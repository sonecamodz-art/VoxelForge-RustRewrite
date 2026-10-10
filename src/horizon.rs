//! The far horizon of an endless world: a 5-level toroidal clipmap of the land beyond the loaded
//! chunks, straight from the generator's terrain functions (height, surface material, sea level,
//! the vegetation sites) - drawn by the primary rays (common.glsl traceHorizon) where the voxels
//! end, so hills, lakes and forests carry on to the horizon under the fog.
//!
//! Level L: N x N texels of TEXEL[L] voxels around the camera, recentred in whole tiles when it
//! moves an eighth of the way; tiles (TILE^2 texels) are made nearest-first on worker threads.
//! A texel is two words (ground, crowns); 0 = not made yet (the shader falls back to the next
//! level).

use std::collections::{HashMap, VecDeque};
use std::sync::Arc;
use std::time::Instant;

use parking_lot::{Condvar, Mutex};

use crate::voxel::world::{wm, FlatWorld, Material};
use crate::worldgen::vegetation::{forest_cover, grow_plant, vegetation_sites, PrimKind, Species, VegSite};
use crate::worldgen::Terrain;

pub const LEVELS: usize = 5;
/// Texels per side per level.
pub const N: i32 = 1024;
/// Texels per tile side.
pub const TILE: i32 = 32;
pub const TILES: i32 = N / TILE;
/// Voxels per texel (0.4 .. 6.4 m, doubling with distance).
pub const TEXEL: [i32; LEVELS] = [4, 8, 16, 32, 64];

/// One tile to upload: level, tile coords in the level's texel grid (absolute), its texels
/// (TILE^2 x {ground, crowns}, row-major) - or none: clear it (it left the window).
pub struct Tile {
    pub level: usize,
    pub tx: i32,
    pub tz: i32,
    pub px: Vec<u32>,
}

/// height * 8 in bits 0-13, water bit 14, canopy bit 15, RGB565 in 16-31.
pub fn pack_texel(height: f32, water: bool, canopy: bool, r: f32, g: f32, b: f32) -> u32 {
    let h = ((height * 8.0).round() as i32).clamp(8, 0x3FFF) as u32; // never 0 (= not made)
    let q = |v: f32, m: f32| (v * m).round().clamp(0.0, m) as u32;
    h | (water as u32) << 14 | (canopy as u32) << 15 | q(b, 31.0) << 16 | q(g, 63.0) << 21 | q(r, 31.0) << 27
}

/// Water: the bed's depth in voxels (0-255) in 16-23, its palette index in 24-31.
pub fn pack_water(height: f32, depth: f32, bed: u8) -> u32 {
    let h = ((height * 8.0).round() as i32).clamp(8, 0x3FFF) as u32;
    h | 1 << 14 | (depth.round().clamp(0.0, 255.0) as u32) << 16 | (bed as u32) << 24
}

/// A texel's crowns (a span floating over the ground, air under it): top (voxels, bits 0-10),
/// thickness (11-18), RGB 4-5-4 (19-31).
pub fn pack_crown(bottom: f32, top: f32, r: f32, g: f32, b: f32) -> u32 {
    let t = (top.round() as i32).clamp(1, 2047) as u32;
    let th = ((top - bottom).round() as i32).clamp(1, 255) as u32;
    let q = |v: f32, m: f32| (v * m).round().clamp(0.0, m) as u32;
    t | th << 11 | q(b, 15.0) << 19 | q(g, 31.0) << 23 | q(r, 15.0) << 28
}

fn hash01(x: i32, z: i32, s: u32) -> f32 {
    let mut h = (x as u32).wrapping_mul(0x8da6_b343) ^ (z as u32).wrapping_mul(0xd816_3841) ^ s.wrapping_mul(0xcb1a_b31f);
    h ^= h >> 13;
    h = h.wrapping_mul(0x5bd1_e995);
    h ^= h >> 15;
    (h & 0xff_ffff) as f32 / 16_777_215.0
}

/// One leaf blob of a plant (the finest level draws the generator's real crowns).
#[derive(Clone, Copy)]
struct Crown {
    x: f32,
    y: f32,
    z: f32,
    r: f32,
    squash: f32,
    c: [f32; 3],
}

struct CrownSet {
    s: Species,
    blobs: Vec<Crown>,
}

/// What the workers read: fixed at start.
struct Gen {
    terrain: Terrain,
    seed: u32,
    flat: Option<FlatWorld>,
    palette: Vec<Material>,
    /// Plants grown for the finest level, by site (a crown reaches several tiles).
    crowns: Mutex<HashMap<(i32, i32), Arc<CrownSet>>>,
}

#[derive(Clone, Copy)]
struct Job {
    level: usize,
    tx: i32,
    tz: i32,
}

#[derive(Default)]
struct Queue {
    /// Most urgent last.
    jobs: Vec<Job>,
    done: VecDeque<Tile>,
    quit: bool,
    stat_ms: [f64; LEVELS],
    stat_n: [usize; LEVELS],
}

struct Shared {
    src: Gen,
    q: Mutex<Queue>,
    cv: Condvar,
}

#[derive(Default)]
pub struct HorizonGen {
    shared: Option<Arc<Shared>>,
    threads: Vec<std::thread::JoinHandle<()>>,
    win: [[i32; 2]; LEVELS],
    focused: bool,
    /// Per level / tile slot: the tile it holds, or none.
    owner: Vec<Option<(i32, i32)>>,
    /// Clears to upload.
    cleared: Vec<Tile>,
    stat_t: Option<Instant>,
}

fn slot(level: usize, tx: i32, tz: i32) -> usize {
    (level * TILES as usize + (tz & (TILES - 1)) as usize) * TILES as usize + (tx & (TILES - 1)) as usize
}

impl HorizonGen {
    /// `flat`: a flat world's layers (else the generator with `seed`).
    pub fn start(&mut self, seed: u32, ny: i32, flat: Option<&FlatWorld>, palette: &[Material]) {
        self.stop();
        let src = Gen { terrain: Terrain::new(seed, 0, ny, 0), seed, flat: flat.cloned(), palette: palette.to_vec(), crowns: Mutex::default() };
        let shared = Arc::new(Shared { src, q: Mutex::default(), cv: Condvar::new() });
        let n = (std::thread::available_parallelism().map_or(1, |n| n.get()) / 6).clamp(1, 3);
        self.threads = (0..n)
            .map(|i| {
                let s = shared.clone();
                std::thread::Builder::new().name(format!("horizon{i}")).spawn(move || s.worker()).expect("spawn a horizon worker")
            })
            .collect();
        self.shared = Some(shared);
        self.owner = vec![None; LEVELS * (TILES * TILES) as usize];
        self.cleared.clear();
        self.focused = false;
    }

    pub fn stop(&mut self) {
        if let Some(s) = self.shared.take() {
            {
                let mut q = s.q.lock();
                q.quit = true;
                q.jobs.clear();
            }
            s.cv.notify_all();
            for t in self.threads.drain(..) {
                let _ = t.join();
            }
        }
    }

    pub fn running(&self) -> bool {
        self.shared.is_some()
    }

    /// A level's window origin (absolute texel coords of its min corner).
    pub fn win(&self, level: usize) -> [i32; 2] {
        self.win[level]
    }

    fn in_window(&self, level: usize, tx: i32, tz: i32) -> bool {
        let (x0, z0) = (self.win[level][0] / TILE, self.win[level][1] / TILE);
        (x0..x0 + TILES).contains(&tx) && (z0..z0 + TILES).contains(&tz)
    }

    /// Every level's missing tiles, nearest the camera first (by distance in voxels, so the fine
    /// level's close tiles come before the coarse level's).
    fn requeue(&self) {
        let Some(sh) = &self.shared else { return };
        let mut list: Vec<(f32, Job)> = Vec::new();
        for (level, &texel) in TEXEL.iter().enumerate() {
            let (x0, z0) = (self.win[level][0] / TILE, self.win[level][1] / TILE);
            let cx = (self.win[level][0] as f32 + N as f32 * 0.5) * texel as f32;
            let cz = (self.win[level][1] as f32 + N as f32 * 0.5) * texel as f32;
            let tw = (TILE * texel) as f32;
            for tz in z0..z0 + TILES {
                for tx in x0..x0 + TILES {
                    if self.owner[slot(level, tx, tz)] == Some((tx, tz)) {
                        continue;
                    }
                    let (dx, dz) = ((tx as f32 + 0.5) * tw - cx, (tz as f32 + 0.5) * tw - cz);
                    // Coarse levels are only seen beyond the finer ones: a little later is fine.
                    list.push(((dx * dx + dz * dz).sqrt() * (1.0 + 0.25 * level as f32), Job { level, tx, tz }));
                }
            }
        }
        list.sort_by(|a, b| b.0.total_cmp(&a.0));
        sh.q.lock().jobs = list.into_iter().map(|e| e.1).collect();
        sh.cv.notify_all();
    }

    /// Main thread, every frame: recentre the levels on the camera (absolute voxels); tiles
    /// that left a window are cleared.
    pub fn set_focus(&mut self, x: f32, z: f32) {
        if !self.running() {
            return;
        }
        let mut moved = false;
        for (level, &t) in TEXEL.iter().enumerate() {
            let (ctx, ctz) = ((x / t as f32).floor() as i32, (z / t as f32).floor() as i32);
            let (cx, cz) = (self.win[level][0] + N / 2, self.win[level][1] + N / 2);
            if self.focused && (ctx - cx).abs() < N / 8 && (ctz - cz).abs() < N / 8 {
                continue;
            }
            self.win[level] = [(ctx - N / 2).div_euclid(TILE) * TILE, (ctz - N / 2).div_euclid(TILE) * TILE];
            for i in 0..(TILES * TILES) as usize {
                let k = level * (TILES * TILES) as usize + i;
                let Some((ox, oz)) = self.owner[k] else { continue };
                if self.in_window(level, ox, oz) {
                    continue;
                }
                self.owner[k] = None;
                self.cleared.push(Tile { level, tx: ox, tz: oz, px: Vec::new() });
            }
            moved = true;
        }
        self.focused = true;
        if moved {
            self.requeue();
        }
    }

    /// Main thread: tiles finished / cleared since the last call.
    pub fn poll(&mut self, out: &mut Vec<Tile>) {
        let Some(sh) = self.shared.clone() else { return };
        out.append(&mut self.cleared);
        let got = {
            let mut q = sh.q.lock();
            let since = *self.stat_t.get_or_insert_with(Instant::now);
            if q.stat_n.iter().take(3).sum::<usize>() > 0 && since.elapsed().as_secs_f32() > 3.0 {
                let avg = |l: usize| if q.stat_n[l] > 0 { q.stat_ms[l] / q.stat_n[l] as f64 } else { 0.0 };
                log::info!(target: "horizon", "tiles made: L0 {} ({:.1} ms each), L1 {} ({:.1}), L2 {} ({:.1}); {} queued", q.stat_n[0], avg(0), q.stat_n[1], avg(1),
                           q.stat_n[2], avg(2), q.jobs.len());
                q.stat_ms = [0.0; LEVELS];
                q.stat_n = [0; LEVELS];
                self.stat_t = Some(Instant::now());
            }
            std::mem::take(&mut q.done)
        };
        for t in got {
            if !self.in_window(t.level, t.tx, t.tz) {
                continue; // stale: the window moved on
            }
            let o = &mut self.owner[slot(t.level, t.tx, t.tz)];
            if let Some((ox, oz)) = *o
                && (ox, oz) != (t.tx, t.tz)
            {
                out.push(Tile { level: t.level, tx: ox, tz: oz, px: Vec::new() });
            }
            *o = Some((t.tx, t.tz));
            out.push(t);
        }
    }

    pub fn pending(&self) -> usize {
        self.shared.as_ref().map_or(0, |s| s.q.lock().jobs.len())
    }
}

impl Drop for HorizonGen {
    fn drop(&mut self) {
        self.stop();
    }
}

/// A texel's ground and crowns while a tile is made.
#[derive(Clone, Copy, Default)]
struct Cell {
    h: f32,
    c: [f32; 3],
    water: bool,
    canopy: bool,
    depth: f32,
    bed: u8,
    /// The crowns' span over the texel (top 0: none) and their colour.
    ct: f32,
    cb: f32,
    cc: [f32; 3],
}

impl Shared {
    fn worker(&self) {
        loop {
            let j = {
                let mut q = self.q.lock();
                while !q.quit && q.jobs.is_empty() {
                    self.cv.wait(&mut q);
                }
                if q.quit {
                    return;
                }
                q.jobs.pop().unwrap()
            };
            let t0 = Instant::now();
            let px = self.src.make_tile(j);
            let ms = t0.elapsed().as_secs_f64() * 1e3;
            let mut q = self.q.lock();
            if q.quit {
                return;
            }
            q.stat_ms[j.level] += ms;
            q.stat_n[j.level] += 1;
            q.done.push_back(Tile { level: j.level, tx: j.tx, tz: j.tz, px });
        }
    }
}

impl Gen {
    fn crowns_of(&self, site: &VegSite, height: &impl Fn(i32, i32) -> i32) -> Arc<CrownSet> {
        let key = (site.x, site.z);
        if let Some(c) = self.crowns.lock().get(&key).filter(|c| c.s == site.s) {
            return c.clone();
        }
        let pl = grow_plant(site.s, site.x, site.z, height, self.seed, site.age);
        let blobs = pl
            .prims
            .iter()
            .filter(|q| q.kind == PrimKind::Blob)
            .map(|q| {
                let m = &self.palette[q.mat as usize];
                let a = &self.palette[if q.accent != 0 { q.accent } else { q.mat } as usize];
                let k = if q.accent != 0 { q.accent_p } else { 0.0 };
                Crown { x: q.a[0], y: q.a[1], z: q.a[2], r: q.ra, squash: q.squash, c: [m.r + (a.r - m.r) * k, m.g + (a.g - m.g) * k, m.b + (a.b - m.b) * k] }
            })
            .collect();
        let set = Arc::new(CrownSet { s: site.s, blobs });
        let mut c = self.crowns.lock();
        if c.len() > 40_000 {
            c.clear(); // a long journey: start over
        }
        c.insert(key, set.clone());
        set
    }

    /// The texels of one tile: the ground (height + surface colour), lakes up to sea level, then
    /// the trees of the generator's own vegetation sites (the same ones the chunks grow):
    /// level 0 each plant's real leaf blobs, levels 1-2 one crown per tree sized like its
    /// species, levels 3-4 a smooth canopy following the forest cover.
    fn make_tile(&self, j: Job) -> Vec<u32> {
        let tr = &self.terrain;
        let n = TILE as usize;
        let mut px = vec![0u32; n * n * 2];
        let t = TEXEL[j.level];
        let vx = |x: i32| (j.tx * TILE + x) * t + t / 2;
        let vz = |z: i32| (j.tz * TILE + z) * t + t / 2;
        if let Some(f) = &self.flat {
            let sea = if f.water > 0 { f.height - 1 + f.water } else { -1 };
            let top = f.layers.last().map_or(wm::GRASS, |l| l.mat);
            let m = &self.palette[top as usize];
            let v = if f.water > 0 { pack_water((sea + 1) as f32, f.water as f32, top) } else { pack_texel(f.height as f32, false, false, m.r, m.g, m.b) };
            for i in (0..px.len()).step_by(2) {
                px[i] = v;
            }
            return px;
        }
        let sea = tr.sea();
        let height = |x: i32, z: i32| tr.height(x, z);
        let mut cell = vec![Cell::default(); n * n];
        for z in 0..TILE {
            for x in 0..TILE {
                let (wx, wz) = (vx(x), vz(z));
                let h = tr.height(wx, wz);
                let c = &mut cell[(z * TILE + x) as usize];
                if h < sea {
                    // A lake / the sea: its surface, its depth and its bed.
                    *c = Cell { h: (sea + 1) as f32, water: true, depth: (sea - h) as f32, bed: tr.top(wx, wz), ..Default::default() };
                    continue;
                }
                let m = &self.palette[tr.top(wx, wz) as usize];
                *c = Cell { h: (h + 1) as f32, c: [m.r, m.g, m.b], ..Default::default() };
            }
        }
        // A crown (ellipsoid: horizontal radius r, vertical r * squash; or a cone, its base at cy)
        // over the texels it covers: its span joins the texel's crowns; the colour of the highest.
        let crown = |cell: &mut [Cell], cx: f32, cy: f32, cz: f32, r: f32, squash: f32, col: [f32; 3], cone: bool| {
            let tf = t as f32;
            let x0 = (((cx - r - tf * 0.5) / tf).floor() as i32 - j.tx * TILE).max(0);
            let x1 = (((cx + r - tf * 0.5) / tf).ceil() as i32 - j.tx * TILE).min(TILE - 1);
            let z0 = (((cz - r - tf * 0.5) / tf).floor() as i32 - j.tz * TILE).max(0);
            let z1 = (((cz + r - tf * 0.5) / tf).ceil() as i32 - j.tz * TILE).min(TILE - 1);
            for z in z0..=z1 {
                for x in x0..=x1 {
                    let (dx, dz) = (vx(x) as f32 - cx, vz(z) as f32 - cz);
                    let d2 = (dx * dx + dz * dz) / (r * r);
                    if d2 >= 1.0 {
                        continue;
                    }
                    let top = if cone { cy + r * squash * (1.0 - d2.sqrt()) } else { cy + r * squash * (1.0 - d2).sqrt() };
                    let c = &mut cell[(z * TILE + x) as usize];
                    if top <= c.h || c.water {
                        continue;
                    }
                    // (a cone's underside: flat at its base, a little lower at the rim - branches hang)
                    let bot = if cone { cy - r * 0.25 * d2.sqrt() } else { cy - r * squash * 0.8 * (1.0 - d2).sqrt() }.max(c.h);
                    if c.ct <= 0.0 {
                        (c.cb, c.ct, c.cc) = (bot, top, col);
                    } else {
                        if top > c.ct {
                            (c.ct, c.cc) = (top, col);
                        }
                        c.cb = c.cb.min(bot);
                    }
                }
            }
        };
        if j.level <= 2 {
            let (x0, z0) = (j.tx * TILE * t, j.tz * TILE * t);
            let mut sites = Vec::new();
            vegetation_sites(0, 0, x0, z0, x0 + TILE * t - 1, z0 + TILE * t - 1, &height, &|x, z| tr.top(x, z), sea, self.seed, &mut sites);
            for s in &sites {
                if matches!(s.s, Species::Log | Species::Snag) {
                    continue; // no leaves
                }
                if j.level == 0 {
                    let mut low = 1e9f32;
                    for c in &self.crowns_of(s, &height).blobs {
                        crown(&mut cell, c.x, c.y, c.z, c.r, c.squash, c.c, false);
                        low = low.min(c.y - c.r * c.squash * 0.5);
                    }
                    // The trunk: its texel's column up into the crowns, in bark.
                    let (tx, tz) = ((s.x as f32 / t as f32).floor() as i32 - j.tx * TILE, (s.z as f32 / t as f32).floor() as i32 - j.tz * TILE);
                    if s.s != Species::Bush && low < 1e8 && (0..TILE).contains(&tx) && (0..TILE).contains(&tz) {
                        let c = &mut cell[(tz * TILE + tx) as usize];
                        if !c.water && low > c.h {
                            let b = &self.palette[match s.s {
                                Species::Birch => wm::BIRCH_BARK,
                                Species::Pine => wm::PINE_BARK,
                                _ => wm::WOOD,
                            } as usize];
                            (c.h, c.c) = (low, [b.r, b.g, b.b]);
                        }
                    }
                    continue;
                }
                if s.s == Species::Bush {
                    continue;
                }
                // Levels 1-2: the species' mean crown (the vegetation growers' sizes).
                let g = tr.height(s.x, s.z) as f32;
                let shade = 0.9 + 0.2 * hash01(s.x, s.z, self.seed + 15);
                if s.s == Species::Pine {
                    let m = &self.palette[wm::NEEDLES as usize];
                    let (h, r) = (3.9 * s.age * 50.0, 3.9 * s.age * 50.0 * 0.2);
                    crown(&mut cell, s.x as f32 + 0.5, g + h * 0.27, s.z as f32 + 0.5, r, (h * 0.73) / r, [m.r * shade, m.g * shade, m.b * shade], true);
                } else {
                    let birch = s.s == Species::Birch;
                    let m = &self.palette[if birch { wm::BIRCH_LEAVES } else { wm::LEAVES } as usize];
                    let sc = s.age * if birch { 2.7 } else { 3.0 };
                    let (trunk_h, r, sq) = (sc * if birch { 23.0 } else { 16.5 }, sc * if birch { 8.5 } else { 15.0 }, if birch { 1.95 } else { 0.85 });
                    crown(&mut cell, s.x as f32 + 0.5, g + trunk_h + r * sq * 0.85, s.z as f32 + 0.5, r * 0.95, sq, [m.r * shade, m.g * shade, m.b * shade], false);
                }
            }
        } else {
            // Levels 3-4: the forest as a smooth canopy - tree height times the cover, leaves'
            // colour in proportion.
            for z in 0..TILE {
                for x in 0..TILE {
                    let c = &mut cell[(z * TILE + x) as usize];
                    if c.water {
                        continue;
                    }
                    let (wx, wz) = (vx(x), vz(z));
                    let grassy = matches!(tr.top(wx, wz), wm::GRASS | wm::DRY_GRASS | wm::MOSS | wm::FLOWERS | wm::TALL_GRASS | wm::COARSE_DIRT);
                    if !grassy {
                        continue;
                    }
                    let cover = ((forest_cover(wx, wz, self.seed) - 0.2) / 0.6).clamp(0.0, 1.0);
                    let pine = ((c.h - sea as f32 - 110.0) / 120.0).clamp(0.0, 1.0);
                    if cover <= 0.02 {
                        continue;
                    }
                    c.h += (140.0 + 50.0 * pine) * cover;
                    let lv = &self.palette[if pine > 0.5 { wm::NEEDLES } else { wm::LEAVES } as usize];
                    let k = (cover * 1.3).min(1.0);
                    for (ch, l) in c.c.iter_mut().zip([lv.r, lv.g, lv.b]) {
                        *ch += (l * 0.85 - *ch) * k;
                    }
                    c.canopy = k > 0.5;
                }
            }
        }
        for z in 0..TILE {
            for x in 0..TILE {
                let c = &cell[(z * TILE + x) as usize];
                let i = ((z * TILE + x) * 2) as usize;
                px[i] = if c.water { pack_water(c.h, c.depth, c.bed) } else { pack_texel(c.h, false, c.canopy, c.c[0], c.c[1], c.c[2]) };
                if c.ct > 0.0 {
                    // Leaf clusters: the voxel crowns' light / dark speckle, a texel at a time.
                    let k = 0.8 + 0.35 * hash01(vx(x), vz(z), self.seed + 21);
                    px[i + 1] = pack_crown(c.cb, c.ct, c.cc[0] * k, c.cc[1] * k, c.cc[2] * k);
                }
            }
        }
        px
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn texel_packing() {
        let t = pack_texel(300.0, false, true, 1.0, 0.0, 0.5);
        assert_eq!(t & 0x3FFF, 2400);
        assert_eq!((t >> 15) & 1, 1);
        assert_eq!(t >> 27, 31);
        assert_eq!(pack_texel(0.0, false, false, 0.0, 0.0, 0.0) & 0x3FFF, 8, "never 0 (= not made)");
        let w = pack_water(233.0, 20.0, 4);
        assert_eq!(((w >> 14) & 1, (w >> 16) & 255, w >> 24), (1, 20, 4));
        let c = pack_crown(300.0, 340.0, 0.0, 1.0, 0.0);
        assert_eq!((c & 0x7FF, (c >> 11) & 255, (c >> 23) & 31), (340, 40, 31));
    }

    #[test]
    fn tiles_stream_around_the_focus() {
        let mut h = HorizonGen::default();
        let palette = crate::voxel::world::default_palette();
        h.start(1337, 1024, None, &palette);
        h.set_focus(1024.0, 1024.0);
        assert!(h.pending() > 0);
        let mut got = Vec::new();
        let t0 = Instant::now();
        while got.iter().filter(|t: &&Tile| t.level == 0 && !t.px.is_empty()).count() < 4 && t0.elapsed().as_secs() < 60 {
            h.poll(&mut got);
            std::thread::sleep(std::time::Duration::from_millis(5));
        }
        let t = got.iter().find(|t| t.level == 0 && !t.px.is_empty()).expect("a level-0 tile");
        assert_eq!(t.px.len(), (TILE * TILE * 2) as usize);
        assert!(t.px.iter().step_by(2).all(|&g| g & 0x3FFF >= 8), "every ground texel made");
        // Moving far: the old window's tiles are cleared.
        h.set_focus(1024.0 + 20000.0, 1024.0);
        let mut more = Vec::new();
        h.poll(&mut more);
        assert!(more.iter().any(|t| t.px.is_empty()));
        h.stop();
    }
}
