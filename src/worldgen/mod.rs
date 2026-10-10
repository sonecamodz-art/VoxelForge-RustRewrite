//! World generation, one CHUNK at a time (a chunk = 128 x 128 columns = 12.8 m, full
//! height). A chunk depends only on (seed, world size, chunk x / z):
//!   * terrain: pure per-column functions (height, surface / sub-surface material);
//!   * features (trees, bushes, logs, forest floor, floating spheres, the building): global
//!     bounds and a global draw order; a chunk draws every feature that reaches it, CLIPPED
//!     to itself, in that order - so chunks never write into each other.
//!
//! The whole world = every chunk, generated in parallel (rayon) into chunk-local volumes and
//! pasted in order: deterministic for any thread count.
//!
//! Generator 1 (the original rolling hills) is ported; it is what the C++ engine's startup
//! world uses.

pub mod vegetation;

use std::sync::atomic::Ordering;
use std::time::Instant;

use rayon::prelude::*;

use crate::voxel::sparse_volume::SparseVolume;
use crate::voxel::world::{wm::*, World};
use vegetation::{grow_plant, raster_plant, vegetation_sites, Plant, Species, VegSite, PLANT_REACH};

pub const CHUNK: i32 = 128;
/// Sea level (voxels) of generator 1.
pub const WATER_LEVEL: i32 = 232;
/// Unbounded worlds put the spawn-area features where a Normal world has them.
pub const SPAWN_AREA: i32 = 2048;

#[inline]
fn hash2(x: i32, y: i32, seed: u32) -> u32 {
    let mut h = seed;
    h ^= (x as u32).wrapping_mul(0x8da6b343);
    h ^= (y as u32).wrapping_mul(0xd8163841);
    h = (h ^ (h >> 13)).wrapping_mul(0x9e3779b1);
    h ^ (h >> 16)
}

#[inline]
fn rand01(h: u32) -> f32 {
    (h & 0xffffff) as f32 / 16_777_216.0
}

fn value_noise(x: f32, y: f32, seed: u32) -> f32 {
    let (xi, yi) = (x.floor() as i32, y.floor() as i32);
    let (fx, fy) = (x - xi as f32, y - yi as f32);
    let (sx, sy) = (fx * fx * (3.0 - 2.0 * fx), fy * fy * (3.0 - 2.0 * fy));
    let v00 = rand01(hash2(xi, yi, seed));
    let v10 = rand01(hash2(xi + 1, yi, seed));
    let v01 = rand01(hash2(xi, yi + 1, seed));
    let v11 = rand01(hash2(xi + 1, yi + 1, seed));
    let a = v00 + (v10 - v00) * sx;
    let b = v01 + (v11 - v01) * sx;
    a + (b - a) * sy
}

fn fbm(x: f32, y: f32, seed: u32) -> f32 {
    let (mut sum, mut amp, mut freq) = (0.0f32, 0.5f32, 1.0f32);
    for i in 0..5u32 {
        sum += amp * value_noise(x * freq, y * freq, seed.wrapping_add(i * 101));
        amp *= 0.5;
        freq *= 2.0;
    }
    sum
}

fn is_grass_kind(m: u8) -> bool {
    matches!(m, GRASS | DRY_GRASS | FLOWERS | TALL_GRASS)
}

/// Terrain as functions of the column; heights may be cached for a region.
/// nx = nz = 0: an unbounded world (streaming).
#[derive(Clone, Default)]
pub struct Terrain {
    pub seed: u32,
    pub nx: i32,
    pub ny: i32,
    pub nz: i32,
    hc: Vec<i16>,
    cx0: i32,
    cz0: i32,
    cw: i32,
    ch: i32,
}

impl Terrain {
    pub fn new(seed: u32, nx: i32, ny: i32, nz: i32) -> Self {
        Self { seed, nx, ny, nz, ..Default::default() }
    }

    pub fn bounded(&self) -> bool {
        self.nx > 0
    }

    pub fn sea(&self) -> i32 {
        WATER_LEVEL
    }

    /// Broad hills (~80 m features, 34 m of relief).
    fn height_raw(&self, x: i32, z: i32) -> i32 {
        let h = fbm(x as f32 * 0.0012, z as f32 * 0.0012, self.seed);
        (136 + (h * 340.0) as i32).clamp(16, self.ny - 160)
    }

    /// Top solid voxel's y (clamped to the world).
    #[inline]
    pub fn height(&self, mut x: i32, mut z: i32) -> i32 {
        if self.bounded() {
            x = x.clamp(0, self.nx - 1);
            z = z.clamp(0, self.nz - 1);
        }
        let (lx, lz) = (x - self.cx0, z - self.cz0);
        if lx >= 0 && lz >= 0 && lx < self.cw && lz < self.ch {
            return self.hc[(lz * self.cw + lx) as usize] as i32;
        }
        self.height_raw(x, z)
    }

    /// The water surface over this column (generator 1: the sea).
    pub fn water(&self, _x: i32, _z: i32) -> i32 {
        WATER_LEVEL
    }

    /// Cache heights over [x0, x1] x [z0, z1] (+ a 4-voxel apron).
    pub fn cache(&mut self, mut x0: i32, mut z0: i32, mut x1: i32, mut z1: i32, parallel: bool) {
        x0 -= 4;
        z0 -= 4;
        x1 += 4;
        z1 += 4;
        if self.bounded() {
            x0 = x0.max(0);
            z0 = z0.max(0);
            x1 = x1.min(self.nx - 1);
            z1 = z1.min(self.nz - 1);
        }
        let w = x1 - x0 + 1;
        let h = z1 - z0 + 1;
        let mut c = vec![0i16; (w * h) as usize];
        let row = |(z, out): (usize, &mut [i16])| {
            for (x, o) in out.iter_mut().enumerate() {
                *o = self.height_raw(x0 + x as i32, z0 + z as i32) as i16;
            }
        };
        if parallel {
            c.par_chunks_mut(w as usize).enumerate().for_each(row);
        } else {
            c.chunks_mut(w as usize).enumerate().for_each(row);
        }
        self.hc = c;
        self.cx0 = x0;
        self.cz0 = z0;
        self.cw = w;
        self.ch = h;
    }

    /// Surface material per column (top voxel + the layer under it).
    pub fn surface(&self, x: i32, z: i32) -> (u8, u8) {
        let h = self.height(x, z);
        let slope = 0.25
            * ((self.height(x + 2, z) - self.height(x - 2, z)).abs()).max((self.height(x, z + 2) - self.height(x, z - 2)).abs()) as f32;
        let grain = rand01(hash2(x, z, self.seed ^ 0x51a7));
        let moist = fbm(x as f32 * 0.0022, z as f32 * 0.0022, self.seed + 1);
        let patch = fbm(x as f32 * 0.021, z as f32 * 0.021, self.seed + 2);
        let (mut top, mut sub) = (GRASS, DIRT);
        if h <= WATER_LEVEL + 4 {
            top = SAND;
            sub = SAND;
            if h > WATER_LEVEL && patch + 0.25 * (grain - 0.5) > 0.56 {
                top = GRAVEL;
            }
        } else if slope + 0.3 * (grain - 0.5) > 1.35 {
            top = STONE;
            sub = STONE;
        } else if slope + 0.4 * (grain - 0.5) > 0.85 {
            top = if grain < 0.6 { GRAVEL } else { COARSE_DIRT };
            sub = STONE;
        } else if patch + 0.06 * (grain - 0.5) > 0.66 {
            top = if patch > 0.70 && grain < 0.45 { GRAVEL } else { COARSE_DIRT };
        } else if h <= WATER_LEVEL + 7 && grain < 0.5 {
            top = COARSE_DIRT;
        } else if moist + 0.05 * (grain - 0.5) < 0.33 {
            top = DRY_GRASS;
        } else if moist > 0.36
            && slope < 0.55
            && fbm(x as f32 * 0.005 + 71.0, z as f32 * 0.005, self.seed + 9) + 0.07 * (grain - 0.5) > 0.6
        {
            top = TALL_GRASS;
        } else if moist > 0.42 && fbm(x as f32 * 0.012 + 31.0, z as f32 * 0.012, self.seed + 5) + 0.08 * (grain - 0.5) > 0.64 {
            top = FLOWERS;
        }
        (top, sub)
    }

    pub fn top(&self, x: i32, z: i32) -> u8 {
        self.surface(x, z).0
    }
}

// ---- the building ----------------------------------------------------------------------
/// The tower: 14 x 14 m, up to 11 floors of 3.2 m, a door, wooden floors and switchback
/// stairs inside, windows and torches, a parapet with a lamp on the roof.
#[derive(Clone, Copy, Default)]
struct Building {
    bx: i32,
    bz: i32,
    w: i32,
    base_y: i32,
    floors: i32,
    top_y: i32,
    zs: i32,
    lo: [i32; 3],
    hi: [i32; 3],
}

fn make_building(t: &Terrain) -> Building {
    let w = 140;
    let (ax, az) = if t.bounded() { (t.nx, t.nz) } else { (SPAWN_AREA, SPAWN_AREA) };
    let bx = ax / 2 - w / 2;
    let bz = az / 2 - w / 2;
    let base_y = t.height(ax / 2, az / 2) + 1;
    let floors = ((t.ny - 48 - base_y) / 32).clamp(1, 11);
    let top_y = base_y + 32 * floors;
    let mut mn = base_y;
    let mut dz = -30;
    while dz < w + 30 {
        let mut dx = -30;
        while dx < w + 30 {
            mn = mn.min(t.height(bx + dx, bz + dz));
            dx += 3;
        }
        dz += 3;
    }
    let min_y = (mn - 1).max(0);
    Building {
        bx,
        bz,
        w,
        base_y,
        floors,
        top_y,
        zs: 12,
        lo: [bx - 30, min_y, bz - 30],
        hi: [bx + w + 29, (t.ny - 1).min(top_y + 14), bz + w + 29],
    }
}

/// None = leave the voxel as it is, else the material (0 = air).
fn building_voxel(b: &Building, t: &Terrain, x: i32, y: i32, z: i32) -> Option<u8> {
    let w = b.w;
    let (dx, dz) = (x - b.bx, z - b.bz);
    let (door0, door1) = (w / 2 - 6, w / 2 + 6);
    let in_foot = dx >= 0 && dx < w && dz >= 0 && dz < w;
    if !in_foot {
        let h = t.height(x, z);
        // Torches either side of the door, in brackets on the outside wall.
        if dz == -1 {
            for tx in [door0 - 5, door1 + 4] {
                if dx != tx {
                    continue;
                }
                match y - b.base_y {
                    16 => return Some(TORCH_STICK),
                    17 => return Some(TORCH_BRACKET_N),
                    18 => return Some(TORCH_HEAD),
                    19 => return Some(TORCH_FLAME),
                    _ => {}
                }
            }
        }
        // Steps down from the door to the ground, or a path cut into a slope.
        if dz < 0 && dz >= -30 && dx >= door0 - 3 && dx < door1 + 3 {
            let d = -dz;
            let ty = b.base_y - 1 - 2 * ((d - 1) / 3);
            if y <= ty && y > h {
                return Some(STONE);
            }
            if y > ty && y <= h && y < b.base_y + 24 {
                return Some(AIR);
            }
            return None;
        }
        // A plinth around the footprint so the building never floats on a slope.
        if dx >= -4 && dx < w + 4 && dz >= -4 && dz < w + 4 && y < b.base_y && y > h {
            return Some(STONE);
        }
        return None;
    }
    let h = t.height(x, z);
    if y < b.base_y - 3 {
        return if y > h { Some(STONE) } else { None };
    }
    let wall = dx < 4 || dz < 4 || dx >= w - 4 || dz >= w - 4;
    if y < b.base_y {
        return Some(if wall { BRICKM } else { STONE });
    }
    if y >= b.top_y {
        // The roof: a parapet with crenels, a lamp in the middle.
        if y >= b.top_y + 12 {
            return None;
        }
        if wall {
            if y >= b.top_y + 10 {
                return None;
            }
            if y >= b.top_y + 6 && ((dx + dz) / 8) % 2 != 0 {
                return Some(AIR);
            }
            return Some(BRICKM);
        }
        if y == b.top_y && dx >= w / 2 - 8 && dx < w / 2 + 8 && dz >= w / 2 - 8 && dz < w / 2 + 8 {
            return Some(LAMP);
        }
        return Some(AIR);
    }
    let k = (y - b.base_y) / 32;
    let fy = (y - b.base_y) % 32;
    if wall {
        if dz < 4 && dx >= door0 && dx < door1 && y < b.base_y + 22 {
            return Some(AIR); // the door
        }
        let along = if dz < 4 || dz >= w - 4 { dx } else { dz };
        if fy > 10 && fy < 24 && along > 10 && along < w - 10 && along % 22 >= 6 && along % 22 < 15 {
            return Some(AIR); // windows
        }
        return Some(BRICKM);
    }
    // Stairwell along the inner east wall: flight A up +z, a landing, flight B back up -z.
    let (sx0, sx1, zs) = (w - 28, w - 4, b.zs);
    let in_well = dx >= sx0 && dx < sx1 && dz >= zs && dz < zs + 36;
    if fy >= 29 && !in_well {
        return Some(if k + 1 >= b.floors { STONE } else { WOOD });
    }
    if in_well {
        let flight_a = dx >= w - 16 && dz < zs + 24;
        let flight_b = dx < w - 16 && dz < zs + 24;
        let surf = if flight_a {
            2 * ((dz - zs) / 3 + 1)
        } else if flight_b {
            16 + 2 * ((zs + 23 - dz) / 3 + 1)
        } else {
            16
        };
        if fy >= surf - 3 && fy <= surf - 1 {
            return Some(WOOD);
        }
        return Some(AIR);
    }
    if dx == sx0 - 1 && dz >= zs && dz < zs + 36 && k >= 1 && fy < 10 {
        return Some(WOOD); // railing by the well
    }
    // Two torches per floor in brackets on the west wall.
    if (dz == w / 3 || dz == 2 * w / 3) && dx == 4 {
        match fy {
            16 => return Some(TORCH_STICK),
            17 => return Some(TORCH_BRACKET_W),
            18 => return Some(TORCH_HEAD),
            19 => return Some(TORCH_FLAME),
            _ => {}
        }
    }
    Some(AIR)
}

#[derive(Clone, Copy)]
struct Sphere {
    x: f32,
    y: f32,
    z: f32,
    r: f32,
    mat: u8,
    lo: [i32; 3],
    hi: [i32; 3],
}

#[derive(Clone, Copy)]
struct Floor {
    x: i32,
    z: i32,
    r: i32,
    crown: f32,
    trunk: f32,
    pine: bool,
    lo: [i32; 2],
    hi: [i32; 2],
}

/// What a region of chunks needs: its terrain, the plants / spheres / building reaching it.
struct Context {
    t: Terrain,
    plants: Vec<Plant>,
    /// Indices into `plants` in global draw order.
    plant_order: Vec<usize>,
    floors: Vec<Floor>,
    spheres: Vec<Sphere>,
    building: Building,
    has_building: bool,
}

fn box_overlap2(lo: &[i32; 3], hi: &[i32; 3], x0: i32, z0: i32, x1: i32, z1: i32) -> bool {
    !(hi[0] < x0 || lo[0] > x1 || hi[2] < z0 || lo[2] > z1)
}

fn build_context(t: Terrain, x0: i32, z0: i32, x1: i32, z1: i32, parallel: bool) -> Context {
    let height = |x: i32, z: i32| t.height(x, z);
    let top = |x: i32, z: i32| t.top(x, z);
    let mut sites: Vec<VegSite> = Vec::new();
    vegetation_sites(t.nx, t.nz, x0, z0, x1, z1, &height, &top, t.sea(), t.seed, &mut sites);
    let grow = |s: &VegSite| grow_plant(s.s, s.x, s.z, &height, t.seed, s.age);
    let plants: Vec<Plant> = if parallel { sites.par_iter().map(grow).collect() } else { sites.iter().map(grow).collect() };
    let floor_of = |i: usize| -> Floor {
        let p = &plants[i];
        let s = &sites[i];
        let r = p.crown_radius.max(p.trunk_radius + 4.0).ceil() as i32;
        Floor {
            x: s.x,
            z: s.z,
            r,
            crown: p.crown_radius,
            trunk: p.trunk_radius,
            pine: s.s == Species::Pine,
            lo: [s.x - r, s.z - r],
            hi: [s.x + r, s.z + r],
        }
    };
    let mut plant_order = Vec::new();
    for (i, p) in plants.iter().enumerate() {
        if p.is_empty() {
            continue;
        }
        let has_floor = !matches!(sites[i].s, Species::Bush | Species::Log);
        let p_reach = box_overlap2(&p.lo, &p.hi, x0, z0, x1, z1);
        let f_reach = has_floor && {
            let f = floor_of(i);
            !(f.hi[0] < x0 || f.lo[0] > x1 || f.hi[1] < z0 || f.lo[1] > z1)
        };
        if p_reach || f_reach {
            plant_order.push(i);
        }
    }
    plant_order.sort_by_key(|&i| sites[i].order);
    let floors = plant_order.iter().filter(|&&i| !matches!(sites[i].s, Species::Bush | Species::Log)).map(|&i| floor_of(i)).collect();

    // Floating coloured spheres (4-8 m radius), 16 per Normal-world area, high up.
    let mut spheres = Vec::new();
    let (ax, az) = if t.bounded() { (t.nx, t.nz) } else { (SPAWN_AREA, SPAWN_AREA) };
    let n_spheres = ((16.0 * ax as f64 * az as f64 / (2048.0 * 2048.0)).round() as i32).max(1);
    for i in 0..n_spheres {
        let seed = t.seed;
        let x = 200 + (rand01(hash2(i, 3, seed ^ 0x5eed)) * (ax - 400) as f32) as i32;
        let z = 200 + (rand01(hash2(i, 5, seed ^ 0x5eed)) * (az - 400) as f32) as i32;
        if (x - ax / 2).abs() < 260 && (z - az / 2).abs() < 260 {
            continue;
        }
        let rad = (24.0 + rand01(hash2(i, 11, seed)) * 24.0) * 1.6;
        let ground = t.height(x, z);
        let cy = (ground as f32 + rad + 120.0 + rand01(hash2(i, 17, seed)) * 260.0).min(t.ny as f32 - rad - 4.0);
        let (sx, sy, sz) = (x as f32 + 0.5, cy + 0.5, z as f32 + 0.5);
        let s = Sphere {
            x: sx,
            y: sy,
            z: sz,
            r: rad,
            mat: SPHERE0 + (i & 7) as u8,
            lo: [(sx - rad).floor() as i32, (sy - rad).floor() as i32, (sz - rad).floor() as i32],
            hi: [(sx + rad).ceil() as i32, (sy + rad).ceil() as i32, (sz + rad).ceil() as i32],
        };
        if box_overlap2(&s.lo, &s.hi, x0, z0, x1, z1) {
            spheres.push(s);
        }
    }
    let building = make_building(&t);
    let has_building = box_overlap2(&building.lo, &building.hi, x0, z0, x1, z1);
    Context { t, plants, plant_order, floors, spheres, building, has_building }
}

/// The terrain of one brick column (local brick coords lbx, lbz) into `out`.
fn terrain_bricks(t: &Terrain, out: &mut SparseVolume, ox: i32, oz: i32, lbx: i32, lbz: i32) {
    let mut hs = [0i32; 64];
    let mut tops = [0u8; 64];
    let mut subs = [0u8; 64];
    let (mut min_h, mut max_h) = (t.ny, 0);
    for lz in 0..8 {
        for lx in 0..8 {
            let (x, z, i) = (ox + lbx * 8 + lx, oz + lbz * 8 + lz, (lz * 8 + lx) as usize);
            hs[i] = t.height(x, z);
            let (tp, sb) = t.surface(x, z);
            tops[i] = tp;
            subs[i] = sb;
            min_h = min_h.min(hs[i]);
            max_h = max_h.max(hs[i]);
        }
    }
    let voxel_for = |i: usize, y: i32| -> u8 {
        let h = hs[i];
        if y <= h {
            if y == h {
                return tops[i];
            }
            if y >= h - 2 {
                return subs[i];
            }
            if y >= h - 12 {
                return if subs[i] == STONE { STONE } else { DIRT };
            }
            return STONE;
        }
        if y <= WATER_LEVEL { WATER } else { AIR }
    };
    let mut vox = [0u8; 512];
    for by in 0..out.bricks_y {
        let (y0, y1) = (by * 8, by * 8 + 7);
        let gi = out.grid_index(lbx, by, lbz);
        if y1 <= min_h - 13 {
            out.grid[gi] = STONE as u32;
            continue;
        }
        if y0 > max_h && (y1 <= WATER_LEVEL || y0 > WATER_LEVEL) {
            out.grid[gi] = if y1 <= WATER_LEVEL { WATER as u32 } else { AIR as u32 };
            continue;
        }
        for lz in 0..8 {
            for ly in 0..8 {
                for lx in 0..8 {
                    vox[(lz * 64 + ly * 8 + lx) as usize] = voxel_for((lz * 8 + lx) as usize, y0 + ly);
                }
            }
        }
        out.set_brick_voxels(gi, &vox);
    }
}

fn build_chunk(c: &Context, cx: i32, cz: i32) -> SparseVolume {
    let t = &c.t;
    let (ox, oz) = (cx * CHUNK, cz * CHUNK);
    let lo = [ox, 0, oz];
    let hi = [ox + CHUNK - 1, t.ny - 1, oz + CHUNK - 1];
    let mut out = SparseVolume::new(CHUNK, t.ny, CHUNK);
    out.track_dirty = false;
    for lbz in 0..CHUNK / 8 {
        for lbx in 0..CHUNK / 8 {
            terrain_bricks(t, &mut out, ox, oz, lbx, lbz);
        }
    }
    let inside = |x: i32, y: i32, z: i32| x >= lo[0] && x <= hi[0] && z >= lo[2] && z <= hi[2] && y >= 0 && y < t.ny;
    // Plants, in their global order (wood overwrites, leaves only fill air).
    for &i in &c.plant_order {
        raster_plant(&c.plants[i], lo, hi, |x, y, z, m, only_air| {
            if only_air && out.get_voxel(x - ox, y, z - oz) != AIR {
                return;
            }
            if inside(x, y, z) {
                out.set_voxel(x - ox, y, z - oz, m);
            }
        });
    }
    // The forest floor: coarse dirt at the trunks, moss / needle litter under the crowns.
    for f in &c.floors {
        if f.hi[0] < lo[0] || f.lo[0] > hi[0] || f.hi[1] < lo[2] || f.lo[1] > hi[2] {
            continue;
        }
        for dz in -f.r..=f.r {
            for dx in -f.r..=f.r {
                let (gx, gz) = (f.x + dx, f.z + dz);
                if gx < lo[0] || gx > hi[0] || gz < lo[2] || gz > hi[2] {
                    continue;
                }
                let gh = t.height(gx, gz);
                if !is_grass_kind(out.get_voxel(gx - ox, gh, gz - oz)) {
                    continue;
                }
                let d = ((dx * dx + dz * dz) as f32).sqrt();
                let r = rand01(hash2(gx * 7 + 17, gz, t.seed ^ 0xf100));
                let m = if d < f.trunk + 4.0 + 3.0 * r {
                    Some(COARSE_DIRT)
                } else if d < f.crown * (0.5 + 0.45 * r) {
                    Some(if f.pine && r < 0.7 { COARSE_DIRT } else { MOSS })
                } else {
                    None
                };
                if let Some(m) = m {
                    if inside(gx, gh, gz) {
                        out.set_voxel(gx - ox, gh, gz - oz, m);
                    }
                }
            }
        }
    }
    for s in &c.spheres {
        if !box_overlap2(&s.lo, &s.hi, lo[0], lo[2], hi[0], hi[2]) {
            continue;
        }
        for z in s.lo[2].max(lo[2])..=s.hi[2].min(hi[2]) {
            for y in s.lo[1].max(0)..=s.hi[1].min(t.ny - 1) {
                for x in s.lo[0].max(lo[0])..=s.hi[0].min(hi[0]) {
                    let (dx, dy, dz) = (x as f32 + 0.5 - s.x, y as f32 + 0.5 - s.y, z as f32 + 0.5 - s.z);
                    if dx * dx + dy * dy + dz * dz <= s.r * s.r {
                        out.set_voxel(x - ox, y, z - oz, s.mat);
                    }
                }
            }
        }
    }
    if c.has_building {
        let b = &c.building;
        for z in b.lo[2].max(lo[2])..=b.hi[2].min(hi[2]) {
            for x in b.lo[0].max(lo[0])..=b.hi[0].min(hi[0]) {
                for y in b.lo[1]..=b.hi[1] {
                    if let Some(m) = building_voxel(b, t, x, y, z) {
                        out.set_voxel(x - ox, y, z - oz, m);
                    }
                }
            }
        }
    }
    out.compact();
    out.dirty_grid.clear();
    out
}

/// The whole world (World::generate): bands of chunk rows, each band's plant recipes built
/// once, its chunks generated in parallel, pasted in order.
pub fn generate_world(w: &mut World, seed: u32) {
    let (ncx, ncz) = (w.nx / CHUNK, w.nz / CHUNK);
    const BAND: i32 = 8;
    let t_all = Instant::now();
    let (mut t_ctx, mut t_chunks, mut t_paste) = (0.0, 0.0, 0.0);
    let mut done = 0;
    let mut cz0 = 0;
    while cz0 < ncz {
        let cz1 = (cz0 + BAND).min(ncz) - 1;
        let t0 = Instant::now();
        let mut t = Terrain::new(seed, w.nx, w.ny, w.nz);
        let (x0, x1, z0, z1) = (0, w.nx - 1, cz0 * CHUNK, (cz1 + 1) * CHUNK - 1);
        t.cache(x0, z0 - PLANT_REACH - 8, x1, z1 + PLANT_REACH + 8, true);
        let ctx = build_context(t, x0, z0, x1, z1, true);
        t_ctx += t0.elapsed().as_secs_f64() * 1e3;
        let t0 = Instant::now();
        let n = ncx * (cz1 - cz0 + 1);
        // rayon's work stealing balances forest / lake / building chunks on its own.
        let vols: Vec<SparseVolume> = (0..n).into_par_iter().map(|i| build_chunk(&ctx, i % ncx, cz0 + i / ncx)).collect();
        t_chunks += t0.elapsed().as_secs_f64() * 1e3;
        let t0 = Instant::now();
        for (i, v) in vols.into_iter().enumerate() {
            let i = i as i32;
            w.vol.paste_chunk(v, (i % ncx) * (CHUNK / 8), 0, (cz0 + i / ncx) * (CHUNK / 8));
        }
        t_paste += t0.elapsed().as_secs_f64() * 1e3;
        done += n;
        if let Some(p) = &w.gen_progress {
            p.store(done * 1000 / (ncx * ncz), Ordering::Relaxed);
        }
        cz0 += BAND;
    }
    w.vol.bricks.shrink_to_fit();
    w.vol.dirty_grid.clear();
    let b = make_building(&Terrain::new(seed, w.nx, w.ny, w.nz));
    log::info!(target: "world", "building: x {}-{}, z {}-{}, ground floor y {}, {} floors", b.bx, b.bx + b.w - 1, b.bz,
               b.bz + b.w - 1, b.base_y, b.floors);
    log::info!(target: "world", "generated {}x{}x{} in {:.0} ms (context {:.0}, chunks {:.0}, paste {:.0}): {} bricks",
               w.nx, w.ny, w.nz, t_all.elapsed().as_secs_f64() * 1e3, t_ctx, t_chunks, t_paste, w.vol.bricks.len());
}

/// One chunk from scratch (streaming): columns [cx*128, +128) x [cz*128, +128) of an
/// nx x ny x nz world (nx = nz = 0: unbounded). Single-threaded (workers call it concurrently).
pub fn generate_chunk(seed: u32, nx: i32, ny: i32, nz: i32, cx: i32, cz: i32) -> SparseVolume {
    let mut t = Terrain::new(seed, nx, ny, nz);
    let (x0, z0) = (cx * CHUNK, cz * CHUNK);
    let (x1, z1) = (x0 + CHUNK - 1, z0 + CHUNK - 1);
    let apron = PLANT_REACH + 8;
    t.cache(x0 - apron, z0 - apron, x1 + apron, z1 + apron, false);
    let ctx = build_context(t, x0, z0, x1, z1, false);
    build_chunk(&ctx, cx, cz)
}

/// A generation context shared by a block of chunks (streaming builds one per tile: the
/// terrain and plant recipes around the tile once, instead of per chunk with a 9x-area apron).
pub struct RegionGen {
    ctx: Context,
    lo: (i32, i32),
    hi: (i32, i32),
}

/// The context for chunks [cx0, cx1] x [cz0, cz1] of an nx x ny x nz world (nx = nz = 0:
/// unbounded). Chunks built from it equal `generate_chunk`'s.
pub fn region_context(seed: u32, nx: i32, ny: i32, nz: i32, (cx0, cz0): (i32, i32), (cx1, cz1): (i32, i32)) -> RegionGen {
    let mut t = Terrain::new(seed, nx, ny, nz);
    let (x0, z0, x1, z1) = (cx0 * CHUNK, cz0 * CHUNK, (cx1 + 1) * CHUNK - 1, (cz1 + 1) * CHUNK - 1);
    let apron = PLANT_REACH + 8;
    t.cache(x0 - apron, z0 - apron, x1 + apron, z1 + apron, false);
    RegionGen { ctx: build_context(t, x0, z0, x1, z1, false), lo: (cx0, cz0), hi: (cx1, cz1) }
}

impl RegionGen {
    pub fn contains(&self, cx: i32, cz: i32) -> bool {
        (self.lo.0..=self.hi.0).contains(&cx) && (self.lo.1..=self.hi.1).contains(&cz)
    }
    /// One chunk of the region (single-threaded; workers share the context).
    pub fn chunk(&self, cx: i32, cz: i32) -> SparseVolume {
        debug_assert!(self.contains(cx, cz));
        build_chunk(&self.ctx, cx, cz)
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn region_chunks_match_standalone_ones() {
        let r = region_context(7, 0, 512, 0, (-1, 3), (0, 4));
        for (cx, cz) in [(-1, 3), (0, 4)] {
            let a = r.chunk(cx, cz);
            let b = generate_chunk(7, 0, 512, 0, cx, cz);
            for z in (0..128).step_by(5) {
                for y in (100..400).step_by(3) {
                    for x in (0..128).step_by(5) {
                        assert_eq!(a.get_voxel(x, y, z), b.get_voxel(x, y, z), "chunk {cx},{cz} at {x},{y},{z}");
                    }
                }
            }
        }
    }

    #[test]
    #[ignore]
    fn bench_region_vs_chunk() {
        let t = Instant::now();
        for i in 0..4 {
            generate_chunk(1337, 0, 1024, 0, 20 + i, 30);
        }
        println!("BENCH standalone {:.0} ms/chunk", t.elapsed().as_secs_f64() * 1e3 / 4.0);
        for n in [2, 3, 4] {
            let t = Instant::now();
            let r = region_context(1337, 0, 1024, 0, (20, 30), (19 + n, 29 + n));
            let tc = t.elapsed().as_secs_f64() * 1e3;
            for j in 0..n {
                for i in 0..n {
                    r.chunk(20 + i, 30 + j);
                }
            }
            println!("BENCH tile {n}x{n}: context {tc:.0} ms, {:.0} ms/chunk all in", t.elapsed().as_secs_f64() * 1e3 / (n * n) as f64);
        }
    }

    #[test]
    fn chunk_matches_world() {
        // A chunk generated alone equals the same chunk of a whole (small) world.
        let mut w = World::new(512, 1024, 512);
        w.init_palette();
        generate_world(&mut w, 99);
        let c = generate_chunk(99, 512, 1024, 512, 1, 2);
        let mut diff = 0;
        for z in (0..128).step_by(3) {
            for y in (150..600).step_by(2) {
                for x in (0..128).step_by(3) {
                    if c.get_voxel(x, y, z) != w.get_voxel(128 + x, y, 256 + z) {
                        diff += 1;
                    }
                }
            }
        }
        assert_eq!(diff, 0);
        // Terrain sanity: ground under the sea is stone / sand, the sea is water.
        let t = Terrain::new(99, 512, 1024, 512);
        let h = t.height(300, 300);
        assert!(h > 16 && h < 1024 - 160);
        assert_ne!(w.get_voxel(300, h - 20, 300), AIR);
    }
}
