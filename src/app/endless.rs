//! Endless worlds (the running session): the stream manager keeping the chunks around the
//! camera resident, and the chunk store holding those that differ from the generator (edited
//! ones; a converted fixed-size save's).
//!
//! Floating origin: the engine works in coordinates relative to an origin that moves in steps
//! of ORIGIN_STEP voxels (6.5 km) - absolute = engine + origin - so floats stay precise however
//! far you travel. The step is whole periods of every toroidal window (the world ring, probes,
//! wind), so none of them move on a rebase. Absolute coordinates: the generator, the chunk
//! store, saves.

use std::path::{Path, PathBuf};
use std::sync::Arc;
use std::time::Instant;

use rayon::prelude::*;

use crate::cvars;
use crate::gfx::context::Context;
use crate::gfx::renderer::Renderer;
use crate::horizon::{HorizonGen, Tile, LEVELS, TEXEL};
use crate::region_file::ChunkStore;
use crate::serialize::{encode_chunk, EndlessInfo};
use crate::stream::{ChunkKey, ChunkSource, StreamManager, StreamParams, CHUNK_VOX};
use crate::voxel::world::{wm, FlatLayer, World};

/// The origin moves in steps of this many voxels.
pub const ORIGIN_STEP: i32 = 65536;

/// The origin step nearest a coordinate.
pub fn origin_for(abs: f32) -> i32 {
    ((abs as f64 / ORIGIN_STEP as f64).round() as i32) * ORIGIN_STEP
}

/// saves/<name>.vfworld -> saves/<name>.chunks (the region files).
pub fn chunk_dir_for(save: &Path) -> PathBuf {
    save.with_extension("chunks")
}

/// Per-second streaming stats (the log).
#[derive(Default)]
struct Stats {
    added: usize,
    removed: usize,
    worst_paste: f64,
    worst_upload: f64,
    since: Option<Instant>,
}

pub struct Endless {
    pub stream: StreamManager,
    pub store: Arc<ChunkStore>,
    pub info: EndlessInfo,
    /// The land beyond the loaded chunks, drawn to the horizon.
    pub horizon: HorizonGen,
    /// A new world: the GPU clipmap (another world's) is cleared first.
    horizon_fresh: bool,
    /// Per ring slot: when its chunk began fading in (None = shown).
    fade_start: Vec<Option<Instant>>,
    stats: Stats,
}

fn params(info: &EndlessInfo, radius: i32) -> StreamParams {
    StreamParams { seed: info.seed, flat: info.flat, flat_cfg: info.flat_cfg.clone(), radius, cache_mb: cvars::STREAM_CACHE.get().max(0) as usize }
}

impl Endless {
    /// Stream `world` (ring mode) as the endless world `info`, chunks stored in `chunk_dir`
    /// (None = not saved yet) and, converting an old save, every chunk of the bounded
    /// `convert_from` (kept as they are; the rest generates). Returns once the chunks within
    /// `inner_r` of (x, z) are in. (x, z) are ABSOLUTE: the origin is set near them (the caller
    /// converts what it places: minus `origin()`).
    pub fn begin(world: &mut World, info: &EndlessInfo, chunk_dir: Option<&Path>, x: f32, z: f32, inner_r: i32, convert_from: Option<&World>) -> Self {
        let mut info = info.clone();
        info.endless = true;
        let store = Arc::new(ChunkStore::default());
        store.open(chunk_dir.unwrap_or(Path::new("")), info.ny);
        if let Some(src) = convert_from {
            // An old save: all its chunks, encoded on every core.
            let t0 = Instant::now();
            let (ncx, ncz) = (src.nx / CHUNK_VOX, src.nz / CHUNK_VOX);
            let blobs: Vec<Vec<u8>> = (0..ncx * ncz).into_par_iter().map(|i| encode_chunk(&src.vol.ring_extract(i % ncx, i / ncx), 6)).collect();
            let bytes: usize = blobs.iter().map(Vec::len).sum();
            for (i, b) in blobs.into_iter().enumerate() {
                let i = i as i32;
                if !b.is_empty() {
                    store.put_blob(i % ncx, i / ncx, b);
                }
            }
            log::info!(target: "saves", "converted a {}x{} world to endless: {} chunks ({:.1} MB) in {:.0} ms, beyond it {}", src.nx, src.nz, ncx * ncz,
                       bytes as f64 / 1048576.0, t0.elapsed().as_secs_f64() * 1e3, if info.flat { "flat".to_string() } else { format!("seed {}", info.seed) });
        }
        world.vol.ny = info.ny; // (the ring is sized from it)
        let mut stream = StreamManager::default();
        let s2 = store.clone();
        stream.on_evict_modified = Some(Box::new(move |cx, cz, v| s2.put(cx, cz, v)));
        let s3 = store.clone();
        let source: ChunkSource = Arc::new(move |cx, cz| s3.get(cx, cz));
        let (ox, oz) = (origin_for(x), origin_for(z));
        stream.set_origin(ox / CHUNK_VOX, oz / CHUNK_VOX);
        stream.start(world, &params(&info, cvars::R_VIEW_DISTANCE.get()), Some(source), 0);
        world.vol.origin_x = ox;
        world.vol.origin_z = oz;
        let t0 = Instant::now();
        let r = inner_r.min(stream.radius());
        stream.load_now(world, x - ox as f32, z - oz as f32, r, &mut Vec::new(), &mut Vec::new());
        log::info!(target: "stream", "{} chunks around {x:.0}, {z:.0} in {:.0} ms ({:.0} ms a chunk on a worker)", world.ring_slots.iter().filter(|s| s.loaded).count(),
                   t0.elapsed().as_secs_f64() * 1e3, stream.avg_chunk_ms());
        let mut horizon = HorizonGen::default();
        horizon.start(info.seed, info.ny, info.flat.then_some(&info.flat_cfg), &world.palette);
        horizon.set_focus(x, z);
        Self { stream, store, info, horizon, horizon_fresh: true, fade_start: Vec::new(), stats: Stats::default() }
    }

    /// The origin in voxels (absolute = engine + origin).
    pub fn origin(&self) -> (i32, i32) {
        let (cx, cz) = self.stream.origin();
        (cx * CHUNK_VOX, cz * CHUNK_VOX)
    }

    /// Move the engine's coordinates by (dx, dz) voxels (multiples of ORIGIN_STEP): the world
    /// ring and the streaming follow. The caller shifts everything else it holds.
    pub fn rebase(&mut self, world: &mut World, dx: i32, dz: i32) {
        let (dcx, dcz) = (dx / CHUNK_VOX, dz / CHUNK_VOX);
        self.stream.rebase(dcx, dcz);
        world.vol.ring_rebase(dcx, dcz);
        let (ox, oz) = self.origin();
        log::info!(target: "stream", "floating origin moved by {dx}, {dz} voxels: now {ox}, {oz}");
    }

    /// Save: the edited resident chunks go to the store, the store to the region files.
    pub fn flush(&mut self, world: &mut World, chunk_dir: &Path) -> Result<(), String> {
        let (ocx, ocz) = self.stream.origin();
        let mut n = 0;
        let edited: Vec<(i32, i32)> = world.ring_slots.iter().filter(|s| s.loaded && s.modified).map(|s| (s.cx, s.cz)).collect();
        for (cx, cz) in edited {
            self.store.put(cx + ocx, cz + ocz, world.vol.ring_extract(cx, cz));
            world.vol.ring_slot_mut(cx, cz).modified = false;
            n += 1;
        }
        let t0 = Instant::now();
        let pending = self.store.pending_chunks();
        let res = self.store.flush(chunk_dir);
        log::info!(target: "saves", "chunks: {n} edited resident, {pending} written to {} in {:.0} ms{}", chunk_dir.display(), t0.elapsed().as_secs_f64() * 1e3,
                   if res.is_ok() { "" } else { " (FAILED)" });
        res
    }

    /// r.view.distance changed: restart streaming around (x, z) with the new radius (edited
    /// chunks kept). True = restarted (the caller re-uploads the scene).
    pub fn apply_view_distance(&mut self, world: &mut World, x: f32, z: f32) -> bool {
        let want = cvars::R_VIEW_DISTANCE.get().clamp(2, 14);
        if !self.stream.running() || want == self.stream.radius() {
            return false;
        }
        let t0 = Instant::now();
        let (ocx, ocz) = self.stream.origin();
        let edited: Vec<(i32, i32)> = world.ring_slots.iter().filter(|s| s.loaded && s.modified).map(|s| (s.cx, s.cz)).collect();
        for (cx, cz) in edited {
            self.store.put(cx + ocx, cz + ocz, world.vol.ring_extract(cx, cz)); // edits survive the restart
        }
        let s3 = self.store.clone();
        let source: ChunkSource = Arc::new(move |cx, cz| s3.get(cx, cz));
        self.stream.start(world, &params(&self.info, want), Some(source), 0);
        self.fade_start.clear();
        self.stream.load_now(world, x, z, want.min(4), &mut Vec::new(), &mut Vec::new());
        log::info!(target: "stream", "view distance {want} chunks: restarted in {:.0} ms", t0.elapsed().as_secs_f64() * 1e3);
        true
    }

    /// Every frame: chunks in / out around the camera, their GPU upload, the fade-in of chunks
    /// arriving at the edge, stats.
    pub fn frame(&mut self, ctx: &Context, r: &mut Renderer, world: &mut World, cam_pos: crate::math::Vec3, cam_fwd: crate::math::Vec3) {
        if !self.stream.running() {
            return;
        }
        let (mut added, mut removed) = (Vec::<ChunkKey>::new(), Vec::<ChunkKey>::new());
        let t0 = Instant::now();
        // One chunk a frame keeps up with walking / normal flight; a backlog takes 3.
        let per = cvars::STREAM_PER_FRAME.get().max(1) as usize;
        let per = if self.stream.ready_count() > 24 { per.max(3) } else { per };
        self.stream.update(world, cam_pos.x, cam_pos.z, cam_fwd.x, cam_fwd.z, cvars::STREAM_BUDGET.get() as f64, per, &mut added, &mut removed);
        let t1 = Instant::now();
        // Batches of 8: bounded staging.
        let n = added.len().max(removed.len());
        for i in (0..n).step_by(8) {
            let a = &added[i.min(added.len())..(i + 8).min(added.len())];
            let b = &removed[i.min(removed.len())..(i + 8).min(removed.len())];
            r.gpu.sync_chunks(ctx, world, a, b);
        }
        let t2 = Instant::now();
        self.fade(world, r, cam_pos, &added, &removed);
        self.update_horizon(ctx, r, cam_pos);
        let st = &mut self.stats;
        st.added += added.len();
        st.removed += removed.len();
        st.worst_paste = st.worst_paste.max((t1 - t0).as_secs_f64() * 1e3);
        st.worst_upload = st.worst_upload.max((t2 - t1).as_secs_f64() * 1e3);
        let since = *st.since.get_or_insert(t0);
        if since.elapsed().as_secs_f32() >= 1.0 {
            if st.added + st.removed > 0 {
                let (cb, cc) = self.stream.cache_size();
                log::info!(target: "stream", "+{} -{} chunks | worst: paste {:.2} ms, upload {:.2} ms | pending {}, {:.1} ms/chunk, {} unsaved | cache {cc} chunks {:.0} MB, {} of {} from it",
                           st.added, st.removed, st.worst_paste, st.worst_upload, self.stream.pending_count(), self.stream.avg_chunk_ms(), self.store.pending_chunks(),
                           cb as f64 / 1048576.0, self.stream.cache_hits(), self.stream.chunks_made());
            }
            *st = Stats { since: Some(Instant::now()), ..Default::default() };
        }
    }

    /// The far horizon: recentre on the camera, upload what the workers finished.
    fn update_horizon(&mut self, ctx: &Context, r: &mut Renderer, cam: crate::math::Vec3) {
        let (ox, oz) = self.origin();
        let mut up: Vec<Tile> = Vec::new();
        if self.horizon_fresh {
            // A new world: whatever the clipmap held is another world's.
            for level in 0..LEVELS {
                for tz in 0..crate::horizon::TILES {
                    for tx in 0..crate::horizon::TILES {
                        up.push(Tile { level, tx, tz, px: Vec::new() });
                    }
                }
            }
            self.horizon_fresh = false;
        }
        self.horizon.set_focus(cam.x + ox as f32, cam.z + oz as f32); // (absolute)
        self.horizon.poll(&mut up);
        // To the engine's texels (the origin is whole windows of every level).
        let mut win = [0i32; 10];
        for level in 0..LEVELS {
            let w = self.horizon.win(level);
            win[level * 2] = w[0] - ox / TEXEL[level];
            win[level * 2 + 1] = w[1] - oz / TEXEL[level];
        }
        // The voxels are trusted inside ~a chunk short of the view distance (the rest may still
        // be streaming in: the horizon covers it meanwhile).
        let rin = ((self.stream.radius() - 1).max(0)) * CHUNK_VOX;
        r.update_horizon(ctx, cvars::R_HORIZON.get(), win, rin, &up);
    }

    /// Shots: wait (up to `secs`) for the horizon tiles queued so far, uploading as they come.
    pub fn settle_horizon(&mut self, ctx: &Context, r: &mut Renderer, cam: crate::math::Vec3, secs: f32) {
        let t0 = Instant::now();
        loop {
            self.update_horizon(ctx, r, cam);
            if self.horizon.pending() == 0 || t0.elapsed().as_secs_f32() > secs {
                break;
            }
            std::thread::sleep(std::time::Duration::from_millis(10));
        }
        std::thread::sleep(std::time::Duration::from_millis(50));
        self.update_horizon(ctx, r, cam); // (the last tiles in flight)
        log::info!(target: "horizon", "settled in {:.1} s, {} tiles still queued", t0.elapsed().as_secs_f32(), self.horizon.pending());
    }

    /// Chunks streaming in near the edge of the view distance fade in instead of popping.
    fn fade(&mut self, world: &World, r: &mut Renderer, cam: crate::math::Vec3, added: &[ChunkKey], removed: &[ChunkKey]) {
        let w = world.ring_w;
        let n = (w * w) as usize;
        if self.fade_start.len() != n {
            self.fade_start = vec![None; n];
        }
        let slot = |cx: i32, cz: i32| ((cz & (w - 1)) * w + (cx & (w - 1))) as usize;
        let now = Instant::now();
        let dur = cvars::R_HORIZON_FADE.get();
        let rin = ((self.stream.radius() - 1).max(0) * CHUNK_VOX) as f32;
        for &(cx, cz) in removed {
            self.fade_start[slot(cx, cz)] = None;
        }
        for &(cx, cz) in added {
            let dx = (cx as f32 + 0.5) * CHUNK_VOX as f32 - cam.x;
            let dz = (cz as f32 + 0.5) * CHUNK_VOX as f32 - cam.z;
            self.fade_start[slot(cx, cz)] = (dur > 0.0 && dx * dx + dz * dz > (rin - 64.0).powi(2)).then_some(now);
        }
        let mut left = vec![0u8; n];
        let mut any = false;
        for (s, f) in self.fade_start.iter_mut().enumerate() {
            let Some(t) = *f else { continue };
            let k = 1.0 - t.elapsed().as_secs_f32() / dur.max(1e-3);
            if k <= 0.0 {
                *f = None;
                continue;
            }
            left[s] = (k * 255.0).round() as u8;
            any = true;
        }
        r.set_chunk_fade(if any { &left } else { &[] });
    }
}

/// What an old (fixed-size) save continues as: flat worlds stay flat (their layers read back
/// from the ground), anything else uses the default seed.
pub fn guess(w: &World) -> EndlessInfo {
    let mut e = EndlessInfo { endless: true, seed: 1337, ny: w.ny, ..Default::default() };
    // Flat: most sampled columns identical (edits and bodies aside).
    let mut votes: std::collections::HashMap<Vec<u8>, usize> = Default::default();
    for j in 0..5 {
        for i in 0..5 {
            let (x, z) = (w.nx * (2 * i + 1) / 10, w.nz * (2 * j + 1) / 10);
            *votes.entry((0..w.ny).map(|y| w.get_voxel(x, y, z)).collect()).or_default() += 1;
        }
    }
    let Some((col, n)) = votes.into_iter().max_by_key(|(_, n)| *n) else { return e };
    if n < 18 {
        return e;
    }
    let top = col.iter().rposition(|&m| m != 0 && m != wm::WATER).map_or(0, |y| y + 1);
    if top == 0 {
        return e;
    }
    e.flat = true;
    e.flat_cfg.height = top as i32;
    let mut y = 0;
    while y < top {
        let k = (y..top).find(|&k| col[k] != col[y]).unwrap_or(top);
        e.flat_cfg.layers.push(FlatLayer { mat: col[y], thickness: (k - y) as i32 });
        y = k;
    }
    e.flat_cfg.water = col[top..].iter().take_while(|&&m| m == wm::WATER).count() as i32;
    e
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::voxel::world::FlatWorld;

    #[test]
    fn origin_steps() {
        assert_eq!(origin_for(1000.0), 0);
        assert_eq!(origin_for(40000.0), 65536);
        assert_eq!(origin_for(-40000.0), -65536);
        assert_eq!(chunk_dir_for(Path::new("saves/My World.vfworld")), PathBuf::from("saves/My World.chunks"));
    }

    #[test]
    fn flat_worlds_are_recognised() {
        let mut w = World::new(256, 128, 256);
        let f = FlatWorld { layers: vec![FlatLayer { mat: wm::STONE, thickness: 10 }, FlatLayer { mat: wm::DIRT, thickness: 3 }], height: 13, water: 2 };
        w.generate_flat(&f);
        w.fill_sphere(30.0, 15.0, 30.0, 6.0, wm::GOLD); // an edit somewhere
        let e = guess(&w);
        assert!(e.flat);
        assert_eq!(e.flat_cfg, f);
    }
}
