//! Streaming worlds: keeps the chunks around a point resident in a World in ring mode,
//! generating or loading the missing ones on worker threads - never on the frame.
//!
//! - Workers take the most urgent request: nearest first, the view direction ahead of the rest.
//!   A chunk comes from the store (a saved chunk), the RAM cache, or the generator
//!   (deterministic, so an untouched chunk is never stored).
//! - `update` (main thread, every frame) re-prioritises around the new centre, evicts resident
//!   chunks beyond radius + 1.5 (hysteresis: walking back and forth over a border doesn't
//!   reload), and pastes finished chunks within a time / count budget, reporting what it pasted
//!   and evicted so the GPU mirror and the lights follow.
//! - The RAM cache keeps the chunks evicted last, compressed by the workers (zlib level 1, ~1/10
//!   the size), least recently evicted dropped first: walking back is a decompression, not a
//!   re-generation. Keyed by absolute chunk, so a rebase keeps it.
//! - Coordinates are the engine's: absolute minus the floating origin. The generator, the store
//!   and `on_evict_modified` get ABSOLUTE chunks; `rebase` follows a move of the origin (chunks
//!   being made then are re-keyed as they arrive).

use std::collections::{BTreeMap, HashMap, HashSet, VecDeque};
use std::sync::atomic::{AtomicUsize, Ordering};
use std::sync::Arc;
use std::time::{Duration, Instant};

use parking_lot::{Condvar, Mutex};

use crate::serialize::{decode_chunk, encode_chunk};
use crate::voxel::sparse_volume::{compute_brick_mask, SparseVolume, BRICK, RING_CHUNK_B};
use crate::voxel::world::{wm, FlatWorld, World};

pub const CHUNK_VOX: i32 = RING_CHUNK_B * BRICK; // 128

pub type ChunkKey = (i32, i32);

#[derive(Clone, Debug, Default)]
pub struct StreamParams {
    pub seed: u32,
    /// A flat world (layers) instead of the generator.
    pub flat: bool,
    pub flat_cfg: FlatWorld,
    /// Resident chunks within this many chunks of the centre (circle).
    pub radius: i32,
    /// RAM cache of evicted chunks (0 = none).
    pub cache_mb: usize,
}

/// Gets each modified chunk (absolute key) as it is evicted.
pub type EvictFn = Box<dyn FnMut(i32, i32, SparseVolume) + Send>;

/// A stored chunk for an absolute (cx, cz) (worker threads), None = generate it.
pub type ChunkSource = Arc<dyn Fn(i32, i32) -> Option<SparseVolume> + Send + Sync>;

/// The ring size (chunk slots per side) a radius needs: room for radius + 1 on both sides so a
/// chunk entering on one side never shares a slot with one still leaving on the other.
pub fn stream_ring_size(radius: i32) -> i32 {
    let mut w = 1;
    while w < 2 * radius + 4 {
        w <<= 1;
    }
    w
}

/// A flat world's chunk.
pub fn flat_chunk(f: &FlatWorld, ny: i32) -> SparseVolume {
    let mut out = SparseVolume::new(CHUNK_VOX, ny, CHUNK_VOX);
    out.track_dirty = false;
    let mut col = vec![0u8; ny as usize];
    let top = f.height.clamp(1, ny - 1);
    let mut y = top;
    for (i, l) in f.layers.iter().enumerate().rev() {
        if y <= 0 {
            break;
        }
        let t = if i == 0 { y } else { l.thickness.min(y) };
        for _ in 0..t {
            y -= 1;
            col[y as usize] = l.mat;
        }
    }
    for k in 0..f.water {
        if top + k < ny {
            col[(top + k) as usize] = wm::WATER;
        }
    }
    for by in 0..out.bricks_y {
        let c = &col[(by * 8) as usize..(by * 8 + 8) as usize];
        let uniform = c.iter().all(|&m| m == c[0]);
        let mut vox = [0u8; 512];
        for z in 0..8 {
            for ly in 0..8 {
                vox[z * 64 + ly * 8..z * 64 + ly * 8 + 8].fill(c[ly]);
            }
        }
        for bz in 0..out.bricks_z {
            for bx in 0..out.bricks_x {
                let gi = out.grid_index(bx, by, bz);
                if uniform {
                    out.grid[gi] = c[0] as u32;
                } else {
                    out.set_brick_voxels(gi, &vox);
                }
            }
        }
    }
    out.dirty_grid.clear();
    out
}

struct Made {
    cx: i32,
    cz: i32,
    /// The origin it was made for (a rebase meanwhile re-keys it).
    ox: i32,
    oz: i32,
    vol: SparseVolume,
}

#[derive(Default)]
struct State {
    /// Most urgent LAST (workers pop).
    queue: Vec<ChunkKey>,
    /// Being made.
    inflight: HashSet<ChunkKey>,
    /// Finished, waiting to be pasted.
    done: VecDeque<Made>,
    /// Evicted, waiting for a worker to compress them into the cache (absolute keys).
    to_encode: HashMap<ChunkKey, SparseVolume>,
    quit: bool,
    ocx: i32,
    ocz: i32,
    make_ms: f64,
    made: usize,
}

/// Compressed evicted chunks, least recently evicted dropped first.
#[derive(Default)]
struct Cache {
    map: HashMap<ChunkKey, (Vec<u8>, u64)>,
    lru: BTreeMap<u64, ChunkKey>,
    stamp: u64,
    bytes: usize,
}

/// Chunks per side of a shared generation context (terrain + plant recipes built once for the
/// tile: half the cost of a chunk generated alone).
const GEN_TILE: i32 = 4;
/// Tile contexts kept (least recently used dropped first).
const GEN_TILES_KEPT: usize = 24;

type Tile = Arc<std::sync::OnceLock<crate::worldgen::RegionGen>>;

struct Shared {
    st: Mutex<State>,
    /// Generation contexts by tile, most recently used last.
    tiles: Mutex<VecDeque<((i32, i32), Tile)>>,
    cv: Condvar,
    done_cv: Condvar,
    cache: Mutex<Cache>,
    cache_cap: usize,
    cache_hits: AtomicUsize,
    params: StreamParams,
    ny: i32,
    source: Option<ChunkSource>,
}

impl Shared {
    /// (Absolute.) Waiting to be compressed: as is; compressed: decoded here (a worker).
    fn take_cached(&self, cx: i32, cz: i32) -> Option<SparseVolume> {
        if let Some(v) = self.st.lock().to_encode.remove(&(cx, cz)) {
            self.cache_hits.fetch_add(1, Ordering::Relaxed);
            return Some(v);
        }
        let blob = {
            let mut c = self.cache.lock();
            let (blob, stamp) = c.map.remove(&(cx, cz))?;
            c.lru.remove(&stamp);
            c.bytes -= blob.len();
            blob
        };
        let v = decode_chunk(&blob, self.ny)?;
        self.cache_hits.fetch_add(1, Ordering::Relaxed);
        Some(v)
    }

    fn to_cache(&self, key: ChunkKey, blob: Vec<u8>) {
        let mut c = self.cache.lock();
        if let Some((old, stamp)) = c.map.remove(&key) {
            c.bytes -= old.len();
            c.lru.remove(&stamp);
        }
        if blob.is_empty() || blob.len() > self.cache_cap {
            return;
        }
        while c.bytes + blob.len() > self.cache_cap {
            let Some((_, k)) = c.lru.pop_first() else { break };
            if let Some((b, _)) = c.map.remove(&k) {
                c.bytes -= b.len();
            }
        }
        c.stamp += 1;
        let stamp = c.stamp;
        c.bytes += blob.len();
        c.lru.insert(stamp, key);
        c.map.insert(key, (blob, stamp));
    }

    /// (Absolute.) From its tile's shared context (built by the first worker that needs it;
    /// others wanting the same tile wait for it instead of building it again).
    fn generate(&self, cx: i32, cz: i32) -> SparseVolume {
        let key = (cx.div_euclid(GEN_TILE), cz.div_euclid(GEN_TILE));
        let tile = {
            let mut tiles = self.tiles.lock();
            let t = match tiles.iter().position(|(k, _)| *k == key) {
                Some(i) => tiles.remove(i).unwrap().1,
                None => Tile::default(),
            };
            tiles.push_back((key, t.clone()));
            while tiles.len() > GEN_TILES_KEPT {
                tiles.pop_front();
            }
            t
        };
        let region = tile.get_or_init(|| {
            let lo = (key.0 * GEN_TILE, key.1 * GEN_TILE);
            crate::worldgen::region_context(self.params.seed, 0, self.ny, 0, lo, (lo.0 + GEN_TILE - 1, lo.1 + GEN_TILE - 1))
        });
        region.chunk(cx, cz)
    }

    /// (Absolute.)
    fn make(&self, cx: i32, cz: i32) -> SparseVolume {
        let (mut v, cached) = match self.take_cached(cx, cz) {
            Some(v) => (v, true),
            None => {
                let v = self.source.as_ref().and_then(|s| s(cx, cz)).unwrap_or_else(|| {
                    if self.params.flat {
                        flat_chunk(&self.params.flat_cfg, self.ny)
                    } else {
                        self.generate(cx, cz)
                    }
                });
                (v, false)
            }
        };
        v.track_dirty = false;
        // The GPU brick masks, off the main thread (a cached chunk keeps its valid ones).
        for b in &mut v.bricks {
            if !cached || !b.mask_ok {
                compute_brick_mask(b);
            }
        }
        v
    }

    fn worker(&self) {
        loop {
            let (key, ox, oz) = {
                let mut st = self.st.lock();
                while !st.quit && st.queue.is_empty() && st.to_encode.is_empty() {
                    self.cv.wait(&mut st);
                }
                if st.quit {
                    return;
                }
                // An evicted chunk into the cache first (it holds ~1 MB until then).
                if let Some(&k) = st.to_encode.keys().next() {
                    let v = st.to_encode.remove(&k).unwrap();
                    drop(st);
                    self.to_cache(k, encode_chunk(&v, 1));
                    continue;
                }
                let k = st.queue.pop().unwrap();
                st.inflight.insert(k);
                (k, st.ocx, st.ocz)
            };
            let t0 = Instant::now();
            let vol = self.make(key.0 + ox, key.1 + oz);
            let ms = t0.elapsed().as_secs_f64() * 1e3;
            let mut st = self.st.lock();
            if st.quit {
                return;
            }
            st.make_ms += ms;
            st.made += 1;
            st.done.push_back(Made { cx: key.0, cz: key.1, ox, oz, vol });
            drop(st);
            self.done_cv.notify_all();
        }
    }
}

/// Keeps a world's ring of chunks filled around a moving centre.
#[derive(Default)]
pub struct StreamManager {
    shared: Option<Arc<Shared>>,
    threads: Vec<std::thread::JoinHandle<()>>,
    radius: i32,
    /// The origin (chunks) for the next start (absolute chunk = engine chunk + origin).
    origin: (i32, i32),
    /// A modified chunk is being evicted (main thread, absolute key): write it out first.
    pub on_evict_modified: Option<EvictFn>,
}

fn in_range(cx: i32, cz: i32, x: f32, z: f32, r: f32) -> bool {
    let dx = (cx as f32 + 0.5) * CHUNK_VOX as f32 - x;
    let dz = (cz as f32 + 0.5) * CHUNK_VOX as f32 - z;
    dx * dx + dz * dz <= (r * CHUNK_VOX as f32).powi(2)
}

fn chunk_of(v: f32) -> i32 {
    (v / CHUNK_VOX as f32).floor() as i32
}

impl StreamManager {
    /// Puts `w` in ring mode and starts the workers (threads 0 = cores - 2).
    pub fn start(&mut self, w: &mut World, p: &StreamParams, source: Option<ChunkSource>, threads: usize) {
        self.stop();
        let mut params = p.clone();
        params.radius = params.radius.clamp(2, 14);
        self.radius = params.radius;
        let ny = w.ny;
        w.vol.ring_init(stream_ring_size(params.radius), ny);
        // Room for the resident circle: pasting never reallocates the whole brick array mid-game.
        let keep = params.radius as f64 + 1.5;
        w.vol.bricks.reserve((std::f64::consts::PI * keep * keep * 2600.0) as usize);
        let threads = if threads == 0 { std::thread::available_parallelism().map_or(2, |n| n.get()).saturating_sub(2).max(1) } else { threads };
        let shared = Arc::new(Shared {
            st: Mutex::new(State { ocx: self.origin.0, ocz: self.origin.1, ..Default::default() }),
            tiles: Mutex::new(VecDeque::new()),
            cv: Condvar::new(),
            done_cv: Condvar::new(),
            cache: Mutex::new(Cache::default()),
            cache_cap: params.cache_mb << 20,
            cache_hits: AtomicUsize::new(0),
            params,
            ny,
            source,
        });
        self.threads = (0..threads)
            .map(|i| {
                let s = shared.clone();
                std::thread::Builder::new().name(format!("stream{i}")).spawn(move || s.worker()).expect("spawn a stream worker")
            })
            .collect();
        log::info!(target: "stream", "streaming: radius {} chunks ({:.0} m), ring {} x {}, {threads} worker threads", self.radius,
                   self.radius as f32 * CHUNK_VOX as f32 * 0.1, w.ring_w, w.ring_w);
        self.shared = Some(shared);
    }

    pub fn stop(&mut self) {
        if let Some(s) = self.shared.take() {
            {
                let mut st = s.st.lock();
                st.quit = true;
                st.queue.clear();
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
    pub fn radius(&self) -> i32 {
        self.radius
    }
    /// The origin in chunks (absolute = engine + origin).
    pub fn origin(&self) -> (i32, i32) {
        self.origin
    }
    /// Before `start` / with nothing resident.
    pub fn set_origin(&mut self, ocx: i32, ocz: i32) {
        self.origin = (ocx, ocz);
        if let Some(s) = &self.shared {
            let mut st = s.st.lock();
            st.ocx = ocx;
            st.ocz = ocz;
        }
    }

    /// The chunks wanted around (x, z), most urgent LAST: by distance, the ones in front first.
    fn want(&self, w: &World, x: f32, z: f32, dx: f32, dz: f32) -> Vec<ChunkKey> {
        let r = self.radius;
        let dl = (dx * dx + dz * dz).sqrt();
        let (dx, dz) = if dl > 1e-4 { (dx / dl, dz / dl) } else { (0.0, 0.0) };
        let (ccx, ccz) = (chunk_of(x), chunk_of(z));
        let mut s: Vec<(f32, ChunkKey)> = Vec::new();
        for j in -r - 1..=r + 1 {
            for i in -r - 1..=r + 1 {
                let (cx, cz) = (ccx + i, ccz + j);
                if !in_range(cx, cz, x, z, r as f32 + 0.5) || w.chunk_loaded(cx, cz) {
                    continue;
                }
                let ox = (cx as f32 + 0.5) * CHUNK_VOX as f32 - x;
                let oz = (cz as f32 + 0.5) * CHUNK_VOX as f32 - z;
                let d = (ox * ox + oz * oz).sqrt();
                let ahead = if d > 1.0 { (ox * dx + oz * dz) / d } else { 1.0 }; // -1 behind .. 1 in front
                s.push((d * (1.25 - 0.45 * ahead), (cx, cz)));
            }
        }
        s.sort_by(|a, b| b.0.total_cmp(&a.0));
        s.into_iter().map(|e| e.1).collect()
    }

    fn evict_far(&mut self, w: &mut World, x: f32, z: f32, evicted: &mut Vec<ChunkKey>) {
        let Some(sh) = self.shared.clone() else { return };
        let keep = self.radius as f32 + 1.5;
        let far: Vec<(i32, i32, bool)> = w.ring_slots.iter().filter(|s| s.loaded && !in_range(s.cx, s.cz, x, z, keep)).map(|s| (s.cx, s.cz, s.modified)).collect();
        for (cx, cz, modified) in far {
            let save = modified && self.on_evict_modified.is_some();
            if save || sh.cache_cap > 0 {
                let out = w.vol.ring_extract(cx, cz);
                let abs = (cx + self.origin.0, cz + self.origin.1);
                if sh.cache_cap > 0 {
                    sh.st.lock().to_encode.insert(abs, out.clone());
                    sh.cv.notify_one();
                }
                if save {
                    (self.on_evict_modified.as_mut().unwrap())(abs.0, abs.1, out);
                }
            }
            w.vol.ring_evict(cx, cz);
            evicted.push((cx, cz));
        }
    }

    /// Every frame (main thread): centre = the camera (voxels), dir = where it looks (x, z).
    /// Pastes finished chunks for at most `budget_ms` / `max_chunks` (at least one).
    #[allow(clippy::too_many_arguments)]
    pub fn update(&mut self, w: &mut World, x: f32, z: f32, dir_x: f32, dir_z: f32, budget_ms: f64, max_chunks: usize, pasted: &mut Vec<ChunkKey>,
                  evicted: &mut Vec<ChunkKey>) {
        let Some(sh) = self.shared.clone() else { return };
        let t0 = Instant::now();
        self.evict_far(w, x, z, evicted);
        let wanted = self.want(w, x, z, dir_x, dir_z);
        let mut ready = {
            let mut st = sh.st.lock();
            let inflight = std::mem::take(&mut st.inflight);
            st.queue = wanted.into_iter().filter(|k| !inflight.contains(k)).collect();
            st.inflight = inflight;
            std::mem::take(&mut st.done)
        };
        sh.cv.notify_all();
        let keep = self.radius as f32 + 1.5;
        let mut finished = Vec::new();
        let n_pasted = pasted.len();
        while let Some(mut r) = ready.pop_front() {
            let (ocx, ocz) = self.origin;
            if (r.ox, r.oz) != (ocx, ocz) {
                // Made before a rebase: to today's coordinates.
                r.cx += r.ox - ocx;
                r.cz += r.oz - ocz;
                (r.ox, r.oz) = (ocx, ocz);
            }
            if !in_range(r.cx, r.cz, x, z, keep) || w.chunk_loaded(r.cx, r.cz) {
                // Left behind while it was made (fast flight): no paste, no budget - into the
                // RAM cache, in case the way leads back. (The C++ engine counted these against
                // the per-frame budget: stale results then held up the ones still wanted.)
                finished.push((r.cx, r.cz));
                if sh.cache_cap > 0 {
                    sh.st.lock().to_encode.insert((r.cx + ocx, r.cz + ocz), r.vol);
                    sh.cv.notify_one();
                }
                continue;
            }
            let done = pasted.len() - n_pasted;
            if done > 0 && (t0.elapsed().as_secs_f64() * 1e3 > budget_ms || done >= max_chunks) {
                ready.push_front(r);
                break;
            }
            finished.push((r.cx, r.cz));
            w.vol.ring_paste(&r.vol, r.cx, r.cz);
            pasted.push((r.cx, r.cz));
        }
        let mut st = sh.st.lock();
        for k in &finished {
            st.inflight.remove(k);
        }
        // Not pasted this frame (budget): back to the front of the done list, in order.
        while let Some(r) = ready.pop_back() {
            st.done.push_front(r);
        }
    }

    /// Block until every chunk within `r` chunks of (x, z) is resident.
    pub fn load_now(&mut self, w: &mut World, x: f32, z: f32, r: i32, pasted: &mut Vec<ChunkKey>, evicted: &mut Vec<ChunkKey>) {
        let Some(sh) = self.shared.clone() else { return };
        let r = r.min(self.radius);
        loop {
            self.update(w, x, z, 0.0, 0.0, 1e9, usize::MAX, pasted, evicted);
            let (ccx, ccz) = (chunk_of(x), chunk_of(z));
            let all = (-r - 1..=r + 1).all(|j| (-r - 1..=r + 1).all(|i| !in_range(ccx + i, ccz + j, x, z, r as f32 + 0.5) || w.chunk_loaded(ccx + i, ccz + j)));
            if all {
                return;
            }
            let mut st = sh.st.lock();
            if st.done.is_empty() {
                sh.done_cv.wait_for(&mut st, Duration::from_millis(20));
            }
        }
    }

    /// Floating origin: the engine's coordinates moved by (dcx, dcz) chunks (the world's ring
    /// moves with them: `SparseVolume::ring_rebase`).
    pub fn rebase(&mut self, dcx: i32, dcz: i32) {
        self.origin.0 += dcx;
        self.origin.1 += dcz;
        let Some(sh) = &self.shared else { return };
        let mut st = sh.st.lock();
        st.ocx += dcx;
        st.ocz += dcz;
        for k in &mut st.queue {
            k.0 -= dcx;
            k.1 -= dcz;
        }
        st.inflight = st.inflight.iter().map(|k| (k.0 - dcx, k.1 - dcz)).collect();
        // `done` keeps its keys: each result carries the origin it was made for.
    }

    pub fn pending_count(&self) -> usize {
        self.shared.as_ref().map_or(0, |s| {
            let st = s.st.lock();
            st.queue.len() + st.inflight.len()
        })
    }
    /// Made, waiting to be pasted.
    pub fn ready_count(&self) -> usize {
        self.shared.as_ref().map_or(0, |s| s.st.lock().done.len())
    }
    /// Average time a worker spends making one chunk (ms).
    pub fn avg_chunk_ms(&self) -> f64 {
        self.shared.as_ref().map_or(0.0, |s| {
            let st = s.st.lock();
            if st.made > 0 { st.make_ms / st.made as f64 } else { 0.0 }
        })
    }
    pub fn chunks_made(&self) -> usize {
        self.shared.as_ref().map_or(0, |s| s.st.lock().made)
    }
    pub fn cache_hits(&self) -> usize {
        self.shared.as_ref().map_or(0, |s| s.cache_hits.load(Ordering::Relaxed))
    }
    /// (bytes, chunks) in the RAM cache.
    pub fn cache_size(&self) -> (usize, usize) {
        self.shared.as_ref().map_or((0, 0), |s| {
            let c = s.cache.lock();
            (c.bytes, c.map.len())
        })
    }
}

impl Drop for StreamManager {
    fn drop(&mut self) {
        self.stop();
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::voxel::world::FlatLayer;

    fn flat_params(radius: i32, cache_mb: usize) -> StreamParams {
        StreamParams {
            flat: true,
            flat_cfg: FlatWorld { layers: vec![FlatLayer { mat: wm::STONE, thickness: 0 }, FlatLayer { mat: wm::GRASS, thickness: 1 }], height: 40, water: 0 },
            radius,
            cache_mb,
            ..Default::default()
        }
    }

    #[test]
    fn ring_sizes() {
        assert_eq!(stream_ring_size(2), 8);
        assert_eq!(stream_ring_size(10), 32);
        assert_eq!(stream_ring_size(14), 32);
    }

    #[test]
    fn flat_chunk_layers() {
        let c = flat_chunk(&flat_params(2, 0).flat_cfg, 64);
        assert_eq!((c.get_voxel(3, 38, 3), c.get_voxel(3, 39, 3), c.get_voxel(3, 40, 3)), (wm::STONE, wm::GRASS, 0));
    }

    #[test]
    fn streams_moves_evicts_and_keeps_edits() {
        let mut w = World::new(256, 64, 256);
        w.init_palette();
        let mut sm = StreamManager::default();
        let saved: Arc<Mutex<HashMap<ChunkKey, SparseVolume>>> = Arc::default();
        let s2 = saved.clone();
        sm.on_evict_modified = Some(Box::new(move |cx, cz, v| {
            s2.lock().insert((cx, cz), v);
        }));
        let s3 = saved.clone();
        let source: ChunkSource = Arc::new(move |cx, cz| s3.lock().get(&(cx, cz)).cloned());
        sm.start(&mut w, &flat_params(3, 64), Some(source), 2);
        let (mut pasted, mut evicted) = (Vec::new(), Vec::new());
        sm.load_now(&mut w, 64.0, 64.0, 3, &mut pasted, &mut evicted);
        assert!(w.chunk_loaded(0, 0) && w.chunk_loaded(3, 0) && w.chunk_loaded(-3, 0));
        assert_eq!(w.get_voxel(-200, 39, 300), wm::GRASS, "negative coordinates stream too");
        // Edit, walk far away (evicted: saved), walk back (loaded from the saved copy).
        w.set_voxel(10, 50, 10, wm::GOLD);
        sm.load_now(&mut w, 64.0 + 30.0 * 128.0, 64.0, 3, &mut pasted, &mut evicted);
        assert!(!w.chunk_loaded(0, 0) && saved.lock().contains_key(&(0, 0)));
        sm.load_now(&mut w, 64.0, 64.0, 3, &mut pasted, &mut evicted);
        assert_eq!(w.get_voxel(10, 50, 10), wm::GOLD, "the edit survived eviction");
        assert!(sm.cache_hits() > 0 || sm.chunks_made() > 0);
        // A rebase: the same absolute place under new engine coordinates.
        sm.rebase(16, 0);
        w.vol.ring_rebase(16, 0);
        sm.load_now(&mut w, 64.0 - 16.0 * 128.0, 64.0, 3, &mut pasted, &mut evicted);
        assert_eq!(w.get_voxel(10 - 16 * 128, 50, 10), wm::GOLD);
        sm.stop();
    }
}
