//! World saves (`saves/<name>.vfworld`), byte-compatible with the C++ engine's format (v1-v6:
//! its saves load here and the other way round).
//!
//! File: [magic "VFFS"][version u32]
//!   v2+: [info size u32][info]            - name, time, thumbnail: read by save listings
//!                                           without inflating the world
//!        [raw size u64][chunks u32]{[raw u32][comp u32][zlib(4 MB of payload)]}...
//!        (v1: [raw size u64][zlib(payload)] - one stream)
//!   payload: palette, world volume (RLE grid + detailed bricks) or an endless world's recipe,
//!   bodies (volume, pose, velocities, glow integrity), the world state (camera, world cvars,
//!   partial water volumes, the next import-colour slot).
//! A load parses everything into temporaries before replacing anything: a failed load leaves
//! the current world untouched. All numbers are little-endian.

use std::io::{Read, Write};
use std::path::Path;

use flate2::read::ZlibDecoder;
use flate2::write::ZlibEncoder;
use flate2::Compression;
use parking_lot::Mutex;
use rayon::prelude::*;

use crate::math::{Mat3, Vec3};
use crate::objects::VoxObject;
use crate::voxel::sparse_volume::{Brick, SparseVolume, BRICK, RING_CHUNK_B};
use crate::voxel::world::{default_palette, ensure_presets, wm, FlatLayer, FlatWorld, Material, World, CUSTOM_COLOR_END, IMPORT_COLOR_BASE};

const MAGIC: u32 = 0x5646_4653; // "VFFS"
/// 1: no info block / world state; 2: u16 water, bounded only; 3: no surfaces; 4: no generator
/// version / height; 5: no world scale.
pub const VERSION: u32 = 6;
/// Bodies stay below the physics shell's 10-bit-per-axis packing.
const MAX_BODY_DIM: i32 = 1024;
/// zlib can't exceed ~1032:1, so a larger declared size is a corrupt file.
const MAX_DEFLATE_RATIO: u64 = 1100;
/// Payload chunks deflated independently (in parallel).
const CHUNK: usize = 4 << 20;
/// zlib level for saves: 2 deflates a generated world ~5x faster than the C++ engine's 6 for a
/// ~30 % larger file (any level reads back everywhere).
const SAVE_LEVEL: u32 = 2;
/// The newest terrain generator this build has.
pub const GEN_LATEST: i32 = 1;

#[derive(Clone, Debug, Default, PartialEq)]
pub struct SceneInfo {
    pub name: String,
    /// Unix seconds.
    pub saved_at: i64,
    /// RGBA8, top-down (empty if none).
    pub thumb_w: u32,
    pub thumb_h: u32,
    pub thumb: Vec<u8>,
}

#[derive(Clone, Copy, Debug, PartialEq)]
pub struct WaterCell {
    pub x: i32,
    pub y: i32,
    pub z: i32,
    pub volume: f32,
}

#[derive(Clone, Debug, Default, PartialEq)]
pub struct SceneState {
    pub has_camera: bool,
    pub cam_pos: [f32; 3],
    pub cam_yaw: f32,
    pub cam_pitch: f32,
    /// World-state cvars (name, value).
    pub cvars: Vec<(String, String)>,
    /// Partial water cells (FluidSim).
    pub water: Vec<WaterCell>,
}

/// v3+: an endless world's file holds no voxels - only how to generate them; the chunks that
/// differ from the generator live in region files beside it.
#[derive(Clone, Debug, PartialEq)]
pub struct EndlessInfo {
    pub endless: bool,
    pub seed: u32,
    pub flat: bool,
    pub flat_cfg: FlatWorld,
    /// v5: the generator version it was made with (older saves: 1).
    pub generator: i32,
    /// v5: its height in voxels (older saves: 1024).
    pub ny: i32,
    /// v6: generator 3's world scale (older saves: 1).
    pub scale: i32,
    /// Saves before v4 / v5 whose custom colours sat in slots now taken by presets / terrain
    /// materials: old -> new for every voxel id (empty = nothing moved). A bounded world and the
    /// bodies are remapped already; an endless world's region files must be too.
    pub remap: Vec<u8>,
}

impl Default for EndlessInfo {
    fn default() -> Self {
        Self { endless: false, seed: 0, flat: false, flat_cfg: FlatWorld::default(), generator: 1, ny: 1024, scale: 1, remap: Vec::new() }
    }
}

#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub enum SceneError {
    Missing,
    NotASave,
    Newer,
    WorldSize,
    Corrupt,
    WriteFailed,
}

impl std::fmt::Display for SceneError {
    /// Player-facing.
    fn fmt(&self, f: &mut std::fmt::Formatter) -> std::fmt::Result {
        f.write_str(match self {
            Self::Missing => "the save file is missing",
            Self::NotASave => "it isn't a VoxelForge world",
            Self::Newer => "it was saved by a newer version of the game",
            Self::WorldSize => "it was saved with a different world size",
            Self::Corrupt => "the file is damaged",
            Self::WriteFailed => "the file couldn't be written (disk full or no permission?)",
        })
    }
}

impl std::error::Error for SceneError {}

pub type SceneResult<T> = Result<T, SceneError>;

// ---- byte sink / source -----------------------------------------------------------------

/// A byte sink. With a `sink`, every full CHUNK of bytes is handed over as it fills (and the
/// sink returns an empty buffer to go on with): a save streams its payload to the compressors
/// instead of building it whole.
#[derive(Default)]
struct Writer<'a> {
    buf: Vec<u8>,
    /// Bytes written in all (handed-over chunks included).
    total: u64,
    sink: Option<ChunkSink<'a>>,
}

/// Takes a full chunk, returns an empty buffer to go on with.
type ChunkSink<'a> = Box<dyn FnMut(Vec<u8>) -> Vec<u8> + 'a>;

impl Writer<'_> {
    fn raw(&mut self, b: &[u8]) {
        self.buf.extend_from_slice(b);
        self.total += b.len() as u64;
        if let Some(sink) = self.sink.as_mut().filter(|_| self.buf.len() >= CHUNK) {
            let rest = self.buf.split_off(CHUNK);
            self.buf = sink(std::mem::take(&mut self.buf));
            self.buf.clear();
            self.buf.extend_from_slice(&rest);
        }
    }
    fn u16(&mut self, v: u16) {
        self.raw(&v.to_le_bytes());
    }
    fn u32(&mut self, v: u32) {
        self.raw(&v.to_le_bytes());
    }
    fn i32(&mut self, v: i32) {
        self.raw(&v.to_le_bytes());
    }
    fn i64(&mut self, v: i64) {
        self.raw(&v.to_le_bytes());
    }
    fn f32(&mut self, v: f32) {
        self.raw(&v.to_le_bytes());
    }
    fn str(&mut self, s: &str) {
        self.u32(s.len() as u32);
        self.raw(s.as_bytes());
    }
}

/// Reads past the end yield zeros and clear `ok` (checked at the end of each section).
struct Reader<'a> {
    b: &'a [u8],
    ok: bool,
}

impl<'a> Reader<'a> {
    fn new(b: &'a [u8]) -> Self {
        Self { b, ok: true }
    }
    fn take(&mut self, n: usize) -> Option<&'a [u8]> {
        if self.b.len() < n {
            self.ok = false;
            self.b = &[];
            return None;
        }
        let (h, t) = self.b.split_at(n);
        self.b = t;
        Some(h)
    }
    fn arr<const N: usize>(&mut self) -> [u8; N] {
        self.take(N).map_or([0; N], |s| s.try_into().unwrap())
    }
    fn u16(&mut self) -> u16 {
        u16::from_le_bytes(self.arr())
    }
    fn u32(&mut self) -> u32 {
        u32::from_le_bytes(self.arr())
    }
    fn i32(&mut self) -> i32 {
        i32::from_le_bytes(self.arr())
    }
    fn i64(&mut self) -> i64 {
        i64::from_le_bytes(self.arr())
    }
    fn f32(&mut self) -> f32 {
        f32::from_le_bytes(self.arr())
    }
    fn vec3(&mut self) -> Vec3 {
        Vec3::new(self.f32(), self.f32(), self.f32())
    }
    fn str(&mut self, max_len: usize) -> String {
        let n = self.u32() as usize;
        if !self.ok || n > max_len {
            self.ok = false;
            return String::new();
        }
        self.take(n).map(|s| String::from_utf8_lossy(s).into_owned()).unwrap_or_default()
    }
}

fn check(ok: bool) -> SceneResult<()> {
    if ok { Ok(()) } else { Err(SceneError::Corrupt) }
}

fn zlib(data: &[u8], level: u32) -> Option<Vec<u8>> {
    let mut e = ZlibEncoder::new(Vec::with_capacity(data.len() / 4 + 64), Compression::new(level));
    e.write_all(data).ok()?;
    e.finish().ok()
}

/// Inflates exactly `raw` bytes (anything else is corruption).
fn unzlib_into(src: &[u8], out: &mut [u8]) -> bool {
    let mut d = ZlibDecoder::new(src);
    if d.read_exact(out).is_err() {
        return false;
    }
    let mut extra = [0u8; 1];
    matches!(d.read(&mut extra), Ok(0))
}

// ---- sections ---------------------------------------------------------------------------

fn write_info(w: &mut Writer, info: &SceneInfo) {
    w.str(&info.name);
    w.i64(info.saved_at);
    let thumb = info.thumb_w > 0 && info.thumb_h > 0 && info.thumb.len() == (info.thumb_w * info.thumb_h * 4) as usize;
    w.u16(if thumb { info.thumb_w as u16 } else { 0 });
    w.u16(if thumb { info.thumb_h as u16 } else { 0 });
    if thumb {
        let c = zlib(&info.thumb, 6).unwrap_or_default();
        w.u32(c.len() as u32);
        w.raw(&c);
    }
}

fn read_info(r: &mut Reader) -> SceneResult<SceneInfo> {
    let mut info = SceneInfo { name: r.str(256), saved_at: r.i64(), thumb_w: r.u16() as u32, thumb_h: r.u16() as u32, thumb: Vec::new() };
    check(r.ok)?;
    if info.thumb_w > 0 && info.thumb_h > 0 {
        let n = r.u32() as usize;
        let src = r.take(n).ok_or(SceneError::Corrupt)?;
        if info.thumb_w > 1024 || info.thumb_h > 1024 {
            return Err(SceneError::Corrupt);
        }
        info.thumb = vec![0; (info.thumb_w * info.thumb_h * 4) as usize];
        if n == 0 || !unzlib_into(src, &mut info.thumb) {
            // A broken thumbnail isn't a broken save.
            info.thumb.clear();
            info.thumb_w = 0;
            info.thumb_h = 0;
        }
    }
    Ok(info)
}

/// RLE over grid words: (count, value) pairs, then a 0xFFFFFFFF terminator.
fn write_grid_rle(w: &mut Writer, g: &[u32]) {
    w.u32(g.len() as u32);
    for run in g.chunk_by(|a, b| a == b) {
        w.u32(run.len() as u32);
        w.u32(run[0]);
    }
    w.u32(u32::MAX);
}

fn read_grid_rle(r: &mut Reader) -> Option<Vec<u32>> {
    let total = r.u32() as usize;
    if !r.ok || total > 1 << 30 {
        return None;
    }
    let mut g = Vec::with_capacity(total);
    loop {
        let count = r.u32();
        if !r.ok {
            return None;
        }
        if count == u32::MAX {
            break;
        }
        let v = r.u32();
        if !r.ok || g.len() + count as usize > total {
            return None;
        }
        g.resize(g.len() + count as usize, v);
    }
    (g.len() == total).then_some(g)
}

/// A volume: dims, the grid with detailed entries renumbered 256 + sequential (no stale brick
/// indices), then those bricks in grid order.
fn write_volume(w: &mut Writer, v: &SparseVolume) {
    w.i32(v.nx);
    w.i32(v.ny);
    w.i32(v.nz);
    let mut detailed: Vec<&Brick> = Vec::new();
    let g: Vec<u32> = v
        .grid
        .iter()
        .map(|&e| {
            if e < 256 {
                return e;
            }
            detailed.push(&v.bricks[(e - 256) as usize]);
            255 + detailed.len() as u32
        })
        .collect();
    write_grid_rle(w, &g);
    w.u32(detailed.len() as u32);
    for b in detailed {
        w.raw(&b.vox);
    }
}

enum DimLimit {
    /// Each axis at most this (bodies, chunks).
    Max(i32),
    /// A world: any size a world can be created at.
    World,
}

fn read_volume(r: &mut Reader, limit: DimLimit) -> SceneResult<SparseVolume> {
    let (nx, ny, nz) = (r.i32(), r.i32(), r.i32());
    if !r.ok || nx <= 0 || ny <= 0 || nz <= 0 {
        return Err(SceneError::Corrupt);
    }
    match limit {
        DimLimit::Max(m) if nx > m || ny > m || nz > m => return Err(SceneError::Corrupt),
        DimLimit::World
            if nx % 128 != 0 || nz % 128 != 0 || ny % 8 != 0 || nx < 256 || nz < 256 || ny < 128 || nx > 16384 || nz > 16384 || ny > 2048 =>
        {
            return Err(SceneError::WorldSize);
        }
        _ => {}
    }
    let mut v = SparseVolume::new(nx, ny, nz);
    let grid = read_grid_rle(r).ok_or(SceneError::Corrupt)?;
    if grid.len() != v.grid.len() {
        return Err(SceneError::Corrupt);
    }
    let n = r.u32() as usize;
    if !r.ok || n > 1 << 24 {
        return Err(SceneError::Corrupt);
    }
    let src = r.take(n * 512).ok_or(SceneError::Corrupt)?;
    v.bricks = src
        .chunks_exact(512)
        .map(|c| {
            let mut b = Brick::filled(0);
            b.vox.copy_from_slice(c);
            b
        })
        .collect();
    if grid.iter().any(|&e| e >= 256 && (e - 256) as usize >= n) {
        return Err(SceneError::Corrupt);
    }
    v.grid = grid;
    Ok(v)
}

/// Every voxel of v through lut (256 entries).
pub fn remap_voxels(v: &mut SparseVolume, lut: &[u8]) {
    if lut.len() < 256 {
        return;
    }
    for g in &mut v.grid {
        if *g < 256 {
            *g = lut[*g as usize] as u32;
        }
    }
    for b in &mut v.bricks {
        for m in &mut b.vox {
            *m = lut[*m as usize];
        }
        b.mask_ok = false;
    }
}

/// Finite pose / velocities and an orthonormal rotation (a NaN or a skewed basis would spread
/// into the physics and the GPU transforms).
fn valid_pose(o: &VoxObject) -> bool {
    let r = o.pose.r.to_cols_array();
    let finite = r.iter().chain(&o.pose.t.to_array()).chain(&o.vel.to_array()).chain(&o.ang_vel.to_array()).all(|f| f.is_finite());
    finite && (o.pose.r.transpose() * o.pose.r - Mat3::IDENTITY).to_cols_array().iter().all(|d| d.abs() <= 1e-3)
}

fn write_material(w: &mut Writer, m: &Material, version: u32) {
    for f in [m.r, m.g, m.b, m.emissive].iter().chain(&m.props) {
        w.f32(*f);
    }
    if version >= 4 {
        for &f in &m.surf {
            w.f32(f);
        }
    }
}

// ---- save ---------------------------------------------------------------------------------

/// What a save holds besides the world.
#[derive(Default)]
pub struct SaveParts<'a> {
    pub info: Option<&'a SceneInfo>,
    pub state: Option<&'a SceneState>,
    /// Set (and `endless`): the world's voxels are not written (its chunks are saved apart).
    pub endless: Option<&'a EndlessInfo>,
    /// 0 = current (tests write older versions).
    pub version: u32,
}

/// The payload, deflated in 4 MB chunks on every core (one stream took seconds for a generated
/// world), written to a temporary file renamed over the old one: a crash or a full disk
/// mid-save leaves the previous save intact.
pub fn save_scene(path: &Path, world: &World, objects: &[VoxObject], parts: &SaveParts) -> SceneResult<()> {
    let version = if parts.version != 0 { parts.version } else { VERSION };
    let mut f = Writer::default();
    f.u32(MAGIC);
    f.u32(version);
    if version >= 2 {
        let mut head = Writer::default();
        write_info(&mut head, parts.info.unwrap_or(&SceneInfo::default()));
        f.u32(head.buf.len() as u32);
        f.raw(&head.buf);
    }
    if version >= 2 {
        // Full chunks deflate on other cores while the payload goes on being written; their
        // buffers come back for reuse (memory stays at a few chunks, not the whole payload).
        type Done = Vec<(usize, Option<Vec<u8>>)>;
        let done: Mutex<Done> = Mutex::new(Vec::new());
        let (free_tx, free_rx) = std::sync::mpsc::channel::<Vec<u8>>();
        let done = &done;
        let total = rayon::in_place_scope(|sc| {
            // Chunks in flight at most (then the writer waits for a buffer back): writing
            // outruns deflating ~10x. A single-thread pool can't wait on itself.
            let threads = rayon::current_num_threads();
            let max_in_flight = if threads > 1 { threads * 2 } else { usize::MAX };
            let (mut next, mut recycled) = (0usize, 0usize);
            let sink = move |full: Vec<u8>| {
                let (i, free_tx) = (next, free_tx.clone());
                next += 1;
                done.lock().push((0, None));
                sc.spawn(move |_| {
                    let c = zlib(&full, SAVE_LEVEL);
                    done.lock()[i] = (full.len(), c);
                    let _ = free_tx.send(full);
                });
                let back = if next - recycled >= max_in_flight { free_rx.recv().ok() } else { free_rx.try_recv().ok() };
                match back {
                    Some(b) => {
                        recycled += 1;
                        b
                    }
                    None => Vec::with_capacity(CHUNK + 4096),
                }
            };
            let mut w = Writer { buf: Vec::with_capacity(CHUNK + 4096), total: 0, sink: Some(Box::new(sink)) };
            write_payload(&mut w, world, objects, parts, version);
            let (last, total) = (std::mem::take(&mut w.buf), w.total);
            drop(w);
            if !last.is_empty() || total == 0 {
                done.lock().push((last.len(), zlib(&last, SAVE_LEVEL)));
            }
            total
        });
        let chunks = std::mem::take(&mut *done.lock());
        f.raw(&total.to_le_bytes());
        f.u32(chunks.len() as u32);
        for (raw, comp) in chunks {
            let comp = comp.ok_or(SceneError::WriteFailed)?;
            f.u32(raw as u32);
            f.u32(comp.len() as u32);
            f.raw(&comp);
        }
    } else {
        let mut w = Writer::default();
        write_payload(&mut w, world, objects, parts, version);
        f.raw(&(w.buf.len() as u64).to_le_bytes());
        f.raw(&zlib(&w.buf, SAVE_LEVEL).ok_or(SceneError::WriteFailed)?);
    }
    write_atomic(path, &f.buf).map_err(|e| {
        log::error!(target: "saves", "writing {}: {e}", path.display());
        SceneError::WriteFailed
    })
}

/// Palette, world (or its endless recipe), bodies, world state.
fn write_payload(w: &mut Writer, world: &World, objects: &[VoxObject], parts: &SaveParts, version: u32) {
    w.u32(world.palette.len() as u32);
    for m in &world.palette {
        write_material(w, m, version);
    }
    match parts.endless.filter(|e| e.endless) {
        Some(e) => {
            w.u32(1);
            w.u32(e.seed);
            w.u32(e.flat as u32);
            w.i32(e.flat_cfg.height);
            w.i32(e.flat_cfg.water);
            w.u32(e.flat_cfg.layers.len() as u32);
            for l in &e.flat_cfg.layers {
                w.u32(l.mat as u32);
                w.i32(l.thickness);
            }
            if version >= 5 {
                w.u32(e.generator as u32);
                w.i32(e.ny);
            }
            if version >= 6 {
                w.u32(e.scale as u32);
            }
        }
        None => {
            if version >= 3 {
                w.u32(0);
            }
            write_volume(w, world);
        }
    }
    w.u32(objects.len() as u32);
    for o in objects {
        write_volume(w, &o.vol);
        for f in o.pose.r.to_cols_array() {
            w.f32(f);
        }
        for v in [o.pose.t, o.vel, o.ang_vel] {
            w.f32(v.x);
            w.f32(v.y);
            w.f32(v.z);
        }
        if version >= 2 {
            w.f32(o.glow_integrity);
        }
    }
    if version >= 2 {
        let none = SceneState::default();
        let st = parts.state.unwrap_or(&none);
        w.u32(st.has_camera as u32);
        for f in st.cam_pos {
            w.f32(f);
        }
        w.f32(st.cam_yaw);
        w.f32(st.cam_pitch);
        w.u32(st.cvars.len() as u32);
        for (k, v) in &st.cvars {
            w.str(k);
            w.str(v);
        }
        w.u32(world.next_import_slot() as u32);
        w.u32(st.water.len() as u32);
        for c in &st.water {
            if version >= 3 {
                w.i32(c.x);
                w.i32(c.y);
                w.i32(c.z);
            } else {
                w.u16(c.x as u16);
                w.u16(c.y as u16);
                w.u16(c.z as u16);
            }
            w.f32(c.volume);
        }
    }
}

/// Write `tmp`, then rename it over `path`.
pub fn write_atomic(path: &Path, data: &[u8]) -> std::io::Result<()> {
    let mut tmp = path.as_os_str().to_owned();
    tmp.push(".tmp");
    let tmp = std::path::PathBuf::from(tmp);
    let res = (|| {
        let mut f = std::fs::File::create(&tmp)?;
        f.write_all(data)?;
        f.sync_all()?;
        std::fs::rename(&tmp, path)
    })();
    if res.is_err() {
        let _ = std::fs::remove_file(&tmp);
    }
    res
}

// ---- load ---------------------------------------------------------------------------------

/// The header: version, info (v2+) and where the payload starts.
fn read_header(file: &[u8]) -> SceneResult<(u32, SceneInfo, usize)> {
    if file.len() < 8 {
        return Err(SceneError::NotASave);
    }
    let magic = u32::from_le_bytes(file[0..4].try_into().unwrap());
    let ver = u32::from_le_bytes(file[4..8].try_into().unwrap());
    if magic != MAGIC {
        return Err(SceneError::NotASave);
    }
    if ver > VERSION {
        return Err(SceneError::Newer);
    }
    if ver < 1 {
        return Err(SceneError::Corrupt);
    }
    if ver < 2 {
        return Ok((ver, SceneInfo::default(), 8));
    }
    if file.len() < 12 {
        return Err(SceneError::Corrupt);
    }
    let n = u32::from_le_bytes(file[8..12].try_into().unwrap()) as usize;
    let block = file.get(12..12 + n).ok_or(SceneError::Corrupt)?;
    let info = read_info(&mut Reader::new(block))?;
    Ok((ver, info, 12 + n))
}

fn read_file(path: &Path, limit: Option<u64>) -> SceneResult<Vec<u8>> {
    let f = std::fs::File::open(path).map_err(|e| if e.kind() == std::io::ErrorKind::NotFound { SceneError::Missing } else { SceneError::Corrupt })?;
    let mut buf = Vec::new();
    let res = match limit {
        Some(n) => f.take(n).read_to_end(&mut buf),
        None => (&f).read_to_end(&mut buf),
    };
    res.map_err(|_| SceneError::Corrupt)?;
    Ok(buf)
}

/// Header only (name, time, thumbnail): cheap, for listing saves.
pub fn read_scene_info(path: &Path) -> SceneResult<SceneInfo> {
    read_header(&read_file(path, Some(4 << 20))?).map(|(_, info, _)| info)
}

/// The file's format version (None = not a readable save).
pub fn scene_file_version(path: &Path) -> Option<u32> {
    read_file(path, Some(4 << 20)).ok().and_then(|f| read_header(&f).ok()).map(|(v, _, _)| v)
}

/// The inflated payload.
fn inflate_payload(ver: u32, body: &[u8]) -> SceneResult<Vec<u8>> {
    let raw = u64::from_le_bytes(body.get(..8).ok_or(SceneError::Corrupt)?.try_into().unwrap());
    let rest = &body[8..];
    if raw > 1 << 34 || raw > rest.len() as u64 * MAX_DEFLATE_RATIO {
        return Err(SceneError::Corrupt);
    }
    let mut data = Vec::new();
    data.try_reserve_exact(raw as usize).map_err(|_| SceneError::Corrupt)?;
    data.resize(raw as usize, 0);
    if ver == 1 {
        return if unzlib_into(rest, &mut data) { Ok(data) } else { Err(SceneError::Corrupt) };
    }
    let mut r = Reader::new(rest);
    let nc = r.u32() as usize;
    if !r.ok || nc == 0 || nc > raw as usize / CHUNK + 2 {
        return Err(SceneError::Corrupt);
    }
    let mut jobs: Vec<(&[u8], &mut [u8])> = Vec::with_capacity(nc);
    let mut out: &mut [u8] = &mut data;
    for _ in 0..nc {
        let (raw_n, comp_n) = (r.u32() as usize, r.u32() as usize);
        let src = r.take(comp_n).ok_or(SceneError::Corrupt)?;
        if raw_n > CHUNK || raw_n > out.len() {
            return Err(SceneError::Corrupt);
        }
        let (dst, tail) = std::mem::take(&mut out).split_at_mut(raw_n);
        out = tail;
        jobs.push((src, dst));
    }
    if !out.is_empty() {
        return Err(SceneError::Corrupt);
    }
    if !jobs.into_par_iter().all(|(src, dst)| unzlib_into(src, dst)) {
        return Err(SceneError::Corrupt);
    }
    Ok(data)
}

/// Everything a save holds, parsed.
pub struct LoadedScene {
    pub palette: Vec<Material>,
    /// A bounded world's voxels (None: endless - the caller streams them).
    pub volume: Option<SparseVolume>,
    pub endless: EndlessInfo,
    pub objects: Vec<VoxObject>,
    pub info: SceneInfo,
    pub state: SceneState,
    /// None: v1 (recompute from the palette).
    pub import_slot: Option<usize>,
}

impl LoadedScene {
    /// Commit into `world` / `objects` (the palette, and a bounded world's voxels; an endless
    /// world's volume is cleared for the caller to stream).
    pub fn apply(self, world: &mut World, objects: &mut Vec<VoxObject>) -> (SceneInfo, SceneState, EndlessInfo) {
        world.palette = self.palette;
        match self.volume {
            Some(mut v) => {
                v.track_dirty = world.vol.track_dirty;
                world.vol = v;
            }
            None => world.vol.clear(),
        }
        world.vol.dirty_grid.clear();
        match self.import_slot {
            Some(i) => world.set_next_import_slot(i),
            None => world.recompute_import_slot(),
        }
        *objects = self.objects;
        (self.info, self.state, self.endless)
    }
}

pub fn load_scene(path: &Path) -> SceneResult<LoadedScene> {
    let file = read_file(path, None)?;
    let (ver, info, at) = read_header(&file)?;
    let data = inflate_payload(ver, &file[at..])?;
    drop(file);
    let mut r = Reader::new(&data);

    let np = r.u32();
    if !r.ok || np != 256 {
        return Err(SceneError::Corrupt);
    }
    let mut palette: Vec<Material> = (0..np)
        .map(|_| {
            let mut m = Material { r: r.f32(), g: r.f32(), b: r.f32(), emissive: r.f32(), ..Default::default() };
            m.props = [r.f32(), r.f32(), r.f32(), r.f32()];
            if ver >= 4 {
                m.surf = [r.f32(), r.f32(), r.f32(), r.f32()];
            }
            m
        })
        .collect();
    check(r.ok)?;
    let lut = migrate_palette(&mut palette, ver);
    ensure_presets(&mut palette);

    let kind = if ver >= 3 { r.u32() } else { 0 };
    if !r.ok || kind > 1 {
        return Err(SceneError::Corrupt);
    }
    let mut en = EndlessInfo::default();
    let mut volume = None;
    if kind == 1 {
        en.endless = true;
        en.seed = r.u32();
        en.flat = r.u32() != 0;
        en.flat_cfg.height = r.i32();
        en.flat_cfg.water = r.i32();
        let nl = r.u32();
        let cfg = &en.flat_cfg;
        if !r.ok || nl > 256 || !(1..=4096).contains(&cfg.height) || !(0..=4096).contains(&cfg.water) {
            return Err(SceneError::Corrupt);
        }
        for _ in 0..nl {
            let l = FlatLayer { mat: r.u32() as u8, thickness: r.i32() };
            if !(0..=4096).contains(&l.thickness) {
                return Err(SceneError::Corrupt);
            }
            en.flat_cfg.layers.push(l);
        }
        if ver >= 5 {
            en.generator = r.u32() as i32;
            en.ny = r.i32();
            if en.generator < 1 || en.ny < 64 || en.ny > 8192 || en.ny & 7 != 0 {
                return Err(SceneError::Corrupt);
            }
            if en.generator > GEN_LATEST {
                return Err(SceneError::Newer); // (a generator this build doesn't have)
            }
        }
        if ver >= 6 {
            en.scale = r.u32() as i32;
            if !(0..=1).contains(&en.scale) {
                return Err(SceneError::Corrupt);
            }
        }
        check(r.ok)?;
    } else {
        let mut v = read_volume(&mut r, DimLimit::World)?;
        remap_voxels(&mut v, &lut);
        volume = Some(v);
    }

    let n_obj = r.u32();
    if !r.ok || n_obj > 100_000 {
        return Err(SceneError::Corrupt);
    }
    let mut objects = Vec::with_capacity(n_obj as usize);
    for _ in 0..n_obj {
        let mut vol = read_volume(&mut r, DimLimit::Max(MAX_BODY_DIM))?;
        remap_voxels(&mut vol, &lut);
        let mut o = VoxObject::new(8, 8, 8);
        o.vol = vol;
        let mut cols = [0.0f32; 9];
        for c in &mut cols {
            *c = r.f32();
        }
        o.pose.r = Mat3::from_cols_array(&cols);
        o.pose.t = r.vec3();
        o.vel = r.vec3();
        o.ang_vel = r.vec3();
        let glow = if ver >= 2 { r.f32() } else { 1.0 };
        if !r.ok || !valid_pose(&o) || !glow.is_finite() {
            return Err(SceneError::Corrupt);
        }
        o.init_body();
        o.glow_integrity = glow.clamp(0.0, 1.0);
        o.asleep = false;
        objects.push(o);
    }

    let mut st = SceneState::default();
    let mut import_slot = None;
    if ver >= 2 {
        st.has_camera = r.u32() != 0;
        st.cam_pos = [r.f32(), r.f32(), r.f32()];
        st.cam_yaw = r.f32();
        st.cam_pitch = r.f32();
        let nc = r.u32();
        if !r.ok || nc > 4096 {
            return Err(SceneError::Corrupt);
        }
        for _ in 0..nc {
            let k = r.str(128);
            let v = r.str(1024);
            st.cvars.push((k, v));
        }
        import_slot = Some(r.u32() as usize);
        let nw = r.u32();
        if !r.ok || nw > 64 << 20 {
            return Err(SceneError::Corrupt);
        }
        st.water = (0..nw)
            .map(|_| {
                let (x, y, z) = if ver >= 3 { (r.i32(), r.i32(), r.i32()) } else { (r.u16() as i32, r.u16() as i32, r.u16() as i32) };
                WaterCell { x, y, z, volume: r.f32() }
            })
            .collect();
        check(r.ok)?;
        if !st.cam_pos.iter().all(|f| f.is_finite()) {
            st.has_camera = false;
        }
    }
    en.remap = lut;
    Ok(LoadedScene { palette, volume, endless: en, objects, info, state: st, import_slot })
}

/// Before v4 saves had no surfaces (built-ins get theirs, custom colours stay matte) and slots
/// 240-255 were custom colours, now the presets; before v5 slots 224-239 were too, now the
/// terrain-2 materials. Colours in use there move to free custom slots: the returned lut maps
/// every voxel id (empty = nothing moved).
fn migrate_palette(palette: &mut [Material], ver: u32) -> Vec<u8> {
    let mut lut = Vec::new();
    if ver >= 5 {
        return lut;
    }
    if ver < 4 {
        let defaults = default_palette();
        for i in 1..IMPORT_COLOR_BASE {
            palette[i].surf = defaults[i].surf;
        }
    }
    let zero = |m: &Material| m.r == 0.0 && m.g == 0.0 && m.b == 0.0 && m.emissive == 0.0 && m.props == [0.0; 4];
    let end = if ver < 4 { 256 } else { wm::PRESET_BASE as usize };
    for i in CUSTOM_COLOR_END..end {
        if zero(&palette[i]) {
            continue;
        }
        if lut.is_empty() {
            lut = (0..=255u8).collect();
        }
        let to = match (IMPORT_COLOR_BASE..CUSTOM_COLOR_END).find(|&j| zero(&palette[j])) {
            Some(j) => {
                palette[j] = palette[i];
                j
            }
            None => {
                // The pool is full: the nearest colour there.
                let d = |j: usize| (palette[j].r - palette[i].r).powi(2) + (palette[j].g - palette[i].g).powi(2) + (palette[j].b - palette[i].b).powi(2);
                (IMPORT_COLOR_BASE..CUSTOM_COLOR_END).min_by(|&a, &b| d(a).total_cmp(&d(b))).unwrap()
            }
        };
        lut[i] = to as u8;
    }
    lut
}

// ---- chunks (region files) ----------------------------------------------------------------

/// One streaming chunk (RING_CHUNK_B*8 x ny x RING_CHUNK_B*8) as a blob: [raw size u64][zlib].
pub fn encode_chunk(chunk: &SparseVolume, level: u32) -> Vec<u8> {
    let mut w = Writer::default();
    write_volume(&mut w, chunk);
    let mut out = (w.buf.len() as u64).to_le_bytes().to_vec();
    match zlib(&w.buf, level) {
        Some(c) => out.extend_from_slice(&c),
        None => return Vec::new(),
    }
    out
}

pub fn decode_chunk(data: &[u8], ny: i32) -> Option<SparseVolume> {
    let raw = u64::from_le_bytes(data.get(..8)?.try_into().ok()?);
    if raw > 256 << 20 || raw > (data.len() as u64 - 8) * MAX_DEFLATE_RATIO {
        return None;
    }
    let mut buf = vec![0u8; raw as usize];
    if !unzlib_into(&data[8..], &mut buf) {
        return None;
    }
    let mut v = read_volume(&mut Reader::new(&buf), DimLimit::Max(4096)).ok()?;
    let side = RING_CHUNK_B * BRICK;
    if v.nx != side || v.nz != side || v.ny != ny {
        return None;
    }
    v.track_dirty = false;
    Some(v)
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::objects::make_crate;

    fn scratch(name: &str) -> std::path::PathBuf {
        let d = std::env::temp_dir().join(format!("vf-serialize-{}-{name}", std::process::id()));
        std::fs::create_dir_all(&d).unwrap();
        d.join("w.vfworld")
    }

    fn sample() -> (World, Vec<VoxObject>) {
        let mut w = World::new(256, 128, 256);
        w.generate_flat(&FlatWorld { layers: vec![FlatLayer { mat: wm::STONE, thickness: 0 }], height: 20, water: 3 });
        w.fill_sphere(100.0, 30.0, 100.0, 9.0, wm::BRICKM); // (presets would migrate in pre-v4 files)
        w.palette[40] = Material { r: 0.5, g: 0.25, b: 0.125, emissive: 0.0, props: [1.0, 0.1, 0.0, 0.0], surf: [0.9, 0.0, 0.04, 0.0] };
        w.set_next_import_slot(41);
        let mut c = make_crate(16);
        c.pose.t = Vec3::new(50.0, 40.0, 60.0);
        c.pose.r = Mat3::from_rotation_y(0.7);
        c.vel = Vec3::new(1.0, 2.0, 3.0);
        c.glow_integrity = 0.5;
        c.init_body();
        (w, vec![c])
    }

    #[test]
    fn round_trip_every_version() {
        for ver in [1, 2, 3, 4, 5, 6] {
            let (w, objs) = sample();
            let path = scratch(&format!("v{ver}"));
            let info = SceneInfo { name: "Test".into(), saved_at: 1234, thumb_w: 2, thumb_h: 1, thumb: vec![1, 2, 3, 255, 4, 5, 6, 255] };
            let st = SceneState {
                has_camera: true,
                cam_pos: [1.0, 2.0, 3.0],
                cam_yaw: 0.5,
                cam_pitch: -0.25,
                cvars: vec![("r.time.hour".into(), "13.5".into())],
                water: vec![WaterCell { x: 5, y: 20, z: 7, volume: 0.4 }],
            };
            save_scene(&path, &w, &objs, &SaveParts { info: Some(&info), state: Some(&st), endless: None, version: ver }).unwrap();
            let l = load_scene(&path).unwrap_or_else(|e| panic!("v{ver}: {e}"));
            let v = l.volume.as_ref().unwrap();
            assert_eq!((v.nx, v.ny, v.nz), (256, 128, 256));
            for p in [(100, 30, 100), (0, 0, 0), (10, 21, 10), (200, 100, 200)] {
                assert_eq!(v.get_voxel(p.0, p.1, p.2), w.get_voxel(p.0, p.1, p.2), "v{ver} voxel {p:?}");
            }
            assert_eq!(l.objects.len(), 1);
            let o = &l.objects[0];
            assert!((o.pose.t - objs[0].pose.t).length() < 1e-6 && (o.vel - objs[0].vel).length() < 1e-6);
            assert_eq!(o.vol.get_voxel(3, 3, 3), objs[0].vol.get_voxel(3, 3, 3));
            if ver >= 2 {
                assert_eq!(l.info, info);
                assert_eq!(o.glow_integrity, 0.5);
                assert_eq!(l.state.cvars, st.cvars);
                assert_eq!(l.state.water, st.water);
                assert_eq!(l.import_slot, Some(41));
            }
            if ver >= 5 {
                assert_eq!(l.palette[40], w.palette[40]);
            }
            assert_eq!(read_scene_info(&path).unwrap().name, if ver >= 2 { "Test" } else { "" });
            let _ = std::fs::remove_dir_all(path.parent().unwrap());
        }
    }

    #[test]
    fn old_custom_colours_move_out_of_the_preset_slots() {
        let (mut w, objs) = sample();
        w.palette[245] = Material { r: 0.9, g: 0.1, b: 0.1, emissive: 0.0, props: [1.0, 0.0, 0.0, 0.0], surf: [0.0; 4] };
        w.set_voxel(5, 25, 5, 245);
        let path = scratch("migrate");
        save_scene(&path, &w, &objs, &SaveParts { version: 3, ..Default::default() }).unwrap();
        let l = load_scene(&path).unwrap();
        let moved = l.volume.as_ref().unwrap().get_voxel(5, 25, 5);
        assert!((IMPORT_COLOR_BASE..CUSTOM_COLOR_END).contains(&(moved as usize)), "moved to {moved}");
        assert_eq!(l.palette[moved as usize].r, 0.9);
        assert_eq!(l.endless.remap[245], moved);
        let _ = std::fs::remove_dir_all(path.parent().unwrap());
    }

    #[test]
    fn endless_header_and_chunks() {
        let (w, _) = sample();
        let path = scratch("endless");
        let en = EndlessInfo {
            endless: true,
            seed: 99,
            flat: true,
            flat_cfg: FlatWorld { layers: vec![FlatLayer { mat: 3, thickness: 4 }], height: 64, water: 0 },
            generator: 1,
            ny: 1024,
            scale: 1,
            remap: Vec::new(),
        };
        save_scene(&path, &w, &[], &SaveParts { endless: Some(&en), ..Default::default() }).unwrap();
        let l = load_scene(&path).unwrap();
        assert!(l.volume.is_none());
        assert_eq!(l.endless, en);
        // A generated world's header (the defaults) loads too.
        let generated = EndlessInfo { endless: true, seed: 5, ..Default::default() };
        save_scene(&path, &w, &[], &SaveParts { endless: Some(&generated), ..Default::default() }).unwrap();
        assert_eq!(load_scene(&path).unwrap().endless, generated);
        let mut c = SparseVolume::new(128, 64, 128);
        c.fill_box(0, 0, 0, 127, 10, 127, 3);
        c.set_voxel(7, 40, 9, 12);
        let blob = encode_chunk(&c, 1);
        let d = decode_chunk(&blob, 64).unwrap();
        assert_eq!((d.get_voxel(7, 40, 9), d.get_voxel(100, 5, 100), d.get_voxel(100, 50, 100)), (12, 3, 0));
        assert!(decode_chunk(&blob, 128).is_none(), "wrong height");
        let _ = std::fs::remove_dir_all(path.parent().unwrap());
    }

    #[test]
    fn damage_is_detected() {
        let (w, objs) = sample();
        let path = scratch("corrupt");
        save_scene(&path, &w, &objs, &SaveParts::default()).unwrap();
        let mut b = std::fs::read(&path).unwrap();
        let n = b.len();
        b[n - 40] ^= 0x5A;
        std::fs::write(&path, &b).unwrap();
        assert_eq!(load_scene(&path).err(), Some(SceneError::Corrupt));
        b.truncate(n / 2);
        std::fs::write(&path, &b).unwrap();
        assert_eq!(load_scene(&path).err(), Some(SceneError::Corrupt));
        std::fs::write(&path, b"not a save").unwrap();
        assert_eq!(load_scene(&path).err(), Some(SceneError::NotASave));
        assert_eq!(load_scene(&path.with_extension("none")).err(), Some(SceneError::Missing));
        let _ = std::fs::remove_dir_all(path.parent().unwrap());
    }
}
#[cfg(test)]
mod bench {
    use super::*;
    #[test]
    #[ignore]
    fn save_timing() {
        let mut w = World::default();
        w.generate(1337);
        let t = std::time::Instant::now();
        let mut wr = Writer::default();
        write_volume(&mut wr, &w);
        println!("BENCH payload {} MB in {:.0} ms", wr.buf.len() >> 20, t.elapsed().as_secs_f64() * 1e3);
        let t = std::time::Instant::now();
        let c: Vec<_> = wr.buf.par_chunks(CHUNK).map(|c| zlib(c, 6).unwrap().len()).collect();
        println!("BENCH zlib6 {} MB in {:.0} ms ({} chunks, {} threads)", c.iter().sum::<usize>() >> 20, t.elapsed().as_secs_f64() * 1e3, c.len(), rayon::current_num_threads());
        for lvl in [1, 2, 3, 4, 5] {
            let t = std::time::Instant::now();
            let c: Vec<_> = wr.buf.par_chunks(CHUNK).map(|c| zlib(c, lvl).unwrap().len()).collect();
            println!("BENCH zlib{lvl} {:.1} MB in {:.0} ms", c.iter().sum::<usize>() as f64 / 1048576.0, t.elapsed().as_secs_f64() * 1e3);
        }
        let p = std::env::temp_dir().join("bench.vfworld");
        let t = std::time::Instant::now();
        save_scene(&p, &w, &[], &SaveParts::default()).unwrap();
        println!("BENCH save_scene {:.0} ms", t.elapsed().as_secs_f64() * 1e3);
        let l = load_scene(&p).unwrap();
        let v = l.volume.as_ref().unwrap();
        assert_eq!(v.grid.len(), w.grid.len());
        for gi in 0..v.grid.len() {
            let (a, b) = (v.grid[gi], w.grid[gi]);
            if a < 256 || b < 256 {
                assert_eq!(a, b, "cell {gi}");
            } else {
                assert!(v.bricks[(a - 256) as usize].vox == w.bricks[(b - 256) as usize].vox, "brick {gi}");
            }
        }
        println!("BENCH verified {} cells", v.grid.len());
        for _ in 0..2 {
            let t = std::time::Instant::now();
            let file = read_file(&p, None).unwrap();
            let t1 = t.elapsed().as_secs_f64() * 1e3;
            let (ver, _, at) = read_header(&file).unwrap();
            let data = inflate_payload(ver, &file[at..]).unwrap();
            let t2 = t.elapsed().as_secs_f64() * 1e3;
            load_scene(&p).unwrap();
            println!("BENCH read {t1:.0} inflate {:.0} | load_scene {:.0} ms ({} MB)", t2 - t1, t.elapsed().as_secs_f64() * 1e3 - t2, data.len() >> 20);
        }
        let t = std::time::Instant::now();
        write_atomic(&p, &vec![0u8; 16 << 20]).unwrap();
        println!("BENCH write_atomic 16MB {:.0} ms", t.elapsed().as_secs_f64() * 1e3);
    }
}
