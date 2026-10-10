//! Chunk storage for endless worlds: only chunks that differ from the generator are kept
//! (generation is deterministic), in region files (the C++ engine's format)
//!   <world>.chunks/r.<rx>.<rz>.vfr    8 x 8 chunks each (1024 x 1024 voxels)
//!   [magic "VFRG"][version u32][64 x {offset u64, size u32, pad u32}][blobs...]
//! a blob = `serialize::encode_chunk`. A region is rewritten whole (temp file + rename: a crash
//! mid-save leaves the old one) when one of its chunks changed.
//!
//! Threads: `get` is called by the stream workers; `put` / `flush` by the main thread. Chunks put
//! since the last flush stay in memory and win over the disk; a store without a folder (a world
//! not saved yet) keeps everything in memory until the first flush gives it one.

use std::collections::{BTreeMap, HashMap};
use std::io::{Read, Seek, SeekFrom};
use std::path::{Path, PathBuf};
use std::sync::Arc;

use parking_lot::Mutex;
use rayon::prelude::*;

use crate::serialize::{decode_chunk, encode_chunk, remap_voxels, write_atomic};
use crate::voxel::sparse_volume::SparseVolume;

/// Chunks per region side.
pub const REGION: i32 = 8;
const MAGIC: u32 = 0x4752_4656; // "VFRG"
/// 1: palette slots 240-255 could hold custom colours (now presets: `migrate_regions`).
const VERSION: u32 = 2;
const SLOTS: usize = (REGION * REGION) as usize;
const HEADER: usize = 8 + SLOTS * 16;

type Key = (i32, i32);

fn region_of(cx: i32, cz: i32) -> Key {
    (cx.div_euclid(REGION), cz.div_euclid(REGION))
}
fn slot_index(cx: i32, cz: i32) -> usize {
    (cz.rem_euclid(REGION) * REGION + cx.rem_euclid(REGION)) as usize
}
fn region_path(dir: &Path, r: Key) -> PathBuf {
    dir.join(format!("r.{}.{}.vfr", r.0, r.1))
}

#[derive(Clone, Copy, Default)]
struct Slot {
    off: u64,
    size: u32,
}

#[derive(Clone)]
enum Entry {
    Raw(Arc<SparseVolume>),
    Blob(Arc<Vec<u8>>),
}

#[derive(Default)]
struct Inner {
    dir: PathBuf,
    ny: i32,
    mem: BTreeMap<Key, Entry>,
    /// Cached region tables (None: the region has no file).
    tables: HashMap<Key, Option<Vec<Slot>>>,
}

#[derive(Default)]
pub struct ChunkStore {
    inner: Mutex<Inner>,
}

fn parse_table(head: &[u8]) -> Option<(u32, Vec<Slot>)> {
    if head.len() < HEADER {
        return None;
    }
    let magic = u32::from_le_bytes(head[0..4].try_into().unwrap());
    let ver = u32::from_le_bytes(head[4..8].try_into().unwrap());
    if magic != MAGIC {
        return None;
    }
    let t = (0..SLOTS)
        .map(|i| {
            let b = &head[8 + i * 16..];
            Slot { off: u64::from_le_bytes(b[0..8].try_into().unwrap()), size: u32::from_le_bytes(b[8..12].try_into().unwrap()) }
        })
        .collect();
    Some((ver, t))
}

/// A whole region file: header + the slots' blobs in order.
fn region_bytes(slots: &[Vec<u8>]) -> (Vec<u8>, Vec<Slot>) {
    let mut out = Vec::with_capacity(HEADER + slots.iter().map(Vec::len).sum::<usize>());
    out.extend_from_slice(&MAGIC.to_le_bytes());
    out.extend_from_slice(&VERSION.to_le_bytes());
    let mut table = Vec::with_capacity(SLOTS);
    let mut off = HEADER as u64;
    for s in slots {
        let e = Slot { off: if s.is_empty() { 0 } else { off }, size: s.len() as u32 };
        out.extend_from_slice(&e.off.to_le_bytes());
        out.extend_from_slice(&e.size.to_le_bytes());
        out.extend_from_slice(&[0; 4]);
        table.push(e);
        off += s.len() as u64;
    }
    for s in slots {
        out.extend_from_slice(s);
    }
    (out, table)
}

impl Inner {
    fn table(&mut self, r: Key) -> Option<&Vec<Slot>> {
        if !self.tables.contains_key(&r) {
            let mut t = None;
            if !self.dir.as_os_str().is_empty() {
                let path = region_path(&self.dir, r);
                let mut head = vec![0u8; HEADER];
                if let Ok(mut f) = std::fs::File::open(&path)
                    && f.read_exact(&mut head).is_ok()
                {
                    match parse_table(&head) {
                        Some((1 | VERSION, tab)) => t = Some(tab),
                        _ => log::warn!(target: "saves", "region {}: not a region file (ignored)", path.display()),
                    }
                }
            }
            self.tables.insert(r, t);
        }
        self.tables[&r].as_ref()
    }

    fn read_blob(&mut self, r: Key, idx: usize) -> Option<Vec<u8>> {
        let s = *self.table(r)?.get(idx)?;
        if s.size == 0 {
            return None;
        }
        let mut f = std::fs::File::open(region_path(&self.dir, r)).ok()?;
        f.seek(SeekFrom::Start(s.off)).ok()?;
        let mut out = vec![0u8; s.size as usize];
        f.read_exact(&mut out).ok()?;
        Some(out)
    }
}

impl ChunkStore {
    /// Folder "" = memory only. Drops every chunk held in memory.
    pub fn open(&self, dir: &Path, ny: i32) {
        *self.inner.lock() = Inner { dir: dir.to_path_buf(), ny, ..Default::default() };
    }

    pub fn dir(&self) -> PathBuf {
        self.inner.lock().dir.clone()
    }

    /// (Worker threads) A stored chunk, None = not stored (generate it).
    pub fn get(&self, cx: i32, cz: i32) -> Option<SparseVolume> {
        let (entry, disk, ny) = {
            let mut g = self.inner.lock();
            let ny = g.ny;
            match g.mem.get(&(cx, cz)).cloned() {
                Some(e) => (Some(e), None, ny),
                None => (None, Some(g.read_blob(region_of(cx, cz), slot_index(cx, cz))?), ny),
            }
        };
        let blob = match entry {
            Some(Entry::Raw(v)) => return Some((*v).clone()),
            Some(Entry::Blob(b)) => b,
            None => Arc::new(disk.unwrap()),
        };
        let v = decode_chunk(&blob, ny);
        if v.is_none() {
            log::warn!(target: "saves", "chunk ({cx}, {cz}) is damaged - generated instead");
        }
        v
    }

    /// (Main thread) A modified chunk (evicted, or copied out for a save).
    pub fn put(&self, cx: i32, cz: i32, chunk: SparseVolume) {
        self.inner.lock().mem.insert((cx, cz), Entry::Raw(Arc::new(chunk)));
    }

    /// An already encoded chunk (a converted save's).
    pub fn put_blob(&self, cx: i32, cz: i32, blob: Vec<u8>) {
        self.inner.lock().mem.insert((cx, cz), Entry::Blob(Arc::new(blob)));
    }

    pub fn pending_chunks(&self) -> usize {
        self.inner.lock().mem.len()
    }

    /// Writes the changed regions into `dir` (a different folder = "save as": the regions
    /// already on disk come along). Err on a write error (nothing in memory is lost).
    pub fn flush(&self, dir: &Path) -> Result<(), String> {
        if dir.as_os_str().is_empty() {
            return Ok(());
        }
        let mut g = self.inner.lock();
        std::fs::create_dir_all(dir).map_err(|e| format!("can't create {}: {e}", dir.display()))?;
        if g.dir != dir {
            // A new folder (first save, save as, overwriting another world): what it held belongs
            // to another world; this world's regions come along (a region file is self-contained).
            let vfr = |d: &Path| -> Vec<PathBuf> {
                std::fs::read_dir(d).map(|rd| rd.flatten().map(|e| e.path()).filter(|p| p.extension().is_some_and(|x| x == "vfr")).collect()).unwrap_or_default()
            };
            for p in vfr(dir) {
                let _ = std::fs::remove_file(p);
            }
            if !g.dir.as_os_str().is_empty() {
                for p in vfr(&g.dir) {
                    let _ = std::fs::copy(&p, dir.join(p.file_name().unwrap()));
                }
            }
            g.tables.clear();
            g.dir = dir.to_path_buf();
        }
        if g.mem.is_empty() {
            return Ok(());
        }
        // Encode the raw chunks on every core.
        let list: Vec<(Key, Entry)> = g.mem.iter().map(|(k, e)| (*k, e.clone())).collect();
        let blobs: Vec<Arc<Vec<u8>>> = list
            .par_iter()
            .map(|(_, e)| match e {
                Entry::Blob(b) => b.clone(),
                Entry::Raw(v) => Arc::new(encode_chunk(v, 6)),
            })
            .collect();
        let mut by_region: BTreeMap<Key, Vec<usize>> = BTreeMap::new();
        for (i, ((cx, cz), _)) in list.iter().enumerate() {
            by_region.entry(region_of(*cx, *cz)).or_default().push(i);
        }
        let mut err = None;
        for (rk, items) in by_region {
            // Every slot's blob: the new ones, else what the file already holds.
            let mut slots = vec![Vec::new(); SLOTS];
            let mut fresh = [false; SLOTS];
            for &i in &items {
                let idx = slot_index(list[i].0.0, list[i].0.1);
                slots[idx] = (*blobs[i]).clone();
                fresh[idx] = true;
            }
            let mut read_ok = true;
            for idx in 0..SLOTS {
                if !fresh[idx] && g.table(rk).is_some_and(|t| t[idx].size > 0) {
                    match g.read_blob(rk, idx) {
                        Some(b) => slots[idx] = b,
                        None => read_ok = false,
                    }
                }
            }
            let path = region_path(dir, rk);
            if !read_ok {
                // Never write a region that would lose chunks it had.
                log::error!(target: "saves", "region {} can't be read back - not rewritten", path.display());
                err = Some(format!("{} can't be read", path.display()));
                continue;
            }
            let (bytes, table) = region_bytes(&slots);
            if let Err(e) = write_atomic(&path, &bytes) {
                log::error!(target: "saves", "writing {}: {e}", path.display());
                err = Some(format!("writing {}: {e}", path.display()));
                continue;
            }
            g.tables.insert(rk, Some(table));
            for &i in &items {
                g.mem.remove(&list[i].0); // on disk now
            }
        }
        err.map_or(Ok(()), Err)
    }
}

/// Version 1 region files of a world whose custom colours moved (`EndlessInfo::remap`): every
/// chunk through lut, rewritten as version 2 (version 2 files are left alone, so running it
/// twice changes nothing). Returns how many files were rewritten.
pub fn migrate_regions(dir: &Path, lut: &[u8], ny: i32) -> usize {
    if lut.len() < 256 {
        return 0;
    }
    let Ok(rd) = std::fs::read_dir(dir) else { return 0 };
    let mut done = 0;
    for path in rd.flatten().map(|e| e.path()).filter(|p| p.extension().is_some_and(|x| x == "vfr")) {
        let Ok(file) = std::fs::read(&path) else { continue };
        let Some((1, table)) = parse_table(&file) else { continue };
        let slots: Option<Vec<Vec<u8>>> = table
            .iter()
            .map(|s| {
                if s.size == 0 {
                    return Some(Vec::new());
                }
                let b = file.get(s.off as usize..(s.off + s.size as u64) as usize)?;
                let mut c = decode_chunk(b, ny)?;
                remap_voxels(&mut c, lut);
                Some(encode_chunk(&c, 6))
            })
            .collect();
        let Some(slots) = slots else {
            log::error!(target: "saves", "region {}: can't migrate (damaged) - left as it is", path.display());
            continue;
        };
        match write_atomic(&path, &region_bytes(&slots).0) {
            Ok(()) => done += 1,
            Err(e) => log::error!(target: "saves", "migrating {}: {e}", path.display()),
        }
    }
    if done > 0 {
        log::info!(target: "saves", "{}: {done} region file(s) migrated (custom colours moved out of the preset slots)", dir.display());
    }
    done
}

#[cfg(test)]
mod tests {
    use super::*;

    fn chunk(ny: i32, mark: u8) -> SparseVolume {
        let mut c = SparseVolume::new(128, ny, 128);
        c.fill_box(0, 0, 0, 127, 7, 127, 3);
        c.set_voxel(5, 20, 6, mark);
        c
    }

    #[test]
    fn put_flush_get_across_regions() {
        let dir = std::env::temp_dir().join(format!("vf-region-{}", std::process::id()));
        let _ = std::fs::remove_dir_all(&dir);
        let s = ChunkStore::default();
        s.open(Path::new(""), 64);
        s.put(0, 0, chunk(64, 10));
        s.put(-1, 9, chunk(64, 11)); // region (-1, 1)
        assert_eq!(s.get(0, 0).unwrap().get_voxel(5, 20, 6), 10, "memory before any folder");
        assert!(s.get(1, 1).is_none());
        s.flush(&dir).unwrap();
        assert_eq!(s.pending_chunks(), 0);
        assert!(dir.join("r.0.0.vfr").exists() && dir.join("r.-1.1.vfr").exists());
        // A second chunk in the same region keeps the first.
        s.put(7, 7, chunk(64, 12));
        s.flush(&dir).unwrap();
        let t = ChunkStore::default();
        t.open(&dir, 64);
        assert_eq!(t.get(0, 0).unwrap().get_voxel(5, 20, 6), 10);
        assert_eq!(t.get(7, 7).unwrap().get_voxel(5, 20, 6), 12);
        assert_eq!(t.get(-1, 9).unwrap().get_voxel(5, 20, 6), 11);
        assert!(t.get(3, 3).is_none());
        // Save as: the regions come along to the new folder.
        let dir2 = dir.with_extension("copy");
        t.put(1, 0, chunk(64, 13));
        t.flush(&dir2).unwrap();
        let u = ChunkStore::default();
        u.open(&dir2, 64);
        assert_eq!(u.get(0, 0).unwrap().get_voxel(5, 20, 6), 10);
        assert_eq!(u.get(1, 0).unwrap().get_voxel(5, 20, 6), 13);
        let _ = std::fs::remove_dir_all(&dir);
        let _ = std::fs::remove_dir_all(&dir2);
    }
}
