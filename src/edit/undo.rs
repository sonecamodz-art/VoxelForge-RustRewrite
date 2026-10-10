//! Undo / redo for everything that edits voxels (brush, paint, detached islands, water pushed
//! aside...), for the world and for physics bodies.
//!
//! An edit is recorded by snapshotting the bricks of the region it can touch BEFORE it runs
//! (`RegionSnapshot`) and diffing afterwards: only bricks that really changed are kept, as
//! before / after states - every code path that writes voxels during the edit is caught. Bodies
//! are recorded whole, by stable id: those an edit changed or removed (before) and changed or
//! created (after).

use std::collections::{HashMap, HashSet};
use std::sync::atomic::{AtomicU32, Ordering};

use crate::objects::VoxObject;
use crate::voxel::sparse_volume::SparseVolume;

/// One brick-grid cell: a uniform value or its 512 voxels.
#[derive(Clone, Debug, PartialEq, Eq)]
pub enum BrickState {
    Uniform(u8),
    Voxels(Box<[u8; 512]>),
}

impl BrickState {
    fn bytes(&self) -> usize {
        match self {
            Self::Uniform(_) => 0,
            Self::Voxels(_) => 512,
        }
    }
}

pub fn capture_brick(v: &SparseVolume, gi: usize) -> BrickState {
    let g = v.grid[gi];
    if g < 256 { BrickState::Uniform(g as u8) } else { BrickState::Voxels(Box::new(v.bricks[(g - 256) as usize].vox)) }
}

/// (Pushes `dirty_grid`.)
pub fn restore_brick(v: &mut SparseVolume, gi: usize, s: &BrickState) {
    match s {
        BrickState::Uniform(m) => v.set_brick_uniform(gi, *m),
        BrickState::Voxels(vox) => v.set_brick_voxels(gi, vox),
    }
}

/// Does grid cell `gi` hold the same voxels as `s`? (A uniform brick equals a detailed one of
/// a single value.)
fn same_as(v: &SparseVolume, gi: usize, s: &BrickState) -> bool {
    let g = v.grid[gi];
    match (g < 256, s) {
        (true, BrickState::Uniform(u)) => g == *u as u32,
        (true, BrickState::Voxels(b)) => b.iter().all(|&x| x as u32 == g),
        (false, BrickState::Voxels(b)) => **b == v.bricks[(g - 256) as usize].vox,
        (false, BrickState::Uniform(u)) => v.bricks[(g - 256) as usize].vox.iter().all(|x| x == u),
    }
}

#[derive(Clone, Debug, Default)]
pub struct RegionSnapshot {
    /// Brick bounds, inclusive (empty: hi < lo).
    blo: [i32; 3],
    bhi: [i32; 3],
    bricks: Vec<BrickState>,
}

impl RegionSnapshot {
    fn index(&self, bx: i32, by: i32, bz: i32) -> usize {
        (((bz - self.blo[2]) * (self.bhi[1] - self.blo[1] + 1) + (by - self.blo[1])) * (self.bhi[0] - self.blo[0] + 1) + (bx - self.blo[0])) as usize
    }
    fn contains(&self, bx: i32, by: i32, bz: i32) -> bool {
        !self.bricks.is_empty() && (0..3).all(|k| [bx, by, bz][k] >= self.blo[k] && [bx, by, bz][k] <= self.bhi[k])
    }
    fn cells(&self) -> impl Iterator<Item = (i32, i32, i32)> + '_ {
        let (lo, hi) = (self.blo, self.bhi);
        (lo[2]..=hi[2]).flat_map(move |z| (lo[1]..=hi[1]).flat_map(move |y| (lo[0]..=hi[0]).map(move |x| (x, y, z))))
    }
}

/// Voxel bounds, inclusive; clamped to the volume (endless: y only - x / z are absolute).
pub fn snapshot_region(v: &SparseVolume, lo: [i32; 3], hi: [i32; 3]) -> RegionSnapshot {
    if (0..3).any(|k| hi[k] < lo[k]) {
        return RegionSnapshot::default();
    }
    let dims = [v.bricks_x, v.bricks_y, v.bricks_z];
    let mut r = RegionSnapshot::default();
    for k in 0..3 {
        let open = v.ring_w != 0 && k != 1;
        r.blo[k] = if open { lo[k] >> 3 } else { (lo[k] >> 3).clamp(0, dims[k] - 1) };
        r.bhi[k] = if open { hi[k] >> 3 } else { (hi[k] >> 3).clamp(0, dims[k] - 1) };
    }
    if v.ring_w != 0 && ((r.bhi[0] - r.blo[0] + 1) as usize) * ((r.bhi[2] - r.blo[2] + 1) as usize) > (v.bricks_x * v.bricks_z) as usize {
        return RegionSnapshot::default(); // wider than the whole window: nothing sensible to snapshot
    }
    r.bricks = r.cells().map(|(bx, by, bz)| capture_brick(v, v.grid_index(bx, by, bz))).collect();
    r
}

/// A changed brick (absolute brick coords: endless worlds keep it through streaming).
#[derive(Clone, Debug)]
pub struct BrickDelta {
    pub b: [i32; 3],
    pub before: BrickState,
    pub after: BrickState,
}

fn diff_region(before: &RegionSnapshot, v: &SparseVolume, out: &mut Vec<BrickDelta>) {
    if before.bricks.is_empty() {
        return;
    }
    for (bx, by, bz) in before.cells() {
        let gi = v.grid_index(bx, by, bz);
        let b = &before.bricks[before.index(bx, by, bz)];
        if !same_as(v, gi, b) {
            out.push(BrickDelta { b: [bx, by, bz], before: b.clone(), after: capture_brick(v, gi) });
        }
    }
}

#[derive(Clone, Default)]
pub struct EditRecord {
    /// "Brush", "Paint"...
    pub label: String,
    pub world: Vec<BrickDelta>,
    /// Bodies changed / removed (state before).
    pub bodies_before: Vec<VoxObject>,
    /// Bodies changed / created (state after).
    pub bodies_after: Vec<VoxObject>,
    /// World voxel bounds of the changes (hi < lo: none).
    pub lo: [i32; 3],
    pub hi: [i32; 3],
}

fn body_bytes(o: &VoxObject) -> usize {
    std::mem::size_of::<VoxObject>() + o.vol.grid.len() * 4 + o.vol.bricks.len() * 600 + o.shell.len() * 4
}

fn brick_key(b: [i32; 3]) -> u64 {
    ((b[0] as u32 as u64 & 0x3FF_FFFF) << 38) | ((b[2] as u32 as u64 & 0x3FF_FFFF) << 12) | (b[1] as u32 as u64 & 0xFFF)
}

impl EditRecord {
    pub fn bytes(&self) -> usize {
        std::mem::size_of::<Self>()
            + self.world.iter().map(|d| std::mem::size_of::<BrickDelta>() + d.before.bytes() + d.after.bytes()).sum::<usize>()
            + self.bodies_before.iter().chain(&self.bodies_after).map(body_bytes).sum::<usize>()
    }

    pub fn is_empty(&self) -> bool {
        self.world.is_empty() && self.bodies_before.is_empty() && self.bodies_after.is_empty()
    }

    fn grow_bounds(&mut self, lo: [i32; 3], hi: [i32; 3]) {
        for k in 0..3 {
            if hi[k] < lo[k] {
                continue;
            }
            if self.hi[k] < self.lo[k] {
                (self.lo[k], self.hi[k]) = (lo[k], hi[k]);
            } else {
                self.lo[k] = self.lo[k].min(lo[k]);
                self.hi[k] = self.hi[k].max(hi[k]);
            }
        }
    }

    /// Fold a later edit into this one (a brush stroke = one undo step): a brick / body keeps
    /// its first "before" and takes the later "after".
    pub fn append(&mut self, later: EditRecord) {
        if self.label.is_empty() {
            self.label = later.label;
        }
        let mut at: HashMap<u64, usize> = self.world.iter().enumerate().map(|(i, d)| (brick_key(d.b), i)).collect();
        for d in later.world {
            match at.get(&brick_key(d.b)) {
                Some(&i) => self.world[i].after = d.after,
                None => {
                    at.insert(brick_key(d.b), self.world.len());
                    self.world.push(d);
                }
            }
        }
        self.grow_bounds(later.lo, later.hi);
        let pos = |v: &[VoxObject], id: u32| v.iter().position(|o| o.id == id);
        for b in later.bodies_before {
            let removed_by_later = pos(&later.bodies_after, b.id).is_none();
            if let Some(i) = pos(&self.bodies_after, b.id) {
                // We already have its state: the later edit starts from ours.
                if removed_by_later {
                    self.bodies_after.remove(i);
                }
                continue;
            }
            if pos(&self.bodies_before, b.id).is_none() {
                self.bodies_before.push(b);
            }
        }
        for a in later.bodies_after {
            match pos(&self.bodies_after, a.id) {
                Some(i) => self.bodies_after[i] = a,
                None => self.bodies_after.push(a),
            }
        }
    }

    /// Floating origin: world coordinates moved by (dx, dz) voxels (multiples of 8).
    pub fn shift(&mut self, dx: i32, dz: i32) {
        for d in &mut self.world {
            d.b[0] -= dx >> 3;
            d.b[2] -= dz >> 3;
        }
        if self.hi[0] >= self.lo[0] {
            self.lo[0] -= dx;
            self.hi[0] -= dx;
        }
        if self.hi[2] >= self.lo[2] {
            self.lo[2] -= dz;
            self.hi[2] -= dz;
        }
        for o in self.bodies_before.iter_mut().chain(&mut self.bodies_after) {
            o.shift_world(dx as f32, dz as f32);
        }
    }
}

static NEXT_BODY_ID: AtomicU32 = AtomicU32::new(1);

/// Gives every body without an id (or with a duplicated one, e.g. a split copy) a fresh id.
pub fn ensure_body_ids(objects: &mut [VoxObject]) {
    let mut seen = HashSet::new();
    for o in objects {
        if o.id == 0 || !seen.insert(o.id) {
            o.id = NEXT_BODY_ID.fetch_add(1, Ordering::Relaxed);
            seen.insert(o.id);
        }
        NEXT_BODY_ID.fetch_max(o.id + 1, Ordering::Relaxed);
    }
}

pub fn find_body(objects: &[VoxObject], id: u32) -> Option<usize> {
    objects.iter().position(|o| o.id == id)
}

/// Records one edit: create it before the edit (region = the world voxels it may touch;
/// `body` = the body it edits), run the edit, then `finish`.
pub struct EditScope {
    snap: Option<RegionSnapshot>,
    extra: Vec<RegionSnapshot>,
    ids_before: Vec<u32>,
    body_before: Option<VoxObject>,
    label: String,
}

impl EditScope {
    pub fn new(world: &SparseVolume, objects: &mut [VoxObject], region: Option<([i32; 3], [i32; 3])>, body: Option<usize>, label: &str) -> Self {
        ensure_body_ids(objects);
        Self {
            snap: region.map(|(lo, hi)| snapshot_region(world, lo, hi)),
            extra: Vec::new(),
            ids_before: objects.iter().map(|o| o.id).collect(),
            body_before: body.and_then(|i| objects.get(i).cloned()),
            label: label.to_string(),
        }
    }

    /// Also record the region lo..hi as it is NOW: changes found during the edit outside its
    /// first region (a felled tree's top, reported before it leaves the world). Bricks already
    /// recorded keep their first state.
    pub fn extend(&mut self, world: &SparseVolume, lo: [i32; 3], hi: [i32; 3]) {
        self.extra.push(snapshot_region(world, lo, hi));
    }

    pub fn finish(self, world: &SparseVolume, objects: &mut [VoxObject]) -> EditRecord {
        let mut r = EditRecord { label: self.label, lo: [0; 3], hi: [-1; 3], ..Default::default() };
        ensure_body_ids(objects);
        if let Some(s) = &self.snap {
            diff_region(s, world, &mut r.world);
        }
        // The extended regions: bricks no earlier snapshot holds (the first state wins).
        let mut seen: HashSet<u64> = r.world.iter().map(|d| brick_key(d.b)).collect();
        let mut claimed: HashSet<u64> = HashSet::new();
        let mut tmp = Vec::new();
        for e in &self.extra {
            tmp.clear();
            diff_region(e, world, &mut tmp);
            for d in tmp.drain(..) {
                let k = brick_key(d.b);
                let in_main = self.snap.as_ref().is_some_and(|s| s.contains(d.b[0], d.b[1], d.b[2]));
                if in_main || claimed.contains(&k) || !seen.insert(k) {
                    continue;
                }
                r.world.push(d);
            }
            if !e.bricks.is_empty() {
                claimed.extend(e.cells().map(|(x, y, z)| brick_key([x, y, z])));
            }
        }
        for i in 0..r.world.len() {
            let b = r.world[i].b.map(|v| v * 8);
            r.grow_bounds(b, b.map(|v| v + 7));
        }
        // Bodies: the edited one (before / after) and any the edit created (split pieces,
        // detached islands).
        if let Some(b) = self.body_before {
            if let Some(i) = find_body(objects, b.id) {
                r.bodies_after.push(objects[i].clone());
            }
            r.bodies_before.push(b);
        }
        let before: HashSet<u32> = self.ids_before.into_iter().collect();
        r.bodies_after.extend(objects.iter().filter(|o| !before.contains(&o.id)).cloned());
        r
    }
}

/// What an undo / redo changed, so the caller can re-sync the GPU.
#[derive(Clone, Debug, Default)]
pub struct Applied {
    pub any: bool,
    pub world: bool,
    pub bodies: bool,
    pub lo: [i32; 3],
    pub hi: [i32; 3],
    pub label: String,
}

pub struct UndoStack {
    undo: Vec<EditRecord>,
    redo: Vec<EditRecord>,
    bytes: usize,
    limit: usize,
}

impl Default for UndoStack {
    fn default() -> Self {
        Self { undo: Vec::new(), redo: Vec::new(), bytes: 0, limit: 512 << 20 }
    }
}

impl UndoStack {
    /// Drops the redo branch; enforces the memory limit (oldest first, the newest kept).
    pub fn push(&mut self, r: EditRecord) {
        if r.is_empty() {
            return;
        }
        self.bytes -= self.redo.drain(..).map(|x| x.bytes()).sum::<usize>();
        self.bytes += r.bytes();
        self.undo.push(r);
        self.trim();
    }

    fn trim(&mut self) {
        let mut drop = 0;
        while self.bytes > self.limit && drop + 1 < self.undo.len() {
            self.bytes -= self.undo[drop].bytes();
            drop += 1;
        }
        self.undo.drain(..drop);
    }

    pub fn set_limit(&mut self, bytes: usize) {
        self.limit = bytes;
        self.trim();
    }

    pub fn clear(&mut self) {
        *self = Self { limit: self.limit, ..Default::default() };
    }

    pub fn can_undo(&self) -> bool {
        !self.undo.is_empty()
    }
    pub fn can_redo(&self) -> bool {
        !self.redo.is_empty()
    }
    pub fn len(&self) -> (usize, usize) {
        (self.undo.len(), self.redo.len())
    }
    pub fn bytes(&self) -> usize {
        self.bytes
    }

    /// Fold `r` into the newest step (a stroke continuing).
    pub fn merge_into_last(&mut self, r: EditRecord) {
        match self.undo.last_mut() {
            Some(last) if !r.is_empty() => {
                self.bytes -= last.bytes();
                last.append(r);
                self.bytes += last.bytes();
                self.trim();
            }
            _ => self.push(r),
        }
    }

    /// Floating origin.
    pub fn shift(&mut self, dx: i32, dz: i32) {
        for r in self.undo.iter_mut().chain(&mut self.redo) {
            r.shift(dx, dz);
        }
    }

    fn apply(r: &EditRecord, forward: bool, world: &mut SparseVolume, objects: &mut Vec<VoxObject>) -> Applied {
        let mut a = Applied { any: true, label: r.label.clone(), lo: [0; 3], hi: [-1; 3], ..Default::default() };
        for d in &r.world {
            // Endless: a brick whose chunk streamed out since is left alone (its chunk was saved
            // as it was; the rest of the edit still undoes).
            if world.ring_w != 0 && !world.chunk_loaded(d.b[0] >> 4, d.b[2] >> 4) {
                continue;
            }
            let gi = world.grid_index(d.b[0], d.b[1], d.b[2]);
            restore_brick(world, gi, if forward { &d.after } else { &d.before });
        }
        if !r.world.is_empty() {
            (a.world, a.lo, a.hi) = (true, r.lo, r.hi);
        }
        // Bodies: remove what the other side has, then put this side's copies back.
        let (remove, add) = if forward { (&r.bodies_before, &r.bodies_after) } else { (&r.bodies_after, &r.bodies_before) };
        for o in remove {
            if let Some(i) = find_body(objects, o.id) {
                objects.remove(i);
                a.bodies = true;
            }
        }
        for o in add {
            let mut o = o.clone();
            o.asleep = false;
            match find_body(objects, o.id) {
                Some(i) => objects[i] = o,
                None => objects.push(o),
            }
            a.bodies = true;
        }
        a
    }

    pub fn undo(&mut self, world: &mut SparseVolume, objects: &mut Vec<VoxObject>) -> Applied {
        let Some(r) = self.undo.pop() else { return Applied::default() };
        let a = Self::apply(&r, false, world, objects);
        self.redo.push(r);
        a
    }

    pub fn redo(&mut self, world: &mut SparseVolume, objects: &mut Vec<VoxObject>) -> Applied {
        let Some(r) = self.redo.pop() else { return Applied::default() };
        let a = Self::apply(&r, true, world, objects);
        self.undo.push(r);
        a
    }

    /// Every voxel byte the history holds (palette slots it still references).
    pub fn for_each_voxels(&self, mut f: impl FnMut(&[u8])) {
        let mut state = |s: &BrickState| match s {
            BrickState::Uniform(m) => f(std::slice::from_ref(m)),
            BrickState::Voxels(v) => f(&v[..]),
        };
        for r in self.undo.iter().chain(&self.redo) {
            for d in &r.world {
                state(&d.before);
                state(&d.after);
            }
        }
        for r in self.undo.iter().chain(&self.redo) {
            for o in r.bodies_before.iter().chain(&r.bodies_after) {
                for &g in &o.vol.grid {
                    if g < 256 {
                        f(&[g as u8]);
                    }
                }
                for b in &o.vol.bricks {
                    f(&b.vox);
                }
            }
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::edit::brush::{apply_brush, BrushBox, BrushShape};
    use crate::objects::make_crate;

    fn world() -> SparseVolume {
        let mut v = SparseVolume::new(64, 32, 64);
        v.fill_box(0, 0, 0, 63, 9, 63, 3);
        v.compact();
        v
    }

    #[test]
    fn brush_undo_redo_and_strokes() {
        let mut v = world();
        let mut objs = Vec::new();
        let mut st = UndoStack::default();
        let b = BrushBox { lo: [10, 6, 10], size: 6, shape: BrushShape::Sphere };
        let (lo, hi) = b.bounds();
        let s = EditScope::new(&v, &mut objs, Some((lo, hi)), None, "Brush");
        apply_brush(&mut v, &b, 0, |_, _, _, _| {});
        let r = s.finish(&v, &mut objs);
        assert!(!r.world.is_empty() && r.world.len() <= 8);
        st.push(r);
        // A second dab of the same stroke folds into it.
        let b2 = BrushBox { lo: [14, 6, 10], ..b };
        let (lo, hi) = b2.bounds();
        let s = EditScope::new(&v, &mut objs, Some((lo, hi)), None, "Brush");
        apply_brush(&mut v, &b2, 0, |_, _, _, _| {});
        st.merge_into_last(s.finish(&v, &mut objs));
        assert_eq!(st.len(), (1, 0));
        assert_eq!(v.get_voxel(13, 9, 13), 0);
        let a = st.undo(&mut v, &mut objs);
        assert!(a.world && a.lo[0] <= 8 && a.hi[0] >= 23);
        assert_eq!((v.get_voxel(13, 9, 13), v.get_voxel(17, 9, 13)), (3, 3), "both dabs undone");
        st.redo(&mut v, &mut objs);
        assert_eq!((v.get_voxel(13, 9, 13), v.get_voxel(17, 9, 13)), (0, 0));
        // A new edit drops the redo branch.
        st.undo(&mut v, &mut objs);
        let s = EditScope::new(&v, &mut objs, Some(([0, 9, 0], [0, 9, 0])), None, "Brush");
        v.set_voxel(0, 9, 0, 0);
        st.push(s.finish(&v, &mut objs));
        assert_eq!(st.len(), (1, 0), "the undone stroke is gone with its redo branch");
    }

    #[test]
    fn bodies_created_and_removed_are_undone() {
        let mut v = world();
        let mut objs = vec![make_crate(8)];
        let mut st = UndoStack::default();
        let s = EditScope::new(&v, &mut objs, None, Some(0), "Split");
        objs[0].vol.set_voxel(0, 0, 0, 0);
        objs.push(make_crate(8)); // a piece split off
        let r = s.finish(&v, &mut objs);
        assert_eq!((r.bodies_before.len(), r.bodies_after.len()), (1, 2));
        st.push(r);
        st.undo(&mut v, &mut objs);
        assert_eq!(objs.len(), 1);
        assert_ne!(objs[0].vol.get_voxel(0, 0, 0), 0);
        st.redo(&mut v, &mut objs);
        assert_eq!(objs.len(), 2);
    }

    #[test]
    fn extended_regions_and_limits() {
        let mut v = world();
        let mut objs = Vec::new();
        let mut s = EditScope::new(&v, &mut objs, Some(([0, 0, 0], [7, 7, 7])), None, "Carve");
        v.set_voxel(1, 1, 1, 0);
        s.extend(&v, [40, 0, 40], [47, 9, 47]);
        v.set_voxel(41, 5, 41, 0);
        let r = s.finish(&v, &mut objs);
        assert_eq!(r.world.len(), 2);
        let mut st = UndoStack::default();
        st.set_limit(1);
        st.push(r.clone());
        st.push(r);
        assert_eq!(st.len().0, 1, "over the limit: only the newest kept");
    }
}
