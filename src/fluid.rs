//! Water as a volume-conserving cellular fluid ("CA water"), CPU-side.
//!
//! Game state stays one byte per voxel: a WATER voxel with no entry in the partial field holds
//! exactly 1.0 volume. Partly full cells (0 < v < 1) keep their volume in a sparse float field;
//! their voxel shows WATER from v >= 0.5 and turns back to AIR below 0.3 (hysteresis, so a cell
//! doesn't blink). Only ACTIVE cells are updated: generated lakes are at rest, and edits
//! (`wake`) or flowing water re-activate cells.
//!
//! Per tick (fixed 20 Hz) each active cell: is refilled from its body's surface if it sits
//! inside one ("pressure"); flows down into the free capacity below; when supported shares with
//! lower horizontal neighbours, (v - vN) / 5 each, until it is a thin puddle; crumbs vanish. A
//! cell whose change stays below the sleep threshold drops out of the active set.
//!
//! Lakes drain level: sideways sharing alone is diffusion, so every LEVEL_TICKS the surface
//! layers touched by the flow are flood-filled and set to their average volume. A layer drained
//! below MIN_SPREAD empties into the water under it. Active cells are a FIFO, capped per tick.

use std::cell::Cell;
use std::collections::VecDeque;

use crate::hash::{VoxMap, VoxSet};
use crate::voxel::sparse_volume::SparseVolume;

pub const WATER: u8 = 5;
const SHOW_AT: f32 = 0.5;
const HIDE_BELOW: f32 = 0.3;
const EVAPORATE: f32 = 1e-4;
/// Thinner water stops spreading sideways.
const MIN_SPREAD: f32 = 0.05;
/// Smaller horizontal transfers are skipped.
const MIN_FLOW: f32 = 1e-4;
const SLEEP: f32 = 1e-3;
pub const TICK_HZ: f32 = 20.0;
const MAX_CELLS_PER_TICK: usize = 50_000;
/// Surface levelling every 0.5 s.
const LEVEL_TICKS: u32 = 10;
/// One layer's flood-fill cap.
const MAX_LEVEL_CELLS: usize = 400_000;
/// Cells a pressure pull may search / pulls per tick.
const PRESSURE_SEARCH: usize = 512;
const MAX_PULLS_PER_TICK: usize = 4096;
/// Cells levelled per tick (a layer is never split).
const LEVEL_BUDGET: usize = 40_000;
const MAX_LEVEL_SEEDS: usize = 4096;

/// 21-bit key fields with a bias, so negative coordinates (endless worlds) pack too.
#[inline]
fn field(v: i32) -> u64 {
    (v.wrapping_add(1 << 20) as u32 & 0x1F_FFFF) as u64
}
#[inline]
fn field_value(f: u64) -> i32 {
    (f & 0x1F_FFFF) as i32 - (1 << 20)
}
#[inline]
fn key(x: i32, y: i32, z: i32) -> u64 {
    field(x) | field(y) << 21 | field(z) << 42
}
#[inline]
fn unkey(k: u64) -> (i32, i32, i32) {
    (field_value(k), field_value(k >> 21), field_value(k >> 42))
}

struct Page {
    v: [f32; 512],
    count: u32,
}

/// Sparse per-8^3-page float field (0 = no entry): the partial volumes. Pages live in a slab;
/// a one-entry cache of the last page skips the hash probe for neighbouring cells.
#[derive(Default)]
pub struct PartialField {
    index: VoxMap<u32>,
    pages: Vec<Page>,
    free: Vec<u32>,
    /// (page key, slab index + 1; 0 = no page)
    last: Cell<(u64, u32)>,
    count: usize,
}

impl PartialField {
    #[inline]
    fn page_key(x: i32, y: i32, z: i32) -> u64 {
        key(x >> 3, y >> 3, z >> 3)
    }
    #[inline]
    fn local(x: i32, y: i32, z: i32) -> usize {
        ((x & 7) | (y & 7) << 3 | (z & 7) << 6) as usize
    }
    #[inline]
    fn page(&self, k: u64) -> Option<usize> {
        let (lk, li) = self.last.get();
        let i = if lk == k {
            li
        } else {
            let i = self.index.get(&k).map_or(0, |&i| i + 1);
            self.last.set((k, i));
            i
        };
        (i != 0).then(|| (i - 1) as usize)
    }

    pub fn get(&self, x: i32, y: i32, z: i32) -> f32 {
        self.page(Self::page_key(x, y, z)).map_or(0.0, |p| self.pages[p].v[Self::local(x, y, z)])
    }

    /// v > 0.
    pub fn set(&mut self, x: i32, y: i32, z: i32, v: f32) {
        let k = Self::page_key(x, y, z);
        let p = match self.page(k) {
            Some(p) => p,
            None => {
                let page = Page { v: [0.0; 512], count: 0 };
                let i = match self.free.pop() {
                    Some(i) => {
                        self.pages[i as usize] = page;
                        i
                    }
                    None => {
                        self.pages.push(page);
                        self.pages.len() as u32 - 1
                    }
                };
                self.index.insert(k, i);
                self.last.set((k, i + 1));
                i as usize
            }
        };
        let page = &mut self.pages[p];
        let c = &mut page.v[Self::local(x, y, z)];
        if *c <= 0.0 {
            page.count += 1;
            self.count += 1;
        }
        *c = v;
    }

    pub fn erase(&mut self, x: i32, y: i32, z: i32) {
        let k = Self::page_key(x, y, z);
        let Some(p) = self.page(k) else { return };
        let page = &mut self.pages[p];
        let c = &mut page.v[Self::local(x, y, z)];
        if *c <= 0.0 {
            return;
        }
        *c = 0.0;
        self.count -= 1;
        page.count -= 1;
        if page.count == 0 {
            self.index.remove(&k);
            self.free.push(p as u32);
            self.last.set((u64::MAX, 0));
        }
    }

    pub fn len(&self) -> usize {
        self.count
    }

    pub fn clear(&mut self) {
        *self = Self::default();
    }

    /// f(x, y, z, volume) for every entry.
    pub fn for_each(&self, mut f: impl FnMut(i32, i32, i32, f32)) {
        for (&k, &i) in &self.index {
            let (px, py, pz) = unkey(k);
            for (j, &v) in self.pages[i as usize].v.iter().enumerate() {
                if v > 0.0 {
                    let j = j as i32;
                    f(px * 8 + (j & 7), py * 8 + ((j >> 3) & 7), pz * 8 + (j >> 6), v);
                }
            }
        }
    }

    pub fn memory_bytes(&self) -> usize {
        self.pages.capacity() * std::mem::size_of::<Page>() + self.index.capacity() * 16
    }
}

/// Visited marks for flood fills on horizontal layers: 64x64 bit tiles per (tile, y).
#[derive(Default)]
struct LayerMarks {
    tiles: VoxMap<Box<[u64; 64]>>,
}

impl LayerMarks {
    #[inline]
    fn tile_key(x: i32, y: i32, z: i32) -> u64 {
        field(x >> 6) | field(y) << 21 | field(z >> 6) << 42
    }
    /// true if it was already set.
    fn test_and_set(&mut self, x: i32, y: i32, z: i32) -> bool {
        let t = self.tiles.entry(Self::tile_key(x, y, z)).or_insert_with(|| Box::new([0; 64]));
        let (w, b) = ((z & 63) as usize, 1u64 << (x & 63));
        let was = t[w] & b != 0;
        t[w] |= b;
        was
    }
    fn clear(&mut self) {
        self.tiles.clear();
    }
    fn memory_bytes(&self) -> usize {
        self.tiles.len() * (512 + 48)
    }
}

/// Endless worlds: absolute x / z; a column that isn't loaded is solid (water waits at the
/// edge of a chunk that streamed out).
#[inline]
fn outside(v: &SparseVolume, x: i32, y: i32, z: i32) -> bool {
    if y < 0 || y >= v.ny {
        return true;
    }
    if v.ring_w != 0 { !v.loaded_at(x, z) } else { x < 0 || z < 0 || x >= v.nx || z >= v.nz }
}

#[inline]
fn solid(v: &SparseVolume, x: i32, y: i32, z: i32) -> bool {
    if outside(v, x, y, z) {
        return true;
    }
    let m = v.get_voxel(x, y, z);
    m != 0 && m != WATER
}

const SIDES: [[i32; 2]; 4] = [[1, 0], [-1, 0], [0, 1], [0, -1]];

#[derive(Default)]
pub struct FluidSim {
    partial: PartialField,
    /// Membership of the active cells; `queue` holds the same cells in activation order.
    active: VoxSet,
    queue: VecDeque<u64>,
    /// Surface cells to level (a set: `level_seed_list` pops them).
    level_seeds: VoxSet,
    level_seed_list: Vec<u64>,
    level_visited: LayerMarks,
    level_cells: Vec<u64>,
    press_queue: Vec<u64>,
    press_visited: VoxSet,
    acc: f32,
    last_cells: usize,
    last_level_cells: usize,
    level_round: bool,
    tick_no: u32,
}

impl FluidSim {
    pub fn volume(&self, v: &SparseVolume, x: i32, y: i32, z: i32) -> f32 {
        if outside(v, x, y, z) {
            return 0.0;
        }
        let p = self.partial.get(x, y, z);
        if p > 0.0 {
            return p;
        }
        if v.get_voxel(x, y, z) == WATER { 1.0 } else { 0.0 }
    }

    pub fn active_count(&self) -> usize {
        self.active.len()
    }
    pub fn idle(&self) -> bool {
        self.active.is_empty() && self.level_seeds.is_empty()
    }
    pub fn partial_count(&self) -> usize {
        self.partial.len()
    }
    pub fn last_tick_cells(&self) -> usize {
        self.last_cells
    }
    pub fn partials(&self) -> &PartialField {
        &self.partial
    }
    /// Restore a saved partial volume (the voxel must already match it).
    pub fn set_partial(&mut self, x: i32, y: i32, z: i32, v: f32) {
        if v > 0.0 && v < 1.0 {
            self.partial.set(x, y, z, v);
        }
    }
    /// Heap held by the sim's tables, for the memory stats.
    pub fn memory_bytes(&self) -> usize {
        self.partial.memory_bytes()
            + self.level_visited.memory_bytes()
            + (self.active.capacity() + self.level_seeds.capacity()) * 9
            + self.queue.capacity() * 8
            + self.level_cells.capacity() * 8
    }

    fn activate(&mut self, x: i32, y: i32, z: i32) {
        if y < 0 {
            return; // (x / z may be negative: endless worlds; outside a fixed world = solid)
        }
        let k = key(x, y, z);
        if self.active.insert(k) {
            self.queue.push_back(k);
        }
    }

    fn wake_around(&mut self, x: i32, y: i32, z: i32) {
        self.activate(x, y, z);
        self.activate(x + 1, y, z);
        self.activate(x - 1, y, z);
        self.activate(x, y + 1, z);
        self.activate(x, y - 1, z);
        self.activate(x, y, z + 1);
        self.activate(x, y, z - 1);
    }

    fn seed_level(&mut self, k: u64) {
        if !self.level_round && self.level_seeds.len() < MAX_LEVEL_SEEDS && self.level_seeds.insert(k) {
            self.level_seed_list.push(k);
        }
    }

    pub fn reset(&mut self) {
        *self = Self::default();
    }

    fn supported(&self, v: &SparseVolume, x: i32, y: i32, z: i32) -> bool {
        solid(v, x, y - 1, z) || self.volume(v, x, y - 1, z) >= 1.0 - 1e-4
    }

    /// Set a cell's volume; updates the voxel (with hysteresis). True if the voxel changed.
    fn set_volume(&mut self, v: &mut SparseVolume, x: i32, y: i32, z: i32, vol: f32) -> bool {
        let is_water = v.get_voxel(x, y, z) == WATER;
        if vol <= 1e-6 {
            self.partial.erase(x, y, z);
            if is_water {
                v.set_voxel(x, y, z, 0);
                return true;
            }
            return false;
        }
        if vol >= 1.0 - 1e-4 {
            self.partial.erase(x, y, z);
            if !is_water {
                v.set_voxel(x, y, z, WATER);
                return true;
            }
            return false;
        }
        self.partial.set(x, y, z, vol);
        if is_water && vol < HIDE_BELOW {
            v.set_voxel(x, y, z, 0);
            return true;
        }
        if !is_water && vol >= SHOW_AT {
            v.set_voxel(x, y, z, WATER);
            return true;
        }
        false
    }

    /// Activate every cell in the box (inclusive): call after an edit near water.
    pub fn wake(&mut self, x0: i32, y0: i32, z0: i32, x1: i32, y1: i32, z1: i32) {
        for z in z0..=z1 {
            for y in y0..=y1 {
                for x in x0..=x1 {
                    self.activate(x, y, z);
                }
            }
        }
    }

    /// A solid voxel is about to be written into (x, y, z): move the water it holds (if any)
    /// up its column, else to the sides, and wake it. Call BEFORE the write.
    pub fn displace(&mut self, v: &mut SparseVolume, x: i32, y: i32, z: i32) {
        let mut amount = self.volume(v, x, y, z);
        self.partial.erase(x, y, z);
        self.wake(x - 1, y - 1, z - 1, x + 1, y + 1, z + 1);
        if amount <= 0.0 {
            return;
        }
        // Up the column (water rises around a placed block); what can't go up goes sideways.
        let mut yy = y + 1;
        while yy < v.ny && amount > 1e-6 {
            if solid(v, x, yy, z) {
                break;
            }
            let cur = self.volume(v, x, yy, z);
            let take = amount.min(1.0 - cur);
            if take > 0.0 {
                self.set_volume(v, x, yy, z, cur + take);
                amount -= take;
                self.activate(x, yy, z);
            }
            yy += 1;
        }
        for [dx, dz] in SIDES {
            if amount <= 1e-6 {
                break;
            }
            let (nx, nz) = (x + dx, z + dz);
            if solid(v, nx, y, nz) {
                continue;
            }
            let cur = self.volume(v, nx, y, nz);
            let take = amount.min(1.0 - cur);
            if take > 0.0 {
                self.set_volume(v, nx, y, nz, cur + take);
                amount -= take;
                self.activate(nx, y, nz);
            }
        }
    }

    /// Flood-fill the surface layer at (x, y, z) (supported water with none above, 4-connected)
    /// and set it to its average volume: the body drains as a whole.
    fn level_layer(&mut self, v: &mut SparseVolume, sx: i32, sy: i32, sz: i32) -> bool {
        let member = |s: &Self, v: &SparseVolume, x: i32, y: i32, z: i32| {
            !solid(v, x, y, z) && s.volume(v, x, y, z) > 0.0 && s.supported(v, x, y, z) && s.volume(v, x, y + 1, z) <= 0.0
        };
        if self.level_visited.test_and_set(sx, sy, sz) || !member(self, v, sx, sy, sz) {
            return false;
        }
        let mut cells = std::mem::take(&mut self.level_cells);
        cells.clear();
        cells.push(key(sx, sy, sz));
        let (mut sum, mut lo, mut hi) = (0.0f64, 2.0f32, -1.0f32);
        let mut i = 0;
        while i < cells.len() {
            let (x, y, z) = unkey(cells[i]);
            i += 1;
            let vol = self.volume(v, x, y, z);
            sum += vol as f64;
            lo = lo.min(vol);
            hi = hi.max(vol);
            if cells.len() >= MAX_LEVEL_CELLS {
                continue; // cap: level what we have
            }
            for [dx, dz] in SIDES {
                let (nx, nz) = (x + dx, z + dz);
                if v.ring_w == 0 && (nx < 0 || nz < 0) {
                    continue;
                }
                if self.level_visited.test_and_set(nx, y, nz) || !member(self, v, nx, y, nz) {
                    continue; // (each tested once)
                }
                cells.push(key(nx, y, nz));
            }
        }
        self.last_level_cells += cells.len();
        let mut changed = false;
        if cells.len() >= 2 && hi - lo >= SLEEP {
            let avg = (sum / cells.len() as f64) as f32;
            if avg >= MIN_SPREAD {
                for &k in &cells {
                    let (x, y, z) = unkey(k);
                    if (self.volume(v, x, y, z) - avg).abs() > 1e-6 {
                        changed |= self.set_volume(v, x, y, z, avg);
                    }
                }
            } else {
                // Drained to a film: empty the layer into the not-full water under it (the
                // drain column), so no volume is lost; what finds no room stays spread.
                let mut left = sum;
                for &k in &cells {
                    let (x, y, z) = unkey(k);
                    if y == 0 || solid(v, x, y - 1, z) {
                        continue;
                    }
                    let below = self.volume(v, x, y - 1, z);
                    let take = left.min((1.0 - below) as f64) as f32;
                    if take <= 0.0 {
                        continue;
                    }
                    changed |= self.set_volume(v, x, y - 1, z, below + take);
                    left -= take as f64;
                    self.activate(x, y - 1, z);
                    if left <= 1e-6 {
                        break;
                    }
                }
                let rest = (left / cells.len() as f64) as f32;
                for &k in &cells {
                    let (x, y, z) = unkey(k);
                    changed |= self.set_volume(v, x, y, z, if rest < EVAPORATE { 0.0 } else { rest });
                }
            }
        }
        self.level_cells = cells;
        changed
    }

    /// One levelling round every LEVEL_TICKS, spread over ticks by a cell budget.
    fn level_surfaces(&mut self, v: &mut SparseVolume) -> bool {
        self.last_level_cells = 0;
        let mut changed = false;
        while self.last_level_cells < LEVEL_BUDGET {
            let Some(k) = self.level_seed_list.pop() else { break };
            self.level_seeds.remove(&k);
            let (x, mut y, z) = unkey(k);
            // The seed may sit in the column under the surface: walk up to the top water cell.
            while y + 1 < v.ny && !solid(v, x, y + 1, z) && self.volume(v, x, y + 1, z) > 0.0 {
                y += 1;
            }
            changed |= self.level_layer(v, x, y, z);
        }
        self.level_round = !self.level_seed_list.is_empty();
        if !self.level_round {
            self.level_visited.clear();
        }
        changed
    }

    /// Breadth-first through the water around (x, y, z), only up and sideways, to the nearest
    /// surface cell above it; take up to `need` from there. Starts only from full neighbours.
    fn pressure_pull(&mut self, v: &mut SparseVolume, x: i32, y: i32, z: i32, need: f32) -> f32 {
        const D: [[i32; 3]; 5] = [[0, 1, 0], [1, 0, 0], [-1, 0, 0], [0, 0, 1], [0, 0, -1]];
        let full = |s: &Self, v: &SparseVolume, a: i32, b: i32, c: i32| !solid(v, a, b, c) && s.volume(v, a, b, c) >= 1.0 - 1e-4;
        let mut queue = std::mem::take(&mut self.press_queue);
        queue.clear();
        self.press_visited.clear();
        self.press_visited.insert(key(x, y, z));
        for [dx, dy, dz] in D {
            let (a, b, c) = (x + dx, y + dy, z + dz);
            if a < 0 || c < 0 || !full(self, v, a, b, c) {
                continue;
            }
            queue.push(key(a, b, c));
            self.press_visited.insert(key(a, b, c));
        }
        let mut side_at_start = 0; // sideways steps at the cell's own level: a few, to reach a column up
        let mut got = 0.0;
        let mut i = 0;
        while i < queue.len() && i < PRESSURE_SEARCH {
            let (a, b, c) = unkey(queue[i]);
            i += 1;
            let here = self.volume(v, a, b, c);
            let top = b + 1 >= v.ny || solid(v, a, b + 1, c) || self.volume(v, a, b + 1, c) <= 0.0;
            if top && b > y && here > 0.0 {
                let take = need.min(here);
                self.set_volume(v, a, b, c, here - take);
                self.activate(a, b, c);
                self.seed_level(key(a, b, c));
                got = take;
                break;
            }
            for [dx, dy, dz] in D {
                let (na, nb, nc) = (a + dx, b + dy, c + dz);
                if na < 0 || nc < 0 || nb >= v.ny || solid(v, na, nb, nc) || self.volume(v, na, nb, nc) <= 0.0 {
                    continue;
                }
                if nb == y && dy == 0 {
                    side_at_start += 1;
                    if side_at_start > 24 {
                        continue;
                    }
                }
                if !self.press_visited.insert(key(na, nb, nc)) {
                    continue;
                }
                queue.push(key(na, nb, nc));
            }
        }
        self.press_queue = queue;
        got
    }

    /// One tick regardless of time. True if any voxel changed.
    pub fn tick(&mut self, v: &mut SparseVolume) -> bool {
        self.tick_no = self.tick_no.wrapping_add(1);
        // Oldest first, capped: cells woken this tick wait behind the backlog, so a huge flood
        // spreads over several ticks without starving any part of it.
        let n = self.queue.len().min(MAX_CELLS_PER_TICK);
        let mut cells: Vec<u64> = self.queue.drain(..n).collect();
        for k in &cells {
            self.active.remove(k);
        }
        // Bottom up (a falling column settles in one pass); the horizontal sweep direction
        // alternates per tick so spreading isn't biased.
        let flip = self.tick_no & 1 != 0;
        cells.sort_unstable_by_key(|&k| {
            let (x, y, z) = unkey(k);
            let (x, z) = if flip { (-x, -z) } else { (x, z) };
            (y, z, x)
        });
        self.last_cells = cells.len();
        let mut changed = false;
        let mut pulls = 0;
        for &k in &cells {
            let (x, y, z) = unkey(k);
            if solid(v, x, y, z) {
                self.partial.erase(x, y, z);
                continue;
            }
            let mut vol = self.volume(v, x, y, z);
            let v0 = vol;
            // 0. Pressure: a cell inside a body that isn't full is refilled from the body's
            //    surface above it, not by slow sideways seepage.
            if vol < 1.0 - 1e-4 && pulls < MAX_PULLS_PER_TICK {
                let got = self.pressure_pull(v, x, y, z, 1.0 - vol);
                pulls += 1;
                if got > 0.0 {
                    vol += got; // (written back below)
                    self.wake_around(x, y, z);
                }
            }
            if vol <= 0.0 {
                continue;
            }
            // 1. Fall into the free capacity below.
            if !solid(v, x, y - 1, z) {
                let below = self.volume(v, x, y - 1, z);
                let f = vol.min(1.0 - below);
                if f > 1e-6 {
                    changed |= self.set_volume(v, x, y - 1, z, below + f);
                    vol -= f;
                    self.wake_around(x, y - 1, z);
                }
            }
            // 2. Supported: share with lower horizontal neighbours. A thin sheet stops spreading
            //    (a puddle keeps its depth) instead of thinning into a film that evaporates.
            if vol > MIN_SPREAD && self.supported(v, x, y, z) {
                let mut nv = [0.0f32; 4];
                let mut open = [false; 4];
                for (i, [dx, dz]) in SIDES.into_iter().enumerate() {
                    open[i] = !solid(v, x + dx, y, z + dz);
                    nv[i] = if open[i] { self.volume(v, x + dx, y, z + dz) } else { 0.0 };
                }
                let here = vol;
                for (i, [dx, dz]) in SIDES.into_iter().enumerate() {
                    if !open[i] || nv[i] >= here {
                        continue;
                    }
                    let f = (here - nv[i]) * 0.2;
                    if f < MIN_FLOW || here - f < MIN_SPREAD * 0.5 {
                        continue;
                    }
                    changed |= self.set_volume(v, x + dx, y, z + dz, nv[i] + f);
                    vol -= f;
                    if f > SLEEP {
                        self.wake_around(x + dx, y, z + dz); // tiny seepage lets it sleep
                    }
                }
            }
            // 3. Crumbs vanish.
            if vol < EVAPORATE {
                vol = 0.0;
            }
            if (vol - v0).abs() > 1e-7 {
                changed |= self.set_volume(v, x, y, z, vol);
            }
            if (vol - v0).abs() > SLEEP {
                self.wake_around(x, y, z);
                self.seed_level(k); // its surface gets levelled
            }
        }
        if self.level_round || self.tick_no.is_multiple_of(LEVEL_TICKS) {
            changed |= self.level_surfaces(v);
            self.shrink_tables();
        }
        changed
    }

    /// Advance by real time dt (fixed ticks, at most 3 per call and no debt beyond that: the
    /// sim runs slower than real time rather than making slow frames slower). True if any
    /// voxel changed.
    pub fn update(&mut self, v: &mut SparseVolume, dt: f32) -> bool {
        if self.idle() {
            self.acc = 0.0;
            return false;
        }
        self.acc = (self.acc + dt.min(0.25)).min(3.0 / TICK_HZ);
        let mut changed = false;
        let mut n = 0;
        while self.acc >= 1.0 / TICK_HZ && n < 3 {
            self.acc -= 1.0 / TICK_HZ;
            changed |= self.tick(v);
            n += 1;
        }
        changed
    }

    /// Tables keep their peak capacity: after a big flood rebuild them at their current size.
    fn shrink_tables(&mut self) {
        if self.active.capacity() > 65536 && self.active.capacity() > 8 * self.active.len() {
            self.active.shrink_to_fit();
        }
        if self.level_seeds.capacity() > 65536 && self.level_seeds.capacity() > 8 * self.level_seeds.len() {
            self.level_seeds.shrink_to_fit();
        }
        if self.queue.is_empty() {
            self.queue.shrink_to_fit();
        }
        if self.level_cells.capacity() > 65536 {
            self.level_cells = Vec::new();
        }
    }

    /// Floating origin: world coordinates moved by (dx, dz) voxels - every table re-keyed (an
    /// unfinished levelling round starts over).
    pub fn shift(&mut self, dx: i32, dz: i32) {
        let mut cells = Vec::with_capacity(self.partial.len());
        self.partial.for_each(|x, y, z, v| cells.push((x - dx, y, z - dz, v)));
        self.partial.clear();
        for (x, y, z, v) in cells {
            self.partial.set(x, y, z, v);
        }
        let mv = |k: u64| {
            let (x, y, z) = unkey(k);
            key(x - dx, y, z - dz)
        };
        self.active = self.active.iter().map(|&k| mv(k)).collect();
        for k in &mut self.level_seed_list {
            *k = mv(*k);
        }
        self.level_seeds = self.level_seed_list.iter().copied().collect();
        for k in &mut self.queue {
            *k = mv(*k);
        }
        self.level_visited.clear();
        self.level_cells.clear();
        self.press_queue.clear();
        self.press_visited.clear();
        self.level_round = false;
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    const STONE: u8 = 3;

    /// A 64x32x64 stone basin: floor at y 0..4, walls 1 thick.
    fn basin() -> SparseVolume {
        let mut v = SparseVolume::new(64, 32, 64);
        v.fill_box(0, 0, 0, 63, 3, 63, STONE);
        v.fill_box(0, 4, 0, 63, 31, 0, STONE);
        v.fill_box(0, 4, 63, 63, 31, 63, STONE);
        v.fill_box(0, 4, 0, 0, 31, 63, STONE);
        v.fill_box(63, 4, 0, 63, 31, 63, STONE);
        v
    }

    fn total(s: &FluidSim, v: &SparseVolume) -> f64 {
        let mut t = 0.0;
        for z in 0..v.nz {
            for y in 0..v.ny {
                for x in 0..v.nx {
                    t += s.volume(v, x, y, z) as f64;
                }
            }
        }
        t
    }

    #[test]
    fn column_falls_spreads_and_conserves_volume() {
        let mut v = basin();
        v.fill_box(30, 10, 30, 33, 25, 33, WATER); // 4x16x4 = 256 cells in the air
        let mut s = FluidSim::default();
        s.wake(29, 9, 29, 34, 26, 34);
        let before = total(&s, &v);
        for _ in 0..2000 {
            s.tick(&mut v);
            if s.idle() {
                break;
            }
        }
        let after = total(&s, &v);
        assert!((before - after).abs() < 0.05 * before, "volume {before} -> {after}");
        // It all came down onto the floor and spread out.
        let high: f64 = (6..32).flat_map(|y| (0..64).flat_map(move |z| (0..64).map(move |x| (x, y, z)))).map(|(x, y, z)| s.volume(&v, x, y, z) as f64).sum();
        assert!(high < 1.0, "water left hanging: {high}");
        assert!(s.volume(&v, 10, 4, 10) > 0.0 || s.volume(&v, 50, 4, 50) > 0.0, "it spread");
    }

    #[test]
    fn displace_moves_water_up() {
        let mut v = basin();
        v.fill_box(1, 4, 1, 62, 5, 62, WATER); // a 2-deep pool
        let mut s = FluidSim::default();
        s.displace(&mut v, 20, 4, 20);
        v.set_voxel(20, 4, 20, STONE);
        assert_eq!(v.get_voxel(20, 6, 20), WATER, "pushed up the column");
        assert!(!s.idle());
    }

    #[test]
    fn partial_field_pages() {
        let mut p = PartialField::default();
        p.set(-3, 5, 700, 0.25);
        p.set(-4, 5, 700, 0.5);
        assert_eq!(p.get(-3, 5, 700), 0.25);
        assert_eq!(p.len(), 2);
        p.erase(-3, 5, 700);
        p.erase(-4, 5, 700);
        assert_eq!((p.len(), p.get(-4, 5, 700)), (0, 0.0));
        p.set(1, 1, 1, 0.75);
        let mut seen = Vec::new();
        p.for_each(|x, y, z, v| seen.push((x, y, z, v)));
        assert_eq!(seen, vec![(1, 1, 1, 0.75)]);
    }

    #[test]
    fn hole_under_a_lake_drains_it_level() {
        let mut v = basin();
        // A lake 4 deep over a floor with a 1x1 drain into a cavern below it.
        v.fill_box(1, 4, 1, 62, 7, 62, WATER);
        v.fill_box(1, 1, 1, 20, 2, 20, 0); // the cavern (inside the floor)
        let mut s = FluidSim::default();
        v.set_voxel(10, 3, 10, 0); // the drain
        s.wake(9, 2, 9, 11, 5, 11);
        for _ in 0..400 {
            s.tick(&mut v);
        }
        assert!(s.volume(&v, 5, 1, 5) > 0.5, "the cavern fills");
        // The surface stays level: no deep crater around the drain.
        let near = s.volume(&v, 12, 7, 10);
        let far = s.volume(&v, 50, 7, 50);
        assert!((near - far).abs() < 0.2, "surface near {near} far {far}");
    }
}
