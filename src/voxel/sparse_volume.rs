//! Sparse brick volume: the storage unit for the static world and every dynamic body.
//!
//! A grid of 8^3-voxel bricks, one `u32` per brick cell:
//!   * `0`        -> empty brick
//!   * `1..=255`  -> uniform brick, entirely material `v`
//!   * `>= 256`   -> detailed brick: index `v - 256` into `bricks`
//!
//! Uniform interiors cost nothing, so memory tracks surface area, not volume.
//!
//! RING mode (streaming worlds, [`SparseVolume::ring_init`]): x / z are unbounded voxel
//! coordinates held in a toroidal window of W x W chunk slots (a chunk = 16 x 16 brick
//! columns, full height). Voxels of chunks that aren't resident read as air and writes to
//! them are dropped. The grid layout is exactly the GPU's toroidal grid texture.

use crate::math::Vec3;

pub const BRICK: i32 = 8;
/// Bricks per streamed chunk side (= worldgen::CHUNK / 8).
pub const RING_CHUNK_B: i32 = 16;

/// One detailed 8^3 brick. `vox` index = lz*64 + ly*8 + lx.
#[derive(Clone)]
pub struct Brick {
    pub vox: [u8; 512],
    /// Atlas slot on the GPU (-1 = not uploaded).
    pub gpu_slot: i32,
    /// Occupancy bits per 4^3 octant (the GPU's brick masks), cached when valid.
    pub mask_ok: bool,
    pub mask: [u32; 16],
}

impl Brick {
    pub fn filled(m: u8) -> Self {
        Self { vox: [m; 512], gpu_slot: -1, mask_ok: false, mask: [0; 16] }
    }

    /// The single material of a brick whose voxels are all equal.
    #[inline]
    pub fn uniform(&self) -> Option<u8> {
        let first = self.vox[0];
        // (chunks of 64 bytes compare as wide vectors)
        self.vox.iter().all(|&v| v == first).then_some(first)
    }
}

/// Occupancy of each 4^3 octant of a brick: octant o = ox | oy<<1 | oz<<2,
/// bit = (x&3) | (y&3)<<2 | (z&3)<<4 (must match traceVolume in common.glsl).
pub fn brick_mask(vox: &[u8; 512]) -> [u32; 16] {
    let mut m = [0u32; 16];
    for z in 0..8 {
        for y in 0..8 {
            let row = &vox[z * 64 + y * 8..z * 64 + y * 8 + 8];
            for (x, &v) in row.iter().enumerate() {
                if v == 0 {
                    continue;
                }
                let oct = (x >> 2) | ((y >> 2) << 1) | ((z >> 2) << 2);
                let bit = (x & 3) | ((y & 3) << 2) | ((z & 3) << 4);
                m[oct * 2 + (bit >> 5)] |= 1 << (bit & 31);
            }
        }
    }
    m
}

pub fn compute_brick_mask(b: &mut Brick) {
    b.mask = brick_mask(&b.vox);
    b.mask_ok = true;
}

#[derive(Clone, Copy, Debug, Default)]
pub struct RayHit {
    pub hit: bool,
    pub t: f32,
    pub cell: [i32; 3],
    pub normal: [i32; 3],
    pub mat: u8,
}

#[derive(Clone, Copy, Debug, Default, PartialEq, Eq)]
pub struct RingSlot {
    /// The absolute chunk held.
    pub cx: i32,
    pub cz: i32,
    pub loaded: bool,
    /// Edited since loaded / saved (must be written before eviction).
    pub modified: bool,
}

#[derive(Clone)]
pub struct SparseVolume {
    /// Voxel dims (multiples of 8); ring: the window span.
    pub nx: i32,
    pub ny: i32,
    pub nz: i32,
    pub bricks_x: i32,
    pub bricks_y: i32,
    pub bricks_z: i32,
    pub grid: Vec<u32>,
    pub bricks: Vec<Brick>,
    /// Grid indices to re-sync to the GPU (may contain duplicates).
    pub dirty_grid: Vec<u32>,
    /// Off for scratch volumes (a chunk being generated).
    pub track_dirty: bool,
    /// Slots per side (0 = a plain bounded volume).
    pub ring_w: i32,
    pub ring_slots: Vec<RingSlot>,
    /// Atlas slots of evicted bricks (the GPU mirror frees them).
    pub released_gpu_slots: Vec<i32>,
    /// Ring: absolute = these coordinates + origin (voxels).
    pub origin_x: i32,
    pub origin_z: i32,
    free_bricks: Vec<u32>,
}

impl SparseVolume {
    pub fn new(nx: i32, ny: i32, nz: i32) -> Self {
        let bx = (nx + BRICK - 1) / BRICK;
        let by = (ny + BRICK - 1) / BRICK;
        let bz = (nz + BRICK - 1) / BRICK;
        Self {
            nx,
            ny,
            nz,
            bricks_x: bx,
            bricks_y: by,
            bricks_z: bz,
            grid: vec![0; (bx as usize) * (by as usize) * (bz as usize)],
            bricks: Vec::new(),
            dirty_grid: Vec::new(),
            track_dirty: true,
            ring_w: 0,
            ring_slots: Vec::new(),
            released_gpu_slots: Vec::new(),
            origin_x: 0,
            origin_z: 0,
            free_bricks: Vec::new(),
        }
    }

    #[inline]
    pub fn grid_index(&self, mut ix: i32, iy: i32, mut iz: i32) -> usize {
        if self.ring_w != 0 {
            ix &= self.bricks_x - 1;
            iz &= self.bricks_z - 1;
        }
        ((iz as usize) * (self.bricks_y as usize) + iy as usize) * (self.bricks_x as usize) + ix as usize
    }

    // --- ring mode -------------------------------------------------------------------------
    #[inline]
    fn ring_slot_index(&self, cx: i32, cz: i32) -> usize {
        let m = self.ring_w - 1;
        ((cz & m) as usize) * (self.ring_w as usize) + (cx & m) as usize
    }
    pub fn ring_slot(&self, cx: i32, cz: i32) -> &RingSlot {
        &self.ring_slots[self.ring_slot_index(cx, cz)]
    }
    pub fn ring_slot_mut(&mut self, cx: i32, cz: i32) -> &mut RingSlot {
        let i = self.ring_slot_index(cx, cz);
        &mut self.ring_slots[i]
    }
    #[inline]
    pub fn chunk_loaded(&self, cx: i32, cz: i32) -> bool {
        if self.ring_w == 0 {
            return true;
        }
        let s = self.ring_slot(cx, cz);
        s.loaded && s.cx == cx && s.cz == cz
    }
    /// Is the voxel column (x, z) resident? (Always true for a bounded volume.)
    #[inline]
    pub fn loaded_at(&self, x: i32, z: i32) -> bool {
        self.ring_w == 0 || self.chunk_loaded(x >> 7, z >> 7)
    }

    pub fn ring_init(&mut self, w: i32, ny: i32) {
        assert!(w > 0 && (w & (w - 1)) == 0, "ring width must be a power of two");
        self.ring_w = w;
        self.nx = w * RING_CHUNK_B * BRICK;
        self.nz = self.nx;
        self.ny = ny;
        self.bricks_x = w * RING_CHUNK_B;
        self.bricks_z = self.bricks_x;
        self.bricks_y = (ny + BRICK - 1) / BRICK;
        self.grid = vec![0; (self.bricks_x as usize) * (self.bricks_y as usize) * (self.bricks_z as usize)];
        self.bricks = Vec::new();
        self.free_bricks = Vec::new();
        self.dirty_grid = Vec::new();
        self.ring_slots = vec![RingSlot::default(); (w * w) as usize];
        self.released_gpu_slots.clear();
    }

    /// Floating origin: the coordinates move by (dcx, dcz) chunks. A multiple of ring_w keeps
    /// every chunk in its slot: only the owners change.
    pub fn ring_rebase(&mut self, dcx: i32, dcz: i32) {
        for s in &mut self.ring_slots {
            s.cx -= dcx;
            s.cz -= dcz;
        }
        self.origin_x += dcx * RING_CHUNK_B * 8;
        self.origin_z += dcz * RING_CHUNK_B * 8;
    }

    /// Absolute brick coordinates of grid cell gi (ring: from its slot's owner).
    pub fn brick_coords(&self, gi: usize) -> (i32, i32, i32) {
        let bx = (gi % self.bricks_x as usize) as i32;
        let by = ((gi / self.bricks_x as usize) % self.bricks_y as usize) as i32;
        let bz = (gi / (self.bricks_x as usize * self.bricks_y as usize)) as i32;
        if self.ring_w == 0 {
            return (bx, by, bz);
        }
        let s = &self.ring_slots[(bz / RING_CHUNK_B) as usize * self.ring_w as usize + (bx / RING_CHUNK_B) as usize];
        (s.cx * RING_CHUNK_B + bx % RING_CHUNK_B, by, s.cz * RING_CHUNK_B + bz % RING_CHUNK_B)
    }

    #[inline]
    fn mark_modified(&mut self, gi: usize) {
        if self.ring_w == 0 {
            return;
        }
        let bxy = self.bricks_x as usize * self.bricks_y as usize;
        let i = ((gi / bxy) / RING_CHUNK_B as usize) * self.ring_w as usize + (gi % self.bricks_x as usize) / RING_CHUNK_B as usize;
        self.ring_slots[i].modified = true;
    }

    // --- voxel access ------------------------------------------------------------------------
    #[inline]
    fn inside(&self, x: i32, y: i32, z: i32) -> bool {
        if self.ring_w != 0 {
            (y as u32) < self.ny as u32 && self.chunk_loaded(x >> 7, z >> 7)
        } else {
            (x as u32) < self.nx as u32 && (y as u32) < self.ny as u32 && (z as u32) < self.nz as u32
        }
    }

    #[inline]
    pub fn get_voxel(&self, x: i32, y: i32, z: i32) -> u8 {
        if !self.inside(x, y, z) {
            return 0;
        }
        let v = self.grid[self.grid_index(x >> 3, y >> 3, z >> 3)];
        if v < 256 {
            return v as u8;
        }
        self.bricks[(v - 256) as usize].vox[((z & 7) * 64 + (y & 7) * 8 + (x & 7)) as usize]
    }

    /// A detailed brick for cell gi filled with `fill` (reusing a freed one).
    fn materialize(&mut self, gi: usize, fill: u8) -> usize {
        let idx = match self.free_bricks.pop() {
            Some(i) => {
                let b = &mut self.bricks[i as usize];
                b.vox = [fill; 512];
                b.gpu_slot = -1;
                b.mask_ok = false;
                i as usize
            }
            None => {
                self.bricks.push(Brick::filled(fill));
                self.bricks.len() - 1
            }
        };
        self.grid[gi] = 256 + idx as u32;
        idx
    }

    pub fn set_voxel(&mut self, x: i32, y: i32, z: i32, m: u8) {
        if !self.inside(x, y, z) {
            return;
        }
        let gi = self.grid_index(x >> 3, y >> 3, z >> 3);
        let v = self.grid[gi];
        let bi = if v < 256 {
            if v as u8 == m {
                return;
            }
            self.materialize(gi, v as u8)
        } else {
            (v - 256) as usize
        };
        let b = &mut self.bricks[bi];
        let slot = &mut b.vox[((z & 7) * 64 + (y & 7) * 8 + (x & 7)) as usize];
        if *slot == m {
            return;
        }
        *slot = m;
        b.mask_ok = false;
        if self.track_dirty {
            self.dirty_grid.push(gi as u32);
        }
        self.mark_modified(gi);
    }

    pub fn fill_sphere(&mut self, cx: f32, cy: f32, cz: f32, r: f32, m: u8) {
        let (x0, x1) = ((cx - r).floor() as i32, (cx + r).ceil() as i32);
        let (y0, y1) = ((cy - r).floor() as i32, (cy + r).ceil() as i32);
        let (z0, z1) = ((cz - r).floor() as i32, (cz + r).ceil() as i32);
        let r2 = r * r;
        for z in z0..=z1 {
            let dz = z as f32 + 0.5 - cz;
            for y in y0..=y1 {
                let dy = y as f32 + 0.5 - cy;
                let rem = r2 - dz * dz - dy * dy;
                if rem < 0.0 {
                    continue;
                }
                for x in x0..=x1 {
                    let dx = x as f32 + 0.5 - cx;
                    if dx * dx <= rem {
                        self.set_voxel(x, y, z, m);
                    }
                }
            }
        }
    }

    pub fn fill_box(&mut self, x0: i32, y0: i32, z0: i32, x1: i32, y1: i32, z1: i32, m: u8) {
        for z in z0..=z1 {
            for y in y0..=y1 {
                for x in x0..=x1 {
                    self.set_voxel(x, y, z, m);
                }
            }
        }
    }

    /// Empty the volume (all cells air, no bricks).
    pub fn clear(&mut self) {
        self.ring_slots.iter_mut().for_each(|s| *s = RingSlot::default());
        self.grid.fill(0);
        self.bricks = Vec::new();
        self.free_bricks = Vec::new();
        self.dirty_grid = Vec::new();
    }

    /// Detailed bricks that became constant -> uniform/empty cells. Only safe before those
    /// bricks were GPU-synced (call right after authoring).
    pub fn compact(&mut self) {
        for gi in 0..self.grid.len() {
            let v = self.grid[gi];
            if v < 256 {
                continue;
            }
            if let Some(m) = self.bricks[(v - 256) as usize].uniform() {
                self.grid[gi] = m as u32;
                self.free_bricks.push(v - 256);
            }
        }
    }

    /// One detailed brick whose voxels are all equal -> a uniform cell again. Returns
    /// `Some(old atlas slot)` (-1 if none) when it compacted.
    pub fn compact_brick(&mut self, gi: usize) -> Option<i32> {
        let v = self.grid[gi];
        if v < 256 {
            return None;
        }
        let b = &mut self.bricks[(v - 256) as usize];
        let m = b.uniform()?;
        let slot = b.gpu_slot;
        b.gpu_slot = -1;
        self.free_bricks.push(v - 256);
        self.grid[gi] = m as u32;
        Some(slot)
    }

    /// Whole-brick write: a uniform cell. Keeps a detailed brick (its atlas slot is still
    /// referenced) - the GPU sync compacts it and releases the slot.
    pub fn set_brick_uniform(&mut self, gi: usize, m: u8) {
        let v = self.grid[gi];
        if v < 256 {
            if v == m as u32 {
                return;
            }
            self.grid[gi] = m as u32;
        } else {
            let b = &mut self.bricks[(v - 256) as usize];
            if b.vox.iter().all(|&x| x == m) {
                return;
            }
            b.vox = [m; 512];
            b.mask_ok = false;
        }
        self.dirty_grid.push(gi as u32);
        self.mark_modified(gi);
    }

    /// Whole-brick write: 512 voxels.
    pub fn set_brick_voxels(&mut self, gi: usize, vox: &[u8; 512]) {
        let v = self.grid[gi];
        let bi = if v < 256 {
            self.materialize(gi, v as u8)
        } else {
            let bi = (v - 256) as usize;
            if &self.bricks[bi].vox == vox {
                return;
            }
            bi
        };
        let b = &mut self.bricks[bi];
        b.vox = *vox;
        b.mask_ok = false;
        if self.track_dirty {
            self.dirty_grid.push(gi as u32);
        }
        self.mark_modified(gi);
    }

    /// Grow by whole bricks on each side. Content shifts by +8*neg; dirty_grid is cleared
    /// (the caller re-uploads this volume's grid).
    pub fn grow_bricks(&mut self, neg: [i32; 3], pos: [i32; 3]) {
        let (nbx, nby, nbz) = (self.bricks_x + neg[0] + pos[0], self.bricks_y + neg[1] + pos[1], self.bricks_z + neg[2] + pos[2]);
        let mut ng = vec![0u32; (nbx * nby * nbz) as usize];
        for bz in 0..self.bricks_z {
            for by in 0..self.bricks_y {
                for bx in 0..self.bricks_x {
                    ng[(((bz + neg[2]) * nby + by + neg[1]) * nbx + bx + neg[0]) as usize] = self.grid[self.grid_index(bx, by, bz)];
                }
            }
        }
        self.grid = ng;
        self.bricks_x = nbx;
        self.bricks_y = nby;
        self.bricks_z = nbz;
        self.nx = nbx * BRICK;
        self.ny = nby * BRICK;
        self.nz = nbz * BRICK;
        self.dirty_grid.clear();
    }

    pub fn detailed_bricks(&self) -> usize {
        self.bricks.len() - self.free_bricks.len()
    }

    /// Copy `src` (brick-aligned) in with its brick (0,0,0) at brick (bx0, by0, bz0). The
    /// target cells must be empty. Takes `src` by value: its bricks move, not copy.
    pub fn paste_chunk(&mut self, src: SparseVolume, bx0: i32, by0: i32, bz0: i32) {
        let base = self.bricks.len() as u32;
        let mut remap = vec![u32::MAX; src.bricks.len()];
        let mut moved = 0u32;
        for bz in 0..src.bricks_z {
            for by in 0..src.bricks_y {
                for bx in 0..src.bricks_x {
                    let (gx, gy, gz) = (bx0 + bx, by0 + by, bz0 + bz);
                    if gx < 0 || gy < 0 || gz < 0 || gx >= self.bricks_x || gy >= self.bricks_y || gz >= self.bricks_z {
                        continue;
                    }
                    let v = src.grid[src.grid_index(bx, by, bz)];
                    let gi = self.grid_index(gx, gy, gz);
                    if v < 256 {
                        self.grid[gi] = v;
                    } else {
                        let r = &mut remap[(v - 256) as usize];
                        if *r == u32::MAX {
                            *r = moved;
                            moved += 1;
                        }
                        self.grid[gi] = 256 + base + *r;
                    }
                }
            }
        }
        // Move the referenced bricks in remap order (a brick referenced twice stays shared).
        let mut order: Vec<(u32, usize)> = remap.iter().enumerate().filter(|(_, r)| **r != u32::MAX).map(|(i, r)| (*r, i)).collect();
        order.sort_unstable();
        let mut src_bricks: Vec<Option<Brick>> = src.bricks.into_iter().map(Some).collect();
        self.bricks.reserve(order.len());
        for (_, i) in order {
            let mut b = src_bricks[i].take().expect("brick moved twice");
            b.gpu_slot = -1;
            self.bricks.push(b);
        }
    }

    /// Put a generated / loaded chunk (a 128 x ny x 128 volume) into its slot, evicting
    /// whatever the slot held. Its bricks reuse freed ones.
    pub fn ring_paste(&mut self, src: &SparseVolume, cx: i32, cz: i32) {
        let s = *self.ring_slot(cx, cz);
        if s.loaded {
            self.ring_evict(s.cx, s.cz);
        }
        let (bx0, bz0) = (cx * RING_CHUNK_B, cz * RING_CHUNK_B);
        for bz in 0..RING_CHUNK_B.min(src.bricks_z) {
            for by in 0..self.bricks_y.min(src.bricks_y) {
                for bx in 0..RING_CHUNK_B.min(src.bricks_x) {
                    let v = src.grid[src.grid_index(bx, by, bz)];
                    let gi = self.grid_index(bx0 + bx, by, bz0 + bz);
                    if v < 256 {
                        self.grid[gi] = v;
                        continue;
                    }
                    let mut b = src.bricks[(v - 256) as usize].clone();
                    b.gpu_slot = -1;
                    let idx = match self.free_bricks.pop() {
                        Some(i) => {
                            self.bricks[i as usize] = b;
                            i
                        }
                        None => {
                            self.bricks.push(b);
                            (self.bricks.len() - 1) as u32
                        }
                    };
                    self.grid[gi] = 256 + idx;
                }
            }
        }
        *self.ring_slot_mut(cx, cz) = RingSlot { cx, cz, loaded: true, modified: false };
    }

    /// Empty a chunk's slot: its bricks are freed (their atlas slots -> released_gpu_slots).
    pub fn ring_evict(&mut self, cx: i32, cz: i32) {
        let s = *self.ring_slot(cx, cz);
        if !s.loaded {
            return;
        }
        let (bx0, bz0) = (s.cx * RING_CHUNK_B, s.cz * RING_CHUNK_B);
        for bz in 0..RING_CHUNK_B {
            for by in 0..self.bricks_y {
                for bx in 0..RING_CHUNK_B {
                    let gi = self.grid_index(bx0 + bx, by, bz0 + bz);
                    let v = self.grid[gi];
                    if v >= 256 {
                        let b = &mut self.bricks[(v - 256) as usize];
                        if b.gpu_slot >= 0 {
                            self.released_gpu_slots.push(b.gpu_slot);
                        }
                        b.gpu_slot = -1;
                        self.free_bricks.push(v - 256);
                    }
                    self.grid[gi] = 0;
                }
            }
        }
        *self.ring_slot_mut(cx, cz) = RingSlot::default();
    }

    /// Copy a resident chunk out (saving): a 128 x ny x 128 volume.
    pub fn ring_extract(&self, cx: i32, cz: i32) -> SparseVolume {
        let mut out = SparseVolume::new(RING_CHUNK_B * BRICK, self.ny, RING_CHUNK_B * BRICK);
        out.track_dirty = false;
        let (bx0, bz0) = (cx * RING_CHUNK_B, cz * RING_CHUNK_B);
        let ok = if self.ring_w != 0 {
            self.chunk_loaded(cx, cz)
        } else {
            bx0 >= 0 && bz0 >= 0 && bx0 + RING_CHUNK_B <= self.bricks_x && bz0 + RING_CHUNK_B <= self.bricks_z && self.ny == out.ny
        };
        if !ok {
            return out;
        }
        for bz in 0..RING_CHUNK_B {
            for by in 0..self.bricks_y {
                for bx in 0..RING_CHUNK_B {
                    let v = self.grid[self.grid_index(bx0 + bx, by, bz0 + bz)];
                    let go = out.grid_index(bx, by, bz);
                    if v < 256 {
                        out.grid[go] = v;
                    } else {
                        let mut b = self.bricks[(v - 256) as usize].clone();
                        b.gpu_slot = -1;
                        out.bricks.push(b);
                        out.grid[go] = 255 + out.bricks.len() as u32;
                    }
                }
            }
        }
        out
    }

    /// Voxel-exact raycast in local (voxel) coordinates. Voxels of `skip_mat` (non-zero) are
    /// passed through (the edit aim looks through water).
    pub fn raycast(&self, ro: Vec3, rd: Vec3, max_t: f32, skip_mat: u8) -> RayHit {
        let mut h = RayHit::default();
        let safe = |v: f32| if v.abs() < 1e-6 { 1e-6 } else { v };
        let inv = [1.0 / safe(rd.x), 1.0 / safe(rd.y), 1.0 / safe(rd.z)];
        let ro_a = [ro.x, ro.y, ro.z];
        let rd_a = [rd.x, rd.y, rd.z];
        let dims = [self.nx, self.ny, self.nz];
        let ring = self.ring_w != 0;
        let (mut t_enter, mut t_exit) = (0.0f32, max_t);
        for a in 0..3 {
            if ring && a != 1 {
                continue; // ring: unbounded in x / z
            }
            let (mut t0, mut t1) = ((0.0 - ro_a[a]) * inv[a], (dims[a] as f32 - ro_a[a]) * inv[a]);
            if t0 > t1 {
                std::mem::swap(&mut t0, &mut t1);
            }
            t_enter = t_enter.max(t0);
            t_exit = t_exit.min(t1);
        }
        if t_exit < t_enter {
            return h;
        }
        let mut t = t_enter + 1e-4;
        let mut cell = [0i32; 3];
        let mut step = [0i32; 3];
        let mut t_max = [0f32; 3];
        let mut t_delta = [0f32; 3];
        for a in 0..3 {
            cell[a] = (ro_a[a] + rd_a[a] * t).floor() as i32;
            if !ring || a == 1 {
                cell[a] = cell[a].clamp(0, dims[a] - 1);
            }
            step[a] = if rd_a[a] < 0.0 { -1 } else { 1 };
            let boundary = cell[a] + if step[a] > 0 { 1 } else { 0 };
            t_max[a] = (boundary as f32 - ro_a[a]) * inv[a];
            t_delta[a] = inv[a].abs();
        }
        let mut normal = [0i32; 3];
        let max_steps = if ring { (3.0 * max_t.min(1e6)) as i32 + 8 } else { 4 * (self.nx + self.ny + self.nz) };
        for _ in 0..max_steps {
            let m = self.get_voxel(cell[0], cell[1], cell[2]);
            if m != 0 && m != skip_mat {
                return RayHit { hit: true, t, cell, normal, mat: m };
            }
            let a = if t_max[0] < t_max[1] && t_max[0] < t_max[2] {
                0
            } else if t_max[1] < t_max[2] {
                1
            } else {
                2
            };
            cell[a] += step[a];
            t = t_max[a];
            t_max[a] += t_delta[a];
            normal = [0; 3];
            normal[a] = -step[a];
            if t > max_t || ((!ring || a == 1) && (cell[a] < 0 || cell[a] >= dims[a])) {
                return h;
            }
        }
        h.hit = false;
        h
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn set_get_compact() {
        let mut v = SparseVolume::new(32, 32, 32);
        assert_eq!(v.get_voxel(3, 4, 5), 0);
        v.set_voxel(3, 4, 5, 7);
        assert_eq!(v.get_voxel(3, 4, 5), 7);
        assert_eq!(v.detailed_bricks(), 1);
        v.set_voxel(3, 4, 5, 0);
        v.compact();
        assert_eq!(v.detailed_bricks(), 0);
        assert_eq!(v.grid[0], 0);
        v.fill_box(8, 8, 8, 15, 15, 15, 3);
        v.compact();
        assert_eq!(v.grid[v.grid_index(1, 1, 1)], 3);
        assert_eq!(v.get_voxel(-1, 0, 0), 0);
    }

    #[test]
    fn mask_bits() {
        let mut vox = [0u8; 512];
        vox[7 * 64 + 7 * 8 + 7] = 1; // (7,7,7): octant 7, bit 63
        let m = brick_mask(&vox);
        assert_eq!(m[15], 1 << 31);
        assert!(m[..15].iter().all(|&x| x == 0));
    }

    #[test]
    fn raycast_hits_floor() {
        let mut v = SparseVolume::new(64, 64, 64);
        v.fill_box(0, 0, 0, 63, 9, 63, 3);
        let h = v.raycast(Vec3::new(20.5, 40.0, 20.5), Vec3::new(0.0, -1.0, 0.0), 100.0, 0);
        assert!(h.hit);
        assert_eq!(h.cell, [20, 9, 20]);
        assert_eq!(h.normal, [0, 1, 0]);
        assert!((h.t - 30.0).abs() < 1e-3);
    }

    #[test]
    fn paste_and_ring() {
        let mut chunk = SparseVolume::new(128, 64, 128);
        chunk.fill_sphere(64.0, 32.0, 64.0, 10.0, 4);
        chunk.compact();
        let mut world = SparseVolume::new(256, 64, 256);
        world.paste_chunk(chunk.clone(), 16, 0, 0);
        assert_eq!(world.get_voxel(128 + 64, 32, 64), 4);
        let mut ring = SparseVolume::new(8, 8, 8);
        ring.ring_init(4, 64);
        ring.ring_paste(&chunk, -3, 5);
        assert!(ring.chunk_loaded(-3, 5));
        assert_eq!(ring.get_voxel(-3 * 128 + 64, 32, 5 * 128 + 64), 4);
        assert_eq!(ring.get_voxel(64, 32, 64), 0);
        let back = ring.ring_extract(-3, 5);
        assert_eq!(back.get_voxel(64, 32, 64), 4);
        ring.ring_evict(-3, 5);
        assert!(!ring.chunk_loaded(-3, 5));
        assert_eq!(ring.get_voxel(-3 * 128 + 64, 32, 5 * 128 + 64), 0);
    }
}
