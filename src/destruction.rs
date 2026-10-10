//! Destruction: after a carve, pieces that no longer connect to the ground become rigid bodies
//! (a cut tree takes its own crown and tips over toward the cut); loose foliage falls, tiny
//! pieces vanish. Bodies split when a carve cuts them in two.

use std::collections::HashMap;
use std::hash::{BuildHasherDefault, Hasher};
use std::time::Instant;

use crate::math::{Mat3, Vec3};
use crate::objects::VoxObject;
use crate::voxel::world::{wm, World};

/// Smaller islands just vanish.
const MIN_ISLAND_VOXELS: usize = 30;
/// rad/s a felled tree starts tipping with (physics keeps it at least that until it leans past
/// its balance).
const TOPPLE_SPIN: f32 = 0.15;
/// Structural voxels one support search may visit (then: it holds).
const SUPPORT_BUDGET: usize = 2_000_000;
/// Foliage voxels a loose clump's search may visit.
const LEAF_BUDGET: usize = 800_000;
/// The undo region around the cut itself (pieces report their own bounds).
pub const DETACH_MARGIN: i32 = 48;
pub const DETACH_MAX_HALF: i32 = 88;

/// Down last: popped first by the depth-first searches.
const DIRS: [[i32; 3]; 6] = [[0, 1, 0], [1, 0, 0], [-1, 0, 0], [0, 0, 1], [0, 0, -1], [0, -1, 0]];

#[derive(Clone, Copy, Debug, PartialEq, Eq)]
struct Cell {
    x: i32,
    y: i32,
    z: i32,
}

const fn cell(x: i32, y: i32, z: i32) -> Cell {
    Cell { x, y, z }
}

/// Hashing for the voxel-key maps: one 64-bit finalizer (murmur3 fmix64) per key. SipHash is
/// several times slower on the millions of lookups a big felling does, and the keys are not
/// attacker-controlled. (A plain multiply would leave the bucket bits blind to x, which sits in
/// the key's high bits.)
#[derive(Default)]
struct VoxHasher(u64);

impl Hasher for VoxHasher {
    fn finish(&self) -> u64 {
        self.0
    }
    fn write(&mut self, bytes: &[u8]) {
        for &b in bytes {
            self.write_u64(self.0 ^ b as u64);
        }
    }
    fn write_u64(&mut self, v: u64) {
        let mut k = v;
        k ^= k >> 33;
        k = k.wrapping_mul(0xff51_afd7_ed55_8ccd);
        k ^= k >> 33;
        k = k.wrapping_mul(0xc4ce_b9fe_1a85_ec53);
        k ^= k >> 33;
        self.0 = k;
    }
}

type FxMap<V> = HashMap<u64, V, BuildHasherDefault<VoxHasher>>;

/// A voxel's key (absolute coordinates; endless worlds are unbounded in x / z).
#[inline]
fn vkey(x: i32, y: i32, z: i32) -> u64 {
    ((x.wrapping_add(0x80_0000) as u32 as u64) << 40) | (((y & 0xFFFF) as u64) << 24) | ((z.wrapping_add(0x80_0000) as u32 & 0xFF_FFFF) as u64)
}

#[inline]
fn structural(m: u8) -> bool {
    m != 0 && m != wm::WATER && !wm::is_foliage_voxel(m)
}
#[inline]
fn foliage(m: u8) -> bool {
    m != 0 && wm::is_foliage_voxel(m)
}
/// Wood and bark: a tree is held only by its own way down to the ground.
#[inline]
fn woody(m: u8) -> bool {
    m != 0 && wm::is_plant(m) && !wm::is_foliage_voxel(m)
}

fn bounds(cells: &[Cell]) -> ([i32; 3], [i32; 3]) {
    let mut lo = [cells[0].x, cells[0].y, cells[0].z];
    let mut hi = lo;
    for c in cells {
        for (a, v) in [c.x, c.y, c.z].into_iter().enumerate() {
            lo[a] = lo[a].min(v);
            hi[a] = hi[a].max(v);
        }
    }
    (lo, hi)
}

/// Unloaded chunks (endless worlds) and everything outside a [`DenseRegion`] read this.
const HOLDS: u8 = 255;

/// A box of the world copied brick by brick (materials) plus a label per voxel: the races
/// below run on it instead of per-voxel lookups and hash maps.
#[derive(Default)]
struct DenseRegion {
    lo: [i32; 3],
    hi: [i32; 3],
    d: [i32; 3],
    mat: Vec<u8>,
    lab: Vec<u8>,
}

impl DenseRegion {
    fn build(w: &World, lo: [i32; 3], hi: [i32; 3]) -> Self {
        let mut r = DenseRegion { lo, hi, ..Default::default() };
        r.lo[1] = r.lo[1].max(0);
        r.hi[1] = r.hi[1].min(w.ny - 1);
        for k in 0..3 {
            r.d[k] = (r.hi[k] - r.lo[k] + 1).max(0);
        }
        let n = r.d[0] as usize * r.d[1] as usize * r.d[2] as usize;
        r.mat = vec![0; n];
        r.lab = vec![0; n];
        if n == 0 {
            return r;
        }
        for bz in r.lo[2] >> 3..=r.hi[2] >> 3 {
            for bx in r.lo[0] >> 3..=r.hi[0] >> 3 {
                let holds = w.ring_w != 0 && !w.chunk_loaded(bx >> 4, bz >> 4);
                let outside = w.ring_w == 0 && (bx < 0 || bz < 0 || bx >= w.bricks_x || bz >= w.bricks_z);
                for by in r.lo[1] >> 3..=r.hi[1] >> 3 {
                    let g = if holds {
                        HOLDS as u32
                    } else if outside {
                        0
                    } else {
                        w.grid[w.grid_index(bx, by, bz)]
                    };
                    if g == 0 {
                        continue;
                    }
                    let vox = (g >= 256).then(|| &w.bricks[(g - 256) as usize].vox);
                    let (x0, x1) = ((bx * 8).max(r.lo[0]), (bx * 8 + 7).min(r.hi[0]));
                    let (y0, y1) = ((by * 8).max(r.lo[1]), (by * 8 + 7).min(r.hi[1]));
                    let (z0, z1) = ((bz * 8).max(r.lo[2]), (bz * 8 + 7).min(r.hi[2]));
                    for z in z0..=z1 {
                        for y in y0..=y1 {
                            let i0 = r.idx(x0, y, z);
                            let row = &mut r.mat[i0..i0 + (x1 - x0 + 1) as usize];
                            match vox {
                                None => row.fill(g as u8),
                                Some(v) => {
                                    let src = ((z & 7) * 64 + (y & 7) * 8) as usize;
                                    for x in x0..=x1 {
                                        row[(x - x0) as usize] = v[src + (x & 7) as usize];
                                    }
                                }
                            }
                        }
                    }
                }
            }
        }
        r
    }

    #[inline]
    fn inside(&self, x: i32, y: i32, z: i32) -> bool {
        x >= self.lo[0] && x <= self.hi[0] && y >= self.lo[1] && y <= self.hi[1] && z >= self.lo[2] && z <= self.hi[2]
    }
    #[inline]
    fn idx(&self, x: i32, y: i32, z: i32) -> usize {
        ((z - self.lo[2]) as usize * self.d[1] as usize + (y - self.lo[1]) as usize) * self.d[0] as usize + (x - self.lo[0]) as usize
    }
    #[inline]
    fn m(&self, x: i32, y: i32, z: i32) -> u8 {
        if self.inside(x, y, z) { self.mat[self.idx(x, y, z)] } else { HOLDS }
    }
    #[inline]
    fn lab_at(&mut self, c: Cell) -> &mut u8 {
        let i = self.idx(c.x, c.y, c.z);
        &mut self.lab[i]
    }
}

/// Marks of the detach searches.
const M_SEARCH: u8 = 1;
const M_ATTACHED: u8 = 2;
const M_ISLAND: u8 = 3;
const M_LEAF_SEEN: u8 = 4;

/// Called with each piece's voxel bounds before it leaves the world (undo).
pub type DetachRegionFn<'a> = &'a mut dyn FnMut([i32; 3], [i32; 3]);

struct Detach<'a> {
    world: &'a mut World,
    objects: &'a mut Vec<VoxObject>,
    center: Vec3,
    r: f32,
    before_remove: Option<DetachRegionFn<'a>>,
    mark: FxMap<u8>,
    comp: Vec<Cell>,
    stack: Vec<Cell>,
    spawned: usize,
    t_support: f64,
    t_wood: f64,
    t_leaves: f64,
    t_spawn: f64,
}

fn ms_since(t: Instant) -> f64 {
    t.elapsed().as_secs_f64() * 1e3
}

/// Build a new body from world cells (world axes, so an identity rotation), taking them out
/// of the world.
fn spawn_from_world_cells(world: &mut World, cells: &[Cell], objects: &mut Vec<VoxObject>) {
    let (mn, mx) = bounds(cells);
    let dim = |a: usize| ((mx[a] - mn[a] + 1 + 7) / 8) * 8;
    let mut o = VoxObject::new(dim(0), dim(1), dim(2));
    for c in cells {
        o.vol.set_voxel(c.x - mn[0], c.y - mn[1], c.z - mn[2], world.get_voxel(c.x, c.y, c.z));
        world.set_voxel(c.x, c.y, c.z, 0);
    }
    o.vol.compact();
    o.pose.r = Mat3::IDENTITY;
    o.pose.t = Vec3::new(mn[0] as f32, mn[1] as f32, mn[2] as f32);
    o.center = o.pose.t;
    o.init_body();
    o.asleep = false;
    objects.push(o);
}

impl Detach<'_> {
    #[inline]
    fn mark_of(&self, x: i32, y: i32, z: i32) -> u8 {
        self.mark.get(&vkey(x, y, z)).copied().unwrap_or(0)
    }
    #[inline]
    fn unloaded(&self, x: i32, z: i32) -> bool {
        self.world.ring_w != 0 && !self.world.loaded_at(x, z)
    }
    #[inline]
    fn mat(&self, x: i32, y: i32, z: i32) -> u8 {
        if y < 0 || y >= self.world.ny { 0 } else { self.world.get_voxel(x, y, z) }
    }

    /// Carve off cells (a body when big enough). Reports the region to the undo first.
    fn remove_or_spawn(&mut self, cells: &[Cell], topple: bool) {
        if cells.is_empty() {
            return;
        }
        let (lo, hi) = bounds(cells);
        if let Some(f) = self.before_remove.as_mut() {
            f(lo, hi);
        }
        if cells.len() < MIN_ISLAND_VOXELS {
            for c in cells {
                self.world.set_voxel(c.x, c.y, c.z, 0);
            }
            return;
        }
        let trunk: Vec<Cell> = if topple { cells.iter().copied().filter(|c| woody(self.world.get_voxel(c.x, c.y, c.z))).collect() } else { Vec::new() };
        spawn_from_world_cells(self.world, cells, self.objects);
        self.spawned += 1;
        if trunk.is_empty() {
            return;
        }
        // Topple: a tall piece standing on its cut would balance; tip it toward the cut side.
        let (tlo, thi) = bounds(&trunk);
        let height = thi[1] - tlo[1] + 1;
        let width = (thi[0] - tlo[0]).max(thi[2] - tlo[2]) + 1;
        if height < 2 * width && height < 40 {
            return;
        }
        // Its base: the cut face (the wood the carve bared) - not its lowest wood, which on a
        // pine is a low branch tip below the cut. (Not touched by the cut: its lowest layers.)
        let (center, r) = (self.center, self.r);
        let mut base: Vec<Cell> = trunk
            .iter()
            .copied()
            .filter(|c| (Vec3::new(c.x as f32 + 0.5, c.y as f32 + 0.5, c.z as f32 + 0.5) - center).length_squared() <= (r + 2.5) * (r + 2.5))
            .collect();
        if base.is_empty() {
            base = trunk.iter().copied().filter(|c| c.y <= tlo[1] + 2).collect();
        }
        let (mut ax, mut az) = (0.0f64, 0.0f64);
        let mut base_y = base[0].y;
        for c in &base {
            ax += c.x as f64 + 0.5;
            az += c.z as f64 + 0.5;
            base_y = base_y.min(c.y);
        }
        ax /= base.len() as f64;
        az /= base.len() as f64;
        let mut dir = Vec3::new((center.x as f64 - ax) as f32, 0.0, (center.z as f64 - az) as f32);
        let l = dir.length();
        if l < 0.3 {
            // Cut straight through the middle: some direction, the same for the same tree.
            let h = (tlo[0].wrapping_mul(73_856_093) as u32) ^ (tlo[2].wrapping_mul(19_349_663) as u32) ^ (tlo[1].wrapping_mul(83_492_791) as u32);
            let a = (h % 6283) as f32 * 0.001;
            dir = Vec3::new(a.cos(), 0.0, a.sin());
        } else {
            dir /= l;
        }
        // The hinge: the trunk's edge toward dir at the cut, held over the stump, so the butt
        // pivots there and the tree falls like a rod hinged at its base.
        let edge = base.iter().map(|c| (c.x as f32 + 0.5 - ax as f32) * dir.x + (c.z as f32 + 0.5 - az as f32) * dir.z).fold(0.0f32, f32::max);
        let hinge = Vec3::new(ax as f32 + dir.x * edge, base_y as f32, az as f32 + dir.z * edge);
        let hx = (hinge.x - dir.x * 0.5).floor() as i32;
        let hz = (hinge.z - dir.z * 0.5).floor() as i32;
        let mut top = base_y - 1; // the stump's top under the hinge
        while top > base_y - 48 && top > 0 && !structural(self.mat(hx, top, hz)) {
            top -= 1;
        }
        let anchored = structural(self.mat(hx, top, hz));
        let body = self.objects.last_mut().unwrap();
        body.ang_vel = Vec3::Y.cross(dir) * TOPPLE_SPIN; // the top moves along dir
        body.hinged = true;
        body.hinge_local = hinge - body.pose.t; // (spawned unrotated)
        body.fall_dir = dir;
        body.hinge_anchor = Vec3::new(hinge.x, if anchored { (top + 1) as f32 } else { -1e9 }, hinge.z);
    }

    /// An island's foliage: what is nearer (through foliage) to it than to any other structure,
    /// by a breadth-first race from both sides; then the orphaned plant voxels around it.
    fn take_foliage(&mut self, island: &mut Vec<Cell>) {
        if island.is_empty() {
            return;
        }
        const MARGIN: i32 = 40;
        let (mut lo, mut hi) = bounds(island);
        for k in 0..3 {
            lo[k] -= MARGIN;
            hi[k] += MARGIN;
        }
        let mut rg = DenseRegion::build(self.world, lo, hi);
        const ISLE: u8 = 1;
        const SET: u8 = 2;
        const MINE: u8 = 4;
        const OTHER: u8 = 6;
        const LAB: u8 = 6;
        let leaf = |rg: &DenseRegion, x: i32, y: i32, z: i32| {
            let m = rg.m(x, y, z);
            m != HOLDS && foliage(m)
        };
        for c in island.iter() {
            if rg.inside(c.x, c.y, c.z) {
                *rg.lab_at(*c) |= ISLE;
            }
        }
        let mut leaves: Vec<Cell> = Vec::new();
        let add = |rg: &mut DenseRegion, leaves: &mut Vec<Cell>, x: i32, y: i32, z: i32| {
            if !rg.inside(x, y, z) {
                return;
            }
            let i = rg.idx(x, y, z);
            if rg.lab[i] & LAB != 0 || !leaf(rg, x, y, z) {
                return;
            }
            rg.lab[i] |= SET;
            leaves.push(cell(x, y, z));
        };
        for c in island.iter() {
            for d in DIRS {
                add(&mut rg, &mut leaves, c.x + d[0], c.y + d[1], c.z + d[2]);
            }
        }
        // Only REACH leaf-steps out from the island; the last layer counts as someone else's
        // where more foliage goes on.
        const REACH: i32 = 32;
        let (mut layer_start, mut layer_end) = (0, leaves.len());
        let mut depth = 1;
        while depth < REACH && layer_start < layer_end {
            for i in layer_start..layer_end {
                let c = leaves[i];
                for d in DIRS {
                    add(&mut rg, &mut leaves, c.x + d[0], c.y + d[1], c.z + d[2]);
                }
            }
            layer_start = layer_end;
            layer_end = leaves.len();
            depth += 1;
        }
        for &c in &leaves[layer_start..layer_end] {
            for d in DIRS {
                let (x, y, z) = (c.x + d[0], c.y + d[1], c.z + d[2]);
                if rg.inside(x, y, z) && rg.lab[rg.idx(x, y, z)] & LAB == 0 && leaf(&rg, x, y, z) {
                    let v = rg.lab_at(c);
                    *v = (*v & !LAB) | OTHER;
                    break;
                }
            }
        }
        let mut q: Vec<Cell> = Vec::new();
        for &c in &leaves {
            // The starting sides.
            if *rg.lab_at(c) & LAB == OTHER {
                q.push(c);
                continue;
            }
            let (mut mine, mut other) = (false, false);
            for d in DIRS {
                let (x, y, z) = (c.x + d[0], c.y + d[1], c.z + d[2]);
                let m = rg.m(x, y, z);
                if m == HOLDS {
                    other = true; // beyond the box: someone's
                    continue;
                }
                if rg.lab[rg.idx(x, y, z)] & ISLE != 0 {
                    mine = true;
                    continue;
                }
                if structural(m) {
                    other = true;
                }
            }
            if mine || other {
                let v = rg.lab_at(c);
                *v = (*v & !LAB) | if other { OTHER } else { MINE };
                q.push(c);
            }
        }
        let mut next = Vec::new();
        while !q.is_empty() {
            next.clear();
            for &c in &q {
                let l = *rg.lab_at(c) & LAB;
                for d in DIRS {
                    let (x, y, z) = (c.x + d[0], c.y + d[1], c.z + d[2]);
                    if !rg.inside(x, y, z) {
                        continue;
                    }
                    let i = rg.idx(x, y, z);
                    if rg.lab[i] & LAB != SET {
                        continue;
                    }
                    rg.lab[i] = (rg.lab[i] & !LAB) | l;
                    next.push(cell(x, y, z));
                }
            }
            std::mem::swap(&mut q, &mut next);
        }
        for &c in &leaves {
            if *rg.lab_at(c) & LAB == MINE {
                island.push(c);
            }
        }

        // Orphans: crowns aren't one face-connected blob (leaves touch by edges and corners, sit
        // past the race, float in loose clusters). Any plant voxel near the felled piece that no
        // longer connects (26-way, through plant voxels) to something holding it goes along too.
        const HELD_FOL: u8 = 8;
        const TAKEN: u8 = 16;
        const GROUP: u8 = 32;
        const ORPHAN_MARGIN: i32 = 40;
        let plant = |m: u8| m != 0 && m != HOLDS && (wm::is_plant(m) || wm::is_foliage_voxel(m));
        let (mut olo, mut ohi) = bounds(island);
        for k in 0..3 {
            olo[k] = (olo[k] - ORPHAN_MARGIN).max(rg.lo[k]);
            ohi[k] = (ohi[k] + ORPHAN_MARGIN).min(rg.hi[k]);
        }
        let in_o = |x: i32, y: i32, z: i32| x >= olo[0] && x <= ohi[0] && y >= olo[1] && y <= ohi[1] && z >= olo[2] && z <= ohi[2];
        for &c in &leaves {
            if *rg.lab_at(c) & LAB == MINE {
                *rg.lab_at(c) |= TAKEN;
            }
        }
        let mut held: Vec<Cell> = Vec::new();
        let mut rest: Vec<Cell> = Vec::new(); // the plant voxels in the box nobody has yet
        for z in olo[2]..=ohi[2] {
            for y in olo[1]..=ohi[1] {
                for x in olo[0]..=ohi[0] {
                    let i = rg.idx(x, y, z);
                    if !plant(rg.mat[i]) || rg.lab[i] & (TAKEN | ISLE) != 0 {
                        continue;
                    }
                    rest.push(cell(x, y, z));
                    let mut anchor = y <= 0;
                    for d in DIRS {
                        if anchor {
                            break;
                        }
                        let (ax, ay, az) = (x + d[0], y + d[1], z + d[2]);
                        if !in_o(ax, ay, az) {
                            continue;
                        }
                        let m = rg.m(ax, ay, az);
                        anchor = structural(m) && !plant(m) && rg.lab[rg.idx(ax, ay, az)] & ISLE == 0;
                    }
                    if anchor {
                        rg.lab[i] |= HELD_FOL;
                        held.push(cell(x, y, z));
                    }
                }
            }
        }
        let flood_held = |rg: &mut DenseRegion, held: &mut Vec<Cell>| {
            while let Some(c) = held.pop() {
                for dz in -1..=1 {
                    for dy in -1..=1 {
                        for dx in -1..=1 {
                            let (x, y, z) = (c.x + dx, c.y + dy, c.z + dz);
                            if !in_o(x, y, z) {
                                continue;
                            }
                            let i = rg.idx(x, y, z);
                            if !plant(rg.mat[i]) || rg.lab[i] & (HELD_FOL | TAKEN | ISLE) != 0 {
                                continue;
                            }
                            rg.lab[i] |= HELD_FOL;
                            held.push(cell(x, y, z));
                        }
                    }
                }
            }
        };
        flood_held(&mut rg, &mut held);
        // What's left and goes on past the box: follow it out through the world (26-way, plant
        // voxels) until something holds it (ground, rock, a building, an unloaded chunk, a held
        // plant voxel in the box, or the budget: a forest's touching crowns stand) - or it runs
        // out: it floats, take it.
        const OUT_BUDGET: usize = 400_000;
        // World voxels outside the box: the group (generation) that last reached each one, or
        // OUT_HELD - reached by a held group, so held too (same component): a later group
        // touching it holds at once instead of re-flooding the neighbour's crown.
        const OUT_HELD: u32 = u32::MAX;
        let mut out_seen: FxMap<u32> = FxMap::default();
        let mut generation = 0u32;
        let (mut group, mut outside, mut gstack) = (Vec::new(), Vec::new(), Vec::new());
        for &c0 in &rest {
            let i0 = rg.idx(c0.x, c0.y, c0.z);
            if rg.lab[i0] & (HELD_FOL | GROUP) != 0 {
                continue;
            }
            group.clear();
            outside.clear();
            gstack.clear();
            generation += 1;
            rg.lab[i0] |= GROUP;
            gstack.push(c0);
            let (mut past, mut holds) = (false, false);
            while let Some(c) = gstack.pop() {
                group.push(c);
                for dz in -1..=1 {
                    for dy in -1..=1 {
                        for dx in -1..=1 {
                            let (x, y, z) = (c.x + dx, c.y + dy, c.z + dz);
                            if !in_o(x, y, z) {
                                if y < 0 || y >= self.world.ny {
                                    continue;
                                }
                                if self.unloaded(x, z) {
                                    holds = true;
                                    continue;
                                }
                                let m = if rg.inside(x, y, z) { rg.mat[rg.idx(x, y, z)] } else { self.mat(x, y, z) };
                                if plant(m) {
                                    let s = out_seen.entry(vkey(x, y, z)).or_insert(0);
                                    if *s == OUT_HELD {
                                        holds = true;
                                    } else if *s != generation {
                                        *s = generation;
                                        outside.push(cell(x, y, z));
                                    }
                                } else if structural(m) {
                                    holds = true;
                                }
                                past |= plant(m);
                                continue;
                            }
                            let i = rg.idx(x, y, z);
                            if !plant(rg.mat[i]) || rg.lab[i] & (GROUP | TAKEN | ISLE) != 0 {
                                continue;
                            }
                            if rg.lab[i] & HELD_FOL != 0 {
                                holds = true;
                                continue;
                            }
                            rg.lab[i] |= GROUP;
                            gstack.push(cell(x, y, z));
                        }
                    }
                }
            }
            if !past && !holds {
                island.extend_from_slice(&group); // closed inside the box: floats
                continue;
            }
            let mut k = 0;
            while k < outside.len() && !holds {
                // Breadth-first out in the world.
                if outside.len() > OUT_BUDGET {
                    holds = true;
                    break;
                }
                let c = outside[k];
                k += 1;
                if c.y <= 0 {
                    holds = true;
                    break;
                }
                'n: for dz in -1..=1 {
                    for dy in -1..=1 {
                        for dx in -1..=1 {
                            let (x, y, z) = (c.x + dx, c.y + dy, c.z + dz);
                            if y < 0 || y >= self.world.ny {
                                continue;
                            }
                            if in_o(x, y, z) {
                                let i = rg.idx(x, y, z);
                                let m = rg.mat[i];
                                if rg.lab[i] & (TAKEN | ISLE) != 0 {
                                    continue;
                                }
                                if rg.lab[i] & HELD_FOL != 0 || (structural(m) && !plant(m)) {
                                    holds = true;
                                } else if plant(m) && rg.lab[i] & GROUP == 0 {
                                    let s = out_seen.entry(vkey(x, y, z)).or_insert(0);
                                    if *s != generation {
                                        *s = generation;
                                        outside.push(cell(x, y, z)); // another group of the box: same fate
                                    }
                                }
                            } else if self.unloaded(x, z) {
                                holds = true;
                            } else {
                                let m = if rg.inside(x, y, z) { rg.mat[rg.idx(x, y, z)] } else { self.mat(x, y, z) };
                                if plant(m) {
                                    let s = out_seen.entry(vkey(x, y, z)).or_insert(0);
                                    if *s == OUT_HELD {
                                        holds = true;
                                    } else if *s != generation {
                                        *s = generation;
                                        outside.push(cell(x, y, z));
                                    }
                                } else if structural(m) {
                                    holds = true;
                                }
                            }
                            if holds {
                                break 'n;
                            }
                        }
                    }
                }
            }
            if holds {
                // The group (and any box voxel it reached) is held.
                for &c in &group {
                    *rg.lab_at(c) |= HELD_FOL;
                    held.push(c);
                }
                for &c in &outside {
                    if !in_o(c.x, c.y, c.z) {
                        out_seen.insert(vkey(c.x, c.y, c.z), OUT_HELD);
                    } else if *rg.lab_at(c) & HELD_FOL == 0 {
                        *rg.lab_at(c) |= HELD_FOL;
                        held.push(c);
                    }
                }
                flood_held(&mut rg, &mut held);
            } else {
                island.extend_from_slice(&group);
                for &c in &outside {
                    if in_o(c.x, c.y, c.z) {
                        let v = rg.lab_at(c);
                        if *v & (GROUP | HELD_FOL) != 0 {
                            continue;
                        }
                        *v |= GROUP;
                    }
                    island.push(c);
                }
            }
        }
    }

    /// Down-first search through structural voxels from (x, y, z) into `comp`. true = attached.
    fn support_search(&mut self, x0: i32, y0: i32, z0: i32) -> bool {
        self.comp.clear();
        self.stack.clear();
        self.stack.push(cell(x0, y0, z0));
        self.mark.insert(vkey(x0, y0, z0), M_SEARCH);
        let mut attached = false;
        'search: while let Some(c) = self.stack.pop() {
            self.comp.push(c);
            if c.y <= 0 || self.comp.len() > SUPPORT_BUDGET {
                attached = true;
                break;
            }
            for d in DIRS {
                let (x, y, z) = (c.x + d[0], c.y + d[1], c.z + d[2]);
                if y >= self.world.ny {
                    continue;
                }
                if self.unloaded(x, z) {
                    attached = true;
                    break 'search;
                }
                let mk = self.mark_of(x, y, z);
                if mk == M_ATTACHED {
                    attached = true;
                    break 'search;
                }
                if mk != 0 {
                    continue;
                }
                let m = self.mat(x, y, z);
                if !structural(m) {
                    continue;
                }
                // Into wood only sideways or down: a cut tree isn't held by climbing its own
                // crown into a neighbour's branches (rock and buildings do hold: arches).
                if d[1] > 0 && woody(m) {
                    continue;
                }
                self.mark.insert(vkey(x, y, z), M_SEARCH);
                self.stack.push(cell(x, y, z));
            }
        }
        if self.comp.len() > SUPPORT_BUDGET {
            log::warn!(target: "detach", "support search stopped at {} voxels - counted as holding", self.comp.len());
        }
        let res = if attached { M_ATTACHED } else { M_ISLAND };
        for c in self.comp.iter().chain(&self.stack) {
            self.mark.insert(vkey(c.x, c.y, c.z), res);
        }
        attached
    }

    /// A piece with wood in it: which wood goes with it, by a breadth-first race through the
    /// connected wood (near the cut) from the cut piece and from every bit of wood touching
    /// other structure. Empty: a whole interlocked forest, it holds.
    fn plant_island(&mut self, core: &[Cell]) -> Vec<Cell> {
        const PLANT_RADIUS: i32 = 72;
        const BELOW: i32 = 24;
        const ABOVE: i32 = 320;
        let (mut cx, mut cz, mut miny) = (0i64, 0i64, core[0].y);
        for c in core {
            cx += c.x as i64;
            cz += c.z as i64;
            miny = miny.min(c.y);
        }
        let (cx, cz) = ((cx / core.len() as i64) as i32, (cz / core.len() as i64) as i32);
        let mut rg = DenseRegion::build(
            self.world,
            [cx - PLANT_RADIUS, miny - BELOW, cz - PLANT_RADIUS],
            [cx + PLANT_RADIUS, miny + ABOVE, cz + PLANT_RADIUS],
        );
        const WOOD: u8 = 1;
        const FALLS: u8 = 2;
        const HELD: u8 = 3;
        let (mut wood, mut q, mut held_seeds) = (Vec::new(), Vec::new(), Vec::new());
        for &c in core {
            if rg.inside(c.x, c.y, c.z) && woody(rg.m(c.x, c.y, c.z)) && *rg.lab_at(c) == 0 {
                *rg.lab_at(c) = FALLS;
                wood.push(c);
                q.push(c);
            }
        }
        let n_core = wood.len();
        let mut i = 0;
        while i < wood.len() {
            let c = wood[i];
            let mut held = c.y <= 0;
            for d in DIRS {
                let (x, y, z) = (c.x + d[0], c.y + d[1], c.z + d[2]);
                let m = rg.m(x, y, z);
                if m == HOLDS {
                    held |= y >= 0; // beyond the box / unloaded: holds
                    continue;
                }
                if woody(m) {
                    let j = rg.idx(x, y, z);
                    if rg.lab[j] == 0 {
                        rg.lab[j] = WOOD;
                        wood.push(cell(x, y, z));
                    }
                    continue;
                }
                held |= structural(m) && self.mark_of(x, y, z) != M_ISLAND;
            }
            if held && i >= n_core {
                *rg.lab_at(c) = HELD;
                held_seeds.push(c);
            }
            i += 1;
        }
        q.extend(held_seeds);
        let mut next = Vec::new();
        while !q.is_empty() {
            next.clear();
            for &c in &q {
                let l = *rg.lab_at(c);
                for d in DIRS {
                    let (x, y, z) = (c.x + d[0], c.y + d[1], c.z + d[2]);
                    if !rg.inside(x, y, z) {
                        continue;
                    }
                    let j = rg.idx(x, y, z);
                    if rg.lab[j] != WOOD {
                        continue;
                    }
                    rg.lab[j] = l;
                    next.push(cell(x, y, z));
                }
            }
            std::mem::swap(&mut q, &mut next);
        }
        // The core's other structure (a stone stuck in the trunk) comes along.
        let mut falls: Vec<Cell> = core.iter().copied().filter(|c| !woody(self.mat(c.x, c.y, c.z))).collect();
        for &c in &wood {
            if *rg.lab_at(c) == FALLS {
                self.mark.insert(vkey(c.x, c.y, c.z), M_ISLAND);
                falls.push(c);
            }
        }
        falls
    }

    fn island_from(&mut self, x: i32, y: i32, z: i32) {
        let t0 = Instant::now();
        let held = self.support_search(x, y, z);
        self.t_support += ms_since(t0);
        if held {
            return;
        }
        let mut island = self.comp.clone();
        if island.iter().any(|c| woody(self.mat(c.x, c.y, c.z))) {
            let t0 = Instant::now();
            let core = std::mem::take(&mut self.comp);
            island = self.plant_island(&core);
            self.comp = core;
            self.t_wood += ms_since(t0);
            if island.is_empty() {
                return; // a whole interlocked forest: holds
            }
        }
        let t0 = Instant::now();
        self.take_foliage(&mut island);
        self.t_leaves += ms_since(t0);
        let t0 = Instant::now();
        self.remove_or_spawn(&island, true);
        self.t_spawn += ms_since(t0);
    }

    fn run(&mut self) {
        let r1 = self.r + 1.8; // seed shell: just outside the carved sphere
        let c = self.center.to_array();
        let mut s0 = c.map(|v| (v - r1).floor() as i32);
        let mut s1 = c.map(|v| (v + r1).ceil() as i32);
        s0[1] = s0[1].max(0);
        s1[1] = s1[1].min(self.world.ny - 1);
        let rin = (self.r - 0.5) * (self.r - 0.5);
        for z in s0[2]..=s1[2] {
            for y in s0[1]..=s1[1] {
                for x in s0[0]..=s1[0] {
                    let d2 = (Vec3::new(x as f32 + 0.5, y as f32 + 0.5, z as f32 + 0.5) - self.center).length_squared();
                    if d2 < rin || d2 > r1 * r1 {
                        continue; // shell only
                    }
                    if self.mark_of(x, y, z) != 0 || self.unloaded(x, z) {
                        continue;
                    }
                    let m = self.mat(x, y, z);
                    if structural(m) {
                        self.island_from(x, y, z);
                        continue;
                    }
                    if !foliage(m) {
                        continue;
                    }
                    // Foliage: whatever structure it touches decides; none at all = a loose clump.
                    let mut clump = vec![cell(x, y, z)];
                    let mut q = vec![cell(x, y, z)];
                    let mut touched = Vec::new();
                    self.mark.insert(vkey(x, y, z), M_LEAF_SEEN);
                    while clump.len() < LEAF_BUDGET {
                        let Some(c) = q.pop() else { break };
                        for d in DIRS {
                            let (nx, ny, nz) = (c.x + d[0], c.y + d[1], c.z + d[2]);
                            let nm = self.mat(nx, ny, nz);
                            if structural(nm) || (nm != 0 && self.unloaded(nx, nz)) {
                                touched.push(cell(nx, ny, nz));
                                continue;
                            }
                            if !foliage(nm) || self.mark_of(nx, ny, nz) != 0 {
                                continue;
                            }
                            self.mark.insert(vkey(nx, ny, nz), M_LEAF_SEEN);
                            clump.push(cell(nx, ny, nz));
                            q.push(cell(nx, ny, nz));
                        }
                    }
                    if touched.is_empty() && clump.len() < LEAF_BUDGET {
                        // Nothing holds it.
                        for c in &clump {
                            self.mark.insert(vkey(c.x, c.y, c.z), M_ISLAND);
                        }
                        self.remove_or_spawn(&clump, false);
                        continue;
                    }
                    for t in touched {
                        // Its tree may have been cut loose: the island takes its leaves.
                        if self.mark_of(t.x, t.y, t.z) == 0 && !self.unloaded(t.x, t.z) {
                            self.island_from(t.x, t.y, t.z);
                        }
                    }
                }
            }
        }
    }
}

/// After a world carve at `center` (radius r): pieces whose structural voxels (all solid but
/// foliage and water) no longer connect down to the bottom of the world become dynamic bodies
/// with the foliage nearest to them; tall ones tip over toward the cut. Loose foliage falls,
/// tiny pieces vanish. Returns the number of bodies spawned (> 0: the caller rebuilds the GPU
/// bodies).
pub fn world_carve_detach<'a>(world: &'a mut World, center: Vec3, r: f32, objects: &'a mut Vec<VoxObject>, before_remove: Option<DetachRegionFn<'a>>) -> usize {
    let mut d = Detach {
        world,
        objects,
        center,
        r,
        before_remove,
        mark: FxMap::with_capacity_and_hasher(1 << 16, Default::default()),
        comp: Vec::new(),
        stack: Vec::new(),
        spawned: 0,
        t_support: 0.0,
        t_wood: 0.0,
        t_leaves: 0.0,
        t_spawn: 0.0,
    };
    d.run();
    if d.spawned > 0 {
        log::info!(target: "detach", "{} piece(s): support {:.1} ms, wood {:.1}, foliage {:.1}, bodies {:.1}", d.spawned, d.t_support,
                   d.t_wood, d.t_leaves, d.t_spawn);
    }
    d.spawned
}

/// Split body `idx` if its volume has come apart: the largest component stays, every other one
/// becomes a new body with the same pose and the rigid-body velocity at its centre of mass.
/// Returns the number of new bodies.
pub fn split_object(objects: &mut Vec<VoxObject>, idx: usize) -> usize {
    let comps = {
        let v = &objects[idx].vol;
        let (nx, ny, nz) = (v.nx, v.ny, v.nz);
        let vi = |x: i32, y: i32, z: i32| (z as usize * ny as usize + y as usize) * nx as usize + x as usize;
        let mut visited = vec![false; nx as usize * ny as usize * nz as usize];
        let mut comps: Vec<Vec<Cell>> = Vec::new();
        // Only the non-empty bricks hold seeds.
        for gi in 0..v.grid.len() {
            if v.grid[gi] == 0 {
                continue;
            }
            let (bx, by, bz) = v.brick_coords(gi);
            for lz in 0..8 {
                for ly in 0..8 {
                    for lx in 0..8 {
                        let (x, y, z) = (bx * 8 + lx, by * 8 + ly, bz * 8 + lz);
                        if visited[vi(x, y, z)] || v.get_voxel(x, y, z) == 0 {
                            continue;
                        }
                        let mut comp = Vec::new();
                        let mut stack = vec![cell(x, y, z)];
                        visited[vi(x, y, z)] = true;
                        while let Some(c) = stack.pop() {
                            comp.push(c);
                            for d in DIRS {
                                let (x, y, z) = (c.x + d[0], c.y + d[1], c.z + d[2]);
                                if x as u32 >= nx as u32 || y as u32 >= ny as u32 || z as u32 >= nz as u32 {
                                    continue;
                                }
                                if visited[vi(x, y, z)] || v.get_voxel(x, y, z) == 0 {
                                    continue;
                                }
                                visited[vi(x, y, z)] = true;
                                stack.push(cell(x, y, z));
                            }
                        }
                        comps.push(comp);
                    }
                }
            }
        }
        comps
    };
    if comps.len() <= 1 {
        return 0;
    }
    let largest = (0..comps.len()).max_by_key(|&i| (comps[i].len(), std::cmp::Reverse(i))).unwrap();
    let mut spawned = 0;
    for (ci, cells) in comps.iter().enumerate() {
        if ci == largest {
            continue;
        }
        let src = &mut objects[idx];
        if cells.len() < MIN_ISLAND_VOXELS {
            for c in cells {
                src.vol.set_voxel(c.x, c.y, c.z, 0);
            }
            continue;
        }
        let (mn, mx) = bounds(cells);
        let dim = |a: usize| ((mx[a] - mn[a] + 1 + 7) / 8) * 8;
        let mut o = VoxObject::new(dim(0), dim(1), dim(2));
        for c in cells {
            o.vol.set_voxel(c.x - mn[0], c.y - mn[1], c.z - mn[2], src.vol.get_voxel(c.x, c.y, c.z));
            src.vol.set_voxel(c.x, c.y, c.z, 0);
        }
        o.vol.compact();
        // Same orientation; placed so nothing moves in the world.
        o.pose.r = src.pose.r;
        o.pose.t = src.pose.t + src.pose.r * Vec3::new(mn[0] as f32, mn[1] as f32, mn[2] as f32);
        o.init_body();
        // Rigid-body velocity at the piece's centre of mass: v + w x r.
        let src_com = src.pose.r * src.com + src.pose.t;
        let piece_com = o.pose.r * o.com + o.pose.t;
        o.vel = src.vel + src.ang_vel.cross(piece_com - src_com);
        o.ang_vel = src.ang_vel;
        o.asleep = false;
        o.glow_integrity = src.glow_integrity; // a broken lamp's pieces stay broken
        objects.push(o);
        spawned += 1;
    }
    let src = &mut objects[idx];
    src.init_body(); // the remaining piece's mass / shell changed
    src.asleep = false;
    src.sleep_timer = 0.0;
    spawned
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::voxel::world::{FlatLayer, FlatWorld};

    fn flat() -> World {
        let mut w = World::new(256, 128, 256);
        w.generate_flat(&FlatWorld { layers: vec![FlatLayer { mat: wm::STONE, thickness: 0 }], height: 16, water: 0 });
        w
    }

    #[test]
    fn cut_pillar_top_falls_as_a_body() {
        let mut w = flat();
        // A 6x6 stone pillar 60 tall on the ground; carve through it at y 40.
        w.fill_box(100, 16, 100, 105, 75, 105, wm::STONE);
        w.fill_sphere(102.5, 40.0, 102.5, 5.0, 0);
        let mut objs = Vec::new();
        let n = world_carve_detach(&mut w, Vec3::new(102.5, 40.0, 102.5), 5.0, &mut objs, None);
        assert_eq!(n, 1);
        assert_eq!(w.get_voxel(102, 60, 102), 0, "the top left the world");
        assert_eq!(w.get_voxel(102, 20, 102), wm::STONE, "the stump stays");
        let o = &objs[0];
        assert!(o.mass > 0.0 && !o.asleep);
        // Unrotated, at its voxels' old place: its top voxel is the pillar's (102, 75, 102).
        assert_eq!(o.pose.r, Mat3::IDENTITY);
        assert!((40.0..47.0).contains(&o.pose.t.y), "bottom at {}", o.pose.t.y);
        let t = o.pose.t;
        assert_eq!(o.vol.get_voxel(102 - t.x as i32, 75 - t.y as i32, 102 - t.z as i32), wm::STONE);
    }

    #[test]
    fn small_pieces_vanish_and_arches_hold() {
        let mut w = flat();
        // A tiny block on a thin stalk: cut the stalk, the block (< 30 voxels) vanishes.
        w.fill_box(50, 16, 50, 50, 30, 50, wm::STONE);
        w.fill_box(49, 31, 49, 51, 32, 51, wm::STONE);
        w.fill_sphere(50.5, 24.0, 50.5, 2.0, 0);
        let mut objs = Vec::new();
        assert_eq!(world_carve_detach(&mut w, Vec3::new(50.5, 24.0, 50.5), 2.0, &mut objs, None), 0);
        assert_eq!(w.get_voxel(50, 32, 50), 0);
        // An arch: cut one leg, the span still holds through the other.
        w.fill_box(150, 16, 150, 152, 50, 152, wm::STONE);
        w.fill_box(180, 16, 150, 182, 50, 152, wm::STONE);
        w.fill_box(150, 51, 150, 182, 54, 152, wm::STONE);
        w.fill_sphere(151.5, 30.0, 151.5, 3.0, 0);
        assert_eq!(world_carve_detach(&mut w, Vec3::new(151.5, 30.0, 151.5), 3.0, &mut objs, None), 0, "the cut leg hangs from the span");
        assert_eq!(w.get_voxel(166, 52, 151), wm::STONE, "the span holds");
    }

    #[test]
    fn split_keeps_largest_and_world_placement() {
        let mut o = VoxObject::new(32, 8, 8);
        o.vol.fill_box(0, 0, 0, 9, 7, 7, wm::STONE); // 640 voxels
        o.vol.fill_box(20, 0, 0, 24, 7, 7, wm::WOOD); // 320 voxels
        o.pose.t = Vec3::new(10.0, 20.0, 30.0);
        o.init_body();
        let mut objs = vec![o];
        assert_eq!(split_object(&mut objs, 0), 1);
        assert_eq!(objs[0].vol.get_voxel(22, 3, 3), 0);
        assert_eq!(objs[0].vol.get_voxel(3, 3, 3), wm::STONE);
        let p = objs[1].pose.apply(Vec3::new(2.5, 3.5, 3.5));
        assert!((p - Vec3::new(32.5, 23.5, 33.5)).length() < 1e-4);
        assert_eq!(objs[1].vol.get_voxel(2, 3, 3), wm::WOOD);
    }
}

#[cfg(test)]
mod generated_world {
    use super::*;

    /// Fells a broadleaf of the seed-1337 world (generation takes ~5 s: run with --ignored).
    #[test]
    #[ignore]
    fn fells_a_generated_tree() {
        let mut w = World::default();
        w.generate(1337);
        assert!(woody(w.get_voxel(650, 253, 657)), "the probe tree moved: worldgen changed?");
        let c = Vec3::new(650.5, 253.5, 656.5);
        w.fill_sphere(c.x, c.y, c.z, 9.0, 0);
        let mut objs = Vec::new();
        let t = Instant::now();
        assert_eq!(world_carve_detach(&mut w, c, 9.0, &mut objs, None), 1);
        println!("detach {:.1} ms, mass {}, shell {}", t.elapsed().as_secs_f64() * 1e3, objs[0].mass, objs[0].shell.len());
        let o = &objs[0];
        assert!(o.tree && o.hinged, "a tall tree tips over its hinge");
        assert!(o.vol.nx > 40 && o.vol.ny > 60, "it took its crown: {}x{}x{}", o.vol.nx, o.vol.ny, o.vol.nz);
        assert_eq!(w.get_voxel(650, 290, 657), 0, "the trunk above the cut left the world");
    }
}
