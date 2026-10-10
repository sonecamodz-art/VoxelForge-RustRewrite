//! Procedural vegetation (1 voxel = 10 cm): oaks, birches, pines, dead snags, bushes and
//! fallen logs at real-world sizes. Each plant is grown ONCE as a recipe - capsules (trunk /
//! limbs / roots) and leaf blobs - and rasterised on demand into any box (a world-generation
//! chunk), so a tree straddling chunks comes out identical from each side.
//!
//! * Broadleaf (oak), birch and dead snags: space colonisation (Runions et al. 2007) with
//!   pipe-model branch radii; leaves in clusters on the twigs.
//! * Conifers: a tapered trunk with whorls of drooping branches, needles in pads.
//! * Every trunk starts at the LOWEST ground under its footprint and spreads roots.

use std::collections::HashMap;

use crate::voxel::world::wm::*;

#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub enum Species {
    Oak,
    Pine,
    Birch,
    Snag,
    Bush,
    Log,
}

#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub enum PrimKind {
    Capsule,
    Blob,
}

/// One drawing primitive of a plant.
#[derive(Clone, Copy, Debug)]
pub struct PlantPrim {
    pub kind: PrimKind,
    pub mat: u8,
    pub accent: u8,
    /// Capsules: 0 plain, 1 birch lenticels, 2 log (moss on top), 3 bark furrows.
    pub style: u8,
    pub a: [f32; 3],
    pub b: [f32; 3],
    /// Capsule radii / blob radius (ra) and holes (rb).
    pub ra: f32,
    pub rb: f32,
    pub squash: f32,
    pub accent_p: f32,
    pub aux: f32,
    pub seed: u32,
    pub lo: [i32; 3],
    pub hi: [i32; 3],
}

#[derive(Clone, Debug, Default)]
pub struct Plant {
    pub prims: Vec<PlantPrim>,
    pub lo: [i32; 3],
    pub hi: [i32; 3],
    pub crown_radius: f32,
    pub trunk_radius: f32,
    pub base_y: i32,
}

impl Plant {
    pub fn is_empty(&self) -> bool {
        self.prims.is_empty()
    }
}

/// How far (horizontally) any plant reaches from its site, in voxels.
pub const PLANT_REACH: i32 = 120;

#[inline]
fn hash3(x: i32, y: i32, z: i32, seed: u32) -> u32 {
    let mut h = seed ^ 0x2c1b3c6d;
    h ^= (x as u32).wrapping_mul(0x8da6b343);
    h = (h ^ (h >> 15)).wrapping_mul(0x2c1b3c6d);
    h ^= (y as u32).wrapping_mul(0xd8163841);
    h = (h ^ (h >> 13)).wrapping_mul(0x297a2d39);
    h ^= (z as u32).wrapping_mul(0xcb1ab31f);
    h = (h ^ (h >> 16)).wrapping_mul(0x9e3779b1);
    h ^ (h >> 15)
}

#[inline]
fn u01(h: u32) -> f32 {
    (h & 0xffffff) as f32 / 16_777_216.0
}

/// Small deterministic xorshift RNG (per plant).
struct Rng(u32);
impl Rng {
    fn new(seed: u32) -> Self {
        Rng(if seed != 0 { seed } else { 1 })
    }
    fn next(&mut self) -> u32 {
        self.0 ^= self.0 << 13;
        self.0 ^= self.0 >> 17;
        self.0 ^= self.0 << 5;
        self.0
    }
    fn f(&mut self) -> f32 {
        u01(self.next())
    }
    fn range(&mut self, a: f32, b: f32) -> f32 {
        a + (b - a) * self.f()
    }
}

// A minimal 3-vector with exactly the C++ arithmetic (no fused ops: deterministic output).
#[derive(Clone, Copy, Debug, Default)]
struct V3 {
    x: f32,
    y: f32,
    z: f32,
}
const fn v3(x: f32, y: f32, z: f32) -> V3 {
    V3 { x, y, z }
}
impl std::ops::Add for V3 {
    type Output = V3;
    fn add(self, b: V3) -> V3 {
        v3(self.x + b.x, self.y + b.y, self.z + b.z)
    }
}
impl std::ops::Sub for V3 {
    type Output = V3;
    fn sub(self, b: V3) -> V3 {
        v3(self.x - b.x, self.y - b.y, self.z - b.z)
    }
}
impl std::ops::Mul<f32> for V3 {
    type Output = V3;
    fn mul(self, s: f32) -> V3 {
        v3(self.x * s, self.y * s, self.z * s)
    }
}
#[inline]
fn dot(a: V3, b: V3) -> f32 {
    a.x * b.x + a.y * b.y + a.z * b.z
}
#[inline]
fn len(a: V3) -> f32 {
    dot(a, a).sqrt()
}
#[inline]
fn norm(a: V3) -> V3 {
    let l = len(a);
    if l > 1e-6 { a * (1.0 / l) } else { v3(0.0, 1.0, 0.0) }
}

// ---- recording primitives ---------------------------------------------------------------
/// Tapered cylinder with round ends. Radii are at least 0.75 (thin diagonals share faces).
fn capsule(pl: &mut Plant, a: V3, b: V3, ra: f32, rb: f32, mat: u8, style: u8, seed: u32, aux: f32) {
    let ra = ra.max(0.75);
    let rb = rb.max(0.75);
    let rm = ra.max(rb);
    pl.prims.push(PlantPrim {
        kind: PrimKind::Capsule,
        mat,
        accent: 0,
        style,
        a: [a.x, a.y, a.z],
        b: [b.x, b.y, b.z],
        ra,
        rb,
        squash: 1.0,
        accent_p: 0.0,
        aux,
        seed,
        lo: [(a.x.min(b.x) - rm).floor() as i32, (a.y.min(b.y) - rm).floor() as i32, (a.z.min(b.z) - rm).floor() as i32],
        hi: [(a.x.max(b.x) + rm).ceil() as i32, (a.y.max(b.y) + rm).ceil() as i32, (a.z.max(b.z) + rm).ceil() as i32],
    });
}

fn cap(pl: &mut Plant, a: V3, b: V3, ra: f32, rb: f32, mat: u8) {
    capsule(pl, a, b, ra, rb, mat, 0, 0, 0.0);
}

/// Noisy leaf cluster: a roughened sphere (squashed vertically) with holes in its shell.
#[allow(clippy::too_many_arguments)]
fn blob(pl: &mut Plant, c: V3, r: f32, m: u8, seed: u32, holes: f32, squash: f32, accent: u8, accent_p: f32) {
    let ri = (r * 1.15 + 1.0).ceil() as i32;
    let ry = ((r * 1.15 + 1.0) * squash.max(1.0)).ceil() as i32;
    let (cx, cy, cz) = (c.x.floor() as i32, c.y.floor() as i32, c.z.floor() as i32);
    pl.prims.push(PlantPrim {
        kind: PrimKind::Blob,
        mat: m,
        accent,
        style: 0,
        a: [c.x, c.y, c.z],
        b: [0.0; 3],
        ra: r,
        rb: holes,
        squash,
        accent_p,
        aux: 0.0,
        seed,
        lo: [cx - ri, cy - ry, cz - ri],
        hi: [cx + ri, cy + ry, cz + ri],
    });
}

fn ground_min(height: &impl Fn(i32, i32) -> i32, x: i32, z: i32, r: i32) -> i32 {
    let mut g = height(x, z);
    for dz in -r..=r {
        for dx in -r..=r {
            if dx * dx + dz * dz <= r * r {
                g = g.min(height(x + dx, z + dz));
            }
        }
    }
    g
}
fn ground_max(height: &impl Fn(i32, i32) -> i32, x: i32, z: i32, r: i32) -> i32 {
    let mut g = height(x, z);
    for dz in -r..=r {
        for dx in -r..=r {
            if dx * dx + dz * dz <= r * r {
                g = g.max(height(x + dx, z + dz));
            }
        }
    }
    g
}

/// Roots: tapering limbs from the trunk base out over the terrain.
#[allow(clippy::too_many_arguments)]
fn roots(pl: &mut Plant, rng: &mut Rng, cx: f32, cz: f32, base_top: i32, trunk_r: f32, height: &impl Fn(i32, i32) -> i32, mat: u8, s: f32) {
    let n = 3 + (rng.f() * 3.0) as i32 + if s > 2.0 { 1 } else { 0 };
    let a0 = rng.f() * std::f32::consts::TAU;
    let steps = 4 + (s * 1.5) as i32;
    for k in 0..n {
        let a = a0 + k as f32 * std::f32::consts::TAU / n as f32 + rng.range(-0.4, 0.4);
        let l = trunk_r * rng.range(2.0, 3.4) + 2.0;
        let mut prev = v3(cx, base_top as f32 + trunk_r * 0.9, cz);
        let mut r_prev = trunk_r * 0.75;
        for st in 1..=steps {
            let t = st as f32 / steps as f32;
            let px = cx + a.cos() * l * t;
            let pz = cz + a.sin() * l * t;
            let gy = height(px.floor() as i32, pz.floor() as i32) as f32;
            let p = v3(px, gy + 0.6 - 0.8 * t, pz);
            let r = trunk_r * 0.75 * (1.0 - t) * (1.0 - t) + 0.6 * (s * 0.4).max(1.0) * t;
            cap(pl, prev, p, r_prev, r, mat);
            prev = p;
            r_prev = r;
        }
    }
}

// ---- space colonisation -----------------------------------------------------------------
#[derive(Clone, Copy)]
struct Node {
    p: V3,
    parent: i32,
    children: i32,
    r: f32,
}
impl Node {
    fn new(p: V3, parent: i32) -> Self {
        Self { p, parent, children: 0, r: 0.0 }
    }
}

struct Lobe {
    c: V3,
    r: f32,
    squash: f32,
}

struct ScParams {
    attractors: i32,
    influence: f32,
    kill: f32,
    step: f32,
    up: f32,
    iterations: i32,
}

/// Grows a branch graph from the trunk chain in `nodes` toward attractors in the lobes.
fn colonise(nodes: &mut Vec<Node>, lobes: &[Lobe], p: &ScParams, rng: &mut Rng) {
    let mut att: Vec<V3> = Vec::new();
    let (mut lo, mut hi) = (v3(1e9, 1e9, 1e9), v3(-1e9, -1e9, -1e9));
    for l in lobes {
        lo = v3(lo.x.min(l.c.x - l.r), lo.y.min(l.c.y - l.r * l.squash), lo.z.min(l.c.z - l.r));
        hi = v3(hi.x.max(l.c.x + l.r), hi.y.max(l.c.y + l.r * l.squash), hi.z.max(l.c.z + l.r));
    }
    let mut tries = 0;
    while tries < p.attractors * 8 && (att.len() as i32) < p.attractors {
        tries += 1;
        let q = v3(rng.range(lo.x, hi.x), rng.range(lo.y, hi.y), rng.range(lo.z, hi.z));
        for l in lobes {
            let mut d = q - l.c;
            d.y /= l.squash;
            if dot(d, d) < l.r * l.r {
                att.push(q);
                break;
            }
        }
    }
    // Uniform grid over the nodes for the nearest-node queries.
    let cell = p.influence;
    let key = |q: V3| -> i64 {
        let ix = (q.x / cell).floor() as i64;
        let iy = (q.y / cell).floor() as i64;
        let iz = (q.z / cell).floor() as i64;
        (ix & 0xFFFFF) | ((iy & 0xFFFFF) << 20) | ((iz & 0xFFFFF) << 40)
    };
    let mut grid: HashMap<i64, Vec<i32>> = HashMap::new();
    for (i, n) in nodes.iter().enumerate() {
        grid.entry(key(n.p)).or_default().push(i as i32);
    }
    let mut pull: Vec<V3> = Vec::new();
    let mut pull_n: Vec<i32> = Vec::new();
    let mut alive = vec![true; att.len()];
    for _ in 0..p.iterations {
        pull.clear();
        pull.resize(nodes.len(), V3::default());
        pull_n.clear();
        pull_n.resize(nodes.len(), 0);
        let mut any = false;
        for (a, &q) in att.iter().enumerate() {
            if !alive[a] {
                continue;
            }
            let mut best = -1i32;
            let mut best_d = p.influence * p.influence;
            let (cx, cy, cz) = ((q.x / cell).floor() as i32, (q.y / cell).floor() as i32, (q.z / cell).floor() as i32);
            for dz in -1..=1 {
                for dy in -1..=1 {
                    for dx in -1..=1 {
                        let k = key(v3(
                            (cx + dx) as f32 * cell + 0.5 * cell,
                            (cy + dy) as f32 * cell + 0.5 * cell,
                            (cz + dz) as f32 * cell + 0.5 * cell,
                        ));
                        let Some(list) = grid.get(&k) else { continue };
                        for &ni in list {
                            let d = q - nodes[ni as usize].p;
                            let dd = dot(d, d);
                            if dd < best_d {
                                best_d = dd;
                                best = ni;
                            }
                        }
                    }
                }
            }
            if best < 0 {
                continue;
            }
            if best_d < p.kill * p.kill {
                alive[a] = false;
                continue;
            }
            let b = best as usize;
            pull[b] = pull[b] + norm(q - nodes[b].p);
            pull_n[b] += 1;
            any = true;
        }
        if !any {
            break;
        }
        let n0 = nodes.len();
        for i in 0..n0 {
            if pull_n[i] == 0 {
                continue;
            }
            let dir = norm(pull[i] * (1.0 / pull_n[i] as f32) + v3(0.0, p.up, 0.0));
            let np = nodes[i].p + dir * p.step;
            nodes[i].children += 1;
            nodes.push(Node::new(np, i as i32));
            grid.entry(key(np)).or_default().push(nodes.len() as i32 - 1);
        }
    }
}

/// Pipe model radii, scaled so the trunk base has radius trunk_r.
fn pipe_radii(nodes: &mut [Node], trunk_r: f32, tip_r: f32) {
    let e = 2.5f32;
    let mut acc = vec![0.0f32; nodes.len()];
    for i in (0..nodes.len()).rev() {
        let r = if nodes[i].children == 0 { tip_r } else { acc[i].powf(1.0 / e) };
        nodes[i].r = r;
        if nodes[i].parent >= 0 {
            acc[nodes[i].parent as usize] += r.powf(e);
        }
    }
    let s = if nodes.is_empty() { 1.0 } else { trunk_r / nodes[0].r.max(1e-3) };
    for n in nodes.iter_mut() {
        n.r = (tip_r * 0.9).max(n.r * s);
    }
    for n in nodes.iter_mut() {
        if n.children == 0 {
            n.r = n.r.min(tip_r * 1.6);
        }
    }
}

// ---- species -----------------------------------------------------------------------------
fn grow_broadleaf(x: i32, z: i32, height: &impl Fn(i32, i32) -> i32, seed: u32, sp: Species, age: f32) -> Plant {
    let mut pl = Plant::default();
    let mut rng = Rng::new(seed);
    let birch = sp == Species::Birch;
    let snag = sp == Species::Snag;
    let s = age * if birch { 2.7 } else if snag { 2.6 } else { 3.0 };
    let d = (s * 0.6).max(1.0);
    let mut trunk_r = s * if birch {
        rng.range(0.9, 1.25)
    } else if snag {
        rng.range(1.1, 1.7)
    } else {
        rng.range(1.5, 2.3)
    };
    trunk_r = trunk_r.max(1.0);
    let g = ground_min(height, x, z, trunk_r.ceil() as i32 + 1);
    let (cx, cz) = (x as f32 + 0.5, z as f32 + 0.5);
    let trunk_h = s * if birch {
        rng.range(18.0, 28.0)
    } else if snag {
        rng.range(10.0, 18.0)
    } else {
        rng.range(13.0, 20.0)
    };
    let crown_r = s * if birch {
        rng.range(7.0, 10.0)
    } else if snag {
        rng.range(8.0, 12.0)
    } else {
        rng.range(12.0, 18.0)
    };
    let squash = if birch { rng.range(1.7, 2.2) } else { rng.range(0.75, 0.95) };
    // (C++ evaluates the braced initializer's arguments left to right)
    let ccx = cx + rng.range(-2.0, 2.0) * s;
    let ccy = g as f32 + trunk_h + crown_r * squash * 0.85;
    let ccz = cz + rng.range(-2.0, 2.0) * s;
    let crown_c = v3(ccx, ccy, ccz);
    let mut lobes = vec![Lobe { c: crown_c, r: crown_r * 0.8, squash }];
    let nl = if birch {
        2 + (rng.f() * 2.0) as i32
    } else {
        4 + (rng.f() * 3.0) as i32 + if s > 2.5 { 2 } else { 0 }
    };
    for _ in 0..nl {
        let a = rng.f() * std::f32::consts::TAU;
        let rr = crown_r * rng.range(0.35, 0.62);
        let dy = rng.range(-0.3, 0.45) * crown_r * squash;
        let c = crown_c + v3(a.cos() * rr, dy, a.sin() * rr);
        lobes.push(Lobe { c, r: crown_r * rng.range(0.42, 0.68), squash });
    }
    // Trunk: a slightly leaning chain of nodes up to the crown's underside.
    let lx = rng.range(-0.12, 0.12);
    let lz = rng.range(-0.12, 0.12);
    let lean = v3(lx, 1.0, lz);
    let mut nodes = vec![Node::new(v3(cx, g as f32 - 1.5, cz), -1)];
    let g_top = ground_max(height, x, z, trunk_r.ceil() as i32 + 1);
    let mut p = v3(cx, g_top as f32 + 2.0, cz);
    nodes.push(Node::new(p, 0));
    nodes[0].children += 1;
    let crown_bottom = crown_c.y - crown_r * squash * 0.7;
    let node_step = 2.5 * d.min(2.0);
    while p.y < crown_bottom {
        p = p + norm(lean) * node_step;
        let parent = nodes.len() as i32 - 1;
        nodes.push(Node::new(p, parent));
        let n = nodes.len();
        nodes[n - 2].children += 1;
    }
    let params = ScParams {
        attractors: ((if birch { 220.0 } else if snag { 70.0 } else { 360.0 }) * (0.3 * s * s).clamp(0.6, 3.2)) as i32,
        influence: (if birch { 11.0 } else { 15.0 }) * (s * 0.72).max(1.0),
        kill: 3.5 * (s * 0.6).max(1.0),
        step: (if birch { 2.0 } else { 2.4 }) * d,
        up: if birch { 0.25 } else { 0.1 },
        iterations: 60 + (28.0 * s) as i32,
    };
    colonise(&mut nodes, &lobes, &params, &mut rng);
    pipe_radii(&mut nodes, trunk_r, if snag { 0.6 } else { 0.55 });
    let bark = if birch { BIRCH_BARK } else if snag { DEAD_WOOD } else { WOOD };
    let style = if birch { 1 } else if snag { 0 } else { 3 };
    // Snags: some limbs broke off - with everything that grew from them.
    let mut dropped = vec![false; nodes.len()];
    for i in 1..nodes.len() {
        let par = nodes[i].parent as usize;
        dropped[i] = dropped[par] || (snag && nodes[i].r < 0.8 * d && u01(hash3(i as i32, 3, 7, seed)) < 0.45);
        if dropped[i] {
            continue;
        }
        let a = nodes[par];
        let st = if a.r > 2.0 { style } else if birch { 1 } else { 0 };
        capsule(&mut pl, a.p, nodes[i].p, a.r, nodes[i].r, bark, st, seed ^ 0x77, 0.0);
    }
    if !snag {
        let leaf = if birch { BIRCH_LEAVES } else { LEAVES };
        for i in 1..nodes.len() {
            let n = nodes[i];
            if n.children > 0 && n.r > 0.8 {
                continue;
            }
            if n.p.y < crown_bottom - 1.0 {
                continue;
            }
            let r = (if birch { rng.range(1.5, 2.5) } else { rng.range(2.3, 3.6) }) * (s * 0.62).max(1.0);
            blob(
                &mut pl,
                n.p + v3(0.0, 0.6 * d, 0.0),
                r,
                leaf,
                seed.wrapping_add((i as u32).wrapping_mul(7919)),
                if birch { 0.35 } else { 0.3 },
                if birch { 1.1 } else { 0.8 },
                0,
                0.0,
            );
        }
    }
    roots(&mut pl, &mut rng, cx, cz, g, trunk_r, height, bark, s);
    pl.crown_radius = if snag { 0.0 } else { crown_r };
    pl.trunk_radius = trunk_r;
    pl.base_y = g - 2;
    pl
}

fn grow_pine(x: i32, z: i32, height: &impl Fn(i32, i32) -> i32, seed: u32, age: f32) -> Plant {
    let mut pl = Plant::default();
    let mut rng = Rng::new(seed);
    let s = 2.5 * age;
    let sh = 3.9 * age;
    let d = (s * 0.75).max(1.0);
    let trunk_r = (s * rng.range(1.3, 1.9)).max(1.0);
    let g = ground_min(height, x, z, trunk_r.ceil() as i32 + 1);
    let gf = g as f32;
    let (cx, cz) = (x as f32 + 0.5, z as f32 + 0.5);
    let h = sh * rng.range(40.0, 60.0);
    let max_l = h * rng.range(0.17, 0.22);
    capsule(&mut pl, v3(cx, gf - 2.0, cz), v3(cx, gf + h, cz), trunk_r, 0.5 * d, PINE_BARK, 3, seed ^ 0x77, 0.0);
    let mut y = gf + h * rng.range(0.22, 0.32);
    let mut rot = rng.f() * std::f32::consts::TAU;
    let mut whorl = 0i32;
    while y < gf + h - 2.5 * d {
        let h_rel = (y - gf) / h;
        let l = max_l * (1.0 - h_rel).powf(0.85) + 1.5 * d;
        let nb = 5 + (rng.f() * 3.0) as i32 + if s > 2.0 { 1 } else { 0 };
        for b in 0..nb {
            let a = rot + b as f32 * std::f32::consts::TAU / nb as f32 + rng.range(-0.25, 0.25);
            let lb = l * rng.range(0.8, 1.15);
            let st = v3(cx, y, cz);
            let dy = -0.26 - rng.range(0.0, 0.2);
            let dir = norm(v3(a.cos(), dy, a.sin()));
            let e = st + dir * lb;
            cap(&mut pl, st, e, 0.6 * d, 0.45, PINE_BARK);
            let bs = seed.wrapping_add((whorl * 97 + b * 13) as u32);
            let pad_step = 1.6 * d.min(1.9);
            let mut t = pad_step * 0.6;
            while t <= lb {
                let k = t / lb;
                let c = st + dir * t + v3(0.0, -0.6 * d * k * k, 0.0);
                blob(
                    &mut pl,
                    c,
                    (1.1 + 1.4 * (1.0 - k) * (lb / (8.0 * d)).min(1.2) + 0.5) * d.min(1.9),
                    NEEDLES,
                    bs.wrapping_add((t * 10.0) as u32),
                    0.14,
                    0.55,
                    0,
                    0.0,
                );
                t += pad_step;
            }
            // Side shoots on long branches.
            if lb > 10.0 * d * 0.8 {
                let ns = 2 + (rng.f() * 2.0) as i32;
                for k2 in 0..ns {
                    let t0 = lb * rng.range(0.3, 0.8);
                    let side = if k2 & 1 != 0 { 1.0 } else { -1.0 };
                    let aa = a + side * rng.range(0.7, 1.1);
                    let s2 = st + dir * t0;
                    let l2 = (lb - t0) * rng.range(0.4, 0.7) + 1.5 * d;
                    let d2 = norm(v3(aa.cos(), -0.3, aa.sin()));
                    cap(&mut pl, s2, s2 + d2 * l2, 0.5 * d, 0.4, PINE_BARK);
                    let mut t = pad_step;
                    while t <= l2 {
                        blob(
                            &mut pl,
                            s2 + d2 * t + v3(0.0, -0.3 * d, 0.0),
                            (1.2 + 0.6 * (1.0 - t / l2)) * d.min(1.9),
                            NEEDLES,
                            bs.wrapping_add(5000).wrapping_add((k2 * 131) as u32).wrapping_add((t * 10.0) as u32),
                            0.14,
                            0.55,
                            0,
                            0.0,
                        );
                        t += pad_step;
                    }
                }
            }
        }
        rot += 2.39996; // golden angle: whorls don't stack
        y += rng.range(2.6, 3.6) * (d * 1.2).min(2.4);
        whorl += 1;
    }
    // Leader tip.
    for ti in 0..4 {
        let t = ti as f32;
        blob(&mut pl, v3(cx, gf + h - 2.5 * d + t * d, cz), (1.6 - t * 0.3) * d, NEEDLES, seed.wrapping_add(11 + ti as u32), 0.0, 1.0, 0, 0.0);
    }
    roots(&mut pl, &mut rng, cx, cz, g, trunk_r, height, PINE_BARK, s);
    pl.crown_radius = max_l + 1.5 * d;
    pl.trunk_radius = trunk_r;
    pl.base_y = g - 2;
    pl
}

fn grow_bush(x: i32, z: i32, height: &impl Fn(i32, i32) -> i32, seed: u32, age: f32) -> Plant {
    let mut pl = Plant::default();
    let mut rng = Rng::new(seed);
    let s = 2.5 * age;
    let g = height(x, z);
    let (cx, cz) = (x as f32 + 0.5, z as f32 + 0.5);
    let r = s * rng.range(2.5, 4.5);
    let berries = rng.f() < 0.35;
    let n = 4 + (rng.f() * 4.0) as i32;
    for i in 0..n {
        let a = rng.f() * std::f32::consts::TAU;
        let dd = rng.range(0.0, r * 0.6);
        let cy = g as f32 + 1.0 + rng.range(0.4, 1.0) * r * 0.75;
        let c = v3(cx + a.cos() * dd, cy, cz + a.sin() * dd);
        cap(&mut pl, v3(cx, g as f32, cz), c, 0.6 * (s * 0.5).max(1.0), 0.5, WOOD);
        let br = r * rng.range(0.42, 0.66);
        blob(&mut pl, c, br, BUSH_LEAVES, seed.wrapping_add((i as u32).wrapping_mul(31)), 0.22, 0.8, if berries { BERRIES } else { 0 }, 0.06);
    }
    pl.crown_radius = r * 0.8;
    pl.base_y = g;
    pl
}

fn grow_log(x: i32, z: i32, height: &impl Fn(i32, i32) -> i32, seed: u32, age: f32) -> Plant {
    let mut pl = Plant::default();
    let mut rng = Rng::new(seed);
    let s = 2.5 * age;
    let a = rng.f() * std::f32::consts::TAU;
    let l = s * rng.range(9.0, 18.0);
    let r = s * rng.range(1.2, 1.9);
    let (x0, z0) = (x as f32 + 0.5, z as f32 + 0.5);
    let (x1, z1) = (x0 + a.cos() * l, z0 + a.sin() * l);
    let h0 = height(x, z);
    let h1 = height(x1.floor() as i32, z1.floor() as i32);
    let tol = (3.0 * s * 0.9) as i32;
    for st in 1..12 {
        let hs = height((x0 + (x1 - x0) * st as f32 / 12.0).floor() as i32, (z0 + (z1 - z0) * st as f32 / 12.0).floor() as i32);
        if (hs - h0).abs() > tol {
            return pl;
        }
    }
    if (h1 - h0).abs() > tol {
        return pl;
    }
    let pa = v3(x0, h0 as f32 + r * 0.7 + 0.5, z0);
    let pb = v3(x1, h1 as f32 + r * 0.7 + 0.5, z1);
    let cy = (pa.y + pb.y) * 0.5;
    capsule(&mut pl, pa, pb, r, r * 0.85, DEAD_WOOD, 2, seed, cy);
    let ns = 1 + (rng.f() * 3.0) as i32;
    for _ in 0..ns {
        let t = rng.range(0.25, 0.85);
        let c = pa + (pb - pa) * t;
        let sign = if rng.f() < 0.5 { 1.0 } else { -1.0 };
        let aa = a + sign * rng.range(0.6, 1.2);
        let dy = rng.range(0.1, 0.7);
        let dd = norm(v3(aa.cos(), dy, aa.sin()));
        let ext = r + s * rng.range(1.5, 3.5);
        cap(&mut pl, c, c + dd * ext, r * 0.35, r * 0.2, DEAD_WOOD);
    }
    pl.base_y = h0.min(h1);
    pl
}

fn smooth01(e0: f32, e1: f32, x: f32) -> f32 {
    let t = ((x - e0) / (e1 - e0)).clamp(0.0, 1.0);
    t * t * (3.0 - 2.0 * t)
}

fn value_noise(x: f32, y: f32, seed: u32) -> f32 {
    let (xi, yi) = (x.floor() as i32, y.floor() as i32);
    let (fx, fy) = (x - xi as f32, y - yi as f32);
    let (sx, sy) = (fx * fx * (3.0 - 2.0 * fx), fy * fy * (3.0 - 2.0 * fy));
    let v00 = u01(hash3(xi, yi, 0, seed));
    let v10 = u01(hash3(xi + 1, yi, 0, seed));
    let v01 = u01(hash3(xi, yi + 1, 0, seed));
    let v11 = u01(hash3(xi + 1, yi + 1, 0, seed));
    let a = v00 + (v10 - v00) * sx;
    let b = v01 + (v11 - v01) * sx;
    a + (b - a) * sy
}

fn fbm3(x: f32, y: f32, seed: u32) -> f32 {
    0.55 * value_noise(x, y, seed) + 0.3 * value_noise(x * 2.1, y * 2.1, seed + 1) + 0.15 * value_noise(x * 4.3, y * 4.3, seed + 2)
}

/// Grow one plant at column (x, z); deterministic in (species, x, z, seed, age). May come out
/// empty (a log on a slope too steep to lie on).
pub fn grow_plant(s: Species, x: i32, z: i32, height: &impl Fn(i32, i32) -> i32, seed: u32, age: f32) -> Plant {
    let h = hash3(x, s as i32, z, seed);
    let mut p = match s {
        Species::Pine => grow_pine(x, z, height, h, age),
        Species::Bush => grow_bush(x, z, height, h, age),
        Species::Log => grow_log(x, z, height, h, age),
        _ => grow_broadleaf(x, z, height, h, s, age),
    };
    if let Some(first) = p.prims.first() {
        let (mut lo, mut hi) = (first.lo, first.hi);
        for q in &p.prims {
            for a in 0..3 {
                lo[a] = lo[a].min(q.lo[a]);
                hi[a] = hi[a].max(q.hi[a]);
            }
        }
        p.lo = lo;
        p.hi = hi;
    }
    p
}

/// Draw the parts of `pl` inside the box [lo, hi] (inclusive): put(x, y, z, mat, only_air).
pub fn raster_plant(pl: &Plant, lo: [i32; 3], hi: [i32; 3], mut put: impl FnMut(i32, i32, i32, u8, bool)) {
    for a in 0..3 {
        if pl.hi[a] < lo[a] || pl.lo[a] > hi[a] {
            return;
        }
    }
    for p in &pl.prims {
        let (x0, x1) = (p.lo[0].max(lo[0]), p.hi[0].min(hi[0]));
        let (y0, y1) = (p.lo[1].max(lo[1]), p.hi[1].min(hi[1]));
        let (z0, z1) = (p.lo[2].max(lo[2]), p.hi[2].min(hi[2]));
        if x0 > x1 || y0 > y1 || z0 > z1 {
            continue;
        }
        match p.kind {
            PrimKind::Capsule => {
                let pa = v3(p.a[0], p.a[1], p.a[2]);
                let ab = v3(p.b[0], p.b[1], p.b[2]) - pa;
                let ab2 = dot(ab, ab).max(1e-6);
                for z in z0..=z1 {
                    for y in y0..=y1 {
                        for x in x0..=x1 {
                            let q = v3(x as f32 + 0.5, y as f32 + 0.5, z as f32 + 0.5);
                            let t = (dot(q - pa, ab) / ab2).clamp(0.0, 1.0);
                            let d = q - (pa + ab * t);
                            let r = p.ra + (p.rb - p.ra) * t;
                            if dot(d, d) > r * r {
                                continue;
                            }
                            let mut m = p.mat;
                            match p.style {
                                1 => {
                                    // birch: dark lenticel marks, wider than tall
                                    if u01(hash3(x / 2, y / 3, z / 2, p.seed)) < 0.13 {
                                        m = PINE_BARK;
                                    }
                                }
                                2 => {
                                    // log: moss on the upper side
                                    if y as f32 + 0.5 > p.aux + p.ra * 0.35 && u01(hash3(x, y, z, p.seed)) < 0.55 {
                                        m = MOSS;
                                    }
                                }
                                3 => {
                                    // bark furrows: darker vertical streaks
                                    if u01(hash3(x, y / 4, z, p.seed)) < 0.2 {
                                        m = if m == WOOD { PINE_BARK } else { WOOD };
                                    }
                                }
                                _ => {}
                            }
                            put(x, y, z, m, false);
                        }
                    }
                }
            }
            PrimKind::Blob => {
                let c = v3(p.a[0], p.a[1], p.a[2]);
                let r_out2 = p.ra * p.ra * 1.3225;
                let r_in2 = p.ra * p.ra * 0.64;
                let shell2 = p.ra * p.ra * 0.3025;
                let inv_sq = 1.0 / p.squash;
                // Whether the blob draws (x, y, z): a voxel past rIn is drawn only if its
                // neighbour one step toward the centre passes the rough edge too.
                let keep = |mut x: i32, mut y: i32, mut z: i32, h_out: &mut u32| -> bool {
                    for guard in 0..64 {
                        let mut d = v3(x as f32 + 0.5, y as f32 + 0.5, z as f32 + 0.5) - c;
                        d.y *= inv_sq;
                        let dd = dot(d, d);
                        if dd > r_out2 {
                            return false;
                        }
                        let h = hash3(x, y, z, p.seed);
                        if guard == 0 {
                            *h_out = h;
                        }
                        if dd > r_in2 {
                            let rr = p.ra * (0.8 + 0.35 * u01(h));
                            if dd > rr * rr {
                                return false;
                            }
                        }
                        if guard == 0 && dd > shell2 && u01(h >> 7) < p.rb {
                            return false;
                        }
                        if dd <= r_in2 {
                            return true;
                        }
                        let (ax, ay, az) = (d.x.abs(), d.y.abs(), d.z.abs());
                        let mx = ax.max(ay).max(az) * 0.4;
                        x -= if ax > mx { if d.x > 0.0 { 1 } else { -1 } } else { 0 };
                        y -= if ay > mx { if d.y > 0.0 { 1 } else { -1 } } else { 0 };
                        z -= if az > mx { if d.z > 0.0 { 1 } else { -1 } } else { 0 };
                    }
                    true
                };
                for z in z0..=z1 {
                    for y in y0..=y1 {
                        for x in x0..=x1 {
                            let mut h = 0u32;
                            if !keep(x, y, z, &mut h) {
                                continue;
                            }
                            // ... and a hole beside it can still leave one alone: no neighbour, no leaf
                            let mut d = v3(x as f32 + 0.5, y as f32 + 0.5, z as f32 + 0.5) - c;
                            d.y *= inv_sq;
                            if dot(d, d) > shell2 {
                                let mut any = false;
                                let mut hn = 0u32;
                                for k in 0..27 {
                                    if k == 13 {
                                        continue;
                                    }
                                    if keep(x + k % 3 - 1, y + k / 3 % 3 - 1, z + k / 9 - 1, &mut hn) {
                                        any = true;
                                        break;
                                    }
                                }
                                if !any {
                                    continue;
                                }
                            }
                            let mm = if p.accent != 0 && u01(h >> 3) < p.accent_p { p.accent } else { p.mat };
                            put(x, y, z, mm, true);
                        }
                    }
                }
            }
        }
    }
}

/// Vegetation sites: jittered grids with biome rules. `order` is global (the same site gets
/// the same number in any region).
#[derive(Clone, Copy, Debug)]
pub struct VegSite {
    pub x: i32,
    pub z: i32,
    pub s: Species,
    pub age: f32,
    pub order: u64,
}

fn fdiv(a: i32, k: i32) -> i32 {
    if a >= 0 { a / k } else { -((-a + k - 1) / k) }
}

/// Sites whose plants can reach the columns [x0, x1] x [z0, z1] of an nx x nz world
/// (nx = 0: unbounded).
#[allow(clippy::too_many_arguments)]
pub fn vegetation_sites(
    nx: i32,
    nz: i32,
    x0: i32,
    z0: i32,
    x1: i32,
    z1: i32,
    height: &impl Fn(i32, i32) -> i32,
    top: &impl Fn(i32, i32) -> u8,
    water_level: i32,
    seed: u32,
    out: &mut Vec<VegSite>,
) {
    let ok = |x: i32, z: i32, slope_r: i32, max_range: i32| -> bool {
        if nx > 0 && (x < 40 || z < 40 || x >= nx - 40 || z >= nz - 40) {
            return false;
        }
        let h = height(x, z);
        if h <= water_level + 4 {
            return false;
        }
        let t = top(x, z);
        if !matches!(t, GRASS | DRY_GRASS | MOSS | FLOWERS | COARSE_DIRT | TALL_GRASS) {
            return false;
        }
        let (mut lo, mut hi) = (h, h);
        let mut dz = -slope_r;
        while dz <= slope_r {
            let mut dx = -slope_r;
            while dx <= slope_r {
                let hh = height(x + dx, z + dz);
                lo = lo.min(hh);
                hi = hi.max(hh);
                dx += slope_r;
            }
            dz += slope_r;
        }
        hi - lo <= max_range
    };
    let forest = |x: i32, z: i32| fbm3(x as f32 * 0.0016, z as f32 * 0.0016, seed + 7);
    let moist = |x: i32, z: i32| fbm3(x as f32 * 0.0022, z as f32 * 0.0022, seed + 1);
    let cells = |k: i32| -> (i32, i32, i32, i32) {
        let (mut c0x, mut c0z) = (fdiv(x0 - PLANT_REACH, k), fdiv(z0 - PLANT_REACH, k));
        let (mut c1x, mut c1z) = (fdiv(x1 + PLANT_REACH, k), fdiv(z1 + PLANT_REACH, k));
        if nx > 0 {
            c0x = c0x.max(0);
            c0z = c0z.max(0);
            c1x = c1x.min(nx / k - 1);
            c1z = c1z.min(nz / k - 1);
        }
        (c0x, c0z, c1x, c1z)
    };
    // Trees: a 64-voxel grid, density from forest noise.
    const K_TREE: i32 = 64;
    let (c0x, c0z, c1x, c1z) = cells(K_TREE);
    for cz in c0z..=c1z {
        for cx in c0x..=c1x {
            let h = hash3(cx, 1, cz, seed);
            let x = cx * K_TREE + 6 + (u01(h) * (K_TREE - 12) as f32) as i32;
            let z = cz * K_TREE + 6 + (u01(h >> 8) * (K_TREE - 12) as f32) as i32;
            let fo = forest(x, z);
            let mo = moist(x, z);
            let p = 0.12 + 0.78 * smooth01(0.36, 0.6, fo);
            if u01(hash3(cx, 2, cz, seed)) >= p {
                continue;
            }
            if !ok(x, z, 8, 14) {
                continue;
            }
            let alt = height(x, z) - water_level;
            let w_pine = 0.15 + 1.6 * smooth01(110.0, 230.0, alt as f32) + 0.6 * smooth01(0.5, 0.3, mo);
            let w_birch = 0.5 * smooth01(0.45, 0.62, mo);
            let w_oak = 1.0;
            let w_snag = 0.07;
            let r = u01(hash3(cx, 3, cz, seed)) * (w_pine + w_birch + w_oak + w_snag);
            let s = if r < w_pine {
                Species::Pine
            } else if r < w_pine + w_birch {
                Species::Birch
            } else if r < w_pine + w_birch + w_oak {
                Species::Oak
            } else {
                Species::Snag
            };
            let ua = u01(hash3(cx, 4, cz, seed));
            let sapling = 0.16 + 0.24 * (1.0 - smooth01(0.4, 0.62, fo));
            let age = if ua < sapling {
                0.35 + 0.3 * (ua / sapling)
            } else if ua > 0.95 && s != Species::Snag {
                1.3
            } else {
                0.85 + 0.25 * u01(hash3(cx, 10, cz, seed))
            };
            out.push(VegSite { x, z, s, age, order: ((cz as u64) << 30) | (cx as u64) });
        }
    }
    // Bushes: a 30-voxel grid, in forests and on the moist side.
    const K_BUSH: i32 = 30;
    let (c0x, c0z, c1x, c1z) = cells(K_BUSH);
    for cz in c0z..=c1z {
        for cx in c0x..=c1x {
            let h = hash3(cx, 5, cz, seed);
            let x = cx * K_BUSH + 3 + (u01(h) * (K_BUSH - 6) as f32) as i32;
            let z = cz * K_BUSH + 3 + (u01(h >> 8) * (K_BUSH - 6) as f32) as i32;
            let p = (0.06 + 0.3 * smooth01(0.38, 0.62, forest(x, z)) * smooth01(0.28, 0.55, moist(x, z))) * 1.0f32.min(0.4 + 1.0);
            if u01(hash3(cx, 6, cz, seed)) >= p || !ok(x, z, 4, 6) {
                continue;
            }
            let age = 0.6 + 0.55 * u01(hash3(cx, 11, cz, seed));
            out.push(VegSite { x, z, s: Species::Bush, age, order: (1u64 << 60) | ((cz as u64) << 30) | (cx as u64) });
        }
    }
    // Fallen logs: a 150-voxel grid, in forests.
    const K_LOG: i32 = 150;
    let (c0x, c0z, c1x, c1z) = cells(K_LOG);
    for cz in c0z..=c1z {
        for cx in c0x..=c1x {
            let h = hash3(cx, 8, cz, seed);
            let x = cx * K_LOG + 12 + (u01(h) * (K_LOG - 24) as f32) as i32;
            let z = cz * K_LOG + 12 + (u01(h >> 8) * (K_LOG - 24) as f32) as i32;
            if u01(hash3(cx, 9, cz, seed)) >= 0.55 * smooth01(0.4, 0.62, forest(x, z)) * 1.0 || !ok(x, z, 8, 7) {
                continue;
            }
            let age = 0.75 + 0.4 * u01(hash3(cx, 12, cz, seed));
            out.push(VegSite { x, z, s: Species::Log, age, order: (2u64 << 60) | ((cz as u64) << 30) | (cx as u64) });
        }
    }
}

/// The chance a tree grows around (x, z) (forest noise; the far map's forest tint).
pub fn forest_cover(x: i32, z: i32, seed: u32) -> f32 {
    0.12 + 0.78 * smooth01(0.36, 0.6, fbm3(x as f32 * 0.0016, z as f32 * 0.0016, seed + 7))
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn plants_are_deterministic_and_rooted() {
        let height = |x: i32, z: i32| 100 + ((x + z) / 50);
        for s in [Species::Oak, Species::Pine, Species::Birch, Species::Snag, Species::Bush, Species::Log] {
            let a = grow_plant(s, 500, 700, &height, 1337, 1.0);
            let b = grow_plant(s, 500, 700, &height, 1337, 1.0);
            assert_eq!(a.prims.len(), b.prims.len());
            if s != Species::Log {
                assert!(!a.is_empty(), "{s:?}");
            }
            if !a.is_empty() {
                assert!(a.lo[1] <= a.base_y + 2 && a.hi[1] > a.base_y, "{s:?}");
                let mut n = 0;
                raster_plant(&a, a.lo, a.hi, |_, _, _, _, _| n += 1);
                assert!(n > 10, "{s:?} drew {n} voxels");
            }
        }
    }
}
