//! Emissive voxels as light sources (lamps in the world, glowing bodies).
//!
//! The renderer samples a short list of lights directly (next-event estimation). A light is a
//! box around its emissive voxels with an emitted power (radiance x visible area):
//!   * world: one per 8^3 brick holding emissive voxels (scanned after generation,
//!     rescanned where edits / streaming changed bricks), touching clusters merged;
//!   * bodies: one per emissive body, in its local grid, moved by its pose.

use std::collections::HashMap;

use bytemuck::{Pod, Zeroable};

use crate::math::Vec3;
use crate::objects::VoxObject;
use crate::voxel::sparse_volume::{SparseVolume, RING_CHUNK_B};
use crate::voxel::world::{wm, Material, World};

/// std140: three vec4s per light (common.glsl _LightsU).
#[repr(C)]
#[derive(Clone, Copy, Debug, Default, Pod, Zeroable)]
pub struct GpuLight {
    pub pos: [f32; 3],
    pub radius: f32,
    pub power: [f32; 3],
    /// Body index, or -1 = world.
    pub owner: f32,
    pub half: [f32; 3],
    /// The renderer's lamp-cache slot (-1 = none).
    pub pad: f32,
}

/// A torch for torchfx.comp: a = (x, head y, z, stick cells below the head),
/// b = (bracket y, bracket side 0 N / 1 E / 2 S / 3 W or -1, flame phase, -).
#[repr(C)]
#[derive(Clone, Copy, Debug, Default, Pod, Zeroable)]
pub struct GpuTorch {
    pub a: [f32; 4],
    pub b: [f32; 4],
}

pub fn light_hash(mut x: u32) -> u32 {
    x ^= x >> 16;
    x = x.wrapping_mul(0x7feb_352d);
    x ^= x >> 15;
    x = x.wrapping_mul(0x846c_a68b);
    x ^= x >> 16;
    x
}

/// Smooth 1D value noise in [0, 1] (lightNoise in common.glsl).
fn light_noise(t: f32, ph: u32) -> f32 {
    let i = t.floor();
    let mut f = t - i;
    f = f * f * (3.0 - 2.0 * f);
    let a = light_hash((i as i32 as u32) ^ ph);
    let b = light_hash(((i as i32).wrapping_add(1) as u32) ^ ph);
    let fa = (a & 0xFFFF) as f32 / 65535.0;
    let fb = (b & 0xFFFF) as f32 / 65535.0;
    fa + (fb - fa) * f
}

/// Emissive light animation (same hash as common.glsl -> same flicker): kind 0 steady,
/// 1 fire block, 2 pulse, 3 torch; damaged bodies (integrity < 0.9) sputter.
pub fn light_anim(kind: f32, ph: u32, t: f32, integrity: f32) -> f32 {
    let mut a = 1.0;
    if kind > 0.5 && kind < 1.5 {
        a = 0.55 + 0.45 * (0.6 * light_noise(t * 8.0, ph) + 0.4 * light_noise(t * 19.0, ph.wrapping_mul(7).wrapping_add(3)));
    } else if kind > 2.5 {
        a = 1.0
            + 0.28
                * (0.5 * light_noise(t * 0.4, ph) + 0.3 * light_noise(t * 1.3, ph.wrapping_mul(5).wrapping_add(1))
                    + 0.2 * light_noise(t * 2.6, ph.wrapping_mul(7).wrapping_add(3))
                    - 0.5);
    } else if kind > 1.5 {
        a = 0.55 + 0.45 * (0.5 + 0.5 * (t * 2.2 + (ph & 1023) as f32 * 0.00614).sin());
    }
    if integrity < 0.9 {
        if integrity < 0.25 {
            return 0.0;
        }
        let d = (0.9 - integrity) / 0.65;
        let n = light_noise(t * (5.0 + 16.0 * d), ph ^ 0x9e37);
        a *= (if n < 0.2 + 0.45 * d { 0.1 } else { 1.0 }) * integrity * integrity;
    }
    a
}

/// Phase of a world light: hash of the 16-voxel cell it is in.
pub fn light_phase_at(p: Vec3) -> u32 {
    let x = ((p.x / 16.0).floor() as i32 & 4095) as u32;
    let y = (p.y / 16.0).floor() as i32 as u32;
    let z = ((p.z / 16.0).floor() as i32 & 4095) as u32;
    light_hash(x.wrapping_mul(73_856_093) ^ y.wrapping_mul(19_349_663) ^ z.wrapping_mul(83_492_791))
}

pub fn light_phase_body(i: i32) -> u32 {
    light_hash((i as u32).wrapping_add(0x51ed))
}

pub fn light_phase_torch(x: i32, y: i32, z: i32) -> u32 {
    light_hash(((x & 0xFFFF) as u32).wrapping_mul(73_856_093) ^ (y as u32).wrapping_mul(19_349_663) ^ ((z & 0xFFFF) as u32).wrapping_mul(83_492_791))
        & 0xFF_FFFF
}

pub const MIN_EMISSIVE: f32 = 0.5;
pub const MAX_LIGHTS: usize = 32;
pub const MAX_TORCHES: usize = 64;

/// Emitters of one brick / body, local coords.
#[derive(Clone, Debug)]
pub struct Stats {
    pub lo: Vec3,
    pub hi: Vec3,
    pub power: Vec3,
    pub count: i32,
    pub anim: f32,
    pub flames: Vec<[i32; 3]>,
}

impl Default for Stats {
    fn default() -> Self {
        Self { lo: Vec3::splat(1e30), hi: Vec3::splat(-1e30), power: Vec3::ZERO, count: 0, anim: 0.0, flames: Vec::new() }
    }
}

impl Stats {
    fn grow(&mut self, lo: Vec3, hi: Vec3) {
        self.lo = self.lo.min(lo);
        self.hi = self.hi.max(hi);
    }
}

fn scan_brick(v: &SparseVolume, pal: &[Material], gi: usize) -> Stats {
    let mut s = Stats::default();
    let g = v.grid[gi];
    if g == 0 || (g < 256 && pal[g as usize].emissive < MIN_EMISSIVE) {
        return s;
    }
    let (bx, by, bz) = v.brick_coords(gi);
    for lz in 0..8 {
        for ly in 0..8 {
            for lx in 0..8 {
                let (x, y, z) = (bx * 8 + lx, by * 8 + ly, bz * 8 + lz);
                let m = if g < 256 { g as u8 } else { v.bricks[(g - 256) as usize].vox[(lz * 64 + ly * 8 + lx) as usize] };
                if m == 0 || pal[m as usize].emissive < MIN_EMISSIVE {
                    continue;
                }
                let mut faces = 0;
                for (dx, dy, dz) in [(1, 0, 0), (-1, 0, 0), (0, 1, 0), (0, -1, 0), (0, 0, 1), (0, 0, -1)] {
                    let (nx, ny, nz) = (x + dx, y + dy, z + dz);
                    let outside = if v.ring_w != 0 {
                        ny < 0 || ny >= v.ny
                    } else {
                        nx < 0 || ny < 0 || nz < 0 || nx >= v.nx || ny >= v.ny || nz >= v.nz
                    };
                    let nb = if outside { 0 } else { v.get_voxel(nx, ny, nz) };
                    if nb == 0 || pal[nb as usize].props[2] == 1.0 {
                        faces += 1; // air or water: it shines out
                    }
                }
                if faces == 0 {
                    continue;
                }
                let mm = &pal[m as usize];
                if m == wm::TORCH_FLAME {
                    s.flames.push([x, y, z]);
                }
                let l = mm.emissive * 0.25 * faces as f32;
                s.power += Vec3::new(mm.r, mm.g, mm.b) * l;
                s.anim = s.anim.max(mm.props[3]);
                let (xf, yf, zf) = (x as f32, y as f32, z as f32);
                if m == wm::TORCH_FLAME {
                    s.grow(Vec3::new(xf - 0.4, yf, zf - 0.4), Vec3::new(xf + 1.4, yf + 1.3, zf + 1.4));
                } else {
                    s.grow(Vec3::new(xf, yf, zf), Vec3::new(xf + 1.0, yf + 1.0, zf + 1.0));
                }
                s.count += 1;
            }
        }
    }
    s
}

fn has_emissive(w: &World, gi: usize) -> bool {
    let g = w.grid[gi];
    if g == 0 {
        return false;
    }
    if g < 256 {
        return w.palette[g as usize].emissive >= MIN_EMISSIVE;
    }
    w.bricks[(g - 256) as usize].vox.iter().any(|&m| m != 0 && w.palette[m as usize].emissive >= MIN_EMISSIVE)
}

#[derive(Default)]
struct BodyCache {
    mass: f32,
    s: Stats,
}

#[derive(Default)]
pub struct LightRegistry {
    world: HashMap<u32, Stats>,
    merged: Vec<Stats>,
    merged_valid: bool,
    bodies: Vec<BodyCache>,
}

impl LightRegistry {
    pub fn scan_world(&mut self, w: &World) {
        self.world.clear();
        self.merged_valid = false;
        // (cheap rejection first: most cells hold no emitter)
        use rayon::prelude::*;
        let found: Vec<(u32, Stats)> = (0..w.grid.len())
            .into_par_iter()
            .filter(|&gi| has_emissive(w, gi))
            .map(|gi| (gi as u32, scan_brick(w, &w.palette, gi)))
            .filter(|(_, s)| s.count > 0)
            .collect();
        self.world.extend(found);
    }

    /// These world grid cells changed: rescan them and their neighbours (exposed faces).
    pub fn rescan_bricks(&mut self, w: &World, cells: &[u32]) {
        let mut todo = Vec::new();
        for &gi in cells {
            let (bx, by, bz) = w.brick_coords(gi as usize);
            for dz in -1..=1 {
                for dy in -1..=1 {
                    for dx in -1..=1 {
                        let (x, y, z) = (bx + dx, by + dy, bz + dz);
                        if y < 0 || y >= w.bricks_y {
                            continue;
                        }
                        if w.ring_w == 0 && (x < 0 || z < 0 || x >= w.bricks_x || z >= w.bricks_z) {
                            continue;
                        }
                        todo.push(w.grid_index(x, y, z) as u32);
                    }
                }
            }
        }
        self.merged_valid = false;
        todo.sort_unstable();
        todo.dedup();
        for gi in todo {
            let s = if has_emissive(w, gi as usize) { scan_brick(w, &w.palette, gi as usize) } else { Stats::default() };
            if s.count > 0 {
                self.world.insert(gi, s);
            } else {
                self.world.remove(&gi);
            }
        }
    }

    /// A streamed chunk's slot changed (loaded / evicted): rescan its cells.
    pub fn rescan_chunk(&mut self, w: &World, cx: i32, cz: i32) {
        self.merged_valid = false;
        let k = RING_CHUNK_B;
        for bz in 0..k {
            for by in 0..w.bricks_y {
                for bx in 0..k {
                    let gi = w.grid_index(cx * k + bx, by, cz * k + bz);
                    let s = if has_emissive(w, gi) { scan_brick(w, &w.palette, gi) } else { Stats::default() };
                    if s.count > 0 {
                        self.world.insert(gi as u32, s);
                    } else {
                        self.world.remove(&(gi as u32));
                    }
                }
            }
        }
    }

    pub fn world_light_count(&self) -> usize {
        self.world.len()
    }

    /// Neighbouring brick clusters (one lamp spanning several bricks) merge into one light.
    fn merge_world(&mut self) {
        self.merged_valid = true;
        // (sorted by key: the same lights merge the same way every time)
        let mut keys: Vec<&u32> = self.world.keys().collect();
        keys.sort_unstable();
        let mut merged: Vec<Stats> = keys.into_iter().map(|k| self.world[k].clone()).collect();
        let mut again = true;
        while again {
            again = false;
            let mut a = 0;
            while a < merged.len() {
                let mut b = a + 1;
                while b < merged.len() {
                    let gap = 1.5;
                    let (al, ah, bl, bh) = (merged[a].lo, merged[a].hi, merged[b].lo, merged[b].hi);
                    let apart = bl.x > ah.x + gap || al.x > bh.x + gap || bl.y > ah.y + gap || al.y > bh.y + gap || bl.z > ah.z + gap || al.z > bh.z + gap;
                    if apart {
                        b += 1;
                        continue;
                    }
                    let bs = merged.swap_remove(b);
                    let am = &mut merged[a];
                    am.grow(bs.lo, bs.hi);
                    am.anim = am.anim.max(bs.anim);
                    am.power += bs.power;
                    am.count += bs.count;
                    am.flames.extend(bs.flames);
                    again = true;
                }
                a += 1;
            }
        }
        self.merged = merged;
    }

    /// The MAX_LIGHTS brightest-by-distance lights near `eye` (world + bodies).
    pub fn gather(&mut self, w: &World, objects: &[VoxObject], eye: Vec3, time: f32) -> Vec<GpuLight> {
        let mut c: Vec<(f32, GpuLight)> = Vec::new();
        let mut add = |p: Vec3, half: Vec3, power: Vec3, owner: f32| {
            let d = p - eye;
            let lum = 0.2126 * power.x + 0.7152 * power.y + 0.0722 * power.z;
            let score = lum / d.dot(d).max(1.0);
            c.push((
                score,
                GpuLight { pos: p.to_array(), radius: half.length(), power: power.to_array(), owner, half: half.to_array(), pad: 0.0 },
            ));
        };
        if !self.merged_valid {
            self.merge_world();
        }
        for s in &self.merged {
            let ctr = (s.lo + s.hi) * 0.5;
            let ph = if s.anim > 2.5 && !s.flames.is_empty() {
                light_phase_torch(s.flames[0][0], s.flames[0][1], s.flames[0][2])
            } else {
                light_phase_at(ctr)
            };
            let a = if s.anim > 0.0 { light_anim(s.anim, ph, time, 1.0) } else { 1.0 };
            add(ctr, (s.hi - s.lo) * 0.5, s.power * a, -1.0);
        }
        self.bodies.resize_with(objects.len(), BodyCache::default);
        for (i, o) in objects.iter().enumerate() {
            let bc = &mut self.bodies[i];
            if bc.mass != o.mass {
                bc.mass = o.mass;
                bc.s = Stats::default();
                for gi in 0..o.vol.grid.len() {
                    if o.vol.grid[gi] == 0 {
                        continue;
                    }
                    let s = scan_brick(&o.vol, &w.palette, gi);
                    if s.count == 0 {
                        continue;
                    }
                    bc.s.power += s.power;
                    bc.s.anim = bc.s.anim.max(s.anim);
                    bc.s.count += s.count;
                    bc.s.grow(s.lo, s.hi);
                }
            }
            if bc.s.count == 0 {
                continue;
            }
            let lc = (bc.s.lo + bc.s.hi) * 0.5;
            let lh = (bc.s.hi - bc.s.lo) * 0.5;
            let r = o.pose.r.to_cols_array();
            let half = Vec3::new(
                r[0].abs() * lh.x + r[3].abs() * lh.y + r[6].abs() * lh.z,
                r[1].abs() * lh.x + r[4].abs() * lh.y + r[7].abs() * lh.z,
                r[2].abs() * lh.x + r[5].abs() * lh.y + r[8].abs() * lh.z,
            );
            let a = light_anim(bc.s.anim, light_phase_body(i as i32), time, o.glow_integrity);
            if a <= 0.0 {
                continue;
            }
            add(o.pose.apply(lc), half, bc.s.power * a, i as f32);
        }
        c.sort_by(|a, b| b.0.total_cmp(&a.0));
        c.into_iter().take(MAX_LIGHTS).map(|(_, l)| l).collect()
    }

    /// Floating origin: world coordinates moved by (dx, dz) voxels.
    pub fn shift(&mut self, dx: f32, dz: f32) {
        for s in self.world.values_mut() {
            s.lo.x -= dx;
            s.hi.x -= dx;
            s.lo.z -= dz;
            s.hi.z -= dz;
            for f in &mut s.flames {
                f[0] -= dx as i32;
                f[2] -= dz as i32;
            }
        }
        self.merged_valid = false;
    }

    /// The MAX_TORCHES torches nearest `eye`.
    pub fn torches(&self, w: &World, eye: Vec3) -> Vec<GpuTorch> {
        let mut c: Vec<(f32, [i32; 3])> = Vec::new();
        for s in self.world.values() {
            for f in &s.flames {
                let d = Vec3::new(f[0] as f32 + 0.5, f[1] as f32 + 0.5, f[2] as f32 + 0.5) - eye;
                c.push((d.dot(d), *f));
            }
        }
        if c.len() > MAX_TORCHES {
            c.select_nth_unstable_by(MAX_TORCHES, |a, b| a.0.total_cmp(&b.0));
            c.truncate(MAX_TORCHES);
        }
        c.sort_by(|a, b| a.0.total_cmp(&b.0));
        let mut out = Vec::new();
        for (_, f) in c {
            let (x, yh, z) = (f[0], f[1] - 1, f[2]);
            if w.get_voxel(x, yh, z) != wm::TORCH_HEAD {
                continue;
            }
            let (mut n, mut by, mut bd) = (0, -1, -1);
            while n < 8 {
                let m = w.get_voxel(x, yh - n - 1, z);
                if !(wm::TORCH_STICK..=wm::TORCH_BRACKET_W).contains(&m) {
                    break;
                }
                if m != wm::TORCH_STICK {
                    by = yh - n - 1;
                    bd = (m - wm::TORCH_BRACKET_N) as i32;
                }
                n += 1;
            }
            out.push(GpuTorch {
                a: [x as f32, yh as f32, z as f32, n as f32],
                b: [by as f32, bd as f32, light_phase_torch(f[0], f[1], f[2]) as f32, 0.0],
            });
        }
        out
    }
}
