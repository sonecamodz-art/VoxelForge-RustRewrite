//! Rigid-body physics: bodies are voxel objects. Collision samples each body's surface-shell
//! voxels against the static world grid (and the other bodies' volumes); contacts are solved
//! with warm-started sequential impulses (accumulated, with two-axis Coulomb friction) under
//! substepping. Felled trees hang on a hinge over their stump until they tip.

use std::collections::HashMap;
use std::time::Instant;

use rayon::prelude::*;

use crate::math::{Mat3, Quat, Transform, Vec3};
use crate::objects::VoxObject;
use crate::voxel::sparse_volume::SparseVolume;
use crate::voxel::world::{wm, World};

const GRAVITY: f32 = -98.0; // voxels/s^2 (~9.8 m/s^2 at 10 cm voxels)
const SUBSTEPS: usize = 4;
const SOLVER_ITERS: usize = 8;
const FRICTION: f32 = 0.7;
/// Velocity damping per second (time-based, so a toppling tree keeps its spin).
const LIN_DAMPING: f32 = 0.05;
const ANG_DAMPING: f32 = 0.1;
/// cos 70 deg: the trunk's up axis this far over, the hinge breaks.
const HINGE_RELEASE: f32 = 0.34;
/// sin 8 deg: a hinged tree leaning less than this toward its fall ...
const FELL_LEAN: f32 = 0.14;
/// ... keeps tipping at least this fast (rad/s).
const FELL_SPIN: f32 = 0.15;
/// s: a tree held up all that time (by a wall, say) is let go of.
const HINGE_LIFE: f32 = 6.0;
/// Baumgarte position-correction factor, allowed penetration, cap on the separation velocity.
const BETA: f32 = 0.2;
const SLOP: f32 = 0.3;
const MAX_BIAS_VEL: f32 = 5.0;
/// World contacts per body after manifold reduction / raw hits collected before it.
const MAX_CONTACTS_PER_BODY: usize = 32;
const MAX_RAW_CONTACTS: usize = 4096;
/// Below these (voxels/s, rad/s) for 0.5 s a body sleeps.
const SLEEP_LIN_VEL: f32 = 2.0;
const SLEEP_ANG_VEL: f32 = 0.15;

/// Timing and counts of the last step (F3 overlay / bench).
#[derive(Clone, Copy, Debug, Default)]
pub struct PhysicsStats {
    pub world_ms: f32,
    pub pairs_ms: f32,
    pub solve_ms: f32,
    pub integrate_ms: f32,
    pub total_ms: f32,
    pub awake: usize,
    pub contacts: usize,
    pub pairs_tested: usize,
    pub pairs_overlapping: usize,
    pub shell_samples: usize,
}

/// Contact between body `a` and body `b` (None = the static world); `n` points from B toward A.
#[derive(Clone, Copy, Default)]
struct Contact {
    a: usize,
    b: Option<usize>,
    /// Contact point relative to each body's centre of mass (world).
    ra: Vec3,
    rb: Vec3,
    n: Vec3,
    pen: f32,
    mass_n: f32,
    acc_n: f32,
    /// Fixed friction tangents (Box2D-style two-axis friction).
    t1: Vec3,
    t2: Vec3,
    mass_t1: f32,
    mass_t2: f32,
    acc_t1: f32,
    acc_t2: f32,
    /// (bodies, voxel cell): matches the same contact next substep.
    key: u64,
}

#[derive(Clone, Copy)]
struct Warm {
    n: f32,
    t1: f32,
    t2: f32,
}

#[derive(Clone, Copy)]
struct BodyCache {
    xcom: Vec3,
    inv_iw: Mat3,
    /// 0 while asleep: a sleeping body is immovable in contacts.
    inv_m: f32,
}

/// The simulation state that outlives a step: warm-start impulses and the last stats.
#[derive(Default)]
pub struct Physics {
    /// Contacts are regenerated every substep; their solved impulses carry over by key and
    /// apply before the iterations (Catto). Without it a resting stack never converges.
    warm: HashMap<u64, Warm>,
    pub stats: PhysicsStats,
    /// Falling bodies pass through the world's foliage and thin branches unless this is set.
    pub foliage_solid: bool,
}

fn contact_key(a: usize, b: Option<usize>, x: i32, y: i32, z: i32) -> u64 {
    let bb = b.map_or(0u64, |b| b as u64 + 1) as u32 as u64;
    let mut k = (a as u32 as u64).wrapping_mul(0x9E37_79B9_7F4A_7C15)
        ^ bb.wrapping_mul(0xC2B2_AE3D_27D4_EB4F)
        ^ (x as u32 as u64 | (y as u32 as u64) << 21 | (z as u32 as u64) << 42);
    k ^= k >> 31;
    k = k.wrapping_mul(0xBF58_476D_1CE4_E5B9);
    k ^= k >> 29;
    k
}

/// Keep a well-spread subset of `v[begin..]`: the deepest contact, then greedy farthest-point
/// sampling over the contact points (deterministic). The first k hits in shell order supported
/// a resting crate along one edge only: it rocked and never slept.
fn reduce_contacts(v: &mut Vec<Contact>, begin: usize, k: usize) {
    let n = v.len() - begin;
    if n <= k {
        return;
    }
    let input: Vec<Contact> = v.drain(begin..).collect();
    let mut dmin = vec![1e30f32; n];
    let mut used = vec![false; n];
    let mut pick = 0;
    for i in 1..n {
        if input[i].pen > input[pick].pen {
            pick = i;
        }
    }
    for _ in 0..k {
        used[pick] = true;
        v.push(input[pick]);
        let (mut next, mut best) = (0, -1.0f32);
        for i in 0..n {
            if used[i] {
                continue;
            }
            let d = input[i].ra - input[pick].ra;
            dmin[i] = dmin[i].min(d.dot(d));
            if dmin[i] > best {
                best = dmin[i];
                next = i;
            }
        }
        if best < 0.0 {
            break;
        }
        pick = next;
    }
}

/// The open side(s) of a solid voxel: the contact normal (local coords of `v`). `toward` (the
/// body's centre of mass relative to the voxel's centre) picks the side where a thin feature is
/// open on both: a 1-voxel wall's faces would cancel and bodies slid through it.
fn volume_normal_at(v: &SparseVolume, x: i32, y: i32, z: i32, skip_water: bool, toward: Vec3) -> Vec3 {
    let open = |dx: i32, dy: i32, dz: i32| {
        let m = v.get_voxel(x + dx, y + dy, z + dz);
        m == 0 || (skip_water && m == wm::WATER)
    };
    let tw = toward.to_array();
    let mut n = [0.0f32; 3];
    for a in 0..3 {
        let e = [(a == 0) as i32, (a == 1) as i32, (a == 2) as i32];
        let pos = open(e[0], e[1], e[2]);
        let neg = open(-e[0], -e[1], -e[2]);
        n[a] = if pos && neg {
            if tw[a] > 0.0 {
                1.0
            } else if tw[a] < 0.0 {
                -1.0
            } else {
                0.0
            }
        } else if pos {
            1.0
        } else if neg {
            -1.0
        } else {
            0.0
        };
    }
    let r = Vec3::from_array(n);
    let l = r.length();
    if l > 1e-4 { r / l } else { Vec3::Y }
}

/// The world voxels bodies collide with: not water, and (unless `foliage_solid`) not foliage.
/// A felled tree also slips through other trees' branches - world wood without wood 2 voxels
/// away on all four horizontal sides; trunks' cores still stop it.
fn blocks_body(w: &World, x: i32, y: i32, z: i32, m: u8, tree: bool, foliage_solid: bool) -> bool {
    if m == 0 || m == wm::WATER {
        return false;
    }
    if foliage_solid || !wm::is_plant(m) {
        return true;
    }
    if wm::is_foliage_voxel(m) {
        return false;
    }
    if !tree {
        return true;
    }
    let wood_at = |ax: i32, az: i32| {
        let n = w.get_voxel(ax, y, az);
        wm::is_plant(n) && !wm::is_foliage_voxel(n)
    };
    wood_at(x + 2, z) && wood_at(x - 2, z) && wood_at(x, z + 2) && wood_at(x, z - 2)
}

#[inline]
fn unpack(pk: u32) -> (i32, i32, i32) {
    ((pk & 1023) as i32, ((pk >> 10) & 1023) as i32, (pk >> 20) as i32)
}

#[inline]
fn floor3(p: Vec3) -> (i32, i32, i32) {
    (p.x.floor() as i32, p.y.floor() as i32, p.z.floor() as i32)
}

/// q advanced by angular velocity w over dt (first order, renormalised).
pub fn q_integrate(q: Quat, w: Vec3, dt: f32) -> Quat {
    let dq = Quat::from_xyzw(w.x * 0.5 * dt, w.y * 0.5 * dt, w.z * 0.5 * dt, 0.0) * q;
    Quat::from_xyzw(q.x + dq.x, q.y + dq.y, q.z + dq.z, q.w + dq.w).normalize()
}

/// Wake bodies whose AABB is within `r` of `pos` (a huge r wakes everything).
pub fn wake_near(objects: &mut [VoxObject], pos: Vec3, r: f32) {
    for o in objects {
        let (mn, mx) = o.world_aabb();
        if pos.cmpgt(mn - r).all() && pos.cmplt(mx + r).all() {
            o.asleep = false;
            o.sleep_timer = 0.0;
        }
    }
}

/// Every brick under the body's AABB (+1 voxel) is empty (or water): it's in free fall.
fn near_world(world: &World, o: &VoxObject) -> bool {
    let (mn, mx) = o.world_aabb();
    let b0 = ((mn - 1.0) / 8.0).floor();
    let b1 = ((mx + 1.0) / 8.0).floor();
    for bz in b0.z as i32..=b1.z as i32 {
        for by in b0.y as i32..=b1.y as i32 {
            if by < 0 || by >= world.bricks_y {
                continue;
            }
            for bx in b0.x as i32..=b1.x as i32 {
                let loaded = if world.ring_w != 0 {
                    world.chunk_loaded(bx >> 4, bz >> 4)
                } else {
                    bx >= 0 && bz >= 0 && bx < world.bricks_x && bz < world.bricks_z
                };
                if !loaded {
                    continue;
                }
                let g = world.grid[world.grid_index(bx, by, bz)];
                if g != 0 && g != wm::WATER as u32 {
                    return true;
                }
            }
        }
    }
    false
}

/// Body `i` against the static world: its contacts (reduced) and the shell samples taken.
fn world_contacts(world: &World, o: &VoxObject, i: usize, c: &BodyCache, foliage_solid: bool) -> (Vec<Contact>, usize) {
    let mut out = Vec::new();
    if !near_world(world, o) {
        return (out, 0);
    }
    let r = o.pose.r;
    // Sample the shell densely: thin world features (1-2 voxel trunks) slip between sparse
    // samples and the body clips through.
    let stride = (o.shell.len() / 16384).max(1);
    let mut samples = 0;
    for &pk in o.shell.iter().step_by(stride) {
        samples += 1;
        let (lx, ly, lz) = unpack(pk);
        let lp = Vec3::new(lx as f32 + 0.5, ly as f32 + 0.5, lz as f32 + 0.5);
        let wp = r * lp + o.pose.t;
        let (gx, gy, gz) = floor3(wp);
        let m = world.get_voxel(gx, gy, gz);
        if !blocks_body(world, gx, gy, gz, m, o.tree, foliage_solid) {
            continue;
        }
        // A body's foliage (a felled crown) meets only the ground, rock and buildings - not
        // other trees' trunks it was wrapped around (they would launch it).
        if !foliage_solid && wm::is_plant(m) && wm::is_foliage_voxel(o.vol.get_voxel(lx, ly, lz)) {
            continue;
        }
        let to_com = c.xcom - Vec3::new(gx as f32 + 0.5, gy as f32 + 0.5, gz as f32 + 0.5);
        let mut n = volume_normal_at(world, gx, gy, gz, true, to_com);
        // A felled tree meets only the buried core of other trunks (no open face: the normal
        // fell back to "up" and the tree climbed its neighbours): push it off sideways.
        if o.tree && wm::is_plant(m) {
            let hz = Vec3::new(to_com.x, 0.0, to_com.z);
            let l = hz.length();
            n = if l > 1e-3 { hz / l } else { Vec3::X };
        }
        let (dx, dy, dz) = floor3(wp + n);
        let deeper = blocks_body(world, dx, dy, dz, world.get_voxel(dx, dy, dz), o.tree, foliage_solid);
        out.push(Contact {
            a: i,
            b: None,
            ra: wp - c.xcom,
            n,
            pen: if deeper { 1.0 } else { 0.15 },
            key: contact_key(i, None, gx, gy, gz),
            ..Default::default()
        });
        if out.len() >= MAX_RAW_CONTACTS {
            break;
        }
    }
    reduce_contacts(&mut out, 0, MAX_CONTACTS_PER_BODY);
    (out, samples)
}

/// The smaller body's shell against the bigger body's volume.
fn pair_contacts(objects: &[VoxObject], cache: &[BodyCache], i: usize, j: usize) -> (Vec<Contact>, usize, bool) {
    let (ia, ib) = if objects[i].shell.len() <= objects[j].shell.len() { (i, j) } else { (j, i) };
    let (a, b) = (&objects[ia], &objects[ib]);
    let b_inv = b.pose.inverse();
    // A-local -> B-local in one transform; world position only on a contact.
    let rab = b_inv.r * a.pose.r;
    let tab = b_inv.r * a.pose.t + b_inv.t;
    // B's box in A-local space (+1 margin): shell points outside it can't touch B.
    let (mut lmn, mut lmx) = (Vec3::splat(1e30), Vec3::splat(-1e30));
    let rba = rab.transpose();
    let tba = -(rba * tab);
    for k in 0..8 {
        let cb = Vec3::new(
            if k & 1 != 0 { b.vol.nx as f32 } else { 0.0 },
            if k & 2 != 0 { b.vol.ny as f32 } else { 0.0 },
            if k & 4 != 0 { b.vol.nz as f32 } else { 0.0 },
        );
        let ca = rba * cb + tba;
        lmn = lmn.min(ca);
        lmx = lmx.max(ca);
    }
    lmn -= 1.0;
    lmx += 1.0;
    let a_com_in_b = b_inv.r * cache[ia].xcom + b_inv.t;
    let mut out = Vec::new();
    let mut samples = 0;
    let stride = (a.shell.len() / 1024).max(1);
    for &pk in a.shell.iter().step_by(stride) {
        samples += 1;
        let (lx, ly, lz) = unpack(pk);
        let lp = Vec3::new(lx as f32 + 0.5, ly as f32 + 0.5, lz as f32 + 0.5);
        if lp.cmplt(lmn).any() || lp.cmpgt(lmx).any() {
            continue;
        }
        let bp = rab * lp + tab;
        let (bx, by, bz) = floor3(bp);
        if b.vol.get_voxel(bx, by, bz) == 0 {
            continue;
        }
        let wp = a.pose.r * lp + a.pose.t;
        let toward = a_com_in_b - Vec3::new(bx as f32 + 0.5, by as f32 + 0.5, bz as f32 + 0.5);
        let n = (b.pose.r * volume_normal_at(&b.vol, bx, by, bz, false, toward)).normalize();
        let (dx, dy, dz) = floor3(bp + b_inv.r * n);
        out.push(Contact {
            a: ia,
            b: Some(ib),
            ra: wp - cache[ia].xcom,
            rb: wp - cache[ib].xcom,
            n,
            pen: if b.vol.get_voxel(dx, dy, dz) != 0 { 1.0 } else { 0.15 },
            key: contact_key(ia, Some(ib), bx, by, bz),
            ..Default::default()
        });
        if out.len() >= MAX_RAW_CONTACTS {
            break;
        }
    }
    let found = !out.is_empty();
    reduce_contacts(&mut out, 0, MAX_CONTACTS_PER_BODY / 2);
    (out, samples, found && (a.asleep || b.asleep))
}

#[inline]
fn apply_impulse(objects: &mut [VoxObject], cache: &[BodyCache], c: &Contact, p: Vec3) {
    let a = &mut objects[c.a];
    a.vel += p * cache[c.a].inv_m;
    a.ang_vel += cache[c.a].inv_iw * c.ra.cross(p);
    if let Some(b) = c.b {
        let o = &mut objects[b];
        o.vel -= p * cache[b].inv_m;
        o.ang_vel += cache[b].inv_iw * c.rb.cross(-p);
    }
}

/// A felled tree's point joint: x / z both ways, y only pushing up (the stump under the butt).
struct Hinge {
    i: usize,
    r: Vec3,
    err: Vec3,
    k: [f32; 3],
    acc: [f32; 3],
}

impl Physics {
    /// Advance every body by `dt` (clamped to 1/30 s, substepped). The world is read-only.
    pub fn step(&mut self, world: &World, objects: &mut [VoxObject], dt: f32) {
        let dt = dt.min(1.0 / 30.0);
        let h = dt / SUBSTEPS as f32;
        let nb = objects.len();
        let foliage_solid = self.foliage_solid;
        let t_start = Instant::now();
        let mut st = PhysicsStats {
            awake: objects.iter().filter(|o| !o.asleep && o.inv_mass != 0.0).count(),
            ..Default::default()
        };
        let lap = |t: &mut Instant| {
            let n = Instant::now();
            let ms = (n - *t).as_secs_f32() * 1e3;
            *t = n;
            ms
        };
        // Endless worlds: a body over ground that isn't loaded is held where it is (it would
        // fall through the missing chunk) until the ground is back.
        let frozen: Vec<bool> = objects
            .iter()
            .map(|o| {
                if world.ring_w == 0 {
                    return false;
                }
                let (mn, mx) = o.world_aabb();
                (0..4).any(|c| {
                    let x = if c & 1 != 0 { mx.x + 8.0 } else { mn.x - 8.0 }.floor() as i32;
                    let z = if c & 2 != 0 { mx.z + 8.0 } else { mn.z - 8.0 }.floor() as i32;
                    !world.loaded_at(x, z)
                })
            })
            .collect();
        let mut cache = vec![BodyCache { xcom: Vec3::ZERO, inv_iw: Mat3::ZERO, inv_m: 0.0 }; nb];

        for _ in 0..SUBSTEPS {
            for (i, o) in objects.iter_mut().enumerate() {
                if o.inv_mass == 0.0 {
                    continue;
                }
                let still = o.asleep || frozen[i];
                if !still {
                    o.vel.y += GRAVITY * h;
                    if o.hinged && (o.pose.r * Vec3::Y).dot(o.fall_dir) < FELL_LEAN {
                        let axis = Vec3::Y.cross(o.fall_dir);
                        let w = o.ang_vel.dot(axis);
                        if w < FELL_SPIN {
                            o.ang_vel += axis * (FELL_SPIN - w);
                        }
                    }
                }
                cache[i] = BodyCache {
                    xcom: o.pose.r * o.com + o.pose.t,
                    inv_iw: if still { Mat3::ZERO } else { o.pose.r * o.inv_inertia_local * o.pose.r.transpose() },
                    inv_m: if still { 0.0 } else { o.inv_mass },
                };
            }
            let mut t_lap = Instant::now();

            // Body vs world, parallel over bodies; merged in body order -> deterministic.
            let per_body: Vec<(Vec<Contact>, usize)> = {
                let objs: &[VoxObject] = objects;
                (0..nb)
                    .into_par_iter()
                    .map(|i| {
                        let o = &objs[i];
                        if o.asleep || frozen[i] || o.inv_mass == 0.0 {
                            return (Vec::new(), 0);
                        }
                        world_contacts(world, o, i, &cache[i], foliage_solid)
                    })
                    .collect()
            };
            let mut contacts: Vec<Contact> = Vec::new();
            for (c, s) in per_body {
                contacts.extend(c);
                st.shell_samples += s;
            }
            st.world_ms += lap(&mut t_lap);

            // Body vs body. Broadphase: sort-and-sweep on x over the world AABBs.
            let mut bmn = vec![Vec3::ZERO; nb];
            let mut bmx = vec![Vec3::ZERO; nb];
            let mut order = Vec::with_capacity(nb);
            for (i, o) in objects.iter().enumerate() {
                if o.inv_mass == 0.0 {
                    continue;
                }
                (bmn[i], bmx[i]) = o.world_aabb();
                order.push(i);
            }
            order.sort_by(|&a, &b| bmn[a].x.total_cmp(&bmn[b].x).then(a.cmp(&b)));
            const MARGIN: f32 = 1.0;
            let mut pairs = Vec::new();
            for (oi, &i0) in order.iter().enumerate() {
                for &j0 in &order[oi + 1..] {
                    if bmn[j0].x >= bmx[i0].x + MARGIN {
                        break; // sorted: no later overlaps
                    }
                    let (i, j) = (i0.min(j0), i0.max(j0));
                    if objects[i].asleep && objects[j].asleep {
                        continue;
                    }
                    st.pairs_tested += 1;
                    let overlap = bmn[i].y < bmx[j].y + MARGIN
                        && bmn[j].y < bmx[i].y + MARGIN
                        && bmn[i].z < bmx[j].z + MARGIN
                        && bmn[j].z < bmx[i].z + MARGIN;
                    if overlap {
                        pairs.push((i, j));
                    }
                }
            }
            st.pairs_overlapping += pairs.len();
            let per_pair: Vec<(Vec<Contact>, usize, bool, usize, usize)> = {
                let objs: &[VoxObject] = objects;
                let cache = &cache;
                pairs
                    .par_iter()
                    .map(|&(i, j)| {
                        let (c, s, wake) = pair_contacts(objs, cache, i, j);
                        (c, s, wake, i, j)
                    })
                    .collect()
            };
            for (c, s, wake, i, j) in per_pair {
                contacts.extend(c);
                st.shell_samples += s;
                if !wake {
                    continue;
                }
                // A contact wakes a sleeper only if the other body is actually moving:
                // "touching an awake body" alone woke resting stacks one body at a time.
                let moving = |o: &VoxObject| !o.asleep && (o.vel.length() > SLEEP_LIN_VEL || o.ang_vel.length() > SLEEP_ANG_VEL);
                if !moving(&objects[i]) && !moving(&objects[j]) {
                    continue;
                }
                for k in [i, j] {
                    objects[k].asleep = false;
                    objects[k].sleep_timer = 0.0;
                }
            }
            st.pairs_ms += lap(&mut t_lap);
            st.contacts += contacts.len();

            // Effective masses along the normal and two fixed tangents.
            let eff = |c: &Contact, d: Vec3| {
                let mut m = cache[c.a].inv_m + d.dot((cache[c.a].inv_iw * c.ra.cross(d)).cross(c.ra));
                if let Some(b) = c.b {
                    m += cache[b].inv_m + d.dot((cache[b].inv_iw * c.rb.cross(d)).cross(c.rb));
                }
                if m > 1e-9 { m } else { 1e9 }
            };
            for c in &mut contacts {
                c.mass_n = eff(c, c.n);
                let ax = if c.n.x.abs() < 0.57 { Vec3::X } else { Vec3::Y };
                c.t1 = c.n.cross(ax).normalize();
                c.t2 = c.n.cross(c.t1);
                c.mass_t1 = eff(c, c.t1);
                c.mass_t2 = eff(c, c.t2);
            }

            let mut hinges: Vec<Hinge> = Vec::new();
            for (i, o) in objects.iter().enumerate() {
                if !o.hinged || cache[i].inv_m == 0.0 {
                    continue;
                }
                let p = o.pose.r * o.hinge_local + o.pose.t;
                let r = p - cache[i].xcom;
                let k = [Vec3::X, Vec3::Y, Vec3::Z].map(|ax| {
                    let k = cache[i].inv_m + ax.dot((cache[i].inv_iw * r.cross(ax)).cross(r));
                    if k > 1e-9 { k } else { 1e9 }
                });
                hinges.push(Hinge { i, r, err: p - o.hinge_anchor, k, acc: [0.0; 3] });
            }

            // Warm start: last substep's impulses for contacts that persist.
            for c in &mut contacts {
                let Some(w) = self.warm.get(&c.key) else { continue };
                c.acc_n = w.n;
                c.acc_t1 = w.t1;
                c.acc_t2 = w.t2;
                let p = c.n * c.acc_n + c.t1 * c.acc_t1 + c.t2 * c.acc_t2;
                apply_impulse(objects, &cache, &*c, p);
            }

            // Sequential impulses with accumulation + friction.
            let rel_vel = |objects: &[VoxObject], c: &Contact| {
                let a = &objects[c.a];
                let va = a.vel + a.ang_vel.cross(c.ra);
                let vb = c.b.map_or(Vec3::ZERO, |b| objects[b].vel + objects[b].ang_vel.cross(c.rb));
                va - vb
            };
            for _ in 0..SOLVER_ITERS {
                for c in &mut contacts {
                    let bias = (BETA / h * (c.pen - SLOP).max(0.0)).min(MAX_BIAS_VEL);
                    let vn = rel_vel(objects, c).dot(c.n);
                    let new_acc = (c.acc_n + (bias - vn) / c.mass_n).max(0.0);
                    let lambda = new_acc - c.acc_n;
                    c.acc_n = new_acc;
                    apply_impulse(objects, &cache, &*c, c.n * lambda);

                    // Friction: two accumulated tangent impulses, the total clamped to the
                    // Coulomb cone |acc_t| <= mu * acc_n.
                    let vrel = rel_vel(objects, c);
                    let mut a1 = c.acc_t1 - vrel.dot(c.t1) / c.mass_t1;
                    let mut a2 = c.acc_t2 - vrel.dot(c.t2) / c.mass_t2;
                    let max_f = FRICTION * c.acc_n;
                    let al = (a1 * a1 + a2 * a2).sqrt();
                    if al > max_f {
                        let sc = max_f / al;
                        a1 *= sc;
                        a2 *= sc;
                    }
                    let pt = c.t1 * (a1 - c.acc_t1) + c.t2 * (a2 - c.acc_t2);
                    c.acc_t1 = a1;
                    c.acc_t2 = a2;
                    apply_impulse(objects, &cache, &*c, pt);
                }
                for g in &mut hinges {
                    let o = &mut objects[g.i];
                    let e = g.err.to_array();
                    for (a, ax) in [Vec3::X, Vec3::Y, Vec3::Z].into_iter().enumerate() {
                        if a == 1 && e[1] >= 0.0 {
                            continue; // above the stump: free to drop onto it
                        }
                        let v = (o.vel + o.ang_vel.cross(g.r)).dot(ax);
                        let bias = (-BETA / h * e[a]).clamp(-MAX_BIAS_VEL, MAX_BIAS_VEL);
                        let mut lambda = (bias - v) / g.k[a];
                        if a == 1 {
                            let acc = (g.acc[1] + lambda).max(0.0);
                            lambda = acc - g.acc[1];
                        }
                        g.acc[a] += lambda;
                        let p = ax * lambda;
                        o.vel += p * cache[g.i].inv_m;
                        o.ang_vel += cache[g.i].inv_iw * g.r.cross(p);
                    }
                }
            }
            self.warm.clear();
            self.warm.extend(contacts.iter().map(|c| (c.key, Warm { n: c.acc_n, t1: c.acc_t1, t2: c.acc_t2 })));
            st.solve_ms += lap(&mut t_lap);

            // Integrate awake bodies.
            for (i, o) in objects.iter_mut().enumerate() {
                if o.asleep || frozen[i] || o.inv_mass == 0.0 {
                    continue;
                }
                o.vel /= 1.0 + h * LIN_DAMPING;
                o.ang_vel /= 1.0 + h * ANG_DAMPING;
                let xcom = cache[i].xcom + o.vel * h;
                o.rot_q = q_integrate(o.rot_q, o.ang_vel, h);
                o.pose.r = Mat3::from_quat(o.rot_q);
                o.pose.t = xcom - o.pose.r * o.com;
                cache[i].xcom = xcom;
            }
            st.integrate_ms += lap(&mut t_lap);
        }

        // Sleep bookkeeping.
        for (i, o) in objects.iter_mut().enumerate() {
            if o.asleep || frozen[i] || o.inv_mass == 0.0 {
                continue;
            }
            if o.hinged {
                o.hinge_age += dt;
                if o.hinge_age > HINGE_LIFE || (o.pose.r * Vec3::Y).y < HINGE_RELEASE {
                    o.hinged = false;
                }
            }
            if o.hinged && (o.pose.r * Vec3::Y).dot(o.fall_dir) < FELL_LEAN {
                o.sleep_timer = 0.0; // still tipping
                continue;
            }
            if o.vel.length() < SLEEP_LIN_VEL && o.ang_vel.length() < SLEEP_ANG_VEL {
                o.sleep_timer += dt;
                if o.sleep_timer > 0.5 {
                    o.asleep = true;
                    o.hinged = false;
                    o.vel = Vec3::ZERO;
                    o.ang_vel = Vec3::ZERO;
                }
            } else {
                o.sleep_timer = 0.0;
            }
            if o.pose.t.y < -200.0 {
                o.asleep = true;
                o.vel = Vec3::ZERO;
                o.ang_vel = Vec3::ZERO;
            }
        }
        st.total_ms = t_start.elapsed().as_secs_f32() * 1e3;
        self.stats = st;
    }
}

/// Pose origin + rotation of a body at a physics step (COM-agnostic: edits move the COM).
#[derive(Clone, Copy)]
pub struct BodyState {
    pub t: Vec3,
    pub q: Quat,
}

impl BodyState {
    pub fn of(o: &VoxObject) -> Self {
        Self { t: o.pose.t, q: o.rot_q }
    }
}

/// The pose between the previous step `p` and the body's current one at `a` in [0, 1]: the
/// centre of mass and rotation blend (a spinning body stays on its path), then the origin.
pub fn interpolated_pose(o: &VoxObject, p: &BodyState, a: f32) -> Transform {
    let com_p = Mat3::from_quat(p.q) * o.com + p.t;
    let com_c = o.pose.r * o.com + o.pose.t;
    let q = if p.q.dot(o.rot_q) < 0.0 { -o.rot_q } else { o.rot_q };
    let r = Mat3::from_quat(p.q.lerp(q, a).normalize());
    let com = com_p + (com_c - com_p) * a;
    Transform { r, t: com - r * o.com }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::objects::{make_crate, make_sphere_object};
    use crate::voxel::world::{FlatLayer, FlatWorld};

    fn flat() -> World {
        let mut w = World::new(256, 128, 256);
        w.generate_flat(&FlatWorld { layers: vec![FlatLayer { mat: wm::STONE, thickness: 0 }], height: 32, water: 0 });
        w
    }

    #[test]
    fn crate_falls_lands_and_sleeps() {
        let w = flat();
        let mut c = make_crate(16);
        c.pose.t = Vec3::new(120.0, 60.0, 120.0);
        c.init_body();
        let mut objs = vec![c];
        let mut ph = Physics::default();
        let mut slept_at = None;
        for f in 0..600 {
            ph.step(&w, &mut objs, 1.0 / 60.0);
            if objs[0].asleep {
                slept_at = Some(f);
                break;
            }
        }
        let o = &objs[0];
        assert!(slept_at.is_some(), "never slept: y {} v {:?}", o.pose.t.y, o.vel);
        // Resting on the floor (top at y 32), upright, inside the slop.
        assert!((o.pose.t.y - 32.0).abs() < 1.5, "rests at y {}", o.pose.t.y);
        assert!((o.pose.r * Vec3::Y).y > 0.99);
    }

    #[test]
    fn spheres_collide_and_stack() {
        let w = flat();
        let mut a = make_sphere_object(8, wm::STONE);
        a.pose.t = Vec3::new(120.0, 34.0, 120.0);
        a.init_body();
        let mut b = make_sphere_object(8, wm::STONE);
        b.pose.t = Vec3::new(120.5, 60.0, 120.0);
        b.init_body();
        let mut objs = vec![a, b];
        let mut ph = Physics::default();
        for _ in 0..240 {
            ph.step(&w, &mut objs, 1.0 / 60.0);
        }
        // Radius 8 on a floor at y 32: the lower centre ~40, the upper one ~16 above it (or
        // rolled off it onto the floor - never through it).
        let com = |o: &VoxObject| o.pose.r * o.com + o.pose.t;
        let (ca, cb) = (com(&objs[0]), com(&objs[1]));
        assert!((ca.y - 40.0).abs() < 1.5, "lower sphere at {}", ca.y);
        assert!(cb.y > 38.5 && (cb - ca).length() > 15.0, "spheres overlap: {ca} {cb}");
    }

    #[test]
    fn wake_near_wakes_only_close_bodies() {
        let mut a = make_crate(8);
        a.asleep = true;
        let mut b = make_crate(8);
        b.pose.t = Vec3::splat(500.0);
        b.asleep = true;
        let mut objs = vec![a, b];
        wake_near(&mut objs, Vec3::splat(4.0), 2.0);
        assert!(!objs[0].asleep && objs[1].asleep);
    }

    #[test]
    fn quaternion_integration_matches_axis_angle() {
        let q = (0..1000).fold(Quat::IDENTITY, |q, _| q_integrate(q, Vec3::new(0.0, 1.0, 0.0), 0.001));
        assert!(q.angle_between(Quat::from_rotation_y(1.0)) < 1e-3);
    }
}
