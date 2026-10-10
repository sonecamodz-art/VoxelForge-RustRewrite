//! The first-person player (1 voxel = 10 cm): an axis-aligned box that walks, jumps, steps up
//! small voxel ledges, crouches (with an edge guard), swims, pushes through foliage, stands on
//! and pushes physics bodies, and - in creative - flies (double-tap Space; flight still
//! collides). Noclip (no collision) is the editor's free camera.
//!
//! Collision is Minecraft-style: the box is swept one axis at a time (Y, X, Z) in sub-steps of
//! under a voxel against the voxel grid, so it never tunnels and never snags on voxel steps.
//! Ground movement is Quake-style (acceleration toward the wished velocity, friction, limited
//! air control) with coyote time and jump buffering.

use crate::math::{Transform, Vec3};
use crate::objects::VoxObject;
use crate::voxel::world::{wm, World};

const EPS: f32 = 1e-3;
/// Integration step.
const SUB_DT: f32 = 1.0 / 120.0;
/// Per collision sub-step: never skips a voxel.
const MAX_AXIS_MOVE: f32 = 0.9;

/// Tunables in voxels / seconds.
#[derive(Clone, Copy, Debug)]
pub struct PlayerTuning {
    pub height: f32,
    pub crouch_height: f32,
    pub width: f32,
    pub eye: f32,
    pub crouch_eye: f32,
    pub gravity: f32,
    pub jump_apex: f32,
    pub terminal: f32,
    pub walk: f32,
    pub sprint: f32,
    pub crouch: f32,
    pub swim: f32,
    pub fly: f32,
    /// Auto step-up / ground snap.
    pub step: f32,
    pub ground_accel: f32,
    pub air_accel: f32,
    /// Steepest walkable ground (degrees); ~5 more slides you.
    pub slope: f32,
}

impl Default for PlayerTuning {
    fn default() -> Self {
        Self {
            height: 18.0,
            crouch_height: 15.0,
            width: 6.0,
            eye: 16.2,
            crouch_eye: 13.2,
            gravity: 200.0,
            jump_apex: 12.0,
            terminal: 600.0,
            walk: 21.5,
            sprint: 50.0,
            crouch: 9.7,
            swim: 22.0,
            fly: 80.0,
            step: 4.0,
            ground_accel: 12.0,
            air_accel: 2.0,
            slope: 45.0,
        }
    }
}

#[derive(Clone, Copy, Debug, Default)]
pub struct PlayerInput {
    /// -1..1 (W/S, D/A).
    pub fwd: f32,
    pub side: f32,
    pub jump_held: bool,
    pub jump_pressed: bool,
    /// Ctrl: crouch (walking), down (flying / swimming).
    pub crouch: bool,
    pub sprint: bool,
}

#[derive(Clone, Copy, Debug)]
pub struct Player {
    /// Bottom centre of the box.
    pub feet: Vec3,
    pub vel: Vec3,
    pub grounded: bool,
    pub flying: bool,
    pub crouched: bool,
    pub in_water: bool,
    pub head_wet: bool,
    /// Current eye height (eases between standing / crouched).
    pub eye_h: f32,
    /// Step / landing smoothing (eases back to 0).
    pub eye_offset: f32,
    pub step_rate: f32,
    pub since_step: f32,
    pub coyote: f32,
    pub jump_buf: f32,
    pub last_space_tap: f64,
    pub bob_phase: f32,
    pub bob_amp: f32,
    /// Share of the box in leaves / bushes (slows you).
    pub foliage: f32,
    /// Standing on this body (None = the world / nothing).
    pub ground_body: Option<usize>,
    pub last_land_speed: f32,
    /// On ground too steep to stand on: sliding down it.
    pub sliding: bool,
}

impl Default for Player {
    fn default() -> Self {
        Self {
            feet: Vec3::ZERO,
            vel: Vec3::ZERO,
            grounded: false,
            flying: false,
            crouched: false,
            in_water: false,
            head_wet: false,
            eye_h: 16.2,
            eye_offset: 0.0,
            step_rate: 0.0,
            since_step: 1.0,
            coyote: 0.0,
            jump_buf: 0.0,
            last_space_tap: -10.0,
            bob_phase: 0.0,
            bob_amp: 0.0,
            foliage: 0.0,
            ground_body: None,
            last_land_speed: 0.0,
            sliding: false,
        }
    }
}

/// The world the player moves in.
pub struct PlayerWorld<'a> {
    pub world: &'a World,
    pub objects: Option<&'a mut Vec<VoxObject>>,
    /// Physics running: walking into a body pushes it.
    pub bodies_dynamic: bool,
}

/// Does this material stop the player? (air, water, foliage, flames don't)
pub fn player_blocks(world: &World, mat: u8) -> bool {
    if mat == 0 || mat == wm::WATER || mat == wm::TORCH || wm::is_torch_part(mat) {
        return false;
    }
    world.palette[mat as usize].props[2] != 3.0 // foliage: pushed through
}

#[derive(Clone, Copy)]
struct PBox {
    lo: [f32; 3],
    hi: [f32; 3],
}

fn box_at(f: Vec3, hw: f32, h: f32) -> PBox {
    PBox { lo: [f.x - hw, f.y, f.z - hw], hi: [f.x + hw, f.y + h, f.z + hw] }
}

/// Solid cells around the player: the world, plus bodies rasterised into world cells over a
/// small region (a body voxel counts where a world cell's centre falls inside it).
struct Collide<'a> {
    world: &'a World,
    lo: [i32; 3],
    dims: [i32; 3],
    /// -1 none, else the body index (< 127).
    body: Vec<i8>,
}

impl<'a> Collide<'a> {
    fn new(world: &'a World) -> Self {
        Self { world, lo: [0; 3], dims: [0; 3], body: Vec::new() }
    }

    fn raster_bodies(&mut self, objects: Option<&[VoxObject]>, b: &PBox, margin: f32) {
        self.dims = [0; 3];
        self.body.clear();
        let Some(objects) = objects else { return };
        let mut rlo = [0i32; 3];
        let mut rhi = [0i32; 3];
        for a in 0..3 {
            rlo[a] = (b.lo[a] - margin).floor() as i32;
            rhi[a] = (b.hi[a] + margin).floor() as i32;
        }
        let mut any = false;
        for (i, o) in objects.iter().enumerate().take(127) {
            let (mn, mx) = o.world_aabb();
            if mx.x < rlo[0] as f32
                || mn.x > (rhi[0] + 1) as f32
                || mx.y < rlo[1] as f32
                || mn.y > (rhi[1] + 1) as f32
                || mx.z < rlo[2] as f32
                || mn.z > (rhi[2] + 1) as f32
            {
                continue;
            }
            if !any {
                any = true;
                for a in 0..3 {
                    self.lo[a] = rlo[a];
                    self.dims[a] = rhi[a] - rlo[a] + 1;
                }
                self.body = vec![-1; (self.dims[0] * self.dims[1] * self.dims[2]) as usize];
            }
            let inv: Transform = o.pose.inverse();
            let (x0, x1) = (rlo[0].max(mn.x.floor() as i32), rhi[0].min(mx.x.floor() as i32));
            let (y0, y1) = (rlo[1].max(mn.y.floor() as i32), rhi[1].min(mx.y.floor() as i32));
            let (z0, z1) = (rlo[2].max(mn.z.floor() as i32), rhi[2].min(mx.z.floor() as i32));
            for z in z0..=z1 {
                for y in y0..=y1 {
                    for x in x0..=x1 {
                        let l = inv.apply(Vec3::new(x as f32 + 0.5, y as f32 + 0.5, z as f32 + 0.5));
                        if o.vol.get_voxel(l.x.floor() as i32, l.y.floor() as i32, l.z.floor() as i32) == 0 {
                            continue;
                        }
                        let idx = (((z - self.lo[2]) * self.dims[1] + (y - self.lo[1])) * self.dims[0] + (x - self.lo[0])) as usize;
                        self.body[idx] = i as i8;
                    }
                }
            }
        }
    }

    /// 0 free, 1 world, 2 + body index.
    fn cell(&self, x: i32, y: i32, z: i32) -> i32 {
        if y < 0 {
            return 1; // the floor
        }
        let w = self.world;
        let outside = if w.ring_w != 0 { !w.loaded_at(x, z) } else { x < 0 || z < 0 || x >= w.nx || z >= w.nz };
        if outside {
            return 1;
        }
        if y < w.ny && player_blocks(w, w.get_voxel(x, y, z)) {
            return 1;
        }
        if self.dims[0] != 0 {
            let (lx, ly, lz) = (x - self.lo[0], y - self.lo[1], z - self.lo[2]);
            if lx >= 0 && ly >= 0 && lz >= 0 && lx < self.dims[0] && ly < self.dims[1] && lz < self.dims[2] {
                let b = self.body[((lz * self.dims[1] + ly) * self.dims[0] + lx) as usize];
                if b >= 0 {
                    return 2 + b as i32;
                }
            }
        }
        0
    }

    /// First solid cell the box overlaps (0 = none; else as cell()).
    fn overlaps(&self, b: &PBox) -> i32 {
        let (x0, x1) = ((b.lo[0] + EPS).floor() as i32, (b.hi[0] - EPS).floor() as i32);
        let (y0, y1) = ((b.lo[1] + EPS).floor() as i32, (b.hi[1] - EPS).floor() as i32);
        let (z0, z1) = ((b.lo[2] + EPS).floor() as i32, (b.hi[2] - EPS).floor() as i32);
        for y in y0..=y1 {
            for z in z0..=z1 {
                for x in x0..=x1 {
                    let c = self.cell(x, y, z);
                    if c != 0 {
                        return c;
                    }
                }
            }
        }
        0
    }
}

#[inline]
fn axis(v: &mut Vec3, a: usize) -> &mut f32 {
    match a {
        0 => &mut v.x,
        1 => &mut v.y,
        _ => &mut v.z,
    }
}

/// Moves the feet along one axis, stopping flush against the first solid cell. Returns what
/// blocked it (0 = nothing).
fn move_axis(c: &Collide, feet: &mut Vec3, a: usize, d: f32, hw: f32, h: f32) -> i32 {
    let (lo_off, hi_off) = if a == 1 { (0.0, h) } else { (-hw, hw) };
    let mut rem = d;
    while rem.abs() > 1e-6 {
        let s = rem.clamp(-MAX_AXIS_MOVE, MAX_AXIS_MOVE);
        let mut np = *feet;
        *axis(&mut np, a) += s;
        let b = box_at(np, hw, h);
        let hit = c.overlaps(&b);
        if hit == 0 {
            *feet = np;
            rem -= s;
            continue;
        }
        // Flush against the layer it entered.
        let old = *axis(feet, a);
        let mut at = if s > 0.0 { (b.hi[a] - EPS).floor() - hi_off } else { (b.lo[a] + EPS).floor() + 1.0 - lo_off };
        at = if s > 0.0 { at.clamp(old, old + s) } else { at.clamp(old + s, old) };
        let mut fp = *feet;
        *axis(&mut fp, a) = at;
        if c.overlaps(&box_at(fp, hw, h)) == 0 {
            *feet = fp;
        }
        return hit;
    }
    0
}

/// Share of the box's cells that are water / foliage.
fn sample_media(w: &World, b: &PBox) -> (f32, f32) {
    let (mut n, mut nw, mut nf) = (0, 0, 0);
    let (x0, x1) = ((b.lo[0] + EPS).floor() as i32, (b.hi[0] - EPS).floor() as i32);
    let (y0, y1) = ((b.lo[1] + EPS).floor() as i32, (b.hi[1] - EPS).floor() as i32);
    let (z0, z1) = ((b.lo[2] + EPS).floor() as i32, (b.hi[2] - EPS).floor() as i32);
    for y in y0..=y1 {
        for z in (z0..=z1).step_by(2) {
            for x in (x0..=x1).step_by(2) {
                let m = w.get_voxel(x, y, z);
                n += 1;
                if m == wm::WATER {
                    nw += 1;
                } else if m != 0 && w.palette[m as usize].props[2] == 3.0 {
                    nf += 1;
                }
            }
        }
    }
    if n == 0 { (0.0, 0.0) } else { (nw as f32 / n as f32, nf as f32 / n as f32) }
}

/// A run still solid 4 m up is a wall / trunk: it says nothing about the slope.
const WALL: f32 = -1e9;

/// Walkable ground height at column (x, z) near feet level y (uphill: the top of the solid
/// run; downhill: the first solid below). Ceilings are never seen.
fn ground_near(w: &World, x: i32, z: i32, y: f32) -> f32 {
    let y0 = y.floor() as i32;
    if player_blocks(w, w.get_voxel(x, y0, z)) {
        for yy in y0 + 1..=y0 + 40 {
            if !player_blocks(w, w.get_voxel(x, yy, z)) {
                return yy as f32;
            }
        }
        return WALL;
    }
    for yy in (y0 - 24..=y0 - 1).rev() {
        if player_blocks(w, w.get_voxel(x, yy, z)) {
            return (yy + 1) as f32;
        }
    }
    (y0 - 25) as f32
}

/// Slope of the ground under the player (rise per run, and the downhill direction).
fn ground_slope(w: &World, feet: Vec3, r: i32) -> (f32, Vec3) {
    let (x, z) = (feet.x.floor() as i32, feet.z.floor() as i32);
    let y = feet.y;
    let h = |dx: i32, dz: i32| {
        let g = ground_near(w, x + dx, z + dz, y);
        if g == WALL { y } else { g.max(y - 24.0) }
    };
    let gx = (h(r, 0) - h(-r, 0)) / (2.0 * r as f32);
    let gz = (h(0, r) - h(0, -r)) / (2.0 * r as f32);
    let g = (gx * gx + gz * gz).sqrt();
    (g, if g > 1e-4 { Vec3::new(-gx / g, 0.0, -gz / g) } else { Vec3::ZERO })
}

pub fn player_eye(p: &Player) -> Vec3 {
    let bob = (p.bob_phase * 2.0).sin() * 0.22 * p.bob_amp;
    Vec3::new(p.feet.x, p.feet.y + p.eye_h + p.eye_offset + bob, p.feet.z)
}

/// Put the player somewhere (spawn, load, teleport): velocity and smoothing reset.
pub fn player_place(p: &mut Player, feet: Vec3, t: &PlayerTuning) {
    p.feet = feet;
    p.vel = Vec3::ZERO;
    p.grounded = false;
    p.eye_offset = 0.0;
    p.step_rate = 0.0;
    p.since_step = 1.0;
    p.eye_h = if p.crouched { t.crouch_eye } else { t.eye };
    p.coyote = 0.0;
    p.jump_buf = 0.0;
    p.ground_body = None;
    p.bob_amp = 0.0;
}

fn cur_height(p: &Player, t: &PlayerTuning) -> f32 {
    if p.crouched { t.crouch_height } else { t.height }
}

/// The box overlaps something solid?
pub fn player_stuck(p: &Player, t: &PlayerTuning, w: &PlayerWorld) -> bool {
    let mut c = Collide::new(w.world);
    let b = box_at(p.feet, t.width * 0.5, cur_height(p, t));
    c.raster_bodies(w.objects.as_deref().map(|v| v.as_slice()), &b, 2.0);
    c.overlaps(&b) != 0
}

/// Move to the nearest free spot (up first, then around). False if none within reach.
pub fn player_unstuck(p: &mut Player, t: &PlayerTuning, w: &PlayerWorld) -> bool {
    let mut c = Collide::new(w.world);
    let (hw, h) = (t.width * 0.5, cur_height(p, t));
    c.raster_bodies(w.objects.as_deref().map(|v| v.as_slice()), &box_at(p.feet, hw, h), 20.0);
    let free_at = |f: Vec3| c.overlaps(&box_at(f, hw, h)) == 0;
    if free_at(p.feet) {
        return true;
    }
    let f0 = Vec3::new(p.feet.x.floor() + 0.5, p.feet.y.floor(), p.feet.z.floor() + 0.5);
    for dy in 0..=64 {
        let f = f0 + Vec3::new(0.0, dy as f32, 0.0);
        if free_at(f) {
            p.feet = f;
            p.vel = Vec3::ZERO;
            return true;
        }
    }
    for r in 1..=16 {
        for dy in (0..=24).step_by(2) {
            for i in 0..8 {
                let a = i as f32 * std::f32::consts::FRAC_PI_4;
                let f = f0 + Vec3::new((a.cos() * r as f32).round(), dy as f32, (a.sin() * r as f32).round());
                if free_at(f) {
                    p.feet = f;
                    p.vel = Vec3::ZERO;
                    return true;
                }
            }
        }
    }
    false
}

/// Would a solid voxel at cell `cc` overlap the player (no placing blocks into yourself)?
pub fn player_overlaps_cell(p: &Player, t: &PlayerTuning, cc: [i32; 3]) -> bool {
    let b = box_at(p.feet, t.width * 0.5, cur_height(p, t));
    (0..3).all(|a| b.lo[a] + EPS < cc[a] as f32 + 1.0 && b.hi[a] - EPS > cc[a] as f32)
}

/// The feet position for standing on the highest walkable surface of column (x, z).
pub fn player_ground_at(world: &World, x: f32, z: f32) -> Vec3 {
    let ix = if world.ring_w != 0 { x.floor() as i32 } else { (x.floor() as i32).clamp(0, world.nx - 1) };
    let iz = if world.ring_w != 0 { z.floor() as i32 } else { (z.floor() as i32).clamp(0, world.nz - 1) };
    for y in (0..world.ny).rev() {
        if player_blocks(world, world.get_voxel(ix, y, iz)) {
            return Vec3::new(x, (y + 1) as f32, z);
        }
    }
    Vec3::new(x, -1.0, z)
}

/// A dry, open place to stand near (x, z): the nearest grassy column (spiral search).
pub fn player_spawn_near(world: &World, x: f32, z: f32) -> Vec3 {
    let top = |ix: i32, iz: i32| -> u8 {
        for y in (0..world.ny).rev() {
            let m = world.get_voxel(ix, y, iz);
            if m != 0 {
                // ... resting on something: a floating sphere / canopy over a lake isn't ground.
                for d in 1..=24 {
                    let b = world.get_voxel(ix, y - d, iz);
                    if b == 0 || b == wm::WATER {
                        return 0;
                    }
                }
                return m;
            }
        }
        0
    };
    for r in (0..=400).step_by(6) {
        let n = if r == 0 { 1 } else { 16 };
        for i in 0..n {
            let a = i as f32 / 16.0 * std::f32::consts::TAU;
            let (mut px, mut pz) = (x + a.cos() * r as f32, z + a.sin() * r as f32);
            if world.ring_w == 0 {
                px = px.clamp(8.0, world.nx as f32 - 8.0);
                pz = pz.clamp(8.0, world.nz as f32 - 8.0);
            } else if !world.loaded_at(px.floor() as i32, pz.floor() as i32) {
                continue;
            }
            let m = top(px as i32, pz as i32);
            if matches!(m, wm::GRASS | wm::DRY_GRASS | wm::FLOWERS | wm::TALL_GRASS | wm::MOSS | wm::COARSE_DIRT) {
                return player_ground_at(world, px.floor() + 0.5, pz.floor() + 0.5);
            }
        }
    }
    player_ground_at(world, x, z)
}

/// Space pressed at `now` (seconds): a second press within 0.3 s toggles flight if allowed.
pub fn player_space_tap(p: &mut Player, now: f64, can_fly: bool) -> bool {
    let dbl = now - p.last_space_tap < 0.3;
    p.last_space_tap = if dbl { -10.0 } else { now };
    if !dbl || !can_fly {
        return false;
    }
    p.flying = !p.flying;
    if p.flying {
        p.vel.y = p.vel.y.max(0.0);
        p.grounded = false;
    }
    true
}

fn lerp_v(a: Vec3, b: Vec3, k: f32) -> Vec3 {
    a + (b - a) * k
}

/// One frame (dt clamped to 0.1 s, integrated in <= 1/120 s steps). `can_fly`: creative /
/// cheats. `noclip`: the editor's free camera.
#[allow(clippy::too_many_arguments)]
pub fn player_step(p: &mut Player, inp: &PlayerInput, yaw: f32, t: &PlayerTuning, w: &mut PlayerWorld, can_fly: bool, noclip: bool, dt: f32) {
    let dt = dt.clamp(0.0, 0.1);
    if dt <= 0.0 {
        return;
    }
    let fwd = Vec3::new(yaw.sin(), 0.0, yaw.cos());
    let right = Vec3::new(-yaw.cos(), 0.0, yaw.sin());
    let mut wish = fwd * inp.fwd + right * inp.side;
    if wish.length() > 1.0 {
        wish = wish.normalize();
    }
    let hw = t.width * 0.5;

    if noclip {
        p.eye_offset *= (-12.0 * dt).exp();
        p.step_rate = 0.0;
        let sp = t.fly * if inp.sprint { 4.0 } else { 1.0 };
        p.feet += wish * (sp * dt);
        p.feet.y += ((inp.jump_held as i32) as f32 - (inp.crouch as i32) as f32) * sp * dt;
        p.vel = Vec3::ZERO;
        p.grounded = false;
        p.ground_body = None;
        p.bob_amp = 0.0;
        let target = if p.crouched { t.crouch_eye } else { t.eye };
        p.eye_h += (target - p.eye_h) * (1.0 - (-14.0 * dt).exp());
        return;
    }
    if !can_fly {
        p.flying = false;
    }
    if player_stuck(p, t, w) {
        player_unstuck(p, t, w);
    }
    if inp.jump_pressed {
        p.jump_buf = 0.12;
    }
    let n_sub = ((dt / SUB_DT).ceil() as i32).max(1);
    let h = dt / n_sub as f32;
    let world = w.world;
    let mut c = Collide::new(world);
    for _ in 0..n_sub {
        // Standing on a moving body: it carries you.
        if let (Some(gb), true, Some(objs)) = (p.ground_body, p.grounded, w.objects.as_deref()) {
            if gb < objs.len() {
                let o = &objs[gb];
                let com = o.pose.apply(o.com);
                let pv = o.vel + o.ang_vel.cross(p.feet - com);
                if pv.length() > 0.01 {
                    c.raster_bodies(Some(objs), &box_at(p.feet, hw, t.height), pv.length() * h + 2.0);
                    let bh = cur_height(p, t);
                    move_axis(&c, &mut p.feet, 1, pv.y * h, hw, bh);
                    move_axis(&c, &mut p.feet, 0, pv.x * h, hw, bh);
                    move_axis(&c, &mut p.feet, 2, pv.z * h, hw, bh);
                }
            }
        }
        // Medium: water / foliage.
        let (water, fol) = sample_media(world, &box_at(p.feet, hw, cur_height(p, t)));
        p.in_water = water >= 0.4;
        p.foliage = fol;
        let e = player_eye(p);
        p.head_wet = world.get_voxel(e.x.floor() as i32, e.y.floor() as i32, e.z.floor() as i32) == wm::WATER;
        if p.in_water || p.grounded {
            p.flying = p.flying && !p.in_water;
        }
        // Crouch (not while flying / swimming); standing up needs headroom.
        c.raster_bodies(w.objects.as_deref().map(|v| v.as_slice()), &box_at(p.feet, hw, t.height), (p.vel.length() * h).max(0.0) + t.step + 2.0);
        let want_crouch = inp.crouch && !p.flying && !p.in_water;
        if want_crouch {
            p.crouched = true;
        } else if p.crouched && c.overlaps(&box_at(p.feet, hw, t.height)) == 0 {
            p.crouched = false;
        }
        let bh = cur_height(p, t);

        // Velocity.
        let foliage_slow = 1.0 - 0.55 * (fol * 4.0).min(1.0);
        let mut speed = if p.flying {
            t.fly * if inp.sprint { 2.0 } else { 1.0 }
        } else if p.in_water {
            t.swim
        } else if p.crouched {
            t.crouch
        } else if inp.sprint && inp.fwd > 0.0 {
            t.sprint
        } else {
            t.walk
        };
        if !p.flying {
            speed *= foliage_slow;
        }
        let target = wish * speed;
        let mut jumped = false;
        let ud = (inp.jump_held as i32) as f32 - (inp.crouch as i32) as f32;
        if p.flying {
            let vy = ud * speed * 0.7;
            p.vel = lerp_v(p.vel, Vec3::new(target.x, vy, target.z), 1.0 - (-8.0 * h).exp());
        } else if p.in_water {
            let k = 1.0 - (-5.0 * h).exp();
            p.vel.x += (target.x - p.vel.x) * k;
            p.vel.z += (target.z - p.vel.z) * k;
            let vy = if inp.jump_held {
                t.swim * 1.1
            } else if inp.crouch {
                -t.swim
            } else {
                -8.0
            };
            p.vel.y += (vy - p.vel.y) * (1.0 - (-3.0 * h).exp());
        } else {
            // Ground steeper than the slope limit (+5 degrees): slide down it.
            p.sliding = false;
            let mut downhill = Vec3::ZERO;
            if p.grounded {
                let lim = (t.slope + 5.0).to_radians().tan();
                let (g, d) = ground_slope(world, p.feet, 8);
                downhill = d;
                p.sliding = g > lim && ground_slope(world, p.feet, 3).0 > lim * 0.75;
            }
            let input = wish.length() > 0.01;
            let k = if p.sliding {
                1.5
            } else if p.grounded {
                t.ground_accel
            } else if input {
                t.air_accel
            } else {
                0.0
            };
            let kk = 1.0 - (-k * h).exp();
            p.vel.x += (target.x - p.vel.x) * kk;
            p.vel.z += (target.z - p.vel.z) * kk;
            if p.sliding {
                p.vel += downhill * (t.gravity * 0.55 * h);
                p.coyote = 0.0;
                p.jump_buf = 0.0;
            }
            p.vel.y = (p.vel.y - t.gravity * h).max(-t.terminal);
            if fol > 0.0 && p.vel.y < 0.0 {
                p.vel.y *= (-6.0 * fol * h).exp(); // leaves break a fall a little
            }
            p.coyote = if p.grounded && !p.sliding { 0.1 } else { p.coyote - h };
            if inp.jump_held && p.grounded && !p.sliding {
                p.jump_buf = p.jump_buf.max(h * 1.5); // held: hop again on landing
            }
            if p.jump_buf > 0.0 && p.coyote > 0.0 {
                p.vel.y = (2.0 * t.gravity * t.jump_apex).sqrt();
                p.coyote = 0.0;
                p.jump_buf = 0.0;
                p.grounded = false;
                jumped = true;
            }
        }
        p.jump_buf = (p.jump_buf - h).max(0.0);

        // Move: Y, then X and Z (step-up / edge guard / pushing).
        let was_grounded = p.grounded;
        let mut stepped = 0.0f32;
        let vy_before = p.vel.y;
        let hit_y = move_axis(&c, &mut p.feet, 1, p.vel.y * h, hw, bh);
        if hit_y != 0 {
            if p.vel.y < 0.0 {
                if !was_grounded {
                    p.last_land_speed = -vy_before;
                }
                if !was_grounded && -vy_before > 90.0 {
                    p.eye_offset -= ((-vy_before - 90.0) * 0.015).min(3.0); // landing dip
                }
            }
            p.vel.y = 0.0;
        }
        {
            let mut probe = p.feet;
            probe.y -= 0.05;
            let g = if p.vel.y <= 0.0 { c.overlaps(&box_at(probe, hw, 0.05)) } else { 0 };
            p.grounded = g != 0;
            p.ground_body = (g >= 2).then(|| (g - 2) as usize);
        }
        let mut pushed_into_wall = false;
        for a in [0usize, 2] {
            let d = *axis(&mut p.vel, a) * h;
            if d.abs() < 1e-7 {
                continue;
            }
            let before = p.feet;
            let mut hit = move_axis(&c, &mut p.feet, a, d, hw, bh);
            // Step up a small ledge - unless the ground keeps rising steeper than the slope
            // limit over the next metre (a hillside, not a curb).
            let mut walkable = true;
            if hit != 0 && p.grounded && !p.flying {
                let dir_s = if d > 0.0 { 1.0 } else { -1.0 };
                let tan_s = t.slope.to_radians().tan();
                let mut front = before;
                *axis(&mut front, a) += dir_s * (hw + 0.5);
                let g0 = ground_near(world, front.x.floor() as i32, front.z.floor() as i32, before.y);
                let mut prev = g0;
                if g0 != WALL {
                    for run in 1..=12 {
                        let mut ahead = front;
                        *axis(&mut ahead, a) += dir_s * run as f32;
                        let ga = ground_near(world, ahead.x.floor() as i32, ahead.z.floor() as i32, before.y);
                        if ga == WALL || ga - prev > t.step {
                            break;
                        }
                        if ga - g0 > tan_s * run as f32 + 2.0 {
                            walkable = false;
                            break;
                        }
                        prev = ga;
                    }
                }
            }
            if hit != 0 && p.grounded && !p.flying && walkable {
                for st in 1..=t.step as i32 {
                    let mut up = before;
                    up.y += st as f32;
                    if c.overlaps(&box_at(up, hw, bh)) != 0 {
                        break; // no headroom
                    }
                    let mut fw = up;
                    *axis(&mut fw, a) += d;
                    if c.overlaps(&box_at(fw, hw, bh)) == 0 {
                        p.feet = fw;
                        p.eye_offset -= st as f32;
                        stepped += st as f32;
                        hit = 0;
                        break;
                    }
                }
            }
            if hit == 0 && p.crouched && p.grounded && !p.flying {
                // edge guard: never walk off a drop while crouched
                let mut under = p.feet;
                under.y -= t.step;
                if c.overlaps(&box_at(under, hw, t.step)) == 0 {
                    p.feet = before;
                    hit = 1;
                }
            }
            if hit != 0 {
                if hit >= 2 && w.bodies_dynamic {
                    if let Some(objs) = w.objects.as_deref_mut() {
                        let o = &mut objs[(hit - 2) as usize];
                        let mut dir = Vec3::ZERO;
                        *axis(&mut dir, a) = if d > 0.0 { 1.0 } else { -1.0 };
                        o.asleep = false;
                        o.sleep_timer = 0.0;
                        if o.vel.dot(dir) < speed * 0.9 {
                            o.vel += dir * (9000.0 / o.mass.max(1.0) * h);
                        }
                    }
                }
                *axis(&mut p.vel, a) = 0.0;
                pushed_into_wall = true;
            }
        }
        // Swimming against the shore: Space climbs out.
        if p.in_water && pushed_into_wall && inp.jump_held && !p.head_wet {
            p.vel.y = p.vel.y.max((2.0 * t.gravity * t.jump_apex).sqrt() * 0.8);
        }
        // Walking down a slope: stay on the ground instead of skipping down it.
        if was_grounded && !p.grounded && !jumped && p.vel.y <= 0.0 && !p.flying && !p.in_water {
            let y0 = p.feet.y;
            let mut f = p.feet;
            if move_axis(&c, &mut f, 1, -t.step, hw, bh) != 0 {
                p.eye_offset += y0 - f.y;
                stepped -= y0 - f.y;
                p.feet = f;
                p.grounded = true;
                p.vel.y = 0.0;
            }
        }
        // The eye eases back after a step (~0.1 s); steps in a row move it at their rate.
        if stepped != 0.0 {
            p.step_rate += stepped / 0.3;
            p.since_step = 0.0;
        } else {
            p.since_step += h;
        }
        p.step_rate *= (-h / if p.since_step > 0.15 { 0.05 } else { 0.3 }).exp();
        {
            let k_ease = 12.0;
            let rest = p.step_rate / k_ease;
            p.eye_offset = rest + (p.eye_offset - rest) * (-k_ease * h).exp();
            if p.step_rate > 0.0 {
                p.eye_offset = p.eye_offset.min(0.5);
            } else if p.step_rate < 0.0 {
                p.eye_offset = p.eye_offset.max(-0.5);
            }
        }
        // Flying down onto the ground lands you.
        if p.flying && p.grounded && inp.crouch {
            p.flying = false;
        }
        // View bob while walking.
        let hs = (p.vel.x * p.vel.x + p.vel.z * p.vel.z).sqrt();
        if p.grounded && !p.flying && hs > 4.0 {
            p.bob_phase += std::f32::consts::PI * 1.9 * (hs / t.walk.max(1.0)).sqrt() * h;
            p.bob_amp += ((hs / t.walk).min(1.0) - p.bob_amp) * (1.0 - (-8.0 * h).exp());
        } else {
            p.bob_amp *= (-8.0 * h).exp();
        }
    }
    p.eye_offset = p.eye_offset.clamp(-6.0, 6.0);
    let target = if p.crouched { t.crouch_eye } else { t.eye };
    p.eye_h += (target - p.eye_h) * (1.0 - (-14.0 * dt).exp());
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::voxel::world::{FlatLayer, FlatWorld};

    fn flat() -> World {
        let mut w = World::new(128, 64, 128);
        w.generate_flat(&FlatWorld { layers: vec![FlatLayer { mat: wm::STONE, thickness: 0 }], height: 10, water: 0 });
        w
    }

    #[test]
    fn falls_lands_walks_and_steps() {
        let mut w = flat();
        let t = PlayerTuning::default();
        let mut p = Player::default();
        player_place(&mut p, Vec3::new(64.0, 30.0, 64.0), &t);
        for _ in 0..120 {
            let mut pw = PlayerWorld { world: &w, objects: None, bodies_dynamic: false };
            player_step(&mut p, &PlayerInput::default(), 0.0, &t, &mut pw, false, false, 1.0 / 60.0);
        }
        assert!(p.grounded && (p.feet.y - 10.0).abs() < 1e-3, "landed at {}", p.feet.y);
        // A 3-voxel ledge ahead (+z) is stepped up.
        w.fill_box(0, 10, 70, 127, 12, 127, wm::STONE);
        let inp = PlayerInput { fwd: 1.0, ..Default::default() };
        for _ in 0..60 {
            let mut pw = PlayerWorld { world: &w, objects: None, bodies_dynamic: false };
            player_step(&mut p, &inp, 0.0, &t, &mut pw, false, false, 1.0 / 60.0);
        }
        assert!(p.feet.z > 72.0 && (p.feet.y - 13.0).abs() < 1e-3, "at {:?}", p.feet);
        // A tall wall stops it.
        w.fill_box(0, 13, 90, 127, 40, 127, wm::STONE);
        for _ in 0..120 {
            let mut pw = PlayerWorld { world: &w, objects: None, bodies_dynamic: false };
            player_step(&mut p, &inp, 0.0, &t, &mut pw, false, false, 1.0 / 60.0);
        }
        assert!(p.feet.z <= 87.0 + 1e-3, "went through the wall: {:?}", p.feet);
    }

    #[test]
    fn jumps_to_apex() {
        let w = flat();
        let t = PlayerTuning::default();
        let mut p = Player::default();
        player_place(&mut p, Vec3::new(64.0, 10.0, 64.0), &t);
        let mut pw = PlayerWorld { world: &w, objects: None, bodies_dynamic: false };
        player_step(&mut p, &PlayerInput::default(), 0.0, &t, &mut pw, false, false, 1.0 / 60.0);
        let jump = PlayerInput { jump_pressed: true, jump_held: true, ..Default::default() };
        player_step(&mut p, &jump, 0.0, &t, &mut pw, false, false, 1.0 / 60.0);
        let mut top = 0.0f32;
        for _ in 0..60 {
            player_step(&mut p, &PlayerInput::default(), 0.0, &t, &mut pw, false, false, 1.0 / 120.0);
            top = top.max(p.feet.y - 10.0);
        }
        assert!((top - t.jump_apex).abs() < 0.6, "apex {top}");
    }
}
