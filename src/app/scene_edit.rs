//! In-world editing: the aim raycast (world + bodies), the brush target and its preview, one
//! brush dab (carve / place / paint) with detach / split, water and GPU sync, recorded for undo.

use std::time::Instant;

use crate::cvars;
use crate::destruction::{split_object, world_carve_detach, DETACH_MARGIN, DETACH_MAX_HALF};
use crate::edit::brush::{apply_brush, apply_paint, brush_for, BrushBox, BrushShape};
use crate::edit::undo::{EditRecord, EditScope};
use crate::fluid::FluidSim;
use crate::gfx::context::Context;
use crate::gfx::renderer::record::BrushPreviewPc;
use crate::gfx::renderer::Renderer;
use crate::math::{Transform, Vec3};
use crate::objects::VoxObject;
use crate::physics::wake_near;
use crate::voxel::sparse_volume::{RayHit, SparseVolume};
use crate::voxel::world::{wm, Material, World};

/// The scene raycast's result: `obj` = None: the world; else `hit` is in that body's space.
#[derive(Clone, Copy, Debug, Default)]
pub struct SceneHit {
    pub hit: RayHit,
    pub obj: Option<usize>,
}

/// Raycast the scene (the world, then every body in its own space): the nearest hit wins. A
/// camera inside solid terrain sees (and edits) from where the ray leaves that solid, like the
/// renderer (gbuffer.comp exitSolid). The ray passes through water (build through it).
pub fn scene_raycast(world: &World, objects: &[VoxObject], ro: Vec3, rd: Vec3, max_t: f32) -> Option<SceneHit> {
    let solid = |c: [i32; 3]| {
        let m = world.get_voxel(c[0], c[1], c[2]);
        m != 0 && m != wm::WATER
    };
    let mut skip = 0.0;
    let start = ro.floor().as_ivec3().to_array();
    if solid(start) {
        // Voxel walk to the first non-solid cell.
        let (o, d) = (ro.to_array(), rd.to_array());
        let mut c = start;
        let (mut st, mut t_max, mut t_del) = ([0i32; 3], [0f32; 3], [0f32; 3]);
        for a in 0..3 {
            let dd = if d[a].abs() < 1e-6 { 1e-6 } else { d[a] };
            st[a] = if dd < 0.0 { -1 } else { 1 };
            t_max[a] = ((c[a] + (st[a] > 0) as i32) as f32 - o[a]) / dd;
            t_del[a] = (1.0 / dd).abs();
        }
        let mut t = 0.0;
        let mut out = false;
        for _ in 0..768 {
            if t >= max_t {
                break;
            }
            if !solid(c) {
                out = true;
                break;
            }
            let a = if t_max[0] < t_max[1] && t_max[0] < t_max[2] { 0 } else if t_max[1] < t_max[2] { 1 } else { 2 };
            c[a] += st[a];
            t = t_max[a];
            t_max[a] += t_del[a];
        }
        if !out {
            return None;
        }
        skip = t + 1e-3;
    }
    let mut best = SceneHit { hit: world.raycast(ro + rd * skip, rd, max_t - skip, wm::WATER), obj: None };
    best.hit.t += skip;
    let mut best_t = if best.hit.hit { best.hit.t } else { max_t };
    for (i, o) in objects.iter().enumerate() {
        let inv = o.pose.inverse();
        let h = o.vol.raycast(inv.r * ro + inv.t, inv.r * rd, best_t, 0);
        if h.hit && h.t < best_t {
            best_t = h.t;
            best = SceneHit { hit: h, obj: Some(i) };
        }
    }
    best.hit.hit.then_some(best)
}

/// Where the brush lands this frame: the same raycast the edit uses, so the preview and the
/// click always agree. The boxes are in the hit volume's own voxel space.
#[derive(Clone, Copy, Debug)]
pub struct BrushTarget {
    pub obj: Option<usize>,
    pub carve: BrushBox,
    pub place: BrushBox,
    pub normal: [i32; 3],
    pub cell: [i32; 3],
    /// Identity for the world.
    pub world_to_local: Transform,
    pub dist: f32,
}

pub fn brush_shape() -> BrushShape {
    if cvars::R_BRUSH_SHAPE.is("cube") { BrushShape::Cube } else { BrushShape::Sphere }
}

pub fn brush_target(world: &World, objects: &[VoxObject], ro: Vec3, rd: Vec3) -> Option<BrushTarget> {
    let sh = scene_raycast(world, objects, ro, rd, cvars::R_BRUSH_REACH.get())?;
    let (size, shape) = (cvars::R_BRUSH_SIZE.get(), brush_shape());
    Some(BrushTarget {
        obj: sh.obj,
        carve: brush_for(sh.hit.cell, sh.hit.normal, size, shape, false),
        place: brush_for(sh.hit.cell, sh.hit.normal, size, shape, true),
        normal: sh.hit.normal,
        cell: sh.hit.cell,
        world_to_local: sh.obj.map_or(Transform::default(), |i| objects[i].pose.inverse()),
        dist: sh.hit.t,
    })
}

/// Preview views: carve cells (red), place ghost, paint (the carve cells tinted).
pub const PREVIEW_CARVE: i32 = 4;
pub const PREVIEW_PLACE: i32 = 8;
pub const PREVIEW_PAINT: i32 = 16;

pub fn brush_preview_pc(t: &BrushTarget, palette: &[Material], mat: u8, out: (u32, u32), views: i32) -> BrushPreviewPc {
    let m = &palette[mat.max(1) as usize];
    BrushPreviewPc {
        world_to_local: t.world_to_local.to_mat4(),
        carve: [t.carve.lo[0], t.carve.lo[1], t.carve.lo[2], t.carve.size],
        place: [t.place.lo[0], t.place.lo[1], t.place.lo[2], t.carve.shape as i32],
        place_col: [m.r, m.g, m.b, (1 | if t.obj.is_some() { 2 } else { 0 } | views) as f32],
        out_res: [out.0 as i32, out.1 as i32, 0, 0],
    }
}

#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub enum BrushOp {
    Carve,
    Place,
    Paint,
}

fn count_emissive(v: &SparseVolume, palette: &[Material]) -> usize {
    let glow = |m: u8| palette[m as usize].emissive > 0.0;
    let mut n = v.grid.iter().filter(|&&g| g < 256 && g != 0 && glow(g as u8)).count() * 512;
    for &g in &v.grid {
        if g >= 256 {
            n += v.bricks[(g - 256) as usize].vox.iter().filter(|&&m| m != 0 && glow(m)).count();
        }
    }
    n
}

/// What a dab needs.
pub struct EditEnv<'a> {
    pub ctx: &'a Context,
    pub r: &'a mut Renderer,
    pub world: &'a mut World,
    pub objects: &'a mut Vec<VoxObject>,
    pub fluid: &'a mut FluidSim,
}

/// One brush dab at `t`. Hitting a body edits it in place (mass recomputed, woken; grows /
/// splits as needed); hitting the world wakes nearby bodies and water, and a carve detaches
/// what it cut loose. Applies the matching GPU sync; returns the undo record.
pub fn brush_edit(env: &mut EditEnv, t: &BrushTarget, op: BrushOp, mat: u8) -> EditRecord {
    let t0 = Instant::now();
    let label = match op {
        BrushOp::Carve => "Carve",
        BrushOp::Place => "Place",
        BrushOp::Paint => "Paint",
    };
    let mut bx = if op == BrushOp::Place { t.place } else { t.carve };
    let c = Vec3::new(bx.centre(0), bx.centre(1), bx.centre(2));
    let reach = bx.size as f32 * 0.5 + 1.0; // the region an edit can influence
    let mut new_bodies = 0;
    let rec = if let Some(oi) = t.obj {
        let scope = EditScope::new(env.world, env.objects, None, Some(oi), label);
        let palette = &env.world.palette;
        let o = &mut env.objects[oi];
        let mut structural = false; // grown or split: buffers resize / bodies added
        let mut glow_changed = false; // header glow integrity re-uploaded
        match op {
            BrushOp::Place => {
                // Grow the volume so you can build past its bounds.
                let g = o.ensure_contains(Vec3::from_array(bx.lo.map(|v| v as f32 - 1.0)), Vec3::from_array(bx.lo.map(|v| (v + bx.size + 1) as f32)));
                structural = g.grew;
                bx.lo = std::array::from_fn(|a| bx.lo[a] + g.shift[a].round() as i32);
                apply_brush(&mut o.vol, &bx, mat, |_, _, _, _| {});
                o.init_body();
                o.asleep = false;
                o.sleep_timer = 0.0;
            }
            BrushOp::Paint => {
                let pre = count_emissive(&o.vol, palette);
                apply_paint(&mut o.vol, &bx, mat);
                glow_changed = pre != count_emissive(&o.vol, palette);
                o.init_body(); // density per material
            }
            BrushOp::Carve => {
                let pre = count_emissive(&o.vol, palette);
                apply_brush(&mut o.vol, &bx, 0, |_, _, _, _| {});
                if pre > 0 {
                    // A glowing body loses the glow of what was carved off.
                    o.glow_integrity *= count_emissive(&o.vol, palette) as f32 / pre as f32;
                    glow_changed = true;
                }
                o.init_body();
                o.asleep = false;
                o.sleep_timer = 0.0;
                // Carving can disconnect the volume: the loose pieces become bodies.
                new_bodies = split_object(env.objects, oi);
                structural = new_bodies > 0;
            }
        }
        let rec = scope.finish(env.world, env.objects);
        if structural {
            env.r.gpu.init_objects(env.ctx, env.world, env.objects);
            env.r.rebind_object_buffers();
        } else {
            env.r.gpu.sync_objects(env.ctx, env.world, env.objects);
            if glow_changed {
                env.r.gpu.update_objects(env.objects);
            }
        }
        rec
    } else {
        // The undo region: the brush box plus what a carve's island detach may remove.
        let m = if op == BrushOp::Carve { (reach as i32 + DETACH_MARGIN).min(DETACH_MAX_HALF) } else { 1 };
        let ci = c.as_ivec3().to_array();
        let lo = ci.map(|v| v - m - bx.size);
        let hi = ci.map(|v| v + m + bx.size);
        let mut scope = EditScope::new(env.world, env.objects, Some((lo, hi)), None, label);
        if op == BrushOp::Paint {
            apply_paint(env.world, &bx, mat);
        } else {
            let fluid = &mut *env.fluid;
            apply_brush(env.world, &bx, if op == BrushOp::Place { mat } else { 0 }, |v, x, y, z| fluid.displace(v, x, y, z));
            // Water next to the edit reacts; nearby bodies fall.
            let (blo, bhi) = bx.bounds();
            env.fluid.wake(blo[0] - 1, blo[1] - 1, blo[2] - 1, bhi[0] + 1, bhi[1] + 1, bhi[2] + 1);
            wake_near(env.objects, c, reach + 12.0);
        }
        if op == BrushOp::Carve {
            // A piece that falls reports its bounds first: the undo records it whole.
            let mut extend = |w: &World, plo: [i32; 3], phi: [i32; 3]| scope.extend(w, plo, phi);
            new_bodies = world_carve_detach(env.world, c, reach, env.objects, Some(&mut extend));
        }
        let rec = scope.finish(env.world, env.objects);
        env.r.gpu.sync(env.ctx, env.world);
        let mm = reach.ceil() as i32;
        env.r.note_world_edit(bx.lo.map(|v| v - mm), bx.lo.map(|v| v + bx.size + mm));
        if new_bodies > 0 {
            env.r.gpu.init_objects(env.ctx, env.world, env.objects);
            env.r.rebind_object_buffers();
        }
        rec
    };
    log::info!(target: "edit", "{label} {} {} {}: {} bricks, +{new_bodies} bodies, {:.2} ms", if t.obj.is_some() { "body" } else { "world" },
               cvars::R_BRUSH_SHAPE.get(), bx.size, rec.world.len(), t0.elapsed().as_secs_f64() * 1e3);
    rec
}
