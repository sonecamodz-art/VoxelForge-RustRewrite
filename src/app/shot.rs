//! Headless rendering: `--shot out.png` drives the renderer for N frames of the startup scene
//! (temporal accumulation) and writes the result; `--bench` times frames.

use ash::vk;
use gpu_allocator::MemoryLocation;

use crate::camera::FlyCamera;
use crate::cvars;
use crate::gfx::context::{Context, Image, COLOR_LAYERS};
use crate::gfx::postfx::PostFx;
use crate::gfx::renderer::record::Rec;
use crate::gfx::renderer::Renderer;
use crate::math::{half_to_f32, Vec3};
use crate::objects::build_self_test_objects;
use crate::voxel::world::{wm, World};

pub struct ShotArgs {
    pub out: String,
    pub w: u32,
    pub h: u32,
    pub frames: u32,
    /// final | albedo | normal | motion
    pub dump: String,
    pub validation: bool,
}

/// Top of the terrain column at (x, z), skipping plants.
pub fn terrain_at(world: &World, x: f32, z: f32) -> f32 {
    for y in (1..world.ny).rev() {
        let m = world.get_voxel(x as i32, y, z as i32);
        if m != 0 && !wm::is_plant(m) {
            return y as f32;
        }
    }
    400.0
}

/// The default view: ~15 m above the spawn area, over the forest toward the building.
pub fn default_camera(world: &World) -> FlyCamera {
    let c = world.mid_x();
    let mut cam = FlyCamera { pos: Vec3::new(c - 260.0, terrain_at(world, c - 260.0, c - 260.0) + 150.0, c - 260.0), ..Default::default() };
    cam.yaw = 0.8;
    cam.pitch = -0.3;
    apply_cam_env(&mut cam, "VF_CAM");
    cam
}

/// VF_CAM="x,y,z,yaw,pitch" overrides the camera (reproduce a windowed view in a shot).
pub fn apply_cam_env(cam: &mut FlyCamera, var: &str) {
    if let Ok(s) = std::env::var(var) {
        let v: Vec<f32> = s.split(',').filter_map(|t| t.trim().parse().ok()).collect();
        if v.len() == 5 {
            cam.pos = Vec3::new(v[0], v[1], v[2]);
            cam.yaw = v[3];
            cam.pitch = v[4];
            log::info!(target: "cam", "{:.0},{:.0},{:.0} yaw {:.2} pitch {:.2}", v[0], v[1], v[2], v[3], v[4]);
        }
    }
}

/// Copy a GENERAL image to host memory (bytes per pixel `bpp`); leaves it GENERAL.
pub fn readback(ctx: &Context, img: &Image, bpp: u32) -> Vec<u8> {
    let (w, h) = (img.w(), img.h());
    let size = (w * h * bpp) as u64;
    let mut buf = ctx.create_buffer("readback", size, vk::BufferUsageFlags::TRANSFER_DST, MemoryLocation::GpuToCpu);
    let dev = &ctx.device;
    let cb = ctx.begin_one_shot();
    use crate::gfx::context::{image_barrier, Access, Stage};
    image_barrier(dev, cb, img.image, Stage::ALL_COMMANDS, Access::MEMORY_WRITE, Stage::COPY, Access::TRANSFER_READ,
                  vk::ImageLayout::GENERAL, vk::ImageLayout::TRANSFER_SRC_OPTIMAL);
    let c = [vk::BufferImageCopy::default().image_subresource(COLOR_LAYERS).image_extent(vk::Extent3D { width: w, height: h, depth: 1 })];
    unsafe { dev.cmd_copy_image_to_buffer(cb, img.image, vk::ImageLayout::TRANSFER_SRC_OPTIMAL, buf.buf, &c) };
    image_barrier(dev, cb, img.image, Stage::COPY, Access::TRANSFER_READ, Stage::ALL_COMMANDS, Access::MEMORY_WRITE,
                  vk::ImageLayout::TRANSFER_SRC_OPTIMAL, vk::ImageLayout::GENERAL);
    ctx.end_one_shot_and_wait(cb);
    let out = buf.mapped()[..size as usize].to_vec();
    buf.destroy(ctx);
    out
}

/// Write RGBA8 pixels as PNG. `bottom_up`: image row 0 is the bottom of the picture (the
/// renderer's convention: ndc.y = -1 at row 0).
pub fn write_png(path: &str, w: u32, h: u32, rgba: &[u8], bottom_up: bool) -> std::io::Result<()> {
    let f = std::fs::File::create(path)?;
    let mut enc = png::Encoder::new(std::io::BufWriter::new(f), w, h);
    enc.set_color(png::ColorType::Rgba);
    enc.set_depth(png::BitDepth::Eight);
    let mut wr = enc.write_header().map_err(std::io::Error::other)?;
    let data: Vec<u8> = if bottom_up {
        (0..h).rev().flat_map(|y| rgba[(y * w * 4) as usize..((y + 1) * w * 4) as usize].iter().copied()).collect()
    } else {
        rgba.to_vec()
    };
    wr.write_image_data(&data).map_err(std::io::Error::other)
}

pub fn run_shot(a: &ShotArgs) -> i32 {
    let (mut ctx, _) = Context::new(&[], a.validation, None);
    let t0 = std::time::Instant::now();
    let mut world = World::default();
    world.generate(cvars::WORLD_SEED.get() as u32);
    let mut objects = build_self_test_objects(&world);
    log::info!(target: "shot", "scene ready in {:.0} ms", t0.elapsed().as_secs_f64() * 1e3);
    let cam = default_camera(&world);
    let mut r = Renderer::new(&ctx, &mut world, &mut objects, a.w, a.h, a.w, a.h);
    let mut post = PostFx::new(&ctx, &r);
    // Headless edit self-tests (the interactive editor's sync path): VF_CARVETEST=radius carves
    // a sphere where the camera looks; VF_PLACETEST=material drops a ball ahead of it.
    if let Ok(v) = std::env::var("VF_CARVETEST") {
        let rad: f32 = v.parse().ok().filter(|r: &f32| *r >= 1.0).unwrap_or(22.0);
        // VF_CARVE_CAM (same format as VF_CAM) aims the carve from elsewhere than the view.
        let mut aim = cam;
        apply_cam_env(&mut aim, "VF_CARVE_CAM");
        // VF_CARVE_AT="x,y,z" carves exactly there instead (a ray from the air stops at water).
        let at: Vec<f32> = std::env::var("VF_CARVE_AT").unwrap_or_default().split(',').filter_map(|t| t.trim().parse().ok()).collect();
        let h = world.raycast(aim.pos, aim.forward(), 600.0, 0);
        let target = if at.len() == 3 {
            Some(crate::math::Vec3::new(at[0], at[1], at[2]))
        } else {
            h.hit.then(|| crate::math::Vec3::new(h.cell[0] as f32 + 0.5, h.cell[1] as f32 + 0.5, h.cell[2] as f32 + 0.5))
        };
        if let Some(c) = target {
            world.fill_sphere(c.x, c.y, c.z, rad, 0);
            let spawned = crate::destruction::world_carve_detach(&mut world, c, rad, &mut objects, None);
            // VF_FLUIDTEST=ticks: the water next to the carve flows in for that many 20 Hz ticks.
            if let Some(ticks) = std::env::var("VF_FLUIDTEST").ok().and_then(|v| v.parse::<u32>().ok()) {
                let mut fl = crate::fluid::FluidSim::default();
                let (lo, hi) = (c - (rad + 1.0), c + (rad + 1.0));
                fl.wake(lo.x as i32, lo.y as i32, lo.z as i32, hi.x as i32, hi.y as i32, hi.z as i32);
                let t = std::time::Instant::now();
                let mut n = 0;
                while n < ticks && !fl.idle() {
                    fl.tick(&mut world);
                    n += 1;
                }
                let wet = (0..=(2.0 * rad) as i32).filter(|&d| fl.volume(&world, c.x as i32, (c.y - rad) as i32 + d, c.z as i32) > 0.0).count();
                log::info!(target: "fluidtest", "{n} ticks in {:.1} ms, {} active, {} partial, {wet} wet cells in the hole's centre column",
                           t.elapsed().as_secs_f64() * 1e3, fl.active_count(), fl.partial_count());
            }
            let n = world.dirty_grid.len();
            r.gpu.sync(&ctx, &mut world);
            if spawned > 0 {
                r.gpu.init_objects(&ctx, &world, &mut objects);
                r.rebind_object_buffers();
            }
            log::info!(target: "carvetest", "carved r={rad} at {c} ({n} dirty cells, {spawned} bodies)");
        }
    }
    if let Ok(v) = std::env::var("VF_PLACETEST") {
        let mat: u8 = v.parse().ok().filter(|m: &u8| *m >= 1).unwrap_or(wm::BRICKM);
        let c = cam.pos + cam.forward() * 90.0;
        world.fill_sphere(c.x, c.y, c.z, 18.0, mat);
        r.gpu.sync(&ctx, &mut world);
        log::info!(target: "placetest", "placed mat={mat} ball at {:.0},{:.0},{:.0}", c.x, c.y, c.z);
    }
    // Physics self-test: VF_SIMTEST=steps runs that many 60 Hz steps so the hovering bodies
    // fall and settle, then uploads their poses.
    if let Some(steps) = std::env::var("VF_SIMTEST").ok().and_then(|v| v.parse::<u32>().ok()) {
        crate::physics::wake_near(&mut objects, crate::math::Vec3::ZERO, 1e9);
        let mut ph = crate::physics::Physics::default();
        let y0 = objects.first().map_or(0.0, |o| o.pose.t.y);
        let t = std::time::Instant::now();
        for _ in 0..steps {
            ph.step(&world, &mut objects, 1.0 / 60.0);
        }
        r.gpu.update_objects(&objects);
        let asleep = objects.iter().filter(|o| o.asleep).count();
        for (i, o) in objects.iter().enumerate() {
            let com = o.pose.apply(o.com);
            log::debug!(target: "simtest", "body {i}: com {:.0},{:.0},{:.0} up.y {:.2} {}{}", com.x, com.y, com.z, (o.pose.r * crate::math::Vec3::Y).y,
                        if o.asleep { "asleep" } else { "awake" }, if o.hinged { " hinged" } else { "" });
        }
        log::info!(target: "simtest", "{} bodies, {steps} steps in {:.1} ms, body0 y {y0:.1} -> {:.1}, {asleep} asleep",
                   objects.len(), t.elapsed().as_secs_f64() * 1e3, objects.first().map_or(0.0, |o| o.pose.t.y));
    }
    let motion = a.dump == "motion";
    let pan = if motion { 0.02 } else { 0.0 };
    let mut cam_now = cam;
    let mut prev = cam;
    let standard = r.standard_mode();
    let raw = cvars::R_DENOISER.is("none");
    let t_frames = std::time::Instant::now();
    let mut gpu_total = 0.0;
    for frame in 0..a.frames.max(1) {
        let pp = (frame & 1) as usize;
        let time = frame as f32 / 60.0;
        r.update_lights(&world, &objects, cam_now.pos, time);
        r.set_frame_uniforms(&cam_now, &prev, frame, cvars::R_EXPOSURE.get(), time, [0.0, 0.0]);
        r.update_sets(&ctx, pp);
        let cb = ctx.begin_one_shot();
        {
            let rec = Rec { ctx: &ctx, cb };
            if standard || raw {
                if standard {
                    r.record_standard(&rec);
                } else {
                    r.record_raw(&rec);
                }
                crate::gfx::context::pass_barrier(&ctx.device, cb);
                r.record_torch_fx(&rec, 1, r.w, r.h);
                post.prepare(&ctx.device, cb, 1, 1.0 / 60.0);
                post.tonemap(&ctx.device, cb, 1, r.w, r.h, cvars::R_EXPOSURE.get(), 0, 1 << 30, -1);
                r.stamp(&rec, "tonemap");
            } else {
                r.record(&rec);
            }
            r.record_dust(&rec, r.w, r.h);
        }
        ctx.end_one_shot_and_wait(cb);
        r.read_timings(&ctx, pp);
        gpu_total = r.pass_ms.iter().map(|p| p.1).sum::<f32>();
        prev = cam_now;
        cam_now.yaw += pan;
    }
    let per = t_frames.elapsed().as_secs_f64() * 1e3 / a.frames.max(1) as f64;
    let passes: Vec<String> = r.pass_ms.iter().map(|(n, ms)| format!("{n}={ms:.2}")).collect();
    log::info!(target: "shot", "{} frames, {per:.1} ms/frame wall; last frame GPU {gpu_total:.2} ms: {}", a.frames, passes.join(" "));
    let pp = ((a.frames.max(1) - 1) & 1) as usize;
    let (img, bpp) = match a.dump.as_str() {
        "albedo" => (&r.t.gb_albedo, 4),
        "normal" => (&r.t.gb_normal[pp], 4),
        "motion" => (&r.t.gb_motion, 4),
        _ => (&r.t.out_tex, 4),
    };
    let mut px = readback(&ctx, img, bpp);
    if motion {
        // RG16F motion: R = +x, G = +y, grey = still (exaggerated).
        let n = (img.w() * img.h()) as usize;
        let mut rgba = vec![0u8; n * 4];
        for i in 0..n {
            let mx = half_to_f32(u16::from_le_bytes([px[i * 4], px[i * 4 + 1]]));
            let my = half_to_f32(u16::from_le_bytes([px[i * 4 + 2], px[i * 4 + 3]]));
            rgba[i * 4] = ((0.5 + mx * 6.0).clamp(0.0, 1.0) * 255.0) as u8;
            rgba[i * 4 + 1] = ((0.5 + my * 6.0).clamp(0.0, 1.0) * 255.0) as u8;
            rgba[i * 4 + 2] = 128;
            rgba[i * 4 + 3] = 255;
        }
        px = rgba;
    }
    let ok = write_png(&a.out, img.w(), img.h(), &px, true);
    match &ok {
        Ok(()) => log::info!(target: "shot", "{} -> {} ({}x{})", a.dump, a.out, img.w(), img.h()),
        Err(e) => log::error!(target: "shot", "writing {}: {e}", a.out),
    }
    post.destroy(&ctx);
    r.destroy(&ctx);
    ctx.destroy();
    if ok.is_ok() { 0 } else { 1 }
}
