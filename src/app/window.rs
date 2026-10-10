//! The game window: winit event loop, swapchain, two frames in flight, the fly camera,
//! in-world editing (carve / place), and the render path chosen by the cvars.
//!
//! The world generates on a worker thread while the window shows a progress screen, so the
//! window is responsive from the first frame.

use std::sync::atomic::{AtomicI32, Ordering};
use std::sync::Arc;
use std::time::Instant;

use ash::vk;
use raw_window_handle::{HasDisplayHandle, HasWindowHandle};
use winit::application::ApplicationHandler;
use winit::event::{DeviceEvent, DeviceId, ElementState, MouseButton, MouseScrollDelta, WindowEvent};
use winit::event_loop::{ActiveEventLoop, ControlFlow, EventLoop};
use winit::keyboard::{KeyCode, PhysicalKey};
use winit::window::{CursorGrabMode, Fullscreen, Window, WindowId};

use super::shot::default_camera;
use super::swapchain::{present_mode_from_cvar, Swapchain};
use crate::camera::{FlyCamera, MoveInput};
use crate::cvars;
use crate::gfx::context::{image_barrier, memory_barrier, vkc, Access, Context, Stage, COLOR_LAYERS, COLOR_RANGE};
use crate::gfx::postfx::PostFx;
use crate::gfx::renderer::record::Rec;
use crate::gfx::renderer::Renderer;
use crate::math::Vec3;
use crate::objects::{build_self_test_objects, VoxObject};
use crate::voxel::world::{wm, World};

const RENDER_SCALES: [f32; 4] = [1.0, 0.75, 0.6, 0.5];
const PICK_REACH: f32 = 800.0;

struct Game {
    world: World,
    objects: Vec<VoxObject>,
    r: Box<Renderer>,
    post: PostFx,
    cam: FlyCamera,
    prev_cam: FlyCamera,
    frame: u32,
    keys: MoveInput,
    captured: bool,
    start: Instant,
    last: Instant,
    /// Smoothed frame time (ms) and the last title refresh.
    frame_ms: f32,
    title_at: Instant,
    brush_radius: f32,
    place_mat: u8,
}

enum Phase {
    Loading { thread: Option<std::thread::JoinHandle<(World, Vec<VoxObject>)>>, progress: Arc<AtomicI32> },
    Game(Box<Game>),
}

struct Running {
    window: Arc<Window>,
    ctx: Context,
    surface: vk::SurfaceKHR,
    sc: Swapchain,
    acquire: [vk::Semaphore; 2],
    fences: [vk::Fence; 2],
    cmds: [vk::CommandBuffer; 2],
    slot: usize,
    need_resize: bool,
    phase: Phase,
}

struct App {
    validation: bool,
    size: (u32, u32),
    frame_limit: u32,
    run: Option<Running>,
    exit_code: i32,
}

pub fn run(validation: bool, w: u32, h: u32, frames: u32) -> i32 {
    let el = match EventLoop::new() {
        Ok(e) => e,
        Err(e) => {
            log::error!(target: "app", "cannot open a window: {e}");
            return 1;
        }
    };
    el.set_control_flow(ControlFlow::Poll);
    let mut app = App { validation, size: (w, h), frame_limit: frames, run: None, exit_code: 0 };
    if let Err(e) = el.run_app(&mut app) {
        log::error!(target: "app", "event loop: {e}");
        return 1;
    }
    app.exit_code
}

impl ApplicationHandler for App {
    fn resumed(&mut self, el: &ActiveEventLoop) {
        if self.run.is_some() {
            return;
        }
        let attrs = Window::default_attributes()
            .with_title("VoxelForge")
            .with_inner_size(winit::dpi::PhysicalSize::new(self.size.0, self.size.1));
        let window = match el.create_window(attrs) {
            Ok(w) => Arc::new(w),
            Err(e) => {
                log::error!(target: "app", "create window: {e}");
                self.exit_code = 1;
                el.exit();
                return;
            }
        };
        if cvars::APP_FULLSCREEN.get() {
            window.set_fullscreen(Some(Fullscreen::Borderless(None)));
        }
        let dh = window.display_handle().expect("display handle").as_raw();
        let wh = window.window_handle().expect("window handle").as_raw();
        let exts = ash_window::enumerate_required_extensions(dh).expect("surface extensions").to_vec();
        let probe = |entry: &ash::Entry, inst: &ash::Instance| unsafe {
            ash_window::create_surface(entry, inst, dh, wh, None).expect("create surface")
        };
        let (ctx, surface) = Context::new(&exts, self.validation, Some(&probe));
        let sz = window.inner_size();
        let sc = Swapchain::new(&ctx, surface, (sz.width, sz.height), present_mode_from_cvar());
        let acquire = [ctx.create_semaphore(), ctx.create_semaphore()];
        let fences = [ctx.create_fence(true), ctx.create_fence(true)];
        let cmds = [ctx.alloc_cmd(), ctx.alloc_cmd()];
        // The world generates on a worker while the window shows progress.
        let progress = Arc::new(AtomicI32::new(0));
        let p2 = progress.clone();
        let thread = std::thread::spawn(move || {
            let mut world = World::default();
            world.gen_progress = Some(p2);
            world.generate(cvars::WORLD_SEED.get() as u32);
            world.gen_progress = None;
            let objects = build_self_test_objects(&world);
            (world, objects)
        });
        self.run = Some(Running {
            window,
            ctx,
            surface,
            sc,
            acquire,
            fences,
            cmds,
            slot: 0,
            need_resize: false,
            phase: Phase::Loading { thread: Some(thread), progress },
        });
    }

    fn window_event(&mut self, el: &ActiveEventLoop, _id: WindowId, ev: WindowEvent) {
        let Some(run) = self.run.as_mut() else { return };
        match ev {
            WindowEvent::CloseRequested => el.exit(),
            WindowEvent::Resized(_) => run.need_resize = true,
            WindowEvent::RedrawRequested => {
                if !run.frame() {
                    el.exit();
                } else if self.frame_limit > 0 {
                    if let Phase::Game(g) = &run.phase {
                        if g.frame >= self.frame_limit {
                            el.exit();
                        }
                    }
                }
            }
            WindowEvent::Focused(false) => run.set_captured(false),
            WindowEvent::KeyboardInput { event, .. } => {
                let PhysicalKey::Code(code) = event.physical_key else { return };
                let down = event.state == ElementState::Pressed;
                if down && !event.repeat && code == KeyCode::Escape {
                    if run.captured() {
                        run.set_captured(false);
                    } else {
                        el.exit();
                    }
                    return;
                }
                run.key(code, down, event.repeat);
            }
            WindowEvent::MouseInput { state: ElementState::Pressed, button, .. } => run.click(button),
            WindowEvent::MouseWheel { delta, .. } => {
                let dy = match delta {
                    MouseScrollDelta::LineDelta(_, y) => y,
                    MouseScrollDelta::PixelDelta(p) => p.y as f32 / 40.0,
                };
                let s = (cvars::IN_FLYSPEED.get() * 1.15f32.powf(dy)).clamp(5.0, 800.0);
                cvars::IN_FLYSPEED.set(s);
            }
            _ => {}
        }
    }

    fn device_event(&mut self, _el: &ActiveEventLoop, _id: DeviceId, ev: DeviceEvent) {
        if let (Some(run), DeviceEvent::MouseMotion { delta }) = (self.run.as_mut(), ev) {
            if let Phase::Game(g) = &mut run.phase {
                if g.captured {
                    let s = cvars::IN_SENSITIVITY.get();
                    let dy = if cvars::IN_INVERT_Y.get() { -delta.1 } else { delta.1 };
                    g.cam.look(delta.0 as f32 * s, dy as f32 * s);
                }
            }
        }
    }

    fn about_to_wait(&mut self, _el: &ActiveEventLoop) {
        if let Some(run) = &self.run {
            run.window.request_redraw();
        }
    }

    fn exiting(&mut self, _el: &ActiveEventLoop) {
        if let Some(mut run) = self.run.take() {
            run.destroy();
        }
    }
}

impl Running {
    fn captured(&self) -> bool {
        matches!(&self.phase, Phase::Game(g) if g.captured)
    }

    fn set_captured(&mut self, on: bool) {
        let Phase::Game(g) = &mut self.phase else { return };
        g.captured = on;
        if on {
            let ok = self.window.set_cursor_grab(CursorGrabMode::Locked).or_else(|_| self.window.set_cursor_grab(CursorGrabMode::Confined));
            if ok.is_err() {
                log::warn!(target: "app", "cannot grab the mouse");
            }
        } else {
            let _ = self.window.set_cursor_grab(CursorGrabMode::None);
        }
        self.window.set_cursor_visible(!on);
        g.keys = MoveInput::default();
    }

    fn key(&mut self, code: KeyCode, down: bool, repeat: bool) {
        let Phase::Game(g) = &mut self.phase else { return };
        let k = &mut g.keys;
        match code {
            KeyCode::KeyW => k.forward = down,
            KeyCode::KeyS => k.back = down,
            KeyCode::KeyA => k.left = down,
            KeyCode::KeyD => k.right = down,
            KeyCode::Space => k.up = down,
            KeyCode::ControlLeft | KeyCode::KeyC => k.down = down,
            KeyCode::ShiftLeft => k.fast = down,
            _ => {}
        }
        if !down || repeat {
            return;
        }
        match code {
            KeyCode::F3 => cvars::UI_OVERLAY.set(!cvars::UI_OVERLAY.get()),
            KeyCode::F4 => cvars::R_DEBUGVIEW.set((cvars::R_DEBUGVIEW.get() + 1) % 10),
            KeyCode::F6 => {
                let cur = cvars::R_SCALE.get();
                let i = RENDER_SCALES.iter().position(|&s| (s - cur).abs() < 0.01).map_or(0, |i| (i + 1) % RENDER_SCALES.len());
                cvars::R_SCALE.set(RENDER_SCALES[i]);
                self.need_resize = true;
            }
            KeyCode::F10 => {
                let Phase::Game(g) = &mut self.phase else { return };
                g.r.reload_shaders(&self.ctx);
            }
            KeyCode::F11 => {
                let fs = self.window.fullscreen().is_none();
                self.window.set_fullscreen(fs.then_some(Fullscreen::Borderless(None)));
                cvars::APP_FULLSCREEN.set(fs);
            }
            KeyCode::KeyL => {
                let std = cvars::R_LIGHTING.is("standard");
                cvars::R_LIGHTING.set(if std { "raytraced" } else { "standard" });
                log::info!(target: "app", "lighting: {}", cvars::R_LIGHTING.get());
            }
            KeyCode::KeyN => {
                let none = cvars::R_DENOISER.is("none");
                cvars::R_DENOISER.set(if none { "svgf" } else { "none" });
                log::info!(target: "app", "denoiser: {}", cvars::R_DENOISER.get());
            }
            KeyCode::BracketLeft => cvars::R_EXPOSURE.set(cvars::R_EXPOSURE.get() / 1.15),
            KeyCode::BracketRight => cvars::R_EXPOSURE.set(cvars::R_EXPOSURE.get() * 1.15),
            KeyCode::Digit1
            | KeyCode::Digit2
            | KeyCode::Digit3
            | KeyCode::Digit4
            | KeyCode::Digit5
            | KeyCode::Digit6
            | KeyCode::Digit7
            | KeyCode::Digit8
            | KeyCode::Digit9 => {
                let Phase::Game(g) = &mut self.phase else { return };
                const KEYS: [KeyCode; 9] = [
                    KeyCode::Digit1, KeyCode::Digit2, KeyCode::Digit3, KeyCode::Digit4, KeyCode::Digit5, KeyCode::Digit6,
                    KeyCode::Digit7, KeyCode::Digit8, KeyCode::Digit9,
                ];
                const MATS: [u8; 9] = [wm::STONE, wm::DIRT, wm::GRASS, wm::SAND, wm::WOOD, wm::BRICKM, wm::LAMP, wm::WATER, wm::GOLD];
                let i = KEYS.iter().position(|&k| k == code).unwrap_or(0);
                g.place_mat = MATS[i];
                log::info!(target: "app", "place material {}", g.place_mat);
            }
            KeyCode::Equal => {
                let Phase::Game(g) = &mut self.phase else { return };
                g.brush_radius = (g.brush_radius * 1.25).min(64.0);
            }
            KeyCode::Minus => {
                let Phase::Game(g) = &mut self.phase else { return };
                g.brush_radius = (g.brush_radius / 1.25).max(1.0);
            }
            _ => {}
        }
    }

    fn click(&mut self, button: MouseButton) {
        if !self.captured() {
            self.set_captured(true);
            return;
        }
        let Phase::Game(g) = &mut self.phase else { return };
        // Aim through water; carve (left) or place (right) a sphere where the ray hits.
        let h = g.world.raycast(g.cam.pos, g.cam.forward(), PICK_REACH, wm::WATER);
        if !h.hit {
            return;
        }
        let rad = g.brush_radius;
        let c = Vec3::new(h.cell[0] as f32 + 0.5, h.cell[1] as f32 + 0.5, h.cell[2] as f32 + 0.5);
        match button {
            MouseButton::Left => g.world.fill_sphere(c.x, c.y, c.z, rad, wm::AIR),
            MouseButton::Right => {
                let n = Vec3::new(h.normal[0] as f32, h.normal[1] as f32, h.normal[2] as f32);
                let p = c + n * rad;
                g.world.fill_sphere(p.x, p.y, p.z, rad, g.place_mat);
            }
            _ => return,
        }
        let t = Instant::now();
        g.r.gpu.sync(&self.ctx, &mut g.world);
        log::info!(target: "edit", "edit at {:?} r {rad}: sync {:.2} ms", h.cell, t.elapsed().as_secs_f64() * 1e3);
    }

    /// Render sizes for the swapchain extent and r.scale.
    fn render_size(&self) -> (u32, u32) {
        let s = cvars::R_SCALE.get();
        let e = self.sc.extent;
        (((e.width as f32 * s) as u32).max(64), ((e.height as f32 * s) as u32).max(64))
    }

    /// One frame. Returns false to quit.
    fn frame(&mut self) -> bool {
        let dev = self.ctx.device.clone();
        // Wait for this slot's previous frame (its command buffer + per-slot resources).
        let slot = self.slot;
        self.ctx.wait_fence(self.fences[slot]);
        if self.need_resize {
            let sz = self.window.inner_size();
            if sz.width == 0 || sz.height == 0 {
                return true; // minimised
            }
            self.sc.recreate(&self.ctx, self.surface, (sz.width, sz.height));
            let (rw, rh) = self.render_size();
            if let Phase::Game(g) = &mut self.phase {
                g.r.resize(&self.ctx, rw, rh, rw, rh);
                g.post.resize(&self.ctx, &g.r);
            }
            self.need_resize = false;
        }
        // Loading: finish the world when the worker is done (then build the renderer).
        if let Phase::Loading { thread, progress } = &mut self.phase {
            if thread.as_ref().is_some_and(|t| t.is_finished()) {
                let (mut world, mut objects) = thread.take().unwrap().join().expect("world generation panicked");
                let (rw, rh) = self.render_size();
                let r = Renderer::new(&self.ctx, &mut world, &mut objects, rw, rh, rw, rh);
                let post = PostFx::new(&self.ctx, &r);
                let cam = default_camera(&world);
                let now = Instant::now();
                self.phase = Phase::Game(Box::new(Game {
                    world,
                    objects,
                    r,
                    post,
                    cam,
                    prev_cam: cam,
                    frame: 0,
                    keys: MoveInput::default(),
                    captured: false,
                    start: now,
                    last: now,
                    frame_ms: 16.0,
                    title_at: now,
                    brush_radius: 6.0,
                    place_mat: wm::STONE,
                }));
                self.window.set_title("VoxelForge - click to look around, Esc to free the mouse");
            } else {
                let p = progress.load(Ordering::Relaxed) as f32 / 1000.0;
                return self.present_clear([0.05 + 0.15 * p, 0.08 + 0.25 * p, 0.12 + 0.4 * p, 1.0]);
            }
        }
        let Phase::Game(g) = &mut self.phase else { return true };
        g.r.read_timings(&self.ctx, slot);
        let sw = self.ctx.swapchain_fn.as_ref().unwrap();
        let (img, suboptimal) = match unsafe { sw.acquire_next_image(self.sc.handle, u64::MAX, self.acquire[slot], vk::Fence::null()) } {
            Ok(v) => v,
            Err(vk::Result::ERROR_OUT_OF_DATE_KHR) => {
                self.need_resize = true;
                return true;
            }
            Err(e) => crate::log::fatal("vk", &format!("acquire: {e:?}")),
        };
        if suboptimal {
            self.need_resize = true;
        }
        vkc(unsafe { dev.reset_fences(&[self.fences[slot]]) }, "reset fence");

        // --- simulation -----------------------------------------------------------------------
        let now = Instant::now();
        let fixed = cvars::APP_TEST_DT.get();
        let dt = if fixed > 0.0 { fixed } else { (now - g.last).as_secs_f32().min(0.1) };
        g.last = now;
        g.frame_ms += (dt * 1e3 - g.frame_ms) * 0.1;
        g.cam.move_speed = cvars::IN_FLYSPEED.get();
        if g.captured {
            g.cam.fly(&g.keys, dt);
        }
        let time = if cvars::R_TIME_FREEZE.get() { 0.0 } else { (now - g.start).as_secs_f32() % 3600.0 };
        if cvars::R_TIME_DAYLENGTH.get() > 0.0 {
            let h = (cvars::R_TIME_HOUR.get() + dt * 24.0 / (cvars::R_TIME_DAYLENGTH.get() * 60.0)) % 24.0;
            cvars::R_TIME_HOUR.set(h);
        }
        apply_time_of_day();
        let exposure = cvars::R_EXPOSURE.get();

        // --- render ------------------------------------------------------------------------------
        let r = &mut g.r;
        r.update_lights(&g.world, &g.objects, g.cam.pos, time);
        r.set_frame_uniforms(&g.cam, &g.prev_cam, g.frame, exposure, time, [0.0, 0.0]);
        r.update_sets(&self.ctx, slot);
        let cb = self.cmds[slot];
        vkc(unsafe { dev.reset_command_buffer(cb, vk::CommandBufferResetFlags::empty()) }, "reset cmd");
        self.ctx.begin_cmd(cb);
        // The previous frame may still execute: its writes visible first (shared histories).
        memory_barrier(&dev, cb, Stage::ALL_COMMANDS, Access::MEMORY_WRITE, Stage::ALL_COMMANDS, Access::MEMORY_READ | Access::MEMORY_WRITE);
        let rec = Rec { ctx: &self.ctx, cb };
        let (w, h) = (r.w, r.h);
        if r.standard_mode() || cvars::R_DENOISER.is("none") {
            if r.standard_mode() {
                r.record_standard(&rec);
            } else {
                r.record_raw(&rec);
            }
            crate::gfx::context::pass_barrier(&dev, cb);
            r.record_torch_fx(&rec, 1, w, h);
            g.post.prepare(&dev, cb, 1, dt);
            g.post.tonemap(&dev, cb, 1, w, h, exposure, 0, 1 << 30, -1);
            r.stamp(&rec, "tonemap");
        } else {
            r.record(&rec);
        }
        r.record_dust(&rec, w, h);
        // out_tex -> swapchain (Y-flip: row 0 is the bottom), then present.
        let simg = self.sc.images[img as usize];
        r.out_to_transfer_src(&dev, cb);
        image_barrier(&dev, cb, simg, Stage::TOP_OF_PIPE, Access::NONE, Stage::BLIT, Access::TRANSFER_WRITE,
                      vk::ImageLayout::UNDEFINED, vk::ImageLayout::TRANSFER_DST_OPTIMAL);
        let e = self.sc.extent;
        let blit = [vk::ImageBlit::default()
            .src_subresource(COLOR_LAYERS)
            .dst_subresource(COLOR_LAYERS)
            .src_offsets([vk::Offset3D { x: 0, y: 0, z: 0 }, vk::Offset3D { x: w as i32, y: h as i32, z: 1 }])
            .dst_offsets([vk::Offset3D { x: 0, y: e.height as i32, z: 0 }, vk::Offset3D { x: e.width as i32, y: 0, z: 1 }])];
        unsafe {
            dev.cmd_blit_image(cb, r.out_image().image, vk::ImageLayout::TRANSFER_SRC_OPTIMAL, simg, vk::ImageLayout::TRANSFER_DST_OPTIMAL,
                               &blit, vk::Filter::LINEAR)
        };
        r.stamp(&rec, "blit");
        image_barrier(&dev, cb, simg, Stage::BLIT, Access::TRANSFER_WRITE, Stage::BOTTOM_OF_PIPE, Access::NONE,
                      vk::ImageLayout::TRANSFER_DST_OPTIMAL, vk::ImageLayout::PRESENT_SRC_KHR);
        r.out_to_general(&dev, cb);
        g.prev_cam = g.cam;
        g.frame += 1;
        self.submit_present(cb, img, slot, Stage::BLIT);
        self.slot ^= 1;
        self.update_title();
        if cvars::APP_FPSCAP.get() > 0 {
            let budget = std::time::Duration::from_secs_f64(1.0 / cvars::APP_FPSCAP.get() as f64);
            let spent = now.elapsed();
            if spent < budget {
                std::thread::sleep(budget - spent);
            }
        }
        true
    }

    fn submit_present(&mut self, cb: vk::CommandBuffer, img: u32, slot: usize, wait_stage: vk::PipelineStageFlags2) {
        let dev = &self.ctx.device;
        vkc(unsafe { dev.end_command_buffer(cb) }, "end cmd");
        let wait = [vk::SemaphoreSubmitInfo::default().semaphore(self.acquire[slot]).stage_mask(wait_stage)];
        let sig = [vk::SemaphoreSubmitInfo::default().semaphore(self.sc.render_finished[img as usize]).stage_mask(Stage::ALL_COMMANDS)];
        let cbi = [vk::CommandBufferSubmitInfo::default().command_buffer(cb)];
        let si = [vk::SubmitInfo2::default().wait_semaphore_infos(&wait).command_buffer_infos(&cbi).signal_semaphore_infos(&sig)];
        vkc(unsafe { dev.queue_submit2(self.ctx.queue, &si, self.fences[slot]) }, "submit");
        let sems = [self.sc.render_finished[img as usize]];
        let scs = [self.sc.handle];
        let idx = [img];
        let pi = vk::PresentInfoKHR::default().wait_semaphores(&sems).swapchains(&scs).image_indices(&idx);
        match unsafe { self.ctx.swapchain_fn.as_ref().unwrap().queue_present(self.ctx.queue, &pi) } {
            Ok(false) => {}
            Ok(true) | Err(vk::Result::ERROR_OUT_OF_DATE_KHR) => self.need_resize = true,
            Err(e) => crate::log::fatal("vk", &format!("present: {e:?}")),
        }
    }

    /// The loading screen: the whole window in one colour (brightening with progress).
    fn present_clear(&mut self, col: [f32; 4]) -> bool {
        let slot = self.slot;
        let sw = self.ctx.swapchain_fn.as_ref().unwrap();
        let img = match unsafe { sw.acquire_next_image(self.sc.handle, u64::MAX, self.acquire[slot], vk::Fence::null()) } {
            Ok((i, _)) => i,
            Err(_) => {
                self.need_resize = true;
                return true;
            }
        };
        let dev = self.ctx.device.clone();
        vkc(unsafe { dev.reset_fences(&[self.fences[slot]]) }, "reset fence");
        let cb = self.cmds[slot];
        vkc(unsafe { dev.reset_command_buffer(cb, vk::CommandBufferResetFlags::empty()) }, "reset cmd");
        self.ctx.begin_cmd(cb);
        let simg = self.sc.images[img as usize];
        image_barrier(&dev, cb, simg, Stage::TOP_OF_PIPE, Access::NONE, Stage::CLEAR, Access::TRANSFER_WRITE,
                      vk::ImageLayout::UNDEFINED, vk::ImageLayout::TRANSFER_DST_OPTIMAL);
        unsafe { dev.cmd_clear_color_image(cb, simg, vk::ImageLayout::TRANSFER_DST_OPTIMAL, &vk::ClearColorValue { float32: col }, &[COLOR_RANGE]) };
        image_barrier(&dev, cb, simg, Stage::CLEAR, Access::TRANSFER_WRITE, Stage::BOTTOM_OF_PIPE, Access::NONE,
                      vk::ImageLayout::TRANSFER_DST_OPTIMAL, vk::ImageLayout::PRESENT_SRC_KHR);
        self.submit_present(cb, img, slot, Stage::CLEAR);
        self.slot ^= 1;
        std::thread::sleep(std::time::Duration::from_millis(16));
        true
    }

    fn update_title(&mut self) {
        let Phase::Game(g) = &mut self.phase else { return };
        if g.title_at.elapsed().as_secs_f32() < 0.5 {
            return;
        }
        g.title_at = Instant::now();
        let gpu: f32 = g.r.pass_ms.iter().map(|p| p.1).sum();
        let mut t = format!(
            "VoxelForge  {:.0} fps ({:.1} ms, GPU {:.1} ms)  {}x{}  {}{}",
            1000.0 / g.frame_ms.max(0.01),
            g.frame_ms,
            gpu,
            g.r.w,
            g.r.h,
            cvars::R_LIGHTING.get(),
            if cvars::R_LIGHTING.is("raytraced") { format!("/{}", cvars::R_DENOISER.get()) } else { String::new() },
        );
        if cvars::UI_OVERLAY.get() {
            let p = g.cam.pos;
            t += &format!("  pos {:.0},{:.0},{:.0} yaw {:.2} pitch {:.2}  brush r{:.0} mat {}", p.x, p.y, p.z, g.cam.yaw, g.cam.pitch,
                          g.brush_radius, g.place_mat);
            let top: Vec<String> = g.r.pass_ms.iter().map(|(n, ms)| format!("{n} {ms:.2}")).collect();
            t += &format!("  [{}]", top.join(", "));
        }
        self.window.set_title(&t);
    }

    fn destroy(&mut self) {
        self.ctx.wait_idle();
        if let Phase::Game(g) = &mut self.phase {
            g.post.destroy(&self.ctx);
            g.r.destroy(&self.ctx);
        }
        let dev = &self.ctx.device;
        unsafe {
            for s in self.acquire {
                dev.destroy_semaphore(s, None);
            }
            for f in self.fences {
                dev.destroy_fence(f, None);
            }
            dev.free_command_buffers(self.ctx.cmd_pool, &self.cmds);
        }
        self.sc.destroy(&self.ctx);
        if let Some(sf) = &self.ctx.surface_fn {
            unsafe { sf.destroy_surface(self.surface, None) };
        }
        self.ctx.destroy();
    }
}

/// r.time.hour drives the sun whenever it changes (sun_from_hour); the sun sliders still work
/// between changes.
fn apply_time_of_day() {
    use std::sync::atomic::AtomicU32;
    static SEEN: AtomicU32 = AtomicU32::new(0);
    let v = crate::cvar::CVar::version(&cvars::R_TIME_HOUR);
    if SEEN.swap(v, Ordering::Relaxed) != v {
        let (el, az) = crate::gfx::renderer::frame::sun_from_hour(cvars::R_TIME_HOUR.get());
        cvars::R_SUN_ELEVATION.set(el);
        cvars::R_SUN_AZIMUTH.set(az);
    }
}
