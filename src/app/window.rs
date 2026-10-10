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
use crate::objects::VoxObject;
use crate::fluid::FluidSim;
use super::endless::{origin_for, Endless, ORIGIN_STEP};
use super::saves;
use super::ui::hud::{self, HudInfo};
use super::ui::menu::{Menu, MenuAction, Screen};
use super::ui::{Gui, ToastKind};
use super::scene_edit::{brush_edit, brush_preview_pc, brush_target, scene_raycast, BrushOp, BrushTarget, EditEnv, PREVIEW_CARVE, PREVIEW_PLACE};
use crate::edit::undo::UndoStack;
use crate::physics::{self, BodyState, Physics};
use crate::player::{self as pl, Player, PlayerInput, PlayerTuning, PlayerWorld};
use crate::voxel::world::{wm, World};

const RENDER_SCALES: [f32; 4] = [1.0, 0.75, 0.6, 0.5];
/// Physics runs at a fixed 60 Hz; bodies render interpolated between the last two steps.
const PHYS_DT: f64 = 1.0 / 60.0;
/// Steps per frame at most: beyond that the simulation slows instead of spiralling.
const MAX_PHYS_STEPS: u32 = 4;

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
    place_mat: u8,
    /// Edit undo history (Ctrl+Z / Ctrl+Y).
    undo: UndoStack,
    /// Mouse buttons held, pressed since the last frame, and the dab repeat timer.
    lmb: bool,
    rmb: bool,
    hit_l: bool,
    hit_r: bool,
    repeat: f32,
    /// Where the brush lands this frame (preview + edit).
    target: Option<BrushTarget>,
    /// A status line shown in the title for a few seconds (saves, errors).
    notice: Option<(String, Instant)>,
    /// Toasts for the UI (taken each frame).
    toasts: Vec<(String, ToastKind)>,
    player: Player,
    /// Where the player put the eye last frame: anything else moving the camera hands it over.
    last_eye: Vec3,
    /// Last frame was the free camera (Tab / in.walk 0).
    was_noclip: bool,
    /// Space went down this frame (a jump).
    space_pressed: bool,
    phys: Physics,
    phys_acc: f64,
    /// Body states at the previous physics step (render interpolation).
    phys_prev: Vec<BodyState>,
    phys_was_on: bool,
    /// The world's water; its voxel changes reach the GPU at most every 100 ms (a sync
    /// re-uploads the touched bricks and rebuilds their distance field).
    fluid: FluidSim,
    fluid_dirty: bool,
    fluid_synced: Instant,
    /// An endless world's streaming session (None: a fixed-size world).
    endless: Option<Endless>,
}

/// What the loading worker hands over.
type StartupWorld = super::Startup;

enum Phase {
    Loading { thread: Option<std::thread::JoinHandle<StartupWorld>>, progress: Arc<AtomicI32> },
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
    /// Keyboard modifiers held (Ctrl+Z...).
    mods: winit::keyboard::ModifiersState,
    /// The UI (dropped before the Vulkan context).
    gui: Option<Gui>,
    menu: Menu,
    /// A menu choice, carried out at the start of the next frame (never while a frame's
    /// commands still reference the world it would replace).
    pending: Option<MenuAction>,
}

struct App {
    validation: bool,
    size: (u32, u32),
    frame_limit: u32,
    load: Option<String>,
    run: Option<Running>,
    exit_code: i32,
}

pub fn run(validation: bool, w: u32, h: u32, frames: u32, load: Option<String>) -> i32 {
    let el = match EventLoop::new() {
        Ok(e) => e,
        Err(e) => {
            log::error!(target: "app", "cannot open a window: {e}");
            return 1;
        }
    };
    el.set_control_flow(ControlFlow::Poll);
    let mut app = App { validation, size: (w, h), frame_limit: frames, load, run: None, exit_code: 0 };
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
        let gui = Gui::new(&ctx, &window, sc.format);
        // The world generates (or loads) on a worker while the window shows progress.
        let progress = Arc::new(AtomicI32::new(0));
        let p2 = progress.clone();
        let load = self.load.take();
        let thread = std::thread::spawn(move || super::startup_world(load.as_deref(), Some(p2)));
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
            mods: Default::default(),
            gui: Some(gui),
            menu: Menu::default(),
            pending: None,
            phase: Phase::Loading { thread: Some(thread), progress },
        });
    }

    fn window_event(&mut self, el: &ActiveEventLoop, _id: WindowId, ev: WindowEvent) {
        let Some(run) = self.run.as_mut() else { return };
        // The UI sees every event (it keeps its own input state); while the game has the mouse
        // and no menu is open, the game gets them.
        let ui_used = match run.gui.as_mut() {
            Some(g) => g.on_event(&run.window, &ev),
            None => false,
        };
        let ui_owns = run.menu.is_open() || (!run.captured() && ui_used);
        match ev {
            WindowEvent::CloseRequested => el.exit(),
            WindowEvent::Resized(_) => run.need_resize = true,
            WindowEvent::RedrawRequested => {
                if !run.frame() {
                    el.exit();
                } else if let Phase::Game(g) = &run.phase
                    && self.frame_limit > 0
                    && g.frame >= self.frame_limit
                {
                    el.exit();
                }
            }
            WindowEvent::Focused(false) => run.set_captured(false),
            WindowEvent::KeyboardInput { event, .. } => {
                let PhysicalKey::Code(code) = event.physical_key else { return };
                let down = event.state == ElementState::Pressed;
                if down && !event.repeat && code == KeyCode::Escape {
                    // Esc: the pause menu; in a menu, back one level (closing it resumes).
                    if run.menu.is_open() {
                        if run.menu.back() {
                            run.set_captured(true);
                        }
                    } else if matches!(run.phase, Phase::Game(_)) {
                        run.set_captured(false);
                        run.menu.open(Screen::Pause);
                    }
                    return;
                }
                if ui_owns {
                    return;
                }
                run.key(code, down, event.repeat);
            }
            WindowEvent::MouseInput { state, button, .. } if !ui_owns => run.mouse(button, state == ElementState::Pressed),
            WindowEvent::ModifiersChanged(m) => run.mods = m.state(),
            WindowEvent::MouseWheel { delta, .. } if !ui_owns => {
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
        if let (Some(run), DeviceEvent::MouseMotion { delta }) = (self.run.as_mut(), ev)
            && let Phase::Game(g) = &mut run.phase
            && g.captured
        {
            let s = cvars::IN_SENSITIVITY.get();
            let dy = if cvars::IN_INVERT_Y.get() { -delta.1 } else { delta.1 };
            g.cam.look(delta.0 as f32 * s, dy as f32 * s);
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

/// A save / load request.
enum SaveLoad {
    QuickSave,
    QuickLoad,
    SaveAs(String),
    Load(std::path::PathBuf),
}

fn tuning() -> PlayerTuning {
    let mut t = PlayerTuning::default();
    t.height = cvars::PLAYER_HEIGHT.get();
    t.crouch_height = (t.height - 3.0).max(4.0);
    t.eye = cvars::PLAYER_EYE.get().min(t.height - 0.8);
    t.crouch_eye = (t.eye - 3.0).max(2.0);
    t.width = cvars::PLAYER_WIDTH.get();
    t.gravity = cvars::PLAYER_GRAVITY.get();
    t.jump_apex = cvars::PLAYER_JUMP.get();
    t.walk = cvars::PLAYER_WALK.get();
    t.sprint = cvars::PLAYER_SPRINT.get();
    t.crouch = t.walk * 0.45; // ~1 m/s
    t.step = cvars::PLAYER_STEP.get();
    t.fly = cvars::IN_FLYSPEED.get();
    t
}

impl Game {
    fn edit_env<'a>(&'a mut self, ctx: &'a Context) -> EditEnv<'a> {
        EditEnv { ctx, r: &mut self.r, world: &mut self.world, objects: &mut self.objects, fluid: &mut self.fluid }
    }

    /// Play-mode building: LMB carves, RMB places (held: a dab every 0.2 s), never into the
    /// player. Also refreshes the brush target for the preview.
    fn build(&mut self, ctx: &Context, dt: f32) {
        const REPEAT: f32 = 0.2;
        self.repeat -= dt;
        let on = self.captured && cvars::R_BUILD.get();
        self.target = if on { brush_target(&self.world, &self.objects, self.cam.pos, self.cam.forward()) } else { None };
        let (hit_l, hit_r) = (std::mem::take(&mut self.hit_l), std::mem::take(&mut self.hit_r));
        let carve = self.lmb || hit_l;
        let place = !carve && (self.rmb || hit_r);
        if !on || (!carve && !place) || (self.repeat > 0.0 && !hit_l && !hit_r) {
            return;
        }
        let Some(t) = self.target else { return };
        self.repeat = REPEAT;
        if place && t.obj.is_none() && cvars::IN_WALK.get() {
            // Not into yourself.
            let tune = tuning();
            if t.place.cells().any(|(x, y, z)| pl::player_overlaps_cell(&self.player, &tune, [x, y, z])) {
                self.notify("You're in the way".into());
                return;
            }
        }
        let (op, mat) = if carve { (BrushOp::Carve, 0) } else { (BrushOp::Place, self.place_mat) };
        let rec = brush_edit(&mut self.edit_env(ctx), &t, op, mat);
        self.undo.push(rec);
        self.phys_prev.clear();
    }

    /// Ctrl+Z / Ctrl+Y.
    fn undo_redo(&mut self, ctx: &Context, redo: bool) {
        let a = if redo { self.undo.redo(&mut self.world, &mut self.objects) } else { self.undo.undo(&mut self.world, &mut self.objects) };
        if !a.any {
            self.notify(format!("Nothing to {}", if redo { "redo" } else { "undo" }));
            return;
        }
        if a.world {
            self.r.gpu.sync(ctx, &mut self.world);
            self.r.note_world_edit(a.lo, a.hi.map(|v| v + 1));
            self.fluid.wake(a.lo[0] - 1, a.lo[1] - 1, a.lo[2] - 1, a.hi[0] + 1, a.hi[1] + 1, a.hi[2] + 1);
        }
        if a.bodies {
            self.r.gpu.init_objects(ctx, &self.world, &mut self.objects);
            self.r.rebind_object_buffers();
            self.phys_prev.clear();
        }
        let (u, r) = self.undo.len();
        self.notify(format!("{}: {} ({u} undo / {r} redo, {} KB)", if redo { "Redo" } else { "Undo" }, a.label, self.undo.bytes() / 1024));
    }

    /// MMB: hold the block under the crosshair.
    fn pick_block(&mut self) {
        let Some(h) = scene_raycast(&self.world, &self.objects, self.cam.pos, self.cam.forward(), 2000.0) else { return };
        let c = h.hit.cell;
        let m = match h.obj {
            Some(i) => self.objects[i].vol.get_voxel(c[0], c[1], c[2]),
            None => self.world.get_voxel(c[0], c[1], c[2]),
        };
        if m != 0 {
            self.place_mat = m;
            self.notify(format!("Holding material {m}"));
        }
    }

    /// Endless worlds: the floating origin, the view distance, chunks in / out around the camera.
    fn stream(&mut self, ctx: &Context) {
        let Some(e) = self.endless.as_mut() else { return };
        // Far out: move the origin (floats stay precise however far you travel).
        let lim = 0.6 * ORIGIN_STEP as f32;
        if self.cam.pos.x.abs() > lim || self.cam.pos.z.abs() > lim {
            let (dx, dz) = (origin_for(self.cam.pos.x), origin_for(self.cam.pos.z));
            e.rebase(&mut self.world, dx, dz);
            let d = Vec3::new(dx as f32, 0.0, dz as f32);
            self.cam.pos -= d;
            self.prev_cam.pos -= d;
            self.player.feet -= d;
            self.last_eye -= d;
            for o in &mut self.objects {
                o.shift_world(dx as f32, dz as f32);
            }
            for p in &mut self.phys_prev {
                p.t -= d;
            }
            self.fluid.shift(dx, dz);
            self.undo.shift(dx, dz);
            self.r.shift_origin(dx, dz);
        }
        if e.apply_view_distance(&mut self.world, self.cam.pos.x, self.cam.pos.z) {
            self.r.gpu.stream_radius = e.stream.radius();
            self.r.reload_scene(ctx, &mut self.world, &mut self.objects);
        }
        e.frame(ctx, &mut self.r, &mut self.world, self.cam.pos, self.cam.forward());
    }

    fn notify(&mut self, msg: String) {
        log::info!(target: "app", "{msg}");
        let kind = if msg.starts_with("Couldn't") { ToastKind::Error } else if msg.starts_with("World saved") || msg.starts_with("Loaded") { ToastKind::Good } else { ToastKind::Info };
        self.toasts.push((msg.clone(), kind));
        self.notice = Some((msg, Instant::now()));
        self.title_at = Instant::now() - std::time::Duration::from_secs(1); // refresh the title now
    }

    /// F5 / F9.
    fn quick_save_load(&mut self, ctx: &Context, save: bool) {
        self.save_load(ctx, if save { SaveLoad::QuickSave } else { SaveLoad::QuickLoad });
    }

    fn save_load(&mut self, ctx: &Context, what: SaveLoad) {
        let mut io = saves::WorldIo { ctx, r: &mut self.r, world: &mut self.world, objects: &mut self.objects, cam: &mut self.cam, fluid: &mut self.fluid, endless: &mut self.endless };
        let saving = matches!(what, SaveLoad::QuickSave | SaveLoad::SaveAs(_));
        let res = match what {
            SaveLoad::QuickSave => saves::quick_save(&mut io),
            SaveLoad::SaveAs(name) => saves::save_world(&mut io, &name),
            SaveLoad::QuickLoad => saves::quick_load(&mut io),
            SaveLoad::Load(path) => saves::load_world(&mut io, &path),
        };
        let msg = if saving {
            match res {
                Ok(n) => format!("World saved: {n}"),
                Err(e) => format!("Couldn't save: {e}"),
            }
        } else {
            match res {
                Ok(n) => {
                    // A new scene: no history to blend from, nothing to undo.
                    self.prev_cam = self.cam;
                    self.undo.clear();
                    self.target = None;
                    self.phys_prev.clear();
                    self.fluid_dirty = false;
                    format!("Loaded {n}")
                }
                Err(e) => format!("Couldn't load: {e}"),
            }
        };
        self.notify(msg);
    }

    /// Fixed-step physics; the GPU gets the bodies interpolated between the last two steps.
    fn simulate(&mut self, ctx: &Context, dt: f32) {
        let on = cvars::GAME_PHYSICS.get();
        if on != self.phys_was_on {
            self.phys_was_on = on;
            if on {
                physics::wake_near(&mut self.objects, Vec3::ZERO, 1e9);
            }
        }
        if !on {
            self.phys_acc = 0.0;
            self.phys_prev.clear();
            return;
        }
        self.phys.foliage_solid = cvars::PHYS_FOLIAGE.get();
        self.phys_acc += dt as f64;
        let mut steps = 0;
        while self.phys_acc >= PHYS_DT && steps < MAX_PHYS_STEPS {
            self.phys_prev.clear();
            self.phys_prev.extend(self.objects.iter().map(BodyState::of));
            self.phys.step(&self.world, &mut self.objects, PHYS_DT as f32);
            self.phys_acc -= PHYS_DT;
            steps += 1;
        }
        if steps == MAX_PHYS_STEPS {
            self.phys_acc = self.phys_acc.min(PHYS_DT);
        }
        // Bodies that fell out of the world or were carved away entirely are removed (they
        // would keep their atlas slots, BVH entries and a place in every ray's object walk).
        let before = self.objects.len();
        self.objects.retain(|o| !(o.pose.t.y < -200.0 || o.mass == 0.0));
        if self.objects.len() != before {
            self.r.gpu.init_objects(ctx, &self.world, &mut self.objects);
            self.r.rebind_object_buffers();
            self.phys_prev.clear();
        }
        if self.phys_prev.len() != self.objects.len() {
            self.phys_prev.clear();
            self.phys_prev.extend(self.objects.iter().map(BodyState::of));
        }
        let a = (self.phys_acc / PHYS_DT) as f32;
        let sim: Vec<_> = self.objects.iter().map(|o| o.pose).collect();
        for (o, p) in self.objects.iter_mut().zip(&self.phys_prev) {
            if o.inv_mass != 0.0 {
                o.pose = physics::interpolated_pose(o, p, a);
            }
        }
        self.r.gpu.update_objects(&self.objects);
        for (o, p) in self.objects.iter_mut().zip(sim) {
            o.pose = p;
        }
    }

    /// The walking player: collision, gravity, steps, swimming; double-tap Space flies.
    fn walk(&mut self, dt: f32) {
        let t = tuning();
        let k = if self.captured { self.keys } else { MoveInput::default() };
        let axis = |a: bool, b: bool| (a as i32 - b as i32) as f32;
        let inp = PlayerInput {
            fwd: axis(k.forward, k.back),
            side: axis(k.right, k.left),
            jump_held: k.up,
            jump_pressed: self.space_pressed && self.captured,
            crouch: k.down,
            sprint: k.fast,
        };
        let mut w = PlayerWorld { world: &self.world, objects: Some(&mut self.objects), bodies_dynamic: cvars::GAME_PHYSICS.get() };
        if (self.cam.pos - self.last_eye).length() > 0.01 || self.was_noclip {
            // Something else moved the camera (start, teleport, the free camera): the player
            // takes over from there, out of the ground, flying if it isn't standing on anything.
            self.player.crouched = false;
            pl::player_place(&mut self.player, self.cam.pos - Vec3::new(0.0, t.eye, 0.0), &t);
            pl::player_unstuck(&mut self.player, &t, &w);
            let mut q = self.player;
            q.feet.y -= 0.5;
            self.player.flying = !pl::player_stuck(&q, &t, &w);
        }
        self.was_noclip = false;
        pl::player_step(&mut self.player, &inp, self.cam.yaw, &t, &mut w, true, false, dt);
        if self.player.feet.y < -50.0 {
            log::info!(target: "app", "fell out of the world");
            let g = pl::player_ground_at(&self.world, self.world.mid_x(), self.world.mid_z());
            pl::player_place(&mut self.player, g, &t);
            pl::player_unstuck(&mut self.player, &t, &w);
        }
        if !cvars::IN_BOB.get() {
            self.player.bob_amp = 0.0;
        }
        self.cam.pos = pl::player_eye(&self.player);
        self.last_eye = self.cam.pos;
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
            KeyCode::Space => {
                let Phase::Game(g) = &mut self.phase else { return };
                if cvars::IN_WALK.get() {
                    g.space_pressed = true;
                    let now = g.start.elapsed().as_secs_f64();
                    if pl::player_space_tap(&mut g.player, now, true) {
                        log::info!(target: "app", "{}", if g.player.flying { "flying (double-tap Space to land)" } else { "walking" });
                    }
                }
            }
            KeyCode::Tab => {
                cvars::IN_WALK.set(!cvars::IN_WALK.get());
                log::info!(target: "app", "{}", if cvars::IN_WALK.get() { "walking" } else { "free camera" });
            }
            KeyCode::KeyP => {
                cvars::GAME_PHYSICS.set(!cvars::GAME_PHYSICS.get());
                log::info!(target: "app", "physics {}", if cvars::GAME_PHYSICS.get() { "running" } else { "paused" });
            }
            KeyCode::F3 => cvars::UI_OVERLAY.set(!cvars::UI_OVERLAY.get()),
            KeyCode::F4 => cvars::R_DEBUGVIEW.set((cvars::R_DEBUGVIEW.get() + 1) % 10),
            KeyCode::F6 => {
                let cur = cvars::R_SCALE.get();
                let i = RENDER_SCALES.iter().position(|&s| (s - cur).abs() < 0.01).map_or(0, |i| (i + 1) % RENDER_SCALES.len());
                cvars::R_SCALE.set(RENDER_SCALES[i]);
                self.need_resize = true;
            }
            KeyCode::F5 | KeyCode::F9 => {
                let Phase::Game(g) = &mut self.phase else { return };
                g.quick_save_load(&self.ctx, code == KeyCode::F5);
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
            KeyCode::Equal | KeyCode::Minus => {
                let s = cvars::R_BRUSH_SIZE.get();
                let step = (s / 4).max(1);
                cvars::R_BRUSH_SIZE.set(if code == KeyCode::Equal { s + step } else { s - step });
            }
            KeyCode::KeyB => {
                cvars::R_BRUSH_SHAPE.set(if cvars::R_BRUSH_SHAPE.is("cube") { "sphere" } else { "cube" });
            }
            KeyCode::KeyF => {
                cvars::R_BUILD.set(!cvars::R_BUILD.get());
                let Phase::Game(g) = &mut self.phase else { return };
                g.notify(format!("Build mode {}", if cvars::R_BUILD.get() { "on" } else { "off" }));
            }
            KeyCode::KeyZ | KeyCode::KeyY if self.mods.control_key() => {
                let redo = code == KeyCode::KeyY || self.mods.shift_key();
                let Phase::Game(g) = &mut self.phase else { return };
                g.undo_redo(&self.ctx, redo);
            }
            _ => {}
        }
    }

    fn mouse(&mut self, button: MouseButton, down: bool) {
        if down && !self.captured() {
            self.set_captured(true);
            return;
        }
        let Phase::Game(g) = &mut self.phase else { return };
        match button {
            MouseButton::Left => {
                g.lmb = down;
                g.hit_l |= down;
            }
            MouseButton::Right => {
                g.rmb = down;
                g.hit_r |= down;
            }
            MouseButton::Middle if down => g.pick_block(),
            _ => {}
        }
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
        if let Some(a) = self.pending.take() {
            match a {
                MenuAction::Quit => return false,
                MenuAction::Resume => {
                    self.menu.back();
                    self.set_captured(true);
                }
                MenuAction::Save(name) => {
                    if let Phase::Game(g) = &mut self.phase {
                        g.save_load(&self.ctx, SaveLoad::SaveAs(name));
                    }
                    self.menu.screen = Screen::Closed;
                    self.set_captured(true);
                }
                MenuAction::Load(path) => {
                    if let Phase::Game(g) = &mut self.phase {
                        g.save_load(&self.ctx, SaveLoad::Load(path));
                    }
                    self.menu.screen = Screen::Closed;
                    self.set_captured(true);
                }
            }
        }
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
                let super::Startup { mut world, mut objects, state, endless } = thread.take().unwrap().join().expect("world generation panicked");
                let (rw, rh) = self.render_size();
                let r = Renderer::new(&self.ctx, &mut world, &mut objects, rw, rh, rw, rh);
                let post = PostFx::new(&self.ctx, &r);
                let mut cam = default_camera(&world);
                let mut fluid = FluidSim::default();
                if let Some(st) = &state {
                    saves::restore_state(st, &mut fluid, &mut cam);
                }
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
                    place_mat: wm::STONE,
                    undo: UndoStack::default(),
                    lmb: false,
                    rmb: false,
                    hit_l: false,
                    hit_r: false,
                    repeat: 0.0,
                    target: None,
                    notice: None,
                    toasts: Vec::new(),
                    player: Player::default(),
                    // Differs from the camera: the first frame places the player under it.
                    last_eye: cam.pos + Vec3::splat(1e6),
                    was_noclip: false,
                    space_pressed: false,
                    phys: Physics::default(),
                    phys_acc: 0.0,
                    phys_prev: Vec::new(),
                    phys_was_on: false,
                    fluid,
                    endless,
                    fluid_dirty: false,
                    fluid_synced: now,
                }));
                self.window.set_title("VoxelForge - click to look around, Esc to free the mouse");
            } else {
                let p = progress.load(Ordering::Relaxed) as f32 / 1000.0;
                return self.present_loading(if p > 0.0 { Some(p) } else { None });
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
        g.simulate(&self.ctx, dt);
        if !cvars::R_TIME_FREEZE.get() && g.fluid.update(&mut g.world, dt) {
            g.fluid_dirty = true;
        }
        if g.fluid_dirty && g.fluid_synced.elapsed().as_secs_f32() >= 0.1 {
            g.r.gpu.sync(&self.ctx, &mut g.world);
            g.fluid_dirty = false;
            g.fluid_synced = Instant::now();
        }
        if cvars::IN_WALK.get() {
            g.walk(dt);
        } else {
            g.was_noclip = true;
            if g.captured {
                g.cam.fly(&g.keys, dt);
            }
        }
        g.space_pressed = false;
        g.stream(&self.ctx);
        g.build(&self.ctx, dt);
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
        if let Some(t) = g.target.filter(|_| cvars::R_BRUSH_PREVIEW.get()) {
            let pc = brush_preview_pc(&t, &g.world.palette, g.place_mat, (w, h), PREVIEW_CARVE | PREVIEW_PLACE);
            r.record_brush_preview(&rec, &pc);
        }
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
        r.out_to_general(&dev, cb);
        // The UI over the frame (it leaves the image ready to present).
        let gpu_ms: f32 = g.r.pass_ms.iter().map(|p| p.1).sum();
        let held = g.world.palette[g.place_mat as usize];
        let info = HudInfo {
            frame_ms: g.frame_ms,
            gpu_ms,
            res: (g.r.w, g.r.h),
            pos: g.cam.pos.to_array(),
            passes: &g.r.pass_ms,
            build: g.captured && cvars::R_BUILD.get(),
            held: g.place_mat,
            held_color: [held.r, held.g, held.b],
            walking: cvars::IN_WALK.get(),
            flying: g.player.flying,
        };
        let gui = self.gui.as_mut().unwrap();
        for (t, k) in g.toasts.drain(..) {
            gui.toasts.push(t, k, None);
        }
        let (menu, playing) = (&mut self.menu, g.captured);
        let mut chosen = None;
        gui.frame(&self.ctx, &self.window, cb, simg, self.sc.views[img as usize], e, slot, |c| {
            if playing && !menu.is_open() {
                if cvars::UI_CROSSHAIR.get() {
                    hud::crosshair(c);
                }
                hud::draw(c, &info);
            }
            if let Some(a) = menu.draw(c) {
                chosen = Some(a);
            }
        });
        if chosen.is_some() {
            self.pending = chosen;
        }
        g.prev_cam = g.cam;
        g.frame += 1;
        self.submit_present(cb, img, slot, Stage::ALL_COMMANDS);
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

    /// The loading screen (the UI over a cleared image).
    fn present_loading(&mut self, progress: Option<f32>) -> bool {
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
        unsafe { dev.cmd_clear_color_image(cb, simg, vk::ImageLayout::TRANSFER_DST_OPTIMAL, &vk::ClearColorValue { float32: [0.047, 0.055, 0.07, 1.0] }, &[COLOR_RANGE]) };
        image_barrier(&dev, cb, simg, Stage::CLEAR, Access::TRANSFER_WRITE, Stage::BLIT, Access::TRANSFER_WRITE,
                      vk::ImageLayout::TRANSFER_DST_OPTIMAL, vk::ImageLayout::TRANSFER_DST_OPTIMAL);
        let what = if cvars::WORLD_ENDLESS.get() { "Loading the world around you" } else { "Generating the world" };
        let gui = self.gui.as_mut().unwrap();
        gui.frame(&self.ctx, &self.window, cb, simg, self.sc.views[img as usize], self.sc.extent, slot, |c| hud::loading(c, what, progress));
        self.submit_present(cb, img, slot, Stage::ALL_COMMANDS);
        self.slot ^= 1;
        std::thread::sleep(std::time::Duration::from_millis(16));
        true
    }

    /// The loading screen without a UI: the whole window in one colour.
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
            t += &format!("  pos {:.0},{:.0},{:.0} yaw {:.2} pitch {:.2}  brush {} {} mat {}", p.x, p.y, p.z, g.cam.yaw, g.cam.pitch,
                          cvars::R_BRUSH_SHAPE.get(), cvars::R_BRUSH_SIZE.get(), g.place_mat);
            let top: Vec<String> = g.r.pass_ms.iter().map(|(n, ms)| format!("{n} {ms:.2}")).collect();
            t += &format!("  [{}]", top.join(", "));
        }
        if let Some((msg, at)) = &g.notice {
            if at.elapsed().as_secs_f32() < 4.0 {
                t = format!("{msg}  |  {t}");
            } else {
                g.notice = None;
            }
        }
        self.window.set_title(&t);
    }

    fn destroy(&mut self) {
        self.ctx.wait_idle();
        self.gui = None; // (its Vulkan objects go before the device)
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
