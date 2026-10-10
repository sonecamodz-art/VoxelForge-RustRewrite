//! The game's UI (egui): menus, settings, saves, the HUD and toasts - drawn over the frame on
//! the swapchain image (dynamic rendering), after the world is blitted there.
//!
//! The look is the C++ engine's: amber on dark, sizes in 1080p pixels scaled with the window.

pub mod hud;
pub mod menu;
pub mod settings;

use std::time::Instant;

use ash::vk;
use egui::{Color32, FontFamily, FontId, TextStyle};
use egui_ash_renderer::{DynamicRendering, Options, RenderMode, Renderer};
use winit::window::Window;

use crate::gfx::context::{image_barrier, Access, Context, Stage};

pub const TEXT: Color32 = Color32::from_rgb(237, 240, 245);
pub const TEXT_DIM: Color32 = Color32::from_rgb(148, 158, 173);
pub const TEXT_FAINT: Color32 = Color32::from_rgb(102, 110, 122);
pub const ACCENT: Color32 = Color32::from_rgb(242, 171, 77); // warm amber
pub const ACCENT_HI: Color32 = Color32::from_rgb(255, 199, 115);
pub const ON_ACCENT: Color32 = Color32::from_rgb(20, 18, 13);
pub const GOOD: Color32 = Color32::from_rgb(120, 210, 140);
pub const ERROR: Color32 = Color32::from_rgb(240, 110, 100);

#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub enum ToastKind {
    Info,
    Good,
    Error,
}

struct Toast {
    text: String,
    kind: ToastKind,
    key: Option<&'static str>,
    at: Instant,
}

/// Short messages at the bottom centre ("World saved"), ~3 s each, newest on top.
#[derive(Default)]
pub struct Toasts {
    list: Vec<Toast>,
}

impl Toasts {
    /// `key`: a new toast replaces a showing one with the same key (mode switches, undo...).
    pub fn push(&mut self, text: impl Into<String>, kind: ToastKind, key: Option<&'static str>) {
        if let Some(k) = key {
            self.list.retain(|t| t.key != Some(k));
        }
        self.list.push(Toast { text: text.into(), kind, key, at: Instant::now() });
        if self.list.len() > 6 {
            self.list.remove(0);
        }
    }

    pub fn draw(&mut self, ctx: &egui::Context) {
        const LIFE: f32 = 3.0;
        self.list.retain(|t| t.at.elapsed().as_secs_f32() < LIFE);
        if self.list.is_empty() {
            return;
        }
        let k = scale(ctx);
        let screen = ctx.content_rect();
        let painter = ctx.layer_painter(egui::LayerId::new(egui::Order::Tooltip, egui::Id::new("toasts")));
        let mut y = screen.bottom() - 150.0 * k;
        for t in self.list.iter().rev() {
            let age = t.at.elapsed().as_secs_f32();
            let alpha = (age / 0.15).min(1.0) * ((LIFE - age) / 0.4).min(1.0);
            let font = FontId::proportional(20.0 * k);
            let g = painter.layout_no_wrap(t.text.clone(), font, TEXT.gamma_multiply(alpha));
            let size = g.size() + egui::vec2(36.0, 18.0) * k;
            let rect = egui::Rect::from_center_size(egui::pos2(screen.center().x, y - size.y * 0.5), size);
            painter.rect_filled(rect, 8.0 * k, Color32::from_black_alpha((190.0 * alpha) as u8));
            let bar = match t.kind {
                ToastKind::Info => ACCENT,
                ToastKind::Good => GOOD,
                ToastKind::Error => ERROR,
            };
            painter.rect_filled(egui::Rect::from_min_size(rect.min, egui::vec2(4.0 * k, size.y)), 2.0, bar.gamma_multiply(alpha));
            painter.galley(rect.min + egui::vec2(18.0, 9.0) * k, g, TEXT);
            y -= size.y + 8.0 * k;
        }
        ctx.request_repaint();
    }
}

/// UI scale: window height / 1080 (the C++ K()).
pub fn scale(ctx: &egui::Context) -> f32 {
    (ctx.content_rect().height() / 1080.0).max(0.5)
}

/// Fira Sans from the data folder when it's there (the C++ engine's menu font), else egui's.
fn install_fonts(ctx: &egui::Context) {
    let dir = crate::paths::data_path("assets/fonts");
    let mut fonts = egui::FontDefinitions::default();
    let mut any = false;
    for (name, file) in [("fira", "FiraSans-Regular.ttf"), ("fira-bold", "FiraSans-SemiBold.ttf")] {
        if let Ok(bytes) = std::fs::read(dir.join(file)) {
            fonts.font_data.insert(name.into(), egui::FontData::from_owned(bytes).into());
            any = true;
        }
    }
    if any {
        if fonts.font_data.contains_key("fira") {
            fonts.families.entry(FontFamily::Proportional).or_default().insert(0, "fira".into());
        }
        if fonts.font_data.contains_key("fira-bold") {
            fonts.families.insert(FontFamily::Name("bold".into()), vec!["fira-bold".into(), "fira".into()]);
        }
        ctx.set_fonts(fonts);
    }
}

/// The menu look, scaled with the window (`height` in points: the styles are set before the
/// frame runs, when the context doesn't know the screen yet).
pub fn apply_style(ctx: &egui::Context, height: f32) {
    let k = (height / 1080.0).max(0.5);
    let mut st = (*ctx.global_style()).clone();
    st.text_styles = [
        (TextStyle::Heading, FontId::proportional(30.0 * k)),
        (TextStyle::Body, FontId::proportional(19.0 * k)),
        (TextStyle::Button, FontId::proportional(19.0 * k)),
        (TextStyle::Small, FontId::proportional(15.0 * k)),
        (TextStyle::Monospace, FontId::monospace(16.0 * k)),
    ]
    .into();
    let v = &mut st.visuals;
    *v = egui::Visuals::dark();
    v.override_text_color = Some(TEXT);
    v.panel_fill = Color32::from_rgba_unmultiplied(14, 16, 20, 235);
    v.window_fill = Color32::from_rgba_unmultiplied(18, 20, 25, 245);
    v.extreme_bg_color = Color32::from_white_alpha(14);
    v.selection.bg_fill = ACCENT.gamma_multiply(0.55);
    v.selection.stroke.color = ACCENT_HI;
    v.slider_trailing_fill = true;
    let r = egui::CornerRadius::same((6.0 * k) as u8);
    for (w, fill) in [
        (&mut v.widgets.inactive, Color32::from_white_alpha(15)),
        (&mut v.widgets.hovered, Color32::from_white_alpha(33)),
        (&mut v.widgets.active, ACCENT.gamma_multiply(0.45)),
        (&mut v.widgets.open, Color32::from_white_alpha(20)),
    ] {
        w.bg_fill = fill;
        w.weak_bg_fill = fill;
        w.corner_radius = r;
        w.bg_stroke = egui::Stroke::NONE;
    }
    v.widgets.noninteractive.bg_stroke.color = Color32::from_white_alpha(20);
    v.widgets.hovered.fg_stroke.color = ACCENT_HI;
    v.widgets.active.fg_stroke.color = TEXT; // (also egui's "strong" text colour)
    st.spacing.item_spacing = egui::vec2(10.0, 8.0) * k;
    st.spacing.button_padding = egui::vec2(12.0, 8.0) * k;
    st.spacing.slider_width = 260.0 * k;
    st.spacing.interact_size.y = 30.0 * k;
    ctx.set_global_style(st);
}

/// egui wired to the window and drawn over the swapchain image.
pub struct Gui {
    pub ctx: egui::Context,
    state: egui_winit::State,
    renderer: Renderer<egui_ash_renderer::allocator::DefaultAllocator>,
    /// Textures egui freed: released once the frames that drew them are done.
    to_free: Vec<Vec<egui::TextureId>>,
    queue: vk::Queue,
    pool: vk::CommandPool,
    last_height: f32,
    pub toasts: Toasts,
}

impl Gui {
    pub fn new(ctx: &Context, window: &Window, format: vk::Format) -> Self {
        let egui_ctx = egui::Context::default();
        install_fonts(&egui_ctx);
        let state = egui_winit::State::new(egui_ctx.clone(), egui::ViewportId::ROOT, window, Some(window.scale_factor() as f32), None, None);
        let renderer = Renderer::with_default_allocator(
            &ctx.instance,
            ctx.phys,
            ctx.device.clone(),
            RenderMode::DynamicRendering(DynamicRendering { color_attachment_format: format, depth_attachment_format: None, stencil_attachment_format: None }),
            Options { in_flight_frames: 2, srgb_framebuffer: false, ..Default::default() },
        )
        .unwrap_or_else(|e| crate::log::fatal("ui", &format!("egui renderer: {e}")));
        Self { ctx: egui_ctx, state, renderer, to_free: vec![Vec::new(), Vec::new()], queue: ctx.queue, pool: ctx.cmd_pool, last_height: 0.0, toasts: Toasts::default() }
    }

    /// Feed a window event; true = the UI used it (the game should not).
    pub fn on_event(&mut self, window: &Window, ev: &winit::event::WindowEvent) -> bool {
        self.state.on_window_event(window, ev).consumed
    }

    pub fn wants_pointer(&self) -> bool {
        self.ctx.egui_wants_pointer_input()
    }
    pub fn wants_keyboard(&self) -> bool {
        self.ctx.egui_wants_keyboard_input()
    }

    /// Build this frame's UI with `f`, then record it into `cb` over `image` (TRANSFER_DST on
    /// entry, PRESENT_SRC on exit). `slot` = the frame-in-flight slot (its fence was waited).
    #[allow(clippy::too_many_arguments)]
    pub fn frame(&mut self, ctx: &Context, window: &Window, cb: vk::CommandBuffer, image: vk::Image, view: vk::ImageView, extent: vk::Extent2D, slot: usize,
                 f: impl FnMut(&egui::Context)) {
        // What this slot's previous frame freed is done with now.
        let free = std::mem::take(&mut self.to_free[slot & 1]);
        if !free.is_empty() {
            for id in free {
                let _ = self.renderer.free_texture(id);
            }
        }
        let input = self.state.take_egui_input(window);
        let h = input.screen_rect.map_or(0.0, |r| r.height());
        if (h - self.last_height).abs() > 0.5 {
            self.last_height = h;
            let c = self.ctx.clone();
            c.request_repaint();
            apply_style(&c, h);
        }
        let mut f = f;
        let toasts = &mut self.toasts;
        let out = self.ctx.run_ui(input, |ui| {
            let c = ui.ctx().clone();
            f(&c);
            toasts.draw(&c);
        });
        self.state.handle_platform_output(window, out.platform_output);
        for (id, deltas) in &out.textures_delta.set {
            for delta in deltas {
                if let Err(e) = self.renderer.set_texture(self.queue, self.pool, *id, delta) {
                    log::error!(target: "ui", "texture upload: {e}");
                }
            }
        }
        self.to_free[slot & 1] = out.textures_delta.free.iter().copied().collect();
        let prims = self.ctx.tessellate(out.shapes, out.pixels_per_point);
        let dev = &ctx.device;
        image_barrier(dev, cb, image, Stage::BLIT, Access::TRANSFER_WRITE, Stage::COLOR_ATTACHMENT_OUTPUT,
                      Access::COLOR_ATTACHMENT_READ | Access::COLOR_ATTACHMENT_WRITE, vk::ImageLayout::TRANSFER_DST_OPTIMAL,
                      vk::ImageLayout::COLOR_ATTACHMENT_OPTIMAL);
        let att = [vk::RenderingAttachmentInfo::default()
            .image_view(view)
            .image_layout(vk::ImageLayout::COLOR_ATTACHMENT_OPTIMAL)
            .load_op(vk::AttachmentLoadOp::LOAD)
            .store_op(vk::AttachmentStoreOp::STORE)];
        let ri = vk::RenderingInfo::default().render_area(vk::Rect2D { offset: vk::Offset2D::default(), extent }).layer_count(1).color_attachments(&att);
        unsafe { dev.cmd_begin_rendering(cb, &ri) };
        if let Err(e) = self.renderer.cmd_draw(cb, extent, out.pixels_per_point, &prims) {
            log::error!(target: "ui", "draw: {e}");
        }
        unsafe { dev.cmd_end_rendering(cb) };
        image_barrier(dev, cb, image, Stage::COLOR_ATTACHMENT_OUTPUT, Access::COLOR_ATTACHMENT_WRITE, Stage::BOTTOM_OF_PIPE, Access::NONE,
                      vk::ImageLayout::COLOR_ATTACHMENT_OPTIMAL, vk::ImageLayout::PRESENT_SRC_KHR);
    }
}
