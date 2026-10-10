//! The in-game HUD: crosshair, FPS counter, performance overlay, the held block / brush line,
//! the loading screen.

use egui::{Align2, Color32, FontId, Stroke};

use super::{scale, ACCENT, TEXT, TEXT_DIM};
use crate::cvars;

/// What the HUD shows this frame.
pub struct HudInfo<'a> {
    pub frame_ms: f32,
    pub gpu_ms: f32,
    pub res: (u32, u32),
    pub pos: [f32; 3],
    pub passes: &'a [(&'static str, f32)],
    pub build: bool,
    pub held: u8,
    pub held_color: [f32; 3],
    pub walking: bool,
    pub flying: bool,
}

/// A small 4-arm reticle at the screen centre, with a shadow so it reads over anything.
pub fn crosshair(ctx: &egui::Context) {
    let p = ctx.layer_painter(egui::LayerId::new(egui::Order::Foreground, "crosshair".into()));
    let c = ctx.content_rect().center();
    let (g, len) = (3.0, 7.0);
    let arms = [(-g - len, 0.0, -g, 0.0), (g, 0.0, g + len, 0.0), (0.0, -g - len, 0.0, -g), (0.0, g, 0.0, g + len)];
    for (x0, y0, x1, y1) in arms {
        let (a, b) = (c + egui::vec2(x0, y0), c + egui::vec2(x1, y1));
        p.line_segment([a + egui::vec2(1.0, 1.0), b + egui::vec2(1.0, 1.0)], Stroke::new(1.0, Color32::from_black_alpha(150)));
        p.line_segment([a, b], Stroke::new(1.0, Color32::from_white_alpha(210)));
    }
}

pub fn draw(ctx: &egui::Context, h: &HudInfo) {
    let k = scale(ctx);
    let screen = ctx.content_rect();
    let p = ctx.layer_painter(egui::LayerId::new(egui::Order::Foreground, "hud".into()));
    if cvars::UI_FPS.get() || cvars::UI_OVERLAY.get() {
        let t = format!("{:.0} fps  {:.1} ms", 1000.0 / h.frame_ms.max(0.01), h.frame_ms);
        p.text(screen.right_top() + egui::vec2(-16.0, 12.0) * k, Align2::RIGHT_TOP, t, FontId::monospace(17.0 * k), TEXT);
    }
    if cvars::UI_OVERLAY.get() {
        let mut lines = vec![
            format!("{}x{}  {}  GPU {:.2} ms", h.res.0, h.res.1, cvars::R_LIGHTING.get(), h.gpu_ms),
            format!("pos {:.0} {:.0} {:.0}", h.pos[0], h.pos[1], h.pos[2]),
        ];
        lines.extend(h.passes.iter().map(|(n, ms)| format!("{n:<16} {ms:6.2}")));
        let font = FontId::monospace(14.0 * k);
        let mut y = screen.top() + 40.0 * k;
        let w = 260.0 * k;
        let bg = egui::Rect::from_min_size(egui::pos2(screen.right() - w - 16.0 * k, y - 6.0 * k), egui::vec2(w + 8.0 * k, lines.len() as f32 * 17.0 * k + 12.0 * k));
        p.rect_filled(bg, 6.0 * k, Color32::from_black_alpha(140));
        for l in lines {
            p.text(egui::pos2(screen.right() - w - 8.0 * k, y), Align2::LEFT_TOP, l, font.clone(), TEXT);
            y += 17.0 * k;
        }
    }
    // The held block and the brush, bottom centre (build mode).
    if h.build {
        let c = screen.center_bottom() - egui::vec2(0.0, 42.0 * k);
        let sw = egui::Rect::from_center_size(c - egui::vec2(120.0 * k, 0.0), egui::vec2(34.0, 34.0) * k);
        let col = Color32::from_rgb((h.held_color[0].clamp(0.0, 1.0).powf(1.0 / 2.2) * 255.0) as u8, (h.held_color[1].clamp(0.0, 1.0).powf(1.0 / 2.2) * 255.0) as u8,
                                    (h.held_color[2].clamp(0.0, 1.0).powf(1.0 / 2.2) * 255.0) as u8);
        p.rect_filled(sw.expand(3.0 * k), 6.0 * k, Color32::from_black_alpha(160));
        p.rect_filled(sw, 4.0 * k, col);
        let t = format!("block {}   {} {}   {}", h.held, cvars::R_BRUSH_SHAPE.get(), cvars::R_BRUSH_SIZE.get(),
                        if !h.walking { "free camera" } else if h.flying { "flying" } else { "walking" });
        let g = p.layout_no_wrap(t, FontId::proportional(18.0 * k), TEXT);
        let r = egui::Rect::from_min_size(egui::pos2(sw.right() + 14.0 * k, c.y - g.size().y * 0.5), g.size()).expand2(egui::vec2(10.0, 6.0) * k);
        p.rect_filled(r, 6.0 * k, Color32::from_black_alpha(140));
        p.galley(r.min + egui::vec2(10.0, 6.0) * k, g, TEXT);
    }
}

/// The loading screen: title and a progress bar (`progress` 0..1, None = indeterminate).
pub fn loading(ctx: &egui::Context, what: &str, progress: Option<f32>) {
    let k = scale(ctx);
    let screen = ctx.content_rect();
    let p = ctx.layer_painter(egui::LayerId::background());
    p.rect_filled(screen, 0.0, Color32::from_rgb(12, 14, 18));
    let c = screen.center();
    p.text(c - egui::vec2(0.0, 60.0 * k), Align2::CENTER_CENTER, "VoxelForge", FontId::proportional(64.0 * k), ACCENT);
    p.text(c + egui::vec2(0.0, 10.0 * k), Align2::CENTER_CENTER, what, FontId::proportional(22.0 * k), TEXT_DIM);
    let bar = egui::Rect::from_center_size(c + egui::vec2(0.0, 56.0 * k), egui::vec2(420.0, 6.0) * k);
    p.rect_filled(bar, 3.0, Color32::from_white_alpha(25));
    let f = match progress {
        Some(f) => f.clamp(0.0, 1.0),
        None => {
            // Indeterminate: a sliding segment.
            let t = ctx.input(|i| i.time) as f32;
            let x = (t * 0.6).fract();
            let seg = egui::Rect::from_min_size(egui::pos2(bar.left() + bar.width() * x * 0.75, bar.top()), egui::vec2(bar.width() * 0.25, bar.height()));
            p.rect_filled(seg, 3.0, ACCENT);
            ctx.request_repaint();
            return;
        }
    };
    p.rect_filled(egui::Rect::from_min_size(bar.min, egui::vec2(bar.width() * f, bar.height())), 3.0, ACCENT);
}
