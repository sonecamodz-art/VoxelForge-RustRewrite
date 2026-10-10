//! The pause menu (Esc): Resume / Save / Load / Options / Quit, the save and load screens (save
//! cards with thumbnails), and the Options pages.

use std::collections::HashMap;
use std::path::PathBuf;
use std::time::{SystemTime, UNIX_EPOCH};

use egui::{Align2, Color32, RichText, Sense, TextureHandle, Ui};

use super::settings::{self, Page};
use super::{scale, ACCENT, ACCENT_HI, ON_ACCENT, TEXT_DIM, TEXT_FAINT};
use crate::app::saves::{self, SaveSlot};

#[derive(Clone, Copy, Debug, PartialEq, Eq, Default)]
pub enum Screen {
    #[default]
    Closed,
    Pause,
    Options(Page),
    Save,
    Load,
}

/// What the menu asks the game to do this frame.
#[derive(Clone, Debug, PartialEq, Eq)]
pub enum MenuAction {
    Resume,
    Quit,
    Save(String),
    Load(PathBuf),
}

#[derive(Default)]
pub struct Menu {
    pub screen: Screen,
    name: String,
    slots: Vec<SaveSlot>,
    thumbs: HashMap<PathBuf, TextureHandle>,
    selected: Option<usize>,
    /// Waiting for "are you sure": overwrite (save) or delete (load).
    confirm: Option<PathBuf>,
    /// The save screen just opened: focus its name field.
    focus_name: bool,
}

/// "5 min ago", "yesterday", "12 days ago".
fn ago(t: i64) -> String {
    let now = SystemTime::now().duration_since(UNIX_EPOCH).map_or(0, |d| d.as_secs() as i64);
    let s = (now - t).max(0);
    match s {
        0..60 => "just now".into(),
        60..3600 => format!("{} min ago", s / 60),
        3600..86400 => format!("{} h ago", s / 3600),
        86400..172800 => "yesterday".into(),
        _ => format!("{} days ago", s / 86400),
    }
}

fn size_text(b: u64) -> String {
    if b >= 1 << 20 { format!("{:.1} MB", b as f64 / 1048576.0) } else { format!("{:.0} KB", b as f64 / 1024.0) }
}

/// A big left-aligned menu button with an amber bar on hover; `hint` = its key.
fn big_button(ui: &mut Ui, label: &str, hint: Option<&str>) -> bool {
    let k = scale(ui.ctx());
    let (rect, resp) = ui.allocate_exact_size(egui::vec2(380.0 * k, 52.0 * k), Sense::click());
    let p = ui.painter();
    if resp.hovered() {
        p.rect_filled(rect, 6.0 * k, Color32::from_white_alpha(20));
        p.rect_filled(egui::Rect::from_min_size(rect.min, egui::vec2(4.0 * k, rect.height())), 2.0, ACCENT);
    }
    p.text(rect.left_center() + egui::vec2(24.0 * k, 0.0), Align2::LEFT_CENTER, label, egui::FontId::proportional(27.0 * k),
           if resp.hovered() { ACCENT_HI } else { super::TEXT });
    if let Some(h) = hint {
        p.text(rect.right_center() - egui::vec2(16.0 * k, 0.0), Align2::RIGHT_CENTER, h, egui::FontId::proportional(17.0 * k), TEXT_FAINT);
    }
    resp.clicked()
}

fn primary(ui: &mut Ui, label: &str, enabled: bool) -> bool {
    let k = scale(ui.ctx());
    let b = egui::Button::new(RichText::new(label).color(ON_ACCENT).strong()).fill(ACCENT).min_size(egui::vec2(160.0 * k, 40.0 * k));
    ui.add_enabled(enabled, b).clicked()
}

impl Menu {
    pub fn is_open(&self) -> bool {
        self.screen != Screen::Closed
    }

    pub fn open(&mut self, screen: Screen) {
        if matches!(screen, Screen::Save | Screen::Load) {
            self.slots = saves::list_saves();
            self.selected = None;
            self.confirm = None;
            let cur = saves::current_world_name();
            self.name = if cur.is_empty() { saves::fresh_world_name() } else { cur };
            self.focus_name = screen == Screen::Save;
        }
        self.screen = screen;
    }

    /// Esc: back one level (closing it from the pause screen). Returns true when the menu closed.
    pub fn back(&mut self) -> bool {
        self.screen = match self.screen {
            Screen::Pause | Screen::Closed => Screen::Closed,
            _ => Screen::Pause,
        };
        if self.screen == Screen::Closed {
            settings::save_settings();
        }
        self.screen == Screen::Closed
    }

    fn thumb(&mut self, ctx: &egui::Context, s: &SaveSlot) -> Option<TextureHandle> {
        if s.info.thumb.is_empty() {
            return None;
        }
        Some(
            self.thumbs
                .entry(s.path.clone())
                .or_insert_with(|| {
                    let img = egui::ColorImage::from_rgba_unmultiplied([s.info.thumb_w as usize, s.info.thumb_h as usize], &s.info.thumb);
                    ctx.load_texture(s.path.display().to_string(), img, egui::TextureOptions::LINEAR)
                })
                .clone(),
        )
    }

    /// Draw the open screen; returns what the player chose.
    pub fn draw(&mut self, ctx: &egui::Context) -> Option<MenuAction> {
        if !self.is_open() {
            return None;
        }
        let k = scale(ctx);
        // Dim the world behind the menu.
        ctx.layer_painter(egui::LayerId::background()).rect_filled(ctx.content_rect(), 0.0, Color32::from_black_alpha(150));
        let mut act = None;
        let screen = self.screen;
        match screen {
            Screen::Closed => {}
            Screen::Pause => {
                egui::Area::new("pause".into()).anchor(Align2::LEFT_CENTER, egui::vec2(120.0 * k, 0.0)).show(ctx, |ui| {
                    ui.label(RichText::new("VoxelForge").size(54.0 * k).strong().color(ACCENT));
                    let cur = saves::current_world_name();
                    ui.label(RichText::new(if cur.is_empty() { "Unsaved world".to_string() } else { cur }).color(TEXT_DIM));
                    ui.add_space(30.0 * k);
                    if big_button(ui, "Resume", Some("Esc")) {
                        act = Some(MenuAction::Resume);
                    }
                    if big_button(ui, "Save world", Some("F5")) {
                        self.open(Screen::Save);
                    }
                    if big_button(ui, "Load world", Some("F9")) {
                        self.open(Screen::Load);
                    }
                    if big_button(ui, "Options", None) {
                        self.screen = Screen::Options(Page::Game);
                    }
                    ui.add_space(20.0 * k);
                    if big_button(ui, "Quit", None) {
                        act = Some(MenuAction::Quit);
                    }
                });
            }
            Screen::Options(page) => self.options(ctx, page),
            Screen::Save => act = self.save_screen(ctx),
            Screen::Load => act = self.load_screen(ctx),
        }
        act
    }

    /// The panel screens share a frame: title, body, Back.
    fn panel(ctx: &egui::Context, title: &str, body: impl FnOnce(&mut Ui)) -> bool {
        let k = scale(ctx);
        let mut back = false;
        let size = ctx.content_rect().size();
        egui::Window::new(title)
            .title_bar(false)
            .resizable(false)
            .collapsible(false)
            .anchor(Align2::CENTER_CENTER, egui::Vec2::ZERO)
            .fixed_size(egui::vec2((1100.0 * k).min(size.x - 40.0), (size.y * 0.82).min(900.0 * k)))
            .frame(egui::Frame::window(&ctx.global_style()).inner_margin(28.0 * k))
            .show(ctx, |ui| {
                ui.horizontal(|ui| {
                    ui.label(RichText::new(title).size(34.0 * k).strong());
                    ui.with_layout(egui::Layout::right_to_left(egui::Align::Center), |ui| {
                        back = ui.button("Back").clicked();
                    });
                });
                ui.add_space(10.0 * k);
                body(ui);
            });
        back
    }

    fn options(&mut self, ctx: &egui::Context, page: Page) {
        let mut page = page;
        let back = Self::panel(ctx, "Options", |ui| {
            ui.horizontal(|ui| {
                for p in Page::ALL {
                    if ui.selectable_label(p == page, RichText::new(p.title()).size(21.0 * scale(ui.ctx()))).clicked() {
                        page = p;
                    }
                }
            });
            ui.separator();
            egui::ScrollArea::vertical().auto_shrink(false).show(ui, |ui| settings::draw_page(ui, page));
        });
        self.screen = if back {
            settings::save_settings();
            Screen::Pause
        } else {
            Screen::Options(page)
        };
    }

    /// A save card: thumbnail, name, when, size. Returns clicked / double-clicked.
    fn card(&mut self, ui: &mut Ui, i: usize) -> (bool, bool) {
        let k = scale(ui.ctx());
        let s = self.slots[i].clone();
        let tex = self.thumb(ui.ctx(), &s);
        let sel = self.selected == Some(i);
        let (rect, resp) = ui.allocate_exact_size(egui::vec2(ui.available_width(), 96.0 * k), Sense::click());
        let p = ui.painter();
        p.rect_filled(rect, 8.0 * k, if sel { ACCENT.gamma_multiply(0.25) } else if resp.hovered() { Color32::from_white_alpha(18) } else { Color32::from_white_alpha(8) });
        let th = egui::Rect::from_min_size(rect.min + egui::vec2(8.0, 8.0) * k, egui::vec2(142.0, 80.0) * k);
        match tex {
            Some(t) => {
                p.image(t.id(), th, egui::Rect::from_min_max(egui::pos2(0.0, 0.0), egui::pos2(1.0, 1.0)), Color32::WHITE);
            }
            None => {
                p.rect_filled(th, 4.0 * k, Color32::from_white_alpha(12));
            }
        }
        let x = th.right() + 18.0 * k;
        p.text(egui::pos2(x, rect.top() + 14.0 * k), Align2::LEFT_TOP, &s.info.name, egui::FontId::proportional(23.0 * k), super::TEXT);
        let sub = match &s.error {
            Some(e) => format!("Can't be loaded: {e}"),
            None => format!("{}  ·  {}", ago(s.info.saved_at), size_text(s.bytes)),
        };
        p.text(egui::pos2(x, rect.top() + 52.0 * k), Align2::LEFT_TOP, sub, egui::FontId::proportional(17.0 * k),
               if s.error.is_some() { super::ERROR } else { TEXT_DIM });
        (resp.clicked(), resp.double_clicked())
    }

    fn save_screen(&mut self, ctx: &egui::Context) -> Option<MenuAction> {
        let mut act = None;
        let back = Self::panel(ctx, "Save world", |ui| {
            let k = scale(ui.ctx());
            ui.horizontal(|ui| {
                ui.label("Name");
                let field = ui.add(egui::TextEdit::singleline(&mut self.name).desired_width(420.0 * k).char_limit(48));
                if std::mem::take(&mut self.focus_name) {
                    field.request_focus();
                }
                let enter = field.lost_focus() && ui.input(|i| i.key_pressed(egui::Key::Enter));
                let path = saves::save_path_for(&self.name);
                let exists = path.exists();
                if self.confirm.as_ref() == Some(&path) {
                    ui.label(RichText::new("Overwrite it?").color(ACCENT));
                    if primary(ui, "Overwrite", true) {
                        act = Some(MenuAction::Save(self.name.clone()));
                    }
                    if ui.button("Cancel").clicked() {
                        self.confirm = None;
                    }
                } else if (primary(ui, "Save", !self.name.trim().is_empty()) || enter) && !self.name.trim().is_empty() {
                    if exists && saves::current_world_name() != saves::sanitize(&self.name) {
                        self.confirm = Some(path);
                    } else {
                        act = Some(MenuAction::Save(self.name.clone()));
                    }
                }
            });
            ui.add_space(12.0 * k);
            ui.label(RichText::new(if self.slots.is_empty() { "No saved worlds yet." } else { "Saved worlds (click to reuse a name):" }).color(TEXT_DIM));
            egui::ScrollArea::vertical().auto_shrink(false).show(ui, |ui| {
                for i in 0..self.slots.len() {
                    if self.card(ui, i).0 {
                        self.selected = Some(i);
                        self.name = self.slots[i].info.name.clone();
                        self.confirm = None;
                    }
                }
            });
        });
        if back {
            self.screen = Screen::Pause;
        }
        act
    }

    fn load_screen(&mut self, ctx: &egui::Context) -> Option<MenuAction> {
        let mut act = None;
        let back = Self::panel(ctx, "Load world", |ui| {
            let k = scale(ui.ctx());
            let sel = self.selected.and_then(|i| self.slots.get(i).cloned());
            ui.horizontal(|ui| {
                let ok = sel.as_ref().is_some_and(|s| s.error.is_none());
                if primary(ui, "Load", ok) {
                    act = sel.as_ref().map(|s| MenuAction::Load(s.path.clone()));
                }
                if let Some(s) = &sel {
                    if self.confirm.as_ref() == Some(&s.path) {
                        ui.label(RichText::new(format!("Delete \"{}\" for good?", s.info.name)).color(super::ERROR));
                        if ui.button("Delete").clicked() {
                            let _ = std::fs::remove_file(&s.path);
                            let _ = std::fs::remove_dir_all(crate::app::endless::chunk_dir_for(&s.path));
                            log::info!(target: "saves", "deleted {}", s.path.display());
                            self.slots = saves::list_saves();
                            self.thumbs.remove(&s.path);
                            self.selected = None;
                            self.confirm = None;
                        }
                        if ui.button("Cancel").clicked() {
                            self.confirm = None;
                        }
                    } else if ui.button("Delete").clicked() {
                        self.confirm = Some(s.path.clone());
                    }
                }
            });
            ui.add_space(12.0 * k);
            if self.slots.is_empty() {
                ui.label(RichText::new("No saved worlds yet - F5 saves this one.").color(TEXT_DIM));
            }
            egui::ScrollArea::vertical().auto_shrink(false).show(ui, |ui| {
                for i in 0..self.slots.len() {
                    let (click, double) = self.card(ui, i);
                    if click {
                        self.selected = Some(i);
                        self.confirm = None;
                    }
                    if double && self.slots[i].error.is_none() {
                        act = Some(MenuAction::Load(self.slots[i].path.clone()));
                    }
                }
            });
        });
        if back {
            self.screen = Screen::Pause;
        }
        act
    }
}
