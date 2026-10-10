//! The Options screen (the C++ settings_ui pages): rows bound to cvars by name - a row whose
//! cvar this build doesn't have is simply not shown. Changes apply at once and are written to
//! settings.cfg when the menu closes.

use egui::{RichText, Ui};

use super::{scale, ACCENT, TEXT_DIM, TEXT_FAINT};
use crate::cvar::{self, CVar};

/// The cvars that are user settings (settings.cfg).
pub const ARCHIVED: &[&str] = &[
    // Game / controls
    "game.mode", "game.cheats", "snd.master", "snd.effects", "snd.ambience", "r.fov", "in.sensitivity", "in.invertY", "in.flyspeed", "in.zoom.fov", "in.bob",
    "in.walk", "ui.fps", "ui.crosshair", "ui.minimap",
    // Display
    "app.fullscreen", "r.present", "app.fpscap", "r.exposure", "r.autoexposure",
    // Graphics
    "r.lighting", "r.denoiser", "r.gi.bounces", "r.gi.cache", "r.scale", "r.view.distance", "r.horizon", "r.std.shadows", "r.std.softshadows",
    "r.std.gispeed", "r.lights", "r.lights.shadows", "r.lights.quality", "r.grass", "r.grass.distance", "r.grass.far", "r.grass.shadows",
    "r.grass.style", "r.water", "r.water.reflect", "r.water.caustics", "r.underwater", "r.fog.quality", "r.bloom", "r.dust", "r.foliage.sss",
    "r.foliage.light", "r.gi.filter",
    // Building
    "r.brush.shape", "r.brush.size", "r.brush.preview", "r.build", "r.brush.reach",
    // World
    "r.time.hour", "r.time.daylength", "r.fog.density", "r.fog.height", "r.fog.falloff", "r.fog.noise", "r.wind.strength", "r.wind.gust",
    "r.wind.speed", "r.wind.dir", "r.wind.set", "r.grass.recover", "r.lights.intensity", "game.physics", "world.endless",
    // Look
    "ui.overlay", "r.sun.azimuth", "r.sun.elevation", "r.water.clarity", "r.water.murk", "r.water.waves", "r.tonemap", "r.saturation", "r.contrast",
    "r.warmth", "r.bloom.strength",
];

pub fn settings_path() -> std::path::PathBuf {
    crate::paths::data_path("settings.cfg")
}

pub fn save_settings() {
    if let Err(e) = cvar::save_archive(&settings_path(), ARCHIVED) {
        log::error!(target: "ui", "saving settings: {e}");
    }
}

#[derive(Clone, Copy, Debug, PartialEq, Eq, Default)]
pub enum Page {
    #[default]
    Game,
    Display,
    Graphics,
    Controls,
    World,
}

impl Page {
    pub const ALL: [Page; 5] = [Page::Game, Page::Display, Page::Graphics, Page::Controls, Page::World];
    pub fn title(self) -> &'static str {
        match self {
            Page::Game => "Game",
            Page::Display => "Display",
            Page::Graphics => "Graphics",
            Page::Controls => "Controls",
            Page::World => "World",
        }
    }
}

fn is(name: &str, v: &str) -> bool {
    cvar::find(name).is_some_and(|c| c.str() == v)
}

fn num(name: &str) -> f32 {
    cvar::find(name).and_then(|c| c.str().parse().ok()).unwrap_or(0.0)
}

thread_local! {
    /// The section header waiting for its first row (a section whose cvars this build doesn't
    /// have isn't shown at all).
    static PENDING: std::cell::RefCell<Option<String>> = const { std::cell::RefCell::new(None) };
}

fn section(_ui: &mut Ui, title: &str) {
    PENDING.with(|p| *p.borrow_mut() = Some(title.to_string()));
}

fn flush_section(ui: &mut Ui) {
    if let Some(title) = PENDING.with(|p| p.borrow_mut().take()) {
        let k = scale(ui.ctx());
        ui.add_space(14.0 * k);
        ui.label(RichText::new(title.to_uppercase()).size(15.0 * k).color(ACCENT).strong());
        ui.separator();
    }
}

/// One labelled row: the label (help on hover, dimmed when disabled), then the control.
fn row(ui: &mut Ui, label: &str, help: &str, enabled: bool, add: impl FnOnce(&mut Ui)) {
    flush_section(ui);
    let k = scale(ui.ctx());
    ui.horizontal(|ui| {
        let l = ui
            .allocate_ui_with_layout(egui::vec2(300.0 * k, 30.0 * k), egui::Layout::left_to_right(egui::Align::Center), |ui| {
                ui.set_min_size(egui::vec2(300.0 * k, 30.0 * k));
                ui.add(egui::Label::new(RichText::new(label).color(if enabled { super::TEXT } else { TEXT_FAINT })).truncate())
            })
            .inner;
        if !help.is_empty() {
            l.on_hover_text(help);
        }
        ui.add_enabled_ui(enabled, add);
    });
}

fn set(cv: &dyn CVar, v: &str) {
    if let Err(e) = cv.set_str(v) {
        log::warn!(target: "ui", "{}: {e}", cv.name());
    }
}

/// A choice among values (`labels` shown); `avail[i]` false = that value greyed out.
fn choice(ui: &mut Ui, label: &str, name: &str, values: &[&str], labels: &[&str], help: &str, enabled: bool) {
    let Some(cv) = cvar::find(name) else { return };
    let cur = cv.str();
    row(ui, label, help, enabled, |ui| {
        for (v, l) in values.iter().zip(labels) {
            if ui.selectable_label(cur == *v, *l).clicked() && cur != *v {
                set(cv, v);
            }
        }
    });
}

fn toggle(ui: &mut Ui, label: &str, name: &str, help: &str, enabled: bool) {
    let Some(cv) = cvar::find(name) else { return };
    let mut on = !matches!(cv.str().as_str(), "0" | "false" | "off");
    row(ui, label, help, enabled, |ui| {
        let text = if on { "On" } else { "Off" };
        if ui.checkbox(&mut on, text).changed() {
            set(cv, if on { "1" } else { "0" });
        }
    });
}

/// A slider; `int` = whole numbers; `zero` = the text shown at 0 ("Off").
#[allow(clippy::too_many_arguments)]
fn slider(ui: &mut Ui, label: &str, name: &str, lo: f32, hi: f32, suffix: &str, help: &str, enabled: bool, int: bool, zero: Option<&str>) {
    let Some(cv) = cvar::find(name) else { return };
    let mut v: f32 = cv.str().parse().unwrap_or(lo);
    row(ui, label, help, enabled, |ui| {
        let decimals = if int || hi - lo >= 50.0 { 0 } else if hi - lo >= 5.0 { 1 } else { 2 };
        let s = egui::Slider::new(&mut v, lo..=hi).max_decimals(decimals).suffix(suffix).custom_formatter(|x, _| match zero {
            Some(z) if x.abs() < 1e-6 => z.to_string(),
            _ => format!("{x:.decimals$}"),
        });
        let s = if int { s.step_by(1.0) } else { s };
        if ui.add(s).changed() {
            set(cv, &if int { format!("{}", v.round() as i64) } else { format!("{v}") });
        }
    });
}

fn key_row(ui: &mut Ui, keys: &str, action: &str) {
    flush_section(ui);
    let k = scale(ui.ctx());
    ui.horizontal(|ui| {
        ui.add_space(18.0 * k);
        let w = ui.horizontal(|ui| {
            for key in keys.split(" / ") {
                egui::Frame::new()
                    .fill(egui::Color32::from_white_alpha(22))
                    .stroke(egui::Stroke::new(1.0, egui::Color32::from_white_alpha(40)))
                    .corner_radius(5.0 * k)
                    .inner_margin(egui::Margin::symmetric((9.0 * k) as i8, (4.0 * k) as i8))
                    .show(ui, |ui| ui.label(RichText::new(key).size(16.0 * k).strong()));
            }
        });
        ui.add_space((250.0 * k - w.response.rect.width()).max(12.0 * k));
        ui.label(action);
    });
}

pub fn draw_page(ui: &mut Ui, page: Page) {
    PENDING.with(|p| *p.borrow_mut() = None);
    match page {
        Page::Game => page_game(ui),
        Page::Display => page_display(ui),
        Page::Graphics => page_graphics(ui),
        Page::Controls => page_controls(ui),
        Page::World => page_world(ui),
    }
}

fn page_game(ui: &mut Ui) {
    section(ui, "Audio");
    slider(ui, "Master volume", "snd.master", 0.0, 1.0, "", "Everything you hear.", true, false, None);
    slider(ui, "Effects", "snd.effects", 0.0, 1.0, "", "Footsteps, blocks breaking and placed, water, objects hitting things.", true, false, None);
    slider(ui, "Ambience", "snd.ambience", 0.0, 1.0, "", "Wind, rustling leaves, birds by day, crickets at night, lapping water.", true, false, None);
    section(ui, "View");
    slider(ui, "Field of view", "r.fov", 50.0, 110.0, "°", "How much of the world you see, top to bottom of the screen.", true, true, None);
    section(ui, "Interface");
    toggle(ui, "Crosshair", "ui.crosshair", "The mark in the middle of the screen.", true);
    toggle(ui, "FPS counter", "ui.fps", "Frame rate and frame time in the top-right corner (F3).", true);
    toggle(ui, "Performance overlay", "ui.overlay", "Detailed timings: every render pass, the camera position.", true);
    toggle(ui, "Brush preview", "r.brush.preview", "In build mode, outline the voxels your next click removes (red) or adds (ghost).", true);
}

fn page_display(ui: &mut Ui) {
    section(ui, "Screen");
    choice(ui, "Display mode", "app.fullscreen", &["0", "1"], &["Windowed", "Borderless fullscreen"], "Borderless fullscreen covers the whole monitor (F11).", true);
    choice(ui, "V-Sync", "r.present", &["fifo", "mailbox", "immediate"], &["On", "Fast", "Off"],
           "On: never above the monitor's refresh rate, no tearing. Fast: as fast as it can without tearing. Off: lowest latency, may tear.", true);
    slider(ui, "Frame rate limit", "app.fpscap", 0.0, 360.0, " fps", "Caps the frame rate to save power and heat.", true, true, Some("Unlimited"));
    section(ui, "Picture");
    slider(ui, "Brightness", "r.exposure", 0.2, 2.5, "", "Overall image brightness ([ and ] keys).", true, false, None);
    toggle(ui, "Eye adaptation", "r.autoexposure", "Your eyes adjust to dim places: rooms, dusk and lamp-lit caves open up slowly.", true);
}

fn page_graphics(ui: &mut Ui) {
    let std_ = is("r.lighting", "standard");
    section(ui, "Lighting");
    choice(ui, "Lighting", "r.lighting", &["standard", "raytraced"], &["Standard", "Ray traced"],
           "Ray traced: every pixel traces its own bounce light, then a denoiser cleans it up. Standard: bounce light from cached light probes, noise-free and fast.", true);
    if !std_ {
        choice(ui, "Denoiser", "r.denoiser", &["svgf", "none"], &["Built-in", "None (raw)"], "Cleans up the ray traced image.", true);
        choice(ui, "Bounce light smoothing", "r.gi.filter", &["normal", "smooth"], &["Normal", "Smooth"],
               "Smooth halves the shimmer of bounce light but softens its detail.", true);
    } else {
        choice(ui, "Shadow quality", "r.std.shadows", &["low", "medium", "high", "ultra"], &["Low", "Medium", "High", "Ultra"],
               "How wide sun shadows' soft edges can get. Low = crisper and cheaper.", true);
        toggle(ui, "Soft shadows", "r.std.softshadows", "Shadow edges soften with the distance to what casts them.", true);
        choice(ui, "Bounce light updates", "r.std.gispeed", &["slow", "normal", "fast"], &["Smooth", "Normal", "Fast"],
               "How quickly bounce light reacts when things move or change.", true);
    }
    choice(ui, "Bounce light", "r.gi.bounces", &["1", "2"], &["Single", "Double"], "How many times light bounces in ray-traced lighting.", !std_);
    toggle(ui, "Bounce light cache", "r.gi.cache", "Double bounce: the second bounce is remembered per patch of surface instead of traced again.",
           !std_ && is("r.gi.bounces", "2"));
    toggle(ui, "Glowing blocks cast light", "r.lights", "Lamps, torches and other glowing blocks light their surroundings, with shadows.", true);
    choice(ui, "Lamp shadows", "r.lights.shadows", &["hard", "soft"], &["Hard", "Soft"], "Soft: lamp shadows get a smooth edge.", is("r.lights", "1") && std_);
    choice(ui, "Lamp light quality", "r.lights.quality", &["full", "half"], &["Full", "Half"], "Half: a quarter of the cost, slightly softer.",
           is("r.lights", "1") && std_);
    section(ui, "Resolution");
    slider(ui, "Render resolution", "r.scale", 0.5, 1.0, "", "The resolution the world is drawn at, as a share of the window (F6).", true, false, None);
    section(ui, "World");
    slider(ui, "View distance", "r.view.distance", 4.0, 14.0, " chunks", "How much of the endless world stays loaded around you (a chunk is 12.8 m).", true, true, None);
    toggle(ui, "Far horizon", "r.horizon", "Past the view distance the land carries on to the horizon (hills, lakes, forests).", true);
    let grass = !is("r.grass", "off");
    section(ui, "Environment");
    choice(ui, "Grass", "r.grass", &["high", "low", "off"], &["High", "Low", "Off"], "Grass blades on the ground.", true);
    slider(ui, "Grass draw distance", "r.grass.distance", 30.0, 300.0, "", "How far away grass blades are drawn (in voxels).", grass, true, None);
    slider(ui, "Far grass", "r.grass.far", 0.0, 4.0, "x", "Simple grass tufts carry the field on past the blade distance.", grass, false, Some("Off"));
    toggle(ui, "Grass shadows", "r.grass.shadows", "Blades cast moving shadows on the ground and on each other.", grass);
    choice(ui, "Grass motion", "r.grass.style", &["smooth", "stepped"], &["Smooth", "Blocky"], "How the blades' little cubes move as they sway.", grass);
    toggle(ui, "Transparent water", "r.water", "See into the water: the lake bed shows through.", true);
    toggle(ui, "Water reflections", "r.water.reflect", "Water mirrors the terrain, trees and lamps.", is("r.water", "1"));
    slider(ui, "Caustics", "r.water.caustics", 0.0, 2.0, "", "The rippling net of sunlight under the surface.", is("r.water", "1"), false, Some("Off"));
    slider(ui, "Under-water effects", "r.underwater", 0.0, 2.0, "", "With your head under water: wobble, depth darkening, blur.", true, false, Some("Off"));
    choice(ui, "Fog quality", "r.fog.quality", &["low", "medium", "high"], &["Low", "Medium", "High"], "Detail of the volumetric fog and its light shafts.", true);
    section(ui, "Effects");
    toggle(ui, "Bloom", "r.bloom", "A soft glow around bright light.", true);
    toggle(ui, "Dust particles", "r.dust", "Tiny dust flakes floating in sun beams near you.", true);
}

fn page_controls(ui: &mut Ui) {
    section(ui, "Mouse");
    slider(ui, "Mouse sensitivity", "in.sensitivity", 0.1, 4.0, "", "How far the view turns for a mouse movement.", true, false, None);
    toggle(ui, "Invert vertical look", "in.invertY", "Moving the mouse up looks down.", true);
    section(ui, "Movement");
    toggle(ui, "Walk", "in.walk", "Walk with collision and gravity; off = the free camera (Tab).", true);
    slider(ui, "Flying speed", "in.flyspeed", 5.0, 800.0, "", "How fast you fly (mouse wheel).", true, true, None);
    toggle(ui, "View bob", "in.bob", "The view sways a little with your steps while walking.", true);
    key_row(ui, "W / A / S / D", "Move");
    key_row(ui, "Space", "Jump - double-tap to fly / land");
    key_row(ui, "Ctrl", "Crouch - flying / swimming: down");
    key_row(ui, "Shift", "Sprint - flying: faster");
    key_row(ui, "Tab", "Free camera on / off");
    section(ui, "Building");
    key_row(ui, "1 / ... / 9", "The block you hold");
    key_row(ui, "Middle mouse", "Pick the block you look at");
    key_row(ui, "F", "Build mode on / off");
    key_row(ui, "Left mouse", "Remove voxels (build mode)");
    key_row(ui, "Right mouse", "Place the held block (build mode)");
    key_row(ui, "B", "Brush shape: sphere / cube");
    key_row(ui, "- / =", "Brush size");
    key_row(ui, "Ctrl + Z / Ctrl + Y", "Undo / redo");
    key_row(ui, "P", "Physics on / off");
    section(ui, "Game");
    key_row(ui, "Esc", "Menu");
    key_row(ui, "F5", "Quick save (the current world)");
    key_row(ui, "F9", "Quick load (the current world, else the newest save)");
    key_row(ui, "F11", "Fullscreen");
    key_row(ui, "F3", "FPS counter");
    key_row(ui, "[ / ]", "Brightness");
    key_row(ui, "L / N / F4 / F6 / F10", "Lighting / denoiser / debug view / render scale / reload shaders");
}

fn page_world(ui: &mut Ui) {
    section(ui, "Weather");
    slider(ui, "Fog", "r.fog.density", 0.0, 12.0, "", "How thick the fog is.", true, false, None);
    slider(ui, "Valley fog", "r.fog.falloff", 0.0, 200.0, "", "0 = the same fog everywhere; else fog gathers in the valleys.", true, false, Some("Off"));
    slider(ui, "Valley fog height", "r.fog.height", 150.0, 500.0, "", "The altitude the valley fog fills up to.", num("r.fog.falloff") > 0.0, false, None);
    slider(ui, "Fog wisps", "r.fog.noise", 0.0, 1.0, "", "Drifting patches and wisps in the fog.", true, false, None);
    slider(ui, "Clouds", "r.clouds.coverage", 0.0, 1.0, "", "How much of the sky is cloudy.", true, false, None);
    slider(ui, "Rain", "r.rain", 0.0, 1.0, "", "Rain falling, wet surfaces.", true, false, Some("Off"));
    slider(ui, "Water clarity", "r.water.clarity", 0.2, 5.0, "", "How deep you can see into the water.", true, false, None);
    slider(ui, "Waves", "r.water.waves", 0.0, 3.0, "", "How big the waves on lakes get.", true, false, Some("Glassy"));
    slider(ui, "Water murk", "r.water.murk", 0.0, 6.0, "", "How hazy it is under water.", true, false, Some("Clear"));
    section(ui, "Time");
    slider(ui, "Time of day", "r.time.hour", 0.0, 24.0, " h", "Moves the sun across the sky; from about 18 h the moon and stars take over.", true, false, None);
    slider(ui, "Day length", "r.time.daylength", 0.0, 60.0, " min", "Real minutes for a full day and night.", true, false, Some("Time stopped"));
    slider(ui, "Sun height", "r.sun.elevation", -40.0, 89.0, "°", "Place the sun by hand. Below 0 = night (moonlight).", true, false, None);
    slider(ui, "Sun direction", "r.sun.azimuth", 0.0, 360.0, "°", "Compass direction of the sun.", true, false, None);
    section(ui, "Wind");
    slider(ui, "Wind", "r.wind.strength", 0.0, 3.0, "", "Wind strength.", true, false, Some("Calm"));
    slider(ui, "Gusts", "r.wind.gust", 0.0, 3.0, "", "Strength of the gusts that roll across the grass.", true, false, None);
    slider(ui, "Gust speed", "r.wind.speed", 0.0, 60.0, "", "How fast gusts travel (voxels per second).", true, false, None);
    slider(ui, "Wind direction", "r.wind.dir", 0.0, 360.0, "°", "Where the wind blows from.", true, false, None);
    slider(ui, "Grass recovery", "r.grass.recover", 5.0, 120.0, " s", "How long trampled or wind-bent grass takes to stand back up.", true, false, None);
    section(ui, "Simulation");
    toggle(ui, "Physics", "game.physics", "Loose objects fall, collide and settle (P).", true);
    toggle(ui, "Endless new worlds", "world.endless", "New worlds stream endlessly around you; off = the fixed 2048 x 2048 world.", true);
    section(ui, "Building");
    slider(ui, "Build reach", "r.brush.reach", 8.0, 200.0, "", "How far away you can remove and place voxels.", true, false, None);
    slider(ui, "Brush size", "r.brush.size", 1.0, 40.0, "", "Voxels across (- / = keys).", true, true, None);
    choice(ui, "Brush shape", "r.brush.shape", &["sphere", "cube"], &["Sphere", "Cube"], "B switches it.", true);
    slider(ui, "Lamp brightness", "r.lights.intensity", 0.0, 12.0, "", "How much light lamps and glowing blocks give off.", true, false, None);
    let _ = TEXT_DIM;
}
