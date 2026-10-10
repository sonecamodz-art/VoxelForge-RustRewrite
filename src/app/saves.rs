//! World saves: the `saves/` folder next to the game, quick save / load (F5 / F9), save
//! listings with thumbnails.
//!
//! A save holds the world (voxels, bodies, palette, water) plus its world state: the camera and
//! the world cvars (time of day, weather, wind, water). Player settings stay in settings.cfg.

use std::path::{Path, PathBuf};
use std::time::{Instant, SystemTime, UNIX_EPOCH};

use parking_lot::Mutex;

use crate::camera::FlyCamera;
use crate::fluid::FluidSim;
use crate::gfx::context::Context;
use crate::gfx::renderer::Renderer;
use crate::objects::VoxObject;
use crate::serialize::{self, SaveParts, SceneInfo, SceneState, WaterCell};
use crate::voxel::world::World;

/// World state saved with a world. (Names this build lacks are skipped.)
const WORLD_CVARS: &[&str] = &[
    "r.time.hour",
    "r.time.daylength",
    "r.time.freeze",
    "r.sun.elevation",
    "r.sun.azimuth",
    "r.fog.density",
    "r.fog.height",
    "r.fog.falloff",
    "r.fog.noise",
    "r.wind.strength",
    "r.wind.gust",
    "r.wind.speed",
    "r.wind.dir",
    "r.grass.recover",
    "r.water.clarity",
    "r.water.murk",
    "game.mode",
    "r.clouds.coverage",
    "r.clouds.thickness",
    "r.rain",
    "game.weather",
];
pub const THUMB_W: u32 = 320;
pub const THUMB_H: u32 = 180;

/// The world last saved / loaded ("" until then).
static CURRENT: Mutex<String> = Mutex::new(String::new());

pub fn current_world_name() -> String {
    CURRENT.lock().clone()
}
pub fn set_current_world_name(name: &str) {
    *CURRENT.lock() = name.to_string();
}

pub fn saves_dir() -> PathBuf {
    crate::paths::data_path("saves")
}

/// A file-system-safe world name (letters, digits, space, - _ ' and non-ASCII; 48 bytes).
pub fn sanitize(name: &str) -> String {
    let mut out: String = name
        .chars()
        .map(|c| if c.is_alphanumeric() || matches!(c, ' ' | '-' | '_' | '\'') || !c.is_ascii() { c } else { '_' })
        .collect();
    while out.len() > 48 {
        out.pop();
    }
    let t = out.trim_start().trim_end_matches([' ', '.']);
    if t.is_empty() { "World".into() } else { t.to_string() }
}

pub fn save_path_for(name: &str) -> PathBuf {
    saves_dir().join(format!("{}.vfworld", sanitize(name)))
}

/// The first free "World N".
pub fn fresh_world_name() -> String {
    (1..1000).map(|n| format!("World {n}")).find(|n| !save_path_for(n).exists()).unwrap_or_else(|| "World".into())
}

/// One save in the folder.
#[derive(Clone, Debug)]
pub struct SaveSlot {
    pub path: PathBuf,
    pub info: SceneInfo,
    pub bytes: u64,
    /// None: readable; else why not.
    pub error: Option<String>,
}

/// Every save in the folder, newest first (thumbnails included).
pub fn list_saves() -> Vec<SaveSlot> {
    let mut out = Vec::new();
    let Ok(rd) = std::fs::read_dir(saves_dir()) else { return out };
    for e in rd.flatten() {
        let path = e.path();
        if !matches!(path.extension().and_then(|x| x.to_str()), Some("vfworld" | "vfscene")) || !path.is_file() {
            continue;
        }
        let meta = e.metadata().ok();
        let mtime = meta.as_ref().and_then(|m| m.modified().ok()).and_then(|t| t.duration_since(UNIX_EPOCH).ok()).map_or(0, |d| d.as_secs() as i64);
        let (mut info, error) = match serialize::read_scene_info(&path) {
            Ok(i) => (i, None),
            Err(e) => (SceneInfo::default(), Some(e.to_string())),
        };
        if info.name.is_empty() {
            info.name = path.file_stem().map(|s| s.to_string_lossy().into_owned()).unwrap_or_default();
        }
        if info.saved_at == 0 {
            info.saved_at = mtime;
        }
        out.push(SaveSlot { bytes: meta.map_or(0, |m| m.len()), path, info, error });
    }
    out.sort_by_key(|s| std::cmp::Reverse(s.info.saved_at));
    out
}

/// The most recently saved readable world.
pub fn newest_save() -> Option<SaveSlot> {
    list_saves().into_iter().find(|s| s.error.is_none())
}

/// Everything a save / load touches.
pub struct WorldIo<'a> {
    pub ctx: &'a Context,
    pub r: &'a mut Renderer,
    pub world: &'a mut World,
    pub objects: &'a mut Vec<VoxObject>,
    pub cam: &'a mut FlyCamera,
    pub fluid: &'a mut FluidSim,
    pub endless: &'a mut Option<super::endless::Endless>,
}

/// The rendered scene (no UI), cropped to 16:9 and box-filtered to the thumbnail (top-down).
fn capture_thumb(ctx: &Context, r: &Renderer) -> Vec<u8> {
    ctx.wait_idle();
    let img = r.out_image();
    let (w, h) = (img.w() as usize, img.h() as usize);
    if w == 0 || h == 0 {
        return Vec::new();
    }
    let src = super::shot::readback(ctx, img, 4);
    let (mut cw, mut ch) = (w, h);
    if cw * 9 > ch * 16 {
        cw = ch * 16 / 9;
    } else {
        ch = cw * 9 / 16;
    }
    let (x0, y0) = ((w - cw) / 2, (h - ch) / 2);
    let (tw, th) = (THUMB_W as usize, THUMB_H as usize);
    let mut out = vec![0u8; tw * th * 4];
    for ty in 0..th {
        let fy = th - 1 - ty; // out_tex rows are bottom-up
        let (sy0, sy1) = (y0 + fy * ch / th, (y0 + (fy + 1) * ch / th).max(y0 + fy * ch / th + 1));
        for tx in 0..tw {
            let (sx0, sx1) = (x0 + tx * cw / tw, (x0 + (tx + 1) * cw / tw).max(x0 + tx * cw / tw + 1));
            let mut acc = [0u32; 3];
            for y in sy0..sy1 {
                for x in sx0..sx1 {
                    let p = &src[(y * w + x) * 4..];
                    for c in 0..3 {
                        acc[c] += p[c] as u32;
                    }
                }
            }
            let n = ((sy1 - sy0) * (sx1 - sx0)) as u32;
            let o = &mut out[(ty * tw + tx) * 4..];
            for c in 0..3 {
                o[c] = (acc[c] / n) as u8;
            }
            o[3] = 255;
        }
    }
    out
}

/// Save the world as `name`. Returns the sanitised name.
pub fn save_world(io: &mut WorldIo, name: &str) -> Result<String, String> {
    let name = sanitize(name);
    let t0 = Instant::now();
    std::fs::create_dir_all(saves_dir()).map_err(|e| format!("can't create {}: {e}", saves_dir().display()))?;
    let thumb = capture_thumb(io.ctx, io.r);
    let info = SceneInfo {
        name: name.clone(),
        saved_at: SystemTime::now().duration_since(UNIX_EPOCH).map_or(0, |d| d.as_secs() as i64),
        thumb_w: if thumb.is_empty() { 0 } else { THUMB_W },
        thumb_h: if thumb.is_empty() { 0 } else { THUMB_H },
        thumb,
    };
    // Saves hold absolute coordinates (an endless world's floating origin added back).
    let (ox, oz) = io.endless.as_ref().map_or((0, 0), |e| e.origin());
    let mut st = SceneState {
        has_camera: true,
        cam_pos: [io.cam.pos.x + ox as f32, io.cam.pos.y, io.cam.pos.z + oz as f32],
        cam_yaw: io.cam.yaw,
        cam_pitch: io.cam.pitch,
        cvars: WORLD_CVARS.iter().filter_map(|&n| crate::cvar::find(n).map(|c| (n.to_string(), c.str()))).collect(),
        water: Vec::new(),
    };
    io.fluid.partials().for_each(|x, y, z, volume| st.water.push(WaterCell { x: x + ox, y, z: z + oz, volume }));
    let shifted: Vec<VoxObject>;
    let bodies: &[VoxObject] = if ox != 0 || oz != 0 {
        shifted = io
            .objects
            .iter()
            .map(|o| {
                let mut o = o.clone();
                o.shift_world(-ox as f32, -oz as f32);
                o
            })
            .collect();
        &shifted
    } else {
        io.objects
    };
    let path = save_path_for(&name);
    // An endless world: its edited chunks into the region files beside the save first.
    if let Some(e) = io.endless.as_mut() {
        e.flush(io.world, &super::endless::chunk_dir_for(&path)).map_err(|e| format!("the chunks couldn't be written: {e}"))?;
    }
    let parts = SaveParts { info: Some(&info), state: Some(&st), endless: io.endless.as_ref().map(|e| &e.info), ..Default::default() };
    serialize::save_scene(&path, io.world, bodies, &parts).map_err(|e| e.to_string())?;
    set_current_world_name(&name);
    log::info!(target: "saves", "saved {} ({} bodies, {} water cells) in {:.0} ms", path.display(), io.objects.len(), st.water.len(),
               t0.elapsed().as_secs_f64() * 1e3);
    Ok(name)
}

/// Load a save over the current world (GPU state, water, world cvars and the camera with it).
/// Nothing changes if it fails. Returns the world's name (its file's: a copied save keeps its
/// old name inside).
pub fn load_world(io: &mut WorldIo, path: &Path) -> Result<String, String> {
    let s = super::world_from_save(path)?;
    *io.endless = None; // (the old session's workers stop)
    *io.world = s.world;
    *io.objects = s.objects;
    *io.endless = s.endless;
    io.r.gpu.stream_radius = io.endless.as_ref().map_or(0, |e| e.stream.radius());
    io.r.reload_scene(io.ctx, io.world, io.objects);
    if let Some(st) = &s.state {
        restore_state(st, io.fluid, io.cam);
    }
    let name = path.file_stem().map(|s| s.to_string_lossy().into_owned()).unwrap_or_default();
    set_current_world_name(&name);
    Ok(name)
}

/// The non-voxel world state of a loaded save: water, world cvars, camera.
pub fn restore_state(st: &SceneState, fluid: &mut FluidSim, cam: &mut FlyCamera) {
    fluid.reset();
    for c in &st.water {
        fluid.set_partial(c.x, c.y, c.z, c.volume);
        fluid.wake(c.x, c.y, c.z, c.x, c.y, c.z); // flowing water keeps flowing
    }
    for (k, v) in &st.cvars {
        if let Some(cv) = crate::cvar::find(k).filter(|_| WORLD_CVARS.contains(&k.as_str())) {
            let _ = cv.set_str(v);
        }
    }
    if st.has_camera {
        cam.pos = crate::math::Vec3::from_array(st.cam_pos);
        cam.yaw = st.cam_yaw;
        cam.pitch = st.cam_pitch;
    }
}

/// F5: save as the current world (a fresh "World N" the first time).
pub fn quick_save(io: &mut WorldIo) -> Result<String, String> {
    let mut name = current_world_name();
    if name.is_empty() {
        name = fresh_world_name();
    }
    save_world(io, &name)
}

/// F9: reload the current world, else the newest save.
pub fn quick_load(io: &mut WorldIo) -> Result<String, String> {
    let cur = current_world_name();
    let path = if !cur.is_empty() && save_path_for(&cur).exists() {
        save_path_for(&cur)
    } else {
        newest_save().ok_or("no saved worlds yet - F5 saves this one")?.path
    };
    load_world(io, &path)
}

/// `--load` argument: a path, or the name of a world in the saves folder.
pub fn resolve(arg: &str) -> PathBuf {
    let p = PathBuf::from(arg);
    if p.exists() { p } else { save_path_for(arg) }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn sanitized_names() {
        assert_eq!(sanitize("My World"), "My World");
        assert_eq!(sanitize("../evil/name."), "___evil_name_");
        assert_eq!(sanitize("   "), "World");
        assert_eq!(sanitize("Ünïcode ok"), "Ünïcode ok");
        assert!(sanitize(&"x".repeat(100)).len() <= 48);
    }
}
