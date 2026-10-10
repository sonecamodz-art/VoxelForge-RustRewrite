//! The application layer: the headless shot / bench paths and the game window.
pub mod endless;
pub mod render_path;
pub mod saves;
pub mod scene_edit;
pub mod shot;
pub mod swapchain;
pub mod ui;
pub mod window;

use std::path::Path;

use crate::objects::{build_self_test_objects, VoxObject};
use crate::serialize::{EndlessInfo, SceneState};
use crate::voxel::world::World;
use endless::Endless;

/// Chunks within this radius of the spawn / a loaded camera are in before play starts.
const INNER_RADIUS: i32 = 4;

/// A world ready to render: its voxels (an endless world's around the start), bodies, a
/// loaded save's state, the endless session.
pub struct Startup {
    pub world: World,
    pub objects: Vec<VoxObject>,
    /// A loaded save's state, in ENGINE coordinates (the floating origin taken off).
    pub state: Option<SceneState>,
    pub endless: Option<Endless>,
}

/// The startup world: the save named by `load` (a path or a world name in saves/), else (or if
/// it can't be loaded) a new world - endless unless world.endless is off - with the self-test
/// bodies.
pub fn startup_world(load: Option<&str>, progress: Option<std::sync::Arc<std::sync::atomic::AtomicI32>>) -> Startup {
    if let Some(arg) = load {
        let path = saves::resolve(arg);
        match world_from_save(&path) {
            Ok(s) => {
                saves::set_current_world_name(&path.file_stem().map(|s| s.to_string_lossy().into_owned()).unwrap_or_default());
                return s;
            }
            Err(e) => log::error!(target: "saves", "can't load {}: {e} - a new world instead", path.display()),
        }
    }
    new_world(progress)
}

pub fn new_world(progress: Option<std::sync::Arc<std::sync::atomic::AtomicI32>>) -> Startup {
    let seed = crate::cvars::WORLD_SEED.get() as u32;
    let mut world = World::default();
    let endless = if crate::cvars::WORLD_ENDLESS.get() {
        world.init_palette();
        let info = EndlessInfo { endless: true, seed, ny: world.ny, ..Default::default() };
        // The spawn area's middle (the tower): the same land as the fixed world there.
        Some(Endless::begin(&mut world, &info, None, 1024.0, 1024.0, INNER_RADIUS, None))
    } else {
        world.gen_progress = progress;
        world.generate(seed);
        world.gen_progress = None;
        None
    };
    let objects = build_self_test_objects(&world);
    Startup { world, objects, state: None, endless }
}

/// A save, ready to render. An endless save streams from its region files; a fixed-size one
/// becomes endless (its land kept chunk for chunk, the rest generated around it) unless
/// world.endless is off.
pub fn world_from_save(path: &Path) -> Result<Startup, String> {
    let t0 = std::time::Instant::now();
    let loaded = crate::serialize::load_scene(path).map_err(|e| e.to_string())?;
    let mut world = World::default();
    let mut objects = Vec::new();
    let (_, mut state, info) = loaded.apply(&mut world, &mut objects);
    let chunk_dir = endless::chunk_dir_for(path);
    let (x, z) = if state.has_camera { (state.cam_pos[0], state.cam_pos[2]) } else { (1024.0, 1024.0) };
    let endless = if info.endless {
        if !info.remap.is_empty() {
            crate::region_file::migrate_regions(&chunk_dir, &info.remap, info.ny); // (old custom colours)
        }
        Some(Endless::begin(&mut world, &info, Some(&chunk_dir), x, z, INNER_RADIUS, None))
    } else if crate::cvars::WORLD_ENDLESS.get() {
        let bounded = std::mem::take(&mut world);
        world.palette = bounded.palette.clone();
        world.set_next_import_slot(bounded.next_import_slot());
        let info = endless::guess(&bounded);
        Some(Endless::begin(&mut world, &info, Some(&chunk_dir), x, z, INNER_RADIUS, Some(&bounded)))
    } else {
        None
    };
    if let Some(e) = &endless {
        // Saves hold absolute coordinates: to the engine's.
        let (ox, oz) = e.origin();
        for o in &mut objects {
            o.shift_world(ox as f32, oz as f32);
        }
        state.cam_pos[0] -= ox as f32;
        state.cam_pos[2] -= oz as f32;
        for c in &mut state.water {
            c.x -= ox;
            c.z -= oz;
        }
    }
    log::info!(target: "saves", "loaded {} ({} bodies{}) in {:.0} ms", path.display(), objects.len(), if endless.is_some() { ", endless" } else { "" },
               t0.elapsed().as_secs_f64() * 1e3);
    Ok(Startup { world, objects, state: Some(state), endless })
}
