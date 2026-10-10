//! The application layer: the headless shot / bench paths and the game window.
pub mod saves;
pub mod shot;
pub mod swapchain;
pub mod window;

use crate::objects::{build_self_test_objects, VoxObject};
use crate::serialize::SceneState;
use crate::voxel::world::World;

/// The startup world: the save named by `load` (a path or a world name in saves/), else (or
/// if it can't be loaded) a generated world with the self-test bodies. A loaded world comes
/// with its state for [`saves::restore_state`].
pub fn startup_world(load: Option<&str>, progress: Option<std::sync::Arc<std::sync::atomic::AtomicI32>>) -> (World, Vec<VoxObject>, Option<SceneState>) {
    let mut world = World::default();
    if let Some(arg) = load {
        let path = saves::resolve(arg);
        match crate::serialize::load_scene(&path) {
            Ok(l) if l.volume.is_some() => {
                let mut objects = Vec::new();
                let (_, st, _) = l.apply(&mut world, &mut objects);
                saves::set_current_world_name(&path.file_stem().map(|s| s.to_string_lossy().into_owned()).unwrap_or_default());
                log::info!(target: "saves", "loaded {} ({} bodies)", path.display(), objects.len());
                return (world, objects, Some(st));
            }
            Ok(_) => log::error!(target: "saves", "{}: an endless world (streaming isn't in this build yet) - generating instead", path.display()),
            Err(e) => log::error!(target: "saves", "can't load {}: {e} - generating instead", path.display()),
        }
    }
    world.gen_progress = progress;
    world.generate(crate::cvars::WORLD_SEED.get() as u32);
    world.gen_progress = None;
    let objects = build_self_test_objects(&world);
    (world, objects, None)
}
