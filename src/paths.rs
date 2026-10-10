//! Where the game keeps its files (settings, logs, caches) and finds its shaders.

use std::path::{Path, PathBuf};
use std::sync::OnceLock;

/// The data directory: `VF_DATA` if set; a cargo build (`target/<profile>/voxelforge`) uses
/// the crate root; a packaged build uses the executable's directory.
pub fn data_dir() -> &'static Path {
    static DIR: OnceLock<PathBuf> = OnceLock::new();
    DIR.get_or_init(|| {
        if let Ok(d) = std::env::var("VF_DATA") {
            return PathBuf::from(d);
        }
        let exe = std::env::current_exe().ok().and_then(|p| p.parent().map(Path::to_path_buf)).unwrap_or_else(|| PathBuf::from("."));
        if let Some(profile_dir) = exe.parent() {
            if profile_dir.file_name().is_some_and(|n| n == "target") {
                if let Some(root) = profile_dir.parent() {
                    return root.to_path_buf();
                }
            }
        }
        exe
    })
}

pub fn data_path(rel: &str) -> PathBuf {
    data_dir().join(rel)
}

/// Directories searched for `shaders/<file>` (first hit wins).
pub fn shader_dirs() -> Vec<PathBuf> {
    let mut v = vec![PathBuf::from("shaders"), PathBuf::from("../shaders"), data_path("shaders")];
    v.push(PathBuf::from(concat!(env!("CARGO_MANIFEST_DIR"), "/shaders")));
    v
}
