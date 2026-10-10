//! VoxelForge: a GPU ray-traced voxel engine (Vulkan compute), Rust rewrite.
//!
//! usage: voxelforge [--winsize WxH] [--validation] [+set <cvar> <value>] [+exec "cmd"]
//!                   [--shot out.png [--frames N] [--dump final|albedo|normal|motion]]

// The port is in progress: engine APIs land ahead of the features that call them.
#![allow(dead_code)]

mod app;
mod camera;
mod cvar;
mod cvars;
mod destruction;
mod fluid;
mod gfx;
mod hash;
mod horizon;
mod lights;
mod log;
mod math;
mod objects;
mod paths;
mod physics;
mod player;
mod region_file;
mod serialize;
mod stream;
mod voxel;
mod worldgen;

fn usage() {
    eprintln!(
        "usage: voxelforge [--winsize WxH] [--validation] [--load <world>] [+set <cvar> <value>] [+exec \"cmd\"]\n\
         \x20                 [--shot out.png [--frames N] [--dump final|albedo|normal|motion]]\n\
         \x20 window keys: WASD walk (Shift sprint, Space jump, 2x Space fly), Tab free camera, mouse look (click to capture, Esc frees),\n\
         \x20              F3 stats, F4 debug view, F6 render scale, F5 / F9 quick save / load, F10 reload shaders, L lighting,\n\
         \x20              N denoiser, [ ] exposure, wheel fly speed\n\
         \x20 cvars: VF_SET=\"name=value;...\" or +set; `help` lists them"
    );
}

fn main() {
    let args: Vec<String> = std::env::args().skip(1).collect();
    let headless = args.iter().any(|a| a == "--shot" || a == "--bench");
    log::init(Some(&paths::data_path("logs")));
    ::log::info!(target: "app", "start: {}", std::env::args().collect::<Vec<_>>().join(" "));
    if !headless {
        let _ = cvar::load_archive(&paths::data_path("settings.cfg"));
    }
    cvar::apply_env();
    let args = cvar::apply_args(&args);

    let mut validation = false;
    let (mut w, mut h, mut frames) = (1280u32, 720u32, 0u32);
    let mut shot = String::new();
    let mut dump = String::from("final");
    let mut load = None;
    let mut i = 0;
    while i < args.len() {
        let a = args[i].as_str();
        let next = |i: &mut usize| -> Option<String> {
            *i += 1;
            args.get(*i).cloned()
        };
        match a {
            "--validation" => validation = true,
            "--shot" => shot = next(&mut i).unwrap_or_default(),
            "--frames" => frames = next(&mut i).and_then(|s| s.parse().ok()).unwrap_or(0),
            "--dump" => dump = next(&mut i).unwrap_or_default(),
            "--load" => load = next(&mut i),
            "--winsize" => {
                if let Some((a, b)) = next(&mut i).as_deref().and_then(|s| s.split_once('x')) {
                    w = a.parse().unwrap_or(w);
                    h = b.parse().unwrap_or(h);
                }
            }
            "--help" | "-h" => {
                usage();
                return;
            }
            _ => {
                usage();
                std::process::exit(1);
            }
        }
        i += 1;
    }
    let code = if !shot.is_empty() {
        app::shot::run_shot(&app::shot::ShotArgs { out: shot, w, h, frames: if frames > 0 { frames } else { 48 }, dump, validation, load })
    } else {
        app::window::run(validation, w, h, frames, load)
    };
    ::log::logger().flush();
    std::process::exit(code);
}
