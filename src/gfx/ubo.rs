//! std140 layouts of the shared uniform blocks (common.glsl). Offsets are asserted at
//! compile time, so a field added on one side only fails the build instead of the image.

use bytemuck::{Pod, Zeroable};
use std::mem::{offset_of, size_of};

/// Scene block (set 2, binding 0): world dims, the look, Standard lighting, streaming, clouds.
#[repr(C)]
#[derive(Clone, Copy, Pod, Zeroable)]
pub struct SceneUbo {
    pub atlas_bricks: [u32; 3],
    pub night: f32,
    pub world_bricks: [i32; 3],
    pub num_objects: i32,
    pub sun_dir: [f32; 3],
    pub grass_dist: f32,
    pub sun_rgb: [f32; 3],
    pub sun_angle: f32,
    pub fog_sun_rgb: [f32; 3],
    pub fog_density: f32,
    pub fog_ambient: [f32; 3],
    pub fog_g: f32,
    pub sky_zenith: [f32; 3],
    pub sky_scale: f32,
    pub sky_horizon: [f32; 3],
    pub gi_scale: f32,
    pub sunset_rgb: [f32; 3],
    pub sky_model: i32,
    pub fog_height: f32,
    pub fog_falloff: f32,
    pub fog_noise: f32,
    pub fog_steps: i32,
    pub fog_shadow_max: f32,
    pub tonemap_op: i32,
    pub saturation: f32,
    pub contrast: f32,
    pub white_balance: [f32; 3],
    pub grass_shadows: i32,
    pub sh_right: [f32; 3],
    pub sh_texel0: f32,
    pub sh_up: [f32; 3],
    pub sh_texel1: f32,
    pub sh_lz_top: f32,
    pub sh_texel2: f32,
    pub sh_n: i32,
    pub std_lighting: i32,
    pub sh_win_a: [i32; 4],
    pub sh_win_b: [i32; 4],
    pub probe_origin: [i32; 3],
    pub probe_spacing: f32,
    pub probe_dims: [i32; 3],
    pub probe_update_offset: i32,
    pub probe_hyst: f32,
    pub probe_normal_bias: f32,
    pub std_soft_taps: f32,
    pub std_cascade_blend: f32,
    pub std_cascade_bias: i32,
    pub std_view_bias: f32,
    pub sh_texel3: f32,
    pub std_soft_min: f32,
    pub probe_origin_c: [i32; 3],
    pub probe_spacing_c: f32,
    pub world_win: [i32; 4],
    pub hor_a: [i32; 4],
    pub hor_b: [i32; 4],
    pub chunk_fade: [u32; 256],
    pub gi_cache_cam: [f32; 3],
    pub gi_cache_lod: f32,
    pub gi_cache_p: [i32; 4],
    pub gi_cache_q: [f32; 4],
    pub hor_c: [i32; 4],
    pub cloud_a: [f32; 4],
    pub cloud_b: [f32; 4],
    pub rain_a: [f32; 4],
}
const _: () = {
    assert!(size_of::<SceneUbo>() == 1536);
    assert!(offset_of!(SceneUbo, fog_height) == 144);
    assert!(offset_of!(SceneUbo, sh_win_a) == 240);
    assert!(offset_of!(SceneUbo, probe_origin) == 272);
    assert!(offset_of!(SceneUbo, world_win) == 352);
    assert!(offset_of!(SceneUbo, chunk_fade) == 400);
    assert!(offset_of!(SceneUbo, gi_cache_cam) == 1424);
    assert!(offset_of!(SceneUbo, hor_c) == 1472);
    assert!(offset_of!(SceneUbo, rain_a) == 1520);
};

/// Per-frame camera block (set 3, binding 15 of every pass).
#[repr(C)]
#[derive(Clone, Copy, Pod, Zeroable)]
pub struct PassUbo {
    pub cam_pos: [f32; 3],
    pub lock_active: f32,
    pub cam_right: [f32; 3],
    pub fog_start: f32,
    pub cam_up: [f32; 3],
    pub haze_k: f32,
    pub cam_fwd: [f32; 3],
    pub tan_half_fov: f32,
    pub prev_cam_pos: [f32; 3],
    pub prev_tan_half_fov: f32,
    pub prev_cam_right: [f32; 3],
    pub haze_sea: f32,
    pub prev_cam_up: [f32; 3],
    pub _pf: f32,
    pub prev_cam_fwd: [f32; 3],
    pub frame_idx: u32,
    pub exposure: f32,
    pub u_step: i32,
    pub u_iteration: i32,
    pub debug_mode: i32,
    pub time: f32,
    pub prev_time: f32,
    pub jitter: [f32; 2],
    /// clarity, refraction, max depth, reflection reach
    pub water_p: [f32; 4],
    /// foliage SSS, dust amount, dust density, leaf light
    pub look_p: [f32; 4],
    /// froxel fog: range, slices, near, render aspect
    pub fog_p: [f32; 4],
    /// eye depth under water, murk, caustics, -
    pub under_p: [f32; 4],
    /// hybrid sun shadows, far grass range, cloud accumulation, half-res reflections
    pub opt_p: [f32; 4],
    /// lake waves: wind dir x, z, wind strength, r.water.waves
    pub wave_p: [f32; 4],
    /// floating-origin shift of last frame's positions
    pub shift_p: [f32; 4],
    /// per-voxel facets
    pub facet_p: [f32; 4],
}
const _: () = {
    assert!(size_of::<PassUbo>() == 288);
    assert!(offset_of!(PassUbo, prev_tan_half_fov) == 76);
    assert!(offset_of!(PassUbo, prev_cam_fwd) == 112);
    assert!(offset_of!(PassUbo, frame_idx) == 124);
    assert!(offset_of!(PassUbo, jitter) == 152);
    assert!(offset_of!(PassUbo, facet_p) == 272);
};

/// Locked-view debug block (set 3, binding 14).
#[repr(C)]
#[derive(Clone, Copy, Pod, Zeroable)]
pub struct LockUbo {
    pub pos: [f32; 3],
    pub tan_half_fov: f32,
    pub right: [f32; 3],
    pub aspect: f32,
    pub up: [f32; 3],
    pub grass_dist: f32,
    pub fwd: [f32; 3],
    pub res_y: f32,
    pub mode: i32,
    pub out_w: i32,
    pub out_h: i32,
    pub valid: i32,
}
const _: () = assert!(size_of::<LockUbo>() == 80);

/// Emissive lights + torches (set 2, binding 1: `_LightsU`).
#[repr(C)]
#[derive(Clone, Copy, Pod, Zeroable)]
pub struct LightsUbo {
    pub count: [i32; 4],
    pub a: [[f32; 4]; 32],
    pub b: [[f32; 4]; 32],
    pub c: [[f32; 4]; 32],
    pub torch_count: [i32; 4],
    pub ta: [[f32; 4]; 64],
    pub tb: [[f32; 4]; 64],
}
const _: () = assert!(size_of::<LightsUbo>() == 16 + 3 * 512 + 16 + 2 * 1024);
