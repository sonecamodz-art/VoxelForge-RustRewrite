//! The full deferred lighting pipeline (all compute), for the headless shot and the window.
//!
//! Two lighting modes:
//!  * ray traced (r.lighting raytraced): per-pixel traced sun + GI rays, denoised by the
//!    in-house SVGF chain (lighting / gi / a-trous / gi_stab / denoise);
//!  * Standard (r.lighting standard): a cached sun shadow map + a DDGI probe grid, one exact
//!    sun ray per pixel softened in screen space, no denoiser.
//!
//! Frames in flight = 2: everything the CPU writes per frame (UBOs, body buffers, set-3
//! descriptor sets, timestamp pools) exists once per frame slot (frame & 1), so recording
//! frame N+1 never touches what frame N is still executing.
//!
//! Usage: `Renderer::new` -> per frame `update_lights`, `set_frame_uniforms`, `update_sets`,
//! `record` / `record_standard` / `record_raw` -> the caller resolves to `out_tex`.

pub mod frame;
pub mod nrd_path;
pub mod passes;
pub mod record;

use ash::vk;
use gpu_allocator::MemoryLocation;

use super::context::{clear_color_f, clear_color_u, image_barrier, to_general, vkc, Access, Buffer, Context, Image, Stage};
use super::gpu_world::{GpuWorld, ABX, ABY};
use super::shader::{self, ComputePipeline, PipelineCache};
use super::ubo::{LightsUbo, LockUbo, PassUbo, SceneUbo};
use crate::cvars;
use crate::lights::LightRegistry;
use crate::math::Vec3;
use crate::objects::VoxObject;
use crate::voxel::world::World;
use passes::{l3_bindings, pass_def, set_layout, L3, L3_COUNT, P, P_COUNT, S, ALL_PASSES, ALL_SETS};

pub const ATROUS_ITERS: usize = 5;
pub const LAMP_SLOTS: usize = 48;
pub const LAMP_CUBE_RES: u32 = 128;
pub const PROBE_DIMS: [i32; 3] = [32, 32, 32];
pub const MAX_PROBES_PER_FRAME: usize = 5120;
pub const PROBE_SPACING_C: f32 = 32.0;
pub const GI_CACHE_SIZE: u32 = 1 << 20;
pub const GI_CACHE_QUEUE_MAX: u32 = 1 << 17;
pub const SHAPE_VOL_DIM: u32 = 256;
pub const MAX_TS: u32 = 32;

/// The macro preamble of every pass (common.glsl's header explains the bridge).
const VK_BRIDGE: &str = "#define SET(x) set = x,\n\
#define BEGIN_SCENE_UNIFORMS layout(set=2, binding=0, std140) uniform _SceneU {\n\
#define END_SCENE_UNIFORMS };\n\
#define BEGIN_PASS_UNIFORMS layout(set=3, binding=15, std140) uniform _PassU {\n\
#define END_PASS_UNIFORMS };\n\
#define UNI\n\
#define DLSS_GUIDES 1\n\
#define PASS_UNIFORMS_BODY vec3 camPos; float lockActive; vec3 camRight; float fogStart; vec3 camUp; float hazeK; vec3 camFwd; \
float tanHalfFov; vec3 prevCamPos; float prevTanHalfFov; vec3 prevCamRight; float hazeSea; vec3 prevCamUp; \
vec3 prevCamFwd; uint frameIdx; float exposure; int uStep; int uIteration; \
int debugMode; float time; float prevTime; vec2 jitter; vec4 waterP; vec4 lookP; vec4 fogP; vec4 underP; vec4 optP; vec4 waveP; vec4 shiftP; vec4 facetP;\n";

fn is_heat_view(v: i32) -> bool {
    v == 2 || v == 3
}

pub fn gi_cache_wanted() -> bool {
    cvars::R_GI_BOUNCES.get() >= 2 && cvars::R_GI_CACHE.get()
}

/// Compile-time switches derived from cvars: pipelines rebuild when this string changes.
pub fn shader_defines() -> String {
    use cvars::*;
    let mut d = String::new();
    let mut def = |s: &str| {
        d += "#define ";
        d += s;
        d += "\n";
    };
    if R_GI_SKYOCC.get() {
        def("SKY_OCC 1");
    }
    if R_GI_BOUNCES.get() >= 2 {
        def("GI_DOUBLE 1");
    }
    if gi_cache_wanted() {
        def("GI_CACHE 1");
    }
    def(&format!("GI_REACH {}.0", R_GI_REACH.get()));
    if is_heat_view(R_DEBUGVIEW.get()) {
        def("DEBUG_HEAT 1");
    }
    if R_TRACE_DISTFIELD.get() {
        def("USE_DIST_FIELD 1");
    }
    if R_TRACE_MASKS.get() {
        def("USE_BRICK_MASKS 1");
    }
    if R_TRACE_OBJBVH.get() {
        def("USE_OBJ_BVH 1");
    }
    if R_AO_HALFRES.get() {
        def("AO_IN_GI 1");
    }
    if R_SVGF_SHADOW.is("soft") {
        def("SVGF_SOFT_SHADOW 1");
    }
    if R_NOISE.is("lds") {
        def("GI_LDS 1");
    }
    if R_GI_BICUBIC.get() {
        def("GI_BICUBIC 1");
    }
    if R_GI_HISTFIX.get() {
        def("GI_HISTFIX 1");
    }
    if !R_GRASS.is("off") {
        def("GRASS_ON 1");
        def(if R_GRASS.is("low") { "GRASS_BLADES 1" } else { "GRASS_BLADES 3" });
    }
    if R_GRASS_STYLE.is("smooth") {
        def("GRASS_SMOOTH 1");
    }
    if R_WATER.get() {
        def("WATER_CLEAR 1");
    }
    if R_REFLECT.get() {
        def("REFL_GLOSSY 1");
    }
    if R_FOLIAGE_SSS.get() > 0.0 {
        def("FOLIAGE_SSS 1");
    }
    if R_FOLIAGE_LIGHT.get() > 0.0 {
        def("FOLIAGE_LIGHT 1");
    }
    if R_FOG_METHOD.is("froxel") {
        def("FOG_FROXEL 1");
    }
    if R_LIGHTS.get() {
        def("LIGHTS_NEE 1");
    }
    if R_LIGHTS_SHADOWS.is("soft") {
        def("LIGHTS_SOFT 1");
    }
    if R_LIGHTS_QUALITY.is("half") {
        def("LIGHTS_HALF 1");
    }
    if R_GI_OBJECTS.get() {
        def("GI_OBJECTS 1");
    }
    let rt_on = RT_ACTIVE.load(std::sync::atomic::Ordering::Relaxed) && rt_any_type();
    if rt_on {
        def("RT_ANY 1");
        for (c, name) in [(&R_RT_SHADOW, "RT_SHADOW"), (&R_RT_GI, "RT_GI"), (&R_RT_REFL, "RT_REFL"), (&R_RT_PROBE, "RT_PROBE"),
                          (&R_RT_LAMP, "RT_LAMP"), (&R_RT_FOG, "RT_FOG"), (&R_RT_PRIMARY, "RT_PRIMARY")] {
            if rt_type(c) {
                def(&format!("{name} 1"));
            }
        }
    }
    if rt_on {
        d.insert_str(0, "#extension GL_EXT_ray_query : require\n");
    }
    d
}

/// The acceleration structures exist (update_rt): the RT_* defines may be on.
static RT_ACTIVE: std::sync::atomic::AtomicBool = std::sync::atomic::AtomicBool::new(false);

/// A ray type on the RT cores (r.rt.hw: all of them).
fn rt_type(c: &crate::cvar::Bool) -> bool {
    cvars::R_RT_HW.get() || c.get()
}
fn rt_any_type() -> bool {
    use cvars::*;
    [&R_RT_SHADOW, &R_RT_GI, &R_RT_REFL, &R_RT_PROBE, &R_RT_LAMP, &R_RT_FOG, &R_RT_PRIMARY].into_iter().any(rt_type)
}
/// Hardware ray queries wanted: the device has them, r.rt, and a ray type asks.
pub fn rt_wanted(ctx: &Context) -> bool {
    ctx.ray_query && cvars::R_RT.get() && rt_any_type()
}

fn fog_slices() -> u32 {
    match cvars::R_FOG_QUALITY.index() {
        0 => 48,
        2 => 96,
        _ => 64,
    }
}

/// The render-resolution images (recreated on resize).
#[derive(Default)]
pub struct Targets {
    pub gb_albedo: Image,
    pub gb_normal: [Image; 2],
    pub gb_pos: [Image; 2],
    pub hist: [Image; 2],
    pub gi_hist: [Image; 2],
    pub gi_mom: [Image; 2],
    pub gi_filt: [Image; 2],
    pub fog_tex: [Image; 2],
    pub out_tex: Image,
    pub gb_motion: Image,
    pub gb_linear_depth: Image,
    pub gb_normal_rough: Image,
    pub gb_spec_albedo: Image,
    pub gb_surf: Image,
    pub gb_diff_albedo: Image,
    pub gb_color_hdr: Image,
    pub dlss_out: Image,
    pub dbg_heat: Image,
    pub gb_depth_hw: Image,
    pub gb_delta: Image,
    pub dlss_out2: Image,
    pub gi_guide: Image,
    pub raw_gi: Image,
    pub lock_pos: Image,
    pub lock_normal: Image,
    pub lock_albedo: Image,
    pub lock_depth: Image,
    pub lock_idx: Image,
    pub shadow_raw: Image,
    pub shadow_tmp: Image,
    pub gi_stab: [Image; 2],
    pub gb_water: Image,
    pub water_add: Image,
    pub refl_half: Image,
    pub cloud_half: Image,
    pub cloud_hist: [Image; 2],
    pub lamp_std: Image,
    pub frox_light: [Image; 2],
    pub frox_vol: Image,
    pub frox_tile: Image,
}

impl Targets {
    fn all_mut(&mut self) -> Vec<&mut Image> {
        let Targets {
            gb_albedo,
            gb_normal,
            gb_pos,
            hist,
            gi_hist,
            gi_mom,
            gi_filt,
            fog_tex,
            out_tex,
            gb_motion,
            gb_linear_depth,
            gb_normal_rough,
            gb_spec_albedo,
            gb_surf,
            gb_diff_albedo,
            gb_color_hdr,
            dlss_out,
            dbg_heat,
            gb_depth_hw,
            gb_delta,
            dlss_out2,
            gi_guide,
            raw_gi,
            lock_pos,
            lock_normal,
            lock_albedo,
            lock_depth,
            lock_idx,
            shadow_raw,
            shadow_tmp,
            gi_stab,
            gb_water,
            water_add,
            refl_half,
            cloud_half,
            cloud_hist,
            lamp_std,
            frox_light,
            frox_vol,
            frox_tile,
        } = self;
        let mut v: Vec<&mut Image> = vec![
            gb_albedo, out_tex, gb_motion, gb_linear_depth, gb_normal_rough, gb_spec_albedo, gb_surf, gb_diff_albedo, gb_color_hdr,
            dlss_out, dbg_heat, gb_depth_hw, gb_delta, dlss_out2, gi_guide, raw_gi, lock_pos, lock_normal, lock_albedo, lock_depth,
            lock_idx, shadow_raw, shadow_tmp, gb_water, water_add, refl_half, cloud_half, lamp_std, frox_vol, frox_tile,
        ];
        for pair in [gb_normal, gb_pos, hist, gi_hist, gi_mom, gi_filt, fog_tex, gi_stab, cloud_hist, frox_light] {
            let [a, b] = pair;
            v.push(a);
            v.push(b);
        }
        v
    }
}

/// Resolution-independent images (created once).
#[derive(Default)]
pub struct Fixed {
    pub shadow_map: Image,
    pub shadow_aux: Image,
    pub probe_irr: Image,
    pub probe_depth: Image,
    pub probe_state: Image,
    pub probe_rays: Image,
    pub probe_data: Image,
    pub wind_disp: Image,
    pub wind_vel: Image,
    pub wind_trample: Image,
    pub lamp_cube: Image,
    pub horizon: Image,
    pub hor_blk: Image,
    pub cloud_pano: Image,
    pub shape_vol: Image,
}

#[derive(Clone, Copy, Debug)]
pub struct LampSlot {
    pub pos: Vec3,
    pub half: Vec3,
    /// -2 = free; -1 = a world light; >= 0 = the body carrying it.
    pub owner: f32,
    pub reach: f32,
    pub used: u32,
    pub ready: bool,
}

impl Default for LampSlot {
    fn default() -> Self {
        Self { pos: Vec3::ZERO, half: Vec3::ZERO, owner: -2.0, reach: 0.0, used: 0, ready: false }
    }
}

#[derive(Clone, Copy, Debug)]
pub struct ShadowRegion {
    pub c: i32,
    pub u0: i32,
    pub v0: i32,
    pub w: i32,
    pub h: i32,
}

/// Standard lighting: CPU side of the cached shadow map + probe grid.
pub struct StdState {
    pub valid: bool,
    pub sun: Vec3,
    pub r: Vec3,
    pub u: Vec3,
    pub lz_top: f32,
    pub win: [[i32; 2]; 4],
    pub edit_seen: u32,
    pub hybrid: bool,
    pub leaf_light: f32,
    pub pending: Vec<ShadowRegion>,
    pub sh_n: i32,
    pub sh_n_alloc: i32,
    pub texel: [f32; 4],
    pub probes_per_frame: i32,
    pub probe_spacing: f32,
    pub probe_origin: [i32; 3],
    pub probe_origin_c: [i32; 3],
    pub rr_cursor: u32,
    pub rr_cursor_c: u32,
    pub boot_frames: i32,
    pub probe_list: Vec<u32>,
    pub edit_boxes: Vec<[i32; 6]>,
    pub taken: Vec<u8>,
}

impl Default for StdState {
    fn default() -> Self {
        Self {
            valid: false,
            sun: Vec3::ZERO,
            r: Vec3::X,
            u: Vec3::Z,
            lz_top: 0.0,
            win: [[0; 2]; 4],
            edit_seen: 0,
            hybrid: false,
            leaf_light: -1.0,
            pending: Vec::new(),
            sh_n: 2048,
            sh_n_alloc: 2048,
            texel: [0.0625, 0.1875, 0.5625, 1.6875],
            probes_per_frame: 2048,
            probe_spacing: 8.0,
            probe_origin: [i32::MIN, 0, 0],
            probe_origin_c: [i32::MIN, 0, 0],
            rr_cursor: 0,
            rr_cursor_c: 0,
            boot_frames: 150,
            probe_list: Vec::new(),
            edit_boxes: Vec::new(),
            taken: Vec::new(),
        }
    }
}

/// Per frame-slot resources.
#[derive(Default)]
pub struct SlotRes {
    pub s1: vk::DescriptorSet,
    pub s2: vk::DescriptorSet,
    pub sets: Vec<vk::DescriptorSet>,
    pub scene_u: Buffer,
    pub main_u: Buffer,
    pub lock_u: Buffer,
    pub probe_list_u: Buffer,
    pub lights_u: Buffer,
    pub ts: vk::QueryPool,
    pub ts_count: u32,
    pub names: Vec<&'static str>,
}

/// Animation state that the C++ kept in function-local statics.
pub struct LookClock {
    pub last_az: f32,
    pub last_el: f32,
    pub drift_x: f64,
    pub drift_z: f64,
    pub evolve: f64,
    pub last: std::time::Instant,
    pub rain_time: f64,
    pub wet: f32,
}

impl Default for LookClock {
    fn default() -> Self {
        Self { last_az: -1.0, last_el: -1000.0, drift_x: 0.0, drift_z: 0.0, evolve: 0.0, last: std::time::Instant::now(), rain_time: 0.0, wet: 0.0 }
    }
}

pub struct Renderer {
    /// Render resolution (all G-buffer / history passes) and its half / quarter.
    pub w: u32,
    pub h: u32,
    pub hw: u32,
    pub hh: u32,
    pub qw: u32,
    pub qh: u32,
    /// Output resolution (out_tex, dlss_out).
    pub ow: u32,
    pub oh: u32,
    pub tan_half_fov: f32,
    pub prev_tan_half_fov: f32,
    /// > 0: the view's FOV this frame instead of r.fov (zoom).
    pub zoom_fov_deg: f32,
    pub gpu: GpuWorld,
    pub pcache: PipelineCache,
    pub t: Targets,
    pub fx: Fixed,
    pub near_s: vk::Sampler,
    pub lin_s: vk::Sampler,

    pub l0: vk::DescriptorSetLayout,
    pub l1: vk::DescriptorSetLayout,
    pub l2: vk::DescriptorSetLayout,
    pub l3: Vec<vk::DescriptorSetLayout>,
    pub pipes: Vec<ComputePipeline>,
    pub pool: vk::DescriptorPool,
    pub s0: vk::DescriptorSet,
    pub s_wind: vk::DescriptorSet,
    pub slots: [SlotRes; 2],
    pub cur: usize,
    pub atrous_ubo: Vec<Buffer>,
    pub sets_dirty: bool,
    pub sets_gi_stab: bool,
    pub compiled_defines: String,

    pub scene_cpu: SceneUbo,
    pub pass_cpu: PassUbo,
    pub lock_cpu: LockUbo,
    pub lights_cpu: Box<LightsUbo>,
    pub lights: LightRegistry,
    pub lamp_slots: Vec<LampSlot>,
    pub lamp_to_trace: Vec<usize>,
    pub lamp_body_box: Vec<(Vec3, Vec3)>,
    pub lamp_frame: u32,
    pub std: StdState,
    pub look: LookClock,
    pub frox_dims: [u32; 3],

    pub last_time: f32,
    pub cam_water_depth: f32,
    pub grass_stamp: [[f32; 4]; 4],
    pub wind_origin: [i32; 2],
    pub wind_reset: bool,
    pub wind_cam: [f32; 2],
    pub wind_dt: f32,
    pub origin_shift: [f32; 2],
    pub chunk_fade: [u32; 256],
    pub hor_win: [i32; 10],
    pub hor_on: i32,
    pub hor_rin: i32,
    pub hor_max_h: i32,
    pub world_sea_level: f32,
    pub world_origin: [i32; 2],

    pub gi_cache_buf: Buffer,
    pub gi_queue_buf: Buffer,
    pub gi_cache_clear: bool,
    pub gi_cache_frame: u32,
    pub gi_cache_boxes: Vec<[i32; 6]>,
    pub gi_cache_reset_all: bool,
    pub gi_cache_sun: Vec3,

    pub lock_valid: bool,
    pub lock_active: bool,
    pub shape_stage: [Buffer; 2],
    pub shape_pending: Vec<u8>,
    pub shape_dims: [u32; 3],
    pub shape_upload: bool,

    pub ts_period_ns: f32,
    /// Last read GPU timings: (label, ms).
    pub pass_ms: Vec<(&'static str, f32)>,
}

fn ubo(ctx: &Context, name: &str, size: usize) -> Buffer {
    ctx.create_buffer(name, size as u64, vk::BufferUsageFlags::UNIFORM_BUFFER, MemoryLocation::CpuToGpu)
}

impl Renderer {
    /// Uploads the scene, creates every image / pipeline / descriptor set for render size
    /// w x h and output size ow x oh.
    pub fn new(ctx: &Context, world: &mut World, objects: &mut [VoxObject], w: u32, h: u32, ow: u32, oh: u32) -> Box<Self> {
        let pcache = PipelineCache::load(ctx);
        let mut r = Box::new(Renderer {
            w,
            h,
            hw: w.div_ceil(2),
            hh: h.div_ceil(2),
            qw: w.div_ceil(4),
            qh: h.div_ceil(4),
            ow,
            oh,
            tan_half_fov: (cvars::R_FOV.get() * 0.5).to_radians().tan(),
            prev_tan_half_fov: 0.0,
            zoom_fov_deg: 0.0,
            gpu: GpuWorld::default(),
            pcache,
            t: Targets::default(),
            fx: Fixed::default(),
            near_s: vk::Sampler::null(),
            lin_s: vk::Sampler::null(),
            l0: vk::DescriptorSetLayout::null(),
            l1: vk::DescriptorSetLayout::null(),
            l2: vk::DescriptorSetLayout::null(),
            l3: Vec::new(),
            pipes: vec![ComputePipeline::default(); P_COUNT],
            pool: vk::DescriptorPool::null(),
            s0: vk::DescriptorSet::null(),
            s_wind: vk::DescriptorSet::null(),
            slots: Default::default(),
            cur: 0,
            atrous_ubo: Vec::new(),
            sets_dirty: true,
            sets_gi_stab: false,
            compiled_defines: String::new(),
            scene_cpu: SceneUbo::zeroed_ubo(),
            pass_cpu: bytemuck::Zeroable::zeroed(),
            lock_cpu: bytemuck::Zeroable::zeroed(),
            lights_cpu: Box::new(bytemuck::Zeroable::zeroed()),
            lights: LightRegistry::default(),
            lamp_slots: vec![LampSlot::default(); LAMP_SLOTS],
            lamp_to_trace: Vec::new(),
            lamp_body_box: Vec::new(),
            lamp_frame: 0,
            std: StdState::default(),
            look: LookClock::default(),
            frox_dims: [0; 3],
            last_time: 0.0,
            cam_water_depth: 0.0,
            grass_stamp: [[0.0; 4]; 4],
            wind_origin: [0; 2],
            wind_reset: true,
            wind_cam: [0.0; 2],
            wind_dt: 0.0,
            origin_shift: [0.0; 2],
            chunk_fade: [0; 256],
            hor_win: [0; 10],
            hor_on: 0,
            hor_rin: 0,
            hor_max_h: 0,
            world_sea_level: crate::worldgen::WATER_LEVEL as f32,
            world_origin: [0; 2],
            gi_cache_buf: Buffer::default(),
            gi_queue_buf: Buffer::default(),
            gi_cache_clear: true,
            gi_cache_frame: u32::MAX,
            gi_cache_boxes: Vec::new(),
            gi_cache_reset_all: false,
            gi_cache_sun: Vec3::ZERO,
            lock_valid: false,
            lock_active: false,
            shape_stage: Default::default(),
            shape_pending: Vec::new(),
            shape_dims: [0; 3],
            shape_upload: false,
            ts_period_ns: ctx.props.limits.timestamp_period,
            pass_ms: Vec::new(),
        });
        for sl in &mut r.slots {
            let qpi = vk::QueryPoolCreateInfo::default().query_type(vk::QueryType::TIMESTAMP).query_count(MAX_TS);
            sl.ts = vkc(unsafe { ctx.device.create_query_pool(&qpi, None) }, "query pool");
        }
        r.gpu.upload(ctx, r.pcache.cache, world, objects);
        r.lights.scan_world(world);
        r.gpu.synced_grid.clear();
        r.alloc_targets(ctx);
        r.near_s = ctx.sampler(vk::Filter::NEAREST);
        r.lin_s = ctx.sampler(vk::Filter::LINEAR);
        r.alloc_fixed(ctx);
        r.invalidate_standard(ctx);
        for i in 0..2 {
            let sl = &mut r.slots[i];
            sl.scene_u = ubo(ctx, "sceneU", std::mem::size_of::<SceneUbo>());
            sl.main_u = ubo(ctx, "passU", std::mem::size_of::<PassUbo>());
            sl.lock_u = ubo(ctx, "lockU", std::mem::size_of::<LockUbo>());
            sl.probe_list_u = ubo(ctx, "probeList", MAX_PROBES_PER_FRAME * 4);
            sl.lights_u = ubo(ctx, "lightsU", std::mem::size_of::<LightsUbo>());
            sl.lights_u.write(0, &vec![0u8; std::mem::size_of::<LightsUbo>()]);
        }
        for i in 0..ATROUS_ITERS {
            let b = ubo(ctx, "atrousU", std::mem::size_of::<PassUbo>());
            let mut ap: PassUbo = bytemuck::Zeroable::zeroed();
            ap.u_step = 1 << i;
            ap.u_iteration = i as i32;
            b.write_pod(0, &ap);
            r.atrous_ubo.push(b);
        }
        r.scene_cpu.atlas_bricks = [ABX as u32, ABY as u32, r.gpu.atlas_bricks_z as u32];
        r.scene_cpu.world_bricks = [r.gpu.wbx, r.gpu.wby, r.gpu.wbz];
        r.scene_cpu.num_objects = r.gpu.num_objects as i32;
        r.set_sun(cvars::R_SUN_AZIMUTH.get(), cvars::R_SUN_ELEVATION.get());
        r.create_layouts(ctx);
        if let Err(e) = r.build_pipelines(ctx, true) {
            crate::log::fatal("shader", &format!("initial pipeline build failed: {e}"));
        }
        r.create_sets(ctx);
        r.write_fixed_sets(ctx);
        r
    }

    // --- images ---------------------------------------------------------------------------------
    pub fn alloc_targets(&mut self, ctx: &Context) {
        use vk::Format as F;
        let (w, h, hw, hh, qw, qh, ow, oh) = (self.w, self.h, self.hw, self.hh, self.qw, self.qh, self.ow, self.oh);
        let i2 = |n: &str, f: F, w: u32, h: u32| ctx.image_2d(n, f, w, h);
        let (f8, f32_, f16, rg16) = (F::R8G8B8A8_UNORM, F::R32G32B32A32_SFLOAT, F::R16G16B16A16_SFLOAT, F::R16G16_SFLOAT);
        let t = &mut self.t;
        t.gb_albedo = i2("gbAlbedo", f8, w, h);
        t.gb_normal = [i2("gbNormal0", f8, w, h), i2("gbNormal1", f8, w, h)];
        t.gb_pos = [i2("gbPos0", f32_, w, h), i2("gbPos1", f32_, w, h)];
        t.hist = [i2("hist0", f32_, w, h), i2("hist1", f32_, w, h)];
        t.gi_hist = [i2("giHist0", f16, hw, hh), i2("giHist1", f16, hw, hh)];
        t.gi_mom = [i2("giMom0", f32_, hw, hh), i2("giMom1", f32_, hw, hh)];
        t.gi_filt = [i2("giFilt0", f16, hw, hh), i2("giFilt1", f16, hw, hh)];
        t.fog_tex = [i2("fogTex0", f16, qw, qh), i2("fogTex1", f16, qw, qh)];
        t.out_tex = i2("outTex", f8, ow, oh);
        t.gb_motion = i2("gbMotion", rg16, w, h);
        t.gb_linear_depth = i2("gbLinearDepth", F::R32_SFLOAT, w, h);
        t.gb_normal_rough = i2("gbNormalRough", f16, w, h);
        t.gb_spec_albedo = i2("gbSpecAlbedo", f8, w, h);
        t.gb_surf = i2("gbSurf", f8, w, h);
        t.gb_diff_albedo = i2("gbDiffAlbedo", f8, w, h);
        t.gb_color_hdr = i2("gbColorHDR", f16, w, h);
        t.dlss_out = i2("dlssOut", f16, ow, oh);
        t.dbg_heat = i2("dbgHeat", f16, w, h);
        t.gb_depth_hw = i2("gbDepthHW", F::R32_SFLOAT, w, h);
        t.gb_delta = i2("gbDelta", f16, w, h);
        t.dlss_out2 = i2("dlssOut2", f16, ow, oh);
        t.gi_guide = i2("giGuide", F::R32G32B32A32_UINT, hw, hh);
        t.raw_gi = i2("rawGi", f16, hw, hh);
        t.lock_pos = i2("lockPos", f32_, w, h);
        t.lock_normal = i2("lockNormal", f8, w, h);
        t.lock_albedo = i2("lockAlbedo", f8, w, h);
        t.lock_depth = i2("lockDepth", F::R32_UINT, w, h);
        t.lock_idx = i2("lockIdx", F::R32_UINT, w, h);
        t.shadow_raw = i2("shadowRaw", rg16, w, h);
        t.shadow_tmp = i2("shadowTmp", rg16, w, h);
        t.gi_stab = [i2("giStab0", f16, hw, hh), i2("giStab1", f16, hw, hh)];
        t.gb_water = i2("gbWater", f16, w, h);
        t.water_add = i2("waterAdd", f16, w, h);
        t.refl_half = i2("reflHalf", f16, w.div_ceil(2), h.div_ceil(2));
        t.cloud_half = i2("cloudHalf", f16, w.div_ceil(2), h.div_ceil(2));
        t.cloud_hist = [i2("cloudHist0", f16, w.div_ceil(2), h.div_ceil(2)), i2("cloudHist1", f16, w.div_ceil(2), h.div_ceil(2))];
        t.lamp_std = i2("lampStd", f16, w, h);
        self.alloc_froxels_inner(ctx);
        self.lock_valid = false;
        self.sets_dirty = true;
        let dev = &ctx.device;
        let cb = ctx.begin_one_shot();
        let imgs: Vec<vk::Image> = self.t.all_mut().into_iter().map(|i| i.image).collect();
        to_general(dev, cb, &imgs);
        self.record_history_clear(dev, cb);
        clear_color_f(dev, cb, self.t.lock_pos.image, [0.0, 0.0, 0.0, -1.0]);
        let t = &self.t;
        for im in [&t.frox_light[0], &t.frox_light[1]] {
            clear_color_f(dev, cb, im.image, [0.0; 4]);
        }
        clear_color_f(dev, cb, t.frox_vol.image, [0.0, 0.0, 0.0, 1.0]); // no fog yet: transmittance 1
        ctx.end_one_shot_and_wait(cb);
    }

    /// Froxel fog volumes: 8-px tiles x r.fog.quality slices (images only; caller clears).
    fn alloc_froxels_inner(&mut self, ctx: &Context) {
        let [l0, l1] = &mut self.t.frox_light;
        for im in [l0, l1, &mut self.t.frox_vol, &mut self.t.frox_tile] {
            im.destroy(ctx);
        }
        self.frox_dims = [self.w.div_ceil(8), self.h.div_ceil(8), fog_slices()];
        let [fx, fy, fz] = self.frox_dims;
        let u = vk::ImageUsageFlags::STORAGE | vk::ImageUsageFlags::SAMPLED | vk::ImageUsageFlags::TRANSFER_DST;
        let f16 = vk::Format::R16G16B16A16_SFLOAT;
        self.t.frox_light = [ctx.image_3d("froxLight0", f16, fx, fy, fz, u), ctx.image_3d("froxLight1", f16, fx, fy, fz, u)];
        self.t.frox_vol = ctx.image_3d("froxVol", f16, fx, fy, fz, u);
        self.t.frox_tile = ctx.image_2d("froxTile", vk::Format::R32_SFLOAT, fx, fy);
    }

    /// r.fog.quality changed: new froxel volumes (cleared).
    fn realloc_froxels(&mut self, ctx: &Context) {
        ctx.wait_idle();
        self.alloc_froxels_inner(ctx);
        let dev = &ctx.device;
        let cb = ctx.begin_one_shot();
        let t = &self.t;
        to_general(dev, cb, &[t.frox_light[0].image, t.frox_light[1].image, t.frox_vol.image, t.frox_tile.image]);
        clear_color_f(dev, cb, t.frox_light[0].image, [0.0; 4]);
        clear_color_f(dev, cb, t.frox_light[1].image, [0.0; 4]);
        clear_color_f(dev, cb, t.frox_vol.image, [0.0, 0.0, 0.0, 1.0]);
        ctx.end_one_shot_and_wait(cb);
        self.sets_dirty = true;
    }

    /// Reset the temporal buffers to their frame-0 state (images already GENERAL).
    pub fn record_history_clear(&mut self, dev: &ash::Device, cb: vk::CommandBuffer) {
        self.gi_cache_clear = true;
        let t = &self.t;
        clear_color_f(dev, cb, t.gb_pos[0].image, [0.0, 0.0, 0.0, -1.0]);
        clear_color_f(dev, cb, t.gb_pos[1].image, [0.0, 0.0, 0.0, -1.0]);
        for im in &t.cloud_hist {
            clear_color_f(dev, cb, im.image, [-1.0; 4]);
        }
        for im in [&t.hist, &t.gi_hist, &t.gi_mom, &t.gi_filt, &t.fog_tex, &t.gi_stab].into_iter().flatten() {
            clear_color_f(dev, cb, im.image, [0.0; 4]);
        }
    }

    fn destroy_targets(&mut self, ctx: &Context) {
        for im in self.t.all_mut() {
            im.destroy(ctx);
        }
    }

    /// Recreate only the resolution-dependent images (window resize / render scale).
    pub fn resize(&mut self, ctx: &Context, w: u32, h: u32, ow: u32, oh: u32) {
        ctx.wait_idle();
        self.destroy_targets(ctx);
        self.w = w;
        self.h = h;
        self.ow = ow;
        self.oh = oh;
        self.hw = w.div_ceil(2);
        self.hh = h.div_ceil(2);
        self.qw = w.div_ceil(4);
        self.qh = h.div_ceil(4);
        self.alloc_targets(ctx);
    }

    fn alloc_fixed(&mut self, ctx: &Context) {
        use vk::Format as F;
        let n = self.std.sh_n_alloc as u32;
        let tiles_x = (PROBE_DIMS[0] * 8) as u32;
        let tiles_y = (PROBE_DIMS[1] * (PROBE_DIMS[2] / 8)) as u32;
        let probes = (2 * PROBE_DIMS[0] * PROBE_DIMS[1] * PROBE_DIMS[2] / 2048) as u32;
        let f = &mut self.fx;
        f.shadow_map = ctx.image_2d("shadowMap", F::R32_SFLOAT, n, 4 * n);
        f.shadow_aux = ctx.image_2d("shadowAux", F::R16_SFLOAT, n, 4 * n);
        f.probe_irr = ctx.image_2d("probeIrr", F::R32G32B32A32_SFLOAT, tiles_x * 10, 2 * tiles_y * 10);
        f.probe_depth = ctx.image_2d("probeDepth", F::R32G32_SFLOAT, tiles_x * 18, 2 * tiles_y * 18);
        f.probe_state = ctx.image_2d("probeState", F::R32_UINT, 2048, probes);
        f.probe_rays = ctx.image_2d("probeRays", F::R16G16B16A16_SFLOAT, 64, MAX_PROBES_PER_FRAME as u32);
        f.probe_data = ctx.image_2d("probeData", F::R16G16B16A16_SFLOAT, 2048, probes);
        f.wind_disp = ctx.image_2d("windDisp", F::R16G16B16A16_SFLOAT, 128, 128);
        f.wind_vel = ctx.image_2d("windVel", F::R16G16B16A16_SFLOAT, 128, 128);
        f.wind_trample = ctx.image_2d("windTrample", F::R16G16B16A16_SFLOAT, 128, 128);
        f.lamp_cube = ctx.image_2d("lampCube", F::R16_SFLOAT, 6 * LAMP_CUBE_RES, LAMP_SLOTS as u32 * LAMP_CUBE_RES);
        f.horizon = ctx.image_2d("horizon", F::R32G32_UINT, 1024, 5 * 1024);
        f.hor_blk = ctx.image_2d("horBlk", F::R32_UINT, 128, 5 * 128);
        f.cloud_pano = ctx.image_2d("cloudPano", F::R16G16B16A16_SFLOAT, 256, 128);
        f.shape_vol = ctx.image_3d("shapeVol", F::R8_UINT, SHAPE_VOL_DIM, SHAPE_VOL_DIM, SHAPE_VOL_DIM,
                                   vk::ImageUsageFlags::SAMPLED | vk::ImageUsageFlags::TRANSFER_DST);
        let dev = &ctx.device;
        let cb = ctx.begin_one_shot();
        let f = &self.fx;
        let imgs = [
            f.shadow_map.image, f.shadow_aux.image, f.probe_irr.image, f.probe_depth.image, f.probe_state.image, f.probe_rays.image,
            f.probe_data.image, f.wind_disp.image, f.wind_vel.image, f.wind_trample.image, f.lamp_cube.image, f.horizon.image,
            f.hor_blk.image, f.cloud_pano.image, f.shape_vol.image,
        ];
        to_general(dev, cb, &imgs);
        clear_color_u(dev, cb, f.horizon.image, [0; 4]);
        clear_color_u(dev, cb, f.hor_blk.image, [0; 4]);
        clear_color_u(dev, cb, f.shape_vol.image, [0; 4]);
        clear_color_f(dev, cb, f.cloud_pano.image, [0.0, 0.0, 0.0, 1.0]);
        ctx.end_one_shot_and_wait(cb);
    }

    /// World replaced / Standard (re)started: re-trace every cascade, reset the probes.
    pub fn invalidate_standard(&mut self, ctx: &Context) {
        self.std.valid = false;
        self.reset_lamp_cache();
        self.std.pending.clear();
        self.std.edit_seen = self.gpu.edit_counter;
        ctx.wait_idle();
        let dev = &ctx.device;
        let cb = ctx.begin_one_shot();
        clear_color_f(dev, cb, self.fx.shadow_map.image, [-1e30, 0.0, 0.0, 0.0]);
        for im in [&self.fx.probe_irr, &self.fx.probe_depth, &self.fx.probe_rays, &self.fx.probe_data] {
            clear_color_f(dev, cb, im.image, [0.0; 4]);
        }
        clear_color_u(dev, cb, self.fx.probe_state.image, [0; 4]);
        self.std.probe_origin[0] = i32::MIN;
        self.std.probe_origin_c[0] = i32::MIN;
        self.std.boot_frames = 150;
        ctx.end_one_shot_and_wait(cb);
    }

    pub fn reset_lamp_cache(&mut self) {
        self.lamp_slots.iter_mut().for_each(|s| *s = LampSlot::default());
        self.lamp_body_box.clear();
    }

    // --- layouts / pipelines ------------------------------------------------------------------------
    fn create_layouts(&mut self, ctx: &Context) {
        use vk::DescriptorType as DT;
        let dev = &ctx.device;
        let cis = DT::COMBINED_IMAGE_SAMPLER;
        let l0: Vec<(u32, DT)> = [0u32, 1, 8, 9, 10, 11, 12, 13, 14, 15, 16, 17, 18, 19, 20, 21, 22, 23, 24, 25, 26, 27].iter().map(|&b| (b, cis)).collect();
        self.l0 = shader::make_set_layout(dev, &l0);
        let sb = DT::STORAGE_BUFFER;
        let mut l1 = vec![(1, sb), (2, sb), (3, sb), (4, sb), (5, sb), (6, sb), (8, sb), (9, sb)];
        if ctx.ray_query {
            // ray queries: 0 the TLAS, 7 the primitive -> brick table (gfx::rt)
            l1.extend([(0, DT::ACCELERATION_STRUCTURE_KHR), (7, sb)]);
        }
        self.l1 = shader::make_set_layout(dev, &l1);
        self.l2 = shader::make_set_layout(dev, &[(0, DT::UNIFORM_BUFFER), (1, DT::UNIFORM_BUFFER)]);
        self.l3 = (0..L3_COUNT)
            .map(|i| {
                let l: L3 = unsafe { std::mem::transmute::<usize, L3>(i) };
                shader::make_set_layout(dev, &l3_bindings(l))
            })
            .collect();
    }

    /// Compile every pass (in parallel, through the SPIR-V cache); swap the pipelines only if
    /// all compiled, so a typo during a live reload keeps the running renderer.
    pub fn build_pipelines(&mut self, ctx: &Context, initial: bool) -> Result<(), String> {
        use rayon::prelude::*;
        let t0 = std::time::Instant::now();
        let macros = shader_defines();
        let bridge = format!("{VK_BRIDGE}{macros}");
        let spvs: Vec<Result<Vec<u32>, String>> = ALL_PASSES
            .par_iter()
            .map(|&p| {
                let d = pass_def(p);
                let mut parts = vec!["common.glsl"];
                parts.extend_from_slice(d.extra);
                parts.push(d.file);
                shader::compile_parts(&parts, &format!("{bridge}{}", d.defs)).map_err(|e| format!("{}:\n{e}", d.file))
            })
            .collect();
        let errors: Vec<&String> = spvs.iter().filter_map(|r| r.as_ref().err()).collect();
        if !errors.is_empty() {
            for e in &errors {
                log::error!(target: "shader", "{e}");
            }
            return Err(format!("{} pass(es) failed{}", errors.len(), if initial { "" } else { " - keeping the previous pipelines" }));
        }
        if !initial {
            ctx.wait_idle();
        }
        let set_layouts = |l: L3| [self.l0, self.l1, self.l2, self.l3[l as usize]];
        for (i, spv) in spvs.into_iter().enumerate() {
            let d = pass_def(ALL_PASSES[i]);
            self.pipes[i].destroy(&ctx.device);
            self.pipes[i] = shader::make_compute_pipeline(&ctx.device, self.pcache.cache, &spv.unwrap(), &set_layouts(d.layout), d.push);
            ctx.name(self.pipes[i].pipeline, d.file);
        }
        self.compiled_defines = macros;
        self.pcache.save(ctx);
        log::info!(target: "shader", "{} {} compute pipelines in {:.0} ms", if initial { "built" } else { "reloaded" }, P_COUNT,
                   t0.elapsed().as_secs_f64() * 1e3);
        Ok(())
    }

    pub fn reload_shaders(&mut self, ctx: &Context) -> bool {
        match self.build_pipelines(ctx, false) {
            Ok(()) => true,
            Err(e) => {
                log::error!(target: "shader", "{e}");
                false
            }
        }
    }

    #[inline]
    pub fn pipe(&self, p: P) -> &ComputePipeline {
        &self.pipes[p as usize]
    }

    #[inline]
    pub fn set(&self, s: S) -> vk::DescriptorSet {
        self.slots[self.cur].sets[s as usize]
    }

    fn create_sets(&mut self, ctx: &Context) {
        use vk::DescriptorType as DT;
        let dev = &ctx.device;
        let sizes = [
            vk::DescriptorPoolSize { ty: DT::COMBINED_IMAGE_SAMPLER, descriptor_count: 560 },
            vk::DescriptorPoolSize { ty: DT::STORAGE_IMAGE, descriptor_count: 280 },
            vk::DescriptorPoolSize { ty: DT::STORAGE_BUFFER, descriptor_count: 40 },
            vk::DescriptorPoolSize { ty: DT::UNIFORM_BUFFER, descriptor_count: 160 },
            vk::DescriptorPoolSize { ty: DT::ACCELERATION_STRUCTURE_KHR, descriptor_count: 4 },
        ];
        self.pool = vkc(unsafe { dev.create_descriptor_pool(&vk::DescriptorPoolCreateInfo::default().max_sets(128).pool_sizes(&sizes), None) }, "descriptor pool");
        let alloc = |l: vk::DescriptorSetLayout| -> vk::DescriptorSet {
            let ls = [l];
            vkc(unsafe { dev.allocate_descriptor_sets(&vk::DescriptorSetAllocateInfo::default().descriptor_pool(self.pool).set_layouts(&ls)) }, "allocate set")[0]
        };
        self.s0 = alloc(self.l0);
        self.s_wind = alloc(self.l3[L3::Wind as usize]);
        for k in 0..2 {
            let s1 = alloc(self.l1);
            let s2 = alloc(self.l2);
            let sets: Vec<vk::DescriptorSet> = ALL_SETS.iter().map(|&s| alloc(self.l3[set_layout(s) as usize])).collect();
            let sl = &mut self.slots[k];
            sl.s1 = s1;
            sl.s2 = s2;
            sl.sets = sets;
        }
    }

    /// Set 0 (scene textures) + the wind set: fixed images, written once (and after a scene
    /// reload or a shadow map resize).
    pub fn write_fixed_sets(&mut self, ctx: &Context) {
        let ro = vk::ImageLayout::SHADER_READ_ONLY_OPTIMAL;
        let genl = vk::ImageLayout::GENERAL;
        let g = &self.gpu;
        let f = &self.fx;
        let tex: Vec<(u32, vk::ImageView, vk::Sampler, vk::ImageLayout)> = vec![
            (0, g.grid.view, g.sampler, ro),
            (1, g.atlas.view, g.sampler, ro),
            (8, g.coarse1.view, g.sampler, ro),
            (9, g.coarse2.view, g.sampler, ro),
            (10, g.dist[0].view, g.sampler, genl),
            (20, g.dist[2].view, g.sampler, genl),
            (11, g.grass_top.view, g.sampler, genl),
            (12, g.grass_tile.view, g.sampler, genl),
            (13, g.grass_info.view, g.sampler, genl),
            (25, g.sky_top.view, g.sampler, genl),
            (14, f.shadow_map.view, self.near_s, genl),
            (15, f.probe_irr.view, self.lin_s, genl),
            (16, f.probe_depth.view, self.lin_s, genl),
            (17, f.probe_state.view, self.near_s, genl),
            (18, f.probe_data.view, self.near_s, genl),
            (19, f.wind_disp.view, self.near_s, genl),
            (21, f.wind_trample.view, self.near_s, genl),
            (22, f.lamp_cube.view, self.near_s, genl),
            (23, f.shadow_aux.view, self.near_s, genl),
            (24, f.horizon.view, self.near_s, genl),
            (26, f.hor_blk.view, self.near_s, genl),
            (27, f.cloud_pano.view, self.lin_s, genl),
        ];
        let mut w = SetWriter::default();
        for (b, v, s, l) in tex {
            w.tex_l(self.s0, b, v, s, l);
        }
        w.img(self.s_wind, 0, f.wind_disp.view);
        w.img(self.s_wind, 1, f.wind_vel.view);
        w.img(self.s_wind, 2, f.wind_trample.view);
        w.flush(&ctx.device);
    }

    /// Bake both slots' set-1/2/3 descriptor sets (waits idle: in-flight frames bind them).
    pub fn write_sets(&mut self, ctx: &Context) {
        ctx.wait_idle();
        let (ns, ls) = (self.near_s, self.lin_s);
        for pp in 0..2 {
            let o = 1 - pp;
            let sl = &self.slots[pp];
            let st = |s: S| sl.sets[s as usize];
            let t = &self.t;
            let g = &self.gpu;
            let mut w = SetWriter::default();
            // set 1 (SSBOs; bodies + BVH per slot) and set 2 (scene UBO, lights per slot)
            w.sbuf(sl.s1, 1, g.palette_buf.buf);
            w.sbuf(sl.s1, 2, g.objects_buf[pp].buf);
            w.sbuf(sl.s1, 3, g.obj_grid_buf.buf);
            w.sbuf(sl.s1, 4, g.mask_buf.buf);
            w.sbuf(sl.s1, 5, g.bvh_node_buf[pp].buf);
            w.sbuf(sl.s1, 6, g.bvh_prim_buf[pp].buf);
            if !self.gi_cache_buf.is_null() {
                w.sbuf(sl.s1, 8, self.gi_cache_buf.buf);
                w.sbuf(sl.s1, 9, self.gi_queue_buf.buf);
            }
            if g.rt.active() {
                // ray queries: the TLAS + the primitive table
                w.sbuf(sl.s1, 7, g.rt.prim_buffer());
                w.accel(sl.s1, 0, g.rt.tlas());
            }
            w.ubuf(sl.s2, 0, sl.scene_u.buf);
            w.ubuf(sl.s2, 1, sl.lights_u.buf);
            let main = sl.main_u.buf;
            // gbuffer (+ guide outputs)
            let s = st(S::Gbuf);
            w.img(s, 0, t.gb_albedo.view);
            w.img(s, 1, t.gb_normal[pp].view);
            w.img(s, 2, t.gb_pos[pp].view);
            w.img(s, 3, t.gb_motion.view);
            w.img(s, 10, t.lock_idx.view);
            w.tex(s, 11, t.lock_pos.view, ns);
            w.tex(s, 12, t.lock_normal.view, ns);
            w.tex(s, 13, t.lock_albedo.view, ns);
            w.img(s, 4, t.gb_linear_depth.view);
            w.img(s, 5, t.gb_normal_rough.view);
            w.img(s, 6, t.gb_spec_albedo.view);
            w.img(s, 7, t.dbg_heat.view);
            w.img(s, 8, t.gb_depth_hw.view);
            w.img(s, 9, t.gb_delta.view);
            w.img(s, 14, t.gb_water.view);
            w.img(s, 16, t.gb_surf.view);
            w.img(s, 17, t.gb_diff_albedo.view);
            w.ubuf(s, 15, main);
            w.tex(st(S::Denoise), 17, t.gb_surf.view, ns);
            w.tex(st(S::Raw), 17, t.gb_surf.view, ns);
            w.tex(st(S::Std), 18, t.gb_surf.view, ns);
            // lighting
            let s = st(S::Light);
            w.img(s, 1, t.hist[pp].view);
            w.tex(s, 2, t.gb_albedo.view, ns);
            w.tex(s, 3, t.gb_normal[pp].view, ns);
            w.tex(s, 4, t.gb_pos[pp].view, ns);
            w.tex(s, 5, t.gb_pos[o].view, ns);
            w.tex(s, 6, t.hist[o].view, ns);
            w.tex(s, 7, t.gb_normal[o].view, ns);
            w.img(s, 8, t.dbg_heat.view);
            w.tex(s, 9, t.gb_motion.view, ns);
            w.tex(s, 10, t.gb_delta.view, ns);
            w.ubuf(s, 15, main);
            // gi
            let s = st(S::Gi);
            w.img(s, 0, t.gi_hist[pp].view);
            w.img(s, 1, t.gi_mom[pp].view);
            w.tex(s, 3, t.gb_normal[pp].view, ns);
            w.tex(s, 4, t.gb_pos[pp].view, ns);
            w.tex(s, 5, t.gb_pos[o].view, ns);
            w.tex(s, 7, t.gb_normal[o].view, ns);
            w.tex(s, 10, t.gi_hist[o].view, ns);
            w.tex(s, 11, t.gi_mom[o].view, ns);
            w.tex(s, 12, t.gb_motion.view, ns);
            w.tex(s, 13, t.gb_delta.view, ns);
            w.img(s, 14, t.gi_guide.view);
            w.ubuf(s, 15, main);
            // a-trous levels (ping-pong; odd level counts end in gi_filt[0])
            for (it, &s) in passes::ATROUS_SETS.iter().enumerate() {
                let s = st(s);
                let (dst, src) = (it & 1, (it + 1) & 1);
                w.img(s, 0, t.gi_filt[dst].view);
                w.tex(s, 2, t.gi_filt[if it == 0 { 0 } else { src }].view, ns);
                w.tex(s, 3, t.gb_normal[pp].view, ns);
                w.tex(s, 4, t.gb_pos[pp].view, ns);
                w.tex(s, 5, t.gi_mom[pp].view, ns);
                w.tex(s, 6, t.gi_hist[pp].view, ns);
                w.tex(s, 7, t.gi_guide.view, ns);
                w.ubuf(s, 15, self.atrous_ubo[it].buf);
            }
            // fog (old quarter-res march)
            let s = st(S::Fog);
            w.img(s, 0, t.fog_tex[pp].view);
            w.tex(s, 4, t.gb_pos[pp].view, ns);
            w.tex(s, 5, t.fog_tex[o].view, ls);
            w.tex(s, 6, t.gb_pos[o].view, ns);
            w.ubuf(s, 15, main);
            // denoise
            let s = st(S::Denoise);
            w.img(s, 0, t.out_tex.view);
            w.img(s, 1, t.gb_color_hdr.view);
            w.tex(s, 2, t.gb_albedo.view, ns);
            w.tex(s, 3, t.gb_normal[pp].view, ns);
            w.tex(s, 4, t.gb_pos[pp].view, ns);
            w.tex(s, 5, t.fog_tex[pp].view, ls);
            w.tex(s, 6, t.hist[pp].view, ns);
            w.tex(s, 7, if cvars::R_GI_STAB.get() { t.gi_stab[pp].view } else { t.gi_filt[0].view }, ns);
            w.tex(s, 8, t.dbg_heat.view, ns);
            w.tex(s, 9, t.gi_guide.view, ns);
            w.tex(s, 10, t.shadow_raw.view, ns);
            w.tex(s, 11, t.gi_hist[pp].view, ns);
            w.tex(s, 12, t.gb_water.view, ns);
            w.tex(s, 13, t.water_add.view, ns);
            w.ubuf(s, 15, main);
            w.tex(st(S::Raw), 12, t.gb_water.view, ns);
            w.tex(st(S::Raw), 13, t.water_add.view, ns);
            w.tex(st(S::Std), 12, t.gb_water.view, ns);
            w.tex(st(S::Std), 13, t.water_add.view, ns);
            // water
            let s = st(S::WaterRefl);
            w.img(s, 0, t.water_add.view);
            w.tex(s, 1, t.gb_water.view, ns);
            w.tex(s, 4, t.gb_pos[pp].view, ns);
            w.ubuf(s, 15, main);
            w.tex(s, 2, t.refl_half.view, ns);
            w.tex(s, 3, t.gb_normal[pp].view, ns);
            w.tex(s, 5, t.gb_surf.view, ns);
            w.tex(s, 6, t.gb_albedo.view, ns);
            for (i, &ts) in passes::TORCH_SETS.iter().enumerate() {
                let s = st(ts);
                let target = [&t.dlss_out, &t.gb_color_hdr, &t.dlss_out2][i];
                w.img(s, 0, target.view);
                w.tex(s, 3, t.gb_normal[pp].view, ns);
                w.tex(s, 4, t.gb_pos[pp].view, ns);
                w.ubuf(s, 15, main);
            }
            let s = st(S::WaterReflHalf);
            w.img(s, 0, t.refl_half.view);
            w.tex(s, 1, t.gb_water.view, ns);
            w.tex(s, 4, t.gb_pos[pp].view, ns);
            w.ubuf(s, 15, main);
            // clouds
            let s = st(S::Clouds);
            w.img(s, 0, t.cloud_half.view);
            w.tex(s, 4, t.gb_pos[pp].view, ns);
            w.ubuf(s, 15, main);
            let s = st(S::CloudPano);
            w.img(s, 0, self.fx.cloud_pano.view);
            w.ubuf(s, 15, main);
            let s = st(S::CloudRes);
            w.img(s, 0, t.cloud_hist[pp].view);
            w.tex(s, 1, t.cloud_half.view, ns);
            w.tex(s, 2, t.cloud_hist[o].view, ls);
            w.ubuf(s, 15, main);
            w.tex(st(S::Raw), 20, t.cloud_hist[pp].view, ls);
            w.tex(st(S::Denoise), 20, t.cloud_hist[pp].view, ls);
            w.tex(st(S::Std), 20, t.cloud_hist[pp].view, ls);
            // brush preview + dust (over out_tex)
            let s = st(S::Brush);
            w.img(s, 0, t.out_tex.view);
            w.tex(s, 3, t.gb_normal[pp].view, ns);
            w.tex(s, 4, t.gb_pos[pp].view, ns);
            w.tex(s, 5, self.fx.shape_vol.view, ns);
            w.ubuf(s, 15, main);
            let s = st(S::Dust);
            w.img(s, 0, t.out_tex.view);
            w.tex(s, 4, t.gb_pos[pp].view, ns);
            w.ubuf(s, 15, main);
            // froxel fog
            w.tex(st(S::Denoise), 14, t.frox_vol.view, ls);
            w.tex(st(S::Raw), 14, t.frox_vol.view, ls);
            w.tex(st(S::Std), 14, t.frox_vol.view, ls);
            for s in [S::Denoise, S::Raw, S::Std] {
                w.tex(st(s), 16, t.gb_delta.view, ns); // .a = body id
            }
            let s = st(S::FroxInj);
            w.img(s, 0, t.frox_light[pp].view);
            w.tex(s, 1, t.frox_light[o].view, ls);
            w.tex(s, 2, t.frox_tile.view, ns);
            w.ubuf(s, 15, main);
            let s = st(S::FroxTile);
            w.img(s, 0, t.frox_tile.view);
            w.tex(s, 4, t.gb_pos[pp].view, ns);
            w.ubuf(s, 15, main);
            let s = st(S::FroxInt);
            w.img(s, 0, t.frox_vol.view);
            w.tex(s, 1, t.frox_light[pp].view, ns);
            w.ubuf(s, 15, main);
            // gi_stab
            let s = st(S::GiStab);
            w.img(s, 0, t.gi_stab[pp].view);
            w.tex(s, 1, t.gi_filt[0].view, ns);
            w.tex(s, 2, t.gi_stab[o].view, ns);
            w.tex(s, 3, t.gb_normal[pp].view, ns);
            w.tex(s, 4, t.gb_pos[pp].view, ns);
            w.tex(s, 5, t.gb_pos[o].view, ns);
            w.tex(s, 6, t.gb_normal[o].view, ns);
            w.tex(s, 7, t.gb_motion.view, ns);
            w.tex(s, 8, t.gb_delta.view, ns);
            w.tex(s, 9, t.gi_guide.view, ns);
            w.ubuf(s, 15, main);
            // raw_shade (the raw 1-spp radiance)
            let s = st(S::Raw);
            w.img(s, 0, t.gb_color_hdr.view);
            w.tex(s, 2, t.gb_albedo.view, ns);
            w.tex(s, 3, t.gb_normal[pp].view, ns);
            w.tex(s, 4, t.gb_pos[pp].view, ns);
            w.tex(s, 5, t.fog_tex[pp].view, ls);
            w.tex(s, 6, t.raw_gi.view, ns);
            w.ubuf(s, 15, main);
            w.img(s, 7, t.lamp_std.view);
            let s = st(S::RawLamp);
            w.img(s, 0, t.gb_color_hdr.view);
            w.img(s, 1, t.lamp_std.view);
            w.tex(s, 3, t.gb_normal[pp].view, ns);
            w.tex(s, 4, t.gb_pos[pp].view, ns);
            w.ubuf(s, 15, main);
            let s = st(S::RawGi);
            w.img(s, 0, t.raw_gi.view);
            w.tex(s, 3, t.gb_normal[pp].view, ns);
            w.tex(s, 4, t.gb_pos[pp].view, ns);
            w.ubuf(s, 15, main);
            // locked view
            let s = st(S::Lock);
            w.img(s, 0, t.out_tex.view);
            w.tex(s, 1, t.gb_pos[pp].view, ns);
            w.tex(s, 2, t.lock_pos.view, ns);
            w.ubuf(s, 14, sl.lock_u.buf);
            w.ubuf(s, 15, main);
            let s = st(S::Splat);
            w.img(s, 0, t.lock_depth.view);
            w.img(s, 1, t.lock_idx.view);
            w.tex(s, 2, t.lock_pos.view, ns);
            w.tex(s, 3, t.lock_normal.view, ns);
            w.ubuf(s, 14, sl.lock_u.buf);
            w.ubuf(s, 15, main);
            // Standard lighting
            let f = &self.fx;
            let s = st(S::Shadow);
            w.img(s, 0, f.shadow_map.view);
            w.img(s, 1, f.shadow_aux.view);
            w.ubuf(s, 15, main);
            let s = st(S::ProbeT);
            w.img(s, 0, f.probe_rays.view);
            w.ubuf(s, 5, sl.probe_list_u.buf);
            w.ubuf(s, 15, main);
            let s = st(S::ProbeU);
            w.img(s, 0, f.probe_irr.view);
            w.img(s, 1, f.probe_depth.view);
            w.img(s, 2, f.probe_state.view);
            w.tex(s, 3, f.probe_rays.view, ns);
            w.img(s, 4, f.probe_data.view);
            w.ubuf(s, 5, sl.probe_list_u.buf);
            w.ubuf(s, 15, main);
            let s = st(S::Std);
            w.img(s, 0, t.gb_color_hdr.view);
            w.tex(s, 2, t.gb_albedo.view, ns);
            w.tex(s, 3, t.gb_normal[pp].view, ns);
            w.tex(s, 4, t.gb_pos[pp].view, ns);
            w.tex(s, 5, t.fog_tex[pp].view, ls);
            w.tex(s, 6, t.shadow_raw.view, ns);
            w.ubuf(s, 15, main);
            w.tex(s, 17, t.lamp_std.view, ns);
            let s = st(S::ShTrace);
            w.img(s, 0, t.shadow_raw.view);
            w.tex(s, 3, t.gb_normal[pp].view, ns);
            w.tex(s, 4, t.gb_pos[pp].view, ns);
            w.ubuf(s, 15, main);
            let s = st(S::Lamp);
            w.img(s, 0, t.lamp_std.view);
            w.tex(s, 3, t.gb_normal[pp].view, ns);
            w.tex(s, 4, t.gb_pos[pp].view, ns);
            w.ubuf(s, 15, main);
            let s = st(S::LampCube);
            w.img(s, 0, f.lamp_cube.view);
            w.ubuf(s, 15, main);
            let s = st(S::ShSoftH);
            w.img(s, 0, t.shadow_tmp.view);
            w.tex(s, 1, t.shadow_raw.view, ns);
            w.tex(s, 3, t.gb_normal[pp].view, ns);
            w.tex(s, 4, t.gb_pos[pp].view, ns);
            w.ubuf(s, 15, main);
            let s = st(S::ShSoftV);
            w.img(s, 0, t.shadow_raw.view);
            w.tex(s, 1, t.shadow_tmp.view, ns);
            w.tex(s, 3, t.gb_normal[pp].view, ns);
            w.tex(s, 4, t.gb_pos[pp].view, ns);
            w.ubuf(s, 15, main);
            w.flush(&ctx.device);
        }
        self.sets_dirty = false;
        self.sets_gi_stab = cvars::R_GI_STAB.get();
    }

    /// Per frame, after set_frame_uniforms: reallocations the settings asked for, pipeline
    /// rebuilds when the shader defines changed, the bodies into this slot, set re-baking.
    /// Per frame, before update_sets: ray queries switched on / off (or their boxes' setting
    /// changed) build / drop the acceleration structures; the shader defines follow.
    pub fn update_rt(&mut self, ctx: &Context, world: &World) {
        let want = rt_wanted(ctx);
        self.gpu.rt_wanted = want;
        if want != self.gpu.rt.active() || (want && self.gpu.rt.stale_boxes()) {
            ctx.wait_idle();
            self.gpu.retire_syncs(ctx, true);
            self.gpu.rt.destroy(ctx);
            self.gpu.rt_rebuild(ctx, world);
            self.sets_dirty = true;
        }
        RT_ACTIVE.store(self.gpu.rt.active(), std::sync::atomic::Ordering::Relaxed);
    }

    pub fn update_sets(&mut self, ctx: &Context, pp: usize) {
        if self.std.sh_n != self.std.sh_n_alloc {
            // r.std.shadows changed the shadow map resolution
            ctx.wait_idle();
            self.fx.shadow_map.destroy(ctx);
            self.fx.shadow_aux.destroy(ctx);
            self.std.sh_n_alloc = self.std.sh_n;
            let n = self.std.sh_n_alloc as u32;
            self.fx.shadow_map = ctx.image_2d("shadowMap", vk::Format::R32_SFLOAT, n, 4 * n);
            self.fx.shadow_aux = ctx.image_2d("shadowAux", vk::Format::R16_SFLOAT, n, 4 * n);
            let cb = ctx.begin_one_shot();
            to_general(&ctx.device, cb, &[self.fx.shadow_map.image, self.fx.shadow_aux.image]);
            ctx.end_one_shot_and_wait(cb);
            self.write_fixed_sets(ctx);
            self.std.valid = false;
            self.sets_dirty = true;
        }
        if self.frox_dims[2] != fog_slices() {
            self.realloc_froxels(ctx);
        }
        if gi_cache_wanted() && self.gi_cache_buf.is_null() {
            ctx.wait_idle();
            let u = vk::BufferUsageFlags::STORAGE_BUFFER
                | vk::BufferUsageFlags::TRANSFER_DST
                | vk::BufferUsageFlags::TRANSFER_SRC
                | vk::BufferUsageFlags::INDIRECT_BUFFER;
            self.gi_cache_buf = ctx.create_buffer("giCache", GI_CACHE_SIZE as u64 * 64, u, MemoryLocation::GpuOnly);
            self.gi_queue_buf = ctx.create_buffer("giCacheQueue", 48 + 2 * GI_CACHE_QUEUE_MAX as u64 * 32, u, MemoryLocation::GpuOnly);
            self.gi_cache_clear = true;
            self.sets_dirty = true;
        }
        if shader_defines() != self.compiled_defines {
            let _ = self.reload_shaders(ctx);
        }
        if self.gpu.flush_slot(ctx, pp) {
            self.sets_dirty = true;
        }
        if cvars::R_GI_STAB.get() != self.sets_gi_stab {
            self.sets_dirty = true;
        }
        if self.sets_dirty {
            self.write_sets(ctx);
        }
        self.cur = pp;
    }

    /// Wholesale scene replacement (a loaded / new world): re-upload, re-point set 0.
    pub fn reload_scene(&mut self, ctx: &Context, world: &mut World, objects: &mut [VoxObject]) {
        ctx.wait_idle();
        self.gpu.destroy(ctx);
        self.gpu.upload(ctx, self.pcache.cache, world, objects);
        self.lights = LightRegistry::default();
        self.lights.scan_world(world);
        self.gpu.synced_grid.clear();
        self.write_fixed_sets(ctx);
        self.scene_cpu.num_objects = self.gpu.num_objects as i32;
        self.scene_cpu.atlas_bricks[2] = self.gpu.atlas_bricks_z as u32;
        self.scene_cpu.world_bricks = [self.gpu.wbx, self.gpu.wby, self.gpu.wbz];
        self.sets_dirty = true;
        self.invalidate_standard(ctx);
        let cb = ctx.begin_one_shot();
        let dev = ctx.device.clone();
        self.record_history_clear(&dev, cb);
        ctx.end_one_shot_and_wait(cb);
    }

    /// The body buffers were recreated (init_objects): re-bake the sets next frame.
    pub fn rebind_object_buffers(&mut self) {
        self.sets_dirty = true;
        self.scene_cpu.num_objects = self.gpu.num_objects as i32;
    }

    pub fn standard_mode(&self) -> bool {
        cvars::R_LIGHTING.is("standard")
    }

    pub fn destroy(&mut self, ctx: &Context) {
        ctx.wait_idle();
        let dev = &ctx.device;
        for p in &mut self.pipes {
            p.destroy(dev);
        }
        unsafe {
            for sl in &mut self.slots {
                dev.destroy_query_pool(sl.ts, None);
                for b in [&mut sl.scene_u, &mut sl.main_u, &mut sl.lock_u, &mut sl.probe_list_u, &mut sl.lights_u] {
                    b.destroy(ctx);
                }
            }
            dev.destroy_descriptor_pool(self.pool, None);
            for l in [self.l0, self.l1, self.l2].iter().chain(self.l3.iter()) {
                dev.destroy_descriptor_set_layout(*l, None);
            }
            dev.destroy_sampler(self.near_s, None);
            dev.destroy_sampler(self.lin_s, None);
        }
        for b in self.atrous_ubo.iter_mut().chain(self.shape_stage.iter_mut()) {
            b.destroy(ctx);
        }
        self.gi_cache_buf.destroy(ctx);
        self.gi_queue_buf.destroy(ctx);
        self.destroy_targets(ctx);
        let f = &mut self.fx;
        for im in [
            &mut f.shadow_map, &mut f.shadow_aux, &mut f.probe_irr, &mut f.probe_depth, &mut f.probe_state, &mut f.probe_rays,
            &mut f.probe_data, &mut f.wind_disp, &mut f.wind_vel, &mut f.wind_trample, &mut f.lamp_cube, &mut f.horizon,
            &mut f.hor_blk, &mut f.cloud_pano, &mut f.shape_vol,
        ] {
            im.destroy(ctx);
        }
        self.gpu.destroy(ctx);
        self.pcache.destroy(ctx);
    }

    /// The image the caller presents / reads back.
    pub fn out_image(&self) -> &Image {
        &self.t.out_tex
    }

    /// Initial barriers so a resized output can be blitted (out_tex stays GENERAL).
    pub fn out_to_transfer_src(&self, dev: &ash::Device, cb: vk::CommandBuffer) {
        image_barrier(dev, cb, self.t.out_tex.image, Stage::COMPUTE_SHADER, Access::SHADER_STORAGE_WRITE, Stage::BLIT | Stage::COPY,
                      Access::TRANSFER_READ, vk::ImageLayout::GENERAL, vk::ImageLayout::TRANSFER_SRC_OPTIMAL);
    }

    pub fn out_to_general(&self, dev: &ash::Device, cb: vk::CommandBuffer) {
        image_barrier(dev, cb, self.t.out_tex.image, Stage::BLIT | Stage::COPY, Access::TRANSFER_READ, Stage::COMPUTE_SHADER,
                      Access::SHADER_STORAGE_WRITE, vk::ImageLayout::TRANSFER_SRC_OPTIMAL, vk::ImageLayout::GENERAL);
    }
}

impl SceneUbo {
    fn zeroed_ubo() -> Self {
        bytemuck::Zeroable::zeroed()
    }
}

/// Collects descriptor writes, then applies them in one `vkUpdateDescriptorSets`.
#[derive(Default)]
pub struct SetWriter {
    ws: Vec<(vk::DescriptorSet, u32, vk::DescriptorType, vk::DescriptorImageInfo, vk::DescriptorBufferInfo)>,
    accels: Vec<(vk::DescriptorSet, u32, vk::AccelerationStructureKHR)>,
}

impl SetWriter {
    pub fn img(&mut self, set: vk::DescriptorSet, b: u32, view: vk::ImageView) {
        let ii = vk::DescriptorImageInfo { sampler: vk::Sampler::null(), image_view: view, image_layout: vk::ImageLayout::GENERAL };
        self.ws.push((set, b, vk::DescriptorType::STORAGE_IMAGE, ii, Default::default()));
    }
    pub fn tex(&mut self, set: vk::DescriptorSet, b: u32, view: vk::ImageView, s: vk::Sampler) {
        self.tex_l(set, b, view, s, vk::ImageLayout::GENERAL);
    }
    pub fn tex_l(&mut self, set: vk::DescriptorSet, b: u32, view: vk::ImageView, s: vk::Sampler, l: vk::ImageLayout) {
        let ii = vk::DescriptorImageInfo { sampler: s, image_view: view, image_layout: l };
        self.ws.push((set, b, vk::DescriptorType::COMBINED_IMAGE_SAMPLER, ii, Default::default()));
    }
    pub fn accel(&mut self, set: vk::DescriptorSet, b: u32, a: vk::AccelerationStructureKHR) {
        self.accels.push((set, b, a));
    }
    pub fn ubuf(&mut self, set: vk::DescriptorSet, b: u32, buf: vk::Buffer) {
        let bi = vk::DescriptorBufferInfo { buffer: buf, offset: 0, range: vk::WHOLE_SIZE };
        self.ws.push((set, b, vk::DescriptorType::UNIFORM_BUFFER, Default::default(), bi));
    }
    pub fn sbuf(&mut self, set: vk::DescriptorSet, b: u32, buf: vk::Buffer) {
        let bi = vk::DescriptorBufferInfo { buffer: buf, offset: 0, range: vk::WHOLE_SIZE };
        self.ws.push((set, b, vk::DescriptorType::STORAGE_BUFFER, Default::default(), bi));
    }
    pub fn flush(self, dev: &ash::Device) {
        let mut writes: Vec<vk::WriteDescriptorSet> = self
            .ws
            .iter()
            .map(|(set, b, ty, ii, bi)| {
                let w = vk::WriteDescriptorSet::default().dst_set(*set).dst_binding(*b).descriptor_type(*ty);
                match *ty {
                    vk::DescriptorType::UNIFORM_BUFFER | vk::DescriptorType::STORAGE_BUFFER => w.buffer_info(std::slice::from_ref(bi)),
                    _ => w.image_info(std::slice::from_ref(ii)),
                }
            })
            .collect();
        let mut aw: Vec<vk::WriteDescriptorSetAccelerationStructureKHR> = self
            .accels
            .iter()
            .map(|(_, _, a)| vk::WriteDescriptorSetAccelerationStructureKHR::default().acceleration_structures(std::slice::from_ref(a)))
            .collect();
        for ((set, b, _), a) in self.accels.iter().zip(aw.iter_mut()) {
            let mut w = vk::WriteDescriptorSet::default().dst_set(*set).dst_binding(*b).descriptor_type(vk::DescriptorType::ACCELERATION_STRUCTURE_KHR);
            w.descriptor_count = 1;
            writes.push(w.push_next(a));
        }
        unsafe { dev.update_descriptor_sets(&writes, &[]) };
    }
}
