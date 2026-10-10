//! Every console variable of the engine, in one table (defaults match the C++ engine).
//! Read them anywhere as `cvars::R_GRASS.get()` / `.is("high")`: lock-free.
#![allow(dead_code)]

use crate::cvar::{Bool, CVar, Float, Int, Str};

macro_rules! cvars {
    ($( $(#[$m:meta])* $id:ident : $ty:ident = $ctor:expr ;)*) => {
        $( $(#[$m])* pub static $id: $ty = $ctor; )*
        /// The registry: every cvar, for the console / settings / crash reports.
        pub static ALL: &[&dyn CVar] = &[$(&$id),*];
    };
}

const BIG: f32 = 1e30;
const IMAX: i32 = i32::MAX;

cvars! {
    // ---- application -------------------------------------------------------------------
    APP_TITLE: Bool = Bool::new("app.title", false, "start on the title screen (0 = straight into the world)");
    APP_FPSCAP: Int = Int::new("app.fpscap", 0, "frame-rate cap (0 = off)", 0, 1000);
    APP_FULLSCREEN: Bool = Bool::new("app.fullscreen", false, "fullscreen window (F11)");
    APP_FRAMES_IN_FLIGHT: Int = Int::new("app.framesinflight", 2, "frames the CPU may record ahead of the GPU", 1, 2);
    APP_TEST_DT: Float = Float::new("app.test.dt", 0.0, "scripted runs: a fixed frame time in seconds (0 = real time)", 0.0, 0.1);
    APP_SCRIPT: Str = Str::new("app.script", "", "console commands run when the window loop starts ('|' separates; `after N cmd` delays)", &[]);
    WORLD_GEN: Int = Int::new("world.gen", 1, "terrain generator for new worlds (1 = original hills)", 1, 1);
    WORLD_SEED: Int = Int::new("world.seed", 1337, "seed of the startup world", 0, IMAX);
    WORLD_ENDLESS: Bool = Bool::new("world.endless", true, "new worlds stream endlessly around you (0 = the fixed 2048 x 2048 world)");
    R_VIEW_DISTANCE: Int = Int::new("r.view.distance", 10, "endless worlds: chunks (12.8 m) kept loaded around you", 4, 14);
    STREAM_CACHE: Int = Int::new("stream.cache", 256, "endless worlds: MB of RAM keeping chunks you left (walking back needs no re-generation)", 0, 4096);
    STREAM_BUDGET: Float = Float::new("stream.budget", 2.0, "endless worlds: ms per frame spent adding finished chunks", 0.2, 20.0);
    STREAM_PER_FRAME: Int = Int::new("stream.perframe", 1, "endless worlds: chunks added (and uploaded) per frame; 3 while finished ones pile up", 1, 64);
    R_HORIZON: Bool = Bool::new("r.horizon", true, "endless worlds: the land beyond the loaded chunks, to ~3 km (far horizon)");
    R_HORIZON_FADE: Float = Float::new("r.horizon.fade", 0.5, "endless worlds: seconds a chunk streaming in at the edge fades in (0 = pop in)", 0.0, 3.0);
    R_EXPOSURE: Float = Float::new("r.exposure", 0.8, "manual exposure ([ and ] keys)", 0.05, 4.0);
    R_PRESENT: Str = Str::new("r.present", "fifo", "present mode: fifo = v-sync, mailbox = low latency, immediate = uncapped (V)",
                              &["fifo", "mailbox", "immediate"]);
    R_SCALE: Float = Float::new("r.scale", 1.0, "render scale (F6 cycles 1/0.75/0.6/0.5); the blit upscales", 0.25, 1.0);
    R_DENOISER: Str = Str::new("r.denoiser", "svgf", "svgf = in-house SVGF chain, none = raw 1-spp radiance (N cycles)",
                               &["svgf", "none"]);
    UI_OVERLAY: Bool = Bool::new("ui.overlay", false, "performance overlay in the window title (F3)");
    IN_SENSITIVITY: Float = Float::new("in.sensitivity", 1.0, "mouse sensitivity (1 = default)", 0.1, 4.0);
    IN_INVERT_Y: Bool = Bool::new("in.invertY", false, "invert the mouse's vertical axis");
    IN_FLYSPEED: Float = Float::new("in.flyspeed", 80.0, "flying speed, voxels per second (wheel changes it)", 5.0, 800.0);
    GAME_PHYSICS: Bool = Bool::new("game.physics", true, "physics running: loose pieces fall, collide and settle (P pauses it)");
    PHYS_FOLIAGE: Bool = Bool::new("phys.foliage", false, "falling bodies collide with the world's foliage and thin branches");
    IN_BOB: Bool = Bool::new("in.bob", true, "view bob while walking");
    IN_WALK: Bool = Bool::new("in.walk", true, "walk with collision and gravity (Tab toggles the free camera)");
    PLAYER_HEIGHT: Float = Float::new("player.height", 18.0, "player height, voxels (1.8 m)", 8.0, 40.0);
    PLAYER_EYE: Float = Float::new("player.eye", 16.2, "eye height, voxels (1.62 m)", 6.0, 38.0);
    PLAYER_WIDTH: Float = Float::new("player.width", 6.0, "player width, voxels (0.6 m)", 2.0, 16.0);
    PLAYER_GRAVITY: Float = Float::new("player.gravity", 200.0, "gravity on the player, voxels/s^2 (20 m/s^2: snappy)", 20.0, 1000.0);
    PLAYER_JUMP: Float = Float::new("player.jump", 12.0, "jump height, voxels (1.2 m)", 0.0, 60.0);
    PLAYER_WALK: Float = Float::new("player.walk", 21.5, "walking speed, voxels/s (2.15 m/s)", 5.0, 400.0);
    PLAYER_SPRINT: Float = Float::new("player.sprint", 50.0, "sprinting speed, voxels/s (Shift, 5 m/s)", 5.0, 400.0);
    PLAYER_STEP: Float = Float::new("player.step", 4.0, "highest ledge walked up without jumping, voxels", 0.0, 12.0);

    // ---- renderer: tracing ---------------------------------------------------------------
    R_DEBUGVIEW: Int = Int::new("r.debugview", 0, "0 normal, 1 object-id tint, 2 heat.primary, 3 heat.shadow, 4 gi only, \
        5 sun shadow, 6 history length, 7 albedo, 8 normals, 9 shadow cascades", 0, 9);
    R_PROFILE_SPLIT: Bool = Bool::new("r.profile.split", false, "serialize lighting/gi/fog with separate GPU timestamps (profiling)");
    R_TRACE_MASKS: Bool = Bool::new("r.trace.masks", false, "detailed-brick traversal via 4^3 occupancy bitmasks");
    R_TRACE_OBJBVH: Bool = Bool::new("r.trace.objbvh", true, "trace objects through a BVH over their AABBs");
    R_TRACE_DISTFIELD: Bool = Bool::new("r.trace.distfield", true, "empty-space skipping with the brick distance field");
    R_GI_FILTER: Str = Str::new("r.gi.filter", "normal", "GI denoise: normal = 3 a-trous levels, smooth = 5", &["normal", "smooth"]);
    R_SVGF_SHADOW: Str = Str::new("r.svgf.shadow", "stochastic", "SVGF sun shadow: soft (exact ray + penumbra filter) or stochastic",
                                  &["soft", "stochastic"]);
    R_SVGF_SHADOW_MAXR: Float = Float::new("r.svgf.shadow.maxr", 96.0, "r.svgf.shadow soft: widest penumbra (pixels)", 4.0, 256.0);
    R_GI_STAB: Bool = Bool::new("r.gi.stab", true, "temporally stabilise the filtered GI");
    R_GI_STAB_GAMMA: Float = Float::new("r.gi.stab.gamma", 1.25, "GI stabilisation clamp width (std-devs)", 0.25, 4.0);
    R_GI_STAB_FRAMES: Float = Float::new("r.gi.stab.frames", 16.0, "GI stabilisation history length (frames)", 2.0, 64.0);
    R_NOISE: Str = Str::new("r.noise", "lds", "GI ray directions: lds (R2 sequence) or white (PCG)", &["lds", "white"]);
    R_GI_BICUBIC: Bool = Bool::new("r.gi.bicubic", true, "GI history reprojection: Catmull-Rom instead of bilinear");
    R_GI_HISTFIX: Bool = Bool::new("r.gi.histfix", true, "freshly revealed texels borrow converged neighbours' GI");
    R_AO_HALFRES: Bool = Bool::new("r.ao.halfres", false, "contact AO applied in the half-res GI pass");
    R_GRASS: Str = Str::new("r.grass", "high", "grass blades: high, low, off", &["high", "low", "off"]);
    R_GRASS_SHADOWS: Bool = Bool::new("r.grass.shadows", true, "grass blades cast animated sun shadows");
    R_GRASS_DISTANCE: Float = Float::new("r.grass.distance", 90.0, "grass blade draw distance (voxels)", 10.0, 600.0);
    R_GRASS_FAR: Float = Float::new("r.grass.far", 2.5, "far grass tufts out to this x the grass distance (0 = off)", 0.0, 4.0);
    R_GRASS_STYLE: Str = Str::new("r.grass.style", "smooth", "grass motion: smooth or stepped", &["smooth", "stepped"]);
    R_GRASS_RECOVER: Float = Float::new("r.grass.recover", 40.0, "seconds for grass to straighten after wind / trampling", 2.0, 300.0);
    R_WATER: Bool = Bool::new("r.water", true, "transparent water (refraction, absorption, traced reflections)");
    R_WATER_CLARITY: Float = Float::new("r.water.clarity", 1.0, "water clarity", 0.2, 5.0);
    R_WATER_REFRACT: Float = Float::new("r.water.refract", 0.5, "how much waves bend the view into the water", 0.0, 1.0);
    R_WATER_MURK: Float = Float::new("r.water.murk", 1.0, "water murkiness", 0.0, 6.0);
    R_WATER_REFLECT_HALF: Bool = Bool::new("r.water.reflect.half", true, "water reflections traced at half resolution");
    R_WATER_WAVES: Float = Float::new("r.water.waves", 1.0, "size of the waves on lakes (0 = glassy)", 0.0, 3.0);
    R_WATER_CAUSTICS: Float = Float::new("r.water.caustics", 1.0, "caustics under the water surface", 0.0, 2.0);
    R_WATER_MAXDEPTH: Float = Float::new("r.water.maxdepth", 64.0, "deepest bed traced through water (voxels)", 8.0, 256.0);
    R_WATER_REFLECT: Bool = Bool::new("r.water.reflect", true, "water reflects the scene (one traced ray per water pixel)");
    R_WATER_REFLECT_DIST: Float = Float::new("r.water.reflect.dist", 600.0, "reflection ray reach (voxels)", 50.0, 4000.0);
    R_FACETS: Float = Float::new("r.facets", 1.0, "per-voxel facets (0 = flat)", 0.0, 3.0);
    R_REFLECT: Bool = Bool::new("r.reflect", true, "glossy surfaces reflect the scene");
    R_FOLIAGE_SSS: Float = Float::new("r.foliage.sss", 1.0, "light through leaves and grass", 0.0, 2.0);
    R_DUST: Float = Float::new("r.dust", 1.0, "sunlit dust motes near the camera (0 = off)", 0.0, 2.0);
    R_DUST_DENSITY: Float = Float::new("r.dust.density", 0.3, "share of 2-voxel cells holding a mote", 0.02, 1.0);
    R_FOLIAGE_LIGHT: Float = Float::new("r.foliage.light", 0.6, "sunlight passing through each leaf", 0.0, 0.9);
    R_WIND_STRENGTH: Float = Float::new("r.wind.strength", 1.0, "wind strength", 0.0, 3.0);
    R_WIND_GUST: Float = Float::new("r.wind.gust", 1.0, "gust strength", 0.0, 3.0);
    R_WIND_SPEED: Float = Float::new("r.wind.speed", 14.0, "gust travel speed (voxels / s)", 0.0, 80.0);
    R_WIND_DIR: Float = Float::new("r.wind.dir", 37.0, "mean wind direction (degrees)", 0.0, 360.0);
    R_WIND_SET: Float = Float::new("r.wind.set", 0.6, "how much of the wind's bend grass keeps", 0.0, 1.0);
    R_FRAME_OFFSET: Int = Int::new("r.frame.offset", 0, "debug: add this to the shader frame counter", 0, 2_000_000_000);

    // ---- look -----------------------------------------------------------------------------
    R_SUN_AZIMUTH: Float = Float::new("r.sun.azimuth", 32.0, "sun azimuth (degrees)", 0.0, 360.0);
    R_SUN_ELEVATION: Float = Float::new("r.sun.elevation", 46.0, "sun elevation (degrees); below 0 = night", -40.0, 89.0);
    R_TIME_HOUR: Float = Float::new("r.time.hour", 14.0, "time of day (hours): sets the sun when changed", 0.0, 24.0);
    R_TIME_DAYLENGTH: Float = Float::new("r.time.daylength", 0.0, "real minutes per day-night cycle (0 = still)", 0.0, 240.0);
    R_TIME_FREEZE: Bool = Bool::new("r.time.freeze", false, "freeze the water/grass/wind animation clock");
    R_SKY_MODEL: Str = Str::new("r.sky.model", "classic", "classic or physical sky", &["classic", "physical"]);
    R_SUN_INTENSITY: Float = Float::new("r.sun.intensity", 1.0, "direct sunlight strength", 0.0, 4.0);
    R_SUN_SOFTNESS: Float = Float::new("r.sun.softness", 1.0, "sun disk size (penumbra width)", 0.1, 4.0);
    R_SKY_INTENSITY: Float = Float::new("r.sky.intensity", 1.0, "sky brightness + sky light", 0.0, 3.0);
    R_GI_INTENSITY: Float = Float::new("r.gi.intensity", 1.0, "bounce light strength", 0.0, 3.0);
    R_FOG_DENSITY: Float = Float::new("r.fog.density", 0.3, "volumetric fog density (x0.0012 / voxel)", 0.0, 12.0);
    R_FOG_HEIGHT: Float = Float::new("r.fog.height", 250.0, "height fog altitude (voxels)", 0.0, 1000.0);
    R_FOG_FALLOFF: Float = Float::new("r.fog.falloff", 0.0, "height fog falloff (0 = uniform)", 0.0, 400.0);
    R_FOG_NOISE: Float = Float::new("r.fog.noise", 0.0, "drifting fog wisps", 0.0, 1.0);
    R_FOG_AMBIENT: Float = Float::new("r.fog.ambient", 1.0, "fog brightness from sky light", 0.0, 4.0);
    R_FOG_METHOD: Str = Str::new("r.fog.method", "froxel", "volumetric fog: froxel or raymarch", &["froxel", "raymarch"]);
    R_FOG_RANGE: Float = Float::new("r.fog.range", 600.0, "froxel fog volume distance (voxels)", 100.0, 1400.0);
    R_CLOUDS: Bool = Bool::new("r.clouds", true, "clouds in the sky");
    R_CLOUDS_TEMPORAL: Bool = Bool::new("r.clouds.temporal", true, "clouds accumulated over frames");
    R_CLOUDS_COVERAGE: Float = Float::new("r.clouds.coverage", 0.3, "sky cloud cover (0 clear .. 1 overcast)", 0.0, 1.0);
    R_CLOUDS_BASE: Float = Float::new("r.clouds.base", 120.0, "cloud base above the sea (metres)", 30.0, 600.0);
    R_CLOUDS_THICKNESS: Float = Float::new("r.clouds.thickness", 45.0, "tallest clouds (metres)", 5.0, 300.0);
    R_CLOUDS_DENSITY: Float = Float::new("r.clouds.density", 1.0, "cloud voxel opacity", 0.1, 4.0);
    R_CLOUDS_SPEED: Float = Float::new("r.clouds.speed", 2.5, "cloud drift speed (m/s)", 0.0, 60.0);
    R_RAIN: Float = Float::new("r.rain", 0.0, "rain (0 none .. 1 heavy)", 0.0, 1.0);
    R_RAIN_PAUSE: Bool = Bool::new("r.rain.pause", false, "dev: the drops hang still");
    R_HAZE_DISTANCE: Float = Float::new("r.haze.distance", 12000.0, "aerial haze visibility (metres; 0 = none)", 0.0, 50000.0);
    R_FOG_START: Float = Float::new("r.fog.start", 1200.0, "fog starts this far from the eye (voxels)", 0.0, 3000.0);
    R_FOG_QUALITY: Str = Str::new("r.fog.quality", "medium", "fog march quality", &["low", "medium", "high"]);
    R_GODRAYS_STRENGTH: Float = Float::new("r.godrays.strength", 1.0, "god-ray strength", 0.0, 6.0);
    R_GODRAYS_FOCUS: Float = Float::new("r.godrays.focus", 0.74, "god-ray anisotropy", 0.0, 0.95);
    R_TONEMAP: Str = Str::new("r.tonemap", "aces", "tonemapper", &["aces", "agx"]);
    R_SATURATION: Float = Float::new("r.saturation", 1.0, "colour saturation", 0.0, 2.0);
    R_WARMTH: Float = Float::new("r.warmth", 0.0, "white balance (+ warm, - cool)", -1.0, 1.0);
    R_CONTRAST: Float = Float::new("r.contrast", 1.0, "contrast", 0.5, 1.8);

    // ---- lighting modes --------------------------------------------------------------------
    R_LIGHTING: Str = Str::new("r.lighting", "raytraced", "raytraced = traced sun + GI rays (SVGF); standard = shadow map + probe GI",
                               &["raytraced", "standard"]);
    R_STD_RETRACEALL: Bool = Bool::new("r.std.retraceall", false, "debug: re-trace every shadow cascade every frame");
    R_STD_VIEWBIAS: Float = Float::new("r.std.viewbias", 0.8, "probe sampling view-vector bias share", 0.0, 1.0);
    R_STD_PROBEFREEZE: Int = Int::new("r.std.probefreeze", 0, "debug: freeze the probe grid from this frame on", 0, 1_000_000_000);
    R_STD_CASCADEBIAS: Int = Int::new("r.std.cascadebias", 0, "debug: shade from this many cascades coarser", 0, 3);
    R_STD_SHADOWS: Str = Str::new("r.std.shadows", "high", "sun shadow quality", &["low", "medium", "high", "ultra"]);
    R_STD_SHADOWDIST: Float = Float::new("r.std.shadowdist", 1.0, "shadow distance scale", 0.5, 3.0);
    R_STD_CASCADEBLEND: Bool = Bool::new("r.std.cascadeblend", true, "blend between shadow cascades");
    R_STD_HYBRIDSHADOWS: Bool = Bool::new("r.std.hybridshadows", true, "sun shadows read the cached map where certain");
    R_LIGHTS_CACHE: Bool = Bool::new("r.lights.cache", true, "lamp shadows from cached cube maps (Standard)");
    R_STD_SOFTSHADOWS: Bool = Bool::new("r.std.softshadows", true, "soft shadows (Standard)");
    R_STD_GISPEED: Str = Str::new("r.std.gispeed", "normal", "how fast bounce light reacts", &["slow", "normal", "fast"]);
    R_FOV: Float = Float::new("r.fov", 70.0, "vertical field of view (degrees)", 50.0, 110.0);
    R_GI_OBJECTS: Bool = Bool::new("r.gi.objects", true, "bodies block and bounce indirect light");
    R_LIGHTS: Bool = Bool::new("r.lights", true, "emissive voxels are sampled as light sources");
    R_LIGHTS_INTENSITY: Float = Float::new("r.lights.intensity", 4.0, "brightness of emissive light sources", -BIG, BIG);
    R_LIGHTS_SHADOWS: Str = Str::new("r.lights.shadows", "hard", "Standard lamp shadows", &["hard", "soft"]);
    R_LIGHTS_QUALITY: Str = Str::new("r.lights.quality", "full", "Standard lamp shadow resolution", &["full", "half"]);
    R_GI_SKYOCC: Bool = Bool::new("r.gi.skyocc", true, "sky light only where the sky is really open");
    R_GI_REACH: Int = Int::new("r.gi.reach", 48, "GI ray reach (voxels)", 32, 1024);
    R_GI_BOUNCES: Int = Int::new("r.gi.bounces", 1, "bounce light: 1 = single, 2 = double", 1, 2);
    R_GI_CACHE: Bool = Bool::new("r.gi.cache", true, "double bounce from a radiance cache");
    R_GI_CACHE_MIN: Int = Int::new("r.gi.cache.min", 4, "radiance cache: samples before a patch is used", 1, 64);
    R_GI_CACHE_CELL: Int = Int::new("r.gi.cache.cell", 0, "radiance cache: log2 of the smallest patch", 0, 4);
    R_GI_CACHE_LOD: Float = Float::new("r.gi.cache.lod", 0.02, "radiance cache: patch size x camera distance", 0.001, 1.0);
    R_GI_CACHE_FRAMES: Int = Int::new("r.gi.cache.frames", 32, "radiance cache: max samples a patch averages", 2, 1024);
    R_GI_CACHE_BUDGET: Int = Int::new("r.gi.cache.budget", 65536, "radiance cache: update rays per frame", 1024, 131_072);
    R_GI_CACHE_EVICT: Int = Int::new("r.gi.cache.evict", 240, "radiance cache: frames before an unused patch drops", 8, 100_000);
    R_GI_CACHE_EDITREACH: Int = Int::new("r.gi.cache.editreach", 128, "radiance cache: edit reset reach (voxels)", 0, 1024);

    // ---- post ------------------------------------------------------------------------------
    R_BLOOM: Bool = Bool::new("r.bloom", false, "bloom");
    R_BLOOM_STRENGTH: Float = Float::new("r.bloom.strength", 0.05, "bloom amount", 0.0, 0.4);
    R_AUTOEXPOSURE: Bool = Bool::new("r.autoexposure", true, "eye adaptation");
    R_UNDERWATER: Float = Float::new("r.underwater", 1.0, "under-water screen effects", 0.0, 2.0);
    R_AUTOEXPOSURE_SPEED: Float = Float::new("r.autoexposure.speed", 1.5, "eye adaptation speed", 0.1, 10.0);
    R_AUTOEXPOSURE_MIN: Float = Float::new("r.autoexposure.min", 1.0, "least auto exposure scale", 0.2, 1.0);
    R_AUTOEXPOSURE_MAX: Float = Float::new("r.autoexposure.max", 8.0, "most auto exposure brightening", 1.0, 8.0);
    R_AUTOEXPOSURE_KEY: Float = Float::new("r.autoexposure.key", 0.18, "auto exposure target luminance", 0.02, 4.0);

    // ---- shader compilation (new in the Rust port) -------------------------------------------
    R_SHADER_OPT: Bool = Bool::new("r.shader.opt", true, "compile shaders with optimisation (cached on disk after the first run)");
    R_SHADER_CACHE: Bool = Bool::new("r.shader.cache", true, "keep compiled SPIR-V in a disk cache (shader_cache/)");
}

#[cfg(test)]
mod tests {
    #[test]
    fn names_unique() {
        let mut names: Vec<_> = super::ALL.iter().map(|c| c.name()).collect();
        names.sort();
        let n = names.len();
        names.dedup();
        assert_eq!(n, names.len(), "duplicate cvar name");
    }
}
