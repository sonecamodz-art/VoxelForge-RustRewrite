//! The compute pass table: every pipeline's shader parts, push-constant size and set-3
//! layout, and every descriptor set's layout. One table drives the initial build and the
//! runtime `shader reload`.

use ash::vk::DescriptorType as DT;

const SI: DT = DT::STORAGE_IMAGE;
const CIS: DT = DT::COMBINED_IMAGE_SAMPLER;
const UB: DT = DT::UNIFORM_BUFFER;

/// Set-3 layouts.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
#[repr(usize)]
pub enum L3 {
    Gbuf,
    Light,
    Gi,
    Atrous,
    Fog,
    Denoise,
    Raw,
    RawGi,
    RawLamp,
    Lock,
    Splat,
    Shadow,
    ProbeT,
    ProbeU,
    Std,
    ShTrace,
    ShSoft,
    GiStab,
    Wind,
    WaterRefl,
    WaterReflHalf,
    Clouds,
    CloudRes,
    CloudPano,
    Torch,
    Brush,
    Dust,
    FroxInj,
    FroxInt,
    FroxTile,
    Lamp,
    LampCube,
}
pub const L3_COUNT: usize = L3::LampCube as usize + 1;

pub fn l3_bindings(l: L3) -> Vec<(u32, DT)> {
    let v: &[(u32, DT)] = match l {
        // 0 albedo, 1 normal, 2 pos, 3 motion, 4 linDepth, 5 normalRough, 6 specAlbedo, 7 dbgHeat,
        // 8 depthHW, 9 delta, 10-13 locked view, 14 water, 16 surface, 17 diffuse guide
        L3::Gbuf => &[
            (0, SI), (1, SI), (2, SI), (3, SI), (4, SI), (5, SI), (6, SI), (7, SI), (8, SI), (9, SI), (10, SI), (11, CIS),
            (12, CIS), (13, CIS), (14, SI), (15, UB), (16, SI), (17, SI),
        ],
        L3::Light => &[(1, SI), (2, CIS), (3, CIS), (4, CIS), (5, CIS), (6, CIS), (7, CIS), (8, SI), (9, CIS), (10, CIS), (15, UB)],
        L3::Gi => &[(0, SI), (1, SI), (3, CIS), (4, CIS), (5, CIS), (7, CIS), (10, CIS), (11, CIS), (12, CIS), (13, CIS), (14, SI), (15, UB)],
        L3::Atrous => &[(0, SI), (2, CIS), (3, CIS), (4, CIS), (5, CIS), (6, CIS), (7, CIS), (15, UB)],
        L3::Fog => &[(0, SI), (4, CIS), (5, CIS), (6, CIS), (15, UB)],
        L3::Denoise => &[
            (0, SI), (1, SI), (2, CIS), (3, CIS), (4, CIS), (5, CIS), (6, CIS), (7, CIS), (8, CIS), (9, CIS), (10, CIS), (11, CIS),
            (12, CIS), (13, CIS), (14, CIS), (15, UB), (16, CIS), (17, CIS), (20, CIS),
        ],
        L3::Raw => &[
            (0, SI), (2, CIS), (3, CIS), (4, CIS), (5, CIS), (6, CIS), (7, SI), (12, CIS), (13, CIS), (14, CIS), (15, UB), (16, CIS),
            (17, CIS), (20, CIS),
        ],
        L3::RawLamp => &[(0, SI), (1, SI), (3, CIS), (4, CIS), (15, UB)],
        L3::RawGi => &[(0, SI), (3, CIS), (4, CIS), (15, UB)],
        L3::Lock => &[(0, SI), (1, CIS), (2, CIS), (14, UB), (15, UB)],
        L3::Splat => &[(0, SI), (1, SI), (2, CIS), (3, CIS), (14, UB), (15, UB)],
        L3::Shadow => &[(0, SI), (1, SI), (15, UB)],
        L3::ProbeT => &[(0, SI), (5, UB), (15, UB)],
        L3::ProbeU => &[(0, SI), (1, SI), (2, SI), (3, CIS), (4, SI), (5, UB), (15, UB)],
        L3::Std => &[
            (0, SI), (2, CIS), (3, CIS), (4, CIS), (5, CIS), (6, CIS), (12, CIS), (13, CIS), (14, CIS), (15, UB), (16, CIS), (17, CIS),
            (18, CIS), (20, CIS),
        ],
        L3::WaterRefl => &[(0, SI), (1, CIS), (2, CIS), (3, CIS), (4, CIS), (5, CIS), (6, CIS), (15, UB)],
        L3::WaterReflHalf => &[(0, SI), (1, CIS), (4, CIS), (15, UB)],
        L3::Clouds => &[(0, SI), (4, CIS), (15, UB)],
        L3::CloudRes => &[(0, SI), (1, CIS), (2, CIS), (15, UB)],
        L3::CloudPano => &[(0, SI), (15, UB)],
        L3::Torch => &[(0, SI), (3, CIS), (4, CIS), (15, UB)],
        L3::Brush => &[(0, SI), (3, CIS), (4, CIS), (5, CIS), (15, UB)],
        L3::Dust => &[(0, SI), (4, CIS), (15, UB)],
        L3::FroxInj => &[(0, SI), (1, CIS), (2, CIS), (15, UB)],
        L3::FroxTile => &[(0, SI), (4, CIS), (15, UB)],
        L3::FroxInt => &[(0, SI), (1, CIS), (15, UB)],
        L3::ShTrace => &[(0, SI), (3, CIS), (4, CIS), (15, UB)],
        L3::Lamp => &[(0, SI), (3, CIS), (4, CIS), (15, UB)],
        L3::LampCube => &[(0, SI), (15, UB)],
        L3::ShSoft => &[(0, SI), (1, CIS), (3, CIS), (4, CIS), (15, UB)],
        L3::Wind => &[(0, SI), (1, SI), (2, SI)],
        L3::GiStab => &[(0, SI), (1, CIS), (2, CIS), (3, CIS), (4, CIS), (5, CIS), (6, CIS), (7, CIS), (8, CIS), (9, CIS), (15, UB)],
    };
    v.to_vec()
}

/// Compute pipelines.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
#[repr(usize)]
pub enum P {
    Gbuf,
    TorchFx,
    Light,
    Gi,
    Atrous,
    Fog,
    Denoise,
    Raw,
    RawGi,
    RawLamp,
    Lock,
    Splat,
    Shadow,
    ProbeT,
    ProbeU,
    Std,
    ShTrace,
    Lamp,
    LampCube,
    ShSoft,
    GiStab,
    Wind,
    WaterRefl,
    WaterReflRR,
    WaterReflHalf,
    Clouds,
    CloudRes,
    CloudPano,
    Brush,
    Dust,
    FroxInj,
    FroxInt,
    FroxTile,
    GiCacheRes,
    GiCacheUpd,
}
pub const P_COUNT: usize = P::GiCacheUpd as usize + 1;

pub struct PassDef {
    pub file: &'static str,
    /// Parts between common.glsl and the pass.
    pub extra: &'static [&'static str],
    /// This pass's own #defines (before common.glsl: they reach its code too).
    pub defs: &'static str,
    pub push: u32,
    pub layout: L3,
}

pub fn pass_def(p: P) -> PassDef {
    let d = |file, extra, layout, push, defs| PassDef { file, extra, defs, push, layout };
    const GRASS: &[&str] = &["grass.glsl"];
    match p {
        P::Gbuf => d("gbuffer.comp", GRASS, L3::Gbuf, 0, ""),
        P::TorchFx => d("torchfx.comp", &["torchfx.glsl", "rain.glsl"], L3::Torch, 0, ""),
        P::Light => d("lighting.comp", GRASS, L3::Light, 0, ""),
        P::Gi => d("gi.comp", &[], L3::Gi, 0, ""),
        P::Atrous => d("atrous.comp", &[], L3::Atrous, 0, ""),
        P::Fog => d("fog.comp", &[], L3::Fog, 0, ""),
        P::Denoise => d("denoise.comp", &["tonemap.glsl", "torchfx.glsl", "rain.glsl"], L3::Denoise, 0, ""),
        P::Raw => d("raw_shade.comp", GRASS, L3::Raw, 0, "#define REFL_STOCHASTIC 1\n"),
        P::RawGi => d("raw_gi.comp", &[], L3::RawGi, 0, ""),
        P::RawLamp => d("raw_lamps.comp", &[], L3::RawLamp, 0, ""),
        P::Lock => d("lockview.comp", &[], L3::Lock, 0, ""),
        P::Splat => d("lock_splat.comp", &[], L3::Splat, 0, ""),
        P::Shadow => d("sunshadow.comp", &[], L3::Shadow, 24, ""),
        P::ProbeT => d("probe_trace.comp", &["probes.glsl"], L3::ProbeT, 0, ""),
        P::ProbeU => d("probe_update.comp", &["probes.glsl"], L3::ProbeU, 0, ""),
        P::Std => d("shade_std.comp", &["grass.glsl", "probes.glsl"], L3::Std, 0, ""),
        P::ShTrace => d("shadow_trace.comp", GRASS, L3::ShTrace, 0, ""),
        P::Lamp => d("lamp_std.comp", &[], L3::Lamp, 0, ""),
        P::LampCube => d("lamp_cube.comp", &[], L3::LampCube, 48, ""),
        P::ShSoft => d("shadow_soften.comp", &[], L3::ShSoft, 16, ""),
        P::GiStab => d("gi_stab.comp", &[], L3::GiStab, 16, ""),
        P::Wind => d("grass_wind.comp", &[], L3::Wind, 128, ""),
        P::WaterRefl => d("water_refl.comp", &["water_refl.glsl"], L3::WaterRefl, 0, ""),
        P::WaterReflRR => d("water_refl.comp", &["water_refl.glsl"], L3::WaterRefl, 0, "#define REFL_STOCHASTIC 1\n"),
        P::WaterReflHalf => d("water_refl_half.comp", &["water_refl.glsl"], L3::WaterReflHalf, 0, ""),
        P::Clouds => d("clouds.comp", &[], L3::Clouds, 0, ""),
        P::CloudRes => d("clouds_resolve.comp", &[], L3::CloudRes, 0, ""),
        P::CloudPano => d("clouds_pano.comp", &[], L3::CloudPano, 0, ""),
        P::Brush => d("brush_preview.comp", &[], L3::Brush, 128, ""),
        P::Dust => d("dust.comp", &[], L3::Dust, 16, ""),
        P::FroxInj => d("froxel_inject.comp", &[], L3::FroxInj, 0, ""),
        P::FroxInt => d("froxel_integrate.comp", &[], L3::FroxInt, 0, ""),
        P::FroxTile => d("froxel_tiles.comp", &[], L3::FroxTile, 0, ""),
        P::GiCacheRes => d("gi_cache_resolve.comp", &[], L3::Wind, 112, ""), // (set 3 unused)
        P::GiCacheUpd => d("gi_cache_update.comp", &[], L3::Wind, 0, ""),
    }
}

pub const ALL_PASSES: [P; P_COUNT] = [
    P::Gbuf, P::TorchFx, P::Light, P::Gi, P::Atrous, P::Fog, P::Denoise, P::Raw, P::RawGi, P::RawLamp, P::Lock, P::Splat,
    P::Shadow, P::ProbeT, P::ProbeU, P::Std, P::ShTrace, P::Lamp, P::LampCube, P::ShSoft, P::GiStab, P::Wind, P::WaterRefl,
    P::WaterReflRR, P::WaterReflHalf, P::Clouds, P::CloudRes, P::CloudPano, P::Brush, P::Dust, P::FroxInj, P::FroxInt,
    P::FroxTile, P::GiCacheRes, P::GiCacheUpd,
];

/// Per-slot descriptor sets (several sets share a layout: the a-trous levels, the two
/// soften directions, the three torch targets).
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
#[repr(usize)]
pub enum S {
    Gbuf,
    Light,
    Gi,
    Atrous0,
    Atrous1,
    Atrous2,
    Atrous3,
    Atrous4,
    Fog,
    Denoise,
    Raw,
    RawGi,
    RawLamp,
    Lock,
    Splat,
    Shadow,
    ProbeT,
    ProbeU,
    Std,
    ShTrace,
    ShSoftH,
    ShSoftV,
    GiStab,
    WaterRefl,
    WaterReflHalf,
    Clouds,
    CloudRes,
    CloudPano,
    Torch0,
    Torch1,
    Torch2,
    Brush,
    Dust,
    FroxInj,
    FroxInt,
    FroxTile,
    Lamp,
    LampCube,
}
pub const S_COUNT: usize = S::LampCube as usize + 1;

pub fn set_layout(s: S) -> L3 {
    match s {
        S::Gbuf => L3::Gbuf,
        S::Light => L3::Light,
        S::Gi => L3::Gi,
        S::Atrous0 | S::Atrous1 | S::Atrous2 | S::Atrous3 | S::Atrous4 => L3::Atrous,
        S::Fog => L3::Fog,
        S::Denoise => L3::Denoise,
        S::Raw => L3::Raw,
        S::RawGi => L3::RawGi,
        S::RawLamp => L3::RawLamp,
        S::Lock => L3::Lock,
        S::Splat => L3::Splat,
        S::Shadow => L3::Shadow,
        S::ProbeT => L3::ProbeT,
        S::ProbeU => L3::ProbeU,
        S::Std => L3::Std,
        S::ShTrace => L3::ShTrace,
        S::ShSoftH | S::ShSoftV => L3::ShSoft,
        S::GiStab => L3::GiStab,
        S::WaterRefl => L3::WaterRefl,
        S::WaterReflHalf => L3::WaterReflHalf,
        S::Clouds => L3::Clouds,
        S::CloudRes => L3::CloudRes,
        S::CloudPano => L3::CloudPano,
        S::Torch0 | S::Torch1 | S::Torch2 => L3::Torch,
        S::Brush => L3::Brush,
        S::Dust => L3::Dust,
        S::FroxInj => L3::FroxInj,
        S::FroxInt => L3::FroxInt,
        S::FroxTile => L3::FroxTile,
        S::Lamp => L3::Lamp,
        S::LampCube => L3::LampCube,
    }
}

pub const ALL_SETS: [S; S_COUNT] = [
    S::Gbuf, S::Light, S::Gi, S::Atrous0, S::Atrous1, S::Atrous2, S::Atrous3, S::Atrous4, S::Fog, S::Denoise, S::Raw, S::RawGi,
    S::RawLamp, S::Lock, S::Splat, S::Shadow, S::ProbeT, S::ProbeU, S::Std, S::ShTrace, S::ShSoftH, S::ShSoftV, S::GiStab,
    S::WaterRefl, S::WaterReflHalf, S::Clouds, S::CloudRes, S::CloudPano, S::Torch0, S::Torch1, S::Torch2, S::Brush, S::Dust,
    S::FroxInj, S::FroxInt, S::FroxTile, S::Lamp, S::LampCube,
];

pub const ATROUS_SETS: [S; 5] = [S::Atrous0, S::Atrous1, S::Atrous2, S::Atrous3, S::Atrous4];
pub const TORCH_SETS: [S; 3] = [S::Torch0, S::Torch1, S::Torch2];

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn tables_consistent() {
        for (i, p) in ALL_PASSES.iter().enumerate() {
            assert_eq!(*p as usize, i);
            let d = pass_def(*p);
            assert!(crate::gfx::shader::load_source(d.file).is_some(), "{}", d.file);
            for e in d.extra {
                assert!(crate::gfx::shader::load_source(e).is_some(), "{e}");
            }
        }
        for (i, s) in ALL_SETS.iter().enumerate() {
            assert_eq!(*s as usize, i);
        }
        for l in 0..L3_COUNT {
            let l: L3 = unsafe { std::mem::transmute(l) };
            let mut b: Vec<u32> = l3_bindings(l).iter().map(|x| x.0).collect();
            let n = b.len();
            b.sort();
            b.dedup();
            assert_eq!(n, b.len(), "{l:?} has a duplicate binding");
        }
    }
}
