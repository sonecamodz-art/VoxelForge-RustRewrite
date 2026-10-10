//! The static world: one big sparse volume plus the 256-entry material palette.

use std::ops::{Deref, DerefMut};
use std::sync::atomic::AtomicI32;

use bytemuck::{Pod, Zeroable};

use super::sparse_volume::SparseVolume;

/// World dims in voxels (1 voxel = 10 cm) of a Normal-size world.
pub const WORLD_NX: i32 = 2048;
pub const WORLD_NY: i32 = 1024;
pub const WORLD_NZ: i32 = 2048;
pub const WORLD_LARGE_NXZ: i32 = 4096;

/// A palette entry, mirrored in GLSL (common.glsl `Material`: 3 x vec4).
///  * rgb: albedo (a metal's reflectance F0), emissive
///  * props: density, per-voxel shade jitter, kind (1 water, 2 grass, 3 foliage),
///    grass variant / light animation (emissive)
///  * surf: roughness, metalness, dielectric F0, detail pattern
#[repr(C)]
#[derive(Clone, Copy, Debug, PartialEq, Pod, Zeroable)]
pub struct Material {
    pub r: f32,
    pub g: f32,
    pub b: f32,
    pub emissive: f32,
    pub props: [f32; 4],
    pub surf: [f32; 4],
}
const _: () = assert!(std::mem::size_of::<Material>() == 48);

impl Default for Material {
    fn default() -> Self {
        Self { r: 0.0, g: 0.0, b: 0.0, emissive: 0.0, props: [0.0; 4], surf: [0.9, 0.0, 0.04, 0.0] }
    }
}

/// Palette indices of the generated world's materials.
pub mod wm {
    pub const AIR: u8 = 0;
    pub const GRASS: u8 = 1;
    pub const DIRT: u8 = 2;
    pub const STONE: u8 = 3;
    pub const SAND: u8 = 4;
    pub const WATER: u8 = 5;
    pub const WOOD: u8 = 6;
    pub const LEAVES: u8 = 7;
    pub const LAMP: u8 = 8;
    pub const BRICKM: u8 = 9;
    pub const GRAVEL: u8 = 10;
    pub const COARSE_DIRT: u8 = 11;
    pub const DRY_GRASS: u8 = 12;
    pub const MOSS: u8 = 13;
    pub const NEEDLES: u8 = 14;
    pub const BIRCH_BARK: u8 = 15;
    pub const SPHERE0: u8 = 16;
    pub const FLOWERS: u8 = 24;
    pub const PINE_BARK: u8 = 25;
    pub const BIRCH_LEAVES: u8 = 26;
    pub const DEAD_WOOD: u8 = 27;
    pub const BUSH_LEAVES: u8 = 28;
    pub const BERRIES: u8 = 29;
    pub const TORCH: u8 = 30;
    pub const TALL_GRASS: u8 = 31;
    pub const TERRAIN2_BASE: u8 = 224;
    pub const SNOW: u8 = 224;
    pub const MUD: u8 = 225;
    pub const CLAY: u8 = 226;
    pub const RED_SAND: u8 = 227;
    pub const SANDSTONE: u8 = 228;
    pub const RED_SANDSTONE: u8 = 229;
    pub const PALE_SANDSTONE: u8 = 230;
    pub const SCREE: u8 = 231;
    pub const REEDS: u8 = 232;
    pub const MARSH_GRASS: u8 = 233;
    pub const LIMESTONE: u8 = 234;
    pub const SHALE: u8 = 235;
    pub const GRANITE: u8 = 236;
    pub const PRESET_BASE: u8 = 240;
    pub const IRON: u8 = 240;
    pub const STEEL: u8 = 241;
    pub const COPPER: u8 = 242;
    pub const GOLD: u8 = 243;
    pub const BRASS: u8 = 244;
    pub const ALUMINIUM: u8 = 245;
    pub const RUSTY_IRON: u8 = 246;
    pub const TORCH_STICK: u8 = 248;
    pub const TORCH_HEAD: u8 = 249;
    pub const TORCH_BRACKET_N: u8 = 250;
    pub const TORCH_BRACKET_E: u8 = 251;
    pub const TORCH_BRACKET_S: u8 = 252;
    pub const TORCH_BRACKET_W: u8 = 253;
    pub const TORCH_FLAME: u8 = 254;

    // Surface detail patterns (Material::surf[3]; gbuffer.comp).
    pub const PAT_NONE: f32 = 0.0;
    pub const PAT_SOIL: f32 = 1.0;
    pub const PAT_GRASS: f32 = 2.0;
    pub const PAT_STONE: f32 = 3.0;
    pub const PAT_SAND: f32 = 4.0;
    pub const PAT_GRAVEL: f32 = 5.0;
    pub const PAT_WOOD: f32 = 6.0;
    pub const PAT_BARK: f32 = 7.0;
    pub const PAT_BRICK: f32 = 8.0;
    pub const PAT_LEAVES: f32 = 9.0;
    pub const PAT_METAL: f32 = 10.0;
    pub const PAT_RUST: f32 = 11.0;

    pub fn is_torch_part(m: u8) -> bool {
        (TORCH_STICK..=TORCH_FLAME).contains(&m)
    }
    pub fn is_torch_solid(m: u8) -> bool {
        (TORCH_STICK..=TORCH_BRACKET_W).contains(&m)
    }
    /// Trees, bushes, logs (not ground).
    pub fn is_plant(m: u8) -> bool {
        matches!(m, WOOD | LEAVES | NEEDLES | BIRCH_BARK | PINE_BARK | BIRCH_LEAVES | DEAD_WOOD | BUSH_LEAVES | BERRIES)
    }
    /// Foliage: held by what it grows on, never holds anything up.
    pub fn is_foliage_voxel(m: u8) -> bool {
        matches!(m, LEAVES | NEEDLES | BIRCH_LEAVES | BUSH_LEAVES | BERRIES | FLOWERS | TALL_GRASS)
    }
}
use wm::*;

/// Physical density per material in t/m^3 (one 10 cm voxel weighs that many kg).
pub fn material_density(m: u8) -> f32 {
    match m {
        3 => 2.6,
        9 => 1.8,
        2 | 11 => 1.5,
        1 | 12 | 13 => 1.4,
        4 => 1.6,
        10 => 1.8,
        6 | 15 | 30 => 0.65,
        25 => 0.45,
        27 => 0.45,
        7 | 14 | 24 | 26 | 28 | 29 | 31 => 0.005,
        8 => 1.2,
        240 | 246 | 250 | 251 | 252 | 253 => 7.9,
        241 => 7.85,
        242 => 8.9,
        243 => 19.3,
        244 => 8.5,
        245 => 2.7,
        248 | 249 => 0.65,
        254 => 0.0,
        228..=230 => 2.3,
        234 | 235 => 2.6,
        236 => 2.7,
        _ => 1.0,
    }
}

/// Palette layout: 0 air, 1-31 world materials, 32-223 custom colours, 224-239 terrain,
/// 240-255 presets.
pub const IMPORT_COLOR_BASE: usize = 32;
pub const CUSTOM_COLOR_END: usize = TERRAIN2_BASE as usize;

/// Custom colour surfaces (the colour picker's choice).
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub enum SurfaceType {
    Matte,
    Satin,
    Glossy,
    Metal,
    PolishedMetal,
}

pub fn surface_of(t: SurfaceType) -> [f32; 4] {
    match t {
        SurfaceType::Matte => [0.90, 0.0, 0.04, PAT_NONE],
        SurfaceType::Satin => [0.50, 0.0, 0.04, PAT_NONE],
        SurfaceType::Glossy => [0.20, 0.0, 0.04, PAT_NONE],
        SurfaceType::Metal => [0.35, 1.0, 0.04, PAT_METAL],
        SurfaceType::PolishedMetal => [0.10, 1.0, 0.04, PAT_NONE],
    }
}

/// Writes the presets (terrain 224-236, metals, torch parts 240-254).
pub fn ensure_presets(p: &mut Vec<Material>) {
    if p.len() < 256 {
        p.resize(256, Material::default());
    }
    let mut ter = |i: u8, r: f32, g: f32, b: f32, jitter: f32, rough: f32, pattern: f32, kind: f32, variant: f32| {
        p[i as usize] = Material {
            r,
            g,
            b,
            emissive: 0.0,
            props: [material_density(i), jitter, kind, variant],
            surf: [rough, 0.0, 0.04, pattern],
        };
    };
    ter(SNOW, 0.86, 0.89, 0.93, 0.05, 0.75, PAT_SAND, 0.0, 0.0);
    ter(MUD, 0.24, 0.18, 0.12, 0.14, 0.55, PAT_SOIL, 0.0, 0.0);
    ter(CLAY, 0.55, 0.42, 0.33, 0.10, 0.85, PAT_SOIL, 0.0, 0.0);
    ter(RED_SAND, 0.74, 0.42, 0.24, 0.12, 0.90, PAT_SAND, 0.0, 0.0);
    ter(SANDSTONE, 0.78, 0.62, 0.42, 0.09, 0.85, PAT_STONE, 0.0, 0.0);
    ter(RED_SANDSTONE, 0.64, 0.33, 0.20, 0.09, 0.85, PAT_STONE, 0.0, 0.0);
    ter(PALE_SANDSTONE, 0.86, 0.76, 0.60, 0.08, 0.85, PAT_STONE, 0.0, 0.0);
    ter(SCREE, 0.47, 0.46, 0.45, 0.30, 0.85, PAT_GRAVEL, 0.0, 0.0);
    ter(LIMESTONE, 0.72, 0.70, 0.64, 0.10, 0.85, PAT_STONE, 0.0, 0.0);
    ter(SHALE, 0.34, 0.32, 0.30, 0.12, 0.80, PAT_STONE, 0.0, 0.0);
    ter(GRANITE, 0.60, 0.55, 0.52, 0.12, 0.80, PAT_STONE, 0.0, 0.0);
    ter(REEDS, 0.24, 0.38, 0.16, 0.20, 0.90, PAT_GRASS, 2.0, 5.0);
    ter(MARSH_GRASS, 0.18, 0.33, 0.15, 0.20, 0.90, PAT_GRASS, 2.0, 6.0);
    let mut set = |i: u8, r: f32, g: f32, b: f32, rough: f32, metal: f32, pattern: f32, e: f32, anim: f32| {
        p[i as usize] = Material {
            r,
            g,
            b,
            emissive: e,
            props: [material_density(i), 0.0, 0.0, anim],
            surf: [rough, metal, 0.04, pattern],
        };
    };
    set(IRON, 0.531, 0.512, 0.496, 0.45, 1.0, PAT_METAL, 0.0, 0.0);
    set(STEEL, 0.669, 0.639, 0.598, 0.30, 1.0, PAT_METAL, 0.0, 0.0);
    set(COPPER, 0.955, 0.638, 0.538, 0.35, 1.0, PAT_METAL, 0.0, 0.0);
    set(GOLD, 1.000, 0.766, 0.336, 0.25, 1.0, PAT_METAL, 0.0, 0.0);
    set(BRASS, 0.910, 0.778, 0.423, 0.35, 1.0, PAT_METAL, 0.0, 0.0);
    set(ALUMINIUM, 0.913, 0.922, 0.924, 0.35, 1.0, PAT_METAL, 0.0, 0.0);
    set(RUSTY_IRON, 0.42, 0.22, 0.12, 0.80, 0.0, PAT_RUST, 0.0, 0.0);
    set(TORCH_STICK, 0.38, 0.26, 0.14, 0.80, 0.0, PAT_WOOD, 0.0, 0.0);
    set(TORCH_HEAD, 0.85, 0.36, 0.10, 0.90, 0.0, PAT_NONE, 0.4, 0.0);
    for i in TORCH_BRACKET_N..=TORCH_BRACKET_W {
        set(i, 0.30, 0.29, 0.28, 0.60, 1.0, PAT_METAL, 0.0, 0.0);
    }
    set(TORCH_FLAME, 1.0, 0.55, 0.18, 0.90, 0.0, PAT_NONE, 7.0, 3.0);
}

/// The default palette: the built-in materials (1-31) and the presets; custom slots empty.
pub fn default_palette() -> Vec<Material> {
    let mut p = vec![Material::default(); 256];
    let mut set_mat = |i: u8, r: f32, g: f32, b: f32, e: f32, jitter: f32, kind: f32, variant: f32| {
        p[i as usize] = Material {
            r,
            g,
            b,
            emissive: e,
            props: [material_density(i), jitter, kind, variant],
            surf: Material::default().surf,
        };
    };
    set_mat(GRASS, 0.30, 0.55, 0.20, 0.0, 0.20, 2.0, 0.0);
    set_mat(DRY_GRASS, 0.46, 0.45, 0.18, 0.0, 0.20, 2.0, 1.0);
    set_mat(MOSS, 0.20, 0.36, 0.13, 0.0, 0.22, 2.0, 2.0);
    set_mat(GRAVEL, 0.50, 0.45, 0.38, 0.0, 0.30, 0.0, 0.0);
    set_mat(COARSE_DIRT, 0.37, 0.28, 0.19, 0.0, 0.20, 0.0, 0.0);
    set_mat(DIRT, 0.42, 0.30, 0.19, 0.0, 0.16, 0.0, 0.0);
    set_mat(STONE, 0.52, 0.52, 0.54, 0.0, 0.09, 0.0, 0.0);
    set_mat(SAND, 0.85, 0.78, 0.55, 0.0, 0.12, 0.0, 0.0);
    set_mat(WATER, 0.15, 0.35, 0.60, 0.0, 0.0, 1.0, 0.0);
    set_mat(WOOD, 0.38, 0.26, 0.14, 0.0, 0.17, 0.0, 0.0);
    set_mat(LEAVES, 0.18, 0.42, 0.16, 0.0, 0.24, 3.0, 0.0);
    set_mat(NEEDLES, 0.10, 0.26, 0.13, 0.0, 0.22, 3.0, 0.0);
    set_mat(BIRCH_BARK, 0.80, 0.78, 0.72, 0.0, 0.10, 0.0, 0.0);
    set_mat(PINE_BARK, 0.27, 0.18, 0.12, 0.0, 0.16, 0.0, 0.0);
    set_mat(BIRCH_LEAVES, 0.36, 0.55, 0.20, 0.0, 0.22, 3.0, 0.0);
    set_mat(DEAD_WOOD, 0.44, 0.40, 0.35, 0.0, 0.18, 0.0, 0.0);
    set_mat(BUSH_LEAVES, 0.16, 0.36, 0.14, 0.0, 0.25, 3.0, 0.0);
    set_mat(BERRIES, 0.62, 0.10, 0.14, 0.0, 0.15, 3.0, 0.0);
    set_mat(FLOWERS, 0.30, 0.55, 0.20, 0.0, 0.20, 2.0, 3.0);
    set_mat(TALL_GRASS, 0.28, 0.50, 0.18, 0.0, 0.20, 2.0, 4.0);
    set_mat(LAMP, 1.0, 0.85, 0.55, 4.0, 0.0, 0.0, 0.0);
    set_mat(TORCH, 1.0, 0.52, 0.16, 4.0, 0.0, 0.0, 1.0);
    set_mat(BRICKM, 0.62, 0.28, 0.22, 0.0, 0.11, 0.0, 0.0);
    const SPHERE_COLS: [[f32; 3]; 8] = [
        [0.90, 0.25, 0.25],
        [0.25, 0.55, 0.90],
        [0.95, 0.75, 0.20],
        [0.60, 0.30, 0.80],
        [0.20, 0.80, 0.70],
        [0.95, 0.45, 0.15],
        [0.85, 0.85, 0.88],
        [0.95, 0.40, 0.65],
    ];
    for (i, c) in SPHERE_COLS.iter().enumerate() {
        set_mat(SPHERE0 + i as u8, c[0], c[1], c[2], 0.0, 0.06, 0.0, 0.0);
    }
    let mut surf = |i: u8, rough: f32, pattern: f32, f0: f32| {
        p[i as usize].surf = [rough, 0.0, f0, pattern];
    };
    for i in [GRASS, DRY_GRASS, FLOWERS, TALL_GRASS] {
        surf(i, 0.90, PAT_GRASS, 0.04);
    }
    surf(MOSS, 1.00, PAT_GRASS, 0.04);
    surf(DIRT, 0.95, PAT_SOIL, 0.04);
    surf(COARSE_DIRT, 0.95, PAT_SOIL, 0.04);
    surf(STONE, 0.75, PAT_STONE, 0.04);
    surf(SAND, 0.90, PAT_SAND, 0.04);
    surf(GRAVEL, 0.80, PAT_GRAVEL, 0.04);
    surf(WATER, 0.02, PAT_NONE, 0.02);
    surf(WOOD, 0.70, PAT_WOOD, 0.04);
    surf(DEAD_WOOD, 0.75, PAT_WOOD, 0.04);
    surf(BIRCH_BARK, 0.80, PAT_BARK, 0.04);
    surf(PINE_BARK, 0.90, PAT_BARK, 0.04);
    for i in [LEAVES, BIRCH_LEAVES, BUSH_LEAVES] {
        surf(i, 0.55, PAT_LEAVES, 0.04);
    }
    surf(NEEDLES, 0.50, PAT_LEAVES, 0.04);
    surf(BERRIES, 0.35, PAT_NONE, 0.04);
    surf(LAMP, 0.40, PAT_NONE, 0.04);
    surf(TORCH, 0.90, PAT_NONE, 0.04);
    surf(BRICKM, 0.85, PAT_BRICK, 0.04);
    for i in 0..8 {
        surf(SPHERE0 + i, 0.45, PAT_NONE, 0.04);
    }
    ensure_presets(&mut p);
    p
}

/// A flat world: layers from the bottom up; the first fills everything below the others.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub struct FlatLayer {
    pub mat: u8,
    pub thickness: i32,
}
#[derive(Clone, Debug, PartialEq, Eq)]
pub struct FlatWorld {
    pub layers: Vec<FlatLayer>,
    /// The surface (the top of the last layer).
    pub height: i32,
    pub water: i32,
}

impl Default for FlatWorld {
    fn default() -> Self {
        Self { layers: Vec::new(), height: 64, water: 0 }
    }
}

pub struct World {
    pub vol: SparseVolume,
    pub palette: Vec<Material>,
    /// generate(): 0..1000 as chunks finish (loading screen).
    pub gen_progress: Option<std::sync::Arc<AtomicI32>>,
    next_import_slot: usize,
}

impl Deref for World {
    type Target = SparseVolume;
    fn deref(&self) -> &SparseVolume {
        &self.vol
    }
}
impl DerefMut for World {
    fn deref_mut(&mut self) -> &mut SparseVolume {
        &mut self.vol
    }
}

impl Default for World {
    fn default() -> Self {
        Self::new(WORLD_NX, WORLD_NY, WORLD_NZ)
    }
}

impl World {
    pub fn new(nx: i32, ny: i32, nz: i32) -> Self {
        Self {
            vol: SparseVolume::new(nx, ny, nz),
            palette: vec![Material::default(); 256],
            gen_progress: None,
            next_import_slot: IMPORT_COLOR_BASE,
        }
    }

    /// A new, empty world of another size (the palette is kept).
    pub fn resize(&mut self, nx: i32, ny: i32, nz: i32) {
        self.vol = SparseVolume::new(nx, ny, nz);
    }

    /// The middle of the world (spawn): a fixed world's centre; endless: the spawn area's.
    pub fn mid_x(&self) -> f32 {
        if self.ring_w != 0 { 1024.0 - self.origin_x as f32 } else { self.nx as f32 * 0.5 }
    }
    pub fn mid_z(&self) -> f32 {
        if self.ring_w != 0 { 1024.0 - self.origin_z as f32 } else { self.nz as f32 * 0.5 }
    }

    pub fn init_palette(&mut self) {
        self.palette = default_palette();
    }

    /// Generate the world (terrain, vegetation, spheres, the building) chunk by chunk, in
    /// parallel.
    pub fn generate(&mut self, seed: u32) {
        self.vol.clear();
        self.init_palette();
        self.next_import_slot = IMPORT_COLOR_BASE;
        crate::worldgen::generate_world(self, seed);
    }

    /// Instant: whole bricks per layer.
    pub fn generate_flat(&mut self, f: &FlatWorld) {
        self.vol.clear();
        self.init_palette();
        self.next_import_slot = IMPORT_COLOR_BASE;
        let ny = self.ny as usize;
        let mut col = vec![0u8; ny];
        let top = f.height.clamp(1, self.ny - 1);
        let mut y = top as usize;
        for (i, l) in f.layers.iter().enumerate().rev() {
            if y == 0 {
                break;
            }
            let t = if i == 0 { y } else { (l.thickness.max(0) as usize).min(y) };
            for _ in 0..t {
                y -= 1;
                col[y] = l.mat;
            }
        }
        for k in 0..f.water.max(0) as usize {
            if top as usize + k < ny {
                col[top as usize + k] = WATER;
            }
        }
        for by in 0..self.bricks_y {
            let ys = (by * 8) as usize;
            let uniform = (0..8).all(|ly| col[ys + ly] == col[ys]);
            let mut vox = [0u8; 512];
            for z in 0..8 {
                for ly in 0..8 {
                    vox[z * 64 + ly * 8..z * 64 + ly * 8 + 8].fill(col[ys + ly]);
                }
            }
            for bz in 0..self.bricks_z {
                for bx in 0..self.bricks_x {
                    let gi = self.grid_index(bx, by, bz);
                    if uniform {
                        self.vol.grid[gi] = col[ys] as u32;
                    } else {
                        self.vol.set_brick_voxels(gi, &vox);
                    }
                }
            }
        }
        self.vol.dirty_grid.clear();
    }

    /// A custom slot with an all-zero entry is free.
    pub fn slot_free(&self, i: usize) -> bool {
        (IMPORT_COLOR_BASE..CUSTOM_COLOR_END).contains(&i) && {
            let m = &self.palette[i];
            m.r == 0.0 && m.g == 0.0 && m.b == 0.0 && m.emissive == 0.0 && m.props == [0.0; 4]
        }
    }

    /// A palette index for (r,g,b): an existing entry within `tol`, or a new one from the
    /// pool. 0 = the pool is full and nothing matches (skip).
    pub fn allocate_color(&mut self, r: f32, g: f32, b: f32, jitter: f32, tol: f32, emissive: f32, surface: SurfaceType) -> u8 {
        let surf = surface_of(surface);
        let mut best = tol * tol;
        let mut best_idx = 0;
        for i in IMPORT_COLOR_BASE..CUSTOM_COLOR_END {
            if self.slot_free(i) {
                continue;
            }
            let m = &self.palette[i];
            if (m.emissive > 0.0) != (emissive > 0.0) {
                continue;
            }
            if (m.surf[0] - surf[0]).abs() > 0.01 || (m.surf[1] - surf[1]).abs() > 0.01 {
                continue;
            }
            let d = (m.r - r).powi(2) + (m.g - g).powi(2) + (m.b - b).powi(2);
            if d < best {
                best = d;
                best_idx = i;
            }
        }
        if best_idx != 0 {
            return best_idx as u8;
        }
        for i in IMPORT_COLOR_BASE..CUSTOM_COLOR_END {
            if self.slot_free(i) {
                self.palette[i] = Material { r, g, b, emissive, props: [1.0, jitter, 0.0, 0.0], surf };
                self.next_import_slot = self.next_import_slot.max(i + 1);
                return i as u8;
            }
        }
        0
    }

    pub fn next_import_slot(&self) -> usize {
        self.next_import_slot
    }
    pub fn set_next_import_slot(&mut self, i: usize) {
        self.next_import_slot = i.clamp(IMPORT_COLOR_BASE, CUSTOM_COLOR_END);
    }
    /// Old saves without the slot: just past the last custom colour in use.
    pub fn recompute_import_slot(&mut self) {
        self.next_import_slot = (IMPORT_COLOR_BASE..CUSTOM_COLOR_END).rev().find(|&i| !self.slot_free(i)).map_or(IMPORT_COLOR_BASE, |i| i + 1);
    }

    /// How far (x, y, z) is under the water surface above it (voxels; 0 = not in water).
    pub fn water_depth_at(&self, x: f32, y: f32, z: f32) -> f32 {
        let (ix, iy, iz) = (x.floor() as i32, y.floor() as i32, z.floor() as i32);
        if self.get_voxel(ix, iy, iz) != WATER {
            return 0.0;
        }
        let mut top = iy + 1;
        while top < self.ny && top < iy + 256 && self.get_voxel(ix, top, iz) == WATER {
            top += 1;
        }
        top as f32 - y
    }
}

// ---- palette reclamation (palette.cpp) ----------------------------------------------------

/// Voxel counts per palette index (a uniform brick counts 512 at once).
pub fn count_palette_use(v: &SparseVolume, count: &mut [u64; 256]) {
    for &g in &v.grid {
        if g < 256 {
            count[g as usize] += 512;
        } else {
            for &m in v.bricks[(g - 256) as usize].vox.iter() {
                count[m as usize] += 1;
            }
        }
    }
}

/// Zero the custom slots with no use; returns how many were freed.
pub fn reclaim_palette(world: &mut World, used: &[u64; 256]) -> usize {
    let mut n = 0;
    for i in IMPORT_COLOR_BASE..CUSTOM_COLOR_END {
        if !world.slot_free(i) && used[i] == 0 {
            world.palette[i] = Material::default();
            n += 1;
        }
    }
    n
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn palette_basics() {
        let p = default_palette();
        assert_eq!(p[GRASS as usize].props[2], 2.0);
        assert_eq!(p[WATER as usize].surf, [0.02, 0.0, 0.02, 0.0]);
        assert_eq!(p[TORCH_FLAME as usize].emissive, 7.0);
        assert_eq!(p[IRON as usize].props[0], 7.9);
        assert!(p[40] == Material::default());
    }

    #[test]
    fn allocate_reuses_and_reclaims() {
        let mut w = World::new(16, 16, 16);
        w.init_palette();
        // fresh custom slots hold Material::default() (surf non-zero) but zero colour/props: free
        assert!(w.slot_free(IMPORT_COLOR_BASE));
        let a = w.allocate_color(0.5, 0.5, 0.5, 0.05, 0.03, 0.0, SurfaceType::Matte);
        assert_eq!(a as usize, IMPORT_COLOR_BASE);
        assert_eq!(w.allocate_color(0.51, 0.5, 0.5, 0.05, 0.03, 0.0, SurfaceType::Matte), a);
        assert_ne!(w.allocate_color(0.5, 0.5, 0.5, 0.05, 0.03, 0.0, SurfaceType::Metal), a);
        let used = [0u64; 256];
        assert_eq!(reclaim_palette(&mut w, &used), 2);
        assert!(w.slot_free(a as usize));
    }

    #[test]
    fn flat_world() {
        let mut w = World::new(32, 64, 32);
        w.generate_flat(&FlatWorld {
            layers: vec![FlatLayer { mat: STONE, thickness: 0 }, FlatLayer { mat: DIRT, thickness: 3 }, FlatLayer { mat: GRASS, thickness: 1 }],
            height: 20,
            water: 2,
        });
        assert_eq!(w.get_voxel(5, 19, 5), GRASS);
        assert_eq!(w.get_voxel(5, 16, 5), DIRT);
        assert_eq!(w.get_voxel(5, 15, 5), STONE);
        assert_eq!(w.get_voxel(5, 0, 5), STONE);
        assert_eq!(w.get_voxel(5, 21, 5), WATER);
        assert_eq!(w.get_voxel(5, 22, 5), AIR);
    }
}
