//! Runtime GLSL -> SPIR-V (shaderc) + compute pipeline / descriptor helpers.
//!
//! Shaders stay plain text on disk (hot-reloadable with `shader reload` / F10), concatenated
//! after a `#version 460` header and a macro preamble, with `#line` markers so compile errors
//! name the real file and line - exactly the C++ engine's scheme, so the GLSL is shared
//! verbatim.
//!
//! Improvements over the C++ engine:
//!  * SPIR-V is optimised (`-O`, the C++ build shipped unoptimised `-O0` SPIR-V);
//!  * compiled SPIR-V is cached on disk by a hash of its full source, so only the first run
//!    (or an edited shader) pays the compile;
//!  * the driver's pipeline cache is persisted too (`pipelines.bin`);
//!  * every shader file is also embedded in the binary: a release build runs without the
//!    `shaders/` directory (disk copies still win, for hot reload).

use std::path::PathBuf;

use ash::vk;

use super::context::{vkc, Context};
use crate::cvars;

/// The shader files, embedded at build time (the fallback when no `shaders/` dir is found).
macro_rules! embed_shaders {
    ($($f:literal),* $(,)?) => {
        const EMBEDDED: &[(&str, &str)] = &[$(($f, include_str!(concat!("../../shaders/", $f)))),*];
    };
}
embed_shaders!(
    "atrous.comp", "bloom.comp", "brickdist.comp", "brush_preview.comp", "clouds.comp", "clouds_pano.comp",
    "clouds_resolve.comp", "common.glsl", "denoise.comp", "dust.comp", "exposure.comp", "fog.comp",
    "froxel_inject.comp", "froxel_integrate.comp", "froxel_tiles.comp", "gbuffer.comp", "gi.comp",
    "gi_cache_resolve.comp", "gi_cache_update.comp", "gi_stab.comp", "grass.glsl", "grass_wind.comp",
    "grasstop.comp", "lamp_cube.comp", "lamp_std.comp", "lighting.comp", "lock_splat.comp", "lockview.comp",
    "nrd_pack.comp", "probe_trace.comp", "probe_update.comp", "probes.glsl", "rain.glsl", "raw_gi.comp",
    "raw_lamps.comp", "raw_shade.comp", "shade_std.comp", "shadow_soften.comp", "shadow_trace.comp",
    "sunshadow.comp", "tonemap.comp", "tonemap.glsl", "torchfx.comp", "torchfx.glsl", "water_refl.comp",
    "water_refl.glsl", "water_refl_half.comp",
);

/// Load `shaders/<name>`: from disk (searching the shader dirs), else the embedded copy.
pub fn load_source(name: &str) -> Option<String> {
    for d in crate::paths::shader_dirs() {
        if let Ok(s) = std::fs::read_to_string(d.join(name)) {
            return Some(s);
        }
    }
    EMBEDDED.iter().find(|(n, _)| *n == name).map(|(_, s)| s.to_string())
}

/// `#version 460` + extensions + the preamble + each part with a `#line 1 "file"` marker.
pub fn assemble(parts: &[&str], preamble: &str) -> Result<String, String> {
    let mut src = String::from(
        "#version 460\n#extension GL_GOOGLE_cpp_style_line_directive : require\n\
         #extension GL_KHR_shader_subgroup_quad : require\n#line 1 \"<preamble>\"\n",
    );
    src += preamble;
    for p in parts {
        let text = load_source(p).ok_or_else(|| format!("missing shader file shaders/{p}"))?;
        src += &format!("#line 1 \"shaders/{p}\"\n");
        src += &text;
        src.push('\n');
    }
    Ok(src)
}

/// 64-bit FNV-1a (the SPIR-V cache key).
fn fnv64(parts: &[&[u8]]) -> u64 {
    let mut h: u64 = 0xcbf2_9ce4_8422_2325;
    for p in parts {
        for &b in *p {
            h ^= b as u64;
            h = h.wrapping_mul(0x100_0000_01b3);
        }
        h ^= 0xff;
        h = h.wrapping_mul(0x100_0000_01b3);
    }
    h
}

fn cache_dir() -> PathBuf {
    crate::paths::data_path("shader_cache")
}

/// Compile a full GLSL compute source to SPIR-V (Vulkan 1.3), through the disk cache.
pub fn compile(source: &str, name: &str) -> Result<Vec<u32>, String> {
    let opt = cvars::R_SHADER_OPT.get();
    let use_cache = cvars::R_SHADER_CACHE.get();
    let key = fnv64(&[source.as_bytes(), if opt { b"O" } else { b"O0" }, b"shaderc-v1"]);
    let path = cache_dir().join(format!("{key:016x}.spv"));
    if use_cache {
        if let Ok(bytes) = std::fs::read(&path) {
            if bytes.len() % 4 == 0 && bytes.len() >= 20 {
                let words: Vec<u32> = bytes.chunks_exact(4).map(|c| u32::from_le_bytes([c[0], c[1], c[2], c[3]])).collect();
                if words[0] == 0x0723_0203 {
                    return Ok(words);
                }
            }
        }
    }
    let compiler = shaderc::Compiler::new().ok_or("shaderc unavailable")?;
    let mut o = shaderc::CompileOptions::new().ok_or("shaderc options")?;
    o.set_source_language(shaderc::SourceLanguage::GLSL);
    o.set_target_env(shaderc::TargetEnv::Vulkan, shaderc::EnvVersion::Vulkan1_3 as u32);
    o.set_target_spirv(shaderc::SpirvVersion::V1_6);
    o.set_optimization_level(if opt { shaderc::OptimizationLevel::Performance } else { shaderc::OptimizationLevel::Zero });
    let art = compiler.compile_into_spirv(source, shaderc::ShaderKind::Compute, name, "main", Some(&o)).map_err(|e| e.to_string())?;
    if art.get_num_warnings() > 0 {
        log::warn!(target: "shader", "{name}: {}", art.get_warning_messages());
    }
    let words = art.as_binary().to_vec();
    if use_cache {
        let _ = std::fs::create_dir_all(cache_dir());
        let tmp = path.with_extension("tmp");
        if std::fs::write(&tmp, bytemuck::cast_slice::<u32, u8>(&words)).is_ok() {
            let _ = std::fs::rename(&tmp, &path);
        }
    }
    Ok(words)
}

/// assemble + compile.
pub fn compile_parts(parts: &[&str], preamble: &str) -> Result<Vec<u32>, String> {
    let src = assemble(parts, preamble)?;
    compile(&src, &parts.join("+"))
}

// --- pipelines ---------------------------------------------------------------------------------

#[derive(Clone, Copy, Default)]
pub struct ComputePipeline {
    pub pipeline: vk::Pipeline,
    pub layout: vk::PipelineLayout,
}

impl ComputePipeline {
    pub fn destroy(&mut self, dev: &ash::Device) {
        unsafe {
            if self.pipeline != vk::Pipeline::null() {
                dev.destroy_pipeline(self.pipeline, None);
            }
            if self.layout != vk::PipelineLayout::null() {
                dev.destroy_pipeline_layout(self.layout, None);
            }
        }
        *self = Self::default();
    }

    pub fn bind(&self, dev: &ash::Device, cb: vk::CommandBuffer, sets: &[vk::DescriptorSet]) {
        unsafe {
            dev.cmd_bind_pipeline(cb, vk::PipelineBindPoint::COMPUTE, self.pipeline);
            dev.cmd_bind_descriptor_sets(cb, vk::PipelineBindPoint::COMPUTE, self.layout, 0, sets, &[]);
        }
    }

    pub fn push<T: bytemuck::Pod>(&self, dev: &ash::Device, cb: vk::CommandBuffer, pc: &T) {
        unsafe { dev.cmd_push_constants(cb, self.layout, vk::ShaderStageFlags::COMPUTE, 0, bytemuck::bytes_of(pc)) };
    }
}

/// The driver's pipeline cache, persisted between runs.
pub struct PipelineCache {
    pub cache: vk::PipelineCache,
}

impl PipelineCache {
    fn path() -> PathBuf {
        cache_dir().join("pipelines.bin")
    }

    pub fn load(ctx: &Context) -> Self {
        let data = if cvars::R_SHADER_CACHE.get() { std::fs::read(Self::path()).unwrap_or_default() } else { vec![] };
        let ci = vk::PipelineCacheCreateInfo::default().initial_data(&data);
        let cache = unsafe { ctx.device.create_pipeline_cache(&ci, None) }
            .or_else(|_| unsafe { ctx.device.create_pipeline_cache(&vk::PipelineCacheCreateInfo::default(), None) });
        Self { cache: vkc(cache, "pipeline cache") }
    }

    pub fn save(&self, ctx: &Context) {
        if !cvars::R_SHADER_CACHE.get() {
            return;
        }
        if let Ok(data) = unsafe { ctx.device.get_pipeline_cache_data(self.cache) } {
            let _ = std::fs::create_dir_all(cache_dir());
            let _ = std::fs::write(Self::path(), data);
        }
    }

    pub fn destroy(&mut self, ctx: &Context) {
        self.save(ctx);
        unsafe { ctx.device.destroy_pipeline_cache(self.cache, None) };
        self.cache = vk::PipelineCache::null();
    }
}

pub fn make_set_layout(dev: &ash::Device, bindings: &[(u32, vk::DescriptorType)]) -> vk::DescriptorSetLayout {
    let b: Vec<_> = bindings
        .iter()
        .map(|&(i, t)| vk::DescriptorSetLayoutBinding::default().binding(i).descriptor_type(t).descriptor_count(1).stage_flags(vk::ShaderStageFlags::COMPUTE))
        .collect();
    vkc(unsafe { dev.create_descriptor_set_layout(&vk::DescriptorSetLayoutCreateInfo::default().bindings(&b), None) }, "set layout")
}

pub fn make_compute_pipeline(
    dev: &ash::Device,
    cache: vk::PipelineCache,
    spv: &[u32],
    set_layouts: &[vk::DescriptorSetLayout],
    push_bytes: u32,
) -> ComputePipeline {
    let pcr = [vk::PushConstantRange::default().stage_flags(vk::ShaderStageFlags::COMPUTE).size(push_bytes)];
    let mut lci = vk::PipelineLayoutCreateInfo::default().set_layouts(set_layouts);
    if push_bytes > 0 {
        lci = lci.push_constant_ranges(&pcr);
    }
    let layout = vkc(unsafe { dev.create_pipeline_layout(&lci, None) }, "pipeline layout");
    let module = vkc(unsafe { dev.create_shader_module(&vk::ShaderModuleCreateInfo::default().code(spv), None) }, "shader module");
    let stage = vk::PipelineShaderStageCreateInfo::default().stage(vk::ShaderStageFlags::COMPUTE).module(module).name(c"main");
    let ci = [vk::ComputePipelineCreateInfo::default().stage(stage).layout(layout)];
    let p = unsafe { dev.create_compute_pipelines(cache, &ci, None) }.map_err(|(_, e)| e);
    let pipeline = vkc(p, "compute pipeline")[0];
    unsafe { dev.destroy_shader_module(module, None) };
    ComputePipeline { pipeline, layout }
}

#[cfg(test)]
mod tests {
    #[test]
    fn every_shader_is_embedded() {
        let dir = concat!(env!("CARGO_MANIFEST_DIR"), "/shaders");
        for e in std::fs::read_dir(dir).unwrap() {
            let n = e.unwrap().file_name().into_string().unwrap();
            assert!(super::EMBEDDED.iter().any(|(f, _)| *f == n), "{n} is not in embed_shaders!");
        }
    }
}
