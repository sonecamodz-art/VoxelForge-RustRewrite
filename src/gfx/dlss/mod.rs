//! NVIDIA DLSS through NGX: Ray Reconstruction (a neural denoiser + upscaler that replaces the
//! SVGF passes) and Super Resolution / DLAA (anti-aliasing and upscaling of an already clean
//! image). Built only with `--features dlss` (the SDK from github.com/NVIDIA/DLSS, see
//! build.rs); without it everything here reports "unavailable" and the renderer runs SVGF.
//! The SDK's runtime libraries (libnvidia-ngx-dlss*.so / nvngx_dlss*.dll) are found next to
//! the executable, in the SDK's lib folder, or in VF_DLSS_LIBS.

use std::ffi::CString;

/// Upscaling quality: how far below the display resolution the frame renders. DLAA is 1:1.
#[derive(Clone, Copy, PartialEq, Eq, Debug)]
pub enum Quality {
    Dlaa,
    Quality,
    Balanced,
    Performance,
    UltraPerf,
}

impl Quality {
    pub fn from_index(i: i32) -> Self {
        [Self::Dlaa, Self::Quality, Self::Balanced, Self::Performance, Self::UltraPerf][i.clamp(0, 4) as usize]
    }
    pub fn name(self) -> &'static str {
        match self {
            Self::Dlaa => "DLAA",
            Self::Quality => "Quality",
            Self::Balanced => "Balanced",
            Self::Performance => "Performance",
            Self::UltraPerf => "Ultra Performance",
        }
    }
}

/// One image NGX reads or writes (VK_IMAGE_LAYOUT_GENERAL).
#[derive(Clone, Copy)]
pub struct GImg {
    pub image: ash::vk::Image,
    pub view: ash::vk::ImageView,
    pub format: ash::vk::Format,
}

#[cfg(feature = "dlss")]
mod ngx;
#[cfg(feature = "dlss")]
pub use ngx::Dlss;

/// NGX's required Vulkan instance and device extensions (before the device exists).
pub fn required_extensions() -> (Vec<CString>, Vec<CString>) {
    #[cfg(feature = "dlss")]
    {
        ngx::required_extensions()
    }
    #[cfg(not(feature = "dlss"))]
    {
        (Vec::new(), Vec::new())
    }
}

#[cfg(not(feature = "dlss"))]
mod stub {
    use super::{GImg, Quality};
    use crate::gfx::context::Context;
    use ash::vk;

    /// Built without the DLSS SDK: nothing is available.
    #[derive(Default)]
    pub struct Dlss {
        pub rr_available: bool,
        pub sr_available: bool,
    }

    impl Dlss {
        pub fn init(_ctx: &Context) -> Self {
            log::info!(target: "dlss", "built without the DLSS SDK (cargo build --features dlss) - Ray Reconstruction / DLAA unavailable");
            Self::default()
        }
        pub fn optimal_render_size(&self, _q: Quality, tw: u32, th: u32) -> (u32, u32) {
            (tw, th)
        }
        pub fn create_rr(&mut self, _ctx: &Context, _cb: vk::CommandBuffer, _r: (u32, u32), _t: (u32, u32), _q: Quality) -> bool {
            false
        }
        pub fn create_sr(&mut self, _ctx: &Context, _cb: vk::CommandBuffer, _r: (u32, u32), _t: (u32, u32), _q: Quality) -> bool {
            false
        }
        #[allow(clippy::too_many_arguments)]
        pub fn evaluate_rr(&self, _cb: vk::CommandBuffer, _color: GImg, _diff: GImg, _spec: GImg, _normal_rough: GImg, _depth: GImg, _motion: GImg,
                           _out: GImg, _reset: bool, _jitter: [f32; 2]) {
        }
        #[allow(clippy::too_many_arguments)]
        pub fn evaluate_sr(&self, _cb: vk::CommandBuffer, _color: GImg, _hw_depth: GImg, _motion: GImg, _out: GImg, _jitter: [f32; 2], _reset: bool) {}
        pub fn release_rr(&mut self) {}
        pub fn release_sr(&mut self) {}
        pub fn rr_active(&self) -> bool {
            false
        }
        pub fn sr_active(&self) -> bool {
            false
        }
        pub fn destroy(&mut self, _ctx: &Context) {}
    }
}
#[cfg(not(feature = "dlss"))]
pub use stub::Dlss;
