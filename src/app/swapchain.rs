//! The window's swapchain (FIFO / mailbox / immediate), recreated on resize.

use ash::vk;

use crate::gfx::context::{vkc, Context};

pub struct Swapchain {
    pub handle: vk::SwapchainKHR,
    pub format: vk::Format,
    pub extent: vk::Extent2D,
    pub images: Vec<vk::Image>,
    /// One per swapchain image (presentation waits on it).
    pub render_finished: Vec<vk::Semaphore>,
    /// The previous swapchain's semaphores: a present may still wait on them.
    retired: Vec<vk::Semaphore>,
    pub present_mode: vk::PresentModeKHR,
    pub has_mailbox: bool,
    pub has_immediate: bool,
}

impl Swapchain {
    pub fn new(ctx: &Context, surface: vk::SurfaceKHR, want: (u32, u32), mode: vk::PresentModeKHR) -> Self {
        let mut sc = Swapchain {
            handle: vk::SwapchainKHR::null(),
            format: vk::Format::UNDEFINED,
            extent: vk::Extent2D::default(),
            images: Vec::new(),
            render_finished: Vec::new(),
            retired: Vec::new(),
            present_mode: mode,
            has_mailbox: false,
            has_immediate: false,
        };
        sc.create(ctx, surface, want);
        sc
    }

    fn create(&mut self, ctx: &Context, surface: vk::SurfaceKHR, want: (u32, u32)) {
        let sf = ctx.surface_fn.as_ref().expect("no surface support");
        let sw = ctx.swapchain_fn.as_ref().expect("no swapchain support");
        let caps = vkc(unsafe { sf.get_physical_device_surface_capabilities(ctx.phys, surface) }, "surface caps");
        let fmts = vkc(unsafe { sf.get_physical_device_surface_formats(ctx.phys, surface) }, "surface formats");
        let chosen = fmts
            .iter()
            .find(|f| f.format == vk::Format::B8G8R8A8_UNORM && f.color_space == vk::ColorSpaceKHR::SRGB_NONLINEAR)
            .or_else(|| fmts.iter().find(|f| f.format == vk::Format::R8G8B8A8_UNORM))
            .copied()
            .unwrap_or(fmts[0]);
        let mut extent = caps.current_extent;
        if extent.width == u32::MAX {
            extent = vk::Extent2D {
                width: want.0.clamp(caps.min_image_extent.width, caps.max_image_extent.width),
                height: want.1.clamp(caps.min_image_extent.height, caps.max_image_extent.height),
            };
        }
        let modes = unsafe { sf.get_physical_device_surface_present_modes(ctx.phys, surface) }.unwrap_or_default();
        self.has_mailbox = modes.contains(&vk::PresentModeKHR::MAILBOX);
        self.has_immediate = modes.contains(&vk::PresentModeKHR::IMMEDIATE);
        if !modes.contains(&self.present_mode) {
            self.present_mode = vk::PresentModeKHR::FIFO;
        }
        let mut n = caps.min_image_count + 1;
        if self.present_mode == vk::PresentModeKHR::MAILBOX {
            n = n.max(3);
        }
        if caps.max_image_count > 0 {
            n = n.min(caps.max_image_count);
        }
        let old = self.handle;
        let ci = vk::SwapchainCreateInfoKHR::default()
            .surface(surface)
            .min_image_count(n)
            .image_format(chosen.format)
            .image_color_space(chosen.color_space)
            .image_extent(extent)
            .image_array_layers(1)
            .image_usage(vk::ImageUsageFlags::COLOR_ATTACHMENT | vk::ImageUsageFlags::TRANSFER_DST | vk::ImageUsageFlags::TRANSFER_SRC)
            .image_sharing_mode(vk::SharingMode::EXCLUSIVE)
            .pre_transform(caps.current_transform)
            .composite_alpha(vk::CompositeAlphaFlagsKHR::OPAQUE)
            .present_mode(self.present_mode)
            .clipped(true)
            .old_swapchain(old);
        self.handle = vkc(unsafe { sw.create_swapchain(&ci, None) }, "create swapchain");
        if old != vk::SwapchainKHR::null() {
            unsafe { sw.destroy_swapchain(old, None) };
        }
        self.format = chosen.format;
        self.extent = extent;
        self.images = vkc(unsafe { sw.get_swapchain_images(self.handle) }, "swapchain images");
        self.render_finished = self.images.iter().map(|_| ctx.create_semaphore()).collect();
    }

    /// Replace the swapchain (resize / present mode): the old one is handed over.
    pub fn recreate(&mut self, ctx: &Context, surface: vk::SurfaceKHR, want: (u32, u32)) {
        ctx.wait_idle();
        for s in self.retired.drain(..) {
            unsafe { ctx.device.destroy_semaphore(s, None) };
        }
        self.retired = std::mem::take(&mut self.render_finished);
        self.create(ctx, surface, want);
    }

    pub fn destroy(&mut self, ctx: &Context) {
        unsafe {
            for s in self.render_finished.drain(..).chain(self.retired.drain(..)) {
                ctx.device.destroy_semaphore(s, None);
            }
            if let Some(sw) = &ctx.swapchain_fn {
                sw.destroy_swapchain(self.handle, None);
            }
        }
        self.handle = vk::SwapchainKHR::null();
    }
}

pub fn present_mode_from_cvar() -> vk::PresentModeKHR {
    match crate::cvars::R_PRESENT.index() {
        1 => vk::PresentModeKHR::MAILBOX,
        2 => vk::PresentModeKHR::IMMEDIATE,
        _ => vk::PresentModeKHR::FIFO,
    }
}
