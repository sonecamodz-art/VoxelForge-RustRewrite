//! Vulkan 1.3 core: instance, device, one graphics+compute+present queue, a command pool,
//! a sub-allocating memory allocator (gpu-allocator: one big block per heap instead of a
//! `vkAllocateMemory` per image like the C++ engine), and resource helpers.
//!
//! Created headless (`Context::new(&[], None, ..)`) or for a window (pass the surface
//! extensions, then `with_surface`).

use std::ffi::{c_char, CStr, CString};
use std::mem::ManuallyDrop;

use ash::{ext, khr, vk};
use gpu_allocator::vulkan::{Allocation, AllocationCreateDesc, AllocationScheme, Allocator, AllocatorCreateDesc};
use gpu_allocator::MemoryLocation;
use parking_lot::Mutex;

/// Unwrap a Vulkan result: failures are programmer errors or a lost device.
#[track_caller]
pub fn vkc<T>(r: Result<T, vk::Result>, what: &str) -> T {
    match r {
        Ok(v) => v,
        Err(e) => crate::log::fatal("vk", &format!("{what} failed: {e:?}")),
    }
}

pub struct Context {
    pub entry: ash::Entry,
    pub instance: ash::Instance,
    pub debug_utils: Option<(ext::debug_utils::Instance, vk::DebugUtilsMessengerEXT)>,
    pub debug_device: Option<ext::debug_utils::Device>,
    pub phys: vk::PhysicalDevice,
    pub props: vk::PhysicalDeviceProperties,
    pub device: ash::Device,
    pub queue_family: u32,
    pub queue: vk::Queue,
    pub cmd_pool: vk::CommandPool,
    pub allocator: ManuallyDrop<Mutex<Allocator>>,
    pub surface_fn: Option<khr::surface::Instance>,
    pub swapchain_fn: Option<khr::swapchain::Device>,
    pub validation: bool,
}

unsafe extern "system" fn debug_cb(
    sev: vk::DebugUtilsMessageSeverityFlagsEXT,
    _ty: vk::DebugUtilsMessageTypeFlagsEXT,
    data: *const vk::DebugUtilsMessengerCallbackDataEXT<'_>,
    _user: *mut std::ffi::c_void,
) -> vk::Bool32 {
    let msg = unsafe { (*data).message_as_c_str() }.map(|c| c.to_string_lossy().into_owned()).unwrap_or_default();
    if sev.contains(vk::DebugUtilsMessageSeverityFlagsEXT::ERROR) {
        log::error!(target: "validation", "{msg}");
    } else {
        log::warn!(target: "validation", "{msg}");
    }
    vk::FALSE
}

impl Context {
    /// Create instance + device. `instance_exts`: extra instance extensions (the window's
    /// surface ones); `surface_probe`: given the instance, creates the window surface so the
    /// queue family can be checked for present support.
    pub fn new(
        instance_exts: &[*const c_char],
        want_validation: bool,
        surface_probe: Option<&dyn Fn(&ash::Entry, &ash::Instance) -> vk::SurfaceKHR>,
    ) -> (Self, vk::SurfaceKHR) {
        let entry = unsafe { ash::Entry::load() }.unwrap_or_else(|e| crate::log::fatal("vk", &format!("no Vulkan loader: {e}")));
        let app_name = CString::new("VoxelForge").unwrap();
        let app = vk::ApplicationInfo::default()
            .application_name(&app_name)
            .application_version(1)
            .engine_name(&app_name)
            .api_version(vk::API_VERSION_1_3);

        let avail_layers = unsafe { entry.enumerate_instance_layer_properties() }.unwrap_or_default();
        let val_name = c"VK_LAYER_KHRONOS_validation";
        let validation = want_validation && avail_layers.iter().any(|l| l.layer_name_as_c_str() == Ok(val_name));
        if want_validation && !validation {
            log::warn!(target: "vk", "validation requested but VK_LAYER_KHRONOS_validation is not installed");
        }
        let avail_exts = unsafe { entry.enumerate_instance_extension_properties(None) }.unwrap_or_default();
        let has_ext = |n: &CStr| avail_exts.iter().any(|e| e.extension_name_as_c_str() == Ok(n));
        let mut exts: Vec<*const c_char> = instance_exts.to_vec();
        let debug_utils = has_ext(ext::debug_utils::NAME);
        if debug_utils {
            exts.push(ext::debug_utils::NAME.as_ptr());
        }
        let layers: Vec<*const c_char> = if validation { vec![val_name.as_ptr()] } else { vec![] };
        let ci = vk::InstanceCreateInfo::default().application_info(&app).enabled_extension_names(&exts).enabled_layer_names(&layers);
        let instance = vkc(unsafe { entry.create_instance(&ci, None) }, "vkCreateInstance");

        let debug = if validation && debug_utils {
            let du = ext::debug_utils::Instance::new(&entry, &instance);
            let mci = vk::DebugUtilsMessengerCreateInfoEXT::default()
                .message_severity(vk::DebugUtilsMessageSeverityFlagsEXT::WARNING | vk::DebugUtilsMessageSeverityFlagsEXT::ERROR)
                .message_type(vk::DebugUtilsMessageTypeFlagsEXT::VALIDATION | vk::DebugUtilsMessageTypeFlagsEXT::PERFORMANCE)
                .pfn_user_callback(Some(debug_cb));
            let m = vkc(unsafe { du.create_debug_utils_messenger(&mci, None) }, "debug messenger");
            Some((du, m))
        } else {
            None
        };

        let surface = surface_probe.map(|f| f(&entry, &instance)).unwrap_or(vk::SurfaceKHR::null());
        let surface_fn = (surface != vk::SurfaceKHR::null()).then(|| khr::surface::Instance::new(&entry, &instance));

        // Physical device: prefer discrete, then integrated, then anything (lavapipe).
        let devs = vkc(unsafe { instance.enumerate_physical_devices() }, "enumerate devices");
        let mut best: Option<(i32, vk::PhysicalDevice, u32)> = None;
        for &pd in &devs {
            let p = unsafe { instance.get_physical_device_properties(pd) };
            if p.api_version < vk::API_VERSION_1_3 {
                continue;
            }
            let qfs = unsafe { instance.get_physical_device_queue_family_properties(pd) };
            let qf = qfs.iter().enumerate().position(|(i, q)| {
                q.queue_flags.contains(vk::QueueFlags::GRAPHICS | vk::QueueFlags::COMPUTE)
                    && surface_fn.as_ref().is_none_or(|sf| unsafe {
                        sf.get_physical_device_surface_support(pd, i as u32, surface).unwrap_or(false)
                    })
            });
            let Some(qf) = qf else { continue };
            let score = match p.device_type {
                vk::PhysicalDeviceType::DISCRETE_GPU => 3,
                vk::PhysicalDeviceType::INTEGRATED_GPU => 2,
                vk::PhysicalDeviceType::VIRTUAL_GPU => 1,
                _ => 0,
            };
            if best.is_none_or(|b| score > b.0) {
                best = Some((score, pd, qf as u32));
            }
        }
        let (_, phys, queue_family) = best.unwrap_or_else(|| crate::log::fatal("vk", "no Vulkan 1.3 device with a graphics+compute queue"));
        let props = unsafe { instance.get_physical_device_properties(phys) };
        let name = props.device_name_as_c_str().map(|c| c.to_string_lossy().into_owned()).unwrap_or_default();
        log::info!(target: "vk", "device: {name} (driver {:#x}, Vulkan {}.{}.{})", props.driver_version,
                   vk::api_version_major(props.api_version), vk::api_version_minor(props.api_version),
                   vk::api_version_patch(props.api_version));

        let prio = [1.0f32];
        let qci = [vk::DeviceQueueCreateInfo::default().queue_family_index(queue_family).queue_priorities(&prio)];
        let mut dev_exts: Vec<*const c_char> = Vec::new();
        if surface_fn.is_some() {
            dev_exts.push(khr::swapchain::NAME.as_ptr());
        }
        let mut f13 = vk::PhysicalDeviceVulkan13Features::default()
            .synchronization2(true)
            .dynamic_rendering(true)
            .robust_image_access(std::env::var("VF_ROBUST").is_ok());
        let mut f12 = vk::PhysicalDeviceVulkan12Features::default();
        // VF_ROBUST=1: bounds-checked buffer access (debugging out-of-range shader reads).
        let f10 = vk::PhysicalDeviceFeatures::default().robust_buffer_access(std::env::var("VF_ROBUST").is_ok());
        let mut feats = vk::PhysicalDeviceFeatures2::default().features(f10).push_next(&mut f12).push_next(&mut f13);
        let dci = vk::DeviceCreateInfo::default().queue_create_infos(&qci).enabled_extension_names(&dev_exts).push_next(&mut feats);
        let device = vkc(unsafe { instance.create_device(phys, &dci, None) }, "vkCreateDevice");
        let queue = unsafe { device.get_device_queue(queue_family, 0) };
        let cmd_pool = vkc(
            unsafe {
                device.create_command_pool(
                    &vk::CommandPoolCreateInfo::default()
                        .queue_family_index(queue_family)
                        .flags(vk::CommandPoolCreateFlags::RESET_COMMAND_BUFFER),
                    None,
                )
            },
            "command pool",
        );
        let allocator = Allocator::new(&AllocatorCreateDesc {
            instance: instance.clone(),
            device: device.clone(),
            physical_device: phys,
            debug_settings: Default::default(),
            buffer_device_address: false,
            allocation_sizes: Default::default(),
        })
        .unwrap_or_else(|e| crate::log::fatal("vk", &format!("allocator: {e}")));
        let swapchain_fn = surface_fn.as_ref().map(|_| khr::swapchain::Device::new(&instance, &device));
        let debug_device = debug.as_ref().map(|_| ext::debug_utils::Device::new(&instance, &device));
        (
            Self {
                entry,
                instance,
                debug_utils: debug,
                debug_device,
                phys,
                props,
                device,
                queue_family,
                queue,
                cmd_pool,
                allocator: ManuallyDrop::new(Mutex::new(allocator)),
                surface_fn,
                swapchain_fn,
                validation,
            },
            surface,
        )
    }

    pub fn device_name(&self) -> String {
        self.props.device_name_as_c_str().map(|c| c.to_string_lossy().into_owned()).unwrap_or_default()
    }

    pub fn wait_idle(&self) {
        let _ = unsafe { self.device.device_wait_idle() };
    }

    // --- command buffers -------------------------------------------------------------------
    pub fn alloc_cmd(&self) -> vk::CommandBuffer {
        let ai = vk::CommandBufferAllocateInfo::default().command_pool(self.cmd_pool).level(vk::CommandBufferLevel::PRIMARY).command_buffer_count(1);
        vkc(unsafe { self.device.allocate_command_buffers(&ai) }, "allocate command buffer")[0]
    }

    pub fn begin_cmd(&self, cb: vk::CommandBuffer) {
        let bi = vk::CommandBufferBeginInfo::default().flags(vk::CommandBufferUsageFlags::ONE_TIME_SUBMIT);
        vkc(unsafe { self.device.begin_command_buffer(cb, &bi) }, "begin command buffer");
    }

    pub fn begin_one_shot(&self) -> vk::CommandBuffer {
        let cb = self.alloc_cmd();
        self.begin_cmd(cb);
        cb
    }

    /// Ends + submits without waiting; `fence` signals when it has run.
    pub fn submit(&self, cb: vk::CommandBuffer, fence: vk::Fence) {
        vkc(unsafe { self.device.end_command_buffer(cb) }, "end command buffer");
        let cbi = [vk::CommandBufferSubmitInfo::default().command_buffer(cb)];
        let si = [vk::SubmitInfo2::default().command_buffer_infos(&cbi)];
        vkc(unsafe { self.device.queue_submit2(self.queue, &si, fence) }, "queue submit");
    }

    pub fn end_one_shot_and_wait(&self, cb: vk::CommandBuffer) {
        let fence = self.create_fence(false);
        self.submit(cb, fence);
        self.wait_fence(fence);
        unsafe {
            self.device.destroy_fence(fence, None);
            self.device.free_command_buffers(self.cmd_pool, &[cb]);
        }
    }

    pub fn create_fence(&self, signaled: bool) -> vk::Fence {
        let fi = vk::FenceCreateInfo::default().flags(if signaled { vk::FenceCreateFlags::SIGNALED } else { vk::FenceCreateFlags::empty() });
        vkc(unsafe { self.device.create_fence(&fi, None) }, "create fence")
    }

    pub fn wait_fence(&self, f: vk::Fence) {
        vkc(unsafe { self.device.wait_for_fences(&[f], true, u64::MAX) }, "wait for fence");
    }

    pub fn create_semaphore(&self) -> vk::Semaphore {
        vkc(unsafe { self.device.create_semaphore(&vk::SemaphoreCreateInfo::default(), None) }, "create semaphore")
    }

    // --- debug labels / names ------------------------------------------------------------------
    pub fn name<H: vk::Handle>(&self, h: H, name: &str) {
        if let Some(d) = &self.debug_device {
            let n = CString::new(name).unwrap_or_default();
            let info = vk::DebugUtilsObjectNameInfoEXT::default().object_handle(h).object_name(&n);
            let _ = unsafe { d.set_debug_utils_object_name(&info) };
        }
    }

    pub fn begin_label(&self, cb: vk::CommandBuffer, label: &str) {
        if let Some(d) = &self.debug_device {
            let n = CString::new(label).unwrap_or_default();
            unsafe { d.cmd_begin_debug_utils_label(cb, &vk::DebugUtilsLabelEXT::default().label_name(&n)) };
        }
    }

    pub fn end_label(&self, cb: vk::CommandBuffer) {
        if let Some(d) = &self.debug_device {
            unsafe { d.cmd_end_debug_utils_label(cb) };
        }
    }

    // --- memory ------------------------------------------------------------------------------------
    fn allocate(&self, name: &str, req: vk::MemoryRequirements, loc: MemoryLocation, linear: bool) -> Allocation {
        self.allocator
            .lock()
            .allocate(&AllocationCreateDesc { name, requirements: req, location: loc, linear, allocation_scheme: AllocationScheme::GpuAllocatorManaged })
            .unwrap_or_else(|e| crate::log::fatal("vk", &format!("allocating {name} ({} bytes): {e}", req.size)))
    }

    pub fn free(&self, a: Allocation) {
        let _ = self.allocator.lock().free(a);
    }

    pub fn create_buffer(&self, name: &str, size: u64, usage: vk::BufferUsageFlags, loc: MemoryLocation) -> Buffer {
        let bci = vk::BufferCreateInfo::default().size(size.max(4)).usage(usage);
        let buf = vkc(unsafe { self.device.create_buffer(&bci, None) }, "create buffer");
        let req = unsafe { self.device.get_buffer_memory_requirements(buf) };
        let alloc = self.allocate(name, req, loc, true);
        vkc(unsafe { self.device.bind_buffer_memory(buf, alloc.memory(), alloc.offset()) }, "bind buffer memory");
        self.name(buf, name);
        let ptr = alloc.mapped_ptr().map(|p| p.as_ptr() as *mut u8).unwrap_or(std::ptr::null_mut());
        Buffer { buf, alloc: Some(alloc), ptr, size }
    }

    /// A host-visible buffer filled with `data`.
    pub fn upload_buffer(&self, name: &str, data: &[u8], usage: vk::BufferUsageFlags) -> Buffer {
        let b = self.create_buffer(name, data.len() as u64, usage, MemoryLocation::CpuToGpu);
        b.write(0, data);
        b
    }

    pub fn create_image(&self, name: &str, ty: vk::ImageType, fmt: vk::Format, extent: vk::Extent3D, usage: vk::ImageUsageFlags) -> Image {
        let ici = vk::ImageCreateInfo::default()
            .image_type(ty)
            .format(fmt)
            .extent(extent)
            .mip_levels(1)
            .array_layers(1)
            .samples(vk::SampleCountFlags::TYPE_1)
            .tiling(vk::ImageTiling::OPTIMAL)
            .usage(usage)
            .initial_layout(vk::ImageLayout::UNDEFINED);
        let image = vkc(unsafe { self.device.create_image(&ici, None) }, "create image");
        let req = unsafe { self.device.get_image_memory_requirements(image) };
        let alloc = self.allocate(name, req, MemoryLocation::GpuOnly, false);
        vkc(unsafe { self.device.bind_image_memory(image, alloc.memory(), alloc.offset()) }, "bind image memory");
        let vt = match ty {
            vk::ImageType::TYPE_3D => vk::ImageViewType::TYPE_3D,
            _ => vk::ImageViewType::TYPE_2D,
        };
        let vci = vk::ImageViewCreateInfo::default().image(image).view_type(vt).format(fmt).subresource_range(COLOR_RANGE);
        let view = vkc(unsafe { self.device.create_image_view(&vci, None) }, "create image view");
        self.name(image, name);
        self.name(view, name);
        Image { image, view, alloc: Some(alloc), format: fmt, extent }
    }

    /// The renderer's standard 2D image: storage + sampled + transfer src/dst (GENERAL).
    pub fn image_2d(&self, name: &str, fmt: vk::Format, w: u32, h: u32) -> Image {
        self.create_image(
            name,
            vk::ImageType::TYPE_2D,
            fmt,
            vk::Extent3D { width: w.max(1), height: h.max(1), depth: 1 },
            vk::ImageUsageFlags::STORAGE | vk::ImageUsageFlags::SAMPLED | vk::ImageUsageFlags::TRANSFER_SRC | vk::ImageUsageFlags::TRANSFER_DST,
        )
    }

    pub fn image_3d(&self, name: &str, fmt: vk::Format, w: u32, h: u32, d: u32, usage: vk::ImageUsageFlags) -> Image {
        self.create_image(name, vk::ImageType::TYPE_3D, fmt, vk::Extent3D { width: w.max(1), height: h.max(1), depth: d.max(1) }, usage)
    }

    pub fn sampler(&self, filter: vk::Filter) -> vk::Sampler {
        let s = vk::SamplerCreateInfo::default()
            .mag_filter(filter)
            .min_filter(filter)
            .mipmap_mode(vk::SamplerMipmapMode::NEAREST)
            .address_mode_u(vk::SamplerAddressMode::CLAMP_TO_EDGE)
            .address_mode_v(vk::SamplerAddressMode::CLAMP_TO_EDGE)
            .address_mode_w(vk::SamplerAddressMode::CLAMP_TO_EDGE);
        vkc(unsafe { self.device.create_sampler(&s, None) }, "create sampler")
    }

    pub fn destroy(&mut self) {
        unsafe {
            self.device.device_wait_idle().ok();
            self.device.destroy_command_pool(self.cmd_pool, None);
            ManuallyDrop::drop(&mut self.allocator);
            self.device.destroy_device(None);
            if let Some((du, m)) = self.debug_utils.take() {
                du.destroy_debug_utils_messenger(m, None);
            }
            self.instance.destroy_instance(None);
        }
    }
}

pub const COLOR_RANGE: vk::ImageSubresourceRange =
    vk::ImageSubresourceRange { aspect_mask: vk::ImageAspectFlags::COLOR, base_mip_level: 0, level_count: 1, base_array_layer: 0, layer_count: 1 };
pub const COLOR_LAYERS: vk::ImageSubresourceLayers =
    vk::ImageSubresourceLayers { aspect_mask: vk::ImageAspectFlags::COLOR, mip_level: 0, base_array_layer: 0, layer_count: 1 };

/// A buffer + its memory. Host-visible ones stay persistently mapped (`ptr`).
pub struct Buffer {
    pub buf: vk::Buffer,
    pub alloc: Option<Allocation>,
    pub ptr: *mut u8,
    pub size: u64,
}

// The mapped pointer is only written by the thread that owns the renderer.
unsafe impl Send for Buffer {}

impl Default for Buffer {
    fn default() -> Self {
        Self { buf: vk::Buffer::null(), alloc: None, ptr: std::ptr::null_mut(), size: 0 }
    }
}

impl Buffer {
    pub fn write(&self, offset: usize, data: &[u8]) {
        assert!(!self.ptr.is_null(), "buffer not host visible");
        assert!(offset + data.len() <= self.size as usize, "buffer write out of range");
        unsafe { std::ptr::copy_nonoverlapping(data.as_ptr(), self.ptr.add(offset), data.len()) };
    }
    pub fn write_pod<T: bytemuck::Pod>(&self, offset: usize, v: &T) {
        self.write(offset, bytemuck::bytes_of(v));
    }
    pub fn mapped(&self) -> &[u8] {
        unsafe { std::slice::from_raw_parts(self.ptr, self.size as usize) }
    }
    pub fn is_null(&self) -> bool {
        self.buf == vk::Buffer::null()
    }
    pub fn destroy(&mut self, ctx: &Context) {
        if self.buf != vk::Buffer::null() {
            unsafe { ctx.device.destroy_buffer(self.buf, None) };
        }
        if let Some(a) = self.alloc.take() {
            ctx.free(a);
        }
        *self = Buffer::default();
    }
}

/// A device-local image + view.
pub struct Image {
    pub image: vk::Image,
    pub view: vk::ImageView,
    pub alloc: Option<Allocation>,
    pub format: vk::Format,
    pub extent: vk::Extent3D,
}

impl Default for Image {
    fn default() -> Self {
        Self { image: vk::Image::null(), view: vk::ImageView::null(), alloc: None, format: vk::Format::UNDEFINED, extent: vk::Extent3D::default() }
    }
}

impl Image {
    pub fn destroy(&mut self, ctx: &Context) {
        unsafe {
            if self.view != vk::ImageView::null() {
                ctx.device.destroy_image_view(self.view, None);
            }
            if self.image != vk::Image::null() {
                ctx.device.destroy_image(self.image, None);
            }
        }
        if let Some(a) = self.alloc.take() {
            ctx.free(a);
        }
        *self = Image::default();
    }
    pub fn w(&self) -> u32 {
        self.extent.width
    }
    pub fn h(&self) -> u32 {
        self.extent.height
    }
}

// --- barriers ---------------------------------------------------------------------------------
pub type Stage = vk::PipelineStageFlags2;
pub type Access = vk::AccessFlags2;

/// sync2 image layout transition.
#[allow(clippy::too_many_arguments)]
pub fn image_barrier(
    dev: &ash::Device,
    cb: vk::CommandBuffer,
    image: vk::Image,
    src_stage: Stage,
    src_access: Access,
    dst_stage: Stage,
    dst_access: Access,
    old: vk::ImageLayout,
    new: vk::ImageLayout,
) {
    let b = [vk::ImageMemoryBarrier2::default()
        .src_stage_mask(src_stage)
        .src_access_mask(src_access)
        .dst_stage_mask(dst_stage)
        .dst_access_mask(dst_access)
        .old_layout(old)
        .new_layout(new)
        .src_queue_family_index(vk::QUEUE_FAMILY_IGNORED)
        .dst_queue_family_index(vk::QUEUE_FAMILY_IGNORED)
        .image(image)
        .subresource_range(COLOR_RANGE)];
    unsafe { dev.cmd_pipeline_barrier2(cb, &vk::DependencyInfo::default().image_memory_barriers(&b)) };
}

/// A global memory barrier.
pub fn memory_barrier(dev: &ash::Device, cb: vk::CommandBuffer, src_stage: Stage, src_access: Access, dst_stage: Stage, dst_access: Access) {
    let b = [vk::MemoryBarrier2::default().src_stage_mask(src_stage).src_access_mask(src_access).dst_stage_mask(dst_stage).dst_access_mask(dst_access)];
    unsafe { dev.cmd_pipeline_barrier2(cb, &vk::DependencyInfo::default().memory_barriers(&b)) };
}

/// Compute -> compute (write-after-write too: ping-pong chains overwrite earlier outputs).
pub fn pass_barrier(dev: &ash::Device, cb: vk::CommandBuffer) {
    memory_barrier(
        dev,
        cb,
        Stage::COMPUTE_SHADER,
        Access::SHADER_STORAGE_WRITE,
        Stage::COMPUTE_SHADER,
        Access::SHADER_STORAGE_READ | Access::SHADER_SAMPLED_READ | Access::SHADER_STORAGE_WRITE,
    );
}

/// UNDEFINED -> GENERAL for freshly created images (one barrier per image).
pub fn to_general(dev: &ash::Device, cb: vk::CommandBuffer, images: &[vk::Image]) {
    let bs: Vec<_> = images
        .iter()
        .map(|&image| {
            vk::ImageMemoryBarrier2::default()
                .src_stage_mask(Stage::TOP_OF_PIPE)
                .dst_stage_mask(Stage::ALL_COMMANDS)
                .dst_access_mask(Access::MEMORY_READ | Access::MEMORY_WRITE)
                .old_layout(vk::ImageLayout::UNDEFINED)
                .new_layout(vk::ImageLayout::GENERAL)
                .src_queue_family_index(vk::QUEUE_FAMILY_IGNORED)
                .dst_queue_family_index(vk::QUEUE_FAMILY_IGNORED)
                .image(image)
                .subresource_range(COLOR_RANGE)
        })
        .collect();
    unsafe { dev.cmd_pipeline_barrier2(cb, &vk::DependencyInfo::default().image_memory_barriers(&bs)) };
}

pub fn clear_color_f(dev: &ash::Device, cb: vk::CommandBuffer, image: vk::Image, v: [f32; 4]) {
    unsafe { dev.cmd_clear_color_image(cb, image, vk::ImageLayout::GENERAL, &vk::ClearColorValue { float32: v }, &[COLOR_RANGE]) };
}

pub fn clear_color_u(dev: &ash::Device, cb: vk::CommandBuffer, image: vk::Image, v: [u32; 4]) {
    unsafe { dev.cmd_clear_color_image(cb, image, vk::ImageLayout::GENERAL, &vk::ClearColorValue { uint32: v }, &[COLOR_RANGE]) };
}
