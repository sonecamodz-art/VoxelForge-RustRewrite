//! The NGX side of DLSS (`--features dlss`): native/ngx_shim.cpp over the SDK's helpers.

use std::ffi::{c_char, c_int, c_uint, c_ulonglong, c_void, CStr, CString};
use std::path::PathBuf;

use ash::vk::{self, Handle};

use super::{GImg, Quality};
use crate::gfx::context::Context;

/// The application's NGX project id (a UUID of our choosing for engines without an NVIDIA app id).
const PROJECT_ID: &CStr = c"a0f4b2c1-1234-5678-9abc-def012345678";

#[repr(C)]
struct VfNgxImg {
    image: u64,
    view: u64,
    format: i32,
    width: c_uint,
    height: c_uint,
}

unsafe extern "C" {
    fn vf_ngx_required_extensions(ic: *mut c_uint, ie: *mut *const *const c_char, dc: *mut c_uint, de: *mut *const *const c_char) -> c_int;
    fn vf_ngx_init(project_id: *const c_char, snippet_dir: *const c_char, data_dir: *const c_char, inst: u64, pd: u64, dev: u64,
                   gipa: *const c_void, gdpa: *const c_void, params: *mut *mut c_void, rr: *mut c_uint, sr: *mut c_uint) -> c_int;
    fn vf_ngx_optimal(params: *mut c_void, rr: c_int, tw: c_uint, th: c_uint, q: c_int, rw: *mut c_uint, rh: *mut c_uint) -> c_int;
    fn vf_ngx_create_rr(dev: u64, cb: u64, params: *mut c_void, rw: c_uint, rh: c_uint, tw: c_uint, th: c_uint, q: c_int, handle: *mut *mut c_void,
                        vram: *mut c_ulonglong) -> c_int;
    fn vf_ngx_create_sr(dev: u64, cb: u64, params: *mut c_void, rw: c_uint, rh: c_uint, tw: c_uint, th: c_uint, q: c_int, handle: *mut *mut c_void,
                        vram: *mut c_ulonglong) -> c_int;
    fn vf_ngx_eval_rr(cb: u64, handle: *mut c_void, params: *mut c_void, color: *const VfNgxImg, diffuse: *const VfNgxImg, specular: *const VfNgxImg,
                      normal_rough: *const VfNgxImg, depth: *const VfNgxImg, motion: *const VfNgxImg, output: *const VfNgxImg, reset: c_int, jx: f32,
                      jy: f32, mvx: f32, mvy: f32) -> c_int;
    fn vf_ngx_eval_sr(cb: u64, handle: *mut c_void, params: *mut c_void, color: *const VfNgxImg, hw_depth: *const VfNgxImg, motion: *const VfNgxImg,
                      output: *const VfNgxImg, reset: c_int, jx: f32, jy: f32, mvx: f32, mvy: f32) -> c_int;
    fn vf_ngx_release(handle: *mut c_void);
    fn vf_ngx_shutdown(dev: u64);
}

/// NVSDK_NGX_Result: success has the high bits clear (NVSDK_NGX_SUCCEED).
fn failed(r: c_int) -> bool {
    (r as u32 & 0xFFF0_0000) == 0xBAD0_0000
}

pub fn required_extensions() -> (Vec<CString>, Vec<CString>) {
    let (mut ic, mut dc) = (0, 0);
    let (mut ie, mut de) = (std::ptr::null(), std::ptr::null());
    let r = unsafe { vf_ngx_required_extensions(&mut ic, &mut ie, &mut dc, &mut de) };
    if failed(r) || ie.is_null() || de.is_null() {
        return (Vec::new(), Vec::new());
    }
    let list = |p: *const *const c_char, n: c_uint| -> Vec<CString> {
        (0..n as usize).map(|i| unsafe { CStr::from_ptr(*p.add(i)) }.to_owned()).collect()
    };
    (list(ie, ic), list(de, dc))
}

/// Where NGX finds its feature libraries (libnvidia-ngx-dlss*.so.<ver> / nvngx_dlss*.dll):
/// $VF_DLSS_LIBS, beside the executable, ../lib/dlss (a packaged build), the SDK this was built
/// against, ./external/DLSS.
fn snippet_dir() -> PathBuf {
    let exe = std::env::current_exe().ok().and_then(|p| p.parent().map(|d| d.to_path_buf())).unwrap_or_default();
    let has_libs = |d: &PathBuf| {
        std::fs::read_dir(d).is_ok_and(|rd| rd.flatten().any(|e| e.file_name().to_string_lossy().contains("ngx_dlss")))
    };
    let rel = if cfg!(windows) { "external/DLSS/lib/Windows_x86_64/rel" } else { "external/DLSS/lib/Linux_x86_64/rel" };
    let mut cands: Vec<PathBuf> = std::env::var_os("VF_DLSS_LIBS").map(PathBuf::from).into_iter().collect();
    cands.extend([exe.clone(), exe.join("../lib/dlss"), PathBuf::from(env!("VF_DLSS_SNIPPETS")), PathBuf::from(rel)]);
    for c in &cands {
        if has_libs(c) {
            return c.canonicalize().unwrap_or_else(|_| c.clone());
        }
    }
    log::warn!(target: "dlss", "the DLSS runtime libraries weren't found (looked in {}); set VF_DLSS_LIBS",
               cands.iter().map(|c| c.display().to_string()).collect::<Vec<_>>().join(", "));
    cands.last().cloned().unwrap_or_default()
}

fn img(g: &GImg, w: u32, h: u32) -> VfNgxImg {
    VfNgxImg { image: g.image.as_raw(), view: g.view.as_raw(), format: g.format.as_raw(), width: w, height: h }
}

/// NGX + the Ray Reconstruction (RR) and Super Resolution / DLAA (SR) features.
pub struct Dlss {
    pub rr_available: bool,
    pub sr_available: bool,
    params: *mut c_void,
    device: vk::Device,
    rr: *mut c_void,
    sr: *mut c_void,
    /// Render (input) and target (output) sizes of each feature.
    rr_size: [(u32, u32); 2],
    sr_size: [(u32, u32); 2],
}

impl Dlss {
    pub fn init(ctx: &Context) -> Self {
        let mut d = Self {
            rr_available: false,
            sr_available: false,
            params: std::ptr::null_mut(),
            device: ctx.device.handle(),
            rr: std::ptr::null_mut(),
            sr: std::ptr::null_mut(),
            rr_size: [(0, 0); 2],
            sr_size: [(0, 0); 2],
        };
        if std::env::var("VF_NO_DLSS").is_ok_and(|v| !v.is_empty() && v != "0") {
            log::info!(target: "dlss", "off (VF_NO_DLSS)");
            return d;
        }
        let snip = snippet_dir();
        let data = crate::paths::data_path("logs/ngx");
        let _ = std::fs::create_dir_all(&data);
        let (snip_c, data_c) = (CString::new(snip.to_string_lossy().as_bytes()).unwrap_or_default(),
                                CString::new(data.to_string_lossy().as_bytes()).unwrap_or_default());
        let gipa = ctx.entry.static_fn().get_instance_proc_addr as *const c_void;
        let gdpa = ctx.instance.fp_v1_0().get_device_proc_addr as *const c_void;
        let (mut rr, mut sr) = (0, 0);
        let r = unsafe {
            vf_ngx_init(PROJECT_ID.as_ptr(), snip_c.as_ptr(), data_c.as_ptr(), ctx.instance.handle().as_raw(), ctx.phys.as_raw(),
                        ctx.device.handle().as_raw(), gipa, gdpa, &mut d.params, &mut rr, &mut sr)
        };
        if failed(r) {
            log::warn!(target: "dlss", "NGX init failed: {r:#010x} (not an RTX GPU / driver, or the runtime libraries are missing) - \
                                        Ray Reconstruction / DLAA unavailable");
            return d;
        }
        d.rr_available = rr != 0;
        d.sr_available = sr != 0;
        log::info!(target: "dlss", "NGX ready ({}): Ray Reconstruction {}, Super Resolution / DLAA {}", snip.display(),
                   if d.rr_available { "yes" } else { "no" }, if d.sr_available { "yes" } else { "no" });
        d
    }

    /// The render size for a display size and quality (DLAA: the display size).
    pub fn optimal_render_size(&self, q: Quality, tw: u32, th: u32) -> (u32, u32) {
        if q == Quality::Dlaa || self.params.is_null() || !(self.rr_available || self.sr_available) {
            return (tw, th);
        }
        let (mut rw, mut rh) = (0, 0);
        let r = unsafe { vf_ngx_optimal(self.params, self.rr_available as c_int, tw, th, q as c_int, &mut rw, &mut rh) };
        if failed(r) || rw == 0 || rh == 0 { (tw, th) } else { (rw, rh) }
    }

    /// Create the Ray Reconstruction feature, render `r` -> target `t`. Records into `cb`.
    pub fn create_rr(&mut self, _ctx: &Context, cb: vk::CommandBuffer, r: (u32, u32), t: (u32, u32), q: Quality) -> bool {
        self.release_rr();
        if !self.rr_available {
            return false;
        }
        let mut vram = 0;
        let res = unsafe { vf_ngx_create_rr(self.device.as_raw(), cb.as_raw(), self.params, r.0, r.1, t.0, t.1, q as c_int, &mut self.rr, &mut vram) };
        if failed(res) {
            log::error!(target: "dlss", "creating Ray Reconstruction failed: {res:#010x}");
            self.rr = std::ptr::null_mut();
            return false;
        }
        self.rr_size = [r, t];
        log::info!(target: "dlss", "Ray Reconstruction: {} render {}x{} -> {}x{} ({:.1} MB)", q.name(), r.0, r.1, t.0, t.1, vram as f64 / 1048576.0);
        true
    }

    /// Create the Super Resolution feature (DLAA when r == t).
    pub fn create_sr(&mut self, _ctx: &Context, cb: vk::CommandBuffer, r: (u32, u32), t: (u32, u32), q: Quality) -> bool {
        self.release_sr();
        if !self.sr_available {
            return false;
        }
        let q = if r == t { Quality::Dlaa } else { q };
        let mut vram = 0;
        let res = unsafe { vf_ngx_create_sr(self.device.as_raw(), cb.as_raw(), self.params, r.0, r.1, t.0, t.1, q as c_int, &mut self.sr, &mut vram) };
        if failed(res) {
            log::error!(target: "dlss", "creating Super Resolution failed: {res:#010x}");
            self.sr = std::ptr::null_mut();
            return false;
        }
        self.sr_size = [r, t];
        log::info!(target: "dlss", "{}: render {}x{} -> {}x{} ({:.1} MB)", if r == t { "DLAA" } else { "Super Resolution" }, r.0, r.1, t.0, t.1,
                   vram as f64 / 1048576.0);
        true
    }

    /// One frame of Ray Reconstruction: noisy HDR `color` + guides -> `out` (display size).
    /// Motion is UV-space (scaled to pixels here); `normal_rough` packs roughness in .w.
    #[allow(clippy::too_many_arguments)]
    pub fn evaluate_rr(&self, cb: vk::CommandBuffer, color: GImg, diff: GImg, spec: GImg, normal_rough: GImg, depth: GImg, motion: GImg, out: GImg,
                       reset: bool, jitter: [f32; 2]) {
        if self.rr.is_null() {
            return;
        }
        let [(w, h), (tw, th)] = self.rr_size;
        let r = unsafe {
            vf_ngx_eval_rr(cb.as_raw(), self.rr, self.params, &img(&color, w, h), &img(&diff, w, h), &img(&spec, w, h), &img(&normal_rough, w, h),
                           &img(&depth, w, h), &img(&motion, w, h), &img(&out, tw, th), reset as c_int, jitter[0], jitter[1], w as f32, h as f32)
        };
        if failed(r) {
            log::error!(target: "dlss", "Ray Reconstruction evaluate failed: {r:#010x}");
        }
    }

    /// One frame of Super Resolution / DLAA: `color` (+ reverse-Z depth, UV motion) -> `out`.
    pub fn evaluate_sr(&self, cb: vk::CommandBuffer, color: GImg, hw_depth: GImg, motion: GImg, out: GImg, jitter: [f32; 2], reset: bool) {
        if self.sr.is_null() {
            return;
        }
        let [(w, h), (tw, th)] = self.sr_size;
        let r = unsafe {
            vf_ngx_eval_sr(cb.as_raw(), self.sr, self.params, &img(&color, w, h), &img(&hw_depth, w, h), &img(&motion, w, h), &img(&out, tw, th),
                           reset as c_int, jitter[0], jitter[1], w as f32, h as f32)
        };
        if failed(r) {
            log::error!(target: "dlss", "Super Resolution evaluate failed: {r:#010x}");
        }
    }

    /// Release a feature (the device must be idle): a new one can then be made at another size.
    pub fn release_rr(&mut self) {
        if !self.rr.is_null() {
            unsafe { vf_ngx_release(self.rr) };
            self.rr = std::ptr::null_mut();
        }
    }
    pub fn release_sr(&mut self) {
        if !self.sr.is_null() {
            unsafe { vf_ngx_release(self.sr) };
            self.sr = std::ptr::null_mut();
        }
    }
    pub fn rr_active(&self) -> bool {
        !self.rr.is_null()
    }
    pub fn sr_active(&self) -> bool {
        !self.sr.is_null()
    }

    pub fn destroy(&mut self, _ctx: &Context) {
        self.release_rr();
        self.release_sr();
        if !self.params.is_null() {
            unsafe { vf_ngx_shutdown(self.device.as_raw()) };
            self.params = std::ptr::null_mut();
        }
    }
}
