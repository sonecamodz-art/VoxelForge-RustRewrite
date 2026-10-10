//! Which lighting / denoiser / resolve runs, at what resolution: one place for the window and
//! the headless shots.
//!
//!   r.denoiser rr    DLSS Ray Reconstruction: the raw 1-spp radiance + guides -> NGX (denoise +
//!                    upscale to the display) -> tonemap
//!   r.denoiser svgf  the in-house SVGF chain; with r.aa dlaa resolved by DLSS Super Resolution
//!                    (DLAA at r.dlss.quality 0, real upscaling below that), else tonemapped by
//!                    denoise.comp at the render resolution (r.scale; the blit upscales)
//!   r.denoiser nrd   NVIDIA NRD (REBLUR + SIGMA) on RR's signals, then like svgf
//!   r.denoiser none  the raw 1-spp radiance (debug)
//!   r.lighting standard: no noise, never RR; DLAA / SR resolve like svgf
//! What isn't available (no DLSS SDK / RTX GPU, no NRD) falls back to svgf, r.aa to off.

use ash::vk;

use crate::cvars;
use crate::gfx::context::{memory_barrier, Access, Context, Stage};
use crate::gfx::dlss::{Dlss, GImg, Quality};
use crate::gfx::postfx::PostFx;
use crate::gfx::renderer::record::Rec;
use crate::gfx::renderer::Renderer;

#[derive(Clone, Copy, PartialEq, Eq, Debug)]
pub enum Denoiser {
    Rr,
    Svgf,
    Nrd,
    None,
}

impl Denoiser {
    pub fn name(self) -> &'static str {
        match self {
            Self::Rr => "DLSS Ray Reconstruction",
            Self::Svgf => "SVGF",
            Self::Nrd => "NVIDIA NRD",
            Self::None => "none (raw)",
        }
    }
}

/// Render (rw x rh) and output (ow x oh) sizes, and which DLSS feature resolves.
#[derive(Clone, Copy, PartialEq, Eq, Debug, Default)]
pub struct ResCfg {
    pub rw: u32,
    pub rh: u32,
    pub ow: u32,
    pub oh: u32,
    pub rr: bool,
    pub sr: bool,
}

pub struct RenderPath {
    pub dlss: Dlss,
    /// NGX's render sizes, by (quality, display size): asked once, not every frame.
    optimal: std::cell::RefCell<Vec<((Quality, u32, u32), (u32, u32))>>,
    pub cfg: ResCfg,
    quality: Quality,
    /// DLSS history is stale (first frame, a feature re-made, a world loaded).
    reset: bool,
}

/// Halton(2, 3) sub-pixel offsets in [-0.5, 0.5), a 32-frame cycle (index 0 = 0,0 skipped).
pub fn halton_jitter(frame: u32) -> [f32; 2] {
    fn halton(mut i: u32, b: u32) -> f32 {
        let (mut f, mut r) = (1.0f32, 0.0f32);
        while i > 0 {
            f /= b as f32;
            r += f * (i % b) as f32;
            i /= b;
        }
        r
    }
    let i = frame % 32 + 1;
    [halton(i, 2) - 0.5, halton(i, 3) - 0.5]
}

fn gimg(im: &crate::gfx::context::Image) -> GImg {
    GImg { image: im.image, view: im.view, format: im.format }
}

impl RenderPath {
    pub fn new(ctx: &Context) -> Self {
        let dlss = Dlss::init(ctx);
        Self { dlss, optimal: Default::default(), cfg: ResCfg::default(), quality: Quality::from_index(cvars::R_DLSS_QUALITY.get()), reset: true }
    }

    /// What a r.denoiser value asks for.
    pub fn denoiser_for(v: &str) -> Denoiser {
        match v {
            "rr" => Denoiser::Rr,
            "nrd" => Denoiser::Nrd,
            "none" => Denoiser::None,
            _ => Denoiser::Svgf,
        }
    }

    /// The denoiser that runs: r.denoiser, or svgf where that isn't available.
    pub fn denoiser(&self, r: Option<&Renderer>) -> Denoiser {
        match cvars::R_DENOISER.get().as_str() {
            "rr" if self.dlss.rr_available => Denoiser::Rr,
            "nrd" if r.is_some_and(|r| r.nrd_available()) => Denoiser::Nrd,
            "none" => Denoiser::None,
            _ => Denoiser::Svgf,
        }
    }

    /// The sizes for a display of tw x th under the current settings.
    pub fn want(&self, r: Option<&Renderer>, tw: u32, th: u32) -> ResCfg {
        let standard = cvars::R_LIGHTING.is("standard");
        let d = self.denoiser(r);
        let rr = !standard && d == Denoiser::Rr;
        let sr = !rr && self.dlss.sr_available && (standard || matches!(d, Denoiser::Svgf | Denoiser::Nrd)) && cvars::R_AA.is("dlaa");
        if rr || sr {
            let q = Quality::from_index(cvars::R_DLSS_QUALITY.get());
            let key = (q, tw, th);
            let hit = self.optimal.borrow().iter().find(|e| e.0 == key).map(|e| e.1);
            let (rw, rh) = hit.unwrap_or_else(|| {
                let v = self.dlss.optimal_render_size(q, tw, th);
                self.optimal.borrow_mut().push((key, v));
                v
            });
            ResCfg { rw, rh, ow: tw, oh: th, rr, sr }
        } else {
            let s = cvars::R_SCALE.get();
            let (rw, rh) = (((tw as f32 * s) as u32).max(64), ((th as f32 * s) as u32).max(64));
            ResCfg { rw, rh, ow: rw, oh: rh, rr: false, sr: false }
        }
    }

    /// Make the DLSS feature `cfg` needs (records + waits). False: it failed (the setting that
    /// asked for it is switched off).
    fn make_feature(&mut self, ctx: &Context) -> bool {
        let c = self.cfg;
        if !c.rr && !c.sr {
            return true;
        }
        let cb = ctx.begin_one_shot();
        let ok = if c.rr {
            self.dlss.create_rr(ctx, cb, (c.rw, c.rh), (c.ow, c.oh), self.quality)
        } else {
            self.dlss.create_sr(ctx, cb, (c.rw, c.rh), (c.ow, c.oh), self.quality)
        };
        ctx.end_one_shot_and_wait(cb);
        if !ok && c.rr {
            log::error!(target: "dlss", "Ray Reconstruction unavailable - r.denoiser svgf");
            cvars::R_DENOISER.set("svgf");
        } else if !ok {
            log::error!(target: "dlss", "DLAA / Super Resolution unavailable - r.aa off");
            cvars::R_AA.set("off");
        }
        ok
    }

    /// The first configuration (before the renderer exists).
    pub fn begin(&mut self, ctx: &Context, tw: u32, th: u32) -> ResCfg {
        self.quality = Quality::from_index(cvars::R_DLSS_QUALITY.get());
        self.cfg = self.want(None, tw, th);
        if !self.make_feature(ctx) {
            self.cfg = self.want(None, tw, th);
        }
        self.reset = true;
        self.cfg
    }

    /// Per frame: a mode / quality / scale / display change retargets the renderer and the
    /// DLSS features. True when it did.
    pub fn retarget(&mut self, ctx: &Context, r: &mut Renderer, post: &mut PostFx, tw: u32, th: u32) -> bool {
        let q = Quality::from_index(cvars::R_DLSS_QUALITY.get());
        let want = self.want(Some(r), tw, th);
        let feature_ok = (!want.rr || self.dlss.rr_active()) && (!want.sr || self.dlss.sr_active());
        if want == self.cfg && q == self.quality && feature_ok {
            return false;
        }
        ctx.wait_idle();
        self.quality = q;
        self.cfg = want;
        self.dlss.release_rr(); // (frees RR's memory when leaving it)
        self.dlss.release_sr();
        if !self.make_feature(ctx) {
            self.cfg = self.want(Some(r), tw, th);
        }
        let c = self.cfg;
        if (c.rw, c.rh, c.ow, c.oh) != (r.w, r.h, r.ow, r.oh) {
            r.resize(ctx, c.rw, c.rh, c.ow, c.oh);
            post.resize(ctx, r);
        }
        self.reset = true;
        log::info!(target: "app", "render {}x{} -> {}x{}  {}{}", c.rw, c.rh, c.ow, c.oh,
                   if cvars::R_LIGHTING.is("standard") { "Standard lighting" } else { self.denoiser(Some(r)).name() },
                   if c.rr { format!(" ({})", self.quality.name()) } else if c.sr { if c.rw == c.ow { " + DLAA".into() } else { format!(" + DLSS {}", self.quality.name()) } } else { String::new() });
        true
    }

    /// History is stale (a world loaded, the camera jumped).
    pub fn reset_history(&mut self) {
        self.reset = true;
    }

    /// This frame's sub-pixel jitter: only when a temporal resolve consumes it (without one it
    /// would just shimmer).
    pub fn jitter(&self, frame: u32) -> [f32; 2] {
        if self.cfg.sr || (self.cfg.rr && cvars::R_RR_JITTER.get()) { halton_jitter(frame) } else { [0.0, 0.0] }
    }

    /// A one-line description for the overlay / title.
    pub fn describe(&self, r: &Renderer) -> String {
        let c = self.cfg;
        let mode = if cvars::R_LIGHTING.is("standard") { "standard".to_string() } else { self.denoiser(Some(r)).name().to_string() };
        if c.rr {
            format!("{mode} {}", self.quality.name())
        } else if c.sr {
            format!("{mode} + {}", if c.rw == c.ow { "DLAA".to_string() } else { format!("DLSS {}", self.quality.name()) })
        } else {
            mode
        }
    }

    /// Record the frame's lighting, denoise and resolve into out_tex. Returns the region of
    /// out_tex that holds the image (what the blit / the overlays use).
    pub fn record(&mut self, rec: &Rec, r: &mut Renderer, post: &mut PostFx, dt: f32, exposure: f32, frame: u32) -> (u32, u32) {
        let dev = &rec.ctx.device;
        let cb = rec.cb;
        let c = self.cfg;
        let jit = self.jitter(frame);
        let ngx_jitter = [jit[0] * cvars::R_AA_JITTER_SX.get(), jit[1] * cvars::R_AA_JITTER_SY.get()];
        // The resolve reads what the passes wrote; the tonemap what the resolve wrote.
        let to_ngx = |cb: vk::CommandBuffer| {
            memory_barrier(dev, cb, Stage::COMPUTE_SHADER, Access::SHADER_STORAGE_WRITE, Stage::ALL_COMMANDS, Access::MEMORY_READ | Access::MEMORY_WRITE)
        };
        let from_ngx = |cb: vk::CommandBuffer| {
            memory_barrier(dev, cb, Stage::ALL_COMMANDS, Access::MEMORY_WRITE, Stage::COMPUTE_SHADER, Access::SHADER_SAMPLED_READ | Access::SHADER_STORAGE_READ)
        };
        let tonemap = |r: &mut Renderer, post: &mut PostFx, input: usize, w: u32, h: u32| {
            r.record_torch_fx(rec, input, w, h); // torch flames + embers into the HDR first (bloom picks them up)
            post.prepare(dev, cb, input, dt);
            post.tonemap(dev, cb, input, w, h, exposure, 0, 1 << 30, -1);
            r.stamp(rec, "tonemap");
        };
        let sr = |s: &mut Self, r: &mut Renderer, post: &mut PostFx| {
            to_ngx(cb);
            let t = &r.t;
            s.dlss.evaluate_sr(cb, gimg(&t.gb_color_hdr), gimg(&t.gb_depth_hw), gimg(&t.gb_motion), gimg(&t.dlss_out), ngx_jitter, s.reset);
            s.reset = false;
            from_ngx(cb);
            r.stamp(rec, if c.rw == c.ow { "DLAA" } else { "DLSS-SR" });
            tonemap(r, post, 0, c.ow, c.oh);
        };
        let d = self.denoiser(Some(r));
        if r.standard_mode() {
            r.record_standard(rec);
            crate::gfx::context::pass_barrier(dev, cb);
            if c.sr {
                sr(self, r, post);
                return (c.ow, c.oh);
            }
            tonemap(r, post, 1, r.w, r.h);
            return (r.w, r.h);
        }
        match d {
            Denoiser::Rr if c.rr => {
                r.record_raw(rec); // raw 1-spp -> guides + the noisy colour
                to_ngx(cb);
                let t = &r.t;
                self.dlss.evaluate_rr(cb, gimg(&t.gb_color_hdr), gimg(&t.gb_diff_albedo), gimg(&t.gb_spec_albedo), gimg(&t.gb_normal_rough),
                                      gimg(&t.gb_linear_depth), gimg(&t.gb_motion), gimg(&t.dlss_out), self.reset, ngx_jitter);
                self.reset = false;
                from_ngx(cb);
                r.stamp(rec, "DLSS-RR");
                tonemap(r, post, 0, c.ow, c.oh);
                (c.ow, c.oh)
            }
            Denoiser::Nrd => {
                r.record_nrd(rec, frame); // NRD -> gb_color_hdr
                crate::gfx::context::pass_barrier(dev, cb);
                if c.sr {
                    sr(self, r, post);
                    return (c.ow, c.oh);
                }
                tonemap(r, post, 1, r.w, r.h);
                (r.w, r.h)
            }
            Denoiser::None => {
                r.record_raw(rec);
                crate::gfx::context::pass_barrier(dev, cb);
                tonemap(r, post, 1, r.w, r.h);
                (r.w, r.h)
            }
            _ => {
                r.record(rec); // the SVGF chain (-> out_tex, and HDR -> gb_color_hdr)
                if c.sr {
                    sr(self, r, post);
                    return (c.ow, c.oh);
                }
                (r.w, r.h)
            }
        }
    }

    pub fn destroy(&mut self, ctx: &Context) {
        self.dlss.destroy(ctx);
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn halton_cycle() {
        let j: Vec<[f32; 2]> = (0..32).map(halton_jitter).collect();
        assert_eq!(j[0], [0.0, 1.0 / 3.0 - 0.5]); // index 1: 1/2, 1/3
        assert!(j.iter().all(|p| p.iter().all(|v| (-0.5..0.5).contains(v))));
        let mean = j.iter().fold([0.0f32; 2], |a, p| [a[0] + p[0] / 32.0, a[1] + p[1] / 32.0]);
        assert!(mean[0].abs() < 0.05 && mean[1].abs() < 0.05, "{mean:?}");
        assert_eq!(halton_jitter(32), j[0]);
    }
}
