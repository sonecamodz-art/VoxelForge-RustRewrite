//! Post-processing after the HDR resolve: auto exposure (eye adaptation), bloom (a 6-level
//! mip chain) and the tonemap (ACES / AgX + saturation / contrast / white balance) into the
//! renderer's out_tex. Inputs: 0 = dlss_out, 1 = gb_color_hdr, 2 = dlss_out2.

use ash::vk;
use bytemuck::{Pod, Zeroable};

use super::context::{clear_color_f, pass_barrier, to_general, vkc, Context, Image};
use super::renderer::frame::warmth_balance;
use super::renderer::{Renderer, SetWriter};
use super::shader::{self, ComputePipeline};
use crate::cvars;

const INPUTS: usize = 3;
const LEVELS: usize = 6;

#[repr(C)]
#[derive(Clone, Copy, Pod, Zeroable)]
struct TonePc {
    exposure: f32,
    x_min: i32,
    x_max: i32,
    divider: i32,
    op: i32,
    sat: f32,
    con: f32,
    bloom: f32,
    auto_expo: i32,
    wb: [f32; 3],
    uw: [f32; 4],
}

#[repr(C)]
#[derive(Clone, Copy, Pod, Zeroable)]
struct BloomPc {
    mode: i32,
    radius: f32,
    threshold: f32,
}

#[repr(C)]
#[derive(Clone, Copy, Pod, Zeroable)]
struct ExpoPc {
    dt: f32,
    speed: f32,
    key: f32,
    reset: i32,
    min_f: f32,
    max_f: f32,
    pad: [f32; 2],
}

pub struct PostFx {
    expo: Image,
    down: Vec<Image>,
    up: Vec<Image>,
    bw: u32,
    bh: u32,
    p_bloom: ComputePipeline,
    p_expo: ComputePipeline,
    p_tone: ComputePipeline,
    l_bloom: vk::DescriptorSetLayout,
    l_expo: vk::DescriptorSetLayout,
    l_tone: vk::DescriptorSetLayout,
    pool: vk::DescriptorPool,
    s_down0: [vk::DescriptorSet; INPUTS],
    s_down: [vk::DescriptorSet; LEVELS],
    s_up: [vk::DescriptorSet; LEVELS - 1],
    s_expo: [vk::DescriptorSet; INPUTS],
    s_tone: [vk::DescriptorSet; INPUTS],
    lin: vk::Sampler,
    expo_reset: bool,
    /// The eye's depth under water (voxels), a clock and the time since it crossed the surface.
    pub underwater_depth: f32,
    pub underwater_time: f32,
    pub underwater_since: f32,
}

impl PostFx {
    pub fn new(ctx: &Context, r: &Renderer) -> Self {
        use vk::DescriptorType as DT;
        let dev = &ctx.device;
        let (cis, si) = (DT::COMBINED_IMAGE_SAMPLER, DT::STORAGE_IMAGE);
        let l_bloom = shader::make_set_layout(dev, &[(0, cis), (1, cis), (2, si)]);
        let l_expo = shader::make_set_layout(dev, &[(0, cis), (1, si)]);
        let l_tone = shader::make_set_layout(dev, &[(0, cis), (1, si), (2, cis), (3, si)]);
        let compile = |parts: &[&str]| shader::compile_parts(parts, "").unwrap_or_else(|e| crate::log::fatal("shader", &e));
        let pc = r.pcache.cache;
        let p_bloom = shader::make_compute_pipeline(dev, pc, &compile(&["bloom.comp"]), &[l_bloom], std::mem::size_of::<BloomPc>() as u32);
        let p_expo = shader::make_compute_pipeline(dev, pc, &compile(&["exposure.comp"]), &[l_expo], std::mem::size_of::<ExpoPc>() as u32);
        let p_tone = shader::make_compute_pipeline(dev, pc, &compile(&["tonemap.glsl", "tonemap.comp"]), &[l_tone], std::mem::size_of::<TonePc>() as u32);
        let n_sets = (INPUTS * 3 + LEVELS + LEVELS - 1) as u32;
        let ps = [vk::DescriptorPoolSize { ty: cis, descriptor_count: n_sets * 2 }, vk::DescriptorPoolSize { ty: si, descriptor_count: n_sets * 2 }];
        let pool = vkc(unsafe { dev.create_descriptor_pool(&vk::DescriptorPoolCreateInfo::default().max_sets(n_sets).pool_sizes(&ps), None) }, "post pool");
        let alloc = |l| {
            let ls = [l];
            vkc(unsafe { dev.allocate_descriptor_sets(&vk::DescriptorSetAllocateInfo::default().descriptor_pool(pool).set_layouts(&ls)) }, "post set")[0]
        };
        let mut p = PostFx {
            expo: ctx.image_2d("autoExposure", vk::Format::R32G32_SFLOAT, 1, 1),
            down: Vec::new(),
            up: Vec::new(),
            bw: 0,
            bh: 0,
            p_bloom,
            p_expo,
            p_tone,
            l_bloom,
            l_expo,
            l_tone,
            pool,
            s_down0: [alloc(l_bloom), alloc(l_bloom), alloc(l_bloom)],
            s_down: [vk::DescriptorSet::null(); LEVELS],
            s_up: [vk::DescriptorSet::null(); LEVELS - 1],
            s_expo: [alloc(l_expo), alloc(l_expo), alloc(l_expo)],
            s_tone: [alloc(l_tone), alloc(l_tone), alloc(l_tone)],
            lin: ctx.sampler(vk::Filter::LINEAR),
            expo_reset: true,
            underwater_depth: 0.0,
            underwater_time: 0.0,
            underwater_since: 100.0,
        };
        for i in 1..LEVELS {
            p.s_down[i] = alloc(l_bloom);
        }
        for i in 0..LEVELS - 1 {
            p.s_up[i] = alloc(l_bloom);
        }
        p.alloc_chain(ctx, r);
        p
    }

    fn alloc_chain(&mut self, ctx: &Context, r: &Renderer) {
        self.bw = (r.ow / 2).max(1);
        self.bh = (r.oh / 2).max(1);
        let f16 = vk::Format::R16G16B16A16_SFLOAT;
        self.down = (0..LEVELS).map(|i| ctx.image_2d("bloomDown", f16, (self.bw >> i).max(1), (self.bh >> i).max(1))).collect();
        self.up = (0..LEVELS - 1).map(|i| ctx.image_2d("bloomUp", f16, (self.bw >> i).max(1), (self.bh >> i).max(1))).collect();
        let dev = &ctx.device;
        let cb = ctx.begin_one_shot();
        let all: Vec<vk::Image> = std::iter::once(&self.expo).chain(self.down.iter()).chain(self.up.iter()).map(|i| i.image).collect();
        to_general(dev, cb, &all);
        for &im in &all {
            clear_color_f(dev, cb, im, [0.0; 4]);
        }
        ctx.end_one_shot_and_wait(cb);
        self.expo_reset = true;
        self.write_sets(ctx, r);
    }

    fn write_sets(&self, ctx: &Context, r: &Renderer) {
        let mut w = SetWriter::default();
        let lin = self.lin;
        for i in 0..INPUTS {
            let input = [r.t.dlss_out.view, r.t.gb_color_hdr.view, r.t.dlss_out2.view][i];
            w.tex(self.s_down0[i], 0, input, lin);
            w.tex(self.s_down0[i], 1, input, lin);
            w.img(self.s_down0[i], 2, self.down[0].view);
            w.tex(self.s_expo[i], 0, input, lin);
            w.img(self.s_expo[i], 1, self.expo.view);
            w.tex(self.s_tone[i], 0, input, lin);
            w.img(self.s_tone[i], 1, r.t.out_tex.view);
            w.tex(self.s_tone[i], 2, self.up[0].view, lin);
            w.img(self.s_tone[i], 3, self.expo.view);
        }
        for i in 1..LEVELS {
            w.tex(self.s_down[i], 0, self.down[i - 1].view, lin);
            w.tex(self.s_down[i], 1, self.down[i - 1].view, lin);
            w.img(self.s_down[i], 2, self.down[i].view);
        }
        for i in 0..LEVELS - 1 {
            let src = if i == LEVELS - 2 { self.down[LEVELS - 1].view } else { self.up[i + 1].view };
            w.tex(self.s_up[i], 0, src, lin);
            w.tex(self.s_up[i], 1, self.down[i].view, lin);
            w.img(self.s_up[i], 2, self.up[i].view);
        }
        w.flush(&ctx.device);
    }

    /// After Renderer::resize: rebuild the bloom chain and re-point every set.
    pub fn resize(&mut self, ctx: &Context, r: &Renderer) {
        ctx.wait_idle();
        for im in self.down.iter_mut().chain(self.up.iter_mut()) {
            im.destroy(ctx);
        }
        self.alloc_chain(ctx, r);
    }

    pub fn bloom_on(&self) -> bool {
        cvars::R_BLOOM.get() && cvars::R_BLOOM_STRENGTH.get() > 0.0
    }

    fn dispatch<T: Pod>(&self, dev: &ash::Device, cb: vk::CommandBuffer, p: &ComputePipeline, s: vk::DescriptorSet, pc: &T, gx: u32, gy: u32) {
        p.bind(dev, cb, &[s]);
        p.push(dev, cb, pc);
        unsafe { dev.cmd_dispatch(cb, gx, gy, 1) };
    }

    /// Exposure + bloom for HDR input `input` (no-ops when disabled). dt in seconds.
    pub fn prepare(&mut self, dev: &ash::Device, cb: vk::CommandBuffer, input: usize, dt: f32) {
        let mut any = false;
        if cvars::R_AUTOEXPOSURE.get() {
            let pc = ExpoPc {
                dt: dt.clamp(0.0, 0.25),
                speed: cvars::R_AUTOEXPOSURE_SPEED.get(),
                key: cvars::R_AUTOEXPOSURE_KEY.get(),
                reset: self.expo_reset as i32,
                min_f: cvars::R_AUTOEXPOSURE_MIN.get().min(cvars::R_AUTOEXPOSURE_MAX.get()),
                max_f: cvars::R_AUTOEXPOSURE_MAX.get(),
                pad: [0.0; 2],
            };
            self.dispatch(dev, cb, &self.p_expo, self.s_expo[input], &pc, 1, 1);
            self.expo_reset = false;
            any = true;
        } else {
            self.expo_reset = true; // re-adapt instantly when turned on again
        }
        if self.bloom_on() {
            let threshold = 1.25 / cvars::R_EXPOSURE.get().max(0.05);
            if any {
                pass_barrier(dev, cb);
            }
            for i in 0..LEVELS {
                let pc = BloomPc { mode: if i == 0 { 0 } else { 1 }, radius: 1.0, threshold };
                let (w, h) = ((self.bw >> i).max(1), (self.bh >> i).max(1));
                let s = if i == 0 { self.s_down0[input] } else { self.s_down[i] };
                self.dispatch(dev, cb, &self.p_bloom, s, &pc, w.div_ceil(8), h.div_ceil(8));
                pass_barrier(dev, cb);
            }
            for i in (0..LEVELS - 1).rev() {
                let pc = BloomPc { mode: 2, radius: 1.0, threshold: 0.0 };
                let (w, h) = ((self.bw >> i).max(1), (self.bh >> i).max(1));
                self.dispatch(dev, cb, &self.p_bloom, self.s_up[i], &pc, w.div_ceil(8), h.div_ceil(8));
                pass_barrier(dev, cb);
            }
            any = true;
        }
        if any {
            pass_barrier(dev, cb);
        }
    }

    /// HDR input -> out_tex over dw x dh (x_min / x_max / divider: split-screen compare).
    #[allow(clippy::too_many_arguments)]
    pub fn tonemap(&self, dev: &ash::Device, cb: vk::CommandBuffer, input: usize, dw: u32, dh: u32, exposure: f32, x_min: i32, x_max: i32, divider: i32) {
        // overcast and rain: soft light, less saturated colour
        let mut grey = ((cvars::R_CLOUDS_COVERAGE.get() - 0.55) / 0.4).clamp(0.0, 1.0);
        grey = grey.max(cvars::R_RAIN.get());
        if !cvars::R_CLOUDS.get() {
            grey = 0.0;
        }
        let pc = TonePc {
            exposure,
            x_min,
            x_max,
            divider,
            op: cvars::R_TONEMAP.is("agx") as i32,
            sat: cvars::R_SATURATION.get() * (1.0 - 0.22 * grey),
            con: cvars::R_CONTRAST.get(),
            bloom: if self.bloom_on() { cvars::R_BLOOM_STRENGTH.get() } else { 0.0 },
            auto_expo: cvars::R_AUTOEXPOSURE.get() as i32,
            wb: warmth_balance(cvars::R_WARMTH.get()).to_array(),
            uw: [self.underwater_depth, self.underwater_time, self.underwater_since, cvars::R_UNDERWATER.get()],
        };
        self.dispatch(dev, cb, &self.p_tone, self.s_tone[input], &pc, dw.div_ceil(8), dh.div_ceil(8));
    }

    pub fn destroy(&mut self, ctx: &Context) {
        ctx.wait_idle();
        let dev = &ctx.device;
        for im in self.down.iter_mut().chain(self.up.iter_mut()) {
            im.destroy(ctx);
        }
        self.expo.destroy(ctx);
        for p in [&mut self.p_bloom, &mut self.p_expo, &mut self.p_tone] {
            p.destroy(dev);
        }
        unsafe {
            dev.destroy_descriptor_pool(self.pool, None);
            for l in [self.l_bloom, self.l_expo, self.l_tone] {
                dev.destroy_descriptor_set_layout(l, None);
            }
            dev.destroy_sampler(self.lin, None);
        }
    }
}
