//! Command recording: the SVGF ray-traced chain, Standard lighting, the raw 1-spp path, and
//! the shared helpers (fog, clouds, water, grass wind, radiance cache, overlays).

use ash::vk;
use bytemuck::{Pod, Zeroable};

use super::passes::{ATROUS_SETS, P, S, TORCH_SETS};
use super::{Renderer, GI_CACHE_QUEUE_MAX, GI_CACHE_SIZE, LAMP_CUBE_RES, MAX_TS};
use crate::camera::FlyCamera;
use crate::cvars;
use crate::gfx::context::{image_barrier, memory_barrier, pass_barrier, Access, Context, Stage, COLOR_LAYERS};
use crate::math::Vec3;

/// Push constants of brush_preview.comp.
#[repr(C)]
#[derive(Clone, Copy, Pod, Zeroable)]
pub struct BrushPreviewPc {
    pub world_to_local: [f32; 16],
    /// corner xyz, size
    pub carve: [i32; 4],
    /// corner xyz, shape (0 sphere, 1 cube)
    pub place: [i32; 4],
    /// rgb, flags (1 on, 2 body, 4 carve, 8 place, 16 paint, 32 volume)
    pub place_col: [f32; 4],
    pub out_res: [i32; 4],
}
const _: () = assert!(std::mem::size_of::<BrushPreviewPc>() == 128);

static DBG_N: std::sync::atomic::AtomicU32 = std::sync::atomic::AtomicU32::new(0);
fn dbg_skip(label: &str) -> bool {
    if std::env::var("VF_SKIP").is_ok_and(|v| v.split(',').any(|s| s == label)) {
        return true;
    }
    let Ok(lim) = std::env::var("VF_MAXPASS") else { return false };
    let n = DBG_N.fetch_add(1, std::sync::atomic::Ordering::Relaxed);
    let skip = n >= lim.parse::<u32>().unwrap_or(u32::MAX);
    if !skip {
        eprintln!("pass {n}: {label}");
    }
    skip
}

/// A recording context: the device + command buffer + the bound frame slot's sets.
pub struct Rec<'a> {
    pub ctx: &'a Context,
    pub cb: vk::CommandBuffer,
}

impl Renderer {
    fn sets4(&self, s3: vk::DescriptorSet) -> [vk::DescriptorSet; 4] {
        let sl = &self.slots[self.cur];
        [self.s0, sl.s1, sl.s2, s3]
    }

    /// Bind pipeline `p` with set 3 = `s`, dispatch (gx, gy, gz).
    fn go(&self, r: &Rec, p: P, s: vk::DescriptorSet, gx: u32, gy: u32, gz: u32, label: &str) {
        if dbg_skip(label) {
            return;
        }
        r.ctx.begin_label(r.cb, label);
        self.pipe(p).bind(&r.ctx.device, r.cb, &self.sets4(s));
        unsafe { r.ctx.device.cmd_dispatch(r.cb, gx, gy, gz) };
        r.ctx.end_label(r.cb);
    }

    fn go_pc<T: Pod>(&self, r: &Rec, p: P, s: vk::DescriptorSet, pc: &T, gx: u32, gy: u32, gz: u32, label: &str) {
        if dbg_skip(label) {
            return;
        }
        r.ctx.begin_label(r.cb, label);
        let pipe = self.pipe(p);
        pipe.bind(&r.ctx.device, r.cb, &self.sets4(s));
        pipe.push(&r.ctx.device, r.cb, pc);
        unsafe { r.ctx.device.cmd_dispatch(r.cb, gx, gy, gz) };
        r.ctx.end_label(r.cb);
    }

    fn barrier(&self, r: &Rec) {
        pass_barrier(&r.ctx.device, r.cb);
    }

    // --- GPU timers ----------------------------------------------------------------------------
    pub fn begin_timers(&mut self, r: &Rec) {
        let sl = &mut self.slots[self.cur];
        unsafe {
            r.ctx.device.cmd_reset_query_pool(r.cb, sl.ts, 0, MAX_TS);
            r.ctx.device.cmd_write_timestamp2(r.cb, Stage::ALL_COMMANDS, sl.ts, 0);
        }
        sl.ts_count = 1;
        sl.names.clear();
    }

    /// End a timed segment.
    pub fn stamp(&mut self, r: &Rec, name: &'static str) {
        let sl = &mut self.slots[self.cur];
        if sl.ts_count == 0 || sl.ts_count >= MAX_TS {
            return;
        }
        sl.names.push(name);
        unsafe { r.ctx.device.cmd_write_timestamp2(r.cb, Stage::ALL_COMMANDS, sl.ts, sl.ts_count) };
        sl.ts_count += 1;
    }

    // --- shared passes ---------------------------------------------------------------------------
    /// Grass wind springs: advance by this frame's time step in substeps of <= 1/120 s,
    /// re-centred on the camera.
    fn record_grass_wind(&mut self, r: &Rec) {
        if cvars::R_GRASS.is("off") {
            return;
        }
        #[repr(C)]
        #[derive(Clone, Copy, Pod, Zeroable)]
        struct Pc {
            ox: i32,
            oz: i32,
            px: i32,
            pz: i32,
            t0: f32,
            h: f32,
            n: i32,
            reset: i32,
            strength: f32,
            gust: f32,
            speed: f32,
            dir: f32,
            set_strength: f32,
            recover: f32,
            pad: [f32; 2],
            stamp: [[f32; 4]; 4],
        }
        const _: () = assert!(std::mem::size_of::<Pc>() == 128);
        let org = [(self.wind_cam[0] / 2.0).floor() as i32 - 64, (self.wind_cam[1] / 2.0).floor() as i32 - 64];
        let n_steps = if self.wind_dt > 0.0 { ((self.wind_dt * 120.0 - 1e-3).ceil() as i32).clamp(1, 8) } else { 0 };
        let pc = Pc {
            ox: org[0],
            oz: org[1],
            px: self.wind_origin[0],
            pz: self.wind_origin[1],
            t0: self.last_time - self.wind_dt,
            h: if n_steps > 0 { self.wind_dt / n_steps as f32 } else { 0.0 },
            n: n_steps,
            reset: self.wind_reset as i32,
            strength: cvars::R_WIND_STRENGTH.get(),
            gust: cvars::R_WIND_GUST.get(),
            speed: cvars::R_WIND_SPEED.get(),
            dir: cvars::R_WIND_DIR.get().to_radians(),
            set_strength: cvars::R_WIND_SET.get(),
            recover: cvars::R_GRASS_RECOVER.get(),
            pad: [0.0; 2],
            stamp: self.grass_stamp,
        };
        self.barrier(r);
        self.go_pc(r, P::Wind, self.s_wind, &pc, 16, 16, 1, "grass_wind");
        self.barrier(r);
        self.wind_origin = org;
        self.wind_reset = false;
    }

    /// Screen-space penumbra over shadow_raw (separable H then V).
    fn record_shadow_soften(&self, r: &Rec, max_r: f32) {
        #[repr(C)]
        #[derive(Clone, Copy, Pod, Zeroable)]
        struct Pc {
            pass: i32,
            max_r: f32,
            min_w: f32,
            pad: i32,
        }
        let (gx, gy) = (self.w.div_ceil(8), self.h.div_ceil(8));
        self.go_pc(r, P::ShSoft, self.set(S::ShSoftH), &Pc { pass: 0, max_r, min_w: 0.04, pad: 0 }, gx, gy, 1, "shadow_soften_h");
        self.barrier(r);
        self.go_pc(r, P::ShSoft, self.set(S::ShSoftV), &Pc { pass: 1, max_r, min_w: 0.04, pad: 0 }, gx, gy, 1, "shadow_soften_v");
        self.barrier(r);
    }

    /// Torch flames + embers (+ rain drops) into an HDR image before its tonemap: `which`
    /// 0 dlss_out, 1 gb_color_hdr, 2 dlss_out2.
    pub fn record_torch_fx(&self, r: &Rec, which: usize, dw: u32, dh: u32) {
        if (self.lights_cpu.torch_count[0] <= 0 && cvars::R_RAIN.get() <= 0.001) || which > 2 {
            return;
        }
        self.go(r, P::TorchFx, self.set(TORCH_SETS[which]), dw.div_ceil(8), dh.div_ceil(8), 1, "torchfx");
        self.barrier(r);
    }

    fn record_clouds(&self, r: &Rec) {
        if !cvars::R_CLOUDS.get() {
            return;
        }
        let (gx, gy) = (self.w.div_ceil(2).div_ceil(8), self.h.div_ceil(2).div_ceil(8));
        self.go(r, P::Clouds, self.set(S::Clouds), gx, gy, 1, "clouds");
        self.barrier(r);
        self.go(r, P::CloudRes, self.set(S::CloudRes), gx, gy, 1, "clouds_resolve");
        self.barrier(r);
    }

    /// Water: the clouds it reflects, half-res reflections, then water_refl (no barrier after).
    fn record_water(&self, r: &Rec, stochastic: bool) {
        if cvars::R_CLOUDS.get() {
            self.go(r, P::CloudPano, self.set(S::CloudPano), 256 / 8, 128 / 8, 1, "clouds_pano");
            self.barrier(r);
        }
        if self.refl_half_on() {
            self.go(r, P::WaterReflHalf, self.set(S::WaterReflHalf), self.w.div_ceil(2).div_ceil(8), self.h.div_ceil(2).div_ceil(8), 1, "water_refl_half");
            self.barrier(r);
        }
        let p = if stochastic { P::WaterReflRR } else { P::WaterRefl };
        self.go(r, p, self.set(S::WaterRefl), self.w.div_ceil(8), self.h.div_ceil(8), 1, "water_refl");
    }

    /// Volumetric fog, part A: froxel tiles + injection (or the old quarter-res march).
    fn record_fog_a(&self, r: &Rec) {
        let [fx, fy, fz] = self.frox_dims;
        if cvars::R_FOG_METHOD.is("froxel") {
            self.go(r, P::FroxTile, self.set(S::FroxTile), fx.div_ceil(8), fy.div_ceil(8), 1, "froxel_tiles");
            self.barrier(r);
            self.go(r, P::FroxInj, self.set(S::FroxInj), fx.div_ceil(8), fy.div_ceil(8), fz, "froxel_inject");
        } else {
            self.go(r, P::Fog, self.set(S::Fog), self.qw.div_ceil(8), self.qh.div_ceil(8), 1, "fog");
        }
    }

    /// Volumetric fog, part B (after a barrier): froxel integration + its own barrier.
    fn record_fog_b(&self, r: &Rec) {
        if !cvars::R_FOG_METHOD.is("froxel") {
            return;
        }
        let [fx, fy, _] = self.frox_dims;
        self.go(r, P::FroxInt, self.set(S::FroxInt), fx.div_ceil(8), fy.div_ceil(8), 1, "froxel_integrate");
        self.barrier(r);
    }

    /// Radiance cache (r.gi.cache), once a frame: resolve last frame's samples, then one update
    /// ray per cell used (indirect dispatch; overlaps the gbuffer).
    fn record_gi_cache(&mut self, r: &Rec) {
        if !super::gi_cache_wanted() || self.gi_cache_buf.is_null() || self.gi_cache_frame == self.scene_cpu.gi_cache_p[0] as u32 {
            return;
        }
        self.gi_cache_frame = self.scene_cpu.gi_cache_p[0] as u32;
        let dev = &r.ctx.device;
        r.ctx.begin_label(r.cb, "gi_cache");
        if self.gi_cache_clear {
            unsafe {
                dev.cmd_fill_buffer(r.cb, self.gi_cache_buf.buf, 0, vk::WHOLE_SIZE, 0);
                dev.cmd_fill_buffer(r.cb, self.gi_queue_buf.buf, 0, vk::WHOLE_SIZE, 0);
            }
            self.gi_cache_clear = false;
            memory_barrier(dev, r.cb, Stage::CLEAR, Access::TRANSFER_WRITE, Stage::COMPUTE_SHADER, Access::SHADER_STORAGE_READ | Access::SHADER_STORAGE_WRITE);
        } else {
            memory_barrier(dev, r.cb, Stage::COMPUTE_SHADER, Access::SHADER_STORAGE_WRITE, Stage::COMPUTE_SHADER,
                           Access::SHADER_STORAGE_READ | Access::SHADER_STORAGE_WRITE);
        }
        #[repr(C)]
        #[derive(Clone, Copy, Pod, Zeroable)]
        struct Pc {
            lo: [[f32; 4]; 3],
            hi: [[f32; 4]; 3],
            n: i32,
            pad: [i32; 3],
        }
        const _: () = assert!(std::mem::size_of::<Pc>() == 112);
        let mut pc = Pc::zeroed();
        let g = cvars::R_GI_CACHE_EDITREACH.get() as f32;
        let put = |pc: &mut Pc, k: usize, b: &[i32; 6]| {
            for a in 0..3 {
                pc.lo[k][a] = b[a] as f32 - g;
                pc.hi[k][a] = b[a + 3] as f32 + g;
            }
        };
        let sd = Vec3::from_array(self.scene_cpu.sun_dir);
        if sd.dot(self.gi_cache_sun) < 0.99999 {
            self.gi_cache_reset_all = true;
        }
        self.gi_cache_sun = sd;
        if self.gi_cache_reset_all {
            pc.n = -1;
        } else if self.gi_cache_boxes.len() <= 3 {
            for (k, b) in self.gi_cache_boxes.iter().enumerate() {
                put(&mut pc, k, b);
            }
            pc.n = self.gi_cache_boxes.len() as i32;
        } else {
            let mut u = self.gi_cache_boxes[0];
            for b in &self.gi_cache_boxes {
                for a in 0..3 {
                    u[a] = u[a].min(b[a]);
                    u[a + 3] = u[a + 3].max(b[a + 3]);
                }
            }
            put(&mut pc, 0, &u);
            pc.n = 1;
        }
        self.gi_cache_boxes.clear();
        self.gi_cache_reset_all = false;
        let sl = &self.slots[self.cur];
        let sets = [self.s0, sl.s1, sl.s2];
        let res = self.pipe(P::GiCacheRes);
        res.bind(dev, r.cb, &sets);
        res.push(dev, r.cb, &pc);
        unsafe { dev.cmd_dispatch(r.cb, GI_CACHE_SIZE / 64, 1, 1) };
        memory_barrier(dev, r.cb, Stage::COMPUTE_SHADER, Access::SHADER_STORAGE_WRITE, Stage::COMPUTE_SHADER | Stage::DRAW_INDIRECT,
                       Access::SHADER_STORAGE_READ | Access::SHADER_STORAGE_WRITE | Access::INDIRECT_COMMAND_READ);
        self.pipe(P::GiCacheUpd).bind(dev, r.cb, &sets);
        unsafe { dev.cmd_dispatch_indirect(r.cb, self.gi_queue_buf.buf, 0) };
        r.ctx.end_label(r.cb);
        let _ = GI_CACHE_QUEUE_MAX;
    }

    // --- the lighting paths ------------------------------------------------------------------
    /// The ray-traced SVGF chain: gbuffer -> lighting + gi + fog -> clouds -> a-trous ->
    /// gi_stab -> denoise (tonemapped into out_tex at render resolution).
    pub fn record(&mut self, r: &Rec) {
        let (gx, gy) = (self.w.div_ceil(8), self.h.div_ceil(8));
        let (hx, hy) = (self.hw.div_ceil(8), self.hh.div_ceil(8));
        self.begin_timers(r);
        self.record_grass_wind(r);
        self.record_gi_cache(r);
        self.go(r, P::Gbuf, self.set(S::Gbuf), gx, gy, 1, "gbuffer");
        self.barrier(r);
        self.stamp(r, "gbuffer");
        // lighting / gi / fog only read the g-buffer and write their own targets: no barriers
        // between them (they overlap on the GPU); one after makes all three visible.
        let soft = cvars::R_SVGF_SHADOW.is("soft");
        let sun = |s: &Self| {
            if soft {
                s.go(r, P::ShTrace, s.set(S::ShTrace), gx, gy, 1, "shadow_trace");
            } else {
                s.go(r, P::Light, s.set(S::Light), gx, gy, 1, "lighting");
            }
            if s.water_on() {
                s.record_water(r, false);
            }
        };
        if cvars::R_PROFILE_SPLIT.get() {
            sun(self);
            self.barrier(r);
            if soft {
                self.record_shadow_soften(r, cvars::R_SVGF_SHADOW_MAXR.get());
            }
            self.stamp(r, "lighting");
            self.go(r, P::Gi, self.set(S::Gi), hx, hy, 1, "gi");
            self.barrier(r);
            self.stamp(r, "gi");
            self.record_fog_a(r);
            self.barrier(r);
            self.record_fog_b(r);
            self.stamp(r, "fog");
            self.record_clouds(r);
            self.stamp(r, "clouds");
        } else {
            sun(self);
            self.go(r, P::Gi, self.set(S::Gi), hx, hy, 1, "gi");
            self.record_fog_a(r);
            self.barrier(r);
            self.record_fog_b(r);
            if soft {
                self.record_shadow_soften(r, cvars::R_SVGF_SHADOW_MAXR.get());
            }
            self.stamp(r, "light+gi+fog");
            self.record_clouds(r);
            self.stamp(r, "clouds");
        }
        let levels = if cvars::R_GI_FILTER.is("smooth") { 5 } else { 3 }; // odd: result in gi_filt[0]
        for &s in ATROUS_SETS.iter().take(levels) {
            self.go(r, P::Atrous, self.set(s), hx, hy, 1, "atrous");
            self.barrier(r);
        }
        self.stamp(r, "atrous");
        if cvars::R_GI_STAB.get() {
            #[repr(C)]
            #[derive(Clone, Copy, Pod, Zeroable)]
            struct Pc {
                gamma: f32,
                max_len: f32,
                min_alpha: f32,
                pad: f32,
            }
            let len = cvars::R_GI_STAB_FRAMES.get();
            let pc = Pc { gamma: cvars::R_GI_STAB_GAMMA.get(), max_len: len, min_alpha: 1.0 / len, pad: 0.0 };
            self.go_pc(r, P::GiStab, self.set(S::GiStab), &pc, hx, hy, 1, "gi_stab");
            self.barrier(r);
            self.stamp(r, "gi_stab");
        }
        self.go(r, P::Denoise, self.set(S::Denoise), gx, gy, 1, "denoise");
        self.stamp(r, "denoise");
    }

    /// The raw 1-spp radiance (r.denoiser none): gbuffer + fog + clouds + water + raw_shade
    /// (+ lamps) -> gb_color_hdr; the caller tonemaps it.
    pub fn record_raw(&mut self, r: &Rec) {
        let (gx, gy) = (self.w.div_ceil(8), self.h.div_ceil(8));
        self.begin_timers(r);
        self.record_grass_wind(r);
        self.record_gi_cache(r);
        self.go(r, P::Gbuf, self.set(S::Gbuf), gx, gy, 1, "gbuffer");
        self.barrier(r);
        self.stamp(r, "gbuffer");
        self.record_fog_a(r);
        self.barrier(r);
        self.record_fog_b(r);
        self.stamp(r, "fog");
        self.record_clouds(r);
        self.stamp(r, "clouds");
        if self.water_on() {
            self.record_water(r, true);
            self.barrier(r);
        }
        self.go(r, P::Raw, self.set(S::Raw), gx, gy, 1, "raw_shade");
        self.stamp(r, "raw_shade");
        if cvars::R_LIGHTS.get() && self.lights_cpu.count[0] > 0 {
            self.barrier(r);
            self.go(r, P::RawLamp, self.set(S::RawLamp), gx, gy, 1, "raw_lamps");
            self.stamp(r, "raw_lamps");
        }
    }

    /// Standard lighting: gbuffer -> cached shadow map (new / edited texels) -> one exact sun
    /// ray per pixel + screen-space penumbra -> probe GI update -> fog -> clouds -> water ->
    /// lamp cubes + lamp light -> shade_std -> gb_color_hdr (the caller tonemaps it).
    pub fn record_standard(&mut self, r: &Rec) {
        let (gx, gy) = (self.w.div_ceil(8), self.h.div_ceil(8));
        let dev = &r.ctx.device;
        self.begin_timers(r);
        self.record_grass_wind(r);
        self.go(r, P::Gbuf, self.set(S::Gbuf), gx, gy, 1, "gbuffer");
        self.barrier(r);
        self.stamp(r, "gbuffer");
        if !self.std.pending.is_empty() {
            #[repr(C)]
            #[derive(Clone, Copy, Pod, Zeroable)]
            struct Pc {
                u0: i32,
                v0: i32,
                w: i32,
                h: i32,
                c: i32,
                texel: f32,
            }
            r.ctx.begin_label(r.cb, "sunshadow");
            let pipe = self.pipe(P::Shadow);
            pipe.bind(dev, r.cb, &self.sets4(self.set(S::Shadow)));
            for rg in &self.std.pending {
                let pc = Pc { u0: rg.u0, v0: rg.v0, w: rg.w, h: rg.h, c: rg.c, texel: self.std.texel[rg.c as usize] };
                pipe.push(dev, r.cb, &pc);
                unsafe { dev.cmd_dispatch(r.cb, (rg.w as u32).div_ceil(8), (rg.h as u32).div_ceil(8), 1) };
            }
            r.ctx.end_label(r.cb);
            self.std.pending.clear();
            self.barrier(r);
            self.stamp(r, "shadowmap");
        }
        self.go(r, P::ShTrace, self.set(S::ShTrace), gx, gy, 1, "shadow_trace");
        self.barrier(r);
        if cvars::R_STD_SOFTSHADOWS.get() {
            let max_r = [6.0, 12.0, 20.0, 32.0][cvars::R_STD_SHADOWS.index()];
            self.record_shadow_soften(r, max_r);
        }
        self.stamp(r, "sun shadow");
        let n_upd = self.std.probe_list.len() as u32;
        if n_upd > 0 {
            self.go(r, P::ProbeT, self.set(S::ProbeT), n_upd, 1, 1, "probe_trace");
            self.barrier(r);
            self.go(r, P::ProbeU, self.set(S::ProbeU), n_upd, 1, 1, "probe_update");
            self.barrier(r);
        }
        self.stamp(r, "probes");
        self.record_fog_a(r);
        self.barrier(r);
        self.record_fog_b(r);
        self.stamp(r, "fog");
        self.record_clouds(r);
        self.stamp(r, "clouds");
        if self.water_on() {
            self.record_water(r, false);
            self.barrier(r);
            self.stamp(r, "water");
        }
        if !self.lamp_to_trace.is_empty() {
            #[repr(C)]
            #[derive(Clone, Copy, Pod, Zeroable)]
            struct Pc {
                c: [f32; 4],
                h: [f32; 4],
                o: [f32; 4],
            }
            r.ctx.begin_label(r.cb, "lamp_cube");
            let pipe = self.pipe(P::LampCube);
            pipe.bind(dev, r.cb, &self.sets4(self.set(S::LampCube)));
            for &s in &self.lamp_to_trace {
                let ls = &self.lamp_slots[s];
                let pc = Pc {
                    c: [ls.pos.x, ls.pos.y, ls.pos.z, s as f32],
                    h: [ls.half.x, ls.half.y, ls.half.z, ls.reach],
                    o: [ls.owner, 0.0, 0.0, 0.0],
                };
                pipe.push(dev, r.cb, &pc);
                unsafe { dev.cmd_dispatch(r.cb, LAMP_CUBE_RES / 8, LAMP_CUBE_RES / 8, 6) };
            }
            r.ctx.end_label(r.cb);
            self.barrier(r);
        }
        if cvars::R_LIGHTS.get() {
            let half = cvars::R_LIGHTS_QUALITY.is("half");
            let (lx, ly) = if half { (self.hw.div_ceil(8), self.hh.div_ceil(8)) } else { (gx, gy) };
            self.go(r, P::Lamp, self.set(S::Lamp), lx, ly, 1, "lamp_std");
            self.barrier(r);
            self.stamp(r, "lamps");
        }
        self.go(r, P::Std, self.set(S::Std), gx, gy, 1, "shade_std");
        self.stamp(r, "shade_std");
    }

    // --- overlays ----------------------------------------------------------------------------------
    /// Sunlit dust motes over out_tex (region w x h); nothing when r.dust is 0.
    pub fn record_dust(&mut self, r: &Rec, w: u32, h: u32) {
        if cvars::R_DUST.get() <= 0.0 {
            return;
        }
        self.barrier(r);
        let pc: [i32; 4] = [w as i32, h as i32, 0, 0];
        self.go_pc(r, P::Dust, self.set(S::Dust), &pc, w.div_ceil(8), h.div_ceil(8), 1, "dust");
        self.stamp(r, "dust");
    }

    /// Edit-brush preview over out_tex.
    pub fn record_brush_preview(&self, r: &Rec, pc: &BrushPreviewPc) {
        self.barrier(r);
        self.go_pc(r, P::Brush, self.set(S::Brush), pc, (pc.out_res[0] as u32).div_ceil(8), (pc.out_res[1] as u32).div_ceil(8), 1, "brush_preview");
    }

    /// Shape tool preview: keep a copy of the voxels (R8 material, up to 256^3); the next
    /// record_shape_preview uploads it inside the frame's command buffer.
    pub fn set_shape_preview(&mut self, vox: &[u8], dims: [u32; 3]) {
        for k in 0..3 {
            self.shape_dims[k] = dims[k].min(super::SHAPE_VOL_DIM);
        }
        let n = (self.shape_dims[0] * self.shape_dims[1] * self.shape_dims[2]) as usize;
        self.shape_pending = vox[..n.min(vox.len())].to_vec();
        self.shape_upload = n > 0;
    }

    pub fn record_shape_preview(&mut self, r: &Rec, pc: &BrushPreviewPc) {
        if self.shape_upload {
            let n = self.shape_pending.len();
            let hb = &mut self.shape_stage[self.cur];
            if (hb.size as usize) < n {
                hb.destroy(r.ctx);
                *hb = r.ctx.create_buffer("shapeStage", n.max(1 << 20) as u64, vk::BufferUsageFlags::TRANSFER_SRC, gpu_allocator::MemoryLocation::CpuToGpu);
            }
            hb.write(0, &self.shape_pending);
            let dev = &r.ctx.device;
            let img = self.fx.shape_vol.image;
            let g = vk::ImageLayout::GENERAL;
            image_barrier(dev, r.cb, img, Stage::COMPUTE_SHADER, Access::SHADER_SAMPLED_READ, Stage::COPY, Access::TRANSFER_WRITE, g, g);
            let c = [vk::BufferImageCopy::default().image_subresource(COLOR_LAYERS).image_extent(vk::Extent3D {
                width: self.shape_dims[0],
                height: self.shape_dims[1],
                depth: self.shape_dims[2],
            })];
            unsafe { dev.cmd_copy_buffer_to_image(r.cb, hb.buf, img, g, &c) };
            image_barrier(dev, r.cb, img, Stage::COPY, Access::TRANSFER_WRITE, Stage::COMPUTE_SHADER, Access::SHADER_SAMPLED_READ, g, g);
            self.shape_upload = false;
        }
        self.record_brush_preview(r, pc);
    }

    /// Locked-view debug: snapshot this frame's G-buffer (rendered from `locked`).
    pub fn record_lock_snapshot(&mut self, r: &Rec, pp: usize, locked: &FlyCamera) {
        let dev = &r.ctx.device;
        self.barrier(r);
        let rg = [vk::ImageCopy::default()
            .src_subresource(COLOR_LAYERS)
            .dst_subresource(COLOR_LAYERS)
            .extent(vk::Extent3D { width: self.w, height: self.h, depth: 1 })];
        let g = vk::ImageLayout::GENERAL;
        let t = &self.t;
        unsafe {
            dev.cmd_copy_image(r.cb, t.gb_pos[pp].image, g, t.lock_pos.image, g, &rg);
            dev.cmd_copy_image(r.cb, t.gb_normal[pp].image, g, t.lock_normal.image, g, &rg);
            dev.cmd_copy_image(r.cb, t.gb_albedo.image, g, t.lock_albedo.image, g, &rg);
        }
        memory_barrier(dev, r.cb, Stage::COPY, Access::TRANSFER_WRITE, Stage::COMPUTE_SHADER, Access::SHADER_SAMPLED_READ);
        let u = &mut self.lock_cpu;
        u.pos = locked.pos.to_array();
        u.right = locked.right().to_array();
        u.up = locked.up().to_array();
        u.fwd = locked.forward().to_array();
        u.tan_half_fov = self.tan_half_fov;
        u.aspect = self.w as f32 / self.h as f32;
        u.res_y = self.h as f32;
        self.slots[self.cur].lock_u.write_pod(0, &self.lock_cpu);
        self.lock_valid = true;
    }

    /// Locked view mode 1: splat the locked surfaces into the ghost camera (gbuffer reads it).
    pub fn record_lock_splat(&self, r: &Rec) {
        let dev = &r.ctx.device;
        crate::gfx::context::clear_color_u(dev, r.cb, self.t.lock_depth.image, [u32::MAX; 4]);
        crate::gfx::context::clear_color_u(dev, r.cb, self.t.lock_idx.image, [u32::MAX; 4]);
        memory_barrier(dev, r.cb, Stage::CLEAR, Access::TRANSFER_WRITE, Stage::COMPUTE_SHADER, Access::SHADER_STORAGE_READ | Access::SHADER_STORAGE_WRITE);
        let (gx, gy) = (self.w.div_ceil(8), self.h.div_ceil(8));
        self.go(r, P::Splat, self.set(S::Splat), gx, gy, 1, "lock_splat");
        self.barrier(r);
        self.go(r, P::Splat, self.set(S::Splat), gx, gy, 2, "lock_splat");
        self.barrier(r);
    }

    /// Locked view mode 2: tint out_tex by what the locked camera renders.
    pub fn record_lock_view(&mut self, r: &Rec, out_w: u32, out_h: u32, mode: i32) {
        let u = &mut self.lock_cpu;
        u.mode = mode;
        u.out_w = out_w as i32;
        u.out_h = out_h as i32;
        u.valid = self.lock_valid as i32;
        u.grass_dist = if cvars::R_GRASS.is("off") { 0.0 } else { cvars::R_GRASS_DISTANCE.get() };
        self.slots[self.cur].lock_u.write_pod(0, &self.lock_cpu);
        self.barrier(r);
        self.go(r, P::Lock, self.set(S::Lock), out_w.div_ceil(8), out_h.div_ceil(8), 1, "lockview");
        self.stamp(r, "lockview");
    }
}
