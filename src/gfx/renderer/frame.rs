//! Per-frame CPU work of the renderer: lights, the camera block, the look (sun / sky / fog /
//! clouds / rain from the cvars), Standard lighting's cascade windows + probe schedule.

use super::{LampSlot, Renderer, ShadowRegion, LAMP_SLOTS, MAX_PROBES_PER_FRAME, PROBE_DIMS, PROBE_SPACING_C};
use crate::camera::FlyCamera;
use crate::cvars;
use crate::lights::GpuLight;
use crate::math::Vec3;
use crate::objects::VoxObject;
use crate::voxel::sparse_volume::RING_CHUNK_B;
use crate::voxel::world::World;

/// Sun position for an hour: rises in the east at 6, peaks (65 deg) at 12, sets at 18.
pub fn sun_from_hour(hour: f32) -> (f32, f32) {
    let a = (hour - 6.0) / 12.0 * std::f32::consts::PI;
    let el = (65.0 * a.sin()).clamp(-40.0, 89.0);
    let az = (90.0 + (hour - 6.0) * 15.0 + 360.0).rem_euclid(360.0);
    (el, az)
}

/// Sun colour through the atmosphere: relative air mass (Kasten-Young) -> transmittance,
/// normalised to the noon sun.
fn sun_transmittance(el_deg: f32) -> Vec3 {
    let e = (el_deg as f64).max(0.5);
    let m = 1.0 / (e.to_radians().sin() + 0.50572 * (e + 6.07995).powf(-1.6364));
    let beta = [0.036, 0.092, 0.215];
    Vec3::new((-beta[0] * (m - 1.0)).exp() as f32, (-beta[1] * (m - 1.0)).exp() as f32, (-beta[2] * (m - 1.0)).exp() as f32)
}

/// White-balance multiplier for r.warmth (1,1,1 at 0), luminance-preserving.
pub fn warmth_balance(w: f32) -> Vec3 {
    if w == 0.0 {
        return Vec3::ONE;
    }
    let c = Vec3::new(1.0 + 0.22 * w, 1.0 + 0.03 * w, 1.0 - 0.30 * w);
    c / (0.2126 * c.x + 0.7152 * c.y + 0.0722 * c.z)
}

fn mixv(a: Vec3, b: Vec3, t: f32) -> Vec3 {
    a + (b - a) * t
}

impl Renderer {
    pub fn set_sun(&mut self, az_deg: f32, el_deg: f32) {
        let (az, el) = (az_deg.to_radians(), el_deg.to_radians());
        let s = Vec3::new(el.cos() * az.cos(), el.sin(), el.cos() * az.sin()).normalize();
        self.scene_cpu.sun_dir = s.to_array();
    }

    pub fn lamp_cache_on(&self) -> bool {
        self.standard_mode() && cvars::R_LIGHTS.get() && cvars::R_LIGHTS_CACHE.get() && !self.lamp_slots.is_empty()
    }

    fn set_lights(&mut self, l: &[GpuLight]) {
        let n = if cvars::R_LIGHTS.get() { l.len().min(32) } else { 0 };
        let li = &mut self.lights_cpu;
        li.count[0] = n as i32;
        let k = cvars::R_LIGHTS_INTENSITY.get();
        for (i, g) in l.iter().take(n).enumerate() {
            li.a[i] = [g.pos[0], g.pos[1], g.pos[2], g.radius];
            li.b[i] = [g.power[0] * k, g.power[1] * k, g.power[2] * k, g.owner];
            li.c[i] = [g.half[0], g.half[1], g.half[2], g.pad];
        }
    }

    /// Per frame, before set_frame_uniforms: rescan world cells the last syncs touched, gather
    /// the lights nearest `eye`, schedule lamp-cube recordings.
    pub fn update_lights(&mut self, world: &World, objects: &[VoxObject], eye: Vec3, time: f32) {
        self.cam_water_depth = world.water_depth_at(eye.x, eye.y, eye.z);
        let cache = self.lamp_cache_on();
        self.lamp_frame += 1;
        self.lamp_to_trace.clear();
        if !cache {
            self.reset_lamp_cache();
        }
        fn stale_near(slots: &mut [LampSlot], lo: Vec3, hi: Vec3) {
            for s in slots.iter_mut().filter(|s| s.ready) {
                let q = s.pos.clamp(lo, hi);
                if (q - s.pos).length() < s.reach {
                    s.ready = false;
                }
            }
        }
        for (cx, cz) in std::mem::take(&mut self.gpu.chunk_events) {
            self.lights.rescan_chunk(world, cx, cz);
            if cache {
                let (x0, z0) = ((cx * RING_CHUNK_B * 8) as f32, (cz * RING_CHUNK_B * 8) as f32);
                let k = (RING_CHUNK_B * 8) as f32;
                stale_near(&mut self.lamp_slots, Vec3::new(x0, 0.0, z0), Vec3::new(x0 + k, world.ny as f32, z0 + k));
            }
        }
        if !self.gpu.synced_grid.is_empty() {
            let synced = std::mem::take(&mut self.gpu.synced_grid);
            if cache {
                for &gi in &synced {
                    let (ib, jb, kb) = world.brick_coords(gi as usize);
                    let b = Vec3::new(ib as f32, jb as f32, kb as f32);
                    stale_near(&mut self.lamp_slots, b * 8.0, (b + 1.0) * 8.0);
                }
            }
            self.lights.rescan_bricks(world, &synced);
        }
        if cache {
            for i in objects.len()..self.lamp_body_box.len() {
                let (a, b) = self.lamp_body_box[i];
                stale_near(&mut self.lamp_slots, a, b);
            }
            self.lamp_body_box.resize(objects.len(), (Vec3::splat(1e30), Vec3::splat(-1e30)));
            for (i, o) in objects.iter().enumerate() {
                let (lo, hi) = o.world_aabb();
                let prev = self.lamp_body_box[i];
                if (lo - prev.0).length() + (hi - prev.1).length() < 0.01 {
                    continue;
                }
                if prev.0.x < 1e29 {
                    stale_near(&mut self.lamp_slots, prev.0, prev.1);
                }
                stale_near(&mut self.lamp_slots, lo, hi);
                self.lamp_body_box[i] = (lo, hi);
            }
        }
        let mut l = self.lights.gather(world, objects, eye, time);
        // Each light keeps a cube slot; a slot not recorded yet is recorded this frame (a few
        // per frame) - until then that light traces rays.
        const TRACE_PER_FRAME: usize = 6;
        let intensity = cvars::R_LIGHTS_INTENSITY.get();
        for g in &mut l {
            g.pad = -1.0;
            if !cache {
                continue;
            }
            let p = Vec3::from_array(g.pos);
            let hf = Vec3::from_array(g.half);
            let mut slot = self.lamp_slots.iter().position(|s| s.owner == g.owner && (g.owner >= 0.0 || (s.pos - p).length() < 0.5));
            if let Some(s) = slot {
                let ls = &mut self.lamp_slots[s];
                if (ls.pos - p).length() + (ls.half - hf).length() > 0.02 {
                    ls.ready = false; // a body light that moved (or a cluster that grew)
                }
            }
            if slot.is_none() {
                slot = self.lamp_slots.iter().position(|s| s.owner == -2.0);
                if slot.is_none() {
                    let frame = self.lamp_frame;
                    slot = (0..LAMP_SLOTS).filter(|&q| self.lamp_slots[q].used != frame).min_by_key(|&q| self.lamp_slots[q].used);
                }
                let Some(s) = slot else { continue };
                self.lamp_slots[s] = LampSlot::default();
            }
            let si = slot.unwrap();
            let s = &mut self.lamp_slots[si];
            s.pos = p;
            s.half = hf;
            s.owner = g.owner;
            s.used = self.lamp_frame;
            let sum = (g.power[0] + g.power[1] + g.power[2]) * intensity;
            let reach = (2.0 * (sum.max(0.0) / (3.0 * std::f32::consts::PI * 0.004)).sqrt() + g.radius).clamp(16.0, 1024.0);
            if s.ready && reach > s.reach {
                s.ready = false;
            }
            if !s.ready && self.lamp_to_trace.len() < TRACE_PER_FRAME {
                s.reach = reach;
                s.ready = true; // recorded by record_standard this frame, before lamp_std reads it
                self.lamp_to_trace.push(si);
            }
            if s.ready {
                g.pad = si as f32;
            }
        }
        self.set_lights(&l);
        let t = self.lights.torches(world, eye);
        let li = &mut self.lights_cpu;
        li.torch_count[0] = t.len().min(64) as i32;
        for (i, tt) in t.iter().take(64).enumerate() {
            li.ta[i] = tt.a;
            li.tb[i] = tt.b;
        }
    }

    /// The look cvars -> the scene block (sun / moon, sky, fog, clouds, rain, display).
    pub fn apply_look(&mut self) {
        use cvars::*;
        let el = R_SUN_ELEVATION.get();
        let moon = el < 0.0;
        let light_fade = if moon { (-el / 6.0).clamp(0.0, 1.0) } else { (el / 3.0).clamp(0.0, 1.0) };
        let night_amt = 1.0 - ((el + 8.0) / 10.0).clamp(0.0, 1.0);
        if R_SUN_AZIMUTH.get() != self.look.last_az || el != self.look.last_el {
            self.look.last_az = R_SUN_AZIMUTH.get();
            self.look.last_el = el;
            if moon {
                self.set_sun((self.look.last_az + 180.0).rem_euclid(360.0), (-el).max(3.0));
            } else {
                self.set_sun(self.look.last_az, el.max(0.5));
            }
        }
        let phys = R_SKY_MODEL.is("physical");
        let mut sun_base = Vec3::new((1.0f64 * 1.7) as f32, (0.95f64 * 1.7) as f32, (0.86f64 * 1.7) as f32);
        let mut fog_sun = Vec3::new((1.0f64 * 2.6) as f32, (0.96f64 * 2.6) as f32, (0.87f64 * 2.6) as f32);
        let mut zen = Vec3::new(0.25, 0.45, 0.85);
        let mut hor = Vec3::new(0.75, 0.82, 0.90);
        let mut sunset = Vec3::ZERO;
        if phys {
            let t = sun_transmittance(el.max(0.5));
            sun_base *= t;
            fog_sun *= t;
            let mut s = 1.0 - (el.max(0.0) / 26.0).clamp(0.0, 1.0);
            s = s * s * (3.0 - 2.0 * s);
            let dim = 1.0 - 0.45 * s;
            zen = mixv(zen, Vec3::new(0.14, 0.22, 0.48), s) * dim;
            hor = mixv(hor, Vec3::new(0.98, 0.66, 0.46), s) * dim;
            sunset = Vec3::new(1.0, 0.45, 0.16) * (0.9 * s);
        }
        if moon {
            sun_base = Vec3::new(0.55, 0.65, 0.95) * 0.35;
            fog_sun = Vec3::new(0.55, 0.65, 0.95) * 0.16;
        }
        sun_base *= light_fade;
        fog_sun *= light_fade;
        zen = mixv(zen, Vec3::new(0.010, 0.016, 0.040), night_amt);
        hor = mixv(hor, Vec3::new(0.028, 0.036, 0.065), night_amt);
        sunset *= 1.0 - night_amt;
        let clouds = R_CLOUDS.get();
        let cover = R_CLOUDS_COVERAGE.get();
        let u = &mut self.scene_cpu;
        u.night = night_amt;
        let over = if clouds { ((cover - 0.5) / 0.45).clamp(0.0, 1.0) } else { 0.0 };
        u.sun_rgb = (sun_base * R_SUN_INTENSITY.get() * (1.0 - 0.6 * over)).to_array();
        // Under cloud the light comes from the whole bright deck: a larger, softer sun.
        let cloudy = if clouds { ((cover - 0.3) / 0.6).clamp(0.0, 1.0) } else { 0.0 };
        u.sun_angle = 0.028 * R_SUN_SOFTNESS.get() * (1.0 + 6.0 * cloudy * cloudy);
        u.fog_sun_rgb = (fog_sun * R_GODRAYS_STRENGTH.get()).to_array();
        u.fog_density = 0.0012 * R_FOG_DENSITY.get();
        u.fog_ambient = ((hor * 0.7 + zen * 0.3) * (0.85 * R_SKY_INTENSITY.get() * R_FOG_AMBIENT.get())).to_array();
        u.fog_g = R_GODRAYS_FOCUS.get();
        {
            // overcast / rain: the sky goes grey and the sun weaker
            let oc = ((cover - 0.55) / 0.4).clamp(0.0, 1.0) * if clouds { 1.0 } else { 0.0 };
            let g = oc * oc * (3.0 - 2.0 * oc);
            let grey = |c: Vec3, lum: f32| {
                let l = (0.2126 * c.x + 0.7152 * c.y + 0.0722 * c.z) * lum;
                c + (Vec3::new(l, l * 1.02, l * 1.06) - c) * g
            };
            let dim = 1.0 - 0.3 * g - 0.2 * R_RAIN.get();
            zen = grey(zen, 0.95) * dim;
            hor = grey(hor, 0.9) * dim;
        }
        u.sky_zenith = zen.to_array();
        u.sky_horizon = hor.to_array();
        u.sunset_rgb = sunset.to_array();
        u.sky_scale = R_SKY_INTENSITY.get();
        u.gi_scale = R_GI_INTENSITY.get();
        u.sky_model = phys as i32;
        u.fog_height = R_FOG_HEIGHT.get() + (self.world_sea_level - 232.0);
        {
            // clouds drift along the wind (accumulated in doubles; the pattern is absolute)
            let now = std::time::Instant::now();
            let lk = &mut self.look;
            let dt = (now - lk.last).as_secs_f64().min(0.25);
            lk.last = now;
            let a = (R_WIND_DIR.get() as f64).to_radians();
            let v = R_CLOUDS_SPEED.get() as f64 * 10.0;
            lk.drift_x = (lk.drift_x + a.sin() * v * dt) % 4.0e6;
            lk.drift_z = (lk.drift_z + a.cos() * v * dt) % 4.0e6;
            lk.evolve = (lk.evolve + dt * 0.0013) % 1000.0;
            u.cloud_a = [cover, self.world_sea_level + R_CLOUDS_BASE.get() * 10.0, R_CLOUDS_THICKNESS.get() * 10.0, 0.029 * R_CLOUDS_DENSITY.get()];
            u.cloud_b = [
                ((self.world_origin[0] as f64 - lk.drift_x) % 4.0e6) as f32,
                ((self.world_origin[1] as f64 - lk.drift_z) % 4.0e6) as f32,
                lk.evolve as f32,
                if clouds && cover > 0.0 { 1.0 } else { 0.0 },
            ];
            if !R_RAIN_PAUSE.get() {
                lk.rain_time = (lk.rain_time + dt) % 3600.0;
            }
            let r = R_RAIN.get();
            lk.wet += if r > 0.05 { (r - lk.wet) * (1.0 - (-dt / 40.0).exp()) as f32 } else { -lk.wet * (1.0 - (-dt / 180.0).exp()) as f32 };
            u.rain_a = [r, lk.rain_time as f32, lk.wet, 0.0];
        }
        u.fog_falloff = R_FOG_FALLOFF.get();
        u.fog_noise = R_FOG_NOISE.get();
        let q = R_FOG_QUALITY.index();
        u.fog_steps = [8, 10, 16][q];
        u.fog_shadow_max = [400.0, 520.0, 900.0][q];
        u.tonemap_op = R_TONEMAP.is("agx") as i32;
        u.saturation = R_SATURATION.get();
        u.contrast = R_CONTRAST.get();
        u.white_balance = warmth_balance(R_WARMTH.get()).to_array();
        u.grass_shadows = (R_GRASS_SHADOWS.get() && !R_GRASS.is("off")) as i32;
    }

    /// The DDGI probe grid: this frame's grid origins and probe update list (fine + coarse).
    fn prepare_probes(&mut self, cam: &FlyCamera, frame: u32, probes_per_frame: i32) {
        let freeze = cvars::R_STD_PROBEFREEZE.get() > 0 && frame >= cvars::R_STD_PROBEFREEZE.get() as u32 && self.std.probe_origin[0] != i32::MIN;
        let spacing = self.std.probe_spacing;
        let cp = cam.pos.to_array();
        let u = &mut self.scene_cpu;
        for k in 0..3 {
            u.probe_origin[k] = if freeze { self.std.probe_origin[k] } else { (cp[k] / spacing).floor() as i32 - PROBE_DIMS[k] / 2 };
            u.probe_origin_c[k] = if freeze { self.std.probe_origin_c[k] } else { (cp[k] / PROBE_SPACING_C).floor() as i32 - PROBE_DIMS[k] / 2 };
            u.probe_dims[k] = PROBE_DIMS[k];
        }
        u.probe_spacing = spacing;
        u.probe_spacing_c = PROBE_SPACING_C;
        u.probe_update_offset = 0;
        let n_probes = (PROBE_DIMS[0] * PROBE_DIMS[1] * PROBE_DIMS[2]) as usize;
        let st = &mut self.std;
        st.probe_list.clear();
        let boot = st.boot_frames > 0;
        if boot {
            st.boot_frames -= 1;
        }
        let fine_budget = if boot { 4096 } else { probes_per_frame };
        let coarse_budget = if boot { 1024 } else { (probes_per_frame / 4).max(256) };
        let new_f = u.probe_origin;
        let new_c = u.probe_origin_c;
        let edit_boxes = std::mem::take(&mut st.edit_boxes);
        let level = |st: &mut super::StdState, new_o: [i32; 3], old_is_c: bool, budget: i32, bit: u32, sp: f32| {
            st.taken.clear();
            st.taken.resize(n_probes, 0);
            let start = st.probe_list.len();
            let full = |l: &Vec<u32>| (l.len() - start) as i32 >= budget;
            let wrap = |c: i32, k: usize| c.rem_euclid(PROBE_DIMS[k]);
            // Probes around this frame's edits re-settle first.
            for b in &edit_boxes {
                let mut c0 = [0; 3];
                let mut c1 = [0; 3];
                for k in 0..3 {
                    c0[k] = ((b[k] as f32 / sp).floor() as i32 - 1).max(new_o[k]);
                    c1[k] = ((b[k + 3] as f32 / sp).floor() as i32 + 1).min(new_o[k] + PROBE_DIMS[k] - 1);
                }
                for z in c0[2]..=c1[2] {
                    for y in c0[1]..=c1[1] {
                        for x in c0[0]..=c1[0] {
                            let lin = (wrap(x, 0) + PROBE_DIMS[0] * (wrap(y, 1) + PROBE_DIMS[1] * wrap(z, 2))) as usize;
                            if st.taken[lin] != 0 || full(&st.probe_list) {
                                continue;
                            }
                            st.probe_list.push(lin as u32 | bit | 0x4000_0000);
                            st.taken[lin] = 1;
                        }
                    }
                }
            }
            let old_o = if old_is_c { st.probe_origin_c } else { st.probe_origin };
            let moved = old_o != new_o;
            if moved && old_o[0] != i32::MIN {
                for lin in 0..n_probes {
                    if full(&st.probe_list) {
                        break;
                    }
                    let sl = [lin as i32 % PROBE_DIMS[0], (lin as i32 / PROBE_DIMS[0]) % PROBE_DIMS[1], lin as i32 / (PROBE_DIMS[0] * PROBE_DIMS[1])];
                    let fresh = (0..3).any(|k| {
                        let d = (sl[k] - new_o[k]).rem_euclid(PROBE_DIMS[k]);
                        let cell = new_o[k] + d;
                        cell < old_o[k] || cell >= old_o[k] + PROBE_DIMS[k]
                    });
                    if fresh && st.taken[lin] == 0 {
                        st.probe_list.push(lin as u32 | bit);
                        st.taken[lin] = 1;
                    }
                }
            }
            if !freeze {
                let mut cursor = if old_is_c { st.rr_cursor_c } else { st.rr_cursor };
                for _ in 0..n_probes {
                    if full(&st.probe_list) {
                        break;
                    }
                    let lin = cursor;
                    cursor = (cursor + 1) % n_probes as u32;
                    if st.taken[lin as usize] == 0 {
                        st.probe_list.push(lin | bit);
                    }
                }
                if old_is_c {
                    st.rr_cursor_c = cursor;
                } else {
                    st.rr_cursor = cursor;
                }
            }
            if old_is_c {
                st.probe_origin_c = new_o;
            } else {
                st.probe_origin = new_o;
            }
        };
        level(st, new_f, false, fine_budget, 0, spacing);
        level(st, new_c, true, coarse_budget, 0x8000_0000, PROBE_SPACING_C);
        st.probe_list.truncate(MAX_PROBES_PER_FRAME);
        let u = &mut self.scene_cpu;
        u.probe_hyst = match cvars::R_STD_GISPEED.index() {
            0 => 0.97,
            2 => 0.90,
            _ => 0.94,
        };
        u.probe_normal_bias = 0.8;
    }

    /// Per frame: light-space basis, cascade windows around the camera (snapped to 64 texels so
    /// strips re-trace in chunks), the texel rects to trace, the probe schedule.
    fn prepare_standard(&mut self, cam: &FlyCamera, frame: u32) {
        let std_on = self.standard_mode();
        self.scene_cpu.std_lighting = std_on as i32;
        self.scene_cpu.sh_n = self.std.sh_n_alloc;
        if !std_on {
            self.std.pending.clear();
            self.gpu.edit_boxes.clear();
            self.std.probe_list.clear();
            self.std.edit_boxes.clear();
            return;
        }
        use cvars::*;
        let want_n = match R_STD_SHADOWS.index() {
            0 => 1024,
            3 => 3072,
            _ => 2048,
        };
        let dist = R_STD_SHADOWDIST.get();
        let ext = [192.0 * dist, 576.0 * dist, 1728.0 * dist, 5184.0 * dist];
        let st = &mut self.std;
        let mut cfg_changed = false;
        if want_n != st.sh_n {
            st.sh_n = want_n;
            cfg_changed = true;
        }
        if R_STD_HYBRIDSHADOWS.get() != st.hybrid {
            st.hybrid = R_STD_HYBRIDSHADOWS.get();
            cfg_changed = true;
        }
        if st.hybrid && R_FOLIAGE_LIGHT.get() != st.leaf_light {
            st.leaf_light = R_FOLIAGE_LIGHT.get();
            cfg_changed = true;
        }
        for c in 0..4 {
            let tx = ext[c] / st.sh_n as f32;
            if tx != st.texel[c] {
                st.texel[c] = tx;
                cfg_changed = true;
            }
        }
        if st.sh_n != st.sh_n_alloc {
            return; // update_sets reallocates the map first
        }
        let n = st.sh_n;
        st.probes_per_frame = match R_STD_GISPEED.index() {
            0 => 1024,
            2 => 4096,
            _ => 2048,
        };
        st.probe_spacing = 8.0;
        let u = &mut self.scene_cpu;
        u.std_soft_taps = if R_STD_SOFTSHADOWS.get() { if R_STD_SHADOWS.index() <= 1 { 8.0 } else { 16.0 } } else { 0.0 };
        u.std_cascade_blend = if R_STD_CASCADEBLEND.get() { 0.25 } else { 0.0 };
        u.std_cascade_bias = R_STD_CASCADEBIAS.get();
        u.std_view_bias = R_STD_VIEWBIAS.get();
        u.std_soft_min = 0.3;
        let l = Vec3::from_array(u.sun_dir);
        let full = !st.valid || l.dot(st.sun) < 0.999_999 || R_STD_RETRACEALL.get() || cfg_changed;
        if full {
            st.sun = l;
            let rf = if l.y.abs() < 0.99 { Vec3::Y } else { Vec3::X };
            st.r = rf.cross(l).normalize();
            st.u = l.cross(st.r);
            let w = [(self.gpu.wbx * 8) as f32, (self.gpu.wby * 8) as f32, (self.gpu.wbz * 8) as f32];
            let mut top = -1e30f32;
            for i in 0..8 {
                let c = Vec3::new(if i & 1 != 0 { w[0] } else { 0.0 }, if i & 2 != 0 { w[1] } else { 0.0 }, if i & 4 != 0 { w[2] } else { 0.0 });
                top = top.max(c.dot(l));
            }
            st.lz_top = top + 4.0;
            st.pending.clear();
        }
        for c in 0..4 {
            let tx = st.texel[c];
            let mut nu = (cam.pos.dot(st.r) / tx).floor() as i32 - n / 2;
            let mut nv = (cam.pos.dot(st.u) / tx).floor() as i32 - n / 2;
            nu = (nu as f32 / 64.0).floor() as i32 * 64;
            nv = (nv as f32 / 64.0).floor() as i32 * 64;
            let (ou, ov) = (st.win[c][0], st.win[c][1]);
            let (du, dv) = (nu - ou, nv - ov);
            let ci = c as i32;
            if full || du.abs() >= n || dv.abs() >= n {
                st.pending.push(ShadowRegion { c: ci, u0: nu, v0: nv, w: n, h: n });
            } else {
                if du > 0 {
                    st.pending.push(ShadowRegion { c: ci, u0: ou + n, v0: nv, w: du, h: n });
                }
                if du < 0 {
                    st.pending.push(ShadowRegion { c: ci, u0: nu, v0: nv, w: -du, h: n });
                }
                if dv > 0 {
                    st.pending.push(ShadowRegion { c: ci, u0: nu, v0: ov + n, w: n, h: dv });
                }
                if dv < 0 {
                    st.pending.push(ShadowRegion { c: ci, u0: nu, v0: nv, w: n, h: -dv });
                }
            }
            st.win[c] = [nu, nv];
        }
        if !full && self.gpu.edit_counter != st.edit_seen {
            // world edits / streamed chunks: re-trace their shadow, box by box
            for e in &self.gpu.edit_boxes {
                for c in 0..4 {
                    let tx = st.texel[c];
                    let (mut u0, mut v0, mut u1, mut v1) = (i32::MAX, i32::MAX, i32::MIN, i32::MIN);
                    for i in 0..8 {
                        let p = Vec3::new(
                            (if i & 1 != 0 { e[3] } else { e[0] }) as f32,
                            (if i & 2 != 0 { e[4] } else { e[1] }) as f32,
                            (if i & 4 != 0 { e[5] } else { e[2] }) as f32,
                        );
                        let pu = (p.dot(st.r) / tx).floor() as i32;
                        let pv = (p.dot(st.u) / tx).floor() as i32;
                        u0 = u0.min(pu);
                        u1 = u1.max(pu);
                        v0 = v0.min(pv);
                        v1 = v1.max(pv);
                    }
                    u0 = (u0 - 1).max(st.win[c][0]);
                    v0 = (v0 - 1).max(st.win[c][1]);
                    u1 = (u1 + 1).min(st.win[c][0] + n - 1);
                    v1 = (v1 + 1).min(st.win[c][1] + n - 1);
                    if u1 >= u0 && v1 >= v0 {
                        st.pending.push(ShadowRegion { c: c as i32, u0, v0, w: u1 - u0 + 1, h: v1 - v0 + 1 });
                    }
                }
            }
        }
        st.edit_boxes = std::mem::take(&mut self.gpu.edit_boxes);
        st.edit_seen = self.gpu.edit_counter;
        st.valid = true;
        let u = &mut self.scene_cpu;
        u.sh_right = st.r.to_array();
        u.sh_up = st.u.to_array();
        u.sh_texel0 = st.texel[0];
        u.sh_texel1 = st.texel[1];
        u.sh_texel2 = st.texel[2];
        u.sh_texel3 = st.texel[3];
        u.sh_lz_top = st.lz_top;
        u.sh_win_a = [st.win[0][0], st.win[0][1], st.win[1][0], st.win[1][1]];
        u.sh_win_b = [st.win[2][0], st.win[2][1], st.win[3][0], st.win[3][1]];
        let ppf = st.probes_per_frame;
        self.prepare_probes(cam, frame, ppf);
    }

    /// Writes this frame's blocks (camera + last frame's camera for reprojection, the scene,
    /// lights, probe list) into frame slot `frame & 1`.
    #[allow(clippy::too_many_arguments)]
    pub fn set_frame_uniforms(&mut self, cam: &FlyCamera, prev: &FlyCamera, frame_idx: u32, exposure: f32, time: f32, jitter: [f32; 2]) {
        use cvars::*;
        self.cur = (frame_idx & 1) as usize;
        let (f, r, u) = (cam.forward(), cam.right(), cam.up());
        let (pf, pr, pu) = (prev.forward(), prev.right(), prev.up());
        let mut p: crate::gfx::ubo::PassUbo = bytemuck::Zeroable::zeroed();
        p.cam_pos = cam.pos.to_array();
        p.cam_right = r.to_array();
        p.cam_up = u.to_array();
        p.cam_fwd = f.to_array();
        let fov = if self.zoom_fov_deg > 0.0 { self.zoom_fov_deg } else { R_FOV.get() };
        self.tan_half_fov = (fov * 0.5).to_radians().tan();
        if self.prev_tan_half_fov <= 0.0 {
            self.prev_tan_half_fov = self.tan_half_fov;
        }
        p.tan_half_fov = self.tan_half_fov;
        p.prev_tan_half_fov = self.prev_tan_half_fov;
        self.prev_tan_half_fov = self.tan_half_fov;
        p.lock_active = if self.lock_active && self.lock_valid { 1.0 } else { 0.0 };
        p.prev_cam_pos = prev.pos.to_array();
        p.prev_cam_right = pr.to_array();
        p.prev_cam_up = pu.to_array();
        p.prev_cam_fwd = pf.to_array();
        p.frame_idx = frame_idx.wrapping_add(R_FRAME_OFFSET.get() as u32);
        // radiance cache: this frame's world edits (before prepare_standard takes them)
        if super::gi_cache_wanted() {
            self.gi_cache_boxes.extend_from_slice(&self.gpu.edit_boxes);
        }
        let s = &mut self.scene_cpu;
        s.gi_cache_cam = cam.pos.to_array();
        s.gi_cache_lod = R_GI_CACHE_LOD.get();
        s.gi_cache_p = [p.frame_idx as i32, R_GI_CACHE_MIN.get(), R_GI_CACHE_CELL.get(), R_GI_CACHE_BUDGET.get().min(super::GI_CACHE_QUEUE_MAX as i32)];
        s.gi_cache_q = [R_GI_CACHE_FRAMES.get() as f32, R_GI_CACHE_EVICT.get() as f32, 0.0, 0.0];
        // A streaming world's window: one ring period of chunks centred on the camera.
        if self.gpu.ring_mode {
            let w = self.gpu.wbx / RING_CHUNK_B;
            let k = (RING_CHUNK_B * 8) as f32;
            let (ccx, ccz) = ((cam.pos.x / k).floor() as i32, (cam.pos.z / k).floor() as i32);
            s.world_win = [(ccx - w / 2) * RING_CHUNK_B, 0, (ccz - w / 2) * RING_CHUNK_B, 1];
        } else {
            s.world_win = [0; 4];
        }
        s.hor_a = [self.hor_win[0], self.hor_win[1], self.hor_win[2], self.hor_win[3]];
        s.hor_b = [self.hor_win[4], self.hor_win[5], if self.gpu.ring_mode && self.hor_on != 0 { (self.hor_max_h + 2).max(2) } else { 0 }, self.hor_rin];
        s.hor_c = [self.hor_win[6], self.hor_win[7], self.hor_win[8], self.hor_win[9]];
        s.chunk_fade = self.chunk_fade;
        p.exposure = exposure;
        p.debug_mode = R_DEBUGVIEW.get();
        p.time = time;
        p.prev_time = if frame_idx == 0 { time } else { self.last_time };
        self.wind_dt = if frame_idx == 0 { 0.0 } else { (time - self.last_time).clamp(0.0, 0.1) };
        self.wind_cam = [cam.pos.x, cam.pos.z];
        self.last_time = time;
        self.scene_cpu.grass_dist = R_GRASS_DISTANCE.get();
        self.apply_look();
        self.prepare_standard(cam, frame_idx);
        p.water_p = [R_WATER_CLARITY.get(), R_WATER_REFRACT.get(), R_WATER_MAXDEPTH.get(), if R_WATER_REFLECT.get() { R_WATER_REFLECT_DIST.get() } else { 0.0 }];
        p.look_p = [R_FOLIAGE_SSS.get(), R_DUST.get(), R_DUST_DENSITY.get(), R_FOLIAGE_LIGHT.get()];
        // Thick fog starts near: never further out than a quarter of what you can see through it.
        let fd = R_FOG_DENSITY.get();
        p.fog_start = R_FOG_START.get().min(if fd > 0.0 { 0.25 * 3.0 / (0.0012 * fd) } else { 1e9 });
        let hd = R_HAZE_DISTANCE.get();
        p.haze_k = (if hd > 0.0 { 3.0 / (hd * 10.0) } else { 0.0 }).max(0.0) + R_RAIN.get() * 3.0 / 15000.0;
        p.haze_sea = self.world_sea_level;
        p.fog_p = [R_FOG_RANGE.get(), if self.frox_dims[2] > 0 { self.frox_dims[2] as f32 } else { 64.0 }, 0.5, self.w as f32 / self.h as f32];
        p.under_p = [self.cam_water_depth, R_WATER_MURK.get(), R_WATER_CAUSTICS.get(), 0.0];
        p.opt_p = [
            if self.standard_mode() && self.std.hybrid { 1.0 } else { 0.0 },
            if R_GRASS.is("off") || R_GRASS_FAR.get() <= 1.0 { 0.0 } else { R_GRASS_FAR.get() },
            if R_CLOUDS_TEMPORAL.get() { 1.0 } else { 0.0 },
            if self.refl_half_on() { 1.0 } else { 0.0 },
        ];
        p.facet_p = [R_FACETS.get(), 0.0, 0.0, 0.0];
        let a = R_WIND_DIR.get().to_radians();
        p.wave_p = [a.cos(), a.sin(), R_WIND_STRENGTH.get(), R_WATER_WAVES.get()];
        p.shift_p = [-self.origin_shift[0], 0.0, -self.origin_shift[1], 0.0];
        self.origin_shift = [0.0; 2];
        p.jitter = jitter;
        self.pass_cpu = p;
        let sl = &self.slots[self.cur];
        sl.scene_u.write_pod(0, &self.scene_cpu);
        sl.lock_u.write_pod(0, &self.lock_cpu);
        sl.main_u.write_pod(0, &p);
        sl.lights_u.write_pod(0, &*self.lights_cpu);
        if !self.std.probe_list.is_empty() {
            sl.probe_list_u.write(0, bytemuck::cast_slice(&self.std.probe_list));
        }
    }

    pub fn water_on(&self) -> bool {
        cvars::R_WATER.get() || cvars::R_REFLECT.get()
    }

    pub fn refl_half_on(&self) -> bool {
        cvars::R_WATER.get() && cvars::R_WATER_REFLECT.get() && cvars::R_WATER_REFLECT_HALF.get()
    }

    /// Floating origin: world coordinates moved by (dx, dz) voxels (multiples of 65536).
    pub fn shift_origin(&mut self, dx: i32, dz: i32) {
        self.origin_shift[0] += dx as f32;
        self.origin_shift[1] += dz as f32;
        self.lights.shift(dx as f32, dz as f32);
        self.gpu.rt.shift(dx, dz);
        self.reset_lamp_cache();
        let st = &mut self.std;
        st.valid = false;
        st.edit_boxes.clear();
        self.gpu.edit_boxes.clear();
        if st.probe_origin[0] != i32::MIN {
            st.probe_origin[0] -= dx / 8;
            st.probe_origin[2] -= dz / 8;
        }
        if st.probe_origin_c[0] != i32::MIN {
            st.probe_origin_c[0] -= dx / 32;
            st.probe_origin_c[2] -= dz / 32;
        }
        self.wind_origin[0] -= dx / 2;
        self.wind_origin[1] -= dz / 2;
        for s in &mut self.grass_stamp {
            s[0] -= dx as f32;
            s[2] -= dz as f32;
        }
        self.lock_valid = false;
        self.gi_cache_clear = true;
    }

    /// A player edit changed the world in voxel box [lo, hi): the Standard probes around it
    /// re-settle next frame (a carved hole frees buried probes; a placed wall buries some).
    pub fn note_world_edit(&mut self, lo: [i32; 3], hi: [i32; 3]) {
        self.std.edit_boxes.push([lo[0], lo[1], lo[2], hi[0], hi[1], hi[2]]);
    }

    /// Endless worlds: the far horizon's state (on, each level's window in the engine's texels,
    /// the radius inside which the voxels are trusted) and its finished / cleared tiles.
    pub fn update_horizon(&mut self, ctx: &crate::gfx::context::Context, on: bool, win: [i32; 10], inner_radius: i32, tiles: &[crate::horizon::Tile]) {
        use crate::gfx::context::{image_barrier, Access, Stage};
        use crate::horizon::{N, TILE};
        use ash::vk;
        self.hor_on = on as i32;
        self.hor_win = win;
        self.hor_rin = inner_radius;
        if tiles.is_empty() {
            return;
        }
        // Tiles land at (texel & (N - 1)) of their level's band; a cleared tile copies zeros.
        // Each texel is two words (ground, crowns); each tile also writes its 4 x 4 blocks'
        // tallest tops (voxels + 1; 0 = not made), which the shader's horizon march skips by.
        const B: i32 = 8; // texels per block side
        let (bt, bn) = (TILE / B, N / B);
        let top_of = |g: u32, c: u32| (((g & 0x3FFF) + 7) >> 3).max(c & 0x7FF) as i32;
        let (mut data, mut blk): (Vec<u32>, Vec<u32>) = (Vec::new(), Vec::new());
        let (mut regions, mut bregions) = (Vec::new(), Vec::new());
        let (mut zero_at, mut bzero_at) = (None, None);
        for t in tiles {
            let (at, bat) = if !t.px.is_empty() {
                let (at, bat) = (data.len(), blk.len());
                data.extend_from_slice(&t.px);
                blk.resize(blk.len() + (bt * bt) as usize, 0);
                for z in 0..TILE {
                    for x in 0..TILE {
                        let i = ((z * TILE + x) * 2) as usize;
                        let top = top_of(t.px[i], t.px[i + 1]) + 1;
                        self.hor_max_h = self.hor_max_h.max(top);
                        let m = &mut blk[bat + ((z / B) * bt + x / B) as usize];
                        *m = (*m).max(top as u32);
                    }
                }
                (at, bat)
            } else {
                let at = *zero_at.get_or_insert_with(|| {
                    data.resize(data.len() + (TILE * TILE * 2) as usize, 0);
                    data.len() - (TILE * TILE * 2) as usize
                });
                let bat = *bzero_at.get_or_insert_with(|| {
                    blk.resize(blk.len() + (bt * bt) as usize, 0);
                    blk.len() - (bt * bt) as usize
                });
                (at, bat)
            };
            let level = t.level as i32;
            regions.push(
                vk::BufferImageCopy::default()
                    .buffer_offset(at as u64 * 4)
                    .image_subresource(crate::gfx::context::COLOR_LAYERS)
                    .image_offset(vk::Offset3D { x: (t.tx * TILE) & (N - 1), y: ((t.tz * TILE) & (N - 1)) + level * N, z: 0 })
                    .image_extent(vk::Extent3D { width: TILE as u32, height: TILE as u32, depth: 1 }),
            );
            bregions.push(
                vk::BufferImageCopy::default()
                    .buffer_offset(bat as u64 * 4)
                    .image_subresource(crate::gfx::context::COLOR_LAYERS)
                    .image_offset(vk::Offset3D { x: (t.tx * bt) & (bn - 1), y: ((t.tz * bt) & (bn - 1)) + level * bn, z: 0 })
                    .image_extent(vk::Extent3D { width: bt as u32, height: bt as u32, depth: 1 }),
            );
        }
        self.gpu.retire_syncs(ctx, false);
        let (b, off) = self.gpu.stage(ctx, bytemuck::cast_slice(&data));
        let (bb, boff) = self.gpu.stage(ctx, bytemuck::cast_slice(&blk));
        for r in &mut regions {
            r.buffer_offset += off;
        }
        for r in &mut bregions {
            r.buffer_offset += boff;
        }
        let cb = self.gpu.begin_sync(ctx);
        let dev = &ctx.device;
        let (hor, hblk) = (self.fx.horizon.image, self.fx.hor_blk.image);
        for im in [hor, hblk] {
            image_barrier(dev, cb, im, Stage::COMPUTE_SHADER, Access::SHADER_SAMPLED_READ, Stage::COPY, Access::TRANSFER_WRITE, vk::ImageLayout::GENERAL,
                          vk::ImageLayout::GENERAL);
        }
        unsafe {
            dev.cmd_copy_buffer_to_image(cb, b, hor, vk::ImageLayout::GENERAL, &regions);
            dev.cmd_copy_buffer_to_image(cb, bb, hblk, vk::ImageLayout::GENERAL, &bregions);
        }
        for im in [hor, hblk] {
            image_barrier(dev, cb, im, Stage::COPY, Access::TRANSFER_WRITE, Stage::COMPUTE_SHADER, Access::SHADER_SAMPLED_READ, vk::ImageLayout::GENERAL,
                          vk::ImageLayout::GENERAL);
        }
        self.gpu.submit_sync(ctx, cb);
    }

    /// Endless worlds: per ring slot, how much of its fade-in is still to go.
    pub fn set_chunk_fade(&mut self, per_slot: &[u8]) {
        self.chunk_fade = [0; 256];
        let bytes: &mut [u8] = bytemuck::cast_slice_mut(&mut self.chunk_fade);
        let n = per_slot.len().min(bytes.len());
        bytes[..n].copy_from_slice(&per_slot[..n]);
    }

    /// Read back slot `slot`'s pass timestamps (non-blocking: keeps the old values if not ready).
    pub fn read_timings(&mut self, ctx: &crate::gfx::context::Context, slot: usize) {
        let sl = &self.slots[slot & 1];
        let n = sl.ts_count as usize;
        if n < 2 {
            return;
        }
        let mut ticks = vec![0u64; n];
        let ok = unsafe { ctx.device.get_query_pool_results(sl.ts, 0, &mut ticks, ash::vk::QueryResultFlags::TYPE_64) };
        if ok.is_err() {
            return;
        }
        self.pass_ms = (0..n - 1).map(|i| (sl.names[i], (ticks[i + 1].wrapping_sub(ticks[i])) as f32 * self.ts_period_ns * 1e-6)).collect();
    }
}
