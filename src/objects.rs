//! Dynamic voxel bodies: their own sparse volume plus a rigid transform.

use crate::math::{mat3_rot_x, mat3_rot_y, Mat3, Quat, Transform, Vec3};
use crate::voxel::sparse_volume::{SparseVolume, BRICK};
use crate::voxel::world::material_density;

#[derive(Clone)]
pub struct VoxObject {
    pub vol: SparseVolume,
    /// Maps local voxel coordinates (0..nx etc.) to world space.
    pub pose: Transform,
    /// Stable identity (undo, selection); 0 = not assigned yet.
    pub id: u32,
    // Demo animation parameters (kinematic test bodies).
    pub center: Vec3,
    pub spin_speed: f32,
    pub tilt: f32,
    pub bob_amp: f32,
    pub bob_phase: f32,
    // Rigid-body state (rebuilt by init_body when the volume's content changes).
    pub rot_q: Quat,
    pub vel: Vec3,
    pub ang_vel: Vec3,
    pub mass: f32,
    pub inv_mass: f32,
    /// Share of its glowing voxels still there (< 0.9 = damaged: dims and sputters).
    pub glow_integrity: f32,
    /// Centre of mass, local coords.
    pub com: Vec3,
    pub inv_inertia_local: Mat3,
    /// Packed surface voxels: x | y<<10 | z<<20.
    pub shell: Vec<u32>,
    pub asleep: bool,
    pub sleep_timer: f32,
}

/// Result of [`VoxObject::ensure_contains`]: `shift` is the offset added to existing content
/// (negative-side growth); `grew` = the volume changed size (the GPU buffers need a rebuild).
#[derive(Clone, Copy, Debug, Default)]
pub struct GrowResult {
    pub shift: Vec3,
    pub grew: bool,
}

impl VoxObject {
    pub fn new(nx: i32, ny: i32, nz: i32) -> Self {
        Self {
            vol: SparseVolume::new(nx, ny, nz),
            pose: Transform::default(),
            id: 0,
            center: Vec3::ZERO,
            spin_speed: 0.0,
            tilt: 0.0,
            bob_amp: 0.0,
            bob_phase: 0.0,
            rot_q: Quat::IDENTITY,
            vel: Vec3::ZERO,
            ang_vel: Vec3::ZERO,
            mass: 0.0,
            inv_mass: 0.0,
            glow_integrity: 1.0,
            com: Vec3::ZERO,
            inv_inertia_local: Mat3::IDENTITY,
            shell: Vec::new(),
            asleep: false,
            sleep_timer: 0.0,
        }
    }

    fn local_center(&self) -> Vec3 {
        Vec3::new(self.vol.nx as f32, self.vol.ny as f32, self.vol.nz as f32) * 0.5
    }

    /// Pose so the volume spins / bobs around `center`.
    pub fn animate(&mut self, t: f32) {
        let rot = mat3_rot_y(self.spin_speed * t) * mat3_rot_x(self.tilt);
        let mut pos = self.center;
        pos.y += self.bob_amp * (t * 0.8 + self.bob_phase).sin();
        self.pose.r = rot;
        self.pose.t = pos - rot * self.local_center();
    }

    /// Grow the volume (whole bricks, capped at 64 per axis) so the local box [lo, hi] fits,
    /// keeping existing content where it is in the world.
    pub fn ensure_contains(&mut self, lo: Vec3, hi: Vec3) -> GrowResult {
        const CAP: i32 = 64;
        let (lof, hif) = (lo.to_array(), hi.to_array());
        let dims = [self.vol.nx, self.vol.ny, self.vol.nz];
        let bdims = [self.vol.bricks_x, self.vol.bricks_y, self.vol.bricks_z];
        let (mut g0, mut g1) = ([0i32; 3], [0i32; 3]);
        for a in 0..3 {
            if lof[a] < 0.0 {
                g0[a] = (-lof[a] / BRICK as f32).ceil() as i32;
            }
            if hif[a] > dims[a] as f32 {
                g1[a] = ((hif[a] - dims[a] as f32) / BRICK as f32).ceil() as i32;
            }
            let mut over = bdims[a] + g0[a] + g1[a] - CAP;
            if over > 0 {
                let r = over.min(g1[a]);
                g1[a] -= r;
                over -= r;
                g0[a] = (g0[a] - over).max(0);
            }
        }
        if g0 == [0; 3] && g1 == [0; 3] {
            return GrowResult::default();
        }
        let old_lc = self.local_center();
        self.vol.grow_bricks(g0, g1);
        let d = Vec3::new((BRICK * g0[0]) as f32, (BRICK * g0[1]) as f32, (BRICK * g0[2]) as f32);
        let new_lc = self.local_center();
        self.center += self.pose.r * (new_lc - old_lc - d);
        self.pose.t -= self.pose.r * d;
        GrowResult { shift: d, grew: true }
    }

    pub fn world_aabb(&self) -> (Vec3, Vec3) {
        let (mut mn, mut mx) = (Vec3::splat(1e30), Vec3::splat(-1e30));
        for i in 0..8 {
            let c = Vec3::new(
                if i & 1 != 0 { self.vol.nx as f32 } else { 0.0 },
                if i & 2 != 0 { self.vol.ny as f32 } else { 0.0 },
                if i & 4 != 0 { self.vol.nz as f32 } else { 0.0 },
            );
            let w = self.pose.apply(c);
            mn = mn.min(w);
            mx = mx.max(w);
        }
        (mn, mx)
    }

    /// Floating origin: world coordinates moved by (dx, dz) voxels.
    pub fn shift_world(&mut self, dx: f32, dz: f32) {
        self.pose.t.x -= dx;
        self.pose.t.z -= dz;
        self.center.x -= dx;
        self.center.z -= dz;
    }

    /// Mass, centre of mass, inertia and the surface shell from the voxels (rigid body setup).
    pub fn init_body(&mut self) {
        let v = &self.vol;
        let (mut m, mut c) = (0.0f64, [0.0f64; 3]);
        let mut cov = [[0.0f64; 3]; 3];
        let mut shell = Vec::new();
        for z in 0..v.nz {
            for y in 0..v.ny {
                for x in 0..v.nx {
                    let mat = v.get_voxel(x, y, z);
                    if mat == 0 {
                        continue;
                    }
                    let d = material_density(mat) as f64;
                    let p = [x as f64 + 0.5, y as f64 + 0.5, z as f64 + 0.5];
                    m += d;
                    for a in 0..3 {
                        c[a] += d * p[a];
                        for b in 0..3 {
                            cov[a][b] += d * p[a] * p[b];
                        }
                    }
                    let exposed = [(1, 0, 0), (-1, 0, 0), (0, 1, 0), (0, -1, 0), (0, 0, 1), (0, 0, -1)]
                        .iter()
                        .any(|&(dx, dy, dz)| v.get_voxel(x + dx, y + dy, z + dz) == 0);
                    if exposed {
                        shell.push(x as u32 | (y as u32) << 10 | (z as u32) << 20);
                    }
                }
            }
        }
        self.shell = shell;
        if m <= 0.0 {
            self.mass = 0.0;
            self.inv_mass = 0.0;
            return;
        }
        let com = [c[0] / m, c[1] / m, c[2] / m];
        // Inertia about the centre of mass (+ each voxel's own 1/6 m a^2 term).
        let mut i = [[0.0f64; 3]; 3];
        let tr = (0..3).map(|a| cov[a][a] - m * com[a] * com[a]).sum::<f64>();
        for a in 0..3 {
            for b in 0..3 {
                let cab = cov[a][b] - m * com[a] * com[b];
                i[a][b] = if a == b { tr - cab + m / 6.0 } else { -cab };
            }
        }
        let im = Mat3::from_cols(
            Vec3::new(i[0][0] as f32, i[1][0] as f32, i[2][0] as f32),
            Vec3::new(i[0][1] as f32, i[1][1] as f32, i[2][1] as f32),
            Vec3::new(i[0][2] as f32, i[1][2] as f32, i[2][2] as f32),
        );
        self.mass = m as f32;
        self.inv_mass = 1.0 / m as f32;
        self.com = Vec3::new(com[0] as f32, com[1] as f32, com[2] as f32);
        self.inv_inertia_local = if im.determinant().abs() > 1e-12 { im.inverse() } else { Mat3::IDENTITY };
        self.rot_q = Quat::from_mat3(&self.pose.r);
    }
}

pub fn make_sphere_object(radius: i32, mat: u8) -> VoxObject {
    let n = ((radius * 2 + 2 + 7) / 8) * 8;
    let mut o = VoxObject::new(n, n, n);
    let c = n as f32 * 0.5;
    o.vol.fill_sphere(c, c, c, radius as f32, mat);
    o.vol.compact();
    o
}

/// A wood box with plank stripes.
pub fn make_crate(size: i32) -> VoxObject {
    let n = ((size + 7) / 8) * 8;
    let mut o = VoxObject::new(n, n, n);
    const WOOD: u8 = 6;
    const DARK: u8 = 2;
    for z in 0..size {
        for y in 0..size {
            for x in 0..size {
                let stripe = y % 8 == 0 || (x % 8 == 0 && z % 8 == 0);
                o.vol.set_voxel(x, y, z, if stripe { DARK } else { WOOD });
            }
        }
    }
    o.vol.compact();
    o
}

/// A w x h x w column.
pub fn make_pillar(w: i32, h: i32, mat: u8) -> VoxObject {
    let (nw, nh) = (((w + 7) / 8) * 8, ((h + 7) / 8) * 8);
    let mut o = VoxObject::new(nw, nh, nw);
    o.vol.fill_box(0, 0, 0, w - 1, h - 1, w - 1, mat);
    o.vol.compact();
    o
}

/// The 7-body self-test layout around the central tower (crates / spheres / pillars, posed at
/// rest) - the C++ engine's startup scene.
pub fn build_self_test_objects(world: &crate::voxel::world::World) -> Vec<VoxObject> {
    let (tx, tz) = (world.mid_x(), world.mid_z());
    let ground_at = |x: f32, z: f32| {
        let rh = world.raycast(Vec3::new(x, world.ny as f32 - 2.0, z), Vec3::new(0.0, -1.0, 0.0), world.ny as f32, 0);
        if rh.hit { rh.cell[1] as f32 } else { 232.0 }
    };
    let mut objects: Vec<VoxObject> = Vec::new();
    let mut add = |mut o: VoxObject, dx: f32, dz: f32, hover: f32, spin: f32, tilt: f32, bob: f32| {
        let (x, z) = (tx + dx, tz + dz);
        o.center = Vec3::new(x, ground_at(x, z) + hover, z);
        o.spin_speed = spin;
        o.tilt = tilt;
        o.bob_amp = bob;
        o.bob_phase = objects.len() as f32 * 1.3;
        o.animate(0.0);
        objects.push(o);
    };
    add(make_crate(24), -165.0, -97.0, 40.0, 0.6, 0.0, 4.0);
    add(make_crate(36), 180.0, 18.0, 55.0, -0.35, 0.4, 6.0);
    add(make_sphere_object(20, 17), -97.0, 180.0, 60.0, 1.2, 0.0, 5.0);
    add(make_sphere_object(28, 21), 76.0, 225.0, 100.0, 0.8, 0.3, 8.0);
    add(make_sphere_object(36, 19), 225.0, -165.0, 130.0, -0.5, 0.2, 10.0);
    add(make_pillar(24, 112, 3), -188.0, 76.0, 90.0, 0.9, 0.5, 5.0);
    add(make_pillar(16, 60, 8), 18.0, -188.0, 70.0, -1.4, 0.9, 7.0);
    for (i, o) in objects.iter_mut().enumerate() {
        o.init_body();
        o.id = i as u32 + 1;
    }
    objects
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn body_mass_and_growth() {
        let mut o = make_pillar(8, 16, 3);
        o.init_body();
        assert!((o.mass - 8.0 * 16.0 * 8.0 * 2.6).abs() < 1e-2);
        assert!((o.com - Vec3::new(4.0, 8.0, 4.0)).length() < 1e-3);
        let before = o.pose.apply(Vec3::new(1.0, 1.0, 1.0));
        let g = o.ensure_contains(Vec3::new(-3.0, 0.0, 0.0), Vec3::new(8.0, 30.0, 8.0));
        assert!(g.grew && g.shift.x == 8.0);
        let after = o.pose.apply(Vec3::new(1.0, 1.0, 1.0) + g.shift);
        assert!((before - after).length() < 1e-4);
        assert_eq!(o.vol.get_voxel(9, 1, 1), 3);
    }
}
