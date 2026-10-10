//! Free-fly camera: yaw around +Y, pitch clamped short of the poles.

use crate::math::Vec3;

#[derive(Clone, Copy, Debug, PartialEq)]
pub struct FlyCamera {
    pub pos: Vec3,
    /// Radians; 0 looks toward +Z.
    pub yaw: f32,
    pub pitch: f32,
    pub move_speed: f32,
    pub mouse_sens: f32,
}

impl Default for FlyCamera {
    fn default() -> Self {
        Self { pos: Vec3::new(80.0, 220.0, 80.0), yaw: 0.8, pitch: -0.35, move_speed: 80.0, mouse_sens: 0.0026 }
    }
}

/// Movement keys held this frame.
#[derive(Clone, Copy, Debug, Default)]
pub struct MoveInput {
    pub forward: bool,
    pub back: bool,
    pub left: bool,
    pub right: bool,
    pub up: bool,
    pub down: bool,
    pub fast: bool,
}

impl FlyCamera {
    pub fn forward(&self) -> Vec3 {
        Vec3::new(self.pitch.cos() * self.yaw.sin(), self.pitch.sin(), self.pitch.cos() * self.yaw.cos())
    }
    pub fn right(&self) -> Vec3 {
        crate::math::normalize_or_up(self.forward().cross(Vec3::Y))
    }
    pub fn up(&self) -> Vec3 {
        self.right().cross(self.forward())
    }

    pub fn look(&mut self, dx: f32, dy: f32) {
        self.yaw -= dx * self.mouse_sens;
        self.pitch = (self.pitch - dy * self.mouse_sens).clamp(-1.55, 1.55);
    }

    /// WASD in the ground plane, Space / Ctrl vertical, Shift = 4x fast.
    pub fn fly(&mut self, k: &MoveInput, dt: f32) {
        let f = self.forward();
        let flat = Vec3::new(f.x, 0.0, f.z).normalize_or_zero();
        let r = self.right();
        let mut d = Vec3::ZERO;
        if k.forward {
            d += flat;
        }
        if k.back {
            d -= flat;
        }
        if k.right {
            d += r;
        }
        if k.left {
            d -= r;
        }
        if k.up {
            d += Vec3::Y;
        }
        if k.down {
            d -= Vec3::Y;
        }
        let speed = self.move_speed * if k.fast { 4.0 } else { 1.0 };
        self.pos += d.normalize_or_zero() * speed * dt;
    }
}
