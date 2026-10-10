//! Math types: glam (SIMD) for vectors / matrices, plus the engine's rigid transform.
//! glam's `Mat3` / `Mat4` are column-major like the C++ `mat3` / `mat4`, so GPU uploads need
//! no reordering.

pub use glam::{Mat3, Quat, Vec3};

/// Rigid transform mapping local points to world: p' = R * p + t.
#[derive(Clone, Copy, Debug, PartialEq)]
pub struct Transform {
    pub r: Mat3,
    pub t: Vec3,
}

impl Default for Transform {
    fn default() -> Self {
        Self { r: Mat3::IDENTITY, t: Vec3::ZERO }
    }
}

impl Transform {
    #[inline]
    pub fn apply(&self, p: Vec3) -> Vec3 {
        self.r * p + self.t
    }
    pub fn inverse(&self) -> Self {
        let rt = self.r.transpose();
        Self { r: rt, t: -(rt * self.t) }
    }
    /// Column-major 4x4 for GPU upload (std430 mat4).
    pub fn to_mat4(&self) -> [f32; 16] {
        let c = self.r.to_cols_array();
        [c[0], c[1], c[2], 0.0, c[3], c[4], c[5], 0.0, c[6], c[7], c[8], 0.0, self.t.x, self.t.y, self.t.z, 1.0]
    }
}

/// Rotation about +Y by `a` radians (the C++ `mat3RotY`).
pub fn mat3_rot_y(a: f32) -> Mat3 {
    Mat3::from_rotation_y(a)
}

/// Rotation about +X by `a` radians (the C++ `mat3RotX`).
pub fn mat3_rot_x(a: f32) -> Mat3 {
    Mat3::from_rotation_x(a)
}

/// Normalize; a zero vector becomes +Y (as the C++ `normalize`).
#[inline]
pub fn normalize_or_up(v: Vec3) -> Vec3 {
    let l = v.length();
    if l > 0.0 { v / l } else { Vec3::Y }
}

/// IEEE half -> f32 (reading back RG16F / RGBA16F images).
pub fn half_to_f32(h: u16) -> f32 {
    let sign = ((h as u32) & 0x8000) << 16;
    let exp = ((h >> 10) & 0x1F) as u32;
    let mut mant = (h & 0x3FF) as u32;
    let bits = if exp == 0 {
        if mant == 0 {
            sign
        } else {
            let mut e = 127 - 15 + 1;
            while mant & 0x400 == 0 {
                mant <<= 1;
                e -= 1;
            }
            sign | (e << 23) | ((mant & 0x3FF) << 13)
        }
    } else if exp == 0x1F {
        sign | 0x7F80_0000 | (mant << 13)
    } else {
        sign | ((exp + 127 - 15) << 23) | (mant << 13)
    };
    f32::from_bits(bits)
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn rotations_match_cpp_convention() {
        // C++ mat3RotY: m[0]=c, m[2]=-s, m[6]=s, m[8]=c (column-major).
        let a = 0.7f32;
        let m = mat3_rot_y(a).to_cols_array();
        assert!((m[0] - a.cos()).abs() < 1e-6 && (m[2] + a.sin()).abs() < 1e-6);
        assert!((m[6] - a.sin()).abs() < 1e-6 && (m[8] - a.cos()).abs() < 1e-6);
        // C++ mat3RotX: m[4]=c, m[5]=s, m[7]=-s, m[8]=c.
        let m = mat3_rot_x(a).to_cols_array();
        assert!((m[5] - a.sin()).abs() < 1e-6 && (m[7] + a.sin()).abs() < 1e-6);
    }

    #[test]
    fn transform_inverse() {
        let t = Transform { r: mat3_rot_y(0.3) * mat3_rot_x(0.2), t: Vec3::new(1.0, 2.0, 3.0) };
        let p = Vec3::new(4.0, -5.0, 6.0);
        assert!((t.inverse().apply(t.apply(p)) - p).length() < 1e-5);
    }

    #[test]
    fn halves() {
        assert_eq!(half_to_f32(0x3C00), 1.0);
        assert_eq!(half_to_f32(0xC000), -2.0);
        assert_eq!(half_to_f32(0x0000), 0.0);
        assert!((half_to_f32(0x0001) - 5.96e-8).abs() < 1e-9);
    }
}
