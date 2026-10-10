//! Edit brush: the exact set of voxels a carve / place / paint touches. Shared by the edit, its
//! tests and - mirrored in shaders/brush_preview.comp - the on-screen preview, so what you see is
//! what the click changes.
//!
//! A brush is a box of `size` cells per axis at integer corner `lo`; the cube is the whole box,
//! the sphere keeps the cells whose centre lies within size/2 of the box centre. Size 1 = exactly
//! one voxel for either shape.

use crate::voxel::sparse_volume::SparseVolume;

const WATER: u8 = 5;

#[derive(Clone, Copy, Debug, PartialEq, Eq, Default)]
#[repr(i32)]
pub enum BrushShape {
    #[default]
    Sphere = 0,
    Cube = 1,
}

#[derive(Clone, Copy, Debug, PartialEq)]
pub struct BrushBox {
    pub lo: [i32; 3],
    pub size: i32,
    pub shape: BrushShape,
}

impl BrushBox {
    pub fn centre(&self, a: usize) -> f32 {
        self.lo[a] as f32 + self.size as f32 * 0.5
    }

    pub fn contains(&self, x: i32, y: i32, z: i32) -> bool {
        let c = [x, y, z];
        if (0..3).any(|a| c[a] < self.lo[a] || c[a] >= self.lo[a] + self.size) {
            return false;
        }
        if self.shape == BrushShape::Cube || self.size <= 2 {
            return true; // 1 / 2 across: every cell
        }
        let r = self.size as f32 * 0.5;
        let d2: f32 = (0..3).map(|a| (c[a] as f32 + 0.5 - self.centre(a)).powi(2)).sum();
        d2 <= r * r
    }

    /// Every cell of the brush, x fastest.
    pub fn cells(&self) -> impl Iterator<Item = (i32, i32, i32)> + '_ {
        let (lo, s) = (self.lo, self.size);
        (lo[2]..lo[2] + s)
            .flat_map(move |z| (lo[1]..lo[1] + s).flat_map(move |y| (lo[0]..lo[0] + s).map(move |x| (x, y, z))))
            .filter(|&(x, y, z)| self.contains(x, y, z))
    }

    /// Voxel bounds, inclusive.
    pub fn bounds(&self) -> ([i32; 3], [i32; 3]) {
        (self.lo, self.lo.map(|v| v + self.size - 1))
    }
}

/// Where a click lands. `hit` = the voxel the aim ray hit, `normal` = the face it entered
/// through. Carve: the brush is centred on the hit voxel (it digs into the surface). Place: it
/// starts at the empty cell in front of that face and extends AWAY from the surface along the
/// normal (centred across it), so it sits on the surface instead of sinking half into it.
pub fn brush_for(hit: [i32; 3], normal: [i32; 3], size: i32, shape: BrushShape, place: bool) -> BrushBox {
    let size = size.max(1);
    let half = (size - 1) / 2; // cells before the centre cell (odd sizes centred)
    let lo = std::array::from_fn(|a| {
        if !place {
            return hit[a] - half;
        }
        let anchor = hit[a] + normal[a];
        match normal[a].signum() {
            1 => anchor,
            -1 => anchor - (size - 1),
            _ => anchor - half,
        }
    });
    BrushBox { lo, size, shape }
}

/// A cell an edit may touch: inside a bounded world; endless: any loaded column.
pub fn in_brush_world(v: &SparseVolume, x: i32, y: i32, z: i32) -> bool {
    if (y as u32) >= v.ny as u32 {
        return false;
    }
    if v.ring_w != 0 { v.loaded_at(x, z) } else { x >= 0 && z >= 0 && x < v.nx && z < v.nz }
}

/// Carve (m = 0) clears the SOLID brush cells (water can't be dug away - it's a fluid); place
/// fills only EMPTY ones - air or water - so a placed shape never replaces terrain. `on_water`
/// is called before a water cell is overwritten by a placed voxel (the fluid sim displaces its
/// volume). Returns the number of voxels changed.
pub fn apply_brush(v: &mut SparseVolume, b: &BrushBox, m: u8, mut on_water: impl FnMut(&mut SparseVolume, i32, i32, i32)) -> usize {
    let mut n = 0;
    for (x, y, z) in b.cells() {
        if !in_brush_world(v, x, y, z) {
            continue;
        }
        let cur = v.get_voxel(x, y, z);
        let empty = cur == 0 || cur == WATER;
        if (m == 0) == empty {
            continue;
        }
        if cur == WATER {
            on_water(v, x, y, z);
        }
        v.set_voxel(x, y, z, m);
        n += 1;
    }
    n
}

/// Paint: recolour the SOLID brush cells (never air, water or a torch's parts) to `m`.
pub fn apply_paint(v: &mut SparseVolume, b: &BrushBox, m: u8) -> usize {
    let mut n = 0;
    for (x, y, z) in b.cells() {
        if !in_brush_world(v, x, y, z) {
            continue;
        }
        let cur = v.get_voxel(x, y, z);
        if cur == 0 || cur == WATER || cur == m || (248..=254).contains(&cur) {
            continue;
        }
        v.set_voxel(x, y, z, m);
        n += 1;
    }
    n
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn shapes_and_anchoring() {
        let s = BrushBox { lo: [0, 0, 0], size: 5, shape: BrushShape::Sphere };
        assert!(s.contains(2, 2, 2) && !s.contains(0, 0, 0));
        let c = BrushBox { shape: BrushShape::Cube, ..s };
        assert_eq!(c.cells().count(), 125);
        assert!(s.cells().count() < 125);
        assert_eq!(BrushBox { lo: [3, 3, 3], size: 1, shape: BrushShape::Sphere }.cells().count(), 1);
        // Place on a floor (hit y = 9, normal +y): the brush starts at y 10 and grows up.
        let p = brush_for([5, 9, 5], [0, 1, 0], 3, BrushShape::Cube, true);
        assert_eq!(p.lo, [4, 10, 4]);
        let p = brush_for([5, 9, 5], [0, -1, 0], 3, BrushShape::Cube, true);
        assert_eq!(p.lo[1], 6);
        assert_eq!(brush_for([5, 9, 5], [0, 1, 0], 3, BrushShape::Cube, false).lo, [4, 8, 4]);
    }

    #[test]
    fn carve_place_paint_rules() {
        let mut v = SparseVolume::new(32, 32, 32);
        v.fill_box(0, 0, 0, 31, 9, 31, 3);
        v.fill_box(0, 10, 0, 31, 11, 31, WATER);
        let b = BrushBox { lo: [4, 8, 4], size: 4, shape: BrushShape::Cube };
        let mut wet = 0;
        let n = apply_brush(&mut v, &b, 9, |_, _, _, _| wet += 1);
        assert_eq!((n, wet), (32, 32), "place fills only the water cells (2 layers of 16)");
        assert_eq!(v.get_voxel(5, 9, 5), 3, "terrain kept");
        assert_eq!(apply_brush(&mut v, &b, 0, |_, _, _, _| {}), 64, "carve: solid only");
        assert_eq!(v.get_voxel(5, 9, 5), 0);
        v.fill_box(4, 8, 4, 7, 9, 7, 3);
        assert_eq!(apply_paint(&mut v, &b, 12), 32);
        assert_eq!(v.get_voxel(5, 8, 5), 12);
    }
}
