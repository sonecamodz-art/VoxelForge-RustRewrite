//! Hardware ray queries: the world as acceleration structures for the RT cores.
//!
//! A TILE = one 16 x 16 brick column of the grid (a streamed chunk's ring slot; a bounded world
//! is cut into the same tiles). Each tile has a BLAS of AABB primitives, one per brick a ray can
//! meet: detailed bricks, water / foliage / light-passing uniform bricks, and solid uniform
//! bricks with a face open to a brick that isn't solid (buried stone is left out). The BLAS is in
//! the tile's local voxel coordinates; the TLAS places it. `rt_prims[custom index + primitive]` =
//! the brick's local coordinates (x | z << 4 | y << 8): the shader (common.glsl traceWorldRQF)
//! looks the brick up in the grid and walks its voxels with the same code as the DDA.
//!
//! Storage: a buffer per BLAS (gpu-allocator sub-allocates their memory, so there's no
//! allocation per BLAS - the C++ engine kept its own arena for that), the primitive table in
//! one SSBO (sub-allocated ranges). A BLAS replaced or released is freed only once the frames
//! that may still trace it are done. Builds are recorded into the world's sync command buffers;
//! the TLAS at the top of each frame.

use std::collections::BTreeMap;

use ash::vk;
use gpu_allocator::MemoryLocation;
use rayon::prelude::*;

use super::context::{memory_barrier, Access, Buffer, Context, Stage};
use crate::voxel::sparse_volume::{SparseVolume, RING_CHUNK_B};
use crate::voxel::world::{wm, Material};

const TILE_B: i32 = RING_CHUNK_B;
/// Primitive table entries (32 MB).
const PRIM_CAP: u64 = 8 << 20;
/// Frees wait this many TLAS builds; also the instance buffers in rotation.
const FRAMES_IN_FLIGHT: u64 = 3;

/// First-fit range allocator (offsets in some unit; merges on free).
#[derive(Default)]
struct Ranges {
    free: BTreeMap<u64, u64>,
}

impl Ranges {
    fn reset(&mut self, size: u64) {
        self.free.clear();
        self.free.insert(0, size);
    }
    fn alloc(&mut self, size: u64) -> Option<u64> {
        let (&start, &len) = self.free.iter().find(|&(_, &len)| len >= size)?;
        self.free.remove(&start);
        if len > size {
            self.free.insert(start + size, len - size);
        }
        Some(start)
    }
    fn release(&mut self, off: u64, size: u64) {
        let (mut off, mut size) = (off, size);
        if let Some((&p, &pl)) = self.free.range(..off).next_back()
            && p + pl == off
        {
            self.free.remove(&p);
            off = p;
            size += pl;
        }
        if let Some(&nl) = self.free.get(&(off + size)) {
            self.free.remove(&(off + size));
            size += nl;
        }
        self.free.insert(off, size);
    }
}

#[derive(Default)]
struct Tile {
    as_: vk::AccelerationStructureKHR,
    buf: Option<Buffer>,
    prim_off: u64,
    prims: u32,
    addr: u64,
    /// The tile's corner in the engine's coordinates (voxels): the instance's translation.
    ox: i32,
    oz: i32,
}

#[derive(Default)]
pub struct RtScene {
    active: bool,
    wbx: i32,
    wby: i32,
    wbz: i32,
    ring_w: i32,
    tiles_x: i32,
    tiles: Vec<Tile>,
    dead: Vec<(u64, Tile)>,
    frame: u64,
    prim_buf: Buffer,
    prim_ranges: Ranges,
    scratch: Buffer,
    tlas: vk::AccelerationStructureKHR,
    tlas_buf: Buffer,
    tlas_scratch: Buffer,
    tlas_cap: u32,
    inst: Vec<Buffer>,
    /// r.rt.tightboxes when the tiles were built.
    tight: bool,
}

fn align_up(v: u64, a: u64) -> u64 {
    v.div_ceil(a) * a
}

impl RtScene {
    pub fn active(&self) -> bool {
        self.active
    }
    pub fn tlas(&self) -> vk::AccelerationStructureKHR {
        self.tlas
    }
    pub fn prim_buffer(&self) -> vk::Buffer {
        self.prim_buf.buf
    }
    pub fn built_tiles(&self) -> usize {
        self.tiles.iter().filter(|t| t.as_ != vk::AccelerationStructureKHR::null()).count()
    }
    pub fn primitives(&self) -> u64 {
        self.tiles.iter().map(|t| t.prims as u64).sum()
    }
    pub fn memory_bytes(&self) -> u64 {
        let bufs = self.tiles.iter().filter_map(|t| t.buf.as_ref()).map(|b| b.size).sum::<u64>();
        bufs + self.prim_buf.size + self.scratch.size + self.tlas_buf.size + self.tlas_scratch.size
    }

    /// (Re)creates everything for a world of wbx x wby x wbz bricks (ring_w > 0: a streaming
    /// ring of ring_w^2 tiles; else bounded tiles). No builds yet.
    pub fn init(&mut self, ctx: &Context, wbx: i32, wby: i32, wbz: i32, ring_w: i32) {
        self.destroy(ctx);
        let Some(acc) = &ctx.accel else { return };
        (self.wbx, self.wby, self.wbz, self.ring_w) = (wbx, wby, wbz, ring_w);
        self.tiles_x = if ring_w > 0 { ring_w } else { (wbx + TILE_B - 1) / TILE_B };
        let tiles_z = if ring_w > 0 { ring_w } else { (wbz + TILE_B - 1) / TILE_B };
        self.tiles = (0..self.tiles_x * tiles_z).map(|_| Tile::default()).collect();
        let addr = vk::BufferUsageFlags::SHADER_DEVICE_ADDRESS;
        self.prim_buf = ctx.create_buffer("rtPrims", PRIM_CAP * 4, vk::BufferUsageFlags::STORAGE_BUFFER | vk::BufferUsageFlags::TRANSFER_DST,
                                          MemoryLocation::GpuOnly);
        self.prim_ranges.reset(PRIM_CAP);
        // The TLAS: every tile (+ room for bodies later).
        self.tlas_cap = self.tiles.len() as u32 + 1024;
        let geo = [vk::AccelerationStructureGeometryKHR::default()
            .geometry_type(vk::GeometryTypeKHR::INSTANCES)
            .geometry(vk::AccelerationStructureGeometryDataKHR { instances: vk::AccelerationStructureGeometryInstancesDataKHR::default() })];
        let bi = vk::AccelerationStructureBuildGeometryInfoKHR::default()
            .ty(vk::AccelerationStructureTypeKHR::TOP_LEVEL)
            .flags(vk::BuildAccelerationStructureFlagsKHR::PREFER_FAST_TRACE)
            .geometries(&geo);
        let mut sz = vk::AccelerationStructureBuildSizesInfoKHR::default();
        unsafe { acc.get_acceleration_structure_build_sizes(vk::AccelerationStructureBuildTypeKHR::DEVICE, &bi, &[self.tlas_cap], &mut sz) };
        self.tlas_buf = ctx.create_buffer("tlas", sz.acceleration_structure_size, vk::BufferUsageFlags::ACCELERATION_STRUCTURE_STORAGE_KHR | addr,
                                          MemoryLocation::GpuOnly);
        let ci = vk::AccelerationStructureCreateInfoKHR::default()
            .buffer(self.tlas_buf.buf)
            .size(sz.acceleration_structure_size)
            .ty(vk::AccelerationStructureTypeKHR::TOP_LEVEL);
        self.tlas = super::context::vkc(unsafe { acc.create_acceleration_structure(&ci, None) }, "create TLAS");
        self.tlas_scratch = ctx.create_buffer("tlasScratch", sz.build_scratch_size + ctx.as_scratch_align,
                                              vk::BufferUsageFlags::STORAGE_BUFFER | addr, MemoryLocation::GpuOnly);
        let inst_bytes = self.tlas_cap as u64 * std::mem::size_of::<vk::AccelerationStructureInstanceKHR>() as u64;
        self.inst = (0..FRAMES_IN_FLIGHT)
            .map(|_| ctx.create_buffer("tlasInstances", inst_bytes, vk::BufferUsageFlags::ACCELERATION_STRUCTURE_BUILD_INPUT_READ_ONLY_KHR | addr,
                                       MemoryLocation::CpuToGpu))
            .collect();
        self.active = true;
        log::info!(target: "rt", "ray queries: {} tiles ({}), TLAS {:.1} MB, primitive table {:.0} MB", self.tiles.len(),
                   if ring_w > 0 { "ring slots" } else { "bounded tiles" }, sz.acceleration_structure_size as f64 / 1048576.0,
                   (PRIM_CAP * 4) as f64 / 1048576.0);
    }

    pub fn destroy(&mut self, ctx: &Context) {
        if !self.active {
            return;
        }
        ctx.wait_idle();
        for i in 0..self.tiles.len() {
            self.free_tile(i);
        }
        self.reap(ctx, true);
        let acc = ctx.accel.as_ref().unwrap();
        unsafe { acc.destroy_acceleration_structure(self.tlas, None) };
        self.tlas = vk::AccelerationStructureKHR::null();
        for b in [&mut self.tlas_buf, &mut self.tlas_scratch, &mut self.scratch, &mut self.prim_buf] {
            b.destroy(ctx);
        }
        for b in &mut self.inst {
            b.destroy(ctx);
        }
        self.inst.clear();
        self.tiles.clear();
        self.active = false;
    }

    /// The tile holding grid cell gi.
    pub fn tile_of_cell(&self, gi: u32) -> usize {
        let gi = gi as usize;
        let (bx, bz) = ((gi % self.wbx as usize) as i32, (gi / (self.wbx as usize * self.wby as usize)) as i32);
        ((bz / TILE_B) * self.tiles_x + bx / TILE_B) as usize
    }

    /// The ring tile of chunk (cx, cz).
    pub fn tile_of_chunk(&self, cx: i32, cz: i32) -> usize {
        let m = self.ring_w - 1;
        ((cz & m) * self.ring_w + (cx & m)) as usize
    }

    /// Floating origin: the engine's coordinates moved by (dx, dz) voxels.
    pub fn shift(&mut self, dx: i32, dz: i32) {
        for t in &mut self.tiles {
            t.ox -= dx;
            t.oz -= dz;
        }
    }

    fn free_tile(&mut self, i: usize) {
        let t = std::mem::take(&mut self.tiles[i]);
        if t.as_ != vk::AccelerationStructureKHR::null() {
            self.dead.push((self.frame, t));
        }
    }

    /// Frees what no frame in flight can still trace.
    fn reap(&mut self, ctx: &Context, all: bool) {
        let frame = self.frame;
        let (gone, keep): (Vec<_>, Vec<_>) = std::mem::take(&mut self.dead).into_iter().partition(|(f, _)| all || frame >= f + FRAMES_IN_FLIGHT + 1);
        self.dead = keep;
        let acc = ctx.accel.as_ref().unwrap();
        for (_, mut t) in gone {
            unsafe { acc.destroy_acceleration_structure(t.as_, None) };
            if let Some(b) = t.buf.as_mut() {
                b.destroy(ctx);
            }
            if t.prims > 0 {
                self.prim_ranges.release(t.prim_off, t.prims as u64);
            }
        }
    }

    fn ensure_scratch(ctx: &Context, buf: &mut Buffer, bytes: u64, name: &str) {
        if bytes <= buf.size {
            return;
        }
        if !buf.is_null() {
            ctx.wait_idle(); // (a build may still use the old one: growth is rare)
            buf.destroy(ctx);
        }
        let cap = align_up(bytes.max(buf.size * 2), 1 << 20);
        *buf = ctx.create_buffer(name, cap, vk::BufferUsageFlags::STORAGE_BUFFER | vk::BufferUsageFlags::SHADER_DEVICE_ADDRESS, MemoryLocation::GpuOnly);
    }

    /// Record the BLAS builds of these tiles from the CPU world (tile index: ring slot, or
    /// tz * tiles_x + tx); a tile that holds nothing now is released. `stage` copies the build
    /// input somewhere that lives until the command buffer has run, and returns the buffer, the
    /// offset and its device address.
    pub fn build_tiles(&mut self, ctx: &Context, cb: vk::CommandBuffer, world: &SparseVolume, palette: &[Material], tiles: &[usize],
                       stage: &mut dyn FnMut(&[u8]) -> (vk::Buffer, u64, u64)) {
        if !self.active || tiles.is_empty() {
            return;
        }
        self.tight = crate::cvars::R_RT_TIGHTBOXES.get();
        let mut tiles = tiles.to_vec();
        tiles.sort_unstable();
        tiles.dedup();
        // A brick hides everything behind it when it's a uniform, opaque, non-foliage material.
        let opaque = |g: u32| {
            g > 0 && g < 256 && g != wm::WATER as u32 && (g as usize) < palette.len() && {
                let m = &palette[g as usize];
                m.props[2] != 3.0 && m.props[2] != 1.0 && m.emissive <= 0.0 && g != wm::TORCH_FLAME as u32 // (lamp rays pass lamps; a flame is air)
            }
        };
        let (wbx, wby, wbz, ring) = (self.wbx, self.wby, self.wbz, self.ring_w > 0);
        let cell = |x: i32, y: i32, z: i32| -> u32 {
            if y < 0 || y >= wby {
                return 0;
            }
            let (x, z) = if ring {
                (x & (wbx - 1), z & (wbz - 1))
            } else if x < 0 || z < 0 || x >= wbx || z >= wbz {
                return 0;
            } else {
                (x, z)
            };
            world.grid[((z * wby + y) * wbx + x) as usize]
        };
        struct Job {
            tile: usize,
            boxes: Vec<vk::AabbPositionsKHR>,
            prims: Vec<u32>,
            ox: i32,
            oz: i32,
        }
        for &t in &tiles {
            if t < self.tiles.len() {
                self.free_tile(t); // (deferred: frames in flight may still trace it)
            }
        }
        let (tiles_x, tight, n_tiles) = (self.tiles_x, self.tight, self.tiles.len());
        let jobs: Vec<Job> = tiles
            .par_iter()
            .filter(|&&t| t < n_tiles)
            .filter_map(|&t| {
                let (tx, tz) = ((t as i32 % tiles_x) * TILE_B, (t as i32 / tiles_x) * TILE_B);
                let (ox, oz) = if ring {
                    let s = &world.ring_slots[t];
                    if !s.loaded {
                        return None; // a ring slot holds a resident chunk, or nothing
                    }
                    (s.cx * TILE_B * 8, s.cz * TILE_B * 8)
                } else {
                    (tx * 8, tz * 8)
                };
                let mut j = Job { tile: t, boxes: Vec::new(), prims: Vec::new(), ox, oz };
                for z in 0..TILE_B {
                    for y in 0..wby {
                        for x in 0..TILE_B {
                            let (bx, bz) = (tx + x, tz + z);
                            if !ring && (bx >= wbx || bz >= wbz) {
                                continue;
                            }
                            let g = cell(bx, y, bz);
                            if g == 0 {
                                continue;
                            }
                            if opaque(g)
                                && opaque(cell(bx - 1, y, bz))
                                && opaque(cell(bx + 1, y, bz))
                                && opaque(cell(bx, y - 1, bz))
                                && opaque(cell(bx, y + 1, bz))
                                && opaque(cell(bx, y, bz - 1))
                                && opaque(cell(bx, y, bz + 1))
                            {
                                continue; // buried: no ray from outside reaches it
                            }
                            let (x0, y0, z0) = ((x * 8) as f32, (y * 8) as f32, (z * 8) as f32);
                            let mut b = vk::AabbPositionsKHR { min_x: x0, min_y: y0, min_z: z0, max_x: x0 + 8.0, max_y: y0 + 8.0, max_z: z0 + 8.0 };
                            if tight && g >= 256 && ((g - 256) as usize) < world.bricks.len() {
                                if let Some((lo, hi)) = brick_bounds(&world.bricks[(g - 256) as usize]) {
                                    // A detailed brick's box = its occupied voxels only: a ray that misses that box can hit
                                    // none of them, so the result is the same - with fewer false candidates (each costs the
                                    // shader a voxel walk). (+0.01 voxel: a ray grazing exactly along a face isn't culled.)
                                    let e = 0.01;
                                    b.min_x = b.min_x.max(x0 + lo[0] as f32 - e);
                                    b.max_x = b.max_x.min(x0 + hi[0] as f32 + 1.0 + e);
                                    b.min_y = b.min_y.max(y0 + lo[1] as f32 - e);
                                    b.max_y = b.max_y.min(y0 + hi[1] as f32 + 1.0 + e);
                                    b.min_z = b.min_z.max(z0 + lo[2] as f32 - e);
                                    b.max_z = b.max_z.min(z0 + hi[2] as f32 + 1.0 + e);
                                }
                            }
                            j.boxes.push(b);
                            j.prims.push(x as u32 | (z as u32) << 4 | (y as u32) << 8);
                        }
                    }
                }
                (!j.boxes.is_empty()).then_some(j)
            })
            .collect();
        self.reap(ctx, false);
        if jobs.is_empty() {
            return;
        }
        let acc = ctx.accel.as_ref().unwrap();
        // One input upload: every job's AABBs, then its primitive words (copied into rt_prims).
        let mut packed: Vec<u8> = Vec::new();
        let mut box_off = Vec::with_capacity(jobs.len());
        for j in &jobs {
            box_off.push(packed.len() as u64);
            packed.extend_from_slice(aabb_bytes(&j.boxes));
        }
        let mut prim_src = Vec::with_capacity(jobs.len());
        for j in &jobs {
            prim_src.push(packed.len() as u64);
            packed.extend_from_slice(bytemuck::cast_slice(&j.prims));
        }
        let (in_buf, in_off, in_addr) = stage(&packed);

        let geos: Vec<[vk::AccelerationStructureGeometryKHR; 1]> = jobs
            .iter()
            .zip(&box_off)
            .map(|(_, &o)| {
                [vk::AccelerationStructureGeometryKHR::default()
                    .geometry_type(vk::GeometryTypeKHR::AABBS)
                    .flags(vk::GeometryFlagsKHR::OPAQUE)
                    .geometry(vk::AccelerationStructureGeometryDataKHR {
                        aabbs: vk::AccelerationStructureGeometryAabbsDataKHR::default()
                            .data(vk::DeviceOrHostAddressConstKHR { device_address: in_addr + o })
                            .stride(std::mem::size_of::<vk::AabbPositionsKHR>() as u64),
                    })]
            })
            .collect();
        let mut infos = Vec::new();
        let mut ranges = Vec::new();
        let mut scratch_off = Vec::new();
        let mut copies = Vec::new();
        let mut scratch_total = 0u64;
        for (i, j) in jobs.iter().enumerate() {
            let bi = vk::AccelerationStructureBuildGeometryInfoKHR::default()
                .ty(vk::AccelerationStructureTypeKHR::BOTTOM_LEVEL)
                .flags(vk::BuildAccelerationStructureFlagsKHR::PREFER_FAST_TRACE)
                .mode(vk::BuildAccelerationStructureModeKHR::BUILD)
                .geometries(&geos[i]);
            let n = j.boxes.len() as u32;
            let mut sz = vk::AccelerationStructureBuildSizesInfoKHR::default();
            unsafe { acc.get_acceleration_structure_build_sizes(vk::AccelerationStructureBuildTypeKHR::DEVICE, &bi, &[n], &mut sz) };
            let Some(prim_off) = self.prim_ranges.alloc(n as u64) else {
                log::warn!(target: "rt", "out of primitive table space: tile {} not built", j.tile);
                continue;
            };
            let buf = ctx.create_buffer("blas", sz.acceleration_structure_size,
                                        vk::BufferUsageFlags::ACCELERATION_STRUCTURE_STORAGE_KHR | vk::BufferUsageFlags::SHADER_DEVICE_ADDRESS,
                                        MemoryLocation::GpuOnly);
            let ci = vk::AccelerationStructureCreateInfoKHR::default()
                .buffer(buf.buf)
                .size(sz.acceleration_structure_size)
                .ty(vk::AccelerationStructureTypeKHR::BOTTOM_LEVEL);
            let as_ = super::context::vkc(unsafe { acc.create_acceleration_structure(&ci, None) }, "create BLAS");
            let addr = unsafe { acc.get_acceleration_structure_device_address(&vk::AccelerationStructureDeviceAddressInfoKHR::default().acceleration_structure(as_)) };
            self.tiles[j.tile] = Tile { as_, buf: Some(buf), prim_off, prims: n, addr, ox: j.ox, oz: j.oz };
            infos.push(bi.dst_acceleration_structure(as_));
            ranges.push([vk::AccelerationStructureBuildRangeInfoKHR::default().primitive_count(n)]);
            scratch_off.push(scratch_total);
            scratch_total = align_up(scratch_total + sz.build_scratch_size, ctx.as_scratch_align);
            copies.push(vk::BufferCopy { src_offset: in_off + prim_src[i], dst_offset: prim_off * 4, size: n as u64 * 4 });
        }
        Self::ensure_scratch(ctx, &mut self.scratch, scratch_total + ctx.as_scratch_align, "blasScratch");
        let base = align_up(ctx.buffer_address(self.scratch.buf), ctx.as_scratch_align);
        for (bi, &o) in infos.iter_mut().zip(&scratch_off) {
            bi.scratch_data = vk::DeviceOrHostAddressKHR { device_address: base + o };
        }
        let dev = &ctx.device;
        if !copies.is_empty() {
            unsafe { dev.cmd_copy_buffer(cb, in_buf, self.prim_buf.buf, &copies) };
        }
        if !infos.is_empty() {
            let range_refs: Vec<&[vk::AccelerationStructureBuildRangeInfoKHR]> = ranges.iter().map(|r| &r[..]).collect();
            unsafe { acc.cmd_build_acceleration_structures(cb, &infos, &range_refs) };
        }
        // Builds + the table -> the next TLAS build / the passes' reads.
        memory_barrier(dev, cb, Stage::ACCELERATION_STRUCTURE_BUILD_KHR | Stage::COPY,
                       Access::ACCELERATION_STRUCTURE_WRITE_KHR | Access::TRANSFER_WRITE,
                       Stage::ACCELERATION_STRUCTURE_BUILD_KHR | Stage::COMPUTE_SHADER,
                       Access::ACCELERATION_STRUCTURE_READ_KHR | Access::SHADER_READ);
    }

    pub fn build_all(&mut self, ctx: &Context, cb: vk::CommandBuffer, world: &SparseVolume, palette: &[Material],
                     stage: &mut dyn FnMut(&[u8]) -> (vk::Buffer, u64, u64)) {
        let all: Vec<usize> = (0..self.tiles.len()).collect();
        self.build_tiles(ctx, cb, world, palette, &all, stage);
    }

    /// r.rt.tightboxes changed since the tiles were built.
    pub fn stale_boxes(&self) -> bool {
        self.active && self.tight != crate::cvars::R_RT_TIGHTBOXES.get()
    }

    /// Every frame, before any pass traces: the TLAS over the built tiles, then a barrier to
    /// compute reads.
    pub fn record_tlas(&mut self, ctx: &Context, cb: vk::CommandBuffer) {
        if !self.active {
            return;
        }
        self.frame += 1;
        self.reap(ctx, false);
        let ib = &self.inst[(self.frame % FRAMES_IN_FLIGHT) as usize];
        let mut inst: Vec<vk::AccelerationStructureInstanceKHR> = Vec::with_capacity(self.tiles.len());
        for t in self.tiles.iter().filter(|t| t.as_ != vk::AccelerationStructureKHR::null()).take(self.tlas_cap as usize) {
            inst.push(vk::AccelerationStructureInstanceKHR {
                transform: vk::TransformMatrixKHR { matrix: [1.0, 0.0, 0.0, t.ox as f32, 0.0, 1.0, 0.0, 0.0, 0.0, 0.0, 1.0, t.oz as f32] },
                instance_custom_index_and_mask: vk::Packed24_8::new(t.prim_off as u32 & 0xFF_FFFF, 0x01), // mask 1: the world
                instance_shader_binding_table_record_offset_and_flags: vk::Packed24_8::new(
                    0,
                    vk::GeometryInstanceFlagsKHR::TRIANGLE_FACING_CULL_DISABLE.as_raw() as u8,
                ),
                acceleration_structure_reference: vk::AccelerationStructureReferenceKHR { device_handle: t.addr },
            });
        }
        let bytes = unsafe { std::slice::from_raw_parts(inst.as_ptr() as *const u8, std::mem::size_of_val(&inst[..])) };
        ib.write(0, bytes);
        let acc = ctx.accel.as_ref().unwrap();
        let geo = [vk::AccelerationStructureGeometryKHR::default().geometry_type(vk::GeometryTypeKHR::INSTANCES).geometry(
            vk::AccelerationStructureGeometryDataKHR {
                instances: vk::AccelerationStructureGeometryInstancesDataKHR::default()
                    .data(vk::DeviceOrHostAddressConstKHR { device_address: ctx.buffer_address(ib.buf) }),
            },
        )];
        let bi = [vk::AccelerationStructureBuildGeometryInfoKHR::default()
            .ty(vk::AccelerationStructureTypeKHR::TOP_LEVEL)
            .flags(vk::BuildAccelerationStructureFlagsKHR::PREFER_FAST_TRACE)
            .mode(vk::BuildAccelerationStructureModeKHR::BUILD)
            .dst_acceleration_structure(self.tlas)
            .geometries(&geo)
            .scratch_data(vk::DeviceOrHostAddressKHR { device_address: align_up(ctx.buffer_address(self.tlas_scratch.buf), ctx.as_scratch_align) })];
        let range = [vk::AccelerationStructureBuildRangeInfoKHR::default().primitive_count(inst.len() as u32)];
        unsafe { acc.cmd_build_acceleration_structures(cb, &bi, &[&range[..]]) };
        memory_barrier(&ctx.device, cb, Stage::ACCELERATION_STRUCTURE_BUILD_KHR, Access::ACCELERATION_STRUCTURE_WRITE_KHR, Stage::COMPUTE_SHADER,
                       Access::ACCELERATION_STRUCTURE_READ_KHR);
    }
}

fn aabb_bytes(b: &[vk::AabbPositionsKHR]) -> &[u8] {
    // (VkAabbPositionsKHR: six f32, no padding)
    unsafe { std::slice::from_raw_parts(b.as_ptr() as *const u8, std::mem::size_of_val(b)) }
}

/// A detailed brick's occupied voxel bounds (inclusive), None when empty.
fn brick_bounds(br: &crate::voxel::sparse_volume::Brick) -> Option<([i32; 3], [i32; 3])> {
    let (mut lo, mut hi) = ([8i32; 3], [-1i32; 3]);
    let mut grow = |a: usize, x0: i32, x1: i32| {
        lo[a] = lo[a].min(x0);
        hi[a] = hi[a].max(x1);
    };
    if br.mask_ok {
        // The occupancy mask (per 4^3 octant 2 words, a word = 4x4x2 voxels, bit = x | y << 2 |
        // z' << 4): the same bounds as the voxel scan, from ~16 words.
        for (w, &m) in br.mask.iter().enumerate() {
            if m == 0 {
                continue;
            }
            let oct = w as i32 >> 1;
            let (ox, oy, oz) = ((oct & 1) * 4, ((oct >> 1) & 1) * 4, (oct >> 2) * 4 + (w as i32 & 1) * 2);
            let (lo16, hi16) = (m & 0xFFFF, m >> 16);
            let h = lo16 | hi16;
            grow(2, if lo16 != 0 { oz } else { oz + 1 }, if hi16 != 0 { oz + 1 } else { oz });
            let ys = (h & 0xF != 0) as u32 | ((h & 0xF0 != 0) as u32) << 1 | ((h & 0xF00 != 0) as u32) << 2 | ((h & 0xF000 != 0) as u32) << 3;
            let xs = (h | h >> 4 | h >> 8 | h >> 12) & 0xF;
            grow(1, oy + ys.trailing_zeros() as i32, oy + 31 - ys.leading_zeros() as i32);
            grow(0, ox + xs.trailing_zeros() as i32, ox + 31 - xs.leading_zeros() as i32);
        }
    } else {
        for lz in 0..8 {
            for ly in 0..8 {
                for lx in 0..8 {
                    if br.vox[(lz * 64 + ly * 8 + lx) as usize] != 0 {
                        grow(0, lx, lx);
                        grow(1, ly, ly);
                        grow(2, lz, lz);
                    }
                }
            }
        }
    }
    (hi[0] >= 0).then_some((lo, hi))
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::voxel::sparse_volume::{compute_brick_mask, Brick};

    #[test]
    fn ranges_merge() {
        let mut r = Ranges::default();
        r.reset(100);
        let a = r.alloc(10).unwrap();
        let b = r.alloc(20).unwrap();
        let c = r.alloc(30).unwrap();
        assert_eq!((a, b, c), (0, 10, 30));
        r.release(b, 20);
        assert_eq!(r.alloc(15), Some(10));
        r.release(10, 15);
        r.release(a, 10);
        r.release(c, 30);
        assert_eq!(r.free.len(), 1);
        assert_eq!(r.free.get(&0), Some(&100));
    }

    #[test]
    fn mask_bounds_match_voxel_scan() {
        let mut seed = 1u32;
        for _ in 0..200 {
            let mut b = Brick::filled(0);
            for _ in 0..(seed % 6 + 1) {
                seed = seed.wrapping_mul(1664525).wrapping_add(1013904223);
                b.vox[(seed >> 8) as usize % 512] = 3;
            }
            let scan = brick_bounds(&b);
            compute_brick_mask(&mut b);
            assert_eq!(brick_bounds(&b), scan);
        }
        assert_eq!(brick_bounds(&Brick::filled(0)), None);
    }
}
