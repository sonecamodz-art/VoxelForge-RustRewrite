//! The Vulkan mirror of the scene for the compute passes.
//!
//! Uploads the static world (brick grid + brick atlas + 2-level coarse occupancy pyramid +
//! palette) and the dynamic bodies (concatenated body grids + per-body headers + a BVH), all
//! sharing one brick atlas. Edits and streamed chunks re-upload incrementally through a
//! persistent mapped staging ring whose submits are retired by fence later (no stalls).
//!
//! Layout contracts with common.glsl: grid encoding (0 empty, 1-255 uniform, 256 + atlas
//! slot), a 64 x 64-brick atlas layer, C1 = 4 / C2 = 16 coarse cells, 64-byte brick masks,
//! the 240-byte ObjHeader and the 32-byte BVH node.

use ash::vk;
use bytemuck::{Pod, Zeroable};
use gpu_allocator::MemoryLocation;
use rayon::prelude::*;

use super::context::{image_barrier, memory_barrier, to_general, vkc, Access, Buffer, Context, Image, Stage, COLOR_LAYERS};
use super::shader::{self, ComputePipeline};
use crate::objects::VoxObject;
use crate::voxel::sparse_volume::{brick_mask, Brick, RING_CHUNK_B};
use crate::voxel::world::{wm, Material, World};

pub const ABX: i32 = 64;
pub const ABY: i32 = 64;
pub const MIN_ABZ: i32 = 128;
pub const MAX_ABZ: i32 = 2048;
pub const C1: i32 = 4;
pub const C2: i32 = 16;
/// Must match DIST_CAP in brickdist.comp.
pub const DIST_CAP: i32 = 16;
const RING_SIZE: u64 = 64 << 20;

/// std430 ObjHeader (common.glsl): 3 x mat4, ivec4, vec4, vec4 = 240 bytes.
#[repr(C)]
#[derive(Clone, Copy, Pod, Zeroable)]
pub struct ObjHeader {
    pub world_to_obj: [f32; 16],
    pub obj_to_world: [f32; 16],
    pub prev_obj_to_world: [f32; 16],
    /// bricksX/Y/Z, grid offset into objGrid.
    pub dims_bricks: [i32; 4],
    /// .w: glow integrity.
    pub aabb_min: [f32; 4],
    pub aabb_max: [f32; 4],
}
const _: () = assert!(std::mem::size_of::<ObjHeader>() == 240);

/// A brick that goes to the atlas: in the world or in body `obj`.
#[derive(Clone, Copy)]
enum BrickRef {
    World(u32),
    Obj(u32, u32),
}

/// Brick bytes as the GPU sees them: a torch's flame cell is air for every ray.
#[inline]
fn gpu_voxels(dst: &mut [u8], src: &[u8]) {
    dst.copy_from_slice(src);
    if src.contains(&wm::TORCH_FLAME) {
        for d in dst.iter_mut() {
            if *d == wm::TORCH_FLAME {
                *d = 0;
            }
        }
    }
}

fn masks_of(b: &Brick) -> [u32; 16] {
    if b.mask_ok { b.mask } else { brick_mask(&b.vox) }
}

struct PendingSync {
    fence: vk::Fence,
    cb: vk::CommandBuffer,
    temps: Vec<Buffer>,
    ring_end: u64,
}

/// One box of bricks (inclusive), absolute brick coordinates.
#[derive(Clone, Copy, Debug)]
struct BrickBox {
    lo: [i32; 3],
    hi: [i32; 3],
}

#[derive(Default)]
pub struct GpuWorld {
    pub grid: Image,
    pub atlas: Image,
    pub coarse1: Image,
    pub coarse2: Image,
    /// [0] the field traversal samples, [1] ping-pong temp, [2] the DRY field (water counts as
    /// empty), [3] its per-brick dry occupancy.
    pub dist: [Image; 4],
    pub grass_top: Image,
    pub grass_tile: Image,
    pub grass_info: Image,
    pub sky_top: Image,
    pub sampler: vk::Sampler,

    dist_pipe: ComputePipeline,
    dist_layout: vk::DescriptorSetLayout,
    dist_pool: vk::DescriptorPool,
    dist_sets: [vk::DescriptorSet; 7],
    grass_pipe: ComputePipeline,
    grass_layout: vk::DescriptorSetLayout,
    grass_pool: vk::DescriptorPool,
    grass_set: vk::DescriptorSet,

    pub palette_buf: Buffer,
    pub obj_grid_buf: Buffer,
    pub mask_buf: Buffer,
    pub objects_buf: [Buffer; 2],
    pub bvh_node_buf: [Buffer; 2],
    pub bvh_prim_buf: [Buffer; 2],
    objects_cap: u64,
    bvh_capacity: u32,
    hdr_cache: Vec<ObjHeader>,
    bvh_nodes_cache: Vec<[f32; 8]>,
    bvh_prims_cache: Vec<u32>,
    obj_version: u64,
    slot_version: [u64; 2],
    grid_offsets: Vec<u32>,
    prev_obj_mat: Vec<[f32; 16]>,

    // staging ring
    ring: Buffer,
    ring_head: u64,
    ring_tail: u64,
    sync_open: bool,
    sync_begin: u64,
    pending: Vec<PendingSync>,
    sync_temps: Vec<Buffer>,
    free_fences: Vec<vk::Fence>,
    free_cbs: Vec<vk::CommandBuffer>,

    pub wbx: i32,
    pub wby: i32,
    pub wbz: i32,
    pub ring_mode: bool,
    /// Sizes the atlas for the chunks a streaming radius keeps resident.
    pub stream_radius: i32,
    pub atlas_slots: u32,
    pub atlas_bricks_z: i32,
    free_slots: Vec<u32>,
    world_slot: Vec<u8>,
    atlas_full_warned: bool,
    pub bricks_compacted: u32,
    pub num_objects: u32,
    coarse1_cpu: Vec<u8>,
    coarse2_cpu: Vec<u8>,
    c1: [i32; 3],
    c2: [i32; 3],

    /// The last world edit's voxel AABB + a counter (Standard lighting re-traces its shadow).
    pub edit_lo: [i32; 3],
    pub edit_hi: [i32; 3],
    pub edit_counter: u32,
    /// Voxel boxes changed since the consumer last took them (edits, streamed chunks).
    pub edit_boxes: Vec<[i32; 6]>,
    /// Chunks synced (in or out) since the lights last looked.
    pub chunk_events: Vec<(i32, i32)>,
    /// Grid cells synced since the light registry last rescanned.
    pub synced_grid: Vec<u32>,
}

impl GpuWorld {
    pub fn atlas_capacity(&self) -> i32 {
        ABX * ABY * self.atlas_bricks_z
    }

    fn alloc_slot(&mut self) -> i32 {
        if let Some(s) = self.free_slots.pop() {
            return s as i32;
        }
        if (self.atlas_slots as i32) < self.atlas_capacity() {
            self.atlas_slots += 1;
            return self.atlas_slots as i32 - 1;
        }
        -1
    }

    /// Grid encoding of a detailed brick: its atlas slot (allocating one). Atlas full: the
    /// brick's majority material as a uniform cell (shape kept, detail lost). Returns
    /// (encoding, needs upload).
    fn encode_brick(&mut self, b: &mut Brick, world: bool) -> (u32, bool) {
        if b.gpu_slot < 0 {
            b.gpu_slot = self.alloc_slot();
        }
        if b.gpu_slot >= 0 {
            if world {
                self.world_slot[b.gpu_slot as usize] = 1;
            }
            return (256 + b.gpu_slot as u32, true);
        }
        if !self.atlas_full_warned {
            log::warn!(target: "gpu-world", "brick atlas full ({} slots) - new detail shown as uniform bricks", self.atlas_slots);
            self.atlas_full_warned = true;
        }
        let mut count = [0u32; 256];
        for &v in b.vox.iter() {
            count[v as usize] += 1;
        }
        let mut best = 1;
        for m in 2..256 {
            if count[m] > count[best] && m != wm::TORCH_FLAME as usize {
                best = m;
            }
        }
        (if count[best] * 2 >= 512 { best as u32 } else { 0 }, false)
    }

    fn brick<'a>(world: &'a World, objects: &'a [VoxObject], r: BrickRef) -> &'a Brick {
        match r {
            BrickRef::World(i) => &world.bricks[i as usize],
            BrickRef::Obj(o, i) => &objects[o as usize].vol.bricks[i as usize],
        }
    }

    // --- whole upload ----------------------------------------------------------------------
    /// Pack + upload the world and the bodies. Assigns gpu_slot to every detailed brick.
    pub fn upload(&mut self, ctx: &Context, pcache: vk::PipelineCache, world: &mut World, objects: &mut [VoxObject]) {
        let t0 = std::time::Instant::now();
        self.wbx = world.bricks_x;
        self.wby = world.bricks_y;
        self.wbz = world.bricks_z;
        self.ring_mode = world.ring_w > 0;
        if self.stream_radius == 0 {
            self.stream_radius = 10;
        }
        // Atlas depth: detailed bricks (world + bodies) x 1.6 for edits; streaming: what the
        // radius keeps resident (~2000 bricks a chunk) + 10 %.
        let mut need: usize = world.grid.par_iter().filter(|&&v| v >= 256).count();
        need += objects.iter().map(|o| o.vol.grid.iter().filter(|&&v| v >= 256).count()).sum::<usize>();
        need = need * 16 / 10;
        if self.ring_mode {
            let r = self.stream_radius as f64 + 1.5;
            need = need.max((std::f64::consts::PI * r * r * 2000.0 * 1.1) as usize);
        }
        let layer = (ABX * ABY) as usize;
        self.atlas_bricks_z = (need.div_ceil(layer) as i32).clamp(MIN_ABZ, MAX_ABZ);
        let cap = self.atlas_capacity() as usize;

        // --- CPU pack: grid encoding + atlas slots (fresh atlas: sequential) ---
        self.atlas_slots = 0;
        self.free_slots.clear();
        self.world_slot = vec![0; cap];
        world.released_gpu_slots.clear();
        self.atlas_full_warned = false;
        let mut slot_bricks: Vec<BrickRef> = Vec::new();
        let mut staged = vec![0u32; world.grid.len()];
        for gi in 0..world.grid.len() {
            let v = world.grid[gi];
            if v < 256 {
                staged[gi] = v;
                continue;
            }
            let bi = v - 256;
            let b = &mut world.vol.bricks[bi as usize];
            b.gpu_slot = -1;
            let (enc, up) = self.encode_brick(b, true);
            staged[gi] = enc;
            if up {
                slot_bricks.push(BrickRef::World(bi));
            }
        }
        // Bodies continue numbering into the same atlas.
        self.num_objects = objects.len() as u32;
        let mut obj_grid: Vec<u32> = Vec::new();
        self.grid_offsets = vec![0; objects.len()];
        for (oi, o) in objects.iter_mut().enumerate() {
            self.grid_offsets[oi] = obj_grid.len() as u32;
            for gi in 0..o.vol.grid.len() {
                let v = o.vol.grid[gi];
                if v < 256 {
                    obj_grid.push(v);
                    continue;
                }
                let b = &mut o.vol.bricks[(v - 256) as usize];
                b.gpu_slot = -1;
                let (enc, up) = self.encode_brick(b, false);
                obj_grid.push(enc);
                if up {
                    slot_bricks.push(BrickRef::Obj(oi as u32, v - 256));
                }
            }
        }
        if obj_grid.is_empty() {
            obj_grid.push(0);
        }
        let headers = self.build_headers(objects, true);

        // --- coarse pyramid ---
        self.c1 = [(self.wbx + C1 - 1) / C1, (self.wby + C1 - 1) / C1, (self.wbz + C1 - 1) / C1];
        self.c2 = [(self.wbx + C2 - 1) / C2, (self.wby + C2 - 1) / C2, (self.wbz + C2 - 1) / C2];
        self.coarse1_cpu = vec![0; (self.c1[0] * self.c1[1] * self.c1[2]) as usize];
        self.coarse2_cpu = vec![0; (self.c2[0] * self.c2[1] * self.c2[2]) as usize];
        for bz in 0..self.wbz {
            for by in 0..self.wby {
                for bx in 0..self.wbx {
                    if world.grid[((bz * self.wby + by) * self.wbx + bx) as usize] == 0 {
                        continue;
                    }
                    self.coarse1_cpu[(((bz / C1) * self.c1[1] + by / C1) * self.c1[0] + bx / C1) as usize] = 1;
                    self.coarse2_cpu[(((bz / C2) * self.c2[1] + by / C2) * self.c2[0] + bx / C2) as usize] = 1;
                }
            }
        }
        let t_pack = t0.elapsed();
        log::info!(target: "vk", "world {}x{}x{} bricks + {} bodies, atlas {} slots ({} z-bricks)", self.wbx, self.wby, self.wbz,
                   self.num_objects, self.atlas_slots, self.atlas_bricks_z);

        // --- GPU resources ---
        let ro3 = vk::ImageUsageFlags::SAMPLED | vk::ImageUsageFlags::TRANSFER_DST;
        let (wbx, wby, wbz) = (self.wbx as u32, self.wby as u32, self.wbz as u32);
        self.grid = ctx.image_3d("worldGrid", vk::Format::R32_UINT, wbx, wby, wbz, ro3);
        self.atlas = ctx.image_3d("brickAtlas", vk::Format::R8_UINT, (ABX * 8) as u32, (ABY * 8) as u32, (self.atlas_bricks_z * 8) as u32, ro3);
        self.coarse1 = ctx.image_3d("coarse1", vk::Format::R8_UINT, self.c1[0] as u32, self.c1[1] as u32, self.c1[2] as u32, ro3);
        self.coarse2 = ctx.image_3d("coarse2", vk::Format::R8_UINT, self.c2[0] as u32, self.c2[1] as u32, self.c2[2] as u32, ro3);
        for (i, n) in ["brickDist", "brickDistTmp", "brickDistDry", "brickDryOccupancy"].iter().enumerate() {
            self.dist[i] = ctx.image_3d(n, vk::Format::R8_UINT, wbx, wby, wbz, ro3 | vk::ImageUsageFlags::STORAGE);
        }
        self.sampler = ctx.sampler(vk::Filter::NEAREST);
        let st2 = |name: &str, f: vk::Format, w: u32, h: u32| {
            ctx.create_image(name, vk::ImageType::TYPE_2D, f, vk::Extent3D { width: w, height: h, depth: 1 },
                             vk::ImageUsageFlags::SAMPLED | vk::ImageUsageFlags::STORAGE)
        };
        self.grass_top = st2("grassTop", vk::Format::R16_UINT, wbx * 8, wbz * 8);
        self.grass_tile = st2("grassTile", vk::Format::R16G16_UINT, wbx, wbz);
        self.grass_info = st2("grassInfo", vk::Format::R8G8B8A8_UINT, wbx * 8, wbz * 8);
        self.sky_top = st2("skyTop", vk::Format::R16_UINT, wbx * 8, wbz * 8);
        self.palette_buf = ctx.upload_buffer("palette", bytemuck::cast_slice(&world.palette), vk::BufferUsageFlags::STORAGE_BUFFER);
        self.make_pipelines(ctx, pcache);
        self.mask_buf = ctx.create_buffer("brickMasks", cap as u64 * 64, vk::BufferUsageFlags::STORAGE_BUFFER | vk::BufferUsageFlags::TRANSFER_DST, MemoryLocation::GpuOnly);
        self.obj_grid_buf = ctx.upload_buffer("objGrid", bytemuck::cast_slice(&obj_grid), vk::BufferUsageFlags::STORAGE_BUFFER);
        self.set_headers(headers);
        self.alloc_object_buffers(ctx);
        self.flush_slot(ctx, 0);
        self.flush_slot(ctx, 1);
        let t_create = t0.elapsed();

        // --- staged uploads: the atlas laid out layer by layer in image order (parallel) ---
        let per_layer = (ABX * ABY) as usize;
        let layers = self.atlas_slots as usize / per_layer + usize::from(self.atlas_slots as usize % per_layer != 0);
        let layer_bytes = (ABX * 8 * ABY * 8 * 8) as usize;
        let mut s_grid = ctx.upload_buffer("stageGrid", bytemuck::cast_slice(&staged), vk::BufferUsageFlags::TRANSFER_SRC);
        drop(staged);
        let mut s_c1 = ctx.upload_buffer("stageC1", &self.coarse1_cpu, vk::BufferUsageFlags::TRANSFER_SRC);
        let mut s_c2 = ctx.upload_buffer("stageC2", &self.coarse2_cpu, vk::BufferUsageFlags::TRANSFER_SRC);
        let mut s_atlas = ctx.create_buffer("stageAtlas", (layers * layer_bytes).max(1) as u64, vk::BufferUsageFlags::TRANSFER_SRC, MemoryLocation::CpuToGpu);
        let mut s_mask = ctx.create_buffer("stageMasks", (slot_bricks.len() * 64).max(4) as u64, vk::BufferUsageFlags::TRANSFER_SRC, MemoryLocation::CpuToGpu);
        {
            let (w, objs) = (&*world, &*objects);
            let atlas_mem = unsafe { std::slice::from_raw_parts_mut(s_atlas.ptr, layers * layer_bytes) };
            let row_pitch = (ABX * 8) as usize;
            let slice_pitch = row_pitch * (ABY * 8) as usize;
            atlas_mem.par_chunks_mut(layer_bytes).enumerate().for_each(|(l, layer)| {
                layer.fill(0);
                for k in 0..per_layer {
                    let s2 = l * per_layer + k;
                    if s2 >= slot_bricks.len() {
                        break;
                    }
                    let (sx, sy) = (k % ABX as usize, k / ABX as usize);
                    let src = &Self::brick(w, objs, slot_bricks[s2]).vox;
                    for lz in 0..8 {
                        for ly in 0..8 {
                            let d = lz * slice_pitch + (sy * 8 + ly) * row_pitch + sx * 8;
                            gpu_voxels(&mut layer[d..d + 8], &src[lz * 64 + ly * 8..lz * 64 + ly * 8 + 8]);
                        }
                    }
                }
            });
            let mask_mem = unsafe { std::slice::from_raw_parts_mut(s_mask.ptr as *mut [u32; 16], slot_bricks.len()) };
            mask_mem.par_iter_mut().enumerate().for_each(|(i, m)| *m = masks_of(Self::brick(w, objs, slot_bricks[i])));
        }
        let dev = &ctx.device;
        let cb = ctx.begin_one_shot();
        let imgs = [self.grid.image, self.atlas.image, self.coarse1.image, self.coarse2.image];
        for &im in &imgs {
            image_barrier(dev, cb, im, Stage::TOP_OF_PIPE, Access::NONE, Stage::COPY, Access::TRANSFER_WRITE,
                          vk::ImageLayout::UNDEFINED, vk::ImageLayout::TRANSFER_DST_OPTIMAL);
        }
        let full = |src: &Buffer, dst: &Image| {
            let c = [vk::BufferImageCopy::default().image_subresource(COLOR_LAYERS).image_extent(dst.extent)];
            unsafe { dev.cmd_copy_buffer_to_image(cb, src.buf, dst.image, vk::ImageLayout::TRANSFER_DST_OPTIMAL, &c) };
        };
        full(&s_grid, &self.grid);
        full(&s_c1, &self.coarse1);
        full(&s_c2, &self.coarse2);
        if layers > 0 {
            let regions: Vec<_> = (0..layers)
                .map(|l| {
                    vk::BufferImageCopy::default()
                        .buffer_offset((l * layer_bytes) as u64)
                        .image_subresource(COLOR_LAYERS)
                        .image_offset(vk::Offset3D { x: 0, y: 0, z: l as i32 * 8 })
                        .image_extent(vk::Extent3D { width: (ABX * 8) as u32, height: (ABY * 8) as u32, depth: 8 })
                })
                .collect();
            unsafe { dev.cmd_copy_buffer_to_image(cb, s_atlas.buf, self.atlas.image, vk::ImageLayout::TRANSFER_DST_OPTIMAL, &regions) };
        }
        if !slot_bricks.is_empty() {
            let r = [vk::BufferCopy { src_offset: 0, dst_offset: 0, size: slot_bricks.len() as u64 * 64 }];
            unsafe { dev.cmd_copy_buffer(cb, s_mask.buf, self.mask_buf.buf, &r) };
        }
        for &im in &imgs {
            image_barrier(dev, cb, im, Stage::COPY, Access::TRANSFER_WRITE, Stage::COMPUTE_SHADER, Access::SHADER_SAMPLED_READ,
                          vk::ImageLayout::TRANSFER_DST_OPTIMAL, vk::ImageLayout::SHADER_READ_ONLY_OPTIMAL);
        }
        memory_barrier(dev, cb, Stage::COPY, Access::TRANSFER_WRITE, Stage::COMPUTE_SHADER, Access::SHADER_STORAGE_READ);
        let genl: Vec<vk::Image> = self.dist.iter().map(|d| d.image).chain([self.grass_top.image, self.grass_tile.image, self.grass_info.image, self.sky_top.image]).collect();
        to_general(dev, cb, &genl);
        let (lo, hi) = ([0, 0, 0], [self.wbx - 1, self.wby - 1, self.wbz - 1]);
        self.record_distance_field(ctx, cb, lo, hi, lo, hi);
        if self.ring_mode {
            let k = RING_CHUNK_B * 8;
            for s in world.ring_slots.iter().filter(|s| s.loaded) {
                self.record_grass_top(ctx, cb, s.cx * k, s.cz * k, (s.cx + 1) * k - 1, (s.cz + 1) * k - 1);
            }
        } else {
            self.record_grass_top(ctx, cb, 0, 0, self.wbx * 8 - 1, self.wbz * 8 - 1);
        }
        ctx.end_one_shot_and_wait(cb);
        for b in [&mut s_grid, &mut s_c1, &mut s_c2, &mut s_atlas, &mut s_mask] {
            b.destroy(ctx);
        }
        log::info!(target: "vk", "world upload: pack {:.0} ms, create {:.0}, total {:.0} ms", t_pack.as_secs_f64() * 1e3,
                   (t_create - t_pack).as_secs_f64() * 1e3, t0.elapsed().as_secs_f64() * 1e3);
        world.dirty_grid.clear();
        for o in objects.iter_mut() {
            o.vol.dirty_grid.clear();
        }
    }

    fn make_pipelines(&mut self, ctx: &Context, pcache: vk::PipelineCache) {
        let dev = &ctx.device;
        let cis = vk::DescriptorType::COMBINED_IMAGE_SAMPLER;
        let si = vk::DescriptorType::STORAGE_IMAGE;
        let sb = vk::DescriptorType::STORAGE_BUFFER;
        // Distance field: 3 normal passes + dry occupancy + 3 dry passes.
        self.dist_layout = shader::make_set_layout(dev, &[(0, cis), (1, si), (2, cis)]);
        let spv = shader::compile_parts(&["brickdist.comp"], "").unwrap_or_else(|e| crate::log::fatal("shader", &e));
        self.dist_pipe = shader::make_compute_pipeline(dev, pcache, &spv, &[self.dist_layout], 32);
        let ps = [vk::DescriptorPoolSize { ty: cis, descriptor_count: 14 }, vk::DescriptorPoolSize { ty: si, descriptor_count: 7 }];
        self.dist_pool = vkc(unsafe { dev.create_descriptor_pool(&vk::DescriptorPoolCreateInfo::default().max_sets(7).pool_sizes(&ps), None) }, "dist pool");
        let layouts = [self.dist_layout; 7];
        let sets = vkc(unsafe { dev.allocate_descriptor_sets(&vk::DescriptorSetAllocateInfo::default().descriptor_pool(self.dist_pool).set_layouts(&layouts)) }, "dist sets");
        self.dist_sets.copy_from_slice(&sets);
        let ro = vk::ImageLayout::SHADER_READ_ONLY_OPTIMAL;
        let genl = vk::ImageLayout::GENERAL;
        let (g, d) = (self.grid.view, [self.dist[0].view, self.dist[1].view, self.dist[2].view, self.dist[3].view]);
        let src_v = [g, d[0], d[1], g, d[3], d[2], d[1]];
        let src_l = [ro, genl, genl, ro, genl, genl, genl];
        let dst_v = [d[0], d[1], d[0], d[3], d[2], d[1], d[2]];
        for i in 0..7 {
            let si_ = [vk::DescriptorImageInfo { sampler: self.sampler, image_view: src_v[i], image_layout: src_l[i] }];
            let di = [vk::DescriptorImageInfo { sampler: vk::Sampler::null(), image_view: dst_v[i], image_layout: genl }];
            let ai = [vk::DescriptorImageInfo { sampler: self.sampler, image_view: self.atlas.view, image_layout: ro }];
            let w = [
                vk::WriteDescriptorSet::default().dst_set(self.dist_sets[i]).dst_binding(0).descriptor_type(cis).image_info(&si_),
                vk::WriteDescriptorSet::default().dst_set(self.dist_sets[i]).dst_binding(1).descriptor_type(si).image_info(&di),
                vk::WriteDescriptorSet::default().dst_set(self.dist_sets[i]).dst_binding(2).descriptor_type(cis).image_info(&ai),
            ];
            unsafe { dev.update_descriptor_sets(&w, &[]) };
        }
        // Grass top map.
        self.grass_layout = shader::make_set_layout(dev, &[(0, cis), (1, cis), (2, sb), (3, si), (4, si), (5, si), (6, si)]);
        let spv = shader::compile_parts(&["grasstop.comp"], "").unwrap_or_else(|e| crate::log::fatal("shader", &e));
        self.grass_pipe = shader::make_compute_pipeline(dev, pcache, &spv, &[self.grass_layout], 32);
        let ps = [
            vk::DescriptorPoolSize { ty: cis, descriptor_count: 2 },
            vk::DescriptorPoolSize { ty: si, descriptor_count: 4 },
            vk::DescriptorPoolSize { ty: sb, descriptor_count: 1 },
        ];
        self.grass_pool = vkc(unsafe { dev.create_descriptor_pool(&vk::DescriptorPoolCreateInfo::default().max_sets(1).pool_sizes(&ps), None) }, "grass pool");
        let l = [self.grass_layout];
        self.grass_set = vkc(unsafe { dev.allocate_descriptor_sets(&vk::DescriptorSetAllocateInfo::default().descriptor_pool(self.grass_pool).set_layouts(&l)) }, "grass set")[0];
        self.write_grass_set(ctx);
    }

    fn write_grass_set(&self, ctx: &Context) {
        let (cis, si, sb) = (vk::DescriptorType::COMBINED_IMAGE_SAMPLER, vk::DescriptorType::STORAGE_IMAGE, vk::DescriptorType::STORAGE_BUFFER);
        let ro = vk::ImageLayout::SHADER_READ_ONLY_OPTIMAL;
        let genl = vk::ImageLayout::GENERAL;
        let gi = [vk::DescriptorImageInfo { sampler: self.sampler, image_view: self.grid.view, image_layout: ro }];
        let ai = [vk::DescriptorImageInfo { sampler: self.sampler, image_view: self.atlas.view, image_layout: ro }];
        let pb = [vk::DescriptorBufferInfo { buffer: self.palette_buf.buf, offset: 0, range: vk::WHOLE_SIZE }];
        let st = |v: vk::ImageView| [vk::DescriptorImageInfo { sampler: vk::Sampler::null(), image_view: v, image_layout: genl }];
        let (t, l, n, s) = (st(self.grass_top.view), st(self.grass_tile.view), st(self.grass_info.view), st(self.sky_top.view));
        let set = self.grass_set;
        let w = [
            vk::WriteDescriptorSet::default().dst_set(set).dst_binding(0).descriptor_type(cis).image_info(&gi),
            vk::WriteDescriptorSet::default().dst_set(set).dst_binding(1).descriptor_type(cis).image_info(&ai),
            vk::WriteDescriptorSet::default().dst_set(set).dst_binding(2).descriptor_type(sb).buffer_info(&pb),
            vk::WriteDescriptorSet::default().dst_set(set).dst_binding(3).descriptor_type(si).image_info(&t),
            vk::WriteDescriptorSet::default().dst_set(set).dst_binding(4).descriptor_type(si).image_info(&l),
            vk::WriteDescriptorSet::default().dst_set(set).dst_binding(5).descriptor_type(si).image_info(&n),
            vk::WriteDescriptorSet::default().dst_set(set).dst_binding(6).descriptor_type(si).image_info(&s),
        ];
        unsafe { ctx.device.update_descriptor_sets(&w, &[]) };
    }

    // --- bodies --------------------------------------------------------------------------------
    fn build_headers(&mut self, objects: &[VoxObject], reset_motion: bool) -> Vec<ObjHeader> {
        let mut headers: Vec<ObjHeader> = objects
            .iter()
            .enumerate()
            .map(|(i, o)| {
                let (mn, mx) = o.world_aabb();
                ObjHeader {
                    world_to_obj: o.pose.inverse().to_mat4(),
                    obj_to_world: o.pose.to_mat4(),
                    prev_obj_to_world: [0.0; 16],
                    dims_bricks: [o.vol.bricks_x, o.vol.bricks_y, o.vol.bricks_z, self.grid_offsets[i] as i32],
                    aabb_min: [mn.x, mn.y, mn.z, o.glow_integrity],
                    aabb_max: [mx.x, mx.y, mx.z, 0.0],
                }
            })
            .collect();
        if reset_motion {
            self.prev_obj_mat.clear();
        }
        // Last frame's pose -> motion vectors (first time: the current pose = no motion).
        let have_prev = self.prev_obj_mat.len() == headers.len();
        for (i, h) in headers.iter_mut().enumerate() {
            h.prev_obj_to_world = if have_prev { self.prev_obj_mat[i] } else { h.obj_to_world };
        }
        self.prev_obj_mat = headers.iter().map(|h| h.obj_to_world).collect();
        headers
    }

    fn set_headers(&mut self, headers: Vec<ObjHeader>) {
        let aabbs: Vec<[f32; 6]> = headers.iter().map(|h| [h.aabb_min[0], h.aabb_min[1], h.aabb_min[2], h.aabb_max[0], h.aabb_max[1], h.aabb_max[2]]).collect();
        self.build_bvh(&aabbs);
        self.hdr_cache = if headers.is_empty() { vec![ObjHeader::zeroed()] } else { headers };
        self.obj_version += 1;
    }

    /// BVH over the bodies' world AABBs: median split along the widest centroid axis.
    /// Nodes: {min.xyz, first child | first prim}, {max.xyz, prim count (0 = interior)}.
    fn build_bvh(&mut self, bb: &[[f32; 6]]) {
        self.bvh_nodes_cache.clear();
        self.bvh_prims_cache.clear();
        if bb.is_empty() {
            return;
        }
        let mut prims: Vec<u32> = (0..bb.len() as u32).collect();
        let mut nodes: Vec<[f32; 8]> = vec![[0.0; 8]];
        fn set_node(nodes: &mut [[f32; 8]], n: usize, mn: [f32; 3], mx: [f32; 3], a: u32, cnt: u32) {
            nodes[n] = [mn[0], mn[1], mn[2], f32::from_bits(a), mx[0], mx[1], mx[2], f32::from_bits(cnt)];
        }
        fn build(bb: &[[f32; 6]], prims: &mut [u32], nodes: &mut Vec<[f32; 8]>, n: usize, first: u32) {
            let count = prims.len() as u32;
            let (mut mn, mut mx) = ([1e30f32; 3], [-1e30f32; 3]);
            for &p in prims.iter() {
                for k in 0..3 {
                    mn[k] = mn[k].min(bb[p as usize][k]);
                    mx[k] = mx[k].max(bb[p as usize][3 + k]);
                }
            }
            if count <= 2 {
                set_node(nodes, n, mn, mx, first, count);
                return;
            }
            let cen = |p: u32, k: usize| bb[p as usize][k] + bb[p as usize][3 + k];
            let (mut cmn, mut cmx) = ([1e30f32; 3], [-1e30f32; 3]);
            for &p in prims.iter() {
                for k in 0..3 {
                    cmn[k] = cmn[k].min(cen(p, k));
                    cmx[k] = cmx[k].max(cen(p, k));
                }
            }
            let mut ax = 0;
            for k in 1..3 {
                if cmx[k] - cmn[k] > cmx[ax] - cmn[ax] {
                    ax = k;
                }
            }
            let mid = prims.len() / 2;
            prims.select_nth_unstable_by(mid, |&a, &b| cen(a, ax).total_cmp(&cen(b, ax)));
            let left = nodes.len();
            nodes.push([0.0; 8]);
            nodes.push([0.0; 8]);
            set_node(nodes, n, mn, mx, left as u32, 0);
            let (l, r) = prims.split_at_mut(mid);
            build(bb, l, nodes, left, first);
            build(bb, r, nodes, left + 1, first + mid as u32);
        }
        build(bb, &mut prims, &mut nodes, 0, 0);
        self.bvh_nodes_cache = nodes;
        self.bvh_prims_cache = prims;
    }

    fn alloc_object_buffers(&mut self, ctx: &Context) {
        for k in 0..2 {
            self.objects_buf[k].destroy(ctx);
            self.bvh_node_buf[k].destroy(ctx);
            self.bvh_prim_buf[k].destroy(ctx);
        }
        self.objects_cap = (self.hdr_cache.len() * 240).max(240) as u64;
        let n = (self.bvh_prims_cache.len() as u32).max(1);
        self.bvh_capacity = 1024.max(n * 2); // headroom for debris
        let sb = vk::BufferUsageFlags::STORAGE_BUFFER;
        for k in 0..2 {
            self.objects_buf[k] = ctx.create_buffer("objects", self.objects_cap, sb, MemoryLocation::CpuToGpu);
            self.bvh_node_buf[k] = ctx.create_buffer("bvhNodes", self.bvh_capacity as u64 * 2 * 32, sb, MemoryLocation::CpuToGpu);
            self.bvh_prim_buf[k] = ctx.create_buffer("bvhPrims", self.bvh_capacity as u64 * 4, sb, MemoryLocation::CpuToGpu);
            self.slot_version[k] = 0;
        }
    }

    /// Copy the body caches into frame slot `slot` if stale. True if the buffers had to be
    /// recreated (bigger): the renderer must re-point its sets.
    pub fn flush_slot(&mut self, ctx: &Context, slot: usize) -> bool {
        if self.slot_version[slot] == self.obj_version {
            return false;
        }
        let mut recreated = false;
        if (self.hdr_cache.len() * 240) as u64 > self.objects_cap || self.bvh_prims_cache.len() as u32 > self.bvh_capacity || self.objects_buf[slot].is_null() {
            ctx.wait_idle();
            self.alloc_object_buffers(ctx);
            recreated = true;
        }
        self.objects_buf[slot].write(0, bytemuck::cast_slice(&self.hdr_cache));
        if !self.bvh_nodes_cache.is_empty() {
            self.bvh_node_buf[slot].write(0, bytemuck::cast_slice(&self.bvh_nodes_cache));
            self.bvh_prim_buf[slot].write(0, bytemuck::cast_slice(&self.bvh_prims_cache));
        }
        self.slot_version[slot] = self.obj_version;
        if recreated {
            for s in 0..2 {
                if s != slot {
                    self.flush_slot(ctx, s);
                }
            }
        }
        recreated
    }

    /// Rebuild every body header (poses + AABBs) + the BVH (call each frame while bodies move).
    pub fn update_objects(&mut self, objects: &[VoxObject]) {
        if objects.is_empty() || self.grid_offsets.len() != objects.len() {
            return;
        }
        let h = self.build_headers(objects, false);
        self.set_headers(h);
    }

    /// Full body rebuild after a STRUCTURAL change (grow / split / spawn / remove): re-packs
    /// every body grid, reclaims atlas slots, recreates the buffers. The renderer must
    /// re-point set 1 afterwards.
    pub fn init_objects(&mut self, ctx: &Context, world: &World, objects: &mut [VoxObject]) {
        ctx.wait_idle();
        // Live = world-owned slots + slots the current bodies still reference.
        let mut live: Vec<u8> = self.world_slot[..self.atlas_slots as usize].to_vec();
        for o in objects.iter() {
            for &v in &o.vol.grid {
                if v >= 256 {
                    let gs = o.vol.bricks[(v - 256) as usize].gpu_slot;
                    if gs >= 0 {
                        live[gs as usize] = 1;
                    }
                }
            }
        }
        self.free_slots = (0..self.atlas_slots).rev().filter(|&s| live[s as usize] == 0).collect();
        let mut all = Vec::new();
        let mut up: Vec<BrickRef> = Vec::new();
        self.grid_offsets = vec![0; objects.len()];
        for (oi, o) in objects.iter_mut().enumerate() {
            self.grid_offsets[oi] = all.len() as u32;
            for gi in 0..o.vol.grid.len() {
                let v = o.vol.grid[gi];
                if v < 256 {
                    all.push(v);
                    continue;
                }
                let (enc, u) = self.encode_brick(&mut o.vol.bricks[(v - 256) as usize], false);
                all.push(enc);
                if u {
                    up.push(BrickRef::Obj(oi as u32, v - 256));
                }
            }
            o.vol.dirty_grid.clear();
        }
        if all.is_empty() {
            all.push(0);
        }
        self.num_objects = objects.len() as u32;
        let h = self.build_headers(objects, true);
        self.obj_grid_buf.destroy(ctx);
        self.obj_grid_buf = ctx.upload_buffer("objGrid", bytemuck::cast_slice(&all), vk::BufferUsageFlags::STORAGE_BUFFER);
        self.set_headers(h);
        self.alloc_object_buffers(ctx);
        self.flush_slot(ctx, 0);
        self.flush_slot(ctx, 1);
        self.upload_bricks_now(ctx, world, objects, &up);
    }

    /// Incremental re-sync of edited body voxels (body count / sizes unchanged).
    pub fn sync_objects(&mut self, ctx: &Context, world: &World, objects: &mut [VoxObject]) {
        if self.grid_offsets.len() != objects.len() {
            return;
        }
        let mut writes: Vec<(usize, u32)> = Vec::new();
        let mut up: Vec<BrickRef> = Vec::new();
        for (oi, o) in objects.iter_mut().enumerate() {
            if o.vol.dirty_grid.is_empty() {
                continue;
            }
            let mut dirty = std::mem::take(&mut o.vol.dirty_grid);
            dirty.sort_unstable();
            dirty.dedup();
            for gi in dirty {
                let v = o.vol.grid[gi as usize];
                let enc = if v < 256 {
                    v
                } else {
                    let (e, u) = self.encode_brick(&mut o.vol.bricks[(v - 256) as usize], false);
                    if u {
                        up.push(BrickRef::Obj(oi as u32, v - 256));
                    }
                    e
                };
                writes.push((self.grid_offsets[oi] as usize + gi as usize, enc));
            }
        }
        if writes.is_empty() {
            return;
        }
        ctx.wait_idle(); // frames in flight read objGrid / the atlas
        for (i, e) in writes {
            self.obj_grid_buf.write(i * 4, &e.to_le_bytes());
        }
        self.upload_bricks_now(ctx, world, objects, &up);
    }

    /// Upload bricks (+ masks) to their atlas slots in one waited submit.
    fn upload_bricks_now(&mut self, ctx: &Context, world: &World, objects: &[VoxObject], up: &[BrickRef]) {
        if up.is_empty() {
            return;
        }
        let mut stage = vec![0u8; up.len() * 512];
        let mut masks = vec![[0u32; 16]; up.len()];
        let mut regions = Vec::with_capacity(up.len());
        let mut mregions = Vec::with_capacity(up.len());
        for (i, &r) in up.iter().enumerate() {
            let b = Self::brick(world, objects, r);
            gpu_voxels(&mut stage[i * 512..i * 512 + 512], &b.vox);
            masks[i] = masks_of(b);
            let s = b.gpu_slot;
            regions.push(Self::brick_region(i as u64 * 512, s));
            mregions.push(vk::BufferCopy { src_offset: i as u64 * 64, dst_offset: s as u64 * 64, size: 64 });
        }
        let mut sa = ctx.upload_buffer("stageBricks", &stage, vk::BufferUsageFlags::TRANSFER_SRC);
        let mut sm = ctx.upload_buffer("stageMasks", bytemuck::cast_slice(&masks), vk::BufferUsageFlags::TRANSFER_SRC);
        let dev = &ctx.device;
        let cb = ctx.begin_one_shot();
        let (ro, dst) = (vk::ImageLayout::SHADER_READ_ONLY_OPTIMAL, vk::ImageLayout::TRANSFER_DST_OPTIMAL);
        image_barrier(dev, cb, self.atlas.image, Stage::COMPUTE_SHADER, Access::SHADER_SAMPLED_READ, Stage::COPY, Access::TRANSFER_WRITE, ro, dst);
        unsafe {
            dev.cmd_copy_buffer_to_image(cb, sa.buf, self.atlas.image, dst, &regions);
            dev.cmd_copy_buffer(cb, sm.buf, self.mask_buf.buf, &mregions);
        }
        image_barrier(dev, cb, self.atlas.image, Stage::COPY, Access::TRANSFER_WRITE, Stage::COMPUTE_SHADER, Access::SHADER_SAMPLED_READ, dst, ro);
        memory_barrier(dev, cb, Stage::COPY, Access::TRANSFER_WRITE, Stage::COMPUTE_SHADER, Access::SHADER_STORAGE_READ);
        ctx.end_one_shot_and_wait(cb);
        sa.destroy(ctx);
        sm.destroy(ctx);
    }

    fn brick_region(off: u64, slot: i32) -> vk::BufferImageCopy {
        let (sx, sy, sz) = (slot % ABX, (slot / ABX) % ABY, slot / (ABX * ABY));
        vk::BufferImageCopy::default()
            .buffer_offset(off)
            .image_subresource(COLOR_LAYERS)
            .image_offset(vk::Offset3D { x: sx * 8, y: sy * 8, z: sz * 8 })
            .image_extent(vk::Extent3D { width: 8, height: 8, depth: 8 })
    }

    /// Re-upload the 256-entry palette (after new colours were allocated).
    pub fn update_palette(&mut self, ctx: &Context, palette: &[Material]) {
        ctx.wait_idle();
        self.palette_buf.write(0, bytemuck::cast_slice(palette));
    }

    // --- staging ring ------------------------------------------------------------------------------
    /// Copy `data` into the staging ring (or a temporary buffer when it doesn't fit); returns
    /// (buffer, offset). Every user must finish through `submit_sync`.
    pub fn stage(&mut self, ctx: &Context, data: &[u8]) -> (vk::Buffer, u64) {
        let size = data.len() as u64;
        if self.ring.is_null() {
            self.ring = ctx.create_buffer("stagingRing", RING_SIZE, vk::BufferUsageFlags::TRANSFER_SRC, MemoryLocation::CpuToGpu);
            self.ring_head = 0;
            self.ring_tail = 0;
        }
        if !self.sync_open {
            if self.pending.is_empty() {
                self.ring_head = 0;
                self.ring_tail = 0;
            }
            self.sync_open = true;
            self.sync_begin = self.ring_head;
        }
        if size + 16 < RING_SIZE {
            loop {
                let at = (self.ring_head + 15) & !15;
                let wrapped = self.ring_head < self.ring_tail;
                let put = if !wrapped {
                    if at + size <= RING_SIZE {
                        Some(at)
                    } else if size < self.ring_tail {
                        Some(0)
                    } else {
                        None
                    }
                } else if at + size < self.ring_tail {
                    Some(at)
                } else {
                    None
                };
                if let Some(p) = put {
                    self.ring.write(p as usize, data);
                    self.ring_head = p + size;
                    return (self.ring.buf, p);
                }
                if self.pending.is_empty() {
                    break;
                }
                self.retire_oldest(ctx);
            }
        }
        let t = ctx.upload_buffer("stagingTemp", data, vk::BufferUsageFlags::TRANSFER_SRC);
        let b = t.buf;
        self.sync_temps.push(t);
        (b, 0)
    }

    /// Finished syncs, oldest first: their resources freed, the ring's tail moved past them.
    pub fn retire_syncs(&mut self, ctx: &Context, wait: bool) {
        while let Some(p) = self.pending.first() {
            let done = if wait {
                ctx.wait_fence(p.fence);
                true
            } else {
                unsafe { ctx.device.get_fence_status(p.fence) }.unwrap_or(false)
            };
            if !done {
                break;
            }
            let mut p = self.pending.remove(0);
            unsafe { ctx.device.reset_fences(&[p.fence]).ok() };
            self.free_fences.push(p.fence);
            self.free_cbs.push(p.cb);
            for t in &mut p.temps {
                t.destroy(ctx);
            }
            self.ring_tail = p.ring_end;
        }
        if self.pending.is_empty() {
            if self.sync_open {
                self.ring_tail = self.sync_begin;
            } else {
                self.ring_head = 0;
                self.ring_tail = 0;
            }
        }
    }

    fn retire_oldest(&mut self, ctx: &Context) {
        if let Some(p) = self.pending.first() {
            ctx.wait_fence(p.fence);
        }
        self.retire_syncs(ctx, false);
    }

    pub fn begin_sync(&mut self, ctx: &Context) -> vk::CommandBuffer {
        let cb = self.free_cbs.pop().unwrap_or_else(|| ctx.alloc_cmd());
        ctx.begin_cmd(cb);
        cb
    }

    pub fn submit_sync(&mut self, ctx: &Context, cb: vk::CommandBuffer) {
        let fence = self.free_fences.pop().unwrap_or_else(|| ctx.create_fence(false));
        ctx.submit(cb, fence);
        self.pending.push(PendingSync { fence, cb, temps: std::mem::take(&mut self.sync_temps), ring_end: self.ring_head });
        self.sync_open = false;
    }

    // --- incremental world sync -----------------------------------------------------------------------
    fn coarse_cell1(&self, world: &World, ci: usize) -> u8 {
        let c = self.c1;
        let (cx, cy, cz) = (ci as i32 % c[0], (ci as i32 / c[0]) % c[1], ci as i32 / (c[0] * c[1]));
        for z in cz * C1..((cz + 1) * C1).min(self.wbz) {
            for y in cy * C1..((cy + 1) * C1).min(self.wby) {
                for x in cx * C1..((cx + 1) * C1).min(self.wbx) {
                    if world.grid[((z * self.wby + y) * self.wbx + x) as usize] != 0 {
                        return 1;
                    }
                }
            }
        }
        0
    }

    fn coarse_cell2(&self, ci: usize) -> u8 {
        let (c, d) = (self.c2, self.c1);
        let r = C2 / C1;
        let (cx, cy, cz) = (ci as i32 % c[0], (ci as i32 / c[0]) % c[1], ci as i32 / (c[0] * c[1]));
        for z in cz * r..((cz + 1) * r).min(d[2]) {
            for y in cy * r..((cy + 1) * r).min(d[1]) {
                for x in cx * r..((cx + 1) * r).min(d[0]) {
                    if self.coarse1_cpu[((z * d[1] + y) * d[0] + x) as usize] != 0 {
                        return 1;
                    }
                }
            }
        }
        0
    }

    fn cell_region(off: u64, x: i32, y: i32, z: i32, w: u32, h: u32, d: u32) -> vk::BufferImageCopy {
        vk::BufferImageCopy::default()
            .buffer_offset(off)
            .image_subresource(COLOR_LAYERS)
            .image_offset(vk::Offset3D { x, y, z })
            .image_extent(vk::Extent3D { width: w, height: h, depth: d })
    }

    /// Incremental re-upload of the world's dirty grid cells + their bricks + the coarse cells,
    /// then the distance field + grass over each cluster of edits. No device stall.
    pub fn sync(&mut self, ctx: &Context, world: &mut World) {
        if world.dirty_grid.is_empty() {
            return;
        }
        self.retire_syncs(ctx, false);
        let mut dirty = std::mem::take(&mut world.vol.dirty_grid);
        dirty.sort_unstable();
        dirty.dedup();
        let mut grid_vals = Vec::with_capacity(dirty.len());
        let mut grid_regions = Vec::with_capacity(dirty.len());
        let mut up: Vec<u32> = Vec::new();
        for &gi in &dirty {
            // An edited brick that became uniform turns back into a uniform cell and frees its slot.
            if let Some(freed) = world.vol.compact_brick(gi as usize) {
                self.bricks_compacted += 1;
                if freed >= 0 {
                    self.world_slot[freed as usize] = 0;
                    self.free_slots.push(freed as u32);
                }
            }
            let v = world.grid[gi as usize];
            let enc = if v < 256 {
                v
            } else {
                let (e, u) = self.encode_brick(&mut world.vol.bricks[(v - 256) as usize], true);
                if u {
                    up.push(v - 256);
                }
                e
            };
            let g = gi as i32;
            let (bx, by, bz) = (g % self.wbx, (g / self.wbx) % self.wby, g / (self.wbx * self.wby));
            grid_regions.push(Self::cell_region(grid_vals.len() as u64 * 4, bx, by, bz, 1, 1, 1));
            grid_vals.push(enc);
        }
        let mut atlas_stage = vec![0u8; up.len() * 512];
        let mut atlas_regions = Vec::with_capacity(up.len());
        let mut mask_data = vec![[0u32; 16]; up.len()];
        let mut mask_regions = Vec::with_capacity(up.len());
        for (i, &bi) in up.iter().enumerate() {
            let b = &world.bricks[bi as usize];
            gpu_voxels(&mut atlas_stage[i * 512..i * 512 + 512], &b.vox);
            atlas_regions.push(Self::brick_region(i as u64 * 512, b.gpu_slot));
            mask_data[i] = masks_of(b);
            mask_regions.push(vk::BufferCopy { src_offset: i as u64 * 64, dst_offset: b.gpu_slot as u64 * 64, size: 64 });
        }
        // Coarse pyramid: recompute the touched cells.
        let mut c1cells: Vec<u32> = Vec::new();
        let mut c2cells: Vec<u32> = Vec::new();
        for &gi in &dirty {
            let g = gi as i32;
            let (bx, by, bz) = (g % self.wbx, (g / self.wbx) % self.wby, g / (self.wbx * self.wby));
            c1cells.push((((bz / C1) * self.c1[1] + by / C1) * self.c1[0] + bx / C1) as u32);
            c2cells.push((((bz / C2) * self.c2[1] + by / C2) * self.c2[0] + bx / C2) as u32);
        }
        c1cells.sort_unstable();
        c1cells.dedup();
        c2cells.sort_unstable();
        c2cells.dedup();
        let mut c1vals = Vec::new();
        let mut c1regions = Vec::new();
        for &ci in &c1cells {
            let occ = self.coarse_cell1(world, ci as usize);
            self.coarse1_cpu[ci as usize] = occ;
            let c = self.c1;
            let ci = ci as i32;
            c1regions.push(Self::cell_region(c1vals.len() as u64, ci % c[0], (ci / c[0]) % c[1], ci / (c[0] * c[1]), 1, 1, 1));
            c1vals.push(occ);
        }
        let mut c2vals = Vec::new();
        let mut c2regions = Vec::new();
        for &ci in &c2cells {
            let occ = self.coarse_cell2(ci as usize);
            self.coarse2_cpu[ci as usize] = occ;
            let c = self.c2;
            let ci = ci as i32;
            c2regions.push(Self::cell_region(c2vals.len() as u64, ci % c[0], (ci / c[0]) % c[1], ci / (c[0] * c[1]), 1, 1, 1));
            c2vals.push(occ);
        }
        // Stage everything.
        let shift = |rs: &mut [vk::BufferImageCopy], base: u64| rs.iter_mut().for_each(|r| r.buffer_offset += base);
        let (b_grid, off) = self.stage(ctx, bytemuck::cast_slice(&grid_vals));
        shift(&mut grid_regions, off);
        let mut b_atlas = None;
        let mut b_mask = None;
        if !up.is_empty() {
            let (b, off) = self.stage(ctx, &atlas_stage);
            shift(&mut atlas_regions, off);
            b_atlas = Some(b);
            let (b, off) = self.stage(ctx, bytemuck::cast_slice(&mask_data));
            mask_regions.iter_mut().for_each(|r| r.src_offset += off);
            b_mask = Some(b);
        }
        let (b_c1, off) = self.stage(ctx, &c1vals);
        shift(&mut c1regions, off);
        let (b_c2, off) = self.stage(ctx, &c2vals);
        shift(&mut c2regions, off);

        let dev = ctx.device.clone();
        let cb = self.begin_sync(ctx);
        // Frames submitted before this may still read what it overwrites.
        memory_barrier(&dev, cb, Stage::ALL_COMMANDS, Access::MEMORY_WRITE, Stage::ALL_COMMANDS, Access::MEMORY_READ | Access::MEMORY_WRITE);
        self.record_copies(&dev, cb, b_grid, &grid_regions, b_atlas, &atlas_regions, b_mask, &mask_regions, b_c1, &c1regions, b_c2, &c2regions);
        // Distance field + grass tops per CLUSTER of dirty bricks (each box grown by the cap).
        let mut boxes: Vec<BrickBox> = Vec::new();
        for &gi in &dirty {
            let (bx, by, bz) = world.brick_coords(gi as usize);
            let b = [bx, by, bz];
            if let Some(x) = boxes.iter_mut().find(|x| (0..3).all(|k| b[k] >= x.lo[k] - DIST_CAP && b[k] <= x.hi[k] + DIST_CAP)) {
                for k in 0..3 {
                    x.lo[k] = x.lo[k].min(b[k]);
                    x.hi[k] = x.hi[k].max(b[k]);
                }
            } else {
                boxes.push(BrickBox { lo: b, hi: b });
            }
        }
        let mut merged = true;
        while merged {
            merged = false;
            'outer: for i in 0..boxes.len() {
                for j in i + 1..boxes.len() {
                    let near = (0..3).all(|k| boxes[j].lo[k] <= boxes[i].hi[k] + 2 * DIST_CAP && boxes[j].hi[k] >= boxes[i].lo[k] - 2 * DIST_CAP);
                    if near {
                        let bj = boxes.remove(j);
                        for k in 0..3 {
                            boxes[i].lo[k] = boxes[i].lo[k].min(bj.lo[k]);
                            boxes[i].hi[k] = boxes[i].hi[k].max(bj.hi[k]);
                        }
                        merged = true;
                        break 'outer;
                    }
                }
            }
        }
        self.edit_lo = [i32::MAX; 3];
        self.edit_hi = [i32::MIN; 3];
        for x in &boxes {
            self.record_grass_top(ctx, cb, x.lo[0] * 8, x.lo[2] * 8, x.hi[0] * 8 + 7, x.hi[2] * 8 + 7);
            for k in 0..3 {
                self.edit_lo[k] = self.edit_lo[k].min(x.lo[k] * 8);
                self.edit_hi[k] = self.edit_hi[k].max(x.hi[k] * 8 + 8);
            }
            self.edit_boxes.push([x.lo[0] * 8, x.lo[1] * 8, x.lo[2] * 8, x.hi[0] * 8 + 8, x.hi[1] * 8 + 8, x.hi[2] * 8 + 8]);
            let lo = [x.lo[0] - DIST_CAP, x.lo[1] - DIST_CAP, x.lo[2] - DIST_CAP];
            let hi = [x.hi[0] + DIST_CAP, x.hi[1] + DIST_CAP, x.hi[2] + DIST_CAP];
            self.record_distance_field(ctx, cb, lo, hi, x.lo, x.hi);
        }
        self.edit_counter += 1;
        self.submit_sync(ctx, cb);
        log::trace!(target: "gpu-world", "sync: {} cells, {} bricks, {} compacted so far", dirty.len(), up.len(), self.bricks_compacted);
        self.synced_grid.extend_from_slice(&dirty);
    }

    #[allow(clippy::too_many_arguments)]
    fn record_copies(
        &self,
        dev: &ash::Device,
        cb: vk::CommandBuffer,
        b_grid: vk::Buffer,
        grid_regions: &[vk::BufferImageCopy],
        b_atlas: Option<vk::Buffer>,
        atlas_regions: &[vk::BufferImageCopy],
        b_mask: Option<vk::Buffer>,
        mask_regions: &[vk::BufferCopy],
        b_c1: vk::Buffer,
        c1regions: &[vk::BufferImageCopy],
        b_c2: vk::Buffer,
        c2regions: &[vk::BufferImageCopy],
    ) {
        let (ro, dst) = (vk::ImageLayout::SHADER_READ_ONLY_OPTIMAL, vk::ImageLayout::TRANSFER_DST_OPTIMAL);
        let imgs = [self.grid.image, self.atlas.image, self.coarse1.image, self.coarse2.image];
        for &im in &imgs {
            image_barrier(dev, cb, im, Stage::COMPUTE_SHADER, Access::SHADER_SAMPLED_READ, Stage::COPY, Access::TRANSFER_WRITE, ro, dst);
        }
        unsafe {
            if !grid_regions.is_empty() {
                dev.cmd_copy_buffer_to_image(cb, b_grid, self.grid.image, dst, grid_regions);
            }
            if let Some(b) = b_atlas {
                if !atlas_regions.is_empty() {
                    dev.cmd_copy_buffer_to_image(cb, b, self.atlas.image, dst, atlas_regions);
                }
            }
            if let Some(b) = b_mask {
                if !mask_regions.is_empty() {
                    dev.cmd_copy_buffer(cb, b, self.mask_buf.buf, mask_regions);
                }
            }
            if !c1regions.is_empty() {
                dev.cmd_copy_buffer_to_image(cb, b_c1, self.coarse1.image, dst, c1regions);
            }
            if !c2regions.is_empty() {
                dev.cmd_copy_buffer_to_image(cb, b_c2, self.coarse2.image, dst, c2regions);
            }
        }
        for &im in &imgs {
            image_barrier(dev, cb, im, Stage::COPY, Access::TRANSFER_WRITE, Stage::COMPUTE_SHADER, Access::SHADER_SAMPLED_READ, dst, ro);
        }
        memory_barrier(dev, cb, Stage::COPY, Access::TRANSFER_WRITE, Stage::COMPUTE_SHADER, Access::SHADER_STORAGE_READ);
    }

    /// Streamed chunks in / out (ring mode): one grid region per chunk, its atlas bricks (runs
    /// of consecutive slots batched), the coarse cells over it, then distance field + grass.
    pub fn sync_chunks(&mut self, ctx: &Context, world: &mut World, loaded: &[(i32, i32)], evicted: &[(i32, i32)]) {
        if !self.ring_mode || (loaded.is_empty() && evicted.is_empty()) {
            return;
        }
        self.retire_syncs(ctx, false);
        if !world.released_gpu_slots.is_empty() {
            let mut rel = std::mem::take(&mut world.vol.released_gpu_slots);
            rel.sort_unstable_by(|a, b| b.cmp(a)); // pops hand out ascending, mostly consecutive slots
            for sl in rel {
                if sl >= 0 && (sl as usize) < self.world_slot.len() && self.world_slot[sl as usize] != 0 {
                    self.world_slot[sl as usize] = 0;
                    self.free_slots.push(sl as u32);
                }
            }
        }
        let cbk = RING_CHUNK_B;
        let (c1, c2) = (self.c1, self.c2);
        let ring_w = world.ring_w;
        let mut grid_vals: Vec<u32> = Vec::new();
        let (mut grid_regions, mut c1regions, mut c2regions) = (Vec::new(), Vec::new(), Vec::new());
        let (mut c1vals, mut c2vals): (Vec<u8>, Vec<u8>) = (Vec::new(), Vec::new());
        let mut up: Vec<u32> = Vec::new();
        let mut boxes: Vec<BrickBox> = Vec::new();
        let mut slot_done = vec![false; (ring_w * ring_w) as usize];
        let mut zero: Option<(usize, usize, usize)> = None;
        for &(cx, cz) in loaded.iter().chain(evicted.iter()) {
            let slot = ((cz & (ring_w - 1)) * ring_w + (cx & (ring_w - 1))) as usize;
            if slot_done[slot] {
                continue;
            }
            slot_done[slot] = true;
            let (tx, tz) = ((cx * cbk) & (self.wbx - 1), (cz * cbk) & (self.wbz - 1));
            boxes.push(BrickBox { lo: [cx * cbk, 0, cz * cbk], hi: [cx * cbk + cbk - 1, self.wby - 1, cz * cbk + cbk - 1] });
            if !world.chunk_loaded(cx, cz) {
                // Evicted, slot not refilled: every copy reads one shared zero block.
                let (zg, z1, z2) = *zero.get_or_insert_with(|| {
                    let r = (grid_vals.len(), c1vals.len(), c2vals.len());
                    grid_vals.resize(grid_vals.len() + (cbk * self.wby * cbk) as usize, 0);
                    c1vals.resize(c1vals.len() + ((cbk / C1) * c1[1] * (cbk / C1)) as usize, 0);
                    c2vals.resize(c2vals.len() + ((cbk / C2) * c2[1] * (cbk / C2)) as usize, 0);
                    r
                });
                grid_regions.push(Self::cell_region(zg as u64 * 4, tx, 0, tz, cbk as u32, self.wby as u32, cbk as u32));
                c1regions.push(Self::cell_region(z1 as u64, tx / C1, 0, tz / C1, (cbk / C1) as u32, c1[1] as u32, (cbk / C1) as u32));
                c2regions.push(Self::cell_region(z2 as u64, tx / C2, 0, tz / C2, (cbk / C2) as u32, c2[1] as u32, (cbk / C2) as u32));
                for z in tz / C1..(tz + cbk) / C1 {
                    for y in 0..c1[1] {
                        let i = ((z * c1[1] + y) * c1[0] + tx / C1) as usize;
                        self.coarse1_cpu[i..i + (cbk / C1) as usize].fill(0);
                    }
                }
                for z in tz / C2..(tz + cbk) / C2 {
                    for y in 0..c2[1] {
                        let i = ((z * c2[1] + y) * c2[0] + tx / C2) as usize;
                        self.coarse2_cpu[i..i + (cbk / C2) as usize].fill(0);
                    }
                }
                continue;
            }
            grid_regions.push(Self::cell_region(grid_vals.len() as u64 * 4, tx, 0, tz, cbk as u32, self.wby as u32, cbk as u32));
            for z in 0..cbk {
                for y in 0..self.wby {
                    for x in 0..cbk {
                        let gi = (((tz + z) * self.wby + y) * self.wbx + tx + x) as usize;
                        let v = world.grid[gi];
                        let enc = if v < 256 {
                            v
                        } else {
                            let (e, u) = self.encode_brick(&mut world.vol.bricks[(v - 256) as usize], true);
                            if u {
                                up.push(v - 256);
                            }
                            e
                        };
                        grid_vals.push(enc);
                    }
                }
            }
            c1regions.push(Self::cell_region(c1vals.len() as u64, tx / C1, 0, tz / C1, (cbk / C1) as u32, c1[1] as u32, (cbk / C1) as u32));
            for z in tz / C1..(tz + cbk) / C1 {
                for y in 0..c1[1] {
                    for x in tx / C1..(tx + cbk) / C1 {
                        let ci = ((z * c1[1] + y) * c1[0] + x) as usize;
                        let occ = self.coarse_cell1(world, ci);
                        self.coarse1_cpu[ci] = occ;
                        c1vals.push(occ);
                    }
                }
            }
            c2regions.push(Self::cell_region(c2vals.len() as u64, tx / C2, 0, tz / C2, (cbk / C2) as u32, c2[1] as u32, (cbk / C2) as u32));
            for z in tz / C2..(tz + cbk) / C2 {
                for y in 0..c2[1] {
                    for x in tx / C2..(tx + cbk) / C2 {
                        let ci = ((z * c2[1] + y) * c2[0] + x) as usize;
                        let occ = self.coarse_cell2(ci);
                        self.coarse2_cpu[ci] = occ;
                        c2vals.push(occ);
                    }
                }
            }
        }
        self.chunk_events.extend_from_slice(loaded);
        self.chunk_events.extend_from_slice(evicted);
        // Atlas: runs of consecutive slots along a row -> one region each.
        up.sort_unstable_by_key(|&bi| world.bricks[bi as usize].gpu_slot);
        let mut atlas_stage = vec![0u8; up.len() * 512];
        let mut atlas_regions = Vec::new();
        let mut mask_data = vec![[0u32; 16]; up.len()];
        let mut mask_regions = Vec::new();
        let mut i = 0;
        while i < up.len() {
            let s0 = world.bricks[up[i] as usize].gpu_slot;
            let mut n = 1usize;
            while i + n < up.len() && world.bricks[up[i + n] as usize].gpu_slot == s0 + n as i32 && (s0 % ABX) + (n as i32) < ABX {
                n += 1;
            }
            let base = i * 512;
            for z in 0..8 {
                for y in 0..8 {
                    for k in 0..n {
                        let d = base + ((z * 8 + y) * n + k) * 8;
                        let src = &world.bricks[up[i + k] as usize].vox[z * 64 + y * 8..z * 64 + y * 8 + 8];
                        gpu_voxels(&mut atlas_stage[d..d + 8], src);
                    }
                }
            }
            atlas_regions.push(
                vk::BufferImageCopy::default()
                    .buffer_offset(base as u64)
                    .buffer_row_length((8 * n) as u32)
                    .buffer_image_height(8)
                    .image_subresource(COLOR_LAYERS)
                    .image_offset(vk::Offset3D { x: (s0 % ABX) * 8, y: ((s0 / ABX) % ABY) * 8, z: (s0 / (ABX * ABY)) * 8 })
                    .image_extent(vk::Extent3D { width: (8 * n) as u32, height: 8, depth: 8 }),
            );
            for k in 0..n {
                mask_data[i + k] = masks_of(&world.bricks[up[i + k] as usize]);
            }
            mask_regions.push(vk::BufferCopy { src_offset: i as u64 * 64, dst_offset: s0 as u64 * 64, size: n as u64 * 64 });
            i += n;
        }
        let shift = |rs: &mut [vk::BufferImageCopy], base: u64| rs.iter_mut().for_each(|r| r.buffer_offset += base);
        let (b_grid, off) = self.stage(ctx, bytemuck::cast_slice(&grid_vals));
        shift(&mut grid_regions, off);
        let (mut b_atlas, mut b_mask) = (None, None);
        if !up.is_empty() {
            let (b, off) = self.stage(ctx, &atlas_stage);
            shift(&mut atlas_regions, off);
            b_atlas = Some(b);
            let (b, off) = self.stage(ctx, bytemuck::cast_slice(&mask_data));
            mask_regions.iter_mut().for_each(|r| r.src_offset += off);
            b_mask = Some(b);
        }
        let (b_c1, off) = self.stage(ctx, &c1vals);
        shift(&mut c1regions, off);
        let (b_c2, off) = self.stage(ctx, &c2vals);
        shift(&mut c2regions, off);
        let dev = ctx.device.clone();
        let cb = self.begin_sync(ctx);
        memory_barrier(&dev, cb, Stage::ALL_COMMANDS, Access::MEMORY_WRITE, Stage::ALL_COMMANDS, Access::MEMORY_READ | Access::MEMORY_WRITE);
        self.record_copies(&dev, cb, b_grid, &grid_regions, b_atlas, &atlas_regions, b_mask, &mask_regions, b_c1, &c1regions, b_c2, &c2regions);
        for b in &boxes {
            let lo = [b.lo[0] - DIST_CAP, b.lo[1] - DIST_CAP, b.lo[2] - DIST_CAP];
            let hi = [b.hi[0] + DIST_CAP, b.hi[1] + DIST_CAP, b.hi[2] + DIST_CAP];
            self.record_distance_field(ctx, cb, lo, hi, b.lo, b.hi);
            self.record_grass_top(ctx, cb, b.lo[0] * 8, b.lo[2] * 8, b.hi[0] * 8 + 7, b.hi[2] * 8 + 7);
            self.edit_boxes.push([b.lo[0] * 8, 0, b.lo[2] * 8, b.hi[0] * 8 + 8, self.wby * 8, b.hi[2] * 8 + 8]);
        }
        self.edit_counter += 1;
        self.submit_sync(ctx, cb);
        log::trace!(target: "gpu-world", "chunks: {} in, {} out, {} bricks, atlas {} / {}", loaded.len(), evicted.len(), up.len(),
                    self.atlas_slots, self.atlas_capacity());
    }

    // --- derived fields ---------------------------------------------------------------------------
    /// The brick Chebyshev distance field over bricks [lo, hi] (separable passes); the dry
    /// occupancy scan covers only the changed bricks [dirty_lo, dirty_hi].
    pub fn record_distance_field(&self, ctx: &Context, cb: vk::CommandBuffer, mut lo: [i32; 3], mut hi: [i32; 3], dirty_lo: [i32; 3], dirty_hi: [i32; 3]) {
        let dims = [self.wbx, self.wby, self.wbz];
        let (mut dlo, mut dhi) = (dirty_lo, dirty_hi);
        for k in 0..3 {
            if self.ring_mode && k != 1 {
                dhi[k] = dhi[k].min(dlo[k] + dims[k] - 1);
                hi[k] = hi[k].min(lo[k] + dims[k] - 1);
                continue;
            }
            lo[k] = lo[k].clamp(0, dims[k] - 1);
            hi[k] = hi[k].clamp(0, dims[k] - 1);
            dlo[k] = dlo[k].clamp(0, dims[k] - 1);
            dhi[k] = dhi[k].clamp(0, dims[k] - 1);
        }
        #[repr(C)]
        #[derive(Clone, Copy, Pod, Zeroable)]
        struct Pc {
            box_min: [i32; 3],
            axis: i32,
            box_max: [i32; 3],
            first_pass: i32,
        }
        let dev = &ctx.device;
        unsafe { dev.cmd_bind_pipeline(cb, vk::PipelineBindPoint::COMPUTE, self.dist_pipe.pipeline) };
        for pass in 0..7 {
            if pass > 0 {
                memory_barrier(dev, cb, Stage::COMPUTE_SHADER, Access::SHADER_STORAGE_WRITE, Stage::COMPUTE_SHADER,
                               Access::SHADER_SAMPLED_READ | Access::SHADER_STORAGE_WRITE);
            }
            let mut pc = Pc { box_min: lo, axis: pass % 3, box_max: hi, first_pass: (pass == 0) as i32 };
            if pass == 3 {
                pc = Pc { box_min: dlo, axis: ABX | (ABY << 10), box_max: dhi, first_pass: 2 };
            }
            if pass >= 4 {
                pc.axis = pass - 4;
                pc.first_pass = 0;
            }
            if self.ring_mode {
                pc.first_pass |= 4;
            }
            unsafe {
                dev.cmd_bind_descriptor_sets(cb, vk::PipelineBindPoint::COMPUTE, self.dist_pipe.layout, 0, &[self.dist_sets[pass as usize]], &[]);
                dev.cmd_push_constants(cb, self.dist_pipe.layout, vk::ShaderStageFlags::COMPUTE, 0, bytemuck::bytes_of(&pc));
                dev.cmd_dispatch(
                    cb,
                    ((pc.box_max[0] - pc.box_min[0] + 4) / 4) as u32,
                    ((pc.box_max[1] - pc.box_min[1] + 4) / 4) as u32,
                    ((pc.box_max[2] - pc.box_min[2] + 4) / 4) as u32,
                );
            }
        }
        memory_barrier(dev, cb, Stage::COMPUTE_SHADER, Access::SHADER_STORAGE_WRITE, Stage::COMPUTE_SHADER, Access::SHADER_SAMPLED_READ);
    }

    /// Rebuild grass-top / sky-top columns [x0..x1] x [z0..z1] (voxels, inclusive) + tiles.
    pub fn record_grass_top(&self, ctx: &Context, cb: vk::CommandBuffer, mut x0: i32, mut z0: i32, mut x1: i32, mut z1: i32) {
        let (nx, nz) = (self.wbx * 8, self.wbz * 8);
        if self.ring_mode {
            x1 = x1.min(x0 + nx - 1);
            z1 = z1.min(z0 + nz - 1);
        } else {
            x0 = x0.clamp(0, nx - 1);
            x1 = x1.clamp(0, nx - 1);
            z0 = z0.clamp(0, nz - 1);
            z1 = z1.clamp(0, nz - 1);
        }
        #[repr(C)]
        #[derive(Clone, Copy, Pod, Zeroable)]
        struct Pc {
            lo: [i32; 2],
            hi: [i32; 2],
            pass: i32,
            abx: i32,
            aby: i32,
            ring: i32,
        }
        let dev = &ctx.device;
        let barrier = || {
            memory_barrier(dev, cb, Stage::COMPUTE_SHADER, Access::SHADER_STORAGE_WRITE, Stage::COMPUTE_SHADER,
                           Access::SHADER_STORAGE_READ | Access::SHADER_SAMPLED_READ)
        };
        let ring = self.ring_mode as i32;
        self.grass_pipe.bind(dev, cb, &[self.grass_set]);
        self.grass_pipe.push(dev, cb, &Pc { lo: [x0, z0], hi: [x1, z1], pass: 0, abx: ABX, aby: ABY, ring });
        unsafe { dev.cmd_dispatch(cb, ((x1 - x0 + 8) / 8) as u32, ((z1 - z0 + 8) / 8) as u32, 1) };
        barrier();
        self.grass_pipe.push(dev, cb, &Pc { lo: [x0 >> 3, z0 >> 3], hi: [x1 >> 3, z1 >> 3], pass: 1, abx: ABX, aby: ABY, ring });
        unsafe { dev.cmd_dispatch(cb, (((x1 >> 3) - (x0 >> 3) + 8) / 8) as u32, (((z1 >> 3) - (z0 >> 3) + 8) / 8) as u32, 1) };
        barrier();
    }

    pub fn destroy(&mut self, ctx: &Context) {
        self.retire_syncs(ctx, true);
        let dev = &ctx.device;
        unsafe {
            for f in self.free_fences.drain(..) {
                dev.destroy_fence(f, None);
            }
            if !self.free_cbs.is_empty() {
                dev.free_command_buffers(ctx.cmd_pool, &self.free_cbs);
            }
            self.free_cbs.clear();
            self.grass_pipe.destroy(dev);
            self.dist_pipe.destroy(dev);
            for p in [self.grass_pool, self.dist_pool] {
                if p != vk::DescriptorPool::null() {
                    dev.destroy_descriptor_pool(p, None);
                }
            }
            for l in [self.grass_layout, self.dist_layout] {
                if l != vk::DescriptorSetLayout::null() {
                    dev.destroy_descriptor_set_layout(l, None);
                }
            }
            if self.sampler != vk::Sampler::null() {
                dev.destroy_sampler(self.sampler, None);
            }
        }
        self.grass_pool = vk::DescriptorPool::null();
        self.dist_pool = vk::DescriptorPool::null();
        self.grass_layout = vk::DescriptorSetLayout::null();
        self.dist_layout = vk::DescriptorSetLayout::null();
        self.sampler = vk::Sampler::null();
        for im in [&mut self.grid, &mut self.atlas, &mut self.coarse1, &mut self.coarse2, &mut self.grass_top, &mut self.grass_tile, &mut self.grass_info, &mut self.sky_top] {
            im.destroy(ctx);
        }
        for d in &mut self.dist {
            d.destroy(ctx);
        }
        for b in [&mut self.palette_buf, &mut self.obj_grid_buf, &mut self.mask_buf, &mut self.ring] {
            b.destroy(ctx);
        }
        for k in 0..2 {
            self.objects_buf[k].destroy(ctx);
            self.bvh_node_buf[k].destroy(ctx);
            self.bvh_prim_buf[k].destroy(ctx);
        }
        self.objects_cap = 0;
        self.bvh_capacity = 0;
        self.slot_version = [0; 2];
    }
}
