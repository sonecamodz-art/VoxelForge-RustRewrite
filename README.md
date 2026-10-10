# VoxelForge (Rust)

A GPU ray-traced voxel engine on Vulkan compute, rewritten in Rust from the C++ engine.
1 voxel = 10 cm; a 2048 x 1024 x 2048 world of sparse 8³ bricks, traced per pixel on the GPU
(brick grid + atlas + a coarse occupancy pyramid + a brick distance field), lit by either

- **ray traced** lighting: traced sun + GI rays per pixel, denoised by an SVGF chain
  (temporal accumulation, à-trous wavelet filter, GI stabilisation), or
- **Standard** lighting: a cached cascaded sun shadow map, a DDGI probe grid and one exact sun
  ray per pixel softened in screen space (no denoiser).

Plus transparent water with traced reflections, procedural grass with wind, froxel volumetric
fog, voxel clouds, emissive lamps / torches, auto exposure, bloom and ACES / AgX tonemapping.

The GLSL compute shaders (`shaders/`) are shared verbatim with the C++ engine.

## Building

Needs a Rust toolchain (1.85+, edition 2024), the Vulkan loader and **shaderc** (part of the
Vulkan SDK).

```sh
cargo build --release
./target/release/voxelforge
```

If the build can't find shaderc, point `SHADERC_LIB_DIR` at the directory holding
`libshaderc_shared.so` (Linux) / `shaderc_shared.lib` (Windows), or set `VULKAN_SDK`.
Ubuntu's `libshaderc-dev` names the library `libshaderc.so`; a symlink fixes that:

```sh
mkdir -p ~/shaderc && ln -s /usr/lib/x86_64-linux-gnu/libshaderc.so.1 ~/shaderc/libshaderc_shared.so
SHADERC_LIB_DIR=~/shaderc cargo build --release
```

Linux windows also need `libxkbcommon-x11` (installed on every desktop).

## Running

```
voxelforge [--winsize WxH] [--validation] [+set <cvar> <value>] [+exec "cmd"]
           [--shot out.png [--frames N] [--dump final|albedo|normal|motion]]
```

Window keys: click to capture the mouse (Esc frees it, Esc again quits), WASD to walk
(Shift sprint, Ctrl / C crouch, Space jump, double-tap Space to fly / land; flying: Space / Ctrl
up / down, wheel = speed), Tab toggles the free camera (no collision), LMB carve / RMB place a sphere (`-` / `=` size, 1-9
material), F3 stats in the title, F4 debug views, F6 render scale, F10 reload shaders,
F11 fullscreen, P pause physics, L ray traced / Standard lighting, N SVGF / raw, `[` `]` exposure.

Every setting is a console variable (`src/cvars.rs`): `+set r.lighting standard`,
`VF_SET="r.fog.density=2;r.clouds.coverage=0.7"`, or `settings.cfg` next to the game.
`--shot` renders headless (no window) and writes a PNG; `VF_CAM="x,y,z,yaw,pitch"` places
its camera.

## What changed from the C++ engine

- **Optimised SPIR-V**: shaders compile with `-O` (the C++ build shipped `-O0`), cached on
  disk by a hash of their full source (`shader_cache/`), plus a persisted Vulkan pipeline
  cache: only the first run (or an edited shader) pays the compile.
- **Shaders embedded in the binary** as a fallback: a release build runs without `shaders/`
  (a `shaders/` directory on disk still wins, for hot reload).
- **Sub-allocated GPU memory** (gpu-allocator) instead of one `vkAllocateMemory` per image /
  buffer.
- **Lock-free settings**: cvars are statics with atomic values; the renderer reads them every
  frame without string lookups or locks.
- **rayon** work-stealing replaces the hand-rolled job pool (chunk generation, the atlas
  packing, the light scan).
- std140 / push-constant layouts are mirrored as `#[repr(C)]` structs with compile-time
  offset checks.

## Status of the port

Ported: the voxel storage (sparse bricks, ring/streaming mode), world generation 1 (terrain,
space-colonisation trees, pines, bushes, logs, the tower, landmark spheres), the GPU world
mirror (upload, incremental edit sync, chunk streaming sync, bodies + BVH), the full renderer
(both lighting modes, water, grass wind, fog, clouds, lamps / torches, radiance cache, debug
views, locked view), post (exposure, bloom, tonemap), the window + headless shots, basic carve
/ place editing, the player controller (walking, steps, crouch, swimming, flying), rigid-body
physics (voxel-shell contacts, warm-started sequential impulses with friction, sleeping,
felled-tree hinges; fixed 60 Hz with interpolated rendering), destruction (carved-loose pieces
become bodies, a cut tree takes its own crown and topples toward the cut; bodies split), water
flow (a volume-conserving cellular fluid: water falls, spreads, refills from its body's surface
and drains lakes level).

Not yet ported: generators 2 / 3, endless streaming (`StreamManager`, far horizon), saves /
region files, the undo stack and edit tools (shapes, lines, brush tools), the ImGui UI (menus,
editor, map, inventory), audio, weather presets, mesh import, NVIDIA DLSS / NRD and hardware
ray queries.

## Debugging

`--validation` enables the Khronos validation layer. `VF_SKIP=gi,fog` skips passes by label,
`VF_MAXPASS=N` records only the first N dispatches, `VF_ROBUST=1` enables robust buffer / image
access, `VF_SIMTEST=steps` settles the bodies before a `--shot` (after `VF_CARVETEST=radius`,
aimed by `VF_CARVE_CAM` or placed by `VF_CARVE_AT=x,y,z`; `VF_FLUIDTEST=ticks` lets water flow
into the hole), `VF_LOG=debug|trace` raises the log level.

Known issue: Mesa's software driver (llvmpipe / lavapipe) crashes in the GI and froxel passes
with emissive-light sampling on (a driver codegen bug reading a uniform block in divergent
code); use `+set r.lights 0` there. Hardware drivers are unaffected.
