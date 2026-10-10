// Shared scene definitions + traversal for all compute passes.
// NOTE: no #version here — vkCompileComputeParts prepends it (+ the macro preamble
// kVkBridge in vk_renderer.cpp) and concatenates this file in front of each pass,
// with #line markers so compile errors name the real file:line.
//
// Descriptor layout: SET(n) expands to "set=n,". Each resource class has its own
// set (0 = scene textures, 1 = scene SSBOs, 2 = scene UBO, 3 = per-pass images +
// the shared PassUBO at binding 15). BEGIN/END_*_UNIFORMS wrap the scalar uniforms
// into those std140 blocks; UNI is empty. (Leftover from the OpenGL original,
// kept because it keeps every pass declaration short.)
layout(SET(0) binding = 0) uniform usampler3D gridTex;  // R32UI, world brick grid
layout(SET(0) binding = 1) uniform usampler3D atlasTex; // R8UI, packed 8^3 bricks
// Coarse empty-space pyramid over the WORLD brick grid (see GpuWorld). A cell is
// non-zero iff any child brick is non-empty, so the outer DDA can skip an empty
// coarse cell in one big step. C1/C2 MUST match GpuWorld's constants.
layout(SET(0) binding = 8) uniform usampler3D coarse1Tex; // R8UI, 4^3-brick cells
layout(SET(0) binding = 9) uniform usampler3D coarse2Tex; // R8UI, 16^3-brick cells
const int C1 = 4;
const int C2 = 16;
const bool kCoarseSkip = true; // debug toggle for the empty-space skip
// Brick Chebyshev distance field (VkGpuWorld, shaders/brickdist.comp): distance in
// bricks from each brick to the nearest non-empty one (0 = occupied). An empty
// brick with distance d sits in an all-empty cube of radius d-1 -> one jump.
layout(SET(0) binding = 10) uniform usampler3D brickDistTex;
#ifdef WATER_CLEAR
// The same field with water counted as empty (brickdist.comp "dry" passes): rays that
// pass through water (skipMat == 5) skip water-only bricks and jump dry-empty space.
layout(SET(0) binding = 20) uniform usampler3D brickDistDryTex;
#endif
// Standard lighting: sun shadow map, 3 cascades stacked vertically (shN x 3*shN,
// R32F = light-space height of the topmost occluder, toroidally addressed; see
// sunshadow.comp). Probe atlases are declared in probes.glsl.
layout(SET(0) binding = 14) uniform sampler2D shadowMapTex;
// Standard lighting: the lamp shadow cache (lamp_cube.comp records, lightVis reads): per lamp
// slot a row of 6 cube faces, R16F distance from the lamp to the first occluder.
layout(SET(0) binding = 22) uniform sampler2D lampCubeTex;
// Hybrid sun shadows: per shadow-map texel whose top is foliage, how far below the top the
// sunlight through the leaves drops under sunVisibility's cut-off (sunshadow.comp).
layout(SET(0) binding = 23) uniform sampler2D shadowAuxTex;
// Far horizon clipmap (horizon.h): 3 levels of 1024^2 packed texels stacked vertically.
layout(SET(0) binding = 24) uniform usampler2D horizonTex;   // (ground, crowns) per texel
layout(SET(0) binding = 26) uniform usampler2D horBlkTex;    // tallest top + 1 of each 8 x 8 texels (0: not made)
layout(SET(0) binding = 27) uniform sampler2D cloudPanoTex;  // the clouds seen from the eye, upper hemisphere (clouds_pano.comp)
layout(SET(0) binding = 25) uniform usampler2D skyTopTex;   // R16UI per column: top of the sky-blocking voxels (grasstop.comp)
const int kLampCubeRes = 128;   // = LitRenderer::kLampCubeRes
// Cube face f (0 +x, 1 -x, 2 +y, 3 -y, 4 +z, 5 -z) at face coords st in [-1, 1] -> direction
// (unnormalised), and back.
vec3 lampCubeDir(int f, vec2 st) {
    return f == 0 ? vec3(1.0, st.y, st.x) : f == 1 ? vec3(-1.0, st.y, st.x)
         : f == 2 ? vec3(st.x, 1.0, st.y) : f == 3 ? vec3(st.x, -1.0, st.y)
         : f == 4 ? vec3(st.x, st.y, 1.0) : vec3(st.x, st.y, -1.0);
}
int lampCubeFace(vec3 v, out vec2 st) {
    vec3 a = abs(v);
    if (a.x >= a.y && a.x >= a.z) { st = vec2(v.z, v.y) / a.x; return v.x > 0.0 ? 0 : 1; }
    if (a.y >= a.z) { st = vec2(v.x, v.z) / a.y; return v.y > 0.0 ? 2 : 3; }
    st = vec2(v.x, v.y) / a.z; return v.z > 0.0 ? 4 : 5;
}

// world.h Material: albedo (a metal's F0) + emissive; density, jitter, kind, variant / animation; the surface:
// roughness, metalness, dielectric F0, detail pattern.
struct Material { vec4 albedoEmissive; vec4 props; vec4 surf; };
layout(std430, SET(1) binding = 1) readonly buffer Palette { Material materials[256]; };

struct ObjHeader {
    mat4 worldToObj;
    mat4 objToWorld;
    mat4 prevObjToWorld;   // last frame's pose -> correct motion vectors for moving bodies
    ivec4 dimsBricks; // bricksX/Y/Z, grid offset
    vec4 aabbMin;
    vec4 aabbMax;
};
layout(std430, SET(1) binding = 2) readonly buffer Objects { ObjHeader objs[]; };
layout(std430, SET(1) binding = 3) readonly buffer ObjGrid { uint objGrid[]; };
// Occupancy of each detailed brick (by atlas slot): 8 octants (4^3 voxels) x 64 bits.
// octant = ox | oy<<1 | oz<<2, bit = (x&3) | (y&3)<<2 | (z&3)<<4 (VkGpuWorld::uploadMasks).
layout(std430, SET(1) binding = 4) readonly buffer BrickMasks { uvec2 brickMasks[]; };
// Object BVH (VkGpuWorld::writeBvh): node = {min.xyz, first}, {max.xyz, count};
// count 0 = interior (children at first, first+1), else a leaf of `count` prims.
struct BvhNode { vec4 a; vec4 b; };
layout(std430, SET(1) binding = 5) readonly buffer ObjBvh { BvhNode bvhNodes[]; };
layout(std430, SET(1) binding = 6) readonly buffer ObjBvhPrims { uint bvhPrims[]; };

BEGIN_SCENE_UNIFORMS
UNI uvec3 atlasBricks; // atlas dims in bricks
UNI float night;       // 0 day .. 1 night: sunDir/sunRGB are then the MOON (applyLook)
UNI ivec3 worldBricks; // world brick-grid dims
UNI int numObjects;
UNI vec3 sunDir;       // direction TO the sun
UNI float grassDist;   // grass blade draw distance (voxels), r.grass.distance
// Look (Options > Lighting & Atmosphere, LitRenderer::applyLook).
UNI vec3 sunRGB;      UNI float sunAngle;    // sun radiance; disk angular radius
UNI vec3 fogSunRGB;   UNI float fogDensity;  // god-ray in-scatter colour; extinction/voxel
UNI vec3 fogAmbient;  UNI float fogG;        // haze in-scatter; HG anisotropy
UNI vec3 skyZenith;   UNI float skyScale;    // physical sky colours; sky brightness
UNI vec3 skyHorizon;  UNI float giScale;     // bounce light strength
UNI vec3 sunsetRGB;   UNI int skyModel;      // glow around a low sun; 0 classic, 1 physical
UNI float fogHeight;  UNI float fogFalloff;  UNI float fogNoise; UNI int fogSteps;
UNI float fogShadowMax; UNI int tonemapOp; UNI float saturation; UNI float contrast;
UNI vec3 whiteBalance;  // r.warmth (display mapping)
UNI int grassShadows;   // r.grass.shadows: sun rays also test the grass blades
// Standard lighting: sun shadow cascades (shadowmap.glsl) + probe grid (probes.glsl).
UNI vec3 shRight; UNI float shTexel0;
UNI vec3 shUp;    UNI float shTexel1;
UNI float shLzTop; UNI float shTexel2; UNI int shN; UNI int stdLighting;
UNI ivec4 shWinA; UNI ivec4 shWinB;
UNI ivec3 probeOrigin; UNI float probeSpacing;
UNI ivec3 probeDims;   UNI int probeUpdateOffset;
UNI float probeHyst; UNI float probeNormalBias; UNI float stdSoftTaps; UNI float stdCascadeBlend;
UNI int stdCascadeBias; UNI float stdViewBias; UNI float shTexel3; UNI float stdSoftMin;
UNI ivec3 probeOriginC; UNI float probeSpacingC;   // coarse probe grid (probes.glsl level 1)
// Streaming worlds (ring mode): the world textures are a toroidal window; worldWin.xz = the
// window's min corner in absolute bricks (one full period, centred on the player: every
// resident chunk appears once), .w = 1. Bounded worlds: 0 (no wrap, origin 0).
UNI ivec4 worldWin;
// The far horizon (endless worlds): level window origins in texels (L0 xz, L1 xz | L2 xz),
// horB.z = on (0 = off; else the tallest horizon height + 2: nothing to hit above it), horB.w = the radius
// (voxels) inside which the voxels are trusted.
UNI ivec4 horA; UNI ivec4 horB;
// Per ring slot (chunk): how much of its fade-in over the horizon is to go, a byte each (0 = done).
UNI uvec4 chunkFade[64];
// Radiance cache (GI_CACHE, r.gi.cache): the camera (cell sizes grow with distance) + the LOD scale;
// x frame, y min samples for a cell to be used, z log2 of the smallest cell (voxels), w queue budget.
UNI vec3 giCacheCam; UNI float giCacheLod;
UNI ivec4 giCacheP;
UNI vec4 giCacheQ;   // x max samples a cell holds (r.gi.cache.frames), y frames unused before it is dropped
UNI ivec4 horC;      // the far horizon's level 3 xz, level 4 xz window origins (texels)
UNI vec4 cloudA;     // clouds: coverage, base (y), thickness, extinction / voxel
UNI vec4 cloudB;     // ... the pattern's offset (engine xz -> drifting absolute), evolution, on
UNI vec4 rainA;      // rain: intensity, time (s), ground wetness, -
END_SCENE_UNIFORMS

// World texture coordinates of absolute brick bc (wraps in a streaming world).
ivec3 gridWrap(ivec3 bc) {
    return worldWin.w != 0 ? ivec3(bc.x & (worldBricks.x - 1), bc.y, bc.z & (worldBricks.z - 1)) : bc;
}
ivec3 worldMinB() { return ivec3(worldWin.x, 0, worldWin.z); }

// --- Far horizon (endless worlds) -----------------------------------------------------------
// Beyond the loaded chunks the land goes on as a heightfield of the generator's terrain (the
// CPU's HorizonGen: 4 / 8 / 16 / 32 / 64 voxels per texel - blocks doubling with distance). A texel is two words:
// the ground = height * 8 (bits 0-13), water (14), canopy (15: levels 3-4's smooth forest), RGB565 (16-31) - 0 = not
// made yet (the next level is used); and the crowns over it (levels 0-2) = top (bits 0-10), thickness (11-18),
// RGB 4-5-4 (19-31) - a span floating over the ground with air under it - 0 = none.
#define HOR_N 1024
#define HOR_LEVELS 5
float horTexelSize(int L) { return float(4 << L); }
ivec2 horWinOf(int L) { return L == 0 ? horA.xy : L == 1 ? horA.zw : L == 2 ? horB.xy : L == 3 ? horC.xy : horC.zw; }
uvec2 horTexel2(int L, ivec2 tc) {
    ivec2 o = horWinOf(L);
    if (any(lessThan(tc, o)) || any(greaterThanEqual(tc, o + HOR_N))) return uvec2(0u);
    return texelFetch(horizonTex, ivec2(tc.x & (HOR_N - 1), (tc.y & (HOR_N - 1)) + L * HOR_N), 0).rg;
}
uint horTexel(int L, ivec2 tc) { return horTexel2(L, tc).x; }
// The tallest top in level L's 8 x 8 block bc (+ 1; 0 = not made / outside the window).
uint horBlkMax(int L, ivec2 bc) {
    ivec2 o = horWinOf(L) >> 3;
    if (any(lessThan(bc, o)) || any(greaterThanEqual(bc, o + HOR_N / 8))) return 0u;
    return texelFetch(horBlkTex, ivec2(bc.x & (HOR_N / 8 - 1), (bc.y & (HOR_N / 8 - 1)) + L * (HOR_N / 8)), 0).r;
}
float horCrownTop(uint c) { return float(c & 0x7FFu); }
float horCrownBottom(uint c) { return float(c & 0x7FFu) - float((c >> 11) & 255u); }
vec3 horCrownColour(uint c) { return vec3(float((c >> 28) & 15u) / 15.0, float((c >> 23) & 31u) / 31.0, float((c >> 19) & 15u) / 15.0); }
// Where the camera is (the finest level's window is centred on it, to a sixteenth).
vec2 horCentre() { return (vec2(horA.xy) + 0.5 * float(HOR_N)) * horTexelSize(0); }
float horHeightOf(uint p) { return float(p & 0x3FFFu) * 0.125; }
// Water texels (bit 14) hold the bed's depth (voxels, bits 16-23) and palette index (24-31) instead of a colour.
float horWaterDepthOf(uint p) { return float((p >> 16) & 255u); }
uint horWaterBedOf(uint p) { return p >> 24; }
vec3 horColourOf(uint p) {
    if ((p & (1u << 14)) != 0u) {   // (land blending into a shore: the bed, darker with depth)
        float k = 1.0 - exp(-horWaterDepthOf(p) * 0.12);
        return mix(materials[horWaterBedOf(p)].albedoEmissive.rgb, materials[5].albedoEmissive.rgb * vec3(0.5, 0.5, 0.6), k);
    }
    return vec3(float((p >> 27) & 31u) / 31.0, float((p >> 21) & 63u) / 63.0, float((p >> 16) & 31u) / 31.0);
}
// Level L at xz, bilinear (height; colour / flags from the texels): false if a texel is missing.
// (h: the top - the crowns' where they stand over the ground: sun shadows and the open sky see the trees)
bool horSampleL(int L, vec2 xz, out float h, out uvec4 q, out vec2 f) {
    vec2 u = xz / horTexelSize(L) - 0.5;
    ivec2 i = ivec2(floor(u));
    f = u - vec2(i);
    uvec2 a = horTexel2(L, i), b = horTexel2(L, i + ivec2(1, 0)), c = horTexel2(L, i + ivec2(0, 1)), d = horTexel2(L, i + ivec2(1, 1));
    q = uvec4(a.x, b.x, c.x, d.x);
    if (q.x == 0u || q.y == 0u || q.z == 0u || q.w == 0u) { h = 0.0; return false; }
    vec4 t = max(vec4(horHeightOf(q.x), horHeightOf(q.y), horHeightOf(q.z), horHeightOf(q.w)),
                 vec4(horCrownTop(a.y), horCrownTop(b.y), horCrownTop(c.y), horCrownTop(d.y)));
    h = mix(mix(t.x, t.y, f.x), mix(t.z, t.w, f.x), f.y);
    return true;
}
// The finest level that covers a point `dist` voxels (horizontally) from the camera.
bool horSample(vec2 xz, float dist, out float h, out int L) {
    uvec4 q; vec2 f;
    for (L = 0; L < HOR_LEVELS - 1 && dist >= 0.42 * HOR_N * horTexelSize(L); ++L) {}
    for (; L < HOR_LEVELS; ++L)
        if (horSampleL(L, xz, h, q, f)) return true;
    h = 0.0;
    return false;
}
// The water of a bilinear sample's texels: true when the nearest is water; its bed's palette index, the depth bilinear
// over the water texels around (land counts 0: the shore shelves up).
bool horWaterOf(uvec4 q, vec2 f, out float depth, out uint bed) {
    uint nq = f.x < 0.5 ? (f.y < 0.5 ? q.x : q.z) : (f.y < 0.5 ? q.y : q.w);
    bvec4 wq = notEqual(q & uvec4(1u << 14), uvec4(0u));
    vec4 wd = mix(vec4(0.0), vec4((q >> 16) & uvec4(255u)), wq);
    depth = mix(mix(wd.x, wd.y, f.x), mix(wd.z, wd.w, f.x), f.y);
    bed = horWaterBedOf(nq);
    return (nq & (1u << 14)) != 0u;
}
// The generator's water at xz from the finest level that has it (false: land there, or nothing made yet).
bool horWaterAt(vec2 xz, out float depth, out uint bed) {
    depth = 0.0; bed = 0u;
    for (int L = 0; L < HOR_LEVELS; ++L) {
        float h; uvec4 q; vec2 f;
        if (horSampleL(L, xz, h, q, f)) return horWaterOf(q, f, depth, bed);
    }
    return false;
}
// How far the chunk at xz has faded in over the horizon (0..1; 1 = done / not endless).
float chunkFadeAt(vec2 xz) {
    int rw = worldBricks.x / 16;   // chunks across the ring
    ivec2 c = ivec2(floor(xz / 128.0)) & (rw - 1);
    int s = c.y * rw + c.x;
    uint w = chunkFade[s >> 4][(s >> 2) & 3];
    return 1.0 - float((w >> ((s & 3) * 8)) & 255u) / 255.0;
}
// Is the column at xz a loaded chunk (the voxels draw it)? Loaded chunks always have ground in
// their bottom brick; the window holds each resident chunk exactly once. A chunk still fading in
// isn't yet (the pixels dithered out of it show the horizon).
// Is p under the open sky? 1 above its column's highest sky-blocking voxel (leaves and water
// don't block), 0 below it (a cave, under a roof), a short blend between. Beyond the loaded
// chunks the generator's terrain (the far horizon) decides; below the world, 0. Used where a
// ray that found nothing within its reach would otherwise count as seeing the sky.
bool colLoaded(vec2 xz) {   // a column of the loaded world (resident chunks have ground in their bottom brick)
    ivec3 bc = ivec3(int(floor(xz.x / 8.0)), 0, int(floor(xz.y / 8.0)));
    ivec3 mn = worldMinB();
    return bc.x >= mn.x && bc.z >= mn.z && bc.x < mn.x + worldBricks.x && bc.z < mn.z + worldBricks.z &&
           texelFetch(gridTex, gridWrap(bc), 0).r != 0u;
}
float skyOpen(vec3 p) {
#ifndef SKY_OCC
    return 1.0;   // (r.gi.skyocc 0)
#endif
    if (p.y < 0.0) return 0.0;
    float top;
    if (colLoaded(p.xz)) {
        ivec2 sz = textureSize(skyTopTex, 0);
        ivec2 c = ivec2(floor(p.xz));
        top = float(texelFetch(skyTopTex, ((c % sz) + sz) % sz, 0).r);
    } else {
        float h;
        int L;
        if (!horSample(p.xz, length(p.xz - horCentre()), h, L)) return 1.0;   // (no horizon: a bounded world's outside)
        top = h;
    }
    return smoothstep(top - 0.5, top + 1.0, p.y);
}
// skyOpen over an area (p and 4 points 12 voxels around it): 0 only where all are covered - under
// rock or a roof - not beside one tree top or under a branch.
float skyOpenWide(vec3 p) {
    return 0.2 * (skyOpen(p) + skyOpen(p + vec3(12.0, 0.0, 0.0)) + skyOpen(p - vec3(12.0, 0.0, 0.0)) +
                  skyOpen(p + vec3(0.0, 0.0, 12.0)) + skyOpen(p - vec3(0.0, 0.0, 12.0)));
}
// A ray that flew its whole reach through the loaded world without a hit is escaping (sky), unless it
// ended below the world or out in the far horizon's terrain (underground there).
float escapedSky(vec3 f) {
    return f.y < 0.0 ? 0.0 : colLoaded(f.xz) ? 1.0 : skyOpen(f);
}
bool horLoaded(vec2 xz) {
    ivec3 bc = ivec3(int(floor(xz.x / 8.0)), 0, int(floor(xz.y / 8.0)));
    ivec3 mn = worldMinB();
    if (bc.x < mn.x || bc.z < mn.z || bc.x >= mn.x + worldBricks.x || bc.z >= mn.z + worldBricks.z) return false;
    return texelFetch(gridTex, gridWrap(bc), 0).r != 0u && chunkFadeAt(xz) >= 1.0;
}
struct HorHit { bool hit; float t; vec3 n; vec3 albedo; bool water; bool canopy; int steps; float depth; uint bed; float top; };   // (water: the bed)
// March the heightfield along a ray that found no voxels: from where it leaves the trusted radius
// out to the coarsest level's reach. Relaxed steps (half the height above the ground, at least
// most of a texel, growing with distance - the fog hides what that misses), then bisection.
// The texels are BLOCKS (Hugo: "not even voxel-like"): each a box with a flat top at its height and vertical sides, of the
// level's size - 0.4 m near the loaded land, doubling out to 6.4 m. The march finds where the ray goes under the
// highest of the texels around it (never skipping a block), then walks the level's grid exactly across that step and
// stops at the first box it enters below its top (a side) or whose top it crosses.
// A level's texel, or where it isn't made yet the coarser level's over it.
uvec2 horTexelAny(int L, ivec2 c) {
    for (int l = L; l < HOR_LEVELS; ++l) {
        uvec2 q = horTexel2(l, c >> (l - L));
        if (q.x != 0u) return q;
    }
    return uvec2(0u);
}
// The exact walk over level L's texels between t0 and t1: each is a box of ground (to its top) and maybe a crown span
// over it; the nearest face the ray meets (outside the loaded chunks) fills r.
bool horTexels(vec3 ro, vec3 rd, float t0, float t1, int L, inout HorHit r) {
    float e = horTexelSize(L);
    vec2 o = ro.xz / e, d = rd.xz / e;   // (grid units: cell i = [i, i + 1))
    ivec2 c = ivec2(floor(o + d * t0));
    ivec2 st = ivec2(sign(d));
    vec2 inv = vec2(abs(d.x) > 1e-9 ? 1.0 / d.x : 1e30, abs(d.y) > 1e-9 ? 1.0 / d.y : 1e30);
    vec2 tNext = vec2(st.x > 0 ? (float(c.x + 1) - o.x) * inv.x : st.x < 0 ? (float(c.x) - o.x) * inv.x : 1e30,
                      st.y > 0 ? (float(c.y + 1) - o.y) * inv.y : st.y < 0 ? (float(c.y) - o.y) * inv.y : 1e30);
    vec2 tDelta = abs(inv);
    float tIn = t0;
    vec3 nIn = vec3(0.0, 1.0, 0.0);
    for (int k = 0; k < 28 && tIn <= t1; ++k) {
        float tOut = min(tNext.x, tNext.y);
        uvec2 q = horTexelAny(L, c);
        if (q.x != 0u) {
            float best = 1e30;
            vec3 nb = vec3(0.0, 1.0, 0.0);
            bool crown = false;
            float yIn = ro.y + rd.y * tIn;
            float hc = horHeightOf(q.x);   // (HorizonGen stores the top face: ground h + 1)
            if (yIn < hc) { best = tIn; nb = nIn; }   // a side (or under, at the start)
            else if (rd.y < 0.0) {
                float tTop = (hc - ro.y) / rd.y;
                if (tTop <= tOut) { best = tTop; nb = vec3(0.0, 1.0, 0.0); }
            }
            if (q.y != 0u) {   // the crowns: a box from their underside to their top
                float cb = horCrownBottom(q.y), ct = horCrownTop(q.y), lo, hi;
                if (abs(rd.y) > 1e-6) {
                    float ta = (cb - ro.y) / rd.y, tb = (ct - ro.y) / rd.y;
                    lo = max(tIn, min(ta, tb)); hi = min(tOut, max(ta, tb));
                } else { lo = tIn; hi = yIn >= cb && yIn <= ct ? tOut : -1.0; }
                if (lo <= hi && lo < best) {
                    best = lo; crown = true;
                    nb = lo <= tIn + 1e-4 ? nIn : rd.y < 0.0 ? vec3(0.0, 1.0, 0.0) : vec3(0.0, -1.0, 0.0);
                }
            }
            if (best < 1e29 && !horLoaded((ro + rd * best).xz)) {
                r.hit = true; r.t = best; r.n = nb;
                if (crown) {
                    r.albedo = horCrownColour(q.y); r.canopy = true; r.top = horCrownTop(q.y); r.water = false;
                } else {
                    r.albedo = horColourOf(q.x);
                    r.water = (q.x & (1u << 14)) != 0u;
                    r.canopy = (q.x & (1u << 15)) != 0u;
                    r.top = hc;
                    if (r.water) { r.depth = horWaterDepthOf(q.x); r.bed = horWaterBedOf(q.x); r.n = vec3(0.0, 1.0, 0.0); }
                }
                return true;
            }
        }
        if (tNext.x < tNext.y) { c.x += st.x; tIn = tNext.x; tNext.x += tDelta.x; nIn = vec3(-float(st.x), 0.0, 0.0); }
        else { c.y += st.y; tIn = tNext.y; tNext.y += tDelta.y; nIn = vec3(0.0, 0.0, -float(st.y)); }
    }
    return false;
}
// The primary ray over the far horizon: level by level outward (each where its texels are about a pixel or two), a 2D
// walk over its 8 x 8-texel blocks - one the ray passes wholly over (above its tallest top) is skipped, the others
// walked texel by texel. Conservative (Tevs et al., maximum-mipmap height fields): it can't step over a thin crown.
// (The old march stepped by the clearance, ~1.5 % of the distance: it skipped the trunks of the far spruces and drew
// their tips as slabs floating in the sky - Hugo's "glitchy" LODs.)
HorHit traceHorizon(vec3 ro, vec3 rd) {
    HorHit r;
    r.hit = false; r.steps = 0; r.t = 0.0; r.n = vec3(0.0, 1.0, 0.0); r.albedo = vec3(0.0); r.water = false; r.canopy = false;
    r.depth = 0.0; r.bed = 0u; r.top = 0.0;
    if (horB.z == 0) return r;
    float hl = length(rd.xz);
    if (hl < 1e-3) return r;
    float t = float(horB.w) / hl;
    float tMax = 0.42 * HOR_N * 64.0 / hl;
    float yTop = float(horB.z);
    int L = 0;
    for (; L < HOR_LEVELS - 1 && t * hl >= 0.42 * HOR_N * horTexelSize(L); ++L) {}
    for (; L < HOR_LEVELS && t < tMax; ++L) {
        float tEnd = L < HOR_LEVELS - 1 ? min(0.42 * HOR_N * horTexelSize(L) / hl, tMax) : tMax;
        if (t >= tEnd) continue;
        float B = 8.0 * horTexelSize(L);
        vec2 o = ro.xz / B, d = rd.xz / B;
        ivec2 c = ivec2(floor(o + d * t));
        ivec2 st = ivec2(sign(d));
        vec2 inv = vec2(abs(d.x) > 1e-9 ? 1.0 / d.x : 1e30, abs(d.y) > 1e-9 ? 1.0 / d.y : 1e30);
        vec2 tNext = vec2(st.x > 0 ? (float(c.x + 1) - o.x) * inv.x : st.x < 0 ? (float(c.x) - o.x) * inv.x : 1e30,
                          st.y > 0 ? (float(c.y + 1) - o.y) * inv.y : st.y < 0 ? (float(c.y) - o.y) * inv.y : 1e30);
        vec2 tDelta = abs(inv);
        float tIn = t;
        for (int k = 0; k < 80 && tIn < tEnd; ++k) {
            float tOut = min(min(tNext.x, tNext.y), tEnd);
            float yIn = ro.y + rd.y * tIn, yOut = ro.y + rd.y * tOut;
            if (rd.y >= 0.0 && yIn > yTop) return r;   // above every hill and crown, going up
            r.steps++;
            uint bm = horBlkMax(L, c);
            if (bm == 0u || min(yIn, yOut) < float(bm)) {
                if (horTexels(ro, rd, tIn, tOut, L, r)) return r;
            }
            if (tNext.x < tNext.y) { c.x += st.x; tIn = tNext.x; tNext.x += tDelta.x; }
            else { c.y += st.y; tIn = tNext.y; tNext.y += tDelta.y; }
        }
        t = tEnd;
    }
    return r;
}

// Sun visibility of a horizon surface (a point beyond the loaded chunks): march the heightfield
// toward the sun - the crowns and hills of the horizon shade each other and the ground. Starts a
// few texels out (no self-shadow from the bilinear surface), steps of half the clearance / at
// least most of a texel, up to ~1.5 km or above every hill. Points in loaded chunks: 1 (the
// voxels' own shadow rays cover them).
float horizonSunVisibility(vec3 o, vec3 d) {
    if (horB.z == 0 || d.y <= 0.0 || horLoaded(o.xz)) return 1.0;
    float t = 0.0;
    {
        float h0; int L0;
        if (!horSample(o.xz, length(o.xz - horCentre()), h0, L0)) return 1.0;
        t = horTexelSize(L0) * 2.5;
    }
    for (int i = 0; i < 64 && t < 15000.0; ++i) {
        vec3 p = o + d * t;
        if (p.y > float(horB.z)) return 1.0;   // (above the tallest horizon height: nothing to block it)
        float h; int L;
        if (!horSample(p.xz, length(p.xz - horCentre()), h, L)) return 1.0;
        float dh = p.y - h;
        if (dh < 0.0) return horLoaded(p.xz) ? 1.0 : 0.0;
        t += max(dh * 0.5, horTexelSize(L) * 0.8);
    }
    return 1.0;
}

// r.water 1 (WATER_CLEAR): water is TRANSPARENT to every ray except the primary one
// that finds its surface (gbuffer.comp traces that one with skipMat 0, then the
// refracted ray to the bed with 5). Shadow / GI / probe / fog rays pass through water;
// the shading passes attenuate what reaches an underwater point. skipMat is a
// CONSTANT at every call site but that one, so the compiler folds it (a runtime
// global cost 8 registers in gbuffer / gi).
#ifdef WATER_CLEAR
const uint kSkipMat = 5u;
#define VOX_SOLID(m) ((m) != 0u && (m) != skipMat)
#else
const uint kSkipMat = 0u;
#define VOX_SOLID(m) ((m) != 0u)
#endif

// Debug heatmaps (r.debugview heat.*): count DDA iterations. Compiled in only
// when DEBUG_HEAT is defined (the renderer rebuilds pipelines for heat views),
// so normal rendering pays nothing.
#ifdef DEBUG_HEAT
int gSteps = 0;
#define HEAT_STEP() gSteps++
#else
#define HEAT_STEP()
#endif

// Exit-axis selection WITHOUT runtime vector indexing (v[ax] with a dynamic ax
// forces the vector into local memory on NVIDIA -> more registers + spills; measured
// via `shader stats`). Ties resolve x, then y, then z like the DDA step's own
// comparisons (tMax.x < tMax.y && tMax.x < tMax.z ? x : tMax.y < tMax.z ? y : z).
bvec3 exitAxis(vec3 tFar) {
    bool x = tFar.x < tFar.y && tFar.x < tFar.z;
    bool y = !x && tFar.y < tFar.z;
    return bvec3(x, y, !x && !y);
}
float pickAxis(vec3 v, bvec3 a) { return a.x ? v.x : (a.y ? v.y : v.z); }

struct Hit {
    bool hit;
    float t;
    vec3 normal;
    uint mat;
    ivec3 cell; // hit voxel in the volume's LOCAL coords (stable per voxel)
    int obj;    // -1 = static world, else object index (debug views)
};

// The atlas is 64 x 64 bricks per layer (VkGpuWorld::ABX / ABY, static_assert there): shifts / masks, not
// integer division by a uniform (called per detailed brick on every ray and per point query).
ivec3 slotBase(uint slot) {
    return ivec3(slot & 63u, (slot >> 6) & 63u, slot >> 12) * 8;
}

// Two-level DDA through one sparse brick volume. `useTex` selects the world
// grid texture; otherwise grid entries come from objGrid at `gridOff`.
// The voxels of detailed brick bc (grid value g >= 256) along the ray from t, where it entered the
// brick (`normal` = the face it crossed): the first solid one goes into h (true). Shared by the DDA
// below and the ray-query path (traceWorldRQ), so both see exactly the same voxels.
bool brickVoxels(vec3 ro, vec3 safeRd, vec3 inv, ivec3 stepDir, ivec3 bc, uint g, float t, vec3 normal,
                 uint skipMat, inout Hit h) {
    ivec3 base = slotBase(g - 256u);
    ivec3 bmin = bc * 8;
    vec3 q = ro + safeRd * (t + 1e-3);
    ivec3 vc = clamp(ivec3(floor(q)), bmin, bmin + 7);
    vec3 tMax1 = ((vec3(vc) + step(vec3(0.0), safeRd)) - ro) * inv;
    vec3 tDelta1 = abs(inv);
    float tv = t;
    vec3 n2 = normal;
#ifdef USE_BRICK_MASKS
    uint slotM = (g - 256u) * 8u;
    int curOct = -1;       // octant whose mask is in `om` (reload on change only)
    uvec2 om = uvec2(0u);
#endif
    for (int j = 0; j < 26; j++) {
        HEAT_STEP();
#ifdef USE_BRICK_MASKS
        // Latency-friendly: the voxel's atlas fetch and (on octant change) the
        // octant mask are issued together; a hit never waits on the mask. The
        // mask only decides, after a miss, whether this 4^3 octant is EMPTY
        // and can be left in one jump (landing clamped to the exit face, exit
        // axis chosen with the DDA's own tie-break -> same voxels visited).
        ivec3 lv = vc - bmin;
        uint m = texelFetch(atlasTex, base + lv, 0).r;
        int oct = (lv.x >> 2) | ((lv.y >> 2) << 1) | ((lv.z >> 2) << 2);
        if (oct != curOct) { om = brickMasks[slotM + uint(oct)]; curOct = oct; }
        if (m == 0u && (om.x | om.y) == 0u) {
            ivec3 oLo = bmin + (lv & ~3);
            vec3 tFar = (mix(vec3(oLo), vec3(oLo + 4), step(vec3(0.0), safeRd)) - ro) * inv;
            bvec3 ax = exitAxis(tFar);
            float tE = pickAxis(tFar, ax);
            ivec3 nv = clamp(ivec3(floor(ro + safeRd * tE)), oLo, oLo + 3);
            vc = mix(nv, mix(oLo - 1, oLo + 4, greaterThan(stepDir, ivec3(0))), ax);
            tv = tE;
            n2 = mix(vec3(0.0), vec3(-stepDir), ax);
            tMax1 = ((vec3(vc) + step(vec3(0.0), safeRd)) - ro) * inv;
            if (any(lessThan(vc, bmin)) || any(greaterThan(vc, bmin + 7))) break;
            continue;
        }
        if (VOX_SOLID(m)) {
#else
        uint m = texelFetch(atlasTex, base + (vc - bmin), 0).r;
        if (VOX_SOLID(m)) {
#endif
            h.hit = true;
            h.t = tv;
            h.normal = n2;
            h.mat = m;
            h.cell = vc;
            return true;
        }
        if (tMax1.x < tMax1.y && tMax1.x < tMax1.z) {
            vc.x += stepDir.x; tv = tMax1.x; tMax1.x += tDelta1.x;
            n2 = vec3(-stepDir.x, 0, 0);
        } else if (tMax1.y < tMax1.z) {
            vc.y += stepDir.y; tv = tMax1.y; tMax1.y += tDelta1.y;
            n2 = vec3(0, -stepDir.y, 0);
        } else {
            vc.z += stepDir.z; tv = tMax1.z; tMax1.z += tDelta1.z;
            n2 = vec3(0, 0, -stepDir.z);
        }
        if (any(lessThan(vc, bmin)) || any(greaterThan(vc, bmin + 7))) break;
    }
    return false;
}

Hit traceVolumeS(vec3 ro, vec3 rd, ivec3 dimsB, bool useTex, uint gridOff, float tCap, uint skipMat) {
    Hit h;
    h.hit = false;
    // The world's box: a streaming world's window (worldWin), else [0, dims); a body: [0, dims).
    ivec3 minB = useTex ? worldMinB() : ivec3(0);
    vec3 dims = vec3(dimsB * 8);

    vec3 safeRd = mix(rd, vec3(1e-5), lessThan(abs(rd), vec3(1e-5)));
    vec3 inv = 1.0 / safeRd;

    vec3 t0 = (vec3(minB * 8) - ro) * inv;
    vec3 t1 = (vec3(minB * 8) + dims - ro) * inv;
    vec3 tmin3 = min(t0, t1), tmax3 = max(t0, t1);
    float tEnter = max(max(tmin3.x, tmin3.y), tmin3.z);
    float tExit = min(min(tmax3.x, tmax3.y), tmax3.z);
    if (tExit < max(tEnter, 0.0) || tEnter > tCap) return h;

    float t = max(tEnter, 0.0);
    vec3 entry = ro + safeRd * (t + 1e-3);
    ivec3 bc = clamp(ivec3(floor(entry * 0.125)), minB, minB + dimsB - 1);
    ivec3 stepDir = ivec3(sign(safeRd));
    vec3 tDelta8 = abs(inv) * 8.0;
    vec3 tMax8 = ((vec3(bc) + step(vec3(0.0), safeRd)) * 8.0 - ro) * inv;

    vec3 normal = vec3(0.0);
    if (tEnter > 0.0) {
        if (tEnter == tmin3.x) normal = vec3(-stepDir.x, 0, 0);
        else if (tEnter == tmin3.y) normal = vec3(0, -stepDir.y, 0);
        else normal = vec3(0, 0, -stepDir.z);
    }

    int maxOuter = 3 * (dimsB.x + dimsB.y + dimsB.z) + 4;
    for (int i = 0; i < maxOuter; i++) {
        if (t > tCap) return h;
        HEAT_STEP();
        uint g = useTex ? texelFetch(gridTex, gridWrap(bc), 0).r
                        : objGrid[gridOff + uint((bc.z * dimsB.y + bc.y) * dimsB.x + bc.x)];
#ifdef WATER_CLEAR
        // A brick holding only water (+ air) is empty for a ray that ignores water:
        // without this, rays skimming a lake crawl voxel by voxel through its surface
        // bricks (measured: water view reflection pass ~4 ms).
        // Uniform bricks need no lookup (only uniform water is dry-empty).
        if (useTex && skipMat == 5u &&
            (g == 5u || (g >= 256u && texelFetch(brickDistDryTex, gridWrap(bc), 0).r > 0u))) g = 0u;
#endif
        if (g >= 256u) {
            if (brickVoxels(ro, safeRd, inv, stepDir, bc, g, t, normal, skipMat, h)) return h;
        } else if (VOX_SOLID(g)) {
            h.hit = true;
            h.t = t;
            h.normal = normal;
            h.mat = g;
            h.cell = ivec3(floor(ro + safeRd * (t + 1e-3)));
            return h;
        }

        // Advance one brick — but on the world path, if this brick is empty,
        // skip the LARGEST empty coarse cell containing it in a single step.
        // The cell is empty by construction, so no geometry can be skipped
        // (output stays identical to the plain brick-by-brick DDA).
        bool stepped = false;
        if (kCoarseSkip && useTex && g == 0u) {
            ivec3 cellB = ivec3(0);
            int Cf = 0;
#ifdef USE_DIST_FIELD
            int d = int(texelFetch(brickDistTex, gridWrap(bc), 0).r);
#ifdef WATER_CLEAR
            if (skipMat == 5u) d = int(texelFetch(brickDistDryTex, gridWrap(bc), 0).r);
#endif
            if (d > 1) {                 // empty cube centred on this brick
                Cf = 2 * d - 1;
                cellB = bc - (d - 1);
            } else
#endif
            {
                // (>> = floor division: absolute bricks can be negative in a streaming world)
                Cf = texelFetch(coarse2Tex, gridWrap(bc) >> 4, 0).r == 0u
                         ? C2
                         : (texelFetch(coarse1Tex, gridWrap(bc) >> 2, 0).r == 0u ? C1 : 0);
                if (Cf > 0) cellB = bc - (bc & (Cf - 1));   // aligned coarse cell (floor, negatives too)
            }
            if (Cf > 0) {
                vec3 cellMin = vec3(cellB) * 8.0;
                vec3 cellMax = cellMin + float(Cf) * 8.0;
                vec3 tFar = (mix(cellMin, cellMax, step(vec3(0.0), safeRd)) - ro) * inv;
                bvec3 ax = exitAxis(tFar);
                float tExit = pickAxis(tFar, ax);
                normal = mix(vec3(0.0), vec3(-stepDir), ax);
                t = tExit;
                // Land in the brick just past the cell. Derive the other axes from
                // the ray position, but FORCE the exit axis past the cell — a raw
                // floor() can round back into the empty cell (float error) and
                // stall the ray (re-skips the same t → missed geometry + wasted
                // iterations, which was the bug).
                // Clamp the non-exit axes into the cell's face (the ray leaves through
                // that face), so float error can't land it in a diagonal neighbour.
                bc = clamp(ivec3(floor((ro + safeRd * (t + 1e-3)) * 0.125)), cellB, cellB + Cf - 1);
                bc = mix(bc, mix(cellB - 1, cellB + Cf, greaterThan(stepDir, ivec3(0))), ax);
                tMax8 = ((vec3(bc) + step(vec3(0.0), safeRd)) * 8.0 - ro) * inv;
                stepped = true;
            }
        }
        if (!stepped) {
            if (tMax8.x < tMax8.y && tMax8.x < tMax8.z) {
                bc.x += stepDir.x; t = tMax8.x; tMax8.x += tDelta8.x;
                normal = vec3(-stepDir.x, 0, 0);
            } else if (tMax8.y < tMax8.z) {
                bc.y += stepDir.y; t = tMax8.y; tMax8.y += tDelta8.y;
                normal = vec3(0, -stepDir.y, 0);
            } else {
                bc.z += stepDir.z; t = tMax8.z; tMax8.z += tDelta8.z;
                normal = vec3(0, 0, -stepDir.z);
            }
        }
        if (any(lessThan(bc, minB)) || any(greaterThanEqual(bc, minB + dimsB))) return h;
    }
    return h;
}

Hit traceVolume(vec3 ro, vec3 rd, ivec3 dimsB, bool useTex, uint gridOff, float tCap) {
    return traceVolumeS(ro, rd, dimsB, useTex, gridOff, tCap, kSkipMat);
}

// Test one object: its world AABB, then its brick volume in object space.
void traceObject(int oi, vec3 ro, vec3 rd, vec3 inv, inout Hit best, inout float limit) {
    vec3 t0 = (objs[oi].aabbMin.xyz - ro) * inv;
    vec3 t1 = (objs[oi].aabbMax.xyz - ro) * inv;
    vec3 tmin3 = min(t0, t1), tmax3 = max(t0, t1);
    float tEnter = max(max(tmin3.x, tmin3.y), tmin3.z);
    float tExit = min(min(tmax3.x, tmax3.y), tmax3.z);
    if (tExit < max(tEnter, 0.0) || tEnter > limit) return;

    vec3 roO = (objs[oi].worldToObj * vec4(ro, 1.0)).xyz;
    vec3 rdO = mat3(objs[oi].worldToObj) * rd;
    Hit oh = traceVolume(roO, rdO, objs[oi].dimsBricks.xyz, false,
                         uint(objs[oi].dimsBricks.w), limit);
    if (oh.hit && oh.t < limit) {
        oh.normal = normalize(mat3(objs[oi].objToWorld) * oh.normal);
        oh.obj = oi;
        best = oh;
        limit = oh.t;
    }
}

// Objects only (dynamic bodies): nearest hit closer than `limit`, merged into
// `best` (world traversal is the caller's). Shared by traceScene and the
// Standard-lighting object shadow ray (the shadow map holds the world only).
// anyHit: return at the first body hit (visibility rays - only "blocked?" is read).
void traceObjectsA(vec3 ro, vec3 rd, inout Hit best, float limit, bool anyHit) {
    if (numObjects == 0) return;
    vec3 safeRd = mix(rd, vec3(1e-5), lessThan(abs(rd), vec3(1e-5)));
    vec3 inv = 1.0 / safeRd;
#ifdef USE_OBJ_BVH
    // BVH over object AABBs: O(log n) boxes per ray instead of every object, and
    // subtrees farther than the current hit are pruned. Near child first.
    int stack[24];
    int sp = 0;
    int node = 0;
    for (int it = 0; it < 512; it++) {
        BvhNode n = bvhNodes[node];
        vec3 t0 = (n.a.xyz - ro) * inv, t1 = (n.b.xyz - ro) * inv;
        vec3 tmin3 = min(t0, t1), tmax3 = max(t0, t1);
        float tEnter = max(max(tmin3.x, tmin3.y), tmin3.z);
        float tExit = min(min(tmax3.x, tmax3.y), tmax3.z);
        bool visit = tExit >= max(tEnter, 0.0) && tEnter <= limit;
        uint first = floatBitsToUint(n.a.w), count = floatBitsToUint(n.b.w);
        if (visit && count == 0u) {
            if (sp < 24) stack[sp++] = int(first) + 1;
            node = int(first);
            continue;
        }
        if (visit)
            for (uint k = 0u; k < count; k++) traceObject(int(bvhPrims[first + k]), ro, rd, inv, best, limit);
        if (anyHit && best.hit) return;
        if (sp == 0) break;
        node = stack[--sp];
    }
#else
    for (int oi = 0; oi < numObjects && !(anyHit && best.hit); oi++) traceObject(oi, ro, rd, inv, best, limit);
#endif
}
void traceObjectsInto(vec3 ro, vec3 rd, inout Hit best, float limit) { traceObjectsA(ro, rd, best, limit, false); }

// Point queries on bodies (contact AO, probes): up to two bodies whose AABB grown by
// `pad` contains P (BVH walk like traceObjectsInto), -1 = none.
bool inBox(vec3 P, vec3 lo, vec3 hi, float pad) {
    return all(greaterThanEqual(P, lo - pad)) && all(lessThanEqual(P, hi + pad));
}
void nearBodies(vec3 P, float pad, out int o0, out int o1) {
    o0 = -1; o1 = -1;
    if (numObjects == 0) return;
#ifdef USE_OBJ_BVH
    int stack[24];
    int sp = 0;
    int node = 0;
    for (int it = 0; it < 256; it++) {
        BvhNode n = bvhNodes[node];
        bool visit = inBox(P, n.a.xyz, n.b.xyz, pad);
        uint first = floatBitsToUint(n.a.w), count = floatBitsToUint(n.b.w);
        if (visit && count == 0u) {
            if (sp < 24) stack[sp++] = int(first) + 1;
            node = int(first);
            continue;
        }
        if (visit)
            for (uint k = 0u; k < count; k++) {
                int oi = int(bvhPrims[first + k]);
                if (!inBox(P, objs[oi].aabbMin.xyz, objs[oi].aabbMax.xyz, pad)) continue;
                if (o0 < 0) o0 = oi;
                else if (oi != o0) { o1 = oi; return; }
            }
        if (sp == 0) break;
        node = stack[--sp];
    }
#else
    for (int oi = 0; oi < numObjects; oi++) {
        if (!inBox(P, objs[oi].aabbMin.xyz, objs[oi].aabbMax.xyz, pad)) continue;
        if (o0 < 0) o0 = oi;
        else { o1 = oi; return; }
    }
#endif
}
// Is world point p inside a solid voxel of body oi?
bool objectSolidAt(int oi, vec3 p) {
    const uint skipMat = kSkipMat;   // VOX_SOLID reads it
    if (oi < 0) return false;
    ivec3 v = ivec3(floor((objs[oi].worldToObj * vec4(p, 1.0)).xyz));
    ivec3 dimsB = objs[oi].dimsBricks.xyz;
    if (any(lessThan(v, ivec3(0))) || any(greaterThanEqual(v, dimsB * 8))) return false;
    ivec3 bc = v >> 3;
    uint g = objGrid[uint(objs[oi].dimsBricks.w) + uint((bc.z * dimsB.y + bc.y) * dimsB.x + bc.x)];
    if (g == 0u) return false;
    if (g < 256u) return VOX_SOLID(g);
    return VOX_SOLID(texelFetch(atlasTex, slotBase(g - 256u) + (v & 7), 0).r);
}

// Indirect-light rays (GI bounce, Standard probes) see bodies too (r.gi.objects): a body
// blocks its own sky / bounce light and shades what is under it.
#ifdef GI_OBJECTS
#define GI_ADD_OBJECTS(o, d, g, cap) { g.obj = -1; traceObjectsInto(o, d, g, min(g.hit ? g.t : (cap), 16.0)); }
#define GI_SUN_BLOCKED(p) traceSceneGi(p, sunDir, 1e30, 0u).hit
#else
#define GI_ADD_OBJECTS(o, d, g, cap)
#define GI_SUN_BLOCKED(p) traceWorldGi(p, sunDir, 1e30, 0u).hit
#endif

Hit traceSceneS(vec3 ro, vec3 rd, float tCap, uint skipMat) {
    Hit best = traceVolumeS(ro, rd, worldBricks, true, 0u, tCap, skipMat);
    best.obj = -1;
    float limit = best.hit ? best.t : tCap;
    traceObjectsInto(ro, rd, best, limit);
    return best;
}
Hit traceScene(vec3 ro, vec3 rd, float tCap) { return traceSceneS(ro, rd, tCap, kSkipMat); }

// --- Hardware ray queries (vk_rt.h; RT_* defines, r.rt.*) -----------------------------------
// The RT cores find the bricks a ray meets (one AABB per brick in a per-chunk BLAS); each candidate
// is resolved exactly like the DDA's brick step (brickVoxels / the uniform-brick test), the nearest
// hit is committed. Bodies stay on their own DDA + BVH (traceObjectsInto).
#ifdef RT_ANY
layout(SET(1) binding = 0) uniform accelerationStructureEXT rtTlas;
layout(std430, SET(1) binding = 7) readonly buffer RtPrims { uint rtPrims[]; };
// Brick bc (absolute) against the ray, from t >= tLo: its first solid voxel into h. Like the DDA, a
// brick counts if the ray ENTERS it by tCap (its voxels may lie a little beyond).
bool rqBrick(vec3 ro, vec3 rd, ivec3 bc, float tLo, float tCap, uint skipMat, inout Hit h) {
    uint g = texelFetch(gridTex, gridWrap(bc), 0).r;
#ifdef WATER_CLEAR
    if (skipMat == 5u && (g == 5u || (g >= 256u && texelFetch(brickDistDryTex, gridWrap(bc), 0).r > 0u))) return false;
#endif
    if (g == 0u) return false;
    vec3 safeRd = mix(rd, vec3(1e-5), lessThan(abs(rd), vec3(1e-5)));
    vec3 inv = 1.0 / safeRd;
    vec3 b0 = vec3(bc * 8);
    vec3 t0 = (b0 - ro) * inv, t1 = (b0 + 8.0 - ro) * inv;
    vec3 mn = min(t0, t1), mx = max(t0, t1);
    float tEnter = max(max(mn.x, mn.y), mn.z), tExit = min(min(mx.x, mx.y), mx.z);
    if (tExit < max(tEnter, tLo)) return false;
    float t = max(tEnter, tLo);
    if (t > tCap) return false;
    ivec3 stepDir = ivec3(sign(safeRd));
    vec3 normal = vec3(0.0);
    if (tEnter > tLo) {
        if (tEnter == mn.x) normal = vec3(-stepDir.x, 0, 0);
        else if (tEnter == mn.y) normal = vec3(0, -stepDir.y, 0);
        else normal = vec3(0, 0, -stepDir.z);
    }
    if (g >= 256u) return brickVoxels(ro, safeRd, inv, stepDir, bc, g, t, normal, skipMat, h);
    if (VOX_SOLID(g)) {
        h.hit = true;
        h.t = t;
        h.normal = normal;
        h.mat = g;
        h.cell = ivec3(floor(ro + safeRd * (t + 1e-3)));
        return true;
    }
    return false;
}
// Two candidates at exactly the same distance (an edge / corner shared by two voxels): a fixed order
// (cell, then face normal, then material) instead of the order the RT cores report them in.
bool rqTieWins(Hit a, Hit b) {
    if (a.cell != b.cell) return a.cell.x != b.cell.x ? a.cell.x < b.cell.x : a.cell.y != b.cell.y ? a.cell.y < b.cell.y : a.cell.z < b.cell.z;
    if (a.normal != b.normal) return a.normal.x != b.normal.x ? a.normal.x < b.normal.x : a.normal.y != b.normal.y ? a.normal.y < b.normal.y : a.normal.z < b.normal.z;
    return a.mat < b.mat;
}
// The world along the ray (= traceVolumeS over the world, through the RT cores).
// flags gl_RayFlagsTerminateOnFirstHitEXT: ANY hit (occlusion tests), else the nearest.
Hit traceWorldRQF(vec3 ro, vec3 rd, float tCap, uint skipMat, uint flags) {
    Hit best;
    best.hit = false;
    best.t = 0.0;
    best.obj = -1;
    float tMax = min(tCap + 14.0, 1.0e7);   // (+ a brick's diagonal: see rqBrick)
    rayQueryEXT q;
    rayQueryInitializeEXT(q, rtTlas, flags, 0x01u, ro, 0.0, rd, tMax);
    while (rayQueryProceedEXT(q)) {
        if (rayQueryGetIntersectionTypeEXT(q, false) != gl_RayQueryCandidateIntersectionAABBEXT) continue;
        HEAT_STEP();
        uint p = rtPrims[uint(rayQueryGetIntersectionInstanceCustomIndexEXT(q, false)) +
                         uint(rayQueryGetIntersectionPrimitiveIndexEXT(q, false))];
        vec3 org = rayQueryGetIntersectionObjectToWorldEXT(q, false)[3];   // the tile's corner (voxels)
        ivec3 bc = ivec3(floor(org * 0.125 + 0.5)) + ivec3(int(p & 15u), int((p >> 8) & 255u), int((p >> 4) & 15u));
        Hit h;
        h.hit = false;
        h.obj = -1;
        // (equal distances - an edge / corner of two voxels - break by cell, not by the order the RT
        // cores report candidates in: the result doesn't depend on the BLAS layout)
        if (rqBrick(ro, rd, bc, 0.0, tCap, skipMat, h) && (!best.hit || h.t < best.t || (h.t == best.t && rqTieWins(h, best)))) {
            best = h;
            rayQueryGenerateIntersectionEXT(q, h.t);
        }
    }
    return best;
}
Hit traceWorldRQ(vec3 ro, vec3 rd, float tCap, uint skipMat) { return traceWorldRQF(ro, rd, tCap, skipMat, gl_RayFlagsNoneEXT); }
Hit traceSceneRQ(vec3 ro, vec3 rd, float tCap, uint skipMat) {
    Hit best = traceWorldRQ(ro, rd, tCap, skipMat);
    best.obj = -1;
    float limit = best.hit ? best.t : tCap;
    traceObjectsInto(ro, rd, best, limit);
    return best;
}
#endif
// Per ray type: the scene (world + bodies) / the world alone, through the RT cores or the DDA.
#ifdef RT_SHADOW
Hit traceSceneShadow(vec3 ro, vec3 rd, float tCap, uint skipMat) { return traceSceneRQ(ro, rd, tCap, skipMat); }
Hit traceWorldShadow(vec3 ro, vec3 rd, float tCap, uint skipMat) { return traceWorldRQ(ro, rd, tCap, skipMat); }
#else
Hit traceSceneShadow(vec3 ro, vec3 rd, float tCap, uint skipMat) { return traceSceneS(ro, rd, tCap, skipMat); }
Hit traceWorldShadow(vec3 ro, vec3 rd, float tCap, uint skipMat) { return traceVolumeS(ro, rd, worldBricks, true, 0u, tCap, skipMat); }
#endif
// Sun / lamp occlusion where only "blocked?" matters: the world, then the bodies only if it's clear.
// On the RT cores the traversal stops at the first hit found (not the nearest).
bool occludedShadow(vec3 ro, vec3 rd, float tCap, uint skipMat) {
#ifdef RT_SHADOW
    if (traceWorldRQF(ro, rd, tCap, skipMat, gl_RayFlagsTerminateOnFirstHitEXT).hit) return true;
#else
    if (traceVolumeS(ro, rd, worldBricks, true, 0u, tCap, skipMat).hit) return true;
#endif
    Hit b;
    b.hit = false;
    b.t = 0.0;
    b.obj = -1;
    traceObjectsA(ro, rd, b, tCap, true);
    return b.hit;
}
#ifdef RT_GI
Hit traceSceneGi(vec3 ro, vec3 rd, float tCap, uint skipMat) { return traceSceneRQ(ro, rd, tCap, skipMat); }
Hit traceWorldGi(vec3 ro, vec3 rd, float tCap, uint skipMat) { return traceWorldRQ(ro, rd, tCap, skipMat); }
// anyHit (a visibility ray: only .hit is read): the RT cores stop at the first hit found, not the nearest.
Hit traceWorldGiA(vec3 ro, vec3 rd, float tCap, uint skipMat, bool anyHit) {
    return traceWorldRQF(ro, rd, tCap, skipMat, anyHit ? gl_RayFlagsTerminateOnFirstHitEXT : gl_RayFlagsNoneEXT);
}
#else
Hit traceSceneGi(vec3 ro, vec3 rd, float tCap, uint skipMat) { return traceSceneS(ro, rd, tCap, skipMat); }
Hit traceWorldGi(vec3 ro, vec3 rd, float tCap, uint skipMat) { return traceVolumeS(ro, rd, worldBricks, true, 0u, tCap, skipMat); }
Hit traceWorldGiA(vec3 ro, vec3 rd, float tCap, uint skipMat, bool anyHit) { return traceVolumeS(ro, rd, worldBricks, true, 0u, tCap, skipMat); }   // (the DDA's first hit is the nearest)
#endif
#ifdef RT_REFL
Hit traceSceneRefl(vec3 ro, vec3 rd, float tCap, uint skipMat) { return traceSceneRQ(ro, rd, tCap, skipMat); }
Hit traceWorldRefl(vec3 ro, vec3 rd, float tCap, uint skipMat) { return traceWorldRQ(ro, rd, tCap, skipMat); }
#else
Hit traceSceneRefl(vec3 ro, vec3 rd, float tCap, uint skipMat) { return traceSceneS(ro, rd, tCap, skipMat); }
Hit traceWorldRefl(vec3 ro, vec3 rd, float tCap, uint skipMat) { return traceVolumeS(ro, rd, worldBricks, true, 0u, tCap, skipMat); }
#endif
#ifdef RT_PROBE
Hit traceSceneProbe(vec3 ro, vec3 rd, float tCap, uint skipMat) { return traceSceneRQ(ro, rd, tCap, skipMat); }
Hit traceWorldProbe(vec3 ro, vec3 rd, float tCap, uint skipMat) { return traceWorldRQ(ro, rd, tCap, skipMat); }
#else
Hit traceSceneProbe(vec3 ro, vec3 rd, float tCap, uint skipMat) { return traceSceneS(ro, rd, tCap, skipMat); }
Hit traceWorldProbe(vec3 ro, vec3 rd, float tCap, uint skipMat) { return traceVolumeS(ro, rd, worldBricks, true, 0u, tCap, skipMat); }
#endif
#ifdef RT_LAMP
Hit traceSceneLamp(vec3 ro, vec3 rd, float tCap, uint skipMat) { return traceSceneRQ(ro, rd, tCap, skipMat); }
Hit traceWorldLamp(vec3 ro, vec3 rd, float tCap, uint skipMat) { return traceWorldRQ(ro, rd, tCap, skipMat); }
#else
Hit traceSceneLamp(vec3 ro, vec3 rd, float tCap, uint skipMat) { return traceSceneS(ro, rd, tCap, skipMat); }
Hit traceWorldLamp(vec3 ro, vec3 rd, float tCap, uint skipMat) { return traceVolumeS(ro, rd, worldBricks, true, 0u, tCap, skipMat); }
#endif
#ifdef RT_FOG
Hit traceSceneFog(vec3 ro, vec3 rd, float tCap, uint skipMat) { return traceSceneRQ(ro, rd, tCap, skipMat); }
Hit traceWorldFog(vec3 ro, vec3 rd, float tCap, uint skipMat) { return traceWorldRQ(ro, rd, tCap, skipMat); }
// Fog sun visibility (only "blocked?"): any hit ends the RT cores' traversal.
bool occludedWorldFog(vec3 ro, vec3 rd, float tCap, uint skipMat) {
    return traceWorldRQF(ro, rd, tCap, skipMat, gl_RayFlagsTerminateOnFirstHitEXT).hit;
}
#else
Hit traceSceneFog(vec3 ro, vec3 rd, float tCap, uint skipMat) { return traceSceneS(ro, rd, tCap, skipMat); }
Hit traceWorldFog(vec3 ro, vec3 rd, float tCap, uint skipMat) { return traceVolumeS(ro, rd, worldBricks, true, 0u, tCap, skipMat); }
bool occludedWorldFog(vec3 ro, vec3 rd, float tCap, uint skipMat) { return traceWorldFog(ro, rd, tCap, skipMat).hit; }
#endif
#ifdef RT_PRIMARY
Hit traceScenePrimary(vec3 ro, vec3 rd, float tCap, uint skipMat) { return traceSceneRQ(ro, rd, tCap, skipMat); }
Hit traceWorldPrimary(vec3 ro, vec3 rd, float tCap, uint skipMat) { return traceWorldRQ(ro, rd, tCap, skipMat); }
#else
Hit traceScenePrimary(vec3 ro, vec3 rd, float tCap, uint skipMat) { return traceSceneS(ro, rd, tCap, skipMat); }
Hit traceWorldPrimary(vec3 ro, vec3 rd, float tCap, uint skipMat) { return traceVolumeS(ro, rd, worldBricks, true, 0u, tCap, skipMat); }
#endif

// Point query: is the WORLD voxel at world-space position p solid? Cheap (no
// DDA) — a grid fetch plus, for detailed bricks, one atlas fetch. Used by the
// baked-style voxel AO (objects are ignored — their AO contribution is minor).
bool voxelSolidAt(vec3 p) {
    const uint skipMat = kSkipMat;
    ivec3 v = ivec3(floor(p));
    if (any(lessThan(v, worldMinB() * 8)) || any(greaterThanEqual(v, (worldMinB() + worldBricks) * 8)))
        return false;
    ivec3 bc = v >> 3;
    uint g = texelFetch(gridTex, gridWrap(bc), 0).r;
    if (g == 0u) return false;
    if (g < 256u) return VOX_SOLID(g); // uniform brick
    return VOX_SOLID(texelFetch(atlasTex, slotBase(g - 256u) + (v & 7), 0).r);
}

// Point query: the WORLD voxel's MATERIAL at p (0 = air). Like voxelSolidAt but
// returns the palette index — used by the water "look through to the bottom" march.
uint voxelMatAt(vec3 p) {
    ivec3 v = ivec3(floor(p));
    if (any(lessThan(v, worldMinB() * 8)) || any(greaterThanEqual(v, (worldMinB() + worldBricks) * 8))) return 0u;
    ivec3 bc = v >> 3;
    uint g = texelFetch(gridTex, gridWrap(bc), 0).r;
    if (g == 0u) return 0u;
    if (g < 256u) return g;                                  // uniform brick
    return texelFetch(atlasTex, slotBase(g - 256u) + (v & 7), 0).r;
}

// A camera inside solid terrain sees through the solid it's in (spectator-style): the
// distance along rd to the first voxel that isn't solid (air or water); -2 if the ray leaves
// the world first (nothing to see), -1 if it's still solid after maxSteps. Whole solid
// bricks are crossed in one step; every step moves across exactly one face (no rounding).
float exitSolid(vec3 ro, vec3 rd, int maxSteps) {
    vec3 safeRd = mix(rd, vec3(1e-5), lessThan(abs(rd), vec3(1e-5)));
    vec3 inv = 1.0 / safeRd;
    ivec3 st = ivec3(sign(safeRd));
    vec3 up = step(vec3(0.0), safeRd);
    ivec3 v = ivec3(floor(ro));
    float t = 0.0;
    for (int i = 0; i < maxSteps; i++) {
        if (any(lessThan(v, worldMinB() * 8)) || any(greaterThanEqual(v, (worldMinB() + worldBricks) * 8))) return -2.0;
        uint g = texelFetch(gridTex, gridWrap(v >> 3), 0).r;
        if (g == 0u || g == 5u) return t;
        int size = 1;
        ivec3 lo = v;
        if (g < 256u) { size = 8; lo = (v >> 3) * 8; }   // a uniform solid brick: its far side
        else {
            uint m = texelFetch(atlasTex, slotBase(g - 256u) + (v & 7), 0).r;
            if (m == 0u || m == 5u) return t;
        }
        vec3 tFar = ((vec3(lo) + up * float(size)) - ro) * inv;
        bvec3 ax = exitAxis(tFar);
        t = pickAxis(tFar, ax);
        // The next cell: across the exit face on the exit axis, the ray's position on the others.
        ivec3 nv = clamp(ivec3(floor(ro + safeRd * t)), lo, lo + (size - 1));
        v = mix(nv, mix(lo - 1, lo + size, greaterThan(st, ivec3(0))), ax);
    }
    return -1.0;
}

// Debug-view palette: t in [0,1] -> blue, cyan, green, yellow, red.
vec3 heatRamp(float t) {
    t = clamp(t, 0.0, 1.0);
    return clamp(vec3(1.5 - abs(4.0 * t - 3.0), 1.5 - abs(4.0 * t - 2.0),
                      1.5 - abs(4.0 * t - 1.0)), 0.0, 1.0);
}

// r.debugview indices (keep in sync with the cvar help in vk_renderer.cpp).
const int DV_CASCADE = 9;   // Standard lighting: shadow cascade per pixel
const int DV_NORMAL = 0, DV_OBJECTS = 1, DV_HEAT_PRIMARY = 2, DV_HEAT_SHADOW = 3,
          DV_GI = 4, DV_SHADOW = 5, DV_HISTORY = 6, DV_ALBEDO = 7, DV_NORMALS = 8;

// Sky radiance. classic = the original fixed gradient; physical = zenith/horizon
// colours + sunset glow from the sun height (LitRenderer::applyLook), sun disk
// tinted by the atmosphere.
// sunDisk = 0: sky light only (bounce/probe rays — the sun itself is added as
// direct light at every hit, so seeing the disk would count it twice).
// Night sky: stars on a fixed direction grid (hashed cells, no twinkle -> nothing for
// the temporal filters to fight), faded near the horizon.
vec3 starField(vec3 rd) {
    vec3 a = abs(rd);
    vec2 uv = a.x > a.y && a.x > a.z ? rd.yz / a.x : a.y > a.z ? rd.xz / a.y : rd.xy / a.z;
    float face = a.x > a.y && a.x > a.z ? (rd.x > 0.0 ? 0.0 : 1.0) : a.y > a.z ? (rd.y > 0.0 ? 2.0 : 3.0)
                                                                         : (rd.z > 0.0 ? 4.0 : 5.0);
    vec2 g = uv * 180.0;
    vec2 c = floor(g);
    uint h = uint(c.x + 400.0) * 73856093u ^ uint(c.y + 400.0) * 19349663u ^ uint(face) * 83492791u;
    h = (h ^ (h >> 16)) * 0x7feb352du;   // (pcgHash is declared further down)
    h = (h ^ (h >> 15)) * 0x846ca68bu;
    h ^= h >> 16;
    if ((h & 1023u) > 3u) return vec3(0.0);
    vec2 sp = vec2(float((h >> 10) & 255u), float((h >> 18) & 255u)) / 255.0;
    float d = length(g - c - sp);
    float b = (0.4 + 0.6 * float((h >> 26) & 63u) / 63.0) * (1.0 - smoothstep(0.05, 0.35, d));
    return vec3(0.85, 0.9, 1.0) * b * 1.2 * smoothstep(0.0, 0.25, rd.y);
}

// The moon (night: sunDir is the moon): a crisp disk ~1 degree in radius
// (about 4x the real one, as games draw it) with faint static maria, instead of the
// sun's soft pow(mu, 800) blob — which at night read as a second, smaller sun.
vec3 moonDisk(vec3 rd) {
    float mu = dot(rd, sunDir);
    if (mu < 0.998) return vec3(0.0);
    const float kR = 0.018;
    float a = sqrt(max(2.0 * (1.0 - mu), 0.0));   // angle from the centre (small-angle)
    float disk = 1.0 - smoothstep(kR * 0.88, kR, a);
    if (disk <= 0.0) return vec3(0.0);
    vec3 t1 = normalize(cross(abs(sunDir.y) < 0.99 ? vec3(0, 1, 0) : vec3(1, 0, 0), sunDir));
    vec3 t2 = cross(sunDir, t1);
    vec2 uv = vec2(dot(rd, t1), dot(rd, t2)) / kR;   // -1..1 across the disk
    // Maria: two octaves of smooth value noise (a fixed pattern: no flicker).
    vec2 q = uv * 2.2 + 3.1;
    vec2 i = floor(q), f = fract(q);
    f = f * f * (3.0 - 2.0 * f);
    #define MH(c) fract(sin(dot(c, vec2(127.1, 311.7))) * 43758.5453)
    float n1 = mix(mix(MH(i), MH(i + vec2(1, 0)), f.x), mix(MH(i + vec2(0, 1)), MH(i + vec2(1, 1)), f.x), f.y);
    q = uv * 5.3 + 7.7; i = floor(q); f = fract(q); f = f * f * (3.0 - 2.0 * f);
    float n2 = mix(mix(MH(i), MH(i + vec2(1, 0)), f.x), mix(MH(i + vec2(0, 1)), MH(i + vec2(1, 1)), f.x), f.y);
    #undef MH
    float maria = 1.0 - 0.38 * smoothstep(0.4, 0.8, 0.65 * n1 + 0.35 * n2);
    float limb = 0.82 + 0.18 * sqrt(max(1.0 - dot(uv, uv), 0.0));   // slightly darker edge
    return vec3(0.92, 0.94, 1.0) * (1.0 * disk * maria * limb);
}

vec3 skyColor(vec3 rd, float sunDisk) {
    float horizon = clamp(rd.y * 0.5 + 0.5, 0.0, 1.0);
    float sunAmt = pow(max(dot(rd, sunDir), 0.0), 800.0) * sunDisk * (1.0 - night);   // day disk (moon: moonDisk)
    vec3 stars = night > 0.0 && sunDisk > 0.0 ? starField(rd) * night * sunDisk + moonDisk(rd) * night * sunDisk : vec3(0.0);   // (GI rays: sunDisk 0)
    // Glow around the disk: the sun's wide aureole; at night only a faint one (the moon's
    // halo in fog comes from the fog's own in-scatter).
    float glow = pow(max(dot(rd, sunDir), 0.0), 8.0) * (1.0 - 0.9 * night);
    if (skyModel == 0) {
        vec3 sky = mix(vec3(0.75, 0.82, 0.90), vec3(0.25, 0.45, 0.85), pow(horizon, 0.7));
        sky = mix(sky, vec3(0.012, 0.02, 0.045), night);   // classic sky darkens at night
        return (sky + vec3(1.0, 0.9, 0.7) * (sunAmt * 20.0 + glow * 0.25) * (1.0 - 0.7 * night) + stars) * skyScale;
    }
    vec3 sky = mix(skyHorizon, skyZenith, pow(horizon, 0.7));
    float mu = max(dot(rd, sunDir), 0.0);
    sky += sunsetRGB * (pow(mu, 3.0) * exp(-max(rd.y, 0.0) * 5.0) + 0.25 * exp(-abs(rd.y) * 9.0));
    vec3 sunTint = sunRGB / max(max(sunRGB.r, sunRGB.g), 1e-3);
    return (sky + sunTint * vec3(1.0, 0.9, 0.7) * (sunAmt * 20.0 + glow * 0.25) * (1.0 - 0.7 * night) + stars) * skyScale;
}
vec3 skyColor(vec3 rd) { return skyColor(rd, 1.0); }

// Depth-aware upsample of the quarter-res fog (scatter.rgb, transmittance.a).
// A plain bilinear tap bleeds the long sky-march in-scatter (fog.comp marches
// sky pixels all the way to the fog cap, surfaces only to their depth) across
// object silhouettes, painting blocky god-ray halos onto near surfaces. Weight
// the 4 covering quarter-res texels by how closely each one's marched length
// matches this pixel's, so a foreground surface never inherits background fog.
// fogTex/gbPos are passed in (both are per-pass SET(3) bindings). The 1400.0
// sentinel mirrors fog.comp's kFogMaxDist (its march length for sky pixels).
// --- Froxel volumetric fog (r.fog.method froxel) ---------------------------------------
// fp = (range: farthest froxel distance, slices, near distance, render aspect). Slice s
// (continuous) sits at distance near * (range/near)^(s/slices): exponential, so froxels
// stay roughly cube-shaped in view (fine near the camera, coarse far away).
float froxelDepth(float s, vec4 fp) { return fp.z * pow(fp.x / fp.z, s / fp.y); }
float froxelSliceOf(float d, vec4 fp) { return fp.y * log(max(d, fp.z) / fp.z) / log(fp.x / fp.z); }
// Henyey-Greenstein phase (fogG), as in fog.comp.
float fogPhase(float cosT) {
    float g = fogG;
    return (1.0 - g * g) / (4.0 * 3.14159265 * pow(max(1.0 + g * g - 2.0 * g * cosT, 1e-3), 1.5));
}
// Fog between the eye and distance `depth` along rd (depth < 0 = sky): the integrated
// froxel volume (froxel_integrate.comp) at this pixel's own depth, trilinear — so a near
// leaf and the far wall next to it each get exactly their own fog — plus, beyond the
// froxel range, an analytic height-fog tail (unshadowed). Returns (in-scatter .rgb,
// transmittance .a), like fogUpsample.
vec4 fogSample(sampler3D vol, float depth, vec3 rd, vec3 eye, ivec2 pix, ivec2 res, vec4 fp, float start, vec2 haze) {
    const float kSkyDist = 1400.0;   // same fog cap as the old march
    float d = depth < 0.0 ? kSkyDist : min(depth, kSkyDist);
    vec2 uv = (vec2(pix) + 0.5) / vec2(res);
    float dv = min(d, fp.x);
    float c = froxelSliceOf(dv, fp);            // depth(c) = dv; slice s stores its far edge (c = s + 1)
    vec4 v;
    if (c < 1.0) {
        vec4 v0 = texture(vol, vec3(uv, 0.5 / fp.y));
        v = mix(vec4(0.0, 0.0, 0.0, 1.0), v0, clamp(dv / froxelDepth(1.0, fp), 0.0, 1.0));
    } else {
        v = texture(vol, vec3(uv, (c - 0.5) / fp.y));
    }
    float t0 = max(fp.x, start);   // (r.fog.start beyond the froxels: the tail starts there)
    if (d > t0) {   // analytic tail: height fog with the density at its midpoint
        float L = d - t0;
        vec3 pm = eye + rd * (t0 + 0.5 * L);
        float dens = fogDensity * (fogFalloff > 0.0 ? exp(-max(pm.y - fogHeight, 0.0) / fogFalloff) : 1.0);
        vec3 src = (fogSunRGB * fogPhase(dot(rd, sunDir)) + fogAmbient) * dens;   // (open air: beyond the march)
        float e = max(dens, 1e-7), tr = exp(-e * L);
        v.rgb += v.a * src * ((1.0 - tr) / e);
        v.a *= tr;
    }
    // Aerial perspective (r.haze.distance): the air itself, thin, thinning with height (~1.5 km scale) - distant land
    // fades into the horizon sky's own colour in that direction. Analytic: the optical depth of an exponential
    // atmosphere along the ray. (Surfaces only: the sky already is that colour at the horizon.)
    if (depth >= 0.0 && haze.x > 0.0) {
        float H = 15000.0, y0 = eye.y - haze.y;
        float a = rd.y * depth / H;   // (the real distance: the fog's march stops at 140 m)
        float od = haze.x * exp(-max(y0, 0.0) / H) * depth * (abs(a) > 1e-4 ? (1.0 - exp(-a)) / a : 1.0 - 0.5 * a);
        float Ta = exp(-od);
        vec3 hz = skyColor(normalize(vec3(rd.x, 0.12, rd.z)), 0.0);   // (bluish: the sky a little above the horizon)
        v.rgb = v.rgb * Ta + hz * (1.0 - Ta);   // (the weather fog's light is seen through the haze beyond it, ~)
        v.a *= Ta;
    }
    return v;
}

vec4 fogUpsample(sampler2D fogTex, sampler2D gbPos, ivec2 pix, float depth, ivec2 res) {
    // Nearest-depth upsampling (the usual fix for low-res volumetrics): bilinear over the
    // 4 covering quarter-res texels, each weighted by how well the depth it marched to
    // (fog.comp samples gbPos at the texel centre = q*4+2) matches this pixel. At a
    // silhouette none of them may match — then take the best-matching texel of the 3x3
    // around, so background fog never bleeds over a near leaf (the old tolerance of
    // 15 % + 8 voxels did, as a blocky, jittering halo).
    ivec2 qres = (res + 3) / 4;                    // matches renderer qw/qh = (w+3)/4
    vec2 qf = (vec2(pix) + 0.5) / 4.0 - 0.5;       // full-res center in quarter-res space
    ivec2 base = ivec2(floor(qf));
    vec2 f = qf - vec2(base);
    float dRef = depth < 0.0 ? 1400.0 : min(depth, 1400.0);
    float sigma = 0.03 * dRef + 0.5;
    vec4 acc = vec4(0.0);
    float wSum = 0.0, wBest = 0.0;
    for (int dy = 0; dy < 2; dy++)
        for (int dx = 0; dx < 2; dx++) {
            ivec2 q = clamp(base + ivec2(dx, dy), ivec2(0), qres - 1);
            float wb = (dx == 0 ? 1.0 - f.x : f.x) * (dy == 0 ? 1.0 - f.y : f.y);
            float dq = texelFetch(gbPos, clamp(q * 4 + 2, ivec2(0), res - 1), 0).w;
            dq = dq < 0.0 ? 1400.0 : min(dq, 1400.0);
            float e = (dq - dRef) / sigma;
            float wD = exp(-e * e);
            wBest = max(wBest, wD);
            float w = wb * wD + 1e-6;
            acc += texelFetch(fogTex, q, 0) * w;
            wSum += w;
        }
    if (wBest > 0.25) return acc / wSum;
    ivec2 bq = clamp(ivec2(floor(qf + 0.5)), ivec2(0), qres - 1);
    float bestE = 1e30;
    for (int dy = -1; dy <= 1; dy++)
        for (int dx = -1; dx <= 1; dx++) {
            ivec2 q = clamp(ivec2(floor(qf + 0.5)) + ivec2(dx, dy), ivec2(0), qres - 1);
            float dq = texelFetch(gbPos, clamp(q * 4 + 2, ivec2(0), res - 1), 0).w;
            dq = dq < 0.0 ? 1400.0 : min(dq, 1400.0);
            float e = abs(dq - dRef);
            if (e < bestE) { bestE = e; bq = q; }
        }
    return texelFetch(fogTex, bq, 0);
}

// HALF-res filter guide (written by gi.comp, read by atrous + giUpsample): exact
// bit copies of the full-res G-buffer texel h*2 that represents half-res texel h —
// world position (float bits) + the RGBA8 normal (packed). Reading this compact
// half-res image instead of scattering over full-res gbPos/gbNormal is what makes
// the filter taps cheap. .w == 0 <=> sky (gbuffer writes normal 0 only for sky).
// Half-res GI texel h -> its representative full-res texel: the first non-sky texel
// of its 2x2 block in a fixed order (h*2 when that is geometry, the common case).
ivec2 repTexel(sampler2D posTex, ivec2 h, ivec2 fullRes) {
    for (int k = 0; k < 4; k++) {
        ivec2 q = min(h * 2 + ivec2(k & 1, k >> 1), fullRes - 1);
        if (texelFetch(posTex, q, 0).w >= 0.0) return q;
    }
    return min(h * 2, fullRes - 1);
}
struct Guide { vec3 P; vec3 N; bool sky; };
Guide unpackGuide(uvec4 g) {
    Guide o;
    o.P = uintBitsToFloat(g.xyz);
    o.N = unpackUnorm4x8(g.w).xyz * 2.0 - 1.0;
    o.sky = g.w == 0u;
    return o;
}

// Rec.709 luma — the scalar signal the SVGF denoiser tracks variance of.
float luminance(vec3 c) { return dot(c, vec3(0.2126, 0.7152, 0.0722)); }

// --- Small RNG (PCG) + sampling helpers for the stochastic passes ---
uint pcgHash(uint v) {
    v = v * 747796405u + 2891336453u;
    v = ((v >> ((v >> 28u) + 4u)) ^ v) * 277803737u;
    return (v >> 22u) ^ v;
}

// Per-pixel, per-frame RNG seed. A chained hash, NOT the old `x*A ^ y*B ^ f*C`:
// that XOR of linear terms correlates pixels along diagonals, and DLSS-RR
// faithfully reconstructs the structured noise as a "dithered"/hatched texture on
// flat surfaces once it has converged on a still camera.
uint pixelSeed(ivec2 pix, uint frame, uint salt) {
    return pcgHash(uint(pix.x) + pcgHash(uint(pix.y) + pcgHash(frame ^ salt)));
}

float rnd(inout uint state) {
    state = pcgHash(state);
    // 24 bits: float(state) rounds values near 2^32 up to exactly 1.0, which made
    // cosineHemisphere / sampleCone produce degenerate (tangent / edge) directions.
    return float(state >> 8u) * (1.0 / 16777216.0);
}

void makeBasis(vec3 n, out vec3 t1, out vec3 t2) {
    t1 = normalize(cross(n, abs(n.y) < 0.9 ? vec3(0, 1, 0) : vec3(1, 0, 0)));
    t2 = cross(n, t1);
}

vec3 sampleCone(vec3 dir, float angRadius, inout uint state) {
    vec3 t1, t2;
    makeBasis(dir, t1, t2);
    float r = angRadius * sqrt(rnd(state));
    float ph = 6.2831853 * rnd(state);
    return normalize(dir + t1 * (r * cos(ph)) + t2 * (r * sin(ph)));
}

vec3 cosineHemisphereU(vec3 n, vec2 u) {
    vec3 t1, t2;
    makeBasis(n, t1, t2);
    float r = sqrt(u.x);
    float ph = 6.2831853 * u.y;
    return normalize(t1 * (r * cos(ph)) + t2 * (r * sin(ph)) + n * sqrt(1.0 - u.x));
}
vec3 cosineHemisphere(vec3 n, inout uint state) {
    float u = rnd(state);
    return cosineHemisphereU(n, vec2(u, rnd(state)));
}

// Spatio-temporal low-discrepancy 2D sample for pixel `pix`, sample `k` of its own
// sequence: a per-pixel random Cranley-Patterson rotation (spatially decorrelated)
// of the R2 sequence (Roberts 2018) over k. Consecutive samples of a pixel cover the
// square evenly instead of clumping like white noise, so the temporal average
// converges ~k^-1 instead of k^-1/2 — less residual crawl for the same rays.
vec2 ldSample2(ivec2 pix, uint k, uint salt) {
    uint h = pcgHash(uint(pix.x) + pcgHash(uint(pix.y) + salt));
    vec2 rot = vec2(float(h >> 8u), float(pcgHash(h) >> 8u)) * (1.0 / 16777216.0);
    return fract(rot + float(k & 0xFFFFu) * vec2(0.7548776662, 0.5698402910));
}

// Cheap deterministic "voxel AO": sample nearby world voxels in the hemisphere
// above the hit; solid samples occlude. No ray march, no temporal, no noise —
// replaces the incoherent cosine AO ray. Two rings; nearer occluders weigh more.
// The body the shaded point lies on (-1 = world), set by the shading pass from
// gbDelta.a (gbuffer.comp): with r.gi.objects its own voxels occlude it too (a hole
// carved into a body darkens) — one body, no search.
int gSelfBody = -1;
int selfBodyOf(float deltaA) { return int(deltaA + 0.5) - 1; }
float voxelAO(vec3 P, vec3 N) {
    vec3 t1, t2;
    makeBasis(N, t1, t2);
    const vec2 disk[8] = vec2[](vec2(1, 0), vec2(0.707, 0.707), vec2(0, 1),
                                vec2(-0.707, 0.707), vec2(-1, 0), vec2(-0.707, -0.707),
                                vec2(0, -1), vec2(0.707, -0.707));
    float occ = 0.0, wsum = 0.0;
    for (int ring = 0; ring < 2; ring++) {
        float rad = ring == 0 ? 1.3 : 2.7; // tangential spread (voxels)
        float up = ring == 0 ? 1.1 : 2.3;  // lift along the normal
        float w = ring == 0 ? 1.0 : 0.5;
        for (int i = 0; i < 8; i++) {
            vec3 sp = P + N * up + (t1 * disk[i].x + t2 * disk[i].y) * rad;
            if (voxelSolidAt(sp)) occ += w;
            wsum += w;
        }
    }
#ifdef GI_OBJECTS
    // On a body: its own voxels occlude too. A separate loop, so world pixels (almost
    // all of them) keep the lean world-only one above.
    if (gSelfBody >= 0)
        for (int ring = 0; ring < 2; ring++) {
            float rad = ring == 0 ? 1.3 : 2.7;
            float up = ring == 0 ? 1.1 : 2.3;
            float w = ring == 0 ? 1.0 : 0.5;
            for (int i = 0; i < 8; i++) {
                vec3 sp = P + N * up + (t1 * disk[i].x + t2 * disk[i].y) * rad;
                if (objectSolidAt(gSelfBody, sp) && !voxelSolidAt(sp)) occ += w;
            }
        }
#endif
    return clamp(1.0 - occ / wsum, 0.0, 1.0);
}

// Medium-range occlusion (4-10 voxels) for Standard lighting: its probes are 8 voxels
// apart, so a hole / crater / alcove smaller than that gets the open air's light. Two
// rings of 8 point samples like voxelAO, farther out (the traced GI rays of the other
// modes see this occlusion themselves). The 16 samples are SHARED by a quad of 4
// neighbouring pixels (4 each, summed with subgroup quad ops; a neighbour counts only
// on the same surface): 16-sample quality at 4 lookups per pixel. Call it from uniform
// control flow (every lane of the quad), `hasSurface` = this lane has one.
float voxelAOFarQuad(vec3 P, vec3 N, bool hasSurface) {
    const vec2 disk[8] = vec2[](vec2(1, 0), vec2(0.707, 0.707), vec2(0, 1),
                                vec2(-0.707, 0.707), vec2(-1, 0), vec2(-0.707, -0.707),
                                vec2(0, -1), vec2(0.707, -0.707));
    uint lane = gl_SubgroupInvocationID & 3u;
    float occ = 0.0, wsum = 0.0;
    if (hasSurface) {
        vec3 t1, t2;
        makeBasis(N, t1, t2);
        for (uint s = 0u; s < 4u; s++) {
            uint i = lane + 4u * s;          // lanes interleave: each covers both rings
            bool outer = i >= 8u;
            vec2 dk = disk[i & 7u];
            if (outer) dk = vec2(dk.x - dk.y, dk.x + dk.y) * 0.7071;   // outer ring rotated 45 deg
            float w = outer ? 0.7 : 1.0;
            vec3 sp = P + N * (outer ? 6.5 : 3.5) + (t1 * dk.x + t2 * dk.y) * (outer ? 8.0 : 4.0);
#ifdef GI_OBJECTS
            if (voxelSolidAt(sp) || objectSolidAt(gSelfBody, sp)) occ += w;
#else
            if (voxelSolidAt(sp)) occ += w;
#endif
            wsum += w;
        }
    }
    vec2 ow = vec2(occ, wsum);
    vec2 acc = vec2(0.0);
#define AO_QUAD_TAKE(q) { \
        vec2 o_ = subgroupQuadBroadcast(ow, q); vec3 p_ = subgroupQuadBroadcast(P, q); \
        vec3 n_ = subgroupQuadBroadcast(N, q); \
        if (dot(p_ - P, p_ - P) < 4.0 && dot(n_, N) > 0.7) acc += o_; }
    AO_QUAD_TAKE(0u) AO_QUAD_TAKE(1u) AO_QUAD_TAKE(2u) AO_QUAD_TAKE(3u)
#undef AO_QUAD_TAKE
    return acc.y > 0.0 ? clamp(1.0 - acc.x / acc.y, 0.0, 1.0) : 1.0;
}

// THE surface shading model, shared by denoise.comp (SVGF path) and raw_shade.comp
// (DLSS-RR COLOR_IN) so both paths render the same lighting: direct sun + 1-bounce
// GI (indirect, contact-darkened) + emissive; water gets a Fresnel sky reflection,
// a see-through body and a sun glint. `shadow` = sun visibility, `ndl` = N·sun.
// sunRGB (scene uniform) is the one sun colour used by every pass.
#ifdef WATER_CLEAR
// --- Transparent water (r.water 1) ---------------------------------------------
// Per-voxel absorption (Beer-Lambert): red goes first, so depth turns blue-green.
// Clarity (r.water.clarity): set by passes from their waterP uniform (pass uniforms
// are declared after this file).
// Extinction = absorption (red goes first) + scattering by what floats in it (r.water.murk,
// grey): at murk 1 you see ~8 m in a lake (blue, 5 % left at ~80 voxels).
float gWaterClarity = 1.0, gWaterMurk = 1.0;
vec3 waterSigma() { return (vec3(0.075, 0.030, 0.021) + 0.015 * gWaterMurk) / max(gWaterClarity, 0.05); }
// The one-number version (green) the froxel volume integrates with; the shading passes fix up
// the per-channel difference on what's seen through it (waterFroxelFix).
float waterExt() { return waterSigma().g; }
vec3 waterTrans(float len) { return exp(-waterSigma() * max(len, 0.0)); }
// Light going DOWN through the water (sun, sky): particles scatter it mostly forward, so it keeps
// going - only a quarter of the scattering counts against it (the view path takes all of it).
vec3 waterTransLight(float len) {
    return exp(-(vec3(0.075, 0.030, 0.021) + 0.25 * 0.015 * gWaterMurk) / max(gWaterClarity, 0.05) * max(len, 0.0));
}
float waterFresnel(vec3 N, vec3 V) { return 0.02 + 0.98 * pow(1.0 - max(dot(N, V), 0.0), 5.0); }
const vec3 kWaterScatter = vec3(0.05, 0.19, 0.24);   // in-scattered body colour (albedo)
// Seen from inside (camera under water): the lit water all around, greener and brighter than the
// body colour seen through the surface from above (no reflection on top, eyes adapted to it).
const vec3 kWaterScatterIn = vec3(0.045, 0.24, 0.21);
const float kWaterSunIn = 0.6;   // share of the sun scattered toward you (x phase)
// Light reaching an underwater point `depth` voxels below the surface: set by the
// shading passes from gbWater before shadeSurface (1 = not underwater).
// gbWater (gbuffer.comp): x = water path length along the view ray; y >= 0: the bed's depth
// below the surface seen from above; y < 0: the camera is under water and the point seen is
// -(y + 1) below the surface (0 = above the water).
float waterLightDepth(vec4 wv) { return wv.x > 0.0 ? (wv.y >= 0.0 ? wv.y : -wv.y - 1.0) : 0.0; }
// Water straight above p, in voxels (to the surface; capped).
float waterAbove(vec3 p, int cap) {
    for (int i = 0; i < cap; i++)
        if (voxelMatAt(p + vec3(0.0, float(i), 0.0)) != 5u) return float(i);
    return float(cap);
}
// --- The sun under water: its light enters refracted (1 -> 1.33), so below the surface it comes
// from sunDirWater() - steeper than the sun in the sky, and where Snell's window shows it. Shading,
// shadows, shafts and caustics under water all use it.
vec3 sunDirWater() {
    if (sunDir.y <= 0.0) return sunDir;
    return -refract(-sunDir, vec3(0.0, 1.0, 0.0), 1.0 / 1.33);
}
// How far p is under the water surface above it (voxels; 0 = not in water).
float waterDepthAt(vec3 p) {
    if (voxelMatAt(p) != 5u) return 0.0;
    return floor(p.y + waterAbove(p, 128)) - p.y;
}
// A sun ray from P `depth` under the surface: along sunDirWater to where that light came in
// (anything under the water in the way shadows it: `blocked`); returns that entry point, just above
// the surface - the origin for the usual sun test toward sunDir.
vec3 sunUnderwater(vec3 P, float depth, out bool blocked) {
    vec3 w = sunDirWater();
    float L = depth / max(w.y, 0.05);
    blocked = occludedShadow(P, w, L, kSkipMat);
    return P + w * L + vec3(0.0, 0.05, 0.0);
}
vec3 gSunTrans = vec3(1.0), gAmbTrans = vec3(1.0);
void setWaterLight(float depth) {
    if (depth <= 0.0) { gSunTrans = vec3(1.0); gAmbTrans = vec3(1.0); return; }
    gSunTrans = waterTransLight(depth / max(sunDirWater().y, 0.1));
    gAmbTrans = waterTransLight(depth * 1.4);   // sky light arrives from all directions
}
// --- Lake waves (surface normals + caustics share them) -----------------------------------
// Eight travelling waves around the wind direction, no two alike: wavelengths 1.5-24 voxels
// (geometric), directions spread +-70 degrees (hashed), amplitude ~ wavelength^0.8 (gentle, a lake),
// speed from deep-water dispersion (w = sqrt(g k): long waves run faster). Their crests are bent by a
// slow domain warp, and a drifting large-scale noise makes glassy patches next to rippled ones (wind
// gusts on a lake). gWave = (wind dir xz, wind strength, r.water.waves) - set by the passes.
vec4 gWave = vec4(0.8, 0.6, 1.0, 1.0);
float wvHash(vec2 p) { p = fract(p * vec2(233.34, 851.73)); p += dot(p, p + 23.45); return fract(p.x * p.y); }
float wvNoise(vec2 p) {   // smooth value noise, 0..1
    vec2 i = floor(p), f = fract(p), u = f * f * (3.0 - 2.0 * f);
    return mix(mix(wvHash(i), wvHash(i + vec2(1, 0)), u.x), mix(wvHash(i + vec2(0, 1)), wvHash(i + vec2(1, 1)), u.x), u.y);
}
vec3 wvNoiseD(vec2 p) {   // value noise and its gradient
    vec2 i = floor(p), f = fract(p), u = f * f * (3.0 - 2.0 * f), du = 6.0 * f * (1.0 - f);
    float a = wvHash(i), b = wvHash(i + vec2(1, 0)), c = wvHash(i + vec2(0, 1)), d = wvHash(i + vec2(1, 1));
    return vec3(a + (b - a) * u.x + (c - a) * u.y + (a - b - c + d) * u.x * u.y,
                du * (vec2(b - a, c - a) + (a - b - c + d) * u.yx));
}
// ---- Clouds -------------------------------------------------------------------------------------------------------
// Cumulus of 1 m voxels (10 x 10 x 10): each 1 m column of the sky holds one cloud span [base, top] (quantised to
// whole voxels) from a drifting, slowly changing noise - big clouds tall, a flat-ish base. A ray crosses the columns
// one by one (a 2D grid walk; coarser cells far away), and each voxel it passes through takes a share of the light:
// 1 - e^(-k x the length inside), so a thin wisp is see-through and a big cloud opaque. Lit by the sun through the
// cloud toward it (Beer), forward-scattering bright round the sun (a silver lining, as the leaves' translucency),
// darker inside (powder), plus the sky's light from above.
bool cloudsOn() { return cloudB.w > 0.5; }
// The cloud field lives in the drifting pattern's coordinates (pp = engine xz + cloudB.xy): whole voxels glide with the
// wind instead of the pattern sliding through a fixed grid (blocks popping on and off).
// The coarse part of the field (the two big octaves) and the most the finer ones can add: an exact test for empty sky.
// Gradient noise (0..1): value noise's axis-aligned ridges drew long straight edges on the clouds.
vec2 cgGrad(vec2 i) {
    uint h = pcgHash(uint(int(i.x)) * 1597334677u ^ uint(int(i.y)) * 3812015801u);
    float a = float(h & 65535u) * (6.2831853 / 65536.0);
    return vec2(cos(a), sin(a));
}
float cgNoise(vec2 p) {
    vec2 i = floor(p), f = p - i, u = f * f * f * (f * (f * 6.0 - 15.0) + 10.0);
    float a = dot(cgGrad(i), f), b = dot(cgGrad(i + vec2(1, 0)), f - vec2(1, 0));
    float c = dot(cgGrad(i + vec2(0, 1)), f - vec2(0, 1)), d = dot(cgGrad(i + vec2(1, 1)), f - vec2(1, 1));
    return 0.5 + 0.7 * mix(mix(a, b, u.x), mix(c, d, u.x), u.y);
}
const mat2 kCloudRot = mat2(0.8, -0.6, 0.6, 0.8);   // (each octave turned: no shared grid)
float cloudCoarse(vec2 p, float ev) {
    return cgNoise(p / 2600.0 + vec2(ev * 0.7, -ev * 0.4)) * 0.55 + cgNoise(kCloudRot * p / 1100.0 + vec2(-ev, ev * 0.6) + 17.3) * 0.28;
}
const float kCloudFineMax = 0.17;
// The span of the column at pattern xz: (base, top) in voxels; empty: top <= base. `quant`: whole voxels.
vec2 cloudColumnP(vec2 p, bool quant) {
    float ev = cloudB.z;
    float n = cloudCoarse(p, ev) + cgNoise(kCloudRot * kCloudRot * p / 420.0 + vec2(ev * 1.6, ev) + 41.7) * 0.12 +
              cgNoise(transpose(kCloudRot) * p / 140.0 + 3.1) * 0.05;
    float d = n - (1.0 - cloudA.x) * 0.92 - 0.04;
    if (d <= 0.0) return vec2(1.0, 0.0);
    float k = clamp(d / 0.35, 0.0, 1.0);
    float th = cloudA.z * pow(k, 0.65);
    // flat bases (the condensation level - the same for every cloud; a base varying in 1 m steps showed each step as a
    // long line seen from below), only the cloud's thin edges a voxel higher
    float base = cloudA.y + (1.0 - smoothstep(0.0, 0.15, k)) * 10.0;
    float top = base + th;
    if (quant) { base = floor(base / 10.0) * 10.0; top = floor(top / 10.0) * 10.0; }
    return vec2(base, top);
}
vec2 cloudColumn(vec2 xz) { return cloudColumnP(xz + cloudB.xy, true); }
// Optical depth of the cloud along a short path toward the sun from p (3 samples)
float cloudSunDepth(vec3 p) {
    float od = 0.0, t = 0.0;
    const float st[3] = float[](25.0, 60.0, 140.0);
    for (int i = 0; i < 3; ++i) {
        t += st[i];
        vec3 q = p + sunDir * t;
        vec2 c = cloudColumn(q.xz);
        if (q.y > c.x && q.y < c.y) od += st[i];
    }
    return od * cloudA.w;
}
// The cloud between the eye and distance tMax along rd: (in-scattered light, transmittance). Marched once per 2x2
// pixels by clouds.comp (cloudComposite upsamples it). Near the eye (< kCloudNear horizontally) a walk over every 1 m
// column; further out the same columns sampled sparser along the ray (steps growing with distance, jittered - the
// temporal passes average it): the shapes are identical, only the sampling thins where a voxel is under a pixel.
const float kCloudNear = 2500.0, kCloudFar = 16000.0;   // (voxels, horizontal: exact walk / fade-out end)
vec4 cloudMarch(vec3 ro, vec3 rd, float tMax, float jit) {
    vec4 res = vec4(0.0, 0.0, 0.0, 1.0);
    if (!cloudsOn()) return res;
    float ylo = cloudA.y - 0.2 * cloudA.z - 20.0, yhi = cloudA.y + cloudA.z + 10.0;
    float hl = length(rd.xz);
    float t0, t1;
    if (abs(rd.y) < 1e-5) {
        if (ro.y < ylo || ro.y > yhi) return res;
        t0 = 0.0; t1 = 1e9;
    } else {
        float a = (ylo - ro.y) / rd.y, b = (yhi - ro.y) / rd.y;
        t0 = max(min(a, b), 0.0); t1 = max(a, b);
    }
    t1 = min(min(t1, tMax), min(60000.0, kCloudFar / max(hl, 1e-4)));   // (past kCloudFar the haze has them)
    if (t0 >= t1) return res;
    // ambient: the sky's light, greyed by the light the cloud scatters inside itself and the land's bounce from below
    vec3 skyUp = skyColor(vec3(0.0, 1.0, 0.0), 0.0);
    float oc = smoothstep(0.55, 0.95, cloudA.x);   // (overcast: the deck lit mostly by itself - a lighter, cool grey)
    vec3 amb = mix(skyUp, vec3(dot(skyUp, vec3(0.2126, 0.7152, 0.0722))) * vec3(0.97, 1.0, 1.06), 0.65) * (1.25 + 0.7 * oc);
    float cosT = dot(rd, sunDir);
    float g1 = 0.5, g2 = -0.25;   // (a softer forward lobe: no grainy glare through thin cloud near the sun)
    float hg = 0.75 * (1.0 - g1 * g1) / pow(1.0 + g1 * g1 - 2.0 * g1 * cosT, 1.5) +
               0.25 * (1.0 - g2 * g2) / pow(1.0 + g2 * g2 - 2.0 * g2 * cosT, 1.5);   // (x 4 pi: 1 = isotropic)
    hg = min(hg, 3.0);
    float sunUp = smoothstep(-0.02, 0.08, sunDir.y);
    float t = t0;
    int steps = 0;
    vec2 o = ro.xz + cloudB.xy;   // (the ray in pattern coordinates)
    float thr = (1.0 - cloudA.x) * 0.92 + 0.04;
    float tNear = kCloudNear / max(hl, 1e-4);
    const float cs = 10.0;   // 1 m voxels
    vec2 lastCb = vec2(1e30);
    float lastCc = 0.0;
    while (t < t1 && steps < 320 && res.a > 0.03) {
        vec2 p = o + rd.xz * t;
        // empty-space skip: a 16 m cell whose coarse field + the finer octaves' most can't reach the threshold
        {
            const float B = 160.0;
            vec2 cb = floor(p / B);
            // (the coarse octaves change by at most ~0.08 over half a 16 m cell: the centre plus a margin is exact)
            if (cb != lastCb) { lastCb = cb; lastCc = cloudCoarse((cb + 0.5) * B, cloudB.z); }
            if (lastCc + kCloudFineMax + 0.12 < thr) {
                vec2 tw = vec2(rd.x > 0.0 ? ((cb.x + 1.0) * B - o.x) / rd.x : rd.x < 0.0 ? (cb.x * B - o.x) / rd.x : 1e30,
                               rd.z > 0.0 ? ((cb.y + 1.0) * B - o.y) / rd.z : rd.z < 0.0 ? (cb.y * B - o.y) / rd.z : 1e30);
                t = max(min(tw.x, tw.y), t + 1e-2);
                ++steps;
                continue;
            }
        }
        float a, b;
        vec2 col;
        if (t < tNear) {   // exact: the ray's stretch through this 1 m column
            vec2 c = floor(p / cs);
            vec2 tw = vec2(rd.x > 0.0 ? ((c.x + 1.0) * cs - o.x) / rd.x : rd.x < 0.0 ? (c.x * cs - o.x) / rd.x : 1e30,
                           rd.z > 0.0 ? ((c.y + 1.0) * cs - o.y) / rd.z : rd.z < 0.0 ? (c.y * cs - o.y) / rd.z : 1e30);
            float tOut = max(min(min(tw.x, tw.y), t1), t + 1e-3);
            col = cloudColumnP((c + 0.5) * cs, true);
            a = t; b = tOut;
        } else {           // sparse: a step ~1 % of the distance, standing for the columns it crosses
            float st = max(cs, 0.016 * t * hl) / max(hl, 0.05);
            float tOut = min(t + st, t1);
            vec2 q = o + rd.xz * mix(t, tOut, jit);
            col = cloudColumnP((floor(q / cs) + 0.5) * cs, true);
            a = t; b = tOut;
        }
        float tNext = b;
        if (col.y > col.x) {
            if (abs(rd.y) > 1e-6) {
                float ta = (col.x - ro.y) / rd.y, tb = (col.y - ro.y) / rd.y;
                a = max(a, min(ta, tb)); b = min(b, max(ta, tb));
            } else if (ro.y < col.x || ro.y > col.y) { b = a; }
            if (b > a) {
                float od = (b - a) * cloudA.w;
                float alpha = (1.0 - exp(-od)) * (1.0 - smoothstep(3000.0, kCloudFar, a * hl));
                vec3 P = ro + rd * (0.5 * (a + b));
                float sod = cloudA.w * 0.7 * max(col.y - P.y, 0.0) / max(sunDir.y, 0.25);
                float Tsun = max(exp(-sod), 0.3 * exp(-sod * 0.25)) * sunUp;
                float powder = 1.0 - 0.6 * exp(-2.0 * (sod + od));
                float hFrac = clamp((P.y - col.x) / max(col.y - col.x, 1.0), 0.0, 1.0);
                vec3 L = sunRGB * Tsun * (0.35 + 0.65 * hg) * powder + amb * (0.45 + 0.55 * hFrac) * 0.9;
                res.rgb += res.a * alpha * L;
                res.a *= 1.0 - alpha;
            }
        }
        t = tNext;
        ++steps;
    }
    return res;
}
// The sky with its clouds in direction d - for reflections (water): clouds_pano.comp marches them from the eye into a
// small panorama of the upper hemisphere each frame (as games reflect their clouds through a sky capture, rather
// than marching them again per reflected ray). Azimuth across, elevation (0 .. 90 deg) down.
vec2 cloudPanoUV(vec3 d) { return vec2(atan(d.z, d.x) / 6.2831853 + 0.5, asin(clamp(d.y, 0.0, 1.0)) / 1.5707963); }
vec3 skyWithClouds(vec3 d) {
    vec3 c = skyColor(d, 0.0);
    if (!cloudsOn() || d.y <= 0.0) return c;
    vec4 p = textureLod(cloudPanoTex, cloudPanoUV(d), 0.0);
    return c * p.a + p.rgb;
}
// Distance along rd at which the ray enters the cloud layer (1e30: never): a surface nearer has no cloud before it.
float cloudEntry(vec3 ro, vec3 rd) {
    float ylo = cloudA.y - 0.2 * cloudA.z - 20.0, yhi = cloudA.y + cloudA.z + 10.0;
    if (ro.y >= ylo && ro.y <= yhi) return 0.0;
    if (abs(rd.y) < 1e-5) return 1e30;
    float a = (ylo - ro.y) / rd.y, b = (yhi - ro.y) / rd.y;
    return max(a, b) < 0.0 ? 1e30 : max(min(a, b), 0.0);
}
// The composite passes: a pixel's clouds from clouds.comp's half-resolution result (none in front of near surfaces).
vec4 cloudComposite(sampler2D cloudTex, ivec2 pix, ivec2 res, vec3 ro, vec3 rd, float depth) {
    if (!cloudsOn()) return vec4(0.0, 0.0, 0.0, 1.0);
    if (depth >= 0.0 && depth <= cloudEntry(ro, rd)) return vec4(0.0, 0.0, 0.0, 1.0);
    return textureLod(cloudTex, (vec2(pix) + 0.5) / vec2(res), 0.0);
}
// Sunlight through the clouds above P (0 under a big cloud .. 1): their shadows on the land.
float cloudShadow(vec3 P) {
    if (!cloudsOn() || sunDir.y <= 0.02) return 1.0;
    float mid = cloudA.y + 0.3 * cloudA.z;
    if (P.y > mid) return 1.0;
    float t = (mid - P.y) / sunDir.y;
    vec3 q = P + sunDir * t;
    vec2 pp = q.xz + cloudB.xy;
    if (cloudCoarse(pp, cloudB.z) + kCloudFineMax < (1.0 - cloudA.x) * 0.92 + 0.04) return 1.0;   // (clear sky above)
    vec2 c = cloudColumnP(floor(pp / 10.0) * 10.0 + 5.0, true);
    float th = max(c.y - c.x, 0.0);
    return mix(0.06, 1.0, exp(-th * cloudA.w * 0.6));   // (a thick cloud lets little direct sun through)
}

struct Wave { vec2 d; float k, a, w, ph; };
Wave lakeWave(int i) {
    float fi = float(i);
    float L = 1.5 * pow(16.0, fi / 7.0);
    float ang = atan(gWave.y, gWave.x) + (wvHash(vec2(fi, 3.7)) * 2.0 - 1.0) * 1.2;
    Wave w;
    w.d = vec2(cos(ang), sin(ang));
    w.k = 6.2831853 / L;
    w.a = 0.01 * pow(L, 0.8) * gWave.w * (0.15 + 0.85 * min(gWave.z, 2.0));
    w.w = sqrt(98.0 * w.k) * 0.7;
    w.ph = wvHash(vec2(fi, 9.1)) * 6.2831853;
    return w;
}
// Where the waves are sampled (crests bent) and how rough the water is here (gust patches).
vec2 lakeWarp(vec2 p, float t) {
    return p + 3.0 * (vec2(wvNoise(p * 0.045 + vec2(t * 0.05, 0.0)), wvNoise(p * 0.045 + vec2(17.3, -t * 0.04))) - 0.5);
}
// Gust patches on the water (cat's paws): drift downwind, like the grass's gust fronts (the sign was upwind).
float lakeRough(vec2 p, float t) {
    return 0.35 + 0.9 * smoothstep(0.25, 0.75, wvNoise(p * 0.012 - gWave.xy * (t * 0.03)));
}
// Surface normal at p (world xz); fp = pixel footprint (voxels): waves shorter than ~2 pixels fade
// out instead of sparkling.
// Rain on water (rainA.x): every 30 cm cell takes a drop now and then - a ring that spreads (~25 cm/s) and fades
// over ~0.7 s (Lagarde's rain ripples); the slope of the rings. Fades out where a ring is under a pixel.
vec2 rainRipples(vec2 p, float fp) {
    float R = rainA.x;
    if (R <= 0.001 || fp > 1.2) return vec2(0.0);
    const float cs = 3.0, life = 0.7, speed = 2.5;
    vec2 c0 = floor(p / cs);
    vec2 g = vec2(0.0);
    for (int j = -1; j <= 1; ++j)
        for (int i = -1; i <= 1; ++i) {
            vec2 c = c0 + vec2(i, j);
            uint k = pcgHash(uint(int(c.x) + 32768) * 1597334677u ^ uint(int(c.y) + 32768) * 3812015801u ^ 4421u);
            float h0 = float(k & 1023u) / 1024.0, h1 = float((k >> 10) & 1023u) / 1024.0;
            float T = mix(0.5, 1.6, h0) / max(R, 0.1);
            float tt = rainA.y + h1 * 17.3;
            float age = mod(tt, T);
            if (age > life) continue;
            uint kn = pcgHash(k ^ uint(floor(tt / T)) * 2654435761u);   // (each drop lands somewhere new)
            vec2 X = (c + 0.15 + 0.7 * vec2(float(kn & 1023u), float((kn >> 10) & 1023u)) / 1024.0) * cs;
            vec2 dv = p - X;
            float r = length(dv);
            float x = r - age * speed;                       // distance behind the ring front
            if (abs(x) > 0.6 || r < 1e-3) continue;
            float f = 1.0 - age / life;
            // two crests behind the front, slope along the radius
            g += dv / r * (cos(x * 10.5) * exp(-x * x * 9.0) * f * f * 0.35);
        }
    return g * (1.0 - smoothstep(0.5, 1.2, fp));
}
vec3 lakeNormal(vec2 p, float t, float fp) {
    vec2 q = lakeWarp(p, t);
    float m = lakeRough(p, t);
    vec2 g = vec2(0.0);
    for (int i = 0; i < 8; i++) {
        Wave w = lakeWave(i);
        float lod = 1.0 - smoothstep(0.5, 1.0, fp * w.k / 3.14159265);
        g += w.d * (lod * m * w.a * w.k * cos(dot(w.d, q) * w.k - w.w * t + w.ph));
    }
    // fine ripples: two octaves of noise drifting with the wind
    float rl = 1.0 - smoothstep(0.4, 1.0, fp * 1.5);
    vec3 n1 = wvNoiseD(q * 1.3 + gWave.xy * (t * 0.8));
    vec3 n2 = wvNoiseD(q * 2.9 - gWave.yx * (t * 1.1));
    g += (n1.yz * 1.3 * 0.035 + n2.yz * 2.9 * 0.012) * m * rl * gWave.w * (0.3 + 0.7 * min(gWave.z, 2.0));
    g += rainRipples(p, fp);
    return normalize(vec3(-g.x, 1.0, -g.y));
}
// Caustics: sunlight focused by the waves onto a point `depth` voxels under the surface. A sun ray
// refracted at slope g of the surface lands displaced by ~depth (1 - 1/1.33) g, so the light
// density down there is 1 / |det J|, J = I + D dg/dx: bright where the surface curvature focuses
// it. g comes from the real surface (lakeNormal: waves, bent crests, gusts, fine ripples) by finite
// differences, so the net is as irregular as the water; ripples finer than the blur of the sun's
// disc at that depth drop out (deeper = softer). The density is softened (lines glow instead of
// razor edges), eased with depth, and red / blue land a hair apart (the prism fringe).
// Sampled where the refracted sun ray came in. `soft` = extra blur (volumetric beams). Mean ~1.
vec2 lakeSlope(vec2 p, float t, float fp) { vec3 n = lakeNormal(p, t, fp); return -n.xz / n.y; }
vec3 waterCausticsRGB(vec3 P, float depth, float t, float soft) {
    if (depth <= 0.2) return vec3(1.0);
    vec3 sw = sunDirWater();
    vec2 p = P.xz - sw.xz / max(sw.y, 0.25) * depth;   // where its light came through the surface
    float fp = 0.25 + depth * 0.012 + soft;             // the sun's disc blurs fine ripples with depth
    const float e = 0.2;
    vec2 g0 = lakeSlope(p, t, fp);
    mat2 dg = mat2((lakeSlope(p + vec2(e, 0.0), t, fp) - g0) / e, (lakeSlope(p + vec2(0.0, e), t, fp) - g0) / e);
    // x3 over the physical 0.248: our surface is smoother than real water, whose capillary ripples
    // (far below a voxel) do most of the focusing.
    float D = depth * 0.248 * 3.0;
    vec3 c;
    for (int k = 0; k < 3; k++) {   // R, G, B: slightly different refraction
        float det = determinant(mat2(1.0) + (D * (1.0 + 0.03 * float(k - 1))) * dg);
        c[k] = 1.3 / (abs(det) + 0.3);   // soft floor; flat water (det 1) = 1
    }
    float sharp = exp(-depth / 70.0) * smoothstep(0.2, 3.0, depth) * 0.85;
    return mix(vec3(1.0), clamp(c, 0.3, 3.0), sharp);
}
float waterCaustics(vec3 P, float depth, float t) { return waterCausticsRGB(P, depth, t, 0.0).g; }
// setWaterLight + caustics on the sun (strength k = r.water.caustics).
void setWaterLightAt(float depth, vec3 P, float t, float k) {
    setWaterLight(depth);
    if (depth > 0.0 && k > 0.0) gSunTrans *= max(mix(vec3(1.0), waterCausticsRGB(P, depth, t, 0.0), k), vec3(0.0));
}
// Camera under water with the froxel fog: the froxels carry the water's in-scatter and a grey
// extinction (waterExt) along the straight view ray for `len` voxels, while gbuffer.comp already
// put the exact per-channel absorption into the albedo; this undoes the grey one (x base).
float waterFroxelFix(float len) { return exp(min(waterExt() * max(len, 0.0), 9.0)); }
#endif

// --- Emissive light animation: the GLSL copy of lightAnim (src/lights.cpp, same hash, so a
// torch's glow and the light it casts flicker together). kind = props.w of an emissive
// material: 0 steady, 1 flicker (fire), 2 pulse; integrity < 0.9 = a damaged glowing body
// (x integrity^2 with random dropouts), < 0.25 dead.
uint lightHash(uint x) {
    x ^= x >> 16; x *= 0x7feb352du;
    x ^= x >> 15; x *= 0x846ca68bu;
    x ^= x >> 16;
    return x;
}
float lightNoise(float t, uint ph) {
    float i = floor(t), f = t - i;
    f = f * f * (3.0 - 2.0 * f);
    uint a = lightHash(uint(int(i)) ^ ph), b = lightHash(uint(int(i) + 1) ^ ph);
    return mix(float(a & 0xFFFFu) / 65535.0, float(b & 0xFFFFu) / 65535.0, f);
}
float lightAnim(float kind, uint ph, float t, float integrity) {
    float a = 1.0;
    if (kind > 0.5 && kind < 1.5)
        a = 0.55 + 0.45 * (0.6 * lightNoise(t * 8.0, ph) + 0.4 * lightNoise(t * 19.0, ph * 7u + 3u));
    else if (kind > 2.5)
        a = 1.0 + 0.28 * (0.5 * lightNoise(t * 0.4, ph) + 0.3 * lightNoise(t * 1.3, ph * 5u + 1u) +
                          0.2 * lightNoise(t * 2.6, ph * 7u + 3u) - 0.5);
    else if (kind > 1.5)
        a = 0.55 + 0.45 * (0.5 + 0.5 * sin(t * 2.2 + float(ph & 1023u) * 0.00614));
    if (integrity < 0.9) {
        if (integrity < 0.25) return 0.0;
        float d = (0.9 - integrity) / 0.65;
        float n = lightNoise(t * (5.0 + 16.0 * d), ph ^ 0x9e37u);
        a *= (n < 0.2 + 0.45 * d ? 0.1 : 1.0) * integrity * integrity;
    }
    return a;
}
uint lightPhaseAt(vec3 p) {
    ivec3 c = ivec3(floor(p / 16.0));
    return lightHash(uint(c.x) * 73856093u ^ uint(c.y) * 19349663u ^ uint(c.z) * 83492791u);
}
uint lightPhaseBody(int oi) { return lightHash(uint(oi) + 0x51edu); }

// --- Emissive light sources (LightRegistry, src/lights.cpp; r.lights) -----------------
// Up to 32 boxes around emissive voxels near the camera: (centre, half diagonal),
// (power rgb, owner body or -1), (half extents xyz, -).
layout(SET(2) binding = 1, std140) uniform _LightsU {
    ivec4 lightCount; vec4 lightA[32]; vec4 lightB[32]; vec4 lightC[32];
    ivec4 torchCount; vec4 torchA[64]; vec4 torchB[64];   // torches (lights.h GpuTorch)
};
// Is point q of light i visible from P along L (distance d)? Hits on the emitter itself
// count as lit: its own body or an emissive world voxel.
bool lightVisible(int i, vec3 P, vec3 L, float d) {
    Hit h = traceSceneLamp(P, L, d, kSkipMat);
    if (!h.hit || h.t > d - 0.05) return true;
    float owner = lightB[i].w;
    if (owner >= 0.0) return h.obj == int(owner);
    return h.obj < 0 && materials[h.mat].albedoEmissive.a >= 0.5;
}
// Irradiance-over-pi from light i at P (radiance units, added like `indirect`), without
// visibility: power x cos / (pi d^2), d clamped to the light's radius.
vec3 lightTerm(int i, vec3 P, vec3 N, vec3 q) {
    vec3 L = q - P;
    float d2 = max(dot(L, L), lightA[i].w * lightA[i].w);
    float c = max(dot(N, normalize(L)), 0.0);
    return lightB[i].rgb * (c / (3.14159265 * d2));
}
#ifdef LIGHTS_NEE
// Below this unshadowed contribution (radiance units) a light is skipped untraced: the
// long shadow rays to far lamps cost more than everything else and add nothing visible.
const float kLightMin = 0.004;
// ...and below this fraction of the light the pixel already gets (sun + sky, set per pixel by the
// shading pass in gLightFloor): a lamp adding under ~2 % to a sunlit / sky-lit surface is
// invisible, but its shadow ray cost as much as anywhere. Caves and nights (no sun, no sky)
// keep every lamp. (Relative NEE culling, like an exposure-aware light cut-off.)
const float kLightRel = 0.02;
float gLightFloor = 0.0;
// The sun + sky light at a surface (the units of lightTerm / pickLight's totals): the lamp floor.
float lightFloorAt(vec3 P, vec3 N, float sunNdlShadow) {
    const vec3 lum = vec3(0.2126, 0.7152, 0.0722);
    float sky = dot(skyHorizon, lum) * skyScale * giScale * skyOpen(P + N * 1.5);
    return kLightRel * (dot(sunRGB, lum) * sunNdlShadow + sky);
}
// Deterministic (Standard lighting: no denoiser, must not flicker): every light that
// matters here, one shadow ray to its centre.
// Visibility of light i from surface point P (0..1). r.lights.shadows hard: one ray to its
// centre; soft (LIGHTS_SOFT): 4 fixed points spread over its box, averaged — a stable,
// noise-free penumbra at 4x the rays.
// Light i seen from P through its recorded cube (lightC[i].w = its slot): the point, pushed off
// its surface by about a texel's footprint (normal offset), is lit if it isn't farther from the
// lamp than the first occluder recorded in its direction. 2x2 comparisons blended (percentage-
// closer filtering): soft, stable edges; soft lamp shadows widen the footprint.
float lampCacheVis(int i, vec3 P, vec3 N) {
    const float R = float(kLampCubeRes);
    int slot = int(lightC[i].w);
    vec3 c = lightA[i].xyz;
    float d0 = length(P - c);
    vec3 v = P + N * (0.2 + 1.2 * d0 / R) - c;
    float d = length(v);
    if (d < 0.5) return 1.0;
    vec2 st;
    int f = lampCubeFace(v, st);
#ifdef LIGHTS_SOFT
    const float spread = 2.0;
#else
    const float spread = 1.0;
#endif
    vec2 x = ((st * 0.5 + 0.5) * R - 0.5) / spread;
    ivec2 b = ivec2(floor(x));
    vec2 w = x - vec2(b);
    float s[4];
    for (int k = 0; k < 4; k++) {
        ivec2 t = clamp(ivec2(round(vec2(b + ivec2(k & 1, k >> 1)) * spread)), ivec2(0), ivec2(kLampCubeRes - 1));
        float D = texelFetch(lampCubeTex, ivec2(f * kLampCubeRes + t.x, slot * kLampCubeRes + t.y), 0).r;
        s[k] = d <= D + 0.3 ? 1.0 : 0.0;
    }
    return mix(mix(s[0], s[1], w.x), mix(s[2], s[3], w.x), w.y);
}
float lightVis(int i, vec3 P, vec3 N) {
    if (lightC[i].w >= 0.0) return lampCacheVis(i, P, N);   // Standard: the lamp shadow cache
    vec3 o = P + N * 0.03;
#ifdef LIGHTS_SOFT
    const vec3 pat[4] = vec3[](vec3(-0.8, -0.8, 0.4), vec3(0.8, -0.4, -0.8),
                               vec3(-0.4, 0.8, -0.8), vec3(0.4, 0.4, 0.8));
    float v = 0.0;
    for (int k = 0; k < 4; k++) {
        vec3 L = lightA[i].xyz + pat[k] * lightC[i].xyz - o;
        float d = length(L);
        if (lightVisible(i, o, L / d, d)) v += 0.25;
    }
    return v;
#else
    vec3 L = lightA[i].xyz - o;
    float d = length(L);
    return lightVisible(i, o, L / d, d) ? 1.0 : 0.0;
#endif
}
// minE: skip lights adding less than this (radiance): each kept one costs a shadow ray.
vec3 lightsDirect(vec3 P, vec3 N, float minE) {
    vec3 sum = vec3(0.0);
    for (int i = 0; i < lightCount.x; i++) {
        vec3 e = lightTerm(i, P, N, lightA[i].xyz);
        if (e.x + e.y + e.z < 3.0 * minE) continue;
        sum += e * lightVis(i, P, N);
    }
    return sum;
}
vec3 lightsDirect(vec3 P, vec3 N) { return lightsDirect(P, N, kLightMin); }
// Pick one light with probability proportional to its unshadowed contribution u(i)
// (no cosine for fog), in one pass (weighted reservoir, one random number rescaled at
// each step). Returns the index, the sum of u and u(pick); -1 when nothing matters.
int pickLight(vec3 P, vec3 N, bool fog, float x, float minTotal, out float total, out float ui) {
    total = 0.0; ui = 0.0;
    int pick = -1;
    for (int i = 0; i < lightCount.x; i++) {
        vec3 L = lightA[i].xyz - P;
        float d2 = max(dot(L, L), lightA[i].w * lightA[i].w);
        float c = fog ? 0.25 : clamp(dot(N, L) * inversesqrt(max(dot(L, L), 1e-6)) + lightA[i].w * inversesqrt(d2), 0.0, 1.0);
        vec3 p = lightB[i].rgb;
        float w = (p.r + p.g + p.b) * (1.0 / 3.0) * c / (3.14159265 * d2);
        if (w <= 0.0) continue;
        total += w;
        float q = w / total;
        if (x < q) { pick = i; ui = w; x /= q; }
        else x = (x - q) / (1.0 - q);
    }
    return total < minTotal ? -1 : pick;
}
// Next-event estimation (stochastic paths): one light picked by contribution, one random
// point in its sphere, one shadow ray; divided by its pick probability.
vec3 sampleLights(vec3 P, vec3 N, inout uint seed) {
    float total, ui;
    int i = pickLight(P, N, false, rnd(seed), max(kLightMin, gLightFloor), total, ui);
    if (i < 0) return vec3(0.0);
    vec3 q = lightA[i].xyz + (vec3(rnd(seed), rnd(seed), rnd(seed)) * 2.0 - 1.0) * lightC[i].xyz;
    vec3 e = lightTerm(i, P, N, q);
    vec3 o = P + N * 0.03;
    vec3 L = q - o;
    float d = length(L);
    return lightVisible(i, o, L / d, d) ? e * (total / ui) : vec3(0.0);
}
// Fog in-scatter at P (isotropic phase) from one light picked by contribution, shadowed;
// multiply by the density. Skipped where even ~30 voxels of this fog would add < kLightMin.
vec3 sampleLightsFog(vec3 P, float dens, inout uint seed) {
    float total, ui;
    int i = pickLight(P, vec3(0.0), true, rnd(seed), kLightMin / (30.0 * dens), total, ui);
    if (i < 0) return vec3(0.0);
    vec3 L = lightA[i].xyz - P;
    float d2 = max(dot(L, L), lightA[i].w * lightA[i].w);
    vec3 e = lightB[i].rgb * (1.0 / (4.0 * 3.14159265 * d2));
    float d = length(L);
    return lightVisible(i, P, L / d, d) ? e * (total / ui) : vec3(0.0);
}
#endif

// Foliage (gbNormal.a 0.8 = leaf voxel, 0.7 = grass blade): thin, passes light through.
// Set by the shading passes before shadeSurface: gFoliage = this pixel is foliage,
// gSss = strength (r.foliage.sss).
float gFoliage = 0.0, gSss = 0.0;
bool isFoliageFlag(float a) { return a > 0.62 && a < 0.88; }
bool isBladeFlag(float a) { return a > 0.62 && a < 0.75; }
// Where a sun-visibility ray for a foliage surface starts: a leaf VOXEL is lit from the
// far side through itself, so start from its centre stepped out toward the sun (a
// blade has no volume: just off its surface). Used when N.L <= 0 (no shadow ray then).
// Sun visibility along d from o, with LEAVES letting light through (r.foliage.light):
// a leaf voxel (palette kind 3, world) passes light with probability / fraction
// `leafLight`; the ray continues from where it leaves that voxel. Stochastic (SVGF /
// RR: pass-or-block at random, averaged by the temporal accumulation -> soft dappled
// light at no extra cost for rays that don't meet leaves) or deterministic (Standard:
// multiply). Up to 6 leaves; tFirst = distance to the first blocker (0 = none).
float sunVisibilityT(vec3 o, vec3 d, float leafLight, inout uint seed, bool stochastic, out float tFirst) {
    tFirst = 0.0;
    float T = 1.0;
    for (int i = 0; i < 6; i++) {
        Hit h = traceSceneShadow(o, d, 1e30, kSkipMat);
        if (!h.hit) return (i == 0 ? horizonSunVisibility(o, d) : T) * cloudShadow(o);   // endless: the far horizon's hills; clouds (both)
        vec3 p = o + d * h.t;
        if (i == 0) tFirst = h.t;
        if (h.obj >= 0 || materials[h.mat].props.z != 3.0 || leafLight <= 0.0) return 0.0;
        if (stochastic) { if (rnd(seed) >= leafLight) return 0.0; }
        else { T *= leafLight; if (T < 0.03) return 0.0; }
        // Leave the leaf voxel: its exit face along d.
        vec3 sd = mix(d, vec3(1e-6), lessThan(abs(d), vec3(1e-6)));
        vec3 tx = (vec3(h.cell) + step(vec3(0.0), sd) - p) / sd;
        o = p + d * (min(min(tx.x, tx.y), tx.z) + 1e-3);
    }
    return 0.0;
}
float sunVisibility(vec3 o, vec3 d, float leafLight, inout uint seed, bool stochastic) {
    float tf;
    return sunVisibilityT(o, d, leafLight, seed, stochastic, tf);
}

vec3 foliageSunOrigin(vec3 P, vec3 N, bool blade) {
    return blade ? P + sunDir * 0.05 : P - N * 0.5 + sunDir * 0.9;
}

// ---------------------------------------------------------------------------
// Surfaces (PBR): a diffuse lobe (albedo x (1 - metalness), less what the specular lobe takes) and a
// specular one (GGX, height-correlated Smith visibility, Schlick Fresnel; F0 = 0.04 for dielectrics, the
// base colour for metals). The shading passes set gRough / gMetal from gbSurf (gbuffer.comp) before
// shadeSurface. Units: sunRGB and `indirect` are irradiance / pi (Lambert = albedo x them).
float gRough = 0.9, gMetal = 0.0;
vec3 gIndW = vec3(0.0);   // set by shadeSurface: what it multiplies `indirect` by (raw_lamps' lamp weight)
void setSurface(vec4 surf) { gRough = clamp(surf.r, 0.02, 1.0); gMetal = surf.g; }
// Per-voxel facets (r.facets, gbuffer.comp): a voxel face's shading normal tilted by f along the face's own axes.
// gbSurf.ba = f * 0.5 + 0.5; 0.5 / 0.5 leaves N exactly as it is.
vec3 facetTilt(vec3 N, vec2 f) {
    if (dot(f, f) < 1e-4) return N;
    vec3 a = abs(N), T, B;
    if (a.y > 0.5) { T = vec3(1.0, 0.0, 0.0); B = vec3(0.0, 0.0, 1.0); }
    else if (a.x > 0.5) { T = vec3(0.0, 0.0, 1.0); B = vec3(0.0, 1.0, 0.0); }
    else { T = vec3(1.0, 0.0, 0.0); B = vec3(0.0, 1.0, 0.0); }
    return normalize(N + T * f.x + B * f.y);
}
vec3 facetNormal(vec3 N, vec4 surf) { return facetTilt(N, surf.ba * 2.0 - 1.0); }
// How far a material's voxels tilt (the tilt's tangent), by its surface kind (Material.surf[3], world.h wm::Pattern):
// grass / turf, leaves and bark most, stone and soil some, metal and plain colours none.
float facetAmount(uint kind) {
    const float k[12] = float[](0.0, 0.2, 0.35, 0.18, 0.12, 0.25, 0.2, 0.3, 0.08, 0.45, 0.0, 0.1);
    return kind < 12u ? k[kind] : 0.0;
}
vec3 surfaceF0(vec3 base, float metal) { return mix(vec3(0.04), base, metal); }
// The specular lobe integrated over the hemisphere for uniform incoming light (the split-sum's environment
// BRDF), analytic fit (Karis, "Physically Based Shading on Mobile", 2014): F0 x A + B.
vec3 envBRDF(vec3 F0, float NdotV, float rough) {
    const vec4 c0 = vec4(-1.0, -0.0275, -0.572, 0.022);
    const vec4 c1 = vec4(1.0, 0.0425, 1.04, -0.04);
    vec4 r = rough * c0 + c1;
    float a004 = min(r.x * r.x, exp2(-9.28 * NdotV)) * r.x + r.y;
    vec2 AB = vec2(-1.04, 1.04) * a004 + r.zw;
    return F0 * AB.x + AB.y;
}
// The specular lobe's share the reflection pass traces (water_refl.comp, r.reflect): glossy surfaces reflect
// the scene; rougher ones keep the environment BRDF on their indirect light only (shadeSurface).
// RR / ground truth (REFL_STOCHASTIC): one ray sampled over the GGX lobe per pixel and frame, denoised by RR with
// the specular guides - glossy up to roughness 0.5 (metals). Otherwise (SVGF / Standard / NRD: added after their
// denoising, so it must be noise-free) one mirror ray: only near-mirrors take most of it, the rest stays rough.
// Never leaves (0.55) or paint (0.45-0.5): a traced ray on every leaf pixel cost ~30 ms.
#if defined(REFL_GLOSSY) && defined(REFL_STOCHASTIC)
float glossyShare(float rough) { return 1.0 - smoothstep(0.3, 0.45, rough); }
#elif defined(REFL_GLOSSY)
float glossyShare(float rough) { return 1.0 - smoothstep(0.08, 0.4, rough); }
#else
float glossyShare(float rough) { return 0.0; }
#endif
// GGX specular BRDF x cos for light from L (radiance per unit irradiance / pi: x pi x sunRGB). alpha clamped:
// a very smooth flat voxel face would turn the sun into one blinding face-sized flash.
vec3 ggxSpec(vec3 N, vec3 V, vec3 L, float rough, vec3 F0) {
    float NdotL = dot(N, L), NdotV = dot(N, V);
    if (NdotL <= 0.0 || NdotV <= 0.0) return vec3(0.0);
    vec3 H = normalize(V + L);
    float NdotH = max(dot(N, H), 0.0), VdotH = max(dot(V, H), 0.0);
    float a = max(rough * rough, 0.04), a2 = a * a;
    float d = NdotH * NdotH * (a2 - 1.0) + 1.0;
    float D = a2 / (3.14159265 * d * d);
    float gv = NdotL * sqrt(NdotV * NdotV * (1.0 - a2) + a2), gl = NdotV * sqrt(NdotL * NdotL * (1.0 - a2) + a2);
    float Vis = 0.5 / max(gv + gl, 1e-6);
    vec3 F = F0 + (1.0 - F0) * pow(1.0 - VdotH, 5.0);
    return D * Vis * F * NdotL * 3.14159265;
}

vec3 shadeSurface(vec3 P, vec3 N, vec3 rd, vec4 alb, bool water, float ndl, float shadow,
                  vec3 indirect, float contact) {
    float emissive = alb.a * 8.0;
#ifdef WATER_CLEAR
    vec3 sunLight = sunRGB * ndl * shadow * gSunTrans;
    indirect *= gAmbTrans;
#else
    vec3 sunLight = sunRGB * ndl * shadow;
#endif
    // Diffuse x (1 - the specular lobe's share) + the specular lobe: the sun's highlight (GGX) and the indirect
    // light (sky, bounce, lamps) through the environment BRDF - a rough reflection of what lights the surface.
    // (Glossy surfaces' sharp reflections: water_refl.comp, added like the water's.)
    vec3 V = -rd;
    float NdotV = max(dot(N, V), 1e-3);
    vec3 F0 = surfaceF0(alb.rgb, gMetal);
    vec3 spec = envBRDF(F0, NdotV, gRough);
    vec3 diff = alb.rgb * (1.0 - gMetal) * (1.0 - spec);
    gIndW = diff + spec * (1.0 - (water ? 0.0 : glossyShare(gRough)));   // (the glossy part: water_refl.comp)
    vec3 sunSpec = ggxSpec(N, V, sunDir, gRough, F0) * (sunLight / max(ndl, 1e-4));
    vec3 base = diff * sunLight + gIndW * (indirect * giScale * contact) + sunSpec + alb.rgb * emissive;
#ifdef FOLIAGE_SSS
    if (gFoliage > 0.5) {
        // Thin-leaf transmission: sunlight that passed through the leaf, strongest when
        // the sun is behind it and brightest looking toward the sun (forward scattering),
        // tinted by the leaf twice (in and out: albedo^1.5 -> yellow-green glow). A soft
        // wrap on the lit side too (thin tissue scatters light around its edges).
        float nl = dot(N, sunDir);
        float back = max(-nl, 0.0), fwd = pow(max(dot(rd, sunDir), 0.0), 6.0);
        vec3 tint = pow(max(alb.rgb, vec3(0.0)), vec3(1.5)) * 1.6;
  #ifdef WATER_CLEAR
        vec3 sunT = sunRGB * gSunTrans;
  #else
        vec3 sunT = sunRGB;
  #endif
        base += tint * sunT * shadow * (0.35 * back + 0.6 * fwd) * gSss;
        base += alb.rgb * sunT * shadow * max(0.25 - abs(nl), 0.0) * 0.8 * gSss;   // wrap at grazing
    }
#endif
    if (water) {
        vec3 V = -rd;
        // SEMI-TRANSPARENT: march down through the water to the bottom, blend by
        // depth (shallow shows the bottom, deep fades to the water tint).
        const vec3 wtint = vec3(0.06, 0.20, 0.28);
        vec3 body = wtint;
        for (float d = 1.0; d < 40.0; d += 1.0) {
            uint m = voxelMatAt(P + rd * d);
            if (m == 5u) continue;                       // still inside the water
            if (m != 0u) {                                // hit the bottom
                vec3 ba = materials[m].albedoEmissive.rgb * (0.35 + 0.55 * max(sunDir.y, 0.0));
                body = mix(wtint, ba, exp(-d * 0.10));
            }
            break;
        }
        vec3 skyR = skyColor(reflect(rd, N)) * skyOpen(P + vec3(0.0, 1.5, 0.0));   // (a pool in a cave)
        float fres = 0.02 + 0.98 * pow(1.0 - max(dot(N, V), 0.0), 5.0);
        vec3 Hh = normalize(V + sunDir);
        vec3 glint = vec3(1.0, 0.96, 0.86) * pow(max(dot(N, Hh), 0.0), 220.0) * shadow * 3.0;
        base = mix(body, skyR, fres) + glint;
    }
    return base;
}

// A GI ray that left the loaded world without a hit: out the top = the open sky; out the bottom = none;
// out a side = the far horizon's terrain decides (below it = underground). (Rays are traced to the
// world's edge - no reach limit, no guess: a sealed room is black, an opening gives what its solid
// angle gives.)
// GI rays reach GI_REACH voxels (r.gi.reach). One that flies that far without a hit is judged by where it
// is heading: sky only if its end AND two points further along (16, 32 voxels) are all unroofed (skyOpen:
// the column's sky-blocking top) - a ray ending under a roof opening but heading sideways into the room,
// or ending in a cave / hall, gets none; one ending in open air outdoors gets the sky. (Unlimited reach
// was +6..12 ms - grazing rays walk the ground brick by brick - and +4 ms even for only rising rays.)
#ifndef GI_REACH
#define GI_REACH 48.0
#endif
// (reach: the segment's length; rising rays that stop under a roof are traced on first - giPath.)
float giEscape(vec3 o, vec3 d, float reach) {
    vec3 lo = vec3(worldMinB() * 8), hi = vec3((worldMinB() + worldBricks) * 8);
    vec3 sd = mix(d, vec3(1e-6), lessThan(abs(d), vec3(1e-6)));
    vec3 t1 = (lo - o) / sd, t2 = (hi - o) / sd;
    vec3 tf = max(t1, t2);
    float tExit = max(min(min(tf.x, tf.y), tf.z), 0.0);
    if (tExit > reach) {   // ran its reach inside the world
        vec3 e = o + d * reach;
        float s = skyOpen(e);
        // (a ray ending under a roof opening may be heading into the room: look further along it)
        if (s > 0.0) s *= skyOpen(e + d * 16.0) * skyOpen(e + d * 32.0);
        return s;
    }
    vec3 f = o + d * tExit;
    if (f.y <= lo.y + 1.0) return 0.0;
    if (f.y >= hi.y - 1.0) return 1.0;
    return skyOpen(f + d * 2.0);
}

#ifdef GI_CACHE
// --- Radiance cache for the second bounce (r.gi.cache; NVIDIA SHaRC-style spatial hash) -----------
// A cell = a patch of surface: a cube of 2^k voxels (k grows with the distance to the camera, so cells
// cover about the same screen area near and far) on one face direction. It holds the mean radiance
// arriving at that patch over the cosine hemisphere = the expected value of what giPath's bounce 2 (+ its
// sun ray) returns there. Filled only by rays traced from that patch: it can't leak through walls.
// giPath reads a warm cell instead of tracing bounce 2; a cold one is traced exactly (and its sample
// kept). Each frame every cell used asks for one more ray (the queue): gi_cache_update.comp traces them
// next frame, gi_cache_resolve.comp folds the frame's samples into the cell (capped running mean).
const uint kGiCacheSize = 1u << 20;      // entries of 64 bytes (= LitRenderer::kGiCacheSize)
const uint kGiCacheQueueMax = 1u << 17;  // requests per queue (= LitRenderer::kGiCacheQueueMax)
struct GiCacheEntry {
    uint check;        // key checksum (0 = empty)
    uint lastUsed;     // frame of the last lookup (eviction)
    uint lastQueued;   // frame it last asked for an update ray
    uint accN;         // samples added this frame (only the first 64 are summed)
    uvec4 acc;         // their sum (rgb, 16.16 fixed point; w unused)
    vec4 val;          // resolved radiance, w = samples it holds (capped)
    vec4 pos;          // a point of the patch (edits reset the patches near them: gi_cache_resolve.comp)
};
layout(std430, SET(1) binding = 8) buffer GiCache { GiCacheEntry giCache[]; };
struct GiCacheReq { vec4 p; vec4 n; };   // p.w = the entry (uint bits)
layout(std430, SET(1) binding = 9) buffer GiCacheQueue {
    uint gcArgs[4];    // the update pass's indirect dispatch (x, y, z, -)
    uint gcCount[2];   // requests per queue (frame & 1)
    uint gcLive[3];    // live patches, per frame % 3 (gi_cache_resolve.comp; r.gi.cache.stats)
    uint gcPad[3];
    GiCacheReq gcReq[];
};
// -> (table slot hash, checksum != 0) of the cell holding surface point p with normal n.
uvec2 giCacheKey(vec3 p, vec3 n) {
    float d = distance(p, giCacheCam);
    int lvl = clamp(int(floor(log2(max(d * giCacheLod, 1e-6)))), giCacheP.z, giCacheP.z + 10);
    ivec3 c = ivec3(floor(p * exp2(-float(lvl))));
    vec3 a = abs(n);
    uint face = a.x >= a.y && a.x >= a.z ? (n.x > 0.0 ? 0u : 1u)
              : a.y >= a.z ? (n.y > 0.0 ? 2u : 3u) : (n.z > 0.0 ? 4u : 5u);
    uint m = uint(lvl + 16) * 8u + face;
    uint h1 = pcgHash(uint(c.x) + pcgHash(uint(c.y) + pcgHash(uint(c.z) + pcgHash(m))));
    uint h2 = pcgHash((m ^ 0x9e3779b9u) + pcgHash(uint(c.z) + pcgHash((uint(c.y) ^ 0x85ebca6bu) +
              pcgHash(uint(c.x) * 0x2545f491u))));
    return uvec2(h1, h2 | 1u);
}
// The cell's slot (inserted if new), -1 if its probe run is full (then traced exactly).
int giCacheFind(uvec2 k) {
    for (uint i = 0u; i < 8u; ++i) {
        uint j = (k.x + i) & (kGiCacheSize - 1u);
        uint c = giCache[j].check;
        if (c == k.y) return int(j);
        if (c == 0u) {
            uint prev = atomicCompSwap(giCache[j].check, 0u, k.y);
            if (prev == 0u || prev == k.y) return int(j);
        }
    }
    return -1;
}
// true: val = the warm cell's radiance. Else idx = the slot the caller's traced sample goes to (-1 none).
bool giCacheLookup(vec3 p, vec3 n, out vec3 val, out int idx) {
    val = vec3(0.0);
    idx = giCacheFind(giCacheKey(p, n));
    if (idx < 0) return false;
    uint f = uint(giCacheP.x);
    giCache[idx].lastUsed = f;
    if (atomicExchange(giCache[idx].lastQueued, f) != f) {   // its one update ray for next frame
        giCache[idx].pos = vec4(p, 0.0);
        uint q = atomicAdd(gcCount[f & 1u], 1u);
        if (q < uint(giCacheP.w))
            gcReq[(f & 1u) * kGiCacheQueueMax + q] = GiCacheReq(vec4(p, uintBitsToFloat(uint(idx))), vec4(n, 0.0));
    }
    vec4 v = giCache[idx].val;
    if (v.w >= float(giCacheP.y)) { val = v.rgb; return true; }
    return false;
}
void giCacheAdd(int idx, vec3 L) {
    if (idx < 0 || any(isnan(L)) || atomicAdd(giCache[idx].accN, 1u) >= 64u) return;
    uvec3 q = uvec3(clamp(L, vec3(0.0), vec3(256.0)) * 65536.0 + 0.5);
    atomicAdd(giCache[idx].acc.x, q.x);
    atomicAdd(giCache[idx].acc.y, q.y);
    atomicAdd(giCache[idx].acc.z, q.z);
}
#endif

// Indirect light along one cosine-sampled direction (the ray-traced modes' GI; RR's giBounceL and SVGF's
// gi.comp): a short path, physically consistent -
//   ray 1 from P: a miss = the sky (if it really escaped: giEscape), a hit = its emission + its direct sun
//   (a shadow ray) and, with r.gi.bounces 2 (GI_DOUBLE), + giScale x the light ray 2 brings it (cosine
//   around the hit's normal: sky, or that hit's emission + direct sun). L = e1 + a1 (s1 + giScale L2).
// No distance fade (radiance is constant along a ray) and no reach limit (rays run to the world's edge;
// the distance field skips open air): the light through an opening is what its solid angle gives, and a
// sealed room is black. All rays - bounce 1, its sun, bounce 2, its sun - go through ONE traversal call
// site (a loop whose bound isn't a compile-time constant, so it isn't unrolled back into four copies:
// inlined traversals cost the shading pass occupancy). Lamps light only the first surface (NEE).
// With GI_CACHE, L2 comes from the radiance cache where its cell is warm (see above).
// (lastStage 1 / traceOn false: one bounce, no trace-on past the reach = exactly bounce 2's own rays -
// gi_cache_update.comp.)
#ifdef GI_HITDIST
// NRD (raw_shade NRD_SPLIT): the first bounce ray's hit distance (voxels; 1e4 = escaped) - REBLUR's hitT,
// which sizes its spatial filters.
float gGiHitT = 1e4;
#endif
vec3 giPathImpl(vec3 P, vec3 N, vec3 giDir, inout uint seed, const int lastStage, const bool traceOn) {
    vec3 L = vec3(0.0), thr = vec3(1.0);
    vec3 ro = P + N * 3e-2, rdir = giDir;
    vec3 hitP = vec3(0.0), hitN = vec3(0.0, 1.0, 0.0), hitAlb = vec3(0.0);
    float hitNdl = 0.0;
    int stage = 0;   // 0 bounce 1, 1 its sun, 2 bounce 2, 3 its sun
    float reach = GI_REACH;
    bool extended = false;   // this bounce ray was traced on past GI_REACH (rising, stopped under a roof)
#ifdef GI_CACHE
    bool hitBody = false;                     // the bounce-1 hit is a body (moves: never cached)
    bool cold = false;                        // bounce 2 traced for a cold cell: L = its own radiance
    vec3 preL = vec3(0.0), preThr = vec3(0.0);
    int cIdx = -1;
#endif
    for (int it = 0; it < 6 + min(numObjects, 0); ++it) {
        bool sunRay = (stage & 1) == 1;
        Hit h = traceWorldGiA(ro, rdir, sunRay ? 1e30 : reach, 0u, sunRay);   // (sun rays: any hit)
#ifdef GI_OBJECTS
        if (!(sunRay && h.hit)) {   // bodies (one call site): nearest within 16 for a bounce, any for the sun
            if (!sunRay) h.obj = -1;
            traceObjectsA(ro, rdir, h, sunRay ? 1e30 : min(h.hit ? h.t : reach, 16.0), sunRay);
        }
#endif
        if (sunRay) {
            if (!h.hit) L += thr * hitAlb * sunRGB * hitNdl;
            if (stage >= lastStage) break;
            // bounce 2 from the first hit
            thr *= hitAlb * giScale;
#ifdef GI_CACHE
            {
                vec3 cv;
                if (!hitBody && giCacheLookup(hitP, hitN, cv, cIdx)) { L += thr * cv; break; }
                cold = true; preL = L; preThr = thr; L = vec3(0.0); thr = vec3(1.0);
            }
#endif
            rdir = cosineHemisphere(hitN, seed);
            ro = hitP + hitN * 3e-2;
            stage = 2;
            reach = GI_REACH;
            extended = false;
            continue;
        }
#ifdef WATER_CLEAR
        // Bounce rays treat water as opaque (the cheap traversal) and a ray that meets water returns the
        // light from above it - right from the lake bed (the shading pass attenuates it).
        if (h.hit && h.mat == 5u) h.hit = false;
#endif
        if (!h.hit) {   // escaped: the sky (its sun disk is the direct term: no double count)
            float esc = giEscape(ro, rdir, reach);
            if (traceOn && esc < 0.999 && !extended && stage == 0 && rdir.y > 0.3) {
                // A rising ray that stopped under a roof (a room toward its window / roof opening): trace it on
                // (it hits the ceiling soon or leaves through the opening) - if any point further along could be
                // open sky; in a cave / deep inside, none is, and it stays dark untraced. Outdoors never.
                vec3 e = ro + rdir * reach;
                if (max(skyOpen(e + rdir * 32.0), max(skyOpen(e + rdir * 64.0), skyOpen(e + rdir * 96.0))) > 0.0) {
                    ro = e;
                    reach = 96.0;
                    extended = true;
                    continue;
                }
            }
            L += thr * min(skyColor(rdir, 0.0), vec3(3.0)) * esc;
            break;
        }
        Material m = materials[h.mat];
        vec3 alb = m.albedoEmissive.rgb;
        float emit = m.albedoEmissive.a * 8.0;
#ifdef LIGHTS_NEE
        if (m.albedoEmissive.a >= 0.5) emit = 0.0;   // lamps: counted by sampleLights (the first surface)
#endif
        L += thr * alb * emit;
        hitP = ro + rdir * h.t;
        hitN = h.normal;
#ifdef GI_HITDIST
        if (stage == 0) gGiHitT = (extended ? GI_REACH : 0.0) + h.t;
#endif
#if defined(GI_CACHE) && defined(GI_OBJECTS)
        if (stage == 0) hitBody = h.obj >= 0;
#endif
        reach = GI_REACH;
        extended = false;
        hitAlb = alb;
        hitNdl = max(dot(hitN, sunDir), 0.0);
        if (hitNdl > 0.0) {   // its direct sun next
            ro = hitP + hitN * 3e-2;
            rdir = sunDir;
            stage += 1;
        } else {   // facing away from the sun: straight to bounce 2 (or done)
            if (stage + 1 >= lastStage) break;
            thr *= hitAlb * giScale;
#ifdef GI_CACHE
            {
                vec3 cv;
                if (!hitBody && giCacheLookup(hitP, hitN, cv, cIdx)) { L += thr * cv; break; }
                cold = true; preL = L; preThr = thr; L = vec3(0.0); thr = vec3(1.0);
            }
#endif
            rdir = cosineHemisphere(hitN, seed);
            ro = hitP + hitN * 3e-2;
            stage = 2;
        }
    }
#ifdef GI_CACHE
    if (cold) {   // bounce 2 was traced: its sample to the cell, then weighted in
        giCacheAdd(cIdx, L);
        L = preL + preThr * L;
    }
#endif
    return L;
}
vec3 giPath(vec3 P, vec3 N, vec3 giDir, inout uint seed) {
#ifdef GI_DOUBLE
    return giPathImpl(P, N, giDir, seed, 3, true);
#else
    return giPathImpl(P, N, giDir, seed, 1, true);
#endif
}
// lamps = false: without the lamps' next-event estimate (raw_shade: raw_lamps.comp adds it in its
// own pass - the lamp ray's traversal code in the big shading pass cost ~3 ms of occupancy even
// where no lamp ray was traced). A constant false drops that code from the caller.
vec3 giBounceL(vec3 P, vec3 N, inout uint seed, const bool lamps) {
    vec3 giDir = cosineHemisphere(N, seed);
    vec3 ind = giPath(P, N, giDir, seed);
#ifdef LIGHTS_NEE
    if (lamps) ind += sampleLights(P, N, seed);   // listed emitters, sampled directly
#endif
    return ind;
}
vec3 giBounce(vec3 P, vec3 N, inout uint seed) { return giBounceL(P, N, seed, true); }

// ---------------------------------------------------------------------------
// Standard lighting: cached sun shadow map (sunshadow.comp writes, these read).
// Light space: x = dot(P, shRight), y = dot(P, shUp), z = dot(P, sunDir) (height
// toward the sun). A texel stores the z of the topmost world occluder on its
// sun-parallel line; P is lit if nothing above it: z(P) >= stored - bias.
// Cascade c covers the window [win, win + shN) texels of size shTexel_c
// (4 cascades, x3 apart; windows centred on the camera, snapped to 64 texels).
float shTexel(int c) { return c == 0 ? shTexel0 : (c == 1 ? shTexel1 : (c == 2 ? shTexel2 : shTexel3)); }
ivec2 shWin(int c) { return c == 0 ? shWinA.xy : (c == 1 ? shWinA.zw : (c == 2 ? shWinB.xy : shWinB.zw)); }
bool shInWindow(int c, vec2 l, int margin) {
    ivec2 u = ivec2(floor(l / shTexel(c))) - shWin(c);
    return all(greaterThanEqual(u, ivec2(margin))) && all(lessThan(u, ivec2(shN - margin)));
}
// Smallest cascade whose window holds light-space point l with `margin` texels to
// spare; -1 if none.
int shCascadeFor(vec2 l, int margin) {
    for (int c = 0; c < 4; c++)
        if (shInWindow(c, l, margin)) return c;
    return -1;
}
// Hand-over radius of cascade c around the camera (a sphere well inside its square
// window, which is re-centred in 64-texel steps).
float shRadius(int c) { return shTexel(c) * (float(shN) * 0.5 - 120.0); }   // 64 snap + 48 margin + slack
float shFetch(int c, ivec2 uAbs) {
    ivec2 t = ((uAbs % shN) + shN) % shN;
    return texelFetch(shadowMapTex, t + ivec2(0, c * shN), 0).r;
}
float shFetchAux(int c, ivec2 uAbs) {
    ivec2 t = ((uAbs % shN) + shN) % shN;
    return texelFetch(shadowAuxTex, t + ivec2(0, c * shN), 0).r;
}
// Hard lookup (fog steps, probe ray hits): 1 = sunlit. Outside all cascades: lit.
float sunShadowHard(vec3 P) {
    vec2 l = vec2(dot(P, shRight), dot(P, shUp));
    int c = shCascadeFor(l, 1);
    if (c < 0) return cloudShadow(P);
    float occ = shFetch(c, ivec2(floor(l / shTexel(c))));
    return dot(P, sunDir) >= occ - 2.0 * shTexel(c) - 0.05 ? cloudShadow(P) : 0.0;
}
