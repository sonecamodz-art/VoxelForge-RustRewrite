// Procedural grass blade layer (primary rays; included by gbuffer.comp only).
//
// Blades are traced as their OWN layer along the camera ray, independent of which
// terrain face the ray ends on, so they overhang step edges and occlude terrain
// correctly (the old shell only existed above the hit top face, which chopped
// blades at every step). Data: grassTopTex = per world column, bits 0-11 the height
// of the grass surface (0 = none), bits 12-15 the grass variant (0 lush, 1 dry,
// 2 moss, 3 flowers) + grassTileTex = per-8x8 (min, max) height (VkGpuWorld, grasstop.comp).
//
// Blades are columns of 1/4-voxel cubes following a bent centre line. r.grass.style:
// STEPPED snaps each cube to the sub-voxel grid (moves in quarter-voxel jumps), SMOOTH
// lets the same cubes glide continuously. Both bend with the simulated wind
// (grass_wind.comp).
//
// PLACEMENT (no lattice): every world column (1x1 voxel cell) holds 0..GRASS_BLADES
// blades at hashed sub-voxel positions. How many, and how tall, comes from
//  - CLUMPS (Ghost of Tsushima, GDC 2021): a jittered cellular field of ~3.5-voxel
//    clumps; each clump has its own height, dryness and lean; blades thin out and
//    shorten toward the clump edge and lean away from its centre -> tufts;
//  - a DENSITY field (~25-voxel noise): lush patches, thin areas and bare spots;
//  - a large-scale HEIGHT field (~90 voxels, once per tile): meadows vs short turf;
//  - the VARIANT of the surface material (dry grass is short and sparse, moss is a
//    dense 1-2 level carpet).
// Bends stay inside the column (no clipping at cell borders). The hit reports the
// same blade point at `prevTime` / last frame's bend, so motion vectors include the wind.

layout(SET(0) binding = 11) uniform usampler2D grassTopTex;   // R16UI per column
layout(SET(0) binding = 12) uniform usampler2D grassTileTex;  // RG16UI per 8x8 (min, max)
layout(SET(0) binding = 13) uniform usampler2D grassInfoTex;  // RGBA8UI baked placement (grasstop.comp)

#ifndef GRASS_BLADES
#define GRASS_BLADES 3                                 // max blades per column
#endif
const float kGSub = 0.25;                              // blade voxel size (1/4 world voxel)
const int   kGCellSubs = 4;                            // sub-voxels per column side
const int   kGMaxLevels = 26;                         // tall meadow grass (variant 4): 65 cm
const float kGMaxH = float(kGMaxLevels) * kGSub;      // tallest blade (voxels)
const float kGMaxHShort = 9.0 * kGSub;                 // every other variant (<= 22 cm)
const uint  kGTileTall = 0x8000u;                      // grassTile.y flag: the tile has tall grass

float gHash(vec2 p, float s) {
    p = fract(p * vec2(0.1031, 0.1030) + s * 0.0973);
    p += dot(p, p.yx + 33.33);
    return fract((p.x + p.y) * p.x);
}

// Smooth value noise (density field).
float gNoise(vec2 p) {
    vec2 i = floor(p), f = fract(p);
    f = f * f * (3.0 - 2.0 * f);
    float a = gHash(i, 11.0), b = gHash(i + vec2(1, 0), 11.0);
    float c = gHash(i + vec2(0, 1), 11.0), d = gHash(i + vec2(1, 1), 11.0);
    return mix(mix(a, b, f.x), mix(c, d, f.x), f.y);
}

// Per variant: density scale, height range (levels), base + tip colour.
struct GVariant { float dens; float hMin, hMax; vec3 base, tip; };
GVariant gVariant(uint v) {
    if (v == 1u) return GVariant(0.90, 2.0, 7.0, vec3(0.26, 0.26, 0.09), vec3(0.70, 0.66, 0.28));  // dry
    if (v == 2u) return GVariant(1.00, 1.0, 2.5, vec3(0.10, 0.19, 0.07), vec3(0.25, 0.41, 0.14));  // moss
    if (v == 3u) return GVariant(1.00, 3.0, 8.0, vec3(0.17, 0.33, 0.11), vec3(0.42, 0.66, 0.25));  // flowers
    if (v == 4u) return GVariant(0.95, 12.0, 26.0, vec3(0.19, 0.31, 0.10), vec3(0.66, 0.64, 0.33)); // tall: straw seed heads
    if (v == 5u) return GVariant(0.80, 16.0, 26.0, vec3(0.16, 0.26, 0.09), vec3(0.52, 0.45, 0.24)); // reeds: tall, brown tips
    if (v == 6u) return GVariant(1.05, 3.0, 10.0, vec3(0.10, 0.22, 0.09), vec3(0.30, 0.50, 0.22));  // marsh grass: dark, lush
    return GVariant(1.05, 3.0, 9.0, vec3(0.17, 0.33, 0.11), vec3(0.46, 0.70, 0.27));              // lush
}

// Density field (0 = bare .. 1 = dense), shared with the far prefilter in gbuffer.comp.
float gDensity(vec2 xz) {
    return smoothstep(0.2, 0.68, gNoise(xz * 0.05 + 17.0));
}

// One world column's grass: ground height, variant, and its clump.
struct GCell {
    float gy; uint variant; int n;
    float clumpH;    // clump height scale
    float edge;      // 0 at the clump centre .. 1 at its edge
    vec2 away;       // unit direction from the clump centre (blades lean this way)
    float dry;       // clump dryness (colour)
};

// A blade = a column of kGSub cubes, one per level, placed where the bent centre
// line passes at that height (snapped to the sub-voxel grid), inside its column.
struct Blade {
    ivec2 baseSub;   // base column (sub-voxel coords)
    int levels;      // blade height in sub-voxels
    vec2 lean;       // rest lean (sub-voxels at the tip)
    float facing;    // blade facing angle (flat side; flutter direction)
    float stiff, phase, freq, tint;
};

// Streaming worlds: the column / tile textures are toroidal like the grid (common.glsl worldWin);
// a tile = 8x8 columns = one brick column, so tiles live in the window's brick range.
ivec2 colWrap(ivec2 c) { return worldWin.w != 0 ? c & (worldBricks.xz * 8 - 1) : c; }
ivec2 tileWrap(ivec2 t) { return worldWin.w != 0 ? t & (worldBricks.xz - 1) : t; }
bool tileInWorld(ivec2 t) { return all(greaterThanEqual(t, worldWin.xz)) && all(lessThan(t, worldWin.xz + worldBricks.xz)); }
uint cellRaw(ivec2 cell) { return texelFetch(grassTopTex, colWrap(cell), 0).r; }

// Placement fields for column `cell` (raw = its grassTopTex texel, non-zero),
// baked per column by grasstop.comp (clumps, density, height field, dryness).
GCell gCell(ivec2 cell, uint raw) {
    GCell c;
    c.gy = float(raw & 0xFFFu);
    c.variant = raw >> 12;
    uvec4 info = texelFetch(grassInfoTex, colWrap(cell), 0);
    c.n = min(int(info.x & 3u), GRASS_BLADES);
    c.dry = float(info.x >> 2) * (1.0 / 63.0);
    c.edge = float(info.y) * (1.0 / 255.0);
    float ang = (float(info.z) * (1.0 / 255.0) - 0.5) * 6.2831853;
    c.away = vec2(cos(ang), sin(ang));
    c.clumpH = float(info.w) * (1.0 / 160.0);
    return c;
}

// Blade i of a column: all randoms from one 32-bit hash.
// Rest pose (Ghost of Tsushima's blade model: height, facing, lean, bend): every blade
// leans 15-30° toward its own random facing, blended toward "away from the clump
// centre", so a patch reads as springy tufts even in calm air. The base sits on the
// side of the column opposite the lean so the whole blade fits in its column.
Blade bladeAt(ivec2 cell, GCell c, int i) {
    // (x / z wrapped to the floating origin's step: the blades survive a rebase)
    uint h = pcgHash(uint(cell.x & 0xFFFF) * 73856093u ^ uint(cell.y & 0xFFFF) * 19349663u ^ uint(i) * 0x9E3779B9u);
    uint h2 = pcgHash(h);
    Blade b;
    float r = float((h >> 4) & 15u) / 15.0;
    GVariant gv = gVariant(c.variant);
    float hs = c.clumpH * mix(1.0, 0.55, c.edge) * mix(0.7, 1.25, r);
    b.levels = clamp(int(mix(gv.hMin, gv.hMax, clamp(hs, 0.0, 1.0)) + 0.5), 1, kGMaxLevels);
    float lv = float(b.levels);
    float face = float(h2 & 255u) * (6.2831853 / 256.0);
    vec2 fdir = normalize(mix(vec2(cos(face), sin(face)), c.away, 0.6 * c.edge) + 1e-4);
    // Small natural variation only (3-10°): the lasting tilt comes from the wind's
    // SET and trampling (grass_wind.comp), not a hard-coded lean.
    float leanAmt = mix(0.05, 0.18, float((h >> 8) & 15u) / 15.0);
    b.lean = fdir * leanAmt * lv;                                             // sub-voxels at the tip
    ivec2 jit = ivec2((h2 >> 8) & 1u, (h2 >> 9) & 1u);
    ivec2 side = ivec2(b.lean.x > 0.0 ? 0 : 3, b.lean.y > 0.0 ? 0 : 3);
    b.baseSub = cell * kGCellSubs + clamp(side + (ivec2(1) - 2 * (side / 3)) * jit, ivec2(0), ivec2(3));
    b.facing = face;
    b.stiff = mix(0.6, 1.3, float((h >> 16) & 15u) / 15.0);
    b.phase = float((h >> 20) & 15u) * (6.2831 / 16.0);
    b.freq = mix(3.0, 6.5, float((h >> 24) & 15u) / 15.0);                   // flutter, rad/s
    b.tint = float((h >> 28) & 15u) / 15.0;
    return b;
}

// --- Wind -------------------------------------------------------------------
// The wind is SIMULATED (grass_wind.comp): a camera-centred 128² grid (one cell per
// 2x2 columns, toroidal) of damped springs driven by travelling gusts, integrated in
// fixed-size substeps from real time — so blades sway at the same speed at any frame
// rate, overshoot and settle after a gust, and neighbours differ. windDisp texel =
// (bend now .xy, bend last frame .zw) in "bend units" (~0.3 calm .. ~1.5 strong gust).
layout(SET(0) binding = 19) uniform sampler2D windDisp;
layout(SET(0) binding = 21) uniform sampler2D windTrample;   // flatten dir*amount .xy, amount .z
const int kWindRes = 128;
const float kWindCell = 2.0;      // world voxels per wind cell
const float kBend = 0.3;          // tip offset (sub-voxels) per level per bend unit
// Camera xz the wind grid is centred on (the pass uniforms are declared after this
// file): set by the callers before traceGrass (gbuffer.comp; grassSunVis below).
vec2 gWindEye = vec2(0.0);

// Bilinear, toroidal; fades to calm at the edge of the simulated area.
vec4 windAt(vec2 xz) {
    vec2 g = xz / kWindCell - 0.5;
    ivec2 i0 = ivec2(floor(g));
    vec2 f = g - vec2(i0);
    vec4 a = texelFetch(windDisp, (i0) & (kWindRes - 1), 0);
    vec4 b = texelFetch(windDisp, (i0 + ivec2(1, 0)) & (kWindRes - 1), 0);
    vec4 c = texelFetch(windDisp, (i0 + ivec2(0, 1)) & (kWindRes - 1), 0);
    vec4 d = texelFetch(windDisp, (i0 + ivec2(1, 1)) & (kWindRes - 1), 0);
    vec4 w = mix(mix(a, b, f.x), mix(c, d, f.x), f.y);
    vec2 dc = abs(xz - gWindEye);
    float edge = float(kWindRes) * kWindCell * 0.5;
    return w * (1.0 - smoothstep(edge - 24.0, edge - 6.0, max(dc.x, dc.y)));
}
// Trampled state at xz (nearest wind cell; changes slowly): dir*amount .xy, amount .z.
vec3 trampleAt(vec2 xz) {
    ivec2 i0 = ivec2(floor(xz / kWindCell)) & (kWindRes - 1);
    vec2 dc = abs(xz - gWindEye);
    float edge = float(kWindRes) * kWindCell * 0.5;
    return texelFetch(windTrample, i0, 0).xyz * (1.0 - step(edge - 6.0, max(dc.x, dc.y)));
}

// Soft limit into [lo, hi] (lo <= 0 <= hi): linear near 0, saturating at the column
// wall instead of sticking to it.
float softLimit(float x, float lo, float hi) {
    if (x >= 0.0) return hi > 1e-3 ? hi * tanh(x / hi) : 0.0;
    return lo < -1e-3 ? lo * tanh(x / lo) : 0.0;
}

// Tip offset (sub-voxels, relative to the base) for bend `disp` at time t: rest
// lean + spring bend + a small flutter at the blade's own constant frequency (its
// amplitude grows with the bend, its speed never changes), kept inside the column.
vec2 bladeTip(Blade b, ivec2 cellSub, vec2 disp, float t, vec3 tr) {
    float lv = float(b.levels);
    disp *= 1.0 - 0.7 * tr.z;                          // flattened blades barely sway
    vec2 fl = vec2(cos(b.facing + 1.57), sin(b.facing + 1.57)) *
              (sin(t * b.freq + b.phase) * (0.1 + 0.3 * min(length(disp), 1.5)) * min(lv, 12.0) / 9.0);
    vec2 o = b.lean + disp * (b.stiff * lv * kBend) + fl + tr.xy * (lv * 0.9);
    vec2 lo = vec2(cellSub - b.baseSub), hi = lo + 3.0;
    return vec2(softLimit(o.x, lo.x, hi.x), softLimit(o.y, lo.y, hi.y));
}

// A trampled blade is squashed down as well as bent (it can't lie flat: it stays in its
// one-voxel column), and grows back as the trampling relaxes.
Blade trampled(Blade b, vec3 tr) {
    b.levels = max(1, int(float(b.levels) * (1.0 - 0.6 * tr.z) + 0.5));
    return b;
}

struct GHit { vec3 n; vec3 pPrev; vec3 alb; };

vec3 bladeAlbedo(Blade b, GCell c, float hf) {
    GVariant gv = gVariant(c.variant);
    // Flower meadow: about a third of the blades carry a bloom at the tip, one hue per
    // clump (read off its lean direction: stable per clump) — yellow, white, violet, red.
    if (c.variant == 3u && b.tint > 0.62 && hf > 0.78) {
        const vec3 hues[4] = vec3[](vec3(0.95, 0.80, 0.18), vec3(0.92, 0.92, 0.88),
                                    vec3(0.55, 0.35, 0.85), vec3(0.85, 0.22, 0.20));
        int k = int(fract(atan(c.away.y, c.away.x) * 0.15915 + 0.5) * 4.0) & 3;
        return hues[k];
    }
    // Grows out of the ground colour: shaded at the base, brighter toward the tip;
    // per-blade value + per-clump dryness variation.
    vec3 alb = mix(gv.base, gv.tip, smoothstep(0.0, 1.0, hf));
    alb *= 0.85 + 0.3 * b.tint;
    return mix(alb, alb * vec3(1.18, 1.02, 0.62), c.dry * 0.55);
}

// --- Blades: a column of 1/4-voxel cubes following the bend ---------------------
// Position (sub-voxels) of level k's cube for a tip offset: quadratic bend. STEPPED
// snaps it to the sub-voxel grid (cubes jump a whole quarter-voxel at a time); SMOOTH
// (r.grass.style smooth) keeps it continuous — the same cubes, gliding as they sway.
vec2 bladeCol(Blade b, vec2 tip, int k) {
    float u = (float(k) + 0.5) / float(b.levels);
#ifdef GRASS_SMOOTH
    return vec2(b.baseSub) + tip * (u * u);
#else
    return vec2(b.baseSub) + floor(tip * (u * u) + 0.5);
#endif
}

// Test one column's blades inside (tmin, tmax) = the ray's span over this column.
// Per blade, only the levels the ray passes at that height can be hit (grazing
// rays: 1-3 levels), so each of those level cubes gets a direct ray-box test —
// no sub-voxel DDA through the whole column. Returns the nearest t (or -1), the
// cube-face normal, the blade colour and the same cube's position last frame.
// anyHit (shadow rays: only "blocked?"): return at the first blade cube met, no hit shading.
float traceCell(ivec2 cell, GCell c, vec3 ro, vec3 rd, float tmin, float tmax, float t, float prevT,
                out GHit hit, const bool anyHit) {
    ivec2 cellSub = cell * kGCellSubs;
    vec4 wd = windAt(vec2(cell) + 0.5);
    vec3 tr = trampleAt(vec2(cell) + 0.5);
    vec3 srd = mix(rd, vec3(1e-6), lessThan(abs(rd), vec3(1e-6)));
    vec3 inv = 1.0 / srd;
    float y0 = ro.y + rd.y * tmin, y1 = ro.y + rd.y * tmax;
    float kLo = (min(y0, y1) - c.gy) / kGSub, kHi = (max(y0, y1) - c.gy) / kGSub;
    float tBest = tmax;
    int bestI = -1, bestK = 0;
    vec3 bestN = vec3(0.0);
    for (int i = 0; i < GRASS_BLADES; i++) {
        if (i >= c.n) break;
        Blade b = trampled(bladeAt(cell, c, i), tr);
        int k0 = max(int(floor(kLo)), 0), k1 = min(int(floor(kHi)), b.levels - 1);
        if (k0 > k1) continue;
        vec2 tip = bladeTip(b, cellSub, wd.xy, t, tr);
        for (int k = k0; k <= k1; k++) {
            vec2 col = bladeCol(b, tip, k);
            vec3 bmin = vec3(col * kGSub, c.gy + float(k) * kGSub).xzy;
            vec3 ta = (bmin - ro) * inv, tb = (bmin + kGSub - ro) * inv;
            vec3 t0v = min(ta, tb), t1v = max(ta, tb);
            float tE = max(max(t0v.x, t0v.y), t0v.z), tX = min(min(t1v.x, t1v.y), t1v.z);
            if (tX < max(tE, tmin) || tE >= tBest) continue;
            if (anyHit) return max(tE, tmin);
            tBest = max(tE, tmin);
            bestI = i; bestK = k;
            bestN = tE == t0v.x ? vec3(-sign(srd.x), 0, 0)
                  : tE == t0v.y ? vec3(0, -sign(srd.y), 0) : vec3(0, 0, -sign(srd.z));
        }
    }
    if (bestI < 0) return -1.0;
    Blade b = trampled(bladeAt(cell, c, bestI), tr);
    hit.alb = bladeAlbedo(b, c, (float(bestK) + 0.5) / float(b.levels));
    hit.n = bestN;
    vec3 P = ro + rd * tBest;
    vec2 now = bladeCol(b, bladeTip(b, cellSub, wd.xy, t, tr), bestK);
    vec2 dc = bladeCol(b, bladeTip(b, cellSub, wd.zw, prevT, tr), bestK) - now;   // where this cube was
    hit.pPrev = P + vec3(dc.x, 0.0, dc.y) * kGSub;
    return tBest;
}

// Walk the columns along the ray inside [t0, t1]; first hit wins.
float grassCells(vec3 ro, vec3 rd, float t0, float t1, float t, float prevT, out GHit hit, const bool anyHit) {
    vec2 o = ro.xz, d = rd.xz;
    d = mix(d, vec2(1e-6), lessThan(abs(d), vec2(1e-6)));
    vec2 inv = 1.0 / d;
    vec2 p0 = o + d * t0;
    ivec2 cell = ivec2(floor(p0));
    ivec2 stp = ivec2(sign(d));
    vec2 tMax = ((vec2(cell) + step(vec2(0.0), d)) - o) * inv;
    vec2 tDel = abs(inv);
    float tc = t0;
    for (int i = 0; i < 160 && tc < t1; i++) {
        float tn = min(min(tMax.x, tMax.y), t1);
        // Cheap reject with the column's own ground height: skip unless the ray is
        // below the tallest possible blade somewhere in the column.
        float yLo = min(ro.y + rd.y * tc, ro.y + rd.y * tn);
        float yHi = max(ro.y + rd.y * tc, ro.y + rd.y * tn);
        uint raw = cellRaw(cell);
        float g0 = float(raw & 0xFFFu);
        // Tallest blade this variant can grow (short dry grass / moss reject early).
        float vMaxH = min(gVariant(raw >> 12).hMax * 1.25 + 0.5, float(kGMaxLevels)) * kGSub;
        if (raw != 0u && yLo < g0 + vMaxH + 0.1 && yHi > g0 - 1.0) {
            GCell c = gCell(cell, raw);
            if (c.n > 0) {
                // A bent blade may be hit slightly outside the column's t-range at
                // grazing angles: allow a small margin.
                float th = traceCell(cell, c, ro, rd, max(tc - 0.05, 0.0), min(tn + 0.05, t1), t, prevT,
                                     hit, anyHit);
                if (th >= 0.0) return th;
            }
        }
        if (tMax.x < tMax.y) { tc = tMax.x; tMax.x += tDel.x; cell.x += stp.x; }
        else { tc = tMax.y; tMax.y += tDel.y; cell.y += stp.y; }
    }
    return -1.0;
}

// Full layer: skip 8x8-column tiles that have no grass or that the ray passes
// above; walk columns only in the rest. tEnd = terrain hit / draw distance.
float traceGrass(vec3 ro, vec3 rd, float tEnd, float t, float prevT, out GHit hit, const bool anyHit) {
    vec2 o = ro.xz / 8.0, d = rd.xz;
    d = mix(d, vec2(1e-6), lessThan(abs(d), vec2(1e-6)));
    vec2 inv = 1.0 / d;
    ivec2 tile = ivec2(floor(o));
    ivec2 stp = ivec2(sign(d));
    vec2 tMax = ((vec2(tile) + step(vec2(0.0), d)) - o) * inv * 8.0;
    vec2 tDel = abs(inv) * 8.0;
    float tc = 0.0;
    for (int i = 0; i < 128 && tc < tEnd; i++) {
        float tn = min(min(tMax.x, tMax.y), tEnd);
        if (tileInWorld(tile)) {
            uvec2 mm = texelFetch(grassTileTex, tileWrap(tile), 0).rg;
            if (mm.y > 0u) {
                // Clip this tile's span to where the ray is inside the band
                // [lowest grass, highest grass + its tallest blade (tall grass only where flagged)].
                float bandMax = float(mm.y & 0x7FFFu) + ((mm.y & kGTileTall) != 0u ? kGMaxH : kGMaxHShort) + 0.1;
                float bandMin = float(mm.x) - 1.0;
                float a = tc, e = tn;
                if (abs(rd.y) > 1e-6) {
                    float tTop = (bandMax - ro.y) / rd.y, tBot = (bandMin - ro.y) / rd.y;
                    a = max(a, min(tTop, tBot));
                    e = min(e, max(tTop, tBot));
                } else if (ro.y > bandMax || ro.y < bandMin) {
                    e = a;   // horizontal ray outside the band
                }
                if (e > a) {
                    float th = grassCells(ro, rd, max(a - 0.05, 0.0), e, t, prevT, hit, anyHit);
                    if (th >= 0.0) return th;
                }
            }
        }
        if (tMax.x < tMax.y) { tc = tMax.x; tMax.x += tDel.x; tile.x += stp.x; }
        else { tc = tMax.y; tMax.y += tDel.y; tile.y += stp.y; }
    }
    return -1.0;
}

// --- Far grass LOD: tufts ---------------------------------------------------------------
// Past the blade distance (r.grass.far x it): one small upright box per 2x2 columns, at a hashed
// spot, as tall as that clump's blades and coloured like them (base -> tip, dryness), present as
// often as the column has blades - so the field keeps its silhouette and patchiness out there at a
// fraction of the cost (one ray-box per 2x2 cell, same tile skipping). Static: no wind.
float tuftCell(ivec2 cell2, vec3 ro, vec3 rd, float tmin, float tmax, out GHit hit) {
    uint h = pcgHash(uint(cell2.x & 0x7FFF) * 0x632BE5ABu ^ uint(cell2.y & 0x7FFF) * 0x85157AF5u);   // (2x2 cells: the origin step / 2)
    ivec2 col = cell2 * 2 + ivec2(h & 1u, (h >> 1) & 1u);
    uint raw = cellRaw(col);
    if (raw == 0u) return -1.0;
    GCell c = gCell(col, raw);
    if (c.n == 0 || float((h >> 2) & 255u) * (float(GRASS_BLADES) / 255.0) >= float(c.n) + 0.25) return -1.0;
    GVariant gv = gVariant(c.variant);
    float ht = mix(gv.hMin, gv.hMax, clamp(c.clumpH * mix(1.0, 0.55, c.edge), 0.0, 1.0)) * kGSub * 0.85;
    float w = mix(0.7, 1.1, float(c.n) / float(GRASS_BLADES));
    vec2 off = vec2(float((h >> 10) & 15u), float((h >> 14) & 15u)) * ((1.0 - w) / 15.0);
    vec3 bmin = vec3(float(col.x) + off.x, c.gy, float(col.y) + off.y);
    vec3 srd = mix(rd, vec3(1e-6), lessThan(abs(rd), vec3(1e-6)));
    vec3 ta = (bmin - ro) / srd, tb = (bmin + vec3(w, ht, w) - ro) / srd;
    vec3 t0v = min(ta, tb), t1v = max(ta, tb);
    float tE = max(max(t0v.x, t0v.y), t0v.z), tX = min(min(t1v.x, t1v.y), t1v.z);
    if (tX < max(tE, tmin) || tE > tmax) return -1.0;
    tE = max(tE, tmin);
    hit.n = tE == t0v.x ? vec3(-sign(srd.x), 0, 0) : tE == t0v.y ? vec3(0, -sign(srd.y), 0) : vec3(0, 0, -sign(srd.z));
    vec3 P = ro + rd * tE;
    vec3 alb = mix(gv.base, gv.tip, smoothstep(0.0, 1.0, (P.y - c.gy) / ht)) * 0.95;
    hit.alb = mix(alb, alb * vec3(1.18, 1.02, 0.62), c.dry * 0.55);
    hit.pPrev = P;
    return tE;
}
// Tufts along the ray inside [t0, tEnd]: the tile skip of traceGrass, then 2x2-column cells.
float traceTufts(vec3 ro, vec3 rd, float t0, float tEnd, out GHit hit) {
    vec2 d = rd.xz;
    d = mix(d, vec2(1e-6), lessThan(abs(d), vec2(1e-6)));
    vec2 inv = 1.0 / d;
    ivec2 stp = ivec2(sign(d));
    vec2 o = ro.xz / 8.0;
    ivec2 tile = ivec2(floor(o + d * (t0 / 8.0)));
    vec2 tMax = ((vec2(tile) + step(vec2(0.0), d)) - o) * inv * 8.0;
    vec2 tDel = abs(inv) * 8.0;
    float tc = t0;
    for (int i = 0; i < 160 && tc < tEnd; i++) {
        float tn = min(min(tMax.x, tMax.y), tEnd);
        if (tileInWorld(tile)) {
            uvec2 mm = texelFetch(grassTileTex, tileWrap(tile), 0).rg;
            if (mm.y > 0u) {
                float bandMax = float(mm.y & 0x7FFFu) + ((mm.y & kGTileTall) != 0u ? kGMaxH : kGMaxHShort) + 0.1;
                float bandMin = float(mm.x) - 1.0;
                float a = tc, e = tn;
                if (abs(rd.y) > 1e-6) {
                    float tTop = (bandMax - ro.y) / rd.y, tBot = (bandMin - ro.y) / rd.y;
                    a = max(a, min(tTop, tBot));
                    e = min(e, max(tTop, tBot));
                } else if (ro.y > bandMax || ro.y < bandMin) {
                    e = a;
                }
                if (e > a) {   // 2x2-column cells inside [a, e]
                    vec2 p0 = (ro.xz + d * a) * 0.5;
                    ivec2 cell = ivec2(floor(p0));
                    vec2 cMax = ((vec2(cell) + step(vec2(0.0), d)) - ro.xz * 0.5) * inv * 2.0;
                    vec2 cDel = abs(inv) * 2.0;
                    float cc = a;
                    for (int j = 0; j < 12 && cc < e; j++) {
                        float cn = min(min(cMax.x, cMax.y), e);
                        float th = tuftCell(cell, ro, rd, max(cc - 0.05, 0.0), cn + 0.05, hit);
                        if (th >= 0.0) return th;
                        if (cMax.x < cMax.y) { cc = cMax.x; cMax.x += cDel.x; cell.x += stp.x; }
                        else { cc = cMax.y; cMax.y += cDel.y; cell.y += stp.y; }
                    }
                }
            }
        }
        if (tMax.x < tMax.y) { tc = tMax.x; tMax.x += tDel.x; tile.x += stp.x; }
        else { tc = tMax.y; tMax.y += tDel.y; tile.y += stp.y; }
    }
    return -1.0;
}

// Prefiltered far grass: expected blade colour + coverage for a grass surface
// point (variant from the material), matching the blade layer's statistics.
vec3 grassFarColour(vec3 groundAlb, vec2 xz, uint variant, float rdY, float far) {
    GVariant gv = gVariant(variant);
    float dens = gDensity(xz) * gv.dens;
    float meanH = mix(gv.hMin, gv.hMax, 0.5) * kGSub;
    float cov = 1.0 - exp(-0.9 * dens * meanH * 0.6 / max(abs(rdY), 0.03));
    vec3 meanBlade = mix(gv.base, gv.tip, 0.55);
    // Near: the ground between blades sits in their shade -> a little darker.
    vec3 a = groundAlb * mix(1.0 - 0.28 * dens, 1.0, far);
    return mix(a, meanBlade, far * cov);
}

// Sun visibility through the grass blade layer (r.grass.shadows): a short ray from
// P toward the sun, only as long as it can still be inside the blade band. Blades
// move with the wind, so their shadows do too. 1 = not shadowed by blades.
// Faded out over the same distance band as the blades themselves.
float grassSunVis(vec3 P, vec3 N, vec3 eye, float t, float prevT) {
    if (grassShadows == 0 || sunDir.y <= 0.02) return 1.0;
    float dist = length(P - eye);
    if (dist > grassDist) return 1.0;
    // As long as this column's tallest blades need (tall meadow grass casts longer shadows).
    uint pRaw = cellRaw(ivec2(floor(P.xz)));
    float tEnd = ((pRaw >> 12) == 4u || (pRaw >> 12) == 5u ? kGMaxH : kGMaxHShort) * 1.8 / max(sunDir.y, 0.08) + 0.5;
    gWindEye = eye.xz;
    GHit gh;
    float th = traceGrass(P + N * 0.03, sunDir, tEnd, t, prevT, gh, true);   // (only "blocked?")
    if (th < 0.0) return 1.0;
    return smoothstep(0.7 * grassDist, grassDist, dist);
}
