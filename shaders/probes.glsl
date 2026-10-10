// Standard lighting: DDGI-style irradiance probes (Majercik et al. 2019), used by
// shade_std.comp (per pixel) and probe_trace.comp (multi-bounce at ray hits).
// Camera-centred grid of probeDims probes, probeSpacing voxels apart, addressed
// toroidally by world cell. Per probe: 8x8 octahedral irradiance (RGBA32F, linear) +
// 16x16 depth moments (RG32F — d² in half float cancels to noise; Chebyshev
// visibility = no light leaking through walls),
// each tile with a 1-texel border so hardware bilinear works (probe_update.comp).
// probeStateTex holds, per probe slot, the key of the world cell it currently
// describes (0 = none / inside solid): a stale slot is simply ignored.

layout(SET(0) binding = 15) uniform sampler2D probeIrrTex;
layout(SET(0) binding = 16) uniform sampler2D probeDepthTex;
layout(SET(0) binding = 17) uniform usampler2D probeStateTex;
// Per probe: xyz = relocation offset (voxels, DDGI "probe relocation": pushed away
// from nearby geometry), w = 1 active / 0 disabled (buried or enclosed: it would
// only see backfaces and smear darkness onto nearby surfaces).
layout(SET(0) binding = 18) uniform sampler2D probeDataTex;

const int kIrrN = 8, kDepN = 16;          // interior texels per tile side
const int kProbeRays = 64;
// Perceptual encoding (DDGI's pow 1/5), used only to JUDGE how much a probe changed.
// Blending stays linear: averaging encoded values (as DDGI stores them) biases the
// result dark where a few bright rays mix with dark ones (a sky shaft: -20 %).
vec3 irrEncode(vec3 e) { return pow(max(e, vec3(0.0)), vec3(0.2)); }

vec2 signNotZero(vec2 v) { return vec2(v.x >= 0.0 ? 1.0 : -1.0, v.y >= 0.0 ? 1.0 : -1.0); }
vec2 octEncode(vec3 n) {                   // unit vector -> [-1,1]^2
    vec2 p = n.xz * (1.0 / (abs(n.x) + abs(n.y) + abs(n.z)));
    return n.y < 0.0 ? (1.0 - abs(p.yx)) * signNotZero(p) : p;
}
vec3 octDecode(vec2 e) {
    vec3 v = vec3(e.x, 1.0 - abs(e.x) - abs(e.y), e.y);
    if (v.y < 0.0) v.xz = (1.0 - abs(v.zx)) * signNotZero(v.xz);
    return normalize(v);
}

// Two probe LEVELS share the atlases: 0 = fine grid (probeSpacing, near the
// camera), 1 = coarse grid (probeSpacingC, the far background). Same dims.
float lvSpacing(int L) { return L == 0 ? probeSpacing : probeSpacingC; }
ivec3 lvOrigin(int L) { return L == 0 ? probeOrigin : probeOriginC; }
ivec3 probeSlot(ivec3 cell) { return ((cell % probeDims) + probeDims) % probeDims; }
int probeLinear(ivec3 slot) { return slot.x + probeDims.x * (slot.y + probeDims.y * slot.z); }
int probeIndex(int L, ivec3 slot) { return probeLinear(slot) + L * probeDims.x * probeDims.y * probeDims.z; }
ivec2 probeStateCoord(int idx) { return ivec2(idx % 2048, idx / 2048); }
// Tile of a probe in the atlases: 8 z-layers side by side, the rest stacked; the
// coarse level below the fine one.
ivec2 probeTile(int L, ivec3 slot) {
    return ivec2(slot.x + (slot.z % 8) * probeDims.x,
                 slot.y + (slot.z / 8) * probeDims.y + L * probeDims.y * (probeDims.z / 8));
}
uint probeKey(int L, ivec3 cell) {
    return (uint(cell.x + 1) & 1023u) | ((uint(cell.y + 1) & 1023u) << 10) |
           ((uint(cell.z + 1) & 1023u) << 20) | (uint(L) << 30);
}
vec3 probeCentre(int L, ivec3 cell) { return (vec3(cell) + 0.5) * lvSpacing(L); }
bool probeInWindow(int L, ivec3 cell) {
    return all(greaterThanEqual(cell, lvOrigin(L))) && all(lessThan(cell, lvOrigin(L) + probeDims));
}
// Decode a probe-update list entry (slot | level << 31) into level, slot, cell.
// Entry: bit 31 = level, bit 30 = re-settle (an edit changed the geometry around it:
// start over like a new cell — position, classification, no history), rest = slot.
bool probeResettle(uint e) { return (e & 0x40000000u) != 0u; }
void probeFromEntry(uint e, out int L, out ivec3 slot, out ivec3 cell) {
    L = int(e >> 31);
    int lin = int(e & 0x3FFFFFFFu);
    slot = ivec3(lin % probeDims.x, (lin / probeDims.x) % probeDims.y, lin / (probeDims.x * probeDims.y));
    ivec3 o = lvOrigin(L);
    cell = o + (((slot - o) % probeDims) + probeDims) % probeDims;
}

// Sky light used where no probe covers the point (beyond the grid).
vec3 probeFallback(vec3 N) {
    return mix(skyHorizon, skyZenith, clamp(N.y * 0.5 + 0.5, 0.0, 1.0)) * skyScale * 0.85;
}

// One probe level at P: visibility-weighted irradiance of its valid probes, and the
// level's COVERAGE (0..1) = how much of P it can vouch for: trilinear weight of the
// probes in its window x each probe's confidence (ramps up over its first 2 updates,
// so a probe that just entered the grid fades in instead of popping) x a fade near
// the window border (so the grid edge is never a line). Disabled probes (buried /
// enclosed) count as covered but carry no light: neighbours are renormalised.
vec3 probeLevel(int L, vec3 P, vec3 N, vec3 V, out float coverage, out bool sealed) {
    float sp = lvSpacing(L);
    vec3 Pb = P + (N * (1.0 - stdViewBias) + V * stdViewBias) * (0.3 * sp);
    vec3 g = Pb / sp - 0.5;
    ivec3 base = ivec3(floor(g));
    vec3 a = g - vec3(base);
    vec3 lo = vec3(lvOrigin(L)), hi = vec3(lvOrigin(L) + probeDims) - 1.0;
    vec3 e3 = min(g - lo, hi - g);
    float edge = smoothstep(0.0, 2.0, min(min(e3.x, e3.y), e3.z));
    vec3 sum = vec3(0.0);
    float wsum = 0.0, cov = 0.0;
    coverage = 0.0;
    sealed = false;
    if (edge <= 0.0) return vec3(0.0);
    vec2 irrSize = vec2(textureSize(probeIrrTex, 0)), depSize = vec2(textureSize(probeDepthTex, 0));
    for (int i = 0; i < 8; i++) {
        ivec3 o = ivec3(i & 1, (i >> 1) & 1, i >> 2);
        ivec3 cell = base + o;
        vec3 tri3 = mix(1.0 - a, a, vec3(o));
        float tri = tri3.x * tri3.y * tri3.z;
        if (!probeInWindow(L, cell)) continue;
        ivec3 slot = probeSlot(cell);
        ivec2 sc = probeStateCoord(probeIndex(L, slot));
        if (texelFetch(probeStateTex, sc, 0).r != probeKey(L, cell)) continue;   // not traced yet
        vec4 pd = texelFetch(probeDataTex, sc, 0);
        if (pd.w < 0.5) { cov += tri; continue; }            // buried / enclosed: covered, no light
        float conf = clamp((pd.w - 1.0) / 2.0, 0.0, 1.0);
        cov += tri * conf;
        vec3 pp = probeCentre(L, cell) + pd.xyz;
        vec3 toProbe = normalize(pp - P);
        float w = (dot(toProbe, N) + 1.0) * 0.5;
        w = w * w + 0.2;
        vec2 tile = vec2(probeTile(L, slot));
        vec3 d = Pb - pp;
        float r = length(d);
        vec2 duv = (tile * float(kDepN + 2) + 1.0 + (octEncode(d / max(r, 1e-4)) * 0.5 + 0.5) * float(kDepN)) / depSize;
        vec2 m = textureLod(probeDepthTex, duv, 0.0).rg;
        if (r > m.x) {                                          // Chebyshev visibility
            float var = abs(m.y - m.x * m.x);
            float cheb = var / (var + (r - m.x) * (r - m.x));
            w *= max(cheb * cheb * cheb, 0.0);
        }
        w = max(w, 1e-5);
        if (w < 0.2) w *= w * w / (0.2 * 0.2);   // crush tiny weights (DDGI)
        w *= tri * conf;
        vec2 iuv = (tile * float(kIrrN + 2) + 1.0 + (octEncode(N) * 0.5 + 0.5) * float(kIrrN)) / irrSize;
        sum += textureLod(probeIrrTex, iuv, 0.0).rgb * w;
        wsum += w;
    }
    // Every covering probe disabled (buried / enclosed): this level has no light to
    // offer (coverage 0, the caller falls back), but it is SEALED — see probeIrradiance.
    sealed = cov > 0.0 && wsum < 1e-7;
    if (wsum < 1e-7) return vec3(0.0);
    coverage = clamp(cov, 0.0, 1.0) * edge;
    return sum / wsum;
}

// Indirect light at P (normal N, toward-viewer V). Fine probes near the camera,
// blended over the coarse background grid, blended over sky light: every hand-over
// is a smooth coverage-weighted mix, so nothing pops as you move.
vec3 probeIrradiance(vec3 P, vec3 N, vec3 V) {
    float cC, cF;
    bool sF, sC;
    vec3 iF = probeLevel(0, P, N, V, cF, sF);
    if (cF >= 1.0) return iF;                 // fully covered by the fine grid
    vec3 iC = probeLevel(1, P, N, V, cC, sC);
    // A point whose coarse probes are all sealed in rock is not under open sky: no
    // sky-light fallback there (lit tight tunnels / caves before). A sealed FINE level
    // (e.g. probes inside a leaf clump) still defers to the coarse grid.
    vec3 outside = sC ? vec3(0.0) : probeFallback(N) * skyOpenWide(P + N * 1.5);   // (none underground)
    return mix(mix(outside, iC, cC), iF, cF);
}

// 64 well-spread directions (spherical Fibonacci), randomly rotated each frame.
vec3 probeRayDir(int i, uint frame) {
    const float golden = 2.39996323;
    float z = 1.0 - (2.0 * float(i) + 1.0) / float(kProbeRays);
    float rr = sqrt(max(0.0, 1.0 - z * z));
    float ph = float(i) * golden;
    vec3 d = vec3(cos(ph) * rr, z, sin(ph) * rr);
    // Random rotation (from a random unit quaternion) shared by every probe this frame.
    uint h = pcgHash(frame * 747796405u + 1u);
    float u1 = float(h & 0xFFFFu) / 65535.0, u2 = float(h >> 16) / 65535.0;
    uint h2 = pcgHash(h);
    float u3 = float(h2 & 0xFFFFu) / 65535.0;
    vec4 q = vec4(sqrt(1.0 - u1) * sin(6.2831853 * u2), sqrt(1.0 - u1) * cos(6.2831853 * u2),
                  sqrt(u1) * sin(6.2831853 * u3), sqrt(u1) * cos(6.2831853 * u3));
    return d + 2.0 * cross(q.xyz, cross(q.xyz, d) + q.w * d);
}
