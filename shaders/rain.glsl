// Rain (rainA.x = intensity, rainA.y = time): drawn into the HDR image after the denoiser / upscaler (torchfx.comp;
// SVGF: denoise.comp), so thin fast streaks are never smeared by temporal accumulation.
//
// The way games do it (Lagarde, "Water drop 2" / Remember Me; Tatarchuk, ToyShop; Garg & Nayar): a streak is a falling
// drop motion-blurred over the exposure - long, thin, faint. A drop is a tiny lens showing the bright sky above it
// (so rain shows against dark trees and almost vanishes against the sky) and glints when backlit by the sun. Rain
// only falls where the sky is open, and never in front of what's nearer.
//
// Drops live in the world, like a game's rain particles in a box round the camera (UE / Unity / Frostbite): five
// world grids of vertical trains (16 cm .. 3.8 m apart), every train a column of drops falling at ~9-10 m/s, leaned by
// the wind. A train keeps its grid for good and is seen out to that grid's range (3 - 70 m, fading): walking, you
// pass the drops; nothing is tied to the distance from you. A ray walks each grid's cells in 2D and tests the train
// it passes closest to. A drop is a few mm wide: far away it covers a fraction of a pixel, so it gets fainter, not
// wider. Splashes: tiny crowns on open ground near the eye. (Ripples on water: lakeNormal.)

float rainHash(uvec3 v) { return float(pcgHash(v.x * 1597334677u ^ v.y * 3812015801u ^ v.z * 2798796415u) & 65535u) / 65536.0; }

vec3 rainDropLight(vec3 rd) {
    // the sky refracted through the drop (it looks up-ish whatever the view), a backlit glint toward the sun
    vec3 sky = skyColor(normalize(vec3(-rd.x * 0.3, 1.0, -rd.z * 0.3)), 0.0);
    float sunUp = smoothstep(-0.02, 0.1, sunDir.y);
    float glint = pow(max(dot(rd, sunDir), 0.0), 8.0) * 3.0 + 0.15;
    return sky * 1.1 + sunRGB * glint * sunUp * 0.25;
}

// The rain over colour `col` along the eye ray (depth: the surface distance, < 0 = sky). `pixAng`: radians per pixel.
vec3 rainDrops(vec3 col, vec3 ro, vec3 rd, float depth, float pixAng, vec4 wave) {   // wave: waveP (wind dir, strength)
    float R = rainA.x;
    if (R <= 0.001) return col;
    float tEnd = depth < 0.0 ? 1e30 : depth;
    float time = rainA.y;
    const float vFall = 95.0;                       // voxels/s (9.5 m/s: terminal velocity of a 2-3 mm drop)
    vec2 wind = wave.xy * (0.25 + 0.6 * min(wave.z, 2.0)) * 30.0;   // horizontal drift (voxels/s)
    vec2 lean = wind / vFall;                       // the train leans: x = xs + lean * y
    // the ray in sheared space (columns vertical there)
    vec2 o = ro.xz - lean * ro.y, d = rd.xz - lean * rd.y;
    float dl = length(d);
    if (dl < 1e-4) return col;                     // straight up / down along the trains: nothing to see
    vec3 Ld = rainDropLight(rd);
    vec3 outc = col;
    // Each drop train belongs to one level for good (its cell in that level's world grid): level L's trains are seen
    // from the eye out to its range, fading over the far half - a drop never pops in or out at a distance from you (the
    // old bands round the eye re-dealt the drops at fixed distances: the rain 'chased' you); walking, you pass them.
    for (int L = 4; L >= 0; --L) {                  // coarse to fine
        float range = 30.0 * pow(2.2, float(L));    // 3 m .. 70 m
        float tA = 3.0, tB = min(range, tEnd);      // (none in the first 30 cm: a drop on your nose)
        if (tA >= tB) continue;
        float cs = 1.6 * pow(2.2, float(L));       // train spacing (voxels): 16 cm .. 3.8 m
        // 2D walk over the cells the ray crosses in [tA, tB]
        vec2 p = o + d * tA;
        vec2 c = floor(p / cs);
        vec2 st = sign(d);
        vec2 tDelta = abs(cs / d);
        vec2 nb = (c + max(st, 0.0)) * cs;
        vec2 tMaxV = vec2(d.x != 0.0 ? (nb.x - o.x) / d.x : 1e30, d.y != 0.0 ? (nb.y - o.y) / d.y : 1e30);
        float best = 0.0;
        for (int i = 0; i < 24; ++i) {
            uint hk = pcgHash(uint(int(c.x) + 32768) * 1597334677u ^ uint(int(c.y) + 32768) * 3812015801u ^ uint(L) * 2798796415u);
            float h0 = float(hk & 255u) / 256.0;
            if (h0 < R * 0.9) {
                // the train: its line, phase, speed, spacing (one hash, four values)
                float h1 = float((hk >> 8) & 255u) / 256.0, h2 = float((hk >> 16) & 255u) / 256.0;
                float h3 = float(hk >> 24) / 256.0;
                vec2 X = (c + 0.15 + 0.7 * vec2(h1, h2)) * cs;
                float ts = dot(X - o, d) / (dl * dl);          // closest approach (sheared xz)
                if (ts > tA && ts < tB) {
                    float dist = length(o + d * ts - X);           // (the common perpendicular of a vertical line and the ray is level)
                    float fp = ts * pixAng;                    // a pixel there (voxels)
                    const float rDrop = 0.03;                  // 3 mm (a voxel is 10 cm)
                    float w = max(rDrop, fp * 0.7);
                    if (dist < w * 1.5) {
                        float v = vFall * (0.85 + 0.3 * h3);
                        float len = v * 0.025 * (0.7 + 0.6 * h1);        // the streak over ~1/40 s
                        float period = len * (2.0 + 4.0 * h2) + 3.0;
                        float y = ro.y + rd.y * ts;
                        float u = mod(y + v * time + h0 * 977.0, period);   // 0 = the drop's head, up the streak
                        if (u < len) {
                            vec3 P = vec3(ro.x + rd.x * ts, y, ro.z + rd.z * ts);
                            float open = skyOpen(P);
                            float a = exp(-2.0 * (dist / w) * (dist / w)) * (rDrop / w)
                                    * smoothstep(0.0, 0.25, u / len) * (1.0 - smoothstep(0.55, 1.0, u / len))
                                    * open * smoothstep(tEnd, tEnd - 3.0, ts)
                                    * (1.0 - smoothstep(0.5 * range, range, ts)) * smoothstep(3.0, 8.0, ts);
                            best = max(best, a);
                        }
                    }
                }
            }
            // next cell
            float tn = min(tMaxV.x, tMaxV.y);
            if (tn > tB) break;
            if (tMaxV.x < tMaxV.y) { tMaxV.x += tDelta.x; c.x += st.x; } else { tMaxV.y += tDelta.y; c.y += st.y; }
        }
        // (a drop never darkens what's behind it: it refracts the bright sky / cloud deck, so it shows on dark trees and
        // nearly vanishes against the sky)
        outc = mix(outc, max(Ld, outc * 1.25 + 0.01), clamp(best * 0.45 * (0.6 + 0.4 * R), 0.0, 1.0));
    }
    return outc;
}

// Splashes: each 25 cm cell of open, up-facing ground near the eye takes a drop now and then - a tiny crown that
// flashes and spreads for a tenth of a second.
vec3 rainSplashes(vec3 col, vec3 P, vec3 N, float dist, float pixAng) {
    float R = rainA.x;
    if (R <= 0.001 || N.y < 0.7 || dist > 300.0) return col;
    if (skyOpen(P + vec3(0.0, 0.6, 0.0)) < 0.5) return col;
    float time = rainA.y;
    const float cs = 2.5;
    vec2 c0 = floor(P.xz / cs);
    float fp = dist * pixAng;
    float acc = 0.0;
    for (int j = -1; j <= 1; ++j)
        for (int i = -1; i <= 1; ++i) {
            vec2 c = c0 + vec2(i, j);
            uvec3 key = uvec3(ivec2(c) + 32768, 977u);
            float h0 = rainHash(key), h1 = rainHash(key + uvec3(0u, 0u, 11u)), h2 = rainHash(key + uvec3(0u, 0u, 23u));
            float T = mix(0.35, 1.4, h0) / max(R, 0.1);   // seconds between drops in this cell
            float age = mod(time + h1 * 13.7, T);
            const float life = 0.14;
            if (age > life) continue;
            float n = floor((time + h1 * 13.7) / T);     // (each drop lands somewhere new)
            vec2 X = (c + 0.2 + 0.6 * vec2(rainHash(key + uvec3(uint(n), 0u, 31u)), h2)) * cs;
            float r = length(P.xz - X);
            float rr = 0.15 + age / life * 0.55;           // crown radius (voxels: ~2 - 7 cm)
            float w = max(0.12, fp * 0.8);
            float ring = exp(-pow((r - rr) / w, 2.0)) * (0.12 / max(w, 0.12));
            acc += ring * (1.0 - age / life);
        }
    vec3 Ls = skyColor(vec3(0.0, 1.0, 0.0), 0.0) * 1.2;   // (spray: lit by the sky)
    return mix(col, Ls, clamp(acc * 0.8 * (1.0 - smoothstep(150.0, 300.0, dist)), 0.0, 0.7));
}
