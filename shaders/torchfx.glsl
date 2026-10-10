// Torch flames and embers, voxel style (torch.h; the light they give is the light list's, lights.h - this is what you
// see). Per torch near the eye (_LightsU torchA / torchB, lights.h GpuTorch), on its head voxel, like a voxel campfire:
// a glowing core cube, and a stream of cubes up to a voxel big, each turned its own way, that rise off it tumbling,
// shrink and cool from yellow-white through orange to red; the stream's size follows the torch's light (lightAnim
// kind 3, the same phase: what you see flicker is what lights the room). Embers: small turned cubes rising off it,
// fading. Each cube face is shaded a little differently, so the cubes read as cubes. Added in HDR before exposure /
// bloom (torchfx.comp; plain SVGF: denoise.comp), after any denoiser.

float tfHash(vec3 p) { return fract(sin(dot(p, vec3(127.1, 311.7, 74.7))) * 43758.5453); }
mat3 tfRot(vec3 e) {   // rotation by angles e (x, then y, then z)
    float cx = cos(e.x), sx = sin(e.x), cy = cos(e.y), sy = sin(e.y), cz = cos(e.z), sz = sin(e.z);
    mat3 rx = mat3(1.0, 0.0, 0.0, 0.0, cx, sx, 0.0, -sx, cx);
    mat3 ry = mat3(cy, 0.0, -sy, 0.0, 1.0, 0.0, sy, 0.0, cy);
    mat3 rz = mat3(cz, sz, 0.0, -sz, cz, 0.0, 0.0, 0.0, 1.0);
    return rz * ry * rx;
}
// The nearest hit (t) of the ray with the cube centre c, half size h, turned by R, before tBest; shade: how lit the
// face it enters looks (facing the eye / up: brighter).
bool tfCube(vec3 ro, vec3 rd, vec3 c, float h, mat3 R, inout float tBest, out float shade) {
    mat3 Rt = transpose(R);
    vec3 o = Rt * (ro - c), d = Rt * rd;
    d = mix(d, vec3(1e-6), lessThan(abs(d), vec3(1e-6)));
    vec3 inv = 1.0 / d;
    vec3 t0 = (-h - o) * inv, t1 = (h - o) * inv;
    vec3 tn = min(t0, t1), tf = max(t0, t1);
    float te = max(max(tn.x, tn.y), tn.z), tx = min(min(tf.x, tf.y), tf.z);
    shade = 1.0;
    if (te > tx || tx < 0.0 || max(te, 0.0) >= tBest) return false;
    tBest = max(te, 0.0);
    vec3 nl = te == tn.x ? vec3(-sign(d.x), 0.0, 0.0) : te == tn.y ? vec3(0.0, -sign(d.y), 0.0) : vec3(0.0, 0.0, -sign(d.z));
    vec3 n = R * nl;
    shade = 0.62 + 0.23 * max(dot(n, -rd), 0.0) + 0.15 * (0.5 + 0.5 * n.y);
    return true;
}

vec3 torchFlames(vec3 ro, vec3 rd, float tMax, float t) {
    vec3 sum = vec3(0.0);
    vec3 safeRd = mix(rd, vec3(1e-6), lessThan(abs(rd), vec3(1e-6)));
    vec3 inv = 1.0 / safeRd;
    const int kN = 7;   // rising cubes per torch
    for (int i = 0; i < torchCount.x; i++) {
        vec4 a = torchA[i], b = torchB[i];
        uint ph = uint(b.z);
        float seed = float(ph & 1023u);
        vec3 base = vec3(a.x + 0.5, a.y + 1.0, a.z + 0.5);   // the head voxel's top
        // quick reject: the flame and its embers' column
        {
            vec3 lo = base - vec3(1.4, 0.2, 1.4), hi = base + vec3(1.4, 9.0, 1.4);
            vec3 t0 = (lo - ro) * inv, t1 = (hi - ro) * inv;
            vec3 tn = min(t0, t1), tf = max(t0, t1);
            if (max(max(max(tn.x, tn.y), tn.z), 0.0) >= min(min(min(tf.x, tf.y), tf.z), tMax)) continue;
        }
        float flick = lightAnim(3.0, ph, t, 1.0);   // the light's flicker, 0.86 .. 1.14
        float grow = 0.75 + 1.8 * (flick - 0.86);   // 0.75 .. 1.25: the flame's size follows it
        float tBest = tMax, shade;
        vec3 col = vec3(0.0);
        // the core: a steady cube on the head, slowly turning
        mat3 Rc = tfRot(vec3(0.35, t * 0.7 + seed, 0.25));
        if (tfCube(ro, rd, base + vec3(0.0, 0.28, 0.0), 0.34 * (0.9 + 0.1 * grow), Rc, tBest, shade))
            col = vec3(1.9, 1.15, 0.32) * shade;
        // the stream: cube k is at age u (0 born on the head .. 1 gone), every life a new cube
        for (int k = 0; k < kN; k++) {
            float life = 0.75 + 0.25 * tfHash(vec3(seed, float(k), 5.0));
            float cyc = t / life + float(k) / float(kN) + tfHash(vec3(seed, float(k), 6.0));
            float u = fract(cyc), id = floor(cyc);
            vec3 r = vec3(tfHash(vec3(seed, id, float(k))), tfHash(vec3(float(k), seed, id)), tfHash(vec3(id, float(k), seed)));
            float h = mix(0.42, 0.06, u) * (0.8 + 0.4 * r.z) * grow;
            vec3 drift = vec3(r.x - 0.5, 0.0, r.y - 0.5) * (0.15 + 0.5 * u);
            vec3 c = base + drift + vec3(0.0, 0.2 + u * 2.2 * grow, 0.0);
            mat3 R = tfRot(r * 6.2831853 + u * vec3(2.1, 2.9, 1.7) * (r.x < 0.5 ? -1.0 : 1.0));
            if (!tfCube(ro, rd, c, h, R, tBest, shade)) continue;
            vec3 hot = mix(vec3(1.9, 1.1, 0.28), vec3(1.8, 0.55, 0.08), smoothstep(0.0, 0.45, u));
            col = mix(hot, vec3(1.1, 0.16, 0.03), smoothstep(0.45, 1.0, u)) * shade;
        }
        sum += col * flick;
        // embers: 4 small turned cubes rising ~0.9 m over ~2 s, drifting, fading
        for (int e = 0; e < 4; e++) {
            float es = seed + float(e) * 17.3;
            float rate = 0.4 + 0.25 * tfHash(vec3(es, 1.0, 0.0));
            float cyc = t * rate + tfHash(vec3(es, 2.0, 0.0));
            float life = fract(cyc), id = floor(cyc);
            vec3 drift = vec3(tfHash(vec3(es, id, 1.0)) - 0.5, 0.0, tfHash(vec3(es, id, 2.0)) - 0.5);
            vec3 pe = base + vec3(0.0, 1.6 + life * 7.5, 0.0) + drift * (0.4 + 2.2 * life);
            mat3 Re = tfRot(vec3(life * 7.0 + es, life * 5.0, es * 0.37));
            float te = tMax;
            if (tfCube(ro, rd, pe, 0.11 * (1.0 - 0.5 * life), Re, te, shade))
                sum += vec3(1.8, 0.5, 0.07) * shade * (1.0 - life);
        }
    }
    return sum;
}
