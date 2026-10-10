// Display mapping shared by tonemap.comp (DLSS/raw paths) and denoise.comp (SVGF
// path): exposed linear HDR -> display (gamma-encoded) RGB. No uniforms here.
//   op 0 = ACES (Narkowicz fit; the original look), 1 = AgX (Sobotka; natural
//   highlight roll-off, bright sun/fog desaturate instead of skewing hue).
//   sat / con = saturation / contrast around mid-grey, applied in display space;
//   wb = white balance (r.warmth) on the exposed HDR.

vec3 aces(vec3 x) {
    return clamp((x * (2.51 * x + 0.03)) / (x * (2.43 * x + 0.59) + 0.14), 0.0, 1.0);
}

vec3 agxContrast(vec3 x) {   // 6th-order fit of the AgX default contrast curve
    vec3 x2 = x * x, x4 = x2 * x2;
    return 15.5 * x4 * x2 - 40.14 * x4 * x + 31.96 * x4 - 6.868 * x2 * x + 0.4298 * x2 +
           0.1191 * x - 0.00232;
}
vec3 agx(vec3 v) {
    const mat3 inset = mat3(0.842479062253094, 0.0423282422610123, 0.0423756549057051,
                            0.0784335999999992, 0.878468636469772, 0.0784336,
                            0.0792237451477643, 0.0791661274605434, 0.879142973793104);
    const float minEv = -12.47393, maxEv = 4.026069;
    v = inset * max(v, vec3(1e-10));
    v = clamp(log2(v), minEv, maxEv);
    v = (v - minEv) / (maxEv - minEv);
    v = agxContrast(v);
    // Outset back toward the display primaries (AgX "base" look), display-encoded.
    const mat3 outset = mat3(1.19687900512017, -0.0528968517574562, -0.0529716355144438,
                             -0.0980208811401368, 1.15190312990417, -0.0980434501171241,
                             -0.0990297440797205, -0.0989611768448433, 1.15107367264116);
    return clamp(outset * v, 0.0, 1.0);   // (EOTF 2.2 then re-encode 1/2.2 cancel)
}

vec3 displayMap(vec3 hdrExposed, int op, float sat, float con, vec3 wb) {
    if (wb != vec3(1.0)) hdrExposed *= wb;
    vec3 c = op == 1 ? agx(hdrExposed) : pow(aces(hdrExposed), vec3(1.0 / 2.2));
    if (sat != 1.0) c = clamp(mix(vec3(dot(c, vec3(0.2126, 0.7152, 0.0722))), c, sat), 0.0, 1.0);
    if (con != 1.0) c = clamp((c - 0.5) * con + 0.5, 0.0, 1.0);
    return c;
}
