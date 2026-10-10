// Transparent water's mirror reflection (water_refl.comp; water_refl_half.comp traces it once per
// 2x2 block with r.water.reflect.half): one mirror ray off the wave normal, traced through the
// scene (terrain, trees, bodies; water skipped) and shaded cheaply at the hit - its own sun ray
// (Standard: the cached shadow map) + sky ambient x contact AO + near lamps. Sky if it escapes.
// Included before the pass uniforms: reach (r.water.reflect.dist) and the clock come as arguments.
// Also the glossy surfaces' reflections (r.reflect: metals, polished / glossy blocks) - the same ray, any
// direction (reflRadianceDir), their hits' sky light kept out of covered places (skyOpen).
#if defined(WATER_CLEAR) || defined(REFL_GLOSSY)
vec3 skyAmbient(vec3 N) {
    // Hemisphere sky light for a normal: the sky toward N, dimmed toward the horizon.
    // (N + up) is zero for a downward face (an overhang / leaf underside seen in the
    // reflection): normalize(0) was NaN -> black swirls in SVGF, white blobs through RR.
    vec3 d = N + vec3(0.0, 1.001, 0.0);
    return skyColor(d * inversesqrt(max(dot(d, d), 1e-6)), 0.0) * (0.55 + 0.25 * N.y) * giScale;
}

vec3 reflRadianceDir(vec3 origin, vec3 R, bool useMap, float reach, float time, bool water) {
    vec3 refl;
    Hit r = traceSceneRefl(origin, R, reach, kSkipMat);
    if (!r.hit) {
        refl = skyWithClouds(R) * skyOpen(origin + R * reach);   // (the clouds too)
    } else {
        Material m = materials[r.mat];
        vec3 Pr = origin + R * r.t;
        // A ray that starts inside the next voxel (a reflection off a staircase step, e.g. a body's sphere) hits at t 0
        // with no face normal: NaN in the AO / sky terms below - each a glowing, flickering blob through RR.
        vec3 Nr = dot(r.normal, r.normal) > 0.5 ? r.normal : -R;
        float ndl = max(dot(Nr, sunDir), 0.0);
        float vis = ndl <= 0.0 ? 0.0 : useMap ? sunShadowHard(Pr + Nr * 0.5)
                                              : (!occludedShadow(Pr + Nr * 0.03, sunDir, 1e30, kSkipMat) ? 1.0 : 0.0);
        vec3 a = m.albedoEmissive.rgb;
        gSelfBody = r.obj;
        // An emitter shows its own radiance (albedo x emissive, as in the direct view;
        // this was x8), animated like gbuffer.comp's.
        float emit = m.albedoEmissive.a;
        if (emit > 0.0 && (m.props.w > 0.5 || r.obj >= 0))
            emit *= lightAnim(m.props.w, r.obj >= 0 ? lightPhaseBody(r.obj) : lightPhaseAt(vec3(r.cell) + 0.5),
                              time, r.obj >= 0 ? objs[r.obj].aabbMin.w : 1.0);
        vec3 lamps = vec3(0.0);
#ifdef LIGHTS_NEE
        // Lamp-lit shore / walls show in the water - near lamps only: every light kept is a
        // shadow ray per reflected pixel (26 far torches cost ~8 ms on a lake view).
        lamps = lightsDirect(Pr, Nr, kLightMin * 10.0);
#endif
        vec3 amb = skyAmbient(Nr) * voxelAO(Pr, Nr);
        if (!water) amb *= skyOpen(Pr + Nr * 1.5);   // (a reflected room isn't lit like the outdoors)
        refl = a * (sunRGB * ndl * vis + amb + lamps) + a * emit;
        // Distant reflections fade into the haze like the direct view would.
        refl = mix(refl, skyWithClouds(R), 1.0 - exp(-r.t * 0.0015));
    }
    return refl;
}
vec3 reflRadiance(vec3 origin, vec3 rd, vec3 Nw, bool useMap, float reach, float time) {
    vec3 R = reflect(rd, Nw);
    R = normalize(vec3(R.x, max(R.y, 0.02), R.z));   // waves never reflect the underside
    return reflRadianceDir(origin, R, useMap, reach, time, true);
}
#endif
