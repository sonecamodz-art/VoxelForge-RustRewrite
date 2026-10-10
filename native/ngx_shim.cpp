// A thin C interface over the DLSS SDK's inline helpers (nvsdk_ngx_helpers*.h), for
// src/gfx/dlss/ngx.rs. The helpers set the NGX parameters each feature reads; calling them
// here instead of re-implementing ~150 parameter names in Rust keeps us on NVIDIA's code.
// Compiled by build.rs only with `--features dlss`.
#include <filesystem>
#include <string>

#include <vulkan/vulkan.h>

// (this order: the dlssd helpers use the plain VK helpers' types and macros)
#include "nvsdk_ngx.h"
#include "nvsdk_ngx_vk.h"
#include "nvsdk_ngx_helpers.h"
#include "nvsdk_ngx_helpers_vk.h"
#include "nvsdk_ngx_helpers_dlssd.h"
#include "nvsdk_ngx_helpers_dlssd_vk.h"

extern "C" {

struct VfNgxImg {
    VkImage image;
    VkImageView view;
    VkFormat format;
    unsigned width, height;
};

static NVSDK_NGX_Resource_VK wrap(const VfNgxImg* g, bool rw) {
    VkImageSubresourceRange range{VK_IMAGE_ASPECT_COLOR_BIT, 0, 1, 0, 1};
    return NVSDK_NGX_Create_ImageView_Resource_VK(g->view, g->image, range, g->format, g->width, g->height, rw);
}

static NVSDK_NGX_PerfQuality_Value perf(int q) {
    switch (q) {
        case 1: return NVSDK_NGX_PerfQuality_Value_MaxQuality;
        case 2: return NVSDK_NGX_PerfQuality_Value_Balanced;
        case 3: return NVSDK_NGX_PerfQuality_Value_MaxPerf;
        case 4: return NVSDK_NGX_PerfQuality_Value_UltraPerformance;
        default: return NVSDK_NGX_PerfQuality_Value_DLAA;
    }
}

int vf_ngx_required_extensions(unsigned* ic, const char*** ie, unsigned* dc, const char*** de) {
    return (int)NVSDK_NGX_VULKAN_RequiredExtensions(ic, ie, dc, de);
}

// Paths are UTF-8. On success *params = the capability parameters, *rr / *sr = availability.
int vf_ngx_init(const char* project_id, const char* snippet_dir, const char* data_dir, VkInstance inst, VkPhysicalDevice pd,
                VkDevice dev, PFN_vkGetInstanceProcAddr gipa, PFN_vkGetDeviceProcAddr gdpa, void** params, unsigned* rr,
                unsigned* sr) {
    namespace fs = std::filesystem;
    std::wstring snip = fs::u8path(snippet_dir).wstring(), data = fs::u8path(data_dir).wstring();
    const wchar_t* paths[] = {snip.c_str()};
    NVSDK_NGX_FeatureCommonInfo fci{};
    fci.PathListInfo.Path = paths;
    fci.PathListInfo.Length = 1;
    fci.LoggingInfo.MinimumLoggingLevel = NVSDK_NGX_LOGGING_LEVEL_OFF;   // (NGX writes its own logs into data_dir)
    NVSDK_NGX_Result r = NVSDK_NGX_VULKAN_Init_with_ProjectID(project_id, NVSDK_NGX_ENGINE_TYPE_CUSTOM, "0.1", data.c_str(), inst, pd,
                                                              dev, gipa, gdpa, &fci);
    if (NVSDK_NGX_FAILED(r)) return (int)r;
    NVSDK_NGX_Parameter* p = nullptr;
    r = NVSDK_NGX_VULKAN_GetCapabilityParameters(&p);
    if (NVSDK_NGX_FAILED(r) || !p) return (int)(NVSDK_NGX_FAILED(r) ? r : NVSDK_NGX_Result_Fail);
    *params = p;
    *rr = *sr = 0;
    NVSDK_NGX_Parameter_GetUI(p, NVSDK_NGX_Parameter_SuperSamplingDenoising_Available, rr);
    NVSDK_NGX_Parameter_GetUI(p, NVSDK_NGX_Parameter_SuperSampling_Available, sr);
    return (int)NVSDK_NGX_Result_Success;
}

// The render size NGX picks for a display size and quality (rr: Ray Reconstruction's table).
int vf_ngx_optimal(void* params, int rr, unsigned tw, unsigned th, int q, unsigned* rw, unsigned* rh) {
    auto* p = (NVSDK_NGX_Parameter*)params;
    unsigned maxW = 0, maxH = 0, minW = 0, minH = 0;
    float sharp = 0.0f;
    *rw = *rh = 0;
    NVSDK_NGX_Result r = rr ? NGX_DLSSD_GET_OPTIMAL_SETTINGS(p, tw, th, perf(q), rw, rh, &maxW, &maxH, &minW, &minH, &sharp)
                            : NGX_DLSS_GET_OPTIMAL_SETTINGS(p, tw, th, perf(q), rw, rh, &maxW, &maxH, &minW, &minH, &sharp);
    return (int)r;
}

// Ray Reconstruction: HDR, auto exposure, motion at render resolution, linear depth, roughness
// packed in the normals' w.
int vf_ngx_create_rr(VkDevice dev, VkCommandBuffer cb, void* params, unsigned rw, unsigned rh, unsigned tw, unsigned th, int q,
                     void** handle, unsigned long long* vram) {
    auto* p = (NVSDK_NGX_Parameter*)params;
    NVSDK_NGX_DLSSD_Create_Params cp{};
    cp.InDenoiseMode = NVSDK_NGX_DLSS_Denoise_Mode_DLUnified;
    cp.InRoughnessMode = NVSDK_NGX_DLSS_Roughness_Mode_Packed;
    cp.InUseHWDepth = NVSDK_NGX_DLSS_Depth_Type_Linear;
    cp.InWidth = rw;
    cp.InHeight = rh;
    cp.InTargetWidth = tw;
    cp.InTargetHeight = th;
    cp.InPerfQualityValue = perf(q);
    cp.InFeatureCreateFlags = NVSDK_NGX_DLSS_Feature_Flags_IsHDR | NVSDK_NGX_DLSS_Feature_Flags_AutoExposure |
                              NVSDK_NGX_DLSS_Feature_Flags_MVLowRes;
    cp.InEnableOutputSubrects = false;
    NVSDK_NGX_Handle* h = nullptr;
    NVSDK_NGX_Result r = NGX_VULKAN_CREATE_DLSSD_EXT1(dev, cb, 1, 1, &h, p, &cp);
    if (NVSDK_NGX_FAILED(r)) return (int)r;
    *handle = h;
    *vram = 0;
    NGX_DLSSD_GET_STATS(p, vram);
    return (int)r;
}

// Super Resolution / DLAA of an already clean image: HDR, motion at render resolution,
// reverse-Z device depth, auto exposure.
int vf_ngx_create_sr(VkDevice dev, VkCommandBuffer cb, void* params, unsigned rw, unsigned rh, unsigned tw, unsigned th, int q,
                     void** handle, unsigned long long* vram) {
    auto* p = (NVSDK_NGX_Parameter*)params;
    NVSDK_NGX_DLSS_Create_Params cp{};
    cp.Feature.InWidth = rw;
    cp.Feature.InHeight = rh;
    cp.Feature.InTargetWidth = tw;
    cp.Feature.InTargetHeight = th;
    cp.Feature.InPerfQualityValue = perf(q);
    cp.InFeatureCreateFlags = NVSDK_NGX_DLSS_Feature_Flags_IsHDR | NVSDK_NGX_DLSS_Feature_Flags_MVLowRes |
                              NVSDK_NGX_DLSS_Feature_Flags_DepthInverted | NVSDK_NGX_DLSS_Feature_Flags_AutoExposure;
    NVSDK_NGX_Handle* h = nullptr;
    NVSDK_NGX_Result r = NGX_VULKAN_CREATE_DLSS_EXT1(dev, cb, 1, 1, &h, p, &cp);
    if (NVSDK_NGX_FAILED(r)) return (int)r;
    *handle = h;
    *vram = 0;
    NGX_DLSS_GET_STATS(p, vram);
    return (int)r;
}

int vf_ngx_eval_rr(VkCommandBuffer cb, void* handle, void* params, const VfNgxImg* color, const VfNgxImg* diffuse,
                   const VfNgxImg* specular, const VfNgxImg* normal_rough, const VfNgxImg* depth, const VfNgxImg* motion,
                   const VfNgxImg* output, int reset, float jx, float jy, float mv_scale_x, float mv_scale_y) {
    NVSDK_NGX_Resource_VK colorR = wrap(color, false), outR = wrap(output, true), diffR = wrap(diffuse, false),
                          specR = wrap(specular, false), normR = wrap(normal_rough, false), depthR = wrap(depth, false),
                          motR = wrap(motion, false);
    NVSDK_NGX_VK_DLSSD_Eval_Params ep{};
    ep.pInColor = &colorR;
    ep.pInOutput = &outR;
    ep.pInDiffuseAlbedo = &diffR;
    ep.pInSpecularAlbedo = &specR;
    ep.pInNormals = &normR;
    ep.pInRoughness = nullptr;   // packed in the normals' w
    ep.pInDepth = &depthR;
    ep.pInMotionVectors = &motR;
    ep.InRenderSubrectDimensions.Width = color->width;
    ep.InRenderSubrectDimensions.Height = color->height;
    ep.InJitterOffsetX = jx;
    ep.InJitterOffsetY = jy;
    ep.InReset = reset;
    ep.InMVScaleX = mv_scale_x;
    ep.InMVScaleY = mv_scale_y;
    return (int)NGX_VULKAN_EVALUATE_DLSSD_EXT(cb, (NVSDK_NGX_Handle*)handle, (NVSDK_NGX_Parameter*)params, &ep);
}

int vf_ngx_eval_sr(VkCommandBuffer cb, void* handle, void* params, const VfNgxImg* color, const VfNgxImg* hw_depth,
                   const VfNgxImg* motion, const VfNgxImg* output, int reset, float jx, float jy, float mv_scale_x,
                   float mv_scale_y) {
    NVSDK_NGX_Resource_VK colorR = wrap(color, false), outR = wrap(output, true), depthR = wrap(hw_depth, false),
                          motR = wrap(motion, false);
    NVSDK_NGX_VK_DLSS_Eval_Params ep{};
    ep.Feature.pInColor = &colorR;
    ep.Feature.pInOutput = &outR;
    ep.pInDepth = &depthR;
    ep.pInMotionVectors = &motR;
    ep.InJitterOffsetX = jx;
    ep.InJitterOffsetY = jy;
    ep.InRenderSubrectDimensions.Width = color->width;
    ep.InRenderSubrectDimensions.Height = color->height;
    ep.InReset = reset;
    ep.InMVScaleX = mv_scale_x;
    ep.InMVScaleY = mv_scale_y;
    return (int)NGX_VULKAN_EVALUATE_DLSS_EXT(cb, (NVSDK_NGX_Handle*)handle, (NVSDK_NGX_Parameter*)params, &ep);
}

void vf_ngx_release(void* handle) {
    if (handle) NVSDK_NGX_VULKAN_ReleaseFeature((NVSDK_NGX_Handle*)handle);
}

void vf_ngx_shutdown(VkDevice dev) { NVSDK_NGX_VULKAN_Shutdown1(dev); }

}  // extern "C"
