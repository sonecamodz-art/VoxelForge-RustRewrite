//! Optional NVIDIA SDKs. Without features this does nothing.
//!
//!   --features dlss   DLSS (NGX): the SDK at $VF_DLSS_SDK or external/DLSS
//!                     (git clone https://github.com/NVIDIA/DLSS external/DLSS). A small C++ shim
//!                     (native/ngx_shim.cpp) calls the SDK's inline helpers; libnvsdk_ngx is linked
//!                     statically. The runtime libraries (libnvidia-ngx-dlss*.so / nvngx_dlss*.dll)
//!                     are found at run time (see src/gfx/dlss).
//!   --features nrd    NRD: see build_nrd below.

use std::env;
use std::path::{Path, PathBuf};

fn main() {
    println!("cargo:rerun-if-changed=build.rs");
    let root = PathBuf::from(env::var("CARGO_MANIFEST_DIR").unwrap());
    if env::var_os("CARGO_FEATURE_DLSS").is_some() {
        build_dlss(&root);
    }
    if env::var_os("CARGO_FEATURE_NRD").is_some() {
        build_nrd(&root);
    }
}

/// $var if set, else root/external/<name>; must contain `probe`.
fn sdk_dir(root: &Path, var: &str, name: &str, probe: &str, how: &str) -> PathBuf {
    println!("cargo:rerun-if-env-changed={var}");
    let dir = env::var_os(var).map(PathBuf::from).unwrap_or_else(|| root.join("external").join(name));
    if !dir.join(probe).exists() {
        panic!("\n\n{name} SDK not found at {} (no {probe}).\n{how}\nor point {var} at it, or build without --features {}.\n\n",
               dir.display(), name.to_lowercase());
    }
    dir.canonicalize().unwrap_or(dir)
}

/// Vulkan headers for the C++ shims: the system's, or the Vulkan SDK's ($VULKAN_SDK/Include).
fn vulkan_include(b: &mut cc::Build) {
    println!("cargo:rerun-if-env-changed=VULKAN_SDK");
    if let Some(sdk) = env::var_os("VULKAN_SDK") {
        for d in ["Include", "include"] {
            let p = PathBuf::from(&sdk).join(d);
            if p.exists() {
                b.include(p);
            }
        }
    }
}

fn build_dlss(root: &Path) {
    let sdk = sdk_dir(root, "VF_DLSS_SDK", "DLSS", "include/nvsdk_ngx.h", "  git clone --depth 1 https://github.com/NVIDIA/DLSS external/DLSS");
    println!("cargo:rerun-if-changed=native/ngx_shim.cpp");
    let mut b = cc::Build::new();
    b.cpp(true).std("c++17").warnings(false).include(sdk.join("include")).file(root.join("native/ngx_shim.cpp"));
    vulkan_include(&mut b);
    b.compile("vf_ngx_shim");
    let windows = env::var("CARGO_CFG_TARGET_OS").as_deref() == Ok("windows");
    if windows {
        // nvsdk_ngx_d: the dynamic-CRT (/MD) build, as Rust links the CRT.
        println!("cargo:rustc-link-search=native={}", sdk.join("lib/Windows_x86_64/x64").display());
        println!("cargo:rustc-link-lib=static=nvsdk_ngx_d");
    } else {
        println!("cargo:rustc-link-search=native={}", sdk.join("lib/Linux_x86_64").display());
        println!("cargo:rustc-link-lib=static=nvsdk_ngx");
        println!("cargo:rustc-link-lib=dylib=dl");
    }
    // Where the runtime libraries are found by default (src/gfx/dlss).
    let rel = if windows { "lib/Windows_x86_64/rel" } else { "lib/Linux_x86_64/rel" };
    println!("cargo:rustc-env=VF_DLSS_SNIPPETS={}", sdk.join(rel).display());
}

fn build_nrd(_root: &Path) {
    panic!("--features nrd: not wired up yet");
}
