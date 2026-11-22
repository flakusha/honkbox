//! Compile GLSL to SPIR-V at build time so the binary embeds no runtime
//! compiler dependency. Only the `ash-compute` feature needs the output.

use std::{env, path::PathBuf, process::Command};

fn main() {
    println!("cargo:rerun-if-changed=shaders/elementwise_add.comp");
    println!("cargo:rerun-if-env-changed=CARGO_FEATURE_ASH_COMPUTE");
    if env::var_os("CARGO_FEATURE_ASH_COMPUTE").is_none() {
        return;
    }

    let out = PathBuf::from(env::var_os("OUT_DIR").expect("cargo sets OUT_DIR"));
    let status = Command::new("glslangValidator")
        .args([
            "-V",
            "-S",
            "comp",
            "--target-env",
            "vulkan1.3",
            "shaders/elementwise_add.comp",
            "-o",
        ])
        .arg(out.join("elementwise_add.spv"))
        .status()
        .expect("glslangValidator required by ash-compute; install glslang-tools");
    assert!(
        status.success(),
        "glslangValidator failed on shaders/elementwise_add.comp"
    );
}
