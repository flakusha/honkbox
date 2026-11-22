//! Compile GLSL to SPIR-V at build time so the binary embeds no runtime
//! compiler dependency. Only the `ash-compute` feature needs the output.

use std::{env, path::PathBuf, process::Command};

fn main() {
    let shaders = [
        ("shaders/elementwise_add.comp", "elementwise_add.spv"),
        ("shaders/matmul.comp", "matmul.spv"),
        ("shaders/softmax.comp", "softmax.spv"),
        ("shaders/layer_norm.comp", "layer_norm.spv"),
        ("shaders/conv2d.comp", "conv2d.spv"),
        ("shaders/silu.comp", "silu.spv"),
        ("shaders/group_norm.comp", "group_norm.spv"),
        ("shaders/conv_transpose.comp", "conv_transpose.spv"),
        ("shaders/ddim_step.comp", "ddim_step.spv"),
        ("shaders/cfg_guidance.comp", "cfg_guidance.spv"),
    ];
    for (source, _) in shaders {
        println!("cargo:rerun-if-changed={source}");
    }
    println!("cargo:rerun-if-env-changed=CARGO_FEATURE_ASH_COMPUTE");
    if env::var_os("CARGO_FEATURE_ASH_COMPUTE").is_none() {
        return;
    }

    let out = PathBuf::from(env::var_os("OUT_DIR").expect("cargo sets OUT_DIR"));
    for (source, output) in shaders {
        let status = Command::new("glslangValidator")
            .args([
                "-V",
                "-S",
                "comp",
                "--target-env",
                "vulkan1.3",
                source,
                "-o",
            ])
            .arg(out.join(output))
            .status()
            .expect("glslangValidator required by ash-compute; install glslang-tools");
        assert!(status.success(), "glslangValidator failed on {source}");
    }
}
