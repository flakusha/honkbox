//! Build script: ensure `web-dist/` exists so `rust-embed` always has a dir.
fn main() {
    println!("cargo::rerun-if-changed=../../web-dist");
    std::fs::create_dir_all("../../web-dist").expect("create web-dist");
}
