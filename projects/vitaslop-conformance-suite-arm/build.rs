//! Rebuild the embedded corpus whenever a case is added, edited or removed.
//!
//! `include_dir!` cannot tell cargo which files it read on stable Rust, so without this a NEW
//! case file left the crate untouched and the harness ran the old corpus - green, and silent
//! about the case it never loaded. Cargo walks a directory named here recursively.
fn main() {
    println!("cargo:rerun-if-changed=cases");
}
