//! >>> THE TRANSPILE'S MEMORY PEAK AND ITS OUTPUT, FOR ONE TITLE, WITH NO EMULATOR.
//!
//! The lenient transpile is the allocation peak of a whole boot (a football title: 1,991 MB of Rust heap
//! for 7.6 M guest instructions, inside a browser worker whose ceiling is 4,096 MB). A change
//! that lowers that peak must not move a single emitted byte, and this probe answers both
//! halves in seconds: it prints the peak, the time, and a hash of the module.
//!
//! ```text
//! VITASLOP_GAME_DIR=<decrypted app dir> cargo test --release -p vitaslop-native \
//!   --test transpile_peak -- --ignored --nocapture
//! ```

use std::hash::Hasher;

use vitaslop_runtime::ingest::pipeline::decrypt_container;
use vitaslop_runtime::ingest::vfs::DirVfs;
use vitaslop_runtime::link::link;

#[global_allocator]
static ALLOC: vitaslop_platform::heap::Counting<std::alloc::System> =
    vitaslop_platform::heap::Counting(std::alloc::System);

#[test]
#[ignore = "needs VITASLOP_GAME_DIR"]
fn transpile_peak() {
    let Ok(dir) = std::env::var("VITASLOP_GAME_DIR") else {
        eprintln!("transpile_peak: VITASLOP_GAME_DIR not set - skipped");
        return;
    };
    let game = decrypt_container(&mut DirVfs::new(&dir)).expect("decrypt container");
    let modules: Vec<_> =
        game.modules.iter().map(|m| vitaslop_loader::load(&m.elf).expect("load module")).collect();
    let linked = link(modules).expect("link");
    let program = linked.shared_program();
    vitaslop_platform::heap::reset_peak();
    let (live0, _) = vitaslop_platform::heap::live_peak_mb();
    let t = std::time::Instant::now();
    let built = vitaslop_transpiler::transpile_lenient(&program);
    let ms = t.elapsed().as_millis();
    let (live, peak) = vitaslop_platform::heap::live_peak_mb();
    let mut h = std::collections::hash_map::DefaultHasher::new();
    h.write(&built.artifact.wasm);
    eprintln!(
        "transpile_peak: {ms} ms, heap before {live0} MB, PEAK {peak} MB, after {live} MB; module {} bytes hash {:016x}, {} funcs, {} stubs",
        built.artifact.wasm.len(),
        h.finish(),
        built.artifact.funcs.len(),
        built.stubbed.len(),
    );
}
