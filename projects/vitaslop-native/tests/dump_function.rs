//! Print the lowered IR of guest functions of a title, and the raw words at their addresses,
//! without booting it - for reading a trap's `guest_block` against the statements that ran.
//!
//!   VITASLOP_GAME_DIR=<extracted app dir> VITASLOP_DUMP_FUNC=<hex>[,<hex>...] \
//!   cargo test -p vitaslop-native --test dump_function -- --ignored --nocapture
//!
//! An address is lifted in Thumb unless that fails, then in ARM; the header says which.

use vitaslop_loader as loader;
use vitaslop_runtime::ingest::pipeline::decrypt_container;
use vitaslop_runtime::ingest::vfs::DirVfs;
use vitaslop_runtime::link::link;

#[test]
#[ignore = "needs VITASLOP_GAME_DIR and VITASLOP_DUMP_FUNC"]
fn dump_function() {
    let (Ok(dir), Ok(list)) = (std::env::var("VITASLOP_GAME_DIR"), std::env::var("VITASLOP_DUMP_FUNC")) else {
        eprintln!("VITASLOP_GAME_DIR / VITASLOP_DUMP_FUNC unset: skipping");
        return;
    };
    let game = decrypt_container(&mut DirVfs::new(&dir)).expect("decrypt");
    let modules: Vec<loader::Module> = game.modules.iter().map(|m| loader::load(&m.elf).expect("load")).collect();
    let linked = link(modules).expect("link");
    let program = linked.shared_program();
    for tok in list.split(',') {
        let addr = u32::from_str_radix(tok.trim().trim_start_matches("0x"), 16).expect("hex address");
        let off = addr.wrapping_sub(linked.base) as usize;
        let words: Vec<String> = (0..16)
            .filter_map(|i| linked.image.get(off + i * 4..off + i * 4 + 4))
            .map(|w| format!("{:08x}", u32::from_le_bytes(w.try_into().unwrap())))
            .collect();
        eprintln!("== words at {addr:#010x}: {}", words.join(" "));
        match vitaslop_transpiler::dump_func(&program, addr) {
            Some(text) => eprint!("{text}"),
            None => eprintln!("no function decodes at {addr:#x}"),
        }
    }
    if std::env::var("VITASLOP_DUMP_IMPORTS").is_ok() {
        for (i, (lib, func)) in linked.imports.iter().enumerate() {
            eprintln!("import {i}: {lib:08x}/{func:08x} {}", vitaslop_runtime::nid::name(*func));
        }
    }
}
