//! TOOL: write the loaded segment that holds a guest address to a file, for offline disassembly.
//!
//! `VITASLOP_GAME_DIR=<extracted dir> VITASLOP_DUMP_ADDR=0x81385eae VITASLOP_DUMP_OUT=<file>
//! [VITASLOP_MAIN_EXEC=app0:<self>] cargo test -p vitaslop-native --test dump_segment -- --ignored --nocapture`
//! Prints every module's segments; writes `<vaddr as 8 hex><bytes>` (4-byte BE vaddr header).

use vitaslop_runtime::ingest::pipeline::decrypt_container;
use vitaslop_runtime::ingest::vfs::DirVfs;

#[test]
#[ignore = "needs VITASLOP_GAME_DIR"]
fn dump_segment() {
    let Ok(dir) = std::env::var("VITASLOP_GAME_DIR") else {
        eprintln!("dump_segment: VITASLOP_GAME_DIR not set - skipped");
        return;
    };
    let addr = std::env::var("VITASLOP_DUMP_ADDR")
        .ok()
        .and_then(|v| u32::from_str_radix(v.trim_start_matches("0x"), 16).ok())
        .expect("VITASLOP_DUMP_ADDR=0x...");
    let out = std::env::var("VITASLOP_DUMP_OUT").expect("VITASLOP_DUMP_OUT=<file>");
    let mut game = decrypt_container(&mut DirVfs::new(&dir)).expect("decrypt container");
    if let Ok(exec) = std::env::var("VITASLOP_MAIN_EXEC") {
        assert!(game.with_main_exec(&exec), "no such executable {exec}");
    }
    for m in &game.modules {
        let module = vitaslop_loader::load(&m.elf).expect("load module");
        for s in &module.segments {
            eprintln!(
                "{} seg {:08x}..{:08x} exec {}",
                module.name,
                s.vaddr,
                s.vaddr + s.mem_size,
                s.executable
            );
            if addr >= s.vaddr && addr < s.vaddr + s.data.len() as u32 {
                let mut bytes = s.vaddr.to_be_bytes().to_vec();
                bytes.extend_from_slice(&s.data);
                std::fs::write(&out, bytes).expect("write");
                eprintln!("dump_segment: wrote {} bytes of {} from {:08x}", s.data.len(), module.name, s.vaddr);
            }
        }
    }
}
