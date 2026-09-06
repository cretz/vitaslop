//! Name the import whose STUB lives at a given guest address.
//!
//! `cargo run -p vitaslop-runtime --example import_at -- <module> <stub_addr_hex> [span]`
//!
//! A `blx` into the stub table is the only trace an import leaves in a disassembly, and the
//! table carries no names: every entry is the same four words. Reading a call site therefore
//! stops at "it calls SOMETHING imported" unless the stub address can be turned back into a
//! (library, NID) pair - which the loader already knows and nothing printed.
fn main() {
    let mut args = std::env::args().skip(1);
    let path = args.next().expect("usage: import_at <module> <stub_addr> [span]");
    let addr = u32::from_str_radix(
        args.next().expect("a stub address").trim_start_matches("0x"),
        16,
    )
    .expect("hex address");
    let span: u32 = args.next().and_then(|v| v.parse().ok()).unwrap_or(16);

    let bytes = std::fs::read(&path).expect("read the module");
    let module = vitaslop_loader::load(&bytes).expect("parse the module");

    let mut hits = 0usize;
    for i in &module.imports {
        if i.stub_addr >= addr.saturating_sub(span) && i.stub_addr <= addr + span {
            println!(
                "stub {:#010x}  lib {:#010x}  nid {:#010x}  {}",
                i.stub_addr,
                i.library_nid,
                i.func_nid,
                vitaslop_runtime::nid::name(i.func_nid)
            );
            hits += 1;
        }
    }
    if hits == 0 {
        println!("no import stub within {span} bytes of {addr:#010x}");
        let mut near: Vec<u32> = module.imports.iter().map(|i| i.stub_addr).collect();
        near.sort_unstable();
        if let (Some(lo), Some(hi)) = (near.first(), near.last()) {
            println!("stub table spans {lo:#010x}..={hi:#010x} ({} imports)", near.len());
        }
    }
}
