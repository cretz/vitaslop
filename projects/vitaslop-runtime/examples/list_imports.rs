//! List a guest module's function imports, named where the NID registry knows them.
//!
//! `cargo run -p vitaslop-runtime --example list_imports -- eboot.bin [filter]`
//!
//! Answers "what does this title actually call from library X" without booting it, which
//! is the difference between implementing a call surface and discovering it one fatal
//! unimplemented-NID at a time.
fn main() {
    let mut args = std::env::args().skip(1);
    let path = args.next().expect("usage: list_imports <module> [name filter]");
    let filter = args.next().unwrap_or_default().to_lowercase();
    let bytes = std::fs::read(&path).expect("read the module");
    let module = vitaslop_loader::load(&bytes).expect("parse the module");

    let mut rows: Vec<(u32, u32, &'static str)> = module
        .imports
        .iter()
        .map(|i| (i.library_nid, i.func_nid, vitaslop_runtime::nid::name(i.func_nid)))
        .collect();
    rows.sort_by_key(|(lib, nid, _)| (*lib, *nid));
    rows.dedup();

    let mut shown = 0usize;
    for (lib, nid, name) in &rows {
        if filter.is_empty() || name.to_lowercase().contains(&filter) {
            println!("lib {lib:#010x}  nid {nid:#010x}  {name}");
            shown += 1;
        }
    }
    eprintln!("{shown} of {} imports shown", rows.len());
}
