//! `playbook` - the tools that need no game running.
//!
//! ```text
//! playbook check [<file.lua> ...]   compile playbooks (default: every committed one)
//! playbook show <tape>              a tape's timeline: phases, splits, logs
//! playbook diff <a.tape> <b.tape>   how far each timeline note moved between two runs
//! ```
//!
//! A playbook RUNS through the desktop headless runner (`vitaslop --game <dir> --headless
//! <out> --playbook <file.lua>`): see the crate README.
use std::path::PathBuf;

use vitaslop_playbook::all_playbooks;
use vitaslop_playbook::tape::{diff_timelines, Tape};

fn read_tape(path: &str) -> Result<Tape, String> {
    let text = std::fs::read_to_string(path).map_err(|e| format!("{path}: {e}"))?;
    Tape::parse(&text).map_err(|e| format!("{path}: {e}"))
}

/// Compiling needs a Lua, and the only one this crate carries is native (a wasm32 build of
/// this tool exists only because a one-invocation wasm clippy builds every target).
#[cfg(target_arch = "wasm32")]
fn check(_: Vec<PathBuf>) -> Result<(), String> {
    Err("check needs the native Lua backend".into())
}

#[cfg(not(target_arch = "wasm32"))]
fn check(paths: Vec<PathBuf>) -> Result<(), String> {
    use vitaslop_playbook::lua_mlua::MluaVm;
    use vitaslop_playbook::{playbook_name, vm};
    let mut bad = 0;
    for path in &paths {
        let name = playbook_name(path);
        let result = std::fs::read_to_string(path)
            .map_err(|e| e.to_string())
            .and_then(|src| vm::load(&mut MluaVm::new(), &name, &src));
        match result {
            Ok(()) => println!("ok    {name}"),
            Err(e) => {
                bad += 1;
                println!("FAIL  {name}: {e}");
            }
        }
    }
    if bad > 0 { Err(format!("{bad} of {} playbooks do not compile", paths.len())) } else { Ok(()) }
}

fn run() -> Result<(), String> {
    let args: Vec<String> = std::env::args().skip(1).collect();
    match args.first().map(String::as_str) {
        Some("check") => {
            let paths: Vec<PathBuf> = if args.len() > 1 {
                args[1..].iter().map(PathBuf::from).collect()
            } else {
                all_playbooks().into_iter().map(|(_, p)| p).collect()
            };
            check(paths)
        }
        Some("show") if args.len() == 2 => {
            let t = read_tape(&args[1])?;
            for (k, v) in &t.header {
                println!("{k} {v}");
            }
            println!("{} input changes, end {:?}", t.changes.len(), t.end);
            for n in &t.notes {
                println!("{:>7} f{:<7} {:>9.3}s  {}", n.input, n.display, n.seconds, n.text);
            }
            Ok(())
        }
        Some("diff") if args.len() == 3 => {
            print!("{}", diff_timelines(&read_tape(&args[1])?, &read_tape(&args[2])?));
            Ok(())
        }
        _ => Err("usage: playbook check [<file.lua> ...] | show <tape> | diff <a.tape> <b.tape>".into()),
    }
}

fn main() {
    if let Err(e) = run() {
        eprintln!("{e}");
        std::process::exit(1);
    }
}
