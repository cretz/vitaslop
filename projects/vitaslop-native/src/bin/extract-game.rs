//! Extract a retail container to a decrypted-dump tree, once, on the native host.
//!
//! The PFS + SELF decrypt over a large pkg is expensive (and prohibitive to redo
//! in a browser's memory), so this tool runs the full ingest pipeline one time
//! and persists the result as the plain tree `decrypt_container` mounts directly
//! (see `pipeline::dump_entries`). Point the desktop app or the web server at the
//! output directory and the title loads with no key material and no crypto.
//!
//! Usage:
//!   extract-game --pkg <file.pkg> --work <work.bin> --out <dir>
//!   extract-game --dir <container-dir> --out <dir>
//!
//! The output directory must not already contain a dump (refuses to overwrite).

use std::path::PathBuf;
use std::process::ExitCode;

use vitaslop_runtime::ingest::pipeline::{decrypt_container, decrypt_pkg, dump_entries, DUMP_MANIFEST};
use vitaslop_runtime::ingest::vfs::DirVfs;

fn main() -> ExitCode {
    let args: Vec<String> = std::env::args().collect();
    let get = |flag: &str| -> Option<PathBuf> {
        args.iter().position(|a| a == flag).and_then(|i| args.get(i + 1)).map(PathBuf::from)
    };
    let (pkg, work, dir, out) = (get("--pkg"), get("--work"), get("--dir"), get("--out"));
    let Some(out) = out else {
        return usage("--out is required");
    };

    let game = match (&pkg, &work, &dir) {
        (Some(pkg), Some(work), None) => {
            let pkg_bytes = match std::fs::read(pkg) {
                Ok(b) => b,
                Err(e) => return fail(&format!("read {}: {e}", pkg.display())),
            };
            let work_bytes = match std::fs::read(work) {
                Ok(b) => b,
                Err(e) => return fail(&format!("read {}: {e}", work.display())),
            };
            match decrypt_pkg(&pkg_bytes, &work_bytes) {
                Ok(g) => g,
                Err(e) => return fail(&format!("decrypt pkg: {e}")),
            }
        }
        (None, None, Some(dir)) => match decrypt_container(&DirVfs::new(dir)) {
            Ok(g) => g,
            Err(e) => return fail(&format!("decrypt container: {e}")),
        },
        _ => return usage("pass either --pkg <file> --work <file>, or --dir <dir>"),
    };

    if out.join(DUMP_MANIFEST).exists() {
        return fail(&format!(
            "{} already contains a dump ({DUMP_MANIFEST} present); refusing to overwrite",
            out.display()
        ));
    }

    let entries = dump_entries(&game);
    let (mut n, mut bytes) = (0usize, 0u64);
    for (rel, data) in &entries {
        let path = out.join(rel.replace('/', std::path::MAIN_SEPARATOR_STR));
        if let Some(parent) = path.parent() {
            if let Err(e) = std::fs::create_dir_all(parent) {
                return fail(&format!("mkdir {}: {e}", parent.display()));
            }
        }
        if let Err(e) = std::fs::write(&path, data) {
            return fail(&format!("write {}: {e}", path.display()));
        }
        n += 1;
        bytes += data.len() as u64;
    }
    println!(
        "extracted {} ({n} entries, {:.1} MB) to {}",
        game.content_id,
        bytes as f64 / 1e6,
        out.display()
    );
    ExitCode::SUCCESS
}

fn usage(msg: &str) -> ExitCode {
    eprintln!("error: {msg}");
    eprintln!("usage: extract-game --pkg <file.pkg> --work <work.bin> --out <dir>");
    eprintln!("       extract-game --dir <container-dir> --out <dir>");
    ExitCode::from(2)
}

fn fail(msg: &str) -> ExitCode {
    eprintln!("error: {msg}");
    ExitCode::FAILURE
}
