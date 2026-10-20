//! Link extracted titles and list the function imports the dispatch table has no arm for.
//!
//! `cargo run --release -p vitaslop-runtime --example unhandled_imports -- [--db <vita-headers db/360 dir>] <extracted-dir>...`
//!
//! Each entry is a call that would stop the run if the title made it. Naming the whole list
//! per title, before booting anything, is what lets a gap be implemented as a family rather
//! than one fatal at a time. With `--db`, NIDs this project has no name for are named from
//! the vita-headers NID database; the last line is the union across every title given.
//!
//! `--db <dir> --coverage [library-name-filter]` instead prints, for every USER library in the
//! database, how many of its functions the dispatch table handles and which it does not - the
//! work list for completing a family no title has been seen to need yet.
use std::collections::{BTreeMap, HashMap};

fn main() {
    let mut args: Vec<String> = std::env::args().skip(1).collect();
    let db = match args.iter().position(|a| a == "--db") {
        Some(i) => {
            let dir = args.remove(i + 1);
            args.remove(i);
            read_db(&dir)
        }
        None => Db::default(),
    };
    if let Some(i) = args.iter().position(|a| a == "--coverage") {
        args.remove(i);
        coverage(&db, args.first().map_or("", String::as_str));
        return;
    }
    assert!(!args.is_empty(), "usage: unhandled_imports [--db <dir>] <extracted-dir>... | --db <dir> --coverage [filter]");
    // (lib, nid) -> how many of the given titles import it.
    let mut union: BTreeMap<(u32, u32), usize> = BTreeMap::new();
    for dir in &args {
        let root = std::path::Path::new(dir);
        // A decrypted dump needs only its manifest and `modules/`: the data files are
        // gigabytes nothing here reads. Anything else is decrypted whole.
        let is_dump = root.join(vitaslop_runtime::ingest::pipeline::DUMP_MANIFEST).exists();
        let mut vfs = vitaslop_runtime::ingest::vfs::MemVfs::new();
        let mut stack = Vec::from([root.to_path_buf()]);
        while let Some(at) = stack.pop() {
            for entry in std::fs::read_dir(&at).expect("read dir") {
                let entry = entry.expect("dir entry");
                let path = entry.path();
                let rel = path
                    .strip_prefix(root)
                    .expect("under root")
                    .components()
                    .map(|c| c.as_os_str().to_string_lossy())
                    .collect::<Vec<_>>()
                    .join("/");
                if is_dump && !(rel == "modules" || rel.starts_with("modules/") || !rel.contains('/') && path.is_file()) {
                    continue;
                }
                if entry.file_type().expect("file type").is_dir() {
                    stack.push(path);
                } else {
                    vfs.insert(rel, std::fs::read(&path).expect("read file"));
                }
            }
        }
        let elfs: Vec<Vec<u8>> = if is_dump {
            let dump = vitaslop_runtime::ingest::pipeline::mount_dump_lazy(&vfs, "").expect("mount dump");
            dump.modules.into_iter().map(|m| m.elf).collect()
        } else {
            let game = vitaslop_runtime::ingest::pipeline::decrypt_container(&mut vfs).expect("decrypt");
            game.modules.into_iter().map(|m| m.elf).collect()
        };
        let modules = elfs
            .iter()
            .map(|elf| vitaslop_loader::load(elf))
            .collect::<Result<Vec<_>, _>>()
            .expect("load module");
        let linked = vitaslop_runtime::link::link(modules).expect("link");
        println!("{dir}: {} unhandled of {} host imports", linked.unhandled_imports.len(), linked.imports.len());
        for &(lib, nid) in &linked.unhandled_imports {
            println!("  {}", db.describe(lib, nid));
            *union.entry((lib, nid)).or_default() += 1;
        }
    }
    if args.len() > 1 {
        println!("union: {} distinct unhandled imports across {} titles (count = titles importing it)", union.len(), args.len());
        let mut rows: Vec<_> = union.into_iter().collect();
        rows.sort_by(|a, b| b.1.cmp(&a.1).then(a.0.cmp(&b.0)));
        for ((lib, nid), n) in rows {
            println!("  {n:>3}  {}", db.describe(lib, nid));
        }
    }
}

/// Per user library: handled/total and the unhandled names. A NID shared by two libraries
/// counts in both (a title imports whichever its SDK linked).
fn coverage(db: &Db, filter: &str) {
    assert!(!db.user_lib_funcs.is_empty(), "--coverage needs --db");
    let (mut all_handled, mut all_total) = (0usize, 0usize);
    for (lib, funcs) in &db.user_lib_funcs {
        if !lib.to_lowercase().contains(&filter.to_lowercase()) {
            continue;
        }
        let missing: Vec<&str> = funcs
            .iter()
            .filter(|(_, nid)| !vitaslop_runtime::vita::dispatch_handles(*nid))
            .map(|(name, _)| name.as_str())
            .collect();
        let handled = funcs.len() - missing.len();
        all_handled += handled;
        all_total += funcs.len();
        println!("{lib}: {handled}/{} handled", funcs.len());
        if !filter.is_empty() {
            for name in missing {
                println!("    missing {name}");
            }
        }
    }
    println!("total: {all_handled}/{all_total} user-library functions handled");
}

#[derive(Default)]
struct Db {
    libs: HashMap<u32, String>,
    funcs: HashMap<u32, String>,
    /// User (non-kernel) library name -> its FUNCTIONS (not variables) as (name, nid).
    user_lib_funcs: BTreeMap<String, Vec<(String, u32)>>,
}

impl Db {
    fn describe(&self, lib: u32, nid: u32) -> String {
        let name = match vitaslop_runtime::nid::name(nid) {
            vitaslop_runtime::nid::UNKNOWN_NAME => self.funcs.get(&nid).map_or("?", String::as_str),
            n => n,
        };
        let lib_name = self.libs.get(&lib).map_or("?", String::as_str);
        format!("{lib:08x}:{nid:08x}  {lib_name}::{name}")
    }
}

/// The vita-headers `db/360/*.yml` files: a library is a `<Name>:` key followed by its own
/// `nid:` line, and a function or variable is `<name>: 0x<nid>`.
fn read_db(dir: &str) -> Db {
    let mut db = Db::default();
    for entry in std::fs::read_dir(dir).expect("read db dir") {
        let path = entry.expect("db entry").path();
        if path.extension().is_none_or(|e| e != "yml") {
            continue;
        }
        let text = std::fs::read_to_string(&path).expect("read db file");
        // The library block being read, whether it is a kernel one, and whether the entries
        // now being read are its `functions:` (not `variables:`).
        let mut current_lib: Option<String> = None;
        let mut lib_name = String::new();
        let mut kernel = false;
        let mut in_functions = false;
        let mut lib_funcs: Vec<(String, u32)> = Vec::new();
        let flush = |name: &str, kernel: bool, funcs: &mut Vec<(String, u32)>, db: &mut Db| {
            if !name.is_empty() && !kernel && !funcs.is_empty() {
                db.user_lib_funcs.entry(name.to_string()).or_default().append(funcs);
            }
            funcs.clear();
        };
        for line in text.lines() {
            let trimmed = line.trim();
            let Some((key, value)) = trimmed.split_once(':') else { continue };
            let value = value.trim();
            if value.is_empty() {
                // A library block opens at six spaces of indent; `functions:`/`variables:` deeper.
                if line.starts_with("      ") && !line.starts_with("       ") {
                    flush(&lib_name, kernel, &mut lib_funcs, &mut db);
                    current_lib = Some(key.to_string());
                    lib_name = key.to_string();
                    kernel = false;
                    in_functions = false;
                } else {
                    in_functions = key == "functions";
                }
                continue;
            }
            if key == "kernel" {
                kernel = value == "true";
                continue;
            }
            let Some(hex) = value.strip_prefix("0x").or_else(|| value.strip_prefix("0X")) else { continue };
            let Ok(n) = u32::from_str_radix(hex, 16) else { continue };
            if key == "nid" {
                if let Some(lib) = current_lib.take() {
                    db.libs.insert(n, lib);
                }
            } else {
                db.funcs.entry(n).or_insert_with(|| key.to_string());
                if in_functions {
                    lib_funcs.push((key.to_string(), n));
                }
            }
        }
        flush(&lib_name, kernel, &mut lib_funcs, &mut db);
    }
    db
}
