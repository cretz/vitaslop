//! rePatch overlays: a player's own modified or decrypted files laid OVER a title, the way the
//! device's rePatch plugin lays `ux0:rePatch/<title id>/` over `app0:` and
//! `ux0:reAddcont/<title id>/<dir>/` over `addcont0:<dir>/`.
//!
//! What the plugin does, and so what this does:
//!
//! - A file in the overlay REPLACES the title's file of the same path; a path the overlay does
//!   not carry falls through to the title. Directories merge, so a listing sees both.
//! - Executables (`eboot.bin`, `sce_module/*.suprx`, a root `*.self`) are replaced too. The
//!   plugin takes them DECRYPTED (an ELF, or an fSELF such as a decrypter writes); an
//!   encrypted SELF is refused at import, where it can be named, rather than at boot.
//! - An `addcont0:` overlay for an add-on directory makes that add-on present, as the plugin
//!   reports a DRM check on a directory it carries as passed.
//!
//! The title itself is never touched. An overlay is stored beside it, and turned off or
//! removed without re-importing anything.
//!
//! Stored layout (both front ends), under the overlay's own directory:
//!
//! - `app/<app0-relative path>`
//! - `addcont/<dir>/<path below the add-on root>`
//! - [`RECORD`], written LAST, so a folder without one is an import that did not finish.

use std::rc::Rc;

use crate::ingest::pipeline::GameModule;
use crate::ingest::stream::{ByteSource, DumpSink, Progress, ZipSource};

/// The overlay's record, written last.
pub const RECORD: &str = "repatch.json";
/// Where `app0:` replacements are stored, below the overlay's directory.
pub const APP_DIR: &str = "app/";
/// Where `addcont0:` replacements are stored, below the overlay's directory.
pub const ADDCONT_DIR: &str = "addcont/";

/// Where one input file goes.
#[derive(Clone, Debug, PartialEq, Eq)]
pub enum Target {
    /// Over `app0:<path>`.
    App(String),
    /// Over `addcont0:<dir>/<path>`.
    Addcont { dir: String, path: String },
}

impl Target {
    /// The path it is stored at, below the overlay's directory.
    pub fn stored(&self) -> String {
        match self {
            Target::App(p) => format!("{APP_DIR}{p}"),
            Target::Addcont { dir, path } => format!("{ADDCONT_DIR}{dir}/{path}"),
        }
    }
}

/// Where each input goes: `(index into paths, target)`, plus how many inputs were left out
/// (files outside the overlay, such as a readme beside a `rePatch` folder).
#[derive(Debug, Default)]
pub struct Plan {
    pub files: Vec<(usize, Target)>,
    pub ignored: usize,
}

/// `ABCD12345`: four letters and five digits, the shape every title id has.
fn looks_like_title_id(s: &str) -> bool {
    let b = s.as_bytes();
    b.len() == 9 && b[..4].iter().all(u8::is_ascii_alphabetic) && b[4..].iter().all(u8::is_ascii_digit)
}

/// What a desktop, an archiver or a file manager leaves beside files, never game content.
fn is_litter(segs: &[&str]) -> bool {
    segs.iter().any(|s| s.eq_ignore_ascii_case("__MACOSX"))
        || segs.last().is_some_and(|n| {
            n.eq_ignore_ascii_case(".DS_Store") || n.eq_ignore_ascii_case("Thumbs.db") || n.eq_ignore_ascii_case("desktop.ini")
        })
}

/// Decide where every input goes, for an overlay of `title_id`.
///
/// `paths` are as the input spells them: relative to the picked folder (with `picked_root`,
/// whose first segment is the picked folder's own name, as a browser folder pick spells it)
/// or to the zip's root. The layouts a player has to hand are all accepted:
///
/// - the device's own tree, anywhere inside the input: `.../rePatch/<id>/...` and
///   `.../reAddcont/<id>/<dir>/...`;
/// - one title's folder: `<id>/...`;
/// - the folder's contents: `eboot.bin`, `data/...`.
///
/// When any path names the title id, only paths below it are taken. An input that names a
/// DIFFERENT title id and never this one is refused: it is an overlay for another game.
pub fn plan(paths: &[String], title_id: &str, picked_root: bool) -> Result<Plan, String> {
    let split: Vec<Vec<&str>> = paths
        .iter()
        .map(|p| {
            let mut s: Vec<&str> = p.split(['/', '\\']).filter(|s| !s.is_empty() && *s != ".").collect();
            if picked_root && !s.is_empty() {
                s.remove(0);
            }
            s
        })
        .collect();
    let id_at = |segs: &[&str]| segs[..segs.len().saturating_sub(1)].iter().position(|s| s.eq_ignore_ascii_case(title_id));
    let named = split.iter().any(|s| id_at(s).is_some());
    if !named
        && let Some(other) = split
            .iter()
            .flat_map(|s| s[..s.len().saturating_sub(1)].iter())
            .find(|s| looks_like_title_id(s))
    {
        return Err(format!("this overlay is for {}, not {title_id}", other.to_ascii_uppercase()));
    }
    let mut plan = Plan::default();
    for (i, segs) in split.iter().enumerate() {
        if segs.is_empty() || is_litter(segs) {
            plan.ignored += 1;
            continue;
        }
        let target = if named {
            let Some(at) = id_at(segs) else {
                plan.ignored += 1;
                continue;
            };
            let rest = &segs[at + 1..];
            if at > 0 && segs[at - 1].eq_ignore_ascii_case("reAddcont") {
                if rest.len() < 2 {
                    plan.ignored += 1;
                    continue;
                }
                Target::Addcont { dir: rest[0].to_string(), path: rest[1..].join("/") }
            } else {
                Target::App(rest.join("/"))
            }
        } else {
            Target::App(segs.join("/"))
        };
        if let Target::Addcont { dir, .. } = &target
            && !dir.bytes().all(|b| b.is_ascii_alphanumeric())
        {
            return Err(format!("{:?} is not an add-on directory name", dir));
        }
        plan.files.push((i, target));
    }
    if plan.files.is_empty() {
        return Err("nothing in it to lay over the game".to_string());
    }
    // Two inputs for one guest path would make the result depend on input order.
    let mut seen = std::collections::HashSet::new();
    for (_, t) in &plan.files {
        if !seen.insert(t.stored().to_ascii_lowercase()) {
            return Err(format!("{} appears twice", t.stored()));
        }
    }
    Ok(plan)
}

/// Whether an `app0:` path is code the title links at boot, rather than data it reads.
pub fn is_code_path(path: &str) -> bool {
    let p = path.to_ascii_lowercase();
    p == "eboot.bin" || (p.starts_with("sce_module/") && p.ends_with(".suprx") && p.matches('/').count() == 1) || (!p.contains('/') && p.ends_with(".self"))
}

/// Refuse an executable the loader cannot take, naming the file.
///
/// The plugin wants decrypted executables, and so does this: an ELF, or an fSELF (what a
/// decrypter writes). An encrypted SELF here would fail at boot, far from the import that
/// could have said why.
pub fn check_code(path: &str, bytes: &[u8]) -> Result<(), String> {
    match vitaslop_loader::load(bytes) {
        Ok(_) => Ok(()),
        Err(e) if bytes.starts_with(b"SCE\0") => Err(format!(
            "{path} is still encrypted ({e:?}) - a rePatch overlay needs DECRYPTED executables, as the device's plugin does"
        )),
        Err(e) => Err(format!("{path} is not a loadable executable: {e:?}")),
    }
}

/// A short, stable name for an overlay's code: what the compile caches key on, so a module
/// compiled with one set of executables is never served for another.
pub fn code_hash<'a>(code: impl IntoIterator<Item = (&'a str, &'a [u8])>) -> String {
    use sha2::{Digest, Sha256};
    let mut items: Vec<(String, &[u8])> = code.into_iter().map(|(p, b)| (p.to_ascii_lowercase(), b)).collect();
    items.sort();
    let mut h = Sha256::new();
    for (p, b) in items {
        h.update((p.len() as u64).to_le_bytes());
        h.update(p.as_bytes());
        h.update((b.len() as u64).to_le_bytes());
        h.update(b);
    }
    h.finalize()[..8].iter().map(|b| format!("{b:02x}")).collect()
}

/// What [`import`] stored.
#[derive(Debug, Default)]
pub struct Imported {
    pub files: u32,
    pub bytes: u64,
    /// The `app0:` executables it replaces or adds, as [`is_code_path`] names them.
    pub code: Vec<String>,
    /// The add-on directories it lays files over.
    pub addcont: Vec<String>,
    /// [`code_hash`] of `code`, or empty when it carries none.
    pub code_hash: String,
    /// Inputs left out: outside the overlay, or litter.
    pub ignored: usize,
}

/// Copy an overlay for `title_id` out of `src` (a picked folder, or one `.zip`) into `sink`,
/// at the stored layout this module describes. The caller writes its record after this
/// returns, which is what makes the copy complete.
///
/// `picked_root`: the paths start with the picked folder's own name (see [`plan`]).
pub fn import(
    src: Rc<dyn ByteSource>,
    title_id: &str,
    picked_root: bool,
    sink: &mut dyn DumpSink,
    progress: &mut dyn FnMut(Progress<'_>),
) -> Result<Imported, String> {
    let names = src.list();
    let (src, picked_root): (Rc<dyn ByteSource>, bool) = match names.as_slice() {
        [one] if one.to_ascii_lowercase().ends_with(".zip") => {
            (Rc::new(ZipSource::open(src.clone(), one).map_err(|e| format!("{one}: {e}"))?), false)
        }
        _ => (src, picked_root),
    };
    let names = src.list();
    let plan = plan(&names, title_id, picked_root)?;
    let total: u64 = plan.files.iter().filter_map(|(i, _)| src.size(&names[*i])).sum();
    let mut out = Imported { ignored: plan.ignored, ..Imported::default() };
    let mut code: Vec<(String, Vec<u8>)> = Vec::new();
    let mut done = 0u64;
    let mut buf = vec![0u8; 1 << 20];
    for (i, target) in &plan.files {
        let from = &names[*i];
        let size = src.size(from).ok_or_else(|| format!("{from}: no size"))?;
        let stored = target.stored();
        let is_code = matches!(target, Target::App(p) if is_code_path(p));
        sink.begin(&stored, size).map_err(|e| format!("{stored}: {e}"))?;
        let mut whole = Vec::new();
        let mut off = 0u64;
        while off < size {
            let want = buf.len().min((size - off) as usize);
            let n = src.read_at(from, off, &mut buf[..want]).map_err(|e| format!("{from}: {e}"))?;
            if n == 0 {
                return Err(format!("{from}: ended at {off} of {size} bytes"));
            }
            sink.write(&buf[..n]).map_err(|e| format!("{stored}: {e}"))?;
            if is_code {
                whole.extend_from_slice(&buf[..n]);
            }
            off += n as u64;
            done += n as u64;
            progress(Progress { stage: "copy", file: from, done, total });
        }
        sink.finish().map_err(|e| format!("{stored}: {e}"))?;
        match target {
            Target::App(p) if is_code => {
                check_code(p, &whole)?;
                out.code.push(p.clone());
                code.push((p.clone(), whole));
            }
            Target::Addcont { dir, .. } if !out.addcont.iter().any(|d| d.eq_ignore_ascii_case(dir)) => out.addcont.push(dir.clone()),
            _ => {}
        }
        out.files += 1;
        out.bytes += size;
    }
    if !code.is_empty() {
        out.code_hash = code_hash(code.iter().map(|(p, b)| (p.as_str(), b.as_slice())));
    }
    Ok(out)
}

/// Read an overlay's executables back out of storage: `read(stored path)` for each of
/// `code` (the record's list), as modules [`apply_code`] takes.
pub fn read_code(code: &[String], mut read: impl FnMut(&str) -> Result<Vec<u8>, String>) -> Result<Vec<GameModule>, String> {
    code.iter()
        .map(|p| Ok(GameModule { path: p.to_ascii_lowercase(), elf: read(&Target::App(p.clone()).stored())? }))
        .collect()
}

/// Lay an overlay's executables over the title's: a module of the same path is replaced, a
/// new `sce_module/*.suprx` or `eboot.bin` is added (libraries before the eboot, as a title's
/// own are ordered), and a root `*.self` replaces or adds the `sceAppMgrLoadExec` target.
pub fn apply_code(modules: &mut Vec<GameModule>, execs: &mut Vec<GameModule>, overlay: Vec<GameModule>) {
    for m in overlay {
        let path = m.path.to_ascii_lowercase();
        if let Some(slot) = modules.iter_mut().find(|x| x.path.eq_ignore_ascii_case(&path)) {
            slot.elf = m.elf;
        } else if let Some(slot) = execs.iter_mut().find(|x| x.path.eq_ignore_ascii_case(&path)) {
            slot.elf = m.elf;
        } else if path.ends_with(".self") {
            execs.push(GameModule { path, elf: m.elf });
        } else {
            modules.push(GameModule { path, elf: m.elf });
        }
    }
    modules.sort_by(|a, b| a.path.cmp(&b.path));
    modules.sort_by_key(|m| m.path == "eboot.bin");
}

#[cfg(test)]
mod tests {
    use super::*;

    fn p(v: &[&str]) -> Vec<String> {
        v.iter().map(|s| s.to_string()).collect()
    }

    fn targets(plan: &Plan) -> Vec<String> {
        plan.files.iter().map(|(_, t)| t.stored()).collect()
    }

    #[test]
    fn the_devices_tree_is_found_anywhere_in_the_input() {
        let plan = plan(
            &p(&[
                "readme.txt",
                "ux0/rePatch/PCSA00000/eboot.bin",
                "ux0/rePatch/pcsa00000/data/a.cpk",
                "ux0/reAddcont/PCSA00000/LABEL01234567890/b.dat",
            ]),
            "PCSA00000",
            false,
        )
        .unwrap();
        assert_eq!(targets(&plan), ["app/eboot.bin", "app/data/a.cpk", "addcont/LABEL01234567890/b.dat"]);
        assert_eq!(plan.ignored, 1, "the readme is outside the overlay");
    }

    #[test]
    fn a_title_folder_or_its_contents_are_the_app_root() {
        let a = plan(&p(&["PCSA00000/data/a.cpk"]), "PCSA00000", false).unwrap();
        assert_eq!(targets(&a), ["app/data/a.cpk"]);
        let b = plan(&p(&["MyMod/data/a.cpk", "MyMod/eboot.bin"]), "PCSA00000", true).unwrap();
        assert_eq!(targets(&b), ["app/data/a.cpk", "app/eboot.bin"], "a folder pick's own name is not a path");
        let c = plan(&p(&["data/a.cpk", ".DS_Store", "__MACOSX/data/._a.cpk"]), "PCSA00000", false).unwrap();
        assert_eq!(targets(&c), ["app/data/a.cpk"]);
        assert_eq!(c.ignored, 2);
    }

    #[test]
    fn another_titles_overlay_is_refused() {
        let e = plan(&p(&["rePatch/PCSA99999/eboot.bin"]), "PCSA00000", false).unwrap_err();
        assert!(e.contains("PCSA99999"), "{e}");
        // A file NAMED like a title id is a file, not a folder of another title's.
        assert!(plan(&p(&["data/PCSA99999"]), "PCSA00000", false).is_ok());
    }

    #[test]
    fn duplicates_and_empty_inputs_are_refused() {
        assert!(plan(&p(&["a/PCSA00000/x", "b/PCSA00000/X"]), "PCSA00000", false).is_err());
        assert!(plan(&p(&["readme.txt"]), "PCSA00000", true).is_err());
    }

    #[test]
    fn code_paths_are_what_a_title_links() {
        assert!(is_code_path("eboot.bin"));
        assert!(is_code_path("sce_module/libfoo.suprx"));
        assert!(is_code_path("game2.self"));
        assert!(!is_code_path("data/eboot.bin"));
        assert!(!is_code_path("sce_module/sub/libfoo.suprx"));
        assert!(!is_code_path("sce_sys/param.sfo"));
    }

    #[test]
    fn overlay_code_replaces_and_adds_in_load_order() {
        let m = |p: &str, b: u8| GameModule { path: p.to_string(), elf: vec![b] };
        let mut modules = vec![m("sce_module/a.suprx", 1), m("eboot.bin", 2)];
        let mut execs = vec![m("other.self", 3)];
        apply_code(&mut modules, &mut execs, vec![m("eboot.bin", 9), m("sce_module/b.suprx", 8), m("Other.self", 7), m("new.self", 6)]);
        let got: Vec<(&str, u8)> = modules.iter().map(|m| (m.path.as_str(), m.elf[0])).collect();
        assert_eq!(got, [("sce_module/a.suprx", 1), ("sce_module/b.suprx", 8), ("eboot.bin", 9)]);
        let got: Vec<(&str, u8)> = execs.iter().map(|m| (m.path.as_str(), m.elf[0])).collect();
        assert_eq!(got, [("other.self", 7), ("new.self", 6)]);
    }

    const CUBE_EBOOT: &[u8] = include_bytes!("../../../vitaslop-conformance-suite-vita/cube-src/cube.eboot.bin");

    fn run_import(files: &[(&str, &[u8])]) -> Result<(Imported, Vec<(String, Vec<u8>)>), String> {
        use crate::ingest::stream::{MemSink, MemSource};
        let src = MemSource { files: files.iter().map(|(p, b)| (p.to_string(), b.to_vec())).collect() };
        let mut sink = MemSink::default();
        let got = import(Rc::new(src), "PCSA00000", true, &mut sink, &mut |_| {})?;
        Ok((got, sink.files))
    }

    #[test]
    fn an_import_copies_the_overlay_and_names_its_code() {
        let (got, mut stored) = run_import(&[
            ("Mod/rePatch/PCSA00000/eboot.bin", CUBE_EBOOT),
            ("Mod/rePatch/PCSA00000/data/x.bin", b"data"),
            ("Mod/reAddcont/PCSA00000/LABEL01234567890/y.bin", b"add"),
            ("Mod/readme.txt", b"hi"),
        ])
        .unwrap();
        stored.sort();
        let paths: Vec<&str> = stored.iter().map(|(p, _)| p.as_str()).collect();
        assert_eq!(paths, ["addcont/LABEL01234567890/y.bin", "app/data/x.bin", "app/eboot.bin"]);
        assert_eq!(stored[2].1, CUBE_EBOOT, "bytes are copied as they are");
        assert_eq!(got.code, ["eboot.bin"]);
        assert_eq!(got.addcont, ["LABEL01234567890"]);
        assert_eq!((got.files, got.ignored), (3, 1));
        assert_eq!(got.code_hash, code_hash([("eboot.bin", CUBE_EBOOT)]));
        let code = read_code(&got.code, |p| Ok(stored.iter().find(|(s, _)| s == p).unwrap().1.clone())).unwrap();
        assert_eq!(code[0].path, "eboot.bin");
    }

    #[test]
    fn an_encrypted_executable_is_refused_by_name() {
        let mut enc = b"SCE\0".to_vec();
        enc.resize(0x400, 0);
        let e = run_import(&[("Mod/sce_module/libx.suprx", &enc)]).unwrap_err();
        assert!(e.contains("sce_module/libx.suprx") && e.contains("encrypted"), "{e}");
    }

    #[test]
    fn the_code_hash_ignores_order_and_case_but_not_content() {
        let a = code_hash([("eboot.bin", &b"x"[..]), ("sce_module/a.suprx", &b"y"[..])]);
        let b = code_hash([("SCE_MODULE/a.suprx", &b"y"[..]), ("eboot.bin", &b"x"[..])]);
        let c = code_hash([("eboot.bin", &b"z"[..]), ("sce_module/a.suprx", &b"y"[..])]);
        assert_eq!(a, b);
        assert_ne!(a, c);
        assert_eq!(a.len(), 16);
    }
}
