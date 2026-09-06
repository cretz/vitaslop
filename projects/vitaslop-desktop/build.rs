//! Embed the browser front end so the desktop binary can serve it (`vitaslop-desktop
//! serve`): one table of `(path, bytes)` over `../vitaslop-web/web`, minus the debug
//! pages. The wasm bundle under `pkg/` is included when it has been built; when it has
//! not, the table still builds and `serve` says the bundle is missing rather than
//! failing the desktop build for a web artefact.

use std::fs;
use std::path::{Path, PathBuf};

fn walk(root: &Path, dir: &Path, out: &mut Vec<(String, PathBuf)>) {
    let Ok(rd) = fs::read_dir(dir) else { return };
    for e in rd.flatten() {
        let p = e.path();
        let rel = p.strip_prefix(root).unwrap().to_string_lossy().replace('\\', "/");
        if p.is_dir() {
            if rel == "debug" || rel == "node_modules" {
                continue;
            }
            walk(root, &p, out);
        } else if !rel.ends_with(".d.ts") && !rel.ends_with(".ttf") {
            out.push((rel, p));
        }
    }
}

fn main() {
    embed_windows_icon();
    let web = Path::new(env!("CARGO_MANIFEST_DIR")).join("../vitaslop-web/web");
    println!("cargo:rerun-if-changed={}", web.display());
    let mut files = Vec::new();
    walk(&web, &web, &mut files);
    files.sort();
    let mut src = String::from("pub static FILES: &[(&str, &[u8])] = &[\n");
    for (rel, p) in &files {
        println!("cargo:rerun-if-changed={}", p.display());
        src.push_str(&format!("    ({:?}, include_bytes!({:?})),\n", rel, p.to_string_lossy()));
    }
    src.push_str("];\n");
    let out = PathBuf::from(std::env::var("OUT_DIR").unwrap()).join("web_files.rs");
    fs::write(out, src).unwrap();
}

/// The executable's icon, Windows only. The `.ico` is rasterised from the web front
/// end's `icon.svg` (the same six shapes) - keep the two in step.
fn embed_windows_icon() {
    let ico = Path::new(env!("CARGO_MANIFEST_DIR")).join("assets/icon.ico");
    println!("cargo:rerun-if-changed={}", ico.display());
    #[cfg(windows)]
    {
        if std::env::var("CARGO_CFG_WINDOWS").is_ok() {
            let mut res = winresource::WindowsResource::new();
            res.set_icon(&ico.to_string_lossy());
            if let Err(e) = res.compile() {
                println!("cargo:warning=vitaslop-desktop: could not embed the icon: {e}");
            }
        }
    }
    #[cfg(not(windows))]
    let _ = ico;
}
