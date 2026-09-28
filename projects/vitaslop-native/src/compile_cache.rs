//! The desktop's COMPILE CACHE: a title's transpiled AND Cranelift-compiled guest module, kept
//! beside the game so only the first boot after a build pays for them - MEASURED on one retail
//! title, 58 s of transpile + validate + compile before the first frame.
//!
//! Layout, inside the game's own directory (so deleting the game deletes it):
//!   `<game>/vitaslop-transpiled/<name>.cwasm`   wasmtime's serialized compiled module
//!   `<game>/vitaslop-transpiled/<name>.layout`  the layout numbers the run needs, written LAST
//! `<name>` = `<build>-<settings>-<exec>`:
//!   build     this executable's identity (its size and modified time), so every new build of
//!             the emulator invalidates every stored module - the browser keys the same way on its
//!             bundle's build stamp (vitaslop-web's README);
//!   settings  the transpiler's resolved emit settings (`codegen_fingerprint`), fuel included;
//!   exec      which executable of the game this is (a title can chain-load another).
//! The game itself is immutable after extraction, so it needs no hash. Storing an entry deletes
//! every entry from another build and every other entry of the same executable, so an
//! invalidated module never lingers on disk.
//!
//! A cache that cannot be read or written is never an error: the boot transpiles as it always
//! did, and the reason is logged.
use std::path::{Path, PathBuf};

use wasmtime::{Engine, Module};

/// The cache's directory name inside a game directory. A walker of the game's files skips it.
pub const DIR: &str = "vitaslop-transpiled";

/// Where one game's compiled modules live, and which of its executables is booting.
pub struct CompileCache {
    dir: PathBuf,
    exec: String,
}

/// One cache entry: a module name inside the game's cache directory.
pub struct Entry {
    dir: PathBuf,
    build: String,
    exec: String,
    name: String,
}

/// What a run needs from the transpile besides the compiled module itself.
pub struct Layout {
    pub funcs: Vec<u32>,
    pub mem_pages: u32,
    pub arm_word_off: Option<u64>,
    pub mirror_off: Option<u64>,
    pub dirty_off: Option<u64>,
    pub stubbed: Vec<u32>,
    pub stub_wasm_indices: Vec<u32>,
    pub decode_gaps: Vec<u32>,
}

fn fnv(bytes: &[u8]) -> u64 {
    let mut h = 0xcbf2_9ce4_8422_2325u64;
    for &b in bytes {
        h = (h ^ u64::from(b)).wrapping_mul(0x0000_0100_0000_01b3);
    }
    h
}

/// This executable's build identity, as hex: its size and modified time. Every rebuild changes it.
fn build_token() -> String {
    static TOKEN: std::sync::OnceLock<String> = std::sync::OnceLock::new();
    TOKEN
        .get_or_init(|| {
            let id = std::env::current_exe().and_then(std::fs::metadata).map(|m| {
                let t = m.modified().ok().and_then(|t| t.duration_since(std::time::UNIX_EPOCH).ok());
                format!("{}:{}", m.len(), t.map_or(0, |d| d.as_nanos()))
            });
            format!("{:016x}", fnv(id.unwrap_or_default().as_bytes()))
        })
        .clone()
}

impl CompileCache {
    /// The cache for the game at `game_dir`, booting executable `exec` (its path in the game).
    pub fn new(game_dir: &Path, exec: &str) -> Self {
        let exec: String = exec.chars().map(|c| if c.is_ascii_alphanumeric() { c } else { '_' }).collect();
        Self { dir: game_dir.join(DIR), exec }
    }

    /// The entry for a transpile under `fingerprint` (`vitaslop_transpiler::codegen_fingerprint`).
    pub fn entry(&self, fingerprint: &str) -> Entry {
        let build = build_token();
        let name = format!("{build}-{:016x}-{}", fnv(fingerprint.as_bytes()), self.exec);
        Entry { dir: self.dir.clone(), build, exec: self.exec.clone(), name }
    }
}

fn hex_list(v: &[u32]) -> String {
    v.iter().map(|x| format!("{x:x}")).collect::<Vec<_>>().join(",")
}

fn parse_list(s: &str) -> Option<Vec<u32>> {
    if s.is_empty() {
        return Some(Vec::new());
    }
    s.split(',').map(|x| u32::from_str_radix(x, 16).ok()).collect()
}

fn opt(v: Option<u64>) -> String {
    v.map_or_else(|| "-".to_string(), |x| format!("{x:x}"))
}

fn parse_opt(s: &str) -> Option<Option<u64>> {
    if s == "-" { Some(None) } else { u64::from_str_radix(s, 16).ok().map(Some) }
}

impl Entry {
    fn path(&self, ext: &str) -> PathBuf {
        self.dir.join(format!("{}.{ext}", self.name))
    }

    /// The stored module and its layout, if this entry exists and loads. `None` on any miss or
    /// failure (logged when it is a failure rather than an absence).
    pub fn load(&self, engine: &Engine) -> Option<(Module, Layout)> {
        let text = std::fs::read_to_string(self.path("layout")).ok()?;
        let field = |key: &str| -> Option<&str> {
            text.lines().find_map(|l| l.strip_prefix(key).and_then(|r| r.strip_prefix('=')))
        };
        let layout = (|| {
            Some(Layout {
                funcs: parse_list(field("funcs")?)?,
                mem_pages: u32::from_str_radix(field("mem_pages")?, 16).ok()?,
                arm_word_off: parse_opt(field("arm_word_off")?)?,
                mirror_off: parse_opt(field("mirror_off")?)?,
                dirty_off: parse_opt(field("dirty_off")?)?,
                stubbed: parse_list(field("stubbed")?)?,
                stub_wasm_indices: parse_list(field("stub_wasm_indices")?)?,
                decode_gaps: parse_list(field("decode_gaps")?)?,
            })
        })();
        let Some(layout) = layout else {
            tracing::warn!(target: "vitaslop::status", "compile cache: {} is unreadable - transpiling", self.path("layout").display());
            return None;
        };
        // SAFETY: the file is one this module wrote with `Module::serialize` into the game's own
        // cache directory, under a name that includes this executable's build; wasmtime also
        // checks that it was compiled by a compatible engine and refuses it otherwise.
        match unsafe { Module::deserialize_file(engine, self.path("cwasm")) } {
            Ok(m) => Some((m, layout)),
            Err(e) => {
                tracing::warn!(target: "vitaslop::status", "compile cache: {} did not load ({e}) - transpiling", self.path("cwasm").display());
                None
            }
        }
    }

    /// Store `module` + `layout` as this entry, first deleting every entry from another build
    /// and every other entry of the same executable. Failures are logged, never raised.
    pub fn store(&self, module: &Module, layout: &Layout) {
        let result = (|| -> Result<(), String> {
            std::fs::create_dir_all(&self.dir).map_err(|e| e.to_string())?;
            for f in std::fs::read_dir(&self.dir).map_err(|e| e.to_string())?.flatten() {
                let n = f.file_name().to_string_lossy().into_owned();
                let stem = n.rsplit_once('.').map_or(n.as_str(), |(s, _)| s);
                let other_build = !stem.starts_with(&format!("{}-", self.build));
                let same_exec = stem.ends_with(&format!("-{}", self.exec));
                if other_build || (same_exec && stem != self.name) {
                    let _ = std::fs::remove_file(f.path());
                }
            }
            let bytes = module.serialize().map_err(|e| e.to_string())?;
            std::fs::write(self.path("cwasm"), &bytes).map_err(|e| e.to_string())?;
            let text = format!(
                "funcs={}\nmem_pages={:x}\narm_word_off={}\nmirror_off={}\ndirty_off={}\nstubbed={}\nstub_wasm_indices={}\ndecode_gaps={}\n",
                hex_list(&layout.funcs),
                layout.mem_pages,
                opt(layout.arm_word_off),
                opt(layout.mirror_off),
                opt(layout.dirty_off),
                hex_list(&layout.stubbed),
                hex_list(&layout.stub_wasm_indices),
                hex_list(&layout.decode_gaps),
            );
            std::fs::write(self.path("layout"), text).map_err(|e| e.to_string())?;
            tracing::info!(
                target: "vitaslop::status",
                "compile cache: stored {} ({} MB) - the next boot of this build skips the transpile and the compile",
                self.path("cwasm").display(),
                bytes.len() / (1024 * 1024)
            );
            Ok(())
        })();
        if let Err(e) = result {
            tracing::warn!(target: "vitaslop::status", "compile cache: could not store {} ({e}) - the next boot transpiles again", self.name);
        }
    }
}
