//! The end-to-end ingest pipeline: a mounted container ([`Vfs`]) in, a fully
//! decrypted [`Game`] out.
//!
//! This is the glue that ties the peeling layers together in the right order:
//!
//! 1. [`detect`](super::detect) sniffs the container shape.
//! 2. For a PFS app dump: parse `files.db` + `unicv.db` (the clean-room container
//!    structure), the RIF (`work.bin`, the license key), build a [`GameData`]
//!    (the F00D-derived per-title key), then PFS-decrypt every file the databases
//!    enumerate. Assets come out as their plain bytes; `eboot.bin` and the
//!    `sce_module/*.suprx` come out as SCE SELF containers.
//! 3. Each SELF executable is unwrapped one more layer by [`self2elf`] into the
//!    plain velf [`vitaslop_loader::load`] parses.
//!
//! The result is a [`Game`]: a plaintext filesystem the guest's file IO reads
//! from, plus the loadable module images in a dependency-respecting order. This
//! is pure and wasm-safe - the browser runs the exact same pipeline into OPFS
//! that native runs onto disk - so a [`Cache`] seam lets a host persist the
//! (expensive) decrypt once and skip it on every later load.

use super::filesdb::FilesDb;
use super::pfs::PfsImage;
use super::pfscrypt::GameData;
use super::rif::Rif;
use super::self2elf::self2elf;
use super::unicv::UnicvDb;
use super::vfs::{MemVfs, Vfs};
use super::{detect, Container, Error};

/// The SCE container magic (`"SCE\0"`) that marks a SELF executable, as the first
/// four bytes. Used to tell an executable module apart from a plain asset.
const SCE_MAGIC: &[u8; 4] = b"SCE\0";

/// One loadable executable module extracted from a game: its app-relative path
/// (e.g. `"eboot.bin"` or `"sce_module/libc.suprx"`) and its decrypted, unwrapped
/// ELF/velf image, ready for [`vitaslop_loader::load`].
pub struct GameModule {
    /// App-root-relative path the module was stored at.
    pub path: String,
    /// The plain ELF/velf bytes (SELF crypto and PFS both peeled off).
    pub elf: Vec<u8>,
}

/// A fully decrypted game: every file as plaintext, plus the loadable modules.
///
/// `files` is a plaintext filesystem keyed by app-root-relative path, backing the
/// guest's file IO once it runs. `modules` are the executable images already
/// unwrapped to ELF, in a load order that places shared libraries (`*.suprx`)
/// before the `eboot.bin` that imports them.
pub struct Game {
    /// The container's content id (from the RIF), e.g.
    /// `"UP4409-PCSE00341_00-OLLIOLLIOLLIOLLI"`.
    pub content_id: String,
    /// Every PFS file, decrypted to plaintext, keyed by app-relative path. The
    /// executable modules are present here too, as their raw SELF bytes; the guest
    /// never reads those, but keeping them keeps the filesystem a faithful mirror.
    pub files: MemVfs,
    /// The executable modules, unwrapped to loadable ELF, shared libraries first.
    pub modules: Vec<GameModule>,
}

impl Game {
    /// The plaintext bytes of an app-relative file, or [`Error::MissingFile`].
    pub fn file(&self, path: &str) -> Result<Vec<u8>, Error> {
        self.files.read(path)
    }

    /// The `eboot.bin` module (the process's main executable), if present.
    pub fn eboot(&self) -> Option<&GameModule> {
        self.modules.iter().find(|m| m.path == "eboot.bin")
    }
}

/// Decrypt and unwrap a mounted container into a [`Game`].
///
/// `vfs` is the container filesystem: a directory on disk (native fixture / the
/// user's extracted dump), the NoNpDrm zip in memory, or OPFS in the browser.
/// Only the PFS app-dump shape is handled here (the shape a real NoNpDrm dump
/// takes); a bare velf or a `.pkg` returns [`Error::UnknownContainer`] for now,
/// since no fixture exercises them.
pub fn decrypt_container(vfs: &dyn Vfs) -> Result<Game, Error> {
    let root = match detect(vfs)? {
        Container::Pfs { root } => root,
        Container::Pkg { .. } | Container::Velf { .. } => {
            return Err(Error::UnknownContainer)
        }
    };
    decrypt_pfs(vfs, &root)
}

/// Join the app root prefix (possibly empty) with an app-relative path.
fn under(root: &str, rel: &str) -> String {
    if root.is_empty() {
        rel.to_string()
    } else {
        format!("{root}/{rel}")
    }
}

/// Decrypt a PFS app dump rooted at `root` inside `vfs`.
fn decrypt_pfs(vfs: &dyn Vfs, root: &str) -> Result<Game, Error> {
    // The PFS metadata and the license key live at fixed paths under the app root.
    let files_db = FilesDb::parse(&vfs.read(&under(root, "sce_pfs/files.db"))?)?;
    let unicv = UnicvDb::parse(&vfs.read(&under(root, "sce_pfs/unicv.db"))?)?;
    let rif = Rif::parse(&vfs.read(&under(root, "sce_sys/package/work.bin"))?)?;

    let image = PfsImage::new(files_db, unicv)?;
    let crypto = GameData::from_klicensee(&rif.key);

    // Decrypt every file the databases enumerate. `image.files()` resolves each
    // node to its app-relative path; the ciphertext sits at that path under the
    // container root. `image.decrypt` returns a plaintext file as-is, so this one
    // loop covers both encrypted files and the few stored in the clear.
    let mut files = MemVfs::new();
    let mut module_paths: Vec<String> = Vec::new();
    for file in image.files() {
        let ciphertext = vfs.read(&under(root, &file.path))?;
        let plaintext = image.decrypt(&file.path, &ciphertext, &rif.key, &crypto)?;
        if is_executable(&file.path, &plaintext) {
            module_paths.push(file.path.clone());
        }
        files.insert(file.path.clone(), plaintext);
    }

    // Unwrap each executable's SELF layer to a loadable ELF. Shared libraries load
    // before the eboot that imports them, so order suprx (sorted for determinism)
    // ahead of eboot.bin. (The precise inter-module dependency order is refined
    // once export resolution exists; this coarse order is already correct because
    // eboot is the sole importer of the suprx here.)
    module_paths.sort();
    module_paths.sort_by_key(|p| p == "eboot.bin");
    let mut modules = Vec::with_capacity(module_paths.len());
    for path in module_paths {
        let self_bytes = files.read(&path)?;
        let elf = self2elf(&self_bytes, &rif.key)?;
        modules.push(GameModule { path, elf });
    }

    Ok(Game {
        content_id: rif.content_id,
        files,
        modules,
    })
}

/// Whether a decrypted file is an executable module we should unwrap: the
/// `eboot.bin` or a `sce_module/*.suprx`, and carrying the SCE SELF magic. (The
/// path check keeps a stray asset that happens to start with `SCE\0` from being
/// treated as code.)
fn is_executable(path: &str, plaintext: &[u8]) -> bool {
    let named_module = path == "eboot.bin"
        || (path.starts_with("sce_module/") && path.ends_with(".suprx"));
    named_module && plaintext.len() >= 4 && &plaintext[..4] == SCE_MAGIC
}

/// A persistence seam for the decrypted output, so the expensive PFS+SELF decrypt
/// runs only on a title's first load. A native host backs this with a directory
/// on disk; the browser backs it with OPFS. Both store the plaintext filesystem
/// and the unwrapped module images; a later load reads them straight back with no
/// key material and no crypto.
///
/// The stored bytes are the user's own decrypted game data - never redistributed,
/// exactly as with any local savegame or backup.
pub trait Cache {
    /// Load a previously-decrypted game for `content_id`, or `None` if not cached.
    fn load(&self, content_id: &str) -> Option<Game>;
    /// Persist a decrypted game under its content id.
    fn store(&self, game: &Game) -> Result<(), Error>;
}

/// Decrypt a container, using `cache` to skip the work when a prior decrypt of the
/// same title is already stored. The content id is read cheaply from the RIF
/// before any bulk decryption, so a cache hit never touches the ciphertext.
pub fn ingest_cached(vfs: &dyn Vfs, cache: &dyn Cache) -> Result<Game, Error> {
    // Peek the content id from the RIF without decrypting anything.
    if let Ok(Container::Pfs { root }) = detect(vfs) {
        if let Ok(work) = vfs.read(&under(&root, "sce_sys/package/work.bin")) {
            if let Ok(rif) = Rif::parse(&work) {
                if let Some(game) = cache.load(&rif.content_id) {
                    return Ok(game);
                }
            }
        }
    }
    let game = decrypt_container(vfs)?;
    cache.store(&game)?;
    Ok(game)
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::ingest::testfix;
    use crate::ingest::vfs::DirVfs;

    /// The whole retail chain over the fixture: mount the extracted app dir,
    /// decrypt every file, and unwrap all six modules to loadable ELFs. Skips
    /// without the fixture, so `cargo test --workspace` stays green everywhere.
    #[test]
    fn decrypts_full_game_offline() {
        let Some(dir) = testfix::game_dir() else {
            return;
        };
        let vfs = DirVfs::new(dir);
        let game = decrypt_container(&vfs).expect("decrypt container");

        assert_eq!(game.content_id, "UP4409-PCSE00341_00-OLLIOLLIOLLIOLLI");

        // All six executables: the eboot plus five shared libraries, each a real
        // ELF, and the eboot ordered last (after its libraries).
        let names: Vec<&str> = game.modules.iter().map(|m| m.path.as_str()).collect();
        assert!(names.contains(&"eboot.bin"));
        for lib in [
            "sce_module/libc.suprx",
            "sce_module/libfios2.suprx",
            "sce_module/libsmart.suprx",
            "sce_module/libult.suprx",
            "sce_module/libface.suprx",
        ] {
            assert!(names.contains(&lib), "missing module {lib}");
        }
        assert_eq!(game.modules.last().unwrap().path, "eboot.bin");
        for m in &game.modules {
            assert_eq!(&m.elf[..4], b"\x7fELF", "module {} is not an ELF", m.path);
        }

        // Every module loads: the loader parses each velf's segments and imports.
        for m in &game.modules {
            let module = vitaslop_loader::load(&m.elf)
                .unwrap_or_else(|e| panic!("load {} failed: {e:?}", m.path));
            assert!(!module.segments.is_empty(), "{} has no segments", m.path);
        }

        // A decrypted asset is readable plaintext: param.sfo begins with its
        // `\0PSF` magic.
        let sfo = game.file("sce_sys/param.sfo").expect("param.sfo present");
        assert_eq!(&sfo[..4], b"\0PSF", "param.sfo not decrypted to plaintext");
    }

    /// Diagnostic: print each module's address layout (base, entry, per-segment
    /// vaddr/size/flags) so the multi-module linker can see whether the six images
    /// overlap and must be relocated to distinct bases. Ignored; run with
    /// `--ignored --nocapture`.
    #[test]
    #[ignore = "diagnostic: needs fixture"]
    fn probe_module_layout() {
        let Some(dir) = testfix::game_dir() else {
            return;
        };
        let game = decrypt_container(&DirVfs::new(dir)).expect("decrypt");
        for m in &game.modules {
            // Raw program headers, including the non-PT_LOAD segments the loader
            // drops (the SCE relocation segment carries the fixups).
            let b = &m.elf;
            let rd32 = |o: usize| u32::from_le_bytes([b[o], b[o + 1], b[o + 2], b[o + 3]]);
            let rd16 = |o: usize| u16::from_le_bytes([b[o], b[o + 1]]);
            let e_phoff = rd32(28) as usize;
            let e_phnum = rd16(44) as usize;
            eprintln!("== {} :: {} phdrs ==", m.path, e_phnum);
            for i in 0..e_phnum {
                let ph = e_phoff + i * 32;
                eprintln!(
                    "  ph{i}: type={:#010x} off={:#010x} vaddr={:#010x} filesz={:#010x} memsz={:#010x} flags={:#x} align={:#x}",
                    rd32(ph), rd32(ph + 4), rd32(ph + 8), rd32(ph + 16),
                    rd32(ph + 20), rd32(ph + 24), rd32(ph + 28),
                );
            }
            // Histogram the relocation codes and entry formats in every
            // PT_SCE_RELA segment, so the linker implements exactly the fixups
            // these modules use.
            use std::collections::BTreeMap;
            let mut by_code: BTreeMap<u8, usize> = BTreeMap::new();
            let mut short_n = 0usize;
            let mut long_n = 0usize;
            let mut piggyback = 0usize;
            for i in 0..e_phnum {
                let ph = e_phoff + i * 32;
                if rd32(ph) != vitaslop_loader::reloc::PT_SCE_RELA {
                    continue;
                }
                let off = rd32(ph + 4) as usize;
                let sz = rd32(ph + 16) as usize;
                let blob = &b[off..off + sz];
                // Count raw formats before decode expands piggybacks.
                let mut o = 0usize;
                while o + 8 <= blob.len() {
                    let w0 = u32::from_le_bytes([blob[o], blob[o + 1], blob[o + 2], blob[o + 3]]);
                    match w0 & 0xF {
                        1 => { short_n += 1; o += 8; }
                        0 => {
                            long_n += 1;
                            if (w0 >> 20) & 0xFF != 0 { piggyback += 1; }
                            o += 12;
                        }
                        _ => break,
                    }
                }
                for r in vitaslop_loader::reloc::decode(blob) {
                    *by_code.entry(r.code).or_default() += 1;
                }
            }
            eprintln!(
                "  relocs: short={short_n} long={long_n} piggyback={piggyback} codes={:?}",
                by_code
            );
            let module = vitaslop_loader::load(&m.elf).expect("load");
            eprintln!(
                "{:<28} nid={:#010x} base={:#010x} entry={:#010x} end={:#010x} imports={}",
                m.path,
                module.module_nid,
                module.base,
                module.entry,
                module.image_end(),
                module.imports.len(),
            );
            for (i, s) in module.segments.iter().enumerate() {
                eprintln!(
                    "    seg{i}: vaddr={:#010x} filesz={:#010x} memsz={:#010x} {}{}",
                    s.vaddr,
                    s.data.len(),
                    s.mem_size,
                    if s.executable { "X" } else { "-" },
                    if s.writable { "W" } else { "-" },
                );
            }
        }
    }
}
