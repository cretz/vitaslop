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
use super::pkg::Pkg;
use super::rif::Rif;
use super::self2elf::self2elf;
use super::unicv::UnicvDb;
use super::vfs::{MemVfs, Vfs};
use super::{detect, Container, Error};

/// Where a NoNpDrm license (the fake RIF) lives, relative to the app root.
const WORK_BIN_PATH: &str = "sce_sys/package/work.bin";

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
    /// `"XXYYYY-ABCD00001_00-0123456789ABCDEF"`.
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
/// user's extracted dump), the NoNpDrm zip in memory, or OPFS in the browser. Two
/// container shapes are handled:
///
/// - a **PFS app dump** (`sce_pfs/files.db` present), decrypted in place;
/// - a **`.pkg`** archive, whose AES-CTR transport layer is peeled to recover the
///   PFS dump inside, then decrypted the same way. A `work.bin` sitting beside the
///   pkg in `vfs` is installed as the license (a NoNpDrm dump ships the usable
///   fake RIF separately); otherwise the RIF packed inside the pkg is used.
///
/// A bare velf still returns [`Error::UnknownContainer`] (no fixture exercises a
/// PFS-less tree). For the common "one `.pkg` plus a standalone `work.bin`" case,
/// [`decrypt_pkg`] takes the two byte blobs directly.
pub fn decrypt_container(vfs: &dyn Vfs) -> Result<Game, Error> {
    match detect(vfs)? {
        Container::Dump { root } => load_dump(vfs, &root),
        Container::Pfs { root } => decrypt_pfs(vfs, &root),
        Container::Pkg { path } => {
            let pkg_bytes = vfs.read(&path)?;
            let work = sibling_work_bin(vfs, &path);
            decrypt_pkg_bytes(&pkg_bytes, work.as_deref())
        }
        Container::Velf { .. } => Err(Error::UnknownContainer),
    }
}

// --- Decrypted-dump trees -------------------------------------------------------
//
// The expensive part of ingestion is the PFS + SELF crypto over hundreds of
// megabytes. A decrypted-dump tree is that work persisted: the plaintext
// filesystem plus the unwrapped module images, laid out so a later load (native
// or browser, where redoing an 800 MB pkg decrypt in-memory is prohibitive) can
// mount the title with no key material and no crypto. The stored bytes are the
// user's own decrypted game data - never redistributed, exactly as with any
// local savegame or backup.

/// The manifest filename marking a decrypted-dump tree.
pub const DUMP_MANIFEST: &str = "vitaslop-dump.txt";
/// First line of the manifest; bumps if the layout ever changes.
const DUMP_MAGIC: &str = "vitaslop-decrypted-dump v1";

/// Serialize `game` as the relative `(path, bytes)` entries of a decrypted-dump
/// tree: the manifest (content id + module load order), every plaintext file under
/// `files/`, and each unwrapped ELF module under `modules/`. The caller persists
/// the entries wherever its storage lives (a directory on disk, OPFS, a zip);
/// [`decrypt_container`] mounts the result directly via [`Container::Dump`].
pub fn dump_entries(game: &Game) -> Vec<(String, Vec<u8>)> {
    let mut manifest = String::new();
    manifest.push_str(DUMP_MAGIC);
    manifest.push('\n');
    manifest.push_str(&format!("content_id={}\n", game.content_id));
    for m in &game.modules {
        manifest.push_str(&format!("module={}\n", m.path));
    }

    let mut out = Vec::with_capacity(2 + game.files.len() + game.modules.len());
    out.push((DUMP_MANIFEST.to_string(), manifest.into_bytes()));
    for path in game.files.list() {
        if let Ok(bytes) = game.files.read(&path) {
            out.push((format!("files/{path}"), bytes));
        }
    }
    for m in &game.modules {
        out.push((format!("modules/{}", m.path), m.elf.clone()));
    }
    out
}

/// Mount a decrypted-dump tree rooted at `root` inside `vfs` (the inverse of
/// [`dump_entries`]): parse the manifest, read the plaintext files and the
/// unwrapped modules back, preserving the manifest's module load order.
fn load_dump(vfs: &dyn Vfs, root: &str) -> Result<Game, Error> {
    let manifest = vfs.read(&under(root, DUMP_MANIFEST))?;
    let manifest = String::from_utf8(manifest).map_err(|_| Error::BadMagic("dump manifest"))?;
    let mut lines = manifest.lines();
    if lines.next() != Some(DUMP_MAGIC) {
        return Err(Error::BadMagic("dump manifest"));
    }
    let mut content_id = String::new();
    let mut module_paths: Vec<String> = Vec::new();
    for line in lines {
        if let Some(id) = line.strip_prefix("content_id=") {
            content_id = id.to_string();
        } else if let Some(p) = line.strip_prefix("module=") {
            module_paths.push(p.to_string());
        }
    }

    let files_root = under(root, "files");
    let files_prefix = format!("{files_root}/");
    let mut files = MemVfs::new();
    for p in vfs.list() {
        if let Some(rel) = p.strip_prefix(&files_prefix) {
            files.insert(rel, vfs.read(&p)?);
        }
    }

    let mut modules = Vec::with_capacity(module_paths.len());
    for path in module_paths {
        let elf = vfs.read(&under(root, &format!("modules/{path}")))?;
        modules.push(GameModule { path, elf });
    }

    Ok(Game {
        content_id,
        files,
        modules,
    })
}

/// Decrypt a NoNpDrm distribution supplied as a `.pkg` archive plus a standalone
/// `work.bin` RIF - the first-class two-file form a NoNpDrm dump ships in.
///
/// The pkg's AES-CTR transport layer is peeled to recover the app's on-disk file
/// tree (still PFS-encrypted), the caller's `work.bin` is installed as the
/// license so PFS decryption keys off the usable offline klicensee, and the
/// standard PFS + SELF pipeline finishes the job.
pub fn decrypt_pkg(pkg_bytes: &[u8], work_bin: &[u8]) -> Result<Game, Error> {
    decrypt_pkg_bytes(pkg_bytes, Some(work_bin))
}

/// The shared pkg path: extract the transport layer, optionally overriding the
/// packed RIF with a caller-supplied `work.bin`, then run the PFS pipeline over
/// the recovered tree.
fn decrypt_pkg_bytes(pkg_bytes: &[u8], work_bin: Option<&[u8]>) -> Result<Game, Error> {
    let pkg = Pkg::open(pkg_bytes)?;
    let mut vfs = pkg.extract()?;
    if let Some(work) = work_bin {
        vfs.insert(WORK_BIN_PATH, work.to_vec());
    }
    // The extracted tree is a PFS raw dump rooted at the pkg's top level.
    match detect(&vfs)? {
        Container::Pfs { root } => decrypt_pfs(&vfs, &root),
        _ => Err(Error::UnknownContainer),
    }
}

/// Find a `work.bin` sitting in the same directory as `pkg_path` inside `vfs`, if
/// any - the standalone NoNpDrm RIF a two-file dump ships next to its pkg.
fn sibling_work_bin(vfs: &dyn Vfs, pkg_path: &str) -> Option<Vec<u8>> {
    let dir = pkg_path.rsplit_once('/').map(|(d, _)| d).unwrap_or("");
    let cand = if dir.is_empty() {
        "work.bin".to_string()
    } else {
        format!("{dir}/work.bin")
    };
    vfs.read(&cand).ok()
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

    /// A decrypted-dump tree roundtrips: serialize a game, mount the entries as a
    /// container, and get the same content id, files, and module order back -
    /// through the public `decrypt_container` front door (which must detect the
    /// tree before the bare-eboot sniff claims its plaintext `files/eboot.bin`).
    #[test]
    fn dump_tree_roundtrips_through_decrypt_container() {
        let mut files = MemVfs::new();
        files.insert("eboot.bin", b"SCE\0garbage".to_vec());
        files.insert("Disc/Data/a.bin", vec![1, 2, 3]);
        let game = Game {
            content_id: "XX0000-ABCD00001_00-TEST".to_string(),
            files,
            modules: vec![
                GameModule { path: "sce_module/libx.suprx".into(), elf: b"\x7fELF-lib".to_vec() },
                GameModule { path: "eboot.bin".into(), elf: b"\x7fELF-main".to_vec() },
            ],
        };

        let mut tree = MemVfs::new();
        for (path, bytes) in dump_entries(&game) {
            tree.insert(format!("dumps/T/{path}"), bytes);
        }
        let back = decrypt_container(&tree).expect("mount dump tree");
        assert_eq!(back.content_id, game.content_id);
        assert_eq!(back.file("Disc/Data/a.bin").unwrap(), vec![1, 2, 3]);
        assert_eq!(back.file("eboot.bin").unwrap(), b"SCE\0garbage".to_vec());
        let order: Vec<&str> = back.modules.iter().map(|m| m.path.as_str()).collect();
        assert_eq!(order, ["sce_module/libx.suprx", "eboot.bin"]);
        assert_eq!(back.modules[1].elf, b"\x7fELF-main".to_vec());
    }

    /// The whole retail chain over a privately-supplied dump: mount the extracted
    /// app dir, decrypt every file, and unwrap the modules to loadable ELFs. Skips
    /// without the fixture, so `cargo test --workspace` stays green everywhere.
    /// Universal invariants only - no title-specific content id or module list.
    #[test]
    fn decrypts_full_game_offline() {
        let Some(dir) = testfix::game_dir() else {
            return;
        };
        let vfs = DirVfs::new(dir);
        let game = decrypt_container(&vfs).expect("decrypt container");

        // A content id is present (its value is title-specific).
        assert!(!game.content_id.is_empty());

        // A game is an eboot plus its shared libraries: more than one module, each a
        // real ELF, and the eboot ordered last (after the libraries it imports).
        let names: Vec<&str> = game.modules.iter().map(|m| m.path.as_str()).collect();
        assert!(names.contains(&"eboot.bin"), "no eboot.bin among {names:?}");
        assert!(game.modules.len() > 1, "expected eboot plus at least one library");
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

    /// Diagnostic: dump the eboot's inner-ELF header + program headers from a
    /// pkg + work.bin dump, to see the ET_SCE_EXEC module-info layout.
    #[test]
    #[ignore = "diagnostic: needs VITASLOP_GAME_PKG + VITASLOP_GAME_WORK"]
    fn probe_eboot_elf() {
        let (Some(pkg_path), Some(work_path)) = (
            std::env::var_os("VITASLOP_GAME_PKG"),
            std::env::var_os("VITASLOP_GAME_WORK"),
        ) else {
            return;
        };
        let pkg = std::fs::read(pkg_path).expect("read pkg");
        let work = std::fs::read(work_path).expect("read work.bin");
        let pkgo = crate::ingest::pkg::Pkg::open(&pkg).expect("open");
        let mut vfs = pkgo.extract().expect("extract");
        vfs.insert("sce_sys/package/work.bin", work.clone());
        // Decrypt just the eboot to its inner ELF, bypassing loader::load.
        use crate::ingest::{filesdb::FilesDb, pfs::PfsImage, rif::Rif, unicv::UnicvDb, self2elf::self2elf};
        let fdb = FilesDb::parse(&vfs.read("sce_pfs/files.db").unwrap()).unwrap();
        let ucv = UnicvDb::parse(&vfs.read("sce_pfs/unicv.db").unwrap()).unwrap();
        let rif = Rif::parse(&work).unwrap();
        let img = PfsImage::new(fdb, ucv).unwrap();
        let crypto = crate::ingest::pfscrypt::GameData::from_klicensee(&rif.key);
        let self_bytes = img.decrypt("eboot.bin", &vfs.read("eboot.bin").unwrap(), &rif.key, &crypto).unwrap();
        let b = self2elf(&self_bytes, &rif.key).expect("self2elf");
        let rd32 = |o: usize| u32::from_le_bytes([b[o], b[o+1], b[o+2], b[o+3]]);
        let rd16 = |o: usize| u16::from_le_bytes([b[o], b[o+1]]);
        eprintln!("e_type={:#06x} e_entry={:#010x} e_phoff={:#x} e_phnum={}", rd16(16), rd32(24), rd32(28), rd16(44));
        let phoff = rd32(28) as usize;
        for i in 0..rd16(44) as usize {
            let ph = phoff + i * 32;
            eprintln!("  ph{i}: type={:#010x} off={:#010x} vaddr={:#010x} paddr={:#010x} filesz={:#x} memsz={:#x} flags={:#x} align={:#x}",
                rd32(ph), rd32(ph+4), rd32(ph+8), rd32(ph+12), rd32(ph+16), rd32(ph+20), rd32(ph+24), rd32(ph+28));
        }
    }

    /// Diagnostic: dump the decrypted-but-still-SELF eboot head (the `SCE\0`
    /// container header) from a PFS dir dump, as a known-plaintext reference.
    #[test]
    #[ignore = "diagnostic: needs VITASLOP_GAME_DIR"]
    fn dump_eboot_self_head() {
        let Some(dir) = testfix::game_dir() else {
            return;
        };
        let game = decrypt_container(&DirVfs::new(dir)).expect("decrypt");
        let eboot = game.file("eboot.bin").expect("eboot self bytes");
        eprintln!("eboot SELF head (48 bytes):");
        for row in eboot[..48].chunks(16) {
            eprintln!("  {row:02x?}");
        }
    }

    /// Diagnostic: dump the pkg header and the extracted file tree so a new
    /// title's layout is visible. Ignored; needs VITASLOP_GAME_PKG.
    #[test]
    #[ignore = "diagnostic: needs VITASLOP_GAME_PKG"]
    fn probe_pkg_layout() {
        let Some(pkg_path) = std::env::var_os("VITASLOP_GAME_PKG") else {
            return;
        };
        let bytes = std::fs::read(pkg_path).expect("read pkg");
        let pkg = crate::ingest::pkg::Pkg::open(&bytes).expect("open pkg");
        eprintln!(
            "header: item_count={} data_offset={:#x} data_size={:#x} key_type={} content_id={}",
            pkg.header().item_count,
            pkg.header().data_offset,
            pkg.header().data_size,
            pkg.header().key_type,
            pkg.header().content_id,
        );
        let items = pkg.items().expect("items");
        eprintln!("--- {} items (first 50) ---", items.len());
        for it in items.iter().take(50) {
            eprintln!(
                "  flags={:#010x} type={:>2} size={:>10} {}",
                it.flags,
                it.flags & 0xff,
                it.data_size,
                it.name
            );
        }
        // Which of the paths the PFS layer needs are present?
        let vfs = pkg.extract().expect("extract");
        for probe in [
            "sce_pfs/files.db",
            "sce_pfs/unicv.db",
            "sce_sys/package/work.bin",
            "sce_sys/param.sfo",
            "eboot.bin",
        ] {
            eprintln!("  present {:>30}: {}", probe, vfs.exists(probe));
        }

        // Now exercise the PFS correlation over this tree.
        use crate::ingest::filesdb::FilesDb;
        use crate::ingest::pfs::PfsImage;
        use crate::ingest::unicv::UnicvDb;
        let fdb = FilesDb::parse(&vfs.read("sce_pfs/files.db").unwrap()).expect("files.db");
        let ucv = UnicvDb::parse(&vfs.read("sce_pfs/unicv.db").unwrap()).expect("unicv.db");
        eprintln!("files.db nodes={} unicv tables={}", fdb.nodes.len(), ucv.tables.len());
        let n_dirs = fdb.nodes.iter().filter(|n| n.is_dir()).count();
        let n_files = fdb.nodes.len() - n_dirs;
        eprintln!("  dirs={n_dirs} files={n_files}");
        let img = PfsImage::new(fdb, ucv).expect("pfs image");
        let files = img.files();
        eprintln!("resolved PFS files: {}", files.len());
        for f in files.iter().take(5) {
            eprintln!("  {} (node.id={} tbl={})", f.path, f.node.id, f.table_index);
        }

        // Decrypt eboot.bin and inspect the head, using the standalone work.bin.
        let Some(work_path) = std::env::var_os("VITASLOP_GAME_WORK") else {
            eprintln!("set VITASLOP_GAME_WORK to decrypt eboot");
            return;
        };
        let work = std::fs::read(work_path).expect("read work.bin");
        let rif = crate::ingest::rif::Rif::parse(&work).expect("parse rif");
        eprintln!("rif: account_id={:#x} fake={} content_id={}", rif.account_id, rif.is_fake(), rif.content_id);
        eprintln!("rif key: {:02x?}", rif.key);

        let fdb_bytes = vfs.read("sce_pfs/files.db").unwrap();
        let h = img.files_db.header;
        eprintln!("files.db: version={} block_size={:#x} key_id={} seed={:#x}", h.version, h.block_size, h.key_id, h.seed);
        eprintln!("unicv.db: version={} block_size={:#x}", img.unicv.header.version, img.unicv.header.block_size);

        // Decisive: does the files.db header ICV validate? This depends only on the
        // drv_key keygen (icv_salt=0, no tweak), so it isolates keygen from cipher.
        let drv = crate::ingest::pfscrypt::f00d_drv_key(&rif.key);
        eprintln!("drv_key = F00D(klicensee) = {:02x?}", drv);
        let secret = crate::ingest::pfscrypt::integrity_secret_pub(&drv, h.seed, 0);
        let mut hdr = fdb_bytes[..0x160].to_vec();
        for b in &mut hdr[0x4c..0x160] { *b = 0; }
        let got = crate::ingest::pfscrypt::hmac_sha1(&secret, &hdr);
        eprintln!("header ICV match={} (got {:02x?} vs {:02x?})", &got[..] == &fdb_bytes[0x4c..0x60], &got[..8], &fdb_bytes[0x4c..0x4c+8]);

        let eboot_file = files.iter().find(|f| f.path == "eboot.bin").expect("eboot node");
        let etbl = &img.unicv.tables[eboot_file.table_index];
        eprintln!("eboot node: ftype={:#x} size={} is_encrypted={}", eboot_file.node.ftype, eboot_file.node.size, eboot_file.node.is_encrypted());
        eprintln!("eboot table: page_no={} page_size={:#x} n_sectors={} iv_seed={:02x?}", etbl.page_no, etbl.page_size, etbl.n_sectors, etbl.iv_seed);
        let ct = vfs.read("eboot.bin").expect("eboot ciphertext");
        eprintln!("eboot ct[0..32]={:02x?}", &ct[..32]);

        // CBC-chaining KPA: pt[16..32] = AES_dec(K, ct[16..32]) XOR ct[0..16].
        // The unknown sector IV cancels, so this isolates whether K == drv_key.
        // Known reference pt[16..24] = header_length = 00 10 00 00 00 00 00 00.
        use aes::cipher::generic_array::GenericArray;
        use aes::cipher::{BlockDecrypt, KeyInit};
        let recover = |key: &[u8; 16]| {
            let c = aes::Aes128::new(GenericArray::from_slice(key));
            let mut b = GenericArray::clone_from_slice(&ct[16..32]);
            c.decrypt_block(&mut b);
            let mut pt = [0u8; 16];
            for j in 0..16 { pt[j] = b[j] ^ ct[j]; }
            pt
        };
        let pt_drv = recover(&drv);
        eprintln!("CBC-chain pt[16..32] with K=drv_key: {:02x?}", pt_drv);
        eprintln!("  expected pt[16..24] = [00,10,00,00,00,00,00,00] -> match={}", &pt_drv[..8] == [0,0x10,0,0,0,0,0,0]);

        // Recover sector-0 IV. AES_dec(drv,ct[0..16]) XOR IV = pt[0..16]. Known pt:
        // 53 43 45 00 03 00 00 00 [kr kr] 01 00 00 06 00 00 (kr = key_revision).
        let dec0 = {
            let c = aes::Aes128::new(GenericArray::from_slice(&drv));
            let mut b = GenericArray::clone_from_slice(&ct[..16]);
            c.decrypt_block(&mut b);
            let mut o = [0u8; 16]; o.copy_from_slice(&b); o
        };
        let known_pt: [Option<u8>; 16] = [
            Some(0x53),Some(0x43),Some(0x45),Some(0x00),Some(0x03),Some(0x00),Some(0x00),Some(0x00),
            None,None,Some(0x01),Some(0x00),Some(0x00),Some(0x06),Some(0x00),Some(0x00),
        ];
        let mut iv = [0u8; 16];
        let mut mask = [false; 16];
        for j in 0..16 {
            if let Some(p) = known_pt[j] { iv[j] = dec0[j] ^ p; mask[j] = true; }
        }
        eprintln!("recovered IV (known bytes; ?? = unknown):");
        eprint!("  ");
        for j in 0..16 { if mask[j] { eprint!("{:02x} ", iv[j]); } else { eprint!("?? "); } }
        eprintln!();

        // Candidate IV formulas.
        let enc = |key: &[u8;16], blk: &[u8;16]| {
            use aes::cipher::BlockEncrypt;
            let c = aes::Aes128::new(GenericArray::from_slice(key));
            let mut b = GenericArray::clone_from_slice(blk);
            c.encrypt_block(&mut b);
            let mut o=[0u8;16]; o.copy_from_slice(&b); o
        };
        let le128 = |n: u64| { let mut b=[0u8;16]; b[..8].copy_from_slice(&n.to_le_bytes()); b };
        let icv_salt = etbl.page_no as u64;
        let cands: Vec<(&str,[u8;16])> = vec![
            ("AES_enc(drv, 0)", enc(&drv, &[0u8;16])),
            ("AES_enc(drv, LE128(icv_salt))", enc(&drv, &le128(icv_salt))),
            ("AES_enc(drv, LE128(node.id))", enc(&drv, &le128(eboot_file.node.id as u64))),
            ("integrity_secret[..16]", { let s = crate::ingest::pfscrypt::integrity_secret_pub(&drv, 0, icv_salt as u32); let mut o=[0u8;16]; o.copy_from_slice(&s[..16]); o }),
            ("tweak_key(zeros)", crate::ingest::pfscrypt::tweak_key_pub(&[0u8;20])),
        ];
        let matches = |c: &[u8;16]| (0..16).all(|j| !mask[j] || c[j]==iv[j]);
        for (name, c) in &cands {
            eprintln!("  cand {:36} = {:02x?} match={}", name, &c[..8], matches(c));
        }

        let _ = (matches, &cands, dec0);

        // Recover FULL sector-0 IVs from files whose first 16 plaintext bytes are
        // fully known (PNG signature + IHDR), across different icv_salt values.
        let png16: [u8;16] = [0x89,0x50,0x4e,0x47,0x0d,0x0a,0x1a,0x0a,0x00,0x00,0x00,0x0d,0x49,0x48,0x44,0x52];
        let recover_iv = |path: &str, known: &[u8;16]| -> Option<([u8;16], u32)> {
            let f = files.iter().find(|f| f.path == path)?;
            if !f.node.is_encrypted() { return None; }
            let ct = vfs.read(path).ok()?;
            let c = aes::Aes128::new(GenericArray::from_slice(&drv));
            let mut b = GenericArray::clone_from_slice(&ct[..16]);
            c.decrypt_block(&mut b);
            let mut ivv = [0u8;16];
            for j in 0..16 { ivv[j] = b[j] ^ known[j]; }
            Some((ivv, img.unicv.tables[f.table_index].page_no))
        };
        for p in ["sce_sys/icon0.png","sce_sys/pic0.png","sce_sys/manual/001.png","sce_sys/manual/002.png","sce_sys/livearea/contents/image_common/background.png"] {
            if let Some((ivv, page_no)) = recover_iv(p, &png16) {
                eprintln!("IV[{}] page_no={} : {:02x?}", p, page_no, ivv);
            }
        }
        eprintln!("IV[eboot] page_no={} : (partial) {:02x?}", etbl.page_no, iv);
    }

    /// The pkg + work.bin chain over a privately-supplied two-file dump: extract
    /// the pkg transport layer, install the standalone RIF, decrypt the PFS tree,
    /// and unwrap the modules. Point `VITASLOP_GAME_PKG` at a `.pkg` and
    /// `VITASLOP_GAME_WORK` at its `work.bin`; skips when either is unset, so
    /// `cargo test --workspace` stays green everywhere. Universal invariants only -
    /// no title-specific content id or module list. Run with `--release` (debug
    /// AES over a multi-hundred-MB pkg is far too slow).
    #[test]
    #[ignore = "needs VITASLOP_GAME_PKG + VITASLOP_GAME_WORK; run with --release"]
    fn decrypts_pkg_plus_workbin_offline() {
        let (Some(pkg_path), Some(work_path)) = (
            std::env::var_os("VITASLOP_GAME_PKG"),
            std::env::var_os("VITASLOP_GAME_WORK"),
        ) else {
            return;
        };
        let pkg = std::fs::read(pkg_path).expect("read pkg");
        let work = std::fs::read(work_path).expect("read work.bin");
        let game = decrypt_pkg(&pkg, &work).expect("decrypt pkg + work.bin");

        assert!(!game.content_id.is_empty());
        let names: Vec<&str> = game.modules.iter().map(|m| m.path.as_str()).collect();
        assert!(names.contains(&"eboot.bin"), "no eboot.bin among {names:?}");
        assert_eq!(game.modules.last().unwrap().path, "eboot.bin");
        for m in &game.modules {
            assert_eq!(&m.elf[..4], b"\x7fELF", "module {} is not an ELF", m.path);
            let module = vitaslop_loader::load(&m.elf)
                .unwrap_or_else(|e| panic!("load {} failed: {e:?}", m.path));
            assert!(!module.segments.is_empty(), "{} has no segments", m.path);
        }
        let sfo = game.file("sce_sys/param.sfo").expect("param.sfo present");
        assert_eq!(&sfo[..4], b"\0PSF", "param.sfo not decrypted to plaintext");
        eprintln!(
            "pkg ingest OK: content_id={} modules={} files={}",
            game.content_id,
            game.modules.len(),
            game.files.len()
        );
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
                for r in vitaslop_loader::reloc::decode(blob).unwrap_or_default() {
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

    /// Diagnostic: decrypt the container and write named plaintext files out to
    /// `VITASLOP_DUMP_DIR`, so a decrypted asset (e.g. an `.at9`) can be inspected
    /// or fed to an external reference decoder. Comma-separated relative paths come
    /// from `VITASLOP_DUMP_FILES`. Nothing is written into the repo. Ignored; run
    /// with `--ignored --nocapture`.
    #[test]
    #[ignore = "diagnostic: needs fixture + VITASLOP_DUMP_DIR/VITASLOP_DUMP_FILES"]
    fn dump_plaintext_files() {
        let Some(dir) = testfix::game_dir() else {
            return;
        };
        let Some(out) = std::env::var_os("VITASLOP_DUMP_DIR") else {
            eprintln!("set VITASLOP_DUMP_DIR");
            return;
        };
        let files = std::env::var("VITASLOP_DUMP_FILES").unwrap_or_default();
        let out = std::path::PathBuf::from(out);
        std::fs::create_dir_all(&out).expect("create dump dir");
        let game = decrypt_container(&DirVfs::new(dir)).expect("decrypt");
        for rel in files.split(',').map(str::trim).filter(|s| !s.is_empty()) {
            match game.file(rel) {
                Ok(bytes) => {
                    let name = rel.rsplit(['/', '\\']).next().unwrap_or(rel);
                    let dest = out.join(name);
                    std::fs::write(&dest, &bytes).expect("write plaintext");
                    eprintln!("dumped {rel} -> {} ({} bytes)", dest.display(), bytes.len());
                }
                Err(e) => eprintln!("skip {rel}: {e:?}"),
            }
        }
    }
}
