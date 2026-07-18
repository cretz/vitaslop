//! ROM ingestion: turn a retail Vita app container into the plaintext velf/ELF
//! images the [`vitaslop_loader`] parses.
//!
//! A real title does not ship as the bare velf the loader understands. It ships
//! as a directory tree whose file bytes are wrapped in one or more encryption
//! layers. This module peels those layers off, entirely off-console, using only
//! published keys:
//!
//! 1. **Outer container.** Either a PFS raw app dump (a `sce_pfs/` directory with
//!    `files.db` + `unicv.db`, every app file individually PFS-encrypted on disk)
//!    or a `.pkg` (a single archive whose entries are AES-CTR encrypted). A
//!    format sniff ([`detect`]) picks the decryptor.
//! 2. **SELF/SCE.** The decrypted `eboot.bin` / `*.suprx` are still SELF
//!    containers with AES-CTR-encrypted, optionally deflated code segments. The
//!    [`self2elf`] layer unwraps those to a plain velf.
//! 3. **velf.** Handed to [`vitaslop_loader::load`], which the pipeline already
//!    understands.
//!
//! Everything here is pure and wasm-safe (RustCrypto primitives, no OS calls), so
//! the browser decrypts into OPFS exactly as native does onto disk. Host file
//! access is abstracted behind [`vfs::Vfs`]; only the concrete disk backend is
//! native-gated.
//!
//! The SELF layer ([`self2elf`]) ports the MIT-licensed `sceutils` (attribution
//! retained per its license). Embedded keys are bare constants (see [`keys`]).

pub mod filesdb;
pub mod keys;
pub mod pfs;
pub mod pfscrypt;
pub mod pipeline;
pub mod pkg;
pub mod rif;
pub mod self2elf;
pub mod unicv;
pub mod vfs;
pub mod zip;

use vfs::Vfs;

/// The recognized container shapes an app can arrive as.
#[derive(Debug, Clone, PartialEq, Eq)]
pub enum Container {
    /// A raw PFS app dump: `<root>/sce_pfs/files.db` present, app files
    /// PFS-encrypted on disk. `root` is the app directory (possibly nested, e.g.
    /// `"app/ABCD00001"` inside a zip).
    Pfs { root: String },
    /// A `.pkg` archive (AES-CTR transport layer): `path` is the entry.
    Pkg { path: String },
    /// A bare velf or SELF at `path` - no outer container.
    Velf { path: String },
    /// A decrypted-dump tree previously written by this pipeline (see
    /// [`pipeline::dump_entries`]): `<root>/vitaslop-dump.txt` present, plaintext
    /// files under `<root>/files/`, unwrapped ELF modules under `<root>/modules/`.
    /// Loading one needs no key material and no crypto.
    Dump { root: String },
}

/// Sniff the container format of a mounted VFS.
///
/// Order: a `sce_pfs/files.db` anywhere means a PFS dump (its parent's parent is
/// the app root); a file beginning with the `\x7fPKG` magic means a pkg; an
/// `eboot.bin` beginning with `SCE\0` or `\x7fELF` means a bare executable.
pub fn detect(vfs: &dyn Vfs) -> Result<Container, Error> {
    // Decrypted dump: its manifest marks the tree unambiguously. Checked first -
    // a dump also contains an (already-plaintext) `files/eboot.bin` that the bare
    // velf sniff below would otherwise claim.
    for p in vfs.list() {
        if p == pipeline::DUMP_MANIFEST || p.ends_with(&format!("/{}", pipeline::DUMP_MANIFEST)) {
            let root = p
                .strip_suffix(pipeline::DUMP_MANIFEST)
                .unwrap_or("")
                .trim_end_matches('/')
                .to_string();
            return Ok(Container::Dump { root });
        }
    }
    // PFS: locate `.../sce_pfs/files.db`.
    for p in vfs.list() {
        if p == "sce_pfs/files.db" || p.ends_with("/sce_pfs/files.db") {
            let root = p
                .strip_suffix("sce_pfs/files.db")
                .unwrap_or("")
                .trim_end_matches('/')
                .to_string();
            return Ok(Container::Pfs { root });
        }
    }
    // PKG: a `.pkg` file whose bytes carry the pkg magic.
    for p in vfs.list() {
        if p.ends_with(".pkg") {
            if let Ok(head) = vfs.read(&p) {
                if head.len() >= 4 && &head[0..4] == b"\x7fPKG" {
                    return Ok(Container::Pkg { path: p });
                }
            }
        }
    }
    // Bare velf/SELF.
    for p in vfs.list() {
        if p == "eboot.bin" || p.ends_with("/eboot.bin") {
            if let Ok(head) = vfs.read(&p) {
                if head.len() >= 4 && (&head[0..4] == b"SCE\0" || &head[0..4] == b"\x7fELF") {
                    return Ok(Container::Velf { path: p });
                }
            }
        }
    }
    Err(Error::UnknownContainer)
}

#[cfg(test)]
mod detect_tests {
    use super::*;

    #[test]
    fn detects_pfs_root_in_zip() {
        let Some(path) = std::env::var_os("VITASLOP_GAME_ZIP") else {
            return;
        };
        let Ok(bytes) = std::fs::read(path) else {
            return;
        };
        let vfs = zip::read_zip(&bytes).expect("read zip");
        // The zip nests the app under app/<TITLE_ID>/; detect finds that root
        // whatever the title id is.
        match detect(&vfs).expect("detect") {
            Container::Pfs { root } => assert!(
                root.starts_with("app/"),
                "detected root {root:?} is not an app/<id> path"
            ),
            other => panic!("expected a PFS container, got {other:?}"),
        }
    }
}

/// Test-fixture access. Real game bytes are never committed; tests read them from
/// an extracted dump pointed to by `VITASLOP_GAME_DIR` and skip when it is unset
/// or a file is absent, so `cargo test --workspace` stays green without the dump.
#[cfg(test)]
pub(crate) mod testfix {
    use std::path::PathBuf;

    /// The extracted app root (e.g. `.../app/ABCD00001`), or `None` if unset.
    pub fn game_dir() -> Option<PathBuf> {
        std::env::var_os("VITASLOP_GAME_DIR").map(PathBuf::from)
    }

    /// Read `rel` under the game dir, or `None` if the dir is unset / file
    /// missing.
    pub fn read(rel: &str) -> Option<Vec<u8>> {
        let p = game_dir()?.join(rel);
        std::fs::read(p).ok()
    }
}

/// Anything that can go wrong turning a container into a velf.
#[derive(Debug, Clone, PartialEq, Eq)]
pub enum Error {
    /// A structure referenced bytes outside its buffer.
    OutOfBounds(&'static str),
    /// A magic number or version we do not handle.
    BadMagic(&'static str),
    /// The container format could not be identified.
    UnknownContainer,
    /// A file named in the database was not present in the container.
    MissingFile(String),
    /// A cryptographic check (HMAC/CMAC/digest) did not match - wrong key or a
    /// corrupt input.
    IntegrityCheck(&'static str),
    /// A key the derivation needs is not available (see the module docs on the
    /// clean-room key-derivation seam).
    MissingKey(&'static str),
}

impl core::fmt::Display for Error {
    fn fmt(&self, f: &mut core::fmt::Formatter<'_>) -> core::fmt::Result {
        match self {
            Error::OutOfBounds(w) => write!(f, "out of bounds: {w}"),
            Error::BadMagic(w) => write!(f, "bad magic/version: {w}"),
            Error::UnknownContainer => write!(f, "unrecognized container format"),
            Error::MissingFile(p) => write!(f, "file not in container: {p}"),
            Error::IntegrityCheck(w) => write!(f, "integrity check failed: {w}"),
            Error::MissingKey(w) => write!(f, "missing key: {w}"),
        }
    }
}

/// Little-endian, bounds-checked reads over a byte slice. Mirrors the loader's
/// reader so the two parsers read alike.
pub(crate) struct Reader<'a> {
    pub bytes: &'a [u8],
}

impl<'a> Reader<'a> {
    pub fn new(bytes: &'a [u8]) -> Self {
        Reader { bytes }
    }

    pub fn u16(&self, at: usize) -> Result<u16, Error> {
        let b = self.bytes.get(at..at + 2).ok_or(Error::OutOfBounds("u16"))?;
        Ok(u16::from_le_bytes([b[0], b[1]]))
    }

    pub fn u32(&self, at: usize) -> Result<u32, Error> {
        let b = self.bytes.get(at..at + 4).ok_or(Error::OutOfBounds("u32"))?;
        Ok(u32::from_le_bytes([b[0], b[1], b[2], b[3]]))
    }

    pub fn u64(&self, at: usize) -> Result<u64, Error> {
        let b = self.bytes.get(at..at + 8).ok_or(Error::OutOfBounds("u64"))?;
        Ok(u64::from_le_bytes([
            b[0], b[1], b[2], b[3], b[4], b[5], b[6], b[7],
        ]))
    }

    /// A fixed-size byte array at `at`.
    pub fn bytes<const N: usize>(&self, at: usize) -> Result<[u8; N], Error> {
        let b = self.bytes.get(at..at + N).ok_or(Error::OutOfBounds("bytes"))?;
        let mut out = [0u8; N];
        out.copy_from_slice(b);
        Ok(out)
    }
}
