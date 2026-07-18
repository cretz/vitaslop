//! A tiny virtual filesystem seam over container contents.
//!
//! Ingestion reads app files by path without caring whether they come from a
//! directory on disk (native), an in-memory zip (the NoNpDrm zip the user
//! supplies), OPFS (the browser), or a RAM map (tests). All of those implement
//! [`Vfs`]; the ingest pipeline and the container sniffer take `&dyn Vfs`.
//!
//! Paths are '/'-separated and relative to the container root (no leading
//! slash), e.g. `"sce_pfs/files.db"`.

use super::Error;
use std::collections::BTreeMap;

/// Read-only random access to container files by path.
pub trait Vfs {
    /// Read a whole file, or [`Error::MissingFile`] if absent.
    fn read(&self, path: &str) -> Result<Vec<u8>, Error>;
    /// Whether a file exists at `path`.
    fn exists(&self, path: &str) -> bool;
    /// Every file path in the container (directories are implicit).
    fn list(&self) -> Vec<String>;
}

/// An in-memory VFS backed by a path->bytes map. Backs the zip reader, the
/// browser (bytes handed in from JS), and tests.
#[derive(Default)]
pub struct MemVfs {
    files: BTreeMap<String, Vec<u8>>,
}

impl MemVfs {
    pub fn new() -> Self {
        MemVfs {
            files: BTreeMap::new(),
        }
    }

    /// Insert (or replace) a file, normalizing the path separators.
    pub fn insert(&mut self, path: impl Into<String>, bytes: Vec<u8>) {
        self.files.insert(normalize(&path.into()), bytes);
    }

    pub fn len(&self) -> usize {
        self.files.len()
    }
    pub fn is_empty(&self) -> bool {
        self.files.is_empty()
    }

    /// Consume the vfs, yielding every `(path, bytes)` pair by move. The bulk
    /// handoff into the guest filesystem uses this so a game's assets (hundreds of
    /// megabytes for a 3D title) are not cloned a second time.
    pub fn into_files(self) -> impl Iterator<Item = (String, Vec<u8>)> {
        self.files.into_iter()
    }
}

impl Vfs for MemVfs {
    fn read(&self, path: &str) -> Result<Vec<u8>, Error> {
        self.files
            .get(&normalize(path))
            .cloned()
            .ok_or_else(|| Error::MissingFile(path.to_string()))
    }
    fn exists(&self, path: &str) -> bool {
        self.files.contains_key(&normalize(path))
    }
    fn list(&self) -> Vec<String> {
        self.files.keys().cloned().collect()
    }
}

/// A VFS over a directory on disk (native only - the fixture dump, and the
/// on-disk decrypt cache).
#[cfg(not(target_arch = "wasm32"))]
pub struct DirVfs {
    root: std::path::PathBuf,
}

#[cfg(not(target_arch = "wasm32"))]
impl DirVfs {
    pub fn new(root: impl Into<std::path::PathBuf>) -> Self {
        DirVfs { root: root.into() }
    }
}

#[cfg(not(target_arch = "wasm32"))]
impl Vfs for DirVfs {
    fn read(&self, path: &str) -> Result<Vec<u8>, Error> {
        std::fs::read(self.root.join(normalize(path)))
            .map_err(|_| Error::MissingFile(path.to_string()))
    }
    fn exists(&self, path: &str) -> bool {
        self.root.join(normalize(path)).is_file()
    }
    fn list(&self) -> Vec<String> {
        let mut out = Vec::new();
        walk(&self.root, &self.root, &mut out);
        out
    }
}

#[cfg(not(target_arch = "wasm32"))]
fn walk(root: &std::path::Path, dir: &std::path::Path, out: &mut Vec<String>) {
    let Ok(entries) = std::fs::read_dir(dir) else {
        return;
    };
    for e in entries.flatten() {
        let p = e.path();
        if p.is_dir() {
            walk(root, &p, out);
        } else if let Ok(rel) = p.strip_prefix(root) {
            out.push(rel.to_string_lossy().replace('\\', "/"));
        }
    }
}

/// Normalize backslashes and strip any leading slash so keys are canonical.
fn normalize(path: &str) -> String {
    path.replace('\\', "/").trim_start_matches('/').to_string()
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn mem_vfs_roundtrip() {
        let mut v = MemVfs::new();
        v.insert("sce_pfs/files.db", vec![1, 2, 3]);
        v.insert("/eboot.bin", vec![4, 5]);
        assert!(v.exists("sce_pfs/files.db"));
        assert!(v.exists("eboot.bin")); // leading slash normalized
        assert_eq!(v.read("eboot.bin").unwrap(), vec![4, 5]);
        assert_eq!(v.list().len(), 2);
        assert!(matches!(
            v.read("missing").unwrap_err(),
            Error::MissingFile(_)
        ));
    }
}
