//! The extracted title on disk, read WHERE IT LIES.
//!
//! # Why this exists
//! `RetailGuest::new` used to read the whole extracted directory into memory and then hand the
//! decrypted copy to the guest filesystem. MEASURED on one retail title (2.8 GB of extracted
//! data): the process held **2,822 MB LIVE at frame 0** and peaked at **4,940 MB** during the
//! load, every megabyte of it a data file the title had not asked for yet. The browser has never
//! done this - `mount_dump_lazy` plus an OPFS backing is its product path, and the reason a
//! gigabyte-plus title fits under wasm32's 4 GB ceiling at all. The desktop is the rig every
//! render and perf decision in this project is measured on, so it was the one engine paying a
//! cost the shipping one does not, and every desktop run was tens of seconds of I/O and
//! gigabytes of footprint before its first frame.
//!
//! This is the same mount with the disk in OPFS's place: [`DiskVfs`] answers the manifest and
//! module reads, [`DiskBacking`] serves the data files a read at a time out of file handles
//! opened once.
//!
//! Positional reads (`seek_read` / `read_at`), not seek-then-read: the preemptive scheduler
//! moves the host between OS threads and two threads sharing one cursor would serve each other's
//! offsets. The handles are opened at mount and kept - a title has 105 files in one extracted
//! tree here and 1,280 in the largest, so a handle each is cheaper than an open per read.

use std::collections::HashMap;
use std::fs::File;
use std::path::{Path, PathBuf};

use vitaslop_runtime::host::{vfs_key, FileBacking};
use vitaslop_runtime::ingest::Error;
use vitaslop_runtime::ingest::vfs::Vfs;

/// Every file under `root`, as `(slash-joined relative path, absolute path)`.
fn walk(root: &Path) -> Result<Vec<(String, PathBuf)>, String> {
    let mut out = Vec::new();
    let mut stack = vec![root.to_path_buf()];
    while let Some(dir) = stack.pop() {
        let entries = std::fs::read_dir(&dir).map_err(|e| format!("read {}: {e}", dir.display()))?;
        for entry in entries {
            let entry = entry.map_err(|e| format!("dir entry: {e}"))?;
            let path = entry.path();
            if entry.file_type().map_err(|e| format!("file type: {e}"))?.is_dir() {
                stack.push(path);
            } else {
                let rel = path
                    .strip_prefix(root)
                    .map_err(|e| format!("strip prefix: {e}"))?
                    .components()
                    .map(|c| c.as_os_str().to_string_lossy())
                    .collect::<Vec<_>>()
                    .join("/");
                out.push((rel, path));
            }
        }
    }
    Ok(out)
}

/// Read into `buf` from `f` at `off` without touching a shared cursor.
fn positional_read(f: &File, off: u64, buf: &mut [u8]) -> usize {
    #[cfg(windows)]
    use std::os::windows::fs::FileExt;
    #[cfg(unix)]
    use std::os::unix::fs::FileExt;
    let mut done = 0usize;
    while done < buf.len() {
        #[cfg(windows)]
        let r = f.seek_read(&mut buf[done..], off + done as u64);
        #[cfg(unix)]
        let r = f.read_at(&mut buf[done..], off + done as u64);
        match r {
            Ok(0) | Err(_) => break,
            Ok(n) => done += n,
        }
    }
    done
}

/// A directory on disk as a [`Vfs`]. Reads go to the file system on every call and nothing is
/// cached, because the only files read through this are the manifest and the loadable modules -
/// a few megabytes - while the data files are served by [`DiskBacking`] instead.
pub struct DiskVfs {
    files: Vec<(String, PathBuf)>,
}

impl DiskVfs {
    pub fn open(root: &Path) -> Result<DiskVfs, String> {
        Ok(DiskVfs { files: walk(root)? })
    }

    fn path_of(&self, path: &str) -> Option<&PathBuf> {
        self.files.iter().find(|(rel, _)| rel == path).map(|(_, p)| p)
    }

    /// The entries whose vfs path starts with `prefix`, as `(app-relative remainder, path on
    /// disk)` - which is exactly what [`DiskBacking`] serves.
    pub fn under(&self, prefix: &str) -> Vec<(String, PathBuf)> {
        self.files
            .iter()
            .filter_map(|(rel, p)| rel.strip_prefix(prefix).map(|r| (r.to_string(), p.clone())))
            .collect()
    }
}

impl Vfs for DiskVfs {
    fn read(&self, path: &str) -> Result<Vec<u8>, Error> {
        let p = self.path_of(path).ok_or_else(|| Error::MissingFile(path.to_string()))?;
        std::fs::read(p).map_err(|e| Error::Io(e.to_string()))
    }
    fn exists(&self, path: &str) -> bool {
        self.path_of(path).is_some()
    }
    fn list(&self) -> Vec<String> {
        self.files.iter().map(|(rel, _)| rel.clone()).collect()
    }
}

/// The title's data files, served a read at a time from handles opened once.
///
/// The key map is built with [`vfs_key`] itself, for the reason that function's own callers
/// document: the filesystem asks with a NORMALISED key while the storage spells paths its own
/// way, and a backing that maps the received key straight onto its storage misses every
/// mixed-case path - silently, because a missing length only skips the file.
pub struct DiskBacking {
    /// normalised key -> (storage spelling, open handle, length)
    entries: HashMap<String, (String, File, usize)>,
}

impl DiskBacking {
    /// `files` is `(app-relative path, path on disk)`, as [`DiskVfs::under`] returns it.
    ///
    /// A file that cannot be opened is DROPPED with a warning rather than failing the mount: a
    /// title that will not boot because one asset is unreadable is a worse outcome than one that
    /// reports the asset missing where it is actually read.
    pub fn new(files: Vec<(String, PathBuf)>) -> DiskBacking {
        let mut entries = HashMap::with_capacity(files.len());
        for (rel, path) in files {
            let Ok(f) = File::open(&path) else {
                tracing::warn!(
                    target: "vitaslop::err",
                    path = %path.display(),
                    "game file could not be opened - the title will read it as missing"
                );
                continue;
            };
            let len = f.metadata().map(|m| m.len() as usize).unwrap_or(0);
            entries.insert(vfs_key(&rel), (rel, f, len));
        }
        DiskBacking { entries }
    }

    /// Total bytes the backing serves - the figure that used to be resident.
    pub fn total_bytes(&self) -> u64 {
        self.entries.values().map(|(_, _, n)| *n as u64).sum()
    }

    /// How many files it serves.
    pub fn file_count(&self) -> usize {
        self.entries.len()
    }
}

impl FileBacking for DiskBacking {
    fn len(&self, key: &str) -> Option<usize> {
        self.entries.get(key).map(|(_, _, n)| *n)
    }

    fn read_at(&self, key: &str, off: usize, buf: &mut [u8]) -> usize {
        let Some((_, f, n)) = self.entries.get(key) else { return 0 };
        if off >= *n {
            return 0;
        }
        let want = buf.len().min(*n - off);
        positional_read(f, off as u64, &mut buf[..want])
    }

    fn keys(&self) -> Vec<String> {
        self.entries.values().map(|(rel, _, _)| rel.clone()).collect()
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    /// The two halves must agree on how a path is spelled, and the check has to be made with a
    /// MIXED-CASE name - that is the case a lowercasing key map gets wrong, and getting it wrong
    /// costs zero bytes on every read rather than an error anyone can see.
    #[test]
    fn a_mixed_case_name_is_served_and_reads_land_where_asked() {
        let dir = std::env::temp_dir().join(format!("vitaslop-diskvfs-{}", std::process::id()));
        let _ = std::fs::create_dir_all(&dir);
        let p = dir.join("MixedCase.DAT");
        std::fs::write(&p, b"0123456789").unwrap();
        let b = DiskBacking::new(vec![("Sub/MixedCase.DAT".to_string(), p.clone())]);
        let key = vfs_key("Sub/MixedCase.DAT");
        assert_eq!(b.len(&key), Some(10), "the normalised key must find the file");
        let mut buf = [0u8; 4];
        assert_eq!(b.read_at(&key, 3, &mut buf), 4);
        assert_eq!(&buf, b"3456", "a positional read must land at the offset asked for");
        // Short at end of file rather than over-reading into whatever follows it on disk.
        let mut tail = [0u8; 8];
        assert_eq!(b.read_at(&key, 6, &mut tail), 4);
        assert_eq!(&tail[..4], b"6789");
        assert_eq!(b.read_at(&key, 10, &mut tail), 0, "a read at the end is empty, not an error");
        let _ = std::fs::remove_file(&p);
    }
}
