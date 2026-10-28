/// One open file descriptor: which path it refers to, the read/write cursor, and
/// the access mode decoded from the open flags.
/// >>> THE BOOT I/O JOURNAL - every guest file operation, in order, for the first [`CAP`] of
/// them, ALWAYS ON.
///
/// # Why
/// 2026-09-26: a baseball title died at frame 328 on the PHONE, deterministically (two runs, identical
/// registers), and booted on every desktop arm. The dump showed the guest's FIOS falling
/// through for every `wadC:` path - the archive mount had failed - but not ONE of the file
/// operations that mount made, so the only way forward left was knob arms on the device. A
/// mount failure lives in individual reads: which offset, how many bytes came back, and
/// whether they were the right bytes. This records exactly that, the page prints it, and the
/// desktop prints the same, so the FIRST line where the two disagree names the defect.
///
/// Each read carries `len` and a hash of its first and last 256 bytes (not the whole
/// buffer: a boot reads tens of MB and this runs on the guest's own thread). Formatting is
/// skipped entirely once the journal is full, so a steady-state read pays one atomic load.
pub mod io_journal {
    use std::sync::Mutex;
    use std::sync::atomic::{AtomicBool, AtomicU64, Ordering};

    /// Operations kept. A boot to the first menu is a few hundred.
    pub const CAP: usize = 4000;
    static LINES: Mutex<Vec<String>> = Mutex::new(Vec::new());
    static FULL: AtomicBool = AtomicBool::new(false);
    static DROPPED: AtomicU64 = AtomicU64::new(0);

    /// Record one operation; `line` is only called while there is room.
    pub fn push(line: impl FnOnce() -> String) {
        if FULL.load(Ordering::Relaxed) {
            DROPPED.fetch_add(1, Ordering::Relaxed);
            return;
        }
        let mut v = LINES.lock().unwrap_or_else(|e| e.into_inner());
        if v.len() >= CAP {
            FULL.store(true, Ordering::Relaxed);
            DROPPED.fetch_add(1, Ordering::Relaxed);
            return;
        }
        let n = v.len();
        v.push(format!("#{n} {}", line()));
    }

    /// `(lines, operations dropped after the cap)`.
    pub fn snapshot() -> (Vec<String>, u64) {
        let v = LINES.lock().unwrap_or_else(|e| e.into_inner()).clone();
        (v, DROPPED.load(Ordering::Relaxed))
    }

    /// A read's result for a journal line: `EBADF`, or the byte count and head/tail hash.
    pub fn result(r: &Option<Vec<u8>>) -> String {
        match r {
            None => "EBADF".into(),
            Some(d) if full_hash() => format!(
                "{} h{:08x} full{:08x}",
                d.len(),
                head_tail_hash(d),
                all_hash(d)
            ),
            Some(d) => format!("{} h{:08x}", d.len(), head_tail_hash(d)),
        }
    }

    /// `VITASLOP_IO_JOURNAL_FULL=1`: every read also carries a hash of ALL its bytes. The
    /// head+tail hash cannot see a file whose MIDDLE differs between two devices - which is
    /// exactly the question when a phone's stored copy of a title is suspect and a desktop's
    /// is not.
    fn full_hash() -> bool {
        static ON: std::sync::OnceLock<bool> = std::sync::OnceLock::new();
        *ON.get_or_init(|| crate::knobs::flag("VITASLOP_IO_JOURNAL_FULL"))
    }

    fn all_hash(d: &[u8]) -> u32 {
        let mut h: u32 = 0x811c_9dc5;
        for &b in d {
            h ^= b as u32;
            h = h.wrapping_mul(0x0100_0193);
        }
        h
    }

    /// FNV-1a over the first and last 256 bytes and the length.
    pub fn head_tail_hash(d: &[u8]) -> u32 {
        let mut h: u32 = 0x811c_9dc5;
        let mut eat = |b: u8| {
            h ^= b as u32;
            h = h.wrapping_mul(0x0100_0193);
        };
        let n = d.len();
        for &b in &d[..n.min(256)] {
            eat(b);
        }
        for &b in &d[n.saturating_sub(256).max(n.min(256))..] {
            eat(b);
        }
        for b in (n as u64).to_le_bytes() {
            eat(b);
        }
        h
    }

    #[cfg(test)]
    mod tests {
        #[test]
        fn hash_sees_head_tail_and_length() {
            let a = vec![7u8; 1000];
            let mut b = a.clone();
            b[999] = 8;
            let mut c = a.clone();
            c[0] = 8;
            assert_ne!(super::head_tail_hash(&a), super::head_tail_hash(&b));
            assert_ne!(super::head_tail_hash(&a), super::head_tail_hash(&c));
            assert_ne!(super::head_tail_hash(&a), super::head_tail_hash(&a[..999]));
            assert_eq!(
                super::head_tail_hash(&[1, 2, 3]),
                super::head_tail_hash(&[1, 2, 3])
            );
        }
    }
}

pub(super) struct OpenFile {
    pub(super) path: String,
    pub(super) cursor: usize,
    pub(super) readable: bool,
    pub(super) writable: bool,
}

/// Serves file bytes the host never loaded into memory.
///
/// # Why a title's files cannot simply be bytes
/// The in-memory map below is the right model for a test fixture and for native, where
/// the host has an address space to spare. It is the wrong model for the browser: a
/// retail container is over a gigabyte, and the emulator's wasm32 heap tops out at four.
/// Measured on a 1719 MB title: Chrome peaked at 8.01 GB during ingest and the worker was
/// killed mid-boot. So the browser stores the title in OPFS and serves reads from there,
/// and only what a guest actually asks for is ever resident.
///
/// # Reads are synchronous, and that is a requirement rather than a preference
/// A guest file read happens inside a host call, on a suspended guest stack that cannot
/// await. OPFS is the only browser storage offering synchronous reads
/// (`FileSystemSyncAccessHandle`, Workers only), which is why it is the backing and
/// Cache Storage / IndexedDB are not.
/// # The key an implementation is asked for is NORMALISED
/// [`keys`](FileBacking::keys) returns paths as the storage spells them, but
/// [`len`](FileBacking::len) and [`read_at`](FileBacking::read_at) are called with the
/// result of [`vfs_key`] - collapsed separators, resolved `.`/`..`, and LOWERCASED,
/// because the Vita filesystem matches case-insensitively. An implementation that maps
/// the key it receives straight back onto its own storage will therefore miss every
/// mixed-case path.
///
/// That is not a hypothetical: it is how this first shipped. The file OPENED (existence
/// is checked against the same normalised key, so that matched) and then every read
/// returned zero bytes, and what surfaced was a guest memory-access trap 31,000 host
/// calls later with nothing pointing at the filesystem. Build the normalised-to-stored
/// map with [`vfs_key`] itself, so the two sides cannot drift.
///
/// `Send` because the native preemptive scheduler moves the whole host between OS
/// threads, and that guarantee is real there - it must not be weakened for every backing
/// just because the browser's cannot honour it. The browser's OPFS backing holds JS
/// handles, which are `!Send`, and asserts `Send` for itself on the grounds that a wasm
/// worker is single-threaded by construction; that assertion is stated where it is made,
/// not hidden here.
pub trait FileBacking: Send {
    /// Length of `key`, or `None` if this backing does not serve it.
    fn len(&self, key: &str) -> Option<usize>;
    /// Read into `buf` starting at `off`; returns the count, short at end of file.
    fn read_at(&self, key: &str, off: usize, buf: &mut [u8]) -> usize;
    /// Every key served, as STORAGE spells them, so existence checks and directory
    /// listings see them all with their real names.
    fn keys(&self) -> Vec<String>;
}

/// A [`FileBacking`] whose keys are another backing's, moved under a mount prefix: what puts
/// an add-on's files (stored relative to its own root) at `addcont0:/<dir>/...`.
pub struct MountedBacking {
    /// The prefix as storage spells it, `addcont0:/LABEL01234567890`.
    prefix: String,
    /// The same prefix as a [`vfs_key`] with its trailing separator, which is what a key
    /// handed to `len`/`read_at` starts with.
    key_prefix: String,
    inner: Box<dyn FileBacking>,
}

impl MountedBacking {
    pub fn new(prefix: &str, inner: Box<dyn FileBacking>) -> MountedBacking {
        let prefix = prefix.trim_end_matches('/').to_string();
        let key_prefix = format!("{}/", vfs_key(&prefix));
        MountedBacking {
            prefix,
            key_prefix,
            inner,
        }
    }

    fn inner_key<'a>(&self, key: &'a str) -> Option<&'a str> {
        key.strip_prefix(&self.key_prefix)
    }
}

impl FileBacking for MountedBacking {
    fn len(&self, key: &str) -> Option<usize> {
        self.inner.len(self.inner_key(key)?)
    }
    fn read_at(&self, key: &str, off: usize, buf: &mut [u8]) -> usize {
        self.inner_key(key)
            .map_or(0, |k| self.inner.read_at(k, off, buf))
    }
    fn keys(&self) -> Vec<String> {
        self.inner
            .keys()
            .into_iter()
            .map(|k| format!("{}/{}", self.prefix, k.trim_start_matches('/')))
            .collect()
    }
}

/// A minimal virtual filesystem backing SceIoFilemgr: a path -> bytes map plus a
/// table of open descriptors. Read files are preloaded by the harness (e.g. a
/// game's data files) or served lazily by a [`FileBacking`]; write opens create or
/// truncate entries here. The console streams (fd 1/2) are handled directly in the IO
/// handlers and are not tracked here. Deterministic and host-only, so a run touches no
/// real filesystem.
#[derive(Default)]
pub struct FileTable {
    pub(super) files: std::collections::HashMap<String, Vec<u8>>,
    /// Files served on demand by [`backing`](FileTable::backing), as key -> length.
    /// A key here is NOT in `files` until something writes to it, at which point it is
    /// faulted in and becomes an ordinary resident file - so a guest write to a game
    /// asset behaves exactly as it always did, at the cost of materialising that one
    /// file.
    pub(super) backed: std::collections::HashMap<String, usize>,
    /// Every installed backing: the title's own files first, then one per add-on (see
    /// [`VitaState::add_addcont`]). `backed_src` names the one serving a key; a key absent
    /// from it is served by backing 0.
    pub(super) backings: Vec<Box<dyn FileBacking>>,
    pub(super) backed_src: std::collections::HashMap<String, usize>,
    /// Read-only PSARC archives found among the title's own files, mounted so that a
    /// path INSIDE one resolves as an ordinary file. See [`crate::psarc`] for why a
    /// system module's open needs this when the title's own asset loading does not.
    pub(super) archives: Vec<crate::psarc::Psarc>,
    /// Normalised key -> (archive, entry). Consulted only after `files` and `backed`
    /// have missed, so an archive can never shadow a real file.
    pub(super) archived: std::collections::HashMap<String, (usize, usize)>,
    /// Original (as-added) spelling per lowercased key, so a directory listing can
    /// return real mixed-case names for a title's own glob matching. Lookup stays
    /// case-insensitive through the lowercased `files` key.
    pub(super) originals: std::collections::HashMap<String, String>,
    pub(super) open: std::collections::HashMap<i32, OpenFile>,
    pub(super) open_dirs: std::collections::HashMap<i32, OpenDir>,
    pub(super) next_fd: i32,
    /// Directories that exist but hold nothing. The map is flat, so a directory is
    /// normally IMPLIED by the keys beneath it - which makes an empty one
    /// unrepresentable, and `sceIoMkdir` followed by `sceIoRmdir` (or by a listing)
    /// unable to see its own work. Recording explicitly-created directories fixes
    /// that without giving up the flat map: a directory exists if it is in here OR
    /// some key lies under it.
    pub(super) dirs: std::collections::HashSet<String>,
    /// Per-path status overrides written by `sceIoChstat`. Absent means "as
    /// synthesized" (a plain readable regular file); present means the guest set it,
    /// and a later `sceIoGetstat` must report what was set.
    pub(super) stats: std::collections::HashMap<String, FileStatOverride>,
    /// Whether anything on a PERSISTED mount has changed since the flag was last
    /// cleared - see [`FileTable::touch`]. This is the whole "does the save need
    /// writing out?" question, answered by a bool rather than by hashing the save every
    /// frame, and it is the reason a run that never writes costs a persistence layer
    /// nothing at all.
    pub(super) dirty: bool,
}

/// The `sceIoChstat`-settable parts of a file's status. Each is `Option` because
/// chstat takes a bit mask of WHICH fields to apply, so a call that sets only the
/// times must not also reset the mode.
#[derive(Clone, Default, Debug, PartialEq, Eq)]
pub struct FileStatOverride {
    pub mode: Option<u32>,
    pub attr: Option<u32>,
    /// `(ctime, atime, mtime)`, each the raw 16-byte guest `SceDateTime`.
    pub times: [Option<[u8; 16]>; 3],
}

/// In-memory savedata slot store: the metadata layer SceAppUtil exposes on top of a
/// title's savedata mount. Real hardware persists each slot's SceAppUtilSaveDataSlotParam
/// (the localized title/subtitle/detail and bookkeeping) to the mounted savedata
/// partition; this keeps the raw param blob per (mount, slot) in memory so a slot the
/// title creates is visible to a later get - faithful read-after-write - without touching
/// a real disk. Deterministic and host-only, like [`FileTable`]. A future backend (native
/// disk, browser OPFS) can persist these blobs; the offline oracle keeps them in RAM.
#[derive(Default)]
pub struct SaveDataStore {
    /// (mount name, slot id) -> the exact SceAppUtilSaveDataSlotParam bytes the title
    /// wrote at create/set time, echoed back verbatim on get.
    pub(super) slots: std::collections::HashMap<(String, u32), Vec<u8>>,
    /// Whether a slot changed since this was last cleared - the same "needs writing out"
    /// flag `FileTable` carries, for the same reason. See [`FileTable::touch`].
    pub(super) dirty: bool,
}

impl SaveDataStore {
    fn key(mount: &str, slot_id: u32) -> (String, u32) {
        (mount.to_string(), slot_id)
    }
    /// Whether a slot exists under this mount.
    pub fn contains(&self, mount: &str, slot_id: u32) -> bool {
        self.slots.contains_key(&Self::key(mount, slot_id))
    }
    /// Store (create or overwrite) a slot's param blob.
    pub fn put(&mut self, mount: &str, slot_id: u32, param: Vec<u8>) {
        self.slots.insert(Self::key(mount, slot_id), param);
        self.dirty = true;
    }

    /// Every slot, ordered by (mount, id) so an export is a function of the state alone.
    pub fn all(&self) -> Vec<(String, u32, Vec<u8>)> {
        let mut out: Vec<(String, u32, Vec<u8>)> = self
            .slots
            .iter()
            .map(|((m, i), p)| (m.clone(), *i, p.clone()))
            .collect();
        out.sort_by(|a, b| (&a.0, a.1).cmp(&(&b.0, b.1)));
        out
    }
    /// Whether a slot changed since [`clear_dirty`](Self::clear_dirty).
    pub fn dirty(&self) -> bool {
        self.dirty
    }

    pub fn clear_dirty(&mut self) {
        self.dirty = false;
    }

    /// The stored param blob for a slot, or `None` if it was never created.
    pub fn get(&self, mount: &str, slot_id: u32) -> Option<&[u8]> {
        self.slots
            .get(&Self::key(mount, slot_id))
            .map(Vec::as_slice)
    }
    /// Remove a slot; returns whether it existed.
    pub fn remove(&mut self, mount: &str, slot_id: u32) -> bool {
        let had = self.slots.remove(&Self::key(mount, slot_id)).is_some();
        self.dirty |= had;
        had
    }
}

/// One entry a directory descriptor yields: the child's original-case name, whether
/// it is a subdirectory (synthesized from deeper paths in the flat map), and the
/// file size (0 for a directory).
#[derive(Clone, Debug, PartialEq, Eq)]
pub struct DirEntry {
    pub name: String,
    pub is_dir: bool,
    pub size: u64,
}

/// An open directory descriptor: the listing snapshotted at `sceIoDopen` time and
/// a read cursor advanced by each `sceIoDread`.
pub(super) struct OpenDir {
    pub(super) entries: Vec<DirEntry>,
    pub(super) cursor: usize,
    /// The (vfs-keyed) path the descriptor was opened on. Kept because a descriptor
    /// can be the subject of a path operation - `_sceFiosKernelOverlayDHChstatSync`
    /// changes the status of the directory a HANDLE names - and status is recorded
    /// per path here.
    pub(super) path: String,
}

// Vita open flags (SCE_O_*), from the MIT vita-headers.
pub(super) const SCE_O_RDONLY: u32 = 0x0001;
pub(super) const SCE_O_WRONLY: u32 = 0x0002;
pub(super) const SCE_O_RDWR: u32 = 0x0003;
pub(super) const SCE_O_APPEND: u32 = 0x0100;
pub(super) const SCE_O_CREAT: u32 = 0x0200;
pub(super) const SCE_O_TRUNC: u32 = 0x0400;

// Seek origins.
pub(super) const SCE_SEEK_SET: i32 = 0;
pub(super) const SCE_SEEK_CUR: i32 = 1;
pub(super) const SCE_SEEK_END: i32 = 2;

/// A generic "no such file/dir" errno, as SCE returns negative on IO failure.
/// The guest only ever tests fd < 0, so the exact value only needs to be
/// negative; this is the real `SCE_ERROR_ERRNO_ENOENT`.
pub(super) const SCE_ERROR_ERRNO_ENOENT: i32 = 0x8001_0002u32 as i32;
/// Bad file descriptor.
pub(super) const SCE_ERROR_ERRNO_EBADF: i32 = 0x8001_0009u32 as i32;
/// The path already exists (mkdir over something, rename onto something).
pub(super) const SCE_ERROR_ERRNO_EEXIST: i32 = 0x8001_0011u32 as i32;
/// A file operation was asked of a directory.
pub(super) const SCE_ERROR_ERRNO_EISDIR: i32 = 0x8001_0015u32 as i32;
/// `sceIoRmdir` on a directory that still holds entries.
pub(super) const SCE_ERROR_ERRNO_ENOTEMPTY: i32 = 0x8001_005Au32 as i32;

/// Reserved descriptors: 0 stdin, 1 stdout, 2 stderr. Real fds start at 3.
pub(super) const FIRST_FD: i32 = 3;
pub const FD_STDOUT: i32 = 1;
pub const FD_STDERR: i32 = 2;

/// Resolve a guest file path to a VFS key.
///
/// Two Vita filesystem facts are folded in here so a title finds its data files:
/// - **Mount point.** A Vita mounts the running app's own read-only data at `app0:`,
///   and the decrypted game files are stored under their paths relative to that mount,
///   so an `app0:` prefix (with or without the separator) is stripped. Other mounts -
///   `ux0:`, `savedata0:` - keep their full path, a distinct, writable space.
/// - **Case-insensitivity.** The Vita's FAT/exFAT filesystem is case-insensitive, so
///   the key is lowercased. Titles routinely request an asset with different casing
///   than the shipped filename (e.g. open `Neondecals.gxt` for the file
///   `neondecals.gxt`); a case-sensitive map would miss it.
///
/// Without this a title that opens `app0:/settings/foo.ini` or `.../Foo.GXT` would
/// miss the file stored as `settings/foo.ini` and take its file-missing (often fatal)
/// path.
///
/// # >>> THE SEPARATOR AFTER A MOUNT IS OPTIONAL, AND TREATING IT AS PART OF THE PATH
/// # LOSES SAVES.
/// On a Vita the colon IS the separator: `savedata0:foo.bin` and `savedata0:/foo.bin`
/// name the same file, and a title uses whichever spelling its own string-joining
/// happens to produce. This split the two apart - the mount was just another path
/// segment, so `savedata0:` + `foo` and `savedata0:foo` folded to different keys.
///
/// The title that found it writes its save through `sceAppUtilSaveDataDataSave`, which
/// builds `savedata0:/data0_US.bin`, and reads it back with `sceIoOpen`
/// (`savedata0:data0_US.bin`). The read got ENOENT on the file the same run had just
/// written, so **every reload started from a fresh profile** however faithfully the
/// container round-tripped - and it round-tripped perfectly, which is what made this look
/// like a persistence bug rather than a path one.
///
/// The mount is therefore canonicalised, not walked: everything up to the first `:` is
/// the mount and is re-joined as `mount:/rest`. That also means a `..` cannot climb out
/// of a mount, which on a real device it cannot either.
pub fn vfs_key(path: &str) -> String {
    // Normalize like a real Vita FS before matching: collapse repeated separators,
    // drop `.` and empty segments, and resolve `..`. Titles build paths by joining a
    // directory (often with a trailing `/`) to a subpath (often with a leading `/`),
    // yielding runs like `usrdir//ui/fonts//x.ttf`; an exact-match store would miss
    // those. Both the store side (`add_file`) and every lookup route through here, so
    // the normalization stays symmetric. Backslashes are folded to `/` for the rare
    // title that uses them. Lowercased last (see the case note above).
    let stripped = strip_app0(path).replace('\\', "/");
    // Split off the mount, if there is one. A mount name carries no separator, so a `:`
    // that appears after the first `/` is part of a filename and not a mount at all.
    let (mount, rest) = match stripped.find(':') {
        Some(i) if !stripped[..i].contains('/') => (&stripped[..=i], &stripped[i + 1..]),
        _ => ("", &stripped[..]),
    };
    let mut segs: Vec<&str> = Vec::new();
    for seg in rest.split('/') {
        match seg {
            "" | "." => {}
            ".." => {
                segs.pop();
            }
            s => segs.push(s),
        }
    }
    let mut key = mount.to_ascii_lowercase();
    if !segs.is_empty() {
        if !key.is_empty() {
            key.push('/');
        }
        key.push_str(&segs.join("/"));
    }
    key.make_ascii_lowercase();
    key
}

/// Strip an `app0:` mount prefix (with or without the separator), case-insensitively:
/// titles spell the mount both `app0:` and `APP0:`. The rest of the path keeps its
/// case - [`vfs_key`] lowercases for lookup, while the original spelling is preserved
/// for directory listings (a title pattern-matches `sceIoDread` names against
/// mixed-case globs like `Foo_*`).
pub(super) fn strip_app0(path: &str) -> &str {
    let b = path.as_bytes();
    if b.len() >= 5 && b[..5].eq_ignore_ascii_case(b"app0:") {
        let rest = &path[5..];
        rest.strip_prefix('/').unwrap_or(rest)
    } else {
        path
    }
}

/// Whether a (already-lowercased [`vfs_key`]) path lives on a writable, persisted
/// Vita mount - `savedata0:` (the title's own save slot) or `ux0:` (the memory
/// card). A write-then-close on one of these is game-authored persistent state, so
/// the egress ledger records it. `app0:` (read-only game data) never qualifies, and
/// is stripped by `vfs_key` anyway.
///
/// # This is also the boundary the game-data container is defined by
/// [`crate::gamedata`] exports exactly the keys this answers `true` for, and REFUSES to
/// import any other - which is what makes an imported container incapable of touching the
/// installed title. Widening this widens both, so it is one definition on purpose: the
/// set the ledger calls "the game persisted state" and the set that survives a reload are
/// the same set, and a third spelling of it would eventually disagree with the other two.
pub fn is_persisted_mount(key: &str) -> bool {
    key.starts_with("savedata") || key.starts_with("ux0:")
}

impl FileTable {
    pub(super) fn new() -> Self {
        FileTable {
            next_fd: FIRST_FD,
            ..Default::default()
        }
    }

    /// Install a lazy backing and register every key it serves. Called at setup, before the
    /// guest runs: once for the title's own files, then once per add-on. A later backing
    /// never takes over a key an earlier one already serves.
    ///
    /// Registering the keys up front (rather than probing the backing on each miss) is
    /// what keeps existence checks, directory listings and `sceIoGetstat` answering from
    /// a plain map lookup - so a backed title behaves identically to a resident one
    /// everywhere except the read itself.
    pub fn set_backing(&mut self, backing: Box<dyn FileBacking>) {
        let src = self.backings.len();
        let mut added = Vec::new();
        for path in backing.keys() {
            // NORMALISE FIRST. `len` and `read_at` take the same key the filesystem will
            // later look up with, not the storage spelling - see `FileBacking`. Asking
            // with the raw path is how this first shipped, and because a missing length
            // merely skipped the file, EVERY file was skipped and the title booted with an
            // empty filesystem. A silent skip is what turned a one-line mistake into a
            // guest memory trap 30,000 host calls away.
            let key = vfs_key(&path);
            let Some(n) = backing.len(&key) else {
                panic!(
                    "file backing lists {path:?} but has no length for its normalised key \
                     {key:?} - the two sides disagree on how a path is spelled, and every \
                     read of this file would silently return nothing"
                )
            };
            if self.backed.contains_key(&key) {
                continue;
            }
            // Keep the as-supplied spelling too: a directory listing hands names back to
            // the title's own glob matching, and a lowercased name would not match.
            self.originals.insert(
                key.clone(),
                strip_app0(&path).trim_start_matches('/').to_string(),
            );
            self.backed.insert(key.clone(), n);
            if src > 0 {
                self.backed_src.insert(key.clone(), src);
            }
            added.push(key);
        }
        self.backings.push(backing);
        self.mount_archives(&added);
    }

    /// Mark the persisted set as changed, if `key` is part of it.
    ///
    /// Called from every mutating path below. Gated on the mount so the ordinary case -
    /// a title faulting one of its own read-only assets resident because it wrote to it -
    /// does not make a save look dirty; only what [`crate::gamedata`] would export can.
    pub(super) fn touch(&mut self, key: &str) {
        if is_persisted_mount(key) {
            self.dirty = true;
        }
    }

    /// Whether anything exportable has changed since [`clear_dirty`](Self::clear_dirty).
    pub(super) fn dirty(&self) -> bool {
        self.dirty
    }

    pub(super) fn clear_dirty(&mut self) {
        self.dirty = false;
    }

    /// Whether `key` exists at all - resident, served by the backing, or inside a
    /// mounted archive.
    pub(super) fn has(&self, key: &str) -> bool {
        self.files.contains_key(key)
            || self.backed.contains_key(key)
            || self.archived.contains_key(key)
    }

    /// Length of `key`, wherever it lives.
    pub(super) fn byte_len(&self, key: &str) -> Option<usize> {
        self.files
            .get(key)
            .map(|d| d.len())
            .or_else(|| self.backed.get(key).copied())
            .or_else(|| self.archived_entry(key).map(|(_, e)| e.size as usize))
    }

    /// The archive and entry serving `key`, if one does.
    pub(super) fn archived_entry(&self, key: &str) -> Option<(&crate::psarc::Psarc, &crate::psarc::Entry)> {
        let &(a, e) = self.archived.get(key)?;
        let archive = self.archives.get(a)?;
        Some((archive, archive.entries.get(e)?))
    }

    /// Mount every `.psarc` among the keys just installed.
    ///
    /// Called from the two places a title's files arrive - [`FileTable::set_backing`] for a
    /// lazily-served container and `add_file` for a resident one - so both front ends get
    /// this without either having to ask for it, and both get it BEFORE the guest runs.
    ///
    /// A refusal is reported and otherwise ignored: the archive stays what it already was,
    /// an opaque file the title reads for itself.
    pub(super) fn mount_archives(&mut self, keys: &[String]) {
        for key in keys {
            if !key.ends_with(".psarc") || self.archives.iter().any(|a| a.key == *key) {
                continue;
            }
            // The archive's own bytes are read back through this table, which is what lets
            // the browser serve a 1.6 GB container out of OPFS without ever holding it.
            let read = |off: usize, len: usize| self.read_range(key, off, len);
            let parsed = match crate::psarc::Psarc::parse(key, &read) {
                Ok(p) => p,
                Err(reason) => {
                    tracing::warn!(
                        target: "vitaslop::io",
                        %reason,
                        "a .psarc file could not be mounted, so any path INSIDE it stays \
                         invisible to a system module that opens by path (the title's own \
                         asset loading is unaffected - it reads the archive itself)"
                    );
                    continue;
                }
            };
            let idx = self.archives.len();
            let mut mounted = 0usize;
            let mut shadowed = 0usize;
            for (i, entry) in parsed.entries.iter().enumerate() {
                let ekey = vfs_key(&entry.name);
                // Resident and backed files win: an archive is the LAST place a path
                // resolves, so mounting one can never change what an existing path means.
                if self.files.contains_key(&ekey) || self.backed.contains_key(&ekey) {
                    shadowed += 1;
                    continue;
                }
                self.originals
                    .entry(ekey.clone())
                    .or_insert_with(|| entry.name.clone());
                self.archived.insert(ekey, (idx, i));
                mounted += 1;
            }
            tracing::info!(
                target: "vitaslop::status",
                archive = %key,
                files = mounted,
                shadowed,
                "psarc mounted: its files now resolve by path, so a system module that opens \
                 one on the title's behalf (sceMp4OpenFile) reads the same bytes the device does"
            );
            self.archives.push(parsed);
        }
    }

    /// Read `[start, start+len)` of `key`, clamped to its end. Serves a resident file
    /// from the map and a backed one straight from storage, so a backed read never
    /// materialises anything but the bytes asked for.
    pub(super) fn read_range(&self, key: &str, start: usize, len: usize) -> Option<Vec<u8>> {
        let _owner = vitaslop_platform::heap::owner_scope(vitaslop_platform::heap::Owner::FileIo);
        if let Some(data) = self.files.get(key) {
            let start = start.min(data.len());
            let end = (start + len).min(data.len());
            return Some(data[start..end].to_vec());
        }
        if let Some(&size) = self.backed.get(key) {
            let backing = self
                .backings
                .get(self.backed_src.get(key).copied().unwrap_or(0))?;
            let start = start.min(size);
            let end = (start + len).min(size);
            let mut out = vec![0u8; end - start];
            let got = backing.read_at(key, start, &mut out);
            out.truncate(got);
            return Some(out);
        }
        // Last: a file that only exists inside a mounted archive. The archive's own bytes
        // come back through this same function - one level down, never two, because an
        // archive inside an archive is not mounted (its key is not among the ones
        // `mount_archives` was handed).
        let (archive, entry) = self.archived_entry(key)?;
        let size = entry.size as usize;
        let start = start.min(size);
        let end = (start + len).min(size);
        let mut out = vec![0u8; end - start];
        let idx = self.archived.get(key)?.1;
        let inner = |off: usize, n: usize| self.read_range(&archive.key, off, n);
        let got = archive.read_entry(idx, start, &mut out, &inner);
        out.truncate(got);
        Some(out)
    }

    /// Backed keys equal to `key` or beneath it as a directory prefix. Used where a key
    /// is about to MOVE, which the backing cannot follow.
    pub(super) fn backed_under(&self, key: &str) -> Vec<String> {
        let prefix = format!("{key}/");
        self.backed
            .keys()
            .filter(|k| *k == key || k.starts_with(&prefix))
            .cloned()
            .collect()
    }

    /// Make `key` resident so it can be written, reading it out of the backing first if
    /// that is where it lives. A no-op for a file that is already resident.
    ///
    /// This is the one place a backed file becomes an ordinary one. It is deliberately
    /// eager and whole-file: a partial write to a lazily-backed file would otherwise need
    /// a copy-on-write overlay, and the case does not arise in practice (titles write
    /// savedata, which they create, not the assets they shipped).
    pub(super) fn make_resident(&mut self, key: &str) -> &mut Vec<u8> {
        if !self.files.contains_key(key) {
            // A backed file and an archived one both fault in whole; only a key that
            // exists nowhere starts empty.
            let bytes = match self.byte_len(key) {
                Some(n) if self.backed.contains_key(key) || self.archived.contains_key(key) => {
                    self.read_range(key, 0, n).unwrap_or_default()
                }
                _ => Vec::new(),
            };
            self.backed.remove(key);
            self.archived.remove(key);
            self.files.insert(key.to_string(), bytes);
        }
        self.files.get_mut(key).expect("just inserted")
    }

    /// Open `path` per the SCE_O_* `flags`; returns a new fd or a negative errno.
    pub(super) fn open(&mut self, path: &str, flags: u32) -> i32 {
        let path = vfs_key(path);
        let readable = flags & SCE_O_RDWR == SCE_O_RDONLY || flags & SCE_O_RDWR == SCE_O_RDWR;
        let writable = flags & SCE_O_WRONLY != 0;
        let exists = self.has(&path);

        if !exists {
            if flags & SCE_O_CREAT != 0 {
                self.files.insert(path.clone(), Vec::new());
                self.touch(&path);
            } else {
                return SCE_ERROR_ERRNO_ENOENT;
            }
        } else if flags & SCE_O_TRUNC != 0 {
            // Truncation discards the contents, so a backed file needs no fault-in -
            // just drop the backing's claim on the key and start empty.
            self.backed.remove(&path);
            self.files.insert(path.clone(), Vec::new());
            self.touch(&path);
        }

        // Append seeks to end; every other open starts at the beginning.
        let cursor = if flags & SCE_O_APPEND != 0 {
            self.byte_len(&path).unwrap_or(0)
        } else {
            0
        };

        let fd = self.next_fd;
        self.next_fd += 1;
        self.open.insert(
            fd,
            OpenFile {
                path,
                cursor,
                readable,
                writable,
            },
        );
        fd
    }

    /// Read up to `len` bytes from `fd` at its cursor; advances it. Returns the
    /// bytes, or None on a bad/unreadable descriptor.
    pub(super) fn read(&mut self, fd: i32, len: usize) -> Option<Vec<u8>> {
        let of = self.open.get_mut(&fd)?;
        if !of.readable {
            return None;
        }
        let (path, cursor) = (of.path.clone(), of.cursor);
        let out = self.read_range(&path, cursor, len)?;
        // Advance by what was actually delivered, so a short read at end of file leaves
        // the cursor at the end rather than past it.
        let end = cursor.min(self.byte_len(&path).unwrap_or(0)) + out.len();
        self.open.get_mut(&fd)?.cursor = end;
        Some(out)
    }

    /// Read up to `len` bytes from `fd` starting at absolute `offset` WITHOUT moving
    /// the descriptor's cursor (sceIoPread). Returns the bytes, or None on a
    /// bad/unreadable descriptor. This is how a streaming reader (e.g. the AT9 music
    /// streamer) pulls the file header and successive chunks from a shared fd.
    pub(super) fn pread(&self, fd: i32, offset: u64, len: usize) -> Option<Vec<u8>> {
        let of = self.open.get(&fd)?;
        if !of.readable {
            return None;
        }
        self.read_range(&of.path.clone(), offset as usize, len)
    }

    /// Write `bytes` to `fd` at absolute `offset` WITHOUT moving the cursor
    /// (sceIoPwrite), extending the file with zeros if needed. Returns the count, or
    /// None on a bad/unwritable descriptor.
    pub(super) fn pwrite(&mut self, fd: i32, offset: u64, bytes: &[u8]) -> Option<usize> {
        let of = self.open.get(&fd)?;
        if !of.writable {
            return None;
        }
        let path = of.path.clone();
        let data = self.make_resident(&path);
        let end = offset as usize + bytes.len();
        if end > data.len() {
            data.resize(end, 0);
        }
        data[offset as usize..end].copy_from_slice(bytes);
        self.touch(&path);
        Some(bytes.len())
    }

    /// Write `bytes` to `fd` at its cursor (extending the file); advances it.
    /// Returns the count written, or None on a bad/unwritable descriptor.
    pub(super) fn write(&mut self, fd: i32, bytes: &[u8]) -> Option<usize> {
        let of = self.open.get_mut(&fd)?;
        if !of.writable {
            return None;
        }
        let (path, cursor) = (of.path.clone(), of.cursor);
        let data = self.make_resident(&path);
        if cursor > data.len() {
            data.resize(cursor, 0);
        }
        let end = cursor + bytes.len();
        if end > data.len() {
            data.resize(end, 0);
        }
        data[cursor..end].copy_from_slice(bytes);
        self.open.get_mut(&fd)?.cursor = end;
        self.touch(&path);
        Some(bytes.len())
    }

    /// Seek `fd` to `offset` from `whence`; returns the new absolute position or a
    /// negative errno.
    pub(super) fn lseek(&mut self, fd: i32, offset: i64, whence: i32) -> i64 {
        // The size lookup borrows `self`, so read the descriptor's path first and let
        // that borrow end before taking the mutable one back for the cursor update.
        let Some(path) = self.open.get(&fd).map(|of| of.path.clone()) else {
            return SCE_ERROR_ERRNO_EBADF as i64;
        };
        let size = self.byte_len(&path).unwrap_or(0) as i64;
        let Some(of) = self.open.get_mut(&fd) else {
            return SCE_ERROR_ERRNO_EBADF as i64;
        };
        let base = match whence {
            SCE_SEEK_SET => 0,
            SCE_SEEK_CUR => of.cursor as i64,
            SCE_SEEK_END => size,
            _ => return SCE_ERROR_ERRNO_EBADF as i64,
        };
        let pos = base + offset;
        if pos < 0 {
            return SCE_ERROR_ERRNO_EBADF as i64;
        }
        of.cursor = pos as usize;
        pos
    }

    /// Close `fd`; returns 0 or a negative errno.
    pub(super) fn close(&mut self, fd: i32) -> i32 {
        if self.open.remove(&fd).is_some() {
            0
        } else {
            SCE_ERROR_ERRNO_EBADF
        }
    }

    /// Open a directory listing (sceIoDopen): returns a new descriptor or a negative
    /// errno if nothing exists under the path.
    ///
    /// The vfs is a flat path->bytes map, so the listing is synthesized: every stored
    /// key under `path/` contributes its next path component, deduplicated - a
    /// component with deeper structure is a subdirectory, a terminal one a file.
    /// Names are returned in their original (as-added) case because titles glob
    /// listings against mixed-case patterns, while dedup/order use the lowercased
    /// form to match the case-insensitive lookup rules. The listing is snapshotted at
    /// open, ordered by lowercased name, and consumed by [`Self::dread`].
    pub(super) fn dopen(&mut self, path: &str) -> i32 {
        let key = vfs_key(path);
        let key = key.trim_end_matches('/');
        let prefix = if key.is_empty() {
            String::new()
        } else {
            format!("{key}/")
        };
        // Child components sit at this index when a full key is split on '/'.
        let comp_index = prefix.matches('/').count();
        let mut children: std::collections::BTreeMap<String, DirEntry> =
            std::collections::BTreeMap::new();
        // Resident and backed files both appear in a listing: a title that stored its
        // assets lazily must still see them when it enumerates a directory, and the
        // sizes have to be right because a title sizes its own read buffers from them.
        let entries: Vec<(&String, usize)> = self
            .files
            .iter()
            .map(|(k, d)| (k, d.len()))
            .chain(
                self.backed
                    .iter()
                    .filter(|(k, _)| !self.files.contains_key(*k))
                    .map(|(k, n)| (k, *n)),
            )
            // A file inside a mounted archive lists like any other: a title that
            // enumerates a directory to find its assets must see the same names the
            // device's own archive layer would hand it. See [`crate::psarc`].
            .chain(self.archived.iter().filter_map(|(k, &(a, e))| {
                let size = self.archives.get(a)?.entries.get(e)?.size as usize;
                Some((k, size))
            }))
            .collect();
        for (k, len) in entries {
            let Some(rest) = k.strip_prefix(&prefix) else {
                continue;
            };
            if rest.is_empty() {
                continue;
            }
            let (comp, is_dir) = match rest.split_once('/') {
                Some((first, _)) => (first, true),
                None => (rest, false),
            };
            let entry = children.entry(comp.to_string()).or_insert_with(|| {
                // Recover the child's original spelling from the as-added path; the
                // lowercased and original paths split identically ('/' survives
                // ASCII lowercasing), so the component index lines up.
                let name = self
                    .originals
                    .get(k)
                    .and_then(|orig| orig.split('/').nth(comp_index))
                    .unwrap_or(comp)
                    .to_string();
                DirEntry {
                    name,
                    is_dir,
                    size: 0,
                }
            });
            if !is_dir {
                entry.size = len as u64;
            }
        }
        // Explicitly-created directories are not implied by any file key, so they have
        // to be folded in separately or a title cannot see the directory it just made.
        for d in &self.dirs {
            let Some(rest) = d.strip_prefix(&prefix) else {
                continue;
            };
            if rest.is_empty() {
                continue;
            }
            let comp = rest.split('/').next().unwrap_or(rest);
            children.entry(comp.to_string()).or_insert_with(|| {
                let name = self
                    .originals
                    .get(d)
                    .and_then(|orig| orig.split('/').nth(comp_index))
                    .unwrap_or(comp)
                    .to_string();
                DirEntry {
                    name,
                    is_dir: true,
                    size: 0,
                }
            });
        }
        // An empty listing is only ENOENT when the directory itself does not exist; a
        // real, explicitly-created empty directory opens fine and reads zero entries.
        if children.is_empty() && !self.dirs.contains(key) {
            return SCE_ERROR_ERRNO_ENOENT;
        }
        let fd = self.next_fd;
        self.next_fd += 1;
        self.open_dirs.insert(
            fd,
            OpenDir {
                entries: children.into_values().collect(),
                cursor: 0,
                path: key.to_string(),
            },
        );
        fd
    }

    /// Read the next entry from a directory descriptor (sceIoDread). Returns
    /// `None` on a bad descriptor, `Some(None)` when the listing is exhausted, and
    /// `Some(Some(entry))` otherwise (advancing the cursor).
    pub(super) fn dread(&mut self, fd: i32) -> Option<Option<DirEntry>> {
        let od = self.open_dirs.get_mut(&fd)?;
        let entry = od.entries.get(od.cursor).cloned();
        if entry.is_some() {
            od.cursor += 1;
        }
        Some(entry)
    }

    /// Whether any stored key lies under `key` (i.e. the directory has content).
    pub(super) fn has_children(&self, key: &str) -> bool {
        let prefix = format!("{key}/");
        self.files.keys().any(|k| k.starts_with(&prefix))
            || self.backed.keys().any(|k| k.starts_with(&prefix))
            || self.dirs.iter().any(|d| d.starts_with(&prefix))
    }

    /// Whether `key` names a directory: one explicitly created, or one implied by a
    /// file stored beneath it.
    pub(super) fn is_dir(&self, key: &str) -> bool {
        self.dirs.contains(key) || self.has_children(key)
    }

    /// int sceIoMkdir(const char *dir, SceMode mode): create a directory.
    ///
    /// Recorded rather than ignored, so the title can see the directory it just made
    /// (list it, stat it, remove it). The parent is not required to exist - the map is
    /// flat and the kernel's own mkdir is likewise a single-level operation whose
    /// failure mode a title checks by return value, which it now gets truthfully.
    pub(super) fn mkdir(&mut self, path: &str) -> i32 {
        let key = vfs_key(path);
        let key = key.trim_end_matches('/').to_string();
        if key.is_empty() {
            return SCE_ERROR_ERRNO_EEXIST;
        }
        if self.has(&key) || self.is_dir(&key) {
            return SCE_ERROR_ERRNO_EEXIST;
        }
        self.originals
            .insert(key.clone(), path.trim_end_matches('/').to_string());
        self.touch(&key);
        self.dirs.insert(key);
        0
    }

    /// int sceIoRmdir(const char *path): remove an EMPTY directory.
    ///
    /// A non-empty one is refused (ENOTEMPTY) exactly as the kernel refuses it -
    /// silently succeeding would tell the title its tree is gone while every file under
    /// it is still readable.
    pub(super) fn rmdir(&mut self, path: &str) -> i32 {
        let key = vfs_key(path);
        let key = key.trim_end_matches('/').to_string();
        if !self.is_dir(&key) {
            return SCE_ERROR_ERRNO_ENOENT;
        }
        if self.has_children(&key) {
            return SCE_ERROR_ERRNO_ENOTEMPTY;
        }
        self.dirs.remove(&key);
        self.originals.remove(&key);
        self.stats.remove(&key);
        self.touch(&key);
        0
    }

    /// int sceIoRemove(const char *file): delete a file.
    ///
    /// Really deletes it: the entry goes, so a later open without SCE_O_CREAT reports
    /// ENOENT and a listing no longer shows it. Refuses a directory (the kernel has
    /// `sceIoRmdir` for that) and an unknown path.
    pub(super) fn remove(&mut self, path: &str) -> i32 {
        let key = vfs_key(path);
        if !self.has(&key) {
            return if self.is_dir(&key) {
                SCE_ERROR_ERRNO_EISDIR
            } else {
                SCE_ERROR_ERRNO_ENOENT
            };
        }
        self.files.remove(&key);
        // A backed file is deleted by forgetting the key: the bytes stay in storage but
        // nothing can reach them, which is what the guest asked for. Storage is the
        // user's installed title and is not ours to erase on a guest unlink.
        self.backed.remove(&key);
        self.originals.remove(&key);
        self.stats.remove(&key);
        self.touch(&key);
        0
    }

    /// int sceIoRename(const char *oldname, const char *newname): move a file or a
    /// whole directory subtree.
    ///
    /// A directory rename has to carry its contents, and in a flat map that means
    /// re-keying every entry under the old prefix - which is the entire reason this
    /// cannot be a two-line map swap. Refuses to overwrite an existing destination,
    /// as the kernel does.
    pub(super) fn rename(&mut self, old: &str, new: &str) -> i32 {
        let (from, to) = (vfs_key(old), vfs_key(new));
        let (from, to) = (
            from.trim_end_matches('/').to_string(),
            to.trim_end_matches('/').to_string(),
        );
        if from == to {
            return 0;
        }
        if self.has(&to) || self.is_dir(&to) {
            return SCE_ERROR_ERRNO_EEXIST;
        }
        let new_orig = new.trim_end_matches('/').to_string();

        // A rename re-keys the file, and the backing only knows its ORIGINAL key - so a
        // backed file has to be faulted in first or its new name would read as empty.
        // Deliberately the expensive path: renaming a shipped asset does not happen in
        // practice, and a silently unreadable file would be far worse than a copy.
        for key in self.backed_under(&from) {
            self.make_resident(&key);
        }

        if let Some(data) = self.files.remove(&from) {
            self.files.insert(to.clone(), data);
            self.originals.remove(&from);
            self.originals.insert(to.clone(), new_orig);
            if let Some(s) = self.stats.remove(&from) {
                self.stats.insert(to.clone(), s);
            }
            // BOTH ends: a move OFF a persisted mount changes the save just as much as a
            // move onto one, and marking only the destination would leave the departure
            // unflushed.
            self.touch(&from);
            self.touch(&to);
            return 0;
        }
        if !self.is_dir(&from) {
            return SCE_ERROR_ERRNO_ENOENT;
        }
        // A directory: re-key the directory itself and everything beneath it.
        let old_prefix = format!("{from}/");
        let rekey = |k: &str| format!("{to}{}", &k[from.len()..]);
        let moved: Vec<String> = self
            .files
            .keys()
            .filter(|k| k.starts_with(&old_prefix))
            .cloned()
            .collect();
        for k in moved {
            let data = self.files.remove(&k).unwrap_or_default();
            let nk = rekey(&k);
            // The original spelling's tail is preserved; only the renamed prefix changes.
            let orig_tail = self
                .originals
                .remove(&k)
                .map(|o| o[from.len().min(o.len())..].to_string());
            self.files.insert(nk.clone(), data);
            if let Some(tail) = orig_tail {
                self.originals
                    .insert(nk.clone(), format!("{new_orig}{tail}"));
            }
            if let Some(s) = self.stats.remove(&k) {
                self.stats.insert(nk, s);
            }
        }
        let moved_dirs: Vec<String> = self
            .dirs
            .iter()
            .filter(|d| **d == from || d.starts_with(&old_prefix))
            .cloned()
            .collect();
        for d in moved_dirs {
            self.dirs.remove(&d);
            self.dirs.insert(rekey(&d));
        }
        self.originals.remove(&from);
        self.touch(&from);
        self.touch(&to);
        self.originals.insert(to, new_orig);
        0
    }

    /// int sceIoChstat(const char *name, SceIoStat *stat, int bits): apply the
    /// selected fields of `stat` to a path. Stored so a later `sceIoGetstat` reports
    /// them back - a chstat whose effect is invisible is worse than an error, because
    /// the title believes the change took.
    pub(super) fn chstat(&mut self, path: &str, over: FileStatOverride) -> i32 {
        let key = vfs_key(path);
        if !self.has(&key) && !self.is_dir(&key) {
            return SCE_ERROR_ERRNO_ENOENT;
        }
        self.touch(&key);
        let e = self.stats.entry(key).or_default();
        if over.mode.is_some() {
            e.mode = over.mode;
        }
        if over.attr.is_some() {
            e.attr = over.attr;
        }
        for (i, t) in over.times.iter().enumerate() {
            if t.is_some() {
                e.times[i] = *t;
            }
        }
        0
    }

    /// The chstat overrides recorded for a path, if any.
    pub(super) fn stat_override(&self, path: &str) -> Option<&FileStatOverride> {
        self.stats.get(&vfs_key(path))
    }

    /// Close a directory descriptor (sceIoDclose); returns 0 or a negative errno.
    pub(super) fn dclose(&mut self, fd: i32) -> i32 {
        if self.open_dirs.remove(&fd).is_some() {
            0
        } else {
            SCE_ERROR_ERRNO_EBADF
        }
    }

    /// The size of `path` if it exists (for sceIoGetstat).
    pub(super) fn size_of(&self, path: &str) -> Option<u64> {
        self.byte_len(&vfs_key(path)).map(|n| n as u64)
    }

    /// The whole contents of `path` if it exists, cloned. For consumers that need a
    /// file's bytes in one shot without managing a descriptor (e.g. loading a font).
    pub(super) fn read_all(&self, path: &str) -> Option<Vec<u8>> {
        let key = vfs_key(path);
        let n = self.byte_len(&key)?;
        self.read_range(&key, 0, n)
    }

    /// The size of the file behind an open descriptor (for sceIoGetstatByFd).
    /// Returns None on a bad descriptor - the path is already the vfs key.
    pub(super) fn size_of_fd(&self, fd: i32) -> Option<u64> {
        let of = self.open.get(&fd)?;
        self.byte_len(&of.path).map(|n| n as u64)
    }

    /// `chstat` on an open descriptor: the same change, addressed by fd.
    pub(super) fn chstat_fd(&mut self, fd: i32, over: FileStatOverride) -> i32 {
        let Some(of) = self.open.get(&fd) else {
            return SCE_ERROR_ERRNO_EBADF;
        };
        let path = of.path.clone();
        self.chstat(&path, over)
    }

    /// Everything on a persisted mount, for [`crate::gamedata`].
    ///
    /// RESIDENT files only, and that is the whole point rather than a shortcut: a backed
    /// file is one the installed title shipped and this must never carry a byte of those.
    /// A guest write faults its subject resident (see [`make_resident`](Self::make_resident)),
    /// so anything the guest actually wrote IS here.
    ///
    /// Sorted, so two exports of an unchanged run are byte-identical.
    pub(super) fn collect_game_data(&self) -> crate::gamedata::GameData {
        let mut out = crate::gamedata::GameData::default();
        for (key, data) in &self.files {
            if is_persisted_mount(key) {
                out.files.push((key.clone(), data.clone()));
            }
        }
        out.files.sort_by(|a, b| a.0.cmp(&b.0));
        out.dirs = self
            .dirs
            .iter()
            .filter(|k| is_persisted_mount(k))
            .cloned()
            .collect();
        out.dirs.sort();
        out.originals = self
            .originals
            .iter()
            .filter(|(k, _)| is_persisted_mount(k))
            .map(|(k, o)| (k.clone(), o.clone()))
            .collect();
        out.originals.sort();
        out.stats = self
            .stats
            .iter()
            .filter(|(k, _)| is_persisted_mount(k))
            .map(|(k, v)| (k.clone(), v.clone()))
            .collect();
        out.stats.sort_by(|a, b| a.0.cmp(&b.0));
        out
    }

    /// Put a container's file half back. Keys are re-checked here even though
    /// [`crate::gamedata::GameData::from_zip`] already refused everything else - this is
    /// the function that actually touches the table, so it is the one that must hold the
    /// invariant even if it is ever called from somewhere new.
    ///
    /// Inserts DIRECTLY rather than through `open`/`write`: going through the guest-facing
    /// path would mark the table dirty and provoke a write-back of what was just read in.
    pub(super) fn restore_game_data(&mut self, data: &crate::gamedata::GameData) -> usize {
        let mut n = 0;
        for (key, bytes) in &data.files {
            if !is_persisted_mount(key) {
                continue;
            }
            self.backed.remove(key);
            self.files.insert(key.clone(), bytes.clone());
            n += 1;
        }
        for key in &data.dirs {
            if is_persisted_mount(key) {
                self.dirs.insert(key.clone());
            }
        }
        for (key, orig) in &data.originals {
            if is_persisted_mount(key) {
                self.originals.insert(key.clone(), orig.clone());
            }
        }
        for (key, over) in &data.stats {
            if is_persisted_mount(key) {
                self.stats.insert(key.clone(), over.clone());
            }
        }
        n
    }
}
