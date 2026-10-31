//! The SceIoFilemgr virtual filesystem in `VitaState`, exercised directly via
//! the public io_* methods. The end-to-end proof on a real velf is
//! `vitaslop-conformance-harness/tests/vita_io.rs`; these cover the paths that
//! artifact does not (SEEK_CUR/END, append, truncate, bad fd, stderr).
use super::*;
use crate::world::DeterministicWorld;

fn state() -> VitaState {
    VitaState::new(0, 64, Box::new(DeterministicWorld::default()))
}

/// A backing built the way a real one must be: it lists paths in their own spelling
/// and resolves lookups through [`vfs_key`], exactly as an OPFS or disk backing has
/// to. Keying it by the raw path instead would make every mixed-case test pass
/// against a backing that could never work in practice.
struct MapBacking {
    /// Storage spelling -> bytes, in listing order.
    stored: Vec<(String, Vec<u8>)>,
    /// Normalised key -> index into `stored`.
    by_key: std::collections::HashMap<String, usize>,
}

impl MapBacking {
    fn new(files: &[(&str, Vec<u8>)]) -> Self {
        let stored: Vec<(String, Vec<u8>)> = files
            .iter()
            .map(|(p, b)| (p.to_string(), b.clone()))
            .collect();
        let by_key = stored
            .iter()
            .enumerate()
            .map(|(i, (p, _))| (vfs_key(p), i))
            .collect();
        MapBacking { stored, by_key }
    }
    fn get(&self, key: &str) -> Option<&[u8]> {
        self.by_key.get(key).map(|&i| self.stored[i].1.as_slice())
    }
}

impl FileBacking for MapBacking {
    fn len(&self, key: &str) -> Option<usize> {
        self.get(key).map(|v| v.len())
    }
    fn read_at(&self, key: &str, off: usize, buf: &mut [u8]) -> usize {
        let Some(v) = self.get(key) else { return 0 };
        let end = (off + buf.len()).min(v.len());
        let n = end.saturating_sub(off);
        buf[..n].copy_from_slice(&v[off..end]);
        n
    }
    fn keys(&self) -> Vec<String> {
        self.stored.iter().map(|(p, _)| p.clone()).collect()
    }
}

fn backed_state(path: &str, bytes: &[u8]) -> VitaState {
    let mut st = state();
    st.set_file_backing(Box::new(MapBacking::new(&[(path, bytes.to_vec())])));
    st
}

/// A lazily-backed file must be indistinguishable from a resident one through the
/// whole SceIo surface. This is the property the browser's memory budget rests on,
/// and every way it can go wrong is silent: a title that reads a truncated asset, or
/// sizes a buffer from a zero stat, fails somewhere else entirely.
#[test]
fn a_backed_file_reads_seeks_and_stats_like_a_resident_one() {
    let data: Vec<u8> = (0..=255u8).collect();
    let mut st = backed_state("ux0:/data/asset.bin", &data);

    // It exists, and its size is known WITHOUT reading it.
    assert_eq!(st.io_size("ux0:/data/asset.bin"), Some(256));

    let fd = st.io_open("ux0:/data/asset.bin", SCE_O_RDONLY);
    assert!(fd >= 0, "a backed file must open for reading");

    // Sequential reads advance the cursor and deliver the real bytes.
    assert_eq!(st.io_read(fd, 4), Some(vec![0, 1, 2, 3]));
    assert_eq!(st.io_read(fd, 4), Some(vec![4, 5, 6, 7]));

    // Seek from the end, then read across it: a short read at end of file must be
    // short, not padded, and must leave the cursor AT the end.
    assert_eq!(st.io_lseek(fd, -2, SCE_SEEK_END), 254);
    assert_eq!(st.io_read(fd, 8), Some(vec![254, 255]));
    assert_eq!(st.io_read(fd, 8), Some(vec![]));

    // Positional reads do not disturb the cursor.
    assert_eq!(st.io_lseek(fd, 10, SCE_SEEK_SET), 10);
    assert_eq!(st.io_pread(fd, 100, 3), Some(vec![100, 101, 102]));
    assert_eq!(st.io_read(fd, 2), Some(vec![10, 11]));
}

/// A write to a backed file faults it in and then behaves exactly as it always did.
/// Worth pinning because the fault-in is the one place the two storage models meet,
/// and a read-after-write that returned the ORIGINAL bytes would look like a
/// corrupted save rather than a filesystem bug.
#[test]
fn writing_a_backed_file_faults_it_in_and_read_after_write_is_exact() {
    let mut st = backed_state("ux0:/data/asset.bin", &[9u8; 16]);
    let fd = st.io_open("ux0:/data/asset.bin", SCE_O_RDWR);
    assert!(fd >= 0);
    assert_eq!(st.io_lseek(fd, 4, SCE_SEEK_SET), 4);
    assert_eq!(st.io_write(fd, &[1, 2, 3]), Some(3));
    assert_eq!(st.io_lseek(fd, 0, SCE_SEEK_SET), 0);
    assert_eq!(st.io_read(fd, 8), Some(vec![9, 9, 9, 9, 1, 2, 3, 9]));
    // Faulted in, so the inspection seam can now see it.
    assert_eq!(
        st.file_bytes("ux0:/data/asset.bin").map(|b| b.len()),
        Some(16)
    );
}

/// A backed file whose stored name has CAPITALS must be reachable by the guest, which
/// asks case-insensitively through an `app0:` mount.
///
/// This is the exact bug that shipped and cost the most to find. The guest filesystem
/// normalises every path (lowercased, `app0:` stripped) before it reaches a backing,
/// while storage keeps the name as written. Get the two out of step and the file is
/// either missing or reads as nothing - and what surfaces is a guest memory-access
/// trap 30,000 host calls later, with nothing anywhere pointing at the filesystem. It
/// went wrong twice in a row, in opposite directions, so both ends are pinned here.
#[test]
fn a_backed_file_is_reachable_through_a_mount_and_a_different_case() {
    let data: Vec<u8> = (0..64u8).collect();
    let mut st = state();
    st.set_file_backing(Box::new(MapBacking::new(&[(
        "PSP2/Data/Blob.BIN",
        data.clone(),
    )])));

    // The spelling the title actually uses: an app0: mount, and whatever case it feels
    // like. Every one of these is the same file.
    for path in [
        "app0:/PSP2/Data/Blob.BIN",
        "app0:/psp2/data/blob.bin",
        "PSP2/Data/Blob.BIN",
    ] {
        assert_eq!(st.io_size(path), Some(64), "size of {path}");
        let fd = st.io_open(path, SCE_O_RDONLY);
        assert!(fd >= 0, "{path} must open (got {fd})");
        assert_eq!(
            st.io_read(fd, 8),
            Some(vec![0, 1, 2, 3, 4, 5, 6, 7]),
            "read of {path}"
        );
    }
}

/// A backed file has to show up in a directory listing with its real size, because a
/// title enumerates its own data directory and sizes read buffers from what it finds.
#[test]
fn a_backed_file_appears_in_a_listing_with_its_size() {
    let mut st = backed_state("ux0:/data/Asset.BIN", &[0u8; 42]);
    let fd = st.io_dopen("ux0:/data");
    assert!(fd >= 0, "the implied directory of a backed file must open");
    let e = st.io_dread(fd).expect("a valid dir fd").expect("one entry");
    assert_eq!(e.name, "Asset.BIN", "the as-supplied spelling must survive");
    assert_eq!(e.size, 42);
    assert!(!e.is_dir);
}

#[test]
fn write_then_read_back_via_the_vfs() {
    let mut st = state();
    let fd = st.io_open("ux0:/a", SCE_O_WRONLY | SCE_O_CREAT);
    assert!(fd >= 3);
    assert_eq!(st.io_write(fd, b"abcdef"), Some(6));
    assert_eq!(st.io_close(fd), 0);
    assert_eq!(st.io_size("ux0:/a"), Some(6));

    let r = st.io_open("ux0:/a", SCE_O_RDONLY);
    assert_eq!(st.io_lseek(r, 2, SCE_SEEK_SET), 2);
    assert_eq!(st.io_read(r, 100).as_deref(), Some(&b"cdef"[..]));
    // At EOF a further read is empty, not an error.
    assert_eq!(st.io_read(r, 100).as_deref(), Some(&b""[..]));
    st.io_close(r);
}

#[test]
fn doubled_slashes_and_dot_segments_resolve_to_the_same_file() {
    // A title that joins a dir with a trailing `/` to a subpath with a leading `/`
    // produces `//`; a real Vita FS collapses it. The font-load path
    // `app0:/usrdir//ui/fonts//x.ttf` regressed a whole title before this
    // normalization landed (font missed -> every glyph "missing" -> null-glyph copy).
    let mut st = state();
    st.add_file("app0:/usrdir/ui/fonts/x.ttf", b"FONTDATA".to_vec());
    assert_eq!(
        st.read_file("app0:/usrdir//ui/fonts//x.ttf").as_deref(),
        Some(&b"FONTDATA"[..])
    );
    // `.` and `..` segments resolve too, and open() shares the same key.
    assert_eq!(
        st.read_file("app0:/usrdir/./ui/bogus/../fonts/x.ttf")
            .as_deref(),
        Some(&b"FONTDATA"[..])
    );
    assert!(st.io_open("app0:/usrdir//ui/fonts//x.ttf", SCE_O_RDONLY) >= 3);
}

#[test]
fn seek_cur_end_and_negative_is_error() {
    let mut st = state();
    st.add_file("app0:/f", b"0123456789".to_vec());
    let fd = st.io_open("app0:/f", SCE_O_RDONLY);
    assert_eq!(st.io_lseek(fd, 3, SCE_SEEK_SET), 3);
    assert_eq!(st.io_lseek(fd, 2, SCE_SEEK_CUR), 5);
    assert_eq!(st.io_lseek(fd, -1, SCE_SEEK_END), 9);
    assert_eq!(st.io_read(fd, 4).as_deref(), Some(&b"9"[..]));
    // Seeking before the start is a negative errno, leaving the cursor put.
    assert!(st.io_lseek(fd, -100, SCE_SEEK_SET) < 0);
}

#[test]
fn append_seeks_to_end_and_truncate_clears() {
    let mut st = state();
    st.add_file("ux0:/log", b"head".to_vec());
    let a = st.io_open("ux0:/log", SCE_O_WRONLY | SCE_O_APPEND);
    assert_eq!(st.io_write(a, b"-tail"), Some(5));
    st.io_close(a);
    assert_eq!(st.file_bytes("ux0:/log"), Some(&b"head-tail"[..]));

    // Opening with TRUNC discards the old contents.
    let t = st.io_open("ux0:/log", SCE_O_WRONLY | SCE_O_TRUNC);
    assert_eq!(st.io_write(t, b"new"), Some(3));
    st.io_close(t);
    assert_eq!(st.file_bytes("ux0:/log"), Some(&b"new"[..]));
}

#[test]
fn missing_file_and_bad_fd_are_errors() {
    let mut st = state();
    assert!(st.io_open("app0:/nope", SCE_O_RDONLY) < 0);
    assert!(st.io_read(999, 4).is_none());
    assert!(st.io_write(999, b"x").is_none());
    assert!(st.io_lseek(999, 0, SCE_SEEK_SET) < 0);
    assert!(st.io_close(999) < 0);
}

#[test]
fn uppercase_app0_mount_finds_the_same_file() {
    // Titles spell the mount both ways; both must resolve to the same key.
    let mut st = state();
    st.add_file("Disc/Data/File.bin", b"x".to_vec());
    assert!(st.io_open("APP0:Disc/Data/File.bin", SCE_O_RDONLY) >= 3);
    assert!(st.io_open("app0:/disc/data/file.bin", SCE_O_RDONLY) >= 3);
    assert_eq!(st.io_size("APP0:Disc/Data/File.bin"), Some(1));
}

/// An add-on's files appear under `addcont0:/<dir>/` beside the title's own (served by a
/// second backing), a listing of `addcont0:` names each installed add-on as a directory
/// in its own spelling, and only an installed add-on is reported entitled.
#[test]
fn an_addon_is_listed_and_read_under_its_mount() {
    let mut st = state();
    st.set_file_backing(Box::new(MapBacking::new(&[("data/base.bin", vec![1; 4])])));
    assert_eq!(
        st.io_dopen("addcont0:"),
        SCE_ERROR_ERRNO_ENOENT,
        "no add-on, no mount"
    );
    st.add_addcont(
        "PACK000000000001",
        Box::new(MapBacking::new(&[
            ("Costume.bcm", vec![7, 8, 9]),
            ("sce_sys/param.sfo", vec![0; 2]),
        ])),
    );
    assert!(st.has_addcont("pack000000000001") && !st.has_addcont("PACK000000000002"));
    let fd = st.io_dopen("addcont0:");
    let e = st.io_dread(fd).unwrap().unwrap();
    assert_eq!((e.name.as_str(), e.is_dir), ("PACK000000000001", true));
    assert_eq!(st.io_dread(fd), Some(None));
    // Both separators, any case - and the title's own files are untouched.
    assert_eq!(
        st.read_file("addcont0:PACK000000000001/costume.bcm")
            .as_deref(),
        Some(&[7u8, 8, 9][..])
    );
    assert_eq!(
        st.read_file("addcont0:/pack000000000001/Costume.bcm")
            .as_deref(),
        Some(&[7u8, 8, 9][..])
    );
    assert_eq!(
        st.read_file("app0:data/base.bin").as_deref(),
        Some(&[1u8; 4][..])
    );
    assert!(
        st.read_file("costume.bcm").is_none(),
        "an add-on's files are not the title's"
    );
}

#[test]
fn dopen_lists_children_with_original_case() {
    let mut st = state();
    st.add_file("Disc/Course/Course_601/a.dat", vec![0; 3]);
    st.add_file("Disc/Course/Course_601/b.dat", vec![0; 4]);
    st.add_file("Disc/Course/Course_700/a.dat", vec![0; 5]);
    st.add_file("Disc/Course/Readme.txt", vec![0; 7]);

    let fd = st.io_dopen("APP0:Disc/Course");
    assert!(fd >= 3);
    // Ordered by lowercased name; subdirectories deduplicated; names keep the
    // original mixed case a title's glob (e.g. `Course_*`) expects.
    let e1 = st.io_dread(fd).unwrap().unwrap();
    assert_eq!(
        (e1.name.as_str(), e1.is_dir, e1.size),
        ("Course_601", true, 0)
    );
    let e2 = st.io_dread(fd).unwrap().unwrap();
    assert_eq!(
        (e2.name.as_str(), e2.is_dir, e2.size),
        ("Course_700", true, 0)
    );
    let e3 = st.io_dread(fd).unwrap().unwrap();
    assert_eq!(
        (e3.name.as_str(), e3.is_dir, e3.size),
        ("Readme.txt", false, 7)
    );
    // Exhausted: end-of-listing, repeatably.
    assert_eq!(st.io_dread(fd), Some(None));
    assert_eq!(st.io_dread(fd), Some(None));
    assert_eq!(st.io_dclose(fd), 0);
    // Closed (and never-opened) descriptors are bad.
    assert!(st.io_dread(fd).is_none());
    assert!(st.io_dclose(fd) < 0);
}

#[test]
fn dopen_of_a_subdir_and_missing_dir() {
    let mut st = state();
    st.add_file("Disc/Course/Course_601/Deep/x.bin", vec![0; 2]);
    // A trailing slash and mixed-case path still resolve.
    let fd = st.io_dopen("app0:Disc/COURSE/Course_601/");
    assert!(fd >= 3);
    let e = st.io_dread(fd).unwrap().unwrap();
    assert_eq!((e.name.as_str(), e.is_dir), ("Deep", true));
    assert_eq!(st.io_dread(fd), Some(None));
    st.io_dclose(fd);
    // Nothing under the path: a negative errno.
    assert!(st.io_dopen("app0:Disc/Nope") < 0);
    // A file fd is not a directory fd and vice versa.
    let ffd = st.io_open("app0:disc/course/course_601/deep/x.bin", SCE_O_RDONLY);
    assert!(st.io_dread(ffd).is_none());
}

#[test]
fn fd_one_and_two_route_to_captured_console() {
    let mut st = state();
    assert_eq!(st.io_write(FD_STDOUT, b"out"), Some(3));
    assert_eq!(st.io_write(FD_STDERR, b"err"), Some(3));
    assert_eq!(st.capture.stdout, b"out");
    assert_eq!(st.capture.stderr, b"err");
}

#[test]
fn pread_is_positioned_and_leaves_the_cursor_put() {
    // sceIoPread reads at an absolute offset without disturbing the descriptor
    // cursor - the AT9 music streamer relies on this to pull the header and then
    // interleave chunk reads on one shared fd.
    let mut st = state();
    st.add_file("app0:/song.at9", b"RIFF....fmt ....data".to_vec());
    let fd = st.io_open("app0:/song.at9", SCE_O_RDONLY);
    // A normal read advances the cursor to 4.
    assert_eq!(st.io_read(fd, 4).as_deref(), Some(&b"RIFF"[..]));
    // pread at offset 8 does not move the cursor...
    assert_eq!(st.io_pread(fd, 8, 4).as_deref(), Some(&b"fmt "[..]));
    // ...so the next sequential read continues from 4.
    assert_eq!(st.io_read(fd, 4).as_deref(), Some(&b"...."[..]));
    // Reading past the end clamps to the available bytes.
    assert_eq!(st.io_pread(fd, 16, 100).as_deref(), Some(&b"data"[..]));
    // A bad fd yields None (mapped to EBADF by the handler).
    assert!(st.io_pread(999, 0, 4).is_none());
}

#[test]
fn pwrite_is_positioned_and_extends_with_zeros() {
    let mut st = state();
    st.add_file("ux0:/save.bin", vec![0u8; 4]);
    let fd = st.io_open("ux0:/save.bin", SCE_O_RDWR);
    // Positioned write past the current end zero-fills the gap.
    assert_eq!(st.io_pwrite(fd, 6, b"AB"), Some(2));
    assert_eq!(
        st.file_bytes("ux0:/save.bin"),
        Some(&[0, 0, 0, 0, 0, 0, b'A', b'B'][..])
    );
    // The cursor is untouched, so a sequential read still starts at 0.
    assert_eq!(st.io_read(fd, 2).as_deref(), Some(&[0, 0][..]));
}
