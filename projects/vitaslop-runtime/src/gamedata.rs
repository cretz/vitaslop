//! The guest's own saved state, as a portable container.
//!
//! # What this is, and the line it does not cross
//! A console keeps two very different things on its storage: the TITLE, which the user
//! installed and which never changes, and the title's SAVED STATE, which the guest writes
//! as it plays. This module carries the second one and NOTHING else.
//!
//! That boundary is not a convention here, it is enforced twice. [`GameData::collect`]
//! only ever reads keys on a persisted mount ([`is_persisted_mount`]) - `savedata0:` and
//! `ux0:` - and [`GameData::from_zip`] REFUSES any entry whose key is not on one, counting
//! the refusal rather than dropping it silently. `app0:` is the game's own read-only tree
//! and its keys are stripped of the mount by [`vfs_key`], so they cannot pass either test.
//! An import is therefore incapable of modifying the installed title, whatever the
//! container claims - which matters because a container is a file a user can hand to
//! another user.
//!
//! # Why a zip
//! It is what the user does things with: downloads it, mails it to themselves, opens it to
//! see whether their save is really in there, uploads it to another browser or another
//! device. Every one of those is a property of the format rather than of this emulator.
//! The writer is [`crate::ingest::zip::write_zip`] - the ROM path already had the reader,
//! so the format costs no dependency and no wasm size worth naming.
//!
//! # The layout inside
//! ```text
//! README.txt        what this file is, in prose, for whoever opens it
//! files/<key>       one entry per resident guest file, ':' escaped as %3A
//! dirs.txt          keys of directories the guest made that hold nothing
//! originals.txt     key<TAB>original-case spelling, for the guest's own globbing
//! stats.bin         sceIoChstat overrides
//! slots.bin         SceAppUtil savedata slot params
//! trophies.bin      unlocked trophy ids and the tick each was earned at
//! ```
//! The binary sidecars are length-prefixed records rather than JSON because this crate has
//! no JSON, and inventing one to carry three integers would be more code than the records.
//! Each is optional: an absent entry means that kind of state is empty, so an older
//! container still imports into a newer build.

use crate::host::{is_persisted_mount, vfs_key, FileStatOverride};
use crate::ingest::vfs::Vfs;

/// The container's format version, written into `README.txt` and checked on import.
/// Bumped only for a change that an older build could MISREAD - a new optional entry is
/// not one, because an unknown entry is ignored and a missing one is empty.
pub const FORMAT: u32 = 1;

/// Everything a run persists, pulled out of the live emulator or read from a container.
///
/// Ordered vectors rather than maps throughout: the container's bytes must be a function
/// of the state alone, so that exporting twice without playing produces identical files
/// and a test can say so. A `HashMap` iteration order would make that false.
#[derive(Default, Debug, Clone, PartialEq, Eq)]
pub struct GameData {
    /// Resident guest files on a persisted mount, as (normalised key, bytes).
    pub files: Vec<(String, Vec<u8>)>,
    /// Directories the guest created that hold nothing - unrepresentable in a flat
    /// path map, so carried explicitly (see `FileTable::dirs`).
    pub dirs: Vec<String>,
    /// (key, the as-created spelling), so a `sceIoDread` after a reload hands the title
    /// back the same mixed-case names it globbed against before one.
    pub originals: Vec<(String, String)>,
    /// `sceIoChstat` overrides, per key.
    pub stats: Vec<(String, FileStatOverride)>,
    /// SceAppUtil savedata slots: (mount, slot id, the title's own param blob).
    pub slots: Vec<(String, u32, Vec<u8>)>,
    /// Unlocked trophies: (NP communication id, [(trophy id, SceRtcTick earned at)]).
    pub trophies: Vec<(String, Vec<(u32, u64)>)>,
}

/// What an import actually did, so a refusal is reported rather than inferred from a save
/// that quietly did not come back.
#[derive(Default, Debug, Clone, PartialEq, Eq)]
pub struct ImportReport {
    pub files: usize,
    pub bytes: usize,
    pub dirs: usize,
    pub stats: usize,
    pub slots: usize,
    pub trophies: usize,
    /// Entries refused because their key is not on a persisted mount - i.e. a container
    /// that tried to write somewhere only the installed title lives. Named, not counted:
    /// one of these means the file came from somewhere it should not have.
    pub refused: Vec<String>,
}

impl ImportReport {
    /// One line for the page and the log.
    pub fn summary(&self) -> String {
        let mut s = format!(
            "{} file(s) ({} bytes), {} dir(s), {} stat(s), {} slot(s), {} trophy set(s)",
            self.files, self.bytes, self.dirs, self.stats, self.slots, self.trophies
        );
        if !self.refused.is_empty() {
            s.push_str(&format!(
                " - REFUSED {} entr(y/ies) that name something outside the guest's saved \
                 state, which this never writes: {}",
                self.refused.len(),
                self.refused.join(", ")
            ));
        }
        s
    }
}

impl GameData {
    /// Whether there is anything at all to persist.
    pub fn is_empty(&self) -> bool {
        self.files.is_empty()
            && self.dirs.is_empty()
            && self.stats.is_empty()
            && self.slots.is_empty()
            && self.trophies.is_empty()
    }

    /// Total bytes of guest file content (not of the container).
    pub fn bytes(&self) -> usize {
        self.files.iter().map(|(_, d)| d.len()).sum()
    }

    /// One line describing the contents, for a status line or a log.
    pub fn summary(&self) -> String {
        format!(
            "{} file(s), {} bytes, {} dir(s), {} slot(s), {} trophy set(s)",
            self.files.len(),
            self.bytes(),
            self.dirs.len(),
            self.slots.len(),
            self.trophies.len()
        )
    }

    /// Serialise to a `.zip`. `title` is the title id, for the README only.
    pub fn to_zip(&self, title: &str) -> Vec<u8> {
        let mut entries: Vec<(String, Vec<u8>)> = Vec::new();
        entries.push((
            "README.txt".to_string(),
            format!(
                "vitaslop game data, format {FORMAT}\r\n\
                 title: {title}\r\n\
                 contents: {}\r\n\
                 \r\n\
                 This archive holds ONLY what the game itself saved - its savedata mount and \
                 its trophy unlocks. It contains no part of the game, and importing one \
                 cannot change the installed title: every entry whose path is not on the \
                 guest's writable mounts is refused on the way back in.\r\n",
                self.summary()
            )
            .into_bytes(),
        ));

        for (key, data) in &self.files {
            entries.push((format!("files/{}", escape(key)), data.clone()));
        }
        if !self.dirs.is_empty() {
            entries.push(("dirs.txt".to_string(), self.dirs.join("\n").into_bytes()));
        }
        if !self.originals.is_empty() {
            let text = self
                .originals
                .iter()
                .map(|(k, o)| format!("{k}\t{o}"))
                .collect::<Vec<_>>()
                .join("\n");
            entries.push(("originals.txt".to_string(), text.into_bytes()));
        }
        if !self.stats.is_empty() {
            let mut out = Vec::new();
            for (key, s) in &self.stats {
                put_str(&mut out, key);
                // One presence byte, then every field in a fixed layout. Writing an absent
                // field as zeros and remembering WHICH were absent is what keeps a chstat
                // that set only the times from also resetting the mode on reload.
                let mut flags = 0u8;
                if s.mode.is_some() {
                    flags |= 1;
                }
                if s.attr.is_some() {
                    flags |= 2;
                }
                for (i, t) in s.times.iter().enumerate() {
                    if t.is_some() {
                        flags |= 4 << i;
                    }
                }
                out.push(flags);
                out.extend_from_slice(&s.mode.unwrap_or(0).to_le_bytes());
                out.extend_from_slice(&s.attr.unwrap_or(0).to_le_bytes());
                for t in &s.times {
                    out.extend_from_slice(&t.unwrap_or([0u8; 16]));
                }
            }
            entries.push(("stats.bin".to_string(), out));
        }
        if !self.slots.is_empty() {
            let mut out = Vec::new();
            for (mount, id, param) in &self.slots {
                put_str(&mut out, mount);
                out.extend_from_slice(&id.to_le_bytes());
                put_bytes(&mut out, param);
            }
            entries.push(("slots.bin".to_string(), out));
        }
        if !self.trophies.is_empty() {
            let mut out = Vec::new();
            for (comm_id, list) in &self.trophies {
                put_str(&mut out, comm_id);
                out.extend_from_slice(&(list.len() as u32).to_le_bytes());
                for (id, tick) in list {
                    out.extend_from_slice(&id.to_le_bytes());
                    out.extend_from_slice(&tick.to_le_bytes());
                }
            }
            entries.push(("trophies.bin".to_string(), out));
        }
        crate::ingest::zip::write_zip(&entries)
    }

    /// Parse a container, refusing anything that names a path outside the guest's own
    /// saved state. Returns the data and what was refused.
    ///
    /// A malformed sidecar is an ERROR, not a silent skip: the alternative is a restore
    /// that loses a title's trophies and reports success.
    pub fn from_zip(bytes: &[u8]) -> Result<(GameData, Vec<String>), String> {
        let vfs = crate::ingest::zip::read_zip(bytes)
            .map_err(|e| format!("this is not a readable zip archive ({e:?})"))?;
        let mut out = GameData::default();
        let mut refused = Vec::new();

        // Sorted, so a container's own entry order cannot change what is produced.
        let mut names = vfs.list();
        names.sort();

        for name in &names {
            let Some(rest) = name.strip_prefix("files/") else { continue };
            let raw = unescape(rest);
            let key = vfs_key(&raw);
            // >>> THE GUARD. A key that is not on a writable mount names the installed
            // title (or, after `vfs_key` has resolved its `..`, nothing at all), and this
            // is the one place a container's own words could otherwise reach storage.
            if key.is_empty() || !is_persisted_mount(&key) {
                refused.push(raw);
                continue;
            }
            let data = vfs.read(name).map_err(|e| format!("unreadable entry {name:?}: {e:?}"))?;
            out.files.push((key, data));
        }

        if let Ok(text) = vfs.read("dirs.txt") {
            for line in String::from_utf8_lossy(&text).lines() {
                let key = vfs_key(line.trim());
                if key.is_empty() {
                    continue;
                }
                if is_persisted_mount(&key) {
                    out.dirs.push(key);
                } else {
                    refused.push(line.trim().to_string());
                }
            }
        }

        if let Ok(text) = vfs.read("originals.txt") {
            for line in String::from_utf8_lossy(&text).lines() {
                let Some((k, orig)) = line.split_once('\t') else { continue };
                let key = vfs_key(k);
                if !key.is_empty() && is_persisted_mount(&key) {
                    out.originals.push((key, orig.to_string()));
                }
            }
        }

        if let Ok(bin) = vfs.read("stats.bin") {
            let mut r = Cursor::new(&bin);
            while !r.done() {
                let key = r.str("stats.bin key")?;
                let flags = r.u8("stats.bin flags")?;
                let mode = r.u32("stats.bin mode")?;
                let attr = r.u32("stats.bin attr")?;
                let mut times = [None; 3];
                for (i, slot) in times.iter_mut().enumerate() {
                    let raw = r.fixed16("stats.bin time")?;
                    if flags & (4 << i) != 0 {
                        *slot = Some(raw);
                    }
                }
                let key = vfs_key(&key);
                if key.is_empty() || !is_persisted_mount(&key) {
                    refused.push(key);
                    continue;
                }
                out.stats.push((
                    key,
                    FileStatOverride {
                        mode: (flags & 1 != 0).then_some(mode),
                        attr: (flags & 2 != 0).then_some(attr),
                        times,
                    },
                ));
            }
        }

        if let Ok(bin) = vfs.read("slots.bin") {
            let mut r = Cursor::new(&bin);
            while !r.done() {
                let mount = r.str("slots.bin mount")?;
                let id = r.u32("slots.bin id")?;
                let param = r.bytes("slots.bin param")?;
                out.slots.push((mount, id, param));
            }
        }

        if let Ok(bin) = vfs.read("trophies.bin") {
            let mut r = Cursor::new(&bin);
            while !r.done() {
                let comm_id = r.str("trophies.bin comm id")?;
                let n = r.u32("trophies.bin count")? as usize;
                let mut list = Vec::with_capacity(n.min(4096));
                for _ in 0..n {
                    let id = r.u32("trophies.bin trophy id")?;
                    let tick = r.u64("trophies.bin tick")?;
                    list.push((id, tick));
                }
                out.trophies.push((comm_id, list));
            }
        }

        Ok((out, refused))
    }
}

/// A ZIP entry name cannot usefully carry the `:` of a Vita mount - Windows refuses to
/// extract one, and an archive a user cannot unpack defeats the point of shipping a zip.
/// Escaped the same way `web/opfs.js` escapes a separator, and reversed by [`unescape`].
/// `%` goes first so the escape is injective.
fn escape(key: &str) -> String {
    key.replace('%', "%25").replace(':', "%3A")
}

fn unescape(name: &str) -> String {
    name.replace("%3A", ":").replace("%25", "%")
}

fn put_str(out: &mut Vec<u8>, s: &str) {
    out.extend_from_slice(&(s.len() as u32).to_le_bytes());
    out.extend_from_slice(s.as_bytes());
}

fn put_bytes(out: &mut Vec<u8>, b: &[u8]) {
    out.extend_from_slice(&(b.len() as u32).to_le_bytes());
    out.extend_from_slice(b);
}

/// A bounds-checked read cursor over a sidecar. Every field names itself, so a truncated
/// container says which record it died in rather than producing a shorter save.
struct Cursor<'a> {
    b: &'a [u8],
    at: usize,
}

impl<'a> Cursor<'a> {
    fn new(b: &'a [u8]) -> Self {
        Cursor { b, at: 0 }
    }
    fn done(&self) -> bool {
        self.at >= self.b.len()
    }
    fn take(&mut self, n: usize, what: &str) -> Result<&'a [u8], String> {
        let end = self.at.checked_add(n).ok_or_else(|| format!("{what}: length overflows"))?;
        let out = self.b.get(self.at..end).ok_or_else(|| {
            format!("{what}: wants {n} bytes at {} of {}, so this container is truncated", self.at, self.b.len())
        })?;
        self.at = end;
        Ok(out)
    }
    fn u8(&mut self, what: &str) -> Result<u8, String> {
        Ok(self.take(1, what)?[0])
    }
    fn u32(&mut self, what: &str) -> Result<u32, String> {
        let b = self.take(4, what)?;
        Ok(u32::from_le_bytes([b[0], b[1], b[2], b[3]]))
    }
    fn u64(&mut self, what: &str) -> Result<u64, String> {
        let b = self.take(8, what)?;
        let mut v = [0u8; 8];
        v.copy_from_slice(b);
        Ok(u64::from_le_bytes(v))
    }
    fn fixed16(&mut self, what: &str) -> Result<[u8; 16], String> {
        let b = self.take(16, what)?;
        let mut v = [0u8; 16];
        v.copy_from_slice(b);
        Ok(v)
    }
    fn bytes(&mut self, what: &str) -> Result<Vec<u8>, String> {
        let n = self.u32(what)? as usize;
        Ok(self.take(n, what)?.to_vec())
    }
    fn str(&mut self, what: &str) -> Result<String, String> {
        let b = self.bytes(what)?;
        String::from_utf8(b).map_err(|_| format!("{what}: not UTF-8"))
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    fn sample() -> GameData {
        GameData {
            files: vec![
                ("savedata0:/data.bin".into(), b"score=1234".to_vec()),
                ("ux0:/data/title/prefs".into(), vec![0u8, 255, 7, 7, 7]),
            ],
            dirs: vec!["savedata0:/empty".into()],
            originals: vec![("savedata0:/data.bin".into(), "DATA.BIN".into())],
            stats: vec![(
                "savedata0:/data.bin".into(),
                FileStatOverride {
                    mode: Some(0x21b6),
                    attr: None,
                    times: [Some([1u8; 16]), None, Some([3u8; 16])],
                },
            )],
            slots: vec![("savedata0:".into(), 3, vec![9, 8, 7])],
            trophies: vec![("NPWR00001_00".into(), vec![(1, 0x1122_3344_5566_7788), (4, 5)])],
        }
    }

    #[test]
    fn a_container_round_trips_every_kind_of_state() {
        let data = sample();
        let zip = data.to_zip("PCSA00000");
        let (back, refused) = GameData::from_zip(&zip).expect("read back");
        assert!(refused.is_empty(), "nothing legitimate should be refused: {refused:?}");
        assert_eq!(back, data);
    }

    #[test]
    fn the_same_state_exports_the_same_bytes_twice() {
        // Two exports of an unchanged run must be byte-identical, or a "has it changed?"
        // check over the container is meaningless and every flush rewrites storage.
        let data = sample();
        assert_eq!(data.to_zip("PCSA00000"), data.to_zip("PCSA00000"));
    }

    #[test]
    fn an_empty_container_round_trips_as_empty() {
        let (back, refused) = GameData::from_zip(&GameData::default().to_zip("X")).expect("read");
        assert!(refused.is_empty());
        assert!(back.is_empty());
    }

    #[test]
    fn a_container_cannot_name_a_path_outside_the_guests_saved_state() {
        // Exactly the attack the guard exists for: an archive whose entries claim to be
        // the installed title, an absolute escape, or a traversal out of the save mount.
        // Every one of them must be REFUSED and NAMED, never imported.
        let hostile = crate::ingest::zip::write_zip(&[
            ("files/eboot.bin".into(), b"pwned".to_vec()),
            ("files/app0%3A/data/x.gxt".into(), b"pwned".to_vec()),
            ("files/../../games/x".into(), b"pwned".to_vec()),
            ("files/savedata0%3A/ok.bin".into(), b"fine".to_vec()),
            ("dirs.txt".into(), b"eboot.bin\nsavedata0:/d".to_vec()),
        ]);
        let (data, refused) = GameData::from_zip(&hostile).expect("read");
        assert_eq!(data.files, vec![("savedata0:/ok.bin".to_string(), b"fine".to_vec())]);
        assert_eq!(data.dirs, vec!["savedata0:/d".to_string()]);
        assert_eq!(refused.len(), 4, "refused: {refused:?}");
    }

    #[test]
    fn a_truncated_sidecar_is_an_error_not_a_shorter_save() {
        // A restore that silently drops a title's trophies and reports success is worse
        // than one that fails: the user plays on and overwrites the good container.
        let mut bin = Vec::new();
        put_str(&mut bin, "NPWR00001_00");
        bin.extend_from_slice(&9u32.to_le_bytes()); // claims 9 trophies
        bin.extend_from_slice(&1u32.to_le_bytes()); // and carries part of one
        let zip = crate::ingest::zip::write_zip(&[("trophies.bin".into(), bin)]);
        let err = GameData::from_zip(&zip).expect_err("must not silently truncate");
        assert!(err.contains("truncated"), "{err}");
    }

    #[test]
    fn an_unknown_entry_is_ignored_so_an_older_build_can_read_a_newer_container() {
        let mut zip_entries = vec![("files/savedata0%3A/a".to_string(), b"a".to_vec())];
        zip_entries.push(("something-from-the-future.bin".into(), vec![1, 2, 3]));
        let (data, refused) = GameData::from_zip(&crate::ingest::zip::write_zip(&zip_entries))
            .expect("read");
        assert!(refused.is_empty());
        assert_eq!(data.files.len(), 1);
    }

    #[test]
    fn a_key_with_a_percent_in_it_survives_the_escaping() {
        let data = GameData {
            files: vec![("savedata0:/100% clear.bin".into(), b"x".to_vec())],
            ..Default::default()
        };
        let (back, _) = GameData::from_zip(&data.to_zip("X")).expect("read");
        assert_eq!(back.files, data.files);
    }

    #[test]
    fn the_container_is_a_zip_other_tools_can_open() {
        // The signature and the end-of-central-directory record, checked here rather than
        // trusted: a game-data export is a file the USER opens, so "our own reader accepts
        // it" is not the property that matters.
        let zip = sample().to_zip("PCSA00000");
        assert_eq!(&zip[0..4], b"PK\x03\x04");
        assert!(zip.windows(4).any(|w| w == b"PK\x05\x06"));
    }
}
