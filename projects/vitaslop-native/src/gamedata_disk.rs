//! The DISK half of the guest's own saved state - what the browser does with OPFS, done
//! with files, so a native run keeps a title's progress across invocations.
//!
//! # There is one definition of what a save IS, and it is not here
//! `vitaslop_runtime::gamedata` decides what may be in a container and
//! `vitaslop_runtime::host::is_persisted_mount` decides what is collected into one. This
//! module moves opaque bytes between that container and a file. It cannot widen the export
//! set, and an imported container is checked by the same parser the browser uses, so the two
//! backends cannot drift into disagreeing about what a save is.
//!
//! # Why a directory per title under a root, rather than one named file
//! The browser stores `gamedata/<titleId>/gamedata.zip`. This mirrors it:
//! `<root>/<title>/gamedata.zip`. Pointing two different titles at one `--save-dir` is then
//! not a way to overwrite one save with another, which a single `--save-file` path would
//! make easy to do by accident and impossible to notice - the container that lands is
//! complete and valid, just the wrong game's.
//!
//! # The write is to one side and then a rename
//! Same reason as the browser's `.part` + `move`: a process killed during the write must not
//! be able to leave a half-written container where a whole one was. `rename` over an
//! existing file is atomic enough for that on both platforms this builds for.

use std::io;
use std::path::{Path, PathBuf};
use std::time::{Duration, Instant};

use vitaslop_runtime::gamedata::GameData;
use vitaslop_runtime::host::VitaState;

/// The container's file name inside a title's save directory. The same name the browser
/// stores under, so a file copied from one to the other is the file the other expects.
const BLOB: &str = "gamedata.zip";
/// Written first, renamed over [`BLOB`]. See the module note.
const PART: &str = "gamedata.zip.part";

/// How long a run waits before writing again after a write. The guest can mark the save
/// dirty on consecutive frames (a title saving a settings blob per menu keypress does), and
/// a container is rewritten whole; without a floor that is a file write per frame.
const FLOOR: Duration = Duration::from_secs(3);

/// One title's save file, and the clock that bounds how often it is rewritten.
pub struct SaveStore {
    dir: PathBuf,
    title: String,
    last_write: Option<Instant>,
    /// Bytes of the container last written, so an unchanged save is not rewritten. The
    /// dirty flag answers "did the guest touch its save", which is not the same question as
    /// "is the result different" - a title that rewrites the same blob raises the flag
    /// every time.
    last_bytes: Option<Vec<u8>>,
    /// Whether this run has written the container at all. Only for REPORTING, and it exists
    /// because the obvious report is wrong: the last flush of a run that saved normally
    /// returns "nothing to do" (the save was already written seconds ago and is not dirty),
    /// which reads as "the game saved nothing this run" - the exact opposite of the truth,
    /// on the line a user checks to find out whether their progress was kept.
    wrote: bool,
}

impl SaveStore {
    /// The store for `title` under `root`. Creates nothing yet: a run that never saves
    /// leaves no directory behind.
    pub fn new(root: &Path, title: &str) -> Self {
        SaveStore {
            dir: root.join(title),
            title: title.to_string(),
            last_write: None,
            last_bytes: None,
            wrote: false,
        }
    }

    /// The name this title's saves are kept under, asked of the TITLE and not of the path.
    ///
    /// >>> THE DIRECTORY NAME IS NOT AN IDENTITY, and the common layout makes that acute:
    /// every title extracted as `<name>/extracted/` has the same last component, so keying
    /// on the path would file every game's save in one directory called `extracted` and let
    /// each overwrite the last. The container carries its own title id; that is the answer,
    /// and it is the same one the browser's launcher keys its OPFS tree on.
    ///
    /// `sfo` is the bytes of the title's `sce_sys/param.sfo`, read out of the guest
    /// filesystem after boot. Falling back to the path when it is missing or unreadable is
    /// deliberate - a save under an imperfect name beats no save - but the caller is
    /// expected to SAY which it used, because the two are not interchangeable.
    pub fn title_for(game_dir: &str, sfo: Option<&[u8]>) -> (String, bool) {
        match sfo.and_then(vitaslop_runtime::ingest::sfo::title_id) {
            Some(id) => (id, true),
            None => (Self::title_from_game_dir(game_dir), false),
        }
    }

    /// A last-resort title name for a game directory: its last path component. Prefer
    /// [`title_for`](SaveStore::title_for), which asks the container.
    pub fn title_from_game_dir(game_dir: &str) -> String {
        Path::new(game_dir)
            .file_name()
            .map(|s| s.to_string_lossy().into_owned())
            .filter(|s| !s.is_empty())
            .unwrap_or_else(|| "unknown".to_string())
    }

    /// Where this title's container lives.
    pub fn path(&self) -> PathBuf {
        self.dir.join(BLOB)
    }

    /// Put a previously written container back into `st`. CALL BEFORE THE GUEST RUNS -
    /// `restore_game_data` replaces files the guest may already hold descriptors on.
    ///
    /// Returns a one-line report, or `None` when this title has never saved here. A
    /// container that exists but does not parse is an ERROR, never a silent fresh start:
    /// the alternative is a user whose save is quietly ignored and who plays on top of
    /// nothing, and the file is still there to be looked at.
    pub fn restore(&mut self, st: &mut VitaState) -> io::Result<Option<String>> {
        let path = self.path();
        let bytes = match std::fs::read(&path) {
            Ok(b) => b,
            Err(e) if e.kind() == io::ErrorKind::NotFound => return Ok(None),
            Err(e) => return Err(e),
        };
        if bytes.is_empty() {
            return Ok(None);
        }
        let (data, refused) = GameData::from_zip(&bytes).map_err(|e| {
            io::Error::new(
                io::ErrorKind::InvalidData,
                format!("{}: not a vitaslop game-data container: {e}", path.display()),
            )
        })?;
        let report = st.restore_game_data(&data);
        // What the parser threw away is named, not counted - a refused entry means the
        // file carries something that is not this title's save.
        let extra = if refused.is_empty() {
            String::new()
        } else {
            format!(" (refused {}: {})", refused.len(), refused.join(", "))
        };
        // The bytes that are now on disk ARE what the state holds, so the first flush of the
        // run has something to compare against and will not rewrite an unchanged file.
        self.last_bytes = Some(st.game_data().to_zip(&self.title));
        Ok(Some(format!("{} bytes, {report}{extra}", bytes.len())))
    }

    /// Whether this run has written the container. See [`SaveStore::wrote`].
    pub fn has_written(&self) -> bool {
        self.wrote
    }

    /// Write the guest's current saved state out, if it has changed.
    ///
    /// `force` skips the rate floor (for the end of a run); it does NOT skip the dirty
    /// check, so a run that saved nothing still writes nothing.
    ///
    /// Returns the size of the container written, or `None` when there was nothing to do.
    pub fn flush(&mut self, st: &mut VitaState, force: bool) -> io::Result<Option<usize>> {
        if !st.game_data_dirty() {
            return Ok(None);
        }
        if !force {
            if let Some(t) = self.last_write {
                if t.elapsed() < FLOOR {
                    return Ok(None);
                }
            }
        }
        // NOT gated on `is_empty`. A guest that deleted its only save collects to nothing,
        // and skipping that would make the one change that matters the one never written -
        // the old container would stay on disk and the deletion would come back undone.
        let bytes = st.game_data().to_zip(&self.title);
        if self.last_bytes.as_deref() == Some(bytes.as_slice()) {
            // Identical to what is already on disk. The flag is still cleared: the change
            // it recorded has been accounted for, and leaving it set would re-serialise the
            // whole container on every subsequent frame for ever.
            st.clear_game_data_dirty();
            self.last_write = Some(Instant::now());
            return Ok(None);
        }
        std::fs::create_dir_all(&self.dir)?;
        let part = self.dir.join(PART);
        std::fs::write(&part, &bytes)?;
        std::fs::rename(&part, self.path())?;
        // ONLY after the bytes are in place. Clearing the flag first and then failing to
        // write takes the record of the change with it, and the next frame sees a clean
        // save that was never stored.
        st.clear_game_data_dirty();
        self.last_write = Some(Instant::now());
        let n = bytes.len();
        self.last_bytes = Some(bytes);
        self.wrote = true;
        Ok(Some(n))
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use vitaslop_runtime::DeterministicWorld;

    fn new_state() -> VitaState {
        VitaState::new(0x1000, 0x10000, Box::new(DeterministicWorld::default()))
    }

    /// The round trip this whole module exists for, over a real directory: a state that
    /// saved is written, a fresh state reads it back, and the second run does not rewrite
    /// an unchanged file.
    #[test]
    fn a_save_written_to_disk_comes_back_in_the_next_run() {
        let root = std::env::temp_dir().join(format!(
            "vitaslop-savestore-{}-{:?}",
            std::process::id(),
            std::thread::current().id()
        ));
        let _ = std::fs::remove_dir_all(&root);

        let mut first = new_state();
        first.clear_game_data_dirty();
        let mut store = SaveStore::new(&root, "PCSA00000");
        assert_eq!(store.flush(&mut first, true).unwrap(), None, "a run that saved nothing writes nothing");
        assert!(!store.path().exists(), "and leaves no directory behind");

        first.add_file("savedata0:/-AUTO-/DATA.BIN", b"progress".to_vec());
        let n = store.flush(&mut first, true).unwrap().expect("a save is written");
        assert!(n > 0);
        assert!(store.path().exists());
        assert!(!first.game_data_dirty(), "the flag is cleared by the write");
        assert_eq!(store.flush(&mut first, true).unwrap(), None, "and not rewritten unchanged");

        let mut second = new_state();
        let mut store2 = SaveStore::new(&root, "PCSA00000");
        let report = store2.restore(&mut second).unwrap().expect("the container is found");
        assert!(report.contains("1 file(s)"), "report was: {report}");
        assert_eq!(
            second.read_file("savedata0:/-AUTO-/DATA.BIN").as_deref(),
            Some(&b"progress"[..]),
        );
        assert!(!second.game_data_dirty(), "a restore is not itself a change to save");
        assert_eq!(store2.flush(&mut second, true).unwrap(), None);
        assert!(store.has_written(), "the run that saved knows it saved");
        assert!(!store2.has_written(), "the run that only restored did not write");

        // A container that is not one is an error, not a silent fresh start.
        std::fs::write(store2.path(), b"not a zip").unwrap();
        let mut third = new_state();
        assert!(SaveStore::new(&root, "PCSA00000").restore(&mut third).is_err());

        let _ = std::fs::remove_dir_all(&root);
    }

    #[test]
    fn the_container_names_the_title_and_the_path_is_only_a_fallback() {
        // The layout that makes this necessary: two different titles, one directory name.
        assert_eq!(SaveStore::title_from_game_dir("C:/games/hotshots/extracted"), "extracted");
        assert_eq!(SaveStore::title_from_game_dir("C:/games/ridgeracer/extracted"), "extracted");

        // A one-entry `param.sfo`, built here rather than exported from the parser's own
        // tests: a builder nothing ships has no business being public API.
        let (key, val) = (b"TITLE_ID ", b"PCSA00009 ");
        let mut sfo = Vec::new();
        sfo.extend_from_slice(b" PSF");
        sfo.extend_from_slice(&0x0101_0000u32.to_le_bytes()); // version
        sfo.extend_from_slice(&36u32.to_le_bytes()); // key table: after header + 1 entry
        sfo.extend_from_slice(&(36u32 + key.len() as u32).to_le_bytes()); // data table
        sfo.extend_from_slice(&1u32.to_le_bytes()); // entry count
        sfo.extend_from_slice(&0u16.to_le_bytes()); // key offset
        sfo.extend_from_slice(&0x0204u16.to_le_bytes()); // format: NUL-terminated text
        sfo.extend_from_slice(&(val.len() as u32).to_le_bytes()); // used length
        sfo.extend_from_slice(&(val.len() as u32).to_le_bytes()); // reserved length
        sfo.extend_from_slice(&0u32.to_le_bytes()); // data offset
        sfo.extend_from_slice(key);
        sfo.extend_from_slice(val);
        assert_eq!(
            SaveStore::title_for("C:/games/hotshots/extracted", Some(&sfo)),
            ("PCSA00009".to_string(), true),
        );
        // No container, or one that carries no usable id: the path, and the caller is told.
        assert_eq!(
            SaveStore::title_for("C:/games/hotshots/extracted", None),
            ("extracted".to_string(), false),
        );
        assert_eq!(
            SaveStore::title_for("/games/hotshots/", Some(b"not a container")),
            ("hotshots".to_string(), false),
        );
    }
}
