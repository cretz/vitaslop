//! The desktop's data directory: imported titles, their records and images, saves,
//! and settings. One layout, spelled here and nowhere else:
//!
//! ```text
//! <home>/library/<TITLE_ID>/          the decrypted dump tree (what `--game` takes)
//! <home>/library/<TITLE_ID>/meta.json the TitleMeta record; icon0.png, pic0.png beside it
//! <home>/saves/<profile>/<TITLE_ID>/  the guest's own saved state (SaveStore)
//! <home>/settings.json                the global settings record
//! <home>/titles/<TITLE_ID>.json       a title's settings patch
//! <home>/logs/vitaslop-<stamp>.log    the shell's run logs, the newest few kept
//! ```
//!
//! `<home>` is `VITASLOP_HOME`, else the platform's per-user data directory. The
//! import is the same streaming ingest the browser uses, over `std::fs`.

use std::fs;
use std::path::{Path, PathBuf};
use std::rc::Rc;
use std::sync::{Arc, Mutex};

use vitaslop_frontend::meta::TitleMeta;
use vitaslop_frontend::settings::{self, Settings};
use vitaslop_runtime::ingest::stream::{self, ByteSource, DumpSink};
use vitaslop_runtime::ingest::Error;

pub fn home() -> PathBuf {
    if let Some(h) = std::env::var_os("VITASLOP_HOME") {
        return PathBuf::from(h);
    }
    let base = if cfg!(target_os = "windows") {
        std::env::var_os("LOCALAPPDATA").map(PathBuf::from)
    } else if cfg!(target_os = "macos") {
        std::env::var_os("HOME").map(|h| PathBuf::from(h).join("Library/Application Support"))
    } else {
        std::env::var_os("XDG_DATA_HOME")
            .map(PathBuf::from)
            .or_else(|| std::env::var_os("HOME").map(|h| PathBuf::from(h).join(".local/share")))
    };
    base.unwrap_or_else(|| PathBuf::from(".")).join("vitaslop")
}

pub fn library_dir() -> PathBuf {
    home().join("library")
}
pub fn title_dir(id: &str) -> PathBuf {
    library_dir().join(id)
}
pub fn saves_dir(profile: &str) -> PathBuf {
    home().join("saves").join(if profile.is_empty() { "default" } else { profile })
}
pub fn logs_dir() -> PathBuf {
    home().join("logs")
}

// ------------------------------- saved data -------------------------------
//
// The browser's game-data panel over the desktop's layout: one `gamedata.zip` per title per
// profile, the same container the browser keeps in OPFS (`vitaslop_native::SaveStore`), so a
// download from either side restores on the other.

/// `<home>/saves/<profile>/<TITLE_ID>/gamedata.zip`.
pub fn gamedata_path(profile: &str, id: &str) -> PathBuf {
    saves_dir(profile).join(id).join("gamedata.zip")
}

/// The saved data's size and last write (Unix ms), `None` when the title has saved nothing.
pub fn gamedata_info(profile: &str, id: &str) -> Option<(u64, u64)> {
    let m = fs::metadata(gamedata_path(profile, id)).ok()?;
    let modified = m.modified().ok().and_then(|t| t.duration_since(std::time::UNIX_EPOCH).ok()).map_or(0, |d| d.as_millis() as u64);
    Some((m.len(), modified))
}

/// What a game-data container holds, from the parser a run restores it with - so what a
/// person is told before an upload is what the next run will actually put back.
pub fn describe_gamedata(zip: &[u8]) -> Result<String, String> {
    let (data, refused) = vitaslop_runtime::gamedata::GameData::from_zip(zip)?;
    let mut out = data.summary();
    if !refused.is_empty() {
        out.push_str(&format!(
            " - and {} entr(y/ies) that name something outside the guest's saved state, which will be REFUSED: {}",
            refused.len(),
            refused.join(", ")
        ));
    }
    Ok(out)
}

/// Replace a title's saved data with `zip`, after checking it parses.
pub fn write_gamedata(profile: &str, id: &str, zip: &[u8]) -> Result<(), String> {
    describe_gamedata(zip)?;
    let p = gamedata_path(profile, id);
    if let Some(d) = p.parent() {
        fs::create_dir_all(d).map_err(|e| e.to_string())?;
    }
    fs::write(&p, zip).map_err(|e| format!("{}: {e}", p.display()))
}

pub fn clear_gamedata(profile: &str, id: &str) -> std::io::Result<()> {
    match fs::remove_file(gamedata_path(profile, id)) {
        Err(e) if e.kind() != std::io::ErrorKind::NotFound => Err(e),
        _ => Ok(()),
    }
}

/// The titles with saved data in `profile`, sorted.
pub fn titles_with_gamedata(profile: &str) -> Vec<String> {
    let mut out: Vec<String> = fs::read_dir(saves_dir(profile))
        .into_iter()
        .flatten()
        .flatten()
        .filter(|e| e.path().join("gamedata.zip").is_file())
        .map(|e| e.file_name().to_string_lossy().into_owned())
        .collect();
    out.sort();
    out
}

/// Every profile that has a folder, plus `default`, sorted - the settings' profile list.
pub fn profiles() -> Vec<String> {
    let mut out: Vec<String> = fs::read_dir(home().join("saves"))
        .into_iter()
        .flatten()
        .flatten()
        .filter(|e| e.path().is_dir())
        .map(|e| e.file_name().to_string_lossy().into_owned())
        .collect();
    if !out.iter().any(|p| p == "default") {
        out.push("default".into());
    }
    out.sort();
    out
}

/// A profile name the browser would accept: letters, digits, `-` and `_`, 1 to 32 of them.
/// Title ids are the same shape, which is what a bundle's entry names are checked against.
pub fn valid_profile(name: &str) -> bool {
    (1..=32).contains(&name.len()) && name.chars().all(|c| c.is_ascii_alphanumeric() || c == '-' || c == '_')
}

/// Every title's saved data in `profile` as one zip of `<TITLE_ID>.zip` entries - the
/// browser's "Download all" bundle, byte for byte the same shape.
pub fn gamedata_bundle(profile: &str) -> Vec<u8> {
    let entries: Vec<(String, Vec<u8>)> = titles_with_gamedata(profile)
        .into_iter()
        .filter_map(|id| fs::read(gamedata_path(profile, &id)).ok().map(|b| (format!("{id}.zip"), b)))
        .collect();
    vitaslop_runtime::ingest::zip::write_zip(&entries)
}

/// The `<TITLE_ID>.zip` entries of a bundle, each checked to parse.
pub fn read_gamedata_bundle(bytes: &[u8]) -> Result<Vec<(String, Vec<u8>)>, String> {
    use vitaslop_runtime::ingest::vfs::Vfs;
    let vfs = vitaslop_runtime::ingest::zip::read_zip(bytes).map_err(|e| e.to_string())?;
    let mut out = Vec::new();
    for name in vfs.list() {
        let Some(id) = name.strip_suffix(".zip").filter(|id| valid_profile(id)) else { continue };
        let zip = vfs.read(&name).map_err(|e| e.to_string())?;
        describe_gamedata(&zip).map_err(|e| format!("{name}: {e}"))?;
        out.push((id.to_string(), zip));
    }
    if out.is_empty() {
        return Err("no <TITLE_ID>.zip entries in this bundle".into());
    }
    Ok(out)
}

/// Bytes under `dir`, recursively (0 when it does not exist).
pub fn dir_bytes(dir: &Path) -> u64 {
    let Ok(rd) = fs::read_dir(dir) else { return 0 };
    rd.flatten()
        .map(|e| match e.metadata() {
            Ok(m) if m.is_dir() => dir_bytes(&e.path()),
            Ok(m) => m.len(),
            Err(_) => 0,
        })
        .sum()
}

// ------------------------------- settings -------------------------------

fn read_json(p: &Path) -> serde_json::Value {
    fs::read_to_string(p).ok().and_then(|s| serde_json::from_str(&s).ok()).unwrap_or(serde_json::json!({}))
}

pub fn global_settings_value() -> serde_json::Value {
    read_json(&home().join("settings.json"))
}

pub fn save_global_settings(s: &Settings) -> std::io::Result<()> {
    fs::create_dir_all(home())?;
    fs::write(home().join("settings.json"), serde_json::to_string_pretty(&s.to_value())?)
}

/// Whether sound is muted - the desktop's form of the web player's mute button, which keeps
/// its choice in `localStorage` beside (not inside) the shared settings; here it is a file
/// beside `settings.json`, so the settings schema stays the browser's.
pub fn muted() -> bool {
    read_json(&home().join("audio.json")).get("muted").and_then(|v| v.as_bool()).unwrap_or(false)
}

pub fn save_muted(muted: bool) -> std::io::Result<()> {
    fs::create_dir_all(home())?;
    fs::write(home().join("audio.json"), serde_json::json!({ "muted": muted }).to_string())
}

/// Where the shell's window was when it last closed, so the next launch opens there. The
/// position and size are PHYSICAL pixels of the window in its normal state (not maximized,
/// fullscreen or minimized - those are kept as flags, and leaving one returns to this rect).
#[derive(Clone, Copy, Debug, PartialEq)]
pub struct WindowPlace {
    pub x: i32,
    pub y: i32,
    pub width: u32,
    pub height: u32,
    pub maximized: bool,
    pub fullscreen: bool,
}

pub fn window_place() -> Option<WindowPlace> {
    let v = read_json(&home().join("window.json"));
    let int = |k: &str| v.get(k).and_then(|n| n.as_i64());
    let flag = |k: &str| v.get(k).and_then(|b| b.as_bool()).unwrap_or(false);
    let (width, height) = (u32::try_from(int("width")?).ok()?, u32::try_from(int("height")?).ok()?);
    if width == 0 || height == 0 {
        return None;
    }
    Some(WindowPlace {
        x: i32::try_from(int("x")?).ok()?,
        y: i32::try_from(int("y")?).ok()?,
        width,
        height,
        maximized: flag("maximized"),
        fullscreen: flag("fullscreen"),
    })
}

pub fn save_window_place(p: &WindowPlace) -> std::io::Result<()> {
    fs::create_dir_all(home())?;
    let v = serde_json::json!({
        "x": p.x, "y": p.y, "width": p.width, "height": p.height,
        "maximized": p.maximized, "fullscreen": p.fullscreen,
    });
    fs::write(home().join("window.json"), v.to_string())
}

pub fn title_patch(id: &str) -> Option<serde_json::Value> {
    let p = home().join("titles").join(format!("{id}.json"));
    p.exists().then(|| read_json(&p))
}

pub fn save_title_patch(id: &str, patch: Option<&serde_json::Value>) -> std::io::Result<()> {
    let dir = home().join("titles");
    let p = dir.join(format!("{id}.json"));
    match patch {
        Some(v) if v.as_object().map(|o| !o.is_empty()).unwrap_or(false) => {
            fs::create_dir_all(&dir)?;
            fs::write(p, serde_json::to_string_pretty(v)?)
        }
        _ => {
            let _ = fs::remove_file(p);
            Ok(())
        }
    }
}

/// The settings a run of `id` uses (or the global ones when `id` is `None`).
pub fn effective(id: Option<&str>) -> Settings {
    let patch = id.and_then(title_patch);
    settings::effective(&global_settings_value(), patch.as_ref())
}

// ------------------------------- the library -------------------------------

pub fn list_titles() -> Vec<TitleMeta> {
    let mut out = Vec::new();
    if let Ok(rd) = fs::read_dir(library_dir()) {
        for e in rd.flatten() {
            if let Some(m) = read_meta(&e.path()) {
                out.push(m);
            }
        }
    }
    out.sort_by_key(|t| std::cmp::Reverse(t.imported_at));
    out
}

pub fn read_meta(dir: &Path) -> Option<TitleMeta> {
    let s = fs::read_to_string(dir.join("meta.json")).ok()?;
    let m: TitleMeta = serde_json::from_str(&s).ok()?;
    (dir.file_name().map(|n| n.to_string_lossy() == m.title_id).unwrap_or(false)).then_some(m)
}

pub fn write_meta(m: &TitleMeta) -> std::io::Result<()> {
    let dir = title_dir(&m.title_id);
    fs::create_dir_all(&dir)?;
    fs::write(dir.join("meta.json"), serde_json::to_string_pretty(m)?)
}

pub fn remove_title(id: &str) -> std::io::Result<()> {
    fs::remove_dir_all(title_dir(id))
}

pub fn now_ms() -> u64 {
    std::time::SystemTime::now().duration_since(std::time::UNIX_EPOCH).map(|d| d.as_millis() as u64).unwrap_or(0)
}

// ------------------------------- the import -------------------------------

/// A directory (or a set of files) as a ByteSource, read with `std::fs`.
pub struct DirSource {
    files: Vec<(String, PathBuf)>,
}

impl DirSource {
    /// Every file under `path` (a directory), or the file itself plus any `work.bin`
    /// beside it (a picked `.pkg`, `.zip` or `.vpk`).
    /// What the import screen's package tab picks: the `.pkg` and its licence, which goes in
    /// as `work.bin` whatever the file was called - the browser's `{ path: "work.bin" }`.
    pub fn pkg_and_licence(pkg: &Path, work: &Path) -> DirSource {
        let name = pkg.file_name().map(|n| n.to_string_lossy().into_owned()).unwrap_or_else(|| "game.pkg".into());
        DirSource { files: vec![(name, pkg.to_path_buf()), ("work.bin".to_string(), work.to_path_buf())] }
    }

    /// Several dropped files and folders at once (a `.pkg` dropped together with its
    /// `work.bin`, say): files by name, folders walked under their own name, as the
    /// browser's drop does. One path is [`DirSource::open`].
    pub fn open_many(paths: &[PathBuf]) -> std::io::Result<DirSource> {
        if let [one] = paths {
            return DirSource::open(one);
        }
        let mut files = Vec::new();
        for p in paths {
            let name = p.file_name().map(|n| n.to_string_lossy().into_owned()).unwrap_or_default();
            if p.is_dir() {
                let mut inner = Vec::new();
                walk(p, p, &mut inner)?;
                files.extend(inner.into_iter().map(|(rel, path)| (format!("{name}/{rel}"), path)));
            } else {
                files.push((name, p.clone()));
            }
        }
        Ok(DirSource { files })
    }

    pub fn open(path: &Path) -> std::io::Result<DirSource> {
        let mut files = Vec::new();
        if path.is_dir() {
            walk(path, path, &mut files)?;
        } else {
            let name = path.file_name().map(|n| n.to_string_lossy().into_owned()).unwrap_or_default();
            files.push((name, path.to_path_buf()));
            if let Some(dir) = path.parent() {
                let w = dir.join("work.bin");
                if w.exists() && !path.ends_with("work.bin") {
                    files.push(("work.bin".to_string(), w));
                }
            }
        }
        Ok(DirSource { files })
    }
}

fn walk(root: &Path, dir: &Path, out: &mut Vec<(String, PathBuf)>) -> std::io::Result<()> {
    for e in fs::read_dir(dir)? {
        let p = e?.path();
        if p.is_dir() {
            // A folder that was already played carries a compile cache; it is not the game's.
            if dir == root && p.file_name().is_some_and(|n| n == vitaslop_native::compile_cache::DIR) {
                continue;
            }
            walk(root, &p, out)?;
        } else {
            let rel = p.strip_prefix(root).unwrap().to_string_lossy().replace('\\', "/");
            out.push((rel, p));
        }
    }
    Ok(())
}

impl ByteSource for DirSource {
    fn list(&self) -> Vec<String> {
        self.files.iter().map(|(n, _)| n.clone()).collect()
    }
    fn size(&self, path: &str) -> Option<u64> {
        let p = &self.files.iter().find(|(n, _)| n == path)?.1;
        fs::metadata(p).ok().map(|m| m.len())
    }
    fn read_at(&self, path: &str, off: u64, buf: &mut [u8]) -> Result<usize, Error> {
        use std::io::{Read, Seek, SeekFrom};
        let p = &self.files.iter().find(|(n, _)| n == path).ok_or_else(|| Error::MissingFile(path.to_string()))?.1;
        let mut f = fs::File::open(p).map_err(|e| Error::Io(e.to_string()))?;
        f.seek(SeekFrom::Start(off)).map_err(|e| Error::Io(e.to_string()))?;
        let mut got = 0;
        while got < buf.len() {
            let n = f.read(&mut buf[got..]).map_err(|e| Error::Io(e.to_string()))?;
            if n == 0 {
                break;
            }
            got += n;
        }
        Ok(got)
    }
}

/// A dump tree on disk as a DumpSink.
pub struct DirSink {
    root: PathBuf,
    cur: Option<std::io::BufWriter<fs::File>>,
}

impl DirSink {
    pub fn new(root: PathBuf) -> DirSink {
        DirSink { root, cur: None }
    }
}

impl DumpSink for DirSink {
    fn begin(&mut self, path: &str, _size: u64) -> Result<(), Error> {
        let p = self.root.join(path);
        if let Some(d) = p.parent() {
            fs::create_dir_all(d).map_err(|e| Error::Io(e.to_string()))?;
        }
        let f = fs::File::create(&p).map_err(|e| Error::Io(format!("{}: {e}", p.display())))?;
        self.cur = Some(std::io::BufWriter::new(f));
        Ok(())
    }
    fn write(&mut self, bytes: &[u8]) -> Result<(), Error> {
        use std::io::Write;
        self.cur.as_mut().ok_or_else(|| Error::Io("write before begin".into()))?.write_all(bytes).map_err(|e| Error::Io(e.to_string()))
    }
    fn finish(&mut self) -> Result<(), Error> {
        use std::io::Write;
        let mut w = self.cur.take().ok_or_else(|| Error::Io("finish before begin".into()))?;
        w.flush().map_err(|e| Error::Io(e.to_string()))
    }
}

/// Live progress of an import running on another thread.
#[derive(Default, Clone)]
pub struct ImportProgress {
    pub stage: String,
    pub file: String,
    pub done: u64,
    pub total: u64,
    pub finished: bool,
    pub error: Option<String>,
    pub title_id: Option<String>,
    /// What the files turned out to be, once the probe has read them - the browser's
    /// `onProbe`, which fills in the progress card part way through the import.
    pub probe: Option<ProbeCard>,
    /// Files and bytes picked, before anything was read.
    pub picked: (usize, u64),
}

/// The import's own identification of what it was given, as the progress card shows it.
#[derive(Default, Clone)]
pub struct ProbeCard {
    pub title: String,
    pub title_id: String,
    pub app_version: String,
    /// `pkg`, `pfs`, `dump` or `vpk`, and whether it came wrapped in a zip.
    pub kind: String,
    pub zipped: bool,
    pub files: usize,
    pub bytes: u64,
    pub icon0: Option<Vec<u8>>,
    /// The library already holds this title: the import replaces it (saved data is kept).
    pub replacing: bool,
}

/// Import `path` into the library on this thread, reporting into `progress`.
pub fn import(path: &Path, progress: &Arc<Mutex<ImportProgress>>) -> Result<TitleMeta, String> {
    import_from(DirSource::open(path).map_err(|e| e.to_string())?, progress)
}

/// Import whatever `src` holds - see [`import`].
pub fn import_from(src: DirSource, progress: &Arc<Mutex<ImportProgress>>) -> Result<TitleMeta, String> {
    progress.lock().unwrap().picked = (src.files.len(), src.files.iter().filter_map(|(_, p)| fs::metadata(p).ok()).map(|m| m.len()).sum());
    let src: Rc<dyn ByteSource> = Rc::new(src);
    progress.lock().unwrap().stage = "reading".into();
    let p = stream::probe(src.clone()).map_err(|e| e.to_string())?;
    let id = p.title_id.clone().ok_or("no param.sfo, so no title id to file this under")?;
    if p.missing_work_bin {
        return Err("this pkg has no work.bin and none was found beside it".into());
    }
    let existing = read_meta(&title_dir(&id));
    progress.lock().unwrap().probe = Some(ProbeCard {
        title: p.title.clone().unwrap_or_else(|| id.clone()),
        title_id: id.clone(),
        app_version: p.app_version.clone().unwrap_or_default(),
        kind: p.kind.to_string(),
        zipped: p.zipped,
        files: p.files,
        bytes: p.bytes,
        icon0: p.icon0.clone(),
        replacing: existing.is_some(),
    });
    let dir = title_dir(&id);
    let _ = fs::remove_dir_all(&dir);
    fs::create_dir_all(&dir).map_err(|e| e.to_string())?;
    let mut sink = DirSink::new(dir.clone());
    let mut report = |pr: stream::Progress<'_>| {
        let mut g = progress.lock().unwrap();
        g.stage = pr.stage.to_string();
        g.file = pr.file.to_string();
        g.done = pr.done;
        g.total = pr.total;
    };
    let content_id = stream::import(src, &mut sink, &mut report).map_err(|e| e.to_string())?;
    if let Some(b) = &p.icon0 {
        let _ = fs::write(dir.join("icon0.png"), b);
    }
    if let Some(b) = &p.pic0 {
        let _ = fs::write(dir.join("pic0.png"), b);
    }
    let meta = TitleMeta {
        title_id: id,
        title: p.title.clone().unwrap_or_else(|| p.title_id.clone().unwrap_or_default()),
        content_id,
        app_version: p.app_version.clone().unwrap_or_default(),
        source_kind: p.kind.to_string(),
        bytes: p.bytes,
        files: p.files as u32,
        has_icon: p.icon0.is_some(),
        has_pic: p.pic0.is_some(),
        imported_at: now_ms(),
        // A re-import keeps when the title was last played, as the browser's does.
        last_played_at: existing.map_or(0, |m| m.last_played_at),
    };
    write_meta(&meta).map_err(|e| e.to_string())?;
    Ok(meta)
}
