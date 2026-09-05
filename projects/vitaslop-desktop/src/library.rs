//! The desktop's data directory: imported titles, their records and images, saves,
//! and settings. One layout, spelled here and nowhere else:
//!
//! ```text
//! <home>/library/<TITLE_ID>/          the decrypted dump tree (what `--game` takes)
//! <home>/library/<TITLE_ID>/meta.json the TitleMeta record; icon0.png, pic0.png beside it
//! <home>/saves/<profile>/<TITLE_ID>/  the guest's own saved state (SaveStore)
//! <home>/settings.json                the global settings record
//! <home>/titles/<TITLE_ID>.json       a title's settings patch
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
    out.sort_by(|a, b| b.imported_at.cmp(&a.imported_at));
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
}

/// Import `path` into the library on this thread, reporting into `progress`.
pub fn import(path: &Path, progress: &Arc<Mutex<ImportProgress>>) -> Result<TitleMeta, String> {
    let src = DirSource::open(path).map_err(|e| e.to_string())?;
    let src: Rc<dyn ByteSource> = Rc::new(src);
    let p = stream::probe(src.clone()).map_err(|e| e.to_string())?;
    let id = p.title_id.clone().ok_or("no param.sfo, so no title id to file this under")?;
    if p.missing_work_bin {
        return Err("this pkg has no work.bin and none was found beside it".into());
    }
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
        last_played_at: 0,
    };
    write_meta(&meta).map_err(|e| e.to_string())?;
    Ok(meta)
}
