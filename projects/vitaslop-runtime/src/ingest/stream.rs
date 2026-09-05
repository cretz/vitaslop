//! Streaming ingest: the same peel as [`super::pipeline::decrypt_container`] - zip,
//! pkg, PFS, SELF - but over a RANDOM-ACCESS source and into a write-once sink, so
//! no container is ever resident.
//!
//! # Why a second path
//! `decrypt_container` takes a [`Vfs`](super::vfs::Vfs) whose `read` returns a whole
//! file, and builds a whole [`Game`](super::pipeline::Game) in memory. That is the
//! right shape for a test and for a desktop with the title already extracted, and
//! the wrong shape for the product's import: a retail pkg is up to 3.3 GB, a wasm32
//! heap is 4 GB with the emulator in it, and a phone has less than that in total.
//! Everything here works in bounded chunks. The pkg layer is AES-CTR, which seeks;
//! the PFS layer is per-page CBC-CTS with a per-page HMAC, which is why
//! [`GameData::decrypt_page`] exists; a zip's stored entries are byte ranges.
//! Only the executables (a few tens of MB) are held whole, because SELF unwrapping
//! needs them whole.
//!
//! Homebrew ships as a VPK: a zip with a plaintext fSELF `eboot.bin` and `param.sfo`
//! at its root and no PFS layer at all. It needs no key and no unwrapping here
//! (the loader unwraps an fSELF itself), so its import is a copy into the same tree.
//!
//! The output is exactly the dump tree `pipeline::dump_entries` writes
//! (`vitaslop-dump.txt`, `files/...`, `modules/...`), so everything downstream -
//! `mount_dump_lazy`, the browser's OPFS backing, the desktop's `load_dump` - is
//! unchanged and cannot tell which path produced the tree. The manifest is written
//! LAST: its presence is what marks the import complete.
//!
//! Platform I/O stays behind [`ByteSource`] and [`DumpSink`]. The browser implements
//! them over `FileReaderSync` + OPFS sync access handles in a worker; the desktop
//! over `std::fs`.

use std::cell::RefCell;
use std::collections::HashMap;
use std::rc::Rc;

use super::filesdb::FilesDb;
use super::pfs::{FileCtx, PfsImage};
use super::pfscrypt::GameData;
use super::pipeline::{DUMP_MAGIC, DUMP_MANIFEST, WORK_BIN_PATH};
use super::pkg::{is_directory, PkgHeader, PkgItem};
use super::rif::Rif;
use super::self2elf::self2elf;
use super::unicv::UnicvDb;
use super::{sfo, Error};
use aes::cipher::{KeyIvInit, StreamCipher, StreamCipherSeek};

/// Random access over a set of named files. `read_at` may return short only at the
/// end of the file.
pub trait ByteSource {
    fn list(&self) -> Vec<String>;
    fn size(&self, path: &str) -> Option<u64>;
    fn read_at(&self, path: &str, off: u64, buf: &mut [u8]) -> Result<usize, Error>;
}

/// Where the dump tree goes. One file at a time: `begin`, any number of `write`s,
/// `finish`. Paths are dump-relative (`files/...`, `modules/...`, the manifest).
pub trait DumpSink {
    fn begin(&mut self, path: &str, size: u64) -> Result<(), Error>;
    fn write(&mut self, bytes: &[u8]) -> Result<(), Error>;
    fn finish(&mut self) -> Result<(), Error>;
}

/// One progress report. `done`/`total` are SOURCE bytes consumed so far against the
/// source's total - the number that moves at the rate the disk reads, which is what
/// a person watching an import needs to see.
pub struct Progress<'a> {
    pub stage: &'static str,
    pub file: &'a str,
    pub done: u64,
    pub total: u64,
}

/// What the source turned out to be and who it is, read BEFORE anything is written.
/// The importer needs the title id to name the destination, and a person needs the
/// name and the size to decide whether to spend the minutes.
pub struct Probe {
    /// `"pkg"`, `"pfs"`, `"dump"`; wrapped in a zip if `zipped`.
    pub kind: &'static str,
    pub zipped: bool,
    pub title_id: Option<String>,
    pub title: Option<String>,
    pub content_id: Option<String>,
    pub app_version: Option<String>,
    /// Source bytes the import will read.
    pub bytes: u64,
    pub files: usize,
    /// `sce_sys/icon0.png` and `sce_sys/pic0.png`, if the title has them.
    pub icon0: Option<Vec<u8>>,
    pub pic0: Option<Vec<u8>>,
    /// The pkg carried no `work.bin` and none was supplied beside it - the import
    /// will fail at the PFS layer, and the UI should say so before starting.
    pub missing_work_bin: bool,
    /// Every dump-relative path [`import`] MAY write, in order, the manifest last. A
    /// sink that must open its files asynchronously (OPFS) opens these before the one
    /// synchronous import call. A superset: a module-named file that turns out not to
    /// be a SELF is listed under `modules/` and never written.
    pub outputs: Vec<String>,
}

const CHUNK: usize = 1 << 20;

fn read_whole(src: &dyn ByteSource, path: &str) -> Result<Vec<u8>, Error> {
    let n = src.size(path).ok_or_else(|| Error::MissingFile(path.to_string()))?;
    let mut out = vec![0u8; n as usize];
    let mut off = 0usize;
    while off < out.len() {
        let got = src.read_at(path, off as u64, &mut out[off..])?;
        if got == 0 {
            return Err(Error::Io(format!("short read of {path} at {off}")));
        }
        off += got;
    }
    Ok(out)
}

fn read_head(src: &dyn ByteSource, path: &str, n: usize) -> Result<Vec<u8>, Error> {
    let size = src.size(path).unwrap_or(0) as usize;
    let mut out = vec![0u8; n.min(size)];
    let mut off = 0usize;
    while off < out.len() {
        let got = src.read_at(path, off as u64, &mut out[off..])?;
        if got == 0 {
            break;
        }
        off += got;
    }
    out.truncate(off);
    Ok(out)
}

fn under(root: &str, rel: &str) -> String {
    if root.is_empty() {
        rel.to_string()
    } else {
        format!("{root}/{rel}")
    }
}

// ============================== zip as a source ==============================

struct ZipEntry {
    method: u16,
    comp_size: u64,
    uncomp_size: u64,
    local_off: u64,
    data_off: RefCell<Option<u64>>,
}

/// A zip's entries as files. Stored entries are byte ranges of the archive; a
/// deflated entry is inflated whole on first touch and kept (a deflated pkg inside a
/// zip is the one shape this cannot stream, and it says so by size).
pub struct ZipSource {
    inner: Rc<dyn ByteSource>,
    path: String,
    entries: HashMap<String, ZipEntry>,
    inflated: RefCell<HashMap<String, Rc<Vec<u8>>>>,
}

const EOCD_SIG: u32 = 0x0605_4b50;
const CDIR_SIG: u32 = 0x0201_4b50;
const LOCAL_SIG: u32 = 0x0403_4b50;

fn le16(b: &[u8], at: usize) -> Result<u16, Error> {
    let s = b.get(at..at + 2).ok_or(Error::OutOfBounds("zip u16"))?;
    Ok(u16::from_le_bytes([s[0], s[1]]))
}
fn le32(b: &[u8], at: usize) -> Result<u32, Error> {
    let s = b.get(at..at + 4).ok_or(Error::OutOfBounds("zip u32"))?;
    Ok(u32::from_le_bytes([s[0], s[1], s[2], s[3]]))
}

impl ZipSource {
    pub fn open(inner: Rc<dyn ByteSource>, path: &str) -> Result<ZipSource, Error> {
        let size = inner.size(path).ok_or_else(|| Error::MissingFile(path.to_string()))?;
        // The end-of-central-directory record is in the last 64 KB + 22 bytes.
        let tail_len = size.min(22 + 0xFFFF) as usize;
        let mut tail = vec![0u8; tail_len];
        let base = size - tail_len as u64;
        let mut off = 0;
        while off < tail_len {
            let got = inner.read_at(path, base + off as u64, &mut tail[off..])?;
            if got == 0 {
                break;
            }
            off += got;
        }
        let eocd = super::zip::find_eocd(&tail).ok_or(Error::BadMagic("zip EOCD"))?;
        if le32(&tail, eocd)? != EOCD_SIG {
            return Err(Error::BadMagic("zip EOCD"));
        }
        let total = le16(&tail, eocd + 10)? as usize;
        let cd_size = le32(&tail, eocd + 12)? as u64;
        let cd_off = le32(&tail, eocd + 16)? as u64;
        if total == 0xFFFF || cd_off == 0xFFFF_FFFF || cd_size == 0xFFFF_FFFF {
            return Err(Error::BadMagic("zip64 archive (not supported) - unzip it first"));
        }
        let mut cd = vec![0u8; cd_size as usize];
        let mut off = 0;
        while off < cd.len() {
            let got = inner.read_at(path, cd_off + off as u64, &mut cd[off..])?;
            if got == 0 {
                return Err(Error::OutOfBounds("zip central directory"));
            }
            off += got;
        }
        let mut entries = HashMap::new();
        let mut at = 0usize;
        for _ in 0..total {
            if le32(&cd, at)? != CDIR_SIG {
                return Err(Error::BadMagic("zip central dir entry"));
            }
            let method = le16(&cd, at + 10)?;
            let comp_size = le32(&cd, at + 20)? as u64;
            let uncomp_size = le32(&cd, at + 24)? as u64;
            let name_len = le16(&cd, at + 28)? as usize;
            let extra_len = le16(&cd, at + 30)? as usize;
            let comment_len = le16(&cd, at + 32)? as usize;
            let local_off = le32(&cd, at + 42)? as u64;
            let name = cd.get(at + 46..at + 46 + name_len).ok_or(Error::OutOfBounds("zip name"))?;
            let name = String::from_utf8_lossy(name).into_owned();
            if !name.ends_with('/') {
                entries.insert(
                    name,
                    ZipEntry { method, comp_size, uncomp_size, local_off, data_off: RefCell::new(None) },
                );
            }
            at += 46 + name_len + extra_len + comment_len;
        }
        Ok(ZipSource { inner, path: path.to_string(), entries, inflated: RefCell::new(HashMap::new()) })
    }

    fn data_off(&self, e: &ZipEntry) -> Result<u64, Error> {
        if let Some(d) = *e.data_off.borrow() {
            return Ok(d);
        }
        let mut hdr = [0u8; 30];
        let got = self.inner.read_at(&self.path, e.local_off, &mut hdr)?;
        if got < 30 || le32(&hdr, 0)? != LOCAL_SIG {
            return Err(Error::BadMagic("zip local header"));
        }
        let name_len = le16(&hdr, 26)? as u64;
        let extra_len = le16(&hdr, 28)? as u64;
        let d = e.local_off + 30 + name_len + extra_len;
        *e.data_off.borrow_mut() = Some(d);
        Ok(d)
    }

    fn inflated(&self, name: &str, e: &ZipEntry) -> Result<Rc<Vec<u8>>, Error> {
        if let Some(v) = self.inflated.borrow().get(name) {
            return Ok(v.clone());
        }
        let d = self.data_off(e)?;
        let mut comp = vec![0u8; e.comp_size as usize];
        let mut off = 0;
        while off < comp.len() {
            let got = self.inner.read_at(&self.path, d + off as u64, &mut comp[off..])?;
            if got == 0 {
                return Err(Error::OutOfBounds("zip entry data"));
            }
            off += got;
        }
        let out = miniz_oxide::inflate::decompress_to_vec_with_limit(&comp, (e.uncomp_size as usize).max(1))
            .map_err(|_| Error::IntegrityCheck("zip deflate"))?;
        let out = Rc::new(out);
        self.inflated.borrow_mut().insert(name.to_string(), out.clone());
        Ok(out)
    }
}

impl ByteSource for ZipSource {
    fn list(&self) -> Vec<String> {
        let mut v: Vec<String> = self.entries.keys().cloned().collect();
        v.sort();
        v
    }
    fn size(&self, path: &str) -> Option<u64> {
        self.entries.get(path).map(|e| e.uncomp_size)
    }
    fn read_at(&self, path: &str, off: u64, buf: &mut [u8]) -> Result<usize, Error> {
        let e = self.entries.get(path).ok_or_else(|| Error::MissingFile(path.to_string()))?;
        match e.method {
            0 => {
                if off >= e.uncomp_size {
                    return Ok(0);
                }
                let want = buf.len().min((e.uncomp_size - off) as usize);
                let d = self.data_off(e)?;
                self.inner.read_at(&self.path, d + off, &mut buf[..want])
            }
            8 => {
                let data = self.inflated(path, e)?;
                let off = off as usize;
                if off >= data.len() {
                    return Ok(0);
                }
                let n = buf.len().min(data.len() - off);
                buf[..n].copy_from_slice(&data[off..off + n]);
                Ok(n)
            }
            _ => Err(Error::BadMagic("zip method")),
        }
    }
}

// ============================== pkg as a source ==============================

type Aes128Ctr = ctr::Ctr128BE<aes::Aes128>;

/// A pkg's items as files, decrypted on read. AES-CTR seeks, so any byte range of
/// any item costs exactly that range. A `work.bin` found BESIDE the pkg in the outer
/// source is presented at the path the PFS layer expects.
pub struct PkgSource {
    inner: Rc<dyn ByteSource>,
    path: String,
    header: PkgHeader,
    session_key: [u8; 16],
    items: HashMap<String, PkgItem>,
    /// `(outer path)` of a work.bin supplied beside the pkg, if any.
    work_bin: Option<String>,
}

impl PkgSource {
    pub fn open(inner: Rc<dyn ByteSource>, path: &str, work_bin: Option<String>) -> Result<PkgSource, Error> {
        let head = read_head(&*inner, path, 0x1000)?;
        let header = PkgHeader::parse(&head)?;
        let session_key = header.session_key()?;
        let mut me = PkgSource { inner, path: path.to_string(), header, session_key, items: HashMap::new(), work_bin };
        let table_len = me.header.item_count as usize * 0x20;
        let table = me.decrypt_at(me.header.data_offset, table_len)?;
        let mut items = HashMap::new();
        for i in 0..me.header.item_count as usize {
            let e = i * 0x20;
            let be32 = |at: usize| -> Result<u32, Error> {
                let s = table.get(at..at + 4).ok_or(Error::OutOfBounds("pkg u32"))?;
                Ok(u32::from_be_bytes([s[0], s[1], s[2], s[3]]))
            };
            let be64 = |at: usize| -> Result<u64, Error> {
                let s = table.get(at..at + 8).ok_or(Error::OutOfBounds("pkg u64"))?;
                Ok(u64::from_be_bytes([s[0], s[1], s[2], s[3], s[4], s[5], s[6], s[7]]))
            };
            let name_offset = be32(e)? as u64;
            let name_size = be32(e + 4)? as usize;
            let data_offset = be64(e + 8)?;
            let data_size = be64(e + 0x10)?;
            let flags = be32(e + 0x18)?;
            let name_bytes = me.decrypt_at(me.header.data_offset + name_offset, name_size)?;
            let name = String::from_utf8_lossy(&name_bytes).into_owned();
            if is_directory(flags) {
                continue;
            }
            items.insert(
                name.clone(),
                PkgItem { name, data_offset: me.header.data_offset + data_offset, data_size, flags },
            );
        }
        me.items = items;
        Ok(me)
    }

    pub fn content_id(&self) -> &str {
        &self.header.content_id
    }

    fn decrypt_at(&self, off: u64, len: usize) -> Result<Vec<u8>, Error> {
        let mut buf = vec![0u8; len];
        let got = self.decrypt_into(off, &mut buf)?;
        buf.truncate(got);
        Ok(buf)
    }

    fn decrypt_into(&self, off: u64, buf: &mut [u8]) -> Result<usize, Error> {
        let mut filled = 0;
        while filled < buf.len() {
            let got = self.inner.read_at(&self.path, off + filled as u64, &mut buf[filled..])?;
            if got == 0 {
                break;
            }
            filled += got;
        }
        let rel = off.checked_sub(self.header.data_offset).ok_or(Error::OutOfBounds("pkg offset before data"))?;
        let mut c = Aes128Ctr::new((&self.session_key).into(), (&self.header.riv).into());
        c.seek(rel);
        c.apply_keystream(&mut buf[..filled]);
        Ok(filled)
    }
}

impl ByteSource for PkgSource {
    fn list(&self) -> Vec<String> {
        let mut v: Vec<String> = self.items.keys().cloned().collect();
        if self.work_bin.is_some() && !self.items.contains_key(WORK_BIN_PATH) {
            v.push(WORK_BIN_PATH.to_string());
        }
        v.sort();
        v
    }
    fn size(&self, path: &str) -> Option<u64> {
        if let Some(i) = self.items.get(path) {
            return Some(i.data_size);
        }
        if path == WORK_BIN_PATH {
            if let Some(p) = &self.work_bin {
                return self.inner.size(p);
            }
        }
        None
    }
    fn read_at(&self, path: &str, off: u64, buf: &mut [u8]) -> Result<usize, Error> {
        if let Some(i) = self.items.get(path) {
            if off >= i.data_size {
                return Ok(0);
            }
            let want = buf.len().min((i.data_size - off) as usize);
            return self.decrypt_into(i.data_offset + off, &mut buf[..want]);
        }
        if path == WORK_BIN_PATH {
            if let Some(p) = &self.work_bin {
                return self.inner.read_at(p, off, buf);
            }
        }
        Err(Error::MissingFile(path.to_string()))
    }
}

// ============================== detection ==============================

/// A source resolved down to the layer the dump is made from.
pub enum Opened {
    /// A dump tree this pipeline wrote: copy through.
    Dump { src: Rc<dyn ByteSource>, root: String },
    /// A PFS app root (raw dump, or a pkg's items): decrypt.
    Pfs { src: Rc<dyn ByteSource>, root: String, kind: &'static str },
    /// A homebrew app root: a plaintext `eboot.bin` beside `sce_sys/`, nothing encrypted.
    Homebrew { src: Rc<dyn ByteSource>, root: String },
}

/// Peel the outer layers - a single zip, a pkg - until a dump or PFS root is found.
pub fn open(src: Rc<dyn ByteSource>) -> Result<(Opened, bool), Error> {
    let names = src.list();
    // One zip and nothing else that is a container: look inside it.
    let zips: Vec<&String> = names
        .iter()
        .filter(|p| {
            let l = p.to_ascii_lowercase();
            l.ends_with(".zip") || l.ends_with(".vpk")
        })
        .collect();
    if zips.len() == 1 && !names.iter().any(|p| p.ends_with("sce_pfs/files.db") || p.ends_with(DUMP_MANIFEST)) {
        let z = ZipSource::open(src.clone(), zips[0])?;
        let (o, _) = open_unzipped(Rc::new(z))?;
        return Ok((o, true));
    }
    open_unzipped(src)
}

fn open_unzipped(src: Rc<dyn ByteSource>) -> Result<(Opened, bool), Error> {
    let names = src.list();
    for p in &names {
        if p == DUMP_MANIFEST || p.ends_with(&format!("/{DUMP_MANIFEST}")) {
            let root = p.strip_suffix(DUMP_MANIFEST).unwrap_or("").trim_end_matches('/').to_string();
            return Ok((Opened::Dump { src, root }, false));
        }
    }
    for p in &names {
        if p == "sce_pfs/files.db" || p.ends_with("/sce_pfs/files.db") {
            let root = p.strip_suffix("sce_pfs/files.db").unwrap_or("").trim_end_matches('/').to_string();
            return Ok((Opened::Pfs { src, root, kind: "pfs" }, false));
        }
    }
    // Homebrew: a plaintext eboot with no PFS around it (checked AFTER the PFS root, since
    // a raw dump also has an eboot.bin - an encrypted one).
    for p in &names {
        if p == "eboot.bin" || p.ends_with("/eboot.bin") {
            let head = read_head(&*src, p, 4)?;
            if head.as_slice() == b"SCE\0" || head.as_slice() == b"\x7fELF" {
                let root = p.strip_suffix("eboot.bin").unwrap_or("").trim_end_matches('/').to_string();
                return Ok((Opened::Homebrew { src, root }, false));
            }
        }
    }
    // A pkg: by magic, not by name - one title on this machine has a 100-character
    // random name with the extension, and a person may have renamed theirs.
    for p in &names {
        let big = src.size(p).unwrap_or(0) >= 0x100;
        if big && !p.ends_with("work.bin") {
            let head = read_head(&*src, p, 4)?;
            if head.as_slice() == b"\x7fPKG" {
                let dir = p.rsplit_once('/').map(|(d, _)| d).unwrap_or("");
                let work = names
                    .iter()
                    .find(|w| w.ends_with("work.bin") && w.rsplit_once('/').map(|(d, _)| d).unwrap_or("") == dir)
                    .or_else(|| names.iter().find(|w| w.ends_with("work.bin")))
                    .cloned();
                let pkg = PkgSource::open(src.clone(), p, work)?;
                return Ok((Opened::Pfs { src: Rc::new(pkg), root: String::new(), kind: "pkg" }, false));
            }
        }
    }
    Err(Error::UnknownContainer)
}

// ============================== PFS over a source ==============================

struct PfsLayer {
    image: PfsImage,
    rif: Rif,
    crypto: GameData,
}

impl PfsLayer {
    fn open(src: &dyn ByteSource, root: &str) -> Result<PfsLayer, Error> {
        let files_db = FilesDb::parse(&read_whole(src, &under(root, "sce_pfs/files.db"))?)?;
        let unicv = UnicvDb::parse(&read_whole(src, &under(root, "sce_pfs/unicv.db"))?)?;
        let rif = Rif::parse(&read_whole(src, &under(root, WORK_BIN_PATH))?)?;
        let image = PfsImage::new(files_db, unicv)?;
        let crypto = GameData::from_klicensee(&rif.key);
        Ok(PfsLayer { image, rif, crypto })
    }

    /// Decrypt one file whole (for the small ones a probe wants).
    fn read_file(&self, src: &dyn ByteSource, root: &str, path: &str) -> Result<Vec<u8>, Error> {
        let ct = read_whole(src, &under(root, path))?;
        self.image.decrypt(path, &ct, &self.rif.key, &self.crypto)
    }

    /// Stream one file's plaintext through `emit`, page by page.
    fn stream_file(
        &self,
        src: &dyn ByteSource,
        root: &str,
        ctx: &FileCtx,
        path: &str,
        mut emit: impl FnMut(&[u8]) -> Result<(), Error>,
        mut tick: impl FnMut(u64),
    ) -> Result<u64, Error> {
        let full = under(root, path);
        let size = src.size(&full).ok_or_else(|| Error::MissingFile(full.clone()))?;
        let mut consumed = 0u64;
        if !ctx.encrypted {
            let mut buf = vec![0u8; CHUNK];
            let mut off = 0u64;
            while off < size {
                let got = src.read_at(&full, off, &mut buf)?;
                if got == 0 {
                    return Err(Error::Io(format!("short read of {full}")));
                }
                emit(&buf[..got])?;
                off += got as u64;
                consumed += got as u64;
                tick(got as u64);
            }
            return Ok(consumed);
        }
        let keys = self.crypto.page_keys(ctx);
        let page = ctx.page_size.max(1) as usize;
        let pages_per_read = (CHUNK / page).max(1);
        let mut buf = vec![0u8; page * pages_per_read];
        let mut off = 0u64;
        let mut idx = 0usize;
        let mut remaining = ctx.plaintext_size as u64;
        while off < size {
            let mut filled = 0usize;
            let want = buf.len().min((size - off) as usize);
            while filled < want {
                let got = src.read_at(&full, off + filled as u64, &mut buf[filled..want])?;
                if got == 0 {
                    break;
                }
                filled += got;
            }
            if filled == 0 {
                return Err(Error::Io(format!("short read of {full}")));
            }
            for chunk in buf[..filled].chunks(page) {
                let pt = self.crypto.decrypt_page(ctx, &keys, idx, chunk)?;
                let take = (pt.len() as u64).min(remaining) as usize;
                if take > 0 {
                    emit(&pt[..take])?;
                }
                remaining -= take as u64;
                idx += 1;
            }
            off += filled as u64;
            consumed += filled as u64;
            tick(filled as u64);
        }
        Ok(consumed)
    }
}

fn is_module_path(path: &str) -> bool {
    path == "eboot.bin" || (path.starts_with("sce_module/") && path.ends_with(".suprx"))
}

// ============================== the probe ==============================

fn sfo_probe(probe: &mut Probe, sfo_bytes: &[u8]) {
    probe.title_id = sfo::title_id(sfo_bytes);
    probe.title = sfo::text_field(sfo_bytes, "TITLE").or_else(|| sfo::text_field(sfo_bytes, "STITLE"));
    probe.app_version = sfo::text_field(sfo_bytes, "APP_VER");
    if probe.content_id.is_none() {
        probe.content_id = sfo::text_field(sfo_bytes, "CONTENT_ID");
    }
}

/// Identify a source without writing anything. Reads only the databases, the
/// license, `param.sfo` and the two images.
pub fn probe(src: Rc<dyn ByteSource>) -> Result<Probe, Error> {
    let (opened, zipped) = open(src)?;
    let mut p = Probe {
        kind: "",
        zipped,
        title_id: None,
        title: None,
        content_id: None,
        app_version: None,
        bytes: 0,
        files: 0,
        icon0: None,
        pic0: None,
        missing_work_bin: false,
        outputs: Vec::new(),
    };
    match opened {
        Opened::Dump { src, root } => {
            p.kind = "dump";
            let prefix = under(&root, "");
            let names: Vec<String> = src.list().into_iter().filter(|n| n.starts_with(&prefix)).collect();
            p.files = names.len();
            p.bytes = names.iter().map(|n| src.size(n).unwrap_or(0)).sum();
            let mut outs: Vec<String> = names.iter().map(|n| n[prefix.len()..].to_string()).filter(|n| n != DUMP_MANIFEST).collect();
            outs.sort();
            outs.push(DUMP_MANIFEST.to_string());
            p.outputs = outs;
            if let Ok(m) = read_whole(&*src, &under(&root, DUMP_MANIFEST)) {
                let m = String::from_utf8_lossy(&m);
                p.content_id = m.lines().find_map(|l| l.strip_prefix("content_id=").map(str::to_string));
            }
            if let Ok(s) = read_whole(&*src, &under(&root, "files/sce_sys/param.sfo")) {
                sfo_probe(&mut p, &s);
            }
            p.icon0 = read_whole(&*src, &under(&root, "files/sce_sys/icon0.png")).ok();
            p.pic0 = read_whole(&*src, &under(&root, "files/sce_sys/pic0.png")).ok();
        }
        Opened::Homebrew { src, root } => {
            p.kind = "vpk";
            let prefix = under(&root, "");
            let names: Vec<String> = src.list().into_iter().filter(|n| n.starts_with(&prefix)).collect();
            p.files = names.len();
            p.bytes = names.iter().map(|n| src.size(n).unwrap_or(0)).sum();
            let rels: Vec<String> = names.iter().map(|n| n[prefix.len()..].to_string()).collect();
            let mut outs: Vec<String> = rels.iter().map(|r| format!("files/{r}")).collect();
            outs.extend(rels.iter().filter(|r| is_module_path(r)).map(|r| format!("modules/{r}")));
            outs.push(DUMP_MANIFEST.to_string());
            p.outputs = outs;
            if let Ok(s) = read_whole(&*src, &under(&root, "sce_sys/param.sfo")) {
                sfo_probe(&mut p, &s);
            }
            p.icon0 = read_whole(&*src, &under(&root, "sce_sys/icon0.png")).ok();
            p.pic0 = read_whole(&*src, &under(&root, "sce_sys/pic0.png")).ok();
        }
        Opened::Pfs { src, root, kind } => {
            p.kind = kind;
            let names = src.list();
            p.files = names.len();
            p.bytes = names.iter().map(|n| src.size(n).unwrap_or(0)).sum();
            if src.size(&under(&root, WORK_BIN_PATH)).is_none() {
                p.missing_work_bin = true;
                // Still name the title: param.sfo is plaintext in every pkg and dump.
                if let Ok(s) = read_whole(&*src, &under(&root, "sce_sys/param.sfo")) {
                    sfo_probe(&mut p, &s);
                }
                p.icon0 = read_whole(&*src, &under(&root, "sce_sys/icon0.png")).ok();
                return Ok(p);
            }
            let pfs = PfsLayer::open(&*src, &root)?;
            p.content_id = Some(pfs.rif.content_id.clone());
            let files = pfs.image.files();
            p.files = files.len();
            p.bytes = files.iter().map(|f| src.size(&under(&root, &f.path)).unwrap_or(0)).sum();
            let mut outs: Vec<String> = files.iter().map(|f| format!("files/{}", f.path)).collect();
            outs.extend(files.iter().filter(|f| is_module_path(&f.path)).map(|f| format!("modules/{}", f.path)));
            outs.push(DUMP_MANIFEST.to_string());
            p.outputs = outs;
            if let Ok(s) = pfs.read_file(&*src, &root, "sce_sys/param.sfo") {
                sfo_probe(&mut p, &s);
            }
            p.icon0 = pfs.read_file(&*src, &root, "sce_sys/icon0.png").ok();
            p.pic0 = pfs.read_file(&*src, &root, "sce_sys/pic0.png").ok();
        }
    }
    Ok(p)
}

// ============================== the import ==============================

/// Peel `src` into `sink` as a dump tree. Returns the content id.
pub fn import(
    src: Rc<dyn ByteSource>,
    sink: &mut dyn DumpSink,
    progress: &mut dyn FnMut(Progress<'_>),
) -> Result<String, Error> {
    let (opened, _) = open(src)?;
    match opened {
        Opened::Dump { src, root } => import_dump(&*src, &root, sink, progress),
        Opened::Pfs { src, root, .. } => import_pfs(&*src, &root, sink, progress),
        Opened::Homebrew { src, root } => import_homebrew(&*src, &root, sink, progress),
    }
}

/// A homebrew app: every file copied under `files/`, the plaintext executables ALSO
/// under `modules/` as they are (the loader unwraps an fSELF), the manifest last.
fn import_homebrew(
    src: &dyn ByteSource,
    root: &str,
    sink: &mut dyn DumpSink,
    progress: &mut dyn FnMut(Progress<'_>),
) -> Result<String, Error> {
    let prefix = under(root, "");
    let mut names: Vec<String> = src.list().into_iter().filter(|n| n.starts_with(&prefix)).collect();
    names.sort();
    let total: u64 = names.iter().map(|n| src.size(n).unwrap_or(0)).sum();
    let mut done = 0u64;
    let mut modules: Vec<String> = Vec::new();
    for n in &names {
        let rel = &n[prefix.len()..];
        copy_through(src, n, &format!("files/{rel}"), sink, |b| {
            done += b;
            progress(Progress { stage: "copy", file: rel, done, total });
        })?;
        if is_module_path(rel) {
            let head = read_head(src, n, 4)?;
            if head.as_slice() == b"SCE\0" || head.as_slice() == b"\x7fELF" {
                modules.push(rel.to_string());
            }
        }
    }
    modules.sort();
    modules.sort_by_key(|p| p == "eboot.bin");
    for m in &modules {
        copy_through(src, &under(root, m), &format!("modules/{m}"), sink, |_| {})?;
    }
    let sfo = read_whole(src, &under(root, "sce_sys/param.sfo")).ok();
    let content_id = sfo
        .as_deref()
        .and_then(|s| sfo::text_field(s, "CONTENT_ID"))
        .filter(|c| !c.is_empty())
        .or_else(|| sfo.as_deref().and_then(sfo::title_id).map(|t| format!("HOMEBREW-{t}")))
        .unwrap_or_else(|| "HOMEBREW".to_string());
    let mut manifest = String::new();
    manifest.push_str(DUMP_MAGIC);
    manifest.push('\n');
    manifest.push_str(&format!("content_id={content_id}\n"));
    for m in &modules {
        manifest.push_str(&format!("module={m}\n"));
    }
    sink.begin(DUMP_MANIFEST, manifest.len() as u64)?;
    sink.write(manifest.as_bytes())?;
    sink.finish()?;
    Ok(content_id)
}

fn copy_through(
    src: &dyn ByteSource,
    from: &str,
    to: &str,
    sink: &mut dyn DumpSink,
    mut tick: impl FnMut(u64),
) -> Result<(), Error> {
    let size = src.size(from).ok_or_else(|| Error::MissingFile(from.to_string()))?;
    sink.begin(to, size)?;
    let mut buf = vec![0u8; CHUNK];
    let mut off = 0u64;
    while off < size {
        let got = src.read_at(from, off, &mut buf)?;
        if got == 0 {
            return Err(Error::Io(format!("short read of {from}")));
        }
        sink.write(&buf[..got])?;
        off += got as u64;
        tick(got as u64);
    }
    sink.finish()
}

fn import_dump(
    src: &dyn ByteSource,
    root: &str,
    sink: &mut dyn DumpSink,
    progress: &mut dyn FnMut(Progress<'_>),
) -> Result<String, Error> {
    let prefix = under(root, "");
    let mut names: Vec<String> = src.list().into_iter().filter(|n| n.starts_with(&prefix)).collect();
    names.sort();
    let total: u64 = names.iter().map(|n| src.size(n).unwrap_or(0)).sum();
    let mut done = 0u64;
    let manifest = under(root, DUMP_MANIFEST);
    let content_id = {
        let m = read_whole(src, &manifest)?;
        let m = String::from_utf8_lossy(&m);
        if m.lines().next() != Some(DUMP_MAGIC) {
            return Err(Error::BadMagic("dump manifest"));
        }
        m.lines().find_map(|l| l.strip_prefix("content_id=").map(str::to_string)).unwrap_or_default()
    };
    for n in names.iter().filter(|n| **n != manifest) {
        let rel = &n[prefix.len()..];
        copy_through(src, n, rel, sink, |b| {
            done += b;
            progress(Progress { stage: "copy", file: rel, done, total });
        })?;
    }
    // Last: the marker.
    copy_through(src, &manifest, DUMP_MANIFEST, sink, |_| {})?;
    Ok(content_id)
}

fn import_pfs(
    src: &dyn ByteSource,
    root: &str,
    sink: &mut dyn DumpSink,
    progress: &mut dyn FnMut(Progress<'_>),
) -> Result<String, Error> {
    let pfs = PfsLayer::open(src, root)?;
    let files = pfs.image.files();
    let total: u64 = files.iter().map(|f| src.size(&under(root, &f.path)).unwrap_or(0)).sum();
    let mut done = 0u64;
    let mut modules: Vec<(String, Vec<u8>)> = Vec::new();
    for f in &files {
        let ctx = pfs.image.ctx_of(f, &pfs.rif.key);
        let out_path = format!("files/{}", f.path);
        sink.begin(&out_path, ctx.plaintext_size as u64)?;
        // An executable is kept whole for the SELF unwrap that follows.
        let mut keep: Option<Vec<u8>> = if is_module_path(&f.path) { Some(Vec::with_capacity(ctx.plaintext_size)) } else { None };
        // Progress per CHUNK, not per file: a title's bytes are not spread evenly over
        // its files, and a counter that only moves when a file finishes sits still for
        // the whole of a gigabyte file, which reads as a hang and was reported as one.
        let before = done;
        let consumed = pfs.stream_file(
            src,
            root,
            &ctx,
            &f.path,
            |pt| {
                if let Some(k) = keep.as_mut() {
                    k.extend_from_slice(pt);
                }
                sink.write(pt)
            },
            |n| {
                done += n;
                progress(Progress { stage: "decrypt", file: &f.path, done, total });
            },
        )?;
        sink.finish()?;
        done = before + consumed;
        if let Some(bytes) = keep {
            if bytes.len() >= 4 && &bytes[..4] == b"SCE\0" {
                modules.push((f.path.clone(), bytes));
            }
        }
    }
    modules.sort_by(|a, b| a.0.cmp(&b.0));
    modules.sort_by_key(|(p, _)| p == "eboot.bin");
    let mut manifest = String::new();
    manifest.push_str(DUMP_MAGIC);
    manifest.push('\n');
    manifest.push_str(&format!("content_id={}\n", pfs.rif.content_id));
    for (path, bytes) in &modules {
        progress(Progress { stage: "unwrap", file: path, done, total });
        let elf = self2elf(bytes, &pfs.rif.key)?;
        let out_path = format!("modules/{path}");
        sink.begin(&out_path, elf.len() as u64)?;
        sink.write(&elf)?;
        sink.finish()?;
        manifest.push_str(&format!("module={path}\n"));
    }
    // Last: the marker.
    sink.begin(DUMP_MANIFEST, manifest.len() as u64)?;
    sink.write(manifest.as_bytes())?;
    sink.finish()?;
    Ok(pfs.rif.content_id.clone())
}

// ============================== in-memory adapters ==============================

/// A [`ByteSource`] over bytes already in memory - for tests and small fixtures.
pub struct MemSource {
    pub files: HashMap<String, Vec<u8>>,
}

impl ByteSource for MemSource {
    fn list(&self) -> Vec<String> {
        let mut v: Vec<String> = self.files.keys().cloned().collect();
        v.sort();
        v
    }
    fn size(&self, path: &str) -> Option<u64> {
        self.files.get(path).map(|b| b.len() as u64)
    }
    fn read_at(&self, path: &str, off: u64, buf: &mut [u8]) -> Result<usize, Error> {
        let b = self.files.get(path).ok_or_else(|| Error::MissingFile(path.to_string()))?;
        let off = off as usize;
        if off >= b.len() {
            return Ok(0);
        }
        let n = buf.len().min(b.len() - off);
        buf[..n].copy_from_slice(&b[off..off + n]);
        Ok(n)
    }
}

/// A [`DumpSink`] that collects into memory - for tests.
#[derive(Default)]
pub struct MemSink {
    pub files: Vec<(String, Vec<u8>)>,
    cur: Option<(String, Vec<u8>)>,
}

impl DumpSink for MemSink {
    fn begin(&mut self, path: &str, size: u64) -> Result<(), Error> {
        self.cur = Some((path.to_string(), Vec::with_capacity(size as usize)));
        Ok(())
    }
    fn write(&mut self, bytes: &[u8]) -> Result<(), Error> {
        self.cur.as_mut().ok_or_else(|| Error::Io("write before begin".into()))?.1.extend_from_slice(bytes);
        Ok(())
    }
    fn finish(&mut self) -> Result<(), Error> {
        let f = self.cur.take().ok_or_else(|| Error::Io("finish before begin".into()))?;
        self.files.push(f);
        Ok(())
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::ingest::pipeline::{decrypt_container, dump_entries};
    use crate::ingest::vfs::{DirVfs, Vfs};

    fn dir_source(root: &std::path::Path) -> MemSource {
        let vfs = DirVfs::new(root);
        let mut files = HashMap::new();
        for p in vfs.list() {
            files.insert(p.clone(), vfs.read(&p).unwrap());
        }
        MemSource { files }
    }

    /// The streaming path must produce BYTE-IDENTICAL output to the resident one on a
    /// real title: same file set, same plaintext, same unwrapped modules, same
    /// manifest. Needs `VITASLOP_GAME_DIR` (a pkg + work.bin dir, or a PFS dump).
    #[test]
    fn streaming_matches_resident_on_a_real_title() {
        let Some(dir) = crate::ingest::testfix::game_dir() else { return };
        let src = Rc::new(dir_source(&dir));
        let mut sink = MemSink::default();
        let mut last = 0;
        let cid = import(src, &mut sink, &mut |p| {
            assert!(p.done >= last, "progress went backwards");
            last = p.done;
        })
        .expect("stream import");
        let mut vfs = DirVfs::new(&dir);
        let game = decrypt_container(&mut vfs).expect("resident decrypt");
        assert_eq!(cid, game.content_id);
        let mut expect = dump_entries(&game);
        expect.sort_by(|a, b| a.0.cmp(&b.0));
        let mut got = sink.files;
        got.sort_by(|a, b| a.0.cmp(&b.0));
        assert_eq!(got.len(), expect.len(), "file count");
        for ((gp, gb), (ep, eb)) in got.iter().zip(expect.iter()) {
            assert_eq!(gp, ep);
            assert!(gb == eb, "{gp}: bytes differ ({} vs {})", gb.len(), eb.len());
        }
    }

    #[test]
    fn probe_names_a_real_title() {
        let Some(dir) = crate::ingest::testfix::game_dir() else { return };
        let src = Rc::new(dir_source(&dir));
        let p = probe(src.clone()).expect("probe");
        let mut sink = MemSink::default();
        import(src, &mut sink, &mut |_| {}).expect("import");
        for (path, _) in &sink.files {
            assert!(p.outputs.contains(path), "{path} written but not planned");
        }
        assert_eq!(p.outputs.last().map(String::as_str), Some(DUMP_MANIFEST));
        assert!(p.title_id.is_some(), "no TITLE_ID");
        assert!(p.icon0.is_some(), "no icon0");
        assert!(p.bytes > 0);
    }

    /// A homebrew VPK (`VITASLOP_VPK=<file>`): probed as `vpk`, imported as files + a
    /// module the loader accepts.
    #[test]
    fn a_vpk_imports_and_its_eboot_loads() {
        let Some(path) = std::env::var_os("VITASLOP_VPK") else { return };
        let bytes = std::fs::read(&path).expect("read vpk");
        let mut files = HashMap::new();
        files.insert("app.vpk".to_string(), bytes);
        let src = Rc::new(MemSource { files });
        let p = probe(src.clone()).expect("probe");
        assert_eq!(p.kind, "vpk");
        assert!(p.zipped);
        assert!(p.title_id.is_some());
        let mut sink = MemSink::default();
        let cid = import(src, &mut sink, &mut |_| {}).expect("import");
        assert!(!cid.is_empty());
        let eboot = sink.files.iter().find(|f| f.0 == "modules/eboot.bin").expect("module");
        vitaslop_loader::load(&eboot.1).expect("the loader takes the fSELF");
        for (path, _) in &sink.files {
            assert!(p.outputs.contains(path), "{path} written but not planned");
        }
    }

    #[test]
    fn a_zip_of_a_dump_round_trips() {
        let mut files = HashMap::new();
        files.insert("d/vitaslop-dump.txt".to_string(), format!("{DUMP_MAGIC}\ncontent_id=X\nmodule=eboot.bin\n").into_bytes());
        files.insert("d/files/a.bin".to_string(), vec![1, 2, 3]);
        files.insert("d/modules/eboot.bin".to_string(), b"\x7fELF".to_vec());
        let entries: Vec<(String, Vec<u8>)> = files.iter().map(|(k, v)| (k.clone(), v.clone())).collect();
        let zip = crate::ingest::zip::write_zip(&entries);
        let mut outer = HashMap::new();
        outer.insert("t.zip".to_string(), zip);
        let src = Rc::new(MemSource { files: outer });
        let p = probe(src.clone()).expect("probe");
        assert_eq!(p.kind, "dump");
        assert!(p.zipped);
        let mut sink = MemSink::default();
        let cid = import(src, &mut sink, &mut |_| {}).expect("import");
        assert_eq!(cid, "X");
        let names: Vec<&str> = sink.files.iter().map(|f| f.0.as_str()).collect();
        assert_eq!(names.last(), Some(&DUMP_MANIFEST));
        assert!(names.contains(&"files/a.bin"));
        assert!(names.contains(&"modules/eboot.bin"));
    }
}
