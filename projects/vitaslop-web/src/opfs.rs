//! The browser's game storage, seen from Rust: a [`FileBacking`] and a [`Vfs`] over the
//! Origin Private File System.
//!
//! # Why the title is not simply loaded
//! A retail Vita container is over a gigabyte. Loading it costs that much in the wasm
//! heap - which tops out at 4 GB for the whole emulator - and the JS side that fetched it
//! holds another copy until it is released. Measured on a 1719 MB title before this
//! existed: Chrome peaked at 8.01 GB during ingest and the worker was killed mid-boot,
//! with no error anywhere. The page simply stopped answering.
//!
//! So the title lives in OPFS and is read in pieces. What crosses into the wasm heap is
//! the few megabytes of loadable modules, and afterwards only the bytes a guest actually
//! asks for.
//!
//! # Synchronous by necessity
//! Every read here is synchronous, because a guest file read happens inside a host call,
//! on a suspended guest stack that cannot await. `FileSystemSyncAccessHandle` is the only
//! browser storage primitive that offers that, and it exists only in Workers - which is
//! where the emulator runs. The handles are opened asynchronously during setup (see
//! `web/opfs.js`); by the time the guest starts, every read is a plain call.

use js_sys::{Function, Object, Reflect, Uint8Array};
use vitaslop_runtime::host::{vfs_key, FileBacking};
use vitaslop_runtime::ingest::vfs::Vfs;
use vitaslop_runtime::ingest::Error;
use wasm_bindgen::prelude::*;

/// The `syncReader` object from `web/opfs.js`: `paths()`, `size(path)`,
/// `read(path, offset, into)`, all synchronous.
pub struct OpfsReader {
    paths: Function,
    size: Function,
    read: Function,
    /// `stats()` if the reader has a ring behind it - see [`OpfsReader::ring_stats`].
    stats: Option<Function>,
    this: Object,
}

// SAFETY: `OpfsReader` holds JS values, which wasm-bindgen marks `!Send` because they
// live in a per-thread heap that cannot be reached from another thread.
//
// This target has no other thread. `wasm32-unknown-unknown` without the atomics feature
// is single-threaded by construction: there is no `std::thread::spawn`, and the emulator's
// own concurrency is guest threads multiplexed onto ONE worker by the JSPI scheduler
// (`browser_sched`), which is why its host is an `Arc<Mutex<VitaEnv>>` whose lock never
// contends. So no `OpfsReader` can be observed from a second thread, and the assertion
// costs nothing that could be lost.
//
// It is asserted here, on the one type that needs it, rather than by dropping the `Send`
// bound from `FileBacking` - because that bound IS load-bearing natively, where the
// preemptive scheduler really does move the host across OS threads.
unsafe impl Send for OpfsReader {}

impl OpfsReader {
    /// Wrap the reader object the worker passes in. Fails loudly on a missing method
    /// rather than degrading to empty reads: a title whose files silently read as zero
    /// bytes does not report an error, it misbehaves thousands of frames later.
    pub fn new(obj: JsValue) -> Result<Self, JsValue> {
        let this: Object = obj.dyn_into()?;
        let get = |name: &str| -> Result<Function, JsValue> {
            Reflect::get(&this, &JsValue::from_str(name))?
                .dyn_into::<Function>()
                .map_err(|_| JsValue::from_str(&format!("OPFS reader has no {name}()")))
        };
        Ok(OpfsReader {
            paths: get("paths")?,
            size: get("size")?,
            read: get("read")?,
            // OPTIONAL, unlike the three above: the in-memory fixture reader has no ring
            // behind it and nothing to report, and an absent counter is a reader without a
            // ring rather than a failure. The three that ARE required stay required - a
            // missing `read` is a title that silently reads zeros.
            stats: {
                let f = Reflect::get(&this, &JsValue::from_str("stats"))
                    .ok()
                    .and_then(|f| f.dyn_into::<Function>().ok());
                // Also parked where the PANEL can reach it. Everything else on this line is
                // counted into a global atomic by `read_at`; the ring's own counters live in
                // JS and cost a call to fetch, so the panel holds the handle and asks once a
                // window instead of the read path asking every time.
                if let Some(f) = f.clone() {
                    RING.with(|r| *r.borrow_mut() = Some((f, this.clone())));
                }
                f
            },
            this,
        })
    }

    /// `(ring hits, misses, ms waited on a miss)` from the storage worker's page ring, or
    /// `None` from a reader that has no ring.
    ///
    /// # Why the read COUNT could not answer this
    /// Once the emulator's reads come out of the ring, the panel's read count says nothing
    /// about what they cost: a hit is a `copy_from_slice` out of shared memory, a miss is a
    /// round trip to another thread with an `Atomics.wait` in the middle, and the count is the
    /// same number either way. A device dump was read with the storage worker live and could
    /// not say whether its 16-page read-ahead was serving anything at all - which is the entire
    /// claim the worker exists to make. Called once per panel window, never per read.
    pub fn ring_stats(&self) -> Option<(u64, u64, f64)> {
        let v = self.stats.as_ref()?.call0(&self.this).ok()?;
        let num = |k: &str| Reflect::get(&v, &JsValue::from_str(k)).ok()?.as_f64();
        Some((num("hits")? as u64, num("misses")? as u64, num("waitMs")?))
    }

    /// Every stored path.
    pub fn paths(&self) -> Vec<String> {
        let Ok(v) = self.paths.call0(&self.this) else { return Vec::new() };
        js_sys::Array::from(&v).iter().filter_map(|p| p.as_string()).collect()
    }

    /// Byte length of `path`, or `None` when it is not stored (the JS side reports -1).
    pub fn size(&self, path: &str) -> Option<usize> {
        let n = self
            .size
            .call1(&self.this, &JsValue::from_str(path))
            .ok()
            .and_then(|v| v.as_f64())?;
        if n < 0.0 {
            None
        } else {
            Some(n as usize)
        }
    }

    /// Read into `buf` at `off`, returning the count actually read.
    pub fn read_at(&self, path: &str, off: usize, buf: &mut [u8]) -> usize {
        // Build the other arguments FIRST. A `Uint8Array` view points into the wasm heap,
        // and anything that grows that heap between the view's creation and its use
        // detaches it - after which the read silently fills nothing and the guest gets
        // zeros. `JsValue::from_str` copies a string out of wasm memory and is exactly the
        // kind of call that can allocate, so it must not happen while the view is live.
        let path = JsValue::from_str(path);
        let off_js = JsValue::from_f64(off as f64);
        // A view over this call's slice, handed to JS to fill in place - no intermediate
        // JS array, no copy. From here to the call there is no allocation at all.
        let view = unsafe { Uint8Array::view_mut_raw(buf.as_mut_ptr(), buf.len()) };
        let n = self
            .read
            .call3(&self.this, &path, &off_js, &view)
            .ok()
            .and_then(|v| v.as_f64())
            .unwrap_or(0.0) as usize;
        OPFS_READS.fetch_add(1, std::sync::atomic::Ordering::Relaxed);
        OPFS_READ_BYTES.fetch_add(n as u64, std::sync::atomic::Ordering::Relaxed);
        n
    }
}

/// Serves the guest's read-only files straight out of OPFS.
///
/// `prefix` is stripped from every stored path before it becomes a guest-visible key, so
/// a decrypted dump stored as `dumps/T/files/Disc/x.bin` is served to the guest as
/// `Disc/x.bin` without a second copy of the path list.
pub struct OpfsBacking {
    reader: OpfsReader,
    /// Guest-visible keys as STORAGE spells them, in listing order.
    keys: Vec<String>,
    /// NORMALISED key -> the full stored path. Built with the runtime's own [`vfs_key`],
    /// because that is what the filesystem will hand back: a lowercased, separator-
    /// collapsed key. Mapping the received key onto storage directly reads nothing at all
    /// for any path with a capital letter in it, and fails as a guest trap far away rather
    /// than as a missing file.
    by_key: std::collections::HashMap<String, String>,
    /// The most recent read-ahead window: the stored path it belongs to, the file offset
    /// it starts at, and its bytes (short only at end of file). See [`Self::read_at`].
    window: std::cell::RefCell<ReadWindow>,
}

/// One read-ahead window over one stored file - see [`OpfsBacking::read_at`].
#[derive(Default)]
struct ReadWindow {
    path: String,
    off: usize,
    data: Vec<u8>,
}

/// OPFS reads made and bytes moved, cumulatively, and how many `read_at` calls the window
/// answered without one. The panel differences two snapshots. See [`opfs_read_counts`].
thread_local! {
    /// The live reader's `stats()` and its receiver, for [`ring_read_counts`]. A
    /// thread_local because this target has exactly one thread - see the `unsafe impl Send`
    /// on `OpfsReader` and the reasoning above it.
    static RING: std::cell::RefCell<Option<(Function, Object)>> =
        const { std::cell::RefCell::new(None) };
}

/// `(ring hits, misses, ms waited on a miss)` for the panel, or `None` before a title with a
/// storage ring is open. See [`OpfsReader::ring_stats`] for why the read count cannot say this.
pub fn ring_read_counts() -> Option<(u64, u64, f64)> {
    RING.with(|r| {
        let borrowed = r.borrow();
        let (f, this) = borrowed.as_ref()?;
        let v = f.call0(this).ok()?;
        let num = |k: &str| Reflect::get(&v, &JsValue::from_str(k)).ok()?.as_f64();
        Some((num("hits")? as u64, num("misses")? as u64, num("waitMs")?))
    })
}

static OPFS_READS: std::sync::atomic::AtomicU64 = std::sync::atomic::AtomicU64::new(0);
static OPFS_READ_BYTES: std::sync::atomic::AtomicU64 = std::sync::atomic::AtomicU64::new(0);
static OPFS_WINDOW_HITS: std::sync::atomic::AtomicU64 = std::sync::atomic::AtomicU64::new(0);

/// `(OPFS reads, bytes read, calls served from the window)` since the page loaded.
pub fn opfs_read_counts() -> (u64, u64, u64) {
    use std::sync::atomic::Ordering::Relaxed;
    (OPFS_READS.load(Relaxed), OPFS_READ_BYTES.load(Relaxed), OPFS_WINDOW_HITS.load(Relaxed))
}

/// Size of the read-ahead window. A title's own archive layer reads 64 KB blocks, and a
/// streamed video's access units are tens of KB, so one window serves a run of the
/// small sequential reads either produces.
const READ_WINDOW: usize = 64 * 1024;

impl OpfsBacking {
    pub fn new(reader: OpfsReader, prefix: impl Into<String>) -> Self {
        let prefix = prefix.into();
        let mut keys = Vec::new();
        let mut by_key = std::collections::HashMap::new();
        for stored in reader.paths() {
            let Some(rel) = stored.strip_prefix(&prefix) else { continue };
            by_key.insert(vfs_key(rel), stored.clone());
            keys.push(rel.to_string());
        }
        OpfsBacking { reader, keys, by_key, window: std::cell::RefCell::new(ReadWindow::default()) }
    }

    /// The stored path a normalised guest key maps back to.
    fn stored(&self, key: &str) -> Option<&str> {
        self.by_key.get(key).map(String::as_str)
    }
}

impl OpfsBacking {
    /// Check that CHUNKED reads of `key` agree with a whole-file read.
    ///
    /// Whole-file reads and OFFSET reads are different paths, and only the first is
    /// exercised by setup (the loadable modules). A wrong offset path complains about
    /// nothing: the guest is handed the wrong bytes and traps deep in its own code tens
    /// of thousands of host calls later, with nothing pointing at storage.
    ///
    /// The chunk size is deliberately not a power of two, so an offset computed with a
    /// mask rather than an addition shows up.
    fn verify_one(&self, key: &str) -> Result<(), String> {
        const CHUNK: usize = 997;
        let n = self.len(key).ok_or_else(|| format!("OPFS has no length for {key:?}"))?;
        let mut whole = vec![0u8; n];
        let got = self.read_at(key, 0, &mut whole);
        if got != n {
            return Err(format!("OPFS whole read of {key:?} gave {got} of {n} bytes"));
        }
        let mut chunked = vec![0u8; n];
        let mut off = 0usize;
        while off < n {
            let end = (off + CHUNK).min(n);
            let got = self.read_at(key, off, &mut chunked[off..end]);
            if got != end - off {
                return Err(format!(
                    "OPFS read of {key:?} at offset {off} gave {got} of {} bytes",
                    end - off
                ));
            }
            off = end;
        }
        match (0..n).find(|&i| whole[i] != chunked[i]) {
            None => Ok(()),
            Some(i) => Err(format!(
                "OPFS chunked read of {key:?} disagrees with the whole-file read at byte {i} \
                 ({:#04x} vs {:#04x}) - offset reads are not landing where they are asked to",
                whole[i], chunked[i]
            )),
        }
    }

    /// Run [`verify_one`](Self::verify_one) over every key this backing serves.
    /// Used by the browser e2e suite against a small fixture.
    ///
    /// Verifies through the NORMALISED key, not the stored spelling, because that is what
    /// the filesystem will actually ask with. Checking the stored spelling instead is what
    /// let a case-mapping bug pass: every read the guest made returned zero bytes while
    /// this reported agreement.
    pub fn verify_all(&self) -> Result<String, String> {
        let keys = self.keys();
        if keys.is_empty() {
            return Err("OPFS backing serves no keys at all - nothing was verified".into());
        }
        let mut bytes = 0usize;
        for stored_spelling in &keys {
            let key = vfs_key(stored_spelling);
            let n = self.len(&key).ok_or_else(|| {
                format!(
                    "OPFS serves {stored_spelling:?} but has no length for its normalised key \
                     {key:?} - the guest will open this file and read nothing"
                )
            })?;
            self.verify_one(&key)?;
            bytes += n;
        }
        Ok(format!("{} files, {bytes} bytes: whole and chunked reads agree", keys.len()))
    }
}

impl FileBacking for OpfsBacking {
    fn len(&self, key: &str) -> Option<usize> {
        self.reader.size(self.stored(key)?)
    }

    /// Serve a read out of the read-ahead window when it can, and refill the window from
    /// OPFS when it cannot.
    ///
    /// # Why a window
    /// Every OPFS read is a boundary crossing - a JS string for the path, a typed-array
    /// view, the call, the sync handle's own read - and the engine is billed in crossings,
    /// not bytes. MEASURED in a V8 worker profile of a retail race: the sync `read` alone
    /// was 2.7% of the whole thread while the title streamed its own video through its own
    /// file layer, which reads a stream in small consecutive pieces. One 64 KB window turns
    /// a run of those into one crossing and a `copy_from_slice` each.
    ///
    /// A read at least a window long goes straight through: buffering it would copy the
    /// bytes twice for nothing. The backing serves the title's read-only files (a save
    /// lives in the resident table and the game-data tree, never here), so a window can
    /// never go stale.
    fn read_at(&self, key: &str, off: usize, buf: &mut [u8]) -> usize {
        let Some(path) = self.stored(key) else { return 0 };
        if buf.len() >= READ_WINDOW {
            return self.reader.read_at(path, off, buf);
        }
        let mut w = self.window.borrow_mut();
        let mut done = 0usize;
        let mut refilled = false;
        // A request can straddle two windows: serve what the current one holds, then
        // refill at the next boundary and carry on, so a mid-file read is never short.
        while done < buf.len() {
            let at = off + done;
            let in_window = w.path == path && at >= w.off && at < w.off + w.data.len();
            if !in_window {
                // Outside the window - or inside its span but past its bytes, which is
                // the end of the file only when the window was already short.
                let short_eof = w.path == path
                    && at >= w.off
                    && at >= w.off + w.data.len()
                    && w.data.len() < READ_WINDOW;
                if short_eof {
                    break;
                }
                let start = at - (at % READ_WINDOW);
                w.data.resize(READ_WINDOW, 0);
                let got = self.reader.read_at(path, start, &mut w.data);
                refilled = true;
                w.data.truncate(got);
                w.off = start;
                w.path.clear();
                w.path.push_str(path);
                if at >= start + got {
                    break;
                }
            }
            let from = at - w.off;
            let n = (buf.len() - done).min(w.data.len() - from);
            buf[done..done + n].copy_from_slice(&w.data[from..from + n]);
            done += n;
        }
        if !refilled {
            OPFS_WINDOW_HITS.fetch_add(1, std::sync::atomic::Ordering::Relaxed);
        }
        done
    }

    fn keys(&self) -> Vec<String> {
        self.keys.clone()
    }
}

/// The same storage seen as an ingest [`Vfs`], for the setup-time reads: the dump
/// manifest and the loadable modules. Whole-file reads, which is right for those - they
/// are a few megabytes and both consumers want the entire image.
pub struct OpfsVfs {
    reader: OpfsReader,
}

impl OpfsVfs {
    pub fn new(reader: OpfsReader) -> Self {
        OpfsVfs { reader }
    }
    /// Hand the reader back so the same open handles can back the guest filesystem,
    /// rather than opening a second set (a sync access handle takes an exclusive lock,
    /// so a second open of the same file would fail rather than merely cost something).
    pub fn into_reader(self) -> OpfsReader {
        self.reader
    }
}

impl Vfs for OpfsVfs {
    fn read(&self, path: &str) -> Result<Vec<u8>, Error> {
        let n = self.reader.size(path).ok_or_else(|| Error::MissingFile(path.to_string()))?;
        let mut out = vec![0u8; n];
        let got = self.reader.read_at(path, 0, &mut out);
        out.truncate(got);
        Ok(out)
    }
    fn exists(&self, path: &str) -> bool {
        self.reader.size(path).is_some()
    }
    fn list(&self) -> Vec<String> {
        self.reader.paths()
    }
}
