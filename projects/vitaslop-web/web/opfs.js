// The browser's game storage: the Origin Private File System.
//
// # Why the container cannot just be bytes in memory
// A retail Vita title is over a gigabyte. Held as JS `ArrayBuffer`s it costs that much
// in the renderer, and copied into the emulator's wasm heap it costs it again - against
// a wasm32 address space that tops out at 4 GB. Measured on a 1719 MB title before this
// existed: Chrome peaked at 8.01 GB during ingest and the worker was killed mid-boot,
// which from outside looks like a page that has stopped answering rather than an
// out-of-memory error.
//
// So the container is never fully resident. It is STREAMED into OPFS - from the user's
// picked `.pkg` + `work.bin`, or from this origin for a test run - and read back in
// pieces. Peak memory during import is one chunk, not one title.
//
// # Why OPFS specifically, and not Cache Storage or IndexedDB
// Only OPFS offers `createSyncAccessHandle`, whose `read` is SYNCHRONOUS. That is not a
// convenience here, it is the requirement: the emulator serves guest file reads inside
// host calls, which are synchronous Rust running on a suspended guest stack. An async
// storage API could not back them without turning every guest read into a suspension.
// Sync access handles are available in Workers only - which is where the emulator runs.

/// Directory under the OPFS root holding one title's files, keyed by `id`.
export async function titleDir(id, { create = true } = {}) {
  const root = await navigator.storage.getDirectory();
  const games = await root.getDirectoryHandle("games", { create });
  return games.getDirectoryHandle(id, { create });
}

/// A file name safe to use as a flat OPFS entry, since OPFS has no path separators of
/// its own and a container's paths do. Reversed by `decodeName`.
export function encodeName(path) {
  return path.replace(/\//g, "%2F");
}

export function decodeName(name) {
  return name.replace(/%2F/g, "/");
}

/// Marker written LAST, after every file is durably stored. Its presence is what makes
/// an import complete: a run interrupted halfway leaves files behind but no marker, and
/// the next boot re-imports rather than mounting a truncated title. Storing the file
/// count too means a partial directory that somehow kept its marker is still caught.
const MANIFEST = "vitaslop-opfs-manifest.json";

/// Whether `id` has a complete import, whatever its size. The marker is written last.
export async function isComplete(id) {
  try {
    const dir = await titleDir(id, { create: false });
    const fh = await dir.getFileHandle(MANIFEST);
    const m = JSON.parse(await (await fh.getFile()).text());
    return m.complete === true;
  } catch {
    return false;
  }
}

/// Remove every stored byte of `id`. The saves are elsewhere and untouched.
export async function removeTitle(id) {
  try {
    const root = await navigator.storage.getDirectory();
    const games = await root.getDirectoryHandle("games", { create: false });
    await games.removeEntry(id, { recursive: true });
  } catch {}
}

/// Whether `id` is already imported and complete. Cheap: one read of the marker.
export async function isImported(id, expectedCount) {
  try {
    const dir = await titleDir(id, { create: false });
    const fh = await dir.getFileHandle(MANIFEST);
    const m = JSON.parse(await (await fh.getFile()).text());
    return m.count === expectedCount && m.complete === true;
  } catch {
    return false;
  }
}

/// Stream one `ReadableStream` (or `Blob`) into `dir` under `path`, without ever
/// holding the whole thing. Returns the bytes written.
///
/// `onBytes(n)` is called with each chunk's length as it passes through. It exists because
/// a container's bytes are NOT spread evenly over its files: one title puts 660 of its 688
/// MB into 28 of its 248 files, so a caller that only learns about a finished file shows a
/// frozen counter for the entire minutes-long download of a single large one - which reads
/// exactly like a hang, and was reported as one.
async function storeOne(dir, path, source, onBytes = () => {}) {
  const fh = await dir.getFileHandle(encodeName(path), { create: true });
  const w = await fh.createWritable();
  const stream = source instanceof Blob ? source.stream() : source;
  const counted = stream.pipeThrough(
    new TransformStream({
      transform(chunk, ctrl) {
        onBytes(chunk.byteLength);
        ctrl.enqueue(chunk);
      },
    })
  );
  // `pipeTo` hands the writable ownership of the stream and closes it on completion,
  // so a failure part-way cannot leave a half-written file locked open.
  await counted.pipeTo(w);
  return (await fh.getFile()).size;
}

/// How much room this origin has, or `null` where the browser will not say.
///
/// # Why this is checked BEFORE an import and not caught after
/// A phone's quota for an origin is a fraction of its free disk, and these containers are
/// 237 MB to 1719 MB. Discovering the limit by hitting it means a `QuotaExceededError`
/// hundreds of megabytes into a download, over wifi, with the partial files still on disk.
/// Asking first costs one call and turns that into a sentence naming both numbers.
export async function storageRoom() {
  if (!navigator.storage || !navigator.storage.estimate) return null;
  const { quota, usage } = await navigator.storage.estimate();
  if (typeof quota !== "number") return null;
  return { quota, usage: usage || 0, free: quota - (usage || 0) };
}

/// Ask the browser to treat this origin's storage as persistent, so an imported title is not
/// evicted under disk pressure - re-importing a gigabyte is not a cost to pay silently.
/// Returns whether it was granted. A refusal is not fatal and is reported, not swallowed.
export async function requestPersistence() {
  if (!navigator.storage || !navigator.storage.persist) return false;
  try {
    return (await navigator.storage.persisted()) || (await navigator.storage.persist());
  } catch {
    return false;
  }
}

/// Import a whole container into OPFS from a list of `{ path, source }`, where `source`
/// is a `Blob`/`File` (the web form) or a `ReadableStream` (a fetch body). Calls
/// `onProgress(done, total, bytes)` as it goes. Idempotent: an already-complete title
/// is left alone.
export async function importTitle(id, entries, onProgress = () => {}) {
  if (await isImported(id, entries.length)) {
    onProgress(entries.length, entries.length, 0, true);
    return { reused: true };
  }
  const dir = await titleDir(id);
  let bytes = 0;
  for (let i = 0; i < entries.length; i++) {
    // Report DURING each file as well as after it, throttled to every 4 MB so a slow large
    // file still moves the counter without flooding the caller. See `storeOne`.
    let inFlight = 0;
    let reported = 0;
    const written = await storeOne(dir, entries[i].path, await entries[i].source(), (n) => {
      inFlight += n;
      if (inFlight - reported >= 4 << 20) {
        reported = inFlight;
        onProgress(i, entries.length, bytes + inFlight, false);
      }
    });
    bytes += written;
    onProgress(i + 1, entries.length, bytes, false);
  }
  // The marker goes last, so "present" always means "everything before it is there".
  const fh = await dir.getFileHandle(MANIFEST, { create: true });
  const w = await fh.createWritable();
  await w.write(JSON.stringify({ complete: true, count: entries.length, bytes }));
  await w.close();
  return { reused: false, bytes };
}

/// Open every stored file of `id` as a SYNCHRONOUS access handle, returned as
/// `{ path: handle }`. Worker-only. This is the async step that buys synchronous reads
/// later - the emulator cannot await inside a guest file read, so every handle it may
/// need has to be open before the guest starts.
export async function openTitleSync(id) {
  // >>> A RELOAD RACES THE PREVIOUS WORKER'S TEARDOWN, AND THAT IS THE NORMAL PATH.
  //
  // A sync access handle is EXCLUSIVE per file. The run worker holds one per file for the
  // whole session, and when the page is reloaded the browser tears that worker down
  // ASYNCHRONOUSLY - while the new page's transpile worker is already starting. The handles
  // are still held for a moment, and the open fails with
  // `NoModificationAllowedError: Access Handles cannot be created if there is another open
  // Access Handle or Writable stream associated with the same file`.
  //
  // Reported by the user on a plain reload of a single tab, with no second tab anywhere:
  // BOOT FAILED, no adapter, no audio, nothing playable. So this is not a multi-tab edge
  // case, it is what reloading a running title does.
  //
  // >>> AND THE FAILURE USED TO POISON EVERY RETRY. If the throw came part-way through the
  // loop, the handles already opened were dropped on the floor still OPEN - held by this
  // worker, for as long as it lives. A second attempt then collided with itself, so the
  // condition could never clear. Closing the partial set is what makes retrying meaningful.
  const DELAYS_MS = [0, 100, 200, 400, 800, 1500];
  let last;
  for (const wait of DELAYS_MS) {
    if (wait) await new Promise((r) => setTimeout(r, wait));
    const dir = await titleDir(id, { create: false });
    const handles = {};
    try {
      for await (const [name, h] of dir.entries()) {
        if (name === MANIFEST || h.kind !== "file") continue;
        handles[decodeName(name)] = await h.createSyncAccessHandle();
      }
      return handles;
    } catch (e) {
      for (const h of Object.values(handles)) {
        try {
          h.close();
        } catch {
          // Already closed, or the handle died with its file. Nothing to do, and it must not
          // mask the original failure below.
        }
      }
      last = e;
      if (e && e.name !== "NoModificationAllowedError") throw e;
    }
  }
  throw new Error(
    `this title's files are still open in another worker, so they cannot be opened again ` +
      `(${last}). That is usually the previous run still shutting down - reloading again ` +
      `after a moment normally clears it. If it does not, close every tab on this address ` +
      `(the handles are released when the last one goes) and open it again.`,
  );
}

/// A reader the wasm side calls: `size(path)` and `read(path, offset, into)`, both
/// synchronous. `into` is a `Uint8Array` view the caller owns; the return is how many
/// bytes were actually read (short at end of file, 0 for an unknown path).
export function syncReader(handles) {
  return {
    paths: () => Object.keys(handles),
    size: (path) => (handles[path] ? handles[path].getSize() : -1),
    read: (path, offset, into) => {
      const h = handles[path];
      return h ? h.read(into, { at: offset }) : 0;
    },
    close: () => {
      for (const h of Object.values(handles)) h.close();
    },
  };
}

// >>> THE RUN'S READER: A PAGE RING FILLED BY A SEPARATE STORAGE WORKER.
//
// `syncReader` above makes every read a synchronous trip through the browser's storage
// stack on the emulator's own thread - about a millisecond per 64 KB where it was measured,
// several times a frame while a title streams. `openTitleCached` moves the handles to
// `storage-worker.js`, which serves 64 KB pages into this ring and reads AHEAD along the
// file between requests, so a sequential stream is answered out of shared memory and the
// emulator waits only on a genuine miss. See the header of storage-worker.js.
//
// Layout of the `SharedArrayBuffer`, as `Int32Array` indices: a small header, one tag per
// slot, then - at `DATA_OFF` bytes - `SLOTS` pages of `PAGE` bytes.
export const PAGE = 64 * 1024;
export const SLOTS = 128;
export const H = {
  REQ: 0, // 1 while a request is posted
  ACK: 1, // count of requests completed
  CLOSE: 2, // 1 asks the worker to close its handles and exit
  ERR: 3, // nonzero once the worker has failed a read; the reader then reports rather than waits
  REQ_PATH: 4,
  REQ_PAGE: 5,
  REQ_COUNT: 6,
  TAGS: 8,
};
export const TAG = { PATH: 0, PAGE: 1, STATE: 2, LEN: 3, STRIDE: 4 };
export const STATE = { READY: 0, FILLING: 1, PINNED: 2 };
export const DATA_OFF = PAGE; // the header and tags fit in the first page with room to spare
/// Pages one request may ask for. A larger read loops; the ring must hold a request plus
/// the worker's read-ahead without lapping itself.
const MAX_REQ = 32;

/// Open `id`'s files on a storage worker and return a reader with the SAME interface as
/// `syncReader` - `paths()`, `size(path)`, `read(path, offset, into)`, `close()` - whose
/// reads come out of the shared page ring. Worker-only, like `syncReader`.
export async function openTitleCached(id) {
  const sab = new SharedArrayBuffer(DATA_OFF + SLOTS * PAGE);
  const h = new Int32Array(sab);
  const data = new Uint8Array(sab, DATA_OFF, SLOTS * PAGE);
  const worker = new Worker("./storage-worker.js", { type: "module" });
  const ready = await new Promise((resolve, reject) => {
    worker.onmessage = (e) => {
      if (e.data.type === "ready") resolve(e.data);
      else if (e.data.type === "error") reject(new Error(e.data.message));
    };
    worker.onerror = (e) => reject(new Error(`storage worker failed: ${e.message || e}`));
    worker.postMessage({ id, sab });
  });
  // From here the worker reports only failures; a read error is also flagged in `H.ERR`,
  // which is what the synchronous reader can see.
  let workerError = null;
  worker.onmessage = (e) => {
    if (e.data.type === "error") {
      workerError = e.data.message;
      console.error(`[storage] ${e.data.message}`);
    }
  };
  const { paths, sizes } = ready;
  const ids = new Map(paths.map((p, i) => [p, i]));

  const tag = (slot, f) => h[H.TAGS + slot * TAG.STRIDE + f];
  /// Pin the slot holding (pid, page): READY -> PINNED, then re-check the tag, since the
  /// worker may have refilled the slot between the scan and the exchange.
  const pin = (pid, page) => {
    for (let s = 0; s < SLOTS; s++) {
      if (tag(s, TAG.PATH) !== pid + 1 || tag(s, TAG.PAGE) !== page) continue;
      const st = H.TAGS + s * TAG.STRIDE + TAG.STATE;
      if (Atomics.compareExchange(h, st, STATE.READY, STATE.PINNED) !== STATE.READY) continue;
      if (tag(s, TAG.PATH) === pid + 1 && tag(s, TAG.PAGE) === page) return s;
      Atomics.store(h, st, STATE.READY);
    }
    return -1;
  };
  const unpin = (s) => Atomics.store(h, H.TAGS + s * TAG.STRIDE + TAG.STATE, STATE.READY);
  // >>> WHETHER THE READ-AHEAD IS ACTUALLY SERVING ANYTHING.
  //
  // The panel's STORAGE line counts the emulator's read CALLS, and once those calls come out
  // of this ring the count says nothing about what they cost: a hit is a memcpy out of shared
  // memory, a miss is a round trip to another thread with an `Atomics.wait` in the middle, and
  // the line shows the same number either way. A device dump was read with the storage worker
  // live and could not say whether the 16-page read-ahead was working at all, which is the
  // whole claim the worker exists to make. A miss already pays for a thread hop, so timing
  // only the misses adds nothing measurable to the hit path.
  let ringHits = 0;
  let ringMisses = 0;
  let ringWaitMs = 0;
  const request = (pid, page, count) => {
    const t0 = performance.now();
    Atomics.store(h, H.REQ_PATH, pid);
    Atomics.store(h, H.REQ_PAGE, page);
    Atomics.store(h, H.REQ_COUNT, count);
    const seq = Atomics.load(h, H.ACK);
    Atomics.store(h, H.REQ, 1);
    Atomics.notify(h, H.REQ);
    // Bounded waits, so a worker that has died leaves an error rather than a hang.
    let waited = 0;
    while (Atomics.load(h, H.ACK) === seq) {
      if (Atomics.load(h, H.ERR) !== 0 || workerError !== null) {
        throw new Error(`storage worker failed: ${workerError || "read error"}`);
      }
      if (Atomics.wait(h, H.ACK, seq, 1000) === "timed-out" && ++waited >= 30) {
        throw new Error(`storage worker did not answer a read of ${paths[pid]} page ${page} in 30 s`);
      }
    }
    ringWaitMs += performance.now() - t0;
  };

  return {
    paths: () => paths.slice(),
    size: (path) => (ids.has(path) ? sizes[ids.get(path)] : -1),
    read: (path, offset, into) => {
      const pid = ids.get(path);
      if (pid === undefined) return 0;
      const size = sizes[pid];
      let done = 0;
      while (done < into.length) {
        const at = offset + done;
        if (at >= size) break;
        const page = Math.floor(at / PAGE);
        let s = pin(pid, page);
        if (s >= 0) {
          ringHits += 1;
        } else {
          ringMisses += 1;
          const count = Math.min(MAX_REQ, Math.ceil((into.length - done + (at % PAGE)) / PAGE));
          request(pid, page, count);
          s = pin(pid, page);
          if (s < 0) throw new Error(`storage ring did not hold ${path} page ${page} after a request`);
        }
        const from = at - page * PAGE;
        const n = Math.min(into.length - done, tag(s, TAG.LEN) - from);
        if (n <= 0) {
          unpin(s);
          break;
        }
        into.set(data.subarray(s * PAGE + from, s * PAGE + from + n), done);
        unpin(s);
        done += n;
      }
      return done;
    },
    // Read once per panel window, never per read - see ringHits above.
    stats: () => ({ hits: ringHits, misses: ringMisses, waitMs: ringWaitMs }),
    close: () => {
      Atomics.store(h, H.CLOSE, 1);
      Atomics.store(h, H.REQ, 1);
      Atomics.notify(h, H.REQ);
      setTimeout(() => worker.terminate(), 2000);
    },
  };
}
