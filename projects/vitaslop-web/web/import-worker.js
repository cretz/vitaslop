// The import: picked files in, a decrypted title in OPFS out, never resident.
//
// Two reasons this is a worker. `FileReaderSync` exists only in workers, and it is
// what lets the Rust streaming ingest pull any byte range of a 3 GB `File` with a
// plain synchronous call. And OPFS synchronous access handles - the only way to
// write a file a chunk at a time without an await between chunks - exist only in
// workers too. So the whole peel (zip, pkg, PFS, SELF) runs here as ONE synchronous
// call into wasm, pulling from the files and pushing into storage, with progress
// posted out as it goes (postMessage from inside a long synchronous call is fine:
// the messages queue on the page's event loop).
//
// ONE JOB, NOT TWO. Identifying the title (`ingest_probe`) and importing it are one
// message: the same worker, the same wasm instance, the same probe. The page used to
// probe in a worker of its own, show a confirmation, then spawn a second worker that
// booted wasm and probed AGAIN before writing a byte. On a phone that first probe is
// a screen that says "Reading" and then sits there - every read of a picked file is a
// round trip to the platform's file provider, and none of them moved a progress bar.
// So the probe now reports the bytes it reads as it reads them, and its result is
// posted to the page mid-job (`{type:"probe"}`) to fill the screen in while the
// import it is already doing continues.
//
// The one wrinkle: a sync access handle can only be OBTAINED asynchronously, and the
// Rust side's `begin(path)` is called from inside the synchronous import. So the
// probe reports every path the import may write (`outputs`), all of them are opened
// here beforehand, and the sink hands them out by name. A path that was planned but
// never begun (a module-named file that was not a SELF) is closed and deleted after.
//
// Messages in:  { type: "import", files: [{ path, file }] }
//               { type: "probe",  files: [{ path, file }] }   (identify only)
// Messages out: { type: "probe", probe } | { type: "progress", stage, file, done, total }
//               { type: "done", titleId, contentId, count } | { type: "error", message }
//               { type: "panic", message }

import init, { ingest_probe, ingest_import } from "./pkg/vitaslop_web.js";
import { titleDir, encodeName, storageRoom } from "./opfs.js";
import { makeFileSource } from "./read-window.js";

globalThis.__vitaslopPanic = (text) => {
  try {
    self.postMessage({ type: "panic", message: text });
  } catch {}
};

const ready = init();

/// The Rust side's ByteSource over the picked files. `onRead(bytes, path)` is called
/// for every range pulled from the provider, which is the only sign of life the
/// identify phase has - and it counts PROVIDER bytes, not bytes served out of the
/// cache, so it stays a measure of real progress.
///
/// The caching, and why a picked file needs any, is in `read-window.js`.
function fileSource(files, onRead = () => {}) {
  return makeFileSource(files, { onRead });
}

/// Throttled progress out. `postMessage` is cheap but the page has to render each one.
/// The FIRST report always goes out: it is the one that says the job is alive, and a
/// short job (identifying a small source on a fast machine) would otherwise finish
/// inside the first window having reported nothing at all.
function throttle(ms) {
  let last = 0;
  return (force, make) => {
    const now = performance.now();
    if (!force && last && now - last < ms) return;
    last = now;
    self.postMessage(make());
  };
}

/// Run `job` over every item with a bounded number in flight. Bounded because these
/// are storage calls: unbounded would put 1,300 OPFS opens in flight at once, and the
/// point is to overlap their latency, not to find the limit of the implementation.
async function inParallel(items, job, width = 16) {
  let next = 0;
  const worker = async () => {
    while (next < items.length) await job(items[next++]);
  };
  await Promise.all(Array.from({ length: Math.min(width, items.length) }, worker));
}

/// The probe as the page needs it: `outputs` is thousands of paths the page has no
/// use for, and `icon0`/`pic0` are already JS-owned copies.
function forPage(p) {
  return {
    kind: p.kind,
    zipped: p.zipped,
    titleId: p.titleId,
    title: p.title,
    contentId: p.contentId,
    appVersion: p.appVersion,
    bytes: p.bytes,
    files: p.files,
    icon0: p.icon0,
    pic0: p.pic0,
    missingWorkBin: p.missingWorkBin,
  };
}

self.onmessage = async (e) => {
  const d = e.data;
  try {
    await ready;
    const post = throttle(100);

    // Identify only (the page's "what is this" path; the import does its own).
    if (d.type === "probe") {
      let read = 0;
      const src = fileSource(d.files, (n) => {
        read += n;
        post(false, () => ({ type: "progress", stage: "reading", file: "", done: read, total: 0 }));
      });
      self.postMessage({ type: "probe", probe: forPage(ingest_probe(src)) });
      return;
    }
    if (d.type !== "import") throw new Error(`unknown message ${d.type}`);

    // ---- identify, reporting the bytes it reads ----
    // The same source serves the import that follows, and the import has a progress
    // report of its own - so the read heartbeat stops the moment the probe is done, or
    // it would talk over it.
    let read = 0;
    let identifying = true;
    const src = fileSource(d.files, (n, path) => {
      if (!identifying) return;
      read += n;
      post(false, () => ({ type: "progress", stage: "reading", file: path, done: read, total: 0 }));
    });
    const probe = ingest_probe(src);
    identifying = false;
    self.postMessage({ type: "probe", probe: forPage(probe) });
    const titleId = probe.titleId;
    if (probe.missingWorkBin) {
      throw new Error("this package needs the work.bin licence that was made for it - pick both files together");
    }
    if (!titleId) throw new Error("no param.sfo was found in these files, so this title cannot be named or stored");
    const outputs = probe.outputs || [];
    if (outputs.length === 0) throw new Error("nothing to import from these files");

    const room = await storageRoom();
    if (room && room.free < probe.bytes * 1.05) {
      throw new Error(
        `this title needs about ${(probe.bytes / 1e6) | 0} MB but this browser will only give this site ` +
          `${(room.free / 1e6) | 0} MB more (quota ${(room.quota / 1e6) | 0} MB, ${(room.usage / 1e6) | 0} MB used). ` +
          `Free up space on the device, or remove a title.`
      );
    }

    // ---- open every output the import may write ----
    self.postMessage({ type: "progress", stage: "preparing", file: "", done: 0, total: 0 });
    const dir = await titleDir(titleId, { create: true });
    // A previous partial import leaves files behind; start from an empty directory
    // so the only entries afterwards are this import's.
    const stale = [];
    for await (const [name] of dir.entries()) stale.push(name);
    await inParallel(stale, (name) => dir.removeEntry(name, { recursive: true }).catch(() => {}));
    // EVERY OUTPUT'S HANDLE, OPENED BEFORE THE FIRST BYTE - and on a phone one open
    // costs 6.5 ms, so a 1,300 file title sat in "preparing storage" for 8.4 seconds
    // doing them one after another. They do not depend on each other, so they go out
    // in parallel, and the stage reports its own progress rather than showing a
    // stopped bar for the whole of it.
    const handles = new Map();
    let opened = 0;
    await inParallel(outputs, async (path) => {
      const fh = await dir.getFileHandle(encodeName(path), { create: true });
      handles.set(path, await fh.createSyncAccessHandle());
      opened += 1;
      // `unit: "files"` because these are handles, not bytes: it keeps the counter out
      // of the byte-rate window and off the "MB/s" line.
      post(false, () => ({
        type: "progress",
        stage: "preparing",
        file: "",
        done: opened,
        total: outputs.length,
        unit: "files",
      }));
    });

    let cur = null;
    let count = 0;
    const sink = {
      begin(path) {
        const h = handles.get(path);
        if (!h) throw new Error(`no handle prepared for ${path}`);
        cur = { path, h, off: 0 };
      },
      write(bytes) {
        cur.off += cur.h.write(bytes, { at: cur.off });
      },
      finish() {
        cur.h.truncate(cur.off);
        cur.h.flush();
        cur.h.close();
        handles.delete(cur.path);
        count += 1;
        cur = null;
      },
    };

    const contentId = ingest_import(src, sink, (stage, file, done, total) => {
      post(done >= total, () => ({ type: "progress", stage, file, done, total }));
    });

    // Planned but never produced.
    for (const [path, h] of handles) {
      try {
        h.close();
      } catch {}
      try {
        await dir.removeEntry(encodeName(path));
      } catch {}
    }
    // The marker opfs.js checks before a run: written LAST.
    const mh = await dir.getFileHandle("vitaslop-opfs-manifest.json", { create: true });
    const w = await mh.createWritable();
    await w.write(JSON.stringify({ count, complete: true }));
    await w.close();
    self.postMessage({ type: "done", titleId, contentId, count });
  } catch (err) {
    self.postMessage({ type: "error", message: String((err && err.message) || err) });
  }
};
