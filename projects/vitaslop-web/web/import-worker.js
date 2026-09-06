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

globalThis.__vitaslopPanic = (text) => {
  try {
    self.postMessage({ type: "panic", message: text });
  } catch {}
};

const ready = init();

/// The Rust side's ByteSource over the picked files. `onRead(bytes, path)` is called
/// for every range pulled, which is the only sign of life the identify phase has.
function fileSource(files, onRead = () => {}) {
  const byPath = new Map(files.map((f) => [f.path, f.file]));
  const reader = new FileReaderSync();
  return {
    list: () => [...byPath.keys()],
    size: (path) => {
      const f = byPath.get(path);
      return f ? f.size : undefined;
    },
    readAt: (path, off, buf) => {
      const f = byPath.get(path);
      if (!f || off >= f.size) return 0;
      const end = Math.min(f.size, off + buf.length);
      const bytes = new Uint8Array(reader.readAsArrayBuffer(f.slice(off, end)));
      buf.set(bytes);
      onRead(bytes.length, path);
      return bytes.length;
    },
  };
}

/// Throttled progress out. `postMessage` is cheap but the page has to render each one.
function throttle(ms) {
  let last = 0;
  return (force, make) => {
    const now = performance.now();
    if (!force && now - last < ms) return;
    last = now;
    self.postMessage(make());
  };
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
    for await (const [name] of dir.entries()) await dir.removeEntry(name, { recursive: true });
    const handles = new Map();
    for (const path of outputs) {
      const fh = await dir.getFileHandle(encodeName(path), { create: true });
      handles.set(path, await fh.createSyncAccessHandle());
    }

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
