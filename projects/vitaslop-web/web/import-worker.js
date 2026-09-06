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
// The one wrinkle: a sync access handle can only be OBTAINED asynchronously, and the
// Rust side's `begin(path)` is called from inside the synchronous import. So the
// probe reports every path the import may write (`outputs`), all of them are opened
// here beforehand, and the sink hands them out by name. A path that was planned but
// never begun (a module-named file that was not a SELF) is closed and deleted after.
//
// Messages in:  { type: "probe",  files: [{ path, file }] }
//               { type: "import", files: [{ path, file }], titleId }
// Messages out: { type: "probe", probe } | { type: "progress", stage, file, done, total }
//               { type: "done", contentId, count } | { type: "error", message }
//               { type: "panic", message }

import init, { ingest_probe, ingest_import } from "./pkg/vitaslop_web.js";
import { titleDir, encodeName } from "./opfs.js";

globalThis.__vitaslopPanic = (text) => {
  try {
    self.postMessage({ type: "panic", message: text });
  } catch {}
};

const ready = init();

/// The Rust side's ByteSource over the picked files.
function fileSource(files) {
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
      return bytes.length;
    },
  };
}

self.onmessage = async (e) => {
  const d = e.data;
  try {
    await ready;
    if (d.type === "probe") {
      self.postMessage({ type: "probe", probe: ingest_probe(fileSource(d.files)) });
      return;
    }
    if (d.type !== "import") throw new Error(`unknown message ${d.type}`);

    const src = fileSource(d.files);
    const probe = ingest_probe(src);
    const outputs = probe.outputs || [];
    if (outputs.length === 0) throw new Error("nothing to import from these files");

    const dir = await titleDir(d.titleId, { create: true });
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

    let lastPost = 0;
    const contentId = ingest_import(src, sink, (stage, file, done, total) => {
      const now = performance.now();
      if (now - lastPost < 100 && done < total) return;
      lastPost = now;
      self.postMessage({ type: "progress", stage, file, done, total });
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
    self.postMessage({ type: "done", contentId, count });
  } catch (err) {
    self.postMessage({ type: "error", message: String((err && err.message) || err) });
  }
};
