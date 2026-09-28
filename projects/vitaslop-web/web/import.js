// The page side of an import: turn what the person picked into `{ path, file }`
// entries, ask the worker what they are, and drive the import with progress.
//
// What people have: a `.pkg` beside a `work.bin`; a folder dumped from a console
// (`sce_pfs/`, `sce_sys/`, `eboot.bin`...); a zip of either; or a dump tree this
// emulator wrote. All of these are "some files", so the picker accepts files OR a
// folder, and the Rust side sniffs what it was given.

import { requestPersistence } from "./opfs.js";

/// Relative paths from a picker's FileList or a drop's items. Folder picks carry
/// `webkitRelativePath` (with the picked folder as the first segment, which is kept -
/// the sniffer finds the root wherever it is).
export function entriesFromFiles(fileList) {
  const out = [];
  for (const f of fileList) {
    const rel = f.webkitRelativePath && f.webkitRelativePath.length ? f.webkitRelativePath : f.name;
    out.push({ path: rel.replace(/\\/g, "/"), file: f });
  }
  return out;
}

/// The same, from a drag-and-drop DataTransfer (walks dropped folders).
export async function entriesFromDrop(dataTransfer) {
  const out = [];
  const items = [...(dataTransfer.items || [])];
  const walk = async (entry, prefix) => {
    if (entry.isFile) {
      const file = await new Promise((res, rej) => entry.file(res, rej));
      out.push({ path: prefix + entry.name, file });
    } else if (entry.isDirectory) {
      const reader = entry.createReader();
      for (;;) {
        const batch = await new Promise((res, rej) => reader.readEntries(res, rej));
        if (!batch.length) break;
        for (const e of batch) await walk(e, prefix + entry.name + "/");
      }
    }
  };
  for (const it of items) {
    const entry = it.webkitGetAsEntry && it.webkitGetAsEntry();
    if (entry) await walk(entry, "");
    else if (it.kind === "file") {
      const f = it.getAsFile();
      if (f) out.push({ path: f.name, file: f });
    }
  }
  return out;
}

function spawn() {
  return new Worker("./import-worker.js", { type: "module" });
}

/// What these files are, without importing them. Resolves to the worker's probe
/// object; `onProgress({ stage, file, done, total })` reports the bytes it reads on
/// the way, which on a phone is the difference between a slow read and a hang.
export function probe(entries, onProgress = () => {}) {
  return new Promise((resolve, reject) => {
    const w = spawn();
    w.onmessage = (e) => {
      const d = e.data;
      if (d.type === "panic") return reject(new Error("import panicked: " + d.message));
      // The identify job reports its reads. Anything but a report ends the job.
      if (d.type === "progress") return onProgress(d);
      w.terminate();
      d.type === "probe" ? resolve(d.probe) : reject(new Error(d.message || "probe failed"));
    };
    w.onerror = (e) => {
      w.terminate();
      reject(new Error(e.message || "the import worker failed to start"));
    };
    w.postMessage({ type: "probe", files: entries });
  });
}

/// Identify AND import in one job: the picked files go straight in, and the worker
/// says what they were (`onProbe`) as soon as it knows, part way through.
///
/// There is no confirmation step. Identifying a package means reading it, on a phone
/// that is slow enough to look like a hang, and doing it twice - once to ask "import
/// this?" and once inside the import - paid that twice for nothing.
/// `onProgress({ stage, file, done, total, rate })`; `stage` is `reading` (identifying,
/// `total` 0), `preparing`, then the import's own stages.
export async function run(entries, onProbe = () => {}, onProgress = () => {}) {
  await requestPersistence();
  return new Promise((resolve, reject) => {
    const w = spawn();
    let panic = null;
    let probe = null;
    // The rate is measured over a TRAILING WINDOW, not since the start. A container is
    // a few big files and thousands of small ones, and they do not read at the same
    // speed: an average taken from the first byte spends the whole import catching up
    // to the current speed, which is what made the estimate fall by minutes at a time.
    // Nothing is reported until the window is wide enough to mean something.
    const WINDOW_MS = 8000;
    const samples = [];
    let smooth = 0;
    w.onmessage = (e) => {
      const d = e.data;
      if (d.type === "progress") {
        let rate = 0;
        // A stage counting FILES (the storage pre-open) has a total, but it is not
        // bytes, so it must not feed the byte-rate window.
        if (d.total && d.unit !== "files") {
          const now = performance.now();
          if (samples.length && samples[0].total !== d.total) samples.length = 0;
          samples.push({ t: now, done: d.done, total: d.total });
          while (samples.length > 2 && now - samples[0].t > WINDOW_MS) samples.shift();
          const secs = (now - samples[0].t) / 1000;
          const bytes = d.done - samples[0].done;
          if (secs >= 2 && bytes > 0) {
            const inst = bytes / secs;
            smooth = smooth ? smooth * 0.7 + inst * 0.3 : inst;
            rate = smooth;
          }
        } else {
          samples.length = 0;
          smooth = 0;
        }
        onProgress({ ...d, rate });
        return;
      }
      if (d.type === "probe") {
        probe = d.probe;
        onProbe(d.probe);
        return;
      }
      if (d.type === "panic") {
        panic = d.message;
        return;
      }
      w.terminate();
      if (d.type === "done") resolve({ ...d, probe });
      else reject(new Error(d.message || panic || "import failed"));
    };
    w.onerror = (e) => {
      w.terminate();
      reject(new Error(panic || e.message || "the import worker died"));
    };
    w.postMessage({ type: "import", files: entries });
  });
}
