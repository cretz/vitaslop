// The page side of an import: turn what the person picked into `{ path, file }`
// entries, ask the worker what they are, and drive the import with progress.
//
// What people have: a `.pkg` beside a `work.bin`; a folder dumped from a console
// (`sce_pfs/`, `sce_sys/`, `eboot.bin`...); a zip of either; or a dump tree this
// emulator wrote. All of these are "some files", so the picker accepts files OR a
// folder, and the Rust side sniffs what it was given.

import { requestPersistence, storageRoom } from "./opfs.js";

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

/// What these files are. Resolves to the worker's probe object.
export function probe(entries) {
  return new Promise((resolve, reject) => {
    const w = spawn();
    w.onmessage = (e) => {
      const d = e.data;
      if (d.type === "panic") return reject(new Error("import panicked: " + d.message));
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

/// Import into `games/<titleId>/`. `onProgress({ stage, file, done, total, rate })`.
export async function run(entries, titleId, needBytes, onProgress = () => {}) {
  const room = await storageRoom();
  if (room && room.free < needBytes * 1.05) {
    throw new Error(
      `this title needs about ${(needBytes / 1e6) | 0} MB but this browser will only give this site ` +
        `${(room.free / 1e6) | 0} MB more (quota ${(room.quota / 1e6) | 0} MB, ${(room.usage / 1e6) | 0} MB used). ` +
        `Free up space on the device, or remove a title.`
    );
  }
  await requestPersistence();
  const t0 = performance.now();
  return new Promise((resolve, reject) => {
    const w = spawn();
    let panic = null;
    w.onmessage = (e) => {
      const d = e.data;
      if (d.type === "progress") {
        const secs = (performance.now() - t0) / 1000;
        onProgress({ ...d, rate: secs > 0.5 ? d.done / secs : 0 });
        return;
      }
      if (d.type === "panic") {
        panic = d.message;
        return;
      }
      w.terminate();
      if (d.type === "done") resolve(d);
      else reject(new Error(d.message || panic || "import failed"));
    };
    w.onerror = (e) => {
      w.terminate();
      reject(new Error(panic || e.message || "the import worker died"));
    };
    w.postMessage({ type: "import", files: entries, titleId });
  });
}
