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
// Messages in:  { type: "import", files: [{ path, file }], addonFor?: titleId }
//               { type: "probe",  files: [{ path, file }] }   (identify only)
//               { type: "repatch", files: [{ path, file }], titleId, pickedRoot, source }
//                 (a rePatch overlay for titleId - see `repatchImport`)
// Messages out: { type: "probe", probe } | { type: "progress", stage, file, done, total }
//               { type: "done", titleId, contentId, count, addon? } | { type: "error", message }
//
// With `addonFor`, the files must be an ADD-ON of that title (param.sfo CATEGORY `ac`, its
// TITLE_ID), and they are stored under `addons/<title>/<dir>/` with an `addon.json` record
// beside them (opfs.js). Without it, an add-on is refused: filed under its title's id it would
// replace the game.
//               { type: "panic", message }

import init, { ingest_probe, ingest_import, repatch_plan, repatch_import } from "./pkg/vitaslop_web.js";
import { titleDir, addonsDir, ADDON_RECORD, isAddonName, encodeName, decodeName, storageRoom, repatchDir, readRepatch, REPATCH_RECORD, listAddons } from "./opfs.js";
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
    category: p.category,
    isAddon: p.isAddon,
    addonDir: p.addonDir,
    isUpdate: p.isUpdate,
    refusedAsTitle: p.refusedAsTitle,
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
    if (d.type === "repatch") {
      await repatchImport(d, post);
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
    const addonFor = d.addonFor || null;
    if (addonFor) {
      if (probe.isUpdate) throw new Error(probe.refusedAsTitle);
      if (!probe.isAddon) {
        throw new Error(`this is not an add-on but a game (${titleId}) - import it from the library`);
      }
      if (titleId !== addonFor) throw new Error(`this add-on belongs to ${titleId}, not ${addonFor}`);
      if (!isAddonName(probe.addonDir)) {
        throw new Error(`this add-on's content id (${probe.contentId}) does not name the folder it installs as`);
      }
    } else if (probe.refusedAsTitle) {
      throw new Error(probe.refusedAsTitle);
    } else if (probe.isAddon) {
      throw new Error(
        `this is an add-on for ${titleId}, not a game - open that game in the library and add it under Add-ons`
      );
    }
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
    const dir = addonFor
      ? await (await addonsDir(titleId, { create: true })).getDirectoryHandle(probe.addonDir, { create: true })
      : await titleDir(titleId, { create: true });
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
    let addon = null;
    if (addonFor) {
      addon = {
        titleId,
        dir: probe.addonDir,
        contentId,
        title: probe.title || probe.addonDir,
        version: probe.appVersion || "",
        sourceKind: probe.kind,
        bytes: probe.bytes,
        files: count,
        importedAt: Date.now(),
      };
      const rh = await dir.getFileHandle(ADDON_RECORD, { create: true });
      const rw = await rh.createWritable();
      await rw.write(JSON.stringify(addon));
      await rw.close();
    }
    self.postMessage({ type: "done", titleId, contentId, count, addon });
  } catch (err) {
    self.postMessage({ type: "error", message: String((err && err.message) || err) });
  }
};

/// Copy a rePatch overlay (a picked folder, or one .zip) for `d.titleId` into a NEW generation
/// directory beside the old one, then point the record at it and drop the old - so a failed copy
/// leaves the overlay that was there. The rules (layout, refusing encrypted executables) are the
/// runtime's: `ingest::repatch`.
async function repatchImport(d, post) {
  const src = fileSource(d.files);
  // What the game holds - its stored paths and add-on directories - so each of the overlay's
  // folders is placed over the game or over an add-on, and one nothing would read is refused.
  const gamePaths = [];
  for await (const [name, h] of (await titleDir(d.titleId, { create: false })).entries()) {
    if (h.kind === "file") gamePaths.push(decodeName(name));
  }
  const addons = (await listAddons(d.titleId)).map((a) => a.dir);
  const planned = repatch_plan(src, d.titleId, !!d.pickedRoot, gamePaths, addons);
  const top = await repatchDir(d.titleId, { create: true });
  const old = await readRepatch(d.titleId);
  const gen = `g${Date.now().toString(36)}`;
  const dir = await top.getDirectoryHandle(gen, { create: true });
  const handles = new Map();
  try {
    let opened = 0;
    await inParallel(planned, async (path) => {
      const fh = await dir.getFileHandle(encodeName(path), { create: true });
      handles.set(path, await fh.createSyncAccessHandle());
      opened += 1;
      post(false, () => ({ type: "progress", stage: "preparing", file: "", done: opened, total: planned.length, unit: "files" }));
    });
    let cur = null;
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
        cur = null;
      },
    };
    const got = repatch_import(src, sink, d.titleId, !!d.pickedRoot, gamePaths, addons, (stage, file, done, total) => {
      post(done >= total, () => ({ type: "progress", stage, file, done, total }));
    });
    const rec = {
      titleId: d.titleId,
      source: d.source || "",
      bytes: got.bytes,
      files: got.files,
      code: got.code,
      addcont: got.addcont,
      appFiles: got.appFiles,
      inferred: got.inferred,
      codeHash: got.codeHash,
      importedAt: Date.now(),
      disabled: false,
      gen,
    };
    const w = await (await top.getFileHandle(REPATCH_RECORD, { create: true })).createWritable();
    await w.write(JSON.stringify(rec));
    await w.close();
    if (old && old.gen !== gen) await top.removeEntry(old.gen, { recursive: true }).catch(() => {});
    self.postMessage({ type: "done", titleId: d.titleId, repatch: rec, ignored: got.ignored });
  } catch (err) {
    for (const h of handles.values()) {
      try {
        h.close();
      } catch {}
    }
    await top.removeEntry(gen, { recursive: true }).catch(() => {});
    throw err;
  }
}
