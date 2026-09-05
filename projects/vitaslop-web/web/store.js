// Where the front end keeps what it knows: settings (global and per title) and the
// library index. Titles' bytes live under `games/<id>/` (opfs.js) and saves under
// `gamedata/` (gamedata.js); this file owns `library/<id>/` and localStorage.
//
// Settings are localStorage because they are small, synchronous to read, and must be
// there before the first frame of the page. The library is OPFS because it is
// per-title files (a JSON record plus two images), and OPFS is what survives with the
// titles - a cleared localStorage must not orphan a gigabyte of imported games.
//
// The RULES for settings (defaults, merge, which knob a checkbox means) are in Rust
// (`vitaslop-frontend`), reached through the wasm exports; this file only stores.

import init, {
  settings_defaults,
  settings_effective,
  settings_run_knobs,
  settings_parse_knobs,
  input_vocabulary,
} from "./pkg/vitaslop_web.js";

let ready = null;
/// The wasm module, initialised once for the page. Cheap after the first call.
export function wasm() {
  if (!ready) ready = init();
  return ready;
}

const GLOBAL_KEY = "vitaslop.settings";
const TITLE_KEY = (id) => `vitaslop.title.${id}`;

function readJson(key) {
  try {
    const s = localStorage.getItem(key);
    return s ? JSON.parse(s) : null;
  } catch {
    return null;
  }
}
function writeJson(key, v) {
  try {
    if (v === null) localStorage.removeItem(key);
    else localStorage.setItem(key, JSON.stringify(v));
  } catch {}
}

/// The stored global record (a partial object; missing keys are defaults).
export const globalSettings = () => readJson(GLOBAL_KEY) || {};
export const saveGlobalSettings = (obj) => writeJson(GLOBAL_KEY, obj);
/// The stored per-title patch (only what differs for this title), or {}.
export const titleSettings = (id) => readJson(TITLE_KEY(id)) || {};
export const saveTitleSettings = (id, patch) =>
  writeJson(TITLE_KEY(id), patch && Object.keys(patch).length ? patch : null);

/// The full default record.
export async function defaults() {
  await wasm();
  return JSON.parse(settings_defaults());
}

/// The settings a run of `id` uses: defaults + global + the title's patch.
export async function effective(id = null) {
  await wasm();
  const g = JSON.stringify(globalSettings());
  const p = id ? JSON.stringify(titleSettings(id)) : null;
  return JSON.parse(settings_effective(g, p));
}

/// The VITASLOP_* map for an effective record.
export async function runKnobs(eff) {
  await wasm();
  return settings_run_knobs(JSON.stringify(eff));
}

export async function parseKnobs(text) {
  await wasm();
  return settings_parse_knobs(text);
}

export async function vocabulary() {
  await wasm();
  return input_vocabulary();
}

// ------------------------------- the library -------------------------------

async function libraryRoot(create = true) {
  const root = await navigator.storage.getDirectory();
  return root.getDirectoryHandle("library", { create });
}

/// Every imported title's record, newest import first. Reads one small JSON per
/// title and no image bytes, so a thousand titles is a thousand short reads.
export async function listTitles() {
  let lib;
  try {
    lib = await libraryRoot(false);
  } catch {
    return [];
  }
  const out = [];
  for await (const [name, handle] of lib.entries()) {
    if (handle.kind !== "directory") continue;
    try {
      const fh = await handle.getFileHandle("meta.json");
      const meta = JSON.parse(await (await fh.getFile()).text());
      if (meta && meta.titleId === name) out.push(meta);
    } catch {
      // A directory without a record is a half-finished import; it is not a title.
    }
  }
  out.sort((a, b) => (b.importedAt || 0) - (a.importedAt || 0));
  return out;
}

export async function readTitle(id) {
  try {
    const lib = await libraryRoot(false);
    const dir = await lib.getDirectoryHandle(id);
    const fh = await dir.getFileHandle("meta.json");
    return JSON.parse(await (await fh.getFile()).text());
  } catch {
    return null;
  }
}

export async function writeTitle(meta, images = {}) {
  const lib = await libraryRoot(true);
  const dir = await lib.getDirectoryHandle(meta.titleId, { create: true });
  for (const [name, bytes] of Object.entries(images)) {
    if (!bytes) continue;
    const fh = await dir.getFileHandle(name, { create: true });
    const w = await fh.createWritable();
    await w.write(bytes);
    await w.close();
  }
  const fh = await dir.getFileHandle("meta.json", { create: true });
  const w = await fh.createWritable();
  await w.write(JSON.stringify(meta));
  await w.close();
}

export async function touchTitle(id, patch) {
  const meta = await readTitle(id);
  if (!meta) return;
  await writeTitle({ ...meta, ...patch });
}

/// An object URL for a title's image, or null. Cached for the page's life so the
/// grid does not re-read a thousand PNGs on every render.
const imageUrls = new Map();
export async function titleImage(id, name = "icon0.png") {
  const key = `${id}/${name}`;
  if (imageUrls.has(key)) return imageUrls.get(key);
  let url = null;
  try {
    const lib = await libraryRoot(false);
    const dir = await lib.getDirectoryHandle(id);
    const fh = await dir.getFileHandle(name);
    url = URL.createObjectURL(await fh.getFile());
  } catch {}
  imageUrls.set(key, url);
  return url;
}

/// Remove a title's record and images (not its bytes and not its saves - the
/// caller does those, deliberately separately).
export async function removeTitleRecord(id) {
  try {
    const lib = await libraryRoot(false);
    await lib.removeEntry(id, { recursive: true });
  } catch {}
  for (const k of [...imageUrls.keys()]) {
    if (k.startsWith(id + "/")) {
      const u = imageUrls.get(k);
      if (u) URL.revokeObjectURL(u);
      imageUrls.delete(k);
    }
  }
}

/// Bytes used by one imported title's files, summed from OPFS.
export async function titleBytes(id) {
  try {
    const root = await navigator.storage.getDirectory();
    const games = await root.getDirectoryHandle("games");
    const dir = await games.getDirectoryHandle(id);
    let n = 0;
    for await (const [, h] of dir.entries()) {
      if (h.kind === "file") n += (await h.getFile()).size;
    }
    return n;
  } catch {
    return 0;
  }
}

/// Human sizes. 1e6-based, because that is what disks and quotas are quoted in.
export function fmtBytes(n) {
  if (!n) return "0 MB";
  if (n < 1e6) return `${(n / 1e3).toFixed(0)} KB`;
  if (n < 1e9) return `${(n / 1e6).toFixed(0)} MB`;
  return `${(n / 1e9).toFixed(2)} GB`;
}
