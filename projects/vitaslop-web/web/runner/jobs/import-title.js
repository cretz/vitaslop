// import-title - stream one title this server offers into the device's OPFS, as the main page's
// import does, so a live job can boot it without anyone touching the phone.
//
// params: { titleId }. Resumable: every file is sized first (HEAD), and a file already stored at
// that size is kept, so a job cut off by its time limit is simply queued again.

import { importTitle, isComplete, storageRoom, requestPersistence } from "../../opfs.js";
import { readTitle, writeTitle } from "../../store.js";

/// The library's record for the title (what the main page's import writes after its files), so it
/// is listed and playable from the library: name, content id and version from its own param.sfo,
/// and its icon0/pic0.
async function writeLibraryRecord(id, url, bytes, files) {
  const get = async (p) => {
    const r = await fetch(url(p));
    return r.ok ? new Uint8Array(await r.arrayBuffer()) : null;
  };
  const sfo = parseSfo(await get("files/sce_sys/param.sfo"));
  const icon0 = await get("files/sce_sys/icon0.png");
  const pic0 = await get("files/sce_sys/pic0.png");
  const had = await readTitle(id);
  const meta = {
    titleId: id,
    title: sfo.TITLE || id,
    contentId: sfo.CONTENT_ID || "",
    appVersion: sfo.APP_VER || "",
    sourceKind: "runner",
    bytes,
    files,
    hasIcon: !!icon0,
    hasPic: !!pic0,
    importedAt: Date.now(),
    lastPlayedAt: had ? had.lastPlayedAt || 0 : 0,
  };
  await writeTitle(meta, { "icon0.png": icon0, "pic0.png": pic0 });
}

/// A param.sfo's string and integer entries by key; {} when it is missing or not an SFO.
function parseSfo(b) {
  const out = {};
  if (!b || b.length < 20 || b[1] !== 0x50 || b[2] !== 0x53 || b[3] !== 0x46) return out;
  const v = new DataView(b.buffer, b.byteOffset, b.byteLength);
  const keys = v.getUint32(8, true), data = v.getUint32(12, true), n = v.getUint32(16, true);
  const cstr = (o, max) => {
    let e = o;
    while (e < o + max && e < b.length && b[e]) e++;
    return new TextDecoder().decode(b.subarray(o, e));
  };
  for (let i = 0; i < n; i++) {
    const e = 20 + i * 16;
    const key = cstr(keys + v.getUint16(e, true), 64);
    const fmt = v.getUint16(e + 2, true), len = v.getUint32(e + 4, true), at = data + v.getUint32(e + 12, true);
    out[key] = fmt === 0x0404 ? v.getUint32(at, true) : cstr(at, len);
  }
  return out;
}

export async function run(params, ctx) {
  const id = params.titleId;
  if (!id) throw new Error("params.titleId is required");
  if (await isComplete(id)) {
    const q = encodeURIComponent(id);
    const url = (p) => `/game/${q}/${p.split("/").map(encodeURIComponent).join("/")}`;
    const had = await readTitle(id);
    if (!had) {
      const paths = await (await fetch(`/game-manifest.json?title=${q}`)).json();
      await writeLibraryRecord(id, url, 0, paths.length);
    }
    return { summary: `${id} already imported` + (had ? "" : "; library record written"), meta: await readTitle(id) };
  }
  const q = encodeURIComponent(id);
  const res = await fetch(`/game-manifest.json?title=${q}`);
  if (!res.ok) throw new Error(`manifest for ${id}: HTTP ${res.status}`);
  const paths = await res.json();
  const url = (p) => `/game/${q}/${p.split("/").map(encodeURIComponent).join("/")}`;
  const sizes = [];
  for (let i = 0; i < paths.length; i++) {
    const h = await fetch(url(paths[i]), { method: "HEAD" });
    const n = Number(h.headers.get("content-length"));
    if (!h.ok || !Number.isFinite(n)) throw new Error(`HEAD ${paths[i]}: HTTP ${h.status}, length ${h.headers.get("content-length")}`);
    sizes.push(n);
    if (i % 50 === 0) ctx.progress(`sizing ${i}/${paths.length}`);
  }
  const total = sizes.reduce((a, b) => a + b, 0);
  const room = await storageRoom();
  const persisted = await requestPersistence();
  const entries = paths.map((path, i) => ({
    path,
    size: sizes[i],
    source: async () => {
      const r = await fetch(url(path));
      if (!r.ok) throw new Error(`GET ${path}: HTTP ${r.status}`);
      return { body: r.body, bytes: sizes[i] };
    },
  }));
  const t0 = performance.now();
  let last = 0;
  const out = await importTitle(id, entries, (done, count, bytes) => {
    const now = performance.now();
    if (now - last < 2000) return;
    last = now;
    ctx.progress(`${done}/${count} files, ${(bytes / 1048576).toFixed(0)}/${(total / 1048576).toFixed(0)} MB`);
  });
  const s = (performance.now() - t0) / 1000;
  await writeLibraryRecord(id, url, total, paths.length);
  return {
    summary: `${id}: ${paths.length} files, ${(total / 1048576).toFixed(0)} MB in ${s.toFixed(0)} s` + (out.reused ? " (already complete)" : ""),
    room,
    persisted,
    complete: await isComplete(id),
  };
}
