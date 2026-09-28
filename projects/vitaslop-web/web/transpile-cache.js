// The TRANSPILE CACHE: a title's transpiled guest module, kept in OPFS so only the first play
// after a build pays the transpile (4-19 s on a phone). Worker-only (sync access handles).
//
// Layout (see the README's "OPFS layout"): inside the title's own directory, so removing or
// re-importing the game deletes it with everything else -
//   games/<titleId>/vitaslop-transpiled/<name>.wasm   the module bytes
//   games/<titleId>/vitaslop-transpiled/<name>.json   its layout numbers, written LAST (the marker)
// `<name>` = `<build>-<settings>-<hostOff>`:
//   build     a hash of the bundle's build stamp (build.mjs writes a new one every build), so any
//             new build of the emulator invalidates every stored module - one re-transpile per
//             title per deploy;
//   settings  the transpiler's resolved emit settings for this run's knobs (Rust
//             `transpile_settings_key`), so an instrumented run never loads a plain module;
//   hostOff   where the run worker reserved the guest region - the module is emitted for it.
//
// A title keeps ONE entry: a miss deletes whatever the directory held before transpiling, and
// every prepare sweeps other titles' entries from older builds, so an invalidated module never
// lingers on disk.
import { titleDir } from "./opfs.js";

const DIR = "vitaslop-transpiled";

/// FNV-1a over a string, as hex - a short, file-name-safe token.
export function tokenOf(text) {
  let h = 0xcbf29ce484222325n;
  for (let i = 0; i < text.length; i++) {
    h ^= BigInt(text.charCodeAt(i));
    h = (h * 0x100000001b3n) & 0xffffffffffffffffn;
  }
  return h.toString(16).padStart(16, "0");
}

/// The build token: a hash of `build-stamp.txt` beside the bundle this worker runs.
export async function buildToken(bundleDir) {
  const r = await fetch(new URL(bundleDir + "build-stamp.txt", self.location.href), { cache: "no-store" });
  if (!r.ok) throw new Error(`build stamp: HTTP ${r.status}`);
  return tokenOf(await r.text());
}

async function cacheDir(titleId, create) {
  const dir = await titleDir(titleId, { create: false });
  return dir.getDirectoryHandle(DIR, { create });
}

async function readAll(fileHandle) {
  const h = await fileHandle.createSyncAccessHandle();
  try {
    const out = new Uint8Array(h.getSize());
    const n = h.read(out, { at: 0 });
    if (n !== out.length) throw new Error(`short read ${n} of ${out.length}`);
    return out;
  } finally {
    h.close();
  }
}

async function writeAll(dir, name, bytes) {
  const fh = await dir.getFileHandle(name, { create: true });
  const h = await fh.createSyncAccessHandle();
  try {
    h.truncate(0);
    let at = 0;
    while (at < bytes.length) at += h.write(bytes.subarray(at), { at });
    h.flush();
  } finally {
    h.close();
  }
}

/// The stored module named `name` for `titleId`: `{ wasm, meta }`, or null when there is none
/// (or it is incomplete - the `.json` is written last and names the wasm's size).
export async function lookup(titleId, name) {
  let dir;
  try {
    dir = await cacheDir(titleId, false);
  } catch {
    return null;
  }
  try {
    const meta = JSON.parse(new TextDecoder().decode(await readAll(await dir.getFileHandle(name + ".json"))));
    const wasm = await readAll(await dir.getFileHandle(name + ".wasm"));
    if (meta.wasmBytes !== wasm.length) return null;
    return { wasm, meta };
  } catch {
    return null;
  }
}

/// Delete every stored module of `titleId` - what an invalidation does before transpiling.
/// Returns how many modules were there (for the prepare line).
export async function clearTitle(titleId) {
  let n = 0;
  try {
    const dir = await cacheDir(titleId, false);
    for await (const [name] of dir.entries()) if (name.endsWith(".wasm")) n++;
    await (await titleDir(titleId, { create: false })).removeEntry(DIR, { recursive: true });
  } catch {}
  return n;
}

/// Store `wasm` + `meta` under `name`, as the title's only entry.
export async function store(titleId, name, wasm, meta) {
  await clearTitle(titleId);
  const dir = await cacheDir(titleId, true);
  await writeAll(dir, name + ".wasm", wasm);
  await writeAll(dir, name + ".json", new TextEncoder().encode(JSON.stringify({ ...meta, wasmBytes: wasm.length })));
}

/// Remove every title's entries that were not made by build `build` - they can never be used
/// again. A directory listing per title; nothing is read. Returns how many modules went.
export async function sweepOtherBuilds(build) {
  let games;
  let swept = 0;
  try {
    games = await (await navigator.storage.getDirectory()).getDirectoryHandle("games", { create: false });
  } catch {
    return swept;
  }
  for await (const [, title] of games.entries()) {
    if (title.kind !== "directory") continue;
    let dir;
    try {
      dir = await title.getDirectoryHandle(DIR, { create: false });
    } catch {
      continue;
    }
    const stale = [];
    for await (const [name] of dir.entries()) if (!name.startsWith(build + "-")) stale.push(name);
    for (const name of stale) {
      await dir.removeEntry(name).catch(() => {});
      if (name.endsWith(".wasm")) swept++;
    }
  }
  return swept;
}
