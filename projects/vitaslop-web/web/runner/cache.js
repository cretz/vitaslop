// cache.js - a VERIFIED on-device copy of a large runner asset.
//
// A 54 MB frame capsule fetched fresh by every job cost ~65 s of each 90 s job on the phone
// (the render itself was 16-25 s). So a big asset is downloaded once into OPFS
// (`runner-cache/`, apart from the title store) and reused - but only after proving it is the
// server's CURRENT content: the server states the SHA-256 (`?sha`), the stored copy's digest is
// recomputed here, and any mismatch (a changed asset, a torn write) is a fresh download. A copy
// is never trusted on its name.

const hex = (buf) => Array.from(new Uint8Array(buf), (b) => b.toString(16).padStart(2, "0")).join("");
const digest = async (bytes) => hex(await crypto.subtle.digest("SHA-256", bytes));

async function dir() {
  const root = await navigator.storage.getDirectory();
  return root.getDirectoryHandle("runner-cache", { create: true });
}

/// The asset's bytes, from the verified local copy when there is one. `progress(text)` reports
/// which way it went. Returns `{ bytes, cached, ms }`.
export async function cachedAsset(path, progress = () => {}) {
  const t0 = performance.now();
  const meta = await (await fetch(`/runner-assets/${path}?sha`)).json();
  if (!meta.sha) throw new Error(`asset ${path}: the server gave no sha (${JSON.stringify(meta)})`);
  const name = path.replace(/[^A-Za-z0-9._-]/g, "_");
  const d = await dir();
  try {
    const fh = await d.getFileHandle(name);
    const file = await fh.getFile();
    if (file.size === meta.size) {
      progress(`checking the cached ${path}`);
      const bytes = new Uint8Array(await file.arrayBuffer());
      if ((await digest(bytes)) === meta.sha) return { bytes, cached: true, ms: performance.now() - t0 };
    }
  } catch {
    // not cached yet
  }
  progress(`downloading ${path} (${(meta.size / 1e6).toFixed(0)} MB, once)`);
  const r = await fetch(`/runner-assets/${path}`);
  if (!r.ok) throw new Error(`asset ${path}: HTTP ${r.status}`);
  const bytes = new Uint8Array(await r.arrayBuffer());
  const got = await digest(bytes);
  if (got !== meta.sha) throw new Error(`asset ${path}: downloaded ${bytes.length} bytes hashing ${got}, the server said ${meta.sha}`);
  try {
    const fh = await d.getFileHandle(name, { create: true });
    const w = await fh.createWritable();
    await w.write(bytes);
    await w.close();
  } catch (e) {
    progress(`could not cache ${path}: ${e.message}`);
  }
  return { bytes, cached: false, ms: performance.now() - t0 };
}
