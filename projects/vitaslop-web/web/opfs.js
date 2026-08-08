// The browser's game storage: the Origin Private File System.
//
// # Why the container cannot just be bytes in memory
// A retail Vita title is over a gigabyte. Held as JS `ArrayBuffer`s it costs that much
// in the renderer, and copied into the emulator's wasm heap it costs it again - against
// a wasm32 address space that tops out at 4 GB. Measured on a 1719 MB title before this
// existed: Chrome peaked at 8.01 GB during ingest and the worker was killed mid-boot,
// which from outside looks like a page that has stopped answering rather than an
// out-of-memory error.
//
// So the container is never fully resident. It is STREAMED into OPFS - from the user's
// picked `.pkg` + `work.bin`, or from this origin for a test run - and read back in
// pieces. Peak memory during import is one chunk, not one title.
//
// # Why OPFS specifically, and not Cache Storage or IndexedDB
// Only OPFS offers `createSyncAccessHandle`, whose `read` is SYNCHRONOUS. That is not a
// convenience here, it is the requirement: the emulator serves guest file reads inside
// host calls, which are synchronous Rust running on a suspended guest stack. An async
// storage API could not back them without turning every guest read into a suspension.
// Sync access handles are available in Workers only - which is where the emulator runs.

/// Directory under the OPFS root holding one title's files, keyed by `id`.
export async function titleDir(id, { create = true } = {}) {
  const root = await navigator.storage.getDirectory();
  const games = await root.getDirectoryHandle("games", { create });
  return games.getDirectoryHandle(id, { create });
}

/// A file name safe to use as a flat OPFS entry, since OPFS has no path separators of
/// its own and a container's paths do. Reversed by `decodeName`.
export function encodeName(path) {
  return path.replace(/\//g, "%2F");
}

export function decodeName(name) {
  return name.replace(/%2F/g, "/");
}

/// Marker written LAST, after every file is durably stored. Its presence is what makes
/// an import complete: a run interrupted halfway leaves files behind but no marker, and
/// the next boot re-imports rather than mounting a truncated title. Storing the file
/// count too means a partial directory that somehow kept its marker is still caught.
const MANIFEST = "vitaslop-opfs-manifest.json";

/// Whether `id` is already imported and complete. Cheap: one read of the marker.
export async function isImported(id, expectedCount) {
  try {
    const dir = await titleDir(id, { create: false });
    const fh = await dir.getFileHandle(MANIFEST);
    const m = JSON.parse(await (await fh.getFile()).text());
    return m.count === expectedCount && m.complete === true;
  } catch {
    return false;
  }
}

/// Stream one `ReadableStream` (or `Blob`) into `dir` under `path`, without ever
/// holding the whole thing. Returns the bytes written.
///
/// `onBytes(n)` is called with each chunk's length as it passes through. It exists because
/// a container's bytes are NOT spread evenly over its files: one title puts 660 of its 688
/// MB into 28 of its 248 files, so a caller that only learns about a finished file shows a
/// frozen counter for the entire minutes-long download of a single large one - which reads
/// exactly like a hang, and was reported as one.
async function storeOne(dir, path, source, onBytes = () => {}) {
  const fh = await dir.getFileHandle(encodeName(path), { create: true });
  const w = await fh.createWritable();
  const stream = source instanceof Blob ? source.stream() : source;
  const counted = stream.pipeThrough(
    new TransformStream({
      transform(chunk, ctrl) {
        onBytes(chunk.byteLength);
        ctrl.enqueue(chunk);
      },
    })
  );
  // `pipeTo` hands the writable ownership of the stream and closes it on completion,
  // so a failure part-way cannot leave a half-written file locked open.
  await counted.pipeTo(w);
  return (await fh.getFile()).size;
}

/// How much room this origin has, or `null` where the browser will not say.
///
/// # Why this is checked BEFORE an import and not caught after
/// A phone's quota for an origin is a fraction of its free disk, and these containers are
/// 237 MB to 1719 MB. Discovering the limit by hitting it means a `QuotaExceededError`
/// hundreds of megabytes into a download, over wifi, with the partial files still on disk.
/// Asking first costs one call and turns that into a sentence naming both numbers.
export async function storageRoom() {
  if (!navigator.storage || !navigator.storage.estimate) return null;
  const { quota, usage } = await navigator.storage.estimate();
  if (typeof quota !== "number") return null;
  return { quota, usage: usage || 0, free: quota - (usage || 0) };
}

/// Ask the browser to treat this origin's storage as persistent, so an imported title is not
/// evicted under disk pressure - re-importing a gigabyte is not a cost to pay silently.
/// Returns whether it was granted. A refusal is not fatal and is reported, not swallowed.
export async function requestPersistence() {
  if (!navigator.storage || !navigator.storage.persist) return false;
  try {
    return (await navigator.storage.persisted()) || (await navigator.storage.persist());
  } catch {
    return false;
  }
}

/// Import a whole container into OPFS from a list of `{ path, source }`, where `source`
/// is a `Blob`/`File` (the web form) or a `ReadableStream` (a fetch body). Calls
/// `onProgress(done, total, bytes)` as it goes. Idempotent: an already-complete title
/// is left alone.
export async function importTitle(id, entries, onProgress = () => {}) {
  if (await isImported(id, entries.length)) {
    onProgress(entries.length, entries.length, 0, true);
    return { reused: true };
  }
  const dir = await titleDir(id);
  let bytes = 0;
  for (let i = 0; i < entries.length; i++) {
    // Report DURING each file as well as after it, throttled to every 4 MB so a slow large
    // file still moves the counter without flooding the caller. See `storeOne`.
    let inFlight = 0;
    let reported = 0;
    const written = await storeOne(dir, entries[i].path, await entries[i].source(), (n) => {
      inFlight += n;
      if (inFlight - reported >= 4 << 20) {
        reported = inFlight;
        onProgress(i, entries.length, bytes + inFlight, false);
      }
    });
    bytes += written;
    onProgress(i + 1, entries.length, bytes, false);
  }
  // The marker goes last, so "present" always means "everything before it is there".
  const fh = await dir.getFileHandle(MANIFEST, { create: true });
  const w = await fh.createWritable();
  await w.write(JSON.stringify({ complete: true, count: entries.length, bytes }));
  await w.close();
  return { reused: false, bytes };
}

/// Open every stored file of `id` as a SYNCHRONOUS access handle, returned as
/// `{ path: handle }`. Worker-only. This is the async step that buys synchronous reads
/// later - the emulator cannot await inside a guest file read, so every handle it may
/// need has to be open before the guest starts.
export async function openTitleSync(id) {
  const dir = await titleDir(id, { create: false });
  const handles = {};
  for await (const [name, h] of dir.entries()) {
    if (name === MANIFEST || h.kind !== "file") continue;
    handles[decodeName(name)] = await h.createSyncAccessHandle();
  }
  return handles;
}

/// A reader the wasm side calls: `size(path)` and `read(path, offset, into)`, both
/// synchronous. `into` is a `Uint8Array` view the caller owns; the return is how many
/// bytes were actually read (short at end of file, 0 for an unknown path).
export function syncReader(handles) {
  return {
    paths: () => Object.keys(handles),
    size: (path) => (handles[path] ? handles[path].getSize() : -1),
    read: (path, offset, into) => {
      const h = handles[path];
      return h ? h.read(into, { at: offset }) : 0;
    },
    close: () => {
      for (const h of Object.values(handles)) h.close();
    },
  };
}
