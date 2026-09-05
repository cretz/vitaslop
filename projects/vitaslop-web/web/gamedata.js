// The GUEST'S OWN SAVED STATE in this browser's storage - and nothing else.
//
// # The separation, which is the whole point of this file
// `opfs.js` owns `games/<titleId>/`: the imported title, hundreds of megabytes of the
// user's own container, written once and read for ever. This file owns
// `gamedata/<titleId>/`: the few kilobytes the GAME writes as it is played. They are
// different top-level directories under the OPFS root, so nothing here can reach the
// title and nothing there can reach a save. "Clear my save" therefore cannot cost a
// re-import, and re-importing a title cannot cost a save.
//
// The rule about what may be INSIDE a container is not enforced here at all - it is
// enforced once, in Rust (`vitaslop_runtime::gamedata`), which refuses any entry that is
// not on the guest's writable mounts. This file moves opaque bytes; it does not know or
// decide what they mean.
//
// # Why one blob rather than a file per guest file
// A save is small - a savedata mount is kilobytes to a few megabytes - and the blob IS the
// download. Keeping it as the same `.zip` the user gets means the file they download is
// byte-for-byte the file the run wrote, rather than something reassembled at download time
// which could differ from what is actually stored.

/// Which profile this module reads and writes. "default" is the unnamed one and keeps
/// the original `gamedata/<id>` path, so a save made before profiles existed is still
/// found; any other name lives under `gamedata-profiles/<name>/<id>`. The page sets it
/// from the settings, and the run worker sets it from its start message - the two
/// sides must agree, and this is the one place the path is spelled.
let profile = "default";
export function setProfile(name) {
  profile = name && name !== "default" ? name : "default";
}
export const currentProfile = () => profile;

/// The directory holding one title's saved state. Deliberately NOT under `games/`.
export async function gameDataDir(id, { create = true } = {}) {
  const root = await navigator.storage.getDirectory();
  if (profile === "default") {
    const all = await root.getDirectoryHandle("gamedata", { create });
    return all.getDirectoryHandle(id, { create });
  }
  const all = await root.getDirectoryHandle("gamedata-profiles", { create });
  const p = await all.getDirectoryHandle(profile, { create });
  return p.getDirectoryHandle(id, { create });
}

/// Every profile that has a directory, plus "default".
export async function listProfiles() {
  const out = ["default"];
  try {
    const root = await navigator.storage.getDirectory();
    const all = await root.getDirectoryHandle("gamedata-profiles", { create: false });
    for await (const [name, h] of all.entries()) if (h.kind === "directory") out.push(name);
  } catch {}
  return out;
}

/// Every title id with a save in the CURRENT profile.
export async function listSaved() {
  const out = [];
  try {
    const root = await navigator.storage.getDirectory();
    let all;
    if (profile === "default") all = await root.getDirectoryHandle("gamedata", { create: false });
    else all = await (await root.getDirectoryHandle("gamedata-profiles", { create: false })).getDirectoryHandle(profile, { create: false });
    for await (const [name, h] of all.entries()) {
      if (h.kind !== "directory") continue;
      try {
        await h.getFileHandle(BLOB);
        out.push(name);
      } catch {}
    }
  } catch {}
  return out;
}

const BLOB = "gamedata.zip";
/// Written first and moved into place, so an interrupted write cannot leave a half a save
/// where a whole one was. See `write`.
const PART = "gamedata.zip.part";

/// The stored container for `id` as a `Uint8Array`, or `null` if this title has never
/// saved anything here.
export async function read(id) {
  try {
    const dir = await gameDataDir(id, { create: false });
    const file = await (await dir.getFileHandle(BLOB)).getFile();
    if (file.size === 0) return null;
    return new Uint8Array(await file.arrayBuffer());
  } catch {
    // A missing directory or file is the normal "no save yet" case, not an error.
    return null;
  }
}

/// Size and last-modified of the stored container, or `null`. For the launcher's card.
export async function info(id) {
  try {
    const dir = await gameDataDir(id, { create: false });
    const file = await (await dir.getFileHandle(BLOB)).getFile();
    if (file.size === 0) return null;
    return { bytes: file.size, modified: file.lastModified };
  } catch {
    return null;
  }
}

/// Store `bytes` (a `Uint8Array`) as this title's saved state.
///
/// >>> WRITTEN TO ONE SIDE AND MOVED INTO PLACE, because the alternative loses saves.
/// A `createWritable` on the live file TRUNCATES it first, so a tab closed during that
/// window leaves a zero-length or half-written container where a good one was - i.e. the
/// crash destroys the save it was about to update. Writing a `.part` and moving it over
/// keeps the previous container intact until a complete new one exists.
///
/// `move()` is not universal; where it is missing this falls back to the direct write and
/// says so ONCE, because that build genuinely does carry the risk above and a silent
/// downgrade would hide it.
let warnedNoMove = false;
export async function write(id, bytes) {
  const dir = await gameDataDir(id);
  const part = await dir.getFileHandle(PART, { create: true });
  const w = await part.createWritable();
  await w.write(bytes);
  await w.close();
  if (typeof part.move === "function") {
    await part.move(dir, BLOB);
    return;
  }
  if (!warnedNoMove) {
    warnedNoMove = true;
    console.warn(
      "[gamedata] this browser has no FileSystemFileHandle.move(), so a save is written " +
        "over the previous one directly. A tab closed mid-write can lose it."
    );
  }
  const live = await dir.getFileHandle(BLOB, { create: true });
  const lw = await live.createWritable();
  await lw.write(bytes);
  await lw.close();
  await dir.removeEntry(PART).catch(() => {});
}

/// Delete this title's saved state. Returns whether there was one.
///
/// Removes the title's directory under `gamedata/` and nothing else - the imported title
/// under `games/` is a different tree and is not reachable from here.
export async function clear(id) {
  try {
    const root = await navigator.storage.getDirectory();
    const all = await root.getDirectoryHandle("gamedata", { create: false });
    await all.removeEntry(id, { recursive: true });
    return true;
  } catch {
    return false;
  }
}

/// A save sink for the run worker: `sink(bytes)` returns immediately and the write happens
/// on the event loop.
///
/// >>> IT COALESCES, AND IT HAS TO. The emulator hands over a whole container each time the
/// guest's save changes, from a frame that cannot wait for storage. Without this, a title
/// that saves in a loop would queue writes faster than OPFS retires them and the backlog
/// would grow for the rest of the run. Only the NEWEST container matters - each one is
/// complete - so a write that arrives while another is in flight replaces any other
/// pending one rather than joining a queue.
///
/// `onError` is called with the failure, so a full quota reaches the page rather than the
/// console: it is the user's progress that is not being kept.
export function sink(id, onError = () => {}) {
  let inFlight = false;
  let pending = null;
  const pump = async () => {
    if (inFlight || pending === null) return;
    inFlight = true;
    const bytes = pending;
    pending = null;
    try {
      await write(id, bytes);
    } catch (err) {
      onError(err);
    } finally {
      inFlight = false;
      // Something may have arrived while this one was being written.
      pump();
    }
  };
  return (bytes) => {
    pending = bytes;
    pump();
  };
}
