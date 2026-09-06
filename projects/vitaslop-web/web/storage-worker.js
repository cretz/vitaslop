// The title's storage, read on its OWN thread.
//
// # Why a second worker
// A guest file read happens inside a host call, on a suspended guest stack, so the emulator
// worker can only read synchronously - and a `FileSystemSyncAccessHandle.read` is a
// synchronous trip through the browser's storage stack for every 64 KB. MEASURED in a V8
// profile of the emulator worker during a retail race on a desktop: about a millisecond per
// 64 KB read, on a machine whose disk moves that in microseconds - the cost is the call, not
// the bytes. A title streaming its track and a billboard movie made 24-40 of them a second
// there and 4-5 PER FRAME on the target phone, where the storage stack is slower again. That
// is several milliseconds of a 16 ms frame spent WAITING on storage, on the thread that runs
// the guest and the renderer.
//
// The device does none of that on its CPU: its I/O layer moves the bytes while the game
// runs. So this worker owns the sync handles, serves whole 64 KB pages into a
// `SharedArrayBuffer` ring, and after every request keeps READING AHEAD along the same file
// while the emulator gets on with the frame. A sequential stream - which is what an archive
// block reader and a movie demuxer both produce - then hits the ring on nearly every read,
// and the emulator thread pays one `Atomics.wait` round trip only on a genuine miss.
//
// # The protocol (see `openTitleCached` in opfs.js for the reading side)
// One request slot in the header: the emulator writes (path id, first page, page count),
// raises `REQ` and waits on `ACK`. This worker fills those pages into ring slots, clears
// `REQ`, bumps `ACK`, and then prefetches the pages that follow until the next request
// arrives. Every slot carries a tag (path id + 1, page, state, byte length); a slot is
// evicted only by winning a compare-exchange from READY to FILLING, and pinned by the reader
// only by winning READY to PINNED, so neither side can ever read or overwrite a slot the
// other is using.
import { openTitleSync, PAGE, SLOTS, DATA_OFF, H, TAG, STATE } from "./opfs.js";

/// Pages to read ahead of the last request, along the same file. Sixteen is a megabyte:
/// enough that a streaming reader stays ahead of the guest, small enough that two
/// interleaved streams (track data and a movie out of one archive) both fit the ring.
const AHEAD = 16;

self.onmessage = async (e) => {
  const { id, sab } = e.data;
  let handles;
  try {
    handles = await openTitleSync(id);
  } catch (err) {
    self.postMessage({ type: "error", message: String((err && err.message) || err) });
    return;
  }
  const paths = Object.keys(handles).sort();
  const sizes = paths.map((p) => handles[p].getSize());
  const hs = paths.map((p) => handles[p]);
  const h = new Int32Array(sab);
  const data = new Uint8Array(sab, DATA_OFF, SLOTS * PAGE);
  self.postMessage({ type: "ready", paths, sizes });

  // A sync access handle reads into an `AllowSharedBufferSource`, so straight into the ring.
  // An older engine that refuses a shared view gets a bounce buffer and one extra copy.
  let bounce = null;
  const readPage = (pid, page, slot) => {
    const at = page * PAGE;
    const want = Math.max(0, Math.min(PAGE, sizes[pid] - at));
    if (want === 0) return 0;
    const view = data.subarray(slot * PAGE, slot * PAGE + want);
    if (bounce === null) {
      try {
        return hs[pid].read(view, { at });
      } catch (err) {
        if (!(err instanceof TypeError)) throw err;
        bounce = new Uint8Array(PAGE);
      }
    }
    const n = hs[pid].read(bounce.subarray(0, want), { at });
    view.set(bounce.subarray(0, n));
    return n;
  };

  const tagOf = (slot, f) => h[H.TAGS + slot * TAG.STRIDE + f];
  const resident = (pid, page) => {
    for (let s = 0; s < SLOTS; s++) {
      if (tagOf(s, TAG.PATH) === pid + 1 && tagOf(s, TAG.PAGE) === page) return true;
    }
    return false;
  };
  // Round-robin victim pointer. A slot that is PINNED by the reader is skipped; with the
  // ring far larger than one request plus its read-ahead, the clock never laps a page the
  // emulator is still copying out of.
  let clock = 0;
  const ensure = (pid, page) => {
    if (resident(pid, page)) return;
    for (let tries = 0; tries < SLOTS * 2; tries++) {
      const s = clock;
      clock = (clock + 1) % SLOTS;
      const st = H.TAGS + s * TAG.STRIDE + TAG.STATE;
      if (Atomics.compareExchange(h, st, STATE.READY, STATE.FILLING) !== STATE.READY) continue;
      // Invalidate the tag BEFORE the bytes move, so a reader that scans mid-fill cannot
      // match the old page against new bytes.
      Atomics.store(h, H.TAGS + s * TAG.STRIDE + TAG.PATH, 0);
      let n = 0;
      let failed = false;
      try {
        n = readPage(pid, page, s);
      } catch (err) {
        failed = true;
        Atomics.store(h, H.ERR, 1);
        self.postMessage({ type: "error", message: `storage read failed for ${paths[pid]} at page ${page}: ${err}` });
      }
      Atomics.store(h, H.TAGS + s * TAG.STRIDE + TAG.PAGE, page);
      Atomics.store(h, H.TAGS + s * TAG.STRIDE + TAG.LEN, n);
      Atomics.store(h, H.TAGS + s * TAG.STRIDE + TAG.PATH, failed ? 0 : pid + 1);
      Atomics.store(h, st, STATE.READY);
      return;
    }
    // Every slot pinned at once cannot happen - the reader pins one slot at a time - so
    // reaching here is a protocol bug, and it must not spin silently.
    Atomics.store(h, H.ERR, 2);
    throw new Error("storage ring: no slot could be claimed");
  };

  let prefetch = null;
  for (;;) {
    if (Atomics.load(h, H.CLOSE) !== 0) break;
    if (Atomics.load(h, H.REQ) !== 0) {
      const pid = Atomics.load(h, H.REQ_PATH);
      const page = Atomics.load(h, H.REQ_PAGE);
      const count = Atomics.load(h, H.REQ_COUNT);
      for (let p = page; p < page + count; p++) ensure(pid, p);
      Atomics.store(h, H.REQ, 0);
      Atomics.add(h, H.ACK, 1);
      Atomics.notify(h, H.ACK);
      const last = Math.ceil(sizes[pid] / PAGE);
      prefetch = { pid, page: page + count, end: Math.min(page + count + AHEAD, last) };
      continue;
    }
    if (prefetch !== null && prefetch.page < prefetch.end) {
      ensure(prefetch.pid, prefetch.page++);
      continue;
    }
    prefetch = null;
    Atomics.wait(h, H.REQ, 0);
  }
  for (const hd of hs) {
    try {
      hd.close();
    } catch {
      // Closing on the way out; a handle that is already gone is not a failure.
    }
  }
  self.postMessage({ type: "closed" });
  self.close();
};
