// The worker half of the OPFS storage test.
//
// It has to be a worker: `createSyncAccessHandle` exists only in workers, which is also
// why the emulator runs there. Testing the read path from the main thread would test a
// different API than the product uses - or, as it did on the first attempt, simply throw.
import init, { opfs_verify } from "./pkg/vitaslop_web.js";
import { importTitle, openTitleSync, openTitleCached, syncReader } from "./opfs.js";

const TITLE = "opfs-test-fixture";

// Sizes chosen against the 997-byte verify chunk: shorter than a chunk, exactly a chunk,
// one over, an awkward multiple, and a large one. A size landing exactly on a boundary
// would hide an off-by-one in the final partial read, so most of these deliberately do
// not. The last two span the cached reader's 64 KB pages: one page plus a byte, and
// several pages with a partial last one, so a read that straddles pages and one that
// runs off the end of a short final page are both exercised through the ring.
const SIZES = [500, 997, 998, 4321, 65_537, 300_001];

// MIXED CASE on purpose - see the note in opfs-test.html's caller. The emulator
// normalises guest paths to lowercase; OPFS does not.
const paths = SIZES.map((_, i) => `files/PSP2/Data/Blob${i}.BIN`);

// A position-dependent pattern, so a read landing at the wrong OFFSET returns wrong
// bytes rather than plausible ones. A constant fill would pass every offset bug there is.
const pattern = (n, seed) => {
  const b = new Uint8Array(n);
  for (let i = 0; i < n; i++) b[i] = (i * 31 + seed * 17 + (i >> 8)) & 0xff;
  return b;
};

self.onmessage = async () => {
  try {
    await init();

    // Start clean, so a rerun tests an import rather than a leftover.
    try {
      const root = await navigator.storage.getDirectory();
      const games = await root.getDirectoryHandle("games", { create: true });
      await games.removeEntry(TITLE, { recursive: true });
    } catch {}

    const entries = SIZES.map((n, i) => ({
      path: paths[i],
      source: async () => new Blob([pattern(n, i)]),
    }));
    await importTitle(TITLE, entries);

    // A second import of a complete title must be a no-op: it is what makes "install
    // once, play many times" true, and what stops the e2e harness rewriting a gigabyte
    // on every run.
    const again = await importTitle(TITLE, entries);
    const reused = again.reused === true;

    const reader = syncReader(await openTitleSync(TITLE));
    const opened = reader.paths().length;

    // The real exported read path, over every fixture file, through the same normalised
    // keys the guest filesystem uses.
    let verify;
    try {
      verify = { ok: true, detail: opfs_verify(reader, "files/") };
    } catch (e) {
      verify = { ok: false, detail: String((e && e.message) || e) };
    }

    // A read running off the end must report what it actually delivered. A backing that
    // silently zero-padded would hand the guest bytes that were never in the file.
    const last = paths[paths.length - 1];
    const size = reader.size(last);
    const shortRead = { got: reader.read(last, size - 10, new Uint8Array(100)), expected: 10 };

    // And an offset read must return the bytes AT that offset - checked against the
    // pattern the fixture was built from, not against another read of the same file.
    const at = 1234;
    const want = pattern(SIZES[3], 3).slice(at, at + 16);
    const into = new Uint8Array(16);
    reader.read(paths[3], at, into);
    const same = want.every((v, i) => v === into[i]);

    reader.close();

    // >>> THE SAME CHECKS THROUGH THE READER THE RUN ACTUALLY USES: the page ring served
    // by the storage worker (`openTitleCached`). The direct reader above is what the
    // transpile worker uses; a run reads through this one, and a ring that hands back the
    // wrong page, the wrong tail of a short page, or stale bytes after an eviction would
    // fail here in seconds rather than deep in a title. The direct handles are closed
    // first: a sync access handle is exclusive per file.
    const cached = await openTitleCached(TITLE);
    const cachedOpened = cached.paths().length;
    let cachedVerify;
    try {
      cachedVerify = { ok: true, detail: opfs_verify(cached, "files/") };
    } catch (e) {
      cachedVerify = { ok: false, detail: String((e && e.message) || e) };
    }
    const cachedShort = { got: cached.read(last, size - 10, new Uint8Array(100)), expected: 10 };
    // Straddle a page boundary and cross into the short final page in one read.
    const big = paths[paths.length - 1];
    const bigAt = 4 * 65_536 - 7;
    const bigWant = pattern(SIZES[SIZES.length - 1], SIZES.length - 1).slice(bigAt, bigAt + 40);
    const bigInto = new Uint8Array(40);
    const bigGot = cached.read(big, bigAt, bigInto);
    const bigSame = bigGot === 40 && bigWant.every((v, i) => v === bigInto[i]);
    // A read that starts past the end reports nothing, not zeros.
    const past = cached.read(big, SIZES[SIZES.length - 1] + 5, new Uint8Array(8));
    const cachedInto = new Uint8Array(16);
    cached.read(paths[3], at, cachedInto);
    const cachedSame = want.every((v, i) => v === cachedInto[i]);
    cached.close();

    self.postMessage({
      written: entries.length,
      opened,
      reused,
      verify,
      shortRead,
      offsetRead: {
        ok: same,
        detail: same ? "" : `at ${at}: want ${[...want].slice(0, 8)} got ${[...into].slice(0, 8)}`,
      },
      cached: {
        opened: cachedOpened,
        verify: cachedVerify,
        shortRead: cachedShort,
        straddle: { ok: bigSame, got: bigGot },
        pastEnd: past,
        offsetRead: cachedSame,
      },
    });
  } catch (e) {
    self.postMessage({ error: String((e && e.stack) || e) });
  }
};
