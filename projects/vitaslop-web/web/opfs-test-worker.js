// The worker half of the OPFS storage test.
//
// It has to be a worker: `createSyncAccessHandle` exists only in workers, which is also
// why the emulator runs there. Testing the read path from the main thread would test a
// different API than the product uses - or, as it did on the first attempt, simply throw.
import init, { opfs_verify } from "./pkg/vitaslop_web.js";
import { importTitle, openTitleSync, syncReader } from "./opfs.js";

const TITLE = "opfs-test-fixture";

// Sizes chosen against the 997-byte verify chunk: shorter than a chunk, exactly a chunk,
// one over, an awkward multiple, and a large one. A size landing exactly on a boundary
// would hide an off-by-one in the final partial read, so most of these deliberately do
// not.
const SIZES = [500, 997, 998, 4321, 65_537];

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
    });
  } catch (e) {
    self.postMessage({ error: String((e && e.stack) || e) });
  }
};
