// The REAL import, run for a bounded number of bytes, twice: once reading the way the
// import used to and once through the read-ahead window.
//
// The other measurement on this page times read, crypto and write as separate
// primitives, which says where the time COULD go. This one runs the actual thing -
// the same `ingest_probe` and `ingest_import` the product calls, the same pkg CTR and
// PFS page decrypt, the same OPFS sync handles - and reports what the import will
// really do on this device. It just stops early, so confirming the speed of a 3 GB
// title does not cost a 3 GB import.
//
// HOW IT STOPS: `DumpSink::write` is allowed to fail, and a failure from the JS side
// comes back through `Reflect::apply` as an `Err`, which unwinds the import cleanly.
// So the sink throws once its budget is spent and this catches that one string. No
// Rust change, and everything up to the stop ran the untouched product path.
//
// THE ARMS RUN OLD FIRST. A second pass over the same bytes can only be helped by a
// cache, so putting the new path second makes the comparison conservative rather than
// flattering. (On the phone that was measured there is no such cache: a warm re-read
// came back at the same 7 MB/s as a cold one.)
//
// Nothing here touches a real title. Output goes to a scratch OPFS directory that is
// deleted afterwards, and it is never read back.
//
// In:  { entries: [{ path, file }] }
// Out: { type: "log", line } ... { type: "done", text }

import init, { ingest_probe, ingest_import } from "../pkg/vitaslop_web.js";
import { encodeName } from "../opfs.js";
import { makeFileSource, MIN_SPAN, MAX_SPAN } from "../read-window.js";

const MB = 1024 * 1024;
const BUDGET_MB = 128; // written bytes per arm
const BUDGET_MS = 20000; // ... or this long, whichever comes first
const HANDLES = 768; // outputs pre-opened; the arm ends if the import needs more
const SCRATCH = "vitaslop-bench-scratch";
const STOP = "BENCH_STOP";

const lines = [];
function log(line) {
  lines.push(line);
  self.postMessage({ type: "log", line });
}

/// The product's source, with the cache made switchable: `span` 0 is the OLD path, one
/// provider read for exactly what was asked. `entries` is what the product's own picker
/// produces (`import.js`, `entriesFromFiles`): the container under its own name and the
/// licence under `work.bin`, or every file of a dumped folder under its relative path.

async function scratchDir() {
  const root = await navigator.storage.getDirectory();
  try {
    await root.removeEntry(SCRATCH, { recursive: true });
  } catch {}
  return await root.getDirectoryHandle(SCRATCH, { create: true });
}

async function dropScratch() {
  const root = await navigator.storage.getDirectory();
  try {
    await root.removeEntry(SCRATCH, { recursive: true });
  } catch {}
}

/// One arm: probe, pre-open handles, then import until the budget runs out.
async function arm(entries, span, maxSpan, label) {
  log("");
  log(`--- ${label} ---`);

  const t0 = performance.now();
  let providerReads = 0;
  let providerBytes = 0;
  // WHERE the window refills, so a pattern that thrashes it can be seen rather than
  // guessed at. Only the first few matter: they name the shape.
  const fills = [];
  const src = makeFileSource(entries, {
    span,
    maxSpan,
    onRead: (n) => {
      providerReads += 1;
      providerBytes += n;
    },
    onFill: (at) => {
      if (fills.length < 24) fills.push(at);
    },
  });

  const probe = ingest_probe(src);
  const probeMs = performance.now() - t0;
  const outputs = (probe.outputs || []).slice(0, HANDLES);
  log(`  identify: ${probeMs.toFixed(0)} ms, ${providerReads} reads, ${(providerBytes / MB).toFixed(1)} MB pulled`);
  // A pkg carries its files encrypted under a key that lives in the licence, so without
  // it there is nothing to measure. Say so HERE: letting it run on hits the container
  // read for work.bin and reports "file not in container", which names the symptom and
  // not the thing the person has to do.
  if (probe.missingWorkBin) {
    log("  this pkg needs its work.bin licence. Pick that too and run it again.");
    return null;
  }
  if (!probe.titleId) {
    log("  no param.sfo in these files, so there is nothing to import");
    return null;
  }

  const dir = await scratchDir();
  const handles = new Map();
  const tOpen = performance.now();
  let next = 0;
  await Promise.all(
    Array.from({ length: 16 }, async () => {
      while (next < outputs.length) {
        const p = outputs[next++];
        const fh = await dir.getFileHandle(encodeName(p), { create: true });
        handles.set(p, await fh.createSyncAccessHandle());
      }
    })
  );
  const openMs = performance.now() - tOpen;
  const each = outputs.length ? `${(openMs / outputs.length).toFixed(2)} ms each` : "none to open";
  log(`  ${outputs.length} handles open: ${openMs.toFixed(0)} ms (${each})`);

  // The import proper. Every counter below is reset here so the rate is the import's
  // and not the identify phase's.
  providerReads = 0;
  providerBytes = 0;
  let written = 0;
  let cur = null;
  const tImp = performance.now();
  const sink = {
    begin(path) {
      const h = handles.get(path);
      if (!h) throw STOP; // past the pre-opened set: an honest end to the measurement
      cur = { h, off: 0 };
    },
    write(bytes) {
      cur.off += cur.h.write(bytes, { at: cur.off });
      written += bytes.length;
      if (written >= BUDGET_MB * MB || performance.now() - tImp > BUDGET_MS) throw STOP;
    },
    finish() {
      cur.h.flush();
      cur = null;
    },
  };

  let stopped = false;
  try {
    ingest_import(src, sink, () => {});
  } catch (e) {
    if (!String((e && e.message) || e).includes(STOP)) throw e;
    stopped = true;
  }
  const impMs = performance.now() - tImp;
  for (const h of handles.values()) {
    try {
      h.close();
    } catch {}
  }
  await dropScratch();

  const mbs = written / MB / (impMs / 1000);
  log(`  import: ${(written / MB).toFixed(0)} MB in ${(impMs / 1000).toFixed(1)} s = ${mbs.toFixed(1)} MB/s${stopped ? "" : " (source ran out first)"}`);
  log(`  provider reads: ${providerReads} for ${(providerBytes / MB).toFixed(0)} MB = ${(providerBytes / MB / Math.max(1, providerReads)).toFixed(2)} MB each`);
  if (fills.length > 1) {
    log(`  window refills at MB: ${fills.map((f) => (f / MB).toFixed(1)).join(" ")}`);
  }
  return { mbs, probeMs, openMs };
}

self.onmessage = async (e) => {
  const entries = e.data.entries;
  const biggest = entries.reduce((a, b) => (a && a.file.size > b.file.size ? a : b), null);
  try {
    await init();
    log(`device: ${navigator.userAgent}`);
    log(`files: ${entries.map((x) => `${x.path} ${(x.file.size / MB).toFixed(1)} MB`).join(", ")}`);
    log(`Running the REAL import twice, ${BUDGET_MB} MB or ${BUDGET_MS / 1000}s per arm.`);

    const before = await arm(entries, 0, 0, "OLD: one provider read per request");
    // A failed first arm is a picked-files problem, not a result: running the rest
    // would print the same complaint three times and bury it.
    const after = before && (await arm(entries, MIN_SPAN, MAX_SPAN, "NEW: adaptive windows, 4 way"));
    // Is the shipped window big enough? Same code, twice the sizes. A device whose
    // reads are all fixed cost wants bigger; one that pays per byte wants smaller.
    const bigger = after && (await arm(entries, 2 * MIN_SPAN, 2 * MAX_SPAN, "BIGGER: 4 MB opening, 32 MB cap"));

    if (before && after) {
      log("");
      log("VERDICT");
      log(`  ${before.mbs.toFixed(1)} MB/s -> ${after.mbs.toFixed(1)} MB/s = ${(after.mbs / before.mbs).toFixed(2)}x`);
      const secs = (n) => `${Math.floor(n / 60)}m ${Math.round(n % 60)}s`;
      const gb = biggest.file.size / MB;
      log(`  this ${(gb / 1024).toFixed(1)} GB title: ${secs(gb / before.mbs)} -> ${secs(gb / after.mbs)}`);
      log(`  identify: ${before.probeMs.toFixed(0)} ms -> ${after.probeMs.toFixed(0)} ms`);
      if (bigger) {
        log(`  bigger windows: ${bigger.mbs.toFixed(1)} MB/s (${(bigger.mbs / after.mbs).toFixed(2)}x the shipped size)`);
      }
    }
    self.postMessage({ type: "done", text: lines.join("\n") });
  } catch (err) {
    log(`FAILED: ${String((err && err.stack) || err)}`);
    await dropScratch();
    self.postMessage({ type: "done", text: lines.join("\n") });
  }
};
