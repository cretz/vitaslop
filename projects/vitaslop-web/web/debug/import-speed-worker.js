// Where the import's time goes on THIS device, measured in the worker the import
// itself runs in.
//
// The import is one synchronous call that reads picked-file byte ranges with
// `FileReaderSync`, decrypts every byte twice (pkg AES-128-CTR, then PFS
// AES-128-CBC-CTS per 0x8000 page) and hashes every byte once (HMAC-SHA1 per page,
// the integrity check), then writes through an OPFS sync access handle. A phone
// reported 4 MB/s against this desktop's 28, and a rate alone cannot say which of
// those three is the cost. So each is timed on its own here:
//
//   READ    the picked file at several chunk sizes. The import uses 1 MB (`CHUNK` in
//           stream.rs). A picked file on Android arrives through a content provider,
//           so a per-read round trip would show up as a rate that CLIMBS with chunk
//           size. A rate that is flat in chunk size is the medium, not the calls.
//   CRYPTO  the same primitives through WebCrypto, which is hardware AES on the
//           phone. Today's cost is the software fixslice AES that the `aes` crate
//           compiles to on wasm32 (wasm has no AES instruction and simd128 does not
//           help it). These numbers are the CEILING a WebCrypto rewrite could buy,
//           and they are measured the way the import would have to use it: per page.
//   WRITE   an OPFS sync access handle, 1 MB at a time, plus the cost of OPENING
//           handles - the import opens one per output file, ~1,300 on a big title,
//           all before it writes a byte, and that is the `preparing storage` stage.
//
// Nothing here touches the real title directory: writes go to a scratch OPFS
// directory that is deleted afterwards.
//
// In:  { file }                  a File the user picked
// Out: { type: "log", line } ... { type: "done", text }

import init, { crypto_bench } from "../pkg/vitaslop_web.js";

const MB = 1024 * 1024;
const PAGE = 0x8000; // the PFS sector, and so the unit of AES-CBC and HMAC work

/// Time-bounded loops keep the whole run near a minute on a slow device: every phase
/// stops at whichever comes first, its byte budget or its seconds.
const READ_BUDGET_MB = 96;
const READ_BUDGET_MS = 6000;
const CRYPTO_MB = 24;
const CRYPTO_BUDGET_MS = 4000;
const WRITE_BUDGET_MB = 96;
const WRITE_BUDGET_MS = 6000;
const HANDLES = 200; // opened, timed, then scaled to the ~1,300 a real title wants

const lines = [];
function log(line) {
  lines.push(line);
  self.postMessage({ type: "log", line });
}

const rate = (bytes, ms) => (ms <= 0 ? "inf" : (bytes / MB / (ms / 1000)).toFixed(1));
const fmt = (bytes, ms) => `${(bytes / MB).toFixed(0)} MB in ${ms.toFixed(0)} ms = ${rate(bytes, ms)} MB/s`;

// ============================ read ============================

/// Sequential read at one chunk size, starting at `from` so a later pass does not
/// simply re-read what an earlier one warmed. Returns [bytes, ms].
function readPass(file, chunk, from) {
  const reader = new FileReaderSync();
  const t0 = performance.now();
  let off = from;
  let got = 0;
  while (off < file.size && got < READ_BUDGET_MB * MB && performance.now() - t0 < READ_BUDGET_MS) {
    const end = Math.min(file.size, off + chunk);
    const buf = reader.readAsArrayBuffer(file.slice(off, end));
    got += buf.byteLength;
    off = end;
  }
  return [got, performance.now() - t0];
}

function readPhase(file) {
  log(`file: ${file.name} ${(file.size / MB).toFixed(1)} MB, type "${file.type || "none"}"`);
  log("");
  log("READ (FileReaderSync, the call the import uses)");

  // The FIRST read of all: a provider that has to do work to hand over the file
  // pays it here, and a first chunk far slower than the rate that follows is that
  // cost rather than a slow medium.
  const reader = new FileReaderSync();
  const t0 = performance.now();
  reader.readAsArrayBuffer(file.slice(0, MB));
  log(`  first 1 MB (cold): ${(performance.now() - t0).toFixed(0)} ms`);

  let spread = 0;
  const rates = [];
  for (const mb of [1, 4, 16]) {
    // Each size starts a sixth of the file further in, so none of them reads a range
    // an earlier pass has already pulled through the OS cache.
    const from = Math.min(file.size - 1, Math.floor((file.size / 6) * rates.length + 1));
    const [bytes, ms] = readPass(file, mb * MB, from);
    rates.push(Number(rate(bytes, ms)));
    log(`  ${String(mb).padStart(2)} MB chunks: ${fmt(bytes, ms)}`);
  }
  // Re-read the range the 1 MB pass just read: if this is much faster, the file is
  // being cached and the numbers above are a cold-provider cost, not the medium.
  const [wb, wms] = readPass(file, MB, Math.min(file.size - 1, 1));
  log(`  1 MB chunks, warm: ${fmt(wb, wms)}`);
  spread = Math.max(...rates) / Math.max(0.01, Math.min(...rates));
  log(`  spread across chunk sizes: ${spread.toFixed(2)}x  (>1.5x means per-read overhead, not the medium)`);
  return rates[0];
}

// ============================ crypto ============================

async function timed(ms_budget, fn) {
  const t0 = performance.now();
  let bytes = 0;
  while (performance.now() - t0 < ms_budget) bytes += await fn();
  return [bytes, performance.now() - t0];
}

async function cryptoPhase() {
  log("");
  log("CRYPTO through WebCrypto (hardware AES: the ceiling a rewrite could buy)");
  if (!self.crypto || !self.crypto.subtle) {
    log("  crypto.subtle is NOT available here, so this page is not on a secure origin.");
    return null;
  }
  const buf = new Uint8Array(CRYPTO_MB * MB);
  crypto.getRandomValues(buf.subarray(0, 65536));
  const raw = new Uint8Array(16);
  crypto.getRandomValues(raw);
  const iv = new Uint8Array(16);

  const ctrKey = await crypto.subtle.importKey("raw", raw, "AES-CTR", false, ["encrypt", "decrypt"]);
  const cbcKey = await crypto.subtle.importKey("raw", raw, "AES-CBC", false, ["encrypt", "decrypt"]);
  const macKey = await crypto.subtle.importKey("raw", raw, { name: "HMAC", hash: "SHA-1" }, false, ["sign"]);

  // pkg layer: one CTR pass over the whole data region, done in big blocks.
  let [b, ms] = await timed(CRYPTO_BUDGET_MS, async () => {
    await crypto.subtle.decrypt({ name: "AES-CTR", counter: iv, length: 64 }, ctrKey, buf);
    return buf.length;
  });
  const ctr = Number(rate(b, ms));
  log(`  AES-128-CTR, ${CRYPTO_MB} MB blocks: ${fmt(b, ms)}`);

  // PFS layer: CBC per 0x8000 page. Per-page is how the import must call it, and a
  // per-call cost that swamps the work is the thing that would kill this rewrite.
  const page = buf.subarray(0, PAGE);
  [b, ms] = await timed(CRYPTO_BUDGET_MS, async () => {
    await crypto.subtle.encrypt({ name: "AES-CBC", iv }, cbcKey, page);
    return PAGE;
  });
  const cbcPage = Number(rate(b, ms));
  log(`  AES-128-CBC, per 32 KB page:  ${fmt(b, ms)}`);

  [b, ms] = await timed(CRYPTO_BUDGET_MS, async () => {
    await crypto.subtle.encrypt({ name: "AES-CBC", iv }, cbcKey, buf);
    return buf.length;
  });
  log(`  AES-128-CBC, ${CRYPTO_MB} MB blocks: ${fmt(b, ms)}   (what batching pages would buy)`);

  // The integrity check: HMAC-SHA1 over each page's ciphertext.
  [b, ms] = await timed(CRYPTO_BUDGET_MS, async () => {
    await crypto.subtle.sign("HMAC", macKey, page);
    return PAGE;
  });
  const macPage = Number(rate(b, ms));
  log(`  HMAC-SHA1, per 32 KB page:    ${fmt(b, ms)}`);

  // The chain as the import runs it: CTR then CBC then HMAC over every byte.
  const chain = 1 / (1 / ctr + 1 / cbcPage + 1 / macPage);
  log(`  => all three per byte, per page: ${chain.toFixed(1)} MB/s`);
  return chain;
}

// ============================ the crypto we ship ============================

/// The same three primitives, through the REAL Rust functions the import calls,
/// compiled to wasm. WebCrypto's numbers above are what a rewrite could reach; these
/// are where it stands today, and the ratio between them is the size of the prize.
async function rustCryptoPhase(webChain) {
  log("");
  log("CRYPTO as the import does it today (Rust, wasm, software AES)");
  await init();
  const r = crypto_bench(1500);
  log(`  AES-128-CTR (pkg layer):      ${r.pkgCtr.toFixed(1)} MB/s`);
  log(`  HMAC-SHA1 per page:           ${r.pfsHmac.toFixed(1)} MB/s`);
  log(`  AES-128-CBC per page:         ${r.pfsCbc.toFixed(1)} MB/s`);
  log(`  all three chained:            ${r.chain.toFixed(1)} MB/s`);
  if (webChain) log(`  => WebCrypto would be ${(webChain / r.chain).toFixed(1)}x this`);
  return r.chain;
}

// ============================ write ============================

async function writePhase() {
  log("");
  log("WRITE (OPFS sync access handle, the call the import uses)");
  const root = await navigator.storage.getDirectory();
  try {
    await root.removeEntry("vitaslop-speed-scratch", { recursive: true });
  } catch {}
  const dir = await root.getDirectoryHandle("vitaslop-speed-scratch", { create: true });
  try {
    // Streaming write into one file, 1 MB at a time, which is what a big title's
    // bytes actually do.
    const fh = await dir.getFileHandle("bulk.bin", { create: true });
    const h = await fh.createSyncAccessHandle();
    const chunk = new Uint8Array(MB);
    const t0 = performance.now();
    let off = 0;
    while (off < WRITE_BUDGET_MB * MB && performance.now() - t0 < WRITE_BUDGET_MS) {
      off += h.write(chunk, { at: off });
    }
    h.flush();
    const ms = performance.now() - t0;
    h.close();
    log(`  1 MB chunks: ${fmt(off, ms)}`);

    // The `preparing storage` stage: one handle per output file, opened up front.
    const t1 = performance.now();
    const handles = [];
    for (let i = 0; i < HANDLES; i++) {
      const f = await dir.getFileHandle(`h${i}.bin`, { create: true });
      handles.push(await f.createSyncAccessHandle());
    }
    const openMs = performance.now() - t1;
    for (const h2 of handles) h2.close();
    const per = openMs / HANDLES;
    log(`  opening ${HANDLES} handles: ${openMs.toFixed(0)} ms = ${per.toFixed(2)} ms each`);
    log(`  => a 1,300 file title would spend ${((per * 1300) / 1000).toFixed(1)} s in "preparing storage"`);
  } finally {
    try {
      await root.removeEntry("vitaslop-speed-scratch", { recursive: true });
    } catch {}
  }
}

// ============================ run ============================

self.onmessage = async (e) => {
  const file = e.data.file;
  try {
    log(`device: ${navigator.userAgent}`);
    log(`cores: ${navigator.hardwareConcurrency || "?"}`);
    try {
      const est = await navigator.storage.estimate();
      log(`storage: ${(est.usage / 1e6) | 0} MB used of ${(est.quota / 1e6) | 0} MB quota`);
    } catch {}

    const readMbs = readPhase(file);
    const chain = await cryptoPhase();
    const rustChain = await rustCryptoPhase(chain);
    await writePhase();

    log("");
    log("WHAT THIS MEANS");
    // The import's own rate is the fourth number, and the user already has it from the
    // import screen. Everything the import does that is NOT one of the three phases
    // above is software crypto, so the gap between its rate and the slowest number here
    // is the size of the prize.
    log(`  Read at the import's 1 MB chunk size: ${readMbs} MB/s.`);
    if (chain) log(`  Hardware crypto would run the whole per-byte chain at ${chain.toFixed(1)} MB/s.`);
    if (rustChain) log(`  The crypto we ship runs it at ${rustChain.toFixed(1)} MB/s.`);
    log(`  For reference this desktop measures 257 read, 296 write, 346 crypto, and imports at 28 MB/s.`);
    log(`  Compare all of these against the rate the import screen itself reports on this device.`);
    self.postMessage({ type: "done", text: lines.join("\n") });
  } catch (err) {
    log(`FAILED: ${String((err && err.stack) || err)}`);
    self.postMessage({ type: "done", text: lines.join("\n") });
  }
};
