// Shared plumbing for the vitaslop-web browser tests: a static file server for
// the built bundle and a Chrome launcher with WebGPU enabled. Used by both the
// cube render test (run.mjs) and the conformance runner (conformance.mjs).

import { chromium } from "playwright";
import { createServer } from "node:http";
import { readFile } from "node:fs/promises";
import { existsSync } from "node:fs";
import { extname, join, dirname } from "node:path";
import { fileURLToPath } from "node:url";

export const here = dirname(fileURLToPath(import.meta.url));
export const webDir = join(here, "..", "web");

const MIME = {
  ".html": "text/html; charset=utf-8",
  ".js": "text/javascript; charset=utf-8",
  ".mjs": "text/javascript; charset=utf-8",
  ".wasm": "application/wasm",
  ".json": "application/json",
  ".css": "text/css; charset=utf-8",
  ".svg": "image/svg+xml",
  ".webmanifest": "application/manifest+json",
  ".png": "image/png",
};

/// Fail early with a clear message if the wasm bundle has not been built.
export function requireBundle() {
  if (!existsSync(join(webDir, "pkg", "vitaslop_web.js"))) {
    console.error("Bundle missing: run ../build.ps1 first (expected web/pkg/vitaslop_web.js).");
    process.exit(2);
  }
}

/// Serve `root` on an ephemeral localhost port. Returns the http.Server.
export function startServer(root, port = 0) {
  const server = createServer(async (req, res) => {
    try {
      const urlPath = decodeURIComponent(req.url.split("?")[0]);
      const rel = urlPath === "/" ? "/index.html" : urlPath;
      const file = join(root, rel);
      if (!file.startsWith(root)) {
        res.writeHead(403).end("forbidden");
        return;
      }
      const body = await readFile(file);
      res.writeHead(200, {
        "content-type": MIME[extname(file)] || "application/octet-stream",
        // A shared WebAssembly.Memory (SharedArrayBuffer) - which the preemptive
        // scheduler imports into every guest instance - is only allowed on a
        // cross-origin-isolated page.
        "Cross-Origin-Opener-Policy": "same-origin",
        "Cross-Origin-Embedder-Policy": "require-corp",
      });
      res.end(body);
    } catch {
      res.writeHead(404).end("not found");
    }
  });
  return new Promise((resolve) => server.listen(port, "127.0.0.1", () => resolve(server)));
}

/// Launch Chrome with WebGPU enabled. Uses the installed Chrome channel by default -
/// Playwright's cached Chromium can lag the pkg version, and we want to test real Chrome
/// anyway.
///
/// These are the CORRECTNESS tests (the cube render and the ARM conformance corpus), so
/// they run headless by default: they compare pixels and pass/fail counts, and a software
/// rasteriser is a legitimate way to produce those. It is not a legitimate way to produce
/// a frame RATE, which is why `game-boot.mjs` runs headed on a real GPU instead. The
/// software fallback stays opt-in here too (ALLOW_SOFTWARE=1), so a box that has lost its
/// GPU says so rather than quietly getting 30x slower.
export function launchChrome() {
  return chromium.launch({
    channel: process.env.PWCHANNEL || "chrome",
    headless: process.env.HEADED ? false : true,
    args: [
      "--enable-unsafe-webgpu",
      "--enable-features=Vulkan",
      "--use-angle=default",
      ...(process.env.ALLOW_SOFTWARE ? ["--enable-unsafe-swiftshader"] : []),
    ],
  });
}

/// A minimal `param.sfo` carrying TITLE_ID and TITLE (the layout `ingest/sfo.rs` reads).
export function sfo(fields) {
  const keys = Object.keys(fields);
  let keyTable = Buffer.alloc(0);
  let dataTable = Buffer.alloc(0);
  const entries = [];
  for (const k of keys) {
    const value = Buffer.from(fields[k] + "\0", "utf8");
    const max = (value.length + 3) & ~3;
    entries.push({ keyOff: keyTable.length, len: value.length, max, dataOff: dataTable.length });
    keyTable = Buffer.concat([keyTable, Buffer.from(k + "\0", "ascii")]);
    dataTable = Buffer.concat([dataTable, value, Buffer.alloc(max - value.length)]);
  }
  while (keyTable.length % 4) keyTable = Buffer.concat([keyTable, Buffer.alloc(1)]);
  const keyTableOff = 20 + entries.length * 16;
  const dataTableOff = keyTableOff + keyTable.length;
  const head = Buffer.alloc(20);
  head.write("\0PSF", 0, "binary");
  head.writeUInt32LE(0x0101, 4);
  head.writeUInt32LE(keyTableOff, 8);
  head.writeUInt32LE(dataTableOff, 12);
  head.writeUInt32LE(entries.length, 16);
  const index = Buffer.alloc(entries.length * 16);
  entries.forEach((e, i) => {
    index.writeUInt16LE(e.keyOff, i * 16);
    index.writeUInt16LE(0x0204, i * 16 + 2);
    index.writeUInt32LE(e.len, i * 16 + 4);
    index.writeUInt32LE(e.max, i * 16 + 8);
    index.writeUInt32LE(e.dataOff, i * 16 + 12);
  });
  return Buffer.concat([head, index, keyTable, dataTable]);
}
