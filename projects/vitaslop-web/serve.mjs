// Play the game in a browser: build the wasm bundle, then serve the play page + a
// game directory over HTTP (with the cross-origin-isolation headers SharedArrayBuffer
// needs) and open it. The emulator runs entirely client-side in a Web Worker - no
// Chrome flags required. Cross-platform (Node only, no shell/OS assumptions).
//
// Usage:
//   node serve.mjs --game <dir> [--port 8080] [--no-build] [--no-open]
//
//   --game <dir>   the extracted app directory (the folder with eboot.bin, sce_pfs, ...)
//   --port <n>     listen port (default 8080)
//   --no-build     skip the wasm rebuild (serve the existing web/pkg)
//   --no-open      do not launch a browser (just print the URL)
import { createServer } from "node:http";
import { spawnSync } from "node:child_process";
import { readFile, readdir, stat } from "node:fs/promises";
import { fileURLToPath } from "node:url";
import { dirname, join, extname, relative, sep } from "node:path";

const crateDir = dirname(fileURLToPath(import.meta.url));
const webDir = join(crateDir, "web");

// --- args ---
const args = process.argv.slice(2);
const opt = (name) => {
  const i = args.indexOf(name);
  return i >= 0 ? args[i + 1] : undefined;
};
const gameDir = opt("--game") || process.env.GAME_DIR;
const port = Number(opt("--port") || process.env.PORT || 8080);
const noBuild = args.includes("--no-build");
const noOpen = args.includes("--no-open");
// --recipe <file>: serve this frame-keyed TAS recipe at /recipe.txt so the page can
// replay a recorded playthrough via play.html?recipe=/recipe.txt (auto-appended below).
const recipeFile = opt("--recipe");

if (!gameDir) {
  console.error("error: --game <dir> is required (the extracted app directory).");
  console.error("usage: node serve.mjs --game <dir> [--port 8080] [--no-build] [--no-open]");
  process.exit(2);
}

// --- build ---
if (!noBuild) {
  console.log("building wasm bundle...");
  const r = spawnSync(process.execPath, [join(crateDir, "build.mjs")], { stdio: "inherit" });
  if (r.status !== 0) {
    console.error("build failed; fix the build or pass --no-build to serve the existing bundle.");
    process.exit(1);
  }
}

const MIME = {
  ".html": "text/html",
  ".js": "text/javascript",
  ".mjs": "text/javascript",
  ".wasm": "application/wasm",
  ".json": "application/json",
  ".css": "text/css",
};

// Recursively list files under `root`, returning forward-slash relative paths.
async function walk(root, dir = root, out = []) {
  for (const name of await readdir(dir)) {
    const full = join(dir, name);
    const s = await stat(full);
    if (s.isDirectory()) await walk(root, full, out);
    else out.push(relative(root, full).split(sep).join("/"));
  }
  return out;
}

const manifest = await walk(gameDir);
const totalMB = (
  (await Promise.all(manifest.map((p) => stat(join(gameDir, p))))).reduce((a, s) => a + s.size, 0) / 1e6
).toFixed(0);
console.log(`game: ${manifest.length} files, ${totalMB} MB in ${gameDir}`);

const server = createServer(async (req, res) => {
  // Cross-origin isolation: required for SharedArrayBuffer (the guest's shared memory).
  const coi = {
    "Cross-Origin-Opener-Policy": "same-origin",
    "Cross-Origin-Embedder-Policy": "require-corp",
  };
  try {
    const url = decodeURIComponent(req.url.split("?")[0]);
    if (url === "/game-manifest.json") {
      res.writeHead(200, { "content-type": "application/json", ...coi });
      return res.end(JSON.stringify(manifest));
    }
    if (url === "/recipe.txt" && recipeFile) {
      res.writeHead(200, { "content-type": "text/plain", ...coi });
      return res.end(await readFile(recipeFile));
    }
    const file = url.startsWith("/game/")
      ? join(gameDir, url.slice("/game/".length))
      : join(webDir, url === "/" ? "/play.html" : url);
    const body = await readFile(file);
    res.writeHead(200, { "content-type": MIME[extname(file)] || "application/octet-stream", ...coi });
    res.end(body);
  } catch {
    res.writeHead(404).end("not found");
  }
});

await new Promise((r) => server.listen(port, "127.0.0.1", r));
const url = `http://127.0.0.1:${port}/${recipeFile ? "?recipe=/recipe.txt" : ""}`;
console.log(`serving at ${url}  (Ctrl+C to stop)`);
if (recipeFile) console.log(`replaying recipe: ${recipeFile}`);

if (!noOpen) {
  // Open the default browser, cross-platform.
  const openers =
    process.platform === "win32"
      ? ["cmd", ["/c", "start", "", url]]
      : process.platform === "darwin"
      ? ["open", [url]]
      : ["xdg-open", [url]];
  const r = spawnSync(openers[0], openers[1], { stdio: "ignore" });
  if (r.error) console.log(`(could not auto-open a browser; open ${url} yourself)`);
}
