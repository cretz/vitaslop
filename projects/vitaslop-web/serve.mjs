// The dev server for playing titles in a browser: build the wasm bundle, discover every
// title on this machine, and serve the player page plus those titles' bytes over HTTP (with
// the cross-origin-isolation headers SharedArrayBuffer needs). The emulator runs entirely
// client-side in a Web Worker - no Chrome flags, no server-side emulation.
//
// Usage:
//   node serve.mjs --games <dir> [--port 8080] [--host 0.0.0.0] [--https] [--no-build] [--open]
//   node serve.mjs --game <dir>  [...]        # one title, the older single-title form
//
//   --games <dir>  a directory of titles; every subtree containing a `vitaslop-dump.txt` is
//                  one title, and its id comes from that file (see `discoverTitles`).
//   --game <dir>   a single title directory (an extracted app dir or a decrypted dump).
//   --recipes <d>  where to look for playable recipes (default: the committed recipe tree).
//   --host <h>     interface to bind. DEFAULT 0.0.0.0, i.e. reachable from other devices on
//                  this network - that is the point (a phone cannot open 127.0.0.1). See the
//                  banner it prints: this exposes the titles' bytes to that network.
//   --port <n>     listen port (default 8080).
//   --https        serve TLS with a self-signed certificate, generated with openssl on first
//                  use. Needed for AUDIO from another device: SharedArrayBuffer requires a
//                  cross-origin-isolated SECURE context, and a plain http:// LAN address is
//                  not one. The page says so either way rather than running silently.
//   --no-build     skip the wasm rebuild (serve the existing web/pkg).
//   --open         launch a local browser at the URL.
import { createServer as createHttpServer } from "node:http";
import { createServer as createHttpsServer } from "node:https";
import { spawnSync } from "node:child_process";
import { readFile, readdir, stat, mkdir, writeFile, appendFile } from "node:fs/promises";
import { existsSync } from "node:fs";
import { networkInterfaces, tmpdir } from "node:os";
import { fileURLToPath } from "node:url";
import { dirname, join, extname, relative, sep, basename } from "node:path";

const crateDir = dirname(fileURLToPath(import.meta.url));
const webDir = join(crateDir, "web");
const projects = dirname(crateDir);
// Device diagnostics land in the SCRATCH area, never in the repo - see the sink endpoints in
// the handler, and `no-repo-scratch-artifacts`. `projects` is <repo>/projects, so the repo root
// is one level up and the scratch area is one level above THAT - writing to
// `<repo>/working-area` (one `dirname` short) drops 44 files inside the git tree, which is
// exactly what this comment exists to prevent and exactly what it did on the first attempt.
const diagDir = join(dirname(dirname(projects)), "working-area", "device-diag");

// --- args ---
const args = process.argv.slice(2);
const opt = (name) => {
  const i = args.indexOf(name);
  return i >= 0 ? args[i + 1] : undefined;
};
const gamesDir = opt("--games") || process.env.GAMES_DIR;
const singleGame = opt("--game") || process.env.GAME_DIR;
const recipesDir = opt("--recipes") || join(projects, "vitaslop-gamerun-recipes", "recipes");
const host = opt("--host") || process.env.HOST || "0.0.0.0";
const port = Number(opt("--port") || process.env.PORT || 8080);
const useHttps = args.includes("--https");
const noBuild = args.includes("--no-build");
const doOpen = args.includes("--open");

if (!gamesDir && !singleGame) {
  console.error("error: pass --games <dir> (a directory of titles) or --game <dir> (one title).");
  console.error("usage: node serve.mjs --games <dir> [--port 8080] [--host 0.0.0.0] [--https]");
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
  ".png": "image/png",
};

/// Recursively list files under `root`, returning forward-slash relative paths.
async function walk(root, dir = root, out = []) {
  for (const name of await readdir(dir)) {
    const full = join(dir, name);
    const s = await stat(full);
    if (s.isDirectory()) await walk(root, full, out);
    else out.push(relative(root, full).split(sep).join("/"));
  }
  return out;
}

/// The title id a directory holds, or null if it is not a title root.
///
/// # Why this is read out of the container and not configured
/// A title's id is IN it: the decrypted-dump manifest carries a
/// `content_id=<region>-<title id>_00-<...>` line, and the id is the middle field. Reading it means no table mapping directory names to ids
/// can go stale, and - the reason that matters - the id is what the recipe tree is keyed by
/// and what OPFS stores the import under, so a wrong one would silently play the wrong
/// recipe or re-import a title that is already stored.
async function titleIdOf(dir) {
  const manifest = join(dir, "vitaslop-dump.txt");
  if (!existsSync(manifest)) return null;
  const text = await readFile(manifest, "utf8");
  const m = /content_id=[^-\s]+-([A-Z]{4}\d{5})_/.exec(text);
  return m ? m[1] : null;
}

/// Every title under `root`, searched to a small depth.
///
/// A title lives at `<root>/<game>/extracted`, `<root>/<game>/dump` or `<root>/<game>/app`
/// depending on how it was produced, and one game directory can hold BOTH an extracted app
/// tree and a decrypted dump - only the dump is readable through the browser's storage path,
/// and it is the one carrying the manifest, so "the directory with the manifest" picks the
/// right one without a special case per title.
async function discoverTitles(root) {
  const found = [];
  const visit = async (dir, depth) => {
    const id = await titleIdOf(dir);
    if (id) {
      found.push({ id, dir });
      return; // a title root is not searched further
    }
    if (depth === 0) return;
    for (const name of await readdir(dir)) {
      const full = join(dir, name);
      if ((await stat(full)).isDirectory()) await visit(full, depth - 1);
    }
  };
  await visit(root, 3);
  return found;
}

/// The friendly names in the committed registry, by title id. Provenance for humans; a title
/// missing from it still plays, listed by its id.
async function readRegistry() {
  const path = join(projects, "vitaslop-gamerun-recipes", "games.toml");
  const names = {};
  try {
    const text = await readFile(path, "utf8");
    let id = null;
    for (const line of text.split(/\r?\n/)) {
      const mid = /^\s*id\s*=\s*"([^"]+)"/.exec(line);
      const mname = /^\s*name\s*=\s*"([^"]+)"/.exec(line);
      if (mid) id = mid[1];
      else if (mname && id) {
        names[id] = mname[1];
        id = null;
      }
    }
  } catch {
    /* no registry: ids are names enough */
  }
  return names;
}

/// The recipes for a title id: every `*.recipe` under `<recipes>/<id>/`, with the `@title`
/// line as its description so the picker says what a recipe DOES rather than only its filename.
async function recipesFor(id) {
  const dir = join(recipesDir, id);
  if (!existsSync(dir)) return [];
  const out = [];
  for (const name of await readdir(dir)) {
    if (!name.endsWith(".recipe")) continue;
    const text = await readFile(join(dir, name), "utf8");
    const m = /^@title\s+(.*)$/m.exec(text);
    out.push({ name, description: m ? m[1].trim() : "" });
  }
  out.sort((a, b) => a.name.localeCompare(b.name));
  return out;
}

// --- discover ---
const registry = await readRegistry();
const roots = gamesDir ? await discoverTitles(gamesDir) : [];
if (singleGame) {
  const id = (await titleIdOf(singleGame)) || basename(singleGame);
  if (!roots.some((t) => t.id === id)) roots.push({ id, dir: singleGame });
}
if (roots.length === 0) {
  console.error(`error: no titles found (looked for a vitaslop-dump.txt under ${gamesDir || singleGame}).`);
  process.exit(2);
}

/// One entry per playable title: its files (for the manifest the page imports into storage),
/// its size, and its recipes.
const titles = [];
for (const { id, dir } of roots) {
  const files = await walk(dir);
  const bytes = (await Promise.all(files.map((p) => stat(join(dir, p))))).reduce((a, s) => a + s.size, 0);
  titles.push({
    id,
    name: registry[id] || id,
    dir,
    files,
    bytes,
    recipes: await recipesFor(id),
  });
  console.log(
    `title ${id} (${registry[id] || "unregistered"}): ${files.length} files, ` +
      `${(bytes / 1e6).toFixed(0)} MB, ${(await recipesFor(id)).length} recipes  [${dir}]`
  );
}
const byId = new Map(titles.map((t) => [t.id, t]));

/// The listing the page renders. Deliberately WITHOUT the file lists (one title is thousands
/// of paths) - the page asks for those per title, once, when it is about to import one.
const listing = titles.map((t) => ({
  id: t.id,
  name: t.name,
  files: t.files.length,
  mb: Math.round(t.bytes / 1e6),
  recipes: t.recipes,
}));

/// A self-signed certificate for `--https`, generated once into the OS temp directory (never
/// into the repo). It exists so another device on this network gets a SECURE context, which is
/// what `SharedArrayBuffer` - and therefore audio - requires; the browser will still warn
/// about the certificate, which has to be accepted once per device.
async function tlsOptions() {
  const dir = join(tmpdir(), "vitaslop-certs");
  const key = join(dir, "key.pem");
  const cert = join(dir, "cert.pem");
  if (!existsSync(key) || !existsSync(cert)) {
    await mkdir(dir, { recursive: true });
    const ips = lanAddresses();
    const san = ["DNS:localhost", "IP:127.0.0.1", ...ips.map((a) => `IP:${a}`)].join(",");
    console.log(`generating a self-signed certificate for ${san}`);
    const r = spawnSync(
      "openssl",
      [
        "req", "-x509", "-newkey", "rsa:2048", "-nodes",
        "-keyout", key, "-out", cert, "-days", "365",
        "-subj", "/CN=vitaslop",
        "-addext", `subjectAltName=${san}`,
      ],
      { stdio: "inherit" }
    );
    if (r.status !== 0) {
      // Named, not swallowed: without openssl the honest outcome is plain http (and no
      // audio), and a silent fallback to that would look like an audio bug later.
      throw new Error(
        "openssl is not available, so --https cannot generate a certificate. " +
          "Run without --https (the run works, but has no audio: SharedArrayBuffer needs a secure context)."
      );
    }
    await writeFile(join(dir, "README.txt"), "Self-signed certs for the vitaslop dev server. Safe to delete.\n");
  }
  return { key: await readFile(key), cert: await readFile(cert) };
}

/// Every non-loopback IPv4 address of this machine, so the banner can print a URL a phone can
/// actually open instead of leaving the user to find it.
function lanAddresses() {
  const out = [];
  for (const list of Object.values(networkInterfaces())) {
    for (const ni of list || []) {
      if (ni.family === "IPv4" && !ni.internal) out.push(ni.address);
    }
  }
  return out;
}

const handler = async (req, res) => {
  // Cross-origin isolation: required for SharedArrayBuffer (the audio ring).
  const coi = {
    "Cross-Origin-Opener-Policy": "same-origin",
    "Cross-Origin-Embedder-Policy": "require-corp",
    // A LAN device caches aggressively and then plays a stale bundle after a rebuild, which
    // looks exactly like a change that did not work.
    "Cache-Control": "no-store",
  };
  try {
    const url = new URL(req.url, "http://x");
    const path = decodeURIComponent(url.pathname);

    // >>> THE DEVICE DIAGNOSTICS SINK.
    //
    // Getting a capture off a phone meant the user hand-copying a 4 KB dump out of an on-screen
    // panel, screenshotting it, and pasting it - per screen, per attempt. That is a tax on the
    // person doing the one thing this project cannot do for itself (run on the target device),
    // and it makes anyone reluctant to take the SECOND capture, which is the one that turns an
    // observation into an A/B.
    //
    // Two endpoints, both dev-server only:
    //   GET  /diag-sink   - capability probe. The page streams ONLY if this answers, so a page
    //                       served from anywhere else (the product, static hosting) never phones
    //                       home and needs no flag to stop it. See
    //                       [[vitaslop-web-is-the-product-not-the-tool]].
    //   POST /diag        - one snapshot of the panel, appended under working-area.
    if (path === "/diag-sink") {
      res.writeHead(200, { "content-type": "text/plain", ...coi });
      return res.end("ok");
    }
    //   POST /diag        - one snapshot of the panel, appended under working-area.
    //   POST /diag-shot   - the canvas at that same moment, as a PNG beside it.
    if ((path === "/diag" || path === "/diag-shot") && req.method === "POST") {
      const chunks = [];
      for await (const c of req) chunks.push(c);
      const body = Buffer.concat(chunks);
      await mkdir(diagDir, { recursive: true });
      const safe = (s, d) => (s || d).replace(/[^A-Za-z0-9_-]/g, "").slice(0, 64);
      const id = safe(url.searchParams.get("run"), "unknown");
      const seq = safe(url.searchParams.get("seq"), "0000");
      const tag = safe(url.searchParams.get("tag"), "tick");
      if (path === "/diag-shot") {
        // One file per shot: a screenshot is only useful next to the dump that describes the
        // same moment, so the seq and tag are the join key between the two.
        await writeFile(join(diagDir, `${id}-${seq}-${tag}.png`), body);
      } else {
        // One file per RUN, appended - so the history is in order in one place and two devices
        // at once do not interleave.
        const stamp = new Date().toISOString().replace("T", " ").slice(0, 19);
        await appendFile(
          join(diagDir, `${id}.txt`),
          `\n===== ${stamp}  seq ${seq}  ${tag} =====\n${body.toString("utf8")}\n`
        );
      }
      res.writeHead(204, coi);
      return res.end();
    }
    if (path === "/titles.json") {
      res.writeHead(200, { "content-type": "application/json", ...coi });
      return res.end(JSON.stringify(listing));
    }
    // The file list for one title. `?title=` names it; with a single title served the
    // parameter is optional, which is what keeps the older pages (play.html, game.html and
    // the e2e harness's own copies) working unchanged.
    if (path === "/game-manifest.json") {
      const t = byId.get(url.searchParams.get("title")) || (titles.length === 1 ? titles[0] : null);
      if (!t) {
        res.writeHead(400, { "content-type": "text/plain", ...coi });
        return res.end("game-manifest.json needs ?title=<TITLE_ID> when more than one title is served");
      }
      res.writeHead(200, { "content-type": "application/json", ...coi });
      return res.end(JSON.stringify(t.files));
    }
    if (path === "/recipe") {
      const t = byId.get(url.searchParams.get("title"));
      const name = url.searchParams.get("name") || "";
      // A traversal-proof name: recipes are picked from a list this server produced, so
      // anything not IN that list is a request nobody meant to make.
      if (!t || !t.recipes.some((r) => r.name === name)) {
        res.writeHead(404, { "content-type": "text/plain", ...coi });
        return res.end("no such recipe");
      }
      res.writeHead(200, { "content-type": "text/plain", ...coi });
      return res.end(await readFile(join(recipesDir, t.id, name)));
    }
    // Title bytes. `/game/<TITLE_ID>/<path>` in multi-title mode; `/game/<path>` still works
    // when a single title is served.
    if (path.startsWith("/game/")) {
      const rest = path.slice("/game/".length);
      const slash = rest.indexOf("/");
      const maybeId = slash > 0 ? rest.slice(0, slash) : "";
      const t = byId.get(maybeId) || (titles.length === 1 ? titles[0] : null);
      if (!t) {
        res.writeHead(404, { ...coi }).end("unknown title");
        return;
      }
      const rel = byId.has(maybeId) ? rest.slice(slash + 1) : rest;
      // Only paths this server listed. A title's manifest is the allowlist.
      if (!t.files.includes(rel)) {
        res.writeHead(404, { ...coi }).end("not in this title's manifest");
        return;
      }
      const body = await readFile(join(t.dir, rel));
      res.writeHead(200, { "content-type": "application/octet-stream", ...coi });
      return res.end(body);
    }

    const file = join(webDir, path === "/" ? "/live.html" : path);
    let body = await readFile(file);
    // >>> THE DIAG SINK ANNOUNCES ITSELF IN THE PAGE, so the page never has to ASK.
    //
    // The page used to probe `GET /diag-sink` at load; on the product (static hosting) that
    // probe is a 404, and a 404 is a red line in an otherwise EMPTY console on every load.
    // A dev server is the only thing that has the sink, so it is the only thing that can
    // say so - one global stamped into the served HTML, read by the page instead of fetched.
    if (extname(file) === ".html") {
      body = body.toString("utf8").replace("<head>", "<head><script>window.__vitaslopDiagSink = true;</script>");
    }
    res.writeHead(200, { "content-type": MIME[extname(file)] || "application/octet-stream", ...coi });
    res.end(body);
  } catch {
    res.writeHead(404, { ...coi }).end("not found");
  }
};

const server = useHttps ? createHttpsServer(await tlsOptions(), handler) : createHttpServer(handler);
// A busy port is an ERROR naming itself, never a silent fallback to another one: the browser
// keys its stored copy of a title by ORIGIN, so a different port re-imports the whole title.
server.on("error", (e) => {
  if (e.code === "EADDRINUSE") {
    console.error(`error: port ${port} is already in use. Stop that server or pass --port <n>.`);
    process.exit(1);
  }
  throw e;
});
await new Promise((r) => server.listen(port, host, r));

const scheme = useHttps ? "https" : "http";
console.log("");
console.log(`serving ${titles.length} title(s) at:`);
console.log(`  ${scheme}://localhost:${port}/`);
for (const a of lanAddresses()) console.log(`  ${scheme}://${a}:${port}/     <- open this on your phone`);
if (host === "0.0.0.0") {
  console.log("");
  console.log("NOTE bound to 0.0.0.0: any device on this network can reach these titles' bytes.");
  console.log("     Pass --host 127.0.0.1 to keep it local.");
}
if (!useHttps) {
  console.log("");
  console.log("NOTE plain http from another device is not a SECURE context, so SharedArrayBuffer");
  console.log("     is unavailable and the run will be SILENT (video and input are unaffected).");
  console.log("     Pass --https for audio; the browser will ask you to accept the certificate.");
}
console.log("");
console.log("Ctrl+C to stop.");

if (doOpen) {
  const url = `${scheme}://localhost:${port}/`;
  const openers =
    process.platform === "win32"
      ? ["cmd", ["/c", "start", "", url]]
      : process.platform === "darwin"
      ? ["open", [url]]
      : ["xdg-open", [url]];
  const r = spawnSync(openers[0], openers[1], { stdio: "ignore" });
  if (r.error) console.log(`(could not auto-open a browser; open ${url} yourself)`);
}
