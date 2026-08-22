// Boot the REAL retail title in the browser through the JSPI preemptive
// scheduler, driven by Playwright. Serves the extracted container dir + the wasm
// bundle (cross-origin isolated for shared memory), lets the page fetch the files and
// call run_game, and reports the boot result.
//
// This runs the real installed Chrome on the real GPU. **USE HEADLESS=1** - it is the
// polite mode (it never steals the foreground) and it is NOT a software run.
//
// MEASURED 2026-08-07, one retail title, same recipe, one variable changed: headed and
// HEADLESS=1 both report `adapter: vendor=nvidia arch=blackwell | GPU`, both reach the
// same frame, both at the same rate. Modern headless Chrome takes the GPU when asked, and
// `--enable-gpu` below asks; the older claim that headless here has no GPU and falls back
// to SwiftShader was left over from a build where that was true, and cost a session's
// worth of foreground windows before anyone re-measured it.
//
// It is still never ASSUMED: SwiftShader is opt-in (ALLOW_SOFTWARE=1), the page refuses to
// render on a software adapter, and this harness refuses to publish a rate from one - so
// the adapter line is a thing the run PROVES, in either mode. Go headed only when you need
// to watch the window.
//
// Env:  GAME_DIR     the extracted app dir (default: $VITASLOP_GAME_DIR)
//       MAX_FRAMES   display flips to run to
//       MAX_ROUNDS   scheduler round cap (default 50_000_000)
//       TARGET_FRAME the flip to screenshot at
//       RECIPE       scripted-input recipe path
//       KNOBS        {"VITASLOP_GXP_LIVE":"1",...} - the browser has no environment
//       WAIT_MS      how long to wait for TARGET_FRAME
//       SHOT_DIR     where the screenshot and summary land
//       HEADLESS=1   run without a window - the DEFAULT CHOICE; still the real GPU (above)
//       ALLOW_SOFTWARE=1  accept a software adapter instead of failing
//       HIDE_SHOW_MS background the tab every N ms (HIDE_FOR_MS, HIDE_FROM_FRAME) and shoot
//                    the frame right after each restore - a hidden tab is a DIFFERENT
//                    environment, and some defects are only reported across that boundary
import { chromium } from "playwright";
import { createServer } from "node:http";
import { readFile, readdir, stat, mkdir } from "node:fs/promises";
import { createWriteStream } from "node:fs";
import { join, extname, relative, sep } from "node:path";
import { fileURLToPath } from "node:url";
import { startProcMon } from "./procmon.mjs";

const here = join(fileURLToPath(import.meta.url), "..");
const webDir = join(here, "..", "web");
const gameDir = process.env.GAME_DIR || process.env.VITASLOP_GAME_DIR || "";
// Live run: render frame-by-frame through the real GXM->WebGPU renderer, driven by a
// scripted recipe that navigates the touch front-end to the Tutorial level. Run past
// the frame the level appears (~175) so the screenshot lands on live gameplay.
const maxFrames = Number(process.env.MAX_FRAMES || 260);
const maxRounds = Number(process.env.MAX_ROUNDS || 50_000_000);
// The display flip by which the tutorial level is on screen; the harness waits for the
// live loop to pass it before screenshotting.
const targetFrame = Number(process.env.TARGET_FRAME || 178);
// The scripted-input recipe (frame-keyed touch taps). Override with RECIPE=<path>, or
// RECIPE="" for a live-input session. No default: pass a recipe from the
// vitaslop-gamerun-recipes crate for the title under test.
const recipePath = process.env.RECIPE ?? "";
const recipe = recipePath ? await readFile(recipePath, "utf8").catch(() => "") : "";

const MIME = { ".html": "text/html", ".js": "text/javascript", ".wasm": "application/wasm", ".json": "application/json" };

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

// Memory is sampled per PROCESS by `procmon.mjs` (see the note at the top of that file
// for why a sum over Chrome processes and a 15-second cadence cannot answer the question
// that decides the diagnosis).

async function main() {
  // The run's output directory, needed BEFORE the browser starts: the telemetry files are
  // opened first so a run that dies during setup still leaves its measurements behind.
  const shotDir = process.env.SHOT_DIR || join(here, "screenshots");
  await mkdir(shotDir, { recursive: true });

  // Per-process memory/CPU telemetry. On by default at 250 ms - it costs one long-lived
  // PowerShell child and a few kB per second, and a run whose failure is invisible without
  // it is a run that has to be repeated. MEM_SAMPLE_MS=0 turns it off.
  const sampleMs = Number(process.env.MEM_SAMPLE_MS ?? 250);
  const procmon = sampleMs > 0 ? startProcMon({ intervalMs: sampleMs, outPath: join(shotDir, "mem.csv") }) : null;
  if (procmon) console.log(`[game] per-process telemetry every ${sampleMs} ms -> ${join(shotDir, "mem.csv")}`);

  const manifest = await walk(gameDir);
  const totalMB = (
    (await Promise.all(manifest.map((p) => stat(join(gameDir, p))))).reduce((a, s) => a + s.size, 0) / 1e6
  ).toFixed(0);
  console.log(`[game] ${manifest.length} files, ${totalMB} MB in ${gameDir}`);

  const server = createServer(async (req, res) => {
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
      const file = url.startsWith("/game/")
        ? join(gameDir, url.slice("/game/".length))
        : join(webDir, url === "/" ? "/game.html" : url);
      const body = await readFile(file);
      res.writeHead(200, { "content-type": MIME[extname(file)] || "application/octet-stream", ...coi });
      res.end(body);
    } catch {
      res.writeHead(404).end("not found");
    }
  });
  // A FIXED port, because OPFS is keyed by ORIGIN.
  //
  // This used to be `listen(0)` - an ephemeral port, so every run was a different origin
  // with its own empty OPFS, and every run re-imported the whole title. The persistent
  // profile did nothing, the "already imported" path was never once taken, and the profile
  // grew by 1719 MB per run: 29 copies and 49 GB before anyone looked. A stable port is
  // what makes "import once, play many times" true here as well as in the product.
  //
  // A port already in use is an ERROR naming the cause, not a silent fallback to a random
  // one - a fallback would restore exactly the behaviour this exists to prevent.
  const port = Number(process.env.PORT || 8787);
  await new Promise((resolve, reject) => {
    server.once("error", (e) =>
      reject(
        e.code === "EADDRINUSE"
          ? new Error(
              `port ${port} is in use - another run is probably still going. OPFS is keyed ` +
                `by origin, so this port must be stable or the title is re-imported ` +
                `(1.7 GB) every run. Stop the other run, or set PORT= and TITLE_ID=.`
            )
          : e
      )
    );
    server.listen(port, "127.0.0.1", resolve);
  });
  const url = `http://127.0.0.1:${port}/`;
  console.log(`[game] serving at ${url}`);

  // WORKER=1 runs the emulator in a Web Worker (the production home). A worker allows
  // synchronous instantiation of the title's large transpiled module at any size, so it
  // needs NO WebAssemblyUnlimitedSyncCompilation flag - we deliberately omit it in worker
  // mode to prove that. On the main thread that flag is still required (large sync
  // instantiate mid-run is disallowed there).
  const useWorker = !!process.env.WORKER;
  const features = useWorker ? "Vulkan" : "Vulkan,WebAssemblyUnlimitedSyncCompilation";
  // HEADLESS=1 runs without a window, which is the polite thing to do on a machine
  // someone is using - it never takes the foreground. Whether it still gets the GPU is
  // NOT assumed either way: Chrome's current headless mode can use one with
  // `--enable-gpu`, older builds could not, and the difference is a 30x speed change and
  // a meaningless frame rate. So it is not asserted here at all - the page reports the
  // adapter it actually got and refuses to render on a software one, which turns this
  // from a thing to believe into a thing the run proves.
  const headless = !!process.env.HEADLESS;
  const allowSoftware = !!process.env.ALLOW_SOFTWARE;
  // A PERSISTENT profile, so OPFS survives between runs.
  //
  // The title is imported into OPFS once and mounted from there every run after. With
  // Playwright's default throwaway profile that import would be paid on every single run
  // - over a gigabyte written per invocation - which would make the harness unusable and,
  // worse, would mean the harness never exercises the path the product actually takes
  // (mounting an already-installed title).
  //
  // PROFILE_DIR is REQUIRED, and there is deliberately no default.
  //
  // The only two candidates are both wrong. The repo is forbidden scratch. The OS temp
  // directory is where a 1.7 GB profile gets swept away between runs, and its whole value is
  // surviving them - a profile that silently vanished would re-import the title and read as a
  // slow run rather than a missing one. So the caller names it, the same way serve.mjs makes
  // the caller name --games: a host path belongs in the invocation, never in a tracked file.
  const profileDir = process.env.PROFILE_DIR;
  if (!profileDir) {
    throw new Error(
      "PROFILE_DIR is required: the persistent Chrome profile holding the OPFS copy of the " +
        "title. Pass an absolute path outside the repo, and pass the SAME one every run or " +
        "the title is re-imported (over a gigabyte) each time."
    );
  }
  // A persistent profile is single-use: a second run while the first still holds it dies
  // with a bare "exitCode=21" and a wall of Chrome flags, which says nothing about the
  // actual cause. Name it.
  const launchPersistent = async (opts) => {
    try {
      return await chromium.launchPersistentContext(profileDir, opts);
    } catch (e) {
      if (/exitCode=21|ProcessSingleton|profile/i.test(String(e.message))) {
        throw new Error(
          `Chrome could not lock the profile at ${profileDir} - another run is almost ` +
            `certainly still using it. Stop it, or pass PROFILE_DIR=<other dir> to run ` +
            `two at once (each gets its own OPFS copy of the title).`
        );
      }
      throw e;
    }
  };
  const context = await launchPersistent({
    channel: process.env.PWCHANNEL || "chrome",
    headless,
    viewport: { width: 1100, height: 800 },
    deviceScaleFactor: 1,
    args: [
      "--enable-unsafe-webgpu",
      `--enable-features=${features}`,
      // A window Chrome believes is hidden gets its timers throttled and its rAF stopped,
      // which reads from outside as "the page died while burning CPU". A long emulator run
      // in a background window hits all three of these.
      "--disable-background-timer-throttling",
      "--disable-backgrounding-occluded-windows",
      "--disable-renderer-backgrounding",
      "--use-angle=default",
      // Audio. A harness has no user gestures, so without this the AudioContext stays
      // suspended and the run is silent no matter how much PCM the guest produces - and
      // the two are indistinguishable from the outside. Headless Chrome also has no
      // output device, so the worklet needs a fake one to be pulled at all; without it
      // `process()` is never called and the ring simply fills up and reports overruns.
      "--autoplay-policy=no-user-gesture-required",
      "--use-fake-device-for-media-stream",
      "--alsa-output-device=default",
      // Ask for the GPU explicitly. Headless Chrome disables it by default; with this it
      // can use a real one, and the page's adapter check says whether it did.
      "--enable-gpu",
      // The software rasteriser is OPT-IN. Always-on, it turns "there is no GPU here" into
      // a silent 30x-slower run that still produces a plausible picture and a meaningless
      // frame rate - the largest unreported fallback in the system.
      ...(allowSoftware ? ["--enable-unsafe-swiftshader"] : []),
      // Chrome's OWN log, on disk. A process that Chrome kills - for memory, for a fatal
      // V8 error, for an unresponsive renderer - says so here and NOWHERE the page can
      // reach: `page.on("crash")`, worker `error` and `unhandledrejection` all report
      // failures the renderer survives long enough to describe. This is the record of the
      // ones it does not. Default verbosity, so it costs a few kB unless something is
      // actually wrong; CHROME_VERBOSE=1 raises it when a quiet log is itself the puzzle.
      "--enable-logging",
      `--log-file=${join(shotDir, "chrome.log")}`,
      ...(process.env.CHROME_VERBOSE ? ["--v=1"] : []),
      // Extra V8/Chrome flags for an experiment, e.g.
      // JS_FLAGS="--no-wasm-tier-up" to hold the guest module in Liftoff code only.
      ...(process.env.JS_FLAGS ? [`--js-flags=${process.env.JS_FLAGS}`] : []),
      ...(process.env.CHROME_FLAGS ? process.env.CHROME_FLAGS.split(/\s+/) : []),
    ],
  });
  // 1:1 device pixels (set on the context above), so an element screenshot of the
  // 960x544 canvas is exactly 960x544 and comparable to the desktop oracle without a
  // resample. A persistent context opens with a page already; reuse it rather than
  // opening a second, so the run has exactly one renderer to account for.
  const page = context.pages()[0] || (await context.newPage());
  // The live loop's own heartbeat, PUSHED over the console rather than polled off the
  // page. Reading `#status` needs the renderer's main thread to answer, and during a long
  // fast-forward it does not answer reliably - which reported a perfectly healthy run as
  // "could not read #status" for its entire duration. A console line arrives whatever the
  // main thread is doing.
  let liveFrame = -1;
  let liveStatus = "";
  let liveAt = 0;
  let ended = false;
  // Every heartbeat's frame number against the wall clock, so the per-process memory
  // samples can be joined to FRAMES. "Did memory climb between frame 580 and 585" is not
  // answerable from a memory series alone, and a run that dies at a fixed frame is
  // describing guest progress, not elapsed time - the join is what makes the two series
  // one measurement. Set VITASLOP_BROWSER_HEARTBEAT_MS=0 for one row per frame.
  const frameCsv = createWriteStream(join(shotDir, "frames.csv"), { flags: "w" });
  frameCsv.write("t_ms,frame\n");
  // The whole run's console, verbatim, on disk. A death nine frames after an event is only
  // attributable if the log around it survived, and a terminal scrollback is not a record.
  const consoleLog = createWriteStream(join(shotDir, "console.log"), { flags: "w" });
  page.on("console", (m) => {
    const text = m.text();
    consoleLog.write(`${Date.now()} ${m.type()} ${text}\n`);
    if (text.startsWith("[live] ")) {
      liveStatus = text.slice(7);
      const f = Number((liveStatus.match(/frame (\d+)/) || [])[1] ?? liveFrame);
      if (f !== liveFrame) frameCsv.write(`${Date.now()},${f}\n`);
      liveFrame = f;
      liveAt = Date.now();
    }
    if (text.startsWith("live run ended at frame")) {
      ended = true;
      liveFrame = Number((text.match(/frame (\d+)/) || [])[1] ?? liveFrame);
      liveStatus = text;
    }
    console.log(`[page:${m.type()}] ${text}`);
  });
  page.on("pageerror", (e) => console.log(`[pageerror] ${e.message}`));
  // A Chrome process VANISHING is the single most diagnostic event available here, and it
  // is reported the moment it happens together with the frame the run had reached. It is
  // what separates "the renderer was killed" (an out-of-memory kill; the lever is
  // footprint) from "the renderer is alive and only its Web Worker stopped" (not memory at
  // all). Those two need opposite fixes and every other symptom of them is identical.
  if (procmon) {
    procmon.onProcessGone(({ pid, kind, lastWs }) => {
      console.log(
        `[game] CHROME PROCESS GONE: pid ${pid} (${kind}), last working set ` +
          `${(lastWs / 1e9).toFixed(2)} GB, at frame ${liveFrame}`
      );
    });
  }
  // A renderer that DIES and a renderer that is merely busy both present as
  // `locator.textContent: Timeout` with `page.isClosed() === false`. Only this event
  // tells them apart, and the difference is the whole diagnosis: one is a performance
  // problem and the other is an out-of-memory kill. Without it a whole session can be
  // spent optimising a run that was never slow.
  let crashed = false;
  page.on("crash", () => {
    crashed = true;
    console.log("[game] RENDERER CRASHED (Chrome killed the tab - almost always memory)");
  });
  // The emulator lives in a Web Worker in WORKER mode, so a worker that dies takes the
  // run with it while the page itself stays perfectly alive and answers nothing useful.
  // A worker closing is only news BEFORE the run is done - it also fires on a normal
  // shutdown, and reporting that as "the emulator's thread is gone" made every clean run
  // look like a failure.
  let runFinished = false;
  page.on("worker", (w) => {
    console.log(`[game] worker started: ${w.url().split("/").pop()}`);
    w.on("close", () => {
      if (!runFinished) console.log("[game] WORKER CLOSED EARLY (the emulator's thread is gone)");
    });
  });

  let ok = false;
  try {
    await page.goto(url + (useWorker ? "game-worker.html" : ""), { waitUntil: "load", timeout: 30000 });
    console.log(`[game] mode: ${useWorker ? "web worker (no sync-compile flag)" : "main thread"}`);
    console.log(`[game] recipe: ${recipe ? `${recipe.split("\n").length} lines` : "(none - live input)"}`);
    // __boot resolves after the one-time decrypt/link/transpile + WebGPU setup, then
    // the live loop runs on the event loop.
    // KNOBS='{"VITASLOP_FRAME_TOPUP":"0"}' - the browser has no environment, so a knob a
    // title needs has to be handed to the page explicitly.
    const knobs = JSON.parse(process.env.KNOBS || "{}");
    if (Object.keys(knobs).length) console.log("[game] knobs:", knobs);
    // The OPFS key for this title. Derived from the game directory so two different
    // titles never share stored files, and stable across runs so the second run of the
    // same title mounts what the first imported instead of re-importing it.
    const titleId = (process.env.TITLE_ID || gameDir).replace(/[^A-Za-z0-9._-]+/g, "_").slice(-64);
    console.log(`[game] OPFS title id: ${titleId} (profile ${profileDir})`);
    const msg = await page.evaluate(
      ([recipe, f, r, k, id]) => window.__boot(recipe, f, r, k, id),
      [recipe, maxFrames, maxRounds, knobs, titleId]
    );
    console.log("[game] SETUP:", msg);
    // The adapter the page ACTUALLY got. The wasm refuses to render on a software one
    // unless VITASLOP_ALLOW_SOFTWARE_GPU is set, so reaching this line already proves a
    // GPU - but print it anyway, because a rate published without the backend that
    // produced it is not a measurement, and this is the line that pairs them.
    // Wait for it rather than sampling: in worker mode the adapter reaches the page as a
    // postMessage after `__boot` has already returned, so an immediate read races it and
    // prints the placeholder - which reads as "unreported" when it is merely early.
    await page
      .waitForFunction(
        () => /vendor=|SOFTWARE|GPU/.test(document.getElementById("adapter")?.textContent || ""),
        undefined,
        { timeout: 60000 }
      )
      .catch(() => {});
    const adapter = (
      await page.locator("#adapter").textContent().catch(() => "adapter: unreported")
    ).trim();
    console.log(`[game] ${adapter}`);
    // Report the live loop's own frame counter while we wait. Without this the run emits
    // NOTHING between setup and the timeout, so "rendering slowly" and "the page died" look
    // identical from outside and the only way to tell them apart is to watch Chrome's CPU
    // time from another process. A retail title's front end is tens of thousands of flips
    // in, so even on a GPU and even fast-forwarding the wait is minutes - which is exactly
    // why the run has to say where it has got to as it goes.
    const t0 = Date.now();
    let lastFrame = -1;
    const ticker = setInterval(async () => {
      // Read the PUSHED heartbeat, never the page. Nothing here can time out, so a
      // progress line always describes the run rather than the harness's ability to
      // interrogate it. Silence now means the run has genuinely stopped emitting, which
      // is a real signal instead of an artefact.
      const s = liveStatus || "(no heartbeat yet)";
      const n = liveFrame;
      const quiet = liveAt ? ((Date.now() - liveAt) / 1000).toFixed(0) : "?";
      const secs = ((Date.now() - t0) / 1000).toFixed(0);
      const rate = n > 0 ? ` (${(n / ((Date.now() - t0) / 1000)).toFixed(1)} frames/s)` : "";
      // Chrome's own accounting for this run's processes. A wasm32 heap tops out at 4 GB
      // and a retail container is over a gigabyte on its own, so "how much is resident"
      // is not a curiosity here - it is the difference between a run that will finish and
      // one that will be killed, and it has to be visible BEFORE the kill.
      console.log(
        `[game] progress: ${s.trim()} at ${secs}s${rate}${n === lastFrame ? ` QUIET ${quiet}s` : ""}` +
          `${procmon ? procmon.summary() : ""}`
      );
      lastFrame = n;
    }, 15000);
    // Wait for the run to pass the target frame, to END, or to go silent for long enough
    // that it is not coming back. All three are decided from the PUSHED heartbeat, so the
    // wait never depends on the page answering a query.
    //
    // The silence bound matters: a worker that dies takes the heartbeat with it and would
    // otherwise leave the harness sitting out the entire WAIT_MS - three hours, for a
    // long run - with nothing to show for it.
    // Periodic screenshots of the live canvas, so ONE run produces a contact sheet
    // instead of a single end-of-run picture.
    //
    // This is what makes a browser run comparable to the desktop oracle, which already
    // writes a shot every N frames. A single final screenshot can say THAT the two
    // engines ended up somewhere different; only a series can say WHERE they parted, and
    // re-running to a different fast-forward target to bisect costs minutes per guess.
    // Each shot is filed under the FRAME the run had reached, taken from the pushed
    // heartbeat, because a browser frame and a wall second are not the same axis.
    const shotEveryMs = Number(process.env.SHOT_EVERY_MS || 0);
    let shotTimer = null;
    if (shotEveryMs > 0) {
      let busy = false;
      shotTimer = setInterval(async () => {
        // Never overlap: an element screenshot on a busy page can take longer than the
        // interval, and a queue of them would stall the harness rather than sample it.
        if (busy || liveFrame < 0) return;
        busy = true;
        try {
          await page.locator("#screen").screenshot({
            path: join(shotDir, `f${String(liveFrame).padStart(6, "0")}.png`),
            timeout: 20000,
          });
        } catch (e) {
          console.log(`[game] periodic shot at frame ${liveFrame} failed: ${e.message}`);
        } finally {
          busy = false;
        }
      }, shotEveryMs);
    }
    // `HIDE_SHOW_MS=N` (optionally `HIDE_FOR_MS=M`, `HIDE_FROM_FRAME=F`): every N ms, put
    // the page in the BACKGROUND for M ms and bring it back.
    //
    // # Why the harness needs this at all
    // A hidden tab is a different execution environment, not a paused one: Chrome stops
    // `requestAnimationFrame`, throttles timers, and may drop the GPU work behind the
    // canvas. The user's own report of the white-out is stated in exactly these terms -
    // "every time I leave the browser and go back, it shows properly before the white light
    // consumes it" - and the device capture records the tab being backgrounded five times in
    // the run that went white. Every automated run before this one held the tab in the
    // foreground for its whole life, so the harness could not produce the one state the
    // defect was described in.
    //
    // Backgrounding is done by focusing a SECOND tab rather than by faking an event: only
    // the real thing gets the real rAF and throttling behaviour, and a dispatched
    // `visibilitychange` with `document.hidden` still false would test nothing.
    const hideShowMs = Number(process.env.HIDE_SHOW_MS || 0);
    const hideForMs = Number(process.env.HIDE_FOR_MS || 3000);
    const hideFromFrame = Number(process.env.HIDE_FROM_FRAME || 0);
    let hideTimer = null;
    let hideCycles = 0;
    let hideVerified = false;
    if (hideShowMs > 0) {
      const other = await page.context().newPage();
      await other.goto("about:blank");
      await page.bringToFront();
      // Freezing through CDP as well as focusing another tab. Focusing another tab is the
      // faithful thing and is what a user does, but it was MEASURED not to background
      // anything under headless Chrome - the emulator ran straight through every hidden
      // window at full rate - so on its own it is an experiment that silently tests nothing.
      const cdp = await page.context().newCDPSession(page).catch(() => null);
      let busy = false;
      hideTimer = setInterval(async () => {
        if (busy || liveFrame < hideFromFrame) return;
        busy = true;
        try {
          const at = liveFrame;
          await other.bringToFront();
          if (cdp) await cdp.send("Page.setWebLifecycleState", { state: "frozen" }).catch(() => {});
          await new Promise((r) => setTimeout(r, hideForMs));
          if (cdp) await cdp.send("Page.setWebLifecycleState", { state: "active" }).catch(() => {});
          await page.bringToFront();
          // >>> THE CYCLE HAS TO PROVE IT HAPPENED.
          // A page that kept running through the hidden window was never backgrounded, and a
          // run full of cycles that did nothing reads exactly like a run that reproduced
          // nothing. Compare the frame counter across the window: a genuinely hidden tab gets
          // no `requestAnimationFrame`, so it advances by almost nothing.
          const advanced = liveFrame - at;
          const expected = (hideForMs / 1000) * 20; // ~20 fps is a conservative floor here
          if (advanced < expected / 4) hideVerified = true;
          hideCycles++;
          console.log(
            `[game] hide/show cycle ${hideCycles} around frame ${at} (hidden ${hideForMs} ms, ` +
              `frames advanced ${advanced} while hidden${advanced < expected / 4 ? "" : " - NOT ACTUALLY BACKGROUNDED"})`
          );
          // A shot right after the restore, on its own schedule: the user's account is that
          // the frame immediately after coming back is CORRECT and it degrades from there,
          // so the periodic sampler is the wrong instrument for the interesting moment.
          await page.locator("#screen").screenshot({
            path: join(shotDir, `f${String(liveFrame).padStart(6, "0")}-shown.png`),
            timeout: 20000,
          }).catch(() => {});
        } catch (e) {
          console.log(`[game] hide/show cycle failed: ${e.message}`);
        } finally {
          busy = false;
        }
      }, hideShowMs);
    }
    const waitMs = Number(process.env.WAIT_MS || 180000);
    const quietLimitMs = Number(process.env.QUIET_MS || 120000);
    const deadline = Date.now() + waitMs;
    while (liveFrame < targetFrame && !ended) {
      if (Date.now() > deadline) throw new Error(`timed out after ${waitMs} ms at frame ${liveFrame}`);
      if (liveAt && Date.now() - liveAt > quietLimitMs) {
        throw new Error(
          `no heartbeat for ${((Date.now() - liveAt) / 1000) | 0}s at frame ${liveFrame} ` +
            `(crashed=${crashed}, pageClosed=${page.isClosed()}) - the run stopped emitting`
        );
      }
      await new Promise((r) => setTimeout(r, 500));
    }
    clearInterval(ticker);
    if (shotTimer) clearInterval(shotTimer);
    if (hideTimer) {
      clearInterval(hideTimer);
      console.log(`[game] ${hideCycles} hide/show cycle(s) during the run`);
      // Said at the END, unconditionally, because this decides whether the run is evidence
      // about backgrounding at all. Without it a null result reads as a negative one.
      if (!hideVerified) {
        console.log(
          "[game] WARNING: NO cycle actually backgrounded the page - the emulator kept running " +
            "through every hidden window. This run says NOTHING about hide/show behaviour."
        );
      }
    }
    // Let a few more live frames present so the FPS meter settles on a real cadence.
    await page.waitForFunction(() => /fps:\s*\d/.test(document.getElementById("fps")?.textContent || ""), { timeout: 15000 }).catch(() => {});
    await page.waitForTimeout(1500);
    const fps = await page.locator("#fps").textContent().catch(() => "?");
    const perf = await page.locator("#perf").textContent().catch(() => "?");
    // The heartbeat, not the DOM: it is the value this run was actually judged on, and it
    // is readable even when the page is not.
    const status = liveStatus || "?";
    // Audio, as COUNTERS rather than as a claim. There is nothing to listen to in a
    // headless run, so the only honest verification is the ring's own bookkeeping:
    // `written` says the guest produced PCM and the sink delivered it, `read` says the
    // device's audio thread actually pulled it, and `underrun`/`overrun` say which side
    // is the slow one. A run with written=0 is a guest that emitted nothing, which is a
    // completely different problem from a ring nobody drains.
    const audio = await page.evaluate(() => (window.__audioStats ? window.__audioStats() : null)).catch(() => null);
    if (audio) {
      const secs = (n) => (n / (audio.sampleRate || 48000)).toFixed(1);
      console.log(
        `[audio] context=${audio.state} written=${audio.written} (${secs(audio.written)}s) ` +
          `read=${audio.read} (${secs(audio.read)}s) underrun=${audio.underrun} overrun=${audio.overrun}`
      );
      // >>> COUNTERS CANNOT TELL MUSIC FROM SILENCE. A frame of zeroes is written and
      // read exactly like a frame of music, and an engine that produced nothing but
      // perfectly paced digital silence reported healthy counters for a long time. The
      // peak below is the cheap half of the answer; AUDIO_DUMP=<file> writes the ring's
      // last ~0.5s as raw s16le so a real signal check can run on it offline.
      // The whole-run high-water mark. This is the load-bearing check: the ring probe
      // below only sees the last half second, and a title with sparse audio (one
      // measured front end is silent 95% of the time) shows an empty ring at the end of
      // a run that was full of sound.
      console.log(
        audio.peak > 0
          ? `[audio] RUN PEAK ${audio.peak.toFixed(4)} (${(20 * Math.log10(audio.peak)).toFixed(1)} dBFS) - the run PRODUCED SOUND`
          : `[audio] RUN PEAK 0 - NOTHING was audible at any point, despite the counters above`
      );
      const probe = await page
        .evaluate(() => {
          if (!window.__audioSamples) return null;
          const s = window.__audioSamples();
          let peak = 0;
          for (let i = 0; i < s.length; i++) peak = Math.max(peak, Math.abs(s[i]));
          return { peak, count: s.length, pcm: Array.from(s) };
        })
        .catch(() => null);
      if (!probe) {
        console.log("[audio] this page exposes no sample probe - cannot tell silence from sound");
      } else if (probe.peak === 0) {
        console.log(
          `[audio] the ring's last 0.5s is silent (${probe.count} samples, every one zero)` +
            (audio.peak > 0
              ? " - which is only a quiet MOMENT, since the run peak above is non-zero"
              : ", and so was the whole run")
        );
      } else {
        console.log(
          `[audio] ring carries SOUND: peak ${probe.peak.toFixed(4)} ` +
            `(${(20 * Math.log10(probe.peak)).toFixed(1)} dBFS) over ${probe.count} samples`
        );
      }
      if (probe && process.env.AUDIO_DUMP) {
        const fsp = await import("node:fs/promises");
        const buf = Buffer.alloc(probe.pcm.length * 2);
        for (let i = 0; i < probe.pcm.length; i++) {
          buf.writeInt16LE(Math.max(-32768, Math.min(32767, Math.round(probe.pcm[i] * 32768))), i * 2);
        }
        await fsp.writeFile(process.env.AUDIO_DUMP, buf);
        console.log(`[audio] wrote ${buf.length} bytes of ring PCM to ${process.env.AUDIO_DUMP}`);
      }
    } else {
      console.log("[audio] no ring on this page - the run was SILENT");
    }
    const fs = await import("node:fs/promises");
    const shot = join(shotDir, "game.png");
    await page.locator("#screen").screenshot({ path: shot });
    // The picture and the numbers that produced it, in one machine-readable place.
    // A screenshot on its own cannot say which adapter drew it or how fast; a shot
    // filed without that pairing is what let a whole session's SwiftShader runs pass
    // for browser results.
    await fs.writeFile(
      join(shotDir, "browser-run.json"),
      JSON.stringify(
        { adapter, fps, perf, status, headless, allowSoftware, targetFrame, maxFrames, recipe: recipePath, knobs, screenshot: shot },
        null,
        2
      ) + "\n"
    );
    console.log(`[game] live render ${fps} | ${perf} | ${status} -> screenshot ${shot}`);
    ok = liveFrame >= targetFrame;
  } catch (e) {
    console.error("[game] error:", e.message);
    console.error("[game] last status:", await page.locator("#status").textContent().catch(() => "?"));
  } finally {
    runFinished = true;
    // The telemetry is flushed BEFORE the browser is torn down, so closing Chrome cannot
    // be mistaken in the CSV for the failure being investigated.
    if (procmon) {
      const deaths = procmon.deaths();
      if (deaths.length) {
        console.log(`[game] ${deaths.length} Chrome process death(s) during the run:`);
        for (const d of deaths) {
          console.log(`[game]   pid ${d.pid} (${d.kind}) last ws ${(d.lastWs / 1e9).toFixed(2)} GB`);
        }
      } else {
        console.log("[game] no Chrome process died during the run (any death is a THREAD, not a process)");
      }
      procmon.stop();
    }
    frameCsv.end();
    consoleLog.end();
    console.log(`[game] telemetry: ${join(shotDir, "mem.csv")}, ${join(shotDir, "frames.csv")}`);
    await context.close();
    server.close();
  }
  console.log(ok ? "[game] PASS" : "[game] FAIL");
  process.exit(ok ? 0 : 1);
}

main();
