// live.js - the runner's LIVE job page script; see live.html for the why.
import { effective, runKnobs } from "../store.js";
import { isComplete } from "../opfs.js";
import { startAudio } from "../audio.js";
import { bundleModule } from "../bundle.js";

const statusEl = document.getElementById("status");
const canvas = document.getElementById("screen");
const up = (m) => parent.postMessage(m, location.origin);
const progress = (text) => {
  statusEl.textContent = text;
  up({ type: "progress", text });
};

async function asset(path) {
  const r = await fetch(`/runner-assets/${path}`);
  if (!r.ok) throw new Error(`asset ${path}: HTTP ${r.status}`);
  return r.text();
}

async function run(p) {
  const titleId = p.titleId;
  if (!titleId) throw new Error("params.titleId is required");
  if (!(await isComplete(titleId))) throw new Error(`${titleId} is not (completely) imported on this device`);
  // The product's own knobs for this title, then the job's over them - the same merge
  // player.js does with a link's `?knobs=`.
  const knobs = { ...(await runKnobs(await effective(titleId))), ...(p.knobs || {}) };
  // `params.shots` are taken by the RENDERER (VITASLOP_SHOT_FRAMES: a full-size copy of the
  // surface before it is presented, posted as a `presentshot` note). The canvas route
  // (`convertToBlob` after the present) failed on the phone for 15 of 15 shots.
  if (p.shots && p.shots.length && !knobs.VITASLOP_SHOT_FRAMES) knobs.VITASLOP_SHOT_FRAMES = p.shots.join(",");
  const recipe = p.recipe ? await asset(p.recipe) : p.recipeText || "";
  const stopFrame = Number(p.stopFrame || 0);
  const t0 = performance.now();
  const reports = {};
  const notes = [];
  let traceDone = !p.waitTrace;
  let lastFrame = 0;
  let error = null;

  const bundleQ = String(knobs.VITASLOP_SMP ?? "") === "1" ? "?smp=1" : "";
  // Compiled once here and handed to both workers, as the player does - see bundle.js.
  progress("loading the emulator...");
  const module = await bundleModule(bundleQ !== "");
  const tBundle = performance.now();
  const worker = new Worker("../worker.js" + bundleQ, { type: "module" });
  progress("reserving...");
  let reserveSplit = `bundle compiled by the page ${Math.round(tBundle - t0)} ms; `;
  const hostOff = await new Promise((resolve, reject) => {
    worker.onmessage = (e) => {
      const d = e.data;
      if (d.type === "reserved") {
        reserveSplit += d.split || "";
        resolve(d.hostOff);
      } else if (d.type === "error" || d.type === "panic") reject(new Error(d.message));
    };
    worker.onerror = (e) => reject(new Error(e.message || "run worker failed to start"));
    worker.postMessage({ type: "reserve", knobs, module });
  });
  const tReserved = performance.now();
  progress("transpiling...");
  const prebuilt = await new Promise((resolve, reject) => {
    const tw = new Worker("../transpile-worker.js" + bundleQ, { type: "module" });
    tw.onmessage = (e) => {
      if (e.data.type === "panic") return reject(new Error("PANIC WHILE PREPARING\n" + e.data.message));
      tw.terminate();
      e.data.type === "built" ? resolve(e.data.built) : reject(new Error(e.data.message));
    };
    tw.onerror = (e) => {
      tw.terminate();
      reject(new Error(e.message || "transpile worker failed to start"));
    };
    tw.postMessage({ titleId, knobs, hostOff, bundleModule: module });
  });
  const tReady = performance.now();
  progress(`transpiled in ${((tReady - t0) / 1000).toFixed(1)} s (${prebuilt && prebuilt.split}); running`);

  // >>> AUDIO, MUTED (`params.audio`). The ring's worklet pulls through a zero-gain node, so
  // the underrun counter measures exactly what a player's ring does without the phone making a
  // sound. Stats are taken when the run crosses `measureFrom` and at the end, so the result
  // states the measured stretch's underrun, not boot's.
  let audio = null;
  if (p.audio) {
    try {
      audio = await startAudio(() => {});
      audio.node.disconnect();
      const mute = audio.context.createGain();
      mute.gain.value = 0;
      audio.node.connect(mute).connect(audio.context.destination);
    } catch (err) {
      notes.push(`[audio] could not start: ${err && err.message}`);
      audio = null;
    }
  }
  let audioAt = null;
  const audioSnap = () => (audio ? { t: performance.now(), frame: lastFrame, ...audio.stats() } : null);

  const meter = [];
  const finished = new Promise((resolve) => {
    const check = () => {
      if (error || (stopFrame && lastFrame >= stopFrame && traceDone)) resolve();
    };
    worker.onmessage = (e) => {
      const d = e.data;
      if (d.type === "report") {
        reports[d.id] = d.text;
        // Every half-second fps meter reading AFTER fast-forward, so the result can state the
        // whole measured stretch (mean, p10, min) instead of the one reading the run ended on.
        if (d.id === "fps") {
          const m = /fps:\s*([0-9.]+)[^(]*\(([0-9.]+)% speed/.exec(d.text);
          if (m && !/fast-forwarding/.test(d.text) && lastFrame >= (p.measureFrom || 0)) meter.push({ frame: lastFrame, fps: Number(m[1]), speed: Number(m[2]) });
        }
        if (d.id === "status") {
          const m = /frame (\d+)/.exec(d.text);
          if (m) lastFrame = Number(m[1]);
          if (audio && !audioAt && lastFrame >= (p.measureFrom || 0) && lastFrame > 0) audioAt = audioSnap();
          statusEl.textContent = `f${lastFrame} ${reports.fps || ""}`;
        }
        check();
      } else if (d.type === "note") {
        notes.push(d.text);
        if (d.text.startsWith("smptrace END")) traceDone = true;
        check();
      } else if (d.type === "error" || d.type === "panic") {
        error = `${d.type}: ${d.message}`;
        check();
      }
    };
    worker.onerror = (e) => {
      error = `worker died: ${e.message || e}`;
      check();
    };
    // Progress to the runner every few seconds, so the phone's log shows the frame.
    const beat = setInterval(() => {
      if (error || (stopFrame && lastFrame >= stopFrame && traceDone)) return clearInterval(beat);
      up({ type: "progress", text: `f${lastFrame} ${(reports.fps || "").replace(/^fps:\s*/, "")}` });
    }, 5000);
  });

  // >>> A WALL-CLOCK WATCHDOG on this page's main thread (the user's point: a hitch that hangs
  // the WHOLE browser stops every timer, the engine's included, so the engine's own numbers
  // show a long frame and nothing else). A 100 ms interval that arrives late records the gap,
  // with the frame the run had reached - big gaps are browser/OS stalls, not our frame time.
  const hangs = [];
  let lastTick = performance.now();
  const watchdog = setInterval(() => {
    const t = performance.now();
    const gap = t - lastTick;
    if (gap > 250) hangs.push({ atMs: Math.round(t - tReady), gapMs: Math.round(gap), frame: lastFrame });
    lastTick = t;
  }, 100);

  const offscreen = canvas.transferControlToOffscreen();
  worker.postMessage(
    { offscreen, titleId, recipe, maxFrames: 0xffffffff, knobs, prebuilt, audioRing: audio ? audio.ring : undefined, profile: p.profile || "runner", noPersist: p.persist !== true, forwardConsole: p.console ? Number(p.console) || 5000 : 0, shots: knobs.VITASLOP_SHOT_FRAMES ? [] : p.shots || [] },
    [offscreen]
  );
  await finished;
  clearInterval(watchdog);
  // The measured stretch's sound: seconds the ring ran dry against seconds of wall, and the
  // seconds the guest produced. Underrun over wall IS the share of time a player hears a gap.
  let audioResult = null;
  if (audio) {
    const end = audioSnap();
    const a = audioAt || end;
    const sr = end.sampleRate || 48000;
    const wallS = (end.t - a.t) / 1000;
    audioResult = {
      state: end.state,
      fromFrame: a.frame,
      toFrame: end.frame,
      wallS: Math.round(wallS * 100) / 100,
      writtenS: Math.round(((end.written - a.written) / sr) * 100) / 100,
      readS: Math.round(((end.read - a.read) / sr) * 100) / 100,
      underrunS: Math.round(((end.underrun - a.underrun) / sr) * 100) / 100,
      underrunPct: wallS > 0 ? Math.round(((end.underrun - a.underrun) / sr / wallS) * 1000) / 10 : null,
      latencySkipS: Math.round(((end.latencySkip - a.latencySkip) / sr) * 100) / 100,
      rejoins: end.rejoins - a.rejoins,
      fillFrames: end.fill,
    };
    audio.context.close().catch(() => {});
  }
  // One more report window, so the panel describes the frames just reached.
  // The measured stretch as a distribution: a single reading is whatever half-second the run
  // happened to end on. p10 is the "bad moments" figure; the mean is what a player gets.
  const stat = (key) => {
    const v = meter.map((x) => x[key]).sort((a, b) => a - b);
    if (!v.length) return null;
    const mean = v.reduce((a, b) => a + b, 0) / v.length;
    return { mean: Math.round(mean * 10) / 10, p10: v[Math.floor(v.length * 0.1)], min: v[0], max: v[v.length - 1] };
  };
  const fpsStat = stat("fps");
  const speedStat = stat("speed");
  const window = fpsStat
    ? ` | over ${meter.length} readings (f${meter[0].frame}-f${meter[meter.length - 1].frame}): fps mean ${fpsStat.mean} p10 ${fpsStat.p10} min ${fpsStat.min}; speed mean ${speedStat.mean}% p10 ${speedStat.p10}% min ${speedStat.min}%`
    : "";
  const audioSummary = audioResult ? ` | audio ${audioResult.state}: underrun ${audioResult.underrunS}s of ${audioResult.wallS}s (${audioResult.underrunPct}%), made ${audioResult.writtenS}s` : "";
  const result = {
    summary: error ? error.slice(0, 200) : `reached f${lastFrame}; ${reports.fps || ""}${window}${audioSummary}`,
    audio: audioResult,
    meter,
    error,
    knobs,
    lastFrame,
    prepareMs: Math.round(tReady - t0),
    prepareSplit: `reserve ${Math.round(tReserved - t0)} ms (${reserveSplit}); worker ready ${prebuilt && prebuilt.readyMs} ms; ${prebuilt && prebuilt.split}; handed back at ${Math.round(tReady - tReserved)} ms`,
    runMs: Math.round(performance.now() - tReady),
    reports,
    notes,
    hangs,
  };
  worker.terminate();
  return result;
}

addEventListener("message", async (e) => {
  if (e.origin !== location.origin || e.data?.type !== "job") return;
  try {
    up({ type: "result", result: await run(e.data.params || {}) });
  } catch (err) {
    up({ type: "error", error: `${err && err.message}\n${(err && err.stack) || ""}`.slice(0, 4000) });
  }
});
up({ type: "ready" });
