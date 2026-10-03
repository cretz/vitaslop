// runner.js - the device test runner's page half. See runner-server.mjs for the why.
//
// The loop: say hello (a fingerprint), ask for the next job, run it in a FRESH Worker
// (job-worker.js), post the result, terminate the worker, repeat. A fresh worker per job because
// a wasm heap only ever grows - terminating is the one reliable way to hand it back.
//
// Built for a phone that comes and goes: nothing runs while the page is hidden, the wake lock
// is re-taken every time it comes back, and an interrupted job is not resumed here - its lease
// lapses on the server and it is offered again. The page also reloads itself when the
// server's copy of the runner changes, so an update reaches a phone nobody is holding.

import { fingerprint } from "./fingerprint.js";

const $ = (id) => document.getElementById(id);
const POLL_MS = 2000;
const HELLO_EVERY_MS = 60_000;

const store = {
  get: (k, d) => {
    try {
      return localStorage.getItem(k) ?? d;
    } catch {
      return d;
    }
  },
  set: (k, v) => {
    try {
      localStorage.setItem(k, v);
    } catch {}
  },
};
let device = store.get("vitaslop.runner.device", "");
if (!/^[0-9a-f]{8}$/.test(device)) {
  device = Array.from(crypto.getRandomValues(new Uint8Array(4)), (b) => b.toString(16).padStart(2, "0")).join("");
  store.set("vitaslop.runner.device", device);
}
const nameBox = $("name");
nameBox.value = store.get("vitaslop.runner.name", "");
nameBox.addEventListener("change", () => {
  store.set("vitaslop.runner.name", nameBox.value.trim());
  lastHello = 0;
});
$("devid").textContent = `id ${device}`;

const lines = [];
const log = (text, cls = "") => {
  const t = new Date().toLocaleTimeString();
  lines.unshift(`<span class="${cls}">${t}  ${text.replace(/[<&]/g, (c) => (c === "<" ? "&lt;" : "&amp;"))}</span>`);
  lines.length = Math.min(lines.length, 60);
  $("log").innerHTML = lines.join("\n");
};
const state = (text, cls = "") => {
  $("state").textContent = text;
  $("state").className = cls;
};

// ----- wake lock: taken on Start, re-taken on every return to the foreground -----
let wakeWanted = false;
let wake = null;
async function takeWake() {
  if (!wakeWanted || document.visibilityState !== "visible") return;
  if (!("wakeLock" in navigator)) {
    $("wake").textContent = "no wake lock in this browser - set the screen timeout long";
    return;
  }
  try {
    wake = await navigator.wakeLock.request("screen");
    $("wake").textContent = "screen kept on";
    wake.addEventListener("release", () => {
      $("wake").textContent = "screen lock released";
    });
  } catch (e) {
    $("wake").textContent = `wake lock refused: ${e.message}`;
  }
}
$("go").addEventListener("click", () => {
  wakeWanted = true;
  store.set("vitaslop.runner.autostart", "1");
  takeWake();
  $("go").textContent = "Running";
  $("go").disabled = true;
});
document.addEventListener("visibilitychange", () => {
  if (document.visibilityState === "visible") {
    takeWake();
    log("back in front - resuming");
  } else {
    log("hidden - paused");
  }
});

// ----- server -----
let fp = null;
let lastHello = 0;
let version = null;
async function hello() {
  fp ??= await fingerprint();
  const r = await fetch(`/runner/hello?device=${device}`, {
    method: "POST",
    body: JSON.stringify({ ...fp, name: nameBox.value.trim() || null }),
  });
  const j = await r.json();
  version ??= j.version;
  lastHello = Date.now();
}

/// Run one job in a fresh Worker (job-worker.js); resolves with the result object (never
/// rejects). A Worker so the page's own timers keep running through a job, and so the kill is
/// exact and returns the job's heap.
function runInWorker(job) {
  return new Promise((resolve) => {
    const w = new Worker("./job-worker.js", { type: "module" });
    let settled = false;
    const finish = (result) => {
      if (settled) return;
      settled = true;
      clearTimeout(timer);
      clearInterval(beat);
      w.terminate();
      resolve(result);
    };
    w.onmessage = (e) => {
      const m = e.data || {};
      if (m.type === "ready") w.postMessage({ job });
      else if (m.type === "progress") state(`running ${job.kind}: ${m.text}`, "warn");
      else if (m.type === "result") finish({ status: "ok", ...m.result });
      else if (m.type === "error") finish({ status: "error", error: m.error });
    };
    w.onerror = (e) => finish({ status: "error", error: `worker failed: ${e.message || e}` });
    const timer = setTimeout(
      () => finish({ status: "timeout", error: `killed at ${job.timeoutMs} ms (the job contract is short runs only)` }),
      job.timeoutMs,
    );
    const beat = setInterval(() => {
      fetch(`/runner/beat?device=${device}&job=${job.id}`, { method: "POST" }).catch(() => {});
      pollSay().catch(() => {});
      if (document.visibilityState !== "visible") finish({ status: "interrupted", error: "page hidden mid-job" });
    }, 5000);
  });
}

/// Run a LIVE job (kind "live") in a full-screen iframe of `live.html`: a live run needs a
/// canvas the compositor shows and workers of its own, which a job Worker cannot give it. Same
/// contract as `runInWorker` - resolves with the result, never rejects; removing the frame
/// ends the run and hands back its memory.
function runInFrame(job) {
  return new Promise((resolve) => {
    const f = document.createElement("iframe");
    f.src = "./live.html";
    f.allow = "cross-origin-isolated; autoplay";
    // In the page, not over it: the runner's log and state stay readable under the game.
    f.style.cssText = "display:block;width:100%;aspect-ratio:960/590;border:0;background:#000;margin:8px 0";
    let settled = false;
    let stopping = false;
    // Sent with every heartbeat, so the desktop sees a stall as soon as the phone does.
    let lastProgress = "";
    const onMsg = (e) => {
      if (e.source !== f.contentWindow) return;
      const m = e.data || {};
      if (m.type === "ready") f.contentWindow.postMessage({ type: "job", params: job.params }, location.origin);
      else if (m.type === "progress") {
        lastProgress = m.text;
        state(`running live: ${m.text}`, "warn");
        log(`  ${m.text}`);
      } else if (m.type === "result") finish({ status: m.result.error ? "error" : "ok", ...m.result });
      else if (m.type === "error") finish({ status: "error", error: m.error });
    };
    const finish = (result) => {
      if (settled) return;
      settled = true;
      clearTimeout(timer);
      clearInterval(beat);
      removeEventListener("message", onMsg);
      f.remove();
      resolve(result);
    };
    addEventListener("message", onMsg);
    document.querySelector("main").insertBefore(f, $("desk"));
    const timer = setTimeout(() => finish({ status: "timeout", error: `killed at ${job.timeoutMs} ms` }), job.timeoutMs);
    const beat = setInterval(() => {
      pollSay().catch(() => {});
      fetch(`/runner/beat?device=${device}&job=${job.id}&p=${encodeURIComponent(lastProgress)}`, { method: "POST" })
        .then((r) => r.json())
        .then((r) => {
          if (!r.cancel || stopping) return;
          // Ask the game page to stop and REPORT (its result carries the reports a bare
          // cancel threw away); end it ourselves only if it cannot answer in 15 s.
          stopping = true;
          f.contentWindow?.postMessage({ type: "stop" }, location.origin);
          setTimeout(() => finish({ status: "cancelled", error: "cancelled from the desktop; the page did not report within 15 s" }), 15_000);
        })
        .catch(() => {});
      if (document.visibilityState !== "visible") finish({ status: "interrupted", error: "page hidden mid-job" });
    }, 5000);
  });
}

// Notes the desktop sends (POST /runner/say), printed in this page's log - the one screen the
// person holding the phone can see. Only notes newer than the page load, plus the last few.
// The newest one is also PINNED, three lines at most, between the game frame and the log
// (#desk): in the log alone it was buried under a running job's progress lines, and polled
// only between jobs - the user saw none. Kept small: the frame must stay on screen.
let sayFrom = -1;
const pin = (m) => {
  $("desk").innerHTML = `<span class="t">${new Date(m.at).toLocaleTimeString()}</span> ${m.text.replace(/[<&]/g, (c) => (c === "<" ? "&lt;" : "&amp;"))}`;
};
async function pollSay() {
  const r = await fetch(`/runner/say?since=${Math.max(sayFrom, 0)}`);
  const says = await r.json();
  if (sayFrom < 0) {
    sayFrom = says.length ? says.at(-1).n : 0;
    for (const m of says.slice(-3)) log(`DESKTOP: ${m.text}`, "say"), pin(m);
    return;
  }
  for (const m of says) {
    if (m.n <= sayFrom) continue;
    log(`DESKTOP: ${m.text}`, "say");
    pin(m);
    sayFrom = m.n;
  }
}

async function tick() {
  if (document.visibilityState !== "visible") return;
  await pollSay().catch(() => {});
  if (Date.now() - lastHello > HELLO_EVERY_MS) await hello();
  const r = await fetch(`/runner/next?device=${device}`);
  const v = r.headers.get("x-runner-version");
  if (version && v && v !== version) {
    log("runner updated on the server - reloading");
    location.reload();
    return;
  }
  if (r.status === 204) {
    state("idle - waiting for jobs", "ok");
    return;
  }
  const job = await r.json();
  log(`job ${job.id} (${job.kind}) ${job.note || ""}`);
  state(`running ${job.kind}...`, "warn");
  const t0 = performance.now();
  const result = await (job.kind === "live" ? runInFrame(job) : runInWorker(job));
  result.ms = Math.round(performance.now() - t0);
  if (result.status === "interrupted") {
    // Not reported: the lease lapses and the job is offered again when the page is back.
    log(`  interrupted - it will run again`, "warn");
    return;
  }
  await fetch(`/runner/result?device=${device}&job=${job.id}`, { method: "POST", body: JSON.stringify(result) });
  log(`  ${result.status} in ${result.ms} ms${result.summary ? " - " + result.summary : ""}${result.error ? " - " + result.error : ""}`, result.status === "ok" ? "ok" : "bad");
  // >>> EVERY LIVE JOB STARTS IN A FRESH TAB. Whatever one game run leaves behind (GPU
  // objects, a worker the browser has not reaped) is the next one's to inherit otherwise: the
  // phone Aw-Snapped at f0 on the seventh back-to-back Uncharted soak in one tab (30b, 059).
  if (job.kind === "live") {
    log("fresh tab for the next live job - reloading");
    location.reload();
  }
}

async function loop() {
  for (;;) {
    try {
      await tick();
    } catch (e) {
      state(`server unreachable (${e.message}) - retrying`, "bad");
    }
    await new Promise((r) => setTimeout(r, POLL_MS));
  }
}

// A reload (an update, a crash) comes back running if it was running before; the wake lock
// still wants a gesture on some browsers, which the button supplies.
if (store.get("vitaslop.runner.autostart", "") === "1") {
  wakeWanted = true;
  takeWake();
}
log(`device ${device} ready`);
loop();
