// cpu-workers - what N concurrently busy browser workers each get from this device's CPU, and
// what a cross-worker wake costs.
//
// The question (2026-10-02, Marvel fight on the phone): the render thread's own guest work cost
// 2.5 ms a frame with ONE SMP worker and 6-10 ms with three - the same code, slower per thread
// once other workers are busy. Big/little cores, a governor or memory contention would all look
// like that; this job measures the device without the emulator in the way.
//
// compute: for each N in `counts`, N workers start together at a barrier and run a fixed integer
//   loop for `ms`; reported per worker as iterations per microsecond (higher = faster core).
// pingpong: one worker wakes another through Atomics.notify and times the reply, with the
//   answering worker asleep in Atomics.wait (the SMP bell) or polling, with and without other
//   workers burning CPU.
//
// params: counts ([1,2,3,4,1]), ms (1500), rounds (400), gapMs (2).

// The worker body is fetched once and started from a Blob URL as a CLASSIC worker: a nested
// MODULE worker from inside the job worker failed on the phone (job 364, a bare "worker
// error"), and the body needs no imports.
let bodyUrl = null;
async function loadBody() {
  if (bodyUrl) return;
  const r = await fetch(new URL("./cpu-workers-w.js", import.meta.url));
  if (!r.ok) throw new Error(`cpu-workers-w.js: HTTP ${r.status}`);
  bodyUrl = URL.createObjectURL(new Blob([await r.text()], { type: "text/javascript" }));
}

function spawn() {
  return new Worker(bodyUrl);
}

function once(w) {
  return new Promise((res, rej) => {
    w.onmessage = (e) => res(e.data);
    w.onerror = (e) => rej(new Error(`worker error: ${e.message || "?"} ${e.filename || ""}:${e.lineno || ""}`));
  });
}

const pct = (xs, p) => {
  const s = [...xs].sort((a, b) => a - b);
  return s[Math.min(s.length - 1, Math.floor(p * s.length))];
};

async function compute(n, ms) {
  const sab = new SharedArrayBuffer(16);
  const ws = Array.from({ length: n }, spawn);
  const outs = ws.map((w, idx) => {
    const p = once(w);
    w.postMessage({ op: "compute", sab, idx, n, ms });
    return p;
  });
  const r = await Promise.all(outs);
  ws.forEach((w) => w.terminate());
  return r.map((x) => +(x.perMs / 1000).toFixed(2));
}

async function pingpong({ spin, burners, rounds, gapMs }) {
  const sab = new SharedArrayBuffer(16);
  const ctl = new Int32Array(sab);
  const burnSab = new SharedArrayBuffer(16);
  const bs = Array.from({ length: burners }, spawn);
  const bdone = bs.map((w) => {
    const p = once(w);
    w.postMessage({ op: "burn", sab: burnSab });
    return p;
  });
  const pong = spawn();
  const pdone = once(pong);
  pong.postMessage({ op: "pong", sab, spin });
  const ping = spawn();
  const pp = once(ping);
  // Let the workers start before the first round.
  await new Promise((r) => setTimeout(r, 200));
  ping.postMessage({ op: "ping", sab, rounds, gapMs });
  const { rt } = await pp;
  Atomics.store(ctl, 0, -1);
  Atomics.add(ctl, 1, 1);
  Atomics.notify(ctl, 1);
  Atomics.store(new Int32Array(burnSab), 0, -1);
  await Promise.all([pdone, ...bdone]);
  [pong, ping, ...bs].forEach((w) => w.terminate());
  const f = (v) => +v.toFixed(3);
  return {
    spin, burners,
    p50: f(pct(rt, 0.5)), p90: f(pct(rt, 0.9)), p99: f(pct(rt, 0.99)), max: f(Math.max(...rt)),
    mean: f(rt.reduce((a, b) => a + b, 0) / rt.length),
  };
}

export async function run(params, { progress }) {
  if (typeof SharedArrayBuffer === "undefined" || !self.crossOriginIsolated) {
    throw new Error("no SharedArrayBuffer here (not cross-origin isolated)");
  }
  await loadBody();
  const counts = params.counts || [1, 2, 3, 4, 1];
  const ms = params.ms || 1500;
  const rounds = params.rounds || 400;
  const gapMs = params.gapMs ?? 2;
  const out = { hw: navigator.hardwareConcurrency, compute: [], pingpong: [] };
  for (const n of counts) {
    progress(`compute x${n}`);
    out.compute.push({ n, perUs: await compute(n, ms) });
  }
  for (const cfg of [
    { spin: false, burners: 0 },
    { spin: true, burners: 0 },
    { spin: false, burners: 2 },
    { spin: true, burners: 2 },
  ]) {
    progress(`pingpong spin=${cfg.spin} burners=${cfg.burners}`);
    out.pingpong.push(await pingpong({ ...cfg, rounds, gapMs }));
  }
  out.summary =
    "compute it/us per worker: " + out.compute.map((c) => `x${c.n} [${c.perUs.join(" ")}]`).join(" ") +
    " | round trip ms p50/p90/max: " +
    out.pingpong.map((p) => `${p.spin ? "spin" : "sleep"}+${p.burners}burn ${p.p50}/${p.p90}/${p.max}`).join(" ");
  return out;
}
