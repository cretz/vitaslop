// cpu-workers-w.js - the worker body of the `cpu-workers` job (not a job kind of its own).
//
// Messages in: {op:"compute", sab, idx, n, ms}  - wait for all `n` workers at the barrier in
//   `sab`, then run a fixed integer loop for `ms` and report iterations per ms.
// {op:"pong", sab, spin}  - answer pings on word 1 by bumping word 2, sleeping in
//   Atomics.wait between them (or polling when `spin`), until word 0 reads -1.
// {op:"ping", sab, rounds} - send `rounds` pings and time each round trip.

function work(ms) {
  let x = 0x12345678 | 0;
  let it = 0;
  const t0 = performance.now();
  let t = t0;
  while (t - t0 < ms) {
    for (let k = 0; k < 4096; k++) {
      x = Math.imul(x ^ (x >>> 15), 0x2c1b3c6d) + k | 0;
    }
    it += 4096;
    t = performance.now();
  }
  return { perMs: it / (t - t0), x };
}

self.onmessage = (e) => {
  const m = e.data;
  const a = new Int32Array(m.sab);
  if (m.op === "compute") {
    Atomics.add(a, 0, 1);
    while (Atomics.load(a, 0) < m.n) {}
    const r = work(m.ms);
    self.postMessage({ idx: m.idx, perMs: r.perMs, x: r.x });
  } else if (m.op === "pong") {
    let seen = Atomics.load(a, 1);
    for (;;) {
      if (m.spin) {
        while (Atomics.load(a, 1) === seen && Atomics.load(a, 0) !== -1) {}
      } else {
        Atomics.wait(a, 1, seen, 1000);
      }
      if (Atomics.load(a, 0) === -1) break;
      const now = Atomics.load(a, 1);
      if (now === seen) continue;
      seen = now;
      Atomics.add(a, 2, 1);
      Atomics.notify(a, 2);
    }
    self.postMessage({ done: true });
  } else if (m.op === "ping") {
    const rt = [];
    for (let i = 0; i < m.rounds; i++) {
      const before = Atomics.load(a, 2);
      const t0 = performance.now();
      Atomics.add(a, 1, 1);
      Atomics.notify(a, 1);
      while (Atomics.load(a, 2) === before) {
        if (m.sleepPing) Atomics.wait(a, 2, before, 1000);
      }
      rt.push(performance.now() - t0);
      // A gap between rounds so the answering worker really goes back to sleep.
      if (m.gapMs) {
        const g = performance.now();
        while (performance.now() - g < m.gapMs) {}
      }
    }
    self.postMessage({ rt });
  } else if (m.op === "burn") {
    // Keep a core busy until word 0 reads -1 (a background load for the ping test).
    while (Atomics.load(a, 0) !== -1) work(5);
    self.postMessage({ done: true });
  }
};
