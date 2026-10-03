// A GUEST WORKER of a parallel run (`VITASLOP_SMP=1` - see src/smp.rs). The run worker
// starts one per Vita core and posts it the emulator's own compiled module, the SHARED linear
// memory every worker runs over, and the title's transpiled guest module. This worker then
// runs guest threads on its own JSPI stacks until the run ends; the host, the guest region and
// the scheduler's state are the same objects the run worker sees, because the memory is.
// Always the THREADS bundle: it is the only one with a shared memory to run over.
import init, { smp_helper_main, smp_resolver_main, smp_sampler_main, get_knob, guest_frame } from "./pkg-threads/vitaslop_web.js";

// `VITASLOP_JS_PROFILE=<start FRAME>:<duration ms>`: a JS Self-Profiling sample of THIS guest
// worker (1 ms interval), aggregated by function - SELF and INCLUSIVE sample counts - and posted
// as one `jsprofile w<n> ...` line. Wasm frames carry the guest's function names (`f_<addr>`) and
// the emulator's, so the line says where a guest thread's time goes on the device itself.
// Needs the page served with `Document-Policy: js-profiling` (serve.mjs does).
function startProfile(worker) {
  let spec;
  try { spec = get_knob("VITASLOP_JS_PROFILE"); } catch { return; }
  if (!spec) return;
  const [start, dur] = spec.split(":").map(Number);
  const wait = setInterval(async () => {
    if (guest_frame() < start) return;
    clearInterval(wait);
    let p;
    try {
      p = new Profiler({ sampleInterval: 1, maxBufferSize: 1000000 });
    } catch (err) {
      self.postMessage({ type: "jsprofile", message: `jsprofile w${worker} UNAVAILABLE ${err}` });
      return;
    }
    setTimeout(async () => {
      const t = await p.stop();
      const self_ = new Map(), incl = new Map();
      const name = (fid) => { const f = t.frames[fid]; return f ? (f.name || `?${f.line}`) : "?"; };
      let n = 0, idle = 0;
      for (const s of t.samples) {
        if (s.stackId === undefined || s.stackId === null) { idle++; continue; }
        n++;
        let st = t.stacks[s.stackId];
        self_.set(name(st.frameId), (self_.get(name(st.frameId)) || 0) + 1);
        const seen = new Set();
        while (st) {
          const nm = name(st.frameId);
          if (!seen.has(nm)) { seen.add(nm); incl.set(nm, (incl.get(nm) || 0) + 1); }
          st = st.parentId === undefined ? null : t.stacks[st.parentId];
        }
      }
      const top = (m, k) => [...m.entries()].sort((a, b) => b[1] - a[1]).slice(0, k).map(([a, b]) => `${a}=${b}`).join(";");
      self.postMessage({ type: "jsprofile", message: `jsprofile w${worker} samples=${n} idle=${idle} SELF ${top(self_, 45)} INCL ${top(incl, 45)}` });
    }, dur);
  }, 250);
}
import { attachTitleCached } from "./opfs.js";

// >>> A GUEST THREAD'S `[trace]` LINES (`VITASLOP_TRACE_BLOCKS`) ARE PRINTED HERE, on the guest
// worker running it - whose console a phone never shows and the run worker's forwarding never
// saw. Relayed on the `jsprofile` channel, which the run worker writes raw to ITS console, where
// a device-runner job's `params.console` forwards it as a note (30c: phone job 093 traced the
// Uncharted job pool and brought back nothing).
for (const level of ["log", "info", "warn"]) {
  const orig = console[level].bind(console);
  console[level] = (...a) => {
    const text = a.map((x) => (typeof x === "string" ? x : String(x))).join(" ");
    if (text.startsWith("[trace]")) {
      try {
        self.postMessage({ type: "jsprofile", message: text });
      } catch {}
    }
    orig(...a);
  };
}

// A Rust panic here reaches the run worker (and from there the page) the same way one on the
// run worker does - see worker.js. Defined before `init` for the same reason it is there.
globalThis.__vitaslopPanic = (text) => {
  try {
    self.postMessage({ type: "panic", message: text });
  } catch {
    // A panic hook that throws replaces a diagnosable crash with an undiagnosable one.
  }
};
self.addEventListener("error", (e) =>
  self.postMessage({ type: "error", message: `guest worker error: ${e.message || e}` })
);
self.addEventListener("unhandledrejection", (e) =>
  self.postMessage({
    type: "error",
    message: `guest worker unhandled rejection: ${(e.reason && (e.reason.stack || e.reason.message)) || e.reason}`,
  })
);

self.onmessage = async (e) => {
  const d = e.data;
  try {
    // The same module over the same memory: this instance's statics, heap and the guest region
    // are the run worker's. A generous shadow stack - host calls run here, and some are deep.
    await init({ module_or_path: d.module, memory: d.memory, thread_stack_size: 8 << 20 });
    // The RESOLVER (src/smp.rs `smp_resolver_main`): no guest threads, one blocking loop that
    // reads each flip's deferred geometry. `ready` is what attaches it.
    // The guest-function SAMPLER (`VITASLOP_GUEST_PROF`, src/smp.rs `smp_sampler_main`).
    if (d.role === "sampler") {
      const report = smp_sampler_main();
      self.postMessage({ type: "note", message: report });
      return;
    }
    if (d.role === "resolver") {
      self.postMessage({ type: "ready" });
      smp_resolver_main();
      self.postMessage({ type: "note", message: "resolver finished" });
      return;
    }
    self.postMessage({ type: "ready", worker: d.worker });
    startProfile(d.worker);
    // This worker's own view of the title's storage ring, when the run worker's reader has one:
    // file reads then run here instead of being forwarded (see `opfs::install_worker_reader`).
    const storage = d.storage ? attachTitleCached(d.storage) : undefined;
    await smp_helper_main(d.worker, d.guest, d.audioRing, storage);
    self.postMessage({ type: "note", message: `guest worker ${d.worker} finished` });
  } catch (err) {
    self.postMessage({ type: "error", message: String((err && (err.stack || err.message)) || err) });
  }
};
