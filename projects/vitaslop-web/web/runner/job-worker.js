// job-worker.js - runs ONE job in its own dedicated Worker and reports to the runner page.
//
// A Worker, not an iframe: a same-origin iframe shares the page's main thread, so a job that
// compiles shaders synchronously froze the runner's own timeout and heartbeats (a 60 s kill
// fired at 81 s). Here the page stays responsive, the kill is exact (`terminate()`), and the
// termination hands back the job's whole heap.
//
// A job kind is a module `./jobs/<kind>.js` exporting `run(params, ctx)`, which returns a plain
// object (it becomes the result JSON; a `summary` string is shown in the page's log).
// `ctx.progress(text)` shows progress on the phone; `ctx.asset(path, as)` fetches from
// `/runner-assets/` and throws on anything but a 200 - an error body is never data.

async function asset(path, as = "text") {
  const r = await fetch(`/runner-assets/${path}`);
  if (!r.ok) throw new Error(`asset ${path}: HTTP ${r.status}`);
  if (as === "json") return r.json();
  if (as === "bytes") return new Uint8Array(await r.arrayBuffer());
  return r.text();
}

self.onmessage = async (e) => {
  const job = e.data?.job;
  if (!job) return;
  try {
    if (!/^[a-z0-9-]+$/.test(job.kind)) throw new Error(`bad job kind ${job.kind}`);
    const mod = await import(`./jobs/${job.kind}.js`);
    const result = await mod.run(job.params || {}, {
      progress: (text) => self.postMessage({ type: "progress", text }),
      asset,
    });
    self.postMessage({ type: "result", result });
  } catch (err) {
    self.postMessage({ type: "error", error: `${(err && err.message) || String(err)}
${(err && err.stack) || ""}`.slice(0, 4000) });
  }
};
self.postMessage({ type: "ready" });
