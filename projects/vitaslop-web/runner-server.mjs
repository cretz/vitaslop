// runner-server.mjs - the DEVICE TEST RUNNER's server half, mounted by serve.mjs.
//
// >>> WHY THIS EXISTS. Every question about a phone used to cost the user a hand-driven run:
// open a page, play to a frame, download a dump, send it. A morning went on four of those for a
// defect one instrumented run named. This turns a phone into a machine the desktop drives: the
// phone keeps `/runner/` open, and jobs queued here (by a shell `curl`, by a script) run on it
// and post their results back - no person in the loop, any number of devices at once.
//
// >>> BUILT FOR PHONES, WHICH DISAPPEAR. A phone is backgrounded, locked, taken away, reloaded.
// So every job is LEASED, not handed over: a device that vanishes mid-job lets its lease lapse
// and the job goes back to the queue. Jobs are SHORT by contract (the page kills one at its
// `timeoutMs`, capped at MAX_TIMEOUT_MS): a phone page is not the place for a multi-minute
// sweep, and one that overruns is reported as a timeout rather than left running.
//
// >>> STATE IS PLAIN FILES under `dir`, so it survives a server restart and a person can read
// it with `ls` and `cat`:
//   devices/<id>.json          fingerprint + lastSeen, one per device that ever said hello
//   queue/<job>.json           a job still to run (with its lease, when one is out)
//   done/<job>.json            a job that finished (for target "all": never moved - see below)
//   results/<job>/<device>.json  what the device reported
//   assets/...                 files jobs fetch (case directories, capsules), at /runner-assets/
//
// TARGETS: a device id (only that device), "any" (the first device to ask), "all" (every
// device that asks, once each - the job stays queued and "done for this device" means a
// result file exists). The API (all JSON):
//   POST /runner/queue          {kind, params, target?, timeoutMs?, note?} -> {id}
//   GET  /runner/next?device=   the next job for this device, leased to it; 204 if none
//   POST /runner/hello?device=  the device's fingerprint
//   POST /runner/beat?device=&job=   extends a lease while a job runs; replies {cancel:true} to end it
//   POST /runner/cancel?job=    cancel a queued or running job
//   POST /runner/result?device=&job= the job's result
//   GET  /runner/status         devices, queue, the latest results
//   GET  /runner/result?job=[&device=]  one job's result(s)
//   GET  /runner/lane.js        the seeded case-input generator (e2e/caseinputs.mjs, ONE copy)
//   GET  /runner-assets/<path>  a file under assets/; a directory with ?list gives its names

import { readFile, writeFile, readdir, mkdir, rename, stat } from "node:fs/promises";
import { existsSync, createReadStream } from "node:fs";
import { pipeline } from "node:stream/promises";
import { join, normalize, sep } from "node:path";
import { createHash } from "node:crypto";

/// A job's longest allowed run. A phone that is needed "here and there" cannot host a sweep.
export const MAX_TIMEOUT_MS = 90_000;
/// A LIVE job (the installed game played by a recipe, web/runner/live.html) has to boot, reach
/// the frames it measures and measure them; still bounded, so a phone is not held for long.
export const MAX_LIVE_TIMEOUT_MS = 360_000;
/// A SOAK (a live job with `params.wallMs`): a defect that needs time to show (a renderer that
/// dies after ~40 min of play) cannot be asked in six minutes. The page stops itself at wallMs.
export const MAX_SOAK_TIMEOUT_MS = 3_300_000;
const DEFAULT_TIMEOUT_MS = 60_000;
/// How long a lease outlives its last heartbeat before the job is offered again.
const LEASE_MS = 25_000;

export function createRunner({ dir, coi, laneSrc, runnerWebDir }) {
  const sub = (...p) => join(dir, ...p);
  let seq = 0;
  const safe = (s) => String(s || "").replace(/[^A-Za-z0-9_-]/g, "").slice(0, 64);

  const readJson = async (p) => JSON.parse(await readFile(p, "utf8"));
  const writeJson = async (p, v) => {
    const tmp = `${p}.tmp`;
    await writeFile(tmp, JSON.stringify(v, null, 2));
    await rename(tmp, p);
  };
  const body = async (req) => {
    const chunks = [];
    for await (const c of req) chunks.push(c);
    return Buffer.concat(chunks).toString("utf8");
  };
  const json = (res, code, v) => {
    res.writeHead(code, { "content-type": "application/json", ...coi });
    res.end(JSON.stringify(v));
  };
  const list = async (d) => {
    try {
      return (await readdir(d)).filter((f) => f.endsWith(".json")).sort();
    } catch {
      return [];
    }
  };

  /// A stamp of the runner PAGE's own files. The page reloads itself when it changes, so an
  /// update reaches a phone nobody is holding. ONLY the page's files: job code (`jobs/`, the
  /// shared modules) is loaded fresh by every job's Worker anyway, and counting it reloaded the
  /// phone on every job edit - which drops the screen wake lock until someone taps Start.
  const PAGE_FILES = ["index.html", "runner.js", "fingerprint.js", "job-worker.js"];
  const runnerVersion = async () => {
    let v = 0;
    for (const f of PAGE_FILES) {
      try {
        v = Math.max(v, (await stat(join(runnerWebDir, f))).mtimeMs);
      } catch {}
    }
    return String(Math.floor(v));
  };

  const shaCache = new Map();
  const shaOf = async (full, st) => {
    const key = `${full}|${st.mtimeMs}|${st.size}`;
    if (shaCache.has(key)) return shaCache.get(key);
    const h = createHash("sha256");
    await pipeline(createReadStream(full), h);
    const sha = h.digest("hex");
    shaCache.set(key, sha);
    return sha;
  };

  const doneFor = (job, device) => existsSync(sub("results", job.id, `${device}.json`));

  /// Whether `device` may take `job` now.
  const offerable = (job, device, now) => {
    // A cancelled job waits only for its running device's report - never hand it out again.
    // MEASURED (10-03a, job 533): the phone ended a cancelled run without a result, polled
    // `next`, was handed the same job back under its own lease, and ran it from the start.
    if (job.cancel) return false;
    const target = job.target || "any";
    if (target !== "any" && target !== "all" && target !== device) return false;
    if (target === "all" && doneFor(job, device)) return false;
    const leases = job.leases || {};
    if (target === "all") {
      const mine = leases[device];
      return !mine || mine.until < now;
    }
    const held = Object.entries(leases).find(([, l]) => l.until >= now);
    return !held || held[0] === device;
  };

  return async function handle(req, res, path, url) {
    if (path === "/runner/lane.js") {
      res.writeHead(200, { "content-type": "text/javascript", ...coi });
      res.end(laneSrc);
      return true;
    }
    if (path.startsWith("/runner-assets/")) {
      const rel = normalize(path.slice("/runner-assets/".length)).replace(/^([/\\])+/, "");
      if (rel.split(sep).includes("..")) {
        json(res, 400, { error: "bad path" });
        return true;
      }
      const full = sub("assets", rel);
      let st;
      try {
        st = await stat(full);
      } catch {
        json(res, 404, { error: `no asset ${rel}` });
        return true;
      }
      if (st.isDirectory()) {
        json(res, 200, (await readdir(full)).sort());
        return true;
      }
      // The content's SHA-256, so a device can keep a copy and prove it current before using
      // it (web/runner/job-worker.js): a 54 MB frame re-downloaded per job was ~65 s of every
      // 90 s job on the phone. Computed once per (path, mtime, size).
      const sha = await shaOf(full, st);
      if (url.searchParams.has("sha")) {
        json(res, 200, { sha, size: st.size });
        return true;
      }
      res.writeHead(200, { "content-type": "application/octet-stream", "content-length": st.size, "x-content-sha256": sha, ...coi });
      await pipeline(createReadStream(full), res);
      return true;
    }
    if (!path.startsWith("/runner/") || path.endsWith(".js") || path.endsWith(".html") || path === "/runner/") {
      return false; // the page's own static files are served by serve.mjs
    }
    for (const d of ["devices", "queue", "done", "results", "assets"]) await mkdir(sub(d), { recursive: true });
    const device = safe(url.searchParams.get("device"));
    const jobId = safe(url.searchParams.get("job"));
    const now = Date.now();

    if (path === "/runner/queue" && req.method === "POST") {
      const spec = JSON.parse(await body(req));
      if (!spec.kind) return json(res, 400, { error: "a job needs a kind" }), true;
      const id = `${new Date().toISOString().replace(/[-:T.Z]/g, "").slice(0, 14)}-${String(++seq).padStart(3, "0")}-${safe(spec.kind)}`;
      const job = {
        id,
        kind: spec.kind,
        params: spec.params || {},
        target: spec.target || "any",
        // An import is resumable only per FILE, so its cap has to fit the largest file: a
        // title's 3 GB archive restarted from zero on every 90-second job and never finished.
        timeoutMs: Math.min(
          Number(spec.timeoutMs) || DEFAULT_TIMEOUT_MS,
          spec.kind === "import-title" ? MAX_SOAK_TIMEOUT_MS
            : spec.kind !== "live" ? MAX_TIMEOUT_MS
            : spec.params?.wallMs ? MAX_SOAK_TIMEOUT_MS : MAX_LIVE_TIMEOUT_MS
        ),
        note: spec.note || "",
        queued: new Date(now).toISOString(),
        leases: {},
      };
      await writeJson(sub("queue", `${id}.json`), job);
      json(res, 200, { id, timeoutMs: job.timeoutMs });
      return true;
    }
    // Notes from the desktop to whoever is holding the phone: the runner page prints them in
    // its log, which is often the only screen that person can see.
    //   POST /runner/say   body = the text      GET /runner/say?since=<n>  -> [{n, at, text}]
    if (path === "/runner/say") {
      const f = sub("say.json");
      let says = [];
      try {
        says = await readJson(f);
      } catch {}
      if (req.method === "POST") {
        const text = (await body(req)).trim().slice(0, 2000);
        says.push({ n: (says.at(-1)?.n || 0) + 1, at: new Date(now).toISOString(), text });
        await writeJson(f, says.slice(-200));
        json(res, 200, { n: says.at(-1).n });
        return true;
      }
      const since = Number(url.searchParams.get("since") || 0);
      json(res, 200, says.filter((m) => m.n > since));
      return true;
    }
    if (path === "/runner/hello" && req.method === "POST") {
      if (!device) return json(res, 400, { error: "device= required" }), true;
      const fp = JSON.parse(await body(req));
      await writeJson(sub("devices", `${device}.json`), { ...fp, device, lastSeen: new Date(now).toISOString() });
      json(res, 200, { ok: true, version: await runnerVersion() });
      return true;
    }
    if (path === "/runner/next") {
      if (!device) return json(res, 400, { error: "device= required" }), true;
      const version = await runnerVersion();
      // Seen, even with nothing to do: `status` shows which phones are listening right now.
      const devFile = sub("devices", `${device}.json`);
      if (existsSync(devFile)) {
        try {
          const d = await readJson(devFile);
          d.lastSeen = new Date(now).toISOString();
          await writeJson(devFile, d);
        } catch {}
      }
      for (const f of await list(sub("queue"))) {
        let job;
        try {
          job = await readJson(sub("queue", f));
        } catch {
          continue;
        }
        if (!offerable(job, device, now)) continue;
        job.leases = job.leases || {};
        job.leases[device] = { until: now + job.timeoutMs + LEASE_MS, taken: new Date(now).toISOString() };
        await writeJson(sub("queue", f), job);
        res.writeHead(200, { "content-type": "application/json", "x-runner-version": version, ...coi });
        res.end(JSON.stringify({ ...job, leases: undefined }));
        return true;
      }
      res.writeHead(204, { "x-runner-version": version, ...coi });
      res.end();
      return true;
    }
    // POST /runner/cancel?job=  - end a queued or running job: a queued one moves to done/ with a
    // `cancelled` result; a running one is told so by its next heartbeat reply, and the page
    // removes the job's frame (runner.js).
    if (path === "/runner/cancel" && req.method === "POST") {
      const f = sub("queue", `${jobId}.json`);
      if (!existsSync(f)) return json(res, 404, { error: `no queued job ${jobId}` }), true;
      const job = await readJson(f);
      job.cancel = true;
      await writeJson(f, job);
      if (!job.leases || Object.keys(job.leases).length === 0) {
        await mkdir(sub("results", jobId), { recursive: true });
        await writeJson(sub("results", jobId, "cancelled.json"), { status: "cancelled", job: jobId, at: new Date(now).toISOString() });
        await rename(f, sub("done", `${jobId}.json`));
      }
      json(res, 200, { ok: true, running: !!job.leases && Object.keys(job.leases).length > 0 });
      return true;
    }
    if (path === "/runner/beat" && req.method === "POST") {
      const f = sub("queue", `${jobId}.json`);
      if (existsSync(f)) {
        const job = await readJson(f);
        if (job.cancel) return json(res, 200, { ok: true, cancel: true }), true;
        if (job.leases?.[device]) {
          job.leases[device].until = now + LEASE_MS;
          // What the device last said it was doing (a live job's frame) - `status` shows it.
          const p = url.searchParams.get("p");
          if (p) job.leases[device].progress = `${new Date(now).toISOString().slice(11, 19)} ${p.slice(0, 200)}`;
          await writeJson(f, job);
        }
      }
      json(res, 200, { ok: true });
      return true;
    }
    if (path === "/runner/result" && req.method === "POST") {
      if (!device || !jobId) return json(res, 400, { error: "device= and job= required" }), true;
      const result = JSON.parse(await body(req));
      await mkdir(sub("results", jobId), { recursive: true });
      await writeJson(sub("results", jobId, `${device}.json`), { ...result, device, job: jobId, at: new Date(now).toISOString() });
      const f = sub("queue", `${jobId}.json`);
      if (existsSync(f)) {
        const job = await readJson(f);
        if (job.target === "all") {
          if (job.leases) delete job.leases[device];
          await writeJson(f, job);
        } else {
          await rename(f, sub("done", `${jobId}.json`));
        }
      }
      json(res, 200, { ok: true });
      return true;
    }
    if (path === "/runner/result" && req.method === "GET") {
      const d = sub("results", jobId);
      const out = {};
      for (const f of await list(d)) {
        if (device && f !== `${device}.json`) continue;
        out[f.replace(/\.json$/, "")] = await readJson(join(d, f));
      }
      json(res, 200, out);
      return true;
    }
    if (path === "/runner/status") {
      const devices = [];
      for (const f of await list(sub("devices"))) {
        const d = await readJson(sub("devices", f));
        devices.push({
          device: d.device,
          name: d.name,
          lastSeen: d.lastSeen,
          agoS: Math.round((now - Date.parse(d.lastSeen)) / 1000),
          gpu: d.adapter,
          ua: d.ua,
        });
      }
      const queue = [];
      for (const f of await list(sub("queue"))) {
        const j = await readJson(sub("queue", f));
        queue.push({ id: j.id, kind: j.kind, target: j.target, note: j.note, leases: j.leases });
      }
      const done = (await list(sub("done"))).slice(-10).map((f) => f.replace(/\.json$/, ""));
      json(res, 200, { version: await runnerVersion(), devices, queue, recentDone: done });
      return true;
    }
    json(res, 404, { error: `no runner route ${path}` });
    return true;
  };
}
