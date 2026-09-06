// Per-PROCESS, per-sample memory and CPU telemetry for a browser run, sampled from
// OUTSIDE the browser.
//
// Why this exists, and why the obvious cheaper thing is not enough:
//
//   * The page cannot report the moment that matters. A renderer or worker that is KILLED
//     does not get to log its own death, and `performance.measureUserAgentSpecificMemory`
//     excludes exactly the allocation under suspicion (a multi-gigabyte wasm heap plus the
//     JIT code for a whole retail title). The OS number includes everything and survives
//     the process.
//
//   * A SUM over `chrome` processes cannot answer the question that decides the
//     diagnosis. "Total Chrome dropped to 0.15 GB" is true whether the renderer PROCESS
//     was killed (an out-of-memory kill, and the lever is footprint) or the renderer is
//     alive and only its Web Worker stopped (not memory at all, and the whole framing has
//     to change). Those need opposite fixes. Only per-PID sampling separates them, by
//     saying whether the renderer's PID is still in the list one sample later.
//
//   * A 15-second cadence cannot see a climb that ends in a kill. The interesting window
//     here is a few seconds wide, so the sampler has to run at a few hundred milliseconds
//     and cost nothing per sample.
//
// One long-lived PowerShell child does the sampling. Spawning a process per sample costs
// ~100 ms on Windows, which is longer than the interval we need, so the loop lives inside
// the child and streams CSV back. Process TYPE (renderer / gpu-process / utility) comes
// from the command line, queried ONCE per newly-seen PID rather than every sample - a
// `Win32_Process` query is far more expensive than `Get-Process` and the type never
// changes.
//
// Output CSV columns:  t_ms,pid,kind,ws_bytes,priv_bytes,cpu_ms
// plus a `#type` comment line the first time each PID is seen.
import { spawn } from "node:child_process";
import { createWriteStream } from "node:fs";

// PowerShell sampler. Emits one CSV row per chrome PID per tick, and a `#type` line the
// first time a PID appears. `--type=` is Chrome's own process-kind flag; the browser
// process has no `--type` at all, hence the `browser` default.
const SCRIPT = `
$ErrorActionPreference = 'SilentlyContinue'
$interval = %INTERVAL%
$kinds = @{}
while ($true) {
  $t = [DateTimeOffset]::UtcNow.ToUnixTimeMilliseconds()
  $procs = Get-Process chrome
  foreach ($p in $procs) {
    if (-not $kinds.ContainsKey($p.Id)) {
      $cl = (Get-CimInstance Win32_Process -Filter "ProcessId=$($p.Id)").CommandLine
      $kind = 'browser'
      if ($cl -match '--type=([a-zA-Z-]+)') { $kind = $Matches[1] }
      if ($kind -eq 'utility' -and $cl -match '--utility-sub-type=([^ ]+)') {
        $kind = 'utility:' + ($Matches[1] -replace '.*\\.', '')
      }
      $kinds[$p.Id] = $kind
      "#type,$($p.Id),$kind"
    }
    $cpu = 0
    try { $cpu = [int]($p.TotalProcessorTime.TotalMilliseconds) } catch { $cpu = -1 }
    "$t,$($p.Id),$($kinds[$p.Id]),$($p.WorkingSet64),$($p.PrivateMemorySize64),$cpu"
  }
  [Console]::Out.Flush()
  Start-Sleep -Milliseconds $interval
}
`;

/**
 * Start sampling every Chrome process into `outPath`.
 *
 * Returns a handle with `stop()` and `snapshot()`. `snapshot()` is the live per-PID view
 * the run's own progress lines use, so a progress line names which process is growing
 * instead of reporting one number that could be any of them.
 */
export function startProcMon({ intervalMs = 250, outPath }) {
  const out = createWriteStream(outPath, { flags: "w" });
  out.write("t_ms,pid,kind,ws_bytes,priv_bytes,cpu_ms\n");

  const child = spawn(
    "powershell.exe",
    ["-NoProfile", "-NonInteractive", "-Command", SCRIPT.replace("%INTERVAL%", String(intervalMs))],
    { stdio: ["ignore", "pipe", "pipe"] }
  );

  /** pid -> { kind, ws, priv, cpu, t } for the most recent sample that named it. */
  const live = new Map();
  /** pid -> kind, kept after a PID disappears so a death can still be named. */
  const kinds = new Map();
  /** PIDs that were present and then vanished, with the sample time they were last seen. */
  const gone = [];

  let tail = "";
  // The tick currently being accumulated, across stream chunks.
  let tickT = null;
  let tickSeen = new Set();
  child.stdout.setEncoding("utf8");
  child.stdout.on("data", (chunk) => {
    out.write(chunk);
    tail += chunk;
    const lines = tail.split("\n");
    tail = lines.pop() ?? "";
    // A tick is the group of rows sharing one timestamp. Deaths are detected by comparing
    // consecutive ticks' PID sets, which is why rows are accumulated per timestamp rather
    // than handled one at a time.
    for (const raw of lines) {
      const line = raw.trim();
      if (!line) continue;
      if (line.startsWith("#type,")) {
        const [, pid, kind] = line.split(",");
        kinds.set(Number(pid), kind);
        continue;
      }
      const [t, pid, kind, ws, priv, cpu] = line.split(",");
      const n = Number(pid);
      if (!Number.isFinite(n)) continue;
      // A tick is only COMPLETE once a row from a LATER tick has arrived. Reconciling the
      // trailing tick of a stream chunk instead reports every process that happened to
      // fall after the chunk boundary as dead, and then alive again on the next chunk -
      // which is a monitor manufacturing exactly the event it exists to detect.
      if (tickT !== null && t !== tickT) {
        reconcile(tickT, tickSeen);
        tickSeen = new Set();
      }
      tickT = t;
      tickSeen.add(n);
      kinds.set(n, kind);
      live.set(n, { kind, ws: Number(ws), priv: Number(priv), cpu: Number(cpu), t: Number(t) });
    }
  });
  child.stderr.setEncoding("utf8");
  child.stderr.on("data", (d) => process.stderr.write(`[procmon] ${d}`));

  // Retire PIDs that were in `live` but absent from a completed tick. A PID vanishing is
  // the single most diagnostic event this monitor produces: it is the difference between
  // a process that was killed and a thread inside a healthy process that stopped.
  const onDeath = [];
  function reconcile(t, seen) {
    // Only reconcile against a tick that actually listed processes; an empty tick means
    // the query failed, not that Chrome exited.
    if (seen.size === 0) return;
    for (const pid of [...live.keys()]) {
      if (!seen.has(pid)) {
        const last = live.get(pid);
        live.delete(pid);
        gone.push({ pid, kind: last.kind, lastWs: last.ws, atMs: Number(t) });
        for (const f of onDeath) f({ pid, kind: last.kind, lastWs: last.ws, atMs: Number(t) });
      }
    }
  }

  return {
    /** Register a callback fired the moment a Chrome PID disappears. */
    onProcessGone(f) {
      onDeath.push(f);
    },
    /** PIDs that have vanished so far, in order. */
    deaths: () => gone.slice(),
    /**
     * A one-line per-kind summary for a progress line: the largest process of each kind,
     * so "which one is growing" is answerable without opening the CSV.
     */
    summary() {
      if (live.size === 0) return "";
      const byKind = new Map();
      for (const [pid, v] of live) {
        const cur = byKind.get(v.kind);
        if (!cur || v.ws > cur.ws) byKind.set(v.kind, { pid, ws: v.ws });
      }
      const parts = [...byKind.entries()]
        .sort((a, b) => b[1].ws - a[1].ws)
        .slice(0, 4)
        .map(([kind, v]) => `${kind}:${(v.ws / 1e9).toFixed(2)}G`);
      return ` | ${parts.join(" ")}`;
    },
    stop() {
      try {
        child.kill();
      } catch {}
      out.end();
    },
  };
}
