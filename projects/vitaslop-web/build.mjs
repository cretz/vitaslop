// Builds the browser bundles, cross-platform (Node, no shell/OS assumptions): compile
// the crate to wasm, then run wasm-bindgen to generate the JS glue + processed wasm
// into web/pkg/ (and web/pkg-threads/). Serve web/ over HTTP afterwards (an ES module +
// WebGPU need an http origin, not file://), e.g.  `npx http-server projects/vitaslop-web/web`,
// then open the page in a WebGPU-capable browser.
//
// Usage:  node build.mjs                    # release (wasm-release profile), BOTH bundles
//         node build.mjs --debug            # dev profile, faster to iterate
//         node build.mjs --single-only      # only web/pkg (the default engine)
//         node build.mjs --threads-only     # only web/pkg-threads (VITASLOP_SMP)
//
// The wasm-bindgen CLI version must equal the crate's wasm-bindgen version (pinned in
// Cargo.toml). If it does not, run:
//   cargo install -f wasm-bindgen-cli --version <the pinned version>
import { spawnSync } from "node:child_process";
import { fileURLToPath } from "node:url";
import { dirname, join } from "node:path";

const crateDir = dirname(fileURLToPath(import.meta.url));
const projects = dirname(crateDir);
const debug = process.argv.includes("--debug");
const profile = debug ? "dev" : "wasm-release";
const profileDir = debug ? "debug" : "wasm-release";

// Run a command, inheriting stdio, and abort the build on a non-zero exit.
function run(cmd, args, env) {
  console.log(`> ${cmd} ${args.join(" ")}`);
  const r = spawnSync(cmd, args, { stdio: "inherit", shell: false, env: env || process.env });
  if (r.error) throw r.error;
  if (r.status !== 0) throw new Error(`${cmd} exited with ${r.status}`);
}

// `PROFILE_SYMBOLS=1` builds the diagnostic variant: a few audio-path functions are kept
// out of line so a V8 worker profile can attribute them separately. See the
// `profile-symbols` feature in vitaslop-runtime/Cargo.toml. Never use it for a timing A/B -
// it is slower by construction; it is for finding out WHERE the time goes, not how much.
const features = process.env.PROFILE_SYMBOLS
  ? ["--features", "vitaslop-runtime/profile-symbols"]
  : [];

// >>> TWO BUNDLES, AND THE DEFAULT ENGINE RUNS ON THE ONE IT ALWAYS DID.
//
// `web/pkg` is the single-threaded build every page loads. `web/pkg-threads` is the same crate
// built with WASM THREADS - a `shared` linear memory - which only `VITASLOP_SMP=1` loads (the
// run and transpile workers pick it from `?smp=1`, and `smp-worker.js` always uses it).
//
// WHY NOT ONE BUNDLE: the threads build costs the one-worker engine 5-8% of its frame
// (MEASURED `ab25a`, desktop Chrome, same source: football 14.4 -> 15.0 ms, fighting 6.3 -> 6.6, golf
// 5.4 -> 5.9) - atomics in every lock, TLS through a global, glue that re-checks a shared
// buffer. A run-time switch that taxed the deterministic default would not be a switch.
//
// WHY THE THREADS BUILD NEEDS ALL OF THIS: every SMP worker executes against the SAME guest
// memory, which lives inside this module's own linear memory (`browser_sched::HostRegion`). A
// memory can only be shared between workers if it is declared `shared`, and Rust only emits
// that when the crate AND `std` are compiled with `+atomics` - which is what `-Z build-std` is
// for (the prebuilt wasm32 `std` is single-threaded). The pinned stable toolchain accepts `-Z`
// under `RUSTC_BOOTSTRAP=1`; it needs the `rust-src` component. `+atomics` alone does NOT make
// the memory shared: the linker has to be told (`--shared-memory`), and wasm-bindgen only
// emits its threads glue - a memory passed in, per-worker TLS and stack - for a module that
// IMPORTS its memory; it finds the TLS and heap boundaries through the exported symbols.
// `--max-memory=4 GiB`: a shared memory must declare its maximum up front, and the linker's
// default for one (1 GiB) is below what a large title's heap reaches.
//
// RUSTFLAGS REPLACES `.cargo/config.toml`'s target rustflags rather than adding to them, so
// `+simd128` is restated; dropping it would silently scalarise every encoder.
const THREAD_FLAGS =
  "-C target-feature=+simd128,+atomics,+bulk-memory,+mutable-globals " +
  "-C link-arg=--shared-memory -C link-arg=--import-memory " +
  "-C link-arg=--max-memory=4294967296 " +
  "-C link-arg=--export=__wasm_init_tls -C link-arg=--export=__tls_size " +
  "-C link-arg=--export=__tls_align -C link-arg=--export=__tls_base " +
  "-C link-arg=--export=__heap_base -C link-arg=--export=__data_end";

const kinds = process.argv.includes("--single-only")
  ? [false]
  : process.argv.includes("--threads-only")
    ? [true]
    : [false, true];

function git(args) {
  const r = spawnSync("git", args, { cwd: projects, encoding: "utf8" });
  return r.status === 0 ? r.stdout.trim() : "?";
}

for (const threads of kinds) {
  const env = { ...process.env };
  const buildStd = [];
  if (threads) {
    env.RUSTC_BOOTSTRAP = "1";
    env.RUSTFLAGS = THREAD_FLAGS;
    buildStd.push("-Z", "build-std=std,panic_abort");
  }
  // A separate target directory per bundle kind: the two are built with different flags,
  // and sharing one would rebuild the world every time the kind changes.
  const targetDir = join(projects, threads ? "target-web-threads" : "target");
  const outName = threads ? "pkg-threads" : "pkg";
  console.log(`Building vitaslop-web (${profile}) for wasm32 -> web/${outName} ` +
    `(${threads ? "wasm THREADS, shared memory" : "single-threaded"})...`);
  run(
    "cargo",
    [
      "build",
      "--manifest-path",
      join(projects, "Cargo.toml"),
      "-p",
      "vitaslop-web",
      "--target",
      "wasm32-unknown-unknown",
      "--profile",
      profile,
      "--target-dir",
      targetDir,
      ...buildStd,
      ...features,
    ],
    env
  );

  const wasm = join(targetDir, "wasm32-unknown-unknown", profileDir, "vitaslop_web.wasm");
  const out = join(crateDir, "web", outName);
  console.log(`Running wasm-bindgen -> ${out}`);
  run("wasm-bindgen", ["--target", "web", "--out-dir", out, "--out-name", "vitaslop_web", wasm]);

  // >>> THE BUNDLE STAMPS ITSELF, so a diagnostic taken on a phone says WHICH BUILD produced it.
  //
  // This was written after a night's three separate renderer fixes all read as "still broken"
  // in a phone dump, and nothing in that dump could say whether the device had actually loaded
  // the bundle those fixes were in. A stale service-worker copy, a cached wasm, a page left open
  // from before the rebuild, or simply the wrong host all produce the same evidence as a fix
  // that did not work - and the second reading is the expensive one, because it sends a night
  // at the wrong problem. The stamp is written HERE, beside the wasm-bindgen output, so it
  // cannot drift from the bytes it names: whatever wrote `vitaslop_web_bg.wasm` wrote this in
  // the same breath.
  const { writeFileSync, statSync } = await import("node:fs");
  const rev = git(["rev-parse", "--short", "HEAD"]);
  const dirty = git(["status", "--porcelain"]).split("\n").filter((l) => l.trim()).length;
  const wasmBytes = statSync(join(out, "vitaslop_web_bg.wasm")).size;
  const stamp =
    `${new Date().toISOString()} ${profile}${threads ? "+threads" : ""} ${rev}` +
    `${dirty ? `+${dirty}dirty` : ""} ${wasmBytes} bytes`;
  writeFileSync(join(out, "build-stamp.txt"), stamp + "\n");
  console.log(`build stamp: ${stamp}`);
}

console.log("Done. Serve projects/vitaslop-web/web over HTTP and open the page.");
