// Builds the browser bundle, cross-platform (Node, no shell/OS assumptions): compile
// the crate to wasm, then run wasm-bindgen to generate the JS glue + processed wasm
// into web/pkg/. Serve web/ over HTTP afterwards (an ES module + WebGPU need an http
// origin, not file://), e.g.  `npx http-server projects/vitaslop-web/web`, then open
// the page in a WebGPU-capable browser.
//
// Usage:  node build.mjs           # release (wasm-release profile), for a perf read
//         node build.mjs --debug   # dev profile, faster to iterate
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
function run(cmd, args) {
  console.log(`> ${cmd} ${args.join(" ")}`);
  const r = spawnSync(cmd, args, { stdio: "inherit", shell: false });
  if (r.error) throw r.error;
  if (r.status !== 0) throw new Error(`${cmd} exited with ${r.status}`);
}

console.log(`Building vitaslop-web (${profile}) for wasm32...`);
// `PROFILE_SYMBOLS=1` builds the diagnostic variant: a few audio-path functions are kept
// out of line so a V8 worker profile can attribute them separately. See the
// `profile-symbols` feature in vitaslop-runtime/Cargo.toml. Never use it for a timing A/B -
// it is slower by construction; it is for finding out WHERE the time goes, not how much.
const features = process.env.PROFILE_SYMBOLS
  ? ["--features", "vitaslop-runtime/profile-symbols"]
  : [];
run("cargo", [
  "build",
  "--manifest-path",
  join(projects, "Cargo.toml"),
  "-p",
  "vitaslop-web",
  "--target",
  "wasm32-unknown-unknown",
  "--profile",
  profile,
  ...features,
]);

const wasm = join(projects, "target", "wasm32-unknown-unknown", profileDir, "vitaslop_web.wasm");
const out = join(crateDir, "web", "pkg");
console.log(`Running wasm-bindgen -> ${out}`);
run("wasm-bindgen", ["--target", "web", "--out-dir", out, "--out-name", "vitaslop_web", wasm]);

console.log("Done. Serve projects/vitaslop-web/web over HTTP and open the page.");
