// >>> THE PLAYBOOK BUILD IS AN OVERLAY ON `web/`, never a second copy of the page.
//
// The shipped bundles (`web/pkg`, `web/pkg-threads`) are built WITHOUT the `playbook` feature: a
// player never runs a playbook, so its engine and exports stay out of the wasm, and the Lua VM
// (wasmoon) stays out of what Pages publishes. `node build.mjs --playbook` builds the same two
// bundles WITH it into `web-playbook/`, beside the vendored wasmoon. A harness that runs playbooks
// (e2e/game-boot.mjs with PLAYBOOK, serve.mjs and its device runner) serves `web/` with these
// paths answered from `web-playbook/` instead - same URLs, so no page or worker knows which build
// it is on except through the bundle's stamp (`+playbook`).
import { existsSync } from "node:fs";
import { dirname, join } from "node:path";
import { fileURLToPath } from "node:url";

export const playbookWebDir = process.env.PLAYBOOK_WEB_DIR || join(dirname(fileURLToPath(import.meta.url)), "web-playbook");

const OVERLAID = ["/pkg/", "/pkg-threads/", "/vendor/wasmoon/"];

/// The `web-playbook/` file that answers `urlPath`, or null when `web/` answers it.
export function playbookFile(urlPath) {
  return OVERLAID.some((p) => urlPath.startsWith(p)) ? join(playbookWebDir, urlPath.slice(1)) : null;
}

/// Exit with a clear message when the playbook bundles have not been built. A harness that fell
/// back to `web/pkg` would fail its first playbook at start with a less useful error.
export function requirePlaybookBundle() {
  for (const dir of ["pkg", "pkg-threads"]) {
    if (!existsSync(join(playbookWebDir, dir, "vitaslop_web.js"))) {
      console.error(`Playbook bundle missing (${join(playbookWebDir, dir)}): run \`node build.mjs --playbook\` first.`);
      process.exit(2);
    }
  }
}
