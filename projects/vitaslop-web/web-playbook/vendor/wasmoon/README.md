# wasmoon 1.16.0 (vendored)

The browser's Lua VM for playbooks (`vitaslop-playbook`): PUC-Rio C Lua 5.4 compiled to wasm
with emscripten, wrapped in a JS API. MIT - see `LICENSE`.

- `wasmoon.js` is the npm package's `dist/index.js` (UMD; importing it defines `self.wasmoon`).
- `glue.wasm` is the package's `dist/glue.wasm`.
- Source: `https://registry.npmjs.org/wasmoon/-/wasmoon-1.16.0.tgz`, sha256
  `7a07cb6d39af10911fabd50740833477b18b47b6f03148f15218edf474c8af56`. Unmodified.

Loaded only by a run that has a playbook (`worker.js`, `loadPlaybookLua`). The engine reaches it
through two functions, so swapping it for another Lua is a change to that one loader. It lives
here, in `web-playbook/`, so only the playbook build serves it and Pages never publishes it
(see `playbook-build.mjs`).
