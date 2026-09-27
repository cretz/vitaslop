# vitaslop-desktop

> Keep this README terse: sectioned and bulleted, not large prose. Explain
> concepts and the why, not exact type names. Update it as the code changes so it
> never goes stale.

The native app: load -> transpile -> run -> capture -> wgpu, in a live winit
window with real keyboard and gamepad input. Same stack as the browser, wrapped
in a window instead of a canvas.

## Why it exists

- The browser is the target, but a browser is a poor place to debug a CPU core.
  This runs the identical path natively, so a difference between the two is a bug
  rather than a fork.
- It is also the **pixel oracle**: the headless mode renders frames through the
  same shared pipeline the canvas uses, which is what makes a render claim
  checkable.

## Live, not replayed

- The guest runs on the cooperative scheduler, one frame stepped per redraw, with
  the real pad injected between frames through the SceCtrl seam. Input genuinely
  reaches the guest.
- `--game <dir>` plays an extracted retail title (decrypt -> link -> transpile);
  without it, the committed clean-room cube.

## Headless

- Renders without a window, driven by a recipe, for scripted runs and shots. This
  is the judging path for titles the software renderer cannot draw - a screenshot
  from a different pass than the game shows is worse than no screenshot.
- Runtime `tracing` diagnostics surface through `RUST_LOG` (e.g.
  `vitaslop::io=trace`); see `KNOBS.md` for the env knobs.

## The shell

- `vitaslop` (the binary this crate builds) with no arguments opens the native shell (egui over the same wgpu
  surface the game presents to): library, title page, settings, import, an Esc menu in
  game, F11 fullscreen.
- `import <pkg|zip|folder>`, `list`, and `serve [--port N]` (hosts the embedded web
  bundle with the isolation headers) are the command-line faces of the same library.
- Data lives under `VITASLOP_HOME` or the per-user data directory: `library/<id>/` (the
  decrypted dump tree plus `meta.json`, `icon0.png`, `pic0.png`), `saves/<profile>/`,
  `settings.json`, `titles/<id>.json` (per-title patches).
- The COMPILE CACHE lives inside each game directory (library or `--game`):
  `<game>/vitaslop-transpiled/<build>-<settings>-<exec>.cwasm` (wasmtime's serialized
  compiled module, hundreds of MB) + `.layout` (written last). `<build>` is this
  executable's identity (size + mtime), so every new build invalidates; `<settings>` the
  transpiler's resolved emit settings; `<exec>` which executable of the game. Storing an
  entry deletes other builds' entries and the same executable's older ones; deleting the
  game deletes it. A hit skips the transpile and the Cranelift compile (MLB: 18.2 s -> 69 ms).
  `VITASLOP_COMPILE_CACHE=0` bypasses it. See `vitaslop_native::compile_cache`.
- `--game <dir>` is unchanged and is what the measurement rigs drive; `session.rs` is
  the loop both it and the shell share.
