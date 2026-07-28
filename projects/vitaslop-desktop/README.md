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
