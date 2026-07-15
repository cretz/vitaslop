// Forward the page's live pointer + keyboard input to the emulator worker. A worker
// has no DOM, so the page listens and posts each event; the worker applies it to the
// run's input world via the exported worker_input_* functions. Pointer events on the
// canvas become front-panel touches (mapped into the Vita's 960x544 screen space, which
// the worker doubles to panel coords); key events carry their KeyboardEvent.code, which
// the worker maps to SceCtrl buttons.
export function forwardInput(worker, canvas) {
  const SCREEN_W = 960;
  const SCREEN_H = 544;

  const toScreen = (e) => {
    const r = canvas.getBoundingClientRect();
    const x = ((e.clientX - r.left) / Math.max(1, r.width)) * SCREEN_W;
    const y = ((e.clientY - r.top) / Math.max(1, r.height)) * SCREEN_H;
    return { x, y };
  };

  const pointer = (down, requireButton) => (e) => {
    if (requireButton && (e.buttons & 1) === 0) return; // only track drags while pressed
    const { x, y } = toScreen(e);
    worker.postMessage({ type: "pointer", x, y, down });
  };
  canvas.addEventListener("pointerdown", pointer(true, false));
  canvas.addEventListener("pointermove", pointer(true, true));
  canvas.addEventListener("pointerup", pointer(false, false));
  canvas.addEventListener("pointerleave", pointer(false, false));

  const scrollers = new Set(["ArrowUp", "ArrowDown", "ArrowLeft", "ArrowRight", "Space", "Enter"]);
  const key = (pressed) => (e) => {
    worker.postMessage({ type: "key", code: e.code, pressed });
    if (scrollers.has(e.code)) e.preventDefault(); // don't scroll the page while playing
  };
  document.addEventListener("keydown", key(true));
  document.addEventListener("keyup", key(false));
}
