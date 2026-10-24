// Forward the page's live pointer + keyboard input to the emulator worker. A worker
// has no DOM, so the page listens and posts each event; the worker applies it to the
// run's input world via the exported worker_input_* functions. Pointer events on the
// canvas become front-panel touches (mapped into the Vita's 960x544 screen space, which
// the worker doubles to panel coords); key events carry their KeyboardEvent.code, which
// the worker maps to SceCtrl buttons.
//
// `opts.rearTouch` (the opt-in "Rear touch panel" setting) adds the BACK panel: a right-button
// drag with a mouse, or any touch while `opts.rearLatched()` says the on-screen Rear toggle is
// on. Off, nothing below differs from a run without the setting.
export function forwardInput(worker, canvas, opts = {}) {
  const SCREEN_W = 960;
  const SCREEN_H = 544;

  // The picture is fitted inside the element (`object-fit: contain`), so the element's
  // box can be wider or taller than the drawn 960x544: the letterbox has to be taken
  // off before scaling, or every touch lands short of where the finger is.
  const toScreen = (e) => {
    const r = canvas.getBoundingClientRect();
    const fit = getComputedStyle(canvas).objectFit;
    let { left, top, width, height } = r;
    if (fit !== "fill" && width > 0 && height > 0) {
      const k = Math.min(width / SCREEN_W, height / SCREEN_H);
      const dw = SCREEN_W * k;
      const dh = SCREEN_H * k;
      left += (width - dw) / 2;
      top += (height - dh) / 2;
      width = dw;
      height = dh;
    }
    const x = Math.min(SCREEN_W, Math.max(0, ((e.clientX - left) / Math.max(1, width)) * SCREEN_W));
    const y = Math.min(SCREEN_H, Math.max(0, ((e.clientY - top) / Math.max(1, height)) * SCREEN_H));
    return { x, y };
  };

  // Every listener is remembered so the caller can take them off again. The KEY ones are on
  // `document`, which outlives any one run: without this, quitting a game and starting
  // another left the first run's handlers installed, so each key posted to a worker that had
  // already been terminated, once more for every game played this session.
  const off = [];
  const on = (target, type, fn, opts) => {
    target.addEventListener(type, fn, opts);
    off.push(() => target.removeEventListener(type, fn, opts));
  };

  const pointer = (down, requireButton) => (e) => {
    if (requireButton && (e.buttons & 1) === 0) return; // only track drags while pressed
    const { x, y } = toScreen(e);
    worker.postMessage({ type: "pointer", x, y, down });
  };
  if (!opts.rearTouch) {
    on(canvas, "pointerdown", pointer(true, false));
    on(canvas, "pointermove", pointer(true, true));
    on(canvas, "pointerup", pointer(false, false));
    on(canvas, "pointerleave", pointer(false, false));
  } else {
    // Which panel each press went to, BY POINTER, so its move and release go to the same one -
    // a second contact made while the Rear toggle changed must not move the first one's panel.
    const panelOf = new Map();
    const toBack = (e) =>
      e.pointerType === "mouse" ? e.button === 2 || (e.buttons & 2) !== 0 : !!(opts.rearLatched && opts.rearLatched());
    const post = (e, back, down) => {
      const { x, y } = toScreen(e);
      worker.postMessage({ type: back ? "back-pointer" : "pointer", x, y, down });
    };
    const release = (e) => {
      if (!panelOf.has(e.pointerId)) return;
      post(e, panelOf.get(e.pointerId), false);
      panelOf.delete(e.pointerId);
    };
    on(canvas, "pointerdown", (e) => {
      const back = toBack(e);
      panelOf.set(e.pointerId, back);
      post(e, back, true);
    });
    on(canvas, "pointermove", (e) => {
      if (!panelOf.has(e.pointerId)) return;
      const back = panelOf.get(e.pointerId);
      // Only a MOUSE reaches the rear panel through its right button; a touch or pen contact is
      // pressed on bit 0 whichever panel the toggle sent it to.
      const held = e.pointerType === "mouse" && back ? 2 : 1;
      if ((e.buttons & held) !== 0) post(e, back, true);
    });
    on(canvas, "pointerup", release);
    on(canvas, "pointerleave", release);
    on(canvas, "pointercancel", release);
    // The right button is the rear panel here, not the browser's menu.
    on(canvas, "contextmenu", (e) => e.preventDefault());
  }

  const scrollers = new Set(["ArrowUp", "ArrowDown", "ArrowLeft", "ArrowRight", "Space", "Enter"]);
  const key = (pressed) => (e) => {
    worker.postMessage({ type: "key", code: e.code, pressed });
    if (scrollers.has(e.code)) e.preventDefault(); // don't scroll the page while playing
  };
  on(document, "keydown", key(true));
  on(document, "keyup", key(false));

  return () => {
    for (const undo of off.splice(0)) undo();
  };
}
