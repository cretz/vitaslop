// On-screen controls and gamepad support for the live player page.
//
// The emulator runs in a Web Worker, which has no DOM, so every input arrives as a message
// (see worker.js). This module produces those messages from three sources:
//
//   - an on-screen d-pad, face buttons and thumb stick, for a phone with no keyboard;
//   - a real gamepad through the Gamepad API, polled once per animation frame;
//   - the page's keyboard, via worker-input.js (unchanged, and still live alongside these).
//
// # Why a thumb stick and not just a d-pad
// One retail racer steers entirely with the LEFT STICK as a turn RATE - a d-pad press is not
// a substitute for it, and until now the browser had no way to send a stick at all. The stick
// is reported in the guest's own 0..255 encoding with 128 centred, and released (not centred)
// on lift, so a scripted recipe keeps steering when nobody is touching the screen.
//
// # Why the controls live OUTSIDE the canvas
// The canvas is the Vita's front TOUCH PANEL - the front end of at least two of these titles
// is driven by tapping it. A button drawn on top of the canvas would deliver a phantom touch
// to the guest every time it was pressed, so the pad is laid out around the canvas instead,
// and its elements swallow their own events.

/// The `KeyboardEvent.code` each on-screen button reports, so the worker's ONE key map
/// (`input::key_button`) stays the single place a button becomes an SceCtrl bit. A second
/// mapping here would be a second thing to get wrong.
const CODES = {
  up: "ArrowUp",
  down: "ArrowDown",
  left: "ArrowLeft",
  right: "ArrowRight",
  cross: "KeyJ",
  circle: "KeyK",
  square: "KeyI",
  triangle: "KeyL",
  l: "KeyQ",
  r: "KeyE",
  start: "Enter",
  select: "ShiftRight",
};

/// Below this fraction of the stick's radius the stick reads as centred. A thumb resting on
/// glass is never exactly at the origin, and without a deadzone every title would see a
/// permanent small steering input.
const STICK_DEADZONE = 0.14;
/// The same for a real gamepad axis, which idles nearer zero but not at it.
const GAMEPAD_DEADZONE = 0.12;

/// Encode a -1..1 axis pair into the guest's 0..255 stick bytes (128 centred).
function encodeAxis(nx, ny) {
  const to = (v) => Math.max(0, Math.min(255, Math.round(128 + v * 127)));
  return { x: to(nx), y: to(ny) };
}

/// Wire the on-screen pad in `root` to `worker`. Returns nothing; the listeners live for the
/// page's lifetime, which is the run's lifetime.
export function installPad(worker, root) {
  const send = (type, extra) => worker.postMessage({ type, ...extra });
  const key = (name, pressed) => {
    const code = CODES[name];
    if (code) send("key", { code, pressed });
  };

  // --- buttons: every element carrying data-btn ---
  for (const el of root.querySelectorAll("[data-btn]")) {
    const name = el.dataset.btn;
    const press = (e) => {
      e.preventDefault();
      e.stopPropagation();
      el.classList.add("held");
      key(name, true);
      // Capture the pointer so a finger that slides off the button still releases it. Without
      // this, dragging off a button leaves it stuck DOWN, and a stuck accelerate reads exactly
      // like an emulator that has hung.
      if (el.setPointerCapture && e.pointerId !== undefined) {
        try {
          el.setPointerCapture(e.pointerId);
        } catch {
          /* a mouse without capture support: the up handler below still fires */
        }
      }
    };
    const release = (e) => {
      e.preventDefault();
      e.stopPropagation();
      el.classList.remove("held");
      key(name, false);
    };
    el.addEventListener("pointerdown", press);
    el.addEventListener("pointerup", release);
    el.addEventListener("pointercancel", release);
    // A button must never also scroll the page or raise the browser's own context menu.
    el.addEventListener("contextmenu", (e) => e.preventDefault());
  }

  // --- thumb stick: the element carrying data-stick ("0" left, "1" right) ---
  for (const pad of root.querySelectorAll("[data-stick]")) {
    const which = Number(pad.dataset.stick);
    const knob = pad.querySelector(".knob");
    let active = null; // the pointerId currently driving this stick

    const move = (e) => {
      const r = pad.getBoundingClientRect();
      const radius = Math.min(r.width, r.height) / 2;
      let nx = (e.clientX - (r.left + r.width / 2)) / radius;
      let ny = (e.clientY - (r.top + r.height / 2)) / radius;
      // Clamp to the circle, not the square: a stick that reads 1.41 diagonally would drive
      // a title past its own full-deflection value.
      const mag = Math.hypot(nx, ny);
      if (mag > 1) {
        nx /= mag;
        ny /= mag;
      }
      const dead = mag < STICK_DEADZONE;
      if (dead) {
        nx = 0;
        ny = 0;
      }
      if (knob) knob.style.transform = `translate(${nx * radius * 0.7}px, ${ny * radius * 0.7}px)`;
      const { x, y } = encodeAxis(nx, ny);
      send("stick", { stick: which, x, y, active: true });
    };

    pad.addEventListener("pointerdown", (e) => {
      e.preventDefault();
      e.stopPropagation();
      active = e.pointerId;
      try {
        pad.setPointerCapture(e.pointerId);
      } catch {
        /* see the button note above */
      }
      move(e);
    });
    pad.addEventListener("pointermove", (e) => {
      if (active !== e.pointerId) return;
      e.preventDefault();
      move(e);
    });
    const lift = (e) => {
      if (active !== e.pointerId) return;
      e.preventDefault();
      active = null;
      if (knob) knob.style.transform = "";
      // RELEASE, not centre. A recipe that is steering keeps steering.
      send("stick", { stick: which, x: 128, y: 128, active: false });
    };
    pad.addEventListener("pointerup", lift);
    pad.addEventListener("pointercancel", lift);
    pad.addEventListener("contextmenu", (e) => e.preventDefault());
  }
}

/// The standard-gamepad button index -> our button name. Only the ones the Vita has.
const GAMEPAD_BUTTONS = {
  0: "cross",
  1: "circle",
  2: "square",
  3: "triangle",
  4: "l",
  5: "r",
  8: "select",
  9: "start",
  12: "up",
  13: "down",
  14: "left",
  15: "right",
};

/// Poll a connected gamepad once per animation frame and forward CHANGES to the worker.
///
/// Only changes: the Gamepad API has no events for button state, so the alternative is
/// re-sending the whole pad sixty times a second across the worker boundary, and that boundary
/// is the expensive one in this system.
///
/// `onStatus` is called with a short description when a pad appears or disappears, so the page
/// can say a controller is connected instead of leaving the user to guess.
export function installGamepad(worker, onStatus = () => {}) {
  if (!navigator.getGamepads) {
    onStatus("no Gamepad API in this browser");
    return;
  }
  const held = new Set();
  let lastStick = [null, null];
  let seen = null;

  const axisOf = (gp, ix, iy) => {
    let nx = gp.axes[ix] ?? 0;
    let ny = gp.axes[iy] ?? 0;
    if (Math.hypot(nx, ny) < GAMEPAD_DEADZONE) {
      nx = 0;
      ny = 0;
    }
    return encodeAxis(nx, ny);
  };

  const tick = () => {
    const pads = navigator.getGamepads();
    const gp = Array.from(pads).find((p) => p && p.connected);
    if (!gp) {
      if (seen !== null) {
        // Let go of everything the pad was holding, or a disconnect mid-corner leaves the
        // guest with a button held forever.
        for (const name of held) worker.postMessage({ type: "key", code: CODES[name], pressed: false });
        held.clear();
        for (const s of [0, 1]) worker.postMessage({ type: "stick", stick: s, x: 128, y: 128, active: false });
        lastStick = [null, null];
        seen = null;
        onStatus("gamepad disconnected");
      }
      requestAnimationFrame(tick);
      return;
    }
    if (seen !== gp.index) {
      seen = gp.index;
      onStatus(`gamepad: ${gp.id}`);
    }
    for (const [ix, name] of Object.entries(GAMEPAD_BUTTONS)) {
      const down = !!(gp.buttons[ix] && gp.buttons[ix].pressed);
      if (down === held.has(name)) continue;
      if (down) held.add(name);
      else held.delete(name);
      worker.postMessage({ type: "key", code: CODES[name], pressed: down });
    }
    for (const [slot, [ix, iy]] of [
      [0, [0, 1]],
      [1, [2, 3]],
    ]) {
      const { x, y } = axisOf(gp, ix, iy);
      const centred = x === 128 && y === 128;
      const prev = lastStick[slot];
      // A centred pad RELEASES the stick rather than holding 128, so an unused gamepad axis
      // does not silently veto a scripted recipe's steering the whole run.
      const now = centred ? null : `${x},${y}`;
      if (now === prev) continue;
      lastStick[slot] = now;
      worker.postMessage({ type: "stick", stick: slot, x, y, active: !centred });
    }
    requestAnimationFrame(tick);
  };
  requestAnimationFrame(tick);
}
