// A real controller through the Gamepad API, mapped by the settings.
//
// The map is `button -> Standard Gamepad control name` (`south`, `dpad_up`...),
// resolved to an index through the vocabulary the Rust side publishes, so the order
// there is the one truth. A press posts the button's KEYBOARD code, which the worker
// resolves through the keyboard map - one lookup table in the worker, not two.
// Sticks are axes 0/1 and 2/3 of the standard layout.
//
// The in-game menu is opened from the pad two ways: the `home` control (index 16 -
// exposed on some pads and browsers, not all) when no Vita button is mapped to it,
// or start+select held together for MENU_HOLD_MS, which every pad has. `onMenu` is
// called once for either; the chord's start and select are released to the game
// first, so the game sees a short press of each and not a stuck pair. While the
// menu is open the pad is SUSPENDED here (everything held is released, nothing is
// posted) and navpad.js owns it; on resume, whatever is still down from the menu -
// the press that closed it - is ignored until it is released.

const HOME = 16;
const MENU_HOLD_MS = 1000;

export function installGamepad(worker, vocab, settings, onStatus = () => {}, onButton = () => {}, onMenu = () => {}) {
  if (!navigator.getGamepads) {
    onStatus("no Gamepad API in this browser");
    return { stop: () => {}, update: () => {}, suspend: () => {} };
  }
  const controls = vocab.gamepadControls;
  let gpmap = settings.gamepad;
  let keymap = settings.keyboard;
  let deadzone = settings.stickDeadzone ?? 0.12;
  const held = new Set();
  let lastStick = [null, null];
  let seen = null;
  let stopped = false;
  let suspended = false;
  // Control indices down at a resume: inert until released once.
  let ignore = new Set();
  let homeDown = false;
  let chordSince = 0;
  const pressedIx = (gp, ix) => ix >= 0 && !!(gp.buttons[ix] && gp.buttons[ix].pressed);
  const releaseAll = () => {
    for (const name of held) post(name, false);
    held.clear();
    for (const s of [0, 1]) worker.postMessage({ type: "stick", stick: s, x: 128, y: 128, active: false });
    lastStick = [null, null];
  };

  const encode = (nx, ny) => {
    const to = (v) => Math.max(0, Math.min(255, Math.round(128 + v * 127)));
    return { x: to(nx), y: to(ny) };
  };
  const post = (name, pressed) => {
    const code = keymap[name];
    if (code) worker.postMessage({ type: "key", code, pressed });
    onButton(name, pressed);
  };

  const tick = () => {
    if (stopped) return;
    const gp = Array.from(navigator.getGamepads()).find((p) => p && p.connected);
    if (!gp) {
      if (seen !== null) {
        releaseAll();
        seen = null;
        ignore = new Set();
        homeDown = false;
        chordSince = 0;
        onStatus("gamepad disconnected");
      }
      requestAnimationFrame(tick);
      return;
    }
    if (seen !== gp.index) {
      seen = gp.index;
      onStatus(`gamepad: ${gp.id}`);
    }
    if (suspended) {
      requestAnimationFrame(tick);
      return;
    }
    for (const ix of ignore) if (!pressedIx(gp, ix)) ignore.delete(ix);
    // Home opens the menu unless a Vita button is mapped to it (then it is the game's).
    const home = pressedIx(gp, HOME) && !ignore.has(HOME);
    const homeMapped = Object.values(gpmap).includes("home");
    if (home && !homeDown && !homeMapped) {
      homeDown = true;
      ignore.add(HOME);
      onMenu();
      requestAnimationFrame(tick);
      return;
    }
    if (!home) homeDown = false;
    // Start+select held together for a second: the universal way in.
    const startIx = controls.indexOf("start");
    const selectIx = controls.indexOf("select");
    if (pressedIx(gp, startIx) && pressedIx(gp, selectIx) && !ignore.has(startIx) && !ignore.has(selectIx)) {
      if (!chordSince) chordSince = performance.now();
      else if (performance.now() - chordSince >= MENU_HOLD_MS) {
        chordSince = 0;
        for (const [name, control] of Object.entries(gpmap)) {
          if ((control === "start" || control === "select") && held.has(name)) {
            held.delete(name);
            post(name, false);
          }
        }
        ignore.add(startIx);
        ignore.add(selectIx);
        onMenu();
        requestAnimationFrame(tick);
        return;
      }
    } else chordSince = 0;
    for (const [name, control] of Object.entries(gpmap)) {
      const ix = controls.indexOf(control);
      if (ignore.has(ix)) continue;
      const down = pressedIx(gp, ix);
      if (down === held.has(name)) continue;
      if (down) held.add(name);
      else held.delete(name);
      post(name, down);
    }
    for (const [slot, [ix, iy]] of [[0, [0, 1]], [1, [2, 3]]]) {
      let nx = gp.axes[ix] ?? 0;
      let ny = gp.axes[iy] ?? 0;
      if (Math.hypot(nx, ny) < deadzone) nx = ny = 0;
      const { x, y } = encode(nx, ny);
      const centred = x === 128 && y === 128;
      const now = centred ? null : `${x},${y}`;
      if (now === lastStick[slot]) continue;
      lastStick[slot] = now;
      worker.postMessage({ type: "stick", stick: slot, x, y, active: !centred });
    }
    requestAnimationFrame(tick);
  };
  requestAnimationFrame(tick);

  return {
    stop: () => {
      stopped = true;
    },
    update: (s) => {
      gpmap = s.gamepad;
      keymap = s.keyboard;
      deadzone = s.stickDeadzone ?? deadzone;
    },
    /// Hand the pad away (true) or take it back (false). Taking it back ignores every
    /// control still down until it is released - see the note at the top.
    suspend: (s) => {
      if (s === suspended) return;
      suspended = s;
      chordSince = 0;
      if (s) {
        releaseAll();
        return;
      }
      ignore = new Set();
      const gp = Array.from(navigator.getGamepads()).find((p) => p && p.connected);
      if (!gp) return;
      for (let ix = 0; ix < gp.buttons.length; ix++) if (pressedIx(gp, ix)) ignore.add(ix);
    },
  };
}
