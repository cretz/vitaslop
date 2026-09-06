// A real controller through the Gamepad API, mapped by the settings.
//
// The map is `button -> Standard Gamepad control name` (`south`, `dpad_up`...),
// resolved to an index through the vocabulary the Rust side publishes, so the order
// there is the one truth. A press posts the button's KEYBOARD code, which the worker
// resolves through the keyboard map - one lookup table in the worker, not two.
// Sticks are axes 0/1 and 2/3 of the standard layout.

export function installGamepad(worker, vocab, settings, onStatus = () => {}, onButton = () => {}) {
  if (!navigator.getGamepads) {
    onStatus("no Gamepad API in this browser");
    return () => {};
  }
  const controls = vocab.gamepadControls;
  let gpmap = settings.gamepad;
  let keymap = settings.keyboard;
  let deadzone = settings.stickDeadzone ?? 0.12;
  const held = new Set();
  let lastStick = [null, null];
  let seen = null;
  let stopped = false;

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
        for (const name of held) post(name, false);
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
    for (const [name, control] of Object.entries(gpmap)) {
      const ix = controls.indexOf(control);
      const down = ix >= 0 && !!(gp.buttons[ix] && gp.buttons[ix].pressed);
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
  };
}
