// The on-screen controls: a d-pad, four faces, two sticks, L/R, Start/Select.
//
// Built here rather than in the HTML so the same controls serve both placements -
// OVER the game (landscape on a phone: the game fills the screen and the controls
// sit translucent at the corners) and BESIDE it (portrait: the game on top, the
// controls below). The placement is a class on the player, set from the settings and
// the orientation; the controls do not know which they are in.
//
// Buttons post the same `{type:"key", code}` a keyboard would, resolved through the
// person's keyboard map so a remap applies here too; sticks post the guest's own
// 0..255 encoding with `active:false` on release (see InputState::left_stick for why
// release is not "send 128").

const STICK_DEADZONE = 0.14;

function encodeAxis(nx, ny) {
  const to = (v) => Math.max(0, Math.min(255, Math.round(128 + v * 127)));
  return { x: to(nx), y: to(ny) };
}

const GLYPH = {
  up: "&#9650;", down: "&#9660;", left: "&#9664;", right: "&#9654;",
  triangle: "&#9651;", circle: "&#9711;", cross: "&#10005;", square: "&#9633;",
  l: "L", r: "R", start: "START", select: "SELECT",
};

/// Build the controls into `root`. `keymap` is the settings' `button -> code` map.
/// Returns { setKeymap(map), setOpacity(v), setScale(v), destroy() }.
export function mountTouchPad(root, worker, keymap, opts = {}) {
  root.innerHTML = `
    <div class="tp tp-l"><div class="tp-btn tp-small" data-btn="l">${GLYPH.l}</div></div>
    <div class="tp tp-r"><div class="tp-btn tp-small" data-btn="r">${GLYPH.r}</div></div>
    <div class="tp tp-left">
      <div class="tp-dpad">
        <i></i><div class="tp-btn" data-btn="up">${GLYPH.up}</div><i></i>
        <div class="tp-btn" data-btn="left">${GLYPH.left}</div><i></i><div class="tp-btn" data-btn="right">${GLYPH.right}</div>
        <i></i><div class="tp-btn" data-btn="down">${GLYPH.down}</div><i></i>
      </div>
      <div class="tp-stick" data-stick="0"><div class="tp-knob"></div></div>
    </div>
    <div class="tp tp-right">
      <div class="tp-faces">
        <i></i><div class="tp-btn" data-btn="triangle">${GLYPH.triangle}</div><i></i>
        <div class="tp-btn" data-btn="square">${GLYPH.square}</div><i></i><div class="tp-btn" data-btn="circle">${GLYPH.circle}</div>
        <i></i><div class="tp-btn" data-btn="cross">${GLYPH.cross}</div><i></i>
      </div>
      <div class="tp-stick" data-stick="1"><div class="tp-knob"></div></div>
    </div>
    <div class="tp tp-mid">
      <div class="tp-btn tp-small" data-btn="select">${GLYPH.select}</div>
      <div class="tp-btn tp-small" data-btn="start">${GLYPH.start}</div>
    </div>`;

  let map = { ...keymap };
  const vibrate = () => {
    if (opts.vibrate && navigator.vibrate) {
      try {
        navigator.vibrate(8);
      } catch {}
    }
  };
  const key = (name, pressed) => {
    const code = map[name];
    if (code) worker.postMessage({ type: "key", code, pressed });
  };

  const stop = (e) => {
    e.preventDefault();
    e.stopPropagation();
  };

  for (const el of root.querySelectorAll("[data-btn]")) {
    const name = el.dataset.btn;
    const press = (e) => {
      stop(e);
      el.classList.add("held");
      key(name, true);
      vibrate();
      try {
        el.setPointerCapture(e.pointerId);
      } catch {}
    };
    const release = (e) => {
      stop(e);
      el.classList.remove("held");
      key(name, false);
    };
    el.addEventListener("pointerdown", press);
    el.addEventListener("pointerup", release);
    el.addEventListener("pointercancel", release);
    el.addEventListener("contextmenu", stop);
  }

  for (const pad of root.querySelectorAll("[data-stick]")) {
    const which = Number(pad.dataset.stick);
    const knob = pad.querySelector(".tp-knob");
    let active = null;
    const move = (e) => {
      const r = pad.getBoundingClientRect();
      const radius = Math.min(r.width, r.height) / 2;
      let nx = (e.clientX - (r.left + r.width / 2)) / radius;
      let ny = (e.clientY - (r.top + r.height / 2)) / radius;
      const mag = Math.hypot(nx, ny);
      if (mag > 1) {
        nx /= mag;
        ny /= mag;
      }
      if (mag < STICK_DEADZONE) nx = ny = 0;
      knob.style.transform = `translate(${nx * radius * 0.6}px, ${ny * radius * 0.6}px)`;
      const { x, y } = encodeAxis(nx, ny);
      worker.postMessage({ type: "stick", stick: which, x, y, active: true });
    };
    pad.addEventListener("pointerdown", (e) => {
      stop(e);
      active = e.pointerId;
      try {
        pad.setPointerCapture(e.pointerId);
      } catch {}
      move(e);
    });
    pad.addEventListener("pointermove", (e) => {
      if (active !== e.pointerId) return;
      stop(e);
      move(e);
    });
    const lift = (e) => {
      if (active !== e.pointerId) return;
      stop(e);
      active = null;
      knob.style.transform = "";
      worker.postMessage({ type: "stick", stick: which, x: 128, y: 128, active: false });
    };
    pad.addEventListener("pointerup", lift);
    pad.addEventListener("pointercancel", lift);
    pad.addEventListener("contextmenu", stop);
  }

  return {
    /// Light a control pressed by something else (a keyboard, a gamepad), so the
    /// picture agrees with the game whichever way the button was pushed.
    setHeld: (name, held) => {
      const el = root.querySelector(`[data-btn="${name}"]`);
      if (el) el.classList.toggle("held", held);
    },
    setKeymap: (m) => {
      map = { ...m };
    },
    setOpacity: (v) => root.style.setProperty("--tp-opacity", String(v)),
    setScale: (v) => root.style.setProperty("--tp-scale", String(v)),
    destroy: () => {
      root.innerHTML = "";
    },
  };
}
