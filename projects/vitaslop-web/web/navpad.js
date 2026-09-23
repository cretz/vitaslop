// A gamepad drives the SITE: the library, a title page, the settings, the import
// screen and, when it is open, the in-game menu.
//
// One focus-based navigator: the d-pad or the left stick moves the document's focus
// among the focusable elements of one ROOT (spatially, by the elements' boxes, with
// DOM order as the fallback when nothing lies in that direction), south activates
// what is focused, east goes back, start opens the settings, and inside a select,
// checkbox or slider left/right change the value instead of moving.
//
// Ownership of the pad is exclusive. In a game the pad belongs to gamepad.js, which
// posts to the worker, and this module is INACTIVE - no root, no polling - so a press
// is never handled twice. Opening the in-game menu hands the pad here (the player
// suspends gamepad.js first); closing it hands the pad back. A press that is already
// down at a hand-over is ignored until it is released, on both sides: the press that
// opened the menu does not also pick "Resume", and the press that closed it does not
// reach the game as a button.
//
// The Gamepad API is polled, not evented (there are no button events), and a pad is
// only visible to the page after its first button press; the poll runs only while a
// root is set.

const FOCUSABLE =
  'a[href], button:not([disabled]), input:not([disabled]):not([type="hidden"]), select:not([disabled]), textarea:not([disabled]), summary, [tabindex]:not([tabindex="-1"])';
const CLASS = "navfocus";
/// Standard Gamepad indices.
const SOUTH = 0;
const EAST = 1;
const START = 9;
const DPAD = { up: 12, down: 13, left: 14, right: 15 };
const STICK = 0.5;
const REPEAT_FIRST_MS = 380;
const REPEAT_MS = 110;

let root = null;
let hooks = {};
let polling = false;
// Buttons (by index, plus the four stick directions as "sx-"... strings) down last frame.
let down = new Set();
// Down at the hand-over: inert until released once.
let ignore = new Set();
let repeatDir = null;
let repeatAt = 0;
let focused = null;

const isVisible = (el) => {
  if (el.closest("[hidden]")) return false;
  const r = el.getBoundingClientRect();
  return r.width > 0 && r.height > 0;
};

/// The elements the pad can reach right now. A key-capture shade (the settings'
/// "press the key" modal) is on top of everything, so while it is up it is the root.
const scope = () => document.querySelector(".capture") || root;
const focusables = () => {
  const s = scope();
  if (!s) return [];
  return [...s.querySelectorAll(FOCUSABLE)].filter(isVisible);
};

const setFocus = (el) => {
  if (focused && focused !== el) focused.classList.remove(CLASS);
  focused = el;
  if (!el) return;
  el.classList.add(CLASS);
  try {
    el.focus({ preventScroll: true });
  } catch {}
  try {
    el.scrollIntoView({ block: "nearest", inline: "nearest" });
  } catch {}
};

/// The element focus is on if it is inside the scope, else nothing.
const current = () => {
  const s = scope();
  const a = document.activeElement;
  if (s && a && a !== document.body && s.contains(a) && isVisible(a)) return a;
  return null;
};

const centre = (r) => ({ x: r.left + r.width / 2, y: r.top + r.height / 2 });
/// Boxes whose tops are within this of each other are one row.
const ROW_TOL = 24;
/// Boxes may overlap by this much and still count as beyond.
const TOL = 4;

/// A sticky element (the settings' Save bar) sits at the viewport's edge whatever the
/// scroll, so by box it is "below" half the page; it is reached only when nothing
/// else lies that way.
const isSticky = (el) => {
  for (let e = el; e && e !== document.body; e = e.parentElement) {
    if (getComputedStyle(e).position === "sticky") return true;
  }
  return false;
};

/// The element in `dir` from `from`, by box. Up and down go to the NEAREST ROW that
/// way (the boxes wholly beyond the current one, grouped by top within ROW_TOL) and,
/// in it, the one nearest across; left and right stay in the current row (boxes that
/// overlap it vertically) and take the nearest. Null when nothing lies that way.
function spatial(from, dir, list) {
  const fr = from.getBoundingClientRect();
  const fc = centre(fr);
  const vertical = dir === "up" || dir === "down";
  const cands = [];
  for (const el of list) {
    if (el === from) continue;
    const r = el.getBoundingClientRect();
    const c = centre(r);
    let along;
    let across;
    if (vertical) {
      along = dir === "down" ? r.top - fr.bottom : fr.top - r.bottom;
      if (along < -TOL) continue;
      across = Math.abs(c.x - fc.x);
    } else {
      along = dir === "right" ? r.left - fr.right : fr.left - r.right;
      if (along < -TOL) continue;
      if (Math.min(r.bottom, fr.bottom) - Math.max(r.top, fr.top) <= 0) continue;
      across = Math.abs(c.y - fc.y);
    }
    cands.push({ el, along, across, sticky: isSticky(el) });
  }
  if (!cands.length) return null;
  const pool = cands.some((c) => !c.sticky) ? cands.filter((c) => !c.sticky) : cands;
  if (!vertical) return pool.sort((a, b) => a.along - b.along || a.across - b.across)[0].el;
  const nearest = Math.min(...pool.map((c) => c.along));
  return pool.filter((c) => c.along <= nearest + ROW_TOL).sort((a, b) => a.across - b.across || a.along - b.along)[0].el;
}

function move(dir) {
  const list = focusables();
  if (!list.length) return;
  const cur = current();
  if (!cur) {
    setFocus(list[0]);
    return;
  }
  // Inside a value control, left/right is the value.
  if (dir === "left" || dir === "right") {
    if (adjust(cur, dir === "right" ? 1 : -1)) return;
  }
  let next = spatial(cur, dir, list);
  if (!next) {
    const i = list.indexOf(cur);
    if (dir === "down" || dir === "right") next = list[i + 1] || null;
    else next = i > 0 ? list[i - 1] : null;
  }
  if (next) setFocus(next);
}

/// Change a select, checkbox or slider by `step`; true if `el` was one of those.
function adjust(el, step) {
  const tag = el.tagName;
  if (tag === "SELECT") {
    const n = el.options.length;
    if (!n) return true;
    el.selectedIndex = (el.selectedIndex + step + n) % n;
    el.dispatchEvent(new Event("change", { bubbles: true }));
    return true;
  }
  if (tag === "INPUT" && el.type === "checkbox") {
    el.click();
    return true;
  }
  if (tag === "INPUT" && (el.type === "range" || el.type === "number")) {
    const st = Number(el.step) || 1;
    const v = (Number(el.value) || 0) + step * st;
    const min = el.min === "" ? -Infinity : Number(el.min);
    const max = el.max === "" ? Infinity : Number(el.max);
    el.value = String(Math.min(max, Math.max(min, Math.round(v / st) * st)));
    el.dispatchEvent(new Event("input", { bubbles: true }));
    el.dispatchEvent(new Event("change", { bubbles: true }));
    return true;
  }
  return false;
}

function activate() {
  const cur = current();
  if (!cur) {
    move("down");
    return;
  }
  if (cur.tagName === "SELECT") {
    adjust(cur, 1);
    return;
  }
  cur.click();
}

function back() {
  const shade = document.querySelector(".capture");
  if (shade) {
    const b = shade.querySelector("button");
    if (b) b.click();
    return;
  }
  if (hooks.onBack) hooks.onBack();
}

/// The buttons and stick directions that read as pressed on `gp`, by key.
function pressed(gp) {
  const out = new Set();
  for (const ix of [SOUTH, EAST, START, DPAD.up, DPAD.down, DPAD.left, DPAD.right]) {
    if (gp.buttons[ix] && gp.buttons[ix].pressed) out.add(ix);
  }
  const x = gp.axes[0] ?? 0;
  const y = gp.axes[1] ?? 0;
  if (y < -STICK) out.add("s-up");
  if (y > STICK) out.add("s-down");
  if (x < -STICK) out.add("s-left");
  if (x > STICK) out.add("s-right");
  return out;
}

const DIR_OF = { [DPAD.up]: "up", [DPAD.down]: "down", [DPAD.left]: "left", [DPAD.right]: "right", "s-up": "up", "s-down": "down", "s-left": "left", "s-right": "right" };

function tick() {
  if (!root) {
    polling = false;
    return;
  }
  requestAnimationFrame(tick);
  const gp = Array.from(navigator.getGamepads()).find((p) => p && p.connected);
  if (!gp) {
    down = new Set();
    ignore = new Set();
    repeatDir = null;
    return;
  }
  const now = pressed(gp);
  for (const k of ignore) if (!now.has(k)) ignore.delete(k);
  const t = performance.now();
  let dirHeld = null;
  for (const k of now) {
    if (ignore.has(k)) continue;
    const dir = DIR_OF[k];
    if (dir) {
      dirHeld = dir;
      if (!down.has(k)) {
        move(dir);
        repeatDir = dir;
        repeatAt = t + REPEAT_FIRST_MS;
      }
      continue;
    }
    if (down.has(k)) continue;
    if (k === SOUTH) activate();
    else if (k === EAST) back();
    else if (k === START && hooks.onStart) hooks.onStart();
  }
  if (dirHeld && dirHeld === repeatDir && t >= repeatAt) {
    move(dirHeld);
    repeatAt = t + REPEAT_MS;
  }
  if (!dirHeld) repeatDir = null;
  down = now;
}

/// Take the pad for `el` (the document for a page, the menu element in a game).
/// `onBack` is east; `onStart` is start. Presses already down are inert until released.
export function attach(el, { onBack, onStart } = {}) {
  if (!navigator.getGamepads) return;
  root = el;
  hooks = { onBack, onStart };
  ignore = new Set();
  const gp = Array.from(navigator.getGamepads()).find((p) => p && p.connected);
  if (gp) ignore = pressed(gp);
  down = new Set();
  repeatDir = null;
  if (focused && !(root && root.contains(focused))) setFocus(null);
  if (!polling) {
    polling = true;
    requestAnimationFrame(tick);
  }
}

/// Give the pad up: no root, no polling, no ring.
export function detach() {
  root = null;
  hooks = {};
  setFocus(null);
}

export const isAttached = () => root !== null;
