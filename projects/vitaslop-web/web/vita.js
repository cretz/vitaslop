// The controller as a picture: a Vita outline with a callout on every control, each
// showing what drives it. In the settings the callouts are live (click a keyboard
// callout, press a key; a gamepad callout is a list); in the in-game menu they are
// read-only, so a person who forgot which key is Circle can look instead of guess.
//
// Positions are percentages of the drawing, so the same layout serves a phone and a
// wide screen.

const LAYOUT = {
  l: { x: 16, y: 6 }, r: { x: 84, y: 6 },
  up: { x: 17, y: 30 }, left: { x: 8, y: 44 }, right: { x: 26, y: 44 }, down: { x: 17, y: 58 },
  triangle: { x: 83, y: 30 }, square: { x: 74, y: 44 }, circle: { x: 92, y: 44 }, cross: { x: 83, y: 58 },
  select: { x: 68, y: 84 }, start: { x: 90, y: 84 },
};

const SVG = `<svg viewBox="0 0 400 190" xmlns="http://www.w3.org/2000/svg" aria-hidden="true">
  <rect x="4" y="12" width="392" height="166" rx="40" fill="#14141f" stroke="#2a2a3a" stroke-width="2"/>
  <rect x="98" y="34" width="204" height="116" rx="4" fill="#0b0b12" stroke="#2a2a3a"/>
  <path d="M40 12 h60 v-6 h-60 z M300 12 h60 v-6 h-60 z" fill="#1b1b29" stroke="#2a2a3a"/>
  <circle cx="52" cy="140" r="14" fill="#1b1b29" stroke="#2a2a3a"/>
  <circle cx="348" cy="140" r="14" fill="#1b1b29" stroke="#2a2a3a"/>
  <g fill="#1b1b29" stroke="#2a2a3a">
    <rect x="46" y="52" width="12" height="36" rx="2"/><rect x="34" y="64" width="36" height="12" rx="2"/>
    <circle cx="348" cy="56" r="6"/><circle cx="332" cy="72" r="6"/><circle cx="364" cy="72" r="6"/><circle cx="348" cy="88" r="6"/>
    <rect x="286" y="160" width="14" height="6" rx="3"/><rect x="340" y="160" width="14" height="6" rx="3"/>
  </g>
</svg>`;

/// `opts`: { buttons: [{name,label}], controls: [names], keyboard, gamepad, mode: "keyboard"|"gamepad",
///          readonly, onKey(name, code), onPad(name, control) }
export function renderVita(root, opts) {
  root.classList.add("vita");
  root.innerHTML = SVG;
  const esc = (s) => String(s ?? "").replace(/[&<>"']/g, (c) => ({ "&": "&amp;", "<": "&lt;", ">": "&gt;", '"': "&quot;", "'": "&#39;" })[c]);
  for (const b of opts.buttons) {
    const pos = LAYOUT[b.name];
    if (!pos) continue;
    const el = document.createElement("div");
    el.className = "vita-callout";
    el.style.left = `${pos.x}%`;
    el.style.top = `${pos.y}%`;
    const kb = opts.keyboard[b.name] || "-";
    const gp = opts.gamepad[b.name] || "-";
    const shown = opts.mode === "gamepad" ? gp : kb;
    if (opts.readonly) {
      el.innerHTML = `<span class="vita-name">${esc(b.label)}</span><span class="vita-val">${esc(kb)}</span><span class="vita-val dim">${esc(gp)}</span>`;
    } else if (opts.mode === "gamepad") {
      el.innerHTML = `<span class="vita-name">${esc(b.label)}</span><select class="vita-sel">${opts.controls.map((c) => `<option ${c === gp ? "selected" : ""}>${esc(c)}</option>`).join("")}</select>`;
      el.querySelector("select").addEventListener("change", (e) => opts.onPad(b.name, e.target.value));
    } else {
      el.innerHTML = `<span class="vita-name">${esc(b.label)}</span><button type="button" class="vita-key">${esc(shown)}</button>`;
      const btn = el.querySelector("button");
      btn.addEventListener("click", () => {
        // A modal: nothing else on the page is reachable until a key is pressed or the
        // capture is cancelled, so a stray click cannot leave a callout listening.
        const shade = document.createElement("div");
        shade.className = "capture";
        shade.innerHTML = `<div class="capture-in"><p class="dim">Press the key for</p><h2>${esc(b.label)}</h2><p class="dim">Esc cancels. Currently <code>${esc(kb)}</code>.</p><button type="button" class="btn small">Cancel</button></div>`;
        document.body.appendChild(shade);
        btn.classList.add("listening");
        const done = (code) => {
          document.removeEventListener("keydown", onKey, true);
          shade.remove();
          btn.classList.remove("listening");
          if (code) {
            btn.textContent = code;
            opts.onKey(b.name, code);
          }
        };
        const onKey = (e) => {
          e.preventDefault();
          e.stopPropagation();
          done(e.code === "Escape" ? null : e.code);
        };
        document.addEventListener("keydown", onKey, true);
        shade.querySelector("button").addEventListener("click", () => done(null));
        shade.addEventListener("click", (e) => e.target === shade && done(null));
      });
    }
    root.appendChild(el);
  }
}
