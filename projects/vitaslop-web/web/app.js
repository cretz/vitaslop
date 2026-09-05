// The app: a hash router over five screens - library, title, settings, import,
// about - and the player. Plain DOM, no framework; each screen is a function that
// renders into #view and wires its own handlers.

import { ensureIsolation, checkFeatures, canPlay } from "./features.js";
import * as store from "./store.js";
import * as imp from "./import.js";
import * as gamedata from "./gamedata.js";
import { removeTitle } from "./opfs.js";
import { writeZip, readZip } from "./zipstore.js";
import { createPlayer } from "./player.js";
import { renderVita } from "./vita.js";

const $ = (id) => document.getElementById(id);
const view = $("view");
const esc = (s) => String(s ?? "").replace(/[&<>"']/g, (c) => ({ "&": "&amp;", "<": "&lt;", ">": "&gt;", '"': "&quot;", "'": "&#39;" })[c]);
const go = (hash) => (location.hash = hash);
const fmtDate = (ms) => (ms ? new Date(ms).toLocaleString() : "never");

let features = [];
let playable = false;
const player = createPlayer({ onExit: () => go(current.titleId ? `#/title/${current.titleId}` : "#/") });
let current = { screen: "library", titleId: null };

/// Breadcrumbs: where you are and the way back. Every screen under a title carries them.
function crumbs(parts) {
  return `<nav class="crumbs">${parts
    .map((p, i) => (i === parts.length - 1 || !p.href ? `<span>${esc(p.text)}</span>` : `<a href="${esc(p.href)}">${esc(p.text)}</a>`))
    .join('<span class="sep">/</span>')}</nav>`;
}

/// The leaves of `mine` that differ from `base`: nested objects recurse, so a title that
/// remaps one key stores one key, and every other global change still reaches it.
function deepDiff(base, mine) {
  const out = {};
  for (const k of new Set([...Object.keys(base || {}), ...Object.keys(mine || {})])) {
    const b = base ? base[k] : undefined;
    const m = mine ? mine[k] : undefined;
    const isObj = (v) => v && typeof v === "object" && !Array.isArray(v);
    if (isObj(b) && isObj(m)) {
      const d = deepDiff(b, m);
      if (Object.keys(d).length) out[k] = d;
    } else if (m === undefined) {
      out[k] = null;
    } else if (JSON.stringify(b) !== JSON.stringify(m)) out[k] = m;
  }
  return out;
}
const leafPaths = (o, prefix = "") => Object.entries(o || {}).flatMap(([k, v]) => (v && typeof v === "object" ? leafPaths(v, prefix + k + ".") : [prefix + k]));

// ------------------------------- router -------------------------------

function route() {
  const h = location.hash || "#/";
  const m = h.match(/^#\/(title|settings|play|import|about)?\/?([A-Za-z0-9_-]*)/);
  const screen = (m && m[1]) || "library";
  const id = (m && m[2]) || null;
  if (player.isRunning() && screen !== "play") player.stop();
  current = { screen, titleId: id };
  document.body.dataset.screen = screen;
  document.title = { library: "Library", settings: "Settings", import: "Add games", about: "About" }[screen] ? `${{ library: "Library", settings: "Settings", import: "Add games", about: "About" }[screen]} - vitaslop` : "vitaslop";
  const fn = { library: renderLibrary, title: renderTitle, settings: renderSettings, import: renderImport, about: renderAbout, play: renderPlay }[screen];
  fn(id).catch((e) => {
    view.innerHTML = `<div class="card error"><h2>Something went wrong</h2><pre>${esc(e && e.stack ? e.stack : e)}</pre></div>`;
  });
}
window.addEventListener("hashchange", route);

// ------------------------------- library -------------------------------

let sortMode = "recent";
let query = "";

async function renderLibrary() {
  const titles = await store.listTitles();
  view.innerHTML = `
    <div class="toolbar">
      <input id="search" type="search" placeholder="Search ${titles.length} title${titles.length === 1 ? "" : "s"}" value="${esc(query)}" autocomplete="off" />
      <select id="sort" title="Sort">
        <option value="recent">Recently added</option>
        <option value="played">Recently played</option>
        <option value="name">Name</option>
        <option value="id">Title id</option>
      </select>
      <a class="btn primary" href="#/import">Add games</a>
    </div>
    ${playable ? "" : browserBlock()}
    <div id="grid" class="grid"></div>
    ${titles.length ? "" : `<div class="empty"><p>No titles yet.</p><p><a class="btn primary" href="#/import">Add a game</a> from a .pkg and work.bin, a dumped folder, or a zip of either.</p></div>`}`;
  $("sort").value = sortMode;
  const grid = $("grid");
  const draw = () => {
    const q = query.trim().toLowerCase();
    let list = titles.filter((t) => !q || `${t.title} ${t.titleId} ${t.contentId}`.toLowerCase().includes(q));
    const by = {
      recent: (a, b) => (b.importedAt || 0) - (a.importedAt || 0),
      played: (a, b) => (b.lastPlayedAt || 0) - (a.lastPlayedAt || 0) || (b.importedAt || 0) - (a.importedAt || 0),
      name: (a, b) => a.title.localeCompare(b.title),
      id: (a, b) => a.titleId.localeCompare(b.titleId),
    }[sortMode];
    list.sort(by);
    grid.innerHTML = list
      .map(
        (t) => `<a class="tile" href="#/title/${esc(t.titleId)}" data-id="${esc(t.titleId)}" title="${esc(t.title)}">
          <span class="icon"><img alt="" data-icon="${esc(t.titleId)}" /></span>
          <span class="name">${esc(t.title)}</span>
          <span class="id">${esc(t.titleId)}</span></a>`
      )
      .join("");
    lazyIcons(grid);
  };
  draw();
  $("search").addEventListener("input", (e) => {
    query = e.target.value;
    draw();
  });
  $("sort").addEventListener("change", (e) => {
    sortMode = e.target.value;
    draw();
  });
}

/// Icons load as tiles scroll into view: a thousand titles is a thousand files, and
/// the library must open instantly regardless.
function lazyIcons(root) {
  const io = new IntersectionObserver(
    (entries) => {
      for (const e of entries) {
        if (!e.isIntersecting) continue;
        const img = e.target;
        io.unobserve(img);
        store.titleImage(img.dataset.icon).then((url) => {
          if (url) img.src = url;
          else img.closest(".icon").classList.add("noicon");
        });
      }
    },
    { rootMargin: "300px" }
  );
  for (const img of root.querySelectorAll("img[data-icon]")) io.observe(img);
}

function browserBlock() {
  const bad = features.filter((f) => f.level === "fatal");
  return `<div class="card error"><h2>This browser cannot run titles here</h2>
    <ul class="checks">${bad.map((f) => `<li><b>${esc(f.text)}</b><br>${esc(f.fix)}</li>`).join("")}</ul>
    <p><a href="#/about">All checks</a></p></div>`;
}

// ------------------------------- title -------------------------------

async function renderTitle(id) {
  const meta = await store.readTitle(id);
  if (!meta) {
    view.innerHTML = `<div class="card"><h2>${esc(id)}</h2><p>This title is not in the library.</p><p><a class="btn" href="#/import">Add it</a></p></div>`;
    return;
  }
  const eff = await store.effective(id);
  gamedata.setProfile(eff.profile);
  document.title = `${meta.title} - vitaslop`;
  const [icon, pic, bytes] = await Promise.all([store.titleImage(id), store.titleImage(id, "pic0.png"), store.titleBytes(id)]);
  view.innerHTML = `
    ${crumbs([{ text: "Library", href: "#/" }, { text: meta.title }])}
    <div class="hero">
      <div class="hero-in">
        <span class="icon big ${icon ? "" : "noicon"}">${icon ? `<img src="${icon}" alt="" />` : ""}</span>
        <div class="hero-text">
          <h1>${esc(meta.title)}</h1>
          <p class="meta">${esc(meta.titleId)}${meta.appVersion ? " &middot; v" + esc(meta.appVersion) : ""} &middot; ${store.fmtBytes(bytes)} &middot; added ${fmtDate(meta.importedAt)} &middot; played ${fmtDate(meta.lastPlayedAt)}</p>
          <div class="actions">
            <button id="play" class="btn primary big" ${playable ? "" : "disabled"}>Play</button>
            <a class="btn" href="#/settings/${esc(id)}">Settings for this title</a>
            <button id="remove" class="btn danger">Remove</button>
          </div>
        </div>
      </div>
    </div>
    ${playable ? "" : browserBlock()}
    <div class="card">
      <h2>Saved data <span class="dim">profile: ${esc(eff.profile)}</span></h2>
      <p id="gd-info" class="dim">checking...</p>
      <div class="actions">
        <button id="gd-dl" class="btn">Download</button>
        <button id="gd-up" class="btn">Upload</button>
        <button id="gd-rm" class="btn danger">Clear</button>
        <input id="gd-file" type="file" accept=".zip,application/zip" hidden />
      </div>
      <p class="dim">What the game saved - its save files and trophies - and nothing of the game itself. A download is a file you own; upload it on another device to continue there.</p>
    </div>`;
  $("play").addEventListener("click", () => play(id, matchMedia("(pointer: coarse)").matches));
  // The backdrop goes on through the element's style property: a `style="..."` attribute
  // in the markup is an inline style the page's Content-Security-Policy refuses.
  if (pic) view.querySelector(".hero").style.backgroundImage = `url('${pic}')`;
  $("remove").addEventListener("click", async () => {
    if (!confirm(`Remove ${meta.title} (${meta.titleId}) from this browser?\n\nThis deletes the imported game (${store.fmtBytes(bytes)}). Saved data is kept; clear it separately if you want it gone.`)) return;
    await removeTitle(id);
    await store.removeTitleRecord(id);
    go("#/");
  });
  const info = $("gd-info");
  const refresh = async () => {
    const i = await gamedata.info(id);
    $("gd-dl").disabled = $("gd-rm").disabled = !i;
    info.textContent = i ? `${store.fmtBytes(i.bytes)}, last written ${fmtDate(i.modified)}` : "nothing saved yet - it appears here the first time the game saves.";
  };
  refresh();
  $("gd-dl").addEventListener("click", async () => {
    const b = await gamedata.read(id);
    if (b) downloadBytes(`vitaslop-${id}-${eff.profile}-gamedata.zip`, b);
  });
  $("gd-up").addEventListener("click", () => $("gd-file").click());
  $("gd-file").addEventListener("change", async (e) => {
    const f = e.target.files && e.target.files[0];
    e.target.value = "";
    if (!f) return;
    try {
      const bytes = new Uint8Array(await f.arrayBuffer());
      const summary = await describeGameData(bytes.slice().buffer);
      const existing = await gamedata.info(id);
      if (!confirm(`Restore this save into ${meta.title}?\n\n${summary}${existing ? `\n\nThis REPLACES the ${store.fmtBytes(existing.bytes)} already saved.` : ""}`)) return;
      await gamedata.write(id, bytes);
    } catch (err) {
      alert(`Not restored: ${err.message || err}`);
    }
    refresh();
  });
  $("gd-rm").addEventListener("click", async () => {
    const i = await gamedata.info(id);
    if (!i || !confirm(`Delete the saved data for ${meta.title}? This cannot be undone.`)) return;
    await gamedata.clear(id);
    refresh();
  });
}

const describeGameData = (zip) =>
  new Promise((resolve, reject) => {
    const w = new Worker("./gamedata-worker.js", { type: "module" });
    w.onmessage = (e) => {
      w.terminate();
      e.data.ok ? resolve(e.data.summary) : reject(new Error(e.data.error));
    };
    w.onerror = (e) => {
      w.terminate();
      reject(new Error(e.message || "the game-data worker failed to start"));
    };
    w.postMessage({ zip }, [zip]);
  });

function downloadBytes(name, bytes, type = "application/zip") {
  const url = URL.createObjectURL(new Blob([bytes], { type }));
  const a = document.createElement("a");
  a.href = url;
  a.download = name;
  document.body.appendChild(a);
  a.click();
  a.remove();
  setTimeout(() => URL.revokeObjectURL(url), 10000);
}

// ------------------------------- play -------------------------------

async function play(id, fullscreen) {
  const meta = await store.readTitle(id);
  if (!meta || !playable) return;
  const eff = await store.effective(id);
  current = { screen: "play", titleId: id };
  history.replaceState(null, "", `#/play/${id}`);
  document.body.dataset.screen = "play";
  view.innerHTML = "";
  await player.start(meta, eff, {
    fullscreen,
    onSetting: (patch) => store.saveGlobalSettings(deepMerge(store.globalSettings(), patch)),
  });
}

async function renderPlay(id) {
  if (player.isRunning()) return;
  // Arrived by URL (a reload, a link): there is no gesture, so the browser will not let
  // audio start and fullscreen cannot be asked for. Land on the title page instead,
  // where the Play button is.
  go(id ? `#/title/${id}` : "#/");
}

function deepMerge(a, b) {
  const out = { ...a };
  for (const [k, v] of Object.entries(b)) {
    out[k] = v && typeof v === "object" && !Array.isArray(v) && out[k] && typeof out[k] === "object" ? deepMerge(out[k], v) : v;
  }
  return out;
}

// ------------------------------- settings -------------------------------

async function renderSettings(id) {
  const vocab = await store.vocabulary();
  const global = await store.effective(null);
  const eff = id ? await store.effective(id) : global;
  const meta = id ? await store.readTitle(id) : null;
  const profiles = await gamedata.listProfiles();
  if (!profiles.includes(eff.profile)) profiles.push(eff.profile);
  const knobsText = Object.entries(eff.knobs).map(([k, v]) => `${k}=${v}`).join("\n");
  const opt = (v, cur, label) => `<option value="${esc(v)}" ${v === cur ? "selected" : ""}>${esc(label ?? v)}</option>`;

  if (meta) document.title = `${meta.title} settings - vitaslop`;
  const patch = id ? store.titleSettings(id) : {};
  const overridden = new Set(leafPaths(patch));
  const kb = { ...eff.keyboard };
  const gp = { ...eff.gamepad };
  let ctlMode = "keyboard";
  view.innerHTML = `
    ${meta ? crumbs([{ text: "Library", href: "#/" }, { text: meta.title, href: `#/title/${id}` }, { text: "Settings" }]) : crumbs([{ text: "Library", href: "#/" }, { text: "Settings" }])}
    <h1>${meta ? `Settings for ${esc(meta.title)}` : "Settings"}</h1>
    ${meta ? `<p class="dim">Only what you change here is kept for this title (marked <b class="ov">changed</b>); everything else follows the <a href="#/settings">global settings</a>, including later changes to them.</p>` : ""}
    <form id="sf" class="settings">
      <section class="card"><h2>General</h2>
        <label class="row"><input type="checkbox" name="pauseOnBlur" ${eff.pauseOnBlur ? "checked" : ""} /><span>Pause when this page is hidden or loses focus<small>Not the game's pause menu: the emulator stops, as a console in a pocket would.</small></span></label>
        <label class="row"><input type="checkbox" name="showFps" ${eff.showFps ? "checked" : ""} /><span>Show the frame rate over the game</span></label>
        <label class="row"><span>Scaling</span><select name="scaling">${opt("fit", eff.scaling, "Fit (smooth)")}${opt("integer", eff.scaling, "Integer (crisp)")}${opt("stretch", eff.scaling, "Stretch")}</select></label>
        <label class="row"><span>Save profile<small>Each profile keeps its own saved data for every game.</small></span>
          <span class="inline"><select name="profile">${profiles.map((p) => opt(p, eff.profile)).join("")}</select><button type="button" id="new-profile" class="btn small">New</button></span></label>
      </section>
      <section class="card"><h2>On-screen controls</h2>
        <p class="dim">The buttons and sticks drawn around the game. They also light up when a keyboard or gamepad presses the same button.</p>
        <label class="row"><span>Placement<small>Automatic shows them only on a touch screen: over the game in landscape, below it in portrait. Hidden on a desktop with a mouse; pick a placement to show them anyway.</small></span><select name="pad.mode">${opt("auto", eff.pad.mode, "Automatic")}${opt("overlay", eff.pad.mode, "Over the game")}${opt("beside", eff.pad.mode, "Beside the game")}${opt("hidden", eff.pad.mode, "Hidden")}</select></label>
        <label class="row"><span>Opacity over the game</span><input type="range" name="pad.opacity" min="0.1" max="1" step="0.05" value="${eff.pad.opacity}" /></label>
        <label class="row"><span>Size</span><input type="range" name="pad.scale" min="0.6" max="1.6" step="0.05" value="${eff.pad.scale}" /></label>
        <label class="row"><input type="checkbox" name="pad.vibrate" ${eff.pad.vibrate ? "checked" : ""} /><span>Vibrate on press</span></label>
        <label class="row"><span>Stick dead zone</span><input type="range" name="stickDeadzone" min="0" max="0.5" step="0.01" value="${eff.stickDeadzone}" /></label>
      </section>
      <section class="card" id="controls"><h2>Controls</h2>
        <div class="seg"><button type="button" class="on" data-ctl="keyboard">Keyboard</button><button type="button" data-ctl="gamepad">Gamepad</button></div>
        <p class="dim" id="ctl-hint">Click a control on the picture, then press the key for it.</p>
        <div id="vita"></div>
        <div class="actions"><button type="button" id="ctl-reset" class="btn small">Reset to defaults</button></div>
      </section>
      <details class="card"><summary><h2>Advanced</h2></summary>
        <label class="col"><span>Knobs<small>One VITASLOP_NAME=value per line. The browser has no environment; this is the only way to reach one.</small></span><textarea name="knobs" rows="4" spellcheck="false">${esc(knobsText)}</textarea></label>
        <label class="col"><span>Recipe<small>A scripted-input recipe, replayed from the first frame; live input still works.</small></span><textarea name="recipe" rows="3" spellcheck="false">${esc(eff.recipe)}</textarea></label>
        <label class="row"><span>Fast-forward to frame<small>Runs unpaced and unpresented to here first.</small></span><input type="number" name="fastForward" min="0" step="100" value="${eff.fastForward}" /></label>
        <label class="row"><input type="checkbox" name="debugCapture" ${eff.debugCapture ? "checked" : ""} /><span>Capture debug timings<small>Times every host call for the diagnostics; roughly doubles the frame cost.</small></span></label>
        <label class="row"><input type="checkbox" name="consoleNotes" ${eff.consoleNotes ? "checked" : ""} /><span>Mirror run notes to the console</span></label>
      </details>
      <div class="actions sticky">
        <button type="submit" class="btn primary">Save</button>
        ${meta ? `<button type="button" id="reset-title" class="btn">Use global settings</button>` : `<button type="button" id="reset-all" class="btn">Reset to defaults</button>`}
        <span id="saved" class="dim"></span>
      </div>
    </form>
    ${meta ? "" : `
    <section class="card"><h2>Saved data</h2>
      <p class="dim">Every game's saved data in the current profile, as one file.</p>
      <div class="actions"><button id="gd-all-dl" class="btn">Download all</button><button id="gd-all-up" class="btn">Upload a bundle</button><button id="gd-all-rm" class="btn danger">Clear all</button><input id="gd-all-file" type="file" accept=".zip" hidden /></div>
      <p id="gd-all-info" class="dim"></p>
    </section>
    <section class="card"><h2>Storage</h2><p id="storage" class="dim">measuring...</p></section>
    <section class="card"><h2>This browser</h2>${featureList()}</section>`}`;

  const form = $("sf");
  const read = () => {
    const fd = new FormData(form);
    return {
      pauseOnBlur: form.pauseOnBlur.checked,
      showFps: form.showFps.checked,
      fpsInTitle: eff.fpsInTitle,
      scaling: fd.get("scaling"),
      pad: { mode: fd.get("pad.mode"), opacity: Number(fd.get("pad.opacity")), scale: Number(fd.get("pad.scale")), vibrate: form["pad.vibrate"].checked },
      keyboard: { ...kb },
      gamepad: { ...gp },
      stickDeadzone: Number(fd.get("stickDeadzone")),
      profile: fd.get("profile"),
      knobs: knobsFromText(fd.get("knobs")),
      recipe: fd.get("recipe"),
      fastForward: Number(fd.get("fastForward")) || 0,
      debugCapture: form.debugCapture.checked,
      consoleNotes: form.consoleNotes.checked,
    };
  };
  const knobsFromText = (t) => {
    const out = {};
    for (const line of String(t || "").split(/\r?\n/)) {
      const m = /^\s*([A-Z0-9_]+)\s*=\s*(.*?)\s*$/.exec(line);
      if (m) out[m[1]] = m[2];
    }
    return out;
  };
  form.addEventListener("submit", (e) => {
    e.preventDefault();
    const v = read();
    if (meta) {
      // Only the LEAVES that differ from the global settings are the title's own.
      store.saveTitleSettings(id, deepDiff(global, v));
    } else store.saveGlobalSettings(v);
    $("saved").textContent = "saved";
    setTimeout(() => ($("saved").textContent = ""), 1500);
  });
  if (meta) $("reset-title").addEventListener("click", () => {
    store.saveTitleSettings(id, null);
    renderSettings(id);
  });
  else $("reset-all").addEventListener("click", () => {
    if (!confirm("Reset every global setting to its default?")) return;
    store.saveGlobalSettings({});
    renderSettings(null);
  });
  $("new-profile").addEventListener("click", () => {
    const name = prompt("Profile name (letters, digits, - and _):");
    if (!name || !/^[A-Za-z0-9_-]{1,32}$/.test(name)) return;
    const sel = form.profile;
    if (![...sel.options].some((o) => o.value === name)) sel.add(new Option(name, name));
    sel.value = name;
  });
  const drawVita = () => {
    $("ctl-hint").textContent = ctlMode === "keyboard" ? "Click a control on the picture, then press the key for it." : "Standard layout positions: south is the bottom face button (A or Cross), east the right one (B or Circle).";
    renderVita($("vita"), {
      buttons: vocab.buttons,
      controls: vocab.gamepadControls,
      keyboard: kb,
      gamepad: gp,
      mode: ctlMode,
      onKey: (name, code) => (kb[name] = code),
      onPad: (name, control) => (gp[name] = control),
    });
  };
  drawVita();
  for (const b of view.querySelectorAll("#controls .seg button")) {
    b.addEventListener("click", () => {
      ctlMode = b.dataset.ctl;
      for (const o of view.querySelectorAll("#controls .seg button")) o.classList.toggle("on", o === b);
      drawVita();
    });
  }
  $("ctl-reset").addEventListener("click", async () => {
    const d = await store.defaults();
    Object.assign(kb, d.keyboard);
    Object.assign(gp, d.gamepad);
    drawVita();
  });
  // Mark what this title overrides.
  for (const el of form.querySelectorAll("[name]")) {
    const key = el.name.replace(/^gamepad\./, "gamepad.");
    if ([...overridden].some((o) => o === key || o.startsWith(key + "."))) el.closest("label")?.classList.add("overridden");
  }
  if ([...overridden].some((o) => o.startsWith("keyboard.") || o.startsWith("gamepad."))) $("controls").classList.add("overridden");

  if (!meta) {
    gamedata.setProfile(eff.profile);
    const refreshAll = async () => {
      const ids = await gamedata.listSaved();
      $("gd-all-info").textContent = ids.length ? `${ids.length} game${ids.length === 1 ? "" : "s"} have saved data in profile "${eff.profile}".` : `no saved data in profile "${eff.profile}".`;
      $("gd-all-dl").disabled = $("gd-all-rm").disabled = !ids.length;
    };
    refreshAll();
    $("gd-all-dl").addEventListener("click", async () => {
      const entries = [];
      for (const tid of await gamedata.listSaved()) entries.push({ name: `${tid}.zip`, bytes: await gamedata.read(tid) });
      downloadBytes(`vitaslop-${eff.profile}-all-gamedata.zip`, writeZip(entries));
    });
    $("gd-all-up").addEventListener("click", () => $("gd-all-file").click());
    $("gd-all-file").addEventListener("change", async (e) => {
      const f = e.target.files && e.target.files[0];
      e.target.value = "";
      if (!f) return;
      try {
        const entries = readZip(new Uint8Array(await f.arrayBuffer())).filter((x) => /^[A-Za-z0-9_-]+\.zip$/.test(x.name));
        if (!entries.length) throw new Error("no <TITLE_ID>.zip entries in this bundle");
        if (!confirm(`Restore saved data for ${entries.length} game${entries.length === 1 ? "" : "s"} into profile "${eff.profile}"? Existing saves for the same games are replaced.`)) return;
        for (const x of entries) await gamedata.write(x.name.slice(0, -4), x.bytes);
      } catch (err) {
        alert(`Not restored: ${err.message || err}`);
      }
      refreshAll();
    });
    $("gd-all-rm").addEventListener("click", async () => {
      const ids = await gamedata.listSaved();
      if (!ids.length || !confirm(`Delete the saved data of ${ids.length} game${ids.length === 1 ? "" : "s"} in profile "${eff.profile}"? This cannot be undone.`)) return;
      for (const tid of ids) await gamedata.clear(tid);
      refreshAll();
    });
    try {
      const est = await navigator.storage.estimate();
      $("storage").textContent = `${store.fmtBytes(est.usage)} used of ${store.fmtBytes(est.quota)} this browser will give this site.`;
    } catch {
      $("storage").textContent = "this browser does not report storage use.";
    }
  }
}

function featureList() {
  return `<ul class="checks">${features.map((f) => `<li class="${f.level}"><b>${esc(f.text)}</b>${f.level !== "ok" && f.fix ? `<br>${esc(f.fix)}` : ""}</li>`).join("")}</ul>`;
}

// ------------------------------- import -------------------------------

async function renderImport() {
  view.innerHTML = `
    <h1>Add games</h1>
    <div class="tabs" id="tabs">
      <button class="tab on" data-mode="pkg">Package (.pkg + work.bin)</button>
      <button class="tab" data-mode="folder">Dumped folder</button>
      <button class="tab" data-mode="zip">Zip</button>
      <button class="tab" data-mode="vpk">Homebrew (.vpk)</button>
    </div>
    <div class="card mode" id="mode-pkg">
      <p class="dim">A <code>.pkg</code> from a console and the <code>work.bin</code> licence that was made for it (from NoNpDrm or a dump). Both are needed: the pkg is encrypted and the licence holds its key.</p>
      <label class="pick"><span>1. The package</span><span class="inline"><button id="pick-pkg" class="btn">Choose .pkg</button><span id="pkg-name" class="dim">none chosen</span></span></label>
      <label class="pick"><span>2. Its licence</span><span class="inline"><button id="pick-work" class="btn">Choose work.bin</button><span id="work-name" class="dim">none chosen</span></span></label>
      <input id="f-pkg" type="file" accept=".pkg,application/octet-stream" hidden />
      <input id="f-work" type="file" accept=".bin,application/octet-stream" hidden />
      <div class="actions"><button id="go-pkg" class="btn primary" disabled>Continue</button></div>
    </div>
    <div class="card mode" id="mode-folder" hidden>
      <p class="dim">A folder dumped from a console (with <code>sce_pfs</code> and <code>sce_sys</code> inside, and the <code>work.bin</code> under <code>sce_sys/package</code>).</p>
      <div class="actions"><button id="pick-dir" class="btn primary">Choose a folder</button></div>
      <input id="f-dir" type="file" webkitdirectory hidden />
    </div>
    <div class="card mode" id="mode-zip" hidden>
      <p class="dim">A zip of either of the above.</p>
      <div class="actions"><button id="pick-zip" class="btn primary">Choose a .zip</button></div>
      <input id="f-zip" type="file" accept=".zip,application/zip" hidden />
    </div>
    <div class="card mode" id="mode-vpk" hidden>
      <p class="dim">A homebrew app as its <code>.vpk</code>, as distributed. Nothing is encrypted, so no licence is needed.</p>
      <div class="actions"><button id="pick-vpk" class="btn primary">Choose a .vpk</button></div>
      <input id="f-vpk" type="file" accept=".vpk,application/zip,application/octet-stream" hidden />
    </div>
    <div id="drop" class="drop"><p class="dim">Or drop files or a folder anywhere here.</p></div>
    <div id="result"></div>`;
  for (const tab of view.querySelectorAll("#tabs .tab")) {
    tab.addEventListener("click", () => {
      for (const t of view.querySelectorAll("#tabs .tab")) t.classList.toggle("on", t === tab);
      for (const m of view.querySelectorAll(".mode")) m.hidden = m.id !== `mode-${tab.dataset.mode}`;
    });
  }
  let pkg = null;
  let work = null;
  const armPkg = () => ($("go-pkg").disabled = !(pkg && work));
  $("pick-pkg").addEventListener("click", () => $("f-pkg").click());
  $("pick-work").addEventListener("click", () => $("f-work").click());
  $("f-pkg").addEventListener("change", (e) => {
    pkg = e.target.files[0] || null;
    $("pkg-name").textContent = pkg ? `${pkg.name} (${store.fmtBytes(pkg.size)})` : "none chosen";
    armPkg();
  });
  $("f-work").addEventListener("change", (e) => {
    work = e.target.files[0] || null;
    $("work-name").textContent = work ? work.name : "none chosen";
    armPkg();
  });
  $("go-pkg").addEventListener("click", () => startImport([{ path: pkg.name, file: pkg }, { path: "work.bin", file: work }]));
  $("pick-dir").addEventListener("click", () => $("f-dir").click());
  $("f-dir").addEventListener("change", (e) => e.target.files.length && startImport(imp.entriesFromFiles(e.target.files)));
  $("pick-vpk").addEventListener("click", () => $("f-vpk").click());
  $("f-vpk").addEventListener("change", (e) => e.target.files.length && startImport(imp.entriesFromFiles(e.target.files)));
  $("pick-zip").addEventListener("click", () => $("f-zip").click());
  $("f-zip").addEventListener("change", (e) => e.target.files.length && startImport(imp.entriesFromFiles(e.target.files)));
  const drop = $("drop");
  drop.addEventListener("dragover", (e) => {
    e.preventDefault();
    drop.classList.add("over");
  });
  drop.addEventListener("dragleave", () => drop.classList.remove("over"));
  drop.addEventListener("drop", async (e) => {
    e.preventDefault();
    drop.classList.remove("over");
    const entries = await imp.entriesFromDrop(e.dataTransfer);
    if (entries.length) startImport(entries);
  });
}

async function startImport(entries) {
  const result = $("result");
  const total = entries.reduce((a, e) => a + e.file.size, 0);
  result.innerHTML = `<div class="card"><p>Reading ${entries.length} file${entries.length === 1 ? "" : "s"} (${store.fmtBytes(total)})...</p></div>`;
  let probe;
  try {
    probe = await imp.probe(entries);
  } catch (err) {
    result.innerHTML = `<div class="card error"><h2>Not a title this emulator recognises</h2><p>${esc(err.message || err)}</p>
      <p class="dim">It looks for a .pkg (with work.bin beside it), a folder with sce_pfs/files.db, or a homebrew .vpk. ${entries.length === 1 && /\.pkg$/i.test(entries[0].path) ? "Only the .pkg was given - it needs its work.bin too." : ""}</p></div>`;
    return;
  }
  const id = probe.titleId;
  const existing = id ? await store.readTitle(id) : null;
  let iconUrl = null;
  if (probe.icon0) iconUrl = URL.createObjectURL(new Blob([probe.icon0], { type: "image/png" }));
  result.innerHTML = `<div class="card confirm">
    <span class="icon big ${iconUrl ? "" : "noicon"}">${iconUrl ? `<img src="${iconUrl}" alt="" />` : ""}</span>
    <div>
      <h2>${esc(probe.title || id || "Unknown title")}</h2>
      <p class="meta">${esc(id || "no title id")}${probe.appVersion ? " &middot; v" + esc(probe.appVersion) : ""} &middot; ${probe.kind === "vpk" ? "homebrew" : esc(probe.kind) + (probe.zipped ? " in a zip" : "")} &middot; ${probe.files} files &middot; ${store.fmtBytes(probe.bytes)}</p>
      ${probe.missingWorkBin ? `<p class="warn">This pkg has no work.bin and none was picked with it. It cannot be decrypted without one - pick both files together.</p>` : ""}
      ${!id ? `<p class="warn">No param.sfo was found, so this cannot be named or stored.</p>` : ""}
      ${existing ? `<p class="warn">${esc(existing.title)} is already in the library; importing again replaces its files (saved data is kept).</p>` : ""}
      <div class="actions"><button id="do-import" class="btn primary" ${probe.missingWorkBin || !id ? "disabled" : ""}>${existing ? "Replace" : "Import"}</button><a class="btn" href="#/">Cancel</a></div>
      <div id="progress" hidden><div class="bar"><div id="bar-fill"></div></div><p id="prog-text" class="dim"></p></div>
    </div></div>`;
  $("do-import").addEventListener("click", async () => {
    $("do-import").disabled = true;
    $("progress").hidden = false;
    const fill = $("bar-fill");
    const text = $("prog-text");
    try {
      const done = await imp.run(entries, id, probe.bytes, (p) => {
        const pct = p.total ? Math.min(100, (100 * p.done) / p.total) : 0;
        fill.style.width = `${pct}%`;
        const left = p.rate > 0 ? `, about ${Math.max(0, Math.round((p.total - p.done) / p.rate))}s left` : "";
        text.textContent = `${p.stage} ${store.fmtBytes(p.done)} / ${store.fmtBytes(p.total)}${p.rate ? ` at ${(p.rate / 1e6).toFixed(0)} MB/s` : ""}${left} - ${p.file}`;
      });
      const meta = {
        titleId: id,
        title: probe.title || id,
        contentId: done.contentId || probe.contentId || "",
        appVersion: probe.appVersion || "",
        sourceKind: probe.kind,
        bytes: probe.bytes,
        files: done.count,
        hasIcon: !!probe.icon0,
        hasPic: !!probe.pic0,
        importedAt: Date.now(),
        lastPlayedAt: existing ? existing.lastPlayedAt || 0 : 0,
      };
      await store.removeTitleRecord(id);
      await store.writeTitle(meta, { "icon0.png": probe.icon0, "pic0.png": probe.pic0 });
      fill.style.width = "100%";
      text.textContent = "done";
      go(`#/title/${id}`);
    } catch (err) {
      text.textContent = "";
      result.insertAdjacentHTML("beforeend", `<div class="card error"><h2>Import failed</h2><pre>${esc(err.message || err)}</pre></div>`);
      $("do-import").disabled = false;
    }
  });
}

// ------------------------------- about -------------------------------

async function renderAbout() {
  view.innerHTML = `
    <h1>About</h1>
    <div class="card">
      <p>vitaslop is a clean-room PlayStation Vita emulator that runs entirely in this browser: a title's ARM code is translated to WebAssembly and run on the browser's own engine, its GPU stream is drawn with WebGPU, and everything you import stays in this browser's private storage. No server, no console firmware, no downloads from anywhere.</p>
      <p><a href="https://github.com/cretz/vitaslop">Source and documentation on GitHub</a>. Titles are never provided; import your own from a console you own.</p>
    </div>
    <section class="card"><h2>This browser</h2>${featureList()}</section>`;
}

// ------------------------------- boot -------------------------------

(async () => {
  if (await ensureIsolation()) return;
  features = await checkFeatures();
  playable = canPlay(features);
  const resume = sessionStorage.getItem("vitaslop.resume");
  if (resume) {
    sessionStorage.removeItem("vitaslop.resume");
    location.hash = resume;
  }
  route();
})();
