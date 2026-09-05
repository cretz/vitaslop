// The play screen: one title, running, with the controls around it.
//
// Owns the boot (transpile worker, then run worker), the page-side services the
// worker cannot reach (audio context, geolocation, input events), the hard pause,
// the orientation and pad placement, the in-game menu and the diagnostics snapshot.
// The emulator itself lives in the worker; this file never touches guest state.
//
// The boot is structurally the same as the debug launcher's (web/debug/live.html)
// and the e2e harness page; a change to the start message or the worker's reports
// has to land in all three.

import { forwardInput } from "./worker-input.js";
import { forwardLocation } from "./location.js";
import { startAudio } from "./audio.js";
import { isComplete } from "./opfs.js";
import * as gamedata from "./gamedata.js";
import { mountTouchPad } from "./touchpad.js";
import { installGamepad } from "./gamepad.js";
import { runKnobs, vocabulary, touchTitle } from "./store.js";
import { renderVita } from "./vita.js";

const $ = (id) => document.getElementById(id);
const MAX_FRAMES = 0xffffffff;

export function createPlayer({ onExit }) {
  const root = $("player");
  const canvas = $("screen");
  const stage = $("stage");
  let worker = null;
  let touch = null;
  let pads = null;
  let audioPause = () => {};
  let audio = null;
  let muted = false;
  const applyMute = () => {
    $("mutebtn").innerHTML = muted ? "&#128263;" : "&#128266;";
    $("mutebtn").title = muted ? "Unmute" : "Mute";
    $("m-mute").checked = muted;
    if (!audio) return;
    try {
      if (muted) audio.node.disconnect();
      else audio.node.connect(audio.context.destination);
    } catch {}
  };
  const setMuted = (m) => {
    muted = m;
    applyMute();
    try {
      localStorage.setItem("vitaslop.muted", m ? "1" : "0");
    } catch {}
  };
  $("mutebtn").addEventListener("click", () => setMuted(!muted));
  $("m-mute").addEventListener("change", (e) => setMuted(e.target.checked));
  let running = false;
  let settings = null;
  let meta = null;
  let fresh = true; // the canvas can be transferred once per page life

  // ----- reports from the worker, kept for the snapshot -----
  const reports = { fps: "", perf: "", adapter: "", status: "", diag: "" };
  const notes = [];
  let fatalText = "";
  let hiddenCount = 0;
  let hardPauses = 0;
  let hardPausedMs = 0;
  document.addEventListener("visibilitychange", () => {
    if (document.visibilityState === "hidden") hiddenCount++;
  });

  const note = (text) => {
    notes.push(text);
    if (notes.length > 400) notes.shift();
    if (settings && settings.consoleNotes) console.log(text);
  };

  const fatal = (text) => {
    fatalText = fatalText ? fatalText + "\n\n" + text : text;
    $("fatal-text").textContent = fatalText;
    $("fatal").hidden = false;
    $("loading").hidden = true;
  };

  // ----- pause -----
  let hardPaused = false;
  let menuOpen = false;
  let pausedAt = 0;
  const wantPause = () =>
    menuOpen || (settings && settings.pauseOnBlur && (document.visibilityState === "hidden" || !document.hasFocus()));
  const applyPause = () => {
    const p = !!wantPause();
    if (p === hardPaused || !worker) return;
    hardPaused = p;
    worker.postMessage({ type: "pause", paused: p });
    audioPause(p);
    if (p) {
      pausedAt = performance.now();
      hardPauses++;
    } else hardPausedMs += performance.now() - pausedAt;
    root.classList.toggle("paused", p);
  };
  document.addEventListener("visibilitychange", applyPause);
  window.addEventListener("blur", applyPause);
  window.addEventListener("focus", applyPause);

  // ----- layout: orientation and pad placement -----
  const landscape = matchMedia("(orientation: landscape)");
  const coarse = matchMedia("(pointer: coarse)").matches;
  const applyLayout = () => {
    if (!settings) return;
    const mode = settings.pad.mode;
    let placement = mode;
    if (mode === "auto") placement = coarse ? (landscape.matches ? "overlay" : "beside") : "hidden";
    root.classList.remove("overlay", "beside", "hidden-pad");
    root.classList.add(placement === "hidden" ? "hidden-pad" : placement);
    root.classList.toggle("landscape", landscape.matches);
    root.classList.toggle("pixelated", settings.scaling === "integer");
    root.classList.toggle("stretch", settings.scaling === "stretch");
    if (settings.scaling === "integer") fitInteger();
    else canvas.style.width = canvas.style.height = "";
  };
  const fitInteger = () => {
    const r = stage.getBoundingClientRect();
    const k = Math.max(1, Math.floor(Math.min(r.width / 960, r.height / 544)));
    canvas.style.width = `${960 * k}px`;
    canvas.style.height = `${544 * k}px`;
  };
  landscape.addEventListener("change", applyLayout);
  window.addEventListener("resize", () => settings && settings.scaling === "integer" && fitInteger());

  // ----- fullscreen -----
  const isFull = () => !!(document.fullscreenElement || document.webkitFullscreenElement);
  const enterFullscreen = async () => {
    try {
      if (root.requestFullscreen) await root.requestFullscreen({ navigationUI: "hide" });
      else if (root.webkitRequestFullscreen) root.webkitRequestFullscreen();
    } catch {}
    try {
      if (coarse && screen.orientation && screen.orientation.lock) await screen.orientation.lock("landscape");
    } catch {}
  };
  const exitFullscreen = async () => {
    try {
      if (isFull()) await document.exitFullscreen();
    } catch {}
    try {
      if (screen.orientation && screen.orientation.unlock) screen.orientation.unlock();
    } catch {}
  };
  document.addEventListener("fullscreenchange", () => {
    $("m-fullscreen").textContent = isFull() ? "Exit fullscreen" : "Fullscreen";
  });

  // ----- the menu -----
  const menu = $("menu");
  const openMenu = (open) => {
    menuOpen = open;
    menu.hidden = !open;
    applyPause();
    if (open) {
      $("m-fps").checked = !!settings.showFps;
      $("m-pad-mode").value = settings.pad.mode;
      $("m-pad-opacity").value = settings.pad.opacity;
      $("m-pause-blur").checked = !!settings.pauseOnBlur;
      $("m-fullscreen").textContent = isFull() ? "Exit fullscreen" : "Fullscreen";
      $("m-diag").textContent = diagText();
      $("m-settings").href = `#/settings/${meta.titleId}`;
      $("m-title").textContent = meta.title;
      if (vocab) renderVita($("m-vita"), { buttons: vocab.buttons, controls: vocab.gamepadControls, keyboard: settings.keyboard, gamepad: settings.gamepad, mode: "keyboard", readonly: true });
    }
  };
  let vocab = null;
  $("menubtn").addEventListener("click", () => openMenu(!menuOpen));
  $("m-resume").addEventListener("click", () => openMenu(false));
  $("m-fullscreen").addEventListener("click", () => (isFull() ? exitFullscreen() : enterFullscreen()));
  $("m-quit").addEventListener("click", () => stop());
  $("m-fps").addEventListener("change", (e) => {
    settings.showFps = e.target.checked;
    $("fpsbadge").hidden = !settings.showFps;
    onRuntimeSetting({ showFps: settings.showFps });
  });
  $("m-pad-mode").addEventListener("change", (e) => {
    settings.pad.mode = e.target.value;
    applyLayout();
    onRuntimeSetting({ pad: { mode: settings.pad.mode } });
  });
  $("m-pad-opacity").addEventListener("input", (e) => {
    settings.pad.opacity = Number(e.target.value);
    if (touch) touch.setOpacity(settings.pad.opacity);
    onRuntimeSetting({ pad: { opacity: settings.pad.opacity } });
  });
  $("m-pause-blur").addEventListener("change", (e) => {
    settings.pauseOnBlur = e.target.checked;
    applyPause();
    onRuntimeSetting({ pauseOnBlur: settings.pauseOnBlur });
  });
  $("m-copy").addEventListener("click", () => copyText(diagText(), $("m-copy")));
  $("m-download").addEventListener("click", () => download(`vitaslop-${meta.titleId}-diag.txt`, diagText(), "text/plain"));
  $("m-shot").addEventListener("click", () => screenshot());
  $("fatal-copy").addEventListener("click", () => copyText(diagText(), $("fatal-copy")));
  $("fatal-quit").addEventListener("click", () => stop());
  document.addEventListener("keydown", (e) => {
    if (running && e.code === "Escape") openMenu(!menuOpen);
  });

  /// A runtime change from the menu is saved as a GLOBAL setting (the person changed
  /// how they want to play, not this title), by the app - the player only reports it.
  let onRuntimeSetting = () => {};

  // ----- diagnostics -----
  const audioLine = () => {
    if (!window.__audioStats) return "audio: no ring - this run is SILENT";
    const a = window.__audioStats();
    const rate = a.sampleRate || 48000;
    const s = (n) => (n / rate).toFixed(2);
    return (
      `audio: context=${a.state} peak=${(a.peak ?? 0).toFixed(4)}${a.peak > 0 ? "" : " (nothing audible yet)"} | ` +
      `written ${s(a.written)}s read ${s(a.read)}s | underrun ${s(a.underrun)}s overrun ${s(a.overrun)}s | ` +
      `backlog ${((1000 * (a.fill ?? 0)) / rate).toFixed(0)}ms`
    );
  };
  const diagText = () =>
    [
      `vitaslop diagnostics`,
      `title: ${meta ? `${meta.title} (${meta.titleId})` : "?"}`,
      `settings: ${JSON.stringify(settings || {})}`,
      `knobs: ${JSON.stringify(window.__runKnobs || {})}`,
      hiddenCount > 0 ? `WARNING: the page was backgrounded ${hiddenCount}x - a hidden page is throttled` : `page stayed in the foreground`,
      hardPauses > 0 ? `hard-paused ${hardPauses}x for ${(hardPausedMs / 1000).toFixed(1)}s in total` : `never hard-paused`,
      `user agent: ${navigator.userAgent}`,
      `screen: ${screen.width}x${screen.height} dpr ${devicePixelRatio} ${landscape.matches ? "landscape" : "portrait"}${isFull() ? " fullscreen" : ""}`,
      `adapter: ${reports.adapter}`,
      `${reports.fps}`,
      `${reports.perf}`,
      `status: ${reports.status}`,
      audioLine(),
      fatalText ? `\nFATAL\n${fatalText}` : ``,
      ``,
      reports.diag,
      ``,
      `notes:`,
      ...notes,
    ].join("\n");

  const copyText = async (text, btn) => {
    const label = btn.textContent;
    try {
      await navigator.clipboard.writeText(text);
      btn.textContent = "Copied";
    } catch {
      btn.textContent = "Copy failed - use Download";
    }
    setTimeout(() => (btn.textContent = label), 2000);
  };
  const download = (name, data, type) => {
    const url = URL.createObjectURL(data instanceof Blob ? data : new Blob([data], { type }));
    const a = document.createElement("a");
    a.href = url;
    a.download = name;
    document.body.appendChild(a);
    a.click();
    a.remove();
    setTimeout(() => URL.revokeObjectURL(url), 10000);
  };
  const screenshot = () =>
    requestAnimationFrame(() => {
      try {
        canvas.toBlob((b) => b && download(`vitaslop-${meta.titleId}-${Date.now()}.png`, b), "image/png");
      } catch (e) {
        note(`[shot] ${e}`);
      }
    });

  // ----- start / stop -----
  async function start(m, eff, { fullscreen = false, onSetting } = {}) {
    if (running) stop();
    if (!fresh) {
      // The canvas was transferred to a previous run's worker and cannot be again:
      // a fresh page is the only way to get a new one.
      sessionStorage.setItem("vitaslop.resume", `#/play/${m.titleId}`);
      location.reload();
      return;
    }
    meta = m;
    settings = eff;
    onRuntimeSetting = onSetting || (() => {});
    running = true;
    fatalText = "";
    $("fatal").hidden = true;
    $("fatal-text").textContent = "";
    root.hidden = false;
    document.title = `${m.title} - vitaslop`;
    document.body.classList.add("playing");
    $("loading").hidden = false;
    $("loading-title").textContent = m.title;
    $("fpsbadge").hidden = !settings.showFps;
    applyLayout();
    if (fullscreen) enterFullscreen();

    const status = (t) => {
      reports.status = t;
      $("loading-status").textContent = t;
    };
    try {
      if (!(await isComplete(m.titleId))) throw new Error("this title's import is incomplete - remove it and import it again");
      const knobs = await runKnobs(settings);
      window.__runKnobs = knobs;
      gamedata.setProfile(settings.profile);

      status("preparing the title (a few seconds on a desktop, up to a minute on a phone)...");
      const prebuilt = await new Promise((resolve, reject) => {
        const tw = new Worker("./transpile-worker.js", { type: "module" });
        tw.onmessage = (e) => {
          if (e.data.type === "panic") {
            fatal("RUST PANIC WHILE PREPARING\n" + e.data.message);
            return;
          }
          tw.terminate();
          e.data.type === "built" ? resolve(e.data.built) : reject(new Error(e.data.message));
        };
        tw.onerror = (e) => {
          tw.terminate();
          reject(new Error(e.message || "the prepare worker failed to start"));
        };
        tw.postMessage({ titleId: m.titleId, knobs });
      });
      if (!running) return;

      status("starting...");
      worker = new Worker("./worker.js", { type: "module" });
      worker.onmessage = (e) => {
        const d = e.data;
        if (d.type === "report") {
          reports[d.id] = d.text;
          if (d.id === "fps") $("fpsbadge").textContent = d.text.replace(/^fps:\s*/, "");
          if (d.id === "status") {
            $("loading-status").textContent = d.text;
            // The first present is the moment the loading screen has nothing to say.
            if (/present|frame|fps/i.test(d.text)) $("loading").hidden = true;
          }
        } else if (d.type === "note") note(d.text);
        else if (d.type === "error") fatal("ERROR\n" + d.message);
        else if (d.type === "panic") fatal("RUST PANIC\n" + d.message);
        else if (d.type === "setup") {
          note(`[setup] ${d.status}`);
          $("loading").hidden = true;
        }
      };
      worker.onerror = (e) => {
        const site = e.filename ? ` at ${e.filename}:${e.lineno || "?"}:${e.colno || "?"}` : "";
        fatal("WORKER DIED\n" + (e.message || "died") + site + (e.error && e.error.stack ? "\n" + e.error.stack : ""));
      };
      const flush = () => worker && worker.postMessage({ type: "flush-game-data" });
      document.addEventListener("visibilitychange", () => document.visibilityState === "hidden" && flush());
      window.addEventListener("pagehide", flush);

      worker.postMessage({ type: "keymap", json: JSON.stringify(settings.keyboard) });
      forwardInput(worker, canvas);
      vocab = await vocabulary();
      touch = mountTouchPad($("pad"), worker, settings.keyboard, { vibrate: settings.pad.vibrate });
      touch.setOpacity(settings.pad.opacity);
      touch.setScale(settings.pad.scale);
      pads = installGamepad(worker, vocab, settings, (msg) => note("[pad] " + msg), (name, down) => touch && touch.setHeld(name, down));
      // Keyboard presses light the on-screen control they map to.
      const byCode = {};
      for (const [name, code] of Object.entries(settings.keyboard)) byCode[code] = name;
      const lightKey = (down) => (e) => {
        const name = byCode[e.code];
        if (name && touch && !menuOpen) touch.setHeld(name, down);
      };
      document.addEventListener("keydown", lightKey(true));
      document.addEventListener("keyup", lightKey(false));
      forwardLocation(worker, note);

      try {
        audio = await startAudio(note);
        audioPause = audio.pause;
        window.__audioStats = audio.stats;
        var audioRing = audio.ring;
        try {
          muted = localStorage.getItem("vitaslop.muted") === "1";
        } catch {}
        applyMute();
      } catch (err) {
        note(`[audio] could not start; this run is SILENT: ${err}`);
      }
      if (!running) return;

      const offscreen = canvas.transferControlToOffscreen();
      fresh = false;
      worker.postMessage(
        { offscreen, titleId: m.titleId, recipe: settings.recipe || "", maxFrames: MAX_FRAMES, knobs, prebuilt, audioRing, profile: settings.profile },
        [offscreen]
      );
      touchTitle(m.titleId, { lastPlayedAt: Date.now() });
      applyPause();
    } catch (err) {
      fatal("COULD NOT START\n" + ((err && (err.stack || err.message)) || err));
    }
  }

  function stop() {
    if (!running) return;
    running = false;
    if (worker) {
      try {
        worker.postMessage({ type: "flush-game-data" });
      } catch {}
      const w = worker;
      worker = null;
      // Give the flush a moment to land before the worker is torn down.
      setTimeout(() => w.terminate(), 500);
    }
    if (touch) touch.destroy();
    if (pads) pads.stop();
    audio = null;
    touch = pads = null;
    audioPause(true);
    openMenu(false);
    exitFullscreen();
    root.hidden = true;
    document.body.classList.remove("playing");
    onExit();
  }

  return { start, stop, isRunning: () => running };
}
