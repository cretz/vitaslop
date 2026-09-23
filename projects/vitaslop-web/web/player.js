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
import * as navpad from "./navpad.js";

const $ = (id) => document.getElementById(id);
const MAX_FRAMES = 0xffffffff;

/// `onExit` is called after a quit; `onRestart(titleId, fullscreen)` after a restart,
/// with the run already torn down, to start the same title through the app's one
/// start path.
export function createPlayer({ onExit, onRestart }) {
  const root = $("player");
  // Replaced per run: a canvas hands its drawing to a worker ONCE (see start).
  let canvas = $("screen");
  const stage = $("stage");
  let worker = null;
  let touch = null;
  let pads = null;
  // Undo functions for listeners installed per run on targets that outlive it.
  const unlisten = [];
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
  // >>> WHICH BUNDLE IS THIS. Written beside the wasm by `build.mjs`; see the note there for
  // why. Read once, here, and never blocking anything: a dump whose build line says
  // `unavailable` is still a dump, but one with no build line at all cannot be told apart from
  // a dump of yesterday's bytes.
  let buildStamp = "not read yet";
  fetch("./pkg/build-stamp.txt", { cache: "no-store" })
    .then((r) => (r.ok ? r.text() : Promise.reject(new Error(`HTTP ${r.status}`))))
    .then((t) => {
      buildStamp = t.trim();
    })
    .catch((e) => {
      buildStamp = `unavailable (${e.message}) - this bundle was built before the stamp existed, or the file was not served`;
    });
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
  // Asked for ONCE per run, from Play (with its gesture) or from the menu button, and
  // never from a resize, orientation, focus or visibility handler: Chrome for Android
  // shows its "swipe down to exit" toast on every fullscreen layout and again every
  // time the window regains focus while fullscreen (FullscreenHtmlApiHandlerBase.
  // onWindowFocusChanged -> FullscreenToast.showNotificationToast), so a re-request or
  // a dim-and-wake is a toast the page caused. The wake lock below removes the dim.
  const isFull = () => !!(document.fullscreenElement || document.webkitFullscreenElement);
  const enterFullscreen = async () => {
    try {
      if (root.requestFullscreen) await root.requestFullscreen({ navigationUI: "hide" });
      else if (root.webkitRequestFullscreen) root.webkitRequestFullscreen();
    } catch {}
    // The lock is a setting: a phone held in portrait is a legitimate way to play with
    // the pad below the screen, and the lock takes that away.
    try {
      if (coarse && settings && settings.lockLandscape !== false && screen.orientation && screen.orientation.lock) {
        await screen.orientation.lock("landscape");
      }
    } catch {}
  };
  // >>> ASKED FOR INSIDE THE TAP, NOT AFTER IT. `play` reads the title and its settings out
  // of storage before `start` runs, and on a freshly reloaded page those reads are cold: the
  // tap's transient activation had lapsed by the time `start` asked, so a phone that pressed
  // Play started in portrait with no fullscreen. The Play handler calls this synchronously
  // and `start` then skips its own request for that run.
  let fullscreenAsked = false;
  const askFullscreenNow = () => {
    fullscreenAsked = true;
    enterFullscreen();
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

  // ----- screen wake lock -----
  // A game is watched, not touched, for minutes at a time; without this the screen
  // dims and every wake is a focus change (and the fullscreen toast again). The lock is
  // dropped by the browser whenever the page is hidden, so it is re-asked for on every
  // return to the foreground while a run is on. Denied or absent, nothing changes.
  let wakeLock = null;
  const holdWake = async () => {
    if (!running || wakeLock || !navigator.wakeLock || document.visibilityState !== "visible") return;
    try {
      const lock = await navigator.wakeLock.request("screen");
      if (!running) {
        lock.release().catch(() => {});
        return;
      }
      wakeLock = lock;
      lock.addEventListener("release", () => {
        if (wakeLock === lock) wakeLock = null;
      });
    } catch (e) {
      note(`[wake] not held: ${e && e.message ? e.message : e}`);
    }
  };
  const dropWake = () => {
    const lock = wakeLock;
    wakeLock = null;
    if (lock) lock.release().catch(() => {});
  };
  document.addEventListener("visibilitychange", () => document.visibilityState === "visible" && holdWake());

  // ----- the menu -----
  const menu = $("menu");
  const confirmBox = $("m-confirm");
  let confirmYes = null;
  // The Yes/No panel that stands in for the menu body until it is answered: a
  // window.confirm is what a fullscreen phone browser hides or shrinks.
  const askConfirm = (text, yes) => {
    confirmYes = yes;
    $("m-confirm-text").textContent = text;
    $("m-body").hidden = true;
    confirmBox.hidden = false;
    $("m-confirm-no").focus();
  };
  const closeConfirm = () => {
    confirmYes = null;
    confirmBox.hidden = true;
    $("m-body").hidden = false;
  };
  const menuBack = () => (confirmYes ? closeConfirm() : openMenu(false));
  const openMenu = (open) => {
    menuOpen = open;
    menu.hidden = !open;
    closeConfirm();
    applyPause();
    // The pad changes hands with the menu: gamepad.js lets go first, then the
    // navigator takes it (and the other way round on close), so no press is seen by both.
    if (open) {
      if (pads) pads.suspend(true);
      navpad.attach(menu, { onBack: menuBack, onStart: () => openMenu(false) });
    } else {
      navpad.detach();
      if (pads) pads.suspend(false);
    }
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
      $("m-resume").focus();
    }
  };
  let vocab = null;
  $("menubtn").addEventListener("click", () => openMenu(!menuOpen));
  $("m-resume").addEventListener("click", () => openMenu(false));
  $("m-fullscreen").addEventListener("click", () => (isFull() ? exitFullscreen() : enterFullscreen()));
  $("m-restart").addEventListener("click", () => askConfirm("Restart this game from the beginning? Anything not saved is lost.", restart));
  $("m-quit").addEventListener("click", () => askConfirm("Quit to the library? Anything not saved is lost.", stop));
  $("m-confirm-yes").addEventListener("click", () => {
    const yes = confirmYes;
    closeConfirm();
    if (yes) yes();
  });
  $("m-confirm-no").addEventListener("click", closeConfirm);
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
  // The same file the menu's Download writes, from the panel that is shown INSTEAD of the menu
  // once the run is over - see the markup for why the clipboard alone was not enough here.
  $("fatal-download").addEventListener("click", () =>
    download(`vitaslop-${meta ? meta.titleId : "unknown"}-fatal.txt`, diagText(), "text/plain"));
  $("fatal-quit").addEventListener("click", () => stop());
  document.addEventListener("keydown", (e) => {
    if (!running || e.code !== "Escape") return;
    if (menuOpen) menuBack();
    else openMenu(true);
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
  /// Knobs armed by the LINK - `?knobs=VITASLOP_A%3D1,VITASLOP_B%3D2` - merged OVER the ones
  /// the settings record produces.
  ///
  /// A device run has to be handed over ALREADY ARMED. Typing a knob name into a text box on a
  /// phone keyboard is where an A/B silently becomes a single arm, and a run that comes back
  /// with a knob misspelled reads exactly like a null result. The armed names go into the
  /// diagnostics file's `knobs:` line below, so what was actually set comes back with the
  /// numbers rather than being taken on trust. A name no reader routes through the override
  /// table still panics on boot, by design - loud beats a knob that did nothing.
  const linkKnobs = () => {
    const raw = new URLSearchParams(location.search).get("knobs");
    if (!raw) return {};
    const out = {};
    for (const part of raw.split(/[,\n]/)) {
      const t = part.trim();
      if (!t) continue;
      const i = t.indexOf("=");
      out[i < 0 ? t : t.slice(0, i)] = i < 0 ? "1" : t.slice(i + 1);
    }
    return out;
  };

  const diagText = () =>
    [
      `vitaslop diagnostics`,
      `title: ${meta ? `${meta.title} (${meta.titleId})` : "?"}`,
      `settings: ${JSON.stringify(settings || {})}`,
      `knobs: ${JSON.stringify(window.__runKnobs || {})}`,
      hiddenCount > 0 ? `WARNING: the page was backgrounded ${hiddenCount}x - a hidden page is throttled` : `page stayed in the foreground`,
      hardPauses > 0 ? `hard-paused ${hardPauses}x for ${(hardPausedMs / 1000).toFixed(1)}s in total` : `never hard-paused`,
      `build: ${buildStamp}`,
      `page loaded: ${new Date(performance.timeOrigin).toISOString()} (${((Date.now() - performance.timeOrigin) / 60000).toFixed(1)} min ago)`,
      `user agent: ${navigator.userAgent}`,
      `screen: ${screen.width}x${screen.height} dpr ${devicePixelRatio} ${landscape.matches ? "landscape" : "portrait"}${isFull() ? " fullscreen" : ""}`,
      `adapter: ${reports.adapter}`,
      // These two are emitted under their own ids at device creation and were being STORED and
      // never printed, so no dump has ever carried them - see the note in `lib.rs` beside
      // `adapter-features` for what that cost.
      `${reports["adapter-compression"] || "adapter compressed-texture support: (not reported)"}`,
      `${reports["adapter-features"] || "adapter features: (not reported)"}`,
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
      // The canvas was transferred to a previous run's worker and cannot be again, so
      // this run gets a new element in its place: same id and size, nothing drawn yet.
      // (A page reload was the old answer, and a reload lands on the title page with no
      // gesture to start from - a restart needs the run to begin here.)
      const next = canvas.cloneNode(false);
      canvas.replaceWith(next);
      canvas = next;
      fresh = true;
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
    // Say on the screen itself what the link armed, so the arm is confirmable without
    // opening the diagnostics file.
    const armed = Object.entries(linkKnobs()).map(([k, v]) => `${k}=${v}`).join(" ");
    $("loading-title").textContent = armed ? `${m.title} - ARMED ${armed}` : m.title;
    $("fpsbadge").hidden = !settings.showFps;
    applyLayout();
    // Already fullscreen (a restart keeps it): no second request, no second toast.
    if (fullscreen && !isFull() && !fullscreenAsked) enterFullscreen();
    fullscreenAsked = false;
    holdWake();

    const status = (t) => {
      reports.status = t;
      $("loading-status").textContent = t;
    };
    try {
      if (!(await isComplete(m.titleId))) throw new Error("this title's import is incomplete - remove it and import it again");
      const knobs = { ...(await runKnobs(settings)), ...linkKnobs() };
      window.__runKnobs = knobs;
      gamedata.setProfile(settings.profile);

      status("preparing the title (a few seconds on a desktop, up to a minute on a phone)...");
      // The RUN worker comes first: it reserves the guest's memory inside its own and says
      // where, and the throwaway transpile worker builds the module for that place (see
      // worker.js's "reserve" message). The run worker then idles until the start message.
      worker = new Worker("./worker.js", { type: "module" });
      const hostOff = await new Promise((resolve, reject) => {
        worker.onmessage = (e) => {
          const d = e.data;
          if (d.type === "reserved") resolve(d.hostOff);
          else if (d.type === "error") reject(new Error(d.message));
          else if (d.type === "panic") reject(new Error("RUST PANIC WHILE RESERVING\n" + d.message));
        };
        worker.onerror = (e) => reject(new Error(e.message || "the run worker failed to start"));
        worker.postMessage({ type: "reserve", knobs });
      });
      if (!running) return;
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
        tw.postMessage({ titleId: m.titleId, knobs, hostOff });
      });
      if (!running) return;

      status("starting...");
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
      const onHide = () => document.visibilityState === "hidden" && flush();
      document.addEventListener("visibilitychange", onHide);
      window.addEventListener("pagehide", flush);
      unlisten.push(() => {
        document.removeEventListener("visibilitychange", onHide);
        window.removeEventListener("pagehide", flush);
      });

      worker.postMessage({ type: "keymap", json: JSON.stringify(settings.keyboard) });
      unlisten.push(forwardInput(worker, canvas));
      vocab = await vocabulary();
      touch = mountTouchPad($("pad"), worker, settings.keyboard, { vibrate: settings.pad.vibrate });
      touch.setOpacity(settings.pad.opacity);
      touch.setScale(settings.pad.scale);
      pads = installGamepad(
        worker,
        vocab,
        settings,
        (msg) => note("[pad] " + msg),
        (name, down) => touch && touch.setHeld(name, down),
        () => running && !menuOpen && openMenu(true)
      );
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

  /// Tear the run down and start the same title again. The screen stays as it is
  /// (fullscreen and the orientation lock included) under the loading panel; the app's
  /// one start path does the rest.
  function restart() {
    if (!running) return;
    const id = meta.titleId;
    const full = isFull();
    stop(false);
    onRestart(id, full);
  }

  /// `exit` false keeps the player on screen for a restart that follows at once.
  function stop(exit = true) {
    if (!running) return;
    running = false;
    dropWake();
    if (worker) {
      try {
        worker.postMessage({ type: "flush-game-data" });
      } catch {}
      const w = worker;
      worker = null;
      // Give the flush a moment to land before the worker is torn down.
      setTimeout(() => w.terminate(), 500);
    }
    // Page-level listeners installed for this run come off with it. `document` and `window`
    // outlive a run, so anything left here is still live for the next game.
    for (const undo of unlisten.splice(0)) undo();
    if (touch) touch.destroy();
    if (pads) pads.stop();
    audio = null;
    touch = pads = null;
    audioPause(true);
    openMenu(false);
    if (!exit) return;
    // >>> THE PLAYER IS HIDDEN ONLY AFTER FULLSCREEN HAS ACTUALLY LET GO. Hiding the
    // fullscreen element and navigating while the exit was still in flight left Chrome for
    // Android routing every touch to the dead fullscreen layer: the library drew, and no
    // link on it answered until a reload. A run started meanwhile owns the player again,
    // so it is left alone.
    exitFullscreen().finally(() => {
      if (running) return;
      root.hidden = true;
      document.body.classList.remove("playing");
      onExit();
    });
  }

  return { start, stop, askFullscreenNow, isRunning: () => running };
}
