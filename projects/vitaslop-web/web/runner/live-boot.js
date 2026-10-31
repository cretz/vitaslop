// live-boot.js - a CLASSIC script that runs before live.js, so a live page whose module never
// loads still says why. live.js announces itself with {type:"ready"}; a module that fails to
// fetch, parse or link never runs at all, and the page sat on its first line with the runner
// waiting for a "ready" that could not come - three phone jobs said nothing but that.
"use strict";
(function () {
  // Shared with live.js, which clears it once the module has loaded.
  const RETRY_KEY = "vitaslop.live.retry";
  const say = (text) => {
    const el = document.getElementById("status");
    if (el) el.textContent = text;
    parent.postMessage({ type: "progress", text }, location.origin);
  };
  // Every file the page has requested so far, with its HTTP status (0 = no response). A module
  // script reports a failure ANYWHERE in its import graph against itself, so the file that
  // actually failed is found here, not in the element's `src`.
  // Only the files that did NOT answer 200 are named (the runner's status line holds 200
  // characters, and a list of every file cut off the very entries that mattered).
  const requested = () => {
    const all = performance.getEntriesByType("resource");
    const bad = all
      .filter((r) => r.responseStatus !== 200)
      .map((r) => `${r.name.replace(location.origin, "")} ${r.responseStatus ?? "?"} ${Math.round(r.duration)}ms`);
    return `${all.length} files, not 200: ${bad.join(", ") || "none"}`;
  };
  // A module script that fails to FETCH fires `error` on its element, which does not bubble -
  // hence the capture listener. A parse or link failure, and anything thrown later, reaches
  // `window` as an ordinary error.
  document.addEventListener(
    "error",
    (e) => {
      const t = e.target;
      if (!t || t.tagName !== "SCRIPT") return;
      say(`page script failed to load: ${t.src}; requested: ${requested()}`);
      // Ask again for each file that got NO response, and say what the network answers now -
      // and whether a service worker sits in between (one registered by the product page at
      // the origin's root controls this page too).
      const failed = performance.getEntriesByType("resource").filter((r) => r.responseStatus === 0);
      const sw = navigator.serviceWorker && navigator.serviceWorker.controller ? "a service worker controls this page" : "no service worker";
      Promise.all(
        failed.map((r) =>
          fetch(r.name, { cache: "no-store" }).then(
            (res) => `${r.name.replace(location.origin, "")} refetch ${res.status}`,
            (err) => `${r.name.replace(location.origin, "")} refetch failed: ${err && err.message}`
          )
        )
      ).then((lines) => {
        say(`${requested()}; ${sw}; ${lines.join("; ") || "no file had status 0"}`);
        // >>> A MODULE THAT FAILED TO FETCH STAYS FAILED FOR THE DOCUMENT'S LIFE (the module
        // map keeps the failure), so the only retry is a fresh document. MEASURED on the phone:
        // two of the graph's parallel fetches got no response at all, and the same files
        // fetched again a second later answered 200. One reload per page session; live.js
        // clears the mark once it has loaded, so a later job gets its own retry.
        let tried = "1";
        try {
          tried = sessionStorage.getItem(RETRY_KEY);
          sessionStorage.setItem(RETRY_KEY, "1");
        } catch {}
        if (tried !== "1") {
          say("reloading the live page once to retry the failed fetch");
          setTimeout(() => location.reload(), 500);
        }
      });
    },
    true
  );
  addEventListener("error", (e) => say(`page error: ${e.message} (${e.filename}:${e.lineno})`));
  addEventListener("unhandledrejection", (e) => say(`page rejection: ${e.reason && (e.reason.message || e.reason)}`));
  setTimeout(() => {
    if (!window.liveModuleLoaded) say(`live.js has not finished loading after 10 s; requested: ${requested()}`);
  }, 10000);
})();
