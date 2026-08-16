// The page half of SceLibLocation: the real `navigator.geolocation`, relayed to the
// emulator worker.
//
// `navigator.geolocation` exists on Window ONLY - a Worker has no such property at all -
// so the worker running the guest cannot call it. This is the same split the page already
// owns for audio (the AudioContext) and for pointer/keyboard input, and it uses the same
// postMessage seam:
//
//   worker -> page   {type:"location-request"}   the guest called sceLocationConfirm
//   worker -> page   {type:"location-release"}   the guest closed its last handle
//   page   -> worker worker_location_fix(...)    a position arrived
//   page   -> worker worker_location_error(code) watchPosition failed
//   page   -> worker worker_location_unavailable() no Geolocation API here at all
//
// The browser raises its permission prompt on the first watchPosition, which is exactly
// when the guest's own permission dialog should be showing. Nothing here answers for the
// user: until the browser calls back, the worker's cell stays "pending", which the guest
// reads as a RUNNING dialog.
//
// WHY watchPosition AND NOT getCurrentPosition: a title asks for position repeatedly
// while it is on a location screen, and getCurrentPosition would start a fresh
// acquisition each time (slow, and battery-expensive on a phone). A single watch delivers
// updates as the device moves, which is also what the guest API's own
// StartLocationCallback models.

/// Install the location relay on `worker`. Returns a function that tears the watch down,
/// for a page that restarts a run.
///
/// `onNote` (optional) receives one-line human-readable status, so a page with a
/// diagnostics panel can show why location is or is not working - the failure mode this
/// replaces is a title silently sitting on "acquiring" for ever with nothing said.
export function forwardLocation(worker, onNote) {
  // Notes go BACK through the worker so they land in the wasm logger's on-page WARN
  // mirror and in the /diag sink. A phone has no console anyone is holding, and the
  // failure this exists to prevent is a title sitting on "acquiring position" for ever
  // with the reason (an insecure origin, a refusal, a timeout) recorded nowhere.
  const note = (s) => {
    console.log("[location] " + s);
    worker.postMessage({ type: "location-note", message: "location: " + s });
    if (onNote) onNote(s);
  };

  let watchId = null;

  // High accuracy: a Vita asking for position through AGPS/GPS is asking for the real
  // thing, and a coarse Wi-Fi fix would be a quieter kind of wrong answer. The timeout is
  // generous because a cold GPS fix genuinely takes tens of seconds outdoors, and a
  // TIMEOUT is reported to the guest as "no fix yet" (not as a refusal), so a slow fix
  // costs nothing but the wait. maximumAge 0: never hand the guest a cached position it
  // would read as current.
  const OPTIONS = { enableHighAccuracy: true, timeout: 60000, maximumAge: 0 };

  const start = () => {
    if (watchId !== null) return; // already watching; do not stack watches
    if (!navigator.geolocation) {
      // A real state, not a failure of ours: an insecure origin does not expose the API.
      note(
        "this context has no Geolocation API (an https origin is required) - the title " +
          "will be told the location provider is unavailable"
      );
      worker.postMessage({ type: "location-unavailable" });
      return;
    }
    note("requesting position - the browser will prompt if it has not already");
    watchId = navigator.geolocation.watchPosition(
      (pos) => {
        const c = pos.coords;
        worker.postMessage({
          type: "location-fix",
          // Latitude/longitude are always present on a Position. The rest are nullable
          // per the spec and NaN on some devices when they are not meaningful (heading
          // and speed are undefined at a standstill); both are forwarded as-is and the
          // Rust side turns either into "unknown", never into zero.
          latitude: c.latitude,
          longitude: c.longitude,
          altitude: c.altitude,
          accuracy: c.accuracy,
          heading: c.heading,
          speed: c.speed,
          timestamp: pos.timestamp,
        });
      },
      (err) => {
        // code 1 = PERMISSION_DENIED (the user said no), 2 = POSITION_UNAVAILABLE,
        // 3 = TIMEOUT. Only 1 is a refusal; the Rust side keeps them apart.
        note(`watchPosition error ${err.code}: ${err.message}`);
        worker.postMessage({ type: "location-error", code: err.code });
      },
      OPTIONS
    );
  };

  const stop = () => {
    if (watchId === null) return;
    navigator.geolocation.clearWatch(watchId);
    watchId = null;
    note("stopped watching position");
  };

  // The worker's own onmessage is installed by the page; this listener runs alongside it
  // rather than replacing it, so neither has to know about the other's messages.
  worker.addEventListener("message", (e) => {
    const d = e.data;
    if (!d) return;
    if (d.type === "location-request") start();
    else if (d.type === "location-release") stop();
  });

  return stop;
}
