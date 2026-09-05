// A service worker whose only job is to add the two headers that make a page
// cross-origin isolated, for hosts that cannot set headers (GitHub Pages).
//
// SharedArrayBuffer - the audio ring, the storage ring, Atomics.wait in the emulator
// worker - exists only on a cross-origin-isolated page, and isolation is granted by
// the Cross-Origin-Opener-Policy and Cross-Origin-Embedder-Policy headers on the
// document and every subresource. A static host serves what it serves. So this worker
// sits between the page and the network and stamps the headers onto every response
// it relays. The page registers it once, reloads, and is isolated from then on.
//
// Nothing here is cached and nothing is rewritten; a response passes through with its
// body untouched. `credentialless` is not used because it is not everywhere yet and
// this site loads nothing cross-origin anyway.

self.addEventListener("install", () => self.skipWaiting());
self.addEventListener("activate", (e) => e.waitUntil(self.clients.claim()));

self.addEventListener("fetch", (e) => {
  const req = e.request;
  if (req.cache === "only-if-cached" && req.mode !== "same-origin") return;
  e.respondWith(
    fetch(req)
      .then((res) => {
        // A 0-status opaque response cannot carry headers; pass it as-is.
        if (res.status === 0) return res;
        const headers = new Headers(res.headers);
        headers.set("Cross-Origin-Embedder-Policy", "require-corp");
        headers.set("Cross-Origin-Opener-Policy", "same-origin");
        headers.set("Cross-Origin-Resource-Policy", "same-origin");
        return new Response(res.body, { status: res.status, statusText: res.statusText, headers });
      })
      .catch((err) => new Response(String(err), { status: 502 }))
  );
});
