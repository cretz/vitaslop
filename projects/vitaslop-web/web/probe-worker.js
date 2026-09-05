// The one OPFS call the emulator needs and cannot make on the main thread: a
// synchronous access handle. A file of its own (not a blob-built worker) so the page's
// Content-Security-Policy can keep `worker-src` at 'self'.
self.onmessage = async () => {
  try {
    const root = await navigator.storage.getDirectory();
    const dir = await root.getDirectoryHandle(".probe", { create: true });
    const fh = await dir.getFileHandle("p", { create: true });
    const h = await fh.createSyncAccessHandle();
    h.write(new Uint8Array([1]));
    h.flush();
    h.close();
    await root.removeEntry(".probe", { recursive: true });
    self.postMessage({ ok: true });
  } catch (e) {
    self.postMessage({ ok: false, err: String(e && e.message ? e.message : e) });
  }
};
