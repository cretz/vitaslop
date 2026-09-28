// The emulator bundle, fetched and compiled ONCE per page and handed to every worker that needs
// it as a `WebAssembly.Module` (the run worker's reserve, the transpile worker).
//
// Each worker used to fetch and compile the 9 MB bundle for itself. On a phone reaching the dev
// server through its self-signed certificate there is no HTTP cache at all, so that was the whole
// 9 MB over the LAN per worker per play - MEASURED (runner, MLB) 8.7 s each, twice before frame 1.
// A compiled module posts between workers without copying its code, so one compile serves both,
// and a second play from the same page pays nothing.
//
// Keyed on the build stamp (read fresh each time, it is 76 bytes): the glue JS beside the wasm is
// never cached, so a rebuild mid-session must not pair the new glue with the old module.
const compiled = new Map();

export async function bundleModule(threads) {
  const dir = threads ? "./pkg-threads/" : "./pkg/";
  let stamp = "";
  try {
    const r = await fetch(new URL(dir + "build-stamp.txt", import.meta.url), { cache: "no-store" });
    if (r.ok) stamp = await r.text();
  } catch {}
  const url = new URL(dir + "vitaslop_web_bg.wasm", import.meta.url).href;
  const key = url + "\n" + stamp;
  let p = compiled.get(key);
  if (!p) {
    compiled.clear();
    p = WebAssembly.compileStreaming(fetch(url)).catch((err) => {
      compiled.delete(key);
      throw err;
    });
    compiled.set(key, p);
  }
  return p;
}
