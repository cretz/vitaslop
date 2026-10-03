// decoder-latency - how many access units this device's WebCodecs H.264 decoder takes in before
// it hands pictures back, per configuration. A console decode API is "submit one unit, poll for
// its picture"; a decoder that holds several units starves a title that recycles its input
// buffers only when their pictures return.
//
// params: { stream: <runner asset: {codec, codedWidth, codedHeight, description (b64 avcC),
//   chunks: [{key, data (b64, length-prefixed)}]}>, waitMs (per unit, default 40), units (all) }

const b64 = (s) => Uint8Array.from(atob(s), (c) => c.charCodeAt(0));

async function one(stream, cfg, waitMs, units) {
  const base = {
    codec: stream.codec,
    codedWidth: stream.codedWidth,
    codedHeight: stream.codedHeight,
    description: b64(stream.description),
  };
  const config = { ...base, ...cfg };
  let supported = null;
  try {
    const s = await VideoDecoder.isConfigSupported(config);
    supported = s.supported;
  } catch (e) {
    return { cfg, supported: false, error: String(e) };
  }
  if (!supported) return { cfg, supported };
  let outputs = 0;
  let error = null;
  const outAt = [];
  let waiter = null;
  const dec = new VideoDecoder({
    output: (f) => {
      outputs++;
      f.close();
      if (waiter) waiter();
    },
    error: (e) => (error = String(e)),
  });
  dec.configure(config);
  const t0 = performance.now();
  const n = Math.min(units || stream.chunks.length, stream.chunks.length);
  for (let i = 0; i < n && !error; i++) {
    const c = stream.chunks[i];
    dec.decode(new EncodedVideoChunk({ type: c.key ? "key" : "delta", timestamp: i * 33333, data: b64(c.data) }));
    // What a title polling once per frame sees: wait for the picture that unit is owed, but no
    // longer than a frame or so.
    const want = i + 1;
    const until = performance.now() + waitMs;
    while (outputs < want && performance.now() < until && !error) {
      await new Promise((r) => {
        waiter = r;
        setTimeout(r, 2);
      });
    }
    outAt.push(outputs);
  }
  // With no more input: does it hand the rest back on its own, or only on flush?
  const held = n - outputs;
  await new Promise((r) => setTimeout(r, 500));
  const afterIdle = outputs;
  let afterFlush = null;
  try {
    await dec.flush();
    afterFlush = outputs;
  } catch (e) {
    error = error || String(e);
  }
  const ms = performance.now() - t0;
  try {
    dec.close();
  } catch {}
  // The steady lag: units in minus pictures out, over the second half of the run.
  const lags = outAt.map((o, i) => i + 1 - o);
  const tail = lags.slice(Math.floor(lags.length / 2));
  return {
    cfg,
    supported,
    error,
    units: n,
    firstPictureAfter: outAt.findIndex((o) => o > 0) + 1,
    maxLag: Math.max(...lags),
    steadyLag: tail.length ? Math.max(...tail) : null,
    heldAtEnd: held,
    afterIdle,
    afterFlush,
    ms: Math.round(ms),
    outAt: outAt.join(","),
  };
}

export async function run(params, ctx) {
  if (typeof VideoDecoder === "undefined") return { summary: "NO WebCodecs VideoDecoder in a worker here" };
  const stream = await ctx.asset(params.stream, "json");
  const waitMs = Number(params.waitMs || 40);
  const configs = params.configs || [
    { hardwareAcceleration: "prefer-software", optimizeForLatency: true },
    { hardwareAcceleration: "prefer-software" },
    { hardwareAcceleration: "no-preference", optimizeForLatency: true },
    { hardwareAcceleration: "prefer-hardware", optimizeForLatency: true },
    { hardwareAcceleration: "prefer-hardware" },
  ];
  const results = [];
  for (const cfg of configs) {
    ctx.progress(`decoder ${JSON.stringify(cfg)}`);
    results.push(await one(stream, cfg, waitMs, params.units));
  }
  const line = (r) =>
    `${r.cfg.hardwareAcceleration}${r.cfg.optimizeForLatency ? "+lowlat" : ""}: ` +
    (r.supported ? `first@${r.firstPictureAfter} steadyLag ${r.steadyLag} maxLag ${r.maxLag} held ${r.heldAtEnd} idle->${r.afterIdle} flush->${r.afterFlush}${r.error ? " ERR " + r.error : ""}` : `UNSUPPORTED${r.error ? " " + r.error : ""}`);
  return { summary: results.map(line).join(" | "), results };
}
