// fingerprint.js - what a device IS, asked of the browser (never read from its name), so a
// result can always be tied to the GPU and browser that produced it and two phones compared.

export async function fingerprint() {
  const fp = {
    ua: navigator.userAgent,
    cores: navigator.hardwareConcurrency ?? null,
    memoryGB: navigator.deviceMemory ?? null,
    // `screen` exists only on a page, not in the job Worker that also asks.
    screen: typeof screen !== "undefined" ? `${screen.width}x${screen.height} dpr ${devicePixelRatio}` : null,
    crossOriginIsolated: self.crossOriginIsolated,
    adapter: null,
    features: [],
    limits: {},
  };
  try {
    const a = navigator.gpu && (await navigator.gpu.requestAdapter());
    if (a) {
      const i = a.info || {};
      fp.adapter = [i.vendor, i.architecture, i.device, i.description].filter(Boolean).join(" / ") || "(no adapter info)";
      fp.features = [...a.features].sort();
      for (const k of [
        "maxTextureDimension2D",
        "maxBindGroups",
        "maxStorageBuffersPerShaderStage",
        "maxStorageBufferBindingSize",
        "maxBufferSize",
        "maxComputeWorkgroupSizeX",
        "maxComputeInvocationsPerWorkgroup",
        "maxColorAttachments",
        "maxInterStageShaderVariables",
      ]) {
        fp.limits[k] = a.limits[k];
      }
    } else {
      fp.adapter = "NO WEBGPU ADAPTER";
    }
  } catch (e) {
    fp.adapter = `adapter error: ${e.message}`;
  }
  return fp;
}
