//! Replay a captured FRAME (`capsule::read_frame`, slim or not) on THIS browser's GPU and hand
//! the pixels back - the browser twin of `vitaslop-native/examples/frame-replay.rs`, for the
//! device runner (`web/runner/jobs/frame-replay.js`).
//!
//! # Why
//! A picture defect that appears only on a phone (MLB's green players, 2026-09) cannot be
//! bisected on the desktop, and a person driving the phone to the frame is the cost the runner
//! exists to remove. With this the phone renders the SAME frame the desktop does - the frame
//! passes one prefix at a time, a pass one draw prefix at a time - and the first prefix whose
//! pixels differ names the pass, then the draw.
//!
//! Same renderer as the live page (`GxmRenderer` + `RenderSceneBuilder`), rendered headless into
//! an RGBA8 texture at the frame's own display size, as the native replay does.

use std::sync::Arc;
use std::sync::atomic::{AtomicBool, Ordering};

use vitaslop_platform::gpu::GxmRenderer;
use vitaslop_runtime::render::RenderSceneBuilder;
use wasm_bindgen::prelude::*;

const OUTPUT_FORMAT: wgpu::TextureFormat = wgpu::TextureFormat::Rgba8Unorm;

/// Render `bytes` (a frame capsule) once per entry of `scene_limits`: 0 = every pass, N = only
/// the first N passes (so pass N-1 is the image). `draw_limit` >= 0 cuts the LAST rendered pass
/// to its first `draw_limit` draws. Returns `[{limit, width, height, rgba: Uint8Array, ms}]`.
#[wasm_bindgen]
pub async fn frame_replay(bytes: Vec<u8>, scene_limits: Vec<u32>, draw_limit: i32) -> Result<js_sys::Array, JsValue> {
    // The renderer's warnings (a refused pipeline, a missing uniform window) go to the console
    // like the live page's, so a replay that differs from a run can say why.
    crate::logging::install_panic_hook();
    crate::logging::init();
    let fc = vitaslop_runtime::capsule::read_frame(&mut &bytes[..])
        .map_err(|e| JsValue::from_str(&format!("frame capsule: {e}")))?;
    drop(bytes);
    let instance = wgpu::Instance::default();
    let adapter = instance
        .request_adapter(&wgpu::RequestAdapterOptions {
            power_preference: wgpu::PowerPreference::HighPerformance,
            force_fallback_adapter: false,
            compatible_surface: None,
            apply_limit_buckets: false,
        })
        .await
        .map_err(|_| JsValue::from_str("no WebGPU adapter"))?;
    let (device, queue) = adapter
        .request_device(&wgpu::DeviceDescriptor {
            label: Some("vitaslop-frame-replay"),
            required_features: vitaslop_platform::gpu::wanted_features(&adapter),
            required_limits: vitaslop_platform::gpu::device_limits(&adapter),
            experimental_features: wgpu::ExperimentalFeatures::disabled(),
            memory_hints: wgpu::MemoryHints::default(),
            trace: wgpu::Trace::Off,
        })
        .await
        .map_err(|e| JsValue::from_str(&format!("request_device: {e}")))?;
    // Out-of-band validation errors, collected and RETURNED: without a handler a refused object
    // silently draws nothing, and a blank replay says nothing about why (the native replay and
    // the live page both install one for the same reason).
    static ERRORS: std::sync::Mutex<Vec<String>> = std::sync::Mutex::new(Vec::new());
    ERRORS.lock().unwrap().clear();
    device.on_uncaptured_error(Arc::new(|e| {
        let msg = e.to_string();
        vitaslop_platform::gpu::note_device_error("wgpu", &msg);
        let mut v = ERRORS.lock().unwrap();
        if v.len() < 20 {
            v.push(msg.chars().take(600).collect());
        }
    }));
    vitaslop_platform::gpu::set_wasm_clock(crate::perf_now);
    let mut gxm = GxmRenderer::new(&device, &queue, OUTPUT_FORMAT);
    let mut builder = RenderSceneBuilder::new();
    let (width, height) = (fc.width, fc.height);
    // Passes completed at their own end-scene already hold their image and are not rendered
    // again - the rule the native replay and the live loop both follow.
    // `VITASLOP_REPLAY_EARLY=1`: render those too - a capsule is the only record of a pass that
    // completed early (the boot light-cell bake on MLB), and replaying it is the question.
    let early = vitaslop_runtime::knobs::var("VITASLOP_REPLAY_EARLY").is_ok_and(|v| v.trim() == "1");
    let scenes: Vec<_> = fc.scenes.iter().filter(|s| early || !s.completed_early).cloned().collect();

    let out = js_sys::Array::new();
    let mut first = true;
    let mut gpu_seq = gxm.ts_latest().map_or(0, |t| t.1);
    for &limit in &scene_limits {
        let n = if limit == 0 || limit as usize > scenes.len() { scenes.len() } else { limit as usize };
        let mut chosen: Vec<_> = scenes[..n].to_vec();
        if draw_limit >= 0 {
            if let Some(last) = chosen.last_mut() {
                last.draws.truncate(draw_limit as usize);
            }
        }
        let t0 = crate::perf_now();
        let color_tex = device.create_texture(&wgpu::TextureDescriptor {
            label: Some("replay colour"),
            size: wgpu::Extent3d { width, height, depth_or_array_layers: 1 },
            mip_level_count: 1,
            sample_count: 1,
            dimension: wgpu::TextureDimension::D2,
            format: OUTPUT_FORMAT,
            usage: wgpu::TextureUsages::RENDER_ATTACHMENT | wgpu::TextureUsages::COPY_SRC,
            view_formats: &[],
        });
        let color_view = color_tex.create_view(&Default::default());
        let depth_tex = device.create_texture(&wgpu::TextureDescriptor {
            label: Some("replay depth"),
            size: wgpu::Extent3d { width, height, depth_or_array_layers: 1 },
            mip_level_count: 1,
            sample_count: 1,
            dimension: wgpu::TextureDimension::D2,
            format: vitaslop_platform::gpu::depth_format(),
            usage: wgpu::TextureUsages::RENDER_ATTACHMENT,
            view_formats: &[],
        });
        let depth_view = depth_tex.create_view(&Default::default());
        builder.begin_frame();
        let built: Vec<_> = chosen.iter().map(|s| builder.build(s)).collect();
        let mut encoder = device.create_command_encoder(&wgpu::CommandEncoderDescriptor { label: None });
        gxm.encode_chain(
            &device,
            &queue,
            &mut encoder,
            &color_view,
            &depth_view,
            &built,
            width,
            height,
            width,
            height,
            fc.clear,
            Some(&color_tex),
        );
        // The pass timestamps ride this submit - see `GpuTimestamps`. Their SUM over a frame is
        // the device's own GPU time for it, which a wall clock around the readback cannot give:
        // that also carries the build, the encode and the event loop.
        gxm.ts_finish_chain(&mut encoder);
        // `VITASLOP_REPLAY_TARGET=<hex guest address>`: hand back that OFFSCREEN target (as the
        // renderer holds it, RGBA8) instead of the display - how a pass whose output only
        // another pass samples is compared across devices. Madden's U4U4U4U4 crowd atlas: the
        // crowd program discards under alpha 0.75 and the phone drew no crowd from identical
        // inputs, while the same pass rendered AS the display matched the desktop.
        let want_target = vitaslop_runtime::knobs::var("VITASLOP_REPLAY_TARGET")
            .ok()
            .and_then(|v| u32::from_str_radix(v.trim().trim_start_matches("0x"), 16).ok());
        let picked = want_target.and_then(|a| gxm.rtt_targets().into_iter().find(|t| t.0 == a).map(|t| (t.1.clone(), t.2, t.3)));
        let (src_tex, width, height) = match (&picked, want_target) {
            (Some((t, w, h)), _) => (t, *w, *h),
            (None, Some(a)) => {
                let four: Vec<String> = gxm.rtt_targets().iter().map(|t| format!("{:#x} {}x{}", t.0, t.2, t.3)).collect();
                let float: Vec<String> = gxm.rtt_float_targets().iter().map(|t| format!("{:#x} {}x{}", t.0, t.2, t.3)).collect();
                return Err(JsValue::from_str(&format!(
                    "frame replay: no 4-byte render target at {a:#x} after this prefix; 4-byte targets [{}], Rgba16Float targets [{}]",
                    four.join(", "),
                    float.join(", ")
                )));
            }
            (None, None) => (&color_tex, width, height),
        };
        let bpr = (width * 4).div_ceil(wgpu::COPY_BYTES_PER_ROW_ALIGNMENT) * wgpu::COPY_BYTES_PER_ROW_ALIGNMENT;
        let readback = device.create_buffer(&wgpu::BufferDescriptor {
            label: Some("replay readback"),
            size: (bpr * height) as u64,
            usage: wgpu::BufferUsages::MAP_READ | wgpu::BufferUsages::COPY_DST,
            mapped_at_creation: false,
        });
        encoder.copy_texture_to_buffer(
            wgpu::TexelCopyTextureInfo { texture: src_tex, mip_level: 0, origin: wgpu::Origin3d::ZERO, aspect: wgpu::TextureAspect::All },
            wgpu::TexelCopyBufferInfo {
                buffer: &readback,
                layout: wgpu::TexelCopyBufferLayout { offset: 0, bytes_per_row: Some(bpr), rows_per_image: Some(height) },
            },
            wgpu::Extent3d { width, height, depth_or_array_layers: 1 },
        );
        queue.submit([encoder.finish()]);
        gxm.ts_map_after_submit();
        // A flag, not a JS promise, so the callback holds nothing `!Send` - this file also
        // builds into the wasm-threads bundle.
        let ready = Arc::new(AtomicBool::new(false));
        let r2 = ready.clone();
        readback.slice(..).map_async(wgpu::MapMode::Read, move |_| r2.store(true, Ordering::Release));
        let mut waited = 0;
        while !ready.load(Ordering::Acquire) {
            crate::sleep_ms(2).await;
            waited += 2;
            if waited > 60_000 {
                return Err(JsValue::from_str("frame replay: the readback never landed (60 s)"));
            }
        }
        let mut rgba = vec![0u8; (width * height * 4) as usize];
        {
            let view = readback
                .slice(..)
                .get_mapped_range()
                .map_err(|e| JsValue::from_str(&format!("frame replay: map the readback: {e:?}")))?;
            for y in 0..height as usize {
                let src = &view[y * bpr as usize..y * bpr as usize + width as usize * 4];
                rgba[y * width as usize * 4..(y + 1) * width as usize * 4].copy_from_slice(src);
            }
        }
        readback.unmap();
        // The timestamp map was asked for after the same submit; give it a moment to land.
        for _ in 0..50 {
            gxm.ts_poll();
            if gxm.ts_latest().map_or(0, |t| t.1) != gpu_seq {
                break;
            }
            crate::sleep_ms(2).await;
        }
        gpu_seq = gxm.ts_latest().map_or(0, |t| t.1);
        // The first render also warms every pipeline; its GPU time is kept out of the report.
        if first {
            let _ = gxm.take_gpu_time_report();
            first = false;
        }
        let o = js_sys::Object::new();
        let set = |k: &str, v: JsValue| js_sys::Reflect::set(&o, &JsValue::from_str(k), &v);
        set("limit", JsValue::from(n as u32))?;
        set("passes", JsValue::from(scenes.len() as u32))?;
        set("draws", JsValue::from(chosen.last().map_or(0, |s| s.draws.len()) as u32))?;
        set("width", JsValue::from(width))?;
        set("height", JsValue::from(height))?;
        set("ms", JsValue::from(crate::perf_now() - t0))?;
        set("rgba", js_sys::Uint8Array::from(&rgba[..]).into())?;
        let errs = js_sys::Array::new();
        for e in ERRORS.lock().unwrap().iter() {
            errs.push(&JsValue::from_str(e));
        }
        set("errors", errs.into())?;
        out.push(&o);
    }
    // The GPU TIME report over every render after the first, on the LAST entry.
    if out.length() > 0 {
        let last = out.get(out.length() - 1);
        js_sys::Reflect::set(&last, &JsValue::from_str("gpu"), &JsValue::from_str(&gxm.take_gpu_time_report()))?;
    }
    Ok(out)
}
