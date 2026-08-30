//! The browser: WebCodecs `VideoDecoder`.
//!
//! WebCodecs is the only way a page reaches the machine's video decoder; everything else
//! (a `<video>` element, MSE) decodes into a compositor surface a program cannot read back
//! cheaply. It is also the one backend here that is genuinely ASYNCHRONOUS: frames arrive
//! on a callback, and the pixels behind a `VideoFrame` only become readable when a second
//! promise (`copyTo`) settles.
//!
//! That asynchrony is not hidden. [`crate::Decoder::receive`] returns what has already
//! landed, and [`crate::Decoder::receive_async`] waits - and it waits on a promise the
//! callbacks resolve, not on a timer. A worker's `setTimeout(0)` is clamped to 4 ms once
//! nested, so a polling loop would cap this decoder at 250 frames a second for no reason.
//!
//! The bindings are declared here by hand rather than taken from `web-sys`, whose WebCodecs
//! types sit behind `--cfg=web_sys_unstable_apis` - a flag every consumer of this crate
//! would then have to set in its own build.

use std::cell::RefCell;
use std::collections::VecDeque;
use std::rc::Rc;

use js_sys::{Function, Object, Promise, Reflect, Uint8Array};
use wasm_bindgen::prelude::*;
use wasm_bindgen_futures::{JsFuture, spawn_local};

use super::{Backend, FramePool, OutputOrder, StreamConfig};
use crate::bitstream::AccessUnit;
use crate::bitstream::avcc;
use crate::error::{Error, Result};
use crate::frame::{Frame, PixelFormat};

#[wasm_bindgen]
extern "C" {
    #[wasm_bindgen(js_name = VideoDecoder)]
    type JsVideoDecoder;

    #[wasm_bindgen(constructor, js_class = "VideoDecoder", catch)]
    fn new(init: &Object) -> std::result::Result<JsVideoDecoder, JsValue>;

    #[wasm_bindgen(method, js_class = "VideoDecoder", js_name = configure, catch)]
    fn configure(this: &JsVideoDecoder, config: &Object) -> std::result::Result<(), JsValue>;

    #[wasm_bindgen(method, js_class = "VideoDecoder", js_name = decode, catch)]
    fn decode(this: &JsVideoDecoder, chunk: &JsEncodedVideoChunk) -> std::result::Result<(), JsValue>;

    #[wasm_bindgen(method, js_class = "VideoDecoder", js_name = flush)]
    fn flush(this: &JsVideoDecoder) -> Promise;

    #[wasm_bindgen(method, js_class = "VideoDecoder", js_name = reset, catch)]
    fn reset(this: &JsVideoDecoder) -> std::result::Result<(), JsValue>;

    #[wasm_bindgen(method, js_class = "VideoDecoder", js_name = close, catch)]
    fn close(this: &JsVideoDecoder) -> std::result::Result<(), JsValue>;

    #[wasm_bindgen(method, getter, js_class = "VideoDecoder", js_name = decodeQueueSize)]
    fn decode_queue_size(this: &JsVideoDecoder) -> u32;

    #[wasm_bindgen(js_name = EncodedVideoChunk)]
    type JsEncodedVideoChunk;

    #[wasm_bindgen(constructor, js_class = "EncodedVideoChunk", catch)]
    fn new(init: &Object) -> std::result::Result<JsEncodedVideoChunk, JsValue>;

    #[wasm_bindgen(js_name = VideoFrame)]
    type JsVideoFrame;

    #[wasm_bindgen(method, getter, js_class = "VideoFrame", js_name = format)]
    fn format(this: &JsVideoFrame) -> Option<String>;

    #[wasm_bindgen(method, getter, js_class = "VideoFrame", js_name = timestamp)]
    fn timestamp(this: &JsVideoFrame) -> f64;

    #[wasm_bindgen(method, getter, js_class = "VideoFrame", js_name = visibleRect)]
    fn visible_rect(this: &JsVideoFrame) -> JsValue;

    #[wasm_bindgen(method, js_class = "VideoFrame", js_name = allocationSize, catch)]
    fn allocation_size(this: &JsVideoFrame, options: &Object) -> std::result::Result<u32, JsValue>;

    #[wasm_bindgen(method, js_class = "VideoFrame", js_name = copyTo)]
    fn copy_to(this: &JsVideoFrame, destination: &Uint8Array, options: &Object) -> Promise;

    #[wasm_bindgen(method, js_class = "VideoFrame", js_name = close)]
    fn close(this: &JsVideoFrame);
}

/// Shared between the backend and the JS callbacks it installs.
#[derive(Default)]
struct Shared {
    ready: VecDeque<Frame>,
    /// The first error the decoder reported. Sticky: WebCodecs closes itself on error.
    error: Option<String>,
    /// `copyTo` promises in flight, i.e. frames decoded but not yet readable.
    copies: usize,
    /// True between `flush()` and the promise it returns settling.
    flushing: bool,
    /// A resolver for whoever is awaiting the next frame.
    waker: Option<Function>,
    /// Buffers handed back for reuse.
    pool: Vec<Vec<u8>>,
    /// What the first decoded frame turned out to be: the format it called itself (which
    /// may be nothing at all) and the plane count the copy actually returned. It is the
    /// one fact about this backend that cannot be known before it runs, it differs between
    /// a software decode and a hardware one, and a run that renders nothing needs it on
    /// the record rather than inferred afterwards.
    layout: Option<String>,
}

impl Shared {
    /// Take the resolver for whoever is awaiting the next frame, to be CALLED once the
    /// cell is no longer borrowed - see [`wake`].
    fn take_waker(&mut self) -> Option<Function> {
        self.waker.take()
    }
}

/// Wake an awaiting `receive_async`, with the shared cell NOT borrowed.
///
/// # Why this is a free function and not a method
///
/// Resolving a promise hands control back to JavaScript, and in a browser every one of
/// this backend's own callbacks is JavaScript: the decoder's `output`, its `error`, and
/// the continuation of a `copyTo`. A `RefCell` guard held across any of them is a panic
/// waiting for the browser to schedule things in the order that exposes it - and a panic
/// in a wasm worker is not an exception, it is the whole engine gone. So the borrow is
/// taken, the resolver is removed from it, the guard is DROPPED, and only then is JS
/// called. Every borrow in this file is scoped that way for the same reason.
fn wake(shared: &Rc<RefCell<Shared>>) {
    let waker = shared.borrow_mut().take_waker();
    if let Some(resolve) = waker {
        let _ = resolve.call0(&JsValue::NULL);
    }
}

/// The WebCodecs backend.
pub struct WebCodecsBackend {
    decoder: JsVideoDecoder,
    shared: Rc<RefCell<Shared>>,
    /// Kept alive for as long as the decoder: dropping a closure detaches its JS function.
    _on_output: Closure<dyn FnMut(JsVideoFrame)>,
    _on_error: Closure<dyn FnMut(JsValue)>,
    /// What to ask WebCodecs for on the next `configure`.
    hardware: bool,
    /// See [`crate::DecoderConfig::low_latency`] - routed to the SOFTWARE decoder here.
    low_latency: bool,
    /// Access units are converted from Annex B to the length-prefixed form the avcC
    /// description implies. Reused across calls.
    scratch: Vec<u8>,
    /// The configuration last given to the decoder, kept so [`Backend::reset`] can put it
    /// back - see there for why it has to.
    config: Option<Object>,
}

// SAFETY: this holds JS values, which wasm-bindgen marks `!Send` because they live in a
// per-thread heap no other thread can reach. On `wasm32-unknown-unknown` without the
// atomics feature there IS no other thread - the whole emulator, including its scheduler,
// runs in one worker - so the values are never reachable from anywhere else. This is the
// same reasoning the browser storage layer records for its own handles.
unsafe impl Send for WebCodecsBackend {}

impl WebCodecsBackend {
    /// Bind a `VideoDecoder`, or report that this browser has none.
    ///
    /// `hardware` becomes WebCodecs' own `hardwareAcceleration` hint on the configuration.
    /// `low_latency` overrides it towards the SOFTWARE decoder - see `configure` for why
    /// the hint alone does not deliver the fixed-function contract.
    pub fn new(hardware: bool, low_latency: bool) -> Result<WebCodecsBackend> {
        let global = js_sys::global();
        let ctor = Reflect::get(&global, &JsValue::from_str("VideoDecoder"))
            .map_err(|_| Error::no_decoder("no global scope to look up VideoDecoder in"))?;
        if ctor.is_undefined() || ctor.is_null() {
            return Err(Error::no_decoder(
                "this browser has no WebCodecs VideoDecoder (Safari < 16.4, Firefox < 130)",
            ));
        }

        let shared = Rc::new(RefCell::new(Shared::default()));

        let out_shared = shared.clone();
        let on_output = Closure::<dyn FnMut(JsVideoFrame)>::new(move |frame: JsVideoFrame| {
            deliver_frame(frame, out_shared.clone());
        });
        let err_shared = shared.clone();
        let on_error = Closure::<dyn FnMut(JsValue)>::new(move |e: JsValue| {
            {
                let mut s = err_shared.borrow_mut();
                if s.error.is_none() {
                    s.error = Some(describe(&e));
                }
            }
            wake(&err_shared);
        });

        let init = Object::new();
        set(&init, "output", on_output.as_ref())?;
        set(&init, "error", on_error.as_ref())?;
        let decoder = JsVideoDecoder::new(&init)
            .map_err(|e| Error::no_decoder(format!("new VideoDecoder failed: {}", describe(&e))))?;

        Ok(WebCodecsBackend {
            decoder,
            shared,
            _on_output: on_output,
            _on_error: on_error,
            hardware,
            low_latency,
            scratch: Vec::new(),
            config: None,
        })
    }

    /// True while the decoder still owes frames: input queued, copies in flight, or a
    /// flush outstanding. [`crate::Decoder::receive_async`] uses this to know whether
    /// waiting can possibly help.
    pub fn work_outstanding(&self) -> bool {
        let busy = {
            let s = self.shared.borrow();
            s.copies > 0 || s.flushing
        };
        busy || self.decoder.decode_queue_size() > 0
    }

    /// A promise that settles the next time a frame lands or an error is reported.
    pub fn next_event(&self) -> Promise {
        let shared = self.shared.clone();
        Promise::new(&mut |resolve, _reject| {
            shared.borrow_mut().waker = Some(resolve);
        })
    }

    /// Move buffers the caller recycled into the shared pool the callbacks allocate from.
    fn refill(&self, pool: &mut FramePool) {
        let mut shared = self.shared.borrow_mut();
        while shared.pool.len() < 4 {
            let buf = pool.take();
            if buf.capacity() == 0 {
                break;
            }
            shared.pool.push(buf);
        }
    }

    fn take_ready(&self, out: &mut Vec<Frame>) -> Result<()> {
        let mut shared = self.shared.borrow_mut();
        if let Some(e) = shared.error.take() {
            return Err(Error::Platform { call: "VideoDecoder", code: 0, detail: e });
        }
        out.extend(shared.ready.drain(..));
        Ok(())
    }
}

impl Backend for WebCodecsBackend {
    fn name(&self) -> &'static str {
        "WebCodecs"
    }

    fn output_order(&self) -> OutputOrder {
        // WebCodecs guarantees presentation order out of the decoder.
        OutputOrder::Presentation
    }

    fn detail(&self) -> Option<String> {
        self.shared.borrow().layout.clone()
    }

    fn configure(&mut self, config: StreamConfig<'_>) -> Result<()> {
        let avcc = config.avcc;
        // `avc1.PPCCLL`: profile, constraint byte, level, as the codec registry spells it.
        let codec = format!(
            "avc1.{:02X}{:02X}{:02X}",
            avcc.profile_idc, avcc.profile_compat, avcc.level_idc
        );
        let description = Uint8Array::from(avcc.to_bytes().as_slice());

        let cfg = Object::new();
        set(&cfg, "codec", &JsValue::from_str(&codec))?;
        set(&cfg, "description", &description)?;
        set(&cfg, "codedWidth", &JsValue::from_f64(config.sps.coded_width() as f64))?;
        set(&cfg, "codedHeight", &JsValue::from_f64(config.sps.coded_height() as f64))?;
        // Without this the browser buffers several frames before emitting any, which for
        // an emulator or a player showing frames as they arrive is pure added latency.
        set(&cfg, "optimizeForLatency", &JsValue::TRUE)?;
        // A hint, not a demand: "prefer-hardware" still falls back rather than failing,
        // which is why `acceleration` never claims hardware from this alone.
        //
        // >>> LOW LATENCY PICKS THE SOFTWARE DECODER, AND `optimizeForLatency` ALONE IS NOT
        // >>> ENOUGH. Chrome's hardware path (D3D11 on Windows, MediaCodec on Android) keeps
        // a several-frame reorder pipeline whatever the hint says. MEASURED on the shipping
        // page: a title whose guest decode API is a fixed-function block - submit one access
        // unit, poll for ITS picture, recycle the ES buffer only when the picture returns -
        // submitted 9 units and got 2 pictures back, hardware holding the rest. The title's
        // own pool then filled with unrecycled buffers (`sceClibMspaceMemalign: 8 REFUSED` on
        // a phone), it stopped pulling units, and the movie froze - and the divergent flow
        // failed the recipe's determinism signature downstream. Chrome's software decoder
        // honours `optimizeForLatency` with a pipeline depth of ~1, and a console-sized
        // picture (<= 960x544) is a few milliseconds in software on any target this runs on.
        let preference = if self.low_latency || !self.hardware {
            "prefer-software"
        } else {
            "no-preference"
        };
        set(&cfg, "hardwareAcceleration", &JsValue::from_str(preference))?;
        self.decoder
            .configure(&cfg)
            .map_err(|e| Error::platform("VideoDecoder.configure", 0, describe(&e)))?;
        self.config = Some(cfg);
        // The visible size is not kept here on purpose: a decoded `VideoFrame` carries its
        // own `visibleRect`, which is the browser's reading of the same SPS cropping, and
        // cropping to a second copy of that number is how the two drift apart.
        Ok(())
    }

    fn send(&mut self, au: &AccessUnit, timestamp: i64) -> Result<()> {
        // The description above is an avcC record, so chunks must be length-prefixed.
        self.scratch.clear();
        avcc::annex_b_to_length_prefixed(&au.data, 4, &mut self.scratch);

        let init = Object::new();
        set(&init, "type", &JsValue::from_str(if au.idr { "key" } else { "delta" }))?;
        // WebCodecs timestamps are microseconds; the key is scaled so two pictures can
        // never collide after the browser rounds.
        set(&init, "timestamp", &JsValue::from_f64((timestamp * 1000) as f64))?;
        set(&init, "data", &Uint8Array::from(self.scratch.as_slice()))?;
        let chunk = JsEncodedVideoChunk::new(&init)
            .map_err(|e| Error::platform("new EncodedVideoChunk", 0, describe(&e)))?;
        self.decoder
            .decode(&chunk)
            .map_err(|e| Error::platform("VideoDecoder.decode", 0, describe(&e)))
    }

    fn poll(&mut self, pool: &mut FramePool, out: &mut Vec<Frame>) -> Result<()> {
        self.refill(pool);
        self.take_ready(out)
    }

    fn drain(&mut self, pool: &mut FramePool, out: &mut Vec<Frame>) -> Result<()> {
        self.refill(pool);
        // The guard is dropped BEFORE `take_ready`, which borrows the same cell: an early
        // return with it still alive is a double borrow, i.e. a panic on the ordinary
        // "drain called twice" path.
        let already_flushing = {
            let mut shared = self.shared.borrow_mut();
            let was = shared.flushing;
            shared.flushing = true;
            was
        };
        if already_flushing {
            return self.take_ready(out);
        }
        let shared = self.shared.clone();
        let promise = self.decoder.flush();
        spawn_local(async move {
            let result = JsFuture::from(promise).await;
            {
                let mut s = shared.borrow_mut();
                s.flushing = false;
                if let Err(e) = result {
                    // An AbortError here is the ordinary consequence of `reset()` racing a
                    // flush, not a decode failure - it is reported rather than swallowed,
                    // but it does not poison the decoder.
                    if s.error.is_none() {
                        s.error = Some(describe(&e));
                    }
                }
            }
            wake(&shared);
        });
        self.take_ready(out)
    }

    fn pending_event(&self) -> Option<Promise> {
        if !self.work_outstanding() && self.shared.borrow().ready.is_empty() {
            return None;
        }
        Some(self.next_event())
    }

    fn reset(&mut self) -> Result<()> {
        self.decoder
            .reset()
            .map_err(|e| Error::platform("VideoDecoder.reset", 0, describe(&e)))?;
        // >>> WebCodecs' `reset()` ALSO UNCONFIGURES THE DECODER, and the other backends'
        // do not.
        //
        // The common layer configures once, on the first access unit that carries parameter
        // sets, and every other backend keeps that configuration across a flush. This one
        // does not: the next `decode` throws "Cannot call 'decode' on an unconfigured
        // codec", so a stream that is reset - a seek, a movie looping, a teardown - never
        // decodes again. Found by running this crate's own conformance suite in a browser,
        // which is the only place it can be found.
        if let Some(cfg) = self.config.as_ref() {
            self.decoder
                .configure(cfg)
                .map_err(|e| Error::platform("VideoDecoder.configure after reset", 0, describe(&e)))?;
        }
        {
            let mut shared = self.shared.borrow_mut();
            shared.ready.clear();
            shared.error = None;
            shared.flushing = false;
        }
        wake(&self.shared);
        Ok(())
    }
}

impl Drop for WebCodecsBackend {
    fn drop(&mut self) {
        // A VideoDecoder left open holds the platform decoder and, on some browsers, a GPU
        // context. Closing an already-closed decoder throws, which is why the error is
        // dropped rather than reported.
        let _ = self.decoder.close();
    }
}

/// Read a decoded `VideoFrame` into a CPU buffer and queue it.
///
/// The copy is asynchronous, so the frame is closed only once it completes. A `VideoFrame`
/// that is never closed pins a decoder slot, and a decoder out of slots simply stops
/// producing - which looks exactly like a stall, so this path never returns early without
/// closing.
fn deliver_frame(frame: JsVideoFrame, shared: Rc<RefCell<Shared>>) {
    // >>> THE COUNTER BUMP CANNOT ASSUME THE CELL IS FREE, and that is not defensive
    // programming - it is what this callback is.
    //
    // `output` is called by the browser, and the browser calls it whenever it likes:
    // measured here, from inside a `VideoFrame.close()` that freed the decoder slot the
    // next picture needed, while this file's own code was part-way through the shared
    // state. A `borrow_mut()` that panics there does not report a bug - it takes the whole
    // worker down, which is how the first ever run of this backend ended. So a busy cell
    // defers the same work to a microtask instead, where nothing of ours is running.
    let Ok(mut s) = shared.try_borrow_mut() else {
        let deferred = shared.clone();
        spawn_local(async move {
            deliver_frame(frame, deferred);
        });
        return;
    };
    s.copies += 1;
    drop(s);
    spawn_local(async move {
        let result = read_frame(&frame, &shared).await;
        frame.close();
        {
            let mut s = shared.borrow_mut();
            s.copies -= 1;
            match result {
                Ok(f) => s.ready.push_back(f),
                Err(e) => {
                    if s.error.is_none() {
                        s.error = Some(e.to_string());
                    }
                }
            }
        }
        wake(&shared);
    });
}

/// Copy one `VideoFrame` into a [`Frame`].
async fn read_frame(frame: &JsVideoFrame, shared: &Rc<RefCell<Shared>>) -> Result<Frame> {
    // >>> THE LAYOUT IS READ BACK, NOT REQUESTED AND NOT ASSUMED.
    //
    // This used to pass `format: "I420"` to `copyTo` so every browser handed back one
    // layout. MEASURED, on the first run this backend ever had: Chrome refuses it -
    // "copyTo() doesn't support explicit copy to non-RGB formats. Remove format parameter
    // to use VideoFrame's pixel format." Every copy failed, so the decoder produced
    // 562,120 access units' worth of nothing.
    //
    // The obvious repair - read `VideoFrame.format` and match on it - is not enough
    // either, because that attribute is NULLABLE: a frame a decoder hands over without
    // exposing its layout reports no format at all, which is exactly what a phone's
    // hardware decoder is entitled to do. So the format string is a HINT used only for
    // the report below; what decides the layout is the plane count `copyTo` itself
    // returns, which is a fact about the copy that just happened.
    let reported = frame.format().unwrap_or_default();

    let rect = frame.visible_rect();
    let (x, y, w, h) = rect_fields(&rect)?;

    // The visible rectangle, so cropping is the browser's reading of the stream's own SPS
    // rather than a second copy of that arithmetic here.
    let options = Object::new();
    let region = Object::new();
    set(&region, "x", &JsValue::from_f64(x))?;
    set(&region, "y", &JsValue::from_f64(y))?;
    set(&region, "width", &JsValue::from_f64(w))?;
    set(&region, "height", &JsValue::from_f64(h))?;
    set(&options, "rect", &region)?;

    let size = frame
        .allocation_size(&options)
        .map_err(|e| Error::platform("VideoFrame.allocationSize", 0, describe(&e)))?;

    let buffer = Uint8Array::new_with_length(size);
    let layout = JsFuture::from(frame.copy_to(&buffer, &options))
        .await
        .map_err(|e| Error::platform("VideoFrame.copyTo", 0, describe(&e)))?;

    // `copyTo` reports where it actually put each plane, and HOW MANY - which is what
    // names the layout. Three planes (or four, the fourth being alpha) is 4:2:0 in
    // separate planes; two is luma plus interleaved chroma.
    let planes = js_sys::Array::from(&layout);
    let pixel_format = match planes.length() {
        3 | 4 => PixelFormat::I420,
        2 => PixelFormat::Nv12,
        // >>> ONE PLANE IS RGBA, AND A DEVICE REALLY DOES THIS.
        //
        // No H.264 decoder produces RGB - the format is 4:2:0 by construction - so a single
        // packed plane means the browser converted before handing it over. MEASURED:
        // Chrome on an Android PowerVR device delivers `RGBA`, where the same browser on a
        // desktop delivers `I420`. Refusing it is refusing video on that device.
        1 if reported.starts_with("RGB") || reported.starts_with("BGR") => PixelFormat::Rgba,
        n => {
            return Err(Error::unsupported(format!(
                "VideoFrame.copyTo returned {n} plane(s) (the frame calls itself                  {reported:?}) - this decoder reads 4:2:0 in two or three planes, or one                  packed RGB plane"
            )));
        }
    };
    {
        let mut s = shared.borrow_mut();
        if s.layout.is_none() {
            let name = if reported.is_empty() {
                "no format of its own".to_string()
            } else {
                reported.clone()
            };
            let note = if planes.length() == 1 {
                " - PACKED RGB, so the decoder converted colour before handing it over and                  a caller that wants 4:2:0 converts it back"
            } else {
                ""
            };
            s.layout = Some(format!(
                "frames arrive as {name}, copied back as {} plane(s){note}",
                planes.length()
            ));
        }
    }

    let recycled = { shared.borrow_mut().pool.pop().unwrap_or_default() };
    let mut out = Frame::alloc(pixel_format, w as u32, h as u32, recycled);
    out.pts = (frame.timestamp() as i64) / 1000;
    let mut copied = vec![0u8; size as usize];
    buffer.copy_to(&mut copied);
    for i in 0..pixel_format.plane_count() {
        let entry = planes.get(i as u32);
        let offset = number(&entry, "offset")? as usize;
        let stride = number(&entry, "stride")? as usize;
        let dst = out.planes[i];
        let rows = dst.rows;
        if offset + stride * (rows - 1) + dst.row_bytes > copied.len() {
            return Err(Error::platform(
                "VideoFrame.copyTo",
                0,
                format!("plane {i} layout runs past the {size}-byte buffer"),
            ));
        }
        let plane = out.plane_mut(i);
        for r in 0..rows {
            let from = offset + r * stride;
            plane[r * dst.stride..r * dst.stride + dst.row_bytes]
                .copy_from_slice(&copied[from..from + dst.row_bytes]);
        }
        // A packed plane the browser calls BGR is the same bytes in the other order.
        // Everything downstream of this crate is told RGBA, so the swap happens once, here,
        // rather than becoming a second format every consumer has to know about.
        if pixel_format == PixelFormat::Rgba && reported.starts_with("BGR") {
            for px in plane.chunks_exact_mut(4) {
                px.swap(0, 2);
            }
        }
    }
    Ok(out)
}

/// Read `x`, `y`, `width`, `height` off a `DOMRectReadOnly`.
fn rect_fields(rect: &JsValue) -> Result<(f64, f64, f64, f64)> {
    Ok((number(rect, "x")?, number(rect, "y")?, number(rect, "width")?, number(rect, "height")?))
}

fn number(object: &JsValue, key: &str) -> Result<f64> {
    Reflect::get(object, &JsValue::from_str(key))
        .ok()
        .and_then(|v| v.as_f64())
        .ok_or_else(|| Error::platform("VideoFrame", 0, format!("no numeric `{key}`")))
}

fn set(object: &Object, key: &str, value: &JsValue) -> Result<()> {
    Reflect::set(object, &JsValue::from_str(key), value)
        .map(|_| ())
        .map_err(|e| Error::platform("Reflect::set", 0, describe(&e)))
}

/// Turn a thrown JS value into something a log line can carry.
fn describe(value: &JsValue) -> String {
    if let Some(s) = value.as_string() {
        return s;
    }
    let name = Reflect::get(value, &JsValue::from_str("name")).ok().and_then(|v| v.as_string());
    let message =
        Reflect::get(value, &JsValue::from_str("message")).ok().and_then(|v| v.as_string());
    match (name, message) {
        (Some(n), Some(m)) => format!("{n}: {m}"),
        (Some(n), None) => n,
        (None, Some(m)) => m,
        (None, None) => format!("{value:?}"),
    }
}
