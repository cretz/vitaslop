//! The page's view of the shared front-end logic: settings, input maps, and the
//! streaming title import. Thin wasm-bindgen wrappers over `vitaslop_frontend` and
//! `vitaslop_runtime::ingest::stream`; the rules live there, not here.
//!
//! # The import crosses the seam as CALLBACKS, not bytes
//! A retail pkg is up to 3.3 GB. The page cannot hand it over: this heap is the
//! 4 GB wasm32 one and the emulator will need most of it. So the page passes an
//! object that READS ranges on demand (`FileReaderSync` over the picked `File`s,
//! in a worker) and one that WRITES the decrypted tree as it is produced (OPFS sync
//! access handles, same worker). The Rust side pulls a megabyte at a time through
//! the first and pushes through the second. Peak wasm heap is a few chunks plus the
//! largest executable.

use std::collections::BTreeMap;
use std::rc::Rc;

use js_sys::{Function, Reflect, Uint8Array};
use vitaslop_frontend::settings::{self, Settings};
use vitaslop_runtime::ingest::stream::{self, ByteSource, DumpSink};
use vitaslop_runtime::ingest::Error;
use wasm_bindgen::prelude::*;

fn js_err(e: JsValue) -> Error {
    Error::Io(e.as_string().unwrap_or_else(|| format!("{e:?}")))
}

fn call(obj: &JsValue, name: &str, args: &[JsValue]) -> Result<JsValue, Error> {
    let f: Function = Reflect::get(obj, &JsValue::from_str(name))
        .map_err(js_err)?
        .dyn_into()
        .map_err(|_| Error::Io(format!("source/sink has no {name}()")))?;
    let arr = js_sys::Array::new();
    for a in args {
        arr.push(a);
    }
    Reflect::apply(&f, obj, &arr).map_err(js_err)
}

/// `{ list(): string[], size(path): number|undefined, readAt(path, off, buf: Uint8Array): number }`
struct JsSource(JsValue);

impl ByteSource for JsSource {
    fn list(&self) -> Vec<String> {
        match call(&self.0, "list", &[]) {
            Ok(v) => js_sys::Array::from(&v).iter().filter_map(|x| x.as_string()).collect(),
            Err(_) => Vec::new(),
        }
    }
    fn size(&self, path: &str) -> Option<u64> {
        call(&self.0, "size", &[JsValue::from_str(path)]).ok()?.as_f64().map(|f| f as u64)
    }
    fn read_at(&self, path: &str, off: u64, buf: &mut [u8]) -> Result<usize, Error> {
        // A fresh view for each call: the wasm memory may have grown (and moved) since
        // the last one, and a view is only valid until it does.
        let view = Uint8Array::new_with_length(buf.len() as u32);
        let n = call(&self.0, "readAt", &[JsValue::from_str(path), JsValue::from_f64(off as f64), view.clone().into()])?
            .as_f64()
            .ok_or_else(|| Error::Io("readAt returned no count".into()))? as usize;
        let n = n.min(buf.len());
        view.subarray(0, n as u32).copy_to(&mut buf[..n]);
        Ok(n)
    }
}

/// `{ begin(path, size), write(bytes: Uint8Array), finish() }`
struct JsSink(JsValue);

impl DumpSink for JsSink {
    fn begin(&mut self, path: &str, size: u64) -> Result<(), Error> {
        call(&self.0, "begin", &[JsValue::from_str(path), JsValue::from_f64(size as f64)]).map(|_| ())
    }
    fn write(&mut self, bytes: &[u8]) -> Result<(), Error> {
        // `Uint8Array::from` copies into a JS-owned buffer, so the sink may keep it.
        call(&self.0, "write", &[Uint8Array::from(bytes).into()]).map(|_| ())
    }
    fn finish(&mut self) -> Result<(), Error> {
        call(&self.0, "finish", &[]).map(|_| ())
    }
}

fn set(obj: &js_sys::Object, k: &str, v: JsValue) {
    let _ = Reflect::set(obj, &JsValue::from_str(k), &v);
}

fn opt_str(s: &Option<String>) -> JsValue {
    match s {
        Some(s) => JsValue::from_str(s),
        None => JsValue::NULL,
    }
}

fn opt_bytes(b: &Option<Vec<u8>>) -> JsValue {
    match b {
        Some(b) => Uint8Array::from(&b[..]).into(),
        None => JsValue::NULL,
    }
}

/// Identify what a set of picked files is, before importing. See `stream::probe`.
#[wasm_bindgen]
pub fn ingest_probe(source: JsValue) -> Result<JsValue, JsValue> {
    crate::logging::install_panic_hook();
    let src: Rc<dyn ByteSource> = Rc::new(JsSource(source));
    let p = stream::probe(src).map_err(|e| JsValue::from_str(&e.to_string()))?;
    let out = js_sys::Object::new();
    set(&out, "kind", JsValue::from_str(p.kind));
    set(&out, "zipped", JsValue::from_bool(p.zipped));
    set(&out, "titleId", opt_str(&p.title_id));
    set(&out, "title", opt_str(&p.title));
    set(&out, "contentId", opt_str(&p.content_id));
    set(&out, "appVersion", opt_str(&p.app_version));
    set(&out, "bytes", JsValue::from_f64(p.bytes as f64));
    set(&out, "files", JsValue::from_f64(p.files as f64));
    set(&out, "icon0", opt_bytes(&p.icon0));
    set(&out, "pic0", opt_bytes(&p.pic0));
    set(&out, "missingWorkBin", JsValue::from_bool(p.missing_work_bin));
    let outs = js_sys::Array::new();
    for o in &p.outputs {
        outs.push(&JsValue::from_str(o));
    }
    set(&out, "outputs", outs.into());
    Ok(out.into())
}

/// Stream a container into a dump tree. `progress(stage, file, done, total)` is
/// called per chunk. Returns the content id.
#[wasm_bindgen]
pub fn ingest_import(source: JsValue, sink: JsValue, progress: Function) -> Result<String, JsValue> {
    crate::logging::install_panic_hook();
    let src: Rc<dyn ByteSource> = Rc::new(JsSource(source));
    let mut sink = JsSink(sink);
    let mut report = |p: stream::Progress<'_>| {
        let _ = progress.call4(
            &JsValue::NULL,
            &JsValue::from_str(p.stage),
            &JsValue::from_str(p.file),
            &JsValue::from_f64(p.done as f64),
            &JsValue::from_f64(p.total as f64),
        );
    };
    stream::import(src, &mut sink, &mut report).map_err(|e| JsValue::from_str(&e.to_string()))
}

/// The default settings record, as JSON.
#[wasm_bindgen]
pub fn settings_defaults() -> String {
    Settings::default().to_value().to_string()
}

/// Defaults, then the global record, then the title patch - the settings a run uses.
#[wasm_bindgen]
pub fn settings_effective(global_json: &str, title_patch_json: Option<String>) -> Result<String, JsValue> {
    let global: serde_json::Value =
        serde_json::from_str(global_json).map_err(|e| JsValue::from_str(&format!("settings: {e}")))?;
    let patch: Option<serde_json::Value> = match title_patch_json {
        Some(s) if !s.trim().is_empty() => {
            Some(serde_json::from_str(&s).map_err(|e| JsValue::from_str(&format!("title settings: {e}")))?)
        }
        _ => None,
    };
    Ok(settings::effective(&global, patch.as_ref()).to_value().to_string())
}

/// The `VITASLOP_*` knobs an effective settings record configures a run with.
#[wasm_bindgen]
pub fn settings_run_knobs(effective_json: &str) -> Result<JsValue, JsValue> {
    let v: serde_json::Value =
        serde_json::from_str(effective_json).map_err(|e| JsValue::from_str(&format!("settings: {e}")))?;
    let knobs: BTreeMap<String, String> = Settings::from_value(&v).run_knobs();
    let out = js_sys::Object::new();
    for (k, v) in knobs {
        set(&out, &k, JsValue::from_str(&v));
    }
    Ok(out.into())
}

/// `NAME=VALUE` lines -> `{NAME: VALUE}`.
#[wasm_bindgen]
pub fn settings_parse_knobs(text: &str) -> JsValue {
    let out = js_sys::Object::new();
    for (k, v) in settings::parse_knobs(text) {
        set(&out, &k, JsValue::from_str(&v));
    }
    out.into()
}

/// The buttons a settings screen lists: `[{name, label, bit}]` in display order, plus
/// the Standard Gamepad control names by index.
#[wasm_bindgen]
pub fn input_vocabulary() -> JsValue {
    use vitaslop_frontend::input::{Button, GAMEPAD_CONTROLS};
    let buttons = js_sys::Array::new();
    for b in Button::ALL {
        let o = js_sys::Object::new();
        set(&o, "name", JsValue::from_str(b.name()));
        set(&o, "label", JsValue::from_str(b.label()));
        set(&o, "bit", JsValue::from_f64(b.bit() as f64));
        buttons.push(&o);
    }
    let controls = js_sys::Array::new();
    for c in GAMEPAD_CONTROLS {
        controls.push(&JsValue::from_str(c));
    }
    let out = js_sys::Object::new();
    set(&out, "buttons", buttons.into());
    set(&out, "gamepadControls", controls.into());
    out.into()
}
