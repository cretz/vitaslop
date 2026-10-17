//! Playbooks in the browser: the wasmoon Lua backend, and the run worker's half of a run.
//!
//! The VM is C Lua 5.4 compiled to wasm (wasmoon), living in the RUN WORKER's JavaScript as
//! `self.__playbookLua` (see `worker.js`), and reached through the two-function seam in
//! `vitaslop_playbook::vm`. A guest worker's pad read ticks the engine but cannot call the VM
//! (it is another worker's JavaScript), so the run worker resumes it here once per present -
//! a request that finishes on input frame N has its successor's input on the first input
//! frame after that. The browser is the real-speed engine, so nothing exact is lost.
use vitaslop_playbook::engine::{self, Event, StartOpts, Status};
use vitaslop_playbook::vm::{self, LuaVm};
use wasm_bindgen::prelude::*;

#[wasm_bindgen]
extern "C" {
    #[wasm_bindgen(js_namespace = __playbookLua, js_name = exec, catch)]
    fn lua_exec(name: &str, source: &str) -> Result<(), JsValue>;
    #[wasm_bindgen(js_namespace = __playbookLua, js_name = call, catch)]
    fn lua_call(func: &str, arg: &str) -> Result<String, JsValue>;
}

/// The run worker's wasmoon state. Holds nothing: the state lives in JavaScript.
struct WebLua;

fn js_error(e: JsValue) -> String {
    e.as_string().unwrap_or_else(|| format!("{e:?}"))
}

impl LuaVm for WebLua {
    fn exec(&mut self, name: &str, source: &str) -> Result<(), String> {
        lua_exec(name, source).map_err(js_error)
    }

    fn call(&mut self, func: &str, arg: &str) -> Result<String, String> {
        lua_call(func, arg).map_err(js_error)
    }
}

fn console(line: &str) {
    web_sys::console::log_1(&JsValue::from_str(line));
}

fn wall_ms() -> f64 {
    js_sys::Date::now()
}

thread_local! {
    static FINISHED: std::cell::Cell<bool> = const { std::cell::Cell::new(false) };
}

/// Start the playbook `{ name, source }` the page sent, if it sent one. Before the guest runs.
pub fn start(playbook: &JsValue) -> Result<bool, JsValue> {
    if playbook.is_undefined() || playbook.is_null() {
        return Ok(false);
    }
    let field = |k: &str| {
        js_sys::Reflect::get(playbook, &JsValue::from_str(k))
            .ok()
            .and_then(|v| v.as_string())
            .ok_or_else(|| JsValue::from_str(&format!("playbook: the start message has no `{k}` string")))
    };
    let (name, source) = (field("name")?, field("source")?);
    vm::load(&mut WebLua, &name, &source).map_err(|e| JsValue::from_str(&format!("playbook {name}: {e}")))?;
    let smp = vitaslop_runtime::knobs::flag("VITASLOP_SMP");
    let opts = StartOpts {
        name: name.clone(),
        header: vec![("mode".into(), if smp { "real-speed (browser, SMP)" } else { "browser (one-baton)" }.into())],
        // Discovery is a desktop tool (`--discover`): it finds cues, which then run anywhere.
        discover: false,
        wall_ms: Some(wall_ms),
        drives_input: true,
    };
    engine::start(opts, &source, None).map_err(|e| JsValue::from_str(&format!("playbook {name}: {e}")))?;
    FINISHED.with(|f| f.set(false));
    console(&format!("[playbook] {name} started"));
    Ok(true)
}

/// The run worker's per-present step: enforce timeouts, resume the VM, surface its output, and
/// print the report and the tape once the playbook is over. Shots are NAMED on the console
/// (`[playbook-shot] <name> at frame <n>`): a worker has no filesystem, so the picture is the
/// harness's to take.
pub fn on_present(frame: u64, guest_us: u64) {
    if FINISHED.with(|f| f.get()) || !engine::active() {
        return;
    }
    engine::on_display_frame(frame, guest_us);
    engine::pump_external(&mut WebLua);
    for ev in engine::take_events() {
        match ev {
            Event::Log(line) => console(&line),
            Event::Shot(name) => console(&format!("[playbook-shot] {name} at frame {frame}")),
        }
    }
    if engine::status() != Some(Status::Running) {
        FINISHED.with(|f| f.set(true));
        if let Some(report) = engine::finish() {
            for line in report.render_text().lines() {
                console(&format!("[playbook-report] {line}"));
            }
            for line in report.tape.render().lines() {
                console(&format!("[playbook-tape] {line}"));
            }
        }
    }
}
