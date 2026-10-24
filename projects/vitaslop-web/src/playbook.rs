//! Playbooks in the browser: the wasmoon Lua backend, and the run worker's half of a run.
//!
//! The VM is C Lua 5.4 compiled to wasm (wasmoon), living in the RUN WORKER's JavaScript as
//! `self.__playbookLua` (see `worker.js`), and reached through the two-function seam in
//! `vitaslop_playbook::vm`. A guest worker's pad read ticks the engine but cannot call the VM
//! (it is another worker's JavaScript), so the run worker resumes it here once per present -
//! a request that finishes on input frame N has its successor's input on the first input
//! frame after that. The browser is the real-speed engine, so nothing exact is lost.
//!
//! >>> ONLY IN THE `playbook` BUILD. The shipped bundles are built without the feature, so the
//! engine, its exports and its per-present work are not in them; [`off`] stands in with the same
//! functions doing nothing, and a run handed a playbook fails at start instead of running unscripted.
//! The e2e harness and the device runner serve the playbook build (`node build.mjs --playbook`).
#[cfg(not(feature = "playbook"))]
pub use off::*;

#[cfg(not(feature = "playbook"))]
mod off {
    use vitaslop_runtime::{CtrlFrame, TouchFrame};
    use wasm_bindgen::JsValue;

    pub fn input(_port: u32) -> Option<(CtrlFrame, TouchFrame)> {
        None
    }
    pub fn wants_pixels() -> bool {
        false
    }
    pub fn on_frame_pixels(_w: u32, _h: u32, _rgba: &[u8]) {}
    pub fn take_shot_request() -> bool {
        false
    }
    pub fn start(playbook: &JsValue) -> Result<bool, JsValue> {
        if playbook.is_undefined() || playbook.is_null() {
            return Ok(false);
        }
        Err(JsValue::from_str(
            "playbook: this bundle was built without playbooks - serve the playbook build (`node build.mjs --playbook`)",
        ))
    }
    pub fn on_present(_frame: u64, _guest_us: u64) {}
    pub fn finish_now() {}
}

#[cfg(feature = "playbook")]
pub use on::*;

#[cfg(feature = "playbook")]
mod on {
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

    pub use vitaslop_playbook::engine::{input, on_frame_pixels, wants_pixels};

    fn console(line: &str) {
        web_sys::console::log_1(&JsValue::from_str(line));
    }

    fn wall_ms() -> f64 {
        js_sys::Date::now()
    }

    thread_local! {
        static FINISHED: std::cell::Cell<bool> = const { std::cell::Cell::new(false) };
        /// Shots named since the renderer last looked - see [`take_shot_request`].
        static SHOT_ASKED: std::cell::Cell<bool> = const { std::cell::Cell::new(false) };
    }

    /// Whether the playbook named a shot since the last call. The renderer's present probe (when
    /// one exists: a device-runner job asks for it) then copies the next present, so a phone run
    /// brings the named picture back as a `presentshot` line - a phone has no harness to take it.
    pub fn take_shot_request() -> bool {
        SHOT_ASKED.with(|s| s.replace(false))
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
                .ok_or_else(|| {
                    JsValue::from_str(&format!("playbook: the start message has no `{k}` string"))
                })
        };
        let (name, source) = (field("name")?, field("source")?);
        vm::load(&mut WebLua, &name, &source)
            .map_err(|e| JsValue::from_str(&format!("playbook {name}: {e}")))?;
        let smp = vitaslop_runtime::knobs::flag("VITASLOP_SMP");
        let opts = StartOpts {
            name: name.clone(),
            header: vec![(
                "mode".into(),
                if smp {
                    "real-speed (browser, SMP)"
                } else {
                    "browser (one-baton)"
                }
                .into(),
            )],
            // Discovery is a desktop tool (`--discover`): it finds cues, which then run anywhere.
            discover: false,
            wall_ms: Some(wall_ms),
            drives_input: true,
        };
        engine::start(opts, &source, None)
            .map_err(|e| JsValue::from_str(&format!("playbook {name}: {e}")))?;
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
                Event::Shot(name) => {
                    console(&format!("[playbook-shot] {name} at frame {frame}"));
                    SHOT_ASKED.with(|s| s.set(true));
                }
            }
        }
        if engine::status() != Some(Status::Running) {
            finish_now();
        }
    }

    /// Print the playbook's report and tape, once. The per-present step calls it when the playbook is
    /// over, and the run's end calls it too (the frame budget, the guest exiting or faulting, a
    /// harness stopping the run): a playbook still running there is reported as STOPPED with
    /// everything it did up to that point, instead of the run ending with no report and no tape.
    pub fn finish_now() {
        if FINISHED.with(|f| f.replace(true)) {
            return;
        }
        if let Some(report) = engine::finish() {
            for line in report.render_text().lines() {
                console(&format!("[playbook-report] {line}"));
            }
            for line in report.tape.render().lines() {
                console(&format!("[playbook-tape] {line}"));
            }
        }
    }

    /// [`finish_now`] for a harness that stops a run itself (the device runner's time cap), called
    /// from the run worker before the result is collected. A run with no playbook prints nothing.
    #[wasm_bindgen]
    pub fn playbook_finish() {
        finish_now();
    }
}
