//! Browser conformance runner. Runs the embedded ARM corpus through [`WebVm`] on
//! the browser's own WebAssembly engine, driving the exact same
//! `vitaslop_conformance_harness::run_all` + golden comparison the native
//! (wasmtime) test uses. A green run here proves the transpiler's output behaves
//! identically on both engines - the whole point of running the suite in-browser.
//!
//! One JS -> wasm call runs the *entire* corpus (transpile + instantiate + run +
//! compare per case, looped in Rust) and returns a JSON summary, so CI pays the
//! browser-launch/module-load cost once, not per case.

use std::cell::RefCell;
use std::rc::Rc;

use vitaslop_conformance_harness::{
    self as harness, CaseRun, Engine, LinuxSvc, BASE, MEM_BYTES, NORETURN_SVC,
};
use vitaslop_conformance_suite_arm as suite;
use vitaslop_transpiler::{transpile, Program};
use wasm_bindgen::prelude::*;

use crate::web_vm::WebVm;

/// The browser engine: transpile each case and run it on `WebVm`, servicing the
/// Linux-EABI `svc` convention via the shared [`LinuxSvc`] handler.
struct WebEngine;

impl Engine for WebEngine {
    fn run_case(
        &mut self,
        bin: &[u8],
        thumb: bool,
        in_regs: &[(usize, u32)],
    ) -> Result<CaseRun, String> {
        let artifact = transpile(&Program {
            code: bin,
            base: BASE,
            thumb,
            entries: &[BASE],
            externs: &[],
            noreturn_svc: NORETURN_SVC,
            mem_bytes: MEM_BYTES,
            // The ARM corpus is tightly controlled and takes no function
            // addresses; keep discovery off so output matches the native runner.
            discover_code_pointers: false,
            import_memory: false,
        })
        .map_err(|e| format!("transpile: {e:?}"))?;

        // Keep an Rc handle to read captured output back after the run.
        let svc = Rc::new(RefCell::new(LinuxSvc::default()));
        let vm = WebVm::new(
            &artifact.wasm,
            bin,
            BASE,
            MEM_BYTES,
            Some(Box::new(svc.clone())),
            None,
        )
        .map_err(|e| jsval_str(&e))?;

        for &(i, v) in in_regs {
            vm.set_reg(i, v);
        }
        vm.call(BASE).map_err(|e| jsval_str(&e))?;

        Ok(CaseRun {
            regs: vm.regs(),
            flags: vm.flags(),
            output: svc.borrow().output.clone(),
        })
    }
}

/// One case's result, JSON-serialized for the page and the Playwright assertion.
#[derive(serde::Serialize)]
struct CaseResult {
    name: String,
    pass: bool,
    detail: Option<String>,
}

/// The whole-corpus summary returned to JS.
#[derive(serde::Serialize)]
struct Summary {
    total: usize,
    passed: usize,
    cases: Vec<CaseResult>,
}

/// Run the entire embedded ARM corpus in the browser and return a JSON summary
/// `{ total, passed, cases: [{ name, pass, detail }] }`. Never throws for a case
/// failure - a failure is reported in the summary so the page can show every one.
#[wasm_bindgen]
pub fn run_conformance() -> Result<String, JsValue> {
    console_error_panic_hook::set_once();
    let cases = suite::embedded_cases().map_err(|e| JsValue::from_str(&e))?;
    let outcomes = harness::run_all(&mut WebEngine, &cases);
    let passed = outcomes.iter().filter(|o| o.pass).count();
    let summary = Summary {
        total: outcomes.len(),
        passed,
        cases: outcomes
            .into_iter()
            .map(|o| CaseResult { name: o.name, pass: o.pass, detail: o.detail })
            .collect(),
    };
    serde_json::to_string(&summary).map_err(|e| JsValue::from_str(&e.to_string()))
}

/// Render a `JsValue` error as a short string for a case-run failure message.
fn jsval_str(e: &JsValue) -> String {
    e.as_string().unwrap_or_else(|| format!("{e:?}"))
}
