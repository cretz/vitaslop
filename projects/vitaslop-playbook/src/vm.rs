//! The Lua VM seam.
//!
//! >>> A BACKEND IS TWO STRING FUNCTIONS. The prelude (`prelude.lua`) turns every playbook API
//! call into a yielded JSON request and every reply into a resume, so the engine never needs a
//! callback registered inside the VM, a userdata, or the VM's coroutine API. What a Lua
//! implementation has to supply is: run a chunk under a name, and call a global function with
//! one string argument and take its string result. That is the whole contract, on purpose -
//! the user expects to swap implementations, and a seam this narrow is one a new VM can fill
//! in an afternoon.
//!
//! Backends: [`crate::lua_mlua`] (native, PUC-Rio C Lua 5.4 through `mlua`) and the browser's
//! wasmoon backend in `vitaslop-web` (the same C Lua 5.4 compiled to wasm). Both run the
//! reference implementation, so a playbook means the same thing in both engines.

/// One Lua state. See the module docs for why this is all a backend implements.
pub trait LuaVm: Send {
    /// Run `source` as a chunk named `name` (errors carry that name).
    fn exec(&mut self, name: &str, source: &str) -> Result<(), String>;
    /// Call the global function `func` with one string argument; return its string result.
    fn call(&mut self, func: &str, arg: &str) -> Result<String, String>;
}

/// The playbook API, run into every VM before a playbook.
pub const PRELUDE: &str = include_str!("prelude.lua");

/// The chunk name the prelude runs under. `prelude.lua`'s error locator skips frames with
/// this source, so it must match there (`=prelude`).
pub const PRELUDE_NAME: &str = "=prelude";

/// Ready `vm` to run the playbook `source` (named `name` in its errors): run the prelude and
/// compile the playbook into its coroutine. Nothing of the playbook has run yet - the first
/// [`resume`] starts it.
pub fn load(vm: &mut dyn LuaVm, name: &str, source: &str) -> Result<(), String> {
    vm.exec(PRELUDE_NAME, PRELUDE)?;
    // The source travels as a JSON string argument; the prelude's own decoder is not involved
    // (it decodes REPLIES), so the call takes the raw text.
    let out = vm.call_load(source, name)?;
    match serde_json::from_str::<serde_json::Value>(&out) {
        Ok(serde_json::Value::Object(o)) => match o.get("error").and_then(|e| e.as_str()) {
            Some(e) => Err(e.to_string()),
            None => Ok(()),
        },
        _ => Err(format!("the prelude answered the load with {out:?}")),
    }
}

/// Resume the playbook with `reply` (JSON) and return its next request (JSON).
pub fn resume(vm: &mut dyn LuaVm, reply: &str) -> Result<String, String> {
    vm.call("__playbook_resume", reply)
}

/// The load call takes two arguments where every other call takes one. Rather than widen the
/// trait for it, the name and the source are passed as one string the prelude splits at its
/// FIRST newline. Not a NUL: a backend may hand the string to C as a C string (wasmoon does).
trait LoadCall {
    fn call_load(&mut self, source: &str, name: &str) -> Result<String, String>;
}

impl<V: LuaVm + ?Sized> LoadCall for V {
    fn call_load(&mut self, source: &str, name: &str) -> Result<String, String> {
        if name.contains('\n') || source.contains('\0') {
            return Err(format!("{name:?}: a playbook name has no newline and its source no NUL"));
        }
        self.call("__playbook_load", &format!("{name}\n{source}"))
    }
}
