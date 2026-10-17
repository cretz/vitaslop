//! The native Lua backend: PUC-Rio Lua 5.4, built from source by `mlua`'s `vendored` feature.
use mlua::{Function, Lua, LuaOptions, StdLib};

use crate::vm::LuaVm;

/// One Lua 5.4 state.
pub struct MluaVm {
    lua: Lua,
}

impl MluaVm {
    pub fn new() -> MluaVm {
        // The libraries a playbook gets. No `io`, `os` or `package`: a playbook reads the GAME,
        // never the machine - a clock or a file read would make two runs of one playbook
        // differ. `debug` is there for the prelude's error locations and tracebacks, which is
        // also why this is `unsafe_new_with` (mlua treats `debug` as unsafe).
        let libs = StdLib::COROUTINE | StdLib::TABLE | StdLib::STRING | StdLib::UTF8 | StdLib::MATH | StdLib::DEBUG;
        // SAFETY: `debug` is the only unsafe library loaded, and the only code run in this state
        // is the prelude and the project's own playbooks.
        let lua = unsafe { Lua::unsafe_new_with(libs, LuaOptions::default()) };
        MluaVm { lua }
    }
}

impl Default for MluaVm {
    fn default() -> Self {
        MluaVm::new()
    }
}

impl LuaVm for MluaVm {
    fn exec(&mut self, name: &str, source: &str) -> Result<(), String> {
        self.lua.load(source).set_name(name).exec().map_err(|e| e.to_string())
    }

    fn call(&mut self, func: &str, arg: &str) -> Result<String, String> {
        let f: Function = self.lua.globals().get(func).map_err(|e| format!("{func}: {e}"))?;
        let s: mlua::String = f.call(self.lua.create_string(arg).map_err(|e| e.to_string())?).map_err(|e| e.to_string())?;
        Ok(s.to_str().map_err(|e| e.to_string())?.to_string())
    }
}
