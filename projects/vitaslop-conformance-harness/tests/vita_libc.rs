//! DIAGNOSTIC probe for the full-newlib (non -nostdlib) path - the first rung of
//! the newlib -> Chocolate Doom arc. libc.velf links REAL newlib, so libc code
//! (malloc, stdio, string, software divide) runs as guest ARM and the newlib
//! syscall bottom surfaces the real host-call demand for full-libc titles.
//!
//! It is `#[ignore]`d because it cannot run to completion yet: the prebuilt
//! vitasdk newlib is NEON-vectorized (memcpy/str* use `vorr`/`vmov-reg`,
//! `vdup.32`, `vadd.i32` on D/Q registers), and the transpiler's NEON decode+lift
//! does not yet cover the logical 3-reg-same family + `vdup`. Until that lands,
//! this probe transpile-fails inside a newlib function with a `Decode` error.
//!
//! What it DOES prove today (run it with `--ignored --nocapture` to watch):
//! transpilation now gets PAST the newlib entry machinery - Thumb->ARM
//! interworking veneers, tail-calls to import stubs, indirect calls (`blx rN` /
//! `bx rN` through the module dispatcher), and `.init_array` constructor
//! discovery all work. The remaining blocker is NEON completeness, not the
//! newlib-linking plumbing. Run:
//!   cargo test -p vitaslop-conformance-harness --test vita_libc -- --ignored --nocapture

use std::cell::RefCell;
use std::rc::Rc;

use vitaslop_loader as loader;
use vitaslop_native::{DeterministicWorld, HostAbi, VitaEnv, Vm};

const LIBC: &[u8] =
    include_bytes!("../../vitaslop-conformance-suite-vita/libc-src/libc.velf");

#[test]
#[ignore = "blocked on NEON decode+lift completeness (prebuilt newlib is vectorized)"]
fn libc_probe() {
    let m = loader::load(LIBC).expect("load libc.velf");
    eprintln!(
        "loaded: name={} base={:#x} entry={:#x} segments={} imports={} init_pointers={:x?}",
        m.name,
        m.base,
        m.entry,
        m.segments.len(),
        m.imports.len(),
        m.init_pointers,
    );

    let inputs = m.program_inputs();
    let imports: Vec<(u32, u32)> =
        m.imports.iter().map(|i| (i.library_nid, i.func_nid)).collect();

    let world = Box::new(DeterministicWorld::default());
    let env = VitaEnv::new(imports, inputs.base, inputs.mem_bytes, world);
    let env = Rc::new(RefCell::new(env));

    let vm = Vm::new(
        &inputs.code,
        inputs.base,
        inputs.thumb_entry,
        &inputs.entries,
        &inputs.externs,
        inputs.mem_bytes,
        &HostAbi::default(),
    );
    let mut vm = match vm {
        Ok(vm) => vm,
        Err(e) => {
            eprintln!("TRANSPILE/INSTANTIATE FAILED (expected until NEON lands): {e:?}");
            return;
        }
    };
    eprintln!("transpiled+instantiated OK");
    vm.set_import_env(Box::new(env.clone()));

    match vm.call(m.entry & !1) {
        Ok(()) => eprintln!("run: clean halt"),
        Err(e) => eprintln!("run: trapped: {e:?}"),
    }

    let env = env.borrow();
    let cap = &env.state.capture;
    eprintln!(
        "calls={} stdout_bytes={}\nunimplemented={:?}\n---stdout---\n{}------------",
        cap.call_count,
        cap.stdout.len(),
        cap.unimplemented,
        String::from_utf8_lossy(&cap.stdout)
    );
}
