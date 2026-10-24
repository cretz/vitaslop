//! Boot a decrypted retail title into the threaded scheduler - the one path every native
//! front end and tool shares.

use vitaslop_loader as loader;
use vitaslop_runtime::ingest::pipeline::decrypt_container;
use vitaslop_runtime::ingest::vfs::{DirVfs, Vfs};
use vitaslop_runtime::link::link;

use crate::{ThreadedScheduler, VitaEnv};

/// Boot a decrypted retail title from its extracted app dir, wired to `world` for
/// input. Shared by the desktop front-end, the boot probe, the resident session and the tools.
pub fn boot_retail(
    dir: &str,
    world: Box<dyn vitaslop_runtime::World + Send>,
    quantum_fuel: u64,
) -> Result<ThreadedScheduler<VitaEnv>, String> {
    boot_retail_exec(dir, world, quantum_fuel, None)
}

/// [`boot_retail`] with `main_exec` - an `sceAppMgrLoadExec` path - in the main executable's
/// place: the process a title's launcher exec'd into. A tool that boots a launcher-style title
/// takes `state.exec_request` when the run finishes and boots again through this.
pub fn boot_retail_exec(
    dir: &str,
    world: Box<dyn vitaslop_runtime::World + Send>,
    quantum_fuel: u64,
    main_exec: Option<&str>,
) -> Result<ThreadedScheduler<VitaEnv>, String> {
    let mut game = decrypt_container(&mut DirVfs::new(dir)).map_err(|e| format!("decrypt: {e:?}"))?;
    if let Some(path) = main_exec
        && !game.with_main_exec(path)
    {
        return Err(format!("sceAppMgrLoadExec(\"{path}\"): the app carries no such executable"));
    }
    let modules: Vec<loader::Module> = game
        .modules
        .iter()
        .map(|m| loader::load(&m.elf).map_err(|e| format!("load module: {e:?}")))
        .collect::<Result<_, _>>()?;
    // This run stands up the THREADED scheduler, so the host mirror block exists and is
    // refreshed at its resume point - which is what lets the RTC tick be read inline.
    // See `vitaslop_runtime::vita::set_preemptive_linking`.
    vitaslop_runtime::vita::set_preemptive_linking(true);
    let linked = link(modules);
    // Reset at once: the flag is process-wide and read only by `link`, so left set it would
    // make a later link in this process (a test, a second title) inline a clock read its
    // scheduler does not refresh.
    vitaslop_runtime::vita::set_preemptive_linking(false);
    let linked = linked.map_err(|e| format!("link: {e:?}"))?;
    let mut env = VitaEnv::new(linked.imports.clone(), linked.base, linked.mem_bytes, world);
    // >>> EVERY NATIVE TOOL DECODES VIDEO, because the one that did not could not SEE the
    // movie path at all.
    //
    // Only the desktop front-end installed this, so `bench`, `session`, `explore` and
    // the boot probe all ran with no decoder. That is not a missing feature in a measurement
    // tool, it is a measurement tool that answers the wrong question: `bench` exists to say
    // where a frame's guest CPU goes, and on a movie screen it reported the cost of a title
    // spinning against a decoder that had refused to be created. The refusal was a `warn` in a
    // log nobody reads when they are reading a phase table.
    env.state.video = Box::new(vitaslop_platform::video::H264Factory);
    env.state.audio_dec = Box::new(vitaslop_platform::audio_dec::AacFactory);
    env.state.set_alloc_base(linked.alloc_base);
    env.state.set_process_param(linked.process_param);
    env.state.set_main_thread_request(linked.main_thread_request());
    env.state.set_modules(linked.loaded_modules.clone());
    env.state.set_unhandled_imports(linked.unhandled_imports.clone());
    // The TLS template seeds each thread's thread-local block; without it an early
    // thread-local access reads uninitialized memory and traps (MemoryOutOfBounds at
    // boot). The boot probe sets this too - keep the shared helper in parity.
    env.state.set_tls_template(linked.tls_template);
    env.state.set_preemptive(true);
    for path in game.files.list() {
        if let Ok(bytes) = game.files.read(&path) {
            env.state.add_file(&path, bytes);
        }
    }
    let (sched, _stubs) = ThreadedScheduler::from_linked(&linked, env, quantum_fuel)
        .map_err(|e| format!("scheduler: {e:?}"))?;
    Ok(sched)
}
