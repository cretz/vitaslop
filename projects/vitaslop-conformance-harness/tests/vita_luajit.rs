//! An embedded LuaJIT, interpreter only, running a fixed script on the retail path (link, then
//! the threaded scheduler) - see `luajit-src/luajit.c` for why this program exists: its opcode
//! handlers are reached only through a dispatch table built at run time, so they are translated
//! when first reached, and they tail-jump to one another for as long as the script runs.
//!
//! The transcript is checked byte for byte against `cases/luajit.toml`, and the run reports how
//! much code it translated at run time.
//!
//! Run with: cargo test -p vitaslop-conformance-harness --test vita_luajit -- --nocapture

use vitaslop_loader as loader;
use vitaslop_native::{DeterministicWorld, RunReport, ThreadedScheduler, VitaEnv};
use vitaslop_runtime::link::link;

const LUAJIT: &[u8] = include_bytes!("../../vitaslop-conformance-suite-vita/luajit-src/luajit.velf");

const EXPECTED: &str = "\
start
ok 0
ok 1
ok 2
ok 3
ok 4
ok 5
ok 6
rung
ok 7
ok 8
55
ok 9
42
ok 10
7
ok 11
-1\t16777216\t5
ok 12
12\t8\t3\t16
ok 13
loop\t1399991
float\t10548.727944
fib\t17711
closure\t3\t2
ipairs\t1000\t333833500
pairs\t500\t125250
string\tababababab|LUA\t14\t11\txbxbxbxbxb|LUA
format\t42  3.14 ok ff
sort\t984\t512\t0
meta\t7
coroutine\t2\t20
pcall\tfalse\tboom
varargs\t3\t1\tnil\t3
bit\t240\t3\t1024\t0000dead
goto\t25
concat\t291
done
";

/// `LJ_DUMP=<hex addr>`: the lowered IR of one function of the probe, as the run lifts it.
#[test]
#[ignore = "diagnostic"]
fn dump_function() {
    let Ok(addr) = std::env::var("LJ_DUMP") else { return };
    let addr = u32::from_str_radix(addr.trim_start_matches("0x"), 16).expect("hex address");
    let linked = link(vec![loader::load(LUAJIT).expect("load")]).expect("link");
    let mut program = linked.program();
    let arm = [addr];
    program.arm_entries = &arm;
    eprintln!("{}", vitaslop_transpiler::dump_func(&program, addr).unwrap_or_default());
}

/// `LJ_CALL=<hex addr>:<r0>:<r1>:...`: call one ARM function of the probe image with the given
/// registers (hex) and print r0..r3 after.
#[test]
#[ignore = "diagnostic"]
fn call_function() {
    let Ok(spec) = std::env::var("LJ_CALL") else { return };
    let parts: Vec<u32> = spec.split(':').map(|s| u32::from_str_radix(s.trim_start_matches("0x"), 16).unwrap()).collect();
    let linked = link(vec![loader::load(LUAJIT).expect("load")]).expect("link");
    let entry = parts[0];
    let mut vm = vitaslop_native::Vm::new(
        &linked.image,
        linked.base,
        false,
        &[entry],
        &[],
        linked.mem_bytes,
        &vitaslop_native::HostAbi::default(),
    )
    .expect("transpile");
    for (i, &v) in parts[1..].iter().enumerate() {
        vm.set_reg(i, v);
    }
    let r = vm.call(entry);
    eprintln!("{r:?} r0={:#x} r1={:#x} r2={:#x} r3={:#x}", vm.get_reg(0), vm.get_reg(1), vm.get_reg(2), vm.get_reg(3));
}

#[test]
fn luajit_interpreter_runs_its_script() {
    let m = loader::load(LUAJIT).expect("load luajit.velf");
    vitaslop_runtime::vita::set_preemptive_linking(true);
    let linked = link(vec![m]);
    vitaslop_runtime::vita::set_preemptive_linking(false);
    let linked = linked.expect("link");

    let mut env = VitaEnv::new(
        linked.imports.clone(),
        linked.base,
        linked.mem_bytes,
        Box::new(DeterministicWorld::default()),
    );
    env.state.set_alloc_base(linked.alloc_base);
    env.state.set_process_param(linked.process_param);
    env.state.set_main_thread_request(linked.main_thread_request());
    env.state.set_modules(linked.loaded_modules.clone());
    env.state.set_unhandled_imports(linked.unhandled_imports.clone());
    env.state.set_tls_template(linked.tls_template);
    env.state.set_preemptive(true);

    let started = std::time::Instant::now();
    let (mut sched, stubbed) = ThreadedScheduler::from_linked(&linked, env, 5_000_000).expect("scheduler");
    let report = sched.run();
    let elapsed = started.elapsed();
    let stats = sched.runtime_stats();
    let host = sched.host();
    let cap = &host.state.capture;
    let output = String::from_utf8_lossy(&cap.stdout);
    eprintln!(
        "---output---\n{output}------------\n{report:?} in {:.0} ms; {} stub(s); runtime code: {} entries ({} swept) \
         in {} side module(s), {} us translating",
        elapsed.as_secs_f64() * 1000.0,
        stubbed.len(),
        stats.entries,
        stats.swept,
        stats.modules,
        stats.busy_us
    );
    eprintln!("unimplemented: {:?}\nstderr: {}", cap.unimplemented, String::from_utf8_lossy(&cap.stderr));
    if vitaslop_runtime::call_table::spec().is_some() {
        eprintln!("{}", vitaslop_runtime::call_table::report());
    }
    assert_eq!(output, EXPECTED);
    assert!(matches!(report, RunReport::Finished(0)), "{report:?}");
}
