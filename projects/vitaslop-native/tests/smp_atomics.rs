//! SMP codegen under REAL parallelism: two OS threads, each running its own instance of one
//! transpiled module over ONE shared wasm memory, both hammering a counter with the guest's
//! own `LDREX`/`add`/`STREX` retry loop - the shape every guest atomic increment compiles to.
//!
//! The ARM conformance corpus (`run_cases_smp`) proves the SMP exclusive forms equal to the
//! qemu oracle, but on ONE thread, where a monitor in one shared word and a per-instance one
//! are indistinguishable. This is the test that can tell them apart: with the per-instance,
//! compare-and-swap monitor every increment lands and the count is exact; with the one-baton
//! build's shared-word monitor and plain load/store, increments are lost.
//!
//! The negative control runs the very same program built WITHOUT SMP and requires that it
//! LOSES increments, which is what shows the test is able to see a race at all. It needs two
//! cores actually running at once; on a machine with one it would be skipped rather than
//! reported, so it says which it did.

use std::sync::{Arc, Barrier};

use vitaslop_transpiler::{self as transpiler, abi};
use wasmtime::{Config, Engine, Instance, Linker, MemoryType, Module, SharedMemory, Store, Val};

/// Thumb, assembled by `arm-none-eabi-as -march=armv7-a` from:
/// ```text
/// again: ldrex r2, [r1]
///        adds  r2, r2, #1
///        strex r3, r2, [r1]
///        cmp   r3, #0
///        bne   again
///        subs  r4, r4, #1
///        bne   again
///        bx    lr
/// ```
/// r1 = the counter's address, r4 = how many increments.
const LOOP: [u8; 20] = [
    0x51, 0xe8, 0x00, 0x2f, 0x01, 0x32, 0x41, 0xe8, 0x00, 0x23, 0x00, 0x2b, 0xf8, 0xd1, 0x01,
    0x3c, 0xf6, 0xd1, 0x70, 0x47,
];

const BASE: u32 = 0x10000;
const MEM_BYTES: u32 = 1 << 20;
/// The counter: a word well inside the guest region, clear of the code.
const COUNTER: u32 = BASE + 0x8000;
const PER_THREAD: u32 = 2_000_000;
const THREADS: usize = 2;

/// Build the loop (SMP or not) as a module that imports a SHARED memory, run it on
/// [`THREADS`] OS threads at once, and return the final counter.
fn race(smp: bool) -> u32 {
    transpiler::set_smp(smp);
    let artifact = transpiler::transpile(&transpiler::Program {
        code: &LOOP,
        base: BASE,
        thumb: true,
        entries: &[BASE],
        arm_entries: &[],
        externs: &[],
        redirects: &[],
        inline_imports: &[],
        noreturn_svc: &[],
        mem_bytes: MEM_BYTES,
        discover_code_pointers: false,
        // One shared memory, one instance per thread: the SMP shape exactly.
        import_memory: true,
        host_off: 0,
    })
    .expect("transpile");
    transpiler::set_smp(false);

    let mut config = Config::new();
    config.wasm_threads(true);
    config.shared_memory(true);
    let engine = Engine::new(&config).expect("engine");
    let module = Module::from_binary(&engine, &artifact.wasm).expect("module");
    let pages = u32::from(u16::MAX).min(artifact.mem_pages);
    let memory =
        SharedMemory::new(&engine, MemoryType::shared(pages, pages)).expect("shared memory");
    // The code image at the rebase origin, and the counter at zero.
    {
        let data = memory.data();
        for (i, b) in LOOP.iter().enumerate() {
            unsafe { *data[i].get() = *b };
        }
    }

    let start = Arc::new(Barrier::new(THREADS));
    let workers: Vec<_> = (0..THREADS)
        .map(|_| {
            let (engine, module, memory, start) =
                (engine.clone(), module.clone(), memory.clone(), start.clone());
            std::thread::spawn(move || {
                let mut store = Store::new(&engine, ());
                let mut linker = Linker::new(&engine);
                let nop1 = |_: i32| {};
                linker.func_wrap(abi::IMPORT_MODULE, abi::SVC_NAME, nop1).unwrap();
                linker.func_wrap(abi::IMPORT_MODULE, abi::IMPORT_NAME, nop1).unwrap();
                linker.func_wrap(abi::IMPORT_MODULE, abi::IMPORT_FAST_NAME, nop1).unwrap();
                linker
                    .func_wrap(abi::IMPORT_MODULE, abi::DISPATCH_MISS_NAME, |_: i32, _: i32| {})
                    .unwrap();
                linker.define(&mut store, abi::IMPORT_MODULE, abi::MEMORY_EXPORT, memory).unwrap();
                let instance: Instance = linker.instantiate(&mut store, &module).expect("instantiate");
                let set = |store: &mut Store<()>, r: usize, v: u32| {
                    instance
                        .get_global(&mut *store, &abi::reg_export(r))
                        .unwrap()
                        .set(&mut *store, Val::I32(v as i32))
                        .unwrap();
                };
                set(&mut store, 1, COUNTER);
                set(&mut store, 4, PER_THREAD);
                set(&mut store, abi::SP, BASE + MEM_BYTES - 64);
                let entry = instance
                    .get_typed_func::<(), ()>(&mut store, &abi::func_export(BASE))
                    .expect("entry");
                start.wait();
                entry.call(&mut store, ()).expect("run");
            })
        })
        .collect();
    for w in workers {
        w.join().expect("worker");
    }
    let off = (COUNTER - BASE) as usize;
    let data = memory.data();
    let bytes: [u8; 4] = std::array::from_fn(|i| unsafe { *data[off + i].get() });
    u32::from_le_bytes(bytes)
}

#[test]
fn smp_exclusives_lose_no_increment_across_real_threads() {
    let want = PER_THREAD * THREADS as u32;
    let got = race(true);
    assert_eq!(got, want, "SMP LDREX/STREX lost {} of {want} increments", want - got);
}

#[test]
fn the_one_baton_build_does_lose_increments_so_the_race_is_visible() {
    if std::thread::available_parallelism().map_or(1, |n| n.get()) < 2 {
        eprintln!("SKIPPED: one core, the negative control cannot race");
        return;
    }
    let want = PER_THREAD * THREADS as u32;
    let got = race(false);
    eprintln!("one-baton build: {got} of {want} increments survived");
    assert!(got < want, "the non-SMP build lost nothing - this test cannot see a race");
}
