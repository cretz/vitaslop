//! Preemptive scheduler substrate test.
//!
//! Drives [`vitaslop_native::ThreadedScheduler`] with a hand-built ARM guest (three
//! functions: `main`, worker `A`, worker `B`) and a mock host that speaks the
//! preemptive `ImportDispatch` protocol (spawn / block / wake / thread-exit /
//! process-halt). It proves the whole substrate end to end WITHOUT the Vita NID
//! stack: two workers really run as their own fibers over one shared linear memory
//! while `main` is parked, a thread blocks on an empty semaphore and is woken by
//! another thread's signal, `main` blocks joining a worker and resumes when it
//! ends, and the observable interleaving is exactly the deterministic order the
//! cooperative round-robin dictates.
//!
//! The guest is pure `bl <stub>` / `bx lr`: the transpiler turns every `bl` to a
//! stub address in the import map into a host call, and a `bx lr` into a structural
//! wasm return, so no stack or real veneer is needed - the control flow is exact.

use std::collections::HashSet;

use vitaslop_native::{RunReport, ThreadSpawn, ThreadedScheduler};
use vitaslop_runtime::{GuestMemory, ImportDispatch, SvcOutcome, VFP_ARG_COUNT};
use vitaslop_transpiler::abi::REG_COUNT;
use vitaslop_transpiler::Extern;

const BASE: u32 = 0x1_0000;
const MEM_BYTES: u32 = 0x10_0000;

// Guest thread entry addresses (ARM, so bit 0 is clear).
const MAIN: u32 = 0x1_0000;
const WORKER_A: u32 = 0x1_0100;
const WORKER_B: u32 = 0x1_0200;

// Guest thread ids the mock assigns. The scheduler gives the main thread id 0.
const TH_A: i32 = 1;
const TH_B: i32 = 2;

// Dense import indices (the `bl` stub each function calls maps to one of these).
mod imp {
    pub const SPAWN_A: u32 = 0;
    pub const SPAWN_B: u32 = 1;
    pub const JOIN_A: u32 = 2;
    pub const JOIN_B: u32 = 3;
    pub const WRITE_M: u32 = 4;
    pub const EXIT: u32 = 5;
    pub const WAIT_SEMA: u32 = 6;
    pub const WRITE_A: u32 = 7;
    pub const WRITE_B: u32 = 8;
    pub const SIGNAL_SEMA: u32 = 9;
}

/// The stub address each import index lives at (any address in the map works; the
/// `bl` never executes the stub, it becomes a host call). Kept apart from the code
/// image so it is obviously never decoded.
const STUB_BASE: u32 = 0x2_0000;
fn stub(index: u32) -> u32 {
    STUB_BASE + index * 4
}

/// `bl <target>` (ARM) encoded so the transpiler routes the call to `target`.
///
/// The transpiler resolves an ARM `BL`/`B` target as `addr + (off << 2)` where
/// yaxpeax's `off` already folds the PC+8 prefetch bias in - i.e. the architectural
/// `addr + 8 + (imm24 << 2)`. So the raw `imm24` encodes `(target - addr - 8) >> 2`.
fn bl(addr: u32, target: u32) -> u32 {
    let imm24 = ((target.wrapping_sub(addr).wrapping_sub(8)) >> 2) & 0x00FF_FFFF;
    0xEB00_0000 | imm24
}

/// `bx lr` (ARM): a structural return in the transpiler.
const BX_LR: u32 = 0xE12F_FF1E;

/// Build the guest image: `main` at 0, `A` at 0x100, `B` at 0x200 (offsets from
/// BASE), each a run of import calls ending in `bx lr`.
fn build_guest() -> Vec<u8> {
    // Cover up to B's last instruction.
    let mut code = vec![0u8; 0x220];
    let put = |code: &mut Vec<u8>, off: u32, word: u32| {
        code[off as usize..off as usize + 4].copy_from_slice(&word.to_le_bytes());
    };
    let at = |guest: u32| guest; // offsets equal guest addr - BASE below

    // main: spawn A, spawn B, join A, join B, write 'M', exit process.
    let m = at(MAIN - BASE);
    put(&mut code, m, bl(MAIN, stub(imp::SPAWN_A)));
    put(&mut code, m + 4, bl(MAIN + 4, stub(imp::SPAWN_B)));
    put(&mut code, m + 8, bl(MAIN + 8, stub(imp::JOIN_A)));
    put(&mut code, m + 12, bl(MAIN + 12, stub(imp::JOIN_B)));
    put(&mut code, m + 16, bl(MAIN + 16, stub(imp::WRITE_M)));
    put(&mut code, m + 20, bl(MAIN + 20, stub(imp::EXIT)));
    put(&mut code, m + 24, BX_LR);

    // A (the waiter): wait on the semaphore, then write 'A'.
    let a = at(WORKER_A - BASE);
    put(&mut code, a, bl(WORKER_A, stub(imp::WAIT_SEMA)));
    put(&mut code, a + 4, bl(WORKER_A + 4, stub(imp::WRITE_A)));
    put(&mut code, a + 8, BX_LR);

    // B (the signaller): write 'B', then signal the semaphore (waking A).
    let b = at(WORKER_B - BASE);
    put(&mut code, b, bl(WORKER_B, stub(imp::WRITE_B)));
    put(&mut code, b + 4, bl(WORKER_B + 4, stub(imp::SIGNAL_SEMA)));
    put(&mut code, b + 8, BX_LR);

    code
}

/// The externs table: every stub address -> its dense import index.
fn build_externs() -> Vec<Extern> {
    (0..10).map(|i| Extern { addr: stub(i), import: i }).collect()
}

/// The mock host: a tiny kernel that speaks the preemptive `ImportDispatch`
/// protocol. It tracks a semaphore, a shared output log, and the wait/wake
/// bookkeeping the scheduler drains.
#[derive(Default)]
struct MockKernel {
    log: Vec<u8>,
    current: i32,
    sema_count: i32,
    sema_waiters: Vec<i32>,
    /// (waiter thid, target thid) for threads parked in a join.
    join_waiters: Vec<(i32, i32)>,
    finished: HashSet<i32>,
    pending_spawns: Vec<ThreadSpawn>,
    pending_wakes: Vec<i32>,
}

impl MockKernel {
    /// A distinct stack top per worker (unused - the guest never pushes - but a
    /// faithful spawn descriptor carries one).
    fn stack_for(thid: i32) -> u32 {
        BASE + MEM_BYTES - (thid as u32) * 0x1000
    }

    fn spawn(&mut self, entry: u32, thid: i32) {
        self.pending_spawns.push(ThreadSpawn {
            entry,
            arg_len: 0,
            arg_ptr: 0,
            stack_top: Self::stack_for(thid),
            thid,
        });
    }

    /// Join `target`: continue if it already finished, else park the current
    /// thread until it does.
    fn join(&mut self, target: i32) -> SvcOutcome {
        if self.finished.contains(&target) {
            SvcOutcome::Continue
        } else {
            self.join_waiters.push((self.current, target));
            SvcOutcome::Block
        }
    }
}

impl ImportDispatch for MockKernel {
    fn dispatch(
        &mut self,
        index: u32,
        _regs: &mut [u32; REG_COUNT],
        _vfp: &mut [u32; VFP_ARG_COUNT],
        _mem: &mut dyn GuestMemory,
        _base: u32,
    ) -> SvcOutcome {
        match index {
            imp::SPAWN_A => {
                self.spawn(WORKER_A, TH_A);
                SvcOutcome::Continue
            }
            imp::SPAWN_B => {
                self.spawn(WORKER_B, TH_B);
                SvcOutcome::Continue
            }
            imp::JOIN_A => self.join(TH_A),
            imp::JOIN_B => self.join(TH_B),
            imp::WRITE_M => {
                self.log.push(b'M');
                SvcOutcome::Continue
            }
            imp::EXIT => SvcOutcome::Halt,
            imp::WAIT_SEMA => {
                if self.sema_count > 0 {
                    self.sema_count -= 1;
                    SvcOutcome::Continue
                } else {
                    self.sema_waiters.push(self.current);
                    SvcOutcome::Block
                }
            }
            imp::WRITE_A => {
                self.log.push(b'A');
                SvcOutcome::Continue
            }
            imp::WRITE_B => {
                self.log.push(b'B');
                SvcOutcome::Continue
            }
            imp::SIGNAL_SEMA => {
                self.sema_count += 1;
                // Release one waiter if any is parked (it consumes the token).
                if let Some(w) = self.sema_waiters.pop() {
                    self.sema_count -= 1;
                    self.pending_wakes.push(w);
                }
                SvcOutcome::Continue
            }
            other => panic!("unexpected import index {other}"),
        }
    }

    fn set_current_thread(&mut self, thid: i32) {
        self.current = thid;
    }

    fn take_spawns(&mut self) -> Vec<ThreadSpawn> {
        std::mem::take(&mut self.pending_spawns)
    }

    fn take_wakes(&mut self) -> Vec<i32> {
        std::mem::take(&mut self.pending_wakes)
    }

    fn set_thread_exit(&mut self, thid: i32, _code: u32) {
        self.finished.insert(thid);
        // Wake anyone joined on this thread.
        let (wake, keep): (Vec<_>, Vec<_>) =
            self.join_waiters.drain(..).partition(|(_, target)| *target == thid);
        self.join_waiters = keep;
        for (waiter, _) in wake {
            self.pending_wakes.push(waiter);
        }
    }
}

#[test]
fn preemptive_block_wake_spawn_ordering() {
    let code = build_guest();
    let externs = build_externs();
    let mut sched = ThreadedScheduler::new(
        &code,
        BASE,
        false, // ARM
        &[MAIN, WORKER_A, WORKER_B],
        &externs,
        MEM_BYTES,
        MockKernel::default(),
    )
    .expect("build scheduler");

    let report = sched.run();
    assert_eq!(report, RunReport::Finished(0), "process should halt cleanly");

    // The only interleaving the cooperative round-robin allows:
    //  - main spawns A and B, then blocks joining A;
    //  - A runs, blocks on the empty semaphore;
    //  - B runs to completion, writing 'B' and signalling (waking A);
    //  - A resumes, writes 'A', ends (waking main's join);
    //  - main resumes, join B is already done, writes 'M', exits.
    // So the shared log is exactly "BAM": B before A (A was blocked on the sema B
    // signalled) and M last (main was blocked joining until both workers ended).
    let log = String::from_utf8(sched.host().log.clone()).unwrap();
    assert_eq!(log, "BAM", "observed interleaving proves real blocking and wakeups");
}
