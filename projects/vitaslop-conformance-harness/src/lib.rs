//! The engine-agnostic ARM conformance runner. It runs the generic corpus
//! (`vitaslop-conformance-suite-arm`) against anything that implements [`Engine`],
//! comparing each case to its golden. Both the native (wasmtime) test below and
//! the browser (WebAssembly) runner in `vitaslop-web` drive this same code over
//! the same embedded corpus, so a green run on both proves the transpiler's output
//! behaves identically on wasmtime and on the browser's own WebAssembly engine.
//!
//! The corpus uses the Linux ARM EABI as its host I/O convention (so a case runs
//! identically under qemu, the golden oracle, and our engine). That convention
//! lives here in [`linux_svc`], defined once and shared by every engine's `svc`
//! wiring - the engine and transpiler stay free of any syscall convention.

use vitaslop_conformance_suite_arm::{Case, Expected, Mode};
use vitaslop_runtime::{Flags, GuestMemory, SvcDispatch, SvcOutcome};
use vitaslop_transpiler::abi;

/// Address every arm case's image loads at (see the suite README).
pub const BASE: u32 = 0x10000;
/// Guest memory each case runs with (image + stack), from `base`. Matches the
/// native default so sp starts at the same place on both engines.
pub const MEM_BYTES: u32 = 64 * 1024 * 1024;

// Linux ARM EABI syscall numbers the corpus uses.
pub const SYS_EXIT: u32 = 1;
pub const SYS_WRITE: u32 = 4;
pub const SYS_EXIT_GROUP: u32 = 248;

/// `svc` numbers (guest r7) that never return, so the transpiler can end decoding
/// at a `svc` with a statically-known one of them.
pub const NORETURN_SVC: &[u32] = &[SYS_EXIT, SYS_EXIT_GROUP];

/// The corpus's Linux-EABI host convention as pure logic over guest state:
/// `write` (r7 = 4) appends `mem[r1..r1+r2]` to `out`; `exit`/`exit_group`
/// (r7 = 1 / 248) halt. The `write` fd (r0) is ignored - all writes are captured
/// output. Shared by every engine's host wiring so the convention is defined once.
pub fn linux_svc(
    regs: &[u32; abi::REG_COUNT],
    mem: &dyn GuestMemory,
    base: u32,
    out: &mut Vec<u8>,
) -> SvcOutcome {
    let (nr, r1, r2) = (regs[7], regs[1], regs[2]);
    match nr {
        SYS_WRITE => {
            if let Some(off) = r1.checked_sub(base) {
                let (off, len) = (off as usize, r2 as usize);
                if off + len <= mem.len() {
                    let mut buf = vec![0u8; len];
                    mem.read(off, &mut buf);
                    out.extend_from_slice(&buf);
                }
            }
            SvcOutcome::Continue
        }
        SYS_EXIT | SYS_EXIT_GROUP => SvcOutcome::Halt,
        _ => SvcOutcome::Continue,
    }
}

/// A stateful [`SvcDispatch`] implementing [`linux_svc`] and accumulating program
/// output. An engine that routes `env.svc` through a dispatcher (the browser
/// `WebVm`) wires one of these; read `output` back after the run.
#[derive(Default)]
pub struct LinuxSvc {
    pub output: Vec<u8>,
}

impl SvcDispatch for LinuxSvc {
    fn svc(
        &mut self,
        _imm: u32,
        regs: &mut [u32; abi::REG_COUNT],
        mem: &mut dyn GuestMemory,
        base: u32,
    ) -> SvcOutcome {
        linux_svc(regs, &*mem, base, &mut self.output)
    }
}

/// The final observable state of a case run: the captured register file, NZCV,
/// and program output.
pub struct CaseRun {
    pub regs: [u32; abi::REG_COUNT],
    pub flags: Flags,
    pub output: Vec<u8>,
}

/// An engine that can run one conformance case: transpile `bin` (loaded at
/// [`BASE`], entry [`BASE`], Thumb per `thumb`) over [`MEM_BYTES`] of guest
/// memory, seed `in_regs`, service the Linux-EABI `svc` convention ([`linux_svc`]),
/// run to halt, and report the final state. The two implementations are the native
/// wasmtime `Vm` (test below) and the browser `WebVm` (in `vitaslop-web`).
pub trait Engine {
    fn run_case(
        &mut self,
        bin: &[u8],
        thumb: bool,
        in_regs: &[(usize, u32)],
    ) -> Result<CaseRun, String>;
}

/// One case's verdict. `detail` is the mismatch/error message when `!pass`.
#[derive(Clone, Debug)]
pub struct Outcome {
    pub name: String,
    pub pass: bool,
    pub detail: Option<String>,
}

/// Run every case through `engine`, comparing each to its golden. Never panics: a
/// run error or a mismatch becomes a `pass = false` outcome with a message, so a
/// caller (native test or browser runner) can report the full set.
pub fn run_all<E: Engine>(engine: &mut E, cases: &[Case]) -> Vec<Outcome> {
    cases
        .iter()
        .map(|case| {
            let in_regs: Vec<(usize, u32)> =
                case.in_regs.iter().map(|(&i, &v)| (i as usize, v)).collect();
            let thumb = case.mode == Mode::Thumb;
            let detail = match engine.run_case(&case.bin, thumb, &in_regs) {
                Ok(run) => check(case, &run).err(),
                Err(e) => Some(format!("run failed: {e}")),
            };
            Outcome { name: case.name.clone(), pass: detail.is_none(), detail }
        })
        .collect()
}

/// Compare a run to a case's golden. `Ok` if it matches, else a mismatch message.
pub fn check(case: &Case, run: &CaseRun) -> Result<(), String> {
    match &case.expected {
        Expected::Output(want) => {
            if run.output != *want {
                return Err(format!("output mismatch: got {:?} want {:?}", run.output, want));
            }
        }
        Expected::Regs { regs, flags } => {
            // Captured registers are r0..r12 and r14; one not listed in the golden
            // is expected to be zero.
            for r in (0u8..=12).chain(std::iter::once(14)) {
                let want = regs.get(&r).copied().unwrap_or(0);
                let got = run.regs[r as usize];
                if got != want {
                    return Err(format!("r{r} mismatch: got {got:#x} want {want:#x}"));
                }
            }
            let g = &run.flags;
            if (g.n, g.z, g.c, g.v) != (flags.n, flags.z, flags.c, flags.v) {
                return Err(format!(
                    "NZCV mismatch: got {:?} want {:?}",
                    (g.n, g.z, g.c, g.v),
                    (flags.n, flags.z, flags.c, flags.v)
                ));
            }
        }
    }
    Ok(())
}

#[cfg(test)]
mod tests {
    use super::*;
    use vitaslop_conformance_suite_arm as suite;
    use vitaslop_native::{run, HostAbi};
    use vitaslop_runtime::SliceMemory;

    /// The native engine's `svc` fn-pointer: wrap wasmtime's rebased memory slice
    /// as a `GuestMemory` and delegate to the shared [`linux_svc`] convention.
    fn native_svc(
        _selector: u32,
        regs: &mut [u32; abi::REG_COUNT],
        mem: &mut [u8],
        base: u32,
        out: &mut Vec<u8>,
    ) -> SvcOutcome {
        let m = SliceMemory(mem);
        linux_svc(regs, &m, base, out)
    }

    /// The native (wasmtime) engine, driving `vitaslop-native::run`.
    struct NativeEngine;

    impl Engine for NativeEngine {
        fn run_case(
            &mut self,
            bin: &[u8],
            thumb: bool,
            in_regs: &[(usize, u32)],
        ) -> Result<CaseRun, String> {
            let abi = HostAbi { noreturn_svc: NORETURN_SVC, svc: native_svc, ..Default::default() };
            let r = run(bin, BASE, BASE, thumb, &[], in_regs, &abi)
                .map_err(|e| format!("{e:?}"))?;
            Ok(CaseRun { regs: r.regs, flags: r.flags, output: r.output })
        }
    }

    /// Run the whole embedded corpus through the native engine and diff each case
    /// against its golden. The browser runs the identical corpus through `WebVm`.
    #[test]
    fn run_cases() {
        let cases = suite::embedded_cases().expect("load arm corpus");
        assert!(!cases.is_empty(), "no cases found");
        let outcomes = run_all(&mut NativeEngine, &cases);
        let failures: Vec<&Outcome> = outcomes.iter().filter(|o| !o.pass).collect();
        assert!(
            failures.is_empty(),
            "{} of {} case(s) failed: {:#?}",
            failures.len(),
            outcomes.len(),
            failures
        );
    }
}
