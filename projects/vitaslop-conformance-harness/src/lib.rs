//! Runs the generic ARM conformance corpus (`vitaslop-conformance-suite-arm`)
//! against vitaslop's engine. This is the only crate that pairs a corpus with
//! vitaslop code; the corpus itself is engine-agnostic.
//!
//! The corpus uses the Linux ARM EABI as its host I/O convention, so a case
//! binary runs identically under qemu (the golden oracle) and our engine. That
//! Linux convention lives here in the harness, injected as the run's `HostAbi` -
//! the engine and transpiler stay free of any syscall convention.

#[cfg(test)]
mod tests {
    use vitaslop_conformance_suite_arm::{self as suite, Expected};
    use vitaslop_native::{HostAbi, SvcOutcome, run_arm};

    /// Address every arm case's image loads at (see the suite README).
    const BASE: u32 = 0x10000;

    // Linux ARM EABI syscall numbers the corpus uses.
    const SYS_EXIT: u32 = 1;
    const SYS_WRITE: u32 = 4;
    const SYS_EXIT_GROUP: u32 = 248;

    /// The corpus's host convention: Linux `write` to the output sink, `exit`
    /// halts. `write`'s fd (r0) is ignored - all writes are captured output.
    fn linux_svc(nr: u32, _r0: u32, r1: u32, r2: u32, mem: &[u8], out: &mut Vec<u8>) -> SvcOutcome {
        match nr {
            SYS_WRITE => {
                let (ptr, len) = (r1 as usize, r2 as usize);
                if let Some(bytes) = ptr.checked_add(len).and_then(|end| mem.get(ptr..end)) {
                    out.extend_from_slice(bytes);
                }
                SvcOutcome::Continue
            }
            SYS_EXIT | SYS_EXIT_GROUP => SvcOutcome::Halt,
            _ => SvcOutcome::Continue,
        }
    }

    fn linux_abi() -> HostAbi<'static> {
        HostAbi {
            noreturn_svc: &[SYS_EXIT, SYS_EXIT_GROUP],
            svc: linux_svc,
        }
    }

    /// Transpile, run, and diff every case against its golden. Flags are not
    /// modeled yet (no CPSR), so `regs` cases check registers only.
    #[test]
    fn run_cases() {
        let cases = suite::load_cases().expect("load arm corpus");
        assert!(!cases.is_empty(), "no cases found");
        let abi = linux_abi();

        for case in &cases {
            let in_regs: Vec<(usize, u32)> = case
                .in_regs
                .iter()
                .map(|(&i, &v)| (i as usize, v))
                .collect();
            let result = run_arm(&case.bin, BASE, BASE, &in_regs, &abi)
                .unwrap_or_else(|e| panic!("{}: run failed: {e:?}", case.name));

            match &case.expected {
                Expected::Output(want) => {
                    assert_eq!(result.output, *want, "{}: output mismatch", case.name);
                }
                Expected::Regs { regs, flags: _ } => {
                    // Captured registers are r0..r12 and r14; one not listed in
                    // the golden is expected to be zero.
                    for r in (0u8..=12).chain(std::iter::once(14)) {
                        let want = regs.get(&r).copied().unwrap_or(0);
                        assert_eq!(
                            result.regs[r as usize], want,
                            "{}: r{r} mismatch",
                            case.name
                        );
                    }
                }
            }
        }
    }
}
