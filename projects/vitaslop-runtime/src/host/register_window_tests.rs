//! >>> THE HOST-CALL REGISTER WINDOW IS A CONTRACT, AND THIS IS WHAT ENFORCES IT.
//!
//! The browser marshals only the registers AAPCS lets a host call reach
//! (`browser_sched::NARROW_REGS`: r0..r3, sp, lr, pc). Reading r4..r11 through a live
//! `GuestCtx` would therefore hand back ZERO rather than the guest's value - a silent
//! wrong answer on the one engine that ships, and invisible on the desktop, which
//! marshals the whole file.
//!
//! A comment cannot hold that. This scans the workspace for the shape that would break
//! it, in the same way `knobs::a_knob_routed_through_this_module_is_reachable_from_the_browser`
//! scans for an unregistered knob - and for the same reason: the failure is silent, so it
//! has to be caught where it is written rather than where it shows.

/// Register indices a host call may read through a `GuestCtx`.
const WINDOW: &[usize] = &[0, 1, 2, 3, 13, 14, 15];

#[test]
fn a_host_handler_only_reads_registers_the_call_marshals() {
    let root = crate::knobs::workspace_root();
    let mut bad = Vec::new();
    for path in crate::knobs::rust_sources(&root) {
        let rel = path
            .strip_prefix(&root)
            .unwrap_or(&path)
            .to_string_lossy()
            .replace('\\', "/");
        // The browser scheduler owns the window and names it; the transpiler and the
        // native engine hold whole register FILES of their own, which are not this.
        // `host.rs` is split across `host/`, and every file of it is a host handler.
        if !rel.starts_with("vitaslop-runtime/src/vita/")
            && !rel.ends_with("vitaslop-runtime/src/host.rs")
            && !rel.starts_with("vitaslop-runtime/src/host/")
        {
            continue;
        }
        let Ok(text) = std::fs::read_to_string(&path) else {
            continue;
        };
        for (i, line) in text.lines().enumerate() {
            // `ctx.regs[N]` / `self.regs[N]` with a literal index. A non-literal index is
            // either the AAPCS argument cursor (bounded to 0..4 where it is written) or a
            // loop over a caller's OWN array, neither of which is a live-context read.
            for m in line.match_indices(".regs[") {
                let rest = &line[m.0 + ".regs[".len()..];
                let digits: String = rest.chars().take_while(|c| c.is_ascii_digit()).collect();
                let Ok(n) = digits.parse::<usize>() else {
                    continue;
                };
                // A range (`regs[0..13]`) is spelt with a literal too; treat its start as
                // the index and let the end be caught by the same rule on the next scan.
                if !WINDOW.contains(&n) {
                    bad.push(format!("{rel}:{} reads r{n}", i + 1));
                }
                if let Some(end) = rest
                    .strip_prefix(&digits)
                    .and_then(|r| r.strip_prefix(".."))
                {
                    let e: String = end.chars().take_while(|c| c.is_ascii_digit()).collect();
                    if let Ok(e) = e.parse::<usize>()
                        && (n..e).any(|k| !WINDOW.contains(&k))
                    {
                        bad.push(format!("{rel}:{} reads r{n}..r{e}", i + 1));
                    }
                }
            }
        }
    }
    bad.sort();
    bad.dedup();
    assert!(
        bad.is_empty(),
        "these read a register the BROWSER does not marshal into a host call, so they \
         would read ZERO there and the guest's value on the desktop:\n  {}\n\
         Either use one of r{WINDOW:?}, or widen `browser_sched::NARROW_REGS` and this \
         list together.",
        bad.join("\n  ")
    );
}
