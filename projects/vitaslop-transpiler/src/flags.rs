//! Condition-flag liveness: which of N, Z, C, V a flag-setting statement has to
//! actually compute.
//!
//! # Why this is the largest single lever on the translated code
//! ARM sets flags constantly and reads them rarely. Every `cmp`, `cmn`, `tst`, `teq`,
//! every `adds`/`subs` and every `movs` writes all four (or three) flags; the consumer
//! is almost always ONE conditional branch or ONE predicated instruction a few
//! instructions later, and a condition code reads at most two flags. `eq` reads Z. `mi`
//! reads N. `lt` reads N and V. Nothing reads all four at once - there is no condition
//! code that does.
//!
//! The emitter computed all of them anyway, because a statement cannot see its own
//! consumers. [`emit_flags_add`](crate::emit) alone is about forty wasm operators, of
//! which the unsigned carry is nine (an i64 widening, two extends, two adds, a shift and
//! a wrap) and the signed overflow is eight. A `cmp` feeding a `beq` needs the two that
//! compute Z, and paid for the other thirty-odd.
//!
//! MEASURED before this pass existed: 108.3 M wasm operators of software fuel per guest
//! frame on the race window, against a Vita CPU that retires about 7.4 M cycles in the
//! same wall time. The expansion factor is where the guest CPU cost lives, and flags are
//! the biggest term in it.
//!
//! # What this pass is, and what it deliberately is not
//! This is **dead-flag elimination**, not lazy flags. It does not defer a computation to
//! its consumer, keep a pending-operation descriptor, or change any runtime state: it
//! runs a backward liveness analysis over the four flags and tells each writer which of
//! them can be observed. A flag that no reachable read can observe is not computed, and
//! its global keeps whatever it held. Everything else is emitted exactly as before, so
//! the emitted code for a flag that IS live is byte-for-byte what it was.
//!
//! That distinction matters because the flags are also part of the guest state the HOST
//! can see (they are exported globals, saved and restored around a host call and a
//! thread suspend). Lazy flags would have to materialise on every one of those
//! boundaries. This pass simply treats every boundary as a reader:
//!
//! - a **host call** (`Import`/`Svc`) reads all four,
//! - a **guest call** (`Call`/`CallIndirect`) reads all four - the callee may test a flag
//!   its caller set, and nothing here proves it does not,
//! - a **return** leaves all four live, because the caller may test them,
//! - and a terminator whose successor is not a block of this function is treated as a
//!   return, because control has left.
//!
//! [`stmt_effect`] matches every statement kind explicitly - **no wildcard arm** - so a
//! kind added later fails to compile until someone decides what it does to the flags.
//! That is deliberate: a wildcard would silently give a new flag WRITER the "reads
//! nothing, writes nothing" treatment, and an unrecorded read is the one direction of
//! this analysis that miscompiles. (An unrecorded WRITE is harmless - it only keeps a
//! live range longer than needed.)
//!
//! # The oracle this has to survive
//! `vitaslop-conformance-harness` compares the FINAL register file and NZCV of each case
//! against the hardware-derived expectation, and every case ends in a return - where all
//! four flags are live. So the last writer on every path out of a function still computes
//! everything, and only writers whose result is provably overwritten or ignored are
//! trimmed. The retail check is stronger still: the headless race render must stay
//! bit-identical, and a wrong flag changes guest control flow within a few frames.

use crate::abi::Flag;
use crate::ir::{ConditionCode, Func, NeonStmt, Stmt, Term, Value, VfpOp};
#[cfg(test)]
use crate::ir::Block;

/// A set of condition flags, one bit per [`Flag`].
#[derive(Clone, Copy, PartialEq, Eq, Debug, Default)]
pub struct FlagMask(u8);

impl FlagMask {
    pub const NONE: FlagMask = FlagMask(0);
    pub const ALL: FlagMask = FlagMask(0b1111);

    pub const fn of(f: Flag) -> FlagMask {
        FlagMask(1 << f as u8)
    }

    pub const fn has(self, f: Flag) -> bool {
        self.0 & (1 << f as u8) != 0
    }

    pub const fn is_empty(self) -> bool {
        self.0 == 0
    }

    pub const fn union(self, other: FlagMask) -> FlagMask {
        FlagMask(self.0 | other.0)
    }

    pub const fn without(self, other: FlagMask) -> FlagMask {
        FlagMask(self.0 & !other.0)
    }
}

/// The flags a condition code tests.
///
/// This is the whole reason the pass pays: no condition code reads more than two flags,
/// and half of them read exactly one. `AL` reads none - an unconditional "condition".
pub fn cond_reads(cond: ConditionCode) -> FlagMask {
    use ConditionCode::*;
    const N: FlagMask = FlagMask::of(Flag::N);
    const Z: FlagMask = FlagMask::of(Flag::Z);
    const C: FlagMask = FlagMask::of(Flag::C);
    const V: FlagMask = FlagMask::of(Flag::V);
    match cond {
        EQ | NE => Z,
        HS | LO => C,
        MI | PL => N,
        VS | VC => V,
        HI | LS => C.union(Z),
        GE | LT => N.union(V),
        GT | LE => N.union(V).union(Z),
        AL => FlagMask::NONE,
    }
}

/// The flags read while evaluating a value expression.
///
/// `Value::Flag` is the only leaf that reads one - it is how `adc`/`sbc` take their
/// carry-in - but it can sit at any depth, so every operand is walked.
fn value_reads(v: &Value) -> FlagMask {
    match v {
        Value::Flag(f) => FlagMask::of(*f),
        Value::Imm(_) | Value::Reg(_) | Value::ThreadPtr | Value::CarryAddResult => FlagMask::NONE,
        Value::Not(a) | Value::Clz(a) | Value::Load { addr: a, .. } => value_reads(a),
        Value::Bin(_, a, b) => value_reads(a).union(value_reads(b)),
    }
}

/// What a statement does to the flags: what it READS while executing, and what it
/// definitely WRITES.
///
/// Only statements whose flag behaviour is certain are modelled; everything else falls to
/// the conservative default of "reads all, writes nothing" (see the module docs). A
/// [`Stmt::Guard`] is not here at all - it is a conditional statement LIST, so it is
/// handled by the walk itself rather than by a single read/write pair.
fn stmt_effect(s: &Stmt) -> (FlagMask, FlagMask) {
    const N: FlagMask = FlagMask::of(Flag::N);
    const Z: FlagMask = FlagMask::of(Flag::Z);
    const C: FlagMask = FlagMask::of(Flag::C);
    match s {
        Stmt::SetReg(_, v) | Stmt::SetThreadPtr(v) | Stmt::Rbit { rm: v, .. } => {
            (value_reads(v), FlagMask::NONE)
        }
        Stmt::Store { addr, data, .. } => {
            (value_reads(addr).union(value_reads(data)), FlagMask::NONE)
        }
        Stmt::MulLong { rn, rm, .. } => (value_reads(rn).union(value_reads(rm)), FlagMask::NONE),
        // Writes all four. Its own operands may read C (the carry-in of `adc`/`sbc`), and
        // that read happens BEFORE the write - the walk below applies them in that order.
        Stmt::FlagsAdd { a, b, cin, .. } => (
            value_reads(a).union(value_reads(b)).union(value_reads(cin)),
            FlagMask::ALL,
        ),
        // N and Z always; C only when a shifter carry-out is supplied. V is left alone,
        // which is ARM's own rule for the logical group and is why it is not written here.
        Stmt::FlagsLogic { value, carry, .. } => {
            let reads = value_reads(value)
                .union(carry.as_ref().map_or(FlagMask::NONE, value_reads));
            let writes = if carry.is_some() { N.union(Z).union(C) } else { N.union(Z) };
            (reads, writes)
        }
        // A register-controlled shift writes N, Z and C when it sets flags - and READS the
        // old C, because a shift by zero leaves the carry untouched and the emitted code
        // selects between them. Missing that read would be a real miscompile, which is why
        // it is spelled out rather than inherited from the default.
        Stmt::ShiftRegFlags { rn, amount, set_flags, .. } => {
            let operands = value_reads(rn).union(value_reads(amount));
            if *set_flags {
                (operands.union(C), N.union(Z).union(C))
            } else {
                (operands, FlagMask::NONE)
            }
        }
        // `vmrs APSR_nzcv, fpscr` overwrites all four from the FP flags and reads none of
        // them. This is the one write that is not an arithmetic flag statement, and
        // recording it lets a comparison whose flags this clobbers be dropped entirely.
        Stmt::Vfp(VfpOp::MrsNzcv) => (FlagMask::NONE, FlagMask::ALL),
        // Every other VFP op works on the FP register file and the FP flags, neither of
        // which is the integer NZCV. None of them carries a `Value`, so none can read a
        // flag either.
        Stmt::Vfp(_) => (FlagMask::NONE, FlagMask::NONE),
        Stmt::VfpMem { addr, .. } => (value_reads(addr), FlagMask::NONE),
        // Of the NEON statements only the element load/store carries an address
        // expression; the rest are register-to-register.
        Stmt::Neon(NeonStmt::ElemMem { addr, .. }) => (value_reads(addr), FlagMask::NONE),
        Stmt::Neon(_) => (FlagMask::NONE, FlagMask::NONE),
        // `uadd8` writes the APSR GE bits and `sel` reads them. GE is modelled separately
        // from NZCV - it lives in a scratch local, not in a flag global - so neither
        // touches anything here. Both take register numbers, not expressions.
        Stmt::Uadd8 { .. } | Stmt::Sel { .. } => (FlagMask::NONE, FlagMask::NONE),
        // Everything that leaves this function - a host call, a guest call, an svc - is
        // treated as reading all four. The callee is not analysed, the host marshals the
        // flags as part of the register file, and a wrong answer here is a miscompile
        // rather than a slow frame.
        Stmt::Import(_) | Stmt::Svc(_) | Stmt::Call { .. } => (FlagMask::ALL, FlagMask::NONE),
        Stmt::CallIndirect { addr, .. } => (FlagMask::ALL.union(value_reads(addr)), FlagMask::NONE),
        // Never reached: `walk_back` handles a guard before calling this, because a guard
        // is a statement LIST and one read/write pair cannot describe it. Spelled out
        // rather than folded into a wildcard so the match stays exhaustive.
        Stmt::Guard(..) => (FlagMask::ALL, FlagMask::NONE),
    }
}

/// What a terminator does: the flags it reads, the blocks control may reach from it inside
/// this function, and whether control may LEAVE the function here.
///
/// # Leaving is the conservative case, and every way of leaving counts
/// A block that can transfer control anywhere this pass cannot see has all four flags live
/// at its end. That covers the obvious `Return`, and three cases that are easy to get
/// wrong:
///
/// - **`Halt`.** It is not only `b .`. It is also a no-return `svc` - where the HOST reads
///   the register file, flags included - and an UNDECODABLE TAIL, where execution simply
///   stops with the guest state standing. The conformance corpus is made of exactly the
///   last kind: a case is one instruction with no terminator, so its block ends in `Halt`
///   and the harness then reads NZCV. Treating `Halt` as "no reader" dropped the flags of
///   every single-instruction case, which is how this was caught.
/// - **`Unreachable`.** A trap. State at a trap is what a person reads to find out why.
/// - **A branch to an address that is not a block of this function.** Control has left by
///   an edge this analysis cannot follow, whatever the terminator's kind.
fn term_effect(func: &Func, b: usize, term: &Term) -> (FlagMask, Vec<usize>, bool) {
    let idx = |addr: u32| func.block_index(addr);
    // Fall-through goes to the next block in address order, which is the order the
    // emitter lays them out in. A last block that falls through runs off the end of the
    // function, which is a way of leaving it.
    let next = if b + 1 < func.blocks.len() { vec![b + 1] } else { Vec::new() };
    let leaves_by_fallthrough = next.is_empty();
    match term {
        Term::Fallthrough => (FlagMask::NONE, next, leaves_by_fallthrough),
        Term::Jump(t) => {
            let succ: Vec<usize> = idx(*t).into_iter().collect();
            let leaves = succ.is_empty();
            (FlagMask::NONE, succ, leaves)
        }
        Term::Branch { cond, taken } => {
            let resolved = idx(*taken);
            let leaves = leaves_by_fallthrough || resolved.is_none();
            let mut succ = next;
            succ.extend(resolved);
            (cond_reads(*cond), succ, leaves)
        }
        Term::BranchZero { taken, .. } => {
            let resolved = idx(*taken);
            let leaves = leaves_by_fallthrough || resolved.is_none();
            let mut succ = next;
            succ.extend(resolved);
            (FlagMask::NONE, succ, leaves)
        }
        Term::Switch { index, targets, default } => {
            let resolved: Vec<usize> = targets.iter().filter_map(|t| idx(*t)).collect();
            let mut leaves = resolved.len() != targets.len();
            let mut succ = resolved;
            if let Some(d) = default {
                match idx(*d) {
                    Some(i) => succ.push(i),
                    None => leaves = true,
                }
            }
            (value_reads(index), succ, leaves)
        }
        Term::Return | Term::Halt | Term::Unreachable => (FlagMask::NONE, Vec::new(), true),
    }
}

/// Walk a statement list backwards, returning the flags live BEFORE it given the flags
/// live after it. With `annotate` set, each flag statement's `live` field is updated to
/// the flags live immediately after it.
fn walk_back(stmts: &mut [Stmt], live_after: FlagMask, annotate: bool) -> FlagMask {
    let mut live = live_after;
    for s in stmts.iter_mut().rev() {
        // A guard is `if cond { inner }`, so control either runs `inner` or steps over it.
        // Both paths reach `live`, so the flags live before the guard are the UNION of the
        // two - and a writer inside the guard sees the guard's own live-after, because if
        // it does not run the old value survives to the same readers.
        if let Stmt::Guard(cond, inner) = s {
            let through_inner = walk_back(inner, live, annotate);
            live = live.union(through_inner).union(cond_reads(*cond));
            continue;
        }
        let (reads, writes) = stmt_effect(s);
        if annotate {
            match s {
                Stmt::FlagsAdd { live: slot, .. }
                | Stmt::FlagsLogic { live: slot, .. }
                | Stmt::ShiftRegFlags { live: slot, .. } => *slot = live,
                _ => {}
            }
        }
        live = live.without(writes).union(reads);
    }
    live
}

/// Whether a statement is guaranteed to leave all four condition flags exactly as it found
/// them.
///
/// Stricter than "writes nothing" from [`stmt_effect`], and deliberately so: that function
/// reports a CALL as writing nothing because for liveness the conservative direction is to
/// keep flags alive across it. Here the conservative direction is the opposite - a callee
/// may leave the flags anywhere - so calls are excluded by name. A nested guard is excluded
/// too, since its body is not examined.
fn cannot_change_flags(s: &Stmt) -> bool {
    match s {
        Stmt::Import(_) | Stmt::Svc(_) | Stmt::Call { .. } | Stmt::CallIndirect { .. } => false,
        Stmt::Guard(..) => false,
        other => stmt_effect(other).1.is_empty(),
    }
}

/// Merge adjacent [`Stmt::Guard`]s that share a condition into one.
///
/// # Why there are so many of them
/// Thumb-2 predicates with `IT`, which covers up to FOUR following instructions with one
/// condition, and lowering makes each of them its own guard. The emitter turns every guard
/// into `<condition>; if ... end`, so a four-instruction `IT` block evaluated the same
/// condition four times and opened four `if` frames. The condition is between one operator
/// (`eq`, `mi`) and five (`gt`, `le`), so that is 8 to 24 operators where 2 to 6 will do.
///
/// # When it is legal
/// Only when the first guard's body cannot change the flags - otherwise the second
/// condition, which the original evaluates AFTER those writes, would be evaluated before
/// them and could take the other arm. In practice the guest almost always allows it,
/// because Thumb suppresses flag-setting inside an `IT` block, which is the case that
/// produces these runs in the first place; lowering already tracks that as `in_it`.
///
/// The merge is a no-op on ARM-mode code, where predication is per instruction and adjacent
/// predicated instructions with the same condition are far less common.
pub fn merge_guards(stmts: &mut Vec<Stmt>) {
    // Depth first: an inner list is merged before the outer one looks at it.
    for s in stmts.iter_mut() {
        if let Stmt::Guard(_, inner) = s {
            merge_guards(inner);
        }
    }
    let mut i = 0;
    while i + 1 < stmts.len() {
        let mergeable = match (&stmts[i], &stmts[i + 1]) {
            (Stmt::Guard(a, body), Stmt::Guard(b, _)) => {
                a == b && body.iter().all(cannot_change_flags)
            }
            _ => false,
        };
        if !mergeable {
            i += 1;
            continue;
        }
        let Stmt::Guard(_, next_body) = stmts.remove(i + 1) else { unreachable!() };
        let Stmt::Guard(_, body) = &mut stmts[i] else { unreachable!() };
        body.extend(next_body);
        // Do not advance: the merged guard may absorb the one after it too.
    }
}

/// Annotate every flag-setting statement in `func` with the flags that can be observed
/// after it, so the emitter can skip computing the rest.
///
/// A backward liveness fixpoint over the block graph. The lattice is four bits and the
/// transfer functions are monotone, so it converges; the loop bound below is a guard
/// against a malformed graph rather than a real limit (the analysis settles in a couple
/// of passes on real functions, one per loop nesting level).
pub fn annotate(func: &mut Func) {
    let n = func.blocks.len();
    if n == 0 {
        return;
    }
    // Flags live at each block's ENTRY. Starts empty and only grows, which is what makes
    // the fixpoint safe to stop at: the first pass that adds nothing is the answer.
    let mut live_in = vec![FlagMask::NONE; n];
    // Precomputed per block, because the CFG does not change during the fixpoint and
    // `block_index` is a linear scan.
    let effects: Vec<(FlagMask, Vec<usize>, bool)> = func
        .blocks
        .iter()
        .enumerate()
        .map(|(b, blk)| term_effect(func, b, &blk.term))
        .collect();

    // The loop runs at most once per block plus one: each pass either grows some
    // `live_in` (and there are only 4n bits to grow) or terminates.
    for _ in 0..=n {
        let mut changed = false;
        // Backwards over the blocks: on a reducible graph in address order this reaches
        // the fixpoint in far fewer passes than a forward sweep would.
        for b in (0..n).rev() {
            let (term_reads, ref succ, leaves) = effects[b];
            let mut live = if leaves { FlagMask::ALL } else { FlagMask::NONE };
            for &s in succ {
                live = live.union(live_in[s]);
            }
            live = live.union(term_reads);
            let entry = walk_back(&mut func.blocks[b].stmts, live, false);
            if entry != live_in[b] {
                live_in[b] = entry;
                changed = true;
            }
        }
        if !changed {
            break;
        }
    }

    // The answer is settled; one more backward walk per block writes it into the
    // statements. Separate from the fixpoint so a statement is never annotated from a
    // half-converged state.
    for b in 0..n {
        let (term_reads, ref succ, leaves) = effects[b];
        let mut live = if leaves { FlagMask::ALL } else { FlagMask::NONE };
        for &s in succ {
            live = live.union(live_in[s]);
        }
        live = live.union(term_reads);
        walk_back(&mut func.blocks[b].stmts, live, true);
    }
}

/// The flags live after each statement of a single block, for tests. Mirrors what
/// [`annotate`] writes, without needing a whole `Func`.
#[cfg(test)]
fn annotate_block(block: &mut Block, live_out: FlagMask) -> FlagMask {
    walk_back(&mut block.stmts, live_out, true)
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::ir::{BinOp, MemSize};

    fn add(live: FlagMask) -> Stmt {
        Stmt::FlagsAdd { a: Value::Reg(0), b: Value::Reg(1), cin: Value::Imm(0), live }
    }

    /// A block for these tests. `arm_count` is the clock's unit of guest work and has no
    /// bearing on liveness, so it is fixed at one rather than being spelled out per block.
    fn blk(addr: u32, stmts: Vec<Stmt>, term: Term) -> Block {
        Block { addr, stmts, term, arm_count: 1 }
    }

    fn live_of(s: &Stmt) -> FlagMask {
        match s {
            Stmt::FlagsAdd { live, .. }
            | Stmt::FlagsLogic { live, .. }
            | Stmt::ShiftRegFlags { live, .. } => *live,
            _ => panic!("not a flag statement"),
        }
    }

    /// A `cmp` and a conditional branch whose two arms each overwrite the flags before
    /// leaving. Both arms are built that way on purpose: a block that reaches the end of
    /// the function without writing the flags keeps all four live (see
    /// [`a_terminal_halt_keeps_every_flag_live`]), which would mask what this is testing.
    fn compare_then_branch(cond: ConditionCode) -> Func {
        Func {
            addr: 0,
            thumb: true,
            stub: false,
            blocks: vec![
                blk(0, vec![add(FlagMask::ALL)], Term::Branch { cond, taken: 8 }),
                blk(4, vec![add(FlagMask::ALL)], Term::Halt),
                blk(8, vec![add(FlagMask::ALL)], Term::Halt),
            ],
        }
    }

    /// The shape the whole pass exists for: `cmp` then `beq`. Only Z can be observed, so
    /// the carry and the overflow - seventeen of the forty operators - are not computed.
    #[test]
    fn a_compare_feeding_an_equality_branch_needs_only_z() {
        let mut f = compare_then_branch(ConditionCode::EQ);
        annotate(&mut f);
        assert_eq!(live_of(&f.blocks[0].stmts[0]), FlagMask::of(Flag::Z));
    }

    /// `lt` reads N and V and nothing else, so the carry is still dropped. This is the
    /// case that proves the mask is per-CONDITION and not a single "some flag is read".
    #[test]
    fn a_signed_branch_needs_n_and_v_but_not_c() {
        let mut f = compare_then_branch(ConditionCode::LT);
        annotate(&mut f);
        let want = FlagMask::of(Flag::N).union(FlagMask::of(Flag::V));
        assert_eq!(live_of(&f.blocks[0].stmts[0]), want);
    }

    /// `Halt` is not only `b .`. It is also a no-return `svc` - where the host reads the
    /// register file - and an UNDECODABLE TAIL, which is what every conformance case is:
    /// one instruction, no terminator, and the harness then compares NZCV. Treating a halt
    /// as unobservable dropped the flags of every single-instruction case in the corpus,
    /// and this is the guard for that. It was confirmed to FAIL against the first version
    /// of this pass.
    #[test]
    fn a_terminal_halt_keeps_every_flag_live() {
        let mut f = Func {
            addr: 0,
            thumb: true,
            stub: false,
            blocks: vec![blk(0, vec![add(FlagMask::ALL)], Term::Halt)],
        };
        annotate(&mut f);
        assert_eq!(live_of(&f.blocks[0].stmts[0]), FlagMask::ALL);
    }

    /// A branch to an address that is not a block of this function is a way OUT of it,
    /// whatever the terminator's kind says.
    #[test]
    fn a_branch_leaving_the_function_keeps_every_flag_live() {
        let mut f = Func {
            addr: 0,
            thumb: true,
            stub: false,
            blocks: vec![
                // 0x1000 is not a block here.
                blk(
                    0,
                    vec![add(FlagMask::ALL)],
                    Term::Branch { cond: ConditionCode::EQ, taken: 0x1000 },
                ),
                blk(4, vec![add(FlagMask::ALL)], Term::Halt),
            ],
        };
        annotate(&mut f);
        assert_eq!(live_of(&f.blocks[0].stmts[0]), FlagMask::ALL);
    }

    /// A flag write nobody can observe costs nothing at all: the second `cmp` overwrites
    /// every flag the first one set, before anything reads one.
    #[test]
    fn a_compare_overwritten_before_any_read_computes_nothing() {
        let mut b = blk(0, vec![add(FlagMask::ALL), add(FlagMask::ALL)], Term::Halt);
        annotate_block(&mut b, FlagMask::NONE);
        assert_eq!(live_of(&b.stmts[0]), FlagMask::NONE);
        assert_eq!(live_of(&b.stmts[1]), FlagMask::NONE);
    }

    /// A return is a reader. The caller may test what this function left behind, and this
    /// pass cannot see the caller - so the last writer on the way out computes all four.
    #[test]
    fn a_return_keeps_every_flag_live() {
        let mut f = Func {
            addr: 0,
            thumb: true,
            stub: false,
            blocks: vec![blk(0, vec![add(FlagMask::ALL)], Term::Return)],
        };
        annotate(&mut f);
        assert_eq!(live_of(&f.blocks[0].stmts[0]), FlagMask::ALL);
    }

    /// A host call is a reader too: the flags are part of the register file the host
    /// marshals, and the callee is not analysed.
    #[test]
    fn a_host_call_keeps_every_flag_live() {
        let mut b = blk(0, vec![add(FlagMask::ALL), Stmt::Import(3)], Term::Halt);
        annotate_block(&mut b, FlagMask::NONE);
        assert_eq!(live_of(&b.stmts[0]), FlagMask::ALL);
    }

    /// `adc` takes the carry IN. The read happens before the write, so a preceding writer
    /// must still produce C even though `FlagsAdd` overwrites it.
    #[test]
    fn a_carry_in_is_a_read_of_the_previous_writers_carry() {
        let mut b = blk(
            0,
            vec![
                add(FlagMask::ALL),
                Stmt::FlagsAdd {
                    a: Value::Reg(0),
                    b: Value::Reg(1),
                    cin: Value::Flag(Flag::C),
                    live: FlagMask::ALL,
                },
            ],
            Term::Halt,
        );
        annotate_block(&mut b, FlagMask::NONE);
        assert_eq!(live_of(&b.stmts[0]), FlagMask::of(Flag::C));
        assert_eq!(live_of(&b.stmts[1]), FlagMask::NONE);
    }

    /// A logical operation never writes V, so it cannot kill a V an earlier compare set
    /// for a later `vs`.
    #[test]
    fn a_logical_write_does_not_kill_the_overflow_flag() {
        let mut b = blk(
            0,
            vec![
                add(FlagMask::ALL),
                Stmt::FlagsLogic { value: Value::Reg(2), carry: None, live: FlagMask::ALL },
            ],
            Term::Branch { cond: ConditionCode::VS, taken: 0 },
        );
        // Live-out of the block includes what the terminator reads; pass it in directly.
        annotate_block(&mut b, cond_reads(ConditionCode::VS));
        assert_eq!(live_of(&b.stmts[0]), FlagMask::of(Flag::V));
    }

    /// A guarded write is conditional, so the value it might not overwrite has to survive
    /// it. The earlier writer stays live even though the guarded one writes everything.
    #[test]
    fn a_guarded_write_does_not_kill_an_earlier_one() {
        let mut b = blk(
            0,
            vec![
                add(FlagMask::ALL),
                Stmt::Guard(ConditionCode::NE, vec![add(FlagMask::ALL)]),
            ],
            Term::Halt,
        );
        annotate_block(&mut b, FlagMask::of(Flag::Z));
        // The guard's own condition reads Z, and Z is live out, so the first write's Z is
        // observable on the path where the guard does not run.
        assert_eq!(live_of(&b.stmts[0]), FlagMask::of(Flag::Z));
    }

    /// A loop's back edge carries liveness backwards into the body. A compare at the top
    /// of a loop whose branch is at the bottom must survive the whole body.
    #[test]
    fn liveness_flows_around_a_back_edge() {
        let mut f = Func {
            addr: 0,
            thumb: true,
            stub: false,
            blocks: vec![
                // 0: the loop header, which sets flags and falls through.
                blk(0, vec![add(FlagMask::ALL)], Term::Fallthrough),
                // 4: the body, which does something flag-free and branches back on Z.
                blk(
                    4,
                    vec![Stmt::SetReg(
                        0,
                        Value::Bin(BinOp::Add, Box::new(Value::Reg(0)), Box::new(Value::Imm(1))),
                    )],
                    Term::Branch { cond: ConditionCode::EQ, taken: 0 },
                ),
                // The exit block overwrites the flags, so nothing flows back from the end
                // of the function and the back edge is the only thing under test.
                blk(8, vec![add(FlagMask::ALL)], Term::Halt),
            ],
        };
        annotate(&mut f);
        assert_eq!(live_of(&f.blocks[0].stmts[0]), FlagMask::of(Flag::Z));
    }

    /// A Thumb `IT` block: four instructions, one condition, four guards - and one `if`
    /// after the merge.
    #[test]
    fn a_run_of_same_condition_guards_becomes_one() {
        let mut stmts = vec![
            Stmt::Guard(ConditionCode::EQ, vec![Stmt::SetReg(0, Value::Imm(1))]),
            Stmt::Guard(ConditionCode::EQ, vec![Stmt::SetReg(1, Value::Imm(2))]),
            Stmt::Guard(ConditionCode::EQ, vec![Stmt::SetReg(2, Value::Imm(3))]),
            Stmt::Guard(ConditionCode::NE, vec![Stmt::SetReg(3, Value::Imm(4))]),
        ];
        merge_guards(&mut stmts);
        assert_eq!(stmts.len(), 2, "the three EQ guards merge, the NE one does not");
        match &stmts[0] {
            Stmt::Guard(ConditionCode::EQ, body) => assert_eq!(body.len(), 3),
            other => panic!("expected a merged EQ guard, got {other:?}"),
        }
    }

    /// A guard whose body SETS the flags must not absorb the next one: the original
    /// evaluates the second condition after those writes, and the merge would evaluate it
    /// before them.
    #[test]
    fn a_guard_that_writes_flags_does_not_absorb_the_next() {
        let mut stmts = vec![
            Stmt::Guard(ConditionCode::EQ, vec![add(FlagMask::ALL)]),
            Stmt::Guard(ConditionCode::EQ, vec![Stmt::SetReg(1, Value::Imm(2))]),
        ];
        merge_guards(&mut stmts);
        assert_eq!(stmts.len(), 2);
    }

    /// A call may leave the flags anywhere, so it blocks the merge for the same reason -
    /// and `stmt_effect` alone would NOT say so, because for liveness a call is modelled
    /// as writing nothing.
    #[test]
    fn a_guard_containing_a_call_does_not_absorb_the_next() {
        let mut stmts = vec![
            Stmt::Guard(ConditionCode::EQ, vec![Stmt::Call { target: 0x1000 }]),
            Stmt::Guard(ConditionCode::EQ, vec![Stmt::SetReg(1, Value::Imm(2))]),
        ];
        merge_guards(&mut stmts);
        assert_eq!(stmts.len(), 2);
    }

    /// A load in an address expression cannot hide a flag read from the walk.
    #[test]
    fn a_flag_read_nested_in_an_address_is_found() {
        let v = Value::Load {
            addr: Box::new(Value::Bin(
                BinOp::Add,
                Box::new(Value::Reg(1)),
                Box::new(Value::Flag(Flag::C)),
            )),
            size: MemSize::Word,
            signed: false,
        };
        assert_eq!(value_reads(&v), FlagMask::of(Flag::C));
    }
}
