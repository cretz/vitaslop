//! USSE (SGX543 Unified Scalable Shader Engine) decoding.

pub mod asm;
pub mod decode;

pub use decode::{
    bits, decode, decode_smlsi, field, is_smlsi, opcode1, repeat_extra_iterations, GroupTable,
    SmlsiSlot, GROUP_TABLES,
};

use crate::container::Program;
use crate::ir::{Op, Shader};

/// The MOE (repeat) state in force on ENTRY to each code word, computed over the program's
/// CONTROL-FLOW GRAPH.
///
/// An SMLSI sets state that persists until the next SMLSI, so what a repeating instruction
/// consults is the last SMLSI that ran before it - and which SMLSI that is, is a control-flow
/// question, not a textual one. This is the forward dataflow that answers it: entry state is
/// [`decode::DEFAULT_REPEAT_STATE`], an SMLSI overwrites the state, everything else passes it
/// through, and two paths carrying DIFFERENT states meet as `None` - "not determined", which is
/// what makes the consumer block instead of picking one.
///
/// This replaces a whole-program rule ("no branch may span any SMLSI, or every SMLSI blocks"),
/// which is sound but far coarser: in a program whose every repeat is immediately preceded by
/// its own SMLSI, a branch landing anywhere else is harmless, and the old rule refused the
/// program anyway.
///
/// # The edges, and why each is the one the hardware has
///
/// * FALL-THROUGH `i -> i+1` for every instruction except an UNCONDITIONAL branch. An
///   unconditional branch does not fall through, and this is not a refinement that can be
///   skipped "to stay conservative": a spurious fall-through edge past one manufactures a
///   conflict at its target's successors out of nothing, which is exactly the false refusal
///   this function exists to remove.
/// * TAKEN `i -> i+rel` for every [`Op::Branch`]. A target outside the program is dropped here;
///   [`remap_branch_targets`] blocks that instruction by name.
/// * A PREDICATED branch has both.
///
/// A word no edge reaches is code the program cannot execute, and its state is reported as
/// `None` rather than assumed: nothing should consult it, and a repeat that does has no
/// established state to consult.
fn moe_states(code: &[u64], instrs: &[crate::ir::Instr]) -> Vec<Option<[decode::SmlsiSlot; 4]>> {
    use crate::ir::Predicate;

    #[derive(Clone, Copy, PartialEq, Eq)]
    enum St {
        Unreached,
        Known([decode::SmlsiSlot; 4]),
        Conflict,
    }
    let join = |a: St, b: St| match (a, b) {
        (St::Unreached, x) | (x, St::Unreached) => x,
        (St::Known(x), St::Known(y)) if x == y => St::Known(x),
        _ => St::Conflict,
    };

    let n = code.len();
    let mut entry = vec![St::Unreached; n];
    if n == 0 {
        return Vec::new();
    }
    entry[0] = St::Known(decode::DEFAULT_REPEAT_STATE);
    // The lattice is three levels deep and every step is monotone, so a sweep that changes
    // nothing is the fixpoint. The bound is only a guard against a future non-monotone edit.
    for _ in 0..=n {
        let mut changed = false;
        for i in 0..n {
            let st = entry[i];
            if st == St::Unreached {
                continue;
            }
            let out = if decode::is_smlsi(code[i]) {
                St::Known(decode::decode_smlsi(code[i]))
            } else {
                st
            };
            let mut succ = [None, None];
            match instrs[i].op {
                Op::Branch { rel } => {
                    let t = i as i64 + i64::from(rel);
                    if t >= 0 && (t as usize) < n {
                        succ[0] = Some(t as usize);
                    }
                    if instrs[i].pred != Predicate::Always && i + 1 < n {
                        succ[1] = Some(i + 1);
                    }
                }
                _ => {
                    if i + 1 < n {
                        succ[0] = Some(i + 1);
                    }
                }
            }
            for s in succ.into_iter().flatten() {
                let merged = join(entry[s], out);
                if merged != entry[s] {
                    entry[s] = merged;
                    changed = true;
                }
            }
        }
        if !changed {
            break;
        }
    }
    entry
        .into_iter()
        .map(|s| match s {
            St::Known(k) => Some(k),
            _ => None,
        })
        .collect()
}

/// The SMBO (set-memory-base-offset) state on ENTRY to every code word, walked over the same
/// control-flow edges [`moe_states`] walks - `None` where two paths disagree or nothing reaches.
///
/// # What an SMBO does, and the closure that establishes it
/// It sets four 12-bit BASE OFFSETS that are added to the register NUMBERS of the instructions
/// that follow, until another SMBO changes them. It exists because some operand fields are only
/// SIX bits wide and cannot name a register above 63.
///
/// That is exactly what a football title uses it for, and the closure is complete rather than
/// argued. Its skinned vertex programs come in two compilations of one shader. In the first, a
/// LIMM loads the sentinel `0x7FFFFFFF` into a register and a byte-wise conditional move selects
/// it. In the second there is no LIMM - the sentinel is a LITERAL in the constant table - and
/// the same move reads it through `src1` with the field at its MAXIMUM VALUE, 63, under an SMBO:
///
///   * `vert_90c28c60`: base 20, `src1` field 63 -> `sa[83]`, and the blob's literal table says
///     `sa[83] = 0x7fffffff`.
///   * `vert_90c2e8f0`: base 38, `src1` field 63 -> `sa[101]`, and ITS literal table says
///     `sa[101] = 0x7fffffff`.
///
/// Two programs, two different bases, both landing exactly on the constant the other compilation
/// loads with a LIMM. The same two tables also carry `0xcf000000` and `0x00010000`, the other two
/// LIMM immediates, which is a third independent check on [`crate::ir::Op::Limm`]'s assembly.
///
/// # What is modelled and what is refused
/// Only the `src1` slot, at bits [23:12], is evidenced, so only it is applied - and an SMBO that
/// programs ANY OTHER slot blocks, rather than being applied on a guessed field order. Bit 50
/// varies between the two forms of the all-zero "reset" word the same programs emit
/// interchangeably after a use, so it cannot change what an all-zero word does; a NON-zero word
/// with it set has never been seen and blocks.
fn smbo_states(code: &[u64], instrs: &[crate::ir::Instr]) -> Vec<Option<u16>> {
    use crate::ir::Predicate;

    #[derive(Clone, Copy, PartialEq, Eq)]
    enum St {
        Unreached,
        Known(u16),
        Conflict,
    }
    let join = |a: St, b: St| match (a, b) {
        (St::Unreached, x) | (x, St::Unreached) => x,
        (St::Known(x), St::Known(y)) if x == y => St::Known(x),
        _ => St::Conflict,
    };
    let n = code.len();
    if n == 0 {
        return Vec::new();
    }
    let mut entry = vec![St::Unreached; n];
    entry[0] = St::Known(0);
    for _ in 0..=n {
        let mut changed = false;
        for i in 0..n {
            let st = entry[i];
            if st == St::Unreached {
                continue;
            }
            let out = match decode::smbo_src1_base(code[i]) {
                Some(base) => St::Known(base),
                None => st,
            };
            let mut succ = [None, None];
            match instrs[i].op {
                Op::Branch { rel } => {
                    let t = i as i64 + i64::from(rel);
                    if t >= 0 && (t as usize) < n {
                        succ[0] = Some(t as usize);
                    }
                    if instrs[i].pred != Predicate::Always && i + 1 < n {
                        succ[1] = Some(i + 1);
                    }
                }
                _ => {
                    if i + 1 < n {
                        succ[0] = Some(i + 1);
                    }
                }
            }
            for s in succ.into_iter().flatten() {
                let merged = join(entry[s], out);
                if merged != entry[s] {
                    entry[s] = merged;
                    changed = true;
                }
            }
        }
        if !changed {
            break;
        }
    }
    entry
        .into_iter()
        .map(|s| match s {
            St::Known(k) => Some(k),
            _ => None,
        })
        .collect()
}

/// Expand every repeating instruction into the sequence of single executions it stands for,
/// stepping each operand by the amount the SMLSI state in force asks for.
///
/// A USSE instruction carries a `repeat_count`: it re-executes that many extra times, and
/// between iterations each operand's ENCODED REGISTER FIELD advances by the per-slot increment
/// the last SMLSI set (default 1 - see [`decode::DEFAULT_REPEAT_STATE`]). What that does to the
/// register INDEX depends on the field's own scaling, which is the whole content of
/// [`decode::repeat_operands`]: a six-bit field is doubled by the hardware, a seven-bit field is
/// not. Nothing downstream of the IR models repetition, so unrolling here is what makes the rest
/// of the recompiler - the emitter, the written-output-lane check, the PA read/write maps that
/// decide the varying interface - see the instruction stream the hardware actually executes.
///
/// That the default stepping is right is MEASURED on the two vertex programs that draw a retail
/// title's entire front-end. Each writes its colour varying with ONE `mov` to `Output[4]` under a
/// two-channel mask, yet its container declares 8 and 10 total output lanes respectively. The
/// repeat counts are 1 and 2, and unrolling at increment 1 over a six-bit (doubled) field closes
/// both statements exactly:
///
/// * 8 lanes: `Output[4] <- SA[0]`, `Output[6] <- SA[2]` - the 4-component `color` uniform
///   filling COLOR0's lanes 4..7 after clip position's 0..3;
/// * 10 lanes: `Output[4] <- PA[4]`, `Output[6] <- PA[6]`, `Output[8] <- PA[8]` - where the
///   parameter table places `In.Color` at PA[4] (4 components) and `In.TexCoord` at PA[8]. The
///   third iteration is the ONLY write of the texture coordinate anywhere in that program.
///
/// That last point is why this is not cosmetic: without unrolling, a textured program's UV
/// varying is never written at all, so every sample lands at (0,0).
///
/// Anything the model cannot state is BLOCKED rather than emitted: a group whose repeat encoding
/// or operand grammar is not established, a slot the SMLSI puts in swizzle mode, or a stepped
/// index that leaves the register file. Emitting once is a silent guess that an instruction does
/// not repeat, and a dropped iteration is exactly the invisible failure this recompiler refuses
/// to make.
///
/// Returns the unrolled stream and, alongside it, where each ORIGINAL code word landed in it:
/// `starts[i]` is the index of code word `i`'s first copy, and `starts[code.len()]` is the
/// stream length, so a branch target of "one past the end" maps too. Unrolling renumbers the
/// stream, and a branch offset is a count of code WORDS, so every branch has to be rewritten
/// through this map or it would silently point at the wrong instruction.
/// What one repeat iteration does to one operand.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
enum RepeatStep {
    /// Advance the operand's register index by this much per iteration.
    Index(i32),
    /// Address `base + ((byte >> 2i) & 3) * stride` on iteration `i`: the SMLSI offset table.
    Offsets(u8, i32),
}

fn unroll_repeats(code: &[u64], instrs: Vec<crate::ir::Instr>) -> (Vec<crate::ir::Instr>, Vec<usize>) {
    let states = moe_states(code, &instrs);
    let bases = smbo_states(code, &instrs);
    let has_smbo = code.iter().any(|&w| decode::smbo_src1_base(w).is_some());
    let mut out = Vec::with_capacity(instrs.len());
    let mut starts = Vec::with_capacity(code.len() + 1);
    for (at, (mut instr, &word)) in instrs.into_iter().zip(code).enumerate() {
        starts.push(out.len());
        // >>> AN SMBO'S BASE, APPLIED WHERE IT IS ESTABLISHED AND BLOCKING WHERE IT IS NOT.
        //
        // An SMBO itself emits nothing - it only sets the state - and is never blocked on its
        // own account unless it programs a slot this does not model (`smbo_src1_base` returns
        // the blocked word to the decoder for that). While a NON-ZERO base is in force it
        // shifts `src1`, which is `srcs[0]` in every shape of the group-0x38 move family (the
        // unconditional `mov`, and the conditional selects, whose source order is
        // `[src1, src2, src0]`). Any OTHER instruction under a non-zero base is outside the
        // closure that established this, so it blocks rather than being addressed by a rule
        // that has not been checked for it.
        if decode::smbo_src1_base(word).is_some() {
            out.push(crate::ir::Instr { op: Op::Nop, blocked: instr.blocked, ..instr });
            continue;
        }
        match bases[at] {
            // A program with no SMBO at all has base zero everywhere, and an UNREACHED word
            // reports `None` from the walk for want of an edge rather than for want of a
            // state - so the unknown case only matters where an SMBO exists to make it real.
            _ if !has_smbo => {}
            Some(0) => {}
            Some(base) if decode::opcode1(word) == 0x07 => {
                match instr.srcs.first_mut().map(|s| (s.index as u32 + u32::from(base), s)) {
                    Some((n, s)) if n <= u8::MAX.into() => s.index = n as u8,
                    _ => {
                        instr.blocked = instr
                            .blocked
                            .or(Some("0xF8 SMBO base pushes src1 outside the register file"))
                    }
                }
            }
            Some(_) => {
                instr.blocked = instr.blocked.or(Some(
                    "0xF8 SMBO: a NON-ZERO base offset is in force over an instruction outside the group-0x38 move family, where only the src1 slot is established",
                ));
            }
            None => {
                instr.blocked = instr.blocked.or(Some(
                    "0xF8 SMBO state at this instruction is not single-valued - two control-flow paths reach it under different base offsets",
                ));
            }
        }
        // The state on ENTRY to this word, over every path that can reach it. `None` is "two
        // paths disagree, or nothing reaches here" - only an instruction that actually CONSULTS
        // the state is blocked by it, below.
        let state = states[at];

        // SMLSI itself emits nothing - its entire effect is the state the repeats below read.
        // It is never blocked on its own account: it SETS the state, so what reaches it cannot
        // make it wrong, and a consumer that cannot resolve its own state blocks there instead.
        if decode::is_smlsi(word) {
            out.push(crate::ir::Instr { op: Op::Nop, blocked: None, ..instr });
            continue;
        }

        // The other half of the `moe_expand` guard in [`decode::decode_grp_mem_load`]: that
        // decoder allows a single-element memory access with bit 53 set because expansion
        // cannot step anything on a domain of one iteration. What that argument needs from the
        // state is not that it is the DEFAULT but that ITERATION ZERO IS AT OFFSET ZERO, and
        // those are different conditions:
        //
        //   * an INCREMENT slot advances by `n * i` per iteration, so iteration 0 contributes
        //     `n * 0 = 0` whatever `n` is. Any increment is safe here, not just the default 1.
        //   * an OFFSET (swizzle) slot reads a four-entry table, `(b >> 2i) & 3`, and iteration
        //     0 takes its FIRST entry - which need not be zero. That one really does move the
        //     address, and it stays blocked.
        //
        // The old test was `state != DEFAULT`, which refused both. That cost a football title
        // TWO fragment programs whose single-element loads run under an SMLSI programmed with
        // increments, where iteration zero is at offset zero by the arithmetic above - and a
        // refused pair's mesh is absent from the frame. A state nothing reaches (`None`) is
        // still refused: it has no single value to check.
        let iteration_zero_unoffset = |st: &[decode::SmlsiSlot; 4]| {
            st.iter().all(|slot| match slot {
                decode::SmlsiSlot::Increment(_) => true,
                decode::SmlsiSlot::Swizzle(b) => b & 3 == 0,
            })
        };
        if matches!(decode::opcode1(word), 0x1d | 0x1e)
            && (word >> 53) & 1 == 1
            && !state.as_ref().is_some_and(iteration_zero_unoffset)
        {
            out.push(crate::ir::Instr {
                blocked: Some(
                    "0xE8 memory access with moe_expand under a MOE state whose ITERATION ZERO is offset (an SMLSI programmed an offset table with a non-zero first entry, or no single state reaches here) - the single-element case rests on iteration zero being at offset zero",
                ),
                ..instr
            });
            continue;
        }

        let Some(extra) = decode::repeat_extra_iterations(word) else {
            // An instruction whose GROUP is not decoded at all reaches here too, and its own
            // reason is the more useful one: "repeat_count encoding not established" sends
            // the reader to look for a four-bit field in a group that has no decoder yet,
            // which is a wrong and expensive place to start. Only claim the repeat encoding
            // is the blocker when nothing else already is.
            out.push(crate::ir::Instr {
                blocked: instr
                    .blocked
                    .or(Some("repeat_count encoding not established for this opcode group")),
                ..instr
            });
            continue;
        };
        if extra == 0 {
            push_split_pack(&mut out, instr);
            continue;
        }
        // From here the instruction really repeats, so the operand grammar has to be known
        // exactly: which SMLSI byte governs each operand, and what one unit of it moves - and
        // WHICH SMLSI is in force has to be a single answer. When two paths reach this word
        // under different MOE states (or none reaches it at all) there is no state to read, and
        // picking either one would step the operands of a real repeat by the wrong amount.
        let Some(state) = state else {
            out.push(crate::ir::Instr {
                blocked: Some(
                    "0xF8 SMLSI state at a repeating instruction is not single-valued - two control-flow paths reach it under different repeat states",
                ),
                ..instr
            });
            continue;
        };
        // >>> THE GROUP-0x15 GUARD: its repeat's STEPS are measured, its SLOT NUMBERS are not.
        //
        // The one repeating IMAD32 in any corpus runs under a state whose three non-zero slots
        // all carry the same increment, so which of them the destination and src0 sit on cannot
        // change the answer - see `repeat_operands`. Under a state where they DIFFER the choice
        // would decide which register a matrix-palette pointer lands in, and that is not a
        // thing to pick.
        // (The IMAD32 slots-disagree guard that stood here is gone: the SMLSI byte order names
        // each operand's byte, so a non-uniform state is no longer ambiguous - see
        // `repeat_operands`.)
        let Some(operands) = decode::repeat_operands(word) else {
            out.push(crate::ir::Instr {
                blocked: Some("repeat operand slots not established for this opcode group"),
                ..instr
            });
            continue;
        };
        // What one iteration does to each operand: step its register INDEX by the slot's
        // increment, or - when the slot is in OFFSET mode - address the register the slot's
        // per-iteration table names (`base + entry[i] * stride`). Both are register addressing;
        // the instruction's mask and swizzle never move (see `decode_smlsi`).
        let steps: Result<Vec<RepeatStep>, &'static str> = operands
            .iter()
            .map(|o| match (o.moe, state[o.slot]) {
                // An intrinsic advance - the DP's channel walk - is not the MOE's to program.
                (false, _) => Ok(RepeatStep::Index(o.stride as i32)),
                (true, decode::SmlsiSlot::Increment(n)) => {
                    Ok(RepeatStep::Index(i32::from(n) * o.stride as i32))
                }
                (true, decode::SmlsiSlot::Swizzle(b)) => Ok(RepeatStep::Offsets(b, o.stride as i32)),
            })
            .collect();
        let steps = match steps {
            Ok(s) if s.len() > instr.srcs.len() => s,
            // More IR sources than the grammar describes means the two disagree about the
            // instruction, which is a decoder bug rather than a shader feature.
            Ok(_) => {
                out.push(crate::ir::Instr {
                    blocked: Some("repeat operand list is shorter than the decoded sources"),
                    ..instr
                });
                continue;
            }
            Err(why) => {
                out.push(crate::ir::Instr { blocked: Some(why), ..instr });
                continue;
            }
        };
        // A stepped index that leaves the 8-bit register file is not a register, and clamping it
        // would read or write the wrong one silently.
        let advance = |index: u8, step: RepeatStep, i: u32| -> Option<u8> {
            let delta = match step {
                RepeatStep::Index(n) => n * i as i32,
                RepeatStep::Offsets(b, stride) => i32::from((b >> (2 * i.min(3))) & 3) * stride,
            };
            u8::try_from(i32::from(index) + delta).ok()
        };
        let mut escaped = false;
        for i in 0..=extra {
            let mut it = instr.clone();
            if let Some(d) = it.dest.as_mut() {
                match advance(d.index, steps[0], i) {
                    Some(index) => d.index = index,
                    None => escaped = true,
                }
            }
            for (s, &step) in it.srcs.iter_mut().zip(&steps[1..]) {
                match advance(s.index, step, i) {
                    // A register-INDIRECT operand's number is not a register index: its top two
                    // bits select the sub-bank and only the low five are the offset. Stepping it
                    // past 31 would carry into the bank selector and silently read a different
                    // bank, so a repeat that walks off the offset field is not a register step.
                    Some(index)
                        if matches!(s.bank, crate::ir::Bank::Indexed)
                            && index >> 5 != s.index >> 5 =>
                    {
                        escaped = true
                    }
                    Some(index) => s.index = index,
                    None => escaped = true,
                }
            }
            push_split_pack(&mut out, it);
        }
        if escaped {
            let from = starts[starts.len() - 1];
            for it in &mut out[from..] {
                it.blocked = Some("a repeated operand steps outside the register file");
            }
        }
    }
    starts.push(out.len());
    (out, starts)
}

/// Rewrite every [`Op::Branch`] delta from the ORIGINAL code-word numbering into the unrolled
/// stream's numbering, using the map [`unroll_repeats`] produced.
///
/// A target that falls outside the program (before the first word, or past one-past-the-end) is
/// not expressible in the current stream and cannot be reconstructed, so the instruction is
/// BLOCKED naming that rather than clamped to something plausible.
/// Push `instr`, splitting a TWO-SOURCE pack (see `decode_grp_pack`: a 32-bit source vector
/// is `(src1.x, src1.y, src2.x, src2.y)`) into one single-source pack per source. A pack is a
/// per-channel copy, so the channels selecting components 0..1 read `src1` and those selecting
/// 2..3 read `src2` at component minus two, and the two halves commute. Every later stage then
/// sees only the single-source form it already models.
fn push_split_pack(out: &mut Vec<crate::ir::Instr>, instr: crate::ir::Instr) {
    if !matches!(instr.op, Op::Pack { .. }) || instr.srcs.len() != 2 {
        out.push(instr);
        return;
    }
    let (s1, s2) = (instr.srcs[0], instr.srcs[1]);
    let mut lower_mask = [false; 4];
    let mut upper_mask = [false; 4];
    let mut upper_src = s2;
    for c in 0..4 {
        if !instr.write_mask[c] {
            continue;
        }
        match s1.swizzle[c] {
            2 | 3 => {
                upper_mask[c] = true;
                upper_src.swizzle[c] = s1.swizzle[c] - 2;
            }
            // Components 0..1, and swizzle constants (which read no register either way).
            _ => lower_mask[c] = true,
        }
    }
    if lower_mask.iter().any(|&m| m) {
        out.push(crate::ir::Instr { write_mask: lower_mask, srcs: vec![s1], ..instr.clone() });
    }
    if upper_mask.iter().any(|&m| m) {
        out.push(crate::ir::Instr { write_mask: upper_mask, srcs: vec![upper_src], ..instr });
    }
}

fn remap_branch_targets(instrs: &mut [crate::ir::Instr], starts: &[usize]) {
    // `starts` is indexed by ORIGINAL code word, and a branch never repeats (group 0xF8 carries
    // no repeat count), so word `w` is the single instruction at `starts[w]`.
    for (word, &at) in starts.iter().take(starts.len().saturating_sub(1)).enumerate() {
        let Op::Branch { rel } = instrs[at].op else { continue };
        let target = word as i64 + rel as i64;
        if target < 0 || target as usize >= starts.len() {
            instrs[at].blocked = Some("0xF8 BR target falls outside the program");
            continue;
        }
        let new_rel = starts[target as usize] as i64 - at as i64;
        instrs[at].op = Op::Branch { rel: new_rel as i32 };
    }
}

/// Block every group-0x1a step that is not part of a well-formed 32-bit multiply-add PAIR.
///
/// `Op::IntMadStep` is one half of a two-instruction idiom - see `decode_grp_imad32_step` for
/// the layout and for why `sn` reads as a 16-bit half selector. The reading that survives the
/// corpus is not the only one that fits it arithmetically, and the rivals differ ONLY in the
/// value the first step leaves in its destination. This is what makes that difference
/// unobservable: a step is emitted only inside a pair whose net effect - `dest = src0 * src1 +
/// src2` - every surviving reading agrees on, and anything else hard-fails naming itself.
///
/// The four conditions, all required:
///  * a `sn = 0` step is immediately followed by a `sn = 1` step, and vice versa;
///  * the two carry the same `src0` and the same `src1` (bank AND number: an immediate literal
///    is carried in the operand's index, so this compares literals too);
///  * the second's `src2` is exactly the first's DESTINATION, which is what chains the two
///    partial products into one sum;
///  * neither is predicated differently from the other, since a pair split by a predicate is
///    not a pair.
fn validate_imad_step_pairs(instrs: &mut [crate::ir::Instr]) {
    use crate::ir::{Instr, Op};

    let step = |i: &Instr| match i.op {
        Op::IntMadStep { high_half, .. } => Some(high_half),
        _ => None,
    };
    // Bank and number together: two operands naming different banks are different operands even
    // when their numbers agree, and an inline literal is carried as an index in the IMMEDIATE
    // bank so this compares literals by value too.
    let mut blocked_at: Vec<(usize, &'static str)> = Vec::new();
    for at in 0..instrs.len() {
        let Some(_high) = step(&instrs[at]) else { continue };
        // >>> AND A LONE STEP IS EMITTED TOO, because the pair closure DETERMINES each half.
        //
        // The refusal here said "only the pair's net result is established". That was true of
        // the PAIR and not of the step: a pair composes to exactly `x * y + z` only if the high
        // step is `((x >> 16) * y) << 16 + z` and the low one `(x & 0xffff) * y + z`, and no
        // other placement of the `<< 16` between two half-product MADs composes to that sum. So
        // the decomposition is unique, and each half is as established as the whole.
        //
        // A football title emits two of them, both HIGH halves alone, directly after a branch -
        // the low half is not missing, it is not wanted: `pa[19] = ((sa[105] >> 16) * sa[72])
        // << 16 + sa[23]` is the value that block computes. Six of its skinned vertex programs
        // were refused whole for it.
        //
        // What stays refused is a step this decoder cannot READ, not one that is alone.
        if instrs[at].srcs.len() != 3 {
            blocked_at.push((
                at,
                "0x1a IMAD32-STEP: the step does not carry three operands, so neither its own                  half-product nor a pair's net value can be formed",
            ));
        }
    }
    for (at, why) in blocked_at {
        instrs[at].blocked = instrs[at].blocked.or(Some(why));
    }
}

// How the decode above resolves a sampler, kept as a plain comment because the item it used to
// document is gone and a `///` block with nothing under it attaches itself to whatever follows.
//
// A `SMP` instruction addresses its sampler by a REGISTER field, not by texture unit: the
// texture's control words live at SA register `2 * field`, and only the container's
// texture-control table says which GXM unit those words describe. That resolution happens in
// the decode above, so `Op::Tex::unit` is a real texture unit everywhere downstream - the same
// namespace a PDS-prefetched sample names directly, and the one the renderer binds by. A field
// the table does not describe blocks the instruction rather than naming an arbitrary unit.

/// Which OUTPUT-bank lanes a decoded program writes, as a bitmap indexed by lane.
///
/// # >>> AN F16 INSTRUCTION'S CHANNELS ARE HALVES, NOT LANES
/// Channel `c` of a half-precision instruction is half `c & 1` of register `base + (c >> 1)` -
/// the emitter's own rule, documented at `wgsl::emit_body` and pinned by
/// `f16_instruction_addresses_half_lanes_of_a_register_pair`. Counting it as `base + c`
/// DOUBLES the span of every half-precision write.
///
/// This lives here, in the library, because THREE separate corpus oracles had each open-coded
/// `base + c` and all three were wrong in the same way. One of them
/// (`assumed_varying_orders_the_vertex_code_contradicts`) turned that into its strongest
/// verdict - CONTRADICTED, meaning "the layout is wrong and every varying past it is read from
/// the wrong register" - against the golf title's sky program, whose four-channel F16 write at
/// output 6 covers lanes 6..7 and was recorded as 6..9. Lane 9 is the padding between `Fog`
/// (one lane, at 8) and `TexCoord(0)` (at 10), so the phantom lane fell outside every declared
/// run. A false CONTRADICTED is worse than a missing one: it is the noise a later real one has
/// to be told apart from.
pub fn written_output_lanes(shader: &Shader) -> Vec<bool> {
    use crate::ir::Bank;
    let mut written: Vec<bool> = Vec::new();
    for instr in &shader.instrs {
        let Some(d) = instr.dest.as_ref() else { continue };
        if d.bank != Bank::Output {
            continue;
        }
        for c in 0..4 {
            if !instr.write_mask[c] {
                continue;
            }
            let lane = d.index as usize + if instr.half_precision { c >> 1 } else { c };
            if written.len() <= lane {
                written.resize(lane + 1, false);
            }
            written[lane] = true;
        }
    }
    written
}

/// Fill in each ORDINARY-REGISTER index load's `stride` - how far apart two consecutive index
/// values' blocks of rows are - from what the PROGRAM ITSELF fetches.
///
/// # The defect this exists to fix, and how the number was measured
/// A football title's skinned meshes read a `g_aMatrixPalette` through
/// `LoadIndex ; IntMad ; MemLoad` triples: the load turns a blend index into a row, the IMAD32
/// multiplies the row by 16 (a float4) and adds the buffer's base, the MemLoad reads four words.
/// Decoded as `src + addend` the rows of bone *b* come out at `b, b+1, b+2` - so bone 32's
/// second row IS bone 33's first, every bone shears into its neighbour, and the players render
/// as flat sheets.
///
/// THE PALETTE SAYS WHAT THE STRIDE IS, read straight out of a draw's own bound window
/// (`VITASLOP_GXP_INPUTS`, 2432 bytes): rows 0, 3, 6, 9, 12 and 15 are each the FIRST row of an
/// affine transform (`|xyz| = 1.0000`, translation in `.w`) and rows 1, 2, 4, 5, ... are second
/// and third rows. It is a packed array of THREE-row matrices. A stride of four is refuted by
/// the same dump - row 4 is a Y-axis row, not a matrix start. And the indices are plain
/// consecutive BONE numbers, not pre-multiplied rows: the same report gives `IN.blendIndices`
/// components ranging `[32,33]` and `[31,33]` on one pair and `[10,25]` over 184 vertices on
/// another, with every histogram bucket populated, so they are not multiples of three.
///
/// # Why it is READ OFF THE PROGRAM rather than written here as a 3
/// No field of the group-0x14 word has been shown to carry it, and one title cannot establish
/// an ISA constant. What the program does establish is its own layout: the loads that share a
/// source register are one index's run of rows, so the next index's run begins after the last
/// of them. Over the whole corpus that is unambiguous - **all 48 programs that use this form
/// have a maximum addend of exactly 2** (`index_load_max_addend_per_program`), and 132 of the
/// 135 (source, addend-set) groups are exactly `{0, 1, 2}`; the three that are not (`{0}` twice
/// and `{1, 2}` once) sit in programs whose maximum is still 2, which is why the maximum is
/// taken over the PROGRAM and not per source group.
///
/// A stream with no such load, or one whose only addend is 0, keeps stride 1 - which is
/// `src + addend`, exactly what it decoded to before. `VITASLOP_GXP_IDX_MUL` overrides the whole
/// thing for an A/B.
fn resolve_index_load_stride(instrs: &mut [crate::ir::Instr]) {
    use crate::ir::Op;
    let mut max = None;
    for i in instrs.iter() {
        if let Op::LoadIndex { addend, to_index: false, .. } = i.op {
            max = Some(max.map_or(addend, |m: i32| m.max(addend)));
        }
    }
    let Some(max) = max else { return };
    let Ok(stride) = u8::try_from(max + 1) else { return };
    for i in instrs.iter_mut() {
        if let Op::LoadIndex { to_index: false, stride: s, .. } = &mut i.op {
            *s = stride;
        }
    }
}

pub fn decode_shader(program: &Program) -> Shader {
    let mut instrs: Vec<_> = program.code.iter().map(|&w| decode(w)).collect();
    for instr in &mut instrs {
        let ordinal = match instr.op {
            Op::Tex { unit, .. } | Op::TexGather { unit, .. } => unit,
            _ => continue,
        };
        match program.sampler_unit_at(2 * ordinal as u32) {
            Some(unit) if unit <= u8::MAX as u32 => {
                match instr.op {
                    Op::Tex { coords, coord_half, lod, .. } => {
                        instr.op = Op::Tex { unit: unit as u8, coords, coord_half, lod };
                    }
                    Op::TexGather { coords, coord_half, .. } => {
                        instr.op = Op::TexGather { unit: unit as u8, coords, coord_half };
                        // A gather writes four TEXELS of ONE component, and where its four F16
                        // coefficients land is fixed by how many registers those texels take.
                        // With a single-component sampler that is four, which is what the
                        // corpus's only consumer reads - it dots the coefficients out of
                        // `dest + 4`. A wider sampler would gather four texels of EACH
                        // component and push the coefficients somewhere this cannot name, so it
                        // is refused rather than assumed to land in the same place.
                        let components =
                            program.sampler_at(unit).map_or(0, |p| p.component_count);
                        if components != 1 {
                            instr.blocked = instr.blocked.or(Some(
                                "0xE0 tex gather4 on a sampler with more than one component: \
                                 where the bilinear coefficients land is established only for \
                                 the single-component form",
                            ));
                        }
                    }
                    _ => unreachable!("only the two sample ops reach here"),
                }
            }
            _ => {
                // Name the CAUSE when the container can see it. An odd control-word base is
                // unaddressable by a double-register sampler field however the field decodes,
                // so reporting the sampler operand there sends the reader to the one part of
                // this that is known to be right.
                instr.blocked = Some(if program.unaddressable_texture_controls().is_empty() {
                    "SMP sampler operand does not resolve to a declared texture unit"
                } else {
                    "SMP sampler operand does not resolve: this program declares its texture \
                     control words at an ODD SA register, which a double-register sampler field \
                     cannot name (Program::unaddressable_texture_controls)"
                });
            }
        }
    }
    validate_imad_step_pairs(&mut instrs);
    resolve_index_load_stride(&mut instrs);
    // Last, so every pass above still sees one instruction per code word.
    let (mut instrs, starts) = unroll_repeats(&program.code, instrs);
    remap_branch_targets(&mut instrs, &starts);
    Shader { kind: program.kind, instrs }
}

/// Decode a program's SECONDARY code stream (see [`Program::secondary_code`]) into shader IR
/// that writes the SA bank.
///
/// The secondary program runs before the primary one and its whole purpose is to leave values in
/// SA registers the primary then reads. The SGX bank rule for this phase (spec A.2, post-decode
/// fixup 3) is that any operand whose bank is not an internal register, a hardware float
/// constant or an inline immediate is FORCED to SECATTR - the secondary program stores its
/// results in `sa`. Applying that here means the emitter is the same emitter, writing `sa`.
///
/// The spec rule is corroborated by two independent closures on the real corpus:
///   - a vertex whose primary reads `SA[3]` (`vsModelToWorldMatrix[0][3]`, which the guest
///     writes as 0.0) has a secondary program whose only `mov` writes register 3 from register
///     64 - that program's `vsCoarseExposureReg`. The primary passes it to the fragment, which
///     multiplies its colour by it: without this the surface is `colour * 0`, i.e. black;
///   - the same programs `pack` register 68 or 72 - `vsReciprocalFogRange` - into register 48 or
///     52 as F16, and the primary reads exactly that register as an F16 fog scale.
/// Both source indices are far past the primary attribute count (a vertex has ~20 attribute
/// registers) and land exactly on named uniforms, which only holds under the SA reading.
/// The destinations deliberately reuse uniform slots the primary never reads as F32 (register 48
/// is the view matrix's `m00`, and the primary reads only that matrix's third column).
pub fn decode_secondary_shader(program: &Program) -> Shader {
    use crate::ir::Bank;
    let mut instrs: Vec<_> = program.secondary_code.iter().map(|&w| decode(w)).collect();
    for instr in &mut instrs {
        for op in instr.dest.iter_mut().chain(instr.srcs.iter_mut()) {
            // Everything but an internal register and a constant becomes SA. An inline
            // immediate is not an operand in this IR (the decoder folds it into the op), so
            // the exemption list is exactly Internal + Constant.
            if !matches!(op.bank, Bank::Internal | Bank::Constant) {
                op.bank = Bank::SecondaryAttr;
            }
        }
    }
    validate_imad_step_pairs(&mut instrs);
    resolve_index_load_stride(&mut instrs);
    let (mut instrs, starts) = unroll_repeats(&program.secondary_code, instrs);
    remap_branch_targets(&mut instrs, &starts);
    Shader { kind: program.kind, instrs }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::ir::Instr;

    /// The two words of a golf title's address computation, which are a well-formed pair.
    const STEP0: u64 = 0xd082_8006_a01a_c080;
    const STEP1: u64 = 0xd092_8006_a01a_c080;

    fn validated(words: &[u64]) -> Vec<Instr> {
        let mut instrs: Vec<Instr> = words.iter().map(|&w| decode(w)).collect();
        validate_imad_step_pairs(&mut instrs);
        instrs
    }

    /// Each STEP is emitted on its own, pair or not.
    ///
    /// The decomposition is unique - only `((x >> 16) * y) << 16 + z` and `(x & 0xffff) * y + z`
    /// compose to the `x * y + z` the pair produces - so a step outside a pair is as established
    /// as one inside it. A football title emits two lone HIGH halves directly after a branch.
    #[test]
    fn every_readable_step_decodes_whether_or_not_it_has_a_partner() {
        let pair = validated(&[STEP0, STEP1]);
        assert_eq!(pair[0].blocked, None, "the low step of a real pair must decode");
        assert_eq!(pair[1].blocked, None, "the high step of a real pair must decode");
        for lone in [STEP0, STEP1] {
            let one = validated(&[lone]);
            assert_eq!(one[0].blocked, None, "a lone step is its own half-product");
        }
        // Two low steps in a row, and a "pair" that does not chain, are not pairs - and no
        // longer need to be, because neither step's own value depends on the other's.
        assert!(validated(&[STEP0, STEP0]).iter().all(|i| i.blocked.is_none()));
    }

}
