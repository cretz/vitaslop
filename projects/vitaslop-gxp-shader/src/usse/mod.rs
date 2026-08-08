//! USSE (SGX543 Unified Scalable Shader Engine) decoding.

pub mod decode;

pub use decode::{
    decode, decode_smlsi, field, is_smlsi, opcode1, repeat_extra_iterations, GroupTable, SmlsiSlot,
    GROUP_TABLES,
};

use crate::container::Program;
use crate::ir::{Op, Shader};

/// Whether the per-instruction SMLSI state can be read off the code stream LINEARLY, i.e. no
/// branch can carry control across an SMLSI.
///
/// SMLSI sets state that persists until the next SMLSI, so what a repeating instruction consults
/// is simply the last SMLSI before it - provided control actually reached it that way. A branch
/// that jumps over an SMLSI (or into the middle of its scope) makes the state at a later
/// instruction path-dependent, and this decoder has no dataflow to resolve that. In that case
/// every SMLSI stays BLOCKED, which is where the model was for every program before this.
///
/// The span is taken as the whole open interval between the branch and its target, in both
/// directions. A backward branch that re-executes its own SMLSI would in fact be safe, but the
/// corpus contains no such program, and a rule that has to reason about re-execution order is
/// not one worth having on no evidence.
fn smlsi_state_is_linear(code: &[u64], instrs: &[crate::ir::Instr]) -> bool {
    let smlsi_at = |lo: i64, hi: i64| {
        code.iter().enumerate().any(|(i, &w)| (i as i64) > lo && (i as i64) < hi && decode::is_smlsi(w))
    };
    !instrs.iter().enumerate().any(|(at, instr)| {
        let Op::Branch { rel } = instr.op else { return false };
        let (at, target) = (at as i64, at as i64 + i64::from(rel));
        smlsi_at(at.min(target), at.max(target))
    })
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
fn unroll_repeats(code: &[u64], instrs: Vec<crate::ir::Instr>) -> (Vec<crate::ir::Instr>, Vec<usize>) {
    let linear = smlsi_state_is_linear(code, &instrs);
    let mut state = decode::DEFAULT_REPEAT_STATE;
    let mut out = Vec::with_capacity(instrs.len());
    let mut starts = Vec::with_capacity(code.len() + 1);
    for (instr, &word) in instrs.into_iter().zip(code) {
        starts.push(out.len());

        // SMLSI itself emits nothing - its entire effect is the state the repeats below read.
        if decode::is_smlsi(word) {
            state = decode::decode_smlsi(word);
            out.push(crate::ir::Instr {
                op: Op::Nop,
                blocked: (!linear).then_some(
                    "0xF8 SMLSI state is not linearly readable - a branch crosses its scope",
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
            out.push(instr);
            continue;
        }
        // From here the instruction really repeats, so the operand grammar has to be known
        // exactly: which SMLSI byte governs each operand, and what one unit of it moves.
        let Some(operands) = decode::repeat_operands(word) else {
            out.push(crate::ir::Instr {
                blocked: Some("repeat operand slots not established for this opcode group"),
                ..instr
            });
            continue;
        };
        let steps: Result<Vec<i32>, &'static str> = operands
            .iter()
            .map(|o| match state[o.slot] {
                decode::SmlsiSlot::Increment(n) => Ok(i32::from(n) * o.stride as i32),
                decode::SmlsiSlot::Swizzle(_) => {
                    Err("0xF8 SMLSI per-iteration SWIZZLE stepping not modeled")
                }
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
        let advance = |index: u8, step: i32, i: u32| -> Option<u8> {
            u8::try_from(i32::from(index) + step * i as i32).ok()
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
            out.push(it);
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

/// Decode a parsed program's USSE code stream into the shader IR.
///
/// A `SMP` instruction addresses its sampler by a REGISTER field, not by texture unit: the
/// texture's control words live at SA register `2 * field`, and only the container's
/// texture-control table says which GXM unit those words describe. That resolution happens here,
/// so `Op::Tex::unit` is a real texture unit everywhere downstream - the same namespace a
/// PDS-prefetched sample names directly, and the one the renderer binds by. A field the table
/// does not describe blocks the instruction rather than naming an arbitrary unit.
pub fn decode_shader(program: &Program) -> Shader {
    let mut instrs: Vec<_> = program.code.iter().map(|&w| decode(w)).collect();
    for instr in &mut instrs {
        let Op::Tex { unit: ordinal, coords, coord_half, lod } = instr.op else { continue };
        match program.sampler_unit_at(2 * ordinal as u32) {
            Some(unit) if unit <= u8::MAX as u32 => {
                instr.op = Op::Tex { unit: unit as u8, coords, coord_half, lod };
            }
            _ => {
                instr.blocked =
                    Some("SMP sampler operand does not resolve to a declared texture unit");
            }
        }
    }
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
    let (mut instrs, starts) = unroll_repeats(&program.secondary_code, instrs);
    remap_branch_targets(&mut instrs, &starts);
    Shader { kind: program.kind, instrs }
}
