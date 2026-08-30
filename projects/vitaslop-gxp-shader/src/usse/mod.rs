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

        // The other half of the `moe_expand` guard in [`decode::decode_grp_mem_load`]: that
        // decoder allows a single-element memory access with bit 53 set because expansion
        // cannot step anything on a domain of one iteration - but that argument also needs the
        // MOE state to be its DEFAULT, and the state is only walkable here. Every captured
        // instance is in a program with no SMLSI at all; one that ran under a programmed
        // stride would be outside the census and must not be decoded on its strength.
        if matches!(decode::opcode1(word), 0x1d | 0x1e)
            && (word >> 53) & 1 == 1
            && state != decode::DEFAULT_REPEAT_STATE
        {
            out.push(crate::ir::Instr {
                blocked: Some(
                    "0xE8 memory access with moe_expand under a PROGRAMMED MOE state (an SMLSI                      is in force) is outside the census the single-element case rests on",
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
            .map(|o| match (o.moe, state[o.slot]) {
                // An intrinsic advance - the DP's channel walk - is not the MOE's to program.
                (false, _) => Ok(o.stride as i32),
                (true, decode::SmlsiSlot::Increment(n)) => Ok(i32::from(n) * o.stride as i32),
                (true, decode::SmlsiSlot::Swizzle(_)) => {
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
    use crate::ir::{Instr, Op, Operand};

    let step = |i: &Instr| match i.op {
        Op::IntMadStep { high_half, .. } => Some(high_half),
        _ => None,
    };
    // Bank and number together: two operands naming different banks are different operands even
    // when their numbers agree, and an inline literal is carried as an index in the IMMEDIATE
    // bank so this compares literals by value too.
    let same = |a: &Operand, b: &Operand| a.bank == b.bank && a.index == b.index;

    let mut blocked_at: Vec<(usize, &'static str)> = Vec::new();
    for at in 0..instrs.len() {
        let Some(high) = step(&instrs[at]) else { continue };
        // Look at the partner this step's own half implies, and let the OTHER end of the pair
        // report its own failure - so a lone step is named once from each side rather than
        // silently half-decoded.
        let partner = if high { at.checked_sub(1) } else { at.checked_add(1) };
        let ok = partner
            .and_then(|q| instrs.get(q).map(|other| (other, step(other))))
            .is_some_and(|(other, other_half)| {
                let (lo, hi) = if high { (other, &instrs[at]) } else { (&instrs[at], other) };
                other_half == Some(!high)
                    && lo.pred == hi.pred
                    && lo.blocked.is_none()
                    && hi.blocked.is_none()
                    && lo.srcs.len() == 3
                    && hi.srcs.len() == 3
                    && same(&lo.srcs[0], &hi.srcs[0])
                    && same(&lo.srcs[1], &hi.srcs[1])
                    && lo.dest.is_some_and(|d| same(&d, &hi.srcs[2]))
            });
        if !ok {
            blocked_at.push((
                at,
                "0x1a IMAD32-STEP: this step is not part of a well-formed multiply-add pair \
                 (an adjacent sn=0 / sn=1 with the same src0 and src1, the second's src2 being \
                 the first's destination). Only the pair's net result is established, so a step \
                 outside one is not emitted",
            ));
        }
    }
    for (at, why) in blocked_at {
        instrs[at].blocked = instrs[at].blocked.or(Some(why));
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
    let (mut instrs, starts) = unroll_repeats(&program.secondary_code, instrs);
    remap_branch_targets(&mut instrs, &starts);
    Shader { kind: program.kind, instrs }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::ir::{Bank, Instr, Operand};

    /// The two words of a golf title's address computation, which are a well-formed pair.
    const STEP0: u64 = 0xd082_8006_a01a_c080;
    const STEP1: u64 = 0xd092_8006_a01a_c080;

    fn validated(words: &[u64]) -> Vec<Instr> {
        let mut instrs: Vec<Instr> = words.iter().map(|&w| decode(w)).collect();
        validate_imad_step_pairs(&mut instrs);
        instrs
    }

    /// A well-formed pair decodes; each half on its own does not. What a single step leaves in
    /// its destination is the one thing the corpus does not pin down (see
    /// `decode_grp_imad32_step`), so a step outside the pair whose NET result is
    /// reading-independent must not be emitted.
    #[test]
    fn a_well_formed_step_pair_survives_and_a_lone_step_does_not() {
        let pair = validated(&[STEP0, STEP1]);
        assert_eq!(pair[0].blocked, None, "the low step of a real pair must decode");
        assert_eq!(pair[1].blocked, None, "the high step of a real pair must decode");

        for lone in [STEP0, STEP1] {
            let one = validated(&[lone]);
            assert!(
                one[0].blocked.is_some_and(|w| w.contains("well-formed multiply-add pair")),
                "a lone step must block: {:?}",
                one[0].blocked
            );
        }
    }

    /// A pair whose second step does not CHAIN through the first's destination is not the idiom
    /// - its net result is not `src0 * src1 + src2` under every reading - so it blocks too.
    #[test]
    fn a_step_pair_that_does_not_chain_blocks() {
        let mut instrs: Vec<Instr> = [STEP0, STEP1].iter().map(|&w| decode(w)).collect();
        // Point the high step's src2 somewhere the low step did not write.
        instrs[1].srcs[2] = Operand::plain(Bank::PrimaryAttr, 9, 2);
        validate_imad_step_pairs(&mut instrs);
        assert!(instrs[0].blocked.is_some(), "the low step of a broken pair must block");
        assert!(instrs[1].blocked.is_some(), "the high step of a broken pair must block");
    }

    /// Two low steps in a row are not a pair either, whichever way they are read.
    #[test]
    fn two_low_steps_are_not_a_pair() {
        let instrs = validated(&[STEP0, STEP0]);
        assert!(instrs.iter().all(|i| i.blocked.is_some()), "neither step may decode");
    }
}
