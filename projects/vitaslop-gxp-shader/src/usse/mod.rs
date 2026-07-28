//! USSE (SGX543 Unified Scalable Shader Engine) decoding.

pub mod decode;

pub use decode::{decode, field, opcode1, repeat_extra_iterations, GroupTable, GROUP_TABLES};

use crate::container::Program;
use crate::ir::{Op, Shader};

/// Turn SMLSI into a no-op in a program where every SMLSI provably sets the DEFAULT stepping on
/// every operand slot a repeat can still consult.
///
/// SMLSI carries no data effect of its own: it sets the per-operand increment/swizzle state
/// that a repeated instruction advances its registers by (spec F8.8, "metadata for repeated
/// instructions; emits nothing directly"). Its danger - and the reason it is blocked by
/// default - is that ignoring it would silently mis-address the operands of any instruction
/// that DOES repeat. [`unroll_repeats`] advances every operand by one of its own widths per
/// iteration, so an SMLSI that asks for exactly that is describing what the unroller already
/// does and is genuinely inert. Two ways that happens, and both are proofs rather than
/// relaxations:
///
///  * nothing in the program can repeat, so no instruction ever reads the state; or
///  * every slot that a repeat CAN read (`slots_repeats_consult`) is set to increment 1, the
///    default width step ([`decode::decode_smlsi`] documents how that unit was measured).
///
/// Anything else - an increment this recompiler has no evidence for, a swizzle-mode operand, an
/// instruction whose repeat encoding or operand grammar is not established - leaves every SMLSI
/// in the program blocked.
///
/// This is deliberately a whole-program test rather than a per-SMLSI scope analysis: SMLSI
/// state persists until the next SMLSI, and the ordering rules around branches are not
/// established, so "every SMLSI here is the default" is the statement the evidence supports.
fn retire_inert_repeat_state(code: &[u64], instrs: &mut [crate::ir::Instr]) {
    let needed = decode::slots_repeats_consult(code);
    let every_smlsi_is_the_default = code.iter().filter(|&&w| decode::is_smlsi(w)).all(|&w| {
        let state = decode::decode_smlsi(w);
        needed
            .iter()
            .zip(state)
            .all(|(&read, slot)| !read || slot == decode::SmlsiSlot::Increment(1))
    });
    if !every_smlsi_is_the_default {
        return;
    }
    for instr in instrs.iter_mut() {
        if matches!(instr.op, Op::Todo("flow smlsi (repeat-state) not modeled")) {
            instr.op = Op::Nop;
            instr.blocked = None;
        }
    }
}

/// Words a single operand of the given precision occupies per execution, which is also the
/// stride a REPEATED instruction advances that operand by between iterations.
///
/// The register banks are addressed in 32-bit words. An F16 operand packs its channels two per
/// word and so occupies ONE word; an F32 operand can only address the low two channels of a
/// packed register (spec A.6 - "F32 into these banks can only address the low two channels"),
/// so it occupies TWO. The width is a property of the operand's precision, not of how many
/// channels the write mask happens to enable.
fn operand_words(half_precision: bool) -> u32 {
    if half_precision {
        1
    } else {
        2
    }
}

/// Expand every repeating instruction into the sequence of single executions it stands for.
///
/// A USSE instruction carries a `repeat_count`: it re-executes that many extra times, and
/// between iterations each operand's register advances by that operand's own width
/// ([`operand_words`]). Nothing downstream of the IR models repetition, so unrolling here is
/// what makes the rest of the recompiler - the emitter, the written-output-lane check, the PA
/// read/write maps that decide the varying interface - see the instruction stream the hardware
/// actually executes.
///
/// MEASURED, on the two vertex programs that draw a retail title's entire front-end. Each
/// writes its colour varying with ONE `mov` to `Output[4]` under a two-channel mask, yet its
/// container declares 8 and 10 total output lanes respectively. The repeat counts are 1 and 2,
/// and unrolling at a stride of two words closes both statements exactly:
///
/// * 8 lanes: `Output[4] <- SA[0]`, `Output[6] <- SA[2]` - the 4-component `color` uniform
///   filling COLOR0's lanes 4..7 after clip position's 0..3;
/// * 10 lanes: `Output[4] <- PA[4]`, `Output[6] <- PA[6]`, `Output[8] <- PA[8]` - where the
///   parameter table places `In.Color` at PA[4] (4 components) and `In.TexCoord` at PA[8]. The
///   third iteration is the ONLY write of the texture coordinate anywhere in that program.
///
/// That last point is why this is not cosmetic: without unrolling, a textured program's UV
/// varying is never written at all, so every sample lands at (0,0). The stride also reproduces
/// the spec's own per-group repeat multipliers for group 0x40, which are stated as
/// `(dest,src1,src2) = (1,2,2)` for a float source - an F16 destination (one word) fed by F32
/// sources (two words each) - and `all 1` otherwise, where every operand is a single word.
///
/// An instruction whose group's repeat encoding is NOT established is BLOCKED rather than
/// emitted once: emitting once is a silent guess that it does not repeat, and a dropped
/// iteration is exactly the invisible failure this recompiler refuses to make.
/// Returns the unrolled stream and, alongside it, where each ORIGINAL code word landed in it:
/// `starts[i]` is the index of code word `i`'s first copy, and `starts[code.len()]` is the
/// stream length, so a branch target of "one past the end" maps too. Unrolling renumbers the
/// stream, and a branch offset is a count of code WORDS, so every branch has to be rewritten
/// through this map or it would silently point at the wrong instruction.
fn unroll_repeats(code: &[u64], instrs: Vec<crate::ir::Instr>) -> (Vec<crate::ir::Instr>, Vec<usize>) {
    let mut out = Vec::with_capacity(instrs.len());
    let mut starts = Vec::with_capacity(code.len() + 1);
    for (instr, &word) in instrs.into_iter().zip(code) {
        starts.push(out.len());
        let Some(extra) = decode::repeat_extra_iterations(word) else {
            out.push(crate::ir::Instr {
                blocked: Some("repeat_count encoding not established for this opcode group"),
                ..instr
            });
            continue;
        };
        if extra == 0 {
            out.push(instr);
            continue;
        }
        let dest_step = operand_words(instr.half_precision);
        let src_step = operand_words(instr.source_half_precision());
        for i in 0..=extra {
            let mut it = instr.clone();
            if let Some(d) = it.dest.as_mut() {
                d.index = d.index.saturating_add((i * dest_step).min(u8::MAX as u32) as u8);
            }
            for s in it.srcs.iter_mut() {
                s.index = s.index.saturating_add((i * src_step).min(u8::MAX as u32) as u8);
            }
            out.push(it);
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
    retire_inert_repeat_state(&program.code, &mut instrs);
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
    retire_inert_repeat_state(&program.secondary_code, &mut instrs);
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
