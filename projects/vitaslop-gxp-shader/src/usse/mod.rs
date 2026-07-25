//! USSE (SGX543 Unified Scalable Shader Engine) decoding.

pub mod decode;

pub use decode::{decode, field, opcode1, GroupTable, GROUP_TABLES};

use crate::container::Program;
use crate::ir::{Op, Shader};

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
    Shader { kind: program.kind, instrs }
}
