//! Case-file plumbing shared by the two producers of GPU execution cases:
//! [`super`]`::execcases` (one case per CAPTURED blob) and `conformance` (one case per AUTHORED
//! program, carrying a stated intent).
//!
//! # Why shared rather than copied
//!
//! A case file carries a SEED, not its thousand input numbers, and the runner
//! (`vitaslop-web/e2e/gxpexec.mjs`) regenerates the inputs from that seed with a JS twin of
//! [`lane_value`]. Three implementations of one generator - Rust here, Rust there, JS in the
//! runner - is two chances to drift, and a drift would feed the GPU different inputs from the
//! reference and report the difference as a translation defect. There is one Rust copy, and the
//! JS twin's own comment pins it.

// Each test binary that includes this module uses a different part of it - the corpus harness
// needs the guest-memory helpers, the conformance suite does not - and an unused item in a
// shared test module is not dead code, it is code the OTHER binary uses.
#![allow(dead_code)]

use std::collections::{BTreeMap, BTreeSet};
use std::path::Path;

use vitaslop_gxp_shader::interp::RegFile;
use vitaslop_gxp_shader::ir::{
    Bank, BitwiseKind, Instr, Op, Operand, Predicate, Shader, TestAlu,
};
use vitaslop_gxp_shader::module::{mem_window_placements, mem_window_vec4_count, MemWindow};
use vitaslop_gxp_shader::wgsl::{bank_prec, Prec, CASE_BANK_LANES};
use vitaslop_gxp_shader::ProgramKind;

/// The input register file is a pure FUNCTION of `(seed, lane)`, computed identically on both
/// sides, so a case file carries a seed rather than a thousand numbers - and the runner can
/// regenerate the exact inputs without trusting a transcription of them.
///
/// The mix is an ordinary integer avalanche; what matters is only that the JS half reproduces
/// it bit for bit (`Math.imul` + `>>> 0`), which its own comment pins.
pub fn lane_bits(seed: u32, lane: u32) -> u32 {
    let mut h = seed ^ lane.wrapping_mul(0x9E37_79B9);
    h ^= h >> 16;
    h = h.wrapping_mul(0x7FEB_352D);
    h ^= h >> 15;
    h = h.wrapping_mul(0x846C_A68B);
    h ^= h >> 16;
    h
}

/// The float a lane holds: the avalanche's top 24 bits mapped onto `[-4, 4)`.
///
/// DELIBERATELY TAME. Seeding the register file with arbitrary bit patterns would feed
/// denormals, infinities and NaNs into every program, and the resulting disagreements would be
/// about IEEE corner cases on two different float units rather than about our translation. The
/// range is wide enough to exercise sign, magnitude and fractional behaviour and narrow enough
/// that a divergence means something.
pub fn lane_value(seed: u32, lane: u32) -> f32 {
    let h = lane_bits(seed, lane);
    (h >> 8) as f32 / 16_777_216.0 * 8.0 - 4.0
}

/// The guest base address handed to window `i`. Chosen by the harness (the real one comes from
/// whatever the guest bound at draw time), well separated so no two windows overlap and far from
/// zero so an address that fell out of a miscomputation does not land inside one by luck.
pub fn window_base(i: usize) -> u32 {
    0x1000_0000 + (i as u32) * 0x0010_0000
}

/// The whole `gxp_mem` binding for `windows`: one header `vec4` per window (lane x = its guest
/// base) then every window's bytes in order - the layout [`mem_window_placements`] describes and
/// the emitted helper reads.
///
/// The data words come from the same avalanche the register file uses, in a separate domain so a
/// window's contents never coincide with a register's value; the module builder bakes THIS SLICE
/// into the module and [`mem_fetch`] reads THIS SLICE for the reference, so there is one copy of
/// the bytes and no second generator to drift.
pub fn mem_words_for(seed: u32, windows: &[MemWindow]) -> Vec<u32> {
    let placements = mem_window_placements(windows);
    let total = mem_window_vec4_count(windows) as usize * 4;
    let mut words = vec![0u32; total];
    for (i, _) in windows.iter().enumerate() {
        words[i * 4] = window_base(i);
    }
    for at in &placements {
        for w in 0..at.words {
            let g = (at.first_word + w) as usize;
            if g < words.len() {
                // TAME FLOATS, for the same reason the register file is seeded with them rather
                // than with arbitrary bit patterns: a window is read as a matrix or a uniform
                // array, and random 32-bit patterns decode to values spanning 1e-38 to 1e36, so
                // any difference between the two sides comes out as a 1e32 number whose size
                // says nothing about its cause. The magnitudes here keep a divergence legible.
                words[g] = lane_value(seed, 0x0002_0000 + g as u32).to_bits();
            }
        }
    }
    words
}

/// Does `addr` land inside any bound window - i.e. will the load read a real word rather than
/// the zero an address outside every window reads?
pub fn placements_hit(addr: u32, windows: &[MemWindow]) -> bool {
    mem_window_placements(windows)
        .iter()
        .enumerate()
        .any(|(i, at)| addr >= window_base(i) && ((addr - window_base(i)) >> 2) < at.words)
}

/// Resolve a guest address against the bound windows exactly as `mem_window_helper` does: the
/// FIRST window containing the address answers, and an address inside none reads zero.
///
/// It must be the same rule, not merely a plausible one - a reference that resolved differently
/// would report a divergence for every load in the program and name the wrong cause.
pub fn mem_fetch(addr: u32, windows: &[MemWindow], words: &[u32]) -> u32 {
    for (i, at) in mem_window_placements(windows).iter().enumerate() {
        let base = window_base(i);
        if addr >= base {
            let w = (addr - base) >> 2;
            if w < at.words {
                return words.get((at.first_word + w) as usize).copied().unwrap_or(0);
            }
        }
    }
    0
}

/// The SOURCE of an instruction that copies whole 32-bit words, or `None` when it computes.
///
/// Two shapes, and the second is not an optimisation nobody writes - it is how this ISA spells
/// an integer move:
///
/// * a MOVE at full precision. A half-precision move packs two channels into one register and
///   is not a word copy at all, which is why the precision is part of the test.
/// * `OR` with the immediate ZERO at 32 lane bits. There is no move opcode in the integer
///   group - the assembler learned the same fact about the float groups, where a move is an ADD
///   OF THE INLINE ZERO - so a compiler emitting "copy this word" emits `x | 0`.
///
/// MEASURED: the second shape is what `cw-rr-corpus__frag_86687980` ends `r[0]` with, and while
/// only the first was recognised that lane was compared as an f32 while holding two halves. The
/// two sides read `0x43d2fe00` against `0x43d27e00` - **one bit apart, the SIGN of a NaN in the
/// low half** - and it was reported as 32,768 ULP.
fn whole_word_copy_source(instr: &Instr, prec: Prec) -> Option<&Operand> {
    let copies = match instr.op {
        Op::Mov => matches!(prec, Prec::F32),
        Op::Bitwise { kind: BitwiseKind::Or, imm: Some(0), lane_bits: 32 } => true,
        _ => false,
    };
    copies.then(|| instr.srcs.first()).flatten()
}

/// The precision each destination lane HOLDS, as a `(bank, lane) -> code` map with
/// 0 = f32, 1 = two packed f16s, 2 = four unorm bytes.
///
/// >>> A PACKED WORD COMPARED AS ONE f32 MISREPORTS ITS OWN ERROR. A register holding two F16s
/// is one 32-bit word, and reading that word as a single float makes a one-ULP difference in
/// the HIGH half read as a difference of thousands in the low mantissa bits of a nonsense
/// number - or hides a real difference in the low half entirely. The runner has to know which
/// view a lane holds before it can say whether the two sides agree, so the case carries it.
///
/// # >>> WHAT A LANE HOLDS IS NOT WHAT ITS LAST WRITER'S PRECISION FIELD SAYS
///
/// This asked each instruction what precision IT operates at and believed the answer for the
/// lane. A full-precision MOVE breaks that: it copies a whole 32-bit word, and if the word held
/// two halves it still holds two halves afterwards. The instruction is F32; the CONTENT is not.
///
/// MEASURED, and it was the two loudest rows in the corpus remainder. `mk-corpus-roof__frag_86c41b20`
/// and `mk-corpus4__frag_9125fc80` both ended with `Mov o[0].xy <- pa[0].xyxy` at full
/// precision, over a `pa[0]` that eleven half-precision instructions had filled. Read as an f32
/// the two sides were `-4.2865e37` against `+4.2699e37` - **4,227,955,712 ULP, the largest
/// numbers in the whole differential**. The raw words were `0xfe00fe00` and `0x7e007e00`: every
/// half a NaN on both sides, differing in the SIGN BIT alone, which IEEE does not specify for a
/// generated NaN and which is not this translation's business. Both programs read PERFECTLY
/// STABLE in the conditioning census, so nothing else could have explained them away.
///
/// So a MOVE propagates its source's view. It is a two-line dataflow rule rather than a
/// per-instruction question, and it needs the input banks tracked too - `pa` is where those
/// programs did their half-precision arithmetic before the move.
///
/// What this still cannot do is follow a value through arithmetic; it does not need to, because
/// an instruction that COMPUTES gives its result the precision it computed at. Only a copy can
/// carry a view its own opcode does not name.
/// The code for a lane whose view is NOT statically determined - see
/// [`written_lane_precision`]'s note on conditional writes.
pub const PREC_AMBIGUOUS: u8 = 4;

pub fn written_lane_precision(shader: &Shader) -> BTreeMap<(u8, usize), u8> {
    // Every writable bank, including the ones the runner does not compare: a move's SOURCE is
    // routinely a `pa` register the program used as scratch, and a rule that could not see
    // those writes would have nothing to propagate.
    let mut all: BTreeMap<(u8, usize), u8> = BTreeMap::new();
    // >>> A WRITE THAT MAY NOT HAVE HAPPENED MUST NOT CLAIM THE LANE'S VIEW.
    //
    // This walk is static and takes the LAST writer in program order. A predicated write, or one
    // inside a branch, may not have run at all - and if it disagrees with the standing view, the
    // lane holds one of two things and nothing here can say which.
    //
    // MEASURED: `gxp-all__hs-gxp__frag_81fa9a10` ends `r[10]` with `r[10] = (sa[34] | 0u)` -
    // this ISA's integer move - inside `if (p[0])`, which did not run. The lane held the F16
    // PAIR an unconditional write left there, and reading it as one f32 turned two NaN halves
    // that agree (`0xfe00fe00` against `0x7e007e00`, the sign bit alone, which IEEE does not
    // specify) into **4,227,955,712 ULP** - the loudest row in the whole differential, and not a
    // defect. Exactly the shape the whole-word-copy rule was written to remove, one level up.
    //
    // Such a lane is marked AMBIGUOUS and the runner grades it in BOTH views, agreeing if either
    // does. That is weaker, so it is COUNTED and reported rather than quietly applied.
    //
    // The conservatism is confined to where it is needed: a lane is only ambiguous if its
    // writers actually DISAGREE about the view, and only when one of them could have been
    // skipped. A straight-line program with no predicates keeps exact last-write-wins.
    let has_branch = shader.instrs.iter().any(|i| matches!(i.op, Op::Branch { .. }));
    let mut conditional: BTreeMap<(u8, usize), bool> = BTreeMap::new();
    let bank_code = |b: Bank| match b {
        Bank::Temp => Some(0u8),
        Bank::Output => Some(1),
        Bank::Internal => Some(2),
        Bank::PrimaryAttr => Some(3),
        Bank::SecondaryAttr => Some(4),
        _ => None,
    };
    // Record one lane's view, merging with whatever is already there: a write that could have
    // been skipped and disagrees leaves the lane ambiguous, and so does a determined write
    // landing on a lane an earlier skippable one already claimed.
    let record = |all: &mut BTreeMap<(u8, usize), u8>,
                      conditional: &mut BTreeMap<(u8, usize), bool>,
                      key: (u8, usize),
                      code: u8,
                      skippable: bool| {
        let merged = match all.get(&key).copied() {
            Some(PREC_AMBIGUOUS) => PREC_AMBIGUOUS,
            Some(prev) if prev != code && (skippable || conditional.get(&key).copied() == Some(true)) => {
                PREC_AMBIGUOUS
            }
            _ => code,
        };
        all.insert(key, merged);
        // A lane last written by a write that definitely happened is determined again, so a
        // later disagreement with IT is an ordinary overwrite rather than an ambiguity.
        conditional.insert(key, skippable);
    };

    for instr in &shader.instrs {
        let Some(dest) = instr.dest.as_ref() else { continue };
        let Some(bank) = bank_code(dest.bank) else { continue };
        let prec = bank_prec(dest.bank, Prec::of(instr));
        // A branch anywhere in the program puts every write in reach of one, and this walk does
        // not build a CFG. The cost of that conservatism is only paid where views DISAGREE.
        let skippable = has_branch || !matches!(instr.pred, Predicate::Always);

        // >>> A WHOLE-WORD COPY CARRIES THE VIEW OF WHAT IT COPIED. Channel `c` reads source
        // lane `index + swizzle[c]` and writes destination lane `index + c`, one whole 32-bit
        // word each, so the destination holds exactly what the source held.
        if let Some(src) = whole_word_copy_source(instr, prec)
            && let Some(sbank) = bank_code(src.bank)
        {
            {
                for c in 0..4 {
                    if !instr.write_mask[c] {
                        continue;
                    }
                    let from = src.index as usize + src.swizzle[c] as usize;
                    let to = dest.index as usize + c;
                    // An unknown source is an f32 lane by default, which is what this function
                    // answered for everything before the rule existed.
                    let code = all.get(&(sbank, from)).copied().unwrap_or(0);
                    record(&mut all, &mut conditional, (bank, to), code, skippable);
                }
                continue;
            }
        }
        let (code, lanes): (u8, Vec<usize>) = match prec {
            Prec::F32 => {
                (0, (0..4).filter(|&c| instr.write_mask[c]).map(|c| dest.index as usize + c).collect())
            }
            // Two channels share a word, so the lane a channel lands in is `index + (c >> 1)`.
            Prec::F16 => (
                1,
                (0..4).filter(|&c| instr.write_mask[c]).map(|c| dest.index as usize + (c >> 1)).collect(),
            ),
            // All four channels are bytes of ONE word.
            Prec::Fx8 => {
                (2, if instr.write_mask.iter().any(|&w| w) { vec![dest.index as usize] } else { vec![] })
            }
        };
        for lane in lanes {
            record(&mut all, &mut conditional, (bank, lane), code, skippable);
        }
    }
    // `sa` is an input the program does not write back and the runner does not read, so it
    // travelled only to make the propagation above possible. `pa` IS compared - see
    // `wrap_compute_module_for` - so its view travels with the case.
    all.into_iter().filter(|((b, _), _)| *b < 4).collect()
}

/// Serialise the non-zero lanes of a bank as a JSON array of `[lane, bits, precision]` triples.
///
/// Both sides start every written bank at zero (the interpreter zeroes its `RegFile`, WGSL
/// zero-initialises a `var<private>` array), so the zeros need no transcription and their
/// agreement is not in question. Only what the program WROTE travels - and the runner still
/// compares every lane, so a lane the GPU wrote and the expectation does not name is a
/// divergence rather than something nobody looked at.
/// [`nonzero_pairs`] for a bank whose baseline is the SEEDED input rather than zero.
///
/// `pa` is loaded with the case's seeded values before the body runs, so "nonzero" says nothing
/// about it - almost every lane is nonzero and almost none was touched. What the case has to
/// carry is what the program CHANGED, and the runner rebuilds the rest from the same seed.
pub fn changed_pairs(
    bank_code: u8,
    base: usize,
    lanes: &[f32],
    seeded: &[f32],
    prec: &BTreeMap<(u8, usize), u8>,
    out: &mut Vec<String>,
) {
    for (n, v) in lanes.iter().enumerate().take(CASE_BANK_LANES) {
        let bits = v.to_bits();
        if seeded.get(n).map(|s| s.to_bits()) != Some(bits) {
            let p = prec.get(&(bank_code, n)).copied().unwrap_or(0);
            out.push(format!("[{},{},{}]", base + n, bits, p));
        }
    }
}

pub fn nonzero_pairs(
    bank_code: u8,
    base: usize,
    lanes: &[f32],
    prec: &BTreeMap<(u8, usize), u8>,
    out: &mut Vec<String>,
) {
    for (n, v) in lanes.iter().enumerate().take(CASE_BANK_LANES) {
        let bits = v.to_bits();
        if bits != 0 {
            let p = prec.get(&(bank_code, n)).copied().unwrap_or(0);
            out.push(format!("[{},{},{}]", base + n, bits, p));
        }
    }
}

/// Write one case's `.wgsl` module and `.json` description into `out`.
///
/// `pairs` is the EXPECTATION the runner holds the GPU to. For a corpus case that is what the
/// reference interpreter computed; for an authored conformance case it is what the program was
/// WRITTEN to mean. The file format does not distinguish them, and it should not: the runner's
/// job is the same either way, and the difference in what a pass PROVES belongs in the test that
/// produced the case, not in the runner.
/// Which GPU rig a case runs in - which decides the pipeline the runner builds for it and the
/// shape of its output buffer.
///
/// It travels in the case file because the runner cannot infer it: both rigs' modules are
/// ordinary WGSL, and a render case fed to a compute pipeline fails with "no entry point
/// `cs_main`" rather than with anything that names the cause.
#[derive(Clone, Copy, PartialEq, Eq, Debug)]
pub enum Stage {
    /// A compute dispatch: four banks out, one entry point `cs_main`.
    Compute,
    /// A render pass over a 1x1 target: `vs_main` + `fs_main`, real texture bindings, and two
    /// extra output words (the kill flag and the fragment depth).
    Fragment,
}

impl Stage {
    fn name(self) -> &'static str {
        match self {
            Stage::Compute => "compute",
            Stage::Fragment => "fragment",
        }
    }
}

#[allow(clippy::too_many_arguments)]
pub fn write_case(
    out: &Path,
    name: &str,
    kind: ProgramKind,
    seed: u32,
    instrs: usize,
    half: usize,
    module: &str,
    mem_words: &[u32],
    pairs: &[String],
    // `inputs`: the lanes whose seeded float this program replaced with a plausible integer or
    // a small index, as `[lane, bits]` over the flat `pa` then `sa` numbering. SPARSE, and empty
    // for a program that needed none. The runner cannot recompute these - they come out of a
    // dataflow analysis over an IR it does not have - so they travel as data, for the same
    // reason the window bytes and the render rig's texels do.
    inputs: &[(usize, u32)],
    stage: Stage,
    // `units`: the sampler units the module declares, ASCENDING - which is the order the render
    // rig assigns its bindings in, so this list is what tells the runner which stand-in texture
    // goes at which binding. Empty for a compute case, whose stand-in is a function.
    units: &[u8],
    // `cubes`: 1 where that slot's binding is a CUBE and 0 where it is flat, one entry per unit.
    // Not recoverable from the unit number - the same unit is a flat texture in one program and
    // a cube in another - and binding the wrong one samples a different texture from the one the
    // module declared, which does not even type-check.
    cubes: &[u8],
) {
    std::fs::write(out.join(format!("{name}.wgsl")), module).expect("write the module");
    let mem_json = if mem_words.is_empty() {
        String::new()
    } else {
        format!(
            ",\"mem\":[{}]",
            mem_words.iter().map(|w| w.to_string()).collect::<Vec<_>>().join(",")
        )
    };
    let inputs_json = if inputs.is_empty() {
        String::new()
    } else {
        format!(
            ",\"inputs\":[{}]",
            inputs
                .iter()
                .map(|(lane, bits)| format!("[{lane},{bits}]"))
                .collect::<Vec<_>>()
                .join(",")
        )
    };
    let list = |v: &[u8]| v.iter().map(|u| u.to_string()).collect::<Vec<_>>().join(",");
    let units_json = if units.is_empty() {
        String::new()
    } else {
        format!(",\"units\":[{}],\"cubes\":[{}]", list(units), list(cubes))
    };
    let json = format!(
        "{{\"name\":\"{name}\",\"kind\":\"{}\",\"stage\":\"{}\",\"seed\":{seed},\"lanes\":{},\
         \"instrs\":{instrs},\"half\":{half}{mem_json}{inputs_json}{units_json},\"expect\":[{}]}}\n",
        match kind {
            ProgramKind::Vertex => "vertex",
            ProgramKind::Fragment => "fragment",
        },
        stage.name(),
        CASE_BANK_LANES,
        pairs.join(",")
    );
    std::fs::write(out.join(format!("{name}.json")), json).expect("write the case");
}

// ---------------------------------------------------------------------------------------------
// >>> WHICH SEEDED INPUT LANE DECIDES AN ADDRESS, A COUNT OR AN INDEX.
//
// The differential's coverage frontier moved off "which blobs are cases" - 1,139 of 1,151 are -
// and onto "are the inputs the cases run on worth anything". Three symptoms, ONE cause:
//
//   * programs whose every written lane is zero,
//   * programs whose every guest-memory load lands OUTSIDE every bound window (and so reads the
//     same zero on both sides, agreeing without checking anything),
//   * programs whose data-driven loop never terminates inside the interpreter's step budget.
//
// All three are the seed not knowing what a register MEANS. `lane_value` fills every lane with a
// float in `[-4, 4)`, and a lane read as an INTEGER - a byte pointer, an element index, a loop
// trip count - then holds `0xC0800000`, i.e. 3,229,614,080. An address computed from it misses
// every window; a loop bounded by it is asked to run three billion times.
//
// The fix is a per-program input, and the fix needs to know which lanes those are. That is a
// dataflow question, so it is ANSWERED by dataflow rather than by a list of register numbers:
// the lanes below are the ones that REACH an integer sink, walked forwards from the seed.
// ---------------------------------------------------------------------------------------------

/// The five banks a case tracks, as the small codes the case file and these analyses share.
///
/// `sa` (4) is included even though the runner never compares it: it is an INPUT bank, and a
/// walk that could not name an `sa` lane could not name where an address came from.
pub fn bank_code(b: Bank) -> Option<u8> {
    match b {
        Bank::Temp => Some(0),
        Bank::Output => Some(1),
        Bank::Internal => Some(2),
        Bank::PrimaryAttr => Some(3),
        Bank::SecondaryAttr => Some(4),
        _ => None,
    }
}

/// True when the code names one of the two banks a case SEEDS - `pa` and `sa`.
pub fn is_seeded_bank(code: u8) -> bool {
    code == 3 || code == 4
}

/// >>> AN ADDRESS OPERAND IS A RAW LANE, NOT A SWIZZLED SPAN, and the two instructions that read
/// >>> one are the two this analysis most needs to get right.
///
/// The interpreter's own note says it: [`Op::MemLoad`]'s pointer and [`Op::LoadIndex`]'s index
/// are read as `bank[index]` exactly - no swizzle selector, no abs/neg, no four-register span. A
/// walk that gave them the ordinary span would name three innocent lanes beside the real one and
/// seed all four as integers.
fn reads_a_raw_lane(op: Op, which: usize) -> bool {
    matches!(op, Op::MemLoad { .. } | Op::LoadIndex { .. }) && which == 0
}

/// The lanes source operand `which` reads, as `(bank code, lanes)`.
///
/// >>> THE SPAN IS THE INSTRUCTION'S OWN, NOT A FIXED FOUR REGISTERS, and the difference is the
/// >>> whole analysis.
///
/// The first version of this took `index..index+4` because a full-precision swizzle selects
/// among four consecutive registers. MEASURED, that is ruinous: a football title's skinning
/// vertex program reads `pa[4..=7]` as a float at one instruction where only `pa[4]` is a float
/// and `pa[7]` is a guest BYTE POINTER, and one such overlap marks every lane that reached the
/// pointer as "read as a float". Every one of the 88 all-miss programs came back with no
/// integer-only lane at all, and the lever this analysis exists to aim looked useless.
///
/// [`Instr::read_channels`] and [`Instr::source_register`] already state the exact rule - which
/// channels an instruction consults and which register each one addresses, F32, F16 and packed
/// byte alike, with a swizzle CONSTANT (selector >= 4) reading no register. They are the
/// emitter's and the linker's own answer, so asking them is one statement rather than a third.
///
/// The one thing they cannot say is the RAW-LANE exception: [`Op::MemLoad`]'s pointer and
/// [`Op::LoadIndex`]'s index are read as `bank[index]` exactly, with no swizzle selector at all
/// - the interpreter carries the same note, having once resolved a different guest address from
/// the same program by reading one through a channel.
pub fn src_lanes(instr: &Instr, which: usize) -> Option<(u8, Vec<usize>)> {
    let src = instr.srcs.get(which)?;
    let bank = bank_code(src.bank)?;
    if reads_a_raw_lane(instr.op, which) {
        return Some((bank, vec![src.index as usize]));
    }
    let channels = instr.read_channels();
    let mut lanes: Vec<usize> = (0..4)
        .filter(|&c| channels[c])
        .filter_map(|c| instr.source_register(src, c).map(|(reg, _)| reg as usize))
        .collect();
    lanes.sort_unstable();
    lanes.dedup();
    Some((bank, lanes))
}

/// The lanes an instruction WRITES, as `(bank code, lanes)`.
///
/// Three ops write a span the write mask cannot describe, and each says so in its own doc:
/// [`Op::MemLoad`] writes `elements` consecutive registers, [`Op::Tex`] writes four, and
/// [`Op::TexGather`] writes six (four texels and two coefficient registers). Everything else
/// goes through the same per-precision rule [`written_lane_precision`] uses.
pub fn dest_lanes(instr: &Instr) -> Option<(u8, Vec<usize>)> {
    let dest = instr.dest.as_ref()?;
    let bank = bank_code(dest.bank)?;
    let base = dest.index as usize;
    match instr.op {
        // The INDEX-register form writes the index file, not a bank, so it writes no lane here.
        Op::LoadIndex { to_index: true, .. } => None,
        Op::MemLoad { elements, .. } => Some((bank, (base..base + elements as usize).collect())),
        Op::Tex { .. } => Some((bank, (base..base + 4).collect())),
        Op::TexGather { .. } => Some((bank, (base..base + 6).collect())),
        _ => {
            let prec = bank_prec(dest.bank, Prec::of(instr));
            let mut lanes: Vec<usize> = match prec {
                Prec::F32 => (0..4).filter(|&c| instr.write_mask[c]).map(|c| base + c).collect(),
                Prec::F16 => {
                    (0..4).filter(|&c| instr.write_mask[c]).map(|c| base + (c >> 1)).collect()
                }
                Prec::Fx8 => {
                    if instr.write_mask.iter().any(|&w| w) { vec![base] } else { vec![] }
                }
            };
            lanes.dedup();
            Some((bank, lanes))
        }
    }
}

/// >>> A SINK THAT READS ITS OPERAND'S BIT PATTERN AS AN INTEGER, and which operand that is.
///
/// Every one of these is established by the op's own documentation, not inferred from a name:
///
/// * [`Op::MemLoad`] src0 IS a guest byte pointer.
/// * [`Op::LoadIndex`] src0 is an element index (the interpreter takes `raw & 0xffff` of it).
/// * [`Op::IntMad`] / [`Op::IntMadStep`] are the integer multiply-adds - every operand.
/// * [`Op::Bitwise`] is the integer bitwise/shift group, read through `bitcast<u32>`.
/// * [`Op::PackFromInt`] and [`Op::PackIntCopy`] read integer halves. [`Op::PackToInt`] reads a
///   FLOAT and is deliberately not here.
///
/// >>> A TEST IS AN INTEGER SINK EXACTLY WHEN ITS ALU FAMILY IS ONE, and [`TestAlu`]'s own docs
/// >>> say which those are: the family decides how the operands are READ, not just what is done
/// >>> to them.
///
/// It is the difference between reaching a loop's trip count and not reaching it. MEASURED on a
/// golf title's world vertex programs - the twelve that spin past the interpreter's step budget:
/// the light counter is `pa[17]`, filled by a `bitwise` from `sa[109]` and compared against
/// `sa[110]` by a `vtst`. With the test counted as a float read, `sa[109]` is "read both ways"
/// and stays a float bit pattern, and the loop is asked to run three billion times.
///
/// `Fx8Sub` is deliberately NOT here. It reads four 8-bit unsigned-normalised channels - not an
/// integer index, count or pointer - and seeding such a lane with `1..=15` would make its four
/// bytes three zeroes and a small one rather than anything a draw would produce.
fn test_alu_reads_integers(alu: TestAlu) -> bool {
    matches!(alu, TestAlu::BitAnd | TestAlu::BitShl | TestAlu::IntSub | TestAlu::IntSub16U)
}

pub fn integer_sink_operands(instr: &Instr) -> Vec<usize> {
    match instr.op {
        Op::MemLoad { .. } | Op::LoadIndex { .. } => vec![0],
        Op::IntMad { .. } | Op::IntMadStep { .. } | Op::Bitwise { .. } => {
            (0..instr.srcs.len()).collect()
        }
        Op::PackFromInt { .. } | Op::PackIntCopy { .. } => vec![0],
        Op::Test { alu, .. } | Op::TestMask { alu, .. } if test_alu_reads_integers(alu) => {
            (0..instr.srcs.len()).collect()
        }
        _ => vec![],
    }
}

/// >>> AN OPERAND THAT IS A FLOAT ON THE WAY TO BEING AN INTEGER.
///
/// [`Op::PackToInt`] is a truncating numeric cast - a float source, an integer destination - and
/// it is how this ISA gets a vertex ATTRIBUTE into an index. That makes its source neither of
/// the other two things: seeding it with an integer bit pattern would feed the cast a denormal
/// and truncate to zero, and leaving it as an ordinary seeded float hands it a number in
/// `[-4, 4)` whose truncation is as likely to be negative as not.
///
/// MEASURED: a football title's atlas vertex programs open with
/// `pack.int pa[3] <- pa[4]` then `imad o[0] = pa[5] * pa[3] + sa[9]` and load through `o[0]`.
/// `pa[4]` is the blend index, and it is read as a float exactly once, by that cast. There are
/// 65 programs in this corpus whose every load misses and this is the shape of them.
pub fn converts_to_int_operands(instr: &Instr) -> Vec<usize> {
    match instr.op {
        Op::PackToInt { .. } => vec![0],
        _ => vec![],
    }
}

/// Every operand of an instruction that is read through a FLOAT view.
///
/// Stated as the COMPLEMENT of [`integer_sink_operands`] rather than as a second list, because
/// two lists drift and what matters is that a lane read BOTH ways is recognised as read both
/// ways - which only holds if the two lists together cover every operand exactly once.
pub fn float_sink_operands(instr: &Instr) -> Vec<usize> {
    let ints = integer_sink_operands(instr);
    let casts = converts_to_int_operands(instr);
    (0..instr.srcs.len()).filter(|n| !ints.contains(n) && !casts.contains(n)).collect()
}

/// >>> WHICH INSTRUCTIONS A BRANCH CAN SKIP - the RANGE, not "the program has a branch in it".
///
/// `written_lane_precision` answers this with a single flag over the whole shader, and for the
/// question IT asks - is a lane's 16-bit view in doubt - that costs nothing: the conservatism
/// only bites where two writers disagree about a view, which is rare.
///
/// >>> HERE IT IS RUINOUS, AND MEASURED SO. This walk decides whether a write REPLACES a lane's
/// provenance or merely adds to it, so a shader-wide flag means no write ever replaces anything
/// and provenance only ever accumulates - the exact failure the live-state walk was written to
/// remove, reintroduced one level up. A football title's 223-instruction atlas vertex program
/// carries ONE predicate, at instruction 205, and every one of its 56 seeded lanes came back
/// "read both ways" - including `pa[5]`, which is a bit mask three instructions in.
///
/// The range is the emitter's own reading: `crate::wgsl::emit_body` turns a FORWARD branch into
/// a WGSL `if` around the instructions it jumps over, so those instructions - and only those -
/// are the conditional ones. A BACKWARD branch is a loop, whose body is entered by falling into
/// it and so runs at least once; repetition is not skipping and does not put a write in doubt.
fn conditionally_executed(shader: &Shader) -> Vec<bool> {
    let mut conditional = vec![false; shader.instrs.len()];
    for (at, instr) in shader.instrs.iter().enumerate() {
        let Op::Branch { rel } = instr.op else { continue };
        if rel <= 0 {
            continue;
        }
        // The words the branch jumps OVER execute when its condition does not hold - so they
        // are the conditional ones, and the branch itself is not.
        for n in at + 1..(at + rel as usize).min(shader.instrs.len()) {
            conditional[n] = true;
        }
    }
    conditional
}

/// >>> WHETHER A WRITE FULLY REPLACES THE LANE, or leaves some of what was there.
///
/// It decides whether the lane's PROVENANCE is replaced or merged, and getting it wrong in the
/// merging direction is not a harmless conservatism: provenance that only ever accumulates makes
/// every lane an origin of every later value, and the first measurement of this analysis said
/// exactly that - 84 of a program's seeded lanes "read both ways" on a program that reads a
/// handful. An answer that says everything is everything answers nothing.
///
/// A write is full when it writes all 32 bits of the lane AND it certainly runs:
///
/// * a 16-BIT write fills one half, so the pair is only replaced when both of its channels are
///   masked; an 8-bit write fills one byte of four.
/// * a PREDICATED write, or any write in a program with a branch in it, may not run at all -
///   this walk is straight-line and builds no CFG - so the lane may still hold what it held.
fn write_replaces_lane(instr: &Instr, skippable: bool, dest_base: usize, lane: usize) -> bool {
    if skippable {
        return false;
    }
    match instr.op {
        Op::MemLoad { .. } | Op::Tex { .. } | Op::TexGather { .. } => true,
        _ => {
            let Some(dest) = instr.dest.as_ref() else { return false };
            match bank_prec(dest.bank, Prec::of(instr)) {
                Prec::F32 => true,
                // Both channels of the pair this lane holds.
                Prec::F16 => {
                    let c = (lane - dest_base) * 2;
                    instr.write_mask.get(c).copied().unwrap_or(false)
                        && instr.write_mask.get(c + 1).copied().unwrap_or(false)
                }
                // All four bytes of the one register.
                Prec::Fx8 => instr.write_mask.iter().all(|&w| w),
            }
        }
    }
}

/// What a seeded input lane is READ AS, over the whole program.
#[derive(Clone, Copy, PartialEq, Eq, Debug, Default)]
pub struct LaneReads {
    /// Something reads a value this lane decides through an INTEGER view.
    pub integer: bool,
    /// Something reads a value this lane decides as a FLOAT that it immediately TRUNCATES to an
    /// integer - see [`converts_to_int_operands`].
    pub float_to_int: bool,
    /// Something reads a value this lane decides through an ordinary FLOAT view, and keeps it a
    /// float. This is the one that forbids every substitution.
    pub float: bool,
}

impl LaneReads {
    /// >>> A LANE READ ONLY AS AN INTEGER IS THE ONLY ONE THAT MAY BE SEEDED AS A RAW ONE.
    ///
    /// A small integer's BIT PATTERN is a denormal - 4 is `5.6e-45` - so seeding a lane that
    /// anything reads as a float would drive that program's arithmetic to zero and trade one
    /// coverage hole for a worse one. A lane read BOTH ways cannot be both, and the float view
    /// is the one that loses everything, so the float view wins.
    pub fn wants_an_integer(self) -> bool {
        self.integer && !self.float && !self.float_to_int
    }

    /// >>> A LANE WHOSE ONLY FLOAT READER TRUNCATES IT WANTS A SMALL NON-NEGATIVE FLOAT.
    ///
    /// Not an integer bit pattern - the cast would truncate that denormal to zero - and not the
    /// seed's `[-4, 4)` either, half of which is negative and indexes BELOW the bound window's
    /// base. `3.0` is an index; `0xC0800000` is not, and neither is `-2.7`.
    pub fn wants_a_small_float(self) -> bool {
        self.float_to_int && !self.float
    }

    /// >>> A LANE READ AS AN INTEGER **AND** AS SOMETHING ELSE - the shape the two narrow roles
    /// >>> above deliberately refuse, and the whole remainder of the coverage hole.
    ///
    /// MEASURED: with both narrow roles in place, 57 programs still have every guest load miss
    /// every bound window and 8 still spin. Their addressing lane is read through an integer
    /// view AND through a float one, so [`wants_an_integer`](Self::wants_an_integer) refuses to
    /// substitute - correctly, because a small integer's bit pattern is a denormal and the float
    /// reader would get zero.
    ///
    /// The refusal is right as a RULE and wrong as a verdict, and the difference is that a rule
    /// has to be safe for every program while a verdict can be MEASURED on the one in front of
    /// it. [`best_plausible_inputs`] offers these lanes a value only to a program that currently
    /// covers nothing - one whose every load misses, or that does not terminate - and keeps the
    /// candidate only if running the reference on it does better. A program that has something
    /// to lose is never offered the trade at all.
    pub fn read_both_ways(self) -> bool {
        self.integer && (self.float || self.float_to_int)
    }
}

/// >>> HOW EACH SEEDED `pa`/`sa` LANE IS READ, following the value through the program.
///
/// ONE forward walk does both halves of the question, and it has to: the origins of a lane are a
/// LIVE thing that a later write replaces, and an analysis that asked for them after the walk
/// finished would attribute every read to the last value the lane ever held. That is not a
/// hypothetical - it is what the first version of this did, and it reported 84 "read both ways"
/// lanes on programs that read a handful.
///
/// * every writable lane carries the set of seeded `pa`/`sa` lanes its CURRENT content came
///   from; a seeded lane nothing has written yet is its own root.
/// * an instruction marks each operand's live roots as read through an integer or a float view,
///   BEFORE its own destination is updated.
/// * a write that fills the whole lane and certainly runs REPLACES those roots; anything else
///   merges (see [`write_replaces_lane`]).
///
/// The walk is straight-line - it builds no CFG - so on a program with branches it runs the
/// bodies of both arms. That over-approximates in the safe direction for this question: the
/// answer decides which lanes get a plausible integer instead of a float, and a lane wrongly
/// called "read as a float" keeps the seed it has today.
///
/// Returns `(pa, sa)`, one entry per lane. A lane read by nothing comes back as
/// [`LaneReads::default`] - neither - and is left exactly as the seed made it: nothing looks at
/// it, so nothing is gained by changing it.
pub fn seeded_lane_reads(shader: &Shader) -> (Vec<LaneReads>, Vec<LaneReads>) {
    let mut pa = vec![LaneReads::default(); CASE_BANK_LANES];
    let mut sa = vec![LaneReads::default(); CASE_BANK_LANES];
    let mut origins: BTreeMap<(u8, usize), BTreeSet<(u8, usize)>> = BTreeMap::new();
    let conditional = conditionally_executed(shader);

    // The roots a lane holds RIGHT NOW: what the program computed into it, or - for a seeded
    // bank nothing has written - the lane itself.
    let live = |origins: &BTreeMap<(u8, usize), BTreeSet<(u8, usize)>>,
                bank: u8,
                lane: usize|
     -> BTreeSet<(u8, usize)> {
        match origins.get(&(bank, lane)) {
            Some(set) => set.clone(),
            None if is_seeded_bank(bank) => [(bank, lane)].into_iter().collect(),
            None => BTreeSet::new(),
        }
    };

    for (at, instr) in shader.instrs.iter().enumerate() {
        let skippable = conditional[at] || !matches!(instr.pred, Predicate::Always);
        // >>> THE READS ARE MARKED BEFORE THE WRITE, because an instruction reads the lane's old
        // contents and writes its new ones, and a program like `r0 = r0 | 0` would otherwise
        // attribute its own result to itself.
        let mut from: BTreeSet<(u8, usize)> = BTreeSet::new();
        let ints = integer_sink_operands(instr);
        let casts = converts_to_int_operands(instr);
        for which in 0..instr.srcs.len() {
            let Some((bank, lanes)) = src_lanes(instr, which) else { continue };
            for lane in lanes {
                let roots = live(&origins, bank, lane);
                for (rb, rl) in &roots {
                    let slot = match rb {
                        3 => pa.get_mut(*rl),
                        4 => sa.get_mut(*rl),
                        _ => None,
                    };
                    if let Some(slot) = slot {
                        if ints.contains(&which) {
                            slot.integer = true;
                        } else if casts.contains(&which) {
                            slot.float_to_int = true;
                        } else {
                            slot.float = true;
                        }
                    }
                }
                from.extend(roots);
            }
        }
        let Some((dbank, dlanes)) = dest_lanes(instr) else { continue };
        let dest_base = instr.dest.as_ref().map(|d| d.index as usize).unwrap_or(0);
        // >>> A VALUE READ OUT OF A BOUND RESOURCE DOES NOT COME FROM THE REGISTER THAT NAMED
        // >>> IT. A load's destination holds guest-memory BYTES and a sample's holds TEXELS;
        // neither is decided by the seeded lanes that computed the address or the coordinate.
        //
        // MEASURED, and it was the whole remainder of the first honest run: a football title's
        // skinning program loads a matrix row through `pa[4]` and then multiplies that row as a
        // FLOAT, so carrying the pointer's roots into the loaded row marked the blend index, the
        // palette stride and the window base "read as a float" - every one of them an integer,
        // and every one of them then ineligible for the plausible value that is the whole point.
        if matches!(instr.op, Op::MemLoad { .. } | Op::Tex { .. } | Op::TexGather { .. }) {
            from.clear();
        }
        for lane in dlanes {
            let slot = origins.entry((dbank, lane)).or_default();
            if write_replaces_lane(instr, skippable, dest_base, lane) {
                *slot = from.clone();
            } else {
                // A seeded lane that has never been written still holds the seed, and a partial
                // write leaves that half of it there - so the lane keeps itself as a root.
                if is_seeded_bank(dbank) && slot.is_empty() {
                    slot.insert((dbank, lane));
                }
                slot.extend(from.iter().copied());
            }
        }
    }
    (pa, sa)
}

/// >>> A PLAUSIBLE VALUE FOR A LANE THE PROGRAM READS AS AN INTEGER: SMALL, POSITIVE, AND NOT
/// >>> THE SAME ONE EVERYWHERE.
///
/// The three things such a lane turns out to be are an element INDEX, a row STRIDE and a loop
/// TRIP COUNT, and one range suits all three. `1..=15`:
///
/// * NOT ZERO. A stride of zero makes every index address the same row, so an emitter that
///   dropped the index entirely would still agree; a trip count of zero runs no loop body.
/// * SMALL. `index * stride` has to stay inside a bound window - the largest it can reach here
///   is 225 - and a trip count has to terminate inside the interpreter's step budget.
/// * VARIED. A stride and an index that are both `4` would hide an emitter that swapped them,
///   which is exactly the class of defect this differential exists to catch, so the value is
///   drawn per lane from the same avalanche the float seed uses.
///
/// `attempt` selects one of several independent draws; see [`plausible_inputs`] for why there
/// are several. The domain is separated from the float seed's so an integer lane's value never
/// coincides with the float that would otherwise have been there - the same separation
/// `mem_words_for` makes.
pub fn plausible_integer(seed: u32, attempt: u32, lane: u32) -> u32 {
    1 + lane_bits(seed ^ attempt.wrapping_mul(0x0500_0001), 0x0003_0000 ^ lane) % 15
}

/// The same small number as a FLOAT, for a lane whose only float reader truncates it.
///
/// Written through the same draw so a program with one of each gets two values that are as
/// unrelated as the two registers are, and EXACT - `3.0` truncates to 3 under any rounding a
/// cast could use, so nothing here depends on which one the hardware picks.
pub fn plausible_small_float(seed: u32, attempt: u32, lane: u32) -> u32 {
    (plausible_integer(seed, attempt, lane) as f32).to_bits()
}

/// How many independent draws [`plausible_inputs`] offers, INCLUDING the empty one at 0.
///
/// MEASURED rather than picked: over the whole corpus, eight draws land 3,388 of 5,514 guest
/// loads inside a bound window, sixteen land 3,390 and thirty-two land 3,391. The curve is flat
/// by eight, so the extra runs buy nothing and the census says so rather than this carrying a
/// number nobody checked.
pub const PLAUSIBLE_ATTEMPTS: u32 = 8;

/// >>> THE WIDENED BAND: the same eight draws again, but also offering a SMALL WHOLE FLOAT to
/// >>> the lanes a narrow draw refuses - the ones read BOTH as an integer and as a float.
///
/// >>> AND A SMALL WHOLE FLOAT IS THE ONLY THING IT MAY OFFER THEM. A second arm handing those
/// lanes a small integer's BIT PATTERN was built, run over the whole corpus, and REFUTED - by
/// the instruction-level trace rather than by argument, which is the only reason it was caught:
///
/// ```text
///   mad-corpus__vert_9006f1b0    first difference arriving at instruction 6    max
///   gxp-all__hs-gxp__vert_81e5df50  first difference arriving at instruction 2  vtst
/// ```
///
/// A small integer's bit pattern IS a subnormal float - `11` is `1.5e-44` - and WGSL lets an
/// implementation flush subnormals to zero. So the reference reads the lane as `1.5e-44` and the
/// GPU reads it as `0`, and the two run DIFFERENT PROGRAMS from the second instruction on: at
/// `vtst` the flush flips a predicate, the GPU takes the other branch, and six registers come
/// back zero that the reference filled. That is not a subnormal-against-zero difference any
/// grader can excuse - it is a different path, and the numbers it produces are ordinary.
///
/// `1.0..=15.0` is NORMAL on both sides, so no flush can reach it, and an integer reader that
/// truncates still gets its index. It is the only value that is both.
///
/// >>> THESE ARE NOT TRIED ON EVERY PROGRAM. [`best_plausible_inputs`] reaches this band only
/// for a program the narrow band left covering NOTHING - see [`PlausibleScore::covers_nothing`].
pub const PLAUSIBLE_ATTEMPTS_WIDENED: u32 = PLAUSIBLE_ATTEMPTS * 2;

/// >>> THE INPUT OVERRIDES FOR ONE PROGRAM at draw `attempt`: the lanes whose seeded float is
/// >>> replaced by a plausible integer, as `(lane, bits)` over the case's flat `pa` then `sa`
/// >>> numbering.
///
/// The seed itself is unchanged - it is still the program's own content hash, and every lane the
/// program does not read as an integer still holds exactly what `lane_value` gave it. What
/// changes is that a lane the program reads ONLY through an integer view stops holding
/// `0xC0800000`, i.e. 3,229,614,080, and starts holding a number a draw could plausibly have put
/// there.
///
/// >>> ATTEMPT 0 IS EMPTY, AND THAT IS WHAT MAKES THE SEARCH SAFE. It is exactly the behaviour
/// >>> before any of this existed, so [`best_plausible_inputs`] can only ever return something
/// >>> that BEAT it on a measured objective. A lever that cannot make a program worse is not a
/// >>> lever that was lucky; it is one whose floor is the control.
///
/// It is SPARSE and it travels in the case file. The alternative - a second generator in the
/// runner that recomputed the analysis - is the thing this crate's case format exists to avoid:
/// a second copy drifts, and a drift feeds the GPU different inputs from the reference and is
/// reported as a translation defect. The runner cannot run a dataflow analysis over an IR it
/// does not have, so the answer travels as data.
/// An `attempt` at or above [`PLAUSIBLE_ATTEMPTS`] is a WIDENED draw - see that constant's
/// neighbours. The draw's own randomness comes from `attempt` either way, so a widened draw and
/// the narrow draw it extends never hand the same lane the same value by construction.
pub fn plausible_inputs(shader: &Shader, seed: u32, attempt: u32) -> Vec<(usize, u32)> {
    if attempt == 0 {
        return Vec::new();
    }
    let widened = attempt >= PLAUSIBLE_ATTEMPTS;
    let (pa, sa) = seeded_lane_reads(shader);
    let mut out = Vec::new();
    for (bank_of_lane, reads) in [(0usize, &pa), (CASE_BANK_LANES, &sa)] {
        for (n, r) in reads.iter().enumerate() {
            let lane = bank_of_lane + n;
            if r.wants_an_integer() {
                out.push((lane, plausible_integer(seed, attempt, lane as u32)));
            } else if r.wants_a_small_float() {
                out.push((lane, plausible_small_float(seed, attempt, lane as u32)));
            } else if widened && r.read_both_ways() {
                // A SMALL WHOLE FLOAT and never the integer's bit pattern - see the note on
                // [`PLAUSIBLE_ATTEMPTS_WIDENED`] for the two traces that settled it. Something
                // reads this lane as a float, and a subnormal is a value the GPU may flush and
                // the reference will not.
                out.push((lane, plausible_small_float(seed, attempt, lane as u32)));
            }
        }
    }
    out
}

/// What one candidate's run of the REFERENCE came to - the objective the search maximises.
///
/// >>> IT IS THE COVERAGE, NOT A PROXY FOR IT. The three symptoms 21d named are three fields
/// here, measured by running the reference interpreter on the candidate inputs, so a candidate
/// is chosen for doing the thing that was wanted rather than for looking like it might.
#[derive(Clone, Copy, PartialEq, Eq, Debug, Default)]
pub struct PlausibleScore {
    /// The interpretation finished - it did not spin past the step budget and was not refused.
    pub terminated: bool,
    /// Guest-memory loads that landed inside a bound window. A load outside every window reads
    /// ZERO on both sides, so it agrees without having checked anything.
    pub loads_inside: u32,
    /// Guest-memory loads the program issued at all. Not part of [`beats`](Self::beats) - a
    /// candidate that issued MORE loads has not thereby covered more - but the denominator
    /// [`covers_nothing`](Self::covers_nothing) needs: a program that issues no load cannot land
    /// one, and calling that a coverage hole would offer it a trade with nothing to win.
    pub loads_total: u32,
    /// The program left at least one written lane different from what it started with.
    pub wrote_something: bool,
    /// >>> A WRITTEN LANE HOLDS A DENORMAL, which makes this candidate UNUSABLE rather than
    /// >>> merely worse - see [`best_plausible_inputs`], which vetoes it outright.
    ///
    /// MEASURED, and it cost a whole run to find: the widened band put small integer bit
    /// patterns into lanes that arithmetic reads, `11` as a float is `1.5e-44`, and the products
    /// of such values are denormals. The reference keeps them. Every GPU the runner reaches
    /// FLUSHES them to zero - WGSL permits exactly that - so the case's expectation is a value
    /// no conforming implementation has to produce.
    ///
    /// 18 of 23 divergences in the first widened run were this and nothing else: a reference of
    /// `0x0000000b` against a GPU zero, reported as 11 ULP of translation defect. The emitter
    /// was right every time.
    pub writes_a_denormal: bool,
}

impl PlausibleScore {
    /// Strictly better, read in the order the three symptoms matter.
    ///
    /// TERMINATION FIRST, and not as a tie-break: a candidate that spins is not a case at all,
    /// and no number of landed loads makes up for one the harness has to skip. Then the loads,
    /// then whether anything was written - which is last because a program can write plenty and
    /// still have read nothing real.
    /// >>> AND A DENORMAL EXPECTATION IS THE LAST TERM, not a veto - which it was for one run,
    /// >>> and the run that followed is why it is not.
    ///
    /// A subnormal expectation is a lane the GPU is free to answer with zero, so it checks
    /// nothing: WGSL permits flushing subnormals and the runner's grader now says so out loud
    /// (`gxpexec.mjs`, the SUBNORMAL FLUSH counter). Preferring a candidate without one is
    /// therefore right; REFUSING one outright is not, because it throws away every other lane
    /// that candidate checks - measured at 89.8% of guest loads landing inside a window against
    /// 61.5% for the same search with the veto.
    ///
    /// Last, because a lane that checks nothing is still worth less than a load that lands.
    pub fn beats(self, other: Self) -> bool {
        (self.terminated, self.loads_inside, self.wrote_something, !self.writes_a_denormal)
            > (other.terminated, other.loads_inside, other.wrote_something, !other.writes_a_denormal)
    }

    /// >>> THIS PROGRAM IS CHECKING NOTHING ABOUT GUEST MEMORY, so it has nothing to lose.
    ///
    /// Exactly the two symptoms that survived the narrow roles, and nothing else: it did not
    /// terminate (the harness skips it entirely), or it issued loads and EVERY one of them read
    /// the zero that lies outside every bound window - which both sides produce without having
    /// agreed about anything.
    ///
    /// It is the gate on the widened band, and that is the whole safety argument for widening.
    /// A program that lands even one load, or that has no load to land, is never offered a value
    /// for a lane some instruction reads as a float - so the 1,100 cases that work today cannot
    /// be traded away for the 65 that do not.
    pub fn covers_nothing(self) -> bool {
        !self.terminated || (self.loads_total > 0 && self.loads_inside == 0)
    }

}

/// >>> CHOOSE ONE PROGRAM'S INPUTS BY MEASURING WHAT THEY DO, not by trusting the rule that
/// >>> proposed them.
///
/// MEASURED, and this is why it is a search rather than a single draw. Giving every integer-only
/// lane one independent small value moved the all-loads-missed population from 88 to 65 and the
/// SPINNING population from 12 to 21 - because a loop's counter and its bound are two lanes, and
/// two independent draws are as likely to order them so the loop never ends as the float bit
/// patterns were. A rule cannot know that; a run can.
///
/// `run` interprets the program with the given overrides applied and reports what happened. The
/// caller owns that because it owns the textures, the windows and the memory words; this owns
/// only the candidates and the comparison.
///
/// >>> AND THE NARROW BAND IS TRIED FIRST, ALWAYS. Only a program the eight narrow draws left
/// >>> covering nothing - see [`PlausibleScore::covers_nothing`] - reaches the widened draws,
/// which are the only ones allowed to hand a value to a lane something reads as a FLOAT. That
/// ordering is what makes the widening safe rather than lucky: a program with coverage to lose
/// is never offered the trade, and one with none has a floor of exactly what it has today.
///
/// Deterministic: the attempts are tried in order, `beats` is strict, and attempt 0 is the empty
/// override set - so the same blob always yields the same inputs, and a program no candidate
/// improves keeps exactly the inputs it had before this existed.
pub fn best_plausible_inputs(
    shader: &Shader,
    seed: u32,
    mut run: impl FnMut(&[(usize, u32)]) -> PlausibleScore,
) -> (Vec<(usize, u32)>, PlausibleScore) {
    let mut best = Vec::new();
    let mut best_score = run(&best);
    // A program with no substitutable lane at all has nothing to try in EITHER band; every
    // attempt would be the empty set again and the runs would be wasted. Both bands are asked,
    // because a program can have no narrow lane and several both-ways ones.
    if plausible_inputs(shader, seed, 1).is_empty()
        && plausible_inputs(shader, seed, PLAUSIBLE_ATTEMPTS).is_empty()
    {
        return (best, best_score);
    }
    // >>> THE CONTROL IS NEVER VETOED, only the candidates. Candidate 0 is what the program had
    // before any of this existed; refusing it would move the floor this whole search stands on,
    // and a program whose own seed happens to produce a denormal is a fact about that program
    // rather than a choice this made.
    for attempt in 1..PLAUSIBLE_ATTEMPTS_WIDENED {
        // The widened draws are offered ONLY to a program the narrow ones left covering
        // nothing - the gate, and the whole safety argument for widening at all.
        if attempt == PLAUSIBLE_ATTEMPTS && !best_score.covers_nothing() {
            break;
        }
        let cand = plausible_inputs(shader, seed, attempt);
        if cand.is_empty() {
            continue;
        }
        let score = run(&cand);
        if score.beats(best_score) {
            best = cand;
            best_score = score;
        }
    }
    (best, best_score)
}

/// Apply chosen overrides to a register file that has already had its seeded fill.
///
/// A helper rather than three call sites, because the ORDER matters and is easy to get wrong in
/// one of them: the overrides land after the seed and BEFORE the window bases, so a window's
/// guest base - an address the harness chose, not a number the program computes - still wins on
/// the lane the driver places it in.
pub fn apply_plausible_inputs(regs: &mut RegFile, overrides: &[(usize, u32)]) {
    for &(lane, bits) in overrides {
        let slot = if lane < CASE_BANK_LANES {
            regs.pa.get_mut(lane)
        } else {
            regs.sa.get_mut(lane - CASE_BANK_LANES)
        };
        if let Some(slot) = slot {
            *slot = f32::from_bits(bits);
        }
    }
}

/// The `pa` baseline a case's expectation is measured against, with the overrides applied.
///
/// `changed_pairs` compares the finished `pa` bank against what the program STARTED with, and
/// that baseline is no longer `lane_value` alone. A baseline that forgot the overrides would
/// report every overridden lane as a lane the program changed, on every case that has one.
pub fn seeded_pa_baseline(seed: u32, overrides: &[(usize, u32)]) -> Vec<f32> {
    let mut pa: Vec<f32> = (0..CASE_BANK_LANES).map(|n| lane_value(seed, n as u32)).collect();
    for &(lane, bits) in overrides {
        if lane < CASE_BANK_LANES {
            pa[lane] = f32::from_bits(bits);
        }
    }
    pa
}
