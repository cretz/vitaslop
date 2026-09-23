//! **The graphics conformance suite: authored shader programs with a STATED INTENT.**
//!
//! # The gap this closes
//!
//! Two instruments already check the shader path, and neither can say what a program MEANS:
//!
//! * The corpus tests (`tests/corpus.rs`) ask whether a captured blob parses, links, and
//!   compiles under naga and Tint. Every graphics defect this project has chased was a module
//!   that did all four perfectly and computed the wrong number.
//! * The corpus DIFFERENTIAL (`tests/execcases.rs` + `vitaslop-web/e2e/gxpexec.mjs`) runs the
//!   emitted module on the real GPU and diffs it against the reference interpreter. That is far
//!   stronger - it found five defects in the reference in one session - but its own header says
//!   the limit plainly: both halves are our reading of the ISA, so **where both are wrong in the
//!   same way, they agree**.
//!
//! A case here is a program this project WROTE, for a reason, with the answer stated in Rust
//! before anything runs. Three parties must then agree - the intent, the reference interpreter,
//! and the real GPU through Tint - and if they do not, the intent says which of the other two is
//! wrong rather than only that they differ.
//!
//! # Where each case came from
//!
//! Every case in the first group is a defect this project actually shipped, found months after
//! the fact because some title drew something visibly wrong. Each is now a five-millisecond
//! question with no title, no recipe and no frame:
//!
//! | case | the defect it would have caught |
//! |---|---|
//! | `the_inline_constants_are_per_channel` | the reference read constant bank 0 for all four channels; a program computing `pos * 1.0` computed `pos * 0.0` - a collapsed mesh |
//! | `fract_is_x_minus_floor_x` | `f32::fract` keeps the sign, WGSL's does not; seven vertex programs were producing outputs ~10^4 times the GPU's |
//! | `a_mad_write_mask_can_select_lane_two_alone` | the two masking tables are ONE per-lane bitmask; the lost bit dropped lane 2 of every skinned vec3 and smeared a title's characters into streaks |
//! | `an_f16_write_packs_two_channels_into_one_register` | the reference held one f32 per lane, so it had no authority over two thirds of the corpus |
//! | `an_f16_store_saturates_where_f32_overflows` | an overflowed F16 store became an infinity, which the next subtract made a NaN, poisoning everything after it |
//! | `a_partial_write_leaves_the_other_lanes_alone` | a half-precision write of ONE lane claimed the whole register and suppressed a literal; a title's batter rendered 100% black |
//!
//! # What a pass here proves, and what it does not
//!
//! It proves the whole shipped path - container parse, USSE decode, link, WGSL emission, Tint,
//! the GPU - carries a stated meaning through unchanged. It does NOT prove that meaning is the
//! real SGX543's, because the program was ASSEMBLED through this project's own reading of the
//! ISA: a field we encode wrong we also decode wrong. Closing THAT gap needs a blob a real
//! `SceShaccCg` produced or a program a real Vita runs, which is the conformance app's job.
//!
//! ```text
//! cargo test -p vitaslop-gxp-shader --test conformance
//! VITASLOP_GXP_CASES_OUT=<dir> cargo test -p vitaslop-gxp-shader --test conformance -- --ignored --nocapture
//! node vitaslop-web/e2e/gxpexec.mjs <dir>
//! ```

use vitaslop_gxp_shader::gxpwrite::{self, ProgramSpec, VertexOutputs};
use vitaslop_gxp_shader::interp::{self, RegFile};
use vitaslop_gxp_shader::ir::{Bank, Op};
use vitaslop_gxp_shader::usse::asm::{self, Dest, Src, MAD_F32_XY};
use vitaslop_gxp_shader::wgsl::{
    wrap_compute_module_for, wrap_render_case_module_ramped, CASE_BANK_LANES, CASE_RAMP_DX,
    CASE_RAMP_DY, CASE_RAMP_LANE,
};
use vitaslop_gxp_shader::{recompile_fragment, recompile_vertex, Program, ProgramKind};

mod common;
use common::{changed_pairs, lane_value, nonzero_pairs, write_case, written_lane_precision, Stage};

/// The seed every conformance case's input register file is generated from.
///
/// ONE seed for the whole suite, fixed here rather than derived from each program's content
/// hash the way a corpus case's is. A conformance case is authored to make a specific
/// disagreement visible, so its inputs must not move when its code is edited - a case that
/// changed its own inputs whenever the program changed could pass for a reason nobody chose.
const SEED: u32 = 0x00C0_FFEE;

/// The input register file every case starts from: PA and SA seeded from [`lane_value`],
/// everything else zero. Identical to the corpus harness's, and reproduced bit for bit in the
/// JS runner from the seed alone.
fn seeded_inputs() -> RegFile {
    let mut regs = RegFile::with_lanes(CASE_BANK_LANES);
    for n in 0..CASE_BANK_LANES {
        regs.pa[n] = lane_value(SEED, n as u32);
        regs.sa[n] = lane_value(SEED, (n + CASE_BANK_LANES) as u32);
    }
    regs
}

/// One lane of the expected result: which bank, which lane, and the exact 32-bit pattern.
///
/// The pattern, not a float, because an F16 lane is a PACKED PAIR of halves and a lane holding
/// four unorm bytes is neither - a comparison made in the wrong view turns a one-ULP difference
/// in a high half into a difference of thousands. See `written_lane_precision`.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
struct Lane {
    bank: Bank,
    lane: usize,
    bits: u32,
}

/// The expected value of an f32 register lane.
fn f32_lane(bank: Bank, lane: usize, v: f32) -> Lane {
    Lane { bank, lane, bits: v.to_bits() }
}

/// The expected value of a register holding two F16 CHANNELS.
///
/// The packing is stated here, in the test, rather than borrowed from the emitter: channel `c`
/// is half `c & 1` of register `base + (c >> 1)`, low half first. That IS the claim a case
/// about F16 storage is making, so restating it is the point - a test that asked the emitter
/// where the halves go could not catch the emitter putting them in the wrong place.
///
/// The half-precision ROUNDING comes from the crate (`f32_to_f16_bits`); every value these
/// cases use is exactly representable in binary16, so no rounding decision is being borrowed.
fn f16_lane(bank: Bank, lane: usize, low: f32, high: f32) -> Lane {
    let bits = u32::from(vitaslop_gxp_shader::fold::f32_to_f16_bits(low))
        | (u32::from(vitaslop_gxp_shader::fold::f32_to_f16_bits(high)) << 16);
    Lane { bank, lane, bits }
}

/// The stand-in texture the reference fetches, in whichever form the EMITTER is emitting.
///
/// The arm is read here rather than at the two call sites, because a call site that forgot it
/// would hand the reference a constant while the module sampled a varying stand-in - and the
/// Rust suite would stay GREEN, because it runs the reference against the intent and never
/// touches the emitted WGSL. Only `gxpexec` would see it, on a GPU, later.
fn case_textures(unit: u8, coord: [f32; 4], lod: interp::TexLodArg) -> Option<[f32; 4]> {
    use vitaslop_gxp_shader::wgsl::{case_tex_varies, case_texture_value, case_texture_value_at};
    Some(if case_tex_varies() {
        case_texture_value_at(unit, [coord[0], coord[1], coord[2]], lod)
    } else {
        case_texture_value(unit)
    })
}

/// One conformance case.
struct Case {
    /// The case's name, and the stem of the files it writes for the GPU runner.
    name: &'static str,
    /// What a failure here would MEAN, in one line. Printed beside a failure, because the
    /// question a case answers is not recoverable from the register numbers.
    checks: &'static str,
    spec: ProgramSpec,
    /// The result the program was written to produce, as a function of the seeded inputs.
    ///
    /// Every lane NOT named here must be zero. That is not a convenience: a case that listed
    /// only the lanes it cared about could not see an instruction writing a register it had no
    /// business touching, which is exactly the shape of the write-mask and register-packing
    /// defects this suite exists for.
    intent: fn(&RegFile) -> Vec<Lane>,
    /// >>> WHAT THE FRAGMENT PIPELINE ITSELF MUST COME TO, for a case that runs on the RENDER
    /// >>> rig. `None` means a compute case, which is what every case in the suite was.
    ///
    /// A discard and a written depth leave the register file untouched, so `intent` cannot say
    /// anything about either: a translation that DROPPED a `kill` agrees on every lane and
    /// still paints a pixel the hardware discards, and one that wrote the depth from the wrong
    /// channel agrees on every lane and sorts the pixel wrongly. They are the whole of what
    /// these cases check, so they are their own field.
    fragment: Option<fn(&RegFile) -> FragmentIntent>,
}

/// The two words a render case carries past the four register banks.
#[derive(Clone, Copy, PartialEq, Debug)]
struct FragmentIntent {
    /// Whether the fragment was discarded. Compared EXACTLY - "within one ULP of killed" is not
    /// a thing - so it travels at precision code 3, the raw-word view.
    killed: bool,
    /// The depth the program WROTE, in the guest's own depth space, or `None` for a program
    /// that replaces no depth. The rig applies its one declared forward map (a clamp) on both
    /// sides; which of the four real maps is in force is a property of the DRAW, not the
    /// program.
    depth: Option<f32>,
}

/// `o[0..3] = pa[0..3]` - the whole path end to end on one instruction, and the case every
/// other one is read against: if this fails, nothing below means anything.
fn case_move() -> Case {
    Case {
        name: "conf_move_copies_four_lanes",
        checks: "a four-lane move carries every channel of its source to its destination",
        spec: vertex_spec(vec![asm::mov(
            false,
            Dest::new(Bank::Output, 0),
            [true; 4],
            Src::reg(Bank::PrimaryAttr, 0),
        )
        .unwrap()]),
        intent: |regs| {
            (0..4).map(|c| f32_lane(Bank::Output, c, regs.pa[c])).collect()
        },
        fragment: None,
    }
}

/// **THE CONSTANT OPERAND'S VALUE IS PER CHANNEL.** Selectors 4..7 are the inline constants
/// 0.0/1.0/2.0/0.5, and they are chosen per channel by the SWIZZLE - not once per operand from
/// its index.
///
/// The corpus differential found this by accident, on a four-instruction program where the
/// shader computed `pos * 1.0` and the reference computed `pos * 0.0` - a clip position of
/// zero, i.e. a collapsed mesh. It is not only an oracle fault: the interpreter decides the
/// per-pass DEPTH FIT and the clip-`w` sign that arms the negative-projection correction, so a
/// constant read wrong is a projection decided wrong on the shipped path.
///
/// Here it is a question with an answer: each channel is multiplied by a DIFFERENT inline
/// constant, so a reader that takes one value for all four cannot pass by coincidence.
fn case_per_channel_constants() -> Case {
    Case {
        name: "conf_inline_constants_are_per_channel",
        checks: "swizzle selectors 4..7 select the inline constants 0.0/1.0/2.0/0.5 PER CHANNEL",
        spec: vertex_spec(vec![asm::alu(
            Op::Mul,
            false,
            Dest::new(Bank::Output, 0),
            [true; 4],
            Src::cnst(0).swz([4, 5, 6, 7]),
            Src::reg(Bank::PrimaryAttr, 0),
        )
        .unwrap()]),
        intent: |regs| {
            const K: [f32; 4] = [0.0, 1.0, 2.0, 0.5];
            (0..4).map(|c| f32_lane(Bank::Output, c, K[c] * regs.pa[c])).collect()
        },
        fragment: None,
    }
}

/// **`fract` IS `x - floor(x)`,** which for a negative operand differs from `x - trunc(x)` by
/// exactly 1.0.
///
/// The reference used Rust's `f32::fract`, which keeps the sign. A fractional part is normally
/// the range reduction in front of a polynomial, so that 1.0 gets squared and scaled: seven
/// 99-instruction vertex programs of one title were producing outputs about 10^4 times the
/// GPU's. The seeded inputs span `[-4, 4)`, so this case has negative operands by construction.
fn case_fract() -> Case {
    Case {
        name: "conf_fract_is_x_minus_floor_x",
        checks: "fract() is x - floor(x) - NOT x - trunc(x), which differs by 1.0 for x < 0",
        spec: vertex_spec(vec![asm::alu(
            Op::Frc,
            false,
            Dest::new(Bank::Output, 0),
            [true; 4],
            Src::reg(Bank::PrimaryAttr, 0),
            Src::reg(Bank::PrimaryAttr, 0),
        )
        .unwrap()]),
        intent: |regs| {
            (0..4)
                .map(|c| {
                    let x = regs.pa[c];
                    f32_lane(Bank::Output, c, x - x.floor())
                })
                .collect()
        },
        fragment: None,
    }
}

/// **A MAD'S WRITE MASK IS A PER-LANE BITMASK, AND ITS THIRD BIT SELECTS LANE 2.**
///
/// The published tables give the mask as two truth tables that between them leave one control
/// bit unused, and reading them as tables loses lane 2. The combination the tables do not cover
/// - `m16=1, m32=0, en=0` - then decodes as an instruction that writes NOTHING, and a shipped
/// compiler does not emit one of those. It is the tail of a skinning accumulate: the z of a
/// vec3 whose first two lanes are the mad before it.
///
/// MEASURED when it was found: with lane 2 lost, a title's characters render as a fan of
/// streaks. Here the two instructions write three different lanes with three different masks,
/// so a mask reading that drops any one of them fails on the lane it dropped.
fn case_mad_write_mask() -> Case {
    let base = asm::mad(
        false,
        Dest::new(Bank::Output, 0),
        [true, true, false, false],
        Src::reg(Bank::PrimaryAttr, 0).swz(MAD_F32_XY),
        Src::reg(Bank::SecondaryAttr, 0).swz(MAD_F32_XY),
        Src::reg(Bank::SecondaryAttr, 2).swz(MAD_F32_XY),
    )
    .unwrap();
    // The combination the two published tables do not cover: lane 2 alone.
    let lane2 = asm::mad(
        false,
        Dest::new(Bank::Output, 0),
        [false, false, true, false],
        Src::reg(Bank::PrimaryAttr, 0).swz(MAD_F32_XY),
        Src::reg(Bank::SecondaryAttr, 0).swz(MAD_F32_XY),
        Src::reg(Bank::SecondaryAttr, 2).swz(MAD_F32_XY),
    )
    .unwrap();
    Case {
        name: "conf_mad_write_mask_selects_lane_two_alone",
        checks: "the mad's three mask bits are ONE per-lane bitmask: en=lane0, m32=lane1, m16=lane2",
        spec: vertex_spec(vec![base, lane2]),
        intent: |regs| {
            // Both instructions carry the `xy` operand swizzle, so every channel reads either
            // channel x or channel y of its source - channel c of the product is
            // `pa[c & 1] * sa[c & 1] + sa[2 + (c & 1)]`.
            let term = |c: usize| regs.pa[c & 1] * regs.sa[c & 1] + regs.sa[2 + (c & 1)];
            vec![
                f32_lane(Bank::Output, 0, term(0)),
                f32_lane(Bank::Output, 1, term(1)),
                f32_lane(Bank::Output, 2, term(2)),
            ]
        },
        fragment: None,
    }
}

/// **AN F16 WRITE PACKS TWO CHANNELS INTO ONE REGISTER,** low half first, so four channels
/// occupy the register PAIR `base, base+1`.
///
/// The reference held one f32 per lane and so had no authority over the 433 of 670 corpus cases
/// that carry half-precision instructions - which is where the F16 varying packing and the F16
/// lighting lanes live. Where the shader reads a half, the reference read the word's bit pattern
/// as a single float: a denormal or a huge number, never the value.
///
/// The inline constants give four DIFFERENT known channel values, so a packing that puts a
/// channel in the wrong half - or in the wrong register of the pair - cannot pass.
fn case_f16_packing() -> Case {
    Case {
        name: "conf_f16_write_packs_two_channels_per_register",
        checks: "an F16 channel c is half (c & 1) of register base + (c >> 1), low half first",
        spec: vertex_spec(vec![asm::alu(
            Op::Mul,
            true,
            Dest::new(Bank::Output, 0),
            [true; 4],
            // 0.0, 1.0, 2.0, 0.5 against the literal 4, so the four channels are
            // 0, 4, 8, 2 - four distinct, exactly-representable values.
            Src::cnst(0).swz([4, 5, 6, 7]),
            Src::imm(4),
        )
        .unwrap()]),
        intent: |_| {
            vec![
                f16_lane(Bank::Output, 0, 0.0, 4.0),
                f16_lane(Bank::Output, 1, 8.0, 2.0),
            ]
        },
        fragment: None,
    }
}

/// **AN F16 STORE SATURATES; IT DOES NOT OVERFLOW TO INFINITY.**
///
/// The reference overflowed to infinity, an infinity propagates, and the next subtract makes it
/// a NaN - so ONE overflowed store poisons everything after it. When the differential found
/// this, 200 diverging lanes held a non-finite reference against a finite GPU value, and the
/// GPU's value was repeatedly 65504: the largest finite binary16.
///
/// The operands are inline LITERALS rather than seeded registers, so the overflow happens
/// whatever the seed - a case that depended on its inputs being large enough would be a case
/// that sometimes checked nothing.
fn case_f16_saturation() -> Case {
    let square = asm::alu(
        Op::Mul,
        true,
        Dest::new(Bank::Temp, 0),
        [true; 4],
        Src::imm(63),
        Src::imm(63),
    )
    .unwrap();
    let cube = asm::alu(
        Op::Mul,
        true,
        Dest::new(Bank::Output, 0),
        [true; 4],
        Src::imm(63),
        Src::reg(Bank::Temp, 0),
    )
    .unwrap();
    Case {
        name: "conf_f16_store_saturates_rather_than_overflowing",
        checks: "an F16 result past binary16's range SATURATES to 65504, never becomes infinity",
        spec: vertex_spec(vec![square, cube]),
        intent: |_| {
            // 63 * 63 = 3969, which binary16 holds as 3968; 3968 * 63 = 249984, far past
            // 65504. Both halves of both destination registers carry the same saturated value,
            // and the intermediate is checked too - a reference that saturated only at the end
            // would still be wrong about what the register in between holds.
            vec![
                f16_lane(Bank::Temp, 0, 3968.0, 3968.0),
                f16_lane(Bank::Temp, 1, 3968.0, 3968.0),
                f16_lane(Bank::Output, 0, 65504.0, 65504.0),
                f16_lane(Bank::Output, 1, 65504.0, 65504.0),
            ]
        },
        fragment: None,
    }
}

/// **A PARTIAL WRITE LEAVES THE OTHER LANES ALONE.**
///
/// A half-precision write of ONE lane once claimed the whole register, which suppressed the
/// literal that belonged in the other half; a title's batter rendered 100% black. The general
/// property is simply that a masked write is masked - and the way to check it is to write a
/// register fully, then write ONE channel of it, and require the other three to survive.
fn case_partial_write() -> Case {
    let fill = asm::mov(
        false,
        Dest::new(Bank::Output, 0),
        [true; 4],
        Src::reg(Bank::PrimaryAttr, 0),
    )
    .unwrap();
    let overwrite_x = asm::alu(
        Op::Mul,
        false,
        Dest::new(Bank::Output, 0),
        [true, false, false, false],
        Src::cnst(0).swz([6, 6, 6, 6]), // the inline 2.0
        Src::reg(Bank::PrimaryAttr, 4),
    )
    .unwrap();
    Case {
        name: "conf_a_partial_write_leaves_the_other_lanes_alone",
        checks: "a masked write touches ONLY its masked channels; the rest keep what was there",
        spec: vertex_spec(vec![fill, overwrite_x]),
        intent: |regs| {
            vec![
                f32_lane(Bank::Output, 0, 2.0 * regs.pa[4]),
                f32_lane(Bank::Output, 1, regs.pa[1]),
                f32_lane(Bank::Output, 2, regs.pa[2]),
                f32_lane(Bank::Output, 3, regs.pa[3]),
            ]
        },
        fragment: None,
    }
}

/// `min`/`max` are per channel, and a source MODIFIER applies before the operation - not after.
///
/// An abs/neg modifier read as a post-operation step is the kind of thing that is right on
/// every operand whose value happens to be positive, which in a seeded register file is half of
/// them.
fn case_modifiers() -> Case {
    let negated = asm::alu(
        Op::Min,
        false,
        Dest::new(Bank::Output, 0),
        [true; 4],
        Src::reg(Bank::PrimaryAttr, 0).negated(),
        Src::reg(Bank::SecondaryAttr, 0),
    )
    .unwrap();
    let absolute = asm::alu(
        Op::Max,
        false,
        Dest::new(Bank::Output, 4),
        [true; 4],
        Src::reg(Bank::PrimaryAttr, 0).absolute(),
        Src::reg(Bank::SecondaryAttr, 0),
    )
    .unwrap();
    Case {
        name: "conf_source_modifiers_apply_before_the_operation",
        checks: "negate and absolute modify the OPERAND, so min(-a, b) and max(|a|, b) per channel",
        spec: vertex_spec(vec![negated, absolute]),
        intent: |regs| {
            let mut out = Vec::new();
            for c in 0..4 {
                out.push(f32_lane(Bank::Output, c, (-regs.pa[c]).min(regs.sa[c])));
                out.push(f32_lane(Bank::Output, 4 + c, regs.pa[c].abs().max(regs.sa[c])));
            }
            out
        },
        fragment: None,
    }
}

/// A source SWIZZLE reads the named channel of the source, per destination channel.
///
/// The swizzle is the operand feature most able to be wrong in a way that looks right: a
/// program whose swizzle is read as identity computes something plausible out of the wrong
/// components, and only a case that asks for a NON-identity permutation can see it.
fn case_swizzle() -> Case {
    Case {
        name: "conf_a_swizzle_permutes_the_source_channels",
        checks: "destination channel c reads source channel swizzle[c], not channel c",
        spec: vertex_spec(vec![asm::alu(
            Op::Add,
            false,
            Dest::new(Bank::Output, 0),
            [true; 4],
            // zxyw: every channel reads a DIFFERENT source channel from its own, except w.
            Src::reg(Bank::PrimaryAttr, 0).swz([2, 0, 1, 3]),
            Src::cnst(0).swz([0, 0, 0, 0]),
        )
        .unwrap()]),
        intent: |regs| {
            const S: [usize; 4] = [2, 0, 1, 3];
            // The second operand is CNST6 entry 0 read through swizzle selector 0, which is
            // constant bank 0's entry 0 - and that entry is zero, so this is a permuting move.
            (0..4).map(|c| f32_lane(Bank::Output, c, regs.pa[S[c]])).collect()
        },
        fragment: None,
    }
}

/// **A NARROWING AND A WIDENING PACK ARE INVERSES, AND THE PACKED PAIR IS THE LAYOUT IN
/// BETWEEN.**
///
/// Real fragment programs are 70-90% half precision, so almost every value in one crosses this
/// instruction twice. It is also where a wrong number is least visible: a conversion that reads
/// the wrong half of a packed register yields a denormal or a huge number, never an error.
///
/// The values are inline literals, all exactly representable in binary16, so the round trip is
/// EXACT and no rounding decision is being borrowed from the code under test. The intermediate
/// register is checked as well as the result - a conversion pair that was wrong in both
/// directions by the same amount would otherwise come back looking correct.
fn case_pack_round_trip() -> Case {
    let fill = asm::mov(false, Dest::new(Bank::Temp, 0), [true; 4], Src::imm(5)).unwrap();
    let narrow = asm::pack(
        Dest::new(Bank::Temp, 8),
        asm::PackFmt::F16,
        Bank::Temp,
        0,
        asm::PackFmt::F32,
        [0, 1, 2, 3],
        [true; 4],
        false,
    )
    .unwrap();
    let widen = asm::pack(
        Dest::new(Bank::Output, 0),
        asm::PackFmt::F32,
        Bank::Temp,
        8,
        asm::PackFmt::F16,
        [0, 1, 2, 3],
        [true; 4],
        false,
    )
    .unwrap();
    Case {
        name: "conf_a_pack_round_trip_through_f16_is_exact_for_a_representable_value",
        checks: "F32 -> F16 -> F32 returns the value, and the register in between holds two halves",
        spec: vertex_spec(vec![fill, narrow, widen]),
        intent: |_| {
            vec![
                // The f32 source the literal filled.
                f32_lane(Bank::Temp, 0, 5.0),
                f32_lane(Bank::Temp, 1, 5.0),
                f32_lane(Bank::Temp, 2, 5.0),
                f32_lane(Bank::Temp, 3, 5.0),
                // The packed intermediate: four channels in TWO registers.
                f16_lane(Bank::Temp, 8, 5.0, 5.0),
                f16_lane(Bank::Temp, 9, 5.0, 5.0),
                // And back, exactly.
                f32_lane(Bank::Output, 0, 5.0),
                f32_lane(Bank::Output, 1, 5.0),
                f32_lane(Bank::Output, 2, 5.0),
                f32_lane(Bank::Output, 3, 5.0),
            ]
        },
        fragment: None,
    }
}

/// **THE NORMALIZED U8 CONVERSION MAPS 0..255 ONTO 0..1, BOTH WAYS.**
///
/// It shares an opcode with the plain truncating cast and differs by one bit, so the two are
/// exactly the pair a decoder can confuse - and confusing them turns a colour of 1.0 into a
/// colour of 255.0, or of 255 into 255/255. Only the two endpoints are used, because they are
/// the two values whose round trip is exact whatever the rounding rule; a case built on 0.5
/// would be asserting a rounding decision instead of the mapping.
fn case_unorm8_round_trip() -> Case {
    let fill = asm::alu(
        Op::Add,
        false,
        Dest::new(Bank::Temp, 0),
        [true; 4],
        Src::cnst(0).swz([4, 5, 4, 5]), // 0.0, 1.0, 0.0, 1.0
        Src::cnst(0).swz([0, 0, 0, 0]), // constant bank 0 entry 0, which is zero
    )
    .unwrap();
    let to_bytes = asm::pack(
        Dest::new(Bank::Temp, 8),
        asm::PackFmt::U8,
        Bank::Temp,
        0,
        asm::PackFmt::F32,
        [0, 1, 2, 3],
        [true; 4],
        true,
    )
    .unwrap();
    let from_bytes = asm::pack(
        Dest::new(Bank::Output, 0),
        asm::PackFmt::F32,
        Bank::Temp,
        8,
        asm::PackFmt::U8,
        [0, 1, 2, 3],
        [true; 4],
        true,
    )
    .unwrap();
    Case {
        name: "conf_the_normalised_u8_conversion_maps_zero_and_one_exactly",
        checks: "the NORMALIZED u8 conversion is byte/255, not a truncating cast - both directions",
        spec: vertex_spec(vec![fill, to_bytes, from_bytes]),
        intent: |_| {
            // A lane whose value is ZERO is not named: both sides start every written bank at
            // zero, so a zero lane and an unwritten one are the same observation and naming it
            // would assert nothing. Only the 1.0 channels appear.
            vec![
                f32_lane(Bank::Temp, 1, 1.0),
                f32_lane(Bank::Temp, 3, 1.0),
                // All four channels are BYTES of ONE register, channel c in byte c with byte 0
                // least significant: (0, 255, 0, 255) is the word 0xff00ff00, not 0x00ff00ff.
                Lane { bank: Bank::Temp, lane: 8, bits: 0xff00_ff00 },
                f32_lane(Bank::Output, 1, 1.0),
                f32_lane(Bank::Output, 3, 1.0),
            ]
        },
        fragment: None,
    }
}

/// **AN EQUAL-WIDTH INTEGER REPACK MOVES BIT PATTERNS AND CONVERTS NOTHING.**
///
/// A `U16 -> U16` VPCK with `scale` clear is a swizzled copy of HALVES: neither end converts,
/// so whatever 16 bits the selector names arrive in the destination half unaltered. The
/// swizzle here SWAPS the two halves of each source register, which makes the case fail in
/// three distinguishable ways rather than one - a copy that ignored the swizzle leaves the
/// word alone, one that read halves in the wrong register order swaps the two registers, and
/// one that widened or sign-extended anything produces a value neither half holds.
///
/// This is the family the reference interpreter refused for 31 corpus programs on the grounds
/// that "this register file cannot hold a packed 16-bit half pair". It can - a lane is a
/// 32-bit word - and the arm that now models it took its element addressing from the same
/// decoder the emitter reads. That is exactly the shape where both halves can be wrong
/// together, so the claim is stated HERE, over an input the case chose, as a third answer.
fn case_int_half_repack() -> Case {
    let swap_halves = asm::pack(
        Dest::new(Bank::Temp, 0),
        asm::PackFmt::U16,
        Bank::PrimaryAttr,
        0,
        asm::PackFmt::U16,
        // Channel c takes element `swizzle[c]` of the four halves that span pa[0] and pa[1]:
        // half 0 is pa[0]'s low, half 1 its high, half 2 pa[1]'s low, half 3 its high.
        [1, 0, 3, 2],
        [true; 4],
        false,
    )
    .unwrap();
    Case {
        name: "conf_an_equal_width_int_repack_moves_bit_patterns",
        checks: "a U16->U16 VPCK copies halves verbatim, addressed by the selector - no convert",
        spec: vertex_spec(vec![swap_halves]),
        intent: |regs| {
            // Destination channel c is half `c & 1` of register `c >> 1` - the same packing
            // `f16_lane` states, restated here because it is part of the claim.
            let rot = |w: u32| w.rotate_left(16);
            vec![
                Lane { bank: Bank::Temp, lane: 0, bits: rot(regs.pa[0].to_bits()) },
                Lane { bank: Bank::Temp, lane: 1, bits: rot(regs.pa[1].to_bits()) },
            ]
        },
        fragment: None,
    }
}

/// **A PARTIALLY-MASKED 16-BIT REPACK WRITES THE HALF ITS CHANNEL NAMES, AND LEAVES THE OTHER
/// HALF OF THAT REGISTER ALONE.**
///
/// >>> THIS IS THE SHAPE THE CORPUS ACTUALLY SHIPS, AND NOTHING COVERED IT. Every
/// `PackIntCopy` in the captured corpus carries a SINGLE-CHANNEL write mask and not one
/// carries a full one, while both authored repack cases above use a full mask - so the form
/// seven real compilers emit was checked by no case at all. The differential cannot stand in
/// for one either: the emitter and the reference now agree here by construction (they read
/// `Instr::dest_raw_packed_bits`, which states the placement once), and where both halves read
/// the same rule they agree whether it is right or wrong.
///
/// The discriminating mask is channel 1 ALONE, because that is where the two candidate rules
/// part company:
///
/// * CHANNEL-INDEXED placement - what the hardware does and what ships - puts channel `c` in
///   half `c & 1` of register `dest + (c >> 1)`. Channel 1 is therefore the HIGH half of
///   `dest`, and the low half keeps whatever was there.
/// * COMPACTED placement - writing the written channels into consecutive halves from the
///   start - puts the one written channel in the LOW half instead. That is not a hypothetical
///   rule: it is what a whole-register store does one level up, and the 16-bit `PackToInt`
///   reference arm did exactly that until this session's predecessor, putting a skinned mesh's
///   four blend indices in four registers where its four bone fetches look in two.
///
/// The SELECTOR is compacted or not by the same argument, so the swizzle is `[3, 0, 0, 0]`
/// rather than something uniform: the written channel's own entry is 0 (the first half of the
/// source span) while entry 0 - what a compacted reading would take - is 3, the last. The two
/// readings therefore name DIFFERENT source halves as well as different destination halves,
/// and a case where they happened to name the same one would pass under either.
///
/// The first instruction is not setup for its own sake: it puts a KNOWN pair of halves in
/// `r[0]` so that "the other half is left alone" is a claim about a value, not about the zero
/// the register file started at. A rule that cleared the partner half would pass a case whose
/// partner half was already zero.
fn case_partial_mask_half_repack() -> Case {
    // r[0] = pa[1], as two halves: channel 0 takes half 2 (pa[1]'s low), channel 1 half 3.
    let seed_pair = asm::pack(
        Dest::new(Bank::Temp, 0),
        asm::PackFmt::U16,
        Bank::PrimaryAttr,
        0,
        asm::PackFmt::U16,
        [2, 3, 0, 0],
        [true, true, false, false],
        false,
    )
    .unwrap();
    // ...then ONE channel: r[0]'s HIGH half takes half 0 (pa[0]'s low), and nothing else moves.
    let one_half = asm::pack(
        Dest::new(Bank::Temp, 0),
        asm::PackFmt::U16,
        Bank::PrimaryAttr,
        0,
        asm::PackFmt::U16,
        [3, 0, 0, 0],
        [false, true, false, false],
        false,
    )
    .unwrap();
    Case {
        name: "conf_a_partial_mask_half_repack_writes_the_half_its_channel_names",
        checks: "a single-channel U16->U16 VPCK writes half (c & 1) of register dest + (c >> 1), \
                 leaves the partner half, and reads the selector entry of that same channel",
        spec: vertex_spec(vec![seed_pair, one_half]),
        intent: |regs| {
            // The halves of the source span, stated here rather than borrowed: half `n` is
            // register `n >> 1`, low half first.
            let half = |n: usize| -> u32 {
                let w = regs.pa[n >> 1].to_bits();
                if n & 1 == 1 { w >> 16 } else { w & 0xffff }
            };
            // Low half: what the first instruction left (half 2). High half: what the second
            // wrote (half 0). `r[1]` and everything else must still be zero, which the suite's
            // whole-register-file comparison is what checks.
            vec![Lane { bank: Bank::Temp, lane: 0, bits: half(2) | (half(0) << 16) }]
        },
        fragment: None,
    }
}

/// **A SIGNED 16-BIT HALF WIDENS WITH ITS SIGN, AND ITS SELECTOR PICKS THE HALF.**
///
/// `S16 -> F32` with `scale` clear is the truncating widen, the mirror of the float->int pack:
/// the named half is read as a signed 16-bit integer and becomes that integer as a float. Two
/// claims in one case, and both have a history:
///
/// * THE SIGN. A half whose top bit is set is a negative number, not a value near 65535. The
///   widening is the same shift pair the packed-operand integer multiply uses, and a third
///   rule anywhere would be a third answer.
/// * THE HALF SELECTION. The reference refused this family partly on the argument that
///   modelling it "would make the oracle agree with an emitter that had the half selection
///   wrong" - a real risk, and the reason the selectors here are `[1, 0, 3, 2]`: a rule that
///   read the halves in register order would produce the same four numbers in the wrong
///   channels, which a straight `[0,1,2,3]` case could not see.
fn case_signed_half_widen() -> Case {
    let widen = asm::pack(
        Dest::new(Bank::Output, 0),
        asm::PackFmt::F32,
        Bank::PrimaryAttr,
        0,
        asm::PackFmt::S16,
        [1, 0, 3, 2],
        [true; 4],
        false,
    )
    .unwrap();
    Case {
        name: "conf_a_signed_int_half_widens_with_its_sign",
        checks: "an S16->F32 VPCK sign-extends the half its selector names, and only that half",
        spec: vertex_spec(vec![widen]),
        intent: |regs| {
            // Half `n` of the span: register `n >> 1`, low half first.
            let half = |n: usize| -> i16 {
                let w = regs.pa[n >> 1].to_bits();
                (if n & 1 == 1 { w >> 16 } else { w & 0xffff }) as u16 as i16
            };
            [1usize, 0, 3, 2]
                .iter()
                .enumerate()
                .map(|(c, &sel)| f32_lane(Bank::Output, c, f32::from(half(sel))))
                .collect()
        },
        fragment: None,
    }
}

/// **A TEXTURE SAMPLE WRITES FOUR CHANNELS, WHATEVER ITS COORDINATE COUNT.**
///
/// The destination EXTENT is part of what the opcode means: a translation that wrote only as
/// many channels as the coordinate had would leave the rest holding whatever was in the
/// register, which is the kind of defect that renders as a plausible wrong colour.
///
/// The bound texture is the harness's constant stand-in, and a constant-valued texture is not
/// an approximation of a real one for this question - a unit bound to a texture whose every
/// texel holds the same value returns that value for any coordinate, at any level, under any
/// filter. What this case therefore does NOT check is the coordinate arithmetic; that needs a
/// texture that varies with position, in a fragment-stage rig.
fn case_texture_sample() -> Case {
    // The sampler ORDINAL is 12, so the instruction names SA register 24. The DATA container
    // starts at SA 20, so the texture-control entry for unit 0 sits at index 4 within it -
    // `20 + 4 = 24`, and that identity is the whole of what the resolution does. A sample whose
    // ordinal and whose table entry do not meet is refused by the emitter, by name.
    let sample = asm::tex(Dest::new(Bank::Temp, 0), 12, Bank::PrimaryAttr, 0, 2).unwrap();
    let store = asm::mov(false, Dest::new(Bank::Output, 0), [true; 4], Src::reg(Bank::Temp, 0))
        .unwrap();
    Case {
        name: "conf_a_texture_sample_writes_all_four_channels",
        checks: "a 2D sample writes RGBA to four consecutive destination channels",
        spec: ProgramSpec::fragment(vec![sample, store])
            .with_parameters(vec![gxpwrite::ParamSpec::sampler("tex0", 0)])
            .with_interpolants(vec![gxpwrite::InterpolantSpec::texcoord(0, 4)])
            .with_texture_control(vec![(4, 0)])
            .with_default_uniform_regs(16)
            .with_containers(vec![
                gxpwrite::ContainerSpec { index: 14, base_sa: 0, size_regs: 16 },
                gxpwrite::ContainerSpec { index: 16, base_sa: 16, size_regs: 4 },
                gxpwrite::ContainerSpec { index: 19, base_sa: 20, size_regs: 8 },
            ])
            .with_registers(4, 28, 8),
        // The sample reads TWO coordinates from `pa[0]`/`pa[1]`, so with the varying stand-in
        // armed the value is a function of those two seeded lanes - which is the point: a
        // translation that read the wrong registers for the coordinate now changes the answer,
        // where against a constant texture it did not.
        intent: |regs| {
            let t = case_textures(0, [regs.pa[0], regs.pa[1], 0.0, 0.0], interp::TexLodArg::IMPLICIT)
                .expect("a stand-in");
            (0..4)
                .flat_map(|c| {
                    [f32_lane(Bank::Temp, c, t[c]), f32_lane(Bank::Output, c, t[c])]
                })
                .collect()
        },
        fragment: None,
    }
}

/// **EACH `lod_mode` IS ITS OWN BUILTIN, FED FROM `src2`.**
///
/// The corpus's grip on three of the four modes is thin - BIAS rests on ONE shipped word, LEVEL
/// on 28, GRADIENT on 7 - so one blob leaving the corpus would take a whole WGSL builtin out of
/// every check. These state the mode and the register it reads outright.
///
/// >>> WHAT A ONE-MIP RIG CAN AND CANNOT SEE. At one mip `textureSample`, `textureSampleBias` and
/// `textureSampleLevel` return the same texel, so a real texture could not tell them apart. The
/// compute rig's stand-in folds the builtin's identity and its LOD operand into the result
/// instead (`case_texture_value_at`), so these catch the wrong builtin, a LOD read from the wrong
/// register, and a gradient's four channels in the wrong slots. What they cannot catch is the
/// SAMPLER's reading of a level - which mip a bias of 0.5 selects - and that is the hardware's
/// arithmetic, not the translation's.
///
/// The LOD operand lives at `pa[4..8]`, a DIFFERENT register from the coordinate at `pa[0..2]`,
/// so a translation that fed the coordinate in as the level changes the answer.
fn lod_sample_case(
    name: &'static str,
    checks: &'static str,
    lod: vitaslop_gxp_shader::ir::TexLod,
    intent: fn(&RegFile) -> Vec<Lane>,
) -> Case {
    let sample =
        asm::tex_lod(Dest::new(Bank::Temp, 0), 12, Bank::PrimaryAttr, 0, 2, lod, Some((Bank::PrimaryAttr, 4)))
            .unwrap();
    let store = asm::mov(false, Dest::new(Bank::Output, 0), [true; 4], Src::reg(Bank::Temp, 0))
        .unwrap();
    Case {
        name,
        checks,
        spec: ProgramSpec::fragment(vec![sample, store])
            .with_parameters(vec![gxpwrite::ParamSpec::sampler("tex0", 0)])
            .with_interpolants(vec![
                gxpwrite::InterpolantSpec::texcoord(0, 4),
                gxpwrite::InterpolantSpec::texcoord(1, 4),
            ])
            .with_texture_control(vec![(4, 0)])
            .with_default_uniform_regs(16)
            .with_containers(vec![
                gxpwrite::ContainerSpec { index: 14, base_sa: 0, size_regs: 16 },
                gxpwrite::ContainerSpec { index: 16, base_sa: 16, size_regs: 4 },
                gxpwrite::ContainerSpec { index: 19, base_sa: 20, size_regs: 8 },
            ])
            .with_registers(8, 28, 8),
        intent,
        fragment: None,
    }
}

/// The four lanes a LOD sample writes to `r[0..4]` and copies to `o[0..4]`, given the stand-in's
/// answer for it.
fn sampled_lanes(t: [f32; 4]) -> Vec<Lane> {
    (0..4)
        .flat_map(|c| [f32_lane(Bank::Temp, c, t[c]), f32_lane(Bank::Output, c, t[c])])
        .collect()
}

fn case_tex_bias() -> Case {
    use vitaslop_gxp_shader::ir::TexLod;
    lod_sample_case(
        "conf_a_biased_sample_is_textureSampleBias_reading_src2_x",
        "lod_mode 1 emits textureSampleBias with src2 channel 0 as the bias",
        TexLod::Bias,
        |regs| {
            let lod = interp::TexLodArg { mode: TexLod::Bias, args: [regs.pa[4], 0.0, 0.0, 0.0] };
            sampled_lanes(case_textures(0, [regs.pa[0], regs.pa[1], 0.0, 0.0], lod).unwrap())
        },
    )
}

fn case_tex_level() -> Case {
    use vitaslop_gxp_shader::ir::TexLod;
    lod_sample_case(
        "conf_an_explicit_level_sample_is_textureSampleLevel_reading_src2_x",
        "lod_mode 2 emits textureSampleLevel with src2 channel 0 as the level",
        TexLod::Level,
        |regs| {
            let lod = interp::TexLodArg { mode: TexLod::Level, args: [regs.pa[4], 0.0, 0.0, 0.0] };
            sampled_lanes(case_textures(0, [regs.pa[0], regs.pa[1], 0.0, 0.0], lod).unwrap())
        },
    )
}

fn case_tex_gradient() -> Case {
    use vitaslop_gxp_shader::ir::TexLod;
    lod_sample_case(
        "conf_a_gradient_sample_reads_ddx_then_ddy_from_src2",
        "lod_mode 3 emits textureSampleGrad with ddx = src2.xy and ddy = src2.zw, in that order",
        TexLod::Gradient,
        |regs| {
            let lod = interp::TexLodArg {
                mode: TexLod::Gradient,
                args: [regs.pa[4], regs.pa[5], regs.pa[6], regs.pa[7]],
            };
            sampled_lanes(case_textures(0, [regs.pa[0], regs.pa[1], 0.0, 0.0], lod).unwrap())
        },
    )
}

/// **A TEST WRITES ONE PREDICATE BIT, AND A PREDICATED WRITE OBEYS IT - AND ITS NEGATION.**
///
/// `vtst` was a real defect family before it could be written a case at all (an F16 test read
/// as one F32 by the reference flipped a branch in five programs), and the predicated write is
/// what every branch-free conditional compiles to. Two tests into two predicate registers, with
/// a DIFFERENT relation and a DIFFERENT reduction channel each, then four writes guarded by
/// `p0`, `!p0`, `p1`, `!p1` - so a swapped predicate register, an inverted negation table, the
/// wrong reduction channel or the wrong relation each moves a whole register of the answer.
///
/// `p0 = pa[0] - pa[4] > 0` (channel 0) and `p1 = pa[1] - pa[5] < 0` (channel 1).
fn case_test_and_predicated_writes() -> Case {
    use vitaslop_gxp_shader::ir::{Predicate, TestAlu, TestCmp, TestReduce};
    let t0 = asm::vtst(TestAlu::Sub, TestCmp::Gt, TestReduce::Channel(0), 0, false, Bank::PrimaryAttr, 0, Bank::PrimaryAttr, 4)
        .unwrap();
    let t1 = asm::vtst(TestAlu::Sub, TestCmp::Lt, TestReduce::Channel(1), 1, false, Bank::PrimaryAttr, 0, Bank::PrimaryAttr, 4)
        .unwrap();
    let guarded = |pred, dest: u8, src: u8| {
        asm::alu_pred(
            Op::Add,
            false,
            pred,
            Dest::new(Bank::Temp, dest),
            [true; 4],
            Src::cnst(0).swz([4, 4, 4, 4]),
            Src::reg(Bank::SecondaryAttr, src),
        )
        .unwrap()
    };
    Case {
        name: "conf_a_test_sets_a_predicate_that_guards_writes_and_their_negation",
        checks: "vtst's relation, reduction channel and predicate register, and ExtVecPredicate's P/NEGP rows",
        spec: vertex_spec(vec![
            t0,
            t1,
            guarded(Predicate::IfP(0), 0, 0),
            guarded(Predicate::IfNotP(0), 4, 4),
            guarded(Predicate::IfP(1), 8, 8),
            guarded(Predicate::IfNotP(1), 12, 12),
        ]),
        intent: |regs| {
            let p0 = regs.pa[0] - regs.pa[4] > 0.0;
            let p1 = regs.pa[1] - regs.pa[5] < 0.0;
            let mut out = Vec::new();
            for (taken, base) in [(p0, 0usize), (!p0, 4), (p1, 8), (!p1, 12)] {
                if taken {
                    out.extend((0..4).map(|c| f32_lane(Bank::Temp, base + c, 0.0 + regs.sa[base + c])));
                }
            }
            out
        },
        fragment: None,
    }
}

/// **THE FOUR TRANSCENDENTALS ARE SCALAR, READ ONE COMPONENT, AND BROADCAST.**
///
/// Group 0x30 picks ONE source channel and writes `f(it)` to every channel its mask names. So
/// each of the four reads a DIFFERENT component under a DIFFERENT mask, and two carry a source
/// modifier - which is what a normalize (`rsq(|dot|)`) and a fog term (`exp2(-d)`) actually
/// compile to. A translation that read the destination channel's own component instead of the
/// selected one, or wrote the unmasked channels, moves a lane the intent pins.
///
/// The GPU's transcendentals are approximate within a few ULP; the grader's four-ULP tolerance
/// is what reads them, and the reference computes the correctly-rounded values.
fn case_transcendentals() -> Case {
    let r = |op, dest: u8, mask, comp, abs, neg| {
        asm::vcomp(op, false, Dest::new(Bank::Temp, dest), mask, Bank::PrimaryAttr, 0, comp, abs, neg).unwrap()
    };
    Case {
        name: "conf_the_transcendentals_broadcast_one_selected_component",
        checks: "rcp/rsq/log2/exp2 op2 numbering, src_comp selection, the abs/neg modifier and the mask",
        spec: vertex_spec(vec![
            r(Op::Rcp, 0, [true, false, false, false], 0, false, false),
            r(Op::Rsq, 2, [true, true, false, false], 1, true, false),
            r(Op::Log, 4, [true; 4], 2, true, false),
            r(Op::Exp, 8, [false, true, false, true], 3, false, true),
        ]),
        intent: |regs| {
            let p = &regs.pa;
            let mut out = vec![f32_lane(Bank::Temp, 0, 1.0 / p[0])];
            out.extend((2..4).map(|l| f32_lane(Bank::Temp, l, 1.0 / p[1].abs().sqrt())));
            out.extend((4..8).map(|l| f32_lane(Bank::Temp, l, p[2].abs().log2())));
            out.extend([9, 11].map(|l| f32_lane(Bank::Temp, l, (-p[3]).exp2())));
            out
        },
        fragment: None,
    }
}

/// **A MOVE IS A TABLE SWIZZLE, A CONDITIONAL MOVE IS A PER-CHANNEL SELECT, AND THE BYTE FORM
/// SELECTS BYTES.**
///
/// Group 0x38's own `mov` has never been authored - `asm::mov` is an ADD of zero in the ALU group
/// - so the vec4-standard swizzle table it reads, including the entry whose channel 3 is the
/// CONSTANT one, was checked by nothing that states it. Then a float select against `< 0`, and
/// the byte-wise select a skinned mesh uses to pick bone indices.
///
/// >>> THE BYTE SELECT PINS THE SHIPPED READING OF AN OPEN QUESTION. Whether VMOVCU8 tests EACH
/// byte of its test operand or tests BYTE 0 and moves every masked byte on that one answer is not
/// settled - every captured word carries the full mask, where the two agree unless the test
/// operand's bytes differ - and the shipped default is the one-byte reading
/// (`module::cmov_u8_tests_each_byte`, `VITASLOP_GXP_CMOVU8=byte` for the other). This case is
/// built to SEPARATE them: its first test register is `r[7] = 1.0` from the move above, bytes
/// `00 00 80 3f`, so the two readings disagree on bytes 2 and 3. Flipping the default must fail
/// here, and whoever flips it restates this intent with the evidence that settled it.
///
/// The second select tests `pa[4]`, whose low byte the seed makes non-zero, so the OTHER arm is
/// taken - under a partial mask, which is four BYTES of one register and not four registers.
fn case_moves() -> Case {
    use asm::MoveKind;
    use vitaslop_gxp_shader::ir::CompareMethod;
    let t = Bank::Temp;
    Case {
        name: "conf_moves_swizzle_by_table_and_select_by_channel_and_byte",
        checks: "VMOV's vec4-standard table (incl. XYZ1), VMOVC's src0 test and arm order, VMOVCU8's byte lanes",
        spec: vertex_spec(vec![
            asm::vmov(MoveKind::Mov, false, Dest::new(t, 0), [true; 4], (Bank::PrimaryAttr, 0), [2, 0, 1, 3], None, None)
                .unwrap(),
            asm::vmov(MoveKind::Mov, false, Dest::new(t, 4), [true; 4], (Bank::PrimaryAttr, 4), [0, 1, 2, 5], None, None)
                .unwrap(),
            asm::vmov(
                MoveKind::Cmov(CompareMethod::LtZero),
                false,
                Dest::new(t, 8),
                [true; 4],
                (Bank::SecondaryAttr, 0),
                [0, 1, 2, 3],
                Some((Bank::SecondaryAttr, 4)),
                Some((Bank::PrimaryAttr, 0)),
            )
            .unwrap(),
            asm::vmov(
                MoveKind::CmovU8(CompareMethod::EqZero),
                false,
                Dest::new(t, 13),
                [true; 4],
                (Bank::SecondaryAttr, 8),
                [0, 1, 2, 3],
                Some((Bank::SecondaryAttr, 9)),
                Some((t, 7)),
            )
            .unwrap(),
            asm::vmov(
                MoveKind::CmovU8(CompareMethod::EqZero),
                false,
                Dest::new(t, 14),
                [true, false, true, false],
                (Bank::SecondaryAttr, 8),
                [0, 1, 2, 3],
                Some((Bank::SecondaryAttr, 9)),
                Some((Bank::PrimaryAttr, 4)),
            )
            .unwrap(),
        ]),
        intent: |regs| {
            let (p, s) = (&regs.pa, &regs.sa);
            let mut out = Vec::new();
            for (l, v) in [p[2], p[0], p[1], p[3]].into_iter().enumerate() {
                out.push(f32_lane(Bank::Temp, l, v));
            }
            for (l, v) in [p[4], p[5], p[6], 1.0].into_iter().enumerate() {
                out.push(f32_lane(Bank::Temp, 4 + l, v));
            }
            for c in 0..4 {
                out.push(f32_lane(Bank::Temp, 8 + c, if p[c] < 0.0 { s[c] } else { s[4 + c] }));
            }
            // The ONE-BYTE reading: byte 0 of the test operand decides every masked byte.
            let (yes, no) = (s[8].to_bits().to_le_bytes(), s[9].to_bits().to_le_bytes());
            let select = |test: f32, mask: [bool; 4]| -> u32 {
                let held = test.to_bits().to_le_bytes()[0] == 0;
                let b: [u8; 4] =
                    std::array::from_fn(|i| if !mask[i] { 0 } else if held { yes[i] } else { no[i] });
                u32::from_le_bytes(b)
            };
            out.push(Lane { bank: Bank::Temp, lane: 13, bits: select(1.0, [true; 4]) });
            out.push(Lane { bank: Bank::Temp, lane: 14, bits: select(p[4], [true, false, true, false]) });
            out
        },
        fragment: None,
    }
}

/// **A LIMM LOADS A RAW 32-BIT PATTERN, WHOLE, INTO ONE LANE - INCLUDING ITS TOP BIT.**
///
/// The immediate is assembled from four fields and its top bit sits in the group's
/// `opcat_extra` position, so the patterns are chosen to exercise every field: `1.5` (a float,
/// top bit clear), `-2^31` as a float (the running-maximum floor a shipped program starts from -
/// top bit SET), and `0x7FFFFFFF` (INT_MAX, every low field full). The destinations are ODD,
/// because the field is undoubled and a doubling defect cannot hide at an odd register.
fn case_limm() -> Case {
    Case {
        name: "conf_limm_loads_a_whole_raw_pattern_into_one_lane",
        checks: "LIMM's four immediate fields incl. the top bit at 54, and its undoubled destination",
        spec: vertex_spec(vec![
            asm::limm(Dest::new(Bank::Temp, 1), 1.5f32.to_bits()).unwrap(),
            asm::limm(Dest::new(Bank::Temp, 3), 0xCF00_0000).unwrap(),
            asm::limm(Dest::new(Bank::Temp, 5), 0x7FFF_FFFF).unwrap(),
        ]),
        intent: |_| {
            vec![
                Lane { bank: Bank::Temp, lane: 1, bits: 1.5f32.to_bits() },
                Lane { bank: Bank::Temp, lane: 3, bits: 0xCF00_0000 },
                Lane { bank: Bank::Temp, lane: 5, bits: 0x7FFF_FFFF },
            ]
        },
        fragment: None,
    }
}

/// **A FORWARD CONDITIONAL BRANCH SKIPS EXACTLY THE WORDS IT JUMPS OVER.**
///
/// `br p0 +2` skips one instruction when `p0` holds - what the emitter reconstructs as a WGSL
/// `if`. The instruction AFTER the target must run either way, which is where an off-by-one in
/// the displacement lands.
fn case_forward_branch() -> Case {
    use vitaslop_gxp_shader::ir::{Predicate, TestAlu, TestCmp, TestReduce};
    let copy = |dest: u8, src: u8| {
        asm::alu(Op::Add, false, Dest::new(Bank::Temp, dest), [true; 4], Src::cnst(0).swz([4, 4, 4, 4]), Src::reg(Bank::SecondaryAttr, src))
            .unwrap()
    };
    Case {
        name: "conf_a_forward_branch_skips_the_words_it_jumps_over",
        checks: "BR's forward displacement in instruction words and its ExtPredicate condition",
        spec: vertex_spec(vec![
            asm::vtst(TestAlu::Sub, TestCmp::Gt, TestReduce::Channel(0), 0, false, Bank::PrimaryAttr, 0, Bank::PrimaryAttr, 4)
                .unwrap(),
            asm::br(Predicate::IfP(0), 2).unwrap(),
            copy(0, 0),
            copy(4, 4),
        ]),
        intent: |regs| {
            let skipped = regs.pa[0] - regs.pa[4] > 0.0;
            let mut out: Vec<Lane> = (0..4).map(|c| f32_lane(Bank::Temp, 4 + c, 0.0 + regs.sa[4 + c])).collect();
            if !skipped {
                out.extend((0..4).map(|c| f32_lane(Bank::Temp, c, 0.0 + regs.sa[c])));
            }
            out
        },
        fragment: None,
    }
}

/// **A BACKWARD BRANCH IS A LOOP, AND IT RUNS AS MANY TIMES AS ITS CONDITION SAYS.**
///
/// The compiler's loop shape - test at the head, a forward exit branch on its negation, the
/// body, an unconditional back edge - which the emitter turns into a WGSL `loop` breaking on the
/// exit. The counter starts at zero and the bound is 2.0, so the body runs TWICE: a translation
/// that ran it once (straight-line) or three times (an off-by-one on either branch) leaves a
/// different multiple of `sa[0..4]` in `r[4..8]`.
///
/// ```text
///   0: r8   = 2.0                  the bound
///   1: p0   = r0.x - r8.x < 0      head
///   2: br !p0 -> 6                 exit
///   3: r0  += 1.0
///   4: r4  += sa[0..4]
///   5: br   -> 1                   back edge
///   6: r16  = r4                   after the loop
/// ```
fn case_loop() -> Case {
    use vitaslop_gxp_shader::ir::{Predicate, TestAlu, TestCmp, TestReduce};
    let t = Bank::Temp;
    Case {
        name: "conf_a_backward_branch_loops_as_many_times_as_its_condition_says",
        checks: "BR's backward (signed) displacement, the loop reconstruction, and its exit polarity",
        spec: vertex_spec(vec![
            asm::alu(Op::Add, false, Dest::new(t, 8), [true; 4], Src::cnst(0).swz([6, 6, 6, 6]), Src::reg(t, 12)).unwrap(),
            asm::vtst(TestAlu::Sub, TestCmp::Lt, TestReduce::Channel(0), 0, false, t, 0, t, 8).unwrap(),
            asm::br(Predicate::IfNotP(0), 4).unwrap(),
            asm::alu(Op::Add, false, Dest::new(t, 0), [true; 4], Src::cnst(0).swz([5, 5, 5, 5]), Src::reg(t, 0)).unwrap(),
            asm::alu(Op::Add, false, Dest::new(t, 4), [true; 4], Src::reg(t, 4), Src::reg(Bank::SecondaryAttr, 0)).unwrap(),
            asm::br(Predicate::Always, -4).unwrap(),
            asm::alu(Op::Add, false, Dest::new(t, 16), [true; 4], Src::cnst(0).swz([4, 4, 4, 4]), Src::reg(t, 4)).unwrap(),
        ]),
        intent: |regs| {
            let mut out = Vec::new();
            for c in 0..4 {
                let acc = (0.0 + regs.sa[c]) + regs.sa[c];
                out.push(f32_lane(Bank::Temp, c, 2.0));
                out.push(f32_lane(Bank::Temp, 4 + c, acc));
                out.push(f32_lane(Bank::Temp, 8 + c, 2.0));
                out.push(f32_lane(Bank::Temp, 16 + c, 0.0 + acc));
            }
            out
        },
        fragment: None,
    }
}

/// **`dsx` IS THE X DERIVATIVE AND `dsy` THE Y ONE, OF THE CHANNEL THE SWIZZLE NAMES.**
///
/// A RENDER case, because only a fragment stage has a quad to difference across - and the rig's
/// SCREEN RAMP on `pa[100]` (`CASE_RAMP_LANE`) is what gives one input a gradient: exactly
/// `CASE_RAMP_DX` along x and `CASE_RAMP_DY` along y, two DIFFERENT powers of two. So
/// `dpdx` emitted for `dsx` and `dpdy` for `dsy` each land a distinct answer, a derivative of the
/// wrong channel lands 0 where the ramp should be (or the ramp where 0 should be), a dropped
/// negate lands the wrong sign, and a derivative of a lane with no ramp must come back +0.
///
/// ```text
///   r[0..4]   = dsx(pa[98].xyzw)    ->  0, 0, DX, 0     (only channel z reads pa[100])
///   r[4..8]   = dsy(pa[98].zzzz)    ->  DY x 4
///   r[8..12]  = dsx(-pa[98].zzzz)   -> -DX x 4
///   r[12..16] = dsy(pa[0].xyzw)     ->  0 x 4
/// ```
fn case_derivatives() -> Case {
    let d = |op, dest: u8, src: Src| {
        asm::alu(op, false, Dest::new(Bank::Temp, dest), [true; 4], src, Src::reg(Bank::PrimaryAttr, 0)).unwrap()
    };
    let pa98 = Src::reg(Bank::PrimaryAttr, 98);
    Case {
        name: "conf_dsx_and_dsy_differentiate_the_channel_their_swizzle_names",
        checks: "ALU opcode2 3/4 are dpdx/dpdy, of src1's swizzled channel, under its negate",
        spec: ProgramSpec::fragment(vec![
            d(Op::Dsx, 0, pa98),
            d(Op::Dsy, 4, pa98.swz([2, 2, 2, 2])),
            d(Op::Dsx, 8, pa98.swz([2, 2, 2, 2]).negated()),
            d(Op::Dsy, 12, Src::reg(Bank::PrimaryAttr, 0)),
        ])
        .with_interpolants(vec![gxpwrite::InterpolantSpec::texcoord(0, 4)])
        .with_registers(104, 0, 16),
        intent: |_| {
            let mut out = vec![f32_lane(Bank::Temp, 2, CASE_RAMP_DX)];
            for c in 0..4 {
                out.push(f32_lane(Bank::Temp, 4 + c, CASE_RAMP_DY));
                out.push(f32_lane(Bank::Temp, 8 + c, -CASE_RAMP_DX));
            }
            out
        },
        fragment: Some(|_| FragmentIntent { killed: false, depth: None }),
    }
}

/// **THE 8-BIT COMBINER: COLOUR AND ALPHA ARE TWO OPERATIONS OVER THE SAME TWO TERMS, AND EACH
/// TERM'S COEFFICIENT IS A SELECTOR, NOT AN OPERAND.**
///
/// Two registers of four unorm bytes, loaded by LIMM so every byte is stated. Then:
///
/// * a LERP - `src1 * src2.a + src2 * (1 - src2.a)`, colour ADD, alpha MAX - the blend a fragment
///   epilogue does when it combines in the byte domain;
/// * a SUBTRACT with a partial mask - colour `src1*src1 - 1*src2` (clamping below zero), alpha
///   MIN, channel y left alone - which exercises the ROTATED write-mask field (alpha stored in
///   bit 0), the complement of a ZERO coefficient, and the op split at channel 3;
/// * the whole-register byte COPY an epilogue ends with.
///
/// Every value is `byte / 255` in, `round(clamp(v) * 255)` out - the rule is stated here, not
/// borrowed; a one-byte disagreement at a rounding boundary is what the runner's u8x4 view reads.
fn case_sop2() -> Case {
    use vitaslop_gxp_shader::ir::{SopFactor as F, SopOp};
    let t = Bank::Temp;
    Case {
        name: "conf_the_byte_combiner_splits_colour_and_alpha_over_selected_coefficients",
        checks: "SOP2M's op/selector/complement fields, its rotated write mask, and the group-0x80 byte copy",
        spec: vertex_spec(vec![
            asm::limm(Dest::new(t, 0), 0xFFC0_8040).unwrap(),
            asm::limm(Dest::new(t, 1), 0x8020_E010).unwrap(),
            asm::sop2(SopOp::Add, SopOp::Max, F::Src2Alpha, false, F::Src2Alpha, true, Dest::new(t, 2), [true; 4], (t, 0), (t, 1))
                .unwrap(),
            asm::sop2(
                SopOp::Sub,
                SopOp::Min,
                F::Src1Color,
                false,
                F::Zero,
                true,
                Dest::new(t, 3),
                [true, false, true, true],
                (t, 0),
                (t, 1),
            )
            .unwrap(),
            asm::copy_fx8(Dest::new(t, 5), (t, 2)).unwrap(),
        ]),
        intent: |_| {
            let (a, b) = (0xFFC0_8040u32, 0x8020_E010u32);
            let ch = |w: u32, c: usize| ((w >> (8 * c)) & 0xff) as f32 / 255.0;
            let store = |v: f32| ((v.clamp(0.0, 1.0) * 255.0) + 0.5) as u32 & 0xff;
            let pack = |f: &dyn Fn(usize) -> Option<f32>| -> u32 {
                (0..4).fold(0u32, |acc, c| acc | f(c).map_or(0, |v| store(v) << (8 * c)))
            };
            let lerp = pack(&|c| {
                let t1 = ch(b, 3) * ch(a, c);
                let t2 = (1.0 - ch(b, 3)) * ch(b, c);
                Some(if c == 3 { t1.max(t2) } else { t1 + t2 })
            });
            let sub = pack(&|c| {
                if c == 1 {
                    return None;
                }
                let t1 = ch(a, c) * ch(a, c);
                let t2 = (1.0 - 0.0) * ch(b, c);
                Some(if c == 3 { t1.min(t2) } else { t1 - t2 })
            });
            [(0, a), (1, b), (2, lerp), (3, sub), (5, lerp)]
                .into_iter()
                .map(|(lane, bits)| Lane { bank: Bank::Temp, lane, bits })
                .collect()
        },
        fragment: None,
    }
}

/// **A GATHER WRITES FOUR TEXELS, THEN FOUR F16 BILINEAR COEFFICIENTS - COEFFICIENT `k` WEIGHTING
/// TEXEL `3 - k`.**
///
/// The shadow filter's instruction, and the only one whose destination is SIX registers: the 2x2
/// footprint's texels (one component each, full precision) in `r[0..4]` in the platform's order
/// `(x0,y1) (x1,y1) (x1,y0) (x0,y0)`, then the four weights in `r[4..6]` as F16 pairs. The
/// pairing is the claim - coefficient 0 is `(1-fx)(1-fy)`, the weight of texel 3 at `(x0,y0)` -
/// and the consumer dots the two groups, so a reversed pairing is a filter that weights every
/// texel by its opposite corner's weight.
///
/// A RENDER case with a REAL texture: the rig's quantiser puts the coordinate a quarter-texel
/// from the footprint boundary with the fraction free in `[0.25, 0.75]`, so every weight varies
/// and none is a constant 0.75 that would check nothing. The texels and fractions come from the
/// rig's own model of its texture (`case_render_gather`); what is stated here is where they go.
fn case_gather() -> Case {
    let gather = asm::tex_gather(Dest::new(Bank::Temp, 0), 12, Bank::PrimaryAttr, 0).unwrap();
    Case {
        name: "conf_a_gather_writes_four_texels_then_coefficients_paired_in_reverse",
        checks: "sb_mode 3: texel order, the six-register destination, F16 coefficient packing, k <-> 3-k",
        spec: ProgramSpec::fragment(vec![gather])
            .with_parameters(vec![gxpwrite::ParamSpec { components: 1, ..gxpwrite::ParamSpec::sampler("shadow", 0) }])
            .with_interpolants(vec![gxpwrite::InterpolantSpec::texcoord(0, 4)])
            .with_texture_control(vec![(4, 0)])
            .with_default_uniform_regs(16)
            .with_containers(vec![
                gxpwrite::ContainerSpec { index: 14, base_sa: 0, size_regs: 16 },
                gxpwrite::ContainerSpec { index: 16, base_sa: 16, size_regs: 4 },
                gxpwrite::ContainerSpec { index: 19, base_sa: 20, size_regs: 8 },
            ])
            .with_registers(4, 28, 8),
        intent: |regs| {
            let (texels, [fx, fy]) =
                vitaslop_gxp_shader::wgsl::case_render_gather(0, [regs.pa[0], regs.pa[1]]);
            let w = [(1.0 - fx) * (1.0 - fy), fx * (1.0 - fy), fx * fy, (1.0 - fx) * fy];
            let mut out: Vec<Lane> = (0..4).map(|c| f32_lane(Bank::Temp, c, texels[c])).collect();
            out.push(f16_lane(Bank::Temp, 4, w[0], w[1]));
            out.push(f16_lane(Bank::Temp, 5, w[2], w[3]));
            out
        },
        fragment: Some(|_| FragmentIntent { killed: false, depth: None }),
    }
}

/// **A MEMORY LOAD READS CONSECUTIVE GUEST WORDS FROM `pointer + offsets`, AN IMMEDIATE OFFSET
/// COUNTING ELEMENTS AND A REGISTER ONE COUNTING BYTES.**
///
/// The skinning idiom's load, through a real uniform-buffer window: the program declares a
/// 64-byte buffer at GXM index 0 and a +0x78 entry placing its pointer in DATA slot 0 (`sa[20]`),
/// and the SHIPPED resolver turns that into the window the harness fills. Two loads:
///
/// * a 4-element BURST at immediate offset 2 - words 2..6 into `r[0..4]`;
/// * one element at `r[8]` (= 20, BYTES) plus immediate 1 (ELEMENTS, so 4 bytes) - word 6.
///
/// A translation that scaled the register offset, or failed to scale the immediate, reads a
/// different word; one that wrote the burst to the wrong registers moves four lanes.
///
/// >>> AND IT PINS THE SHIPPED WIDTH OF A REGISTER OFFSET: 16 BITS. That rule rests on one
/// program shape and the guest's own memory (`wgsl::emit_mem_load`, arm
/// `VITASLOP_GXP_MEM_OFFSET16`), not on a published field width - so `r[8]` carries a HIGH HALF
/// (`0x0001_0014`) and the two readings part: the low 16 bits land on word 6, the whole register
/// 64 KiB past the window. Flipping the arm must fail here.
fn case_mem_load() -> Case {
    use asm::IntSrc::{Imm, Reg};
    let t = Bank::Temp;
    Case {
        name: "conf_a_memory_load_reads_pointer_plus_element_and_byte_offsets",
        checks: "0xE8's burst length, destination run, immediate-in-elements vs register-in-bytes offsets",
        spec: vertex_spec(vec![
            asm::limm(Dest::new(t, 8), 0x0001_0014).unwrap(),
            asm::ldmem(Dest::new(t, 0), 4, (Bank::SecondaryAttr, 20), Imm(2), Imm(0)).unwrap(),
            asm::ldmem(Dest::new(t, 4), 1, (Bank::SecondaryAttr, 20), Reg(t, 8), Imm(1)).unwrap(),
        ])
        .with_parameters(vec![
            gxpwrite::ParamSpec::attribute("IN.position", 0, 4),
            gxpwrite::ParamSpec::attribute("IN.texcoord", 4, 4),
            gxpwrite::ParamSpec::uniform_buffer("g_Palette", 0, 64),
        ])
        .with_ub_bindings(vec![(0, 0)]),
        intent: |_| {
            // The window this program declares, stated here: 64 bytes, pointer at sa[20].
            let win = vitaslop_gxp_shader::module::MemWindow { buffer_index: 0, bytes: 64, base_sa: 20, base_offset: 0 };
            let words = common::mem_words_for(SEED, std::slice::from_ref(&win));
            let at = |word: u32| common::mem_fetch(common::window_base(0) + 4 * word, std::slice::from_ref(&win), &words);
            let mut out: Vec<Lane> = (0..4).map(|c| Lane { bank: Bank::Temp, lane: c, bits: at(2 + c as u32) }).collect();
            out.push(Lane { bank: Bank::Temp, lane: 4, bits: at(6) });
            out.push(Lane { bank: Bank::Temp, lane: 8, bits: 0x0001_0014 });
            out
        },
        fragment: None,
    }
}

// ---------------------------------------------------------------------------------------
// THE F16 GROUP. A shipped fragment program is 70-90% half precision, and a half-precision
// operand is a DIFFERENT ADDRESSING RULE - channel `c` is half `c & 1` of register
// `index + (c >> 1)` - so an F32 case says nothing about it. Every case here builds its F16
// inputs by PACKING seeded F32 lanes first: reading a seeded f32 lane directly as two halves
// would feed arbitrary bit patterns (NaNs, subnormals the GPU flushes) into the comparison.
// ---------------------------------------------------------------------------------------

/// A value as it survives an F32 -> F16 pack: rounded to binary16 and read back.
fn h(x: f32) -> f32 {
    vitaslop_gxp_shader::wgsl::f16_bits_to_f32(vitaslop_gxp_shader::fold::f32_to_f16_bits(x))
}

/// Pack `pa[base..base+4]` to F16 in `r[dest]`, `r[dest+1]`.
fn pack_f16(dest: u8, base: u8) -> u64 {
    asm::pack(Dest::new(Bank::Temp, dest), asm::PackFmt::F16, Bank::PrimaryAttr, base, asm::PackFmt::F32, [0, 1, 2, 3], [true; 4], false)
        .unwrap()
}

/// The two registers an F16 vec4 occupies, as lanes.
fn f16_vec(base: usize, v: [f32; 4]) -> [Lane; 2] {
    [f16_lane(Bank::Temp, base, v[0], v[1]), f16_lane(Bank::Temp, base + 1, v[2], v[3])]
}

/// **THE F16 ALU: EVERY BINARY OP, THE DOT AND THE MAD, ON PACKED HALVES.**
///
/// `a = r[0..2]` and `b = r[2..4]` are four halves each; every op reads both through the F16
/// addressing rule and writes a packed pair. The DOT is the ALU group's own opcode-7 form - the
/// only one the F16 pipeline has - and it broadcasts to every masked channel.
fn case_f16_alu() -> Case {
    let t = Bank::Temp;
    let op = |o, dest: u8| asm::alu(o, true, Dest::new(t, dest), [true; 4], Src::reg(t, 0), Src::reg(t, 2)).unwrap();
    Case {
        name: "conf_the_f16_alu_reads_and_writes_packed_halves",
        checks: "0x10 group add/min/max/frc/dot and the F16 mad, all through the half-per-channel addressing",
        spec: vertex_spec(vec![
            pack_f16(0, 0),
            pack_f16(2, 4),
            op(Op::Add, 6),
            op(Op::Min, 8),
            op(Op::Max, 10),
            op(Op::Frc, 12),
            op(Op::Dot { components: 4 }, 14),
            asm::mad(true, Dest::new(t, 16), [true; 4], Src::reg(t, 0), Src::reg(t, 2), Src::reg(t, 0)).unwrap(),
        ]),
        intent: |regs| {
            let a: [f32; 4] = std::array::from_fn(|c| h(regs.pa[c]));
            let b: [f32; 4] = std::array::from_fn(|c| h(regs.pa[4 + c]));
            let each = |f: &dyn Fn(f32, f32) -> f32| -> [f32; 4] { std::array::from_fn(|c| f(a[c], b[c])) };
            let mut d = 0.0f32;
            for c in 0..4 {
                d += a[c] * b[c];
            }
            let mut out = Vec::new();
            out.extend(f16_vec(0, a));
            out.extend(f16_vec(2, b));
            out.extend(f16_vec(6, each(&|x, y| x + y)));
            out.extend(f16_vec(8, each(&|x, y| x.min(y))));
            out.extend(f16_vec(10, each(&|x, y| x.max(y))));
            out.extend(f16_vec(12, each(&|x, _| x - x.floor())));
            out.extend(f16_vec(14, [d; 4]));
            out.extend(f16_vec(16, each(&|x, y| x * y + x)));
            out
        },
        fragment: None,
    }
}

/// **THE F16 SCALAR, MOVE AND TEST FORMS.**
///
/// The transcendentals select ONE F16 component (half `comp & 1` of register `comp >> 1`) and
/// broadcast it; the moves swizzle and select halves; the test and the test-mask compare halves.
/// `a`, `b`, `e` are three packed vec4s (`r[0..2]`, `r[2..4]`, `r[4..6]`).
///
/// ```text
///   r[20]     = rcp(a.y)          mask xy
///   r[22..24] = rsq(|a.z|)        mask xyzw
///   r[24..26] = log2(|a.w|)       mask x_z_   (a half in EACH register)
///   r[26..28] = exp2(-a.x)        mask xyzw
///   r[28..30] = a.zxyw            (vec4-standard table entry 10)
///   r[30..32] = a < 0 ? b : e     per channel
///   r[32..34] = a - e >= 0 ? 1:0  the F16 numeric mask
///   p0 = a.y - b.y < 0, then r[40..44] = sa[0..4] where p0 holds
/// ```
fn case_f16_scalar_move_test() -> Case {
    use asm::MoveKind;
    use vitaslop_gxp_shader::ir::{CompareMethod, Predicate, TestAlu, TestCmp, TestReduce};
    let t = Bank::Temp;
    let comp = |o, dest: u8, mask, c, abs, neg| asm::vcomp(o, true, Dest::new(t, dest), mask, t, 0, c, abs, neg).unwrap();
    Case {
        name: "conf_the_f16_scalar_move_and_test_forms_address_halves",
        checks: "F16 vcomp component selection, VMOV/VMOVC at F16, VTST and VTSTMSK on halves",
        spec: vertex_spec(vec![
            pack_f16(0, 0),
            pack_f16(2, 4),
            pack_f16(4, 8),
            comp(Op::Rcp, 20, [true, true, false, false], 1, false, false),
            comp(Op::Rsq, 22, [true; 4], 2, true, false),
            comp(Op::Log, 24, [true, false, true, false], 3, true, false),
            comp(Op::Exp, 26, [true; 4], 0, false, true),
            asm::vmov(MoveKind::Mov, true, Dest::new(t, 28), [true; 4], (t, 0), [2, 0, 1, 3], None, None).unwrap(),
            asm::vmov(MoveKind::Cmov(CompareMethod::LtZero), true, Dest::new(t, 30), [true; 4], (t, 2), [0, 1, 2, 3], Some((t, 4)), Some((t, 0)))
                .unwrap(),
            asm::vtstmsk(TestAlu::Sub, TestCmp::Ge, true, Dest::new(t, 32), (t, 0), false, (t, 4)).unwrap(),
            asm::vtst(TestAlu::Sub, TestCmp::Lt, TestReduce::Channel(1), 0, true, t, 0, t, 2).unwrap(),
            asm::alu_pred(Op::Add, false, Predicate::IfP(0), Dest::new(t, 40), [true; 4], Src::cnst(0).swz([4, 4, 4, 4]), Src::reg(Bank::SecondaryAttr, 0))
                .unwrap(),
        ]),
        intent: |regs| {
            let v = |base: usize| -> [f32; 4] { std::array::from_fn(|c| h(regs.pa[base + c])) };
            let (a, b, e) = (v(0), v(4), v(8));
            let mut out = Vec::new();
            out.extend(f16_vec(0, a));
            out.extend(f16_vec(2, b));
            out.extend(f16_vec(4, e));
            out.push(f16_lane(Bank::Temp, 20, 1.0 / a[1], 1.0 / a[1]));
            out.extend(f16_vec(22, [1.0 / a[2].abs().sqrt(); 4]));
            let l = a[3].abs().log2();
            out.push(f16_lane(Bank::Temp, 24, l, 0.0));
            out.push(f16_lane(Bank::Temp, 25, l, 0.0));
            out.extend(f16_vec(26, [(-a[0]).exp2(); 4]));
            out.extend(f16_vec(28, [a[2], a[0], a[1], a[3]]));
            out.extend(f16_vec(30, std::array::from_fn(|c| if a[c] < 0.0 { b[c] } else { e[c] })));
            out.extend(f16_vec(32, std::array::from_fn(|c| if a[c] - e[c] >= 0.0 { 1.0 } else { 0.0 })));
            if a[1] - b[1] < 0.0 {
                out.extend((0..4).map(|c| f32_lane(Bank::Temp, 40 + c, 0.0 + regs.sa[c])));
            }
            out.retain(|l| l.bits != 0);
            out
        },
        fragment: None,
    }
}

/// **A SAMPLE'S COORDINATE WIDTH AND RESULT WIDTH ARE INDEPENDENT.**
///
/// Three samples of one unit: an F16 coordinate with an F32 result, an F32 coordinate with an F16
/// result, and both F16. A translation that tied the two widths together - reading an F16 UV as
/// one f32, or storing an F16 RGBA as four whole lanes over its neighbours - moves a lane in one
/// of the three. The coordinate is packed from the seeded lanes, so its halves are real values.
fn case_f16_samples() -> Case {
    let t = Bank::Temp;
    let s = |dest: u8, cb: Bank, cr: u8, ch: bool, rh: bool| {
        asm::tex_typed(Dest::new(t, dest), 12, cb, cr, 2, ch, rh, vitaslop_gxp_shader::ir::TexLod::Implicit, None).unwrap()
    };
    Case {
        name: "conf_a_samples_coordinate_and_result_widths_are_independent",
        checks: "0xE0 src0_type (F16 coordinate) and fconv_type (F16 result) read and written independently",
        spec: ProgramSpec::fragment(vec![
            pack_f16(0, 0),
            s(4, t, 0, true, false),
            s(8, Bank::PrimaryAttr, 0, false, true),
            s(12, t, 0, true, true),
        ])
        .with_parameters(vec![gxpwrite::ParamSpec::sampler("tex0", 0)])
        .with_interpolants(vec![gxpwrite::InterpolantSpec::texcoord(0, 4)])
        .with_texture_control(vec![(4, 0)])
        .with_default_uniform_regs(16)
        .with_containers(vec![
            gxpwrite::ContainerSpec { index: 14, base_sa: 0, size_regs: 16 },
            gxpwrite::ContainerSpec { index: 16, base_sa: 16, size_regs: 4 },
            gxpwrite::ContainerSpec { index: 19, base_sa: 20, size_regs: 8 },
        ])
        .with_registers(4, 28, 16),
        intent: |regs| {
            let (u16c, v16c) = (h(regs.pa[0]), h(regs.pa[1]));
            let at = |u: f32, v: f32| case_textures(0, [u, v, 0.0, 0.0], interp::TexLodArg::IMPLICIT).unwrap();
            let t16 = at(u16c, v16c);
            let t32 = at(regs.pa[0], regs.pa[1]);
            let mut out: Vec<Lane> = f16_vec(0, std::array::from_fn(|c| h(regs.pa[c]))).to_vec();
            out.extend((0..4).map(|c| f32_lane(Bank::Temp, 4 + c, t16[c])));
            out.extend(f16_vec(8, t32));
            out.extend(f16_vec(12, t16));
            out
        },
        fragment: None,
    }
}

// ---------------------------------------------------------------------------------------
// THE INTEGER GROUP. Every lane below holds an integer's BIT PATTERN, loaded by LIMM, so the
// intent is plain integer arithmetic on stated values rather than a function of the seed.
// ---------------------------------------------------------------------------------------

/// **THE BITWISE OPS ACT ON THE LANE'S BITS, AND THE 16-BIT FORM ON ITS LOW HALF.**
///
/// All six operations, a register and an immediate second source, and the 16-bit lane - where a
/// shift WRAPS at 16 bits and an arithmetic shift extends bit 15, not bit 31. The values are
/// chosen so each wrong reading moves the answer: `0x8000_0010` makes `>>` and arithmetic `>>`
/// differ, `0xF000` in the low half makes the 16-bit arithmetic shift fill with ones, and
/// `0x5678 << 12` overflows 16 bits.
fn case_bitwise() -> Case {
    use asm::IntSrc::{Imm, Reg};
    use vitaslop_gxp_shader::ir::BitwiseKind::*;
    let t = Bank::Temp;
    let bw = |kind, lane16, dest: u8, src: u8, s2| asm::bitwise(kind, lane16, Dest::new(t, dest), (t, src), s2).unwrap();
    Case {
        name: "conf_bitwise_ops_act_on_the_lane_bits_and_the_16_bit_form_on_its_low_half",
        checks: "VBW opcode/op2 numbering, the 16-bit immediate's three fields, and the 16-bit lane's wrap and sign",
        spec: vertex_spec(vec![
            asm::limm(Dest::new(t, 0), 0x1234_5678).unwrap(),
            asm::limm(Dest::new(t, 1), 0x0F0F_00FF).unwrap(),
            asm::limm(Dest::new(t, 8), 0x8000_0010).unwrap(),
            asm::limm(Dest::new(t, 11), 0x0000_F000).unwrap(),
            bw(And, false, 2, 0, Reg(t, 1)),
            bw(Or, false, 3, 0, Imm(0xA5A5)),
            bw(Xor, false, 4, 0, Reg(t, 1)),
            bw(Shl, false, 5, 0, Imm(4)),
            bw(Shr, false, 6, 8, Imm(4)),
            bw(Asr, false, 7, 8, Imm(4)),
            bw(Shl, true, 9, 0, Imm(12)),
            bw(Asr, true, 10, 11, Imm(4)),
        ]),
        intent: |_| {
            let (a, b, n, h) = (0x1234_5678u32, 0x0F0F_00FFu32, 0x8000_0010u32, 0x0000_F000u32);
            [
                (0, a),
                (1, b),
                (8, n),
                (11, h),
                (2, a & b),
                (3, a | 0xA5A5),
                (4, a ^ b),
                (5, a << 4),
                (6, n >> 4),
                (7, ((n as i32) >> 4) as u32),
                (9, ((a & 0xffff) << 12) & 0xffff),
                (10, ((((h & 0xffff) as u16 as i16) >> 4) as u16) as u32),
            ]
            .into_iter()
            .map(|(lane, bits)| Lane { bank: Bank::Temp, lane, bits })
            .collect()
        },
        fragment: None,
    }
}

/// **AN IMAD32 MULTIPLIES A HALF OF src0, AND ITS SIGN BIT DECIDES HOW THAT HALF WIDENS.**
///
/// `r[0] = 0x0003_FFF9`: its low half is 65529 unsigned and -7 signed, its high half 3. Four
/// multiply-adds read it every way the two selectors allow, one with an inline-literal
/// multiplier and one with `src1_high` - so a translation that read the whole register, the wrong
/// half, or ignored the sign moves one of four lanes.
///
/// Then the group-0x1a PAIR, which builds a full 32x32 product out of two 16x32 steps:
/// `r[9] = r[0] * r[1] + r[2]` in wrapping arithmetic, the low step into `r[9]` and the high step
/// adding its shifted product to it.
fn case_imad() -> Case {
    use asm::IntSrc::{Imm, Reg};
    let t = Bank::Temp;
    Case {
        name: "conf_imad_multiplies_a_half_of_src0_and_a_step_pair_builds_the_whole_product",
        checks: "IMAD32 src0_high/src1_high/signed, its 7-bit literal, and the 0x1a low/high step pair",
        spec: vertex_spec(vec![
            asm::limm(Dest::new(t, 0), 0x0003_FFF9).unwrap(),
            asm::limm(Dest::new(t, 1), 0x0005_03E8).unwrap(),
            asm::limm(Dest::new(t, 2), 0x0000_0011).unwrap(),
            asm::imad(false, false, false, Dest::new(t, 4), (t, 0), Reg(t, 1), Reg(t, 2)).unwrap(),
            asm::imad(true, false, false, Dest::new(t, 5), (t, 0), Reg(t, 1), Reg(t, 2)).unwrap(),
            asm::imad(false, true, false, Dest::new(t, 6), (t, 0), Imm(9), Reg(t, 2)).unwrap(),
            asm::imad(false, false, true, Dest::new(t, 7), (t, 0), Reg(t, 1), Imm(3)).unwrap(),
            asm::imad_step(false, false, Dest::new(t, 9), (t, 0), Reg(t, 1), Reg(t, 2)).unwrap(),
            asm::imad_step(true, false, Dest::new(t, 9), (t, 0), Reg(t, 1), Reg(t, 9)).unwrap(),
        ]),
        intent: |_| {
            let (a, b, d) = (0x0003_FFF9u32, 0x0005_03E8u32, 0x11u32);
            let lo_u = a & 0xffff;
            let lo_s = (a & 0xffff) as u16 as i16 as i32;
            [
                (0, a),
                (1, b),
                (2, d),
                (4, lo_u.wrapping_mul(b).wrapping_add(d)),
                (5, lo_s.wrapping_mul(b as i32).wrapping_add(d as i32) as u32),
                (6, (a >> 16).wrapping_mul(9).wrapping_add(d)),
                (7, lo_u.wrapping_mul(b >> 16).wrapping_add(3)),
                (9, a.wrapping_mul(b).wrapping_add(d)),
            ]
            .into_iter()
            .map(|(lane, bits)| Lane { bank: Bank::Temp, lane, bits })
            .collect()
        },
        fragment: None,
    }
}

/// **AN INDEX-ADD INTO AN ORDINARY REGISTER IS `low16(src) * STRIDE + addend`, THE STRIDE BEING
/// ONE MORE THAN THE PROGRAM'S LARGEST ADDEND.**
///
/// This is how a skinned mesh turns a blend index into a bone's matrix row: three consecutive
/// registers a bone, loaded as addends 0, 1 and 2 - so the stride is three, and bone 5's rows are
/// 15, 16, 17. The stride is not in the word; the decoder infers it from the program, and that
/// inference is exactly what a case must state. The source's HIGH half is set (`0x0007_0005`) so
/// a translation that used the whole register rather than its low 16 bits lands elsewhere.
fn case_index_add() -> Case {
    let t = Bank::Temp;
    Case {
        name: "conf_an_index_add_is_low16_times_the_programs_row_stride_plus_its_addend",
        checks: "group 0x14's ordinary-register form: low-16 source, [25:21] destination, the inferred stride",
        spec: vertex_spec(vec![
            asm::limm(Dest::new(t, 0), 0x0007_0005).unwrap(),
            asm::index_add(Dest::new(t, 4), t, 0, 0).unwrap(),
            asm::index_add(Dest::new(t, 5), t, 0, 1).unwrap(),
            asm::index_add(Dest::new(t, 6), t, 0, 2).unwrap(),
        ]),
        intent: |_| {
            vec![
                Lane { bank: Bank::Temp, lane: 0, bits: 0x0007_0005 },
                Lane { bank: Bank::Temp, lane: 4, bits: 15 },
                Lane { bank: Bank::Temp, lane: 5, bits: 16 },
                Lane { bank: Bank::Temp, lane: 6, bits: 17 },
            ]
        },
        fragment: None,
    }
}

/// **A REPEATING PACK STEPS ITS DESTINATION BY THE SMLSI's DEST BYTE.**
///
/// A football title's face/skin program packs its four blend indices as int16 pairs with ONE
/// `PackToInt` repeated twice, under `SMLSI [9,1,1,1]`. The destination byte (9) sends the
/// second pair nine registers on, into a register the program has finished with; reading the
/// destination off the src0 byte (1) instead lands it one register on - on the program's
/// TANGENT - and the index copies below read the texture coordinate as bone indices. Half of
/// every skinned face's palette loads then landed outside the palette and the faces smeared
/// into vertical streaks.
///
/// Here the two readings name different registers (`r[9]` against `r[1]`), and the intent
/// states the whole written set, so only one of them passes.
fn case_repeating_pack_steps_by_the_dest_byte() -> Case {
    use asm::PackFmt;
    use vitaslop_gxp_shader::usse::decode::SmlsiSlot::Increment;
    let t = Bank::Temp;
    let pack = asm::pack(Dest::new(t, 0), PackFmt::S16, Bank::PrimaryAttr, 4, PackFmt::F32, [0, 1, 0, 0], [true, true, false, false], false)
        .unwrap();
    Case {
        name: "conf_a_repeating_pack_steps_its_destination_by_the_smlsi_dest_byte",
        checks: "VPCK repeat under SMLSI [9,1,1,1]: destination r0 then r9 (dest byte), source pa4 then pa6",
        spec: vertex_spec(vec![
            asm::smlsi([Increment(9), Increment(1), Increment(1), Increment(1)]).unwrap(),
            asm::with_repeat_count(pack, 1).unwrap(),
        ]),
        intent: |regs| {
            let p = &regs.pa;
            let h = |v: f32| (v.trunc() as i32 as u16) as u32;
            vec![
                Lane { bank: Bank::Temp, lane: 0, bits: h(p[4]) | (h(p[5]) << 16) },
                Lane { bank: Bank::Temp, lane: 9, bits: h(p[6]) | (h(p[7]) << 16) },
            ]
            .into_iter()
            .filter(|l| l.bits != 0)
            .collect()
        },
        fragment: None,
    }
}

/// **A REPEATING INDEX-ADD WRITES ONE ROW PER ITERATION, WALKING THE SMLSI's OFFSET TABLE.**
///
/// Bits 46:44 of a group-0x14 word are its repeat count - bit 45 was known to vary without
/// touching the first iteration's value, which is exactly what a repeat looks like. The same face
/// program runs one index-add three times under an SMLSI whose dest and src0 slots are in OFFSET
/// mode `0x34` = (0,1,3): it writes `pa[18], pa[19], pa[21]` from `pa[11], pa[12], pa[14]`, the
/// only producers the program has for three of its bone-row offsets. Read as running once, two
/// of those registers kept the raw float bits of a vertex colour and a blend index.
///
/// The stride is the program's (one more than its largest addend), so two plain index-adds with
/// addends 1 and 2 are there to make it three, and the sources' HIGH halves are set so a read of
/// the whole register lands elsewhere.
fn case_repeating_index_add_walks_the_offset_table() -> Case {
    use vitaslop_gxp_shader::usse::decode::SmlsiSlot::{Increment, Swizzle};
    let t = Bank::Temp;
    let rep = asm::index_add(Dest::new(t, 8), t, 0, 0).unwrap();
    Case {
        name: "conf_a_repeating_index_add_writes_one_row_per_iteration",
        checks: "group 0x14 repeat (bits 46:44 = 2) under SMLSI offset mode (0,1,3) on dest and src0",
        spec: vertex_spec(vec![
            asm::limm(Dest::new(t, 0), 0x0007_0005).unwrap(),
            asm::limm(Dest::new(t, 1), 0x0009_0002).unwrap(),
            asm::limm(Dest::new(t, 2), 0x0006_0006).unwrap(),
            asm::limm(Dest::new(t, 3), 0x0004_0008).unwrap(),
            asm::index_add(Dest::new(t, 20), t, 0, 1).unwrap(),
            asm::index_add(Dest::new(t, 21), t, 0, 2).unwrap(),
            asm::smlsi([Swizzle(0x34), Swizzle(0x34), Increment(0), Increment(0)]).unwrap(),
            asm::with_repeat_count(rep, 2).unwrap(),
        ]),
        intent: |_| {
            vec![
                Lane { bank: Bank::Temp, lane: 0, bits: 0x0007_0005 },
                Lane { bank: Bank::Temp, lane: 1, bits: 0x0009_0002 },
                Lane { bank: Bank::Temp, lane: 2, bits: 0x0006_0006 },
                Lane { bank: Bank::Temp, lane: 3, bits: 0x0004_0008 },
                Lane { bank: Bank::Temp, lane: 20, bits: 16 },
                Lane { bank: Bank::Temp, lane: 21, bits: 17 },
                // Iterations 0, 1, 2 at offsets 0, 1, 3: r8 <- r0, r9 <- r1, r11 <- r3.
                Lane { bank: Bank::Temp, lane: 8, bits: 15 },
                Lane { bank: Bank::Temp, lane: 9, bits: 6 },
                Lane { bank: Bank::Temp, lane: 11, bits: 24 },
            ]
        },
        fragment: None,
    }
}

/// **A TEST-MASK WRITES 1.0 OR 0.0 PER CHANNEL - FOUR CHANNELS FOR A FLOAT FAMILY.**
///
/// The shadow-filter idiom: compare four samples against a reference, then average the mask. Two
/// instructions, a subtract `> 0` and an add `>= 0` with the source-1 NEGATE (`test_flag_2`), so
/// the relation table, the negate bit and the direct destination each move a lane if misread.
fn case_test_mask() -> Case {
    use vitaslop_gxp_shader::ir::{TestAlu, TestCmp};
    let t = Bank::Temp;
    Case {
        name: "conf_a_test_mask_writes_one_or_zero_per_channel",
        checks: "VTSTMSK's numeric float form: per-channel relation, the src1 negate at bit 50, the direct destination",
        spec: vertex_spec(vec![
            asm::vtstmsk(TestAlu::Sub, TestCmp::Gt, false, Dest::new(t, 4), (Bank::PrimaryAttr, 0), false, (Bank::SecondaryAttr, 0))
                .unwrap(),
            asm::vtstmsk(TestAlu::Add, TestCmp::Ge, false, Dest::new(t, 9), (Bank::PrimaryAttr, 0), true, (Bank::SecondaryAttr, 8))
                .unwrap(),
        ]),
        intent: |regs| {
            let (p, s) = (&regs.pa, &regs.sa);
            let one = |b: bool| if b { 1.0 } else { 0.0 };
            let mut out = Vec::new();
            for c in 0..4 {
                out.push(f32_lane(Bank::Temp, 4 + c, one(p[c] - s[c] > 0.0)));
                out.push(f32_lane(Bank::Temp, 9 + c, one(-p[c] + s[8 + c] >= 0.0)));
            }
            out.retain(|l| l.bits != 0);
            out
        },
        fragment: None,
    }
}

/// **A FLOAT-TO-INT PACK TRUNCATES TOWARD ZERO AND STORES THE INTEGER'S BITS, PACKED BY WIDTH.**
///
/// `scale` clear is the plain cast, not a normalise. At 16 bits two channels share a register,
/// half `c & 1` of `dest + (c >> 1)`; at 8 bits all four are the BYTES of one. The seeded inputs
/// are in `[-4, 4)`, so both signs are present and truncation (toward zero) differs from `floor`
/// on every negative non-integer.
fn case_pack_to_int() -> Case {
    use asm::PackFmt;
    let t = Bank::Temp;
    Case {
        name: "conf_a_float_to_int_pack_truncates_and_packs_by_width",
        checks: "VPCK F32->S16 and F32->S8 with scale clear: truncation, two's complement, half/byte placement",
        spec: vertex_spec(vec![
            asm::pack(Dest::new(t, 0), PackFmt::S16, Bank::PrimaryAttr, 4, PackFmt::F32, [0, 1, 2, 3], [true; 4], false)
                .unwrap(),
            asm::pack(Dest::new(t, 4), PackFmt::S8, Bank::PrimaryAttr, 12, PackFmt::F32, [3, 2, 1, 0], [true; 4], false)
                .unwrap(),
        ]),
        intent: |regs| {
            let p = &regs.pa;
            let i = |v: f32| v.trunc() as i32;
            let h = |v: f32| (i(v) as u16) as u32;
            let b = |v: f32| (i(v) as u8) as u32;
            vec![
                Lane { bank: Bank::Temp, lane: 0, bits: h(p[4]) | (h(p[5]) << 16) },
                Lane { bank: Bank::Temp, lane: 1, bits: h(p[6]) | (h(p[7]) << 16) },
                Lane {
                    bank: Bank::Temp,
                    lane: 4,
                    bits: b(p[15]) | (b(p[14]) << 8) | (b(p[13]) << 16) | (b(p[12]) << 24),
                },
            ]
            .into_iter()
            .filter(|l| l.bits != 0)
            .collect()
        },
        fragment: None,
    }
}

// ---------------------------------------------------------------------------------------
// THE MADDEN GROUP. Each of these is a shape a football title's stadium actually draws, and
// each was a defect that took a session of title-level debugging to name. They are here so
// the next regression costs five milliseconds instead.
// ---------------------------------------------------------------------------------------

/// **AN INDEXED UNIFORM READ IS `sa[i0 + offset]`, AT PAIR SCALE ONE.**
///
/// This is the crowd-sprite idiom end to end: an index load computes a row, and the instruction
/// after it reads a uniform ARRAY at that row. The scale between the index register's value and
/// the register it selects is the number that was wrong - it defaulted to 2, fitted to one
/// program under a source reading that is now refuted, and under that scale a title's six-vertex
/// fan sprites read past every register anything writes, came back zero, and collapsed to a
/// point: 23 draws and 59,202 indices that rasterised NOTHING every frame.
///
/// The case is deliberately arithmetic rather than pictorial. `r[0]` is zero (nothing has
/// written it), so the index is exactly the ADDEND, and the read lands on a register the intent
/// can name outright. At scale 2 the same program reads a different set of lanes, so the two
/// readings cannot both pass.
fn case_indexed_uniform_read() -> Case {
    // i0 = int(r[0]) + 5. Nothing has written r[0], so its integer value is 0 and the index is
    // exactly 5 - deterministic, and independent of the seed.
    let load = asm::load_index(Bank::Temp, 0, 5).unwrap();
    let read = asm::mov(
        false,
        Dest::new(Bank::Output, 0),
        [true; 4],
        Src::indexed(Bank::SecondaryAttr, 10, 0),
    )
    .unwrap();
    Case {
        name: "conf_an_indexed_uniform_read_selects_the_row_its_index_names",
        checks: "sa[i0 + offset] reads register i0 + offset at pair scale ONE, not i0 * 2 + offset",
        spec: vertex_spec(vec![load, read]),
        intent: |regs| {
            // i0 = 5, offset = 10, so channel c reads sa[15 + c]. Under a pair scale of 2 the
            // same program would read sa[20 + c] - four different lanes.
            (0..4).map(|c| f32_lane(Bank::Output, c, regs.sa[15 + c])).collect()
        },
        fragment: None,
    }
}

/// **THE INDEX LOAD'S ADDEND SELECTS THE ROW, AND CONSECUTIVE ADDENDS SELECT CONSECUTIVE
/// ROWS.**
///
/// The addend's field position was established by arithmetic closure against one title's
/// parameter table and then corroborated against a second - real evidence, but evidence with
/// no test. Skinning is where it bites: three consecutive registers per bone against addends
/// that are exactly 0, 1 and 2, so a hidden multiplier anywhere in the encoding would produce
/// rows 0, 3, 6 instead and every bone matrix would be assembled out of three unrelated rows.
///
/// Two loads with addends one apart, each feeding its own read, is that property stated
/// directly: the two results must be one register apart, no more and no less.
fn case_index_addend_steps_by_one() -> Case {
    let row0 = asm::load_index(Bank::Temp, 0, 3).unwrap();
    let read0 = asm::mov(
        false,
        Dest::new(Bank::Output, 0),
        [true, true, false, false],
        Src::indexed(Bank::SecondaryAttr, 0, 0),
    )
    .unwrap();
    let row1 = asm::load_index(Bank::Temp, 0, 4).unwrap();
    let read1 = asm::mov(
        false,
        Dest::new(Bank::Output, 4),
        [true, true, false, false],
        Src::indexed(Bank::SecondaryAttr, 0, 0),
    )
    .unwrap();
    Case {
        name: "conf_consecutive_index_addends_select_consecutive_rows",
        checks: "addends one apart select registers one apart - no hidden multiplier in the encoding",
        spec: vertex_spec(vec![row0, read0, row1, read1]),
        intent: |regs| {
            vec![
                f32_lane(Bank::Output, 0, regs.sa[3]),
                f32_lane(Bank::Output, 1, regs.sa[4]),
                f32_lane(Bank::Output, 4, regs.sa[4]),
                f32_lane(Bank::Output, 5, regs.sa[5]),
            ]
        },
        fragment: None,
    }
}

/// **A SIX-CORNER SPRITE EXPANDS TO SIX DIFFERENT POSITIONS.**
///
/// The stadium crowd is drawn as six-vertex fan sprites: each vertex carries a corner number,
/// a corner table sits in the uniform block at a known stride, and the vertex program looks its
/// own corner up and offsets the sprite centre by it. When the lookup is wrong, every vertex
/// reads the SAME table row, all six land on one point, and the sprite has no area at all -
/// which is exactly what was measured, and exactly what a per-draw coverage query saw as a draw
/// that rasterised nothing.
///
/// The case runs the lookup at two different corners in one program and requires the two
/// results to DIFFER - the property a collapsed sprite violates, stated without needing a
/// picture, a frame or a stadium.
fn case_sprite_corner_expansion() -> Case {
    let mut code = Vec::new();
    // Corner 0 and corner 2 of a table at stride 2, as two index loads with the addends that
    // `2 * corner + base` produces: base 8, so corners 0 and 2 are addends 8 and 12.
    for (corner_addend, out_reg) in [(8u8, 0u8), (12u8, 4u8)] {
        code.push(asm::load_index(Bank::Temp, 0, corner_addend).unwrap());
        code.push(
            asm::mov(
                false,
                Dest::new(Bank::Output, out_reg),
                [true, true, false, false],
                Src::indexed(Bank::SecondaryAttr, 0, 0),
            )
            .unwrap(),
        );
    }
    Case {
        name: "conf_a_six_corner_sprite_reads_a_different_row_per_corner",
        checks: "two corners of one sprite read DIFFERENT table rows - a collapsed sprite is a lookup that does not move",
        spec: vertex_spec(code),
        intent: |regs| {
            vec![
                f32_lane(Bank::Output, 0, regs.sa[8]),
                f32_lane(Bank::Output, 1, regs.sa[9]),
                f32_lane(Bank::Output, 4, regs.sa[12]),
                f32_lane(Bank::Output, 5, regs.sa[13]),
            ]
        },
        fragment: None,
    }
}

/// A FRAGMENT program, so the fragment half of the link is exercised too: its inputs arrive as
/// interpolants rather than vertex iterators, and its PA register map comes from the varyings
/// block rather than the parameter table.
fn case_fragment() -> Case {
    Case {
        name: "conf_fragment_reads_its_interpolants",
        checks: "a fragment program's PA registers are its declared interpolants, in order",
        spec: ProgramSpec::fragment(vec![asm::alu(
            Op::Mul,
            false,
            Dest::new(Bank::Output, 0),
            [true; 4],
            Src::reg(Bank::PrimaryAttr, 0),
            Src::reg(Bank::PrimaryAttr, 4),
        )
        .unwrap()])
        .with_interpolants(vec![
            gxpwrite::InterpolantSpec::texcoord(0, 4),
            gxpwrite::InterpolantSpec::texcoord(1, 4),
        ])
        .with_registers(8, 0, 8),
        intent: |regs| {
            (0..4).map(|c| f32_lane(Bank::Output, c, regs.pa[c] * regs.pa[4 + c])).collect()
        },
        fragment: None,
    }
}

/// The standard vertex container the arithmetic cases share: a four-lane position attribute, a
/// four-lane second attribute, a clip position and one texcoord output.
fn vertex_spec(code: Vec<u64>) -> ProgramSpec {
    ProgramSpec::vertex(code)
        .with_parameters(vec![
            gxpwrite::ParamSpec::attribute("IN.position", 0, 4),
            gxpwrite::ParamSpec::attribute("IN.texcoord", 4, 4),
        ])
        .with_default_uniform_regs(16)
        .with_containers(vec![
            gxpwrite::ContainerSpec { index: 14, base_sa: 0, size_regs: 16 },
            gxpwrite::ContainerSpec { index: 16, base_sa: 16, size_regs: 4 },
            gxpwrite::ContainerSpec { index: 19, base_sa: 20, size_regs: 4 },
        ])
        .with_registers(8, 24, 8)
        .with_outputs(VertexOutputs { color0: false, texcoords: vec![(0, 4)] })
}

// ---------------------------------------------------------------------------------------
// THE FRAGMENT-PIPELINE GROUP. Two effects that leave the register file untouched, and which
// nothing therefore checks unless a case says so out loud.
//
// >>> THIS IS THE ONE PART OF THE RENDER RIG NO CAPTURED PROGRAM REACHES. 29 corpus blobs
// carry a `kill` and every one of them is the same alpha-test idiom, of which 11 actually
// discard under the seeded inputs and 18 carry the instruction behind a predicate that does
// not fire. NOT ONE corpus program writes a fragment depth at all - so `Op::DepthF`'s
// reference arm and the rig's depth capture were pinned by a Rust test and checked on the GPU
// by nothing.
//
// An authored case closes both, and it is worth more than a captured one here for the usual
// reason: a discard that both sides drop is a discard both sides AGREE about, and only a
// stated intent can tell that apart from a discard that happened.
// ---------------------------------------------------------------------------------------

/// The fragment container these two share: two four-lane interpolants and no sampler.
fn fragment_pipeline_spec(code: Vec<u64>) -> ProgramSpec {
    ProgramSpec::fragment(code)
        .with_interpolants(vec![
            gxpwrite::InterpolantSpec::texcoord(0, 4),
            gxpwrite::InterpolantSpec::texcoord(1, 4),
        ])
        .with_registers(8, 0, 8)
}

/// **AN UNPREDICATED `kill` DISCARDS, AND EVERYTHING BEFORE IT STILL HAPPENED.**
///
/// The instruction ends the fragment, and the two halves of that are separately wrong-able: a
/// translation that dropped the discard leaves every register agreeing and paints a pixel the
/// hardware throws away, while one that ended the program too early loses the write above it.
/// So the case writes a register FIRST and then kills, and states both.
///
/// >>> AND THE RIG MODELS THE KILL RATHER THAN EXECUTING IT, which is the only honest choice
/// and is why the flag exists at all. WGSL's `discard` demotes the invocation to a helper, and
/// a helper's stores DO NOT LAND - so a case whose program really discarded would read its
/// output buffer back untouched and compare as "the GPU computed zero" on every lane. The
/// wrapper rewrites it to "record the kill, write the register file, leave", and the reference
/// ends its walk at the same instruction.
fn case_kill_discards_after_writing() -> Case {
    let write = asm::alu(
        Op::Mul,
        false,
        Dest::new(Bank::Output, 0),
        [true; 4],
        Src::reg(Bank::PrimaryAttr, 0),
        Src::reg(Bank::PrimaryAttr, 4),
    )
    .unwrap();
    let kill = asm::kill(vitaslop_gxp_shader::ir::Predicate::Always).unwrap();
    // A write AFTER the kill, which must NOT land: the fragment ended.
    let after =
        asm::mov(false, Dest::new(Bank::Output, 4), [true; 4], Src::reg(Bank::PrimaryAttr, 0))
            .unwrap();
    Case {
        name: "conf_kill_discards_and_ends_the_program",
        checks: "an unpredicated kill discards the fragment, keeps the writes above it, and ends it",
        spec: fragment_pipeline_spec(vec![write, kill, after]),
        intent: |regs| (0..4).map(|c| f32_lane(Bank::Output, c, regs.pa[c] * regs.pa[4 + c])).collect(),
        fragment: Some(|_| FragmentIntent { killed: true, depth: None }),
    }
}

/// **A `kill` WHOSE PREDICATE DOES NOT FIRE IS NOT A KILL**, and the program runs on past it.
///
/// 18 of the corpus's 29 `kill` programs are exactly this - the instruction is there and the
/// predicate does not hold - so a translation that discarded unconditionally would leave those
/// 18 agreeing on every register while erasing the surfaces they draw. Nothing could see that
/// but a case that states which way the predicate went.
///
/// The predicate register is never written, so `p[1]` is false and `IfNotP(1)` - the encoding a
/// real alpha test uses - DOES fire. `IfP(0)` over an unwritten `p[0]` is the one that does not,
/// and it is encodable: it is the fourth value of KILL's own two-bit field.
fn case_predicated_kill_that_does_not_fire() -> Case {
    let kill = asm::kill(vitaslop_gxp_shader::ir::Predicate::IfP(0)).unwrap();
    let after = asm::alu(
        Op::Add,
        false,
        Dest::new(Bank::Output, 0),
        [true; 4],
        Src::reg(Bank::PrimaryAttr, 0),
        Src::reg(Bank::PrimaryAttr, 4),
    )
    .unwrap();
    Case {
        name: "conf_a_predicated_kill_that_does_not_fire_runs_on",
        checks: "a kill whose predicate is clear discards nothing and the program continues",
        spec: fragment_pipeline_spec(vec![kill, after]),
        intent: |regs| (0..4).map(|c| f32_lane(Bank::Output, c, regs.pa[c] + regs.pa[4 + c])).collect(),
        fragment: Some(|_| FragmentIntent { killed: false, depth: None }),
    }
}

/// **A WRITTEN FRAGMENT DEPTH COMES FROM THE REGISTER THE INSTRUCTION NAMES**, and from its
/// channel 0.
///
/// NOT ONE CORPUS PROGRAM WRITES A DEPTH, so before this case the whole path - decode, the
/// reference's `DepthF` arm, the emitter's `gxp_frag_depth` assignment, the rig's capture of it
/// - was checked by a Rust unit test on one shipped word and by nothing on a GPU.
///
/// The depth is computed rather than copied, so a translation that took the wrong register or
/// the wrong channel produces a different number instead of the same one by luck: the value is
/// `pa[0] * pa[4]`, and the two seeded lanes differ.
///
/// The rig's declared forward map is a CLAMP to `[0, 1]`, applied on both sides - which of the
/// four real maps is in force is a property of the DRAW - so the intent applies it too, with
/// `min`/`max` rather than `f32::clamp` because WGSL's `clamp` is `min(max(e,lo),hi)` and the
/// two differ on a NaN.
fn case_depthf_replaces_the_fragment_depth() -> Case {
    // >>> THE DEPTH MUST LAND STRICTLY INSIDE `[0, 1)`, OR THE CASE CHECKS THE CLAMP AND NOT
    // >>> THE VALUE. Written first as `pa[0] * pa[4]`, the product came to 1.0 after the rig's
    // clamp - and a saturated expectation agrees with EVERY value at or above one, which is
    // most of the ways a translation could get this wrong. `fract` is the range reduction that
    // fixes it: `x - floor(x)` is in `[0, 1)` for any finite input, so the clamp becomes a
    // no-op and what the two sides compare is the number the program computed.
    //
    // It also writes a register the instruction does NOT name - `r[1]`, from the same `frc`'s
    // second channel - so a translation that took the depth from the wrong register of the
    // group produces a different number rather than the same one by luck.
    let compute = asm::alu(
        Op::Frc,
        false,
        Dest::new(Bank::Temp, 0),
        [true, true, false, false],
        Src::reg(Bank::PrimaryAttr, 0),
        Src::reg(Bank::PrimaryAttr, 0),
    )
    .unwrap();
    let depth = asm::depthf(Bank::Temp, 0).unwrap();
    Case {
        name: "conf_depthf_writes_the_register_it_names",
        checks: "a written fragment depth is channel 0 of the register DEPTHF names, not its neighbour",
        spec: fragment_pipeline_spec(vec![compute, depth]),
        intent: |regs| {
            (0..2)
                .map(|c| {
                    let x = regs.pa[c];
                    f32_lane(Bank::Temp, c, x - x.floor())
                })
                .collect()
        },
        fragment: Some(|regs| {
            let x = regs.pa[0];
            FragmentIntent { killed: false, depth: Some(x - x.floor()) }
        }),
    }
}

// ---------------------------------------------------------------------------------------
// THE DOT GROUP. A vertex program's FIRST arithmetic, and until now unwritable.
//
// >>> THE REPEAT COUNT ON THIS GROUP IS THE DIFFERENCE BETWEEN A TITLE'S GEOMETRY AND A BLACK
// FRAME, and the reading that settled it is a corpus census plus a destination-closure
// argument - a claim about what four shipped words must have meant, with no case anywhere
// stating what ONE word means before it runs. The census cannot supply one: it can only report
// what some compiler emitted, never what the answer should be.
//
// These two say it. They are worth more than the 1,884 corpus DOTs for the usual reason - where
// the emitter and the interpreter are wrong the same way, they agree, and only a stated intent
// tells that apart from being right.
// ---------------------------------------------------------------------------------------

/// **A DOT REDUCES ITS CHANNELS TO A SCALAR, and its second operand is an INTERNAL register.**
///
/// The plain form first, because the repeating one below is meaningless if this is wrong. The
/// internal register is written by the instruction above it - the only way a DOT's op2 can come
/// to hold anything, and itself a path the emitter can get wrong, since `i0` is four lanes of a
/// bank that no parameter fills and no case had ever written from outside the corpus.
///
/// Four channels and not three, with the three-channel form's own distinct swizzle table left to
/// the assembler sweep: what a GPU can add that a Rust test cannot is the REDUCTION ORDER, and
/// both forms sum the same way.
fn case_dot_reduces_four_channels() -> Case {
    let define_i0 = asm::alu(
        Op::Add,
        false,
        Dest::new(Bank::Internal, 0),
        [true; 4],
        Src::reg(Bank::PrimaryAttr, 0),
        Src::reg(Bank::PrimaryAttr, 4),
    )
    .unwrap();
    let dot = asm::dot(
        4,
        Dest::new(Bank::Output, 0),
        [true, false, false, false],
        Src::reg(Bank::PrimaryAttr, 4),
        Src::reg(Bank::Internal, 0),
        0,
    )
    .unwrap();
    Case {
        name: "conf_dot_reduces_four_channels_to_a_scalar",
        checks: "a 4-channel DOT sums four products into ONE destination channel, reading its \
                 second operand from an internal register written above it",
        spec: vertex_spec(vec![define_i0, dot]),
        intent: |regs| {
            // i0 = pa[0..3] + pa[4..7], then o[0] = dot(pa[4..7], i0). The internal register is
            // named too - the harness requires every lane the program moves, which is what stops
            // a case from quietly tolerating a write it did not intend.
            let acc: f32 = (0..4).map(|c| regs.pa[4 + c] * (regs.pa[c] + regs.pa[4 + c])).sum();
            (0..4)
                .map(|c| f32_lane(Bank::Internal, c, regs.pa[c] + regs.pa[4 + c]))
                .chain(std::iter::once(f32_lane(Bank::Output, 0, acc)))
                .collect()
        },
        fragment: None,
    }
}

/// **A REPEATING DOT IS A MATRIX TRANSFORM: the destination walks one CHANNEL per iteration
/// while the vector source advances a WHOLE VECTOR.**
///
/// >>> THIS IS THE ONE THAT RENDERS A TITLE OR DOES NOT. A retail title's world vertex programs
/// contain exactly one DOT each, with three extra iterations, sourced from a
/// `WorldViewProjection` uniform declared `F32[4]` x 4 and writing the four lanes of clip
/// position that nothing else in the program writes. Read as running ONCE, the program emits a
/// single scalar where a whole clip position belongs and the title renders BLACK.
///
/// The two strides are the claim, and they are separately wrong-able in ways that both produce
/// four written lanes: a destination that stepped a whole vector writes four registers instead
/// of four channels, and a source that stepped one lane transposes the matrix. So the sixteen
/// uniform values are distinct by construction (they are `lane_value`'s, which never repeat),
/// and a transposed read cannot coincide with the right answer.
fn case_repeating_dot_is_a_matrix_transform() -> Case {
    // `min(x, x)` is x, and it is the shortest way to get four known lanes into an internal
    // register - a DOT's op2 can come from nowhere else, and no parameter fills that bank.
    let define_i0 = asm::alu(
        Op::Min,
        false,
        Dest::new(Bank::Internal, 0),
        [true; 4],
        Src::reg(Bank::PrimaryAttr, 0),
        Src::reg(Bank::PrimaryAttr, 0),
    )
    .unwrap();
    // Four executions: destination o[0], o[1], o[2], o[3]; source sa[0..3], sa[4..7],
    // sa[8..11], sa[12..15]; the internal operand standing still.
    let transform = asm::dot(
        4,
        Dest::new(Bank::Output, 0),
        [true, false, false, false],
        Src::reg(Bank::SecondaryAttr, 0),
        Src::reg(Bank::Internal, 0),
        3,
    )
    .unwrap();
    Case {
        name: "conf_repeating_dot_walks_lanes_and_vectors",
        checks: "a DOT repeating three extra times steps its DESTINATION one channel and its \
                 vector SOURCE a whole vector per iteration - a 4x4 transform, not one scalar",
        spec: vertex_spec(vec![define_i0, transform]),
        intent: |regs| {
            // i0 = min(pa[0..3], pa[0..3]) = pa[0..3] - and it must stand still across all four
            // iterations, which is the third of the three strides this case states.
            (0..4)
                .map(|c| f32_lane(Bank::Internal, c, regs.pa[c]))
                .chain((0..4).map(|k| {
                    let acc: f32 = (0..4).map(|c| regs.sa[4 * k + c] * regs.pa[c]).sum();
                    f32_lane(Bank::Output, k, acc)
                }))
                .collect()
        },
        fragment: None,
    }
}

/// Every case in the suite.
fn all_cases() -> Vec<Case> {
    vec![
        case_move(),
        case_dot_reduces_four_channels(),
        case_repeating_dot_is_a_matrix_transform(),
        case_per_channel_constants(),
        case_fract(),
        case_mad_write_mask(),
        case_f16_packing(),
        case_f16_saturation(),
        case_partial_write(),
        case_modifiers(),
        case_swizzle(),
        case_pack_round_trip(),
        case_unorm8_round_trip(),
        case_int_half_repack(),
        case_signed_half_widen(),
        case_partial_mask_half_repack(),
        case_texture_sample(),
        case_tex_bias(),
        case_tex_level(),
        case_tex_gradient(),
        case_test_and_predicated_writes(),
        case_transcendentals(),
        case_moves(),
        case_limm(),
        case_forward_branch(),
        case_loop(),
        case_derivatives(),
        case_sop2(),
        case_gather(),
        case_mem_load(),
        case_f16_alu(),
        case_f16_scalar_move_test(),
        case_f16_samples(),
        case_bitwise(),
        case_imad(),
        case_index_add(),
        case_repeating_pack_steps_by_the_dest_byte(),
        case_repeating_index_add_walks_the_offset_table(),
        case_test_mask(),
        case_pack_to_int(),
        case_indexed_uniform_read(),
        case_index_addend_steps_by_one(),
        case_sprite_corner_expansion(),
        case_fragment(),
        case_kill_discards_after_writing(),
        case_predicated_kill_that_does_not_fire(),
        case_depthf_replaces_the_fragment_depth(),
    ]
}

/// What running one case produced: the shader, the emitted body and the interpreter's final
/// register file.
struct Ran {
    shader: vitaslop_gxp_shader::ir::Shader,
    body: String,
    regs: RegFile,
    kind: ProgramKind,
    /// The guest-memory windows the program's memory loads read, and the bytes behind them.
    windows: Vec<vitaslop_gxp_shader::module::MemWindow>,
    mem_words: Vec<u32>,
}

/// Assemble, parse, recompile and interpret one case, through the SAME entry points the
/// renderer calls on a captured blob. Nothing test-only is in this path: a case that passes
/// here passed through the shipped container parser, the shipped decoder and the shipped
/// emitter.
fn run(case: &Case) -> Ran {
    let bytes = gxpwrite::write(&case.spec);
    let program = Program::parse(&bytes)
        .unwrap_or_else(|e| panic!("{}: the assembled blob does not parse: {e:?}", case.name));
    assert_eq!(program.kind, case.spec.kind, "{}: the blob's kind", case.name);

    let (shader, body) = match case.spec.kind {
        ProgramKind::Vertex => recompile_vertex(&bytes).map(|r| (r.shader, r.wgsl_body)),
        ProgramKind::Fragment => recompile_fragment(&bytes).map(|r| (r.shader, r.wgsl_body)),
    }
    .unwrap_or_else(|e| panic!("{}: the emitter refused the assembled program: {e:?}", case.name));

    let mut regs = seeded_inputs();
    // The value BOTH wrappers pin `gxp_front_facing` to. A `kill` carries its own predicate and
    // a facing test is an ordinary case, so the reference needs the same pinned value the module
    // gets - and a caller that has none still gets the refusal rather than a default.
    regs.facing = Some(true);
    // A RENDER case runs under the rig's screen ramp, and the reference is told which lane.
    if case.fragment.is_some() {
        regs.ramp = Some(CASE_RAMP_LANE);
    }
    // Each rig's reference fetches what that rig's module samples: the compute rig's stand-in
    // function, or the render rig's REAL texels (nearest, one mip) and its gather footprint.
    let textures = case_textures;
    let render_sample = |unit: u8, coord: [f32; 4], _lod: interp::TexLodArg| {
        Some(vitaslop_gxp_shader::wgsl::case_render_sample(unit, coord))
    };
    let render_gather =
        |unit: u8, uv: [f32; 2]| Some(vitaslop_gxp_shader::wgsl::case_render_gather(unit, uv));
    let env = if case.fragment.is_some() {
        interp::TexEnv { sample: &render_sample, gather: Some(&render_gather) }
    } else {
        interp::TexEnv::sampling(&textures)
    };
    // GUEST MEMORY, resolved by the SHIPPED resolver from the program's own +0x78 table and
    // parameter table - the same windows the renderer would bind. Each window's pointer is placed
    // in its SA register exactly as the module's prologue places it, and the bytes are the
    // harness's one avalanche, read by the reference and baked into the case for the runner.
    let windows = match case.spec.kind {
        ProgramKind::Vertex => vitaslop_gxp_shader::mem_windows_for_vertex_blob(&bytes),
        ProgramKind::Fragment => vitaslop_gxp_shader::mem_windows_for_fragment_blob(&bytes),
    };
    for (i, win) in windows.iter().enumerate() {
        regs.sa[win.base_sa as usize] = f32::from_bits(common::window_base(i));
    }
    let mem_words = common::mem_words_for(SEED, &windows);
    let memory = |addr: u32| common::mem_fetch(addr, &windows, &mem_words);
    let mem: Option<&dyn Fn(u32) -> u32> = if windows.is_empty() { None } else { Some(&memory) };
    interp::run_traced_env(&shader, &mut regs, &env, mem, &mut |_, _| {})
        .unwrap_or_else(|e| panic!("{}: the reference refused the program: {e:?}", case.name));

    Ran { shader, body, regs, kind: case.spec.kind, windows, mem_words }
}

/// The lanes of a bank the case's intent does NOT mention must be zero. Stated as its own
/// function because it is the half of the assertion that catches an instruction writing
/// somewhere it should not - the shape of the write-mask and register-packing defects.
fn lanes_of(bank: Bank, regs: &RegFile) -> &[f32] {
    match bank {
        Bank::Output => &regs.o,
        Bank::Temp => &regs.r,
        Bank::Internal => &regs.i,
        // `pa` is an OUTPUT bank too - a fragment colour can land there
        // (`ColorOutput::NonNativePa`) and the GPU runner compares it - so an intent may name
        // it. It starts SEEDED rather than zero, which is why `unmoved_baseline` exists.
        Bank::PrimaryAttr => &regs.pa,
        other => panic!("a conformance intent cannot name the {other:?} bank"),
    }
}

/// What a bank holds where the program wrote nothing: zero for the destination banks, and the
/// SEEDED input for `pa`, which the case loads before the body runs.
fn unmoved_baseline(bank: Bank, lane: usize) -> u32 {
    match bank {
        Bank::PrimaryAttr => lane_value(SEED, lane as u32).to_bits(),
        _ => 0,
    }
}

/// **THE SUITE.** Each case's stated intent is held against what the reference interpreter
/// computes, over the whole register file - every lane, not only the named ones.
#[test]
fn every_conformance_case_computes_what_it_was_written_to_compute() {
    let mut failures = Vec::new();
    for case in all_cases() {
        let ran = run(&case);
        let want = (case.intent)(&seeded_inputs());

        for lane in &want {
            let got = lanes_of(lane.bank, &ran.regs)[lane.lane].to_bits();
            if got != lane.bits {
                failures.push(format!(
                    "{}: {:?}[{}] = {:#010x} ({}), intent {:#010x} ({})\n      the case checks: {}",
                    case.name,
                    lane.bank,
                    lane.lane,
                    got,
                    f32::from_bits(got),
                    lane.bits,
                    f32::from_bits(lane.bits),
                    case.checks,
                ));
            }
        }
        // >>> AND THE FRAGMENT PIPELINE'S OWN RESULT, which no register can carry. A discard
        // that both sides drop is a discard both sides AGREE about; only a stated intent tells
        // that apart from a discard that happened.
        if let Some(fragment) = case.fragment {
            let want = fragment(&seeded_inputs());
            if ran.regs.killed != want.killed {
                failures.push(format!(
                    "{}: killed = {}, intent {}
      the case checks: {}",
                    case.name, ran.regs.killed, want.killed, case.checks,
                ));
            }
            // Compared as BITS: a depth is a number, and `None == None` is a claim about the
            // program too - a translation that wrote a depth where the program writes none
            // makes the whole shader depth-replacing.
            let got = ran.regs.frag_depth;
            if got.map(f32::to_bits) != want.depth.map(f32::to_bits) {
                failures.push(format!(
                    "{}: frag_depth = {got:?}, intent {:?}
      the case checks: {}",
                    case.name, want.depth, case.checks,
                ));
            }
        } else if ran.regs.killed || ran.regs.frag_depth.is_some() {
            // A COMPUTE case that discarded or wrote a depth is not a compute case, and the
            // rig it runs in cannot represent either - so it must not happen quietly.
            failures.push(format!(
                "{}: a case with no fragment intent killed={} depth={:?}",
                case.name, ran.regs.killed, ran.regs.frag_depth,
            ));
        }
        // And nothing else moved. A case that only checked the lanes it named could not see an
        // instruction writing a register it had no business touching.
        for bank in [Bank::Output, Bank::Temp, Bank::Internal, Bank::PrimaryAttr] {
            for (n, v) in lanes_of(bank, &ran.regs).iter().enumerate().take(CASE_BANK_LANES) {
                if v.to_bits() != unmoved_baseline(bank, n)
                    && !want.iter().any(|l| l.bank == bank && l.lane == n)
                {
                    failures.push(format!(
                        "{}: {bank:?}[{n}] = {} moved from its baseline, and the intent names \
                         no such lane\n      the case checks: {}",
                        case.name,
                        v,
                        case.checks,
                    ));
                }
            }
        }
    }
    assert!(
        failures.is_empty(),
        "\n>>> {} CONFORMANCE FAILURE(S) - the reference does not compute what the program means:\n    {}\n",
        failures.len(),
        failures.join("\n    ")
    );
}

/// The operation KINDS the emitter translates, as [`op_kind`] names them - one per WGSL shape,
/// so a texture sample is four kinds (one per builtin) and a pack is several (one per
/// conversion). Every one of these must be exercised by at least one authored case.
const EMITTABLE_KINDS: &[&str] = &[
    "mad", "mul", "add", "frc", "dsx", "dsy", "min", "max", "dot", "rcp", "rsq", "log", "exp",
    "mov", "cmov", "cmov.u8", "limm", "bitwise", "sop2.fx8", "mov.fx8", "pack", "pack.int",
    "unpack.int", "pack.int.copy", "pack.unorm8", "unpack.unorm8", "imad", "imad.step0",
    "imad.step1", "loadidx", "idxadd", "ldmem", "vtst", "vtstmsk", "kill", "depthf", "br",
    "tex.implicit", "tex.bias", "tex.level", "tex.grad", "tex.gather4", "predicated",
];

/// The kinds that also have an F16 form the emitter translates, each of which must be exercised
/// AT F16 too. Real fragment programs are 70-90% F16, and a half-precision operand is a DIFFERENT
/// addressing rule (channel `c` is half `c & 1` of register `index + (c >> 1)`), so an F32 case
/// says nothing about it. `tex.coord16` is an F16 COORDINATE, independent of the result's width.
///
/// Not listed, with the reason: `dsx`/`dsy` at F16 (the render rig's screen ramp is an F32 lane,
/// and an F16 read of it would take a half of an f32 bit pattern - there is no F16 input with a
/// known gradient); `pack` (its source and destination widths are formats, which the pack cases
/// sweep directly).
const F16_KINDS: &[&str] = &[
    "mul.f16", "add.f16", "frc.f16", "min.f16", "max.f16", "dot.f16", "mad.f16", "rcp.f16",
    "rsq.f16", "log.f16", "exp.f16", "mov.f16", "cmov.f16", "vtst.f16", "vtstmsk.f16",
    "tex.implicit.f16", "tex.coord16",
];

/// >>> WHAT IS STILL UNAUTHORED, stated exactly. The gauge below FAILS if this list is wrong in
/// EITHER direction: a kind a new case covers must come off it, and a kind nothing covers must be
/// on it. So it cannot go stale, and "the conformance suite is complete" is the statement that it
/// is empty.
const NOT_YET_AUTHORED: &[&str] = &[];

/// The kind an instruction counts as, for the coverage gauge.
fn op_kind(op: Op) -> String {
    use vitaslop_gxp_shader::ir::TexLod;
    match op {
        Op::Tex { lod, .. } => match lod {
            TexLod::Implicit => "tex.implicit",
            TexLod::Bias => "tex.bias",
            TexLod::Level => "tex.level",
            TexLod::Gradient => "tex.grad",
        }
        .to_string(),
        other => other.mnemonic().to_string(),
    }
}

/// **THE SUITE'S COMPLETENESS, MEASURED.** Which emittable operation kinds no authored case
/// exercises. Printed every run, and pinned to [`NOT_YET_AUTHORED`] exactly.
#[test]
fn every_emittable_operation_kind_has_an_authored_case() {
    use std::collections::{BTreeMap, BTreeSet};
    let mut by_kind: BTreeMap<String, Vec<&'static str>> = BTreeMap::new();
    for case in all_cases() {
        let ran = run(&case);
        for i in &ran.shader.instrs {
            if matches!(i.op, Op::Nop) {
                continue;
            }
            by_kind.entry(op_kind(i.op)).or_default().push(case.name);
            if i.half_precision {
                by_kind.entry(format!("{}.f16", op_kind(i.op))).or_default().push(case.name);
            }
            if matches!(i.op, Op::Tex { coord_half: true, .. }) {
                by_kind.entry("tex.coord16".to_string()).or_default().push(case.name);
            }
            if i.pred != vitaslop_gxp_shader::ir::Predicate::Always {
                by_kind.entry("predicated".to_string()).or_default().push(case.name);
            }
        }
    }
    let all: Vec<&str> = EMITTABLE_KINDS.iter().chain(F16_KINDS).copied().collect();
    let uncovered: BTreeSet<&str> = all.iter().copied().filter(|k| !by_kind.contains_key(*k)).collect();
    println!("\nconformance coverage: {} of {} operation kinds authored", all.len() - uncovered.len(), all.len());
    for k in &all {
        match by_kind.get(*k) {
            Some(cases) => {
                let mut c = cases.clone();
                c.dedup();
                println!("  {k:<14} {} case(s), e.g. {}", c.len(), c[0]);
            }
            None => println!("  {k:<14} NONE"),
        }
    }
    // An `.f16` key outside `F16_KINDS` is fine (a pack's destination width, say); its BASE must
    // still be a listed kind.
    let unknown: Vec<&String> = by_kind
        .keys()
        .filter(|k| {
            let base = k.strip_suffix(".f16").unwrap_or(k);
            !EMITTABLE_KINDS.contains(&base) && !F16_KINDS.contains(&k.as_str())
        })
        .collect();
    assert!(unknown.is_empty(), "a case exercises a kind the gauge does not list: {unknown:?}");
    let listed: BTreeSet<&str> = NOT_YET_AUTHORED.iter().copied().collect();
    assert_eq!(
        uncovered, listed,
        "NOT_YET_AUTHORED must name exactly the kinds no case exercises"
    );
}

/// Every case's program must be one the EMITTER accepts and the decoder does not block, and
/// every instruction must be classified. An authored program that quietly decoded to
/// `Unsupported` would make its case vacuous - it would agree with the intent only because
/// nothing ran.
#[test]
fn every_conformance_program_is_fully_decoded_and_emittable() {
    for case in all_cases() {
        let ran = run(&case);
        assert!(
            !ran.shader.instrs.is_empty(),
            "{}: the assembled program decoded to nothing",
            case.name
        );
        assert!(
            ran.shader.fully_supported(),
            "{}: an instruction is blocked or unclassified: {:?}",
            case.name,
            ran.shader.instrs.iter().find(|i| !i.is_supported()).map(|i| (i.op, i.blocked)),
        );
        assert!(
            !ran.body.trim().is_empty(),
            "{}: the emitter produced an empty body",
            case.name
        );
    }
}

/// **A CASE THAT CANNOT FAIL IS NOT A TEST.** Every case here passes today, because the defects
/// it describes were all fixed - so the only thing that makes the suite worth running is that
/// each case would have FAILED before its fix. That rests on properties of the seeded inputs,
/// and those properties are stated here rather than assumed.
///
/// The failure mode this guards against is real and quiet: change [`SEED`], or narrow
/// [`lane_value`]'s range to the positives, and `fract` and the source modifiers would still
/// pass while checking nothing at all.
#[test]
fn the_seeded_inputs_make_every_case_non_vacuous() {
    let regs = seeded_inputs();

    // `fract`: x - floor(x) and x - trunc(x) differ only for a NEGATIVE operand, so the case
    // needs at least one.
    assert!(
        (0..4).any(|c| regs.pa[c] < 0.0),
        "conf_fract_is_x_minus_floor_x checks nothing unless some input is negative: {:?}",
        &regs.pa[0..4]
    );
    // The source modifiers: `-a` differs from `a`, and `|a|` from `a`, only where a is
    // non-zero, and the two must disagree with each other somewhere or min and max coincide.
    assert!(
        (0..4).any(|c| regs.pa[c] < 0.0) && (0..4).any(|c| regs.pa[c] > 0.0),
        "conf_source_modifiers_apply_before_the_operation needs inputs of both signs"
    );
    // The swizzle permutation `zxyw`: it must actually move something, which needs the
    // permuted channels to hold different values.
    assert!(
        regs.pa[0] != regs.pa[1] && regs.pa[1] != regs.pa[2] && regs.pa[0] != regs.pa[2],
        "conf_a_swizzle_permutes_the_source_channels needs distinct source channels"
    );
    // The write-mask case reads `sa` as well as `pa`, and a zero there would make the mad's
    // addend invisible.
    assert!(
        (0..4).all(|c| regs.sa[c] != 0.0),
        "conf_mad_write_mask_selects_lane_two_alone needs non-zero uniform lanes"
    );
    // The partial-write case: the lane it overwrites must end up DIFFERENT from what the full
    // write left there, or "the write happened" and "the write did not happen" look the same.
    assert!(
        2.0 * regs.pa[4] != regs.pa[0],
        "conf_a_partial_write_leaves_the_other_lanes_alone needs the overwrite to change lane 0"
    );
    // The two 16-bit HALF cases swap the halves of each source register, and a swap of two
    // EQUAL halves is invisible - it would pass against a reading that ignored the selector
    // entirely. All four halves must therefore be distinct.
    let halves: Vec<u16> = (0..4)
        .map(|n: usize| {
            let w = regs.pa[n >> 1].to_bits();
            (if n & 1 == 1 { w >> 16 } else { w & 0xffff }) as u16
        })
        .collect();
    assert!(
        halves.iter().enumerate().all(|(i, a)| halves[i + 1..].iter().all(|b| a != b)),
        "conf_an_equal_width_int_repack_moves_bit_patterns and \
         conf_a_signed_int_half_widens_with_its_sign need four DISTINCT halves, \
         or swapping them asserts nothing: {halves:04x?}"
    );
    // The TEXTURE sample, under the varying stand-in: the two coordinate lanes must be
    // non-zero and DIFFERENT, or the varying part adds nothing (or the same thing to two
    // channels) and the case degenerates into the constant one it replaced.
    assert!(
        regs.pa[0] != 0.0 && regs.pa[1] != 0.0 && regs.pa[0] != regs.pa[1],
        "conf_a_texture_sample_writes_all_four_channels checks no UV unless its two coordinate          lanes are non-zero and distinct: {:?}",
        &regs.pa[0..2]
    );
    // The PARTIAL-MASK repack: the four ways it can go wrong have to land on four different
    // answers, and with the halves as `h0..h3` those answers are
    //   correct                  r[0] = h2 | (h0 << 16)
    //   compacted DESTINATION    r[0] = h0 | (h3 << 16)
    //   compacted SELECTOR       r[0] = h2 | (h3 << 16)   (i.e. the setup, unchanged)
    // so `h0 != h2`, `h0 != h3` and `h2 != h0` are what separate them. The distinctness
    // assertion above covers all three, and this states which pairs the case actually leans on
    // so that narrowing the guard later cannot silently make this one vacuous.
    assert!(
        halves[0] != halves[2] && halves[0] != halves[3],
        "conf_a_partial_mask_half_repack_writes_the_half_its_channel_names cannot tell \
         channel-indexed placement from compacted placement unless h0, h2 and h3 differ: {halves:04x?}"
    );
    // And the SIGN half of that pair needs a half of each sign, or sign extension and a plain
    // unsigned widen agree on every channel the case looks at.
    assert!(
        halves.iter().any(|h| *h & 0x8000 != 0) && halves.iter().any(|h| *h & 0x8000 == 0),
        "conf_a_signed_int_half_widens_with_its_sign needs a half with its top bit SET and one \
         without, or a sign-extending widen and an unsigned one cannot be told apart: {halves:04x?}"
    );
    // The FLOAT->INT PACK: every one of its eight inputs must truncate to a NON-ZERO integer, or
    // that channel reads the same zero whether it was converted or never read at all - which is
    // exactly how an assembler bug that fed channels 2 and 3 from the wrong register went unseen
    // (the old inputs there were -0.205 and -0.192). And a negative must be among them, or two's
    // complement is not in the comparison.
    let pack_in: Vec<f32> = (4..8).chain(12..16).map(|n| regs.pa[n]).collect();
    assert!(
        pack_in.iter().all(|v| v.trunc() != 0.0) && pack_in.iter().any(|v| *v < -1.0),
        "conf_a_float_to_int_pack_truncates_and_packs_by_width needs non-zero truncations and a negative: {pack_in:?}"
    );
    // The SCREEN RAMP: the ramp lane plus both steps must stay in the lane's own binade, or
    // `x + step` rounds and the quad's difference is not exactly the step.
    let x = regs.pa[CASE_RAMP_LANE];
    let top = x + CASE_RAMP_DX + CASE_RAMP_DY;
    assert!(
        top.abs().log2().floor() == x.abs().log2().floor() || top.abs() < x.abs(),
        "the render rig's ramp lane pa[{CASE_RAMP_LANE}] = {x} crosses a binade under its steps"
    );
    // The TEST MASK: each instruction's four answers must include both a 1 and a 0, or a
    // translation that wrote a constant mask agrees with it.
    for (name, mask) in [
        ("sub > 0", (0..4).map(|c| regs.pa[c] - regs.sa[c] > 0.0).collect::<Vec<_>>()),
        ("-a + b >= 0", (0..4).map(|c| -regs.pa[c] + regs.sa[8 + c] >= 0.0).collect::<Vec<_>>()),
    ] {
        assert!(
            mask.contains(&true) && mask.contains(&false),
            "conf_a_test_mask_writes_one_or_zero_per_channel: `{name}` must be mixed: {mask:?}"
        );
    }
    // The BYTE SELECT: its second test register must have a non-zero low byte, or both selects
    // take the same arm and the `no` operand is never read.
    assert!(
        regs.pa[4].to_bits() & 0xff != 0,
        "conf_moves_swizzle_by_table_and_select_by_channel_and_byte needs pa[4]'s low byte \
         non-zero: {:08x}",
        regs.pa[4].to_bits()
    );
    // The PREDICATE case: p0 and p1 must take DIFFERENT values, or a swapped predicate register
    // writes the same registers the right one does.
    let p0 = regs.pa[0] - regs.pa[4] > 0.0;
    let p1 = regs.pa[1] - regs.pa[5] < 0.0;
    assert!(
        p0 != p1,
        "conf_a_test_sets_a_predicate_that_guards_writes_and_their_negation cannot see a swapped \
         predicate register unless p0 != p1: pa[0,1,4,5] = {:?}",
        [regs.pa[0], regs.pa[1], regs.pa[4], regs.pa[5]]
    );
}

/// Case names must be unique: they are the stems of the files the GPU runner reads, so a
/// collision would silently drop one case and report the other twice.
#[test]
fn conformance_case_names_are_unique() {
    let mut names: Vec<&str> = all_cases().iter().map(|c| c.name).collect();
    names.sort_unstable();
    let before = names.len();
    names.dedup();
    assert_eq!(names.len(), before, "two conformance cases share a name");
}

/// Write every case out for the GPU runner, with the INTENT as the expectation.
///
/// The corpus differential writes the interpreter's answer here; this writes the authored one.
/// That is the whole difference, and it is what turns the GPU run from "the two of them agree"
/// into "the GPU computes what the program means".
///
/// ```text
/// VITASLOP_GXP_CASES_OUT=<dir> cargo test -p vitaslop-gxp-shader --test conformance -- --ignored --nocapture
/// node vitaslop-web/e2e/gxpexec.mjs <dir>
/// ```
#[test]
#[ignore = "writes case files for the browser GPU runner; needs VITASLOP_GXP_CASES_OUT"]
fn write_every_conformance_case_for_the_gpu_runner() {
    let Some(out) = std::env::var_os("VITASLOP_GXP_CASES_OUT").map(std::path::PathBuf::from) else {
        eprintln!("VITASLOP_GXP_CASES_OUT unset - nothing to do");
        return;
    };
    std::fs::create_dir_all(&out).expect("create the case output directory");

    let mut written = 0usize;
    let mut wrote_textures = false;
    for case in all_cases() {
        let ran = run(&case);
        // >>> A CASE WITH A FRAGMENT INTENT GOES TO THE RENDER RIG, because a compute dispatch
        // has no discard and no depth to write. The rest stay on the compute rig, where the
        // varying texture stand-in sees a UV error of any size - moving them across would LOSE
        // coverage, which is the same call the corpus writer makes.
        let mut units: Vec<u8> = Vec::new();
        let (module, stage) = if case.fragment.is_some() {
            // The render rig binds REAL textures for the units the program samples - flat ones,
            // since no authored case declares a cube.
            let want = vitaslop_gxp_shader::wgsl::tex_units(&ran.shader, |_| false);
            units = want.iter().map(|b| b.unit).collect();
            // With the SCREEN RAMP: every authored render case gets it, and `run` hands the
            // reference the same lane, so a derivative has a stated answer.
            let (m, rewrites) = wrap_render_case_module_ramped(&ran.body, ran.kind, &ran.windows, &want, true, true)
                .unwrap_or_else(|why| panic!("{}: the render rig refused it: {why}", case.name));
            let gathers = ran.shader.instrs.iter().filter(|i| matches!(i.op, Op::TexGather { .. })).count();
            assert_eq!(
                rewrites.gathers, gathers,
                "{}: the render rig quantised {} of {gathers} gather coordinate(s)",
                case.name, rewrites.gathers
            );
            // >>> THE REWRITE IS COUNTED AGAINST THE INSTRUCTION STREAM, not trusted. A `kill`
            // the textual parse missed stays a real WGSL `discard`, which COMPILES - and a
            // discarded invocation is a helper whose stores do not land, so the case would read
            // its whole output buffer back as zero and report it as a translation defect.
            let kills = ran.shader.instrs.iter().filter(|i| matches!(i.op, Op::Kill)).count();
            assert_eq!(
                rewrites.kills, kills,
                "{}: the render rig rewrote {} of {kills} kill(s)",
                case.name, rewrites.kills
            );
            (m, Stage::Fragment)
        } else {
            let (m, _units) = wrap_compute_module_for(&ran.body, ran.kind, &ran.windows);
            (m, Stage::Compute)
        };
        assert!(
            stage != Stage::Compute || !module.contains("textureSample"),
            "{}: a sample the constant stand-in could not parse",
            case.name
        );

        // The intent, laid into a register file, then serialised by the SAME writer the corpus
        // cases use - so the runner cannot tell an authored case from a captured one, and the
        // two kinds of case can sit in one directory and be run together.
        let mut want = RegFile::with_lanes(CASE_BANK_LANES);
        // `pa` starts at the SEEDED input on both sides, so the expectation is laid over that
        // and only what the program CHANGED travels in the case - the same shape the corpus
        // writer uses. A `RegFile` starts zeroed, which for `pa` would claim the program wiped
        // every lane it never touched.
        for n in 0..CASE_BANK_LANES {
            want.pa[n] = lane_value(SEED, n as u32);
        }
        for lane in (case.intent)(&seeded_inputs()) {
            let slot = match lane.bank {
                Bank::Output => &mut want.o,
                Bank::Temp => &mut want.r,
                Bank::Internal => &mut want.i,
                Bank::PrimaryAttr => &mut want.pa,
                other => panic!("a conformance intent cannot name the {other:?} bank"),
            };
            slot[lane.lane] = f32::from_bits(lane.bits);
        }

        let prec = written_lane_precision(&ran.shader);
        let mut pairs: Vec<String> = Vec::new();
        nonzero_pairs(0, 0, &want.r, &prec, &mut pairs);
        nonzero_pairs(1, CASE_BANK_LANES, &want.o, &prec, &mut pairs);
        nonzero_pairs(2, CASE_BANK_LANES * 2, &want.i, &prec, &mut pairs);
        let seeded_pa: Vec<f32> = (0..CASE_BANK_LANES).map(|n| lane_value(SEED, n as u32)).collect();
        changed_pairs(3, CASE_BANK_LANES * 3, &want.pa, &seeded_pa, &prec, &mut pairs);

        // The two words a render case carries past the four banks, in the rig's own order: the
        // kill flag at precision code 3 (a flag, compared EXACTLY) and the depth the rig's one
        // declared forward map produced. `min`/`max` rather than `f32::clamp`, because WGSL's
        // `clamp` is `min(max(e,lo),hi)` and the two differ on a NaN.
        if let Some(fragment) = case.fragment {
            let want = fragment(&seeded_inputs());
            pairs.push(format!("[{},{},3]", CASE_BANK_LANES * 4, u32::from(want.killed)));
            #[allow(clippy::manual_clamp)]
            let depth = want.depth.map(|v| v.max(0.0).min(1.0)).unwrap_or(0.0);
            pairs.push(format!("[{},{},0]", CASE_BANK_LANES * 4 + 1, depth.to_bits()));
        }

        let half = ran.shader.instrs.iter().filter(|i| i.half_precision).count();
        write_case(
            &out,
            case.name,
            ran.kind,
            SEED,
            ran.shader.instrs.len(),
            half,
            &module,
            &ran.mem_words,
            &pairs,
            // >>> AN AUTHORED CASE NEEDS NO INPUT OVERRIDES, and that is a statement about what
            // it is rather than a gap. The corpus rig substitutes plausible counts and indices
            // because a CAPTURED program's registers came from a draw nobody has; an authored
            // program's inputs are part of what the author WROTE, and a harness that quietly
            // replaced one would be grading a program nobody intended.
            &[],
            stage,
            // The render rig's bindings, ascending by unit, all flat.
            &units,
            &vec![0u8; units.len()],
        );
        if !units.is_empty() {
            wrote_textures = true;
        }
        written += 1;
        println!("  {:<52} {}", case.name, case.checks);
    }

    // The render rig's texels travel as BYTES beside the cases, exactly as the corpus writer
    // ships them - the runner uploads what the reference read, not a second generator's output.
    if wrote_textures {
        std::fs::write(out.join("casetex.bin"), vitaslop_gxp_shader::wgsl::case_tex_bytes())
            .expect("write the render rig's texels");
    }
    println!("\n=== {written} conformance case(s) written to {} ===", out.display());
    println!("  Each carries its AUTHORED intent as the expectation, not the reference's answer,");
    println!("  so a divergence names which side is wrong rather than only that they differ.");
    println!("\n  run them:  node vitaslop-web/e2e/gxpexec.mjs {}", out.display());
}

