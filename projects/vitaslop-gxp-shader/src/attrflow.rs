//! WHAT CONSTANT A VERTEX ATTRIBUTE'S SURPLUS LANES MUST CARRY, answered from the programs
//! themselves rather than from a global default.
//!
//! # The problem this exists to solve
//! A guest vertex attribute is often bound NARROWER than the shader declares - a `vec4`
//! parameter fed by an `F32x2` stream. The lanes above the binding are not written by the
//! vertex fetch, and the recompiled pipeline has to put SOMETHING in them. That constant was a
//! single global for the whole renderer, and **two shipping titles need opposite values**:
//!
//!   * one title's sky/background family forwards a four-component `In.UV1` - bound `F16x2` -
//!     straight into a varying its fragment reads as a colour MODULATE. A zero in the surplus
//!     lanes is a DEAD CHANNEL there; the whole outdoor world reads green. It needs 1.0.
//!   * another title's particle/effect programs read a four-component colour and a
//!     four-component sprite CORNER INDEX, both bound two-wide. A 1.0 in those lanes is a
//!     saturated colour channel and a stretched quad. They need 0.
//!
//! The guest FORMAT does not separate them (both titles have over-wide attributes on plain
//! F32), and the binding width is read straight off the guest's own `SceGxmVertexAttribute`,
//! so it is not a mis-read either. The only thing left that distinguishes the two cases is
//! **what the surplus lane FEEDS**, and that is a dataflow question over the linked pair.
//!
//! # The rule, and why it is this one
//! The fill is the IDENTITY of the operation the lane's contribution enters:
//!
//!   * a lane that only ever passes through value-preserving steps - a move, a format repack,
//!     a conditional select - and MULTIPLICATIONS is one whose contribution is scaling
//!     something else. Its identity is **1.0**, and supplying it makes the lane vanish from
//!     the result exactly as an unwritten channel of a modulate should.
//!   * a lane reaching anything else - an ADD, a multiply-ADD (the product is a term of a sum,
//!     so the sum's identity is what matters, not the product's), a dot product, a texture
//!     coordinate, a compare, a bitwise op - contributes ADDITIVELY or positionally. Its
//!     identity is **0**.
//!
//! A `mad` is deliberately on the second list even though the lane is one of its multiplicands:
//! `a*b + c` with `a = 1` is `b + c`, which is not `c`. That is the ordinary vertex TRANSFORM -
//! a surplus position lane multiplying a matrix column - and 1.0 there adds a whole column.
//!
//! # Analysis shape
//! A forward taint walk at 16-bit HALF granularity over the vertex program, then across the
//! varying interface into the fragment program. Taint that reaches a blocking use anywhere
//! answers ZERO for that lane; taint that reaches only transparent uses (or nothing at all)
//! answers IDENTITY. Unread lanes answer IDENTITY because the fill cannot be observed, which
//! also keeps this change confined to the lanes it can actually explain.
//!
//! The walk is CONSERVATIVE towards zero in every place it cannot be exact - an unresolvable
//! indexed read of a tainted bank, an operand mode it does not model - because a wrong 1.0 is
//! the defect this module was written for. The FOURTH component is the one exception and is
//! pinned to 1.0 without asking the walk; see [`lane_fill`] for the frame that settled that.
//!
//! # What it changes, measured
//! Ridge Racer, the title whose picture fixed the old 1.0 default, is **BIT-IDENTICAL** under
//! this analysis and under `VITASLOP_GXP_ATTR_FILL=one`: 11 of 11 frame-pinned headless shots
//! at zero mean absolute error, including `f004800`, the front end that a wrong answer blanks
//! completely. Mortal Kombat's boot and logo stretch is identical too (4 of 4 shots); its
//! gameplay is an attract-mode demo that picks a different match per run, so a pixel A/B over
//! that stretch answers nothing and none was claimed
//! [[vitaslop-a-verdict-over-a-broken-key-is-void]].

use crate::ir::{Bank, Instr, Op, Operand, Predicate, Shader};
use crate::wgsl::BANK_REGS;

/// The constant a surplus attribute lane is fed.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum Fill {
    /// 1.0 - the lane scales something, so the identity of a multiply is what makes it vanish.
    Identity,
    /// 0.0 - the lane reaches an additive, positional or otherwise non-scaling use.
    Zero,
    /// **NOTHING READS THIS LANE**, so no value it is given can be observed.
    ///
    /// Its `value()` is 1.0, exactly what [`Fill::Identity`] gave before this variant existed,
    /// so nothing that writes a fill changes - and that is the point: this splits a claim that
    /// was already being made, it does not make a new one. What it adds is a caller that can
    /// ask "does this lane need MY constant?" and be told no.
    ///
    /// The distinction pays for itself in the renderer's vertex path. Feeding a surplus lane a
    /// chosen constant means WRITING it, which means repacking the guest's vertex row into a
    /// buffer this code owns; a lane nobody reads can be left to whatever the hardware's own
    /// vertex fetch supplies, and the row can then be bound as it stands. MEASURED on a golf
    /// title, where this is the difference: 102 of its pipelines were held off that path by a
    /// lane that turned out to be unread.
    Unobserved,
}

impl Fill {
    pub fn value(self) -> f32 {
        match self {
            Fill::Identity | Fill::Unobserved => 1.0,
            Fill::Zero => 0.0,
        }
    }

    /// Whether this lane's value is READ at all - i.e. whether `value()` is a decision or an
    /// arbitrary choice among equals.
    pub fn observed(self) -> bool {
        !matches!(self, Fill::Unobserved)
    }
}

/// Where one interpolated vertex OUTPUT lane lands in the fragment stage. Mirrors the linker's
/// own `ComponentDest`, which is private to it; the conversion happens at the call site so this
/// analysis stays independent of the interface planner.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum FragLand {
    /// A whole 32-bit PA register (an F32 interpolant's component).
    Register(u32),
    /// One 16-bit half of a PA register (an F16 interpolant packs two components per register).
    Half { register: u32, slot: u32 },
    /// Consumed by the PDS as a texture-sample coordinate - it never reaches the fragment code,
    /// and a coordinate is positional, so this is a BLOCKING landing.
    SampleCoord,
}

/// Banks this walk tracks taint in. SA is a uniform binding and the INDEX file and predicates
/// are only ever reached through a blocking use, so four banks cover every path a vertex
/// attribute's value can take.
///
/// INTERNAL is in the list and that is not a detail: the corpus's particle programs stage
/// almost every intermediate through `i0`/`i8`, so a walk that dropped taint there would lose
/// the value halfway and report the lane unread - an answer of 1.0 arrived at by not looking.
const TRACKED_BANKS: usize = 4;

fn bank_slot(b: Bank) -> Option<usize> {
    match b {
        Bank::Temp => Some(0),
        Bank::PrimaryAttr => Some(1),
        Bank::Output => Some(2),
        Bank::Internal => Some(3),
        _ => None,
    }
}

/// A taint set over `(bank, register, 16-bit half)`.
#[derive(Clone)]
struct Taint {
    t: Vec<bool>,
}

impl Taint {
    fn new() -> Taint {
        Taint { t: vec![false; TRACKED_BANKS * BANK_REGS * 2] }
    }
    fn index(bank: usize, reg: usize, half: usize) -> usize {
        (bank * BANK_REGS + reg) * 2 + half
    }
    fn get(&self, bank: usize, reg: usize, half: usize) -> bool {
        reg < BANK_REGS && self.t[Taint::index(bank, reg, half)]
    }
    fn set(&mut self, bank: usize, reg: usize, half: usize, v: bool) {
        if reg < BANK_REGS {
            self.t[Taint::index(bank, reg, half)] = v;
        }
    }
    fn any_in_bank(&self, bank: usize) -> bool {
        self.t[bank * BANK_REGS * 2..(bank + 1) * BANK_REGS * 2].iter().any(|b| *b)
    }
}

/// How a source operand's value reaches the instruction's result.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
enum Role {
    /// The value passes through with 1.0 as the identity: a move, a repack, a select arm, or a
    /// multiplicand of a PLAIN multiply.
    Through,
    /// Anything else. See the module docs for why `mad`'s multiplicands are here.
    Blocking,
}

fn role_of(op: Op, src: usize) -> Role {
    match op {
        // Value-preserving copies. A format repack changes the STORAGE width, not the number.
        Op::Mov | Op::Pack { .. } | Op::PackToInt { .. } | Op::PackUnorm8 { .. } | Op::CopyFx8 => {
            Role::Through
        }
        // `dest = src1 * src2`: both multiplicands scale, so 1.0 is the identity for either.
        Op::Mul => Role::Through,
        // `select(src2, src1, test(src0))` - the two arms are copies, the test is not.
        Op::Cmov { .. } => {
            if src < 2 {
                Role::Through
            } else {
                Role::Blocking
            }
        }
        _ => Role::Blocking,
    }
}

/// Whether an operation propagates a source channel to the SAME destination channel. Only the
/// [`Role::Through`] ops do, and they all do it per channel.
fn propagates_per_channel(op: Op) -> bool {
    matches!(
        op,
        Op::Mov
            | Op::Pack { .. }
            | Op::PackToInt { .. }
            | Op::PackUnorm8 { .. }
            | Op::CopyFx8
            | Op::Mul
            | Op::Cmov { .. }
    )
}

/// The `(register, halves)` a source operand's channel `c` reads, at the instruction's own
/// source precision. Mirrors the linker's read model exactly - an F32 channel reads register
/// `index + selector`, the four F16 channels share a register PAIR, and a packed-byte operand
/// keeps all four channels in ONE register.
fn src_reg(instr: &Instr, src: &Operand, c: usize) -> Option<(usize, std::ops::Range<usize>)> {
    let sel = src.swizzle[c] as usize;
    if sel > 3 {
        return None; // a swizzle constant reads no register
    }
    Some(if instr.source_packed_bytes() {
        (src.index as usize, 0..2)
    } else if instr.source_half_precision() {
        (src.index as usize + (sel >> 1), (sel & 1)..(sel & 1) + 1)
    } else {
        (src.index as usize + sel, 0..2)
    })
}

/// The `(register, halves)` an instruction's destination channel `c` writes.
fn dest_reg(instr: &Instr, d: &Operand, c: usize) -> (usize, std::ops::Range<usize>) {
    if instr.half_precision {
        (d.index as usize + (c >> 1), (c & 1)..(c & 1) + 1)
    } else {
        (d.index as usize + c, 0..2)
    }
}

/// One stage's walk. Returns `true` when the taint reached a BLOCKING use.
fn walk(shader: &Shader, taint: &mut Taint) -> bool {
    let mut blocked = false;
    for instr in &shader.instrs {
        let read = crate::link::read_channels(instr);
        // An INDEXED source addresses `bank[i + offset]`, and the index register's value is not
        // known here. If any taint remains in the bank it addresses, this read may be of the
        // tainted lane and its role cannot be established - answer zero rather than guess.
        for src in &instr.srcs {
            if src.bank == Bank::Indexed {
                let sub = crate::ir::indexed_sub_bank(src.index);
                if bank_slot(sub).is_some_and(|b| taint.any_in_bank(b)) {
                    blocked = true;
                }
            }
        }

        // Which destination channels a transparent source propagates taint into.
        let mut propagate = [false; 4];
        for (i, src) in instr.srcs.iter().enumerate() {
            let Some(bank) = bank_slot(src.bank) else { continue };
            for c in 0..4 {
                if !read[c] {
                    continue;
                }
                let Some((reg, halves)) = src_reg(instr, src, c) else { continue };
                if !halves.clone().any(|h| taint.get(bank, reg, h)) {
                    continue;
                }
                match role_of(instr.op, i) {
                    Role::Through if propagates_per_channel(instr.op) => propagate[c] = true,
                    // A `Through` role on an op with no per-channel destination model would
                    // lose the value; there is no such op today, but a future one must not
                    // silently drop the taint.
                    Role::Through => blocked = true,
                    Role::Blocking => blocked = true,
                }
            }
        }

        // Assign the destination. A write KILLS the taint of the lanes it overwrites - unless
        // it is conditional, where the old value survives on the other path.
        let Some(d) = instr.dest.as_ref() else { continue };
        let Some(dbank) = bank_slot(d.bank) else {
            // A destination outside the tracked banks (an index register, a predicate) that any
            // taint reached is already counted blocking above.
            continue;
        };
        let conditional = instr.pred != Predicate::Always;
        for c in 0..4 {
            if !instr.write_mask[c] {
                continue;
            }
            let (reg, halves) = dest_reg(instr, d, c);
            for h in halves {
                if propagate[c] {
                    taint.set(dbank, reg, h, true);
                } else if !conditional {
                    taint.set(dbank, reg, h, false);
                }
            }
        }
    }
    blocked
}

/// The OUTPUT-bank slot index in a [`Taint`].
const OUT_BANK: usize = 2;
const PA_BANK: usize = 1;

/// Decide the fill for ONE surplus lane of a vertex attribute: the PA register `lane` of the
/// vertex program, followed through the vertex stage, the varying interface and the fragment
/// stage.
///
/// `varyings` maps a vertex OUTPUT lane to where it lands in the fragment stage; lanes 0..3 are
/// clip POSITION and are not in it.
pub fn lane_fill(
    vsh: &Shader,
    fsh: &Shader,
    base_lane: u32,
    component: u32,
    varyings: &[(u32, FragLand)],
) -> Fill {
    // THE FOURTH COMPONENT IS 1.0 AND THE ANALYSIS DOES NOT GET A VOTE. `w = 1` is what the
    // hardware, WebGPU, GL and D3D all supply for an unbound fourth lane, so a three-component
    // bind into a declared `vec4` is a convention rather than a choice - and the convention is
    // the one thing the two readings this module arbitrates between AGREE on.
    //
    // It is pinned because letting the walk answer it was MEASURED to destroy a title. Ridge
    // Racer binds a three-component POSITION into a four-component parameter on four pairs; the
    // walk correctly finds `w` reaching a multiply-ADD (it multiplies the transform's fourth
    // matrix column) and answered ZERO, which is the projection's translation term deleted. The
    // whole front end went BLACK, every frame from f000800 on, against a control run of the same
    // recipe. The rule the walk implements is about lanes that carry DATA; `w` carries the
    // homogeneous coordinate, and its identity is 1 for exactly the reason a position is not
    // data.
    //
    // >>> THE PIN IS ON THE VALUE, NOT ON THE QUESTION. The walk still runs for `w`, and its
    // answer is narrowed to the two that carry 1.0 - `Identity` when anything reads the lane,
    // `Unobserved` when nothing does. Zero, the answer that destroyed the title above, remains
    // unreachable for `w`. The distinction matters to a caller that is not choosing a constant
    // at all: a `w` nobody reads is a lane the hardware may fill with whatever falls under it,
    // which is what lets a three-component F16 attribute be fetched by a four-component format.
    let pinned = component == 3;
    // Every answer below leaves through here, so the pin is applied ONCE and cannot be missed by
    // a path added later - which is the failure this shape exists to prevent, given what the one
    // wrong answer for `w` cost.
    let pin = |f: Fill| match (pinned, f) {
        (true, Fill::Unobserved) => Fill::Unobserved,
        (true, _) => Fill::Identity,
        (false, f) => f,
    };
    let lane = base_lane + component;
    let mut taint = Taint::new();
    taint.set(PA_BANK, lane as usize, 0, true);
    taint.set(PA_BANK, lane as usize, 1, true);
    if walk(vsh, &mut taint) {
        return pin(Fill::Zero);
    }

    // Clip POSITION. A surplus lane that survives to a position lane is positional by
    // definition, whatever it passed through on the way.
    for l in 0..4 {
        if taint.get(OUT_BANK, l, 0) || taint.get(OUT_BANK, l, 1) {
            return pin(Fill::Zero);
        }
    }

    // Cross the interface. A vertex output lane the fragment does not read drops the taint,
    // which is correct: a lane nobody reads cannot decide anything.
    let mut ftaint = Taint::new();
    let mut crossed = false;
    for (vlane, land) in varyings {
        let l = *vlane as usize;
        if !(taint.get(OUT_BANK, l, 0) || taint.get(OUT_BANK, l, 1)) {
            continue;
        }
        crossed = true;
        match *land {
            FragLand::Register(r) => {
                ftaint.set(PA_BANK, r as usize, 0, true);
                ftaint.set(PA_BANK, r as usize, 1, true);
            }
            FragLand::Half { register, slot } => {
                ftaint.set(PA_BANK, register as usize, (slot & 1) as usize, true);
            }
            // A texture coordinate is positional.
            FragLand::SampleCoord => return pin(Fill::Zero),
        }
    }
    if !crossed {
        // NOTHING DOWNSTREAM CAN SEE THIS LANE. Either the vertex program never moved it to an
        // output at all, or the output lane it reached is one the fragment stage does not read -
        // and the position lanes, the one output that is read without being a varying, were
        // excluded just above. The value is unobservable, which is a stronger and more useful
        // statement than the 1.0 this used to return: see [`Fill::Unobserved`].
        return pin(Fill::Unobserved);
    }
    if walk(fsh, &mut ftaint) {
        return pin(Fill::Zero);
    }
    pin(Fill::Identity)
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::container::ProgramKind;
    use crate::ir::{Instr, Op, Operand};

    fn instr(op: Op, dest: Option<Operand>, srcs: Vec<Operand>) -> Instr {
        Instr {
            op,
            pred: Predicate::Always,
            dest,
            write_mask: [true; 4],
            srcs,
            half_precision: false,
            raw: 0,
            group: 0,
            blocked: None,
        }
    }

    fn shader(instrs: Vec<Instr>) -> Shader {
        Shader { kind: ProgramKind::Vertex, instrs }
    }

    fn pa(i: u8) -> Operand {
        Operand::plain(Bank::PrimaryAttr, i, 2)
    }
    fn out(i: u8) -> Operand {
        Operand::plain(Bank::Output, i, 1)
    }
    fn sa(i: u8) -> Operand {
        Operand::plain(Bank::SecondaryAttr, i, 3)
    }

    /// A lane the vertex forwards into a varying the fragment MODULATES: 1.0, the multiplicative
    /// identity - the sky/background case this module exists for.
    #[test]
    fn forwarded_into_a_fragment_modulate_is_identity() {
        let vsh = shader(vec![instr(Op::Mov, Some(out(4)), vec![pa(0)])]);
        let fsh = Shader {
            kind: ProgramKind::Fragment,
            instrs: vec![instr(Op::Mul, Some(out(0)), vec![pa(0), sa(0)])],
        };
        // Vertex output lane 4+2 = the surplus lane's own copy; the varying map says where it
        // lands. Lane 2 of the attribute at base 0 -> o[6] -> fragment pa[2].
        let varyings = [(6u32, FragLand::Register(2))];
        assert_eq!(lane_fill(&vsh, &fsh, 0, 2, &varyings), Fill::Identity);
    }

    /// The same lane reaching a multiply-ADD is ZERO: the product is a term of a sum, and 1.0
    /// there adds a whole matrix column.
    #[test]
    fn a_multiplicand_of_a_mad_is_zero() {
        let vsh = shader(vec![instr(Op::Mad, Some(out(0)), vec![pa(2), sa(0), sa(4)])]);
        let fsh = Shader { kind: ProgramKind::Fragment, instrs: vec![] };
        assert_eq!(lane_fill(&vsh, &fsh, 0, 2, &[]), Fill::Zero);
    }

    /// A lane that survives into clip POSITION is positional whatever it passed through.
    #[test]
    fn reaching_clip_position_is_zero() {
        let vsh = shader(vec![instr(Op::Mov, Some(out(0)), vec![pa(2)])]);
        let fsh = Shader { kind: ProgramKind::Fragment, instrs: vec![] };
        assert_eq!(lane_fill(&vsh, &fsh, 0, 2, &[]), Fill::Zero);
    }

    /// The FOURTH lane is 1.0 whatever the walk would have said - the convention every graphics
    /// API shares, and the one this module's two readings agree on. See [`lane_fill`] for the
    /// title that went black when the walk was allowed to answer it.
    #[test]
    fn the_fourth_lane_is_identity_without_asking() {
        let vsh = shader(vec![instr(Op::Mov, Some(out(0)), vec![pa(0)])]);
        let fsh = Shader { kind: ProgramKind::Fragment, instrs: vec![] };
        assert_eq!(lane_fill(&vsh, &fsh, 0, 3, &[]), Fill::Identity);
    }

    /// `w` REACHING A POSITION is still 1.0 - the exact case that blacked out a title's front
    /// end when the walk was allowed to answer ZERO for it. The pin narrows the answer; it does
    /// not stop the walk running.
    #[test]
    fn a_fourth_lane_that_multiplies_a_matrix_column_stays_identity() {
        // The shape the walk calls ZERO for any other lane: a multiplicand of a mad. The SAME
        // shape is written for lane 2 and for lane 3, so the only difference between the two
        // answers is the pin.
        let fsh = Shader { kind: ProgramKind::Fragment, instrs: vec![] };
        let lane2 = shader(vec![instr(Op::Mad, Some(out(0)), vec![pa(2), sa(0), sa(4)])]);
        let lane3 = shader(vec![instr(Op::Mad, Some(out(0)), vec![pa(3), sa(0), sa(4)])]);
        assert_eq!(lane_fill(&lane2, &fsh, 0, 2, &[]), Fill::Zero, "the rule for an ordinary lane");
        assert_eq!(lane_fill(&lane3, &fsh, 0, 3, &[]), Fill::Identity, "and the pin for `w`");
    }

    /// A `w` NOTHING reads reports itself unobserved, while still valuing 1.0.
    ///
    /// This is the distinction that lets a three-component F16 attribute be fetched by a
    /// four-component vertex format: the fourth lane picks up whatever guest bytes follow, and
    /// that is only admissible when nothing can look at it.
    #[test]
    fn an_unread_fourth_lane_is_unobserved() {
        let vsh = shader(vec![instr(Op::Mov, Some(out(4)), vec![pa(0)])]);
        let fsh = Shader { kind: ProgramKind::Fragment, instrs: vec![] };
        let f = lane_fill(&vsh, &fsh, 0, 3, &[(4, FragLand::Register(0))]);
        assert_eq!(f, Fill::Unobserved);
        assert_eq!(f.value(), 1.0);
    }

    /// A lane NOTHING reads is `Unobserved`, not `Identity`.
    ///
    /// The two carry the same 1.0 and are still different answers: `Identity` says a constant
    /// was CHOSEN and any other would show, `Unobserved` says no choice is being made at all -
    /// which is what lets the renderer bind the guest's vertex row as it stands instead of
    /// rewriting it to place a value nobody will look at.
    #[test]
    fn a_lane_nothing_reads_is_unobserved() {
        // The vertex program moves a DIFFERENT lane; lane 2's taint reaches no output.
        let vsh = shader(vec![instr(Op::Mov, Some(out(4)), vec![pa(0)])]);
        let fsh = Shader { kind: ProgramKind::Fragment, instrs: vec![] };
        assert_eq!(lane_fill(&vsh, &fsh, 0, 2, &[(4, FragLand::Register(0))]), Fill::Unobserved);
        assert!(!lane_fill(&vsh, &fsh, 0, 2, &[(4, FragLand::Register(0))]).observed());
        // And it is still the same 1.0 to anyone that writes a fill, so the repack is unchanged.
        assert_eq!(Fill::Unobserved.value(), Fill::Identity.value());
    }

    /// A lane the vertex DOES forward, into a varying the fragment does not read, is unobserved
    /// too - the interface is where the observation is lost, not the vertex stage.
    #[test]
    fn a_varying_the_fragment_never_reads_is_unobserved() {
        let vsh = shader(vec![instr(Op::Mov, Some(out(6)), vec![pa(2)])]);
        let fsh = Shader { kind: ProgramKind::Fragment, instrs: vec![] };
        // o[6] carries the lane, and the varying map does not mention o[6] at all.
        assert_eq!(lane_fill(&vsh, &fsh, 0, 2, &[(4, FragLand::Register(0))]), Fill::Unobserved);
    }

    /// A write to the lane's own register before it is read does NOT make the attribute's value
    /// observable - but a write cannot travel backwards either, so the taint must survive the
    /// reads that precede it.
    #[test]
    fn a_later_overwrite_does_not_erase_an_earlier_use() {
        let vsh = shader(vec![
            instr(Op::Mad, Some(out(0)), vec![pa(2), sa(0), sa(4)]),
            instr(Op::Mov, Some(pa(2)), vec![sa(1)]),
        ]);
        let fsh = Shader { kind: ProgramKind::Fragment, instrs: vec![] };
        assert_eq!(lane_fill(&vsh, &fsh, 0, 2, &[]), Fill::Zero);
    }

    /// A texture COORDINATE is positional: a fill of 1.0 there samples the wrong texel.
    #[test]
    fn a_prefetched_sample_coordinate_is_zero() {
        // `Mov o[4] <- pa[2]` puts the lane's own channel 0 in o[4].
        let vsh = shader(vec![instr(Op::Mov, Some(out(4)), vec![pa(2)])]);
        let fsh = Shader { kind: ProgramKind::Fragment, instrs: vec![] };
        assert_eq!(lane_fill(&vsh, &fsh, 0, 2, &[(4, FragLand::SampleCoord)]), Fill::Zero);
    }
}
