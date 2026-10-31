//! PER-DRAW memory-window extents: how far past each window's base ANY invocation of one draw
//! can load, computed from that draw's own inputs.
//!
//! # Why a window needs this
//! A memory window is a snapshot of guest bytes the host uploads with the draw (see
//! [`crate::module::MemWindow`]). Its size has to cover every byte a load can address, and for
//! most programs the program's own arithmetic bounds that ([`crate::module`]'s `hard_reach`).
//! Some programs' addresses are bounded only by values that arrive WITH the draw: a
//! depth-reconstruct pass loads `ptr + (stride * y + x) * 32`, where `x, y` come from the
//! fragment's window position and `stride` from a uniform - so the reach depends on the target
//! the draw rasterises into and on the uniform's value, and nothing in the program bounds it.
//! Sized as declared, its four windows would be 17 MiB each, uploaded every draw.
//!
//! # What this computes, and why it is a BOUND rather than an estimate
//! An INTERVAL evaluation of the whole program (secondary then primary, the order the emitted
//! module runs them in) over a register file whose every 32-bit lane holds a range of values:
//!
//! * the SA registers the draw's uniform image covers, its container literals and its window
//!   pointers hold their EXACT values - the module initialises them from exactly these;
//! * the POSITION interpolant's x and y hold `[0, width]` and `[0, height]` of the target the
//!   draw rasterises into - a fragment is never generated outside it;
//! * everything else - every other varying, every temporary, every texture result - holds
//!   "anything".
//!
//! The integer and float operations an address is built from are evaluated on the ranges
//! (floats with outward rounding plus one ulp, integers only while no product or sum can wrap),
//! and EVERY other instruction sets each lane it may write to "anything". A predicated
//! instruction joins its result with the old value, since the predicate is not modelled. A
//! program that BRANCHES or writes through an index register is not evaluated at all
//! ([`DrawReach::build`] declines it), because a straight-line walk would not visit what it runs.
//!
//! Each memory load then contributes the range of addresses it can touch, and a window's extent
//! is the furthest of those that lands inside its DECLARED range. A load whose address is
//! "anything" reaches every window's whole declared extent - which is exactly the size it binds
//! at without this - so the analysis never makes a window smaller than what a load can address.
//! An address inside no window reads zero exactly as it did before.

use crate::container::{Program, ProgramKind, VaryingUsage};
use crate::ir::{Bank, BitwiseKind, Instr, Op, Operand, Predicate};
use crate::wgsl::{bank_prec, cnst6_channel_value, Prec};

/// One 32-bit lane's abstract value.
#[derive(Debug, Clone, Copy, PartialEq)]
enum V {
    /// Any bit pattern at all.
    Top,
    /// An unsigned integer bit pattern in `[lo, hi]`. An exact pattern is `I(v, v)`.
    I(u32, u32),
    /// A finite f32 in `[lo, hi]` (bit patterns unknown beyond that: `-0.0` and `+0.0` alike).
    F(f32, f32),
    /// Two independent 16-bit HALVES: the low one in `[.0, .1]`, the high one in `[.2, .3]`.
    /// What a 16-bit pack leaves - one half written, the other still holding whatever it held -
    /// and the only form in which reading the written half back loses nothing.
    P(u32, u32, u32, u32),
}

const LANES: usize = 256;

/// The register banks the evaluation tracks, one [`V`] per 32-bit lane.
struct Regs {
    banks: [Vec<V>; 5],
}

fn bank_slot(bank: Bank) -> Option<usize> {
    Some(match bank {
        Bank::Temp => 0,
        Bank::PrimaryAttr => 1,
        Bank::SecondaryAttr => 2,
        Bank::Output => 3,
        Bank::Internal => 4,
        _ => return None,
    })
}

impl Regs {
    fn new() -> Regs {
        Regs { banks: std::array::from_fn(|_| vec![V::Top; LANES]) }
    }
    fn get(&self, bank: Bank, lane: u32) -> V {
        bank_slot(bank).and_then(|b| self.banks[b].get(lane as usize).copied()).unwrap_or(V::Top)
    }
    fn set(&mut self, bank: Bank, lane: u32, v: V) {
        if let Some(slot) = bank_slot(bank).and_then(|b| self.banks[b].get_mut(lane as usize)) {
            *slot = v;
        }
    }
}

/// The smallest range holding both.
fn join(a: V, b: V) -> V {
    match (a, b) {
        (V::I(a0, a1), V::I(b0, b1)) => V::I(a0.min(b0), a1.max(b1)),
        (V::F(a0, a1), V::F(b0, b1)) => V::F(a0.min(b0), a1.max(b1)),
        (V::P(..), V::P(..) | V::I(..)) | (V::I(..), V::P(..)) => {
            let ((a0, a1), (a2, a3)) = halves(a);
            let ((b0, b1), (b2, b3)) = halves(b);
            V::P(a0.min(b0), a1.max(b1), a2.min(b2), a3.max(b3))
        }
        _ => V::Top,
    }
}

/// `v` as an unsigned integer range.
fn int(v: V) -> (u32, u32) {
    match v {
        V::I(lo, hi) => (lo, hi),
        // Only a NORMAL float has one bit pattern a float operation is bound to produce: a zero
        // may carry either sign and a denormal may have been flushed.
        V::F(lo, hi) if lo.to_bits() == hi.to_bits() && lo.is_normal() => (lo.to_bits(), lo.to_bits()),
        V::P(l0, l1, h0, h1) => ((h0 << 16) | l0, (h1 << 16) | l1),
        _ => (0, u32::MAX),
    }
}

/// `v`'s low and high 16-bit halves as ranges.
fn halves(v: V) -> ((u32, u32), (u32, u32)) {
    match v {
        V::P(l0, l1, h0, h1) => ((l0, l1), (h0, h1)),
        v => {
            let i = int(v);
            (field(i, 0, 16), (i.0 >> 16, i.1 >> 16))
        }
    }
}

/// [`field`] of a lane's value, read inside one half where the field lies in one.
fn vfield(v: V, shift: u32, bits: u32) -> (u32, u32) {
    if let V::P(..) = v
        && shift % 16 + bits <= 16
    {
        let (lo, hi) = halves(v);
        return field(if shift >= 16 { hi } else { lo }, shift % 16, bits);
    }
    field(int(v), shift, bits)
}

/// [`insert`] into a lane's value, keeping the halves apart when a whole half is replaced.
fn vinsert(old: V, shift: u32, bits: u32, e: (u32, u32)) -> V {
    if bits == 16 && (shift == 0 || shift == 16) {
        let (lo, hi) = halves(old);
        let (lo, hi) = if shift == 0 { (e, hi) } else { (lo, e) };
        return V::P(lo.0, lo.1, hi.0, hi.1);
    }
    let (lo, hi) = insert(int(old), shift, bits, e);
    V::I(lo, hi)
}

/// `v` as a float range, when it is one: a float range, or an exact pattern whose float is
/// finite.
fn float(v: V) -> Option<(f32, f32)> {
    match v {
        V::F(lo, hi) => Some((lo, hi)),
        V::I(..) | V::P(..) => {
            let (lo, hi) = int(v);
            let f = f32::from_bits(lo);
            (lo == hi && f.is_finite()).then_some((f, f))
        }
        V::Top => None,
    }
}

fn next_down(f: f32) -> f32 {
    if f == 0.0 {
        return -f32::from_bits(1);
    }
    if f > 0.0 { f32::from_bits(f.to_bits() - 1) } else { f32::from_bits(f.to_bits() + 1) }
}

fn next_up(f: f32) -> f32 {
    -next_down(-f)
}

/// An f64 range rounded OUTWARD to f32 and widened by one more ulp each side, with zero
/// included wherever a bound lies in the denormal range (a GPU may flush it). That covers a
/// correctly rounded result, a fused or unfused multiply-add, and a flushed denormal alike.
fn widen(lo: f64, hi: f64) -> V {
    if !lo.is_finite() || !hi.is_finite() || lo > hi {
        return V::Top;
    }
    let mut l = lo as f32;
    if f64::from(l) > lo {
        l = next_down(l);
    }
    let mut h = hi as f32;
    if f64::from(h) < hi {
        h = next_up(h);
    }
    let (mut l, mut h) = (next_down(l), next_up(h));
    if !l.is_finite() || !h.is_finite() {
        return V::Top;
    }
    if l.abs() < f32::MIN_POSITIVE {
        l = l.min(0.0);
    }
    if h.abs() < f32::MIN_POSITIVE {
        h = h.max(0.0);
    }
    if l > 0.0 && l < f32::MIN_POSITIVE {
        l = 0.0;
    }
    if h < 0.0 && h > -f32::MIN_POSITIVE {
        h = 0.0;
    }
    V::F(l, h)
}

fn fmul(a: (f32, f32), b: (f32, f32)) -> (f64, f64) {
    let c = [
        f64::from(a.0) * f64::from(b.0),
        f64::from(a.0) * f64::from(b.1),
        f64::from(a.1) * f64::from(b.0),
        f64::from(a.1) * f64::from(b.1),
    ];
    (c.iter().copied().fold(f64::INFINITY, f64::min), c.iter().copied().fold(f64::NEG_INFINITY, f64::max))
}

/// The `bits`-wide field at `shift` of a value in `[lo, hi]`: exact while every value in the
/// range agrees on the bits above the field, the whole field's range otherwise.
fn field(v: (u32, u32), shift: u32, bits: u32) -> (u32, u32) {
    let mask = if bits >= 32 { u32::MAX } else { (1u32 << bits) - 1 };
    let top = shift + bits;
    let same_above = top >= 32 || (v.0 >> top) == (v.1 >> top);
    if same_above { ((v.0 >> shift) & mask, (v.1 >> shift) & mask) } else { (0, mask) }
}

/// A value in `old` with its `bits`-wide field at `shift` replaced by one in `e`. The three
/// parts occupy disjoint bits, so each part's own bounds bound the whole.
fn insert(old: (u32, u32), shift: u32, bits: u32, e: (u32, u32)) -> (u32, u32) {
    let top = shift + bits;
    let above = |x: u32| if top >= 32 { 0 } else { u64::from(x >> top) << top };
    let below = if shift == 0 { (0, 0) } else { field(old, 0, shift) };
    let lo = above(old.0) | (u64::from(e.0) << shift) | u64::from(below.0);
    let hi = above(old.1) | (u64::from(e.1) << shift) | u64::from(below.1);
    (lo as u32, hi as u32)
}

/// `a * b + c` on unsigned ranges, or "anything" when it can wrap.
fn mad(a: (u32, u32), b: (u32, u32), c: (u32, u32)) -> V {
    let hi = u64::from(a.1) * u64::from(b.1) + u64::from(c.1);
    if hi > u64::from(u32::MAX) {
        return V::Top;
    }
    V::I((u64::from(a.0) * u64::from(b.0) + u64::from(c.0)) as u32, hi as u32)
}

/// The smallest all-ones number at or above `x`.
fn ones_above(x: u32) -> u32 {
    if x == 0 { 0 } else { u32::MAX >> x.leading_zeros() }
}

/// Each window's extent in bytes for ONE DRAW of the fragment program `blob`: its per-draw
/// reach ([`DrawReach::extents`]) for a [`MemWindow::per_draw`] set, its fixed
/// [`MemWindow::bytes`] for every other. `windows` is what [`crate::module::resolve_mem_windows`]
/// returned for the program, `sa_image`/`addrs`/`target` as [`DrawReach::extents`] takes them.
///
/// The live capture (which reads that many guest bytes) and the renderer (which binds them) both
/// ask this with the same inputs, so they agree. The analysis is built once per program and kept;
/// a draw costs one walk of the program over ranges.
///
/// [`MemWindow::per_draw`]: crate::module::MemWindow::per_draw
/// [`MemWindow::bytes`]: crate::module::MemWindow::bytes
pub fn per_draw_extents(
    blob: &[u8],
    windows: &[crate::module::MemWindow],
    sa_image: &[u8],
    addrs: &[u32],
    target: (u32, u32),
) -> Vec<u32> {
    per_draw_extents_memo(blob, windows, sa_image, addrs, target).0
}

/// [`per_draw_extents`], and whether the answer came from the memo rather than a walk - for a
/// caller's phase table.
pub fn per_draw_extents_memo(
    blob: &[u8],
    windows: &[crate::module::MemWindow],
    sa_image: &[u8],
    addrs: &[u32],
    target: (u32, u32),
) -> (Vec<u32>, bool) {
    use std::collections::HashMap;
    use std::sync::{Arc, Mutex, OnceLock};
    let fixed = || (windows.iter().map(|w| w.bytes).collect::<Vec<u32>>(), false);
    if !crate::module::mem_window_per_draw(windows) {
        return fixed();
    }
    type Memo = Mutex<HashMap<u64, Option<Arc<DrawReach>>>>;
    static MEMO: OnceLock<Memo> = OnceLock::new();
    // >>> AND EACH DRAW'S ANSWER, BY EXACTLY THE INPUTS THE WALK READS. A depth-reconstruct
    // pass draws with the same uniform image, the same window pointers and the same target
    // every frame, and both the capture and the renderer ask - so without this every such draw
    // walked the whole program over ranges twice a frame for an answer it already had. The key
    // is the inputs themselves, compared whole (no fingerprint), so a hit IS the walk's result.
    type Answers = Mutex<HashMap<(u64, ReachInputs), Vec<u32>>>;
    static ANSWERS: OnceLock<Answers> = OnceLock::new();
    // A pair's uniform image can move every draw (a per-object constant), so the table is
    // bounded and shed whole: losing it costs one walk per draw, never an answer.
    const ANSWERS_CAP: usize = 4096;
    let mut key: u64 = 0xcbf2_9ce4_8422_2325;
    for b in blob {
        key ^= u64::from(*b);
        key = key.wrapping_mul(0x0000_0100_0000_01b3);
    }
    let memo = MEMO.get_or_init(Default::default);
    let held = memo.lock().unwrap_or_else(|e| e.into_inner()).get(&key).cloned();
    let reach = match held {
        Some(r) => r,
        None => {
            let built = Program::parse(blob)
                .ok()
                .and_then(|p| DrawReach::build(&p, &crate::usse::decode_shader(&p), windows))
                .map(Arc::new);
            memo.lock().unwrap_or_else(|e| e.into_inner()).insert(key, built.clone());
            built
        }
    };
    let Some(r) = reach else {
        return fixed();
    };
    if !reach_memo_enabled() {
        return (r.extents(sa_image, addrs, target), false);
    }
    let answers = ANSWERS.get_or_init(Default::default);
    let akey = (key, r.inputs(sa_image, addrs, target));
    if let Some(e) = answers.lock().unwrap_or_else(|e| e.into_inner()).get(&akey) {
        return (e.clone(), true);
    }
    let e = r.extents(sa_image, addrs, target);
    let mut g = answers.lock().unwrap_or_else(|e| e.into_inner());
    if g.len() >= ANSWERS_CAP {
        g.clear();
    }
    g.insert(akey, e.clone());
    (e, false)
}

/// `VITASLOP_REACH_MEMO=0` walks the program for every draw, the arm back from the answer memo
/// in [`per_draw_extents`].
fn reach_memo_enabled() -> bool {
    static ON: std::sync::OnceLock<bool> = std::sync::OnceLock::new();
    *ON.get_or_init(|| std::env::var("VITASLOP_REACH_MEMO").map(|v| v.trim() != "0").unwrap_or(true))
}

/// Everything [`DrawReach::extents`] reads from one draw, as an exact memo key: the SA words the
/// module carries (an image shorter than that reads as zero-padded, as the walk reads it), each
/// window's address, and the target extent.
#[derive(Debug, Clone, PartialEq, Eq, Hash)]
struct ReachInputs {
    sa: Vec<u32>,
    addrs: Vec<u32>,
    target: (u32, u32),
}

/// What a draw's memory loads can reach - see the module docs. Built once per program by
/// [`DrawReach::build`], evaluated per draw by [`DrawReach::extents`].
#[derive(Debug, Clone)]
pub struct DrawReach {
    /// Secondary then primary, the order the emitted module runs them in.
    instrs: Vec<Instr>,
    literals: Vec<(u32, u32)>,
    /// How many SA registers the module initialises from the draw's uniform image.
    sa_carried: u32,
    /// The POSITION interpolant's first PA register and register count.
    position: Option<(u32, u32)>,
    /// Each window's pointer register and declared (maximum) extent, in window order.
    windows: Vec<(u32, u32)>,
    /// The emitter's readings, fixed at build so a per-draw evaluation reads no knob.
    halves_src1: bool,
    offset16: bool,
    src1_whole: bool,
}

/// One memory load's reach: the range its first address can take, and its byte extent past it.
#[derive(Debug, Clone, Copy, PartialEq)]
struct Load {
    addr: Option<(u64, u64)>,
    bytes: u64,
}

impl DrawReach {
    /// The analysis for a FRAGMENT program's windows, or `None` when it cannot be walked
    /// straight through: a branch, a write through an index register, a blocked instruction.
    pub fn build(program: &Program, shader: &crate::ir::Shader, windows: &[crate::module::MemWindow]) -> Option<DrawReach> {
        if program.kind != ProgramKind::Fragment || windows.is_empty() {
            return None;
        }
        let instrs = crate::module::with_secondary(program, shader).instrs;
        if instrs.iter().any(|i| {
            i.blocked.is_some()
                || matches!(i.op, Op::Branch { .. })
                || i.dest.as_ref().is_some_and(|d| matches!(d.bank, Bank::Indexed))
        }) {
            return None;
        }
        let position = program
            .interpolants
            .iter()
            .find(|it| it.usage == VaryingUsage::Position && !it.half && it.register_count > 0)
            .map(|it| (u32::from(it.pa_base), u32::from(it.register_count)));
        Some(DrawReach {
            instrs,
            literals: program.literals.clone(),
            sa_carried: program.sa_carried_extent(),
            position,
            windows: windows.iter().map(|w| (w.base_sa, w.bytes)).collect(),
            halves_src1: crate::link::ima32_step_halves_src1(),
            offset16: crate::link::arm_on(crate::link::MEM_OFFSET16_ARM),
            src1_whole: crate::link::arm(crate::link::IMAD_SRC1_WHOLE_ARM).is_some_and(|v| v.trim() == "1"),
        })
    }

    /// Each window's extent in bytes for one draw: `sa_image` is the draw's SA uniform image
    /// (absolute SA registers, little-endian words), `addrs` each window's guest address - the
    /// value its pointer register holds - and `target` the width and height of the surface the
    /// draw rasterises into. Never more than a window's declared extent, never less than 4.
    pub fn extents(&self, sa_image: &[u8], addrs: &[u32], target: (u32, u32)) -> Vec<u32> {
        self.walk(sa_image, addrs, target)
    }

    /// The part of one draw's inputs [`Self::extents`] reads - see [`ReachInputs`].
    fn inputs(&self, sa_image: &[u8], addrs: &[u32], target: (u32, u32)) -> ReachInputs {
        let sa = (0..self.sa_carried.min(LANES as u32))
            .map(|k| {
                let at = k as usize * 4;
                sa_image.get(at..at + 4).map_or(0, |w| u32::from_le_bytes([w[0], w[1], w[2], w[3]]))
            })
            .collect();
        // Only the POSITION interpolant's lanes read the target.
        let target = match self.position {
            Some((_, n)) if n > 1 => target,
            Some((_, 1)) => (target.0, 0),
            _ => (0, 0),
        };
        ReachInputs { sa, addrs: addrs.to_vec(), target }
    }

    fn walk(&self, sa_image: &[u8], addrs: &[u32], target: (u32, u32)) -> Vec<u32> {
        let declared: Vec<u32> = self.windows.iter().map(|&(_, b)| b).collect();
        if addrs.len() != self.windows.len() {
            return declared;
        }
        let mut regs = Regs::new();
        // The module copies the SA registers its uniform binding carries and nothing past them,
        // and the renderer zero-pads an image shorter than the binding.
        for k in 0..self.sa_carried.min(LANES as u32) {
            let at = k as usize * 4;
            let v = sa_image.get(at..at + 4).map_or(0, |w| u32::from_le_bytes([w[0], w[1], w[2], w[3]]));
            regs.set(Bank::SecondaryAttr, k, V::I(v, v));
        }
        for &(reg, v) in &self.literals {
            regs.set(Bank::SecondaryAttr, reg, V::I(v, v));
        }
        for (&(base_sa, _), &a) in self.windows.iter().zip(addrs) {
            regs.set(Bank::SecondaryAttr, base_sa, V::I(a, a));
        }
        if let Some((base, count)) = self.position {
            if count > 0 {
                regs.set(Bank::PrimaryAttr, base, V::F(0.0, target.0 as f32));
            }
            if count > 1 {
                regs.set(Bank::PrimaryAttr, base + 1, V::F(0.0, target.1 as f32));
            }
        }
        let mut loads = Vec::new();
        for instr in &self.instrs {
            self.step(&mut regs, instr, &mut loads);
        }
        let mut need = vec![4u32; self.windows.len()];
        for (i, (&a, &d)) in addrs.iter().zip(&declared).enumerate() {
            let (b, end) = (u64::from(a), u64::from(a) + u64::from(d));
            for l in &loads {
                let reach = match l.addr {
                    None => end,
                    Some((lo, hi)) if lo < end && hi + l.bytes > b => (hi + l.bytes).min(end),
                    Some(_) => continue,
                };
                need[i] = need[i].max((reach - b) as u32);
            }
            // Whole words, as the window is placed; never past the declared extent.
            need[i] = (need[i].div_ceil(4) * 4).min(d).max(4);
        }
        need
    }

    /// Read operand `o`'s channel `c` the way an F32 float instruction does.
    fn fsrc(regs: &Regs, o: &Operand, c: usize) -> Option<(f32, f32)> {
        let sel = *o.swizzle.get(c)?;
        let (lo, hi) = match sel {
            4 => (0.0, 0.0),
            5 => (1.0, 1.0),
            6 => (2.0, 2.0),
            7 => (0.5, 0.5),
            0..=3 => match o.bank {
                Bank::Immediate => (f32::from(o.index), f32::from(o.index)),
                Bank::Constant => {
                    let f = cnst6_channel_value(o.index, sel, false, false)?;
                    (f, f)
                }
                b => float(regs.get(b, u32::from(o.index) + u32::from(sel)))?,
            },
            _ => return None,
        };
        // A denormal input may be FLUSHED to zero by the GPU before it is used.
        let (lo, hi) = (if lo.abs() < f32::MIN_POSITIVE { lo.min(0.0) } else { lo }, if hi.abs() < f32::MIN_POSITIVE { hi.max(0.0) } else { hi });
        let (lo, hi) = if o.abs {
            if lo >= 0.0 {
                (lo, hi)
            } else if hi <= 0.0 {
                (-hi, -lo)
            } else {
                (0.0, (-lo).max(hi))
            }
        } else {
            (lo, hi)
        };
        Some(if o.neg { (-hi, -lo) } else { (lo, hi) })
    }

    /// Operand `o` as the raw integer an integer instruction reads: an immediate's number, or
    /// the register's own lane.
    fn isrc(regs: &Regs, o: &Operand) -> Option<V> {
        if o.abs || o.neg {
            return None;
        }
        match o.bank {
            Bank::Immediate => Some(V::I(u32::from(o.index), u32::from(o.index))),
            Bank::Temp | Bank::PrimaryAttr | Bank::SecondaryAttr | Bank::Output | Bank::Internal => {
                let sel = *o.swizzle.first()?;
                (sel < 4).then(|| regs.get(o.bank, u32::from(o.index) + u32::from(sel)))
            }
            _ => None,
        }
    }

    fn step(&self, regs: &mut Regs, instr: &Instr, loads: &mut Vec<Load>) {
        let Some(dest) = instr.dest.as_ref() else {
            return;
        };
        if let Op::MemLoad { elements, offset_bytes } = instr.op {
            loads.push(self.load(regs, instr, offset_bytes, elements));
            for k in 0..u32::from(elements) {
                regs.set(dest.bank, u32::from(dest.index) + k, V::Top);
            }
            return;
        }
        // Every lane the instruction writes, with its new value - or `None` for "every lane it
        // MAY write is unknown".
        let writes = self.eval(regs, instr, dest);
        let conditional = !matches!(instr.pred, Predicate::Always);
        match writes {
            Some(ws) => {
                for (lane, v) in ws {
                    let v = if conditional { join(regs.get(dest.bank, lane), v) } else { v };
                    regs.set(dest.bank, lane, v);
                }
            }
            None => {
                for lane in Self::may_write(instr, dest) {
                    regs.set(dest.bank, lane, V::Top);
                }
            }
        }
    }

    /// Every lane an instruction the walk does not model may write, by the emitter's own
    /// placement rules: a packed-integer element at `ir::packed_dest_slot`, an F32 channel `c`
    /// at `dest + c`, an F16 one at `dest + c / 2`, a byte one in `dest` itself - over its
    /// write mask, or all four channels for a texture sample (whose emitter stores all four).
    /// A memory load, which writes a run of registers, is handled by the caller.
    fn may_write(instr: &Instr, dest: &Operand) -> Vec<u32> {
        let base = u32::from(dest.index);
        let prec = bank_prec(dest.bank, Prec::of(instr));
        let all = matches!(instr.op, Op::Tex { .. } | Op::TexGather { .. });
        let mut lanes: Vec<u32> = (0..4u32)
            .filter(|&c| all || instr.write_mask[c as usize])
            .map(|c| {
                base + match instr.dest_raw_packed_bits() {
                    Some(bits) => crate::ir::packed_dest_slot(bits, c).0,
                    None => match prec {
                        Prec::F32 => c,
                        Prec::F16 => c >> 1,
                        Prec::Fx8 => 0,
                    },
                }
            })
            .collect();
        if let Op::LocalLoad { .. } = instr.op {
            lanes.push(base);
        }
        lanes
    }

    fn load(&self, regs: &Regs, instr: &Instr, offset_bytes: u32, elements: u8) -> Load {
        let bytes = u64::from(elements) * 4;
        let mut addr = (u64::from(offset_bytes), u64::from(offset_bytes));
        for (k, o) in instr.srcs.iter().enumerate() {
            // The pointer and each register offset are read as the emitter reads them: the
            // register's own lane, an offset narrowed to 16 bits under the default reading.
            let v = match bank_slot(o.bank) {
                Some(_) => regs.get(o.bank, u32::from(o.index)),
                None => return Load { addr: None, bytes },
            };
            let v = if k > 0 && self.offset16 { vfield(v, 0, 16) } else { int(v) };
            if v == (0, u32::MAX) {
                return Load { addr: None, bytes };
            }
            addr = (addr.0 + u64::from(v.0), addr.1 + u64::from(v.1));
        }
        // The emitted address is a u32 sum: one that can wrap can land anywhere.
        if addr.1 + bytes > u64::from(u32::MAX) {
            return Load { addr: None, bytes };
        }
        Load { addr: Some(addr), bytes }
    }

    /// The lanes `instr` writes and their new values, or `None` when it is not modelled.
    fn eval(&self, regs: &Regs, instr: &Instr, dest: &Operand) -> Option<Vec<(u32, V)>> {
        bank_slot(dest.bank)?;
        let base = u32::from(dest.index);
        let channels = || (0..4).filter(|&c| instr.write_mask[c]);
        let float_op = !instr.half_precision
            && Prec::of(instr) == Prec::F32
            && Prec::src_of(instr) == Prec::F32
            && bank_prec(dest.bank, Prec::F32) == Prec::F32;
        match instr.op {
            Op::Mov | Op::Add | Op::Mul | Op::Mad | Op::Min | Op::Max if float_op => {
                let mut out = Vec::new();
                for c in channels() {
                    let s = |n: usize| instr.srcs.get(n).and_then(|o| Self::fsrc(regs, o, c));
                    let v = match instr.op {
                        // A move goes through the FLOAT view (`emit_unary`), so it carries a
                        // float range and nothing else: an integer pattern's bits are not
                        // promised to survive it (a denormal can flush, a NaN canonicalise).
                        Op::Mov => s(0).map_or(V::Top, |(l, h)| V::F(l, h)),
                        Op::Add => match (s(0), s(1)) {
                            (Some(a), Some(b)) => widen(f64::from(a.0) + f64::from(b.0), f64::from(a.1) + f64::from(b.1)),
                            _ => V::Top,
                        },
                        Op::Mul => match (s(0), s(1)) {
                            (Some(a), Some(b)) => {
                                let (l, h) = fmul(a, b);
                                widen(l, h)
                            }
                            _ => V::Top,
                        },
                        Op::Mad => match (s(0), s(1), s(2)) {
                            (Some(a), Some(b), Some(d)) => {
                                let (l, h) = fmul(a, b);
                                widen(l + f64::from(d.0), h + f64::from(d.1))
                            }
                            _ => V::Top,
                        },
                        Op::Min | Op::Max => match (s(0), s(1)) {
                            (Some(a), Some(b)) if instr.op == Op::Min => V::F(a.0.min(b.0), a.1.min(b.1)),
                            (Some(a), Some(b)) => V::F(a.0.max(b.0), a.1.max(b.1)),
                            _ => V::Top,
                        },
                        _ => V::Top,
                    };
                    out.push((base + c as u32, v));
                }
                Some(out)
            }
            Op::PackToInt { bits, signed, src_half: false, norm: false } if !instr.half_precision => {
                let bits = u32::from(bits);
                if bits != 8 && bits != 16 && bits != 32 {
                    return None;
                }
                let mask = if bits >= 32 { u32::MAX } else { (1u32 << bits) - 1 };
                let mut out: Vec<(u32, V)> = Vec::new();
                for c in channels() {
                    let e = match instr.srcs.first().and_then(|o| Self::fsrc(regs, o, c)) {
                        Some((lo, hi)) => {
                            let (tl, th) = (f64::from(lo).trunc(), f64::from(hi).trunc());
                            let (tl, th) = if signed {
                                (tl.max(-2_147_483_000.0), th.min(2_147_483_000.0))
                            } else {
                                (tl.max(0.0), th.min(4_294_967_000.0))
                            };
                            if tl >= 0.0 && th <= f64::from(mask) {
                                (tl as u32, th as u32)
                            } else {
                                (0, mask)
                            }
                        }
                        None => (0, mask),
                    };
                    Self::place(regs, &mut out, dest.bank, base, bits, c as u32, e);
                }
                Some(out)
            }
            Op::PackIntCopy { bits, dest_bits, signed } => {
                let (bits, dest_bits) = (u32::from(bits), u32::from(dest_bits));
                if (bits != 16 && bits != 8) || (dest_bits != 16 && dest_bits != 8) {
                    return None;
                }
                let s1 = instr.srcs.first()?;
                bank_slot(s1.bank)?;
                let per_reg = 32 / bits;
                let dmask = (1u32 << dest_bits) - 1;
                let mut out: Vec<(u32, V)> = Vec::new();
                for c in channels() {
                    let sel = u32::from(s1.swizzle[c]);
                    let word = regs.get(s1.bank, u32::from(s1.index) + sel / per_reg);
                    let e = vfield(word, (sel % per_reg) * bits, bits);
                    // A widening signed copy carries the sign up: only a non-negative element is
                    // the same number unsigned.
                    let e = if signed && dest_bits > bits && e.1 >> (bits - 1) != 0 { (0, dmask) } else { e };
                    let e = (e.0.min(dmask), e.1.min(dmask));
                    let e = if e.1 > dmask { (0, dmask) } else { e };
                    Self::place(regs, &mut out, dest.bank, base, dest_bits, c as u32, e);
                }
                Some(out)
            }
            Op::Bitwise { kind, imm, lane_bits } if instr.write_mask == [true, false, false, false] => {
                let lane_bits = u32::from(lane_bits);
                let lmask = if lane_bits >= 32 { u32::MAX } else { (1u32 << lane_bits) - 1 };
                let av = Self::isrc(regs, instr.srcs.first()?)?;
                let a = vfield(av, 0, lane_bits.min(32));
                let b = match imm {
                    Some(v) => (v & lmask, v & lmask),
                    None => vfield(Self::isrc(regs, instr.srcs.get(1)?)?, 0, lane_bits.min(32)),
                };
                let shift_mask = lane_bits.saturating_sub(1);
                let r = match kind {
                    BitwiseKind::And if b.0 == b.1 && (b.0 == u32::MAX || (b.0 & b.0.wrapping_add(1)) == 0) => {
                        if a.1 <= b.0 { a } else { vfield(av, 0, b.0.count_ones()) }
                    }
                    BitwiseKind::And => (0, a.1.min(b.1)),
                    BitwiseKind::Or if b == (0, 0) => a,
                    BitwiseKind::Or => (a.0.max(b.0), ones_above(a.1.max(b.1))),
                    BitwiseKind::Xor => (0, ones_above(a.1.max(b.1))),
                    BitwiseKind::Shl if b.0 == b.1 => {
                        let s = b.0 & shift_mask;
                        if (u64::from(a.1) << s) <= u64::from(lmask) { (a.0 << s, a.1 << s) } else { (0, lmask) }
                    }
                    BitwiseKind::Shr if b.0 == b.1 => {
                        let s = b.0 & shift_mask;
                        (a.0 >> s, a.1 >> s)
                    }
                    BitwiseKind::Asr if b.0 == b.1 && lane_bits >= 1 && a.1 >> (lane_bits - 1) == 0 => {
                        let s = b.0 & shift_mask;
                        (a.0 >> s, a.1 >> s)
                    }
                    _ => (0, lmask),
                };
                Some(vec![(base, V::I(r.0 & lmask, r.1.min(lmask)))])
            }
            Op::IntMad { signed, bits: 32, src0_high, src1_high } => {
                let a = Self::isrc(regs, instr.srcs.first()?)?;
                let b = Self::isrc(regs, instr.srcs.get(1)?)?;
                let d = Self::isrc(regs, instr.srcs.get(2)?)?;
                let half = |v: V, high: bool| vfield(v, if high { 16 } else { 0 }, 16);
                let b_whole = !src1_high && self.src1_whole;
                let (a, d) = (half(a, src0_high), int(d));
                let b = if b_whole { int(b) } else { half(b, src1_high) };
                // A signed multiply agrees with the unsigned one while every operand is
                // non-negative in its signed reading and the sum stays below the sign bit.
                let nonneg = a.1 < 0x8000 && b.1 < if b_whole { 0x8000_0000 } else { 0x8000 } && d.1 < 0x8000_0000;
                let r = match mad(a, b, d) {
                    V::I(_, hi) if signed && (!nonneg || hi >= 0x8000_0000) => V::Top,
                    r => r,
                };
                Some(vec![(base, r)])
            }
            Op::IntMadStep { high_half, .. } => {
                let a = Self::isrc(regs, instr.srcs.first()?)?;
                let b = Self::isrc(regs, instr.srcs.get(1)?)?;
                let d = Self::isrc(regs, instr.srcs.get(2)?)?;
                // Signedness does not change the low 32 bits of a product, and only a result
                // that does not wrap is kept, so one evaluation serves both.
                let (w, h) = if self.halves_src1 { (a, b) } else { (b, a) };
                let (w, d) = (int(w), int(d));
                let r = if high_half {
                    match mad(w, vfield(h, 16, 16), (0, 0)) {
                        V::I(lo, hi) if u64::from(hi) << 16 <= u64::from(u32::MAX) => {
                            mad((lo, hi), (1 << 16, 1 << 16), d)
                        }
                        _ => V::Top,
                    }
                } else {
                    mad(w, vfield(h, 0, 16), d)
                };
                Some(vec![(base, r)])
            }
            _ => None,
        }
    }

    /// Store element `e` of channel `c` the way a packed-integer write places it: a `bits`-wide
    /// field of a register for 8 and 16 bits (`ir::packed_dest_slot`), a whole lane for 32.
    fn place(regs: &Regs, out: &mut Vec<(u32, V)>, dest: Bank, base: u32, bits: u32, c: u32, e: (u32, u32)) {
        if bits >= 32 {
            out.push((base + c, V::I(e.0, e.1)));
            return;
        }
        let (reg_off, shift) = crate::ir::packed_dest_slot(bits, c);
        let lane = base + reg_off;
        // The rest of the register keeps what it held - which is this same instruction's
        // earlier channel when two channels share the register.
        let old = out.iter().rev().find(|(l, _)| *l == lane).map_or_else(|| regs.get(dest, lane), |&(_, v)| v);
        out.push((lane, vinsert(old, shift, bits, e)));
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::gxpwrite::{self, ContainerSpec, InterpolantSpec, ParamSpec, ProgramSpec};
    use crate::usse::asm::{self, Dest, IntSrc, PackFmt};

    /// The `attribute_info` usage nibble of the POSITION interpolant.
    const POSITION: u32 = 0xd;
    const DECLARED: u32 = 17_825_792;

    /// The depth-reconstruct reader's address arithmetic, in the program's own instructions:
    /// `x, y = int16(position)`, `index = stride * y.lo16 + x` (an IMA32 step with the stride
    /// whole in `src0`), then `ptr + index * 32` (a lone step, `#32` in `src1`) and an 8-word
    /// load. `y` reaches the step through a 16-bit copy that leaves its register's HIGH half
    /// holding the float position - which only the `src1` reading ignores.
    fn reader(extra_tail: Vec<u64>) -> Vec<u8> {
        let (pa, sa, t) = (Bank::PrimaryAttr, Bank::SecondaryAttr, Bank::Temp);
        let mut code = vec![
            asm::pack(Dest::new(pa, 0), PackFmt::S16, pa, 0, PackFmt::F32, [0, 1, 0, 0], [true, true, false, false], false).unwrap(),
            asm::pack(Dest::new(pa, 1), PackFmt::U16, pa, 0, PackFmt::U16, [1, 0, 0, 0], [true, false, false, false], false).unwrap(),
            asm::bitwise(BitwiseKind::And, false, Dest::new(pa, 0), (pa, 0), IntSrc::Imm(0xffff)).unwrap(),
            asm::imad_step(false, false, Dest::new(pa, 1), (sa, 1), IntSrc::Reg(pa, 1), IntSrc::Reg(pa, 0)).unwrap(),
            asm::imad_step(false, false, Dest::new(t, 0), (pa, 1), IntSrc::Imm(32), IntSrc::Reg(sa, 11)).unwrap(),
            asm::ldmem(Dest::new(t, 4), 8, (t, 0), IntSrc::Imm(0), IntSrc::Imm(0)).unwrap(),
        ];
        code.extend(extra_tail);
        gxpwrite::write(
            &ProgramSpec::fragment(code)
                .with_parameters(vec![ParamSpec::uniform("inVU.mPixelStride", 0, 1), ParamSpec::uniform_buffer("", 0, DECLARED)])
                .with_default_uniform_regs(4)
                .with_containers(vec![
                    ContainerSpec { index: 14, base_sa: 0, size_regs: 4 },
                    ContainerSpec { index: 19, base_sa: 10, size_regs: 4 },
                ])
                .with_ub_bindings(vec![(0, 1)])
                .with_interpolants(vec![InterpolantSpec { usage: POSITION, registers: 2, half: false }])
                .with_registers(2, 14, 12),
        )
    }

    fn windows(blob: &[u8]) -> Vec<crate::module::MemWindow> {
        let p = Program::parse(blob).unwrap();
        crate::module::resolve_mem_windows(&p, &crate::usse::decode_shader(&p)).unwrap()
    }

    /// `sa[1] = 960` in the draw's uniform image.
    fn image() -> Vec<u8> {
        [0u32, 960, 0, 0].iter().flat_map(|w| w.to_le_bytes()).collect()
    }

    /// The reader's window is unbounded by the program and so sized PER DRAW: a 240x136 target
    /// reaches `(960 * 136 + 240) * 32` plus the 32-byte load and not a byte more, a 960x544 one
    /// proportionally further, and both stay inside the declared 17 MiB.
    #[test]
    fn an_index_bounded_by_the_target_and_a_uniform_is_sized_per_draw() {
        let blob = reader(vec![]);
        let w = windows(&blob);
        assert_eq!(w.len(), 1);
        assert!(w[0].per_draw && w[0].bytes == DECLARED, "{w:?}");
        assert!(crate::module::mem_window_storage(&w));
        let at = 0x9000_0000u32;
        let small = per_draw_extents(&blob, &w, &image(), &[at], (240, 136));
        assert_eq!(small, vec![(960 * 136 + 240) * 32 + 32]);
        let big = per_draw_extents(&blob, &w, &image(), &[at], (960, 544));
        assert_eq!(big, vec![(960 * 544 + 960) * 32 + 32]);
        // A stride the image does not carry is "anything": the declared extent stands.
        let p = Program::parse(&blob).unwrap();
        let r = DrawReach::build(&p, &crate::usse::decode_shader(&p), &w).unwrap();
        assert_eq!(r.extents(&[], &[at], (240, 136)), vec![240 * 32 + 32], "an absent image reads zero");
        let huge: Vec<u8> = [0u32, 0x7fff_ffff, 0, 0].iter().flat_map(|w| w.to_le_bytes()).collect();
        assert_eq!(r.extents(&huge, &[at], (240, 136)), vec![DECLARED], "a product that can wrap reaches everything");
    }

    /// A program that BRANCHES is not walked straight through, so its window keeps the declared
    /// extent rather than a bound the walk could not establish.
    #[test]
    fn a_branching_program_is_not_sized_per_draw() {
        let blob = reader(vec![asm::br(Predicate::IfP(0), 1).unwrap()]);
        let w = windows(&blob);
        assert!(!w[0].per_draw && w[0].bytes == DECLARED, "{w:?}");
    }

    /// The per-draw declaration and helper form a valid module: a runtime-sized storage array,
    /// each window's extent and placement read from its header.
    #[test]
    fn the_per_draw_helper_validates() {
        let w = windows(&reader(vec![]));
        let decl = crate::module::mem_window_decl(1, 1, "gxp_fmem", &w);
        assert_eq!(decl, "@group(1) @binding(1) var<storage, read> gxp_fmem: array<vec4<u32>>;\n");
        let src = format!(
            "{decl}{}@fragment fn main() -> @location(0) vec4<u32> {{ return gxp_fmem_quad(gxp_fmem_word(16u)); }}\n",
            crate::module::mem_window_helper_named(&w, "gxp_fmem")
        );
        assert!(src.contains("if (w0 < h0.y)") && src.contains("let g0 = h0.z + w0;"), "{src}");
        let parsed = naga::front::wgsl::parse_str(&src).unwrap_or_else(|e| panic!("{e}\n{src}"));
        naga::valid::Validator::new(naga::valid::ValidationFlags::all(), naga::valid::Capabilities::all())
            .validate(&parsed)
            .unwrap_or_else(|e| panic!("{e:?}\n{src}"));
    }

    /// The range arithmetic the walk is built on.
    #[test]
    fn ranges_round_outward_and_refuse_a_wrap() {
        let V::F(lo, hi) = widen(1.0, 2.0) else { panic!() };
        assert!(lo < 1.0 && hi > 2.0 && lo > 0.99 && hi < 2.01);
        assert_eq!(widen(f64::NAN, 1.0), V::Top);
        assert_eq!(mad((0, 0xffff), (0, 0x1_0000), (0, 0xffff)), V::I(0, 0xffff_ffff));
        assert_eq!(mad((0, 0xffff), (0, 0x1_0001), (0, 0xffff)), V::Top);
        assert_eq!(field((0x0001_0005, 0x0001_0009), 0, 16), (5, 9));
        assert_eq!(field((0x0001_0005, 0x0002_0009), 0, 16), (0, 0xffff));
        assert_eq!(field((0x0001_0005, 0x0002_0009), 16, 16), (1, 2));
        assert_eq!(insert((0x4300_0000, 0x4310_0000), 0, 16, (3, 7)), (0x4300_0003, 0x4310_0007));
        assert_eq!(ones_above(5), 7);
    }
}
