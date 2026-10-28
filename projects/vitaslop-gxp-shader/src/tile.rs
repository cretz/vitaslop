//! The TILE special registers - `GLOBAL[43]` (`TILEXY`), `GLOBAL[23]` (`QUADRANTX`) and
//! `GLOBAL[24]` (`TILEY`) - and the proof that lets a program reading them be emitted.
//!
//! # What is established, and what is not
//! The vendor's special-bank definitions NAME these registers and say nothing of their
//! contents. What a program settles is narrower: the one fragment program in any corpus that
//! reads them (a shooter's edge-mask pass) forms
//!
//! ```text
//!   X = 16 * TILEXY.byte0 + QUADRANTX
//!   Y = 16 * TILEXY.byte1 + TILEY
//! ```
//!
//! and stores one word per fragment at `(Y * stride + X) * 4` into a buffer of exactly
//! `stride * 136 * 4` bytes, drawn over a 240x136 quad. The only reading under which every
//! fragment gets its own in-range word is `X` = the fragment's pixel x and `Y` = its pixel y;
//! quad units collide four to one and a 32-pixel tile with a 0..31 remainder collides across
//! tiles. So the SUMS are the pixel coordinates. How the hardware splits each sum between the
//! registers (16-pixel units with a 0..15 remainder, or 32-pixel tiles counted in 16-pixel units
//! with a 0..31 remainder, or anything else) is NOT established, and a program reading one
//! register alone would see the difference.
//!
//! # Why emitting a split is nevertheless exact - for the programs [`tile_sums_only`] admits
//! The emitter supplies ONE split - `TILEXY = (x >> 4) | (y >> 4) << 8`, `QUADRANTX = x & 15`,
//! `TILEY = y & 15` - whose sums are the pixel coordinates. That is exact for a program whose
//! every observable result depends on the registers ONLY THROUGH the two sums, because every
//! split with those sums then gives the same results. [`tile_sums_only`] is the check, and it is
//! a proof, not a heuristic:
//!
//! * Each 32-bit lane is tracked as either SPLIT-INDEPENDENT (`Free`), or as a linear form
//!   `r0*TXlo + r1*TXhi + r2*Q + r3*TY + F` modulo 2^32 with `F` split-independent (`Partial`),
//!   where `TXlo`/`TXhi` are `TILEXY`'s low two bytes and `Q`/`TY` the other two registers.
//! * The form is COMPLETE when `r0 = 16*r2` and `r1 = 16*r3`: it is then `r2*X + r3*Y + F`,
//!   identical under every split, and becomes `Free`. Arithmetic modulo 2^32 is a ring
//!   homomorphism, so wrapping never breaks the identity.
//! * Only operations that are LINEAR in the forms are followed - the IMA32 step multiply-add by
//!   a constant, the byte mask that extracts `TXlo`, the byte copy that extracts `TXhi` - and a
//!   16-bit truncation only where the value is provably below 2^16 for EVERY byte value of the
//!   atoms (so whatever bytes the hardware holds, the truncation is the identity).
//! * Any other operation that reads an incomplete form, any write that would mix one into a
//!   lane partially, any read of `TILEXY`'s upper bytes, a branch, or an incomplete form left in
//!   an output register at the end refuses the program, naming why.
//!
//! A program refused here keeps the emitter's existing hard failure (`UnmodeledGlobal`).

use crate::ir::{Bank, BitwiseKind, Instr, Op, Operand, Predicate, Shader};

/// `GLOBAL[43]`, `TILEXY`.
pub const TILEXY: u8 = 43;
/// `GLOBAL[23]`, `QUADRANTX`.
pub const QUADRANTX: u8 = 23;
/// `GLOBAL[24]`, `TILEY`.
pub const TILEY: u8 = 24;

/// Whether `index` is one of the three tile registers.
pub fn is_tile_register(index: u8) -> bool {
    matches!(index, TILEXY | QUADRANTX | TILEY)
}

/// Whether `shader` reads any tile register.
pub fn reads_tile_registers(shader: &Shader) -> bool {
    shader
        .instrs
        .iter()
        .flat_map(|i| i.srcs.iter())
        .any(|o| o.bank == Bank::Global && is_tile_register(o.index))
}

/// The value the emitted split gives each tile register for the fragment at pixel `(x, y)` -
/// the ONE place the split is written down for the Rust side (the reference interpreter and the
/// tests); the WGSL side is [`crate::wgsl::TILE_REGISTERS_DECL`]. Only meaningful for a program
/// [`tile_sums_only`] admitted.
pub fn tile_register_value(index: u8, x: u32, y: u32) -> Option<u32> {
    Some(match index {
        TILEXY => ((x >> 4) & 0xff) | (((y >> 4) & 0xff) << 8),
        QUADRANTX => x & 15,
        TILEY => y & 15,
        _ => return None,
    })
}

/// Open the emitter's tile-register scope (`wgsl::tile_sums_scope`) for one FRAGMENT program's
/// emission, with [`tile_sums_only`]'s verdict over its secondary and primary streams together.
/// `shader` is the primary stream exactly as it will be emitted. Every emission of a fragment
/// program's streams - `recompile_fragment`'s and the linker's - goes through this, so the two
/// cannot disagree about whether the registers are admitted.
pub fn fragment_tile_scope(program: &crate::container::Program, shader: &Shader) -> crate::wgsl::TileSumsScope {
    let verdict = if program.kind == crate::container::ProgramKind::Fragment {
        tile_sums_only(&crate::module::with_secondary(program, shader), &program.literals)
    } else {
        Ok(false)
    };
    crate::wgsl::tile_sums_scope(verdict)
}

/// The atoms a [`T::Partial`] form is over.
const TXLO: usize = 0;
const TXHI: usize = 1;
const Q: usize = 2;
const TY: usize = 3;

/// One 32-bit lane's standing.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
enum T {
    /// Independent of the split; the exact value when it is a known constant.
    Free(Option<u32>),
    /// `r . atoms + k` (mod 2^32), `k` split-independent and known when `Some`.
    Partial { r: [u32; 4], k: Option<u32> },
    /// The low byte is atom `atom` (a byte of `TILEXY`); the other 24 bits are split-independent.
    LowByte(usize),
}

impl T {
    fn atom(a: usize) -> T {
        let mut r = [0u32; 4];
        r[a] = 1;
        T::Partial { r, k: Some(0) }
    }

    /// A complete form is split-independent.
    fn normal(self) -> T {
        match self {
            T::Partial { r, .. } if r[TXLO] == r[Q].wrapping_mul(16) && r[TXHI] == r[TY].wrapping_mul(16) => {
                T::Free(None)
            }
            t => t,
        }
    }

    /// The value's range for EVERY byte value of the two `TILEXY` atoms, when it is a form over
    /// those atoms alone with a known constant part and no wrap - `None` otherwise.
    fn byte_range(self) -> Option<(u64, u64)> {
        match self {
            T::Free(Some(v)) => Some((u64::from(v), u64::from(v))),
            T::Partial { r, k: Some(k) } if r[Q] == 0 && r[TY] == 0 => {
                let hi = u64::from(k) + 255 * u64::from(r[TXLO]) + 255 * u64::from(r[TXHI]);
                (hi <= u64::from(u32::MAX)).then_some((u64::from(k), hi))
            }
            _ => None,
        }
    }
}

fn add(a: T, b: T) -> Result<T, &'static str> {
    Ok(match (a, b) {
        (T::Free(x), T::Free(y)) => T::Free(x.zip(y).map(|(x, y)| x.wrapping_add(y))),
        (T::Partial { r, k }, T::Free(y)) | (T::Free(y), T::Partial { r, k }) => {
            T::Partial { r, k: k.zip(y).map(|(k, y)| k.wrapping_add(y)) }
        }
        (T::Partial { r: ra, k: ka }, T::Partial { r: rb, k: kb }) => T::Partial {
            r: std::array::from_fn(|i| ra[i].wrapping_add(rb[i])),
            k: ka.zip(kb).map(|(a, b)| a.wrapping_add(b)),
        },
        _ => return Err("a byte of TILEXY is added before it is masked out of its register"),
    }
    .normal())
}

fn mul(a: T, b: T) -> Result<T, &'static str> {
    Ok(match (a, b) {
        (T::Free(x), T::Free(y)) => T::Free(x.zip(y).map(|(x, y)| x.wrapping_mul(y))),
        (T::Partial { r, k }, T::Free(Some(c))) | (T::Free(Some(c)), T::Partial { r, k }) => {
            T::Partial { r: r.map(|v| v.wrapping_mul(c)), k: k.map(|k| k.wrapping_mul(c)) }
        }
        (T::Partial { .. }, T::Free(None)) | (T::Free(None), T::Partial { .. }) => {
            return Err("an incomplete tile-register sum is multiplied by a value that is not a known constant");
        }
        (T::Partial { .. }, T::Partial { .. }) => return Err("two tile-register terms are multiplied together"),
        _ => return Err("a byte of TILEXY is multiplied before it is masked out of its register"),
    }
    .normal())
}

/// The low (`high = false`) or high 16 bits of a lane, as the IMA32 step reads its halved input.
fn half(t: T, high: bool) -> Result<T, &'static str> {
    match t {
        T::Free(v) => Ok(T::Free(v.map(|v| if high { v >> 16 } else { v & 0xffff }))),
        t => match t.byte_range() {
            Some((_, hi)) if hi <= 0xffff => Ok(if high { T::Free(Some(0)) } else { t }),
            _ => Err("a 16-bit half of an incomplete tile-register sum is taken where it may not be the whole value"),
        },
    }
}

/// The register state the walk tracks, one [`T`] per lane of each ordinary bank.
struct Lanes {
    banks: [Vec<T>; 5],
}

const LANES: usize = 512;

fn slot(bank: Bank) -> Option<usize> {
    Some(match bank {
        Bank::Temp => 0,
        Bank::PrimaryAttr => 1,
        Bank::SecondaryAttr => 2,
        Bank::Output => 3,
        Bank::Internal => 4,
        _ => return None,
    })
}

impl Lanes {
    fn get(&self, bank: Bank, lane: u32) -> T {
        slot(bank).and_then(|b| self.banks[b].get(lane as usize).copied()).unwrap_or(T::Free(None))
    }
    fn set(&mut self, bank: Bank, lane: u32, t: T) {
        if let Some(s) = slot(bank).and_then(|b| self.banks[b].get_mut(lane as usize)) {
            *s = t;
        }
    }
    fn any_tracked(&self) -> bool {
        self.banks.iter().any(|b| b.iter().any(|t| !matches!(t, T::Free(_))))
    }
}

/// Whether `shader` - the SECONDARY stream followed by the primary, as
/// [`crate::module::with_secondary`] builds it - reads the tile registers, and if so whether it
/// depends on them ONLY through the two pixel-coordinate sums (see the module docs).
///
/// `Ok(false)`: no tile register is read. `Ok(true)`: they are read and the proof holds, so the
/// emitter's split is exact. `Err`: they are read and the proof does not hold - the reason.
/// `literals` are the program's container literals, the only SA values known at emit time.
pub fn tile_sums_only(shader: &Shader, literals: &[(u32, u32)]) -> Result<bool, &'static str> {
    if !reads_tile_registers(shader) {
        return Ok(false);
    }
    if shader.instrs.iter().any(|i| matches!(i.op, Op::Branch { .. })) {
        return Err("the program branches, and the tile-register proof walks straight-line code only");
    }
    let mut lanes = Lanes { banks: std::array::from_fn(|_| vec![T::Free(None); LANES]) };
    for &(reg, v) in literals {
        lanes.set(Bank::SecondaryAttr, reg, T::Free(Some(v)));
    }
    for instr in &shader.instrs {
        step(&mut lanes, instr)?;
    }
    // The end of the program reads every output: an incomplete form left where the colour may
    // be read is an observable that depends on the split.
    for bank in [Bank::Output, Bank::PrimaryAttr] {
        if (0..LANES as u32).any(|l| !matches!(lanes.get(bank, l), T::Free(_))) {
            return Err("an incomplete tile-register sum is left in an output register");
        }
    }
    Ok(true)
}

/// A raw-lane operand's value: an immediate, a register lane (`index` exactly, as the integer
/// emitters read it), or a tile register.
fn raw(lanes: &Lanes, o: &Operand) -> Result<T, &'static str> {
    if o.abs || o.neg {
        return Err("a tile-register sum is read through a modifier");
    }
    Ok(match o.bank {
        Bank::Immediate => T::Free(Some(u32::from(o.index))),
        Bank::Global if o.index == QUADRANTX => T::atom(Q),
        Bank::Global if o.index == TILEY => T::atom(TY),
        Bank::Global if o.index == TILEXY => return Err("TILEXY is read whole, not one of its low two bytes"),
        Bank::Global | Bank::Constant => T::Free(None),
        Bank::Indexed => {
            if lanes.any_tracked() {
                return Err("an indexed read while a tile-register term is live");
            }
            T::Free(None)
        }
        b => lanes.get(b, u32::from(o.index)),
    })
}

fn step(lanes: &mut Lanes, instr: &Instr) -> Result<(), &'static str> {
    if let Some(t) = modelled(lanes, instr)? {
        let d = instr.dest.as_ref().ok_or("a modelled operation without a destination")?;
        let lane = u32::from(d.index);
        if !matches!(instr.pred, Predicate::Always) && (!matches!(t, T::Free(_)) || !matches!(lanes.get(d.bank, lane), T::Free(_))) {
            return Err("a predicated write involves a tile-register term");
        }
        let t = if matches!(instr.pred, Predicate::Always) { t } else { T::Free(None) };
        lanes.set(d.bank, lane, t);
        return Ok(());
    }
    // Not modelled: it must neither read a tile register nor any lane holding a term, and it
    // must not write into a lane holding one (a partial write would mix it).
    if instr.srcs.iter().any(|o| o.bank == Bank::Global && is_tile_register(o.index)) {
        return Err("a tile register is read by an operation the sum proof does not model");
    }
    for (bank, lane) in read_lanes(instr) {
        if bank == Bank::Indexed {
            if lanes.any_tracked() {
                return Err("an indexed read while a tile-register term is live");
            }
            continue;
        }
        if !matches!(lanes.get(bank, lane), T::Free(_)) {
            return Err("an incomplete tile-register sum is read by an operation the sum proof does not model");
        }
    }
    for (bank, lane) in write_lanes(instr) {
        if !matches!(lanes.get(bank, lane), T::Free(_)) {
            return Err("an operation the sum proof does not model overwrites an incomplete tile-register sum");
        }
        lanes.set(bank, lane, T::Free(None));
    }
    Ok(())
}

/// The result of an operation the proof follows, or `None` when it is not one of them (or does
/// not touch a tracked value, in which case the generic path is exact).
fn modelled(lanes: &Lanes, instr: &Instr) -> Result<Option<T>, &'static str> {
    let Some(dest) = instr.dest.as_ref() else { return Ok(None) };
    if slot(dest.bank).is_none() {
        return Ok(None);
    }
    let src = |n: usize| instr.srcs.get(n).ok_or("a modelled operation is missing an operand");
    match instr.op {
        Op::IntMadStep { high_half, .. } => {
            let (a, b, c) = (raw(lanes, src(0)?)?, raw(lanes, src(1)?)?, raw(lanes, src(2)?)?);
            // The emitter's own reading of which input is halved (`wgsl::emit_int_mad_step`).
            let (whole, halved) = if crate::link::ima32_step_halves_src1() { (a, b) } else { (b, a) };
            let p = mul(whole, half(halved, high_half)?)?;
            let p = if high_half { mul(p, T::Free(Some(1 << 16)))? } else { p };
            Ok(Some(add(p, c)?))
        }
        Op::Bitwise { kind, imm, lane_bits } if instr.write_mask == [true, false, false, false] => {
            let s = src(0)?;
            // `TILEXY & 0xff` is the low byte - the one whole-register read admitted.
            if s.bank == Bank::Global && s.index == TILEXY {
                return match (kind, imm, lane_bits) {
                    (BitwiseKind::And, Some(0xff), 32) if !s.abs && !s.neg => Ok(Some(T::atom(TXLO))),
                    _ => Err("TILEXY is read by a bitwise operation other than the low-byte mask"),
                };
            }
            let a = raw(lanes, s)?;
            let b = match imm {
                Some(v) => T::Free(Some(v)),
                None => raw(lanes, src(1)?)?,
            };
            match (a, b) {
                (T::Free(_), T::Free(_)) => Ok(Some(T::Free(None))),
                (T::LowByte(atom), T::Free(Some(0xff))) if kind == BitwiseKind::And && lane_bits == 32 => Ok(Some(T::atom(atom))),
                (t, T::Free(Some(m))) if kind == BitwiseKind::And && lane_bits == 32 && m.wrapping_add(1).is_power_of_two() => {
                    match t.byte_range() {
                        Some((_, hi)) if hi <= u64::from(m) => Ok(Some(t)),
                        _ => Err("an incomplete tile-register sum is masked where the mask may change it"),
                    }
                }
                _ => Err("an incomplete tile-register sum is read by a bitwise operation"),
            }
        }
        // A byte of TILEXY copied into byte 0 of a lane whose other bits are split-independent -
        // the one shape the proof follows. A copy from an ordinary register takes the generic
        // path, which checks its reads and writes.
        Op::PackIntCopy { bits: 8, dest_bits: 8, signed: false }
            if instr.srcs.first().is_some_and(|s| s.bank == Bank::Global && s.index == TILEXY) =>
        {
            let s = src(0)?;
            if instr.write_mask != [true, false, false, false] || s.abs || s.neg {
                return Err("a TILEXY byte is copied into a register shape the sum proof does not follow");
            }
            let atom = match s.swizzle[0] {
                0 => TXLO,
                1 => TXHI,
                _ => return Err("an upper byte of TILEXY is read; its contents are not established"),
            };
            if !matches!(lanes.get(dest.bank, u32::from(dest.index)), T::Free(_)) {
                return Err("a TILEXY byte is copied over an incomplete tile-register sum");
            }
            Ok(Some(T::LowByte(atom)))
        }
        _ => Ok(None),
    }
}

/// Every register lane an operation may READ, conservatively: each source's lanes under any of
/// the three placements (`index + sel` for F32, `index + sel / 2` for F16, `index` for a byte or
/// raw-lane read) for each channel it reads. An indexed operand is reported as `Bank::Indexed`.
fn read_lanes(instr: &Instr) -> Vec<(Bank, u32)> {
    let chans = instr.read_channels();
    let mut out = Vec::new();
    for o in &instr.srcs {
        if o.bank == Bank::Indexed {
            out.push((Bank::Indexed, 0));
            continue;
        }
        if slot(o.bank).is_none() {
            continue;
        }
        let base = u32::from(o.index);
        out.push((o.bank, base));
        for (c, &on) in chans.iter().enumerate() {
            let sel = u32::from(o.swizzle[c]);
            if on && sel <= 3 {
                out.push((o.bank, base + sel));
                out.push((o.bank, base + (sel >> 1)));
            }
        }
        // A multi-register read (a texture coordinate, a dot product's second vector) stays
        // within four registers of its base.
        if matches!(instr.op, Op::Tex { .. } | Op::TexGather { .. } | Op::Dot { .. }) {
            out.extend((0..8).map(|k| (o.bank, base + k)));
        }
    }
    out
}

/// Every register lane an operation may WRITE, by the emitter's own placement rules (the same
/// rules `reach::DrawReach` sizes windows by): a packed-integer element at
/// `ir::packed_dest_slot`, an F32 channel `c` at `dest + c`, an F16 one at `dest + c / 2`, a byte
/// one in `dest` itself - over the write mask, or all four channels for a sample; a memory load
/// its `elements` registers, a gather its six.
fn write_lanes(instr: &Instr) -> Vec<(Bank, u32)> {
    use crate::wgsl::{bank_prec, Prec};
    let Some(d) = instr.dest.as_ref() else { return Vec::new() };
    if slot(d.bank).is_none() {
        return Vec::new();
    }
    let base = u32::from(d.index);
    match instr.op {
        Op::MemLoad { elements, .. } => return (0..u32::from(elements)).map(|k| (d.bank, base + k)).collect(),
        Op::LocalLoad { .. } => return vec![(d.bank, base)],
        Op::TexGather { .. } => return (0..6).map(|k| (d.bank, base + k)).collect(),
        _ => {}
    }
    let prec = bank_prec(d.bank, Prec::of(instr));
    let all = matches!(instr.op, Op::Tex { .. });
    (0..4u32)
        .filter(|&c| all || instr.write_mask[c as usize])
        .map(|c| {
            let off = match instr.dest_raw_packed_bits() {
                Some(bits) => crate::ir::packed_dest_slot(bits, c).0,
                None => match prec {
                    Prec::F32 => c,
                    Prec::F16 => c >> 1,
                    Prec::Fx8 => 0,
                },
            };
            (d.bank, base + off)
        })
        .collect()
}

#[cfg(test)]
mod tests {
    use super::*;

    fn op(bank: Bank, index: u8) -> Operand {
        Operand::plain(bank, index, 0)
    }

    fn ins(op: Op, dest: Operand, srcs: Vec<Operand>) -> Instr {
        Instr {
            op,
            pred: Predicate::Always,
            dest: Some(dest),
            write_mask: [true, false, false, false],
            srcs,
            half_precision: false,
            raw: 0,
            group: 0,
            blocked: None,
        }
    }

    fn step_ins(dest: Operand, a: Operand, b: Operand, c: Operand, high: bool) -> Instr {
        ins(Op::IntMadStep { signed: false, high_half: high }, dest, vec![a, b, c])
    }

    /// The edge-mask program's own address arithmetic: the secondary's byte extraction, then
    /// `Y = 1 * (16 * TXhi) + TILEY` (a step pair) and `X = TXlo * 16 + QUADRANTX`.
    fn edge_mask(tail: Vec<Instr>) -> Shader {
        let (sa, i, pa) = (Bank::SecondaryAttr, Bank::Internal, Bank::PrimaryAttr);
        let g = |n| op(Bank::Global, n);
        let imm = |n| op(Bank::Immediate, n);
        let mut pack = ins(Op::PackIntCopy { bits: 8, dest_bits: 8, signed: false }, op(sa, 1), vec![g(TILEXY)]);
        pack.srcs[0].swizzle = [1, 0, 0, 0];
        let mut instrs = vec![
            pack,
            ins(Op::Bitwise { kind: BitwiseKind::And, imm: Some(0xff), lane_bits: 32 }, op(i, 0), vec![op(sa, 1)]),
            step_ins(op(sa, 1), op(i, 0), imm(16), imm(0), false),
            ins(Op::Bitwise { kind: BitwiseKind::And, imm: Some(0xff), lane_bits: 32 }, op(sa, 13), vec![g(TILEXY)]),
            step_ins(op(i, 0), op(sa, 12), op(sa, 1), g(TILEY), false),
            step_ins(op(i, 4), op(sa, 12), op(sa, 1), op(i, 0), true),
            step_ins(op(i, 0), op(sa, 13), imm(16), g(QUADRANTX), false),
            step_ins(op(pa, 3), op(i, 4), op(sa, 0), op(i, 0), false),
            step_ins(op(pa, 0), op(pa, 3), imm(4), op(sa, 2), false),
        ];
        instrs.extend(tail);
        Shader { kind: crate::container::ProgramKind::Fragment, instrs }
    }

    #[test]
    fn the_edge_mask_arithmetic_depends_only_on_the_sums() {
        assert_eq!(tile_sums_only(&edge_mask(vec![]), &[(12, 1)]), Ok(true));
    }

    /// Without the literal `sa12 = 1` the step pair's multiplier is not a known constant, and an
    /// incomplete sum multiplied by an unknown is not provably split-independent.
    #[test]
    fn an_unknown_multiplier_of_an_incomplete_sum_refuses() {
        assert!(tile_sums_only(&edge_mask(vec![]), &[]).is_err());
    }

    /// Reading QUADRANTX on its own into an output exposes the split.
    #[test]
    fn a_lone_register_reaching_an_output_refuses() {
        let lone = step_ins(op(Bank::Output, 0), op(Bank::Immediate, 1), op(Bank::Immediate, 1), op(Bank::Global, QUADRANTX), false);
        let sh = Shader { kind: crate::container::ProgramKind::Fragment, instrs: vec![lone] };
        assert!(tile_sums_only(&sh, &[]).is_err());
        // ...and so does an incomplete form read by an operation the proof does not follow.
        let x = ins(Op::Mov, op(Bank::Temp, 0), vec![op(Bank::SecondaryAttr, 13)]);
        assert!(tile_sums_only(&edge_mask(vec![x]), &[(12, 1)]).is_err());
    }

    /// The emitted split satisfies the sums for every pixel the guest can address.
    #[test]
    fn the_emitted_split_has_the_pixel_sums() {
        for (x, y) in [(0u32, 0u32), (15, 16), (239, 135), (959, 543), (4095, 4095)] {
            let xy = tile_register_value(TILEXY, x, y).unwrap();
            assert_eq!(16 * (xy & 0xff) + tile_register_value(QUADRANTX, x, y).unwrap(), x);
            assert_eq!(16 * ((xy >> 8) & 0xff) + tile_register_value(TILEY, x, y).unwrap(), y);
        }
    }
}
