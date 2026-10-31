//! Writes RANDOM cases: short sequences of ARM-state integer instructions, the families a C
//! compiler emits (data processing over every shifter form, multiplies, extends, bitfields,
//! saturating and packed arithmetic), under random condition codes and random seed registers.
//!
//! Only the human half is written - `regen` adds qemu's golden like it does for any case - so a
//! fuzz case is checked by the same oracle and the same harness as a hand-written one, and a
//! failure can be copied into `cases/` as it stands once it is minimised.
//!
//!   fuzz <dir> <count> <seed>
//!   ARM_CASES_DIR=<dir> regen
//!   ARM_FUZZ_DIR=<dir> cargo test -p vitaslop-conformance-harness --lib fuzz_dir -- --ignored --nocapture

use std::fmt::Write as _;
use std::fs;
use std::path::Path;

/// xorshift64*: deterministic for a seed, and no dependency.
struct Rng(u64);

impl Rng {
    fn next(&mut self) -> u64 {
        self.0 ^= self.0 >> 12;
        self.0 ^= self.0 << 25;
        self.0 ^= self.0 >> 27;
        self.0.wrapping_mul(0x2545_F491_4F6C_DD1D)
    }
    fn below(&mut self, n: u32) -> u32 {
        (self.next() % u64::from(n)) as u32
    }
    fn pick<'a>(&mut self, items: &[&'a str]) -> &'a str {
        items[self.below(items.len() as u32) as usize]
    }
    fn chance(&mut self, percent: u32) -> bool {
        self.below(100) < percent
    }
    /// A register value: mostly edge cases, where a lowering is most likely wrong.
    fn value(&mut self) -> u32 {
        const EDGES: [u32; 12] =
            [0, 1, 2, 31, 32, 33, 0x7fff_ffff, 0x8000_0000, 0xffff_ffff, 0xffff_fffe, 0x0000_ffff, 0xffff_8000];
        match self.below(4) {
            0 => EDGES[self.below(EDGES.len() as u32) as usize],
            1 => self.below(64),
            _ => self.next() as u32,
        }
    }
}

const CONDS: [&str; 14] = ["eq", "ne", "cs", "cc", "mi", "pl", "vs", "vc", "hi", "ls", "ge", "lt", "gt", "le"];

fn reg(r: &mut Rng) -> String {
    format!("r{}", r.below(13))
}

// `thumb`: the same families as Thumb-2 spells them - see `main`. Set once, before any case.
static THUMB: std::sync::atomic::AtomicBool = std::sync::atomic::AtomicBool::new(false);
fn thumb() -> bool {
    THUMB.load(std::sync::atomic::Ordering::Relaxed)
}
// The condition the instruction being built took, so a Thumb case can open its IT block.
thread_local! {
    static PENDING_IT: std::cell::Cell<&'static str> = const { std::cell::Cell::new("") };
}

fn cond(r: &mut Rng) -> &'static str {
    let c = if r.chance(30) { CONDS[r.below(CONDS.len() as u32) as usize] } else { "" };
    PENDING_IT.with(|p| p.set(c));
    c
}

/// A modified immediate: an 8-bit value rotated right by an even amount. Thumb-2's own
/// immediates cannot wrap a value around the word, so there it is an 8-bit value shifted left.
fn imm(r: &mut Rng) -> u32 {
    if thumb() {
        return r.below(256) << r.below(25);
    }
    (r.below(256)).rotate_right(2 * r.below(16))
}

/// The flexible second operand, in every form.
fn op2(r: &mut Rng) -> String {
    // Thumb-2 has no register-shifted register operand (only the shift instructions take one).
    match r.below(if thumb() { 4 } else { 5 }) {
        0 => format!("#{}", imm(r)),
        1 => reg(r),
        2 => {
            let (sh, n) = match r.below(4) {
                0 => ("lsl", r.below(32)),
                1 => ("lsr", 1 + r.below(32)),
                2 => ("asr", 1 + r.below(32)),
                _ => ("ror", 1 + r.below(31)),
            };
            format!("{}, {sh} #{n}", reg(r))
        }
        3 => format!("{}, rrx", reg(r)),
        _ => format!("{}, {} {}", reg(r), r.pick(&["lsl", "lsr", "asr", "ror"]), reg(r)),
    }
}

fn insn(r: &mut Rng) -> String {
    let c = cond(r);
    let s = if r.chance(40) { "s" } else { "" };
    // Thumb-2 has no 32-bit flag-setting multiply of any width.
    let ms = if thumb() { "" } else { s };
    match r.below(16) {
        0..=4 => {
            let ops: &[&str] = if thumb() {
                &["and", "eor", "sub", "rsb", "add", "adc", "sbc", "orr", "bic", "orn"]
            } else {
                &["and", "eor", "sub", "rsb", "add", "adc", "sbc", "rsc", "orr", "bic"]
            };
            let op = r.pick(ops);
            format!("{op}{s}{c} {}, {}, {}", reg(r), reg(r), op2(r))
        }
        5 => match r.below(4) {
            0 | 1 => format!("{}{s}{c} {}, {}", r.pick(&["mov", "mvn"]), reg(r), op2(r)),
            // The shift instructions themselves, by an immediate, by a register, and `rrx`.
            2 => {
                let (op, n) = match r.below(4) {
                    0 => ("lsl", r.below(32)),
                    1 => ("lsr", 1 + r.below(32)),
                    2 => ("asr", 1 + r.below(32)),
                    _ => ("ror", 1 + r.below(31)),
                };
                if r.chance(50) {
                    format!("{op}{s}{c} {}, {}, #{n}", reg(r), reg(r))
                } else {
                    format!("{op}{s}{c} {}, {}, {}", reg(r), reg(r), reg(r))
                }
            }
            _ => format!("rrx{s}{c} {}, {}", reg(r), reg(r)),
        },
        6 => format!("{}{c} {}, {}", r.pick(&["tst", "teq", "cmp", "cmn"]), reg(r), op2(r)),
        7 => match r.below(3) {
            0 => format!("mul{ms}{c} {}, {}, {}", reg(r), reg(r), reg(r)),
            1 => format!("mla{ms}{c} {}, {}, {}, {}", reg(r), reg(r), reg(r), reg(r)),
            _ => format!("mls{c} {}, {}, {}, {}", reg(r), reg(r), reg(r), reg(r)),
        },
        8 => {
            let lo = r.below(13);
            let hi = (lo + 1 + r.below(12)) % 13;
            let op = r.pick(&["umull", "smull", "umlal", "smlal"]);
            format!("{op}{ms}{c} r{lo}, r{hi}, {}, {}", reg(r), reg(r))
        }
        9 => format!("{}{c} {}, {}", r.pick(&["clz", "rev", "rev16", "revsh", "rbit"]), reg(r), reg(r)),
        10 => {
            let rot = [0, 8, 16, 24][r.below(4) as usize];
            if r.chance(50) {
                let op = r.pick(&["uxtb", "uxth", "sxtb", "sxth", "uxtb16", "sxtb16"]);
                format!("{op}{c} {}, {}, ror #{rot}", reg(r), reg(r))
            } else {
                let op = r.pick(&["uxtab", "uxtah", "sxtab", "sxtah", "uxtab16", "sxtab16"]);
                format!("{op}{c} {}, {}, {}, ror #{rot}", reg(r), reg(r), reg(r))
            }
        }
        11 => {
            let lsb = r.below(32);
            let width = 1 + r.below(32 - lsb);
            match r.below(4) {
                0 => format!("ubfx{c} {}, {}, #{lsb}, #{width}", reg(r), reg(r)),
                1 => format!("sbfx{c} {}, {}, #{lsb}, #{width}", reg(r), reg(r)),
                2 => format!("bfi{c} {}, {}, #{lsb}, #{width}", reg(r), reg(r)),
                _ => format!("bfc{c} {}, #{lsb}, #{width}", reg(r)),
            }
        }
        12 => format!("{}{c} {}, #{}", r.pick(&["movw", "movt"]), reg(r), r.below(65536)),
        13 => match r.below(3) {
            0 => format!("usat{c} {}, #{}, {}, lsl #{}", reg(r), r.below(32), reg(r), r.below(32)),
            1 => format!("ssat{c} {}, #{}, {}, asr #{}", reg(r), 1 + r.below(32), reg(r), 1 + r.below(31)),
            _ => format!("{}{c} {}, {}, {}", r.pick(&["qadd", "qsub", "qdadd", "qdsub"]), reg(r), reg(r), reg(r)),
        },
        14 => {
            let op = r.pick(&[
                "smulbb", "smulbt", "smultb", "smultt", "smulwb", "smulwt", "smmul", "smmulr", "smuad", "smusd",
                "smuadx", "smusdx",
            ]);
            if r.chance(50) {
                format!("{op}{c} {}, {}, {}", reg(r), reg(r), reg(r))
            } else {
                let op = r.pick(&["smlabb", "smlatt", "smlawb", "smlawt", "smmla", "smmls", "smlad", "smlsd"]);
                format!("{op}{c} {}, {}, {}, {}", reg(r), reg(r), reg(r), reg(r))
            }
        }
        _ => match r.below(4) {
            0 => {
                let op = r.pick(&[
                    "uadd8", "usub8", "uadd16", "usub16", "sadd8", "ssub8", "sadd16", "ssub16", "uqadd8", "uqsub8",
                    "qadd8", "qsub16", "uhadd8", "shsub16", "uasx", "usax", "sasx", "ssax",
                ]);
                format!("{op}{c} {}, {}, {}", reg(r), reg(r), reg(r))
            }
            1 => format!("sel{c} {}, {}, {}", reg(r), reg(r), reg(r)),
            2 => {
                if r.chance(50) {
                    format!("pkhbt{c} {}, {}, {}, lsl #{}", reg(r), reg(r), reg(r), r.below(32))
                } else {
                    format!("pkhtb{c} {}, {}, {}, asr #{}", reg(r), reg(r), reg(r), 1 + r.below(31))
                }
            }
            _ => format!("usad8{c} {}, {}, {}", reg(r), reg(r), reg(r)),
        },
    }
}

/// The MEMORY window a `mem` case addresses: the seed registers pushed twice, 104 bytes the
/// oracle and the engine both hold the same bytes in, reached through `r11` (= sp after the
/// pushes). Every access is kept inside it, so an address never enters the comparison.
const WINDOW: u32 = 104;

/// A data register a memory case may write: anything but the base (`r11`).
fn data_reg(r: &mut Rng) -> u32 {
    let n = r.below(12);
    if n == 11 { 12 } else { n }
}

/// One load or store against the window. `base` is r11's offset into it, moved by writeback.
fn mem_insn(r: &mut Rng, base: &mut u32) -> String {
    let c = cond(r);
    // A conditional writeback might not happen, which would lose track of the base: only an
    // unconditional access writes the base back.
    let wb_ok = c.is_empty();
    let (op, size) = match r.below(10) {
        0 => ("ldr", 4),
        1 => ("str", 4),
        2 => ("ldrb", 1),
        3 => ("strb", 1),
        4 => ("ldrh", 2),
        5 => ("strh", 2),
        6 => ("ldrsb", 1),
        7 => ("ldrsh", 2),
        8 => ("ldrd", 8),
        _ => ("strd", 8),
    };
    if r.chance(15) {
        // A block transfer of up to four registers, ascending order.
        let mut regs: Vec<u32> = (0..4).map(|_| data_reg(r)).collect();
        regs.sort_unstable();
        regs.dedup();
        let bytes = regs.len() as u32 * 4;
        let list: Vec<String> = regs.iter().map(|n| format!("r{n}")).collect();
        let load = r.chance(50);
        let room = WINDOW.saturating_sub(*base);
        if room < bytes || !(*base).is_multiple_of(4) {
            return format!("mov{c} r{}, #{}", data_reg(r), r.below(256));
        }
        let wb = wb_ok && r.chance(40) && *base + 2 * bytes <= WINDOW;
        if wb {
            *base += bytes;
        }
        return format!(
            "{}ia{c} r11{}, {{{}}}",
            if load { "ldm" } else { "stm" },
            if wb { "!" } else { "" },
            list.join(", ")
        );
    }
    if *base + size > WINDOW {
        return format!("mov{c} r{}, #{}", data_reg(r), r.below(256));
    }
    // An even pair below the base register.
    let rt = if size == 8 { 2 * r.below(5) } else { data_reg(r) };
    let regs = if size == 8 { format!("r{rt}, r{}", rt + 1) } else { format!("r{rt}") };
    // A reachable offset whose ADDRESS is aligned (the base may sit anywhere after a byte
    // writeback), so even `ldrd` never faults on alignment.
    let align = size.min(4);
    let span = WINDOW - size - *base;
    let first = (align - *base % align) % align;
    if first > span {
        return format!("mov{c} r{}, #{}", data_reg(r), r.below(256));
    }
    let off = first + r.below((span - first) / align + 1) * align;
    // Thumb-2's doubleword immediate is a multiple of four, not just an aligned address.
    if thumb() && size == 8 && !off.is_multiple_of(4) {
        return format!("mov{c} r{}, #{}", data_reg(r), r.below(256));
    }
    // The halfword and doubleword forms take only an 8-bit immediate; the word and byte
    // forms take twelve. A register offset is set right before use, to a value inside.
    let imm_ok = off < 256 || matches!(size, 1 | 4) && !op.ends_with("sb");
    match r.below(4) {
        0 if imm_ok => format!("{op}{c} {regs}, [r11, #{off}]"),
        // Thumb-2's pre-indexed writeback takes an 8-bit offset only.
        1 if imm_ok && wb_ok && rt != 11 && size != 8 && (!thumb() || off < 256) => {
            *base += off;
            format!("{op} {regs}, [r11, #{off}]!")
        }
        2 if imm_ok && wb_ok && size != 8 && off < 256 => {
            let s = format!("{op} {regs}, [r11], #{off}");
            *base += off;
            s
        }
        // Thumb-2's doubleword forms take no register offset; the window keeps `off` in range.
        _ if thumb() && size == 8 => format!("{op}{c} {regs}, [r11, #{off}]"),
        _ => {
            let idx = loop {
                let n = data_reg(r);
                if n != rt && (size != 8 || n != rt + 1) {
                    break n;
                }
            };
            if size == 4 && off.is_multiple_of(4) && r.chance(50) {
                format!("mov{c} r{idx}, #{}\n    {op}{c} {regs}, [r11, r{idx}, lsl #2]", off / 4)
            } else {
                format!("mov{c} r{idx}, #{off}\n    {op}{c} {regs}, [r11, r{idx}]")
            }
        }
    }
}

/// One VFP instruction over d0-d7 (and the s0-s15 they alias), conditional like any other.
fn vfp_insn(r: &mut Rng) -> String {
    let c = cond(r);
    let d = |r: &mut Rng| format!("d{}", r.below(8));
    let s = |r: &mut Rng| format!("s{}", r.below(16));
    let (t, x) = if r.chance(60) { ("f64", 0) } else { ("f32", 1) };
    let v = |r: &mut Rng| if x == 0 { d(r) } else { s(r) };
    match r.below(12) {
        0..=2 => {
            let op = r.pick(&["vadd", "vsub", "vmul", "vdiv", "vnmul"]);
            format!("{op}{c}.{t} {}, {}, {}", v(r), v(r), v(r))
        }
        3 => {
            let op = r.pick(&["vmla", "vmls", "vnmla", "vnmls"]);
            format!("{op}{c}.{t} {}, {}, {}", v(r), v(r), v(r))
        }
        4 => format!("{}{c}.{t} {}, {}", r.pick(&["vabs", "vneg", "vsqrt", "vmov"]), v(r), v(r)),
        5 => format!("vmov{c}.{t} {}, #{}", v(r), r.pick(&["1.0", "0.5", "-2.0", "3.0", "-0.25", "10.0", "0.125"])),
        6 => match r.below(6) {
            0 => format!("vcvt{c}.f64.s32 {}, {}", d(r), s(r)),
            1 => format!("vcvt{c}.f64.u32 {}, {}", d(r), s(r)),
            2 => format!("vcvt{c}.s32.f64 {}, {}", s(r), d(r)),
            3 => format!("vcvt{c}.u32.f64 {}, {}", s(r), d(r)),
            4 => format!("vcvt{c}.f32.f64 {}, {}", s(r), d(r)),
            _ => format!("vcvt{c}.f64.f32 {}, {}", d(r), s(r)),
        },
        7 => match r.below(3) {
            0 => format!("vcvt{c}.s32.f32 {}, {}", s(r), s(r)),
            1 => format!("vcvt{c}.f32.s32 {}, {}", s(r), s(r)),
            _ => format!("vcvtr{c}.s32.f64 {}, {}", s(r), d(r)),
        },
        8 | 9 => {
            let cmp = if r.chance(30) {
                format!("{}{c}.{t} {}, #0", r.pick(&["vcmp", "vcmpe"]), v(r))
            } else {
                format!("{}{c}.{t} {}, {}", r.pick(&["vcmp", "vcmpe"]), v(r), v(r))
            };
            format!("{cmp}\n    vmrs{c} APSR_nzcv, fpscr")
        }
        10 => format!("vmov{c} r{}, {}", r.below(13), s(r)),
        _ => format!("vmov{c} {}, r{}", s(r), r.below(13)),
    }
}

/// In Thumb state, open an IT block for the conditional instruction(s) just built: one more
/// `t` per extra line, since a memory form can be a register set-up and the access, both
/// conditional.
fn it_block(thumb_mode: bool, i: String) -> String {
    let c = PENDING_IT.with(|p| p.replace(""));
    if !thumb_mode || c.is_empty() {
        return i;
    }
    let extra = "t".repeat(i.matches('\n').count());
    format!("it{extra} {c}\n    {i}")
}

fn main() {
    let args: Vec<String> = std::env::args().collect();
    if !(4..=6).contains(&args.len()) {
        eprintln!("usage: fuzz <dir> <count> <seed> [instructions per case] [mem|vfp|thumb|thumbmem]");
        std::process::exit(2);
    }
    // `mem`: loads and stores against a window of the stack, between the ALU instructions.
    // `vfp`: floating point over d0-d7, made from the seed registers and moved back to them.
    // `thumb`: the integer families in Thumb state - Thumb-2 encodings, each conditional
    // instruction in its own IT block. Retail code is mostly Thumb, and its lowering was the
    // part never fuzzed.
    // `thumbmem`: the memory mode in Thumb state.
    let mem = args.get(5).is_some_and(|m| m == "mem" || m == "thumbmem");
    let vfp = args.get(5).is_some_and(|m| m == "vfp");
    let thumb_mode = args.get(5).is_some_and(|m| m == "thumb" || m == "thumbmem");
    THUMB.store(thumb_mode, std::sync::atomic::Ordering::Relaxed);
    let dir = Path::new(&args[1]);
    let count: u32 = args[2].parse().expect("count");
    let seed: u64 = args[3].parse().expect("seed");
    // One instruction per case names the failing instruction outright; a longer sequence
    // finds what only a combination gets wrong (a flag one instruction leaves for the next).
    let fixed: Option<u32> = args.get(4).map(|s| s.parse().expect("instructions per case"));
    fs::create_dir_all(dir).expect("create dir");
    let mut rng = Rng(seed.wrapping_mul(0x9E37_79B9_7F4A_7C15) | 1);
    for n in 0..count {
        let len = fixed.unwrap_or_else(|| 4 + rng.below(9));
        let mut asm = String::new();
        // Flags from the seeds first, so conditions and carries start from a varied state.
        let first = loop {
            let first = match rng.below(4) {
                0 => format!("cmp {}, {}", reg(&mut rng), reg(&mut rng)),
                1 => format!("cmn {}, {}", reg(&mut rng), reg(&mut rng)),
                2 => format!("adds {}, {}, {}", reg(&mut rng), reg(&mut rng), reg(&mut rng)),
                _ => format!("subs {}, {}, {}", reg(&mut rng), reg(&mut rng), reg(&mut rng)),
            };
            if !mem || !first.contains("r11") {
                break first;
            }
        };
        if mem {
            let _ = writeln!(asm, "    push {{r0-r12}}\n    push {{r0-r12}}\n    mov r11, sp");
        }
        let _ = writeln!(asm, "    {first}");
        if vfp {
            // Exact doubles from the seed integers, and now and then raw bits.
            for i in 0..8 {
                if rng.chance(15) {
                    let _ = writeln!(asm, "    vmov d{i}, r{}, r{}", rng.below(13), rng.below(13));
                } else {
                    let _ = writeln!(asm, "    vmov s{}, r{i}\n    vcvt.f64.s32 d{i}, s{}", 2 * i + 1, 2 * i + 1);
                }
            }
        }
        let mut base = 0u32;
        for _ in 0..len {
            let i = if vfp && rng.chance(80) {
                vfp_insn(&mut rng)
            } else if mem && rng.chance(60) {
                let i = mem_insn(&mut rng, &mut base);
                it_block(thumb_mode, i)
            } else {
                loop {
                    let i = insn(&mut rng);
                    // The ALU instructions of a memory case leave the base alone.
                    if !mem || !i.contains("r11") {
                        break it_block(thumb_mode, i);
                    }
                }
            };
            let _ = writeln!(asm, "    {i}");
        }
        if vfp {
            for i in 0..6 {
                let _ = writeln!(asm, "    vmov r{}, r{}, d{i}", 2 * i, 2 * i + 1);
            }
            let _ = writeln!(asm, "    vmov r12, s14");
        }
        if mem {
            // The base as an offset into the window, which both engines agree on; and sp
            // back where the case found it.
            // (Thumb-2 takes sp only as the FIRST operand of a subtraction, so there the offset
            // is read the other way round - the same value on both engines either way.)
            let back = if thumb_mode { "sub r11, sp, r11" } else { "sub r11, r11, sp" };
            let _ = writeln!(asm, "    {back}\n    add sp, sp, #{WINDOW}");
        }
        let mut regs = String::new();
        for i in 0..13 {
            let _ = write!(regs, "{}r{i} = {}", if i == 0 { "" } else { ", " }, rng.value());
        }
        let text = format!(
            "description = \"fuzz seed {seed} case {n}\"\n\nmode = \"{mode}\"\n\nasm = '''\n{asm}'''\n\n[in]\nregs = {{ {regs} }}\n",
            mode = if thumb_mode { "thumb" } else { "arm" }
        );
        fs::write(dir.join(format!("fuzz_{seed}_{n:04}.toml")), text).expect("write case");
    }
    println!("wrote {count} case(s) to {}", dir.display());
}
