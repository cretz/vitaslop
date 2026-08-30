//! Find every place a loaded guest module NAMES a given address.
//!
//! `cargo run -p vitaslop-runtime --example refs -- <module> <addr_hex> [span] [--arm]`
//!
//! A static object with no symbols is found by its ADDRESS: the uniform writer reads its sky
//! parameters from `0x81d50cd8`, so the question "who fills that struct" is the question "what
//! code names it". Grepping the file on disk answers nothing - the module is packed, and the
//! loader is what produces the image whose text carries final addresses - so this loads the
//! module the same way the engine does and scans the image.
//!
//! Two forms carry an address in ARM code and both are matched:
//! * a `MOVW`/`MOVT` pair, which is how a Thumb-2 compiler materialises a 32-bit constant. The
//!   pair need not be adjacent (the scheduler interleaves other work between the halves), so
//!   the last `MOVW` per register is remembered until a `MOVT` on the same register completes
//!   it.
//! * a literal WORD equal to the address - a literal pool entry, a vtable slot, a relocated
//!   pointer in initialised data. These are reported separately because a data hit is not a
//!   code reference: it is a place some code will LOAD the address from.
//!
//! `span` (default 0x40) widens the match to `[addr, addr+span)`, because a compiler names the
//! BASE of a struct and reaches a field by offset - asking only about the field's own address
//! finds nothing at all.

fn main() {
    let mut args = std::env::args().skip(1);
    let path = args.next().expect("usage: refs <module> <addr> [span] [--arm]");
    let rest: Vec<String> = args.collect();
    let arm = rest.iter().any(|a| a == "--arm");
    let mut numbers = rest.iter().filter(|a| !a.starts_with("--"));
    let hex = |v: &String| u32::from_str_radix(v.trim_start_matches("0x"), 16).expect("hex");
    let target = numbers.next().map(hex).expect("an address to look for");
    let span = numbers.next().map(hex).unwrap_or(0x40);
    let hit = |a: u32| a >= target && a < target.wrapping_add(span);

    let bytes = std::fs::read(&path).expect("read the module");
    let module = vitaslop_loader::load(&bytes).expect("parse the module");

    // The image's own extent, printed first: an address ABOVE every segment is not a static at
    // all - it is heap, and "no code names it" then means something entirely different.
    for seg in &module.segments {
        eprintln!(
            "segment {:#010x}..{:#010x} file-backed, zero-filled .bss to {:#010x}",
            seg.vaddr,
            seg.vaddr + seg.data.len() as u32,
            seg.vaddr + seg.mem_size
        );
    }

    for seg in &module.segments {
        let base = seg.vaddr;
        let data = &seg.data;

        // MOVW/MOVT. `pending[r]` is the low half most recently materialised into register r.
        let mut pending = [None::<(u32, u32)>; 16]; // (pc of the MOVW, imm16)
        let mut at = 0usize;
        while at + 4 <= data.len() {
            let hw0 = u16::from_le_bytes([data[at], data[at + 1]]);
            let hw1 = u16::from_le_bytes([data[at + 2], data[at + 3]]);
            let pc = base + at as u32;
            if arm {
                // ARM A2 encodings: MOVW `cccc 0011 0000 imm4 Rd imm12`, MOVT `... 0011 0100 ...`.
                let w = u32::from_le_bytes([data[at], data[at + 1], data[at + 2], data[at + 3]]);
                let op = (w >> 20) & 0xff;
                let rd = ((w >> 12) & 0xf) as usize;
                let imm16 = ((w >> 4) & 0xf000) | (w & 0xfff);
                if op == 0x30 {
                    pending[rd] = Some((pc, imm16));
                } else if op == 0x34 {
                    if let Some((lo_pc, lo)) = pending[rd] {
                        let addr = (imm16 << 16) | lo;
                        if hit(addr) {
                            println!("{pc:#010x}  movt r{rd} (movw at {lo_pc:#010x})  -> {addr:#010x}");
                        }
                    }
                }
                at += 4;
                continue;
            }
            // Thumb-2 T3 MOVW / T1 MOVT: the same second halfword layout, different op bits.
            let is_movw = (hw0 & 0xfbf0) == 0xf240;
            let is_movt = (hw0 & 0xfbf0) == 0xf2c0;
            if is_movw || is_movt {
                let i = ((hw0 >> 10) & 1) as u32;
                let imm4 = (hw0 & 0xf) as u32;
                let imm3 = ((hw1 >> 12) & 0x7) as u32;
                let rd = ((hw1 >> 8) & 0xf) as usize;
                let imm8 = (hw1 & 0xff) as u32;
                let imm16 = (imm4 << 12) | (i << 11) | (imm3 << 8) | imm8;
                if is_movw {
                    pending[rd] = Some((pc, imm16));
                } else if let Some((lo_pc, lo)) = pending[rd] {
                    let addr = (imm16 << 16) | lo;
                    if hit(addr) {
                        println!("{pc:#010x}  movt r{rd} (movw at {lo_pc:#010x})  -> {addr:#010x}");
                    }
                }
                at += 4;
                continue;
            }
            // Not a wide instruction we care about: step one halfword, since Thumb is a mix of
            // 2- and 4-byte encodings and a fixed stride desynchronises (see `disasm`).
            at += if (0b11101..=0b11111).contains(&(hw0 >> 11)) { 4 } else { 2 };
        }

        // Literal words.
        for off in (0..data.len().saturating_sub(3)).step_by(4) {
            let w = u32::from_le_bytes([data[off], data[off + 1], data[off + 2], data[off + 3]]);
            if hit(w) {
                println!("{:#010x}  word  -> {w:#010x}", base + off as u32);
            }
        }
    }
}
