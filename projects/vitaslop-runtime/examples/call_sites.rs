//! Find where a guest module calls a given library's imports, in address order.
//!
//! `cargo run -p vitaslop-runtime --example call_sites -- <module> <library_nid_hex>`
//!
//! Recovering an undocumented library means reading its call sites, and the first thing
//! needed is WHERE they are and in what ORDER a caller makes them - the sequence is most of
//! the API. Booting a title to find out costs a run per discovery; this costs nothing and
//! shows every site at once, including the ones a particular run never reaches.
//!
//! Only branch instructions are decoded, not the whole instruction set: a full disassembler
//! is a different tool, and every call site is a `BL`/`BLX` whose encoding is small enough
//! to match directly.

use std::collections::HashMap;

fn main() {
    let mut args = std::env::args().skip(1);
    let path = args.next().expect("usage: call_sites <module> [library_nid_hex | --callers addr]");
    let second = args.next();
    let callers_of = if second.as_deref() == Some("--callers") {
        Some(
            u32::from_str_radix(
                args.next().expect("an address").trim_start_matches("0x"),
                16,
            )
            .expect("hex address"),
        )
    } else {
        None
    };
    let library = second.as_ref().filter(|v| *v != "--callers").map(|v| {
        u32::from_str_radix(v.trim_start_matches("0x"), 16).expect("library nid in hex")
    });

    let bytes = std::fs::read(&path).expect("read the module");
    let module = vitaslop_loader::load(&bytes).expect("parse the module");

    // "Who calls this function" - the other direction, and the one that says what a
    // recovered call's RESULT is used for.
    if let Some(target) = callers_of {
        let mut found = 0usize;
        for segment in &module.segments {
            let data = &segment.data;
            let mut at = 0usize;
            while at + 4 <= data.len() {
                let pc = segment.vaddr + at as u32;
                let first = u16::from_le_bytes([data[at], data[at + 1]]);
                let second = u16::from_le_bytes([data[at + 2], data[at + 3]]);
                if let Some(dest) = thumb_bl_target(pc, first, second)
                    && dest & !1 == target & !1
                {
                    println!("{pc:#010x} calls {target:#010x}");
                    found += 1;
                }
                at += 2;
            }
        }
        println!("{found} callers");
        return;
    }

    // Stub address -> the import it stands for. The low bit of a Thumb function pointer is
    // the instruction-set bit, so both forms are indexed.
    let mut stubs: HashMap<u32, (u32, u32)> = HashMap::new();
    for import in &module.imports {
        if library.is_none_or(|l| import.library_nid == l) {
            stubs.insert(import.stub_addr & !1, (import.library_nid, import.func_nid));
            stubs.insert(import.stub_addr | 1, (import.library_nid, import.func_nid));
        }
    }
    if stubs.is_empty() {
        eprintln!("no imports match that library");
        return;
    }

    let mut sites: Vec<(u32, u32, u32)> = Vec::new(); // (site, library, nid)
    for segment in &module.segments {
        let base = segment.vaddr;
        let data = &segment.data;
        let mut at = 0usize;
        while at + 4 <= data.len() {
            let pc = base + at as u32;
            let first = u16::from_le_bytes([data[at], data[at + 1]]);
            let second = u16::from_le_bytes([data[at + 2], data[at + 3]]);
            if let Some(target) = thumb_bl_target(pc, first, second)
                && let Some(&(lib, nid)) = stubs.get(&target)
            {
                sites.push((pc, lib, nid));
            }
            // ARM `BL` is one 32-bit word; a module is one instruction set throughout, but
            // matching both costs nothing and a mixed image would otherwise be missed.
            let word = u32::from_le_bytes([data[at], data[at + 1], data[at + 2], data[at + 3]]);
            if let Some(target) = arm_bl_target(pc, word)
                && let Some(&(lib, nid)) = stubs.get(&target)
            {
                sites.push((pc, lib, nid));
            }
            at += 2;
        }
    }
    sites.sort();
    sites.dedup();

    // A title usually links a static shim: one tiny wrapper function per export, so the
    // direct call sites are all wrappers and the INTERESTING callers are one level up.
    // Each wrapper is found by walking back to its `push {..., lr}` prologue, and then the
    // module is scanned again for calls to that.
    let mut wrappers: HashMap<u32, u32> = HashMap::new(); // function start -> nid
    for (site, _lib, nid) in &sites {
        if let Some(start) = function_start(&module, *site) {
            wrappers.insert(start | 1, *nid);
            wrappers.insert(start & !1, *nid);
        }
    }
    let mut callers: Vec<(u32, u32)> = Vec::new();
    for segment in &module.segments {
        let base = segment.vaddr;
        let data = &segment.data;
        let mut at = 0usize;
        while at + 4 <= data.len() {
            let pc = base + at as u32;
            let first = u16::from_le_bytes([data[at], data[at + 1]]);
            let second = u16::from_le_bytes([data[at + 2], data[at + 3]]);
            if let Some(target) = thumb_bl_target(pc, first, second)
                && let Some(&nid) = wrappers.get(&target)
            {
                callers.push((pc, nid));
            }
            at += 2;
        }
    }
    callers.sort();
    callers.dedup();

    println!("{} call sites", sites.len());
    let mut previous = 0u32;
    for (site, _lib, nid) in &sites {
        // A gap suggests a different caller; a blank line makes the groups readable, which
        // is the whole point - consecutive calls are a sequence.
        if previous != 0 && site.saturating_sub(previous) > 0x100 {
            println!();
        }
        println!("{site:#010x}  {}  ({nid:#010x})", vitaslop_runtime::nid::name(*nid));
        previous = *site;
    }

    println!("
{} calls through wrappers (the real usage):", callers.len());
    let mut previous = 0u32;
    for (site, nid) in &callers {
        if previous != 0 && site.saturating_sub(previous) > 0x80 {
            println!();
        }
        println!("{site:#010x}  {}  ({nid:#010x})", vitaslop_runtime::nid::name(*nid));
        previous = *site;
    }
}

/// Walk back from a call site to the `push {..., lr}` that starts its function.
fn function_start(module: &vitaslop_loader::Module, site: u32) -> Option<u32> {
    let segment = module.segments.iter().find(|s| site >= s.vaddr && site < s.vaddr + s.data.len() as u32)?;
    let offset = (site - segment.vaddr) as usize;
    let lowest = offset.saturating_sub(0x80);
    let mut at = offset;
    while at > lowest {
        at -= 2;
        let half = u16::from_le_bytes([segment.data[at], segment.data[at + 1]]);
        // PUSH {..., lr}: 1011 0101 xxxx xxxx
        if half & 0xff00 == 0xb500 {
            return Some(segment.vaddr + at as u32);
        }
        // PUSH.W {..., lr}: e92d with bit 14 of the register list set
        if at >= 2 {
            let prev = u16::from_le_bytes([segment.data[at - 2], segment.data[at - 1]]);
            if prev == 0xe92d && half & 0x4000 != 0 {
                return Some(segment.vaddr + at as u32 - 2);
            }
        }
    }
    None
}

/// Target of a Thumb-2 `BL`/`BLX` pair, if this is one.
fn thumb_bl_target(pc: u32, first: u16, second: u16) -> Option<u32> {
    if first & 0xf800 != 0xf000 {
        return None;
    }
    let blx = match second & 0xf800 {
        0xf800 => false, // BL
        0xe800 => true,  // BLX (to ARM state)
        _ => return None,
    };
    let s = ((first >> 10) & 1) as u32;
    let imm10 = (first & 0x3ff) as u32;
    let j1 = ((second >> 13) & 1) as u32;
    let j2 = ((second >> 11) & 1) as u32;
    let imm11 = (second & 0x7ff) as u32;
    let i1 = !(j1 ^ s) & 1;
    let i2 = !(j2 ^ s) & 1;
    let mut offset = (s << 24) | (i1 << 23) | (i2 << 22) | (imm10 << 12) | (imm11 << 1);
    if s == 1 {
        offset |= 0xfe00_0000; // sign extend
    }
    let target = (pc + 4).wrapping_add(offset);
    Some(if blx { target & !3 } else { target | 1 })
}

/// Target of an ARM `BL`, if this is one.
fn arm_bl_target(pc: u32, word: u32) -> Option<u32> {
    if word & 0x0f00_0000 != 0x0b00_0000 {
        return None;
    }
    let imm24 = word & 0x00ff_ffff;
    let mut offset = imm24 << 2;
    if imm24 & 0x0080_0000 != 0 {
        offset |= 0xfc00_0000;
    }
    Some((pc + 8).wrapping_add(offset))
}
