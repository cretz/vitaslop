//! Disassemble a range of a loaded guest module.
//!
//! `cargo run -p vitaslop-runtime --example disasm -- <module> <addr_hex> [count] [--arm]`
//!
//! Recovering an undocumented library means reading its call sites, and reading a call site
//! means seeing the instructions around it: which registers are set up, what the return
//! value is compared against, which struct offsets the result is stored into. The transpiler
//! decodes all of this already - this just prints it.

use yaxpeax_arch::{Decoder, U8Reader};
use yaxpeax_arm::armv7::InstDecoder;

fn main() {
    let mut args = std::env::args().skip(1);
    let path = args.next().expect("usage: disasm <module> <addr> [count] [--arm]");
    let rest: Vec<String> = args.collect();
    let thumb = !rest.iter().any(|a| a == "--arm");
    let mut numbers = rest.iter().filter(|a| !a.starts_with("--"));
    let start = numbers
        .next()
        .map(|v| u32::from_str_radix(v.trim_start_matches("0x"), 16).expect("hex address"))
        .expect("an address to start at");
    let count: usize = numbers.next().and_then(|v| v.parse().ok()).unwrap_or(48);

    let bytes = std::fs::read(&path).expect("read the module");
    let module = vitaslop_loader::load(&bytes).expect("parse the module");
    let address = start & !1;

    let segment = module
        .segments
        .iter()
        .find(|s| address >= s.vaddr && address < s.vaddr + s.data.len() as u32)
        .expect("that address is not in any segment");

    let decoder = InstDecoder::default().with_thumb_mode(thumb);
    let mut at = (address - segment.vaddr) as usize;
    for _ in 0..count {
        let pc = segment.vaddr + at as u32;
        let mut reader = U8Reader::new(&segment.data[at..]);
        match decoder.decode(&mut reader) {
            Ok(inst) => {
                // The instruction's own length is what advances the cursor; Thumb is a mix
                // of 2- and 4-byte encodings and stepping by a fixed stride desynchronises.
                let length = if thumb {
                    let half = u16::from_le_bytes([segment.data[at], segment.data[at + 1]]);
                    if (0b11101..=0b11111).contains(&(half >> 11)) { 4 } else { 2 }
                } else {
                    4
                };
                let raw: Vec<String> =
                    segment.data[at..at + length].iter().map(|b| format!("{b:02x}")).collect();
                println!("{pc:#010x}  {:<12}  {inst}", raw.join(""));
                at += length;
            }
            Err(e) => {
                println!("{pc:#010x}  <undecodable: {e}>");
                at += if thumb { 2 } else { 4 };
            }
        }
        if at + 4 > segment.data.len() {
            break;
        }
    }
}
