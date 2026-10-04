//! TOOL: Thumb-2 disassembly of a segment dumped by `vitaslop-native/tests/dump_segment.rs`.
//!
//! `VITASLOP_DISASM_FILE=<dump> VITASLOP_DISASM_RANGE=0x81385eae-0x81385fc0
//! cargo test -p vitaslop-transpiler --test disasm_dump -- --ignored --nocapture`

use yaxpeax_arch::{Decoder, U8Reader};
use yaxpeax_arm::armv7::InstDecoder;

#[test]
#[ignore = "needs VITASLOP_DISASM_FILE"]
fn disasm_dump() {
    let Ok(file) = std::env::var("VITASLOP_DISASM_FILE") else { return };
    let range = std::env::var("VITASLOP_DISASM_RANGE").expect("VITASLOP_DISASM_RANGE=lo-hi");
    let (lo, hi) = range.split_once('-').expect("lo-hi");
    let parse = |v: &str| u32::from_str_radix(v.trim_start_matches("0x"), 16).expect("hex");
    let (lo, hi) = (parse(lo) & !1, parse(hi));
    let bytes = std::fs::read(file).expect("read dump");
    let base = u32::from_be_bytes([bytes[0], bytes[1], bytes[2], bytes[3]]);
    let code = &bytes[4..];
    let decoder = InstDecoder::default_thumb();
    let mut pc = lo;
    while pc < hi {
        let off = (pc - base) as usize;
        let mut reader = U8Reader::new(&code[off..]);
        match decoder.decode(&mut reader) {
            Ok(inst) => {
                let len = if (code[off + 1] & 0xf8) >= 0xe8 { 4 } else { 2 };
                println!("{pc:08x}: {inst}");
                pc += len;
            }
            Err(e) => {
                println!("{pc:08x}: <{e:?}> {:02x}{:02x}", code[off + 1], code[off]);
                pc += 2;
            }
        }
    }
}
