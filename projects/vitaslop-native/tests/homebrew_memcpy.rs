//! newlib's hand-written ARM `memcpy` (every vitasdk homebrew links it; no retail title
//! runs it - Sony's libc has its own) executed from a real homebrew image, over every
//! length up to a few hundred bytes and every source/destination alignment, against a
//! plain byte copy.
//!
//! The routine reaches three shapes the retail corpus never exercised together: an ARM
//! `add pc, pc, rN` computed jump into an unrolled `vld1/vst1 {d0}, [rN]!` ladder, the
//! aligned `vst1.8 {d0-d3}, [ip:64]!` quad-register stores of its bulk loop, and
//! conditional `ldr`/`str`/`pop` tails. A mis-lift in any of them corrupts the copy
//! silently and the title dies much later (a wild asset pointer, a heap check).
//!
//! Needs `VITASLOP_ARM_FUNC=<image.bin>:<image base hex>:<func hex>:<len hex>` - the
//! image is game-derived and lives outside the repo.

use vitaslop_native::{CallOutcome, HostAbi, Vm, DEFAULT_MEM_BYTES};

/// Scratch window well past the code: `[SRC, SRC+WIN)` and `[DST, DST+WIN)`.
const WIN: u32 = 0x1000;

fn spec() -> Option<(Vec<u8>, u32, u32)> {
    let v = std::env::var("VITASLOP_ARM_FUNC").ok()?;
    // Split from the right: a Windows path carries a drive-letter colon.
    let mut it = v.rsplitn(4, ':');
    let hex = |s: &str| u32::from_str_radix(s.trim_start_matches("0x"), 16).ok();
    let len = hex(it.next()?)?;
    let func = hex(it.next()?)?;
    let base = hex(it.next()?)?;
    let path = it.next()?;
    let image = std::fs::read(path).ok()?;
    let off = func.wrapping_sub(base) as usize;
    Some((image[off..off + len as usize].to_vec(), func, len))
}

#[test]
fn newlib_memcpy_copies_every_length_and_alignment() {
    let Some((code, func, _len)) = spec() else {
        eprintln!("VITASLOP_ARM_FUNC unset or unreadable: skipping");
        return;
    };
    if std::env::var("VITASLOP_DUMP_IR").is_ok() {
        let program = vitaslop_transpiler::Program {
            code: &code,
            base: func,
            thumb: false,
            entries: &[func],
            arm_entries: &[func],
            externs: &[],
            redirects: &[],
            inline_imports: &[],
            noreturn_svc: &[],
            mem_bytes: DEFAULT_MEM_BYTES,
            discover_code_pointers: false,
            import_memory: false,
        };
        eprintln!("{}", vitaslop_transpiler::dump_func(&program, func).unwrap_or_default());
    }
    let abi = HostAbi::default();
    let mut vm =
        Vm::new(&code, func, false, &[func], &[], DEFAULT_MEM_BYTES, &abi).expect("build vm");
    let src_base = func + 0x10000;
    let dst_base = src_base + WIN * 2;
    let pattern: Vec<u8> = (0..WIN).map(|i| (i.wrapping_mul(7) ^ (i >> 5)) as u8).collect();
    let mut failures = Vec::new();
    for &sa in &[0u32, 1, 2, 3, 4, 5, 7, 8, 12, 16] {
        for &da in &[0u32, 1, 2, 3, 4, 6, 7, 8, 9, 16] {
            for n in (0..=600u32).step_by(1) {
                let fill = vec![0xEEu8; WIN as usize];
                vm.write_mem(src_base, &pattern).unwrap();
                vm.write_mem(dst_base, &fill).unwrap();
                let src = src_base + sa;
                let dst = dst_base + da + 64;
                vm.set_reg(0, dst);
                vm.set_reg(1, src);
                vm.set_reg(2, n);
                vm.set_reg(13, func + 0x8000);
                match vm.call_bounded(func, 10_000_000) {
                    CallOutcome::Returned => {}
                    other => {
                        failures.push(format!("sa={sa} da={da} n={n}: {other:?}"));
                        continue;
                    }
                }
                let got = vm.read_mem(dst_base, WIN as usize).unwrap();
                let mut want = fill.clone();
                let off = (da + 64) as usize;
                want[off..off + n as usize]
                    .copy_from_slice(&pattern[sa as usize..(sa + n) as usize]);
                if got != want {
                    let first = got.iter().zip(&want).position(|(a, b)| a != b).unwrap();
                    failures.push(format!(
                        "sa={sa} da={da} n={n}: first bad byte at dst+{} (got {:#04x} want {:#04x}); r0={:#x}",
                        first as i64 - off as i64,
                        got[first],
                        want[first],
                        vm.get_reg(0)
                    ));
                    if failures.len() > 40 {
                        break;
                    }
                }
            }
        }
    }
    assert!(failures.is_empty(), "memcpy mis-copies:\n{}", failures.join("\n"));
}
