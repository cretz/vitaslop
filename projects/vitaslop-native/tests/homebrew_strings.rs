//! newlib's hand-written Thumb-2 `strlen` and `strcmp` (`strlen-armv7.S`,
//! `strcmp-armv7.S`: `ldrd` with post-increment, `uadd8`/`sel`, `orn`, `rev`, `clz`,
//! `tbb`, `it` blocks), executed from a real homebrew image over every alignment and a
//! range of lengths, against Rust's answer. Every vitasdk homebrew links them; no
//! retail title runs them (Sony's libc has its own).
//!
//! Needs `VITASLOP_HB_IMAGE=<image.bin>:<base hex>` plus `VITASLOP_HB_STRLEN=<hex>` and/or
//! `VITASLOP_HB_STRCMP=<hex>` (Thumb entries, bit 0 clear).

use vitaslop_native::{CallOutcome, HostAbi, Vm, DEFAULT_MEM_BYTES};

const WIN: u32 = 0x400;

fn image() -> Option<(Vec<u8>, u32)> {
    let v = std::env::var("VITASLOP_HB_IMAGE").ok()?;
    let (path, base) = v.rsplit_once(':')?;
    let base = u32::from_str_radix(base.trim_start_matches("0x"), 16).ok()?;
    Some((std::fs::read(path).ok()?, base))
}

fn entry(var: &str) -> Option<u32> {
    let v = std::env::var(var).ok()?;
    u32::from_str_radix(v.trim_start_matches("0x"), 16).ok()
}

/// A VM over the whole image (the routines read past string ends in aligned words, so
/// the image, not a slice, is the code), with the routine at `func` as the entry.
fn vm(image: &[u8], base: u32, func: u32) -> Vm {
    let abi = HostAbi::default();
    Vm::new(image, base, true, &[func], &[], DEFAULT_MEM_BYTES, &abi).expect("build vm")
}

#[test]
fn newlib_strlen_counts_every_length_and_alignment() {
    let (Some((image, base)), Some(func)) = (image(), entry("VITASLOP_HB_STRLEN")) else {
        eprintln!("VITASLOP_HB_IMAGE/VITASLOP_HB_STRLEN unset: skipping");
        return;
    };
    let mut vm = vm(&image, base, func);
    let buf = base + 0x400000;
    let mut failures = Vec::new();
    for align in 0..16u32 {
        for len in 0..80u32 {
            // Non-zero filler before and after the string, so a lost terminator or a
            // short count both show.
            let mut mem = vec![0x41u8; WIN as usize];
            let s = (align as usize)..(align + len) as usize;
            for (i, b) in mem[s.clone()].iter_mut().enumerate() {
                *b = b'a' + (i % 26) as u8;
            }
            mem[(align + len) as usize] = 0;
            vm.write_mem(buf, &mem).unwrap();
            vm.set_reg(0, buf + align);
            vm.set_reg(13, base + 0x300000);
            match vm.call_bounded(func, 1_000_000) {
                CallOutcome::Returned => {}
                other => {
                    failures.push(format!("align={align} len={len}: {other:?}"));
                    continue;
                }
            }
            let got = vm.get_reg(0);
            if got != len {
                failures.push(format!("align={align} len={len}: got {got}"));
            }
        }
    }
    assert!(failures.is_empty(), "strlen mis-counts:\n{}", failures.join("\n"));
}

#[test]
fn newlib_strcmp_orders_every_pair() {
    let (Some((image, base)), Some(func)) = (image(), entry("VITASLOP_HB_STRCMP")) else {
        eprintln!("VITASLOP_HB_IMAGE/VITASLOP_HB_STRCMP unset: skipping");
        return;
    };
    let mut vm = vm(&image, base, func);
    let a_buf = base + 0x400000;
    let b_buf = a_buf + WIN;
    let words: [&[u8]; 9] = [
        b"", b"a", b"truetype", b"truetyp", b"truetypes", b"smooth", b"smootg", b"abcdefghijklmnopqrstuvwxyz0123456789", b"abcdefghijklmnopqrstuvwxyz0123456788",
    ];
    let mut failures = Vec::new();
    for &a in &words {
        for &b in &words {
            for aa in [0u32, 1, 2, 3, 4, 5, 7, 8, 9, 13] {
                for ba in [0u32, 1, 2, 3, 4, 6, 7, 8, 11, 15] {
                    let mut ma = vec![0x7fu8; WIN as usize];
                    let mut mb = vec![0x7fu8; WIN as usize];
                    ma[aa as usize..aa as usize + a.len()].copy_from_slice(a);
                    ma[aa as usize + a.len()] = 0;
                    mb[ba as usize..ba as usize + b.len()].copy_from_slice(b);
                    mb[ba as usize + b.len()] = 0;
                    vm.write_mem(a_buf, &ma).unwrap();
                    vm.write_mem(b_buf, &mb).unwrap();
                    vm.set_reg(0, a_buf + aa);
                    vm.set_reg(1, b_buf + ba);
                    vm.set_reg(13, base + 0x300000);
                    match vm.call_bounded(func, 1_000_000) {
                        CallOutcome::Returned => {}
                        other => {
                            failures.push(format!("{a:?}@{aa} vs {b:?}@{ba}: {other:?}"));
                            continue;
                        }
                    }
                    let got = vm.get_reg(0) as i32;
                    let want = a.cmp(b);
                    let ok = match want {
                        std::cmp::Ordering::Less => got < 0,
                        std::cmp::Ordering::Equal => got == 0,
                        std::cmp::Ordering::Greater => got > 0,
                    };
                    if !ok {
                        failures.push(format!("{:?}@{aa} vs {:?}@{ba}: got {got} want {want:?}", String::from_utf8_lossy(a), String::from_utf8_lossy(b)));
                    }
                }
            }
        }
    }
    assert!(failures.is_empty(), "strcmp mis-orders:\n{}", failures.join("\n"));
}

#[test]
fn strcmp_probe_single_cases() {
    let (Some((image, base)), Some(func)) = (image(), entry("VITASLOP_HB_STRCMP")) else { return };
    let mut vm = vm(&image, base, func);
    let a_buf = base + 0x400000;
    let b_buf = a_buf + WIN;
    for (a, b) in [(&b"\0"[..], &b"a\0"[..]), (b"a\0", b"b\0"), (b"b\0", b"a\0"), (b"a\0", b"a\0"), (b"truetype\0", b"truetyp\0")] {
        vm.write_mem(a_buf, a).unwrap();
        vm.write_mem(b_buf, b).unwrap();
        vm.set_reg(0, a_buf);
        vm.set_reg(1, b_buf);
        vm.set_reg(2, 0xdead);
        vm.set_reg(3, 0xbeef);
        vm.set_reg(13, base + 0x300000);
        let out = vm.call_bounded(func, 1_000_000);
        eprintln!("{:?} vs {:?}: {out:?} r0={:#x} r1={:#x} r2={:#x} r3={:#x}", String::from_utf8_lossy(a), String::from_utf8_lossy(b), vm.get_reg(0), vm.get_reg(1), vm.get_reg(2), vm.get_reg(3));
    }
}
