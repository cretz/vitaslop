//! newlib's `qsort` from a real homebrew image, sorting an integer array through a
//! comparator that is ALSO the image's (a `blx fp` through the dispatcher), against
//! Rust's sort. Exercises `mul`/`mla`, `ands rd, rn, rm, asr #32` + `it lo`, the
//! swap loops with `lr` as a scratch register, and self-recursion.
//!
//! Needs `VITASLOP_HB_IMAGE=<image.bin>:<base hex>`, `VITASLOP_HB_QSORT=<hex>` and
//! `VITASLOP_HB_CMP=<hex>` (a `cmp(const u32*, const u32*)` in the image; Thumb, bit 0
//! clear).

use vitaslop_native::{CallOutcome, HostAbi, Vm, DEFAULT_MEM_BYTES};

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

#[test]
fn newlib_qsort_sorts_integers() {
    let (Some((image, base)), Some(qsort), Some(cmp)) =
        (image(), entry("VITASLOP_HB_QSORT"), entry("VITASLOP_HB_CMP"))
    else {
        eprintln!("VITASLOP_HB_IMAGE/VITASLOP_HB_QSORT/VITASLOP_HB_CMP unset: skipping");
        return;
    };
    if std::env::var("VITASLOP_DUMP_IR").is_ok() {
        let program = vitaslop_transpiler::Program {
            code: &image,
            base,
            thumb: true,
            entries: &[qsort, cmp],
            arm_entries: &[],
            externs: &[],
            redirects: &[],
            inline_imports: &[],
            noreturn_svc: &[],
            mem_bytes: DEFAULT_MEM_BYTES,
            discover_code_pointers: false,
            import_memory: false,
        };
        let d = entry("VITASLOP_HB_DUMP").unwrap_or(qsort); eprintln!("{}", vitaslop_transpiler::dump_func(&program, d).unwrap_or_default());
    }
    let abi = HostAbi::default();
    let mut vm = Vm::new(&image, base, true, &[qsort, cmp], &[], DEFAULT_MEM_BYTES, &abi)
        .expect("build vm");
    let arr = base + 0x2400000;
    if let Ok(t) = vm.read_mem(base + DEFAULT_MEM_BYTES, 16) {
        eprintln!("addr table: {t:02x?}");
    } else {
        eprintln!("addr table unreadable via read_mem");
    }
    let mut failures = Vec::new();
    // A simple LCG so the arrays are reproducible; lengths straddle qsort's
    // insertion-sort cutoff (7) and its median-of-3 / median-of-9 thresholds (40).
    let mut seed = 0x1234_5678u32;
    let mut next = || {
        seed = seed.wrapping_mul(1_103_515_245).wrapping_add(12345);
        (seed >> 8) & 0xffff
    };
    let bit = std::env::var("VITASLOP_HB_CMP_BIT").map(|v| v == "0").unwrap_or(false);
    for n in [0u32, 1, 2, 3, 5, 6, 7, 8, 9, 15, 16, 31, 40, 41, 64, 72, 100, 257] {
        let vals: Vec<u32> = (0..n).map(|_| next()).collect();
        let bytes: Vec<u8> = vals.iter().flat_map(|v| v.to_le_bytes()).collect();
        vm.write_mem(arr, &bytes).unwrap();
        vm.set_reg(0, arr);
        vm.set_reg(1, n);
        vm.set_reg(2, 4);
        vm.set_reg(3, if bit { cmp } else { cmp | 1 });
        vm.set_reg(13, base + 0x2300000);
        match vm.call_bounded(qsort, 50_000_000) {
            CallOutcome::Returned => {}
            other => {
                let regs: Vec<String> = (0..16).map(|i| format!("r{i}={:#x}", vm.get_reg(i))).collect();
                failures.push(format!("n={n}: {other:?}
      {}", regs.join(" ")));
                continue;
            }
        }
        let out = vm.read_mem(arr, (n * 4) as usize).unwrap();
        let got: Vec<u32> =
            out.chunks(4).map(|c| u32::from_le_bytes([c[0], c[1], c[2], c[3]])).collect();
        let mut want = vals.clone();
        want.sort();
        if got != want {
            failures.push(format!("n={n}: got {got:x?}\n      want {want:x?}"));
        }
    }
    assert!(failures.is_empty(), "qsort mis-sorts:\n{}", failures.join("\n"));
}

/// Block trace for one run: set `VITASLOP_TRACE_BLOCKS=<lo>-<hi>` (emit-time) and
/// `VITASLOP_HB_N=<n>`; every block entry prints the register file.
fn trace_svc(
    selector: u32,
    regs: &mut [u32; 16],
    _mem: &mut [u8],
    _base: u32,
    _out: &mut Vec<u8>,
) -> vitaslop_native::SvcOutcome {
    let r: Vec<String> = regs.iter().map(|v| format!("{v:08x}")).collect();
    eprintln!("B {selector:08x} {}", r.join(" "));
    vitaslop_native::SvcOutcome::Continue
}

#[test]
fn qsort_block_trace() {
    let (Some((image, base)), Some(qsort), Some(cmp), Ok(n)) =
        (image(), entry("VITASLOP_HB_QSORT"), entry("VITASLOP_HB_CMP"), std::env::var("VITASLOP_HB_N"))
    else {
        return;
    };
    let n: u32 = n.parse().unwrap();
    let abi = HostAbi { noreturn_svc: &[], svc: trace_svc, import: trace_svc };
    let mut vm = Vm::new(&image, base, true, &[qsort, cmp], &[], DEFAULT_MEM_BYTES, &abi)
        .expect("build vm");
    let arr = base + 0x2400000;
    let mut seed = 0x1234_5678u32;
    let vals: Vec<u32> = (0..n)
        .map(|_| {
            seed = seed.wrapping_mul(1_103_515_245).wrapping_add(12345);
            (seed >> 8) & 0xffff
        })
        .collect();
    let bytes: Vec<u8> = vals.iter().flat_map(|v| v.to_le_bytes()).collect();
    vm.write_mem(arr, &bytes).unwrap();
    vm.set_reg(0, arr);
    vm.set_reg(1, n);
    vm.set_reg(2, 4);
    vm.set_reg(3, cmp | 1);
    vm.set_reg(13, base + 0x2300000);
    let out = vm.call_bounded(qsort, 50_000_000);
    eprintln!("outcome: {out:?}");
}
