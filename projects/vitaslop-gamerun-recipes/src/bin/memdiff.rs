//! `memdiff` - diff guest-memory dumps from two runs that differ only in their
//! input, and report exactly which state the input touched.
//!
//! # Why this is the right instrument
//! The obvious way to find a live value ("hold the throttle and see what changes")
//! does not work on a running game: over any span of frames, thousands of unrelated
//! slots move - timers, animation cursors, particle systems, allocator bumps, audio
//! phase. Measured on a real title, a scan for "changed while the throttle was held"
//! and the same scan with NO throttle held return the same order of magnitude of
//! candidates. The signal is buried in churn, and no amount of narrowing separates
//! them, because both are "things that changed over time".
//!
//! Two RUNS remove the churn completely. The emulator is deterministic, so two runs
//! that replay the same prefix are byte-identical in memory at the branch frame.
//! Let them differ only in what is pressed from there, dump the same region from
//! each at the same frame, and every differing byte is CAUSED by that input. There
//! is no background at all: both runs did the same amount of it.
//!
//! An EMPTY diff is a first-class answer, and often the important one: the input
//! reached no game state whatsoever, so no amount of retiming or holding it longer
//! will help - the problem is elsewhere.
//!
//! Usage:
//!   memdiff <baseline.bin> <variant.bin> [more-variants...] [options]
//!
//! Options:
//!   --limit <N>      report at most N differing clusters (default 40)
//!   --min <N>        ignore clusters shorter than N bytes (default 4)
//!   --gap <N>        merge differing bytes into one cluster when the identical gap
//!                    between them is under N bytes (default 16), so one object with
//!                    a few unchanged fields reads as a single finding
//!   --type <t>       how to print values: u32|i32|f32|hex (default f32)
//!
//! Dumps come from a session's `dump` command; `explore --dump` produces a matched
//! set automatically.

use std::path::{Path, PathBuf};
use std::process::ExitCode;

/// One run of differing bytes, after gap merging.
struct Cluster {
    addr: u32,
    len: usize,
}

fn main() -> ExitCode {
    let args: Vec<String> = std::env::args().collect();
    let mut files: Vec<PathBuf> = Vec::new();
    let mut limit = 40usize;
    let mut min_len = 4usize;
    let mut gap = 16usize;
    let mut ty = "f32".to_string();

    let mut i = 1;
    while i < args.len() {
        let a = args[i].as_str();
        let mut next = || {
            i += 1;
            args.get(i).cloned()
        };
        match a {
            "--limit" => limit = next().and_then(|s| s.parse().ok()).unwrap_or(limit),
            "--min" => min_len = next().and_then(|s| s.parse().ok()).unwrap_or(min_len),
            "--gap" => gap = next().and_then(|s| s.parse().ok()).unwrap_or(gap),
            "--type" => ty = next().unwrap_or(ty),
            "-h" | "--help" => {
                eprintln!("usage: memdiff <baseline.bin> <variant.bin> [more...] [--limit N] [--min N] [--gap N] [--type f32]");
                return ExitCode::from(2);
            }
            other => files.push(PathBuf::from(other)),
        }
        i += 1;
    }

    if files.len() < 2 {
        eprintln!("error: need a baseline dump and at least one variant dump");
        return ExitCode::from(2);
    }
    let (base_addr, base) = match load(&files[0]) {
        Ok(v) => v,
        Err(e) => {
            eprintln!("error: {e}");
            return ExitCode::FAILURE;
        }
    };

    for f in &files[1..] {
        let (addr, other) = match load(f) {
            Ok(v) => v,
            Err(e) => {
                eprintln!("error: {e}");
                return ExitCode::FAILURE;
            }
        };
        // Comparing dumps of different regions would produce confident nonsense.
        if addr != base_addr || other.len() != base.len() {
            eprintln!(
                "error: {} covers {addr:#010x}+{:#x} but the baseline covers {base_addr:#010x}+{:#x}",
                f.display(),
                other.len(),
                base.len()
            );
            return ExitCode::FAILURE;
        }
        let clusters = diff(&base, &other, base_addr, gap, min_len);
        let total: usize = clusters.iter().map(|c| c.len).sum();
        println!(
            "\n=== {} vs {} : {} differing bytes in {} clusters ===",
            files[0].file_stem().unwrap_or_default().to_string_lossy(),
            f.file_stem().unwrap_or_default().to_string_lossy(),
            total,
            clusters.len()
        );
        if clusters.is_empty() {
            println!(
                "IDENTICAL. The two runs reached the same memory state, so this input changed \
                 NOTHING in the guest - not one byte. Do not retime it or hold it longer; it is \
                 not being consumed at all at this point in the game."
            );
            continue;
        }
        for c in clusters.iter().take(limit) {
            println!("{:#010x} +{:<6} {}", c.addr, c.len, values(&base, &other, base_addr, c, &ty));
        }
        if clusters.len() > limit {
            println!("... {} more clusters", clusters.len() - limit);
        }
    }
    ExitCode::SUCCESS
}

/// Read a `VDMP` dump: `(guest base address, bytes)`.
fn load(path: &Path) -> Result<(u32, Vec<u8>), String> {
    let raw = std::fs::read(path).map_err(|e| format!("reading {}: {e}", path.display()))?;
    if raw.len() < 16 || &raw[..4] != b"VDMP" {
        return Err(format!("{} is not a session memory dump", path.display()));
    }
    let addr = u32::from_le_bytes([raw[4], raw[5], raw[6], raw[7]]);
    let len = u64::from_le_bytes(raw[8..16].try_into().unwrap()) as usize;
    if raw.len() != 16 + len {
        return Err(format!("{} is truncated ({} bytes of {len})", path.display(), raw.len() - 16));
    }
    Ok((addr, raw[16..].to_vec()))
}

/// Differing byte runs, merged across gaps shorter than `gap` and filtered to those
/// at least `min_len` bytes long.
fn diff(a: &[u8], b: &[u8], base: u32, gap: usize, min_len: usize) -> Vec<Cluster> {
    let mut out: Vec<Cluster> = Vec::new();
    let mut i = 0;
    while i < a.len() {
        if a[i] == b[i] {
            i += 1;
            continue;
        }
        let start = i;
        let mut end = i + 1;
        let mut j = end;
        // Extend while any difference appears within `gap` bytes.
        while j < a.len() && j - end <= gap {
            if a[j] != b[j] {
                end = j + 1;
            }
            j += 1;
        }
        if end - start >= min_len {
            out.push(Cluster { addr: base.wrapping_add(start as u32), len: end - start });
        }
        i = end.max(start + 1);
    }
    out
}

/// Render a cluster's before/after values, at most a few words wide.
fn values(a: &[u8], b: &[u8], base: u32, c: &Cluster, ty: &str) -> String {
    let off = c.addr.wrapping_sub(base) as usize;
    let words = (c.len / 4).min(4);
    let mut s = String::new();
    for w in 0..words {
        let at = off + w * 4;
        if at + 4 > a.len() {
            break;
        }
        let x = &a[at..at + 4];
        let y = &b[at..at + 4];
        s.push_str(&format!("{} -> {}   ", show(x, ty), show(y, ty)));
    }
    if c.len / 4 > words {
        s.push_str("...");
    }
    s
}

fn show(w: &[u8], ty: &str) -> String {
    let raw = u32::from_le_bytes([w[0], w[1], w[2], w[3]]);
    match ty {
        "u32" => format!("{raw}"),
        "i32" => format!("{}", raw as i32),
        "hex" => format!("{raw:#010x}"),
        _ => format!("{}", f32::from_le_bytes([w[0], w[1], w[2], w[3]])),
    }
}
