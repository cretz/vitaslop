//! Regenerates the machine-checkable lower half of every case: the assembled
//! `[bin]` and the qemu-observed `[out]`. The human-authored top of each file
//! (above the generated marker) is preserved verbatim; only the block below is
//! rewritten, deterministically, so an unchanged case produces no git diff.
//!
//! This runs the ARM toolchain and qemu from `PATH` (override with `ARM_AS`,
//! `ARM_OBJCOPY`, `ARM_LD`, `QEMU`). It knows nothing about how those tools are
//! provided - on a Linux CI runner they come from apt; from a Windows checkout
//! you run this inside WSL. See README.md ("Regenerating").

use std::collections::BTreeMap;
use std::fs;
use std::path::Path;
use std::process::Command;

use serde::Serialize;
use vitaslop_conformance_suite_arm as suite;

fn main() {
    let code = run_all();
    // Best-effort cleanup of this run's scratch dir, whatever the outcome.
    let _ = fs::remove_dir_all(workdir());
    std::process::exit(code);
}

fn run_all() -> i32 {
    let mut changed = 0usize;
    let mut files: Vec<_> = fs::read_dir(suite::cases_dir())
        .expect("read cases dir")
        .flatten()
        .map(|e| e.path())
        .filter(|p| p.extension().is_some_and(|x| x == "toml"))
        .collect();
    files.sort();

    for path in &files {
        match regen_one(path) {
            Ok(true) => {
                changed += 1;
                println!("updated {}", path.display());
            }
            Ok(false) => println!("unchanged {}", path.display()),
            Err(e) => {
                eprintln!("error: {}: {e}", path.display());
                return 1;
            }
        }
    }
    println!("{} case(s), {changed} updated", files.len());
    0
}

/// Human-authored fields (everything above the generated marker).
#[derive(serde::Deserialize)]
struct Human {
    #[allow(dead_code)]
    description: String,
    asm: String,
    mode: Option<String>,
    capture: Option<String>,
    #[serde(rename = "in")]
    input: Option<HumanIn>,
}

#[derive(serde::Deserialize, Default)]
struct HumanIn {
    #[serde(default)]
    regs: BTreeMap<String, i64>,
}

fn regen_one(path: &Path) -> Result<bool, String> {
    let text = fs::read_to_string(path).map_err(|e| e.to_string())?;

    // Keep everything above the marker verbatim; regenerate below it.
    let human_text = match text.split_once(suite::GENERATED_MARK) {
        Some((above, _)) => above,
        None => &text,
    };
    let human: Human = toml::from_str(human_text).map_err(|e| e.to_string())?;

    let mode = human.mode.as_deref().unwrap_or("arm");
    let capture = human.capture.as_deref().unwrap_or("regs");
    let seeds = parse_seeds(&human.input.unwrap_or_default().regs)?;

    // [bin]: the bare user instructions our engine will run.
    let bin = assemble_bin(&human.asm, mode)?;

    // [out]: run the wrapped/seeded program under qemu and observe.
    let gen_out = match capture {
        "regs" => {
            let dump = run_qemu(&harness_regs(&human.asm, mode, &seeds))?;
            out_from_dump(&dump, &seeds)?
        }
        "output" => {
            let stdout = run_qemu(&harness_output(&human.asm, mode, &seeds))?;
            let text =
                String::from_utf8(stdout).map_err(|_| "output is not valid utf-8".to_string())?;
            GenOut {
                regs: BTreeMap::new(),
                flags: None,
                output: Some(text),
            }
        }
        other => return Err(format!("unknown capture {other:?}")),
    };

    // `[bin]` is hand-formatted: the base64 is a wrapped multiline literal, which
    // the toml serializer cannot emit. `[out]` goes through serde for correct
    // string escaping. Both are deterministic.
    let bin_block = format!(
        "[bin]\nbase64 = '''\n{}\n'''\n",
        wrap(&suite::encode_base64(&bin), suite::BASE64_WRAP)
    );
    let out_block = toml::to_string(&OutDoc { out: gen_out }).map_err(|e| e.to_string())?;

    let new_text = format!(
        "{}\n\n{}\n{}\n{}",
        human_text.trim_end(),
        suite::GENERATED_MARK,
        bin_block,
        out_block.trim_end()
    ) + "\n";

    if new_text == text {
        return Ok(false);
    }
    fs::write(path, new_text).map_err(|e| e.to_string())?;
    Ok(true)
}

// --- generated `[out]` block (serialized with serde+toml) ---

#[derive(Serialize)]
struct OutDoc {
    out: GenOut,
}

#[derive(Serialize)]
struct GenOut {
    #[serde(skip_serializing_if = "BTreeMap::is_empty")]
    regs: BTreeMap<String, u32>,
    #[serde(skip_serializing_if = "Option::is_none")]
    flags: Option<GenFlags>,
    #[serde(skip_serializing_if = "Option::is_none")]
    output: Option<String>,
}

#[derive(Serialize)]
struct GenFlags {
    n: bool,
    z: bool,
    c: bool,
    v: bool,
}

/// Wrap a string into lines of at most `width` characters (base64 is ASCII).
fn wrap(s: &str, width: usize) -> String {
    s.as_bytes()
        .chunks(width)
        .map(|c| std::str::from_utf8(c).unwrap())
        .collect::<Vec<_>>()
        .join("\n")
}

/// Registers captured by the harness: r0..r12 and r14.
const CAPTURED: [u8; 14] = [0, 1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12, 14];

fn out_from_dump(dump: &[u8], seeds: &BTreeMap<u8, u32>) -> Result<GenOut, String> {
    // Layout written by the harness epilogue: r14, cpsr, r0..r12 (15 LE words).
    if dump.len() != 60 {
        return Err(format!(
            "expected 60-byte register dump, got {}",
            dump.len()
        ));
    }
    let word = |i: usize| u32::from_le_bytes(dump[i * 4..i * 4 + 4].try_into().unwrap());
    let r14 = word(0);
    let cpsr = word(1);
    let value = |r: u8| -> u32 {
        match r {
            14 => r14,
            _ => word(2 + r as usize),
        }
    };

    // Emit a register if it was seeded or ended non-zero; else it is implicitly 0.
    let mut regs = BTreeMap::new();
    for &r in &CAPTURED {
        let v = value(r);
        if v != 0 || seeds.contains_key(&r) {
            regs.insert(format!("r{r}"), v);
        }
    }

    Ok(GenOut {
        regs,
        flags: Some(GenFlags {
            n: (cpsr >> 31) & 1 != 0,
            z: (cpsr >> 30) & 1 != 0,
            c: (cpsr >> 29) & 1 != 0,
            v: (cpsr >> 28) & 1 != 0,
        }),
        output: None,
    })
}

// --- assembly harnesses ---

fn header(mode: &str) -> String {
    let m = if mode == "thumb" { ".thumb" } else { ".arm" };
    format!(".syntax unified\n.arch armv7-a\n{m}\n.text\n")
}

/// Seed r0..r12 and r14 to their seed value (0 if unseeded), so an unseeded
/// register is a deterministic 0 rather than qemu's startup garbage.
fn seed_all(seeds: &BTreeMap<u8, u32>) -> String {
    let mut s = String::new();
    for &r in &CAPTURED {
        let v = seeds.get(&r).copied().unwrap_or(0);
        s.push_str(&format!("movw r{r}, #{}\n", v & 0xFFFF));
        if v >> 16 != 0 {
            s.push_str(&format!("movt r{r}, #{}\n", (v >> 16) & 0xFFFF));
        }
    }
    s
}

/// Seed only the explicitly-listed registers (for whole-program cases).
fn seed_listed(seeds: &BTreeMap<u8, u32>) -> String {
    let mut s = String::new();
    for (&r, &v) in seeds {
        s.push_str(&format!("movw r{r}, #{}\n", v & 0xFFFF));
        if v >> 16 != 0 {
            s.push_str(&format!("movt r{r}, #{}\n", (v >> 16) & 0xFFFF));
        }
    }
    s
}

/// The bare instructions as our engine runs them (no scaffold): what goes in [bin].
fn assemble_bin(asm: &str, mode: &str) -> Result<Vec<u8>, String> {
    assemble(&format!("{}{}\n", header(mode), asm.trim()))
}

/// Seed, run the instructions, then dump r0..r12/cpsr/r14 via a write syscall.
fn harness_regs(asm: &str, mode: &str, seeds: &BTreeMap<u8, u32>) -> String {
    format!(
        "{}.global _start\n_start:\n{}{}\n\
         push {{r0-r12}}\n\
         mrs r0, cpsr\n\
         push {{r0}}\n\
         push {{r14}}\n\
         mov r0, #1\n\
         mov r1, sp\n\
         mov r2, #60\n\
         mov r7, #4\n\
         svc #0\n\
         mov r0, #0\n\
         mov r7, #1\n\
         svc #0\n",
        header(mode),
        seed_all(seeds),
        asm.trim()
    )
}

/// Seed (listed only), then run a self-exiting program; qemu captures its stdout.
fn harness_output(asm: &str, mode: &str, seeds: &BTreeMap<u8, u32>) -> String {
    format!(
        "{}.global _start\n_start:\n{}{}\n",
        header(mode),
        seed_listed(seeds),
        asm.trim()
    )
}

// --- toolchain (native; tools resolved from PATH) ---

fn tool(env: &str, default: &str) -> String {
    std::env::var(env).unwrap_or_else(|_| default.to_string())
}

fn workdir() -> std::path::PathBuf {
    std::env::temp_dir().join("vitaslop-regen")
}

fn assemble(src: &str) -> Result<Vec<u8>, String> {
    let dir = workdir();
    fs::create_dir_all(&dir).map_err(|e| e.to_string())?;
    let s = dir.join("bin.s");
    let o = dir.join("bin.o");
    let b = dir.join("bin.bin");
    fs::write(&s, src).map_err(|e| e.to_string())?;
    run(
        &tool("ARM_AS", "arm-none-eabi-as"),
        &["-march=armv7-a", path(&s), "-o", path(&o)],
    )?;
    run(
        &tool("ARM_OBJCOPY", "arm-none-eabi-objcopy"),
        &["-O", "binary", path(&o), path(&b)],
    )?;
    fs::read(&b).map_err(|e| e.to_string())
}

fn run_qemu(src: &str) -> Result<Vec<u8>, String> {
    let dir = workdir();
    fs::create_dir_all(&dir).map_err(|e| e.to_string())?;
    let s = dir.join("h.s");
    let o = dir.join("h.o");
    let elf = dir.join("h.elf");
    fs::write(&s, src).map_err(|e| e.to_string())?;
    run(
        &tool("ARM_AS", "arm-none-eabi-as"),
        &["-march=armv7-a", path(&s), "-o", path(&o)],
    )?;
    run(
        &tool("ARM_LD", "arm-none-eabi-ld"),
        &["-static", "-e", "_start", path(&o), "-o", path(&elf)],
    )?;

    let out = Command::new(tool("QEMU", "qemu-arm"))
        .arg(path(&elf))
        .output()
        .map_err(|e| format!("qemu spawn: {e}"))?;
    if !out.status.success() {
        return Err(format!(
            "qemu failed ({}): {}",
            out.status,
            String::from_utf8_lossy(&out.stderr)
        ));
    }
    Ok(out.stdout)
}

fn path(p: &Path) -> &str {
    p.to_str().expect("utf-8 path")
}

fn run(cmd: &str, args: &[&str]) -> Result<(), String> {
    let out = Command::new(cmd)
        .args(args)
        .output()
        .map_err(|e| format!("spawn {cmd}: {e}"))?;
    if !out.status.success() {
        return Err(format!(
            "{cmd} failed ({}): {}",
            out.status,
            String::from_utf8_lossy(&out.stderr)
        ));
    }
    Ok(())
}

fn parse_seeds(m: &BTreeMap<String, i64>) -> Result<BTreeMap<u8, u32>, String> {
    m.iter()
        .map(|(k, v)| {
            let i = suite::reg_index(k).ok_or_else(|| format!("bad register {k:?}"))?;
            if i == 13 || i == 15 {
                return Err(format!("register {k} is not seedable (harness uses sp/pc)"));
            }
            Ok((i, *v as u32))
        })
        .collect()
}
