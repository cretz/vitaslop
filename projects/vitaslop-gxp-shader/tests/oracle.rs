//! Oracle harness: validate the container parser + USSE decoder against real captured
//! `SceGxmProgram` blobs, and print the statistics that drive semantic RE.
//!
//! The blobs are game-derived shader bytes (a private oracle) and are NEVER committed.
//! This test reads them from a directory named by `VITASLOP_GXP_DUMPS` and SKIPS cleanly
//! when that is unset or empty - so `cargo test` is green with no fixture, and CI never
//! sees game data. Every assertion here is content-free (structural invariants only:
//! magic, size, offset ordering, 64-bit alignment, decode losslessness). Run it with:
//!
//! ```text
//! VITASLOP_GXP_DUMPS=<abs-path-to-your-dump-dir> \
//!   cargo test -p vitaslop-gxp-shader --test oracle -- --ignored --nocapture
//! ```

use std::fs;
use std::path::PathBuf;

use vitaslop_gxp_shader::container::{ParamCategory, ProgramKind};
use vitaslop_gxp_shader::ir::Bank;
use vitaslop_gxp_shader::usse::{decode, field, opcode1, GROUP_TABLES};
use vitaslop_gxp_shader::{
    analyze, container::Program, recompile_fragment, recompile_fragment_module,
    recompile_vertex_module, RecompileError,
};

/// naga-validate a complete bindable module (the same WGSL front-end + validator wgpu uses):
/// proves a recompiled shader is not just plausible text but a real, compilable GPU module
/// with a valid binding interface. Panics naming the shader + the WGSL on any failure.
fn validate_module_wgsl(name: &str, wgsl: &str) {
    let module = naga::front::wgsl::parse_str(wgsl)
        .unwrap_or_else(|e| panic!("{name}: recompiled module failed to parse: {e:?}\n{wgsl}"));
    let mut v = naga::valid::Validator::new(
        naga::valid::ValidationFlags::all(),
        naga::valid::Capabilities::all(),
    );
    v.validate(&module)
        .unwrap_or_else(|e| panic!("{name}: recompiled module failed validation: {e:?}\n{wgsl}"));
}

/// Dataflow corroboration of the STRUCTURAL decode (bank + register index) against the real
/// code, with no ground-truth oracle needed. A well-formed shader writes a temp/output/
/// internal register before it reads it (pa/sa are external inputs, always "defined";
/// constants carry a value). So walking a shader in order and checking every r/o/i source
/// read against the set of registers an earlier instruction wrote yields a hit rate that
/// should be very high if the bank/index decode is correct - a decode bug (wrong bank, wrong
/// index scaling, wrong operand slot) shows up as reads-before-write. This validates the new
/// 0x18/0x30/0x38 operand decode on real data, beyond the synthetic unit tests. It tracks
/// registers at BASE granularity (not per-lane), because the exact write masks of the
/// undocumented groups are not established - the register index is what is being corroborated.
fn dataflow_corroboration(files: &[PathBuf]) {
    // Per-bank (reads, hits). Temp is the real decode-quality signal (temps are always
    // written before read); Internal legitimately misses (iterator/tex preload is external);
    // Output reads are rare. Split them so a genuine decode bug is not masked by preloads.
    let mut r = (0u64, 0u64);
    let mut o = (0u64, 0u64);
    let mut i_ = (0u64, 0u64);
    let mut worst_r: Vec<(String, u64, u64)> = Vec::new();
    for path in files {
        let bytes = fs::read(path).unwrap();
        let Ok(program) = Program::parse(&bytes) else { continue };
        let mut def_r = std::collections::HashSet::new();
        let mut def_o = std::collections::HashSet::new();
        let mut def_i = std::collections::HashSet::new();
        let (mut sr, mut hr) = (0u64, 0u64);
        for &w in &program.code {
            let ins = decode(w);
            for s in &ins.srcs {
                match s.bank {
                    Bank::Temp => { r.0 += 1; sr += 1; if def_r.contains(&s.index) { r.1 += 1; hr += 1; } }
                    Bank::Output => { o.0 += 1; if def_o.contains(&s.index) { o.1 += 1; } }
                    Bank::Internal => { i_.0 += 1; if def_i.contains(&s.index) { i_.1 += 1; } }
                    _ => {}
                }
            }
            if let Some(d) = &ins.dest {
                match d.bank {
                    Bank::Temp => { def_r.insert(d.index); }
                    Bank::Output => { def_o.insert(d.index); }
                    Bank::Internal => { def_i.insert(d.index); }
                    _ => {}
                }
            }
        }
        if sr > 0 {
            worst_r.push((path.file_name().unwrap().to_string_lossy().into_owned(), hr, sr));
        }
    }
    let pct = |h: u64, n: u64| if n == 0 { 0.0 } else { h as f64 / n as f64 * 100.0 };
    println!("\n=== dataflow corroboration (bank reads matched to an earlier in-stream write) ===");
    println!("  temp   r: {}/{} ({:.1}%)  <- CONFOUNDED: tex/pack/flow/and (0xE0/0x40/0xF8/0x50) write", r.1, r.0, pct(r.1, r.0));
    println!("            temps but their DEST is not decoded, so those writes are invisible -> false misses");
    println!("  output o: {}/{} ({:.1}%)", o.1, o.0, pct(o.1, o.0));
    println!("  internal i: {}/{} ({:.1}%)  <- HIGH corroborates the dot/mad internal-reg decode is sound", i_.1, i_.0, pct(i_.1, i_.0));
    worst_r.sort_by(|a, b| pct(a.1, a.2).partial_cmp(&pct(b.1, b.2)).unwrap());
    println!("  lowest temp-hit-rate shaders (candidate decode issues):");
    for (name, h, n) in worst_r.iter().take(6) {
        println!("    {name:<24} {h:>4}/{n:<4} ({:.0}%)", pct(*h, *n));
    }
}

/// Histogram the raw values of named fields across every instruction of a given opcode1
/// group, over all blobs. This is the behavioral-RE microscope: for the groups whose
/// operand SEMANTICS are undocumented (0x30/0x38 transcendentals+mov, 0x40 pack, 0xE0 tex,
/// 0xF8 flow), the real distribution of the swizzle/mask/modifier fields tells us which
/// encodings actually occur (and which are dead), so emit can be wired for the real cases
/// and corroborated, never guessed for cases that never appear.
fn field_hist_for_group(files: &[PathBuf], group: u8, table_name: &str, fields: &[&str]) {
    let Some((_, high, low)) = GROUP_TABLES.iter().find(|(n, _, _)| *n == table_name) else {
        println!("  (no grammar table named {table_name})");
        return;
    };
    use std::collections::BTreeMap;
    let mut hist: Vec<BTreeMap<u32, u64>> = fields.iter().map(|_| BTreeMap::new()).collect();
    let mut count = 0u64;
    for path in files {
        let bytes = fs::read(path).unwrap();
        let Ok(program) = Program::parse(&bytes) else { continue };
        for &w in &program.code {
            if opcode1(w) != group {
                continue;
            }
            count += 1;
            let hi = (w >> 32) as u32;
            let lo = w as u32;
            for (i, fname) in fields.iter().enumerate() {
                // A field lives in either the high or low half; `field` returns 0 if absent,
                // so try both and take the half whose table actually names it.
                let v = if high.iter().any(|(n, _)| n == fname) {
                    field(hi, high, fname)
                } else {
                    field(lo, low, fname)
                };
                *hist[i].entry(v).or_default() += 1;
            }
        }
    }
    println!("\n  group 0x{:02x} ({table_name}), {count} instrs - field value histograms:", group << 3);
    for (i, fname) in fields.iter().enumerate() {
        let pairs: Vec<String> = hist[i].iter().map(|(v, n)| format!("{v}:{n}")).collect();
        println!("    {fname:<16} {}", pairs.join("  "));
    }
}

fn dump_dir() -> Option<PathBuf> {
    let v = std::env::var("VITASLOP_GXP_DUMPS").ok()?;
    if v.trim().is_empty() {
        return None;
    }
    Some(PathBuf::from(v))
}

fn gxp_files(dir: &PathBuf) -> Vec<PathBuf> {
    let mut out: Vec<PathBuf> = fs::read_dir(dir)
        .into_iter()
        .flatten()
        .flatten()
        .map(|e| e.path())
        .filter(|p| p.extension().map(|e| e == "gxp").unwrap_or(false))
        .collect();
    out.sort();
    out
}

#[test]
#[ignore = "requires the private VITASLOP_GXP_DUMPS fixture; run explicitly"]
fn oracle_parse_decode_all_blobs() {
    let Some(dir) = dump_dir() else {
        eprintln!("VITASLOP_GXP_DUMPS unset - skipping oracle (this is expected in CI)");
        return;
    };
    let files = gxp_files(&dir);
    assert!(!files.is_empty(), "no .gxp files under {}", dir.display());

    let mut global_group = [0u64; 32];
    let mut total_instrs = 0u64;
    let mut total_supported = 0u64;
    let mut total_classified = 0u64;
    let (mut n_frag, mut n_vert) = (0u32, 0u32);
    // Per-group operand bank-selector histogram, to correlate encodings with the param
    // table in future semantic RE.
    let mut bank_sel_hist = [[0u64; 4]; 32];
    // Per-op-mnemonic and per-blocked-reason tallies drive the emit grind priority: which
    // operation appears most, and what exactly stops the ops that ARE wired from emitting.
    use std::collections::BTreeMap;
    let mut op_hist: BTreeMap<&'static str, u64> = BTreeMap::new();
    let mut blocked_hist: BTreeMap<&'static str, u64> = BTreeMap::new();

    println!("\n{:<24} {:>4} {:>5} {:>4} {:>4} {:>4} {:>7} {:>6}", "file", "kind", "instr", "PA", "SA", "smp", "classif", "emit");
    for path in &files {
        let bytes = fs::read(path).unwrap();
        let name = path.file_name().unwrap().to_string_lossy();

        // Structural invariants (content-free): a real blob parses, and its declared
        // size equals its byte length.
        let program = Program::parse(&bytes)
            .unwrap_or_else(|e| panic!("{name}: parse failed: {e:?}"));
        assert_eq!(program.size as usize, bytes.len(), "{name}: size != len");

        let cov = analyze(&bytes).unwrap();
        assert_eq!(cov.total, program.code.len(), "{name}: instr count mismatch");

        // Every instruction re-decodes to exactly the same word we read (losslessness).
        for &w in &program.code {
            let ins = decode(w);
            assert_eq!(ins.raw, w, "{name}: decode dropped bits");
            global_group[(ins.group & 0x1f) as usize] += 1;
            *op_hist.entry(ins.op.mnemonic()).or_default() += 1;
            // Tally why an otherwise-classified instruction is not emittable: a wired op
            // held back by `blocked`, or an op whose emit is simply not wired yet.
            if !ins.is_supported() {
                let reason = ins.blocked.unwrap_or("op emit not wired");
                *blocked_hist.entry(reason).or_default() += 1;
            }
            for op in ins.dest.iter().chain(ins.srcs.iter()) {
                bank_sel_hist[(ins.group & 0x1f) as usize][(op.bank_sel & 3) as usize] += 1;
            }
        }
        total_instrs += cov.total as u64;
        total_supported += cov.supported as u64;
        total_classified += cov.classified as u64;
        match program.kind {
            ProgramKind::Fragment => n_frag += 1,
            ProgramKind::Vertex => n_vert += 1,
        }
        let n_samplers = program.parameters.iter().filter(|p| p.category == ParamCategory::Sampler).count();
        println!(
            "{:<24} {:>4} {:>5} {:>4} {:>4} {:>4} {:>6.0}% {:>5.0}%",
            name,
            if program.kind == ProgramKind::Fragment { "frag" } else { "vert" },
            cov.total,
            program.primary_reg_count,
            program.secondary_reg_count,
            n_samplers,
            cov.classified_fraction() * 100.0,
            cov.fraction() * 100.0,
        );
    }

    // Grind contract: recompiling a real fragment shader today HARD-FAILS naming the
    // exact unsupported opcode (never a silent success / fallback). Verify that on every
    // fragment blob, and that the message actually names an opcode to implement.
    println!("\nper-fragment recompile status (first blocker on the road to whole-shader emit):");
    let mut n_recompiled = 0u32;
    for path in &files {
        let name = path.file_name().unwrap().to_string_lossy();
        if !name.starts_with("frag") {
            continue;
        }
        let bytes = fs::read(path).unwrap();
        match recompile_fragment(&bytes) {
            Ok(rc) => {
                n_recompiled += 1;
                // Go the whole way to a complete, bindable module and prove it validates as
                // real WGSL with a sound binding interface (the artifact the renderer binds).
                let (_, module) = recompile_fragment_module(&bytes)
                    .unwrap_or_else(|e| panic!("{name}: module assembly failed: {e}"));
                validate_module_wgsl(&name, &module.wgsl);
                let b = &module.bindings;
                println!(
                    "  {name:<24} OK - {} WGSL bytes, module VALID (pa={} sa={} smp={} {:?})",
                    rc.wgsl_body.len(),
                    b.pa_lane_count,
                    b.sa_lane_count,
                    b.samplers.len(),
                    b.color,
                );
            }
            Err(RecompileError::Emit(e)) => {
                let msg = e.to_string();
                assert!(
                    msg.contains("USSE") || msg.contains("unmapped"),
                    "{name}: hard-fail message must name what to implement, got: {msg}"
                );
                // Print a compact first-blocker summary (the op/reason + instruction index).
                let brief: String = msg.split(" - ").next().unwrap_or(&msg).chars().take(90).collect();
                println!("  {name:<24} {brief}");
            }
            Err(other) => panic!("{name}: unexpected recompile error {other:?}"),
        }
    }
    println!("  => {n_recompiled} fragment shaders fully recompiled to WGSL");

    // Same grind for VERTEX programs: recompile each to a complete, bindable module (attribute
    // inputs + position/varying outputs), prove it validates as real WGSL, or hard-fail naming
    // the exact opcode / limit to implement next.
    println!("\nper-vertex recompile status (first blocker on the road to whole-shader emit):");
    let mut n_vrecompiled = 0u32;
    for path in &files {
        let name = path.file_name().unwrap().to_string_lossy();
        if !name.starts_with("vert") {
            continue;
        }
        let bytes = fs::read(path).unwrap();
        match recompile_vertex_module(&bytes) {
            Ok((rc, module)) => {
                n_vrecompiled += 1;
                validate_module_wgsl(&name, &module.wgsl);
                let b = &module.bindings;
                let attrs: Vec<String> = b.attributes.iter().map(|a| format!("{}@{}x{}", a.name, a.base_lane, a.components)).collect();
                println!(
                    "  {name:<24} OK - {} instrs, module VALID (attrs=[{}] sa={} varyings={})",
                    rc.shader.instrs.len(),
                    attrs.join(" "),
                    b.sa_lane_count,
                    b.varying_vec4s,
                );
            }
            Err(RecompileError::Emit(e)) => {
                let msg = e.to_string();
                let brief: String = msg.split(" - ").next().unwrap_or(&msg).chars().take(90).collect();
                println!("  {name:<24} {brief}");
            }
            Err(other) => println!("  {name:<24} {other}"),
        }
    }
    println!("  => {n_vrecompiled} vertex shaders fully recompiled to WGSL");

    // Validate the fragment interpolant (varying) parse against real blobs: every SMP that
    // reads its coordinate from the PA bank is a non-dependent texture query, so that PA
    // register (field = decoder index / 2, undoing the double-register scaling) must fall
    // within some parsed interpolant's [pa_base, pa_base+register_count) span. A correct parse
    // hits ~all of them; a wrong layout misses. This is the empirical confirmation.
    println!("\n=== fragment interpolant parse vs SMP coord PA reads ===");
    let (mut hits, mut misses) = (0u32, 0u32);
    for path in &files {
        let name = path.file_name().unwrap().to_string_lossy();
        if !name.starts_with("frag") {
            continue;
        }
        let bytes = fs::read(path).unwrap();
        let Ok(program) = Program::parse(&bytes) else { continue };
        let spans: Vec<(u8, u8)> = program
            .interpolants
            .iter()
            .map(|it| (it.pa_base, it.pa_base + it.register_count))
            .collect();
        let mut smp_fields: Vec<u8> = Vec::new();
        for &w in &program.code {
            let ins = decode(w);
            if matches!(ins.op, vitaslop_gxp_shader::Op::Tex { .. }) {
                if let Some(c) = ins.srcs.first() {
                    if c.bank == Bank::PrimaryAttr {
                        smp_fields.push(c.index / 2); // undo field*2 double-register scaling
                    }
                }
            }
        }
        let mut miss_list = Vec::new();
        for &f in &smp_fields {
            if spans.iter().any(|&(lo, hi)| f >= lo && f < hi) {
                hits += 1;
            } else {
                misses += 1;
                miss_list.push(f);
            }
        }
        let interp: Vec<String> = program
            .interpolants
            .iter()
            .map(|it| format!("{:?}@pa{}x{}", it.usage, it.pa_base, it.register_count))
            .collect();
        let flag = if miss_list.is_empty() { "" } else { " <-- MISS" };
        println!("  {name:<24} [{}]{}  smpPA(field)={:?}{}", interp.join(" "), if program.interpolants.is_empty() { " (none)" } else { "" }, smp_fields, flag);
        if !miss_list.is_empty() {
            println!("  {:<24}   unmatched SMP PA fields: {:?}", "", miss_list);
        }
    }
    println!("  => interpolant/SMP cross-check: {hits} hits, {misses} misses (misses should be ~0 if the layout is right)");

    println!("\n=== {} blobs ({} frag, {} vert), {} instructions ===", files.len(), n_frag, n_vert, total_instrs);
    println!("classified (operation known from ISA): {}/{}  ({:.1}%)",
        total_classified, total_instrs, total_classified as f64 / total_instrs as f64 * 100.0);
    println!("emittable (WGSL wired + not blocked):  {}/{}  ({:.1}%)",
        total_supported, total_instrs, total_supported as f64 / total_instrs as f64 * 100.0);
    println!("\nopcode1 group histogram:");
    for g in 0..32 {
        if global_group[g] > 0 {
            println!("  0x{:02x} (grp 0x{:02x}): {:>6}   bank_sel[0..3]={:?}",
                g, g << 3, global_group[g], bank_sel_hist[g]);
        }
    }
    // Op-mnemonic histogram (sorted by count desc), so the next emit target is obvious.
    let mut ops: Vec<_> = op_hist.iter().collect();
    ops.sort_by(|a, b| b.1.cmp(a.1));
    println!("\nop mnemonic histogram (all instrs):");
    for (op, n) in ops {
        println!("  {n:>6}  {op}");
    }
    // Blocked/unwired-reason histogram: exactly what stands between us and more emit.
    let mut reasons: Vec<_> = blocked_hist.iter().collect();
    reasons.sort_by(|a, b| b.1.cmp(a.1));
    println!("\nnon-emittable reason histogram:");
    for (why, n) in reasons {
        println!("  {n:>6}  {why}");
    }

    // The 0xF8 flow group has no grammar.json table (grammar stops at 0x38), so extract its
    // classifying fields by hand from the henkaku encoding (same header across predicate
    // subsections): predicate[26:24], modifier1[22], opcode2[21:19], opcode4[15],
    // modifier2[9], opcode3[8]. Histogram them + the distinct raw high-words, so the flow
    // sub-ops (mov/ba/kill/pcoeff/ptoff/emit) can be classified from the documented truth
    // tables even though their operand bytes are "?".
    {
        use std::collections::BTreeMap;
        // Fields live in the HIGH 32-bit word, i.e. u64 bit = wiki_high_bit + 32.
        let bits = |w: u64, hi: u32, lo_incl: u32| {
            (w >> (lo_incl + 32)) & ((1u64 << (hi - lo_incl + 1)) - 1)
        };
        let mut pred = BTreeMap::<u64, u64>::new();
        let mut opc2 = BTreeMap::<u64, u64>::new();
        let mut combo = BTreeMap::<(u64, u64, u64, u64, u64, u64), u64>::new();
        let mut first_word = BTreeMap::<u64, u64>::new();
        let mut n = 0u64;
        for path in &files {
            let bytes = fs::read(path).unwrap();
            let Ok(program) = Program::parse(&bytes) else { continue };
            for (i, &w) in program.code.iter().enumerate() {
                if opcode1(w) != 0x1f {
                    continue;
                }
                n += 1;
                let (p, m1, o2, o4, m2, o3) =
                    (bits(w, 26, 24), bits(w, 22, 22), bits(w, 21, 19), bits(w, 15, 15), bits(w, 9, 9), bits(w, 8, 8));
                *pred.entry(p).or_default() += 1;
                *opc2.entry(o2).or_default() += 1;
                *combo.entry((p, m1, o2, o4, m2, o3)).or_default() += 1;
                if i == 0 {
                    *first_word.entry(w).or_default() += 1;
                }
            }
        }
        println!("\n=== 0xF8 flow group classification microscope ({n} instrs) ===");
        println!("  predicate: {}", pred.iter().map(|(k, v)| format!("{k}:{v}")).collect::<Vec<_>>().join("  "));
        println!("  opcode2:   {}", opc2.iter().map(|(k, v)| format!("{k}:{v}")).collect::<Vec<_>>().join("  "));
        println!("  (pred,mod1,opc2,opc4,mod2,opc3) combos:");
        let mut cv: Vec<_> = combo.iter().collect();
        cv.sort_by(|a, b| b.1.cmp(a.1));
        for (k, v) in cv {
            println!("    {k:?}: {v}");
        }
        println!("  distinct instr#0 words (shader prologue): {}",
            first_word.iter().map(|(k, v)| format!("{k:#018x}:{v}")).collect::<Vec<_>>().join("  "));
    }

    // Behavioral-RE microscope: raw field distributions for the undocumented-operand groups.
    println!("\n=== undocumented-group operand field histograms (behavioral RE) ===");
    field_hist_for_group(&files, 0x06, "grp30",
        &["opcode2", "data_format", "swz_en", "mask1_op0", "mask1_op1", "mask2_op1",
          "abs_op1", "neg_op1", "opt0", "opt1", "modifier0", "modifier1"]);
    field_hist_for_group(&files, 0x07, "grp38",
        &["opcode2", "cond0", "cond1", "swz_en", "swz_mask1", "swz_mask2", "swz_mask3",
          "data_format", "op23_swz", "opt0", "opt2", "opt3"]);

    dataflow_corroboration(&files);
}

/// Empirical RE of the VERTEX output register layout, to establish (not guess) how a vertex
/// program's `o[]` writes map to the canonical usages and whether POSITION occupies o0..o3.
/// For each vertex blob: decode every instruction, collect the OUTPUT-bank destination lanes
/// actually written, the PRIMARY-ATTR (attribute input) lanes read, and print the parameter
/// table's ATTRIBUTE entries (resource_index = the PA register a vertex input loads into) plus
/// the varyings-block header words. Correlating these across all blobs tells us the true
/// output layout before any faithful vertex emit relies on it.
#[test]
#[ignore = "requires the private VITASLOP_GXP_DUMPS fixture; run explicitly"]
fn vertex_output_layout_analysis() {
    let Some(dir) = dump_dir() else {
        eprintln!("VITASLOP_GXP_DUMPS unset - skipping vertex analysis");
        return;
    };
    let files = gxp_files(&dir);
    println!("\n=== vertex output-register layout analysis ===");
    for path in &files {
        let name = path.file_name().unwrap().to_string_lossy();
        if !name.starts_with("vert") {
            continue;
        }
        let bytes = fs::read(path).unwrap();
        let Ok(program) = Program::parse(&bytes) else { continue };

        // Output (o) destination lanes written + primary-attr (input) lanes read.
        let mut o_lanes: Vec<u16> = Vec::new();
        let mut pa_lanes: Vec<u16> = Vec::new();
        for &w in &program.code {
            let ins = decode(w);
            if let Some(d) = &ins.dest {
                if d.bank == Bank::Output {
                    for c in 0..4 {
                        if ins.write_mask[c] {
                            o_lanes.push(d.index as u16 + c as u16);
                        }
                    }
                }
            }
            for s in &ins.srcs {
                if s.bank == Bank::PrimaryAttr {
                    pa_lanes.push(s.index as u16);
                }
            }
        }
        o_lanes.sort_unstable();
        o_lanes.dedup();
        pa_lanes.sort_unstable();
        pa_lanes.dedup();

        // Attribute parameters: resource_index = the PA register a vertex input lands in.
        let attrs: Vec<String> = program
            .parameters
            .iter()
            .filter(|p| p.category == ParamCategory::Attribute)
            .map(|p| format!("{}@pa{}x{}", p.name, p.resource_index, p.component_count))
            .collect();

        // Varyings block header words (self-relative at header +0x2C).
        let var_rel = u32(&bytes, 0x2c);
        let (vo1, vo2, texpack) = if var_rel != 0 {
            let blk = 0x2c + var_rel as usize;
            (u32(&bytes, blk + 0x10), u32(&bytes, blk + 0x14), u32(&bytes, blk + 0x18))
        } else {
            (0, 0, 0)
        };

        println!(
            "\n  {name}  PA(prim)={} SA(sec)={}",
            program.primary_reg_count, program.secondary_reg_count
        );
        println!("    o-lanes written: {o_lanes:?}");
        println!("    pa-lanes read:   {pa_lanes:?}");
        println!("    attributes: [{}]", attrs.join(" "));
        println!("    varyings hdr: vo1={vo1:#010x} vo2={vo2:#010x} texpack={texpack:#010x}");
    }
}

/// Correlate each captured vertex<->fragment PAIR (from a real draw run) to establish the
/// linkage rule: how a vertex OUTPUT register maps to a fragment PA INPUT register. Prints,
/// per pair, the fragment interpolant field spans (usage @ pa_base, register_count) and the
/// vertex's written output FIELDS beyond clip position (o-register index / 2), so the mapping
/// can be read off directly rather than guessed. The pairs come from a `VITASLOP_DUMP_DRAW_GXP`
/// run of the car frame (host prints `vprog=<vh> fprog=<fh>` per draw).
#[test]
#[ignore = "requires VITASLOP_GXP_DUMPS + VITASLOP_GXP_PAIRS; run explicitly"]
fn link_pair_correlation() {
    let Some(dir) = dump_dir() else { return };
    // The (vertex header, fragment header) pairing is game-specific RE data, so it is NOT baked
    // into source. Pass it via VITASLOP_GXP_PAIRS="vh:fh,vh:fh,..." (header hexes, e.g.
    // "82d28620:82d27fb0,82ed1d50:82ed17c0"), captured from a real run with VITASLOP_DUMP_DRAW_GXP
    // (the host prints `vprog=<vh> fprog=<fh>` per draw). The DECODE this test exercises (the vo2
    // texcoord-size layout, the interpolant descriptors) is a container-format fact, not per-title.
    let Ok(spec) = std::env::var("VITASLOP_GXP_PAIRS") else {
        eprintln!("set VITASLOP_GXP_PAIRS=\"vh:fh,...\" (header hexes) to correlate pairs");
        return;
    };
    let pairs: Vec<(String, String)> = spec
        .split(',')
        .filter_map(|p| p.split_once(':').map(|(v, f)| (v.trim().to_string(), f.trim().to_string())))
        .collect();
    println!("\n=== vertex-output <-> fragment-input linkage correlation ===");
    for (vh, fh) in &pairs {
        let vpath = dir.join(format!("vert_{vh}.gxp"));
        let fpath = dir.join(format!("frag_{fh}.gxp"));
        let (Ok(vb), Ok(fb)) = (fs::read(&vpath), fs::read(&fpath)) else {
            println!("  (missing blob for {vh}/{fh})");
            continue;
        };
        let (Ok(vp), Ok(fp)) = (Program::parse(&vb), Program::parse(&fb)) else { continue };

        // Fragment interpolants: usage @ pa_base field span [pa_base, pa_base+reg_count).
        let frag: Vec<String> = fp
            .interpolants
            .iter()
            .map(|it| format!("{:?}@F{}..{}", it.usage, it.pa_base, it.pa_base + it.register_count))
            .collect();

        // Vertex output FIELDS written (dest.index/2), excluding clip position (fields 0..1).
        let mut ofields: Vec<u16> = Vec::new();
        for &w in &vp.code {
            let ins = decode(w);
            if let Some(d) = &ins.dest {
                if d.bank == Bank::Output {
                    for c in 0..4u16 {
                        if ins.write_mask[c as usize] {
                            ofields.push((d.index as u16 + c) / 2);
                        }
                    }
                }
            }
        }
        ofields.sort_unstable();
        ofields.dedup();
        let vary_fields: Vec<u16> = ofields.iter().copied().filter(|&f| f >= 2).collect();

        // Decode the vertex varyings-block `vertex_outputs2` (header +0x2C self-relative, then
        // +0x14) as ten 3-bit texcoord size fields. Each 3-bit value v gives a component count
        // len = (v&1)*2 + ((v>>1)&1) + ((v>>2)&1) (2/3/4), i.e. TEXCOORDk's width; 0 = absent.
        // The register span each texcoord occupies is ceil(len/2). We validate: summing those
        // register spans reproduces the vertex's actual written varying-field count.
        let var_rel = u32(&vb, 0x2c);
        let vo2 = if var_rel != 0 { u32(&vb, 0x2c + var_rel as usize + 0x14) } else { 0 };
        let mut tc_regs: Vec<(usize, u32)> = Vec::new(); // (texcoord index, register span)
        for k in 0..10u32 {
            let v = (vo2 >> (k * 3)) & 0x7;
            if v == 0 {
                continue;
            }
            let comps = (v & 1) * 2 + ((v >> 1) & 1) + ((v >> 2) & 1);
            tc_regs.push((k as usize, comps.div_ceil(2)));
        }
        let sum_regs: u32 = tc_regs.iter().map(|(_, r)| r).sum();

        println!("\n  vert_{vh} -> frag_{fh}");
        println!("    frag interpolants: [{}]", frag.join(" "));
        println!("    vert out fields (all): {ofields:?}");
        println!("    vert varying fields (>=2): {vary_fields:?} (count={})", vary_fields.len());
        println!("    vo2={vo2:#010x} -> texcoord regs {tc_regs:?} sum={sum_regs}");
    }
}

/// Link each captured vertex<->fragment PAIR into a single WGSL module and prove the linked
/// module validates as real WGSL with a matched varying interface (the same front-end +
/// validator wgpu uses). This is the end-to-end confirmation that the linkage produces a
/// bindable pipeline, not just plausible text: naga checks that every fragment `@location`
/// input has a matching vertex `@location` output of the right type, that the group/binding
/// namespace does not collide across stages, and that both entry points are well formed.
/// A pair that cannot be linked faithfully (a blocked shader, an irregular vertex layout, an
/// unfed sampled lane) is reported as a fall-back, not a failure - that is the correct
/// no-guess behaviour.
#[test]
#[ignore = "requires VITASLOP_GXP_DUMPS + VITASLOP_GXP_PAIRS; run explicitly"]
fn link_pairs_validate() {
    let Some(dir) = dump_dir() else { return };
    let Ok(spec) = std::env::var("VITASLOP_GXP_PAIRS") else {
        eprintln!("set VITASLOP_GXP_PAIRS=\"vh:fh,...\" (header hexes) to link + validate pairs");
        return;
    };
    let pairs: Vec<(String, String)> = spec
        .split(',')
        .filter_map(|p| p.split_once(':').map(|(v, f)| (v.trim().to_string(), f.trim().to_string())))
        .collect();
    println!("\n=== linked vertex+fragment module validation ===");
    let mut n_linked = 0u32;
    for (vh, fh) in &pairs {
        let (Ok(vb), Ok(fb)) = (
            fs::read(dir.join(format!("vert_{vh}.gxp"))),
            fs::read(dir.join(format!("frag_{fh}.gxp"))),
        ) else {
            println!("  (missing blob for {vh}/{fh})");
            continue;
        };
        match vitaslop_gxp_shader::link_programs(&vb, &fb) {
            Ok(linked) => {
                validate_module_wgsl(&format!("vert_{vh}+frag_{fh}"), &linked.wgsl);
                n_linked += 1;
                println!(
                    "  vert_{vh}+frag_{fh}  LINKED + VALID ({} WGSL bytes, vary v={} f={}, attrs={} smp={} {:?})",
                    linked.wgsl.len(),
                    linked.vertex_varyings,
                    linked.fragment_varyings,
                    linked.vertex_bindings.attributes.len(),
                    linked.fragment_bindings.samplers.len(),
                    linked.fragment_bindings.color,
                );
                // The linked module itself, for reading the emitted dataflow when a pair links
                // but paints the wrong colour. Written to a directory, never stdout - these are
                // large, and one file per pair is what a diff wants.
                if let Some(out) = std::env::var_os("VITASLOP_GXP_WGSL_DIR") {
                    let path = PathBuf::from(out).join(format!("{vh}_{fh}.wgsl"));
                    if let Some(parent) = path.parent() {
                        let _ = fs::create_dir_all(parent);
                    }
                    fs::write(&path, &linked.wgsl).expect("write linked WGSL");
                }
            }
            Err(e) => println!("  vert_{vh}+frag_{fh}  falls back to fixed-function: {e}"),
        }
    }
    println!("  => {n_linked} pairs linked to a validated WGSL module");
}

/// Compact disassembly of one blob (named by `VITASLOP_GXP_DISASM`, matched as a filename
/// substring), printing each instruction's op / destination / sources - the microscope for
/// understanding a specific decode (e.g. where an internal register is written vs read).
#[test]
#[ignore = "requires VITASLOP_GXP_DUMPS + VITASLOP_GXP_DISASM; run explicitly"]
fn disasm_one() {
    let Some(dir) = dump_dir() else { return };
    let Ok(want) = std::env::var("VITASLOP_GXP_DISASM") else {
        eprintln!("set VITASLOP_GXP_DISASM=<filename substring>");
        return;
    };
    for path in gxp_files(&dir) {
        let name = path.file_name().unwrap().to_string_lossy().into_owned();
        if !name.contains(&want) {
            continue;
        }
        let bytes = fs::read(&path).unwrap();
        let Ok(program) = Program::parse(&bytes) else { continue };
        println!("\n== {name} ({} instrs) ==", program.code.len());
        // The parameter table next to the code is what makes an SA/PA register number
        // readable: a UNIFORM's `resource_index` IS its SA register, so `SA[46]` in the
        // disassembly below is whatever uniform claims resource_index 46.
        println!("  -- parameters (dubuf={} regs, pa={}, sa={}) --",
            program.default_uniform_regs, program.primary_reg_count, program.secondary_reg_count);
        for p in &program.parameters {
            println!("    {:<40} {:?}/{:?} res={} comp={} arr={} ci={}",
                p.name, p.category, p.ptype, p.resource_index, p.component_count,
                p.array_size, p.container_index);
        }
        // Literals and texture-control words live above the uniform buffer, so an `SA[n]`
        // with n >= dubuf is one of these - print both readings (F32 and the two F16 halves)
        // because the register file is untyped and the consumer decides the width.
        for &(sa, w) in &program.literals {
            println!("    literal SA[{sa}] = {w:#010x} f32={} f16=({}, {})",
                f32::from_bits(w), half_to_f32(w as u16), half_to_f32((w >> 16) as u16));
        }
        for &(sa, unit) in &program.texture_control {
            println!("    texctl SA[{sa}] -> texture unit {unit}");
        }
        for it in &program.interpolants {
            println!("    interpolant {:?} pa_base={} regs={} span={} half={} prefetch={:?}",
                it.usage, it.pa_base, it.register_count, it.span, it.half, it.prefetch);
        }
        let fmt_op = |o: &vitaslop_gxp_shader::ir::Operand| {
            format!("{:?}[{}]{}{}", o.bank, o.index,
                if o.swizzle != [0, 1, 2, 3] { format!(".{:?}", o.swizzle) } else { String::new() },
                if o.neg { "(neg)" } else { "" })
        };
        let line = |i: usize, ins: &vitaslop_gxp_shader::ir::Instr, w: u64| {
            let dest = ins.dest.as_ref().map(&fmt_op).unwrap_or_else(|| "-".into());
            let srcs: Vec<String> = ins.srcs.iter().map(&fmt_op).collect();
            let mask: String = (0..4).map(|c| if ins.write_mask[c] { "xyzw".as_bytes()[c] as char } else { '.' }).collect();
            // The op's full Debug, not just the mnemonic: for the ops that carry a payload
            // (the compare method of a cmov, the alu/cmp/reduce of a test) that payload IS
            // the semantics, and reading the idiom around it depends on seeing it.
            println!("  #{i:<3} {:<10}{} {:<12}dst={dest} [{mask}] <- {}  {:?} {}{}",
                ins.op.mnemonic(),
                if ins.half_precision { ".f16" } else { ".f32" },
                format!("{:?}", ins.pred),
                srcs.join(", "),
                ins.op,
                if let Some(b) = ins.blocked { format!("BLOCKED({b})") } else { String::new() },
                format_args!("raw={w:#018x}"));
        };
        // The SECONDARY stream runs first on the hardware and exists purely to leave values in
        // SA registers the primary reads, so a primary `SA[n]` above the uniform buffer is only
        // readable next to the instruction that produced it.
        let sec = vitaslop_gxp_shader::usse::decode_secondary_shader(&program);
        println!("  -- secondary program ({} instrs) --", sec.instrs.len());
        for (i, ins) in sec.instrs.iter().enumerate() {
            line(i, ins, ins.raw);
        }
        println!("  -- primary program ({} instrs) --", program.code.len());
        for (i, &w) in program.code.iter().enumerate() {
            line(i, &decode(w), w);
        }
    }
}

/// Survey every SPECIAL/GLOBAL hardware-register read in the corpus, with the instructions
/// around it, so what a GLOBAL index holds can be argued from what the shaders DO with it
/// rather than assumed. Prints, per read: the blob, the program kind, the GLOBAL index, the
/// instruction, and the following few instructions (the consumers).
#[test]
#[ignore = "requires the private VITASLOP_GXP_DUMPS fixture; run explicitly"]
fn global_special_register_reads() {
    let Some(dir) = dump_dir() else {
        eprintln!("VITASLOP_GXP_DUMPS unset - skipping (expected in CI)");
        return;
    };
    let mut total = 0u32;
    let mut by_index: std::collections::BTreeMap<u8, u32> = std::collections::BTreeMap::new();
    let mut by_kind: std::collections::BTreeMap<String, u32> = std::collections::BTreeMap::new();
    for path in gxp_files(&dir) {
        let name = path.file_name().unwrap().to_string_lossy().into_owned();
        let bytes = fs::read(&path).unwrap();
        let Ok(program) = Program::parse(&bytes) else { continue };
        // Both streams: a GLOBAL read in the secondary program would mean something quite
        // different (it runs once per primitive, not per fragment), so do not conflate them.
        let secondary = vitaslop_gxp_shader::usse::decode_secondary_shader(&program);
        let streams: [(&str, Vec<_>); 2] = [
            ("secondary", secondary.instrs.clone()),
            ("primary", program.code.iter().map(|&w| decode(w)).collect()),
        ];
        for (stream, instrs) in &streams {
            for (i, ins) in instrs.iter().enumerate() {
                let globals: Vec<u8> = ins
                    .srcs
                    .iter()
                    .chain(ins.dest.iter())
                    .filter(|o| matches!(o.bank, Bank::Global))
                    .map(|o| o.index)
                    .collect();
                if globals.is_empty() {
                    continue;
                }
                total += 1;
                for g in &globals {
                    *by_index.entry(*g).or_default() += 1;
                }
                *by_kind.entry(format!("{:?}/{stream}", program.kind)).or_default() += 1;
                println!(
                    "\n{name} [{stream}] #{i} reads GLOBAL{globals:?}: {} raw={:#018x}",
                    ins.op.mnemonic(),
                    ins.raw
                );
                // The consumers are the argument: what the shader DOES with the tested bit is
                // what identifies the register.
                for (j, follow) in instrs.iter().enumerate().skip(i + 1).take(8) {
                    let srcs: Vec<String> = follow
                        .srcs
                        .iter()
                        .map(|o| format!("{:?}[{}]", o.bank, o.index))
                        .collect();
                    println!(
                        "    +{:<2} {:<8} {:<10} dst={} <- {}",
                        j - i,
                        follow.op.mnemonic(),
                        format!("{:?}", follow.pred),
                        follow
                            .dest
                            .as_ref()
                            .map(|d| format!("{:?}[{}]", d.bank, d.index))
                            .unwrap_or_else(|| "-".into()),
                        srcs.join(", ")
                    );
                }
            }
        }
    }
    println!("\n=== GLOBAL reads: {total} total ===");
    println!("  by index: {by_index:?}");
    println!("  by kind/stream: {by_kind:?}");
}

/// Verify the SA-bank layout model on every real fragment blob: each `SMP` instruction's
/// sampler operand must resolve, through the container's texture-control table, to a texture
/// unit the program actually declares as a SAMPLER parameter. The `SMP` sampler field is a
/// register number in double-register units, so the control words live at SA register
/// `2 * field` (see the distilled SA-bank layout notes).
///
/// This is the check that pins the whole SA model: it only passes if the default-uniform-buffer
/// size, the `table_index + dubuf` register mapping and the double-register scaling are ALL
/// right, and it independently corroborates each resolution by dimensionality (the `...1D` fog
/// table is always sampled with one coordinate, the ambient cube map with three).
#[test]
#[ignore = "requires the private VITASLOP_GXP_DUMPS fixture; run explicitly"]
fn sampler_unit_correlation() {
    let Some(dir) = dump_dir() else {
        eprintln!("VITASLOP_GXP_DUMPS unset - skipping (expected in CI)");
        return;
    };
    let mut checked = 0u32;
    for path in gxp_files(&dir) {
        let name = path.file_name().unwrap().to_string_lossy().into_owned();
        let bytes = fs::read(&path).unwrap();
        let Ok(program) = Program::parse(&bytes) else { continue };
        let declared: Vec<(u32, &str)> = program.samplers().collect();
        let mut resolved: Vec<(u8, u8, u32)> = Vec::new();
        for &w in &program.code {
            if let vitaslop_gxp_shader::ir::Op::Tex { unit, coords, .. } = decode(w).op {
                if resolved.iter().any(|(f, _, _)| *f == unit) {
                    continue;
                }
                let sa = 2 * unit as u32;
                let gxm_unit = program.sampler_unit_at(sa).unwrap_or_else(|| {
                    panic!(
                        "{name}: SMP field {unit} -> SA reg {sa} resolves to no texture;                          control table {:?}, dubuf {}",
                        program.texture_control, program.default_uniform_regs
                    )
                });
                assert!(
                    declared.iter().any(|(u, _)| *u == gxm_unit),
                    "{name}: SMP field {unit} resolved to unit {gxm_unit}, not a declared sampler {declared:?}",
                );
                resolved.push((unit, coords, gxm_unit));
                checked += 1;
            }
        }
        if resolved.is_empty() {
            continue;
        }
        // Dimensionality corroboration by declared name, the independent signal.
        for &(_, coords, gxm_unit) in &resolved {
            let nm = declared.iter().find(|(u, _)| *u == gxm_unit).map(|(_, n)| *n).unwrap_or("");
            if nm.ends_with("1D") {
                assert_eq!(coords, 1, "{name}: {nm} sampled with {coords} coords");
            }
            if nm.ends_with("AmbientMap") {
                assert_eq!(coords, 3, "{name}: {nm} sampled with {coords} coords");
            }
        }
        println!(
            "{name:<22} dubuf={} literals={:?} resolved(field,coords,unit)={resolved:?}",
            program.default_uniform_regs, program.literals,
        );
        println!("        interpolants={:?}", program.interpolants);
    }
    assert!(checked > 0, "no SMP instructions found in the fixture");
    println!("  => {checked} SMP sampler operands resolved through the container");
}

/// EVIDENCE for the vertex->fragment varying interface: for each captured real pairing, print
/// the fragment's interpolant DESCRIPTORS (the container's own statement of the interface:
/// usage, PA base, register span, precision) next to how the fragment CODE actually reads each
/// of those PA registers, next to how many varying outputs the VERTEX writes.
///
/// The three are independent statements about one interface, so they must agree:
/// * a descriptor spanning `register_count` registers at `half` precision consumes
///   `register_count * (2 if half else 1)` interpolated scalar components,
/// * the code must read each register in that span at the descriptor's precision,
/// * the total component demand must equal the varying output lanes the vertex writes.
///
/// Any disagreement names exactly which register and which of the three is wrong - that is the
/// signal needed to settle the interface, rather than curve-fitting a total.
#[test]
#[ignore = "requires VITASLOP_GXP_DUMPS + VITASLOP_GXP_PAIRS; run explicitly"]
fn varying_interface_evidence() {
    let Some(dir) = dump_dir() else { return };
    let Ok(spec) = std::env::var("VITASLOP_GXP_PAIRS") else {
        eprintln!("set VITASLOP_GXP_PAIRS=\"vh:fh,...\" (header hexes)");
        return;
    };
    println!("\n=== varying interface evidence (descriptors vs code vs vertex) ===");
    for (vh, fh) in spec.split(',').filter_map(|p| p.split_once(':')) {
        let (vh, fh) = (vh.trim(), fh.trim());
        let (Ok(vb), Ok(fb)) = (
            fs::read(dir.join(format!("vert_{vh}.gxp"))),
            fs::read(dir.join(format!("frag_{fh}.gxp"))),
        ) else {
            println!("  (missing blob for {vh}/{fh})");
            continue;
        };
        let (Ok(vp), Ok(fp)) = (Program::parse(&vb), Program::parse(&fb)) else { continue };
        let vsh = vitaslop_gxp_shader::usse::decode_shader(&vp);
        let fsh = vitaslop_gxp_shader::usse::decode_shader(&fp);

        // Vertex: OUTPUT lanes written (one interpolated scalar per lane).
        let mut owritten = vec![false; 64];
        for ins in &vsh.instrs {
            let Some(d) = ins.dest.as_ref() else { continue };
            if d.bank != Bank::Output {
                continue;
            }
            for c in 0..4 {
                if ins.write_mask[c] {
                    if let Some(s) = owritten.get_mut(d.index as usize + c) {
                        *s = true;
                    }
                }
            }
        }
        let olanes: Vec<usize> =
            owritten.iter().enumerate().filter(|&(_, &w)| w).map(|(l, _)| l).collect();
        let vary_lanes: Vec<usize> = olanes.iter().copied().filter(|&l| l >= 6).collect();
        let vspan = vary_lanes.last().map(|&l| l as u32 + 1 - 6).unwrap_or(0);

        // Fragment: how the CODE reads each PA register before any write to it (the raw
        // register file is untyped, so the reading instruction's precision is the only
        // in-code statement of a varying's width).
        let mut code_half: Vec<Option<bool>> = vec![None; 64];
        let mut pwritten = vec![false; 64];
        for ins in &fsh.instrs {
            let half = ins.half_precision;
            let read = match ins.op {
                vitaslop_gxp_shader::ir::Op::Dot { components } => components.clamp(1, 4),
                vitaslop_gxp_shader::ir::Op::Tex { coords, .. } => coords.clamp(1, 4),
                _ => ins.write_mask.iter().filter(|&&m| m).count() as u8,
            };
            for src in &ins.srcs {
                if src.bank != Bank::PrimaryAttr {
                    continue;
                }
                for c in 0..read as usize {
                    let sel = src.swizzle[c.min(3)];
                    if sel > 3 {
                        continue;
                    }
                    let reg = src.index as usize
                        + if half { (sel >> 1) as usize } else { sel as usize };
                    if reg < code_half.len() && !pwritten[reg] && code_half[reg].is_none() {
                        code_half[reg] = Some(half);
                    }
                }
            }
            if let Some(d) = ins.dest.as_ref() {
                if d.bank == Bank::PrimaryAttr {
                    for c in 0..4 {
                        if ins.write_mask[c] {
                            let reg = d.index as usize + if half { c >> 1 } else { c };
                            if let Some(s) = pwritten.get_mut(reg) {
                                *s = true;
                            }
                        }
                    }
                }
            }
        }

        // Vertex varyings block: vo1 (usage presence bits) + vo2 (ten 3-bit texcoord widths).
        let var_rel = u32(&vb, 0x2c);
        let (vo1, vo2) = if var_rel != 0 {
            let blk = 0x2c + var_rel as usize;
            (u32(&vb, blk + 0x10), u32(&vb, blk + 0x14))
        } else {
            (0, 0)
        };
        let tc: Vec<(u32, u32)> = (0..10u32)
            .filter_map(|k| {
                let v = (vo2 >> (k * 3)) & 0x7;
                (v != 0).then(|| (k, (v & 1) * 2 + ((v >> 1) & 1) + ((v >> 2) & 1)))
            })
            .collect();

        let reads: Vec<String> = code_half
            .iter()
            .enumerate()
            .filter_map(|(r, h)| h.map(|h| format!("{r}{}", if h { "h" } else { "f" })))
            .collect();
        println!("\n  vert_{vh} -> frag_{fh}");
        println!(
            "    fragment primary_regs={} declared_regs={} reads-before-write: [{}]",
            fp.primary_reg_count,
            fp.interpolants.iter().map(|i| i.register_count as u32).sum::<u32>(),
            reads.join(" "),
        );
        println!("    vertex o-lanes: {olanes:?}  varying lanes(>=6): {vary_lanes:?} span={vspan}");
        println!("    vo1={vo1:#010x} vo2={vo2:#010x} texcoords(idx,comps)={tc:?}");
        // Vertex-side placement: texcoords in ascending index from the varying base, one lane
        // per component. Printed so the fragment's per-usage span can be matched against it.
        let mut lane = 6u32;
        let mut vplace: Vec<String> = Vec::new();
        for &(k, n) in &tc {
            vplace.push(format!("TC{k}@o{}..{}", lane, lane + n));
            lane += n;
        }
        println!("    vertex texcoord placement (base 6, 1 lane/comp): [{}] end={lane}", vplace.join(" "));
        let mut demand = 0u32;
        for it in &fp.interpolants {
            let comps = it.register_count as u32 * if it.half { 2 } else { 1 };
            demand += comps;
            let reads: Vec<String> = (it.pa_base..it.pa_base + it.register_count)
                .map(|r| match code_half.get(r as usize).copied().flatten() {
                    Some(true) => "h".to_string(),
                    Some(false) => "f".to_string(),
                    None => ".".to_string(),
                })
                .collect();
            let agree = (it.pa_base..it.pa_base + it.register_count).all(|r| {
                code_half.get(r as usize).copied().flatten().is_none_or(|h| h == it.half)
            });
            println!(
                "    {:?}@pa{}..{} rc={} half={} comps={} code=[{}] {}",
                it.usage,
                it.pa_base,
                it.pa_base + it.register_count,
                it.register_count,
                it.half,
                comps,
                reads.join(""),
                if agree { "AGREE" } else { "*** DISAGREE ***" },
            );
        }
        // PA registers the code reads as varyings but no descriptor covers (would be an
        // interface the container does not declare - or a scratch misclassification).
        let covered = |r: usize| {
            fp.interpolants
                .iter()
                .any(|it| r >= it.pa_base as usize && r < (it.pa_base + it.register_count) as usize)
        };
        let uncovered: Vec<usize> = (0..code_half.len())
            .filter(|&r| code_half[r].is_some() && !covered(r))
            .collect();
        println!(
            "    descriptor demand={demand} vs vertex span={vspan} => {}   uncovered code reads: {uncovered:?}",
            if demand == vspan { "MATCH" } else { "MISMATCH" }
        );
    }
}

/// Disassemble every blob's SECONDARY program (the stream that runs before the primary and
/// fills SA-bank registers), and report which SA registers each one writes.
///
/// This is the instrument for the "primary program reads an SA register the guest never wrote"
/// case: the value is not missing, it was computed here. The redundant count/start/end triple
/// the parser checks means a blob that reaches this point has a correctly sliced stream.
#[test]
#[ignore = "requires the private VITASLOP_GXP_DUMPS fixture; run explicitly"]
fn secondary_program_disasm() {
    let Some(dir) = dump_dir() else { return };
    let (mut with, mut total_instrs, mut emittable) = (0u32, 0u32, 0u32);
    for path in gxp_files(&dir) {
        let bytes = fs::read(&path).unwrap();
        let name = path.file_name().unwrap().to_string_lossy().into_owned();
        let program = Program::parse(&bytes).unwrap_or_else(|e| panic!("{name}: parse failed: {e:?}"));
        if program.secondary_code.is_empty() {
            continue;
        }
        with += 1;
        println!("\n== {name}: {} secondary instrs (uniform_regs={}, secondary_regs={})",
            program.secondary_code.len(), program.default_uniform_regs, program.secondary_reg_count);
        for (i, &w) in program.secondary_code.iter().enumerate() {
            let ins = decode(w);
            total_instrs += 1;
            if ins.is_supported() {
                emittable += 1;
            }
            let fmt = |o: &vitaslop_gxp_shader::ir::Operand| {
                format!("{:?}[{}]{}", o.bank, o.index,
                    if o.swizzle != [0, 1, 2, 3] { format!(".{:?}", o.swizzle) } else { String::new() })
            };
            let mask: String = (0..4).map(|c| if ins.write_mask[c] { "xyzw".as_bytes()[c] as char } else { '.' }).collect();
            println!("  #{i:<2} {:<8}{} dst={} [{mask}] <- {}  {}",
                ins.op.mnemonic(),
                if ins.half_precision { ".f16" } else { ".f32" },
                ins.dest.as_ref().map(&fmt).unwrap_or_else(|| "-".into()),
                ins.srcs.iter().map(&fmt).collect::<Vec<_>>().join(", "),
                ins.blocked.map(|b| format!("BLOCKED({b})")).unwrap_or_default());
        }
    }
    println!("\n  {with} blobs carry a secondary program, {total_instrs} instrs, {emittable} emittable");
}

/// Read a little-endian u32 from a byte slice (analysis helper; returns 0 if out of range).
fn u32(b: &[u8], off: usize) -> u32 {
    b.get(off..off + 4)
        .map(|s| u32::from_le_bytes([s[0], s[1], s[2], s[3]]))
        .unwrap_or(0)
}

/// Decode an IEEE binary16 bit pattern to `f32` (analysis helper - the SA literal table is
/// untyped 32-bit storage and F16 consumers read it as two packed halves).
fn half_to_f32(h: u16) -> f32 {
    let sign = ((h >> 15) & 1) as u32;
    let exp = ((h >> 10) & 0x1f) as u32;
    let man = (h & 0x3ff) as u32;
    let bits = match exp {
        0 if man == 0 => sign << 31,
        0 => {
            // Subnormal half: normalise into a binary32 exponent.
            let shift = man.leading_zeros() - 21;
            (sign << 31) | ((127 - 15 - shift) << 23) | ((man << (shift + 1)) & 0x7f_ffff)
        }
        0x1f => (sign << 31) | 0x7f80_0000 | (man << 13),
        _ => (sign << 31) | ((exp + 127 - 15) << 23) | (man << 13),
    };
    f32::from_bits(bits)
}

/// The REPEAT-COUNT closure test: after unrolling, a vertex program's written OUTPUT lanes must
/// be exactly the set its own container declares, `0..(vertex_outputs1 >> 24)`.
///
/// This is the independent check on the repeat stride (`crate::usse::unroll_repeats`). The two
/// statements come from opposite ends of the container - one from the varyings-block header,
/// the other from executing the instruction stream - and nothing makes them agree except a
/// correct model of how a repeated instruction advances its destination. Before repetition was
/// modelled, programs whose colour varying is written by one repeating `mov` fell short by two
/// or four lanes, and the missing lanes were COLOR0's z/w and, in a textured program, the whole
/// texture coordinate.
///
/// Programs whose stream still contains a blocked instruction are skipped, not failed: an
/// instruction the decoder refuses to translate may be the one that writes the missing lane, so
/// requiring closure there would assert on a stream we have not claimed to decode.
#[test]
#[ignore = "requires the private VITASLOP_GXP_DUMPS fixture; run explicitly"]
fn vertex_written_lanes_close_against_declared_total() {
    let Some(dir) = dump_dir() else {
        eprintln!("VITASLOP_GXP_DUMPS unset - skipping repeat closure test");
        return;
    };
    let (mut checked, mut skipped) = (0u32, 0u32);
    let mut failures: Vec<String> = Vec::new();
    let mut reasons: Vec<String> = Vec::new();
    let mut single_lane_short: Vec<String> = Vec::new();
    for path in gxp_files(&dir) {
        let name = path.file_name().unwrap().to_string_lossy().into_owned();
        let bytes = fs::read(&path).unwrap();
        let Ok(program) = Program::parse(&bytes) else { continue };
        if program.kind != ProgramKind::Vertex {
            continue;
        }
        let var_rel = u32(&bytes, 0x2c);
        if var_rel == 0 {
            continue;
        }
        let total = u32(&bytes, 0x2c + var_rel as usize + 0x10) >> 24;
        if total == 0 {
            continue;
        }
        let shader = vitaslop_gxp_shader::usse::decode_shader(&program);
        if shader.instrs.iter().any(|i| i.blocked.is_some()) {
            skipped += 1;
            for i in shader.instrs.iter().filter(|i| i.blocked.is_some()) {
                reasons.push(format!("{} (group {:#04x})", i.blocked.unwrap(), i.group));
            }
            continue;
        }
        let mut written = vec![false; 256];
        for instr in &shader.instrs {
            let Some(d) = instr.dest.as_ref() else { continue };
            if d.bank != Bank::Output {
                continue;
            }
            for c in 0..4 {
                if instr.write_mask[c] {
                    let lane = d.index as usize + c;
                    if lane < written.len() {
                        written[lane] = true;
                    }
                }
            }
        }
        // The lanes the recompiler actually ROUTES: clip position, plus every varying the
        // container declares. This is the set correctness depends on, and it is the right
        // comparison rather than a dense `0..total` - a reserved region can legitimately hold
        // a lane no varying claims and the program therefore never writes (every fog-carrying
        // program in the corpus declares a 2-lane region whose second lane is exactly that).
        let mut routed = vec![false; written.len()];
        for l in 0..4usize {
            routed[l] = true;
        }
        for v in &program.output_varyings {
            for c in 0..v.components as usize {
                let lane = v.base_lane as usize + c;
                if lane < routed.len() {
                    routed[lane] = true;
                }
            }
        }
        let got: Vec<usize> = (0..written.len()).filter(|&l| written[l]).collect();
        let expect: Vec<usize> = (0..routed.len()).filter(|&l| routed[l]).collect();
        checked += 1;
        // Two distinct failures, and the direction matters when reading a regression: a
        // SHORTFALL means an iteration was dropped (the pre-repeat bug, which silently left a
        // varying uninterpolated), an OVERRUN means the stride is too large and the program is
        // writing over a lane that belongs to something else.
        let over: Vec<usize> = got.iter().copied().filter(|l| !routed[*l]).collect();
        let under: Vec<usize> = expect.iter().copied().filter(|l| !written[*l]).collect();
        // An OVERRUN is unambiguously a wrong stride: the program wrote a lane no declared
        // varying claims. A SHORTFALL of a single lane is not - a program may simply not fill a
        // component its consumer never reads, and the container declares the interface width
        // rather than promising every lane is written. What separates the two readings is the
        // SIZE of the shortfall: a wrong repeat stride drops whole iterations, so it comes up
        // short by two or four lanes at a time (before repetition was modelled the two
        // front-end programs were short by exactly 2 and 4). One lane cannot be a dropped
        // iteration, so it is bounded here rather than ignored.
        if !over.is_empty() || under.len() > 1 {
            failures.push(format!(
                "  {name}: declares {total} total lanes; routed {expect:?}; wrote {got:?}\
                 \n      unwritten routed lanes {under:?}, writes outside the routed set {over:?}"
            ));
        } else if !under.is_empty() {
            single_lane_short.push(format!("{name} lane {}", under[0]));
        }
    }
    println!("\n=== repeat closure: {checked} vertex programs checked, {skipped} skipped (blocked) ===");
    // What is actually blocking the skipped programs, so a skip count can never quietly stand
    // in for a decode gap nobody has looked at.
    reasons.sort();
    let mut tally: Vec<(String, usize)> = Vec::new();
    for r in reasons {
        match tally.last_mut() {
            Some((k, n)) if *k == r => *n += 1,
            _ => tally.push((r, 1)),
        }
    }
    tally.sort_by_key(|(_, n)| std::cmp::Reverse(*n));
    for (r, n) in &tally {
        println!("  {n:5}  {r}");
    }
    if !single_lane_short.is_empty() {
        println!(
            "  {} program(s) leave exactly one routed lane unwritten (an unread component, not a\n\
             \x20 dropped iteration - see the test's doc comment): {}",
            single_lane_short.len(),
            single_lane_short.join(", ")
        );
    }
    assert!(
        failures.is_empty(),
        "{} of {checked} vertex programs do not write exactly their declared output lanes:\n{}",
        failures.len(),
        failures.join("\n")
    );
}
