//! Offline analysis of a captured `.gxp` corpus.
//!
//! Point `VITASLOP_GXP_CORPUS` at a directory of `vert_*.gxp` / `frag_*.gxp` blobs (what
//! `VITASLOP_DUMP_GXP_BIN` writes) and run:
//!
//! ```text
//! VITASLOP_GXP_CORPUS=<dir> cargo test -p vitaslop-gxp-shader --test corpus -- --ignored
//! --nocapture
//! ```
//!
//! # Why this exists
//!
//! Every question about why a shader will not recompile was costing a full replay of the
//! title - minutes per question, for an answer that depends on nothing but the blob. This
//! answers the same questions in under a second, over the WHOLE corpus at once, which also
//! turns "why does this one fail" into "how many fail, and on what" - the ranking that says
//! what to implement next.
//!
//! It is `#[ignore]`d because the corpus is captured game bytes: it never exists in CI, and
//! a test that needs it must not fail there.

use std::collections::BTreeMap;
use std::path::PathBuf;

use vitaslop_gxp_shader::ir::{Bank, Op, TestAlu, TestCmp};
use vitaslop_gxp_shader::wgsl::{HALF_HI_FN, HALF_LO_FN, HALF_PK_FN, HALF_QUANT_FN};
use vitaslop_gxp_shader::{link_programs, recompile_fragment, recompile_vertex, Program, ProgramKind};

fn corpus_dir() -> Option<PathBuf> {
    std::env::var_os("VITASLOP_GXP_CORPUS").map(PathBuf::from)
}

/// Every blob in the corpus, as `(file stem, bytes)`.
fn blobs(dir: &PathBuf) -> Vec<(String, Vec<u8>)> {
    let mut out = Vec::new();
    let Ok(rd) = std::fs::read_dir(dir) else { return out };
    for e in rd.flatten() {
        let p = e.path();
        if p.extension().and_then(|s| s.to_str()) != Some("gxp") {
            continue;
        }
        if let Ok(b) = std::fs::read(&p) {
            out.push((p.file_stem().unwrap_or_default().to_string_lossy().into_owned(), b));
        }
    }
    out.sort_by(|a, b| a.0.cmp(&b.0));
    out
}

/// Does `want` name this blob - either by file stem (`vert_867062a0`) or by the CONTENT hash
/// the live renderer prints (`gxp pair <key>: vprog hash <h>, fprog hash <h>`)?
///
/// The file stem is a guest ADDRESS, which differs between the run that captured the corpus and
/// the run that printed the key. The hash does not, so it is the only reliable way to take a
/// pair seen in a frame back to the two blobs an offline test can open.
fn blob_matches(name: &str, bytes: &[u8], want: &str) -> bool {
    // An empty selector means EVERY blob, so a question that has to be asked of the whole
    // corpus at once - "which program contains this idiom" - costs one command rather than one
    // per blob.
    if want.is_empty() || name == want {
        return true;
    }
    let want = want.trim_start_matches("0x");
    u64::from_str_radix(want, 16)
        .ok()
        .is_some_and(|h| Program::parse(bytes).is_ok_and(|p| p.hash == h))
}

/// Print every blob's content hash beside its file name, so a `gxp pair` line from a live run
/// can be turned into blob names in one command.
#[test]
#[ignore = "needs a captured corpus (game bytes); set VITASLOP_GXP_CORPUS"]
fn tabulate_blob_hashes() {
    let Some(dir) = corpus_dir() else { return };
    for (name, bytes) in blobs(&dir) {
        match Program::parse(&bytes) {
            Ok(p) => println!("{:016x}  {name}  {:?}  uniform_regs={}", p.hash, p.kind, p.default_uniform_regs),
            Err(e) => println!("{:>16}  {name}  parse failed: {e:?}", "-"),
        }
    }
}

/// One line per blob: its name and a hash of the WGSL the recompiler emits for it.
///
/// # What this is for, and why a hash rather than the text
/// A DECODER change's reach is "which programs does it change the emitted code of", and until
/// now the only answers available were "does the recompile still succeed" (which a wrong
/// operand does not disturb at all) and a picture from a fifteen-minute replay. Running this
/// under both arms of a change and diffing names EVERY program it touches, across every corpus
/// on disk, in seconds - which is what turns "this looks safe" into a list.
///
/// The hash is FNV-1a over the emitted string, so a blob whose output is byte-identical hashes
/// identically and a blob whose output moved by one register does not.
#[test]
#[ignore = "needs a captured corpus (game bytes); set VITASLOP_GXP_CORPUS"]
fn hash_every_blob_wgsl() {
    let Some(dir) = corpus_dir() else {
        eprintln!("VITASLOP_GXP_CORPUS not set - nothing to analyse");
        return;
    };
    let all = blobs(&dir);
    assert!(!all.is_empty(), "no .gxp blobs under {}", dir.display());
    let fnv = |s: &str| {
        let mut h: u64 = 0xcbf2_9ce4_8422_2325;
        for b in s.as_bytes() {
            h ^= u64::from(*b);
            h = h.wrapping_mul(0x1000_0000_01b3);
        }
        h
    };
    for (name, bytes) in &all {
        let Ok(p) = Program::parse(bytes) else {
            println!("{name} PARSE-FAIL");
            continue;
        };
        let out = match p.kind {
            ProgramKind::Vertex => recompile_vertex(bytes).map(|m| m.wgsl_body),
            ProgramKind::Fragment => recompile_fragment(bytes).map(|m| m.wgsl_body),
        };
        match out {
            Ok(w) => println!("{name} {:016x}", fnv(&w)),
            Err(e) => println!("{name} FAIL {e}"),
        }
    }
}

/// Recompile every blob on its own and rank the failures by cause.
///
/// A single-stage failure is a decoder or emitter gap and is independent of pairing, so it is
/// the cheapest thing to fix and the right thing to count first.
#[test]
#[ignore = "needs a captured corpus (game bytes); set VITASLOP_GXP_CORPUS"]
fn recompile_every_blob_and_rank_the_failures() {
    let Some(dir) = corpus_dir() else {
        eprintln!("VITASLOP_GXP_CORPUS not set - nothing to analyse");
        return;
    };
    let all = blobs(&dir);
    assert!(!all.is_empty(), "no .gxp blobs under {}", dir.display());

    let mut by_reason: BTreeMap<String, Vec<String>> = BTreeMap::new();
    let (mut ok, mut failed) = (0usize, 0usize);
    for (name, bytes) in &all {
        let kind = match Program::parse(bytes) {
            Ok(p) => p.kind,
            Err(e) => {
                by_reason.entry(format!("container parse: {e:?}")).or_default().push(name.clone());
                failed += 1;
                continue;
            }
        };
        let result = match kind {
            ProgramKind::Vertex => recompile_vertex(bytes).map(|_| ()),
            ProgramKind::Fragment => recompile_fragment(bytes).map(|_| ()),
        };
        match result {
            Ok(()) => ok += 1,
            Err(e) => {
                failed += 1;
                by_reason.entry(format!("{e}")).or_default().push(name.clone());
            }
        }
    }
    println!("corpus: {} blobs, {ok} recompile on their own, {failed} do not", all.len());
    let mut ranked: Vec<_> = by_reason.iter().collect();
    ranked.sort_by_key(|r| std::cmp::Reverse(r.1.len()));
    for (reason, names) in ranked {
        println!("  {} blobs - {reason}", names.len());
        for n in names.iter().take(4) {
            println!("      {n}");
        }
        if names.len() > 4 {
            println!("      ... and {} more", names.len() - 4);
        }
    }
}

/// Print the container reflection of every FRAGMENT blob: its interpolants, the PA registers
/// they cover, and its samplers.
///
/// This is the view needed to settle a `PaReadUnfed` - the fragment reads a PA register that
/// no declared interpolant feeds - because it shows what the container actually declares next
/// to what the code reads, which is the comparison the error is making.
#[test]
#[ignore = "needs a captured corpus (game bytes); set VITASLOP_GXP_CORPUS"]
fn describe_fragment_interpolants() {
    let Some(dir) = corpus_dir() else { return };
    for (name, bytes) in blobs(&dir) {
        let Ok(p) = Program::parse(&bytes) else { continue };
        if p.kind != ProgramKind::Fragment {
            continue;
        }
        // A fragment that declares NO interpolant at all cannot be fed by any vertex program,
        // so it is the whole population of a `PaReadUnfed` failure and worth listing on its
        // own. Everything else is printed only when it also fails to recompile standalone.
        let no_interpolants = p.interpolants.is_empty();
        let err = match recompile_fragment(&bytes) {
            Ok(_) if !no_interpolants => continue,
            Ok(_) => match p.varyings_error {
                Some(why) => format!("NO interpolants - the varyings block did not decode: {why}"),
                None => "NO interpolants - the program declares none".to_string(),
            },
            Err(e) => format!("{e}"),
        };
        println!("\n{name}: {err}");
        println!("  primary_reg_count={} interpolants={}", p.primary_reg_count, p.interpolants.len());
        for (i, d) in vitaslop_gxp_shader::container::raw_varying_descriptors(&bytes).iter().enumerate() {
            println!(
                "    raw[{i}] attribute_info={:#010x} resource_index={:#x} size={:#x} component_info={:#x}",
                d[0], d[1], d[2], d[3]
            );
        }
        for it in &p.interpolants {
            println!(
                "    usage={:?} pa_base={} registers={} half={} prefetch={:?}",
                it.usage, it.pa_base, it.register_count, it.half, it.prefetch
            );
        }
    }
}

/// Tabulate every fragment interpolant's `(half, register_count)` against its usage, and every
/// vertex program's per-usage component width, so the two sides' UNITS can be compared.
///
/// # The question this settles
/// `plan_interface` hard-fails when a vertex produces more components than the fragment's
/// declaration spans, on the reasoning that the surplus would land on the next interpolant.
/// One title's `tutorial-drive` hits that on a real pair, and the failure has a suspiciously
/// uniform shape: every instance reads "the fragment spans **1** PA register at F16", never
/// any other count. Two readings explain it and they need opposite fixes - the hardware
/// tolerates a fragment consuming a PREFIX of a wider varying, or `register_count` is being
/// parsed in the wrong UNIT for a half-precision varying (the trap
/// `vitaslop-f16-half-granularity-varyings` records once already).
///
/// A count is what tells them apart: if EVERY half varying in the corpus declares exactly one
/// register whatever its width, the field is not a register count.
#[test]
#[ignore = "needs a captured corpus (game bytes); set VITASLOP_GXP_CORPUS"]
fn tabulate_interpolant_register_counts_by_precision() {
    let Some(dir) = corpus_dir() else { return };
    // (half, register_count) -> how many interpolants declare it.
    let mut by_shape: BTreeMap<(bool, u8), usize> = BTreeMap::new();
    // The same, split by usage, so a usage-specific rule would show.
    let mut by_usage: BTreeMap<(String, bool, u8), usize> = BTreeMap::new();
    let mut frags = 0usize;
    for (_, bytes) in blobs(&dir) {
        let Ok(p) = Program::parse(&bytes) else { continue };
        if p.kind != ProgramKind::Fragment {
            continue;
        }
        frags += 1;
        for it in &p.interpolants {
            *by_shape.entry((it.half, it.register_count)).or_default() += 1;
            *by_usage
                .entry((format!("{:?}", it.usage), it.half, it.register_count))
                .or_default() += 1;
        }
    }
    println!("{frags} fragment blobs");
    println!("  (half, register_count) -> count");
    for ((half, regs), n) in &by_shape {
        println!("    half={half} registers={regs}: {n}");
    }
    println!("  by usage:");
    for ((usage, half, regs), n) in &by_usage {
        println!("    {usage:<12} half={half} registers={regs}: {n}");
    }
}

/// Tabulate every FRAGMENT program's varying DECLARATION ORDER, and every VERTEX program's
/// output-lane accounting, so the two can be compared.
///
/// The vertex block states WHICH varyings a program outputs and how WIDE each texcoord is; it
/// does not state the ORDER they occupy the output bank in, and two titles' programs demand
/// opposite orders for the same declared set. The fragment's descriptor array DOES carry an
/// order - its entries accumulate a PA base in declaration order - so if the vertex lane order
/// is the fragment's declaration order, this tabulation shows the two titles' fragments
/// declaring their varyings in opposite orders, and the contradiction is not one.
#[test]
#[ignore = "needs a captured corpus (game bytes); set VITASLOP_GXP_CORPUS"]
fn tabulate_fragment_varying_declaration_order() {
    let Some(dir) = corpus_dir() else {
        eprintln!("VITASLOP_GXP_CORPUS not set - nothing to analyse");
        return;
    };
    let mut by_order: BTreeMap<String, Vec<String>> = BTreeMap::new();
    for (name, bytes) in blobs(&dir) {
        let Ok(p) = Program::parse(&bytes) else { continue };
        if p.kind != ProgramKind::Fragment {
            continue;
        }
        if p.interpolants.is_empty() {
            continue;
        }
        let order: Vec<String> = p
            .interpolants
            .iter()
            .map(|it| format!("{:?}@{}+{}", it.usage, it.pa_base, it.register_count))
            .collect();
        let usages: Vec<String> =
            p.interpolants.iter().map(|it| format!("{:?}", it.usage)).collect();
        println!("{name}: primary_reg_count={} {}", p.primary_reg_count, order.join(" "));
        by_order.entry(usages.join(",")).or_default().push(name);
    }
    println!("\n-- declaration orders, by how many fragment programs use them --");
    for (order, names) in &by_order {
        println!("  {:<3} [{order}]", names.len());
        if names.len() <= 6 {
            println!("        {}", names.join(" "));
        }
    }
}

/// How many `Assumed` vertex orders does the program's OWN CODE contradict?
///
/// # The gap this measures
/// `link::plan_interface` asks `convention_agrees_with_the_code` - the written-lane witness -
/// only for [`VaryingOrder::Ambiguous`] programs (the declared-COLOR1 case). Every
/// [`VaryingOrder::Assumed`] program takes the canonical order UNCHECKED, even though the same
/// witness is available and is precise: a varying whose run begins on a lane the program never
/// writes is a run that is not there, and a written lane outside every run is a run the layout
/// does not account for.
///
/// So the convention is verified exactly where it is already distrusted and trusted blind
/// everywhere else. This counts the programs where that blind trust is contradicted by the
/// blob itself - each one is a pair drawing a confident, wrong picture with every varying read
/// from the wrong register.
///
/// # TWO VERDICTS, and only one of them is a defect
/// An earlier version of this test folded them together and reported "8 CONTRADICTED" on one
/// title, which was written up in the notes as WIDTH errors in the texcoord pack field and
/// carried as a session's worth of work. **It was a false alarm, and this test's own output
/// refuted it.** The two shapes are:
///
/// - **CONTRADICTED** - a lane the program WRITES that falls outside every declared run, below
///   the top of the layout. Nothing but a wrong layout can produce that: the write has to
///   belong to some varying, and the layout says no varying is there. This is the defect.
/// - **UNWRITTEN** - a declared run the program never starts, with every OTHER run landing
/// exactly on written lanes. That is not a layout error, it is a varying the shader declares
/// and does not produce, which hardware allows and the fragment stage reads as whatever is in
/// the register. **The tell is decisive: a wrong WIDTH shifts everything after it, so the runs
/// past the gap would land on unwritten lanes too - and they do not.** MEASURED on the the race
/// corpus corpus, all 8 flagged programs: `vert_868a2c50` writes 4..16 and 20..28 while its
/// convention puts TexCoord(5)(6)(7) at exactly 20..22, 22..25, 25..28; six others write
///   4..18 against a layout whose runs end at 18 with one more declared above it. Every run
///   lands where the convention says. The layouts are right.
///
/// Reported separately for that reason. A non-zero CONTRADICTED count is worth a session; a
/// non-zero UNWRITTEN count is worth nothing at all.
#[test]
#[ignore = "needs a captured corpus (game bytes); set VITASLOP_GXP_CORPUS"]
fn assumed_varying_orders_the_vertex_code_contradicts() {
    use vitaslop_gxp_shader::container::VaryingOrder;
    

    let Some(dir) = corpus_dir() else {
        eprintln!("VITASLOP_GXP_CORPUS not set - nothing to analyse");
        return;
    };
    // The clip position owns the first lanes of the output bank and is never a varying.
    const POSITION_LANES: usize = 4;

    let (mut checked, mut agree, mut contradicted, mut unwritten_only) = (0usize, 0usize, 0usize, 0usize);
    for (name, bytes) in blobs(&dir) {
        let Ok(p) = Program::parse(&bytes) else { continue };
        if p.kind != ProgramKind::Vertex || p.output_varyings.len() < 2 {
            continue;
        }
        // Only the orders nothing has checked. `Known` is read from the attributes and
        // `Ambiguous` already goes through this same witness in the linker.
        if p.output_order != VaryingOrder::Assumed {
            continue;
        }
        checked += 1;

        let shader = vitaslop_gxp_shader::usse::decode_shader(&p);
        let written = vitaslop_gxp_shader::usse::written_output_lanes(&shader);
        let in_a_run = |lane: usize| {
            lane < POSITION_LANES
                || p.output_varyings.iter().any(|v| {
                    let lo = v.base_lane as usize;
                    lane >= lo && lane < lo + v.components as usize
                })
        };
        let stray: Vec<usize> =
            written.iter().enumerate().filter(|&(l, &w)| w && !in_a_run(l)).map(|(l, _)| l).collect();
        let unstarted: Vec<String> = p
            .output_varyings
            .iter()
            .filter(|v| !written.get(v.base_lane as usize).copied().unwrap_or(false))
            .map(|v| format!("{:?}@{}", v.usage, v.base_lane))
            .collect();
        // A lane ABOVE every declared run is weak evidence on its own: clip planes and point
        // size sit at the TOP of the output bank, take lanes, and are not varyings - and
        // `Program` does not carry their count, so this test cannot subtract them. A lane
        // written BELOW the top of the layout cannot be one of those, so only those count.
        let top = p
            .output_varyings
            .iter()
            .map(|v| v.base_lane as usize + v.components as usize)
            .max()
            .unwrap_or(0);
        let (above, inside): (Vec<usize>, Vec<usize>) = stray.iter().partition(|&&l| l >= top);
        if inside.is_empty() && unstarted.is_empty() {
            agree += 1;
            continue;
        }
        // The verdict, split - see the doc comment. A run the program never starts is not
        // evidence against the LAYOUT unless something else is written where no run is.
        let verdict = if inside.is_empty() {
            unwritten_only += 1;
            "UNWRITTEN (a declared varying the program does not produce - the layout is not in \
             question: every other run lands on written lanes)"
        } else {
            contradicted += 1;
            "CONTRADICTED"
        };
        let layout: Vec<String> = p
            .output_varyings
            .iter()
            .map(|v| format!("{:?}@{}..{}", v.usage, v.base_lane, v.base_lane + v.components))
            .collect();
        println!("{name} {verdict}  convention says {}", layout.join(" "));
        let live: Vec<usize> =
            written.iter().enumerate().filter(|&(_, &w)| w).map(|(l, _)| l).collect();
        println!("  written lanes {live:?}");
        if !inside.is_empty() {
            println!("  written OUTSIDE every declared run, BELOW the top of the layout: {inside:?}");
        }
        if !above.is_empty() {
            println!("  also written above the layout ({above:?}) - may be clip planes / psize");
        }
        if !unstarted.is_empty() {
            println!("  declared runs whose FIRST lane is never written: {}", unstarted.join(" "));
        }
    }
    println!(
        "\nassumed-order vertex programs: {checked} checked, {agree} agree with their own code, \
         {unwritten_only} declare a varying they never write (NOT a layout defect), \
         {contradicted} CONTRADICTED"
    );
    // The one number that is a defect. Zero across every corpus captured so far, which is what
    // says the canonical order is right where nothing checks it - the question this test was
    // written to answer.
    assert_eq!(
        contradicted, 0,
        "an assumed varying order writes a lane no declared run covers - the layout is wrong \
         and every varying past it is read from the wrong register"
    );
}

/// Does the VERTEX's decoded output-lane order agree with the order the FRAGMENT it is
/// paired with declares its interpolants in?
///
/// # The question this settles
/// `parse_vertex_output_varyings` places a vertex program's varyings in a CANONICAL order
/// (colours, fog, then texcoords ascending) whenever the attributes do not cover the
/// declared set exactly. That is a convention, not a reading - and
/// `tabulate_fragment_varying_declaration_order` shows the fragment side using orders no
/// single convention can produce: `[Color0,TexCoord(1)]` and `[TexCoord(2),Color0]` in one
/// title, `[..,Fog,TexCoord(3)]` and `[..,TexCoord(3),Fog]` in another, and one program
/// declaring TexCoord(2) before TexCoord(0).
///
/// Both stages describe the SAME interface, and the fragment states its order explicitly
/// (its descriptors are in PA order, which mirrors the vertex's lane order). So every pair
/// where the two disagree is a pair whose vertex lanes are assigned wrongly - each varying
/// read from the wrong register, silently, with a picture that still draws.
///
/// This counts them, which is the number that says whether the convention is a small
/// rough edge or the wrong mechanism.
#[test]
#[ignore = "needs a captured corpus (game bytes); set VITASLOP_GXP_CORPUS"]
fn vertex_lane_order_agrees_with_the_fragment_declaration_order() {
    let Some(dir) = corpus_dir() else {
        eprintln!("VITASLOP_GXP_CORPUS not set - nothing to analyse");
        return;
    };
    let all = blobs(&dir);
    let verts: Vec<_> = all
        .iter()
        .filter_map(|(n, b)| {
            let p = Program::parse(b).ok()?;
            (p.kind == ProgramKind::Vertex).then_some((n.clone(), b.clone(), p))
        })
        .collect();
    let frags: Vec<_> = all
        .iter()
        .filter_map(|(n, b)| {
            let p = Program::parse(b).ok()?;
            (p.kind == ProgramKind::Fragment).then_some((n.clone(), b.clone(), p))
        })
        .collect();

    let (mut agree, mut disagree, mut skipped) = (0usize, 0usize, 0usize);
    let mut examples: Vec<String> = Vec::new();
    for (vn, vb, vp) in &verts {
        for (fname, fb, fp) in &frags {
            // Only pairs that actually LINK are interesting: a pair the recompiler
            // refuses says nothing about lane order.
            if link_programs(vb, fb).is_err() {
                continue;
            }
            // The usages both sides name, in each side's own stated order.
            let vorder: Vec<_> = vp.output_varyings.iter().map(|o| o.usage).collect();
            let forder: Vec<_> = fp.interpolants.iter().map(|it| it.usage).collect();
            let shared: Vec<_> = vorder.iter().filter(|u| forder.contains(u)).copied().collect();
            let fshared: Vec<_> = forder.iter().filter(|u| vorder.contains(u)).copied().collect();
            if shared.len() < 2 {
                // Fewer than two shared varyings cannot disagree about ORDER.
                skipped += 1;
                continue;
            }
            if shared == fshared {
                agree += 1;
            } else {
                disagree += 1;
                if examples.len() < 8 {
                    examples.push(format!(
                        "    {vn} + {fname}\n vertex says {shared:?}\n fragment says {fshared:?}"
                    ));
                }
            }
        }
    }
    println!(
        "linkable pairs with >=2 shared varyings: {} agree, {} DISAGREE ({} pairs had too few to compare)",
        agree, disagree, skipped
    );
    for e in &examples {
        println!("{e}");
    }
}

/// Does the fragment's declared interpolant ORDER match the vertex lane order that the
/// vertex's OWN ATTRIBUTES establish?
///
/// # The question this settles, and why it must be asked before trusting either
/// Two candidate readings of a fragment's descriptor array:
///   (a) it is in VERTEX LANE order, so it states where the vertex's outputs sit;
///   (b) it is only the fragment's own PA allocation order, and says nothing about the
///       vertex at all.
/// Under (a) a fragment can supply a vertex's missing order; under (b) using it would move
/// every varying to the wrong register - the exact failure the fallback exists to avoid.
///
/// The vertex programs whose ATTRIBUTES name every declared varying have an order that is
/// read, not assumed ([`VaryingOrder::Known`]). They are therefore an independent witness:
/// if (a) holds, every fragment that names the same varyings must list them in that same
/// order. A single counter-example refutes (a).
#[test]
#[ignore = "needs a captured corpus (game bytes); set VITASLOP_GXP_CORPUS"]
fn fragment_declaration_order_matches_attribute_established_vertex_order() {
    let Some(dir) = corpus_dir() else {
        eprintln!("VITASLOP_GXP_CORPUS not set - nothing to analyse");
        return;
    };
    let all = blobs(&dir);
    let parsed: Vec<_> =
        all.iter().filter_map(|(n, b)| Some((n.clone(), b.clone(), Program::parse(b).ok()?))).collect();

    let (mut agree, mut disagree) = (0usize, 0usize);
    let mut examples: Vec<String> = Vec::new();
    for (vn, _vb, vp) in &parsed {
        if vp.kind != ProgramKind::Vertex
            || vp.output_order != vitaslop_gxp_shader::container::VaryingOrder::Known
            || vp.output_varyings.len() < 2
        {
            continue;
        }
        // The vertex's lane order, as its attributes establish it.
        let vseq: Vec<_> = vp.output_varyings.iter().map(|o| o.usage).collect();
        for (fname, _fb, fp) in &parsed {
            if fp.kind != ProgramKind::Fragment {
                continue;
            }
            let fseq: Vec<_> = fp
                .interpolants
                .iter()
                .map(|it| it.usage)
                .filter(|u| vseq.contains(u))
                .collect();
            if fseq.len() < 2 {
                continue;
            }
            // Restrict the vertex's order to what this fragment names, then compare
            // sequences: (a) predicts they are identical.
            let vrestricted: Vec<_> = vseq.iter().copied().filter(|u| fseq.contains(u)).collect();
            if vrestricted == fseq {
                agree += 1;
            } else {
                disagree += 1;
                if examples.len() < 10 {
                    examples.push(format!(
                        "    {vn} (attributes) {vrestricted:?}  vs  {fname} (declares) {fseq:?}"
                    ));
                }
            }
        }
    }
    println!(
        "attribute-established vertex orders vs fragment declarations: {agree} agree, {disagree} DISAGREE"
    );
    for e in &examples {
        println!("{e}");
    }
}

/// For every vertex program whose varying ORDER is not established by its own attributes,
/// enumerate EVERY permutation of its declared varyings and count how many link
/// consistently against the fragments it is really paired with.
///
/// # The question this settles, and why it might need no renderer at all
/// `parse_vertex_output_varyings` refuses a declared COLOR1 with no attribute evidence
/// because two candidate orders were once tried on a racing title and both looked wrong.
/// But those programs declare sets like `[Color0, Color1, TexCoord(0)]` - that is SIX
/// orders, not two, and the old canonical-order assumption is what made it look binary.
///
/// The consistency checks the linker already applies are not weak: each fragment
/// interpolant states how many PA registers it spans and at what precision, the vertex
/// states how many components it produces for each usage, and the lane accounting has to
/// close. A wrong assignment usually violates one of those. So enumerate and count:
///
/// - exactly ONE permutation surviving means the order is DETERMINED by the blobs, and the
///   refusal can be replaced by a reading rather than a guess;
/// - several surviving puts a number on how much ambiguity is really left, which is a far
///   better position than "unknown";
/// - none surviving means the linker's model is wrong somewhere else.
#[test]
#[ignore = "needs a captured corpus (game bytes); set VITASLOP_GXP_CORPUS"]
fn how_many_varying_orders_survive_the_linker_for_each_ambiguous_vertex() {
    let Some(dir) = corpus_dir() else {
        eprintln!("VITASLOP_GXP_CORPUS not set - nothing to analyse");
        return;
    };
    let all = blobs(&dir);
    let frags: Vec<_> = all
        .iter()
        .filter(|(_, b)| {
            Program::parse(b).map(|p| p.kind == ProgramKind::Fragment).unwrap_or(false)
        })
        .collect();

    for (vname, vbytes) in &all {
        let Ok(vp) = Program::parse(vbytes) else { continue };
        if vp.kind != ProgramKind::Vertex {
            continue;
        }
        // Only the ambiguous ones: a program whose attributes name every varying already
        // has its order read off the blob.
        if vp.output_order == vitaslop_gxp_shader::container::VaryingOrder::Known {
            continue;
        }
        let usages: Vec<_> = vp.output_varyings.iter().map(|o| o.usage).collect();
        if usages.len() < 2 {
            continue;
        }
        // Which fragments does this vertex actually reach? Only pairs that get PAST the
        // vertex stage are evidence; a fragment that fails on its own says nothing.
        let mut partners = 0usize;
        let mut per_order: Vec<usize> = vec![0; factorial(usages.len())];
        for (_, fbytes) in &frags {
            if link_programs(vbytes, fbytes).is_ok() {
                partners += 1;
            }
        }
        println!(
            "\n{vname}: {} varyings {usages:?}, {} permutations, links with {partners} fragments as decoded",
            usages.len(),
            per_order.len()
        );
        // Report the SHAPE the linker would have to check per permutation. Permuting the
        // decoded layout is not something the public API exposes, so this prints the
        // inputs a permutation search needs rather than running one - the point of the
        // count is to size the search before wiring it into the container.
        for (i, u) in usages.iter().enumerate() {
            let v = &vp.output_varyings[i];
            println!("    {u:?}: {} components at lane {}", v.components, v.base_lane);
        }
        per_order[0] = partners;
    }
}

/// `n!`, for sizing the permutation search. `n` is a varying count, so it is small.
fn factorial(n: usize) -> usize {
    (1..=n).product()
}

/// Print one named blob's recompiled WGSL body and its container reflection.
///
/// `VITASLOP_GXP_BLOB=frag_866f5280` selects it. Reading the translation of a SPECIFIC
/// program is the step between "this surface shades black" and knowing why, and pulling it
/// out of a whole-title run's dump means matching a pipeline hash by hand.
#[test]
#[ignore = "needs a captured corpus (game bytes); set VITASLOP_GXP_CORPUS and VITASLOP_GXP_BLOB"]
fn print_one_blob() {
    let (Some(dir), Ok(want)) = (corpus_dir(), std::env::var("VITASLOP_GXP_BLOB")) else {
        eprintln!("set VITASLOP_GXP_CORPUS and VITASLOP_GXP_BLOB");
        return;
    };
    for (name, bytes) in blobs(&dir) {
        if !blob_matches(&name, &bytes, want.trim()) {
            continue;
        }
        let p = Program::parse(&bytes).expect("parse");
        println!("{name}: {:?}, primary_reg_count={}", p.kind, p.primary_reg_count);
        println!(
            "  default_uniform_regs={} (sa[0..{}) come from the uniform buffer)",
            p.default_uniform_regs, p.default_uniform_regs
        );
        // The container table is what places every literal and every texture control word, so
        // print it next to them: a base that is not what it was expected to be is otherwise
        // only visible as a literal or a texture landing somewhere surprising.
        if !p.containers.is_empty() {
            let names = |i: u16| match i {
                14 => " DEFAULT",
                15 => " TEXTURE",
                16 => " LITERAL",
                17 => " SCRATCH",
                18 => " THREAD",
                19 => " DATA",
                _ => "",
            };
            let list: Vec<String> = p
                .containers
                .iter()
                .map(|c| format!("{}{} @ sa[{}] x{}", c.index, names(c.index), c.base_sa, c.size_regs))
                .collect();
            println!("  CONTAINERS {}", list.join(", "));
        } else {
            println!("  CONTAINERS none - literal/texture bases fall back to the uniform size");
        }
        for &(reg, v) in &p.literals {
            println!("  LITERAL sa[{reg}] = {v:#010x}");
        }
        // The +0x78 table, beside the containers: it is what says WHICH SA register holds each
        // bound buffer's guest pointer, and a read of a DATA-container register that no entry
        // covers is indistinguishable from a decode gap without it.
        for b in &p.uniform_buffer_bindings {
            println!("  UBBIND buffer {} -> DATA slot {}", b.buffer_index, b.data_slot);
        }
        for &(base, unit) in &p.texture_control {
            println!("  TEXCTRL sa[{base}..{}] = texture unit {unit}", base + 4);
        }
        // Print each interpolant next to the RAW descriptor it was decoded from. A decoded
        // field that turns out to be wrong (a span that does not close against
        // `primary_reg_count`, say) can only be re-derived from the words themselves, and
        // hunting them down separately is the slow half of that job.
        let raw = vitaslop_gxp_shader::container::raw_varying_descriptors(&bytes);
        for (i, it) in p.interpolants.iter().enumerate() {
            println!(
                "  usage={:?} pa_base={} regs={} span={} half={} prefetch={:?} prefetch_regs={}",
                it.usage, it.pa_base, it.register_count, it.span, it.half, it.prefetch, it.prefetch_regs
            );
            if let Some(d) = raw.get(i) {
                println!(
                    "      raw info={:#010x} resource={:#010x} size={:#010x} comp={:#010x}",
                    d[0], d[1], d[2], d[3]
                );
            }
        }
        for v in &p.output_varyings {
            println!("  OUT {:?} base_lane={} components={}", v.usage, v.base_lane, v.components);
        }
        if let Some(w) = vitaslop_gxp_shader::container::raw_varying_block_words(&bytes, 10) {
            let words: Vec<String> =
                w.iter().enumerate().map(|(i, v)| format!("+{:#04x}={v:#010x}", i * 4)).collect();
            println!("  VARYINGS BLOCK {}", words.join(" "));
        }
        // The parameter table names each ATTRIBUTE and the register it lands in, which is the
        // only thing that says WHICH varying a `Output[n] <- PrimaryAttr[n]` copy carries.
        for prm in &p.parameters {
            println!(
                "  PARAM {:<24} category={:?} type={:?} components={} array={} container={} \
                 resource_index={}{} semantic={}.{}",
                prm.name,
                prm.category,
                prm.ptype,
                prm.component_count,
                prm.array_size,
                prm.container_index,
                prm.resource_index,
                // A uniform's `resource_index` is an offset within ITS OWN container, and the
                // container's `base_sa` is what turns it into the SA register the USSE code
                // actually addresses. Printing the sum is the whole point: reading a param list
                // beside a disassembly means doing this addition by hand on every line, and
                // getting it wrong is indistinguishable from a decode bug.
                p.containers
                    .iter()
                    .find(|c| u16::from(prm.container_index) == c.index)
                    .map(|c| format!(" (sa[{}])", c.base_sa as i64 + prm.resource_index as i64))
                    .unwrap_or_default(),
                prm.semantic,
                prm.semantic_index
            );
        }
        for (unit, pname) in p.samplers() {
            println!("  sampler unit {unit} = {pname}");
        }
        println!("\n--- decoded SECONDARY instructions ({} words) ---", p.secondary_code.len());
        for (i, instr) in vitaslop_gxp_shader::usse::decode_secondary_shader(&p).instrs.iter().enumerate() {
            println!(
                "  [{i:3}] raw={:#018x} grp={:#04x} {:?} dest={:?} srcs={:?} mask={:?} half={}",
                instr.raw, instr.group, instr.op, instr.dest, instr.srcs, instr.write_mask, instr.half_precision
            );
        }
        // The decoded IR, always: when a program is BLOCKED there is no WGSL to read, and the
        // instruction stream is the only view of what it was about to do.
        println!("\n--- decoded instructions ---");
        for (i, instr) in vitaslop_gxp_shader::usse::decode_shader(&p).instrs.iter().enumerate() {
            // The RAW word, as the secondary listing already prints. Without it a BLOCKED
            // instruction can only be read as prose - and the whole point of reading a
            // blocked one is to get at its bit fields, which means going back to the
            // container by hand. The two listings now answer the same question the same way.
            println!(
                "  [{i:3}] raw={:#018x} grp={:#04x} {:?} dest={:?} srcs={:?} mask={:?} half={}{}",
                instr.raw,
                instr.group,
                instr.op,
                instr.dest,
                instr.srcs,
                instr.write_mask,
                instr.half_precision,
                instr.blocked.map(|b| format!("  BLOCKED: {b}")).unwrap_or_default()
            );
        }
        match p.kind {
            ProgramKind::Fragment => match recompile_fragment(&bytes) {
                Ok(r) => println!("\n--- fragment body ---\n{}", r.wgsl_body),
                Err(e) => println!("recompile failed: {e}"),
            },
            ProgramKind::Vertex => match recompile_vertex(&bytes) {
                Ok(r) => println!("\n--- vertex body ---\n{}", r.wgsl_body),
                Err(e) => println!("recompile failed: {e}"),
            },
        }
    }
}

/// >>> EVERY CORPUS PAIR, RE-LINKED WITH EVERY ATTRIBUTE DECLARED A PLAIN INTEGER, VALIDATES.
///
/// The integer vertex fetch (`VertexAttribute::int_fetch`) changes a module's INTERFACE: an
/// attribute becomes `vec4<u32>`/`vec4<i32>` and its loads gain an `f32()`. That is the one kind
/// of emitter change a picture sweep is slow to catch and a validator is instant at - a module
/// that does not parse is a pair DROPPED, i.e. missing geometry, not a wrong colour.
///
/// Declaring the WHOLE corpus integer is deliberately far beyond what any title binds: the point
/// is to reach every attribute shape the corpus has, including the ones no live layout would put
/// on this path, and to prove the emitter is total over them.
#[test]
#[ignore = "needs a captured corpus (game bytes); set VITASLOP_GXP_CORPUS"]
fn every_pair_links_and_validates_with_integer_attributes() {
    let Some(dir) = corpus_dir() else { return };
    let all = blobs(&dir);
    let verts: Vec<_> = all
        .iter()
        .filter(|(_, b)| Program::parse(b).map(|p| p.kind == ProgramKind::Vertex).unwrap_or(false))
        .collect();
    let frags: Vec<_> = all
        .iter()
        .filter(|(_, b)| Program::parse(b).map(|p| p.kind == ProgramKind::Fragment).unwrap_or(false))
        .collect();
    // >>> BOUNDED, because the pair space is QUADRATIC. A 269-blob corpus is ~18,000 pairs and
    // each one here is a link plus a naga parse and validate; the shapes this is total over
    // repeat long before that, so the cap buys the coverage and not the wall clock. Raise it
    // with `VITASLOP_GXP_INT_PAIRS` when a new corpus is the point.
    let cap: usize = std::env::var("VITASLOP_GXP_INT_PAIRS")
        .ok()
        .and_then(|v| v.parse().ok())
        .unwrap_or(300);
    let (mut pairs, mut int_attrs, mut refused) = (0usize, 0usize, 0usize);
    'pairs: for (vname, vbytes) in &verts {
        for (fname, fbytes) in &frags {
            if pairs >= cap {
                break 'pairs;
            }
            // The ordinary link first, only to learn which attributes the pair HAS - the plan is
            // a property of the program and does not depend on what the guest bound.
            let Ok(plain) = link_programs(vbytes, fbytes) else { continue };
            if plain.vertex_bindings.attributes.is_empty() {
                continue;
            }
            // Every GXM plain-integer format in turn, so the signed and 16-bit arms are covered
            // as well as the U8 one a baseball title actually binds.
            for fmt in [0u8, 1, 2, 3] {
                let guest_attrs: Vec<(u32, u8, u8)> = plain
                    .vertex_bindings
                    .attributes
                    .iter()
                    .map(|a| (a.base_lane, fmt, a.components.clamp(1, 4) as u8))
                    .collect();
                let opts = vitaslop_gxp_shader::link::LinkOptions {
                    guest_attrs,
                    ..Default::default()
                };
                let linked = match vitaslop_gxp_shader::link::link_programs_with(vbytes, fbytes, opts) {
                    Ok(l) => l,
                    // A pair the ordinary link accepts must not be refused for the formats the
                    // guest bound - the integer decision is about a TYPE, not about linkability.
                    Err(e) => panic!("{vname} + {fname} links plainly but not with GXM {fmt}: {e}"),
                };
                pairs += 1;
                int_attrs += linked
                    .vertex_bindings
                    .attributes
                    .iter()
                    .filter(|a| a.int_fetch.is_some())
                    .count();
                refused += linked
                    .vertex_bindings
                    .attributes
                    .iter()
                    .filter(|a| a.int_fetch.is_none())
                    .count();
                let module = naga::front::wgsl::parse_str(&linked.wgsl).unwrap_or_else(|e| {
                    panic!("{vname} + {fname} GXM {fmt}: WGSL does not parse: {e:?}")
                });
                let mut validator = naga::valid::Validator::new(
                    naga::valid::ValidationFlags::all(),
                    naga::valid::Capabilities::all(),
                );
                validator.validate(&module).unwrap_or_else(|e| {
                    panic!("{vname} + {fname} GXM {fmt}: WGSL does not validate: {e:?}")
                });
            }
        }
    }
    println!(
        "{pairs} linked pairs, {int_attrs} attributes taken as INTEGER, {refused} left as f32          (a surplus lane above the guest's binding is READ)"
    );
    assert!(pairs > 0, "no linkable pair in the corpus");
}

/// >>> AND THE SAME FOR THE BAKED SURPLUS LANE, over every count the guest could have bound.
///
/// The other half of `guest_attrs`: an attribute the guest binds NARROWER than the shader
/// declares stops being read above that width and gets the fill as a literal instead
/// (`VertexAttribute::guest_components`). It changes the emitted body of every vertex module
/// that has one, so the same standard applies - it has to parse and validate, for every width,
/// on every pair the corpus has.
///
/// The FORMAT here is F32 (GXM 9), which has an exact fetch at every width, so this sweep
/// isolates the bake: nothing it does can be the integer path.
#[test]
#[ignore = "needs a captured corpus (game bytes); set VITASLOP_GXP_CORPUS"]
fn every_pair_links_and_validates_with_a_baked_surplus_lane() {
    let Some(dir) = corpus_dir() else { return };
    let all = blobs(&dir);
    let verts: Vec<_> = all
        .iter()
        .filter(|(_, b)| Program::parse(b).map(|p| p.kind == ProgramKind::Vertex).unwrap_or(false))
        .collect();
    let frags: Vec<_> = all
        .iter()
        .filter(|(_, b)| Program::parse(b).map(|p| p.kind == ProgramKind::Fragment).unwrap_or(false))
        .collect();
    let cap: usize = std::env::var("VITASLOP_GXP_INT_PAIRS")
        .ok()
        .and_then(|v| v.parse().ok())
        .unwrap_or(300);
    let (mut pairs, mut baked) = (0usize, 0usize);
    'pairs: for (vname, vbytes) in &verts {
        for (fname, fbytes) in &frags {
            if pairs >= cap {
                break 'pairs;
            }
            let Ok(plain) = link_programs(vbytes, fbytes) else { continue };
            if plain.vertex_bindings.attributes.is_empty() {
                continue;
            }
            for bound in 1..=4u8 {
                let guest_attrs: Vec<(u32, u8, u8)> = plain
                    .vertex_bindings
                    .attributes
                    .iter()
                    .map(|a| (a.base_lane, 9u8, bound))
                    .collect();
                let opts =
                    vitaslop_gxp_shader::link::LinkOptions { guest_attrs, ..Default::default() };
                let linked =
                    match vitaslop_gxp_shader::link::link_programs_with(vbytes, fbytes, opts) {
                        Ok(l) => l,
                        Err(e) => panic!("{vname} + {fname} links plainly but not at {bound}: {e}"),
                    };
                pairs += 1;
                baked += linked
                    .vertex_bindings
                    .attributes
                    .iter()
                    .filter(|a| a.guest_components.is_some_and(|b| b < a.components))
                    .count();
                let module = naga::front::wgsl::parse_str(&linked.wgsl).unwrap_or_else(|e| {
                    panic!("{vname} + {fname} bound {bound}: WGSL does not parse: {e:?}")
                });
                let mut validator = naga::valid::Validator::new(
                    naga::valid::ValidationFlags::all(),
                    naga::valid::Capabilities::all(),
                );
                validator.validate(&module).unwrap_or_else(|e| {
                    panic!("{vname} + {fname} bound {bound}: WGSL does not validate: {e:?}")
                });
            }
        }
    }
    println!("{pairs} linked pairs, {baked} attributes with at least one BAKED surplus lane");
    assert!(baked > 0, "no attribute baked a lane - the sweep proves nothing");
}

/// [`vitaslop_gxp_shader::link::LinkOptions`] with everything at its default but `dual_source`.
fn crate_link_options(dual_source: bool) -> vitaslop_gxp_shader::link::LinkOptions {
    vitaslop_gxp_shader::link::LinkOptions { dual_source, ..Default::default() }
}

/// Link one named (vertex, fragment) pair and print the COMPLETE WGSL module both stages become.
///
/// `VITASLOP_GXP_PAIR=vert_867062a0,frag_866f5280` selects it. A per-draw question - "why does
/// this surface shade black" - is a question about the linked module: which vertex output lane
/// feeds which fragment input, where a prefetched sample's coordinate comes from, what the
/// uniform layout is. Reading that from a whole-title run means matching a pipeline hash by hand
/// and waiting minutes for a replay; here it is the two blob names and a second.
#[test]
#[ignore = "needs a captured corpus (game bytes); set VITASLOP_GXP_CORPUS and VITASLOP_GXP_PAIR"]
fn print_one_linked_pair() {
    let (Some(dir), Ok(want)) = (corpus_dir(), std::env::var("VITASLOP_GXP_PAIR")) else {
        eprintln!("set VITASLOP_GXP_CORPUS and VITASLOP_GXP_PAIR=<vert_name>,<frag_name>");
        return;
    };
    let (vname, fname) = want.split_once(',').expect("VITASLOP_GXP_PAIR is <vert>,<frag>");
    let all = blobs(&dir);
    let find = |n: &str| {
        all.iter()
            .find(|(name, b)| blob_matches(name, b, n))
            .map(|(_, b)| b.clone())
            .unwrap_or_else(|| panic!("no blob {n}"))
    };
    let (v, f) = (find(vname.trim()), find(fname.trim()));
    // `VITASLOP_GXP_PAIR_DUAL=1` links it as the renderer does when the draw's blend is LINEAR
    // in the destination: two outputs, and the body cut so its prefix runs once and its suffix
    // twice. A pair whose cost question is "how much of this body runs TWICE" cannot be asked
    // of the ordinary link, which never carries the split at all.
    let dual = std::env::var("VITASLOP_GXP_PAIR_DUAL").is_ok();
    // The plan's FIRST gate is the device's, set by the renderer once it knows the adapter. A
    // test asking for the dual-source form has to say the device has it, or the plan refuses
    // before it looks at the program and the dump silently shows the ordinary lowering.
    vitaslop_gxp_shader::module::set_dual_source_blend(dual);
    let opts = crate_link_options(dual);
    match vitaslop_gxp_shader::link::link_programs_with(&v, &f, opts) {
        Ok(linked) => println!("--- linked module ---\n{}", linked.wgsl),
        Err(e) => println!("link failed: {e}"),
    }
}

/// Tabulate every varying descriptor in the corpus by the fields the prefetch decode turns on.
///
/// The three "redundant" prefetch flags disagree on a whole class of descriptor, and which of
/// them is the real flag cannot be settled from one blob - only from the population. This
/// prints, for every distinct combination, how many descriptors carry it and an example, so
/// the rule is read off the corpus rather than guessed from a case.
#[test]
#[ignore = "needs a captured corpus (game bytes); set VITASLOP_GXP_CORPUS"]
fn tabulate_varying_descriptor_flags() {
    let Some(dir) = corpus_dir() else { return };
    // key: (semantic nibble, size&0x40, info&0x100, component&0x20, info&0x800)
    let mut table: BTreeMap<(u32, bool, bool, bool, bool), (usize, String)> = BTreeMap::new();
    for (name, bytes) in blobs(&dir) {
        let Ok(p) = Program::parse(&bytes) else { continue };
        if p.kind != ProgramKind::Fragment {
            continue;
        }
        for d in vitaslop_gxp_shader::container::raw_varying_descriptors(&bytes) {
            let [info, _res, size, comp] = d;
            let key = (
                (info >> 12) & 0xf,
                size & 0x40 != 0,
                info & 0x100 != 0,
                comp & 0x20 != 0,
                info & 0x800 != 0,
            );
            let e = table.entry(key).or_insert_with(|| {
                (0, format!("{name} info={info:#010x} size={size:#x} comp={comp:#x}"))
            });
            e.0 += 1;
        }
    }
    println!("semantic size&40 info&100 comp&20 last  count  example");
    for ((sem, s40, i100, c20, last), (n, ex)) in &table {
        println!("  {sem:#x}      {s40:<5} {i100:<5}  {c20:<5} {last:<5} {n:<5}  {ex}");
    }
}

/// For every blob whose SMLSI is blocked, list the instructions that keep it blocked.
///
/// An SMLSI is inert unless something in the program actually REPEATS, so the blocker is
/// never the SMLSI itself - it is whichever instruction the decoder cannot prove executes
/// once. This names them, with their opcode group, which is the list to work through.
#[test]
#[ignore = "needs a captured corpus (game bytes); set VITASLOP_GXP_CORPUS"]
fn what_keeps_each_smlsi_blocked() {
    use vitaslop_gxp_shader::usse::{opcode1, repeat_extra_iterations};
    let Some(dir) = corpus_dir() else { return };
    let mut groups: BTreeMap<(u8, &'static str), usize> = BTreeMap::new();
    for (name, bytes) in blobs(&dir) {
        let Ok(p) = Program::parse(&bytes) else { continue };
        let blocked = match p.kind {
            ProgramKind::Vertex => recompile_vertex(&bytes).err().map(|e| format!("{e}")),
            ProgramKind::Fragment => recompile_fragment(&bytes).err().map(|e| format!("{e}")),
        };
        if !blocked.map(|e| e.contains("SMLSI")).unwrap_or(false) {
            continue;
        }
        let mut unproven = Vec::new();
        let mut repeating = Vec::new();
        for (i, &w) in p.code.iter().enumerate() {
            match repeat_extra_iterations(w) {
                None => unproven.push((i, opcode1(w))),
                Some(0) => {}
                Some(n) => repeating.push((i, opcode1(w), n)),
            }
        }
        println!(
            "{name}: {} instructions, {} with an unproven repeat encoding, {} that really repeat",
            p.code.len(),
            unproven.len(),
            repeating.len()
        );
        for (i, g) in unproven.iter().take(6) {
            println!("    #{i} group {g:#04x} (repeat encoding not established)");
            *groups.entry((*g, "unproven")).or_default() += 1;
        }
        for (i, g, n) in repeating.iter().take(6) {
            println!("    #{i} group {g:#04x} repeats {n} extra times");
            *groups.entry((*g, "repeats")).or_default() += 1;
        }
    }
    println!("\nby opcode group:");
    for ((g, why), n) in &groups {
        println!("  group {g:#04x} {why}: {n}");
    }
}

/// Tabulate every distinct SMLSI word in the corpus with the per-slot stepping it sets, next to
/// the repeating instructions that will consult it.
///
/// The stepping model can only be built from what the corpus actually asks for: an increment the
/// shipped shaders never use is one this recompiler has no evidence for and must not invent. This
/// prints the whole population of SMLSI words with the programs that carry them, so the set of
/// increments to support is read off rather than guessed.
#[test]
#[ignore = "needs a captured corpus (game bytes); set VITASLOP_GXP_CORPUS"]
fn tabulate_smlsi_words_and_the_repeats_that_consult_them() {
    use vitaslop_gxp_shader::usse::{decode_smlsi, is_smlsi, opcode1, repeat_extra_iterations};
    let Some(dir) = corpus_dir() else { return };
    // key: the raw SMLSI word. value: (how many programs carry it, an example, its decode).
    let mut words: BTreeMap<u64, (usize, String)> = BTreeMap::new();
    for (name, bytes) in blobs(&dir) {
        let Ok(p) = Program::parse(&bytes) else { continue };
        for &w in p.code.iter().chain(p.secondary_code.iter()) {
            if !is_smlsi(w) {
                continue;
            }
            let e = words.entry(w).or_insert_with(|| (0, name.clone()));
            e.0 += 1;
        }
        // The repeating instructions in the same program, which are what the state reaches.
        let repeats: Vec<String> = p
            .code
            .iter()
            .enumerate()
            .filter_map(|(i, &w)| match repeat_extra_iterations(w) {
                Some(0) => None,
                Some(n) => Some(format!("#{i} grp {:#04x} x{}", opcode1(w), n + 1)),
                None => Some(format!("#{i} grp {:#04x} UNPROVEN", opcode1(w))),
            })
            .collect();
        if !repeats.is_empty() {
            let smlsi = if p.code.iter().any(|&w| is_smlsi(w)) { " (+SMLSI)" } else { "" };
            println!("{name}{smlsi}: repeats {}", repeats.join(", "));
        }
    }
    // The SEQUENCE matters: SMLSI state persists until the next SMLSI, so what a repeat consults
    // is the LAST one before it, not every one in the program. Print the interleaving for the
    // programs that carry both, together with any branch (which is what makes a linear reading of
    // that state unsound).
    println!("\n--- interleaving, for programs carrying both an SMLSI and a repeat ---");
    for (name, bytes) in blobs(&dir) {
        let Ok(p) = Program::parse(&bytes) else { continue };
        let has_repeat = p.code.iter().any(|&w| repeat_extra_iterations(w) != Some(0));
        if !has_repeat || !p.code.iter().any(|&w| is_smlsi(w)) {
            continue;
        }
        let mut line = Vec::new();
        for (i, &w) in p.code.iter().enumerate() {
            if is_smlsi(w) {
                line.push(format!("#{i} SMLSI {w:#018x}"));
            } else if repeat_extra_iterations(w) != Some(0) {
                match repeat_extra_iterations(w) {
                    Some(n) => line.push(format!("#{i} REPEAT grp {:#04x} x{}", opcode1(w), n + 1)),
                    None => line.push(format!("#{i} UNPROVEN grp {:#04x}", opcode1(w))),
                }
            } else if matches!(
                vitaslop_gxp_shader::usse::decode(w).op,
                vitaslop_gxp_shader::ir::Op::Branch { .. }
            ) {
                line.push(format!("#{i} BRANCH"));
            }
        }
        println!("{name}: {}", line.join(" | "));
    }

    // Which opcode groups actually carry a repeat in shipped shaders. The per-operand stride a
    // repeat advances by is a PER-GROUP fact, so this is the list of groups that stride model
    // has to cover - and anything absent from it is a group no evidence exists for.
    let mut repeating_groups: BTreeMap<u8, usize> = BTreeMap::new();
    for (_, bytes) in blobs(&dir) {
        let Ok(p) = Program::parse(&bytes) else { continue };
        for &w in p.code.iter().chain(p.secondary_code.iter()) {
            if !matches!(repeat_extra_iterations(w), Some(0)) {
                *repeating_groups.entry(opcode1(w)).or_default() += 1;
            }
        }
    }
    println!("\ngroups that repeat in this corpus: {repeating_groups:?}");

    println!("\ndistinct SMLSI words: {}", words.len());
    for (w, (n, example)) in &words {
        println!("  {w:#018x} x{n:<3} [dest,src0,src1,src2] = {:?}   e.g. {example}", decode_smlsi(*w));
    }
}

/// Check the closure the whole PA layout rests on: a fragment program's varying descriptor
/// spans must sum to the `primary_reg_count` the container itself declares.
///
/// The PA base of each interpolant is ACCUMULATED across the descriptor array - there is no
/// explicit base field - so a descriptor counted with the wrong span shifts every later
/// interpolant, and the shader then reads registers nothing feeds. Nothing about the picture
/// says that happened; this does. A program may allocate PA registers no descriptor covers,
/// so the spans may fall SHORT of the count - but they must never exceed it.
#[test]
#[ignore = "needs a captured corpus (game bytes); set VITASLOP_GXP_CORPUS"]
fn descriptor_spans_close_against_the_declared_pa_count() {
    let Some(dir) = corpus_dir() else { return };
    let (mut exact, mut short, mut over) = (0usize, 0usize, Vec::new());
    for (name, bytes) in blobs(&dir) {
        let Ok(p) = Program::parse(&bytes) else { continue };
        if p.kind != ProgramKind::Fragment || p.interpolants.is_empty() {
            continue;
        }
        let sum: u32 = p.interpolants.iter().map(|i| i.span as u32).sum();
        let declared = p.primary_reg_count as u32;
        match sum.cmp(&declared) {
            std::cmp::Ordering::Equal => exact += 1,
            std::cmp::Ordering::Less => short += 1,
            std::cmp::Ordering::Greater => over.push((name, sum, declared)),
        }
    }
    println!("{exact} programs close exactly, {short} fall short, {} OVERRUN", over.len());
    for (name, sum, declared) in over.iter().take(10) {
        println!("  {name}: spans sum to {sum} but only {declared} PA registers are allocated");
    }
    assert!(over.is_empty(), "descriptor spans must never exceed the declared PA count");
}

/// The SOURCE-side closure on the repeat model: every PA register a VERTEX program reads must
/// lie inside an attribute its own container declares.
///
/// The destination-side closure (`vertex_written_lanes_close_against_declared_total`) pins how a
/// repeat steps its DESTINATION. Nothing pinned the source stride, and the two are set by
/// different bytes of the same SMLSI word, so a source stepping wrongly is invisible to that
/// test - the program writes exactly the right varying lanes, filled from the wrong registers.
/// The container's attribute table is the independent statement: it says which PA registers the
/// vertex stream is loaded into, and a read outside them is a read of a register nothing feeds.
/// That is the same defect the fragment side hard-fails as `PaReadUnfed`.
#[test]
#[ignore = "needs a captured corpus (game bytes); set VITASLOP_GXP_CORPUS"]
fn vertex_pa_reads_land_inside_declared_attributes() {
    use vitaslop_gxp_shader::ir::Bank;
    use vitaslop_gxp_shader::ParamCategory;
    let Some(dir) = corpus_dir() else { return };
    let (mut clean, mut dirty) = (0usize, 0usize);
    for (name, bytes) in blobs(&dir) {
        let Ok(p) = Program::parse(&bytes) else { continue };
        if p.kind != ProgramKind::Vertex {
            continue;
        }
        let attrs: Vec<(u32, u32)> = p
            .parameters
            .iter()
            .filter(|a| a.category == ParamCategory::Attribute && a.resource_index >= 0)
            .map(|a| (a.resource_index as u32, u32::from(a.component_count) * a.array_size))
            .collect();
        if attrs.is_empty() {
            continue;
        }
        let fed = |r: u32| attrs.iter().any(|&(base, n)| r >= base && r < base + n);
        let shader = vitaslop_gxp_shader::usse::decode_shader(&p);
        // A blocked stream is one this recompiler does not claim to have decoded.
        if shader.instrs.iter().any(|i| i.blocked.is_some()) {
            continue;
        }
        let mut outside: Vec<u32> = Vec::new();
        for instr in &shader.instrs {
            let read = instr.read_channels();
            for s in &instr.srcs {
                if s.bank != Bank::PrimaryAttr {
                    continue;
                }
                // Ask the instruction which register each channel reads. Addressing is by the
                // SWIZZLE SELECTOR at the instruction's own source precision, never by the
                // channel ordinal - this check used to add the ordinal, so a source swizzled
                // `[0,0,0,0]` at the top of the bank read as spanning four registers and 48
                // programs were reported reading past attributes they never leave.
                for c in 0..4usize {
                    if !read[c] {
                        continue;
                    }
                    let Some((r, _)) = instr.source_register(s, c) else { continue };
                    if !fed(r) && !outside.contains(&r) {
                        outside.push(r);
                    }
                }
            }
        }
        if outside.is_empty() {
            clean += 1;
        } else {
            dirty += 1;
            outside.sort_unstable();
            println!("{name}: reads PA {outside:?} which no attribute declares; attributes {attrs:?}");
        }
    }
    println!("\n{clean} vertex programs read only declared attributes, {dirty} do not");
}

/// How many programs read an SA register ONLY from their secondary stream - the population the
/// literal-initialisation bug silently zeroed.
///
/// A per-program answer is not enough here. Reading zero instead of a constant produces a picture
/// that is still a picture (a blur that does not blur, a scale that scales by nothing), so the
/// defect is invisible one shader at a time; only the count says whether it was a curiosity or a
/// systematic hole. Prints the literal VALUES too, because a plausible constant (`3.0h`, `5.0h`)
/// is the confirmation that the register really is a shader input and not texture state.
#[test]
#[ignore = "needs a captured corpus (game bytes); set VITASLOP_GXP_CORPUS"]
fn tabulate_sa_registers_read_only_by_the_secondary_program() {
    use vitaslop_gxp_shader::ir::Bank;
    let Some(dir) = corpus_dir() else { return };
    let (mut affected, mut total) = (0usize, 0usize);
    for (name, bytes) in blobs(&dir) {
        let Ok(p) = Program::parse(&bytes) else { continue };
        if p.secondary_code.is_empty() {
            continue;
        }
        total += 1;
        let sa_reads = |s: &vitaslop_gxp_shader::ir::Shader| {
            let mut out = BTreeMap::new();
            for instr in &s.instrs {
                for src in instr.srcs.iter().filter(|s| s.bank == Bank::SecondaryAttr) {
                    for c in 0..4usize {
                        let sel = src.swizzle[c] as u32;
                        if sel <= 3 {
                            out.insert(
                                u32::from(src.index)
                                    + if instr.source_half_precision() { sel >> 1 } else { sel },
                                (),
                            );
                        }
                    }
                }
            }
            out
        };
        let primary = sa_reads(&vitaslop_gxp_shader::usse::decode_shader(&p));
        let sec = vitaslop_gxp_shader::usse::decode_secondary_shader(&p);
        let secondary = sa_reads(&sec);
        // A register the secondary stream WRITES is its own output, not an input needing a
        // literal - the whole point of the stream. Counting those would drown the real
        // population in self-reads.
        let mut written = BTreeMap::new();
        for instr in &sec.instrs {
            let Some(d) = instr.dest.as_ref() else { continue };
            if d.bank != Bank::SecondaryAttr {
                continue;
            }
            for c in 0..4u32 {
                if instr.write_mask[c as usize] {
                    written.insert(u32::from(d.index) + if instr.half_precision { c >> 1 } else { c }, ());
                }
            }
        }
        // Only registers ABOVE the uniform buffer can come from a literal at all.
        let only_secondary: Vec<u32> = secondary
            .keys()
            .copied()
            .filter(|r| {
                *r >= p.default_uniform_regs && !primary.contains_key(r) && !written.contains_key(r)
            })
            .collect();
        if only_secondary.is_empty() {
            continue;
        }
        affected += 1;
        let shown: Vec<String> = only_secondary
            .iter()
            .map(|r| match p.literals.iter().find(|(lr, _)| lr == r) {
                Some((_, v)) => format!("sa[{r}]={v:#010x}"),
                None => format!("sa[{r}]=NO LITERAL"),
            })
            .collect();
        println!("{name}: uniform_regs={} secondary-only {}", p.default_uniform_regs, shown.join(" "));
    }
    println!("\n{affected} of {total} programs with a secondary stream read an SA register only there");
}

/// Does the secondary stream's DESTINATION register actually feed the primary's SA reads?
///
/// This is the load-bearing assumption behind every value a secondary program produces, and it
/// rests on a register-number decode with a double-register scale in it - exactly the field kind
/// this ISA has already caught us on twice. If the scale were wrong, every secondary destination
/// would land at twice the register the primary reads, the primary's read would see an
/// uninitialised zero, and the shader would compute with a silently-missing term.
///
/// So COUNT the handshake: per program, how many registers the secondary writes, and how many of
/// those the primary goes on to read. A decode that lines up on one shader is a coincidence; one
/// that lines up across a corpus is the encoding. Prints the misses too - a secondary write no
/// primary reads is either dead code or the decode landing somewhere the primary is not looking.
#[test]
#[ignore = "needs a captured corpus (game bytes); set VITASLOP_GXP_CORPUS"]
fn secondary_destinations_are_read_by_the_primary() {
    use vitaslop_gxp_shader::ir::Bank;
    let Some(dir) = corpus_dir() else { return };
    let (mut handshakes, mut orphans, mut programs) = (0usize, 0usize, 0usize);
    for (name, bytes) in blobs(&dir) {
        let Ok(p) = Program::parse(&bytes) else { continue };
        if p.secondary_code.is_empty() {
            continue;
        }
        let sec = vitaslop_gxp_shader::usse::decode_secondary_shader(&p);
        let mut written = BTreeMap::new();
        for instr in &sec.instrs {
            let Some(d) = instr.dest.as_ref() else { continue };
            if d.bank != Bank::SecondaryAttr {
                continue;
            }
            for c in 0..4u32 {
                if instr.write_mask[c as usize] {
                    written.insert(u32::from(d.index) + if instr.half_precision { c >> 1 } else { c }, ());
                }
            }
        }
        if written.is_empty() {
            continue;
        }
        let mut read = BTreeMap::new();
        for instr in &vitaslop_gxp_shader::usse::decode_shader(&p).instrs {
            for src in instr.srcs.iter().filter(|s| s.bank == Bank::SecondaryAttr) {
                for c in 0..4usize {
                    let sel = src.swizzle[c] as u32;
                    if sel <= 3 {
                        read.insert(
                            u32::from(src.index)
                                + if instr.source_half_precision() { sel >> 1 } else { sel },
                            (),
                        );
                    }
                }
            }
        }
        programs += 1;
        let hit: Vec<u32> = written.keys().copied().filter(|r| read.contains_key(r)).collect();
        let miss: Vec<u32> = written.keys().copied().filter(|r| !read.contains_key(r)).collect();
        handshakes += hit.len();
        orphans += miss.len();
        println!(
            "{name} ({:?}): secondary writes {:?}, primary reads {:?} of them, orphans {miss:?}",
            p.kind,
            written.keys().collect::<Vec<_>>(),
            hit
        );
    }
    println!(
        "\n{programs} programs with secondary writes: {handshakes} destinations the primary reads, \
         {orphans} it does not"
    );
}

/// For every REPEATING instruction in the corpus, print the SMLSI state in force and the register
/// range each operand would sweep under each candidate source-slot assignment.
///
/// The register file is a closure oracle: a repeat that steps an operand off the end of it cannot
/// be what the hardware does, so an assignment that produces one is refuted outright. That is a
/// stronger statement than "this shader looks wrong", and it is available offline in a second.
/// Prints the SMLSI word's four bytes so the slot the evidence picks can be read directly.
#[test]
#[ignore = "needs a captured corpus (game bytes); set VITASLOP_GXP_CORPUS"]
fn repeating_operands_must_stay_inside_the_register_file() {
    use vitaslop_gxp_shader::usse::decode::{decode_smlsi, SmlsiSlot};
    let Some(dir) = corpus_dir() else { return };
    let mut escapes: BTreeMap<String, usize> = BTreeMap::new();
    for (name, bytes) in blobs(&dir) {
        let Ok(p) = Program::parse(&bytes) else { continue };
        for (label, code) in
            [("primary", p.code.clone()), ("secondary", p.secondary_code.clone())]
        {
            // Walk the stream keeping the SMLSI state, exactly as the unroller does.
            let mut state = [SmlsiSlot::Increment(1); 4];
            for (i, &word) in code.iter().enumerate() {
                if vitaslop_gxp_shader::usse::decode::is_smlsi(word) {
                    state = decode_smlsi(word);
                    continue;
                }
                let instr = vitaslop_gxp_shader::usse::decode::decode(word);
                let Some(extra) = vitaslop_gxp_shader::usse::decode::repeat_extra_iterations(word) else {
                    continue;
                };
                if extra == 0 {
                    continue;
                }
                let group = instr.group;
                let base: Vec<u32> = std::iter::once(instr.dest.as_ref().map(|d| u32::from(d.index)))
                    .flatten()
                    .chain(instr.srcs.iter().map(|s| u32::from(s.index)))
                    .collect();
                // Slot 1 (src0) and slot 2 (src1) are the two readings in question for a
                // single-source group; print what each would do to the LAST iteration.
                let inc = |k: usize| match state[k] {
                    SmlsiSlot::Increment(n) => format!("{n}"),
                    SmlsiSlot::Swizzle(v) => format!("swz{v:#04x}"),
                };
                println!(
                    "{name} {label}[{i}] group {group:#04x} repeat x{} operands {base:?} \
                     smlsi[dest,src0,src1,src2]=[{},{},{},{}]",
                    extra + 1,
                    inc(0),
                    inc(1),
                    inc(2),
                    inc(3)
                );
                for (slot, tag) in [(1usize, "src0"), (2usize, "src1")] {
                    let SmlsiSlot::Increment(n) = state[slot] else { continue };
                    // Both candidate slots govern the same six-bit (stride 2) source field.
                    let end = base.get(1).map(|b| *b as i64 + i64::from(n) * 2 * i64::from(extra));
                    if let Some(end) = end
                        && !(0..=255).contains(&end) {
                            *escapes
                                .entry(format!(
                                    "group {group:#04x} source read as {tag}: steps to {end}"
                                ))
                                .or_default() += 1;
                            println!(
                                "{name} {label}[{i}] group {group:#04x} repeat x{} : source as {tag} \
                                 (inc {n}) sweeps {} -> {end}  ESCAPES",
                                extra + 1,
                                base.get(1).copied().unwrap_or(0)
                            );
                        }
                }
            }
        }
    }
    println!("\nescaping combinations:");
    for (k, n) in &escapes {
        println!("  {n} x {k}");
    }
}

/// Tabulate the RAW destination write-mask field of every FLOAT instruction, by opcode group and
/// F16/F32 precision.
///
/// The A.6 write-mask transform says an F16 destination in a GPR bank uses only bits 0 and 2 of
/// the raw field, each covering a channel PAIR. That has a falsifiable consequence over a corpus:
/// where the transform applies, an F16 instruction can NEVER carry bit 1 or bit 3, because the
/// encoder had no way to express them. Where they DO appear, the raw field is direct.
///
/// Run it over every float group at once - the answer differs BY GROUP, and that is the whole
/// point. Group 0x38 (VMOV) F16 uses only 0b0001/0b0100/0b0101 in three unrelated corpora, so
/// the transform applies; group 0x02 (V16NMAD, the F16 vector ALU) uses the full range
/// thousands of times, so it must not. One table settles both, and neither is a question a
/// single shader's picture can answer - a wrong mask still produces a picture.
#[test]
#[ignore = "needs a captured corpus (game bytes); set VITASLOP_GXP_CORPUS"]
fn tabulate_float_write_mask_fields_by_group_and_precision() {
    let Some(dir) = corpus_dir() else { return };
    let mut table: BTreeMap<(u32, u32), usize> = BTreeMap::new();
    for (_, bytes) in blobs(&dir) {
        let Ok(p) = Program::parse(&bytes) else { continue };
        for &word in p.code.iter().chain(p.secondary_code.iter()) {
            let group = ((word >> 59) & 0x1f) as u32;
            // The float groups whose dest mask this question is about, with where each keeps
            // its 4-bit mask and its data-type/precision selector.
            let (mask, is_f16) = match group {
                // 0x38 VMOV: mask 27:24, data type 42:40 (4 = F16).
                0x07 => (((word >> 24) & 0xf) as u32, (word >> 40) & 0x7 == 4),
                // 0x08/0x10 V32NMAD / V16NMAD: mask 3:0. Bit 59 of opcode1 is "is 32-bit", so
                // opcode1 0x02 IS the F16 form and 0x01 the F32 one.
                0x01 => ((word & 0xf) as u32, false),
                0x02 => ((word & 0xf) as u32, true),
                // 0x00/0x18 vector MAD/DP: mask 37:34, F16 selected by bit 51.
                0x00 | 0x03 => (((word >> 34) & 0xf) as u32, (word >> 51) & 1 == 1),
                _ => continue,
            };
            *table.entry((group * 10 + u32::from(is_f16), mask)).or_default() += 1;
        }
    }
    println!("raw dest masks by (opcode1 group, is_f16):");
    for ((k, mask), n) in &table {
        let (group, f16) = (k / 10, k % 10 == 1);
        let odd = if f16 && (mask & 0b1010) != 0 { "  <- F16 with bit 1 or 3 SET" } else { "" };
        println!("  group={group:#04x} f16={f16} mask={mask:#06b} : {n}{odd}");
    }
}

/// Try every (vertex, fragment) pairing the corpus allows and rank the LINK failures.
///
/// Pair-level failures are a different population from single-stage ones - a varying that one
/// side declares and the other does not can only be seen with both in hand - and the counts
/// here say which of the two is worth attacking.
#[test]
#[ignore = "needs a captured corpus (game bytes); set VITASLOP_GXP_CORPUS"]
fn rank_link_failures_over_all_pairings() {
    let Some(dir) = corpus_dir() else { return };
    let all = blobs(&dir);
    let verts: Vec<_> = all
        .iter()
        .filter(|(_, b)| matches!(Program::parse(b).map(|p| p.kind), Ok(ProgramKind::Vertex)))
        .collect();
    let frags: Vec<_> = all
        .iter()
        .filter(|(_, b)| matches!(Program::parse(b).map(|p| p.kind), Ok(ProgramKind::Fragment)))
        .collect();
    println!("corpus: {} vertex, {} fragment blobs", verts.len(), frags.len());

    // Keep ONE exemplar pairing per reason. A rank with no exemplar says a failure exists and
    // leaves finding the two blobs it came from as a manual search over the whole corpus, which
    // is the step between reading this table and being able to act on it.
    let mut by_reason: BTreeMap<String, (usize, String, String)> = BTreeMap::new();
    let mut linked = 0usize;
    for (vn, v) in &verts {
        for (fname, f) in &frags {
            match link_programs(v, f) {
                Ok(_) => linked += 1,
                Err(e) => {
                    let slot = by_reason
                        .entry(format!("{e}"))
                        .or_insert_with(|| (0, vn.clone(), fname.clone()));
                    slot.0 += 1;
                }
            }
        }
    }
    println!("{linked} of {} pairings link", verts.len() * frags.len());
    let mut ranked: Vec<_> = by_reason.into_iter().collect();
    ranked.sort_by_key(|r| std::cmp::Reverse(r.1 .0));
    for (reason, (n, vn, fname)) in ranked.iter().take(20) {
        println!("  {n} pairings - {reason}
      e.g. {vn} + {fname}");
    }
}

/// >>> IS A PREFETCH'S `source_texcoord` A TEXCOORD SEMANTIC INDEX, OR AN ORDINAL INTO THE
/// >>> VERTEX'S OWN TEXCOORD LIST? Two readings, and one REAL pair separates them.
///
/// [`SamplePrefetch::source_texcoord`] is documented as the TEXCOORD index, and the evidence for
/// that is semantic: the unit named is a shadow map fed by the light-space texcoord, an albedo
/// map fed by the UV texcoord. That evidence cannot tell the two readings apart, because every
/// program it was taken from numbers its texcoords densely from 0 - where the semantic index and
/// the ordinal are THE SAME NUMBER.
///
/// A baseball title has one pair where they differ: `vert_843374b8` produces exactly
/// `[Color0, TexCoord(1)]` and its fragment's prefetch names source 0. Read as a semantic index
/// that is a texcoord the vertex does not produce, the link is refused, and the draw - one of
/// six a scene on the user's device - renders NOTHING. Read as an ordinal it is that vertex's
/// FIRST texcoord, which is `TexCoord(1)`, and the pair links.
///
/// So this asks the closure question over the pairs a run actually draws: does the ordinal
/// reading agree with the semantic one everywhere the semantic one WORKS? If it does, it is
/// compatible with all the evidence the semantic reading rests on and additionally explains the
/// pair that reading cannot - which is what a better reading looks like. If it disagrees
/// anywhere, it is refuted and the failure needs a different answer.
#[test]
#[ignore = "needs a captured corpus AND a run's real pair list"]
fn a_prefetch_source_texcoord_read_as_an_ordinal_agrees_wherever_the_semantic_reading_works() {
    let Some(dir) = corpus_dir() else {
        println!("set VITASLOP_GXP_CORPUS");
        return;
    };
    let Some(list) = std::env::var_os("VITASLOP_GXP_REAL_PAIRS") else {
        println!("set VITASLOP_GXP_REAL_PAIRS");
        return;
    };
    let text = std::fs::read_to_string(PathBuf::from(list)).expect("read real-pair list");
    let mut wanted: Vec<(u64, u64)> = text
        .lines()
        .filter_map(|l| {
            let (v, f) = l.split_once(", fprog hash ")?;
            let v = v.rsplit("vprog hash ").next()?;
            Some((
                u64::from_str_radix(v.trim(), 16).ok()?,
                u64::from_str_radix(f.split_whitespace().next()?, 16).ok()?,
            ))
        })
        .collect();
    wanted.sort_unstable();
    wanted.dedup();

    let mut by_hash: BTreeMap<u64, (String, Program)> = BTreeMap::new();
    for (name, bytes) in blobs(&dir) {
        if let Ok(p) = Program::parse(&bytes) {
            by_hash.insert(p.hash, (name, p));
        }
    }

    let (mut agree, mut both_miss, mut ordinal_only, mut semantic_only, mut disagree) =
        (0usize, 0usize, 0usize, 0usize, 0usize);
    let mut rows: Vec<String> = Vec::new();
    for (vh, fh) in &wanted {
        let (Some((vn, vp)), Some((fname, fp))) = (by_hash.get(vh), by_hash.get(fh)) else {
            continue;
        };
        // The vertex's texcoords, in DECLARATION order - which is what an ordinal indexes.
        let vtex: Vec<u8> = vp
            .output_varyings
            .iter()
            .filter_map(|o| match o.usage {
                vitaslop_gxp_shader::container::VaryingUsage::TexCoord(k) => Some(k),
                _ => None,
            })
            .collect();
        for it in &fp.interpolants {
            let Some(pf) = it.prefetch else { continue };
            let s = pf.source_texcoord;
            let semantic = vtex.contains(&s);
            let ordinal = vtex.get(s as usize).copied();
            match (semantic, ordinal) {
                // The two readings name the SAME varying - the dense-from-zero case, which is
                // every program the original evidence was taken from.
                (true, Some(o)) if o == s => agree += 1,
                (true, Some(o)) => {
                    disagree += 1;
                    rows.push(format!(
                        "  DISAGREE {vn} -> {fname}: source {s}, vertex texcoords {vtex:?} - semantic says TexCoord({s}), ordinal says TexCoord({o})"
                    ));
                }
                (true, None) => semantic_only += 1,
                (false, Some(o)) => {
                    ordinal_only += 1;
                    rows.push(format!(
                        "  ORDINAL ONLY {vn} -> {fname}: source {s}, vertex texcoords {vtex:?} - semantic finds nothing, ordinal says TexCoord({o})"
                    ));
                }
                (false, None) => both_miss += 1,
            }
        }
    }
    println!("prefetch coordinate readings over {} real pairs:", wanted.len());
    println!("  {agree} agree (the vertex numbers its texcoords densely from 0)");
    println!("  {disagree} DISAGREE - a reading that differs here is refuted by the other");
    println!("  {ordinal_only} the ORDINAL resolves and the semantic index does not");
    println!("  {semantic_only} the semantic index resolves and the ordinal does not");
    println!("  {both_miss} neither resolves");
    for r in rows.iter().take(20) {
        println!("{r}");
    }
}

/// >>> WHY THE PAIRS A RUN ACTUALLY DRAWS FAIL TO LINK, AND WHAT THEIR TWO PROGRAMS DECLARE.
///
/// `rank_link_failures_over_all_pairings` ranks the CROSS PRODUCT, which is mostly pairings the
/// title never makes. This takes the run's own list (`VITASLOP_GXP_REAL_PAIRS`, the
/// `vprog hash <h>, fprog hash <h>` lines a run prints with `VITASLOP_GXP_PAIRS=1`) and links
/// only those - so every row is a draw the title issues and the engine drops.
///
/// It exists because a device capture said every scene of a baseball title's gameplay reports
/// `N draws, N carry a shader payload, N-6 recompiled+prepared`, with one reason:
/// `fragment reads a TexCoord(0) varying that the vertex program does not produce`. A dropped
/// draw renders NOTHING, which is what a black loading card looks like.
///
/// The two readings of that failure need opposite fixes and the message alone picks neither:
///   * the vertex GENUINELY produces no such varying, and the hardware feeds the fragment
///     whatever the PA allocation held - in which case refusing costs the draw for nothing and
///     the surplus-register default ([`Interface::defaults`]) is already the established
///     answer for exactly this shape one register at a time;
///   * or `parse_vertex_output_varyings` MISSED an output the block does declare, and feeding a
///     default would paper over a routing bug with a confident wrong picture.
/// So this prints what each side declares, which is the evidence that separates them.
#[test]
#[ignore = "needs a captured corpus AND a run's real pair list"]
fn rank_link_failures_over_the_pairs_the_title_actually_draws() {
    let Some(dir) = corpus_dir() else {
        println!("set VITASLOP_GXP_CORPUS");
        return;
    };
    let Some(list) = std::env::var_os("VITASLOP_GXP_REAL_PAIRS") else {
        println!("set VITASLOP_GXP_REAL_PAIRS to a file of `vprog hash <h>, fprog hash <h>` lines");
        return;
    };
    let text = std::fs::read_to_string(PathBuf::from(list)).expect("read real-pair list");
    let mut wanted: Vec<(u64, u64)> = text
        .lines()
        .filter_map(|l| {
            let (v, f) = l.split_once(", fprog hash ")?;
            let v = v.rsplit("vprog hash ").next()?;
            Some((
                u64::from_str_radix(v.trim(), 16).ok()?,
                u64::from_str_radix(f.split_whitespace().next()?, 16).ok()?,
            ))
        })
        .collect();
    wanted.sort_unstable();
    wanted.dedup();
    assert!(!wanted.is_empty(), "no pairs parsed - the list format changed");

    let mut by_hash: BTreeMap<u64, (String, Vec<u8>, Program)> = BTreeMap::new();
    for (name, bytes) in blobs(&dir) {
        if let Ok(p) = Program::parse(&bytes) {
            by_hash.insert(p.hash, (name, bytes, p));
        }
    }
    println!("{} distinct real pairs, {} blobs indexed by hash", wanted.len(), by_hash.len());

    let mut by_reason: BTreeMap<String, Vec<(u64, u64)>> = BTreeMap::new();
    let (mut linked, mut missing) = (0usize, 0usize);
    for (vh, fh) in &wanted {
        let (Some((_, vb, _)), Some((_, fb, _))) = (by_hash.get(vh), by_hash.get(fh)) else {
            missing += 1;
            continue;
        };
        match link_programs(vb, fb) {
            Ok(_) => linked += 1,
            Err(e) => by_reason.entry(format!("{e}")).or_default().push((*vh, *fh)),
        }
    }
    println!("{linked} link, {} fail, {missing} have a blob the corpus does not hold", wanted.len() - linked - missing);
    let mut ranked: Vec<_> = by_reason.into_iter().collect();
    ranked.sort_by_key(|r| std::cmp::Reverse(r.1.len()));
    for (reason, pairs) in &ranked {
        println!("\n  {} REAL pairs - {reason}", pairs.len());
        // Every distinct pair, with both sides' declarations: the whole point is to separate
        // "the vertex really has no such output" from "the block parse missed one".
        for (vh, fh) in pairs.iter().take(6) {
            let (vn, _, vp) = &by_hash[vh];
            let (fname, _, fp) = &by_hash[fh];
            println!(
                "      {vn} -> {fname}\n        vertex outputs: {:?}\n        fragment reads : {:?}",
                vp.output_varyings
                    .iter()
                    .map(|o| (o.usage, o.base_lane, o.components))
                    .collect::<Vec<_>>(),
                fp.interpolants
                    .iter()
                    .map(|it| (it.usage, it.pa_base, it.register_count, it.half))
                    .collect::<Vec<_>>(),
            );
        }
    }
}

/// Tabulate every VERTEX program's varyings-block output words against the layout they are
/// supposed to describe, so the RESERVED region between the clip position and the texcoords can
/// be settled from the corpus rather than guessed.
///
/// `parse_vertex_output_varyings` derives that region by ARITHMETIC - total lanes minus the
/// texcoord widths minus the four position lanes - and then names it by its width alone (2 lanes
/// = FOG, 4 = COLOR0). That is a one-item inference, and one title's front-end vertex program
/// leaves EIGHT reserved lanes, which the arithmetic cannot name: its whole 2D primitive family
/// then declares no COLOR0 output and every fragment that reads one falls back. This prints the
/// two words next to the derived region so the bits that name it can be found.
#[test]
#[ignore = "needs a captured corpus (game bytes); set VITASLOP_GXP_CORPUS"]
fn tabulate_vertex_varying_output_words() {
    let Some(dir) = corpus_dir() else {
        eprintln!("VITASLOP_GXP_CORPUS not set - nothing to analyse");
        return;
    };
    let mut by_reserved: BTreeMap<u32, usize> = BTreeMap::new();
    for (name, bytes) in blobs(&dir) {
        let Ok(p) = Program::parse(&bytes) else { continue };
        if p.kind != ProgramKind::Vertex {
            continue;
        }
        let Some((vo1, vo2)) = vitaslop_gxp_shader::container::raw_vertex_varying_words(&bytes)
        else {
            continue;
        };
        let total = vo1 >> 24;
        let widths: Vec<(u32, u32)> = (0..=9u32)
            .filter_map(|k| {
                let v = (vo2 >> (k * 3)) & 0x7;
                (v != 0).then(|| (k, (v & 1) * 2 + ((v >> 1) & 1) + ((v >> 2) & 1)))
            })
            .collect();
        let tex: u32 = widths.iter().map(|&(_, n)| n).sum();
        let reserved = total.saturating_sub(tex).saturating_sub(4);
        *by_reserved.entry(reserved).or_default() += 1;
        let blk = vitaslop_gxp_shader::container::raw_varying_block_words(&bytes, 6)
            .unwrap_or_default()
            .iter()
            .map(|v| format!("{v:#010x}"))
            .collect::<Vec<_>>()
            .join(" ");
        let usages: Vec<String> =
            p.output_varyings.iter().map(|v| format!("{:?}@{}", v.usage, v.base_lane)).collect();
        println!(
            "{name}: blk[{blk}] total={total} tex={tex} RESERVED={reserved} decoded=[{}]",
            usages.join(" ")
        );
    }
    println!("\n-- reserved-region widths, by how many programs have them --");
    for (r, n) in &by_reserved {
        println!("  reserved={r:<3} {n} programs");
    }
}

/// >>> Does the size we hand the GUEST cover every uniform the program declares?
///
/// `sceGxmProgramGetDefaultUniformBufferSize` answers `default_uniform_regs * 4`, straight out
/// of the container header (+0x64). A title uses that answer as the LENGTH of the `memcpy` that
/// fills the buffer `sceGxmReserveFragmentDefaultUniformBuffer` just handed it - so a uniform
/// whose registers lie past that length is NEVER WRITTEN BY THE GUEST, and the shader reads
/// whatever the recycled reservation ring happened to hold.
///
/// That failure does not look like missing data. The ring holds the PREVIOUS draw's uniforms,
/// which drift smoothly frame over frame, so the stale lane reads as a plausible animated value
/// - and it differs between engines, because the two run different draw orders. This is the
/// exact shape of the `screenTintColour` white-out.
///
/// The extent has to be measured in REGISTERS, not components: an F16 packs two components per
/// 32-bit register, so an `F16[3]` at register 4 ends at register 5, not register 7.
#[test]
#[ignore = "needs a captured corpus (game bytes); set VITASLOP_GXP_CORPUS"]
fn every_declared_uniform_fits_the_size_we_report_to_the_guest() {
    let Some(dir) = corpus_dir() else { return };
    let mut over = 0usize;
    let mut total = 0usize;
    let mut elsewhere = std::collections::BTreeMap::<u8, usize>::new();
    for (name, bytes) in blobs(&dir) {
        let Ok(p) = Program::parse(&bytes) else { continue };
        total += 1;
        for prm in &p.parameters {
            if prm.category != vitaslop_gxp_shader::container::ParamCategory::Uniform {
                continue;
            }
            // >>> A UNIFORM IS NOT NECESSARILY IN THE DEFAULT BUFFER, and this test used to
            // assume it was. `Parameter::container_index` says which block it lives in, and its
            // `resource_index` is an offset within THAT block - so measuring a buffer-3
            // parameter against the DEFAULT buffer's declared size compares two different
            // address spaces. It made the report unreadable: one program's `g_BoneMatrix` is
            // `F32[2160]`, which "overruns" a 28-register default buffer by 2,132 registers and
            // is not in it at all. Counted by container instead, so the ones that ARE in the
            // default buffer stand out.
            if prm.container_index != DEFAULT_UNIFORM_CONTAINER {
                *elsewhere.entry(prm.container_index).or_default() += 1;
                continue;
            }
            let Some(cb) = prm.ptype.component_bytes() else { continue };
            let components = (prm.component_count as u32).max(1) * prm.array_size.max(1);
            // Registers this parameter spans, from its start register, rounding a partly
            // filled last register up: that register still has to be copied for the
            // components in it to arrive.
            let regs = (components * cb).div_ceil(4);
            let end = (prm.resource_index.max(0) as u32) + regs;
            if end > p.default_uniform_regs {
                over += 1;
                println!(
                    "{name}: {} {:?}[{}] at reg {} spans {} regs -> needs {} but header declares \
                     {} (guest memcpys {} bytes; {} registers NEVER arrive)",
                    prm.name,
                    prm.ptype,
                    components,
                    prm.resource_index,
                    regs,
                    end,
                    p.default_uniform_regs,
                    p.default_uniform_regs * 4,
                    end - p.default_uniform_regs,
                );
            }
        }
    }
    println!("\n-- {over} declared uniforms lie past the reported size, over {total} programs --");
    println!(
        "-- and {} uniforms are NOT in the default buffer at all, by container: {elsewhere:?} \
         (their resource_index is an offset in their own block and says nothing about this) --",
        elsewhere.values().sum::<usize>(),
    );
}

/// The container index the DEFAULT uniform buffer occupies - see [`Container::index`], where
/// 0..13 are the ordinary uniform buffers and 14 is the default one.
const DEFAULT_UNIFORM_CONTAINER: u8 = 14;

/// >>> How does a `SMP`'s sampler FIELD address the texture-control table? Ask the whole corpus.
///
/// `decode_shader` resolves it as `sa_register = 2 * field`. That rule is only ever exercised
/// where it cannot be told apart from `field + default_uniform_regs` or from `field + 2`,
/// because every program that uses it happens to sample its FIRST declared texture - and one
/// blob (`frag_866a1840`) breaks it. Three data points that agree only because they are all the
/// same case are one data point.
///
/// This prints, for every SMP in every blob, the raw field beside what each candidate rule would
/// resolve to and what the container's own texture-control table actually says. A rule that
/// reproduces the table on EVERY row is established; one that does not is dead. That is a
/// decision the corpus can make offline, where a live run can only ever show one program.
#[test]
#[ignore = "needs a captured corpus (game bytes); set VITASLOP_GXP_CORPUS"]
fn how_a_smp_sampler_field_addresses_the_texture_control_table() {
    use vitaslop_gxp_shader::usse::decode;
    use vitaslop_gxp_shader::ir::Op;
    let Some(dir) = corpus_dir() else { return };
    // Candidate rules, each `field -> sa_register`.
    let rules: [(&str, fn(u32, u32) -> u32); 4] = [
        ("2*field", |f, _| 2 * f),
        ("field", |f, _| f),
        ("field+dubuf", |f, d| f + d),
        ("2*field+dubuf", |f, d| 2 * f + d),
    ];
    let mut hits = [0usize; 4];
    let mut rows = 0usize;
    for (name, bytes) in blobs(&dir) {
        let Ok(p) = Program::parse(&bytes) else { continue };
        for (i, &w) in p.code.iter().enumerate() {
            let Op::Tex { unit: field, .. } = decode(w).op else { continue };
            rows += 1;
            let field = field as u32;
            let resolved: Vec<String> = rules
                .iter()
                .enumerate()
                .map(|(k, (label, f))| {
                    let sa = f(field, p.default_uniform_regs);
                    let ok = p.sampler_unit_at(sa).is_some();
                    if ok {
                        hits[k] += 1;
                    }
                    format!("{label}->sa{sa}{}", if ok { "*" } else { "" })
                })
                .collect();
            println!(
                "{name} #{i}: field={field} dubuf={} texctl={:?}  {}",
                p.default_uniform_regs,
                p.texture_control,
                resolved.join("  "),
            );
        }
    }
    println!("\n-- {rows} SMP instructions; how often each rule lands on a DECLARED texture --");
    for (k, (label, _)) in rules.iter().enumerate() {
        println!("  {label:<16} {}/{rows}", hits[k]);
    }
}

/// >>> THE VERTEX PROGRAM'S OWN CODE AS EVIDENCE FOR THE VARYING ORDER.
///
/// # The problem this attacks
/// A vertex program's varyings block states WHICH varyings it outputs and how wide each is,
/// but not their ORDER. Two candidate readings of the containers have each REFUTED the
/// other (see `VaryingOrder` and the two tabulate tests above), and a permutation search
/// over the linker's own consistency checks does not discriminate either - on one title
/// 126 pairings have all six orders surviving, because the varyings all have the same
/// width. So the order needs evidence from OUTSIDE the containers.
///
/// # The evidence this looks for
/// The vertex program's USSE code. Whatever a varying is called, the code has to COMPUTE
/// it, and what it computes it FROM is a fact the container does not carry. Concretely: run
/// the program, perturb ONE vertex attribute, and see which OUTPUT LANES change. A lane
/// that moves when TEXCOORD0 moves is carrying something derived from TEXCOORD0.
///
/// A sensitivity analysis rather than a static read of the operands, because it sees
/// THROUGH arbitrary arithmetic: a UV that arrives at an output lane via a matrix multiply,
/// a scale-and-bias, or a chain of temporaries is still a lane that moves when the UV
/// moves. A dataflow walk would have to model every op to say the same thing.
///
/// This test only REPORTS. It settles nothing on its own - it says whether the instrument
/// can be built at all, which is the question that has to be answered before anything is
/// built on it.
#[test]
#[ignore = "needs a captured corpus (game bytes); set VITASLOP_GXP_CORPUS"]
fn tabulate_vertex_output_sensitivity_to_attributes() {
    use vitaslop_gxp_shader::container::{ParamCategory, VaryingOrder};
    use vitaslop_gxp_shader::interp::{run, RegFile};

    let Some(dir) = corpus_dir() else {
        println!("set VITASLOP_GXP_CORPUS to a directory of .gxp blobs");
        return;
    };

    // Enough lanes for any bank a captured program addresses; the interpreter indexes
    // `bank[base + lane]` directly and a short bank would fault rather than answer.
    const LANES: usize = 512;
    /// How far to move an attribute lane. Large and irrational-ish so a perturbation cannot
    /// coincidentally land back on the baseline through a wrap, a saturate or a fract.
    const KICK: f32 = 0.618_034;

    let mut interpretable = 0usize;
    let mut refused: BTreeMap<String, usize> = BTreeMap::new();
    let mut vertex_programs = 0usize;

    for (name, bytes) in blobs(&dir) {
        let Ok(program) = Program::parse(&bytes) else { continue };
        if program.kind != ProgramKind::Vertex {
            continue;
        }
        vertex_programs += 1;
        let Ok(rc) = recompile_vertex(&bytes) else {
            *refused.entry("recompile_vertex refused".into()).or_default() += 1;
            continue;
        };
        let shader = &rc.shader;

        // The attributes, each at its own PA base lane (`resource_index`).
        let attrs: Vec<_> = program
            .parameters
            .iter()
            .filter(|p| p.category == ParamCategory::Attribute)
            .collect();

        // Baseline: every PA lane distinct, so a lane that does not move is not merely
        // hidden by two inputs happening to be equal.
        let seed = |f: &mut RegFile| {
            for (i, v) in f.pa.iter_mut().enumerate() {
                *v = 0.125 + i as f32 * 0.0314159;
            }
            for (i, v) in f.sa.iter_mut().enumerate() {
                *v = 0.5 + i as f32 * 0.0271828;
            }
        };

        let mut base = RegFile::with_lanes(LANES);
        seed(&mut base);
        if let Err(e) = run(shader, &mut base) {
            *refused.entry(format!("interpreter: {e:?}")).or_default() += 1;
            continue;
        }
        interpretable += 1;

        // Which output lanes this program writes at all - the rest are not varyings.
        let mut written: Vec<usize> = Vec::new();
        for instr in &shader.instrs {
            let Some(d) = instr.dest.as_ref() else { continue };
            if d.bank != vitaslop_gxp_shader::ir::Bank::Output {
                continue;
            }
            for c in 0..4 {
                if instr.write_mask[c] {
                    written.push(d.index as usize + c);
                }
            }
        }
        written.sort_unstable();
        written.dedup();

        // One perturbed run per attribute; record which written output lanes moved.
        let mut moves: Vec<(String, Vec<usize>)> = Vec::new();
        for a in &attrs {
            let mut f = RegFile::with_lanes(LANES);
            seed(&mut f);
            let lo = a.resource_index.max(0) as usize;
            for c in 0..(a.component_count as usize).clamp(1, 4) {
                if lo + c < f.pa.len() {
                    f.pa[lo + c] += KICK;
                }
            }
            if run(shader, &mut f).is_err() {
                continue;
            }
            let moved: Vec<usize> = written
                .iter()
                .copied()
                .filter(|&l| l < f.o.len() && (f.o[l] - base.o[l]).abs() > 1e-6)
                .collect();
            moves.push((format!("{}[sem {}.{}]", a.name, a.semantic, a.semantic_index), moved));
        }

        let order = program.output_order;
        let tag = match order {
            VaryingOrder::Known => "KNOWN",
            VaryingOrder::Assumed => "assumed",
            VaryingOrder::Ambiguous => ">>> AMBIGUOUS",
        };
        println!("\n{name} {tag}  declared varyings:");
        for v in &program.output_varyings {
            println!(
                "    {:?} lanes {}..{}",
                v.usage,
                v.base_lane,
                v.base_lane + v.components
            );
        }
        println!("  written output lanes: {written:?}");
        for (who, moved) in &moves {
            println!("  moving {who:<34} -> lanes {moved:?}");
        }
        // The lanes NO attribute reaches: computed from uniforms or literals alone. A COLOR1
        // with no attribute evidence is expected to be exactly this, which is why it is
        // worth naming separately rather than leaving as an empty row.
        let reached: std::collections::BTreeSet<usize> =
            moves.iter().flat_map(|(_, m)| m.iter().copied()).collect();
        let unreached: Vec<usize> =
            written.iter().copied().filter(|l| !reached.contains(l)).collect();
        println!("  lanes NO attribute reaches: {unreached:?}");

        // >>> AND WHICH UNIFORM FEEDS THEM. A uniform-fed varying is not evidence-free: the
        // container NAMES its uniforms, and a lane group that moves when `diffuseColour`
        // moves is a colour whatever the varyings block calls it. This is the half of the
        // evidence that reaches the varyings an attribute never touches - which is exactly
        // the case (a declared COLOR1 with no attribute) that stops one of the titles.
        if !unreached.is_empty() {
            for u in program
                .parameters
                .iter()
                .filter(|p| p.category == ParamCategory::Uniform)
            {
                let mut f = RegFile::with_lanes(LANES);
                seed(&mut f);
                let lo = u.resource_index.max(0) as usize;
                let n = (u.component_count as usize).clamp(1, 4) * u.array_size.max(1) as usize;
                for c in 0..n {
                    if lo + c < f.sa.len() {
                        f.sa[lo + c] += KICK;
                    }
                }
                if run(shader, &mut f).is_err() {
                    continue;
                }
                let moved: Vec<usize> = unreached
                    .iter()
                    .copied()
                    .filter(|&l| l < f.o.len() && (f.o[l] - base.o[l]).abs() > 1e-6)
                    .collect();
                if !moved.is_empty() {
                    println!("    uniform {:<28} sa{lo}+{n} -> lanes {moved:?}", u.name);
                }
            }
        }
    }

    println!(
        "\n-- {vertex_programs} vertex programs; {interpretable} could be INTERPRETED --"
    );
    for (why, n) in &refused {
        println!("  {n:>4}  {why}");
    }
}

/// >>> AND THE FRAGMENT SIDE OF THE SAME QUESTION: WHICH VARYING IS A TEXTURE COORDINATE?
///
/// The vertex sensitivity test above says which output lane group each vertex ATTRIBUTE
/// reaches. That fixes the group boundaries but cannot label two groups of equal width -
/// on one title's ambiguous programs, Color0 and Color1 are four lanes each and nothing
/// on the vertex side tells them apart.
///
/// The fragment can. A varying used as a SAMPLER COORDINATE is a UV whatever the container
/// calls it, and the interpreter already has the hook to see it: [`interp::TexFetch`] is
/// handed the coordinate of every sample. So perturb one interpolant's PA registers, re-run,
/// and watch which sample COORDINATES move.
///
/// Together the two tests state a cross-stage constraint the containers do not: the vertex
/// lane group carrying the geometry's UV attribute has to arrive at the fragment interpolant
/// the fragment samples with. That is evidence about the ORDER, from the code, which is what
/// this whole question has been missing.
#[test]
#[ignore = "needs a captured corpus (game bytes); set VITASLOP_GXP_CORPUS"]
fn tabulate_fragment_varyings_used_as_texture_coordinates() {
    use std::cell::RefCell;
    use vitaslop_gxp_shader::interp::{run_watching_for_nan_with_textures, RegFile};

    let Some(dir) = corpus_dir() else {
        println!("set VITASLOP_GXP_CORPUS to a directory of .gxp blobs");
        return;
    };

    const LANES: usize = 512;
    const KICK: f32 = 0.618_034;

    let mut interpretable = 0usize;
    let mut fragment_programs = 0usize;
    let mut refused: BTreeMap<String, usize> = BTreeMap::new();

    for (name, bytes) in blobs(&dir) {
        let Ok(program) = Program::parse(&bytes) else { continue };
        if program.kind != ProgramKind::Fragment {
            continue;
        }
        fragment_programs += 1;
        let Ok(rc) = recompile_fragment(&bytes) else {
            *refused.entry("recompile_fragment refused".into()).or_default() += 1;
            continue;
        };
        let shader = &rc.shader;

        let seed = |f: &mut RegFile| {
            for (i, v) in f.pa.iter_mut().enumerate() {
                *v = 0.125 + i as f32 * 0.0314159;
            }
            for (i, v) in f.sa.iter_mut().enumerate() {
                *v = 0.5 + i as f32 * 0.0271828;
            }
        };

        // Record every coordinate the shader samples with, and return a value DERIVED from
        // it - so a dependency does not stop at the sample, and a varying that only reaches
        // the output through a texture fetch is still visible downstream.
        fn sample(
            log: &RefCell<Vec<[f32; 4]>>,
        ) -> impl Fn(u8, [f32; 4], vitaslop_gxp_shader::interp::TexLodArg) -> Option<[f32; 4]> + '_ {
            move |_unit: u8, c: [f32; 4], _lod| {
                log.borrow_mut().push(c);
                Some([c[0] * 0.5 + 0.25, c[1] * 0.25 + 0.5, c[2] * 0.125, 0.75])
            }
        }

        let base_log = RefCell::new(Vec::new());
        let mut base = RegFile::with_lanes(LANES);
        seed(&mut base);
        let f = sample(&base_log);
        if let Err(e) = run_watching_for_nan_with_textures(shader, &mut base, &f) {
            *refused.entry(format!("interpreter: {e:?}")).or_default() += 1;
            continue;
        }
        interpretable += 1;
        let base_coords = base_log.borrow().clone();

        println!("\n{name}  interpolants:");
        let mut rows: Vec<String> = Vec::new();
        for it in &program.interpolants {
            // The interpolant's PA registers. `half` packs two components per register, but
            // the interpreter's register file has no packing anywhere, so perturbing the
            // register lanes is what moves the value either way.
            let lo = it.pa_base as usize;
            let hi = lo + it.register_count.max(1) as usize;

            let log = RefCell::new(Vec::new());
            let mut f2 = RegFile::with_lanes(LANES);
            seed(&mut f2);
            for l in lo..hi.min(f2.pa.len()) {
                f2.pa[l] += KICK;
            }
            let fetch = sample(&log);
            if run_watching_for_nan_with_textures(shader, &mut f2, &fetch).is_err() {
                rows.push(format!("    {:?} pa{lo}..{hi}: (re-run refused)", it.usage));
                continue;
            }
            let coords = log.borrow().clone();
            // Which SAMPLES moved. Same shader, same control flow (the seed only shifts
            // values), so the two logs line up index for index; a length change is itself
            // worth reporting rather than silently zipping to the shorter.
            let moved: Vec<usize> = if coords.len() != base_coords.len() {
                Vec::new()
            } else {
                base_coords
                    .iter()
                    .zip(coords.iter())
                    .enumerate()
                    .filter(|(_, (a, b))| {
                        a.iter().zip(b.iter()).any(|(x, y)| (x - y).abs() > 1e-6)
                    })
                    .map(|(i, _)| i)
                    .collect()
            };
            // The fragment's colour is whatever the program left in a FIXED register at the
            // end - `o0` when the stream writes the OUTPUT bank ("native colour"), else
            // `pa0`. Comparing the wrong one reports "nothing moved" for every non-native
            // shader, which is a silent zero, not a finding.
            //
            // When the colour is in `pa0` the comparison also has to skip the lanes this
            // perturbation WROTE: an interpolant at pa0 would otherwise register as
            // affecting the colour purely because it was the thing kicked.
            let native = shader.instrs.iter().any(|i| {
                i.dest.as_ref().is_some_and(|d| d.bank == vitaslop_gxp_shader::ir::Bank::Output)
            });
            let (a_bank, b_bank) =
                if native { (&base.o, &f2.o) } else { (&base.pa, &f2.pa) };
            let out_moved = (0..4usize)
                .filter(|&i| native || !(lo..hi).contains(&i))
                .filter(|&i| (a_bank[i] - b_bank[i]).abs() > 1e-6)
                .collect::<Vec<_>>();
            rows.push(format!(
                "    {:?} pa{lo}..{hi} {}-> samples {moved:?}, colour ({}) channels {out_moved:?}{}",
                it.usage,
                if it.half { "(f16) " } else { "" },
                if native { "o0" } else { "pa0" },
                if moved.is_empty() { "   [NOT a texture coordinate]" } else { "   <<< UV" },
            ));
        }
        for r in rows {
            println!("{r}");
        }
        println!("  samples taken: {}", base_coords.len());
    }

    println!("\n-- {fragment_programs} fragment programs; {interpretable} INTERPRETED --");
    for (why, n) in &refused {
        println!("  {n:>4}  {why}");
    }
}

/// >>> THE FALSIFIER FOR THE TWO INSTRUMENTS ABOVE: on every vertex program whose order the
/// >>> CONTAINER establishes independently, the code has to say the same thing.
///
/// A sensitivity analysis is only evidence if it agrees with the cases that are already
/// settled. `VaryingOrder::Known` means the program's own ATTRIBUTES named every declared
/// varying, so the lane layout is established without any of this - which makes those
/// programs the known-good arm. For each one, perturbing the attribute that carries a
/// varying's semantic must move exactly that varying's lanes and no other varying's.
///
/// If this fails, the instrument is wrong and nothing built on it may be believed. If it
/// passes over a whole corpus, the same instrument's answer on an AMBIGUOUS program is
/// evidence of the same kind.
#[test]
#[ignore = "needs a captured corpus (game bytes); set VITASLOP_GXP_CORPUS"]
fn attribute_sensitivity_agrees_with_the_container_on_known_programs() {
    use vitaslop_gxp_shader::container::{
        ParamCategory, VaryingOrder, VaryingUsage, SEMANTIC_COLOR,
    };
    use vitaslop_gxp_shader::interp::{run, RegFile};

    let Some(dir) = corpus_dir() else {
        println!("set VITASLOP_GXP_CORPUS to a directory of .gxp blobs");
        return;
    };

    const LANES: usize = 512;
    const KICK: f32 = 0.618_034;
    /// `Parameter::semantic` for TEXCOORD - see `container::SEMANTIC_*`.
    const SEMANTIC_TEXCOORD: u8 = 14;

    let mut checked = 0usize;
    let mut agreed = 0usize;
    let mut disagreed: Vec<String> = Vec::new();
    let mut partial: Vec<String> = Vec::new();

    for (name, bytes) in blobs(&dir) {
        let Ok(program) = Program::parse(&bytes) else { continue };
        if program.kind != ProgramKind::Vertex || program.output_order != VaryingOrder::Known {
            continue;
        }
        // Fewer than two varyings is `Known` trivially and orders nothing.
        if program.output_varyings.len() < 2 {
            continue;
        }
        let Ok(rc) = recompile_vertex(&bytes) else { continue };

        let seed = |f: &mut RegFile| {
            for (i, v) in f.pa.iter_mut().enumerate() {
                *v = 0.125 + i as f32 * 0.0314159;
            }
            for (i, v) in f.sa.iter_mut().enumerate() {
                *v = 0.5 + i as f32 * 0.0271828;
            }
        };
        let mut base = RegFile::with_lanes(LANES);
        seed(&mut base);
        if run(&rc.shader, &mut base).is_err() {
            continue;
        }

        for v in &program.output_varyings {
            // The attribute carrying this varying's semantic, if there is one.
            let want = match v.usage {
                VaryingUsage::Color0 => Some((SEMANTIC_COLOR, 0u8)),
                VaryingUsage::Color1 => Some((SEMANTIC_COLOR, 1)),
                VaryingUsage::TexCoord(k) => Some((SEMANTIC_TEXCOORD, k)),
                _ => None,
            };
            let Some((sem, idx)) = want else { continue };
            let Some(a) = program.parameters.iter().find(|p| {
                p.category == ParamCategory::Attribute && p.semantic == sem && p.semantic_index == idx
            }) else {
                continue;
            };

            let mut f = RegFile::with_lanes(LANES);
            seed(&mut f);
            let lo = a.resource_index.max(0) as usize;
            for c in 0..(a.component_count as usize).clamp(1, 4) {
                if lo + c < f.pa.len() {
                    f.pa[lo + c] += KICK;
                }
            }
            if run(&rc.shader, &mut f).is_err() {
                continue;
            }
            let moved: Vec<usize> = (0..base.o.len())
                .filter(|&l| (f.o[l] - base.o[l]).abs() > 1e-6)
                .collect();
            let lanes: Vec<usize> =
                (v.base_lane as usize..(v.base_lane + v.components) as usize).collect();

            checked += 1;
            // The claim is that the varying STARTS where the container says: moving the
            // attribute that names it must move its BASE lane.
            //
            // Not "every lane of it", which was tried and is wrong: a shader legitimately
            // writes constants into some channels of a colour, and one title's
            // `vert_82c63d50` does exactly that - `in_colour` moves lanes 8 and 9 of a
            // Color0 declared at 8..12, with 10 and 11 written from literals. That is a
            // varying correctly placed at lane 8, not a disagreement about order.
            //
            // Nor is exclusivity asserted: an attribute legitimately reaches other outputs
            // too (a position built from a normal, a fog term derived from a UV).
            let base_moved = moved.contains(&(v.base_lane as usize));
            if base_moved {
                agreed += 1;
                if !lanes.iter().all(|l| moved.contains(l)) {
                    partial.push(format!(
                        "{name}: {:?} at lanes {lanes:?} - {} moves only {:?} (the rest are written from constants)",
                        v.usage,
                        a.name,
                        lanes.iter().filter(|l| moved.contains(l)).collect::<Vec<_>>(),
                    ));
                }
            } else {
                disagreed.push(format!(
                    "{name}: {:?} declared at lanes {lanes:?}, but moving {} moved {moved:?}",
                    v.usage, a.name
                ));
            }
        }
    }

    println!("-- container-established varyings checked against the code: {agreed}/{checked} agree --");
    for d in &disagreed {
        println!("  {d}");
    }
    for p in &partial {
        println!("  (partial, not a disagreement) {p}");
    }
    if checked == 0 {
        // A corpus can legitimately contain no multi-varying program whose attributes name
        // every varying. Say so rather than failing: this test proves nothing here, and a
        // green result would be the more misleading of the two outcomes.
        println!("  NOTHING CHECKABLE in this corpus - this run proves nothing either way");
        return;
    }
    assert!(
        disagreed.is_empty(),
        "{} of {checked} container-established varyings disagree with the code; the \
         sensitivity instrument cannot be trusted on the ambiguous ones",
        disagreed.len()
    );
}

/// >>> THE REFUTATION ABOVE WAS MEASURED OVER THE WRONG POPULATION. THIS RE-RUNS IT OVER
/// >>> THE PAIRS A TITLE ACTUALLY DRAWS.
///
/// `fragment_declaration_order_matches_attribute_established_vertex_order` and
/// `vertex_lane_order_agrees_with_the_fragment_declaration_order` both iterate the CROSS
/// PRODUCT of every vertex blob against every fragment blob, and count any combination the
/// linker does not reject. Most of those combinations are pairings the title never makes -
/// a shadow vertex program against a UI fragment - and a disagreement between two programs
/// that are never drawn together says nothing about the hardware.
///
/// The notes drew that conclusion themselves without acting on it: one title's world
/// renders correctly today under the convention, which it could not if 96% of its REAL
/// pairs were mis-ordered, "so its REAL pairs are among the ones that agree, and the
/// cross-product count overstates the problem".
///
/// So this takes the real pairs. `VITASLOP_GXP_REAL_PAIRS` points at a file of
/// `vprog hash <h>, fprog hash <h>` lines - exactly what a run prints with
/// `VITASLOP_GXP_PAIRS=1` - and only those pairings are compared. If they agree, the
/// fragment's declaration order IS a statement about vertex lanes for pairs that exist,
/// and it can supply the order a vertex container leaves unstated.
#[test]
#[ignore = "needs a captured corpus AND a run's real pair list"]
fn the_fragment_declaration_order_agrees_on_pairs_the_title_actually_draws() {
    let Some(dir) = corpus_dir() else {
        println!("set VITASLOP_GXP_CORPUS");
        return;
    };
    let Some(list) = std::env::var_os("VITASLOP_GXP_REAL_PAIRS") else {
        println!("set VITASLOP_GXP_REAL_PAIRS to a file of `vprog hash <h>, fprog hash <h>` lines");
        return;
    };
    let text = std::fs::read_to_string(PathBuf::from(list)).expect("read real-pair list");
    let wanted: Vec<(u64, u64)> = text
        .lines()
        .filter_map(|l| {
            let (v, f) = l.split_once(", fprog hash ")?;
            let v = v.trim().strip_prefix("vprog hash ")?;
            Some((u64::from_str_radix(v.trim(), 16).ok()?, u64::from_str_radix(f.trim(), 16).ok()?))
        })
        .collect();
    assert!(!wanted.is_empty(), "no pairs parsed - the list format changed");

    // Index the corpus by content hash, which is what a run prints and what survives the
    // guest addresses differing between runs.
    let mut by_hash: BTreeMap<u64, (String, Vec<u8>, Program)> = BTreeMap::new();
    for (name, bytes) in blobs(&dir) {
        if let Ok(p) = Program::parse(&bytes) {
            by_hash.insert(p.hash, (name, bytes, p));
        }
    }

    let (mut agree, mut disagree, mut few, mut missing) = (0usize, 0usize, 0usize, 0usize);
    let mut rows: Vec<String> = Vec::new();
    for (vh, fh) in &wanted {
        let (Some((vn, _, vp)), Some((fname, _, fp))) = (by_hash.get(vh), by_hash.get(fh)) else {
            missing += 1;
            continue;
        };
        let vorder: Vec<_> = vp.output_varyings.iter().map(|o| o.usage).collect();
        let forder: Vec<_> = fp.interpolants.iter().map(|it| it.usage).collect();
        let shared: Vec<_> = vorder.iter().filter(|u| forder.contains(u)).copied().collect();
        let fshared: Vec<_> = forder.iter().filter(|u| vorder.contains(u)).copied().collect();
        if shared.len() < 2 {
            few += 1;
            continue;
        }
        if shared == fshared {
            agree += 1;
        } else {
            disagree += 1;
            rows.push(format!(
                "    {vn} + {fname}\n vertex order   {shared:?}\n fragment order {fshared:?}"
            ));
        }
    }
    println!(
        "REAL pairs: {} agree, {} DISAGREE, {} had <2 shared varyings, {} not in this corpus \
         (of {} listed)",
        agree,
        disagree,
        few,
        missing,
        wanted.len()
    );
    for r in &rows {
        println!("{r}");
    }
}

/// Every (vertex, fragment) pairing in the corpus, tallied by whether it LINKS and - when it
/// does not - by the reason. One number per reason, so a change to the linker can be read as
/// "this many pairings moved" instead of by eye.
#[test]
#[ignore = "needs a captured corpus (game bytes); set VITASLOP_GXP_CORPUS"]
fn tally_pair_link_outcomes() {
    let Some(dir) = corpus_dir() else {
        println!("set VITASLOP_GXP_CORPUS");
        return;
    };
    let all = blobs(&dir);
    let verts: Vec<_> = all
        .iter()
        .filter(|(_, b)| {
            Program::parse(b).map(|p| p.kind == ProgramKind::Vertex).unwrap_or(false)
        })
        .collect();
    let frags: Vec<_> = all
        .iter()
        .filter(|(_, b)| {
            Program::parse(b).map(|p| p.kind == ProgramKind::Fragment).unwrap_or(false)
        })
        .collect();

    let mut ok = 0usize;
    let mut by_reason: BTreeMap<String, usize> = BTreeMap::new();
    for (_, vb) in &verts {
        for (_, fb) in &frags {
            match link_programs(vb, fb) {
                Ok(_) => ok += 1,
                Err(e) => {
                    // Collapse the variant's payload so the tally groups by CAUSE.
                    let s = format!("{e}");
                    let head: String = s.chars().take(60).collect();
                    *by_reason.entry(head).or_default() += 1;
                }
            }
        }
    }
    println!(
        "{} vertex x {} fragment = {} pairings: {ok} LINK",
        verts.len(),
        frags.len(),
        verts.len() * frags.len()
    );
    let mut rows: Vec<_> = by_reason.into_iter().collect();
    rows.sort_by_key(|r| std::cmp::Reverse(r.1));
    for (why, n) in rows.iter().take(12) {
        println!("  {n:>6}  {why}");
    }
}

/// Which output lanes a vertex program copies STRAIGHT out of a named attribute, and whether
/// the varying sitting on those lanes carries that attribute's semantic.
///
/// # Why this witness, and why it is not `attribute_order`
/// `container::attribute_order` already reads an order off the attributes, but only when the
/// semantics cover the declared varying set EXACTLY - a passthrough program. Almost no real
/// program is one: it forwards two of its four inputs and computes the rest, so the cover
/// fails and the whole reading is discarded, convention or nothing.
///
/// A forwarding MOVE is evidence even when the cover is not exact. `mov Output[n] <-
/// PrimaryAttr[m]` says lane `n` receives the attribute holding register `m`, and that
/// attribute's semantic says which varying that is. It is a statement about ONE lane, so it
/// survives the other varyings being computed - and one such statement is enough to refuse a
/// layout that puts a TEXCOORD attribute into a COLOR varying.
///
/// This test does not change the linker. It counts how often each candidate order - the
/// canonical convention, and the paired fragment's declaration order - is CONTRADICTED by a
/// program's own forwarding moves, per corpus. That is the measurement the varying-order
/// question has been missing: both candidate readings have a render oracle that likes them and
/// a render oracle that does not, and neither has ever been asked what the vertex code says.
#[test]
#[ignore = "needs a captured corpus (game bytes); set VITASLOP_GXP_CORPUS"]
fn forwarding_moves_that_contradict_a_varying_order() {
    use vitaslop_gxp_shader::container::{
        VaryingUsage, SEMANTIC_COLOR, SEMANTIC_FOGCOORD, SEMANTIC_TEXCOORD,
    };
    use vitaslop_gxp_shader::ir::{Bank, Op};

    let Some(dir) = corpus_dir() else {
        eprintln!("VITASLOP_GXP_CORPUS not set - nothing to analyse");
        return;
    };

    // The varying an attribute's semantic names, or `None` for one that is not a varying usage
    // (position, normals, blend weights - consumed rather than forwarded).
    fn attr_usage(p: &vitaslop_gxp_shader::container::Parameter) -> Option<VaryingUsage> {
        match p.semantic {
            SEMANTIC_FOGCOORD => Some(VaryingUsage::Fog),
            SEMANTIC_COLOR => match p.semantic_index {
                0 => Some(VaryingUsage::Color0),
                1 => Some(VaryingUsage::Color1),
                _ => None,
            },
            SEMANTIC_TEXCOORD => Some(VaryingUsage::TexCoord(p.semantic_index)),
            _ => None,
        }
    }

    let (mut programs, mut with_evidence, mut convention_ok, mut convention_bad) =
        (0usize, 0usize, 0usize, 0usize);
    for (name, bytes) in blobs(&dir) {
        let Ok(p) = Program::parse(&bytes) else { continue };
        if p.kind != ProgramKind::Vertex || p.output_varyings.len() < 2 {
            continue;
        }
        programs += 1;

        // PA register -> the attribute that holds it. An attribute occupies
        // `resource_index .. resource_index + component_count` registers of the PA bank.
        let attr_at = |reg: usize| {
            p.parameters.iter().find(|a| {
                a.category == vitaslop_gxp_shader::container::ParamCategory::Attribute
                    && a.resource_index >= 0
                    && reg >= a.resource_index as usize
                    && reg < a.resource_index as usize + a.component_count as usize
            })
        };

        // The forwarding moves: `Output[n].c <- PrimaryAttr[m]`, one claim per written channel.
        // Only a MOVE counts - anything that computes with the value says nothing about which
        // varying the value IS.
        let shader = vitaslop_gxp_shader::usse::decode_shader(&p);
        let mut claims: Vec<(usize, VaryingUsage, String)> = Vec::new();
        for instr in &shader.instrs {
            if !matches!(instr.op, Op::Mov) {
                continue;
            }
            let (Some(d), Some(s)) = (instr.dest.as_ref(), instr.srcs.first()) else { continue };
            if d.bank != Bank::Output || s.bank != Bank::PrimaryAttr {
                continue;
            }
            for c in 0..4 {
                if !instr.write_mask[c] {
                    continue;
                }
                let src_reg = s.index as usize + s.swizzle[c] as usize;
                let Some(a) = attr_at(src_reg) else { continue };
                let Some(u) = attr_usage(a) else { continue };
                claims.push((d.index as usize + c, u, a.name.clone()));
            }
        }
        if claims.is_empty() {
            continue;
        }
        with_evidence += 1;

        // Judge the CONVENTION (the order the container placed) against them.
        let usage_at = |lane: usize| {
            p.output_varyings
                .iter()
                .find(|v| {
                    lane >= v.base_lane as usize
                        && lane < v.base_lane as usize + v.components as usize
                })
                .map(|v| v.usage)
        };
        let bad: Vec<String> = claims
            .iter()
            .filter_map(|(lane, want, aname)| match usage_at(*lane) {
                // A lane inside no declared run is not a varying claim at all (scratch use of
                // the output bank, which this corpus does contain), so it is not evidence.
                None => None,
                Some(got) if got == *want => None,
                Some(got) => Some(format!("lane {lane} <- {aname} ({want:?}) but the layout says {got:?}")),
            })
            .collect();
        if bad.is_empty() {
            convention_ok += 1;
            continue;
        }
        convention_bad += 1;
        let layout: Vec<String> = p
            .output_varyings
            .iter()
            .map(|v| format!("{:?}@{}..{}", v.usage, v.base_lane, v.base_lane + v.components))
            .collect();
        println!("{name} CONTRADICTED ({:?})  layout {}", p.output_order, layout.join(" "));
        for b in &bad {
            println!("    {b}");
        }
    }
    println!(
        "\nforwarding-move witness: {programs} vertex programs, {with_evidence} forward at least \
         one named attribute, {convention_ok} agree with the placed order, {convention_bad} \
         CONTRADICT it"
    );
}

/// Every instruction of the USSE MEMORY-ACCESS family in the corpus, with the fields the
/// distilled spec establishes - so the field that is NOT established can be looked at across
/// real shipped shaders instead of reasoned about.
///
/// # Why this is the next step and not an emitter
/// The USSE memory-access notes (reference material, held outside the repo) have the whole
/// 64-bit layout of group 0x1d (load) and 0x1e (store), which share one format, and it is
/// explicit that emit must stay blocked: the general case reads ARBITRARY GUEST MEMORY through
/// a register-held byte pointer, which WGSL cannot express without a storage-buffer binding,
/// and **the variant selector that would say which of the three address spaces (absolute /
/// local / thread) applies is not established.** The only unclaimed multi-value fields are
/// `mode` (41:40) and `addr_mode` (43:42). Resolving those comes BEFORE any emitter.
///
/// This is the cheapest evidence available for that: what values those two fields actually
/// take in shipped shaders, and what they co-vary with. A field that is constant across every
/// instruction in every title is not a selector; one that tracks the source BANK is a strong
/// hint about what it selects.
///
/// It asserts nothing. It is an instrument, and it prints a table.
#[test]
#[ignore = "needs a captured corpus (game bytes); set VITASLOP_GXP_CORPUS"]
fn usse_memory_group_field_census() {
    let Some(dir) = corpus_dir() else {
        eprintln!("VITASLOP_GXP_CORPUS not set - nothing to analyse");
        return;
    };
    // The spec's bit table. Named here rather than imported because the decoder does not
    // decode this group's operands at all - that is the whole point of the census.
    let f = |w: u64, hi: u32, lo: u32| ((w >> lo) & ((1u64 << (hi - lo + 1)) - 1)) as u32;
    // `(direction, mode, addr_mode, data_type, src0 bank+ext, mask_count+1)` -> how many, and
    // one example name.
    let mut rows: BTreeMap<(u32, u32, u32, u32, u32, u32), (usize, String)> = BTreeMap::new();
    let mut programs = 0usize;
    // Second table, for the ONE field that still blocks this group on a shipped title:
    // `moe_expand` (bit 53) against the element count, and against whether ANY SMLSI has
    // executed earlier in the same program. "Expansion" can only be the identity while the
    // MOE state is its default `Increment(1)` and there is a single element to expand, so
    // those two columns are what decide whether the blocked case needs a semantics at all.
    let mut moe: BTreeMap<(u32, u32, u32, bool), (usize, String)> = BTreeMap::new();
    for (name, bytes) in blobs(&dir) {
        let Ok(p) = Program::parse(&bytes) else { continue };
        // >>> BOTH STREAMS. This census walked the PRIMARY only, and every memory load of a
        // shipped baseball title's world materials is in the SECONDARY - the driver-run
        // program that fills the SA file from the bound uniform buffers. So the reading it
        // established for `moe_expand` ("set only ever with a single element, and only in
        // programs where no SMLSI has run") was measured on a domain that excluded the
        // instructions the question is actually about.
        let shader = vitaslop_gxp_shader::usse::decode_shader(&p);
        let secondary = vitaslop_gxp_shader::usse::decode_secondary_shader(&p);
        let mut hit = false;
        let mut smlsi_seen = false;
        for instr in secondary.instrs.iter().chain(shader.instrs.iter()) {
            if vitaslop_gxp_shader::usse::decode::is_smlsi(instr.raw) {
                smlsi_seen = true;
            }
            if instr.group != 0x1d && instr.group != 0x1e {
                continue;
            }
            hit = true;
            let w = instr.raw;
            let key = (
                f(w, 60, 59),                      // direction: 01 load, 10 store
                f(w, 41, 40),                      // mode - NOT ESTABLISHED
                f(w, 43, 42),                      // addr_mode - NOT ESTABLISHED
                f(w, 37, 36),                      // data_type
                f(w, 50, 50) * 2 + f(w, 34, 34),   // src0 bank ext:bank
                f(w, 47, 44) + 1,                  // elements
            );
            let e = rows.entry(key).or_insert((0, name.clone()));
            e.0 += 1;
            let mk = (f(w, 60, 59), f(w, 53, 53), f(w, 47, 44) + 1, smlsi_seen);
            let e = moe.entry(mk).or_insert((0, name.clone()));
            e.0 += 1;
        }
        if hit {
            programs += 1;
        }
    }
    println!(
        "\n-- USSE memory group census: {} distinct field combinations over {programs} programs --",
        rows.len()
    );
    println!("  dir mode addr type src0(ext:bank)  elems  count  example");
    for ((dir_, mode, addr_mode, ty, src0, elems), (n, example)) in &rows {
        let dirn = match dir_ {
            1 => "LD",
            2 => "ST",
            _ => "??",
        };
        let bank = match src0 {
            0 => "TEMP",
            1 => "PRIMATTR",
            2 => "OUTPUT",
            _ => "SECATTR",
        };
        println!(
            "  {dirn}  {mode}    {addr_mode}    {ty}    {bank:<9} {elems:<5}  {n:<5}  {example}"
        );
    }
    // >>> EVERY `moe_expand` LOAD'S IMMEDIATE OFFSET, against the program's own declared
    // parameters. The bit is allowed through on the argument that a SINGLE element has no
    // second iteration to step to; if that is right the offset it reads must land where a
    // parameter does, and if it is wrong the offsets will sit systematically beside one.
    {
        let mut off_rows: BTreeMap<(u32, bool), (usize, String)> = BTreeMap::new();
        for (name, bytes) in blobs(&dir) {
            let Ok(p) = Program::parse(&bytes) else { continue };
            // Every register a declared uniform STARTS at, in this program's own terms.
            let starts: std::collections::BTreeSet<i32> =
                p.parameters.iter().map(|q| q.resource_index).collect();
            let sec = vitaslop_gxp_shader::usse::decode_secondary_shader(&p);
            let pri = vitaslop_gxp_shader::usse::decode_shader(&p);
            for instr in sec.instrs.iter().chain(pri.instrs.iter()) {
                let w = instr.raw;
                if (instr.group != 0x1d && instr.group != 0x1e) || f(w, 53, 53) == 0 {
                    continue;
                }
                let off = f(w, 13, 7) + f(w, 6, 0);
                let on_start = starts.contains(&(off as i32));
                let prev_start = starts.contains(&(off as i32 - 1));
                let e = off_rows.entry((off, on_start || prev_start)).or_insert((0, name.clone()));
                e.0 += 1;
                let _ = prev_start;
            }
        }
    // >>> DOES A `moe_expand` LOAD CLOBBER A LITERAL THE PROGRAM STILL NEEDS?
    //
    // The destination of one of these is an SA register, and this title's DATA container also
    // places LITERALS in SA registers. If the load's destination lands on a literal that a LATER
    // instruction reads, then either the guest is deliberately reusing the register (legitimate
    // - the literal's last use is before the load) or the DESTINATION is mis-decoded the way the
    // offset was. The two are told apart by WHERE the literal is last read: after the load, the
    // program would be reading data it did not put there.
    {
        let mut rows: BTreeMap<(bool, bool), (usize, String)> = BTreeMap::new();
        for (name, bytes) in blobs(&dir) {
            let Ok(p) = Program::parse(&bytes) else { continue };
            let lits: std::collections::BTreeSet<u32> =
                p.literals.iter().map(|l| l.0).collect();
            if lits.is_empty() {
                continue;
            }
            let sec = vitaslop_gxp_shader::usse::decode_secondary_shader(&p);
            let pri = vitaslop_gxp_shader::usse::decode_shader(&p);
            for instr in sec.instrs.iter() {
                let w = instr.raw;
                if (instr.group != 0x1d && instr.group != 0x1e) || f(w, 53, 53) == 0 {
                    continue;
                }
                let Some(d) = instr.dest.as_ref() else { continue };
                let dst = d.index as u32;
                let on_literal = lits.contains(&dst);
                // Is that register read by the PRIMARY, i.e. after every secondary load?
                let read_later = pri.instrs.iter().any(|i| {
                    i.srcs.iter().any(|o| {
                        matches!(o.bank, vitaslop_gxp_shader::ir::Bank::SecondaryAttr)
                            && o.index as u32 == dst
                    })
                });
                let e = rows.entry((on_literal, read_later)).or_insert((0, name.clone()));
                e.0 += 1;
            }
        }
        println!("
-- moe_expand LOAD destinations: does the dest hold a LITERAL, and is it read by the PRIMARY? --");
        println!("  dest_is_a_literal  read_by_primary  count  example");
        for ((lit, later), (n, example)) in &rows {
            println!("  {lit:<18} {later:<16} {n:<6} {example}");
        }
    }
        println!("
-- moe_expand LOAD offsets (in registers): odd? on a declared parameter start? --");
        println!("  offset  odd    lands_on_or_after_a_param_start  count  example");
        for ((off, near), (n, example)) in &off_rows {
            println!("  {off:<7} {:<6} {near:<31} {n:<6} {example}", off % 2 == 1);
        }
    }
    println!("
-- moe_expand (bit 53) x elements x SMLSI-earlier-in-program --");
    println!("  dir  moe_expand  elems  smlsi_before  count  example");
    for ((dir_, m, elems, smlsi), (n, example)) in &moe {
        let dirn = match dir_ {
            1 => "LD",
            2 => "ST",
            _ => "??",
        };
        println!("  {dirn}   {m}           {elems:<5}  {smlsi:<12}  {n:<5}  {example}");
    }
}

/// >>> WHICH SAMPLER VARIANTS THE CORPUS ACTUALLY CONTAINS - by `lod_mode`, sub-behaviour,
/// >>> coordinate count and coordinate precision, straight off the WORDS.
///
/// The emitter picks a different WGSL builtin per LOD mode - `textureSample`,
/// `textureSampleBias`, `textureSampleLevel`, `textureSampleGrad` - and picking the wrong one
/// is a visibly wrong image rather than a compile error. Which of those four paths any test has
/// ever run was not recorded anywhere, so "the sampler group is covered" was a claim about the
/// group and not about its variants.
///
/// >>> IT COUNTS WORDS, NOT DECODED INSTRUCTIONS, and that distinction is the whole point. A
/// variant the decoder BLOCKS never becomes an `Op::Tex`, so a census over decoded programs
/// would report exactly zero of precisely the variants that are unimplemented - the ones worth
/// knowing about. The fields here are read at the positions `decode_grp_tex` documents.
#[test]
#[ignore = "needs a captured corpus (game bytes); set VITASLOP_GXP_CORPUS"]
fn sampler_variant_census() {
    use std::collections::BTreeMap;
    let Some(dir) = corpus_dir() else { return };
    let f = |w: u64, hi: u32, lo: u32| (w >> lo) & ((1u64 << (hi - lo + 1)) - 1);
    let mut lod: BTreeMap<u64, usize> = BTreeMap::new();
    let mut sb: BTreeMap<u64, usize> = BTreeMap::new();
    let mut dim: BTreeMap<u64, usize> = BTreeMap::new();
    let mut blocked: BTreeMap<String, usize> = BTreeMap::new();
    let (mut words, mut progs) = (0usize, 0usize);
    for (_, bytes) in blobs(&dir) {
        let Ok(p) = Program::parse(&bytes) else { continue };
        let mut here = 0usize;
        for w in p.code.iter().chain(p.secondary_code.iter()) {
            if (w >> 59) & 0x1f != 0x1c {
                continue;
            }
            words += 1;
            here += 1;
            *lod.entry(f(*w, 41, 40)).or_default() += 1;
            *sb.entry(f(*w, 38, 37)).or_default() += 1;
            *dim.entry(f(*w, 43, 42) + 1).or_default() += 1;
            // What the DECODER makes of the same word - the other half of the question, and the
            // half that says whether a variant the corpus contains is one we can translate.
            if let Some(why) = vitaslop_gxp_shader::usse::decode(*w).blocked {
                *blocked.entry(why.to_string()).or_default() += 1;
            }
        }
        if here > 0 {
            progs += 1;
        }
    }
    let name = |m: u64| match m {
        0 => "implicit (textureSample)",
        1 => "bias     (textureSampleBias)",
        2 => "level    (textureSampleLevel)",
        _ => "gradient (textureSampleGrad)",
    };
    println!("\n=== SAMPLER VARIANTS IN THE CORPUS: {words} group-0xE0 words in {progs} programs ===");
    println!("  by lod_mode (41:40) - each is a DIFFERENT WGSL builtin:");
    for m in 0..4u64 {
        println!("    {} {:>6}", name(m), lod.get(&m).copied().unwrap_or(0));
    }
    println!("  by sb_mode (38:37) - 0 is the ordinary sample, 3 the gather:");
    for (k, n) in &sb {
        println!("    {k} {n:>6}");
    }
    println!("  by coordinate count (dim 43:42, base-1):");
    for (k, n) in &dim {
        println!("    {k}D {n:>6}");
    }
    println!("  of those words, the ones the DECODER refuses, by reason:");
    if blocked.is_empty() {
        println!("    (none - every sampler word in the corpus decodes)");
    }
    let mut rows: Vec<_> = blocked.iter().collect();
    rows.sort_by_key(|(_, n)| std::cmp::Reverse(**n));
    for (why, n) in rows {
        println!("    {n:>6}  {why}");
    }
    assert!(words > 0, "no sampler words in this corpus - the census measured nothing");
}

/// Census of the 0x18 DOT group's bits 47:44 - the four bits every group with a documented
/// `repeat_count` puts it at, and which this group's own field table names `unk7`, `abs_op2`,
/// `swz_en_strange1`, `swz_en_strange0`.
#[test]
#[ignore = "needs a captured corpus (game bytes); set VITASLOP_GXP_CORPUS"]
fn dot_repeat_field_census() {
    use std::collections::BTreeMap;
    let Some(dir) = corpus_dir() else { return };
    let mut hist: BTreeMap<u64, usize> = BTreeMap::new();
    let mut blobs_with: BTreeMap<String, Vec<(usize, u64)>> = BTreeMap::new();
    for (name, bytes) in blobs(&dir) {
        let Ok(p) = Program::parse(&bytes) else { continue };
        for (i, w) in p.code.iter().chain(p.secondary_code.iter()).enumerate() {
            if (w >> 59) & 0x1f != 0x03 {
                continue;
            }
            // opcode2 (word bit 53) splits DOT (0) from MAD (1).
            if (w >> 53) & 1 != 0 {
                continue;
            }
            let f = (w >> 44) & 0xf;
            *hist.entry(f).or_default() += 1;
            if f != 0 {
                blobs_with.entry(name.clone()).or_default().push((i, *w));
            }
        }
    }
    println!("0x18 DOT bits 47:44 histogram:");
    for (v, n) in &hist {
        println!("  {v:#04x} (unk7={} abs_op2={} strange1={} strange0={}): {n}", v >> 3, (v >> 2) & 1, (v >> 1) & 1, v & 1);
    }
    println!("blobs with a non-zero field: {}", blobs_with.len());
    for (name, ws) in blobs_with.iter().take(20) {
        let list: Vec<String> = ws.iter().map(|(i, w)| format!("#{i}={w:#018x}")).collect();
        println!("  {name}: {}", list.join(" "));
    }
}

/// Census of the 0xD0 `mad` group (opcode1 = 0x1a): every word in the corpus, every bit
/// position's set-count, and the distinct words with the programs they appear in.
///
/// The wiki gives this group's opcode1, its 3-bit predicate (58:56), a reserved-zero at bit 53,
/// a `modifier` at 52 (`s0`/`s1`) and a `data_format` at 41 (u32/i32) - and marks EVERY operand
/// byte "?". This is the raw material for settling the rest.
#[test]
#[ignore = "needs a captured corpus (game bytes); set VITASLOP_GXP_CORPUS"]
fn group_d0_word_census() {
    let Some(dir) = corpus_dir() else { return };
    let mut hist: BTreeMap<u64, Vec<(String, usize)>> = BTreeMap::new();
    let mut bit_set = [0usize; 64];
    let mut total = 0usize;
    for (name, bytes) in blobs(&dir) {
        let Ok(p) = Program::parse(&bytes) else { continue };
        for (i, w) in p.code.iter().chain(p.secondary_code.iter()).enumerate() {
            if (w >> 59) & 0x1f != 0x1a {
                continue;
            }
            total += 1;
            for b in 0..64 {
                if (w >> b) & 1 != 0 {
                    bit_set[b] += 1;
                }
            }
            hist.entry(*w).or_default().push((name.clone(), i));
        }
    }
    println!("\n-- 0xD0 group: {total} words, {} distinct --", hist.len());
    println!("bit set-counts (bit: count/total):");
    for b in (0..64).rev() {
        let c = bit_set[b];
        let tag = if c == 0 { "ZERO" } else if c == total { "ONE " } else { "vary" };
        println!("  bit {b:>2}: {c:>4}/{total}  {tag}");
    }
    println!("\ndistinct words:");
    for (w, uses) in &hist {
        let names: Vec<String> =
            uses.iter().take(6).map(|(n, i)| format!("{n}#{i}")).collect();
        println!("  {w:#018x}  x{:<4} {}", uses.len(), names.join(" "));
    }
}

/// Census of INTERNAL-REGISTER def-use across the corpus, split by the PRECISION of the
/// instruction that writes and the one that reads.
///
/// The emitter reads every bank at the instruction's own precision, so an F16 instruction
/// reading `i0` takes two 32-bit registers as four packed halves while an F32 one takes four
/// registers as four floats. If a single internal register is routinely written at one
/// precision and read at the other, that model cannot be what the hardware does - the two
/// readings do not even touch the same registers.
#[test]
#[ignore = "needs a captured corpus (game bytes); set VITASLOP_GXP_CORPUS"]
fn internal_register_def_use_precision_census() {
    use vitaslop_gxp_shader::ir::{Bank, Op};
    let Some(dir) = corpus_dir() else { return };
    // (writer half, reader half) -> (uses, one example)
    let mut pairs: BTreeMap<(bool, bool), (usize, String)> = BTreeMap::new();
    let mut mixed_programs = 0usize;
    let mut programs = 0usize;
    for (name, bytes) in blobs(&dir) {
        let Ok(p) = Program::parse(&bytes) else { continue };
        for shader in [
            vitaslop_gxp_shader::usse::decode_shader(&p),
            vitaslop_gxp_shader::usse::decode_secondary_shader(&p),
        ] {
            if shader.instrs.is_empty() {
                continue;
            }
            programs += 1;
            // The precision the last writer of each internal LANE used, or `None` if no
            // instruction has written it yet in stream order.
            let mut wrote: [Option<bool>; 16] = [None; 16];
            let mut mixed_here = false;
            for instr in &shader.instrs {
                if matches!(instr.op, Op::Branch { .. } | Op::Nop) {
                    continue;
                }
                for src in &instr.srcs {
                    if !matches!(src.bank, Bank::Internal) {
                        continue;
                    }
                    // Every lane this read touches under the CURRENT model, so the tally is
                    // about the model actually in the emitter and not an idealised one.
                    for c in 0..4usize {
                        if !instr.write_mask[c] {
                            continue;
                        }
                        let half = instr.source_half_precision();
                        let lane = src.index as usize
                            + if half { c >> 1 } else { c };
                        let Some(Some(w)) = wrote.get(lane).copied() else { continue };
                        let e = pairs.entry((w, half)).or_insert((0, name.clone()));
                        e.0 += 1;
                        if w != half {
                            mixed_here = true;
                        }
                    }
                }
                let Some(d) = instr.dest else { continue };
                if !matches!(d.bank, Bank::Internal) {
                    continue;
                }
                for c in 0..4usize {
                    if !instr.write_mask[c] {
                        continue;
                    }
                    let lane = d.index as usize
                        + if instr.half_precision { c >> 1 } else { c };
                    if let Some(slot) = wrote.get_mut(lane) {
                        *slot = Some(instr.half_precision);
                    }
                }
            }
            if mixed_here {
                mixed_programs += 1;
            }
        }
    }
    println!(
        "\n-- internal-register def-use precision over {programs} programs \
         ({mixed_programs} mix precisions on one lane) --"
    );
    for ((w, r), (n, example)) in &pairs {
        let name = |h: &bool| if *h { "f16" } else { "f32" };
        println!("  written {} -> read {}: {n:<6} e.g. {example}", name(w), name(r));
    }
}

/// Every instruction that touches an internal register, in the programs that read one at a
/// DIFFERENT precision from the one that wrote it - the whole evidence for what an internal
/// register's storage format is.
#[test]
#[ignore = "needs a captured corpus (game bytes); set VITASLOP_GXP_CORPUS"]
fn internal_register_mixed_precision_programs() {
    use vitaslop_gxp_shader::ir::{Bank, Op};
    let Some(dir) = corpus_dir() else { return };
    for (name, bytes) in blobs(&dir) {
        let Ok(p) = Program::parse(&bytes) else { continue };
        let shader = vitaslop_gxp_shader::usse::decode_shader(&p);
        let touches = |i: &vitaslop_gxp_shader::ir::Instr| {
            i.dest.is_some_and(|d| matches!(d.bank, Bank::Internal))
                || i.srcs.iter().any(|s| matches!(s.bank, Bank::Internal))
        };
        let halves: Vec<bool> = shader
            .instrs
            .iter()
            .filter(|i| touches(i) && !matches!(i.op, Op::Branch { .. } | Op::Nop))
            .map(|i| i.half_precision)
            .collect();
        if halves.iter().any(|&h| h) && halves.iter().any(|&h| !h) {
            println!("\n== {name}");
            for (at, instr) in shader.instrs.iter().enumerate() {
                if !touches(instr) {
                    continue;
                }
                println!(
                    "  #{at:<3} {:<10} half={} mask={:?} dst={:?} srcs={:?}",
                    instr.op.mnemonic(),
                    instr.half_precision,
                    instr.write_mask,
                    instr.dest.map(|d| (d.bank, d.index)),
                    instr
                        .srcs
                        .iter()
                        .map(|s| (s.bank, s.index, s.swizzle))
                        .collect::<Vec<_>>()
                );
            }
        }
    }
}

/// A digest of every blob's recompiled WGSL, so an emitter change can be diffed over the whole
/// corpus at once: run it before and after, and the blobs whose line differs are exactly the
/// ones the change touched.
#[test]
#[ignore = "needs a captured corpus (game bytes); set VITASLOP_GXP_CORPUS"]
fn digest_every_blob_wgsl() {
    let Some(dir) = corpus_dir() else { return };
    for (name, bytes) in blobs(&dir) {
        let Ok(p) = Program::parse(&bytes) else {
            println!("{name} PARSE-FAIL");
            continue;
        };
        let body = match p.kind {
            ProgramKind::Vertex => recompile_vertex(&bytes).map(|m| m.wgsl_body),
            _ => recompile_fragment(&bytes).map(|m| m.wgsl_body),
        };
        match body {
            Ok(w) => {
                // FNV-1a over the emitted text: stable, dependency-free, and enough to say
                // "this blob's output changed".
                let mut h: u64 = 0xcbf29ce484222325;
                for b in w.as_bytes() {
                    h ^= *b as u64;
                    h = h.wrapping_mul(0x100000001b3);
                }
                println!("{name} {h:016x} {}", w.len());
            }
            Err(e) => println!("{name} BLOCKED {e}"),
        }
    }
}

/// Every distinct word of one opcode group across the corpus, with the programs it appears in.
/// Set `VITASLOP_GXP_GROUP` to the 5-bit `opcode1` value in hex (e.g. `0f` for VTSTMSK).
///
/// Reads the parsed CODE region rather than the file, so string tables and parameter data
/// cannot masquerade as instructions - which a raw byte scan for a top-five-bit pattern
/// otherwise does, in quantity.
#[test]
#[ignore = "needs a captured corpus (game bytes); set VITASLOP_GXP_CORPUS and VITASLOP_GXP_GROUP"]
fn group_word_census() {
    let Some(dir) = corpus_dir() else { return };
    let Some(group) = std::env::var("VITASLOP_GXP_GROUP")
        .ok()
        .and_then(|v| u64::from_str_radix(v.trim_start_matches("0x"), 16).ok())
    else {
        eprintln!("set VITASLOP_GXP_GROUP=<opcode1 in hex>");
        return;
    };
    let mut hist: BTreeMap<u64, Vec<(String, usize)>> = BTreeMap::new();
    let mut bit_set = [0usize; 64];
    let mut total = 0usize;
    for (name, bytes) in blobs(&dir) {
        let Ok(p) = Program::parse(&bytes) else { continue };
        for (i, w) in p.code.iter().chain(p.secondary_code.iter()).enumerate() {
            if (w >> 59) & 0x1f != group {
                continue;
            }
            total += 1;
            for (b, slot) in bit_set.iter_mut().enumerate() {
                if (w >> b) & 1 != 0 {
                    *slot += 1;
                }
            }
            hist.entry(*w).or_default().push((name.clone(), i));
        }
    }
    println!("\n-- group {group:#04x}: {total} words, {} distinct --", hist.len());
    let varying: Vec<usize> =
        (0..64).rev().filter(|&b| bit_set[b] != 0 && bit_set[b] != total).collect();
    let ones: Vec<usize> = (0..64).rev().filter(|&b| bit_set[b] == total && total > 0).collect();
    println!("  always set: {ones:?}");
    println!("  varying:    {varying:?}");
    for (w, uses) in &hist {
        let names: Vec<String> = uses.iter().take(6).map(|(n, i)| format!("{n}#{i}")).collect();
        println!("  {w:#018x}  x{:<4} {}", uses.len(), names.join(" "));
    }
}

/// Every program that carries a 0xE8 memory load, with the uniform-buffer shape the memory
/// window resolves against - or the reason it does not.
#[test]
#[ignore = "needs a captured corpus (game bytes); set VITASLOP_GXP_CORPUS"]
fn memory_window_resolution_census() {
    use vitaslop_gxp_shader::container::ParamCategory;
    use vitaslop_gxp_shader::ir::Op;
    let Some(dir) = corpus_dir() else { return };
    for (name, bytes) in blobs(&dir) {
        let Ok(p) = Program::parse(&bytes) else { continue };
        let shader = vitaslop_gxp_shader::usse::decode_shader(&p);
        if !shader.instrs.iter().any(|i| matches!(i.op, Op::MemLoad { .. })) {
            continue;
        }
        let sa_resident = p.sa_uniform_buffers();
        println!("\n== {name} ({:?})", p.kind);
        println!("  window: {:?}", vitaslop_gxp_shader::module::resolve_mem_windows(&p, &shader));
        println!("  sa-resident buffers: {sa_resident:?}");
        println!("  +0x78 bindings: {:?}", p.uniform_buffer_bindings);
        for c in &p.containers {
            println!("    CONTAINER {} @ sa[{}] x{}", c.index, c.base_sa, c.size_regs);
        }
        for par in &p.parameters {
            if par.category != ParamCategory::UniformBuffer {
                continue;
            }
            println!(
                "    UB param name={:?} resource_index={} array_size={} container={}",
                par.name, par.resource_index, par.array_size, par.container_index
            );
        }
    }
}

/// For every program with a 0xE8 memory load: which SA registers at or above the DATA
/// container it READS, and which instruction reads each. The +0x78 table places a bound
/// buffer's guest ADDRESS in one of those registers, so this is what says whether a table
/// entry is live.
#[test]
#[ignore = "needs a captured corpus (game bytes); set VITASLOP_GXP_CORPUS"]
fn data_container_sa_reads_census() {
    use vitaslop_gxp_shader::ir::{Bank, Op};
    let Some(dir) = corpus_dir() else { return };
    for (name, bytes) in blobs(&dir) {
        let Ok(p) = Program::parse(&bytes) else { continue };
        let shader = vitaslop_gxp_shader::usse::decode_shader(&p);
        if !shader.instrs.iter().any(|i| matches!(i.op, Op::MemLoad { .. })) {
            continue;
        }
        let Some(data) = p.containers.iter().find(|c| c.index == 19) else { continue };
        println!("\n== {name}: DATA @ sa[{}] x{}", data.base_sa, data.size_regs);
        let lo = u32::from(data.base_sa);
        let hi = lo + u32::from(data.size_regs);
        let mut seen: BTreeMap<u32, Vec<String>> = BTreeMap::new();
        for (at, instr) in shader.instrs.iter().enumerate() {
            for src in &instr.srcs {
                if src.bank != Bank::SecondaryAttr {
                    continue;
                }
                let r = u32::from(src.index);
                if r < lo || r >= hi {
                    continue;
                }
                seen.entry(r).or_default().push(format!("#{at} {}", instr.op.mnemonic()));
            }
        }
        for (r, who) in &seen {
            let lit = p.literals.iter().find(|(reg, _)| *reg == *r).map(|(_, v)| *v);
            println!("  sa[{r}] slot {} literal={lit:?} read by {}", r - lo, who.join(", "));
        }
    }
}

/// Every AMBIGUOUS-order vertex program with the two verdicts the linker's convention gate can
/// return: the STRICT one it uses today (every written lane inside a run AND every run's first
/// lane written) and the one `assumed_varying_orders_the_vertex_code_contradicts` established
/// (only a write BELOW the top of the layout that no run covers refutes a layout).
#[test]
#[ignore = "needs a captured corpus (game bytes); set VITASLOP_GXP_CORPUS"]
fn ambiguous_order_convention_gate_census() {
    use vitaslop_gxp_shader::container::VaryingOrder;
    
    const POSITION_LANES: usize = 4;
    let Some(dir) = corpus_dir() else { return };
    let (mut n, mut both, mut relaxed_only, mut neither) = (0usize, 0usize, 0usize, 0usize);
    for (name, bytes) in blobs(&dir) {
        let Ok(p) = Program::parse(&bytes) else { continue };
        if p.kind != ProgramKind::Vertex || p.output_order != VaryingOrder::Ambiguous {
            continue;
        }
        n += 1;
        let shader = vitaslop_gxp_shader::usse::decode_shader(&p);
        let written = vitaslop_gxp_shader::usse::written_output_lanes(&shader);
        let in_a_run = |lane: usize| {
            lane < POSITION_LANES
                || p.output_varyings.iter().any(|v| {
                    let lo = v.base_lane as usize;
                    lane >= lo && lane < lo + v.components as usize
                })
        };
        let top = p
            .output_varyings
            .iter()
            .map(|v| v.base_lane as usize + v.components as usize)
            .max()
            .unwrap_or(0);
        let stray: Vec<usize> =
            written.iter().enumerate().filter(|&(l, &w)| w && !in_a_run(l)).map(|(l, _)| l).collect();
        let inside: Vec<usize> = stray.iter().copied().filter(|&l| l < top).collect();
        let above: Vec<usize> = stray.iter().copied().filter(|&l| l >= top).collect();
        let unstarted: Vec<String> = p
            .output_varyings
            .iter()
            .filter(|v| !written.get(v.base_lane as usize).copied().unwrap_or(false))
            .map(|v| format!("{:?}@{}", v.usage, v.base_lane))
            .collect();
        let strict = stray.is_empty() && unstarted.is_empty();
        let relaxed = inside.is_empty();
        match (strict, relaxed) {
            (true, _) => both += 1,
            (false, true) => relaxed_only += 1,
            (false, false) => neither += 1,
        }
        println!(
            "{name}: varyings={} strict={strict} relaxed={relaxed} inside={inside:?} \
             above={above:?} unstarted=[{}]",
            p.output_varyings.len(),
            unstarted.join(" ")
        );
    }
    println!(
        "\nambiguous-order vertex programs: {n}; {both} pass both gates, {relaxed_only} pass only \
         the established (relaxed) one, {neither} pass neither"
    );
}

/// The closure behind the group-0x1a reading: EVERY group-0x1a instruction in the corpus is
/// part of a well-formed multiply-add pair, so the step semantics are only ever used where the
/// pair's net result is the same under every reading of `sn` that survives (see
/// `decode_grp_imad32_step`).
///
/// This is the statement that would break first if a new title used the group differently, and
/// it is the one that makes the decode safe rather than merely plausible - so it ASSERTS.
#[test]
#[ignore = "needs a captured corpus (game bytes); set VITASLOP_GXP_CORPUS"]
fn every_imad_step_is_part_of_a_pair() {
    use vitaslop_gxp_shader::ir::Op;
    let Some(dir) = corpus_dir() else { return };
    let (mut steps, mut programs) = (0usize, 0usize);
    let mut lone: Vec<String> = Vec::new();
    for (name, bytes) in blobs(&dir) {
        let Ok(p) = Program::parse(&bytes) else { continue };
        let shader = vitaslop_gxp_shader::usse::decode_shader(&p);
        let mut here = 0usize;
        for (at, instr) in shader.instrs.iter().enumerate() {
            if !matches!(instr.op, Op::IntMadStep { .. }) {
                continue;
            }
            here += 1;
            steps += 1;
            if let Some(why) = instr.blocked {
                lone.push(format!("{name}#{at}: {why}"));
            }
        }
        if here > 0 {
            programs += 1;
        }
    }
    println!("group 0x1a: {steps} steps over {programs} programs, {} not in a pair", lone.len());
    assert!(lone.is_empty(), "a group-0x1a step outside a well-formed pair:\n{}", lone.join("\n"));
}

/// The closure behind the group-0xE0 GATHER reading: every gather in the corpus samples a
/// ONE-component texture, which is what fixes where its four bilinear coefficients land
/// (`dest + 4`, after the four gathered texels). A wider sampler would put them somewhere this
/// decode cannot name, so the day one appears is the day this fails and says so.
#[test]
#[ignore = "needs a captured corpus (game bytes); set VITASLOP_GXP_CORPUS"]
fn every_gather4_samples_a_single_component_texture() {
    use vitaslop_gxp_shader::ir::Op;
    let Some(dir) = corpus_dir() else { return };
    let mut found = 0usize;
    let mut wrong: Vec<String> = Vec::new();
    for (name, bytes) in blobs(&dir) {
        let Ok(p) = Program::parse(&bytes) else { continue };
        let shader = vitaslop_gxp_shader::usse::decode_shader(&p);
        for (at, instr) in shader.instrs.iter().enumerate() {
            let Op::TexGather { unit, .. } = instr.op else { continue };
            found += 1;
            let comps = p.sampler_at(u32::from(unit)).map_or(0, |s| s.component_count);
            if comps != 1 || instr.blocked.is_some() {
                wrong.push(format!("{name}#{at}: unit {unit} has {comps} component(s), blocked={:?}", instr.blocked));
            }
        }
    }
    println!("group 0xE0 gather4: {found} instructions, {} outside the established shape", wrong.len());
    assert!(wrong.is_empty(), "a gather4 outside the established shape:\n{}", wrong.join("\n"));
}

/// Every program that declares a uniform BUFFER neither path feeds: not SA-resident (the
/// driver copies it into the register file) and not a memory window (the program chases its
/// address). Such a program reads ZEROES where the guest put data, silently.
///
/// This is the offline form of the renderer's `report_unfed_uniforms`, so the same question
/// can be answered over the whole corpus instead of one run's pairs.
#[test]
#[ignore = "needs a captured corpus (game bytes); set VITASLOP_GXP_CORPUS"]
fn uniform_buffers_neither_path_feeds() {
    use vitaslop_gxp_shader::container::ParamCategory;
    let Some(dir) = corpus_dir() else { return };
    let (mut declared, mut unfed_programs) = (0usize, 0usize);
    for (name, bytes) in blobs(&dir) {
        let Ok(p) = Program::parse(&bytes) else { continue };
        let bufs: Vec<&vitaslop_gxp_shader::container::Parameter> = p
            .parameters
            .iter()
            .filter(|q| q.category == ParamCategory::UniformBuffer)
            .collect();
        if bufs.is_empty() {
            continue;
        }
        declared += 1;
        let resident = p.sa_uniform_buffers();
        let windows = vitaslop_gxp_shader::mem_windows_for_vertex_blob(&bytes);
        let unfed: Vec<String> = bufs
            .iter()
            .filter(|q| {
                q.resource_index < 0
                    || !(resident.iter().any(|b| b.buffer_index == q.resource_index as u32)
                        || windows.iter().any(|w| w.buffer_index == q.resource_index as u32))
            })
            .map(|q| format!("buffer {} ({} bytes, container {})", q.resource_index, q.array_size, q.container_index))
            .collect();
        if !unfed.is_empty() {
            unfed_programs += 1;
            println!("{name} ({:?}): {}", p.kind, unfed.join(", "));
        }
    }
    println!(
        "\n{declared} programs declare a uniform buffer; {unfed_programs} declare one neither \
         path feeds"
    );
}

/// The DEFAULT uniform buffer's SA container ends exactly on a PARAMETER boundary, over every
/// blob in the corpus that has both a container 14 and a pointer to the same buffer.
///
/// This is the oracle for [`MemWindow::base_offset`]. The reading it pins is that the driver
/// copies `container 14`'s `size_regs` into the SA file and points the DATA slot at the FIRST
/// REGISTER IT DID NOT COPY, so a program reaches its overflow parameters at small offsets from
/// that pointer. If that is right, the cut must fall between two parameters and never through
/// one - a container ending mid-`float4` would leave half a parameter in each address space and
/// no offset could name it.
///
/// It is what turns two hand-checked programs into a corpus statement. The two:
/// `vert_820d6730` carries 31 of 34 registers and its leftover is exactly `sunColor` (reg 31,
/// 3 components); `vert_81d72040` carries 14 of 28 and its leftover is exactly
/// `g_DiffuseRange` + `g_Material.{diffuse,fresnel,ambient}` (regs 14..28).
#[test]
#[ignore = "needs a captured corpus (game bytes); set VITASLOP_GXP_CORPUS"]
fn the_default_uniform_container_ends_on_a_parameter_boundary() {
    use vitaslop_gxp_shader::container::ParamCategory;
    let Some(dir) = corpus_dir() else { return };
    let mut checked = 0usize;
    let mut reached = 0usize;
    let mut fully_carried_with_a_pointer: Vec<String> = Vec::new();
    let mut straddled: Vec<String> = Vec::new();
    for (name, bytes) in blobs(&dir) {
        let Ok(p) = Program::parse(&bytes) else { continue };
        let Some(c) = p.containers.iter().find(|c| c.index == 14) else { continue };
        let carried = u32::from(c.size_regs);
        // Only programs that actually reach past the copy - otherwise the boundary is the end
        // of the buffer and says nothing.
        //
        // But COUNT the other case, because it is where this change could newly REFUSE a
        // program that used to link: with nothing left past the copy the window is zero bytes,
        // and `resolve_mem_windows` treats a zero-size buffer whose pointer register is READ as
        // unestablished rather than guessing. If any program in a shipped title is that shape,
        // the fix would drop it to fixed-function and the sweep would have to catch it.
        if carried >= p.default_uniform_regs {
            let shader = vitaslop_gxp_shader::usse::decode_shader(&p);
            let points_at_default = p
                .uniform_buffer_bindings
                .iter()
                .any(|b| b.buffer_index == 14);
            let loads = shader
                .instrs
                .iter()
                .any(|i| matches!(i.op, vitaslop_gxp_shader::ir::Op::MemLoad { .. }));
            if points_at_default && loads {
                fully_carried_with_a_pointer.push(name.clone());
            }
            continue;
        }
        checked += 1;
        // A parameter STRADDLES the cut when it starts before `carried` and ends after it. Its
        // extent is components x array, in the register units `resource_index` is expressed in
        // for an F32 uniform - the only type any of these declare.
        let straddling: Vec<String> = p
            .parameters
            .iter()
            .filter(|q| {
                q.category == ParamCategory::Uniform
                    && q.container_index == 14
                    && q.resource_index >= 0
            })
            .filter_map(|q| {
                let start = q.resource_index as u32;
                // The extent in REGISTERS, not components: an F16 uniform packs TWO components
                // into one 32-bit register, so `float4` there is 2 registers and not 4. Counting
                // components reported two of this corpus's fragment programs as straddling a cut
                // that in fact falls exactly at the end of their only parameter.
                // [[vitaslop-uniform-extent-is-registers-not-components]]
                let width = q.ptype.component_bytes().unwrap_or(4);
                let regs = (u32::from(q.component_count.max(1)) * q.array_size.max(1) * width)
                    .div_ceil(4);
                (start < carried && start + regs > carried)
                    .then(|| format!("{} at reg {start} x{regs}", q.name))
            })
            .collect();
        // Whether the offset actually REACHES this program - i.e. whether it resolves a window
        // for buffer 14 at all. A program with a partly-carried buffer but no memory load never
        // chases the pointer, so the boundary above is true of it and inert for it, and the
        // no-regression claim for the other titles rests on this column rather than on a sweep.
        let shader = vitaslop_gxp_shader::usse::decode_shader(&p);
        let window = vitaslop_gxp_shader::module::resolve_mem_windows(&p, &shader)
            .unwrap_or_default()
            .into_iter()
            .find(|w| w.buffer_index == 14);
        if window.is_some() {
            reached += 1;
        }
        println!(
            "{name}: container 14 carries {carried} of {} declared registers - {}{}",
            p.default_uniform_regs,
            match &window {
                Some(w) => format!("pointer at +{} over {} bytes", w.base_offset, w.bytes),
                None => "NO window (it never chases the pointer)".to_string(),
            },
            if straddling.is_empty() { String::new() } else { format!("  STRADDLED BY {}", straddling.join(", ")) }
        );
        if !straddling.is_empty() {
            straddled.push(format!("{name}: {}", straddling.join(", ")));
        }
    }
    println!(
        "
{checked} programs keep part of their default uniform buffer past container 14;          {reached} of them actually chase the pointer, and only those are affected by          `MemWindow::base_offset`"
    );
    assert!(
        fully_carried_with_a_pointer.is_empty(),
        "these programs carry their WHOLE default uniform buffer in container 14 yet still take a pointer to it and load memory - nothing is left past the copy, so `base_offset` would make their window zero bytes and `resolve_mem_windows` would refuse them: {:#?}",
        fully_carried_with_a_pointer
    );
    assert!(
        straddled.is_empty(),
        "the default uniform container cuts THROUGH a parameter, so the pointer's offset cannot name it and `MemWindow::base_offset` is the wrong reading: {straddled:#?}"
    );
}

/// Every blob whose decoded stream contains a BACKWARD branch - i.e. every blob whose recompile
/// depends on the loop reconstruction rather than on the straight-line skip structuring.
#[test]
#[ignore = "needs a captured corpus (game bytes); set VITASLOP_GXP_CORPUS"]
fn blobs_with_a_backward_branch() {
    use vitaslop_gxp_shader::ir::Op;
    let Some(dir) = corpus_dir() else { return };
    let mut n = 0usize;
    for (name, bytes) in blobs(&dir) {
        let Ok(p) = Program::parse(&bytes) else { continue };
        let shader = vitaslop_gxp_shader::usse::decode_shader(&p);
        let back: Vec<usize> = shader
            .instrs
            .iter()
            .enumerate()
            .filter(|(at, i)| matches!(i.op, Op::Branch { rel } if *at as i64 + rel as i64 <= *at as i64))
            .map(|(at, _)| at)
            .collect();
        if !back.is_empty() {
            n += 1;
            println!("{name} ({:?}): backward branches at {back:?}", p.kind);
        }
    }
    println!("
{n} blob(s) carry a backward branch");
}

/// Every blob whose +0x78 uniform-buffer binding table has more than ONE entry - the only
/// blobs whose parse can differ between an 8-byte and a 16-byte entry stride - with the
/// SA-RESIDENT buffer list that table filters.
#[test]
#[ignore = "needs a captured corpus (game bytes); set VITASLOP_GXP_CORPUS"]
fn multi_entry_uniform_buffer_binding_tables() {
    let Some(dir) = corpus_dir() else { return };
    let mut n = 0usize;
    for (name, bytes) in blobs(&dir) {
        let Ok(p) = Program::parse(&bytes) else { continue };
        if p.uniform_buffer_bindings.len() < 2 {
            continue;
        }
        n += 1;
        println!(
            "{name} ({:?}): {:?} | sa-resident {:?}",
            p.kind, p.uniform_buffer_bindings, p.sa_uniform_buffers()
        );
    }
    println!("
{n} blob(s) have more than one +0x78 entry");
}

/// >>> THE WORD THAT PANICS THE RECOMPILER ON A REAL PHONE, REPRODUCED OFFLINE IN A TEST.
///
/// `gxp pair 7089f16e34be693f ... blocked USSE instruction #24 at code byte 0xc0
/// (raw 0x7802019271f6a839): 0x78 VTSTMSK mask type other than NUMERIC`, hit playing
/// CHALLENGE on the third course. The pair is in no local corpus and its word appears in no
/// file of the extracted container (the shaders are packed inside the `.xb` archives).
///
/// **It does not need to be.** The refusal is a DECODE refusal - `decode` sets `blocked` and
/// the emitter hard-fails on it, before any GPU work - so the whole crash reproduces from the
/// 64 bits, with no game, no device and no capture. That makes the turnaround on this a
/// second instead of a play session, and it is the reason to reach for the word rather than
/// the blob whenever the failure is in the decoder.
///
/// What the word CANNOT answer, and what a capture is still owed for, is the pair of open
/// questions below - both of which need the surrounding PROGRAM, not the instruction.
#[test]
fn the_word_that_blocks_the_recompiler_decodes_the_way_the_panic_says() {
    use vitaslop_gxp_shader::usse::decode::decode;
    const W: u64 = 0x7802019271f6a839;
    let i = decode(W);
    assert_eq!(i.group, 0x0f, "group 0x0f is the TEST family (0x78 VTSTMSK)");
    assert_eq!(i.raw, W);

    // Every field the panic rests on, read back from the word so a decoder change that moves
    // any of them fails HERE rather than on a phone. Bit positions are the group's documented
    // layout (see `decode_grp_test_mask`).
    let b = |hi: u32, lo: u32| ((W >> lo) & ((1u64 << (hi - lo + 1)) - 1)) as u32;
    assert_eq!(b(37, 36), 1, "tst_mask_type: 1, NOT the NUMERIC 2 - this is what blocks");
    assert_eq!(b(19, 18), 1, "alu_sel: the INTEGER family, not float");
    assert_eq!(b(17, 14), 10, "alu_op: the unsigned 16-bit SUBTRACT");
    assert_eq!((b(43, 42), b(41, 40), b(39, 39)), (0, 1, 1), "sign/zero/crcomb: the relation is == 0");
    assert_eq!(b(20, 20), 1, "test_wben: the destination IS written");
    assert_eq!(b(50, 50), 0, "test_flag_2 clear, so that open flag is not in play here");
    assert_eq!(b(58, 56), 0, "no predicate");
    // One operand is the special bank at index 16 - GLOBAL[16], which this project and the
    // vendor's own header independently name the BACK-FACE control register. So whatever the
    // mask turns out to be, this instruction is a FACING TEST.
    assert_eq!((b(31, 30), b(49, 49), b(13, 7)), (1, 1, 80), "src1: special bank, 64+16");

    // >>> AND THE TRANSLATION, WHICH IS NO LONGER BLOCKED.
    //
    // Both questions that held it are answered, and neither by guessing:
    //
    // 1. WHICH REGISTERS IT TOUCHES - settled by the CORPUS, not by either document. The two
    //    clean-room passes disagreed about operand doubling for the non-float families
    //    (`SA[57]`/`PA[15]` against `SA[114]`/`PA[30]`).
    //    `test_group_operand_numbering_evidence` finds 12 cases over 4 programs and two
    //    unrelated titles where a non-float TEST operand stays inside its program's declared
    //    register file ONLY if it is not doubled, and none the other way. Not doubled.
    // 2. WHAT `tst_mask_type = 1` WRITES - the two readings DISAGREE about the mechanism and
    //    AGREE about this combination's numbers. "A mask at the ALU precision" and "all-ones
    //    at the type width" both give 0xFFFF/0x0000 for the precision-mask form at an unsigned
    //    16-bit precision. The translation therefore does not depend on which rule is true,
    //    which is why it can ship while the mechanism stays open. The combinations where the
    //    two rules give DIFFERENT numbers are still refused - see `decode_grp_test_mask`.
    assert!(i.blocked.is_none(), "no longer blocked: {:?}", i.blocked);
    assert!(
        matches!(i.op, Op::TestMask { alu: TestAlu::IntSub16U, cmp: TestCmp::Eq }),
        "an unsigned 16-bit equality test, got {:?}",
        i.op
    );
    // The operands, at the numbering the corpus established.
    let d = i.dest.as_ref().expect("a mask writes a destination");
    assert_eq!((d.bank, d.index), (Bank::PrimaryAttr, 15), "PA[15], not PA[30]");
    assert_eq!((i.srcs[0].bank, i.srcs[0].index), (Bank::Global, 16), "GLOBAL[16], the facing register");
    assert_eq!((i.srcs[1].bank, i.srcs[1].index), (Bank::SecondaryAttr, 57), "SA[57], not SA[114]");
    // ONE channel. The count comes from the ALU family, not the mask type: four channels for
    // the float form (whose consumer dots them against bilinear coefficients), one for this.
    assert_eq!(i.write_mask, [true, false, false, false], "channel x only");
}

/// The combinations where the two readings of the mask field give DIFFERENT numbers must stay
/// refused. This is the half of the fix that is easy to lose: unblocking the established case
/// is only correct while its neighbours remain blocked, and a later "tidy-up" that widens the
/// gate would ship a silently wrong shadow or facing term.
#[test]
fn the_mask_forms_whose_readings_disagree_are_still_refused() {
    use vitaslop_gxp_shader::usse::decode::decode;
    const W: u64 = 0x7802019271f6a839;
    // Rewrite ONLY `tst_mask_type` (bits 37:36) and check each other value is refused.
    for mt in [0u64, 2, 3] {
        let w = (W & !(3 << 36)) | (mt << 36);
        let i = decode(w);
        assert!(
            i.blocked.is_some(),
            "mask type {mt} with the u16 family must stay blocked - the 8-bit-mask form and the numeric form are where the vendor naming and the emulator rule diverge"
        );
    }
    // And the established word is still fine, so the loop above is not passing by accident.
    assert!(decode(W).blocked.is_none());
}

/// >>> CAN THE CORPUS SETTLE THE TEST-GROUP OPERAND NUMBERING WITHOUT THE CRASHING BLOB?
///
/// The two clean-room passes disagree about whether the TEST group's register fields are
/// DOUBLED for the non-float ALU families: one doubles only for float, the other says the
/// field counts 32-bit registers with the low bit always zero and doubles unconditionally.
/// The word that panics is integer, so the two readings name different registers and cannot
/// be translated between.
///
/// This is the cheap oracle to try before capturing anything: a program declares how many
/// PRIMARY and SECONDARY attribute registers it uses, so a TEST-group instruction naming an
/// index that only ONE rule keeps inside the declared file refutes the other - no device, no
/// capture, no gameplay. Census only; it asserts nothing, because what it finds is the input
/// to a decision rather than the decision.
#[test]
#[ignore = "census over a corpus dir; run with VITASLOP_GXP_CORPUS=<dir>"]
fn test_group_operand_numbering_evidence() {
    use vitaslop_gxp_shader::usse::decode::decode;
    let Some(dir) = corpus_dir() else {
        println!("no VITASLOP_GXP_CORPUS set");
        return;
    };
    // (alu_sel, alu_op) -> how many words, and one example with its context.
    let mut fams: BTreeMap<(u32, u32), usize> = BTreeMap::new();
    let mut masks: BTreeMap<u32, usize> = BTreeMap::new();
    let mut decisive = 0usize;
    let mut total = 0usize;
    let mut progs = 0usize;
    for (name, bytes) in blobs(&dir) {
        let Ok(p) = Program::parse(&bytes) else { continue };
        progs += 1;
        let (pa, sa) = (p.primary_reg_count as u32, p.secondary_reg_count as u32);
        for (label, code) in [("primary", p.code.clone()), ("secondary", p.secondary_code.clone())] {
            for (i, &w) in code.iter().enumerate() {
                let g = ((w >> 59) & 0x1f) as u8;
                // 0x09 = VTST (predicate), 0x0f = VTSTMSK (per-channel mask).
                if g != 0x09 && g != 0x0f {
                    continue;
                }
                total += 1;
                let b = |hi: u32, lo: u32| ((w >> lo) & ((1u64 << (hi - lo + 1)) - 1)) as u32;
                let (alu_sel, alu_op) = (b(19, 18), b(17, 14));
                *fams.entry((alu_sel, alu_op)).or_default() += 1;
                if g == 0x0f {
                    *masks.entry(b(37, 36)).or_default() += 1;
                }
                // The evidence: a NON-FLOAT test whose doubled index leaves the declared file
                // while its undoubled one stays inside (or vice versa) decides the rule.
                if alu_sel == 0 {
                    continue;
                }
                let (s1b, s1n) = (b(31, 30), b(13, 7));
                let (s2b, s2n) = (b(29, 28), b(6, 0));
                for (banksel, n) in [(s1b, s1n), (s2b, s2n)] {
                    // bank 2 = pa (primary attr), 3 = sa (secondary attr) on this selector.
                    let limit = match banksel {
                        2 => pa,
                        3 => sa,
                        _ => continue,
                    };
                    if limit == 0 {
                        continue;
                    }
                    let (single, doubled) = (n < limit, n * 2 < limit);
                    if single != doubled {
                        decisive += 1;
                        println!(
                            "DECISIVE {name} {label}#{i} w={w:#018x} bank_sel={banksel} n={n} \
                             limit={limit} -> undoubled {}, doubled {}",
                            if single { "IN" } else { "OUT" },
                            if doubled { "IN" } else { "OUT" },
                        );
                    }
                }
                let _ = decode(w);
            }
        }
    }
    println!("\n{progs} programs, {total} TEST-group word(s)");
    println!("ALU (sel, op) populations: {fams:?}");
    println!("VTSTMSK tst_mask_type populations: {masks:?}");
    println!("decisive non-float operands found: {decisive}");
}

/// `0x48090881a00cc79f` - the BITWISE SHIFT-LEFT test that panicked a user's run one fix after
/// the VTSTMSK above. Decoded from the 64 bits alone: no game, no device, no capture.
///
/// Both spec sources give the BITWISE family's `alu_op` 3 as SHIFT LEFT and disagree about
/// nothing at or below 3, and the idiom refutes the only competing reading (the separate VBW
/// group's numbering, where 3 is XOR) because XOR makes the test unconditionally true and the
/// two complementary facing masks above it dead. See the arm in `decode_grp_test`.
#[test]
fn the_bitwise_shift_test_decodes_as_a_shift_of_the_facing_mask() {
    use vitaslop_gxp_shader::ir::{Bank, Op, TestAlu, TestCmp, TestReduce};
    use vitaslop_gxp_shader::usse::decode::decode;
    const W: u64 = 0x48090881a00cc79f;
    let i = decode(W);
    assert!(i.blocked.is_none(), "must translate: {:?}", i.blocked);
    assert_eq!(
        i.op,
        Op::Test {
            alu: TestAlu::BitShl,
            cmp: TestCmp::Gt,
            reduce: TestReduce::Channel(0),
            pdst: 0,
            write_back: false
        }
    );
    // src1 is pa[15] UNDOUBLED - the register the program's own VTSTMSK writes two
    // instructions earlier. A float decode would double it to pa[30], which this program does
    // not declare, and the census records that as one of its decisive cases.
    assert_eq!(i.srcs[0].bank, Bank::PrimaryAttr);
    assert_eq!(i.srcs[0].index, 15);
    // src2 is the inline immediate 31: the shift amount, not a register.
    assert_eq!(i.srcs[1].bank, Bank::Immediate);
    assert_eq!(i.srcs[1].index, 31);
    // The sibling word three instructions later is the same test on the complementary mask.
    let j = decode(0x48090881a00cc31f);
    assert!(j.blocked.is_none());
    assert_eq!(j.srcs[0].index, 6, "the NE mask's register");
}

/// A SHIFT whose amount is not provably in range stays REFUSED. WGSL leaves a shift of 32 or
/// more indeterminate and neither spec source says what the device does, so masking the amount
/// into range would be inventing a semantics - the one thing this decoder does not do.
///
/// Rewrites only the src2 fields of the real word, so what is being tested is the amount and
/// nothing else.
#[test]
fn a_shift_left_test_with_an_unprovable_amount_is_refused() {
    use vitaslop_gxp_shader::usse::decode::decode;
    const W: u64 = 0x48090881a00cc79f;
    // src2_n (bits 6:0) raised to 32 and to 127: both out of WGSL's defined range.
    for n in [32u64, 63, 127] {
        let w = (W & !0x7f) | n;
        assert!(
            decode(w).blocked.is_some(),
            "a shift by {n} must stay blocked - 32 or more is indeterminate"
        );
    }
    // src2_ext (bit 48) cleared makes src2 a REGISTER, whose value cannot be bounded here.
    assert!(
        decode(W & !(1 << 48)).blocked.is_some(),
        "a register shift amount must stay blocked"
    );
    // The real word, and every amount below 32, still translate - so the loop above is not
    // passing because the whole arm is refused.
    for n in [0u64, 1, 31] {
        let w = (W & !0x7f) | n;
        assert!(decode(w).blocked.is_none(), "a shift by {n} is in range");
    }
}

/// CENSUS: every TEST-group word in the BITWISE ALU family (`alu_sel` 3), with its operands
/// and the two words either side of it.
///
/// The family's `alu_op` numbering is the open question. `(3, 0)` is modelled as AND; the
/// corpus also carries `(3, 3)`, and nothing we hold states what 3 is. What decides it is not
/// a document but the IDIOM: these tests read a value the program itself just produced, so the
/// neighbours name what src1 holds, and an op that turns that value into a CONSTANT cannot be
/// the one the compiler emitted. Census only - it asserts nothing.
#[test]
#[ignore = "census over a corpus dir; run with VITASLOP_GXP_CORPUS=<dir>"]
fn bitwise_family_test_words_with_their_context() {
    use vitaslop_gxp_shader::usse::decode::decode;
    let Some(dir) = corpus_dir() else {
        println!("no VITASLOP_GXP_CORPUS set");
        return;
    };
    for (name, bytes) in blobs(&dir) {
        let Ok(p) = Program::parse(&bytes) else { continue };
        for (label, code) in [("primary", p.code.clone()), ("secondary", p.secondary_code.clone())] {
            for (i, &w) in code.iter().enumerate() {
                let g = ((w >> 59) & 0x1f) as u8;
                if g != 0x09 && g != 0x0f {
                    continue;
                }
                let b = |hi: u32, lo: u32| ((w >> lo) & ((1u64 << (hi - lo + 1)) - 1)) as u32;
                if b(19, 18) != 3 {
                    continue;
                }
                let bank = |s: u32| match s {
                    0 => "temp",
                    1 => "out/global",
                    2 => "pa",
                    _ => "sa",
                };
                println!(
                    "{name} {label}#{i} w={w:#018x} alu_op={} cmp(sign={},zero={},and={}) \
                     chan_cc={} pdst={} wben={} prec={}",
                    b(17, 14),
                    b(43, 42),
                    b(41, 40),
                    b(39, 39),
                    b(38, 36),
                    b(35, 34),
                    b(20, 20),
                    b(47, 47),
                );
                println!(
                    "    src1 {}[{}] ext={}  src2 {}[{}] ext={} (ext+bank2 = IMMEDIATE {})",
                    bank(b(31, 30)),
                    b(13, 7),
                    b(49, 49),
                    bank(b(29, 28)),
                    b(6, 0),
                    b(48, 48),
                    b(6, 0),
                );
                for j in i.saturating_sub(2)..(i + 3).min(code.len()) {
                    let mark = if j == i { ">>" } else { "  " };
                    println!("    {mark} #{j} {:?}", decode(code[j]));
                }
            }
        }
    }
}

/// Census of the 0x14 (I16MAD) group: every word, its bit variation, and the instructions
/// around it.
///
/// The decoder models exactly one word of this group (an index-register load fixed by
/// arithmetic closure) and hard-fails on any other, which is what blocks four vertex programs
/// of a shipped title. The group has no published layout, so the only way to widen it is to
/// see what the OTHER words are and what surrounds them - a group-0x14 instruction whose
/// result is consumed by an indexed read is doing the same job as the modeled one whatever
/// its fields say, and the context is what establishes that.
#[test]
#[ignore = "needs a captured corpus (game bytes); set VITASLOP_GXP_CORPUS"]
fn group_14_word_census() {
    let Some(dir) = corpus_dir() else { return };
    let mut hist: BTreeMap<u64, Vec<(String, usize)>> = BTreeMap::new();
    let mut bit_set = [0usize; 64];
    let mut total = 0usize;
    // Per occurrence, the surrounding raw words so the job of the instruction can be read
    // off its neighbours.
    let mut context: Vec<(String, usize, u64, Vec<u64>, Vec<u64>)> = Vec::new();
    for (name, bytes) in blobs(&dir) {
        let Ok(p) = Program::parse(&bytes) else { continue };
        for (which, code) in [("primary", &p.code), ("secondary", &p.secondary_code)] {
            for (i, w) in code.iter().enumerate() {
                if (w >> 59) & 0x1f != 0x14 {
                    continue;
                }
                total += 1;
                for b in 0..64 {
                    if (w >> b) & 1 != 0 {
                        bit_set[b] += 1;
                    }
                }
                let tag = format!("{name}/{which}");
                hist.entry(*w).or_default().push((tag.clone(), i));
                let lo = i.saturating_sub(4);
                context.push((
                    tag,
                    i,
                    *w,
                    code[lo..i].to_vec(),
                    code[i + 1..(i + 5).min(code.len())].to_vec(),
                ));
            }
        }
    }
    println!("\n-- 0x14 group: {total} words, {} distinct --", hist.len());
    for b in (0..64).rev() {
        let c = bit_set[b];
        if c != 0 && c != total {
            println!("  bit {b:>2}: {c:>4}/{total}  vary");
        }
    }
    println!("\ndistinct words:");
    for (w, uses) in &hist {
        let names: Vec<String> = uses.iter().take(8).map(|(n, i)| format!("{n}#{i}")).collect();
        println!("  {w:#018x}  x{:<4} {}", uses.len(), names.join(" "));
    }
    println!("\ncontext (4 before, 4 after; group in brackets):");
    for (name, i, w, before, after) in &context {
        println!("  {name} #{i}  {w:#018x}  [grp {:#04x}]", (w >> 59) & 0x1f);
        for (k, b) in before.iter().enumerate() {
            println!("    -{}  {b:#018x}  [grp {:#04x}]", before.len() - k, (b >> 59) & 0x1f);
        }
        for (k, a) in after.iter().enumerate() {
            println!("    +{}  {a:#018x}  [grp {:#04x}]", k + 1, (a >> 59) & 0x1f);
        }
    }
}

/// Print one blob's decoded USSE stream and its parameter table.
///
/// `VITASLOP_GXP_BLOB=vert_86473960` selects it. This is the reading step a blocked group
/// needs: the decoder refuses one instruction, and what settles what that instruction MEANS
/// is the arithmetic of the instructions around it against the container's own parameter
/// table - which is what this prints, side by side, with no recompile in the way.
#[test]
#[ignore = "needs a captured corpus (game bytes); set VITASLOP_GXP_CORPUS and VITASLOP_GXP_BLOB"]
fn print_one_blob_disassembly() {
    let Some(dir) = corpus_dir() else { return };
    let want = std::env::var("VITASLOP_GXP_BLOB").unwrap_or_default();
    for (name, bytes) in blobs(&dir) {
        if !blob_matches(&name, &bytes, &want) {
            continue;
        }
        let Ok(p) = Program::parse(&bytes) else { continue };
        println!("\n===== {name} =====");
        println!(
            "kind {:?}  pa_regs {}  sa_regs {}  temps {}",
            p.kind, p.primary_reg_count, p.secondary_reg_count, p.temp_reg_count
        );
        println!("-- containers (index, base_sa, size_in_f32) --");
        for c in &p.containers {
            println!("  {c:?}");
        }
        println!("  sa_base_from_container = {}", p.sa_base_from_container);
        // The SA tables, printed BY REGISTER. A register the secondary neither writes nor
        // finds a literal for reads zero in the emitted WGSL, and nothing else in this listing
        // says whether that zero is a table the blob does carry or a genuine gap - which is
        // exactly the question a shader multiplying its colour by an uninitialised lane raises.
        println!("-- literals (sa register = value) --");
        for (r, v) in &p.literals {
            let (lo, hi) = (f16_to_f32(*v as u16), f16_to_f32((*v >> 16) as u16));
            println!("  sa[{r:<3}] = {v:#010x}   f32 {:<14}  f16 pair ({lo}, {hi})", f32::from_bits(*v));
        }
        println!("-- texture control (base sa register, unit) --");
        for (base, unit) in &p.texture_control {
            println!("  sa[{base:<3}] unit {unit}");
        }
        println!("-- parameters --");
        for prm in &p.parameters {
            println!(
                "  {:<28} {:?} {:?} comps {} array {} container {} resource_index {} semantic {}/{}",
                prm.name,
                prm.category,
                prm.ptype,
                prm.component_count,
                prm.array_size,
                prm.container_index,
                prm.resource_index,
                prm.semantic,
                prm.semantic_index
            );
        }
        // The VARYING INTERFACE, both directions. A fragment reads its interpolants out of the
        // PA bank by REGISTER, and which vertex output feeds each register is decided by the
        // usage tables below - a listing without them shows a program multiplying pa[4] with no
        // way to say whether pa[4] is a colour or a texture coordinate.
        println!("-- fragment interpolants (usage, pa_base, regs, span, half, prefetch) --");
        for it in &p.interpolants {
            println!("  {it:?}");
        }
        println!("-- vertex output varyings (usage, base_lane, components) order {:?} --", p.output_order);
        for v in &p.output_varyings {
            println!("  {v:?}");
        }
        if let Some(why) = p.varyings_error {
            println!("  varyings_error: {why}");
        }
        println!("-- raw code words ({} primary, {} secondary) --", p.code.len(), p.secondary_code.len());
        for (i, w) in p.code.iter().enumerate() {
            println!("  code[{i:<4}] {w:#018x}  grp {:#04x}", (w >> 59) & 0x1f);
        }
        // The SECONDARY stream first, because that is where a program that chases a bound
        // buffer's pointer puts its memory loads - and a listing that shows only the primary
        // shows a program with no loads at all.
        let secondary = vitaslop_gxp_shader::usse::decode_secondary_shader(&p);
        println!("-- decoded SECONDARY stream ({} instrs) --", secondary.instrs.len());
        for (i, ins) in secondary.instrs.iter().enumerate() {
            println!(
                "  s#{i:<4} {:#018x} grp {:#04x}  {:?}  dest {:?} mask {:?} srcs {:?}{}",
                ins.raw, ins.group, ins.op, ins.dest, ins.write_mask, ins.srcs,
                match ins.blocked { Some(b) => format!("  BLOCKED: {b}"), None => String::new() }
            );
        }
        let shader = vitaslop_gxp_shader::usse::decode_shader(&p);
        println!("-- decoded stream ({} instrs) --", shader.instrs.len());
        for (i, ins) in shader.instrs.iter().enumerate() {
            println!(
                "  #{i:<4} {:#018x} grp {:#04x}  {:?}  dest {:?} mask {:?} srcs {:?} half {} pred {:?}{}",
                ins.raw,
                ins.group,
                ins.op,
                ins.dest,
                ins.write_mask,
                ins.srcs,
                ins.half_precision,
                ins.pred,
                match ins.blocked {
                    Some(b) => format!("  BLOCKED: {b}"),
                    None => String::new(),
                }
            );
        }
    }
}

/// Which SMLSI slot governs a repeating VPCK's DESTINATION, decided by which assignment keeps
/// every shipped instance inside the register file.
///
/// The source slot is settled (slot 1, by the halves-to-floats closure in `repeat_operands`).
/// The destination's is not, because in every instance the closure was fitted to, slots 0 and 1
/// carry the SAME increment - so those words cannot tell the two apart. A fourth title supplies
/// a VPCK whose slots differ in SIGN (`[-10, -10, +6, +6]`) and whose destination field is 0, so
/// under slot 0 its second iteration writes register -10. That is not a register, and a rule
/// that produces one is wrong somewhere.
///
/// This walks every repeating VPCK in the corpus and, for each candidate destination slot,
/// counts the iterations that leave the file. A slot that never leaves it on any shipped
/// instruction is the one the hardware uses; one that does cannot be.
#[test]
#[ignore = "needs a captured corpus (game bytes); set VITASLOP_GXP_CORPUS"]
fn vpck_repeat_destination_slot_census() {
    use vitaslop_gxp_shader::usse::decode::{decode_smlsi, is_smlsi, opcode1, SmlsiSlot};
    let Some(dir) = corpus_dir() else { return };
    let f = |w: u64, hi: u32, lo: u32| ((w >> lo) & ((1u64 << (hi - lo + 1)) - 1)) as u32;
    // Per candidate destination slot, how many iterations leave the 0..=255 register file.
    let mut escapes = [0usize; 4];
    let mut rows = 0usize;
    for (name, bytes) in blobs(&dir) {
        let Ok(p) = Program::parse(&bytes) else { continue };
        for code in [&p.code, &p.secondary_code] {
            let mut state = [1i32; 4];
            let mut swizzle = [false; 4];
            for (i, w) in code.iter().enumerate() {
                if is_smlsi(*w) {
                    for (k, s) in decode_smlsi(*w).iter().enumerate() {
                        match s {
                            SmlsiSlot::Increment(n) => {
                                state[k] = i32::from(*n);
                                swizzle[k] = false;
                            }
                            SmlsiSlot::Swizzle(_) => swizzle[k] = true,
                        }
                    }
                    continue;
                }
                if opcode1(*w) != 0x08 {
                    continue;
                }
                let extra = f(*w, 47, 44);
                if extra == 0 {
                    continue;
                }
                let dest_n = f(*w, 27, 21) as i32;
                let src_n = f(*w, 13, 8) as i32;
                rows += 1;
                println!(
                    "  {name}#{i} {w:#018x} extra {extra} dest_n {dest_n} src_n {src_n} \
                     smlsi {state:?} swizzle {swizzle:?}"
                );
                // >>> WHICH CANDIDATE'S SECOND DESTINATION IS ACTUALLY READ.
                //
                // A repeated instruction whose later iterations write registers nothing ever
                // reads is a compiler emitting dead work; a candidate under which those
                // registers ARE read afterwards is the one the program was written against.
                // The IR's decoded destination gives the bank and the iteration-0 index, and
                // the step is in the destination field's own units (stride 1 for this group),
                // so the candidate destination is just `index + slot_step * iteration`.
                {
                    let decoded = vitaslop_gxp_shader::usse::decode::decode(*w);
                    if let Some(d0) = decoded.dest {
                        // Every source register read by a LATER instruction of this stream,
                        // in the same bank. Repeats are not unrolled here - the question is
                        // which registers the program mentions at all.
                        // LIVE, not merely mentioned: a register that is overwritten before it
                        // is read is dead however often the program names it later. The scan
                        // stops at the first instruction that writes the candidate.
                        let live = |idx: u8| -> bool {
                            for later in &code[i + 1..] {
                                let ins = vitaslop_gxp_shader::usse::decode::decode(*later);
                                if ins.srcs.iter().any(|s| s.bank == d0.bank && s.index == idx) {
                                    return true;
                                }
                                if ins.dest.is_some_and(|d| d.bank == d0.bank && d.index == idx) {
                                    return false;
                                }
                            }
                            false
                        };
                        let marks: Vec<String> = (0..4)
                            .map(|slot| {
                                let d = i32::from(d0.index) + state[slot] * extra as i32;
                                match u8::try_from(d) {
                                    Ok(idx) if live(idx) => format!("slot{slot}:{idx} LIVE"),
                                    Ok(idx) => format!("slot{slot}:{idx} -"),
                                    Err(_) => format!("slot{slot}:{d} OUT"),
                                }
                            })
                            .collect();
                        println!("      dest bank {:?} base {} -> {}", d0.bank, d0.index, marks.join("  "));
                    }
                }
                for slot in 0..4 {
                    for it in 1..=extra as i32 {
                        // The destination field is SEVEN bits (stride 1); the source is SIX
                        // (stride 2). Both are counted in their own field's units here.
                        let d = dest_n + state[slot] * it;
                        if !(0..128).contains(&d) {
                            escapes[slot] += 1;
                        }
                    }
                    let s = src_n + state[1] * extra as i32;
                    if !(0..64).contains(&s) {
                        println!("    (source slot 1 escapes: {s})");
                    }
                }
            }
        }
    }
    println!("\n-- {rows} repeating VPCKs; iterations leaving the register file per candidate destination slot --");
    for (slot, n) in escapes.iter().enumerate() {
        println!("  slot {slot}: {n}");
    }
}

/// Every blob whose LINK resolves a guest-memory window must also get one from the
/// draw-time scan, and the other way round.
///
/// These are two different functions with two different jobs - one runs at link time over a
/// decoded shader, the other once per registered program over raw bytes - and the renderer
/// requires them to agree exactly: a pipeline built with N windows drops any draw that arrives
/// with a different number, because feeding a memory load zeroes would paint a wrong picture
/// with nothing to say so. They disagreed, and the disagreement was invisible in every corpus
/// test until a title put its load in the SECONDARY stream: the link found it, the draw-time
/// scan looked only at the primary, and every world draw of that title was dropped.
#[test]
#[ignore = "needs a captured corpus (game bytes); set VITASLOP_GXP_CORPUS"]
fn the_draw_time_window_scan_agrees_with_the_link() {
    let Some(dir) = corpus_dir() else { return };
    let mut checked = 0usize;
    let mut with_windows = 0usize;
    for (name, bytes) in blobs(&dir) {
        let Ok(p) = Program::parse(&bytes) else { continue };
        if p.kind != vitaslop_gxp_shader::container::ProgramKind::Vertex {
            continue;
        }
        checked += 1;
        let shader = vitaslop_gxp_shader::usse::decode_shader(&p);
        let linked = vitaslop_gxp_shader::module::resolve_mem_windows(&p, &shader)
            .unwrap_or_default();
        let at_draw = vitaslop_gxp_shader::mem_windows_for_vertex_blob(&bytes);
        assert_eq!(
            linked.len(),
            at_draw.len(),
            "{name}: the link resolves {} window(s) and the draw-time scan finds {} - a draw \
             carrying the second number is DROPPED by a pipeline built from the first",
            linked.len(),
            at_draw.len()
        );
        if !linked.is_empty() {
            with_windows += 1;
            assert_eq!(linked, at_draw, "{name}: same count, different windows");
        }
    }
    println!("{checked} vertex blobs, {with_windows} of them with guest-memory windows");
    assert!(checked > 0, "no vertex blobs in the corpus");
}

/// Every vertex blob's guest-memory WINDOWS beside the uniform buffers its parameter table
/// declares, so a window that is too small to hold what the program reads through it is
/// visible without a run.
///
/// The window's byte count is what the emitted `gxp_mem_word` bounds-checks against, and an
/// address past it reads ZERO - silently, in the shader. A title whose world programs read a
/// bone matrix by index will miss most of its reads if the count is short, and the frame comes
/// out empty with nothing anywhere saying why.
#[test]
#[ignore = "needs a captured corpus (game bytes); set VITASLOP_GXP_CORPUS"]
fn tabulate_memory_windows_against_the_declared_buffers() {
    let Some(dir) = corpus_dir() else { return };
    for (name, bytes) in blobs(&dir) {
        let Ok(p) = Program::parse(&bytes) else { continue };
        if p.kind != vitaslop_gxp_shader::container::ProgramKind::Vertex {
            continue;
        }
        let shader = vitaslop_gxp_shader::usse::decode_shader(&p);
        let windows =
            vitaslop_gxp_shader::module::resolve_mem_windows(&p, &shader).unwrap_or_default();
        if windows.is_empty() {
            continue;
        }
        println!("\n{name}: temp_regs {} default_uniform_regs {}", p.secondary_reg_count, p.default_uniform_regs);
        for c in &p.containers {
            println!("  container {:>2} base_sa {:>3} size_regs {}", c.index, c.base_sa, c.size_regs);
        }
        for prm in &p.parameters {
            if matches!(prm.category, vitaslop_gxp_shader::container::ParamCategory::UniformBuffer) {
                println!(
                    "  UniformBuffer parameter: container {} resource_index {} comps {} array_size {}",
                    prm.container_index, prm.resource_index, prm.component_count, prm.array_size
                );
            }
        }
        for w in &windows {
            println!(
                "  WINDOW buffer_index {} bytes {} base_sa {} base_offset {}",
                w.buffer_index, w.bytes, w.base_sa, w.base_offset
            );
        }
    }
}

/// Every group-0xF8 LIMM in the corpus, with its field breakdown and its neighbourhood.
///
/// LIMM is the last blocked member of the flow group, and it blocks a shipped skinning vertex
/// program, so it stops a real draw. Its published field list cannot be read literally - the
/// reference puts `imm[31:26]` at bits 57:52, which is where `op2` (58:56) and `opcat` (53:52)
/// select the member, and those bits are what identify the instruction AS a LIMM. So the
/// encoding has to come off the words themselves, and this is the inventory that makes that
/// possible: what varies between two LIMMs in one program is the immediate and the destination,
/// and what does not vary is the member selector.
#[test]
#[ignore = "needs a captured corpus (game bytes); set VITASLOP_GXP_CORPUS"]
fn tabulate_limm_words_and_their_neighbourhood() {
    use vitaslop_gxp_shader::usse::{bits, decode, opcode1};
    let Some(dir) = corpus_dir() else { return };
    let is_limm = |w: u64| {
        opcode1(w) == 0x1f && bits(w, 58, 56) == 0b100 && bits(w, 53, 52) == 0b10
    };
    let mut total = 0usize;
    for (name, bytes) in blobs(&dir) {
        let Ok(p) = Program::parse(&bytes) else { continue };
        for (stream, code) in [("primary", &p.code), ("secondary", &p.secondary_code)] {
            for (i, &w) in code.iter().enumerate() {
                if !is_limm(w) {
                    continue;
                }
                total += 1;
                println!(
                    "\n{name} {stream} #{i}  {w:#018x}\n  \
                     [63:59]={:#07b} op2[58:56]={:#05b} b55={} extra[54]={} opcat[53:52]={:#04b} \
                     [51:48]={:#06b} [47:44]={:#06b} [43:40]={:#06b} [39:36]={:#06b} \
                     [35:32]={:#06b} [31:0]={:#010x}",
                    bits(w, 63, 59),
                    bits(w, 58, 56),
                    bits(w, 55, 55),
                    bits(w, 54, 54),
                    bits(w, 53, 52),
                    bits(w, 51, 48),
                    bits(w, 47, 44),
                    bits(w, 43, 40),
                    bits(w, 39, 36),
                    bits(w, 35, 32),
                    bits(w, 31, 0),
                );
                for j in i.saturating_sub(8)..(i + 10).min(code.len()) {
                    let marker = if j == i { ">>" } else { "  " };
                    let d = decode(code[j]);
                    println!(
                        "  {marker} #{j:<4} {:#018x} grp {:#04x} {:?} dest {:?} mask {:?} srcs {:?}",
                        code[j],
                        opcode1(code[j]),
                        d.op,
                        d.dest.map(|o| (o.bank, o.index)),
                        d.write_mask,
                        d.srcs.iter().map(|o| (o.bank, o.index, o.swizzle)).collect::<Vec<_>>(),
                    );
                }
            }
        }
    }
    println!("\n{total} LIMM word(s) in this corpus");
}

/// For every program that contains a LIMM, the writable-bank registers it READS before
/// anything writes them.
///
/// This is how the LIMM's destination is established without reading its disputed field list:
/// a Temp / Output / Internal register a program reads but never writes has to be written by
/// the one instruction the decoder cannot yet model, because those banks hold nothing on entry
/// (attributes and uniforms arrive in PrimaryAttr / SecondaryAttr, which are therefore excluded
/// here). If exactly one such register appears, the LIMM's destination is not a guess.
#[test]
#[ignore = "needs a captured corpus (game bytes); set VITASLOP_GXP_CORPUS"]
fn registers_read_before_written_in_programs_carrying_a_limm() {
    use vitaslop_gxp_shader::usse::{bits, decode, opcode1};
    let Some(dir) = corpus_dir() else { return };
    let is_limm =
        |w: u64| opcode1(w) == 0x1f && bits(w, 58, 56) == 0b100 && bits(w, 53, 52) == 0b10;
    // `Bank` is not ordered, and only the three banks that hold nothing on entry matter here.
    let writable = |b: Bank| match b {
        Bank::Temp => Some(0u8),
        Bank::Output => Some(1),
        Bank::Internal => Some(2),
        _ => None,
    };
    let bank_name = |k: u8| ["Temp", "Output", "Internal"][k as usize];
    for (name, bytes) in blobs(&dir) {
        let Ok(p) = Program::parse(&bytes) else { continue };
        if !p.code.iter().chain(p.secondary_code.iter()).any(|&w| is_limm(w)) {
            continue;
        }
        // The SECONDARY runs first and leaves its results in the register file, so a register it
        // writes is not "unwritten" when the primary reads it.
        let mut written: std::collections::BTreeSet<(u8, u8)> = Default::default();
        let mut first_read: BTreeMap<(u8, u8), String> = BTreeMap::new();
        for (stream, code) in [("secondary", &p.secondary_code), ("primary", &p.code)] {
            for (i, &w) in code.iter().enumerate() {
                let d = decode(w);
                for s in &d.srcs {
                    if let Some(k) = writable(s.bank)
                        && !written.contains(&(k, s.index)) {
                            first_read.entry((k, s.index)).or_insert(format!("{stream} #{i}"));
                        }
                }
                if let Some(dst) = d.dest
                    && let Some(k) = writable(dst.bank) {
                        written.insert((k, dst.index));
                    }
            }
        }
        let limms: Vec<String> = p
            .code
            .iter()
            .enumerate()
            .filter(|&(_, &w)| is_limm(w))
            .map(|(i, &w)| format!("#{i} {w:#018x}"))
            .collect();
        println!("
{name}: LIMMs at {}", limms.join(", "));
        if first_read.is_empty() {
            println!("  no writable-bank register is read before it is written");
        }
        for ((k, idx), at) in &first_read {
            println!("  READ-BEFORE-WRITE  {}[{idx}]  first read at {at}", bank_name(*k));
        }
    }
}

/// Every repeating 16-bit PACK and the group-0x15 IMAD32s that read what it wrote.
///
/// This is the closure that decides the repeating pack's destination MULTIPLIER for a 16-bit
/// destination, and it is a whole-program argument rather than a fit. A skinned mesh converts
/// its four float blend indices to integers with ONE `PackToInt` repeated twice over a
/// two-channel mask, then multiplies each by the bone-matrix stride. So the four IMAD32s that
/// follow must between them name FOUR DISTINCT index values - a mesh does not blend one bone
/// against itself - and the pack must have put four distinct values where they look.
///
/// The two readings differ exactly here:
///   * ONE VALUE PER REGISTER, multiplier 2: iteration k writes `dest+2k` and `dest+2k+1`.
///   * TWO VALUES PER REGISTER (packed halves), multiplier 1: iteration k writes the two
///     HALVES of `dest+k`, and `src0_high` (bit 56) is what picks one back out.
/// `src0_high` is the discriminator: a corpus where the consumers never set it cannot tell the
/// readings apart, and one where they do can only be explained by the packed form.
#[test]
#[ignore = "needs a captured corpus (game bytes); set VITASLOP_GXP_CORPUS"]
fn repeating_16bit_packs_and_the_imads_that_read_them() {
    use vitaslop_gxp_shader::usse::{bits, decode, opcode1, repeat_extra_iterations};
    let Some(dir) = corpus_dir() else { return };
    for (name, bytes) in blobs(&dir) {
        let Ok(p) = Program::parse(&bytes) else { continue };
        for (stream, code) in [("primary", &p.code), ("secondary", &p.secondary_code)] {
            // A repeating group-0x08 pack whose DESTINATION format is one of the 16-bit ones
            // (3 = U16, 4 = S16, 5 = F16).
            let packs: Vec<usize> = code
                .iter()
                .enumerate()
                .filter(|&(_, &w)| {
                    opcode1(w) == 0x08
                        && !matches!(repeat_extra_iterations(w), Some(0) | None)
                        && matches!(bits(w, 40, 38), 3..=5)
                })
                .map(|(i, _)| i)
                .collect();
            if packs.is_empty() {
                continue;
            }
            for i in packs {
                let w = code[i];
                let d = decode(w);
                let extra = repeat_extra_iterations(w).unwrap_or(0);
                let lanes = d.write_mask.iter().filter(|m| **m).count();
                let base = d.dest.map_or(255u8, |o| o.index);
                println!(
                    "\n{name} {stream} #{i} {w:#018x}: PACK dest_fmt {} dest {:?}[{base}] \
                     mask {lanes} lane(s), runs {} time(s)\n  \
                     one-per-register (x2) writes {:?}\n  packed halves (x1) writes {:?}",
                    bits(w, 40, 38),
                    d.dest.map(|o| o.bank),
                    extra + 1,
                    (0..=extra)
                        .flat_map(|k| (0..lanes as u32).map(move |l| base as u32 + 2 * k + l))
                        .collect::<Vec<_>>(),
                    (0..=extra)
                        .flat_map(|k| (0..lanes as u32)
                            .map(move |l| format!("{}.{}", base as u32 + k, if l == 0 { "lo" } else { "hi" })))
                        .collect::<Vec<_>>(),
                );
                // Every IMAD32 in the same stream, with the source register and half it names.
                for (j, &iw) in code.iter().enumerate() {
                    if opcode1(iw) != 0x15 {
                        continue;
                    }
                    println!(
                        "    #{j:<4} IMAD32 src0 = {}[{}] half {} (bit56={}), src1 {:?}, dest {:?}",
                        if bits(iw, 34, 34) == 0 { "Temp" } else { "PrimaryAttr" },
                        bits(iw, 20, 14),
                        if bits(iw, 56, 56) == 1 { "HIGH" } else { "low" },
                        bits(iw, 56, 56),
                        decode(iw).srcs.get(1).map(|o| (o.bank, o.index)),
                        decode(iw).dest.map(|o| (o.bank, o.index)),
                    );
                }
            }
        }
    }
}

/// Every group-0x15 IMAD32 in the corpus, and whether its `src0` register is one a 16-bit
/// PACK wrote. `src0_high` (bit 56) only means "the high half of a packed pair" if src0 is
/// always such a pair; a program whose src0 is a full 32-bit value would refute that.
#[test]
#[ignore = "needs a captured corpus (game bytes); set VITASLOP_GXP_CORPUS"]
fn imad32_sources_against_the_packs_that_feed_them() {
    use vitaslop_gxp_shader::usse::{bits, opcode1};
    let Some(dir) = corpus_dir() else { return };
    let (mut total, mut fed, mut high) = (0usize, 0usize, 0usize);
    for (name, bytes) in blobs(&dir) {
        let Ok(p) = Program::parse(&bytes) else { continue };
        for code in [&p.code, &p.secondary_code] {
            // Registers a 16-bit pack writes, under the PACKED-HALVES reading: one register
            // per iteration starting at the destination.
            let mut packed: std::collections::BTreeSet<(u8, u32)> = Default::default();
            for &w in code.iter() {
                if opcode1(w) != 0x08 || !matches!(bits(w, 40, 38), 3..=5) {
                    continue;
                }
                let d = vitaslop_gxp_shader::usse::decode(w);
                let Some(dst) = d.dest else { continue };
                let iters = vitaslop_gxp_shader::usse::repeat_extra_iterations(w).unwrap_or(0);
                let bank = if matches!(dst.bank, Bank::Temp) { 0u8 } else { 1 };
                for k in 0..=iters {
                    packed.insert((bank, dst.index as u32 + k));
                }
            }
            for &w in code.iter() {
                if opcode1(w) != 0x15 {
                    continue;
                }
                total += 1;
                if bits(w, 56, 56) == 1 {
                    high += 1;
                }
                let bank = if bits(w, 34, 34) == 0 { 0u8 } else { 1 };
                if packed.contains(&(bank, bits(w, 20, 14))) {
                    fed += 1;
                } else {
                    println!(
                        "  {name}: IMAD32 {w:#018x} src0 = {}[{}] half {} is NOT written by any \
                         16-bit pack in its stream",
                        if bank == 0 { "Temp" } else { "PrimaryAttr" },
                        bits(w, 20, 14),
                        bits(w, 56, 56),
                    );
                }
            }
        }
    }
    println!("IMAD32: {total} total, {fed} whose src0 a 16-bit pack wrote, {high} with src0_high set");
}

/// Programs whose parameter table names a PARTICLE input, and the group-0x14 LoadIndex
/// instructions they carry with the addend each decodes.
///
/// A particle vertex program expands one point into a quad by adding four corner offsets, so a
/// wrong offset does not make a small error - it makes a screen-sized quad in the particle's
/// own colour, appearing and vanishing with the particle. That is the shape of the stray solid
/// polygons, so this lists the programs that could produce one and the field that decides it.
#[test]
#[ignore = "needs a captured corpus (game bytes); set VITASLOP_GXP_CORPUS"]
fn particle_programs_and_their_load_index_addends() {
    use vitaslop_gxp_shader::usse::{bits, decode, opcode1};
    let Some(dir) = corpus_dir() else { return };
    for (name, bytes) in blobs(&dir) {
        let Ok(p) = Program::parse(&bytes) else { continue };
        let names: Vec<String> = p.parameters.iter().map(|q| q.name.clone()).collect();
        let particle = names.iter().any(|n| {
            let l = n.to_ascii_lowercase();
            l.contains("subuv") || l.contains("particle") || l.contains("corner")
                || l.contains("sprite")
        });
        let idx: Vec<String> = p
            .code
            .iter()
            .chain(p.secondary_code.iter())
            .enumerate()
            .filter(|&(_, &w)| matches!(decode(w).op, Op::LoadIndex { .. }) || opcode1(w) == 0x02)
            .map(|(i, &w)| format!("#{i} {:?} raw={w:#018x} [6:0]={}", decode(w).op, bits(w, 6, 0)))
            .collect();
        if !particle {
            continue;
        }
        println!("\n{name}  <-- PARTICLE parameters");
        println!("  params: {}", names.join(", "));
        for l in idx {
            println!("    {l}");
        }
    }
}

/// How many of a corpus's FRAGMENT programs still read the destination colour under each
/// lowering setting - which is exactly how many render-pass SPLITS a frame drawing each of
/// them once would pay.
///
/// A split ends the pass, copies the whole attachment and begins a new one with `LoadOp::Load`,
/// so on a tiling GPU it is a full store-and-reload of the framebuffer's tiles. The phone
/// measured 54 of them in a single frame. Every program the lowering recognises is one split
/// that never happens, so the question "what would turning the third shape on buy" has an exact
/// offline answer, and this is it.
#[test]
#[ignore = "needs a captured corpus (game bytes); set VITASLOP_GXP_CORPUS"]
fn fragment_programs_that_still_read_the_destination_under_each_lowering() {
    use vitaslop_gxp_shader::module;
    let Some(dir) = corpus_dir() else { return };
    let all = blobs(&dir);
    let mut rows: Vec<(String, bool, bool, bool)> = Vec::new();
    for (name, bytes) in &all {
        let Ok(p) = Program::parse(bytes) else { continue };
        if p.kind != ProgramKind::Fragment {
            continue;
        }
        let reads = |forms: u32| {
            module::set_dest_blend_lowering(forms);
            vitaslop_gxp_shader::fragment_reads_dest_color(bytes)
        };
        let off = reads(0);
        let default = reads(module::FORM_DEFAULT);
        let all_forms = reads(module::FORM_ALL);
        if off {
            rows.push((name.clone(), off, default, all_forms));
        }
    }
    module::set_dest_blend_lowering(module::FORM_DEFAULT);
    let n_off = rows.len();
    let n_default = rows.iter().filter(|r| r.2).count();
    let n_all = rows.iter().filter(|r| r.3).count();
    println!(
        "\nfragment programs READING THE DESTINATION (each one a render-pass split per draw):\n  \
         lowering OFF      {n_off}\n  lowering DEFAULT  {n_default} (lerp + modulate)\n  \
         lowering ALL      {n_all} (+ additive)"
    );
    for (name, _, d, a) in &rows {
        if *d && !*a {
            println!("  the ADDITIVE shape would lower: {name}");
        }
    }
    for (name, _, d, a) in &rows {
        if *d && *a {
            println!("  still reads the destination under every shape: {name}");
        }
    }
}

/// The fragment body of one blob under each dest-blend lowering setting, so a shape that is
/// about to be turned on can be READ before it is believed.
#[test]
#[ignore = "needs a captured corpus (game bytes); set VITASLOP_GXP_CORPUS and VITASLOP_GXP_BLOB"]
fn fragment_body_under_each_dest_blend_lowering() {
    use vitaslop_gxp_shader::module;
    let (Some(dir), Ok(want)) = (corpus_dir(), std::env::var("VITASLOP_GXP_BLOB")) else { return };
    for (name, bytes) in blobs(&dir) {
        if !blob_matches(&name, &bytes, &want) {
            continue;
        }
        for (label, forms) in [("OFF", 0), ("DEFAULT", module::FORM_DEFAULT), ("ALL", module::FORM_ALL)] {
            module::set_dest_blend_lowering(forms);
            match recompile_fragment(&bytes) {
                Ok(r) => println!(
                    "\n=== {name} lowering {label}: reads_dest={} blend={:?}\n{}",
                    vitaslop_gxp_shader::fragment_reads_dest_color(&bytes),
                    r.dest_blend,
                    r.wgsl_body
                ),
                Err(e) => println!("\n=== {name} lowering {label}: recompile failed: {e:?}"),
            }
        }
        module::set_dest_blend_lowering(module::FORM_DEFAULT);
    }
}

/// Whether the LIMM-carrying program's two words can be told apart field by field.
///
/// `0xF8 LIMM` is the last blocked member of the flow group and it stops a shipped skinning
/// pair mid-fight. Its published field list cannot be read literally - the reference puts
/// `imm[31:26]` at bits 57:52, which is where `op2` (58:56) and `opcat` (53:52) select the
/// member - so the encoding has to come off the words. This prints, for every LIMM in the
/// corpus, which bits VARY between them: a field that never varies cannot be separated from a
/// constant, and with only two words most of the layout is exactly that. It is the record of
/// what the evidence can and cannot decide, so the next corpus that carries a third LIMM can
/// be pointed straight at it.
#[test]
#[ignore = "needs a captured corpus (game bytes); set VITASLOP_GXP_CORPUS"]
fn which_bits_separate_the_limm_words() {
    use vitaslop_gxp_shader::usse::{bits, opcode1};
    let Some(dir) = corpus_dir() else { return };
    let is_limm =
        |w: u64| opcode1(w) == 0x1f && bits(w, 58, 56) == 0b100 && bits(w, 53, 52) == 0b10;
    let mut words: Vec<(String, usize, u64)> = Vec::new();
    for (name, bytes) in blobs(&dir) {
        let Ok(p) = Program::parse(&bytes) else { continue };
        for (i, &w) in p.code.iter().chain(p.secondary_code.iter()).enumerate() {
            if is_limm(w) {
                words.push((name.clone(), i, w));
            }
        }
    }
    println!("\n{} LIMM word(s):", words.len());
    for (n, i, w) in &words {
        println!("  {n} #{i} {w:#018x}");
    }
    if words.len() < 2 {
        println!("  fewer than two - nothing can be separated");
        return;
    }
    let (mut varies, mut fixed) = (Vec::new(), Vec::new());
    for b in (0..64).rev() {
        let v: Vec<u32> = words.iter().map(|(_, _, w)| bits(*w, b, b)).collect();
        if v.windows(2).any(|p| p[0] != p[1]) { varies.push(b) } else { fixed.push((b, v[0])) }
    }
    println!("\n  bits that VARY (the only ones any field can be read from): {varies:?}");
    println!(
        "  bits that are FIXED across every LIMM (a field here is indistinguishable from a \
         constant): {}",
        fixed.iter().map(|(b, v)| format!("{b}={v}")).collect::<Vec<_>>().join(" ")
    );
}

/// Programs whose parameters name a VELOCITY or MOTION-BLUR input.
///
/// A velocity pass writes a 2D motion vector into a render target AS COLOUR, so if one of its
/// draws reaches the display target the frame gets a smooth two-channel gradient over the
/// moving geometry - flat green with blue running across it, in polygon-shaped patches. That is
/// exactly the shape of this title's stray coloured masses, so this is the list of programs
/// that could produce one.
#[test]
#[ignore = "needs a captured corpus (game bytes); set VITASLOP_GXP_CORPUS"]
fn programs_that_compute_a_velocity() {
    let Some(dir) = corpus_dir() else { return };
    for (name, bytes) in blobs(&dir) {
        let Ok(p) = Program::parse(&bytes) else { continue };
        let names: Vec<String> = p.parameters.iter().map(|q| q.name.clone()).collect();
        if !names.iter().any(|n| {
            let l = n.to_ascii_lowercase();
            l.contains("velocity") || l.contains("blur") || l.contains("previous") || l.contains("motion")
        }) {
            continue;
        }
        println!("{name} ({:?}): {}", p.kind, names.join(", "));
    }
}

/// Which of this session's decoder changes can reach each corpus at all: programs carrying a
/// 16-bit PACK (whose destination is now half-granular), a group-0x15 IMAD32 (whose `src0` is
/// now a half), and an INDEXED read of the SA bank (which now pulls in every container
/// literal). A corpus with none of them cannot have moved, so a picture that did move names
/// the change that moved it.
#[test]
#[ignore = "needs a captured corpus (game bytes); set VITASLOP_GXP_CORPUS"]
fn which_programs_this_sessions_changes_can_reach() {
    use vitaslop_gxp_shader::usse::{bits, decode, opcode1};
    let Some(dir) = corpus_dir() else { return };
    let (mut pack16, mut imad, mut indexed, mut total) = (0, 0, 0, 0);
    let mut indexed_names = Vec::new();
    for (name, bytes) in blobs(&dir) {
        let Ok(p) = Program::parse(&bytes) else { continue };
        total += 1;
        let words: Vec<u64> = p.code.iter().chain(p.secondary_code.iter()).copied().collect();
        if words.iter().any(|&w| opcode1(w) == 0x08 && matches!(bits(w, 40, 38), 3 | 4)) {
            pack16 += 1;
        }
        if words.iter().any(|&w| opcode1(w) == 0x15) {
            imad += 1;
        }
        if words.iter().any(|&w| {
            decode(w).srcs.iter().any(|s| {
                s.bank == Bank::Indexed
                    && vitaslop_gxp_shader::ir::indexed_sub_bank(s.index) == Bank::SecondaryAttr
            })
        }) {
            indexed += 1;
            indexed_names.push(format!("{name} ({} literals)", p.literals.len()));
        }
    }
    println!(
        "\nof {total} blobs: {pack16} carry a 16-bit PACK, {imad} an IMAD32, {indexed} an \
         INDEXED SA read"
    );
    for n in &indexed_names {
        println!("  indexed SA read: {n}");
    }
}

/// What [`vitaslop_gxp_shader::attrflow`] decides for every attribute of every pair that links,
/// so the per-lane fill can be read off a corpus instead of a device.
///
/// One line per attribute whose analysis says ZERO for any lane, plus a census, because the
/// interesting answer is the CHANGE from the renderer's standing 1.0 - an attribute the walk
/// leaves at identity is the old behaviour and says nothing new.
///
/// `VITASLOP_GXP_ATTR_ALL=1` prints every attribute instead, which is what to use when the
/// question is "why is THIS one still 1.0".
#[test]
#[ignore = "needs a captured corpus (game bytes); set VITASLOP_GXP_CORPUS"]
fn tabulate_surplus_attribute_fills() {
    use vitaslop_gxp_shader::attrflow::Fill;
    let Some(dir) = corpus_dir() else {
        eprintln!("VITASLOP_GXP_CORPUS not set - nothing to analyse");
        return;
    };
    let all = blobs(&dir);
    let verts: Vec<_> = all
        .iter()
        .filter(|(_, b)| Program::parse(b).map(|p| p.kind == ProgramKind::Vertex).unwrap_or(false))
        .collect();
    let frags: Vec<_> = all
        .iter()
        .filter(|(_, b)| Program::parse(b).map(|p| p.kind == ProgramKind::Fragment).unwrap_or(false))
        .collect();
    let show_all = std::env::var_os("VITASLOP_GXP_ATTR_ALL").is_some();

    let (mut pairs, mut attrs, mut zeroed) = (0usize, 0usize, 0usize);
    for (vname, vbytes) in &verts {
        for (fname, fbytes) in &frags {
            let Ok(linked) = link_programs(vbytes, fbytes) else { continue };
            pairs += 1;
            for a in &linked.vertex_bindings.attributes {
                attrs += 1;
                let any_zero = a.surplus_fill.contains(&Fill::Zero);
                if any_zero {
                    zeroed += 1;
                }
                if !any_zero && !show_all {
                    continue;
                }
                let lanes: Vec<String> = (0..4)
                    .map(|c| format!("{}={}", ["x", "y", "z", "w"][c], a.surplus_fill[c].value()))
                    .collect();
                println!(
                    "{vname} + {fname}: @location {} {} ({} declared components, base lane {}) -> {}",
                    a.location,
                    a.name,
                    a.components,
                    a.base_lane,
                    lanes.join(" ")
                );
            }
        }
    }
    println!("\n{pairs} linked pairs, {attrs} attributes, {zeroed} with at least one ZERO lane");
}

/// The complete LINKED WGSL for one pair, selected by `VITASLOP_GXP_VERT` + `VITASLOP_GXP_FRAG`
/// (file stems or content hashes, the same selector `print_one_blob` takes).
///
/// The live renderer can write this with `VITASLOP_GXP_WGSL_DIR`, but that costs a replay to the
/// frame that binds the pair - minutes for a pair only a menu screen reaches. Two blob names and
/// a second is the same answer.
#[test]
#[ignore = "needs a captured corpus (game bytes); set VITASLOP_GXP_CORPUS"]
fn print_linked_pair_wgsl() {
    let Some(dir) = corpus_dir() else { return };
    let (want_v, want_f) = (
        std::env::var("VITASLOP_GXP_VERT").unwrap_or_default(),
        std::env::var("VITASLOP_GXP_FRAG").unwrap_or_default(),
    );
    if want_v.is_empty() || want_f.is_empty() {
        eprintln!("set VITASLOP_GXP_VERT and VITASLOP_GXP_FRAG");
        return;
    }
    let all = blobs(&dir);
    let find = |want: &str| {
        all.iter().find(|(n, b)| blob_matches(n, b, want.trim())).map(|(n, b)| (n.clone(), b.clone()))
    };
    let (Some((vn, vb)), Some((fnm, fb))) = (find(&want_v), find(&want_f)) else {
        eprintln!("one of the two blobs is not in this corpus");
        return;
    };
    println!("===== {vn} + {fnm} =====");
    if let Ok(vp) = Program::parse(&vb) {
        println!("// vertex output order {:?}", vp.output_order);
        // WHICH OUTPUT LANES THE CODE ACTUALLY WRITES. The declared layout says where each
        // varying SITS; only this says which of those lanes carry a value at all, and a
        // fragment read that lands on a lane outside this set shades from whatever the
        // previous draw left. Printing them side by side is what makes a layout question
        // decidable by looking rather than by reasoning.
        let vsh = vitaslop_gxp_shader::usse::decode_shader(&vp);
        let mut w = [false; 64];
        for i in &vsh.instrs {
            let Some(d) = i.dest.as_ref() else { continue };
            if format!("{:?}", d.bank) != "Output" {
                continue;
            }
            for c in 0..4 {
                if i.write_mask[c] && (d.index as usize + c) < w.len() {
                    w[d.index as usize + c] = true;
                }
            }
        }
        let written: Vec<String> = (0..40).map(|l| if w[l] { "#".into() } else { ".".to_string() }).collect();
        println!("//   vertex WRITES lanes 0..40: {}", written.join(""));
        for v in &vp.output_varyings {
            println!("//   vertex output {:?} lanes {}..{}", v.usage, v.base_lane, v.base_lane + v.components);
        }
    }
    if let Ok(fp) = Program::parse(&fb) {
        for it in &fp.interpolants {
            println!("//   fragment interpolant {:?}", it);
        }
    }
    match link_programs(&vb, &fb) {
        Ok(l) => {
            for a in &l.vertex_bindings.attributes {
                println!(
                    "// attribute @location {} {} components {} base lane {} fill {:?}",
                    a.location, a.name, a.components, a.base_lane, a.surplus_fill
                );
            }
            println!("// dest_blend {:?}  reads_dest_color {}", l.dest_blend, l.reads_dest_color);
            println!("{}", l.wgsl);
        }
        Err(e) => println!("LINK FAILED: {e:?}"),
    }
}

/// Which VERTEX programs the forwarding resolver's lane RESERVATION moves, and where to.
///
/// The reservation (`layout_from_forwarding_claims`, and its `VITASLOP_GXP_VARYING_RESOLVE=
/// noreserve` arm) changes which varying sits in which lane, so it can change any title's
/// picture. This names the programs it can reach BEFORE a render run does, which is what turns
/// "check every title" into "check the titles that have one".
///
/// Run it twice, once per arm, and diff: a corpus whose output is identical under both cannot
/// have regressed, whatever its frames look like.
#[test]
#[ignore = "needs a captured corpus (game bytes); set VITASLOP_GXP_CORPUS"]
fn tabulate_resolved_varying_layouts() {
    let Some(dir) = corpus_dir() else { return };
    let all = blobs(&dir);
    let frags: Vec<_> = all
        .iter()
        .filter(|(_, b)| Program::parse(b).map(|p| p.kind == ProgramKind::Fragment).unwrap_or(false))
        .collect();
    let mut lines: BTreeMap<String, String> = BTreeMap::new();
    for (vn, vb) in &all {
        let Ok(vp) = Program::parse(vb) else { continue };
        if vp.kind != ProgramKind::Vertex {
            continue;
        }
        // A layout is a property of the vertex program, but only a LINK exposes it - so pair it
        // with the first fragment it links against and read the vertex side.
        // The pairing must be one that actually WIRES varyings: a fragment that declares none
        // renders the layout invisible, and picking the first link that happens to be one of
        // those reports "nothing changed" for a program whose layout moved.
        let Some(l) = frags.iter().filter_map(|(_, fb)| link_programs(vb, fb).ok()).find(|l| {
            l.wgsl.lines().any(|s| s.trim_start().starts_with("out.v"))
        }) else {
            continue;
        };
        let declared: Vec<String> = vp
            .output_varyings
            .iter()
            .map(|v| format!("{:?}@{}x{}", v.usage, v.base_lane, v.components))
            .collect();
        // The linked module's own varying wiring is the layout that was actually used.
        let used: Vec<&str> = l
            .wgsl
            .lines()
            .filter(|s| s.trim_start().starts_with("out.v"))
            .map(|s| s.trim())
            .collect();
        lines.insert(
            vn.clone(),
            format!("{vn}  declared [{}]  wired {:?}", declared.join(" "), used),
        );
    }
    for l in lines.values() {
        println!("{l}");
    }
    println!("{} vertex programs", lines.len());
}

/// Every 0xE8 memory LOAD that carries a register offset, with the instruction that computed
/// that offset and - when its addend is a container LITERAL - the literal's value.
///
/// # The question this exists to answer
/// A load's address is `pointer + immediate + register offset`, and one title's particle
/// programs put a literal `0x00010000` into the register offset, which is 64 KB past a 512-byte
/// window (see `wgsl::emit_mem_load`). Whether that operand can be narrowed to 16 bits is a
/// question about EVERY program that has one, not about the one that misbehaves: this lists the
/// whole population so a candidate rule can be checked against it before a run rather than
/// after.
#[test]
#[ignore = "needs a captured corpus (game bytes); set VITASLOP_GXP_CORPUS"]
fn mem_load_register_offsets_and_what_computes_them() {
    use vitaslop_gxp_shader::ir::Op;
    let Some(dir) = corpus_dir() else { return };
    let mut seen: BTreeMap<String, String> = BTreeMap::new();
    for (name, bytes) in blobs(&dir) {
        let Ok(p) = Program::parse(&bytes) else { continue };
        for (stream, sh) in [
            ("primary", vitaslop_gxp_shader::usse::decode_shader(&p)),
            ("secondary", vitaslop_gxp_shader::usse::decode_secondary_shader(&p)),
        ] {
            for (i, ins) in sh.instrs.iter().enumerate() {
                let Op::MemLoad { elements, offset_bytes } = ins.op else { continue };
                let Some(base) = ins.srcs.first() else { continue };
                let offs: Vec<&vitaslop_gxp_shader::ir::Operand> = ins.srcs.iter().skip(1).collect();
                if offs.is_empty() {
                    continue;
                }
                for o in offs {
                    // What last wrote the offset register before this load, and - if it is an
                    // IMAD - what its addend actually holds.
                    let mut how = "(not written in this stream)".to_string();
                    for prev in sh.instrs[..i].iter().rev() {
                        let Some(d) = prev.dest else { continue };
                        if d.bank != o.bank || d.index != o.index {
                            continue;
                        }
                        how = format!("{:?}", prev.op);
                        if let Op::IntMad { .. } = prev.op
                            && let Some(s2) = prev.srcs.get(2)
                        {
                            let lit = p
                                .literals
                                .iter()
                                .find(|(r, _)| {
                                    s2.bank == vitaslop_gxp_shader::ir::Bank::SecondaryAttr
                                        && *r == u32::from(s2.index)
                                })
                                .map(|(_, v)| format!("{v:#010x}"))
                                .unwrap_or_else(|| "NOT a literal (driver-written)".into());
                            how = format!(
                                "{how} addend {:?}[{}] = {lit}",
                                s2.bank, s2.index as u32
                            );
                        }
                        break;
                    }
                    seen.insert(
                        format!("{how} | base {:?}[{}] +{offset_bytes} x{elements}", base.bank, base.index as u32),
                        format!("{name} {stream}#{i}: offset {:?}[{}] <- {how}", o.bank, o.index as u32),
                    );
                }
            }
        }
    }
    println!("\n-- {} distinct (offset producer, load shape) --", seen.len());
    for (k, v) in &seen {
        println!("  {v}\n      shape: {k}");
    }
}

/// Census: for every fragment blob that SOURCES the output bank, say whether that read can
/// see the DESTINATION at all - i.e. whether the program had already written the register.
///
/// `reads_output_bank` is deliberately conservative: any Output source anywhere counts, and a
/// program that uses `o[n]` as scratch after writing it therefore asks the renderer for a
/// full-attachment copy and a render-pass SPLIT it does not need. MEASURED on the phone, one
/// title's menu: `54.47 DESTINATION-COLOUR pass splits (108.51 MB copied)` over 56 draws.
/// This prints the split of genuine reads from scratch reads so the refinement is decided on
/// counts rather than on a guess.
#[test]
#[ignore = "needs a captured corpus (game bytes); set VITASLOP_GXP_CORPUS"]
fn tabulate_output_bank_reads_before_and_after_a_write() {
    let Some(dir) = corpus_dir() else {
        eprintln!("VITASLOP_GXP_CORPUS not set - nothing to analyse");
        return;
    };
    let (mut frags, mut readers, mut genuine, mut scratch) = (0usize, 0usize, 0usize, 0usize);
    for (name, bytes) in blobs(&dir) {
        let Ok(p) = Program::parse(&bytes) else { continue };
        if p.kind != ProgramKind::Fragment {
            continue;
        }
        frags += 1;
        let sh = vitaslop_gxp_shader::usse::decode_shader(&p);
        if !vitaslop_gxp_shader::module::reads_output_bank(&sh) {
            continue;
        }
        readers += 1;
        // Straight-line write-before-read, per register. A branch anywhere makes the order
        // unprovable from the instruction index, so such a program stays a genuine reader.
        let branches = sh.instrs.iter().any(|i| matches!(i.op, Op::Branch { .. }));
        let mut written: [bool; 256] = [false; 256];
        let mut first_unwritten: Option<(usize, u8)> = None;
        for (i, instr) in sh.instrs.iter().enumerate() {
            for s in &instr.srcs {
                if s.bank == Bank::Output && !written[s.index as usize] && first_unwritten.is_none()
                {
                    first_unwritten = Some((i, s.index));
                }
            }
            if let Some(d) = instr.dest
                && d.bank == Bank::Output
                && instr.write_mask.iter().all(|m| *m)
                && instr.pred == vitaslop_gxp_shader::ir::Predicate::Always
            {
                written[d.index as usize] = true;
            }
        }
        // What the RECOMPILE path actually does with it: `lower_dest_blend` rewrites the
        // recognised blend shapes into pipeline state, and only what is left pays a split.
        let lowered = vitaslop_gxp_shader::recompile_fragment(&bytes)
            .ok()
            .map(|r| r.dest_blend);
        let tag = match lowered {
            Some(Some(b)) => format!("LOWERED {b:?}"),
            Some(None) => "SPLIT (not lowered)".to_string(),
            None => "recompile FAILED".to_string(),
        };
        println!("           {name}: {tag}");
        match (branches, first_unwritten) {
            (false, None) => {
                scratch += 1;
                println!("  SCRATCH  {name} ({} instrs) - every Output source follows a full write", sh.instrs.len());
            }
            (b, fu) => {
                genuine += 1;
                println!(
                    "  DEST     {name} ({} instrs) branches={b} first unwritten read: {:?}",
                    sh.instrs.len(),
                    fu
                );
            }
        }
    }
    println!(
        "\n{frags} fragment blobs, {readers} source the output bank: {genuine} genuine destination reads, {scratch} scratch-only"
    );
}

/// The INDEX REGISTER counts REGISTER PAIRS, proved from the one program whose indexed table
/// is closed - see `wgsl::emit_load_index`.
///
/// A fighting title's particle-streak vertex program (`vert_869002f0`) holds EIGHT one-hot
/// `vec4` literals: the corner selection table of a quad. Two `LoadIndex` instructions (addends
/// 21 and 30) feed two indexed dot products against the SubUV extents `(umin, umax, vmin,
/// vmax)`, with a corner index shifted left by one in between.
///
/// Under the single-register reading the table is never touched: the reads land at SA 35..45,
/// which is the TAIL of the default uniform container followed by the DATA container's
/// uniform-buffer POINTERS. A pointer used as a texture-coordinate weight is a coordinate in
/// the millions, and the draw came out as a full-screen neon moire.
///
/// Under `idx = (src + addend) * 2` the two dots read SA `56 + 4c` and `72 + 4c`, which select
/// `(umin,vmin) (umin,vmax) (umax,vmax) (umax,vmin)` - a quad's winding, using all eight
/// entries of the table exactly once with no overrun. This test asserts that closure, which is
/// what makes the scale a measurement rather than a fit.
#[test]
#[ignore = "needs a captured corpus (game bytes); set VITASLOP_GXP_CORPUS"]
fn the_indexed_corner_table_closes_only_when_the_index_counts_register_pairs() {
    let Some(dir) = corpus_dir() else {
        eprintln!("VITASLOP_GXP_CORPUS not set - nothing to analyse");
        return;
    };
    let Some((name, bytes)) = blobs(&dir).into_iter().find(|(n, b)| blob_matches(n, b, "e3a37124470fbb63")) else {
        eprintln!("this corpus does not hold vert e3a37124470fbb63 - nothing to check");
        return;
    };
    let p = Program::parse(&bytes).expect("parse");
    let sh = vitaslop_gxp_shader::usse::decode_shader(&p);
    // The literals, as the SA registers they land in.
    let lits: BTreeMap<u32, f32> =
        p.literals.iter().map(|&(r, w)| (r, f32::from_bits(w))).collect();
    let one_hot = |base: u32| -> Option<usize> {
        let w: Vec<f32> = (0..4).map(|k| lits.get(&(base + k)).copied().unwrap_or(f32::NAN)).collect();
        if w.iter().any(|x| !(*x == 0.0 || *x == 1.0)) || w.iter().sum::<f32>() != 1.0 {
            return None;
        }
        w.iter().position(|x| *x == 1.0)
    };
    let addends: Vec<i32> = sh
        .instrs
        .iter()
        .filter_map(|i| match i.op {
            Op::LoadIndex { addend, .. } => Some(addend),
            _ => None,
        })
        .collect();
    assert!(addends.contains(&21) && addends.contains(&30), "{name}: addends {addends:?}");
    let scale = vitaslop_gxp_shader::module::index_register_scale() as u32;
    // dot1 reads `idx + 14`; dot2 reads `idx + 12`. Corner `c` shifts the source by `2c`.
    let mut u = Vec::new();
    let mut v = Vec::new();
    for c in 0..4u32 {
        let i1 = (2 * c + 21) * scale + 14;
        let i2 = (2 * c + 30) * scale + 12;
        u.push(one_hot(i1).unwrap_or_else(|| panic!("{name}: corner {c} dot1 at sa[{i1}] is not one-hot")));
        v.push(one_hot(i2).unwrap_or_else(|| panic!("{name}: corner {c} dot2 at sa[{i2}] is not one-hot")));
    }
    // Lanes 0/1 are the U extents and 2/3 the V extents, so the first dot must choose a U and
    // the second a V - and together they must name four DISTINCT corners.
    assert!(u.iter().all(|l| *l < 2), "{name}: dot1 lanes {u:?} are not U extents");
    assert!(v.iter().all(|l| *l >= 2), "{name}: dot2 lanes {v:?} are not V extents");
    let corners: std::collections::BTreeSet<(usize, usize)> =
        u.iter().zip(v.iter()).map(|(a, b)| (*a, *b)).collect();
    assert_eq!(corners.len(), 4, "{name}: corners {u:?}/{v:?} are not four distinct ones");
    println!("{name}: scale {scale} -> corners u={u:?} v={v:?}");
}

/// Census: every float->float `Pack` that writes MORE THAN ONE component, with the source
/// swizzle it carries - and whether that swizzle is a consecutive run.
///
/// A VPCK that writes `n` contiguous destination lanes reads `n` source registers, and when the
/// instruction REPEATS the source advances by the same `n`. So a two-lane pack inside a repeat
/// of stride two must read two CONSECUTIVE source registers, or the repeat's stride and the
/// read width disagree and the iterations overlap. This prints the population that decides
/// whether the decoded `c1` field means what the emitter reads it as.
#[test]
#[ignore = "needs a captured corpus (game bytes); set VITASLOP_GXP_CORPUS"]
fn multi_lane_float_pack_swizzles() {
    let Some(dir) = corpus_dir() else { return };
    let mut counts: BTreeMap<String, usize> = BTreeMap::new();
    let mut examples: BTreeMap<String, String> = BTreeMap::new();
    for (name, bytes) in blobs(&dir) {
        let Ok(p) = Program::parse(&bytes) else { continue };
        for (i, ins) in vitaslop_gxp_shader::usse::decode_shader(&p).instrs.iter().enumerate() {
            if !matches!(ins.op, Op::Pack { .. }) {
                continue;
            }
            let lanes = ins.write_mask.iter().filter(|m| **m).count();
            if lanes < 2 {
                continue;
            }
            let Some(s) = ins.srcs.first() else { continue };
            let used: Vec<u8> = s.swizzle[..lanes].to_vec();
            let consecutive = used.windows(2).all(|w| w[1] == w[0] + 1);
            let key = format!(
                "{} lanes, src bank {:?}, swizzle {:?} -> {}",
                lanes,
                s.bank,
                used,
                if consecutive { "consecutive" } else { "NOT consecutive" }
            );
            *counts.entry(key.clone()).or_default() += 1;
            examples.entry(key).or_insert_with(|| format!("{name} #{i} raw {:#018x}", ins.raw));
        }
    }
    let mut ranked: Vec<_> = counts.iter().collect();
    ranked.sort_by_key(|(_, n)| std::cmp::Reverse(**n));
    for (k, n) in ranked {
        println!("  {n:>4}  {k}\n          e.g. {}", examples[k]);
    }
}

/// Every INDEXED source operand in the corpus, with the op that reads it and the swizzle it
/// decodes to - the population that says whether an indexed operand carries a swizzle at all.
#[test]
#[ignore = "needs a captured corpus (game bytes); set VITASLOP_GXP_CORPUS"]
fn indexed_source_swizzles() {
    let Some(dir) = corpus_dir() else { return };
    let mut counts: BTreeMap<String, usize> = BTreeMap::new();
    for (_name, bytes) in blobs(&dir) {
        let Ok(p) = Program::parse(&bytes) else { continue };
        let mut all = vitaslop_gxp_shader::usse::decode_shader(&p).instrs;
        all.extend(vitaslop_gxp_shader::usse::decode_secondary_shader(&p).instrs);
        for ins in &all {
            for s in &ins.srcs {
                if s.bank != Bank::Indexed {
                    continue;
                }
                let lanes = ins.write_mask.iter().filter(|m| **m).count().max(1);
                *counts
                    .entry(format!(
                        "{:?} lanes={lanes} swizzle {:?} abs={} neg={}",
                        ins.op, s.swizzle, s.abs, s.neg
                    ))
                    .or_default() += 1;
            }
        }
    }
    let mut ranked: Vec<_> = counts.iter().collect();
    ranked.sort_by_key(|(_, n)| std::cmp::Reverse(**n));
    for (k, n) in ranked {
        println!("  {n:>4}  {k}");
    }
}

/// Link every `<key>.vert.gxp` / `<key>.frag.gxp` PAIR in a directory and rank what stops them.
///
/// A blob that recompiles alone can still fail to LINK: the interface between the two stages is
/// where the varying layout, the uniform banks and the memory windows are decided, and none of
/// them is visible to a single-stage test. `recompile_every_blob_and_rank_the_failures` reports
/// the decoder's frontier; this reports the RENDERER's, which is the one a black frame is about.
///
/// The directory is built straight out of a run's dropped-pair reports, so the corpus is exactly
/// the pairs a frame lost: each report names a `<key>` and the two blobs behind it, and the two
/// are written as `<key>.vert.gxp` and `<key>.frag.gxp`. The extraction is a few lines against
/// that log format and belongs to whoever is holding the run - it is not part of this crate, and
/// naming a path outside the repository here would only go stale.
#[test]
#[ignore = "needs a captured pair corpus (game bytes); set VITASLOP_GXP_PAIR_CORPUS"]
fn link_every_pair_and_rank_the_failures() {
    let Some(dir) = std::env::var_os("VITASLOP_GXP_PAIR_CORPUS").map(PathBuf::from) else {
        eprintln!("VITASLOP_GXP_PAIR_CORPUS not set - nothing to analyse");
        return;
    };
    let mut keys: Vec<String> = std::fs::read_dir(&dir)
        .expect("pair corpus dir")
        .filter_map(|e| {
            let n = e.ok()?.file_name().to_string_lossy().into_owned();
            n.strip_suffix(".vert.gxp").map(str::to_string)
        })
        .collect();
    keys.sort();

    let mut ok = 0usize;
    let mut by_reason: BTreeMap<String, Vec<String>> = BTreeMap::new();
    // One VERBATIM example per class, kept beside the collapsed shape: the digits are what say
    // WHICH register and which extent, and a ranking that hides them cannot be acted on.
    let mut verbatim: BTreeMap<String, String> = BTreeMap::new();
    for key in &keys {
        let v = std::fs::read(dir.join(format!("{key}.vert.gxp"))).expect("vert");
        let f = std::fs::read(dir.join(format!("{key}.frag.gxp"))).expect("frag");
        match link_programs(&v, &f) {
            Ok(_) => ok += 1,
            Err(e) => {
                // Rank by the SHAPE of the failure, not the instance: the numbers in a message
                // are one pair's registers, and grouping on them hides that ten pairs are one
                // gap. Digits collapse to `N`.
                let msg = e.to_string();
                let msg: String = msg.split_whitespace().collect::<Vec<_>>().join(" ");
                let shape_of = |m: &str| -> String {
                    let mut sh = String::new();
                    let mut pd = false;
                    for c in m.chars() {
                        if c.is_ascii_digit() {
                            if !pd { sh.push('N'); }
                            pd = true;
                        } else { sh.push(c); pd = false; }
                    }
                    sh
                };
                let mut shape = String::new();
                let mut prev_digit = false;
                for c in msg.chars() {
                    if c.is_ascii_digit() {
                        if !prev_digit {
                            shape.push('N');
                        }
                        prev_digit = true;
                    } else {
                        shape.push(c);
                        prev_digit = false;
                    }
                }
                by_reason.entry(shape).or_default().push(key.clone());
                verbatim.entry(shape_of(&msg)).or_insert_with(|| format!("{key}: {msg}"));
            }
        }
    }

    println!("\n{} of {} pairs LINK", ok, keys.len());
    let mut ranked: Vec<_> = by_reason.into_iter().collect();
    ranked.sort_by_key(|(_, v)| std::cmp::Reverse(v.len()));
    for (reason, pairs) in &ranked {
        println!("\n{:3} pairs: {}", pairs.len(), reason);
        println!("        e.g. {}", pairs.iter().take(4).cloned().collect::<Vec<_>>().join(" "));
        if let Some(v) = verbatim.get(reason) {
            println!("        ONE: {v}");
        }
    }
}

/// Rank a pair corpus by the F16 EMULATION it emits: the `gxp_h*` stores and `unpack2x16float`.
///
/// WGSL without the `f16` extension has no half type, so every 16-bit register in the USSE file
/// is a packed `u32` and every read and write of one is a CONVERSION. A desktop GPU's compiler
/// folds most of them away and a desktop measurement therefore prices this at zero
/// [[vitaslop-desktop-cannot-price-a-count-win]]; a tiler does not, and mlb's world pass is
/// where that bill lands [[vitaslop-f16-emulation-is-the-phones-world-pass]].
///
/// This is the STATIC instrument for that bill: per pair, the conversions in the linked module
/// and the body's line count, ranked. It cannot say what a frame costs - a pair's price is its
/// conversions times its SAMPLES, and only a coverage run knows the second factor - but it is
/// what says whether an emitter change moved the count at all, on every pair at once and with
/// no device in the loop.
///
/// `VITASLOP_GXP_PAIR_CORPUS=<dir>` selects the corpus, the same `<key>.vert.gxp` /
/// `<key>.frag.gxp` directory [`link_every_pair_and_rank_the_failures`] reads.
#[test]
#[ignore = "needs a captured pair corpus (game bytes); set VITASLOP_GXP_PAIR_CORPUS"]
fn rank_pairs_by_emitted_f16_conversions() {
    let Some(dir) = std::env::var_os("VITASLOP_GXP_PAIR_CORPUS").map(PathBuf::from) else {
        eprintln!("VITASLOP_GXP_PAIR_CORPUS not set - nothing to analyse");
        return;
    };
    let mut keys: Vec<String> = std::fs::read_dir(&dir)
        .expect("pair corpus dir")
        .filter_map(|e| {
            let n = e.ok()?.file_name().to_string_lossy().into_owned();
            n.strip_suffix(".vert.gxp").map(str::to_string)
        })
        .collect();
    keys.sort();

    let count = |hay: &str, needle: &str| hay.matches(needle).count();
    let mut rows: Vec<(usize, usize, usize, usize, usize, usize, usize, usize, String)> =
        Vec::new();
    let (mut linked_ok, mut failed) = (0usize, 0usize);
    for key in &keys {
        let v = std::fs::read(dir.join(format!("{key}.vert.gxp"))).expect("vert");
        let f = std::fs::read(dir.join(format!("{key}.frag.gxp"))).expect("frag");
        let Ok(linked) = link_programs(&v, &f) else {
            failed += 1;
            continue;
        };
        linked_ok += 1;
        let w = &linked.wgsl;
        // VALIDATED, not merely emitted: a text pass over the emitted WGSL (the half-register
        // unpacking, the bank sizing) can produce a module that links and does not COMPILE, and
        // in a real run that surfaces a whole pass away from here as dropped draws. naga is the
        // same front end wgpu hands the device.
        match naga::front::wgsl::parse_str(w) {
            Ok(module) => {
                let mut v = naga::valid::Validator::new(
                    naga::valid::ValidationFlags::all(),
                    naga::valid::Capabilities::all(),
                );
                if let Err(e) = v.validate(&module) {
                    panic!("{key}: linked module failed validation: {e:?}
{w}");
                }
            }
            Err(e) => panic!("{key}: linked module failed to parse: {e:?}
{w}"),
        }
        // >>> STORES ARE THE `gxp_h*` HELPERS NOW, NOT `pack2x16float`. See
        // [`count_conversions`]: the builtin survives only in the preamble, so counting it here
        // would report near-zero conversions for the whole corpus and read as a win.
        let packs = count(w, &format!("{HALF_LO_FN}("))
            + count(w, &format!("{HALF_HI_FN}("))
            + count(w, &format!("{HALF_PK_FN}(")) * 2;
        let unpacks = count(w, "unpack2x16float(");
        // The unpacked half-register file's ROUNDING is a conversion too, and it is what that
        // pass trades the pack/unpack pairs for. Counting only the two calls it removes would
        // make that pass look free, which is the one way this instrument could flatter its
        // subject.
        let quant = count(w, "gxp_q2(") * 2;
        // The guest's own instruction count, beside the WGSL the fragment becomes: the ratio is
        // what says whether a body is big because the PROGRAM is big or because the emission is.
        let frag_instrs = vitaslop_gxp_shader::recompile_fragment(&f)
            .map(|r| r.shader.instrs.len())
            .unwrap_or(0);
        // Texture samples and per-fragment memory-window words are the two costs a conversion
        // cut does not touch, so they are ranked in the same table rather than looked up later.
        let samples = count(w, "textureSample") + count(w, "textureSampleLevel");
        rows.push((
            packs + unpacks + quant,
            packs,
            unpacks,
            quant,
            frag_instrs,
            samples,
            declared_register_words(w),
            w.lines().count(),
            key.clone(),
        ));
    }
    rows.sort_by_key(|r| std::cmp::Reverse(r.0));

    let total: usize = rows.iter().map(|r| r.0).sum();
    println!(
        "
{linked_ok} pairs linked ({failed} did not), {total} f16 conversions emitted in total"
    );
    println!(
        "  {:>6} {:>6} {:>6} {:>6} {:>6} {:>4} {:>5} {:>6}  pair",
        "conv", "pack", "unpack", "quant", "instrs", "smpl", "words", "lines"
    );
    for (conv, packs, unpacks, quant, instrs, samples, words, lines, key) in &rows {
        println!(
            "  {conv:>6} {packs:>6} {unpacks:>6} {quant:>6} {instrs:>6} {samples:>4} {words:>5}              {lines:>6}  {key}"
        );
    }
    // A corpus whose pairs emit NO conversions is one where this whole lever is absent, and
    // that is a result too - it is why the same change is inert on another title.
    if total == 0 {
        println!("  no pair in this corpus emits a single f16 conversion");
    }
}

/// The function-local register STORAGE a module declares, in 32-bit words.
///
/// The unpacked half-register file buys conversions with SPACE: a half pair that was one `u32`
/// becomes two `f32`. On a phone that is not free - a program that needs more registers runs
/// fewer threads at once, and the bank SIZING that cut one title's warm render from 10.13 to
/// 4.08 ms is the same lever pointing the other way. Counted here so the trade is visible in the
/// same table as the win rather than discovered on a device.
fn declared_register_words(w: &str) -> usize {
    let mut words = 0usize;
    for line in w.lines() {
        let t = line.trim();
        let Some(rest) = t.strip_prefix("var ").and_then(|r| r.split_once(": array<")) else {
            continue;
        };
        let (ty, count) = rest.1.split_once(", ").unwrap_or(("", ""));
        let n: usize = count.trim_end_matches(">;").trim().parse().unwrap_or(0);
        words += n * match ty {
            "u32" | "i32" | "f32" => 1,
            "vec2<f32>" => 2,
            "vec4<f32>" => 4,
            // `array<bool, 4>` and anything else: not register-file storage worth counting.
            _ => 0,
        };
    }
    words
}

/// Every f16 conversion a linked module EXECUTES: the reads (`unpack2x16float`), the STORES
/// (the `gxp_h*` helpers) and the rounding the unpacked half-register file does.
///
/// >>> IT COUNTS THE HELPER CALLS, NOT `pack2x16float`, AND THE DIFFERENCE IS THE WHOLE VALUE
/// >>> OF THIS FUNCTION. Until the rounding fix a store WAS a `pack2x16float` spelled inline,
/// and counting that call was counting stores. It is now a call to `gxp_hlo`/`gxp_hhi`/
/// `gxp_hpk`, whose DEFINITIONS live in the preamble - so a counter still looking for the
/// builtin reads the preamble's own two or three occurrences and reports essentially ZERO
/// conversions for every program in the corpus. An instrument that silently reads zero is
/// worse than none, because the number it prints looks like an improvement
/// [[vitaslop-a-drop-count-needs-its-draw-count]].
///
/// A DEFINITION is not a call, so each helper's own `fn` line is subtracted. `gxp_q2` is two
/// roundings behind one name and is weighed as two; `gxp_hpk` is two narrowings behind one name
/// and is weighed the same way, because what this ranks is the arithmetic a fragment runs and
/// not the source lines it is spelled in.
fn count_conversions(w: &str) -> usize {
    // A call minus its own definition, for a helper named `f`.
    let calls = |f: &str| {
        w.matches(&format!("{f}(")).count() - usize::from(w.contains(&format!("fn {f}(")))
    };
    // `unpack2x16float(` contains `pack2x16float(`, so the reads are counted on the longer name
    // and the two-halves-at-once store helper is counted separately.
    let reads = w.matches("unpack2x16float(").count() - usize::from(w.contains("fn gxp_hq("));
    // A single-half store is one narrowing; a folded pair is two.
    let stores = calls(HALF_LO_FN) + calls(HALF_HI_FN) + calls(HALF_PK_FN) * 2;
    // The unpacked-half-register arm rounds without packing: `gxp_q2` is two, `gxp_hq` one.
    // `gxp_hq` is also called from inside `gxp_q2`'s body, which is a definition, not a site.
    let q2 = calls("gxp_q2");
    // `gxp_q2`'s BODY spells `gxp_hq` twice, and a body is not a call site. Its CALL sites are
    // spelled `gxp_q2(` and contain no `gxp_hq(` at all, so what comes off here is those two
    // occurrences once - not two per call, which would undercount every module that uses both.
    let hq = calls(HALF_QUANT_FN).saturating_sub(usize::from(w.contains("fn gxp_q2(")) * 2);
    reads + stores + q2 * 2 + hq
}

/// >>> THE CONVERSION COUNTER IS ITSELF A THING THAT CAN SILENTLY READ ZERO, so it is checked
/// >>> against a module whose conversions were counted by hand.
///
/// It already did read zero once, for a whole session's worth of would-be measurements: it
/// counted `pack2x16float`, the emitter stopped spelling stores that way, and nothing failed.
/// The only defence against that is a case where the right answer is known independently of
/// the function, which is what this is.
#[test]
fn the_conversion_counter_counts_calls_and_not_definitions() {
    use vitaslop_gxp_shader::wgsl::add_half_helpers;

    // Three stores and one read, counted by hand: the PAIR helper is two narrowings, the
    // single-half helper one, and the read one. Total 4.
    let body = "  r[0] = gxp_hpk(a, b);
  r[1] = gxp_hlo(r[1], c);
  r[2] = gxp_hhi(r[2], d);
                   let x = unpack2x16float(r[3])[0];
";
    let bare = count_conversions(body);
    assert_eq!(bare, 2 + 1 + 1 + 1, "counted by hand from the body alone: {body}");

    // >>> AND ADDING THE HELPER DEFINITIONS MUST NOT CHANGE THE COUNT. The preamble spells
    // every helper name once more and `gxp_hq`'s body spells `unpack2x16float` - so a counter
    // that did not subtract definitions would charge a module for conversions it never runs,
    // and would charge a DIFFERENT amount depending on which rounding arm the device chose.
    let with_helpers = add_half_helpers(body.to_string());
    assert!(with_helpers.contains("fn gxp_hpk("), "the helpers were added:
{with_helpers}");
    assert_eq!(
        count_conversions(&with_helpers),
        bare,
        "the preamble is definitions, not call sites:
{with_helpers}"
    );

    // The unpacked half-register file's rounding: `gxp_q2` is two, `gxp_hq` one, and neither
    // the helper preamble nor `gxp_q2`'s own body may be counted as a site.
    let q = "fn gxp_q2(v: vec2<f32>) -> vec2<f32> {
  return vec2<f32>(gxp_hq(v.x), gxp_hq(v.y));
}
               r_h[0] = gxp_q2(vec2<f32>(a, b));
  r_h[1][0] = gxp_hq(c);
";
    assert_eq!(count_conversions(&add_half_helpers(q.to_string())), 2 + 1, "got:
{q}");

    // And a module with no half work at all is zero, not "one because the word appears".
    assert_eq!(count_conversions("  r[0] = bitcast<u32>(1.0);
"), 0);
}

/// Price the F16 emulation in CONVERSIONS PER FRAME, by weighing each pair's emitted count with
/// the SAMPLES it actually painted.
///
/// [`rank_pairs_by_emitted_f16_conversions`] ranks programs; a frame does not care about
/// programs, it cares about fragments. One pair painting 78% of a pass's samples decides the
/// bill and a dozen pairs with bigger bodies and a handful of fragments do not
/// [[vitaslop-f16-emulation-is-the-phones-world-pass]]. This joins the two halves:
///
/// * `VITASLOP_GXP_COVERAGE_LOG` - a run log with `VITASLOP_GXM_DRAW_COVERAGE=1`, which reports
///   each pass's painted pairs and their sample counts, and with the `gxp pair` lines that name
///   each key's vertex and fragment blob HASHES;
/// * `VITASLOP_GXP_CORPUS` - that title's blob corpus, so the pair can be linked here.
///
/// Run it with `VITASLOP_GXP_HALF_REGS` in each position to price an emitter change in the only
/// unit that means anything: conversions a frame stops executing. It is still a STATIC count -
/// it cannot know what a driver folds, and a tiler and a desktop fold differently - so it ranks
/// and bounds, it does not predict milliseconds.
#[test]
#[ignore = "needs a coverage log and that title's corpus; set VITASLOP_GXP_COVERAGE_LOG"]
fn weigh_f16_conversions_by_draw_coverage() {
    let (Some(dir), Ok(log)) = (corpus_dir(), std::env::var("VITASLOP_GXP_COVERAGE_LOG")) else {
        eprintln!("set VITASLOP_GXP_CORPUS and VITASLOP_GXP_COVERAGE_LOG");
        return;
    };
    let text = std::fs::read_to_string(&log).expect("coverage log");

    // key -> (vertex hash, fragment hash), from the `gxp pair` reports.
    let mut blob_of: BTreeMap<String, (String, String)> = BTreeMap::new();
    for line in text.lines() {
        let Some(at) = line.find("gxp pair ") else { continue };
        let rest = &line[at + "gxp pair ".len()..];
        let Some((key, rest)) = rest.split_once(": vprog hash ") else { continue };
        let Some((vh, rest)) = rest.split_once(", fprog hash ") else { continue };
        let fh: String = rest.chars().take_while(|c| c.is_ascii_hexdigit()).collect();
        if fh.is_empty() {
            continue;
        }
        blob_of.insert(key.trim().to_string(), (vh.trim().to_string(), fh));
    }

    // key -> samples painted, summed over every pass the log reports, and how many pass reports
    // that was - the divisor that turns a log total into a FRAME.
    let mut samples: BTreeMap<String, u64> = BTreeMap::new();
    let mut first_passes = 0u64;
    for line in text.lines() {
        let Some(at) = line.find("gxm draw coverage: pass #") else { continue };
        if line[at..].starts_with("gxm draw coverage: pass #0:") {
            first_passes += 1;
        }
        let Some(list) = line.split_once("PAINTED: [").map(|(_, r)| r) else { continue };
        let list = list.split(']').next().unwrap_or("");
        for entry in list.split(", ") {
            // `<key> xN (M samples) at i`
            let mut it = entry.split_whitespace();
            let (Some(key), Some(_times), Some(count)) = (it.next(), it.next(), it.next()) else {
                continue;
            };
            let Ok(n) = count.trim_start_matches('(').parse::<u64>() else { continue };
            *samples.entry(key.to_string()).or_default() += n;
        }
    }
    let frames = first_passes.max(1);

    let all = blobs(&dir);
    let find = |h: &str| all.iter().find(|(name, b)| blob_matches(name, b, h)).map(|(_, b)| b.clone());

    let mut rows: Vec<(u64, u64, usize, String, String)> = Vec::new();
    let (mut total, mut unknown) = (0u64, 0u64);
    for (key, n) in &samples {
        let Some((vh, fh)) = blob_of.get(key) else {
            unknown += n;
            continue;
        };
        let (Some(v), Some(f)) = (find(vh), find(fh)) else {
            unknown += n;
            continue;
        };
        let Ok(linked) = link_programs(&v, &f) else {
            unknown += n;
            continue;
        };
        let conv = count_conversions(&linked.wgsl);
        total += n * conv as u64;
        rows.push((n * conv as u64, *n, conv, key.clone(), String::new()));
    }
    rows.sort_by_key(|r| std::cmp::Reverse(r.0));

    println!("\n{} pass reports, taken as {frames} frames", first_passes);
    println!(
        "  >>> {:.1}M F16 CONVERSIONS PER FRAME over {:.2}M samples",
        total as f64 / frames as f64 / 1e6,
        samples.values().sum::<u64>() as f64 / frames as f64 / 1e6
    );
    if unknown > 0 {
        // A pair whose blobs are not in this corpus is not a zero - saying so is the difference
        // between "this is the whole bill" and "this is the part that could be priced".
        println!(
            "  ({:.2}M samples/frame could NOT be priced - their blobs are not in this corpus)",
            unknown as f64 / frames as f64 / 1e6
        );
    }
    println!("  {:>12} {:>12} {:>6}  pair", "conv/frame", "samples/fr", "conv");
    for (weighted, n, conv, key, _) in rows.iter().take(12) {
        println!(
            "  {:>12.0} {:>12.0} {conv:>6}  {key}",
            *weighted as f64 / frames as f64,
            *n as f64 / frames as f64
        );
    }
}

/// Census the RAW prefetch-bearing words of every fragment descriptor, unreduced to flags.
///
/// [`tabulate_varying_descriptor_flags`] asks whether three named BITS agree. When they do not,
/// the next question is not which bit is right but whether the field is one bit at all: a
/// two-bit field whose corpus has only ever shown its low value looks exactly like a flag until
/// a program uses the high one. This prints `attribute_info`'s bits 8..11 and `component_info`'s
/// bits 4..7 as VALUES, against the size word, so the shape of the field is read off the
/// population instead of assumed.
///
/// Point `VITASLOP_GXP_CORPUS` at each corpus in turn; the interesting rows are the ones that
/// appear in only one title, which is what says a reading was never measured rather than agreed.
#[test]
#[ignore = "needs a captured corpus (game bytes); set VITASLOP_GXP_CORPUS"]
fn tabulate_prefetch_field_values() {
    let Some(dir) = corpus_dir() else { return };
    // key: (semantic nibble, info bits 11:8, component_info bits 7:4, size bits 7:6)
    let mut table: BTreeMap<(u32, u32, u32, u32), (usize, String)> = BTreeMap::new();
    for (name, bytes) in blobs(&dir) {
        let Ok(p) = Program::parse(&bytes) else { continue };
        if p.kind != ProgramKind::Fragment {
            continue;
        }
        for d in vitaslop_gxp_shader::container::raw_varying_descriptors(&bytes) {
            let [info, res, size, comp] = d;
            let key = ((info >> 12) & 0xf, (info >> 8) & 0xf, (comp >> 4) & 0xf, (size >> 6) & 0x3);
            let e = table.entry(key).or_insert_with(|| {
                (0, format!("{name} info={info:#010x} res={res} size={size:#x} comp={comp:#x}"))
            });
            e.0 += 1;
        }
    }
    println!("sem info[11:8] comp[7:4] size[7:6]  count  example");
    for ((sem, i, c, s), (n, ex)) in &table {
        println!("  {sem:#x}   {i:#04x}      {c:#04x}      {s}          {n:<5}  {ex}");
    }
}

/// The prefetch LOOKUP-KIND field (`attribute_info` bits 10:8) against the sampler it names:
/// is the unit a CUBE, and how many components its name/semantics suggest. See
/// `container::PrefetchLookup` for the reading this census backs.
#[test]
#[ignore = "needs a captured corpus (game bytes); set VITASLOP_GXP_CORPUS"]
fn census_prefetch_lookup_kind_against_the_sampler() {
    let Some(dir) = corpus_dir() else { return };
    let mut table: BTreeMap<(u32, bool), (usize, String)> = BTreeMap::new();
    for (name, bytes) in blobs(&dir) {
        let Ok(p) = Program::parse(&bytes) else { continue };
        if p.kind != ProgramKind::Fragment {
            continue;
        }
        for d in vitaslop_gxp_shader::container::raw_varying_descriptors(&bytes) {
            let [info, res, _size, _comp] = d;
            let kind = (info >> 8) & 0x7;
            if kind == 0 {
                continue;
            }
            let sampler = p.parameters.iter().find(|q| {
                q.category == vitaslop_gxp_shader::container::ParamCategory::Sampler
                    && q.resource_index == res as i32
            });
            let cube = sampler.is_some_and(|s| s.sampler_cube);
            let e = table.entry((kind, cube)).or_insert_with(|| {
                (0, format!("{name} unit {res} {:?} info={info:#010x}", sampler.map(|s| s.name.as_str())))
            });
            e.0 += 1;
        }
    }
    println!("kind cube  count  example");
    for ((k, c), (n, ex)) in &table {
        println!("  {k}   {c:5}  {n:<5}  {ex}");
    }
}

/// For every blob, the SA registers its code READS that no declared home covers, classified by
/// where they fall: inside the DATA container, inside the default uniform buffer's span, or
/// past everything.
///
/// `secondary_attr_init` refuses these one at a time, naming a register. That is the right
/// behaviour and the wrong diagnostic for asking WHAT the register is: a single number cannot
/// say whether one title has one unexplained slot or every title has scattered ones, and those
/// are different defects - the first is a layout slot nobody has identified, the second is the
/// register-addressing model. Run it over each corpus in turn and compare.
#[test]
#[ignore = "needs a captured corpus (game bytes); set VITASLOP_GXP_CORPUS"]
fn sa_reads_no_declared_home_covers() {
    let Some(dir) = corpus_dir() else { return };
    let mut total = 0usize;
    for (name, bytes) in blobs(&dir) {
        let Ok(p) = Program::parse(&bytes) else { continue };
        // Every home a read can legitimately have, as the linker counts them.
        let lits: std::collections::BTreeSet<u32> = p.literals.iter().map(|&(r, _)| r).collect();
        let texc: std::collections::BTreeSet<u32> =
            p.texture_control.iter().flat_map(|&(b, _)| b..b + 4).collect();
        let ptrs: std::collections::BTreeSet<u32> = p
            .containers
            .iter()
            .find(|c| c.index == 19)
            .map(|c| {
                p.uniform_buffer_bindings
                    .iter()
                    .map(|b| u32::from(c.base_sa) + u32::from(b.data_slot))
                    .collect()
            })
            .unwrap_or_default();
        let carried = p.sa_carried_extent();
        let data = p.containers.iter().find(|c| c.index == 19).map(|c| {
            (u32::from(c.base_sa), u32::from(c.base_sa) + u32::from(c.size_regs))
        });

        // Registers the code reads, and registers it writes, by the linker's own addressing.
        let (mut reads, mut writes) = (std::collections::BTreeSet::new(), std::collections::BTreeSet::new());
        for stream in [
            vitaslop_gxp_shader::usse::decode_secondary_shader(&p),
            vitaslop_gxp_shader::usse::decode_shader(&p),
        ] {
            for i in &stream.instrs {
                for s in &i.srcs {
                    if s.bank == Bank::SecondaryAttr {
                        for c in 0..4 {
                            let sel = s.swizzle[c] as u32;
                            if sel < 4 {
                                reads.insert(s.index as u32 + sel);
                            }
                        }
                    }
                }
                if let Some(d) = i.dest.as_ref()
                    && d.bank == Bank::SecondaryAttr {
                        let n = match i.op {
                            Op::MemLoad { elements, .. } => u32::from(elements),
                            _ => 4,
                        };
                        writes.extend((0..n).map(|k| d.index as u32 + k));
                    }
            }
        }
        let orphans: Vec<u32> = reads
            .iter()
            .copied()
            .filter(|r| {
                *r >= carried
                    && !lits.contains(r)
                    && !texc.contains(r)
                    && !ptrs.contains(r)
                    && !writes.contains(r)
            })
            .collect();
        if orphans.is_empty() {
            continue;
        }
        total += orphans.len();
        let where_ = |r: u32| match data {
            Some((lo, hi)) if r >= lo && r < hi => format!("DATA+{}", r - lo),
            _ => "past-everything".to_string(),
        };
        println!(
            "{name}: {}",
            orphans.iter().map(|&r| format!("sa[{r}]({})", where_(r))).collect::<Vec<_>>().join(" ")
        );
    }
    println!("\n{total} orphan SA reads in this corpus");
}

/// Print a digest of every blob's recompiled WGSL, so two builds can be compared exactly.
///
/// A count of what recompiles cannot see a change that alters a REGISTER an instruction reads:
/// the blob still recompiles, and the picture is wrong. This turns "is this decoder change inert
/// on the titles that already work" into a diff of two text files, which is the only form of
/// that question that can actually be answered before a ten-minute replay.
/// [[vitaslop-identical-output-is-evidence]]
#[test]
#[ignore = "needs a captured corpus (game bytes); set VITASLOP_GXP_CORPUS"]
fn digest_every_blobs_wgsl() {
    let Some(dir) = corpus_dir() else { return };
    for (name, bytes) in blobs(&dir) {
        let Ok(p) = Program::parse(&bytes) else { continue };
        let body = match p.kind {
            ProgramKind::Vertex => recompile_vertex(&bytes).map(|r| r.wgsl_body),
            ProgramKind::Fragment => recompile_fragment(&bytes).map(|r| r.wgsl_body),
        };
        match body {
            // FNV-1a over the emitted text: any changed register, swizzle or statement moves it.
            Ok(w) => {
                let mut h: u64 = 0xcbf2_9ce4_8422_2325;
                for b in w.as_bytes() {
                    h ^= u64::from(*b);
                    h = h.wrapping_mul(0x1000_0000_01b3);
                }
                println!("{name} {h:016x} {}", w.len());
            }
            Err(e) => println!("{name} FAILED {e}"),
        }
    }
}

/// For every fragment descriptor that declares a PREFETCH, print the texcoord it names beside
/// the TexCoord semantics the same program declares - the two candidate readings of the field
/// side by side.
///
/// `source_texcoord` is read as an ABSOLUTE `TexCoord(n)` semantic. The alternative is that it
/// INDEXES the program's own declared texcoords, and the two agree on every program whose
/// texcoords start at 0 - which is most of them. This prints the disagreements.
#[test]
#[ignore = "needs a captured corpus (game bytes); set VITASLOP_GXP_CORPUS"]
fn prefetch_source_against_the_texcoords_the_program_declares() {
    let Some(dir) = corpus_dir() else { return };
    let (mut agree, mut differ) = (0usize, 0usize);
    for (name, bytes) in blobs(&dir) {
        let Ok(p) = Program::parse(&bytes) else { continue };
        if p.kind != ProgramKind::Fragment {
            continue;
        }
        // The TexCoord semantics this program declares, in descriptor order.
        let declared: Vec<u32> = p
            .interpolants
            .iter()
            .filter_map(|i| match i.usage {
                vitaslop_gxp_shader::container::VaryingUsage::TexCoord(n) => Some(u32::from(n)),
                _ => None,
            })
            .collect();
        for it in &p.interpolants {
            let Some(pf) = it.prefetch else { continue };
            let src = u32::from(pf.source_texcoord);
            let absolute_ok = declared.contains(&src);
            let indexed = declared.get(src as usize).copied();
            if absolute_ok {
                agree += 1;
            } else {
                differ += 1;
                println!(
                    "{name}: prefetch source {src} - declared TexCoords {declared:?}; \
                     as an ABSOLUTE semantic it is NOT declared, as an INDEX it is {indexed:?}"
                );
            }
        }
    }
    println!("\n{agree} prefetches whose source is a declared TexCoord semantic, {differ} not");
}

/// IEEE half -> f32, so a literal table can be read in the width the program reads it in.
fn f16_to_f32(bits: u16) -> f32 {
    let sign = u32::from(bits >> 15) << 31;
    let exp = u32::from((bits >> 10) & 0x1f);
    let man = u32::from(bits & 0x3ff);
    let out = match exp {
        0 if man == 0 => sign,
        0 => {
            // Subnormal: normalise it into an f32 exponent.
            let shift = man.leading_zeros() - 21;
            sign | ((127 - 15 - shift) << 23) | ((man << (shift + 1)) & 0x7f_ffff)
        }
        0x1f => sign | 0x7f80_0000 | (man << 13),
        _ => sign | ((exp + 127 - 15) << 23) | (man << 13),
    };
    f32::from_bits(out)
}

/// One line per blob: the SA-register INITIALISER list the linker would build for it.
///
/// The counterpart of [`hash_every_blob_wgsl`] for the LINK stage. A change to which container
/// literals survive into a program's prologue moves nothing in a blob's own WGSL body - that is
/// built before linking - so the WGSL hash census reads IDENTICAL across both arms of such a
/// change and says nothing at all. Run this under both arms and diff, and it names every blob
/// on disk whose uniform prologue moves, in seconds.
#[test]
#[ignore = "needs a captured corpus (game bytes); set VITASLOP_GXP_CORPUS"]
fn hash_every_blob_sa_literals() {
    let Some(dir) = corpus_dir() else {
        eprintln!("VITASLOP_GXP_CORPUS not set - nothing to analyse");
        return;
    };
    let all = blobs(&dir);
    assert!(!all.is_empty(), "no .gxp blobs under {}", dir.display());
    for (name, bytes) in &all {
        let Ok(p) = Program::parse(bytes) else {
            println!("{name} PARSE-FAIL");
            continue;
        };
        match vitaslop_gxp_shader::link::sa_literal_init(&p) {
            Ok(mut l) => {
                l.sort_unstable();
                let regs: Vec<String> =
                    l.iter().map(|(r, v)| format!("sa[{r}]={v:#010x}")).collect();
                println!("{name} {}", regs.join(" "));
            }
            Err(e) => println!("{name} FAIL {e:?}"),
        }
    }
}

/// Every instruction whose destination write mask decodes to NOTHING, by group and by the
/// mask-control bits that produced it.
///
/// **A shipped compiler does not emit an instruction that writes nothing.** So every row this
/// prints is either a decode gap or a form worth naming, and the emitter is SILENT about them -
/// it emits no statement and reports nothing, which is exactly how a lost lane in the middle of
/// a skinning chain leaves the rest of the chain reading a stale accumulator.
#[test]
#[ignore = "needs a captured corpus (game bytes); set VITASLOP_GXP_CORPUS"]
fn instructions_whose_write_mask_is_empty() {
    use vitaslop_gxp_shader::usse::decode::{field, GROUP_TABLES};
    let Some(dir) = corpus_dir() else {
        eprintln!("VITASLOP_GXP_CORPUS not set - nothing to analyse");
        return;
    };
    let g00 = GROUP_TABLES.iter().find(|(n, _, _)| *n == "grp00_mad").expect("grp00_mad");
    // group -> count, and for group 0x00 the (data_format, swz_mask16, swz_mask32, swz_en) tally.
    let mut by_group: BTreeMap<u8, usize> = BTreeMap::new();
    let mut g00_bits: BTreeMap<(u32, u32, u32, u32), usize> = BTreeMap::new();
    let mut total = 0usize;
    let mut blobs_hit: std::collections::BTreeSet<String> = Default::default();
    for (name, bytes) in blobs(&dir) {
        let Ok(p) = Program::parse(&bytes) else { continue };
        for shader in [
            vitaslop_gxp_shader::usse::decode_shader(&p),
            vitaslop_gxp_shader::usse::decode_secondary_shader(&p),
        ] {
            for ins in &shader.instrs {
                total += 1;
                if ins.dest.is_none() || ins.write_mask.iter().any(|w| *w) || ins.blocked.is_some() {
                    continue;
                }
                *by_group.entry(ins.group).or_default() += 1;
                blobs_hit.insert(name.clone());
                if ins.group == 0x00 {
                    let (hi, lo) = ((ins.raw >> 32) as u32, ins.raw as u32);
                    let f = |n: &str| field(hi, g00.1, n);
                    *g00_bits
                        .entry((f("data_format"), f("swz_mask16"), f("swz_mask32"), f("swz_en")))
                        .or_default() += 1;
                    let _ = lo;
                }
            }
        }
    }
    println!("{total} instructions decoded; {} blobs carry an empty-mask write", blobs_hit.len());
    for (g, n) in &by_group {
        println!("  group {g:#04x}: {n}");
    }
    println!("group 0x00 by (data_format, swz_mask16, swz_mask32, swz_en):");
    for ((df, m16, m32, en), n) in &g00_bits {
        println!("  df={df} m16={m16} m32={m32} en={en}: {n}");
    }
}

/// Hash the LINKED WGSL of every vert x frag pairing in a corpus, so a LINK-STAGE change can be
/// censused the way `hash_every_blob_wgsl` censuses a DECODER change.
///
/// A per-blob hash is blind to anything the linker decides - the varying layout, the prefetch
/// coordinates, the SA literal initialiser - because none of it exists until two programs are
/// put together [[vitaslop-a-blob-recompiling-is-not-a-pair-linking]]. This links every
/// combination that links at all and prints one line per successful pairing, which turns
/// "what does this link change touch" into a diff of two text files.
#[test]
#[ignore = "needs a captured corpus (game bytes); set VITASLOP_GXP_CORPUS"]
fn hash_every_pair_link() {
    let Some(dir) = corpus_dir() else { return };
    let all = blobs(&dir);
    let verts: Vec<_> = all
        .iter()
        .filter(|(_, b)| Program::parse(b).map(|p| p.kind == ProgramKind::Vertex).unwrap_or(false))
        .collect();
    let frags: Vec<_> = all
        .iter()
        .filter(|(_, b)| Program::parse(b).map(|p| p.kind == ProgramKind::Fragment).unwrap_or(false))
        .collect();
    let mut n = 0usize;
    for (vn, vb) in &verts {
        for (fnm, fb) in &frags {
            let Ok(l) = link_programs(vb, fb) else { continue };
            let mut h: u64 = 0xcbf2_9ce4_8422_2325;
            for b in l.wgsl.as_bytes() {
                h ^= u64::from(*b);
                h = h.wrapping_mul(0x1000_0000_01b3);
            }
            println!("{vn}+{fnm} {h:016x} {}", l.wgsl.len());
            n += 1;
        }
    }
    println!("{n} linked pairings");
}

/// >>> WHICH OF A TITLE'S DESTINATION-READING FRAGMENT PROGRAMS STILL COST A PASS SPLIT, AND
/// WHAT SHAPE THE ONES THAT DO NOT LOWER ACTUALLY ARE.
///
/// A sports title measured **~230 destination-colour splits in one frame** on a phone, against
/// 9 for the title `lower_dest_blend` was built on. Each split is a full store and reload of
/// every tile on a tiling GPU, and that title is GPU-BOUND BY 2x - so the splits are the first
/// thing to price. The live renderer's `gxp dest colour` report says how many splits a pass took
/// and by which pair, but it FIRES ONCE and it can only name pairs it actually drew, so it
/// cannot say what the unlowered programs have in common.
///
/// This is the INVENTORY instead ([[an-inventory-beats-a-serial-hunt]]): every fragment blob in
/// the corpus, asked the same two questions the renderer asks, in the same order -
///   1. does it source the output bank at all (is it a destination reader), and
///   2. does `lower_dest_blend` turn it into pipeline state?
/// A program that answers yes then no is one split per draw, every frame, forever. The tail
/// instructions of each such program are printed because the ANSWER to "why did it not lower"
/// is a shape, and a shape has to be read to be added to `match_dest_blend`.
#[test]
#[ignore = "needs a captured corpus (game bytes); set VITASLOP_GXP_CORPUS"]
fn rank_the_destination_readers_that_do_not_lower() {
    let Some(dir) = corpus_dir() else {
        eprintln!("VITASLOP_GXP_CORPUS not set - nothing to analyse");
        return;
    };
    let all = blobs(&dir);
    assert!(!all.is_empty(), "no .gxp blobs under {}", dir.display());

    let mut frags = 0usize;
    let mut readers = 0usize;
    let mut lowered: BTreeMap<String, usize> = BTreeMap::new();
    let mut stuck: Vec<(String, vitaslop_gxp_shader::Shader)> = Vec::new();

    for (name, bytes) in &all {
        let Ok(program) = Program::parse(bytes) else { continue };
        if program.kind != ProgramKind::Fragment {
            continue;
        }
        frags += 1;
        let mut shader = vitaslop_gxp_shader::usse::decode_shader(&program);
        // BEFORE the rewrite: is this a destination reader at all? Asking after would count a
        // lowered program as never having been one, which is the number we are trying to split.
        if !vitaslop_gxp_shader::module::declares_dest_color(&shader) {
            continue;
        }
        readers += 1;
        match vitaslop_gxp_shader::module::lower_dest_blend(&mut shader) {
            Some(b) => *lowered.entry(format!("{:?}/{:?}", b.color, b.alpha)).or_default() += 1,
            None => stuck.push((name.clone(), shader)),
        }
    }

    println!("\n{frags} fragment programs, {readers} read the DESTINATION colour");
    println!("  LOWERED to pipeline state: {}", readers - stuck.len());
    for (shape, n) in &lowered {
        println!("    x{n:<3} {shape}");
    }
    println!("  >>> STILL SPLIT THE PASS: {}", stuck.len());

    // The tail is where the blend is: `match_dest_blend` matches the last instructions of the
    // program. Twelve is enough to hold every shape the matcher knows and the context around it.
    for (name, shader) in &stuck {
        // >>> AND IS IT LINEAR IN THE DESTINATION? That is what decides whether a DUAL-SOURCE
        // lowering (`out = dst*F + G`, with F emitted as src1) can take the program, or
        // whether it keeps its render-pass split.
        let linear = vitaslop_gxp_shader::module::dest_is_linear(shader);
        println!(
            "
  --- {name}: {} instructions, LINEAR IN DEST: {}{}",
            shader.instrs.len(),
            linear,
            if linear { "  <- a dual-source lowering can take this" } else { "" }
        );
        println!("      tail:");
        let from = shader.instrs.len().saturating_sub(12);
        for (i, instr) in shader.instrs.iter().enumerate().skip(from) {
            println!(
                "    [{i:3}] {:?} dest={:?} srcs={:?} mask={:?} pred={:?} half={}",
                instr.op, instr.dest, instr.srcs, instr.write_mask, instr.pred, instr.half_precision
            );
        }
    }
}

/// Print the DUAL-SOURCE plan (or the reason there is none) for every destination reader in
/// the corpus, with `VITASLOP_GXP_DUAL_TRACE=1` walking the analysis instruction by
/// instruction. A diagnostic, not an assertion.
#[test]
#[ignore]
fn print_dual_source_plans() {
    let Some(dir) = corpus_dir() else { return };
    vitaslop_gxp_shader::module::set_dual_source_blend(true);
    for (name, bytes) in &blobs(&dir) {
        let Ok(program) = Program::parse(bytes) else { continue };
        if program.kind != ProgramKind::Fragment {
            continue;
        }
        let mut shader = vitaslop_gxp_shader::usse::decode_shader(&program);
        if !vitaslop_gxp_shader::module::declares_dest_color(&shader) {
            continue;
        }
        if vitaslop_gxp_shader::module::lower_dest_blend(&mut shader).is_some() {
            continue;
        }
        eprintln!("== {name}");
        eprintln!("   {:?}", vitaslop_gxp_shader::fragment_dual_source_plan(bytes));
    }
}

/// EVERY blocked instruction in every blob, not just the first one the recompiler reports.
///
/// The recompiler stops at the first refusal, so a program with three unmodelled instructions
/// looks exactly like one with a single unmodelled instruction: wire the first and the next
/// appears. This walks the FULLY DECODED stream (`decode_shader` / `decode_secondary_shader`,
/// i.e. after repeat unrolling and every validation pass, which is where several refusals are
/// actually raised) and lists them all, so "what does this program still need" is one run
/// rather than one run per instruction.
#[test]
#[ignore = "needs a captured corpus (game bytes); set VITASLOP_GXP_CORPUS"]
fn every_blocked_instruction_in_every_blob() {
    let Some(dir) = corpus_dir() else { return };
    let mut tally: BTreeMap<&'static str, usize> = BTreeMap::new();
    for (name, bytes) in blobs(&dir) {
        let Ok(p) = Program::parse(&bytes) else { continue };
        let streams = [
            ("primary", vitaslop_gxp_shader::usse::decode_shader(&p)),
            ("secondary", vitaslop_gxp_shader::usse::decode_secondary_shader(&p)),
        ];
        let mut any = false;
        for (stream, shader) in &streams {
            for (i, instr) in shader.instrs.iter().enumerate() {
                let Some(why) = instr.blocked else { continue };
                if !any {
                    println!("\n{name}:");
                    any = true;
                }
                println!("  {stream} #{i:<4} {:#018x} {:?}\n      {why}", instr.raw, instr.op);
                *tally.entry(why).or_default() += 1;
            }
        }
        if !any {
            println!("\n{name}: nothing blocked");
        }
    }
    println!("\nby reason:");
    for (why, n) in &tally {
        println!("  {n:>4}  {why}");
    }
}

/// Every branch, its target, and every SMLSI / repeating instruction, in code-word numbering.
///
/// The per-instruction MOE (repeat) state is only readable off the stream when every path that
/// reaches a repeating instruction carries the SAME last SMLSI. That is a control-flow question,
/// so it needs the control-flow graph, and this is the listing the graph is read from.
#[test]
#[ignore = "needs a captured corpus (game bytes); set VITASLOP_GXP_CORPUS"]
fn branch_targets_against_the_smlsis_and_repeats_they_span() {
    use vitaslop_gxp_shader::usse::{decode, is_smlsi, opcode1, repeat_extra_iterations};
    let Some(dir) = corpus_dir() else { return };
    for (name, bytes) in blobs(&dir) {
        let Ok(p) = Program::parse(&bytes) else { continue };
        for (stream, code) in [("primary", &p.code), ("secondary", &p.secondary_code)] {
            let mut lines = Vec::new();
            for (i, &w) in code.iter().enumerate() {
                let d = decode(w);
                if is_smlsi(w) {
                    lines.push(format!("  #{i:<4} SMLSI {w:#018x}"));
                } else if let Op::Branch { rel } = d.op {
                    lines.push(format!(
                        "  #{i:<4} BRANCH rel {rel:+} -> #{} pred {:?}",
                        i as i64 + rel as i64,
                        d.pred
                    ));
                } else if repeat_extra_iterations(w).is_some_and(|e| e > 0) {
                    lines.push(format!(
                        "  #{i:<4} REPEAT grp {:#04x} x{}",
                        opcode1(w),
                        repeat_extra_iterations(w).unwrap() + 1
                    ));
                }
            }
            if !lines.is_empty() {
                println!("\n{name} {stream} ({} words):", code.len());
                for l in lines {
                    println!("{l}");
                }
            }
        }
    }
}

/// The four candidate LIMM destinations, scored by LIVENESS over every LIMM in the corpus.
///
/// A LIMM writes a constant. The register it writes must therefore be READ before anything
/// overwrites it - a compiler does not emit a load whose value nothing consumes. So for each
/// candidate (which field holds the NUMBER, and whether that number is double-register scaled)
/// this walks forward from the LIMM to the FIRST instruction that touches the candidate
/// register and reports whether that touch is a read (LIVE) or a write (DEAD). A reading under
/// which any LIMM in a shipped program is dead code is refuted.
///
/// The two number candidates are `[27:21]`, where every other group puts a destination number,
/// and the five upper bits `47,46,38,37,36` - which are the only upper bits that VARY across
/// the corpus once the always-set ones are excluded, and so the only ones that could hold a
/// number under the reading that spends the whole low word on the immediate.
#[test]
#[ignore = "needs a captured corpus (game bytes); set VITASLOP_GXP_CORPUS"]
fn limm_layout_candidates_by_liveness() {
    use vitaslop_gxp_shader::usse::{bits, decode, opcode1};
    let Some(dir) = corpus_dir() else { return };
    let is_limm =
        |w: u64| opcode1(w) == 0x1f && bits(w, 58, 56) == 0b100 && bits(w, 53, 52) == 0b10;
    let bank = |sel: u32| ["Temp", "Output", "PrimaryAttr", "(index mode)"][(sel & 3) as usize];
    let mut tally: BTreeMap<&'static str, (usize, usize, usize)> = BTreeMap::new();
    let mut sweep: BTreeMap<(u32, u32), (usize, usize, usize)> = BTreeMap::new();
    let mut total = 0usize;
    for (name, bytes) in blobs(&dir) {
        let Ok(p) = Program::parse(&bytes) else { continue };
        for (i, &w) in p.code.iter().enumerate() {
            if !is_limm(w) {
                continue;
            }
            total += 1;
            let sel = bits(w, 33, 32);
            let bn = bank(sel).to_string();
            let low = bits(w, 27, 21);
            let upper = (bits(w, 47, 47) << 4)
                | (bits(w, 46, 46) << 3)
                | (bits(w, 38, 38) << 2)
                | (bits(w, 37, 37) << 1)
                | bits(w, 36, 36);
            // A BRUTE-FORCE SWEEP over every 7-bit window in the word, scored the same way.
            // If one window is live everywhere and every other is dead somewhere, the field is
            // found rather than chosen.
            for b in 0..58u32 {
                let n = bits(w, b + 6, b);
                for (scale, idx) in [(1u32, n), (2, n * 2)] {
                    let idx = idx as u8;
                    let mut verdict = 2; // untouched
                    for &v in p.code.iter().skip(i + 1) {
                        let d = decode(v);
                        let same = |o: &vitaslop_gxp_shader::ir::Operand| {
                            o.index == idx && format!("{:?}", o.bank) == bn
                        };
                        if d.srcs.iter().any(same) {
                            verdict = 0;
                            break;
                        }
                        // A LATER LIMM IS A WRITE TOO. It decodes with `dest: None` because it
                        // is blocked, so a walk that only looks at `dest` cannot see the one
                        // thing that kills a constant's live range - another constant landing
                        // in the same register. Under the candidate being scored, a LIMM whose
                        // own bank and number match is exactly that write.
                        let later_limm = is_limm(v)
                            && bits(v, 33, 32) == sel
                            && (bits(v, b + 6, b) * scale) as u8 == idx;
                        if later_limm || d.dest.as_ref().is_some_and(same) {
                            verdict = 1;
                            break;
                        }
                    }
                    let e = sweep.entry((b, scale)).or_insert((0usize, 0usize, 0usize));
                    match verdict {
                        0 => e.0 += 1,
                        1 => e.1 += 1,
                        _ => e.2 += 1,
                    }
                }
            }
            let mut line = format!("{name} #{i} {w:#018x} bank {bn}");
            for (label, n) in [("low[27:21]", low), ("upper5", upper)] {
                for (scale, idx) in [("x1", n), ("x2", n * 2)] {
                    let key: &'static str = match (label, scale) {
                        ("low[27:21]", "x1") => "low[27:21] undoubled",
                        ("low[27:21]", _) => "low[27:21] doubled",
                        (_, "x1") => "upper5 undoubled",
                        _ => "upper5 doubled",
                    };
                    let idx = idx as u8;
                    let mut verdict = "untouched";
                    for &v in p.code.iter().skip(i + 1) {
                        let d = decode(v);
                        let same = |o: &vitaslop_gxp_shader::ir::Operand| {
                            o.index == idx && format!("{:?}", o.bank) == bn
                        };
                        if d.srcs.iter().any(same) {
                            verdict = "LIVE";
                            break;
                        }
                        if d.dest.as_ref().is_some_and(same) {
                            verdict = "DEAD";
                            break;
                        }
                    }
                    let e = tally.entry(key).or_default();
                    match verdict {
                        "LIVE" => e.0 += 1,
                        "DEAD" => e.1 += 1,
                        _ => e.2 += 1,
                    }
                    line.push_str(&format!("  | {key} -> {bn}[{idx}] {verdict}"));
                }
            }
            println!("{line}");
        }
    }
    println!("
{total} LIMMs. By candidate (live / dead / untouched):");
    for (k, (l, d, u)) in &tally {
        println!("  {k:<22} {l:>4} live  {d:>4} DEAD  {u:>4} untouched");
    }
    println!("
EVERY 7-bit window, no dead LIMM under it, fewest untouched first:");
    let mut rows: Vec<_> = sweep.iter().filter(|(_, v)| v.1 == 0).collect();
    rows.sort_by_key(|(_, v)| v.2);
    for ((b, scale), (l, d, u)) in rows.iter().take(14) {
        println!("  [{:>2}:{:>2}] x{scale}  {l:>4} live  {d:>4} DEAD  {u:>4} untouched", b + 6, b);
    }
}

/// For every LIMM in the corpus: the destination each candidate field reading names, and what
/// the program does with that register afterwards.
///
/// This is the evidence a LIMM decode has to rest on, collected so the next pass does not
/// re-derive it. A LIMM is `dest = <32-bit immediate>`, so the two questions are WHERE the
/// destination is encoded and HOW the immediate is assembled, and they are not equally
/// answerable from this corpus:
///
/// * THE DESTINATION BANK IS AT BITS[33:32], and this is as close to settled as two words can
///   make it. Every other group in this decoder reads a 2-bit destination bank selector there
///   (VPCK, VMOV, VTSTMSK all do), the corpus's two LIMMs carry 1 and 2, and 1/2 are OUTPUT and
///   PRIMATTR in the table every one of those groups uses. The first LIMM is followed two words
///   later by three reads of `Output[0]` - the ONLY writable-bank OUTPUT register the program
///   reads before anything writes it - so a LIMM with bank OUTPUT and destination number 0 is
///   exactly what that program is missing.
/// * THE IMMEDIATE IS NOT DERIVABLE HERE, and no part of this prints a decode. The two words
///   differ in only 31 bits, the ISA note says the 32-bit value is assembled from THREE fields
///   without saying which (its own field positions collide with the opcode discriminant, so it
///   is wrong on its face), and two mutually exclusive layouts fit both words:
///     - immediate = bits[31:0] verbatim, giving the two canonical constants `0x00000000` and
///       `0x00FFFFFF`, with the destination NUMBER then having to live in the five varying
///       upper bits (47, 46, 38, 37, 36) - which no 7-bit field covers;
///     - destination number at bits[27:21] (where every other group puts it), giving 0 and 7,
///       with the immediate's top 11 bits then living in the upper half - and the only varying
///       upper bits are those same five.
///   They cannot both be right, and the corpus has exactly TWO LIMM words in it (this listing
///   prints the count), both from one program, with no third to separate them. A LIMM decoded
///   under the wrong one writes a WRONG CONSTANT into a real register, silently, which is
///   strictly worse than the dropped draw it would replace - so it stays blocked.
///
/// >>> AND THE SECOND LIMM REFUTES BOTH NUMBER READINGS, which is why this is a negative result
/// rather than a near miss. Its `number[27:21]` is 7 and its bank is PRIMATTR:
///   * DOUBLED (`PrimaryAttr[14]`) is read six times after it and never written - a perfect
///     live range, except that two of those six readers are the moves that take BLEND INDEX 2
///     and BLEND INDEX 3 into the bone-matrix lookup. A skinned mesh does not blend against a
///     constant bone, so a LIMM that overwrites that register cannot be what the program means.
///   * UNDOUBLED (`PrimaryAttr[7]`) is read by NOTHING before the next instruction writes it,
///     so under that reading the LIMM is dead code - which no compiler emits either.
/// The FIRST LIMM, by contrast, closes under every reading at once (`Output[0]`, number 0,
/// doubled or not, read five times and written by nothing), so it cannot separate them. That is
/// the shape of the remaining gap: the BANK is evidenced, the NUMBER has two readings and the
/// corpus refutes both, and the immediate's assembly has no evidence at all.
#[test]
#[ignore = "needs a captured corpus (game bytes); set VITASLOP_GXP_CORPUS"]
fn limm_destination_candidates_against_what_the_program_reads() {
    use vitaslop_gxp_shader::usse::{bits, decode, opcode1};
    let Some(dir) = corpus_dir() else { return };
    let is_limm =
        |w: u64| opcode1(w) == 0x1f && bits(w, 58, 56) == 0b100 && bits(w, 53, 52) == 0b10;
    let bank = |sel: u32| ["Temp", "Output", "PrimaryAttr", "(index mode)"][(sel & 3) as usize];
    let mut total = 0usize;
    for (name, bytes) in blobs(&dir) {
        let Ok(p) = Program::parse(&bytes) else { continue };
        for (i, &w) in p.code.iter().enumerate() {
            if !is_limm(w) {
                continue;
            }
            total += 1;
            let sel = bits(w, 33, 32);
            let n = bits(w, 27, 21);
            println!("\n{name} primary #{i} {w:#018x}");
            println!("  bank[33:32] = {sel} -> {}", bank(sel));
            println!(
                "  number[27:21] = {n}  -> {}[{n}] undoubled, {}[{}] doubled",
                bank(sel),
                bank(sel),
                n * 2
            );
            println!("  low word [31:0] = {:#010x}", bits(w, 31, 0));
            println!(
                "  varying upper bits: b47={} b46={} b38={} b37={} b36={}",
                bits(w, 47, 47),
                bits(w, 46, 46),
                bits(w, 38, 38),
                bits(w, 37, 37),
                bits(w, 36, 36)
            );
            // What the program does with each candidate register AFTER the LIMM, which is the
            // only thing that can tell a live destination from a dead one.
            for (label, idx) in [("undoubled", n as u8), ("doubled", (n * 2) as u8)] {
                let mut reads = Vec::new();
                let mut writes = Vec::new();
                for (j, &v) in p.code.iter().enumerate().skip(i + 1).take(48) {
                    let d = decode(v);
                    let same = |o: &vitaslop_gxp_shader::ir::Operand| {
                        o.index == idx
                            && format!("{:?}", o.bank) == bank(sel).replace("(index mode)", "?")
                    };
                    if d.srcs.iter().any(same) {
                        reads.push(j);
                    }
                    if d.dest.as_ref().is_some_and(same) {
                        writes.push(j);
                    }
                }
                println!(
                    "  {label} {}[{idx}]: read at {reads:?}, written at {writes:?}",
                    bank(sel)
                );
            }
        }
    }
    println!("\n{total} LIMM word(s) in this corpus - the whole evidence base for its layout");
}

/// Every `LoadIndex` whose index register NOTHING in the stream then reads - and the two
/// instructions that follow it, raw.
///
/// An index register is loaded to be USED: the guest's compiler does not emit a
/// `LoadIndex` and then address nothing with it. So a program where `idx` is written and never
/// read is a program where the CONSUMER's indexed operand decoded as a plain register, and the
/// emitted body silently reads a fixed register where the hardware reads a computed one. That
/// is invisible to every other check here - the shader recompiles, links and draws.
#[test]
#[ignore = "needs a captured corpus (game bytes); set VITASLOP_GXP_CORPUS"]
fn load_index_whose_index_register_nothing_reads() {
    use vitaslop_gxp_shader::ir::{Bank, Op};
    let Some(dir) = corpus_dir() else { return };
    let (mut progs, mut with_li, mut orphan) = (0usize, 0usize, 0usize);
    let mut shown = 0usize;
    for (name, bytes) in blobs(&dir) {
        let Ok(p) = Program::parse(&bytes) else { continue };
        for (which, shader) in [
            ("primary", vitaslop_gxp_shader::usse::decode_shader(&p)),
            ("secondary", vitaslop_gxp_shader::usse::decode_secondary_shader(&p)),
        ] {
            if shader.instrs.is_empty() {
                continue;
            }
            progs += 1;
            let loads: Vec<usize> = shader
                .instrs
                .iter()
                .enumerate()
                .filter(|(_, i)| matches!(i.op, Op::LoadIndex { .. }))
                .map(|(at, _)| at)
                .collect();
            if loads.is_empty() {
                continue;
            }
            with_li += 1;
            let reads = shader
                .instrs
                .iter()
                .filter(|i| i.srcs.iter().any(|s| matches!(s.bank, Bank::Indexed)))
                .count();
            if reads > 0 {
                continue;
            }
            orphan += 1;
            if shown >= 12 {
                continue;
            }
            shown += 1;
            println!(
                "\n== {name} {which}: {} LoadIndex, 0 indexed reads",
                loads.len()
            );
            for at in loads.iter().take(2) {
                for k in 0..3usize {
                    let Some(i) = shader.instrs.get(at + k) else { continue };
                    println!(
                        "   #{:<4} raw={:#018x} grp {:#04x} {:<12} dst={:?} srcs={:?}",
                        at + k,
                        i.raw,
                        i.group,
                        i.op.mnemonic(),
                        i.dest.map(|d| (d.bank, d.index)),
                        i.srcs.iter().map(|s| (s.bank, s.index)).collect::<Vec<_>>()
                    );
                }
            }
        }
    }
    println!(
        "\n-- {orphan} of {with_li} programs that load an index register read NOTHING through it \
         ({progs} programs in corpus) --"
    );
}

/// Bit 55 of a group-0x15 IMAD32, against whether a `LoadIndex` sits just above it.
///
/// The field is not in the group's reserved-bit check and not in any of its decoded operands,
/// so if it correlates with an index load it is the operand mode the decode is missing.
#[test]
#[ignore = "needs a captured corpus (game bytes); set VITASLOP_GXP_CORPUS"]
fn imad32_bit55_against_a_preceding_load_index() {
    use vitaslop_gxp_shader::ir::Op;
    let Some(dir) = corpus_dir() else { return };
    // (bit55, a LoadIndex within 3 instructions above) -> count
    let mut tally: BTreeMap<(u8, bool), usize> = BTreeMap::new();
    let mut example: BTreeMap<(u8, bool), String> = BTreeMap::new();
    let mut words: Vec<(bool, u64)> = Vec::new();
    for (name, bytes) in blobs(&dir) {
        let Ok(p) = Program::parse(&bytes) else { continue };
        for shader in [
            vitaslop_gxp_shader::usse::decode_shader(&p),
            vitaslop_gxp_shader::usse::decode_secondary_shader(&p),
        ] {
            for (at, i) in shader.instrs.iter().enumerate() {
                if i.group != 0x15 {
                    continue;
                }
                let b55 = ((i.raw >> 55) & 1) as u8;
                let near = (1..=3usize).any(|k| {
                    at.checked_sub(k)
                        .and_then(|j| shader.instrs.get(j))
                        .is_some_and(|j| matches!(j.op, Op::LoadIndex { .. }))
                });
                words.push((near, i.raw));
                *tally.entry((b55, near)).or_default() += 1;
                example
                    .entry((b55, near))
                    .or_insert_with(|| format!("{name} #{at} raw={:#018x}", i.raw));
            }
        }
    }
    println!("\n-- group-0x15 IMAD32: bit 55 against a LoadIndex within 3 instructions above --");
    for ((b55, near), n) in &tally {
        println!(
            "  bit55={b55} load_index_above={near:<5} {n:<6} e.g. {}",
            example[&(*b55, *near)]
        );
    }
    println!("\n-- every bit that varies, near a LoadIndex vs not --");
    for bit in 0..64u32 {
        let mut c = [[0usize; 2]; 2];
        for (near, w) in &words {
            c[*near as usize][((w >> bit) & 1) as usize] += 1;
        }
        if c[0][0] + c[1][0] == 0 || c[0][1] + c[1][1] == 0 {
            continue;
        }
        let perfect = (c[0][0] == 0 && c[1][1] == 0) || (c[0][1] == 0 && c[1][0] == 0);
        println!(
            "  bit {bit:<2} near:[0={} 1={}] far:[0={} 1={}]{}",
            c[1][0],
            c[1][1],
            c[0][0],
            c[0][1],
            if perfect { "   <<< SPLITS PERFECTLY" } else { "" }
        );
    }
}

/// The group-0x14 index load against the group-0x15 IMAD32 that consumes it: does the load's
/// `[25:21]` field name the register the IMAD32 reads as `src0`, and does `[34:33]` name its
/// bank?
///
/// `decode_grp_i16mad` writes `Bank::Index` and the consumer reads a PLAIN register, so today
/// the index register is written and never read
/// ([`load_index_whose_index_register_nothing_reads`]). If these two fields line up with the
/// consumer on every occurrence, the load's destination is an ordinary register and the pairing
/// is closed by the corpus rather than assumed.
#[test]
#[ignore = "needs a captured corpus (game bytes); set VITASLOP_GXP_CORPUS"]
fn the_index_load_destination_against_its_consumer() {
    use vitaslop_gxp_shader::ir::Op;
    let Some(dir) = corpus_dir() else { return };
    let f = |w: u64, hi: u32, lo: u32| ((w >> lo) & ((1u64 << (hi - lo + 1)) - 1)) as u32;
    let (mut pairs, mut idx_agree) = (0usize, 0usize);
    let mut bank_map: BTreeMap<(u32, String), usize> = BTreeMap::new();
    let mut disagree: Vec<String> = Vec::new();
    for (name, bytes) in blobs(&dir) {
        let Ok(p) = Program::parse(&bytes) else { continue };
        for shader in [
            vitaslop_gxp_shader::usse::decode_shader(&p),
            vitaslop_gxp_shader::usse::decode_secondary_shader(&p),
        ] {
            for (at, li) in shader.instrs.iter().enumerate() {
                if !matches!(li.op, Op::LoadIndex { .. }) {
                    continue;
                }
                let Some(consumer) = (1..=3usize)
                    .filter_map(|k| shader.instrs.get(at + k))
                    .find(|i| i.group == 0x15)
                else {
                    continue;
                };
                let Some(src0) = consumer.srcs.first() else { continue };
                pairs += 1;
                let dest_field = f(li.raw, 25, 21);
                if dest_field == src0.index as u32 {
                    idx_agree += 1;
                } else if disagree.len() < 8 {
                    disagree.push(format!(
                        "{name} #{at}: load [25:21]={dest_field} but consumer src0 = {:?}[{}] (load raw {:#018x})",
                        src0.bank, src0.index, li.raw
                    ));
                }
                *bank_map.entry((f(li.raw, 34, 33), format!("{:?}", src0.bank))).or_default() += 1;
            }
        }
    }
    println!("\n-- {idx_agree} of {pairs} index loads have [25:21] == the consumer's src0 register --");
    for d in &disagree {
        println!("  {d}");
    }
    println!("-- load [34:33] -> consumer src0 bank --");
    for ((sel, bank), n) in &bank_map {
        println!("  [34:33]={sel} -> {bank:<14} {n}");
    }
}

/// The group-0x14 index load's undecoded variable bits, split by the `(b8,b51)` form flag -
/// what is left unexplained once `[19:18]`/`[17:14]` (source), `[25:21]`/`33` (destination) and
/// `[6:0]` (addend) are accounted for.
#[test]
#[ignore = "needs a captured corpus (game bytes); set VITASLOP_GXP_CORPUS"]
fn index_load_leftover_bits_by_form() {
    use vitaslop_gxp_shader::ir::Op;
    let Some(dir) = corpus_dir() else { return };
    let f = |w: u64, hi: u32, lo: u32| ((w >> lo) & ((1u64 << (hi - lo + 1)) - 1)) as u32;
    let mut tally: BTreeMap<(u32, u32, u32, u32, u32), usize> = BTreeMap::new();
    for (_, bytes) in blobs(&dir) {
        let Ok(p) = Program::parse(&bytes) else { continue };
        for shader in [
            vitaslop_gxp_shader::usse::decode_shader(&p),
            vitaslop_gxp_shader::usse::decode_secondary_shader(&p),
        ] {
            for i in &shader.instrs {
                if !matches!(i.op, Op::LoadIndex { .. }) {
                    continue;
                }
                *tally
                    .entry((f(i.raw, 8, 8), f(i.raw, 51, 51), f(i.raw, 45, 45), f(i.raw, 54, 54), f(i.raw, 34, 34)))
                    .or_default() += 1;
            }
        }
    }
    println!("\n-- index load: (b8,b51) form vs the undecoded b45 / b54 / b34 --");
    for ((b8, b51, b45, b54, b34), n) in &tally {
        println!("  b8={b8} b51={b51} | b45={b45} b54={b54} b34={b34}  {n}");
    }
}

/// Does bit 34 of a group-0x14 index load SELECT A HALF of its source register?
///
/// The closure available is a collision: within one program, two loads with the SAME source
/// register and the SAME addend must fetch the same matrix row for the same bone - so if they
/// exist and feed DIFFERENT destinations, something in the word must tell them apart, and bit
/// 34 is the only field left. If every such collision is resolved by bit 34 and no pair agrees
/// on all four, the bit is a source selector; if collisions survive it, it is not.
#[test]
#[ignore = "needs a captured corpus (game bytes); set VITASLOP_GXP_CORPUS"]
fn index_load_bit34_resolves_same_source_same_addend_collisions() {
    use vitaslop_gxp_shader::ir::Op;
    let Some(dir) = corpus_dir() else { return };
    let f = |w: u64, hi: u32, lo: u32| ((w >> lo) & ((1u64 << (hi - lo + 1)) - 1)) as u32;
    let (mut collisions, mut split_by_b34, mut unresolved) = (0usize, 0usize, 0usize);
    let mut examples: Vec<String> = Vec::new();
    for (name, bytes) in blobs(&dir) {
        let Ok(p) = Program::parse(&bytes) else { continue };
        for shader in [
            vitaslop_gxp_shader::usse::decode_shader(&p),
            vitaslop_gxp_shader::usse::decode_secondary_shader(&p),
        ] {
            // (source bank, source reg, addend) -> the (bit34, destination) of each load
            let mut groups: BTreeMap<(u32, u32, u32), Vec<(u32, u32)>> = BTreeMap::new();
            for i in &shader.instrs {
                if !matches!(i.op, Op::LoadIndex { .. }) || f(i.raw, 8, 8) != 1 {
                    continue;
                }
                groups
                    .entry((f(i.raw, 19, 18), f(i.raw, 17, 14), f(i.raw, 6, 0)))
                    .or_default()
                    .push((f(i.raw, 34, 34), f(i.raw, 25, 21) | (f(i.raw, 33, 33) << 8)));
            }
            for (k, v) in &groups {
                if v.len() < 2 {
                    continue;
                }
                collisions += 1;
                let b34s: std::collections::BTreeSet<u32> = v.iter().map(|(b, _)| *b).collect();
                if b34s.len() == v.len() {
                    split_by_b34 += 1;
                } else {
                    unresolved += 1;
                    if examples.len() < 6 {
                        examples.push(format!("{name} src bank{}[{}] addend {} -> {v:?}", k.0, k.1, k.2));
                    }
                }
            }
        }
    }
    println!(
        "\n-- (source, addend) collisions among (b8=1) index loads: {collisions} groups, \
         {split_by_b34} told apart by bit 34, {unresolved} NOT --"
    );
    for e in &examples {
        println!("  {e}");
    }
}

/// Is bit 34 of a `b8=1` index load CONSTANT across the three rows of one bone?
///
/// The three loads that share a source register are the three rows of ONE bone
/// (addends 0,1,2), so whatever names the bone cannot change between them. A bit that VARIES
/// inside such a group is therefore not a source-half or bone selector at all.
#[test]
#[ignore = "needs a captured corpus (game bytes); set VITASLOP_GXP_CORPUS"]
fn index_load_bit34_across_the_rows_of_one_bone() {
    use vitaslop_gxp_shader::ir::Op;
    let Some(dir) = corpus_dir() else { return };
    let f = |w: u64, hi: u32, lo: u32| ((w >> lo) & ((1u64 << (hi - lo + 1)) - 1)) as u32;
    let (mut groups_n, mut varies) = (0usize, 0usize);
    let mut examples: Vec<String> = Vec::new();
    for (name, bytes) in blobs(&dir) {
        let Ok(p) = Program::parse(&bytes) else { continue };
        for shader in [
            vitaslop_gxp_shader::usse::decode_shader(&p),
            vitaslop_gxp_shader::usse::decode_secondary_shader(&p),
        ] {
            let mut groups: BTreeMap<(u32, u32), Vec<(u32, u32)>> = BTreeMap::new();
            for i in &shader.instrs {
                if !matches!(i.op, Op::LoadIndex { .. }) || f(i.raw, 8, 8) != 1 {
                    continue;
                }
                groups
                    .entry((f(i.raw, 19, 18), f(i.raw, 17, 14)))
                    .or_default()
                    .push((f(i.raw, 6, 0), f(i.raw, 34, 34)));
            }
            for (k, v) in &groups {
                if v.len() < 2 {
                    continue;
                }
                groups_n += 1;
                let set: std::collections::BTreeSet<u32> = v.iter().map(|(_, b)| *b).collect();
                if set.len() > 1 {
                    varies += 1;
                    if examples.len() < 6 {
                        examples.push(format!("{name} src bank{}[{}] (addend,b34) {v:?}", k.0, k.1));
                    }
                }
            }
        }
    }
    println!(
        "\n-- bit 34 across the rows of one bone: {groups_n} multi-row groups, \
         bit 34 VARIES inside {varies} of them --"
    );
    for e in &examples {
        println!("  {e}");
    }
}

/// Which reading of a `b8=1` index load's SOURCE names a register an earlier instruction wrote?
///
/// An index load reads a blend index the program packed a few instructions above it, so under
/// the right field reading every source is a register already written IN STREAM ORDER. Two
/// readings are compared: today's (`bank [19:18]`, number `[17:14]`) and the six-bit
/// (`bank` = bit 34, number `[19:14]`). A reading that names unwritten registers is refuted.
#[test]
#[ignore = "needs a captured corpus (game bytes); set VITASLOP_GXP_CORPUS"]
fn index_load_source_readings_against_what_the_program_wrote() {
    use vitaslop_gxp_shader::ir::{Bank, Op};
    let Some(dir) = corpus_dir() else { return };
    let f = |w: u64, hi: u32, lo: u32| ((w >> lo) & ((1u64 << (hi - lo + 1)) - 1)) as u32;
    // Bank as a small ordinal, so a (bank, register) pair can live in a set.
    let ord = |b: Bank| match b {
        Bank::Temp => 0u8,
        Bank::Output => 1,
        Bank::PrimaryAttr => 2,
        Bank::SecondaryAttr => 3,
        _ => 9,
    };
    let bank4 = |s: u32| match s & 3 {
        0 => 0u8,
        1 => 1,
        2 => 2,
        _ => 3,
    };
    let (mut n, mut ok_old, mut ok_new) = (0usize, 0usize, 0usize);
    let mut bad_new: Vec<String> = Vec::new();
    let mut bad_old: Vec<String> = Vec::new();
    for (name, bytes) in blobs(&dir) {
        let Ok(p) = Program::parse(&bytes) else { continue };
        for shader in [
            vitaslop_gxp_shader::usse::decode_shader(&p),
            vitaslop_gxp_shader::usse::decode_secondary_shader(&p),
        ] {
            // Every (bank ordinal, flat register) an instruction has written so far. A PA
            // register an attribute supplies counts as written from the start, so a PA source
            // is accepted under either reading and only TEMP/OUTPUT test stream order.
            let mut written: std::collections::BTreeSet<(u8, u32)> = Default::default();
            for i in &shader.instrs {
                if matches!(i.op, Op::LoadIndex { .. }) && f(i.raw, 8, 8) == 1 {
                    n += 1;
                    let old = (bank4(f(i.raw, 19, 18)), f(i.raw, 17, 14));
                    let new = (if f(i.raw, 34, 34) == 0 { 0u8 } else { 2 }, f(i.raw, 19, 14));
                    if old.0 == 2 || written.contains(&old) {
                        ok_old += 1;
                    } else if bad_old.len() < 6 {
                        bad_old.push(format!("{name}: OLD bank{}[{}] never written (raw {:#018x})", old.0, old.1, i.raw));
                    }
                    if new.0 == 2 || written.contains(&new) {
                        ok_new += 1;
                    } else if bad_new.len() < 6 {
                        bad_new.push(format!("{name}: SIX-BIT bank{}[{}] never written (raw {:#018x})", new.0, new.1, i.raw));
                    }
                }
                if let Some(d) = i.dest {
                    for c in 0..4usize {
                        if i.write_mask[c] {
                            written.insert((ord(d.bank), d.index as u32 + c as u32));
                        }
                    }
                }
            }
        }
    }
    println!(
        "
-- {n} b8=1 index loads: source already written under the OLD reading {ok_old},          under the SIX-BIT reading {ok_new} --"
    );
    for b in &bad_old {
        println!("  {b}");
    }
    for b in &bad_new {
        println!("  {b}");
    }
}

/// Which reading of a group-0x08 PACK's source register names one an earlier instruction wrote?
///
/// The source is decoded as an R6 number `[13:8]` scaled by two, so it can only name EVEN
/// registers and the component selector reaches the odd halves. A football title's blend-index
/// extraction says otherwise: two identical `PackIntCopy` words differ only in bits 8 and 7,
/// and the one with bit 7 set must read `r[17]` - a register the R6 reading cannot name at all
/// (it decodes it as `r[16]`, which nothing in the program has written).
///
/// This asks the whole corpus which reading survives: the R6 one, or a seven-bit `[13:7]`
/// register number. Only TEMP and OUTPUT sources test anything - a PA register an attribute
/// supplies is written from the start under either reading.
#[test]
#[ignore = "needs a captured corpus (game bytes); set VITASLOP_GXP_CORPUS"]
fn pack_source_readings_against_what_the_program_wrote() {
    use vitaslop_gxp_shader::ir::Bank;
    let Some(dir) = corpus_dir() else { return };
    let f = |w: u64, hi: u32, lo: u32| ((w >> lo) & ((1u64 << (hi - lo + 1)) - 1)) as u32;
    let ord = |b: Bank| match b {
        Bank::Temp => 0u8,
        Bank::Output => 1,
        Bank::PrimaryAttr => 2,
        Bank::SecondaryAttr => 3,
        _ => 9,
    };
    let (mut n, mut ok_r6, mut ok_r7, mut bit7_set) = (0usize, 0usize, 0usize, 0usize);
    let mut bad_r6: Vec<String> = Vec::new();
    let mut bad_r7: Vec<String> = Vec::new();
    for (name, bytes) in blobs(&dir) {
        let Ok(p) = Program::parse(&bytes) else { continue };
        for shader in [
            vitaslop_gxp_shader::usse::decode_shader(&p),
            vitaslop_gxp_shader::usse::decode_secondary_shader(&p),
        ] {
            let mut written: std::collections::BTreeSet<(u8, u32)> = Default::default();
            for i in &shader.instrs {
                // Only the wired pack group, only a plain register source (no extension row),
                // and only the source formats where bit 7 is not already the component's high
                // bit (`src_fmt == 6`).
                if i.group == 0x08 && f(i.raw, 49, 49) == 0 && f(i.raw, 43, 41) != 6
                    && let Some(s) = i.srcs.first()
                        && matches!(s.bank, Bank::Temp | Bank::Output) {
                            n += 1;
                            if f(i.raw, 7, 7) == 1 {
                                bit7_set += 1;
                            }
                            let b = ord(s.bank);
                            let r6 = s.index as u32;
                            let r7 = f(i.raw, 13, 7);
                            if written.contains(&(b, r6)) {
                                ok_r6 += 1;
                            } else if bad_r6.len() < 6 {
                                bad_r6.push(format!("{name}: R6 bank{b}[{r6}] unwritten (raw {:#018x})", i.raw));
                            }
                            if written.contains(&(b, r7)) {
                                ok_r7 += 1;
                            } else if bad_r7.len() < 6 {
                                bad_r7.push(format!("{name}: R7 bank{b}[{r7}] unwritten (raw {:#018x})", i.raw));
                            }
                        }
                if let Some(d) = i.dest {
                    for c in 0..4usize {
                        if i.write_mask[c] {
                            written.insert((ord(d.bank), d.index as u32 + c as u32));
                        }
                    }
                }
            }
        }
    }
    println!(
        "\n-- {n} pack sources in TEMP/OUTPUT ({bit7_set} with bit 7 set): already written under \
         the R6 reading {ok_r6}, under the seven-bit [13:7] reading {ok_r7} --"
    );
    for b in bad_r6.iter().chain(bad_r7.iter()) {
        println!("  {b}");
    }
}

/// Bits 1 and 7 of a group-0x08 PACK with a plain register source, split by source format -
/// the two candidates for comp0's HIGH selector bit.
#[test]
#[ignore = "needs a captured corpus (game bytes); set VITASLOP_GXP_CORPUS"]
fn pack_comp0_high_bit_candidates_by_source_format() {
    let Some(dir) = corpus_dir() else { return };
    let f = |w: u64, hi: u32, lo: u32| ((w >> lo) & ((1u64 << (hi - lo + 1)) - 1)) as u32;
    let mut tally: BTreeMap<(u32, u32, u32), usize> = BTreeMap::new();
    for (_, bytes) in blobs(&dir) {
        let Ok(p) = Program::parse(&bytes) else { continue };
        for shader in [
            vitaslop_gxp_shader::usse::decode_shader(&p),
            vitaslop_gxp_shader::usse::decode_secondary_shader(&p),
        ] {
            for i in &shader.instrs {
                if i.group == 0x08 && f(i.raw, 49, 49) == 0 {
                    *tally
                        .entry((f(i.raw, 43, 41), f(i.raw, 1, 1), f(i.raw, 7, 7)))
                        .or_default() += 1;
                }
            }
        }
    }
    println!("\n-- pack (src_fmt, bit1, bit7) --");
    for ((fmt, b1, b7), n) in &tally {
        println!("  src_fmt={fmt} bit1={b1} bit7={b7}  {n}");
    }
}

/// The content hash of every blob that carries a `b8=1` index load - the SKINNING programs, in
/// the form a live run's `gxp pair <key>: vprog hash <h>` line names them by.
#[test]
#[ignore = "needs a captured corpus (game bytes); set VITASLOP_GXP_CORPUS"]
fn hashes_of_every_blob_that_skins() {
    use vitaslop_gxp_shader::ir::Op;
    let Some(dir) = corpus_dir() else { return };
    for (name, bytes) in blobs(&dir) {
        let Ok(p) = Program::parse(&bytes) else { continue };
        let n = vitaslop_gxp_shader::usse::decode_shader(&p)
            .instrs
            .iter()
            .filter(|i| matches!(i.op, Op::LoadIndex { .. }) && (i.raw >> 8) & 1 == 1)
            .count();
        if n > 0 {
            println!("{:016x}  {name}  {n} index loads", p.hash);
        }
    }
}

/// Do the `b8=1` index loads of ONE program agree on bit 34?
///
/// If a program mixes the two values, whatever bit 34 selects is a per-LOAD property; if every
/// program is uniform in it, it is a per-PROGRAM one, and a global arm over the whole title is
/// a fair test of what the factor should be.
#[test]
#[ignore = "needs a captured corpus (game bytes); set VITASLOP_GXP_CORPUS"]
fn index_load_bit34_within_one_program() {
    use vitaslop_gxp_shader::ir::Op;
    let Some(dir) = corpus_dir() else { return };
    let (mut uniform0, mut uniform1, mut mixed) = (0usize, 0usize, 0usize);
    let mut examples: Vec<String> = Vec::new();
    for (name, bytes) in blobs(&dir) {
        let Ok(p) = Program::parse(&bytes) else { continue };
        let mut seen = [false; 2];
        for shader in [
            vitaslop_gxp_shader::usse::decode_shader(&p),
            vitaslop_gxp_shader::usse::decode_secondary_shader(&p),
        ] {
            for i in &shader.instrs {
                if matches!(i.op, Op::LoadIndex { .. }) && (i.raw >> 8) & 1 == 1 {
                    seen[((i.raw >> 34) & 1) as usize] = true;
                }
            }
        }
        match seen {
            [true, true] => {
                mixed += 1;
                if examples.len() < 6 {
                    examples.push(name.clone());
                }
            }
            [true, false] => uniform0 += 1,
            [false, true] => uniform1 += 1,
            _ => {}
        }
    }
    println!(
        "\n-- programs with b8=1 index loads: {uniform0} all bit34=0, {uniform1} all bit34=1, \
         {mixed} MIXED --"
    );
    for e in &examples {
        println!("  mixed: {e}");
    }
}

/// The `b8=1` index loads of the programs whose bit 34 is SET, printed with what the two
/// readings of that bit would name and whether an earlier instruction wrote it.
#[test]
#[ignore = "needs a captured corpus (game bytes); set VITASLOP_GXP_CORPUS"]
fn index_loads_of_the_bit34_set_programs() {
    use vitaslop_gxp_shader::ir::{Bank, Op};
    let Some(dir) = corpus_dir() else { return };
    let f = |w: u64, hi: u32, lo: u32| ((w >> lo) & ((1u64 << (hi - lo + 1)) - 1)) as u32;
    let ord = |b: Bank| match b {
        Bank::Temp => 0u8,
        Bank::Output => 1,
        Bank::PrimaryAttr => 2,
        Bank::SecondaryAttr => 3,
        _ => 9,
    };
    let mut shown = 0usize;
    for (name, bytes) in blobs(&dir) {
        let Ok(p) = Program::parse(&bytes) else { continue };
        let shader = vitaslop_gxp_shader::usse::decode_shader(&p);
        if !shader
            .instrs
            .iter()
            .any(|i| matches!(i.op, Op::LoadIndex { .. }) && (i.raw >> 8) & 1 == 1 && (i.raw >> 34) & 1 == 1)
        {
            continue;
        }
        if shown >= 3 {
            continue;
        }
        shown += 1;
        println!("\n== {name} (declared pa_regs {})", p.primary_reg_count);
        let mut written: std::collections::BTreeSet<(u8, u32)> = Default::default();
        for i in &shader.instrs {
            if matches!(i.op, Op::LoadIndex { .. }) && (i.raw >> 8) & 1 == 1 {
                let n = f(i.raw, 19, 14);
                println!(
                    "   raw {:#018x} src n={n} as PA written={} as TEMP written={} addend={}",
                    i.raw,
                    written.contains(&(2, n)),
                    written.contains(&(0, n)),
                    f(i.raw, 6, 0)
                );
            }
            if let Some(d) = i.dest {
                for c in 0..4usize {
                    if i.write_mask[c] {
                        written.insert((ord(d.bank), d.index as u32 + c as u32));
                    }
                }
            }
        }
    }
}

/// The set of ADDENDS the `b8=1` index loads of one program use for ONE source register - the
/// rows of one bone's matrix, and therefore the STRIDE between bones.
///
/// A bone's matrix is a run of consecutive float4 rows, so the loads that share a source
/// register are that run, and the number of them is how far the NEXT bone's matrix starts. If
/// every group is exactly `{0, 1, ..., n-1}` the stride is readable off the program itself
/// rather than assumed.
#[test]
#[ignore = "needs a captured corpus (game bytes); set VITASLOP_GXP_CORPUS"]
fn index_load_addend_sets_per_source() {
    use vitaslop_gxp_shader::ir::Op;
    let Some(dir) = corpus_dir() else { return };
    let f = |w: u64, hi: u32, lo: u32| ((w >> lo) & ((1u64 << (hi - lo + 1)) - 1)) as u32;
    let mut shapes: BTreeMap<String, usize> = BTreeMap::new();
    for (_, bytes) in blobs(&dir) {
        let Ok(p) = Program::parse(&bytes) else { continue };
        for shader in [
            vitaslop_gxp_shader::usse::decode_shader(&p),
            vitaslop_gxp_shader::usse::decode_secondary_shader(&p),
        ] {
            let mut groups: BTreeMap<(u32, u32), std::collections::BTreeSet<u32>> = BTreeMap::new();
            for i in &shader.instrs {
                if matches!(i.op, Op::LoadIndex { .. }) && f(i.raw, 8, 8) == 1 {
                    groups
                        .entry((f(i.raw, 34, 34), f(i.raw, 19, 14)))
                        .or_default()
                        .insert(f(i.raw, 6, 0));
                }
            }
            for set in groups.values() {
                let v: Vec<u32> = set.iter().copied().collect();
                *shapes.entry(format!("{v:?}")).or_default() += 1;
            }
        }
    }
    println!("\n-- addend sets per (source bank, source register), b8=1 loads --");
    for (shape, n) in &shapes {
        println!("  {shape}  x{n}");
    }
}

/// The MAXIMUM addend any `b8=1` index load of a program uses, per program - the candidate
/// for the stride between one index's block of rows and the next.
#[test]
#[ignore = "needs a captured corpus (game bytes); set VITASLOP_GXP_CORPUS"]
fn index_load_max_addend_per_program() {
    use vitaslop_gxp_shader::ir::Op;
    let Some(dir) = corpus_dir() else { return };
    let mut tally: BTreeMap<u32, usize> = BTreeMap::new();
    let mut odd: Vec<String> = Vec::new();
    for (name, bytes) in blobs(&dir) {
        let Ok(p) = Program::parse(&bytes) else { continue };
        let mut max: Option<u32> = None;
        for shader in [
            vitaslop_gxp_shader::usse::decode_shader(&p),
            vitaslop_gxp_shader::usse::decode_secondary_shader(&p),
        ] {
            for i in &shader.instrs {
                if matches!(i.op, Op::LoadIndex { .. }) && (i.raw >> 8) & 1 == 1 {
                    let a = (i.raw & 0x7f) as u32;
                    max = Some(max.map_or(a, |m: u32| m.max(a)));
                }
            }
        }
        if let Some(m) = max {
            *tally.entry(m).or_default() += 1;
            if m != 2 && odd.len() < 8 {
                odd.push(format!("{name}: max addend {m}"));
            }
        }
    }
    println!("\n-- max addend per program (b8=1 loads) --");
    for (m, n) in &tally {
        println!("  max {m}: {n} programs");
    }
    for o in &odd {
        println!("  {o}");
    }
}

/// >>> WHAT A COLOUR-NO-OP MEMO MISS COSTS, AND WHAT THE BLOB-ONLY PREAMBLE WAS OF IT.
///
/// # Why this is a test and not a browser arm
/// The fold was the largest single CPU item in a frame on a baseball title (`key` 62% of
/// `prepare`, `colour-fold` 5-11 ms) and TWO attempts to make it cheaper were reverted for
/// aiming at the wrong half. The census that finally named the halves
/// (`EncodeWork::fold_asks`) says the per-draw BYTE HASH reads **128-144 bytes an ask** and
/// that the asks MISS about 8 times a frame at gameplay - so the cost is a miss, and a miss
/// used to run `recompile_fragment`: a full USSE decode AND a full WGSL emission.
///
/// Proving that by running the browser twice costs an hour a pair of arms and compares two
/// windows that are never quite the same scene. This measures the thing itself, on the real
/// blobs, in seconds - and it is an A/B on ONE build, which a cross-session millisecond
/// comparison is not [[vitaslop-compare-against-the-same-build-twice]].
///
/// It PRINTS rather than asserts a speed-up: a timing threshold in a test suite is a flake on
/// a busy machine, and the number wanted here is the RATIO, which the reader can see.
#[test]
#[ignore = "needs a captured corpus (game bytes); set VITASLOP_GXP_CORPUS"]
fn what_a_colour_fold_memo_miss_costs_with_and_without_the_prepared_program() {
    use std::time::Instant;
    use vitaslop_gxp_shader::fold::{
        fragment_colour_terms, fragment_colour_terms_prepared, DrawUniforms, FoldProgram,
    };
    let Some(dir) = corpus_dir() else { return };

    // The eligible programs are the only ones that reach the fold at all - everything else is
    // short-circuited by level one and pays none of this.
    let eligible: Vec<(String, Vec<u8>)> = blobs(&dir)
        .into_iter()
        .filter(|(_, b)| matches!(Program::parse(b).map(|p| p.kind), Ok(ProgramKind::Fragment)))
        .filter(|(_, b)| FoldProgram::prepare(b).is_some())
        .collect();
    if eligible.is_empty() {
        println!("  NO ELIGIBLE PROGRAM in this corpus - this run proves nothing either way");
        return;
    }

    let block = vec![0u8; 1024];
    let windows: Vec<(u32, &[u8])> = (0..4).map(|k| (0x1000_0000 + k * 0x1000, &block[..])).collect();
    let u = DrawUniforms { frag_sa: &block, windows: &windows };

    // Enough repeats that one program's answer is not a single clock tick. A MISS is what is
    // being priced, so the OLD path runs end to end every time - that is exactly what it did.
    const REPS: u32 = 20;
    let (mut old_ns, mut new_ns, mut prep_ns) = (0u128, 0u128, 0u128);
    let mut checked = 0usize;
    let mut disagreed: Vec<String> = Vec::new();

    for (name, bytes) in &eligible {
        // OLD: everything per miss.
        let t = Instant::now();
        let mut old_last = None;
        for _ in 0..REPS {
            old_last = fragment_colour_terms(bytes, &u);
        }
        old_ns += t.elapsed().as_nanos();

        // NEW: the blob-only preamble ONCE (which is what the renderer's per-blob memo holds),
        // then the per-draw evaluation per miss.
        let t = Instant::now();
        let fp = FoldProgram::prepare(bytes).expect("filtered to the eligible above");
        prep_ns += t.elapsed().as_nanos();
        let t = Instant::now();
        let mut new_last = None;
        for _ in 0..REPS {
            new_last = fragment_colour_terms_prepared(&fp, &u, &mut None);
        }
        new_ns += t.elapsed().as_nanos();

        // >>> AND THE ANSWERS MUST BE IDENTICAL. That is the whole claim of the split: it moves
        // work, it does not change a verdict. A speed-up that came with a different answer
        // would be a regression wearing a benchmark's clothes.
        checked += 1;
        if format!("{old_last:?}") != format!("{new_last:?}") {
            disagreed.push(format!("{name}: old {old_last:?} vs prepared {new_last:?}"));
        }
    }

    let per = |n: u128| n as f64 / (eligible.len() as f64 * REPS as f64) / 1000.0;
    println!(
        "-- a colour-fold MISS over {} eligible programs x{REPS}: OLD {:.1} us, PREPARED {:.1} us \
         (+{:.1} us ONCE per blob) = {:.1}x --",
        eligible.len(),
        per(old_ns),
        per(new_ns),
        prep_ns as f64 / eligible.len() as f64 / 1000.0,
        per(old_ns) / per(new_ns).max(1e-9),
    );
    for d in &disagreed {
        println!("  DISAGREED {d}");
    }
    assert!(
        disagreed.is_empty(),
        "{} of {checked} programs fold to a DIFFERENT answer through the prepared path - the \
         split was supposed to move work, not change a verdict",
        disagreed.len()
    );
}

#[test]
#[ignore = "needs a captured corpus (game bytes); set VITASLOP_GXP_CORPUS"]
fn the_colour_no_op_fold_over_every_eligible_program() {
    use vitaslop_gxp_shader::fold::{
        fragment_can_be_identity, fragment_colour_is_destination, fragment_colour_terms, DrawUniforms,
    };
    let Some(dir) = corpus_dir() else { return };
    let (mut frags, mut eligible) = (0usize, 0usize);
    let mut shapes: BTreeMap<String, usize> = BTreeMap::new();
    for (name, bytes) in blobs(&dir) {
        let Ok(p) = Program::parse(&bytes) else { continue };
        if p.kind != ProgramKind::Fragment {
            continue;
        }
        frags += 1;
        let can = fragment_can_be_identity(&bytes);
        if can {
            eligible += 1;
        }
        // Two blocks, neither of which needs a running game: all ZEROS, and all f16 ONES. A
        // window at four plausible base addresses, because the base is what the program's own
        // pointer register is seeded with and a wrong one simply leaves the loads unknown -
        // which is the conservative direction.
        for (label, fill) in [("zeros", 0u8), ("f16 ones", 0x3c)] {
            let mut block = vec![0u8; 1024];
            if fill != 0 {
                for (i, b) in block.iter_mut().enumerate() {
                    *b = if i % 2 == 1 { fill } else { 0 };
                }
            }
            let windows: Vec<(u32, &[u8])> =
                (0..4).map(|k| (0x1000_0000 + k * 0x1000, &block[..])).collect();
            let u = DrawUniforms { frag_sa: &block, windows: &windows };
            let verdict = fragment_colour_is_destination(&bytes, &u);
            assert!(
                !verdict || can,
                "{name}: folded to the identity under {label} but is not reported ELIGIBLE - the                  renderer short-circuits on eligibility, so such a program would never be asked"
            );
            if !can {
                continue;
            }
            let show = |v: [Option<f32>; 4]| {
                v.iter()
                    .map(|c| c.map_or_else(|| "?".into(), |x| format!("{x}")))
                    .collect::<Vec<String>>()
                    .join(",")
            };
            let shape = match fragment_colour_terms(&bytes, &u) {
                Some((g, f)) => format!("{label}: G=[{}] F=[{}]{}", show(g), show(f), if verdict { "  ELIDED" } else { "" }),
                None => format!("{label}: refused (not linear / control flow this fold does not model)"),
            };
            *shapes.entry(shape).or_default() += 1;
        }
    }
    println!("
-- colour-no-op fold over {frags} fragment blobs: {eligible} could EVER be an identity --");
    for (shape, n) in &shapes {
        println!("  {n:<4} {shape}");
    }
}

/// Write ONE linked WGSL module per FRAGMENT blob to `VITASLOP_GXP_WGSL_OUT` - the first vertex
/// in the corpus that links with it, which is enough for any question about the FRAGMENT entry.
///
/// # What this is for: the only Tint check that does not cost a play session
/// naga accepts WGSL that Tint refuses, and the difference is not academic - a `dpdx` in
/// non-uniform control flow compiles on the desktop and kills the browser's run worker
/// [[vitaslop-tint-rejects-what-naga-accepts]]. The corpus tests validate with naga, and the only
/// other Tint in reach is a whole browser replay of the title that binds the pair. This writes
/// every module to disk in a second, so a directory of them can be handed to Chrome's own
/// `createShaderModule` - one page, one device, `createShaderModule` per file and the
/// compilation info read back - and every emitted module checked at once, whether or not any
/// recipe reaches the draw that uses it.
///
/// The vertex is whichever one links, so the VERTEX entry in these files is not necessarily one
/// the title ever pairs with that fragment. That is deliberate and stated: this answers questions
/// about the fragment entry, and a pair a title really draws is `print_linked_pair_wgsl`.
#[test]
#[ignore = "needs a captured corpus (game bytes); set VITASLOP_GXP_CORPUS + VITASLOP_GXP_WGSL_OUT"]
fn write_every_linked_pair_wgsl() {
    let Some(dir) = corpus_dir() else { return };
    let Some(out) = std::env::var_os("VITASLOP_GXP_WGSL_OUT") else {
        println!("set VITASLOP_GXP_WGSL_OUT to a directory");
        return;
    };
    let out = PathBuf::from(out);
    std::fs::create_dir_all(&out).expect("create the output directory");
    let all = blobs(&dir);
    let verts: Vec<_> = all
        .iter()
        .filter(|(_, b)| matches!(Program::parse(b).map(|p| p.kind), Ok(ProgramKind::Vertex)))
        .collect();
    let (mut wrote, mut unlinkable) = (0usize, 0usize);
    for (fname, fb) in &all {
        if !matches!(Program::parse(fb).map(|p| p.kind), Ok(ProgramKind::Fragment)) {
            continue;
        }
        let linked = verts.iter().find_map(|(vn, vb)| link_programs(vb, fb).ok().map(|l| (vn, l)));
        match linked {
            Some((vn, l)) => {
                let path = out.join(format!("{fname}.wgsl"));
                std::fs::write(&path, format!("// {fname} linked with {vn}\n{}", l.wgsl))
                    .expect("write the module");
                wrote += 1;
            }
            None => {
                unlinkable += 1;
                println!("  no vertex in this corpus links with {fname}");
            }
        }
    }
    println!("\n-- wrote {wrote} linked modules to {}; {unlinkable} fragment blobs link with no vertex here --", out.display());
}

/// Which +0x78 entries name a container index that is NOT a guest-bindable uniform buffer?
///
/// # The question, and why the answer is a defect rather than a curiosity
/// `Container`'s own doc gives the format's fixed numbering: **0..13 are the ordinary uniform
/// buffers, 14 the DEFAULT uniform buffer, 15 TEXTURE, 16 LITERAL, 17 SCRATCH, 18 THREAD,
/// 19 DATA**. `sceGxmSet{Vertex,Fragment}UniformBuffer` takes an index in 0..13 - the runtime's
/// `MAX_UNIFORM_BUFFERS` - so an entry naming 15 or above names a block the DRIVER owns and the
/// guest cannot bind. A window resolved for one can therefore NEVER be fed: the capture reads
/// the guest's binding table at that index, finds nothing, withholds every window the program
/// has and drops the draw, for the whole life of the title.
///
/// `sa_uniform_buffers` already applies this rule (`if buffer_index >= 14 { continue }`, with
/// the comment "14 upward are the default buffer and the driver's own blocks").
/// `resolve_mem_windows` does not, which is the disagreement this measures.
#[test]
#[ignore = "needs a captured corpus (game bytes); set VITASLOP_GXP_CORPUS"]
fn which_plus_78_entries_name_a_driver_block_not_a_guest_buffer() {
    let Some(dir) = corpus_dir() else {
        eprintln!("VITASLOP_GXP_CORPUS not set - nothing to analyse");
        return;
    };
    let name_of = |i: u32| match i {
        0..=13 => "ordinary uniform buffer",
        14 => "DEFAULT uniform buffer",
        15 => "TEXTURE (driver)",
        16 => "LITERAL (driver)",
        17 => "SCRATCH (driver)",
        18 => "THREAD (driver)",
        19 => "DATA (driver)",
        _ => "UNKNOWN",
    };
    let mut by_index: BTreeMap<u32, Vec<String>> = BTreeMap::new();
    let (mut parsed, mut with_windows) = (0usize, 0usize);
    for (name, bytes) in blobs(&dir) {
        let kind = if name.starts_with("frag") { ProgramKind::Fragment } else { ProgramKind::Vertex };
        let Ok(p) = Program::parse(&bytes) else { continue };
        parsed += 1;
        let windows = match kind {
            ProgramKind::Vertex => vitaslop_gxp_shader::mem_windows_for_vertex_blob(&bytes),
            ProgramKind::Fragment => vitaslop_gxp_shader::mem_windows_for_fragment_blob(&bytes),
        };
        if windows.is_empty() {
            continue;
        }
        with_windows += 1;
        for w in &windows {
            if w.buffer_index <= 14 {
                continue;
            }
            // Everything a fix would need to know, per offending window.
            let has_container = p.containers.iter().any(|c| u32::from(c.index) == w.buffer_index);
            by_index.entry(w.buffer_index).or_default().push(format!(
                "{name}: buffer {} = {}, {} bytes at sa[{}] (+{}), container present: {}, \
                 literals declared: {}, this blob's other windows: {:?}",
                w.buffer_index,
                name_of(w.buffer_index),
                w.bytes,
                w.base_sa,
                w.base_offset,
                has_container,
                p.literals.len(),
                windows.iter().map(|o| o.buffer_index).collect::<Vec<_>>()
            ));
        }
    }
    println!(
        "{parsed} blobs parsed, {with_windows} resolve at least one memory window.",
    );
    if by_index.is_empty() {
        println!("NONE of them names a driver block - every window is a guest-bindable buffer.");
        return;
    }
    let total: usize = by_index.values().map(|v| v.len()).sum();
    println!("{total} window(s) name a DRIVER block, which the guest cannot bind:");
    for (index, rows) in &by_index {
        println!("  index {index} ({}) - {} window(s)", name_of(*index), rows.len());
        for r in rows {
            println!("    {r}");
        }
    }
}

/// Everything a +0x78 entry naming a DRIVER block needs, for one named blob: the container
/// table, the raw +0x78 entries, the literal table, and every memory load with the SA register
/// it chases. `PROBE_BLOB=<stem>`.
///
/// # Why raw bytes and not a verdict
/// Two readings explain a window on container 16 equally well from the resolved form alone -
/// the driver really does hand the program a pointer to its own literal pool, or the +0x78
/// field this code reads as a buffer index is something else in these blobs. Only the entries
/// themselves, beside what the code loads through the register they place, separate them.
#[test]
#[ignore = "needs a captured corpus (game bytes); set VITASLOP_GXP_CORPUS"]
fn probe_a_driver_block_binding() {
    let Some(dir) = corpus_dir() else { return };
    let want = std::env::var("PROBE_BLOB").unwrap_or_default();
    if want.is_empty() {
        eprintln!("set PROBE_BLOB=<file stem or content hash>");
        return;
    }
    for (name, bytes) in blobs(&dir) {
        if !blob_matches(&name, &bytes, &want) {
            continue;
        }
        let Ok(p) = Program::parse(&bytes) else {
            println!("{name}: PARSE FAILED");
            continue;
        };
        println!("\n===== {name} ({:?}) temp_regs {} default_uniform_regs {}",
            p.kind, p.temp_reg_count, p.default_uniform_regs);
        for c in &p.containers {
            println!("  container {:2} base_sa {:3} size_regs {:3}", c.index, c.base_sa, c.size_regs);
        }
        for b in &p.uniform_buffer_bindings {
            println!("  +0x78 entry: buffer_index {:2} data_slot {:2}", b.buffer_index, b.data_slot);
        }
        for pm in &p.parameters {
            println!("  param {:?} resource_index {} array_size {} name {:?}",
                pm.category, pm.resource_index, pm.array_size, pm.name);
        }
        println!("  literals ({}):", p.literals.len());
        for (reg, v) in &p.literals {
            println!("    sa[{reg}] = {v:#010x} ({})", f32::from_bits(*v));
        }
        for (reg, unit) in &p.texture_control {
            println!("  texture control sa[{reg}] -> unit {unit}");
        }
        // >>> THE SAME TWO DECODES `mem_windows_for_blob` RESOLVES AGAINST, not a recompiled
        // shader. The recompile REWRITES loads, so printing its instructions answers a
        // different question than the one the window resolution asked - and answering the
        // wrong one here read as "nothing loads through that pointer" when something does.
        let primary = vitaslop_gxp_shader::usse::decode_shader(&p);
        let secondary = vitaslop_gxp_shader::usse::decode_secondary_shader(&p);
        for (label, sh) in [("PRIMARY", &primary), ("SECONDARY", &secondary)] {
            for i in &sh.instrs {
                if let Op::MemLoad { elements, offset_bytes } = i.op {
                    println!(
                        "  {label} MemLoad ptr {:?} elements {elements} offset {offset_bytes} extra_srcs {} -> {:?}",
                        i.srcs.first().map(|s| (s.bank, s.index)),
                        i.srcs.len().saturating_sub(1),
                        i.dest.as_ref().map(|d| (d.bank, d.index))
                    );
                }
            }
        }
        for w in match p.kind {
            ProgramKind::Fragment => vitaslop_gxp_shader::mem_windows_for_fragment_blob(&bytes),
            ProgramKind::Vertex => vitaslop_gxp_shader::mem_windows_for_vertex_blob(&bytes),
        } {
            println!("  RESOLVED WINDOW {w:?}");
        }
    }
}

/// WHICH INDEX-REGISTER SCALE LANDS AN INDEXED SA READ ON A REGISTER THE PROGRAM POPULATES?
///
/// The index register counts REGISTER PAIRS (`idx = (src + addend) * 2`) on the evidence of one
/// title's corner table; a football title's CROWD needs 1, and both cannot be right. The
/// question is decidable without a render: an indexed read of `sa[idx + base]` can only be
/// meaningful if the register it names is one the driver actually loads - a container LITERAL,
/// a uniform register, or one a memory-window load fills. A scale that puts every reachable
/// index ABOVE everything the program declares is reading uninitialised scratch, which no
/// shipped shader does on purpose.
///
/// Prints, per program: the addend, the read's base, the literal register span, the uniform
/// register count, and where scale 1 and scale 2 land for `src = 0`.
#[test]
#[ignore = "needs a captured corpus (game bytes); set VITASLOP_GXP_CORPUS"]
fn which_index_scale_lands_inside_the_declared_layout() {
    use vitaslop_gxp_shader::ir::{Bank, Op};
    let Some(dir) = corpus_dir() else { return };
    let (mut n, mut ok1, mut ok2) = (0usize, 0usize, 0usize);
    for (name, bytes) in blobs(&dir) {
        let Ok(p) = Program::parse(&bytes) else { continue };
        let lit_lo = p.literals.iter().map(|(r, _)| *r).min();
        let lit_hi = p.literals.iter().map(|(r, _)| *r).max();
        let uni = p.secondary_reg_count as u32;
        for shader in [
            vitaslop_gxp_shader::usse::decode_shader(&p),
            vitaslop_gxp_shader::usse::decode_secondary_shader(&p),
        ] {
            let mut pending: Option<i32> = None;
            for i in &shader.instrs {
                if let Op::LoadIndex { addend, to_index: true, .. } = i.op {
                    pending = Some(addend);
                    continue;
                }
                let Some(addend) = pending else { continue };
                let Some(src) = i.srcs.iter().find(|o| o.bank == Bank::Indexed) else { continue };
                let base = vitaslop_gxp_shader::ir::indexed_offset(src.index);
                let sub = vitaslop_gxp_shader::ir::indexed_sub_bank(src.index);
                if sub != Bank::SecondaryAttr {
                    continue;
                }
                let (s1, s2) = (addend + base as i32, addend * 2 + base as i32);
                let inside = |v: i32| {
                    v >= 0
                        && (lit_lo.is_some_and(|lo| v as u32 >= lo && v as u32 <= lit_hi.unwrap_or(0))
                            || (v as u32) < uni)
                };
                n += 1;
                ok1 += usize::from(inside(s1));
                ok2 += usize::from(inside(s2));
                println!(
                    "  {name}: addend {addend} base {base} literals {:?}..{:?} uniform_regs {uni} -> scale1 sa[{s1}] {} | scale2 sa[{s2}] {}",
                    lit_lo, lit_hi,
                    if inside(s1) { "INSIDE" } else { "outside" },
                    if inside(s2) { "INSIDE" } else { "outside" },
                );
                pending = None;
            }
        }
    }
    println!("
-- {n} indexed SA reads: scale 1 lands inside on {ok1}, scale 2 on {ok2} --");
}

/// EVERY INDEXED SA READ MUST NAME A REGISTER THE PROGRAM POPULATES. A GUARD, NOT A CENSUS.
///
/// This is the invariant the index-register decode broke, and it is checkable without a render,
/// without a game and without hardware: an indexed read of `sa[idx + base]` is meaningful only
/// if some index the program can produce names a register the driver actually loads - a
/// container LITERAL, a uniform, or one a memory-window load fills. When the SOURCE field was
/// read as four bits under a hardcoded bank, and again when the index was scaled by two, the
/// reachable range sat entirely ABOVE everything the program declares, which on hardware is
/// uninitialised scratch [[vitaslop-an-unwritten-sa-register-is-uninitialised-scratch]] and on
/// screen was a stadium of collapsed crowd sprites.
///
/// A title taking a whole session to surface that is the failure mode this test exists to end:
/// the corpus knows the answer in a second. Skipped (not failed) when no corpus is configured,
/// like every other test here.
#[test]
#[ignore = "needs a captured corpus (game bytes); set VITASLOP_GXP_CORPUS"]
fn every_indexed_sa_read_names_a_register_the_program_populates() {
    use vitaslop_gxp_shader::ir::{Bank, Op};
    let Some(dir) = corpus_dir() else { return };
    let scale = vitaslop_gxp_shader::module::index_register_scale();
    let mut bad: Vec<String> = Vec::new();
    let mut checked = 0usize;
    for (name, bytes) in blobs(&dir) {
        let Ok(p) = Program::parse(&bytes) else { continue };
        let lit_lo = p.literals.iter().map(|(r, _)| *r).min();
        let lit_hi = p.literals.iter().map(|(r, _)| *r).max();
        let uni = p.secondary_reg_count as u32;
        for shader in [
            vitaslop_gxp_shader::usse::decode_shader(&p),
            vitaslop_gxp_shader::usse::decode_secondary_shader(&p),
        ] {
            let mut pending: Option<i32> = None;
            for i in &shader.instrs {
                if let Op::LoadIndex { addend, to_index: true, .. } = i.op {
                    pending = Some(addend);
                    continue;
                }
                let Some(addend) = pending else { continue };
                let Some(src) = i.srcs.iter().find(|o| o.bank == Bank::Indexed) else { continue };
                if vitaslop_gxp_shader::ir::indexed_sub_bank(src.index) != Bank::SecondaryAttr {
                    continue;
                }
                let base = vitaslop_gxp_shader::ir::indexed_offset(src.index) as i32;
                // The index the program produces is not known statically, so the test asks the
                // weakest honest question: does the read land inside the layout for ANY index
                // the register file can hold? A reading that fails even that is reading nothing.
                let reachable = (0..=127i32).any(|src_v| {
                    let at = (src_v + addend) * scale + base;
                    at >= 0
                        && (lit_lo.is_some_and(|lo| at as u32 >= lo && at as u32 <= lit_hi.unwrap_or(0))
                            || (at as u32) < uni)
                });
                checked += 1;
                if !reachable {
                    bad.push(format!(
                        "{name}: indexed sa read base {base} addend {addend} scale {scale} reaches                          nothing the program declares (literals {lit_lo:?}..{lit_hi:?}, uniform regs {uni})"
                    ));
                }
                pending = None;
            }
        }
    }
    assert!(bad.is_empty(), "{checked} indexed SA reads checked, {} unreachable:
  {}", bad.len(), bad.join("
  "));
    println!("-- {checked} indexed SA reads, every one reaches the declared layout at scale {scale} --");
}

/// THE BLAST RADIUS of reading an index load's SOURCE as six bits under bit 34's bank: how many
/// index-register loads does it MOVE, and in which programs?
///
/// The old reading took four bits and hardcoded the PrimaryAttr bank, so the two agree wherever
/// bits [19:18] are 0 and bit 34 is 1 - which is every word the decode tests pin. A word with
/// bit 34 CLEAR names a temporary register instead, and that is the entire set this changes.
#[test]
#[ignore = "needs a captured corpus (game bytes); set VITASLOP_GXP_CORPUS"]
fn which_index_loads_the_six_bit_source_reading_moves() {
    use vitaslop_gxp_shader::ir::Op;
    let Some(dir) = corpus_dir() else { return };
    let f = |w: u64, hi: u32, lo: u32| ((w >> lo) & ((1u64 << (hi - lo + 1)) - 1)) as u32;
    let (mut total, mut moved) = (0usize, 0usize);
    let mut names: Vec<String> = Vec::new();
    for (name, bytes) in blobs(&dir) {
        let Ok(p) = Program::parse(&bytes) else { continue };
        for shader in [
            vitaslop_gxp_shader::usse::decode_shader(&p),
            vitaslop_gxp_shader::usse::decode_secondary_shader(&p),
        ] {
            for i in &shader.instrs {
                if !matches!(i.op, Op::LoadIndex { to_index: true, .. }) {
                    continue;
                }
                total += 1;
                let same_bank = f(i.raw, 34, 34) == 1;
                let same_number = f(i.raw, 19, 18) == 0;
                if !(same_bank && same_number) {
                    moved += 1;
                    if !names.contains(&name) {
                        names.push(name.clone());
                    }
                }
            }
        }
    }
    println!(
        "
-- {total} index-register loads, {moved} MOVED by the six-bit reading, in {} program(s): {:?} --",
        names.len(),
        names
    );
}

/// **WHERE THE SHADER PIPELINE'S CPU TIME GOES, PER PROGRAM, OVER THE WHOLE CORPUS.**
///
/// The pipeline runs once per program rather than once per draw, but "once" happens while a
/// title is drawing - which is why it shows up as a HITCH rather than as a frame rate, and why
/// a mean over a session hides it entirely.
///
/// # Why measure this at all, when the notes say the compile cost is the backend
///
/// They do, and that is the point: `shader-compile-cost-is-the-backend-not-the-parse` says the
/// expensive half is the WGSL compiler, not our parse. That claim has never been checked with a
/// number on OUR half - it was inferred from where the wall clock went - and "our half is small"
/// is exactly the kind of belief that quietly stops being true. This prints our half's
/// distribution so the next person can compare it against the backend's rather than assume.
///
/// # What it deliberately does NOT do
///
/// It does not rank programs by emitted operation COUNT as a stand-in for GPU cost.
/// `operator-count-is-not-browser-time` and `desktop-cannot-price-a-count-win` are both
/// measured refutations of that substitution, and a census that made it would read as a
/// performance finding while measuring nothing.
///
/// It also prints p50/p90/MAX rather than a mean: a mean cannot see the hitch, and the hitch is
/// the whole phenomenon (`a-mean-cannot-see-a-dip-fuel-names-its-owner`).
///
/// ```text
/// VITASLOP_GXP_CORPUS=<dir> cargo test --release -p vitaslop-gxp-shader \
///   --test corpus -- --ignored --nocapture where_the_shader_pipeline_spends
/// ```
#[test]
#[ignore = "needs a captured corpus (game bytes); set VITASLOP_GXP_CORPUS"]
fn where_the_shader_pipeline_spends_its_cpu_time_per_program() {
    use std::time::Instant;
    let Some(dir) = corpus_dir() else { return };

    // Enough repeats that one program's answer is not a single clock tick, and few enough that
    // a 1,151-blob corpus still finishes in seconds.
    const REPS: u32 = 10;

    /// One measured stage: every per-program sample, in nanoseconds.
    struct Stage {
        name: &'static str,
        ns: Vec<u128>,
        worst: (u128, String),
    }
    impl Stage {
        fn new(name: &'static str) -> Stage {
            Stage { name, ns: Vec::new(), worst: (0, String::new()) }
        }
        fn push(&mut self, ns: u128, who: &str) {
            if ns > self.worst.0 {
                self.worst = (ns, who.to_string());
            }
            self.ns.push(ns);
        }
        /// The percentile at `q` (0..1), in microseconds. Sorts in place.
        fn us(&mut self, q: f64) -> f64 {
            if self.ns.is_empty() {
                return 0.0;
            }
            self.ns.sort_unstable();
            let i = ((self.ns.len() - 1) as f64 * q).round() as usize;
            self.ns[i] as f64 / 1000.0
        }
        fn total_ms(&self) -> f64 {
            self.ns.iter().sum::<u128>() as f64 / 1_000_000.0
        }
    }

    let mut parse = Stage::new("Program::parse");
    let mut decode = Stage::new("usse::decode_shader");
    let mut emit = Stage::new("wgsl::emit_body");
    let mut whole = Stage::new("recompile_* (all three)");
    let (mut programs, mut refused) = (0usize, 0usize);
    let mut emitted_bytes = 0usize;

    for (name, bytes) in blobs(&dir) {
        let Ok(program) = Program::parse(&bytes) else { continue };
        let kind = program.kind;

        // The stages, each timed on its own so the split is measured rather than subtracted -
        // a subtracted stage carries every other stage's noise.
        let t = Instant::now();
        for _ in 0..REPS {
            let _ = std::hint::black_box(Program::parse(&bytes));
        }
        parse.push(t.elapsed().as_nanos() / u128::from(REPS), &name);

        let t = Instant::now();
        for _ in 0..REPS {
            let _ = std::hint::black_box(vitaslop_gxp_shader::usse::decode_shader(&program));
        }
        decode.push(t.elapsed().as_nanos() / u128::from(REPS), &name);

        let shader = vitaslop_gxp_shader::usse::decode_shader(&program);
        if vitaslop_gxp_shader::wgsl::emit_body(&shader).is_ok() {
            let t = Instant::now();
            for _ in 0..REPS {
                let _ = std::hint::black_box(vitaslop_gxp_shader::wgsl::emit_body(&shader));
            }
            emit.push(t.elapsed().as_nanos() / u128::from(REPS), &name);
            if let Ok(body) = vitaslop_gxp_shader::wgsl::emit_body(&shader) {
                emitted_bytes += body.len();
            }
        }

        let t = Instant::now();
        let mut ok = true;
        for _ in 0..REPS {
            let r = match kind {
                ProgramKind::Vertex => {
                    vitaslop_gxp_shader::recompile_vertex(&bytes).map(|r| r.wgsl_body.len())
                }
                ProgramKind::Fragment => {
                    vitaslop_gxp_shader::recompile_fragment(&bytes).map(|r| r.wgsl_body.len())
                }
            };
            ok = r.is_ok();
            let _ = std::hint::black_box(r);
        }
        if ok {
            whole.push(t.elapsed().as_nanos() / u128::from(REPS), &name);
        } else {
            refused += 1;
        }
        programs += 1;
    }

    println!("\n=== SHADER PIPELINE CPU COST over {programs} program(s) ({refused} the emitter refuses) ===");
    println!("    {} of emitted WGSL in total\n", human_bytes(emitted_bytes));
    println!("  {:<26} {:>10} {:>10} {:>10} {:>10}", "stage", "p50 us", "p90 us", "max us", "total ms");
    for s in [&mut parse, &mut decode, &mut emit, &mut whole] {
        let (p50, p90, max, total) = (s.us(0.50), s.us(0.90), s.us(1.0), s.total_ms());
        println!("  {:<26} {p50:>10.1} {p90:>10.1} {max:>10.1} {total:>10.1}", s.name);
    }
    for s in [&parse, &decode, &emit, &whole] {
        if !s.worst.1.is_empty() {
            println!("    worst {:<22} {:>8.1} us  {}", s.name, s.worst.0 as f64 / 1000.0, s.worst.1);
        }
    }
    println!(
        "\n  >>> READ THIS AGAINST THE BACKEND, NOT ON ITS OWN. These are OUR stages; the WGSL\n      \
         compiler that consumes the emitted text is a separate and (per the notes) larger cost.\n      \
         What this says is how much of a first-sight hitch is ours to remove."
    );
}

/// Bytes in a form a reader can compare at a glance.
fn human_bytes(n: usize) -> String {
    if n >= 1 << 20 {
        format!("{:.1} MB", n as f64 / (1 << 20) as f64)
    } else {
        format!("{:.1} KB", n as f64 / 1024.0)
    }
}

/// Census for the repeating-VPCK DESTINATION's SMLSI slot: every straight-line read of a PA or
/// TEMP lane that nothing earlier in the program wrote and no attribute declares. Run it once per
/// `VITASLOP_GXP_PACK_DEST_SLOT` and compare - a reading of the destination slot that is right
/// removes reads of unwritten registers (the consumer of the second iteration finds its value)
/// and never adds them. Coarse on purpose: lanes are 32-bit, a 16-bit half marks its whole lane,
/// and every swizzle entry of every source counts, so ONLY THE DIFFERENCE between two runs
/// means anything.
#[test]
#[ignore = "needs a captured corpus (game bytes); set VITASLOP_GXP_CORPUS"]
fn census_reads_of_unwritten_registers() {
    use vitaslop_gxp_shader::ParamCategory;
    let Some(dir) = corpus_dir() else {
        eprintln!("VITASLOP_GXP_CORPUS not set - nothing to analyse");
        return;
    };
    let slot = std::env::var("VITASLOP_GXP_PACK_DEST_SLOT").unwrap_or_else(|_| "default".into());
    let (mut blobs_n, mut total) = (0usize, 0usize);
    for (name, bytes) in blobs(&dir) {
        let Ok(p) = Program::parse(&bytes) else { continue };
        blobs_n += 1;
        let sh = vitaslop_gxp_shader::usse::decode_shader(&p);
        let mut pa_w = [false; 256];
        let mut t_w = [false; 256];
        for prm in &p.parameters {
            if prm.category == ParamCategory::Attribute && prm.resource_index >= 0 {
                let n = u32::from(prm.component_count.max(1)) * prm.array_size.max(1);
                for k in 0..n {
                    if let Some(w) = pa_w.get_mut(prm.resource_index as usize + k as usize) {
                        *w = true;
                    }
                }
            }
        }
        let mut bad = 0usize;
        for instr in &sh.instrs {
            let half = instr.half_precision
                || matches!(instr.op, Op::PackToInt { .. } | Op::PackIntCopy { .. } | Op::LoadIndex { .. });
            for s in &instr.srcs {
                let arr = match s.bank {
                    Bank::PrimaryAttr => &pa_w,
                    Bank::Temp => &t_w,
                    _ => continue,
                };
                let mut lanes = std::collections::BTreeSet::new();
                for c in 0..4 {
                    if !instr.write_mask[c] {
                        continue;
                    }
                    let sw = u32::from(s.swizzle[c].min(3));
                    let lane = u32::from(s.index) + if half { sw / 2 } else { sw };
                    lanes.insert(lane);
                }
                for l in lanes {
                    if !arr.get(l as usize).copied().unwrap_or(true) {
                        bad += 1;
                    }
                }
            }
            if let Some(d) = instr.dest {
                let arr = match d.bank {
                    Bank::PrimaryAttr => &mut pa_w,
                    Bank::Temp => &mut t_w,
                    _ => continue,
                };
                for c in 0..4 {
                    if instr.write_mask[c] {
                        let lane = u32::from(d.index) + if half { c as u32 / 2 } else { c as u32 };
                        if let Some(w) = arr.get_mut(lane as usize) {
                            *w = true;
                        }
                    }
                }
            }
        }
        if bad > 0 {
            println!("  {name}: {bad}");
        }
        total += bad;
    }
    println!("\nslot={slot}: {blobs_n} blobs, {total} reads of an unwritten PA/TEMP lane");
}

/// For every REPEATING pack (two or more consecutive unrolled copies of one code word), is each
/// later iteration's destination READ before anything overwrites it? Run once per
/// `VITASLOP_GXP_PACK_DEST_SLOT`: the slot that governs the destination is the one under which
/// those writes are live. A dead later iteration is a value the program computed for nothing -
/// or, when it lands on an attribute lane, one it destroyed.
#[test]
#[ignore = "needs a captured corpus (game bytes); set VITASLOP_GXP_CORPUS"]
fn census_repeating_pack_destinations_live_or_dead() {
    let Some(dir) = corpus_dir() else {
        eprintln!("VITASLOP_GXP_CORPUS not set - nothing to analyse");
        return;
    };
    let slot = std::env::var("VITASLOP_GXP_PACK_DEST_SLOT").unwrap_or_else(|_| "default".into());
    let (mut live, mut dead) = (0usize, 0usize);
    for (name, bytes) in blobs(&dir) {
        let Ok(p) = Program::parse(&bytes) else { continue };
        let sh = vitaslop_gxp_shader::usse::decode_shader(&p);
        let n = sh.instrs.len();
        let mut i = 0;
        while i < n {
            let w = sh.instrs[i].raw;
            let is_pack = matches!(
                sh.instrs[i].op,
                Op::Pack { .. } | Op::PackToInt { .. } | Op::PackIntCopy { .. } | Op::PackFromInt { .. } | Op::PackUnorm8 { .. }
            );
            let mut j = i + 1;
            while j < n && sh.instrs[j].raw == w {
                j += 1;
            }
            if is_pack && j - i >= 2 {
                for k in i + 1..j {
                    let Some(d) = sh.instrs[k].dest else { continue };
                    // Read before overwritten, scanning past the whole repeat.
                    let mut verdict = "DEAD(end)";
                    'scan: for later in &sh.instrs[j..] {
                        for s in &later.srcs {
                            if s.bank == d.bank && s.index.abs_diff(d.index) <= 1 {
                                verdict = "LIVE";
                                break 'scan;
                            }
                        }
                        if let Some(ld) = later.dest
                            && ld.bank == d.bank
                            && ld.index == d.index
                            && later.write_mask.iter().filter(|m| **m).count() >= 2
                        {
                            verdict = "DEAD(overwritten)";
                            break;
                        }
                    }
                    if verdict == "LIVE" {
                        live += 1;
                    } else {
                        dead += 1;
                    }
                    println!("  {name} #{k} {:?}[{}] {verdict}", d.bank, d.index);
                }
            }
            i = j;
        }
    }
    println!("\nslot={slot}: later pack iterations LIVE {live}, DEAD {dead}");
}
