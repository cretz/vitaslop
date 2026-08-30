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
    if name == want {
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
    ranked.sort_by(|a, b| b.1.len().cmp(&a.1.len()));
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
    use vitaslop_gxp_shader::ir::Bank;

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
                        "    {vn} + {fname}\n      vertex says {shared:?}\n      fragment says {fshared:?}"
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
    match link_programs(&v, &f) {
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
            for s in &instr.srcs {
                if s.bank != Bank::PrimaryAttr {
                    continue;
                }
                // Read the channels the instruction actually reads, mirroring the emitter: a
                // source lane is `index + channel` for the channels the write mask enables.
                for c in 0..4u32 {
                    if !instr.write_mask[c as usize] && instr.write_mask.iter().any(|&m| m) {
                        continue;
                    }
                    let r = u32::from(s.index) + c;
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
                    if let Some(end) = end {
                        if !(0..=255).contains(&end) {
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
    ranked.sort_by(|a, b| b.1 .0.cmp(&a.1 .0));
    for (reason, (n, vn, fname)) in ranked.iter().take(20) {
        println!("  {n} pairings - {reason}
      e.g. {vn} + {fname}");
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
    const KICK: f32 = 0.6180339887;

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
    const KICK: f32 = 0.6180339887;

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
        ) -> impl Fn(u8, [f32; 4]) -> Option<[f32; 4]> + '_ {
            move |_unit: u8, c: [f32; 4]| {
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
                .filter(|&i| !(!native && (lo..hi).contains(&i)))
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
    const KICK: f32 = 0.6180339887;
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
                        "{name}: {:?} at lanes {lanes:?} - {} moves only {:?} (the rest are                          written from constants)",
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
                "    {vn} + {fname}\n      vertex order   {shared:?}\n      fragment order {fshared:?}"
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
    rows.sort_by(|a, b| b.1.cmp(&a.1));
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
        let shader = vitaslop_gxp_shader::usse::decode_shader(&p);
        let mut hit = false;
        let mut smlsi_seen = false;
        for instr in &shader.instrs {
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
            "  {dirn}  {mode}    {addr_mode}    {ty}    {bank:<9}       {elems:<5}  {n:<5}  {example}"
        );
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
        println!("  {v:#03x} (unk7={} abs_op2={} strange1={} strange0={}): {n}", v >> 3, (v >> 2) & 1, (v >> 1) & 1, v & 1);
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
    use vitaslop_gxp_shader::ir::Bank;
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
                let width = u32::from(q.ptype.component_bytes().unwrap_or(4));
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
        "these programs carry their WHOLE default uniform buffer in container 14 yet still take a          pointer to it and load memory - nothing is left past the copy, so `base_offset` would          make their window zero bytes and `resolve_mem_windows` would refuse them: {:#?}",
        fully_carried_with_a_pointer
    );
    assert!(
        straddled.is_empty(),
        "the default uniform container cuts THROUGH a parameter, so the pointer's offset cannot          name it and `MemWindow::base_offset` is the wrong reading: {straddled:#?}"
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
