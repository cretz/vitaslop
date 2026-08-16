//! Offline analysis of a captured `.gxp` corpus.
//!
//! Point `VITASLOP_GXP_CORPUS` at a directory of `vert_*.gxp` / `frag_*.gxp` blobs (what
//! `VITASLOP_DUMP_GXP_BIN` writes) and run:
//!
//! ```text
//! VITASLOP_GXP_CORPUS=<dir> cargo test -p vitaslop-gxp-shader --test corpus -- --ignored --nocapture
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
        for &(reg, v) in &p.literals {
            println!("  LITERAL sa[{reg}] = {v:#010x}");
        }
        for &(base, unit) in &p.texture_control {
            println!("  TEXCTRL sa[{base}..{}] = texture unit {unit}", base + 4);
        }
        for it in &p.interpolants {
            println!(
                "  usage={:?} pa_base={} regs={} span={} half={} prefetch={:?} prefetch_regs={}",
                it.usage, it.pa_base, it.register_count, it.span, it.half, it.prefetch, it.prefetch_regs
            );
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
                "  PARAM {:<24} category={:?} type={:?} components={} array={} resource_index={} \
                 semantic={}.{}",
                prm.name,
                prm.category,
                prm.ptype,
                prm.component_count,
                prm.array_size,
                prm.resource_index,
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
            println!(
                "  [{i:3}] grp={:#04x} {:?} dest={:?} srcs={:?} mask={:?} half={}{}",
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

    let mut by_reason: BTreeMap<String, usize> = BTreeMap::new();
    let mut linked = 0usize;
    for (_, v) in &verts {
        for (_, f) in &frags {
            match link_programs(v, f) {
                Ok(_) => linked += 1,
                Err(e) => *by_reason.entry(format!("{e}")).or_default() += 1,
            }
        }
    }
    println!("{linked} of {} pairings link", verts.len() * frags.len());
    let mut ranked: Vec<_> = by_reason.into_iter().collect();
    ranked.sort_by(|a, b| b.1.cmp(&a.1));
    for (reason, n) in ranked.iter().take(20) {
        println!("  {n} pairings - {reason}");
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
    for (name, bytes) in blobs(&dir) {
        let Ok(p) = Program::parse(&bytes) else { continue };
        total += 1;
        for prm in &p.parameters {
            if prm.category != vitaslop_gxp_shader::container::ParamCategory::Uniform {
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
}

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
