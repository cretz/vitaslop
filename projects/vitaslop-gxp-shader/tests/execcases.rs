//! Write an EXECUTION case per corpus blob: the emitted WGSL as a runnable compute module,
//! plus the register file [`vitaslop_gxp_shader::interp`] computes for the same inputs.
//!
//! # Why this exists
//!
//! Everything this project checks over the whole corpus today is STRUCTURAL - it links, it
//! parses, it compiles under naga, it compiles under Tint. None of that can catch a wrong
//! NUMBER, and every graphics defect chased here one title at a time was a wrong number: a
//! write mask that dropped a lane, a load offset biased by one register, an index scale of 2
//! where the hardware uses 1, a literal suppressed by a partial write. Each of those was found
//! because some title drew something visibly wrong, months after the blob was first captured.
//!
//! This turns that into a corpus-wide question that needs no title, no recipe and no frame.
//! Two independently-written implementations of the same USSE semantics already exist in this
//! crate:
//!
//! * [`vitaslop_gxp_shader::wgsl`] - a string emitter, whose output runs on the real GPU;
//! * [`vitaslop_gxp_shader::interp`] - a numeric evaluator, whose output is an `f32` register
//!   file on the CPU.
//!
//! They share the decoder and nothing else. Where they DISAGREE, one of them is wrong, and the
//! disagreement is visible without anything having to render. This test writes the cases; the
//! GPU half runs in the browser (`vitaslop-web/e2e/gxpexec.mjs`), because Tint - not naga - is
//! the compiler the product ships through, and the browser is where the pixels are.
//!
//! # What this can and cannot prove
//!
//! It CANNOT prove fidelity to the real SGX543: both halves are our own reading of the ISA, so
//! a misread that reached the decoder appears in both and this agrees with itself. That gap is
//! the conformance-app's job (an authored shader with a known intent), not this file's.
//!
//! It CAN prove that the shader the GPU actually runs computes what our own reference says the
//! program means - which is the half the pixels depend on, and the half that has been failing.
//!
//! ```text
//! VITASLOP_GXP_CORPUS=<dir> VITASLOP_GXP_CASES_OUT=<dir> \
//!   cargo test -p vitaslop-gxp-shader --test execcases -- --ignored --nocapture
//! ```

use std::collections::BTreeMap;
use std::fmt::Write as _;
use std::path::PathBuf;

use vitaslop_gxp_shader::interp::{self, RegFile};
use vitaslop_gxp_shader::ir::{Bank, BitwiseKind, Instr, Op, Operand, Predicate, Shader};
use vitaslop_gxp_shader::module::{mem_window_placements, MemWindow};
use vitaslop_gxp_shader::wgsl::{
    self as wgsl, case_render_gather, case_render_sample, case_render_sample_cube, case_tex_bytes, case_tex_varies,
    case_texture_value, case_texture_value_at, wrap_compute_module_for,
    wrap_render_case_module_for, RenderRewrites, CASE_BANK_LANES, GLOBAL_FACING,
};
use vitaslop_gxp_shader::{
    mem_windows_for_fragment_blob, mem_windows_for_vertex_blob, recompile_fragment,
    recompile_vertex, Program, ProgramKind,
};

mod common;
use common::{
    apply_plausible_inputs, best_plausible_inputs, changed_pairs, dest_lanes,
    integer_sink_operands, lane_value, mem_fetch, mem_words_for, nonzero_pairs, placements_hit,
    plausible_inputs, seeded_lane_reads, seeded_pa_baseline, src_lanes, window_base, write_case,
    written_lane_precision, PlausibleScore, Stage, PLAUSIBLE_ATTEMPTS,
    PLAUSIBLE_ATTEMPTS_WIDENED, PREC_AMBIGUOUS,
};

/// >>> THE INPUTS ONE PROGRAM'S CASE RUNS ON: the seeded fill, plus whatever plausible counts,
/// >>> indices and pointers [`best_plausible_inputs`] CHOSE for it by running the reference.
///
/// ONE statement, and it has to be: the case writer, the trace writer and the conditioning
/// census all interpret the same program, and a trace taken on different inputs from the case's
/// locates a divergence that is not there while a conditioning verdict over different inputs is
/// about a program the case never ran. Three copies of a search is three chances to answer
/// differently - the same argument `lane_value` itself carries one level down.
///
/// >>> THE SEARCH SCORES ON THE COMPUTE RIG'S STAND-IN even for a program that will end up on
/// the render rig. It is choosing an INDEX, a STRIDE and a TRIP COUNT, none of which a texture
/// decides, and building the render rig's texture environment per candidate would cost eight
/// times over for an answer that cannot change. What a case is finally WRITTEN from is the real
/// run at the call site, with that rig's own fetcher.
/// >>> DOES THIS RUN LEAVE A DENORMAL WHERE THE CASE WILL CHECK ONE?
///
/// The expectation a case carries is the reference's finished register file, and a denormal in
/// it is a value the GPU is free not to produce: WGSL permits flushing subnormals to zero, and
/// every backend the runner reaches does. See [`PlausibleScore::writes_a_denormal`] - this cost
/// a full corpus run to find, as 18 divergences that were all the same non-defect.
///
/// Every bank the grader reads, because a denormal is just as unreachable in whichever one it
/// lands in. `is_subnormal` and not a magnitude threshold: the boundary is the format's, not a
/// number chosen here.
fn writes_a_denormal(regs: &RegFile) -> bool {
    regs.r
        .iter()
        .chain(regs.o.iter())
        .chain(regs.i.iter())
        .chain(regs.pa.iter())
        .any(|v| v.is_subnormal())
}

fn chosen_case_inputs(
    shader: &Shader,
    seed: u32,
    windows: &[MemWindow],
    mem_words: &[u32],
) -> Vec<(usize, u32)> {
    best_plausible_inputs(shader, seed, |overrides| {
        let mut regs = seeded_case_regs(seed, windows, overrides);
        let baseline = seeded_pa_baseline(seed, overrides);
        // BOTH counts, because the widened band's gate needs the denominator: a program that
        // issued no load at all has no coverage to win and must not be offered the trade.
        let loads = std::cell::Cell::new((0u32, 0u32));
        let memory = |addr: u32| {
            let (total, inside) = loads.get();
            loads.set((total + 1, inside + u32::from(placements_hit(addr, windows))));
            mem_fetch(addr, windows, mem_words)
        };
        let mem_arg: Option<&dyn Fn(u32) -> u32> =
            if windows.is_empty() { None } else { Some(&memory) };
        let textures = case_textures;
        let outcome = interp::run_watching_for_nan_with_env(shader, &mut regs, &textures, mem_arg);
        let (loads_total, loads_inside) = loads.get();
        let mut score =
            PlausibleScore { loads_inside, loads_total, ..PlausibleScore::default() };
        if outcome.is_ok() {
            score.terminated = true;
            score.wrote_something = regs
                .r
                .iter()
                .chain(regs.o.iter())
                .chain(regs.i.iter())
                .any(|v| v.to_bits() != 0)
                || regs.pa.iter().zip(baseline.iter()).any(|(a, b)| a.to_bits() != b.to_bits());
            score.writes_a_denormal = writes_a_denormal(&regs);
        }
        score
    })
    .0
}

/// The register file a case STARTS from, in the one order that is correct.
///
/// Seeded fill, then the chosen overrides, then the window bases - and the order is not a
/// preference. The seed would overwrite a base placed before it, and a base is an address the
/// harness chose rather than a number the program computes, so it wins over an override on the
/// same lane. The emitted prologue puts the bases on last for the same reason, so this is the
/// two agreeing rather than a second rule.
fn seeded_case_regs(seed: u32, windows: &[MemWindow], overrides: &[(usize, u32)]) -> RegFile {
    let mut regs = RegFile::with_lanes(CASE_BANK_LANES);
    for n in 0..CASE_BANK_LANES {
        regs.pa[n] = lane_value(seed, n as u32);
        regs.sa[n] = lane_value(seed, (n + CASE_BANK_LANES) as u32);
    }
    apply_plausible_inputs(&mut regs, overrides);
    // The value BOTH wrappers pin `gxp_front_facing` to. Pinned here rather than defaulted in
    // the register file, so a caller that has no facing to give still gets the refusal.
    regs.facing = Some(true);
    for (i, win) in windows.iter().enumerate() {
        if let Some(slot) = regs.sa.get_mut(win.base_sa as usize) {
            *slot = f32::from_bits(window_base(i));
        }
    }
    regs
}

/// True when the program reads the per-fragment FACING flag, `GLOBAL[16]`.
///
/// Every OTHER global index is pipeline state neither side holds and is refused by
/// `rig_excludes`; this one is pinned to the same value by both wrappers and by the reference,
/// which is what makes such a program an ordinary case.
fn reads_the_facing_global(shader: &Shader) -> bool {
    shader
        .instrs
        .iter()
        .any(|i| i.srcs.iter().any(|s| matches!(s.bank, Bank::Global) && s.index == GLOBAL_FACING))
}

/// The name of a program's BACK-FACING case. A suffix rather than a flag inside the file,
/// because the runner prints NAMES and "the same program, the other way up" has to be readable
/// there.
fn back_name(name: &str) -> String {
    format!("{name}__backfacing")
}

fn corpus_dir() -> Option<PathBuf> {
    std::env::var_os("VITASLOP_GXP_CORPUS").map(PathBuf::from)
}

fn out_dir() -> Option<PathBuf> {
    std::env::var_os("VITASLOP_GXP_CASES_OUT").map(PathBuf::from)
}

/// Why a blob is not a case this harness can judge. Each is a REAL limitation of the compute
/// rig, not a defect - and the census of them is itself the coverage number to report.
#[derive(Clone, Copy, PartialEq, Eq, PartialOrd, Ord, Debug)]
enum Skip {
    /// The blob does not parse, or is not the kind its name claims.
    NotAProgram,
    /// The emitter refuses it (an unestablished op). Already ranked elsewhere; not new news.
    EmitRefused,
    /// A fragment-stage instruction (a gather, a derivative, a kill, a written depth) in a
    /// program that is not a FRAGMENT program. The render rig has a fragment stage and nothing
    /// else to put such a program in, and the emitter refuses a written depth outside one.
    RenderNotFragment,
    /// A fragment-stage program that samples a 3D or a RAW integer texture. The render rig
    /// binds a flat and a CUBE stand-in set per unit and quantises a 2D coordinate or a cube
    /// DIRECTION; a volume coordinate needs a third texture set and a third quantiser, and a
    /// raw binding is read with textureLoad rather than sampled at all. Binding a flat texture
    /// for either would sample something the program did not ask for.
    RenderTexKind,
    /// The render rig's rewrite did not reach every sample, gather or discard the instruction
    /// stream contains. An unrewritten sample is not a compile error - it is a real
    /// `textureSample` on an unquantised coordinate, whose texel the hardware's subtexel
    /// rounding may pick differently from the reference. Refused rather than shipped.
    RenderUnrewritten,
    /// A sample whose emitted statement did not match the shape the constant stand-in parses,
    /// so the module would reference a binding the wrapper never declared.
    UnparsedSample,
    /// A 0xE8 guest-memory load whose window the linker cannot resolve. The shipped path refuses
    /// the same program (the draw is dropped rather than fed zeroes), so there is nothing to
    /// check here either.
    MemWindowUnresolved,
    /// A hardware GLOBAL register neither side holds any state for.
    ///
    /// `GLOBAL[16]`, the per-fragment FACING flag, is NOT this any more: both wrappers pin
    /// `gxp_front_facing` to `true` and the reference is pinned to the same value, so such a
    /// program is an ordinary case. Every other index is pipeline state, and a fabricated value
    /// for one would be a number the hardware never produced.
    GlobalRegister,
    /// The interpreter refuses it, naming the op. The EMITTER accepted it, so this is a
    /// coverage gap in the reference, and it is reported as one - except for the step-budget
    /// entry, which is a seed artifact and is called out as such where the census prints it.
    InterpRefused,
}

impl Skip {
    fn label(self) -> &'static str {
        match self {
            Skip::NotAProgram => "not a program",
            Skip::EmitRefused => "emitter refused",
            Skip::RenderNotFragment => "a fragment-stage instruction outside a fragment program",
            Skip::RenderTexKind => "a fragment-stage program sampling a 3D or raw texture",
            Skip::RenderUnrewritten => "a sample, gather or discard the render rig did not rewrite",
            Skip::UnparsedSample => "a sample the constant stand-in could not parse",
            Skip::MemWindowUnresolved => "guest memory load whose window does not resolve",
            Skip::GlobalRegister => "global register",
            Skip::InterpRefused => "interpreter refused (emitter did not)",
        }
    }
}

/// The rig's own exclusions, decided from the IR before either side runs.
/// The stand-in texture the reference fetches, in whichever form the emitter is emitting.
///
/// The arm is read HERE rather than at each of the three call sites, because a call site that
/// forgot it would hand the reference a constant while the module sampled a varying stand-in -
/// and the differential would report that as a translation defect on every program that samples
/// anything.
fn case_textures(unit: u8, coord: [f32; 4], lod: interp::TexLodArg) -> Option<[f32; 4]> {
    Some(if case_tex_varies() {
        case_texture_value_at(unit, [coord[0], coord[1], coord[2]], lod)
    } else {
        case_texture_value(unit)
    })
}

fn rig_excludes(shader: &Shader) -> Option<Skip> {
    for instr in &shader.instrs {
        for s in &instr.srcs {
            // >>> THE FACING FLAG IS NOT A GAP ANY MORE. Both wrappers pin
            // `gxp_front_facing` to `true` - a compute dispatch has no facing, and the render
            // rig draws one triangle - so the reference is pinned to the same value and the
            // program is a case like any other. What that does NOT do is exercise the
            // back-facing arm, which is why the coverage line says so.
            //
            // Every OTHER global stays refused: it is pipeline state neither side holds, and a
            // fabricated value is a number the hardware never produced.
            if matches!(s.bank, Bank::Global) && s.index != GLOBAL_FACING {
                return Some(Skip::GlobalRegister);
            }
        }
    }
    None
}

/// Does this program need the RENDER rig - a real fragment stage - rather than the compute one?
///
/// Four instruction families cannot run in a compute dispatch at all, and each was excluded by
/// name for as long as the compute rig was the only one:
///
///  * a GATHER returns a four-texel FOOTPRINT, so it needs a real texture: the whole point of
///    the instruction is that the four texels differ, which no stand-in can represent;
///  * a screen-space DERIVATIVE differences a value across the rasteriser's 2x2 quad;
///  * `kill` and `depthf` are fragment pipeline state.
///
/// A program with none of them stays on the compute rig, deliberately - see the render rig's
/// own note on why its UV sensitivity is coarser than the varying stand-in's.
fn needs_fragment_stage(shader: &Shader) -> bool {
    shader.instrs.iter().any(|i| {
        matches!(i.op, Op::TexGather { .. } | Op::Dsx | Op::Dsy | Op::Kill | Op::DepthF)
    })
}

/// How many instructions of each family the render rig's textual rewrite must have reached.
fn render_rewrite_targets(shader: &Shader) -> RenderRewrites {
    let count = |f: &dyn Fn(&Op) -> bool| shader.instrs.iter().filter(|i| f(&i.op)).count();
    RenderRewrites {
        samples: count(&|o| matches!(o, Op::Tex { .. })),
        gathers: count(&|o| matches!(o, Op::TexGather { .. })),
        kills: count(&|o| matches!(o, Op::Kill)),
    }
}

/// Print every guest address ONE program's loads resolve to, and whether each lands inside a
/// bound window - the seam where the reference and the emitted module can disagree without
/// either of them saying so.
///
/// A load that falls outside every window reads ZERO on both sides by design, so a program whose
/// addresses all miss is not "checked and agreeing" - it is a program whose loads never
/// happened. That is invisible in the differential's pass/fail and is exactly what this prints.
///
/// ```text
/// VITASLOP_GXP_CORPUS=<dir> VITASLOP_GXP_BLOB=<stem> \
///   cargo test -p vitaslop-gxp-shader --test execcases -- --ignored --nocapture memory
/// ```
#[test]
#[ignore = "needs a captured corpus and VITASLOP_GXP_BLOB"]
fn print_one_programs_memory_addresses() {
    let (Some(dir), Ok(want)) = (corpus_dir(), std::env::var("VITASLOP_GXP_BLOB")) else {
        eprintln!("set VITASLOP_GXP_CORPUS and VITASLOP_GXP_BLOB");
        return;
    };
    let path = dir.join(format!("{}.gxp", want.trim()));
    let Ok(bytes) = std::fs::read(&path) else {
        eprintln!("no such blob: {}", path.display());
        return;
    };
    let program = Program::parse(&bytes).expect("parse");
    let kind = program.kind;
    let (shader, _body) = match kind {
        ProgramKind::Vertex => recompile_vertex(&bytes).map(|r| (r.shader, r.wgsl_body)),
        ProgramKind::Fragment => recompile_fragment(&bytes).map(|r| (r.shader, r.wgsl_body)),
    }
    .expect("recompile");
    let windows = match kind {
        ProgramKind::Vertex => mem_windows_for_vertex_blob(&bytes),
        ProgramKind::Fragment => mem_windows_for_fragment_blob(&bytes),
    };
    let seed = (program.hash ^ (program.hash >> 32)) as u32 | 1;
    let words = mem_words_for(seed, &windows);

    println!("{want}: {kind:?}, {} instrs, {} window(s)", shader.instrs.len(), windows.len());
    for (i, w) in windows.iter().enumerate() {
        let at = mem_window_placements(&windows)[i];
        println!(
            "  window {i}: base {:#010x} .. {:#010x} ({} bytes, {} words at gxp_mem word {}), pointer in sa[{}], buffer_index {}, base_offset {}",
            window_base(i),
            window_base(i) + w.bytes,
            w.bytes,
            at.words,
            at.first_word,
            w.base_sa,
            w.buffer_index,
            w.base_offset
        );
    }

    let mut regs = RegFile::with_lanes(CASE_BANK_LANES);
    for n in 0..CASE_BANK_LANES {
        regs.pa[n] = lane_value(seed, n as u32);
        regs.sa[n] = lane_value(seed, (n + CASE_BANK_LANES) as u32);
    }
    for (i, win) in windows.iter().enumerate() {
        if let Some(slot) = regs.sa.get_mut(win.base_sa as usize) {
            *slot = f32::from_bits(window_base(i));
        }
    }
    let seen = std::cell::RefCell::new(Vec::<(u32, bool)>::new());
    let textures = case_textures;
    let memory = |addr: u32| {
        let inside = mem_window_placements(&windows)
            .iter()
            .enumerate()
            .any(|(i, at)| addr >= window_base(i) && ((addr - window_base(i)) >> 2) < at.words);
        seen.borrow_mut().push((addr, inside));
        mem_fetch(addr, &windows, &words)
    };
    let mem_arg: Option<&dyn Fn(u32) -> u32> = if windows.is_empty() { None } else { Some(&memory) };
    match interp::run_watching_for_nan_with_env(&shader, &mut regs, &textures, mem_arg) {
        Ok(_) => {}
        Err(e) => println!("  interpretation stopped: {e}"),
    }
    let seen = seen.borrow();
    let hits = seen.iter().filter(|(_, inside)| *inside).count();
    println!("\n  {} load(s), {hits} inside a window, {} OUTSIDE (which read zero)", seen.len(), seen.len() - hits);
    for (addr, inside) in seen.iter().take(24) {
        println!("    {addr:#010x}  {}", if *inside { "inside" } else { "OUTSIDE -> 0" });
    }
}

#[test]
#[ignore = "needs a captured corpus (VITASLOP_GXP_CORPUS) and an output directory"]
fn write_every_blob_as_an_execution_case() {
    let Some(dir) = corpus_dir() else {
        eprintln!("VITASLOP_GXP_CORPUS unset - nothing to do");
        return;
    };
    let Some(out) = out_dir() else {
        eprintln!("VITASLOP_GXP_CASES_OUT unset - nothing to do");
        return;
    };
    std::fs::create_dir_all(&out).expect("create the case output directory");

    let mut files: Vec<PathBuf> = std::fs::read_dir(&dir)
        .expect("read the corpus directory")
        .flatten()
        .map(|e| e.path())
        .filter(|p| p.extension().and_then(|s| s.to_str()) == Some("gxp"))
        .collect();
    files.sort();

    let mut skipped: BTreeMap<Skip, Vec<String>> = BTreeMap::new();
    let mut written = 0usize;
    // How many of those go through the RENDER rig, so the coverage line can say which half of
    // the corpus each instrument carries.
    let mut rendered = 0usize;
    // How many programs got a SECOND case at the opposite facing - see the arm below.
    let mut back_facing = 0usize;
    // A program whose every written lane is zero proves nothing by agreeing, so it is counted
    // separately rather than padding the pass rate.
    let mut trivial = 0usize;
    let mut interp_refusals: BTreeMap<String, usize> = BTreeMap::new();
    // Which texture kind turned a fragment-stage program away, so the census ranks the work.
    let mut tex_kinds: BTreeMap<&str, usize> = BTreeMap::new();
    let (mut loads_total, mut loads_inside, mut loads_all_missed) = (0u32, 0u32, 0usize);

    for path in &files {
        let name = path.file_stem().unwrap_or_default().to_string_lossy().into_owned();
        let Ok(bytes) = std::fs::read(path) else { continue };
        let Ok(program) = Program::parse(&bytes) else {
            skipped.entry(Skip::NotAProgram).or_default().push(name);
            continue;
        };
        let kind = program.kind;
        let recompiled = match kind {
            ProgramKind::Vertex => recompile_vertex(&bytes).map(|r| (r.shader, r.wgsl_body)),
            ProgramKind::Fragment => recompile_fragment(&bytes).map(|r| (r.shader, r.wgsl_body)),
        };
        let Ok((shader, body)) = recompiled else {
            skipped.entry(Skip::EmitRefused).or_default().push(name);
            continue;
        };
        if let Some(why) = rig_excludes(&shader) {
            skipped.entry(why).or_default().push(name);
            continue;
        }
        let render = needs_fragment_stage(&shader);
        if render && !matches!(kind, ProgramKind::Fragment) {
            skipped.entry(Skip::RenderNotFragment).or_default().push(name);
            continue;
        }
        // A program whose loads reach a window the linker cannot RESOLVE is refused on the
        // shipped path too (`LinkError::MemWindowUnresolved` drops the draw rather than feed it
        // zeroes), so there is nothing here for this harness to check either.
        let windows = match kind {
            ProgramKind::Vertex => mem_windows_for_vertex_blob(&bytes),
            ProgramKind::Fragment => mem_windows_for_fragment_blob(&bytes),
        };
        if body.contains("gxp_mem") && windows.is_empty() {
            skipped.entry(Skip::MemWindowUnresolved).or_default().push(name);
            continue;
        }

        // The seed is the program's own content hash, so a case is reproducible from the blob
        // alone and two different programs never share an input file.
        let seed = (program.hash ^ (program.hash >> 32)) as u32 | 1;
        let mem_words = mem_words_for(seed, &windows);
        // >>> AND THE LANES THE PROGRAM READS AS COUNTS, INDICES AND POINTERS GET PLAUSIBLE
        // >>> VALUES - CHOSEN BY RUNNING THE REFERENCE, not by trusting the rule that named
        // >>> them. See `best_plausible_inputs`; its candidate 0 is the empty set, so a program
        // >>> no candidate improves keeps exactly the inputs it had before this existed.
        let inputs = chosen_case_inputs(&shader, seed, &windows, &mem_words);
        let mut regs = seeded_case_regs(seed, &windows, &inputs);

        let sampled: Vec<u8> = shader
            .instrs
            .iter()
            .filter_map(|i| match i.op {
                Op::Tex { unit, .. } => Some(unit),
                _ => None,
            })
            .collect();
        // >>> A PROGRAM THAT NEEDS A FRAGMENT STAGE GOES TO THE RENDER RIG, and everything else
        // stays on the compute one. Not a replacement: the compute rig's varying stand-in sees a
        // UV error of any size, while the render rig's real texture can only see one of at least
        // a texel - so moving an ordinary sampling program over would LOSE coverage.
        let mut cube_slots: Vec<u8> = Vec::new();
        let (module, units) = if render {
            // The bindings the program needs, with the container's own answer for which units
            // are cube maps - the same question the shipped pipeline builder asks.
            let want = wgsl::tex_units(&shader, |u| program.sampler_is_cube(u32::from(u)));
            let (m, rewrites) = match wrap_render_case_module_for(&body, kind, &windows, &want) {
                Ok(v) => v,
                Err(why) => {
                    *tex_kinds.entry(why).or_default() += 1;
                    skipped.entry(Skip::RenderTexKind).or_default().push(name);
                    continue;
                }
            };
            // >>> THE REWRITE IS COUNTED AGAINST THE INSTRUCTION STREAM, not trusted. A sample
            // the textual parse missed stays a real `textureSample` on an unquantised
            // coordinate, which COMPILES - so nothing downstream would notice, and the device's
            // subtexel rounding would be reported as a translation defect.
            if rewrites != render_rewrite_targets(&shader) {
                skipped.entry(Skip::RenderUnrewritten).or_default().push(name);
                continue;
            }
            // The binding KINDS travel with the case: the runner has one 2D texture set and one
            // CUBE set per unit, and which to bind at a slot is not recoverable from the unit
            // number - the same unit is a flat texture in one program and a cube in another.
            cube_slots = want.iter().map(|b| u8::from(b.cube && b.coords >= 3)).collect();
            (m, want.iter().map(|b| b.unit).collect::<Vec<u8>>())
        } else {
            // The module and the reference must be handed the SAME constant per unit. The
            // wrapper reports which units it replaced; anything the parse missed would leave a
            // real `textureSample` in a compute module, which cannot compile - so the two are
            // checked against each other here rather than assumed to agree.
            let (m, u) = wrap_compute_module_for(&body, kind, &windows);
            if m.contains("textureSample") {
                skipped.entry(Skip::UnparsedSample).or_default().push(name);
                continue;
            }
            (m, u)
        };
        if sampled.iter().any(|u| !units.contains(u)) {
            skipped.entry(Skip::UnparsedSample).or_default().push(name);
            continue;
        }
        // The window's guest base goes into the SA register the driver places it in, AFTER the
        // seeded fill - the same order the emitted prologue uses, and a seed left there instead
        // would send every load to a random address outside every window.
        for (i, win) in windows.iter().enumerate() {
            if let Some(slot) = regs.sa.get_mut(win.base_sa as usize) {
                *slot = f32::from_bits(window_base(i));
            }
        }
        // >>> EACH RIG'S REFERENCE FETCHES WHAT THAT RIG'S MODULE SAMPLES. The compute rig's
        // module carries a stand-in function; the render rig's binds a real texture, so the
        // reference reads that texture's own texels. A call site that handed one rig the other's
        // fetcher would report every sampling program as a translation defect.
        // A unit sampled as a CUBE reads the cube set, and which units those are is the
        // container's answer - the same one the module's binding type was built from. Reading
        // the flat set for a cube unit would disagree with the module on every sample.
        let is_cube = |u: u8| {
            cube_slots.iter().zip(units.iter()).any(|(&c, &un)| un == u && c == 1)
        };
        // The render rig binds ONE mip, so every LOD form returns level 0 there and the operand
        // is not part of what it can check - the compute rig's stand-in is where it is checked.
        let render_sample = |unit: u8, coord: [f32; 4], _lod: interp::TexLodArg| {
            Some(if is_cube(unit) {
                case_render_sample_cube(unit, coord)
            } else {
                case_render_sample(unit, coord)
            })
        };
        let render_gather = |unit: u8, uv: [f32; 2]| Some(case_render_gather(unit, uv));
        let texenv = if render {
            interp::TexEnv { sample: &render_sample, gather: Some(&render_gather) }
        } else {
            interp::TexEnv::sampling(&case_textures)
        };
        // >>> COUNT WHETHER THE LOADS ACTUALLY LANDED. A load outside every window reads ZERO on
        // both sides by design, so a program whose addresses all miss AGREES without having
        // checked anything - a coverage illusion that pass/fail cannot show. The addresses come
        // from the program's own arithmetic over a SEEDED register file, and a register that
        // holds a small index in a real draw holds a float's bit pattern here, so misses are
        // expected; what must not happen is nobody knowing how many.
        let loads = std::cell::Cell::new((0u32, 0u32));
        let memory = |addr: u32| {
            let inside = placements_hit(addr, &windows);
            let (t, i) = loads.get();
            loads.set((t + 1, i + u32::from(inside)));
            mem_fetch(addr, &windows, &mem_words)
        };
        let mem_arg: Option<&dyn Fn(u32) -> u32> = if windows.is_empty() { None } else { Some(&memory) };
        if let Err(e) =
            interp::run_traced_env(&shader, &mut regs, &texenv, mem_arg, &mut |_, _| {}).map(|_| ())
        {
            let op = match e {
                interp::InterpError::UnsupportedOp { op, .. } => op.to_string(),
                interp::InterpError::Blocked { reason, .. } => format!("blocked: {reason}"),
                interp::InterpError::OutOfRange { .. } => "operand out of range".to_string(),
            };
            *interp_refusals.entry(op).or_default() += 1;
            skipped.entry(Skip::InterpRefused).or_default().push(name);
            continue;
        }

        let (load_total, load_inside) = loads.get();
        if load_total > 0 {
            loads_total += load_total;
            loads_inside += load_inside;
            if load_inside == 0 {
                loads_all_missed += 1;
            }
        }
        let prec = written_lane_precision(&shader);
        let mut pairs: Vec<String> = Vec::new();
        nonzero_pairs(0, 0, &regs.r, &prec, &mut pairs);
        nonzero_pairs(1, CASE_BANK_LANES, &regs.o, &prec, &mut pairs);
        nonzero_pairs(2, CASE_BANK_LANES * 2, &regs.i, &prec, &mut pairs);
        // >>> `pa` IS AN OUTPUT TOO. A fragment program's colour can live there
        // (`ColorOutput::NonNativePa`), and while this bank went uncaptured such a program's
        // whole effect was invisible: 176 of 1,025 cases expected an all-zero register file.
        // Its baseline is the SEEDED input, so only what the program changed travels.
        // The baseline is the seed AS THE CASE ACTUALLY STARTED IT - overrides included. A
        // baseline that forgot them would report every overridden `pa` lane as a lane the
        // program changed, on every case that has one.
        let seeded_pa = seeded_pa_baseline(seed, &inputs);
        changed_pairs(3, CASE_BANK_LANES * 3, &regs.pa, &seeded_pa, &prec, &mut pairs);
        if pairs.is_empty() {
            trivial += 1;
        }
        // >>> THE TWO FRAGMENT-STAGE RESULTS ARE PART OF THE ANSWER, not bookkeeping. A
        // translation that dropped a `kill` leaves every register agreeing and still paints a
        // pixel the hardware discards; one that wrote the depth from the wrong channel leaves
        // every register agreeing and sorts the pixel wrongly.
        //
        // The KILL flag is compared as a raw word (precision code 3): it is a flag, and "within
        // one ULP of killed" is not a thing.
        //
        // The DEPTH carries the rig's pinned forward map applied to what the reference computed
        // - see `wrap_render_case_module_for`, whose helper is that same clamp. `min`/`max`
        // rather than `f32::clamp`, because WGSL's `clamp` is defined as `min(max(e,lo),hi)` and
        // the two differ on a NaN.
        if render {
            pairs.push(format!("[{},{},3]", CASE_BANK_LANES * 4, u32::from(regs.killed)));
            #[allow(clippy::manual_clamp)]
            let depth = regs.frag_depth.map(|v| v.max(0.0).min(1.0)).unwrap_or(0.0);
            pairs.push(format!("[{},{},0]", CASE_BANK_LANES * 4 + 1, depth.to_bits()));
            rendered += 1;
        }

        // HALF-PRECISION IS THE REFERENCE'S KNOWN BLIND SPOT: the interpreter's register file is
        // one `f32` per lane, so a program that packs two F16 values into a lane means something
        // the reference cannot represent, while the emitter models it. Carry the count so a
        // divergence can be attributed to that gap instead of being read as an emitter defect.
        //
        // The window's words travel as DATA, not as literals in the module - see
        // `wrap_compute_module_for`. The runner uploads them to the storage binding the module
        // declares, so both sides read the identical bytes with no second generator.
        let half = shader.instrs.iter().filter(|i| i.half_precision).count();
        write_case(
            &out,
            &name,
            kind,
            seed,
            shader.instrs.len(),
            half,
            &module,
            &mem_words,
            &pairs,
            &inputs,
            if render { Stage::Fragment } else { Stage::Compute },
            if render { &units } else { &[] },
            if render { &cube_slots } else { &[] },
        );
        written += 1;

        // >>> AND THE BACK-FACING ARM, for the programs that read the facing global.
        //
        // `GLOBAL[16]` is pinned to `true` by BOTH wrappers and by the reference, which is what
        // makes such a program an ordinary case at all - neither side models a rasteriser. The
        // cost of that pinning is that every one of those programs runs FRONT-FACING only, so
        // whichever way its facing test branches, the other arm is executed by nothing.
        //
        // A second case at `false` runs it. It is a second CASE rather than a second draw
        // because the value is a pinned constant on both sides: there is nothing a reversed
        // winding would add that the constant does not already say, and a winding would bring
        // the framebuffer-space orientation convention into a rig that has no need of one.
        if reads_the_facing_global(&shader) {
            let mut back = seeded_case_regs(seed, &windows, &inputs);
            back.facing = Some(false);
            let back_module = if render {
                let want = wgsl::tex_units(&shader, |u| program.sampler_is_cube(u32::from(u)));
                match wgsl::wrap_render_case_module_facing(&body, kind, &windows, &want, false) {
                    // The front arm already passed both checks on the same body, so a refusal or
                    // an under-rewrite here cannot normally happen - and if it did, dropping the
                    // back arm silently would make its absence invisible. It is counted.
                    Ok((m, rewrites)) if rewrites == render_rewrite_targets(&shader) => Some(m),
                    _ => {
                        skipped.entry(Skip::RenderUnrewritten).or_default().push(back_name(&name));
                        None
                    }
                }
            } else {
                Some(wgsl::wrap_compute_module_facing(&body, kind, &windows, false).0)
            };
            let loads = std::cell::Cell::new(0u32);
            let memory = |addr: u32| {
                loads.set(loads.get() + 1);
                mem_fetch(addr, &windows, &mem_words)
            };
            let mem_arg: Option<&dyn Fn(u32) -> u32> =
                if windows.is_empty() { None } else { Some(&memory) };
            let ran = interp::run_traced_env(&shader, &mut back, &texenv, mem_arg, &mut |_, _| {})
                .is_ok();
            if let (Some(back_module), true) = (back_module, ran) {
                let prec = written_lane_precision(&shader);
                let mut pairs: Vec<String> = Vec::new();
                nonzero_pairs(0, 0, &back.r, &prec, &mut pairs);
                nonzero_pairs(1, CASE_BANK_LANES, &back.o, &prec, &mut pairs);
                nonzero_pairs(2, CASE_BANK_LANES * 2, &back.i, &prec, &mut pairs);
                changed_pairs(3, CASE_BANK_LANES * 3, &back.pa, &seeded_pa, &prec, &mut pairs);
                if render {
                    pairs.push(format!("[{},{},3]", CASE_BANK_LANES * 4, u32::from(back.killed)));
                    #[allow(clippy::manual_clamp)]
                    let depth = back.frag_depth.map(|v| v.max(0.0).min(1.0)).unwrap_or(0.0);
                    pairs.push(format!("[{},{},0]", CASE_BANK_LANES * 4 + 1, depth.to_bits()));
                }
                write_case(
                    &out,
                    &back_name(&name),
                    kind,
                    seed,
                    shader.instrs.len(),
                    half,
                    &back_module,
                    &mem_words,
                    &pairs,
                    &inputs,
                    if render { Stage::Fragment } else { Stage::Compute },
                    if render { &units } else { &[] },
                    if render { &cube_slots } else { &[] },
                );
                written += 1;
                back_facing += 1;
                if render {
                    rendered += 1;
                }
            }
        }
    }

    // >>> THE RENDER RIG'S TEXELS TRAVEL AS A FILE, not as a generator the runner reimplements.
    // A second copy of the avalanche in JS would be a second thing to keep in step, and a drift
    // would hand the GPU different texels from the reference and be reported as a translation
    // defect - the same reason the guest-memory windows travel as bytes.
    if rendered > 0 {
        let tex = out.join("casetex.bin");
        if let Err(e) = std::fs::write(&tex, case_tex_bytes()) {
            println!("  WARNING: could not write {}: {e}", tex.display());
        }
    }

    let total = files.len();
    let skipped_total: usize = skipped.values().map(|v| v.len()).sum();
    println!("\n=== execution cases from {total} blobs in {} ===", dir.display());
    println!(
        "  {written} cases written ({trivial} of them write nothing at all); \
         {rendered} on the RENDER rig, {} on the compute rig",
        written - rendered
    );
    if back_facing > 0 {
        println!(
            "  {back_facing} program(s) also got a BACK-FACING case: they read GLOBAL[16], \n             and with it pinned to `true` the other arm of their facing test ran nowhere."
        );
    }
    println!("  {skipped_total} skipped:");
    for (why, names) in &skipped {
        println!("    {:>5}  {}", names.len(), why.label());
        // Name a handful so a category that should not be large can be chased without a re-run.
        for n in names.iter().take(4) {
            println!("             {n}");
        }
        if names.len() > 4 {
            println!("             ... and {} more", names.len() - 4);
        }
    }
    if !tex_kinds.is_empty() {
        println!("
  >>> AND WHICH TEXTURE KIND TURNED A FRAGMENT-STAGE PROGRAM AWAY:");
        let mut ranked: Vec<_> = tex_kinds.iter().collect();
        ranked.sort_by(|a, b| b.1.cmp(a.1).then(a.0.cmp(b.0)));
        for (kind, n) in ranked {
            println!("    {n:>5}  {kind}");
        }
    }
    if !interp_refusals.is_empty() {
        println!("\n  >>> THE REFERENCE'S OWN COVERAGE GAPS - the EMITTER accepted these and the");
        println!("      interpreter did not, so they are shader code that ships with no oracle:");
        let mut ranked: Vec<_> = interp_refusals.iter().collect();
        ranked.sort_by(|a, b| b.1.cmp(a.1).then(a.0.cmp(b.0)));
        for (op, n) in ranked {
            println!("    {n:>5}  {op}");
        }
        // >>> AND ONE ENTRY IN THAT LIST IS NOT A COVERAGE GAP AT ALL, so it is named rather
        // >>> than left to read as one.
        //
        // The step budget is hit by a golf title's world vertex programs, which run a loop
        // around a memory load whose trip count comes from a REGISTER. In a real draw that
        // register holds a light count; here it holds whatever the seed put there - a float's
        // bit pattern read as an integer - so the loop is asked to run billions of times.
        //
        // The emitted WGSL has the same real loop (only USSE's repeat field is unrolled, not a
        // data-driven branch), so the GPU side would spin on the identical seed. Skipping is
        // therefore correct on BOTH sides, and there is nothing in the reference to implement:
        // what it would take is a seed chosen per program so that a loop-controlling register
        // holds a plausible count, and nothing here knows which register that is.
        if let Some(n) = interp_refusals
            .iter()
            .find(|(op, _)| op.contains("step budget"))
            .map(|(_, n)| *n)
        {
            println!(
                "      NOTE: {n} of those are the step budget, which is a property of the SEED \
                 and not of the reference - the emitted shader carries the same data-driven \
                 loop and would spin on the same inputs. Not an op to implement."
            );
        }
    }
    if loads_total > 0 {
        println!(
            "\n  >>> GUEST-MEMORY COVERAGE: {loads_inside} of {loads_total} loads landed inside a bound window \
             ({:.0}%); {loads_all_missed} program(s) had EVERY load miss.",
            100.0 * f64::from(loads_inside) / f64::from(loads_total)
        );
        println!(
            "      A load outside every window reads ZERO on both sides, so those agree without\n      \
             checking anything - the addresses come from the program's own arithmetic over a\n      \
             SEEDED register file, where a register that holds a small index in a real draw holds\n      \
             a float's bit pattern instead."
        );
    }
    // >>> AND THE SKIPS TRAVEL WITH THE CASES, BECAUSE THE RUNNER REPORTS THE PASS RATE AND
    // >>> THE PASS RATE IS NOT THE COVERAGE.
    //
    // A blob that never became a case is not a program that agreed - it is a program nothing
    // checked. `gxpexec` prints "1,029 cases: 861 exact ..." and has no way to know that 122
    // blobs were never handed to it, so the number reads as though the corpus were covered.
    // The writer is the only thing that knows, so it says so in a file the runner reads.
    //
    // NOT a `.json`: the runner globs `*.json` for CASES and would try to run this one.
    let mut summary = String::from("{\n");
    let _ = writeln!(summary, "  \"blobs\": {total},");
    let _ = writeln!(summary, "  \"written\": {written},");
    let _ = writeln!(summary, "  \"rendered\": {rendered},");
    let _ = writeln!(summary, "  \"trivial\": {trivial},");
    let _ = writeln!(summary, "  \"skipped\": {skipped_total},");
    let _ = writeln!(summary, "  \"reasons\": {{");
    let labelled: Vec<String> = skipped
        .iter()
        .map(|(why, names)| format!("    \"{}\": {}", why.label(), names.len()))
        .collect();
    let _ = writeln!(summary, "{}", labelled.join(",\n"));
    let _ = writeln!(summary, "  }}\n}}");
    let cov = out.join("coverage.summary");
    if let Err(e) = std::fs::write(&cov, &summary) {
        // A summary that failed to write must not pass for "nothing was skipped".
        println!("  WARNING: could not write {}: {e}", cov.display());
    }

    println!("\n  run them:  node vitaslop-web/e2e/gxpexec.mjs {}", out.display());
}

/// >>> A CONDITIONAL WRITE THAT DISAGREES ABOUT A LANE'S VIEW LEAVES IT AMBIGUOUS, and a
/// >>> straight-line program is untouched by that conservatism.
///
/// MEASURED on `gxp-all__hs-gxp__frag_81fa9a10`: `r[10]` is written as an F16 PAIR, then copied
/// whole (`r[10] = sa[34] | 0u`, this ISA's integer move) inside an `if` that did not run. The
/// static walk took the copy's view, the runner read two NaN halves as one f32, and the report
/// carried **4,227,955,712 ULP** for a lane on which both sides agree.
///
/// The three cases below are the whole rule: a determined overwrite still wins, a skippable
/// writer that AGREES changes nothing, and only a skippable DISAGREEMENT goes ambiguous.
#[test]
fn a_conditional_writer_that_disagrees_leaves_the_lane_ambiguous() {
    use vitaslop_gxp_shader::ir::{Instr, Operand, Predicate};

    let at = |op: Op, dest_reg: u8, half: bool, pred: Predicate| Instr {
        op,
        pred,
        dest: Some(Operand::plain(Bank::Temp, dest_reg, 0)),
        write_mask: [true, false, false, false],
        srcs: vec![Operand::plain(Bank::PrimaryAttr, 0, 0)],
        half_precision: half,
        raw: 0,
        group: 0,
        blocked: None,
    };
    let sh = |instrs: Vec<Instr>| Shader { kind: ProgramKind::Fragment, instrs };
    let view = |s: &Shader| written_lane_precision(s).get(&(0u8, 0usize)).copied();

    // Unconditional f16 then unconditional f32: the later one determines the lane.
    assert_eq!(
        view(&sh(vec![
            at(Op::Mov, 0, true, Predicate::Always),
            at(Op::Mov, 0, false, Predicate::Always),
        ])),
        Some(0),
        "a determined overwrite still decides the view"
    );
    // Unconditional f16 then a PREDICATED f32 copy: the lane holds one or the other.
    assert_eq!(
        view(&sh(vec![
            at(Op::Mov, 0, true, Predicate::Always),
            at(Op::Mov, 0, false, Predicate::IfP(0)),
        ])),
        Some(PREC_AMBIGUOUS),
        "a skippable writer that disagrees must not claim the lane"
    );
    // A predicated writer that AGREES with the standing view costs nothing.
    assert_eq!(
        view(&sh(vec![
            at(Op::Mov, 0, true, Predicate::Always),
            at(Op::Mov, 0, true, Predicate::IfP(0)),
        ])),
        Some(1),
        "agreement needs no leniency"
    );
}

/// One f32 ULP up (or down) from `v`, by bit pattern. A zero steps to the smallest subnormal
/// of the requested sign, and a non-finite value is left alone.
fn step_ulp(v: f32, up: bool) -> f32 {
    if !v.is_finite() {
        return v;
    }
    let b = v.to_bits();
    // Same-signed float bit patterns are monotonic, so a step is +-1 on the pattern - except
    // across zero, where the sign bit flips and the neighbour is the other sign's smallest
    // subnormal.
    let nb = if v == 0.0 {
        if up {
            1
        } else {
            0x8000_0001
        }
    } else if (b & 0x8000_0000 != 0) == up {
        b - 1
    } else {
        b + 1
    };
    f32::from_bits(nb)
}

/// Distance between two f32 values in ULPs, the same measure `gxpexec.mjs` reports.
fn ulp_gap(a: f32, b: f32) -> u64 {
    if !a.is_finite() || !b.is_finite() {
        return if a.to_bits() == b.to_bits() { 0 } else { u64::MAX };
    }
    let ord = |v: f32| -> i64 {
        let x = v.to_bits() as i64;
        if v.to_bits() & 0x8000_0000 != 0 {
            -(x & 0x7fff_ffff)
        } else {
            x
        }
    };
    ord(a).abs_diff(ord(b))
}

/// One f16 ULP up (or down) from the 16-bit pattern `h`, the same way [`step_ulp`] steps an
/// f32: same-signed patterns are monotonic, so a step is +-1 on the pattern, and a zero steps
/// to the smallest subnormal of the requested sign. A non-finite half is left alone.
fn step_half_ulp(h: u16, up: bool) -> u16 {
    if h & 0x7c00 == 0x7c00 {
        return h; // infinity or NaN
    }
    if h & 0x7fff == 0 {
        return if up { 1 } else { 0x8001 };
    }
    if (h & 0x8000 != 0) == up {
        h - 1
    } else {
        h + 1
    }
}

/// A register word with BOTH of its 16-bit halves stepped one f16 ULP.
///
/// The seeded register file holds one `f32` per lane, but a 16-bit instruction reads that
/// lane's BIT PATTERN as two halves - so the perturbation a half-precision program can actually
/// see is a step of each half, not of the f32 the lane prints as.
fn step_both_halves(v: f32, up: bool) -> f32 {
    let b = v.to_bits();
    let lo = step_half_ulp(b as u16, up) as u32;
    let hi = step_half_ulp((b >> 16) as u16, up) as u32;
    f32::from_bits((hi << 16) | lo)
}

/// >>> AT WHICH PRECISION IS EACH SEEDED REGISTER ACTUALLY READ?
///
/// The census nudges the inputs and watches the output move. For that to mean anything the
/// nudge has to be a step the program can SEE, and one f32 ULP is roughly 8192 times finer
/// than the f16 quantum - so on a 16-bit program it rounds away at the first read and the
/// census reports the program perfectly stable. It did: the f16 residue clustered at exactly
/// 24576 / 32768 / 40960 ULP, which are f16 ULPs wearing f32 clothes, while the instrument
/// said those cases were quiet.
///
/// So each PA and SA register is classified by the widest read that reaches it:
/// `Some(true)` = read ONLY through 16-bit instructions, `Some(false)` = something reads it at
/// full precision, `None` = never read at all. A register read at BOTH widths takes the f32
/// step, because the full-precision read is the one that can see it - the nudge must be the
/// FINEST step the program notices, or a stable program is reported unstable and the census
/// stops being one-sided.
///
/// The spans are the emitter's own: a four-channel F32 operand reaches `index..index+4`,
/// because its swizzle selects among four consecutive registers; a 16-bit one reaches
/// `index..index+2`, because its four channels are the two halves of two registers. Swizzle
/// selectors 4..7 are CONSTANTS and read no register at all, which is why the span is fixed
/// rather than taken from the selectors.
fn seeded_read_precision(shader: &Shader) -> (Vec<Option<bool>>, Vec<Option<bool>>) {
    let mut pa = vec![None; CASE_BANK_LANES];
    let mut sa = vec![None; CASE_BANK_LANES];
    for instr in &shader.instrs {
        let half = instr.source_half_precision();
        for src in &instr.srcs {
            let bank = match src.bank {
                Bank::PrimaryAttr => &mut pa,
                Bank::SecondaryAttr => &mut sa,
                _ => continue,
            };
            for k in 0..(if half { 2 } else { 4 }) {
                let Some(slot) = bank.get_mut(src.index as usize + k) else { continue };
                // `false` (a full-precision read) wins, because it is the finer step.
                *slot = Some(slot.unwrap_or(true) && half);
            }
        }
    }
    (pa, sa)
}

/// >>> CAN A CASE EVEN ANSWER THE QUESTION IT IS ASKED? A DIVERGENCE OVER AN ILL-CONDITIONED
/// >>> PROGRAM MEANS NOTHING, AND NOTHING IN THE RIG COULD SAY WHICH ONES THOSE WERE.
///
/// `gxpexec` compares the GPU register file against this crate's reference and reports the
/// difference in ULPs. That comparison assumes the program is STABLE: that a legitimate
/// last-bit disagreement between two float units stays a last-bit disagreement at the output.
/// Plenty of real shader code is not. One corpus vertex program carries a `log2` and an `exp2`
/// over a SEEDED register file, and there a one-ULP wobble on the input leaves as a factor of
/// ten million - reported as 2,728,658,196 ULP and read, for a whole session, as a translation
/// defect.
///
/// So this measures the program's own amplification: nudge every seeded input lane by ONE f32
/// ULP and re-run THE REFERENCE AGAINST ITSELF. Whatever the output moves by is motion no
/// comparison against a second float unit can attribute, because the GPU is entitled to that
/// much disagreement per operation before it begins.
///
/// It is ONE-SIDED, deliberately. A large movement PROVES the case cannot discriminate - its
/// own reference disagrees with itself by that much. A small movement does NOT certify the
/// program: the perturbation is applied to the INPUTS while a GPU's freedom is per operation.
/// A quiet case is evidence, not a certificate.
///
/// The window BASE registers are left alone: they hold an ADDRESS bit pattern rather than a
/// number, and stepping one sends every load somewhere else.
///
/// ```text
/// VITASLOP_GXP_CORPUS=<dir> cargo test --release -p vitaslop-gxp-shader --test execcases \
///   -- --ignored --nocapture how_far_a_one_ulp_input_wobble_moves_each_case
/// ```
#[test]
#[ignore = "a census over the whole corpus; needs VITASLOP_GXP_CORPUS"]
fn how_far_a_one_ulp_input_wobble_moves_each_case() {
    let Some(dir) = corpus_dir() else {
        eprintln!("VITASLOP_GXP_CORPUS unset - nothing to do");
        return;
    };
    let mut files: Vec<PathBuf> = std::fs::read_dir(&dir)
        .expect("read the corpus directory")
        .flatten()
        .map(|e| e.path())
        .filter(|p| p.extension().and_then(|s| s.to_str()) == Some("gxp"))
        .collect();
    files.sort();

    let mut rows: Vec<(String, u64, u64, usize, usize)> = Vec::new();
    for path in &files {
        let name = path.file_stem().unwrap_or_default().to_string_lossy().into_owned();
        let Ok(bytes) = std::fs::read(path) else { continue };
        let Ok(program) = Program::parse(&bytes) else { continue };
        let kind = program.kind;
        let recompiled = match kind {
            ProgramKind::Vertex => recompile_vertex(&bytes).map(|r| (r.shader, r.wgsl_body)),
            ProgramKind::Fragment => recompile_fragment(&bytes).map(|r| (r.shader, r.wgsl_body)),
        };
        let Ok((shader, body)) = recompiled else { continue };
        if rig_excludes(&shader).is_some() || body.contains("gxp_depth") {
            continue;
        }
        let windows = match kind {
            ProgramKind::Vertex => mem_windows_for_vertex_blob(&bytes),
            ProgramKind::Fragment => mem_windows_for_fragment_blob(&bytes),
        };
        if body.contains("gxp_mem") && windows.is_empty() {
            continue;
        }
        let seed = (program.hash ^ (program.hash >> 32)) as u32 | 1;
        let mem_words = mem_words_for(seed, &windows);
        let textures = case_textures;

        // The same setup the case writer uses, with every seeded lane optionally stepped. The
        // window bases are written AFTER the fill and are never stepped.
        let (pa_prec, sa_prec) = seeded_read_precision(&shader);
        // A program every one of whose seeded reads is 16-bit is nudged at the f16 quantum;
        // see `seeded_read_precision`. Counted so the census can say how many it measured each
        // way rather than leaving the reader to assume one.
        let half_nudged = pa_prec.iter().chain(sa_prec.iter()).filter(|p| **p == Some(true)).count();
        let full_nudged = pa_prec.iter().chain(sa_prec.iter()).filter(|p| **p == Some(false)).count();
        // >>> THE NUDGE IS APPLIED TO THE INPUTS THE CASE ACTUALLY RUNS ON, which are no longer
        // the seed alone. A conditioning verdict measured over different inputs from the case's
        // is a verdict about a program the case never ran - and this census exists precisely to
        // say whether a case can answer the question it is asked.
        let chosen = chosen_case_inputs(&shader, seed, &windows, &mem_words);
        let chosen_pa = seeded_pa_baseline(seed, &chosen);
        let chosen_sa: Vec<f32> = {
            let mut v: Vec<f32> = (0..CASE_BANK_LANES)
                .map(|n| lane_value(seed, (n + CASE_BANK_LANES) as u32))
                .collect();
            for &(lane, bits) in &chosen {
                if lane >= CASE_BANK_LANES {
                    v[lane - CASE_BANK_LANES] = f32::from_bits(bits);
                }
            }
            v
        };
        // >>> AN OVERRIDDEN LANE IS NOT NUDGED. It holds a count, an index or a pointer, and
        // stepping one of those by an ULP is not a small perturbation - it is a DIFFERENT index,
        // which would report the program's own addressing as instability. The window bases are
        // left alone for exactly the same reason, and always were.
        let overridden: std::collections::BTreeSet<usize> =
            chosen.iter().map(|&(l, _)| l).collect();
        let build = |nudge: Option<bool>| -> RegFile {
            let mut regs = RegFile::with_lanes(CASE_BANK_LANES);
            for n in 0..CASE_BANK_LANES {
                let mut pa = chosen_pa[n];
                let mut sa = chosen_sa[n];
                let pinned =
                    overridden.contains(&n) || overridden.contains(&(n + CASE_BANK_LANES));
                if let Some(up) = nudge
                    && !pinned
                {
                    // One ULP of the format THIS lane is read at. An unread lane takes the f32
                    // step; it changes nothing either way, and guessing otherwise would make
                    // the untouched lanes a second, silent, rule.
                    pa = if pa_prec[n] == Some(true) { step_both_halves(pa, up) } else { step_ulp(pa, up) };
                    sa = if sa_prec[n] == Some(true) { step_both_halves(sa, up) } else { step_ulp(sa, up) };
                }
                regs.pa[n] = pa;
                regs.sa[n] = sa;
            }
            for (i, win) in windows.iter().enumerate() {
                if let Some(slot) = regs.sa.get_mut(win.base_sa as usize) {
                    *slot = f32::from_bits(window_base(i));
                }
            }
            regs
        };
        let run = |regs: &mut RegFile| -> bool {
            let memory = |addr: u32| mem_fetch(addr, &windows, &mem_words);
            let mem_arg: Option<&dyn Fn(u32) -> u32> =
                if windows.is_empty() { None } else { Some(&memory) };
            interp::run_watching_for_nan_with_env(&shader, regs, &textures, mem_arg).is_ok()
        };

        let mut base = build(None);
        if !run(&mut base) {
            continue;
        }
        let mut worst = 0u64;
        let mut lanes_moved = 0u64;
        for up in [true, false] {
            let mut alt = build(Some(up));
            if !run(&mut alt) {
                continue;
            }
            let b_all = base.r.iter().chain(base.o.iter()).chain(base.i.iter());
            let a_all = alt.r.iter().chain(alt.o.iter()).chain(alt.i.iter());
            for (b, a) in b_all.zip(a_all) {
                let g = ulp_gap(*b, *a);
                if g > 0 {
                    lanes_moved += 1;
                }
                worst = worst.max(g);
            }
        }
        rows.push((name, worst, lanes_moved, half_nudged, full_nudged));
    }

    // Buckets chosen against what the comparison can survive: a case whose own reference moves
    // by more than a few ULP cannot be read at a 1-ULP tolerance, and one that moves by
    // millions cannot be read at all.
    let bucket = |w: u64| -> &'static str {
        if w == u64::MAX {
            "non-finite flip"
        } else if w == 0 {
            "0 (stable)"
        } else if w <= 16 {
            "1-16"
        } else if w <= 1_000 {
            "17-1k"
        } else if w <= 100_000 {
            "1k-100k"
        } else {
            ">100k"
        }
    };
    let mut tally: BTreeMap<&str, usize> = BTreeMap::new();
    for (_, w, _, _, _) in &rows {
        *tally.entry(bucket(*w)).or_default() += 1;
    }
    // >>> HOW MANY PROGRAMS WERE NUDGED AT WHICH QUANTUM, said out loud. Before this the
    // census applied one f32 ULP to everything and understated every 16-bit program by about
    // 8192x, and nothing in its output could have told you so.
    let any_half = rows.iter().filter(|r| r.3 > 0).count();
    let only_half = rows.iter().filter(|r| r.3 > 0 && r.4 == 0).count();
    println!("\n=== how far a ONE-ULP input wobble moves the reference against itself ===");
    println!("  {} programs measured\n", rows.len());
    for (k, v) in &tally {
        println!("  {v:>5}  {k}");
    }
    println!(
        "\n  nudged at the f16 quantum on at least one lane: {any_half} programs \
         ({only_half} read NOTHING at full precision); the rest at one f32 ULP",
    );
    println!("\n  the 20 loudest (worst output ULP movement, lanes moved, name):");
    let mut ranked = rows.clone();
    ranked.sort_by_key(|r| std::cmp::Reverse(r.1));
    for (n, w, l, _, _) in ranked.iter().take(20) {
        println!("    {w:>12}  {l:>4}  {n}");
    }
    // One line per program, so a divergence list can be JOINED against this instead of
    // eyeballed - which is the whole point of measuring it.
    println!("\n=== per program: name worst_ulp lanes_moved half_lanes full_lanes ===");
    for (n, w, l, h, f) in &rows {
        println!("COND {n} {w} {l} {h} {f}");
    }
}

/// >>> A FULL-PRECISION MOVE CARRIES THE VIEW OF WHAT IT MOVED, and this is the rule that says
/// >>> so. See [`written_lane_precision`].
///
/// Without it the two loudest rows in the whole corpus differential were a NaN's SIGN BIT read
/// as a 4.2-billion-ULP disagreement, because the lane's last writer was an f32 move over a word
/// eleven half-precision instructions had filled.
#[test]
fn a_full_precision_move_carries_the_view_of_what_it_moved() {
    use vitaslop_gxp_shader::ir::{Instr, Operand, Predicate};

    let make = |op: Op, dest: Operand, srcs: Vec<Operand>, mask: [bool; 4], half: bool| Instr {
        op,
        pred: Predicate::Always,
        dest: Some(dest),
        write_mask: mask,
        srcs,
        half_precision: half,
        raw: 0,
        group: 0,
        blocked: None,
    };
    let pa = |i: u8| Operand::plain(Bank::PrimaryAttr, i, 2);
    let o = |i: u8| Operand::plain(Bank::Output, i, 1);

    // Half-precision arithmetic fills `pa[0]`/`pa[1]`, then a FULL-PRECISION move copies those
    // two whole words into `o[0]`/`o[1]` - the exact shape `mk-corpus-roof__frag_86c41b20` ends
    // with, and the one that made a NaN sign bit read as 4,227,955,712 ULP.
    let mut mov = make(Op::Mov, o(0), vec![pa(0)], [true, true, false, false], false);
    mov.srcs[0].swizzle = [0, 1, 0, 1];
    let shader = Shader {
        kind: ProgramKind::Fragment,
        instrs: vec![
            make(Op::Mul, pa(0), vec![pa(4), pa(6)], [true; 4], true),
            mov,
        ],
    };
    let prec = written_lane_precision(&shader);
    assert_eq!(prec.get(&(1, 0)), Some(&1), "o[0] holds two halves, not an f32: {prec:?}");
    assert_eq!(prec.get(&(1, 1)), Some(&1), "...and so does o[1]: {prec:?}");
    // `pa` IS compared (a fragment colour can land there), so its view travels; `sa` is an
    // input the runner never reads and was tracked only to make the propagation possible.
    assert!(prec.keys().all(|(b, _)| *b < 4), "sa must not travel: {prec:?}");
    assert_eq!(prec.get(&(3, 0)), Some(&1), "the pa lane the arithmetic filled is half: {prec:?}");

    // >>> AND THE INTEGER MOVE IS `OR` WITH ZERO, which is how this ISA spells a word copy -
    // there is no move opcode in the integer group. `cw-rr-corpus__frag_86687980` ends `r[0]`
    // with one, over a word holding two halves, and while it went unrecognised the two sides
    // read `0x43d2fe00` against `0x43d27e00`: one bit apart, the SIGN of a NaN.
    let or_zero = Op::Bitwise { kind: vitaslop_gxp_shader::ir::BitwiseKind::Or, imm: Some(0), lane_bits: 32 };
    let t = |i: u8| Operand::plain(Bank::Temp, i, 0);
    let shader = Shader {
        kind: ProgramKind::Fragment,
        instrs: vec![
            make(Op::Mul, t(4), vec![pa(4), pa(6)], [true; 4], true),
            make(or_zero, t(0), vec![t(4)], [true, false, false, false], false),
        ],
    };
    assert_eq!(
        written_lane_precision(&shader).get(&(0, 0)),
        Some(&1),
        "an OR with zero copies the word, so r[0] holds what t[4] held"
    );

    // A move whose SOURCE the program never wrote is an f32 lane, which is what this answered
    // for everything before the rule existed - the rule may not invent a view.
    let plain = Shader {
        kind: ProgramKind::Fragment,
        instrs: vec![make(Op::Mov, o(0), vec![pa(0)], [true, false, false, false], false)],
    };
    assert_eq!(written_lane_precision(&plain).get(&(1, 0)), Some(&0));

    // And a move at HALF precision is not a whole-word copy at all - it packs two channels into
    // one register, and the existing rule already answers for it.
    let half_mov = Shader {
        kind: ProgramKind::Fragment,
        instrs: vec![make(Op::Mov, o(0), vec![pa(0)], [true, true, false, false], true)],
    };
    assert_eq!(written_lane_precision(&half_mov).get(&(1, 0)), Some(&1));
}

/// >>> WHERE DO THE TWO SIDES FIRST DISAGREE? The differential says a 183-instruction program's
/// >>> register file is wrong at the end, which is not something anyone can fix.
///
/// This records a checksum of the whole register file at every TOP-LEVEL instruction boundary,
/// on both sides, and writes a case the browser runner (`gxptrace.mjs`) grades: the first
/// instruction index whose incoming state differs, and the instruction just before it, which is
/// the culprit. Both sides key by INDEX rather than by a step counter, so a taken branch cannot
/// put the two traces out of phase.
///
/// ```text
/// VITASLOP_GXP_CORPUS=<dir> VITASLOP_GXP_BLOB=<stem> VITASLOP_GXP_CASES_OUT=<dir> \
///   cargo test --release -p vitaslop-gxp-shader --test execcases -- --ignored --nocapture \
///   write_one_blob_as_a_trace_case
/// node vitaslop-web/e2e/gxptrace.mjs <dir>
/// ```
#[test]
#[ignore = "writes ONE blob's trace case; needs VITASLOP_GXP_CORPUS, VITASLOP_GXP_BLOB and VITASLOP_GXP_CASES_OUT"]
fn write_one_blob_as_a_trace_case() {
    use vitaslop_gxp_shader::wgsl::{emit_body_marked, trace_checksum, wrap_trace_module_for};

    let (Some(dir), Ok(want), Some(out)) =
        (corpus_dir(), std::env::var("VITASLOP_GXP_BLOB"), out_dir())
    else {
        eprintln!("set VITASLOP_GXP_CORPUS, VITASLOP_GXP_BLOB and VITASLOP_GXP_CASES_OUT");
        return;
    };
    std::fs::create_dir_all(&out).expect("create the case output directory");
    let path = dir.join(format!("{}.gxp", want.trim()));
    let bytes = std::fs::read(&path).unwrap_or_else(|e| panic!("read {}: {e}", path.display()));
    let program = Program::parse(&bytes).expect("parse");
    let kind = program.kind;
    let recompiled = match kind {
        ProgramKind::Vertex => recompile_vertex(&bytes).map(|r| r.shader),
        ProgramKind::Fragment => recompile_fragment(&bytes).map(|r| r.shader),
    };
    let shader = recompiled.expect("recompile");
    let marked = emit_body_marked(&shader).expect("emit");
    let windows = match kind {
        ProgramKind::Vertex => mem_windows_for_vertex_blob(&bytes),
        ProgramKind::Fragment => mem_windows_for_fragment_blob(&bytes),
    };
    let (module, _units, indices) = wrap_trace_module_for(&marked, kind, &windows);

    // The SAME seeded file the ordinary case uses - the CHOSEN OVERRIDES INCLUDED - so the
    // trace and the verdict are about one run of one program rather than two different ones.
    let seed = (program.hash ^ (program.hash >> 32)) as u32 | 1;
    let mem_words = mem_words_for(seed, &windows);
    let inputs = chosen_case_inputs(&shader, seed, &windows, &mem_words);
    let mut regs = seeded_case_regs(seed, &windows, &inputs);
    let end = indices.last().copied().unwrap_or(0);
    let mut trace = vec![0u32; end + 1];
    {
        let memory = |addr: u32| mem_fetch(addr, &windows, &mem_words);
        let mem_arg: Option<&dyn Fn(u32) -> u32> =
            if windows.is_empty() { None } else { Some(&memory) };
        let mut observe = |index: usize, r: &RegFile| {
            // LAST WRITE WINS, exactly as the emitted `gxp_trace[k] = ...` does in a loop.
            if let Some(slot) = trace.get_mut(index) {
                *slot = trace_checksum(&r.r, &r.o, &r.i, &r.pa, &r.p, &r.idx);
            }
        };
        let textures = case_textures;
        interp::run_traced(&shader, &mut regs, &textures, mem_arg, &mut observe)
            .expect("the reference must run to write a trace");
        trace[end] = trace_checksum(&regs.r, &regs.o, &regs.i, &regs.pa, &regs.p, &regs.idx);
    }

    let name = want.trim();
    let listing: Vec<String> = shader
        .instrs
        .iter()
        .enumerate()
        .map(|(i, ins)| format!("{i}: {} {:?}", ins.op.mnemonic(), ins.write_mask))
        .collect();
    let json = format!(
        "{{\"name\":\"{name}\",\"seed\":{seed},\"lanes\":{CASE_BANK_LANES},\"inputs\":[{}],\
         \"checkpoints\":[{}],\"trace\":[{}],\"mem\":[{}],\"instrs\":[{}]}}",
        // The chosen per-program overrides, so the runner feeds the trace module exactly what
        // the reference walked - see `chosen_case_inputs`.
        inputs.iter().map(|(l, v)| format!("[{l},{v}]")).collect::<Vec<_>>().join(","),
        indices.iter().map(|i| i.to_string()).collect::<Vec<_>>().join(","),
        trace.iter().map(|v| v.to_string()).collect::<Vec<_>>().join(","),
        // The guest-memory WINDOW words travel too: a program with 0xE8 loads declares a third
        // binding, and a runner that bound only the input, the output and the trace failed the
        // whole dispatch with "Number of entries (3) did not match the expected number of
        // entries (4)" - which reads as "the trace module did not run", and it ran nothing.
        mem_words.iter().map(|w| w.to_string()).collect::<Vec<_>>().join(","),
        listing.iter().map(|l| format!("{l:?}")).collect::<Vec<_>>().join(","),
    );
    std::fs::write(out.join(format!("{name}.trace.json")), json).expect("write the trace case");
    std::fs::write(out.join(format!("{name}.trace.wgsl")), &module).expect("write the module");
    println!(
        "wrote {name}: {} instructions, {} checkpoints, into {}",
        shader.instrs.len(),
        indices.len(),
        out.display()
    );
}

/// >>> A HALF-PRECISION FLOAT TEST READS ITS OPERANDS AS HALVES, and one hardwired default in
/// >>> the reference meant it did not.
///
/// `read_channel` is `read_channel_prec(.., Prec::F32)`, so the reference read a VTST in the F16
/// pipeline as whole 32-bit floats while `emit_test` read it at `Prec::of(instr)` - as halves.
/// The result is a PREDICATE, one bit that decides which arm of a branch runs, so everything
/// downstream disagreed. It was the FIRST divergence in five of the corpus remainder's programs
/// across three titles, and the per-instruction trace named `vtst` in every one.
///
/// # What makes this discriminating rather than merely green
/// The operand's two HALVES and the f32 the same word spells have OPPOSITE SIGN, so a reference
/// reading the wrong width does not get a nearby number - it gets the other side of the
/// comparison and the predicate inverts. A word whose halves and whose f32 agreed in sign would
/// pass under either reading.
#[test]
fn a_half_precision_float_test_reads_its_operands_as_halves() {
    use vitaslop_gxp_shader::ir::{Instr, Operand, Predicate, TestAlu, TestCmp, TestReduce};

    // A word whose LOW HALF is positive (0x3c00 = +1.0h) and which, read as one f32, is
    // NEGATIVE. 0xbf80_3c00 is -1.0000019 as an f32; its low half is +1.0h.
    const WORD: u32 = 0xbf80_3c00;
    let mut regs = RegFile::with_lanes(CASE_BANK_LANES);
    regs.sa[0] = f32::from_bits(WORD);
    regs.sa[1] = f32::from_bits(0x0000_0000); // zero in both views

    // `p0 = (sa[0] - sa[1]) >= 0`, in the F16 pipeline, then a move predicated on it so the
    // predicate is observable in a register.
    let sa = |i: u8| Operand::plain(Bank::SecondaryAttr, i, 3);
    let mk = |op: Op, dest: Option<Operand>, srcs: Vec<Operand>, half: bool, pred: Predicate| Instr {
        op,
        pred,
        dest,
        write_mask: [true, false, false, false],
        srcs,
        half_precision: half,
        raw: 0,
        group: 0,
        blocked: None,
    };
    let test = mk(
        Op::Test {
            alu: TestAlu::Sub,
            cmp: TestCmp::Ge,
            reduce: TestReduce::Channel(0),
            pdst: 0,
            write_back: false,
        },
        None,
        vec![sa(0), sa(1)],
        true,
        Predicate::Always,
    );
    let marker = mk(
        Op::Mov,
        Some(Operand::plain(Bank::Temp, 0, 0)),
        vec![sa(0)],
        false,
        Predicate::IfP(0),
    );
    let shader = Shader { kind: ProgramKind::Fragment, instrs: vec![test, marker] };
    let textures = case_textures;
    interp::run_watching_for_nan_with_env(&shader, &mut regs, &textures, None).expect("run");

    // The LOW HALF is +1.0h, so `+1 - 0 >= 0` holds and the predicated move runs.
    assert!(
        regs.p[0],
        "a half-precision test must read the LOW HALF (+1.0h), not the whole word as an f32 \
         ({}): the f32 view is negative and would clear the predicate",
        f32::from_bits(WORD)
    );
    assert_eq!(
        regs.r[0].to_bits(),
        WORD,
        "...and the predicated move must therefore have run"
    );

    // The same test at FULL precision reads the whole word, which is negative, so it does NOT
    // hold - which is what makes the assertion above a statement about the PRECISION rather
    // than about this particular word.
    let mut regs32 = RegFile::with_lanes(CASE_BANK_LANES);
    regs32.sa[0] = f32::from_bits(WORD);
    regs32.sa[1] = 0.0;
    let test32 = mk(
        Op::Test {
            alu: TestAlu::Sub,
            cmp: TestCmp::Ge,
            reduce: TestReduce::Channel(0),
            pdst: 0,
            write_back: false,
        },
        None,
        vec![sa(0), sa(1)],
        false,
        Predicate::Always,
    );
    let shader32 = Shader { kind: ProgramKind::Fragment, instrs: vec![test32] };
    interp::run_watching_for_nan_with_env(&shader32, &mut regs32, &textures, None).expect("run");
    assert!(
        !regs32.p[0],
        "the f32 reading of the same word is negative, so the two precisions must disagree - \
         otherwise this case cannot tell them apart"
    );
}

/// >>> WHICH SEEDED LANES DOES EACH PROGRAM READ AS AN INTEGER, AND WHAT DOES CHOOSING A
/// >>> PLAUSIBLE ONE ACTUALLY BUY - BOTH ARMS, ONE BUILD, ONE RUN.
///
/// 21d's own "do this next" named one lever and three symptoms: cases whose every written lane
/// is zero, cases whose every guest-memory load misses every bound window, and cases whose
/// data-driven loop never terminates. It also named the cause - the seed does not know what a
/// register MEANS - and proposed the fix.
///
/// This is the instrument for that claim, and it carries its OWN CONTROL: every program is run
/// twice in the same process off the same build, once on the seed alone and once on the inputs
/// [`best_plausible_inputs`] chose, and the three counters are reported side by side. A lever
/// that moved nothing would say so here rather than in a pass rate that cannot tell coverage
/// from agreement.
///
/// The "made worse" line is the one that matters. It is zero BY CONSTRUCTION - the search's own
/// candidate 0 is the empty override set - and it is printed anyway, because a structural
/// guarantee that is never measured is a structural guarantee nobody would notice losing.
///
/// ```text
/// VITASLOP_GXP_CORPUS=<dir> cargo test --release -p vitaslop-gxp-shader --test execcases \
///   -- --ignored --nocapture which_seeded_lanes_each_program_reads_as_integers
/// ```
#[test]
#[ignore = "a census over the whole corpus; needs VITASLOP_GXP_CORPUS"]
fn which_seeded_lanes_each_program_reads_as_integers() {
    let Some(dir) = corpus_dir() else {
        eprintln!("VITASLOP_GXP_CORPUS unset - nothing to do");
        return;
    };
    let mut files: Vec<PathBuf> = std::fs::read_dir(&dir)
        .expect("read the corpus directory")
        .flatten()
        .map(|e| e.path())
        .filter(|p| p.extension().and_then(|s| s.to_str()) == Some("gxp"))
        .collect();
    files.sort();

    #[derive(Default, Clone, Copy)]
    struct Arm {
        loads_total: u64,
        loads_inside: u64,
        all_missed: bool,
        spun: bool,
        wrote_nothing: bool,
        /// >>> WRITTEN LANES HOLDING A NON-ZERO VALUE - the counter the widened band needs and
        /// `wrote_nothing` is too coarse to be.
        ///
        /// The widened band hands a lane a small INTEGER BIT PATTERN that something reads as a
        /// float, and `4` as a float is the denormal `5.6e-45`. A program whose arithmetic that
        /// flattens still terminates, still lands its loads and still writes SOMETHING - so
        /// every counter above says it got better while its float results quietly went to zero,
        /// and a case whose expectation is all zeros agrees with an emitter that does nothing.
        ///
        /// This is that trade made visible. It is not part of the search's objective, because a
        /// program is not better for having big numbers in it; it is the number that would have
        /// to FALL for the widening to have been a bad bargain, and it is printed either way.
        nonzero_written: u64,
    }
    #[derive(Default)]
    struct Totals {
        loads_total: u64,
        loads_inside: u64,
        all_missed: usize,
        spun: usize,
        wrote_nothing: usize,
        nonzero_written: u64,
    }
    impl Totals {
        fn add(&mut self, a: Arm) {
            self.loads_total += a.loads_total;
            self.loads_inside += a.loads_inside;
            self.all_missed += usize::from(a.all_missed);
            self.spun += usize::from(a.spun);
            self.wrote_nothing += usize::from(a.wrote_nothing);
            self.nonzero_written += a.nonzero_written;
        }
    }

    let (mut ran, mut with_int, mut int_lanes_total) = (0usize, 0usize, 0usize);
    // How many programs the SEARCH actually moved off its own candidate 0.
    let mut chosen = 0usize;
    // Of those, how many the NARROW band could not have chosen - see the widened band's gate.
    let mut widened = 0usize;
    let (mut base, mut fixed) = (Totals::default(), Totals::default());
    let mut still_missing: Vec<String> = Vec::new();
    let mut still_spinning: Vec<String> = Vec::new();
    let mut still_zero: Vec<String> = Vec::new();
    let mut regressed: Vec<String> = Vec::new();

    for path in &files {
        let name = path.file_stem().unwrap_or_default().to_string_lossy().into_owned();
        let Ok(bytes) = std::fs::read(path) else { continue };
        let Ok(program) = Program::parse(&bytes) else { continue };
        let kind = program.kind;
        let recompiled = match kind {
            ProgramKind::Vertex => recompile_vertex(&bytes).map(|r| (r.shader, r.wgsl_body)),
            ProgramKind::Fragment => recompile_fragment(&bytes).map(|r| (r.shader, r.wgsl_body)),
        };
        let Ok((shader, body)) = recompiled else { continue };
        if rig_excludes(&shader).is_some() {
            continue;
        }
        let windows = match kind {
            ProgramKind::Vertex => mem_windows_for_vertex_blob(&bytes),
            ProgramKind::Fragment => mem_windows_for_fragment_blob(&bytes),
        };
        if body.contains("gxp_mem") && windows.is_empty() {
            continue;
        }
        let seed = (program.hash ^ (program.hash >> 32)) as u32 | 1;
        ran += 1;
        let candidate_lanes = plausible_inputs(&shader, seed, 1).len();
        int_lanes_total += candidate_lanes;
        if candidate_lanes > 0 {
            with_int += 1;
        }

        let mem_words = mem_words_for(seed, &windows);
        // ONE runner, used both for the search's scoring and for the two reported arms, so the
        // thing the search optimised and the thing the census reports cannot come apart.
        let run = |overrides: &[(usize, u32)]| -> (Arm, PlausibleScore) {
            let mut regs = RegFile::with_lanes(CASE_BANK_LANES);
            for n in 0..CASE_BANK_LANES {
                regs.pa[n] = lane_value(seed, n as u32);
                regs.sa[n] = lane_value(seed, (n + CASE_BANK_LANES) as u32);
            }
            apply_plausible_inputs(&mut regs, overrides);
            regs.facing = Some(true);
            for (i, win) in windows.iter().enumerate() {
                if let Some(slot) = regs.sa.get_mut(win.base_sa as usize) {
                    *slot = f32::from_bits(window_base(i));
                }
            }
            let baseline = seeded_pa_baseline(seed, overrides);
            let loads = std::cell::Cell::new((0u32, 0u32));
            let memory = |addr: u32| {
                let inside = placements_hit(addr, &windows);
                let (t, i) = loads.get();
                loads.set((t + 1, i + u32::from(inside)));
                mem_fetch(addr, &windows, &mem_words)
            };
            let mem_arg: Option<&dyn Fn(u32) -> u32> =
                if windows.is_empty() { None } else { Some(&memory) };
            let textures = case_textures;
            let outcome =
                interp::run_watching_for_nan_with_env(&shader, &mut regs, &textures, mem_arg);
            let (lt, li) = loads.get();
            let mut arm =
                Arm { loads_total: u64::from(lt), loads_inside: u64::from(li), ..Arm::default() };
            let mut score =
                PlausibleScore { loads_inside: li, loads_total: lt, ..PlausibleScore::default() };
            match outcome {
                Err(interp::InterpError::UnsupportedOp { op, .. }) if op.contains("step budget") => {
                    arm.spun = true;
                }
                Err(_) => {}
                Ok(_) => {
                    score.terminated = true;
                    arm.all_missed = lt > 0 && li == 0;
                    let wrote = regs.r.iter().chain(regs.o.iter()).chain(regs.i.iter()).any(|v| v.to_bits() != 0)
                        || regs.pa.iter().zip(baseline.iter()).any(|(a, b)| a.to_bits() != b.to_bits());
                    arm.wrote_nothing = !wrote;
                    score.wrote_something = wrote;
                    // >>> A DENORMAL IS NOT A ZERO, and counting it as one would hide exactly the
                    // thing this counter exists to see: `4` read as a float is `5.6e-45`, and a
                    // program whose arithmetic that flattens produces values that are not zero
                    // and not distinguishable from it by any expectation worth checking.
                    score.writes_a_denormal = writes_a_denormal(&regs);
                    arm.nonzero_written = regs
                        .r
                        .iter()
                        .chain(regs.o.iter())
                        .chain(regs.i.iter())
                        .filter(|v| v.is_normal() || v.is_infinite())
                        .count() as u64;
                }
            }
            (arm, score)
        };
        let (a, _) = run(&[]);
        let (picked, _) = best_plausible_inputs(&shader, seed, |o| run(o).1);
        if !picked.is_empty() {
            chosen += 1;
        }
        // A chosen set the NARROW band could not have produced: it names a lane no narrow role
        // takes. Asked of the answer rather than of the search, so it says what the case file
        // will actually carry.
        if !picked.is_empty() {
            let (pa_reads, sa_reads) = seeded_lane_reads(&shader);
            let role_of = |lane: usize| {
                if lane < CASE_BANK_LANES {
                    pa_reads.get(lane).copied()
                } else {
                    sa_reads.get(lane - CASE_BANK_LANES).copied()
                }
            };
            if picked.iter().any(|&(lane, _)| role_of(lane).is_some_and(|r| r.read_both_ways())) {
                widened += 1;
            }
        }
        let (b, _) = run(&picked);
        // >>> LIKE WINDOWS. A load count from an interpretation that SPUN is 1.8 million loads
        // of a loop the program never runs, and it swamps both arms' totals with one number that
        // is not about coverage at all. Only a program that terminated in BOTH arms contributes
        // its loads, so the percentage compares the same programs on both sides.
        // >>> AND THE SAME WINDOW FOR `nonzero_written`, for the same reason and a sharper one:
        // an arm that SPUN wrote nothing this counter can see, so a program that spun before and
        // terminates now would add its whole register file to one side of a comparison meant to
        // show whether values were LOST. That is a rise manufactured by the fix it is auditing.
        let comparable = !a.spun && !b.spun;
        let windowed =
            |x: Arm| if comparable { x } else { Arm { loads_total: 0, loads_inside: 0, nonzero_written: 0, ..x } };
        base.add(windowed(a));
        fixed.add(windowed(b));
        if b.all_missed {
            still_missing.push(name.clone());
        }
        if b.spun {
            still_spinning.push(name.clone());
        }
        if b.wrote_nothing {
            // >>> WHAT THE PROGRAM IS, not only that it wrote nothing. Measured for the 17: a
            // program that reads no seeded lane at all cannot be reached by ANY choice of
            // inputs, so counting it against the seed is counting it against the wrong thing.
            let effective =
                shader.instrs.iter().filter(|i| !matches!(i.op, Op::Nop | Op::Branch { .. })).count();
            let reads = {
                let (pa, sa) = seeded_lane_reads(&shader);
                pa.iter().chain(sa.iter()).filter(|r| r.integer || r.float || r.float_to_int).count()
            };
            still_zero.push(format!("{name}  ({effective} effective instr(s), {reads} seeded lane(s) read)"));
        }
        if (b.all_missed && !a.all_missed)
            || (b.spun && !a.spun)
            || (b.wrote_nothing && !a.wrote_nothing)
            || b.loads_inside < a.loads_inside
        {
            regressed.push(name.clone());
        }
    }

    println!("\n=== HOW EACH PROGRAM READS ITS SEEDED LANES, AND WHAT A PLAUSIBLE ONE BUYS ===");
    println!(
        "  {ran} programs the rig can run; {with_int} of them have at least one INTEGER-ONLY \
         lane ({int_lanes_total} such lanes in total); the search chose a non-empty set for \
         {chosen}"
    );
    let pct = |i: u64, t: u64| if t == 0 { 0.0 } else { 100.0 * i as f64 / t as f64 };
    println!("\n{:<38}{:>16}{:>16}", "", "seed alone", "+ plausible");
    println!(
        "{:<38}{:>16}{:>16}",
        "  loads landing inside a window",
        format!("{} / {}", base.loads_inside, base.loads_total),
        format!("{} / {}", fixed.loads_inside, fixed.loads_total)
    );
    println!(
        "{:<38}{:>15.1}%{:>15.1}%",
        "    as a percentage",
        pct(base.loads_inside, base.loads_total),
        pct(fixed.loads_inside, fixed.loads_total)
    );
    for (label, a, b) in [
        ("  programs where EVERY load missed", base.all_missed, fixed.all_missed),
        ("  programs that SPUN past the budget", base.spun, fixed.spun),
        ("  programs that WROTE NOTHING", base.wrote_nothing, fixed.wrote_nothing),
    ] {
        println!("{label:<38}{a:>16}{b:>16}");
    }
    // >>> THE COUNTER THAT WOULD HAVE TO FALL FOR THIS TO HAVE BEEN A BAD BARGAIN. See
    // `Arm::nonzero_written`: the widened band trades a lane's float view for its integer one,
    // and the way that goes wrong is silent - every counter above improves while the arithmetic
    // flattens to denormals and the case's expectation becomes zeros that agree with anything.
    println!(
        "{:<38}{:>16}{:>16}",
        "  written lanes holding a NORMAL value", base.nonzero_written, fixed.nonzero_written
    );
    println!(
        "
  the WIDENED band - the only draws that may substitute a lane something reads as a          FLOAT - chose the inputs for {widened} program(s); it is offered only to a program the          narrow draws left covering nothing at all"
    );
    println!(
        "\n  >>> AND WHAT IT MADE WORSE: {} program(s). Zero by construction - the search's own \
         candidate 0 is the empty set - and measured anyway.",
        regressed.len()
    );
    for n in regressed.iter().take(12) {
        println!("      {n}");
    }
    if regressed.len() > 12 {
        println!("      ... and {} more", regressed.len() - 12);
    }
    for (label, rows) in [
        ("EVERY LOAD STILL MISSES", &still_missing),
        ("STILL SPINS", &still_spinning),
        ("STILL WRITES NOTHING", &still_zero),
    ] {
        println!("\n  {label}: {} program(s)", rows.len());
        for n in rows.iter().take(20) {
            println!("      {n}");
        }
        if rows.len() > 20 {
            println!("      ... and {} more", rows.len() - 20);
        }
    }
}

/// >>> WHAT ONE PROGRAM READS EACH SEEDED LANE AS, instruction by instruction.
///
/// The corpus census says which POPULATIONS a per-program seed could reach. When the answer is
/// "none of them", the next question is why - and a count of lanes cannot answer it. This prints
/// the stream with each instruction's operands classified, and then every seeded lane with the
/// verdict the analysis reached, so the disagreement between "this is plainly an address" and
/// "this lane is read as a float" can be read off rather than argued about.
///
/// ```text
/// VITASLOP_GXP_CORPUS=<dir> VITASLOP_GXP_BLOB=<stem> \
///   cargo test --release -p vitaslop-gxp-shader --test execcases -- --ignored --nocapture \
///   print_one_programs_seeded_lane_roles
/// ```
#[test]
#[ignore = "needs a captured corpus and VITASLOP_GXP_BLOB"]
fn print_one_programs_seeded_lane_roles() {
    let (Some(dir), Ok(want)) = (corpus_dir(), std::env::var("VITASLOP_GXP_BLOB")) else {
        eprintln!("set VITASLOP_GXP_CORPUS and VITASLOP_GXP_BLOB");
        return;
    };
    let path = dir.join(format!("{}.gxp", want.trim()));
    let Ok(bytes) = std::fs::read(&path) else {
        eprintln!("no such blob: {}", path.display());
        return;
    };
    let program = Program::parse(&bytes).expect("parse");
    let kind = program.kind;
    let (shader, _body) = match kind {
        ProgramKind::Vertex => recompile_vertex(&bytes).map(|r| (r.shader, r.wgsl_body)),
        ProgramKind::Fragment => recompile_fragment(&bytes).map(|r| (r.shader, r.wgsl_body)),
    }
    .expect("recompile");
    let windows = match kind {
        ProgramKind::Vertex => mem_windows_for_vertex_blob(&bytes),
        ProgramKind::Fragment => mem_windows_for_fragment_blob(&bytes),
    };
    println!("{want}: {kind:?}, {} instrs, {} window(s)", shader.instrs.len(), windows.len());
    for (i, w) in windows.iter().enumerate() {
        println!("  window {i}: base {:#010x}, pointer in sa[{}]", window_base(i), w.base_sa);
    }
    let bank_name = |c: u8| match c {
        0 => "r",
        1 => "o",
        2 => "i",
        3 => "pa",
        4 => "sa",
        _ => "?",
    };
    println!("\n  == the stream, with each operand's view ==");
    for (n, instr) in shader.instrs.iter().enumerate() {
        let ints = integer_sink_operands(instr);
        let mut parts: Vec<String> = Vec::new();
        if let Some((b, lanes)) = dest_lanes(instr) {
            let (lo, hi) = (lanes.first().copied().unwrap_or(0), lanes.last().copied().unwrap_or(0));
            parts.push(format!("-> {}[{lo}..={hi}]", bank_name(b)));
        }
        for which in 0..instr.srcs.len() {
            let Some((b, lanes)) = src_lanes(instr, which) else {
                parts.push(format!("s{which}=<other bank>"));
                continue;
            };
            let (lo, hi) = (lanes.first().copied().unwrap_or(0), lanes.last().copied().unwrap_or(0));
            parts.push(format!(
                "s{which}={}[{lo}..={hi}]{}",
                bank_name(b),
                if ints.contains(&which) { " INT" } else { "" }
            ));
        }
        println!(
            "    #{n:<4} {:<14} {:<26} {}",
            instr.op.mnemonic(),
            format!("{:?}", instr.pred),
            parts.join("  ")
        );
    }
    let (pa_reads, sa_reads) = seeded_lane_reads(&shader);
    println!("\n  == how each seeded lane is read ==");
    for (label, reads) in [("pa", &pa_reads), ("sa", &sa_reads)] {
        for (n, r) in reads.iter().enumerate() {
            if !r.integer && !r.float {
                continue;
            }
            println!(
                "    {label}[{n}]  {}{}{}",
                if r.integer { "INT " } else { "" },
                if r.float { "float " } else { "" },
                if r.wants_an_integer() { " <- would be seeded as an integer" } else { "" }
            );
        }
    }
}

// =================================================================================================
// >>> THE PER-PROGRAM INPUT RULES, PINNED.
//
// Each of the four below is a rule that was WRONG first and measured second, and each of them
// fails loudly under this test with the wrong version restored. A dataflow analysis is exactly
// the kind of code that agrees with itself while answering nothing - the first version of this
// one reported 84 of a program's 56 seeded lanes "read both ways" and looked like a finished
// instrument - so the rules are asserted rather than described.
// =================================================================================================

/// Build one instruction for the rule tests below.
fn pin_instr(op: Op, dest: Option<Operand>, write_mask: [bool; 4], srcs: Vec<Operand>) -> Instr {
    Instr {
        op,
        pred: Predicate::Always,
        dest,
        write_mask,
        srcs,
        half_precision: false,
        raw: 0,
        group: 0,
        blocked: None,
    }
}

fn pin_shader(instrs: Vec<Instr>) -> Shader {
    Shader { kind: ProgramKind::Vertex, instrs }
}

/// >>> AN OPERAND READS THE REGISTERS ITS SWIZZLE NAMES, NOT FOUR CONSECUTIVE ONES.
///
/// MEASURED: with the fixed four-register span, a football title's skinning program read
/// `pa[4..=7]` as a float at an instruction that reads only `pa[4]` - and `pa[7]` is a guest byte
/// pointer three instructions later. Every one of the 88 programs whose loads all missed came
/// back with no integer-only lane at all, and the whole lever read as useless.
///
/// The source swizzled `[0,0,0,0]` is the discriminating one: a fixed span says four registers,
/// the selector says one.
#[test]
fn a_source_operand_reads_the_registers_its_swizzle_names() {
    let mut src = Operand::plain(Bank::PrimaryAttr, 4, 0);
    src.swizzle = [0, 0, 0, 0];
    let instr = pin_instr(
        Op::Add,
        Some(Operand::plain(Bank::Temp, 0, 0)),
        [true, true, true, true],
        vec![src, Operand::plain(Bank::SecondaryAttr, 0, 0)],
    );
    assert_eq!(src_lanes(&instr, 0), Some((3, vec![4])), "one selector names one register");

    // And a plain `.xyzw` really does reach four - the narrowing must not have become a
    // blanket "one lane", which would miss every operand that reads a vector.
    let plain = pin_instr(
        Op::Add,
        Some(Operand::plain(Bank::Temp, 0, 0)),
        [true, true, true, true],
        vec![Operand::plain(Bank::PrimaryAttr, 4, 0), Operand::plain(Bank::SecondaryAttr, 0, 0)],
    );
    assert_eq!(src_lanes(&plain, 0), Some((3, vec![4, 5, 6, 7])));

    // A memory load's pointer is a RAW LANE - no selector, whatever the swizzle says.
    let mut ptr = Operand::plain(Bank::PrimaryAttr, 4, 0);
    ptr.swizzle = [2, 2, 2, 2];
    let load = pin_instr(
        Op::MemLoad { elements: 4, offset_bytes: 0 },
        Some(Operand::plain(Bank::Temp, 0, 0)),
        [true; 4],
        vec![ptr],
    );
    assert_eq!(src_lanes(&load, 0), Some((3, vec![4])), "an address ignores the swizzle");
}

/// >>> A VALUE READ OUT OF A BOUND RESOURCE DOES NOT COME FROM THE REGISTER THAT NAMED IT.
///
/// MEASURED, and it was the entire remainder of the first honest run: a skinning program loads a
/// matrix row through `pa[4]` and multiplies that row as a FLOAT. Carrying the pointer's roots
/// into the loaded row marked the blend index, the palette stride and the window base "read as a
/// float" - all three integers, all three then ineligible.
///
/// Here `pa[9]` is an address (a raw-lane integer read), `r[0]` is what the load returned, and
/// the float multiply of `r[0]` must NOT reach back to `pa[9]`.
#[test]
fn a_loaded_value_does_not_inherit_its_addresss_origins() {
    let shader = pin_shader(vec![
        pin_instr(
            Op::MemLoad { elements: 1, offset_bytes: 0 },
            Some(Operand::plain(Bank::Temp, 0, 0)),
            [true; 4],
            vec![Operand::plain(Bank::PrimaryAttr, 9, 0)],
        ),
        pin_instr(
            Op::Mul,
            Some(Operand::plain(Bank::Output, 0, 0)),
            [true, false, false, false],
            vec![Operand::plain(Bank::Temp, 0, 0), Operand::plain(Bank::SecondaryAttr, 0, 0)],
        ),
    ]);
    let (pa, _) = seeded_lane_reads(&shader);
    assert!(pa[9].integer, "the address is read as an integer");
    assert!(!pa[9].float, "the loaded row's float multiply must not reach the pointer");
    assert!(pa[9].wants_an_integer(), "so the pointer is eligible for a plausible integer");
}

/// >>> A LANE READ AS A FLOAT BY ANYTHING KEEPS ITS FLOAT, AND ONE WHOSE ONLY FLOAT READER
/// >>> TRUNCATES IT GETS A SMALL INDEX INSTEAD.
///
/// The two halves are one rule and they fail in opposite directions. A lane that some float
/// multiply reads cannot hold an integer bit pattern - `4` is the denormal `5.6e-45` and drives
/// that program's arithmetic to zero. A lane whose only float reader is `pack.int` cannot hold
/// one either - the cast would truncate the denormal to zero - and must not keep the seed's
/// `[-4, 4)`, half of which is negative and indexes below a window's base.
#[test]
fn only_an_unmixed_lane_is_substituted_and_a_truncated_one_gets_a_float() {
    // pa[4] is read ONLY by the float-to-int cast; pa[8] by an ordinary float multiply as well.
    let shader = pin_shader(vec![
        pin_instr(
            Op::PackToInt { bits: 32, signed: true, src_half: false },
            Some(Operand::plain(Bank::Temp, 0, 0)),
            [true, false, false, false],
            vec![Operand::plain(Bank::PrimaryAttr, 4, 0)],
        ),
        pin_instr(
            Op::Bitwise { kind: BitwiseKind::Or, imm: Some(0), lane_bits: 32 },
            Some(Operand::plain(Bank::Temp, 1, 0)),
            [true, false, false, false],
            vec![Operand::plain(Bank::PrimaryAttr, 8, 0)],
        ),
        pin_instr(
            Op::Mul,
            Some(Operand::plain(Bank::Output, 0, 0)),
            [true, false, false, false],
            vec![Operand::plain(Bank::PrimaryAttr, 8, 0), Operand::plain(Bank::SecondaryAttr, 0, 0)],
        ),
    ]);
    let (pa, _) = seeded_lane_reads(&shader);
    assert!(pa[4].float_to_int && !pa[4].float);
    assert!(pa[4].wants_a_small_float(), "a truncated attribute wants a small non-negative float");
    assert!(!pa[4].wants_an_integer(), "and NOT an integer bit pattern the cast would flatten");

    assert!(pa[8].integer && pa[8].float, "read both ways");
    assert!(!pa[8].wants_an_integer() && !pa[8].wants_a_small_float(), "so it is left alone");

    // And the substituted value really is the small float, exactly representable.
    let inputs = plausible_inputs(&shader, 0x1234_5678, 1);
    let (_, bits) = inputs.iter().find(|(l, _)| *l == 4).copied().expect("pa[4] substituted");
    let v = f32::from_bits(bits);
    assert!((1.0..=15.0).contains(&v) && v.fract() == 0.0, "a small whole index, got {v}");
    assert!(inputs.iter().all(|(l, _)| *l != 8), "pa[8] must not be substituted");
}

/// >>> THE WIDENED BAND IS OFFERED ONLY TO A PROGRAM THAT COVERS NOTHING, and it is the ONLY
/// >>> thing that can hand a value to a lane something reads as a float.
///
/// The narrow roles refuse a both-ways lane for a good reason - a small integer's bit pattern is
/// a denormal, and a float reader would get zero - and that refusal left 57 programs whose every
/// guest load misses every window and 8 that never terminate. The refusal is right as a RULE and
/// wrong as a VERDICT, and the difference is measurable: a program already landing loads has
/// something to lose, a program landing none does not.
///
/// Both halves are pinned here, because the gate failing OPEN is the whole risk. A program that
/// covers nothing must reach the band; a program that covers something must never see it, no
/// matter how much the widened draw might have improved some other number.
#[test]
fn the_widened_band_reaches_a_both_ways_lane_only_when_there_is_nothing_to_lose() {
    // pa[8] addresses a load AND is multiplied as a float - the shape the narrow roles refuse.
    let shader = pin_shader(vec![
        pin_instr(
            Op::MemLoad { elements: 1, offset_bytes: 0 },
            Some(Operand::plain(Bank::Temp, 0, 0)),
            [true; 4],
            vec![Operand::plain(Bank::PrimaryAttr, 8, 0)],
        ),
        pin_instr(
            Op::Mul,
            Some(Operand::plain(Bank::Output, 0, 0)),
            [true, false, false, false],
            vec![Operand::plain(Bank::PrimaryAttr, 8, 0), Operand::plain(Bank::SecondaryAttr, 0, 0)],
        ),
    ]);
    let (pa, _) = seeded_lane_reads(&shader);
    assert!(pa[8].read_both_ways(), "the lane addresses a load and is read as a float");
    assert!(!pa[8].wants_an_integer() && !pa[8].wants_a_small_float(), "no narrow role takes it");
    assert!(plausible_inputs(&shader, 1, 1).is_empty(), "and the narrow band offers nothing");

    // >>> EVERY WIDENED DRAW HANDS IT A SMALL WHOLE FLOAT, AND NEVER A SUBNORMAL. A second arm
    // offering the integer's raw BIT PATTERN was built and refuted on the GPU: `11` as a float
    // is `1.5e-44`, the GPU flushes it and the reference does not, and at a `vtst` that flips a
    // predicate and sends the two sides down different branches from instruction 2. See
    // `PLAUSIBLE_ATTEMPTS_WIDENED`. So the range is asserted here, on every draw in the band -
    // a single sampled draw would not notice one arm of it going subnormal again.
    for attempt in PLAUSIBLE_ATTEMPTS..PLAUSIBLE_ATTEMPTS_WIDENED {
        let got = plausible_inputs(&shader, 1, attempt);
        assert_eq!(got.len(), 1, "draw {attempt} substitutes the both-ways lane");
        assert_eq!(got[0].0, 8);
        let v = f32::from_bits(got[0].1);
        assert!(
            v.is_normal() && (1.0..=15.0).contains(&v) && v.fract() == 0.0,
            "draw {attempt} must hand a both-ways lane a small whole NORMAL float, got {v} \
             ({:#010x})",
            got[0].1
        );
    }

    // >>> THE GATE, CLOSED. A program that lands a load is never offered the band, even when
    // every widened draw would have scored better - which is exactly what this scorer says.
    let mut widened_seen = false;
    let (picked, _) = best_plausible_inputs(&shader, 1, |o| {
        widened_seen |= !o.is_empty();
        PlausibleScore {
            terminated: true,
            loads_inside: if o.is_empty() { 1 } else { 99 },
            loads_total: 1,
            wrote_something: true,
            writes_a_denormal: false,
        }
    });
    assert!(!widened_seen, "the narrow band offers nothing here, so NOTHING may be tried");
    assert!(picked.is_empty(), "and a program with coverage keeps the inputs it had");

    // >>> THE GATE, OPEN. The same program landing none of its loads reaches the band, and the
    // draw that lands one wins.
    let (picked, score) = best_plausible_inputs(&shader, 1, |o| PlausibleScore {
        terminated: true,
        loads_inside: u32::from(!o.is_empty()),
        loads_total: 1,
        wrote_something: true,
        writes_a_denormal: false,
    });
    assert_eq!(picked.len(), 1, "the widened draw is taken when the control covered nothing");
    assert_eq!(picked[0].0, 8);
    assert_eq!(score.loads_inside, 1);

    // And a program that issues NO load at all is not a coverage hole, so it is not offered
    // the trade either - `covers_nothing` needs the denominator, not only the numerator.
    let mut widened_seen = false;
    best_plausible_inputs(&shader, 1, |o| {
        widened_seen |= !o.is_empty();
        PlausibleScore {
            terminated: true,
            loads_inside: 0,
            loads_total: 0,
            wrote_something: true,
            writes_a_denormal: false,
        }
    });
    assert!(!widened_seen, "no load issued is not a load missed");
}

/// >>> A CANDIDATE WHOSE EXPECTATION IS A DENORMAL LOSES A TIE, and nothing more than a tie.
///
/// MEASURED, and it cost a whole corpus run: the widened band's first run turned 5 divergences
/// into 23, and **18 of the new ones were one non-defect** - a reference of `0x0000000b` against
/// a GPU zero, graded as 11 ULP of translation error. WGSL permits flushing subnormals, every
/// backend the runner reaches does it, and the emitter was right every time.
///
/// >>> AND IT WAS A VETO FOR EXACTLY ONE RUN, which is the part worth keeping. Refusing such a
/// candidate outright also threw away every OTHER lane it checked: the same search scored 89.8%
/// of guest loads landing inside a window as a preference and 61.5% as a veto. The right place
/// for the rule turned out to be the GRADER - `gxpexec.mjs` now names a subnormal-against-zero
/// as the permitted flush it is and counts it - after which the search only has to PREFER a
/// candidate that checks real values, and last, behind every load it could land.
#[test]
fn a_candidate_whose_expectation_is_a_denormal_loses_a_tie_and_only_a_tie() {
    let shader = pin_shader(vec![pin_instr(
        Op::MemLoad { elements: 1, offset_bytes: 0 },
        Some(Operand::plain(Bank::Temp, 0, 0)),
        [true; 4],
        vec![Operand::plain(Bank::PrimaryAttr, 9, 0)],
    )]);
    assert!(!plausible_inputs(&shader, 1, 1).is_empty(), "there is something to substitute");

    // A candidate that lands loads STILL WINS even though its expectation carries a denormal:
    // the flush costs one lane's check, and the loads it lands are every other lane's.
    let (picked, score) = best_plausible_inputs(&shader, 1, |o| PlausibleScore {
        terminated: true,
        loads_inside: if o.is_empty() { 0 } else { 999 },
        loads_total: 1,
        wrote_something: true,
        writes_a_denormal: !o.is_empty(),
    });
    assert!(!picked.is_empty(), "a denormal lane does not outweigh 999 loads that landed");
    assert_eq!(score.loads_inside, 999);

    // >>> BUT AT EQUAL COVERAGE THE CLEAN ONE WINS. Every candidate lands one load; only the
    // first draw's expectation carries a denormal, so the search must move off it.
    let mut seen_clean = false;
    let (picked, score) = best_plausible_inputs(&shader, 1, |o| {
        let denormal = !o.is_empty() && !seen_clean;
        if !o.is_empty() {
            seen_clean = true;
        }
        PlausibleScore {
            terminated: true,
            loads_inside: u32::from(!o.is_empty()),
            loads_total: 1,
            wrote_something: true,
            writes_a_denormal: denormal,
        }
    });
    assert!(!picked.is_empty(), "a candidate that lands a load beats the control");
    assert!(!score.writes_a_denormal, "and at equal coverage the one without a denormal wins");
}

/// >>> THE SEARCH CANNOT RETURN SOMETHING WORSE THAN THE INPUTS THERE WERE BEFORE IT EXISTED.
///
/// That is the whole safety argument for substituting anything at all, and it rests on one line:
/// candidate 0 is the EMPTY override set. MEASURED that it matters - handing every integer-only
/// lane one independent small value without a search moved the all-loads-missed population from
/// 88 to 65 and the SPINNING population from 12 to 21, because a loop's counter and its bound
/// are two lanes and two independent draws order them wrongly as easily as the float bit
/// patterns did.
///
/// The scorer here refuses every non-empty candidate, which is the adversarial case: the search
/// must come back with nothing rather than with the last thing it tried.
#[test]
fn the_input_search_falls_back_to_the_inputs_that_were_there_before() {
    let shader = pin_shader(vec![pin_instr(
        Op::MemLoad { elements: 1, offset_bytes: 0 },
        Some(Operand::plain(Bank::Temp, 0, 0)),
        [true; 4],
        vec![Operand::plain(Bank::PrimaryAttr, 9, 0)],
    )]);
    assert!(!plausible_inputs(&shader, 1, 1).is_empty(), "there is something to substitute");

    let mut tried = 0usize;
    let (picked, score) = best_plausible_inputs(&shader, 1, |o| {
        tried += 1;
        // Everything but the empty set scores as a program that did not even terminate.
        PlausibleScore {
            terminated: o.is_empty(),
            loads_inside: 0,
            loads_total: 0,
            wrote_something: o.is_empty(),
            writes_a_denormal: false,
        }
    });
    assert!(picked.is_empty(), "no candidate beat the control, so the control stands");
    assert!(score.terminated);
    assert_eq!(tried, PLAUSIBLE_ATTEMPTS as usize, "every draw is offered, and only better wins");

    // And when one IS better, it is taken.
    let (picked, _) = best_plausible_inputs(&shader, 1, |o| PlausibleScore {
        terminated: true,
        loads_inside: u32::from(!o.is_empty()),
        loads_total: 1,
        wrote_something: true,
        writes_a_denormal: false,
    });
    assert!(!picked.is_empty(), "a candidate that lands a load must win");
}

/// >>> A BRANCH SKIPS A RANGE, NOT A PROGRAM.
///
/// `written_lane_precision` answers "is a write in reach of a branch" with one flag over the
/// whole shader, and for the question IT asks that costs nothing. Here it costs everything: this
/// walk uses the answer to decide whether a write REPLACES a lane's provenance, so a shader-wide
/// flag means nothing is ever replaced and provenance only accumulates - the exact failure the
/// live-state walk exists to remove, one level up.
#[test]
fn a_branch_puts_only_the_words_it_jumps_over_in_doubt() {
    // #0 writes pa[0] from a float; #1 branches over #2; #3 is past the branch entirely.
    let shader = pin_shader(vec![
        pin_instr(
            Op::Mul,
            Some(Operand::plain(Bank::Temp, 0, 0)),
            [true, false, false, false],
            vec![Operand::plain(Bank::PrimaryAttr, 0, 0), Operand::plain(Bank::SecondaryAttr, 0, 0)],
        ),
        pin_instr(Op::Branch { rel: 2 }, None, [false; 4], vec![]),
        pin_instr(
            Op::Mul,
            Some(Operand::plain(Bank::Temp, 1, 0)),
            [true, false, false, false],
            vec![Operand::plain(Bank::PrimaryAttr, 1, 0), Operand::plain(Bank::SecondaryAttr, 0, 0)],
        ),
        // A determined whole-lane write from a fresh integer source; it must REPLACE r[2]'s
        // provenance, so the float lane pa[2] below it does not leak into the load's address.
        pin_instr(
            Op::Bitwise { kind: BitwiseKind::Or, imm: Some(0), lane_bits: 32 },
            Some(Operand::plain(Bank::Temp, 2, 0)),
            [true, false, false, false],
            vec![Operand::plain(Bank::PrimaryAttr, 3, 0)],
        ),
        pin_instr(
            Op::MemLoad { elements: 1, offset_bytes: 0 },
            Some(Operand::plain(Bank::Output, 0, 0)),
            [true; 4],
            vec![Operand::plain(Bank::Temp, 2, 0)],
        ),
    ]);
    let (pa, _) = seeded_lane_reads(&shader);
    assert!(pa[3].wants_an_integer(), "the pointer's source is integer-only");
    assert!(pa[0].float && !pa[0].integer, "a lane before the branch stays what it was read as");
}
