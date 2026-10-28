//! The clip position: interpreting a pair's vertex program over its own mesh to learn its
//! projection, and the fixup that maps GXM's clip space onto WebGPU's.

use super::*;

/// The depth encoding to assume for a pass whose own draws could not produce one: `-1/w`,
/// i.e. `a = 0, c = -1`. It is the encoding with no projection in it, so a frame built on
/// it is visibly a fallback rather than a plausible-looking guess.
pub(super) const DEPTH_FIT_RECIP_W: (f32, f32) = (0.0, -1.0);

/// What interpreting a pair's vertex program over its OWN mesh says about the projection
/// behind it. See [`count_clip_w_signs`].
#[derive(Clone, Debug, Default)]
pub(super) struct ClipStats {
    /// Sampled vertices landing inside the frustum with clip `w > 0`, and with `w < 0`.
    pub(super) in_front: usize,
    pub(super) behind: usize,
    pub(super) sampled: usize,
    /// NDC bounding box under `x/|w|`.
    pub(super) bbox: [f32; 4],
    /// `(a, c)` of `z_clip = a * w_clip + c`, and the spread of `w` the fit was taken over.
    ///
    /// A projection matrix makes clip `z` an AFFINE function of clip `w` - both are the same
    /// dot product against the same point, differing only in which matrix column they use -
    /// so this holds EXACTLY, per vertex, for every draw sharing one projection. It is what
    /// gives the guest's own window depth `z/w = a + c/w` without reflecting the matrix, and
    /// therefore what lets a depth surface be re-encoded the way the guest wrote it rather
    /// than the way we guessed. `None` when this draw has no two vertices with different
    /// `w` to fit through.
    pub(super) depth_fit: Option<(f32, f32)>,
    pub(super) w_spread: f32,
    /// The FIRST sampled vertex, as `(model-space position, clip vector)`.
    ///
    /// A bounding box says a draw covers nothing; it cannot say WHY, and the two answers -
    /// "the position it was handed is wrong" and "the transform it applied is wrong" - want
    /// opposite investigations. One vertex in and its four clip components out separates
    /// them by inspection, and a wrong SCALE (the common shape) is visible immediately.
    pub(super) first_vertex: Option<([f32; 4], [f32; 4])>,
    /// Guest-memory reads this interpretation made, and how many landed in NO window and
    /// therefore read ZERO. A transform read as zeroes collapses every vertex onto the
    /// origin, which cannot be told from a draw that was never issued by looking at the
    /// frame - so the count is what separates "the address arithmetic is wrong" from "the
    /// geometry is somewhere else". See `read_mem` in `count_clip_w_signs`.
    pub(super) mem_reads: u32,
    pub(super) mem_misses: u32,
    /// The first address that missed, the first that hit, and the windows' own bases.
    pub(super) first_miss: u32,
    pub(super) first_hit: u32,
    pub(super) window_bases: Vec<(u32, u32)>,
}

/// Interpret a pair's vertex program over its OWN mesh and measure what it says about the
/// projection: how many vertices land in front of the eye and how many behind, and how clip
/// `z` relates to clip `w`. `None` when the program cannot be interpreted at all.
///
/// Sampled with an even STRIDE through the mesh rather than as a prefix: vertex 0 of a
/// world mesh is routinely behind the camera, so a prefix answers a different question
/// than "does this draw cover anything".
pub(super) fn count_clip_w_signs(gxp: &GxpRecompile, max_samples: usize) -> Result<ClipStats, String> {
    let (mut in_front, mut behind, mut sampled) = (0usize, 0usize, 0usize);
    let mut bbox = [f32::MAX, f32::MIN, f32::MAX, f32::MIN];
    // Least-squares accumulators for `z = a*w + c` over the sampled vertices, in f64: the
    // fit's `c` is a small difference between two large dot products (on one retail racer,
    // `z` and `w` agree to four decimal places and `c` is about -1), and in f32 that
    // subtraction loses most of its significant digits.
    let (mut sw, mut sz, mut sww, mut swz) = (0f64, 0f64, 0f64, 0f64);
    let (mut wlo, mut whi) = (f32::MAX, f32::MIN);
    let mut first_vertex: Option<([f32; 4], [f32; 4])> = None;
    let pick = |nverts: usize| -> Result<Vec<usize>, String> {
        let step = (nverts / max_samples.max(1)).max(1);
        Ok((0..nverts).step_by(step).collect())
    };
    let mut visit = |v: usize, first_in: [f32; 4], regs: &vitaslop_gxp_shader::interp::RegFile, nan: Option<&vitaslop_gxp_shader::interp::NanSite>| {
        // Diagnostic (`VITASLOP_GXP_NAN_SITES=1`): the FIRST instruction of this pair's
        // vertex program that produced a non-finite value, on the reference interpreter
        // with this draw's own inputs - once per pair. A NaN varying paints a surface
        // black with nothing refused, and walking back to its origin by probing the GPU
        // costs a replay per register; this names the instruction in one.
        if let Some(site) = nan
            && crate::knobs::flag("VITASLOP_GXP_NAN_SITES")
        {
            use std::sync::{Mutex, OnceLock};
            static SEEN: OnceLock<Mutex<HashSet<u64>>> = OnceLock::new();
            let seen = SEEN.get_or_init(|| Mutex::new(HashSet::default()));
            let key = vitaslop_gxp_shader::Program::parse(&gxp.vprog).map_or(0, |p| p.hash);
            if seen.lock().unwrap_or_else(|e| e.into_inner()).insert(key) {
                report_warn!(
                    "gxp nan: vertex program {key:016x} vertex {v}: first non-finite at #{} {} -> {}.{} = {} from [{}]",
                    site.index,
                    site.op,
                    site.dest,
                    site.channel,
                    site.value,
                    site.sources.join(", ")
                );
            }
        }
        sampled += 1;
        // Count the vertices this draw would actually PUT ON SCREEN under each reading of
        // the sign, rather than the sign alone. A mesh that straddles the camera plane has
        // both signs in it, and a vote on the sign alone leaves it uncorrected - which
        // clips away exactly the half that is in view and keeps the half behind the eye.
        // Inside the frustum means `|x| <= |w|` and `|y| <= |w|` on the side being tested.
        let (x, y, w) = (regs.o[0], regs.o[1], regs.o[3]);
        if first_vertex.is_none() {
            first_vertex = Some((first_in, [x, y, regs.o[2], w]));
            // The INTERNAL registers at program end. A clip position is a matrix times a
            // point, and when the matrix checks out (dump the SA file) the point is the
            // only thing left - these are where a transformed position lives on its way to
            // the output bank, so this is the number that separates "wrong matrix" from
            // "wrong point" without reading the disassembly.
            if crate::knobs::flag("VITASLOP_GXP_CLIP_DUMP_SA") {
                let iregs: Vec<String> = regs
                    .i
                    .iter()
                    .take(12)
                    .enumerate()
                    .map(|(k, v)| format!("i{k}:{v:.4}"))
                    .collect();
                let rregs: Vec<String> = regs
                    .r
                    .iter()
                    .take(8)
                    .enumerate()
                    .map(|(k, v)| format!("r{k}:{v:.4}"))
                    .collect();
                report_warn!(
                    "gxp clip: first vertex END STATE {} | {}",
                    iregs.join(" "),
                    rregs.join(" ")
                );
            }
        }
        let inside = x.abs() <= w.abs() && y.abs() <= w.abs();
        if w > 0.0 && inside {
            in_front += 1;
        } else if w < 0.0 && inside {
            behind += 1;
        }
        // Where the draw lands once the sign is corrected: `x/|w|`, which is the same NDC
        // either reading resolves to for the vertices that are actually on screen.
        if w != 0.0 {
            let (nx, ny) = (x / w.abs(), y / w.abs());
            bbox = [bbox[0].min(nx), bbox[1].max(nx), bbox[2].min(ny), bbox[3].max(ny)];
        }
        let z = regs.o[2];
        if w.is_finite() && z.is_finite() {
            let (wd, zd) = (w as f64, z as f64);
            sw += wd;
            sz += zd;
            sww += wd * wd;
            swz += wd * zd;
            wlo = wlo.min(w);
            whi = whi.max(w);
        }
    };
    let mem = interpret_vertices(gxp, &pick, &mut visit)?;
    let n = sampled as f64;
    let denom = n * sww - sw * sw;
    // A mesh whose vertices all share one `w` (a 2D overlay, a billboard) pins no line
    // through them, and inventing one from a degenerate system would put a wild `c` into
    // the depth encoding. Requiring a real spread of `w` is what keeps such a draw from
    // being asked a question it cannot answer.
    let depth_fit = (sampled >= 2 && denom.abs() > 1e-9 && (whi - wlo).abs() > 1e-3).then(|| {
        let a = (n * swz - sw * sz) / denom;
        let c = (sz - a * sw) / n;
        (a as f32, c as f32)
    });
    Ok(ClipStats {
        in_front,
        behind,
        sampled,
        first_vertex,
        bbox,
        depth_fit,
        w_spread: if whi > wlo { whi - wlo } else { 0.0 },
        mem_reads: mem.reads,
        mem_misses: mem.misses,
        first_miss: mem.first_miss,
        first_hit: mem.first_hit,
        window_bases: mem.window_bases,
    })
}

/// What [`interpret_vertices`] saw of the draw's guest-memory windows - see the fields of the
/// same names on [`ClipStats`].
struct MemStats {
    reads: u32,
    misses: u32,
    first_miss: u32,
    first_hit: u32,
    window_bases: Vec<(u32, u32)>,
}

/// What interpreting a vertex BLOB needs that does not depend on the draw: its decoded
/// primary and secondary programs, its container literals and its memory-window layout.
struct VertexInterpPlan {
    primary: vitaslop_gxp_shader::Shader,
    secondary: Option<vitaslop_gxp_shader::Shader>,
    literals: Vec<(u32, u32)>,
    windows: Vec<vitaslop_gxp_shader::module::MemWindow>,
}

/// [`VertexInterpPlan`] for a blob, memoised. `draw_pixel_bounds` asks it for every
/// self-blending draw of every frame, and decoding the blob each time was most of that cost.
///
/// Keyed by the blob's CONTENT (hashed, and compared byte for byte on a hit), not its address:
/// every draw carries its own copy of the blob, so an address key missed on every draw and
/// re-decoded the program each time - MEASURED as 6 of the 7.5 ms a 128-draw pass spent here.
/// Cleared wholesale at a bound: the key is the value, so every entry rebuilds identically.
fn vertex_interp_plan(vprog: &std::sync::Arc<[u8]>) -> Result<std::sync::Arc<VertexInterpPlan>, String> {
    use std::sync::{Arc, Mutex, OnceLock};
    type Memo = HashMap<Arc<[u8]>, Result<Arc<VertexInterpPlan>, String>>;
    static MEMO: OnceLock<Mutex<Memo>> = OnceLock::new();
    const CAP: usize = 1024;
    let mut g = MEMO.get_or_init(|| Mutex::new(HashMap::default())).lock().unwrap_or_else(|e| e.into_inner());
    if let Some(r) = g.get(vprog) {
        return r.clone();
    }
    let built = vitaslop_gxp_shader::recompile_vertex(vprog)
        .map_err(|e| format!("the vertex program does not recompile: {e}"))
        .map(|vrc| {
            Arc::new(VertexInterpPlan {
                secondary: Some(vitaslop_gxp_shader::usse::decode_secondary_shader(&vrc.program)),
                literals: vrc.program.literals.clone(),
                windows: vitaslop_gxp_shader::mem_windows_for_vertex_blob(vprog),
                primary: vrc.shader,
            })
        });
    if g.len() >= CAP {
        g.clear();
    }
    g.insert(vprog.clone(), built.clone());
    built
}

/// `*dst = src.clone()` without allocating: every vertex of a draw starts from the same
/// register file, and five fresh 512-lane banks per vertex were a third of the interpretation.
/// The destructuring is EXHAUSTIVE on purpose - a field added to the register file fails to
/// compile here rather than leaking one vertex's value into the next.
fn reset_regs(dst: &mut vitaslop_gxp_shader::interp::RegFile, src: &vitaslop_gxp_shader::interp::RegFile) {
    let vitaslop_gxp_shader::interp::RegFile { r, pa, sa, o, i, p, idx, killed, facing, frag_depth, ramp, tile_pixel, stores } = src;
    dst.r.clone_from(r);
    dst.pa.clone_from(pa);
    dst.sa.clone_from(sa);
    dst.o.clone_from(o);
    dst.i.clone_from(i);
    dst.p = *p;
    dst.idx = *idx;
    dst.killed = *killed;
    dst.facing = *facing;
    dst.frag_depth = *frag_depth;
    dst.ramp = *ramp;
    dst.tile_pixel = *tile_pixel;
    dst.stores.clone_from(stores);
}

/// Run a draw's vertex program on the reference interpreter, once per vertex `pick` names
/// (it is handed the vertex count of the draw's stream), against exactly the inputs the
/// linked module is given: the default uniform buffer, the container literals, the driver's
/// pointer registers, the secondary program, the draw's memory windows and vertex textures.
/// `visit` sees each vertex's index, its first attribute as fed, the register file at program
/// end, and where a non-finite value first appeared.
fn interpret_vertices(
    gxp: &GxpRecompile,
    pick: &dyn Fn(usize) -> Result<Vec<usize>, String>,
    visit: &mut dyn FnMut(usize, [f32; 4], &vitaslop_gxp_shader::interp::RegFile, Option<&vitaslop_gxp_shader::interp::NanSite>),
) -> Result<MemStats, String> {
    let plan = vertex_interp_plan(&gxp.vprog)?;
    let VertexInterpPlan { primary, secondary, literals, windows } = &*plan;
    // >>> A PROGRAM THAT PACKS 16-BIT HALVES IS INTERPRETED NOW. This refused every such
    // program on the premise that the interpreter held one value per lane, so a
    // four-channel F16 or 16-bit INTEGER pack drifted its register layout from the shader's
    // - true when written, and it produced a confident fiction on a fighting title (every
    // skinned vertex "at w == 0"). The premise is DEAD: the interpreter's lane accessors
    // read and write packed halves and bytes (`read_reg_lane`/`write_reg_lane`), the corpus
    // differential holds 803 half-precision programs to the GPU with 0 divergences, and the
    // authored F16 conformance cases pin the addressing rule itself. What the refusal cost
    // was the reverse fiction: a football title's play-call panel is 25 skinned draws that
    // rasterise nothing, and the one instrument that says where a draw's vertices go was
    // blind to exactly those programs.
    let mut base = vitaslop_gxp_shader::interp::RegFile::with_lanes(512);
    for (k, c) in gxp.vert_sa.chunks_exact(4).enumerate() {
        if k < base.sa.len() {
            base.sa[k] = f32::from_le_bytes([c[0], c[1], c[2], c[3]]);
        }
    }
    // The container's LITERALS, which the module writes with `emit_secondary_attrs` BEFORE
    // it places the pointer registers. Order matters: a literal that names the same SA
    // register as a buffer pointer must lose to the pointer here exactly as it does there.
    for &(reg, value) in literals {
        if let Some(slot) = base.sa.get_mut(reg as usize) {
            *slot = f32::from_bits(value);
        }
    }
    // The DRIVER-placed pointer registers, exactly as the linked module initialises them
    // (`sa[base_sa] = gxp_mem[i].x`). Without them the program's address arithmetic runs
    // against a zero pointer and its loads read nothing at all - which is not the same
    // program the frame runs, and this is the reference that has to agree with it.
    for (w, (addr, _)) in windows.iter().zip(&gxp.mem_windows) {
        if let Some(slot) = base.sa.get_mut(w.base_sa as usize) {
            *slot = f32::from_bits(*addr);
        }
    }
    // The draw's own guest-memory windows, resolved by ADDRESS exactly as the emitted
    // `gxp_mem_word` helper resolves them - first window that contains the address wins,
    // and an address inside none reads ZERO.
    // >>> AND IT COUNTS THE READS THAT LAND IN NO WINDOW. An address inside none reads
    // ZERO, in the shader and here alike - and a transform read as zeroes collapses every
    // vertex onto the origin, which on screen cannot be told from a draw that was never
    // issued. The count is what separates them: a program whose every vertex comes out with
    // `w == 0` and whose memory reads all MISSED is reading through a pointer or an offset
    // this renderer got wrong; one that misses nothing has a different problem.
    let mem_reads = std::cell::Cell::new(0u32);
    let mem_misses = std::cell::Cell::new(0u32);
    // The FIRST address that landed in no window, and the first that landed in one. Two
    // numbers beside the window's own base say which way a wrong base is wrong, which a
    // miss COUNT on its own cannot: four hits and sixteen misses out of five four-word
    // loads is the same count whether the base is 64 bytes high or 64 bytes low.
    let first_miss = std::cell::Cell::new(u32::MAX);
    let first_hit = std::cell::Cell::new(u32::MAX);
    let read_mem = |addr: u32| -> u32 {
        mem_reads.set(mem_reads.get() + 1);
        for (w, (base_addr, bytes)) in windows.iter().zip(&gxp.mem_windows) {
            let Some(off) = addr.checked_sub(*base_addr) else { continue };
            if off >= w.bytes {
                continue;
            }
            let at = (off & !3) as usize;
            if let Some(b) = bytes.get(at..at + 4) {
                if first_hit.get() == u32::MAX {
                    first_hit.set(addr);
                }
                return u32::from_le_bytes([b[0], b[1], b[2], b[3]]);
            }
        }
        mem_misses.set(mem_misses.get() + 1);
        if first_miss.get() == u32::MAX {
            first_miss.set(addr);
        }
        0
    };
    // A program with memory loads and no window bytes is not interpretable: reading zeroes
    // it was never given would fabricate a vertex position, which is the one thing this
    // measurement may not do.
    if windows.len() != gxp.mem_windows.len() {
        return Err(format!(
            "the program declares {} guest-memory window(s) and the draw carries {}",
            windows.len(),
            gxp.mem_windows.len()
        ));
    }
    let mem: Option<vitaslop_gxp_shader::interp::MemFetch<'_>> =
        (!windows.is_empty()).then_some(&read_mem);
    // The SECONDARY program runs before the primary and overwrites SA registers the
    // primary reads, so skipping it interprets the primary against uniforms the real
    // module has already replaced. Its literals were written further up, ahead of the
    // pointer registers, in the module's own order.
    //
    // >>> IT RUNS AFTER THE POINTER REGISTERS AND WITH THE MEMORY READER, because the
    // emitted module does exactly that: `sa[base_sa] = gxp_mem[i].x` is placed BEFORE the
    // body, and the body begins with the secondary. A secondary that loads memory - the
    // natural place for it, since the prologue fetches once and leaves the result in the SA
    // file - therefore reads through a pointer that is already in place. Running it first,
    // with no reader, interpreted a DIFFERENT program from the one the frame runs, and it
    // did not fail quietly: every world program of a fighting title came back "its
    // SECONDARY program does not interpret: ldmem (no guest-memory window supplied to this
    // interpretation)", so the pass had no perspective evidence at all and took its
    // projection verdict from whatever 2D overlay quad did interpret.
    if let Some(secondary) = secondary.as_ref() {
        // A secondary program samples no textures - it runs before any varying exists.
        let no_tex = |_unit: u8, _coord: [f32; 4], _lod: vitaslop_gxp_shader::interp::TexLodArg| -> Option<[f32; 4]> { None };
        if let Err(e) = vitaslop_gxp_shader::interp::run_watching_for_nan_with_env(
            secondary,
            &mut base,
            &no_tex,
            mem,
        ) {
            return Err(format!("its SECONDARY program does not interpret: {e}"));
        }
        // Diagnostic (`VITASLOP_GXP_CLIP_DUMP_SA=1`): the SA file the primary is about to
        // read, AFTER the prologue has filled it from the bound windows.
        //
        // This is the one state a wrong clip position can hide in that nothing else prints.
        // The window dump says what the guest BOUND; this says what the program actually
        // has in its registers, and the difference between the two is the prologue - the
        // loads, their destinations and the repeats that shuffle them. A transform whose
        // z and w are right while x and y are not is a statement about four specific
        // registers, and this is what names them.
        if crate::knobs::flag("VITASLOP_GXP_CLIP_DUMP_SA") {
            let row: Vec<String> = base
                .sa
                .iter()
                .take(80)
                .enumerate()
                .map(|(i, v)| format!("{i}:{v:.4}"))
                .collect();
            report_warn!("gxp clip: SA after the secondary = [{}]", row.join(" "));
        }
    }
    let stride = gxp.vertex_stride.max(1) as usize;
    let nverts = gxp.vertices.len() / stride;
    if nverts == 0 {
        return Err(format!(
            "the draw carries no vertices ({} stream bytes at stride {stride})",
            gxp.vertices.len()
        ));
    }
    let verts = pick(nverts)?;
    // The VERTEX stage's own textures, so a program that builds its geometry from a fetch
    // can be interpreted at all. Nearest, wrapped, no mip: this is a measurement of where
    // the geometry LANDS, and a filter would change nothing about that while costing four
    // texel reads per sample.
    // Level 0 whatever the LOD operand says: the texture here is one decoded mip.
    let fetch = |unit: u8, coord: [f32; 4], _lod: vitaslop_gxp_shader::interp::TexLodArg| -> Option<[f32; 4]> {
        let t = &gxp.vertex_textures.iter().find(|t| t.unit == unit)?.tex;
        let (w, h) = (t.width.max(1), t.height.max(1));
        let wrap = |v: f32, n: u32| {
            let n = n as f32;
            let p = (v * n).floor().rem_euclid(n);
            p as u32
        };
        let (x, y) = (wrap(coord[0], w), wrap(coord[1], h));
        let bpt = t.texel.bytes_per_texel();
        let o = (y * w + x) as usize * bpt;
        let px = t.rgba.get(o..o + bpt)?;
        // Read through the seam the texture was decoded onto. Narrowing here would put
        // back exactly the precision loss the half seam exists to avoid, and on the one
        // path whose whole job is to say where a vertex-fetched geometry LANDS.
        Some(match t.texel {
            TexelSeam::Rgba8 => [
                px[0] as f32 / 255.0,
                px[1] as f32 / 255.0,
                px[2] as f32 / 255.0,
                px[3] as f32 / 255.0,
            ],
            TexelSeam::Rgba16Float => {
                let h = |i: usize| half_to_f32(u16::from_le_bytes([px[i * 2], px[i * 2 + 1]]));
                [h(0), h(1), h(2), h(3)]
            }
            TexelSeam::Rg32Float => {
                let f = |i: usize| f32::from_le_bytes([px[i], px[i + 1], px[i + 2], px[i + 3]]);
                [f(0), f(4), 0.0, 1.0]
            }
        })
    };
    // Which PA lanes an attribute actually supplies (see the default fill below).
    let mut claimed = vec![false; base.pa.len()];
    for a in gxp.attributes.iter() {
        for c in 0..a.components as usize {
            if let Some(slot) = claimed.get_mut(a.reg_index as usize + c) {
                *slot = true;
            }
        }
    }
    let mut regs = base.clone();
    for (n, v) in verts.into_iter().enumerate() {
        if n > 0 {
            reset_regs(&mut regs, &base);
        }
        for a in gxp.attributes.iter() {
            let vbase = v * stride + a.offset as usize;
            // All FOUR lanes of the attribute's register, because that is what the pipeline
            // feeds the real shader: the linked module reads `in.aN.xyzw`, and WebGPU fills
            // the components a `Float32x3` vertex format does not supply with (0,0,0,1). An
            // interpretation that zero-fills instead reads `w = 0` where the GPU reads 1,
            // which on a 3-component POSITION drops the projection's translation term - i.e.
            // it measures a different clip `w` than the frame it is supposed to explain.
            for c in 0..4usize {
                let lane = a.reg_index as usize + c;
                if lane >= regs.pa.len() {
                    continue;
                }
                if c < a.components as usize {
                    regs.pa[lane] = read_attr_component(&gxp.vertices, vbase, a.gxm_format, c);
                } else if !claimed[lane] {
                    // Only where no OTHER attribute owns the lane - two attributes packed
                    // two lanes apart would otherwise overwrite each other's components.
                    regs.pa[lane] = if c == 3 { 1.0 } else { 0.0 };
                }
            }
        }
        // BEFORE the run: the primary uses the PA bank as scratch, so reading the
        // attribute back afterwards reports whatever the shader last left there. That is
        // not a detail - it made this very diagnostic print a screen-space number for a
        // model-space mesh.
        let first_in: [f32; 4] = gxp.attributes.first().map_or([0.0; 4], |a| {
            let b = a.reg_index as usize;
            std::array::from_fn(|c| regs.pa.get(b + c).copied().unwrap_or(0.0))
        });
        let run = vitaslop_gxp_shader::interp::run_watching_for_nan_with_env(primary, &mut regs, &fetch, mem);
        match run {
            Err(e) => return Err(format!("{e}")),
            Ok(site) => visit(v, first_in, &regs, site.as_ref()),
        }
    }
    Ok(MemStats {
        reads: mem_reads.get(),
        misses: mem_misses.get(),
        first_miss: first_miss.get(),
        first_hit: first_hit.get(),
        window_bases: windows
            .iter()
            .zip(&gxp.mem_windows)
            .map(|(w, (addr, _))| (*addr, w.bytes))
            .collect(),
    })
}

/// >>> WHERE A DRAW CAN PUT PIXELS, from its own vertices, for the destination-copy
/// >>> coalescing in `encode_pass` - in ATTACHMENT texels, `[x0, y0, x1, y1)`.
///
/// A self-blending draw on a 64-bit target costs a render-pass split and an attachment copy,
/// unless it touches nothing written since the last copy (`DestDirty`). That test was asked of
/// the draw's VIEWPORT, which for a world effect is the whole target, so every reader met
/// every earlier draw and every reader cut the pass. MEASURED on an action title's courtyard:
/// 128 splits over 128 draws in one pass, each a store and reload of a 720x408 64-bit
/// attachment and its depth, 28-36 ms of a phone frame (it surfaced as the `depth-convert`
/// after it, the first timed pass to wait for those untimed segments).
///
/// The draw's vertex program is run on the reference interpreter over EVERY vertex its index
/// list names, with exactly the clip fixup the linked module applies (`sign` and `y_sense`, the
/// two per-draw lanes of `gxp_clipfix`); each triangle is clipped to the clip volume's side
/// planes as the rasteriser clips it (a ground decal reaches behind the eye, and its corners
/// there project nowhere useful), and the box of the clipped, projected corners is widened by
/// a texel each side - the interpreter and the GPU may differ in the last bit of a position,
/// never by a texel. A rasterised triangle lies inside the hull of its clipped corners, so the
/// box bounds every pixel the draw can cover. Everything this cannot bound exactly answers
/// `None` (the viewport stands, as before): a non-finite position, a line or
/// point topology (a width or a point size reaches past the corners), a vertex stage that
/// SAMPLES (the interpreter fetches nearest, the GPU may filter), a memory read that landed in
/// no window, a disabled viewport, an index past the stream, or more than `max_verts`
/// distinct vertices (the cost bound).
pub(super) fn draw_pixel_bounds(
    gxp: &GxpRecompile,
    sign: f32,
    yflip: bool,
    viewport: (f32, f32, f32, f32),
    max_verts: usize,
) -> Option<[f32; 4]> {
    if !gxm_topology_has_faces(gxp.primitive)
        || gxm_topology(gxp.primitive) != wgpu::PrimitiveTopology::TriangleList
        || !gxp.vertex_textures.is_empty()
        || gxp.viewport_enable == crate::gpu::SCE_GXM_VIEWPORT_DISABLED
        || !(sign == 1.0 || sign == -1.0 || sign == 2.0)
    {
        return None;
    }
    // The viewport's vertical sense, exactly as `prepare` hands it to the module.
    let y_sense: f32 = if gxp.viewport[3] > 0.0 { -1.0 } else { 1.0 };
    let pick = |nverts: usize| -> Result<Vec<usize>, String> {
        let n = gxp.index_count as usize;
        let mut seen: Vec<usize> = Vec::with_capacity(n.min(max_verts + 1));
        let mut mark = vec![false; nverts];
        for k in 0..n {
            let i = if gxp.index_u32 {
                let b = gxp.indices.get(4 * k..4 * k + 4).ok_or("short index list")?;
                u32::from_le_bytes([b[0], b[1], b[2], b[3]]) as usize
            } else {
                let b = gxp.indices.get(2 * k..2 * k + 2).ok_or("short index list")?;
                u16::from_le_bytes([b[0], b[1]]) as usize
            };
            let slot = mark.get_mut(i).ok_or("an index past the vertex stream")?;
            if !*slot {
                *slot = true;
                seen.push(i);
                if seen.len() > max_verts {
                    return Err("too many vertices".into());
                }
            }
        }
        Ok(seen)
    };
    // Each vertex's clip position after the fixup, `(x, y, w)`, by its index in the stream.
    let mut clip: HashMap<usize, [f64; 3]> = HashMap::default();
    let mut finite = true;
    let mut visit = |v: usize, _in: [f32; 4], regs: &vitaslop_gxp_shader::interp::RegFile, _nan: Option<&vitaslop_gxp_shader::interp::NanSite>| {
        // `gxp_clipfix`, x/y/w only: `-1` negates all four lanes, `2` negates `w` alone.
        let (mut x, mut y, mut w) = (regs.o[0], regs.o[1], regs.o[3]);
        if sign < 0.0 {
            (x, y, w) = (-x, -y, -w);
        } else if sign > 1.5 {
            w = -w;
        }
        y *= y_sense;
        if yflip {
            y = -y;
        }
        if !x.is_finite() || !y.is_finite() || !w.is_finite() {
            finite = false;
        }
        clip.insert(v, [x as f64, y as f64, w as f64]);
    };
    let mem = interpret_vertices(gxp, &pick, &mut visit).ok()?;
    if !finite || mem.misses > 0 {
        return None;
    }
    let index = |k: usize| -> usize {
        if gxp.index_u32 {
            u32::from_le_bytes([gxp.indices[4 * k], gxp.indices[4 * k + 1], gxp.indices[4 * k + 2], gxp.indices[4 * k + 3]]) as usize
        } else {
            u16::from_le_bytes([gxp.indices[2 * k], gxp.indices[2 * k + 1]]) as usize
        }
    };
    let n = gxp.index_count as usize;
    if !n.is_multiple_of(3) {
        return None;
    }
    let mut tris: Vec<[[f64; 3]; 3]> = Vec::with_capacity(n / 3);
    for t in 0..n / 3 {
        let corner = |k: usize| clip.get(&index(3 * t + k)).copied();
        tris.push([corner(0)?, corner(1)?, corner(2)?]);
    }
    Some(clipped_pixel_box(&tris, viewport))
}

/// The pixel box `[x0, y0, x1, y1]` (fractional, widened by a texel each side) that every
/// triangle of `tris` - clip positions `(x, y, w)` - can cover through `viewport`, or all
/// zeroes when none survives the clip. See [`draw_pixel_bounds`].
///
/// Every triangle is clipped to the four SIDE planes of the clip volume (`-w <= x, y <= w`),
/// exactly as the rasteriser clips it, then projected: what is left lies inside the hull of the
/// clipped corners, all of which have `w >= |x|, |y|`. A corner the clipping leaves AT `w = 0`
/// (the eye point itself) has no projection; it is taken as reaching the whole viewport. The
/// depth planes are not applied - they can only remove more.
pub(super) fn clipped_pixel_box(tris: &[[[f64; 3]; 3]], viewport: (f32, f32, f32, f32)) -> [f32; 4] {
    let (mut lo, mut hi) = ([f64::MAX; 2], [f64::MIN; 2]);
    let mut poly: Vec<[f64; 3]> = Vec::with_capacity(9);
    let mut next: Vec<[f64; 3]> = Vec::with_capacity(9);
    for tri in tris {
        poly.clear();
        poly.extend_from_slice(tri);
        // Plane value >= 0 inside: w - x, w + x, w - y, w + y.
        for plane in 0..4 {
            let d = |p: &[f64; 3]| match plane {
                0 => p[2] - p[0],
                1 => p[2] + p[0],
                2 => p[2] - p[1],
                _ => p[2] + p[1],
            };
            next.clear();
            for i in 0..poly.len() {
                let (a, b) = (poly[i], poly[(i + 1) % poly.len()]);
                let (da, db) = (d(&a), d(&b));
                if da >= 0.0 {
                    next.push(a);
                }
                if (da >= 0.0) != (db >= 0.0) {
                    let s = da / (da - db);
                    next.push([a[0] + (b[0] - a[0]) * s, a[1] + (b[1] - a[1]) * s, a[2] + (b[2] - a[2]) * s]);
                }
            }
            std::mem::swap(&mut poly, &mut next);
            if poly.is_empty() {
                break;
            }
        }
        for p in &poly {
            let (nx, ny) = if p[2] > 1e-9 { ((p[0] / p[2]).clamp(-1.0, 1.0), (p[1] / p[2]).clamp(-1.0, 1.0)) } else {
                lo = [-1.0, -1.0];
                hi = [1.0, 1.0];
                continue;
            };
            lo = [lo[0].min(nx), lo[1].min(ny)];
            hi = [hi[0].max(nx), hi[1].max(ny)];
        }
    }
    if lo[0] > hi[0] {
        // Nothing survives the clip: the draw rasterises nothing.
        return [0.0; 4];
    }
    // WebGPU's viewport transform: ndc +1 is the right edge and the TOP of the rectangle.
    let (vx, vy, vw, vh) = (viewport.0 as f64, viewport.1 as f64, viewport.2 as f64, viewport.3 as f64);
    let px = |n: f64| vx + (n + 1.0) * 0.5 * vw;
    let py = |n: f64| vy + (1.0 - n) * 0.5 * vh;
    [(px(lo[0]) - 1.0) as f32, (py(hi[1]) - 1.0) as f32, (px(hi[0]) + 1.0) as f32, (py(lo[1]) + 1.0) as f32]
}

#[cfg(test)]
mod pixel_box_tests {
    use super::clipped_pixel_box;

    const VP: (f32, f32, f32, f32) = (0.0, 0.0, 100.0, 50.0);

    #[test]
    fn a_triangle_in_front_bounds_its_projected_corners_widened_by_a_texel() {
        // ndc x in [-0.5, 0.5], y in [0, 0.5] at w = 2.
        let b = clipped_pixel_box(&[[[-1.0, 0.0, 2.0], [1.0, 0.0, 2.0], [0.0, 1.0, 2.0]]], VP);
        assert_eq!(b, [24.0, 11.5, 76.0, 26.0]);
    }

    #[test]
    fn a_triangle_reaching_behind_the_eye_is_clipped_not_refused() {
        // One corner behind the eye (w < 0): clipped at the side planes, the box is the part in
        // view, and never past the viewport.
        let b = clipped_pixel_box(&[[[0.0, 0.0, 1.0], [0.2, 0.0, 1.0], [5.0, 0.0, -1.0]]], VP);
        assert!(b[0] >= 49.0 - 1e-3 && b[2] <= 101.0 + 1e-3, "{b:?}");
        assert!(b[1] >= 24.0 - 1e-3 && b[3] <= 26.0 + 1e-3, "{b:?}");
    }

    #[test]
    fn a_triangle_wholly_outside_covers_nothing() {
        let b = clipped_pixel_box(&[[[3.0, 0.0, 1.0], [4.0, 0.0, 1.0], [3.5, 0.5, 1.0]]], VP);
        assert_eq!(b, [0.0; 4]);
        // Wholly behind the eye too.
        let b = clipped_pixel_box(&[[[0.0, 0.0, -1.0], [0.1, 0.0, -1.0], [0.0, 0.1, -1.0]]], VP);
        assert_eq!(b, [0.0; 4]);
    }

    #[test]
    fn a_triangle_through_the_eye_point_reaches_the_whole_viewport() {
        let b = clipped_pixel_box(&[[[0.0, 0.0, 0.0], [1.0, 0.0, 1.0], [0.0, 1.0, 1.0]]], VP);
        assert_eq!(b, [-1.0, -1.0, 101.0, 51.0]);
    }
}

/// Weigh, ONCE per shader pair, how many of that pair's vertices land in the frustum with
/// each sign of clip `w`. Reports what it saw unconditionally: a correction this large
/// that applied itself silently would be indistinguishable from a faithful projection by
/// looking at the frame.
///
/// `None` = the pair's vertex program does not interpret at all, so it carries no evidence
/// and never will.
pub(super) fn measure_clip(gxp: &GxpRecompile, key: u64) -> Option<ClipStats> {
    match count_clip_w_signs(gxp, 256) {
        Ok(s) => {
            report!(
                "gxp clip: key {key:x}: of {} sampled vertices, {} land in the frustum with w>0 \
                 and {} with w<0; ndc x[{:.2},{:.2}] y[{:.2},{:.2}]; z = {}",
                s.sampled,
                s.in_front,
                s.behind,
                s.bbox[0],
                s.bbox[1],
                s.bbox[2],
                s.bbox[3],
                match s.depth_fit {
                    Some((a, c)) => format!("{a}*w + {c} over w spread {}", s.w_spread),
                    None => "(no w spread to fit through)".into(),
                }
            );
            // >>> A PAIR WHOSE EVERY SAMPLED VERTEX LANDS NOWHERE IS A FINDING, AND IT
            // >>> HAS TO BE VISIBLE AT `warn`.
            //
            // The measurement above is at `debug`, which is right for a pair that measures
            // sanely and wrong for one that does not: a run at the default level then sees
            // the REFUSALS and not the measurements, so "0 of 435 vertices land in the
            // frustum" - the whole explanation of an empty stage - is invisible in exactly
            // the run that needs it.
            // [[vitaslop-a-diagnostic-at-debug-is-a-diagnostic-that-does-not-exist]]
            //
            // A 2D overlay quad legitimately measures nothing (its `w` is constant, so no
            // vertex is "in the frustum" in this sense), so the warning is limited to a
            // pair with a real mesh: more than a handful of vertices, and not one of them
            // inside under either sign of `w`.
            // ...and NOT for a mesh that merely projects OUTSIDE the frustum. A stadium is
            // drawn in sections all the way round the camera, and the sections behind
            // and beside it are valid positions the hardware culls - MEASURED on a
            // baseball title as 206 draws of one pair "covering nothing" with ndc boxes
            // at x -113..-7. A position is degenerate when there is no perspective in it
            // (no `w` spread) or it is not finite; otherwise it is a position, off-screen.
            let degenerate = s.w_spread == 0.0 || !s.bbox.iter().all(|v| v.is_finite());
            // Once per KEY: this measures per pipeline variant and per re-measure, and a
            // culled section is the same fact every time.
            static OUTSIDE_SEEN: std::sync::Mutex<Option<HashSet<u64>>> = std::sync::Mutex::new(None);
            let first_outside = || {
                let mut g = OUTSIDE_SEEN.lock().unwrap_or_else(|e| e.into_inner());
                g.get_or_insert_with(HashSet::default).insert(key)
            };
            // >>> AND THE SAME DEDUPE FOR THE **DEGENERATE** BRANCH BELOW, WHICH HAD NONE.
            //
            // That one is a `report_warn!` carrying a dozen formatted fields and a THIRD
            // `Program::parse` of the blob (for the corpus hash), and it fired on every
            // re-measure - thirteen times a FRAME on a baseball title's composite pass, for
            // the whole run. A per-frame warning is not a warning, it is a cost with a
            // message attached [[vitaslop-a-diagnostic-can-bury-the-findings]], and this one
            // was buried in exactly the log its own subject would have to be found in.
            static DEGENERATE_SEEN: std::sync::Mutex<Option<HashSet<u64>>> = std::sync::Mutex::new(None);
            let first_degenerate = || {
                let mut g = DEGENERATE_SEEN.lock().unwrap_or_else(|e| e.into_inner());
                g.get_or_insert_with(HashSet::default).insert(key)
            };
            if s.sampled >= 16 && s.in_front == 0 && s.behind == 0 && !degenerate && first_outside() {
                report_status!(
                    "gxp clip: key {key:x}: all {} sampled vertices project OUTSIDE the frustum (ndc x[{:.2},{:.2}] y[{:.2},{:.2}], w spread {}) - a valid position, off-screen at this sample, which the hardware culls too",
                    s.sampled,
                    s.bbox[0],
                    s.bbox[1],
                    s.bbox[2],
                    s.bbox[3],
                    s.w_spread
                );
            } else if s.sampled >= 16 && s.in_front == 0 && s.behind == 0 && first_degenerate() {
                report_warn!(
                    "gxp clip: key {key:x}: NONE of its {} sampled vertices lands in the frustum under either sign of w, and the ndc box is degenerate (x[{:.2},{:.2}] y[{:.2},{:.2}]) - the clip position this program computes is not a position, so the draw covers no pixel however it is shaded. Its guest-memory reads: {} of {} landed in NO bound window and read ZERO. Its vertex DEFAULT UNIFORM BUFFER is {} bytes and its first 16 registers - where a local-to-world matrix lives - are {}. Its VERTEX BLOB's content hash is {:016x}, which is the name the offline corpus files it under.{}",
                    s.sampled,
                    s.bbox[0],
                    s.bbox[1],
                    s.bbox[2],
                    s.bbox[3],
                    s.mem_misses,
                    s.mem_reads,
                    gxp.vert_sa.len(),
                    // The uniform buffer is the OTHER place a vertex position can come
                    // from, and the two are not interchangeable: a program that transforms
                    // by a matrix in the SA file and one that chases a pointer read their
                    // world transform from different bytes. All-zero here is a complete
                    // explanation of a clip position of zero, and non-zero rules it out.
                    if gxp.vert_sa.iter().take(64).all(|b| *b == 0) {
                        "ALL ZERO"
                    } else {
                        "not zero"
                    },
                    // The hash is what turns a pair key from a live frame into the `.gxp`
                    // an offline test can open, which is the difference between a
                    // thirteen-minute run per question and a one-second one.
                    vitaslop_gxp_shader::Program::parse(&gxp.vprog)
                        .map(|p| p.hash)
                        .unwrap_or(0),
                    // ONE vertex, in and out. The box says the draw covers nothing; this
                    // says whether the position it was handed is wrong or the transform it
                    // applied is, which are different investigations.
                    match s.first_vertex {
                        Some((p, c)) => format!(
                            " FIRST SAMPLED VERTEX: attribute0 [{:.4}, {:.4}, {:.4}, {:.4}] -> clip [{:.4}, {:.4}, {:.4}, {:.4}] (ndc {:.3}, {:.3})",
                            p[0], p[1], p[2], p[3], c[0], c[1], c[2], c[3],
                            c[0] / c[3], c[1] / c[3]
                        ),
                        None => String::new(),
                    }
                );
                if s.mem_misses > 0 {
                    report_warn!(
                        "gxp clip: key {key:x}: its guest-memory windows are at {:x?} (base, bytes); the FIRST address that missed is {:#x} and the first that hit is {:#x}. The difference between a miss and the window's own base is the offset this renderer has wrong.",
                        s.window_bases,
                        s.first_miss,
                        s.first_hit
                    );
                }
            } else if s.mem_misses > 0 {
                // A pair that DOES land somewhere but still reads outside its windows is
                // a different, quieter defect: part of its data is zeroes it was never
                // given.
                report_warn!(
                    "gxp clip: key {key:x}: {} of its {} guest-memory reads landed in NO bound window and read ZERO, so part of what this program computes is zeroes this renderer supplied rather than the guest's data. Its windows are at {:x?} (base, bytes); the FIRST address that missed is {:#x} and the first that HIT is {:#x} - the difference between a miss and a window's own base is the offset this renderer has wrong. Its VERTEX BLOB's content hash is {:016x}.",
                    s.mem_misses,
                    s.mem_reads,
                    s.window_bases,
                    s.first_miss,
                    s.first_hit,
                    vitaslop_gxp_shader::Program::parse(&gxp.vprog)
                        .map(|p| p.hash)
                        .unwrap_or(0)
                );
            }
            Some(s)
        }
        // The REASON, not just the fact. A pass whose every perspective draw is
        // unmeasurable takes its projection verdict from whatever 2D overlay quad does
        // interpret, and on a golf title that is the whole difference between a world and a
        // black frame - so "could not be measured" on its own is the shape of diagnostic
        // that leaves a title dark for a session.
        Err(why) => {
            report_warn!(
                "gxp clip: key {key:x}: NOT MEASURED - {why}. It contributes no evidence \
                 about this pass's projection, so the pass is decided by its other draws"
            );
            None
        }
    }
}

/// Inject the GXM->WebGPU clip fixup into a linked module: wrap the vertex stage's
/// `out.position` assignment in a helper that remaps clip Z (and optionally flips Y).
///
/// The linker emits exactly one `  out.position = <expr>;` statement, but the exact shape of
/// `<expr>` is the emitter's business and has changed. So this matches the STATEMENT and
/// wraps whatever it assigns. Returns `None` when the statement is not found, which makes
/// the pair fall back rather than render with no depth remap at all: this transform is not
/// cosmetic, it is what puts the guest's clip depth inside WebGPU's `0 <= z <= w` clip
/// volume, and without it the hardware clips away every triangle whose raw clip z runs past
/// w (this title's does, by roughly 5x) - which looks like a mesh mysteriously missing its
/// far half, not like a broken depth buffer.
pub(super) fn inject_clip_fixup(wgsl: &str, zfix: ZFix, yflip: bool, solid: bool, keycolor: Option<u64>, ret: bool) -> Option<String> {
    // Replace the guest's clip z with the SAME depth the fixed-function path writes, so
    // recompiled and fixed-function draws share one comparable depth buffer: the projected
    // view distance through `-1/w`, mapped linearly onto [0,1] over the scene's visible
    // range (see `render::project` for why the guest's own clip z is not a depth here).
    // Keeping xy exact leaves the real shader's projection untouched.
    //
    // A remap that DIVIDES by `c.w` has to guard on `c.w > 0.0` and leaves behind-the-eye
    // vertices on a different scale from their own triangle's other corners; the default
    // arm no longer divides, for the reason `ZFix::Viewport` records. The arms below that
    // still do are diagnostics, and each says so.
    let z = match zfix {
        ZFix::Range => {
            "  if (c.w > 0.0) { let q = -1.0 / c.w;\n    r.z = clamp((q - gxp_depth.range.x) * gxp_depth.range.y, 0.0, 1.0) * c.w; }\n"
        }
        // The ORDINARY GL->WebGPU depth remap: the guest's own clip z, which GXM reads in
        // [-w, w], mapped into WebGPU's [0, w]. It needs no scene statistics at all, so it
        // cannot be thrown off by a depth range measured through a different projection
        // than the one the recompiled shader actually uses.
        ZFix::Gl => "  r.z = (c.z + c.w) * 0.5;\n",
        // The guest's own window depth, CLAMPED rather than clipped (see `ZFix::Clamp`).
        // `c.w <= 0` is behind the eye and left to wgpu's own clip, exactly as `Range` does -
        // which is the defect `ZFix::Viewport` records, kept here because this arm exists to
        // reproduce the OLD behaviour. It is a diagnostic; do not adopt it as a default.
        ZFix::Clamp => "  if (c.w > 0.0) { r.z = clamp(c.z / c.w, 0.0, 1.0) * c.w; }\n",
        // The guest's OWN viewport depth mapping (`fit.z` = zScale, `fit.w` = zOffset),
        // applied PROJECTIVELY. See `ZFix::Viewport` for why it is not a divide.
        ZFix::Viewport => "  r.z = c.z * gxp_depth.fit.z + c.w * gxp_depth.fit.w;\n",
        ZFix::Off => "",
    };
    // >>> WITHOUT `depth-clip-control` THE CLAMP HAS TO BE HERE, OR THE PRIMITIVE IS CLIPPED.
    //
    // PowerVR CLAMPS window depth; it does not clip on it. The default arm leaves that clamp
    // to the rasteriser (`unclipped_depth`), which only a device offering
    // `depth-clip-control` can grant - and a PowerVR phone does not offer it. There every
    // vertex whose depth fell outside [0, w] was CLIPPED: a baseball title's boot splash quads sit at clip
    // z = -1 with no viewport set, so all 1,400 frames of logos and the health warning were
    // clipped away and the only thing left on screen was the title's black fade quad
    // (measured on the device: `VITASLOP_GXP_KEYCOLOR` + `VITASLOP_GXP_NODEPTH` with the fade
    // pair excluded rasterised NOTHING; the fixed-function path, which clamps, showed the
    // warning screen).
    //
    // Projective, `w > 0` only: `clamp(z, 0, w)` is the per-vertex window-depth clamp, and a
    // vertex behind the eye is left alone for the reason `ZFix::Viewport` records.
    let z_clamp = if matches!(zfix, ZFix::Viewport) && !crate::gpu::unclipped_depth_available() {
        "  if (c.w > 0.0) { r.z = clamp(r.z, 0.0, c.w); }\n"
    } else {
        ""
    };
    let z = format!("{z}{z_clamp}");
    // The guest viewport's VERTICAL SENSE, applied per DRAW (`gxp_depth.vp.x`, +1 or -1).
    //
    // GXM maps ndc y to the framebuffer as `screen = yOffset + yScale * ndc`, so a pass
    // that sets `yScale > 0` puts ndc `+1` at the BOTTOM of its rectangle. wgpu's
    // `set_viewport` requires a positive height and always puts ndc `+1` at the TOP, so
    // there is nowhere else this can be expressed - `gxm_viewport_rect` takes `|yScale|`
    // and used to REPORT the flip as uncorrectable.
    //
    // >>> AND A TITLE DOES IT, on the one pass where a mirrored image is invisible.
    // MEASURED on the golf title: its 1536x1536 SHADOW MAP pass sets
    // `viewport [744, 712, 744, 712, 0.5, 0.5]` - `yScale = +712`. The map came out
    // vertically mirrored, which no frame shows, and every later lookup into it read the
    // wrong row: the character's own shadow reference depth is 0.941 while the row it
    // sampled holds 0.2-0.5, so its shadow term was 0 over every pixel and the whole
    // character rendered as a BLACK SILHOUETTE for five sessions. The terrain was
    // unaffected because its reference depth is past the map's range and never consults
    // it. A pass whose output is only ever SAMPLED is exactly where a mirror hides.
    //
    // Multiplying by +1.0 is exact in IEEE, so a title whose viewports all have the
    // ordinary negative `yScale` is bit-identical to before this existed.
    let y = if yflip { "  r.y = -c.y * gxp_depth.vp.x;\n" } else { "  r.y = c.y * gxp_depth.vp.x;\n" };
    // >>> A DISABLED VIEWPORT MEANS THE POSITION IS ALREADY IN WINDOW PIXELS (`gxp_depth.vp.y`).
    //
    // `sceGxmSetViewportEnable(SCE_GXM_VIEWPORT_DISABLED)` turns off the clip-to-window
    // transform, and a title then writes window coordinates straight from its vertex
    // program: x right and y DOWN in pixels of the target, divided by w like any clip
    // position. A 2D overlay is the natural user - one title's dialog text emits
    // `pos.xy = pixel + offset` with no scale at all. Read as clip space, every such quad
    // landed hundreds of ndc units off screen and the dialog showed its box and no words.
    // So the pixels are mapped onto clip space here, projectively (`x*2/W - w`,
    // `w - y*2/H`), and the pass's own vertical flip still applies after.
    let y = format!(
        "{y}  if (gxp_depth.vp.y > 0.5) {{ r.x = c.x * gxp_depth.vp.z - c.w; r.y = {}(c.w - c.y * gxp_depth.vp.w); }}\n",
        if yflip { "-" } else { "" }
    );
    // The clip-w SIGN correction (`gxp_depth.range.z`, +1 or -1; see `NegW`). A title whose
    // projection puts clip `w` NEGATIVE in front of the camera names every visible point on
    // the half of the homogeneous line WebGPU clips away. Negating all four components names
    // the SAME point with `w > 0` (homogeneous coordinates are scale-invariant) and moves
    // behind-camera geometry to `w < 0`, where it belongs. It is one multiply by a UNIFORM,
    // which is what keeps the decision PER DRAW rather than per vertex - a per-vertex
    // `if (w < 0) { c = -c; }` would pull the geometry behind the eye into view.
    //
    // `gxp_depth.range.z` selects the correction: 1 = none, -1 = negate the whole vector,
    // 2 = flip the sign of `w` alone. The two are DIFFERENT renders, not two spellings of
    // one: negating all four components leaves `x/w` exactly as it was (that is what makes
    // it a rename of the same point), so it fixes the clip and nothing else, while flipping
    // `w` alone MIRRORS the draw in both axes. Which one a title needs is a measurement.
    //
    // The `GxpDepth` block itself is declared by the LINKER (`link::GXP_DEPTH_DECL`), not
    // here: the fragment stage reads the same block to reconstruct the guest's window
    // position, and a linked module has to be independently compilable.
    let helper = format!(
        "fn gxp_clipfix(cin: vec4<f32>) -> vec4<f32> {{\n\
         \x20 var c = cin;\n\
         \x20 if (gxp_depth.range.z < 0.0) {{ c = -cin; }}\n\
         \x20 else if (gxp_depth.range.z > 1.5) {{ c = vec4<f32>(cin.x, cin.y, cin.z, -cin.w); }}\n\
         \x20 var r = c;\n{z}{y}  return r;\n}}\n"
    );
    const ASSIGN: &str = "\n  out.position = ";
    let at = wgsl.find(ASSIGN)?;
    let rhs_start = at + ASSIGN.len();
    let rhs_end = rhs_start + wgsl[rhs_start..].find(";")?;
    let mut patched = String::with_capacity(wgsl.len() + 64);
    patched.push_str(&wgsl[..rhs_start]);
    patched.push_str("gxp_clipfix(");
    patched.push_str(&wgsl[rhs_start..rhs_end]);
    patched.push(')');
    patched.push_str(&wgsl[rhs_end..]);
    // Diagnostic (`VITASLOP_GXP_KEYCOLOR`): shade every pair a flat colour derived from its
    // own key, keeping its real geometry, depth and blend. One run then answers "WHICH pair
    // owns this region of the screen" for every region at once - the question that otherwise
    // costs one `VITASLOP_GXP_KEYS` run per candidate, and the one that has to be answered
    // before any question about what a surface's shader computes.
    // `VITASLOP_GXP_RETURN=<wgsl vec4 expression>`: the fragment's final return replaced by
    // any expression over its own locals (`in.v0`, `textureSample(t0, s0, in.v1.xy)`, ...) -
    // `SOLID` generalised, for asking a DEVICE what one term of one pipeline evaluates to. A
    // pair whose module cannot compile the expression falls back, which is the point: only the
    // pairs that have those names are asked.
    if let Some(eps) = texcoord_nudge() {
        patched = nudge_texcoords(&patched, eps);
    }
    // `VITASLOP_GXP_RETURN_KEYS=<keys | !keys>` restricts it to those pairs (`ret`).
    if let Some(expr) = crate::knobs::var("VITASLOP_GXP_RETURN").ok().filter(|e| ret && !e.trim().is_empty())
        && let Some(at) = patched.rfind("
  return ") {
            let end = patched[at + 1..].find(";
").map(|e| at + 1 + e + 1).unwrap_or(patched.len());
            patched.replace_range(at + 1..end, &format!("  return {};", expr.trim()));
        }
    if let Some(k) = keycolor {
        let chan = |shift: u32| ((k >> shift) & 0xff) as f32 / 255.0;
        let (r, g, b) = (0.25 + 0.75 * chan(0), 0.25 + 0.75 * chan(21), 0.25 + 0.75 * chan(42));
        match patched.rfind("\n  return ") {
            Some(at) => {
                let end = patched[at + 1..].find(";\n").map(|e| at + 1 + e + 1).unwrap_or(patched.len());
                // Keep the pair's REAL alpha. Forcing it to 1.0 does not merely recolour the
                // frame, it changes WHICH pair owns a pixel: a blended draw that contributes
                // a few percent becomes an opaque cover, and one late fullscreen quad then
                // hides every pair behind it. That is the exact question this instrument
                // exists to answer, so the alpha is carried through rather than replaced.
                let expr = patched[at + 1 + "  return ".len()..end - 1].trim().to_string();
                patched.replace_range(
                    at + 1..end,
                    &format!("  return vec4<f32>({r:.3}, {g:.3}, {b:.3}, ({expr}).w);"),
                );
                // Print the assignment: reading the colour back OFF the frame means undoing
                // whatever transfer function the target applied, and a near-match to the
                // wrong key is not distinguishable from a match to the right one.
                //
                // STATUS, not `report!`. At `debug` this line does not exist at the default
                // filter, and the browser - where the regions that need naming are - runs at
                // `warn`: a key-coloured browser frame came back with 131 pairs drawn and
                // ZERO colour lines to match them against. It is emitted only when the knob
                // is set and only once per pair, so it cannot bury anything
                // [[vitaslop-a-diagnostic-at-debug-is-a-diagnostic-that-does-not-exist]].
                report_status!("gxp keycolor: key {k:x} -> linear rgb({r:.3}, {g:.3}, {b:.3})");
            }
            // At WARN, for the same reason as the `solid` arm below: an instrument that
            // reports its own absence below the default filter reports nothing at all.
            None => report_warn!(
                "gxp build: VITASLOP_GXP_KEYCOLOR found no fragment return to replace for key \
                 {k:x} - the module shape changed; this pair is NOT key-coloured"
            ),
        }
    }
    if solid {
        // Diagnostic: force the fragment to solid magenta so any on-screen triangle is
        // visible regardless of shading. The colour expression depends on which register
        // holds the colour and at what precision, so this matches the fragment entry's
        // LAST `return` statement structurally rather than by its exact text - an
        // enumerate-the-spellings version silently stopped substituting anything the first
        // time the emitter's return line changed, which made the diagnostic lie.
        match patched.rfind("\n  return ") {
            Some(at) => {
                let end = patched[at + 1..]
                    .find(";\n")
                    .map(|e| at + 1 + e + 1)
                    .unwrap_or(patched.len());
                patched.replace_range(at + 1..end, "  return vec4<f32>(1.0, 0.0, 1.0, 1.0);");
            }
            // At WARN. This is the instrument reporting that it did NOTHING, and it was
            // at `debug` while every repro command says `VITASLOP_LOG=warn` - so a
            // `VITASLOP_GXP_SOLID` run came back BIT-IDENTICAL to the unpatched one
            // (0 of 522240 pixels), the substitution silently absent, and the frame read
            // as "this geometry does not rasterize". It rasterizes fine. The comment
            // above about an enumerate-the-spellings version that "made the diagnostic
            // lie" describes this failure exactly, and structural matching did not cure
            // it - only saying so out loud does.
            None => report_warn!(
                "gxp build: VITASLOP_GXP_SOLID found no fragment return to replace - \
                 the module shape changed; solid-fill is NOT in effect"
            ),
        }
    }
    // A module's `enable` directives (a dual-source module carries one) must precede
    // every global declaration, the helper included - so they are hoisted over it.
    let mut directives = String::new();
    let mut body = patched.as_str();
    while body.starts_with("enable ") {
        let end = body.find('\n').map_or(body.len(), |i| i + 1);
        directives.push_str(&body[..end]);
        body = &body[end..];
    }
    Some(format!("{directives}{helper}{body}"))
}
