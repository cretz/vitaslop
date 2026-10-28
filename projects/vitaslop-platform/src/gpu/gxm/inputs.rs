//! What a draw was FED: the per-draw input reports, the uniform substitution that tests one
//! input's causality, and the decode of a uniform through its own declared type.

use super::*;

/// Diagnostic (`VITASLOP_GXP_INPUTS=<hex-key>[,<hex-key>]` or `=all`): print, ONCE per
/// shader pair, every value the guest actually fed that draw - each declared uniform
/// decoded through its own declared type, and the observed range of every vertex
/// attribute component over the draw's whole stream.
///
/// This exists because the two halves of a wrong picture are indistinguishable from the
/// frame: a coordinate that comes out wrong is either a uniform we are reading through the
/// wrong layout or an attribute we are decoding wrong, and every other diagnostic here
/// (`_PROBE`, `_VPROBE`, `_KEYCOLOR`) observes the shader's OUTPUT, which is downstream of
/// both. A uniform printed as its parameter declares it - `mainScaleBias F16[4] = (1, 1,
/// 0.25, 0.25)` - settles in one line what a register probe cannot settle at all.
/// `VITASLOP_GXP_QUADS=<hex-key>:<xmin>,<xmax>,<ymin>,<ymax>` - every SUBMISSION of the
/// named pair whose position lane intersects the box, with every vertex's every lane.
///
/// # Why this exists when `VITASLOP_GXP_INPUTS_VERTS` already dumps vertices
/// That dump is deduplicated per distinct INPUT SET, which for a UI pair submitted a
/// thousand times a frame means the element under investigation is almost never the one
/// printed - a trap that cost an hour on a smudged text label ("scanning its output for
/// 'which draw covers this box' silently sees a sample"). This one filters by WHERE the
/// draw lands instead, in the draw's OWN coordinate space (a UI pair's positions are in
/// the guest's UI space, not the shot's - convert before setting the box, and sanity-check
/// the conversion against an element whose position is known). The box is what bounds the
/// output; the cap below is the backstop that keeps a mis-set box from burying a run.
pub(super) fn report_quads(key: u64, gxp: &GxpRecompile) {
    use std::sync::atomic::{AtomicUsize, Ordering};
    use std::sync::OnceLock;
    static SPEC: OnceLock<Option<(u64, [f32; 4])>> = OnceLock::new();
    let Some((want, bx)) = SPEC.get_or_init(|| {
        let s = crate::knobs::var("VITASLOP_GXP_QUADS").ok()?;
        let (k, rest) = s.split_once(':')?;
        let k = u64::from_str_radix(k.trim().trim_start_matches("0x"), 16).ok()?;
        let mut it = rest.split(',').map(|v| v.trim().parse::<f32>().ok());
        let b = [it.next()??, it.next()??, it.next()??, it.next()??];
        Some((k, b))
    }) else {
        return;
    };
    if *want != key {
        return;
    }
    // The position lane is the attribute at register 0 - the convention every linked
    // vertex program here follows. A pair without one has nothing to filter on, and
    // saying so once beats silently printing nothing.
    let Some(pos) = gxp.attributes.iter().find(|a| a.reg_index == 0) else {
        report_knob!("gxp quads {key:016x}: no attribute at register 0 - cannot filter");
        return;
    };
    let stride = gxp.vertex_stride.max(1) as usize;
    let nverts = gxp.vertices.len() / stride;
    let (mut x0, mut x1, mut y0, mut y1) = (f32::MAX, f32::MIN, f32::MAX, f32::MIN);
    for v in 0..nverts {
        let x = read_attr_component(&gxp.vertices, v * stride + pos.offset as usize, pos.gxm_format, 0);
        let y = read_attr_component(&gxp.vertices, v * stride + pos.offset as usize, pos.gxm_format, 1);
        (x0, x1) = (x0.min(x), x1.max(x));
        (y0, y1) = (y0.min(y), y1.max(y));
    }
    // CONTAINMENT, not intersection: a UI scene's full-screen backdrop quads intersect
    // every box, and on an engine that renders every frame they burn the whole dump cap
    // before the element under investigation prints once. "Lands in the box" means the
    // draw's extent FITS in it.
    if nverts == 0 || x0 < bx[0] || x1 > bx[1] || y0 < bx[2] || y1 > bx[3] {
        return;
    }
    // Backstop, not the bound: a box over a busy region can still name hundreds of draws.
    const QUAD_DUMP_CAP: usize = 256;
    static PRINTED: AtomicUsize = AtomicUsize::new(0);
    let n = PRINTED.fetch_add(1, Ordering::Relaxed);
    if n == QUAD_DUMP_CAP {
        report_knob!(
            "gxp quads {key:016x}: over the {QUAD_DUMP_CAP}-draw cap - later matches are \
             NOT printed; tighten the box"
        );
    }
    if n >= QUAD_DUMP_CAP {
        return;
    }
    report_knob!(
        "gxp quads {key:016x} draw #{n}: {nverts} verts, x {x0:.1}..{x1:.1} y {y0:.1}..{y1:.1}"
    );
    for v in 0..nverts.min(8) {
        let cols: Vec<String> = gxp
            .attributes
            .iter()
            .map(|a| {
                let comps = a.components.clamp(1, 4) as usize;
                let vals: Vec<String> = (0..comps)
                    .map(|c| {
                        format!(
                            "{:.4}",
                            read_attr_component(&gxp.vertices, v * stride + a.offset as usize, a.gxm_format, c)
                        )
                    })
                    .collect();
                format!("lane{}=({})", a.reg_index, vals.join(","))
            })
            .collect();
        report_knob!("gxp quads {key:016x} draw #{n} v{v}: {}", cols.join(" "));
    }
}

pub(super) fn report_inputs(key: u64, gxp: &GxpRecompile) {
    // Through the knob table, not `std::env::var`: the browser has no environment, and this
    // is the diagnostic that answers "what was this draw FED" - which is the live question
    // for a defect that reproduces in the browser and nowhere a file can be written.
    // Resolved once - see `KeySpec` for the per-draw cost this used to pay in every run.
    use std::sync::OnceLock;
    static SPEC: OnceLock<KeySpec> = OnceLock::new();
    if !SPEC.get_or_init(|| KeySpec::resolve("VITASLOP_GXP_INPUTS")).wants(key) {
        return;
    }
    // Dedupe on the pair AND on the inputs themselves, not on the pair alone: one pair is
    // submitted many times a frame with DIFFERENT uniforms (that is what a per-draw uniform
    // buffer is for), and reporting only the first submission is how a diagnostic ends up
    // describing a draw that is not the one being investigated.
    //
    // >>> THE VERTEX BYTES ARE PART OF "the inputs", and leaving them out of this hash made
    // this diagnostic LIE. It reports attribute ranges and per-vertex values, so a draw whose
    // only per-frame change is in the vertex stream - a UI element faded through a per-vertex
    // COLOUR, which is how this title animates its text - matched an earlier submission on
    // uniforms and textures alone and was never printed again. The frame showed ten distinct
    // alphas while this said there were four sets of inputs, and the honest reading of that
    // contradiction was that the fade did not come through a uniform. It cost a wrong
    // conclusion before it cost anything else.
    //
    // Hashing the whole vertex buffer is real work on a world mesh. It is affordable because
    // nothing here runs unless the knob is set, and a diagnostic that silently under-reports
    // is worth less than one that is slow.
    use std::hash::{Hash, Hasher};
    use std::sync::Mutex;
    static SEEN: OnceLock<Mutex<HashSet<(u64, u64)>>> = OnceLock::new();
    let mut h = std::collections::hash_map::DefaultHasher::new();
    gxp.vert_sa.hash(&mut h);
    gxp.frag_sa.hash(&mut h);
    gxp.vertices.hash(&mut h);
    gxp.vertex_stride.hash(&mut h);
    // The memory windows too, for the reason the vertex bytes are here: on a program whose
    // uniforms lie past the SA container this is where the per-draw values LIVE, so a
    // submission that differs only in them is a different set of inputs and has to print.
    gxp.mem_windows.hash(&mut h);
    for t in gxp.textures.iter().chain(gxp.vertex_textures.iter()) {
        (t.unit, t.tex.data_addr, t.tex.width, t.tex.height).hash(&mut h);
    }
    let inputs_hash = h.finish();
    let seen = SEEN.get_or_init(|| Mutex::new(HashSet::default()));
    if !seen.lock().unwrap_or_else(|e| e.into_inner()).insert((key, inputs_hash)) {
        return;
    }
    // >>> AND A CAP ON HOW MANY DISTINCT INPUT SETS ONE PAIR MAY PRINT.
    //
    // The dedupe above is per (pair, INPUTS), which for a pair whose uniforms change every
    // submission - a stage mesh drawn 63 times a frame with its own `LocalToWorld` each
    // time - is not a dedupe at all: every draw is a new set, and each one prints its whole
    // memory window as hex. MEASURED: `VITASLOP_GXP_INPUTS` aimed at two keys wrote **914 MB
    // in under eight minutes** and was still growing, on a title whose whole run log is
    // otherwise a megabyte. An operator cannot aim a diagnostic that fills the disk before
    // it answers, and the run it buries is the run it was set for
    // [[vitaslop-a-diagnostic-can-bury-the-findings]].
    //
    // `VITASLOP_GXP_INPUTS_SETS=<n>` raises it for a case that genuinely needs the sweep;
    // the default is the number a human reads.
    const INPUT_SETS_DEFAULT: usize = 8;
    static SETS_CAP: OnceLock<usize> = OnceLock::new();
    let cap = *SETS_CAP.get_or_init(|| {
        crate::knobs::var("VITASLOP_GXP_INPUTS_SETS")
            .ok()
            .and_then(|v| v.trim().parse::<usize>().ok())
            .unwrap_or(INPUT_SETS_DEFAULT)
    });
    static PRINTED: OnceLock<Mutex<HashMap<u64, usize>>> = OnceLock::new();
    let printed = PRINTED.get_or_init(|| Mutex::new(HashMap::default()));
    {
        let mut p = printed.lock().unwrap_or_else(|e| e.into_inner());
        let n = p.entry(key).or_insert(0);
        *n += 1;
        if *n == cap + 1 {
            report_knob!(
                "gxp inputs {key:016x}: over the {cap}-input-set cap - later submissions of \
                 this pair are NOT printed. Raise it with VITASLOP_GXP_INPUTS_SETS=<n>, or \
                 aim at the submission you want with VITASLOP_GXP_QUADS."
            );
        }
        if *n > cap {
            return;
        }
    }
    for (stage, bytes, blob) in
        [("vertex", &gxp.vert_sa, &gxp.vprog), ("fragment", &gxp.frag_sa, &gxp.fprog)]
    {
        let Ok(program) = vitaslop_gxp_shader::Program::parse(blob) else { continue };
        // The raw words too, not only the decoded parameters. A parameter is decoded through
        // its declared offset/type, so a value that reads wrong is either the bytes or that
        // decode - and only the bytes can tell the two apart.
        // The fragment bank's guest ADDRESS goes on the line too - it is what a store watch
        // is pointed at, and without it "who wrote this uniform" needs a second run to
        // discover the address before it can even start.
        let at = match stage {
            "fragment" if gxp.frag_sa_addr != 0 => format!(" at {:#x}", gxp.frag_sa_addr),
            _ => String::new(),
        };
        report_knob!(
            "gxp inputs {key:016x} {stage}{at}: default uniform buffer is {} bytes for {} declared registers, raw = {}",
            bytes.len(),
            program.default_uniform_regs,
            bytes
                .chunks(4)
                .map(|c| {
                    let mut w = [0u8; 4];
                    w[..c.len()].copy_from_slice(c);
                    format!("{:08x}", u32::from_le_bytes(w))
                })
                .collect::<Vec<_>>()
                .join(" ")
        );
        for p in &program.parameters {
            use vitaslop_gxp_shader::container::ParamCategory;
            if p.category != ParamCategory::Uniform {
                continue;
            }
            let vals = decode_uniform(bytes, p, &program.containers);
            // The CONTAINER and the absolute SA register, not just `resource_index`: the
            // index alone is ambiguous across blocks (see `decode_uniform`), and the
            // absolute register is what the disassembly's `sa[k]` reads say.
            let at = program
                .containers
                .iter()
                .find(|c| c.index == u16::from(p.container_index))
                .map(|c| format!(" sa[{}]", c.base_sa as i64 + p.resource_index as i64))
                .unwrap_or_default();
            report_knob!(
                "gxp inputs {key:016x} {stage}:   {} {:?}[{}] in container {} at reg {}{} = {}",
                p.name, p.ptype, p.component_count, p.container_index, p.resource_index, at, vals
            );
        }
    }
    // The vertex program's GUEST-MEMORY WINDOWS, which the uniform lines above cannot show.
    //
    // A uniform that lies past its container's carried extent does not reach the shader
    // through the SA file at all - the compiler emits a memory load through the buffer's
    // own address instead, and the value the draw is fed is then entirely in these bytes.
    // The golf title's sky program is exactly that shape: 34 declared registers, 31 carried
    // in container 14, and `sunColor` at register 31 reached only through the window. With
    // no line here the parameter reads `<past the end of the buffer>` and the report has
    // nothing to say about the one input the final colour is made of.
    // BOUNDED, because a window is as big as the guest's buffer: one title's instance
    // array is 8,640 bytes and another's is 20 KB, and printing all of it once per input
    // set is most of what made this report unusable. The head is what a layout question
    // needs; the length is still reported in full.
    const MEM_WINDOW_WORDS: usize = 64;
    // BOTH STAGES. A fragment program reaches its own buffer the same way, and a generic
    // blend shader - whose SRC/DST coefficients all arrive through the window - has its
    // whole answer in these bytes and nothing in its uniforms.
    for (stage, windows) in
        [("vertex", &gxp.mem_windows), ("fragment", &gxp.frag_mem_windows)]
    {
    for (i, (addr, bytes)) in windows.iter().enumerate() {
        let words = bytes.len().div_ceil(4);
        report_knob!(
            "gxp inputs {key:016x} {stage}: memory window {i} at {addr:#x}, {} bytes ({} words{}) = {}",
            bytes.len(),
            words,
            if words > MEM_WINDOW_WORDS {
                format!(", first {MEM_WINDOW_WORDS} shown")
            } else {
                String::new()
            },
            bytes
                .chunks(4)
                .take(MEM_WINDOW_WORDS)
                .map(|c| {
                    let mut w = [0u8; 4];
                    w[..c.len()].copy_from_slice(c);
                    format!("{:08x}", u32::from_le_bytes(w))
                })
                .collect::<Vec<_>>()
                .join(" ")
        );
    }
    }
    // The guest's VIEWPORT, which is what turns the vertex program's clip output into
    // The guest's CULL MODE, which no pipeline here applies (every one of them is built
    // with `cull_mode: None`). It belongs in this report for the same reason the viewport
    // does: it is state the draw was FED, it is not visible in the shader or the uniforms,
    // and ignoring it changes the picture. A fullscreen overlay triangle that the guest
    // expects to be culled is drawn instead, over everything.
    report_knob!(
        "gxp inputs {key:016x} cull: mode={} ({}){}",
        gxp.cull_mode,
        match gxp.cull_mode {
            0 => "NONE",
            1 => "CW",
            2 => "CCW",
            _ => "unknown",
        },
        match gxm_cull_face(gxp.cull_mode) {
            Some(wgpu::Face::Front) => "  - APPLIED, front faces discarded",
            Some(wgpu::Face::Back) => "  - APPLIED, back faces discarded",
            None => "",
        }
    );
    // target pixels. A post-process pass that samples a source at a scale/bias only lands
    // right if the source was RENDERED where the pass thinks it was, and the viewport is
    // the one piece of that which is neither in the shader nor in the uniforms.
    report_knob!(
        "gxp inputs {key:016x} viewport: xOffset={} xScale={} yOffset={} yScale={} zOffset={} zScale={}{}",
        gxp.viewport[0], gxp.viewport[1], gxp.viewport[2],
        gxp.viewport[3], gxp.viewport[4], gxp.viewport[5],
        if gxp.viewport.iter().all(|v| *v == 0.0) { "  (all zero = the guest left the default)" } else { "" }
    );
    // What each declared sampler is actually bound to. A post-process pass's whole content
    // is the geometry of the buffer it reads, and "unit 1" in a shader and "unit 1" in the
    // guest's texture state are only the same thing if the binding says so.
    //
    // BOTH stages, and the vertex one is not a footnote: a vertex program that samples
    // builds its GEOMETRY from what it reads, so an unbound or empty vertex texture is a
    // draw with no mesh rather than a surface with the wrong colours. Leaving it out of
    // this report is what left "the campaign map body is black" a content question nobody
    // could answer - the fragment side said everything was bound, and it was.
    for (stage, blob, bound) in [
        ("sampler", &gxp.fprog, &gxp.textures),
        ("vertex sampler", &gxp.vprog, &gxp.vertex_textures),
    ] {
        let Ok(program) = vitaslop_gxp_shader::Program::parse(blob) else { continue };
        for p in &program.parameters {
            if p.category != vitaslop_gxp_shader::container::ParamCategory::Sampler {
                continue;
            }
            match bound.iter().find(|t| t.unit as i32 == p.resource_index) {
                // The first few DECODED texels, not just the binding. "Which texture" and
                // "what is in it" are different questions, and for a texture the shader
                // reads as DATA rather than as a picture - a vector canvas, a lookup
                // table, an index map - the second one is the whole answer, and the
                // binding line alone has already sent one investigation down the wrong
                // road. Also says outright when every sampled texel is the same value.
                Some(t) => report_knob!(
                    "gxp inputs {key:016x} {stage}: {} unit {} <- {:#x} {}x{} faces={} \
                     fmt={:#04x} swz={:#x} gamma={} filter={} wrap=({},{}) seam={:?} texels[0..4]={:?} {}{}",
                    p.name,
                    p.resource_index,
                    t.tex.data_addr,
                    t.tex.width,
                    t.tex.height,
                    t.tex.faces,
                    t.tex.base_format,
                    t.tex.swizzle,
                    t.tex.gamma,
                    if t.tex.filter_linear { "linear" } else { "point" },
                    t.tex.addr_mode_u,
                    t.tex.addr_mode_v,
                    t.tex.texel,
                    // Read through the seam, so a half texture prints the values the shader
                    // actually gets rather than pairs of its bytes read as texels.
                    seam_texels(&t.tex).into_iter().take(4).collect::<Vec<_>>(),
                    channel_spread_seam(&t.tex),
                    // "carries no image" is a statement about the GUEST BYTES, and saying
                    // only that has now sent two investigations down the same wrong road:
                    // a shadow map or any other render-to-texture surface reads empty here
                    // BY CONSTRUCTION - the guest never writes it, the GPU attachment does,
                    // and the renderer substitutes that at bind time. The line has to carry
                    // its own caveat, because the reading it invites otherwise ("the shadow
                    // map is empty, that is why this draw is black") is both wrong and
                    // extremely plausible.
                    match seam_texels(&t.tex).first() {
                        Some(first) if seam_texels(&t.tex).iter().all(|c| c == first) =>
                            " (every texel of the GUEST BYTES is identical - which is \
                             EXPECTED of a render-to-texture or depth surface, whose \
                             contents live in a GPU attachment substituted at bind time, \
                             and is a finding only for a texture the guest uploads itself)",
                        _ => "",
                    }
                ),
                None => report_knob!(
                    "gxp inputs {key:016x} {stage}: {} unit {} <- NOTHING BOUND",
                    p.name, p.resource_index
                ),
            }
            // `VITASLOP_GXP_INPUTS_DIR=<dir>`: the decoded texture itself, as a PNG named by
            // the parameter. A channel spread says a channel is flat; only the image says
            // whether the picture in the other three is the one the shader expects, and for
            // a texture read as DATA (a displacement canvas, a lookup table) that is the
            // difference between "we decoded it wrong" and "the producer never ran".
            if let (Ok(dir), Some(t)) =
                (std::env::var("VITASLOP_GXP_INPUTS_DIR"), bound.iter().find(|t| t.unit as i32 == p.resource_index))
            {
                let _ = std::fs::create_dir_all(&dir);
                let safe: String = p.name.chars().filter(|c| c.is_ascii_alphanumeric()).collect();
                // Raw RGBA8 with the dimensions in the NAME, not a PNG: this crate has no
                // image encoder and must not grow one for a diagnostic. The size is in the
                // filename so the bytes are self-describing.
                let path = std::path::Path::new(&dir).join(format!(
                    "{key:016x}_{}_u{}_{safe}_{}x{}.rgba",
                    stage.replace(' ', "-"),
                    p.resource_index,
                    t.tex.width,
                    t.tex.height * t.tex.faces
                ));
                let _ = std::fs::write(path, t.tex.rgba.as_slice());
            }
        }
    }
    // Attribute RANGES, not the first vertex: a scale/bias question is a question about the
    // span of the coordinate across the mesh, and one vertex cannot answer it.
    let Ok(vprogram) = vitaslop_gxp_shader::Program::parse(&gxp.vprog) else { return };
    let stride = gxp.vertex_stride.max(1) as usize;
    let nverts = gxp.vertices.len() / stride;
    // THE VERTICES THE DRAW ACTUALLY REFERENCES, not every vertex in the uploaded stream.
    //
    // A stream commonly holds more than one draw's worth of geometry - a top ring and a
    // skirt ring, an LOD chain, a shared pool - and only the INDEX BUFFER says which of it
    // this draw touches. Ranging over the whole stream reports a component as reaching a
    // value no drawn triangle ever sees, which is a statement about the buffer wearing the
    // clothes of a statement about the picture. It also makes our own index expansion
    // unfalsifiable: if we index vertices the guest never asked for, the two populations
    // differ and nothing here could ever have said so.
    let indexed: Vec<usize> = {
        let mut seen = std::collections::BTreeSet::new();
        let b = gxp.indices.as_ref();
        let n = gxp.index_count as usize;
        if gxp.index_u32 {
            for i in 0..n.min(b.len() / 4) {
                seen.insert(u32::from_le_bytes([b[i * 4], b[i * 4 + 1], b[i * 4 + 2], b[i * 4 + 3]]) as usize);
            }
        } else {
            for i in 0..n.min(b.len() / 2) {
                seen.insert(u16::from_le_bytes([b[i * 2], b[i * 2 + 1]]) as usize);
            }
        }
        seen.into_iter().filter(|&v| v < nverts).collect()
    };
    // The TOPOLOGY, and the first triangle it forms. A mesh whose vertices are right can
    // still be assembled wrong: read a triangle LIST as a STRIP and every triangle after the
    // first straddles two of the source's, which shows up as smooth INTERPOLATION across
    // faces that should each be flat - a shading artefact that looks like a lighting bug and
    // is nothing of the kind. The guest's primitive word never appeared in any report, so
    // that possibility could not be checked at all.
    report_knob!(
        "gxp inputs {key:016x} topology: guest primitive {:#x}, {} indices -> {} triangles \
         over {nverts} vertices; first triangle = {:?}",
        gxp.primitive,
        gxp.index_count,
        gxp.index_count / 3,
        {
            let b = gxp.indices.as_ref();
            let rd = |i: usize| -> u32 {
                if gxp.index_u32 {
                    b.get(i * 4..i * 4 + 4).map(|c| u32::from_le_bytes([c[0], c[1], c[2], c[3]])).unwrap_or(0)
                } else {
                    b.get(i * 2..i * 2 + 2).map(|c| u16::from_le_bytes([c[0], c[1]]) as u32).unwrap_or(0)
                }
            };
            [rd(0), rd(1), rd(2), rd(3), rd(4), rd(5)]
        }
    );
    if !indexed.is_empty() && indexed.len() != nverts {
        report_knob!(
            "gxp inputs {key:016x} vertices: the stream holds {nverts} but the draw's indices \
             reference only {} of them ({}..={}) - the ranges below are over the INDEXED set",
            indexed.len(),
            indexed.first().copied().unwrap_or(0),
            indexed.last().copied().unwrap_or(0)
        );
    }
    // Fall back to the whole stream only when there is no index buffer to narrow it.
    let sample: Vec<usize> = if indexed.is_empty() { (0..nverts).collect() } else { indexed };
    let nverts = sample.len();
    for a in gxp.attributes.iter() {
        let name = vprogram
            .parameters
            .iter()
            .find(|p| {
                p.category == vitaslop_gxp_shader::container::ParamCategory::Attribute
                    && p.resource_index == a.reg_index as i32
            })
            .map(|p| p.name.as_str())
            .unwrap_or("<unnamed>");
        let comps = a.components.clamp(1, 4) as usize;
        let mut lo = [f32::INFINITY; 4];
        let mut hi = [f32::NEG_INFINITY; 4];
        let mut vals: Vec<Vec<f32>> = vec![Vec::new(); comps];
        for &v in &sample {
            for c in 0..comps {
                let f = read_attr_component(&gxp.vertices, v * stride + a.offset as usize, a.gxm_format, c);
                lo[c] = lo[c].min(f);
                hi[c] = hi[c].max(f);
                vals[c].push(f);
            }
        }
        // A RANGE says a component reaches -1 somewhere; it does not say whether that is one
        // vertex or half the mesh, and those two readings send an investigation in opposite
        // directions. Eight equal buckets across the span answer it in a bounded line - the
        // reason this is not the per-vertex dump, which is capped out on any real mesh.
        let shape = |c: usize| -> String {
            if !(lo[c].is_finite() && hi[c].is_finite()) || hi[c] <= lo[c] {
                return "constant".into();
            }
            let mut bins = [0usize; 8];
            let span = hi[c] - lo[c];
            for &f in &vals[c] {
                let b = (((f - lo[c]) / span) * 8.0) as usize;
                bins[b.min(7)] += 1;
            }
            let pct: Vec<String> = bins
                .iter()
                .map(|&n| format!("{:.0}", 100.0 * n as f32 / nverts.max(1) as f32))
                .collect();
            pct.join("/")
        };
        let ranges: Vec<String> = (0..comps)
            .map(|c| format!("[{:.4}, {:.4}] {}%", lo[c], hi[c], shape(c)))
            .collect();
        report_knob!(
            "gxp inputs {key:016x} attribute: {name} lane {} at byte {} of a {}-byte vertex, \
             fmt {} x{} over {nverts} vertices = {}",
            a.reg_index,
            a.offset,
            gxp.vertex_stride,
            a.gxm_format,
            comps,
            ranges.join(" ")
        );
    }
    // Per-VERTEX values, behind their OWN knob (`VITASLOP_GXP_INPUTS_VERTS=<key>|all`).
    // A component RANGE only pins down an attribute if the mesh maps it affinely onto the
    // screen, and a post-process DISTORTION GRID is exactly the mesh that does not - so
    // sometimes the vertices themselves are the answer.
    //
    // # Why naming the pair is NOT enough to ask for this
    // It used to be: naming a pair explicitly (rather than `all`) turned the dump on, capped
    // at `MAX_DUMPED_VERTICES` = 512 so it could not "bury the frame's other reports in a
    // megabyte of numbers". That cap is sized for a LOG FILE. The browser's diagnostics panel
    // keeps a bounded number of DISTINCT lines - 96 - and this title's final composite is a
    // 288-vertex grid, so one input set of per-vertex lines evicts every other finding in the
    // panel. MEASURED on the user's phone: `10378 earlier DISTINCT line(s) dropped`, and the
    // four uniform lines the run was taken FOR were among them. The knob answered, and its
    // answer was destroyed by the same knob's other output.
    //
    // So the dump is opt-in on its own name. Naming a pair in `VITASLOP_GXP_INPUTS` now gets
    // the uniforms, the samplers, the viewport and the attribute RANGES - a bounded number of
    // lines that fits any sink. [[vitaslop-a-diagnostic-can-bury-the-findings]]
    let verts_wanted = crate::knobs::var("VITASLOP_GXP_INPUTS_VERTS")
        .map(|s| {
            s.split(',').any(|k| {
                let k = k.trim();
                k == "all" || u64::from_str_radix(k.trim_start_matches("0x"), 16) == Ok(key)
            })
        })
        .unwrap_or(false);
    if !verts_wanted {
        return;
    }
    // A MESH TOO BIG TO DUMP STILL GETS ITS FIRST FEW VERTICES. The cap exists so one draw
    // cannot bury a frame's other findings, and that is right - but returning EMPTY makes the
    // instrument silent exactly on the meshes worth asking about, and "the knob printed
    // nothing" then reads as "there is nothing there". The question a raw record answers -
    // is this field really at this byte, is this value really the float it decodes to - is
    // answered by eight vertices as well as by seventeen thousand.
    // [[vitaslop-instrument-failure-imitating-its-subject]]
    const HEAD: usize = 8;
    let shown = if nverts > MAX_DUMPED_VERTICES { HEAD.min(nverts) } else { nverts };
    if shown < nverts {
        report_knob!(
            "gxp inputs {key:016x} vertices: {nverts} is over the {MAX_DUMPED_VERTICES} dump \
             cap - printing the FIRST {shown} only; the attribute RANGES above cover all of them"
        );
    }
    for v in 0..shown {
        let cols: Vec<String> = gxp
            .attributes
            .iter()
            .map(|a| {
                let comps = a.components.clamp(1, 4) as usize;
                let vals: Vec<String> = (0..comps)
                    .map(|c| {
                        format!(
                            "{:.4}",
                            read_attr_component(&gxp.vertices, v * stride + a.offset as usize, a.gxm_format, c)
                        )
                    })
                    .collect();
                format!("lane{}=({})", a.reg_index, vals.join(","))
            })
            .collect();
        // The RAW vertex bytes too. A guest vertex often carries fields no declared attribute
        // names, and an attribute read at the wrong byte offset produces plausible numbers -
        // so the only way to tell "this really is the mesh's UV" from "this is some other
        // field that happens to look like one" is to see the whole record.
        let raw: String = gxp
            .vertices
            .get(v * stride..(v + 1) * stride)
            .map(|b| b.chunks(4).map(|c| {
                let mut w = [0u8; 4];
                w[..c.len()].copy_from_slice(c);
                format!("{}", f32::from_le_bytes(w))
            }).collect::<Vec<_>>().join(" "))
            .unwrap_or_default();
        report_knob!("gxp inputs {key:016x} vertex {v}: {}   raw-as-f32 [{raw}]", cols.join(" "));
    }
}

/// Most vertices `VITASLOP_GXP_INPUTS` will print individually. A post-process grid or a UI
/// quad is small enough to read; a world mesh is not, and dumping one would bury the frame's
/// other reports in a megabyte of numbers.
pub(super) const MAX_DUMPED_VERTICES: usize = 512;

/// Diagnostic (`VITASLOP_GXP_INPUTS_ORDER=<hex-key>[,<hex-key>]`): print ONE compact line per
/// SUBMISSION of that pair, in submission order, never deduped.
///
/// # Why this exists next to [`report_inputs`], which already prints every input
/// [`report_inputs`] deduplicates on `(pair, inputs-hash)` for the whole run, so what it
/// prints is the SET of distinct input sets in FIRST-SEEN order. That is the right shape for
/// "what is this draw fed" and the wrong shape for "in what order is it fed", and the two are
/// indistinguishable when read off the same log: a first-seen list looks monotonic whatever
/// the real order is, because a value's second appearance is exactly what the dedupe drops.
///
/// This was not a hypothetical. A fade animated through a per-vertex COLOUR was read off that
/// deduped list as "two clean monotonic ramps", which excluded the guest as the source of a
/// flicker whose whole content is which alpha lands in which frame. The ordering question
/// needs an instrument that repeats itself; that is this one.
///
/// Deliberately one line per submission with RANGES only - a per-vertex dump repeated at draw
/// rate is not readable, and a ramp is a question about the span, not about vertex 7.
///
/// # Why the SMALL declared uniforms are printed by VALUE and not only as a hash
/// This line used to carry a single `sa <hash>` for the whole uniform block, which makes it
/// able to say WHEN a uniform changed and never WHAT it changed to. That is half an
/// instrument, and the half it is missing is the half the ordering question needs: the
/// project's own notes told the next session to "use `_INPUTS_ORDER` if the order must be
/// exact", and following that instruction produces a column of opaque hashes. The values
/// then have to be recovered from [`report_inputs`], whose output is DEDUPED and
/// first-seen-ordered - which is the exact instrument the ordering question was trying to
/// get away from. A fade was read off that pairing as running upward when it runs downward.
///
/// Only uniforms of at most [`MAX_ORDERED_UNIFORM_COMPONENTS`] components are printed.
/// That is not a cosmetic budget: the uniforms a fade or a transition is driven through are
/// scalars and short vectors (`screenTintColour`, `bloomFactor`, `posOffset`), while the
/// ones that would drown the line are matrices and sample-offset tables (a `worldViewProj`
/// is 16 components and one map pair declares 94 registers). Printing everything at draw
/// rate would bury the finding in exactly the way
/// [[vitaslop-a-diagnostic-can-bury-the-findings]] describes. The hash still covers the
/// WHOLE block, so a change in a table that is not printed still shows up as a hash change
/// and cannot be mistaken for "nothing moved".
pub(super) fn report_inputs_order(key: u64, gxp: &GxpRecompile) {
    // Through the knob table - see `report_inputs` for why - and resolved ONCE, see
    // `KeySpec`. This is the hottest of these: it runs per SUBMISSION, not per pair.
    use std::sync::OnceLock;
    static SPEC: OnceLock<KeySpec> = OnceLock::new();
    if !SPEC.get_or_init(|| KeySpec::resolve("VITASLOP_GXP_INPUTS_ORDER")).wants(key) {
        return;
    }
    use std::sync::atomic::{AtomicU64, Ordering};
    static SEQ: AtomicU64 = AtomicU64::new(0);
    let seq = SEQ.fetch_add(1, Ordering::Relaxed);
    let stride = gxp.vertex_stride.max(1) as usize;
    let nverts = gxp.vertices.len() / stride;
    let mut cols: Vec<String> = Vec::new();
    for a in gxp.attributes.iter() {
        let comps = a.components.clamp(1, 4) as usize;
        let mut lo = [f32::INFINITY; 4];
        let mut hi = [f32::NEG_INFINITY; 4];
        for v in 0..nverts {
            for c in 0..comps {
                let f = read_attr_component(
                    &gxp.vertices,
                    v * stride + a.offset as usize,
                    a.gxm_format,
                    c,
                );
                lo[c] = lo[c].min(f);
                hi[c] = hi[c].max(f);
            }
        }
        let r: Vec<String> = (0..comps)
            .map(|c| {
                if lo[c] == hi[c] {
                    format!("{:.4}", lo[c])
                } else {
                    format!("{:.4}..{:.4}", lo[c], hi[c])
                }
            })
            .collect();
        cols.push(format!("lane{}=({})", a.reg_index, r.join(",")));
    }
    // The uniform bytes fold in too: a fade driven through a uniform and one driven through a
    // vertex colour produce the same picture, and this line has to be able to tell them apart
    // without a second run.
    use std::hash::{Hash, Hasher};
    let mut h = std::collections::hash_map::DefaultHasher::new();
    gxp.vert_sa.hash(&mut h);
    gxp.frag_sa.hash(&mut h);
    // The SHORT declared uniforms by value, so this line can answer "what did it change to"
    // and not only "it changed" - see the doc comment above for why the long ones are left
    // to the hash.
    let mut named: Vec<String> = Vec::new();
    for (stage, bytes, blob) in
        [("v", &gxp.vert_sa, &gxp.vprog), ("f", &gxp.frag_sa, &gxp.fprog)]
    {
        let Ok(program) = vitaslop_gxp_shader::Program::parse(blob) else { continue };
        for p in &program.parameters {
            use vitaslop_gxp_shader::container::ParamCategory;
            if p.category != ParamCategory::Uniform
                || p.component_count as usize > MAX_ORDERED_UNIFORM_COMPONENTS
            {
                continue;
            }
            named.push(format!("{stage}:{}={}", p.name, decode_uniform(bytes, p, &program.containers)));
        }
    }
    report_knob!(
        "gxp order {key:016x} #{seq}: {nverts} verts, sa {:016x}, {}{}",
        h.finish(),
        cols.join(" "),
        if named.is_empty() { String::new() } else { format!(" | {}", named.join(" ")) }
    );
}

/// The widest declared uniform [`report_inputs_order`] prints by VALUE on its per-submission
/// line. See that function's doc comment: short vectors are what animations are driven
/// through, matrices and lookup tables are what would bury the line.
pub(super) const MAX_ORDERED_UNIFORM_COMPONENTS: usize = 8;

/// Diagnostic (`VITASLOP_GXP_SA=<key>:<v|f>:<reg>=<hexword>[,...]`): replace a default-uniform
/// register with a chosen 32-bit word before the draw is submitted.
///
/// This is the causality half of [`report_inputs`]. Reading a uniform tells you what a shader
/// was fed; it does not tell you whether that value is what makes the picture wrong, because
/// every downstream stage is a candidate too. Substituting the value and re-rendering is the
/// one experiment that separates them, and it takes one run instead of a session of reasoning.
/// A substitution is REPORTED, once per (pair, stage, register): a run whose frame came from
/// values the guest never wrote must never be mistaken for a run of the real thing.
pub(super) fn override_sa<'a>(key: u64, stage: char, bytes: &'a [u8]) -> std::borrow::Cow<'a, [u8]> {
    // Through the knob table - see `report_inputs`. This one especially: it is the
    // causality experiment, and the defect it is aimed at reproduces only in the browser.
    // >>> THE OFF PATH IS FREE, and it was not: this runs TWICE PER DRAW (vertex and
    // fragment), and reading the knob here cost a mutex plus an `std::env::var` allocation
    // each time - ~2,500 per frame on the user's device - before re-parsing and
    // re-VALIDATING a spec that is almost always absent. Only the presence check is cached;
    // when the knob IS set the full parse still runs per draw, keeping every fail-loud
    // panic below exactly as it was. A diagnostic nobody enabled must not be on the bill.
    use std::sync::OnceLock;
    static SET: OnceLock<bool> = OnceLock::new();
    if !*SET.get_or_init(|| crate::knobs::var("VITASLOP_GXP_SA").is_ok()) {
        return std::borrow::Cow::Borrowed(bytes);
    }
    let Ok(spec) = crate::knobs::var("VITASLOP_GXP_SA") else {
        return std::borrow::Cow::Borrowed(bytes);
    };
    let mut out = std::borrow::Cow::Borrowed(bytes);
    for item in spec.split(',').filter(|s| !s.trim().is_empty()) {
        let parts: Vec<&str> = item.trim().split(':').collect();
        let [k, st, assign] = parts[..] else {
            panic!("VITASLOP_GXP_SA item {item:?} is not <key>:<v|f>:<reg>=<hexword>");
        };
        // `*` is EVERY pair. The pair key is a pointer-identity cache built at draw time,
        // so it is not the same number in a different process - which makes it unusable in
        // the capsule replay (`vitaslop_runtime::capsule`), where a substitution is exactly
        // what one wants and there is only ever ONE pair to aim it at. Naming a specific key
        // still works and is still what a live run should use, because there a wildcard
        // would substitute the register in every shader in the frame.
        let want_key = if k.trim() == "*" {
            key
        } else {
            match u64::from_str_radix(k.trim_start_matches("0x"), 16) {
                Ok(v) => v,
                Err(_) => panic!("VITASLOP_GXP_SA item {item:?} has a non-hex pair key (or `*` for every pair)"),
            }
        };
        let want_stage = match st {
            "v" => 'v',
            "f" => 'f',
            other => panic!("VITASLOP_GXP_SA item {item:?} names stage {other:?}, not v or f"),
        };
        let Some((reg, word)) = assign.split_once('=') else {
            panic!("VITASLOP_GXP_SA item {item:?} is missing the =<hexword>");
        };
        let (Ok(reg), Ok(word)) = (
            reg.trim().parse::<usize>(),
            u32::from_str_radix(word.trim().trim_start_matches("0x"), 16),
        ) else {
            panic!("VITASLOP_GXP_SA item {item:?} has a bad register or word");
        };
        if want_key != key || want_stage != stage {
            continue;
        }
        let buf = out.to_mut();
        if buf.len() < (reg + 1) * 4 {
            buf.resize((reg + 1) * 4, 0);
        }
        buf[reg * 4..reg * 4 + 4].copy_from_slice(&word.to_le_bytes());
        report_sa_override(key, stage, reg, word);
    }
    out
}

/// Report - once per (pair, stage, register) - that a uniform register was substituted by
/// `VITASLOP_GXP_SA`. See [`override_sa`] for why this is never silent.
pub(super) fn report_sa_override(key: u64, stage: char, reg: usize, word: u32) {
    use std::sync::{Mutex, OnceLock};
    static SEEN: OnceLock<Mutex<HashSet<(u64, char, usize)>>> = OnceLock::new();
    let seen = SEEN.get_or_init(|| Mutex::new(HashSet::default()));
    let mut seen = seen.lock().unwrap_or_else(|e| e.into_inner());
    if seen.insert((key, stage, reg)) {
        // At WARN, not `report!`. This one carries a correctness guarantee - "a run whose
        // frame came from values the guest never wrote must never be mistaken for a run of
        // the real thing" - and `report!` is BELOW the browser's default filter, so on the
        // engine where this knob was just made reachable the guarantee was silently void.
        // The first substituted run there produced a frame with no indication it was one.
        report_warn!(
            "gxp pair {key:016x}: {} uniform register {reg} SUBSTITUTED with {word:#010x} - \
             this frame is NOT what the guest asked for",
            if stage == 'v' { "vertex" } else { "fragment" }
        );
    }
}

/// One uniform parameter's values, read out of the SA REGISTER FILE IMAGE through its OWN
/// declared type. `resource_index` is a 4-byte register offset; the components are packed
/// from there at the type's own component width, which is how an F16 float4 fits in two
/// registers and an F32 float4 needs four.
///
/// >>> THE OFFSET IS RELATIVE TO THE PARAMETER'S OWN CONTAINER, NOT TO THE IMAGE. `bytes`
/// > > > is what `VitaState::sa_uniform_image` lays out: the default buffer at container 14's
/// > > > stored `base_sa` and every SA-resident buffer at its own. A program whose default
/// > > > container sits at `sa[88]` with an 80-register buffer underneath it therefore has TWO
/// > > > parameters at `resource_index = 0`, in different address spaces, and reading both from
/// > > > offset 0 prints one of them over the other. This diagnostic exists to say what a draw
/// > > > was FED, so a value it prints from the wrong block is worse than no line at all - it
/// > > > reads as a measurement. A parameter naming a container this program does not declare
/// > > > gets `None` and is reported that way rather than guessed at zero.
pub(super) fn decode_uniform(
    bytes: &[u8],
    p: &vitaslop_gxp_shader::container::Parameter,
    containers: &[vitaslop_gxp_shader::container::Container],
) -> String {
    use vitaslop_gxp_shader::container::ParamType;
    let Some(width) = p.ptype.component_bytes() else {
        return format!("<{:?} has no fixed component width>", p.ptype);
    };
    let base_sa = if containers.is_empty() {
        // No container table at all: the whole image IS the default buffer at register 0,
        // which is the shape `sa_uniform_image` returns unchanged.
        0
    } else {
        match containers.iter().find(|c| c.index == u16::from(p.container_index)) {
            Some(c) => c.base_sa,
            // >>> "NOT DECLARED" IS THE WRONG WORD FOR IT, and reading it as the right one
            // costs a session. A container below 14 is an ordinary BOUND UNIFORM BUFFER:
            // the program reaches it by chasing the pointer the driver placed in an SA
            // register, so it is absent from the container table BY CONSTRUCTION and its
            // bytes are in the `memory window` lines this same report prints a few lines
            // later. The old wording said the program had nothing to do with the
            // parameter - for `UFP_LightColorAndLightMult`, the term a lit surface is made
            // of, on the very program being investigated.
            //
            // So name the buffer and hand over the two offsets a reader actually needs:
            // the BYTE offset to index the printed window hex with, and the F16 LANE, which
            // is the numbering `VITASLOP_GXP_FMEM` and the capsule's `--fmem` both take.
            // Neither is derivable from this line without knowing the convention.
            None => {
                let byte = p.resource_index.max(0) as usize * 4;
                return format!(
                    "<in bound uniform buffer {} - see this pair's `memory window` lines, \
                     at byte {byte} (F16 lane {})>",
                    p.container_index,
                    byte / 2,
                );
            }
        }
    };
    let base = (base_sa as usize + p.resource_index.max(0) as usize) * 4;
    let n = p.component_count as usize * p.array_size.max(1) as usize;
    let mut out: Vec<String> = Vec::with_capacity(n);
    for i in 0..n {
        let o = base + i * width as usize;
        let Some(raw) = bytes.get(o..o + width as usize) else {
            out.push("<past the end of the buffer>".to_string());
            break;
        };
        out.push(match p.ptype {
            ParamType::F32 => format!("{}", f32::from_le_bytes([raw[0], raw[1], raw[2], raw[3]])),
            ParamType::F16 => format!("{}", half_to_f32(u16::from_le_bytes([raw[0], raw[1]]))),
            ParamType::U32 => format!("{}", u32::from_le_bytes([raw[0], raw[1], raw[2], raw[3]])),
            ParamType::S32 => format!("{}", i32::from_le_bytes([raw[0], raw[1], raw[2], raw[3]])),
            ParamType::U16 | ParamType::C10 => format!("{}", u16::from_le_bytes([raw[0], raw[1]])),
            ParamType::S16 => format!("{}", i16::from_le_bytes([raw[0], raw[1]])),
            ParamType::U8 => format!("{}", raw[0]),
            ParamType::S8 => format!("{}", raw[0] as i8),
            ParamType::Aggregate | ParamType::Unknown(_) => unreachable!("no component width"),
        });
    }
    format!("({})", out.join(", "))
}

#[cfg(test)]
mod decode_uniform_tests {
    use super::decode_uniform;
    use vitaslop_gxp_shader::container::{Container, ParamCategory, ParamType, Parameter};

    fn uniform(name: &str, container: u8, resource_index: i32, components: u8) -> Parameter {
        Parameter {
            name: name.to_string(),
            category: ParamCategory::Uniform,
            ptype: ParamType::F32,
            component_count: components,
            container_index: container,
            sampler_cube: false,
            array_size: 1,
            resource_index,
            semantic: 0,
            semantic_index: 0,
        }
    }

    /// An image of `regs` registers whose every word is its own register number as a float,
    /// so a decoded value names the register it was read from and nothing else can.
    fn image(regs: usize) -> Vec<u8> {
        (0..regs).flat_map(|r| (r as f32).to_le_bytes()).collect()
    }

    /// The case this exists for: two parameters at `resource_index = 0` in DIFFERENT
    /// containers. Reading both from offset 0 - which is what this did before the container
    /// base was applied - prints one of them over the other, and both lines read as
    /// measurements. The golf title's sky program is exactly this shape.
    #[test]
    fn a_uniform_is_read_from_its_own_containers_base() {
        let containers = [
            Container { index: 0, base_sa: 0, size_regs: 80 },
            Container { index: 14, base_sa: 88, size_regs: 31 },
        ];
        let img = image(119);
        assert_eq!(decode_uniform(&img, &uniform("g.WVP", 0, 0, 1), &containers), "(0)");
        assert_eq!(decode_uniform(&img, &uniform("wvp", 14, 0, 1), &containers), "(88)");
        // And an offset WITHIN the non-zero container, which is where an off-by-a-base
        // lands on a plausible neighbouring value rather than an obviously wrong one.
        assert_eq!(decode_uniform(&img, &uniform("topColor", 14, 26, 3), &containers), "(114, 115, 116)");
    }

    /// A parameter past its container's carried extent must say so rather than read the
    /// NEXT container's bytes - on this title that neighbour is the DATA container, and a
    /// pointer word decoded as a colour is a value nobody would question.
    #[test]
    fn a_uniform_past_the_end_of_the_image_reports_it() {
        let containers = [Container { index: 14, base_sa: 88, size_regs: 31 }];
        let img = image(119);
        let vals = decode_uniform(&img, &uniform("sunColor", 14, 31, 3), &containers);
        assert!(vals.contains("past the end of the buffer"), "{vals}");
    }

    /// A container the program does not declare is reported, not silently taken as zero:
    /// this title declares `g_PointLight[..]` in container 4 and never carries container 4.
    ///
    /// And it is reported as what it IS - a bound uniform BUFFER whose bytes are in the
    /// report's own `memory window` lines - with the byte offset and the F16 LANE to reach
    /// it by. The old wording ("is not declared by this program") read as "this parameter
    /// has nothing to do with this program", which for the term a lit surface is made of
    /// sent a whole investigation the wrong way.
    #[test]
    fn an_undeclared_container_is_named_as_the_bound_buffer_it_is() {
        let containers = [Container { index: 14, base_sa: 88, size_regs: 31 }];
        let vals = decode_uniform(&image(119), &uniform("g_PointLight", 4, 6, 4), &containers);
        assert!(vals.contains("bound uniform buffer 4"), "{vals}");
        // resource_index 6 is register 6 = byte 24 = F16 lane 12, the numbering
        // `VITASLOP_GXP_FMEM` takes.
        assert!(vals.contains("byte 24"), "{vals}");
        assert!(vals.contains("F16 lane 12"), "{vals}");
    }

    /// The overwhelmingly common shape - no container table at all - still reads the image
    /// as the default buffer at register 0, which is what `sa_uniform_image` returns then.
    #[test]
    fn no_container_table_means_the_image_is_the_default_buffer() {
        assert_eq!(decode_uniform(&image(8), &uniform("m", 0, 3, 1), &[]), "(3)");
    }
}
