//! Re-render a captured draw ([`vitaslop_runtime::capsule`]) offline, in a second, instead of
//! replaying the title to reach it.
//!
//! ```text
//! # capture, once, during a normal run:
//! VITASLOP_GXP_CAPSULE=c8d63e65363d1c30:/tmp/caps:3  vitaslop-desktop --game ... --headless ...
//!
//! # then ask as many questions as you like, each in about a second:
//! cargo run --release -p vitaslop-native --example capsule-replay -- /tmp/caps/v....capsule out.png
//! VITASLOP_GXP_VPROBE=1 cargo run ... -- /tmp/caps/v....capsule probe-v1.png
//! VITASLOP_GXP_SA=...   cargo run ... -- /tmp/caps/v....capsule subst.png
//! ```
//!
//! Every `VITASLOP_GXP_*` knob applies here exactly as it does live, because this runs the same
//! recompiler, the same linker and the same pipeline build - that is the whole point. What it
//! cannot reproduce is everything OUTSIDE the one draw, which it prints on every run rather
//! than leaving to be discovered; see [`vitaslop_runtime::capsule::CAVEATS`].
//!
//! It also prints the dominant colours of what it rendered. Reading a probe by eye is how a
//! whole session's attribution went to the wrong shader pair
//! (`vitaslop-keycolor-needs-noblend`), and a histogram is not fooled by a dark blue that
//! looks black.

use std::collections::HashMap;
use vitaslop_native::GeneralRenderer;
use vitaslop_runtime::capsule::{Capsule, CAVEATS};
use vitaslop_runtime::capture::Scene;

/// `VITASLOP_CAPSULE_DUMP_SA=1`: print this draw's uniform banks - `frag_sa` with the GUEST
/// ADDRESS it was read from, `vert_sa`, and every vertex memory window - as f32 registers.
///
/// The address is the whole point. A capsule answers "what did this draw get" and a live
/// `UNIFORM_WATCH` answers "what did the guest write", but neither alone answers
/// whether the guest's write landed in the buffer THIS draw bound - and `frag_sa_addr` is the
/// join between them. Printing the image beside it also settles the cheaper half offline: a
/// register reading zero among neighbours that are plainly correct is a value the guest itself
/// zeroed, not a plumbing fault.
fn dump_sa_requested() -> bool {
    std::env::var_os("VITASLOP_CAPSULE_DUMP_SA").is_some()
}

fn main() {
    // A subscriber, or every report the renderer makes about THIS draw is silent - including
    // the substitution warning ("this frame is NOT what the guest asked for") and any pair that
    // falls back. A replay tool that hides those is worse than no tool: it produces pictures
    // with no indication of what produced them. `VITASLOP_LOG` overrides the default.
    let _ = tracing_subscriber::fmt()
        .with_env_filter(tracing_subscriber::EnvFilter::new(
            std::env::var("VITASLOP_LOG").unwrap_or_else(|_| "warn".to_string()),
        ))
        .with_writer(std::io::stderr)
        .try_init();

    // `--tex <unit>=<r>,<g>,<b>,<a>` (0..255) or `--tex <unit>=<float>` rewrites every texel of
    // one bound sampler unit before the draw runs.
    //
    // THE FILL IS RAW GUEST BYTES, NOT A COLOUR. `BoundTexture::pixels` is the texture exactly
    // as it sits in guest memory, so what four bytes MEAN is decided by the unit's format. On
    // an 8888 colour map `255,255,255,255` is white; on a FLOAT format - a depth-only shadow
    // map is one - the same four bytes are `0xffffffff`, which is a NaN, and every comparison
    // against a NaN is false. That reads exactly like "this texture is not the cause": a whole
    // session concluded a shadow map was innocent from a fill that was silently a NaN. Hence
    // the single-value form, which writes a real f32, and the format printed on every fill.
    //
    // This is the texture half of what `VITASLOP_GXP_SA` is for uniforms: a sampler that reads
    // zero everywhere multiplies a surface to black just as surely as a zero uniform does, and
    // until now the only way to ask "is THIS texture the zero" was to reason about the shader.
    // It lives in the replay tool rather than in the renderer deliberately: it is an experiment
    // on a captured draw, and a knob that rewrites texels in a live run is a foot-gun with no
    // matching question.
    let mut tex_fill: Vec<(u32, [u8; 4])> = Vec::new();
    let mut positional: Vec<String> = Vec::new();
    let mut it = std::env::args().skip(1);
    while let Some(a) = it.next() {
        if a == "--tex" {
            let spec = it.next().unwrap_or_default();
            let Some((u, rgba)) = spec.split_once('=') else {
                eprintln!("--tex wants <unit>=<r>,<g>,<b>,<a>");
                std::process::exit(2);
            };
            if u.trim().parse::<u32>().is_err() {
                eprintln!("--tex wants <unit>=<r>,<g>,<b>,<a> or <unit>=<float>");
                std::process::exit(2);
            }
            // No comma: one FLOAT, written as its four little-endian bytes. This is the form a
            // depth or any other float-format target wants, and it is the one that stops a fill
            // meant as "fully lit" from landing as a NaN.
            let bytes = if rgba.contains(',') {
                let v: Vec<u8> = rgba.split(',').filter_map(|c| c.trim().parse().ok()).collect();
                if v.len() != 4 {
                    eprintln!("--tex wants <unit>=<r>,<g>,<b>,<a> with four 0..255 values");
                    std::process::exit(2);
                }
                [v[0], v[1], v[2], v[3]]
            } else {
                match rgba.trim().parse::<f32>() {
                    Ok(f) => f.to_le_bytes(),
                    Err(_) => {
                        eprintln!("--tex wants <unit>=<r>,<g>,<b>,<a> or <unit>=<float>");
                        std::process::exit(2);
                    }
                }
            };
            tex_fill.push((u.trim().parse().unwrap(), bytes));
        } else {
            positional.push(a);
        }
    }
    let [path, out] = match positional.as_slice() {
        [p, o] => [p.clone(), o.clone()],
        [p] => [p.clone(), "capsule.png".to_string()],
        _ => {
            eprintln!("usage: capsule-replay [--tex <unit>=<r>,<g>,<b>,<a>] <file.capsule> [out.png]");
            std::process::exit(2);
        }
    };

    let cap = match Capsule::load(std::path::Path::new(&path)) {
        Ok(c) => c,
        Err(e) => {
            eprintln!("capsule-replay: cannot read {path}: {e}");
            std::process::exit(1);
        }
    };

    eprintln!(
        "capsule {path}\n  {}\n  {} vertices bytes, {} indices, {} textures, {} vertex textures\n  \
         vert_sa {}B  frag_sa {}B ({} windows)\n  CAVEAT: {CAVEATS}",
        cap.note,
        cap.draw.vertices.len(),
        cap.draw.index_count,
        cap.draw.textures.len(),
        cap.draw.vertex_textures.len(),
        cap.draw.vert_sa.len(),
        cap.draw.frag_sa.len(),
        cap.draw.mem_windows.len(),
    );
    // The state that decides whether a BLACK draw is a bug at all. An additive or alpha-blended
    // draw contributes to what is already there, so black is a legitimate "adds nothing"; only
    // a REPLACE draw owns its pixels outright. A capsule renders against a bare clear, so this
    // line is the reader's only warning that the frame did not.
    let rs = &cap.draw.render_state;
    eprintln!(
        "  blend: mask {:#04x} colour func {} src {} dst {} | alpha func {} src {} dst {}
           depth: func {} write {} | cull {} | fragment program {}",
        cap.draw.blend.color_mask,
        cap.draw.blend.color_func,
        cap.draw.blend.color_src,
        cap.draw.blend.color_dst,
        cap.draw.blend.alpha_func,
        cap.draw.blend.alpha_src,
        cap.draw.blend.alpha_dst,
        rs.front_depth_func,
        rs.front_depth_write,
        rs.cull_mode,
        rs.front_fragment_program_enable,
    );

    // THE UNIFORM IMAGE AND THE ADDRESS IT CAME FROM. A probe can read what the shader saw and
    // a live `UNIFORM_WATCH` can read what the guest wrote, but neither alone answers
    // "did the guest's write land in the buffer THIS draw bound" - they are two halves of one
    // question, and the join between them is `frag_sa_addr`. Printing the image beside it makes
    // a capsule enough to ask half of it offline: a register that reads 0 here, in a block whose
    // neighbours are plainly correct, is a value the guest itself zeroed rather than a plumbing
    // fault. Both banks are printed as f32 because a default uniform buffer is floats; a probe
    // is still the way to read one as F16 pairs.
    let dump_bank = |name: &str, addr: u32, bytes: &[u8]| {
        if bytes.is_empty() {
            return;
        }
        let words: Vec<String> = bytes
            .chunks_exact(4)
            .enumerate()
            .map(|(i, w)| {
                let v = f32::from_le_bytes([w[0], w[1], w[2], w[3]]);
                format!("{i}:{v}")
            })
            .collect();
        eprintln!("  {name} @ {addr:#010x} ({} regs) = [{}]", words.len(), words.join(", "));
    };
    if dump_sa_requested() {
        dump_bank("frag_sa", cap.draw.frag_sa_addr, &cap.draw.frag_sa);
        dump_bank("vert_sa", 0, &cap.draw.vert_sa);
        // The vertex's 0xE8 memory windows are the OTHER path a material reaches a shader by -
        // a lit vertex program on this title reads its whole material through one - so a dump
        // that showed only the SA banks would answer half the question and look like all of it.
        for (i, (addr, bytes)) in cap.draw.mem_windows.iter().enumerate() {
            dump_bank(&format!("mem_window[{i}]"), *addr, bytes);
        }
    }

    // Name any GXP knob that is in force, so a picture from a probed or substituted run is
    // never mistaken for a picture of the real thing - the same guarantee the live path's
    // substitution report carries.
    let knobs: Vec<String> = std::env::vars()
        .filter(|(k, _)| (k.starts_with("VITASLOP_GXP") || k.starts_with("VITASLOP_GXM")) && k != "VITASLOP_GXP_LIVE")
        .map(|(k, v)| format!("{k}={v}"))
        .collect();
    if !knobs.is_empty() {
        eprintln!("  KNOBS IN FORCE: {} - this is NOT a picture of the real thing", knobs.join(" "));
    }

    // A capsule IS the recompiled path - the payload it carries is the guest's own shaders -
    // so the knob that enables that path is implied rather than left to be remembered. Without
    // it the draw silently does not rasterise at all, which reads exactly like "the geometry is
    // off-screen" and cost the first four replays of this tool.
    if std::env::var_os("VITASLOP_GXP_LIVE").is_none() {
        // SAFETY: single-threaded, before any renderer thread exists.
        unsafe { std::env::set_var("VITASLOP_GXP_LIVE", "1") };
    }

    let Some(mut gpu) = GeneralRenderer::new() else {
        eprintln!("capsule-replay: no GPU adapter available");
        std::process::exit(1);
    };

    let mut draw = cap.draw.clone();
    if !tex_fill.is_empty() {
        let mut texs: Vec<_> = draw.textures.iter().cloned().collect();
        for (unit, rgba) in &tex_fill {
            match texs.iter_mut().find(|t| t.unit == *unit) {
                Some(t) => {
                    let n = t.pixels.len() / 4;
                    t.pixels = std::sync::Arc::from(
                        rgba.iter().copied().cycle().take(n * 4).collect::<Vec<u8>>(),
                    );
                    // The FORMAT is on this line because the bytes only mean something next
                    // to it, and both readings are printed so a fill can never be mistaken for
                    // a colour it is not in this texture's format.
                    eprintln!(
                        "  --tex: unit {unit} ({}x{}) base_format {:#04x} filled with bytes                          {rgba:?} - reads as rgba8({},{},{},{}) or f32 {} - NOT the guest's texels",
                        t.width,
                        t.height,
                        t.base_format,
                        rgba[0],
                        rgba[1],
                        rgba[2],
                        rgba[3],
                        f32::from_le_bytes(*rgba)
                    );
                }
                // Refused loudly. Filling a unit the draw does not bind changes nothing, and a
                // silent no-op here reads as "that texture is not the cause".
                None => {
                    eprintln!(
                        "  --tex: this draw binds NO sampler unit {unit} (it binds {:?}) - refusing",
                        texs.iter().map(|t| t.unit).collect::<Vec<_>>()
                    );
                    std::process::exit(2);
                }
            }
        }
        draw.textures = std::sync::Arc::from(texs);
    }

    let scene = Scene {
        precompile: std::sync::Arc::new(Vec::new()),
        color: None,
        depth: None,
        multisample: 0,
        draws: vec![draw],
    };

    let fb = gpu.render_scene(&scene, cap.width, cap.height, cap.clear);
    match std::fs::write(&out, fb.to_png()) {
        Ok(()) => eprintln!("  -> {out} ({}x{})", cap.width, cap.height),
        Err(e) => {
            eprintln!("capsule-replay: cannot write {out}: {e}");
            std::process::exit(1);
        }
    }

    // The histogram, on the same grid `pxtop.py` uses. Printed here so the tool answers the
    // question without a second program: "is this black, or is it a dark colour".
    let px = &fb.rgba;
    let (w, h) = (fb.width as usize, fb.height as usize);
    let mut counts: HashMap<[u8; 3], usize> = HashMap::new();
    let mut total = 0usize;
    for y in (0..h).step_by(4) {
        for x in (0..w).step_by(4) {
            let o = (y * w + x) * 4;
            if o + 3 <= px.len() {
                *counts.entry([px[o], px[o + 1], px[o + 2]]).or_default() += 1;
                total += 1;
            }
        }
    }
    let mut top: Vec<_> = counts.into_iter().collect();
    top.sort_by_key(|&(_, n)| std::cmp::Reverse(n));
    eprintln!("  dominant colours ({total} sampled):");
    for (c, n) in top.iter().take(8) {
        eprintln!("    {:5.1}%  rgb({}, {}, {})", 100.0 * *n as f64 / total as f64, c[0], c[1], c[2]);
    }
    // The CLEAR is in that histogram and is not part of the draw. Saying so costs one line and
    // stops "40% of the frame is this colour" from being read as a finding about the shader.
    eprintln!(
        "    (the clear is rgb({}, {}, {}) - pixels the draw did not cover)",
        cap.clear[0], cap.clear[1], cap.clear[2]
    );
}
