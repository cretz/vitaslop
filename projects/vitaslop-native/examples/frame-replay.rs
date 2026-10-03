//! Re-render a captured FRAME ([`vitaslop_runtime::capsule::write_frame`]) offline - every scene
//! of it, through the same `GeneralRenderer` a headless shot uses - in about a second, instead
//! of replaying the title to that frame.
//!
//! ```text
//! # capture, once, during a headless run (every SHOT frame is written):
//! VITASLOP_FRAME_CAPSULE=<dir> vitaslop --game ... --headless <shots>
//!
//! # then, after every renderer change, rebuild this and:
//! cargo run --release -p vitaslop-native --example frame-replay -- <dir>/f003600.frame out.png
//! VITASLOP_GPU_CHAIN_DIR=<dir> ...   also dumps every target the frame rendered
//! ```
//!
//! Every `VITASLOP_*` renderer knob applies exactly as it does live. What it cannot reproduce is
//! anything an EARLIER frame left behind - a target rendered on a previous frame and sampled on
//! this one, the guest bytes a write-back would have refreshed - which it prints on every run.

use vitaslop_native::GeneralRenderer;

fn main() {
    // The recompiled-shader path, as every headless run uses it (`anyrun` sets it): without it
    // the frame renders through the fixed-function approximation and is not the frame that was
    // captured. Set explicitly to 0 to see that approximation.
    if std::env::var_os("VITASLOP_GXP_LIVE").is_none() {
        // SAFETY: single-threaded, before anything reads the environment.
        unsafe { std::env::set_var("VITASLOP_GXP_LIVE", "1") };
    }
    // The renderer's own diagnostics are `tracing` events (`VITASLOP_GXP_KEYCOLOR`'s key -> colour
    // table among them); with no subscriber they do not exist, and a key-coloured frame with no
    // table cannot be read back. Filtered by `RUST_LOG`, warn by default.
    let _ = tracing_subscriber::fmt()
        .with_env_filter(tracing_subscriber::EnvFilter::new(
            std::env::var("RUST_LOG").unwrap_or_else(|_| "warn".into()),
        ))
        .with_writer(std::io::stderr)
        .try_init();
    let args: Vec<String> = std::env::args().skip(1).collect();
    let Some(path) = args.first() else {
        eprintln!("usage: frame-replay <file.frame> [out.png]");
        std::process::exit(2);
    };
    let bytes = std::fs::read(path).unwrap_or_else(|e| {
        eprintln!("frame-replay: cannot read {path}: {e}");
        std::process::exit(1);
    });
    let mut fc = vitaslop_runtime::capsule::read_frame(&mut &bytes[..]).unwrap_or_else(|e| {
        eprintln!("frame-replay: {path}: {e}");
        std::process::exit(1);
    });
    // `--truncate <scene> <n>`: keep only the first `n` draws of that scene - bisect which draw
    // puts a defect on screen (render at n, look at the pixel, halve). Later scenes still run,
    // so a composite that samples the truncated target shows the truncated picture.
    if let Some(at) = args.iter().position(|a| a == "--truncate") {
        let parse = |i: usize| args.get(at + i).and_then(|v| v.parse::<usize>().ok());
        let (Some(si), Some(n)) = (parse(1), parse(2)) else {
            eprintln!("frame-replay: --truncate <scene> <n>");
            std::process::exit(2);
        };
        let Some(scene) = fc.scenes.get_mut(si) else {
            eprintln!("frame-replay: --truncate: no scene {si}");
            std::process::exit(2);
        };
        scene.draws.truncate(n);
        eprintln!("  scene {si} truncated to its first {} draw(s)", scene.draws.len());
    }
    eprintln!(
        "frame {} - {} scene(s), {} draw(s), {}x{}",
        fc.frame,
        fc.scenes.len(),
        fc.scenes.iter().map(|s| s.draws.len()).sum::<usize>(),
        fc.width,
        fc.height
    );
    let list = args.iter().any(|a| a == "--list");
    for (i, s) in fc.scenes.iter().enumerate() {
        let target = s.color.map_or("no colour".to_string(), |c| {
            format!("{:#x} {}x{} pitch {} fmt {:#x}", c.data_addr, c.width, c.height, c.stride_pixels, c.format)
        });
        let depth = s.depth.map_or(String::new(), |d| {
            format!(", depth {:#x} (stencil {:#x}, zls {:#x})", d.depth_addr, d.stencil_addr, d.zls_control)
        });
        eprintln!("  scene {i}: {} draw(s) into {target}{depth} msaa {}", s.draws.len(), s.multisample);
        // `--list`: every draw's bound textures, as the capture decoded them.
        if list {
            for (di, d) in s.draws.iter().enumerate() {
                let h = |b: &[u8]| vitaslop_gxp_shader::Program::parse(b).map(|p| p.hash).unwrap_or(0);
                let rs = &d.render_state;
                eprintln!(
                    "    draw {di} vprog {:016x} fprog {:016x} depth func {} write {} cull {}",
                    h(&d.vprog), h(&d.fprog), rs.front_depth_func, rs.front_depth_write, rs.cull_mode
                );
                for t in d.textures.iter() {
                    eprintln!(
                        "    draw {di} unit {} tex {:#x} {}x{} type {} fmt {:#x} stride {} faces {} min {} mag {} mip {} mips {} addr {}/{}",
                        t.unit, t.data_addr, t.width, t.height, t.tex_type, t.base_format, t.stride, t.faces,
                        t.min_filter, t.mag_filter, t.mip_filter, t.levels, t.u_addr_mode, t.v_addr_mode
                    );
                }
            }
        }
    }
    // `--prog <scene> <draw> <dir>`: write that draw's vertex and fragment program blobs to
    // `<dir>/<hash>.v.gxp` / `<dir>/<hash>.f.gxp`, for the shader crate's disassembly and
    // linked-WGSL tools - the one step between "this draw is wrong" and reading its program.
    // `--progs <dir>`: every DISTINCT program of the whole frame, the same way - the frame's own
    // shader corpus, for a census of one instruction shape across everything it draws with.
    if let Some(dir) = args.iter().position(|a| a == "--progs").and_then(|at| args.get(at + 1)) {
        std::fs::create_dir_all(dir).expect("create --progs dir");
        let mut seen = std::collections::HashSet::new();
        for d in fc.scenes.iter().flat_map(|s| s.draws.iter()) {
            for (blob, kind) in [(&d.vprog, "v"), (&d.fprog, "f")] {
                let hash = vitaslop_gxp_shader::Program::parse(blob).map(|p| p.hash).unwrap_or(0);
                if seen.insert((hash, kind)) {
                    std::fs::write(std::path::Path::new(dir).join(format!("{hash:016x}.{kind}.gxp")), &blob[..])
                        .expect("write program blob");
                }
            }
        }
        eprintln!("  wrote {} distinct programs to {dir}", seen.len());
    }
    if let Some(at) = args.iter().position(|a| a == "--prog") {
        let (Some(si), Some(di), Some(dir)) = (
            args.get(at + 1).and_then(|s| s.parse::<usize>().ok()),
            args.get(at + 2).and_then(|s| s.parse::<usize>().ok()),
            args.get(at + 3),
        ) else {
            eprintln!("frame-replay: --prog <scene> <draw> <dir>");
            std::process::exit(2);
        };
        let Some(d) = fc.scenes.get(si).and_then(|s| s.draws.get(di)) else {
            eprintln!("frame-replay: --prog: no draw {di} in scene {si}");
            std::process::exit(2);
        };
        std::fs::create_dir_all(dir).expect("create --prog dir");
        for (blob, kind) in [(&d.vprog, "v"), (&d.fprog, "f")] {
            let hash = vitaslop_gxp_shader::Program::parse(blob).map(|p| p.hash).unwrap_or(0);
            let path = std::path::Path::new(dir).join(format!("{hash:016x}.{kind}.gxp"));
            std::fs::write(&path, &blob[..]).expect("write program blob");
            eprintln!("  wrote {}", path.display());
        }
    }
    // `--tex <addr> <dir>`: every DISTINCT binding of the texture at that guest address, as the
    // capture decoded it - format, swizzle, a hash of its snapshotted bytes, and the decoded
    // RGBA8 written to `<dir>/tex-<addr>-<n>.png`. Two frames' dumps of one address say whether
    // a picture changed because the BYTES did or because the DECODE did.
    if let Some(at) = args.iter().position(|a| a == "--tex") {
        let (Some(want), Some(dir)) = (args.get(at + 1), args.get(at + 2)) else {
            eprintln!("frame-replay: --tex <hex addr> <dir>");
            std::process::exit(2);
        };
        let want = u32::from_str_radix(want.trim_start_matches("0x"), 16).unwrap_or_else(|_| {
            eprintln!("frame-replay: --tex wants a hex address, got {want}");
            std::process::exit(2);
        });
        std::fs::create_dir_all(dir).expect("create the texture directory");
        let mut seen: Vec<u64> = Vec::new();
        for s in fc.scenes.iter() {
            for t in s.draws.iter().flat_map(|d| d.textures.iter()).filter(|t| t.data_addr == want) {
                use std::hash::{Hash, Hasher};
                let mut h = std::collections::hash_map::DefaultHasher::new();
                (t.base_format, t.swizzle, t.width, t.height, t.stride, &t.pixels[..]).hash(&mut h);
                let id = h.finish();
                if seen.contains(&id) {
                    continue;
                }
                seen.push(id);
                let (w, hgt, rgba) = vitaslop_runtime::render::decode_texture_rgba8(t);
                let out = format!("{dir}/tex-{want:08x}-{}.png", seen.len() - 1);
                let fb = vitaslop_runtime::render::Framebuffer { width: w, height: hgt, rgba };
                std::fs::write(&out, fb.to_png()).expect("write the texture png");
                eprintln!(
                    "  tex {want:#x} {}x{} fmt {:#x} swizzle {:#x} type {} bytes {} hash {id:016x} -> {out}",
                    t.width, t.height, t.base_format, t.swizzle, t.tex_type, t.pixels.len()
                );
            }
        }
        return;
    }
    // `--ir <scene> <draw> [from] [to]`: that draw's VERTEX program decoded - index, raw word,
    // op, destination and sources - the listing the NaN-site report's `#N` indexes.
    if let Some(at) = args.iter().position(|a| a == "--ir") {
        let si: usize = args.get(at + 1).and_then(|v| v.parse().ok()).unwrap_or(0);
        let di: usize = args.get(at + 2).and_then(|v| v.parse().ok()).unwrap_or(0);
        let from: usize = args.get(at + 3).and_then(|v| v.parse().ok()).unwrap_or(0);
        let to: usize = args.get(at + 4).and_then(|v| v.parse().ok()).unwrap_or(usize::MAX);
        let Some(d) = fc.scenes.get(si).and_then(|s| s.draws.get(di)) else {
            eprintln!("frame-replay: no scene {si} draw {di}");
            std::process::exit(2);
        };
        let prog = vitaslop_gxp_shader::Program::parse(&d.vprog).expect("parse the vertex program");
        let sh = vitaslop_gxp_shader::usse::decode_shader(&prog);
        for (i, ins) in sh.instrs.iter().enumerate().skip(from).take(to.saturating_sub(from).saturating_add(1)) {
            println!("#{i} {:016x} g{:#04x} {:?} dest {:?} mask {:?} srcs {:?}{}", ins.raw, ins.group, ins.op, ins.dest, ins.write_mask, ins.srcs, ins.blocked.map_or(String::new(), |b| format!(" BLOCKED {b}")));
        }
        return;
    }
    // `--sa <scene> <draw>`: print that draw's captured SA banks (vertex and fragment) as floats,
    // and its uniform windows - what a recompiled shader reads for its uniforms.
    if let Some(at) = args.iter().position(|a| a == "--sa") {
        let si: usize = args.get(at + 1).and_then(|v| v.parse().ok()).unwrap_or(0);
        let di: usize = args.get(at + 2).and_then(|v| v.parse().ok()).unwrap_or(0);
        let Some(d) = fc.scenes.get(si).and_then(|s| s.draws.get(di)) else {
            eprintln!("frame-replay: no scene {si} draw {di}");
            std::process::exit(2);
        };
        let floats = |b: &[u8]| -> String {
            b.chunks_exact(4)
                .map(|c| format!("{:.4}", f32::from_le_bytes([c[0], c[1], c[2], c[3]])))
                .collect::<Vec<_>>()
                .join(" ")
        };
        let h = |b: &[u8]| vitaslop_gxp_shader::Program::parse(b).map(|p| p.hash).unwrap_or(0);
        println!("scene {si} draw {di}: vprog {:016x} fprog {:016x}", h(&d.vprog), h(&d.fprog));
        println!("scene {si} draw {di}: vert_sa {} B: {}", d.vert_sa.len(), floats(&d.vert_sa));
        println!("  frag_sa {} B (from {:#x}): {}", d.frag_sa.len(), d.frag_sa_addr, floats(&d.frag_sa));
        println!(
            "  frag_sa as f16: {}",
            d.frag_sa.chunks_exact(2).map(|c| format!("{:.4}", half_to_f32(u16::from_le_bytes([c[0], c[1]])))).collect::<Vec<_>>().join(" ")
        );
        for (a, b) in d.mem_windows.iter() {
            println!("  vwindow {a:#x} {} B: {}", b.len(), floats(b));
        }
        println!("  uniforms (lanes 0..16): {:?}", &d.uniforms[..d.uniforms.len().min(16)]);
        for t in d.vertex_textures.iter() {
            println!("  VERTEX texture unit {} {}x{} fmt {:#x} data {:#x} {} B", t.unit, t.width, t.height, t.base_format, t.data_addr, t.pixels.len());
            let px = &t.pixels;
            let h = |i: usize| half_to_f32(u16::from_le_bytes([px[i], px[i + 1]]));
            let f = |i: usize| f32::from_le_bytes([px[i], px[i + 1], px[i + 2], px[i + 3]]);
            for texel in [0usize, 1, 2, 3, 4, 5, 2048, 2049] {
                let o = texel * 8;
                if o + 8 <= px.len() {
                    println!(
                        "    texel {texel}: as f16x4 [{:.4} {:.4} {:.4} {:.4}] as f32x2 [{:.4} {:.4}]",
                        h(o), h(o + 2), h(o + 4), h(o + 6), f(o), f(o + 4)
                    );
                }
            }
        }
        for t in d.textures.iter() {
            println!("  fragment texture unit {} {}x{} fmt {:#x} data {:#x} {} B", t.unit, t.width, t.height, t.base_format, t.data_addr, t.pixels.len());
        }
        for (stage, blob) in [("vertex", &d.vprog), ("fragment", &d.fprog)] {
            if let Ok(p) = vitaslop_gxp_shader::Program::parse(blob) {
                for q in &p.parameters {
                    println!(
                        "  {stage} param {:?} {:?} x{} [{}] at sa {}",
                        q.name, q.category, q.component_count, q.array_size, q.resource_index
                    );
                }
            }
        }
        return;
    }
    // `--extract <vprog content hash> <dir>`: write every draw whose VERTEX program hashes to that
    // value as its own draw capsule, for `capsule-replay` (named uniforms, per-draw probes) - the
    // per-draw instrument, without another title run to capture the draw on its own.
    // `--params <vprog hash>`: that vertex program's parameter table, its +0x78 buffer bindings
    // and the memory windows the capture resolves for it - what sizes a window it reads past.
    if let Some(at) = args.iter().position(|a| a == "--params") {
        let want = args.get(at + 1).and_then(|w| u64::from_str_radix(w.trim_start_matches("0x"), 16).ok());
        let Some(want) = want else {
            eprintln!("frame-replay: --params <vprog hash>");
            std::process::exit(2);
        };
        for s in &fc.scenes {
            for d in &s.draws {
                let Ok(p) = vitaslop_gxp_shader::Program::parse(&d.vprog) else { continue };
                if p.hash != want {
                    continue;
                }
                eprintln!("vprog {want:016x}: default uniform regs {}", p.default_uniform_regs);
                for q in &p.parameters {
                    eprintln!(
                        "  param {:<32} {:?} type {:?} comps {} array {} res {} semantic {}/{} container {}",
                        q.name, q.category, q.ptype, q.component_count, q.array_size, q.resource_index, q.semantic,
                        q.semantic_index, q.container_index
                    );
                }
                for b in &p.uniform_buffer_bindings {
                    eprintln!("  +0x78 binding {b:?}");
                }
                for w in vitaslop_gxp_shader::mem_windows_for_vertex_blob(&d.vprog) {
                    eprintln!("  window {w:?}");
                }
                for (a, b) in &d.mem_windows {
                    eprintln!("  captured window at {a:#x}, {} bytes", b.len());
                }
                eprintln!("  vertex stride {} bytes, {} bytes captured", d.vertex_stride, d.vertices.len());
                for at in d.attributes.iter() {
                    eprintln!("  attr {at:?}");
                    let stride = d.vertex_stride.max(1) as usize;
                    let raw: Vec<String> = (0..4)
                        .filter_map(|v| {
                            let o = v * stride + at.offset as usize;
                            d.vertices.get(o..o + 8).map(|b| format!("{b:02x?}"))
                        })
                        .collect();
                    eprintln!("    first 4 vertices, 8 bytes at its offset: {}", raw.join(" "));
                }
                return;
            }
        }
        eprintln!("frame-replay: no draw binds vprog {want:016x}");
        return;
    }
    if let Some(at) = args.iter().position(|a| a == "--extract") {
        let (Some(want), Some(dir)) = (args.get(at + 1), args.get(at + 2)) else {
            eprintln!("frame-replay: --extract <vprog hash> <dir>");
            std::process::exit(2);
        };
        let want = u64::from_str_radix(want.trim_start_matches("0x"), 16).unwrap_or_else(|_| {
            eprintln!("frame-replay: --extract wants a hex vertex-program hash, got {want}");
            std::process::exit(2);
        });
        std::fs::create_dir_all(dir).expect("create the extract directory");
        let mut n = 0;
        for (si, s) in fc.scenes.iter().enumerate() {
            for (di, d) in s.draws.iter().enumerate() {
                let hash = vitaslop_gxp_shader::Program::parse(&d.vprog).map(|p| p.hash).unwrap_or(0);
                if hash != want {
                    continue;
                }
                let (width, height) = s.color.map_or((fc.width, fc.height), |c| (c.width, c.height));
                let cap = vitaslop_runtime::capsule::Capsule {
                    draw: d.clone(),
                    width,
                    height,
                    clear: fc.clear,
                    key: 0,
                    frame: fc.frame,
                    draw_index: di as u32,
                    note: format!("extracted from frame {} scene {si} draw {di}", fc.frame),
                };
                let out = format!("{dir}/s{si}-d{di}.capsule");
                let mut f = std::fs::File::create(&out).expect("create the capsule file");
                cap.write(&mut f).expect("write the capsule");
                eprintln!("  extracted scene {si} draw {di} -> {out}");
                n += 1;
            }
        }
        eprintln!("  {n} draw(s) extracted");
        return;
    }
    // `--slim <out.frame>`: rewrite this frame as a SLIM frame (every distinct large byte field
    // once - see `capsule::write_frame_slim`), the form a PHONE can be sent. Verified before it
    // says so: the slim file read back and re-written in the old format must hash exactly like
    // the input, so a conversion that lost a byte is reported, never shipped.
    if let Some(at) = args.iter().position(|a| a == "--slim") {
        let Some(out) = args.get(at + 1) else {
            eprintln!("frame-replay: --slim <out.frame>");
            std::process::exit(2);
        };
        struct Hash(std::collections::hash_map::DefaultHasher, u64);
        impl std::io::Write for Hash {
            fn write(&mut self, b: &[u8]) -> std::io::Result<usize> {
                use std::hash::Hasher;
                self.0.write(b);
                self.1 += b.len() as u64;
                Ok(b.len())
            }
            fn flush(&mut self) -> std::io::Result<()> {
                Ok(())
            }
        }
        let hash_of = |scenes: &[vitaslop_runtime::capture::Scene], w, h, c, f| {
            use std::hash::Hasher;
            let mut hw = Hash(Default::default(), 0);
            vitaslop_runtime::capsule::write_frame(&mut hw, scenes, w, h, c, f).expect("hash a frame");
            (hw.0.finish(), hw.1)
        };
        let mut slim = Vec::new();
        vitaslop_runtime::capsule::write_frame_slim(&mut slim, &fc.scenes, fc.width, fc.height, fc.clear, fc.frame)
            .expect("write the slim frame");
        let back = vitaslop_runtime::capsule::read_frame(&mut &slim[..]).expect("read the slim frame back");
        let want = hash_of(&fc.scenes, fc.width, fc.height, fc.clear, fc.frame);
        let got = hash_of(&back.scenes, back.width, back.height, back.clear, back.frame);
        if want != got {
            eprintln!("frame-replay: --slim VERIFY FAILED: the slim frame reads back as a different frame ({want:?} vs {got:?}) - not written");
            std::process::exit(1);
        }
        std::fs::write(out, &slim).expect("write the slim frame file");
        eprintln!(
            "  slim: {} -> {} bytes ({:.1}x smaller), verified identical on read-back -> {out}",
            bytes.len(),
            slim.len(),
            bytes.len() as f64 / slim.len() as f64
        );
        return;
    }
    // `--scene-only <i> <out.frame>`: a SLIM frame holding scene `i` alone, sized to its own
    // target - so that target becomes the display a replay (or a phone frame-replay job's PNG)
    // hands back. For judging one offscreen pass across devices from identical inputs.
    // `--prepend-scene <other.frame> <i> <out.frame>`: a SLIM frame of scene `i` of another
    // capsule followed by every scene of this one - a target an earlier frame painted, carried
    // into this frame's replay on devices that cannot take `--before` (a phone job).
    if let Some(at) = args.iter().position(|a| a == "--prepend-scene") {
        let (Some(p), Some(i), Some(out)) =
            (args.get(at + 1), args.get(at + 2).and_then(|v| v.parse::<usize>().ok()), args.get(at + 3))
        else {
            eprintln!("frame-replay: --prepend-scene <other.frame> <scene index> <out.frame>");
            std::process::exit(2);
        };
        let b = std::fs::read(p).expect("read the other frame");
        let other = vitaslop_runtime::capsule::read_frame(&mut &b[..]).expect("parse the other frame");
        let Some(s) = other.scenes.get(i) else {
            eprintln!("frame-replay: {p} has no scene {i}");
            std::process::exit(2);
        };
        let mut scenes = vec![s.clone()];
        scenes.extend(fc.scenes.iter().cloned());
        let mut slim = Vec::new();
        vitaslop_runtime::capsule::write_frame_slim(&mut slim, &scenes, fc.width, fc.height, fc.clear, fc.frame)
            .expect("write the combined frame");
        std::fs::write(out, &slim).expect("write the combined frame file");
        eprintln!("  scene {i} of {p} + {} scenes -> {out}", fc.scenes.len());
        return;
    }
    // `--keep-vprog <hash> <out.frame>`: a SLIM frame keeping scene 0 whole (an offscreen pass
    // the kept draws may sample) and, in every later scene, only the draws whose VERTEX program
    // hashes to `hash` - one shader pair's geometry alone, to judge across devices.
    if let Some(at) = args.iter().position(|a| a == "--keep-vprog") {
        let (Some(want), Some(out)) = (args.get(at + 1), args.get(at + 2)) else {
            eprintln!("frame-replay: --keep-vprog <hex vprog hash> <out.frame>");
            std::process::exit(2);
        };
        // `!<hash>` DROPS that program's draws instead (from every scene) - which program's
        // removal makes an artifact vanish names the program that draws it.
        let drop = want.starts_with('!');
        let want = u64::from_str_radix(want.trim_start_matches('!').trim_start_matches("0x"), 16).expect("a hex vertex-program hash");
        let mut scenes = fc.scenes.clone();
        let mut kept = 0usize;
        for s in scenes.iter_mut().skip(usize::from(!drop)) {
            s.draws.retain(|d| (vitaslop_gxp_shader::Program::parse(&d.vprog).map(|p| p.hash).unwrap_or(0) == want) != drop);
            kept += s.draws.len();
        }
        let mut slim = Vec::new();
        vitaslop_runtime::capsule::write_frame_slim(&mut slim, &scenes, fc.width, fc.height, fc.clear, fc.frame)
            .expect("write the filtered frame");
        std::fs::write(out, &slim).expect("write the filtered frame file");
        eprintln!("  kept {kept} draw(s) of vprog {want:016x} after scene 0 -> {out}");
        return;
    }
    if let Some(at) = args.iter().position(|a| a == "--scene-only") {
        let (Some(i), Some(out)) = (args.get(at + 1).and_then(|v| v.parse::<usize>().ok()), args.get(at + 2)) else {
            eprintln!("frame-replay: --scene-only <scene index> <out.frame>");
            std::process::exit(2);
        };
        let Some(s) = fc.scenes.get(i) else {
            eprintln!("frame-replay: no scene {i} (the frame has {})", fc.scenes.len());
            std::process::exit(2);
        };
        let (w, h) = s.color.as_ref().map_or((fc.width, fc.height), |c| (c.width, c.height));
        let mut slim = Vec::new();
        vitaslop_runtime::capsule::write_frame_slim(&mut slim, std::slice::from_ref(s), w, h, fc.clear, fc.frame)
            .expect("write the scene-only frame");
        std::fs::write(out, &slim).expect("write the scene-only frame file");
        eprintln!("  scene {i} alone ({w}x{h}, {} draws) -> {out}", s.draws.len());
        return;
    }
    eprintln!(
        "  CAVEAT: one frame only - a target an EARLIER frame rendered is not in it, so a sampler \
         reaching for one reads the guest bytes captured with the draw."
    );
    let Some(mut gpu) = GeneralRenderer::new() else {
        eprintln!("frame-replay: no GPU adapter available");
        std::process::exit(1);
    };
    // `--before <file.frame>` (repeatable, in order): render these frames first, through the
    // SAME renderer, so the targets they draw persist into the main frame exactly as they do
    // live - a light probe or lookup map a title renders once at load and samples for the rest
    // of the level is otherwise read from guest bytes the GPU never wrote. Capsules from two
    // runs of one recipe line up: the emulator is deterministic.
    for (i, a) in args.iter().enumerate() {
        if a != "--before" {
            continue;
        }
        let Some(p) = args.get(i + 1) else { continue };
        let b = std::fs::read(p).unwrap_or_else(|e| {
            eprintln!("frame-replay: cannot read {p}: {e}");
            std::process::exit(1);
        });
        let pre = vitaslop_runtime::capsule::read_frame(&mut &b[..]).unwrap_or_else(|e| {
            eprintln!("frame-replay: {p}: {e}");
            std::process::exit(1);
        });
        let _ = gpu.render_frame(&pre.scenes, pre.width, pre.height, pre.clear);
        eprintln!("  rendered frame {} first ({p})", pre.frame);
    }
    let t = std::time::Instant::now();
    let fb = gpu.render_frame(&fc.scenes, fc.width, fc.height, fc.clear);
    eprintln!("  rendered in {:.0} ms", t.elapsed().as_secs_f64() * 1000.0);
    // `--gpu-time <n>`: render the frame n MORE times (the first render above warmed every
    // pipeline) and print the GPU TIME report over those n - with `VITASLOP_GPU_TIME_ALL=1` and
    // `VITASLOP_GXP_PASS_SPLIT_EVERY=<k>`, the GPU cost of each k-draw slice of every pass.
    if let Some(n) = args.iter().position(|a| a == "--gpu-time").and_then(|i| args.get(i + 1)).and_then(|v| v.parse::<u32>().ok()) {
        let _ = gpu.take_gpu_time_report();
        for _ in 0..n {
            let _ = gpu.render_frame(&fc.scenes, fc.width, fc.height, fc.clear);
        }
        eprintln!("  gpu time: {}", gpu.take_gpu_time_report());
    }
    // The first positional argument after the frame, skipping every option's operands. Only
    // the options that reach this render are listed: `--tex`, `--ir`, `--sa`, `--slim`,
    // `--prepend-scene`, `--keep-vprog` and `--scene-only` return before it.
    let operands = |opt: &str| match opt {
        "--before" | "--extract" | "--gpu-time" | "--progs" | "--params" => 1,
        "--truncate" => 2,
        "--prog" => 3,
        _ => 0,
    };
    let mut skip = 0;
    let out = args.iter().skip(1).find(|a| {
        if skip > 0 {
            skip -= 1;
            return false;
        }
        skip = operands(a);
        !a.starts_with("--")
    });
    if let Some(out) = out {
        std::fs::write(out, fb.to_png()).unwrap_or_else(|e| {
            eprintln!("frame-replay: cannot write {out}: {e}");
            std::process::exit(1);
        });
        eprintln!("  -> {out}");
    }
}

fn half_to_f32(h: u16) -> f32 {
    let s = ((h >> 15) & 1) as u32;
    let e = ((h >> 10) & 0x1f) as u32;
    let m = (h & 0x3ff) as u32;
    let bits = if e == 0 {
        if m == 0 {
            s << 31
        } else {
            let mut e = 127 - 15 + 1;
            let mut m = m;
            while m & 0x400 == 0 {
                m <<= 1;
                e -= 1;
            }
            (s << 31) | (e << 23) | ((m & 0x3ff) << 13)
        }
    } else if e == 31 {
        (s << 31) | (0xff << 23) | (m << 13)
    } else {
        (s << 31) | ((e + 127 - 15) << 23) | (m << 13)
    };
    f32::from_bits(bits)
}
