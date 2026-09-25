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
    let fc = vitaslop_runtime::capsule::read_frame(&mut &bytes[..]).unwrap_or_else(|e| {
        eprintln!("frame-replay: {path}: {e}");
        std::process::exit(1);
    });
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
        eprintln!("  scene {i}: {} draw(s) into {target}{depth}", s.draws.len());
        // `--list`: every draw's bound textures, as the capture decoded them.
        if list {
            for (di, d) in s.draws.iter().enumerate() {
                for t in d.textures.iter() {
                    eprintln!(
                        "    draw {di} unit {} tex {:#x} {}x{} type {} fmt {:#x} stride {} faces {}",
                        t.unit, t.data_addr, t.width, t.height, t.tex_type, t.base_format, t.stride, t.faces
                    );
                }
            }
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
    // `--extract <vprog content hash> <dir>`: write every draw whose VERTEX program hashes to that
    // value as its own draw capsule, for `capsule-replay` (named uniforms, per-draw probes) - the
    // per-draw instrument, without another title run to capture the draw on its own.
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
    // The first positional argument after the frame, skipping every option's VALUE.
    let out = args.iter().enumerate().skip(1).find_map(|(i, a)| {
        let is_value = matches!(args[i - 1].as_str(), "--before" | "--extract");
        (!a.starts_with("--") && !is_value).then_some(a)
    });
    if let Some(out) = out {
        std::fs::write(out, fb.to_png()).unwrap_or_else(|e| {
            eprintln!("frame-replay: cannot write {out}: {e}");
            std::process::exit(1);
        });
        eprintln!("  -> {out}");
    }
}
