//! Headless boot probe for a real, multi-module retail Vita title.
//!
//! This drives one end-to-end run of a linked commercial game through the whole
//! pipeline: PFS/self2elf decrypt -> multi-module link -> lenient transpile ->
//! wasm instantiate -> run the shared-library constructors and the executable entry
//! under a preemptive scheduler -> software-rasterize whatever GXM scenes were
//! captured to PNGs. It is a *probe*, not a pass/fail conformance test: it always
//! dumps a reviewable report (how far the boot got, which NIDs are still missing,
//! how many frames were captured) plus one PNG per captured scene, so progress
//! toward "it renders" is visible and the exact remaining host-call surface is
//! enumerated.
//!
//! The probe is content-free and game-agnostic: it embeds no title, no game bytes,
//! and no per-game expectations - it runs whatever decrypted app directory the
//! `VITASLOP_GAME_DIR` fixture points at. It is `#[ignore]`d and skips when that env
//! var is absent, so `cargo test --workspace` stays green for everyone.
//!
//! Run:
//!   VITASLOP_GAME_DIR=/path/to/decrypted/app/<TITLE_ID> \
//!   VITASLOP_SHOT_DIR=/some/dir/outside/the/repo \
//!   cargo test -p vitaslop-native --test retail_boot_probe -- --ignored --nocapture
//!
//! All file artifacts (frame PNGs, the call trace, an optional image dump) are
//! written ONLY when `VITASLOP_SHOT_DIR` is set, and only there - the probe never
//! creates a directory or writes a file inside the repo. Without it, the probe just
//! prints its report to stderr.
//!
//! Optional diagnostics (all off by default, zero cost when unset - the reusable
//! telemetry for bringing up a new title):
//! - `VITASLOP_TRANSPILE_REPORT` lists the functions the transpiler could not build
//!   (decode vs lift gaps, with addresses).
//! - `VITASLOP_DUMP_STUBS` lists the trapping-stub wasm indices.
//! - `VITASLOP_DUMP_IMAGE` writes the combined image to `VITASLOP_SHOT_DIR`.
//! - `VITASLOP_TRACE_IO` logs every `sceIoOpen` (path, flags, result) - the fastest
//!   way to spot a title that quits because a required data file is missing (a wrong
//!   mount prefix or filename case). See `host::vfs_key`.
//! - `VITASLOP_TRACE_EXIT` dumps, when the guest calls `sceKernelExitProcess`, the
//!   exit code, the caller (LR) + a stack window with code-range words flagged, and
//!   the last 30 host calls tagged by thread - so a clean early-exit is traceable to
//!   the deciding code instead of looking like a crash. (Runtime: `libkernel::trace_exit`.)
//! - `VITASLOP_TRACE_FILE=<path>` writes the WHOLE thread-tagged call trace there, so
//!   the pre-exit decision region is examinable, not just the exit-machinery tail.
//! - `VITASLOP_DUMP_MEM=addr:len,...` / `VITASLOP_CHECK_ADDRS=hex,...` /
//!   `VITASLOP_WASM_INDICES=n,...` inspect guest memory, ask whether an address is a
//!   discovered/lifted/stub function, and map a trap backtrace's wasm indices to
//!   guest addresses (also done automatically for any `RunReport::Error`).

use vitaslop_loader as loader;
use vitaslop_native::{render, CtrlFrame, RunReport, ThreadedScheduler, VitaEnv, World};
use vitaslop_runtime::ingest::pipeline::decrypt_container;
use vitaslop_runtime::ingest::vfs::{DirVfs, Vfs};
use vitaslop_runtime::link::link;

const WIDTH: u32 = 960;
const HEIGHT: u32 = 544;
const CLEAR: [u8; 4] = [0, 0, 0, 255];
/// Instructions a thread retires before the scheduler preempts it (the interleave
/// granularity). Large enough that a run of guest code between host calls finishes
/// in one slice, small enough to keep the round budget meaningful.
const QUANTUM_FUEL: u64 = 5_000_000;
/// Frame flips (display queue entries) to capture before stopping the run.
const MAX_FRAMES: u64 = 3;
/// Fiber-poll backstop so a busy-waiting guest cannot run unbounded.
const MAX_ROUNDS: u64 = 200_000;

/// A minimal host world: a monotonic clock advancing one 60Hz tick per poll, no
/// input, deterministic zero randomness. Enough to boot; input scripting comes
/// with the recipe-driven validation harness later.
#[derive(Default)]
struct BootWorld {
    polls: u64,
}
impl World for BootWorld {
    fn monotonic_us(&mut self) -> u64 {
        self.polls = self.polls.wrapping_add(1);
        self.polls.wrapping_mul(16_666)
    }
    fn wall_us(&mut self) -> u64 {
        0
    }
    fn poll_ctrl(&mut self, _port: u32) -> CtrlFrame {
        CtrlFrame::default()
    }
    fn fill_random(&mut self, buf: &mut [u8]) {
        buf.fill(0);
    }
}

#[test]
#[ignore = "boot probe: needs VITASLOP_GAME_DIR fixture"]
fn retail_boot_probe() {
    let Ok(dir) = std::env::var("VITASLOP_GAME_DIR") else {
        eprintln!("VITASLOP_GAME_DIR not set; skipping");
        return;
    };
    // Artifacts are written only when a shot dir is provided, and only there - never
    // inside the repo. Without it the probe is print-only.
    let shot_dir: Option<String> = std::env::var("VITASLOP_SHOT_DIR").ok();
    if let Some(dir) = &shot_dir {
        std::fs::create_dir_all(dir).expect("create shot dir");
    }

    // 1. Decrypt + link the whole title.
    let game = decrypt_container(&DirVfs::new(&dir)).expect("decrypt container");
    let files = game.files.list();
    let modules: Vec<loader::Module> = game
        .modules
        .iter()
        .map(|m| loader::load(&m.elf).expect("load module"))
        .collect();
    let linked = link(modules).expect("link");
    eprintln!(
        "== retail boot probe ==\nimage={} KiB base={:#x} alloc_base={:#x} \
         modules={} host_imports={} redirects={}",
        linked.image.len() / 1024,
        linked.base,
        linked.alloc_base,
        linked.module_inits.len(),
        linked.imports.len(),
        linked.redirects.len(),
    );

    if std::env::var("VITASLOP_TRANSPILE_REPORT").is_ok() {
        let report = vitaslop_transpiler::transpile_report(&linked.program());
        eprintln!("transpile_report: {} ok, {} failed", report.ok.len(), report.failures.len());
        // Split failures into decode gaps vs lift gaps, and for decode gaps print the
        // real failing address + bytes (so a misaligned example word cannot mislead).
        let mut decode_addrs: Vec<u32> = Vec::new();
        let mut lift_count = 0u32;
        for f in &report.failures {
            let a = format!("{:?}", f.error);
            if let Some(addr) = a
                .strip_prefix("Decode { addr: ")
                .and_then(|s| s.trim_end_matches(" }").parse::<u32>().ok())
            {
                decode_addrs.push(addr);
            } else {
                lift_count += 1;
            }
        }
        decode_addrs.sort_unstable();
        eprintln!("decode gaps: {}, lift gaps: {}", decode_addrs.len(), lift_count);
        // Lift gaps: an instruction that decodes but is not lowered yet. Print the
        // function root, the offending instruction address, and the opcode.
        for f in &report.failures {
            let s = format!("{:?}", f.error);
            if !s.starts_with("Decode") {
                eprintln!("  lift-gap in g_{:08x}: {s}", f.root);
            }
        }
        for addr in decode_addrs.iter().take(60) {
            let off = (addr - linked.base) as usize;
            if off + 4 <= linked.image.len() {
                let hw1 = u16::from_le_bytes([linked.image[off], linked.image[off + 1]]);
                let hw2 = u16::from_le_bytes([linked.image[off + 2], linked.image[off + 3]]);
                eprintln!("  decode-gap @ {addr:#x}: hw1={hw1:#06x} hw2={hw2:#06x}");
            }
        }
    }
    // Map wasm function indices (from a trap backtrace) to guest addresses. The
    // lenient build orders functions by address and assigns wasm index
    // IMPORT_FUNC_COUNT + i, so a fresh build reproduces the same mapping. (Set
    // VITASLOP_WASM_NAMES to have the backtrace print `g_<addr>` directly instead.)
    let import_funcs = vitaslop_transpiler::abi::IMPORT_FUNC_COUNT as usize;
    if let Ok(list) = std::env::var("VITASLOP_WASM_INDICES") {
        let built = vitaslop_transpiler::transpile_lenient(&linked.shared_program());
        for tok in list.split(',') {
            if let Ok(widx) = tok.trim().parse::<usize>() {
                let addr = widx.checked_sub(import_funcs).and_then(|i| built.artifact.funcs.get(i)).map(|fe| fe.addr);
                match addr {
                    Some(a) => eprintln!("  wasm[{widx}] = guest {a:#x}"),
                    None => eprintln!("  wasm[{widx}] = <out of range>"),
                }
            }
        }
    }
    // Report whether specific guest addresses were discovered as functions
    // (VITASLOP_CHECK_ADDRS=hex,hex,...). Answers "is this indirect-call target in
    // the dispatcher's table?" - an undiscovered target traps the dispatcher.
    if let Ok(list) = std::env::var("VITASLOP_CHECK_ADDRS") {
        let built = vitaslop_transpiler::transpile_lenient(&linked.shared_program());
        let addrs: std::collections::BTreeSet<u32> =
            built.artifact.funcs.iter().map(|f| f.addr).collect();
        let stubs: std::collections::BTreeSet<u32> = built.stubbed.iter().copied().collect();
        for tok in list.split(',') {
            let a = u32::from_str_radix(tok.trim().trim_start_matches("0x"), 16).unwrap_or(0);
            let masked = a & !1;
            let status = if !addrs.contains(&masked) {
                "NOT DISCOVERED (dispatcher would trap)"
            } else if stubs.contains(&masked) {
                "discovered but STUB (unlifted)"
            } else {
                "discovered + lifted"
            };
            eprintln!("  check {a:#010x} (masked {masked:#010x}): {status}");
        }
    }
    // Dump a discovered function's lowered IR (VITASLOP_DUMP_FUNC=hex[,hex...]): the
    // authoritative decode/lowering the emitter uses, so a trap's `guest_block` can be
    // read against the exact statements (not a naive linear disassembly that may
    // misalign on undecoded ops). Prints each block's address, statements, terminator.
    if let Ok(list) = std::env::var("VITASLOP_DUMP_FUNC") {
        let program = linked.shared_program();
        if list.trim() == "all" {
            // Dump every discovered function's IR to <shot_dir>/allfuncs.ir, so a
            // global address's readers/writers can be found by grep (e.g. who stores
            // to an uninitialized singleton slot).
            let built = vitaslop_transpiler::transpile_lenient(&program);
            let mut all = String::new();
            for fe in &built.artifact.funcs {
                if let Some(text) = vitaslop_transpiler::dump_func(&program, fe.addr) {
                    all.push_str(&text);
                }
            }
            match &shot_dir {
                Some(dir) => {
                    let path = format!("{dir}/allfuncs.ir");
                    std::fs::write(&path, &all).unwrap();
                    eprintln!("wrote {} funcs IR to {path}", built.artifact.funcs.len());
                }
                None => eprintln!("VITASLOP_DUMP_FUNC=all needs VITASLOP_SHOT_DIR"),
            }
        } else {
            for tok in list.split(',') {
                let want = u32::from_str_radix(tok.trim().trim_start_matches("0x"), 16).unwrap_or(0);
                match vitaslop_transpiler::dump_func(&program, want) {
                    Some(text) => eprint!("{text}"),
                    None => eprintln!("VITASLOP_DUMP_FUNC: no function decodes at {want:#x}"),
                }
            }
        }
    }
    // Dump host-import index -> (library nid, function nid) for the indices listed in
    // VITASLOP_DUMP_IMPORTS (decimal, comma-separated), so an `Import(N)` seen in a
    // dumped block or trap can be resolved to a NID (and thence a name via the db).
    if let Ok(list) = std::env::var("VITASLOP_DUMP_IMPORTS") {
        if list.trim() == "all" {
            for (idx, (lib, nid)) in linked.imports.iter().enumerate() {
                eprintln!("  import[{idx}] = lib={lib:#010x} nid={nid:#010x}");
            }
        } else {
            for tok in list.split(',') {
                if let Ok(idx) = tok.trim().parse::<usize>() {
                    match linked.imports.get(idx) {
                        Some((lib, nid)) => eprintln!("  import[{idx}] = lib={lib:#010x} nid={nid:#010x}"),
                        None => eprintln!("  import[{idx}] = <out of range ({} imports)>", linked.imports.len()),
                    }
                }
            }
        }
    }
    eprintln!("process_param at {:#x}", linked.process_param);
    eprintln!("main_entry (executable module_start) = {:#x}", linked.main_entry);
    if std::env::var("VITASLOP_DUMP_IMAGE").is_ok() {
        match &shot_dir {
            Some(dir) => {
                let path = format!("{dir}/image.bin");
                std::fs::write(&path, &linked.image).unwrap();
                eprintln!("dumped image ({} bytes) to {path}", linked.image.len());
            }
            None => eprintln!("VITASLOP_DUMP_IMAGE set but VITASLOP_SHOT_DIR is not; skipping dump"),
        }
    }

    // 2. Host environment: NID dispatch + capture + the decrypted filesystem, in
    //    PREEMPTIVE mode so blocking primitives really park a thread (a real title
    //    boots its render loop on threads that a synchronous run-to-completion model
    //    cannot interleave). The heap is moved above the whole linked image.
    // Input: a scripted TAS recipe when VITASLOP_INPUT_RECIPE points at a recipe
    // file (frame-keyed button/analog directives; see `vitaslop_runtime::recipe`),
    // else the deterministic input-free BootWorld. The recipe drives menus/dialogs
    // reproducibly (e.g. dismiss the "not signed in" dialog, then move the cursor).
    let world: Box<dyn World + Send> = match std::env::var("VITASLOP_INPUT_RECIPE") {
        Ok(path) => {
            let text = std::fs::read_to_string(&path).expect("read input recipe");
            match vitaslop_runtime::RecipeWorld::parse(&text) {
                Ok(w) => {
                    eprintln!("input: scripted recipe from {path}");
                    Box::new(w)
                }
                Err(e) => panic!("input recipe parse error: {e}"),
            }
        }
        Err(_) => Box::new(BootWorld::default()),
    };
    let mut env = VitaEnv::new(linked.imports.clone(), linked.base, linked.mem_bytes, world);
    env.state.set_alloc_base(linked.alloc_base);
    env.state.set_process_param(linked.process_param);
    env.state.set_preemptive(true);
    for path in &files {
        if let Ok(bytes) = game.files.read(path) {
            env.state.add_file(path, bytes);
        }
    }
    eprintln!("preloaded {} files into the guest filesystem", files.len());

    // 3. Transpile (lenient) + stand up the preemptive scheduler. The main thread
    //    runs every module_start in load order, then the executable's; spawned
    //    threads run concurrently, switched at their blocking points and frame flips.
    let (mut sched, stubbed) =
        ThreadedScheduler::from_linked(&linked, env, QUANTUM_FUEL).expect("scheduler");
    eprintln!("transpiled + instantiated; {} trapping stubs (unlifted funcs)", stubbed.len());
    if std::env::var("VITASLOP_DUMP_STUBS").is_ok() {
        let stub_by_windex: std::collections::BTreeMap<u32, u32> =
            stubbed.iter().map(|&(addr, widx)| (widx, addr)).collect();
        for (widx, addr) in &stub_by_windex {
            eprintln!("  stub wasm[{widx}] = guest {addr:#x}");
        }
    }

    // 4. Run until a few frames flip (a title's render loop never returns), or the
    //    process ends / deadlocks. Bounded by rounds so a busy-wait cannot hang.
    //    `VITASLOP_MAX_FRAMES` overrides the frame budget (fewer = faster diagnostics).
    let max_frames = std::env::var("VITASLOP_MAX_FRAMES")
        .ok()
        .and_then(|s| s.parse().ok())
        .unwrap_or(MAX_FRAMES);
    let max_rounds = std::env::var("VITASLOP_MAX_ROUNDS")
        .ok()
        .and_then(|s| s.parse().ok())
        .unwrap_or(MAX_ROUNDS);
    let report = sched.run_frames(max_frames, max_rounds);
    eprintln!("run report: {report:?}");

    // If the run trapped, auto-map every "wasm function N" in the backtrace to its
    // guest address, so a fault is immediately readable without a second run.
    if let RunReport::Error(msg) = &report {
        let built = vitaslop_transpiler::transpile_lenient(&linked.shared_program());
        let mut seen = std::collections::BTreeSet::new();
        let mut rest = msg.as_str();
        while let Some(p) = rest.find("wasm function ") {
            rest = &rest[p + "wasm function ".len()..];
            let n: String = rest.chars().take_while(|c| c.is_ascii_digit()).collect();
            if let Ok(widx) = n.parse::<usize>() {
                if seen.insert(widx) {
                    let addr = widx
                        .checked_sub(import_funcs)
                        .and_then(|i| built.artifact.funcs.get(i))
                        .map(|fe| fe.addr);
                    match addr {
                        Some(a) => eprintln!("  backtrace wasm[{widx}] = guest {a:#010x}"),
                        None => eprintln!("  backtrace wasm[{widx}] = <dispatcher or out of range>"),
                    }
                }
            }
        }
    }

    // Post-mortem guest memory dump: VITASLOP_DUMP_MEM=addr:len[,addr:len...] (hex
    // addr, decimal len) prints those guest ranges as words. The shared image
    // survives a trap, so this inspects object state at the fault point.
    if let Ok(spec) = std::env::var("VITASLOP_DUMP_MEM") {
        for item in spec.split(',') {
            let Some((a, l)) = item.split_once(':') else { continue };
            let addr = u32::from_str_radix(a.trim().trim_start_matches("0x"), 16).unwrap_or(0);
            let len: usize = l.trim().parse().unwrap_or(64);
            let bytes = sched.read_guest(addr, len);
            if bytes.is_empty() {
                eprintln!("mem[{addr:#010x}..+{len}]: <out of bounds>");
                continue;
            }
            eprintln!("mem[{addr:#010x}..+{len}]:");
            for (i, chunk) in bytes.chunks(16).enumerate() {
                let words: Vec<String> = chunk
                    .chunks(4)
                    .map(|w| {
                        let mut b = [0u8; 4];
                        b[..w.len()].copy_from_slice(w);
                        format!("{:08x}", u32::from_le_bytes(b))
                    })
                    .collect();
                eprintln!("  {:#010x}: {}", addr + (i * 16) as u32, words.join(" "));
            }
        }
    }

    // 5. Report the boot ladder and render whatever was captured.
    let host = sched.host();
    let st = &host.state;
    let reached_main = matches!(
        report,
        RunReport::Finished(_) | RunReport::FramesReached(_) | RunReport::Deadlock(_)
    );
    eprintln!("\n== result ==");
    eprintln!("reached executable main: {reached_main}");
    eprintln!("GXM scenes captured: {}", st.capture.scenes.len());
    // VITASLOP_DUMP_DRAWS dumps every captured scene's draw stream - primitive,
    // vertex stride, index count, the attribute layout (format/components/offset/
    // reg), the uniform vector, any bound fragment textures, and a few decoded
    // vertices. This is the ground-truth tool for bringing up a new title's real
    // geometry (2D sprites, UI quads) versus the placeholder cube.
    if std::env::var("VITASLOP_DUMP_DRAWS").is_ok() {
        for (si, scene) in st.capture.scenes.iter().enumerate() {
            eprintln!(
                "-- scene {si}: color={:?} draws={}",
                scene.color, scene.draws.len()
            );
            for (di, d) in scene.draws.iter().enumerate() {
                eprintln!(
                    "  draw {di}: prim={} idxfmt={} idxcount={} stride={} verts={}B indices={}B uniforms={} textures={}",
                    d.primitive, d.index_format, d.index_count, d.vertex_stride,
                    d.vertices.len(), d.indices.len(), d.uniforms.len(), d.textures.len(),
                );
                for a in &d.attributes {
                    eprintln!(
                        "      attr: stream={} offset={} format={} components={} reg={}",
                        a.stream_index, a.offset, a.format, a.component_count, a.reg_index
                    );
                }
                for (ti, t) in d.textures.iter().enumerate() {
                    let head: Vec<String> =
                        t.pixels.iter().take(16).map(|b| format!("{b:02x}")).collect();
                    eprintln!(
                        "      tex[{}] unit={} base_format={:#04x} swizzle={:#08x} type={:#x} {}x{} stride={} data={:#x} px={}B head=[{}]",
                        ti, t.unit, t.base_format, t.swizzle, t.tex_type, t.width, t.height,
                        t.stride, t.data_addr, t.pixels.len(), head.join(" ")
                    );
                }
                if !d.uniforms.is_empty() {
                    let n = d.uniforms.len().min(20);
                    eprintln!("      uniforms[..{n}]: {:?}", &d.uniforms[..n]);
                }
                // Decode the first few vertices as f32 lanes so the actual position
                // coordinate space (NDC vs pixels vs clip) and UV range are visible.
                let stride = d.vertex_stride.max(1) as usize;
                let nverts = if stride > 0 { d.vertices.len() / stride } else { 0 };
                for vi in 0..nverts.min(4) {
                    let base = vi * stride;
                    let lanes: Vec<f32> = (0..stride / 4)
                        .map(|l| {
                            let o = base + l * 4;
                            f32::from_le_bytes([
                                d.vertices[o], d.vertices[o + 1],
                                d.vertices[o + 2], d.vertices[o + 3],
                            ])
                        })
                        .collect();
                    eprintln!("      v{vi}: {lanes:?}");
                }
            }
        }
    }
    eprintln!("unimplemented NIDs hit ({}):", st.capture.unimplemented.len());
    for (lib, func, name) in &st.capture.unimplemented {
        eprintln!("    lib={lib:#010x} nid={func:#010x} {name}");
    }

    eprintln!("total host calls serviced: {}", st.capture.call_count);
    if !st.capture.stdout.is_empty() {
        eprintln!("--- guest stdout ---\n{}", String::from_utf8_lossy(&st.capture.stdout));
    }
    if !st.capture.stderr.is_empty() {
        eprintln!("--- guest stderr ---\n{}", String::from_utf8_lossy(&st.capture.stderr));
    }
    // Histogram of serviced calls by name, most frequent first.
    let trace = &st.capture.trace;
    let mut hist: std::collections::BTreeMap<&str, u32> = std::collections::BTreeMap::new();
    for &nid in trace {
        *hist.entry(vitaslop_runtime::nid::name(nid)).or_default() += 1;
    }
    let mut hist: Vec<_> = hist.into_iter().collect();
    hist.sort_by(|a, b| b.1.cmp(&a.1));
    eprintln!("--- serviced-call histogram ---");
    for (name, n) in &hist {
        eprintln!("  {n:>5}  {name}");
    }
    // Per-thread call counts (from the thread-id trace) - shows whether the hot
    // spin is one thread or a producer/consumer pair handing off.
    let mut by_thread: std::collections::BTreeMap<i32, u64> = std::collections::BTreeMap::new();
    for &thid in &st.capture.trace_thid {
        *by_thread.entry(thid).or_default() += 1;
    }
    eprintln!("--- calls by thread ---");
    for (thid, n) in &by_thread {
        eprintln!("  thread {thid:>3}: {n}");
    }
    // The last few calls each thread made, in order. A stalled producer/consumer
    // shows its final blocking primitive here (what a worker parked on, what the
    // main loop polls) - the ground truth for a livelock on the loading screen.
    {
        let trace = &st.capture.trace;
        let thids = &st.capture.trace_thid;
        for thid in by_thread.keys() {
            let mut last: Vec<String> = Vec::new();
            for i in (0..trace.len()).rev() {
                if thids.get(i).copied() == Some(*thid) {
                    let nid = trace[i];
                    let nm = vitaslop_runtime::nid::name(nid);
                    last.push(if nm == "<unknown>" { format!("nid:{nid:#010x}") } else { nm.to_string() });
                    if last.len() >= 16 {
                        break;
                    }
                }
            }
            last.reverse();
            eprintln!("  thread {thid:>3} last: {}", last.join(" "));
        }
    }
    vitaslop_runtime::vita::dump_call_sites(14);
    eprintln!("--- sceKernelWaitLwCond samples (work, timeout_ptr, timeout) ---");
    for (work, tp, to) in &st.capture.lwcond_wait_samples {
        eprintln!("  work={work:#010x} timeout_ptr={tp:#010x} timeout={to}");
    }
    // Artifacts (call trace + frame PNGs) go to the shot dir only when one was given.
    match &shot_dir {
        Some(dir) => {
            let trace_txt: String = trace
                .iter()
                .enumerate()
                .map(|(i, &nid)| format!("{i} {} {nid:#010x}\n", vitaslop_runtime::nid::name(nid)))
                .collect();
            let _ = std::fs::write(format!("{dir}/trace.txt"), trace_txt);
            let mut frames_written = 0;
            for (i, scene) in st.capture.scenes.iter().enumerate() {
                let fb = render::render_scene(scene, WIDTH, HEIGHT, CLEAR);
                std::fs::write(format!("{dir}/frame_{i:04}.png"), fb.to_png()).expect("write png");
                frames_written += 1;
            }
            eprintln!("wrote {frames_written} frame PNG(s) + trace to {dir}/");
            // VITASLOP_GPU also renders each scene through the general GXM->WebGPU
            // renderer (the browser's real path) to frame_gpu_XXXX.png, so the GPU
            // output can be compared to the software oracle on the real title's frames.
            if std::env::var("VITASLOP_GPU").is_ok() {
                match vitaslop_native::GeneralRenderer::new() {
                    Some(mut gpu) => {
                        eprintln!("GPU render via {}", gpu.adapter_name);
                        for (i, scene) in st.capture.scenes.iter().enumerate() {
                            let fb = gpu.render_scene(scene, WIDTH, HEIGHT, CLEAR);
                            std::fs::write(format!("{dir}/frame_gpu_{i:04}.png"), fb.to_png())
                                .expect("write gpu png");
                        }
                        eprintln!("wrote {} GPU frame PNG(s) to {dir}/", st.capture.scenes.len());
                    }
                    None => eprintln!("VITASLOP_GPU set but no GPU adapter available"),
                }
            }
        }
        None => eprintln!(
            "VITASLOP_SHOT_DIR unset: not writing frames/trace ({} scene(s) captured)",
            st.capture.scenes.len()
        ),
    }

    // The probe always "passes" - its value is the report and the artifacts. The
    // one hard assertion is that the pipeline got far enough to instantiate and run
    // (a trap on the very first constructor would be a real regression).
    assert!(
        !st.capture.scenes.is_empty() || !st.capture.unimplemented.is_empty() || reached_main,
        "boot made no observable progress at all",
    );
}
