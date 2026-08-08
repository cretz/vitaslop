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
/// The span `VITASLOP_FIND_WORD` searches: from the image base up through the guest heap.
/// Wide enough to cover both the loaded modules and everything allocated above them.
const IMAGE_SCAN_BASE: u32 = vitaslop_runtime::link::IMAGE_BASE;
const IMAGE_SCAN_LEN: usize = 0x1000_0000;
/// Fiber-poll backstop so a busy-waiting guest cannot run unbounded.
const MAX_ROUNDS: u64 = 200_000;

/// A minimal host world: a monotonic clock advancing one 60Hz tick per poll, no
/// input, deterministic zero randomness. Enough to boot; input scripting comes
/// with the recipe-driven validation harness later.
#[derive(Default)]
struct BootWorld {
    polls: u64,
    frame: u64,
}
impl World for BootWorld {
    fn monotonic_us(&mut self) -> u64 {
        self.polls = self.polls.wrapping_add(1);
        self.polls.wrapping_mul(16_666)
    }
    fn wall_us(&mut self) -> u64 {
        // A faithful wall clock ADVANCES with real time. Driving it from the display
        // frame count (60 Hz -> 16.6 ms/frame) atop a fixed calendar epoch gives the
        // game a monotonically advancing sceRtcGetCurrentTick. A frozen wall clock
        // (the old `0`) makes any timeout the game measures via the wall clock never
        // elapse - e.g. the offline-determination timeout behind "Communicating with
        // server...", which then waits forever even though process time advances.
        1_500_000_000_000_000u64.wrapping_add(self.frame.wrapping_mul(16_666))
    }
    fn set_frame(&mut self, frame: u64) {
        self.frame = frame;
    }
    fn poll_ctrl(&mut self, _port: u32) -> CtrlFrame {
        // Diagnostic: VITASLOP_HOLD_BUTTONS=0xHEX holds those SceCtrl buttons down every
        // poll after VITASLOP_HOLD_FROM polls (default 0). Lets the probe test whether an
        // idle attract/title screen is waiting on input. Pulses ~30 polls on / 30 off so a
        // title reacting to an edge (press, not hold) still sees a rising edge repeatedly.
        let hold = std::env::var("VITASLOP_HOLD_BUTTONS")
            .ok()
            .and_then(|s| u32::from_str_radix(s.trim().trim_start_matches("0x"), 16).ok());
        if let Some(buttons) = hold {
            let from = std::env::var("VITASLOP_HOLD_FROM").ok().and_then(|s| s.parse::<u64>().ok()).unwrap_or(0);
            if self.polls >= from && (self.polls / 30) % 2 == 0 {
                let mut f = CtrlFrame::default();
                f.buttons = buttons;
                return f;
            }
        }
        CtrlFrame::default()
    }
    fn poll_touch(&mut self, _port: u32) -> vitaslop_native::TouchFrame {
        // Diagnostic: VITASLOP_HOLD_TOUCH=x,y pulses a front-panel finger at (x,y) ~30
        // polls on / 30 off after VITASLOP_HOLD_FROM. Tests a touch-driven attract screen.
        if let Ok(spec) = std::env::var("VITASLOP_HOLD_TOUCH") {
            if let Some((xs, ys)) = spec.split_once(',') {
                if let (Ok(x), Ok(y)) = (xs.trim().parse::<u16>(), ys.trim().parse::<u16>()) {
                    let from = std::env::var("VITASLOP_HOLD_FROM").ok().and_then(|s| s.parse::<u64>().ok()).unwrap_or(0);
                    if self.polls >= from && (self.polls / 30) % 2 == 0 {
                        return vitaslop_native::TouchFrame::single(x, y);
                    }
                }
            }
        }
        vitaslop_native::TouchFrame::default()
    }
    fn fill_random(&mut self, buf: &mut [u8]) {
        buf.fill(0);
    }
}

/// One memory address to sample each frame in observation mode: guest address, how
/// to interpret the bytes, and a human label for the CSV column.
struct Watch {
    addr: u32,
    ty: &'static str,
    label: String,
}

/// Parse `VITASLOP_WATCH_MEM=addr:type:label,addr:type:label,...` into watches.
/// `addr` is hex (with or without `0x`), `type` is one of u8|u16|u32|i32|f32, and
/// `label` is the CSV column name. Silently drops malformed entries.
fn parse_watches(spec: Option<&str>) -> Vec<Watch> {
    let Some(spec) = spec else { return Vec::new() };
    let mut out = Vec::new();
    for item in spec.split(',') {
        let item = item.trim();
        if item.is_empty() {
            continue;
        }
        let mut parts = item.splitn(3, ':');
        let (Some(a), Some(t)) = (parts.next(), parts.next()) else { continue };
        let Ok(addr) = u32::from_str_radix(a.trim().trim_start_matches("0x"), 16) else { continue };
        let ty = match t.trim() {
            "u8" => "u8",
            "u16" => "u16",
            "u32" => "u32",
            "i32" => "i32",
            "f32" => "f32",
            _ => continue,
        };
        let label = parts.next().map(|s| s.trim().to_string()).unwrap_or_else(|| format!("{addr:#x}"));
        out.push(Watch { addr, ty, label });
    }
    out
}

/// Read and format one watched value from current guest memory.
fn sample_watch(sched: &ThreadedScheduler<VitaEnv>, w: &Watch) -> String {
    let n = match w.ty {
        "u8" => 1,
        "u16" => 2,
        _ => 4,
    };
    let b = sched.read_guest(w.addr, n);
    if b.len() < n {
        return "oob".to_string();
    }
    match w.ty {
        "u8" => format!("{}", b[0]),
        "u16" => format!("{}", u16::from_le_bytes([b[0], b[1]])),
        "u32" => format!("{}", u32::from_le_bytes([b[0], b[1], b[2], b[3]])),
        "i32" => format!("{}", i32::from_le_bytes([b[0], b[1], b[2], b[3]])),
        "f32" => format!("{}", f32::from_le_bytes([b[0], b[1], b[2], b[3]])),
        _ => "?".to_string(),
    }
}

#[test]
#[ignore = "boot probe: needs VITASLOP_GAME_DIR or VITASLOP_GAME_PKG+_WORK"]
fn retail_boot_probe() {
    // Surface the runtime's `tracing` diagnostics (VITASLOP_LOG=vitaslop::io=trace, ...);
    // `RUST_LOG` still works as the fallback - see `knobs::log_filter`.
    let _ = tracing_subscriber::fmt()
        .with_env_filter(tracing_subscriber::EnvFilter::new(vitaslop_runtime::knobs::log_filter()))
        .with_writer(std::io::stderr)
        .try_init();
    // Two ingest sources: an extracted PFS dir (VITASLOP_GAME_DIR) or a NoNpDrm
    // pkg + standalone work.bin (VITASLOP_GAME_PKG + VITASLOP_GAME_WORK).
    let pkg = std::env::var("VITASLOP_GAME_PKG").ok();
    let dir = std::env::var("VITASLOP_GAME_DIR").ok();
    if pkg.is_none() && dir.is_none() {
        eprintln!("neither VITASLOP_GAME_DIR nor VITASLOP_GAME_PKG set; skipping");
        return;
    }
    // Artifacts are written only when a shot dir is provided, and only there - never
    // inside the repo. Without it the probe is print-only.
    let shot_dir: Option<String> = std::env::var("VITASLOP_SHOT_DIR").ok();
    if let Some(dir) = &shot_dir {
        std::fs::create_dir_all(dir).expect("create shot dir");
    }

    // 1. Decrypt + link the whole title.
    let game = if let Some(pkg) = &pkg {
        let work = std::env::var("VITASLOP_GAME_WORK").expect("VITASLOP_GAME_WORK for pkg");
        let pkg_bytes = std::fs::read(pkg).expect("read pkg");
        let work_bytes = std::fs::read(work).expect("read work.bin");
        vitaslop_runtime::ingest::pipeline::decrypt_pkg(&pkg_bytes, &work_bytes).expect("decrypt pkg")
    } else {
        decrypt_container(&mut DirVfs::new(dir.as_ref().unwrap())).expect("decrypt container")
    };
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
        let gap_cap: usize = std::env::var("VITASLOP_GAP_CAP")
            .ok()
            .and_then(|s| s.parse().ok())
            .unwrap_or(60);
        for addr in decode_addrs.iter().take(gap_cap) {
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
    env.state.set_modules(linked.loaded_modules.clone());
    env.state.set_tls_template(linked.tls_template);
    env.state.set_preemptive(true);
    let mut preloaded = 0usize;
    let dump_paths = std::env::var("VITASLOP_DUMP_PATHS").is_ok();
    for (path, bytes) in game.files.into_files() {
        if dump_paths {
            eprintln!("  vfs: {path} ({} bytes)", bytes.len());
        }
        env.state.add_file(&path, bytes);
        preloaded += 1;
    }
    eprintln!("preloaded {preloaded} files into the guest filesystem");

    // 3. Transpile (lenient) + stand up the preemptive scheduler. The main thread
    //    runs every module_start in load order, then the executable's; spawned
    //    threads run concurrently, switched at their blocking points and frame flips.
    // VITASLOP_QUANTUM_FUEL overrides the preemption granularity. Set it huge to run
    // thread 0 uninterrupted (it runs essentially alone until the config-resolution
    // crash), making the boot deterministic so block/watch tracing does not shift the
    // schedule and change the outcome.
    let quantum = std::env::var("VITASLOP_QUANTUM_FUEL")
        .ok()
        .and_then(|s| s.parse::<u64>().ok())
        .unwrap_or(QUANTUM_FUEL);
    let (mut sched, stubbed) =
        ThreadedScheduler::from_linked(&linked, env, quantum).expect("scheduler");
    eprintln!("transpiled + instantiated; {} trapping stubs (unlifted funcs)", stubbed.len());
    // How many completed scenes to KEEP. The probe used to keep every one, which is fine
    // for a 3-frame bring-up run and ruinous for a long one: this title submits six
    // scenes a frame, so a 30,000-frame boot retained 180,000 of them - gigabytes of
    // captured geometry that nothing ever reads, on a run whose whole purpose is to get
    // further. Eviction folds into the determinism signature, so a limit never changes
    // `@sig` (see `Capture::record_scene`).
    //
    // The default keeps exactly what this run can actually look at: everything when
    // frames are being written and no window was given, `VITASLOP_SHOT_LAST`'s window
    // when there is one, and a single scene when nothing is being written at all.
    // `VITASLOP_SCENE_LIMIT=0` restores unlimited retention.
    {
        let shot_last: Option<usize> =
            std::env::var("VITASLOP_SHOT_LAST").ok().and_then(|s| s.parse().ok());
        // How many completed GXM scenes the probe KEEPS (0 = unlimited). A long boot
        // submits hundreds of thousands; keeping them all costs gigabytes and nothing
        // reads them. Eviction folds into the determinism signature, so this never
        // changes what a run reports about itself.
        let requested = std::env::var("VITASLOP_SCENE_LIMIT").ok().and_then(|s| s.parse::<usize>().ok());
        let limit = match requested {
            Some(0) => None,
            Some(n) => Some(n),
            None => match (&shot_dir, shot_last) {
                (None, _) => Some(1),
                (Some(_), Some(k)) => Some(k.max(1)),
                (Some(_), None) => None,
            },
        };
        sched.host().state.capture.scene_limit = limit;
    }
    // qemu-diff faithfulness (VITASLOP_PATCH_STUBS): the on-disk inter-module import
    // stubs are unresolved placeholders (e.g. `mvn r0,#0; bx lr` = return -1); our
    // transpiler resolves those calls by redirection at transpile time and never writes
    // the resolved target into the guest stub. An external reference CPU (qemu) replaying
    // the raw snapshot would therefore run the placeholder and diverge. Patch each stub
    // with a tiny ARM trampoline to its resolved guest target, so qemu follows the SAME
    // routine our engine reaches. Harmless to our own run - we never execute the memory
    // stub (the call is redirected in wasm). The trampoline is `ldr pc, [pc, #-4]; .word
    // target|1`: LDR-to-PC interworks (selects Thumb from bit0) in ARMv7 and clobbers NO
    // general register, so it leaves the exact same register state our direct redirect does
    // (a `bx ip` veneer would perturb r12/ip). Assumes the stub is entered in ARM via `blx`
    // (true for the observed placeholders); the diff flags any exception.
    if std::env::var("VITASLOP_PATCH_STUBS").is_ok() {
        let mut patched = 0u32;
        for r in &linked.redirects {
            let mut tramp = Vec::with_capacity(8);
            tramp.extend_from_slice(&0xe51f_f004u32.to_le_bytes()); // ldr pc, [pc, #-4]
            tramp.extend_from_slice(&(r.target | 1).to_le_bytes()); // target (thumb bit set)
            sched.write_guest(r.addr & !1, &tramp);
            patched += 1;
        }
        eprintln!("[qdiff] patched {patched} import stubs with resolved trampolines");
    }
    // VITASLOP_PREPOKE=0xaddr=0xval,... write words to guest memory once before the run
    // starts (causality test for uninitialized globals the CRT should have set).
    if let Ok(s) = std::env::var("VITASLOP_PREPOKE") {
        for item in s.split(',') {
            if let Some((a, v)) = item.split_once('=') {
                let addr = u32::from_str_radix(a.trim().trim_start_matches("0x"), 16).unwrap();
                let val = u32::from_str_radix(v.trim().trim_start_matches("0x"), 16).unwrap();
                sched.write_guest(addr, &val.to_le_bytes());
                eprintln!("[prepoke] {addr:#x} = {val:#x}");
            }
        }
    }
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
    // Observation/steering mode (VITASLOP_WATCH_MEM=addr:type:label,...): step the
    // run one display flip at a time and sample the listed guest addresses each
    // frame, logging them to <shot>/watch.csv (and stderr). This is the frame-scrub
    // loop for reverse-engineering a title's live state (skater x, score, a
    // level-complete flag) without pixels - the values guide input authoring. Types:
    // u8|u16|u32|i32|f32. When unset, the run is a single bounded call as before.
    let watch = parse_watches(std::env::var("VITASLOP_WATCH_MEM").ok().as_deref());
    // Generic RE tool: VITASLOP_DUMP_REGION=hexaddr:len writes the raw guest bytes of that
    // region to <shot>/mem/f<frame>.bin on each stepped frame (optionally only within
    // VITASLOP_DUMP_REGION_RANGE=lo-hi). Diffing those dumps across a known behaviour
    // (skater grounded vs airborne) finds the address of a live state value - the
    // value-search an agent needs to steer a game precisely. Also triggers frame-stepping.
    let dump_region: Option<(u32, usize)> = std::env::var("VITASLOP_DUMP_REGION").ok().and_then(|s| {
        let (a, l) = s.split_once(':')?;
        let addr = u32::from_str_radix(a.trim().trim_start_matches("0x"), 16).ok()?;
        Some((addr, l.trim().parse().ok()?))
    });
    let dump_range: Option<(u64, u64)> = std::env::var("VITASLOP_DUMP_REGION_RANGE").ok().and_then(|s| {
        let (lo, hi) = s.split_once('-')?;
        Some((lo.trim().parse().ok()?, hi.trim().parse().ok()?))
    });
    let report = if watch.is_empty() && dump_region.is_none() {
        // Causality probe VITASLOP_STALL_WAKE=<id|0xwork>,...: run the boot in
        // `VITASLOP_STALL_CHUNK` (default 200k) round chunks, and each time the run
        // stalls (RoundLimit - one thread spinning while the rest are parked) force
        // deliver a signal to the listed semaphores (decimal id) and lightweight conds
        // (0x-prefixed work pointer), then continue, up to VITASLOP_STALL_WAVES (default
        // 40). This tests whether a boot freeze is a missing/lost worker wakeup: if
        // injecting the signal a real title's producer would have sent unblocks the
        // boot, the freeze is a sync-handoff bug, not a data/lift bug.
        let r = if let Ok(spec) = std::env::var("VITASLOP_STALL_WAKE") {
            let mut sema_ids: Vec<i32> = Vec::new();
            let mut cond_works: Vec<u32> = Vec::new();
            for t in spec.split(',').map(str::trim).filter(|s| !s.is_empty()) {
                if let Some(h) = t.strip_prefix("0x") {
                    if let Ok(w) = u32::from_str_radix(h, 16) { cond_works.push(w); }
                } else if let Ok(id) = t.parse::<i32>() { sema_ids.push(id); }
            }
            let chunk: u64 = std::env::var("VITASLOP_STALL_CHUNK").ok()
                .and_then(|s| s.parse().ok()).unwrap_or(200_000);
            let max_waves: u32 = std::env::var("VITASLOP_STALL_WAVES").ok()
                .and_then(|s| s.parse().ok()).unwrap_or(40);
            eprintln!("STALL_WAKE: semas={sema_ids:?} conds={cond_works:?} chunk={chunk} waves<={max_waves}");
            let mut done;
            let mut waves = 0u32;
            loop {
                let rep = sched.run_frames(max_frames, chunk);
                match rep {
                    RunReport::RoundLimit if waves < max_waves => {
                        {
                            let mut h = sched.host();
                            for &id in &sema_ids { h.state.sema_signal_wake(id, 1); }
                            for &w in &cond_works { h.state.lwcond_signal(w, true); }
                        }
                        waves += 1;
                        eprintln!("  wave {waves}: signalled at frame {}", sched.frames());
                    }
                    other => { done = other; break; }
                }
            }
            eprintln!("STALL_WAKE done: {waves} wake wave(s), frames={}", sched.frames());
            done
        } else {
            sched.run_frames(max_frames, max_rounds)
        };
        eprintln!("run report: {r:?}");
        // VITASLOP_DUMP_MAP=<hex ptr-to-map-object>: after the run, in-order walk a
        // libstdc++/Sony std::map<string,V> red-black tree and print every key. Node
        // layout (from RE): parent@+0, left@+4, right@+8, key SSO buffer@+0x10,
        // size@+0x20 (data is inline at +0x10 when size<=15 else a heap ptr at +0x10).
        // Map object: node_count@+0x10, root@+0x14. Telemetry for "why did find() miss".
        if let Ok(mv) = std::env::var("VITASLOP_DUMP_MAP") {
            let rdu = |a: u32| -> u32 {
                if a < 0x8000_0000 { return 0; }
                let b = sched.read_guest(a, 4);
                if b.len() < 4 { return 0; }
                u32::from_le_bytes([b[0], b[1], b[2], b[3]])
            };
            let rdkey = |node: u32| -> String {
                let sz = (rdu(node + 0x20) as usize).min(256);
                let dptr = if sz <= 15 { node + 0x10 } else { rdu(node + 0x10) };
                if dptr < 0x8000_0000 { return String::new(); }
                let raw = sched.read_guest(dptr, sz);
                String::from_utf8_lossy(&raw).into_owned()
            };
            for tok in mv.split(',') {
                let holder = u32::from_str_radix(tok.trim().trim_start_matches("0x"), 16).unwrap_or(0);
                let mapobj = rdu(holder); // the global holds a pointer to the map object
                eprintln!("MAP holder @{holder:#x} -> obj {mapobj:#x}");
                let count = rdu(mapobj + 0x10);
                let root = rdu(mapobj + 0x14);
                eprintln!("MAP @{mapobj:#x}: count={count} root={root:#x}");
                // Show the raw first-8-words of the map object header and the root node so
                // the true link layout is visible (in-order offsets were guessed wrong).
                for base in [mapobj, root, 0x8775e650u32, 0x8775e7d0u32, 0x8775e760u32] {
                    let words: Vec<String> = (0..12).map(|i| format!("{:#x}", rdu(base + i * 4))).collect();
                    eprintln!("  node@{base:#x} words: {}", words.join(" "));
                }
                // Cycle-safe DFS: treat +4 and +8 (and +0xc) as candidate child links,
                // dedupe via a visited set, cap at 400 nodes. Print each node's key under
                // all plausible interpretations so the layout is unambiguous.
                let _ = root;
                // The nodes are a contiguous pool. Linear-scan a window for node-like
                // records: a valid key (size@+0x20 in 1..=200, printable path bytes at
                // +0x10 inline or via heap ptr). Collect distinct keys with their addrs.
                let is_key_at = |p: u32| -> Option<String> {
                    let sz = rdu(p + 0x20);
                    if sz == 0 || sz > 200 { return None; }
                    let dptr = if (sz as usize) <= 15 { p + 0x10 } else { rdu(p + 0x10) };
                    if dptr < 0x8000_0000 { return None; }
                    let cap = rdu(p + 0x24);
                    if (sz as usize) <= 15 && cap != 15 { return None; }
                    let raw = sched.read_guest(dptr, sz as usize);
                    if raw.len() != sz as usize { return None; }
                    if !raw.iter().all(|&b| b == b'/' || b == b'.' || b == b'_' || b == b'-' || b == b' ' || b.is_ascii_alphanumeric()) { return None; }
                    Some(String::from_utf8_lossy(&raw).into_owned())
                };
                let mut found: Vec<(u32, String)> = Vec::new();
                let mut p = mapobj & !0xf;
                let end = p + 0x20000; // 128 KiB window over the node pool
                while p < end {
                    if let Some(k) = is_key_at(p) { found.push((p, k)); }
                    p += 4;
                }
                found.sort_by(|a, b| a.1.cmp(&b.1));
                for (a, k) in &found { eprintln!("  key @{a:#x} {k:?}"); }
                eprintln!("MAP scan found {} node-like keys (count field={count})", found.len());
                // Count occurrences of the crash lookup key to spot duplicates/variants.
                let target = "configs/msrc.cfg";
                let hits: Vec<&(u32,String)> = found.iter().filter(|(_,k)| k.eq_ignore_ascii_case(target)).collect();
                eprintln!("  target {target:?}: {} case-insensitive match(es): {:?}", hits.len(), hits);
            }
        }
        r
    } else {
        // Per-frame round budget: one flip's worth of guest work. The overall
        // max_rounds still applies per step, but a flip rarely needs the full budget.
        let per_frame_rounds = std::env::var("VITASLOP_ROUNDS_PER_FRAME")
            .ok()
            .and_then(|s| s.parse().ok())
            .unwrap_or(4_000_000u64);
        let header: String =
            std::iter::once("frame".to_string()).chain(watch.iter().map(|w| w.label.clone())).collect::<Vec<_>>().join(",");
        let mut csv = format!("{header}\n");
        eprintln!("watch: {header}");
        if let (Some((a, l)), Some(dir)) = (dump_region, &shot_dir) {
            std::fs::create_dir_all(format!("{dir}/mem")).ok();
            eprintln!("dump region {a:#x}..+{l} to {dir}/mem/f<frame>.bin");
        }
        // VITASLOP_WATCH_FROM=N runs the uninteresting prefix (frames 0..N) as ONE fast
        // batched call, then steps/watches from N on. Cuts a watch run's wall time from
        // ~stepping-every-frame to just the window that matters - the difference between
        // a ~90s and a ~30s iteration when the prefix is a long menu+lesson navigation.
        let mut last = RunReport::FramesReached(0);
        if let Some(from) = std::env::var("VITASLOP_WATCH_FROM").ok().and_then(|s| s.parse::<u64>().ok()) {
            if from > 0 && from < max_frames {
                last = sched.run_frames(from, max_rounds);
                eprintln!("batched prefix to frame {} ({last:?})", sched.frames());
            }
        }
        // Diagnostic steering: VITASLOP_POKE=addr:frame:value (hex addr, decimal frame,
        // decimal u32 value) writes `value` to guest `addr` once, at the start of the
        // given frame. Lets a probe force a stuck state variable to test causality.
        let poke: Option<(u32, u64, u32)> = std::env::var("VITASLOP_POKE").ok().and_then(|s| {
            let mut it = s.split(':');
            let addr = u32::from_str_radix(it.next()?.trim().trim_start_matches("0x"), 16).ok()?;
            let frame: u64 = it.next()?.trim().parse().ok()?;
            let value: u32 = it.next()?.trim().parse().ok()?;
            Some((addr, frame, value))
        });
        // Diagnostic steering: VITASLOP_SET_EVF=id:frame[:bits],... force-sets event
        // flag `id` (bits default all-ones) at the start of every frame >= `frame`,
        // waking any parked waiter exactly as a guest sceKernelSetEventFlag would. This
        // tests, without guessing at the producer, whether waking a stuck worker thread
        // actually unblocks the boot (causality check for a never-fired completion event).
        let set_evfs: Vec<(i32, u64, u32)> = std::env::var("VITASLOP_SET_EVF")
            .ok()
            .map(|s| {
                s.split(',')
                    .filter_map(|item| {
                        let mut it = item.split(':');
                        let id: i32 = it.next()?.trim().parse().ok()?;
                        let frame: u64 = it.next()?.trim().parse().ok()?;
                        let bits: u32 = it
                            .next()
                            .and_then(|b| u32::from_str_radix(b.trim().trim_start_matches("0x"), 16).ok())
                            .unwrap_or(0xFFFF_FFFF);
                        Some((id, frame, bits))
                    })
                    .collect()
            })
            .unwrap_or_default();
        // VITASLOP_HOLD_MEM=addr:value[:fromframe],... writes `value` (hex or dec) to guest
        // `addr` at the START of every frame >= fromframe (default 0). Unlike POKE (one shot),
        // this re-asserts each frame, so a per-frame-cleared word (e.g. a completion mask) can
        // be held set long enough for that frame's consumers to observe it.
        let hold_mems: Vec<(u32, u32, u64)> = std::env::var("VITASLOP_HOLD_MEM")
            .ok()
            .map(|s| {
                s.split(',')
                    .filter_map(|item| {
                        let mut it = item.split(':');
                        let addr = u32::from_str_radix(it.next()?.trim().trim_start_matches("0x"), 16).ok()?;
                        let vs = it.next()?.trim();
                        let value = u32::from_str_radix(vs.trim_start_matches("0x"), 16)
                            .or_else(|_| vs.parse::<u32>())
                            .ok()?;
                        let from: u64 = it.next().and_then(|f| f.trim().parse().ok()).unwrap_or(0);
                        Some((addr, value, from))
                    })
                    .collect()
            })
            .unwrap_or_default();
        // VITASLOP_FORCE_READY=vtable[:lo-hi] (hex): at the start of every frame, scan the
        // guest heap range [lo,hi) (default 0x82000000-0x82c00000) for any 4-byte word equal
        // to `vtable`, and if found at offset 0 of an object, set that object's +84 byte-word
        // to 1. This simulates "resource completion always succeeds" for EVERY resObj of that
        // class (not just one fixed address), so the boot coroutine can flow through its whole
        // sequence of sequential resource waits. Causality test for the never-firing streaming
        // completion: if the boot advances through all screens / the state machine leaves 2,
        // the completion model is confirmed and this is the forcing path to make real.
        let force_ready: Option<(u32, u32, u32)> = std::env::var("VITASLOP_FORCE_READY")
            .ok()
            .and_then(|s| {
                let (vt_s, range) = s.split_once(':').unwrap_or((s.as_str(), ""));
                let vt = u32::from_str_radix(vt_s.trim().trim_start_matches("0x"), 16).ok()?;
                let (lo, hi) = if let Some((a, b)) = range.split_once('-') {
                    (
                        u32::from_str_radix(a.trim().trim_start_matches("0x"), 16).ok()?,
                        u32::from_str_radix(b.trim().trim_start_matches("0x"), 16).ok()?,
                    )
                } else {
                    (0x8200_0000u32, 0x82c0_0000u32)
                };
                Some((vt, lo, hi))
            });
        // VITASLOP_FORCE_READY_V2=vtable2[:lo-hi]: like FORCE_READY but matches the SHARED
        // base-class vtable at object offset +4 (0x811fbba4 for RR's resource items), so it
        // covers EVERY resource-item subclass (text/font/etc) at once, and only marks an item
        // ready if its data pointer (+92) is non-null (i.e. it actually finished loading).
        // This is the general "HLE resource completion" stand-in for the streaming subsystem
        // whose real completion trigger never fires under our synchronous HLE.
        let force_ready_v2: Option<(u32, u32, u32)> = std::env::var("VITASLOP_FORCE_READY_V2")
            .ok()
            .and_then(|s| {
                let (vt_s, range) = s.split_once(':').unwrap_or((s.as_str(), ""));
                let vt = u32::from_str_radix(vt_s.trim().trim_start_matches("0x"), 16).ok()?;
                let (lo, hi) = if let Some((a, b)) = range.split_once('-') {
                    (
                        u32::from_str_radix(a.trim().trim_start_matches("0x"), 16).ok()?,
                        u32::from_str_radix(b.trim().trim_start_matches("0x"), 16).ok()?,
                    )
                } else {
                    (0x8200_0000u32, 0x8300_0000u32)
                };
                Some((vt, lo, hi))
            });
        while sched.frames() < max_frames {
            if let Some((vt, lo, hi)) = force_ready {
                let bytes = sched.read_guest(lo, (hi - lo) as usize);
                let mut hits = 0u32;
                let mut i = 0usize;
                while i + 4 <= bytes.len() {
                    let w = u32::from_le_bytes([bytes[i], bytes[i + 1], bytes[i + 2], bytes[i + 3]]);
                    if w == vt {
                        let obj = lo + i as u32;
                        sched.write_guest(obj + 84, &1u32.to_le_bytes());
                        hits += 1;
                    }
                    i += 4;
                }
                if sched.frames() < 3 {
                    eprintln!("FORCE_READY frame {}: set +84=1 on {hits} obj(s) with vtable {vt:#x}", sched.frames());
                }
            }
            if let Some((vt, lo, hi)) = force_ready_v2 {
                let bytes = sched.read_guest(lo, (hi - lo) as usize);
                let mut hits = 0u32;
                let mut i = 4usize; // candidate obj+4 position
                while i + 4 <= bytes.len() {
                    let w = u32::from_le_bytes([bytes[i], bytes[i + 1], bytes[i + 2], bytes[i + 3]]);
                    // match vtable2 at +4; object starts 4 bytes earlier
                    if w == vt && i + 92 <= bytes.len() {
                        let data = u32::from_le_bytes([bytes[i + 88], bytes[i + 89], bytes[i + 90], bytes[i + 91]]);
                        let ready = u32::from_le_bytes([bytes[i + 80], bytes[i + 81], bytes[i + 82], bytes[i + 83]]);
                        if data != 0 && ready == 0 {
                            let obj = lo + (i as u32) - 4;
                            sched.write_guest(obj + 84, &1u32.to_le_bytes());
                            hits += 1;
                        }
                    }
                    i += 4;
                }
                if sched.frames() < 3 || hits > 0 {
                    eprintln!("FORCE_READY_V2 frame {}: set +84 on {hits} loaded item(s) (vtable2 {vt:#x})", sched.frames());
                }
            }
            if let Some((addr, frame, value)) = poke {
                if sched.frames() == frame {
                    sched.write_guest(addr, &value.to_le_bytes());
                    eprintln!("POKE {addr:#x} = {value} at frame {frame}");
                }
            }
            for &(addr, value, from) in &hold_mems {
                if sched.frames() >= from {
                    sched.write_guest(addr, &value.to_le_bytes());
                }
            }
            for &(id, frame, bits) in &set_evfs {
                if sched.frames() >= frame {
                    let mut h = sched.host();
                    h.state.event_set_wake(id, bits);
                }
            }
            let target = sched.frames() + 1;
            last = sched.run_frames(target, per_frame_rounds);
            let f = sched.frames();
            let mut row = vec![f.to_string()];
            for w in &watch {
                row.push(sample_watch(&sched, w));
            }
            let line = row.join(",");
            csv.push_str(&line);
            csv.push('\n');
            if !watch.is_empty() {
                eprintln!("w {line}");
            }
            // Per-frame raw region dump (for value-search diffing), bounded to the range.
            if let (Some((a, l)), Some(dir)) = (dump_region, &shot_dir) {
                let in_range = dump_range.map(|(lo, hi)| f >= lo && f <= hi).unwrap_or(true);
                if in_range {
                    let bytes = sched.read_guest(a, l);
                    if !bytes.is_empty() {
                        let _ = std::fs::write(format!("{dir}/mem/f{f:05}.bin"), &bytes);
                    }
                }
            }
            // Stop early if the guest actually finished/trapped (not just a flip).
            if !matches!(last, RunReport::FramesReached(_)) {
                break;
            }
        }
        eprintln!("run report: {last:?}");
        if let Some(dir) = &shot_dir {
            let _ = std::fs::write(format!("{dir}/watch.csv"), &csv);
            eprintln!("wrote watch.csv to {dir}/");
        }
        last
    };

    // Block-visit histogram (VITASLOP_BLOCK_HIST, with VITASLOP_TRACE_BLOCKS emitting
    // the hooks): the empirical map of a hot loop's structure - hottest blocks, trip
    // counts, and the exact repeating cycle whose head is the loop head to snapshot.
    let hist_top = std::env::var("VITASLOP_BLOCK_HIST")
        .ok()
        .and_then(|s| s.trim().parse::<usize>().ok())
        .unwrap_or(40);
    vitaslop_native::dump_block_hist(hist_top);

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

    // Post-mortem word search: VITASLOP_FIND_WORD=0xVALUE[,0xVALUE...] reports every
    // 4-byte-aligned guest address holding that word. Written for "who holds this
    // function pointer?" after a dispatch miss - the answer separates a pointer the
    // linker wrote into a static table (which static discovery could find) from one the
    // guest computed at runtime (which it cannot), and those need different fixes.
    if let Ok(spec) = std::env::var("VITASLOP_FIND_WORD") {
        let wants: Vec<u32> = spec
            .split(',')
            .filter_map(|t| u32::from_str_radix(t.trim().trim_start_matches("0x"), 16).ok())
            .collect();
        // The whole guest region in one borrow; a chunked read would miss a word
        // straddling the boundary and costs another copy.
        let image = sched.read_guest(IMAGE_SCAN_BASE, IMAGE_SCAN_LEN);
        for want in wants {
            let mut hits = 0;
            for (i, w) in image.chunks_exact(4).enumerate() {
                if u32::from_le_bytes([w[0], w[1], w[2], w[3]]) == want {
                    eprintln!("  find {want:#010x}: at {:#010x}", IMAGE_SCAN_BASE + 4 * i as u32);
                    hits += 1;
                    if hits >= 64 {
                        eprintln!("  find {want:#010x}: ... (more than 64 hits)");
                        break;
                    }
                }
            }
            if hits == 0 {
                eprintln!("  find {want:#010x}: not present in guest memory");
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

    // Post-mortem word scan: VITASLOP_SCAN_WORD=0xVAL[:lo-hi] finds every guest
    // address whose 32-bit word == VAL. Reloc-filled function-pointer tables (bound
    // completion callbacks / dispatch vectors) are invisible to the static image and
    // IR - their pointers are written at load time - so scanning live memory is the
    // only way to find who holds a given handler and where it is dispatched from.
    if let Ok(spec) = std::env::var("VITASLOP_SCAN_WORD") {
        let (val_s, range) = spec.split_once(':').unwrap_or((spec.as_str(), ""));
        let val = u32::from_str_radix(val_s.trim().trim_start_matches("0x"), 16).unwrap_or(0);
        let (lo, hi) = if let Some((a, b)) = range.split_once('-') {
            (
                u32::from_str_radix(a.trim().trim_start_matches("0x"), 16).unwrap_or(0x8120_0000),
                u32::from_str_radix(b.trim().trim_start_matches("0x"), 16).unwrap_or(0x8160_0000),
            )
        } else {
            (0x8120_0000u32, 0x8300_0000u32)
        };
        eprintln!("scan for {val:#010x} in [{lo:#010x}, {hi:#010x}):");
        let bytes = sched.read_guest(lo, (hi - lo) as usize);
        let mut hits = 0u32;
        let mut a = 0usize;
        while a + 4 <= bytes.len() {
            let w = u32::from_le_bytes([bytes[a], bytes[a + 1], bytes[a + 2], bytes[a + 3]]);
            if w == val {
                eprintln!("  hit @ {:#010x}", lo + a as u32);
                hits += 1;
                if hits >= 200 {
                    eprintln!("  ...(capped at 200)");
                    break;
                }
            }
            a += 4;
        }
        eprintln!("  {hits} hit(s)");
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
    // The TRUE count, not the retained one: `scene_limit` bounds how many are kept,
    // and reporting the retained figure as "captured" would understate a long run by
    // orders of magnitude and read as a regression when retention was bounded.
    eprintln!(
        "GXM scenes captured: {} ({} retained, {} evicted)",
        st.capture.scenes.len() as u64 + st.capture.retired_scenes,
        st.capture.scenes.len(),
        st.capture.retired_scenes,
    );
    // The game->OS egress ledger: the human-readable milestones a conformance recipe
    // asserts on (savedata writes, trophies, score submits), each frame-tagged. This
    // is the content-free "the game reached a real state" signal - no pixels needed.
    eprintln!("egress ledger ({} events):", st.capture.egress.len());
    for ev in &st.capture.egress {
        eprintln!("  f{:>5} {:?}", ev.frame, ev.kind);
    }
    // Determinism signature: a cheap FNV-1a hash over the captured render stream
    // (per scene: color target + every draw's vertex/index/uniform bytes) and the
    // egress ledger. Two runs of the same recipe must print the SAME signature - that
    // is what makes a recipe a reproducible test rather than a one-off. Run the probe
    // twice with a recipe and diff this line to prove the run is deterministic.
    // Use `Capture::signature()` - the ONE authority - rather than folding
    // `capture.scenes` here. The probe used to hand-roll its own FNV over the retained
    // scenes, which was a second implementation of the same idea and, worse, was not
    // invariant under `scene_limit`: once retention is bounded the retained scenes are
    // only the tail of the run, so the hand-rolled value silently described a few
    // scenes instead of all of them and compared unequal to an unlimited run of the
    // very same boot. `Capture::signature` folds `retired_digest` (every evicted scene)
    // first, so it is identical at any retention setting - verified by running the same
    // boot at `VITASLOP_SCENE_LIMIT=0` and at the default.
    //
    // NOTE this is a different hash from the probe's old hand-rolled one, so probe
    // signatures recorded before 2026-07-28 are not comparable to these.
    let h = st.capture.signature();
    eprintln!(
        "determinism signature: {h:#018x} (scenes={}, egress={})",
        st.capture.scenes.len() as u64 + st.capture.retired_scenes,
        st.capture.egress.len()
    );
    // Phase timings (`VITASLOP_PERF=1`). A long boot's wall clock is the thing that
    // decides how far into a title anything can get, so the probe reports where it went.
    if vitaslop_runtime::perf::enabled() {
        eprintln!("--- perf phases (VITASLOP_PERF) ---");
        for ph in vitaslop_runtime::perf::Phase::all() {
            let (ns, calls, bytes) = vitaslop_runtime::perf::read(ph);
            if calls == 0 {
                continue;
            }
            eprintln!(
                "  {:38} {:>9.1} ms  {:>10} calls  {:>12} bytes",
                ph.label(),
                ns as f64 / 1e6,
                calls,
                bytes
            );
        }
    }
    // Clock calibration input: `QUANTUM_CPU_US` is one frame divided by the quanta a
    // steadily-rendering frame takes, so this ratio is what sets it. It also reads as a
    // health check - a run whose quanta hugely outnumber its flips is a title spinning
    // in guest code rather than rendering.
    {
        let (quanta, flips) = st.quantum_flip_counts();
        let per_frame = if flips > 0 { quanta as f64 / flips as f64 } else { f64::NAN };
        eprintln!(
            "clock: virtual_us={} quanta={quanta} flips={flips} quanta/flip={per_frame:.2}",
            st.now_us(),
        );
    }
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
    vitaslop_runtime::vita::dump_call_sites(80);
    eprintln!("--- sceKernelWaitLwCond samples (work, timeout_ptr, timeout) ---");
    for (work, tp, to) in &st.capture.lwcond_wait_samples {
        eprintln!("  work={work:#010x} timeout_ptr={tp:#010x} timeout={to}");
    }
    // Final preemptive-sync state: which threads are parked on which primitive (vs.
    // absent from every waiter list = spinning in pure compute). The decisive read for
    // a boot that stalls before its first frame - is the main thread blocked on a lock a
    // worker holds, or busy-waiting on a datum a worker never produces?
    eprintln!("--- final sync state ---\n{}", st.debug_sync_dump());
    // Scene introspection (VITASLOP_DUMP_SCENES=N): print the last N captured scenes'
    // structure - color surface, and per draw the primitive/index/vertex shape,
    // uniform count, attributes, and bound textures. This answers "the frame is
    // black: what is the game actually drawing?" without rendering anything.
    if let Some(n) = std::env::var("VITASLOP_DUMP_SCENES").ok().and_then(|s| s.parse::<usize>().ok()) {
        let total = st.capture.scenes.len();
        for (i, scene) in st.capture.scenes.iter().enumerate().skip(total.saturating_sub(n)) {
            match &scene.color {
                Some(c) => eprintln!(
                    "scene[{i}]: color fmt={:#x} {}x{} stride={} addr={:#010x}, {} draw(s)",
                    c.format, c.width, c.height, c.stride_pixels, c.data_addr,
                    scene.draws.len()
                ),
                None => eprintln!("scene[{i}]: NO color surface, {} draw(s)", scene.draws.len()),
            }
            for (j, d) in scene.draws.iter().enumerate() {
                let rs = &d.render_state;
                eprintln!(
                    "  draw[{j}]: prim={} idxfmt={} idxcount={} vstride={} vbytes={} uniforms={} depth_func={:#x} depth_write={:#x} cull={:#x} attrs={:?}",
                    d.primitive, d.index_format, d.index_count, d.vertex_stride,
                    d.vertices.len(), d.uniforms.len(), rs.front_depth_func, rs.front_depth_write, rs.cull_mode, d.attributes,
                );
                if !d.uniforms.is_empty() {
                    let n = d.uniforms.len().min(32);
                    eprintln!("    uniforms[..{n}]: {:?}", &d.uniforms[..n]);
                }
                for t in &d.textures {
                    eprintln!(
                        "    tex unit={} base_fmt={:#04x} swizzle={:#08x} type={:#x} {}x{} stride={} addr={:#010x} pixels={} minf={} magf={}",
                        t.unit, t.base_format, t.swizzle, t.tex_type, t.width, t.height, t.stride,
                        t.data_addr, t.pixels.len(), t.min_filter, t.mag_filter
                    );
                    // VITASLOP_DUMP_TEX: also decode the first few vertices as f32 lanes
                    // + trailing bytes, so the real attribute semantics (which lane is
                    // uv vs color) are visible.
                    if std::env::var("VITASLOP_DUMP_TEX").is_ok() {
                        let stride = d.vertex_stride.max(1) as usize;
                        let nv = (d.vertices.len() / stride).min(16);
                        for vi in 0..nv {
                            let b = &d.vertices[vi * stride..(vi + 1) * stride];
                            let lanes: Vec<String> = (0..stride / 4)
                                .map(|k| {
                                    let o = k * 4;
                                    let f = f32::from_le_bytes([b[o], b[o + 1], b[o + 2], b[o + 3]]);
                                    format!("{f:.3}")
                                })
                                .collect();
                            let bytes: Vec<String> = b.iter().map(|x| format!("{x:02x}")).collect();
                            eprintln!("      v{vi} f32=[{}] bytes=[{}]", lanes.join(" "), bytes.join(" "));
                        }
                    }
                    // VITASLOP_DUMP_TEX: write each bound texture's decoded RGBA8 + raw
                    // bytes to the shot dir, so the atlas can be inspected directly.
                    if std::env::var("VITASLOP_DUMP_TEX").is_ok() {
                        if let Ok(dir) = std::env::var("VITASLOP_SHOT_DIR") {
                            let (tw, th, rgba) = render::decode_texture_rgba8(t);
                            let raw = format!("{dir}/tex_s{i}_d{j}_{tw}x{th}.rgba");
                            let _ = std::fs::write(&raw, &rgba);
                            let g = format!("{dir}/tex_s{i}_d{j}_{}x{}.gray", t.width, t.height);
                            let _ = std::fs::write(&g, &t.pixels);
                        }
                    }
                }
            }
        }
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
            // Rendering + PNG-encoding every scene is the dominant per-run cost, but when
            // iterating on input timing only the OUTCOME frames matter. `VITASLOP_SHOT_LAST=K`
            // renders just the last K scenes (keeping their real frame indices), so a
            // tuning run that only needs the final PASS/FAIL screen is a few PNGs, not
            // thousands. Unset = render all (full-fidelity diagnostic).
            let shot_last: Option<usize> = std::env::var("VITASLOP_SHOT_LAST").ok().and_then(|s| s.parse().ok());
            let total = st.capture.scenes.len();
            let start = shot_last.map(|k| total.saturating_sub(k)).unwrap_or(0);
            let mut frames_written = 0;
            for (i, scene) in st.capture.scenes.iter().enumerate().skip(start) {
                let fb = render::render_scene(scene, WIDTH, HEIGHT, CLEAR);
                std::fs::write(format!("{dir}/frame_{i:04}.png"), fb.to_png()).expect("write png");
                frames_written += 1;
            }
            eprintln!("wrote {frames_written} frame PNG(s) (of {total}) + trace to {dir}/");
            // VITASLOP_DUMP_DRAW=i,j,k prints those scene-draw indices' raw vertex format
            // (attributes, stride, primitive) and the first few vertices as f32 + f16 + u8
            // lanes - to diagnose a mesh that decodes to scatter (wrong position attr,
            // multi-stream, or a half-float layout the decoder mis-reads).
            if let Ok(list) = std::env::var("VITASLOP_DUMP_DRAW") {
                if let Some(scene) = st.capture.scenes.last() {
                    let h2f = |h: u16| -> f32 {
                        let s = (h >> 15) & 1;
                        let e = ((h >> 10) & 0x1f) as i32;
                        let m = (h & 0x3ff) as f32;
                        let v = if e == 0 { m * 2f32.powi(-24) } else { (m / 1024.0 + 1.0) * 2f32.powi(e - 15) };
                        if s == 1 { -v } else { v }
                    };
                    for idx in list.split(',').filter_map(|s| s.trim().parse::<usize>().ok()) {
                        let Some(d) = scene.draws.get(idx) else { continue };
                        eprintln!(
                            "DRAW {idx}: prim={:#010x} idxfmt={} idxcount={} stride={} nverts={} cull={:#x} two_sided={:#x} depth_write={:#x} depth_func={:#x}",
                            d.primitive, d.index_format, d.index_count, d.vertex_stride,
                            d.vertices.len() / d.vertex_stride.max(1) as usize,
                            d.render_state.cull_mode, d.render_state.two_sided,
                            d.render_state.front_depth_write, d.render_state.front_depth_func
                        );
                        for a in &d.attributes {
                            eprintln!("   attr stream={} off={} fmt={} comps={} reg={}",
                                a.stream_index, a.offset, a.format, a.component_count, a.reg_index);
                        }
                        let stride = d.vertex_stride.max(1) as usize;
                        for vi in 0..4.min(d.vertices.len() / stride) {
                            let b = &d.vertices[vi * stride..(vi + 1) * stride];
                            let f32s: Vec<String> = (0..stride / 4)
                                .map(|k| format!("{:.2}", f32::from_le_bytes([b[k*4], b[k*4+1], b[k*4+2], b[k*4+3]])))
                                .collect();
                            let f16s: Vec<String> = (0..stride / 2)
                                .map(|k| format!("{:.2}", h2f(u16::from_le_bytes([b[k*2], b[k*2+1]]))))
                                .collect();
                            eprintln!("   v{vi}: bytes={:02x?}", b);
                            eprintln!("        f32={f32s:?}");
                            eprintln!("        f16={f16s:?}");
                        }
                    }
                }
            }
            // VITASLOP_DUMP_RENDERSCENE prints, for the last captured scene, how the GPU
            // builder classifies each draw (space / opaque / exposure / textured) plus, for
            // MVP draws, the transformed NDC-z range and on-screen vertex count - the data
            // that tells us why an opaque 3D draw might render on the software oracle but
            // vanish on the GPU (wrong pipeline, off-screen, or out-of-range depth).
            if std::env::var("VITASLOP_DUMP_RENDERSCENE").is_ok() {
                if let Some(scene) = st.capture.scenes.last() {
                    let rs = vitaslop_runtime::render::RenderSceneBuilder::new().build(scene);
                    eprintln!("RENDERSCENE last scene: {} draws", rs.draws.len());
                    let lanef = |b: &[u8], o: usize| f32::from_le_bytes([b[o], b[o + 1], b[o + 2], b[o + 3]]);
                    for (di, d) in rs.draws.iter().enumerate() {
                        let (sp, mvp) = match d.space {
                            vitaslop_platform::gpu::DrawSpace::Mvp(m) => ("Mvp", Some(m)),
                            vitaslop_platform::gpu::DrawSpace::Ndc => ("Ndc", None),
                            vitaslop_platform::gpu::DrawSpace::Pixel => ("Pixel", None),
                        };
                        let mut extra = String::new();
                        if let Some(m) = mvp {
                            let nv = d.vertices.len() / 24;
                            let (mut zmin, mut zmax) = (f32::INFINITY, f32::NEG_INFINITY);
                            let (mut wpos, mut onscreen) = (0usize, 0usize);
                            for i in 0..nv {
                                let o = i * 24;
                                let (x, y, z) = (lanef(&d.vertices, o), lanef(&d.vertices, o + 4), lanef(&d.vertices, o + 8));
                                let cx = m[0] * x + m[4] * y + m[8] * z + m[12];
                                let cy = m[1] * x + m[5] * y + m[9] * z + m[13];
                                let cz = m[2] * x + m[6] * y + m[10] * z + m[14];
                                let cw = m[3] * x + m[7] * y + m[11] * z + m[15];
                                if cw > 0.0 {
                                    wpos += 1;
                                    let (nx, ny, nz) = (cx / cw, cy / cw, cz / cw);
                                    zmin = zmin.min(nz);
                                    zmax = zmax.max(nz);
                                    if nx.abs() <= 1.0 && ny.abs() <= 1.0 {
                                        onscreen += 1;
                                    }
                                }
                            }
                            extra = format!("nv={nv} w>0={wpos} onscreen={onscreen} ndcz=[{zmin:.3},{zmax:.3}]");
                        }
                        // First vertex color + uv, and the decoded texture's dims + center texel.
                        let vc = if d.vertices.len() >= 24 {
                            format!("vcol0=[{},{},{},{}] uv0=({:.2},{:.2})",
                                d.vertices[20], d.vertices[21], d.vertices[22], d.vertices[23],
                                lanef(&d.vertices, 12), lanef(&d.vertices, 16))
                        } else { String::new() };
                        let tx = match &d.texture {
                            Some(t) => {
                                let n = t.rgba.len();
                                let c = (n / 2) & !3;
                                format!("tex={}x{} centerTexel=[{},{},{},{}]",
                                    t.width, t.height,
                                    t.rgba.get(c).copied().unwrap_or(0), t.rgba.get(c+1).copied().unwrap_or(0),
                                    t.rgba.get(c+2).copied().unwrap_or(0), t.rgba.get(c+3).copied().unwrap_or(0))
                            }
                            None => "tex=none".into(),
                        };
                        eprintln!(
                            "  [{di:3}] {sp:5} opaque={} exp={:.2} idx={} {extra} {vc} {tx}",
                            d.opaque, d.exposure, d.index_count
                        );
                    }
                }
            }
            // VITASLOP_GPU also renders each scene through the general GXM->WebGPU
            // renderer (the browser's real path) to frame_gpu_XXXX.png, so the GPU
            // output can be compared to the software oracle on the real title's frames.
            if std::env::var("VITASLOP_GPU").is_ok() {
                match vitaslop_native::GeneralRenderer::new() {
                    Some(mut gpu) => {
                        eprintln!("GPU render via {}", gpu.adapter_name);
                        // Honor VITASLOP_SHOT_LAST here too (keeping real frame indices), so
                        // a tuning run renders a few GPU frames, not thousands.
                        let mut gpu_written = 0;
                        for (i, scene) in st.capture.scenes.iter().enumerate().skip(start) {
                            let fb = gpu.render_scene(scene, WIDTH, HEIGHT, CLEAR);
                            // Mean per-channel abs diff vs the software oracle for this scene,
                            // so GPU/oracle agreement on the real title's frames is a number,
                            // not just an eyeball of the two PNGs.
                            let sw = render::render_scene(scene, WIDTH, HEIGHT, CLEAR);
                            let sum: u64 = sw
                                .rgba
                                .iter()
                                .zip(&fb.rgba)
                                .map(|(a, b)| (*a as i32 - *b as i32).unsigned_abs() as u64)
                                .sum();
                            let mean = sum as f64 / sw.rgba.len() as f64;
                            eprintln!("frame {i}: mean_abs_diff sw-vs-gpu = {mean:.3}");
                            std::fs::write(format!("{dir}/frame_gpu_{i:04}.png"), fb.to_png())
                                .expect("write gpu png");
                            gpu_written += 1;
                        }
                        eprintln!("wrote {gpu_written} GPU frame PNG(s) (of {total}) to {dir}/");
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
