//! The retail-title desktop path: load an extracted Vita game directory, run it LIVE
//! in a native window, and play it with the keyboard + mouse. This is the native twin
//! of the browser's `run_game` - the same decrypt -> link -> transpile -> preemptive
//! scheduler -> general GXM->WebGPU render pipeline, wrapped in a winit window instead
//! of a canvas. The guest is stepped one display frame per 1/60 s of wall time with
//! real input injected through the `World` seam, so menus (touch, via the mouse) and
//! gameplay (buttons, via the keyboard/gamepad) reach the guest exactly as on hardware.

use std::path::{Path, PathBuf};
use std::sync::{Arc, Mutex};
use std::time::{Duration, Instant};

use vitaslop_native::{RunReport, ThreadedScheduler};
use vitaslop_platform::gpu::{GxmRenderer, DEPTH_FORMAT};
use vitaslop_runtime::capture::Scene;
use vitaslop_runtime::ingest::pipeline::decrypt_container;
use vitaslop_runtime::ingest::vfs::MemVfs;
use vitaslop_runtime::link::link;
use vitaslop_runtime::render::RenderSceneBuilder;
use vitaslop_runtime::{CtrlFrame, TouchFrame, VitaEnv, World};
use vitaslop_loader as loader;

use pollster::block_on;
use winit::application::ApplicationHandler;
use winit::dpi::LogicalSize;
use winit::event::{ElementState, MouseButton, WindowEvent};
use winit::event_loop::{ActiveEventLoop, ControlFlow, EventLoop};
use winit::keyboard::{KeyCode, PhysicalKey};
use winit::window::{Window, WindowId};

use crate::input::Input;

/// The Vita display resolution the guest draws in (pixel-space 2D coords are in this
/// space); the window opens here and the general renderer projects against it. The
/// front touch panel is twice this in each axis.
const GAME_W: u32 = 960;
const GAME_H: u32 = 544;
const PANEL_SCALE: f32 = 2.0;

/// Background clear color, matching the software oracle and the browser.
const CLEAR: [u8; 4] = [16, 16, 24, 255];

/// Step one guest frame per 1/60 s of wall time (60 Hz), regardless of the monitor's
/// refresh, so the game runs at its intended speed.
const FRAME_DT: Duration = Duration::from_micros(16_666);

/// Scheduler quantum + per-frame round cap. The quantum matches the retail boot probe
/// (a fuel slice large enough that most between-host-call work finishes in one slice);
/// the per-frame round cap is generous - the first frame runs the whole boot.
const QUANTUM_FUEL: u64 = vitaslop_runtime::host::QUANTUM_FUEL;
const PER_FRAME_ROUNDS: u64 = 60_000_000;

/// The live input the window writes and the guest's world reads: the merged controller
/// frame plus the current mouse-as-touch (front panel). Shared behind a mutex because
/// `World` is `Send` (single-threaded here, so it never contends).
#[derive(Clone, Copy, Default)]
pub struct DesktopInput {
    pub ctrl: CtrlFrame,
    pub touch: Option<TouchFrame>,
}

pub type SharedInput = Arc<Mutex<DesktopInput>>;

/// The world the retail guest polls: a virtual 60 Hz clock, a seeded PRNG, and input
/// drawn from the shared [`DesktopInput`] cell the window updates each frame. An
/// optional [`RecipeWorld`] overlays a scripted TAS recipe on top of live input, so a
/// recorded playthrough replays in the window while the user watches (and can still
/// nudge with the keyboard/mouse) - the native twin of the browser's `BrowserWorld`.
struct DesktopWorld {
    input: SharedInput,
    /// A scripted recipe driving input as a function of frame, if `--recipe` was given.
    /// Frame-keyed, so it replays identically on desktop, browser, and headless.
    recipe: Option<vitaslop_runtime::RecipeWorld>,
    monotonic_us: u64,
    wall_us: u64,
    rng: u64,
}

impl DesktopWorld {
    fn new(input: SharedInput) -> Self {
        DesktopWorld {
            input,
            recipe: None,
            monotonic_us: 0,
            wall_us: 1_500_000_000_000_000,
            rng: 0x9E37_79B9_7F4A_7C15,
        }
    }

    /// With a scripted recipe overlaid on the live input.
    fn with_recipe(input: SharedInput, recipe: vitaslop_runtime::RecipeWorld) -> Self {
        let mut w = DesktopWorld::new(input);
        w.recipe = Some(recipe);
        w
    }
}

impl World for DesktopWorld {
    fn monotonic_us(&mut self) -> u64 {
        self.monotonic_us
    }
    fn wall_us(&mut self) -> u64 {
        self.wall_us
    }
    fn poll_ctrl(&mut self, port: u32) -> CtrlFrame {
        let live = self.input.lock().unwrap().ctrl;
        let Some(r) = self.recipe.as_mut() else { return live };
        // Recipe drives; live input is additive so the watcher can nudge. Buttons OR
        // together; a stick the recipe leaves centered falls back to the live stick.
        let scripted = r.poll_ctrl(port);
        CtrlFrame {
            buttons: scripted.buttons | live.buttons,
            lx: if scripted.lx != 128 { scripted.lx } else { live.lx },
            ly: if scripted.ly != 128 { scripted.ly } else { live.ly },
            rx: if scripted.rx != 128 { scripted.rx } else { live.rx },
            ry: if scripted.ry != 128 { scripted.ry } else { live.ry },
        }
    }
    fn poll_touch(&mut self, port: u32) -> TouchFrame {
        if port != 0 {
            return TouchFrame::default();
        }
        let live = self.input.lock().unwrap().touch.unwrap_or_default();
        let Some(r) = self.recipe.as_mut() else { return live };
        // Prefer the recipe's scripted finger; fall back to the live mouse-as-touch.
        let scripted = r.poll_touch(port);
        if scripted.count > 0 {
            scripted
        } else {
            live
        }
    }
    fn fill_random(&mut self, buf: &mut [u8]) {
        // Prefer the recipe's deterministic PRNG when replaying, so a scripted run is
        // reproducible; otherwise SplitMix64, matching the runtime worlds.
        if let Some(r) = self.recipe.as_mut() {
            r.fill_random(buf);
            return;
        }
        for chunk in buf.chunks_mut(8) {
            self.rng = self.rng.wrapping_add(0x9E37_79B9_7F4A_7C15);
            let mut z = self.rng;
            z = (z ^ (z >> 30)).wrapping_mul(0xBF58_476D_1CE4_E5B9);
            z = (z ^ (z >> 27)).wrapping_mul(0x94D0_49BB_1331_11EB);
            z ^= z >> 31;
            for (i, b) in chunk.iter_mut().enumerate() {
                *b = (z >> (i * 8)) as u8;
            }
        }
    }
    fn set_frame(&mut self, frame: u64) {
        if let Some(r) = self.recipe.as_mut() {
            r.set_frame(frame);
        }
        self.monotonic_us = frame.wrapping_mul(16_666);
        self.wall_us = 1_500_000_000_000_000u64.wrapping_add(self.monotonic_us);
    }
}

/// Recursively collect every file under `root` as `(forward-slash relative path, bytes)`
/// - the shape `decrypt_container` expects for the extracted app directory.
fn read_dir_files(root: &Path) -> Result<Vec<(String, Vec<u8>)>, String> {
    let mut out = Vec::new();
    let mut stack = vec![root.to_path_buf()];
    while let Some(dir) = stack.pop() {
        let entries = std::fs::read_dir(&dir).map_err(|e| format!("read {}: {e}", dir.display()))?;
        for entry in entries {
            let entry = entry.map_err(|e| format!("dir entry: {e}"))?;
            let path = entry.path();
            let ty = entry.file_type().map_err(|e| format!("file type: {e}"))?;
            if ty.is_dir() {
                stack.push(path);
            } else {
                let rel = path
                    .strip_prefix(root)
                    .map_err(|e| format!("strip prefix: {e}"))?
                    .components()
                    .map(|c| c.as_os_str().to_string_lossy())
                    .collect::<Vec<_>>()
                    .join("/");
                let bytes = std::fs::read(&path).map_err(|e| format!("read {}: {e}", path.display()))?;
                out.push((rel, bytes));
            }
        }
    }
    Ok(out)
}

/// The retail guest, stepped live on the preemptive scheduler. Owns the scheduler and
/// tracks the newest presented frame and whether the run has ended.
pub struct RetailGuest {
    sched: ThreadedScheduler<VitaEnv>,
    /// Every scene of the newest presented frame, in submission order.
    scenes: Vec<Scene>,
    finished: bool,
    err: Option<String>,
    /// The report that ENDED the run, when one did.
    ///
    /// Without this a run that stops early prints only the frame it reached, and every
    /// terminal outcome looks identical to "the target was reached": a deadlock, a thread
    /// exiting, a round budget running out and a clean finish all produce the same line.
    /// MEASURED cost of not having it: a retail headless run that stops at frame 1 was
    /// read as a clock pathology and worked around three times before anyone asked what
    /// actually ended it.
    ended_by: Option<String>,
    /// Decrypt + link + transpile + instantiate time, measured once at construction.
    pub build_ms: f64,
}

impl RetailGuest {
    /// Load, decrypt, link, transpile, and instantiate the title in `dir` for live
    /// execution over the shared `input` cell. An optional scripted `recipe` (a
    /// frame-keyed TAS text) is overlaid on the live input so a recorded playthrough
    /// replays in the window.
    pub fn new(dir: &Path, input: SharedInput, recipe: Option<&str>) -> Result<RetailGuest, String> {
        let t0 = Instant::now();
        let files = read_dir_files(dir)?;
        let mut vfs = MemVfs::new();
        for (path, bytes) in files {
            vfs.insert(path, bytes);
        }
        let game = decrypt_container(&mut vfs).map_err(|e| format!("decrypt: {e:?}"))?;
        let modules = game
            .modules
            .iter()
            .map(|m| loader::load(&m.elf))
            .collect::<Result<Vec<_>, _>>()
            .map_err(|e| format!("load module: {e:?}"))?;
        let linked = link(modules).map_err(|e| format!("link: {e:?}"))?;

        let world: Box<dyn World + Send> = match recipe {
            Some(text) => {
                let r = vitaslop_runtime::RecipeWorld::parse(text)
                    .map_err(|e| format!("recipe parse: {e}"))?;
                Box::new(DesktopWorld::with_recipe(input, r))
            }
            None => Box::new(DesktopWorld::new(input)),
        };
        let mut env = VitaEnv::new(linked.imports.clone(), linked.base, linked.mem_bytes, world);
        // Movie playback decodes on this machine's own H.264 decoder. Installing the port
        // here rather than inside the engine is what keeps the engine free of a decoder:
        // the browser build installs the WebCodecs-backed one through the same seam.
        env.state.video = Box::new(vitaslop_platform::video::H264Factory);
        env.state.audio_dec = Box::new(vitaslop_platform::audio_dec::AacFactory);
        env.state.set_alloc_base(linked.alloc_base);
        env.state.set_process_param(linked.process_param);
        env.state.set_modules(linked.loaded_modules.clone());
        env.state.set_tls_template(linked.tls_template);
        env.state.set_preemptive(true);
        // Move (not clone) the decrypted assets into the guest filesystem - for a
        // large 3D title this is hundreds of megabytes.
        for (path, bytes) in game.files.into_files() {
            env.state.add_file(&path, bytes);
        }

        let (mut sched, _stubs) = ThreadedScheduler::from_linked(&linked, env, QUANTUM_FUEL)
            .map_err(|e| format!("scheduler: {e:?}"))?;
        // >>> NO DETERMINISM SIGNATURE unless something will read it. The fold hashes every
        // retired scene's vertices, indices and uniforms - 3.5 MB a frame on a race, MEASURED
        // at 8.0% of a desktop frame - and this path never prints or compares one. The
        // BROWSER has been gated since 2026-08-19e; the desktop was not, so `--headless`
        // renders (the GPU oracle every render decision here is judged by) were paying it.
        // `VITASLOP_SIGNATURE=1` asks for one back, the same spelling both other engines use.
        sched
            .host()
            .state
            .capture
            .set_signature_wanted(vitaslop_runtime::knobs::flag("VITASLOP_SIGNATURE"));
        let build_ms = t0.elapsed().as_secs_f64() * 1000.0;
        Ok(RetailGuest { sched, scenes: Vec::new(), finished: false, err: None, ended_by: None, build_ms })
    }

    /// Step the guest one display frame. Keeps the newest captured scene and drops the
    /// rest (render-to-texture intermediates); marks the run finished on end/trap.
    pub fn advance(&mut self) {
        if self.finished {
            return;
        }
        let target = self.sched.frames() + 1;
        let report = self.sched.run_frames(target, PER_FRAME_ROUNDS);
        {
            let mut host = self.sched.host();
            let cap = &mut host.state.capture;
            // Diagnostic (VITASLOP_DUMP_SCENES): a frame can contain several GXM scenes
            // (render-to-texture passes plus the display pass). Only the last is kept, so
            // knowing how many there were - and how big each is - is the difference between
            // "the geometry is missing" and "the geometry is in a pass we discard".
            if std::env::var_os("VITASLOP_DUMP_SCENES").is_some() && !cap.scenes.is_empty() {
                let shape: Vec<String> = cap
                    .scenes
                    .iter()
                    .map(|s| match &s.color {
                        Some(c) => format!("{}draws@{}x{}", s.draws.len(), c.width, c.height),
                        None => format!("{}draws@no-surface", s.draws.len()),
                    })
                    .collect();
                eprintln!("SCENES frame={}: [{}]", self.sched.frames(), shape.join(", "));
            }
            // Keep the WHOLE frame, not just its last scene. A 3D title renders its world
            // (and shadow maps, reflections, post chains) into offscreen surfaces and then
            // composites them; keeping only the last scene keeps only the composite, and
            // the world never gets drawn at all - a retail racer's race was a live HUD over
            // black for exactly this reason. An empty list means the guest submitted no
            // scene this frame, in which case the previous frame's stays on screen.
            if !cap.scenes.is_empty() {
                self.scenes = std::mem::take(&mut cap.scenes);
            }
            cap.scenes.clear();
            cap.trace.clear();
            cap.trace_thid.clear();
            cap.presents.clear();
        }
        match report {
            RunReport::FramesReached(_) => {}
            RunReport::Error(e) => {
                self.finished = true;
                self.err = Some(e);
            }
            other => {
                self.finished = true;
                self.ended_by = Some(format!("{other:?}"));
            }
        }
    }

    /// The guest's OWN diagnostic output: everything it has written to fd 1/2 this run.
    ///
    /// Retail titles ship a lot of their own logging, and it is the developer's account of
    /// what the game thinks is happening - which beats reverse-engineering the answer, and
    /// has already identified a hang on another title in one line. The engine has been
    /// capturing this all along with nothing in the headless path to read it.
    /// The guest's virtual clock, in microseconds.
    ///
    /// Worth reporting next to the frame count: a title that paces its simulation off the
    /// wall clock but caps how far it will step per frame goes WRONG QUIETLY when this
    /// runs fast - the race timer counts up several times too quickly while the car
    /// crawls, and neither symptom names the clock. One frame is 1/60 s, so the ratio of
    /// this to `frames / 60` should be 1.
    pub fn clock_us(&mut self) -> u64 {
        self.sched.host().state.now_us()
    }

    /// `(from quanta, from the frame top-up, from idle jumps)` microseconds of game clock.
    /// Printed next to the clock so a cross-engine comparison can see WHICH source
    /// differs, not merely that the totals do.
    pub fn clock_sources(&mut self) -> (u64, u64, u64) {
        self.sched.host().state.clock_sources()
    }

    /// Seconds of sound the guest has submitted through `sceAudioOutOutput`.
    pub fn audio_produced_seconds(&mut self) -> f64 {
        self.sched.host().state.audio_produced_seconds()
    }

    /// Every LIVE thread that is parked at the end of the run, with the wait it is parked in.
    ///
    /// The counterpart of [`idle_attribution`](Self::idle_attribution), which accounts only for
    /// waits that BOUGHT clock: a thread parked on something untimed buys none, so it is absent
    /// there and present here. Printed beside it so a desktop run and a browser run - whose
    /// panel prints the same list from the same runtime function - can be diffed thread for
    /// thread, which is how a browser-ONLY stall gets named.
    pub fn blocked_threads(&mut self) -> String {
        let v = self.sched.host().state.blocked_threads();
        if v.is_empty() {
            return String::new();
        }
        let mut s = format!("headless: {} thread(s) parked at the end of the run:
", v.len());
        for (thid, name, state) in &v {
            s.push_str(&format!("  thid {thid:#x} {name:?}: {state}
"));
        }
        s
    }

    /// Where the IDLE part of the game clock went, largest owner first. See
    /// `VitaState::idle_attribution` for why a total is not enough.
    pub fn idle_attribution(&mut self) -> String {
        let v = self.sched.host().state.idle_attribution();
        if v.is_empty() {
            return String::new();
        }
        let total: u64 = v.iter().map(|(_, us, _)| us).sum();
        let mut s = format!("headless: the idle clock ({:.1}s) was bought by:\n", total as f64 / 1e6);
        for (owner, us, jumps) in v.iter().take(8) {
            s.push_str(&format!(
                "  {:>8.3}s over {jumps:>8} jump(s), mean {:>7.1}us - {} on thread {:#x}\n",
                *us as f64 / 1e6,
                *us as f64 / jumps.max(&1).to_owned() as f64,
                owner.kind.name(),
                owner.thid,
            ));
        }
        s
    }

    /// Who got the CPU, and how much of the device's parallelism the run used. The two
    /// belong together: a lopsided share is only a problem if the starved threads were
    /// READY, and the second report is what says so.
    pub fn scheduler_report(&self) -> String {
        format!("{}{}", self.sched.cpu_share_report(), self.sched.runnable_report())
    }

    /// One line per guest memory space (`sceClibMspace*`): how full it is and whether the
    /// title is DRAINING it. See `vitaslop_runtime::mspace::Mspace`.
    pub fn mspace_report(&mut self) -> Vec<String> {
        self.sched.host().state.mspace_report()
    }

    pub fn guest_stdout(&mut self) -> Vec<u8> {
        self.sched.host().state.capture.stdout.clone()
    }

    /// The size of the buffer the guest last handed `sceDisplaySetFrameBuf`, which is what a
    /// frame must be RENDERED at. It is not always the panel: a title may render its front
    /// end smaller and let the display controller stretch it, and composing such a frame at
    /// the panel size puts the picture in a corner. See `VitaState::display_size`.
    pub fn display_size(&mut self) -> (u32, u32) {
        self.sched.host().state.display_size()
    }

    /// The newest presented frame's scenes, in submission order; empty until the guest
    /// has flipped once.
    pub fn current(&self) -> &[Scene] {
        &self.scenes
    }
    pub fn finished(&self) -> bool {
        self.finished
    }
    pub fn frames(&self) -> u64 {
        self.sched.frames()
    }

    /// `(total fuel burned, samples, largest single burn)` - the accounting behind the
    /// game clock's CPU charge. See [`vitaslop_runtime::sched::SchedCore::fuel_report`].
    pub fn fuel_report(&self) -> (u64, u64, u64) {
        self.sched.fuel_report()
    }
    pub fn error(&self) -> Option<&str> {
        self.err.as_deref()
    }
    /// Why the run ended, when it ended before the frame target. See [`Self::ended_by`].
    pub fn ended_by(&self) -> Option<&str> {
        self.ended_by.as_deref()
    }

    /// Guest memory at `addr`, for `VITASLOP_PEEK`.
    pub fn peek(&self, addr: u32, len: usize) -> Vec<u8> {
        self.sched.read_guest(addr, len)
    }
}

/// `VITASLOP_PEEK=<hex addr>[/<dec off>]*:<len>[,...]`: hex-dump guest memory at the END of a
/// headless run, as words with their addresses.
///
/// The alternative is a store watchpoint, and the two answer different questions. A watchpoint
/// says WHO wrote a word; this says WHAT IS THERE, which is what you need when the suspicion is
/// about the neighbours - "is this a table of records, and does the one next door hold the value
/// this draw should have had". A watchpoint cannot show you a record nobody wrote.
///
/// # The `/off` hops, and why they are not a convenience
/// What is actually being looked for is almost never at a fixed address: it is at the end of a
/// POINTER CHAIN rooted in one. `0x816F07B8/0/4` reads the word at `0x816F07B8+0`, then the word
/// at `that+4`, and dumps from there. Without the hops each link costs its own headless replay -
/// the same run, three times, to walk two pointers - and the addresses in between are heap, so
/// they move and the runs cannot even be mixed.
///
/// Every hop is PRINTED, and a hop through a null or unmapped word STOPS and says so, because
/// the failure it replaces is silent: `read_guest` on a bad address returns zeros, so a broken
/// chain otherwise hex-dumps a screenful of `00000000` that reads exactly like a real region
/// nobody has written yet.
fn peek_regions(guest: &RetailGuest) {
    let Ok(spec) = std::env::var("VITASLOP_PEEK") else { return };
    for part in spec.split(',') {
        let Some((a, n)) = part.trim().rsplit_once(':') else { continue };
        let mut hops = a.trim().split('/');
        let root = hops.next().unwrap_or_default();
        let (Ok(mut addr), Ok(len)) = (
            u32::from_str_radix(root.trim().trim_start_matches("0x"), 16),
            n.trim().parse::<usize>(),
        ) else {
            println!("peek: {part:?} is not <hex addr>[/<dec off>]*:<len>");
            continue;
        };
        // Walk the chain, reporting each link. `continue 'part` is spelled as a flag because
        // the dump below is shared with the no-hop case.
        let mut broken = false;
        for hop in hops {
            let Ok(off) = hop.trim().parse::<u32>() else {
                println!("peek: {hop:?} is not a decimal offset in {part:?}");
                broken = true;
                break;
            };
            let at = addr.wrapping_add(off);
            let w = guest.peek(at, 4);
            let next = u32::from_le_bytes([w[0], w[1], w[2], w[3]]);
            println!("peek hop: [{at:#010x}] = {next:#010x}");
            if next == 0 {
                println!("peek: the chain in {part:?} hops through a NULL at {at:#010x} - stopping \
                          rather than dumping address 0, which reads as an unwritten region.");
                broken = true;
                break;
            }
            addr = next;
        }
        if broken {
            continue;
        }
        let bytes = guest.peek(addr, len);
        for (i, row) in bytes.chunks(32).enumerate() {
            let words: Vec<String> = row
                .chunks(4)
                .map(|c| {
                    let mut w = [0u8; 4];
                    w[..c.len()].copy_from_slice(c);
                    format!("{:08x}", u32::from_le_bytes(w))
                })
                .collect();
            println!("peek {:#010x}: {}", addr + i as u32 * 32, words.join(" "));
        }
    }
}

/// The window's GPU surface + the general GXM renderer. Presents a captured scene each
/// frame through the same `GxmRenderer` the browser canvas uses (so pixels match). The
/// guest draws in 960x544 game space; the renderer projects against that and fills the
/// (possibly larger) window, stretching to fit.
struct RetailGfx {
    surface: wgpu::Surface<'static>,
    device: wgpu::Device,
    queue: wgpu::Queue,
    config: wgpu::SurfaceConfiguration,
    gxm: GxmRenderer,
    builder: RenderSceneBuilder,
    depth: wgpu::TextureView,
    render_format: wgpu::TextureFormat,
    adapter_name: String,
}

impl RetailGfx {
    fn new(window: Arc<Window>) -> Result<RetailGfx, String> {
        let size = window.inner_size();
        let (w, h) = (size.width.max(1), size.height.max(1));

        let instance = wgpu::Instance::default();
        let surface = instance.create_surface(window).map_err(|e| format!("create_surface: {e}"))?;
        let adapter = block_on(instance.request_adapter(&wgpu::RequestAdapterOptions {
            power_preference: wgpu::PowerPreference::HighPerformance,
            force_fallback_adapter: false,
            compatible_surface: Some(&surface),
            apply_limit_buckets: false,
        }))
        .map_err(|_| "no GPU adapter for the window surface".to_string())?;
        let adapter_name = adapter.get_info().name;

        let (device, queue) = block_on(adapter.request_device(&wgpu::DeviceDescriptor {
            label: Some("vitaslop-desktop-gxm"),
            required_features: vitaslop_platform::gpu::wanted_features(&adapter),
            // Raise resolution-derived limits to the adapter's: a real title binds
            // textures past the 2048 downlevel floor (some titles have a ~2480px atlas).
            required_limits: wgpu::Limits::downlevel_defaults().using_resolution(adapter.limits()),
            experimental_features: wgpu::ExperimentalFeatures::disabled(),
            memory_hints: wgpu::MemoryHints::default(),
            trace: wgpu::Trace::Off,
        }))
        .map_err(|e| format!("request_device: {e}"))?;

        let caps = surface.get_capabilities(&adapter);
        let format = caps.formats[0];
        // The GXM color decode already yields final display-ready (sRGB-encoded) byte
        // values - the software oracle just writes them into an `Rgba8Unorm` PNG. If the
        // surface's preferred format is an sRGB variant (native Vulkan/DX usually prefer
        // `Bgra8UnormSrgb`), writing those straight values applies a SECOND sRGB encode
        // and washes the image out. Render through a non-sRGB view of the same surface so
        // the bytes land verbatim, matching the oracle. On WebGPU the preferred canvas
        // format is already non-sRGB, so this is a no-op there.
        let render_format = format.remove_srgb_suffix();
        let view_formats = if render_format == format { vec![] } else { vec![render_format] };
        let config = wgpu::SurfaceConfiguration {
            usage: wgpu::TextureUsages::RENDER_ATTACHMENT,
            format,
            color_space: wgpu::SurfaceColorSpace::Auto,
            width: w,
            height: h,
            present_mode: wgpu::PresentMode::Fifo,
            alpha_mode: caps.alpha_modes[0],
            view_formats,
            desired_maximum_frame_latency: 2,
        };
        surface.configure(&device, &config);

        let mut gxm = GxmRenderer::new(&device, &queue, render_format);
        // ONE sample by default, because that is what the guest asks for. The display buffer's
        // render target is created `SCE_GXM_MULTISAMPLE_NONE` on every title measured here -
        // the console composites the front buffer at one sample - and the antialiasing the
        // title DOES ask for now happens where it asked for it, on the render targets it
        // created multisampled (`gpu::gxm_sample_count`).
        //
        // This used to default to 2, and the desktop defaulted differently from the browser,
        // which is the shape of bug that makes a phone look broken next to a review shot. It
        // was also not doing what it was believed to do: MEASURED on the front end, a 2x
        // supersampled GAME MODE differs from a 1x one by 3.31% of pixels and a mean of
        // 0.20/255, and its text edges come out very slightly SOFTER (mean |dx| 1.807 against
        // 1.900). Supersampling the display was never the answer to a sharpness complaint.
        //
        // `VITASLOP_SSAA` survives as a review instrument - it is what the software oracle's
        // parity probe compares against - not as a setting anything should ship with.
        let ssaa = std::env::var("VITASLOP_SSAA").ok().and_then(|s| s.parse::<u32>().ok()).filter(|&n| n >= 1).unwrap_or(1);
        gxm.set_supersample(ssaa);
        let depth = make_depth(&device, w, h);
        Ok(RetailGfx { surface, device, queue, config, gxm, builder: RenderSceneBuilder::new(), depth, render_format, adapter_name })
    }

    fn resize(&mut self, w: u32, h: u32) {
        if w == 0 || h == 0 {
            return;
        }
        self.config.width = w;
        self.config.height = h;
        self.surface.configure(&self.device, &self.config);
        self.depth = make_depth(&self.device, w, h);
    }

    /// `display` is the size the GUEST declared to `sceDisplaySetFrameBuf`, which is what
    /// the frame is projected against. See the `encode_chain` call below.
    fn present(&mut self, scenes: &[Scene], display: (u32, u32)) {
        let built: Vec<_> = scenes.iter().map(|s| self.builder.build(s)).collect();
        let frame = match self.surface.get_current_texture() {
            wgpu::CurrentSurfaceTexture::Success(t) | wgpu::CurrentSurfaceTexture::Suboptimal(t) => t,
            _ => return,
        };
        let view = frame.texture.create_view(&wgpu::TextureViewDescriptor {
            format: Some(self.render_format),
            ..Default::default()
        });
        let mut encoder = self.device.create_command_encoder(&wgpu::CommandEncoderDescriptor { label: None });
        // Project against the resolution the GUEST declared (not the window size), so
        // pixel-space 2D coords map correctly; the window-sized framebuffer stretches the
        // resolution-independent clip output to fill the window. That stretch is also what
        // makes a title presenting a buffer SMALLER than the panel come out full-screen -
        // the hardware's own upscale, for free, on this path.
        let (dw, dh) = display;
        let (fw, fh) = (frame.texture.width(), frame.texture.height());
        self.gxm.encode_chain(&self.device, &self.queue, &mut encoder, &view, &self.depth, &built, dw, dh, fw, fh, CLEAR);
        self.queue.submit([encoder.finish()]);
        self.queue.present(frame);
    }
}

fn make_depth(device: &wgpu::Device, w: u32, h: u32) -> wgpu::TextureView {
    let tex = device.create_texture(&wgpu::TextureDescriptor {
        label: Some("depth"),
        size: wgpu::Extent3d { width: w, height: h, depth_or_array_layers: 1 },
        mip_level_count: 1,
        sample_count: 1,
        dimension: wgpu::TextureDimension::D2,
        format: DEPTH_FORMAT,
        usage: wgpu::TextureUsages::RENDER_ATTACHMENT,
        view_formats: &[],
    });
    tex.create_view(&Default::default())
}

/// Headless self-check of the retail path (NO window): load `dir`, optionally drive a
/// touch script through the shared input cell, and render the final scene via the
/// native GPU renderer (the same `GxmRenderer`) to `<shot_dir>/desktop.png`. This
/// exercises the whole load -> decrypt -> link -> transpile -> run -> input -> render
/// chain minus the winit window glue, so the desktop path is verifiable without a
/// display.
///
/// Env knobs (so one entry point serves any title without recompiling):
/// - `VITASLOP_HEADLESS_FRAMES` - display flips to run to (default 180). A title that
///   boots through intro/attract screens before its first interactive frame needs a
///   higher target.
/// - `VITASLOP_HEADLESS_RECIPE` - a frame-keyed TAS recipe to drive input with. When
///   set, the recipe supplies the touches/buttons; when unset, the built-in Tutorial
///   tap script runs (the historical title navigation), unless
///   `VITASLOP_HEADLESS_NO_TAPS` is set, which runs input-free to the title screen.
/// - `VITASLOP_HEADLESS_TIMING` - report per-frame GUEST cost (the emulated CPU work
///   behind one display flip) and the GPU cost of rendering the final captured scene.
/// - `VITASLOP_HEADLESS_SHOT_EVERY` - also write `<shot_dir>/fNNNNNN.png` every N display
///   flips, so one run shows the whole boot SEQUENCE rather than only its last frame.
/// - `VITASLOP_HEADLESS_SHOT_FROM` / `VITASLOP_HEADLESS_SHOT_TO` - restrict those shots to an
///   inclusive frame window (default: the whole run). A defect that lives in a TRANSITION -
///   a fade, a wipe, a one-frame flash - is invisible at any interval coarse enough to cover
///   a boot, and every-frame-from-boot is gigabytes of stored-deflate PNG. An empty window is
///   an ERROR rather than a run that quietly writes nothing.
///
/// NOTE what the timing does and does not measure. The guest advances every frame but the
/// scene is RENDERED ONCE, at the end - so a wall-clock total over a headless run is CPU
/// only, and dividing it by the frame count says nothing about the render. The two costs
/// are reported separately for that reason. Neither is a frame rate on its own: a title
/// sitting on a menu waiting for input costs almost nothing per frame, so a number
/// measured there is not a gameplay number.
/// Print the guest-CPU frame-cost distribution and the one-off render cost.
///
/// Reported as PERCENTILES over the whole run and separately over the last 300 frames,
/// because a headless run is two different workloads glued together: a boot sequence that
/// does heavy one-off work, then whatever screen it settles on. The median of the tail is
/// the closest thing here to "what a frame costs on this screen", and it is only a frame
/// RATE if the title is actually doing work on that screen rather than idling for input.
/// One rendered frame of the shot window: what it cost, what it drew, and what the encoder
/// had to CREATE for it. See [`report_hiccups`].
struct Hiccup {
    frame: u64,
    guest_ms: f64,
    render_ms: f64,
    scenes: usize,
    draws: usize,
    pipelines_built: u64,
    tex_uploaded: u64,
    tex_upload_bytes: u64,
    bind_groups_built: u64,
    buffer_bytes: u64,
}

impl Hiccup {
    fn total_ms(&self) -> f64 {
        self.guest_ms + self.render_ms
    }
}

/// Name the frames that HITCHED, with the two halves of each one's cost side by side.
///
/// # Why a ranked list and not another percentile
/// `report_frame_timing` already prints p50/p95/max over the run, and a hiccup is exactly what
/// that cannot describe: a distribution says how often an expensive frame happens, never which
/// one or why, and a max is one number with no context attached. A player reporting "little
/// hiccups" is reporting a handful of named frames, so this prints those frames -
/// [[vitaslop-a-range-is-not-a-distribution]] applies to a p95 as much as to a min/max.
///
/// Rows are ranked by TOTAL (guest + render) because that is what a frame costs, and both halves
/// are printed because they have opposite fixes. The scene and draw counts sit on the same row so
/// a frame that costs 5x while drawing 5x as much - which is work, not a hiccup - can be told
/// apart from one that costs 5x drawing the same thing, which is not.
///
/// The BASELINE is the median of the same window, not of the run: a window inside a race and a
/// window over a boot are different workloads, and a ratio against the wrong one is meaningless.
fn report_hiccups(rows: &[Hiccup]) {
    if rows.len() < 8 {
        // Under a handful of frames there is no baseline to be an outlier against, and ranking
        // three frames by cost is just printing them.
        return;
    }
    let mut totals: Vec<f64> = rows.iter().map(Hiccup::total_ms).collect();
    totals.sort_by(|a, b| a.partial_cmp(b).unwrap_or(std::cmp::Ordering::Equal));
    let median = totals[totals.len() / 2];
    let mut worst: Vec<&Hiccup> = rows.iter().collect();
    worst.sort_by(|a, b| b.total_ms().partial_cmp(&a.total_ms()).unwrap_or(std::cmp::Ordering::Equal));
    // How much of the window is spent in frames that cost more than twice the median - the one
    // number that says whether the hitches are a curiosity or the experience.
    let spikes: Vec<&&Hiccup> = worst.iter().filter(|r| r.total_ms() > 2.0 * median).collect();
    let window_ms: f64 = totals.iter().sum();
    let spike_ms: f64 = spikes.iter().map(|r| r.total_ms()).sum();
    println!(
        "hiccups: {} frames rendered in the shot window, median {median:.2} ms/frame; {} frames \
         cost MORE THAN 2x that and they are {:.1}% of the window's time",
        rows.len(),
        spikes.len(),
        100.0 * spike_ms / window_ms.max(1e-9),
    );
    println!(
        "hiccups: the 12 most expensive frames. `built` is what the encoder had to CREATE for \
         that frame - a pipeline, a texture upload, a bind group - and it is what separates a \
         hitch that will not repeat from one that will."
    );
    for r in worst.iter().take(12) {
        println!(
            "hiccups:   f{:06}  guest {:7.2} + render {:7.2} = {:7.2} ms ({:5.1}x median)  \
             {:2} scenes / {:4} draws  built: {} pipelines, {} textures ({:.2} MB), \
             {} bind groups, {:.2} MB buffers",
            r.frame,
            r.guest_ms,
            r.render_ms,
            r.total_ms(),
            r.total_ms() / median.max(1e-9),
            r.scenes,
            r.draws,
            r.pipelines_built,
            r.tex_uploaded,
            r.tex_upload_bytes as f64 / (1024.0 * 1024.0),
            r.bind_groups_built,
            r.buffer_bytes as f64 / (1024.0 * 1024.0),
        );
    }
    // And the same counters over the WHOLE window, so a per-frame row can be read as "this
    // frame's share" rather than as a number on its own.
    let (p, t, b) = rows.iter().fold((0u64, 0u64, 0u64), |a, r| {
        (a.0 + r.pipelines_built, a.1 + r.tex_uploaded, a.2 + r.bind_groups_built)
    });
    println!(
        "hiccups: over the whole window the encoder built {p} pipelines, uploaded {t} textures \
         and built {b} bind groups"
    );
    // ...and what those pipeline builds actually SPENT, split into the two halves that have
    // different fixes. A 2.5-second first frame is not actionable until it says which half.
    let (module_ms, create_ms) = vitaslop_platform::gpu::take_pipeline_build_split();
    let pre_ms = vitaslop_platform::gpu::take_precompile_ms();
    println!(
        "hiccups: the pipeline builds of the WHOLE RUN cost {module_ms:.0} ms compiling WGSL + \
         {create_ms:.0} ms creating pipelines, IN the frame that drew them, plus {pre_ms:.0} ms \
         compiling WGSL AHEAD of any draw (when the guest's shader patcher named the pair, which \
         is where the device does its shader work). The in-frame WGSL figure is what the \
         preparation failed to catch; the pipeline half also needs that draw's blend, depth, \
         cull, format and sample count, and cannot be moved without them."
    );
}

fn report_frame_timing(
    frame_ms: &[f64],
    cold_render_ms: f64,
    warm_render_ms: &[f64],
    split: vitaslop_native::RenderSplit,
    // Encode counters already taken per frame by the hiccup log, folded back in so this
    // report still covers the whole run.
    enc_window: &vitaslop_platform::gpu::EncodeWork,
) {
    if frame_ms.is_empty() {
        println!("timing: no frames advanced");
        return;
    }
    let pct = |v: &[f64], p: f64| -> f64 {
        let mut s = v.to_vec();
        s.sort_by(|a, b| a.partial_cmp(b).unwrap());
        s[(((s.len() - 1) as f64) * p).round() as usize]
    };
    let total: f64 = frame_ms.iter().sum();
    let tail = &frame_ms[frame_ms.len().saturating_sub(300)..];
    println!(
        "timing: guest CPU over {} frames - total {:.0} ms, p50 {:.2} ms, p95 {:.2} ms, max {:.2} ms",
        frame_ms.len(),
        total,
        pct(frame_ms, 0.50),
        pct(frame_ms, 0.95),
        pct(frame_ms, 1.0)
    );
    println!(
        "timing: guest CPU over the last {} frames - p50 {:.2} ms ({:.0} fps if CPU-bound), p95 {:.2} ms",
        tail.len(),
        pct(tail, 0.50),
        1000.0 / pct(tail, 0.50).max(1e-6),
        pct(tail, 0.95)
    );
    println!(
        "timing: GPU render of the final scene - cold {cold_render_ms:.1} ms (builds pipelines, \
         compiles shaders, uploads textures)"
    );
    if !warm_render_ms.is_empty() {
        let p50 = pct(warm_render_ms, 0.50);
        println!(
            "timing: GPU render of the final scene - warm p50 {:.2} ms ({:.0} fps if GPU-bound), \
             p95 {:.2} ms over {} samples",
            p50,
            1000.0 / p50.max(1e-6),
            pct(warm_render_ms, 0.95),
            warm_render_ms.len()
        );
    }
    // WHERE the warm render went. A total says the recompiler path is slower than the
    // fixed-function one; only the split says whether that is per-draw CPU work building GPU
    // objects or the GPU genuinely shading more, and those call for opposite fixes.
    let p = split.phases;
    println!(
        "timing: warm render split - build {:.2} ms (scene decode, CPU), encode {:.2} ms (CPU), \
         submit+wait {:.2} ms (contains the GPU)",
        split.build_ms, split.encode_ms, split.submit_ms
    );
    println!(
        "timing: encode phases - prepare {:.2} ms, upload {:.2} ms, pass {:.2} ms over \
         {} recompiled + {} fixed-function draws",
        p.prepare_ms, p.upload_ms, p.pass_ms, p.gxp_draws, p.fixed_draws
    );
    // ...and WHAT `build` did, in counts. The browser prints the same line from the same
    // counters, so "build costs 21 ms there and 1.6 ms here" can be answered by comparing
    // work instead of comparing two machines' clocks.
    let n = warm_render_ms.len().max(1) as u64;
    println!("timing: {}", vitaslop_runtime::render::take_build_work().line(n));
    // ...and what `encode` did, in the same units, for the same reason. `encode` is 84% of the
    // browser's render on a burst frame and the three phase timings do not say what is IN it -
    // upload volume and per-call boundary overhead live in the same phase and have opposite
    // fixes. The browser prints this identical line, which is what makes the two comparable.
    let mut enc = vitaslop_platform::gpu::take_encode_work();
    enc.add(enc_window);
    println!("timing: {}", enc.line(n));
    // ...and inside `prepare`, which is most of `encode`, WHERE. Only when asked for: the split
    // reads a clock six times a draw, which is affordable here and is not in the browser.
    //
    // This reader sees the WARM RE-RENDER LOOP only - the shot window drains the same counters
    // per frame for its own report. One scene rendered sixty times hits every cache, so read
    // this as the floor: the hashing and the arena copies are real per-frame costs and the
    // repack is priced at zero here in a way a moving frame's is not.
    let prep = vitaslop_platform::gpu::take_prepare_split();
    if !prep.is_empty() {
        println!("timing: warm re-render {}", prep.line(n));
    }
    println!("timing: {}", vitaslop_runtime::render::decode_by_format_line());
    let (bg_hit, bg_new) = vitaslop_platform::gpu::take_sampler_bg_counts();
    println!(
        "timing: sampler bind groups - {:.1} reused / {:.1} BUILT per render",
        bg_hit as f64 / n as f64,
        bg_new as f64 / n as f64,
    );
    // WHICH host calls the run made, when the profiler is on. The browser prints the same report
    // from the same counters; this path did not print it at all, so the one engine that can be
    // driven by a recipe in a loop could not answer "which NIDs" without going through the
    // interactive session. The only way to spend less at the host-call boundary is to cross it
    // fewer times, and that needs the ranking.
    if vitaslop_runtime::knobs::flag("VITASLOP_DBG_CALLSITES") {
        println!("{}", vitaslop_runtime::vita::call_sites_report(25));
    }
    // The honest caveat, printed rather than left to the reader: a guest that is waiting for
    // input does no work, so a low per-frame CPU cost on a menu is not a gameplay figure.
    println!(
        "timing: NOTE the CPU figures are whatever the title was doing on this screen - if it \
         was idling on a menu they are not a gameplay frame rate."
    );
}

/// How many extra renders of the captured scene to time for the warm figure. Enough to get a
/// stable median past the first-render pipeline/upload cost, cheap enough to always run.
const WARM_RENDER_SAMPLES: usize = 60;

pub fn headless_check(
    dir: PathBuf,
    shot_dir: PathBuf,
    // The `--recipe` flag's contents, if it was given. Taken as a parameter rather than read
    // from the environment here so the FLAG and the env knob land in the same place - the flag
    // used to be parsed by `main` and then never passed, which is silent and unfalsifiable
    // (see the comment at its parse site).
    flag_recipe: Option<String>,
) -> Result<(), String> {
    let input: SharedInput = Arc::new(Mutex::new(DesktopInput::default()));
    let env_recipe = std::env::var("VITASLOP_HEADLESS_RECIPE")
        .ok()
        .map(|p| std::fs::read_to_string(&p).map_err(|e| format!("read recipe {p}: {e}")))
        .transpose()?;
    // Both spellings exist and the notes use both. If they are BOTH given they must agree,
    // because silently preferring one is how a run ends up replaying a recipe nobody named.
    if let (Some(a), Some(b)) = (&flag_recipe, &env_recipe) {
        if a != b {
            return Err("both --recipe and VITASLOP_HEADLESS_RECIPE are set and they are \
                        different recipes - pass one, not two".into());
        }
    }
    let recipe = flag_recipe.or(env_recipe);
    // Say which way input is being driven, unconditionally. "Did the recipe apply" was
    // previously answerable only by recognising the final screenshot.
    println!(
        "headless: input is {}",
        if recipe.is_some() { "a RECIPE" } else { "the built-in tap script (no recipe given)" }
    );
    let mut guest = RetailGuest::new(&dir, input.clone(), recipe.as_deref())?;
    let target: u64 = std::env::var("VITASLOP_HEADLESS_FRAMES")
        .ok()
        .and_then(|s| s.parse().ok())
        .unwrap_or(180);
    println!("headless: loaded (build {:.0} ms), running to frame {target}...", guest.build_ms);

    // >>> THE STALL WATCHDOG (`VITASLOP_STALL_WATCHDOG=<seconds>`).
    //
    // Armed AFTER the load, because the load is legitimately tens of seconds (13 s of build on
    // one title) and no display frame flips during it - a watchdog armed before it would fire
    // on the transpiler every time. Everything from here on is supposed to be producing frames.
    //
    // Said out loud either way: "the run is under a 120 s watchdog" and "the run is not" are
    // both things the reader of a log needs, and only one of them is visible from an absence.
    match vitaslop_native::watchdog::spawn(&format!("{} / {}", dir.display(), match &recipe {
        Some(_) => "recipe",
        None => "built-in taps",
    }))? {
        true => println!(
            "headless: STALL WATCHDOG armed at {} s - if no frame flips for that long the run              dumps what the guest is calling and stops with exit {}.",
            vitaslop_native::watchdog::budget_secs()?.unwrap_or(0),
            vitaslop_native::watchdog::STALL_EXIT_CODE
        ),
        false => println!(
            "headless: no stall watchdog (set VITASLOP_STALL_WATCHDOG=<seconds> to arm one;              without it a guest that spins hangs this run forever)."
        ),
    }

    // The built-in Tutorial tap script (title menu navigation): used only when no
    // recipe is given and taps are not disabled. (active_from, active_until, x, y) - a
    // sticky tap held across a few frames then released.
    let use_builtin_taps = recipe.is_none() && std::env::var_os("VITASLOP_HEADLESS_NO_TAPS").is_none();
    let taps = [
        (12u64, 19u64, 450u16, 674u16),
        (30, 37, 230, 230),
        (80, 87, 930, 674),
        (112, 121, 1620, 376),
        (128, 137, 630, 870),
    ];
    // Per-frame guest cost, in milliseconds, when timing is on. Kept per frame rather than
    // as a running total so the report can separate the typical frame from the worst one - a
    // mean over a run that includes boot (or an idle menu) describes neither.
    let timing = std::env::var_os("VITASLOP_HEADLESS_TIMING").is_some();
    // Periodic GPU shots (`VITASLOP_HEADLESS_SHOT_EVERY=N`). Without this the headless path
    // renders exactly one frame - the last - which answers "what is on screen at frame N"
    // and nothing else. A title being brought up moves through a SEQUENCE (legal notice,
    // attract, title, menu), and finding the frame a given screen lives on by bisecting on
    // one final shot per run costs a whole boot per guess. The renderer is built once and
    // reused, so the extra cost is one render per sampled frame.
    //
    // Note this renders through `GeneralRenderer`, the same GPU path as the final shot, so
    // a sampled frame is directly comparable to it - and to the browser, which shares the
    // renderer. The software rasterizer is NOT used here and could not show a title whose
    // draws are fragment-shader passes.
    let shot_every: u64 = std::env::var("VITASLOP_HEADLESS_SHOT_EVERY")
        .ok()
        .and_then(|s| s.parse().ok())
        .unwrap_or(0);
    // The WINDOW the periodic shots apply to (`VITASLOP_HEADLESS_SHOT_FROM` /
    // `..._SHOT_TO`, inclusive, defaulting to the whole run). What this exists for is the
    // class of defect that lives in a TRANSITION - a fade, a wipe, a one-frame flash -
    // which a sampling interval coarse enough to survive a whole boot steps straight over.
    // Shooting every frame from boot instead is not the answer: these PNGs are written with
    // stored (uncompressed) deflate, so a 1400-frame run is gigabytes. A window makes
    // `SHOT_EVERY=1` affordable exactly where the question is.
    let shot_from: u64 = std::env::var("VITASLOP_HEADLESS_SHOT_FROM")
        .ok()
        .and_then(|s| s.parse().ok())
        .unwrap_or(0);
    let shot_to: u64 = std::env::var("VITASLOP_HEADLESS_SHOT_TO")
        .ok()
        .and_then(|s| s.parse().ok())
        .unwrap_or(u64::MAX);
    if shot_to < shot_from {
        return Err(format!(
            "VITASLOP_HEADLESS_SHOT_TO={shot_to} is before VITASLOP_HEADLESS_SHOT_FROM={shot_from}, \
             so the window is empty and the run would silently write no shots"
        ));
    }
    // `VITASLOP_CALLSITES_WINDOW=<from>-<to>`: clear the call-site histogram at display frame
    // `from` and print it at `to`, so the ranking describes those frames and not the boot.
    //
    // The end-of-run report below is cumulative, and a cumulative tally cannot rank STEADY
    // work: boot is thousands of frames of loading, and its call mix is a different program
    // from the one a frame rate is made of. That has mis-ranked this project's host-call list
    // repeatedly - the fix has always been "sample twice and difference", and nobody does,
    // because it costs two runs. One window costs none.
    let callsite_window: Option<(u64, u64)> = match std::env::var("VITASLOP_CALLSITES_WINDOW") {
        Err(_) => None,
        Ok(s) => {
            let (a, b) = s.split_once('-').ok_or_else(|| {
                format!("VITASLOP_CALLSITES_WINDOW={s} is not <from>-<to>")
            })?;
            let from: u64 = a.trim().parse().map_err(|_| format!("bad window start in {s:?}"))?;
            let to: u64 = b.trim().parse().map_err(|_| format!("bad window end in {s:?}"))?;
            if to <= from {
                return Err(format!(
                    "VITASLOP_CALLSITES_WINDOW={s} ends at or before it starts, so it would \
                     report an empty window"
                ));
            }
            if !vitaslop_runtime::knobs::flag("VITASLOP_DBG_CALLSITES") {
                return Err(
                    "VITASLOP_CALLSITES_WINDOW needs VITASLOP_DBG_CALLSITES=1 - nothing records \
                     call sites without it, so the window would report nothing"
                        .into(),
                );
            }
            Some((from, to))
        }
    };
    let mut periodic: Option<vitaslop_native::GeneralRenderer> = None;
    if shot_every > 0 {
        std::fs::create_dir_all(&shot_dir).map_err(|e| format!("mkdir: {e}"))?;
        periodic = Some(vitaslop_native::GeneralRenderer::new().ok_or("no GPU adapter")?);
    }
    let mut frame_ms: Vec<f64> = Vec::new();
    // >>> THE HICCUP LOG: per-frame `(frame, guest ms, render ms, scenes, draws)` for every
    // frame INSIDE the shot window, where the renderer runs on every frame rather than only on
    // the sampled ones.
    //
    // A percentile distribution says a run has expensive frames; it never says WHICH frame or
    // WHY, and "the run has a p99 of 40 ms" is not something anyone can act on. A hiccup is one
    // named frame, and the two halves next to each other are what separate a guest stall (a
    // streamed load, a GC sweep, a lock) from a render stall (a pipeline built mid-race, a
    // texture transcode, a scene that suddenly carries three times the draws). The draw and
    // scene counts are on the same row for the same reason: a frame that costs 5x while drawing
    // 5x as much is not a hiccup, it is work.
    //
    // Outside the shot window nothing is rendered, so a row there would report a render cost of
    // zero and read as a guest-only frame. Only the window is logged.
    let mut hiccups: Vec<Hiccup> = Vec::new();
    // Encode counters folded up across the window's per-frame takes, so the run-total report
    // below still sees every one of them. `take_encode_work` RESETS, and the hiccup rows need
    // per-frame deltas, so the two readers have to cooperate rather than race.
    let mut enc_total = vitaslop_platform::gpu::EncodeWork::default();
    // The same, for the sub-phases INSIDE `prepare` (`VITASLOP_PREPARE_SPLIT=1`). Folded up
    // over the WINDOW rather than read from the warm re-render loop, because the warm loop
    // renders one scene sixty times: its packed-vertex cache hits every time, so it prices the
    // hashing and the arena copies honestly and prices the REPACK at zero. A race frame moves
    // its geometry, and this is the reader that sees that.
    let mut prep_total = vitaslop_platform::gpu::PrepareSplit::default();
    let mut prep_frames = 0u64;
    while guest.frames() < target && !guest.finished() {
        let f = guest.frames();
        if let Some((from, to)) = callsite_window {
            if f == from {
                vitaslop_runtime::vita::reset_call_sites();
                println!("callsites: window {from}..{to} OPEN (counts before this are discarded)");
            }
            if f == to {
                println!(
                    "callsites: window {from}..{to} = {} display frames\n{}",
                    to - from,
                    vitaslop_runtime::vita::call_sites_report(30)
                );
            }
        }
        let in_shot_window = f >= shot_from && f <= shot_to;
        // The size the guest declared for THIS frame. It is read per frame rather than once
        // because a title changes it: one front end presents 640x368 and its world 960x544,
        // through the same three buffers.
        let display = guest.display_size();
        // >>> RENDER EVERY FRAME IN THE WINDOW, WRITE ONLY THE SAMPLED ONES.
        //
        // # A sampled render is not a faithful picture, and the way it fails is invisible
        // A render target is guest memory: it keeps what was last written into it, and a title
        // routinely paints something ONCE - at a screen transition - and samples it for the next
        // thousand frames. The renderer serves that from `rtt_rendered` plus the targets still
        // resident from earlier frames, so a target is only correct here if this process actually
        // RENDERED the frame that painted it.
        //
        // Rendering only at the sampling interval breaks exactly that. MEASURED on one
        // title's event briefing: sampled every 500 frames, the whole menu background is
        // BLACK; rendering the 500 frames up to the same flip contiguously, it is the light
        // grey panel the browser draws and the device draws. Nothing in the shot, the log or
        // the scene composition says a layer is missing - the draw counts are identical (8
        // scenes / 49 draws either way), because the guest submitted the same work and it was
        // OUR history that was empty.
        //
        // So every frame inside the window is rendered and only the sampled ones are written.
        // The window is the knob that keeps this affordable: it is already there to make
        // `SHOT_EVERY=1` cheap enough to catch a one-frame flash, and it now also bounds the
        // cost of being faithful. Outside the window nothing is rendered, exactly as before.
        let mut render_ms = 0.0f64;
        let mut frame_shape = (0usize, 0usize);
        if let (Some(r), true) = (periodic.as_mut(), shot_every > 1 && in_shot_window && f % shot_every != 0) {
            let scenes = guest.current();
            if !scenes.is_empty() {
                frame_shape = (scenes.len(), scenes.iter().map(|s| s.draws.len()).sum());
                let t = std::time::Instant::now();
                let _ = r.render_frame(scenes, display.0, display.1, CLEAR);
                render_ms = t.elapsed().as_secs_f64() * 1000.0;
            }
        }
        if let (Some(r), true) = (periodic.as_mut(), shot_every > 0 && in_shot_window && f % shot_every == 0) {
            let scenes = guest.current();
            // The frame's COMPOSITION, taken while the scenes are borrowed and printed below
            // with the clock. A frame that loses most of its picture from one flip to the next
            // is either the guest drawing less or us placing less, and those are opposite bugs
            // - these two counts separate them without another run. The scene and surface
            // REPORTS cannot: they dedupe by design, so they say nothing about a given frame.
            let scene_count = scenes.len();
            let draw_count: usize = scenes.iter().map(|s| s.draws.len()).sum();
            // ...and WHICH scenes, by the colour address each one rasterises into, in the
            // order the guest submitted them. A count says a frame changed shape; only the
            // addresses say whether a pass APPEARED, whether it targets the display buffer or
            // an offscreen, and which pass is the one that stopped contributing. The scene
            // reports above cannot answer that - they fire once per surface for a whole run.
            let composition: Vec<String> = scenes
                .iter()
                .map(|s| match &s.color {
                    Some(c) => format!("{:#x}:{}", c.data_addr, s.draws.len()),
                    // A scene with no resolvable colour surface renders NOWHERE, and one of
                    // those is already known to be load-bearing on this title (a later pass
                    // samples its depth). Name it rather than letting it read as absent.
                    None => format!("NO-COLOUR:{}", s.draws.len()),
                })
                .collect();
            frame_shape = (scene_count, draw_count);
            if !scenes.is_empty() {
                let t = std::time::Instant::now();
                let fb = r.render_frame(scenes, display.0, display.1, CLEAR);
                render_ms = t.elapsed().as_secs_f64() * 1000.0;
                let path = shot_dir.join(format!("f{f:06}.png"));
                // Written at PANEL size whatever the guest declared, so a shot sequence is
                // comparable frame to frame and against the browser. `scaled_to` is a copy
                // when the two already agree, which is every title but one.
                let fb = fb.scaled_to(GAME_W, GAME_H);
                std::fs::write(&path, fb.to_png()).map_err(|e| format!("write png: {e}"))?;
            }
            // The guest clock beside the frame, so its LOCAL rate is visible. A run-total
            // ratio of 1.00x hides a stretch that ran five times fast against a stretch
            // that stalled, and a title paced off the clock behaves very differently in
            // the two.
            let (clk_q, clk_topup, clk_idle) = guest.clock_sources();
            println!(
                "shot f{f:06}: guest clock {:.3}s ({:.3}s quanta + {:.3}s topup + {:.3}s idle), \
                 {scene_count} scenes / {draw_count} draws [{}]",
                guest.clock_us() as f64 / 1e6,
                clk_q as f64 / 1e6,
                clk_topup as f64 / 1e6,
                clk_idle as f64 / 1e6,
                composition.join(" "),
            );
        }
        if use_builtin_taps {
            let touch = taps
                .iter()
                .find(|(a, b, _, _)| f >= *a && f < *b)
                .map(|(_, _, x, y)| TouchFrame::single(*x, *y));
            input.lock().unwrap().touch = touch;
        }
        if timing {
            let t = std::time::Instant::now();
            guest.advance();
            let guest_ms = t.elapsed().as_secs_f64() * 1000.0;
            frame_ms.push(guest_ms);
            if in_shot_window && periodic.is_some() {
                // The encode counters THIS frame moved. A hitch with a pipeline build in it and
                // a hitch with a megabyte of texture in it look identical in milliseconds and
                // need different fixes, and neither is visible in a run total.
                let e = vitaslop_platform::gpu::take_encode_work();
                hiccups.push(Hiccup {
                    frame: f,
                    guest_ms,
                    render_ms,
                    scenes: frame_shape.0,
                    draws: frame_shape.1,
                    pipelines_built: e.pipelines_built,
                    tex_uploaded: e.tex_uploaded,
                    tex_upload_bytes: e.tex_upload_bytes,
                    bind_groups_built: e.bind_groups_built,
                    buffer_bytes: e.buffer_bytes,
                });
                enc_total.add(&e);
                prep_total.add(&vitaslop_platform::gpu::take_prepare_split());
                prep_frames += 1;
            }
        } else {
            guest.advance();
        }
    }
    // Block-visit histogram (`VITASLOP_BLOCK_HIST=<top>`, with `VITASLOP_TRACE_BLOCKS=lo-hi`
    // emitting the hooks). It was only reachable from the boot-probe test, which cannot
    // replay a recipe - so the one case it is built for, "the title reached screen N and
    // stopped", could not be measured at all. Restrict the hook range to the suspect
    // functions and the tail of the recorded sequence IS the stuck loop.
    if let Ok(top) = std::env::var("VITASLOP_BLOCK_HIST") {
        vitaslop_native::dump_block_hist(top.trim().parse().unwrap_or(40));
    }
    // Sampler-unit bindings dropped at capture time, ranked by cause. Unconditional: a
    // dropped unit is why a recompiled shader falls back, and the per-cause reports are
    // deduped so only this says which cause is worth fixing first.
    vitaslop_runtime::host::report_texture_drops();
    // How many vblank wait loops were parked rather than spun through. Silent when none
    // were, which is the reading for a title that does not wait that way.
    vitaslop_runtime::host::report_vblank_spin_parks();
    // What the NGS mix carried, and how much of it was audible.
    vitaslop_runtime::vita::at9::report_mix();
    // Each guest memory space, and above all its allocs against its frees - a pool that only
    // ever fills is a release path this engine does not implement, and one that churns is the
    // title's own business. The warning on a failed allocation guesses between those two; this
    // is the measurement.
    for line in guest.mspace_report() {
        println!("{line}");
    }
    // `VITASLOP_CPU_SHARE=1`: who got the CPU, and how many threads were READY when the
    // scheduler handed the baton on. The second half is what the game clock divides by
    // (the device runs three of them at once), so a clock that looks wrong on a loading
    // screen is answered here or nowhere.
    //
    // ...and ALWAYS when the run ended in a guest error, knob or no knob. A guest fault is
    // almost always a question about ordering - which thread published what before which
    // other thread read it - and the CPU share plus the runnable histogram are the only
    // description of the schedule this run leaves behind. Asking for it after the fact costs
    // a whole replay, and on a fault whose frame moves with the clock that replay is not
    // guaranteed to reproduce the same fault.
    if std::env::var_os("VITASLOP_CPU_SHARE").is_some() || guest.error().is_some() {
        print!("{}", guest.scheduler_report());
    }
    // VITASLOP_DUMP_STDOUT=<path>: write everything the GUEST logged to fd 1/2 this run.
    // Written before the error check, because a run that ended in a trap or a hang is exactly
    // the one whose log matters most.
    if let Ok(path) = std::env::var("VITASLOP_DUMP_STDOUT") {
        let bytes = guest.guest_stdout();
        let len = bytes.len();
        std::fs::write(&path, bytes).map_err(|e| format!("write guest stdout: {e}"))?;
        println!("headless: wrote {len} bytes of GUEST log to {path}");
    }
    // Also before the error check, and for the same reason: a guest fault is the run whose
    // memory you most need to look at. `VITASLOP_PEEK` used to be evaluated only after a
    // clean render, so the one question it exists to answer - "what was in that record when
    // the guest dereferenced it" - was the one question it could not be asked. It reads guest
    // memory and prints; it cannot fail the run.
    peek_regions(&guest);
    if let Some(e) = guest.error() {
        return Err(format!("guest error at frame {}: {e}", guest.frames()));
    }
    let scenes = guest.current().to_vec();
    if scenes.is_empty() {
        return Err("no scene captured".into());
    }

    let mut renderer = vitaslop_native::GeneralRenderer::new().ok_or("no GPU adapter")?;
    let render_t = std::time::Instant::now();
    let display = guest.display_size();
    let fb = renderer.render_frame(&scenes, display.0, display.1, CLEAR);
    let cold_render_ms = render_t.elapsed().as_secs_f64() * 1000.0;
    if timing {
        // Re-render the SAME scene to separate one-off cost from per-frame cost. The first
        // render builds every pipeline, compiles every recompiled shader and uploads every
        // texture; a steady frame does none of that. Reporting only the cold number would
        // overstate the per-frame cost by orders of magnitude, and reporting only the warm
        // one would hide a startup hitch users would actually feel.
        let mut warm: Vec<f64> = Vec::new();
        for _ in 0..WARM_RENDER_SAMPLES {
            let t = std::time::Instant::now();
            let _ = renderer.render_frame(&scenes, display.0, display.1, CLEAR);
            warm.push(t.elapsed().as_secs_f64() * 1000.0);
        }
        report_frame_timing(&frame_ms, cold_render_ms, &warm, renderer.last_split(), &enc_total);
        report_hiccups(&hiccups);
        // Where `prepare` went, averaged over the WINDOW's real frames. `encode` is most of a
        // render and `prepare` is most of `encode`; this is the line that says what in it.
        if !prep_total.is_empty() {
            println!("hiccups: {}", prep_total.line(prep_frames));
        }
    }
    std::fs::create_dir_all(&shot_dir).map_err(|e| format!("mkdir: {e}"))?;
    let path = shot_dir.join("desktop.png");
    // Panel size, whatever the guest declared - see the sampled-shot write above.
    let fb = fb.scaled_to(GAME_W, GAME_H);
    std::fs::write(&path, fb.to_png()).map_err(|e| format!("write png: {e}"))?;
    // Diagnostic (VITASLOP_SOFTWARE=1): also render the frame's FINAL scene through the
    // software rasterizer and write it beside the GPU shot. The two paths share the scene
    // but nothing else, so a defect present in both is upstream of rendering (geometry,
    // capture, uniforms) rather than a shading bug - and the software path is the one that
    // reports per-draw `DSTAT` coverage under VITASLOP_DRAW_STATS. It renders one scene, so
    // on a title that composites offscreen passes it shows the composite only.
    if std::env::var_os("VITASLOP_SOFTWARE").is_some() {
        let sw = vitaslop_runtime::render::render_scene(scenes.last().unwrap(), GAME_W, GAME_H, CLEAR);
        let sw_path = shot_dir.join("software.png");
        std::fs::write(&sw_path, sw.to_png()).map_err(|e| format!("write png: {e}"))?;
        println!("headless: wrote {}", sw_path.display());
    }
    let frames = guest.frames();
    let clock_s = guest.clock_us() as f64 / 1e6;
    if let Some(why) = guest.ended_by() {
        println!(
            "headless: the run ENDED BEFORE the frame target ({frames} of {target}): {why}.              That is the guest stopping, not the target being met - a deadlock, a thread              exiting or a round budget running out all land here."
        );
    }
    println!(
        "headless: reached frame {frames}, wrote {} (guest clock {clock_s:.1}s over {frames} displayed frames = {:.2} display periods each). THAT COUNT IS THE TITLE'S OWN VBLANK DIVISOR and should be a WHOLE number: 1.00 for a 60 fps title, 2.00 for a 30 fps one. A fraction, or one more period than the title's own limiter waits for, is game time no displayed frame accounted for - and since audio is billed in clock time, on a device that is what fills the audio ring and drops the excess.",
        path.display(),
        clock_s * 60.0 / frames.max(1) as f64,
    );
    print!("{}", guest.idle_attribution());
    print!("{}", guest.blocked_threads());
    // AUDIO AGAINST THE CLOCK IT IS PACED ON. `sceAudioOutOutput` parks one grain of
    // VIRTUAL time, so this is 1.00 on a healthy path whatever the frame rate - and it stays
    // 1.00 when the CLOCK itself is wrong, which is why the period count above is the other
    // half of the pair. See `AudioState::produced_seconds`.
    let audio_s = guest.audio_produced_seconds();
    if audio_s > 0.0 && clock_s > 0.0 {
        println!(
            "headless: the guest produced {audio_s:.1}s of SOUND against {clock_s:.1}s of emulated clock ({:.2}x). Audio is billed in clock time, so this is 1.00 on a healthy path at any frame rate; it is the PERIODS-PER-FRAME figure above, not this one, that catches a clock running fast.",
            audio_s / clock_s,
        );
    }
    // The fuel accounting's own totals, next to the clock they price. A mean burn near the
    // preemption interval means most suspends really are full slices; a mean far below it
    // with a huge total means the clock is being driven by the NUMBER of suspends, which is
    // the thing charging per unit of fuel exists to stop. A max above the interval is not a
    // busy title, it is a broken reading - the engine preempts at the interval.
    let (fuel, samples, max) = guest.fuel_report();
    if samples > 0 {
        println!(
            "headless: fuel {fuel} over {samples} suspends (mean {}, max {max}, interval {})",
            fuel / samples,
            vitaslop_runtime::host::QUANTUM_FUEL,
        );
    }
    // The emitted work counter against wasmtime's own metering, over the same intervals.
    // Both engines preempt on that counter and the game clock is billed from it, and
    // nothing in a BROWSER run can say whether it agrees with a real engine - this is
    // where that is checked. It reads a little under 1.00 by construction: wasmtime bills
    // the counter's own operators, which the counter does not bill itself.
    if let Some((sw, wt, n)) = vitaslop_native::threaded::software_fuel_report() {
        println!(
            "headless: software fuel {sw} vs wasmtime {wt} over {n} samples (ratio {:.2}x)",
            sw as f64 / wt.max(1) as f64,
        );
    }
    // Zero unless the engine suspended a fiber without running any of our code, which on
    // a build with the emitted work check should be impossible.
    let stray = vitaslop_native::threaded::unattributed_suspends();
    if stray != 0 {
        println!("headless: {stray} suspends were not produced by the emitted work check");
    }
    Ok(())
}

/// Run the retail title in `dir` in a live window until the window closes or the guest
/// exits. Blocks until the event loop ends. With `recipe` set, a scripted playthrough
/// replays in the window (live keyboard/mouse still nudges it).
pub fn run(dir: PathBuf, recipe: Option<String>) -> Result<(), String> {
    let input: SharedInput = Arc::new(Mutex::new(DesktopInput::default()));
    let guest = RetailGuest::new(&dir, input.clone(), recipe.as_deref())?;
    println!("loaded {} (decrypt + link + transpile {:.0} ms)", dir.display(), guest.build_ms);
    if recipe.is_some() {
        println!("replaying scripted recipe (keyboard/mouse still nudge live)");
    }
    println!("controls:");
    println!("  menus  : click the window (the front-end is touch driven)");
    println!("  d-pad  : arrow keys        faces: Z cross / X circle / A square / S triangle");
    println!("  L / R  : Q / E             Start: Enter   Select: Shift");
    println!("  a gamepad also works; Space pauses, Esc quits");

    let event_loop = EventLoop::new().map_err(|e| format!("create event loop: {e}"))?;
    event_loop.set_control_flow(ControlFlow::Poll);
    let mut app = RetailApp {
        guest,
        input_shared: input,
        input: Input::new(),
        window: None,
        gfx: None,
        paused: false,
        cursor: (0.0, 0.0),
        mouse_down: false,
        acc: Duration::ZERO,
        last_tick: Instant::now(),
        fps_since: Instant::now(),
        fps_frames: 0,
        guest_frames_since: 0,
        fps: 0.0,
        guest_fps: 0.0,
        reported_exit: false,
    };
    event_loop.run_app(&mut app).map_err(|e| format!("run event loop: {e}"))?;
    Ok(())
}

/// The retail window app: the live guest, the merged input, mouse-as-touch state, the
/// window + its general-renderer surface, and 60 Hz pacing + FPS bookkeeping.
struct RetailApp {
    guest: RetailGuest,
    input_shared: SharedInput,
    input: Input,
    window: Option<Arc<Window>>,
    gfx: Option<RetailGfx>,
    paused: bool,
    /// Last cursor position in physical window pixels, and whether the left button is
    /// held - together the mouse-as-touch source.
    cursor: (f64, f64),
    mouse_down: bool,
    acc: Duration,
    last_tick: Instant,
    fps_since: Instant,
    /// Frames PRESENTED since the meter window opened (redraws), and the guest display
    /// flips retired in the same window. These are different numbers and the difference is
    /// the point: the guest is stepped against real time with a catch-up budget, so when
    /// emulation cannot keep up the window keeps presenting at the display rate while the
    /// guest falls behind. Reporting only the present rate would show a steady 60 while the
    /// title ran in slow motion.
    fps_frames: u32,
    guest_frames_since: u64,
    fps: f64,
    /// Emulated display flips per second, and that as a fraction of the pace the guest is
    /// being stepped at (`FRAME_DT`) - i.e. how close to real-time speed it is running.
    guest_fps: f64,
    reported_exit: bool,
}

impl ApplicationHandler for RetailApp {
    fn resumed(&mut self, event_loop: &ActiveEventLoop) {
        if self.window.is_some() {
            return;
        }
        let attrs = Window::default_attributes()
            .with_title("vitaslop")
            .with_inner_size(LogicalSize::new(GAME_W, GAME_H));
        let window = Arc::new(event_loop.create_window(attrs).expect("create window"));
        match RetailGfx::new(window.clone()) {
            Ok(g) => {
                println!("presenting on GPU: {}", g.adapter_name);
                self.gfx = Some(g);
            }
            Err(e) => {
                eprintln!("failed to init GPU surface: {e}");
                event_loop.exit();
                return;
            }
        }
        self.last_tick = Instant::now();
        self.window = Some(window);
    }

    fn window_event(&mut self, event_loop: &ActiveEventLoop, _id: WindowId, event: WindowEvent) {
        match event {
            WindowEvent::CloseRequested => event_loop.exit(),
            WindowEvent::Resized(size) => {
                if let Some(g) = self.gfx.as_mut() {
                    g.resize(size.width, size.height);
                }
            }
            WindowEvent::KeyboardInput { event, .. } => {
                let pressed = event.state == ElementState::Pressed;
                if let PhysicalKey::Code(code) = event.physical_key {
                    match code {
                        KeyCode::Escape if pressed => {
                            event_loop.exit();
                            return;
                        }
                        KeyCode::Space if pressed && !event.repeat => self.paused = !self.paused,
                        _ => {}
                    }
                    self.input.set_key(code, pressed);
                }
            }
            WindowEvent::CursorMoved { position, .. } => {
                self.cursor = (position.x, position.y);
            }
            WindowEvent::MouseInput { state, button, .. } => {
                if button == MouseButton::Left {
                    self.mouse_down = state == ElementState::Pressed;
                }
            }
            WindowEvent::RedrawRequested => self.render(),
            _ => {}
        }
    }

    fn about_to_wait(&mut self, _event_loop: &ActiveEventLoop) {
        if let Some(w) = self.window.as_ref() {
            w.request_redraw();
        }
    }
}

impl RetailApp {
    /// The mouse-as-touch frame this instant: a front-panel finger at the cursor when
    /// the left button is held, else no touch. Cursor physical pixels map through the
    /// window size to 960x544 game space, then double to panel coords.
    fn mouse_touch(&self) -> Option<TouchFrame> {
        if !self.mouse_down {
            return None;
        }
        let (ww, wh) = self
            .window
            .as_ref()
            .map(|w| {
                let s = w.inner_size();
                (s.width.max(1) as f64, s.height.max(1) as f64)
            })
            .unwrap_or((GAME_W as f64, GAME_H as f64));
        let sx = (self.cursor.0 / ww * GAME_W as f64).clamp(0.0, GAME_W as f64);
        let sy = (self.cursor.1 / wh * GAME_H as f64).clamp(0.0, GAME_H as f64);
        Some(TouchFrame::single((sx as f32 * PANEL_SCALE) as u16, (sy as f32 * PANEL_SCALE) as u16))
    }

    fn render(&mut self) {
        self.input.pump_gamepad();
        let ctrl = self.input.ctrl_frame();
        let touch = self.mouse_touch();
        *self.input_shared.lock().unwrap() = DesktopInput { ctrl, touch };

        let now = Instant::now();
        self.acc += now.duration_since(self.last_tick);
        self.last_tick = now;

        if self.paused {
            self.acc = Duration::ZERO;
        } else {
            if self.guest.current().is_empty() {
                self.guest.advance(); // bootstrap the first frame (runs the whole boot)
            }
            // >>> ONE GUEST FRAME PER REDRAW, SO EVERY FRAME COMPUTED IS A FRAME SHOWN.
            //
            // This used to allow four, and present once - so three of the four were computed
            // and discarded. The budget looks like a safety cap and behaves like a frame-rate
            // halver, because the loop has a stable fixed point at EVERY count: at two frames
            // per redraw the redraw takes twice as long, so `acc` arrives at twice the size,
            // so it runs two again. Nothing pushes it back down, and one hitch is enough to
            // settle it there for the rest of the run.
            //
            // MEASURED in the browser, whose loop had the identical shape (see `live_loop` in
            // vitaslop-web): a race at 100% emulated speed was showing 31 of the 60 frames a
            // second it computed, on a machine with headroom. One frame per tick showed 60.
            //
            // Catch-up is not lost, it is recognised as unavailable: sprinting through extra
            // frames is only possible for a machine with spare time, and a machine with spare
            // time is already keeping up. `acc` still carries the deficit, and the clamp below
            // still drops it when a boot frame makes it absurd.
            if self.acc >= FRAME_DT {
                self.acc -= FRAME_DT;
                self.guest.advance();
            }
            if self.acc > FRAME_DT * 4 {
                self.acc = Duration::ZERO;
            }
        }

        if self.guest.finished() && !self.reported_exit {
            self.reported_exit = true;
            match self.guest.error() {
                Some(e) => eprintln!("guest exited with error: {e}"),
                None => println!("guest exited after {} frames", self.guest.frames()),
            }
        }

        // Read before the renderer is borrowed: both live on `self`.
        let display = self.guest.display_size();
        if let Some(gfx) = self.gfx.as_mut() {
            let scenes = self.guest.current();
            if !scenes.is_empty() {
                gfx.present(scenes, display);
            }
        }
        self.update_title(now);
    }

    /// Publish both rates to the window title, four times a second.
    ///
    /// Two numbers, deliberately: `fps` is what the window PRESENTED, `guest` is how many
    /// emulated display flips actually retired, and `speed` is that against the pace the
    /// guest is stepped at. On a machine with headroom they agree and speed sits at 100%;
    /// when emulation cannot keep up, present stays pinned to the display rate and only the
    /// guest number drops - which is the number a user actually wants when asking "is this
    /// running full speed?".
    fn update_title(&mut self, now: Instant) {
        self.fps_frames += 1;
        let since = now.duration_since(self.fps_since);
        if since < Duration::from_millis(250) {
            return;
        }
        let secs = since.as_secs_f64();
        let guest_now = self.guest.frames();
        self.fps = self.fps_frames as f64 / secs;
        self.guest_fps = guest_now.saturating_sub(self.guest_frames_since) as f64 / secs;
        self.fps_frames = 0;
        self.guest_frames_since = guest_now;
        self.fps_since = now;
        let Some(w) = self.window.as_ref() else { return };
        let state = if self.guest.finished() {
            " [exited]"
        } else if self.paused {
            " [paused]"
        } else {
            ""
        };
        // The step pace is the reference for "full speed": one FRAME_DT per emulated flip.
        let target = 1.0 / FRAME_DT.as_secs_f64();
        let speed = if self.paused || self.guest.finished() {
            String::new()
        } else {
            format!("  |  speed {:.0}%", self.guest_fps / target * 100.0)
        };
        w.set_title(&format!(
            "vitaslop  |  {:.0} fps present  |  {:.0} fps guest{speed}{state}  |  frame {}",
            self.fps,
            self.guest_fps,
            guest_now
        ));
    }
}
