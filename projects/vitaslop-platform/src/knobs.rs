//! Process-wide overrides for the `VITASLOP_*` knobs, for a platform that has no
//! environment to read them from.
//!
//! # Why this exists, and why it lives in the LOWEST crate
//! `wasm32-unknown-unknown` has no environment at all: `std::env::var` always returns
//! `NotPresent` and `std::env::set_var` fails outright. So in the browser every knob
//! reads as unset, and there is no way to tell that apart from "the knob is off". That
//! is not a diagnostic inconvenience - it silently changes what the emulator IS. The
//! renderer's master switch `VITASLOP_GXP_LIVE` is read this way, so before this table
//! existed the browser could only ever draw the fixed-function APPROXIMATION, while the
//! desktop oracle it is supposed to match drew the guest's real shaders. Two different
//! renderers, no message, and a browser frame that looked plausible and was wrong.
//!
//! The table lives here rather than in `vitaslop-runtime` because the readers span
//! crates in both directions: `vitaslop-runtime` depends on `vitaslop-platform`, so the
//! renderer in [`crate::gpu`] cannot reach a table owned by the runtime.
//! `vitaslop_runtime::knobs` re-exports these so the public API is unchanged, and keeps
//! the generated `KNOBS.md` index, which needs the whole workspace on disk.
//!
//! # Fail loudly, never partially
//! [`set_override`] PANICS on a name whose reader still calls `std::env::var` directly.
//! A silently-ignored override would leave the caller believing it had configured a run
//! it had not - the exact failure this module exists to prevent. A name earns its place
//! in [`OVERRIDABLE`] only once its reader goes through [`var`] / [`var_os`] / [`flag`].

use std::collections::HashMap;
use std::sync::{Mutex, OnceLock};

/// Knobs whose readers go through this module, so [`set_override`] can reach them.
///
/// Grouped by what a browser run actually needs:
/// - `VITASLOP_FRAME_TOPUP` - one retail racer never finishes loading without it.
/// - `VITASLOP_GXP_*` - the shader recompiler's master switch and its diagnostics. Without
///   `VITASLOP_GXP_LIVE` the browser renders a different picture than the desktop oracle.
/// - `VITASLOP_BROWSER_*` - knobs only the browser build reads; they have no environment
///   to come from by construction.
pub const OVERRIDABLE: &[&str] = &[
    "VITASLOP_ALLOW_SOFTWARE_GPU",
    "VITASLOP_BROWSER_FASTFORWARD",
    "VITASLOP_BROWSER_FUEL",
    "VITASLOP_BROWSER_HEARTBEAT_MS",
    // Whether a finished guest thread's module instance may be reused. On by default and
    // overridable because it is the single variable that decides whether a title creating
    // a thread per frame instantiates the whole module sixty times a second.
    "VITASLOP_BROWSER_INSTANCE_POOL",
    "VITASLOP_BROWSER_QUANTUM_CALLS",
    "VITASLOP_BROWSER_SUPERSAMPLE",
    // The per-NID / per-call-site host-call histogram. Reachable from the browser because that
    // is where the host-call boundary costs the most: a phone spends roughly 16 ms of a 56 ms
    // guest frame on ~4,950 calls, and the only way to spend less is to make fewer of them -
    // which needs to know WHICH ones. Read through this seam rather than `std::env` for exactly
    // that reason; see `vitaslop_runtime::vita::DBG_CALLSITES`.
    "VITASLOP_DBG_CALLSITES",
    // ONE switch for every expensive instrument, chosen before a run starts and never turned on
    // by anything but a human asking for it. The individual knobs above and `VITASLOP_PERF` below
    // still exist for a harness that wants exactly one of them; this is what a person picks, and
    // its default of OFF is the point - profiling machinery does not belong in an ordinary run.
    "VITASLOP_DEBUG_CAPTURE",
    // Budget for the decoded-texture cache, in MB. Reachable from the browser because that
    // is where outgrowing it costs the most: a wholesale clear there re-decodes hundreds of
    // textures inside one frame's `build`.
    "VITASLOP_DECODE_CACHE_MB",
    // The dispatch ABLATION: route even a fallthrough through the function's `br_table`.
    // Browser-reachable because the question it answers is a V8 branch-prediction question -
    // the module carries one indirect branch per 10.5 guest instructions and nothing this
    // project usually counts can price one.
    "VITASLOP_DISPATCH_ALL",
    // The A/B arm for `emit_flags_add`'s carry and overflow forms. It is here because the
    // BROWSER is the engine that has to answer: `flags-add` was 39% of every operator the
    // transpiler emitted, the closed forms cut the module 5.3% and executed operators 8.7%,
    // and three interleaved desktop repeats put the wall-clock difference inside the noise.
    // Encode only draws `lo..=hi` of every pass. Browser-reachable because the question it
    // answers - "which draw put this on screen, and which one covered it" - is asked of a
    // PICTURE, and the pictures that need it are the ones only a device or a browser produces.
    // `VITASLOP_CHAIN_LIMIT` bisects by PASS and cannot touch a title whose frame is one pass.
    "VITASLOP_DRAW_RANGE",
    "VITASLOP_FLAGS_WIDE_C",
    // The one frame whose per-SCENE digests are printed, so a cross-engine difference lands on
    // a PASS instead of on a whole frame.
    "VITASLOP_FRAME_DIGEST",
    "VITASLOP_FRAME_TOPUP",
    // The clock's core model, so a browser run can be A/B'd against native without an
    // environment to set it in.
    "VITASLOP_GUEST_CORES",
    // Force every pass to ONE sample, whatever `SceGxmMultisampleMode` the guest asked for.
    // Reachable from the browser because that is the ONLY place the cost of multisampling can
    // be priced: the phone is the target hardware and its GPU is a tile-based PowerVR, where
    // MSAA is cheap for entirely different reasons than on this desktop. A render change that
    // cannot be turned off on the machine that pays for it cannot be measured at all.
    //
    // Missing from this list when the knob was added, which is the third time that has
    // happened here (the call-site profiler, then the inline-imports switch). It is not a
    // silent omission: `set_override` PANICS on an unregistered name, so a phone run that
    // typed it into the knobs box died on boot with a black canvas and no output.
    "VITASLOP_GXM_NO_MULTISAMPLE",
    // Poisons a freshly reserved default uniform buffer, so a lane the guest never wrote is
    // distinguishable from one it wrote as zero. NOTE it only covers the RESERVE path, never a
    // precomputed state's guest-owned buffer - so its silence is not evidence until the pattern
    // is seen SOMEWHERE. Browser-reachable because the value it has to decide about
    // (`screenTintColour`) only ever appears there.
    "VITASLOP_GXM_UNIFORM_POISON",
    "VITASLOP_GXP_ALLOW_FIXED_FUNCTION",
    // What an attribute lane the vertex stream does not supply is FILLED with. Browser-reachable
    // because the fill value is a picture question and the phone is where wrong pictures are
    // reported from.
    "VITASLOP_GXP_ATTR_FILL",
    // The capsule capture. Reachable from the browser like every other knob here, but the
    // WRITE will fail there - a browser worker has no filesystem to put a capsule on - and the
    // capture reports that failure by name rather than dropping the draw in silence.
    "VITASLOP_GXP_CAPSULE",
    "VITASLOP_GXP_CAPSULE_MIN_INDICES",
    "VITASLOP_GXP_CAPSULE_SKIP",
    "VITASLOP_GXP_CULL",
    "VITASLOP_GXP_DUMP",
    "VITASLOP_GXP_EXCLUDE",
    "VITASLOP_GXP_FORCE",
    // What a draw was FED - its default uniform bank decoded per parameter, its attribute
    // ranges and its bound textures. Reachable from the browser because the defect it is
    // pointed at (a composite that blows out to white from measurably correct inputs)
    // reproduces THERE, and the values it prints come from the guest, which is the half that
    // differs between engines.
    "VITASLOP_GXP_INPUTS",
    // The same, one line per SUBMISSION in order - which is how the frame's LAST pair (the
    // composite) is identified at all.
    "VITASLOP_GXP_INPUTS_ORDER",
    // The per-VERTEX half of `..._INPUTS`, on its own name because it is unbounded in the one
    // place that cannot afford it: the browser panel keeps 96 distinct lines, and a 288-vertex
    // composite grid evicts every other finding - including the uniforms the run was taken for.
    "VITASLOP_GXP_INPUTS_VERTS",
    "VITASLOP_GXP_KEYS",
    "VITASLOP_GXP_LIVE",
    // Turns OFF the generated mip chain. Found missing by
    // `a_knob_routed_through_this_module_is_reachable_from_the_browser`, which is the FIFTH
    // instance of this omission - and it belongs here for the same reason `VITASLOP_TEX_COMPRESS`
    // does: the chain is a third of every uploaded RGBA8 texture's bytes, so on the device that
    // runs out of GPU memory it is both a memory lever and the A/B for whether the chain is what
    // prevents speckle.
    "VITASLOP_GXP_MIPS",
    "VITASLOP_GXP_NEGW",
    "VITASLOP_GXP_NOBLEND",
    "VITASLOP_GXP_NODEPTH",
    "VITASLOP_GXP_ONLY",
    // Substitute a default-uniform register before a draw is submitted - the causality half of
    // `VITASLOP_GXP_INPUTS`. Reachable from the browser for the same reason as that one: the
    // white-out it is aimed at reproduces there and nowhere a file can be written.
    // For a title whose `sceGxmShaderPatcherCreateFragmentProgram` passes a NULL vertexProgram -
    // so the call names no shader PAIR and nothing can be prepared from it - offer the CROSS
    // PRODUCT of its created fragment and vertex programs and keep the ones that LINK.
    // The shader PAIRS a run linked, one line each. Browser-reachable because a pair that links
    // on the desktop and not on the device is exactly the failure this names.
    "VITASLOP_GXP_PAIRS",
    // Compile a title's shader pairs AHEAD of the draw that needs them. Browser-reachable
    // because an in-frame shader compile costs the most there - it is the hitch itself.
    "VITASLOP_GXP_PRECOMPILE",
    // Speculative work paid on a loading screen; how much of it is wasted is a per-title
    // measurement, which is what this exists to take. Reachable from the browser because that
    // is where an in-frame shader compile costs the most.
    "VITASLOP_GXP_PRECOMPILE_CROSS",
    // Every SUBMISSION of one pair that lands in a screen-space box, with its full vertex
    // record. The per-DRAW half `..._INPUTS_VERTS` cannot be: that dump dedupes by input
    // set, so a UI pair submitted a thousand times a frame almost never prints the element
    // under investigation. Browser-reachable for the same reason the other input dumps are.
    "VITASLOP_GXP_QUADS",
    "VITASLOP_GXP_SA",
    // The SHADER EMITTER's two arms, forwarded into `vitaslop_gxp_shader::link` by
    // `set_override` below rather than read through `var` - that crate has no dependencies by
    // design and cannot see this table. Both are here because of the black race: a phone whose
    // driver refused four pipelines could not be handed either arm, and `SIZE_BANKS=0` alone
    // would have bisected it in one run. `SA_DIRECT` also takes `unroll`, which is the control
    // that separates a value change from a driver-codegen one.
    "VITASLOP_GXP_SA_DIRECT",
    "VITASLOP_GXP_SIZE_BANKS",
    "VITASLOP_GXP_SOLID",
    "VITASLOP_GXP_YFLIP",
    "VITASLOP_GXP_ZFIX",
    "VITASLOP_LOG",
    // The SCOPED switch for the three `sceClibMem*` bulk primitives. Reachable from the
    // browser on both counts at once: they are 13% of a real title's host calls, so pricing
    // them is a device question; and they are the only inline forms that write a range the
    // GUEST sizes and the only ones that stamp the guest-store dirty map, which is a path
    // that exists on the browser and nowhere else. Read at LINK time; set it before the run.
    // Refuse BC texture formats and take the transcode path the phone's GPU forces. Browser-
    // reachable because the phone has no BC at all ([[vitaslop-phone-gpu-has-no-bc]]), so this
    // is how a desktop browser is made to render what the device renders.
    // The movie diagnostics, and they belong here more than most: the phone is where a movie
    // has looked wrong, the phone has no environment, and a rendered frame is not an oracle
    // for a movie (which picture a run lands on is decided by the host's decoder).
    // `..._PICTURE_HASH` says which picture reached guest memory and what its mean luma was;
    // `..._DUMP_DIR` writes the picture out, which separates "the decoder produced black"
    // from "the conversion is wrong" from "the draw never sampled it".
    "VITASLOP_MOVIE_DUMP_DIR",
    "VITASLOP_MOVIE_DUMP_EVERY",
    "VITASLOP_MOVIE_PICTURE_HASH",
    // Withholds a movie's AUDIO units from the title: the A/B arm for any title whose
    // own demux behaves differently once a movie turns out to have a second stream.
    // Opens a different movie than the title asked for. The one way to exercise the movie
    // AUDIO path without first playing most of a game: a front-screen movie may have no
    // audio track while the ones that do are behind thousands of frames of menus.
    "VITASLOP_MOVIE_SUBSTITUTE",
    "VITASLOP_MP4_AUDIO",
    // The falsifier for the voice-handle LOOKUP: with it off, every query for a rack's
    // voice allocates a fresh handle again, which is what left 8,138 voices in the bank and
    // 318 of them playing every grain. It is here so a device can price the difference.
    "VITASLOP_NGS_VOICE_HANDLE_MEMO",
    "VITASLOP_NO_BC",
    "VITASLOP_NO_INLINE_CLIB",
    // The A/B switch for the whole inline-import mechanism. Reachable from the browser for the
    // same reason `VITASLOP_DBG_CALLSITES` is, and more sharply: inlining exists to stop paying
    // for the host-call CROSSING, the crossing is a large share of a phone frame and a small one
    // here, so "what did inlining buy" is a question only the browser can answer honestly. It is
    // read at LINK time, so it must be set before the run starts, not toggled during it.
    "VITASLOP_NO_INLINE_IMPORTS",
    // The SCOPED version of the switch above, for the lightweight-mutex lock/unlock pair.
    // Reachable from the browser for a sharper reason than the whole-mechanism one: turning
    // everything off moves ~11,000 calls a frame and every preemption point with them, so a
    // family worth ~1,000 calls cannot be priced against that baseline. This one changes
    // nothing else. Read at LINK time; set it before the run, not during it.
    "VITASLOP_NO_INLINE_LWMUTEX",
    // The SCOPED switch for the two default-uniform RESERVES, which are the largest single
    // family of host calls a gameplay frame still made before they were inlined (1,189 a
    // frame on one title, 53% of everything it calls). Reachable from the browser for both
    // of the reasons the two switches around it are separately: the phone is the only machine
    // where a count-based win is worth measuring, and this form is the first that hands the
    // guest an ADDRESS rather than answering a question, so it is the one to fall back to if
    // a title's uniforms ever look wrong. Read at LINK time; set it before the run.
    "VITASLOP_NO_INLINE_RESERVE",
    // The SCOPED switch for the constant-return STUBS (the NGS patch/volume calls and their
    // neighbours, 32% of a race frame's host calls on one title). Reachable from the browser
    // for the usual price-tag reason, and for a diagnostic one the other switches do not have:
    // an inlined call leaves the call histogram, and the histogram is how "which unimplemented
    // calls does this title make" is answered. Read at LINK time; set it before the run.
    "VITASLOP_NO_INLINE_STUBS",
    // The SCOPED switch for the fragment-texture bind. Reachable from the browser because it
    // is not a perf question at all: the inline copy form replaced a handler that a title's
    // every texture bind went through, so if a texture goes MISSING on a device, this knob is
    // the one-run answer to "is it the inline form" - and the device is the only place the
    // report came from. Read at LINK time; set it before the run, not during it.
    "VITASLOP_NO_INLINE_TEXTURE",
    // The SCOPED switch for `sceGxmSetUniformDataF`, which after every other GXM inlining is
    // the largest single call a real title still makes (1,106 a frame on a race, 58% of the
    // remainder). Reachable from the browser because that is where a count-based win is
    // worth having, and because this form writes the bytes a SHADER READS - a fault in it is
    // a wrong picture, and the phone is where wrong pictures have been reported from. Read
    // at LINK time; set it before the run.
    "VITASLOP_NO_INLINE_UNIFORM_DATA",
    // The A/B arm that turns the NGS decode-and-mix OFF. Browser-reachable because that is
    // where the audio path had to be priced - it decodes and mixes up to ~100 voices a grain,
    // which is guest-CPU work on the machine that has the least of it.
    "VITASLOP_NO_NGS_MIX",
    "VITASLOP_PERF",
    // Whether the per-window performance report is ALSO written to the browser console. The
    // panel and the sink always get it; this is the console copy, off by default because the
    // page is the product and eight multi-line blocks a window is a firehose. Overridable
    // because a HARNESS reads the console and nothing else - a browser measurement with this
    // unset reports a frame rate and no breakdown, which is the sixth instance of this
    // omission and cost a run to find.
    "VITASLOP_PERF_CONSOLE",
    // Sample the PRESENTED surface every N presents and describe it in the diagnostics panel.
    // Reachable from the browser because it exists for a defect only the browser has: a blank
    // screen over a healthy set of render counters, where the fault is either a blank picture
    // or a picture the compositor never showed and no counter upstream of the surface can say
    // which. The device has no screenshot tool and no console, so the answer has to arrive as
    // text in the panel. Read when the surface is configured; set it before the run.
    // Break `prepare`'s milliseconds down INSIDE one draw (hash / repack / arena copy /
    // uniforms / samplers / depth) and count the bytes each phase moved. Reachable from the
    // browser because that is where a per-draw cost is amplified, and gated because it is the
    // one instrument here that reads a CLOCK on a path making no WebGPU call - six reads a
    // draw across several hundred draws a frame would move the number they report.
    "VITASLOP_PREPARE_SPLIT",
    "VITASLOP_PRESENT_PROBE",
    // Hold the ARM register file in wasm LOCALS along each straight-line run instead of on
    // its globals (`transpiler::promote`). Reachable from the browser because the browser is
    // the ONLY place it can be priced: promotion adds operators and removes none, so fuel,
    // the code-expansion factor and the guest clock are all blind to it by construction, and
    // matched-frame V8 wall-clock is the only instrument left. An emit-time knob, so it is
    // read once before the transpile rather than by a running thread.
    "VITASLOP_PROMOTE_REGS",
    // Whether PVRTC decodes a whole face at a time or one texel at a time. Reachable from
    // the browser because that is where PVRTC decode volume costs the most, so that is where
    // the exactness falsifier has to be runnable.
    "VITASLOP_PVRTC_DECODE",
    // Keep geometry that has not changed since the renderer first saw it RESIDENT on the GPU
    // instead of copying it into a per-frame arena and uploading it again. `0` sends every draw
    // back through the arenas, which is the A/B arm. Reachable from the browser because that is
    // where a per-frame upload costs the most.
    "VITASLOP_RESIDENT_GEOM",
    // The byte budget for each of the two resident geometry heaps, in MB (default 48). A heap
    // that fills at its budget is RESET and says so; a reset every few frames means the working
    // set does not fit and this is the number to change.
    "VITASLOP_RESIDENT_GEOM_MB",
    "VITASLOP_RTT_BG_CACHE",
    // DIAGNOSTIC, not a shipped behaviour: cap how many runnable threads may hold the baton,
    // by priority, as the console's core count would. Unset (the default) keeps the current
    // discipline, where the spin cooldown eventually admits every runnable thread whatever its
    // priority - which is why a below-third-priority thread can run in a quantum the hardware
    // would not have given it. Set to 3 to ask whether a bug depends on that.
    //
    // It is a knob rather than a default because a strict core cap can LIVELOCK on priority
    // inversion (a high-priority thread spinning on a lock a capped-out low-priority thread
    // holds), and the cooldown it replaces is the anti-starvation mechanism. Answering the
    // question is worth a run that may hang; shipping it is not, until it has an escape hatch.
    // The negative control for narrowing a draw's texture decode to the units its fragment
    // program DECLARES. Browser-reachable because the cost it removes - a decode per bound slot
    // per draw - is only large where the engine runs at wasm speed.
    "VITASLOP_SAMPLER_NARROW",
    "VITASLOP_SCHED_CORES",
    // Round-robin the scheduler's pick instead of the priority discipline, and TRACE what it
    // picked. Both browser-reachable because the scheduler behaves differently there (JSPI
    // suspends, no fuel) and a scheduling question asked on the desktop answers about the
    // desktop.
    "VITASLOP_SCHED_RR",
    "VITASLOP_SCHED_TRACE",
    // Fold the determinism signature on a browser recipe run that does NOT declare `@sig`.
    // The fold hashes every retired scene's vertices, indices and uniforms - about 3 MB a
    // frame on a race, MEASURED at 7.7% of the guest window - and the only consumer is an
    // `@sig` assertion, so a recipe without one pays for a number nothing compares. Set this
    // when the point of the run is to LEARN the signature and bless it into a recipe.
    "VITASLOP_SIGNATURE",
    // How often the RUNNING signature is printed (`sigtrace f<frame> <hash>`), so a
    // browser-only divergence can be bisected against the desktop's identical line instead of
    // by re-running the pair per halving.
    "VITASLOP_SIGNATURE_EVERY",
    // Microseconds of artificial cost added to every guest frame, so a machine with headroom
    // can exercise the live loop's behind-the-clock pacing. Browser-reachable because the loop
    // it tests only exists there, and because the device this models has no console.
    "VITASLOP_SLOW_FRAME_US",
    // Byte budget for retained texture snapshots. Reachable from the browser because
    // exceeding it there costs a full re-decode of the working set in one frame.
    "VITASLOP_SNAPSHOT_BUDGET_MB",
    // Path of the font that STANDS IN for the console's system font, which is not shipped.
    // Listed here so the name is not a browser-boot panic, though the browser cannot open a
    // path: there it supplies the bytes directly instead (`font::system::set_bytes`).
    "VITASLOP_SYSTEM_FONT",
    // How often a retained texture snapshot is re-checked against guest memory: `scene`
    // (the default, exact) or `frame` (faster, one scene of staleness the first time a
    // texture changes). Reachable from the browser because that is where the check costs
    // the most.
    "VITASLOP_TEXTURE_CHECK",
    "VITASLOP_TEX_CACHE_MB",
    // Turns the compressed-texture upload OFF, for an A/B against the plain decode. That is the
    // whole surface: the feature is ON, its measurement PRINTS, and there is nothing here to
    // turn on. It began as four knobs - a passthrough switch, a transcode switch defaulting to
    // OFF, a mip probe and a working-set report - and every one of them was a decision or a
    // measurement that belonged in the default path rather than behind a flag the user would
    // never set. Browser-reachable because the A/B is only interesting on the device whose GPU
    // allocation fails at 274 MB and draws WHITE.
    "VITASLOP_TEX_COMPRESS",
    // The A/B arm for the texture memos surviving a scene: `1` restores the per-scene clear.
    // Browser-reachable because the desktop CANNOT see this win (0.28 ms/frame under a 5% noise
    // floor) - the browser pays it as sceGxmDraw* handler time at wasm speed, which is the
    // entire point of the change.
    // How much of a re-read texture the guest had actually written, and the falsifier for
    // the page-granular re-read that census motivated.
    "VITASLOP_TEX_DIRTY_CENSUS",
    "VITASLOP_TEX_MEMO_PER_SCENE",
    // Names the guest code that WROTE a uniform, by parameter name, with its `lr`. Needed in
    // the browser because `screenTintColour` - the white-out - is written there and never on
    // the desktop.
    "VITASLOP_TEX_PAGE_READ",
    "VITASLOP_UNIFORM_WATCH",
    // The A/B arm for the vblank SPIN GUARD (`=0` restores the bare mirror read). Here
    // because the browser is where the spin costs the most - 26% of all translated guest
    // code on a retail racer's race - and because a default that cannot be turned off on
    // the machine that pays for it cannot be measured at all.
    "VITASLOP_VBLANK_PARK",
    // The guest-address name section. Browser-reachable so a V8 CPU profile taken in the
    // browser can NAME the guest functions it samples instead of reporting `wasm-function[N]`.
    // The arm for vertex interning: giving a rebuilt-but-identical vertex stream the identity
    // of the buffer already held. Browser-reachable because what it buys is identity-keyed
    // caches hitting instead of a whole-stream hash and memcmp per draw, which only costs where
    // the engine runs at wasm speed.
    "VITASLOP_VERTEX_INTERN",
    "VITASLOP_WASM_NAMES",
];

/// The override map, consulted by every reader in this module BEFORE the environment.
fn overrides() -> &'static Mutex<HashMap<String, String>> {
    static CELL: OnceLock<Mutex<HashMap<String, String>>> = OnceLock::new();
    CELL.get_or_init(|| Mutex::new(HashMap::new()))
}

/// Read a knob: the process-wide override if one is set, else the environment.
pub fn var(name: &str) -> Result<String, std::env::VarError> {
    if let Some(v) = overrides().lock().unwrap_or_else(|e| e.into_inner()).get(name) {
        return Ok(v.clone());
    }
    std::env::var(name)
}

/// Whether a knob is SET at all, regardless of value - the `std::env::var_os(..).is_some()`
/// shape most flags here use.
pub fn var_os(name: &str) -> Option<String> {
    if let Some(v) = overrides().lock().unwrap_or_else(|e| e.into_inner()).get(name) {
        return Some(v.clone());
    }
    std::env::var_os(name).map(|v| v.to_string_lossy().into_owned())
}

/// The `tracing` filter directive every vitaslop binary installs: `VITASLOP_LOG` if it is set
/// and non-empty, else `RUST_LOG`, else nothing.
///
/// `VITASLOP_LOG` is the primary name because it is the only one that works on BOTH engines:
/// the browser has no environment to read and takes its knobs through the override table
/// above, which is keyed by `VITASLOP_*` names. Every note and repro command in this project
/// is written against it, so a desktop binary answering only to `RUST_LOG` turns a documented
/// invocation into silence - which is indistinguishable from a diagnostic that never fires,
/// and that is exactly how it was found. `RUST_LOG` still works, for the ordinary Rust habit.
pub fn log_filter() -> String {
    match var("VITASLOP_LOG") {
        Ok(v) if !v.is_empty() => v,
        _ => match std::env::var("RUST_LOG") {
            Ok(v) if !v.is_empty() => v,
            // WARN by default, not silence. Everything this engine approximates is required
            // to report itself - a shader pair that falls back to fixed-function, a dropped
            // draw, an unplaced scene - and those reports are `warn`. With an empty filter
            // they were all discarded, so the DEFAULT invocation was the one that could not
            // see them: a run that quietly drew a whole title through the fallback looked
            // exactly like a run that recompiled it. A knob nobody sets is not a report.
            _ => "warn".to_string(),
        },
    }
}

/// A boolean knob: set means on, EXCEPT for the values that plainly mean off.
///
/// # `NAME=0` used to mean ON, and that cost a whole measurement
/// This was a pure presence flag - set to anything, including `0` and the empty string,
/// meant on - on the reasoning that that is what a shell does. It is also what nobody
/// expects, and the failure is silent in the worst way: an A/B needs an OFF arm, and
/// `VITASLOP_PROMOTE_REGS=0` typed into the page's knobs box produced a promoted build in
/// BOTH arms. The run compared a build against itself and reported no difference - a clean,
/// plausible, meaningless zero (27 matched frame pairs, +0.12%, median ratio exactly
/// 1.0000). Nothing about it looked wrong.
///
/// The same trap was latent on `VITASLOP_GXP_LIVE`, the renderer's master switch, where
/// `=0` would have turned the recompiler ON.
///
/// So the off values are honoured: `0`, `false`, `no`, `off` (any case, surrounding space
/// ignored). Everything else set - including the empty string, which is how a shell writes
/// "present" - is on, so `NAME=` and `NAME=1` are unchanged. The transpiler's own
/// environment readers already worked this way; this makes the two agree.
pub fn flag(name: &str) -> bool {
    match var(name) {
        Err(_) => false,
        Ok(v) => !matches!(v.trim().to_ascii_lowercase().as_str(), "0" | "false" | "no" | "off"),
    }
}

/// Set a knob for this process, for a platform where the environment cannot be.
///
/// Panics on a name not in [`OVERRIDABLE`] - see the module docs for why a silently
/// ignored override is worse than a missing one.
pub fn set_override(name: &str, value: &str) {
    assert!(
        OVERRIDABLE.contains(&name),
        "{name} is not overridable - its reader still calls std::env::var directly. \
         Route that reader through vitaslop_platform::knobs and add the name to \
         vitaslop_platform::knobs::OVERRIDABLE."
    );
    overrides()
        .lock()
        .unwrap_or_else(|e| e.into_inner())
        .insert(name.to_string(), value.to_string());
    // The shader emitter keeps its own arm table: `vitaslop-gxp-shader` has NO dependencies on
    // purpose (it is the wasm-safe, game-data-free half of the renderer), so it cannot read
    // this one. Forwarding here is what makes those arms reachable from the browser at all,
    // and `set_arm` ignores every name but its own two.
    #[cfg(feature = "gpu")]
    vitaslop_gxp_shader::link::set_arm(name, value);
}

#[cfg(test)]
mod tests {
    use super::*;

    /// The list is the contract `set_override` enforces, so a duplicate or an unsorted
    /// entry is a maintenance hazard rather than a style point: both make "is this name
    /// already here?" answerable only by reading every line.
    #[test]
    fn overridable_is_sorted_and_unique() {
        let mut sorted = OVERRIDABLE.to_vec();
        sorted.sort_unstable();
        sorted.dedup();
        assert_eq!(sorted, OVERRIDABLE);
    }

    /// `NAME=0` must mean OFF.
    ///
    /// This is pinned because the opposite behaviour is invisible when it is wrong: a
    /// boolean knob that reads `0` as ON turns the OFF arm of an A/B into a second copy of
    /// the ON arm, and the run then reports "no difference" - which is exactly what a
    /// correct null result looks like. It cost a full measurement once already.
    #[test]
    fn a_boolean_knob_reads_zero_and_friends_as_off() {
        // `set_override` is the platform-independent way in, and every name it accepts must
        // be in OVERRIDABLE - so this uses one that is.
        const NAME: &str = "VITASLOP_GXP_LIVE";
        for off in ["0", "false", "no", "off", "OFF", " 0 ", "False"] {
            set_override(NAME, off);
            assert!(!flag(NAME), "{off:?} must read as OFF");
        }
        for on in ["1", "", "yes", "true", "2"] {
            set_override(NAME, on);
            assert!(flag(NAME), "{on:?} must read as ON");
        }
    }

    #[test]
    fn an_override_is_visible_to_every_reader_shape() {
        set_override("VITASLOP_GXP_LIVE", "1");
        assert_eq!(var("VITASLOP_GXP_LIVE").as_deref(), Ok("1"));
        assert_eq!(var_os("VITASLOP_GXP_LIVE").as_deref(), Some("1"));
        assert!(flag("VITASLOP_GXP_LIVE"));
    }

    /// The empty string is how a shell spells "set but valueless", and the house flag
    /// convention is presence, not truthiness - so it must read as ON.
    #[test]
    fn an_empty_override_still_reads_as_set() {
        set_override("VITASLOP_GXP_SOLID", "");
        assert!(flag("VITASLOP_GXP_SOLID"));
    }

    #[test]
    #[should_panic(expected = "not overridable")]
    fn setting_an_unrouted_knob_panics() {
        set_override("VITASLOP_NOT_ROUTED_ANYWHERE", "1");
    }
}
