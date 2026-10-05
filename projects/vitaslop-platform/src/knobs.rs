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
    // `0`: pass arenas are never dropped (grow-only for the run, the old behaviour).
    "VITASLOP_ARENA_TRIM",
    "VITASLOP_ARENA_UPLOAD_PROBE",
    "VITASLOP_ARM_AT_FRAME",
    // `0`: recompiled pipelines are created synchronously in the frame that first draws them
    // (the old rule), not started with `createRenderPipelineAsync` ahead of it - see
    // `GxmRenderer::warm_pipelines`.
    "VITASLOP_ASYNC_PIPELINES",
    // `0`: sceAudioOutOutput sleeps one grain from the call instead of following the port schedule.
    "VITASLOP_AUDIO_SCHEDULE",
    // `0`: sceAudioOutOutput paces on the guest's virtual clock instead of the host's wall.
    "VITASLOP_AUDIO_WALL",
    // Wall-ms cap on the browser SMP run worker's wait for a movie decoder to answer down to
    // the stream's reorder depth before a decode call; `0` is the arm back (no wait). See
    // `vitaslop_runtime::vita::avcdec::decoder_behind`.
    "VITASLOP_AVCDEC_CATCH_UP_MS",
    // An exiting guest thread's stack is unwound so its instance is pooled (browser_sched
    // `exit_unwinds`); `0` parks it forever and drops the instance, as before.
    "VITASLOP_BROWSER_EXIT_UNWIND",
    "VITASLOP_BROWSER_FASTFORWARD",
    "VITASLOP_BROWSER_FUEL",
    "VITASLOP_BROWSER_HEARTBEAT_MS",
    // Whether a finished guest thread's module instance may be reused. On by default and
    // overridable because it is the single variable that decides whether a title creating
    // a thread per frame instantiates the whole module sixty times a second.
    "VITASLOP_BROWSER_INSTANCE_POOL",
    // How many guest-module instances each SMP worker builds AHEAD of any thread (default 2;
    // `0` = none): a new thread's first run otherwise pays 9-21 ms of instantiation on a phone.
    "VITASLOP_BROWSER_INSTANCE_RESERVE",
    "VITASLOP_BROWSER_QUANTUM_CALLS",
    // The frame a FAST-FORWARD starts RENDERING at, even while it is still unpaced. The
    // browser's half of `VITASLOP_HEADLESS_RENDER_FROM`: render history is state, and a run
    // that renders nothing before its window samples render targets that are empty.
    "VITASLOP_BROWSER_RENDER_FROM",
    // `=1`: give the guest a linear memory OF ITS OWN, exactly sized, reached through typed
    // arrays - the form before the guest region moved inside the emulator's memory (see
    // `vitaslop_web::reserve_guest_region`). The diagnostic arm: a guest pointer past its
    // region traps here instead of reading the emulator's heap. Read by the RUN worker
    // before it reserves, so it must be set for the run, not the link.
    "VITASLOP_BROWSER_SPLIT_MEMORY",
    "VITASLOP_BROWSER_SUPERSAMPLE",
    // `=1`: a measurement mode - the live loop runs the next frame as soon as the last one is
    // presented, never waiting for the wall clock, so `fps` reads capacity (speed goes past
    // 100%). How two engines with headroom are compared; audio overruns by design.
    "VITASLOP_BROWSER_UNPACED",
    // The negative control for what `RenderSceneBuilder` stopped re-deriving and stopped
    // timing every frame - see `render::build_fastpath`. Overridable because the only rig
    // that can price it is a browser run.
    "VITASLOP_BUILD_FASTPATH",
    // `<from>-<to>` display frames: the diagnostics panel's CALL TABLE - every distinct
    // (host call, thread, return address) with counts. See `vitaslop_runtime::call_table`.
    "VITASLOP_CALL_TABLE",
    // `1`: the CALL TABLE also prints r0..r12 at each row's first occurrence.
    "VITASLOP_CALL_TABLE_REGS",
    // `0` drops an unpresented frame's scenes whole instead of carrying its offscreen renders (lib.rs).
    "VITASLOP_CARRY_UNPRESENTED",
    // `VITASLOP_SMP`: `0` lets the game clock fall behind real time (no wall-clock floor).
    "VITASLOP_CLOCK_WALL_FLOOR",
    // Per-tick clamp (ms) of that floor; held (gated/paused) time is never counted.
    "VITASLOP_CLOCK_WALL_STEP_MS",
    // `0` = the negative control for the sparse-draw compaction (the min..max window read).
    "VITASLOP_COMPACT_SPARSE",
    // Mirror the run's status notes and heartbeat to the browser console. Off by default:
    // a clean run's console shows warnings and nothing else; the panel carries the rest.
    "VITASLOP_CONSOLE",
    // Start the scheduler CPU-share counters at this frame - see `sched::cpu_share_from`.
    "VITASLOP_CPU_SHARE_FROM",
    // `1` = a blocking sceCtrlReadBuffer* returns only the samples since the last read.
    "VITASLOP_CTRL_READ_NEW",
    // Hard-pause the emulator when the window loses focus or the tab is hidden. ON by
    // default; `0` keeps the guest running unfocused. Read on both engines: the desktop's
    // window event and the browser page's visibility/blur handlers.
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
    // Every `sceKernelDelayThread` tallied by (call site, requested microseconds). Browser-
    // reachable for the same reason the call-site profiler is: a polling thread's cost is
    // iterations times crossings, and a crossing costs twenty times more on the phone than
    // here - so the phase where the count matters is the one only that engine can be asked.
    // Read a draw's vertices and indices at `sceGxmEndScene` rather than at the draw call -
    // ON by default. Browser-reachable because it is a DEFAULT-BEARING arm over what the
    // engine reads, not a diagnostic, and the browser is where the picture is judged.
    "VITASLOP_DEFER_GEOMETRY",
    // The negative control for skipping the draw-time snapshot of a guest-memory window the
    // end-of-scene resolve replaces anyway - see `vitaslop_runtime`'s `defer_window_bytes`.
    "VITASLOP_DEFER_WINDOW_BYTES",
    "VITASLOP_DELAY_CENSUS",
    // The A/B arm for the guest-store dirty map's RUN COALESCER: `=0` gives every store its
    // own mark again. Browser-reachable because the browser is where it has to be priced -
    // the coalescer only removes operators, so neither the game clock (billed in guest
    // instructions) nor the expansion factor can see it, and V8 wall-clock is the only
    // instrument that can. VALUE-sensitive: anything but `0` is ON.
    "VITASLOP_DIALOG_RUNNING_FRAMES",
    // `0`: an SMP build's guest-store mark stores unconditionally again (the false-sharing fix
    // off; transpiler `emit::dirty_mark_tested`).
    "VITASLOP_DIRTY_MARK_TEST",
    "VITASLOP_DIRTY_RUN_MARK",
    // The dispatch ABLATION: route even a fallthrough through the function's `br_table`.
    // Browser-reachable because the question it answers is a V8 branch-prediction question -
    // the module carries one indirect branch per 10.5 guest instructions and nothing this
    // project usually counts can price one.
    "VITASLOP_DISPATCH_ALL",
    // Every draw of the listed frames (program, blend, vertices, uniforms, textures) to the console (lib.rs).
    "VITASLOP_DRAW_NOTE_AT",
    // With VITASLOP_DRAW_NOTE_AT: only scenes whose colour target is <w>x<h> (lib.rs).
    "VITASLOP_DRAW_NOTE_TARGET",
    // With VITASLOP_DRAW_NOTE_AT: also every bound texture's bytes, hex (lib.rs).
    "VITASLOP_DRAW_NOTE_TEX",
    // The A/B arm for `emit_flags_add`'s carry and overflow forms. It is here because the
    // BROWSER is the engine that has to answer: `flags-add` was 39% of every operator the
    // transpiler emitted, the closed forms cut the module 5.3% and executed operators 8.7%,
    // and three interleaved desktop repeats put the wall-clock difference inside the noise.
    // Encode only draws `lo..=hi` of every pass. Browser-reachable because the question it
    // answers - "which draw put this on screen, and which one covered it" - is asked of a
    // PICTURE, and the pictures that need it are the ones only a device or a browser produces.
    // `VITASLOP_CHAIN_LIMIT` bisects by PASS and cannot touch a title whose frame is one pass.
    "VITASLOP_DRAW_RANGE",
    // The per-draw dump of one display frame, readable from the browser's knob box.
    "VITASLOP_DUMP_DRAW_GXP",
    // The TOP arm of the non-suspending-trap A/B: the DERIVED fast set - every `cont!`
    // `VITASLOP_SMP`: serve parked GPU waits before each present, waiting up to <ms> for one; -1 = after it.
    "VITASLOP_EARLY_GRACE_MS",
    // dispatch arm, ~870 NIDs - instead of the 23 hand-picked ones that are still the default.
    // `1`: an early completion waits for every in-flight readback of its targets (old rule).
    "VITASLOP_EARLY_WAIT_ANY",
    // Browser-reachable for the same reason `VITASLOP_NO_FAST_IMPORT` is - the two traps only
    // differ there, and the JSPI stack switch this removes is what a phone pays most for - and
    // together with that knob it gives one build all three points: nothing fast, today's 23,
    // and every admissible arm. Read at LINK time; set it before the run, not during it.
    "VITASLOP_FAST_IMPORT_CURATED",
    // `1`: a fiber's backing thread runs at its runner's priority and mask (default: priority 0xA0, any core).
    "VITASLOP_FIBER_RUNNER_PRIORITY",
    "VITASLOP_FLAGS_WIDE_C",
    // The one frame whose per-SCENE digests are printed, so a cross-engine difference lands on
    // a PASS instead of on a whole frame.
    "VITASLOP_FRAME_CAPSULE",
    "VITASLOP_FRAME_DIGEST",
    "VITASLOP_FRAME_TOPUP",
    // The GPU budget's arm back (`=0`), and the desktop rig that saturates a GPU so the budget
    // can be exercised off the phone - see `LivePlayback::gpu_budget` and `GxmRenderer::gpu_burn`.
    "VITASLOP_GPU_BUDGET",
    "VITASLOP_GPU_BURN",
    // `1`: the run worker counts live WebGPU objects by kind from the JS side and logs a
    // `[gpu-census]` line every 10 s (web/gpu-census.js). Read by the worker, not by Rust.
    "VITASLOP_GPU_CENSUS",
    // How many GPU submits may be in flight before a present declines to make another.
    // Default 2, `0` disables the bound. Browser-reachable because the browser is the only
    // engine where an unbounded queue is a HANG rather than latency: `queue.write_buffer`
    // blocks the worker thread when the staging ring cannot retire, and a blocked worker turns
    // no event loop at all. See `LivePlayback::gpu_in_flight`.
    "VITASLOP_GPU_QUEUE_DEPTH",
    // `1` = the GPU budget's stale rule as it was (a second without a new measurement, even
    // with no readback in flight) - the arm back for the latch fix.
    "VITASLOP_GPU_STALE_OLD",
    // `1` lists every pass in the GPU TIME report, in frame order (default: the costliest five).
    "VITASLOP_GPU_TIME_ALL",
    // The clock's core model, so a browser run can be A/B'd against native without an
    // environment to set it in.
    "VITASLOP_GUEST_CORES",
    // `VITASLOP_SMP`: `<from frame>:<frames>` samples which guest function each worker runs (on-device profiler).
    "VITASLOP_GUEST_PROF",
    // The per-pass arena's minimum size in KB. Browser-reachable because the only machine
    // whose `write_buffer` stalls on a growing arena is the user's phone.
    "VITASLOP_GXM_ALPHA_SINGLE",
    "VITASLOP_GXM_ARENA_FLOOR_KB",
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
    // Per-draw occlusion counts: how many samples each draw of the frame actually wrote. See
    // `gpu::DrawCoverage`. Registered here so a browser run that sets it does not panic on
    // boot, though the READBACK is native-only and the run says so.
    // The destination-colour SPLIT stopwatch: alternate the pass cut on and off every <n> frames
    // within one run and report the GPU time each arm measured. Browser-reachable because the
    // question it answers - what a tile store/reload costs - only a TILING GPU can answer.
    // `=1` pools the six per-pass staging arenas across passes, capacity bounded, instead of a
    // fresh `Vec::new()` each. OFF by default: it buys the desktop 0.2 ms and the BROWSER
    // nothing, and the unbounded version of it cost the target device real frame rate. Reachable
    // from the browser because a DEVICE run is the only thing that can settle it, and a desktop
    // measurement already failed to [[vitaslop-desktop-cannot-price-a-count-win]].
    "VITASLOP_GXM_ARENA_POOL",
    // What share of arena uploads repeat last frame's bytes - the residency question, asked
    // of the engine that pays for it.
    "VITASLOP_GXM_ARENA_REPEAT",
    "VITASLOP_GXM_BACKGROUND_DEPTH",
    // `0` is the arm back to the old (wrong) colour-mask bit order - see `gpu::gxm_color_mask`.
    "VITASLOP_GXM_COLOR_MASK_ORDER",
    "VITASLOP_GXM_DEST_SPLIT_AB",
    "VITASLOP_GXM_DRAW_COVERAGE",
    "VITASLOP_GXM_FLOAT_TARGETS",
    "VITASLOP_GXM_FLOAT_TARGETS_WIDE",
    // `0`: a frame whose last scene is not a flipped (display) buffer shows that LAST SCENE
    // (an offscreen pass) - the old rule. See `GxmRenderer::display_choice`.
    "VITASLOP_GXM_HOLD_FLIP",
    "VITASLOP_GXM_NO_MULTISAMPLE",
    // Restores the pre-2026-09-12 behaviour in which the frame's FIRST pass into an offscreen
    // render target CLEARED it, discarding what earlier frames had rendered there. The default
    // is to LOAD, which is what a GXM tiler does at `sceGxmBeginScene`. See
    // `gpu::rtt_clear_every_frame` and `GxmRenderer::rtt_ever_rendered`.
    "VITASLOP_GXM_RTT_CLEAR_EVERY_FRAME",
    // The render-target writeback cap in texels; `0` is the arm back. See
    // `vitaslop_runtime::rtt_writeback`.
    "VITASLOP_GXM_RTT_WRITEBACK",
    // The arm back to a fresh vertex program per identical create - see `gxm::create_vertex_program`.
    "VITASLOP_GXM_SHARE_PROGRAMS",
    // `0` is the arm back to the pre-staging texture upload path. Its own doc says the browser
    // is the ONLY place the old path's hundreds of milliseconds are visible - and it was not
    // listed here, so setting it in the browser PANICKED the run on boot. An arm that cannot be
    // taken where the effect lives is not an arm.
    "VITASLOP_GXM_STAGING",
    "VITASLOP_GXM_STALE_UNIFORMS",
    // `=1` runs the packed-geometry budget check and the promotion-map sweep every frame
    // instead of every eighth - the OFF arm of that cut.
    // `0` rebinds a draw's whole state even when nothing changed - the off arm for the
    // per-pass redundant-state elimination in `gpu::BoundState`. Browser-reachable because the
    // browser frame is where the encode cost lands and the phone is where a state change is
    // dearest.
    "VITASLOP_GXM_STATE_DEDUP",
    "VITASLOP_GXM_SWEEP_EVERY",
    // Poisons a freshly reserved default uniform buffer, so a lane the guest never wrote is
    // distinguishable from one it wrote as zero. NOTE it only covers the RESERVE path, never a
    // precomputed state's guest-owned buffer - so its silence is not evidence until the pattern
    // is seen SOMEWHERE. Browser-reachable because the value it has to decide about
    // (`screenTintColour`) only ever appears there.
    "VITASLOP_GXM_TEX_UNWRITTEN",
    "VITASLOP_GXM_UNIFORM_POISON",
    // `0` stops the guest's `sceGxmSetViewport` reaching the FIXED-FUNCTION GPU arm and the
    // SOFTWARE rasteriser - the negative control for that change, scoped to the two paths it
    // touched (the recompiled arm always applied it). See `gpu::viewport_reaches_fixed_function`.
    "VITASLOP_GXM_VIEWPORT",
    "VITASLOP_GXM_ZLS",
    // The negative control for 4-ALIGNING each stream's column in the interleaved vertex row and
    // the row's own stride. Without it a stream whose column is not a multiple of four puts every
    // later attribute at an offset WebGPU refuses, and the guest's row has to be copied rather
    // than bound - 154 of one title's 185 pipelines. See `vitaslop_runtime`'s `align_packed_row`.
    "VITASLOP_GXP_ALIGN_ROW",
    "VITASLOP_GXP_ALLOW_FIXED_FUNCTION",
    "VITASLOP_GXP_ARENA_RING",
    // What an attribute lane the vertex stream does not supply is FILLED with. Browser-reachable
    // because the fill value is a picture question and the phone is where wrong pictures are
    // reported from.
    "VITASLOP_GXP_ATTR_FILL",
    // The negative control for laying the interleaved vertex row out per ATTRIBUTE rather than
    // per stream column - see `vitaslop_runtime`'s `attr_packed_row`.
    "VITASLOP_GXP_ATTR_ROW",
    // Which sampler descriptors the bind-group builder DECODED out of guest bytes and which
    // fell back. Browser-reachable because a wrong texture is reported from the phone.
    "VITASLOP_GXP_BIND_TRACE",
    // `0`: a bank read dynamically keeps all 512 registers - `link::bound_dynamic_reads`.
    "VITASLOP_GXP_BOUND_READS",
    // The capsule capture. Reachable from the browser like every other knob here, but the
    // WRITE will fail there - a browser worker has no filesystem to put a capsule on - and the
    // capture reports that failure by name rather than dropping the draw in silence.
    "VITASLOP_GXP_CAPSULE",
    "VITASLOP_GXP_CAPSULE_MIN_INDICES",
    "VITASLOP_GXP_CAPSULE_SKIP",
    // The interpreter's SA file after the clip sampler has run the prologue - see
    // `gpu::count_clip_w_signs`. Overridable so a browser run can ask it too.
    "VITASLOP_GXP_CLIP_DUMP_SA",
    // The OFF arm of the clip-verdict retry backoff: re-measure a no-evidence pair on EVERY
    // appearance. It was unreachable from the browser, which is where the cut was aimed.
    "VITASLOP_GXP_CLIP_RETRY_EVERY_FRAME",
    "VITASLOP_GXP_CULL",
    "VITASLOP_GXP_DEPTH24",
    "VITASLOP_GXP_DEST",
    "VITASLOP_GXP_DEST_BLEND",
    // `0` = the arm back for the repeating-DP bit-48 rule (both internal sources walk when bit 47
    // is clear) - two normalised vectors in one repeat; see `usse::decode::repeat_operands`.
    "VITASLOP_GXP_DP_B48_BOTH",
    // `0` = the arm back for the bit-47 DP repeat rule (intrinsic four-register source walk for
    // every repeating DP). A decode rule that changes skinned meshes on three titles has to be
    // switchable where the phone runs it. Forwarded to the emitter by `set_override`.
    "VITASLOP_GXP_DP_MOE_BIT47",
    // Whether a destination-reading fragment program linear in the destination is lowered
    // to a dual-source blend instead of a pass split; `0` is the arm back.
    "VITASLOP_GXP_DUAL_SOURCE",
    "VITASLOP_GXP_DUMP",
    // The negative control for eliding a draw whose colour is provably the destination
    // unchanged (default ON). Browser-reachable because the saving is the PHONE's - it is fill
    // in a GPU-bound world pass - so the arm has to be switchable where it is felt.
    "VITASLOP_GXP_ELIDE_NOOP",
    "VITASLOP_GXP_EXCLUDE",
    // `1`/`0`: the f16 saturation test as float compares (browser default) or the bit test (native).
    "VITASLOP_GXP_F16_FCMP",
    // The f16 rounding arm (`vitaslop_gxp_shader::link::F16_ROUND_ARM`): `0` = the device's own
    // `pack2x16float` mode. Missing here, setting it in the browser PANICKED the run worker at
    // "reserving the guest region" (`guest region: unreachable`, 2026-09-25).
    "VITASLOP_GXP_F16_RTE",
    "VITASLOP_GXP_FMEM",
    "VITASLOP_GXP_FORCE",
    // Whether a 16-bit guest index buffer stays 16-bit to the GPU. Browser-reachable because
    // the bytes it saves are `writeBuffer` bytes, and `writeBuffer` is the hottest single item
    // in a profile of the browser worker - the arm belongs where its price is measured.
    // `0` keeps every 16-bit register in a PACKED word, so each half read and write stays a
    // `pack2x16float`/`unpack2x16float` conversion - the off arm for the unpacked half-register
    // file (`vitaslop_gxp_shader::link::unpack_half_registers`). Browser-reachable because a
    // TILER is what pays for those conversions and a desktop compiler folds most of them away,
    // so the engine that can price this arm is the one on the phone.
    // The negative control for TELLING THE LINK WHAT THE GUEST BOUND, which is what lets an
    // attribute be fetched as an integer (`VertexAttribute::int_fetch`) and its surplus lanes be
    // baked as constants (`VertexAttribute::guest_components`). With it off both are unavailable
    // and the streams carrying either go back on the repack, which is what this did before.
    // Browser-reachable because the repack it removes is only large at wasm speed.
    "VITASLOP_GXP_GUEST_ATTRS",
    "VITASLOP_GXP_HALF_REGS",
    // The three decoder arms of the 2026-09-15a skinning work: the group-0x14 index load's
    // ORDINARY-REGISTER destination, the stride it multiplies its source by, and bit 7 as
    // comp0's high selector bit on a 16-bit PACK source. Browser-reachable for the reason every
    // other emission arm is - a picture that only goes wrong on the engine that ships cannot be
    // bisected from the desktop.
    "VITASLOP_GXP_IDX_MUL",
    "VITASLOP_GXP_IDX_REGDEST",
    // `0` = rcp/rsq as the bare WGSL builtins (their IEEE edges undecided) - see wgsl::IEEE_HELPERS.
    "VITASLOP_GXP_IEEE_RCP",
    // `1` = a 16-bit integer MAD's src1 read WHOLE when its half bit is clear (the old reading).
    "VITASLOP_GXP_IMAD_SRC1_WHOLE",
    "VITASLOP_GXP_INDEX16",
    // `0`: an indexed table's window sized as declared - `module::indexed_reach`.
    "VITASLOP_GXP_INDEXED_REACH",
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
    // How many DISTINCT INPUT SETS one pair may print under `VITASLOP_GXP_INPUTS` (default 8).
    // The report dedupes per (pair, inputs), which on a pair resubmitted with new uniforms every
    // draw is no dedupe at all: unaimed, it wrote 914 MB in eight minutes. Browser-reachable
    // like the report it bounds.
    "VITASLOP_GXP_INPUTS_SETS",
    "VITASLOP_GXP_INPUTS_VERTS",
    // Shade every pair a flat colour derived from its key, so ONE run says which pair owns
    // which region of the screen. Browser-reachable because that is where the regions that
    // need naming are: this title's black sideline characters, its green scorebar slot and its
    // black slab all reproduce in the browser and NOT natively, and without this the only
    // attribution instrument for them was one that cannot run there.
    "VITASLOP_GXP_KEYCOLOR",
    "VITASLOP_GXP_KEYS",
    "VITASLOP_GXP_LIVE",
    // Turns OFF the generated mip chain. Found missing by
    // `a_knob_routed_through_this_module_is_reachable_from_the_browser`, which is the FIFTH
    // instance of this omission - and it belongs here for the same reason `VITASLOP_TEX_COMPRESS`
    // does: the chain is a third of every uploaded RGBA8 texture's bytes, so on the device that
    // runs out of GPU memory it is both a memory lever and the A/B for whether the chain is what
    // prevents speckle.
    // Whether a 0xE8 memory load's REGISTER offset is read as 16 bits (the default) or
    // full-width. Browser-reachable because it decides a picture - one title's particle quads
    // read 64 KB past their window under the full-width reading - and a wrong picture is
    // reported from the phone.
    "VITASLOP_GXP_MEM_OFFSET16",
    // Print the guest words at a memory window's base and at base + this many bytes (hex),
    // once per (vertex program, buffer). What settled whether a load's addend is a byte
    // displacement at all - see `wgsl::emit_mem_load`.
    "VITASLOP_GXP_MEM_PEEK",
    // `0`: four consecutive memory-window word loads stay four `_word` calls (`module::quad_mem_reads`).
    "VITASLOP_GXP_MEM_QUAD",
    "VITASLOP_GXP_MIPS",
    // Report the first non-finite instruction per vertex program (`count_clip_w_signs`).
    "VITASLOP_GXP_NAN_SITES",
    // The portable f16 narrowing on a device that offers `shader-f16` - see `gpu.rs`.
    "VITASLOP_GXP_NATIVE_F16",
    "VITASLOP_GXP_NEGW",
    "VITASLOP_GXP_NOBLEND",
    "VITASLOP_GXP_NODEPTH",
    // The arm back to a depth-only attachment with no stencil - see `gpu::depth_format`.
    "VITASLOP_GXP_NO_STENCIL",
    "VITASLOP_GXP_ONLY",
    // Substitute a default-uniform register before a draw is submitted - the causality half of
    // `VITASLOP_GXP_INPUTS`. Reachable from the browser for the same reason as that one: the
    // white-out it is aimed at reproduces there and nowhere a file can be written.
    // For a title whose `sceGxmShaderPatcherCreateFragmentProgram` passes a NULL vertexProgram -
    // so the call names no shader PAIR and nothing can be prepared from it - offer the CROSS
    // PRODUCT of its created fragment and vertex programs and keep the ones that LINK.
    // The shader PAIRS a run linked, one line each. Browser-reachable because a pair that links
    // on the desktop and not on the device is exactly the failure this names.
    "VITASLOP_GXP_PACK_COMP0",
    "VITASLOP_GXP_PAIRS",
    "VITASLOP_GXP_PASS_SPLIT_EVERY",
    // Compile a title's shader pairs AHEAD of the draw that needs them. Browser-reachable
    // because an in-frame shader compile costs the most there - it is the hitch itself.
    "VITASLOP_GXP_PRECOMPILE",
    // Speculative work paid on a loading screen; how much of it is wasted is a per-title
    // measurement, which is what this exists to take. Reachable from the browser because that
    // is where an in-frame shader compile costs the most.
    "VITASLOP_GXP_PRECOMPILE_CROSS",
    // `0` = a one-register prefetch is always one full-precision component (the old reading).
    "VITASLOP_GXP_PREFETCH_U8",
    // >>> THE SHADER PROBES. The instruments that say WHICH term of a lit material is the
    // zero, and they were desktop-only: read through `std::env::var`, which the browser does
    // not have. That is backwards - the pictures under investigation are browser pictures.
    // Forwarded to the emitter's own arm table by `set_override`; see `link::set_arm`.
    "VITASLOP_GXP_PROBE",
    "VITASLOP_GXP_PROBE_SCALE",
    // `0`: an unpacked pair store narrows as two scalar round trips, not one vector one.
    "VITASLOP_GXP_Q2_VEC",
    // Every SUBMISSION of one pair that lands in a screen-space box, with its full vertex
    // record. The per-DRAW half `..._INPUTS_VERTS` cannot be: that dump dedupes by input
    // set, so a UI pair submitted a thousand times a frame almost never prints the element
    // under investigation. Browser-reachable for the same reason the other input dumps are.
    "VITASLOP_GXP_QUADS",
    // Replace every recompiled fragment's final return with this WGSL expression (gpu.rs).
    "VITASLOP_GXP_RETURN",
    // With VITASLOP_GXP_RETURN: only these pair keys (`!keys` = all but these) (gpu.rs).
    "VITASLOP_GXP_RETURN_KEYS",
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
    // The strict stop, so a browser run can be made to fail at the first pair the recompiler
    // cannot translate instead of dropping that pair's draws - see `gpu::report_fallback` for
    // the three choices and why dropping is the default.
    "VITASLOP_GXP_STRICT",
    // Add this epsilon to every 2-D textureSample coordinate (gpu.rs `nudge_texcoords`).
    "VITASLOP_GXP_TEXCOORD_NUDGE",
    // Restores `unclipped_depth` on every recompiled pipeline - the NEAR and FAR z clip planes
    // off together, which is what `depth-clip-control` actually buys. The default is OFF because
    // the near half of that bargain drew geometry BEHIND the eye at the plane, magnified, with
    // its depth clamped to nearest: a flat field decal straddling the camera plane covered 55%
    // of a world pass. Browser-reachable so the far-plane half can be re-measured where it is
    // seen.
    "VITASLOP_GXP_UNCLIPPED_DEPTH",
    "VITASLOP_GXP_VARYING_CODE_ORDER",
    // The arm back for binding the guest's own vertex row instead of repacking it to f32.
    // Browser-reachable because the browser is where its cost lives: the repack it removes was
    // most of a `prepare` stage that reached 82% of a 265 ms render frame there, so the A/B has
    // to be runnable on the engine that pays for it.
    "VITASLOP_GXP_VERTEX_PASSTHROUGH",
    // Return an interpolated VARYING as the colour - the half of the probe family a register
    // probe cannot reach, because a texture coordinate is consumed straight out of the varying
    // and never lands in a register.
    "VITASLOP_GXP_VPROBE",
    "VITASLOP_GXP_VP_TRACE",
    // Every linked GXP module's WGSL, at WARN, as it is built (gpu.rs).
    "VITASLOP_GXP_WGSL_LOG",
    // A directory of hand-edited `<key>.wgsl` modules that replace the generated ones.
    "VITASLOP_GXP_WGSL_OVERRIDE_DIR",
    // The WGSL override as a knob string, "<hex key>\n<wgsl>" (a browser has no override dir).
    "VITASLOP_GXP_WGSL_OVERRIDE_TEXT",
    "VITASLOP_GXP_YFLIP",
    "VITASLOP_GXP_ZFIX",
    // Rig: spin the browser run worker `<ms>` once after frame `<frame>` (`<frame>:<ms>`).
    "VITASLOP_HITCH_AT",
    // Report every write a HOST CALL makes to these guest addresses / values (host.rs).
    "VITASLOP_HOST_WRITE_WATCH",
    // The negative control for the instanced expansion memo - see
    // `vitaslop_runtime`'s `instanced_memo_enabled`.
    // `0` skips the in-place compare of a dirty snapshot range (its exactness control).
    "VITASLOP_INPLACE_COMPARE",
    "VITASLOP_INSTANCED_MEMO",
    // The storage model (vitaslop-runtime vita/iofilemgr.rs): bandwidth (`0` = model off),
    // per-request cost, park threshold. Their readers already go through `knobs::var`, but
    // they were missing here, so setting one in the browser ASSERTED in `set_override` and
    // the run died reserving its guest region ("unreachable", phone jobs 451/452/472/473).
    "VITASLOP_IO_BANDWIDTH_KIBPS",
    "VITASLOP_IO_PARK_THRESHOLD_US",
    "VITASLOP_IO_REQUEST_US",
    // `0`: the wall-clock floor no longer advances the storage clock (host.rs `wall_floor_tick`).
    "VITASLOP_IO_WALL_FLOOR",
    // `<start frame>:<duration ms>`: JS Self-Profiling sample of each SMP guest worker (smp-worker.js).
    "VITASLOP_JS_PROFILE",
    "VITASLOP_LINK_VAR_ADDEND",
    "VITASLOP_LOG",
    // The executable a title's `sceAppMgrLoadExec` replaced the process with - set BY THE
    // PAGE when it reboots the emulator for that exec (see vitaslop-web `mount_and_link`), the
    // browser twin of native's `RetailGuest::new_with_exec`. Not a tuning knob.
    "VITASLOP_MAIN_EXEC",
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
    "VITASLOP_MEM_DUMP",
    "VITASLOP_MEM_DUMP_AT",
    // Scan guest memory for byte patterns at a frame (lib.rs `mem_find_at`).
    "VITASLOP_MEM_FIND",
    "VITASLOP_MOVIE_AUDIO_BACKLOG",
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
    "VITASLOP_MP4_AUDIO_READ_AHEAD_MS",
    "VITASLOP_MP4_GUEST_IO_KB",
    "VITASLOP_MP4_GUEST_OPS",
    "VITASLOP_MP4_READ_AHEAD_MS",
    // One params state per voice: a params block and a lock/unlock write the same thing.
    "VITASLOP_NGS_BLOCK_STATE",
    "VITASLOP_NGS_VOICE_HANDLE_MEMO",
    "VITASLOP_NGS_ZERO_LEVEL",
    // The falsifier for the voice-handle LOOKUP: with it off, every query for a rack's
    // voice allocates a fresh handle again, which is what left 8,138 voices in the bank and
    // 318 of them playing every grain. It is here so a device can price the difference.
    // The CROSS-ENGINE host-call digest. Browser-reachable by construction: the whole point
    // is to line a browser run's frames up against a desktop run's and find the first one
    // that differs, and half of that comparison has no environment to read a knob from.
    "VITASLOP_NID_DIGEST",
    "VITASLOP_NO_BC",
    "VITASLOP_NO_FAST_IMPORT",
    "VITASLOP_NO_INLINE_CLIB",
    "VITASLOP_NO_INLINE_DELAY",
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
    // Route the HEAVYWEIGHT mutex lock/unlock pair through the host instead of emitting it
    // inline. Browser-reachable because that is where the pair costs what it costs: the pair
    // is two of the three crossings in each of a title's two movie-phase poll loops, and a
    // crossing there is twenty times a desktop one.
    "VITASLOP_NO_INLINE_MUTEX",
    // Routes every host call back through the SUSPENDING trap (`env.import`) instead of
    // sending the never-blocking NIDs through `env.import_fast`. Reachable from the browser
    // because the browser is the only engine where the two traps differ: the JSPI stack
    // switch per call is what the fast trap removes, and only a phone can price it. Read
    // at LINK time; set it before the run, not during it.
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
    // Canvas size in device pixels for a page that sends none (debug/runner); see present_scale.rs.
    "VITASLOP_OUTPUT_SIZE",
    // The repacked-geometry cache's bound in MEGABYTES, and `0` is the arm back to the entry
    // `0`: the live loop charges the wall floor's clock gain as game time (the old pacing).
    "VITASLOP_PACE_FLOOR_FREE",
    // `0`: the live loop charges short frames a whole period with no refund (the old pacing).
    "VITASLOP_PACE_REFUND",
    // cap alone. Browser-reachable because the browser is where the bound is load-bearing: an
    // entry cap of 4,096 meshes let this heap grow 40-80 MB per rendered frame on one title,
    // 252 MB to 3,751 MB in a hundred frames, and wasm32's ceiling is 4,096 MB.
    "VITASLOP_PACKED_CACHE_MB",
    "VITASLOP_PAUSE_ON_BLUR",
    "VITASLOP_PERF",
    // Whether the per-window performance report is ALSO written to the browser console. The
    // panel and the sink always get it; this is the console copy, off by default because the
    // page is the product and eight multi-line blocks a window is a firehose. Overridable
    // because a HARNESS reads the console and nothing else - a browser measurement with this
    // unset reports a frame rate and no breakdown, which is the sixth instance of this
    // omission and cost a run to find.
    "VITASLOP_PERF_CONSOLE",
    "VITASLOP_PIPELINE_DEFER",
    // How long declined presents may run before the present waits for its compiles.
    "VITASLOP_PIPELINE_DEFER_MAX_MS",
    // One status line per pipeline built, with its baked state - `gpu::log_pipe`.
    "VITASLOP_PIPE_LOG",
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
    // Per-present note for the first N frames: flipped buffers vs scene targets (lib.rs).
    "VITASLOP_PRESENT_LOG",
    "VITASLOP_PRESENT_PROBE",
    // With VITASLOP_PRESENT_PROBE: each sampled surface, half size, as a base64 console line (lib.rs).
    "VITASLOP_PRESENT_SHOT",
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
    "VITASLOP_RAW_WORD_VIEW",
    // Keep geometry that has not changed since the renderer first saw it RESIDENT on the GPU
    // instead of copying it into a per-frame arena and uploading it again. `0` sends every draw
    // back through the arenas, which is the A/B arm. Reachable from the browser because that is
    // where a per-frame upload costs the most.
    // A/B arm for making the region clip SCENE state: `0` restores the old behaviour, in which
    // a rectangle set for one render target stays in the GXM context and scissors the next
    // scene. Browser-reachable because the picture defect it fixes was reported from a phone.
    "VITASLOP_REGION_CLIP_SCENE",
    // `1`: the frame replay renders early-completed scenes too (frame_replay.rs).
    "VITASLOP_REPLAY_EARLY",
    // frame-replay (web): read back this offscreen target (hex guest address) instead of the display.
    "VITASLOP_REPLAY_TARGET",
    // `0`: a resident heap below its budget always DOUBLES when full instead of compacting a
    // mostly-dead heap at its current size.
    "VITASLOP_RESIDENT_COMPACT_EARLY",
    "VITASLOP_RESIDENT_GEOM",
    // The byte budget for each of the two resident geometry heaps, in MB (default 128). A heap
    // that fills at its budget COMPACTS (or, if its live set is most of it, RESETS and says so);
    // a reset every few frames means the working set does not fit and this is the number to change.
    "VITASLOP_RESIDENT_GEOM_MB",
    // Frames a resident geometry slice may go unbound before it (and the guest stream it pins on
    // the Rust heap) is forgotten (default 600); `0` = never, the old unbounded behaviour.
    "VITASLOP_RESIDENT_IDLE_FRAMES",
    // `<draws>` the resolver reads per hold of the snapshot lock (default 64); `0` = the whole job.
    "VITASLOP_RESOLVE_CHUNK",
    "VITASLOP_RTT_BG_CACHE",
    // The render-target CLEAR probe. Browser-reachable like the rest of the RTT diagnostics.
    "VITASLOP_RTT_CLEAR_PROBE",
    // Stamp the early-completion grid lines with the frame, at the first completion at or after each listed frame.
    "VITASLOP_RTT_GRID_FRAMES",
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
    // Per-frame trace of a written-back render target's probe texel and mean - `1` for every
    // target, or a `+`-separated hex address list. See `vitaslop_runtime::rtt_writeback`.
    // Name every written-back target containing one of these rrggbb colours (rtt_writeback.rs).
    "VITASLOP_RTT_PROBE_FIND",
    "VITASLOP_RTT_PROBE_LOG",
    // Hold each browser RTT writeback copy this long after capture - see `RttWriteback`.
    "VITASLOP_RTT_STALE_EXTENT",
    "VITASLOP_RTT_SUBRECT",
    "VITASLOP_RTT_WRITEBACK_DELAY_MS",
    "VITASLOP_RTT_WRITEBACK_FLOAT",
    // Decline a present while the oldest RTT writeback copy is older than this - see `LivePlayback::wb_max_age_ms`.
    "VITASLOP_RTT_WRITEBACK_MAX_AGE_MS",
    // Wait up to this long for render-target copies before the next guest frame (browser).
    "VITASLOP_RTT_WRITEBACK_SYNC_MS",
    "VITASLOP_SAMPLER_NARROW",
    "VITASLOP_SCHED_CORES",
    // Round-robin the scheduler's pick instead of the priority discipline, and TRACE what it
    // picked. Both browser-reachable because the scheduler behaves differently there (JSPI
    // suspends, no fuel) and a scheduling question asked on the desktop answers about the
    // desktop.
    "VITASLOP_SCHED_RR",
    "VITASLOP_SCHED_TRACE",
    // Guest frames to photograph (comma list): full-size presentshot lines; the runner's params.shots.
    "VITASLOP_SHOT_FRAMES",
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
    // `=1`: run the guest's threads IN PARALLEL on several browser workers (the Vita's three
    // cores) instead of one at a time on the run worker. OFF by default, and off is the
    // DETERMINISTIC engine every recipe, capsule and bisection assumes: a parallel run's
    // interleaving depends on host timing and does not replay. See `vitaslop_web::smp`.
    "VITASLOP_SMP",
    // `1` moves the flip's geometry resolve from the guest render thread to the run worker.
    "VITASLOP_SMP_ASYNC_RESOLVE",
    // `VITASLOP_SMP`: presents frame N while the guest workers build N+1 (the gate runs one
    // frame ahead). ON by default under SMP; `=0` stops every worker for each present instead.
    // `VITASLOP_SMP`: comma-separated NID-name prefixes to FORWARD to the run worker in addition
    // to `vita::smp_owner_only`'s own list - bisects a defect that appears when a family runs
    // on the guest workers.
    "VITASLOP_SMP_DEFER_TEXTURES",
    // `VITASLOP_SMP`: `1` pauses every guest for a small-target completion's whole render (default: only its write).
    "VITASLOP_SMP_EARLY_PAUSE_ALL",
    // `1`: a fiber's backing thread is placed on its runner's worker (default: like any new thread).
    "VITASLOP_SMP_FIBER_NEAR",
    "VITASLOP_SMP_FORWARD",
    // Spin x times each guest slice on the SMP workers: a phone-slow guest on the desktop.
    "VITASLOP_SMP_GUEST_SLOW",
    // `0`: a slow present holds the whole guest at the frame gate (default: it runs on in real
    // time and the next present takes the newest frame - `smp::late_present`).
    "VITASLOP_SMP_LATE_PRESENT",
    // `0`: a parallel link refuses the lightweight-mutex inline form (vita/mod.rs).
    "VITASLOP_SMP_LWMUTEX_INLINE",
    "VITASLOP_SMP_OVERLAP",
    // `VITASLOP_SMP`: thread placement. `apart` (default) = fewest live threads first, never
    // beside the main thread when the mask allows another worker; `spread` = fewest live
    // threads first; `load` = least recent work first (the old default).
    "VITASLOP_SMP_PLACE",
    // `0`: a real-time (wall-parked, e.g. audio output) thread stops at a shut frame gate too.
    "VITASLOP_SMP_RT_THROUGH_GATE",
    // Rig: leave GXM capture time out of the VITASLOP_SMP_GUEST_SLOW spin (see perf.rs).
    "VITASLOP_SMP_SLOW_EXCLUDE_GXM",
    // `VITASLOP_SMP`: ceiling in microseconds on the ADAPTIVE poll a guest worker makes before
    // sleeping, when at least half its recent mid-frame waits ended within it (default 400; 0 off).
    "VITASLOP_SMP_SPIN_CAP_US",
    // `VITASLOP_SMP`: microseconds a guest worker polls its doorbell before sleeping on it
    // (default 0) - trades CPU for the wake-up latency of a cross-worker handoff.
    "VITASLOP_SMP_SPIN_US",
    // `0`: a thread released by a timed wait expiring (or a sleep ending) competes for its
    // worker at once, instead of yielding to threads already runnable (the old rule). See the
    // note in `smp.rs` `drain`.
    "VITASLOP_SMP_TIMEOUT_YIELDS",
    // `VITASLOP_SMP`: `<from frame>:<frames>` records every resume span, wake, idle clock jump
    // and flip for that window and prints it as `smptrace` console lines - the frame timeline.
    "VITASLOP_SMP_TRACE",
    // `0`: a thread pinned to a core that already carries a better-priority thread keeps the
    // pin instead of being placed as if unpinned (smp.rs `place`).
    "VITASLOP_SMP_UNPIN_STARVED",
    // How many guest workers `VITASLOP_SMP` runs (default 3, one per Vita core).
    "VITASLOP_SMP_WORKERS",
    // Byte budget for retained texture snapshots. Reachable from the browser because
    // exceeding it there costs a full re-decode of the working set in one frame.
    "VITASLOP_SNAPSHOT_BUDGET_MB",
    // `0`: surplus free staging-belt chunks are never destroyed (the old unbounded free list).
    "VITASLOP_STAGING_TRIM",
    // `0`: sub-rectangle copy textures are dropped each frame instead of pooled.
    "VITASLOP_SUBRECT_POOL",
    // `0`: a sync point (sceGxmFinish) reads its deferred geometry inline under the host lock.
    "VITASLOP_SYNC_RESOLVE_ASYNC",
    // Path of the font that STANDS IN for the console's system font, which is not shipped.
    // `VITASLOP_SMP`: `0` serves a sync point's resolve-only park on the run worker, not the guest's own.
    "VITASLOP_SYNC_RESOLVE_ON_WORKER",
    // Listed here so the name is not a browser-boot panic, though the browser cannot open a
    // path: there it supplies the bytes directly instead (`font::system::set_bytes`).
    "VITASLOP_SYSTEM_FONT",
    // The texture-expansion scratch sized to the texture at hand when the kept largest-seen
    // buffers do not fit (texenc.rs `shrink_scratch_to_fit`); `0` = refuse to the CPU, as before.
    "VITASLOP_TEXENC_SHRINK_SCRATCH",
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
    // Milliseconds of one frame the INLINE block encode may spend. Default 2.0; `0` removes
    // the budget. A texture refused by it takes the ordinary RGBA8 path and is encoded on a
    // later frame. See `render::inline_encode_budget_ms`.
    "VITASLOP_TEX_ENCODE_BUDGET_MS",
    // The deferred ETC2 encode (gpu.rs `tex_encode_defer`) and its per-frame budget. Browser-
    // reachable because only a phone's GPU makes the inline encode a freeze.
    "VITASLOP_TEX_ENCODE_DEFER",
    "VITASLOP_TEX_ENCODE_DEFER_UNITS",
    // `0` restores the one-shot encode: an encode that starts runs to completion however long
    // it takes, and a texture too large for a frame is refused outright. The A/B arm for the
    // RESUMABLE encode - see `render::PartialEncode`.
    "VITASLOP_TEX_ENCODE_RESUME",
    "VITASLOP_TEX_MEMO_PER_SCENE",
    // Names the guest code that WROTE a uniform, by parameter name, with its `lr`. Needed in
    // the browser because `screenTintColour` - the white-out - is written there and never on
    // the desktop.
    "VITASLOP_TEX_PAGE_READ",
    // `0`: a texture unused THIS frame is a retention-eviction candidate again, however recently
    // it was used - see `gpu::tex_recent_frames`.
    "VITASLOP_TEX_RECENT_FRAMES",
    // How many MB of GPU texture the recompiler's VIEW cache may retain across frames, as
    // distinct from the budget above - which also gates the BC -> ETC2 re-encode, so it cannot
    // be lowered to bound memory without silently trading picture. See
    // `gpu::tex_retain_budget_bytes`. Browser-reachable because the device measured at 343 MB
    // retained against a 14 MB working set is a phone.
    "VITASLOP_TEX_RETAIN_MB",
    // The per-basic-block execution tracer's address ranges. Browser-reachable because the
    // trace is emitted at TRANSPILE time and the browser transpiles in a throwaway worker
    // with no environment: without this the one instrument that says "which path did this
    // function actually take" could not be pointed at the engine whose behaviour DIVERGES
    // from the desktop's, which is the only reason anyone asks.
    "VITASLOP_TRACE_BLOCKS",
    // Per-block guest-PC tracking, so a TRAP's register dump names the faulting guest
    // instruction instead of reporting `pc=0`. Browser-reachable for the same reason the
    // tracer above is - it is emitted at transpile time - and because a browser fault the
    // desktop does not reproduce is the case where a wasm stack alone leaves you
    // disassembling by hand.
    "VITASLOP_TRACK_PC",
    // `0`: bypass the browser's transpile cache (web/transpile-cache.js) - neither read nor write.
    "VITASLOP_TRANSPILE_CACHE",
    // Rig: hold a landed GPU timestamp readback until it has been in flight this many ms (a
    // phone's late map callbacks, on a desktop) - see the GPU budget in the web runner.
    "VITASLOP_TS_DELAY_MS",
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
    // Whether a content-index HIT on the single-stream snapshot path counts as a USE. `=0` is
    // the old behaviour, in which it did not and the index evicted what it was answering from.
    "VITASLOP_VERTEX_INTERN_USE",
    // `1`: a desktop guest movie decodes on the HARDWARE decoder (default software - video.rs
    // `movie_hardware`; the browser keeps WebCodecs' own choice).
    "VITASLOP_VIDEO_HARDWARE",
    // The format-keyed pipeline warm fallback (gpu.rs `warm_by_format`); `=0` is the arm back.
    "VITASLOP_WARM_BY_FORMAT",
    "VITASLOP_WASM_NAMES",
    "VITASLOP_WATCH_READ",
    "VITASLOP_WATCH_READ_SKIP",
    // The emit-time diagnostic family: the store and read watchpoints and the frame gate that
    // arms them. Browser-reachable for the reason the two above are - they are woven into the
    // guest's code at TRANSPILE time - and because "who wrote this word, and when" is the
    // question a browser-only divergence from a working desktop always comes down to.
    "VITASLOP_WATCH_STORE",
    "VITASLOP_WATCH_STORE_ARM",
    "VITASLOP_WATCH_STORE_LOG",
    "VITASLOP_WATCH_STORE_NZ",
    "VITASLOP_WATCH_STORE_SKIP",
    // The uniform-rewrite watch (SA buffers rewritten after their draw was recorded) - the
    // instrument that cleared "stale data" for a phone-only skinning defect; it reads through
    // `var`, so it must be listed or setting it in the browser panics the run on boot.
    "VITASLOP_WATCH_UNIFORM_REWRITES",
    // Turn a profile address into module+offset and a NID. A browser PROFILE is the thing that
    // produces those addresses, so this has to be settable there.
    "VITASLOP_WHICH_EXPORT",
    // Measure the window re-read census's slot-blank count exactly under the default rule.
    "VITASLOP_WINDOW_CENSUS",
    // The arm back to suspending on EVERY `sceKernelDelayThread(0)`, including the ones with
    // no other runnable thread to yield to. Browser-reachable because the browser is where
    // this is priced: a suspend there is a JSPI round trip, and a football title takes 2,377
    // of them a frame from one guest spin.
    // WHICH windows the guest's own GPU wait re-reads - see `vitaslop_runtime`'s
    // `window_wants_reread`. `blank` (the DEFAULT) is the standing whole-window rule, `slot`
    // also takes a window with an unwritten 16-byte row, `all` is the negative control.
    // `slot` is REFUTED as a fix - see the census in `window_reread_rule`.
    "VITASLOP_WINDOW_REREAD",
    // `1`: a draw's window capture waits for a busy snapshot lock instead of reading uncached.
    "VITASLOP_WINDOW_WAIT",
    // `0`: render-target write-backs go into guest memory after the present (old rule), not
    // at the next flip - see `writeback_at_flip` in vitaslop-web.
    "VITASLOP_WRITEBACK_AT_FLIP",
    "VITASLOP_YIELD_ELIDE",
];

/// The override map, consulted by every reader in this module BEFORE the environment.
fn overrides() -> &'static Mutex<HashMap<String, String>> {
    static CELL: OnceLock<Mutex<HashMap<String, String>>> = OnceLock::new();
    CELL.get_or_init(|| Mutex::new(HashMap::new()))
}

/// Read a knob: the process-wide override if one is set, else the environment.
/// Whether the guest's own vertex attribute layout is handed to the GXP link (so an attribute
/// can be integer-fetched or have its surplus lanes baked). OFF under any of the three arms that
/// convert vertex data on the CPU instead: `VITASLOP_GXP_GUEST_ATTRS=0`,
/// `VITASLOP_GXP_VERTEX_PASSTHROUGH=0`, and `VITASLOP_GXP_ATTR_FILL`. One definition, read by the
/// renderer's draw path AND the runtime's patcher-time prelink, so the two build the same
/// link options.
pub fn gxp_guest_attrs_in_link() -> bool {
    static ON: std::sync::OnceLock<bool> = std::sync::OnceLock::new();
    *ON.get_or_init(|| {
        var("VITASLOP_GXP_GUEST_ATTRS").ok().as_deref() != Some("0")
            && var("VITASLOP_GXP_VERTEX_PASSTHROUGH").map(|v| v.trim() != "0").unwrap_or(true)
            && var("VITASLOP_GXP_ATTR_FILL").is_err()
    })
}

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
/// The target `report_status!` emits on. Forced ON below whenever a filter was NAMED.
///
/// Status rides at `info` so it never claims to be a warning, but a channel the DEFAULT filter
/// switches off is a channel that does not exist - and that is precisely the trade that put this
/// material at `warn` in the first place. The web logger forces the same directive; this is the
/// native half, and without it every `VITASLOP_LOG=warn` desktop run (which is all of them,
/// including every headless oracle) silently loses the frame-shape trace, the precompile counts
/// and the pair-hash lines.
pub const STATUS_TARGET_DIRECTIVE: &str = "vitaslop::status=info";

pub fn log_filter() -> String {
    // >>> A RUN NOBODY CONFIGURED PRINTS NOTHING UNLESS SOMETHING IS WRONG.
    //
    // The status channel used to be appended to EVERY filter, the default included, so a
    // clean run of a working title wrote a dozen `INFO vitaslop::status` lines (the image
    // span, the mounted archive, the adapter's texture families, every NGS rack) before its
    // first frame. Those answer questions a developer asks; the product's answer to "how is
    // it going" is silence. So with no filter NAMED the directive is `warn` alone, and
    // status is forced on only when one was - `VITASLOP_LOG=warn` is a debugging session,
    // and every recorded repro command keeps exactly the output it had.
    //
    // Appended LAST so it wins: a user narrowing `VITASLOP_LOG` must not lose the status
    // channel by accident, and a user who deliberately names `vitaslop::status` still can
    // (their directive is more specific than the level default they are overriding).
    match log_filter_base() {
        Some(base) if base.contains("vitaslop::status") => base,
        Some(base) => format!("{base},{STATUS_TARGET_DIRECTIVE}"),
        None => "warn".to_string(),
    }
}

/// The filter the user NAMED, or `None` when neither knob is set.
fn log_filter_base() -> Option<String> {
    match var("VITASLOP_LOG") {
        Ok(v) if !v.is_empty() => Some(v),
        _ => match std::env::var("RUST_LOG") {
            Ok(v) if !v.is_empty() => Some(v),
            // WARN by default, not silence. Everything this engine approximates is required
            // to report itself - a shader pair that falls back to fixed-function, a dropped
            // draw, an unplaced scene - and those reports are `warn`. With an empty filter
            // they were all discarded, so the DEFAULT invocation was the one that could not
            // see them: a run that quietly drew a whole title through the fallback looked
            // exactly like a run that recompiled it. A knob nobody sets is not a report.
            _ => None,
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
    // and `set_arm` ignores every name it does not own.
    #[cfg(feature = "gpu")]
    vitaslop_gxp_shader::link::set_arm(name, value);
}

/// >>> WHICH SHADER EMISSION ARMS THIS RUN IS ON, for a diagnostic to name itself with.
///
/// Same bridge as [`set_override`] and for the same reason: the emitter's arm table lives in a
/// crate that depends on nothing, so the only way a browser diagnostic can report the arm it is
/// running is through here. Without the feature there is no emitter and no arm to report.
pub fn shader_arms_line() -> String {
    #[cfg(feature = "gpu")]
    {
        vitaslop_gxp_shader::link::arms_line()
    }
    #[cfg(not(feature = "gpu"))]
    {
        "no shader emitter in this build".to_string()
    }
}

/// >>> HOW MUCH MEMORY THE MACHINE THIS IS RUNNING ON ACTUALLY HAS, AND WHAT THE CACHES DO
/// >>> ABOUT IT.
///
/// # The problem this exists for, in the user's own words
/// "My whole phone feels sluggish", "my phone angry as a whole" - said across three separate
/// sessions about the target device, a phone, while every cache budget in this project was an
/// absolute constant fitted on a desktop:
/// ```text
///   texture snapshots (guest bytes)   192 MB
///   decode cache (RGBA8)              256 MB
///   vertex + index snapshots           64 MB each
///   GPU texture views                 477 MB   (the CONSOLE's own resident ceiling)
/// ```
/// Close to a gigabyte of caching before the guest's own memory, the wasm heap (measured at
/// 438 MB and never returned to the OS) or the render targets. MEASURED here on a 48,000-frame
/// session: a browser renderer process at **1.53 GB**. Nothing anywhere read a single property
/// of the device.
///
/// # Why it could not be fixed before, and can be now
/// Lowering these budgets used to make things WORSE, because every one of these caches dropped
/// EVERYTHING at its cap: a smaller budget bought more cliffs rather than less memory pressure.
/// They evict per entry now, oldest-first, so a smaller budget costs proportionally more
/// re-decode of the COLDEST entries instead of a full rebuild of the working set.
///
/// # The scale, and why a desktop cannot move
/// `navigator.deviceMemory` is capped at 8 by its own specification, so every desktop and most
/// tablets report exactly 8 and get a scale of 1.0 - the budgets they have today, unchanged, by
/// construction. A 4 GB phone gets 0.5 and a 2 GB one 0.25, which is the floor: below that the
/// caches stop being able to hold a frame's working set, and a cache that cannot hold one frame
/// re-decodes inside the frame, which is the failure the floors on the individual budgets exist
/// to prevent.
///
/// Unknown (nothing called the setter - the desktop and the headless oracle) is 1.0, so those
/// paths are exactly what they were.
static DEVICE_MEMORY_TENTHS: std::sync::atomic::AtomicU32 = std::sync::atomic::AtomicU32::new(0);

/// Record what the host says this device has, in gigabytes. Called once by the frontend at
/// startup, BEFORE the first frame - the budgets read the scale on every call rather than
/// caching it, so a late call is merely late rather than ignored.
pub fn set_device_memory_gb(gb: f64) {
    // NaN included: `gb <= 0.0` would let a NaN through, and a NaN budget is worse than none.
    if gb.is_nan() || gb <= 0.0 {
        return;
    }
    let tenths = (gb * 10.0).round().clamp(1.0, 2550.0) as u32;
    DEVICE_MEMORY_TENTHS.store(tenths, std::sync::atomic::Ordering::Relaxed);
}

/// What every memory budget in this project is multiplied by. `1.0` when the device is a
/// desktop, is unknown, or reports the specification's 8 GB cap.
pub fn memory_scale() -> f64 {
    let tenths = DEVICE_MEMORY_TENTHS.load(std::sync::atomic::Ordering::Relaxed);
    if tenths == 0 {
        return 1.0;
    }
    ((tenths as f64 / 10.0) / 8.0).clamp(0.25, 1.0)
}

/// The device memory that was reported, for the diagnostics panel. `None` when nothing set it.
pub fn device_memory_gb() -> Option<f64> {
    let tenths = DEVICE_MEMORY_TENTHS.load(std::sync::atomic::Ordering::Relaxed);
    (tenths != 0).then(|| tenths as f64 / 10.0)
}

/// Apply [`memory_scale`] to a byte budget. Kept as one function so every budget scales the same
/// way and a reader can find them all from here.
pub fn scale_budget(bytes: usize) -> usize {
    let scale = memory_scale();
    if scale >= 1.0 {
        return bytes;
    }
    ((bytes as f64) * scale) as usize
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

    /// >>> THE OVERRIDE TABLE IS PROCESS-GLOBAL AND `cargo test` IS PARALLEL.
    ///
    /// Two tests below set the SAME knob and then read it back, and nothing stopped one of them
    /// running between the other's write and its assertion. It failed exactly that way in a
    /// workspace run - `left: Some("false"), right: Some("1")` - and passed on its own and under
    /// `--test-threads=1`, which is the shape of a flake that gets re-run rather than fixed.
    /// Every test that WRITES an override takes this first.
    fn override_lock() -> std::sync::MutexGuard<'static, ()> {
        static LOCK: std::sync::Mutex<()> = std::sync::Mutex::new(());
        LOCK.lock().unwrap_or_else(|e| e.into_inner())
    }

    /// `NAME=0` must mean OFF.
    ///
    /// This is pinned because the opposite behaviour is invisible when it is wrong: a
    /// boolean knob that reads `0` as ON turns the OFF arm of an A/B into a second copy of
    /// the ON arm, and the run then reports "no difference" - which is exactly what a
    /// correct null result looks like. It cost a full measurement once already.
    #[test]
    fn a_boolean_knob_reads_zero_and_friends_as_off() {
        let _guard = override_lock();
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
        let _guard = override_lock();
        set_override("VITASLOP_GXP_LIVE", "1");
        assert_eq!(var("VITASLOP_GXP_LIVE").as_deref(), Ok("1"));
        assert_eq!(var_os("VITASLOP_GXP_LIVE").as_deref(), Some("1"));
        assert!(flag("VITASLOP_GXP_LIVE"));
    }

    /// The empty string is how a shell spells "set but valueless", and the house flag
    /// convention is presence, not truthiness - so it must read as ON.
    #[test]
    fn an_empty_override_still_reads_as_set() {
        let _guard = override_lock();
        set_override("VITASLOP_GXP_SOLID", "");
        assert!(flag("VITASLOP_GXP_SOLID"));
    }

    #[test]
    #[should_panic(expected = "not overridable")]
    fn setting_an_unrouted_knob_panics() {
        set_override("VITASLOP_NOT_ROUTED_ANYWHERE", "1");
    }
}
