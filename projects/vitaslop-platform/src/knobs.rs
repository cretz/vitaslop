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
    // The A/B arm for `emit_flags_add`'s carry and overflow forms. It is here because the
    // BROWSER is the engine that has to answer: `flags-add` was 39% of every operator the
    // transpiler emitted, the closed forms cut the module 5.3% and executed operators 8.7%,
    // and three interleaved desktop repeats put the wall-clock difference inside the noise.
    "VITASLOP_FLAGS_WIDE_C",
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
    // Speculative work paid on a loading screen; how much of it is wasted is a per-title
    // measurement, which is what this exists to take. Reachable from the browser because that
    // is where an in-frame shader compile costs the most.
    "VITASLOP_GXP_PRECOMPILE_CROSS",
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
    "VITASLOP_PERF",
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
    "VITASLOP_SCHED_CORES",
    // Fold the determinism signature on a browser recipe run that does NOT declare `@sig`.
    // The fold hashes every retired scene's vertices, indices and uniforms - about 3 MB a
    // frame on a race, MEASURED at 7.7% of the guest window - and the only consumer is an
    // `@sig` assertion, so a recipe without one pays for a number nothing compares. Set this
    // when the point of the run is to LEARN the signature and bless it into a recipe.
    "VITASLOP_SIGNATURE",
    // Byte budget for retained texture snapshots. Reachable from the browser because
    // exceeding it there costs a full re-decode of the working set in one frame.
    "VITASLOP_SNAPSHOT_BUDGET_MB",
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
    // Names the guest code that WROTE a uniform, by parameter name, with its `lr`. Needed in
    // the browser because `screenTintColour` - the white-out - is written there and never on
    // the desktop.
    "VITASLOP_UNIFORM_WATCH",
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
