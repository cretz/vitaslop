//! The renderer's once-per-subject reports: render-target aliasing and lifetime, the texture
//! working set, depth conversion and sampling, gamma surfaces, uniforms nothing feeds, and
//! the vertex plans.

use super::*;

/// Report, ONCE PER SHADER PAIR, that a draw is rendering the fixed-function
/// approximation instead of the guest's own translated shaders, and exactly why.
///
/// This is deliberately not gated behind a debug flag. The recompiler's contract is
/// no-guess: an unknown opcode hard-fails naming its raw word, an unfed shader input
/// hard-fails, a clip fixup that finds nothing to patch hard-fails. But the RENDERER
/// answers all of those by drawing the older capture-based approximation for that pair,
/// and a silent approximation is indistinguishable on screen from a faithful render -
/// which is how session 8's "36 draws render with their real guest shaders" turned out
/// to be 36 draws that linked and never ran. The reason line is the only signal that a
/// pair is still owed work, so it must be visible in an ordinary run.
///
/// The dedupe is global rather than per-renderer: a pair is submitted hundreds of times a
/// frame, and nothing is gained by repeating it for a second render target.
/// How far past a render target's base address a sampled texture may start and still be
/// that target. One 960-pixel row at 4 bytes is 3840, so this covers "the same buffer,
/// described from a slightly different origin" and cannot reach the next surface: this
/// title's own targets are 16 KiB apart.
pub(super) const RTT_ALIAS_SLACK: u32 = 4096;

/// The rendered target `addr` refers to, allowing for a small positive offset into it.
///
/// A title does not always sample a render target through a texture describing exactly that
/// target. This one aliases its 960x544 world buffer as a **1920x1088** texture starting 256
/// bytes in - the 2x supersampled view of the same memory - and reads it that way from every
/// post-process pass. Matching the data address EXACTLY misses that completely: the sample
/// falls through to the guest bytes, which the GPU never wrote, and the light, blur and
/// composite passes all render black. Matching by RANGE binds the render they meant.
///
/// Reports the first alias it resolves per (pair, unit): substituting a different buffer
/// than the one the guest named is exactly the kind of helpfulness that must not be silent.
pub(super) fn rendered_alias(
    rendered: &HashMap<u32, wgpu::TextureView>,
    addr: u32,
    key: u64,
    unit: u8,
    guest_bytes_unwritten: bool,
) -> Option<u32> {
    if addr == 0 {
        return None;
    }
    if rendered.contains_key(&addr) {
        return Some(addr);
    }
    let base = rendered
        .keys()
        .copied()
        .filter(|&b| addr > b && addr - b <= RTT_ALIAS_SLACK)
        .max_by_key(|&b| b)?;
    // >>> AN OFFSET INTO A TARGET IS ONLY THAT TARGET WHILE THE GUEST HAS NOT WRITTEN THERE.
    //
    // A live target's pixels are on the GPU, so the guest's bytes at a sub-rectangle of it
    // read EMPTY - the witness `rtt_alias_block` already uses for the exact-address case
    // [[vitaslop-a-render-target-reads-empty-in-guest-memory]]. That case only ever looks at
    // textures bound at a target's OWN address, so a recycled allocation that starts a few
    // bytes INTO a freed target came through here unchecked and was handed the old target.
    // MEASURED (a football title, desktop browser, 2026-09-25): the scoreboard's publisher badge bound
    // `rendered@0x9257ab00` - a 32x64 target last drawn in the front end - and drew as a flat
    // RED SQUARE; native bound the guest's decode only because a slower run had already aged
    // the target out after its one-minute TTL. Real guest bytes: sample those.
    if !guest_bytes_unwritten {
        return None;
    }
    use std::sync::{Mutex, OnceLock};
    static SEEN: OnceLock<Mutex<HashSet<(u64, u8)>>> = OnceLock::new();
    let seen = SEEN.get_or_init(|| Mutex::new(HashSet::default()));
    let mut seen = seen.lock().unwrap_or_else(|e| e.into_inner());
    if seen.insert((key, unit)) {
        report!(
            "gxp pair {key:016x}: sampler unit {unit} names {addr:#x}, which is {} bytes into \
             the render target at {base:#x} - binding that target's render (the guest is \
             describing the same buffer from a different origin)",
            addr - base
        );
    }
    Some(base)
}

/// A pipeline linked with a RAW unit found no raw-rendered target at that unit's address on
/// this draw - the draw is dropped rather than bound short. Once per pair and unit: the
/// pipeline key folds the raw mask, so a draw whose texture changed builds another pipeline
/// and this is a genuine disagreement between the plan and the frame.
pub(super) fn report_raw_slot_unbound(key: u64, unit: u8) {
    use std::sync::{Mutex, OnceLock};
    static SEEN: OnceLock<Mutex<HashSet<(u64, u8)>>> = OnceLock::new();
    let seen = SEEN.get_or_init(|| Mutex::new(HashSet::default()));
    let mut seen = seen.lock().unwrap_or_else(|e| e.into_inner());
    if seen.insert((key, unit)) {
        report_warn!(
            "gxp pair {key:016x}: sampler unit {unit} was linked as a RAW 64-bit read but no \
             raw-rendered target is bound there on this draw - the draw is DROPPED"
        );
    }
}

/// A fixed-function draw landed in a RAW 64-bit pass, which has no fixed-function
/// pipelines; it is dropped. Once per target address.
pub(super) fn report_raw64_fixed_dropped(addr: u32) {
    use std::sync::{Mutex, OnceLock};
    static SEEN: OnceLock<Mutex<HashSet<u32>>> = OnceLock::new();
    let seen = SEEN.get_or_init(|| Mutex::new(HashSet::default()));
    let mut seen = seen.lock().unwrap_or_else(|e| e.into_inner());
    if seen.insert(addr) {
        report_warn!(
            "gxm rtt {addr:#x}: a fixed-function draw (no recompiled pair) into this 64-bit \
             colour surface is DROPPED - the raw Rg32Uint pass serves recompiled draws only"
        );
    }
}

/// A draw into a RAW 64-bit target carried a guest blend state, which an integer
/// attachment cannot apply: the words are stored unblended. Once per pair.
pub(super) fn report_raw64_blend_dropped(key: u64) {
    use std::sync::{Mutex, OnceLock};
    static SEEN: OnceLock<Mutex<HashSet<u64>>> = OnceLock::new();
    let seen = SEEN.get_or_init(|| Mutex::new(HashSet::default()));
    let mut seen = seen.lock().unwrap_or_else(|e| e.into_inner());
    if seen.insert(key) {
        report_warn!(
            "gxp pair {key:016x}: draws into a 64-bit colour surface with a blend enabled, \
             and the raw Rg32Uint attachment cannot blend - its words are stored unblended"
        );
    }
}

/// The faces of a cube map. Named because `6` appears in three places that must agree -
/// the layer count of the assembled texture, the progression test that finds the faces,
/// and `GxmTexture::faces`, which is how the guest's own texture declares itself a cube.
pub(super) const CUBE_FACES: u32 = 6;

/// Say - once per cube - that a cube map is being served from the six targets its faces
/// were RENDERED into rather than from guest memory, and on what evidence.
///
/// Worth a line because it is the moment a whole class of draw stops sampling stale bytes,
/// and because the spacing is the load-bearing derivation: printing it is what would let a
/// wrong one be spotted rather than admired as a working reflection.
pub(super) fn report_rendered_cube(base: u32, size: u32, stride: u32, format: wgpu::TextureFormat) {
    report_warn!(
        "gxm cube: {base:#x} is a cube map the guest RENDERS - its six faces were drawn into \
         six {size}x{size} targets spaced {stride:#x} bytes apart ({format:?}), so they are \
         assembled into one cube texture and sampled from THAT. Before this they fell \
         through to an upload of guest memory, which the GPU wrote and the guest did not, \
         so the reflection sampled stale or empty bytes."
    );
}

/// Say - once per cube - that a cube map is served from ONE target its six faces were
/// rendered into, stacked. See `GxmRenderer::assemble_stacked_cube`.
pub(super) fn report_stacked_cube(base: u32, size: u32, height: u32, format: wgpu::TextureFormat) {
    report_warn!(
        "gxm cube: {base:#x} is a cube map the guest RENDERS as one {size}x{height} target \
         with its six {size}x{size} faces stacked in its first {} rows ({format:?}), so the \
         faces are copied into one cube texture and sampled from THAT instead of from guest \
         memory the GPU wrote and the guest did not",
        size * CUBE_FACES
    );
}

/// Report - once, with the high-water mark - that ONE FRAME's texture working set does not
/// fit the cache budget, so the budget had to be exceeded to finish the frame.
///
/// The eviction policy will not throw away a texture the frame in flight has already used
/// (see [`GxpLive::views_used`]), so this is the case where there was nothing else to give.
/// It needs saying because the consequence is not a slow frame - it is unbounded GPU memory
/// on a device that kills a worker for it without an error, a crash event or a log line.
/// `VITASLOP_TEX_CACHE_MB` raises the budget; the real fix is a smaller working set.
pub(super) fn report_texture_budget_exceeded(
    bytes: usize,
    views_used: &HashMap<(u64, SamplerDim), (u64, usize, u32, Residency)>,
    epoch: u64,
) {
    use std::sync::atomic::{AtomicUsize, Ordering};
    static WORST: AtomicUsize = AtomicUsize::new(0);
    // >>> A NEW HIGH-WATER MARK IS NOT A NEW FINDING, and reporting every one of them
    // turned this into the second-largest source of noise in a device capture: the working
    // set climbs a megabyte at a time (256, 260, 261, 263, 264 ...) and each step was a
    // fresh `warn`. MEASURED: 1,846 lines of it in one run, against a panel that keeps 96.
    // A step is required, so the line fires when the picture MATERIALLY changes.
    const STEP_MB: usize = 32;
    let mb = bytes / (1024 * 1024);
    let prev = WORST.load(Ordering::Relaxed);
    if mb < prev + if prev == 0 { 1 } else { STEP_MB } {
        return;
    }
    WORST.store(mb, Ordering::Relaxed);
    // >>> AND IT NAMES ITS OWN CAUSE. "Reduce the working set" is not actionable without
    // knowing what the working set IS: a Vita cannot reference 264 MB of texture, so the
    // interesting question is which formats INFLATE on the way in - a PVRTC surface has no
    // WebGPU format and is decoded to RGBA8, which is an 8x expansion, while a BC one is
    // uploaded compressed and expands not at all. Without this breakdown the next step is a
    // guess; with it the top row IS the next piece of work.
    report_warn!(
        "gxm textures: one frame's working set is {mb} MB against a \
         {} MB budget, and every entry is in use by this frame - the budget is being \
         EXCEEDED to finish it. Raise VITASLOP_TEX_CACHE_MB or reduce the working set. \
         Biggest contributors this frame: {}",
        tex_cache_budget_bytes() / (1024 * 1024),
        texture_composition(views_used, epoch).join(", ")
    );
}

/// How a texture is actually resident on the GPU, for the working-set breakdown.
#[derive(Clone, Copy, PartialEq, Eq, Hash, Debug)]
pub(super) enum Residency {
    /// CPU-decoded to RGBA8 (or RGBA16F) - the expansion the budget is about.
    Decoded,
    /// The guest's own blocks, handed over unchanged.
    Passthrough,
    /// Blocks we re-encoded from decoded texels.
    Transcoded,
}

/// The frame's texture working set BY FORMAT, biggest first - the breakdown both texture
/// reports print, written once so they cannot describe the same frame differently.
pub(super) fn texture_composition(
    views_used: &HashMap<(u64, SamplerDim), (u64, usize, u32, Residency)>,
    epoch: u64,
) -> Vec<String> {
    let mut by_format: HashMap<(u32, Residency), (usize, usize)> = HashMap::default();
    for (used, b, fmt, resident) in views_used.values() {
        if *used == epoch {
            let e = by_format.entry((*fmt, *resident)).or_insert((0, 0));
            e.0 += *b;
            e.1 += 1;
        }
    }
    let mut rows: Vec<((u32, Residency), usize, usize)> =
        by_format.into_iter().map(|(f, (b, n))| (f, b, n)).collect();
    rows.sort_by_key(|r| std::cmp::Reverse(r.1));
    rows.iter()
        .take(8)
        .map(|((f, resident), b, n)| {
            format!("{:#04x} {} x{n} ({:.1} MB)", f, format_note(*f, *resident), *b as f64 / 1048576.0)
        })
        .collect()
}

/// Report one frame's texture working set and its composition at each new high-water mark,
/// WHETHER OR NOT it fits the budget.
///
/// # A report that only fires on failure cannot measure a fix
/// [`report_texture_budget_exceeded`] carries the same breakdown and used to be the only
/// thing that ever printed it - so the moment a change brought the working set under budget,
/// the number that would have shown by how much disappeared with it. Measuring the decode
/// path against the compressed upload gave 274 MB for the arm that FAILED and silence for
/// the two that worked, which is the one shape of result an A/B must not have.
///
/// # NOT behind a knob, because a knob nobody sets is the same as no report
/// This is the number the whole texture-memory story is told in, and gating it would mean the
/// engine that needs it most - a phone, which has no environment to set a knob in - never
/// emits it at all. The volume is bounded instead: a 32 MB step over a 274 MB run is nine
/// lines, against a device panel that keeps 96. The budget report's own history is why the
/// step matters - it fired on every 1 MB change and produced 1,846 lines in one run.
pub(super) fn report_texture_working_set(
    bytes: usize,
    views_used: &HashMap<(u64, SamplerDim), (u64, usize, u32, Residency)>,
    epoch: u64,
) {
    use std::sync::atomic::{AtomicUsize, Ordering};
    static WORST: AtomicUsize = AtomicUsize::new(0);
    const STEP_MB: usize = 32;
    // Publish the frame's working set whether or not it is a new high, so the CPU side can
    // ask whether compressing anything further is still buying something. See
    // [`texture_budget_pressure`].
    //
    // >>> AT LEAST ONE BYTE: ZERO IS RESERVED FOR "NO FRAME HAS FINISHED YET", WHICH COUNTS
    // >>> AS PRESSURE. A frame that bound no textures - a loading screen, a black fade -
    // published a real 0 here, so the NEXT screen's first frames read as "unknown" and
    // re-encoded every BC texture they bound to ETC2 on a phone whose working set was 22 MB
    // of 477. MEASURED (a fighting title, runner job 085): the two worst display gaps of the run, 1.46 s
    // and 1.40 s, were 46 and 56 textures transcoded at a screen's first frame, almost all
    // `Bc -> Etc2Rgba8` - a second lossy step the budget never asked for, at ~17 ms of GPU
    // each on that device.
    crate::gpu::LAST_WORKING_SET.store(bytes.max(1), Ordering::Relaxed);
    let mb = bytes / (1024 * 1024);
    let prev = WORST.load(Ordering::Relaxed);
    if mb < prev + if prev == 0 { 1 } else { STEP_MB } {
        return;
    }
    WORST.store(mb, Ordering::Relaxed);
    // >>> NAME THE ADAPTER WHEN THE ADAPTER IS THE REASON.
    //
    // Without this the composition reads "x94 PVRTC, ~8x EXPANDED" on a device that CANNOT
    // do anything else, and every reader has to work out from the ABSENCE of refusal reports
    // that the format gate shut before any of them could fire. Absence of evidence is the
    // worst possible way to publish the single fact that decides whether this whole path is
    // available - MEASURED on the target phone, whose img-tec D-series adapter offers ASTC
    // and ETC2 but not BC, so the passthrough and the transcode are both unreachable there.
    let why = if crate::gpu::block_compression_available() {
        ""
    } else {
        " NOTE: this adapter has NO BC support, so every compressed guest format is decoded \
          to RGBA8 and no passthrough or transcode is possible here - the fix for this device \
          is an ETC2/ASTC encoder, not a refusal to chase."
    };
    report_status!(
        "gxm textures: one frame's texture working set is {mb} MB (new high). \
         Composition: {}.{why}",
        texture_composition(views_used, epoch).join(", ")
    );
}

/// Whether a guest base format costs GPU memory beyond its compressed size, for the
/// working-set breakdown. The distinction is the whole point of that report: a format with
/// no WebGPU equivalent is CPU-decoded to RGBA8 and lands on the GPU expanded, and that
/// expansion - not the number of textures - is what puts a phone over budget.
/// # This used to state an aspiration and read as a measurement
/// `0x85..=0x88` was annotated "BC, uploaded compressed" while nothing in the project
/// uploaded anything compressed - both devices asked for `Features::empty()` and
/// `upload_gxp_texture` only ever created `Rgba8Unorm`. So the one report whose whole job is
/// to say WHERE the megabytes are told the reader that a third of them were already as small
/// as they could be. The flag now comes from the upload decision itself.
pub(super) fn format_note(base_format: u32, resident: Residency) -> &'static str {
    match resident {
        // The difference between a lossless handover and a second lossy pass, in the one
        // report anyone reads to find out where the megabytes went.
        Residency::Passthrough => return "uploaded COMPRESSED (guest blocks, guest mips)",
        // >>> NAME THE FAMILY IT WAS ENCODED TO, not the one this line was written for.
        // This said "RE-ENCODED to BC" unconditionally, and a device capture caught it
        // saying so two lines under its own `RE-ENCODED to Etc2Rgb8` warning. The working-set
        // breakdown is the one report anyone reads to find out where the megabytes went, and
        // on the device that needs ETC2 it was naming the format that device cannot take.
        Residency::Transcoded => {
            return match crate::gpu::block_family() {
                BlockFamily::Etc2 => "RE-ENCODED to ETC2 (lossy)",
                _ => "RE-ENCODED to BC (lossy)",
            }
        }
        Residency::Decoded => {}
    }
    match base_format {
        0x80..=0x84 => "PVRTC/PVRTC2 -> RGBA8, ~8x EXPANDED",
        0x85..=0x8b => "BC -> RGBA8, ~4-8x EXPANDED (no passthrough - see the report above)",
        _ => "raw",
    }
}

/// Report - once per (pair, unit) - that a recompiled draw is sampling a texture with a NULL
/// data pointer, i.e. the 1x1 zero texel the runtime substitutes for an uninitialised
/// `SceGxmTexture` handle.
///
/// This is not a fallback: the draw runs, with the guest's own shader, and produces a
/// perfectly valid picture of nothing. That is exactly why it needs saying out loud - a
/// render target that comes out uniformly black looks like a geometry, depth or blend
/// problem, and this is none of those.
pub(super) fn report_zero_texel_sample(key: u64, unit: u8) {
    use std::sync::{Mutex, OnceLock};
    static SEEN: OnceLock<Mutex<HashSet<(u64, u8)>>> = OnceLock::new();
    let seen = SEEN.get_or_init(|| Mutex::new(HashSet::default()));
    let mut seen = seen.lock().unwrap_or_else(|e| e.into_inner());
    if seen.insert((key, unit)) {
        report!(
            "gxp pair {key:016x}: sampler unit {unit} is bound to a 1x1 ZERO texel (the guest \
             handle had null control words) - everything this draw derives from that sample \
             is zero"
        );
    }
}

/// The guest's own depth mapping `(zScale, zOffset)` for a draw, from the viewport it was
/// submitted under: the hardware's window depth is `z/w * zScale + zOffset`.
///
/// `(1, 0)` - the identity, which is what this renderer applied for every draw of every
/// title before there was a reading - when the guest never called `sceGxmSetViewport` for
/// this pass (the all-zero sentinel), or when it left a degenerate `zScale` of 0 that would
/// collapse the whole scene onto one plane. Both cases are REPORTED once per pair: an
/// assumed depth convention is exactly the kind of thing that looks right until a title
/// whose faces sort by nine ten-thousandths of the buffer draws itself inside out.
pub(super) fn gxm_viewport_depth(vp: &[f32; 6], key: u64) -> (f32, f32) {
    let (zo, zs) = (vp[4], vp[5]);
    if vp.iter().all(|v| *v == 0.0) {
        report_viewport_depth_assumed(key, "the guest set no viewport for this draw");
        return (1.0, 0.0);
    }
    if !zs.is_finite() || !zo.is_finite() || zs == 0.0 {
        report_viewport_depth_assumed(key, "its zScale is zero or non-finite");
        return (1.0, 0.0);
    }
    (zs, zo)
}

/// Once per pair: this draw has the viewport transform DISABLED, so its position is taken
/// as window pixels of the target (see `inject_clip_fixup`). Said because the reading only
/// exists since the constant was corrected to the SDK's `0x10000`, so which titles take it
/// is worth seeing in a run's own log.
pub(super) fn report_window_space_pair(key: u64, surf: (u32, u32)) {
    use std::collections::HashSet;
    use std::sync::Mutex;
    static SEEN: Mutex<Option<HashSet<u64>>> = Mutex::new(None);
    let mut g = SEEN.lock().unwrap_or_else(|e| e.into_inner());
    if !g.get_or_insert_with(HashSet::new).insert(key) {
        return;
    }
    report_status!(
        "gxp pair {key:x}: drawn with the viewport DISABLED - its clip position is read as              WINDOW pixels of the {}x{} target",
        surf.0,
        surf.1
    );
}

/// One line per pair whose depth mapping had to be assumed rather than read. See
/// [`gxm_viewport_depth`].
pub(super) fn report_viewport_depth_assumed(key: u64, why: &str) {
    use std::collections::HashSet;
    use std::sync::Mutex;
    static SEEN: Mutex<Option<HashSet<u64>>> = Mutex::new(None);
    let mut g = SEEN.lock().unwrap_or_else(|e| e.into_inner());
    if !g.get_or_insert_with(HashSet::new).insert(key) {
        return;
    }
    report!(
        "gxp pair {key:x}: depth mapped with the IDENTITY (zScale 1, zOffset 0) because \
         {why} - the guest's own zScale/zOffset is what decides whether a LESS_EQUAL test \
         can separate two faces at all"
    );
}

/// This renderer's view of [`super::viewport_rect`] - the ONE statement of GXM's
/// offset/half-scale convention, shared with the software rasteriser so the two backends
/// cannot drift.
///
/// The DEPTH half (`zOffset`, `zScale`) is deliberately not applied here. The recompiled
/// vertex stage already writes a depth in whichever convention `VITASLOP_GXP_ZFIX` selects,
/// so handing the guest's z mapping to `min_depth`/`max_depth` as well would apply it twice.
pub(super) fn gxm_viewport_rect(
    enable: u32,
    vp: &[f32; 6],
    w: u32,
    h: u32,
) -> Option<(f32, f32, f32, f32)> {
    crate::gpu::viewport_rect(enable, vp, w, h)
}

/// Report - once per (surface, new mode) - that the guest CHANGED a live target's
/// gamma-correct mode.
///
/// Worth a line of its own because it is the event that used to desynchronise the
/// renderer from itself: the recorded mode was written at creation and never updated, so
/// after a change the pass encoded through one view while every sampler decoded through
/// the other, and on a feedback path the mismatch compounded to white or to black. It is
/// handled now, and this says when a title exercises it - a fact no other line carries.
/// Why a render target was (re)created, deduped by ADDRESS AND REASON so a target that is
/// rebuilt every frame says so once rather than three thousand times.
///
/// See `ensure_rtt` for what a recreation costs: it bumps `rtt_epoch`, which invalidates
/// every cached sampler bind group naming ANY target, and it discards the target's own
/// contents. `rtt_created` alone cannot name the cause because the reasons have nothing in
/// common - a DEPTH companion is a legitimate one-off, and a SAMPLE-COUNT change is this
/// renderer's own per-frame `sample_depth` answer flipping.
///
/// >>> "SIZE" WAS THE THIRD REASON AND IT IS GONE, because it was never a reason - it was
/// >>> two render targets sharing one colour address being mistaken for one that resized.
///
/// `rtt` is keyed by `(address, width, height)` now, so a different extent at a known
/// address is a different target and a first sighting. MEASURED on a baseball title: this line
/// fired every frame on the three stadium-board addresses 0x8e55a090, 0x8e79c2e0 and
/// 0x8ec4a8e0, where a 4-draw 512x1024 pass and a 41-draw 256x256 pass share a colour
/// surface - and each recreation threw the other pass's image away. Those boards rendered
/// BLACK. See `rtt`.
/// Once per (target, sample count): a target kept its picture across a change of sample
/// count - see `ensure_rtt`. Said because a target that alternates every frame allocates
/// a multisampled pair every frame, which is worth seeing in a run's own log.
pub(super) fn report_rtt_samples_swapped(addr: u32, samples: u32) {
    use std::collections::HashSet;
    use std::sync::{Mutex, OnceLock};
    static SEEN: OnceLock<Mutex<HashSet<(u32, u32)>>> = OnceLock::new();
    let seen = SEEN.get_or_init(|| Mutex::new(HashSet::new()));
    if !seen.lock().unwrap_or_else(|e| e.into_inner()).insert((addr, samples)) {
        return;
    }
    report_status!(
        "gxm rtt: render target {addr:#010x} now rasterises at {samples} sample(s) - its              resolved image is KEPT, only the multisampled attachments were replaced"
    );
}

pub(super) fn report_rtt_recreated(addr: u32, why: &'static str) {
    if why == "first sighting" {
        return;
    }
    use std::collections::HashSet;
    use std::sync::{Mutex, OnceLock};
    static SEEN: OnceLock<Mutex<HashSet<(u32, &'static str)>>> = OnceLock::new();
    let seen = SEEN.get_or_init(|| Mutex::new(HashSet::new()));
    if !seen.lock().unwrap_or_else(|e| e.into_inner()).insert((addr, why)) {
        return;
    }
    report_warn!(
        "gxm rtt: render target {addr:#010x} was RE-CREATED because its {why} changed. Every recreation bumps `rtt_epoch`, which invalidates every cached sampler bind group naming any target - so a target that does this every frame makes every one of those groups a first sighting, every frame. It also DISCARDS the target's contents. Reported once per target per reason."
    );
}

/// A frame whose scenes straddle a display FLIP, and which passes were rescued by it.
///
/// Reported once. It is a property of how a title's scene list lines up with its flips
/// rather than an event, so repeating it would bury everything else in the log - and the
/// interesting question is whether it happens at all, which one line answers.
pub(super) fn report_frame_straddles_a_flip(extra: &HashSet<u32>, presents: usize) {
    use std::sync::atomic::{AtomicBool, Ordering};
    static SAID: AtomicBool = AtomicBool::new(false);
    if SAID.swap(true, Ordering::Relaxed) {
        return;
    }
    let list: Vec<String> = extra.iter().map(|a| format!("{a:#010x}")).collect();
    report_warn!(
        "gxm: this frame's scenes STRADDLE A DISPLAY FLIP - the guest presented {presents} different buffers while it was being captured, and {} of its passes draw into one that is not the last scene's target ([{}]). Those passes used to be classified as OFFSCREEN and dropped, which shows as a HUD over a black world; they are now composited into the same display image, in scene order. Reported once.",
        extra.len(),
        list.join(", ")
    );
}

/// A texture-expansion batch that survived to the next frame - see `texenc::RawBatch`.
/// Reported once: it is a defect in this file's own flush discipline, not a title's doing.
pub(super) fn report_raw_batch_not_flushed(pending: u32) {
    use std::sync::atomic::{AtomicBool, Ordering};
    static SAID: AtomicBool = AtomicBool::new(false);
    if SAID.swap(true, Ordering::Relaxed) {
        return;
    }
    report_warn!(
        "gxm: {pending} texture expansions were still UNSUBMITTED at the start of a frame, so the frame before this one drew with textures nothing had written yet. They are submitted now, but the flush belongs at the end of `encode_chain` and something returned from it without reaching that line. Reported once."
    );
}

/// Render targets released as abandoned - see [`GxmRenderer::reclaim_stale_rtt`].
///
/// Reported the FIRST time only. That it happens at all is the finding (a title whose target
/// count grows with run length), and after that the standing count is in the caches line,
/// where it can be read against the occupancy rather than scrolled through.
pub(super) fn report_rtt_reclaimed(n: usize, bytes: u64, left: usize, which: &[(u32, u32, u32)]) {
    use std::sync::atomic::{AtomicBool, Ordering};
    static SAID: AtomicBool = AtomicBool::new(false);
    if SAID.swap(true, Ordering::Relaxed) {
        return;
    }
    // >>> AND WHICH ONES. A count says memory was returned; it cannot say whether the thing
    // returned was a target something still SAMPLES, and those are different events with
    // the same number. A title whose crowd atlas is painted once during loading and read
    // for the rest of the run has exactly one symptom if this releases it - an empty
    // picture where that atlas was - and no line in the log to connect the two.
    // [[vitaslop-a-drop-count-needs-its-draw-count]]
    let named: Vec<String> =
        which.iter().take(24).map(|(a, w, h)| format!("{a:#010x}({w}x{h})")).collect();
    report_warn!(
        "gxm: released {n} render targets ({} MB) that had gone a full minute without being rendered into or sampled, leaving {left}: {}{}. This map is keyed by guest address and had no removal path at all, so a title that moves between screens accumulated their colour and depth attachments for the whole run - MEASURED at 304 targets on a 48,000-frame session. Reported once; the running total is in the caches line.",
        bytes / (1024 * 1024),
        named.join(" "),
        if which.len() > named.len() { " ..." } else { "" }
    );
}

/// >>> A DRAW BOUND A TEXTURE AT AN ADDRESS THIS RENDERER STILL HOLDS A RENDER TARGET FOR,
/// >>> AND THE TWO ARE NOT THE SAME SIZE.
///
/// The guest reuses memory. A render target it has finished with can be freed and an
/// ordinary texture allocated over it, and `rtt` - offered to the
/// sampler path through `sample_views` - will then hand the draw the OLD TARGET's pixels
/// instead of the texture the guest actually put there. The picture that produces is a
/// surface wearing something else's image, which is what "a garbage texture inside the
/// banner" looks like from the outside.
///
/// A disagreeing EXTENT is the cheap evidence for it: a pass sampling a real target binds it
/// at the size it was rendered. **This changes NOTHING yet** - not what is bound, and not
/// whether the address counts as a use. It exists to find out whether the case occurs at all,
/// because both repairs it would justify (refusing the alias, or letting the target be
/// reclaimed) risk taking a target away from a title that is legitimately sampling it through
/// a descriptor of another size.
///
/// Deduped per address per shape pair: a title that does this does it every frame.
///
/// >>> `target` IS ONE OF POSSIBLY SEVERAL EXTENTS HELD AT THAT ADDRESS - the smallest.
/// `rtt` is keyed by `(address, width, height)` (see its doc comment), so an address can
/// hold two live targets, and the caller fires this only when the bound extent matches
/// NONE of them. For the single-target case - every title measured when this report was
/// written - it prints exactly what it always did.
pub(super) fn report_rtt_extent_mismatch(addr: u32, target: (u32, u32), bound: (u32, u32), guest_empty: bool) {
    use std::sync::{Mutex, OnceLock};
    static SEEN: OnceLock<Mutex<HashSet<(u32, u32, u32)>>> = OnceLock::new();
    let seen = SEEN.get_or_init(|| Mutex::new(HashSet::default()));
    if !seen.lock().unwrap_or_else(|e| e.into_inner()).insert((addr, bound.0, bound.1)) {
        return;
    }
    let text = format!(
        "gxm rtt: {addr:#010x} is held as a {}x{} RENDER TARGET, but a draw binds a texture of {}x{} at that address, and the guest's own bytes there are {}. {} Reported once per address per bound shape.",
        target.0,
        target.1,
        bound.0,
        bound.1,
        if guest_empty { "ALL ZERO" } else { "NOT all zero" },
        if guest_empty {
            "That is the signature of a LIVE target sampled through a differently-sized descriptor (its pixels are on the GPU, so guest memory reads empty), so the alias STANDS and the target counts as used."
        } else {
            "So the guest has really allocated something over the freed target: the alias is REFUSED - this draw samples the guest's own texture - and the stale target stops counting as used, so it is reclaimed within its TTL."
        }
    );
    // A standing alias is the RESOLVED case - the evidence said live target, the target
    // was bound - so it is status. A refused one changes what the draw samples and stays
    // a warning.
    if guest_empty {
        report_status!("{text}");
    } else {
        report_warn!("{text}");
    }
}

pub(super) fn report_gamma_mode_changed(addr: u32, gamma: bool) {
    use std::sync::{Mutex, OnceLock};
    static SEEN: OnceLock<Mutex<HashSet<(u32, bool)>>> = OnceLock::new();
    let seen = SEEN.get_or_init(|| Mutex::new(HashSet::default()));
    let mut seen = seen.lock().unwrap_or_else(|e| e.into_inner());
    if seen.insert((addr, gamma)) {
        report!(
            "gxm surface: {addr:#x} changed GAMMA-CORRECT mode to {gamma} on a live target - \
             the recorded mode and its cached bind groups are refreshed, so what samples this \
             target decodes the way the pass that wrote it encoded"
        );
    }
}

/// Report - once per surface - that a pass is rendering into a GAMMA-CORRECT colour
/// surface, and whether that is being honoured.
///
/// `honoured == false` means the colour format has no sRGB twin, so the ROP encode the
/// hardware performs is not happening and this surface (and everything sampling it) reads
/// darker than the title intends. That has to be said out loud - it is precisely the kind
/// of uniform darkening that gets chased as a lighting bug.
/// Report - once per surface - that the DISPLAY pass is rendering into a surface the guest
/// put in GAMMA-CORRECT mode, which this arm does not honour. See the call site.
/// Report - once per target - that a scene carrying no draws was skipped instead of being
/// begun, cleared and stored. See the call site for the measurement.
pub(super) fn report_empty_scene_skipped(addr: u32, w: u32, h: u32) {
    use std::sync::{Mutex, OnceLock};
    static SEEN: OnceLock<Mutex<HashSet<u32>>> = OnceLock::new();
    let seen = SEEN.get_or_init(|| Mutex::new(HashSet::default()));
    let mut seen = seen.lock().unwrap_or_else(|e| e.into_inner());
    if seen.insert(addr) {
        report_status!(
            "gxm chain: the scene targeting {addr:#x} ({w}x{h}) carries NO DRAWS, so no render pass is encoded for it. Beginning one would allocate tiles, clear the attachment and STORE it - a full-target write on a tiling GPU - to produce an image nothing in the frame wrote. The target is left holding what it already held, and it is NOT marked as rendered, so a later scene into the same address this frame is still its first pass and still clears."
        );
    }
}

pub(super) fn report_gamma_display_surface(addr: u32, w: u32, h: u32, honoured: bool) {
    use std::sync::{Mutex, OnceLock};
    static SEEN: OnceLock<Mutex<HashSet<u32>>> = OnceLock::new();
    let seen = SEEN.get_or_init(|| Mutex::new(HashSet::default()));
    let mut seen = seen.lock().unwrap_or_else(|e| e.into_inner());
    if seen.insert(addr) {
        if honoured {
            report_status!(
                "gxm display: {addr:#x} ({w}x{h}) is a GAMMA-CORRECT DISPLAY surface - rendering through the display image's sRGB view, so every store is sRGB-encoded after blending as the console's ROP does"
            );
        } else {
            // A WARNING, because it is a claim that the picture is WRONG and owed a
            // fix - every pixel of every draw stored linear where the hardware
            // encodes. It was written as `report!` (which is `tracing::debug!`) and
            // so did not exist at the default filter: three runs went by with the
            // display arm ignoring gamma mode and nothing said a word.
            // [[vitaslop-a-diagnostic-at-debug-is-a-diagnostic-that-does-not-exist]]
            report_warn!(
                "gxm display: {addr:#x} ({w}x{h}) is the DISPLAY surface and the guest put it in GAMMA-CORRECT mode, which this pass could NOT honour - it renders through a linear view, so every store keeps the value the shader produced where the ROP would sRGB-ENCODE it after blending. The whole frame reaches the panel DARKER than the title intends (a linear 0.5 shown where an encoded 0.73 belongs), uniformly, on every draw. The reachable causes are a SUPERSAMPLED pass or a bare host view (neither owns a display image to take an sRGB view of) and a colour format with no sRGB twin."
            );
        }
    }
}

pub(super) fn report_gamma_surface(addr: u32, honoured: bool) {
    use std::sync::{Mutex, OnceLock};
    static SEEN: OnceLock<Mutex<HashSet<u32>>> = OnceLock::new();
    let seen = SEEN.get_or_init(|| Mutex::new(HashSet::default()));
    let mut seen = seen.lock().unwrap_or_else(|e| e.into_inner());
    if seen.insert(addr) {
        if honoured {
            report!(
                "gxm surface: {addr:#x} is GAMMA-CORRECT - rendering through an sRGB view, \
                 so writes are sRGB-encoded after blending as the hardware does"
            );
        } else {
            report!(
                "gxm surface: {addr:#x} is GAMMA-CORRECT but the render format has no sRGB \
                 twin - its writes stay LINEAR where the hardware would encode them, so it \
                 and anything sampling it read darker than the title intends"
            );
        }
    }
}

/// Report - once per (pair, unit) - that a sampler was bound to a render target's DEPTH
/// rather than to any colour buffer.
///
/// The same class of event as [`rendered_alias`]: the runtime is substituting something
/// other than the guest bytes at the address the shader named, and that substitution
/// decides what the pass computes.
/// Say - once per address - that one guest address is registered BOTH as a colour target
/// this frame rendered and as a converted depth surface, and that the colour answer was
/// taken. See the depth/colour ordering in `plan_sampler_binds` for why this cannot be
/// true of a correct registration, and what it cost when it was.
pub(super) fn report_depth_is_also_colour(addr: u32) {
    use std::sync::{Mutex, OnceLock};
    static SEEN: OnceLock<Mutex<HashSet<u32>>> = OnceLock::new();
    let seen = SEEN.get_or_init(|| Mutex::new(HashSet::default()));
    if seen.lock().unwrap_or_else(|e| e.into_inner()).insert(addr) {
        report_warn!(
            "gxm rtt {addr:#x} is registered as BOTH a rendered colour target and a converted DEPTH surface. Binding the COLOUR: a sampler naming a colour target wants the image, and the depth path is consulted first, so this would otherwise hand it a distance. One of the two registrations is wrong."
        );
    }
}

/// Say - once per address - that a colour pass began into an address that held a converted
/// DEPTH surface, which was retired so no later sampler binds it there. See the scene loop.
pub(super) fn report_depth_retired_by_colour(addr: u32) {
    use std::sync::{Mutex, OnceLock};
    static SEEN: OnceLock<Mutex<HashSet<u32>>> = OnceLock::new();
    let seen = SEEN.get_or_init(|| Mutex::new(HashSet::default()));
    if seen.lock().unwrap_or_else(|e| e.into_inner()).insert(addr) {
        report_status!(
            "gxm rtt {addr:#x}: a COLOUR pass began into memory a depth pass used, so the depth converted there is retired - a sampler naming {addr:#x} now gets the colour image (or the guest's bytes), not a distance"
        );
    }
}

/// Say - once per address - that a carried-forward DEPTH surface was NOT bound for a
/// sampler naming its address, because the bound texture is a compressed format no GPU pass
/// can have written: the guest re-allocated that memory, and its own texture is decoded.
pub(super) fn report_depth_alias_refused_compressed(addr: u32, base_format: u32) {
    use std::sync::{Mutex, OnceLock};
    static SEEN: OnceLock<Mutex<HashSet<u32>>> = OnceLock::new();
    let seen = SEEN.get_or_init(|| Mutex::new(HashSet::default()));
    if seen.lock().unwrap_or_else(|e| e.into_inner()).insert(addr) {
        report!(
            "gxm rtt {addr:#x} is held as a render target's DEPTH surface, but a draw binds a                  COMPRESSED texture (base format {base_format:#x}) there - the guest re-allocated                  the memory, so the alias is refused and its own texture is sampled"
        );
    }
}

pub(super) fn report_depth_sample_bound(key: u64, unit: u8, addr: u32) {
    use std::sync::{Mutex, OnceLock};
    static SEEN: OnceLock<Mutex<HashSet<(u64, u8)>>> = OnceLock::new();
    let seen = SEEN.get_or_init(|| Mutex::new(HashSet::default()));
    let mut seen = seen.lock().unwrap_or_else(|e| e.into_inner());
    if seen.insert((key, unit)) {
        report!(
            "gxp pair {key:016x}: sampler unit {unit} names {addr:#x}, which is a render \
             target's DEPTH surface - binding this frame's converted depth"
        );
    }
}

/// Report - once per target - that a render target's depth was converted into the guest's
/// encoding for a later pass to sample, and which encoding that was.
///
/// Unconditional. Which value a GXM depth surface holds is REVERSE-ENGINEERED, not read
/// off a spec we hold, so every frame built on that reading has to say which reading it
/// used - otherwise a soft-particle fade that comes out subtly wrong looks like a shading
/// bug rather than a choice made here.
pub(super) fn report_depth_conversion(addr: u32, mode: u32, konst: f32, depth_min: f32, depth_scale: f32) {
    use std::sync::{Mutex, OnceLock};
    static SEEN: OnceLock<Mutex<HashSet<(u32, u32)>>> = OnceLock::new();
    let seen = SEEN.get_or_init(|| Mutex::new(HashSet::default()));
    let mut seen = seen.lock().unwrap_or_else(|e| e.into_inner());
    if seen.insert((addr, mode)) {
        if mode == 5 {
            report!(
                "gxm depth: target {addr:#x} depth REPLACED by the constant {konst} - this is \
                 a causality probe, and any frame built on it is not a real frame"
            );
            return;
        }
        let name = match mode {
            0 => "the guest's own window depth (a + c/w, measured from this pass)",
            1 => "w",
            2 => "-w",
            3 => "-1/w",
            6 => "1/w",
            _ => "the stored [0,1] depth",
        };
        let degenerate = if depth_scale == 0.0 {
            "  (the scene has NO depth range, so w cannot be recovered and the stored depth \
             is passed through unchanged)"
        } else {
            ""
        };
        report!(
            "gxm depth: target {addr:#x} depth re-encoded as {name} for a later pass to \
             sample (depth_min={depth_min}, depth_scale={depth_scale}){degenerate}"
        );
    }
}

/// Report - once per address - that a pass samples a depth buffer the frame did render,
/// but through the DISPLAY target rather than an offscreen one, where no converted copy
/// exists.
pub(super) fn report_unconverted_depth_sample(addr: u32) {
    use std::sync::{Mutex, OnceLock};
    static SEEN: OnceLock<Mutex<HashSet<u32>>> = OnceLock::new();
    let seen = SEEN.get_or_init(|| Mutex::new(HashSet::default()));
    let mut seen = seen.lock().unwrap_or_else(|e| e.into_inner());
    if seen.insert(addr) {
        report!(
            "gxm depth: a draw samples the depth at {addr:#x}, which belongs to a scene \
             rendered into the DISPLAY target - that pass keeps no depth copy, so this \
             sample reads guest bytes the GPU never wrote"
        );
    }
}

/// Report - once per pair and stage - a program that reads uniforms this renderer does not
/// feed it: anything declared in a NON-DEFAULT container, or a `UniformBuffer` binding.
///
/// # Why the bind site cannot say this
/// `sceGxmSet{Vertex,Fragment}UniformBuffer` and its precomputed twins already warn, once
/// per run, that a non-default buffer was bound and that the capture does not carry it. That
/// warning names the CALL, and the call is not the defect: a title may bind a buffer that no
/// shader on screen ever reads, in which case nothing is wrong and the warning is a false
/// alarm that cannot be cleared. The defect is a PROGRAM whose parameter table says it reads
/// one, because that program renders with zeros where the guest put data - silently, with
/// zero fallbacks and zero WebGPU errors, which is this project's signature failure shape.
///
/// So this is keyed by pair and stage and names the parameters, which makes it falsifiable:
/// a run with the bind-site warning and none of these is a run where the gap costs nothing.
pub(super) fn report_unfed_uniforms(key: u64, stage: &str, blob: &[u8]) {
    use std::sync::{Mutex, OnceLock};
    use vitaslop_gxp_shader::container::ParamCategory;
    static SEEN: OnceLock<Mutex<HashSet<(u64, String)>>> = OnceLock::new();
    let Ok(program) = vitaslop_gxp_shader::Program::parse(blob) else { return };
    // ONLY a `UniformBuffer` parameter, which is unambiguous: it IS a buffer binding.
    //
    // This deliberately does NOT also flag `Uniform` parameters by `container_index`. That
    // was the first version, on the assumption that container 0 is the default uniform
    // buffer and anything else is a bound buffer - and it fired on nearly every pair in the
    // title, naming ordinary uniforms (`bloomFactor`, `blendweights`) that demonstrably DO
    // reach the shader, because the picture is right. So container 14 is not "some other
    // buffer" and the assumption was wrong.
    //
    // What `container_index` actually indexes is not established here, and a diagnostic
    // built on an unverified reading of a field is worse than no diagnostic: it produced
    // 200 lines of confident noise on one run. It stays out until someone reads the
    // container table and can say what the index means.
    // A declared buffer is FED by either of the two paths that exist, and both are per
    // BUFFER rather than per program - so the two are subtracted from the declaration list
    // rather than used to suppress the whole report. Reporting a program because ONE of its
    // four buffers is unfed is right; reporting it because it declares four when three are
    // resident and the fourth is a window is a false alarm, and that is what this said
    // about every world program of a golf title while they rendered correctly.
    //
    // * SA-RESIDENT: the driver copies the buffer into the SA register file and
    //   `VitaState::sa_uniform_image` lays those bytes out per draw.
    // * A MEMORY WINDOW: the vertex program chases the buffer's bound address with 0xE8
    //   loads and the draw snapshots its bytes (`VitaState::capture_mem_windows`).
    //
    // >>> BOTH STAGES HAVE MEMORY WINDOWS, AND ASKING ONLY THE VERTEX ONE MADE THIS REPORT
    // >>> ACCUSE 47 PAIRS OF READING ZERO WHEN THE QUESTION WAS NEVER PUT.
    // The fragment stage got its own window (`gxp_fmem`, captured from
    // `sceGxmSetFragmentUniformBuffer` - a different table from the vertex stage's) after
    // this report was written, and the `else` arm here was left handing back an empty list.
    // A fragment buffer fed entirely through its window therefore came out UNFED on every
    // world pair of a title whose world was black for an unrelated reason, which is the
    // worst thing a diagnostic can do: it agreed with the symptom.
    let resident = program.sa_uniform_buffers();
    // * A STORE WINDOW: a fragment program writes the buffer (`resolve_store_windows`) and the
    //   draw snapshots it to seed the renderer's mirror (`capture_store_windows`).
    let windows = if stage == "vertex" {
        vitaslop_gxp_shader::mem_windows_for_vertex_blob(blob)
    } else {
        let mut w = vitaslop_gxp_shader::mem_windows_for_fragment_blob(blob);
        w.extend(vitaslop_gxp_shader::store_windows_for_fragment_blob(blob));
        w
    };
    let fed = |index: i32| {
        index >= 0
            && (resident.iter().any(|b| b.buffer_index == index as u32)
                || windows.iter().any(|w| w.buffer_index == index as u32))
    };
    let unfed: Vec<String> = program
        .parameters
        .iter()
        .filter(|p| matches!(p.category, ParamCategory::UniformBuffer))
        .filter(|p| !fed(p.resource_index))
        .map(|p| format!("{} (buffer {}, container {})", p.name, p.resource_index, p.container_index))
        .collect();
    if unfed.is_empty() {
        return;
    }
    let seen = SEEN.get_or_init(|| Mutex::new(HashSet::default()));
    if !seen.lock().unwrap_or_else(|e| e.into_inner()).insert((key, stage.to_string())) {
        return;
    }
    report_warn!(
        "gxp pair {key:016x} {stage}: the program reads {} uniform(s) from a NON-DEFAULT \
         buffer, which the capture does not carry - those registers read ZERO in the \
         recompiled shader: {}",
        unfed.len(),
        unfed.join(", ")
    );
}

/// A linked attribute WIDER than what the guest binds, whose surplus components are fed
/// [`attr_fill`] rather than guest data.
///
/// # Why this is worth a line of its own
/// The fill is a CHOICE (1.0, not the graphics API's `(0,0,0,1)` - see `attr_fill`), it is
/// invisible in the frame, and a shader that reads a filled lane reads a constant where the
/// device read geometry. It cost a wrong conclusion once already: a `VPROBE` frame that
/// came out white was read as "this UV is the attribute fill" when the guest binds the UV
/// in every submission and the white lane was a pixel coordinate. Naming the pair, the
/// location and the exact lane range makes that a lookup instead of an inference.
///
/// Once per `(pair, location)`, like every other report here - a per-draw line on a hot
/// pair buries the findings it exists to surface.
pub(super) fn report_attr_fill(
    key: u64,
    location: u32,
    base_lane: u32,
    bound: u8,
    slots: u8,
    guest_format: u8,
    fill: [f32; 4],
) {
    use std::sync::{Mutex, OnceLock};
    static SEEN: OnceLock<Mutex<HashSet<(u64, u32)>>> = OnceLock::new();
    let seen = SEEN.get_or_init(|| Mutex::new(HashSet::default()));
    if !seen.lock().unwrap_or_else(|e| e.into_inner()).insert((key, location)) {
        return;
    }
    let lanes: Vec<String> = (bound..slots)
        .map(|c| {
            format!("{}={}", ["x", "y", "z", "w"][c as usize], attr_fill(c as usize, fill[c as usize]))
        })
        .collect();
    // A 3-component bind into a declared `vec4` fills only `w`, and `w = 1` is what both the
    // hardware and every graphics API do with a position or a normal - it is ORDINARY, and
    // warning about it at the same volume as the rest is how a diagnostic buries the finding
    // it exists to surface. Anything WIDER than that is the case worth a warning: the fill
    // is then reaching `z` or beyond, where this renderer's 1.0 is a CHOICE (see `attr_fill`,
    // which records the title that lost its colour to the API's zero) rather than a
    // convention, and a shader reading that lane reads a constant where the device read
    // geometry.
    let only_w = bound == 3 && slots == 4;
    let subject = format!(
        "pair {key:016x} @location {location} (base lane {base_lane}), {slots} declared \
         vs {bound} bound, guest format {guest_format}, filling {}",
        lanes.join(", ")
    );
    if only_w {
        report!(
            "gxp attribute fill: {subject} - the ordinary vec3-into-vec4 case, where the \
             fill is the convention and not a choice."
        );
    } else {
        // >>> EVERY SUBJECT, ONE LINE EACH, AT DEBUG - because the census below keeps only
        // >>> the FIRST and a count, and the count is not the question.
        //
        // A phone dump reads `count=88`: eighty-eight attributes whose surplus lanes this
        // renderer fills with 1.0, of which the panel names one. Whether that fill is right
        // depends entirely on WHAT EACH LANE FEEDS - identity for a colour modulate, badly
        // wrong for a position or a corner offset, where a 1.0 in place of a 0 stretches
        // geometry. That question cannot be asked of a count, and answering it by having a
        // person toggle `VITASLOP_GXP_ATTR_FILL` on a device is guesswork: the subjects are
        // enumerable here, and each pair key names a program in the offline corpus whose
        // parameter list says what the attribute at that location IS.
        //
        // Debug, so the phone panel is unchanged and a desktop run with
        // `RUST_LOG=vitaslop::gxm=debug` produces the whole list.
        report!("gxp attribute fill SUBJECT: {subject}");
        // One census rather than one warning per (pair, location): a single round of one
        // title reaches 177 of these, which is more than the page's whole panel. See
        // [`Census`] for the capture that measured it.
        static CENSUS: crate::gpu::Census = crate::gpu::Census::new();
        // STATUS, not a warning: the 1.0 fill is the MEASURED value of the lanes the
        // device leaves unbound, so a wider declaration is the expected shape of this
        // title's shaders, not a fault owed a fix. It still reaches the panel.
        CENSUS.note_status(
            "gxp attribute fill: a linked attribute is declared WIDER than the guest \
             binds, so its surplus lane(s) read this renderer's fill of 1.0 rather than \
             guest data - a shader reading one of those lanes reads a constant where the \
             device read geometry.",
            &subject,
        );
    }
}

/// A pipeline whose vertex layout could NOT be bound against the guest's own row, and why.
///
/// # Why a fallback nobody can see still has to say so
/// The two plans draw the SAME PICTURE - that is the whole claim being made - so a layout
/// that quietly repacks is invisible in every frame, every shot and every hash. Only the
/// COST differs, and cost is exactly what this change is for: a title left on the repack
/// keeps paying the ~7x expansion, and without a line naming the attribute that put it there
/// the only way to find out is to go and read its shaders. One line per pair, with the FIRST
/// disqualifying attribute, turns that into a lookup.
///
/// Status rather than a warning: a format WebGPU has no equivalent for is not a fault owed a
/// fix, it is the reason the repack exists. Censused for the same reason `report_attr_fill`
/// is - one title reaches enough of these to fill a phone panel on its own.
pub(super) fn report_vertex_repack(key: u64, why: &str) {
    use std::sync::{Mutex, OnceLock};
    static SEEN: OnceLock<Mutex<HashSet<u64>>> = OnceLock::new();
    let seen = SEEN.get_or_init(|| Mutex::new(HashSet::default()));
    if !seen.lock().unwrap_or_else(|e| e.into_inner()).insert(key) {
        return;
    }
    // The census keeps the FIRST subject and a count, and the count is not the question:
    // "44 pipelines repack" says nothing about what to go and widen, while the reasons
    // themselves are a short histogram - a format with no equivalent, an offset, a fill.
    // At debug, so an ordinary run's panel is unchanged.
    report!("gxp vertex repack SUBJECT: pair {key:016x}: {why}");
    static CENSUS: crate::gpu::Census = crate::gpu::Census::new();
    CENSUS.note_status(
        "gxp vertex expand: some of this pipeline's attributes cannot keep the guest's own \
         component format, so every draw of it widens them to f32 - about 7x the bytes for \
         each one that takes it. See `passthrough_vertex_format`.",
        &format!("pair {key:016x}: {why}"),
    );
}

/// A pipeline that needs NO f32 expansion, whichever of the two ways it got there.
///
/// # Why the GOOD outcome is reported at all, and at status
/// [`report_vertex_repack`] fires only on the outcome worth fixing. That is right for a
/// warning and useless as a CONTROL: a title where every layout works produces no line at
/// all, so "it worked everywhere" reads exactly like "it never armed". It cost a verdict -
/// a racer's A/B came back 12 of 12 frames bit-identical with zero lines in EITHER arm, and
/// nothing in either log could show the arms had done anything different
/// [[vitaslop-report-that-only-fires-on-failure]].
///
/// ONE census for both good outcomes, so its count against `gxp vertex expand`'s is the
/// whole split - and one static, because two `static`s in two blocks are two counters that
/// each report a fraction and look like the total.
pub(super) fn report_vertex_plan_ok(subject: &str) {
    static CENSUS: crate::gpu::Census = crate::gpu::Census::new();
    CENSUS.note_status(
        "gxp vertex plans: this pipeline needs NO f32 expansion - its attributes keep the \
         guest's own component formats and the vertex fetch converts them. Read this count \
         against the `gxp vertex expand` one for the split.",
        subject,
    );
}
