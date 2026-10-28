//! What one encode spent and did: the phase split, the sampler bind-group counts and the
//! per-frame work tally.

use super::*;

/// Where one `encode` went, in milliseconds of CPU.
///
/// The recompiler path renders this title's main screen in ~30 ms warm against the
/// fixed-function path's ~8 ms, and the interesting question is WHICH SIDE that
/// delta is on: per-draw CPU work building GPU objects, or genuinely heavier
/// shaders on the GPU. They call for opposite fixes, so guessing is expensive.
///
/// `prepare` is the scene walk - for a recompiled draw that means creating its
/// buffers and bind groups, where the fixed-function path only appends bytes to a
/// grow-only arena. `upload` is the arena writes, `pass` is command encoding. All
/// three are CPU; GPU time shows up in the caller's submit-and-wait, which is why
/// the caller times that separately. Recorded unconditionally - three `Instant`s a
/// frame cost nothing next to the milliseconds they measure, and a diagnostic that
/// has to be switched on is one nobody has on when they need it.
#[derive(Clone, Copy, Debug, Default, PartialEq)]
pub struct EncodePhases {
    pub prepare_ms: f64,
    pub upload_ms: f64,
    pub pass_ms: f64,
    /// `upload` split in two, because the combined number named no fix. A course-load frame
    /// measured **2,642 ms of `upload` while uploading ZERO textures and ZERO bytes** and
    /// creating no buffer - so whatever it was, it was not "the arena writes" the doc above
    /// claims. The two halves have different fixes, so they are timed apart: `arena` is the
    /// recompiled path's three `write_buffer`s, `ubo_bg` is the per-pipeline uniform bind
    /// groups. Anything the two do not account for is inside the fixed-function block.
    pub arena_ms: f64,
    /// `arena_ms` split again, because its two halves have DIFFERENT fixes and the numbers
    /// alone cannot tell them apart: on the load frame the slot count and the write count
    /// are 1:1 (209 passes -> 643 buffers created and 627 `write_buffer` calls), so
    /// "2.2 ms each" is true of both and names neither. `create` is answered by sharing one
    /// arena across the frame's passes; `write` is answered by fewer, larger calls.
    pub arena_create_ms: f64,
    pub arena_write_ms: f64,
    pub ubo_bg_ms: f64,
    /// >>> THE PARTS OF `encode_chain` THAT ARE NOT A PASS, which used to be nothing but a
    /// >>> gap between `encode` and the three phases above.
    ///
    /// MEASURED on a long device run: `encode 19.3` against `prepare 12.8 + upload 0.8 +
    /// pass 1.5`, and on that run's WORST frame **107.3 ms against 8.8**. A split that
    /// leaves ninety-eight milliseconds unnamed is worse than none, because it reads as
    /// complete. These three are what runs there, per FRAME rather than per draw, so the
    /// clock reads cost nothing.
    ///
    /// `resident_ms` is the one to watch on a long run: it covers `grow_or_reset`, which
    /// COMPACTS the resident geometry heap by allocating a whole new buffer and copying
    /// every live slice into it.
    pub precompile_ms: f64,
    pub retire_ms: f64,
    pub resident_ms: f64,
    /// >>> THE THREE WALL TIMES THAT MAKE THE SPLIT AN ACCOUNT RATHER THAN A LIST.
    ///
    /// Every field above is a span SOMEBODY CHOSE TO TIME, so their sum can only ever be a
    /// lower bound on the encode - and on a baseball title's gameplay frame it was 24% of
    /// it. A residual that large says nothing about WHERE the rest is, and the obvious next
    /// step - "name another span" - is a guess whose candidate list is the whole function.
    ///
    /// These three close the frame instead of adding to the list. `encode_chain` is exactly
    /// HEAD + SCENE LOOP + TAIL, and the loop is exactly the `encode_pass` calls plus the
    /// per-scene setup around them, so:
    ///
    ///   head_other = chain_head - (precompile + retire + resident)
    ///   loop_other = scene_loop - pass_wall            (per-scene RTT setup, snapshots)
    ///   pass_other = pass_wall  - (prepare + upload + pass)
    ///   tail                                          (diagnostics, blit, resolve, flush)
    ///
    /// and those four plus the named spans are the encode, with nothing unaccounted. Four
    /// clock reads a FRAME and one per PASS, which is why this is affordable in the browser
    /// too. [[vitaslop-contradiction-means-look-between]]
    pub chain_head_ms: f64,
    pub scene_loop_ms: f64,
    /// Wall time inside `encode_pass` / `encode_depth_only_pass`, timed by the CALLER, so
    /// the gap between it and the per-pass named spans is the pass's own unnamed work
    /// (the dest-blend scan, `decide_scene_negw`, the summary diagnostics).
    pub pass_wall_ms: f64,
    pub chain_tail_ms: f64,
    /// The pre-prepare CLIP VERDICT - `GxpLive::decide_scene_negw` - and how many vertex
    /// programs it INTERPRETED to reach it (`count_clip_w_signs` over 256 vertices each).
    ///
    /// Split out of `pass_wall` because that residual pointed at a block with four
    /// candidates in it and this is the only one that can cost milliseconds: a pass whose
    /// draws produce no clip evidence is deliberately NOT cached, so it re-interprets every
    /// one of its pairs every frame, for ever. The COUNT is the load-bearing half - it is
    /// zero on a settled frame whatever the machine is doing.
    pub negw_ms: f64,
    pub clip_measures: usize,
    /// `chain_head_ms` broken into its three candidates, so "head unnamed 0.6 ms" is an
    /// answer rather than another place to guess. All three are per FRAME.
    ///
    /// `head_sweep_ms` is the one that can GROW WITH RUN LENGTH: it sums the packed-geometry
    /// maps' bytes and walks both promotion maps, so it is proportional to the CACHES, not to
    /// the frame. `head_scan_ms` is proportional to the frame's DRAWS (three walks of every
    /// draw's every sampler). `head_rtt_ms` is proportional to the live render targets.
    pub head_sweep_ms: f64,
    pub head_scan_ms: f64,
    pub head_rtt_ms: f64,
    /// Draws that took the recompiled path, and draws that took the fixed-function
    /// path - the denominator for any per-draw cost.
    pub gxp_draws: usize,
    pub fixed_draws: usize,
}

impl EncodePhases {
    /// Fold one PASS's phases into a whole-FRAME total.
    ///
    /// A frame is a CHAIN of passes, and reporting only the last one's phases describes
    /// whichever pass happened to be last - on this title's race frame that is an
    /// eleven-draw composite, so the split read `prepare 0.25 ms` while the frame was
    /// spending sixteen milliseconds encoding four hundred draws it never mentioned.
    pub(super) fn add(&mut self, pass: EncodePhases) {
        self.prepare_ms += pass.prepare_ms;
        self.upload_ms += pass.upload_ms;
        self.pass_ms += pass.pass_ms;
        self.arena_ms += pass.arena_ms;
        self.arena_create_ms += pass.arena_create_ms;
        self.arena_write_ms += pass.arena_write_ms;
        self.ubo_bg_ms += pass.ubo_bg_ms;
        self.precompile_ms += pass.precompile_ms;
        self.retire_ms += pass.retire_ms;
        self.resident_ms += pass.resident_ms;
        // NOT `chain_head_ms` / `scene_loop_ms` / `pass_wall_ms` / `chain_tail_ms`: those are
        // FRAME spans written straight onto `chain_phases` by `encode_chain`, and a fold runs
        // after they are known. Adding a pass's zeroes would be harmless today and would
        // silently clear them the first time a pass learned how to set one.
        self.negw_ms += pass.negw_ms;
        self.clip_measures += pass.clip_measures;
        // NOT the three head spans, for the same reason as the four wall times above.
        self.gxp_draws += pass.gxp_draws;
        self.fixed_draws += pass.fixed_draws;
    }
}

/// Sampler bind groups BUILT and sampler bind groups reused, since the last read.
///
/// A count, not a time: the browser has no `Instant` inside `encode`, and "how many GPU
/// objects did this frame create" is the question either way. See `GxpLive::sampler_bgs`.
/// >>> TWO ATOMICS, NOT A MUTEX. This is on the per-DRAW path - every draw of every
/// > > > frame passes through `note_sampler_bg`, several hundred a frame - and a `Mutex` there
/// > > > is a lock/unlock pair per draw to move a counter that is only ever added to. The two
/// > > > halves are read together by `take_sampler_bg_counts`, which is a report boundary and
/// > > > does not need them to be one atomic transaction: a count that lands in the next window
/// > > > instead of this one is a rounding error in a diagnostic, where the lock was real work
/// > > > on the hottest path in the renderer.
pub(super) static SAMPLER_BG_REUSED: std::sync::atomic::AtomicU64 = std::sync::atomic::AtomicU64::new(0);
pub(super) static SAMPLER_BG_BUILT: std::sync::atomic::AtomicU64 = std::sync::atomic::AtomicU64::new(0);

/// Draws that took the PREVIOUS draw's group without resolving a single unit - see
/// `make_sampler_bg`. Separate from the reuse count above, because the two say different
/// things: that one is "the group already existed", this one is "the work that finds it did
/// not run either", and only the second is what the fingerprint bought.
pub(super) static SAMPLER_BG_PREV: std::sync::atomic::AtomicU64 = std::sync::atomic::AtomicU64::new(0);

/// Draws answered from a group THIS PASS already decided, but not by the draw immediately
/// before - see `GxpLive::sampler_pre`. Counted apart from `SAMPLER_BG_PREV` because the
/// two measure different caches, and the whole reason the second exists is that the first
/// was measured to miss two draws in three.
pub(super) static SAMPLER_BG_PASS: std::sync::atomic::AtomicU64 = std::sync::atomic::AtomicU64::new(0);

/// Record one draw answered from the pass-wide fingerprint map, and count it as a reuse.
pub(crate) fn note_sampler_bg_pass() {
    SAMPLER_BG_PASS.fetch_add(1, std::sync::atomic::Ordering::Relaxed);
    note_sampler_bg(true);
}

/// Take and reset [`SAMPLER_BG_PASS`]. The caller owns the window.
pub fn take_sampler_bg_pass() -> u64 {
    SAMPLER_BG_PASS.swap(0, std::sync::atomic::Ordering::Relaxed)
}

/// Record one draw answered from the previous draw's group, and count it as a reuse too.
pub(crate) fn note_sampler_bg_prev() {
    SAMPLER_BG_PREV.fetch_add(1, std::sync::atomic::Ordering::Relaxed);
    note_sampler_bg(true);
}

/// Take and reset [`SAMPLER_BG_PREV`]. The caller owns the window.
pub fn take_sampler_bg_prev() -> u64 {
    SAMPLER_BG_PREV.swap(0, std::sync::atomic::Ordering::Relaxed)
}

/// Record one sampler bind group as reused (`hit`) or freshly built.
pub(crate) fn note_sampler_bg(hit: bool) {
    let slot = if hit { &SAMPLER_BG_REUSED } else { &SAMPLER_BG_BUILT };
    slot.fetch_add(1, std::sync::atomic::Ordering::Relaxed);
    // Sampler groups are also bind groups, and the encode tally's job is to account for
    // EVERY GPU object the frame creates. Counted in both places on purpose: this pair has
    // its own line because group2 is the one a draw can accidentally make per draw.
    enc(if hit { &ENC.bind_groups_reused } else { &ENC.bind_groups_built }, 1);
}

/// Take and reset `(reused, built)` sampler bind-group counts.
pub fn take_sampler_bg_counts() -> (u64, u64) {
    (
        SAMPLER_BG_REUSED.swap(0, std::sync::atomic::Ordering::Relaxed),
        SAMPLER_BG_BUILT.swap(0, std::sync::atomic::Ordering::Relaxed),
    )
}

/// What one `encode_chain` DID, counted rather than timed.
///
/// # Why a count when the phases are now timed on both engines
/// A time says WHICH phase, never WHAT IN IT. `encode` costs 143 ms on a burst frame in the
/// browser and 8 ms on the desktop, and the two obvious stories - "it uploads 178 MB of
/// expanded texture" and "it makes six WebGPU calls per draw across the wasm/JS boundary"
/// - live in the same phase, scale with different things, and have opposite fixes. Bytes
/// and call counts tell them apart in one run; a millisecond cannot, ever.
///
/// It is also the only figure that survives a comparison BETWEEN the two engines, which is
/// the comparison this project actually makes. Same title, same frame: if one side uploads
/// forty times the bytes or makes four times the calls, that is the finding. If both do the
/// identical work and one is twenty times slower, the cost is per-call overhead and the fix
/// is batching, not volume - and that conclusion is unreachable from two clocks on two
/// machines.
///
/// Every counter is a relaxed atomic add on a path that already makes a WebGPU call, so the
/// instrument costs nothing next to what it measures. It is NOT behind a knob, for the usual
/// reason: a diagnostic you have to switch on is one nobody has on when the surprising frame
/// is already in front of them.
///
/// # A MAXIMUM IS NOT A SUM, AND THIS MACRO USED TO ADD ONE
/// Every field here is folded across the window by [`EncodeWork::add`], and `add` did
/// `+=` to all of them. That is right for a count and WRONG for a maximum: a per-frame
/// worst call summed over a window is a number with no meaning, and it read as one that
/// had - a device dump reported `1.44 MB written in 15.2 write_buffer CALLS, WORST SINGLE
/// CALL 1.0 ms for 14080 KB`, i.e. a single call carrying ten times the bytes of the whole
/// frame it happened in. The `@max` section below is folded with `max` instead. Put a
/// counter there if it is a high-water mark and in the list above if it is a tally; there
/// is no third kind.
macro_rules! encode_work {
    ($($(#[$m:meta])* $name:ident),+ $(,)? ; @max $($(#[$mm:meta])* $mname:ident),+ $(,)?) => {
        #[derive(Default, Clone, Copy, Debug)]
        pub struct EncodeWork {
            $($(#[$m])* pub $name: u64,)+
            $($(#[$mm])* pub $mname: u64,)+
        }

        pub(super) struct EncodeCounters {
            $(pub(super) $name: std::sync::atomic::AtomicU64,)+
            $(pub(super) $mname: std::sync::atomic::AtomicU64,)+
        }

        pub(super) static ENC: EncodeCounters = EncodeCounters {
            $($name: std::sync::atomic::AtomicU64::new(0),)+
            $($mname: std::sync::atomic::AtomicU64::new(0),)+
        };

        /// Everything any `take_encode_work` has taken, for the whole run - never reset.
        static RUN: EncodeCounters = EncodeCounters {
            $($name: std::sync::atomic::AtomicU64::new(0),)+
            $($mname: std::sync::atomic::AtomicU64::new(0),)+
        };

        /// Take and RESET every encode counter. The caller owns the window it divides by.
        /// What it takes is also folded into the run's total (see [`run_encode_work`]).
        pub fn take_encode_work() -> EncodeWork {
            use std::sync::atomic::Ordering::Relaxed;
            let w = EncodeWork {
                $($name: ENC.$name.swap(0, Relaxed),)+
                $($mname: ENC.$mname.swap(0, Relaxed),)+
            };
            $(RUN.$name.fetch_add(w.$name, Relaxed);)+
            $(RUN.$mname.fetch_max(w.$mname, Relaxed);)+
            w
        }

        /// >>> THE RUN'S CUMULATIVE ENCODE WORK, SAFE TO DIFFERENCE ACROSS ANY SPAN.
        ///
        /// `peek_encode_work` reads counters the live renderer TAKES (resets) on every
        /// present, so a difference of two peeks across a present reads the reset as zero:
        /// the slowest-frames list and the stall note both said "passes 0 draws 0" for a
        /// 165 ms present. This is the taken total plus what is not yet taken.
        pub fn run_encode_work() -> EncodeWork {
            use std::sync::atomic::Ordering::Relaxed;
            EncodeWork {
                $($name: RUN.$name.load(Relaxed) + ENC.$name.load(Relaxed),)+
                $($mname: RUN.$mname.load(Relaxed).max(ENC.$mname.load(Relaxed)),)+
            }
        }

        /// Read every encode counter WITHOUT resetting it - for a caller that differences two
        /// reads around one piece of work (a slow early batch) while the panel's window keeps
        /// its own `take`.
        pub fn peek_encode_work() -> EncodeWork {
            EncodeWork {
                $($name: ENC.$name.load(std::sync::atomic::Ordering::Relaxed),)+
                $($mname: ENC.$mname.load(std::sync::atomic::Ordering::Relaxed),)+
            }
        }

        impl EncodeWork {
            /// Fold another tally in, so a caller can accumulate per PRESENT.
            pub fn add(&mut self, o: &EncodeWork) {
                $(self.$name += o.$name;)+
                // >>> MAXIMA, NOT TOTALS. See the macro's own doc comment.
                $(self.$mname = self.$mname.max(o.$mname);)+
            }
        }
    };
}

encode_work! {
    /// Render passes begun, and the draw/state calls inside them. Each one is a call
    /// across the wasm/JS boundary in the browser and a function call on the desktop,
    /// which is the whole reason to count them separately from the bytes.
    passes,
    draw_calls,
    /// Draws NOT issued because their colour was provably the destination unchanged and
    /// they wrote no depth - see `draw_colour_is_noop`. Counted rather than assumed: a
    /// title with no such draws and a proof that never fires read the same from outside,
    /// and they want completely different next steps.
    draws_elided,
    /// >>> THE COLOUR-NO-OP FOLD'S OWN CENSUS, AND WHY IT IS HERE RATHER THAN IN
    /// >>> `PrepareSplit`.
    ///
    /// `draw_colour_is_noop` is the largest single CPU item in a frame on a baseball title
    /// (5-11 ms, eliding 0.0 draws) and two attempts to make it cheaper were reverted for
    /// guessing at the repeat structure. The question a fix needs answered first is which
    /// half costs the time - the per-draw BYTE HASH in front of the memo, or the FOLD a
    /// memo miss runs - and that is a question about COUNTS and BYTES, not about time.
    ///
    /// So these are plain relaxed adds on a path that already takes a mutex, always on,
    /// and readable from an ordinary browser run or a device dump. `PrepareSplit` cannot
    /// answer it: it is gated because its CLOCK READS are themselves slow enough to move
    /// the picture on this very title [[vitaslop-instrument-failure-imitating-its-subject]],
    /// so the one instrument that could name this cost is the one that must not be armed
    /// while measuring it.
    ///
    /// `fold_asks` counts draws that got PAST level one (the per-blob "can this program
    /// ever be an identity"), i.e. the ones that pay the hash at all. `fold_hash_bytes`
    /// is what that hash actually read, which is the number the notes asked for.
    fold_asks,
    fold_memo_hits,
    fold_memo_misses,
    fold_hash_bytes,
    /// Memory windows walked by the hash, summed over asks - so the bytes can be read as
    /// "how many windows" against "how big each one is", which want different fixes.
    fold_hash_windows,
    /// Draws that never reached the hash because level one said the program can never be
    /// an identity. The denominator: without it, a small `fold_asks` reads as a cheap
    /// fold rather than as a title the short-circuit is carrying.
    fold_l1_rejects,
    pipeline_sets,
    bind_group_sets,
    vertex_buffer_sets,
    viewport_sets,
    /// `set_scissor_rect` calls, from the guest's `sceGxmSetRegionClip`. Zero on a title
    /// that never scissors, which is what makes a nonzero value here worth seeing: a
    /// scissor changes WHICH PIXELS a draw may touch, so a frame that looks wrong on a
    /// title with a nonzero count has a suspect this counter names.
    scissor_sets,
    /// Textures UPLOADED (a `write_texture`) and the RGBA8 bytes that went with them,
    /// against the ones a warm view cache served for free. This is the counter that
    /// answers "is encode the 178 MB".
    tex_uploaded,
    tex_upload_bytes,
    tex_view_cached,
    /// Times a view cache went over its BYTE budget and ran an EVICTION PASS, and how many
    /// entries those passes dropped.
    ///
    /// # These were `tex_view_clears`, and the name outlived the policy
    /// It counted WHOLESALE clears. When this cache went per-entry the counting site moved
    /// with it and the name did not, so a healthy cache shedding cold entries under steady
    /// pressure - which is the design - reported "2.10 view-cache clears per frame" and read
    /// as the cliff the rewrite had removed. The decode cache had exactly the same stale
    /// name and cost a session to the same misreading; this is the other half of that fix.
    ///
    /// **Neither of these is the cliff signal.** [`Self::tex_reuploaded_after_evict`] is.
    tex_view_evict_passes,
    tex_view_evicted,
    /// Uploads released because a newer upload of the SAME guest texture took their slot -
    /// see `GxpLive::view_slots`. Not evictions: nothing can ask for those bytes again.
    tex_view_superseded,
    /// Textures PUT INTO the recompiler's view cache. The third term of the handle ledger
    /// and the one that was missing: `report_texture_handle_drift` asserted "created =
    /// cached + destroyed" with no measurement of how many creations ever reached the
    /// cache, so a creation that went somewhere else was indistinguishable from a leak.
    tex_view_inserted,
    /// Times the FIXED-FUNCTION renderer's view cache was cleared WHOLESALE on reaching its
    /// byte budget. This is the last wholesale clear in the engine and it is a real cliff:
    /// once the working set exceeds the budget it fires part-way through every frame and
    /// throws away exactly what the rest of that frame is about to ask for.
    ///
    /// It is counted separately from the per-entry passes above precisely so the two are not
    /// added together - they are different failure shapes and only one of them is a bug. It
    /// reads zero on any title whose draws all recompile (`0 fixed`), which is why it has not
    /// been worth rewriting; a nonzero value here means a title is on the fallback path AND
    /// over budget, and then it should be rewritten the way the GXP one was.
    tex_view_wholesale_clears,
    /// Textures UPLOADED whose view this run had already EVICTED - the cache thrashing.
    ///
    /// An upload is expensive and a cold upload is unavoidable, so the count of uploads
    /// cannot separate "this frame reached new content" from "we are re-uploading what we
    /// threw away last frame". This can. A cache under healthy pressure evicts what is not
    /// coming back and leaves this near zero however many passes run.
    tex_reuploaded_after_evict,
    /// Bind groups and pipelines CREATED against ones a cache served. A GPU object per
    /// draw is the mistake this project has made twice.
    bind_groups_built,
    bind_groups_reused,
    pipelines_built,
    /// Of the pipelines built, those for a PAIR that already had a pipeline under another
    /// baked state (cull, samples, format, layout, raster). The module is shared, so on a
    /// native driver these are cheap; a browser's GPU process may still compile the
    /// backend shader per pipeline. A race that keeps building these mid-lap is the case
    /// for creating the common variants ahead of the draw.
    pipelines_built_variant,
    /// Pipelines STARTED ahead of the frame with `createRenderPipelineAsync` - see
    /// `GxmRenderer::warm_pipelines` - and, of `pipelines_built`, those the pre-pass did NOT
    /// predict and the encode therefore created synchronously, on the browser's GPU thread,
    /// with async creation on. The second is the pre-pass's miss count: it should be ~0.
    pipelines_async,
    pipelines_unwarmed,
    /// Buffers created, and bytes written into a buffer (`write_buffer` plus the
    /// create-with-contents arenas). Vertex/index/uniform volume, separable from texture
    /// volume because the two have completely different fixes.
    buffers_created,
    buffer_bytes,
    /// `queue.write_buffer` CALLS, which is the unit this cost is actually billed in -
    /// see the write site in `ensure_gxp_arena`. Reported beside the byte count because a
    /// frame cannot be diagnosed from either alone: 27 MB in three calls and 27 MB in six
    /// hundred are the same bytes and, measured, nothing like the same milliseconds.
    buffer_writes,
    /// Buffers explicitly `destroy()`ed (the previous frame's arenas, released at the start
    /// of the next one). Reported beside `buffers_created` on purpose: the two should track
    /// each other in a steady frame, and a persistent gap between them is the signature of
    /// GPU memory the renderer has stopped naming but has not released.
    buffers_destroyed,
    /// Offscreen render targets created or resized, targets SNAPSHOTTED (a full
    /// texture-to-texture copy so a pass can sample what it also draws into) and the
    /// bytes in those copies, and depth surfaces converted to a sampleable float.
    /// All three are per-frame GPU work no draw count predicts.
    rtt_created,
    /// Offscreen render targets `destroy()`ed because the guest rebuilt them (a resize, a
    /// new depth reader, a changed sample count). Beside `rtt_created` for the same reason
    /// `buffers_destroyed` sits beside `buffers_created`: a target is several textures, so
    /// a standing gap between the two is a large leak on the one engine where Drop does not
    /// free - see [`RttSurface::destroy`].
    rtt_destroyed,
    rtt_snapshots,
    rtt_snapshot_bytes,
    /// Render-pass SPLITS taken so a draw whose fragment program reads the DESTINATION
    /// colour could be given a copy of the attachment - and the bytes those copies moved.
    ///
    /// Reported because it is the one cost this renderer pays that the draw count cannot
    /// predict and that a TILING GPU pays far more for than a desktop one: every split is a
    /// full store and reload of the framebuffer's tiles. See `GxmRenderer::dest_color`.
    dest_splits,
    dest_split_bytes,
    /// Destination reads that needed NO split: nothing written since the last copy meets
    /// the reader's footprint, so the copy already holds what it reads - see `DestDirty`.
    dest_reads_shared,
    /// Draws whose destination-copy footprint came from their own projected VERTICES
    /// (`clip::draw_pixel_bounds`), draws it could not bound (the viewport stood), and the
    /// CPU microseconds the interpretation cost.
    dest_geom_bounded,
    dest_geom_refused,
    dest_geom_us,
    depth_converts,
    /// Wholesale clears of the depth-range bind-group cache ([`GxpLive::depth_bgs`]).
    ///
    /// A nonzero value is NOT a fault - it is the bound doing its job on a screen whose
    /// depth range moves. It is reported because the alternative reading of a cache that
    /// never hits is a cache that is never bounded, and this one was: the counter is what
    /// makes "entries are being minted and never reused" visible from a capture instead of
    /// from a memory graph taken hours later.
    depth_bg_cache_clears,
    /// Textures explicitly `destroy()`ed on eviction from the recompiler's view cache. Reads
    /// zero wherever the cache never evicts (the desktop, on this title); on a device under
    /// budget pressure it should track the eviction count.
    textures_destroyed,
    /// GPU textures CREATED. Against `textures_destroyed` and the cache size this is the
    /// only thing that can see a handle dropped instead of released - see
    /// `report_texture_handle_drift`.
    textures_created,
    /// Of the uploads above, how many handed the guest's own BLOCKS to the GPU instead of a
    /// decoded RGBA8 image.
    ///
    /// This has to be counted separately from the megabytes beside it. A working set that
    /// does not shrink has two completely different causes - the passthrough never fired
    /// (this reads 0), or it fired and the remaining formats are the weight (this reads
    /// high) - and the byte total alone cannot tell them apart.
    tex_uploaded_compressed,
    /// Of those, how many had their blocks BUILT ON THE GPU from the guest's own bytes,
    /// with no CPU decode and no CPU encode ([`crate::texenc`]).
    ///
    /// The number this is read against is `tex_gpu_encode_refused`. A transition that is
    /// still slow with this high is slow for some other reason; a transition that is still
    /// slow with the refusals high is one where the shaders declined the shapes that matter,
    /// and those two have nothing in common to fix.
    tex_encoded_on_gpu,
    /// Uploads that asked for a GPU transcode and were declined by it, falling back to the
    /// CPU decode. Never a wrong picture - only a slower one.
    tex_gpu_encode_refused,
    /// Of `tex_encoded_on_gpu`, the ones the ETC2 ENCODER made (`texenc::Transcoder::run`),
    /// as opposed to a decode straight to RGBA8 (`expand_rgba8`). The two cost very
    /// differently on a phone GPU and the shared counter could not say which a stall was.
    tex_etc2_encoded,
    ;
    @max
    /// >>> THE WORST SINGLE `queue.write_buffer` OF THE WINDOW, IN MICROSECONDS, AND THE
    /// >>> BYTES THAT SAME CALL CARRIED.
    ///
    /// The pair that says WHAT this call is waiting on, which neither a total nor a call
    /// count can. MEASURED on the user's phone, one gameplay window: `arena 81.1 = write
    /// 81.1` over `2.88 MB in 11.9 calls` - 6.8 ms a call for 242 KB, and a worst frame of
    /// 2,427 ms for 3.36 MB in 12. No copy of 242 KB costs 6.8 ms and no crossing into the
    /// browser costs it either, so the call is BLOCKING; these two say whether it is one
    /// call blocking (a wait for something specific - the previous submit, a staging ring
    /// that has to be recycled) or all of them equally (a per-call cost that is simply
    /// enormous). Those have opposite fixes, and the notes have been carrying the question
    /// unanswered because nothing timed the calls individually.
    ///
    /// # THE TWO HALVES HAVE TO COME FROM THE SAME CALL, so they are ONE counter
    /// This was a pair of independent `fetch_max`es, which is not a pair at all: the
    /// microseconds came from the slowest call and the bytes from the FATTEST, and nothing
    /// said they were the same one. That is the exact question this counter exists to
    /// answer - a slow call carrying few bytes is a BLOCK, a slow call carrying many is a
    /// COPY - so an unpaired reading cannot answer it even when both numbers are correct.
    /// They are packed into one `u64` as `us << 32 | bytes` and maximised TOGETHER, so the
    /// winner is the slowest call and the bytes are the ones IT carried.
    ///
    /// It lives in the `@max` section because [`EncodeWork::add`] folds a window with `+=`
    /// otherwise, and a summed maximum is what produced `WORST SINGLE CALL 1.0 ms for
    /// 14080 KB` on a frame that wrote 1.44 MB in total.
    ///
    /// Read it with [`EncodeWork::buffer_write_worst_us_kb`].
    buffer_write_worst,
}

#[cfg(test)]
mod encode_work_tests {
    use super::EncodeWork;

    /// A window is folded with [`EncodeWork::add`], and a MAXIMUM folded with `+=` is not a
    /// maximum. This is the test the shipped instrument did not have: it reported `WORST
    /// SINGLE CALL 1.0 ms for 14080 KB` on a frame whose every write together came to
    /// 1.44 MB, and nothing failed.
    #[test]
    fn a_window_maxes_the_worst_write_and_sums_the_rest() {
        let frame = |us: u64, bytes: u64, writes: u64| EncodeWork {
            buffer_writes: writes,
            buffer_write_worst: (us << 32) | bytes,
            ..EncodeWork::default()
        };
        let mut w = EncodeWork::default();
        w.add(&frame(1_000, 200 * 1024, 5));
        w.add(&frame(6_800, 242 * 1024, 7));
        w.add(&frame(300, 900 * 1024, 3));

        assert_eq!(w.buffer_writes, 15, "a tally is still summed across the window");
        let (ms, kb) = w.buffer_write_worst_us_kb();
        assert!((ms - 6.8).abs() < 1e-9, "the window's worst call is 6.8 ms, got {ms}");
        // >>> AND THE BYTES ARE THAT CALL'S. The 900 KB frame is the FATTEST write of the
        // window and must not appear here: a slow call carrying little is a BLOCK and a slow
        // call carrying a lot is a COPY, and pairing the two maxima independently reports
        // the second when the truth is the first.
        assert!((kb - 242.0).abs() < 1e-9, "expected the slow call's own 242 KB, got {kb}");
    }
}
