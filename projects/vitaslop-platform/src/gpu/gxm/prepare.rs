//! `VITASLOP_PREPARE_SPLIT`: where the milliseconds inside one `prepare` go, and the
//! one-line reports of the prepare and encode tallies.

use super::*;

/// Where the milliseconds INSIDE one `prepare` go, plus the bytes each phase moved.
///
/// `EncodePhases` says `prepare` is 7.66 ms of a 9.08 ms encode. That names the phase and
/// nothing in it, and the candidates inside call for opposite fixes: hashing the whole
/// vertex stream to key a cache is bytes-per-frame the cache exists to avoid, copying the
/// packed result into the pass arena is bytes-per-frame the GPU genuinely needs uploaded,
/// and building a bind group is a GPU object. A time alone cannot separate them, so the
/// bytes ride along - the same argument `EncodeWork` is built on.
///
/// # Why this one is behind a knob when no other counter here is
/// Every other counter in this module is a relaxed add on a path that already makes a WebGPU
/// call. These are CLOCK READS on a path that makes none: in the browser `performance.now()`
/// is a wasm/JS boundary crossing ([[vitaslop-browser-host-call-cost]]), and six per draw
/// across several hundred draws a frame would cost more than the phase they measure and move
/// the number they exist to report. `VITASLOP_PREPARE_SPLIT=1` asks for them.
macro_rules! prepare_split {
    ($($(#[$m:meta])* $name:ident),+ $(,)?) => {
        #[derive(Default, Clone, Copy, Debug)]
        pub struct PrepareSplit { $($(#[$m])* pub $name: u64,)+ }

        pub(super) struct PrepareCounters { $(pub(super) $name: std::sync::atomic::AtomicU64,)+ }

        pub(super) static PREP: PrepareCounters = PrepareCounters {
            $($name: std::sync::atomic::AtomicU64::new(0),)+
        };

        /// Take and RESET every prepare sub-counter. The caller owns the window.
        pub fn take_prepare_split() -> PrepareSplit {
            PrepareSplit {
                $($name: PREP.$name.swap(0, std::sync::atomic::Ordering::Relaxed),)+
            }
        }

        impl PrepareSplit {
            /// Fold another tally in, so a caller can accumulate per PRESENT.
            pub fn add(&mut self, o: &PrepareSplit) {
                $(self.$name += o.$name;)+
            }
        }
    };
}

prepare_split! {
    /// >>> THE WHOLE OF `prepare`, SO THE SPLIT CAN BE HELD TO ACCOUNT.
    ///
    /// A phase table is only an argument if it accounts for the thing it divides. Without
    /// this the seven named phases below summed to ~0.7 ms a frame while the browser's own
    /// `RENDER SPLIT` put `prepare` at 2.5 - so 72% of the function had no instrument on it
    /// and nothing in the report could say so. Adding an eighth named phase on a hunch, with
    /// a residual that large, is guessing.
    ///
    /// Charged by a `Drop` guard rather than an explicit end call: `prepare` leaves through
    /// half a dozen `?` and `return None` paths (link refused, key filtered, no vertices,
    /// pipeline poisoned), and timing only the path that reaches the bottom under-reports
    /// exactly the CHEAP draws while the residual silently absorbs the difference.
    total_ns,
    /// Calls to `prepare`, so the total can be read per DRAW. A per-frame millisecond figure
    /// hides that a frame's draw count moves with what is on screen.
    draws,
    /// Nanoseconds in the cache-key and pipeline-lookup preamble, WITHOUT the build below.
    key_ns,
    /// Inside [`Self::key_ns`]: computing the draw's PAIR KEY.
    key_pair_ns,
    /// Inside [`Self::key_ns`]: `draw_colour_is_noop`, i.e. the colour-no-op FOLD and the
    /// two-level memo in front of it - which hashes this draw's uniform bytes and every
    /// bound memory window, per draw.
    key_fold_ns,
    /// Arena bytes written by a draw whose pipeline binds the GUEST ROW unchanged, by one
    /// that REPACKS, and - of the repacked ones - how many of those bytes exist only
    /// because an attribute had to widen to f32.
    ///
    /// >>> THE THIRD IS THE ONE THAT SAYS WHETHER THE EXPANSION IS WORTH REMOVING. "47
    /// pipelines expand" is a count of PIPELINES, and a pipeline that draws one 40-vertex
    /// quad costs the same in that count as the one that draws the stadium. The excess is
    /// `(packed_stride - the stride the same row would have with every attribute in the
    /// guest's own format) * vertices`, charged only where the arena is actually written -
    /// a resident draw is not re-uploaded and must not be counted as if it were.
    /// Of the content-cache HITS below, how many arrived in a DIFFERENT allocation than
    /// the one the entry was built from, and how many in the SAME one.
    ///
    /// >>> THESE TWO SEPARATE THE ONLY TWO EXPLANATIONS FOR A FULL-STREAM HASH. The
    /// allocation-keyed cache is the cheap path and the content hash is what a miss costs,
    /// so a frame that hashes megabytes is a frame whose allocation keys are not hitting -
    /// and that is either geometry arriving in a fresh buffer every frame (the interning
    /// upstream is not holding it) or an allocation-map entry that was EVICTED. The two
    /// have unrelated fixes and neither is visible in the hit/miss counts.
    packed_hit_new_alloc,
    packed_hit_same_alloc,
    /// Arena bytes that are INDICES rather than vertices. The two share one counter's
    /// worth of time and are read as if the vertex half were all of it; on a baseball
    /// title's stadium the index half is the larger of the two.
    arena_bytes_index,
    /// Arena bytes that are GUEST-MEMORY WINDOW blocks pushed into the uniform arena. The
    /// third contributor, and the one that made the byte line not add up: vertices plus
    /// indices came to little more than half of `arena_bytes` and nothing said what the
    /// rest was.
    arena_bytes_memwin,
    /// Of [`arena_bytes_memwin`], the bytes this FRAME had already pushed for the same
    /// (pipeline, window set) in an EARLIER PASS - see `GxpLive::mem_push_frame`. The size
    /// of the prize a frame-scoped window arena would win, measured before building one.
    arena_bytes_memwin_repeat,
    arena_bytes_passthrough,
    arena_bytes_repacked,
    arena_bytes_expand_excess,
    /// Nanoseconds spent BUILDING a pipeline this draw found missing - link, WGSL emit,
    /// `create_shader_module`, `create_render_pipeline`.
    ///
    /// >>> SEPARATE FROM `key_ns` BECAUSE POOLING THEM MADE THE TABLE LIE. The build was
    /// inside the key stopwatch, and a build is rare and enormous: one frame of a baseball
    /// title's shot window built 153 pipelines and rendered in 359 ms, which put `key` at
    /// 6.4 ms/frame averaged over the window and 10.2 ms/render averaged over warm
    /// re-renders that had built nothing at all. A lookup that costs a steady frame ~0.1 ms
    /// read as the largest phase in `prepare` and as the obvious thing to optimise next.
    pipe_build_ns,
    /// Nanoseconds in the whole GEOMETRY path - the packed-vertex cache probe, the resident
    /// heap lookup, the promotion bookkeeping and the index side of the same - with
    /// `hash`/`repack`/`arena` nested INSIDE it as sub-phases.
    ///
    /// Split out because those three are the work the caches exist to AVOID: when they are
    /// small and this is not, the cost is the bookkeeping itself (several map probes per
    /// draw over ~490 draws), which is a different fix from making a copy faster.
    vpath_ns,
    /// Nanoseconds hashing the guest vertex stream to key the packed-vertex cache, and the
    /// bytes that hash read. This is pure cache overhead: it is paid on a HIT as well as a
    /// miss, and it scales with the frame's whole vertex volume rather than with its misses.
    hash_ns,
    hash_bytes,
    /// Nanoseconds repacking a guest vertex stream the packed cache did not have, and the
    /// guest bytes those repacks read.
    repack_ns,
    repack_bytes,
    /// Nanoseconds copying packed vertices and guest indices into the pass arenas, and the
    /// bytes copied. Paid every frame for every draw, including one whose geometry has not
    /// changed since the renderer started.
    arena_ns,
    arena_bytes,
    /// Nanoseconds pushing the two SA blocks into the pass's uniform arena.
    uni_ns,
    /// The `uni_ns` phase SUB-SPLIT, because "uniforms is the biggest phase" does not name a
    /// fix. MEASURED on one title's race: `uniforms 0.95 ms` was 37% of a 2.56 ms `prepare`,
    /// and the two halves want OPPOSITE fixes - arena growth is a pooling question (already
    /// tried and reverted once for pinning a loading-frame peak in a wasm heap that never
    /// shrinks), while the guest-memory windows are a per-draw copy question. Arithmetic put
    /// the append at ~0.2 ms and could not account for the rest, so it is measured.
    ///
    /// `uni_sa_ns` is the two [`push_sa`] calls; `uni_mem_ns` is [`push_mem_windows`];
    /// `uni_sa_bytes` is what the SA pushes actually wrote, INCLUDING the alignment padding
    /// each one zero-fills - a draw needing 16 bytes still consumes a 256-byte slot, and
    /// whether that padding is the cost is the whole question.
    uni_sa_ns,
    uni_mem_ns,
    /// Window pushes answered by the per-pass memo (`push_mem_windows_memo`) - no copy.
    uni_mem_hits,
    uni_sa_bytes,
    /// `push_sa` calls that wrote nothing because the stage declares no SA lanes. A pair
    /// with none still pays the call and the alignment probe.
    uni_sa_empty,
    /// Nanoseconds in `make_sampler_bg` - group2, the one that is a GPU object per draw when
    /// its cache misses.
    sampler_ns,
    /// Nanoseconds in the group3 depth bind group.
    depth_ns,
    /// Draws that reached each of the two packed-vertex outcomes.
    packed_hits,
    packed_misses,
    /// Draws that repacked NOTHING because their pipeline binds the guest's own interleaved
    /// row (`GxpPipeline::passthrough`). Counted beside the two above because it is the
    /// outcome that makes them BOTH smaller: such a draw pays no hash, no repack and no
    /// cache entry, so a fall in `repack`/`hash` without this number rising would mean the
    /// frame drew less, not that it repacked less.
    packed_passthrough,
    /// Draws whose vertices, and draws whose indices, were bound where they already LIVE -
    /// the resident heaps ([`ResidentHeap`]) - against the bytes newly placed in them. A
    /// healthy steady state is high hit counts against near-zero placement: the placement is
    /// what a title pays as it reaches new geometry, and it should stop.
    resident_v_hits,
    resident_i_hits,
    resident_placed_bytes,
}

impl PrepareSplit {
    /// Whether anything was recorded at all - i.e. whether the knob was on.
    pub fn is_empty(&self) -> bool {
        self.key_ns == 0
            && self.pipe_build_ns == 0
            && self.hash_ns == 0
            && self.arena_ns == 0
            && self.sampler_ns == 0
            && self.packed_hits == 0
            && self.packed_misses == 0
            && self.packed_passthrough == 0
    }

    /// One line, per FRAME (the caller divides), naming every sub-phase above.
    pub fn line(&self, frames: u64) -> String {
        let n = frames.max(1) as f64;
        // The top-level phases only. `hash`/`repack`/`arena` are nested inside `vpath` and
        // adding them here would double-count them into a negative residual.
        let residual = self
            .total_ns
            .saturating_sub(self.key_ns)
            .saturating_sub(self.pipe_build_ns)
            .saturating_sub(self.vpath_ns)
            .saturating_sub(self.uni_ns)
            .saturating_sub(self.sampler_ns)
            .saturating_sub(self.depth_ns);
        let ms = |v: u64| v as f64 / n / 1.0e6;
        let mb = |v: u64| v as f64 / n / (1024.0 * 1024.0);
        let per = |v: u64| v as f64 / n;
        format!(
            "prepare split/frame: key {:.2} ms (of which pair-key {:.2} + colour-fold {:.2}, \
             the rest being the pipeline-cache lookup and the per-draw preamble around it) \
             (+ pipeline BUILDS {:.2} ms), vertex-hash {:.2} ms ({:.2} MB hashed), \
             repack {:.2} ms ({:.2} MB read, {:.0} misses vs {:.0} hits ({:.0} of them in a FRESH allocation, {:.0} in one the                  allocation cache had EVICTED), {:.0} draws bound \
             straight through), arena copy {:.2} ms \
             ({:.2} MB: {:.2} guest-row + {:.2} repacked vertices, of which {:.2} MB is f32 EXPANSION, + {:.2} MB of INDICES + {:.2} MB of guest-memory WINDOW blocks), uniforms {:.2} ms [SA {:.2} ms writing {:.2} MB over {:.0} pushes, \
             {:.0} of them empty; mem-windows {:.2} ms ({} pushes answered by the pass memo, {:.2} MB of the block bytes ALREADY PUSHED THIS FRAME in an earlier pass)], samplers {:.2} ms, depth {:.2} ms; RESIDENT \
             {:.0} vertex + {:.0} index draws bound in place, {:.2} MB newly placed. \
             TOTAL {:.2} ms over {:.0} draws ({:.1} us/draw), of which geom path {:.2} ms \
             (its own lookup+bookkeeping {:.2} ms, i.e. what is left after hash/repack/arena) \
             and {:.2} ms UNATTRIBUTED ({:.0}%)",
            ms(self.key_ns),
            ms(self.key_pair_ns),
            ms(self.key_fold_ns),
            ms(self.pipe_build_ns),
            ms(self.hash_ns),
            mb(self.hash_bytes),
            ms(self.repack_ns),
            mb(self.repack_bytes),
            per(self.packed_misses),
            per(self.packed_hits),
            per(self.packed_hit_new_alloc),
            per(self.packed_hit_same_alloc),
            per(self.packed_passthrough),
            ms(self.arena_ns),
            mb(self.arena_bytes),
            mb(self.arena_bytes_passthrough),
            mb(self.arena_bytes_repacked),
            mb(self.arena_bytes_expand_excess),
            mb(self.arena_bytes_index),
            mb(self.arena_bytes_memwin),
            ms(self.uni_ns),
            ms(self.uni_sa_ns),
            mb(self.uni_sa_bytes),
            // Two pushes per draw, so this is the call count, not the draw count.
            per(self.draws * 2),
            per(self.uni_sa_empty),
            ms(self.uni_mem_ns),
            per(self.uni_mem_hits),
            mb(self.arena_bytes_memwin_repeat),
            ms(self.sampler_ns),
            ms(self.depth_ns),
            per(self.resident_v_hits),
            per(self.resident_i_hits),
            mb(self.resident_placed_bytes),
            ms(self.total_ns),
            per(self.draws),
            if self.draws == 0 { 0.0 } else { self.total_ns as f64 / self.draws as f64 / 1000.0 },
            ms(self.vpath_ns),
            // What the geometry path costs BEYOND the three phases nested in it: the map
            // probes, the residency test and the promotion bookkeeping.
            ms(self
                .vpath_ns
                .saturating_sub(self.hash_ns)
                .saturating_sub(self.repack_ns)
                .saturating_sub(self.arena_ns)),
            // >>> `saturating_sub`, because the parts and the total are separate relaxed
            // counters: a window taken between two writes could otherwise go negative and
            // print as an enormous number, and an instrument that reads as a finding is the
            // one failure mode it must not have.
            ms(residual),
            if self.total_ns == 0 { 0.0 } else { 100.0 * residual as f64 / self.total_ns as f64 },
        )
    }
}

/// Whether [`take_prepare_split`]'s counters are being fed. Read once - see the macro above
/// for why this instrument is the one thing here that has to be asked for.
pub(crate) fn prepare_split_on() -> bool {
    use std::sync::OnceLock;
    static CELL: OnceLock<bool> = OnceLock::new();
    *CELL.get_or_init(|| crate::knobs::var("VITASLOP_PREPARE_SPLIT").map(|v| v.trim() != "0").unwrap_or(false))
}

/// A stopwatch that exists only when the split is on, so the OFF path reads no clock.
#[inline]
pub(super) fn split_start() -> Option<Stopwatch> {
    prepare_split_on().then(Stopwatch::start)
}

/// Charge `t`'s elapsed time to `c`. A no-op when the split is off, because `t` is `None`.
#[inline]
pub(super) fn split_end(t: Option<Stopwatch>, c: &std::sync::atomic::AtomicU64) {
    if let Some(t) = t {
        enc(c, (t.ms() * 1.0e6) as u64);
    }
}

/// Charge `t`'s elapsed time to `c` MINUS `less_ns`, which a nested phase has already
/// taken. Saturating: a nested measurement can only be shorter than the span containing it,
/// and if clock jitter ever says otherwise the answer is zero rather than a huge unsigned
/// wrap that would dwarf every other row of the table.
#[inline]
pub(super) fn split_end_less(t: Option<Stopwatch>, c: &std::sync::atomic::AtomicU64, less_ns: u64) {
    if let Some(t) = t {
        enc(c, ((t.ms() * 1.0e6) as u64).saturating_sub(less_ns));
    }
}

/// Charge `n` to a split counter, only when the split is on - so a reader never sees bytes
/// beside a zero time and reads the phase as free.
#[inline]
pub(super) fn split_add(c: &std::sync::atomic::AtomicU64, n: u64) {
    if prepare_split_on() {
        enc(c, n);
    }
}

/// Charges the WHOLE of `prepare` to [`PrepareSplit::total_ns`] however the function leaves.
///
/// A guard rather than a call at the bottom, because `prepare` returns early on half a dozen
/// paths and every one of them is a draw that happened. Timing only the complete path
/// under-reports the cheap draws and inflates the unattributed remainder by exactly the
/// amount that would have explained it.
pub(super) struct SplitTotal(Option<Stopwatch>);

impl SplitTotal {
    #[inline]
    pub(super) fn start() -> Self {
        let t = split_start();
        if t.is_some() {
            enc(&PREP.draws, 1);
        }
        SplitTotal(t)
    }
}

impl Drop for SplitTotal {
    fn drop(&mut self) {
        split_end(self.0.take(), &PREP.total_ns);
    }
}

impl EncodeWork {
    /// Unpack [`EncodeWork::buffer_write_worst`] into `(milliseconds, KB)` - the slowest
    /// single `queue.write_buffer` of the window and the bytes THAT call carried.
    ///
    /// Packed rather than stored as a pair so the two cannot drift onto different calls;
    /// see the field's own comment for what that cost.
    pub fn buffer_write_worst_us_kb(&self) -> (f64, f64) {
        let us = self.buffer_write_worst >> 32;
        let bytes = self.buffer_write_worst & 0xffff_ffff;
        (us as f64 / 1000.0, bytes as f64 / 1024.0)
    }

    /// One line, per FRAME (the caller divides), naming every unit above.
    pub fn line(&self, frames: u64) -> String {
        let n = frames.max(1) as f64;
        let mb = |v: u64| v as f64 / n / (1024.0 * 1024.0);
        let per = |v: u64| v as f64 / n;
        format!(
            "encode work/frame: {:.1} passes, {:.0} draws, {:.0} pipeline + {:.0} bind-group \
             + {:.0} vertex-buffer + {:.0} viewport + {:.0} scissor sets, textures {:.1} UPLOADED \
             ({:.2} MB, {:.1} COMPRESSED passthrough, {:.1} ENCODED ON THE GPU, {:.1} GPU \
             encodes refused, {:.1} RE-uploaded after eviction) / {:.1} cached ({:.2} view evict \
             passes dropping {:.1} entries, {:.1} superseded in place, {:.2} WHOLESALE clears, {:.1} DESTROYED), bind groups {:.1} built \
             / {:.1} reused, {:.2} pipelines built ({:.2} of them state variants of a pair \
             already built; {:.2} started ASYNC ahead, {:.2} built sync UNWARMED), buffers {:.1} created / {:.1} destroyed ({:.2} MB \
             written in {:.1} write_buffer CALLS, WORST SINGLE CALL {:.1} ms for {:.0} KB), rtt {:.2} created / {:.2} destroyed / {:.2} snapshots ({:.2} MB), {:.2} DESTINATION-COLOUR pass splits ({:.2} MB copied; {:.2} destination reads SHARED the copy before them; footprints from VERTICES for {:.1} draws, {:.1} refused to the viewport, {:.2} ms CPU), {:.2} depth \
             conversions, {:.2} depth-bind-cache clears, {:.1} draws ELIDED (colour = destination, no depth write; {} in this window - a per-frame mean that rounds to 0.0 cannot tell NEVER from RARELY, and those want opposite next steps), COLOUR-FOLD census/frame: {:.1} asks ({:.1} short-circuited by level one), {:.1} memo hits / {:.1} MISSES, {:.2} MB hashed over {:.1} windows ({:.0} B per ask) - the misses are what the FOLD costs and the bytes are what the HASH costs, and a fix that targets the wrong one has been reverted twice",
            per(self.passes),
            per(self.draw_calls),
            per(self.pipeline_sets),
            per(self.bind_group_sets),
            per(self.vertex_buffer_sets),
            per(self.viewport_sets),
            per(self.scissor_sets),
            per(self.tex_uploaded),
            mb(self.tex_upload_bytes),
            per(self.tex_uploaded_compressed),
            per(self.tex_encoded_on_gpu),
            per(self.tex_gpu_encode_refused),
            per(self.tex_reuploaded_after_evict),
            per(self.tex_view_cached),
            per(self.tex_view_evict_passes),
            per(self.tex_view_evicted),
            per(self.tex_view_superseded),
            per(self.tex_view_wholesale_clears),
            per(self.textures_destroyed),
            per(self.bind_groups_built),
            per(self.bind_groups_reused),
            per(self.pipelines_built),
            per(self.pipelines_built_variant),
            per(self.pipelines_async),
            per(self.pipelines_unwarmed),
            per(self.buffers_created),
            per(self.buffers_destroyed),
            mb(self.buffer_bytes),
            per(self.buffer_writes),
            // NOT divided by the window: this is a MAX over it, and dividing a maximum by
            // the number of frames produces a number that is nothing at all. It is also
            // FOLDED with `max` rather than `+=` - see the `@max` section of the macro.
            self.buffer_write_worst_us_kb().0,
            self.buffer_write_worst_us_kb().1,
            per(self.rtt_created),
            per(self.rtt_destroyed),
            per(self.rtt_snapshots),
            mb(self.rtt_snapshot_bytes),
            per(self.dest_splits),
            mb(self.dest_split_bytes),
            per(self.dest_reads_shared),
            per(self.dest_geom_bounded),
            per(self.dest_geom_refused),
            per(self.dest_geom_us) / 1000.0,
            per(self.depth_converts),
            per(self.depth_bg_cache_clears),
            per(self.draws_elided),
            self.draws_elided,
            per(self.fold_asks),
            per(self.fold_l1_rejects),
            per(self.fold_memo_hits),
            per(self.fold_memo_misses),
            mb(self.fold_hash_bytes),
            per(self.fold_hash_windows),
            // Bytes per ASK, not per frame: the per-frame figure moves with the draw count
            // and this one does not, so it is the one that says whether a single draw's
            // hash is reading a few dozen bytes or a few kilobytes.
            self.fold_hash_bytes as f64 / (self.fold_asks.max(1) as f64),
        )
    }
}
