//! The GPU's own clock: timestamp queries around every pass of a frame.

/// Timestamps at both ends of every render pass of a frame, so a run can say how long the
/// GPU itself took - the one figure the work-done latency cannot give.
///
/// # Why this exists
/// The browser's `GPU WORK-DONE LATENCY` times a submit to the moment the event loop
/// dispatches the queue's completion callback. On a worker that yields the loop about
/// twice a frame, that figure carries up to a whole tick of dispatch delay on top of the
/// GPU's own time, and cannot say which of the two it is looking at. A device reading 44 ms
/// of latency against a 27 ms period could be a GPU taking 17 ms and a callback landing
/// late, or a GPU taking 40 ms - and those need opposite fixes. `timestamp-query` writes
/// the GPU's own clock at the start and end of each pass.
///
/// # Shape
/// One query pair per pass, up to [`Self::PAIRS`] a frame. After the chain is encoded the
/// pairs are resolved into a buffer, copied to a mappable one on the same submit, and the
/// map is requested after the submit; the values are read on a LATER present, whenever the
/// map has completed, so nothing here waits on the GPU. One readback is in flight at a
/// time: while it is, the frame's passes take no pairs, which is why the report says how
/// many frames it measured.
///
/// The report is the per-pass execution time SUMMED (what the GPU spent on this frame) and
/// the SPAN from the first pass's start to the last pass's end (what the frame occupied,
/// including any gaps the GPU idled through). Sum near the frame budget is a GPU that
/// cannot make the rate whatever the CPU does; a span well above the sum is a GPU waiting
/// on something between passes.
pub struct GpuTimestamps {
    set: std::sync::Arc<wgpu::QuerySet>,
    resolve: wgpu::Buffer,
    read: wgpu::Buffer,
    /// Nanoseconds per timestamp tick, from the queue.
    period_ns: f32,
    /// Pairs taken by the chain being encoded, and what each pass was.
    next: u32,
    labels: Vec<String>,
    /// Whether the chain being encoded may take pairs: set when no readback is in flight.
    armed: bool,
    /// The readback in flight: how many pairs, their labels, and where the map is.
    ///
    /// The `bool` is the [`super::dest_split_armed`] arm the frame was ENCODED under. It has
    /// to ride along: the readback lands frames later, by which time the arm has flipped, so
    /// attributing the time to whatever arm is current at poll time would credit half of
    /// each arm's frames to the other one and drive the difference to zero.
    pub(super) in_flight: Option<(u32, Vec<String>, bool)>,
    /// 0 nothing to map, 1 copy encoded and awaiting the submit, 2 map requested,
    /// 3 mapped, 4 map failed.
    state: std::sync::Arc<std::sync::atomic::AtomicU8>,
    /// The window's statistics: measured frames, summed pass time, its max, summed span.
    win_frames: u32,
    win_sum_ms: f64,
    win_max_ms: f64,
    win_span_ms: f64,
    /// The mean GPU ms per measured frame of the window `take_report` LAST returned, kept
    /// after the reset so a caller that knows the display period can state the saturation
    /// share. Without it the panel's reader has to cross-reference PACING and divide.
    pub(super) last_mean_ms: f64,
    /// The NEWEST measured frame's summed pass time, and how many frames have been measured
    /// this run. Read by the page's GPU budget (see `GxmRenderer::ts_latest`), which needs
    /// the latest figure as it lands rather than a window's mean, and the sequence number to
    /// tell "no new measurement" from "the same cost again".
    pub(super) latest_ms: f64,
    pub(super) latest_seq: u64,
    /// `VITASLOP_GXM_DEST_SPLIT_AB`'s two arms: `(frames, summed GPU ms)` for the arm that
    /// TOOK its destination-colour splits and for the arm that did not. Cumulative over the
    /// run rather than per window, because the answer wanted is one number per arm and a
    /// window's worth of either is a small sample of a moving scene.
    ab_split: (u32, f64),
    ab_nosplit: (u32, f64),
    /// The most recently measured frame's passes, in SUBMISSION ORDER, as
    /// `(ordinal, label, ms, gap_before_ms)`.
    ///
    /// # THE ORDINAL AND THE GAP ARE WHY THIS IS NOT JUST `(label, ms)`
    /// A phone dump showed `1024x512 0 draws 30.28 ms` beside `640x368 117 draws 1.11 ms` -
    /// a pass drawing NOTHING costing twenty-seven times the pass drawing the whole frame.
    /// Nothing about a clear-and-store of a 2 MB attachment costs 30 ms, so the duration is
    /// not this pass's own work, and the report as it stood could not say what it was
    /// instead. The two candidates need different fixes and one number separates them:
    ///
    ///  * it is the FIRST pass of the frame, and on a tile-based GPU the begin-of-pass
    ///    timestamp is written when the command buffer reaches it while the tile work runs
    ///    later - so a queue that is BACKED UP has its wait absorbed by whichever pass is
    ///    first, and the fix is upstream (bound the queue depth; see
    ///    `LivePlayback::gpu_in_flight`), not in this pass at all;
    ///  * it is a pass in the MIDDLE with a large gap before it, which is the GPU genuinely
    ///    idling between passes waiting on this thread.
    ///
    /// A list sorted by cost, with no ordinal and no gap, is consistent with both. So the
    /// ordinal is printed and the gap from the previous pass's end is computed here, where
    /// the raw GPU ticks still exist - the report only sees milliseconds.
    last_passes: Vec<(usize, String, f64, f64)>,
    /// The guest frame (see [`super::set_frame_tag`]) the readback in flight was encoded on.
    in_flight_tag: u64,
    /// >>> THE COSTLIEST MEASURED FRAMES OF THE RUN, WITH EVERY PASS - `(summed ms, span ms,
    /// >>> guest frame, passes)`, costliest first, at most [`Self::WORST_KEPT`].
    ///
    /// `last_passes` is whichever frame landed last, so the one frame worth reading - a fighting
    /// title's round start, measured at 1,034 ms of GPU on the phone (job 078) - was overwritten a
    /// frame later and the panel could only say it had happened. A frame that expensive is
    /// rare by construction, so it is KEPT, named by guest frame so it lines up with the
    /// page's LONGEST DISPLAY GAPS, and printed pass by pass - including the texture
    /// transcodes, which take pairs of their own (see [`lent_ts_pair`]).
    worst: Vec<(f64, f64, u64, Vec<(usize, String, f64, f64)>)>,
}

thread_local! {
    /// The frame's [`GpuTimestamps`], LENT for the span of `GxpLive::prepare` - see
    /// [`lent_ts_pair`].
    static LENT_TS: std::cell::RefCell<Option<GpuTimestamps>> = const { std::cell::RefCell::new(None) };
}

/// >>> A TEXTURE TRANSCODE IS GPU WORK IN THE FRAME, AND UNTIMED IT HID INSIDE THE NEXT PASS.
///
/// The transcoder (`crate::texenc`) submits its compute passes while a frame's textures are
/// prepared, ahead of the frame's own render passes. On a tiling GPU the first timed pass
/// after them then absorbs their whole cost, so a fighting title's round start read `1,034 ms` against a
/// render pass with the transcodes invisible. The renderer lends its timestamps here around
/// the prepare, and the transcoder takes a labelled pair per pass like any render pass.
pub(crate) fn lent_ts_pair(
    label: impl FnOnce() -> String,
) -> Option<(std::sync::Arc<wgpu::QuerySet>, u32)> {
    LENT_TS.with(|c| c.borrow_mut().as_mut().and_then(|t| t.take(label)))
}

pub(super) fn lend_ts(ts: Option<GpuTimestamps>) {
    LENT_TS.with(|c| *c.borrow_mut() = ts);
}

pub(super) fn reclaim_ts() -> Option<GpuTimestamps> {
    LENT_TS.with(|c| c.borrow_mut().take())
}

impl GpuTimestamps {
    /// Query pairs a frame may take. A retail frame is 11-14 passes; the rest is headroom
    /// for a title that draws more scenes, and anything past it is simply unmeasured.
    const PAIRS: u32 = 1024;
    /// Frames kept in [`Self::worst`].
    const WORST_KEPT: usize = 4;

    pub(super) fn new(device: &wgpu::Device, queue: &wgpu::Queue) -> Option<Self> {
        if !device.features().contains(wgpu::Features::TIMESTAMP_QUERY) {
            return None;
        }
        let set = device.create_query_set(&wgpu::QuerySetDescriptor {
            label: Some("gxm-frame-timestamps"),
            ty: wgpu::QueryType::Timestamp,
            count: Self::PAIRS * 2,
        });
        let bytes = (Self::PAIRS * 2 * 8) as u64;
        let resolve = device.create_buffer(&wgpu::BufferDescriptor {
            label: Some("gxm-frame-timestamps-resolve"),
            size: bytes,
            usage: wgpu::BufferUsages::QUERY_RESOLVE | wgpu::BufferUsages::COPY_SRC,
            mapped_at_creation: false,
        });
        let read = device.create_buffer(&wgpu::BufferDescriptor {
            label: Some("gxm-frame-timestamps-read"),
            size: bytes,
            usage: wgpu::BufferUsages::MAP_READ | wgpu::BufferUsages::COPY_DST,
            mapped_at_creation: false,
        });
        Some(GpuTimestamps {
            set: std::sync::Arc::new(set),
            resolve,
            read,
            period_ns: queue.get_timestamp_period(),
            next: 0,
            labels: Vec::new(),
            armed: false,
            in_flight: None,
            state: std::sync::Arc::new(std::sync::atomic::AtomicU8::new(0)),
            win_frames: 0,
            win_sum_ms: 0.0,
            win_max_ms: 0.0,
            win_span_ms: 0.0,
            last_mean_ms: 0.0,
            latest_ms: 0.0,
            latest_seq: 0,
            ab_split: (0, 0.0),
            ab_nosplit: (0, 0.0),
            last_passes: Vec::new(),
            in_flight_tag: 0,
            worst: Vec::new(),
        })
    }

    /// Take the next pair for a pass, or `None` when the frame is not being measured.
    /// The label is built only when a pair is taken.
    pub(super) fn take(&mut self, label: impl FnOnce() -> String) -> Option<(std::sync::Arc<wgpu::QuerySet>, u32)> {
        if !self.armed || self.next >= Self::PAIRS {
            return None;
        }
        let i = self.next;
        self.next += 1;
        self.labels.push(label());
        Some((self.set.clone(), i))
    }

    /// A new chain: it may take pairs only if the previous readback has been collected.
    pub(super) fn begin_chain(&mut self) {
        self.next = 0;
        self.labels.clear();
        self.armed = self.in_flight.is_none();
    }

    /// Encode the resolve and the copy to the mappable buffer, after the chain's passes.
    pub(super) fn finish_chain(&mut self, encoder: &mut wgpu::CommandEncoder) {
        if !self.armed || self.next == 0 {
            return;
        }
        let n = self.next;
        encoder.resolve_query_set(&self.set, 0..n * 2, &self.resolve, 0);
        encoder.copy_buffer_to_buffer(&self.resolve, 0, &self.read, 0, (n * 2 * 8) as u64);
        self.in_flight = Some((n, std::mem::take(&mut self.labels), crate::gpu::dest_split_armed()));
        self.in_flight_tag = crate::gpu::frame_tag();
        self.state.store(1, std::sync::atomic::Ordering::Relaxed);
        self.armed = false;
    }

    /// Request the map, once the submit carrying the copy has been made.
    pub(super) fn map_after_submit(&mut self) {
        use std::sync::atomic::Ordering;
        let Some((n, _, _)) = self.in_flight.as_ref() else { return };
        if self.state.load(Ordering::Relaxed) != 1 {
            return;
        }
        self.state.store(2, Ordering::Relaxed);
        let state = self.state.clone();
        let bytes = (*n * 2 * 8) as u64;
        self.read.slice(0..bytes).map_async(wgpu::MapMode::Read, move |r| {
            state.store(if r.is_ok() { 3 } else { 4 }, Ordering::Relaxed);
        });
    }

    /// Collect a completed readback into the window's statistics.
    pub(super) fn poll(&mut self) {
        use std::sync::atomic::Ordering;
        match self.state.load(Ordering::Relaxed) {
            3 => {}
            4 => {
                // A failed map is dropped and the next frame measures again; the failure is
                // not silent because the frame count in the report stays low.
                self.state.store(0, Ordering::Relaxed);
                self.in_flight = None;
                return;
            }
            _ => return,
        }
        let Some((n, labels, ab_armed)) = self.in_flight.take() else { return };
        let bytes = (n * 2 * 8) as u64;
        let words: Vec<u64> = match self.read.slice(0..bytes).get_mapped_range() {
            Ok(view) => view
                .chunks_exact(8)
                .map(|c| u64::from_le_bytes([c[0], c[1], c[2], c[3], c[4], c[5], c[6], c[7]]))
                .collect(),
            Err(_) => Vec::new(),
        };
        self.read.unmap();
        self.state.store(0, Ordering::Relaxed);
        if words.len() < (n * 2) as usize {
            return;
        }
        let ns = self.period_ns as f64;
        let mut sum = 0.0f64;
        let mut first = u64::MAX;
        let mut last = 0u64;
        // The previous TIMED pass's end, in raw GPU ticks, for the gap - see `last_passes`.
        // `None` until one has been seen, so the first pass reports no gap rather than a
        // gap measured from zero.
        let mut prev_end: Option<u64> = None;
        self.last_passes.clear();
        for (i, label) in labels.into_iter().enumerate() {
            let (b, e) = (words[i * 2], words[i * 2 + 1]);
            // A driver may hand back a zero for a query it could not time; leave such a
            // pass out of the sum rather than credit it with the whole epoch.
            if b == 0 || e == 0 || e < b {
                continue;
            }
            let ms = (e - b) as f64 * ns / 1.0e6;
            let gap = prev_end.map_or(0.0, |p| b.saturating_sub(p) as f64 * ns / 1.0e6);
            prev_end = Some(e);
            sum += ms;
            first = first.min(b);
            last = last.max(e);
            self.last_passes.push((i, label, ms, gap));
        }
        if self.last_passes.is_empty() {
            return;
        }
        let span = (last - first) as f64 * ns / 1.0e6;
        if self.worst.len() < Self::WORST_KEPT || self.worst.last().is_some_and(|w| w.0 < sum) {
            self.worst.push((sum, span, self.in_flight_tag, self.last_passes.clone()));
            self.worst.sort_by(|a, b| b.0.partial_cmp(&a.0).unwrap_or(std::cmp::Ordering::Equal));
            self.worst.truncate(Self::WORST_KEPT);
        }
        self.win_frames += 1;
        self.latest_ms = sum;
        self.latest_seq += 1;
        self.win_sum_ms += sum;
        self.win_max_ms = self.win_max_ms.max(sum);
        self.win_span_ms += span;
        if crate::gpu::dest_split_ab().is_some() {
            let arm = if ab_armed { &mut self.ab_split } else { &mut self.ab_nosplit };
            arm.0 += 1;
            arm.1 += sum;
        }
    }

    /// The window's report, resetting it. `None` until a frame has been measured.
    pub(super) fn take_report(&mut self) -> Option<String> {
        if self.win_frames == 0 {
            // >>> AND ZERO THE MEAN A CALLER MIGHT STATE A SHARE FROM. Leaving the previous
            // window's figure here would let a "nothing was measured" report carry a
            // confident saturation percentage about a window that measured nothing - an
            // instrument lying rather than going quiet, which is worse.
            self.last_mean_ms = 0.0;
            return None;
        }
        let n = self.win_frames as f64;
        let mut passes = self.last_passes.clone();
        // `VITASLOP_GPU_TIME_ALL=1`: EVERY pass, in frame order, instead of the costliest
        // five - with `VITASLOP_GXP_PASS_SPLIT_EVERY=<n>` that is the GPU cost of each
        // n-draw slice of a big pass, which is how a main pass's time is attributed to the
        // draws (and so the shaders) that spend it.
        let all = crate::gpu::gpu_time_all_passes();
        if !all {
            passes.sort_by(|a, b| b.2.partial_cmp(&a.2).unwrap_or(std::cmp::Ordering::Equal));
        }
        let top: Vec<String> = passes
            .iter()
            .take(if all { usize::MAX } else { 5 })
            .map(|(i, l, ms, gap)| format!("#{i} {l} {ms:.2} ms (gap before {gap:.2})"))
            .collect();
        // `VITASLOP_GXM_DEST_SPLIT_AB`: the verdict, prepended so it is the first thing
        // read and cannot be lost in the tail of a long line. BOTH arms must have frames
        // before it says anything - one arm alone is not an A/B, it is a reading.
        let ab = match (self.ab_split, self.ab_nosplit) {
            ((sn, sms), (nn, nms)) if sn > 0 && nn > 0 => {
                let (a, b) = (sms / sn as f64, nms / nn as f64);
                format!(
                    ">>> DEST-SPLIT A/B: splits ON {a:.1} ms/frame over {sn} frames, OFF \
                     {b:.1} ms over {nn} frames (difference {:.1} ms). >>> DO NOT READ THAT \
                     DIFFERENCE AS THE COST OF THE SPLITS. It sums time over PASSES, and that \
                     sum is dominated by the queue wait the FIRST pass absorbs - one phone \
                     frame timed `#0 128x128 1 draw` at 422 ms beside a real main pass at 94 \
                     ms. So it reads whichever arm sat behind a deeper queue and its SIGN IS \
                     ARBITRARY: +3.0 ms on one title and -6.9 ms on another, i.e. removing \
                     work made the GPU faster. Price a pass cut by COUNTS - `passes/frame`, \
                     `DESTINATION-COLOUR pass splits`, MB copied - which cannot absorb a \
                     wait. The ALTERNATION is sound and validated (the ON arm is \
                     bit-identical to a normal run); it is this METRIC that is void. And \
                     never arm this beside VITASLOP_PERF: that halves the frame rate and so \
                     changes the queue depth being measured. ",
                    a - b
                )
            }
            _ => String::new(),
        };
        let out = format!(
            "{ab}GPU execution {:.1} ms per frame (max {:.1}) summed over its passes, spanning \
             {:.1} ms from the first pass's start to the last pass's end, over {} measured \
             frame(s) this window; the last measured frame's costliest passes: {}. This is \
             the GPU's OWN clock from a timestamp query, so unlike the work-done latency it \
             carries no event-loop delay. A sum near the display period is a GPU that \
             cannot make the rate whatever the CPU does; a span well above the sum is a GPU \
             idling between passes, i.e. waiting on this thread. >>> `#N` IS THE PASS'S                  POSITION IN THE FRAME AND IT DECIDES HOW TO READ A LARGE ONE: a big `#0`,                  especially on a pass with few or no draws, is NOT that pass's work - on a                  tiling GPU the first pass absorbs the wait for a queue that is already deep,                  so read it beside GPU BACKPRESSURE and the work-done latency and fix the                  depth, not the pass. A big pass in the MIDDLE, or a large `gap before`, is                  real: that is GPU work, or the GPU idling on this thread between passes.",
            self.win_sum_ms / n,
            self.win_max_ms,
            self.win_span_ms / n,
            self.win_frames,
            top.join(", "),
        );
        // The run's costliest measured frames, every pass of each in frame order - see `worst`.
        let worst: Vec<String> = self
            .worst
            .iter()
            .map(|(sum, span, tag, passes)| {
                let list: Vec<String> = passes
                    .iter()
                    .map(|(i, l, ms, gap)| {
                        if *gap >= 1.0 {
                            format!("#{i} {l} {ms:.1} (gap {gap:.1})")
                        } else {
                            format!("#{i} {l} {ms:.1}")
                        }
                    })
                    .collect();
                format!("f{tag} {sum:.1} ms summed / {span:.1} span: {}", list.join(", "))
            })
            .collect();
        let out = if worst.is_empty() {
            out
        } else {
            format!(
                "{out} | >>> COSTLIEST MEASURED FRAMES THIS RUN (guest frame, GPU ms summed                      over passes / first-start-to-last-end span, then EVERY pass in frame order                      with its ms; a `texenc` pass is a texture transcode): {}",
                worst.join(" || ")
            )
        };
        self.last_mean_ms = self.win_sum_ms / n;
        self.win_frames = 0;
        self.win_sum_ms = 0.0;
        self.win_max_ms = 0.0;
        self.win_span_ms = 0.0;
        Some(out)
    }
}
