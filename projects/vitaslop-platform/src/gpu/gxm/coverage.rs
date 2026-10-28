//! `VITASLOP_GXM_DRAW_COVERAGE`: the samples each draw of a frame actually wrote.

/// `VITASLOP_GXM_DRAW_COVERAGE=1`: how many SAMPLES each draw of the frame actually wrote,
/// straight from the GPU's own occlusion counter.
///
/// # The one question a black frame asks that nothing else here can answer
/// Every other diagnostic in this file describes a draw's INPUTS - its pair, its bindings,
/// its blend, its clip box, its viewport. None of them says whether the draw put anything on
/// the screen, and a frame that comes back black is consistent with all of:
///
///   * the geometry never rasterised (degenerate transform, wrong viewport, depth-rejected);
///   * it rasterised and the fragment program wrote black;
///   * it rasterised, wrote colour, and a later draw painted over it.
///
/// Those have completely different fixes, and telling them apart used to cost a
/// `VITASLOP_DRAW_RANGE` bisection - one twelve-minute replay per guess, on a title whose
/// world pass is 560 draws. An occlusion query answers the first one outright and for every
/// draw at once: a count of zero is a draw the rasteriser threw away, and a large count on a
/// black frame moves the question to the fragment stage, where it belongs.
///
/// Occlusion queries are CORE WebGPU - no adapter feature, no fallback - so this works in the
/// browser exactly as it does here.
///
/// # Why the readback is synchronous
/// It is a diagnostic that already changes the frame's cost (a query around every draw), and
/// the answer is wanted on the frame that was asked about rather than two frames later. The
/// timestamp harness above is asynchronous because it is meant to be affordable at all times;
/// this is not, and pretending otherwise would only make the report harder to read.
pub(super) struct DrawCoverage {
    pub(super) set: std::sync::Arc<wgpu::QuerySet>,
    resolve: wgpu::Buffer,
    read: wgpu::Buffer,
    /// The next free query index in the frame being encoded.
    next: u32,
    /// One entry per taken index, in encode order: the pass ordinal in the frame, the
    /// draw's position in its pass, its pair key, and the index count it was issued with.
    ///
    /// The INDEX COUNT is here because zero samples has two innocent causes and one
    /// interesting one, and this separates them without another run: a draw issued with no
    /// indices at all never had geometry (a capture or repack defect), while a draw issued
    /// with thousands of indices and covering nothing is a transform, viewport or depth
    /// question.
    taken: Vec<(usize, usize, u64, u32)>,
    /// Passes encoded so far in the frame being encoded, so a report can say WHICH pass a
    /// draw belongs to. A frame here is a chain of passes and the same pair is routinely
    /// drawn in several of them; a bare draw index would merge them.
    pub(super) pass: usize,
}

impl DrawCoverage {
    /// Draws a frame may measure. A retail frame here is under a thousand; past this the
    /// remaining draws are simply unmeasured and the report says how many it covered, so a
    /// truncated frame cannot be mistaken for a complete one.
    const MAX: u32 = 4096;

    pub(super) fn new(device: &wgpu::Device) -> Self {
        let set = device.create_query_set(&wgpu::QuerySetDescriptor {
            label: Some("gxm-draw-coverage"),
            ty: wgpu::QueryType::Occlusion,
            count: Self::MAX,
        });
        let bytes = (Self::MAX * 8) as u64;
        DrawCoverage {
            set: std::sync::Arc::new(set),
            resolve: device.create_buffer(&wgpu::BufferDescriptor {
                label: Some("gxm-draw-coverage-resolve"),
                size: bytes,
                usage: wgpu::BufferUsages::QUERY_RESOLVE | wgpu::BufferUsages::COPY_SRC,
                mapped_at_creation: false,
            }),
            read: device.create_buffer(&wgpu::BufferDescriptor {
                label: Some("gxm-draw-coverage-read"),
                size: bytes,
                usage: wgpu::BufferUsages::MAP_READ | wgpu::BufferUsages::COPY_DST,
                mapped_at_creation: false,
            }),
            next: 0,
            taken: Vec::new(),
            pass: 0,
        }
    }

    pub(super) fn begin_chain(&mut self) {
        self.next = 0;
        self.pass = 0;
        self.taken.clear();
    }

    /// Claim the next query index for a draw, or `None` when the frame's budget is spent.
    pub(super) fn take(&mut self, pass: usize, draw: usize, key: u64, indices: u32) -> Option<u32> {
        if self.next >= Self::MAX {
            return None;
        }
        let i = self.next;
        self.next += 1;
        self.taken.push((pass, draw, key, indices));
        Some(i)
    }

    /// Encode the resolve and the copy to the mappable buffer, after the chain's passes and
    /// on the SAME encoder, so the counts describe the frame that was just built.
    pub(super) fn finish_chain(&mut self, encoder: &mut wgpu::CommandEncoder) {
        if self.next == 0 {
            return;
        }
        encoder.resolve_query_set(&self.set, 0..self.next, &self.resolve, 0);
        encoder.copy_buffer_to_buffer(&self.resolve, 0, &self.read, 0, (self.next * 8) as u64);
    }

    /// After the submit that carried [`Self::finish_chain`]: map the counts and report them.
    ///
    /// Blocking, and it says so - see the type comment. `device.poll` is what makes the map
    /// complete on a native backend; the browser has no such call and this whole report is
    /// therefore native-only, which the knob's own line states rather than leaving a browser
    /// run silently unmeasured.
    pub(super) fn report_blocking(&mut self, device: &wgpu::Device) {
        if self.next == 0 {
            return;
        }
        let bytes = (self.next * 8) as u64;
        let (tx, rx) = std::sync::mpsc::channel();
        self.read.slice(0..bytes).map_async(wgpu::MapMode::Read, move |r| {
            let _ = tx.send(r.is_ok());
        });
        let _ = device.poll(wgpu::PollType::wait_indefinitely());
        if rx.recv() != Ok(true) {
            report_warn!(
                "gxm draw coverage: the readback map FAILED, so this frame is unmeasured"
            );
            return;
        }
        let counts: Vec<u64> = match self.read.slice(0..bytes).get_mapped_range() {
            Ok(v) => v
                .chunks_exact(8)
                .map(|c| u64::from_le_bytes([c[0], c[1], c[2], c[3], c[4], c[5], c[6], c[7]]))
                .collect(),
            Err(_) => Vec::new(),
        };
        self.read.unmap();
        if counts.len() < self.taken.len() {
            return;
        }
        // Per PASS, because that is the unit a reader is asking about ("the world pass draws
        // 560 things and the screen is black"), and per PAIR within it, because a pair is
        // what a fix acts on. A per-draw list of 600 lines would be unreadable and would say
        // no more: draws of one pair in one pass either all cover something or all do not,
        // and where they do NOT the interesting number is how many indices were thrown away.
        let mut by_pass: std::collections::BTreeMap<usize, Vec<(usize, u64, u32, u64)>> =
            std::collections::BTreeMap::new();
        for (i, &(p, d, key, idx)) in self.taken.iter().enumerate() {
            by_pass.entry(p).or_default().push((d, key, idx, counts[i]));
        }
        for (p, rows) in by_pass {
            let drawn: usize = rows.iter().filter(|r| r.3 > 0).count();
            let samples: u64 = rows.iter().map(|r| r.3).sum();
            // Every pair that covered NOTHING, with the draws and indices it threw away.
            // This is the list the report exists for; a pair that covered something needs no
            // explaining and would only bury it.
            // ...WITH THE DRAW ORDINALS, as ranges: `seq` in the per-draw dump
            // (`VITASLOP_DUMP_DRAW_GXP`) and the position here are the same count, so
            // this is the join between a pair key and the guest's own program handles.
            let mut empty: std::collections::BTreeMap<u64, (usize, u64, Vec<usize>)> =
                std::collections::BTreeMap::new();
            let mut painted: std::collections::BTreeMap<u64, (usize, u64, Vec<usize>)> =
                std::collections::BTreeMap::new();
            for &(d, key, idx, n) in &rows {
                let e = if n == 0 { &mut empty } else { &mut painted };
                let slot = e.entry(key).or_insert((0, 0, Vec::new()));
                slot.0 += 1;
                slot.1 += if n == 0 { idx as u64 } else { n };
                slot.2.push(d);
            }
            let ranges = |v: &[usize]| {
                let mut out: Vec<String> = Vec::new();
                let mut i = 0;
                while i < v.len() {
                    let mut j = i;
                    while j + 1 < v.len() && v[j + 1] == v[j] + 1 {
                        j += 1;
                    }
                    out.push(if j > i { format!("{}-{}", v[i], v[j]) } else { v[i].to_string() });
                    i = j + 1;
                }
                out.join(",")
            };
            let fmt = |m: &std::collections::BTreeMap<u64, (usize, u64, Vec<usize>)>, what: &str| {
                let mut v: Vec<_> = m.iter().collect();
                v.sort_by_key(|(_, (_, n, _))| std::cmp::Reverse(*n));
                v.iter()
                    .take(12)
                    .map(|(k, (d, n, at))| format!("{k:016x} x{d} ({n} {what}) at {}", ranges(at)))
                    .collect::<Vec<_>>()
                    .join(", ")
            };
            // >>> AND THE BIGGEST INDIVIDUAL DRAWS, because the per-pair total cannot find a
            // >>> single draw that covers half the screen.
            //
            // The grouping above rests on "draws of one pair either all cover something or
            // all do not", which is true of the question it was built for (a black frame)
            // and NOT of the question "which one draw painted over everything". MEASURED on
            // a baseball title: a pair whose eight draws total 1,520,090 samples hides whichever of
            // them is the 160,000-sample one, and reading the per-pair list named the WRONG
            // draw - a suppression run then showed the artefact still there. One ranked line
            // per pass costs nothing and answers it outright.
            let mut top: Vec<&(usize, u64, u32, u64)> = rows.iter().filter(|r| r.3 > 0).collect();
            top.sort_by_key(|r| std::cmp::Reverse(r.3));
            let biggest: Vec<String> = top
                .iter()
                .take(8)
                .map(|(d, key, idx, n)| format!("#{d} {key:016x} {n} samples ({idx} idx)"))
                .collect();
            report_warn!(
                "gxm draw coverage: pass #{p}: {drawn} of {} draws covered any sample at all, {samples} samples in total. PAINTED: [{}]. COVERED NOTHING: [{}]. BIGGEST SINGLE DRAWS: [{}]. A pair in the second list rasterised no fragment - its geometry, viewport, scissor or depth test threw it away, and no amount of looking at its shader will explain the black; a pair in the first list DID write pixels, so if the frame is still black the colour it wrote is the question.",
                rows.len(),
                fmt(&painted, "samples"),
                fmt(&empty, "indices thrown away"),
                biggest.join(", "),
            );
        }
    }
}
