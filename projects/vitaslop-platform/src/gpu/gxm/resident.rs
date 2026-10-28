//! The resident geometry heaps: geometry uploaded once and left on the GPU.

use super::*;

/// One bump-allocated GPU buffer holding geometry that has NOT CHANGED since the renderer
/// first saw it, uploaded once and left there.
///
/// # Why this exists: the largest steady-state item in a gameplay frame
/// The pass arenas are per FRAME. Every recompiled draw copies its repacked vertices and
/// its expanded indices into them, and the whole arena is `write_buffer`ed, every frame,
/// for every draw - including a track mesh that has been byte-identical since the race
/// started. MEASURED on a retail race, per frame: **4.75 ms copying 6.94
/// MB into the arenas** out of a 7.9 ms `prepare`, and 11.23 MB of buffer writes behind it.
/// That is the biggest thing the renderer does in a frame, and 93% of it (541 draws of 582)
/// is geometry it already had.
///
/// It is also a divergence from the device, not just a cost. GXM does not upload geometry:
/// the guest allocates its vertex and index buffers in shared memory and the GPU reads them
/// where they lie. Nothing on the hardware is proportional to a frame's vertex VOLUME. So
/// paying that per frame is the wrong SHAPE as well as the wrong price, which is the same
/// argument that moved shader compilation to the patcher call.
///
/// # What makes an address a sound key
/// An entry holds a strong `Arc` to the guest stream it was built from, so that allocation
/// cannot be freed while the entry lives and no later stream can be handed the same address.
/// `Arc<[u8]>` has no interior mutability, so the bytes behind a live address cannot change.
/// The lookup asserts it with `Arc::ptr_eq` anyway rather than trusting the argument: a
/// geometry cache that is wrong draws ANOTHER MESH, confidently, and nothing reports it.
/// This project has already paid for that once ([`GxpLive::packed`]).
///
/// # Why it never grows or resets mid-frame
/// A pass records into a command encoder that is submitted at the END of the chain, and a
/// prepared draw carries only an OFFSET - the buffer it addresses is read at encode time.
/// Recreating the buffer while a frame is in flight would silently re-point every draw
/// already prepared. So `place` only ever declines, and the grow or reset it asks for
/// happens at the top of the next `encode_chain`.
pub(super) struct ResidentHeap {
    pub(super) buf: Option<wgpu::Buffer>,
    cap: u64,
    used: u64,
    /// Every slice handed out: key -> (the guest allocation that owns it, offset, length,
    /// the [`Self::stamp`] of the last frame that bound it). Holding the `Arc` is what
    /// keeps that allocation's ADDRESS from being handed to a later buffer, which is the
    /// whole soundness argument for an address key. The stamp is what lets a full heap
    /// tell a LIVE slice from one belonging to a screen the title left behind.
    slices: HashMap<(u64, usize, usize), (std::sync::Arc<[u8]>, u64, u64, u64)>,
    /// Frame counter, bumped once per [`Self::grow_or_reset`] call (the frame boundary).
    stamp: u64,
    /// Set by a `place` that could not fit, cleared by the frame boundary that acts on it.
    want_grow: bool,
    /// Times this heap was RESET wholesale because it filled at its budget, and frames since
    /// the last one. A reset is not a fault - a title that reaches new geometry has to
    /// displace old - but a reset every few frames is thrash, and then the budget is the
    /// finding rather than the heap.
    resets: u64,
    frames_since_reset: u64,
    /// Bytes uploaded into this heap, ever. Against the per-frame arena volume it replaced.
    uploaded: u64,
    /// The buffer the last COMPACTION moved out of, kept (at the same `cap`) as the next
    /// compaction's destination instead of going to the graveyard. A buffer's FIRST touch
    /// is what stalls a browser for hundreds of milliseconds (seconds on the phone) - see
    /// `ensure_buffer` - and a heap at its budget compacts every few thousand frames in
    /// steady play, so without this every compaction paid a first touch of a full-budget
    /// buffer. Reusing it is safe for the same reason the compaction is: it runs at a frame
    /// boundary and its copies are submitted after every command that read the old one.
    spare: Option<wgpu::Buffer>,
}

impl ResidentHeap {
    pub(super) fn new() -> Self {
        ResidentHeap {
            spare: None,
            buf: None,
            cap: 0,
            used: 0,
            slices: HashMap::default(),
            stamp: 0,
            want_grow: false,
            resets: 0,
            frames_since_reset: 0,
            uploaded: 0,
        }
    }

    /// The slice `key` already owns, or `None`.
    ///
    /// A key whose stored `Arc` is not the caller's is treated as absent, never served. See
    /// the type's doc comment: that check is the whole soundness argument for an address key.
    pub(super) fn get(&mut self, key: &(u64, usize, usize), src: &std::sync::Arc<[u8]>) -> Option<(u64, u64)> {
        let stamp = self.stamp;
        let (stored, off, len, last_used) = self.slices.get_mut(key)?;
        if !std::sync::Arc::ptr_eq(stored, src) {
            return None;
        }
        *last_used = stamp;
        Some((*off, *len))
    }

    /// Copy `bytes` into the heap and remember them under `key`, or decline.
    ///
    /// Declining is not a failure - the caller falls back to the pass arena, which is what
    /// every draw did before this existed. It asks for a grow on the way out so the frame
    /// boundary can make room without moving a buffer a live command encoder names.
    #[allow(clippy::too_many_arguments)]
    pub(super) fn place(
        &mut self,
        queue: &wgpu::Queue,
        key: (u64, usize, usize),
        src: &std::sync::Arc<[u8]>,
        bytes: &[u8],
    ) -> Option<(u64, u64)> {
        let buf = self.buf.as_ref()?;
        let len = bytes.len() as u64;
        // `write_buffer` copies whole 4-byte units and every consumer of these ranges wants
        // 4-byte alignment (an index is four bytes; a packed vertex stride is a whole number
        // of f32s). Round the RESERVATION, never the data.
        let need = len.next_multiple_of(wgpu::COPY_BUFFER_ALIGNMENT).max(4);
        if self.used + need > self.cap {
            self.want_grow = true;
            return None;
        }
        let off = self.used;
        // The tail past `len` is reserved and never read, so a short write is fine - but
        // `write_buffer` itself needs a whole number of copy units, so pad the DATA when the
        // caller's slice is not one. Padding here rather than at every call site keeps the
        // alignment argument in one place.
        if len == need {
            queue.write_buffer(buf, off, bytes);
        } else {
            let mut padded = bytes.to_vec();
            padded.resize(need as usize, 0);
            queue.write_buffer(buf, off, &padded);
        }
        enc(&ENC.buffer_bytes, need);
        split_add(&PREP.resident_placed_bytes, need);
        self.uploaded += need;
        self.used += need;
        self.slices.insert(key, (src.clone(), off, len, self.stamp));
        Some((off, len))
    }

    /// Make room, at a FRAME BOUNDARY and nowhere else. Returns the buffer being replaced,
    /// for the caller's graveyard - dropping a `wgpu::Buffer` on the web backend only makes
    /// it collectable, and the last frame's commands still name it until its submit retires.
    /// How many slices the heap is holding - see `GxmRenderer::cache_sizes`. It is pruned
    /// only by a compaction or a reset, so on a long run it is one of the few numbers here
    /// that can climb without bound between them.
    pub(super) fn slice_count(&self) -> usize {
        self.slices.len()
    }

    /// A COUNT IS NOT A COST: the bytes the slices' `Arc`s keep alive on the RUST heap (each
    /// distinct allocation once), and the GPU buffer's used/cap. The first is not bounded by
    /// the second's budget in time - it only drops when the heap fills and compacts.
    pub(super) fn pinned_bytes(&self) -> (u64, u64, u64) {
        let mut seen: HashSet<usize> = HashSet::default();
        let pinned = self
            .slices
            .values()
            .filter(|(a, ..)| seen.insert(std::sync::Arc::as_ptr(a) as *const u8 as usize))
            .map(|(a, ..)| a.len() as u64)
            .sum();
        (pinned, self.used, self.cap)
    }

    /// >>> FORGET THE SLICES NO DRAW HAS BOUND FOR `idle` FRAMES - their `Arc`s are RUST
    /// >>> MEMORY, and nothing else ever releases it until the GPU heap fills.
    ///
    /// A slice pins the guest stream it was built from (see the type's doc comment), and
    /// the only thing that dropped a slice was a compaction or a reset - both triggered by
    /// the GPU buffer filling at its 128 MB budget. MEASURED in the desktop browser, a
    /// fighting title walking fights (`long1`): 35,703 vertex + 26,864 index slices at f19986 and
    /// no compaction yet, the Rust heap climbing 572 -> 899 MB at ~1 MB per second of play
    /// while every byte-budgeted cache sat under its budget. The snapshot caches had evicted
    /// those streams long before; the slices were what kept them allocated.
    ///
    /// Dropping the RECORD is sound at a frame boundary: a prepared draw carries an offset,
    /// not the record, and none is in flight here. Its GPU bytes become dead space that the
    /// next compaction drops, exactly like any other dead slice, and a mesh that returns is
    /// promoted and placed again. `VITASLOP_RESIDENT_IDLE_FRAMES=0` is the arm back.
    pub(super) fn forget_idle(&mut self, idle: u64) {
        if idle == 0 || !self.stamp.is_multiple_of(RESIDENT_SWEEP_EVERY) {
            return;
        }
        let keep_from = self.stamp.saturating_sub(idle);
        self.slices.retain(|_, (.., used)| *used >= keep_from);
    }

    pub(super) fn grow_or_reset(
        &mut self,
        device: &wgpu::Device,
        queue: &wgpu::Queue,
        budget: u64,
        usage: wgpu::BufferUsages,
        label: &str,
    ) -> Option<wgpu::Buffer> {
        self.stamp += 1;
        self.frames_since_reset += 1;
        self.forget_idle(resident_idle_frames());
        if !self.want_grow && self.buf.is_some() {
            return None;
        }
        self.want_grow = false;
        // The live set: every slice some draw bound in the last two frames.
        let keep_from = self.stamp.saturating_sub(2);
        let live: u64 = self
            .slices
            .values()
            .filter(|(_, _, _, used)| *used >= keep_from)
            .map(|(_, _, len, _)| len.next_multiple_of(wgpu::COPY_BUFFER_ALIGNMENT).max(4))
            .sum();
        // First use: start small and let the title's own working set size this.
        let want = if self.buf.is_none() {
            (self.cap.max(1024 * 1024)).min(budget)
        } else if self.cap < budget
            && live <= self.cap / 2
            && crate::gpu::memory_bound_on("VITASLOP_RESIDENT_COMPACT_EARLY")
        {
            // >>> A HEAP THAT FILLED WITH DEAD GEOMETRY COMPACTS; IT DOES NOT DOUBLE.
            // A bump allocator fills with meshes the title stopped drawing, so "full" says
            // nothing about the working set. Doubling on every fill took both heaps to their
            // 128 MB budget and then kept a same-size compaction spare beside each: MEASURED
            // in the desktop browser (a golf title, `long3` GPU census) 287 + 143 MB of
            // resident-heap buffers holding a live set of 15-25 MB. Grow only when the live
            // set itself needs the room.
            self.compact(device, queue, keep_from, label);
            return None;
        } else if self.cap < budget {
            (self.cap * 2).min(budget)
        } else {
            // Already at budget and still short. What fills a heap in practice is not the
            // frame's working set but the DEAD tail - meshes belonging to screens the
            // title left behind, held forever by a bump allocator that cannot free.
            // MEASURED in a browser on a retail sports title: the index heap filled its
            // 48 MB with 11,000+ meshes every ~1,400 frames and reset WHOLESALE, and every
            // mesh the frame actually used was re-uploaded over the following frames.
            //
            // So: COMPACT. Keep every slice some draw bound in the last two frames (the
            // live set), copy it into a fresh buffer GPU-side - this runs at a frame
            // boundary, so no prepared draw holds an offset into the old buffer, the same
            // safety argument the wholesale reset always relied on - and drop the rest.
            // The copies are submitted HERE, before the frame's own encoder: submission
            // order is execution order, and the old buffer's last reads were submitted
            // with the previous frame.
            //
            // If the live set alone is most of the budget, compaction buys nothing and the
            // heap is genuinely too small: reset wholesale and SAY so, exactly as before.
            if live >= self.cap / 4 * 3 {
                report_resident_heap_reset(label, self.cap, self.slices.len(), self.frames_since_reset);
                self.resets += 1;
                self.frames_since_reset = 0;
                self.used = 0;
                self.slices.clear();
                return None;
            }
            self.compact(device, queue, keep_from, label);
            return None;
        };
        enc_buffer_created();
        // COPY_SRC so a later compaction (above) can copy the live set out of this buffer.
        let t_create = Stopwatch::start();
        let new = device.create_buffer(&wgpu::BufferDescriptor {
            label: Some(label),
            size: want,
            usage: usage | wgpu::BufferUsages::COPY_DST | wgpu::BufferUsages::COPY_SRC,
            mapped_at_creation: false,
        });
        enc_create_us((t_create.ms() * 1000.0) as u64);
        self.cap = want;
        // The compaction spare is the OLD size now and can never be taken again - see
        // `compact`. Destroy it rather than leave it for a collector: its last use (a
        // compaction's copy) was submitted at an earlier frame boundary.
        if let Some(s) = self.spare.take() {
            s.destroy();
        }
        // >>> CARRY THE LIVE BYTES OVER ON THE GPU, AT THE SAME OFFSETS. The new buffer is
        // larger, so every slice fits where it was, and ONE copy of `[0, used)` keeps them
        // all valid. This used to clear every slice and let the draws re-place them, which
        // re-uploaded the title's whole resident working set from the CPU over the next
        // frames - a phone's worst present read `buffers 1.0 created, 13.81 MB written`
        // against ~1.9 MB normally, with `prepare` at 1,990 ms. Submitted here, before the
        // frame's own encoder, like the compaction above.
        if let Some(old_buf) = self.buf.as_ref().filter(|_| self.used > 0) {
            let mut encoder = device.create_command_encoder(&wgpu::CommandEncoderDescriptor {
                label: Some("resident-heap-grow"),
            });
            encoder.copy_buffer_to_buffer(old_buf, 0, &new, 0, self.used);
            queue.submit([encoder.finish()]);
        } else {
            self.used = 0;
            self.slices.clear();
        }
        self.buf.replace(new)
    }

    /// Copy the slices bound since `keep_from` to the front of a same-size buffer and drop
    /// the rest - at a frame boundary only, for the reasons in `grow_or_reset`.
    fn compact(&mut self, device: &wgpu::Device, queue: &wgpu::Queue, keep_from: u64, label: &str) {
        let usage = self.buf.as_ref().map_or(wgpu::BufferUsages::empty(), |b| b.usage());
        {
            let new = match self.spare.take() {
                Some(b) if b.size() == self.cap => b,
                _ => {
                    enc_buffer_created();
                    let t_create = Stopwatch::start();
                    let b = device.create_buffer(&wgpu::BufferDescriptor {
                        label: Some(label),
                        size: self.cap,
                        usage: usage | wgpu::BufferUsages::COPY_DST | wgpu::BufferUsages::COPY_SRC,
                        mapped_at_creation: false,
                    });
                    enc_create_us((t_create.ms() * 1000.0) as u64);
                    b
                }
            };
            let old = self.buf.replace(new);
            let (Some(old_buf), Some(new_buf)) = (old.as_ref(), self.buf.as_ref()) else {
                unreachable!("compaction only runs with a live buffer");
            };
            let mut encoder = device.create_command_encoder(&wgpu::CommandEncoderDescriptor {
                label: Some("resident-heap-compact"),
            });
            let before = self.slices.len();
            let mut off = 0u64;
            self.slices.retain(|_, (_, slice_off, len, used)| {
                if *used < keep_from {
                    return false;
                }
                let need = len.next_multiple_of(wgpu::COPY_BUFFER_ALIGNMENT).max(4);
                // The aligned tail is within the slice's own reservation - `place`
                // reserved (and wrote) whole copy units - so copying it reads nothing
                // that belongs to a neighbour.
                encoder.copy_buffer_to_buffer(old_buf, *slice_off, new_buf, off, need);
                *slice_off = off;
                off += need;
                true
            });
            queue.submit([encoder.finish()]);
            self.used = off;
            self.frames_since_reset = 0;
            tracing::debug!(
                target: "vitaslop::gxm",
                "resident geometry: the {label} heap filled at {:.1} MB and was COMPACTED - \
                 {} live meshes ({:.1} MB) copied in place on the GPU, {} dead ones dropped",
                self.cap as f64 / (1024.0 * 1024.0),
                self.slices.len(),
                off as f64 / (1024.0 * 1024.0),
                before - self.slices.len(),
            );
            // Kept as the next compaction's destination rather than retired - see `spare`.
            self.spare = old;
        }
    }
}

/// Say, once per heap, that a resident geometry heap FILLED at its budget and was dropped.
///
/// Unconditional, at `warn`, for the usual reason: this is the renderer throwing away work
/// it will have to redo, and the alternative to saying so is a frame that is mysteriously
/// slower than its neighbours with every counter reading healthy.
pub(super) fn report_resident_heap_reset(label: &str, cap: u64, entries: usize, frames: u64) {
    report_warn!(
        "resident geometry: the {label} heap filled at {:.1} MB ({entries} meshes) and was \
         reset after {frames} frames - every mesh it held is uploaded again as its draw comes \
         round. One reset as a title reaches new content is expected; one every few frames is \
         thrash, and then VITASLOP_RESIDENT_GEOM_MB is the number to change.",
        cap as f64 / (1024.0 * 1024.0)
    );
}
