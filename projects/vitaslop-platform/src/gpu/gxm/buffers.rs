use super::*;

/// >>> RETIRED GPU BUFFERS, KEPT AND HANDED BACK OUT INSTEAD OF FREED AND ALLOCATED AGAIN.
///
/// # The measurement this exists for
/// A `queue.write_buffer` on the user's phone blocked for **5,980 ms**, and a run of them
/// cost **25.7 s of wall clock in one session** while the whole browser froze. Everything a
/// stalling frame does was instrumented one at a time and all of it came back clean: an
/// EMPTY queue (0 submits in flight), no textures uploaded, `destroy()` at 0 ms, and on the
/// worst one no pipeline build either. One thing is present at every single stall and
/// nothing else is: the frame CREATED BUFFERS - 1, 17, 47 of them - and the block lands on
/// the first write that follows.
///
/// A GPU buffer is allocated lazily, so the first write into a fresh one pays for finding
/// its pages, in the GPU PROCESS, which is shared with every other tab - which is exactly
/// what a browser-wide hitch looks like and why the bytes in the blocked call are
/// irrelevant (one of them carried a single kilobyte).
///
/// A buffer taken from here has already been written to, so it is already backed. The pool
/// does not make allocation faster; it removes it.
///
/// # Why this is safe where a free list usually is not
/// Buffers arrive here through the SAME graveyard that used to destroy them - at the frame
/// boundary, after the submit that last named them has been made. That invariant is what
/// made `destroy()` correct there, and reuse needs exactly the same one, so nothing new is
/// being assumed.
///
/// # And why it is capped in BYTES
/// A cache capped in entries is not bounded in memory, which is how this project last put a
/// browser into an out-of-memory tab. The pool holds at most [`Self::CAP_BYTES`] and
/// destroys anything that does not fit, so the worst case is the behaviour it replaced.
/// Create a GPU buffer and ZERO IT ON THE QUEUE, so the first `write_buffer` into it is a
/// plain copy rather than the call that has to initialise it.
///
/// See [`BufferPool::create`] for the measurement, and `gpu::buffer_preinit` for the arm
/// that takes it off. Every buffer the renderer creates per frame goes through here, which
/// is what makes the A/B one knob rather than a survey of call sites.
pub(super) fn new_buffer(
    device: &wgpu::Device,
    queue: &wgpu::Queue,
    desc: &wgpu::BufferDescriptor,
) -> wgpu::Buffer {
    let _ = queue;
    let t_create = Stopwatch::start();
    let b = device.create_buffer(desc);
    enc_create_us((t_create.ms() * 1000.0) as u64);
    b
}

#[derive(Default)]
pub(super) struct BufferPool {
    /// Keyed by (size, usage bits): a reused buffer must be byte-for-byte substitutable for
    /// the one being asked for, and a buffer created without `INDEX` cannot be bound as one.
    pub(super) free: HashMap<(u64, u32), Vec<wgpu::Buffer>>,
    /// What the pool is holding, and what it has saved, for the panel.
    pub(super) bytes: u64,
    pub(super) reused: u64,
    pub(super) created: u64,
}

impl BufferPool {
    /// 64 MB. Large enough to hold a scene transition's arenas (the 47-buffer frame is
    /// about 12 MB of them), small enough that it is not a second heap.
    pub(super) const CAP_BYTES: u64 = 64 * 1024 * 1024;

    /// A buffer of exactly this size and usage, if one is waiting.
    /// What one pooled buffer is CHARGED against [`Self::CAP_BYTES`]: its size, but never less
    /// than [`Self::ENTRY_FLOOR`].
    ///
    /// >>> A BYTE CAP NEVER FIRES FOR TINY BUFFERS. Charged at their own size, 32-byte
    /// parameter blocks could fill the pool by the MILLION before reaching 64 MB, and every
    /// one is a separate GPU allocation, which a driver places at its own granularity (64 KB
    /// for a D3D12 placed buffer) rather than at 32 bytes. The floor makes the cap bound what
    /// the pool really holds.
    ///
    /// `VITASLOP_POOL_SMALL_BUFFERS=0` is the arm back (charged at their size, and the stencil
    /// row copy's parameter blocks created fresh - see `encode_row_copy`).
    fn charge(size: u64) -> u64 {
        if crate::gpu::memory_bound_on("VITASLOP_POOL_SMALL_BUFFERS") { size.max(Self::ENTRY_FLOOR) } else { size }
    }

    /// The smallest charge per pooled buffer - see [`Self::charge`].
    pub(super) const ENTRY_FLOOR: u64 = 64 * 1024;

    pub(super) fn take(&mut self, size: u64, usage: wgpu::BufferUsages) -> Option<wgpu::Buffer> {
        let v = self.free.get_mut(&(size, usage.bits()))?;
        let b = v.pop()?;
        self.bytes = self.bytes.saturating_sub(Self::charge(size));
        self.reused += 1;
        Some(b)
    }

    /// Keep a retired buffer, or destroy it if the pool is full.
    pub(super) fn put(&mut self, buf: wgpu::Buffer) {
        let size = buf.size();
        if self.bytes + Self::charge(size) > Self::CAP_BYTES {
            buf.destroy();
            return;
        }
        self.bytes += Self::charge(size);
        self.free.entry((size, buf.usage().bits())).or_default().push(buf);
    }

    /// Create through the pool: a hit is free, a miss is the allocation that was going to
    /// happen anyway, counted so the panel can say which it was.
    ///
    /// # >>> AND THE MISS IS ZERO-INITIALISED ON THE GPU BEFORE ANYONE WRITES IT
    /// `VITASLOP_GXM_BUFFER_PREINIT=0` takes this off, which is the A/B.
    ///
    /// WebGPU guarantees a new buffer reads as zero, so an implementation must clear one
    /// before its contents can be observed - and it defers that until first use. A
    /// `write_buffer` that fills only PART of a fresh buffer is exactly that first use, and
    /// it has to zero the rest before it can land. That is the only story left that fits
    /// every reading here: the blocked calls are the first write into a buffer created in
    /// that same frame (3 new buffers -> 286 ms, 1 new buffer -> 138 ms for 128 KB), while
    /// `create_buffer` itself times 0 ms, `destroy` times 0 ms, the queue is empty and the
    /// byte count of the blocked write predicts nothing.
    ///
    /// `clear_buffer` does the same zeroing as a GPU operation on the queue, where it costs
    /// a fill rather than a stall, and satisfies the initialisation so the write that
    /// follows is a plain copy. If the stalls survive this, zero-init is not the mechanism
    /// and the next suspect is the wire's own transfer buffer.
    pub(super) fn create(
        &mut self,
        device: &wgpu::Device,
        queue: &wgpu::Queue,
        desc: &wgpu::BufferDescriptor,
    ) -> wgpu::Buffer {
        if let Some(b) = self.take(desc.size, desc.usage) {
            return b;
        }
        self.created += 1;
        new_buffer(device, queue, desc)
    }
}
