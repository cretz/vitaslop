//! GXM occlusion queries: where the GPU writes visibility results, and the counts a scene
//! accumulates for them.
//!
//! [`VisibilityQueries`] owns the buffer, the per-scene counts and each context's BACK-face
//! test. The FRONT-face test lives in the guest's context block (the inline forms write it),
//! so the caller reads it there and hands it in. It needs nothing else from the host.

use super::GuestCtx;
use crate::fasthash::FxHashMap;
use std::collections::BTreeMap;

/// Occlusion-query state. See the module docs.
#[derive(Default)]
pub(super) struct VisibilityQueries {
    /// `sceGxmSetVisibilityBuffer(context, bufferBase, stridePerCore)`: where the GPU
    /// writes occlusion-query results, and the per-core stride. 0 = no buffer bound.
    buffer: u32,
    stride: u32,
    /// Per visibility-test index, the sample count accumulated over the current scene
    /// (see [`VisibilityQueries::accumulate`]). Flushed into the visibility buffer when
    /// the scene ends, because that is when the GPU would have written it.
    counts: BTreeMap<u32, u32>,
    /// Per GXM context: the BACK-face visibility test `[enable, index, op]` - see
    /// `vita::gxm::set_back_visibility_test`.
    back: FxHashMap<u32, [u32; 3]>,
}

impl VisibilityQueries {
    pub(super) fn set_buffer(&mut self, base: u32, stride_per_core: u32) {
        self.buffer = base;
        self.stride = stride_per_core;
        self.counts.clear();
    }

    /// Set one field (0 enable, 1 index, 2 op) of `context`'s back-face test.
    pub(super) fn set_back_test(&mut self, context: u32, field: usize, value: u32) {
        self.back.entry(context).or_insert([0; 3])[field] = value;
    }

    /// One recorded draw of `index_count` on `context`, whose FRONT test is `front` as
    /// `(enabled, index)`: count it into the slot the live test names. See
    /// `VitaState::accumulate_visibility` for why this is a vertex count, deliberately.
    pub(super) fn accumulate(&mut self, context: u32, front: (bool, u32), index_count: u32) {
        // The capture does not rasterize, so it cannot tell a front face from a back one (see the
        // warning below): a draw counts into the FRONT test's slot when that test is on, else into
        // the BACK test's slot when only that one is - the same approximation, now with the slot
        // a back-only query asked for rather than nothing.
        let back = self.back.get(&context).copied().unwrap_or([0; 3]);
        let (front_on, front_index) = front;
        if (!front_on && back[0] == 0) || self.buffer == 0 {
            return;
        }
        static REPORTED: std::sync::atomic::AtomicBool = std::sync::atomic::AtomicBool::new(false);
        // A load first: a swap is a WRITE, on every draw while a query is live.
        if !REPORTED.load(std::sync::atomic::Ordering::Relaxed)
            && !REPORTED.swap(true, std::sync::atomic::Ordering::Relaxed)
        {
            // >>> STAYS AT `warn`. It was briefly moved to `debug` on the argument that a
            // deliberate decision is not a defect, and that argument is wrong: the OUTPUT is
            // still incorrect. Occluded geometry reports visible, guest code branches on it,
            // and a title can therefore behave differently here than on hardware. "We chose
            // this" describes why it has not been fixed, not whether it is broken - and a
            // warning that is quiet because the cause is known is a warning that has been
            // silenced. It fires ONCE per run, which is the right volume for a standing
            // incorrect result. Fix it (a real query plus a determinism story) or leave it
            // loud; do not lower it.
            tracing::warn!(
                target: "vitaslop::gxm",
                "an occlusion query is live; the capture does not rasterize, so the visibility \
                 buffer receives the SUBMITTED vertex count, not the count of samples that \
                 passed - occluded geometry still reports visible (e.g. a flare that should be \
                 hidden behind scenery is not). This is a DELIBERATE approximation: the result \
                 is read by guest code and branched on, so a real GPU query would make guest \
                 control flow depend on which adapter is present. See `accumulate_visibility`."
            );
        }
        let slot = if front_on { front_index } else { back[1] };
        *self.counts.entry(slot).or_insert(0) += index_count;
    }

    /// Write the scene's accumulated counts into the guest's visibility buffer and start the
    /// next scene from zero.
    pub(super) fn flush(&mut self, ctx: &mut GuestCtx) {
        if self.buffer == 0 {
            return;
        }
        for (&slot, &count) in &self.counts {
            ctx.write_u32(self.buffer.wrapping_add(slot * 4), count);
        }
        self.counts.clear();
    }
}

/// What the guest's render targets were created with. Owned here; `VitaState` forwards.
#[derive(Default)]
pub(super) struct RenderTargets {
    /// `SceGxmRenderTarget` handle -> the `driverMemBlock` UID its params carried, so
    /// `sceGxmRenderTargetGetDriverMemBlock` hands back exactly what the guest gave.
    mem_blocks: std::collections::HashMap<u32, u32>,
    /// `SceGxmRenderTarget` handle -> the `(width, height)` its params declared. This
    /// is the authoritative extent of every scene begun on that target: GXM
    /// rasterizes the render target's region, and the colour surface only describes
    /// where the pixels land (its data pointer, format and stride).
    extents: std::collections::HashMap<u32, (u32, u32)>,
}

impl RenderTargets {
    pub(super) fn set_mem_block(&mut self, render_target: u32, mem_block: u32) {
        self.mem_blocks.insert(render_target, mem_block);
    }

    /// The `driverMemBlock`, or `SCE_UID_INVALID_UID` (-1) when none was recorded.
    pub(super) fn mem_block(&self, render_target: u32) -> u32 {
        self.mem_blocks
            .get(&render_target)
            .copied()
            .unwrap_or(0xffff_ffff)
    }

    pub(super) fn set_extent(&mut self, render_target: u32, width: u32, height: u32) {
        self.extents.insert(render_target, (width, height));
    }

    pub(super) fn extent(&self, render_target: u32) -> Option<(u32, u32)> {
        self.extents.get(&render_target).copied()
    }
}
