//! The display: what the guest told the display controller, and the model of the scanout its
//! flips are paced against.
//!
//! [`DisplayController`] holds the registers `sceDisplaySetFrameBuf` writes - the buffer
//! description, its geometry, the sync mode, and the vblank edge the named buffer latches on.
//! [`FlipPacer`] is the scanout's side of a flip: the frames the guest has queued and the panel
//! has not shown yet, the vblank floor the next latch may not precede, and the queue depth the
//! guest declared. Both own their fields; the `VitaState` methods in `state_display.rs` and
//! `state_clock.rs` hand them the clock and do the parking. Neither needs anything else from the
//! host.

use std::collections::VecDeque;

/// The first vblank edge at or after `t`, on a grid of `period` starting at 0.
pub(super) fn at_or_after_vblank(t: u64, period: u64) -> u64 {
    if period == 0 {
        return t;
    }
    t.div_ceil(period).saturating_mul(period)
}

/// The display controller's registers, as the guest last set them. See the module docs.
pub(super) struct DisplayController {
    /// The vblank edge the buffer named by the last `sceDisplaySetFrameBuf` latches on, until
    /// a `sceDisplayWaitSetFrameBuf*` takes it - see `VitaState::take_frame_buf_latch`.
    frame_buf_latch_us: Option<u64>,
    /// The `sync` argument of the most recent `sceDisplaySetFrameBuf`, which is what
    /// decides whether a present waits for the scanout. See `VitaState::set_display_sync`.
    sync: u32,
    /// Whether the guest has called `sceDisplaySetFrameBuf` at all, so the diagnostic can
    /// tell "asked for the default" from "never asked".
    sync_seen: bool,
    /// The `(width, height)` of the buffer the guest last handed `sceDisplaySetFrameBuf`.
    ///
    /// >>> A TITLE MAY PRESENT A BUFFER SMALLER THAN THE PANEL AND LET THE HARDWARE SCALE IT.
    /// > > > The display controller stretches whatever this struct describes onto the 960x544 panel,
    /// > > > so a front end that wants the frame rate more than the pixels renders 640x368 and
    /// > > > declares it. Composing that pass at the size it rasterised and stopping there puts the
    /// > > > whole screen in the top-left corner with black around it - MEASURED on one title's
    /// > > > menus. Every frontend that presents has to ask for this and scale.
    ///
    /// Starts at the panel size, which is what every title that renders today declares, so a
    /// run that never calls the setter behaves as it always did.
    size: (u32, u32),
    /// The whole `SceDisplayFrameBuf` the guest last handed `sceDisplaySetFrameBuf`, as
    /// its six words `{ size, base, pitch, fmt, width, height }`. Kept because
    /// `sceDisplayGetFrameBuf` reads it back, and the only source for what it should
    /// report is what the guest set: the fields it did not write (`pitch`, `fmt`) are
    /// nowhere else in this engine. `None` until the first set.
    frame_buf: Option<[u32; 6]>,
}

impl DisplayController {
    /// A controller before the guest has set anything: the panel's size and `sync`.
    pub(super) fn new(panel: (u32, u32), sync: u32) -> Self {
        DisplayController {
            frame_buf_latch_us: None,
            sync,
            sync_seen: false,
            size: panel,
            frame_buf: None,
        }
    }

    pub(super) fn set_frame_buf_latch(&mut self, latch: u64) {
        self.frame_buf_latch_us = Some(latch);
    }

    pub(super) fn take_frame_buf_latch(&mut self) -> Option<u64> {
        self.frame_buf_latch_us.take()
    }

    /// Record `(w, h)` unless either axis is zero - that describes no buffer.
    pub(super) fn set_geometry(&mut self, w: u32, h: u32) {
        if w != 0 && h != 0 {
            self.size = (w, h);
        }
    }

    pub(super) fn size(&self) -> (u32, u32) {
        self.size
    }

    pub(super) fn set_frame_buf(&mut self, fb: [u32; 6]) {
        self.frame_buf = Some(fb);
    }

    pub(super) fn frame_buf(&self) -> Option<[u32; 6]> {
        self.frame_buf
    }

    /// Record `sync`, returning `Some(first)` when it is worth reporting - the first call, or a
    /// change - and `None` for a repeat.
    pub(super) fn set_sync(&mut self, sync: u32) -> Option<bool> {
        if self.sync_seen && self.sync == sync {
            return None;
        }
        let first = !self.sync_seen;
        self.sync_seen = true;
        self.sync = sync;
        Some(first)
    }

    pub(super) fn sync(&self) -> u32 {
        self.sync
    }
}

/// The scanout's side of a flip. See the module docs and `VitaState::pace_flip`.
#[derive(Default)]
pub(super) struct FlipPacer {
    /// Display flips observed, for the calibration diagnostic beside the quantum count.
    flip_count: u64,
    /// Earliest game-clock time the display may latch the NEXT frame - the vsync floor.
    next_flip_us: u64,
    /// Latch times, in game-clock microseconds, of the frames the guest has queued and the
    /// scanout has not yet shown. The DEPTH of this is what a title is allowed to run ahead
    /// by.
    latches: VecDeque<u64>,
    /// `SceGxmInitializeParams::displayQueueMaxPendingCount`, as the guest declared it.
    /// Zero until `sceGxmInitialize` is seen, which [`FlipPacer::pace`] reads as one.
    queue_max_pending: u32,
    /// Whether a frame has been presented since the last `sceDisplayWaitSetFrameBuf`.
    present_since_wait: bool,
}

impl FlipPacer {
    pub(super) fn count_flip(&mut self) {
        self.flip_count = self.flip_count.saturating_add(1);
    }

    pub(super) fn flip_count(&self) -> u64 {
        self.flip_count
    }

    pub(super) fn set_queue_depth(&mut self, count: u32) {
        self.queue_max_pending = count;
    }

    /// The queue depth a flip is paced against: the guest's own, read as one before it says.
    pub(super) fn depth(&self) -> usize {
        self.queue_max_pending.clamp(1, 8) as usize
    }

    pub(super) fn note_present(&mut self) {
        self.present_since_wait = true;
    }

    pub(super) fn take_present_since_wait(&mut self) -> bool {
        core::mem::take(&mut self.present_since_wait)
    }

    /// Pace one flip at `now` on a vblank grid of `frame_us`, given `owed` frames the guest's
    /// own display callback still has to retire beyond the queue depth: returns how long the
    /// flipping thread must park, and books the frame's latch. See `VitaState::pace_flip`.
    pub(super) fn pace(&mut self, now: u64, frame_us: u64, owed: usize) -> u64 {
        // Retire the entries the scanout has already shown.
        while self.latches.front().is_some_and(|&t| t <= now) {
            self.latches.pop_front();
        }
        let depth = self.depth();
        let mut park = 0;
        if owed > 0 {
            park = park.max((owed as u64).saturating_mul(frame_us));
        }
        if self.latches.len() >= depth {
            // Full: wait for the oldest outstanding frame to reach the panel.
            if let Some(oldest) = self.latches.pop_front() {
                park = park.max(oldest.saturating_sub(now));
            }
        }
        // Where the clock will be when the thread resumes - the park is scheduled, not taken.
        let resumed = now.saturating_add(park);
        let latch = at_or_after_vblank(self.next_flip_us.max(resumed), frame_us);
        self.next_flip_us = latch.saturating_add(frame_us);
        self.latches.push_back(latch);
        park
    }

    /// The latch times of the frames queued and not yet shown, for the tests that pin them.
    #[cfg(test)]
    pub(super) fn latches(&self) -> Vec<u64> {
        self.latches.iter().copied().collect()
    }
}
