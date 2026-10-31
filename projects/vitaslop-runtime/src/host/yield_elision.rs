//! What decides whether a `sceKernelDelayThread(0)` yield has anyone to give the CPU to: the
//! runnable count at the pick, the run of yields elided since, and the host-mirror word that
//! lets the inlined yield answer the same question without a host call.
//!
//! [`YieldElision`] owns that state. `VitaState` (`state_clock.rs`) supplies the one input it
//! cannot see - whether this slice has woken or spawned anything - and keeps the knob and the
//! counters; see `VitaState::yield_would_repick_this_thread` for why the answer is exact.

/// See the module docs.
#[derive(Default)]
pub(super) struct YieldElision {
    /// How many OTHER threads were RUNNABLE at the pick that resumed the current one.
    runnable_others: usize,
    /// Consecutive yields elided since this thread was last picked - bounded by
    /// `VitaState::ELIDE_RUN_CAP`.
    elided_run: u32,
    /// Guest address of the host-mirror block, once the scheduler has one - see
    /// `VitaState::publish_yield_free_change` for the one word the host writes there between
    /// resumes.
    mirror_base: Option<u32>,
    /// Whether the mirror's `SLOT_YIELD_FREE` word was published as 1 at the last refresh and
    /// has not been cleared since.
    published: bool,
}

impl YieldElision {
    /// A pick: record the runnable set's size; a real suspend ends the elided run.
    #[inline]
    pub(super) fn note_pick(&mut self, runnable_others: usize) {
        self.runnable_others = runnable_others;
        self.elided_run = 0;
    }

    /// A host call made another thread runnable.
    #[inline]
    pub(super) fn note_woken(&mut self) {
        self.runnable_others = self.runnable_others.saturating_add(1);
    }

    /// Whether nobody was runnable at the pick and nothing has been woken since.
    #[inline]
    pub(super) fn alone(&self) -> bool {
        self.runnable_others == 0
    }

    #[inline]
    pub(super) fn elided_run(&self) -> u32 {
        self.elided_run
    }

    /// Take one more yield as elided, unless the run is at `cap` - which resets it and says
    /// the yield must really suspend.
    #[inline]
    pub(super) fn elide_within(&mut self, cap: u32) -> bool {
        if self.elided_run >= cap {
            self.elided_run = 0;
            return false;
        }
        self.elided_run += 1;
        true
    }

    /// The inlined yield reached the cap on its own: restart the host's run with it.
    #[inline]
    pub(super) fn reset_run(&mut self) {
        self.elided_run = 0;
    }

    #[inline]
    pub(super) fn mirror_base(&self) -> Option<u32> {
        self.mirror_base
    }

    #[inline]
    pub(super) fn set_mirror_base(&mut self, addr: u32) {
        self.mirror_base = Some(addr);
    }

    #[inline]
    pub(super) fn published(&self) -> bool {
        self.published
    }

    #[inline]
    pub(super) fn set_published(&mut self, free: bool) {
        self.published = free;
    }
}
