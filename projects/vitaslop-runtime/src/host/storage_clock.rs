//! The modelled storage device's own clock: the transfer time a title has accrued, the threads
//! parked paying it, and how much of a frame's worth of storage time the quanta have already
//! charged since the last flip.
//!
//! [`StorageClock`] owns all of it and reports which threads a step released; `VitaState` turns
//! those into wakes (`state_clock.rs`), because waking is the scheduler's business.

/// See the module docs.
#[derive(Default)]
pub(super) struct StorageClock {
    /// Accrued modelled storage-transfer time not yet paid by parking a thread.
    debt_us: u64,
    /// The device's clock, in microseconds. Deliberately NOT the game clock (`virtual_us`): a
    /// park charged against `virtual_us` livelocks, because that clock only advances on a
    /// display flip or on the scheduler's nothing-is-runnable idle path, and a title waiting
    /// for a load produces neither. This clock instead advances on any evidence that time
    /// passed at all - see [`StorageClock::advance`] - so a modelled transfer always completes.
    now_us: u64,
    /// `(thid, now_us deadline)` for each thread parked paying its storage-latency debt.
    waiters: Vec<(i32, u64)>,
    /// Storage time already charged by quantum boundaries since the last display flip, so
    /// the per-flip advance can top up to exactly one frame rather than double-count. See
    /// [`StorageClock::frame`].
    charged_since_flip: u64,
}

impl StorageClock {
    /// Add `us` to the accrued debt and return the new total.
    pub(super) fn add_debt(&mut self, us: u64) -> u64 {
        self.debt_us = self.debt_us.saturating_add(us);
        self.debt_us
    }

    pub(super) fn debt(&self) -> u64 {
        self.debt_us
    }

    /// Clear the accrued debt, returning it (the caller is about to park for it).
    pub(super) fn take_debt(&mut self) -> u64 {
        core::mem::take(&mut self.debt_us)
    }

    /// The device's clock now.
    pub(super) fn now(&self) -> u64 {
        self.now_us
    }

    /// Park `thid` until the device has spent `us` more on its transfer.
    pub(super) fn park(&mut self, thid: i32, us: u64) {
        let deadline = self.now_us.saturating_add(us);
        self.waiters.push((thid, deadline));
    }

    /// Advance the clock by `us`; returns, in park order, every thread whose transfer has now
    /// completed.
    pub(super) fn advance(&mut self, us: u64) -> Vec<i32> {
        self.now_us = self.now_us.saturating_add(us);
        let now = self.now_us;
        let mut woken = Vec::new();
        self.waiters.retain(|&(thid, deadline)| {
            if deadline <= now {
                woken.push(thid);
                false
            } else {
                true
            }
        });
        woken
    }

    /// One display flip: advance to one full frame of progress, minus what the quanta already
    /// charged since the last flip.
    pub(super) fn frame(&mut self, frame_us: u64) -> Vec<i32> {
        let owed = frame_us.saturating_sub(core::mem::take(&mut self.charged_since_flip));
        self.advance(owed)
    }

    /// Time passed without a flip (a quantum, an idle jump): advance, and remember it so the
    /// next flip nets it out.
    pub(super) fn charge(&mut self, us: u64) -> Vec<i32> {
        self.charged_since_flip = self.charged_since_flip.saturating_add(us);
        self.advance(us)
    }

    /// How much more time the EARLIEST outstanding transfer owes, `None` with nothing in flight.
    pub(super) fn earliest_remaining(&self) -> Option<u64> {
        self.waiters
            .iter()
            .map(|&(_, d)| d.saturating_sub(self.now_us))
            .min()
    }

    /// Complete the earliest outstanding transfer; `None` when nothing is in flight.
    pub(super) fn release_earliest(&mut self) -> Option<Vec<i32>> {
        let deadline = self.waiters.iter().map(|&(_, d)| d).min()?;
        Some(self.advance(deadline.saturating_sub(self.now_us)))
    }

    pub(super) fn has_waiters(&self) -> bool {
        !self.waiters.is_empty()
    }

    /// The deadline `thid` is parked until, if it is parked here.
    pub(super) fn deadline_of(&self, thid: i32) -> Option<u64> {
        self.waiters
            .iter()
            .find(|(t, _)| *t == thid)
            .map(|&(_, d)| d)
    }

    /// Drop `thid` from the waiters (it is going away).
    pub(super) fn forget(&mut self, thid: i32) {
        self.waiters.retain(|&(t, _)| t != thid);
    }
}
