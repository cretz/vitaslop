//! A serial guest-callback queue: the display queue's callback thread and the NGS
//! synthesizer's callback thread are the same machine.
//!
//! Each has one dedicated guest stack (one callback runs at a time, so one stack serves every
//! run), a ring of guest-memory slots the callback's argument is copied into (so the caller can
//! reuse its own buffer immediately), a FIFO of runs waiting their turn, and the one in flight.
//! [`CallbackQueue`] owns all of that; `VitaState` decides what a run carries, spawns the guest
//! thread for it, and reports.
//!
//! Two smaller delivery shapes live beside it: [`NgsSyncStash`], the NGS callbacks a
//! `sceAudioOutOutput` runs on its own calling thread, and [`OneShotCallback`], a service-state
//! callback that is delivered exactly once.

use std::collections::{HashMap, VecDeque};

/// One serial guest-callback queue. `T` is what one queued run carries. See the module docs.
pub(super) struct CallbackQueue<T> {
    /// The callback thread's guest stack, and the argument ring - 0 until first use, and the
    /// ring is `u32::MAX` once the arena refused them (the queue is then OFF for the run).
    stack: u32,
    slots: u32,
    next_slot: u32,
    queue: VecDeque<T>,
    running: Option<i32>,
    /// Threads parked until the queue is empty with nothing running - see
    /// [`CallbackQueue::park_until_idle`].
    idle_waiters: Vec<i32>,
    /// Whether the backlog-deeper-than-the-ring report has been made (it is made once).
    overrun_reported: bool,
}

impl<T> Default for CallbackQueue<T> {
    fn default() -> Self {
        CallbackQueue {
            stack: 0,
            slots: 0,
            next_slot: 0,
            queue: VecDeque::new(),
            running: None,
            idle_waiters: Vec::new(),
            overrun_reported: false,
        }
    }
}

impl<T> CallbackQueue<T> {
    /// Whether the queue is OFF: an earlier [`ensure`](Self::ensure) found the arena unable to
    /// spare its stack and ring.
    pub(super) fn is_off(&self) -> bool {
        self.slots == u32::MAX
    }

    /// Make sure the stack and a ring of `slot_count` slots of `slot_bytes` exist, allocating
    /// them through `galloc(size, align)` on first use and recording the stack's range with
    /// `note_stack(lo, hi)`. `false` when the queue is OFF - the arena could not spare them,
    /// now or on an earlier call.
    pub(super) fn ensure(
        &mut self,
        slot_bytes: u32,
        slot_count: u32,
        stack_bytes: u32,
        mut galloc: impl FnMut(u32, u32) -> u32,
        note_stack: impl FnOnce(u32, u32),
    ) -> bool {
        if self.slots == u32::MAX {
            return false;
        }
        if self.slots == 0 {
            self.slots = galloc(slot_bytes * slot_count, 8);
            self.stack = galloc(stack_bytes, 16);
            note_stack(self.stack, self.stack.wrapping_add(stack_bytes));
            if self.slots == 0 || self.stack == 0 {
                self.slots = u32::MAX;
                return false;
            }
        }
        true
    }

    /// The next argument slot of a ring of `slot_count` slots of `slot_bytes`, reused after
    /// `slot_count` more. Only valid after [`ensure`](Self::ensure) said yes.
    pub(super) fn take_slot(&mut self, slot_bytes: u32, slot_count: u32) -> u32 {
        let slot = self.slots + (self.next_slot % slot_count) * slot_bytes;
        self.next_slot = self.next_slot.wrapping_add(1);
        slot
    }

    pub(super) fn push(&mut self, run: T) {
        self.queue.push_back(run);
    }

    /// Runs waiting their turn (the one in flight not counted).
    pub(super) fn queued(&self) -> usize {
        self.queue.len()
    }

    pub(super) fn running(&self) -> Option<i32> {
        self.running
    }

    /// Whether a run is queued or in flight.
    pub(super) fn busy(&self) -> bool {
        self.running.is_some() || !self.queue.is_empty()
    }

    /// Whether this is the first time the backlog reached the ring's depth - the caller
    /// reports it then, and only then.
    pub(super) fn first_overrun(&mut self, slot_count: u32) -> bool {
        if self.queue.len() >= slot_count as usize && !self.overrun_reported {
            self.overrun_reported = true;
            return true;
        }
        false
    }

    /// The next run and the stack top to start it on, if nothing is in flight; `thid` - minted
    /// only when there is one to start - becomes the run in flight.
    pub(super) fn start_next(
        &mut self,
        stack_bytes: u32,
        mint: impl FnOnce() -> i32,
    ) -> Option<(T, u32, i32)> {
        if self.running.is_some() {
            return None;
        }
        let run = self.queue.pop_front()?;
        let thid = mint();
        self.running = Some(thid);
        Some((run, self.stack + stack_bytes, thid))
    }

    /// Thread `thid` ended: if it was the run in flight, nothing is any more. Whether it was.
    pub(super) fn finished(&mut self, thid: i32) -> bool {
        if self.running == Some(thid) {
            self.running = None;
            return true;
        }
        false
    }

    /// Record `thid` as parked until the queue drains.
    pub(super) fn park_until_idle(&mut self, thid: i32) {
        self.idle_waiters.push(thid);
    }

    /// Every thread parked until the queue drained, once it has - empty while it is busy.
    pub(super) fn take_idle_waiters(&mut self) -> Vec<i32> {
        if self.busy() {
            return Vec::new();
        }
        std::mem::take(&mut self.idle_waiters)
    }
}

/// Per thread: the NGS callbacks its `sceAudioOutOutput` is running on it, as
/// `(entry, info slot)`, and the info ring those callbacks read their `SceNgsCallbackInfo`
/// from - see `VitaState::ngs_sync_stash`, which allocates the ring and runs the batches.
#[derive(Default)]
pub(super) struct NgsSyncStash {
    batches: HashMap<i32, Vec<(u32, u32)>>,
    /// The ring's base: 0 until first use, `u32::MAX` once the arena refused it.
    slots: u32,
    next: u32,
}

impl NgsSyncStash {
    /// Whether the ring has not been asked for yet.
    pub(super) fn needs_ring(&self) -> bool {
        self.slots == 0
    }

    /// Install the ring the arena handed out; 0 (refused) turns the stash off for the run.
    pub(super) fn set_ring(&mut self, base: u32) {
        self.slots = if base == 0 { u32::MAX } else { base };
    }

    /// Whether the arena refused the ring.
    pub(super) fn is_off(&self) -> bool {
        self.slots == u32::MAX
    }

    /// The next info slot of a ring of `count` slots `bytes` long each.
    pub(super) fn take_slot(&mut self, bytes: u32, count: u32) -> u32 {
        let slot = self.slots + (self.next % count) * bytes;
        self.next = self.next.wrapping_add(1);
        slot
    }

    pub(super) fn stash(&mut self, thid: i32, batch: Vec<(u32, u32)>) {
        self.batches.insert(thid, batch);
    }

    pub(super) fn is_replaying(&self, thid: i32) -> bool {
        self.batches.contains_key(&thid)
    }

    pub(super) fn batch(&self, thid: i32) -> Option<Vec<(u32, u32)>> {
        self.batches.get(&thid).cloned()
    }

    pub(super) fn finish(&mut self, thid: i32) {
        self.batches.remove(&thid);
    }
}

/// A registered service-state callback that is delivered exactly once: `Some((entry,
/// userdata))` once registered, and `delivered` once it has been run. A re-registration
/// arms it again.
#[derive(Default)]
pub(super) struct OneShotCallback {
    registered: Option<(u32, u32)>,
    delivered: bool,
}

impl OneShotCallback {
    /// Register `entry` (0 clears it) and arm it for one delivery.
    pub(super) fn register(&mut self, entry: u32, userdata: u32) {
        self.registered = (entry != 0).then_some((entry, userdata));
        self.delivered = false;
    }

    /// The registered `(entry, userdata)`, if it has not been delivered yet.
    pub(super) fn undelivered(&self) -> Option<(u32, u32)> {
        if self.delivered {
            None
        } else {
            self.registered
        }
    }

    pub(super) fn mark_delivered(&mut self) {
        self.delivered = true;
    }
}
