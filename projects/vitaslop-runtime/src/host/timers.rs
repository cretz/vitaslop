//! The kernel's virtual timers (`sceKernelCreateTimer` and friends).
//!
//! A timer is a stopwatch over the same virtual clock every other guest-visible time call
//! reads, NOT a scheduled event source: the titles seen so far create one, start it and read
//! it back for their own profiling. Timer EVENTS (`sceKernelSetTimerEvent`, which wakes a
//! thread on a schedule) are deliberately not here - nothing has imported them, and a stub
//! that accepted one silently would be a timer that never fires.
//!
//! [`KernelTimers`] owns the timers outright; the clock reading each operation needs is handed
//! in as `now`. The `VitaState` methods the NID handlers call forward to it from
//! `state_kobjects.rs`. It needs nothing from the rest of the host.

/// A live virtual timer. It is a stopwatch, so what is stored is `accrued_us` (banked
/// by every stop) plus the clock reading it was last started at; the current count is
/// derived, never ticked. That is what keeps it exact under a scheduler that does not
/// run in real time - a timer that were incremented per frame would measure frames.
struct TimerRec {
    uid: i32,
    /// The name passed to `sceKernelCreateTimer`, which is the key `sceKernelOpenTimer`
    /// resolves - a lookup key, not just a label.
    name: String,
    attr: u32,
    running: bool,
    /// Microseconds banked by previous stops.
    accrued_us: u64,
    /// The clock reading at the last start; meaningless unless `running`.
    started_us: u64,
}

/// Every live kernel timer. See the module docs.
#[derive(Default)]
pub(super) struct KernelTimers {
    timers: Vec<TimerRec>,
}

impl KernelTimers {
    pub(super) fn create(&mut self, uid: i32, name: &str, attr: u32) {
        self.timers.push(TimerRec {
            uid,
            name: name.to_string(),
            attr,
            running: false,
            accrued_us: 0,
            started_us: 0,
        });
    }

    pub(super) fn by_name(&self, name: &str) -> Option<i32> {
        self.timers.iter().find(|t| t.name == name).map(|t| t.uid)
    }

    pub(super) fn start(&mut self, uid: i32, now: u64) -> Option<bool> {
        let t = self.timers.iter_mut().find(|t| t.uid == uid)?;
        let already = t.running;
        if !already {
            t.running = true;
            t.started_us = now;
        }
        Some(already)
    }

    pub(super) fn stop(&mut self, uid: i32, now: u64) -> Option<bool> {
        let t = self.timers.iter_mut().find(|t| t.uid == uid)?;
        let was_running = t.running;
        if was_running {
            t.accrued_us += now.saturating_sub(t.started_us);
            t.running = false;
        }
        Some(was_running)
    }

    pub(super) fn time_us(&self, uid: i32, now: u64) -> Option<u64> {
        let t = self.timers.iter().find(|t| t.uid == uid)?;
        Some(if t.running {
            t.accrued_us + now.saturating_sub(t.started_us)
        } else {
            t.accrued_us
        })
    }

    pub(super) fn set_time_us(&mut self, uid: i32, us: u64, now: u64) -> Option<u64> {
        let t = self.timers.iter_mut().find(|t| t.uid == uid)?;
        let before = if t.running {
            t.accrued_us + now.saturating_sub(t.started_us)
        } else {
            t.accrued_us
        };
        t.accrued_us = us;
        t.started_us = now;
        Some(before)
    }

    pub(super) fn delete(&mut self, uid: i32) -> bool {
        let n = self.timers.len();
        self.timers.retain(|t| t.uid != uid);
        self.timers.len() != n
    }

    pub(super) fn name(&self, uid: i32) -> &str {
        self.timers
            .iter()
            .find(|t| t.uid == uid)
            .map(|t| t.name.as_str())
            .unwrap_or("")
    }

    pub(super) fn attr(&self, uid: i32) -> u32 {
        self.timers
            .iter()
            .find(|t| t.uid == uid)
            .map(|t| t.attr)
            .unwrap_or(0)
    }

    pub(super) fn running(&self, uid: i32) -> bool {
        self.timers.iter().any(|t| t.uid == uid && t.running)
    }
}
