use super::*;

impl VitaState {
    /// What thread `thid` is parked on, as one line, or `RUNNABLE` if it appears in no
    /// waiter list at all. "Runnable" is the interesting answer as often as the blocked
    /// ones are: a title that renders but never progresses has one thread spinning on a
    /// flag in pure guest compute, and that thread is exactly the one nothing here
    /// mentions.
    pub fn thread_wait_state(&self, thid: i32) -> String {
        if let Some(state) = self.sync.wait_state(thid) {
            return state;
        }
        if let Some(deadline) = self.storage.deadline_of(thid) {
            return format!("awaiting storage io_us={deadline} (now {})", self.storage.now());
        }
        if let Some((_, deadline)) = self.sleep_waiters.iter().find(|(t, _)| *t == thid) {
            return format!("sleeping until_us={deadline} (now {})", self.virtual_us);
        }
        if let Some((_, target, _)) = self.join_waiters.iter().find(|(t, _, _)| *t == thid) {
            return format!("joining thid={target:#x}");
        }
        // `sceKernelWaitSignal` is UNTIMED: it buys no idle clock, so
        // [`idle_attribution`](Self::idle_attribution) cannot see it, and a thread parked here
        // read as RUNNABLE in this dump - the one report whose job is to say which thread is
        // stopped. A wait with no deadline is exactly the shape that hangs forever, so it is
        // the last one that should be invisible.
        if self.signal_waiters.contains(&thid) {
            return "blocked on sceKernelWaitSignal (UNTIMED - only another thread can wake it)"
                .to_string();
        }
        "RUNNABLE".to_string()
    }

    /// Every LIVE thread that is parked, with the wait it is parked in - the report a stall
    /// needs, small enough for a phone's diagnostics panel.
    ///
    /// # Why this exists next to [`debug_sync_dump`](Self::debug_sync_dump)
    /// That dump prints every primitive's whole waiter list and runs to hundreds of lines: it
    /// is a file to read, and the browser panel keeps a bounded number of DISTINCT lines
    /// (`vitaslop_web::logging::PAGE_LOG_CAP`), so pasting it there would evict every other
    /// finding. This is the same question asked the other way round - per THREAD rather than
    /// per object - and answers "who is stopped, and on what" in as many lines as there are
    /// parked threads.
    ///
    /// Runnable threads are omitted deliberately: on a healthy frame nearly every thread is
    /// parked in something ordinary, and it is the empty-vs-not that matters when a title
    /// stops making progress. A thread absent from BOTH this list and the runnable set does
    /// not exist; a thread that is runnable and getting nowhere is spinning in guest code,
    /// which is a different instrument's question (`VITASLOP_BLOCK_HIST`).
    pub fn blocked_threads(&self) -> Vec<(i32, String, String)> {
        let mut out = Vec::new();
        let mut push = |thid: i32, name: String, state: String| {
            if state != "RUNNABLE" {
                out.push((thid, name, state));
            }
        };
        push(
            MAIN_THID,
            "main".to_string(),
            self.thread_wait_state(MAIN_THID),
        );
        for t in &self.threads {
            if t.exit_code.is_some() {
                continue;
            }
            let name = if t.name.is_empty() {
                "-".to_string()
            } else {
                t.name.clone()
            };
            // A thread the guest created but never started is not blocked on anything; saying
            // so is the difference between "waiting for a signal" and "never launched".
            let state = if t.started {
                self.thread_wait_state(t.uid)
            } else {
                "DORMANT (created, never started)".to_string()
            };
            push(t.uid, name, state);
        }
        out
    }

    /// Human-readable dump of every preemptive sync primitive's ownership/waiter state
    /// plus the sleep/timed-wait queues, for diagnosing a boot that stalls without
    /// reaching a frame. The decisive question it answers: is a given thread parked in
    /// some waiter list (blocked, and on what) or absent from all of them (running -
    /// i.e. spinning in pure guest compute)? Content-free (ids/addresses only).
    pub fn debug_sync_dump(&self) -> String {
        self.debug_sync_dump_with(None)
    }

    /// [`debug_sync_dump`](Self::debug_sync_dump), optionally able to read guest memory - see
    /// [`crate::ImportDispatch::sync_dump_with`] for why that changes what it can say.
    pub fn debug_sync_dump_with(&self, read_word: Option<&dyn Fn(u32) -> u32>) -> String {
        use std::fmt::Write;
        let mut s = String::new();
        let _ = writeln!(
            s,
            "current thread = {:#x}, virtual_us = {}",
            self.current, self.virtual_us
        );
        // The threads themselves come first: the primitive lists below say who is
        // waiting on each object, but only this says, for every live thread, whether
        // it is parked at all - and a thread absent from every waiter list is the
        // one actually running. A stall dump without it forces the reader to
        // cross-reference by hand and silently omits any thread blocked on a
        // primitive whose list is not printed.
        let _ = writeln!(
            s,
            "threads ({} live, {} parked fiber threads ended by a finalize, {} of those not \
             yet dropped by the scheduler):",
            1 + self
                .threads
                .iter()
                .filter(|t| t.exit_code.is_none())
                .count(),
            self.fibers.terminated(),
            self.pending_terminations.len()
        );
        // The MAIN thread is listed first and by hand, because it is the one thread
        // that has no `ThreadRec`: `create_thread` records the threads the GUEST
        // creates, and the initial thread was created by the loader before any of
        // that. Omitting it made this dump read as "the game logic thread is gone"
        // for a title whose main thread was in fact alive and spinning - a wrong
        // diagnosis that a stall dump exists precisely to prevent. Its thid is 0.
        let _ = writeln!(
            s,
            "  thid={:#x} \"main\" (the initial thread - no create record) {}",
            MAIN_THID,
            self.thread_wait_state(MAIN_THID),
        );
        for t in &self.threads {
            if t.exit_code.is_some() {
                continue;
            }
            let name = if t.name.is_empty() {
                "-".to_string()
            } else {
                format!("{:?}", t.name)
            };
            let _ = writeln!(
                s,
                "  thid={:#x} {name} entry={:#010x} prio={:#x} {}{}",
                t.uid,
                t.entry,
                t.priority,
                if t.started { "" } else { "DORMANT " },
                self.thread_wait_state(t.uid),
            );
        }
        self.sync.dump_into(&mut s, read_word);
        let _ = writeln!(
            s,
            "sleep waiters (thid, deadline_us): {:x?}",
            self.sleep_waiters
        );
        s
    }
}
