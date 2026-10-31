//! `sceKernelRegisterThreadEventHandler` registrations and the bookkeeping that makes a
//! thread's EXIT handlers run exactly once.
//!
//! [`ThreadEvents`] owns the registrations, which threads' ends have been claimed, and whether
//! the initial thread is still in an early `module_start`. Whether a thread is the title's at
//! all is `VitaState`'s question, asked before it consults this. It needs nothing else from the
//! host.

use super::{
    GuestCall, MAIN_THID, THREAD_EVENT_ALL_IN_PROCESS, THREAD_EVENT_EXIT, ThreadEventHandler,
};

/// Thread-event registrations and exit claims. See the module docs.
#[derive(Default)]
pub(super) struct ThreadEvents {
    /// `sceKernelRegisterThreadEventHandler` registrations, in registration order.
    handlers: Vec<ThreadEventHandler>,
    /// Threads whose EXIT handlers have been handed out: each thread ends once, so a handler
    /// that itself calls `sceKernelExitThread`, or an exit followed by the engine's own
    /// end-of-thread, does not run them again. A restart is a new run that ends again, so it
    /// clears the thread's entry.
    exit_claimed: std::collections::HashSet<i32>,
    /// The initial thread is still running a module's `module_start` before the last one: an
    /// `sceKernelExitThread` there ends that initializer, not the thread, so it owes no EXIT
    /// handlers.
    main_initializing: bool,
}

impl ThreadEvents {
    pub(super) fn register(&mut self, uid: i32, target: i32, mask: u32, handler: u32, common: u32) {
        self.handlers.push(ThreadEventHandler {
            uid,
            target,
            mask,
            handler,
            common,
        });
    }

    pub(super) fn unregister(&mut self, uid: i32) -> bool {
        let before = self.handlers.len();
        self.handlers.retain(|h| h.uid != uid);
        self.handlers.len() != before
    }

    /// Thread `thid` is starting one of its entries; `last` when it is its final one.
    pub(super) fn note_entry(&mut self, thid: i32, last: bool) {
        if thid == MAIN_THID {
            self.main_initializing = !last;
        }
    }

    /// Claim thread `thid`'s END for its EXIT handlers: true the first time, false after - and
    /// false, claiming nothing, while the initial thread is in an early `module_start`.
    pub(super) fn claim_exit(&mut self, thid: i32) -> bool {
        if thid == MAIN_THID && self.main_initializing {
            return false;
        }
        self.exit_claimed.insert(thid)
    }

    /// Thread `thid` is starting a new run, which will end again.
    pub(super) fn forget_exit(&mut self, thid: i32) {
        self.exit_claimed.remove(&thid);
    }

    /// The EXIT handler calls owed thread `thid` ending with `code`, in registration order.
    /// The handler's id rides as a fifth (stack) argument the handler never reads, so two
    /// registrations of the same function with the same `common` are still two distinct
    /// calls - `VitaState::call_guest` matches a replayed call by its arguments.
    pub(super) fn exit_calls(&self, thid: i32, code: u32) -> Vec<GuestCall> {
        self.handlers
            .iter()
            .filter(|h| {
                h.mask & THREAD_EVENT_EXIT != 0
                    && (h.target == THREAD_EVENT_ALL_IN_PROCESS || h.target == thid)
            })
            .map(|h| GuestCall {
                entry: h.handler,
                args: vec![THREAD_EVENT_EXIT, thid as u32, code, h.common, h.uid as u32],
            })
            .collect()
    }
}
