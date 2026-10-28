//! The replay records behind `VitaState::call_guest`: a host call that calls back into the
//! title runs once per guest call it makes, each run answering the calls already made from its
//! record.
//!
//! [`GuestCalls`] owns every thread's stack of records and which one the running dispatch is
//! using; the replay contract itself is documented on `VitaState::call_guest`, which forwards
//! here with the calling thread. It needs nothing else from the host.

use super::{GuestCall, GuestCallResult, GuestMemory};
use crate::fasthash::FxHashMap;

/// One host call in progress that is making guest calls: what they answered so far, and the
/// one it is waiting on.
#[derive(Default)]
struct GuestCallFrame {
    /// Each finished call WITH the call it answers - a replay hands a result back only to the
    /// same call (see [`GuestCalls::call`]).
    results: Vec<(GuestCall, GuestCallResult)>,
    /// How many of `results` the running replay has consumed.
    cursor: usize,
    /// The call asked for and not yet run, with the region to capture when it returns.
    pending: Option<(GuestCall, Option<(u32, u32)>)>,
    /// Set when the pending call has returned: the next dispatch on this thread is the
    /// replay, and runs against this frame rather than pushing one.
    resume: bool,
}

/// Every thread's guest-call records. See the module docs.
#[derive(Default)]
pub(super) struct GuestCalls {
    /// Host calls in progress that are calling back into the title, by thread, innermost
    /// last. A stack because the guest function a call runs may itself make host calls that
    /// call back, and each needs a record of its own.
    frames: FxHashMap<i32, Vec<GuestCallFrame>>,
    /// The frame the RUNNING dispatch records into and replays from, as an index into its
    /// thread's stack; `None` until the handler first asks for a guest call. Set per
    /// dispatch by [`GuestCalls::begin`], under the host lock, so one field serves every
    /// thread.
    active: Option<usize>,
}

impl GuestCalls {
    /// `call` on behalf of the running dispatch on thread `thid`: its recorded result when
    /// this dispatch has had it run, else `None` with the call left pending.
    pub(super) fn call(
        &mut self,
        thid: i32,
        call: GuestCall,
        capture: Option<(u32, u32)>,
    ) -> Option<GuestCallResult> {
        let frames = self.frames.entry(thid).or_default();
        let idx = match self.active {
            Some(i) => i,
            None => {
                frames.push(GuestCallFrame::default());
                let i = frames.len() - 1;
                self.active = Some(i);
                i
            }
        };
        let frame = &mut frames[idx];
        // >>> A RESULT GOES BACK ONLY TO THE CALL IT ANSWERS, NOT TO WHOEVER ASKS NEXT.
        //
        // The replay is positional when the handler repeats itself exactly - but a handler may
        // NOT repeat itself: one that caches what an earlier pass read skips that read on the
        // next pass, and the call after it would then be handed the SKIPPED call's result.
        // MEASURED on a fighting title's intro movie: the audio look-ahead reads several samples per call,
        // the first was served from the read cache on the replay, and the second sample
        // received the first one's bytes - corrupt units, 4 decodes, a black silent movie.
        if frame.cursor < frame.results.len() && frame.results[frame.cursor].0 == call {
            frame.cursor += 1;
            return Some(frame.results[frame.cursor - 1].1.clone());
        }
        if let Some((_, r)) = frame.results.iter().find(|(c, _)| *c == call) {
            return Some(r.clone());
        }
        frame.pending = Some((call, capture));
        None
    }

    /// Whether the running dispatch on `thid` has asked for a guest call it does not have yet.
    pub(super) fn pending(&self, thid: i32) -> bool {
        let Some(i) = self.active else {
            return false;
        };
        self.frames
            .get(&thid)
            .and_then(|f| f.get(i))
            .is_some_and(|f| f.pending.is_some())
    }

    /// Before a dispatch on `thid`: pick the record it replays against - the innermost frame
    /// when that frame's call has just returned, none otherwise.
    pub(super) fn begin(&mut self, thid: i32) {
        self.active = None;
        if self.frames.is_empty() {
            return;
        }
        if let Some(frames) = self.frames.get_mut(&thid)
            && let Some(top) = frames.last_mut()
            && top.resume
        {
            top.resume = false;
            top.cursor = 0;
            self.active = Some(frames.len() - 1);
        }
    }

    /// After a dispatch on `thid`: whether it is waiting on a guest call. A dispatch that
    /// finished drops its record.
    pub(super) fn end(&mut self, thid: i32) -> bool {
        let Some(idx) = self.active.take() else {
            return false;
        };
        let Some(frames) = self.frames.get_mut(&thid) else {
            return false;
        };
        if frames.get(idx).is_some_and(|f| f.pending.is_some()) {
            return true;
        }
        frames.truncate(idx);
        if frames.is_empty() {
            self.frames.remove(&thid);
        }
        false
    }

    /// The guest call thread `thid`'s last dispatch is waiting on.
    pub(super) fn take(&mut self, thid: i32) -> Option<GuestCall> {
        let top = self.frames.get_mut(&thid)?.last_mut()?;
        top.pending.as_ref().map(|(c, _)| c.clone())
    }

    /// The guest call thread `thid` was waiting on has returned with `r0`/`r1`: record its
    /// result, with the capture it asked for read out of `mem` (rebased at `base`).
    pub(super) fn returned(
        &mut self,
        thid: i32,
        r0: u32,
        r1: u32,
        mem: &dyn GuestMemory,
        base: u32,
    ) {
        let Some(top) = self.frames.get_mut(&thid).and_then(|f| f.last_mut()) else {
            return;
        };
        let Some((call, capture)) = top.pending.take() else {
            return;
        };
        let bytes = match capture {
            Some((addr, len)) if len > 0 => {
                let mut b = vec![0u8; len as usize];
                mem.read(addr.wrapping_sub(base) as usize, &mut b);
                b
            }
            _ => Vec::new(),
        };
        top.results.push((call, GuestCallResult { r0, r1, bytes }));
        top.resume = true;
    }

    /// Whether the running dispatch is a REPLAY of a host call that already asked for one.
    pub(super) fn replaying(&self) -> bool {
        self.active.is_some()
    }

    /// Thread `thid` is gone: drop its records.
    pub(super) fn forget_thread(&mut self, thid: i32) {
        self.frames.remove(&thid);
    }

    /// Whether no thread has a record - a finished call leaves none.
    #[cfg(test)]
    pub(super) fn is_empty(&self) -> bool {
        self.frames.is_empty()
    }
}
