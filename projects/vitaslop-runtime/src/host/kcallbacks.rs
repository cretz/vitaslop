//! Kernel callback objects (`sceKernelCreateCallback` and friends) and the threads delivering
//! them.
//!
//! [`KernelCallbacks`] owns every callback and the record of which delivery thread is running
//! which one. Notification is pure bookkeeping here; delivery - spawning the guest call and
//! parking the owner until it returns - is `VitaState::kcb_deliver_one`'s, which asks this type
//! for the next pending callback. Vblank notifications are counted against the clock the
//! caller hands in. It needs nothing else from the host.

/// A kernel callback object (`sceKernelCreateCallback`). It belongs to the thread that
/// created it and runs ONLY on that thread's behalf, at a point the thread chooses
/// (`sceKernelCheckCallback`, or a `...CB` wait): `func(uid, notifyCount, notifyArg, common)`.
/// A notification only records - `count` accumulates and `notify_arg` is the latest - and a
/// delivery hands both over and clears them.
struct KCallback {
    uid: i32,
    owner: i32,
    name: String,
    func: u32,
    common: u32,
    count: u32,
    notify_arg: u32,
    /// `Some(virtual us of the last vblank accounted)` while registered with
    /// `sceDisplayRegisterVblankStartCallback`: every vblank edge since is one notification.
    vblank_since: Option<u64>,
}

/// One callback taken for delivery: `(uid, func, count, notify arg, common)`.
pub(super) type Delivery = (i32, u32, u32, u32, u32);

/// Every kernel callback and its delivery threads. See the module docs.
#[derive(Default)]
pub(super) struct KernelCallbacks {
    callbacks: Vec<KCallback>,
    /// `(callback thread, callback uid)` for every callback currently being delivered, so
    /// its return value can be applied when the delivery thread ends.
    delivering: Vec<(i32, i32)>,
}

impl KernelCallbacks {
    pub(super) fn create(&mut self, uid: i32, owner: i32, name: String, func: u32, common: u32) {
        self.callbacks.push(KCallback {
            uid,
            owner,
            name,
            func,
            common,
            count: 0,
            notify_arg: 0,
            vblank_since: None,
        });
    }

    pub(super) fn delete(&mut self, uid: i32) -> bool {
        let before = self.callbacks.len();
        self.callbacks.retain(|c| c.uid != uid);
        before != self.callbacks.len()
    }

    pub(super) fn notify(&mut self, uid: i32, arg: u32) -> bool {
        match self.callbacks.iter_mut().find(|c| c.uid == uid) {
            Some(c) => {
                c.count = c.count.saturating_add(1);
                c.notify_arg = arg;
                true
            }
            None => false,
        }
    }

    pub(super) fn cancel(&mut self, uid: i32) -> bool {
        match self.callbacks.iter_mut().find(|c| c.uid == uid) {
            Some(c) => {
                c.count = 0;
                c.notify_arg = 0;
                true
            }
            None => false,
        }
    }

    /// The pending notification count of `uid` at `now`, vblank notifications included.
    pub(super) fn count(&mut self, uid: i32, now: u64, period: u64) -> Option<u32> {
        self.account_vblanks(now, period);
        self.callbacks
            .iter()
            .find(|c| c.uid == uid)
            .map(|c| c.count)
    }

    /// Register `uid` for vblank notifications from `now` (`on`), or stop them.
    pub(super) fn set_vblank(&mut self, uid: i32, on: bool, now: u64) -> bool {
        match self.callbacks.iter_mut().find(|c| c.uid == uid) {
            Some(c) => {
                c.vblank_since = on.then_some(now);
                true
            }
            None => false,
        }
    }

    /// Turn the vblank edges that passed since each registered callback was last accounted
    /// into notifications. Vblanks are the virtual clock's grid of `period`, not events, so
    /// they are counted when a count is wanted.
    fn account_vblanks(&mut self, now: u64, period: u64) {
        for c in self.callbacks.iter_mut() {
            if let Some(since) = c.vblank_since {
                let edges = (now / period).saturating_sub(since / period);
                if edges > 0 {
                    c.count = c.count.saturating_add(edges as u32);
                    c.notify_arg = 0;
                    c.vblank_since = Some(now);
                }
            }
        }
    }

    /// Take ONE pending callback of thread `owner` at `now` for delivery, clearing its count:
    /// `None` when it has nothing pending.
    pub(super) fn take_pending(&mut self, owner: i32, now: u64, period: u64) -> Option<Delivery> {
        self.account_vblanks(now, period);
        let c = self
            .callbacks
            .iter_mut()
            .find(|c| c.owner == owner && c.count > 0)?;
        let delivery = (c.uid, c.func, c.count, c.notify_arg, c.common);
        tracing::trace!(target: "vitaslop::cb", uid = format_args!("{:#x}", c.uid), name = %c.name, count = c.count, "kernel callback delivered");
        c.count = 0;
        c.notify_arg = 0;
        Some(delivery)
    }

    /// Thread `thid` is delivering callback `uid`.
    pub(super) fn note_delivery(&mut self, thid: i32, uid: i32) {
        self.delivering.push((thid, uid));
    }

    /// Whether `thid` is a callback delivery thread.
    pub(super) fn is_delivery_thread(&self, thid: i32) -> bool {
        self.delivering.iter().any(|(t, _)| *t == thid)
    }

    /// Thread `thid` ended with `code`: if it was delivering a callback, a NON-ZERO return
    /// deletes that callback, which is the kernel's contract for a one-shot handler.
    pub(super) fn delivery_ended(&mut self, thid: i32, code: u32) {
        if let Some(pos) = self.delivering.iter().position(|(t, _)| *t == thid) {
            let (_, uid) = self.delivering.remove(pos);
            if code != 0 {
                self.callbacks.retain(|c| c.uid != uid);
            }
        }
    }
}
