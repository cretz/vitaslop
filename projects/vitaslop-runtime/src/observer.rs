//! A process-wide observer of every serviced host call - the one seam an outside driver of a
//! run (a playbook, see `vitaslop-playbook`) needs inside the engine.
//!
//! # Why a hook and not a dependency
//! The driver reacts to what the guest DOES: a call it makes, a value in its memory at the
//! moment it reads the pad. Both are only in hand at [`crate::host::VitaEnv`]'s dispatch, so
//! the engine has to call out from there. Calling a trait object keeps the dependency pointing
//! the right way - the driver depends on the runtime, never the reverse - which is what lets
//! the driver's crate also hold the tools that boot a title.
//!
//! # Cost
//! One relaxed atomic load per host call while nothing is enabled. The observer is installed
//! once per process (a `&'static`); a run turns it on and off with [`set_enabled`].
//!
//! # Threads
//! Under SMP the dispatch runs on whichever guest thread made the call, so an observer is
//! `Send + Sync` and does its own locking. It must not call back into the engine.
use std::sync::OnceLock;
use std::sync::atomic::{AtomicBool, Ordering};

use crate::guest_read::GuestRead;

/// One serviced host call, as the guest made it - before the handler runs, so the argument
/// registers are still the guest's own.
pub struct ObservedCall<'a> {
    /// The function NID ([`crate::nid::name`] names it).
    pub nid: u32,
    /// The guest's integer registers at the call (`r0..` are the arguments).
    pub regs: &'a [u32],
    /// Guest memory, by guest address.
    pub mem: &'a dyn GuestRead,
    /// Display flips completed so far.
    pub frame: u64,
    /// The guest's monotonic clock, in microseconds.
    pub guest_us: u64,
}

/// Something that watches every host call. See the module docs for the threading contract.
pub trait HostCallObserver: Send + Sync {
    fn on_call(&self, call: &ObservedCall<'_>);
}

static OBSERVER: OnceLock<&'static dyn HostCallObserver> = OnceLock::new();
static ENABLED: AtomicBool = AtomicBool::new(false);

/// Install the process's observer. The first install wins; a later one returns false and
/// changes nothing, so two drivers can never both believe they are watching.
pub fn install(observer: &'static dyn HostCallObserver) -> bool {
    OBSERVER.set(observer).is_ok()
}

/// Turn delivery on or off. Off by default; enabling with nothing installed delivers nothing.
pub fn set_enabled(on: bool) {
    ENABLED.store(on, Ordering::Release);
}

/// Whether calls are being delivered. The dispatch fast path.
#[inline]
pub fn enabled() -> bool {
    ENABLED.load(Ordering::Relaxed)
}

/// Deliver one call to the installed observer, if any.
pub fn notify(call: &ObservedCall<'_>) {
    if let Some(o) = OBSERVER.get() {
        o.on_call(call);
    }
}
