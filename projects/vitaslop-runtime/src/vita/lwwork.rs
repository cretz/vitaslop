//! The lightweight-mutex state, laid out in the guest's own WORK AREA.
//!
//! # Why it lives in guest memory
//! A lightweight mutex has no kernel handle. Its state lives in a caller-provided
//! `SceKernelLwMutexWork` (`SceInt64 data[4]`, 32 bytes) that libc embeds directly in its
//! own structures, and on the device `sceKernelLockLwMutex` is a USERSPACE function: it
//! compare-and-swaps that work area and only enters the kernel on CONTENTION. Keeping the
//! ownership host-side instead was wrong twice over:
//!
//!   1. **It was not faithful.** A host-side record keyed by the work-area ADDRESS cannot
//!      follow a byte copy of the work area, which is the exact antipattern that produced
//!      the "wait on an uncreated LwCond" deadlock - fixed for the cond, still open here
//!      until now. See `vitaslop-host-call-reference-semantics`.
//!   2. **A lock whose state lives on the host cannot be inlined.** With the state here,
//!      the uncontended take is a handful of wasm instructions and the boundary crossing
//!      is gone; contention still reaches the host, which is the only side that can park
//!      and wake a thread. Measured on a retail title after the GXM draw state was inlined, the
//!      lock/unlock pair was the largest single block of host calls left: 101,155 crossings
//!      in one profile window, 28,316 of each at one call site.
//!
//! Same move as [`super::gxmctx`], for the same two reasons in the same order.
//!
//! # Layout
//! Four words at the front of the 32-byte work area. They are load-bearing in two places -
//! the handlers in [`super::lwsync`] and the `InlineOp::LwMutexLock` form the transpiler
//! emits - so [`layout`] hands the SAME constants to the emitter rather than letting it
//! carry a second copy.
//!
//! # What the host still owns
//! The parked-waiter QUEUE: a list of thread ids in arrival order, which is not a thing
//! guest memory can hold usefully. [`off::WAITERS`] is its LENGTH, written by the host and
//! never read by it - it exists so guest code can tell "nobody is parked" (the case it may
//! serve itself) from "somebody is" (the case only the host can). Treat it as a projection
//! of the host's queue, not as a second home for it.
//!
//! # The lockCount argument is NOT modelled, here or before
//! `sceKernelLockLwMutex(work, lockCount, timeout)` can acquire a recursive mutex several
//! times at once. The handlers have always performed exactly one acquisition per call and
//! still do; what changed is only WHERE the count is kept. The fast paths below therefore
//! refuse any count but 1 and let the handler answer, so the two agree exactly wherever
//! both can run. Fixing the argument properly is a separate change with its own test.

use crate::host::GuestWords;
use vitaslop_transpiler::LwMutexLayout;

/// Byte offset of each word from the work-area pointer.
pub mod off {
    /// The work area's own address, stamped by [`super::init`] - the identity the kernel
    /// keeps INSIDE the work area, so a byte copy staged elsewhere can be resolved back to
    /// the original. A zero here means "never created", which is a real and different case.
    pub const ID: u32 = 0x00;
    /// The owning thread's SceUID. Meaningful only while [`COUNT`] is non-zero: thid 0 is
    /// the main thread by convention, so no owner value can mean "nobody" and freeness has
    /// to be read off the count instead.
    pub const OWNER: u32 = 0x04;
    /// Recursion depth. Zero means free.
    pub const COUNT: u32 = 0x08;
    /// How many threads the host has parked on this mutex. Non-zero sends every operation
    /// to the host, which is the only side that can wake one.
    pub const WAITERS: u32 = 0x0c;
}

/// Bytes the state occupies at the front of the work area.
pub const BYTES: u32 = 0x10;

/// >>> THE CONTENDED BIT, the top bit of the COUNT word: the host has a thread parked here.
///
/// Under the browser's parallel run the inline lock and unlock are COMPARE-AND-SWAPS of the
/// count word on several workers at once (`InlineOp::LwMutexLockSmp`), racing the host's
/// slow path. A waiter count in a SEPARATE word cannot be made safe against that without a
/// two-word handshake: an owner checking `waiters == 0` and then storing `count = 0` can
/// interleave with a contender that read the count as held and is about to park - and the
/// parked thread is never woken. With the bit IN the count word, the contender parks only if
/// its CAS from the held value it read succeeds, and the owner's release is a CAS from a
/// value WITHOUT the bit - so exactly one of them wins, and the loser goes to the host, which
/// settles it under its mutex. Set and cleared only by the host; [`count`] masks it.
/// [`off::WAITERS`] is still published for the single-baton inline form, which reads it.
pub const CONTENDED: u32 = 1 << 31;

/// The owner word of a free mutex. Every release to zero writes it BEFORE the count, so a
/// thread can never read its own id from a mutex someone else has just taken (the take writes
/// the owner AFTER its compare-and-swap of the count). Thread 0 is a real thread, so the old
/// "leave it stale" rule could not be kept once takes and releases run on several workers.
pub const NO_OWNER: u32 = u32::MAX;

/// Size of `SceKernelLwMutexWork` (`SceInt64 data[4]`), the storage the guest supplies.
pub const WORK_SIZE: u32 = 32;

/// The offsets, packaged for the transpiler. One definition, two readers.
pub fn layout() -> LwMutexLayout {
    LwMutexLayout { id: off::ID, owner: off::OWNER, count: off::COUNT, waiters: off::WAITERS }
}

fn get(w: &dyn GuestWords, work: u32, offset: u32) -> u32 {
    w.word(work.wrapping_add(offset))
}

fn set(w: &mut dyn GuestWords, work: u32, offset: u32, value: u32) {
    w.set_word(work.wrapping_add(offset), value);
}

/// Lay a free, unowned mutex out at `work` and stamp its identity.
///
/// Stamped LAST, so a partially written work area is never mistaken for a complete one -
/// and so a fast path that reached it mid-init would see `id != work` and defer to the
/// host rather than take a lock whose count word is still whatever the guest left there.
pub fn init(w: &mut dyn GuestWords, work: u32) {
    set(w, work, off::OWNER, NO_OWNER);
    set(w, work, off::COUNT, 0);
    set(w, work, off::WAITERS, 0);
    set(w, work, off::ID, work);
}

/// Forget the mutex at `work` (`sceKernelDeleteLwMutex`): clear the identity stamp, so a
/// stale work area cannot be taken inline by a guest that kept the pointer.
pub fn clear(w: &mut dyn GuestWords, work: u32) {
    set(w, work, off::ID, 0);
}

/// Whether the work area at `work` names ITSELF - i.e. this pointer is the canonical
/// mutex rather than a byte copy of one (a copy carries the original's id) or a work area
/// no create ever touched (zero).
pub fn is_mutex(w: &dyn GuestWords, work: u32) -> bool {
    work != 0 && get(w, work, off::ID) == work
}

/// The id this work area carries, whoever it belongs to. Non-zero means it was copied
/// from - or is - a created mutex.
pub fn carried_id(w: &dyn GuestWords, work: u32) -> u32 {
    get(w, work, off::ID)
}

/// The recursion depth; zero means free. The [`CONTENDED`] bit is not part of it.
pub fn count(w: &dyn GuestWords, work: u32) -> u32 {
    get(w, work, off::COUNT) & !CONTENDED
}

/// The count word as stored, [`CONTENDED`] bit and all - the value a compare-and-swap of it
/// has to expect.
pub fn count_word(w: &dyn GuestWords, work: u32) -> u32 {
    get(w, work, off::COUNT)
}

/// Take a FREE mutex for `thid`, atomically against the inline parallel form: a
/// compare-and-swap of the count word from the free value just read, then the owner.
/// `contended` states the [`CONTENDED`] bit to leave behind (the host's queue is not empty -
/// a barge past parked threads). Returns false if the mutex was not free when the swap ran;
/// the caller re-reads and decides again.
pub fn take_free(w: &mut dyn GuestWords, work: u32, thid: i32, contended: bool) -> bool {
    let raw = get(w, work, off::COUNT);
    if raw & !CONTENDED != 0 {
        return false;
    }
    let new = 1 | if contended { CONTENDED } else { 0 };
    if w.cas_word(work.wrapping_add(off::COUNT), raw, new) != raw {
        return false;
    }
    set(w, work, off::OWNER, thid as u32);
    true
}

/// Set the recursion depth of a mutex held by a thread that cannot be running guest code
/// right now (the caller of this host call, or a parked thread being handed it), with the
/// [`CONTENDED`] bit stated. A plain store is exact here: only the owner changes a held count,
/// and the owner is not running.
pub fn set_held(w: &mut dyn GuestWords, work: u32, held: u32, contended: bool) {
    set(w, work, off::COUNT, held | if contended { CONTENDED } else { 0 });
}

/// Mark a HELD mutex contended, if its count word still reads `raw` (a held value without
/// the bit, as just read). False means it changed - released, or the owner recursed - and the
/// caller must decide again rather than park: parking on a mutex that was released in
/// between would wait for a release that already happened.
pub fn mark_contended(w: &mut dyn GuestWords, work: u32, raw: u32) -> bool {
    w.cas_word(work.wrapping_add(off::COUNT), raw, raw | CONTENDED) == raw
}

/// Release fully: the owner word first ([`NO_OWNER`]), then the count - see [`NO_OWNER`].
pub fn release(w: &mut dyn GuestWords, work: u32) {
    set(w, work, off::OWNER, NO_OWNER);
    set(w, work, off::COUNT, 0);
}

/// The owning thread, meaningful only while [`count`] is non-zero.
pub fn owner(w: &dyn GuestWords, work: u32) -> i32 {
    get(w, work, off::OWNER) as i32
}

/// Publish how many threads the host has parked on this mutex. Written by the host from
/// its own queue; the host never reads it back.
pub fn set_waiters(w: &mut dyn GuestWords, work: u32, n: usize) {
    set(w, work, off::WAITERS, n as u32);
}

/// HAND a mutex the releasing caller still holds to the parked thread `thid`, at depth 1.
/// The count never passes through zero, so no take can slip in between: the owner word
/// changes first, then the count (with the [`CONTENDED`] bit as the queue now stands).
pub fn hand_to(w: &mut dyn GuestWords, work: u32, thid: i32, contended: bool) {
    set(w, work, off::OWNER, thid as u32);
    set_held(w, work, 1, contended);
}

/// Take the mutex without the host's help, if it is uncontended and this pointer is the
/// canonical work area. Returns whether it was taken.
///
/// **This is the definition the emitted `InlineOp::LwMutexLock` is held against**, term
/// for term - see `lwmutex_fast_paths_match_the_emitted_form`. The handler calls it first
/// for exactly that reason: the inline and host paths then take the same decision from the
/// same words, and the only thing the host adds is what to do when the answer is no.
pub fn fast_lock(w: &mut dyn GuestWords, work: u32, thid: i32, lock_count: u32) -> bool {
    if lock_count != 1 || !is_mutex(w, work) || get(w, work, off::WAITERS) != 0 {
        return false;
    }
    let held = get(w, work, off::COUNT);
    if held & CONTENDED != 0 {
        return false;
    }
    if held == 0 {
        // A compare-and-swap, so the host's take is atomic against the inline parallel form
        // (a single-baton backing's swap is a plain read and write - the same thing there).
        return take_free(w, work, thid, false);
    }
    if get(w, work, off::OWNER) != thid as u32 {
        return false;
    }
    // Recursion: only the owner changes a held count, and the owner is the caller.
    set(w, work, off::COUNT, held + 1);
    true
}

/// Release a mutex this thread holds, if nothing is parked on it. Returns whether it was
/// released. The mirror of [`fast_lock`], and the definition
/// `InlineOp::LwMutexUnlock` is held against.
///
/// `owner` is deliberately left alone when the count reaches zero: every reader tests the
/// count first, so a stale owner is unobservable, and clearing it would need a sentinel
/// that thid 0 already rules out.
pub fn fast_unlock(w: &mut dyn GuestWords, work: u32, thid: i32, unlock_count: u32) -> bool {
    if unlock_count != 1 || !is_mutex(w, work) || get(w, work, off::WAITERS) != 0 {
        return false;
    }
    let held = get(w, work, off::COUNT);
    if held == 0 || get(w, work, off::OWNER) != thid as u32 {
        return false;
    }
    set(w, work, off::COUNT, held - 1);
    true
}

#[cfg(test)]
mod tests {
    use super::*;
    use std::collections::BTreeMap;

    /// A sparse word-addressed memory, so a test can seed one word without an image.
    #[derive(Default)]
    struct Words(BTreeMap<u32, u32>);

    impl GuestWords for Words {
        fn word(&self, addr: u32) -> u32 {
            self.0.get(&addr).copied().unwrap_or(0)
        }
        fn set_word(&mut self, addr: u32, value: u32) {
            self.0.insert(addr, value);
        }
    }

    const WORK: u32 = 0x8000_1000;

    /// Every word of the state is claimed by exactly one field, with no gaps, and the whole
    /// of it fits the work area the guest is required to supply. A second field at one
    /// offset is a lock whose count word is somebody else's owner - deadlock, with nothing
    /// anywhere to say why.
    #[test]
    fn the_layout_is_closed_and_fits_the_work_area() {
        let mut claimed = [
            (off::ID, "id"),
            (off::OWNER, "owner"),
            (off::COUNT, "count"),
            (off::WAITERS, "waiters"),
        ];
        claimed.sort();
        for pair in claimed.windows(2) {
            assert_eq!(
                pair[1].0,
                pair[0].0 + 4,
                "{} at {:#x} and {} at {:#x} must be adjacent",
                pair[0].1,
                pair[0].0,
                pair[1].1,
                pair[1].0
            );
        }
        assert_eq!(claimed[0].0, 0, "the state starts at the work pointer");
        assert_eq!(claimed[3].0 + 4, BYTES, "BYTES must end just past the last word");
        assert!(BYTES <= WORK_SIZE, "{BYTES} bytes against a SceKernelLwMutexWork of {WORK_SIZE}");
        assert_eq!(layout().top() + 4, BYTES, "the emitter's pointer bound covers every word");
    }

    /// A created mutex identifies itself; an untouched work area does not, and neither does
    /// a byte COPY of one - which is the whole reason the id is inside the work area.
    #[test]
    fn only_the_canonical_work_area_names_itself() {
        let mut w = Words::default();
        assert!(!is_mutex(&w, WORK), "untouched memory is not a mutex");
        init(&mut w, WORK);
        assert!(is_mutex(&w, WORK));
        // A byte copy staged 0x100 further on carries the ORIGINAL's id.
        let copy = WORK + 0x100;
        for i in 0..BYTES / 4 {
            let v = w.word(WORK + i * 4);
            w.set_word(copy + i * 4, v);
        }
        assert!(!is_mutex(&w, copy), "a copy must not pass as the canonical mutex");
        assert_eq!(carried_id(&w, copy), WORK, "...but it names the one it came from");
        clear(&mut w, WORK);
        assert!(!is_mutex(&w, WORK), "a deleted mutex cannot be taken");
    }

    /// The uncontended take and release, including recursion, and the fact that the count
    /// is what says "free" - not the owner, which is left stale on purpose.
    #[test]
    fn the_fast_paths_take_release_and_recurse() {
        let mut w = Words::default();
        init(&mut w, WORK);
        assert!(fast_lock(&mut w, WORK, 7, 1), "a free mutex is taken");
        assert_eq!(count(&w, WORK), 1);
        assert_eq!(owner(&w, WORK), 7);
        assert!(fast_lock(&mut w, WORK, 7, 1), "the owner recurses");
        assert_eq!(count(&w, WORK), 2);
        assert!(!fast_lock(&mut w, WORK, 9, 1), "another thread must reach the host");
        assert_eq!(count(&w, WORK), 2, "and must not have changed anything");
        assert!(fast_unlock(&mut w, WORK, 7, 1));
        assert_eq!(count(&w, WORK), 1, "still held after one of two releases");
        assert!(!fast_unlock(&mut w, WORK, 9, 1), "a non-owner cannot release it");
        assert!(fast_unlock(&mut w, WORK, 7, 1));
        assert_eq!(count(&w, WORK), 0, "free");
        assert_eq!(owner(&w, WORK), 7, "the owner word is left stale, by design");
        assert!(!fast_unlock(&mut w, WORK, 7, 1), "releasing a free mutex is the handler's case");
        assert!(fast_lock(&mut w, WORK, 9, 1), "a free mutex goes to whoever asks, stale owner or not");
        assert_eq!(owner(&w, WORK), 9);
    }

    /// Every term of the guard refuses on its own. Each of these is a case where taking the
    /// lock inline would be silently wrong rather than visibly broken - a copy diverging
    /// from its original, a parked thread never woken, a multi-count acquire released early.
    #[test]
    fn each_guard_term_refuses_by_itself() {
        for (why, prepare, lock_count) in [
            ("a lockCount the handler defines", (|_: &mut Words| {}) as fn(&mut Words), 2u32),
            ("a zero lockCount", (|_: &mut Words| {}) as fn(&mut Words), 0),
            ("an unstamped work area", (|w: &mut Words| clear(w, WORK)) as fn(&mut Words), 1),
            (
                "a parked waiter",
                (|w: &mut Words| set_waiters(w, WORK, 1)) as fn(&mut Words),
                1,
            ),
        ] {
            let mut w = Words::default();
            init(&mut w, WORK);
            prepare(&mut w);
            assert!(!fast_lock(&mut w, WORK, 7, lock_count), "lock must refuse: {why}");
            assert_eq!(count(&w, WORK), 0, "and must not have written anything: {why}");

            // ...and the same for release, from a held mutex.
            let mut w = Words::default();
            init(&mut w, WORK);
            assert!(fast_lock(&mut w, WORK, 7, 1));
            prepare(&mut w);
            assert!(!fast_unlock(&mut w, WORK, 7, lock_count), "unlock must refuse: {why}");
            assert_eq!(count(&w, WORK), 1, "and must not have written anything: {why}");
        }
    }
}
