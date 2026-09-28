//! The KERNEL mutex's ownership state, laid out in a table inside the HOST MIRROR block.
//!
//! # Why it is not on the host any more
//! `sceKernelLockMutex`/`sceKernelUnlockMutex` are the largest block of host calls left in one
//! title's opening MOVIE, and they are there because the title runs two 8 kHz poll loops:
//!
//! ```text
//!   producer:  lock(m) ; signalCond(c) ; unlock(m)          158,345 of each / 800 frames
//!   consumer:  lock(m) ; unlock(m) ; delayThread(30 us)     156,308 of each / 800 frames
//! ```
//!
//! The 30 microseconds is the TITLE's own number (`VITASLOP_DELAY_CENSUS` reads it off the
//! call), so the iteration count is not something this engine can argue with - only the cost of
//! an iteration is. Three crossings per iteration at ~20 us each on the user's phone is most of
//! that phase's 45.8 ms frame; inlining the uncontended lock and unlock takes it to one.
//!
//! A lock whose state lives on the host cannot be inlined, for the reason
//! [`super::lwwork`] states at length: the emitted code cannot ask the host what it owns
//! without making the crossing the inlining exists to avoid. So the state moves to memory both
//! sides can reach, exactly as the lightweight mutex's did - and the layout is deliberately the
//! SAME four words, so the two emitted forms are one piece of code.
//!
//! # Why a TABLE, and not a work area
//! A lightweight mutex is named by a pointer to storage the guest supplies. A kernel mutex is
//! named by an `SceUID` and has no guest storage at all, so the emitted form has to turn an id
//! into an address with no host call: it indexes a fixed table by `uid & (ENTRIES - 1)`.
//!
//! UIDs are handed out sequentially across every kernel object, so two mutexes can collide.
//! That is handled by ENTRY OWNERSHIP rather than by hoping: an entry records the uid it
//! belongs to, the emitted form refuses any entry whose id is not the one in `r0`, and
//! [`crate::host::VitaState::create_mutex`] gives a colliding newcomer NO entry at all -
//! it keeps its state on the host and every operation on it takes the handler, which is
//! the behaviour every kernel mutex had before this existed.
//!
//! # What the host still owns
//! The parked-waiter QUEUE, for the same reason [`super::lwwork`] keeps it: a list of thread
//! ids in arrival order is not a thing this table can hold usefully, and only the host can wake
//! one. [`off::WAITERS`] is its LENGTH - written by the host, never read by it - so that guest
//! code can tell "nobody is parked" (the case it may serve itself) from "somebody is".

use crate::host::GuestWords;
use vitaslop_transpiler::LwMutexLayout;

/// Byte offset of each word from an entry's base. The SAME layout the lightweight mutex uses
/// ([`super::lwwork::off`]), so one emitted form serves both.
pub mod off {
    /// The `SceUID` this entry belongs to. Zero means the entry is free; any other value that
    /// is not the uid being operated on means the entry belongs to a DIFFERENT mutex that
    /// collided with it, and the operation belongs to the host.
    pub const ID: u32 = 0x00;
    /// The owning thread's SceUID. Meaningful only while [`COUNT`] is non-zero: thread id 0 is
    /// the main thread by convention, so no owner value can mean "nobody" and freeness has to
    /// be read off the count instead.
    pub const OWNER: u32 = 0x04;
    /// Recursion depth. Zero means free.
    pub const COUNT: u32 = 0x08;
    /// How many threads the host has parked on this mutex. Non-zero sends every operation to
    /// the host, which is the only side that can wake one.
    pub const WAITERS: u32 = 0x0c;
}

/// Bytes per entry.
pub const ENTRY_BYTES: u32 = 0x10;

/// Entries in the table. A power of two, because the emitted index is a mask rather than a
/// division - and generous, because a collision costs a mutex its inline form for the life of
/// the run and uids climb past a thousand on a title that creates threads as it loads.
pub const ENTRIES: u32 = 1024;

/// The table's own size in 32-bit mirror slots.
pub const SLOTS: u32 = ENTRIES * (ENTRY_BYTES / 4);

/// The offsets, packaged for the transpiler. One definition, two readers.
pub fn layout() -> LwMutexLayout {
    LwMutexLayout { id: off::ID, owner: off::OWNER, count: off::COUNT, waiters: off::WAITERS }
}

/// Which entry `uid` maps to. The emitted form computes exactly this.
pub fn index_of(uid: i32) -> u32 {
    (uid as u32) & (ENTRIES - 1)
}

/// The guest address of `uid`'s entry, given the table's base.
pub fn entry_addr(table: u32, uid: i32) -> u32 {
    table.wrapping_add(index_of(uid).wrapping_mul(ENTRY_BYTES))
}

fn get(w: &dyn GuestWords, entry: u32, offset: u32) -> u32 {
    w.word(entry.wrapping_add(offset))
}

fn set(w: &mut dyn GuestWords, entry: u32, offset: u32, value: u32) {
    w.set_word(entry.wrapping_add(offset), value);
}

/// Whether `uid`'s entry is the one that holds its state - it exists and names this uid.
///
/// False for a mutex that lost the entry to a collision and for a run with no table at all
/// (nothing linked an inline mutex form, so no block was reserved). Both cases keep their
/// state on the host, and this is the single question that decides which home to read.
pub fn owns_entry(w: &dyn GuestWords, table: u32, uid: i32) -> bool {
    table != 0 && get(w, entry_addr(table, uid), off::ID) == uid as u32
}

/// Claim `uid`'s entry, if it is free. Returns whether the claim succeeded; a `false` leaves
/// the mutex host-resident for its whole life.
///
/// There is no matching release, and that is not an omission: `sceKernelDeleteMutex` is a no-op
/// in this engine (see `super::sync::delete_object` - nothing destroys a kernel object), so an
/// entry is claimed once and holds its mutex's state for the run. A title that creates more
/// than [`ENTRIES`] kernel objects wraps the index and the later mutexes find their entries
/// taken; they fall back to the host, which is slower and correct.
pub fn claim(w: &mut dyn GuestWords, table: u32, uid: i32) -> bool {
    if table == 0 {
        return false;
    }
    let entry = entry_addr(table, uid);
    if get(w, entry, off::ID) != 0 {
        return false;
    }
    set(w, entry, off::ID, uid as u32);
    set(w, entry, off::OWNER, 0);
    set(w, entry, off::COUNT, 0);
    set(w, entry, off::WAITERS, 0);
    true
}

/// The recursion depth. Zero means free.
pub fn count(w: &dyn GuestWords, table: u32, uid: i32) -> i32 {
    get(w, entry_addr(table, uid), off::COUNT) as i32
}

pub fn set_count(w: &mut dyn GuestWords, table: u32, uid: i32, v: i32) {
    set(w, entry_addr(table, uid), off::COUNT, v as u32);
}

/// The owning thread, or `None` while the mutex is free. Read off the COUNT, never off the
/// owner word: thread id 0 is a real thread, so no owner value can mean "nobody".
pub fn owner(w: &dyn GuestWords, table: u32, uid: i32) -> Option<i32> {
    let entry = entry_addr(table, uid);
    (get(w, entry, off::COUNT) != 0).then(|| get(w, entry, off::OWNER) as i32)
}

pub fn set_owner(w: &mut dyn GuestWords, table: u32, uid: i32, thid: i32) {
    set(w, entry_addr(table, uid), off::OWNER, thid as u32);
}

/// Publish how many threads the host has parked on this mutex. Non-zero is what keeps every
/// operation on the host while anyone is waiting.
pub fn set_waiters(w: &mut dyn GuestWords, table: u32, uid: i32, n: usize) {
    set(w, entry_addr(table, uid), off::WAITERS, n as u32);
}

/// Take `uid`'s lock at its table `entry`, if nothing is parked on it and it is free or
/// already this thread's. Returns whether it was taken.
///
/// The DEFINITION `InlineOp::KernelMutexLock` is held against, and the twin of
/// [`super::lwwork::fast_lock`] with one rule changed: a lightweight work area names ITSELF
/// (`id == its own address`), while a table entry names the UID whose state it holds
/// (`id == uid`). That difference is the whole of what the collision check is.
///
/// Takes the entry ADDRESS rather than the table base, so a test can place one anywhere.
pub fn fast_lock(w: &mut dyn GuestWords, entry: u32, uid: i32, thid: i32, lock_count: u32) -> bool {
    if lock_count != 1
        || get(w, entry, off::ID) != uid as u32
        || get(w, entry, off::WAITERS) != 0
    {
        return false;
    }
    let held = get(w, entry, off::COUNT);
    if held != 0 && get(w, entry, off::OWNER) != thid as u32 {
        return false;
    }
    // Correct on both arms at once: `held + 1` takes a free mutex to 1 and a recursive one to
    // n+1, and re-writing an owner that already reads `thid` changes nothing.
    set(w, entry, off::OWNER, thid as u32);
    set(w, entry, off::COUNT, held + 1);
    true
}

/// Release a lock this thread holds at `entry`, if nothing is parked on it. The mirror of
/// [`fast_lock`] and the definition `InlineOp::KernelMutexUnlock` is held against.
///
/// `owner` is deliberately left alone when the count reaches zero: every reader tests the
/// count first, so a stale owner is unobservable, and clearing it would need a sentinel that
/// thread id 0 already rules out.
pub fn fast_unlock(
    w: &mut dyn GuestWords,
    entry: u32,
    uid: i32,
    thid: i32,
    unlock_count: u32,
) -> bool {
    if unlock_count != 1
        || get(w, entry, off::ID) != uid as u32
        || get(w, entry, off::WAITERS) != 0
    {
        return false;
    }
    let held = get(w, entry, off::COUNT);
    if held == 0 || get(w, entry, off::OWNER) != thid as u32 {
        return false;
    }
    set(w, entry, off::COUNT, held - 1);
    true
}
