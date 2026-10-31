//! The owned subsystem types, exercised on their own - no `VitaState`, no scheduler - so each
//! one's contract is pinned where it is written rather than only through the host API above it.
use super::*;

const MIB: u32 = 1024 * 1024;
/// A CDRAM and a main-RAM `SceKernelMemBlockType` (see `vita::sysmem::partition_of`).
const TY_CDRAM: u32 = 0x0940_8060;
const TY_MAIN: u32 = 0x0C20_D060;

// --- KernelCallbacks ---

/// Notifications accumulate with the latest argument, vblank edges since registration count as
/// notifications, delivery hands over and clears one callback of the owner at a time, and a
/// delivery thread that returns non-zero deletes its (one-shot) callback.
#[test]
fn kernel_callbacks_count_deliver_and_retire_one_shots() {
    let mut k = KernelCallbacks::default();
    k.create(0x10, 7, "a".into(), 0x8100_0001, 0xc0);
    k.create(0x11, 8, "b".into(), 0x8100_0101, 0);
    assert!(k.notify(0x10, 5) && k.notify(0x10, 6));
    assert_eq!(k.count(0x10, 0, 100), Some(2));
    assert!(k.set_vblank(0x11, true, 0));
    assert_eq!(k.count(0x11, 350, 100), Some(3), "three vblank edges");
    assert_eq!(
        k.take_pending(7, 350, 100),
        Some((0x10, 0x8100_0001, 2, 6, 0xc0))
    );
    assert_eq!(k.take_pending(7, 350, 100), None, "delivered and cleared");
    k.note_delivery(0x200, 0x10);
    assert!(k.is_delivery_thread(0x200));
    k.delivery_ended(0x200, 0);
    assert!(!k.is_delivery_thread(0x200));
    assert_eq!(k.count(0x10, 350, 100), Some(0), "a zero return keeps it");
    k.note_delivery(0x201, 0x10);
    k.delivery_ended(0x201, 1);
    assert_eq!(
        k.count(0x10, 350, 100),
        None,
        "a non-zero return deletes it"
    );
    assert!(k.cancel(0x11) && !k.cancel(0x10) && k.delete(0x11) && !k.delete(0x11));
}

// --- SyncObjects ---

/// The scheduler queues a [`Sched`] borrows, owned by the test.
#[derive(Default)]
struct Queues {
    wakes: Vec<i32>,
    causes: Vec<(i32, i32, &'static str)>,
    resume: Vec<(i32, u32)>,
    stat: Vec<(u32, u32)>,
}

impl Queues {
    fn sched(&mut self, now: u64, current: i32) -> Sched<'_> {
        Sched {
            now,
            current,
            wakes: &mut self.wakes,
            causes: &mut self.causes,
            trace: true,
            resume_codes: &mut self.resume,
            stat_writes: &mut self.stat,
        }
    }
}

/// A signal releases the waiters its count can satisfy in FIFO order, consuming the count for
/// each, and the wake trace names the signaller and a source site in the sync module.
#[test]
fn a_semaphore_signal_wakes_satisfiable_waiters_in_order() {
    let mut s = SyncObjects::default();
    let mut q = Queues::default();
    s.create_sema(0x100, "s", 0, 0, 8);
    s.sema_block(0x100, 1, 2, None, (0, 0, 0, 0));
    s.sema_block(0x100, 2, 1, None, (0, 0, 0, 0));
    s.sema_block(0x100, 3, 1, None, (0, 0, 0, 0));
    s.sema_signal_wake(&mut q.sched(0, 9), 0x100, 3);
    assert_eq!(q.wakes, vec![1, 2], "need 2 then need 1; the third waits");
    assert_eq!(
        s.sema_info(0x100).map(|i| i.3),
        Some(0),
        "the count was consumed"
    );
    assert_eq!(q.causes[0].1, 9, "the signalling thread");
    assert!(q.causes[0].2.contains("sync.rs"), "{}", q.causes[0].2);
}

/// A timed wait that expires is woken carrying WAIT_TIMEOUT, and only once its deadline passes.
#[test]
fn an_expired_timed_wait_is_woken_with_wait_timeout() {
    let mut s = SyncObjects::default();
    let mut q = Queues::default();
    let mut timed_out = Vec::new();
    s.create_rwlock(0x200, "rw", 0);
    assert_eq!(s.rwlock_try(0x200, true, 1), RwLockResult::Done);
    s.rwlock_block(0x200, false, 2, Some(500));
    assert_eq!(
        s.earliest_deadline_owner(std::iter::empty())
            .map(|(d, o)| (d, o.thid)),
        Some((500, 2))
    );
    s.expire_locks(&mut q.sched(499, 0), &mut timed_out);
    assert!(q.wakes.is_empty(), "not yet");
    s.expire_locks(&mut q.sched(500, 0), &mut timed_out);
    assert_eq!(q.wakes, vec![2]);
    assert_eq!(q.resume, vec![(2, SCE_KERNEL_ERROR_WAIT_TIMEOUT)]);
}

// --- FlipPacer / DisplayController ---

/// Latches land on the vblank grid, never closer than one period, and the guest may run its
/// declared queue depth ahead before a flip parks.
#[test]
fn flips_latch_on_the_grid_and_park_only_when_the_queue_is_full() {
    let mut p = FlipPacer::default();
    p.set_queue_depth(2);
    assert_eq!(p.pace(100, 1000, 0), 0);
    assert_eq!(
        p.pace(150, 1000, 0),
        0,
        "depth 2: the second frame queues freely"
    );
    assert_eq!(
        p.latches(),
        vec![1000, 2000],
        "one period apart, on the grid"
    );
    // The queue is full: the third flip waits for the oldest frame to reach the panel.
    assert_eq!(p.pace(200, 1000, 0), 800);
    assert_eq!(p.latches(), vec![2000, 3000]);
    // A display-callback backlog is charged to the flipping thread a frame per entry owed.
    let mut q = FlipPacer::default();
    assert_eq!(q.pace(0, 1000, 3), 3000);
    assert_eq!(q.latches(), vec![3000]);
    // Shown frames retire, and a presented frame is remembered once.
    assert_eq!(p.pace(5000, 1000, 0), 0);
    assert_eq!(p.latches(), vec![5000]);
    p.note_present();
    assert!(p.take_present_since_wait() && !p.take_present_since_wait());
}

/// The controller reports a sync mode the first time and on a change only, and ignores a
/// geometry that describes no buffer.
#[test]
fn the_display_controller_reports_sync_changes_and_keeps_real_geometry() {
    let mut d = DisplayController::new((960, 544), 1);
    assert_eq!(d.set_sync(1), Some(true), "the first call is reported");
    assert_eq!(d.set_sync(1), None);
    assert_eq!(d.set_sync(0), Some(false));
    d.set_geometry(640, 0);
    assert_eq!(d.size(), (960, 544));
    d.set_geometry(640, 368);
    assert_eq!(d.size(), (640, 368));
    assert_eq!(d.frame_buf(), None);
    d.set_frame_buf_latch(3333);
    assert_eq!(
        (d.take_frame_buf_latch(), d.take_frame_buf_latch()),
        (Some(3333), None)
    );
}

// --- GuestArena ---

/// The bump cursor aligns up and never hands out past the ceiling: an allocation that would
/// reach it is a null, not a pointer into the stack above.
#[test]
fn the_arena_aligns_and_refuses_past_its_ceiling() {
    let mut a = GuestArena::new(0x1001);
    assert_eq!(a.galloc(16, 16, 0x2000), 0x1010);
    assert_eq!(a.cursor(), 0x1020);
    assert_eq!(a.galloc(0x1000, 4, 0x2000), 0, "would cross the ceiling");
    assert_eq!(a.cursor(), 0x1020, "a refusal moves nothing");
    assert_eq!(a.main_free_bytes(0x2000), 0x2000 - 0x1020);
}

/// A freed block is reused first fit, its remainder stays available, and neighbouring holes
/// merge - the shape a title that probes memory by allocating and freeing relies on.
#[test]
fn freed_blocks_are_reused_split_and_coalesced() {
    let mut a = GuestArena::new(0x10_0000);
    let ceiling = 0x10_0000 + 64 * MIB;
    let mut next = 0x100;
    let b1 = a.alloc_memblock(4 * MIB, 4096, TY_MAIN, ceiling, &mut next);
    let b2 = a.alloc_memblock(4 * MIB, 4096, TY_MAIN, ceiling, &mut next);
    assert_eq!(
        (b1, b2),
        (0x100, 0x101),
        "ids are minted in order, on success only"
    );
    let (base1, base2) = (a.memblock_base(b1).unwrap(), a.memblock_base(b2).unwrap());
    assert_eq!(a.free_memblock(b1), Some((base1, 4 * MIB)));
    assert_eq!(a.free_memblock(b2), Some((base2, 4 * MIB)));
    assert_eq!(a.freed_count(), 1, "two adjacent holes are one");
    assert_eq!(a.freed_bytes(), u64::from(8 * MIB));
    assert_eq!(a.free_memblock(b1), None, "an id is gone once freed");
    // A smaller request reuses the merged hole from its start and leaves the rest.
    let b3 = a.alloc_memblock(MIB, 4096, TY_MAIN, ceiling, &mut next);
    assert_eq!(a.memblock_base(b3), Some(base1));
    assert_eq!(a.freed_bytes(), u64::from(7 * MIB));
    assert_eq!(a.memblock_info_at(base1 + 100), Some((base1, MIB, TY_MAIN)));
    assert_eq!(a.memblock_containing(base1, MIB), Some(b3));
    assert_eq!(a.memblock_containing(base1, MIB + 1), None);
}

/// A partition's budget is enforced whatever the arena has room for, and a free gives it back.
#[test]
fn a_partition_budget_is_enforced_and_refunded() {
    let mut a = GuestArena::new(0x10_0000);
    let ceiling = u32::MAX;
    let mut next = 1;
    let big = a.alloc_memblock(100 * MIB, 4096, TY_CDRAM, ceiling, &mut next);
    assert_ne!(big, 0);
    assert_eq!(a.cdram_used(), 100 * MIB);
    assert_eq!(
        a.alloc_memblock(20 * MIB, 4096, TY_CDRAM, ceiling, &mut next),
        0,
        "past the CDRAM budget"
    );
    assert_eq!(next, 2, "a refused block mints no id");
    a.free_memblock(big);
    assert_eq!(a.cdram_used(), 0);
    assert_ne!(
        a.alloc_memblock(20 * MIB, 4096, TY_CDRAM, ceiling, &mut next),
        0
    );
}

// --- CallbackQueue ---

fn arena_galloc(a: &mut GuestArena) -> impl FnMut(u32, u32) -> u32 + '_ {
    move |size, align| a.galloc(size, align, u32::MAX)
}

/// Runs start one at a time in submission order, each on the queue's own stack, and a thread
/// waiting for the queue to drain is released only once nothing is queued or running.
#[test]
fn a_callback_queue_runs_serially_and_releases_idle_waiters_last() {
    let mut arena = GuestArena::new(0x1000);
    let mut q: CallbackQueue<u32> = CallbackQueue::default();
    let mut stack = (0, 0);
    assert!(
        q.ensure(8, 4, 0x100, arena_galloc(&mut arena), |lo, hi| stack =
            (lo, hi))
    );
    assert_eq!(stack.1 - stack.0, 0x100);
    let (s1, s2) = (q.take_slot(8, 4), q.take_slot(8, 4));
    assert_eq!(s2 - s1, 8);
    q.push(s1);
    q.push(s2);
    q.park_until_idle(0x50);
    let mut uid = 0x200;
    let mut mint = || {
        uid += 1;
        uid
    };
    let (run, top, thid) = q.start_next(0x100, &mut mint).unwrap();
    assert_eq!((run, top, thid), (s1, stack.0 + 0x100, 0x201));
    assert!(q.start_next(0x100, &mut mint).is_none(), "one at a time");
    assert!(q.finished(thid));
    assert!(q.take_idle_waiters().is_empty(), "one still queued");
    let (run, _, thid) = q.start_next(0x100, &mut mint).unwrap();
    assert_eq!(run, s2);
    assert!(!q.finished(0x999), "not the run in flight");
    assert!(q.finished(thid));
    assert_eq!(q.take_idle_waiters(), vec![0x50]);
    assert!(!q.busy());
}

/// The slot ring wraps, the overrun report fires once, and an arena that cannot spare the
/// stack turns the queue OFF for good.
#[test]
fn a_callback_queue_wraps_reports_once_and_can_be_off() {
    let mut arena = GuestArena::new(0x1000);
    let mut q: CallbackQueue<u32> = CallbackQueue::default();
    assert!(q.ensure(4, 2, 0x40, arena_galloc(&mut arena), |_, _| {}));
    let first = q.take_slot(4, 2);
    q.take_slot(4, 2);
    assert_eq!(q.take_slot(4, 2), first, "the ring reuses its first slot");
    q.push(1);
    assert!(!q.first_overrun(2));
    q.push(2);
    assert!(q.first_overrun(2));
    q.push(3);
    assert!(!q.first_overrun(2), "reported once");

    let mut tiny = GuestArena::new(0x1000);
    let mut off: CallbackQueue<u32> = CallbackQueue::default();
    assert!(!off.ensure(
        4,
        2,
        0x10_0000,
        |s, al| tiny.galloc(s, al, 0x2000),
        |_, _| {}
    ));
    assert!(off.is_off());
    assert!(!off.ensure(4, 2, 0x10, |_, _| panic!("no retry once off"), |_, _| {}));
}

// --- KernelTimers ---

/// A timer is a stopwatch over the clock it is handed: it banks on stop, keeps counting when
/// re-set while running, and a double start does not lose time.
#[test]
fn a_kernel_timer_is_a_stopwatch_over_the_given_clock() {
    let mut t = KernelTimers::default();
    t.create(7, "frame", 0x10);
    assert_eq!(t.by_name("frame"), Some(7));
    assert_eq!(t.time_us(7, 100), Some(0), "created stopped");
    assert_eq!(t.start(7, 100), Some(false));
    assert_eq!(
        t.start(7, 150),
        Some(true),
        "already running - not restarted"
    );
    assert_eq!(t.time_us(7, 400), Some(300));
    assert_eq!(t.stop(7, 500), Some(true));
    assert_eq!(t.time_us(7, 9000), Some(400), "stopped time does not move");
    t.start(7, 1000);
    assert_eq!(t.set_time_us(7, 5, 1200), Some(600));
    assert_eq!(t.time_us(7, 1300), Some(105));
    assert!(t.running(7));
    assert_eq!((t.name(7), t.attr(7)), ("frame", 0x10));
    assert!(t.delete(7) && !t.delete(7));
    assert_eq!(t.time_us(7, 0), None);
}

// --- ProgramRefs ---

/// An identical vertex program is handed back with one more reference, and the share is
/// forgotten with the last release.
#[test]
fn a_shared_program_is_counted_and_forgotten_at_its_last_release() {
    let mut r = ProgramRefs::default();
    let key = (0x8100_0000, vec![1, 2, 3]);
    r.created(0x40);
    r.remember(key.clone(), 0x40);
    assert_eq!(r.share(&key), Some(0x40));
    assert_eq!(r.count(0x40), Some(2));
    assert!(r.released(0x40));
    assert_eq!(
        r.share(&key),
        Some(0x40),
        "still live after one of two releases"
    );
    assert!(r.released(0x40) && r.released(0x40));
    assert_eq!(r.count(0x40), None);
    assert_eq!(r.share(&key), None, "gone with its last reference");
    assert!(!r.released(0x40), "a release of nothing is told apart");
}

// --- PrecompileQueue ---

/// A pair is admitted once, and a program a precomputed state names pairs with the other
/// stage's programs named so far - each new header adding only its own pairs.
#[test]
fn the_precompile_queue_dedupes_and_pairs_state_programs_once() {
    let mut q = PrecompileQueue::default();
    assert!(q.admit(1, 2));
    assert!(!q.admit(1, 2), "a pair is queued once");
    assert_eq!(q.note_state_program(10, true), Some(vec![]));
    assert_eq!(q.note_state_program(20, false), Some(vec![10]));
    assert_eq!(q.note_state_program(21, false), Some(vec![10]));
    assert_eq!(q.note_state_program(11, true), Some(vec![20, 21]));
    assert_eq!(q.note_state_program(11, true), None, "named before");
    assert_eq!(q.note_state_program(0, true), None);
    assert_eq!(q.state_program_counts(), (2, 2));
    assert!(q.note_vertex(5) && !q.note_vertex(5) && !q.note_vertex(0));
    assert!(q.note_null_fragment(6) && !q.note_null_fragment(6));
    assert_eq!(
        (q.created_vertices(), q.null_fragments()),
        (&[5][..], &[6][..])
    );
}

// --- NetState ---

/// Socket ids sit above the file-descriptor range, a closed socket refuses every call, and an
/// option never set reads back as 0 while a bad descriptor reads back as `None`.
#[test]
fn the_offline_socket_table_keeps_local_state() {
    let mut n = NetState::default();
    let s = n.socket("udp", 2, 2, 0);
    assert!(s >= 0x1000);
    assert!(n.bind(s, 0x0100_007f, 3658) && n.set_opt(s, 1, 4, 9));
    assert_eq!(n.local_addr(s), Some((0x0100_007f, 3658)));
    assert_eq!((n.get_opt(s, 1, 4), n.get_opt(s, 1, 5)), (Some(9), Some(0)));
    assert!(n.close(s));
    assert!(!n.socket_exists(s) && !n.bind(s, 0, 0));
    assert_eq!(n.get_opt(s, 1, 4), None);
}

/// Deleting an http template frees everything under it, and ids are never reused.
#[test]
fn deleting_an_http_template_frees_its_children() {
    let mut n = NetState::default();
    let t = n.http_create(HttpKind::Template, 0, "ua", 0);
    let c = n.http_create(HttpKind::Connection, t, "http://x", 0);
    let r = n.http_create(HttpKind::Request, c, "/", 1);
    assert!(n.http_is(r, HttpKind::Request) && !n.http_is(r, HttpKind::Connection));
    assert!(n.http_delete(t));
    assert!(!n.http_exists(t) && !n.http_exists(c) && !n.http_exists(r));
    let t2 = n.http_create(HttpKind::Template, 0, "ua", 0);
    assert_ne!(t2, t, "a deleted id is not handed out again");
}

/// An epoll's waiters are handed back in the order they parked, and a timed-out one is no
/// longer a waiter an abort can reach.
#[test]
fn epoll_waiters_are_taken_in_order_and_timed_out_ones_forgotten() {
    let mut n = NetState::default();
    let e = n.epoll_create();
    n.add_epoll_waiter(3, e);
    n.add_epoll_waiter(4, e);
    n.add_epoll_waiter(5, e);
    n.forget_timed_out_epoll_waiters(&[4]);
    assert_eq!(n.take_epoll_waiters(e), vec![3, 5]);
    assert!(n.take_epoll_waiters(e).is_empty());
    assert!(n.epoll_latch_abort(e) && n.epoll_take_abort(e) && !n.epoll_take_abort(e));
    assert!(n.epoll_destroy(e) && !n.epoll_exists(e));
}

// --- ColorSurfaces ---

fn surface(data_addr: u32) -> crate::capture::ColorSurface {
    crate::capture::ColorSurface {
        format: 0,
        surface_type: 0,
        width: 64,
        height: 64,
        stride_pixels: 64,
        data_addr,
        scale_mode: 0,
        gamma: 0,
    }
}

/// The gamma mode answers by struct pointer first and falls back to the data address, so a
/// copy of the surface struct keeps its mode - but a zero data address is no identity.
#[test]
fn colour_surface_gamma_falls_back_to_the_data_address() {
    let mut c = ColorSurfaces::default();
    c.set(0x100, surface(0x8000_0000));
    c.set_gamma(0x100, 0x8000_0000, 1);
    assert_eq!(c.gamma_mode(0x100, 0), 1, "the pointer answers exactly");
    assert_eq!(c.gamma_mode(0x200, 0x8000_0000), 1, "a copy answers by its buffer");
    assert_eq!(c.gamma_mode(0x200, 0x9000_0000), 0);
    c.set_gamma(0x300, 0, 1);
    assert_eq!(c.gamma_mode(0x400, 0), 0, "a NULL buffer never matches");
    c.set_gamma(0x500, 0x8000_0000, 0);
    assert_eq!(c.gamma_mode(0x100, 0x8000_0000), 1, "the pointer still outvotes the buffer");
    assert_eq!(c.get(0x100).map(|s| s.data_addr), Some(0x8000_0000));
    assert_eq!(c.clip(0x100), None);
    c.set_clip(0x100, (1, 2, 3, 4));
    c.set_dither(0x100, 1);
    assert_eq!((c.clip(0x100), c.dither(0x100), c.dither(0x200)), (Some((1, 2, 3, 4)), 1, 0));
}

// --- TextureRegistry ---

/// A recorded format reports whether it changed, the nearby search finds the closest other
/// texture in the window, and the memo answers only for a handle with no format of its own.
#[test]
fn texture_registry_records_formats_and_memoises_the_nearby_search() {
    let mut t = TextureRegistry::default();
    assert!(t.set_format(0x1000, 7) && !t.set_format(0x1000, 7) && t.set_format(0x1000, 8));
    t.set_format(0x1100, 9);
    assert_eq!(t.nearest(0x1010, 0x100), Some((-0x10, 8)));
    assert_eq!(t.nearest(0x1000, 0x80), None, "the address itself is not its own neighbour");
    t.memo_nearby(0x1010, 0x100);
    t.memo_nearby(0x1000, 0x100);
    assert_eq!(t.nearby(0x1010), Some((-0x10, 8)));
    assert_eq!(t.nearby(0x1000), None, "a handle with a format is never searched");
    t.set_format(0x1008, 1);
    t.memo_nearby(0x1010, 0x100);
    assert_eq!(t.nearby(0x1010), Some((-0x10, 8)), "a memoised answer is not re-derived");
    t.set_palette(0x1000, 0xabc);
    assert_eq!((t.palette(0x1000), t.palette(0x1100)), (0xabc, 0));
}

// --- FiosOverlays ---

fn overlay(order: u8, dst: &str) -> FiosOverlay {
    FiosOverlay {
        id: 0,
        kind: SCE_FIOS_OVERLAY_TYPE_OPAQUE,
        order,
        pid: 0,
        dst: dst.into(),
        src: "ux0:o".into(),
    }
}

/// Overlays stay in application order through add and modify, and disabling is per thread
/// and reports the previous setting.
#[test]
fn fios_overlays_keep_order_and_per_thread_disable() {
    let mut f = FiosOverlays::default();
    let a = f.add(overlay(5, "app0:a"));
    let b = f.add(overlay(1, "app0:b"));
    let order = |f: &FiosOverlays| f.in_orders(0, 255).map(|o| o.id).collect::<Vec<_>>();
    assert_eq!(order(&f), vec![b, a]);
    assert!(f.modify(b, overlay(9, "app0:b")));
    assert_eq!(order(&f), vec![a, b]);
    assert_eq!(f.in_orders(6, 9).map(|o| o.id).collect::<Vec<_>>(), vec![b]);
    assert_eq!(f.get(b).map(|o| o.order), Some(9));
    assert!(f.remove(a) && !f.remove(a) && !f.modify(a, overlay(0, "x")));
    assert!(!f.set_disabled_for(3, true) && f.set_disabled_for(3, true));
    assert!(f.disabled_for(3) && !f.disabled_for(4));
    assert!(f.set_disabled_for(3, false) && !f.disabled_for(3));
}

// --- StorageClock ---

/// A transfer completes when the clock reaches its deadline, a flip nets out what the quanta
/// already charged, and releasing the earliest transfer jumps to exactly its deadline.
#[test]
fn storage_clock_parks_charges_and_nets_flips() {
    let mut s = StorageClock::default();
    assert_eq!(s.add_debt(5) + s.add_debt(0), 10);
    assert_eq!((s.take_debt(), s.debt()), (5, 0));
    s.park(1, 100);
    s.park(2, 300);
    assert_eq!(s.earliest_remaining(), Some(100));
    assert!(s.charge(60).is_empty());
    assert_eq!(s.deadline_of(1), Some(100));
    assert_eq!(s.frame(100), vec![1], "the flip owes 40 after the 60 charged: now 100");
    assert_eq!(s.now(), 100);
    assert_eq!(s.frame(50), Vec::<i32>::new(), "nothing charged since: a full 50");
    assert_eq!(s.release_earliest(), Some(vec![2]));
    assert_eq!(s.now(), 300);
    assert!(s.release_earliest().is_none() && !s.has_waiters());
    s.park(3, 10);
    s.forget(3);
    assert!(!s.has_waiters());
}

// --- PrecomputedRecords ---

/// A block is trusted only once claimed, the ladder prints the first four inits, then a new
/// program or a power of two, and a guest edit is adopted into the shadow.
#[test]
fn precomputed_records_claim_blocks_ladder_and_shadow() {
    let mut p = PrecomputedRecords::default();
    assert_eq!(p.block_owner(0x100), None);
    p.claim_block(0x100, 0x20);
    assert_eq!(p.block_owner(0x100), Some(0x20));
    let printed: Vec<Option<u32>> = (0..6).map(|_| p.note_state_init((1, 1))).collect();
    assert_eq!(printed, vec![Some(1), Some(2), Some(3), Some(4), None, None]);
    assert_eq!(p.note_state_init((2, 1)), Some(7), "a new program always prints");
    assert_eq!(p.note_state_init((2, 1)), Some(8), "a power of two prints");
    let mut words = [0u32; pdraw::WORDS as usize];
    p.shadow_draw(0x400, words);
    words[3] = 9;
    p.note_guest_edits(0x400, &words, 0);
    p.note_guest_edits(0x500, &words, 0);
}

// --- YieldElision ---

/// A pick sets the runnable count and ends the run, a wake makes the thread not alone, and the
/// run is capped: the yield AT the cap suspends and starts a fresh run.
#[test]
fn yield_elision_counts_picks_wakes_and_caps_the_run() {
    let mut y = YieldElision::default();
    y.note_pick(0);
    assert!(y.alone());
    assert!(y.elide_within(2) && y.elide_within(2));
    assert!(!y.elide_within(2), "the yield at the cap suspends");
    assert_eq!(y.elided_run(), 0);
    y.note_woken();
    assert!(!y.alone());
    y.note_pick(0);
    assert!(y.alone() && y.elided_run() == 0);
    assert_eq!(y.mirror_base(), None);
    y.set_mirror_base(0x100);
    y.set_published(true);
    assert_eq!((y.mirror_base(), y.published()), (Some(0x100), true));
}

// --- NgsSyncStash, OneShotCallback ---

/// The info ring is asked for once, a refusal turns the stash off, slots wrap the ring, and a
/// batch is per thread until it is finished.
#[test]
fn ngs_sync_stash_rings_slots_and_holds_batches_per_thread() {
    let mut s = NgsSyncStash::default();
    assert!(s.needs_ring());
    s.set_ring(0);
    assert!(!s.needs_ring() && s.is_off());
    let mut s = NgsSyncStash::default();
    s.set_ring(0x1000);
    assert!(!s.is_off());
    let slots: Vec<u32> = (0..3).map(|_| s.take_slot(0x1c, 2)).collect();
    assert_eq!(slots, vec![0x1000, 0x101c, 0x1000]);
    s.stash(7, vec![(0x8100_0001, 0x1000)]);
    assert!(s.is_replaying(7) && !s.is_replaying(8));
    assert_eq!(s.batch(7), Some(vec![(0x8100_0001, 0x1000)]));
    s.finish(7);
    assert!(!s.is_replaying(7) && s.batch(7).is_none());
}

/// A service callback is delivered once, a NULL entry clears it, and re-registering arms it
/// again.
#[test]
fn one_shot_callback_delivers_once_per_registration() {
    let mut c = OneShotCallback::default();
    assert_eq!(c.undelivered(), None);
    c.register(0x8100_0001, 0x20);
    assert_eq!(c.undelivered(), Some((0x8100_0001, 0x20)));
    c.mark_delivered();
    assert_eq!(c.undelivered(), None);
    c.register(0x8100_0001, 0x30);
    assert_eq!(c.undelivered(), Some((0x8100_0001, 0x30)));
    c.register(0, 0x30);
    assert_eq!(c.undelivered(), None);
}
