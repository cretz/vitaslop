use super::*;

impl VitaState {
    // --- lightweight condition variables + virtual clock -------------------
    //
    // Lightweight conds (`sceKernelWaitLwCond`/`SignalLwCond`) are identified by a
    // guest work pointer, not a kernel handle, so they park by work address here
    // rather than through `CondRec`. A parked wait yields the fiber (the caller
    // returns `Block`), which lets the scheduler run the thread that will set the
    // awaited condition and signal it - the whole point, since a non-yielding wait
    // starves the producer and busy-spins.

    /// The virtual monotonic clock in microseconds.
    pub fn now_us(&self) -> u64 {
        self.virtual_us
    }

    /// >>> THE CLOCK EVERY GUEST-VISIBLE TIME CALL MUST READ, AND THERE USED TO BE TWO.
    ///
    /// `World::monotonic_us`/`World::wall_us` are stamped `frame * 16666` by every world
    /// that drives a retail run, so they are CONSTANT for the whole duration of a display
    /// frame and step once at the flip. That is not a clock, it is the frame counter in
    /// microsecond clothing, and a title that paces itself off one cannot be paced by it.
    ///
    /// MEASURED on a retail racer's race, 100 display frames of a headless run: the title's
    /// update loop - `sceCtrlReadBufferPositive`, `sceRtcGetCurrentTick`, `sceMotionGetState`
    /// and its two network callbacks, all at exactly the same count - ran **12.85 times per
    /// rendered frame**, while `sceGxmBeginScene`/`EndScene`/`Draw` ran exactly ONCE per
    /// frame's worth. The loop is `t0 = sceKernelLibcClock(); do { step(); } while
    /// (sceKernelLibcClock() - t0 < one frame);` and that condition cannot become false
    /// inside a frame when the clock only moves AT a frame - so it spun, stepping the
    /// simulation every time round. The game's own lap timer advanced 12.4x real
    /// ([[vitaslop-game-clock-runs-12-8x-the-emulated-clock]] measured the ratio without
    /// naming the cause), which is what "the race is way sped up" is.
    ///
    /// [`now_us`](Self::now_us) is the clock the scheduler actually advances - by charged
    /// quanta, by the idle jump, and by the vblank floor at a flip - so it moves WHILE guest
    /// code runs and a self-timing loop terminates against it. `sceKernelGetProcessTimeWide`
    /// and `sceDisplayGetVcount` were already on it; these are the calls that were not.
    ///
    /// The run-to-completion host has no scheduler and therefore no such clock (nothing
    /// charges quanta and nothing paces a flip), so there the world's frame-stamped value is
    /// still the only thing that moves and is still what it gets.
    pub fn guest_mono_us(&mut self) -> u64 {
        if self.is_preemptive() {
            self.virtual_us
        } else {
            self.world.monotonic_us()
        }
    }

    /// The guest's WALL clock: a fixed epoch plus [`guest_mono_us`](Self::guest_mono_us).
    ///
    /// Same argument as that function - a wall clock that only ticks at a display flip is
    /// no more usable for a delta than a monotonic one, and this title reads
    /// `sceRtcGetCurrentTick` once per update iteration alongside `sceKernelLibcClock`.
    pub fn guest_wall_us(&mut self) -> u64 {
        if self.is_preemptive() {
            GUEST_WALL_EPOCH_US.wrapping_add(self.virtual_us)
        } else {
            self.world.wall_us()
        }
    }

    /// The tick `sceRtcGetCurrentTick` writes: microseconds since the SceRtc epoch.
    ///
    /// The SAME expression the handler uses, factored out so the HOST MIRROR block and
    /// `crate::vita::services::rtc_get_current_tick` cannot drift - the inline form is a
    /// second implementation of the call and nothing else in the system would notice them
    /// disagreeing (`mirror_matches_its_handlers` holds them to it).
    ///
    /// `&self`, unlike [`guest_wall_us`](Self::guest_wall_us), because the mirror snapshot has
    /// only a shared borrow - which is also the reason the SINGLE-THREAD arm is not mirrored:
    /// there the wall clock is the host's own and moves while guest code runs, so
    /// [`crate::vita::services::inline_op`] withholds the inline form rather than serve a value
    /// the mirror contract does not cover.
    pub fn guest_wall_tick(&self) -> u64 {
        crate::vita::services::RTC_UNIX_EPOCH_TICKS
            .wrapping_add(GUEST_WALL_EPOCH_US.wrapping_add(self.virtual_us))
    }

    /// Add `us` to the accrued storage-latency debt and return the new total. See
    /// `vita::iofilemgr::charge_read`: the debt exists so the modelled transfer time is
    /// charged in full without a scheduler round trip per read.
    #[inline]
    pub fn add_io_debt_us(&mut self, us: u64) -> u64 {
        self.storage.add_debt(us)
    }

    /// The accrued storage-latency debt, without touching it. See
    /// `vita::iofilemgr::park_for_debt`, which decides whether it is worth a context switch
    /// before clearing it.
    #[inline]
    pub fn io_debt_us(&self) -> u64 {
        self.storage.debt()
    }

    /// Clear the accrued storage-latency debt (the caller is about to park for it).
    #[inline]
    pub fn take_io_debt_us(&mut self) -> u64 {
        self.storage.take_debt()
    }

    /// Park the current thread on the STORAGE clock for `us` of modelled transfer time.
    ///
    /// Not [`sleep_park`](Self::sleep_park): a sleep's deadline is in `virtual_us`, which
    /// only moves on a display flip or on the scheduler's idle path. A title blocked
    /// waiting for the load produces no flips, and a sibling thread polling with a
    /// zero-length delay keeps the idle path finding a deadline that buys no time, so the
    /// game clock stands still and the load can never finish - the measured livelock. The
    /// storage clock has no such dependency.
    #[inline]
    pub fn io_park(&mut self, us: u64) {
        self.storage.park(self.current, us);
    }

    /// Wake every thread a storage-clock step released (see [`StorageClock`]).
    fn wake_io(&mut self, woken: Vec<i32>) {
        for t in woken {
            push_wake!(self, t);
        }
    }

    /// One display flip happened: bring the storage clock up to one full frame of
    /// progress, minus whatever the quantum charges already contributed since the last
    /// flip. While a title renders, this is the whole model - the device streams one
    /// frame's worth of bytes per rendered frame, which is what a loading screen's
    /// spinner is telling the player.
    pub fn advance_io_frame(&mut self, frame_us: u64) {
        let woken = self.storage.frame(frame_us);
        self.wake_io(woken);
    }

    /// A thread burned a whole scheduler quantum (or yielded) without a flip. Time
    /// passed on the device too. This is what keeps a title that spins in guest code
    /// waiting for a load - never blocking, never flipping - from waiting forever;
    /// [`advance_io_frame`](Self::advance_io_frame) nets these out so a normally
    /// rendering title still advances exactly one frame of storage time per frame.
    pub fn charge_io_quantum(&mut self, us: u64) {
        let woken = self.storage.charge(us);
        self.wake_io(woken);
    }

    /// A thread burned a whole scheduler quantum of guest execution. **Game time passed
    /// while it did**, and this is the only thing that says so when the title is neither
    /// blocking nor flipping.
    ///
    /// Without this the game clock advances only on a display flip or on the scheduler's
    /// nothing-is-runnable idle path, so a guest busy-wait ON THE CLOCK ITSELF - the
    /// `do { v = sceDisplayGetVcount(); } while (v == last);` vblank spin that is
    /// ordinary, correct guest code - can never be satisfied: the thread never blocks,
    /// so the scheduler is never idle, and no flip can happen because the flip is what
    /// the spin is waiting for. Two such threads livelock the whole title. That is not a
    /// slow boot, it is a stopped clock, and it is what made this title's boot take
    /// hours while the clock advanced 60 ms.
    ///
    /// [`advance_time_frame`](Self::advance_time_frame) nets these charges out, so a
    /// title that renders normally still advances exactly one frame of game time per
    /// rendered frame and its own timers keep 60 Hz - the charges only become visible
    /// when the guest burns more CPU than a frame's worth, which is the honest answer.
    pub fn charge_cpu_quantum(&mut self, us: u64) {
        self.quantum_count = self.quantum_count.saturating_add(1);
        self.cpu_charged_since_flip = self.cpu_charged_since_flip.saturating_add(us);
        let target = self.virtual_us.saturating_add(us);
        self.clock_source = ClockSource::Quantum;
        self.advance_time_to(target);
    }

    /// Seconds of sound the guest has submitted through `sceAudioOut`. See
    /// [`crate::vita::audio::AudioState::produced_seconds`] for why this belongs next to the
    /// clock split and not in the audio backend's own counters.
    pub fn audio_produced_seconds(&self) -> f64 {
        self.audio_state.produced_seconds()
    }

    /// >>> THE GAME CLOCK MAY NOT RUN SLOWER THAN REAL TIME (SMP browser runs).
    ///
    /// The clock advances by charged quanta (a flat amount per quantum of guest execution) and
    /// by idle jumps. On a device whose guest executes slower than a Vita's core, the quanta
    /// arrive slower than real time, so the clock - and every vblank edge on it - falls behind
    /// the wall. MEASURED on the phone (a baseball at-bat, trace 059): the main thread parked in
    /// `sceCtrlReadBufferPositive` (next vblank) for 23.8 ms/f and was released only ~5 ms
    /// after the render thread flipped, because the edge it waited for was reached only by the
    /// render thread's own quanta. On hardware the vblank is real time and main's next update
    /// overlaps render's draw; here the two were serialised, and the game also ran in slow
    /// motion.
    ///
    /// So the clock is floored at the wall: the floor advances by elapsed wall time and never
    /// sits below the clock itself (a guest running ahead of real time is left alone - the live
    /// loop's gate paces it). Returns whether the clock moved.
    ///
    /// >>> ONLY WALL TIME THE GUEST SPENT RUNNING. Each step is clamped to `MAX_STEP_MS`, which
    /// is far above the gap between two ticks of a running guest (every run end ticks) and far
    /// below a stop. A 50 ms clamp let the time the live loop HELD the guest at its gate come
    /// back as game time on the next tick; the loop charges a frame the game time it advanced,
    /// so it held the guest longer, which advanced the clock further - MEASURED on the phone
    /// (062): frames alternating ~30 / ~100 ms with 40-60 ms of nothing running, 16 fps.
    ///
    /// >>> AND NOT A LINE ANCHORED TO THE WALL. Tried 27c (phone job 025, a baseball title's pitches): with
    /// the whole wall recovered the clock read 97% speed, but the guest's own frame period went
    /// from 45 to 80 ms (11 fps), the audio thread fell to 0.66x of the clock and the ring
    /// underran 35% of the time - worse on every axis the player sees. Pushing the clock does
    /// not make the guest faster; it makes every timed wait expire early.
    ///
    /// >>> HELD TIME IS SKIPPED, SO THE CLAMP CAN COVER A WHOLE QUANTUM. `held` (the frame gate
    /// is shut or the guest is paused) moves the tick's anchor without advancing anything, so
    /// the clamp no longer has to be what keeps a hold out. It was 4 ms, and a phone resume
    /// runs 5-7 ms between two ticks: MEASURED (029, a baseball title's pitches) the main thread parked 17 ms
    /// for a vblank while the render thread drew, released only by the render thread's own
    /// quanta, because every tick threw away a third of the wall that had passed.
    /// `VITASLOP_CLOCK_WALL_STEP_MS` sets the clamp (default 12).
    pub fn wall_floor_tick(&mut self, wall_ms: f64, held: bool) -> bool {
        let max_step_ms = wall_step_ms();
        let (last, floor) = match self.wall_floor {
            None => {
                self.wall_floor = Some((wall_ms, self.virtual_us));
                return false;
            }
            Some(v) => v,
        };
        if held {
            self.wall_floor = Some((wall_ms, floor));
            return false;
        }
        let step = (wall_ms - last).clamp(0.0, max_step_ms);
        // >>> THE FLOOR TRAILS A CLOCK THAT RAN AHEAD; IT DOES NOT JUMP UP TO IT. It used to be
        // `max(floor + step, clock)`: every idle jump (the clock running AHEAD of the wall) reset
        // the floor to the clock, and the next ticks added wall on top - a ratchet that pushes a
        // guest past real time. MEASURED on the phone (042, a baseball title's pitches) once ticks came often
        // (idle workers now wake for wall parks): `% speed` mean 104.7%. Kept at most
        // `WALL_FLOOR_LAG_US` below the clock, so an idle jump is paid back by wall time first
        // and a long idle stretch is not banked.
        const WALL_FLOOR_LAG_US: u64 = 67_000;
        let floor =
            (floor + (step * 1000.0) as u64).max(self.virtual_us.saturating_sub(WALL_FLOOR_LAG_US));
        self.wall_floor = Some((wall_ms, floor));
        if floor > self.virtual_us {
            // >>> THE STORAGE CLOCK GETS THE SAME FLOOR. A Vita's card streams in real time; the
            // storage clock otherwise moves only on flips, quanta and idle jumps, and a title
            // that polls a load with short delays from two threads (a fighting title's sound banks: main and
            // render each `sceKernelDelayThread(100)` ~2000 times a frame) is never globally
            // idle and never flips. MEASURED on the phone (job 098): AsyncIOSystem parked
            // 180-240 ms per 128 KB pread modelled at ~2.7 ms, frames of 530 ms; the desktop
            // goes idle between polls and never saw it. Charged like an idle jump, so a flip
            // nets it out and a rendering title still streams one frame per frame.
            // `VITASLOP_IO_WALL_FLOOR=0` is the arm back.
            if io_wall_floor() {
                self.charge_io_idle(floor - self.virtual_us);
            }
            self.clock_source = ClockSource::Wall;
            self.advance_time_to(floor);
            return true;
        }
        false
    }

    /// Game clock gained from the wall-clock floor so far, in microseconds.
    pub fn clock_from_wall_us(&self) -> u64 {
        self.clock_from_wall_us
    }

    /// `(from quanta, from the frame top-up, from idle jumps)` microseconds of game clock
    /// so far.
    ///
    /// The split, not the total. Two engines sharing this scheduler can reach very
    /// different clocks from the same guest, and the fix depends entirely on which
    /// bucket differs: a smaller `quanta` bucket means the preemption rate is wrong,
    /// while a smaller `idle` bucket means the run never goes idle - some thread is
    /// still runnable - and no amount of re-calibrating the quantum will touch it.
    pub fn clock_sources(&self) -> (u64, u64, u64) {
        (
            self.clock_from_quanta_us,
            self.clock_from_topup_us,
            self.clock_from_idle_us,
        )
    }

    /// One display flip happened: bring the game clock up to one full frame of progress,
    /// minus whatever [`charge_cpu_quantum`](Self::charge_cpu_quantum) already
    /// contributed since the last flip. The storage clock's
    /// [`advance_io_frame`](Self::advance_io_frame) with the game clock's units.
    ///
    /// # The top-up hands out time the guest was never given the CPU to use
    /// >>> IT IS OFF BY DEFAULT AND `VITASLOP_FRAME_TOPUP=1` IS WHAT TURNS IT ON. This is
    /// > > > stated in that direction deliberately: it used to read "`=0` switches it off", which
    /// > > > implies a default of ON, and `frame_topup()` tests for `Ok("1")`. Which way round it is
    /// > > > decides what a displayed frame MEANS, so the sentence has to be unambiguous.
    /// > > > With the top-up on, a flip advances the
    /// > > > clock a whole frame however little guest code ran, so the render thread's next
    /// > > > vblank wait is already satisfied and it flips again at once. Measured on a
    /// > > > retail title's loading screen: **3120 fps, 8 thread resumes a frame, barely one
    /// > > > whole scheduler quantum per frame** - the guest gets about a TENTH of the CPU a
    /// > > > console frame carries, so a load that costs hardware 3,000 frames costs us
    /// > > > 30,000+, and the wall time goes on per-frame overhead instead of on the loader.
    ///
    /// With it off, a flip advances the clock by nothing it did not earn. The render
    /// thread's vblank wait is then a real wait: its spin burns quanta, every other
    /// runnable thread is scheduled between them, and the frame ends when a frame's
    /// worth of guest execution has actually happened. Neither of the two ways the
    /// clock can stall applies - a spinning thread charges quanta, and a title that
    /// blocks everything reaches the scheduler's idle path, which advances the clock to
    /// the earliest deadline.
    ///
    /// It is a knob rather than the default because it moves every title's timing and
    /// every determinism signature.
    pub fn advance_time_frame(&mut self, frame_us: u64) {
        self.flip_pacer.count_flip();
        let charged = core::mem::take(&mut self.cpu_charged_since_flip);
        if !frame_topup() {
            return;
        }
        let owed = frame_us.saturating_sub(charged);
        let target = self.virtual_us.saturating_add(owed);
        self.clock_source = ClockSource::FrameTopup;
        self.advance_time_to(target);
    }

    /// Display flips completed so far - the frame number scripted input and a shot count
    /// in. Diagnostics stamp it so an event can be placed on the timeline.
    #[inline]
    pub fn flip_count(&self) -> u64 {
        self.flip_pacer.flip_count()
    }

    /// Scheduler quanta and display flips seen so far. `QUANTUM_CPU_US` is calibrated
    /// from their ratio on a title that is rendering steadily: one quantum is worth one
    /// frame divided by the quanta a frame actually takes.
    pub fn quantum_flip_counts(&self) -> (u64, u64) {
        (self.quantum_count, self.flip_pacer.flip_count())
    }

    /// How much more storage time the EARLIEST outstanding transfer still owes, or
    /// `None` when nothing is in flight.
    ///
    /// This is the number the idle path compares against the next timed wait. Without it
    /// the scheduler could only ask "is a transfer outstanding", and the answer to that
    /// question is the same for a transfer with 200 us left and one with 40 ms left -
    /// which is how a modelled read came to cost no display frames at all whatever its
    /// size. See [`crate::sched`]'s idle path.
    #[inline]
    pub fn earliest_io_remaining_us(&self) -> Option<u64> {
        self.storage.earliest_remaining()
    }

    /// Nothing is runnable and the game clock is being jumped to `us` ahead: the device
    /// was reading for that whole idle interval too, so credit it.
    ///
    /// Charged exactly like a quantum - and netted out of the next flip for the same
    /// reason - so a title that renders normally still advances exactly one frame of
    /// storage time per rendered frame however its idle fell.
    pub fn charge_io_idle(&mut self, us: u64) {
        self.charge_io_quantum(us);
    }

    /// Nothing is runnable and no timed wait completes sooner: complete the earliest
    /// outstanding transfer. Returns whether anything was released.
    ///
    /// **The caller must have established that no timed wait expires first.** The
    /// earlier rule here was "an outstanding transfer comes FIRST, before any timed
    /// wait", justified as "no guest code can run until something completes, so nothing
    /// observable is lost". That reasoning is wrong, and the thing it loses is the
    /// display. The render thread's vblank wait IS a timed wait, so completing the
    /// transfer ahead of it means a transfer always finishes before the next frame flips,
    /// whatever its modelled size - and the storage clock only advances on flips and
    /// quanta, so the size then buys nothing at all. Measured on a retail racer's course
    /// load: the modelled cost of a 1,996,800-byte read moved from 4,554 us to 25,020 us
    /// across a 250x bandwidth sweep and the load landed on the SAME guest frame every
    /// time, because the idle path handed it back its time immediately. The transfer and
    /// the timed wait are now honoured in time order.
    pub fn release_earliest_io(&mut self) -> bool {
        let Some(woken) = self.storage.release_earliest() else {
            return false;
        };
        self.wake_io(woken);
        true
    }

    /// Whether any thread is parked paying storage latency.
    #[inline]
    pub fn has_io_waiters(&self) -> bool {
        self.storage.has_waiters()
    }

    /// Park the current thread until `now + us` on the virtual clock, woken only by
    /// time. Used for `sceKernelDelayThread` and `sceAudioOutOutput` grain pacing.
    pub fn sleep_park(&mut self, us: u64) {
        self.sleep_waiters
            .push((self.current, self.virtual_us.wrapping_add(us)));
    }

    /// >>> PARK THE CURRENT THREAD UNTIL A HOST WALL-CLOCK TIME, not a guest-clock one.
    ///
    /// For a wait the DEVICE times in real time whatever the guest clock does - the audio
    /// output's DAC (see `vita::audio::out_output`). A virtual-clock park there stretches with
    /// every stall of the virtual clock: MEASURED on the phone (a baseball title's pitches, 040) a ~10 ms audio
    /// park lasted 50-85 ms of wall 27 times in 4.4 s, while every guest thread was parked and
    /// the run worker presented. Served only by an engine that expires them
    /// ([`wall_parks_served`]); the caller falls back to [`Self::sleep_park`] otherwise.
    pub fn wall_park(&mut self, until_wall_us: u64) {
        self.wall_waiters.push((self.current, until_wall_us));
        note_next_wall_park(&self.wall_waiters);
    }

    /// Wake every wall-parked thread whose time has come. Called by the engine that serves them.
    /// Returns the threads it woke - the scheduler treats a thread that waits on real time as a
    /// real-time thread (see `smp`'s gate).
    pub fn expire_wall_parks(&mut self, now_wall_us: u64) -> Vec<i32> {
        let mut woke = Vec::new();
        if self.wall_waiters.is_empty() {
            return woke;
        }
        let mut i = 0;
        while i < self.wall_waiters.len() {
            if self.wall_waiters[i].1 <= now_wall_us {
                let (thid, _) = self.wall_waiters.swap_remove(i);
                push_wake!(self, thid);
                woke.push(thid);
            } else {
                i += 1;
            }
        }
        note_next_wall_park(&self.wall_waiters);
        woke
    }

    /// Whether any thread is parked on the wall clock (an idle guest is then waiting, not stuck).
    pub fn has_wall_parks(&self) -> bool {
        !self.wall_waiters.is_empty()
    }

    /// Record the scheduler's runnable set size at the pick - see
    /// [`crate::sched::ImportDispatch::note_runnable_others`].
    #[inline]
    pub fn note_runnable_others(&mut self, n: usize) {
        // A pick means a real suspend has just happened, so the run of elided ones is over.
        self.yields.note_pick(n);
    }

    /// Consecutive `delay(0)` yields one thread may have returned to it before the next one
    /// really suspends.
    ///
    /// >>> THIS BOUND IS NOT A TUNING PARAMETER - IT IS THE ONE CASE THE REASONING ABOVE DOES
    /// >>> NOT COVER.
    ///
    /// That reasoning says nothing can make another thread runnable while this one runs except
    /// this one's own host calls. It is a statement about the SCHEDULER's state, and there is a
    /// wake that does not pass through it: [`VitaState::wake_thread`] exists for exactly that -
    /// "make a parked thread runnable from OUTSIDE a host call", which is how the frontend's
    /// asynchronous scene completion releases the thread that ended the scene. A guest spinning
    /// on such a release, and a browser worker that never returns to its event loop to receive
    /// one, are the same shape: the loop has to be let go of for the wake to arrive.
    ///
    /// So the elision is bounded rather than absolute: at most this many in a row, then a real
    /// suspend whatever the runnable set says. At 2,674 of these a frame the bound still
    /// removes better than 99% of them, and nothing can be deferred by more than one run.
    ///
    /// NOT measured as necessary - both engines ran a football title's whole scripted run with the
    /// unbounded form. It is here because the hazard is nameable, not because it was seen.
    ///
    /// >>> RAISED 256 -> 4096 (2026-09-17) because a SUSPEND IS THE EXPENSIVE UNIT. MEASURED in
    /// the browser on that football title with the yield inlined: forcing one suspend per 256
    /// elided yields (`inline_elide_run_at_cap`) added ~37 suspends a frame and ~3.5 ms of
    /// host-call time - about 0.1 ms per JSPI suspend and resume. The fuel preemption still
    /// bounds every deferral; this cap is the safety net behind it, and 4096 iterations of a
    /// pacing spin is ~25 ms of guest time, under the preemption interval's own worst case.
    ///
    /// The cap also sets how often a pacing spin hands the CPU back, which is what decides
    /// when a sleeping thread's deadline is next checked: a football title's browser timeline
    /// moved with it (256 = a huddle, 1024 = the hike, 4096 = a huddle at the same frame
    /// number), while both arms billed the same clock per flip. Not a defect anyone can name
    /// without hardware; judge that title by game state, not frame number.
    pub const ELIDE_RUN_CAP: u32 = 4096;

    /// The elide predicate as ONE WORD, for the host-mirror block
    /// (`vita::mirror::SLOT_YIELD_FREE`): 1 when a `sceKernelDelayThread(<=1)` from the thread
    /// about to run would be elided by [`Self::yield_would_repick_this_thread`] - the same
    /// terms, minus the run cap, which the emitted code tests against its own slot.
    pub fn yield_free_word(&self) -> u32 {
        (yield_elide_on()
            && self.yields.alone()
            && self.pending_wakes.is_empty()
            && self.pending_spawns.is_empty()) as u32
    }

    /// The consecutive-elide count for the mirror's `SLOT_ELIDE_RUN` - the emitted
    /// `DelayYield` increments the word in place and stops eliding at
    /// [`Self::ELIDE_RUN_CAP`], where the handler takes over with its own count.
    #[inline]
    pub fn elided_run_word(&self) -> u32 {
        self.yields.elided_run()
    }

    /// Where the mirror block is, so [`Self::publish_yield_free_change`] can write into it.
    #[inline]
    pub fn set_mirror_block_base(&mut self, addr: u32) {
        self.yields.set_mirror_base(addr);
    }

    /// Whether the inlined `DelayYield` has already elided a whole cap of yields this slice
    /// (the mirror's `SLOT_ELIDE_RUN` word), so the yield that just reached the host is the
    /// one the cap means to suspend. Resets the host's own count with it: the resume that
    /// follows republishes zero and the inline form starts a fresh run.
    pub fn inline_elide_run_at_cap(&mut self, ctx: &GuestCtx) -> bool {
        let Some(base) = self.yields.mirror_base() else {
            return false;
        };
        let run = ctx.read_u32(base.wrapping_add(vita::mirror::SLOT_ELIDE_RUN * 4));
        if run < Self::ELIDE_RUN_CAP {
            return false;
        }
        self.yields.reset_run();
        true
    }

    /// Record what the last refresh published for `SLOT_YIELD_FREE`.
    #[inline]
    pub fn note_yield_free_published(&mut self, free: bool) {
        self.yields.set_published(free);
    }

    /// >>> THE ONE MIRROR WORD A HOST CALL CAN CHANGE, KEPT HONEST.
    ///
    /// `SLOT_YIELD_FREE` says "nobody to yield to". Between two resumes the only thing that
    /// can make that false is THIS thread's own host call waking or spawning another thread -
    /// and an inlined `sceKernelDelayThread(1)` reading a stale 1 would elide a yield that had
    /// somewhere to go, deferring that thread's turn to the elide cap. So every host call
    /// return passes through here: if the word was published as 1 and the predicate has since
    /// turned false, the word is cleared IN GUEST MEMORY, once, and the next inlined delay
    /// falls through to the handler exactly as it would have before this existed.
    ///
    /// One branch per host call in the common case; the write itself is rare - it happens at
    /// most once per slice, on the first wake.
    pub fn publish_yield_free_change(&mut self, ctx: &mut GuestCtx) {
        if !self.yields.published() || self.yield_free_word() != 0 {
            return;
        }
        if let Some(base) = self.yields.mirror_base() {
            ctx.write_u32(base.wrapping_add(vita::mirror::SLOT_YIELD_FREE * 4), 0);
        }
        self.yields.set_published(false);
    }

    /// Note that a host call just made a thread RUNNABLE, so a yield from here has somewhere
    /// to go even though the pick said it did not.
    ///
    /// Called from the wake paths rather than inferred: the scheduler's count is exact AT THE
    /// PICK, and the only thing that can change it while this thread runs is this thread's own
    /// host calls - which are these.
    #[inline]
    pub fn note_thread_woken(&mut self) {
        self.yields.note_woken();
    }

    /// >>> WHETHER A `sceKernelDelayThread(0)` HAS ANYONE TO GIVE THE CPU TO.
    ///
    /// The call means "yield"; the kernel returns to the caller when no other thread is ready.
    /// This scheduler could not say that, so every one of them cost a full suspend and resume
    /// and the scheduler then re-picked the same thread. MEASURED in the browser, where a
    /// suspend is the most expensive thing a scheduler round does: a football title runs
    /// **9.6 million QUANTUM suspends** over 4,047 flips - **2,377 a frame** - against a
    /// baseball title's 4, and its call-site histogram puts **10.4 million** of them at ONE
    /// guest address inside its lock code.
    ///
    /// The answer is exact rather than heuristic. The count comes from the PICK that resumed
    /// this thread, and this scheduler runs one thread at a time, so the only thing that can
    /// make another thread runnable in between is this thread's own host calls - which report
    /// it through [`Self::note_thread_woken`]. A TIMED wake needs the clock to advance and the
    /// clock advances at suspensions, and the ENGINE's own fuel preemption still takes those,
    /// so no wake can be deferred past one preemption interval by this.
    ///
    /// `VITASLOP_YIELD_ELIDE=0` is the arm back to always suspending.
    pub fn yield_would_repick_this_thread(&mut self) -> bool {
        if !yield_elide_on() {
            YIELDS_TAKEN.fetch_add(1, std::sync::atomic::Ordering::Relaxed);
            return false;
        }
        if !self.yields.alone() {
            YIELDS_TAKEN.fetch_add(1, std::sync::atomic::Ordering::Relaxed);
            return false;
        }
        // ...and anything THIS SLICE has released or started. `SchedCore::drain` takes these
        // after a resume, not after a host call, so a thread this slice woke is still sitting
        // here - and it is exactly the thread a yield now has somewhere to go to. One check
        // covers every wake path rather than each of them having to remember.
        if !self.pending_wakes.is_empty() || !self.pending_spawns.is_empty() {
            YIELDS_TAKEN.fetch_add(1, std::sync::atomic::Ordering::Relaxed);
            return false;
        }
        // ...and the bound, which is what keeps a spin that waits on something outside this
        // reasoning from never giving the loop back. See `ELIDE_RUN_CAP`.
        if !self.yields.elide_within(Self::ELIDE_RUN_CAP) {
            YIELDS_TAKEN.fetch_add(1, std::sync::atomic::Ordering::Relaxed);
            return false;
        }
        YIELDS_ELIDED.fetch_add(1, std::sync::atomic::Ordering::Relaxed);
        true
    }

    /// Pace a display flip to the scanout: park the calling thread until the display can
    /// actually latch the frame it just queued, and reserve the vblank after that for the
    /// one to come. `frame_us` is one display period.
    ///
    /// # Vsync is a FLOOR on when a flip may complete, not a grant of free time
    /// Neither of the two behaviours this replaces was physical, and they failed in
    /// opposite directions.
    ///
    /// With the per-flip top-up ON, a flip advanced the clock a whole frame however
    /// little guest code had run. The render thread's next vblank wait was therefore
    /// already satisfied and it flipped again at once: measured on a retail loading
    /// screen, **3120 fps and barely one scheduler quantum per frame**, so the guest got
    /// about a TENTH of the CPU a console frame carries and a load that costs hardware
    /// 3,000 frames cost 30,000+.
    ///
    /// With it OFF, a flip advanced the clock by nothing it had not earned - but nothing
    /// stopped the guest flipping again immediately either, so the frame rate became
    /// "however fast the guest can draw". Measured on this title's front end: **4.64 ms
    /// of game clock per flip, 216 fps**, against a 60 Hz panel. Every wall-clock wait in
    /// the title then cost 3.6x the flips hardware needs, and since a frame number is
    /// what scripted input, a screenshot and `--max-frames` are keyed to, the whole timeline
    /// stretched by that factor. The race was unaffected (33.5 ms/flip, a true 30 fps),
    /// which is what makes this a pacing bug and not a speed one.
    ///
    /// The floor fixes both. A flip may not complete before the next vblank, so a cheap
    /// frame WAITS - and it waits by blocking, not by being handed time: while this
    /// thread is parked the scheduler runs every other runnable thread (charging real
    /// quanta) and only jumps the clock when nothing at all is runnable. The guest
    /// therefore gets a whole frame's worth of CPU per frame, which is the half the
    /// top-up never supplied.
    ///
    /// # A latch happens ON a vblank, not one period after the request
    /// The floor above is a duration; the scanout is a GRID. A frame that takes 17 ms
    /// of a 16.67 ms period does not latch at 17 ms - it misses the vblank and waits
    /// for the next one, so it costs 33.3 ms and the title runs at 30 Hz. Rounding the
    /// latch up to the grid is what produces the console's characteristic 60/30/20 Hz
    /// quantisation instead of a continuum, and it is also what lets a title that
    /// renders in 5 ms and then waits for vblank stay PHASE-LOCKED rather than drifting
    /// by its own render time every frame. `sceDisplayGetVcount` is already defined off
    /// this same grid ([`crate::vita::display::vcount`]), so the two agree by
    /// construction.
    ///
    /// Returns the microseconds parked, for the caller's diagnostics.
    pub fn pace_flip(&mut self, frame_us: u64) -> u64 {
        // >>> THE QUEUE HAS THE DEPTH THE GUEST ASKED FOR, AND THAT IS NOT A THROUGHPUT
        // >>> TUNABLE - IT IS A WHOLE FRAME OF GAME CLOCK PER FRAME.
        //
        // This used to park the caller until the next vblank on EVERY call, on the
        // reasoning that `displayQueueMaxPendingCount` only raises throughput and cannot
        // raise the latch rate above the panel, so the depth could not matter to the frame
        // count or the clock. The first half is true and the second does not follow.
        //
        // MEASURED on a retail sports title, which sets a depth of three and paces itself:
        // its main loop waits on `sceDisplayGetVcount` for its own target rate (a 100 us
        // poll loop, disassembled), THEN queues the frame. With a forced depth of one, the
        // queue call parked it for a further whole vblank that hardware would have spent
        // scanning out in parallel - so a frame cost THREE display periods where the title
        // asked for two, and the guest's clock ran **3.01x its displayed frames, dead
        // steady**. That is not a throughput loss. The clock is what a title's own timers
        // and its AUDIO are billed in: the same run produced **232 s of sound against 78 s
        // of picture (2.96x)**, which on a device fills the audio ring, drops the excess,
        // and then starves - reported as a broken audio backend and caused here.
        //
        // So the guest may be `displayQueueMaxPendingCount` frames ahead of the scanout,
        // exactly as GXM promises, and parks only when the queue is genuinely full. What
        // the floor still guarantees is the thing it was written for: latches land ON the
        // vblank grid and never closer together than one period, so a title that queues as
        // fast as it can draw still latches at 60 Hz and no faster (the 216 fps case).
        //
        // [`FlipPacer::pace`] retires the latches the scanout has already shown, parks for a
        // full queue, and books this frame's latch on the vblank grid.
        //
        // >>> THE SAME DEPTH BOUNDS THE GUEST'S OWN DISPLAY CALLBACK, AND IT WAS NOT BOUNDED
        // >>> AT ALL.
        //
        // GXM runs the display-queue callback once per queued entry, and `AddEntry` blocks
        // while `displayQueueMaxPendingCount` of them are still outstanding - that is the whole
        // of the guest's back-pressure on its own display path. This engine ran the callback on
        // a serial callback thread and never bounded the queue behind it, so a title whose
        // callback completes less often than it queues runs away.
        //
        // MEASURED on a role-playing title's front-end movie (`displayQueueMaxPendingCount = 2`): the
        // guest queued 60 entries a second and the callback completed 30, and by frame 2700 the
        // backlog was **1,348 entries deep**. The callback-data ring is
        // [`Self::DISPLAY_CB_SLOT_COUNT`] slots, so it had wrapped 168 times and EVERY callback
        // was handed a much newer entry's data - the title's own buffer bookkeeping, fed
        // garbage, for the whole run.
        //
        // So the shortfall is charged back to the thread that queued: parking it for the frames
        // the callback still owes makes the guest's flip rate the rate its own display path can
        // retire, exactly as the blocking `AddEntry` does on hardware. In steady state the
        // backlog is at most `depth` and this adds nothing.
        let owed = self
            .display_callback_backlog()
            .saturating_sub(self.flip_pacer.depth());
        let park = self.flip_pacer.pace(self.virtual_us, frame_us, owed);
        // >>> ALWAYS PARK, EVEN FOR ZERO. The scheduler BLOCKS the flipping thread at
        // `Stop::Flip` unconditionally (that is what stops a title flipping as fast as it
        // can draw), so the sleep waiter registered here is the only thing that will ever
        // make it runnable again. Registering it only on the full-queue path deadlocked the
        // first frame of a title with a deeper queue: the guest queued, the scheduler
        // blocked it, and nothing named a wake.
        self.sleep_park(park);
        park
    }

    /// Take the "a frame has been presented since the last display wait" flag.
    ///
    /// # What it is for, and the measurement behind it
    /// `sceDisplayWaitSetFrameBuf` waits for the buffer the last `sceDisplaySetFrameBuf`
    /// named to reach the panel. In THIS engine a frame reaches the panel synchronously,
    /// inside `sceGxmDisplayQueueAddEntry` (`present`), and the panel's latch RATE is
    /// already enforced there, on the queueing thread, by [`pace_flip`](Self::pace_flip).
    /// So a wait that follows a present has nothing left to wait for, and parking it a full
    /// vblank anyway charges the title a second display period for one frame.
    ///
    /// MEASURED on a retail sports title whose display-queue callback is the GXM sample
    /// pattern (`sceDisplaySetFrameBuf`, then wait): that second period went into EVERY
    /// frame, on top of the two its own 30 Hz limiter asks for. Its game clock ran **2.99x
    /// its displayed frames**, dead steady, and it produced **2.95 seconds of sound for
    /// every second of picture** - which on a device fills the audio ring, drops a third of
    /// what the guest made, and starves on the next hitch, reading as a broken audio
    /// backend. Not parking after a present gives **1.99x** on both, i.e. exactly the two
    /// periods a 30 fps title spends, and cuts the run's scheduler suspends by 4.8x.
    ///
    /// A wait with NO present behind it still parks a vblank, and that is the case the
    /// unconditional park was really defending: a title that SPINS on this call livelocked
    /// when it returned at once (frame 3, 34.3 million thread resumes).
    #[inline]
    pub fn take_present_since_wait(&mut self) -> bool {
        self.flip_pacer.take_present_since_wait()
    }
}
