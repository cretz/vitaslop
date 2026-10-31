use super::*;

impl VitaState {
    /// `sceDisplaySetFrameBuf` named a buffer: it reaches the panel AT ONCE with an immediate
    /// sync, and at the next vblank edge otherwise - the edge `sceDisplayWaitSetFrameBuf` waits
    /// for.
    pub fn note_set_frame_buf(&mut self, immediate: bool) {
        let latch = if immediate {
            self.virtual_us
        } else {
            Self::at_or_after_vblank(
                self.virtual_us.saturating_add(1),
                crate::vita::display::VBLANK_US,
            )
        };
        self.display.set_frame_buf_latch(latch);
    }

    /// The latch edge of the last `sceDisplaySetFrameBuf`, taken: `None` when no buffer was
    /// named since the last wait, which keeps the older present-based rule for that case.
    ///
    /// >>> A WAIT AFTER A SetFrameBuf WAITS FOR THAT BUFFER'S LATCH, PRESENT OR NOT. The rule
    /// that a wait after a present has nothing left to wait for was written for a callback
    /// whose only wait is this one. A fighting title's display callback, in its 30 fps
    /// cinematics, is `SetFrameBuf`, `WaitSetFrameBuf`, then `WaitVblankStart` - two vblanks a
    /// frame on hardware, the latch and the one after it. Skipping the latch made it ONE, and
    /// the cinematics ran at 60 fps: twice their speed, each one cut away before its voice line
    /// finished (a 5-6 s line stopped 22% in, measured). Its fights' callback has no
    /// `WaitVblankStart` and stays at 60.
    pub fn take_frame_buf_latch(&mut self) -> Option<u64> {
        self.display.take_frame_buf_latch()
    }

    /// Record `SceGxmInitializeParams::displayQueueMaxPendingCount`. See [`Self::pace_flip`].
    pub fn set_display_queue_depth(&mut self, count: u32) {
        self.flip_pacer.set_queue_depth(count);
        tracing::debug!(
            target: "vitaslop::gxm",
            count,
            "sceGxmInitialize: displayQueueMaxPendingCount - how many frames the guest may run ahead of the scanout"
        );
    }

    /// `SceDisplaySetBufSync`: the buffer change takes effect immediately (and tears).
    pub const SETBUF_IMMEDIATE: u32 = 0;
    /// `SceDisplaySetBufSync`: the buffer change takes effect at the next vblank.
    pub const SETBUF_NEXTFRAME: u32 = 1;

    /// Record the `sync` argument of a `sceDisplaySetFrameBuf`.
    ///
    /// # It is RECORDED and REPORTED, and deliberately not acted on
    /// `SceDisplaySetBufSync` has exactly two values, IMMEDIATE and NEXTFRAME, and the
    /// difference between them is WHEN the scanout's framebuffer pointer changes - mid-
    /// scan, which tears, or at a vblank, which does not. It is not a swap interval, it
    /// cannot ask for 30 Hz, and it does not change how fast a title can present: the
    /// panel still scans at 60 Hz and the guest still has to draw each frame. We present
    /// whole frames, so we do not model tearing, and there is nothing left for the
    /// argument to change.
    ///
    /// Both behaviours it seemed to imply were TRIED and both were wrong, on a retail
    /// title that really does ask for IMMEDIATE:
    /// - dropping [`pace_flip`](Self::pace_flip)'s vblank floor for it re-created exactly
    ///   the bug the floor exists to fix, and
    /// - letting `sceDisplayWaitSetFrameBuf` return at once (nothing to wait for, if the
    ///   buffer changed when it was asked for) LIVELOCKED the title: it spins on that
    ///   call, and the run reached **frame 3** with **34.3 million thread resumes**, one
    ///   thread taking 25.7 million of them and burning 699 whole quanta. On hardware
    ///   that call waits for the DISPLAY to update, which is a vblank event whichever
    ///   sync mode set the buffer.
    ///
    /// So it is kept as a diagnostic: it says what the title asked for, which is worth
    /// knowing and was previously invisible.
    /// The Vita's panel, which is what the display controller scales a smaller buffer onto.
    pub const PANEL_W: u32 = 960;
    pub const PANEL_H: u32 = 544;

    /// Record the geometry the guest declared to `sceDisplaySetFrameBuf`. A zero in either
    /// axis is ignored rather than stored: it describes no buffer, and storing it would
    /// present nothing.
    #[inline]
    pub fn set_display_geometry(&mut self, w: u32, h: u32) {
        self.display.set_geometry(w, h)
    }

    /// The geometry the guest last declared. Render a frame at THIS size and scale it to
    /// [`Self::PANEL_W`] x [`Self::PANEL_H`] to present, which is what the hardware does.
    #[inline]
    pub fn display_size(&self) -> (u32, u32) {
        self.display.size()
    }

    /// Record the whole `SceDisplayFrameBuf` a `sceDisplaySetFrameBuf` declared, so
    /// `sceDisplayGetFrameBuf` can report the scanout the way the display controller
    /// holds it. Called with the six words as read from guest memory.
    #[inline]
    pub fn set_display_frame_buf(&mut self, fb: [u32; 6]) {
        self.display.set_frame_buf(fb)
    }

    /// The scanout description to report back, or `None` if the guest has never set one.
    /// A caller must NOT invent a buffer for the `None` case: before a title's first set,
    /// what is on screen belongs to the shell, and this engine has no shell.
    #[inline]
    pub fn display_frame_buf(&self) -> Option<[u32; 6]> {
        self.display.frame_buf()
    }

    pub fn set_display_sync(&mut self, sync: u32) {
        // Report the first call as well as every change, not changes alone. A run that
        // never logs is otherwise ambiguous between "the title asks for the default" and
        // "the title never called this at all", and those are different worlds: the
        // second means the presents are reaching the display down some other path and
        // the sync mode being honoured here is honouring nothing.
        if let Some(first) = self.display.set_sync(sync) {
            tracing::info!(
                target: "vitaslop::gxm",
                sync,
                first,
                "display: sceDisplaySetFrameBuf sync = {} ({})",
                sync,
                if sync == Self::SETBUF_IMMEDIATE {
                    "IMMEDIATE - the buffer change takes effect at once, so presents do not wait for vblank"
                } else {
                    "NEXTFRAME - presents latch at a vblank"
                },
            );
        }
    }

    /// The `sync` mode the guest last asked for. See [`Self::set_display_sync`].
    #[inline]
    pub fn display_sync(&self) -> u32 {
        self.display.sync()
    }

    /// The first vblank edge at or after `t`, on a grid of `period` starting at 0.
    #[inline]
    pub(super) fn at_or_after_vblank(t: u64, period: u64) -> u64 {
        at_or_after_vblank(t, period)
    }

    /// Park the calling thread until the `n`th vblank edge STRICTLY after now, and
    /// return the microseconds parked. `n` of 0 parks for nothing.
    ///
    /// The distinction from `sleep_park(n * period)` is the whole point: a vblank is a
    /// scanout heartbeat the guest joins, not a stopwatch it starts. Parking a full
    /// period from wherever the guest happened to call over-waits by half a period on
    /// average and can never phase-lock, so a title that renders in 5 ms and waits for
    /// vblank ran at 5 + 16.67 ms (46 fps) instead of the 60 the hardware gives it.
    /// Count one vblank-spin park - see [`report_vblank_spin_parks`].
    pub fn note_vblank_spin_park(&mut self) {
        VBLANK_SPIN_PARKS.fetch_add(1, std::sync::atomic::Ordering::Relaxed);
    }

    pub fn vblank_park(&mut self, n: u64, period: u64) -> u64 {
        if n == 0 {
            return 0;
        }
        // STRICTLY after: a wait issued exactly on an edge waits for the next one, or a
        // loop of them would return instantly forever and never advance the clock.
        let first = Self::at_or_after_vblank(self.virtual_us.saturating_add(1), period);
        let edge = first.saturating_add((n - 1).saturating_mul(period));
        let park = edge.saturating_sub(self.virtual_us);
        self.sleep_park(park);
        park
    }

    /// Guest stack for the display-callback thread. One callback runs at a time,
    /// so a single stack serves every invocation.
    pub(super) const DISPLAY_CB_STACK_BYTES: u32 = 0x1_0000;
    /// Callback-data copy slots. The queue is drained one entry per callback run;
    /// the ring only needs to cover entries submitted while earlier ones are still
    /// pending (the display queue itself is at most a few frames deep).
    pub(super) const DISPLAY_CB_SLOT_COUNT: u32 = 8;

    /// Queue the registered sceGxmInitialize display callback to run as guest code
    /// with this entry's `callback_data` (preemptive mode only).
    ///
    /// On hardware every `sceGxmDisplayQueueAddEntry` eventually runs the
    /// callback - typically `sceDisplaySetFrameBuf` plus the game's own buffer
    /// bookkeeping - on the display queue's internal thread. A double-buffered
    /// title waits (often a bare memory-poll loop, no host calls) for that
    /// bookkeeping to release the older buffer, so a host that never runs the
    /// callback stalls it after exactly two frames. GXM copies `callback_data`
    /// into the queue at AddEntry time; mirrored here via a slot ring so the
    /// caller can immediately reuse its buffer.
    pub fn enqueue_display_callback(&mut self, ctx: &mut GuestCtx, callback_data: u32) {
        if !self.preemptive || self.display_queue_cb == 0 {
            return;
        }
        let size = self.display_queue_cb_data_size.max(4);
        let ceiling = self.heap_ceiling();
        let (arena, stacks) = (&mut self.arena, &self.guest_stacks);
        if !self.display_callbacks.ensure(
            size,
            Self::DISPLAY_CB_SLOT_COUNT,
            Self::DISPLAY_CB_STACK_BYTES,
            |bytes, align| arena.galloc(bytes, align, ceiling),
            |lo, hi| stacks.note(lo, hi),
        ) {
            self.display_queue_cb = 0; // OOM: disable rather than corrupt
            return;
        }
        let slot = self
            .display_callbacks
            .take_slot(size, Self::DISPLAY_CB_SLOT_COUNT);
        if callback_data != 0 {
            let bytes = ctx.read_bytes(callback_data, size as usize);
            ctx.write_bytes(slot, &bytes);
        }
        // >>> A QUEUE DEEPER THAN THE SLOT RING IS SILENT DATA CORRUPTION.
        //
        // The slot the entry just took is reused after `DISPLAY_CB_SLOT_COUNT` more entries,
        // so a queue that long hands an older, still-pending callback a newer entry's bytes.
        // `pace_flip` charges the shortfall back to the queueing thread, which keeps the depth
        // at the guest's own `displayQueueMaxPendingCount`; if it is ever exceeded anyway, the
        // guest's display bookkeeping is being lied to and only this says so.
        if self
            .display_callbacks
            .first_overrun(Self::DISPLAY_CB_SLOT_COUNT)
        {
            tracing::warn!(
                target: "vitaslop::gxm",
                pending = self.display_callbacks.queued(),
                slots = Self::DISPLAY_CB_SLOT_COUNT,
                "the display-queue CALLBACK backlog is as deep as its data ring, so a pending callback will be handed a newer entry's callbackData. The guest's own display bookkeeping runs on that data. Reported once."
            );
        }
        self.display_callbacks.push(slot);
        self.pump_display_callback();
    }

    /// How many display-queue entries are waiting for their guest callback, including the
    /// one in flight. See [`Self::enqueue_display_callback`]; GXM runs the callback once per
    /// entry, so anything but a small constant here is a backlog.
    pub fn display_callback_backlog(&self) -> usize {
        self.display_callbacks.queued() + usize::from(self.display_callbacks.running().is_some())
    }

    /// The thread running the display-queue callback right now, if one is.
    #[inline]
    pub(super) fn display_callback_thread(&self) -> Option<i32> {
        self.display_callbacks.running()
    }

    /// Start the next queued display callback if none is in flight. Chained from
    /// [`Self::set_thread_exit`] when the in-flight one ends, preserving
    /// submission order.
    /// `sceGxmDisplayQueueFinish`: whether the caller must park until every display-queue
    /// entry submitted so far has had its callback run. Parks it (the caller returns
    /// [`SvcOutcome::Block`]) when one is queued or running; the last one to finish wakes it.
    ///
    /// >>> RETURNING AT ONCE WAS A USE-AFTER-FREE. The callback is the title's own code and
    /// > > > reads the entry's callback data, which can name the title's buffers; Finish is how a
    /// > > > title knows no callback still holds one before it frees it. MEASURED (a golf title,
    /// > > > a mode change): Finish returned with a callback still queued, the title freed a
    /// > > > 192x192 RGB565 image and re-allocated the block, the queued callback then wrote the
    /// > > > image's pixels over the new chunk's heap header, and the next `free` faulted inside
    /// > > > the system C library.
    ///
    /// Never parks the callback thread itself: a Finish from inside the callback would wait
    /// on its own return.
    pub fn display_queue_finish(&mut self) -> bool {
        if !self.display_callbacks.busy() {
            return false;
        }
        if self.display_callbacks.running() == Some(self.current) {
            return false;
        }
        tracing::debug!(
            target: "vitaslop::display",
            thid = self.current,
            queued = self.display_callbacks.queued(),
            running = self.display_callbacks.running().is_some(),
            frame = self.flip_count(),
            "sceGxmDisplayQueueFinish waits for the display callbacks still pending"
        );
        self.display_callbacks.park_until_idle(self.current);
        true
    }

    pub(super) fn pump_display_callback(&mut self) {
        let next_uid = &mut self.next_uid;
        let Some((data, stack_top, thid)) = self
            .display_callbacks
            .start_next(Self::DISPLAY_CB_STACK_BYTES, || {
                let uid = *next_uid;
                *next_uid += 1;
                uid
            })
        else {
            return;
        };
        self.pending_spawns.push(Reentry {
            entry: self.display_queue_cb & !1,
            arg_len: data, // r0: the callback's `const void *callbackData`
            arg_ptr: 0,
            r2: 0,
            r3: 0,
            stack_top,
            thid,
            // Above game threads, so a queued frame is presented promptly.
            //
            // >>> RAISING THIS TO `SCE_KERNEL_HIGHEST_PRIORITY_USER` WAS TRIED AND REVERTED.
            // GXM creates this thread itself and the SDK's samples put it at the top of the
            // user band, so that looked more faithful - and it MEASURED as no change at all:
            // The role-playing title's callback rate stayed at one per two vblanks (it is the title's own
            // TWO `sceDisplayWaitVblankStart` calls that set it, see `pace_flip`), and a
            // racer's race was identical either way. What it did do is move the schedule, so
            // it is not carried on an argument alone.
            priority: DEFAULT_THREAD_PRIORITY - 0x10,
        });
    }

    /// Guest stack for the NGS callback thread. One callback runs at a time.
    pub(super) const NGS_CB_STACK_BYTES: u32 = 0x1_0000;
    /// `SceNgsCallbackInfo` is seven words: voice, rack, module id, callback data,
    /// callback data 2, a callback pointer, and the user data.
    ///
    /// EVIDENCE for the two fields past the first four, from the callback one title
    /// registers on its player module: it reads its OWN object out of `info + 0x18` (and
    /// dereferences it, so that is the user data it registered) and forwards `info + 0x10`
    /// to its buffer handler beside the constant it uses for "a buffer completed" - the
    /// second callback datum, the buffer index. A five-word layout with the user data at
    /// `+0x10` would hand that title a buffer index where it expects its object.
    pub(super) const NGS_CB_INFO_BYTES: u32 = 0x1c;
    /// Info slots in the ring. Callbacks are short and drain serially; the ring only has to
    /// cover the events one mix can raise before the first callback of the batch runs.
    pub(super) const NGS_CB_SLOT_COUNT: u32 = 32;

    /// Queue a guest NGS callback `entry(const SceNgsCallbackInfo *)` with `info` as the
    /// struct's bytes, to run as guest code on the NGS callback thread (preemptive mode
    /// only - a run-to-completion host has no thread to run it on).
    ///
    /// On the device the synthesizer raises these from `sceNgsSystemUpdate` - a player
    /// finishing a buffer, a voice finishing its data - and a streaming title does its
    /// refills IN the callback. So without delivery the stream primes two slots, plays them,
    /// and ends; with it, the title keeps the ring full. Serial and in order, the way one
    /// update thread would run them.
    pub fn enqueue_ngs_callback(&mut self, ctx: &mut GuestCtx, entry: u32, info: &[u8]) {
        if !self.preemptive || entry == 0 {
            return;
        }
        if self.ngs_callbacks.is_off() {
            return;
        }
        let ceiling = self.heap_ceiling();
        let (arena, stacks) = (&mut self.arena, &self.guest_stacks);
        if !self.ngs_callbacks.ensure(
            Self::NGS_CB_INFO_BYTES,
            Self::NGS_CB_SLOT_COUNT,
            Self::NGS_CB_STACK_BYTES,
            |bytes, align| arena.galloc(bytes, align, ceiling),
            |lo, hi| stacks.note(lo, hi),
        ) {
            tracing::warn!(
                target: "vitaslop::ngs",
                "NGS callback delivery is OFF: the guest heap could not spare a stack and \
                 an info ring for it, so every module/finished callback from here on is \
                 dropped and a streaming voice will end after its primed buffers"
            );
            return;
        }
        let slot = self
            .ngs_callbacks
            .take_slot(Self::NGS_CB_INFO_BYTES, Self::NGS_CB_SLOT_COUNT);
        let mut bytes = [0u8; 0x1c];
        let n = info.len().min(bytes.len());
        bytes[..n].copy_from_slice(&info[..n]);
        ctx.write_bytes(slot, &bytes);
        self.ngs_callbacks.push((entry, slot));
        self.pump_ngs_callback();
    }

    /// Whether an NGS callback is queued or running.
    pub fn ngs_callbacks_pending(&self) -> bool {
        self.ngs_callbacks.busy()
    }

    /// Info slots in the [`NgsSyncStash`] ring: more than any one grain raises (a few per
    /// streaming voice); a batch past it goes to the serial thread instead.
    pub(super) const NGS_SYNC_SLOTS: u32 = 64;

    /// Stash the callbacks one `sceAudioOutOutput` grain raised, to run on the CALLING thread
    /// before it returns - see `vita::audio::out_output`. Each gets its own info slot, so no
    /// two calls of one batch are the same [`GuestCall`] (the replay matches results by call).
    /// False - nothing stashed, the caller delivers them the old way - outside the preemptive
    /// engine, for a batch past the ring, or when the heap cannot spare the ring.
    pub fn ngs_sync_stash(&mut self, ctx: &mut GuestCtx, calls: &[(u32, [u8; 0x1c])]) -> bool {
        if !self.preemptive || calls.len() as u32 > Self::NGS_SYNC_SLOTS {
            return false;
        }
        if calls.is_empty() {
            return true;
        }
        if self.ngs_sync.needs_ring() {
            let base = self.galloc(Self::NGS_CB_INFO_BYTES * Self::NGS_SYNC_SLOTS, 8);
            self.ngs_sync.set_ring(base);
        }
        if self.ngs_sync.is_off() {
            return false;
        }
        let mut batch = Vec::with_capacity(calls.len());
        for (entry, info) in calls {
            let slot = self
                .ngs_sync
                .take_slot(Self::NGS_CB_INFO_BYTES, Self::NGS_SYNC_SLOTS);
            ctx.write_bytes(slot, info);
            batch.push((*entry, slot));
        }
        self.ngs_sync.stash(self.current, batch);
        true
    }

    /// Whether this dispatch is a REPLAY of a `sceAudioOutOutput` still running its callbacks.
    #[inline]
    pub fn ngs_sync_replaying(&self) -> bool {
        self.ngs_sync.is_replaying(self.current)
    }

    /// Run the current thread's stashed callbacks as guest calls, in order. True when every one
    /// has returned (the stash is gone); false when one is being run - the handler returns
    /// [`crate::SvcOutcome::CallGuest`] and is dispatched again when it comes back.
    pub fn ngs_sync_run(&mut self) -> bool {
        let Some(batch) = self.ngs_sync.batch(self.current) else {
            return true;
        };
        for (entry, slot) in batch {
            if self
                .call_guest(
                    GuestCall {
                        entry,
                        args: vec![slot],
                    },
                    None,
                )
                .is_none()
            {
                return false;
            }
        }
        self.ngs_sync.finish(self.current);
        true
    }

    /// Start the next queued NGS callback if none is in flight. Chained from
    /// [`Self::set_thread_exit`] when the in-flight one ends.
    pub(super) fn pump_ngs_callback(&mut self) {
        let next_uid = &mut self.next_uid;
        let Some(((entry, info), stack_top, thid)) = self
            .ngs_callbacks
            .start_next(Self::NGS_CB_STACK_BYTES, || {
                let uid = *next_uid;
                *next_uid += 1;
                uid
            })
        else {
            return;
        };
        self.pending_spawns.push(Reentry {
            entry: entry & !1,
            arg_len: info, // r0: `const SceNgsCallbackInfo *`
            arg_ptr: 0,
            r2: 0,
            r3: 0,
            stack_top,
            thid,
            // Above the game's threads, as the synthesizer's own update runs: a refill that
            // waits behind a frame of game work is a refill that arrives after the ring ran
            // dry.
            priority: DEFAULT_THREAD_PRIORITY - 0x10,
        });
    }

    /// Guest stack bytes for a one-shot service-state callback delivery.
    pub(super) const SERVICE_CB_STACK_BYTES: u32 = 0x8000;

    /// Record the SceNpManager service-state callback a title registered
    /// (`sceNpRegisterServiceStateCallback`); pumped by
    /// [`Self::pump_np_callback`].
    #[inline]
    pub fn set_np_service_callback(&mut self, entry: u32, userdata: u32) {
        self.np_service_cb.register(entry, userdata);
    }

    /// Record the SceNetCtl inet-state callback a title registered
    /// (`sceNetCtlInetRegisterCallback`).
    #[inline]
    pub fn set_net_inet_callback(&mut self, entry: u32, userdata: u32) {
        self.net_inet_cb.register(entry, userdata);
    }

    /// Spawn a one-shot guest callback `entry(r0, r1, r2)` on its own fresh stack
    /// (preemptive mode). Used to deliver the service-state notifications a title
    /// waits for; the callback typically just records the state into a global the
    /// boot state machine polls. Returns false if not spawned (single-thread mode
    /// or heap exhausted).
    pub(super) fn spawn_oneshot_callback(&mut self, entry: u32, r0: u32, r1: u32, r2: u32) -> bool {
        if !self.preemptive || entry == 0 {
            return false;
        }
        let stack = self.galloc(Self::SERVICE_CB_STACK_BYTES, 16);
        if stack == 0 {
            return false;
        }
        self.guest_stacks
            .note(stack, stack + Self::SERVICE_CB_STACK_BYTES);
        let thid = self.new_uid();
        self.pending_spawns.push(Reentry {
            entry: entry & !1,
            arg_len: r0,
            arg_ptr: r1,
            r2,
            r3: 0,
            stack_top: stack + Self::SERVICE_CB_STACK_BYTES,
            thid,
            priority: DEFAULT_THREAD_PRIORITY,
        });
        true
    }

    /// Deliver any registered-but-undelivered service-state callback (called from
    /// the per-frame `sce*CheckCallback` pumps). Each fires exactly once, with the
    /// offline state the title expects: NP signed-out, net disconnected. Without
    /// this a title that gates its boot on the notification waits forever.
    ///
    /// `state` is the `SceNpServiceState` enum to deliver (signed-out offline).
    pub fn pump_np_callback(&mut self, state: u32) {
        if let Some((entry, userdata)) = self.np_service_cb.undelivered() {
            // The registered NP callback is a C++ member-function thunk taking its
            // `this`/userdata in r2, the service-state enum in r0, and a second
            // reserved argument in r1 (verified by disassembling the game's handler,
            // which does `switch(r0)` on the state): callback(state, 0, userdata).
            if self.spawn_oneshot_callback(entry, state, 0, userdata) {
                self.np_service_cb.mark_delivered();
            }
        }
    }

    /// `sceKernelCreateCallback`: a new callback owned by the calling thread.
    pub fn kcb_create(&mut self, name: String, func: u32, common: u32) -> i32 {
        let uid = self.new_uid();
        let owner = self.current;
        tracing::debug!(target: "vitaslop::cb", uid = format_args!("{uid:#x}"), owner = format_args!("{owner:#x}"), %name, func = format_args!("{func:#x}"), "sceKernelCreateCallback");
        self.kcallbacks.create(uid, owner, name, func, common);
        uid
    }

    /// `sceKernelDeleteCallback`. False for an unknown uid.
    #[inline]
    pub fn kcb_delete(&mut self, uid: i32) -> bool {
        self.kcallbacks.delete(uid)
    }

    /// `sceKernelNotifyCallback`: one more notification, `arg` the latest argument.
    #[inline]
    pub fn kcb_notify(&mut self, uid: i32, arg: u32) -> bool {
        self.kcallbacks.notify(uid, arg)
    }

    /// `sceKernelCancelCallback`: drop the pending notifications.
    #[inline]
    pub fn kcb_cancel(&mut self, uid: i32) -> bool {
        self.kcallbacks.cancel(uid)
    }

    /// `sceKernelGetCallbackCount`, vblank notifications included.
    #[inline]
    pub fn kcb_count(&mut self, uid: i32) -> Option<u32> {
        self.kcallbacks
            .count(uid, self.virtual_us, crate::vita::display::VBLANK_US)
    }

    /// `sceDisplayRegisterVblankStartCallback` (`on`) / `...Unregister...` (`!on`).
    #[inline]
    pub fn kcb_vblank(&mut self, uid: i32, on: bool) -> bool {
        self.kcallbacks.set_vblank(uid, on, self.virtual_us)
    }

    /// Deliver ONE pending callback of the calling thread: spawn it as `func(uid, count,
    /// arg, common)` at the caller's priority and park the caller until it returns, so the
    /// callback runs on the owner's behalf and the owner does not continue past its callback
    /// point meanwhile. Returns whether one was delivered (the caller then returns `Block`).
    pub fn kcb_deliver_one(&mut self) -> bool {
        if !self.preemptive {
            return false;
        }
        let owner = self.current;
        let Some((uid, func, count, arg, common)) =
            self.kcallbacks
                .take_pending(owner, self.virtual_us, crate::vita::display::VBLANK_US)
        else {
            return false;
        };
        let stack = self.galloc(Self::SERVICE_CB_STACK_BYTES, 16);
        if stack == 0 || func == 0 {
            return false;
        }
        self.guest_stacks
            .note(stack, stack + Self::SERVICE_CB_STACK_BYTES);
        let thid = self.new_uid();
        let priority = self.current_priority();
        self.pending_spawns.push(Reentry {
            entry: func & !1,
            arg_len: uid as u32,
            arg_ptr: count,
            r2: arg,
            r3: common,
            stack_top: stack + Self::SERVICE_CB_STACK_BYTES,
            thid,
            priority,
        });
        self.kcallbacks.note_delivery(thid, uid);
        self.join_waiters.push((owner, thid, 0));
        true
    }

    /// Deliver the net inet-state callback once with `event` (a `SceNetCtlState`).
    /// The SceNetCtl callback ABI is `cb(int event_type, void *arg)`: r0=event,
    /// r1=arg (the registered userdata), r2 unused.
    pub fn pump_net_callback(&mut self, event: u32) {
        if let Some((entry, userdata)) = self.net_inet_cb.undelivered()
            && self.spawn_oneshot_callback(entry, event, userdata, 0)
        {
            self.net_inet_cb.mark_delivered();
        }
    }

    /// The earliest pending timed-wake deadline across every timed blocking wait -
    /// lightweight-cond, heavyweight cond, semaphore, event flag - and pure sleeps.
    /// The scheduler uses this to jump the clock forward when every thread is blocked,
    /// instead of declaring a deadlock. A timed wait omitted here would be turned into
    /// an infinite park (it could never wake by timeout), so every timed-wait kind
    /// must contribute its deadline.
    pub fn earliest_lwcond_deadline(&self) -> Option<u64> {
        self.earliest_deadline_owner().map(|(d, _)| d)
    }

    /// The earliest pending timed-wake deadline AND who is waiting on it.
    ///
    /// The owner is what [`idle_attribution`](Self::idle_attribution) is built on: an idle
    /// jump is game time nobody executed for, and a total alone cannot say which of a
    /// title's threads bought it.
    pub fn earliest_deadline_owner(&self) -> Option<(u64, IdleOwner)> {
        let sleepers = self.sleep_waiters.iter().map(|&(thid, d)| {
            (
                d,
                IdleOwner {
                    kind: IdleKind::Sleep,
                    thid,
                },
            )
        });
        self.sync.earliest_deadline_owner(sleepers)
    }

    /// Where the idle-path clock time went, as `(owner, microseconds, jumps)`, largest first.
    ///
    /// # Why this exists
    /// `clock_sources` splits the game clock into quanta / top-up / idle, and on one retail
    /// title **the idle bucket is 80% of the clock**: 4.0 s of idle against 1.0 s of quanta
    /// per 100 displayed frames, for frames worth 1.67 s. The clock therefore ran 3.0x the
    /// display, dead steady - which is what a title paces its AUDIO off, so it produced three
    /// times as much sound as the picture called for and the phone's ring both overran and
    /// starved. The split named the PATH and stopped there; nothing said which wait was
    /// buying the time, and "the scheduler jumped to the earliest deadline" is true of every
    /// healthy run as well.
    ///
    /// An idle jump is the one kind of game time no guest instruction was executed for, so
    /// it is also the one kind that can be wrong without any guest code being wrong. Naming
    /// the waiting thread and the wait FAMILY turns that from an inference into a lookup.
    pub fn idle_attribution(&self) -> Vec<(IdleOwner, u64, u64)> {
        let mut v = self.idle_by_owner.clone();
        v.sort_by_key(|e| std::cmp::Reverse(e.1));
        v
    }

    /// Take the `r0` value owed to thread `thid` when it resumes from a block (a timed
    /// wait that expired), if any. Drained once, by the engine, at the point the thread
    /// resumes; `None` (a signal wake) leaves the thread's pre-park return value intact.
    pub fn take_resume_code(&mut self, thid: i32) -> Option<u32> {
        let idx = self
            .pending_resume_codes
            .iter()
            .position(|&(t, _)| t == thid)?;
        Some(self.pending_resume_codes.remove(idx).1)
    }

    /// Advance the virtual clock to at least `to_us` and wake every timed wait whose
    /// deadline has now passed. Called by the scheduler when no thread is runnable but
    /// a timed wait can still fire. Every kind of timed *wait* (lightweight cond,
    /// heavyweight cond, semaphore, event flag) is woken with
    /// `SCE_KERNEL_ERROR_WAIT_TIMEOUT` delivered through the resume-code channel; a
    /// pure `sceKernelDelayThread` sleep instead completes with success (0), since a
    /// delay elapsing is not a timeout. A timed-out cond wait additionally re-acquires
    /// its mutex before resuming (it may re-block on the mutex first).
    pub fn advance_time_to(&mut self, to_us: u64) {
        // WHERE the game clock's time came from, cumulatively. See
        // [`clock_sources`](Self::clock_sources): a total alone cannot say whether one
        // engine's clock runs slower because it charges less per quantum or because it
        // never reaches the idle path that JUMPS the clock to the next deadline, and
        // those have nothing in common as bugs.
        let gained = to_us.saturating_sub(self.virtual_us);
        match self.clock_source {
            ClockSource::Quantum => self.clock_from_quanta_us += gained,
            ClockSource::FrameTopup => self.clock_from_topup_us += gained,
            ClockSource::Wall => self.clock_from_wall_us += gained,
            ClockSource::Idle => {
                self.clock_from_idle_us += gained;
                // ...and WHO it was gained for. The owner is looked up before the deadline
                // lists are drained below, because draining is what removes it.
                if let Some((_, owner)) = self.earliest_deadline_owner() {
                    match self.idle_by_owner.iter_mut().find(|(o, _, _)| *o == owner) {
                        Some(slot) => {
                            slot.1 += gained;
                            slot.2 += 1;
                        }
                        // Bounded by (wait kinds) x (live threads), which is tens - a title
                        // that somehow exceeds that would be reporting a thread leak anyway.
                        None => self.idle_by_owner.push((owner, gained, 1)),
                    }
                }
            }
        }
        // Back to the default immediately: the tag describes THIS call only, and letting
        // it persist would credit the next untagged advance to whichever path happened to
        // run last - which is precisely the kind of quiet misattribution this split exists
        // to prevent.
        self.clock_source = ClockSource::Idle;
        self.virtual_us = self.virtual_us.max(to_us);
        let now = self.virtual_us;
        // Every wake pushed below is released by TIME - see `take_timeout_wakes`.
        let wakes_before = self.pending_wakes.len();
        // The sync objects' timed waits expire in two halves with the pure sleeps between
        // them, which is the order the scheduler has always seen the wakes in.
        {
            let mut cx = sched!(self);
            self.sync.expire_locks(&mut cx, &mut self.timeout_wakes);
        }
        // A pure sleep (sceKernelDelayThread / audio grain pacing) that elapses is a
        // successful completion, not a timed-out wait: wake it with its return value
        // (0) unchanged - no resume code.
        let mut slept: Vec<i32> = Vec::new();
        self.sleep_waiters.retain(|&(thid, deadline)| {
            if deadline <= now {
                push_wake!(self, thid);
                slept.push(thid);
                false
            } else {
                true
            }
        });
        // A timed `sceNetEpollWait` that ran out is no longer a waiter an abort can interrupt.
        self.net.forget_timed_out_epoll_waiters(&slept);
        let clock = (
            self.clock_from_quanta_us,
            self.clock_from_wall_us,
            self.clock_from_idle_us,
        );
        let frame = self.cur_frame();
        {
            let mut cx = sched!(self);
            self.sync
                .expire_waits(&mut cx, &mut self.timeout_wakes, clock, frame);
        }
        if self.pending_wakes.len() > wakes_before {
            let woke = self.pending_wakes[wakes_before..].to_vec();
            self.timeout_wakes.extend(woke);
        }
    }
}
