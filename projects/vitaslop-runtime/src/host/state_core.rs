use super::*;

impl VitaState {
    /// Tell the texture cache that the ENGINE just wrote `bytes` into guest memory at `addr`,
    /// so no draw has to discover it. See [`TextureSnapshots::author`] - the contract is that
    /// `bytes` is exactly what guest memory now holds there.
    pub(crate) fn author_texture_bytes(&mut self, ctx: &GuestCtx, addr: u32, bytes: Arc<[u8]>) {
        snaps!(self).author(ctx, addr, bytes);
    }

    /// New state for a run over `[base, base + mem_bytes)`. Allocations start
    /// above the image (at base + 1 MiB) and grow up, well below the stack that
    /// starts at the top of the region.
    pub fn new(base: u32, mem_bytes: u32, world: Box<dyn World + Send>) -> Self {
        // The main thread's stack descends from the top of the guest region through the reserve
        // `galloc` keeps clear of it - see `MAIN_STACK_RESERVE_BYTES`.
        let guest_stacks = GuestStacks::default();
        let top = base.wrapping_add(mem_bytes);
        guest_stacks.note(top.wrapping_sub(Self::MAIN_STACK_RESERVE_BYTES), top);
        VitaState {
            texture_snapshots: Arc::new(std::sync::Mutex::new(TextureSnapshots::with_stacks(
                guest_stacks.clone(),
            ))),
            guest_stacks,
            yuv_profiles: [0, 0],
            precomputed: PrecomputedRecords::default(),
            complete_scene_now: None,
            complete_scene_async: false,
            pending_early: None,
            base,
            mem_bytes,
            arena: GuestArena::new(base + 0x0010_0000),
            next_handle: HANDLE_BASE + 1,
            next_uid: 0x100,
            storage: StorageClock::default(),
            charge_rem: (0, 0),
            cpu_charged_since_flip: 0,
            flip_pacer: FlipPacer::default(),
            display: DisplayController::new(
                (Self::PANEL_W, Self::PANEL_H),
                Self::SETBUF_NEXTFRAME,
            ),
            flip_candidates: std::collections::HashSet::new(),
            exec_request: None,
            main_signals: 0,
            quantum_count: 0,
            shader_patcher: ShaderPatcherRecords::default(),
            program_memos: ProgramMemos::default(),
            precompile: PrecompileQueue::default(),
            color_surfaces: ColorSurfaces::default(),
            display_queue_cb: 0,
            display_queue_cb_data_size: 0,
            scene: None,
            gxm_context: 0,
            deferred_contexts: FxHashMap::default(),
            command_lists: Vec::new(),
            reported_no_gxm_context: false,
            reported_draw_without_textures: false,
            texture_writes: TextureSlotWrites::default(),
            last_draw_was_precomputed: false,
            last_precomputed_block: 0,
            pdraw_binding: None,
            deferred_geometry: Vec::new(),
            pending_geometry: Vec::new(),
            flip_resolve_upto: None,
            deferred_serial: 0,
            sa_bank: 0,
            bound_binding_scratch: Vec::new(),
            threads: Vec::new(),
            free_stacks: Vec::new(),
            runaway_threads_reported: false,
            reported_no_video: false,
            reported_unit_layout: false,
            movie: None,
            avplayers: Default::default(),
            avcdec: crate::vita::avcdec::AvcdecState::default(),
            mspace_exhausted: std::collections::HashSet::new(),
            pending_reentry: None,
            guest_calls: GuestCalls::default(),
            mp4_io_scratch: 0,
            sync: SyncObjects::default(),
            timers: KernelTimers::default(),
            open_dialogs: 0,
            dialog_opened_flip: [0; 16],
            fios_overlays: FiosOverlays::default(),
            gpo: 0,
            touch: Default::default(),
            preemptive: false,
            current: MAIN_THID,
            signal_waiters: Vec::new(),
            display_callbacks: CallbackQueue::default(),
            ngs_callbacks: CallbackQueue::default(),
            ngs_sync: NgsSyncStash::default(),
            np_service_cb: OneShotCallback::default(),
            net_inet_cb: OneShotCallback::default(),
            join_waiters: Vec::new(),
            kcallbacks: KernelCallbacks::default(),
            thread_events: ThreadEvents::default(),
            pending_spawns: Vec::new(),
            spawn_near: Vec::new(),
            pending_wakes: Vec::new(),
            pending_terminations: Vec::new(),
            timeout_wakes: Vec::new(),
            wake_causes: Vec::new(),
            trace_wakes: false,
            pending_stat_writes: Vec::new(),
            pending_resume_codes: Vec::new(),
            fs: FileTable::new(),
            addcont: Vec::new(),
            savedata: SaveDataStore::default(),
            safe_memory: Vec::new(),
            safe_memory_dirty: false,
            trophies: crate::trophy::TrophyStore::default(),
            mspaces: crate::mspace::MspaceStore::default(),
            fonts: crate::font::FontLibrary::default(),
            capture: Capture::new(),
            world,
            audio: Box::new(crate::audio::NullSink::default()),
            video: Box::new(vitaslop_platform::video::NoVideo),
            audio_dec: Box::new(vitaslop_platform::audio_dec::NoAudioDecode),
            audio_state: crate::vita::audio::AudioState::default(),
            voice: crate::vita::voice::VoiceState::default(),
            jpeg: crate::vita::jpeg::JpegState::default(),
            program_refs: ProgramRefs::default(),
            audiodec: crate::vita::audiodec::AudiodecState::default(),
            location: crate::vita::location::LocationState::default(),
            halt_on_terminate: false,
            process_param: 0,
            main_priority: DEFAULT_THREAD_PRIORITY,
            main_affinity: 0,
            modules: Vec::new(),
            unhandled_imports: Vec::new(),
            tls: ThreadLocalStorage::default(),
            bound_textures: Vec::new(),
            bound_vertex_textures: Vec::new(),
            vertex_texture_span: [0; crate::vita::gxmctx::TEXTURE_ARRAY_BYTES as usize],
            tex_scene_seq: 0,
            deferred_tex_last: None,
            deferred_vert_last: None,
            vertex_texture_gen: 0,
            render_state_memo: None,
            uniform_float_scratch: Vec::new(),
            textures: TextureRegistry::default(),
            texture_unit_scratch: Vec::new(),
            snaps_frames_pending: std::sync::atomic::AtomicU32::new(0),
            async_resolve: Arc::default(),
            uniform_ring_wrapped: false,
            reported_reserve_without_context: false,
            reported_stale_uniforms: FxHashMap::default(),
            mem_windows_zero_tested: FxHashSet::default(),
            frame_raw_targets: Vec::new(),
            notification_region: 0,
            visibility: VisibilityQueries::default(),
            gxm_mappings: std::collections::BTreeMap::new(),
            render_targets: RenderTargets::default(),
            photo_exports: 0,
            net: NetState::default(),
            motion: Default::default(),
            power: Default::default(),
            shared_fb: None,
            xml: Default::default(),
            np: Default::default(),
            net_libs: Default::default(),
            ime: Default::default(),
            gesture: Default::default(),
            fibers: FiberTable::default(),
            sleep_waiters: Vec::new(),
            wall_waiters: Vec::new(),
            yields: YieldElision::default(),
            virtual_us: 0,
            clock_source: ClockSource::Idle,
            clock_from_quanta_us: 0,
            clock_from_topup_us: 0,
            clock_from_idle_us: 0,
            clock_from_wall_us: 0,
            wall_floor: None,
            idle_by_owner: Vec::new(),
            cur_frame: 0,
            ctrl: Default::default(),
        }
    }

    /// The current display-frame index (set by the scheduler each flip). Read by the
    /// egress ledger to frame-tag events and by an observation harness sampling per
    /// frame.
    pub fn cur_frame(&self) -> u64 {
        self.cur_frame
    }

    /// Set the current display-frame index. Called by the scheduler's frame boundary
    /// so egress events and per-frame samples carry the right frame number.
    pub fn set_cur_frame(&mut self, frame: u64) {
        self.cur_frame = frame;
    }

    /// Record a registered shader program (its guest `SceGxmProgram*` header) and
    /// return the opaque `shaderPatcherId` handle the guest will pass around.
    pub fn register_shader_program(&mut self, program_header: u32) -> u32 {
        let handle = self.new_handle();
        self.shader_patcher.register(handle, program_header);
        // A registration is the one moment a header address can begin to mean a
        // different program (the guest may have freed and reallocated the blob), so
        // nothing reflected earlier may be trusted past it.
        self.invalidate_program_reflection();
        handle
    }

    /// The `SceGxmProgram*` header for a `shaderPatcherId`, or 0 if unknown.
    #[inline]
    pub fn shader_program(&self, id: u32) -> u32 {
        self.shader_patcher.registered(id)
    }

    /// Store the shader patcher's user-data word, replacing any previous one.
    #[inline]
    pub fn set_shader_patcher_user_data(&mut self, patcher: u32, user_data: u32) {
        self.shader_patcher.set_user_data(patcher, user_data);
    }

    /// Read it back. A patcher that was never given one holds NULL, which is what GXM
    /// hands back for a patcher whose `SceGxmShaderPatcherParams.userData` was null.
    #[inline]
    pub fn shader_patcher_user_data(&self, patcher: u32) -> u32 {
        self.shader_patcher.user_data(patcher)
    }

    /// Record the guest address of the main module's `SceProcessParam` (from
    /// [`crate::link::LinkedProgram::process_param`]) so `sceKernelGetProcessParam`
    /// can hand it to libc. Set once before the run.
    pub fn set_process_param(&mut self, addr: u32) {
        self.process_param = addr;
    }

    /// Record what the title asks its main thread to run at
    /// ([`crate::link::LinkedProgram::main_thread_request`]); `None` keeps the default. Set
    /// once before the run, beside [`Self::set_process_param`]. Every scheduler starts the
    /// main thread at [`Self::main_thread_priority`], and a thread the main thread creates
    /// with mask 0 inherits [`Self::main_affinity`].
    pub fn set_main_thread_request(&mut self, request: (Option<i32>, Option<i32>)) {
        if let Some(p) = request.0 {
            self.main_priority = p;
        }
        if let Some(m) = request.1 {
            self.main_affinity = m;
        }
    }

    /// The priority the main thread starts at - [`DEFAULT_THREAD_PRIORITY`] unless the title's
    /// `SceProcessParam` names one.
    pub fn main_thread_priority(&self) -> i32 {
        self.main_priority
    }

    /// The main thread's CPU affinity mask (0 = every user core).
    pub fn main_affinity(&self) -> i32 {
        self.main_affinity
    }

    /// Record the linked image's modules so the kernel's module queries can answer
    /// from them. Set once before the run, alongside [`Self::set_process_param`].
    pub fn set_modules(&mut self, modules: Vec<crate::link::LoadedModule>) {
        self.modules = modules;
    }

    /// Record the linked image's unhandled function imports, for the unimplemented-NID report.
    pub fn set_unhandled_imports(&mut self, imports: Vec<(u32, u32)>) {
        self.unhandled_imports = imports;
    }

    /// The linked image's unhandled function imports - see [`Self::set_unhandled_imports`].
    pub fn unhandled_imports(&self) -> &[(u32, u32)] {
        &self.unhandled_imports
    }

    /// The module whose segments contain `addr`, and its SceUID, or `None` if the
    /// address is in no loaded module (guest heap, a stack, a memory block).
    ///
    /// The UID is the module's index plus a base: modules are placed once at link
    /// time and never unloaded here, so the index is a stable identity, and offsetting
    /// it keeps a module id from colliding with the small ids threads and sync objects
    /// use.
    pub fn module_by_addr(&self, addr: u32) -> Option<(i32, &crate::link::LoadedModule)> {
        self.modules
            .iter()
            .position(|m| m.contains(addr))
            .map(|i| (MODULE_UID_BASE + i as i32, &self.modules[i]))
    }

    /// The `SceProcessParam` address (0 if the title carries none).
    pub fn process_param(&self) -> u32 {
        self.process_param
    }

    /// Record the main executable's TLS template (from
    /// [`crate::link::LinkedProgram::tls_template`]) so each thread can be given its
    /// own thread-local-storage block. Set once before the run.
    #[inline]
    pub fn set_tls_template(&mut self, template: (u32, u32, u32)) {
        self.tls.set_template(template)
    }

    /// The thread pointer (`TPIDRURO`) for thread `thid`: the base of its private TLS
    /// block, allocating the block on first request. The block is `memsz` bytes; its
    /// `.tbss` tail is zero (guest memory starts zeroed) and its `.tdata` head is
    /// copied from the template by the engine at thread instantiation (it owns guest
    /// memory). Returns 0 when the title has no TLS - the compiler then never reads
    /// the thread pointer, so 0 is never dereferenced.
    pub fn ensure_tls_block(&mut self, thid: i32) -> u32 {
        let ceiling = self.heap_ceiling();
        let arena = &mut self.arena;
        self.tls
            .block(thid, |size, align| arena.galloc(size, align, ceiling))
    }

    /// The TLS init image `(source address, initialized byte count)` for the template,
    /// so the engine can copy `.tdata` into a freshly allocated block. `(_, 0)` when
    /// there is nothing to copy (a pure `.tbss` template stays zero).
    #[inline]
    pub fn tls_init_image(&self) -> (u32, u32) {
        self.tls.init_image()
    }

    /// What the engine needs to instantiate thread `thid`: `(thread pointer, init image
    /// source, bytes to copy)`. The copy length is the template's `.tdata` size the FIRST
    /// time a block is handed out and 0 every time after.
    ///
    /// >>> A FIBER'S THREAD POINTER IS ITS RUNNER'S, AND THAT BLOCK IS ALREADY LIVE.
    /// `fiber_dispatch` binds a fiber's backing thread to its runner's block, so seeding
    /// `.tdata` on every instantiation re-copied the template over storage the runner had
    /// been writing for its whole life. MEASURED (one title, its first mission load): the
    /// head of the block holds a per-thread scratch-stack `{base, top}` pair its worker set
    /// up at start; the first job fiber's instantiation reset it to the template's zeros and
    /// the fiber's expression evaluator pushed through a NULL stack - MemoryOutOfBounds on
    /// the loader. On hardware a fiber is a context switch inside its thread: nothing
    /// reinitialises that thread's TLS.
    pub fn tls_block_to_instantiate(&mut self, thid: i32) -> (u32, u32, u32) {
        let ceiling = self.heap_ceiling();
        let arena = &mut self.arena;
        self.tls
            .block_to_instantiate(thid, |size, align| arena.galloc(size, align, ceiling))
    }

    /// The stable guest address of TLS slot `key` for the currently running thread (see
    /// [`tls_addr_of`](Self::tls_addr_of)).
    pub fn tls_addr(&mut self, key: u32) -> Option<u32> {
        let current = self.current;
        self.tls_addr_of(current, key)
    }

    /// The TLS slot `key` of thread `thid` (`sceKernelGetThreadTLSAddr`; `thid` 0 is the
    /// caller): word `key` of that thread's slot array, allocated zeroed on the thread's first
    /// query, so a never-written slot reads back as NULL. `None` for a key past
    /// [`TLS_KEY_MAX`], which has no slot.
    pub fn tls_addr_of(&mut self, thid: i32, key: u32) -> Option<u32> {
        let thid = if thid == 0 { self.current } else { thid };
        // A fiber shares its runner's thread-local storage: on hardware it runs ON that
        // thread, so a slot it reads must be the same slot the runner writes.
        let thread = self.logical_thread(thid);
        let ceiling = self.heap_ceiling();
        let arena = &mut self.arena;
        self.tls
            .slot_addr(thread, key, |size, align| arena.galloc(size, align, ceiling))
    }
}
