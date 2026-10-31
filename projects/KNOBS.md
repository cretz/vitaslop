# VITASLOP_* environment knobs

GENERATED - do not edit by hand. Regenerate with:

```text
VITASLOP_BLESS_KNOBS=1 cargo test -p vitaslop-runtime --lib knobs
```

Every knob the workspace reads, with the file that reads it and the first
line of that code's own documentation. A knob read at TRANSPILE time only
takes effect when the module is built, so it must be set for the whole run,
not just the frame you care about. Trapping diagnostics can be held inert
until a chosen display frame with `VITASLOP_ARM_AT_FRAME`, which is what
makes a first-hit watchpoint usable deep inside a game instead of firing
during boot.

608 knobs.

| knob | read in | what it does |
|---|---|---|
| `VITASLOP_A` | vitaslop-frontend/src/settings.rs | - |
| `VITASLOP_AAC_DIR` | vitaslop-aac/tests/oracle.rs | The `AudioSpecificConfig` an ADTS header describes: 5 bits of object type, 4 of |
| `VITASLOP_ALLOW_SOFTWARE_GPU` | vitaslop-web/src/lib.rs | Whether a run may proceed on a software rasteriser (`VITASLOP_ALLOW_SOFTWARE_GPU`). |
| `VITASLOP_AMBIENT_PROBE` | vitaslop-frontend/src/settings.rs | The knobs every run starts from. |
| `VITASLOP_ARENA_TRIM` | vitaslop-platform/src/gpu.rs | The 2026-09-28 memory bounds, each with its arm back (`=0`): `VITASLOP_SUBRECT_POOL` |
| `VITASLOP_ARENA_UPLOAD_PROBE` | vitaslop-platform/src/gpu.rs | `VITASLOP_ARENA_UPLOAD_PROBE=1` - time the SAME bytes through the other upload path, a |
| `VITASLOP_ARM_AT_FRAME` | vitaslop-native/src/threaded.rs | Linear-memory offset of the "diagnostics armed" word, when this build was |
| `VITASLOP_ARM_FUNC` | vitaslop-native/tests/homebrew_memcpy.rs | Scratch window well past the code: `[SRC, SRC+WIN)` and `[DST, DST+WIN)`. |
| `VITASLOP_ASYNC_PIPELINES` | vitaslop-platform/src/gpu.rs | >>> PIPELINES ARE STARTED ASYNC, AHEAD OF THE FRAME THAT DRAWS THEM - see |
| `VITASLOP_AT9_DIR` | vitaslop-atrac9/tests/oracle.rs | Decode a whole AT9 payload the way a superframe consumer does: for each |
| `VITASLOP_AUDIO_MUTE` | vitaslop-desktop/src/session.rs | - |
| `VITASLOP_AUDIO_RAW` | vitaslop-runtime/src/vita/audio.rs | Optional raw-s16le capture of the output, ONE FILE PER PORT (env `VITASLOP_AUDIO_RAW` |
| `VITASLOP_AUDIO_SCHEDULE` | vitaslop-runtime/src/vita/audio.rs | Whether `sceAudioOutOutput` paces by the port's play-out schedule (the default) - see |
| `VITASLOP_AUDIO_WALL` | vitaslop-runtime/src/vita/audio.rs | The host wall clock in microseconds when `sceAudioOutOutput` should pace on it (see |
| `VITASLOP_AVCDEC_CATCH_UP_MS` | vitaslop-web/src/smp.rs | `VITASLOP_AVCDEC_CATCH_UP_MS`: how long a forwarded movie decode may wait for its decoder |
| `VITASLOP_B` | vitaslop-frontend/src/settings.rs | - |
| `VITASLOP_BACKTRACE` | vitaslop-runtime/src/vita/mod.rs | Print the guest call chain the first time a chosen NID is called from each thread |
| `VITASLOP_BLESS_GXMCONF` | vitaslop-gxp-shader/tests/conformance.rs | A 64-bit surface stores an ALIGNED register pair, whichever half the program wrote last. |
| `VITASLOP_BLOCK_HIST` | vitaslop-native/src/threaded.rs | The first and last display frame each block was entered on. |
| `VITASLOP_BLOCK_HIST_FIRST` | vitaslop-native/src/threaded.rs | The first and last display frame each block was entered on. |
| `VITASLOP_BLOCK_HIST_SEQ` | vitaslop-native/src/threaded.rs | Print the block-visit histogram gathered under `VITASLOP_BLOCK_HIST`: the `top` |
| `VITASLOP_BROWSER_EXIT_UNWIND` | vitaslop-web/src/browser_sched.rs | Whether a guest thread that EXITS through a host call (`sceKernelExitThread`, |
| `VITASLOP_BROWSER_FASTFORWARD` | vitaslop-runtime/src/vita/audio.rs | The host wall clock in microseconds when `sceAudioOutOutput` should pace on it (see |
| `VITASLOP_BROWSER_FUEL` | vitaslop-web/src/browser_sched.rs | Guest work a thread may execute before the browser preempts it, in WASMTIME FUEL UNITS |
| `VITASLOP_BROWSER_HEARTBEAT_MS` | vitaslop-web/src/lib.rs | - |
| `VITASLOP_BROWSER_INSTANCE_POOL` | vitaslop-web/src/browser_sched.rs | Whether a finished thread's module instance may be REUSED by the next thread |
| `VITASLOP_BROWSER_INSTANCE_RESERVE` | vitaslop-web/src/browser_sched.rs | Stand up one guest thread: an instance importing the shared memory and a |
| `VITASLOP_BROWSER_QUANTUM_CALLS` | vitaslop-web/src/browser_sched.rs | Host calls one guest thread may make before the browser preempts it |
| `VITASLOP_BROWSER_RENDER_FROM` | vitaslop-web/src/lib.rs | Frame from which a FAST-FORWARD still RENDERS (`VITASLOP_BROWSER_RENDER_FROM`), even though |
| `VITASLOP_BROWSER_SPLIT_MEMORY` | vitaslop-web/src/browser_sched.rs | >>> THE GUEST REGION INSIDE THIS MODULE'S OWN LINEAR MEMORY. |
| `VITASLOP_BROWSER_SUPERSAMPLE` | vitaslop-web/src/lib.rs | Supersample factor for the live browser render (`VITASLOP_BROWSER_SUPERSAMPLE`). |
| `VITASLOP_BROWSER_UNPACED` | vitaslop-web/src/lib.rs | - |
| `VITASLOP_BUILD_FASTPATH` | vitaslop-runtime/src/render.rs | `VITASLOP_BUILD_FASTPATH=0`: the NEGATIVE CONTROL arm for what `build` stopped doing. |
| `VITASLOP_CALLSITES_WINDOW` | vitaslop-desktop/src/retail.rs | Where the idle clock went SINCE `before` - the windowed reading. |
| `VITASLOP_CALL_TABLE` | vitaslop-runtime/src/call_table.rs | The frame window, or `None` when the table is off. |
| `VITASLOP_CALL_TABLE_REGS` | vitaslop-runtime/src/call_table.rs | r0..r12 at the FIRST occurrence, printed under `VITASLOP_CALL_TABLE_REGS=1` - the |
| `VITASLOP_CAPSULE_DUMP_PROGS` | vitaslop-native/examples/capsule-replay.rs | - |
| `VITASLOP_CAPSULE_DUMP_SA` | vitaslop-native/examples/capsule-replay.rs | `VITASLOP_CAPSULE_DUMP_SA=1`: print this draw's uniform banks - `frag_sa` with the GUEST |
| `VITASLOP_CAPSULE_DUMP_VERTS` | vitaslop-native/examples/capsule-replay.rs | - |
| `VITASLOP_CAPSULE_EXTENT` | vitaslop-native/examples/capsule-replay.rs | - |
| `VITASLOP_CAPSULE_TEX_DIR` | vitaslop-native/examples/capsule-replay.rs | - |
| `VITASLOP_CARRY_UNPRESENTED` | vitaslop-web/src/lib.rs | `VITASLOP_CARRY_UNPRESENTED=0`: the arm back - drop an unpresented frame's scenes whole. |
| `VITASLOP_CHAIN_DRAWS` | vitaslop-platform/src/gpu/gxm/renderer_pass.rs | - |
| `VITASLOP_CHAIN_LIMIT` | vitaslop-native/tests/gpu_rtt_gamma.rs | Render a chain of `feedback` sample-and-write-back passes over the offscreen target and |
| `VITASLOP_CHAIN_SKIP` | vitaslop-native/src/wgpu_render.rs | - |
| `VITASLOP_CHECK_ADDRS` | vitaslop-native/tests/retail_boot_probe.rs | - |
| `VITASLOP_CLOCK_EVERY` | vitaslop-desktop/src/session.rs | `VITASLOP_CLOCK_EVERY` - see [`Self::report_clock`]: `(frame, wall, game us, audio s)`. |
| `VITASLOP_CLOCK_TRACE` | vitaslop-runtime/src/sched.rs | - |
| `VITASLOP_CLOCK_WALL_FLOOR` | vitaslop-native/src/smp.rs | The game clock is floored at the wall - see `VitaState::wall_floor_tick`. |
| `VITASLOP_CLOCK_WALL_STEP_MS` | vitaslop-runtime/src/host/state_clock.rs | >>> THE GAME CLOCK MAY NOT RUN SLOWER THAN REAL TIME (SMP browser runs). |
| `VITASLOP_CODE_RANGE` | vitaslop-runtime/src/vita/mod.rs | The guest code range scanned for the game-level caller in [`dispatch`] (env |
| `VITASLOP_COMPACT_SPARSE` | vitaslop-runtime/src/host/snapshot_cache.rs | `VITASLOP_COMPACT_SPARSE=0` is the NEGATIVE CONTROL for the compaction: the `min..=max` |
| `VITASLOP_COMPILE_CACHE` | vitaslop-desktop/src/retail.rs | - |
| `VITASLOP_CONSOLE` | vitaslop-web/src/logging.rs | `VITASLOP_CONSOLE=1`: mirror the run's status notes - the setup summary, the adapter and |
| `VITASLOP_CPU_SHARE` | vitaslop-desktop/src/retail.rs | - |
| `VITASLOP_CPU_SHARE_FROM` | vitaslop-runtime/src/sched.rs | `VITASLOP_CPU_SHARE_FROM=<frame>` - see the reset in `SchedCore::on_suspended`. |
| `VITASLOP_CTRL_READ_NEW` | vitaslop-runtime/src/vita/ctrl.rs | A blocking read returns `count` samples (the default) unless `VITASLOP_CTRL_READ_NEW=1`. |
| `VITASLOP_DBG_CALLSITES` | vitaslop-runtime/src/vita/mod.rs | Diagnostic call-site profiler (`VITASLOP_DBG_CALLSITES`): counts host calls |
| `VITASLOP_DEBUG_CAPTURE` | vitaslop-frontend/src/settings.rs | The `VITASLOP_*` map a run of these settings is configured with: the base set, |
| `VITASLOP_DECODE_CACHE_MB` | vitaslop-runtime/src/host/snapshot_cache.rs | Byte budget for retained texture snapshots, scaled to the device |
| `VITASLOP_DEFER_GEOMETRY` | vitaslop-runtime/src/host/resolve.rs | Whether a draw's VERTEX AND INDEX BYTES are read at `sceGxmEndScene` rather than at the |
| `VITASLOP_DEFER_WINDOW_BYTES` | vitaslop-runtime/src/host/snapshot_cache.rs | `VITASLOP_GXP_ATTR_ROW=0` - the negative control for laying the interleaved row out per |
| `VITASLOP_DELAY_CENSUS` | vitaslop-runtime/src/vita/threadmgr.rs | Diagnostic (`VITASLOP_DELAY_CENSUS=1`): every `sceKernelDelayThread` tallied by (call site, |
| `VITASLOP_DEVICE_BUDGET` | vitaslop-desktop/src/retail.rs | Headless self-check of the retail path (NO window): load `dir`, optionally drive a |
| `VITASLOP_DIALOG_RUNNING_FRAMES` | vitaslop-runtime/src/vita/services.rs | How many display flips an opened dialog reports RUNNING for before FINISHED |
| `VITASLOP_DIRTY_MARK_TEST` | vitaslop-transpiler/src/emit.rs | >>> UNDER SMP THE MARK IS WRITTEN ONLY WHEN THE BYTE DIFFERS FROM THE EPOCH. |
| `VITASLOP_DIRTY_PAGES` | vitaslop-native/src/threaded.rs | Linear-memory offset of the guest-store dirty block, when this build was |
| `VITASLOP_DIRTY_RUN_MARK` | vitaslop-transpiler/src/emit.rs | Turn the coalesced run-mark on or off for modules emitted on this thread after this |
| `VITASLOP_DIRTY_SP_MARKS` | vitaslop-transpiler/src/emit.rs | `VITASLOP_DIRTY_SP_MARKS=1`: mark SP-relative stores like any other - the arm back from |
| `VITASLOP_DISASM_ARM` | vitaslop-transpiler/tests/disasm_dump.rs | - |
| `VITASLOP_DISASM_FILE` | vitaslop-transpiler/tests/disasm_dump.rs | - |
| `VITASLOP_DISASM_RANGE` | vitaslop-transpiler/tests/disasm_dump.rs | - |
| `VITASLOP_DISPATCH_ALL` | vitaslop-transpiler/src/emit.rs | The ablation arm that prices a dispatch re-entry: `VITASLOP_DISPATCH_ALL=1` sends even a |
| `VITASLOP_DISPLAY_SNAPSHOT` | vitaslop-platform/src/gpu.rs | `VITASLOP_DISPLAY_SNAPSHOT=0`: the arm back to a display pass that samples its own buffer |
| `VITASLOP_DRAW_NOTE_AT` | vitaslop-web/src/lib.rs | Render one freshly-executed FRAME - every scene the guest submitted between |
| `VITASLOP_DRAW_NOTE_TARGET` | vitaslop-web/src/lib.rs | - |
| `VITASLOP_DRAW_NOTE_TEX` | vitaslop-web/src/lib.rs | - |
| `VITASLOP_DRAW_ONLY` | vitaslop-runtime/src/render.rs | - |
| `VITASLOP_DRAW_RANGE` | vitaslop-platform/src/gpu/gxm/coverage.rs | `VITASLOP_GXM_DRAW_COVERAGE=1`: how many SAMPLES each draw of the frame actually wrote, |
| `VITASLOP_DRAW_STATS` | vitaslop-runtime/src/render.rs | `VITASLOP_DRAW_STATS` (diagnostic), cached: asked per scene and per draw batch. |
| `VITASLOP_DRV_KEY` | vitaslop-runtime/src/ingest/pfscrypt.rs | `F00D(klicensee)` for the title, from `VITASLOP_DRV_KEY` (32 hex chars), or |
| `VITASLOP_DUMP_ADDR` | vitaslop-native/tests/dump_segment.rs | - |
| `VITASLOP_DUMP_DIR` | vitaslop-runtime/src/ingest/pipeline.rs | Diagnostic: write the eboot's decrypted inner ELF to `VITASLOP_DUMP_DIR/eboot.elf`, so |
| `VITASLOP_DUMP_DRAW` | vitaslop-native/tests/retail_boot_probe.rs | - |
| `VITASLOP_DUMP_DRAWS` | vitaslop-native/tests/retail_boot_probe.rs | - |
| `VITASLOP_DUMP_DRAW_GXP` | vitaslop-gxp-shader/tests/oracle.rs | Correlate each captured vertex<->fragment PAIR (from a real draw run) to establish the |
| `VITASLOP_DUMP_DRAW_GXP_CAP` | vitaslop-runtime/src/host/state_resolve.rs | `vhandle` and `streams` are the DRAW's bindings - the snapshot's, which a precomputed |
| `VITASLOP_DUMP_DRAW_GXP_FULL` | vitaslop-runtime/src/host/state_resolve.rs | - |
| `VITASLOP_DUMP_EXPORTS` | vitaslop-runtime/src/link.rs | - |
| `VITASLOP_DUMP_FILES` | vitaslop-runtime/src/ingest/pipeline.rs | Diagnostic: decrypt the container and write named plaintext files out to |
| `VITASLOP_DUMP_FPROG` | vitaslop-runtime/src/host/reflect.rs | Diagnostic (`VITASLOP_DUMP_FPROG`): dump each fragment program's samplers. |
| `VITASLOP_DUMP_FUNC` | vitaslop-native/tests/dump_function.rs | - |
| `VITASLOP_DUMP_GXP_BIN` | vitaslop-platform/src/gpu/gxm/fallback.rs | >>> A REFUSAL THAT DOES NOT HAND OVER THE EVIDENCE COSTS A PLAY SESSION. |
| `VITASLOP_DUMP_IMAGE` | vitaslop-native/tests/retail_boot_probe.rs | - |
| `VITASLOP_DUMP_IMPORTS` | vitaslop-native/tests/dump_function.rs | - |
| `VITASLOP_DUMP_IR` | vitaslop-native/tests/homebrew_memcpy.rs | Scratch window well past the code: `[SRC, SRC+WIN)` and `[DST, DST+WIN)`. |
| `VITASLOP_DUMP_MAP` | vitaslop-native/tests/retail_boot_probe.rs | - |
| `VITASLOP_DUMP_MEM` | vitaslop-native/tests/retail_boot_probe.rs | - |
| `VITASLOP_DUMP_OUT` | vitaslop-native/tests/dump_segment.rs | - |
| `VITASLOP_DUMP_PATHS` | vitaslop-native/tests/retail_boot_probe.rs | - |
| `VITASLOP_DUMP_REGION` | vitaslop-native/tests/retail_boot_probe.rs | - |
| `VITASLOP_DUMP_REGION_RANGE` | vitaslop-native/tests/retail_boot_probe.rs | - |
| `VITASLOP_DUMP_RENDERSCENE` | vitaslop-native/tests/retail_boot_probe.rs | - |
| `VITASLOP_DUMP_SCENES` | vitaslop-desktop/src/retail.rs | Step the guest one display frame. |
| `VITASLOP_DUMP_SLOW_PAIRS` | vitaslop-gxp-shader/tests/corpus.rs | Time each pair a run dumped as a SLOW in-frame translation (`VITASLOP_DUMP_SLOW_PAIRS`: |
| `VITASLOP_DUMP_STDOUT` | vitaslop-desktop/src/retail.rs | - |
| `VITASLOP_DUMP_STREAM_BYTES` | vitaslop-runtime/src/host/resolve.rs | `VITASLOP_DUMP_STREAM_BYTES[=<n>]`: with the per-draw dump on, print each vertex stream's |
| `VITASLOP_DUMP_STUBS` | vitaslop-native/tests/retail_boot_probe.rs | - |
| `VITASLOP_DUMP_TEX` | vitaslop-native/tests/retail_boot_probe.rs | - |
| `VITASLOP_DUMP_TEX_DIR` | vitaslop-runtime/src/host/state_resolve.rs | - |
| `VITASLOP_DUMP_TEX_MAX_TEXELS` | vitaslop-runtime/src/host/state_resolve.rs | - |
| `VITASLOP_DUMP_TRIS` | vitaslop-runtime/src/render.rs | - |
| `VITASLOP_DUMP_VPROG` | vitaslop-runtime/src/host/reflect.rs | Diagnostic (`VITASLOP_DUMP_VPROG`): dump each vertex program's parameter table. |
| `VITASLOP_DUMP_VUBUF` | vitaslop-runtime/src/host/state_resolve.rs | - |
| `VITASLOP_DUMP_WASM` | vitaslop-native/tests/transpile_peak.rs | - |
| `VITASLOP_EAGER_FILES` | vitaslop-desktop/src/retail.rs | Load, decrypt, link, transpile, and instantiate the title in `dir` for live |
| `VITASLOP_EARLY_GRACE_MS` | vitaslop-web/src/lib.rs | >>> SERVE A PARKED GPU WAIT BEFORE THE PRESENT, NOT AFTER IT. |
| `VITASLOP_EARLY_WAIT_ANY` | vitaslop-web/src/lib.rs | `VITASLOP_EARLY_WAIT_ANY=1`: an early completion waits for EVERY in-flight readback of its |
| `VITASLOP_EXPLICIT_BOUNDS` | vitaslop-native/src/threaded.rs | `VITASLOP_EXPLICIT_BOUNDS=1` - check every guest memory access in code instead of catching |
| `VITASLOP_FAST_IMPORT_CURATED` | vitaslop-runtime/src/vita/mod.rs | Whether `func_nid`'s handler can only ever CONTINUE, so the transpiler may route the |
| `VITASLOP_FIBER_RUNNER_PRIORITY` | vitaslop-runtime/src/host.rs | Whether a fiber's backing thread runs at its runner's priority (the hardware: a fiber IS its |
| `VITASLOP_FIND_WORD` | vitaslop-native/tests/retail_boot_probe.rs | The span `VITASLOP_FIND_WORD` searches: from the image base up through the guest heap. |
| `VITASLOP_FLAGS_WIDE_C` | vitaslop-transpiler/src/emit.rs | The A/B arm for [`emit_flags_add`]'s carry and overflow forms: `VITASLOP_FLAGS_WIDE_C=1` |
| `VITASLOP_FLAG_POISON` | vitaslop-transpiler/src/emit.rs | `VITASLOP_FLAG_POISON=0/1` - the FALSIFIER for the flag-liveness pass |
| `VITASLOP_FORCE_READY` | vitaslop-native/tests/retail_boot_probe.rs | - |
| `VITASLOP_FORCE_READY_V2` | vitaslop-native/tests/retail_boot_probe.rs | - |
| `VITASLOP_FORCE_RET` | vitaslop-transpiler/src/emit.rs | Diagnostic forced return. |
| `VITASLOP_FRAME_CAPSULE` | vitaslop-runtime/src/capsule.rs | `VITASLOP_FRAME_CAPSULE=<dir>`: write every frame a headless run renders for a SHOT to |
| `VITASLOP_FRAME_DIGEST` | vitaslop-desktop/src/retail.rs | The running determinism signature at frame `f`, as the browser prints it: `sigtrace` every |
| `VITASLOP_FRAME_TOPUP` | vitaslop-runtime/src/host/env.rs | `VITASLOP_FRAME_TOPUP=1`: top the game clock up to a full frame at a display flip. |
| `VITASLOP_FUEL` | vitaslop-native/src/threaded.rs | This thread's SOFTWARE fuel counter (`abi::FUEL_EXPORT`), present only when the |
| `VITASLOP_GALLOC_SKEW` | vitaslop-runtime/src/host/arena.rs | Whether `VITASLOP_GALLOC_SKEW` has been applied - see [`GuestArena::galloc`]. |
| `VITASLOP_GAME_DIR` | vitaslop-runtime/src/ingest/mod.rs | Test-fixture access. |
| `VITASLOP_GAME_PKG` | vitaslop-runtime/src/ingest/pipeline.rs | Diagnostic: write the eboot's decrypted inner ELF to `VITASLOP_DUMP_DIR/eboot.elf`, so |
| `VITASLOP_GAME_WORK` | vitaslop-runtime/src/ingest/pipeline.rs | Diagnostic: write the eboot's decrypted inner ELF to `VITASLOP_DUMP_DIR/eboot.elf`, so |
| `VITASLOP_GAME_ZIP` | vitaslop-runtime/src/ingest/mod.rs | - |
| `VITASLOP_GAP_CAP` | vitaslop-native/tests/retail_boot_probe.rs | - |
| `VITASLOP_GESTURE_DRAG_SLOP` | vitaslop-runtime/src/vita/gesture.rs | [`DRAG_SLOP`], or `VITASLOP_GESTURE_DRAG_SLOP=<panel units>` to begin a drag only once the |
| `VITASLOP_GPU` | vitaslop-native/tests/retail_boot_probe.rs | - |
| `VITASLOP_GPU_BUDGET` | vitaslop-web/src/lib.rs | >>> THE GPU BUDGET: a present costs the GPU milliseconds the newest timestamp query |
| `VITASLOP_GPU_BURN` | vitaslop-platform/src/gpu/gxm/mod.rs | `VITASLOP_GPU_BURN`'s compute work, built on first use - see [`Self::gpu_burn`]. |
| `VITASLOP_GPU_CHAIN_DIR` | vitaslop-native/src/wgpu_render.rs | `VITASLOP_GPU_CHAIN_DIR=<dir>`: write every offscreen target of the frame just |
| `VITASLOP_GPU_QUEUE_DEPTH` | vitaslop-web/src/lib.rs | How many submits may be in flight before a present declines to make another. |
| `VITASLOP_GPU_STALE_OLD` | vitaslop-web/src/lib.rs | `VITASLOP_GPU_STALE_OLD=1`: the stale rule as it was - see the check in `present`. |
| `VITASLOP_GPU_TIME_ALL` | vitaslop-platform/src/gpu.rs | `VITASLOP_GXP_PASS_SPLIT_EVERY=<n>` cuts a render pass every `n` draws, with no shader |
| `VITASLOP_GUARD_REG` | vitaslop-transpiler/src/emit.rs | Diagnostic callee-saved-register guard. |
| `VITASLOP_GUEST_CORES` | vitaslop-runtime/src/host/env.rs | CPU cores a Vita gives a GAME. |
| `VITASLOP_GUEST_PROF` | vitaslop-transpiler/src/abi.rs | SMP, `VITASLOP_GUEST_PROF` only: worker `w`'s GUEST-FUNCTION slot is host-mirror slot |
| `VITASLOP_GXM` | vitaslop-native/examples/capsule-replay.rs | - |
| `VITASLOP_GXM_ALPHA_SINGLE` | vitaslop-conformance-harness/tests/vita_gxmconf_real.rs | **SCENE 10 - A U8_A SURFACE STORES THE FRAGMENT'S ALPHA.** |
| `VITASLOP_GXM_ARENA_FLOOR_KB` | vitaslop-platform/src/gpu.rs | `VITASLOP_GXM_DEST_SPLIT_AB=<n>`: alternate the destination-colour pass SPLIT on and off every |
| `VITASLOP_GXM_ARENA_POOL` | vitaslop-platform/src/gpu/gxm/vertex.rs | `VITASLOP_GXM_ARENA_POOL=1` pools the six per-pass staging arenas. |
| `VITASLOP_GXM_ARENA_REPEAT` | vitaslop-platform/src/gpu/gxm/caches.rs | >>> REPACKS OF GEOMETRY THIS RUN HAD ALREADY EVICTED - the cache THRASHING. |
| `VITASLOP_GXM_BACKGROUND_DEPTH` | vitaslop-conformance-harness/tests/vita_gxmconf.rs | **SCENE 8 - A REVERSED DEPTH RANGE: THE PASS STARTS FROM THE SURFACE'S BACKGROUND DEPTH.** |
| `VITASLOP_GXM_BUFFER_PREINIT` | vitaslop-platform/src/gpu/gxm/buffers.rs | Create through the pool: a hit is free, a miss is the allocation that was going to |
| `VITASLOP_GXM_COLOR_MASK_ORDER` | vitaslop-conformance-harness/tests/vita_gxmconf_real.rs | **SCENE 14 - THE COLOUR MASK NUMBERS ALPHA AS BIT 0.** |
| `VITASLOP_GXM_DEPTH_ENC` | vitaslop-platform/src/gpu/gxm/mod.rs | Which value a later pass reads out of a render target's depth |
| `VITASLOP_GXM_DEST_COALESCE` | vitaslop-platform/src/gpu.rs | `VITASLOP_GXM_DEST_COALESCE=0`: cut the pass and take a WHOLE destination copy in front of |
| `VITASLOP_GXM_DEST_GEOM` | vitaslop-platform/src/gpu.rs | `VITASLOP_GXM_DEST_GEOM=0`: bound the destination-copy coalescing by each draw's VIEWPORT |
| `VITASLOP_GXM_DEST_GEOM_GATE` | vitaslop-platform/src/gpu/gxm/renderer_pass.rs | The CPU budget for vertex footprints in this encode of pass `pass`, or `None` for coarse |
| `VITASLOP_GXM_DEST_GEOM_SLICE` | vitaslop-platform/src/gpu/gxm/clip.rs | The instructions of `primary` that can reach the POSITION (`o0..o3`) - what a caller |
| `VITASLOP_GXM_DEST_SPLIT_AB` | vitaslop-platform/src/gpu/gxm/timestamps.rs | `VITASLOP_GXM_DEST_SPLIT_AB`'s two arms: `(frames, summed GPU ms)` for the arm that |
| `VITASLOP_GXM_DRAW_COVERAGE` | vitaslop-gxp-shader/src/link.rs | `1` - WHERE DID THIS DRAW'S VERTICES GO? Draw the mesh at CLAMPED normalised coordinates and |
| `VITASLOP_GXM_FLOAT_TARGETS` | vitaslop-platform/src/gpu.rs | Whether a guest `SceGxmColorFormat` is F11F11F10, the packed FLOATING-POINT format a title |
| `VITASLOP_GXM_FLOAT_TARGETS_WIDE` | vitaslop-platform/src/gpu.rs | Whether a guest `SceGxmColorFormat` is F11F11F10, the packed FLOATING-POINT format a title |
| `VITASLOP_GXM_GUEST_STENCIL` | vitaslop-platform/src/gpu.rs | Whether a scene whose depth-stencil surface FORCE_LOADs a separate STENCIL plane that a |
| `VITASLOP_GXM_HOLD_FLIP` | vitaslop-platform/src/gpu/gxm/renderer_misc.rs | Which image this frame shows when its LAST scene is not a display buffer - see the |
| `VITASLOP_GXM_NO_MULTISAMPLE` | vitaslop-platform/src/gpu/gxm/rtt.rs | A/B instrument: force every pass to ONE sample, whatever the guest asked for. |
| `VITASLOP_GXM_PVS_BULK_TABLE` | vitaslop-runtime/src/vita/gxm.rs | `VITASLOP_GXM_PVS_BULK_TABLE=1` - restore the VERTEX bind's old WHOLESALE table copy, zeros |
| `VITASLOP_GXM_RTT_CLEAR_EVERY_FRAME` | vitaslop-platform/src/gpu.rs | `VITASLOP_GXM_RTT_CLEAR_EVERY_FRAME=1` - restore the pre-2026-09-12 behaviour, in which the |
| `VITASLOP_GXM_RTT_WRITEBACK` | vitaslop-desktop/src/retail.rs | `display` is the size the GUEST declared to `sceDisplaySetFrameBuf`, which is what |
| `VITASLOP_GXM_SHARE_PROGRAMS` | vitaslop-runtime/src/vita/gxm.rs | int sceGxmShaderPatcherCreateFragmentProgram(patcher, programId, outputFormat, |
| `VITASLOP_GXM_STAGING` | vitaslop-platform/src/gpu/gxm/mod.rs | >>> THE ARENA UPLOAD'S STAGING BELT - the path that replaces `queue.write_buffer` |
| `VITASLOP_GXM_STALE_UNIFORMS` | vitaslop-desktop/src/retail.rs | Headless self-check of the retail path (NO window): load `dir`, optionally drive a |
| `VITASLOP_GXM_STATE_DEDUP` | vitaslop-platform/src/gpu/gxm/rtt.rs | What a render pass already has bound, so a draw that changes nothing rebinds nothing. |
| `VITASLOP_GXM_SWEEP_EVERY` | vitaslop-platform/src/gpu.rs | `VITASLOP_GXM_SWEEP_EVERY=<n>` - how many frames apart the packed-geometry budget check and |
| `VITASLOP_GXM_TEX_UNWRITTEN` | vitaslop-runtime/src/render.rs | `VITASLOP_GXM_TEX_UNWRITTEN=0` - the arm back for reading ANY uniform 4-byte fill as |
| `VITASLOP_GXM_TILE_SHADOW` | vitaslop-platform/src/gpu.rs | `VITASLOP_GXM_TILE_SHADOW=0` is the arm back for the half-precision tile shadow of an 8-bit |
| `VITASLOP_GXM_UNIFORM_POISON` | vitaslop-gxp-shader/src/module.rs | `:bits=<hex>` - paint a lane 1.0 when that register's RAW BITS equal this word, 0.0 |
| `VITASLOP_GXM_VIEWPORT` | vitaslop-platform/src/gpu.rs | `VITASLOP_GXM_VIEWPORT=0` - the NEGATIVE CONTROL for the guest viewport reaching the |
| `VITASLOP_GXM_ZLS` | vitaslop-platform/src/gpu/gxm/mod.rs | >>> THE DEPTH-STENCIL A FORCE_STORE PASS LEFT IN MEMORY, by (depth address, width, |
| `VITASLOP_GXP` | vitaslop-native/examples/capsule-replay.rs | - |
| `VITASLOP_GXP_` | vitaslop-native/examples/capsule-replay.rs | - |
| `VITASLOP_GXP_ALIGN_ROW` | vitaslop-runtime/src/host/snapshot_cache.rs | Whether a draw decodes only the texture units its fragment program DECLARES - see |
| `VITASLOP_GXP_ALLOW_FIXED_FUNCTION` | vitaslop-platform/src/gpu/gxm/fallback.rs | >>> A PAIR THIS RECOMPILER CANNOT TRANSLATE DROPS ITS DRAWS. |
| `VITASLOP_GXP_ARENA_RING` | vitaslop-platform/src/gpu.rs | `VITASLOP_GXP_ARENA_RING=<n>`: how many COPIES of each render pass's recompiled-path |
| `VITASLOP_GXP_ATTR_ALL` | vitaslop-gxp-shader/tests/corpus.rs | What [`vitaslop_gxp_shader::attrflow`] decides for every attribute of every pair that links, |
| `VITASLOP_GXP_ATTR_FILL` | vitaslop-gxp-shader/src/module.rs | Bytes the driver adds to the DEFAULT uniform buffer's bound address before writing it into |
| `VITASLOP_GXP_ATTR_ROW` | vitaslop-runtime/src/host/snapshot_cache.rs | `VITASLOP_GXP_ATTR_ROW=0` - the negative control for laying the interleaved row out per |
| `VITASLOP_GXP_BIND_TRACE` | vitaslop-platform/src/gpu.rs | Diagnostic (`VITASLOP_GXP_BIND_TRACE=1`): say, once per pair and plan shape, WHICH view each |
| `VITASLOP_GXP_BLOB` | vitaslop-gxp-shader/tests/corpus.rs | Print one named blob's recompiled WGSL body and its container reflection. |
| `VITASLOP_GXP_BOUND_READS` | vitaslop-gxp-shader/src/link.rs | `0` keeps a bank read dynamically at the full [`BANK_REGS`] - the A/B arm for |
| `VITASLOP_GXP_CAPSULE` | vitaslop-runtime/src/capsule.rs | Diagnostic (`VITASLOP_GXP_CAPSULE=<vprog-hash>[,<vprog-hash>]:<dir>[:N]`): write the first |
| `VITASLOP_GXP_CAPSULE_MIN_INDICES` | vitaslop-runtime/src/capsule.rs | Diagnostic (`VITASLOP_GXP_CAPSULE=<vprog-hash>[,<vprog-hash>]:<dir>[:N]`): write the first |
| `VITASLOP_GXP_CAPSULE_SKIP` | vitaslop-runtime/src/capsule.rs | Diagnostic (`VITASLOP_GXP_CAPSULE_SKIP=<n>`): ignore the first `n` matching submissions |
| `VITASLOP_GXP_CASES_OUT` | vitaslop-gxp-shader/tests/conformance.rs | Write every case out for the GPU runner, with the INTENT as the expectation. |
| `VITASLOP_GXP_CASE_TEX` | vitaslop-gxp-shader/src/wgsl.rs | >>> DOES THE STAND-IN TEXTURE VARY WITH THE COORDINATE? |
| `VITASLOP_GXP_CLAIM_WIDTH` | vitaslop-gxp-shader/src/link.rs | Re-order a convention-placed layout so that every forwarded attribute's usage STARTS at the |
| `VITASLOP_GXP_CLIP_DUMP_SA` | vitaslop-platform/src/gpu/gxm/clip.rs | - |
| `VITASLOP_GXP_CLIP_RETRY_EVERY_FRAME` | vitaslop-platform/src/gpu.rs | `VITASLOP_GXP_CLIP_RETRY_EVERY_FRAME=1` - re-measure a pair whose clip measurement found no |
| `VITASLOP_GXP_CMOVU8` | vitaslop-gxp-shader/src/module.rs | Whether [`crate::wgsl`]'s byte-wise conditional move tests EACH BYTE of its test operand |
| `VITASLOP_GXP_CORPUS` | vitaslop-gxp-shader/src/link.rs | Every vertex program in the captured corpora (`VITASLOP_GXP_CORPUS`, `;`-separated |
| `VITASLOP_GXP_COVERAGE_LOG` | vitaslop-gxp-shader/tests/corpus.rs | Price the F16 emulation in CONVERSIONS PER FRAME, by weighing each pair's emitted count with |
| `VITASLOP_GXP_CULL` | vitaslop-platform/src/gpu.rs | `VITASLOP_GXP_CULL=0` restores the pre-2026-08-19b "draw both windings". |
| `VITASLOP_GXP_DEBUG` | vitaslop-platform/src/gpu/gxm/build_pipeline.rs | - |
| `VITASLOP_GXP_DEFAULT_UNIFORM_OFFSET` | vitaslop-gxp-shader/src/module.rs | Bytes the driver adds to the DEFAULT uniform buffer's bound address before writing it into |
| `VITASLOP_GXP_DEPTH24` | vitaslop-platform/src/gpu.rs | - |
| `VITASLOP_GXP_DEPTH_PROBE` | vitaslop-gxp-shader/src/link.rs | Turn each stage's SA-bank marker into the reads the body actually needs. |
| `VITASLOP_GXP_DERIVED_CLAIMS` | vitaslop-gxp-shader/src/link.rs | Is the DERIVATION reading in force (`VITASLOP_GXP_DERIVED_CLAIMS=0` turns it off)? An A/B |
| `VITASLOP_GXP_DEST` | vitaslop-gxp-shader/src/module.rs | Whether a fragment program that reads the destination colour is DECLARED as reading it. |
| `VITASLOP_GXP_DEST_BLEND` | vitaslop-gxp-shader/src/module.rs | Whether [`lower_dest_blend`] is allowed to run. |
| `VITASLOP_GXP_DEST_POISON` | vitaslop-gxp-shader/src/module.rs | `VITASLOP_GXP_DEST_POISON=<r,g,b,a>` - the constant [`dest_color_init`] seeds the output |
| `VITASLOP_GXP_DEST_PROBE` | vitaslop-gxp-shader/src/link.rs | Turn each stage's SA-bank marker into the reads the body actually needs. |
| `VITASLOP_GXP_DEST_SEED_LIVE` | vitaslop-gxp-shader/src/module.rs | `VITASLOP_GXP_DEST_SEED_LIVE=0`: the arm back to [`reads_output_bank`] alone - every program |
| `VITASLOP_GXP_DISASM` | vitaslop-gxp-shader/tests/oracle.rs | Compact disassembly of one blob (named by `VITASLOP_GXP_DISASM`, matched as a filename |
| `VITASLOP_GXP_DP_B48_BOTH` | vitaslop-gxp-shader/src/link.rs | `0` holds a repeating DP's internal op2 when bit 47 is clear and bit 48 is set (the old |
| `VITASLOP_GXP_DP_MOE_BIT47` | vitaslop-gxp-shader/src/link.rs | `0` sends every repeating DP back to the intrinsic four-register source walk, bit 47 or not |
| `VITASLOP_GXP_DUAL_SOURCE` | vitaslop-gxp-shader/src/module.rs | Whether a destination-reading program that is LINEAR in the destination may be lowered to a |
| `VITASLOP_GXP_DUAL_TRACE` | vitaslop-gxp-shader/tests/corpus.rs | Print the DUAL-SOURCE plan (or the reason there is none) for every destination reader in |
| `VITASLOP_GXP_DUMP` | vitaslop-platform/src/gpu/gxm/gxp_types.rs | Diagnostic (`VITASLOP_GXP_KEYS=<hex>,<hex>`): recompile ONLY these shader-pair keys |
| `VITASLOP_GXP_DUMPS` | vitaslop-gxp-shader/tests/oracle.rs | Histogram the raw values of named fields across every instruction of a given opcode1 |
| `VITASLOP_GXP_ELIDE_NOOP` | vitaslop-platform/src/gpu.rs | `VITASLOP_GXP_ELIDE_NOOP=0` - the negative control for dropping the colour work of a draw |
| `VITASLOP_GXP_EXCLUDE` | vitaslop-platform/src/gpu/gxm/gxp_types.rs | Pairs forced down the fixed-function path (`VITASLOP_GXP_EXCLUDE`). |
| `VITASLOP_GXP_F16_FCMP` | vitaslop-gxp-shader/src/link.rs | `1`/`0`: the f16 SATURATION as a finite test and a `clamp` (`1`) or as a bit-pattern test |
| `VITASLOP_GXP_F16_RTE` | vitaslop-gxp-shader/src/link.rs | >>> HOW AN f32 NARROWS TO AN f16 - THE NEGATIVE CONTROL FOR THE ROUNDING FIX. |
| `VITASLOP_GXP_FMEM` | vitaslop-platform/src/gpu/gxm/uniforms.rs | Diagnostic (`VITASLOP_GXP_FMEM=<lane>=<value>` / `<lane>*<factor>`, comma separated): |
| `VITASLOP_GXP_FOLD_LITERALS` | vitaslop-gxp-shader/src/link.rs | >>> A WORD KNOWN AT TRANSLATION TIME IS UNPACKED AT TRANSLATION TIME. |
| `VITASLOP_GXP_FORCE` | vitaslop-platform/src/gpu/gxm/gxp_types.rs | Diagnostic (`VITASLOP_GXP_FORCE`): bind a neutral fallback texture for a sampler |
| `VITASLOP_GXP_FRAG` | vitaslop-gxp-shader/tests/corpus.rs | The complete LINKED WGSL for one pair, selected by `VITASLOP_GXP_VERT` + `VITASLOP_GXP_FRAG` |
| `VITASLOP_GXP_FRC_TWO_SOURCE` | vitaslop-gxp-shader/src/wgsl.rs | Whether FRC reads its SECOND source as the value to floor (`src1 - floor(src2)`), the |
| `VITASLOP_GXP_GROUP` | vitaslop-gxp-shader/tests/corpus.rs | Every distinct word of one opcode group across the corpus, with the programs it appears in. |
| `VITASLOP_GXP_HALF_REGS` | vitaslop-gxp-shader/src/link.rs | Give every register the program only ever uses as a PACKED F16 PAIR an UNPACKED home, so |
| `VITASLOP_GXP_IDX_MUL` | vitaslop-gxp-shader/src/usse/mod.rs | Fill in each ORDINARY-REGISTER index load's `stride` - how far apart two consecutive index |
| `VITASLOP_GXP_IDX_PAIRS` | vitaslop-gxp-shader/src/ir.rs | How many registers ONE count of the index register spans for this [`Bank::Indexed`] |
| `VITASLOP_GXP_IDX_REGDEST` | vitaslop-gxp-shader/src/link.rs | `0` sends EVERY group-0x14 index load to the index register, which is what this decoder did |
| `VITASLOP_GXP_IDX_REPEAT` | vitaslop-gxp-shader/src/usse/decode.rs | Whether a load-index word's bits 46:44 repeat it (`VITASLOP_GXP_IDX_REPEAT=0` is the arm |
| `VITASLOP_GXP_IDX_SCALE` | vitaslop-gxp-shader/src/module.rs | How many REGISTERS one count of an index register spans - see [`crate::wgsl`]'s |
| `VITASLOP_GXP_IEEE_RCP` | vitaslop-gxp-shader/src/wgsl.rs | >>> `rcp` AND `rsq` WITH THEIR ZEROES AND INFINITIES DECIDED IN INTEGERS. |
| `VITASLOP_GXP_IMA32_STEP_SRC0` | vitaslop-gxp-shader/src/ir.rs | One STEP of a 32-bit integer multiply-add (group 0x1a, the second 32-bit form): |
| `VITASLOP_GXP_IMAD_SRC1_WHOLE` | vitaslop-gxp-shader/src/link.rs | `1` reads a 16-bit integer MAD's src1 as the WHOLE register when its half-select bit is clear |
| `VITASLOP_GXP_INDEX16` | vitaslop-runtime/src/render.rs | `VITASLOP_GXP_INDEX16=0` restores the OLD behaviour: every index widened to u32 whatever |
| `VITASLOP_GXP_INDEXED_REACH` | vitaslop-gxp-shader/src/module.rs | `VITASLOP_GXP_INDEXED_REACH=0`: size an indexed table's window as declared again - see |
| `VITASLOP_GXP_INPUTS` | vitaslop-gxp-shader/src/usse/mod.rs | Fill in each ORDINARY-REGISTER index load's `stride` - how far apart two consecutive index |
| `VITASLOP_GXP_INPUTS_DIR` | vitaslop-platform/src/gpu/gxm/mod.rs | Whether the once-per-pair `gxp pair <key>: vprog hash ..., fprog hash ...` INDEX should be |
| `VITASLOP_GXP_INPUTS_ORDER` | vitaslop-platform/src/gpu/gxm/inputs.rs | Diagnostic (`VITASLOP_GXP_INPUTS_ORDER=<hex-key>[,<hex-key>]`): print ONE compact line per |
| `VITASLOP_GXP_INPUTS_SETS` | vitaslop-platform/src/gpu/gxm/inputs.rs | - |
| `VITASLOP_GXP_INPUTS_VERTS` | vitaslop-platform/src/gpu/gxm/inputs.rs | Diagnostic (`VITASLOP_GXP_INPUTS=<hex-key>[,<hex-key>]` or `=all`): print, ONCE per |
| `VITASLOP_GXP_INTERP` | vitaslop-platform/src/gpu/gxm/build_pipeline.rs | - |
| `VITASLOP_GXP_INT_PAIRS` | vitaslop-gxp-shader/tests/corpus.rs | >>> EVERY CORPUS PAIR, RE-LINKED WITH EVERY ATTRIBUTE DECLARED A PLAIN INTEGER, VALIDATES. |
| `VITASLOP_GXP_KEEP_MODULES` | vitaslop-platform/src/gpu/gxm/build_pipeline.rs | Whether a linked pair's shader module stays in `GxpLive::modules` once a pipeline has been |
| `VITASLOP_GXP_KEYCOLOR` | vitaslop-platform/src/gpu/gxm/clip.rs | Inject the GXM->WebGPU clip fixup into a linked module: wrap the vertex stage's |
| `VITASLOP_GXP_KEYS` | vitaslop-platform/src/gpu/gxm/gxp_types.rs | Diagnostic (`VITASLOP_GXP_KEYS=<hex>,<hex>`): recompile ONLY these shader-pair keys |
| `VITASLOP_GXP_LIVE` | vitaslop-platform/src/gpu/gxm/gxp_types.rs | Master switch (`VITASLOP_GXP_LIVE`). |
| `VITASLOP_GXP_MAD_MASK16` | vitaslop-gxp-shader/src/usse/decode.rs | The mad-group destination write mask: a BITMASK over the destination's REGISTER LANES. |
| `VITASLOP_GXP_MEM_OFFSET16` | vitaslop-gxp-shader/src/module.rs | The largest byte offset a memory load's REGISTER offset can carry when the instruction that |
| `VITASLOP_GXP_MEM_PEEK` | vitaslop-runtime/src/host/state_mem_windows.rs | `VITASLOP_MEM_DUMP=<hex addr>:<bytes>[,...]`: write those guest byte ranges to |
| `VITASLOP_GXP_MEM_QUAD` | vitaslop-gxp-shader/src/module.rs | Rewrite every run of four consecutive word loads from ONE address - `X = {binding}_word(gxp_aN |
| `VITASLOP_GXP_MEM_STORE` | vitaslop-gxp-shader/src/link.rs | `0` refuses a fragment program that STORES to guest memory (0xF0 in the absolute address |
| `VITASLOP_GXP_MEM_WINDOW_PER_DRAW` | vitaslop-gxp-shader/src/link.rs | `0` sizes a fragment program's statically unbounded memory windows at their DECLARED extent |
| `VITASLOP_GXP_MIPS` | vitaslop-platform/src/gpu/gxm/caches.rs | Whether a chain is built for a seam, ignoring the per-texture exception above. |
| `VITASLOP_GXP_MIPS_GUEST_LEVELS` | vitaslop-platform/src/gpu/gxm/caches.rs | Whether [`upload_gxp_texture`] builds a mip chain for this texture. |
| `VITASLOP_GXP_NAN_SITES` | vitaslop-platform/src/gpu/gxm/clip.rs | Interpret a pair's vertex program over its OWN mesh and measure what it says about the |
| `VITASLOP_GXP_NATIVE_F16` | vitaslop-platform/src/gpu/gxm/renderer_new.rs | Build both pipelines for a `color_format` render target. |
| `VITASLOP_GXP_NEGW` | vitaslop-platform/src/gpu/gxm/gxp_types.rs | How to choose the clip-`w` sign correction (`VITASLOP_GXP_NEGW`). |
| `VITASLOP_GXP_NOBLEND` | vitaslop-platform/src/gpu/gxm/gxp_types.rs | Diagnostic (`VITASLOP_GXP_NOBLEND`): force every recompiled pipeline to REPLACE with |
| `VITASLOP_GXP_NODEPTH` | vitaslop-platform/src/gpu/gxm/gxp_types.rs | Diagnostic (`VITASLOP_GXP_NODEPTH`): every recompiled draw keeps its real shading and |
| `VITASLOP_GXP_NO_STENCIL` | vitaslop-conformance-harness/tests/vita_gxmconf.rs | **SCENE 7 - A STENCIL MASK CONFINES A LATER DRAW TO WHAT AN INVISIBLE DRAW MARKED.** |
| `VITASLOP_GXP_ONLY` | vitaslop-platform/src/gpu/gxm/gxp_types.rs | Render ONLY recompiled draws, skipping the fixed-function draw for any call that |
| `VITASLOP_GXP_PACK_COMP0` | vitaslop-gxp-shader/src/link.rs | `0` restores bit 1 as comp0's high selector bit for a 16-bit PACK source, which is what this |
| `VITASLOP_GXP_PACK_DEST_SLOT` | vitaslop-gxp-shader/src/usse/decode.rs | The SMLSI slot a repeating 0x40 VPCK's DESTINATION steps under: slot 0, the DEST byte |
| `VITASLOP_GXP_PACK_DEST_X2` | vitaslop-gxp-shader/src/usse/decode.rs | - |
| `VITASLOP_GXP_PACK_INTERNAL` | vitaslop-gxp-shader/src/usse/decode.rs | Where each operand of `word` sits for repeat purposes: the destination, then each source in |
| `VITASLOP_GXP_PAIR` | vitaslop-gxp-shader/tests/corpus.rs | Link one named (vertex, fragment) pair and print the COMPLETE WGSL module both stages become. |
| `VITASLOP_GXP_PAIRS` | vitaslop-gxp-shader/tests/corpus.rs | >>> WHY THE PAIRS A RUN ACTUALLY DRAWS FAIL TO LINK, AND WHAT THEIR TWO PROGRAMS DECLARE. |
| `VITASLOP_GXP_PAIR_CORPUS` | vitaslop-gxp-shader/tests/corpus.rs | Rank a pair corpus by the F16 EMULATION it emits: the `gxp_h*` stores and `unpack2x16float`. |
| `VITASLOP_GXP_PAIR_DUAL` | vitaslop-gxp-shader/tests/corpus.rs | Link one named (vertex, fragment) pair and print the COMPLETE WGSL module both stages become. |
| `VITASLOP_GXP_PASS_SPLIT_EVERY` | vitaslop-platform/src/gpu.rs | `VITASLOP_GXP_PASS_SPLIT_EVERY=<n>` cuts a render pass every `n` draws, with no shader |
| `VITASLOP_GXP_POSPROBE` | vitaslop-gxp-shader/src/link.rs | `1` - WHERE DID THIS DRAW'S VERTICES GO? Draw the mesh at CLAMPED normalised coordinates and |
| `VITASLOP_GXP_POSPROBE_DIV` | vitaslop-gxp-shader/src/link.rs | >>> HOW FAR OFF-SCREEN, NOT ONLY THAT IT IS OFF-SCREEN. |
| `VITASLOP_GXP_PRECOMPILE` | vitaslop-platform/src/gpu.rs | Whether the shader pairs the guest's patcher names are TRANSLATED ahead of the draws that bind |
| `VITASLOP_GXP_PRECOMPILE_CROSS` | vitaslop-runtime/src/host/state_shaders.rs | `VITASLOP_GXP_PRECOMPILE_CROSS`: for a title whose `sceGxmShaderPatcherCreateFragmentProgram` |
| `VITASLOP_GXP_PREFETCH_CLAIM` | vitaslop-gxp-shader/src/link.rs | >>> A PREFETCH COORDINATE THAT LANDS ON LANES THE VERTEX NEVER WRITES IS RE-POINTED TO THE |
| `VITASLOP_GXP_PREFETCH_PROJECTIVE` | vitaslop-gxp-shader/src/link.rs | Whether a projective prefetch divides by its `w` - see `container::PrefetchLookup::Projective`. |
| `VITASLOP_GXP_PREFETCH_U8` | vitaslop-gxp-shader/src/link.rs | `0` reads a one-register prefetch as one full-precision component even where the program |
| `VITASLOP_GXP_PREFETCH_UNFED` | vitaslop-gxp-shader/src/link.rs | >>> THE VERTEX PROGRAM PRODUCES NO SUCH TEXCOORD AT ALL, so the coordinate is the |
| `VITASLOP_GXP_PROBE` | vitaslop-gxp-shader/src/module.rs | Diagnostic (`VITASLOP_GXP_PROBE=<bank><idx>[@<instr>][:f32/:bits=<hex>]`, e.g. |
| `VITASLOP_GXP_PROBE_SCALE` | vitaslop-gxp-shader/src/link.rs | Divide a probed value before it is written, so an HDR term reads back under the |
| `VITASLOP_GXP_Q2_VEC` | vitaslop-gxp-shader/src/link.rs | >>> [`GXP_Q2`] ON THE NATIVE ARM: BOTH HALVES IN ONE CONVERSION, ONE PACK AND ONE UNPACK. |
| `VITASLOP_GXP_QUADS` | vitaslop-platform/src/gpu/gxm/inputs.rs | Diagnostic (`VITASLOP_GXP_INPUTS=<hex-key>[,<hex-key>]` or `=all`): print, ONCE per |
| `VITASLOP_GXP_REAL_PAIRS` | vitaslop-gxp-shader/tests/corpus.rs | >>> WHY THE PAIRS A RUN ACTUALLY DRAWS FAIL TO LINK, AND WHAT THEIR TWO PROGRAMS DECLARE. |
| `VITASLOP_GXP_RECOMPILE` | vitaslop-runtime/src/host/state_resolve.rs | - |
| `VITASLOP_GXP_RETURN` | vitaslop-platform/src/gpu/gxm/clip.rs | - |
| `VITASLOP_GXP_RETURN_KEYS` | vitaslop-platform/src/gpu/gxm/build_pipeline.rs | - |
| `VITASLOP_GXP_SA` | vitaslop-platform/src/gpu/gxm/inputs.rs | Diagnostic (`VITASLOP_GXP_SA=<key>:<v/f>:<reg>=<hexword>[,...]`): replace a default-uniform |
| `VITASLOP_GXP_SA_DIRECT` | vitaslop-gxp-shader/src/link.rs | `0` restores the SA copy loop, `unroll` the constant-subscript copy - see [`resolve_sa_init`]. |
| `VITASLOP_GXP_SA_LITERAL_ALWAYS` | vitaslop-gxp-shader/src/link.rs | Is a container literal laid down for EVERY read that names its register |
| `VITASLOP_GXP_SA_UNCLAIMED` | vitaslop-gxp-shader/src/link.rs | Validate that every SA register a stage reads is either inside its default uniform buffer, |
| `VITASLOP_GXP_SIZE_BANKS` | vitaslop-gxp-shader/src/link.rs | `VITASLOP_GXP_SIZE_BANKS=0` restores the pre-2026-08-20b emission - every register bank |
| `VITASLOP_GXP_SOLID` | vitaslop-platform/src/gpu/gxm/gxp_types.rs | Diagnostic (`VITASLOP_GXP_SOLID`): every recompiled draw outputs solid magenta with |
| `VITASLOP_GXP_STATIC_MEM` | vitaslop-gxp-shader/src/module.rs | Resolve at EMIT time every memory load whose address is a window's own base register plus a |
| `VITASLOP_GXP_STREAM_COLUMN_MAX` | vitaslop-runtime/src/host/state_gxm_programs.rs | - |
| `VITASLOP_GXP_STRICT` | vitaslop-platform/src/gpu/gxm/fallback.rs | >>> A PAIR THIS RECOMPILER CANNOT TRANSLATE DROPS ITS DRAWS. |
| `VITASLOP_GXP_TEXCOORD_NUDGE` | vitaslop-platform/src/gpu/gxm/mod.rs | `VITASLOP_GXP_TEXCOORD_NUDGE=<eps>`: add `eps` (normalised units) to every 2-D |
| `VITASLOP_GXP_TSTMSK_DEST` | vitaslop-gxp-shader/src/usse/decode.rs | - |
| `VITASLOP_GXP_UNCLIPPED_DEPTH` | vitaslop-platform/src/gpu.rs | - |
| `VITASLOP_GXP_UNFED_TEXCOORD_DEFAULT` | vitaslop-gxp-shader/src/link.rs | Match the vertex's declared varying outputs to the fragment's declared interpolants BY |
| `VITASLOP_GXP_VARYING_CODE_ORDER` | vitaslop-gxp-shader/src/link.rs | `0` keeps a vertex program's output varyings in the container's order (attribute order or |
| `VITASLOP_GXP_VARYING_LAYOUT` | vitaslop-gxp-shader/src/link.rs | Diagnostic (`VITASLOP_GXP_VARYING_LAYOUT=<vhash>:<usage>@<lane>x<comps>,...`): plan ONE |
| `VITASLOP_GXP_VARYING_ORDER` | vitaslop-gxp-shader/src/link.rs | The vertex lane order the paired FRAGMENT's declaration implies, or `None` when the two |
| `VITASLOP_GXP_VARYING_RESOLVE` | vitaslop-gxp-shader/tests/corpus.rs | Which VERTEX programs the forwarding resolver's lane RESERVATION moves, and where to. |
| `VITASLOP_GXP_VERT` | vitaslop-gxp-shader/tests/corpus.rs | The complete LINKED WGSL for one pair, selected by `VITASLOP_GXP_VERT` + `VITASLOP_GXP_FRAG` |
| `VITASLOP_GXP_VERTEX_PASSTHROUGH` | vitaslop-platform/src/gpu/gxm/gxp_types.rs | This pipeline's vertex layout addresses the GUEST's interleaved row DIRECTLY - its |
| `VITASLOP_GXP_VPROBE` | vitaslop-gxp-shader/src/link.rs | `1` - WHERE DID THIS DRAW'S VERTICES GO? Draw the mesh at CLAMPED normalised coordinates and |
| `VITASLOP_GXP_VP_TRACE` | vitaslop-conformance-harness/tests/vita_gxmconf.rs | WHERE a scene painted, in one line: the count in each quadrant-edge half plus the pixel at |
| `VITASLOP_GXP_WGSL_DIR` | vitaslop-gxp-shader/tests/corpus.rs | The complete LINKED WGSL for one pair, selected by `VITASLOP_GXP_VERT` + `VITASLOP_GXP_FRAG` |
| `VITASLOP_GXP_WGSL_LOG` | vitaslop-platform/src/gpu/gxm/build_pipeline.rs | - |
| `VITASLOP_GXP_WGSL_OUT` | vitaslop-gxp-shader/tests/corpus.rs | Write ONE linked WGSL module per FRAGMENT blob to `VITASLOP_GXP_WGSL_OUT` - the first vertex |
| `VITASLOP_GXP_WGSL_OVERRIDE_DIR` | vitaslop-platform/src/gpu/gxm/renderer_warm.rs | >>> NATIVE: TRANSLATE AND COMPILE THE PAIRS THE PATCHER NAMED, ON BACKGROUND THREADS, |
| `VITASLOP_GXP_WGSL_OVERRIDE_TEXT` | vitaslop-platform/src/gpu/gxm/renderer_warm.rs | >>> NATIVE: TRANSLATE AND COMPILE THE PAIRS THE PATCHER NAMED, ON BACKGROUND THREADS, |
| `VITASLOP_GXP_YFLIP` | vitaslop-platform/src/gpu/gxm/gxp_types.rs | Flip clip Y (`VITASLOP_GXP_YFLIP`, default off). |
| `VITASLOP_GXP_ZFIX` | vitaslop-platform/src/gpu/gxm/gxp_types.rs | Apply the GXM (GL-style, NDC z in [-1,1]) -> WebGPU (z in [0,1]) clip-depth remap |
| `VITASLOP_HB_CMP` | vitaslop-native/tests/homebrew_qsort.rs | Block trace for one run: set `VITASLOP_TRACE_BLOCKS=<lo>-<hi>` (emit-time) and |
| `VITASLOP_HB_CMP_BIT` | vitaslop-native/tests/homebrew_qsort.rs | - |
| `VITASLOP_HB_DUMP` | vitaslop-native/tests/homebrew_qsort.rs | - |
| `VITASLOP_HB_IMAGE` | vitaslop-native/tests/homebrew_strings.rs | A VM over the whole image (the routines read past string ends in aligned words, so |
| `VITASLOP_HB_N` | vitaslop-native/tests/homebrew_qsort.rs | Block trace for one run: set `VITASLOP_TRACE_BLOCKS=<lo>-<hi>` (emit-time) and |
| `VITASLOP_HB_QSORT` | vitaslop-native/tests/homebrew_qsort.rs | Block trace for one run: set `VITASLOP_TRACE_BLOCKS=<lo>-<hi>` (emit-time) and |
| `VITASLOP_HB_STRCMP` | vitaslop-native/tests/homebrew_strings.rs | - |
| `VITASLOP_HB_STRLEN` | vitaslop-native/tests/homebrew_strings.rs | A VM over the whole image (the routines read past string ends in aligned words, so |
| `VITASLOP_HEADLESS_FRAMES` | vitaslop-desktop/src/retail.rs | Headless self-check of the retail path (NO window): load `dir`, optionally drive a |
| `VITASLOP_HEADLESS_NO_TAPS` | vitaslop-desktop/src/retail.rs | Headless self-check of the retail path (NO window): load `dir`, optionally drive a |
| `VITASLOP_HEADLESS_RENDER_FROM` | vitaslop-desktop/src/retail.rs | - `VITASLOP_HEADLESS_SHOT_EVERY` - also write `<shot_dir>/fNNNNNN.png` every N display |
| `VITASLOP_HEADLESS_SHOT_EVERY` | vitaslop-desktop/src/retail.rs | - `VITASLOP_HEADLESS_SHOT_EVERY` - also write `<shot_dir>/fNNNNNN.png` every N display |
| `VITASLOP_HEADLESS_SHOT_FROM` | vitaslop-desktop/src/retail.rs | - `VITASLOP_HEADLESS_SHOT_EVERY` - also write `<shot_dir>/fNNNNNN.png` every N display |
| `VITASLOP_HEADLESS_SHOT_TO` | vitaslop-desktop/src/retail.rs | - `VITASLOP_HEADLESS_SHOT_EVERY` - also write `<shot_dir>/fNNNNNN.png` every N display |
| `VITASLOP_HEADLESS_TIMING` | vitaslop-desktop/src/retail.rs | Headless self-check of the retail path (NO window): load `dir`, optionally drive a |
| `VITASLOP_HEAP_BIG_GROW` | vitaslop-platform/src/heap.rs | >>> THE THREADS BUNDLE'S HEAP: dlmalloc, GROWING THE WASM MEMORY 16 MB AT A TIME. |
| `VITASLOP_HEAP_OWNERS` | vitaslop-platform/src/heap.rs | # The OWNER ledger (`VITASLOP_HEAP_OWNERS=<min KB>`, both engines) |
| `VITASLOP_HEAP_TRACE` | vitaslop-platform/src/heap.rs | # The large-allocation ledger (`VITASLOP_HEAP_TRACE=<min MB>`, native only) |
| `VITASLOP_HITCH_AT` | vitaslop-web/src/lib.rs | - |
| `VITASLOP_HOLD_BUTTONS` | vitaslop-native/tests/retail_boot_probe.rs | A minimal host world: a monotonic clock advancing one 60Hz tick per poll, no |
| `VITASLOP_HOLD_FROM` | vitaslop-native/tests/retail_boot_probe.rs | A minimal host world: a monotonic clock advancing one 60Hz tick per poll, no |
| `VITASLOP_HOLD_MEM` | vitaslop-native/tests/retail_boot_probe.rs | - |
| `VITASLOP_HOLD_TOUCH` | vitaslop-native/tests/retail_boot_probe.rs | - |
| `VITASLOP_HOME` | vitaslop-desktop/src/library.rs | - |
| `VITASLOP_HOSTCALL_WATCH` | vitaslop-runtime/src/host.rs | The `SceGxmPrecomputedDraw` block of that draw, for the per-draw dump - which names the |
| `VITASLOP_HOST_WRITE_WATCH` | vitaslop-runtime/src/host/memory.rs | `VITASLOP_HOST_WRITE_WATCH=<hex addr>[,...]`: report every write a HOST CALL makes to one |
| `VITASLOP_INGEST_DEBUG` | vitaslop-runtime/src/ingest/filesdb.rs | Resolve every non-directory node to its full '/'-separated path (no |
| `VITASLOP_INPLACE_COMPARE` | vitaslop-runtime/src/host/resolve.rs | Whether a draw's VERTEX AND INDEX BYTES are read at `sceGxmEndScene` rather than at the |
| `VITASLOP_INSTANCED_MEMO` | vitaslop-runtime/src/host/snapshot_cache.rs | Whether a draw decodes only the texture units its fragment program DECLARES - see |
| `VITASLOP_IO_BANDWIDTH_KIBPS` | vitaslop-runtime/src/vita/iofilemgr.rs | Modelled sequential read bandwidth, in KiB per second |
| `VITASLOP_IO_JOURNAL_FULL` | vitaslop-runtime/src/host/files.rs | `VITASLOP_IO_JOURNAL_FULL=1`: every read also carries a hash of ALL its bytes. |
| `VITASLOP_IO_PARK_THRESHOLD_US` | vitaslop-runtime/src/vita/iofilemgr.rs | >>> THROUGH THE KNOB SEAM, NOT `std::env` - THE BROWSER HAS NO ENVIRONMENT. |
| `VITASLOP_IO_REQUEST_US` | vitaslop-runtime/src/vita/iofilemgr.rs | Fixed per-request cost in microseconds (`VITASLOP_IO_REQUEST_US`): the command |
| `VITASLOP_IO_WALL_FLOOR` | vitaslop-runtime/src/host/snapshot_cache.rs | `VITASLOP_IO_WALL_FLOOR` (default on): the wall-clock floor that advances the game clock |
| `VITASLOP_JPEG_BENCH` | vitaslop-runtime/src/vita/jpeg.rs | What this decoder costs, on a real image, so "is it fast enough" is a number. |
| `VITASLOP_JS_PROFILE` | vitaslop-web/src/lib.rs | Read a knob from JavaScript - a guest worker's script shares the override table (it is in |
| `VITASLOP_LATE_RETIRE_SUPERSEDED` | vitaslop-runtime/src/capture.rs | `VITASLOP_LATE_RETIRE_SUPERSEDED=0`: the arm back for [`Capture::retire_oldest_untaken_frame`] |
| `VITASLOP_LINK_NO_BODY_MEMO` | vitaslop-gxp-shader/src/link.rs | A stage body's WGSL text, emitted once per distinct program blob: the text is a function of |
| `VITASLOP_LINK_NO_MEMO` | vitaslop-gxp-shader/src/link.rs | >>> A TRANSLATION IS A FUNCTION OF ITS INPUTS, SO IT IS DONE ONCE AND AHEAD WHEN IT CAN BE. |
| `VITASLOP_LINK_NO_RC_MEMO` | vitaslop-gxp-shader/src/link.rs | A program's decode, done once per distinct blob. |
| `VITASLOP_LINK_OLD_F16` | vitaslop-gxp-shader/src/link.rs | Whether `expr` can only ever hold a value that is ALREADY exactly a 16-bit float, so the |
| `VITASLOP_LINK_OLD_PARSE` | vitaslop-gxp-shader/src/link.rs | Split `line` into the store it performs, if it is one of the recognised shapes, and the |
| `VITASLOP_LINK_OLD_REWRITE` | vitaslop-gxp-shader/src/link.rs | [`rewrite_half_line`] appended to `out` - and a line with nothing to rewrite (no kept store, |
| `VITASLOP_LINK_OLD_STRAY` | vitaslop-gxp-shader/src/link.rs | Every bank subscript in `text`: `Ok` names a register, `Err` a bank subscripted by something |
| `VITASLOP_LINK_PROFILE` | vitaslop-gxp-shader/src/link.rs | `VITASLOP_LINK_PROFILE=1` (native only): cumulative microseconds per link phase - |
| `VITASLOP_LINK_VAR_ADDEND` | vitaslop-runtime/src/link.rs | Whether variable-import fixups apply the code word's addend - see [`apply_var_fixups`]. |
| `VITASLOP_LOG` | vitaslop-desktop/src/log.rs | Whether captured events are also written to stderr. |
| `VITASLOP_MAIN_EXEC` | vitaslop-web/src/lib.rs | The title's modules with the executable a `sceAppMgrLoadExec` named in the main |
| `VITASLOP_MAX_FRAMES` | vitaslop-native/tests/retail_boot_probe.rs | - |
| `VITASLOP_MAX_ROUNDS` | vitaslop-native/tests/retail_boot_probe.rs | - |
| `VITASLOP_MEM_CHANGE_WATCH` | vitaslop-runtime/src/host/state_mem_windows.rs | `VITASLOP_MEM_CHANGE_WATCH=<hex addr>:<bytes>[,...]`: at every scene end, report each |
| `VITASLOP_MEM_DUMP` | vitaslop-runtime/src/host/state_mem_windows.rs | `VITASLOP_MEM_DUMP=<hex addr>:<bytes>[,...]`: write those guest byte ranges to |
| `VITASLOP_MEM_DUMP_AT` | vitaslop-runtime/src/host/state_mem_windows.rs | `VITASLOP_MEM_DUMP=<hex addr>:<bytes>[,...]`: write those guest byte ranges to |
| `VITASLOP_MEM_FIND` | vitaslop-runtime/src/host/state_mem_windows.rs | `VITASLOP_MEM_SCAN=<frame>:<hex word>[,<hex word>...]`: at the first scene end at or |
| `VITASLOP_MEM_SCAN` | vitaslop-runtime/src/host/state_mem_windows.rs | `VITASLOP_MEM_SCAN=<frame>:<hex word>[,<hex word>...]`: at the first scene end at or |
| `VITASLOP_MEM_WINDOW_FROM_TARGET` | vitaslop-platform/src/gpu/gxm/renderer_rtt.rs | >>> A MEMORY WINDOW OVER A 64-BIT TARGET THIS FRAME RENDERED READS THAT TARGET. |
| `VITASLOP_MEM_WINDOW_SKIP_COVERED` | vitaslop-platform/src/gpu.rs | `VITASLOP_MEM_WINDOW_SKIP_COVERED=0`: the arm back to reading, copying and uploading every |
| `VITASLOP_MOVIE` | vitaslop-runtime/src/vita/video.rs | The conversions are 90 kHz <-> microseconds and invert each other on whole ticks. |
| `VITASLOP_MOVIE_AUDIO_BACKLOG` | vitaslop-runtime/src/vita/video.rs | How many decoded frames to hold for a title that is not collecting them. |
| `VITASLOP_MOVIE_DUMP_DIR` | vitaslop-runtime/src/vita/avcdec.rs | >>> AND WHAT THE PICTURE ACTUALLY LOOKS LIKE, because "a picture arrived" and "the movie |
| `VITASLOP_MOVIE_DUMP_EVERY` | vitaslop-runtime/src/vita/avcdec.rs | >>> AND WHAT THE PICTURE ACTUALLY LOOKS LIKE, because "a picture arrived" and "the movie |
| `VITASLOP_MOVIE_PICTURE_HASH` | vitaslop-runtime/src/vita/avcdec.rs | Pictures handed to the guest so far, which is what `VITASLOP_MOVIE_PICTURE_HASH` |
| `VITASLOP_MOVIE_SUBSTITUTE` | vitaslop-runtime/src/vita/video.rs | >>> OPEN A DIFFERENT MOVIE THAN THE TITLE ASKED FOR |
| `VITASLOP_MP4_AUDIO` | vitaslop-runtime/src/vita/video.rs | The tracks this engine will hand units for, as cursors, in the order they appear in the |
| `VITASLOP_MP4_AUDIO_READ_AHEAD_MS` | vitaslop-runtime/src/vita/video.rs | Read-ahead for a player that ENABLED an audio stream - see `movie_unit_wait_us`. |
| `VITASLOP_MP4_GUEST_IO_KB` | vitaslop-runtime/src/vita/video.rs | The I/O buffer a callback-read movie is read through, and the largest single read. |
| `VITASLOP_MP4_GUEST_OPS` | vitaslop-runtime/src/vita/video.rs | The body of [`mp4_open_file`], as a plain function so it can use guard clauses. |
| `VITASLOP_MP4_READ_AHEAD_MS` | vitaslop-runtime/src/vita/video.rs | [`READ_AHEAD_US`], or `VITASLOP_MP4_READ_AHEAD_MS` - the arm for a player that keeps its |
| `VITASLOP_MP4_UNITS` | vitaslop-runtime/src/vita/video.rs | `VITASLOP_MP4_UNITS=none`: never return an access unit. |
| `VITASLOP_NAME` | vitaslop-desktop/src/shell.rs | - |
| `VITASLOP_NEON_CACHE` | vitaslop-transpiler/src/emit.rs | Whether emitted modules hold the low NEON bank in locals across a run of vector |
| `VITASLOP_NGS_BLOCK_STATE` | vitaslop-runtime/src/vita/ngs.rs | SceInt32 sceNgsVoiceSetParamsBlock(SceNgsHVoice voice, const SceNgsModuleParamHeader |
| `VITASLOP_NGS_CB_SYNC` | vitaslop-runtime/src/vita/audio.rs | Whether `sceAudioOutOutput` runs the NGS callbacks its mix raised on its own thread before |
| `VITASLOP_NGS_VOICE_HANDLE_MEMO` | vitaslop-runtime/src/vita/ngs.rs | SceInt32 sceNgsSystemRelease(SceNgsHSynSystem system): every rack goes with the system, so |
| `VITASLOP_NGS_VOICE_PEAKS` | vitaslop-runtime/src/vita/at9.rs | `VITASLOP_NGS_VOICE_PEAKS=1` only: this voice's lifetime source peak, sum of squares and |
| `VITASLOP_NGS_VOICE_RAW` | vitaslop-runtime/src/vita/at9.rs | `VITASLOP_NGS_VOICE_RAW=<base>`: every voice's decoded grain, BEFORE any gain, appended to |
| `VITASLOP_NGS_ZERO_LEVEL` | vitaslop-runtime/src/vita/at9.rs | A level of exactly 0.0 is a field the title never wrote, so unity stands |
| `VITASLOP_NID_DIGEST` | vitaslop-runtime/src/host/env.rs | The cross-engine host-call digest for the frame in progress - see [`NidDigest`]. |
| `VITASLOP_NONLOCAL_RETURNS` | vitaslop-transpiler/src/emit.rs | >>> A RETURN THAT DOES NOT GO TO ITS CALLER IS A NON-LOCAL TRANSFER, AND IS FOLLOWED THERE. |
| `VITASLOP_NO_BC` | vitaslop-runtime/src/render.rs | Decode a whole BC1/BC2/BC3 block to its sixteen RGBA8 texels at once. |
| `VITASLOP_NO_FAST_IMPORT` | vitaslop-runtime/src/vita/mod.rs | Whether `func_nid`'s handler can only ever CONTINUE, so the transpiler may route the |
| `VITASLOP_NO_INLINE_CLIB` | vitaslop-runtime/src/vita/mod.rs | `VITASLOP_NO_INLINE_CLIB`: route `sceClibMemcpy`, `sceClibMemset` and `sceClibMemcmp` |
| `VITASLOP_NO_INLINE_DELAY` | vitaslop-runtime/src/vita/mod.rs | `VITASLOP_NO_INLINE_DELAY`: route every `sceKernelDelayThread` through the host, leaving |
| `VITASLOP_NO_INLINE_IMPORTS` | vitaslop-runtime/src/host.rs | >>> WHO HAS WRITTEN THE CONTEXT'S TEXTURE SLOTS, counted for the failure report above. |
| `VITASLOP_NO_INLINE_LWMUTEX` | vitaslop-runtime/src/vita/mod.rs | `VITASLOP_NO_INLINE_LWMUTEX`: route the lightweight-mutex lock and unlock through the |
| `VITASLOP_NO_INLINE_MUTEX` | vitaslop-runtime/src/vita/mod.rs | `VITASLOP_NO_INLINE_MUTEX`: route `sceKernelLockMutex`/`sceKernelUnlockMutex` through the |
| `VITASLOP_NO_INLINE_RESERVE` | vitaslop-runtime/src/vita/mod.rs | `VITASLOP_NO_INLINE_RESERVE`: route `sceGxmReserve{Vertex,Fragment}DefaultUniformBuffer` |
| `VITASLOP_NO_INLINE_STUBS` | vitaslop-runtime/src/vita/mod.rs | `VITASLOP_NO_INLINE_STUBS`: route the constant-return stubs through the host, leaving |
| `VITASLOP_NO_INLINE_TEXTURE` | vitaslop-runtime/src/host/state_gxm_bind.rs | Report - ONCE - a draw whose fragment program DECLARES samplers and whose texture |
| `VITASLOP_NO_INLINE_UNIFORM_DATA` | vitaslop-runtime/src/vita/mod.rs | `VITASLOP_NO_INLINE_UNIFORM_DATA`: route `sceGxmSetUniformDataF` through the host, |
| `VITASLOP_NO_NGS_MIX` | vitaslop-runtime/src/vita/audio.rs | `VITASLOP_NO_NGS_MIX`: skip the NGS decode-and-mix entirely, leaving the guest's |
| `VITASLOP_OUTPUT_SIZE` | vitaslop-web/src/lib.rs | - |
| `VITASLOP_PACE_FLOOR_FREE` | vitaslop-web/src/lib.rs | Whether the live loop leaves the wall floor's clock gain out of a frame's charge - see the |
| `VITASLOP_PACE_REFUND` | vitaslop-web/src/lib.rs | Whether the pacing loop refunds the one-period floor it charged short frames out of later |
| `VITASLOP_PACKED_CACHE_MB` | vitaslop-platform/src/gpu/gxm/caches.rs | >>> AND A CAP IN ENTRIES IS NOT A BOUND ON MEMORY. |
| `VITASLOP_PARALLEL_PIPELINES` | vitaslop-platform/src/gpu/gxm/renderer_warm.rs | >>> NATIVE: BUILD EVERY PIPELINE THESE SCENES WILL NEED AND DO NOT HAVE, ON SEVERAL |
| `VITASLOP_PATCH_STUBS` | vitaslop-native/tests/retail_boot_probe.rs | - |
| `VITASLOP_PAUSE_ON_BLUR` | vitaslop-desktop/src/retail.rs | Run the retail title in `dir` in a live window until the window closes or the guest |
| `VITASLOP_PEEK` | vitaslop-desktop/src/retail.rs | Guest memory at `addr`, for `VITASLOP_PEEK`. |
| `VITASLOP_PERF` | vitaslop-desktop/src/retail.rs | The `top` host calls by total time since the profiler was last reset, one line each - |
| `VITASLOP_PERF_CONSOLE` | vitaslop-web/src/lib.rs | Whether the per-window performance report is also written to the browser CONSOLE |
| `VITASLOP_PERF_FROM` | vitaslop-desktop/src/retail.rs | The frame (and the moment) the profilers were reset at - see `VITASLOP_PERF_FROM`. |
| `VITASLOP_PIPELINE_DEFER` | vitaslop-web/src/lib.rs | >>> THE PIPELINES THIS FRAME NEEDS, CREATED ASYNC - see `GxmRenderer::warm_pipelines` |
| `VITASLOP_PIPELINE_DEFER_MAX_MS` | vitaslop-web/src/lib.rs | `VITASLOP_PIPELINE_DEFER_MAX_MS`: overrides [`PIPE_DEFER_MAX_MS_DEFAULT`]. |
| `VITASLOP_PIPE_LOG` | vitaslop-platform/src/gpu/gxm/gxp_types.rs | `VITASLOP_PIPE_LOG=1`: one status line per pipeline built - the frame, the pair, and the |
| `VITASLOP_PIXEL_TRACE` | vitaslop-runtime/src/render.rs | Draw one scene onto an EXISTING framebuffer and depth buffer, composing with whatever |
| `VITASLOP_PKG` | vitaslop-runtime/src/ingest/stream.rs | A pkg's item table, read over a source that only ever hands out RANGES |
| `VITASLOP_POISON_UNRESOLVED_VARS` | vitaslop-runtime/src/link.rs | - |
| `VITASLOP_POKE` | vitaslop-native/tests/retail_boot_probe.rs | - |
| `VITASLOP_POLL_ADDR` | vitaslop-native/src/threaded.rs | Guest address to sample after each host call, from `VITASLOP_POLL_ADDR` (hex). |
| `VITASLOP_POOL_SMALL_BUFFERS` | vitaslop-platform/src/gpu/gxm/buffers.rs | A buffer of exactly this size and usage, if one is waiting. |
| `VITASLOP_PREPARE_SPLIT` | vitaslop-platform/src/gpu/gxm/prepare.rs | Where the milliseconds INSIDE one `prepare` go, plus the bytes each phase moved. |
| `VITASLOP_PREPOKE` | vitaslop-native/tests/retail_boot_probe.rs | - |
| `VITASLOP_PRESENT_LOG` | vitaslop-web/src/lib.rs | Render one freshly-executed FRAME - every scene the guest submitted between |
| `VITASLOP_PRESENT_PROBE` | vitaslop-web/src/lib.rs | Reads back WHAT WE PRESENTED, when `VITASLOP_PRESENT_PROBE` asks for it. |
| `VITASLOP_PRESENT_SHOT` | vitaslop-web/src/lib.rs | If a mapped read is ready, describe it and release the buffer. |
| `VITASLOP_PROMOTE_POISON` | vitaslop-transpiler/src/emit.rs | `VITASLOP_PROMOTE_POISON=<n>` - the FALSIFIER for register promotion. |
| `VITASLOP_PROMOTE_REGS` | vitaslop-native/src/threaded.rs | A concise trap description (kind + message), matching the sync `Vm`'s detail. |
| `VITASLOP_PSARC` | vitaslop-runtime/src/psarc.rs | Read a REAL archive: `VITASLOP_PSARC=<path to a .psarc>`, optionally |
| `VITASLOP_PSARC_FILE` | vitaslop-runtime/src/psarc.rs | Read a REAL archive: `VITASLOP_PSARC=<path to a .psarc>`, optionally |
| `VITASLOP_PVRTC_DECODE` | vitaslop-runtime/src/render.rs | Whether PVRTC decodes a whole face at a time (the default) or one texel at a time. |
| `VITASLOP_QUANTUM_CPU_US` | vitaslop-runtime/src/host/env.rs | Game-clock time charged for one [`QUANTUM_ARM`] of guest execution, in microseconds. |
| `VITASLOP_QUANTUM_FUEL` | vitaslop-native/tests/retail_boot_probe.rs | - |
| `VITASLOP_RAW_WORD_VIEW` | vitaslop-platform/src/gpu.rs | Whether a sampler reading one 32-bit word of a raw 64-bit surface as a double-width texture |
| `VITASLOP_REACH_MEMO` | vitaslop-gxp-shader/src/reach.rs | `VITASLOP_REACH_MEMO=0` walks the program for every draw, the arm back from the answer memo |
| `VITASLOP_REGION_CLIP_SCENE` | vitaslop-runtime/src/vita/gxm.rs | - |
| `VITASLOP_REGTRACE` | vitaslop-native/src/threaded.rs | `VITASLOP_REGTRACE=<lo>-<hi>:<path>` - append the reg+flag file per block entry in |
| `VITASLOP_REGTRACE_MAX` | vitaslop-native/src/threaded.rs | `VITASLOP_REGTRACE_MAX=<n>` caps the register trace at `n` lines (0 = unbounded). |
| `VITASLOP_REGTRACE_VFP` | vitaslop-native/src/threaded.rs | Append one `pc r0..r15 n z c v [mADDR=VAL...] tTHID` line (all hex, flags 0/1) to the |
| `VITASLOP_REGTRACE_WATCH` | vitaslop-native/src/threaded.rs | The `VITASLOP_REGTRACE_WATCH` words, formatted as ` mADDR=VALUE` fields ready to append |
| `VITASLOP_REPLAY_BGRA` | vitaslop-web/src/frame_replay.rs | - |
| `VITASLOP_REPLAY_EARLY` | vitaslop-web/src/frame_replay.rs | - |
| `VITASLOP_REPLAY_TARGET` | vitaslop-web/src/frame_replay.rs | - |
| `VITASLOP_RESIDENT_COMPACT_EARLY` | vitaslop-platform/src/gpu.rs | The 2026-09-28 memory bounds, each with its arm back (`=0`): `VITASLOP_SUBRECT_POOL` |
| `VITASLOP_RESIDENT_GEOM` | vitaslop-platform/src/gpu/gxm/gxp_types.rs | Repacked vertices and expanded indices that have not changed since the renderer first |
| `VITASLOP_RESIDENT_GEOM_MB` | vitaslop-platform/src/gpu/gxm/gxp_types.rs | The byte budget for each of the two heaps (`VITASLOP_RESIDENT_GEOM_MB`, per heap). |
| `VITASLOP_RESIDENT_IDLE_FRAMES` | vitaslop-platform/src/gpu/gxm/resident.rs | >>> FORGET THE SLICES NO DRAW HAS BOUND FOR `idle` FRAMES - their `Arc`s are RUST |
| `VITASLOP_RESOLVE_CHUNK` | vitaslop-runtime/src/host/resolve.rs | `VITASLOP_RESOLVE_CHUNK=<draws>`: how many draws the RESOLVER reads per hold of the snapshot |
| `VITASLOP_ROUNDS_PER_FRAME` | vitaslop-native/tests/retail_boot_probe.rs | - |
| `VITASLOP_RTT_BG_CACHE` | vitaslop-platform/src/gpu.rs | `VITASLOP_RTT_BG_CACHE=0` restores the OLD behaviour: a sampler bind group naming a render |
| `VITASLOP_RTT_CLEAR_PROBE` | vitaslop-native/src/wgpu_render.rs | - |
| `VITASLOP_RTT_GAMMA_BY_TEXTURE` | vitaslop-platform/src/gpu.rs | Whether a sampler over a GAMMA-CORRECT render target decodes by the TEXTURE's own gamma mode |
| `VITASLOP_RTT_GRID_FRAMES` | vitaslop-web/src/browser_sched.rs | - |
| `VITASLOP_RTT_PROBE_FIND` | vitaslop-runtime/src/rtt_writeback.rs | TEMPORARY TELEMETRY (`VITASLOP_RTT_PROBE_FIND=<rrggbb>[+<rrggbb>...]`): name every |
| `VITASLOP_RTT_PROBE_LOG` | vitaslop-runtime/src/rtt_writeback.rs | TEMPORARY TELEMETRY (`VITASLOP_RTT_PROBE_LOG=1`): print every written-back target's |
| `VITASLOP_RTT_SELF_WINDOW` | vitaslop-platform/src/gpu.rs | `VITASLOP_RTT_SELF_WINDOW=0` is the arm back for a draw sampling a window of the target its |
| `VITASLOP_RTT_SELF_WINDOW_SCENE_START` | vitaslop-platform/src/gpu.rs | Whether a self window lying under its reader's own pixels is filled from the attachment as |
| `VITASLOP_RTT_STALE_EXTENT` | vitaslop-platform/src/gpu.rs | Whether a held render target is refused as a sampler source when this frame has written its |
| `VITASLOP_RTT_SUBRECT` | vitaslop-conformance-harness/tests/vita_gxmconf_real.rs | **SCENES 16 + 17 - A TEXTURE NAMING A SUB-RECTANGLE OF A RENDERED TARGET SAMPLES THAT RECTANGLE.** |
| `VITASLOP_RTT_SUBRECT_HELD` | vitaslop-conformance-harness/tests/vita_gxmconf_real.rs | **SCENES 16 + 17, AT THE BASE - A TEXTURE AT A TARGET'S OWN ADDRESS, SMALLER THAN IT, IS ITS |
| `VITASLOP_RTT_SUBRECT_HELD_INTERIOR` | vitaslop-conformance-harness/tests/vita_gxmconf_real.rs | **SCENES 16 + 17, ACROSS FRAMES - A SUB-RECTANGLE OF A TARGET AN EARLIER FRAME RENDERED IS |
| `VITASLOP_RTT_WRITEBACK_DELAY_MS` | vitaslop-web/src/lib.rs | `VITASLOP_RTT_WRITEBACK_DELAY_MS`: a copy is not handed over, and its slot stays mapped, |
| `VITASLOP_RTT_WRITEBACK_FLOAT` | vitaslop-runtime/src/rtt_writeback.rs | Write ONE target's rendered pixels (`rgba`, tightly packed `w*h*4`, memory order R,G,B,A) |
| `VITASLOP_RTT_WRITEBACK_MAX_AGE_MS` | vitaslop-web/src/lib.rs | >>> THE WRITEBACK AGE BOUND: a present DECLINES while the oldest render-target copy |
| `VITASLOP_RTT_WRITEBACK_SYNC_MS` | vitaslop-web/src/lib.rs | - |
| `VITASLOP_SAMPLER_NARROW` | vitaslop-runtime/src/host/snapshot_cache.rs | Whether a draw decodes only the texture units its fragment program DECLARES - see |
| `VITASLOP_SCAN_WORD` | vitaslop-native/tests/retail_boot_probe.rs | - |
| `VITASLOP_SCENE_LIMIT` | vitaslop-native/tests/retail_boot_probe.rs | - |
| `VITASLOP_SCHED_CORES` | vitaslop-runtime/src/sched.rs | `VITASLOP_SCHED_CORES=<n>`: cap the baton to the top `n` runnable PRIORITIES, as the |
| `VITASLOP_SCHED_RR` | vitaslop-runtime/src/sched.rs | `VITASLOP_SCHED_RR=1`: round-robin every runnable thread, ignoring priority. |
| `VITASLOP_SCHED_TRACE` | vitaslop-runtime/src/sched.rs | `VITASLOP_SCHED_TRACE=<from>-<to>` (display frames, inclusive): print one line per |
| `VITASLOP_SEMA_TRAIL` | vitaslop-runtime/src/vita/sync.rs | TEMPORARY DIAGNOSTIC. |
| `VITASLOP_SET_EVF` | vitaslop-native/tests/retail_boot_probe.rs | - |
| `VITASLOP_SHELL_MENU_AT` | vitaslop-desktop/src/shell.rs | `VITASLOP_SHELL_SHOT=<png>` (+ `VITASLOP_SHELL_SCREEN=library/settings/settings:<ID>/import/ |
| `VITASLOP_SHELL_SCREEN` | vitaslop-desktop/src/shell.rs | `VITASLOP_SHELL_SHOT=<png>` (+ `VITASLOP_SHELL_SCREEN=library/settings/settings:<ID>/import/ |
| `VITASLOP_SHELL_SHOT` | vitaslop-desktop/src/shell.rs | `VITASLOP_SHELL_SHOT` - see [`ShellShot`]. |
| `VITASLOP_SHELL_SHOT_AT` | vitaslop-desktop/src/shell.rs | `VITASLOP_SHELL_SHOT=<png>` (+ `VITASLOP_SHELL_SCREEN=library/settings/settings:<ID>/import/ |
| `VITASLOP_SHOT_DIR` | vitaslop-native/tests/retail_boot_probe.rs | Read and format one watched value from current guest memory. |
| `VITASLOP_SHOT_FRAMES` | vitaslop-web/src/lib.rs | `VITASLOP_SHOT_FRAMES=<f>,<f>,...`: GUEST frames to photograph, oldest first - the first |
| `VITASLOP_SHOT_LAST` | vitaslop-native/tests/retail_boot_probe.rs | - |
| `VITASLOP_SIGNATURE` | vitaslop-desktop/src/retail.rs | - |
| `VITASLOP_SIGNATURE_EVERY` | vitaslop-desktop/src/retail.rs | The running determinism signature at frame `f`, as the browser prints it: `sigtrace` every |
| `VITASLOP_SLOW_FRAME_US` | vitaslop-web/src/lib.rs | >>> THE WRITEBACK AGE BOUND: a present DECLINES while the oldest render-target copy |
| `VITASLOP_SLOW_PAIRS` | vitaslop-gxp-shader/tests/corpus.rs | Time each pair a run dumped as a SLOW in-frame translation (`VITASLOP_DUMP_SLOW_PAIRS`: |
| `VITASLOP_SMP` | vitaslop-conformance-harness/src/lib.rs | The SAME corpus, transpiled as an SMP build (`VITASLOP_SMP`'s codegen: atomic |
| `VITASLOP_SMP_ASYNC_RESOLVE` | vitaslop-native/src/smp.rs | The flip's geometry and texture-set reads on a RESOLVER thread of their own, applied by the |
| `VITASLOP_SMP_DEFER_TEXTURES` | vitaslop-runtime/src/host/resolve.rs | >>> A DRAW'S TEXTURE BINDINGS, PROVEN WHERE ITS GEOMETRY IS READ. |
| `VITASLOP_SMP_EARLY_PAUSE_ALL` | vitaslop-web/src/smp.rs | Pause every guest worker for a small-target completion's whole render + readback |
| `VITASLOP_SMP_FIBER_NEAR` | vitaslop-native/src/smp.rs | `VITASLOP_SMP_FIBER_NEAR=1`: a fiber's backing thread is placed on its runner's worker. |
| `VITASLOP_SMP_FORWARD` | vitaslop-runtime/src/vita/mod.rs | [`smp_owner_only`] for a run that knows whether its guest workers each hold their own view |
| `VITASLOP_SMP_GUEST_SLOW` | vitaslop-runtime/src/perf.rs | `VITASLOP_SMP_SLOW_EXCLUDE_GXM=1` - a MEASUREMENT RIG for the phone proxy only. |
| `VITASLOP_SMP_LATE_PRESENT` | vitaslop-native/src/smp.rs | A SLOW PRESENT DOES NOT FREEZE THE GAME. |
| `VITASLOP_SMP_LOCKSTEP` | vitaslop-native/src/smp.rs | >>> THE EMULATED CORES RUN IN LOCKSTEP. |
| `VITASLOP_SMP_LWMUTEX_INLINE` | vitaslop-runtime/src/vita/mod.rs | >>> WHICH INLINE FORMS SURVIVE PARALLEL GUEST EXECUTION, and what the rest become. |
| `VITASLOP_SMP_OVERLAP` | vitaslop-native/src/smp.rs | Present frame N while the guest workers build N+1: the gate stands ONE frame ahead of the |
| `VITASLOP_SMP_PLACE` | vitaslop-native/src/smp.rs | `VITASLOP_SMP_PLACE` - the browser's placement modes, `apart` the default. |
| `VITASLOP_SMP_RT_THROUGH_GATE` | vitaslop-native/src/smp.rs | Real-time (wall-parked) threads run behind a shut gate. |
| `VITASLOP_SMP_SLOW_EXCLUDE_GXM` | vitaslop-runtime/src/perf.rs | `VITASLOP_SMP_SLOW_EXCLUDE_GXM=1` - a MEASUREMENT RIG for the phone proxy only. |
| `VITASLOP_SMP_SPIN_CAP_US` | vitaslop-web/src/smp.rs | >>> ADAPTIVE SPIN: the ceiling, in microseconds, on how long a guest worker polls its doorbell |
| `VITASLOP_SMP_SPIN_US` | vitaslop-web/src/smp.rs | `VITASLOP_SMP_SPIN_US`: how long a guest worker with nothing to run polls its doorbell |
| `VITASLOP_SMP_TIMEOUT_YIELDS` | vitaslop-native/src/smp.rs | A thread released by TIME yields to the threads already runnable on its worker - see |
| `VITASLOP_SMP_TRACE` | vitaslop-web/src/smp.rs | Whether `VITASLOP_SMP_TRACE` is armed at all - for a caller that would otherwise take |
| `VITASLOP_SMP_UNPIN_STARVED` | vitaslop-native/src/smp.rs | `VITASLOP_SMP_UNPIN_STARVED=1`: the browser's opt-in starvation unpin. |
| `VITASLOP_SMP_WORKERS` | vitaslop-native/src/smp.rs | Guest workers: `VITASLOP_SMP_WORKERS`, default 3 - the Vita gives a title three cores, and |
| `VITASLOP_SNAPSHOT` | vitaslop-native/src/threaded.rs | `VITASLOP_SNAPSHOT=<hexpc>:<path>` - dump full state on first entry to block `hexpc`. |
| `VITASLOP_SNAPSHOT_BUDGET_MB` | vitaslop-runtime/src/host/snapshot_cache.rs | Byte budget for retained texture snapshots, scaled to the device |
| `VITASLOP_SNAPSHOT_DENSE` | vitaslop-native/src/threaded.rs | Dump the full guest state (all non-zero pages + r0..r15 + NZCV) to `path`, in the |
| `VITASLOP_SNAPSHOT_SKIP` | vitaslop-native/src/threaded.rs | `VITASLOP_SNAPSHOT_SKIP=<n>` - skip the first `n` entries to the snapshot block before |
| `VITASLOP_SOFTWARE` | vitaslop-desktop/src/retail.rs | - |
| `VITASLOP_SSAA` | vitaslop-platform/src/gpu/gxm/renderer_new.rs | Set the supersample factor: 1 (default) renders the scene straight into the caller's |
| `VITASLOP_STAGING_TRIM` | vitaslop-platform/src/gpu.rs | The 2026-09-28 memory bounds, each with its arm back (`=0`): `VITASLOP_SUBRECT_POOL` |
| `VITASLOP_STALL_CHUNK` | vitaslop-native/tests/retail_boot_probe.rs | - |
| `VITASLOP_STALL_WAKE` | vitaslop-native/tests/retail_boot_probe.rs | - |
| `VITASLOP_STALL_WATCHDOG` | vitaslop-native/src/watchdog.rs | The configured stall budget in seconds, from `VITASLOP_STALL_WATCHDOG`. |
| `VITASLOP_STALL_WAVES` | vitaslop-native/tests/retail_boot_probe.rs | - |
| `VITASLOP_STAMP_VERIFY` | vitaslop-native/src/threaded.rs | A host write into guest memory - the probe's POKE knob, and the render-target WRITEBACK |
| `VITASLOP_STRICT_DRAWS` | vitaslop-runtime/src/render.rs | Why [`RenderSceneBuilder::build`] discarded draws from a captured scene. |
| `VITASLOP_STRUCTURED_CF` | vitaslop-transpiler/src/emit.rs | Multi-block functions emitted as structured control flow (`structured`), and those |
| `VITASLOP_SUBRECT_POOL` | vitaslop-platform/src/gpu.rs | The 2026-09-28 memory bounds, each with its arm back (`=0`): `VITASLOP_SUBRECT_POOL` |
| `VITASLOP_SWITCH_WHY` | vitaslop-transpiler/src/lower.rs | Whether the table-branch diagnostic is on for this address |
| `VITASLOP_SW_CHAIN` | vitaslop-native/src/wgpu_render.rs | `VITASLOP_GPU_CHAIN_DIR=<dir>`: write every offscreen target of the frame just |
| `VITASLOP_SW_CHAIN_DIR` | vitaslop-runtime/src/render.rs | - |
| `VITASLOP_SW_POST` | vitaslop-runtime/src/render.rs | - |
| `VITASLOP_SYNC_RESOLVE_ASYNC` | vitaslop-runtime/src/host/resolve.rs | `VITASLOP_SYNC_RESOLVE_ASYNC=0` - the arm back to reading a sync point's geometry inline under |
| `VITASLOP_SYNC_RESOLVE_ON_WORKER` | vitaslop-web/src/browser_sched.rs | >>> A SYNC POINT'S RESOLVE-ONLY PARK, SETTLED ON THE GUEST'S OWN WORKER. |
| `VITASLOP_SYSTEM_FONT` | vitaslop-runtime/src/font/system.rs | The resolved substitute: its bytes and a human-readable account of where they came from. |
| `VITASLOP_TAIL_CALLS` | vitaslop-transpiler/src/emit.rs | >>> A CALL THAT IS IMMEDIATELY FOLLOWED BY A RETURN IS EMITTED AS A WASM TAIL CALL. |
| `VITASLOP_TEXENC_SHRINK_SCRATCH` | vitaslop-platform/src/texenc.rs | >>> HOW BIG THE SHARED SOURCE BUFFER SHOULD BE for a texture needing `need_src` bytes, given |
| `VITASLOP_TEXTURE_CHECK` | vitaslop-runtime/src/host/snapshot_cache.rs | How a retained texture snapshot is re-validated (`VITASLOP_TEXTURE_CHECK`): `scene` |
| `VITASLOP_TEX_CACHE_MB` | vitaslop-platform/src/gpu/gxm/mod.rs | Upper bound on the cross-frame texture caches. |
| `VITASLOP_TEX_COMPRESS` | vitaslop-runtime/src/render.rs | Whether compressed textures reach the GPU compressed at all. |
| `VITASLOP_TEX_DIRTY_CENSUS` | vitaslop-runtime/src/host/memory.rs | >>> WHICH PARTS of `[off, off + len)` the guest may have stored into since `stamp`, |
| `VITASLOP_TEX_ENCODE_BUDGET_MS` | vitaslop-runtime/src/render.rs | >>> HOW MUCH OF ONE FRAME THE INLINE BLOCK ENCODE MAY SPEND, in milliseconds. |
| `VITASLOP_TEX_ENCODE_DEFER` | vitaslop-platform/src/gpu.rs | >>> A TEXTURE'S ETC2 ENCODE IS SPREAD OVER LATER FRAMES, AND THE FRAME THAT BINDS IT DRAWS |
| `VITASLOP_TEX_ENCODE_DEFER_UNITS` | vitaslop-platform/src/gpu.rs | Block WORK UNITS the deferred encoder spends per frame (`VITASLOP_TEX_ENCODE_DEFER_UNITS`): |
| `VITASLOP_TEX_ENCODE_RESUME` | vitaslop-runtime/src/render.rs | `VITASLOP_TEX_ENCODE_RESUME=0` restores the OLD behaviour: an encode that starts runs to |
| `VITASLOP_TEX_MEMO_PER_SCENE` | vitaslop-runtime/src/host/snapshot_cache.rs | A whole DRAW's worth of snapshotted textures, by the bindings that produced it - kept |
| `VITASLOP_TEX_PAGE_READ` | vitaslop-runtime/src/host/snapshots.rs | Record that this entry's bytes are current as of THIS SCENE, so a later |
| `VITASLOP_TEX_RECENT_FRAMES` | vitaslop-platform/src/gpu.rs | >>> A TEXTURE IS KEPT WHILE THE GUEST STILL HOLDS ITS BYTES - THE VITA HAS NO TEXTURE CACHE. |
| `VITASLOP_TEX_RETAIN_MB` | vitaslop-platform/src/gpu.rs | How many bytes of GPU texture the view cache may RETAIN, as opposed to how many the |
| `VITASLOP_THREAD_TRAILS_AT` | vitaslop-runtime/src/host/env.rs | `VITASLOP_THREAD_TRAILS_AT=<frame>` - see its use in the host-call dispatch. |
| `VITASLOP_THREAD_TRAILS_NID` | vitaslop-runtime/src/host/env.rs | `VITASLOP_THREAD_TRAILS_NID=<hex>` - see its use in the host-call dispatch. |
| `VITASLOP_TRACE_BLOCKS` | vitaslop-native/src/threaded.rs | `VITASLOP_TRACE_FRAMES=<from>-<to>` (decimal display frames, inclusive) - print the |
| `VITASLOP_TRACE_EXIT` | vitaslop-native/tests/retail_boot_probe.rs | - |
| `VITASLOP_TRACE_FILE` | vitaslop-runtime/src/vita/libkernel.rs | Diagnostic (`RUST_LOG=vitaslop::exit=debug`): when the guest calls |
| `VITASLOP_TRACE_FRAMES` | vitaslop-native/src/threaded.rs | `VITASLOP_TRACE_FRAMES=<from>-<to>` (decimal display frames, inclusive) - print the |
| `VITASLOP_TRACE_FUNCS` | vitaslop-native/src/threaded.rs | Bind `env.svc`. |
| `VITASLOP_TRACE_INDIRECT` | vitaslop-transpiler/src/emit.rs | Diagnostic indirect-call tracer. |
| `VITASLOP_TRACE_IO` | vitaslop-native/tests/retail_boot_probe.rs | - |
| `VITASLOP_TRACE_ORDER` | vitaslop-runtime/src/vita/mod.rs | Ordered-timeline trace (env `VITASLOP_TRACE_ORDER`): print every *meaningful* |
| `VITASLOP_TRACE_ORDER_FULL` | vitaslop-runtime/src/vita/mod.rs | - |
| `VITASLOP_TRACK_PC` | vitaslop-transpiler/src/abi.rs | Exported name of the diagnostic guest-PC tracker global. |
| `VITASLOP_TRANSPILE_REPORT` | vitaslop-native/src/threaded.rs | - |
| `VITASLOP_TRAP_HALT` | vitaslop-transpiler/src/emit.rs | When `VITASLOP_TRAP_HALT` is set, a `Term::Halt` (a block that ran off the end of decoded |
| `VITASLOP_TS_DELAY_MS` | vitaslop-web/src/lib.rs | `VITASLOP_TS_DELAY_MS`: a test rig - see where it holds the poll in `present`. |
| `VITASLOP_UNIFORM_WATCH` | vitaslop-runtime/src/vita/gxm.rs | `VITASLOP_UNIFORM_WATCH=<hex address>/<parameter name substring>[,...]`: report every |
| `VITASLOP_UV_DEBUG` | vitaslop-runtime/src/render.rs | Draw one scene onto an EXISTING framebuffer and depth buffer, composing with whatever |
| `VITASLOP_VBLANK_PARK` | vitaslop-runtime/src/vita/display.rs | Whether an inlined `sceDisplayGetVcount` carries the spin guard (`VITASLOP_VBLANK_PARK`, |
| `VITASLOP_VERTEX_INTERN` | vitaslop-runtime/src/host/snapshot_cache.rs | A cheap, allocation-free fingerprint of a vertex stream, for [`TextureSnapshots:: |
| `VITASLOP_VERTEX_INTERN_USE` | vitaslop-runtime/src/host/snapshot_cache.rs | A cheap, allocation-free fingerprint of a vertex stream, for [`TextureSnapshots:: |
| `VITASLOP_VIDEO_HARDWARE` | vitaslop-platform/src/video.rs | >>> A GUEST MOVIE DECODES IN SOFTWARE ON THE DESKTOP. |
| `VITASLOP_VIEWPORT_SCENE` | vitaslop-runtime/src/vita/gxm.rs | - |
| `VITASLOP_VPK` | vitaslop-runtime/src/ingest/stream.rs | A homebrew VPK (`VITASLOP_VPK=<file>`): probed as `vpk`, imported as files + a |
| `VITASLOP_WARM_BY_FORMAT` | vitaslop-platform/src/gpu.rs | Whether `warm_pipelines` predicts a never-encoded target's pipelines from its guest colour |
| `VITASLOP_WARM_NEW_TARGETS` | vitaslop-platform/src/gpu/gxm/renderer_warm.rs | >>> START, WITH `createRenderPipelineAsync`, EVERY PIPELINE THESE SCENES WILL NEED |
| `VITASLOP_WASM_INDICES` | vitaslop-native/src/threaded.rs | Rewrite `<wasm function N>` in a trap backtrace to name the GUEST function it is. |
| `VITASLOP_WASM_NAMES` | vitaslop-transpiler/src/emit.rs | When `VITASLOP_WASM_NAMES` is set, emit a wasm `name` custom section labelling |
| `VITASLOP_WATCH_` | vitaslop-transpiler/src/emit.rs | Number of matching store-watchpoint hits to skip before trapping (`VITASLOP_WATCH_ |
| `VITASLOP_WATCH_FROM` | vitaslop-native/tests/retail_boot_probe.rs | - |
| `VITASLOP_WATCH_MEM` | vitaslop-native/tests/retail_boot_probe.rs | Parse `VITASLOP_WATCH_MEM=addr:type:label,addr:type:label,...` into watches. |
| `VITASLOP_WATCH_READ` | vitaslop-transpiler/src/emit.rs | Diagnostic read watchpoint. |
| `VITASLOP_WATCH_READ_` | vitaslop-transpiler/src/emit.rs | Optional guest-PC EXCLUDE window for the read watchpoint (`VITASLOP_WATCH_READ_ |
| `VITASLOP_WATCH_READ_NZ` | vitaslop-transpiler/src/emit.rs | Emit the read-watchpoint trap check. |
| `VITASLOP_WATCH_READ_PC_EXCL` | vitaslop-transpiler/src/emit.rs | Optional guest-PC EXCLUDE window for the read watchpoint (`VITASLOP_WATCH_READ_ |
| `VITASLOP_WATCH_READ_SKIP` | vitaslop-transpiler/src/emit.rs | WASM global index of the read-watchpoint match counter, appended after the guest-PC |
| `VITASLOP_WATCH_STORE` | vitaslop-native/src/threaded.rs | `VITASLOP_REGTRACE_WATCH=<hex guest addr>[,<hex guest addr>...]` - append the WORD |
| `VITASLOP_WATCH_STORE_ARM` | vitaslop-transpiler/src/emit.rs | The emit-time knobs [`set_emit_knob`] accepts, so a caller that forwards a whole table |
| `VITASLOP_WATCH_STORE_LOG` | vitaslop-runtime/src/capture.rs | GUEST ADDRESS the bytes above were read from, or 0 when there is no bound buffer. |
| `VITASLOP_WATCH_STORE_MODE` | vitaslop-transpiler/src/emit.rs | Store-watchpoint mode, from `VITASLOP_WATCH_STORE_MODE` (default `any`): |
| `VITASLOP_WATCH_STORE_NZ` | vitaslop-transpiler/src/emit.rs | `VITASLOP_WATCH_STORE_LOG` - LOG each store to the watched address (the storing |
| `VITASLOP_WATCH_STORE_SKIP` | vitaslop-transpiler/src/emit.rs | Linear-memory byte offset of the store-watchpoint MATCH COUNTER, or 0 when this |
| `VITASLOP_WATCH_UNIFORM_REWRITES` | vitaslop-runtime/src/host/resolve.rs | `VITASLOP_WATCH_UNIFORM_REWRITES=1`: watch every SA uniform buffer a draw read for a guest |
| `VITASLOP_WHICH_EXPORT` | vitaslop-runtime/src/link.rs | - |
| `VITASLOP_WINDOW_CENSUS` | vitaslop-runtime/src/host/snapshot_cache.rs | `VITASLOP_WINDOW_CENSUS=1`: measure the census's `slot-blank` count exactly even under the |
| `VITASLOP_WINDOW_FRAMES` | vitaslop-desktop/src/retail.rs | Run the retail title in `dir` in a live window until the window closes or the guest |
| `VITASLOP_WINDOW_REREAD` | vitaslop-runtime/src/host/snapshot_cache.rs | Which windows the guest's GPU wait re-reads - see [`window_wants_reread`]. |
| `VITASLOP_WINDOW_SHOT` | vitaslop-desktop/src/retail.rs | The frame rendered OFFSCREEN through this window's own renderer - its targets, caches and |
| `VITASLOP_WINDOW_SHOT_EVERY` | vitaslop-desktop/src/retail.rs | `VITASLOP_WINDOW_SHOT_EVERY` - see [`shot_every`]. |
| `VITASLOP_WINDOW_SHOT_FROM` | vitaslop-desktop/src/retail.rs | `VITASLOP_WINDOW_SHOT_FROM=<frame>`: the periodic shots start at this frame - a dense |
| `VITASLOP_WINDOW_SIZE` | vitaslop-desktop/src/retail.rs | `VITASLOP_WINDOW_SIZE=<w>x<h>`: open a window (the `--game` window or the library shell) at |
| `VITASLOP_WINDOW_WAIT` | vitaslop-runtime/src/host/snapshot_cache.rs | `VITASLOP_WINDOW_WAIT=1`: the draw-time window capture waits for a busy snapshot lock |
| `VITASLOP_WRITEBACK_AT_FLIP` | vitaslop-web/src/lib.rs | >>> RENDER-TARGET WRITE-BACKS GO INTO GUEST MEMORY AT THE FLIP, NOT AFTER THE PRESENT. |
| `VITASLOP_X` | vitaslop-frontend/src/settings.rs | Parse a `NAME=VALUE` per line knobs box into a map. |
| `VITASLOP_XML_DUMP` | vitaslop-runtime/src/vita/sce_xml.rs | `VITASLOP_XML_DUMP=<dir>`: write every document handed to `parse` into `<dir>` as |
| `VITASLOP_Y` | vitaslop-frontend/src/settings.rs | Parse a `NAME=VALUE` per line knobs box into a map. |
| `VITASLOP_YIELD_ELIDE` | vitaslop-runtime/src/host/state_clock.rs | >>> WHETHER A `sceKernelDelayThread(0)` HAS ANYONE TO GIVE THE CPU TO. |
| `VITASLOP_ZLS_PROBE` | vitaslop-platform/src/gpu/gxm/renderer_chain.rs | `VITASLOP_ZLS_PROBE=<hex depth address>[@<guest frame>]`: one line per pass into that depth |
