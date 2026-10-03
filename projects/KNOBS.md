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

550 knobs.

| knob | read in | what it does |
|---|---|---|
| `VITASLOP_A` | vitaslop-frontend/src/settings.rs:292 | - |
| `VITASLOP_AAC_DIR` | vitaslop-aac/tests/oracle.rs:69 | The `AudioSpecificConfig` an ADTS header describes: 5 bits of object type, 4 of |
| `VITASLOP_ALLOW_SOFTWARE_GPU` | vitaslop-web/src/lib.rs:2401 | Whether a run may proceed on a software rasteriser (`VITASLOP_ALLOW_SOFTWARE_GPU`). |
| `VITASLOP_AMBIENT_PROBE` | vitaslop-frontend/src/settings.rs:148 | The knobs every run starts from. |
| `VITASLOP_ARENA_TRIM` | vitaslop-platform/src/gpu.rs:1034 | The 2026-09-28 memory bounds, each with its arm back (`=0`): `VITASLOP_SUBRECT_POOL` |
| `VITASLOP_ARENA_UPLOAD_PROBE` | vitaslop-platform/src/gpu.rs:26142 | `VITASLOP_ARENA_UPLOAD_PROBE=1` - time the SAME bytes through the other upload path, a |
| `VITASLOP_ARM_AT_FRAME` | vitaslop-native/src/threaded.rs:342 | Linear-memory offset of the "diagnostics armed" word, when this build was |
| `VITASLOP_ARM_FUNC` | vitaslop-native/tests/homebrew_memcpy.rs:21 | Scratch window well past the code: `[SRC, SRC+WIN)` and `[DST, DST+WIN)`. |
| `VITASLOP_ASYNC_PIPELINES` | vitaslop-platform/src/gpu.rs:1327 | >>> PIPELINES ARE STARTED ASYNC, AHEAD OF THE FRAME THAT DRAWS THEM - see |
| `VITASLOP_AT9_DIR` | vitaslop-atrac9/tests/oracle.rs:66 | Decode a whole AT9 payload the way a superframe consumer does: for each |
| `VITASLOP_AUDIO_RAW` | vitaslop-runtime/src/vita/audio.rs:103 | Optional raw-s16le capture of the output, ONE FILE PER PORT (env `VITASLOP_AUDIO_RAW` |
| `VITASLOP_AUDIO_SCHEDULE` | vitaslop-runtime/src/vita/audio.rs:641 | Whether `sceAudioOutOutput` paces by the port's play-out schedule (the default) - see |
| `VITASLOP_AUDIO_WALL` | vitaslop-runtime/src/vita/audio.rs:625 | The host wall clock in microseconds when `sceAudioOutOutput` should pace on it (see |
| `VITASLOP_AVCDEC_CATCH_UP_MS` | vitaslop-web/src/smp.rs:2200 | `VITASLOP_AVCDEC_CATCH_UP_MS`: how long a forwarded movie decode may wait for its decoder |
| `VITASLOP_B` | vitaslop-frontend/src/settings.rs:292 | - |
| `VITASLOP_BACKTRACE` | vitaslop-runtime/src/vita/mod.rs:844 | Print the guest call chain the first time a chosen NID is called from each thread |
| `VITASLOP_BLESS_GXMCONF` | vitaslop-gxp-shader/tests/conformance.rs:3662 | The C header `gxmconf.c` includes: every container as a 64-byte-aligned byte array. |
| `VITASLOP_BLOCK_HIST` | vitaslop-native/src/recipe_runner.rs:167 | Dump the per-PC block-entry histogram gathered under `VITASLOP_BLOCK_HIST`, for |
| `VITASLOP_BLOCK_HIST_SEQ` | vitaslop-native/src/threaded.rs:1899 | Print the block-visit histogram gathered under `VITASLOP_BLOCK_HIST`: the `top` |
| `VITASLOP_BROWSER_EXIT_UNWIND` | vitaslop-web/src/browser_sched.rs:3537 | Whether a guest thread that EXITS through a host call (`sceKernelExitThread`, |
| `VITASLOP_BROWSER_FASTFORWARD` | vitaslop-runtime/src/vita/audio.rs:626 | The host wall clock in microseconds when `sceAudioOutOutput` should pace on it (see |
| `VITASLOP_BROWSER_FUEL` | vitaslop-web/src/browser_sched.rs:952 | Guest work a thread may execute before the browser preempts it, in WASMTIME FUEL UNITS |
| `VITASLOP_BROWSER_HEARTBEAT_MS` | vitaslop-web/src/lib.rs:5648 | - |
| `VITASLOP_BROWSER_INSTANCE_POOL` | vitaslop-web/src/browser_sched.rs:2139 | Whether a finished thread's module instance may be REUSED by the next thread |
| `VITASLOP_BROWSER_INSTANCE_RESERVE` | vitaslop-web/src/browser_sched.rs:2655 | Stand up one guest thread: an instance importing the shared memory and a |
| `VITASLOP_BROWSER_QUANTUM_CALLS` | vitaslop-web/src/browser_sched.rs:130 | Host calls one guest thread may make before the browser preempts it |
| `VITASLOP_BROWSER_RENDER_FROM` | vitaslop-web/src/lib.rs:2384 | Frame from which a FAST-FORWARD still RENDERS (`VITASLOP_BROWSER_RENDER_FROM`), even though |
| `VITASLOP_BROWSER_SPLIT_MEMORY` | vitaslop-web/src/browser_sched.rs:1237 | >>> THE GUEST REGION INSIDE THIS MODULE'S OWN LINEAR MEMORY. |
| `VITASLOP_BROWSER_SUPERSAMPLE` | vitaslop-web/src/lib.rs:2157 | Supersample factor for the live browser render (`VITASLOP_BROWSER_SUPERSAMPLE`). |
| `VITASLOP_BROWSER_UNPACED` | vitaslop-web/src/lib.rs:5603 | - |
| `VITASLOP_BUILD_FASTPATH` | vitaslop-runtime/src/render.rs:6472 | `VITASLOP_BUILD_FASTPATH=0`: the NEGATIVE CONTROL arm for what `build` stopped doing. |
| `VITASLOP_CALLSITES_WINDOW` | vitaslop-desktop/src/retail.rs:600 | Where the idle clock went SINCE `before` - the windowed reading. |
| `VITASLOP_CALL_TABLE` | vitaslop-runtime/src/call_table.rs:50 | The frame window, or `None` when the table is off. |
| `VITASLOP_CALL_TABLE_REGS` | vitaslop-runtime/src/call_table.rs:31 | r0..r12 at the FIRST occurrence, printed under `VITASLOP_CALL_TABLE_REGS=1` - the |
| `VITASLOP_CAPSULE_DUMP_PROGS` | vitaslop-native/examples/capsule-replay.rs:579 | - |
| `VITASLOP_CAPSULE_DUMP_SA` | vitaslop-native/examples/capsule-replay.rs:29 | `VITASLOP_CAPSULE_DUMP_SA=1`: print this draw's uniform banks - `frag_sa` with the GUEST |
| `VITASLOP_CAPSULE_DUMP_VERTS` | vitaslop-native/examples/capsule-replay.rs:488 | - |
| `VITASLOP_CAPSULE_EXTENT` | vitaslop-native/examples/capsule-replay.rs:684 | - |
| `VITASLOP_CAPSULE_TEX_DIR` | vitaslop-native/examples/capsule-replay.rs:386 | - |
| `VITASLOP_CARRY_UNPRESENTED` | vitaslop-web/src/lib.rs:2236 | `VITASLOP_CARRY_UNPRESENTED=0`: the arm back - drop an unpresented frame's scenes whole. |
| `VITASLOP_CHAIN_DRAWS` | vitaslop-platform/src/gpu.rs:24637 | - |
| `VITASLOP_CHAIN_LIMIT` | vitaslop-native/tests/gpu_rtt_gamma.rs:179 | Render a chain of `feedback` sample-and-write-back passes over the offscreen target and |
| `VITASLOP_CHAIN_SKIP` | vitaslop-native/src/wgpu_render.rs:374 | - |
| `VITASLOP_CHECK_ADDRS` | vitaslop-native/tests/retail_boot_probe.rs:43 | - |
| `VITASLOP_CLOCK_TRACE` | vitaslop-runtime/src/sched.rs:1175 | - |
| `VITASLOP_CLOCK_WALL_FLOOR` | vitaslop-web/src/smp.rs:109 | The game clock is floored at the wall - see `VitaState::wall_floor_tick`. |
| `VITASLOP_CLOCK_WALL_STEP_MS` | vitaslop-runtime/src/host.rs:12181 | >>> THE GAME CLOCK MAY NOT RUN SLOWER THAN REAL TIME (SMP browser runs). |
| `VITASLOP_CODE_RANGE` | vitaslop-runtime/src/vita/mod.rs:827 | The guest code range scanned for the game-level caller in [`dispatch`] (env |
| `VITASLOP_COMPACT_SPARSE` | vitaslop-runtime/src/host.rs:3622 | `VITASLOP_COMPACT_SPARSE=0` is the NEGATIVE CONTROL for the compaction: the `min..=max` |
| `VITASLOP_COMPILE_CACHE` | vitaslop-desktop/src/retail.rs:380 | - |
| `VITASLOP_CONSOLE` | vitaslop-web/src/logging.rs:216 | `VITASLOP_CONSOLE=1`: mirror the run's status notes - the setup summary, the adapter and |
| `VITASLOP_CPU_SHARE` | vitaslop-native/src/recipe_runner.rs:111 | Who actually got the CPU over the run, when `VITASLOP_CPU_SHARE` is set - see |
| `VITASLOP_CPU_SHARE_FROM` | vitaslop-runtime/src/sched.rs:1578 | `VITASLOP_CPU_SHARE_FROM=<frame>` - see the reset in `SchedCore::on_suspended`. |
| `VITASLOP_CTRL_READ_NEW` | vitaslop-runtime/src/vita/ctrl.rs:340 | A blocking read returns `count` samples (the default) unless `VITASLOP_CTRL_READ_NEW=1`. |
| `VITASLOP_DBG_CALLSITES` | vitaslop-runtime/src/vita/mod.rs:785 | Diagnostic call-site profiler (`VITASLOP_DBG_CALLSITES`): counts host calls |
| `VITASLOP_DEBUG_CAPTURE` | vitaslop-frontend/src/settings.rs:196 | The `VITASLOP_*` map a run of these settings is configured with: the base set, |
| `VITASLOP_DECODE_CACHE_MB` | vitaslop-runtime/src/render.rs:7033 | Budget for the decode cache, in BYTES of decoded RGBA8, before it is cleared wholesale. |
| `VITASLOP_DEFER_GEOMETRY` | vitaslop-runtime/src/host.rs:24965 | Whether a draw's VERTEX AND INDEX BYTES are read at `sceGxmEndScene` rather than at the |
| `VITASLOP_DEFER_WINDOW_BYTES` | vitaslop-runtime/src/host.rs:3702 | `VITASLOP_GXP_ATTR_ROW=0` - the negative control for laying the interleaved row out per |
| `VITASLOP_DELAY_CENSUS` | vitaslop-runtime/src/vita/threadmgr.rs:240 | Diagnostic (`VITASLOP_DELAY_CENSUS=1`): every `sceKernelDelayThread` tallied by (call site, |
| `VITASLOP_DEVICE_BUDGET` | vitaslop-desktop/src/retail.rs:1076 | Headless self-check of the retail path (NO window): load `dir`, optionally drive a |
| `VITASLOP_DIALOG_RUNNING_FRAMES` | vitaslop-runtime/src/vita/services.rs:277 | How many display flips an opened dialog reports RUNNING for before FINISHED |
| `VITASLOP_DIRTY_MARK_TEST` | vitaslop-transpiler/src/emit.rs:1140 | >>> UNDER SMP THE MARK IS WRITTEN ONLY WHEN THE BYTE DIFFERS FROM THE EPOCH. |
| `VITASLOP_DIRTY_PAGES` | vitaslop-native/src/threaded.rs:128 | Linear-memory offset of the guest-store dirty block, when this build was |
| `VITASLOP_DIRTY_RUN_MARK` | vitaslop-transpiler/src/emit.rs:872 | Turn the coalesced run-mark on or off for modules emitted on this thread after this |
| `VITASLOP_DISASM_FILE` | vitaslop-transpiler/tests/disasm_dump.rs:3 | - |
| `VITASLOP_DISASM_RANGE` | vitaslop-transpiler/tests/disasm_dump.rs:3 | - |
| `VITASLOP_DISPATCH_ALL` | vitaslop-transpiler/src/emit.rs:1812 | The ablation arm that prices a dispatch re-entry: `VITASLOP_DISPATCH_ALL=1` sends even a |
| `VITASLOP_DRAW_NOTE_AT` | vitaslop-web/src/lib.rs:3128 | Render one freshly-executed FRAME - every scene the guest submitted between |
| `VITASLOP_DRAW_NOTE_TARGET` | vitaslop-web/src/lib.rs:3147 | - |
| `VITASLOP_DRAW_NOTE_TEX` | vitaslop-web/src/lib.rs:3200 | - |
| `VITASLOP_DRAW_ONLY` | vitaslop-runtime/src/render.rs:5544 | - |
| `VITASLOP_DRAW_RANGE` | vitaslop-platform/src/gpu.rs:801 | `VITASLOP_RTT_BG_CACHE=0` restores the OLD behaviour: a sampler bind group naming a render |
| `VITASLOP_DRAW_STATS` | vitaslop-runtime/src/render.rs:10596 | `VITASLOP_DRAW_STATS` (diagnostic), cached: asked per scene and per draw batch. |
| `VITASLOP_DRV_KEY` | vitaslop-runtime/src/ingest/pfscrypt.rs:344 | `F00D(klicensee)` for the title, from `VITASLOP_DRV_KEY` (32 hex chars), or |
| `VITASLOP_DUMP_ADDR` | vitaslop-native/tests/dump_segment.rs:3 | - |
| `VITASLOP_DUMP_DIR` | vitaslop-runtime/src/ingest/pipeline.rs:566 | Diagnostic: write the eboot's decrypted inner ELF to `VITASLOP_DUMP_DIR/eboot.elf`, so |
| `VITASLOP_DUMP_DRAW` | vitaslop-native/tests/retail_boot_probe.rs:1287 | - |
| `VITASLOP_DUMP_DRAWS` | vitaslop-native/tests/retail_boot_probe.rs:1076 | - |
| `VITASLOP_DUMP_DRAW_GXP` | vitaslop-gxp-shader/tests/oracle.rs:653 | Correlate each captured vertex<->fragment PAIR (from a real draw run) to establish the |
| `VITASLOP_DUMP_DRAW_GXP_CAP` | vitaslop-runtime/src/host.rs:19507 | Whether `VITASLOP_DUMP_DRAW_GXP` names display frame `disp`: "all", a single frame |
| `VITASLOP_DUMP_DRAW_GXP_FULL` | vitaslop-runtime/src/host.rs:19797 | - |
| `VITASLOP_DUMP_EXPORTS` | vitaslop-runtime/src/link.rs:651 | - |
| `VITASLOP_DUMP_FILES` | vitaslop-runtime/src/ingest/pipeline.rs:921 | Diagnostic: decrypt the container and write named plaintext files out to |
| `VITASLOP_DUMP_FPROG` | vitaslop-runtime/src/host.rs:19053 | Diagnostic (VITASLOP_DUMP_FPROG): print the bound fragment program's sampler |
| `VITASLOP_DUMP_FUNC` | vitaslop-native/tests/retail_boot_probe.rs:317 | - |
| `VITASLOP_DUMP_GXP_BIN` | vitaslop-platform/src/gpu.rs:17544 | >>> A REFUSAL THAT DOES NOT HAND OVER THE EVIDENCE COSTS A PLAY SESSION. |
| `VITASLOP_DUMP_IMAGE` | vitaslop-native/tests/retail_boot_probe.rs:33 | - |
| `VITASLOP_DUMP_IMPORTS` | vitaslop-native/tests/retail_boot_probe.rs:353 | - |
| `VITASLOP_DUMP_IR` | vitaslop-native/tests/homebrew_memcpy.rs:40 | Scratch window well past the code: `[SRC, SRC+WIN)` and `[DST, DST+WIN)`. |
| `VITASLOP_DUMP_MAP` | vitaslop-native/tests/retail_boot_probe.rs:590 | - |
| `VITASLOP_DUMP_MEM` | vitaslop-native/tests/retail_boot_probe.rs:43 | - |
| `VITASLOP_DUMP_OUT` | vitaslop-native/tests/dump_segment.rs:3 | - |
| `VITASLOP_DUMP_PATHS` | vitaslop-native/tests/retail_boot_probe.rs:413 | - |
| `VITASLOP_DUMP_REGION` | vitaslop-native/tests/retail_boot_probe.rs:528 | - |
| `VITASLOP_DUMP_REGION_RANGE` | vitaslop-native/tests/retail_boot_probe.rs:530 | - |
| `VITASLOP_DUMP_RENDERSCENE` | vitaslop-native/tests/retail_boot_probe.rs:1328 | - |
| `VITASLOP_DUMP_SCENES` | vitaslop-desktop/src/retail.rs:469 | Step the guest one display frame. |
| `VITASLOP_DUMP_STDOUT` | vitaslop-desktop/src/retail.rs:2097 | - |
| `VITASLOP_DUMP_STREAM_BYTES` | vitaslop-runtime/src/host.rs:24289 | `VITASLOP_DUMP_STREAM_BYTES[=<n>]`: with the per-draw dump on, print each vertex stream's |
| `VITASLOP_DUMP_STUBS` | vitaslop-native/tests/retail_boot_probe.rs:32 | - |
| `VITASLOP_DUMP_TEX` | vitaslop-native/tests/retail_boot_probe.rs:1230 | - |
| `VITASLOP_DUMP_TEX_DIR` | vitaslop-runtime/src/host.rs:19803 | - |
| `VITASLOP_DUMP_TEX_MAX_TEXELS` | vitaslop-runtime/src/host.rs:19812 | - |
| `VITASLOP_DUMP_TRIS` | vitaslop-runtime/src/render.rs:5551 | - |
| `VITASLOP_DUMP_VPROG` | vitaslop-runtime/src/host.rs:18811 | Diagnostic (VITASLOP_DUMP_VPROG): reflect the bound vertex program's parameter |
| `VITASLOP_DUMP_VUBUF` | vitaslop-runtime/src/host.rs:19613 | - |
| `VITASLOP_EAGER_FILES` | vitaslop-desktop/src/retail.rs:257 | [`Self::new`] with the app's executable `main_exec` (an `sceAppMgrLoadExec` path) in |
| `VITASLOP_EARLY_GRACE_MS` | vitaslop-web/src/lib.rs:2211 | >>> SERVE A PARKED GPU WAIT BEFORE THE PRESENT, NOT AFTER IT. |
| `VITASLOP_EARLY_WAIT_ANY` | vitaslop-web/src/lib.rs:2194 | `VITASLOP_EARLY_WAIT_ANY=1`: an early completion waits for EVERY in-flight readback of its |
| `VITASLOP_FAST_IMPORT_CURATED` | vitaslop-runtime/src/vita/mod.rs:381 | Whether `func_nid`'s handler can only ever CONTINUE, so the transpiler may route the |
| `VITASLOP_FIBER_RUNNER_PRIORITY` | vitaslop-runtime/src/host.rs:1265 | Whether a fiber's backing thread runs at its runner's priority (the hardware: a fiber IS its |
| `VITASLOP_FIND_WORD` | vitaslop-native/tests/retail_boot_probe.rs:63 | The span `VITASLOP_FIND_WORD` searches: from the image base up through the guest heap. |
| `VITASLOP_FLAGS_WIDE_C` | vitaslop-transpiler/src/emit.rs:1785 | The A/B arm for [`emit_flags_add`]'s carry and overflow forms: `VITASLOP_FLAGS_WIDE_C=1` |
| `VITASLOP_FLAG_POISON` | vitaslop-transpiler/src/emit.rs:1743 | `VITASLOP_FLAG_POISON=0/1` - the FALSIFIER for the flag-liveness pass |
| `VITASLOP_FORCE_READY` | vitaslop-native/tests/retail_boot_probe.rs:736 | - |
| `VITASLOP_FORCE_READY_V2` | vitaslop-native/tests/retail_boot_probe.rs:759 | - |
| `VITASLOP_FORCE_RET` | vitaslop-transpiler/src/emit.rs:2128 | Diagnostic forced return. |
| `VITASLOP_FRAME_CAPSULE` | vitaslop-runtime/src/capsule.rs:753 | `VITASLOP_FRAME_CAPSULE=<dir>`: write every frame a headless run renders for a SHOT to |
| `VITASLOP_FRAME_DIGEST` | vitaslop-native/src/recipe_runner.rs:375 | - |
| `VITASLOP_FRAME_TOPUP` | vitaslop-runtime/src/host.rs:8274 | The per-flip top-up ([`VitaState::advance_time_frame`]), which is OPT-IN: |
| `VITASLOP_FUEL` | vitaslop-native/src/threaded.rs:147 | This thread's SOFTWARE fuel counter (`abi::FUEL_EXPORT`), present only when the |
| `VITASLOP_GAME_DIR` | vitaslop-runtime/src/ingest/mod.rs:140 | Test-fixture access. |
| `VITASLOP_GAME_ID` | vitaslop-gamerun-recipes/tests/conformance.rs:30 | - |
| `VITASLOP_GAME_PKG` | vitaslop-runtime/src/ingest/pipeline.rs:568 | Diagnostic: write the eboot's decrypted inner ELF to `VITASLOP_DUMP_DIR/eboot.elf`, so |
| `VITASLOP_GAME_WORK` | vitaslop-runtime/src/ingest/pipeline.rs:568 | Diagnostic: write the eboot's decrypted inner ELF to `VITASLOP_DUMP_DIR/eboot.elf`, so |
| `VITASLOP_GAME_ZIP` | vitaslop-runtime/src/ingest/mod.rs:120 | - |
| `VITASLOP_GAP_CAP` | vitaslop-native/tests/retail_boot_probe.rs:266 | - |
| `VITASLOP_GESTURE_EVENT_KIND` | vitaslop-runtime/src/vita/gesture.rs:814 | `VITASLOP_GESTURE_EVENT_KIND`: write this byte at [`EVENT_KIND_OFF`]. |
| `VITASLOP_GESTURE_EVENT_STATE` | vitaslop-runtime/src/vita/gesture.rs:395 | The bits written into [`EVENT_STATE_OFF`]. |
| `VITASLOP_GESTURE_PRIMITIVE_STATE` | vitaslop-runtime/src/vita/gesture.rs:796 | `VITASLOP_GESTURE_PRIMITIVE_STATE`: write this halfword at [`PRIMITIVE_STATE_OFF`]. |
| `VITASLOP_GESTURE_TAP_ON_RELEASE` | vitaslop-runtime/src/vita/gesture.rs:862 | `VITASLOP_GESTURE_TAP_ON_RELEASE`: report a type-1 recognizer's event on the frame the |
| `VITASLOP_GESTURE_TYPE_MASK` | vitaslop-runtime/src/vita/gesture.rs:414 | Recognizer types allowed to report events (`VITASLOP_GESTURE_TYPE_MASK`, a bitmask |
| `VITASLOP_GPU` | vitaslop-native/tests/retail_boot_probe.rs:1391 | - |
| `VITASLOP_GPU_BUDGET` | vitaslop-web/src/lib.rs:1183 | >>> THE GPU BUDGET: a present costs the GPU milliseconds the newest timestamp query |
| `VITASLOP_GPU_BURN` | vitaslop-platform/src/gpu.rs:6051 | `VITASLOP_GPU_BURN`'s compute work, built on first use - see [`Self::gpu_burn`]. |
| `VITASLOP_GPU_CHAIN_DIR` | vitaslop-native/src/wgpu_render.rs:611 | `VITASLOP_GPU_CHAIN_DIR=<dir>`: write every offscreen target of the frame just |
| `VITASLOP_GPU_QUEUE_DEPTH` | vitaslop-web/src/lib.rs:1081 | How many submits may be in flight before a present declines to make another. |
| `VITASLOP_GPU_STALE_OLD` | vitaslop-web/src/lib.rs:1290 | `VITASLOP_GPU_STALE_OLD=1`: the stale rule as it was - see the check in `present`. |
| `VITASLOP_GPU_TIME_ALL` | vitaslop-platform/src/gpu.rs:853 | `VITASLOP_GXP_PASS_SPLIT_EVERY=<n>` cuts a render pass every `n` draws, with no shader |
| `VITASLOP_GUARD_REG` | vitaslop-transpiler/src/emit.rs:1982 | Diagnostic callee-saved-register guard. |
| `VITASLOP_GUEST_CORES` | vitaslop-runtime/src/host.rs:22397 | CPU cores a Vita gives a GAME. |
| `VITASLOP_GUEST_PROF` | vitaslop-transpiler/src/abi.rs:293 | SMP, `VITASLOP_GUEST_PROF` only: worker `w`'s GUEST-FUNCTION slot is host-mirror slot |
| `VITASLOP_GXM` | vitaslop-native/examples/capsule-replay.rs:598 | - |
| `VITASLOP_GXM_ALPHA_SINGLE` | vitaslop-conformance-harness/tests/vita_gxmconf_real.rs:116 | **SCENE 10 - A U8_A SURFACE STORES THE FRAGMENT'S ALPHA.** |
| `VITASLOP_GXM_ARENA_FLOOR_KB` | vitaslop-platform/src/gpu.rs:898 | `VITASLOP_GXM_DEST_SPLIT_AB=<n>`: alternate the destination-colour pass SPLIT on and off every |
| `VITASLOP_GXM_ARENA_POOL` | vitaslop-platform/src/gpu.rs:14864 | `VITASLOP_GXM_ARENA_POOL=1` pools the six per-pass staging arenas. |
| `VITASLOP_GXM_ARENA_REPEAT` | vitaslop-platform/src/gpu.rs:4387 | >>> REPACKS OF GEOMETRY THIS RUN HAD ALREADY EVICTED - the cache THRASHING. |
| `VITASLOP_GXM_BACKGROUND_DEPTH` | vitaslop-conformance-harness/tests/vita_gxmconf.rs:621 | **SCENE 8 - A REVERSED DEPTH RANGE: THE PASS STARTS FROM THE SURFACE'S BACKGROUND DEPTH.** |
| `VITASLOP_GXM_BUFFER_PREINIT` | vitaslop-platform/src/gpu.rs:19633 | Create through the pool: a hit is free, a miss is the allocation that was going to |
| `VITASLOP_GXM_COLOR_MASK_ORDER` | vitaslop-conformance-harness/tests/vita_gxmconf_real.rs:179 | **SCENE 14 - THE COLOUR MASK NUMBERS ALPHA AS BIT 0.** |
| `VITASLOP_GXM_DEPTH_ENC` | vitaslop-platform/src/gpu.rs:3750 | Which value a later pass reads out of a render target's depth |
| `VITASLOP_GXM_DEST_SPLIT_AB` | vitaslop-platform/src/gpu.rs:873 | `VITASLOP_GXM_DEST_SPLIT_AB=<n>`: alternate the destination-colour pass SPLIT on and off every |
| `VITASLOP_GXM_DRAW_COVERAGE` | vitaslop-gxp-shader/src/link.rs:5504 | `1` - WHERE DID THIS DRAW'S VERTICES GO? Draw the mesh at CLAMPED normalised coordinates and |
| `VITASLOP_GXM_DRAW_PROBE` | vitaslop-platform/src/gpu.rs:3812 | >>> TEMPORARY TELEMETRY (`VITASLOP_GXM_DRAW_PROBE=<keyspec>`). |
| `VITASLOP_GXM_FLOAT_TARGETS` | vitaslop-platform/src/gpu.rs:2825 | Whether a guest `SceGxmColorFormat` is F11F11F10, the packed FLOATING-POINT format a title |
| `VITASLOP_GXM_FLOAT_TARGETS_WIDE` | vitaslop-platform/src/gpu.rs:2844 | Whether a guest `SceGxmColorFormat` is F11F11F10, the packed FLOATING-POINT format a title |
| `VITASLOP_GXM_HOLD_FLIP` | vitaslop-platform/src/gpu.rs:1025 | Whether a frame whose last scene is not a flipped (display) buffer shows its last display |
| `VITASLOP_GXM_NO_MULTISAMPLE` | vitaslop-platform/src/gpu.rs:8686 | A/B instrument: force every pass to ONE sample, whatever the guest asked for. |
| `VITASLOP_GXM_PVS_BULK_TABLE` | vitaslop-runtime/src/vita/gxm.rs:749 | `VITASLOP_GXM_PVS_BULK_TABLE=1` - restore the VERTEX bind's old WHOLESALE table copy, zeros |
| `VITASLOP_GXM_RTT_CLEAR_EVERY_FRAME` | vitaslop-platform/src/gpu.rs:26114 | `VITASLOP_GXM_RTT_CLEAR_EVERY_FRAME=1` - restore the pre-2026-09-12 behaviour, in which the |
| `VITASLOP_GXM_RTT_WRITEBACK` | vitaslop-native/src/wgpu_render.rs:511 | The rendered pixels of every offscreen target small enough to hand back to the GUEST, |
| `VITASLOP_GXM_SHARE_PROGRAMS` | vitaslop-runtime/src/vita/gxm.rs:1660 | Report - once per program - that a blend came from the SHADER rather than from GXM. |
| `VITASLOP_GXM_STAGING` | vitaslop-platform/src/gpu.rs:1049 | Whether the arena upload goes through the STAGING BELT (default) rather than |
| `VITASLOP_GXM_STALE_UNIFORMS` | vitaslop-desktop/src/retail.rs:1090 | Headless self-check of the retail path (NO window): load `dir`, optionally drive a |
| `VITASLOP_GXM_STATE_DEDUP` | vitaslop-platform/src/gpu.rs:8602 | What a render pass already has bound, so a draw that changes nothing rebinds nothing. |
| `VITASLOP_GXM_SWEEP_EVERY` | vitaslop-platform/src/gpu.rs:26165 | `VITASLOP_GXM_SWEEP_EVERY=<n>` - how many frames apart the packed-geometry budget check and |
| `VITASLOP_GXM_TEX_UNWRITTEN` | vitaslop-runtime/src/render.rs:6995 | `VITASLOP_GXM_TEX_UNWRITTEN=0` - the arm back for reading ANY uniform 4-byte fill as |
| `VITASLOP_GXM_UNIFORM_POISON` | vitaslop-gxp-shader/src/module.rs:392 | `:bits=<hex>` - paint a lane 1.0 when that register's RAW BITS equal this word, 0.0 |
| `VITASLOP_GXM_VIEWPORT` | vitaslop-platform/src/gpu.rs:2390 | `VITASLOP_GXM_VIEWPORT=0` - the NEGATIVE CONTROL for the guest viewport reaching the |
| `VITASLOP_GXM_ZLS` | vitaslop-platform/src/gpu.rs:2926 | Whether a sampler reading one 32-bit word of a raw 64-bit surface as a double-width texture |
| `VITASLOP_GXP` | vitaslop-native/examples/capsule-replay.rs:598 | - |
| `VITASLOP_GXP_` | vitaslop-native/examples/capsule-replay.rs:14 | - |
| `VITASLOP_GXP_ALIGN_ROW` | vitaslop-runtime/src/host.rs:2774 | ONE COPY the gather makes: `len` bytes from byte `src` of stream `stream`'s row into byte |
| `VITASLOP_GXP_ALLOW_FIXED_FUNCTION` | vitaslop-platform/src/gpu.rs:17611 | >>> A PAIR THIS RECOMPILER CANNOT TRANSLATE DROPS ITS DRAWS. |
| `VITASLOP_GXP_ARENA_RING` | vitaslop-platform/src/gpu.rs:1060 | `VITASLOP_GXP_ARENA_RING=<n>`: how many COPIES of each render pass's recompiled-path |
| `VITASLOP_GXP_ATTR_ALL` | vitaslop-gxp-shader/tests/corpus.rs:5053 | What [`vitaslop_gxp_shader::attrflow`] decides for every attribute of every pair that links, |
| `VITASLOP_GXP_ATTR_FILL` | vitaslop-gxp-shader/src/module.rs:2348 | Bytes the driver adds to the DEFAULT uniform buffer's bound address before writing it into |
| `VITASLOP_GXP_ATTR_ROW` | vitaslop-runtime/src/host.rs:3700 | `VITASLOP_GXP_ATTR_ROW=0` - the negative control for laying the interleaved row out per |
| `VITASLOP_GXP_BIND_TRACE` | vitaslop-platform/src/gpu.rs:813 | Diagnostic (`VITASLOP_GXP_BIND_TRACE=1`): say, once per pair and plan shape, WHICH view each |
| `VITASLOP_GXP_BLOB` | vitaslop-gxp-shader/tests/corpus.rs:689 | Print one named blob's recompiled WGSL body and its container reflection. |
| `VITASLOP_GXP_BOUND_READS` | vitaslop-gxp-shader/src/link.rs:5402 | `0` keeps a bank read dynamically at the full [`BANK_REGS`] - the A/B arm for |
| `VITASLOP_GXP_CAPSULE` | vitaslop-runtime/src/capsule.rs:951 | Diagnostic (`VITASLOP_GXP_CAPSULE=<vprog-hash>[,<vprog-hash>]:<dir>[:N]`): write the first |
| `VITASLOP_GXP_CAPSULE_MIN_INDICES` | vitaslop-runtime/src/capsule.rs:964 | Diagnostic (`VITASLOP_GXP_CAPSULE=<vprog-hash>[,<vprog-hash>]:<dir>[:N]`): write the first |
| `VITASLOP_GXP_CAPSULE_SKIP` | vitaslop-runtime/src/capsule.rs:984 | Diagnostic (`VITASLOP_GXP_CAPSULE_SKIP=<n>`): ignore the first `n` matching submissions |
| `VITASLOP_GXP_CASES_OUT` | vitaslop-gxp-shader/tests/conformance.rs:3341 | Write every case out for the GPU runner, with the INTENT as the expectation. |
| `VITASLOP_GXP_CASE_TEX` | vitaslop-gxp-shader/src/wgsl.rs:2259 | >>> DOES THE STAND-IN TEXTURE VARY WITH THE COORDINATE? |
| `VITASLOP_GXP_CLAIM_WIDTH` | vitaslop-gxp-shader/src/link.rs:1892 | Re-order a convention-placed layout so that every forwarded attribute's usage STARTS at the |
| `VITASLOP_GXP_CLIP_DUMP_SA` | vitaslop-platform/src/gpu.rs:15949 | - |
| `VITASLOP_GXP_CLIP_RETRY_EVERY_FRAME` | vitaslop-platform/src/gpu.rs:26180 | `VITASLOP_GXP_CLIP_RETRY_EVERY_FRAME=1` - re-measure a pair whose clip measurement found no |
| `VITASLOP_GXP_CMOVU8` | vitaslop-gxp-shader/src/module.rs:676 | Whether [`crate::wgsl`]'s byte-wise conditional move tests EACH BYTE of its test operand |
| `VITASLOP_GXP_CORPUS` | vitaslop-gxp-shader/src/link.rs:6348 | Every vertex program in the captured corpora (`VITASLOP_GXP_CORPUS`, `;`-separated |
| `VITASLOP_GXP_COVERAGE_LOG` | vitaslop-gxp-shader/tests/corpus.rs:5859 | Price the F16 emulation in CONVERSIONS PER FRAME, by weighing each pair's emitted count with |
| `VITASLOP_GXP_CULL` | vitaslop-platform/src/gpu.rs:1294 | `VITASLOP_GXP_CULL=0` restores the pre-2026-08-19b "draw both windings". |
| `VITASLOP_GXP_DEBUG` | vitaslop-platform/src/gpu.rs:18438 | - |
| `VITASLOP_GXP_DEFAULT_UNIFORM_OFFSET` | vitaslop-gxp-shader/src/module.rs:2345 | Bytes the driver adds to the DEFAULT uniform buffer's bound address before writing it into |
| `VITASLOP_GXP_DEPTH24` | vitaslop-platform/src/gpu.rs:458 | - |
| `VITASLOP_GXP_DEPTH_PROBE` | vitaslop-gxp-shader/src/link.rs:4050 | Turn each stage's SA-bank marker into the reads the body actually needs. |
| `VITASLOP_GXP_DERIVED_CLAIMS` | vitaslop-gxp-shader/src/link.rs:1603 | Is the DERIVATION reading in force (`VITASLOP_GXP_DERIVED_CLAIMS=0` turns it off)? An A/B |
| `VITASLOP_GXP_DEST` | vitaslop-gxp-shader/src/module.rs:757 | Whether a fragment program that reads the destination colour is DECLARED as reading it. |
| `VITASLOP_GXP_DEST_BLEND` | vitaslop-gxp-shader/src/module.rs:625 | Whether [`lower_dest_blend`] is allowed to run. |
| `VITASLOP_GXP_DEST_POISON` | vitaslop-gxp-shader/src/module.rs:1754 | `VITASLOP_GXP_DEST_POISON=<r,g,b,a>` - the constant [`dest_color_init`] seeds the output |
| `VITASLOP_GXP_DEST_PROBE` | vitaslop-gxp-shader/src/link.rs:4050 | Turn each stage's SA-bank marker into the reads the body actually needs. |
| `VITASLOP_GXP_DISASM` | vitaslop-gxp-shader/tests/oracle.rs:860 | Compact disassembly of one blob (named by `VITASLOP_GXP_DISASM`, matched as a filename |
| `VITASLOP_GXP_DP_B48_BOTH` | vitaslop-gxp-shader/src/link.rs:5578 | `0` holds a repeating DP's internal op2 when bit 47 is clear and bit 48 is set (the old |
| `VITASLOP_GXP_DP_MOE_BIT47` | vitaslop-gxp-shader/src/link.rs:5575 | `0` sends every repeating DP back to the intrinsic four-register source walk, bit 47 or not |
| `VITASLOP_GXP_DUAL_SOURCE` | vitaslop-gxp-shader/src/module.rs:774 | Whether a destination-reading program that is LINEAR in the destination may be lowered to a |
| `VITASLOP_GXP_DUAL_TRACE` | vitaslop-gxp-shader/tests/corpus.rs:6480 | Print the DUAL-SOURCE plan (or the reason there is none) for every destination reader in |
| `VITASLOP_GXP_DUMP` | vitaslop-platform/src/gpu.rs:9230 | Diagnostic (`VITASLOP_GXP_KEYS=<hex>,<hex>`): recompile ONLY these shader-pair keys |
| `VITASLOP_GXP_DUMPS` | vitaslop-gxp-shader/tests/oracle.rs:145 | Histogram the raw values of named fields across every instruction of a given opcode1 |
| `VITASLOP_GXP_ELIDE_NOOP` | vitaslop-platform/src/gpu.rs:823 | `VITASLOP_GXP_ELIDE_NOOP=0` - the negative control for dropping the colour work of a draw |
| `VITASLOP_GXP_EXCLUDE` | vitaslop-platform/src/gpu.rs:9234 | Pairs forced down the fixed-function path (`VITASLOP_GXP_EXCLUDE`). |
| `VITASLOP_GXP_F16_FCMP` | vitaslop-gxp-shader/src/link.rs:5446 | `1`/`0`: the f16 SATURATION as a finite test and a `clamp` (`1`) or as a bit-pattern test |
| `VITASLOP_GXP_F16_RTE` | vitaslop-gxp-shader/src/link.rs:5432 | >>> HOW AN f32 NARROWS TO AN f16 - THE NEGATIVE CONTROL FOR THE ROUNDING FIX. |
| `VITASLOP_GXP_FMEM` | vitaslop-platform/src/gpu.rs:13150 | Diagnostic (`VITASLOP_GXP_FMEM=<lane>=<value>` / `<lane>*<factor>`, comma separated): |
| `VITASLOP_GXP_FOLD_LITERALS` | vitaslop-gxp-shader/src/link.rs:5705 | >>> A WORD KNOWN AT TRANSLATION TIME IS UNPACKED AT TRANSLATION TIME. |
| `VITASLOP_GXP_FORCE` | vitaslop-platform/src/gpu.rs:9201 | Diagnostic (`VITASLOP_GXP_FORCE`): bind a neutral fallback texture for a sampler |
| `VITASLOP_GXP_FRAG` | vitaslop-gxp-shader/tests/corpus.rs:5105 | The complete LINKED WGSL for one pair, selected by `VITASLOP_GXP_VERT` + `VITASLOP_GXP_FRAG` |
| `VITASLOP_GXP_GROUP` | vitaslop-gxp-shader/tests/corpus.rs:3409 | Every distinct word of one opcode group across the corpus, with the programs it appears in. |
| `VITASLOP_GXP_HALF_REGS` | vitaslop-gxp-shader/src/link.rs:4178 | Give every register the program only ever uses as a PACKED F16 PAIR an UNPACKED home, so |
| `VITASLOP_GXP_IDX_MUL` | vitaslop-gxp-shader/src/usse/mod.rs:724 | Fill in each ORDINARY-REGISTER index load's `stride` - how far apart two consecutive index |
| `VITASLOP_GXP_IDX_REGDEST` | vitaslop-gxp-shader/src/link.rs:5554 | `0` sends EVERY group-0x14 index load to the index register, which is what this decoder did |
| `VITASLOP_GXP_IDX_REPEAT` | vitaslop-gxp-shader/src/usse/decode.rs:4959 | Whether a load-index word's bits 46:44 repeat it (`VITASLOP_GXP_IDX_REPEAT=0` is the arm |
| `VITASLOP_GXP_IDX_SCALE` | vitaslop-gxp-shader/src/module.rs:715 | How many REGISTERS one count of an index register spans - see [`crate::wgsl`]'s |
| `VITASLOP_GXP_IEEE_RCP` | vitaslop-gxp-shader/src/wgsl.rs:1021 | >>> `rcp` AND `rsq` WITH THEIR ZEROES AND INFINITIES DECIDED IN INTEGERS. |
| `VITASLOP_GXP_IMAD_SRC1_WHOLE` | vitaslop-gxp-shader/src/link.rs:5587 | `1` reads a 16-bit integer MAD's src1 as the WHOLE register when its half-select bit is clear |
| `VITASLOP_GXP_INDEX16` | vitaslop-runtime/src/render.rs:1278 | `VITASLOP_GXP_INDEX16=0` restores the OLD behaviour: every index widened to u32 whatever |
| `VITASLOP_GXP_INDEXED_REACH` | vitaslop-gxp-shader/src/module.rs:3705 | `VITASLOP_GXP_INDEXED_REACH=0`: size an indexed table's window as declared again - see |
| `VITASLOP_GXP_INPUTS` | vitaslop-gxp-shader/src/usse/mod.rs:705 | Fill in each ORDINARY-REGISTER index load's `stride` - how far apart two consecutive index |
| `VITASLOP_GXP_INPUTS_DIR` | vitaslop-platform/src/gpu.rs:3955 | Whether the once-per-pair `gxp pair <key>: vprog hash ..., fprog hash ...` INDEX should be |
| `VITASLOP_GXP_INPUTS_ORDER` | vitaslop-platform/src/gpu.rs:145 | The output of a diagnostic whose own KNOB is already the gate. |
| `VITASLOP_GXP_INPUTS_SETS` | vitaslop-platform/src/gpu.rs:13409 | - |
| `VITASLOP_GXP_INPUTS_VERTS` | vitaslop-platform/src/gpu.rs:13265 | Diagnostic (`VITASLOP_GXP_INPUTS=<hex-key>[,<hex-key>]` or `=all`): print, ONCE per |
| `VITASLOP_GXP_INTERP` | vitaslop-platform/src/gpu.rs:18888 | - |
| `VITASLOP_GXP_INT_PAIRS` | vitaslop-gxp-shader/tests/corpus.rs:864 | >>> EVERY CORPUS PAIR, RE-LINKED WITH EVERY ATTRIBUTE DECLARED A PLAIN INTEGER, VALIDATES. |
| `VITASLOP_GXP_KEYCOLOR` | vitaslop-platform/src/gpu.rs:3964 | Whether the once-per-pair `gxp pair <key>: vprog hash ..., fprog hash ...` INDEX should be |
| `VITASLOP_GXP_KEYS` | vitaslop-platform/src/gpu.rs:9229 | Diagnostic (`VITASLOP_GXP_KEYS=<hex>,<hex>`): recompile ONLY these shader-pair keys |
| `VITASLOP_GXP_LIVE` | vitaslop-platform/src/gpu.rs:2139 | The guest's real vertex+fragment shaders + their draw inputs, for the GXP->WGSL |
| `VITASLOP_GXP_MAD_MASK16` | vitaslop-gxp-shader/src/usse/decode.rs:1810 | The mad-group destination write mask: a BITMASK over the destination's REGISTER LANES. |
| `VITASLOP_GXP_MEM_OFFSET16` | vitaslop-gxp-shader/tests/conformance.rs:1395 | **A MEMORY LOAD READS CONSECUTIVE GUEST WORDS FROM `pointer + offsets`, AN IMMEDIATE OFFSET |
| `VITASLOP_GXP_MEM_PEEK` | vitaslop-runtime/src/host.rs:18084 | `VITASLOP_MEM_DUMP=<hex addr>:<bytes>[,...]`: write those guest byte ranges to |
| `VITASLOP_GXP_MEM_QUAD` | vitaslop-gxp-shader/src/module.rs:2177 | Rewrite every run of four consecutive word loads from ONE address - `X = {binding}_word(gxp_aN |
| `VITASLOP_GXP_MIPS` | vitaslop-platform/src/gpu.rs:4238 | Whether a chain is built for a seam, ignoring the per-texture exception above. |
| `VITASLOP_GXP_NAN_SITES` | vitaslop-platform/src/gpu.rs:16071 | - |
| `VITASLOP_GXP_NATIVE_F16` | vitaslop-platform/src/gpu.rs:19836 | Build both pipelines for a `color_format` render target. |
| `VITASLOP_GXP_NEGW` | vitaslop-platform/src/gpu.rs:9397 | How to choose the clip-`w` sign correction (`VITASLOP_GXP_NEGW`). |
| `VITASLOP_GXP_NOBLEND` | vitaslop-platform/src/gpu.rs:9219 | Diagnostic (`VITASLOP_GXP_NOBLEND`): force every recompiled pipeline to REPLACE with |
| `VITASLOP_GXP_NODEPTH` | vitaslop-platform/src/gpu.rs:9211 | Diagnostic (`VITASLOP_GXP_NODEPTH`): every recompiled draw keeps its real shading and |
| `VITASLOP_GXP_NO_STENCIL` | vitaslop-conformance-harness/tests/vita_gxmconf.rs:579 | **SCENE 7 - A STENCIL MASK CONFINES A LATER DRAW TO WHAT AN INVISIBLE DRAW MARKED.** |
| `VITASLOP_GXP_ONLY` | vitaslop-platform/src/gpu.rs:9191 | Render ONLY recompiled draws, skipping the fixed-function draw for any call that |
| `VITASLOP_GXP_PACK_COMP0` | vitaslop-gxp-shader/src/link.rs:5560 | `0` restores bit 1 as comp0's high selector bit for a 16-bit PACK source, which is what this |
| `VITASLOP_GXP_PACK_DEST_SLOT` | vitaslop-gxp-shader/src/usse/decode.rs:4936 | The SMLSI slot a repeating 0x40 VPCK's DESTINATION steps under: slot 0, the DEST byte |
| `VITASLOP_GXP_PACK_INTERNAL` | vitaslop-gxp-shader/src/usse/decode.rs:4922 | Where each operand of `word` sits for repeat purposes: the destination, then each source in |
| `VITASLOP_GXP_PAIR` | vitaslop-gxp-shader/tests/corpus.rs:1014 | Link one named (vertex, fragment) pair and print the COMPLETE WGSL module both stages become. |
| `VITASLOP_GXP_PAIRS` | vitaslop-gxp-shader/tests/corpus.rs:1847 | >>> WHY THE PAIRS A RUN ACTUALLY DRAWS FAIL TO LINK, AND WHAT THEIR TWO PROGRAMS DECLARE. |
| `VITASLOP_GXP_PAIR_CORPUS` | vitaslop-gxp-shader/tests/corpus.rs:5633 | Rank a pair corpus by the F16 EMULATION it emits: the `gxp_h*` stores and `unpack2x16float`. |
| `VITASLOP_GXP_PAIR_DUAL` | vitaslop-gxp-shader/tests/corpus.rs:1035 | Link one named (vertex, fragment) pair and print the COMPLETE WGSL module both stages become. |
| `VITASLOP_GXP_PASS_SPLIT_EVERY` | vitaslop-platform/src/gpu.rs:842 | `VITASLOP_GXP_PASS_SPLIT_EVERY=<n>` cuts a render pass every `n` draws, with no shader |
| `VITASLOP_GXP_POSPROBE` | vitaslop-gxp-shader/src/link.rs:5515 | `1` - WHERE DID THIS DRAW'S VERTICES GO? Draw the mesh at CLAMPED normalised coordinates and |
| `VITASLOP_GXP_POSPROBE_DIV` | vitaslop-gxp-shader/src/link.rs:5529 | >>> HOW FAR OFF-SCREEN, NOT ONLY THAT IT IS OFF-SCREEN. |
| `VITASLOP_GXP_PRECOMPILE` | vitaslop-platform/src/gpu.rs:1303 | Whether the shader pairs the guest's patcher names are TRANSLATED ahead of the draws that bind |
| `VITASLOP_GXP_PRECOMPILE_CROSS` | vitaslop-runtime/src/host.rs:18492 | `VITASLOP_GXP_PRECOMPILE_CROSS`: for a title whose `sceGxmShaderPatcherCreateFragmentProgram` |
| `VITASLOP_GXP_PREFETCH_CLAIM` | vitaslop-gxp-shader/src/link.rs:2297 | >>> A PREFETCH COORDINATE THAT LANDS ON LANES THE VERTEX NEVER WRITES IS RE-POINTED TO THE |
| `VITASLOP_GXP_PREFETCH_PROJECTIVE` | vitaslop-gxp-shader/src/link.rs:2266 | Whether a projective prefetch divides by its `w` - see `container::PrefetchLookup::Projective`. |
| `VITASLOP_GXP_PREFETCH_U8` | vitaslop-gxp-shader/src/link.rs:5412 | `0` reads a one-register prefetch as one full-precision component even where the program |
| `VITASLOP_GXP_PREFETCH_UNFED` | vitaslop-gxp-shader/src/link.rs:1141 | >>> THE VERTEX PROGRAM PRODUCES NO SUCH TEXCOORD AT ALL, so the coordinate is the |
| `VITASLOP_GXP_PROBE` | vitaslop-gxp-shader/src/module.rs:365 | Diagnostic (`VITASLOP_GXP_PROBE=<bank><idx>[@<instr>][:f32/:bits=<hex>]`, e.g. |
| `VITASLOP_GXP_PROBE_SCALE` | vitaslop-gxp-shader/src/link.rs:5498 | Divide a probed value before it is written, so an HDR term reads back under the |
| `VITASLOP_GXP_Q2_VEC` | vitaslop-gxp-shader/src/link.rs:5046 | >>> [`GXP_Q2`] ON THE NATIVE ARM: BOTH HALVES IN ONE CONVERSION, ONE PACK AND ONE UNPACK. |
| `VITASLOP_GXP_QUADS` | vitaslop-platform/src/gpu.rs:13262 | Diagnostic (`VITASLOP_GXP_INPUTS=<hex-key>[,<hex-key>]` or `=all`): print, ONCE per |
| `VITASLOP_GXP_REAL_PAIRS` | vitaslop-gxp-shader/tests/corpus.rs:1846 | >>> WHY THE PAIRS A RUN ACTUALLY DRAWS FAIL TO LINK, AND WHAT THEIR TWO PROGRAMS DECLARE. |
| `VITASLOP_GXP_RECOMPILE` | vitaslop-runtime/src/host.rs:19850 | - |
| `VITASLOP_GXP_RETURN` | vitaslop-platform/src/gpu.rs:16462 | - |
| `VITASLOP_GXP_RETURN_KEYS` | vitaslop-platform/src/gpu.rs:3877 | - |
| `VITASLOP_GXP_SA` | vitaslop-platform/src/gpu.rs:13154 | Diagnostic (`VITASLOP_GXP_FMEM=<lane>=<value>` / `<lane>*<factor>`, comma separated): |
| `VITASLOP_GXP_SA_DIRECT` | vitaslop-gxp-shader/src/link.rs:5404 | `0` restores the SA copy loop, `unroll` the constant-subscript copy - see [`resolve_sa_init`]. |
| `VITASLOP_GXP_SA_LITERAL_ALWAYS` | vitaslop-gxp-shader/src/link.rs:3095 | Is a container literal laid down for EVERY read that names its register |
| `VITASLOP_GXP_SA_UNCLAIMED` | vitaslop-gxp-shader/src/link.rs:3049 | Validate that every SA register a stage reads is either inside its default uniform buffer, |
| `VITASLOP_GXP_SIZE_BANKS` | vitaslop-gxp-shader/src/link.rs:5288 | `VITASLOP_GXP_SIZE_BANKS=0` restores the pre-2026-08-20b emission - every register bank |
| `VITASLOP_GXP_SOLID` | vitaslop-platform/src/gpu.rs:9207 | Diagnostic (`VITASLOP_GXP_SOLID`): every recompiled draw outputs solid magenta with |
| `VITASLOP_GXP_STATIC_MEM` | vitaslop-gxp-shader/src/module.rs:2257 | Resolve at EMIT time every memory load whose address is a window's own base register plus a |
| `VITASLOP_GXP_STRICT` | vitaslop-platform/src/gpu.rs:17624 | >>> A PAIR THIS RECOMPILER CANNOT TRANSLATE DROPS ITS DRAWS. |
| `VITASLOP_GXP_TEXCOORD_NUDGE` | vitaslop-platform/src/gpu.rs:3821 | >>> TEMPORARY TELEMETRY (`VITASLOP_GXM_DRAW_PROBE=<keyspec>`). |
| `VITASLOP_GXP_TSTMSK_DEST` | vitaslop-gxp-shader/src/usse/decode.rs:1144 | - |
| `VITASLOP_GXP_UNCLIPPED_DEPTH` | vitaslop-platform/src/gpu.rs:421 | - |
| `VITASLOP_GXP_UNFED_TEXCOORD_DEFAULT` | vitaslop-gxp-shader/src/link.rs:1225 | Match the vertex's declared varying outputs to the fragment's declared interpolants BY |
| `VITASLOP_GXP_VARYING_CODE_ORDER` | vitaslop-gxp-shader/src/link.rs:5584 | `0` keeps a vertex program's output varyings in the container's order (attribute order or |
| `VITASLOP_GXP_VARYING_LAYOUT` | vitaslop-gxp-shader/src/link.rs:2161 | Diagnostic (`VITASLOP_GXP_VARYING_LAYOUT=<vhash>:<usage>@<lane>x<comps>,...`): plan ONE |
| `VITASLOP_GXP_VARYING_ORDER` | vitaslop-gxp-shader/src/link.rs:1499 | The vertex lane order the paired FRAGMENT's declaration implies, or `None` when the two |
| `VITASLOP_GXP_VARYING_RESOLVE` | vitaslop-gxp-shader/tests/corpus.rs:5180 | Which VERTEX programs the forwarding resolver's lane RESERVATION moves, and where to. |
| `VITASLOP_GXP_VERT` | vitaslop-gxp-shader/tests/corpus.rs:5105 | The complete LINKED WGSL for one pair, selected by `VITASLOP_GXP_VERT` + `VITASLOP_GXP_FRAG` |
| `VITASLOP_GXP_VERTEX_PASSTHROUGH` | vitaslop-platform/src/gpu.rs:1276 | `VITASLOP_GXP_VERTEX_PASSTHROUGH=0` makes every recompiled draw repack its vertex stream into |
| `VITASLOP_GXP_VPROBE` | vitaslop-gxp-shader/src/link.rs:5510 | `1` - WHERE DID THIS DRAW'S VERTICES GO? Draw the mesh at CLAMPED normalised coordinates and |
| `VITASLOP_GXP_VP_TRACE` | vitaslop-conformance-harness/tests/vita_gxmconf.rs:81 | WHERE a scene painted, in one line: the count in each quadrant-edge half plus the pixel at |
| `VITASLOP_GXP_WGSL_DIR` | vitaslop-gxp-shader/tests/corpus.rs:5108 | The complete LINKED WGSL for one pair, selected by `VITASLOP_GXP_VERT` + `VITASLOP_GXP_FRAG` |
| `VITASLOP_GXP_WGSL_LOG` | vitaslop-platform/src/gpu.rs:18520 | - |
| `VITASLOP_GXP_WGSL_OUT` | vitaslop-gxp-shader/tests/corpus.rs:7707 | Write ONE linked WGSL module per FRAGMENT blob to `VITASLOP_GXP_WGSL_OUT` - the first vertex |
| `VITASLOP_GXP_WGSL_OVERRIDE_DIR` | vitaslop-platform/src/gpu.rs:19017 | - |
| `VITASLOP_GXP_WGSL_OVERRIDE_TEXT` | vitaslop-platform/src/gpu.rs:19037 | - |
| `VITASLOP_GXP_YFLIP` | vitaslop-platform/src/gpu.rs:9197 | Flip clip Y (`VITASLOP_GXP_YFLIP`, default off). |
| `VITASLOP_GXP_ZFIX` | vitaslop-platform/src/gpu.rs:9195 | Apply the GXM (GL-style, NDC z in [-1,1]) -> WebGPU (z in [0,1]) clip-depth remap |
| `VITASLOP_HB_CMP` | vitaslop-native/tests/homebrew_qsort.rs:115 | Block trace for one run: set `VITASLOP_TRACE_BLOCKS=<lo>-<hi>` (emit-time) and |
| `VITASLOP_HB_CMP_BIT` | vitaslop-native/tests/homebrew_qsort.rs:67 | - |
| `VITASLOP_HB_DUMP` | vitaslop-native/tests/homebrew_qsort.rs:48 | - |
| `VITASLOP_HB_IMAGE` | vitaslop-native/tests/homebrew_strings.rs:36 | A VM over the whole image (the routines read past string ends in aligned words, so |
| `VITASLOP_HB_N` | vitaslop-native/tests/homebrew_qsort.rs:99 | Block trace for one run: set `VITASLOP_TRACE_BLOCKS=<lo>-<hi>` (emit-time) and |
| `VITASLOP_HB_QSORT` | vitaslop-native/tests/homebrew_qsort.rs:115 | Block trace for one run: set `VITASLOP_TRACE_BLOCKS=<lo>-<hi>` (emit-time) and |
| `VITASLOP_HB_STRCMP` | vitaslop-native/tests/homebrew_strings.rs:8 | - |
| `VITASLOP_HB_STRLEN` | vitaslop-native/tests/homebrew_strings.rs:35 | A VM over the whole image (the routines read past string ends in aligned words, so |
| `VITASLOP_HEADLESS_FRAMES` | vitaslop-desktop/src/retail.rs:1041 | Headless self-check of the retail path (NO window): load `dir`, optionally drive a |
| `VITASLOP_HEADLESS_NO_TAPS` | vitaslop-desktop/src/retail.rs:1047 | Headless self-check of the retail path (NO window): load `dir`, optionally drive a |
| `VITASLOP_HEADLESS_RECIPE` | vitaslop-desktop/src/retail.rs:1044 | Headless self-check of the retail path (NO window): load `dir`, optionally drive a |
| `VITASLOP_HEADLESS_RENDER_FROM` | vitaslop-desktop/src/retail.rs:1140 | - `VITASLOP_HEADLESS_SHOT_EVERY` - also write `<shot_dir>/fNNNNNN.png` every N display |
| `VITASLOP_HEADLESS_SHOT_EVERY` | vitaslop-desktop/src/retail.rs:1133 | - `VITASLOP_HEADLESS_SHOT_EVERY` - also write `<shot_dir>/fNNNNNN.png` every N display |
| `VITASLOP_HEADLESS_SHOT_FROM` | vitaslop-desktop/src/retail.rs:1135 | - `VITASLOP_HEADLESS_SHOT_EVERY` - also write `<shot_dir>/fNNNNNN.png` every N display |
| `VITASLOP_HEADLESS_SHOT_TO` | vitaslop-desktop/src/retail.rs:1135 | - `VITASLOP_HEADLESS_SHOT_EVERY` - also write `<shot_dir>/fNNNNNN.png` every N display |
| `VITASLOP_HEADLESS_TIMING` | vitaslop-desktop/src/retail.rs:1048 | Headless self-check of the retail path (NO window): load `dir`, optionally drive a |
| `VITASLOP_HEAP_TRACE` | vitaslop-platform/src/heap.rs:51 | # The large-allocation ledger (`VITASLOP_HEAP_TRACE=<min MB>`, native only) |
| `VITASLOP_HITCH_AT` | vitaslop-web/src/lib.rs:5596 | - |
| `VITASLOP_HOLD_BUTTONS` | vitaslop-native/tests/retail_boot_probe.rs:96 | A minimal host world: a monotonic clock advancing one 60Hz tick per poll, no |
| `VITASLOP_HOLD_FROM` | vitaslop-native/tests/retail_boot_probe.rs:97 | A minimal host world: a monotonic clock advancing one 60Hz tick per poll, no |
| `VITASLOP_HOLD_MEM` | vitaslop-native/tests/retail_boot_probe.rs:715 | - |
| `VITASLOP_HOLD_TOUCH` | vitaslop-native/tests/retail_boot_probe.rs:114 | - |
| `VITASLOP_HOME` | vitaslop-desktop/src/library.rs:12 | - |
| `VITASLOP_HOSTCALL_WATCH` | vitaslop-runtime/src/vita/mod.rs:898 | `VITASLOP_HOSTCALL_WATCH=<hex addr>[,<hex addr>...]` - print every host call that passes one |
| `VITASLOP_HOST_WRITE_WATCH` | vitaslop-runtime/src/host.rs:819 | `VITASLOP_HOST_WRITE_WATCH=<hex addr>[,...]`: report every write a HOST CALL makes to one |
| `VITASLOP_INGEST_DEBUG` | vitaslop-runtime/src/ingest/filesdb.rs:172 | Resolve every non-directory node to its full '/'-separated path (no |
| `VITASLOP_INPLACE_COMPARE` | vitaslop-runtime/src/host.rs:24985 | Whether a draw's VERTEX AND INDEX BYTES are read at `sceGxmEndScene` rather than at the |
| `VITASLOP_INPUT_RECIPE` | vitaslop-native/tests/retail_boot_probe.rs:388 | - |
| `VITASLOP_INSTANCED_MEMO` | vitaslop-runtime/src/host.rs:3692 | Whether a draw decodes only the texture units its fragment program DECLARES - see |
| `VITASLOP_IO_BANDWIDTH_KIBPS` | vitaslop-runtime/src/vita/iofilemgr.rs:22 | Modelled sequential read bandwidth, in KiB per second |
| `VITASLOP_IO_PARK_THRESHOLD_US` | vitaslop-runtime/src/vita/iofilemgr.rs:88 | >>> THROUGH THE KNOB SEAM, NOT `std::env` - THE BROWSER HAS NO ENVIRONMENT. |
| `VITASLOP_IO_REQUEST_US` | vitaslop-runtime/src/vita/iofilemgr.rs:81 | Fixed per-request cost in microseconds (`VITASLOP_IO_REQUEST_US`): the command |
| `VITASLOP_IO_WALL_FLOOR` | vitaslop-runtime/src/host.rs:3786 | `VITASLOP_IO_WALL_FLOOR` (default on): the wall-clock floor that advances the game clock |
| `VITASLOP_JPEG_BENCH` | vitaslop-runtime/src/vita/jpeg.rs:546 | What this decoder costs, on a real image, so "is it fast enough" is a number. |
| `VITASLOP_JS_PROFILE` | vitaslop-web/src/lib.rs:5131 | Read a knob from JavaScript - a guest worker's script shares the override table (it is in |
| `VITASLOP_LINK_NO_BODY_MEMO` | vitaslop-gxp-shader/src/link.rs:556 | A stage body's WGSL text, emitted once per distinct program blob: the text is a function of |
| `VITASLOP_LINK_NO_MEMO` | vitaslop-gxp-shader/src/link.rs:480 | >>> A TRANSLATION IS A FUNCTION OF ITS INPUTS, SO IT IS DONE ONCE AND AHEAD WHEN IT CAN BE. |
| `VITASLOP_LINK_NO_RC_MEMO` | vitaslop-gxp-shader/src/link.rs:528 | A program's decode, done once per distinct blob. |
| `VITASLOP_LINK_OLD_F16` | vitaslop-gxp-shader/src/link.rs:4729 | Whether `expr` can only ever hold a value that is ALREADY exactly a 16-bit float, so the |
| `VITASLOP_LINK_OLD_PARSE` | vitaslop-gxp-shader/src/link.rs:4241 | Split `line` into the store it performs, if it is one of the recognised shapes, and the |
| `VITASLOP_LINK_OLD_REWRITE` | vitaslop-gxp-shader/src/link.rs:4628 | [`rewrite_half_line`] appended to `out` - and a line with nothing to rewrite (no kept store, |
| `VITASLOP_LINK_OLD_STRAY` | vitaslop-gxp-shader/src/link.rs:4343 | Every bank subscript in `text`: `Ok` names a register, `Err` a bank subscripted by something |
| `VITASLOP_LINK_PROFILE` | vitaslop-gxp-shader/src/link.rs:576 | `VITASLOP_LINK_PROFILE=1` (native only): cumulative microseconds per link phase - |
| `VITASLOP_LINK_VAR_ADDEND` | vitaslop-runtime/src/link.rs:846 | Whether variable-import fixups apply the code word's addend - see [`apply_var_fixups`]. |
| `VITASLOP_LOG` | vitaslop-desktop/src/log.rs:30 | Whether captured events are also written to stderr. |
| `VITASLOP_MAIN_EXEC` | vitaslop-web/src/lib.rs:4809 | The title's modules with the executable a `sceAppMgrLoadExec` named in the main |
| `VITASLOP_MAX_FRAMES` | vitaslop-native/tests/retail_boot_probe.rs:512 | - |
| `VITASLOP_MAX_ROUNDS` | vitaslop-native/tests/retail_boot_probe.rs:517 | - |
| `VITASLOP_MEM_DUMP` | vitaslop-runtime/src/host.rs:18076 | `VITASLOP_MEM_DUMP=<hex addr>:<bytes>[,...]`: write those guest byte ranges to |
| `VITASLOP_MEM_DUMP_AT` | vitaslop-runtime/src/host.rs:18078 | `VITASLOP_MEM_DUMP=<hex addr>:<bytes>[,...]`: write those guest byte ranges to |
| `VITASLOP_MEM_FIND` | vitaslop-web/src/lib.rs:2298 | TEMPORARY TELEMETRY (`VITASLOP_MEM_FIND=<frame>:<hex bytes>[,<hex bytes>...]`): at the first |
| `VITASLOP_MOVIE` | vitaslop-runtime/src/vita/video.rs:2562 | A track whose codec this engine does not decode is not offered at all: the title's |
| `VITASLOP_MOVIE_AUDIO_BACKLOG` | vitaslop-runtime/src/vita/video.rs:1229 | How many decoded frames to hold for a title that is not collecting them. |
| `VITASLOP_MOVIE_DUMP_DIR` | vitaslop-runtime/src/vita/avcdec.rs:1292 | >>> AND WHAT THE PICTURE ACTUALLY LOOKS LIKE, because "a picture arrived" and "the movie |
| `VITASLOP_MOVIE_DUMP_EVERY` | vitaslop-runtime/src/vita/avcdec.rs:1292 | >>> AND WHAT THE PICTURE ACTUALLY LOOKS LIKE, because "a picture arrived" and "the movie |
| `VITASLOP_MOVIE_PICTURE_HASH` | vitaslop-runtime/src/vita/avcdec.rs:168 | Pictures handed to the guest so far, which is what `VITASLOP_MOVIE_PICTURE_HASH` |
| `VITASLOP_MOVIE_SUBSTITUTE` | vitaslop-runtime/src/vita/video.rs:416 | >>> OPEN A DIFFERENT MOVIE THAN THE TITLE ASKED FOR |
| `VITASLOP_MP4_AUDIO` | vitaslop-runtime/src/vita/video.rs:1618 | The tracks this engine will hand units for, as cursors, in the order they appear in the |
| `VITASLOP_MP4_AUDIO_READ_AHEAD_MS` | vitaslop-runtime/src/vita/video.rs:2160 | Read-ahead for a player that ENABLED an audio stream - see `movie_unit_wait_us`. |
| `VITASLOP_MP4_GUEST_IO_KB` | vitaslop-runtime/src/vita/video.rs:152 | The I/O buffer a callback-read movie is read through, and the largest single read. |
| `VITASLOP_MP4_GUEST_OPS` | vitaslop-runtime/src/vita/video.rs:310 | The body of [`mp4_open_file`], as a plain function so it can use guard clauses. |
| `VITASLOP_MP4_READ_AHEAD_MS` | vitaslop-runtime/src/vita/video.rs:2182 | [`READ_AHEAD_US`], or `VITASLOP_MP4_READ_AHEAD_MS` - the arm for a player that keeps its |
| `VITASLOP_MP4_UNITS` | vitaslop-runtime/src/vita/video.rs:1792 | `VITASLOP_MP4_UNITS=none`: never return an access unit. |
| `VITASLOP_NAME` | vitaslop-desktop/src/shell.rs:735 | - |
| `VITASLOP_NEON_CACHE` | vitaslop-transpiler/src/emit.rs:3408 | Whether emitted modules hold the low NEON bank in locals across a run of vector |
| `VITASLOP_NGS_BLOCK_STATE` | vitaslop-runtime/src/vita/ngs.rs:832 | SceInt32 sceNgsVoiceSetParamsBlock(SceNgsHVoice voice, const SceNgsModuleParamHeader |
| `VITASLOP_NGS_NEG_LOOP` | vitaslop-runtime/src/vita/at9.rs:1447 | A NEGATIVE `nLoopCount` means "play this buffer once and move on", not "repeat it |
| `VITASLOP_NGS_VOICE_HANDLE_MEMO` | vitaslop-runtime/src/vita/ngs.rs:455 | SceInt32 sceNgsRackGetVoiceHandle(SceNgsHRack rack, SceUInt32 index, SceNgsHVoice *handle) |
| `VITASLOP_NGS_VOICE_PEAKS` | vitaslop-runtime/src/vita/at9.rs:349 | `VITASLOP_NGS_VOICE_PEAKS=1` only: this voice's lifetime source peak, sum of squares and |
| `VITASLOP_NGS_ZERO_LEVEL` | vitaslop-runtime/src/vita/at9.rs:1453 | A NEGATIVE `nLoopCount` means "play this buffer once and move on", not "repeat it |
| `VITASLOP_NID_DIGEST` | vitaslop-runtime/src/host.rs:21444 | The cross-engine host-call digest for the frame in progress - see [`NidDigest`]. |
| `VITASLOP_NO_BC` | vitaslop-runtime/src/render.rs:2319 | Decode a whole BC1/BC2/BC3 block to its sixteen RGBA8 texels at once. |
| `VITASLOP_NO_FAST_IMPORT` | vitaslop-runtime/src/vita/mod.rs:379 | Whether `func_nid`'s handler can only ever CONTINUE, so the transpiler may route the |
| `VITASLOP_NO_INLINE_CLIB` | vitaslop-runtime/src/vita/mod.rs:582 | `VITASLOP_NO_INLINE_CLIB`: route `sceClibMemcpy`, `sceClibMemset` and `sceClibMemcmp` |
| `VITASLOP_NO_INLINE_DELAY` | vitaslop-runtime/src/vita/mod.rs:317 | `VITASLOP_NO_INLINE_DELAY`: route every `sceKernelDelayThread` through the host, leaving |
| `VITASLOP_NO_INLINE_IMPORTS` | vitaslop-runtime/src/host.rs:8566 | >>> WHO HAS WRITTEN THE CONTEXT'S TEXTURE SLOTS, counted for the failure report above. |
| `VITASLOP_NO_INLINE_LWMUTEX` | vitaslop-runtime/src/vita/mod.rs:684 | `VITASLOP_NO_INLINE_LWMUTEX`: route the lightweight-mutex lock and unlock through the |
| `VITASLOP_NO_INLINE_MUTEX` | vitaslop-runtime/src/vita/mod.rs:704 | `VITASLOP_NO_INLINE_MUTEX`: route `sceKernelLockMutex`/`sceKernelUnlockMutex` through the |
| `VITASLOP_NO_INLINE_RESERVE` | vitaslop-runtime/src/vita/mod.rs:619 | `VITASLOP_NO_INLINE_RESERVE`: route `sceGxmReserve{Vertex,Fragment}DefaultUniformBuffer` |
| `VITASLOP_NO_INLINE_STUBS` | vitaslop-runtime/src/vita/mod.rs:556 | `VITASLOP_NO_INLINE_STUBS`: route the constant-return stubs through the host, leaving |
| `VITASLOP_NO_INLINE_TEXTURE` | vitaslop-runtime/src/host.rs:8566 | >>> WHO HAS WRITTEN THE CONTEXT'S TEXTURE SLOTS, counted for the failure report above. |
| `VITASLOP_NO_INLINE_UNIFORM_DATA` | vitaslop-runtime/src/vita/mod.rs:653 | `VITASLOP_NO_INLINE_UNIFORM_DATA`: route `sceGxmSetUniformDataF` through the host, |
| `VITASLOP_NO_NGS_MIX` | vitaslop-runtime/src/vita/audio.rs:333 | `VITASLOP_NO_NGS_MIX`: skip the NGS decode-and-mix entirely, leaving the guest's |
| `VITASLOP_OUTPUT_SIZE` | vitaslop-web/src/lib.rs:3419 | - |
| `VITASLOP_PACE_FLOOR_FREE` | vitaslop-web/src/lib.rs:2181 | Whether the live loop leaves the wall floor's clock gain out of a frame's charge - see the |
| `VITASLOP_PACE_REFUND` | vitaslop-web/src/lib.rs:2188 | Whether the pacing loop refunds the one-period floor it charged short frames out of later |
| `VITASLOP_PACKED_CACHE_MB` | vitaslop-platform/src/gpu.rs:4281 | >>> AND A CAP IN ENTRIES IS NOT A BOUND ON MEMORY. |
| `VITASLOP_PATCH_STUBS` | vitaslop-native/tests/retail_boot_probe.rs:467 | - |
| `VITASLOP_PAUSE_ON_BLUR` | vitaslop-desktop/src/retail.rs:2314 | Run the retail title in `dir` in a live window until the window closes or the guest |
| `VITASLOP_PEEK` | vitaslop-desktop/src/retail.rs:729 | Guest memory at `addr`, for `VITASLOP_PEEK`. |
| `VITASLOP_PERF` | vitaslop-native/src/perf.rs:43 | Is perf accounting on (`VITASLOP_PERF` set)? Read once and cached. |
| `VITASLOP_PERF_CONSOLE` | vitaslop-web/src/lib.rs:2414 | Whether the per-window performance report is also written to the browser CONSOLE |
| `VITASLOP_PIPELINE_DEFER` | vitaslop-web/src/lib.rs:8722 | >>> THE PIPELINES THIS FRAME NEEDS, CREATED ASYNC - see `GxmRenderer::warm_pipelines` |
| `VITASLOP_PIPELINE_DEFER_MAX_MS` | vitaslop-web/src/lib.rs:8879 | `VITASLOP_PIPELINE_DEFER_MAX_MS`: overrides [`PIPE_DEFER_MAX_MS_DEFAULT`]. |
| `VITASLOP_PIPE_LOG` | vitaslop-platform/src/gpu.rs:8824 | `VITASLOP_PIPE_LOG=1`: one status line per pipeline built - the frame, the pair, and the |
| `VITASLOP_PIXEL_TRACE` | vitaslop-runtime/src/render.rs:5525 | Draw one scene onto an EXISTING framebuffer and depth buffer, composing with whatever |
| `VITASLOP_PKG` | vitaslop-runtime/src/ingest/stream.rs:1003 | A pkg's item table, read over a source that only ever hands out RANGES |
| `VITASLOP_POISON_UNRESOLVED_VARS` | vitaslop-runtime/src/link.rs:475 | - |
| `VITASLOP_POKE` | vitaslop-native/tests/retail_boot_probe.rs:683 | - |
| `VITASLOP_POLL_ADDR` | vitaslop-native/src/threaded.rs:1990 | Guest address to sample after each host call, from `VITASLOP_POLL_ADDR` (hex). |
| `VITASLOP_PREPARE_SPLIT` | vitaslop-platform/src/gpu.rs:7772 | Where the milliseconds INSIDE one `prepare` go, plus the bytes each phase moved. |
| `VITASLOP_PREPOKE` | vitaslop-native/tests/retail_boot_probe.rs:490 | - |
| `VITASLOP_PRESENT_LOG` | vitaslop-web/src/lib.rs:3107 | Render one freshly-executed FRAME - every scene the guest submitted between |
| `VITASLOP_PRESENT_PROBE` | vitaslop-web/src/lib.rs:1024 | Reads back WHAT WE PRESENTED, when `VITASLOP_PRESENT_PROBE` asks for it. |
| `VITASLOP_PRESENT_SHOT` | vitaslop-web/src/lib.rs:1817 | If a mapped read is ready, describe it and release the buffer. |
| `VITASLOP_PROBE_SAMPLE` | vitaslop-platform/src/gpu.rs:932 | >>> TEMPORARY SCAFFOLDING (`VITASLOP_PROBE_SAMPLE=<hex>+<hex>...`). |
| `VITASLOP_PROMOTE_POISON` | vitaslop-transpiler/src/emit.rs:3488 | `VITASLOP_PROMOTE_POISON=<n>` - the FALSIFIER for register promotion. |
| `VITASLOP_PROMOTE_REGS` | vitaslop-native/src/threaded.rs:2602 | A concise trap description (kind + message), matching the sync `Vm`'s detail. |
| `VITASLOP_PSARC` | vitaslop-runtime/src/psarc.rs:430 | Read a REAL archive: `VITASLOP_PSARC=<path to a .psarc>`, optionally |
| `VITASLOP_PSARC_FILE` | vitaslop-runtime/src/psarc.rs:431 | Read a REAL archive: `VITASLOP_PSARC=<path to a .psarc>`, optionally |
| `VITASLOP_PVRTC_DECODE` | vitaslop-runtime/src/render.rs:7145 | Whether PVRTC decodes a whole face at a time (the default) or one texel at a time. |
| `VITASLOP_QUANTUM_CPU_US` | vitaslop-runtime/src/host.rs:22352 | Game-clock time charged for one [`QUANTUM_ARM`] of guest execution, in microseconds. |
| `VITASLOP_QUANTUM_FUEL` | vitaslop-native/tests/retail_boot_probe.rs:426 | - |
| `VITASLOP_RAW_WORD_VIEW` | vitaslop-platform/src/gpu.rs:2924 | Whether a sampler reading one 32-bit word of a raw 64-bit surface as a double-width texture |
| `VITASLOP_REGION_CLIP_SCENE` | vitaslop-runtime/src/vita/gxm.rs:1888 | - |
| `VITASLOP_REGTRACE` | vitaslop-native/src/threaded.rs:1590 | `VITASLOP_REGTRACE=<lo>-<hi>:<path>` - append the reg+flag file per block entry in |
| `VITASLOP_REGTRACE_MAX` | vitaslop-native/src/threaded.rs:1755 | `VITASLOP_REGTRACE_MAX=<n>` caps the register trace at `n` lines (0 = unbounded). |
| `VITASLOP_REGTRACE_VFP` | vitaslop-native/src/threaded.rs:1794 | Append one `pc r0..r15 n z c v [mADDR=VAL...] tTHID` line (all hex, flags 0/1) to the |
| `VITASLOP_REGTRACE_WATCH` | vitaslop-native/src/threaded.rs:1515 | The `VITASLOP_REGTRACE_WATCH` words, formatted as ` mADDR=VALUE` fields ready to append |
| `VITASLOP_REPLAY_EARLY` | vitaslop-web/src/frame_replay.rs:76 | - |
| `VITASLOP_REPLAY_TARGET` | vitaslop-web/src/frame_replay.rs:135 | - |
| `VITASLOP_RESIDENT_COMPACT_EARLY` | vitaslop-platform/src/gpu.rs:1033 | The 2026-09-28 memory bounds, each with its arm back (`=0`): `VITASLOP_SUBRECT_POOL` |
| `VITASLOP_RESIDENT_GEOM` | vitaslop-platform/src/gpu.rs:9494 | Repacked vertices and expanded indices that have not changed since the renderer first |
| `VITASLOP_RESIDENT_GEOM_MB` | vitaslop-platform/src/gpu.rs:9537 | The byte budget for each of the two heaps (`VITASLOP_RESIDENT_GEOM_MB`, per heap). |
| `VITASLOP_RESIDENT_IDLE_FRAMES` | vitaslop-platform/src/gpu.rs:5058 | >>> FORGET THE SLICES NO DRAW HAS BOUND FOR `idle` FRAMES - their `Arc`s are RUST |
| `VITASLOP_RESOLVE_CHUNK` | vitaslop-runtime/src/host.rs:24475 | `VITASLOP_RESOLVE_CHUNK=<draws>`: how many draws the RESOLVER reads per hold of the snapshot |
| `VITASLOP_ROUNDS_PER_FRAME` | vitaslop-native/tests/retail_boot_probe.rs:661 | - |
| `VITASLOP_RTT_BG_CACHE` | vitaslop-platform/src/gpu.rs:796 | `VITASLOP_RTT_BG_CACHE=0` restores the OLD behaviour: a sampler bind group naming a render |
| `VITASLOP_RTT_CLEAR_PROBE` | vitaslop-native/src/wgpu_render.rs:721 | - |
| `VITASLOP_RTT_GRID_FRAMES` | vitaslop-web/src/browser_sched.rs:4740 | - |
| `VITASLOP_RTT_PROBE_FIND` | vitaslop-runtime/src/rtt_writeback.rs:135 | TEMPORARY TELEMETRY (`VITASLOP_RTT_PROBE_FIND=<rrggbb>[+<rrggbb>...]`): name every |
| `VITASLOP_RTT_PROBE_LOG` | vitaslop-runtime/src/rtt_writeback.rs:85 | TEMPORARY TELEMETRY (`VITASLOP_RTT_PROBE_LOG=1`): print every written-back target's |
| `VITASLOP_RTT_STALE_EXTENT` | vitaslop-platform/src/gpu.rs:2880 | Whether a held render target is refused as a sampler source when this frame has written its |
| `VITASLOP_RTT_SUBRECT` | vitaslop-conformance-harness/tests/vita_gxmconf_real.rs:230 | **SCENES 16 + 17 - A TEXTURE NAMING A SUB-RECTANGLE OF A RENDERED TARGET SAMPLES THAT RECTANGLE.** |
| `VITASLOP_RTT_WRITEBACK_DELAY_MS` | vitaslop-web/src/lib.rs:650 | `VITASLOP_RTT_WRITEBACK_DELAY_MS`: a copy is not handed over, and its slot stays mapped, |
| `VITASLOP_RTT_WRITEBACK_FLOAT` | vitaslop-runtime/src/rtt_writeback.rs:294 | Write ONE target's rendered pixels (`rgba`, tightly packed `w*h*4`, memory order R,G,B,A) |
| `VITASLOP_RTT_WRITEBACK_MAX_AGE_MS` | vitaslop-web/src/lib.rs:1135 | >>> THE WRITEBACK AGE BOUND: a present DECLINES while the oldest render-target copy |
| `VITASLOP_RTT_WRITEBACK_SYNC_MS` | vitaslop-web/src/lib.rs:5768 | - |
| `VITASLOP_SAMPLER_NARROW` | vitaslop-runtime/src/host.rs:3688 | Whether a draw decodes only the texture units its fragment program DECLARES - see |
| `VITASLOP_SCAN_WORD` | vitaslop-native/tests/retail_boot_probe.rs:963 | - |
| `VITASLOP_SCENE_LIMIT` | vitaslop-native/tests/retail_boot_probe.rs:447 | - |
| `VITASLOP_SCHED_CORES` | vitaslop-runtime/src/sched.rs:644 | `VITASLOP_SCHED_CORES=<n>`: cap the baton to the top `n` runnable PRIORITIES, as the |
| `VITASLOP_SCHED_RR` | vitaslop-runtime/src/sched.rs:660 | `VITASLOP_SCHED_RR=1`: round-robin every runnable thread, ignoring priority. |
| `VITASLOP_SCHED_TRACE` | vitaslop-runtime/src/sched.rs:670 | `VITASLOP_SCHED_TRACE=<from>-<to>` (display frames, inclusive): print one line per |
| `VITASLOP_SEMA_TRAIL` | vitaslop-runtime/src/vita/sync.rs:378 | TEMPORARY DIAGNOSTIC. |
| `VITASLOP_SET_EVF` | vitaslop-native/tests/retail_boot_probe.rs:693 | - |
| `VITASLOP_SHOT_DIR` | vitaslop-native/tests/retail_boot_probe.rs:207 | Read and format one watched value from current guest memory. |
| `VITASLOP_SHOT_FRAMES` | vitaslop-web/src/lib.rs:1708 | `VITASLOP_SHOT_FRAMES=<f>,<f>,...`: GUEST frames to photograph, oldest first - the first |
| `VITASLOP_SHOT_LAST` | vitaslop-native/tests/retail_boot_probe.rs:445 | - |
| `VITASLOP_SIGNATURE` | vitaslop-native/src/recipe_runner.rs:101 | The determinism signature over the observable output (render stream + egress), |
| `VITASLOP_SIGNATURE_EVERY` | vitaslop-native/src/recipe_runner.rs:506 | `VITASLOP_SIGNATURE_EVERY=<n>`: print the RUNNING determinism signature every `n` stepped |
| `VITASLOP_SLOW_FRAME_US` | vitaslop-web/src/lib.rs:1147 | >>> THE WRITEBACK AGE BOUND: a present DECLINES while the oldest render-target copy |
| `VITASLOP_SMP` | vitaslop-conformance-harness/src/lib.rs:214 | The SAME corpus, transpiled as an SMP build (`VITASLOP_SMP`'s codegen: atomic |
| `VITASLOP_SMP_ASYNC_RESOLVE` | vitaslop-runtime/src/host.rs:19274 | >>> THE FLIP'S RESOLVE, READ OFF THE HOST LOCK: take the job (under the lock, cheap). |
| `VITASLOP_SMP_DEFER_TEXTURES` | vitaslop-runtime/src/host.rs:24890 | >>> A DRAW'S TEXTURE BINDINGS, PROVEN WHERE ITS GEOMETRY IS READ. |
| `VITASLOP_SMP_EARLY_PAUSE_ALL` | vitaslop-web/src/smp.rs:99 | Pause every guest worker for a small-target completion's whole render + readback |
| `VITASLOP_SMP_FIBER_NEAR` | vitaslop-web/src/smp.rs:532 | Whether a fiber's backing thread is placed on its runner's worker (`VitaState::spawn_near`). |
| `VITASLOP_SMP_FORWARD` | vitaslop-runtime/src/vita/mod.rs:207 | [`smp_owner_only`] for a run that knows whether its guest workers each hold their own view |
| `VITASLOP_SMP_GUEST_SLOW` | vitaslop-runtime/src/perf.rs:786 | `VITASLOP_SMP_SLOW_EXCLUDE_GXM=1` - a MEASUREMENT RIG for the phone proxy only. |
| `VITASLOP_SMP_LWMUTEX_INLINE` | vitaslop-runtime/src/vita/mod.rs:138 | >>> WHICH INLINE FORMS SURVIVE PARALLEL GUEST EXECUTION, and what the rest become. |
| `VITASLOP_SMP_OVERLAP` | vitaslop-web/src/smp.rs:69 | Present frame N while the guest workers build N+1 - the gate runs ONE frame ahead of the |
| `VITASLOP_SMP_PLACE` | vitaslop-web/src/smp.rs:456 | `VITASLOP_SMP_PLACE=spread`: bind a new thread whose mask allows several workers to the one |
| `VITASLOP_SMP_RT_THROUGH_GATE` | vitaslop-web/src/smp.rs:863 | Waits on REAL time (it has been woken from a wall park - the audio output thread): it |
| `VITASLOP_SMP_SLOW_EXCLUDE_GXM` | vitaslop-runtime/src/perf.rs:785 | `VITASLOP_SMP_SLOW_EXCLUDE_GXM=1` - a MEASUREMENT RIG for the phone proxy only. |
| `VITASLOP_SMP_SPIN_CAP_US` | vitaslop-web/src/smp.rs:429 | >>> ADAPTIVE SPIN: the ceiling, in microseconds, on how long a guest worker polls its doorbell |
| `VITASLOP_SMP_SPIN_US` | vitaslop-web/src/smp.rs:415 | `VITASLOP_SMP_SPIN_US`: how long a guest worker with nothing to run polls its doorbell |
| `VITASLOP_SMP_TIMEOUT_YIELDS` | vitaslop-web/src/smp.rs:128 | Whether a thread released by TIME (a timed wait expiring, a sleep ending) yields to threads |
| `VITASLOP_SMP_TRACE` | vitaslop-web/src/smp.rs:676 | Whether `VITASLOP_SMP_TRACE` is armed at all - for a caller that would otherwise take |
| `VITASLOP_SMP_UNPIN_STARVED` | vitaslop-web/src/smp.rs:521 | Whether a thread pinned to a core that already carries a better-priority thread is placed |
| `VITASLOP_SMP_WORKERS` | vitaslop-web/src/smp.rs:264 | How many guest workers a parallel run uses: `VITASLOP_SMP_WORKERS`, default 3 - the Vita |
| `VITASLOP_SNAPSHOT` | vitaslop-native/src/threaded.rs:1576 | `VITASLOP_SNAPSHOT=<hexpc>:<path>` - dump full state on first entry to block `hexpc`. |
| `VITASLOP_SNAPSHOT_BUDGET_MB` | vitaslop-runtime/src/host.rs:4405 | Byte budget for retained texture snapshots, scaled to the device |
| `VITASLOP_SNAPSHOT_DENSE` | vitaslop-native/src/threaded.rs:1692 | Dump the full guest state (all non-zero pages + r0..r15 + NZCV) to `path`, in the |
| `VITASLOP_SNAPSHOT_SKIP` | vitaslop-native/src/threaded.rs:1644 | `VITASLOP_SNAPSHOT_SKIP=<n>` - skip the first `n` entries to the snapshot block before |
| `VITASLOP_SOFTWARE` | vitaslop-desktop/src/retail.rs:2188 | - |
| `VITASLOP_SSAA` | vitaslop-platform/src/gpu.rs:20301 | Set the supersample factor: 1 (default) renders the scene straight into the caller's |
| `VITASLOP_STAGING_TRIM` | vitaslop-platform/src/gpu.rs:1035 | The 2026-09-28 memory bounds, each with its arm back (`=0`): `VITASLOP_SUBRECT_POOL` |
| `VITASLOP_STALL_CHUNK` | vitaslop-native/tests/retail_boot_probe.rs:544 | - |
| `VITASLOP_STALL_WAKE` | vitaslop-native/tests/retail_boot_probe.rs:543 | - |
| `VITASLOP_STALL_WATCHDOG` | vitaslop-native/src/watchdog.rs:99 | The configured stall budget in seconds, from `VITASLOP_STALL_WATCHDOG`. |
| `VITASLOP_STALL_WAVES` | vitaslop-native/tests/retail_boot_probe.rs:547 | - |
| `VITASLOP_STRICT_DRAWS` | vitaslop-runtime/src/render.rs:6789 | Why [`RenderSceneBuilder::build`] discarded draws from a captured scene. |
| `VITASLOP_SUBRECT_POOL` | vitaslop-platform/src/gpu.rs:1032 | The 2026-09-28 memory bounds, each with its arm back (`=0`): `VITASLOP_SUBRECT_POOL` |
| `VITASLOP_SWITCH_WHY` | vitaslop-transpiler/src/lower.rs:1225 | Whether the table-branch diagnostic is on for this address |
| `VITASLOP_SW_CHAIN` | vitaslop-native/src/wgpu_render.rs:614 | `VITASLOP_GPU_CHAIN_DIR=<dir>`: write every offscreen target of the frame just |
| `VITASLOP_SW_CHAIN_DIR` | vitaslop-runtime/src/render.rs:5330 | - |
| `VITASLOP_SW_POST` | vitaslop-runtime/src/render.rs:5377 | - |
| `VITASLOP_SYNC_RESOLVE_ASYNC` | vitaslop-runtime/src/host.rs:19199 | >>> A SYNC POINT'S RESOLVE, HANDED TO THE RESOLVER WORKER INSTEAD OF READ UNDER THE HOST |
| `VITASLOP_SYNC_RESOLVE_ON_WORKER` | vitaslop-web/src/browser_sched.rs:4489 | >>> A SYNC POINT'S RESOLVE-ONLY PARK, SETTLED ON THE GUEST'S OWN WORKER. |
| `VITASLOP_SYSTEM_FONT` | vitaslop-runtime/src/font/system.rs:75 | The resolved substitute: its bytes and a human-readable account of where they came from. |
| `VITASLOP_TEXENC_SHRINK_SCRATCH` | vitaslop-platform/src/texenc.rs:534 | >>> HOW BIG THE SHARED SOURCE BUFFER SHOULD BE for a texture needing `need_src` bytes, given |
| `VITASLOP_TEXTURE_CHECK` | vitaslop-runtime/src/host.rs:4138 | How a retained texture snapshot is re-validated (`VITASLOP_TEXTURE_CHECK`): `scene` |
| `VITASLOP_TEX_CACHE_MB` | vitaslop-platform/src/gpu.rs:666 | The texture-cache budget in bytes: [`GAME_RESIDENT_CEILING_MB`] unless |
| `VITASLOP_TEX_COMPRESS` | vitaslop-runtime/src/render.rs:1563 | Whether compressed textures reach the GPU compressed at all. |
| `VITASLOP_TEX_DIRTY_CENSUS` | vitaslop-runtime/src/host.rs:229 | >>> WHICH PARTS of `[off, off + len)` the guest may have stored into since `stamp`, |
| `VITASLOP_TEX_ENCODE_BUDGET_MS` | vitaslop-runtime/src/render.rs:1141 | >>> HOW MUCH OF ONE FRAME THE INLINE BLOCK ENCODE MAY SPEND, in milliseconds. |
| `VITASLOP_TEX_ENCODE_DEFER` | vitaslop-platform/src/gpu.rs:1134 | >>> A TEXTURE'S ETC2 ENCODE IS SPREAD OVER LATER FRAMES, AND THE FRAME THAT BINDS IT DRAWS |
| `VITASLOP_TEX_ENCODE_DEFER_UNITS` | vitaslop-platform/src/gpu.rs:1141 | Block WORK UNITS the deferred encoder spends per frame (`VITASLOP_TEX_ENCODE_DEFER_UNITS`): |
| `VITASLOP_TEX_ENCODE_RESUME` | vitaslop-runtime/src/render.rs:1270 | `VITASLOP_TEX_ENCODE_RESUME=0` restores the OLD behaviour: an encode that starts runs to |
| `VITASLOP_TEX_MEMO_PER_SCENE` | vitaslop-runtime/src/host.rs:3335 | A whole DRAW's worth of snapshotted textures, by the bindings that produced it - kept |
| `VITASLOP_TEX_PAGE_READ` | vitaslop-runtime/src/host.rs:4748 | Record that this entry's bytes are current as of THIS SCENE, so a later |
| `VITASLOP_TEX_RECENT_FRAMES` | vitaslop-platform/src/gpu.rs:750 | >>> A TEXTURE IS KEPT WHILE THE GUEST STILL HOLDS ITS BYTES - THE VITA HAS NO TEXTURE CACHE. |
| `VITASLOP_TEX_RETAIN_MB` | vitaslop-platform/src/gpu.rs:693 | How many bytes of GPU texture the view cache may RETAIN, as opposed to how many the |
| `VITASLOP_TRACE_BLOCKS` | vitaslop-native/src/threaded.rs:1490 | `VITASLOP_TRACE_FRAMES=<from>-<to>` (decimal display frames, inclusive) - print the |
| `VITASLOP_TRACE_EXIT` | vitaslop-native/tests/retail_boot_probe.rs:37 | - |
| `VITASLOP_TRACE_FILE` | vitaslop-runtime/src/vita/libkernel.rs:47 | Diagnostic (`RUST_LOG=vitaslop::exit=debug`): when the guest calls |
| `VITASLOP_TRACE_FRAMES` | vitaslop-native/src/threaded.rs:1487 | `VITASLOP_TRACE_FRAMES=<from>-<to>` (decimal display frames, inclusive) - print the |
| `VITASLOP_TRACE_FUNCS` | vitaslop-native/src/threaded.rs:1356 | Bind `env.svc`. |
| `VITASLOP_TRACE_INDIRECT` | vitaslop-transpiler/src/emit.rs:2026 | Diagnostic indirect-call tracer. |
| `VITASLOP_TRACE_IO` | vitaslop-native/tests/retail_boot_probe.rs:34 | - |
| `VITASLOP_TRACE_ORDER` | vitaslop-runtime/src/vita/mod.rs:882 | Ordered-timeline trace (env `VITASLOP_TRACE_ORDER`): print every *meaningful* |
| `VITASLOP_TRACE_ORDER_FULL` | vitaslop-runtime/src/vita/mod.rs:1253 | - |
| `VITASLOP_TRACK_PC` | vitaslop-transpiler/src/abi.rs:200 | Exported name of the diagnostic guest-PC tracker global. |
| `VITASLOP_TRANSPILE_REPORT` | vitaslop-native/src/threaded.rs:972 | - |
| `VITASLOP_TRAP_HALT` | vitaslop-transpiler/src/emit.rs:2154 | When `VITASLOP_TRAP_HALT` is set, a `Term::Halt` (a block that ran off the end of decoded |
| `VITASLOP_TS_DELAY_MS` | vitaslop-web/src/lib.rs:1282 | `VITASLOP_TS_DELAY_MS`: a test rig - see where it holds the poll in `present`. |
| `VITASLOP_UNIFORM_WATCH` | vitaslop-runtime/src/vita/gxm.rs:2327 | `VITASLOP_UNIFORM_WATCH=<hex address>/<parameter name substring>[,...]`: report every |
| `VITASLOP_UV_DEBUG` | vitaslop-runtime/src/render.rs:5539 | Draw one scene onto an EXISTING framebuffer and depth buffer, composing with whatever |
| `VITASLOP_VBLANK_PARK` | vitaslop-runtime/src/vita/display.rs:215 | Whether an inlined `sceDisplayGetVcount` carries the spin guard (`VITASLOP_VBLANK_PARK`, |
| `VITASLOP_VERTEX_INTERN` | vitaslop-runtime/src/host.rs:3662 | A cheap, allocation-free fingerprint of a vertex stream, for [`TextureSnapshots:: |
| `VITASLOP_VERTEX_INTERN_USE` | vitaslop-runtime/src/host.rs:3664 | A cheap, allocation-free fingerprint of a vertex stream, for [`TextureSnapshots:: |
| `VITASLOP_VPK` | vitaslop-runtime/src/ingest/stream.rs:1057 | A homebrew VPK (`VITASLOP_VPK=<file>`): probed as `vpk`, imported as files + a |
| `VITASLOP_WARM_BY_FORMAT` | vitaslop-platform/src/gpu.rs:1158 | Whether `warm_pipelines` predicts a never-encoded target's pipelines from its guest colour |
| `VITASLOP_WASM_INDICES` | vitaslop-native/src/threaded.rs:2700 | Rewrite `<wasm function N>` in a trap backtrace to name the GUEST function it is. |
| `VITASLOP_WASM_NAMES` | vitaslop-transpiler/src/emit.rs:1550 | When `VITASLOP_WASM_NAMES` is set, emit a wasm `name` custom section labelling |
| `VITASLOP_WATCH_` | vitaslop-transpiler/src/emit.rs:1938 | Number of matching store-watchpoint hits to skip before trapping (`VITASLOP_WATCH_ |
| `VITASLOP_WATCH_FROM` | vitaslop-native/tests/retail_boot_probe.rs:673 | - |
| `VITASLOP_WATCH_MEM` | vitaslop-native/tests/retail_boot_probe.rs:139 | Parse `VITASLOP_WATCH_MEM=addr:type:label,addr:type:label,...` into watches. |
| `VITASLOP_WATCH_READ` | vitaslop-transpiler/src/emit.rs:1488 | Diagnostic read watchpoint. |
| `VITASLOP_WATCH_READ_` | vitaslop-transpiler/src/emit.rs:1964 | Optional guest-PC EXCLUDE window for the read watchpoint (`VITASLOP_WATCH_READ_ |
| `VITASLOP_WATCH_READ_NZ` | vitaslop-transpiler/src/emit.rs:3888 | Emit the read-watchpoint trap check. |
| `VITASLOP_WATCH_READ_PC_EXCL` | vitaslop-transpiler/src/emit.rs:1974 | Optional guest-PC EXCLUDE window for the read watchpoint (`VITASLOP_WATCH_READ_ |
| `VITASLOP_WATCH_READ_SKIP` | vitaslop-transpiler/src/emit.rs:1668 | WASM global index of the read-watchpoint match counter, appended after the guest-PC |
| `VITASLOP_WATCH_STORE` | vitaslop-native/src/threaded.rs:1611 | `VITASLOP_REGTRACE_WATCH=<hex guest addr>[,<hex guest addr>...]` - append the WORD |
| `VITASLOP_WATCH_STORE_ARM` | vitaslop-transpiler/src/emit.rs:1280 | The emit-time knobs [`set_emit_knob`] accepts, so a caller that forwards a whole table |
| `VITASLOP_WATCH_STORE_LOG` | vitaslop-runtime/src/capture.rs:554 | GUEST ADDRESS the bytes above were read from, or 0 when there is no bound buffer. |
| `VITASLOP_WATCH_STORE_MODE` | vitaslop-transpiler/src/emit.rs:2169 | Store-watchpoint mode, from `VITASLOP_WATCH_STORE_MODE` (default `any`): |
| `VITASLOP_WATCH_STORE_NZ` | vitaslop-transpiler/src/emit.rs:2192 | `VITASLOP_WATCH_STORE_LOG` - LOG each store to the watched address (the storing |
| `VITASLOP_WATCH_STORE_SKIP` | vitaslop-transpiler/src/emit.rs:647 | Linear-memory byte offset of the store-watchpoint MATCH COUNTER, or 0 when this |
| `VITASLOP_WATCH_UNIFORM_REWRITES` | vitaslop-runtime/src/host.rs:24282 | `VITASLOP_WATCH_UNIFORM_REWRITES=1`: watch every SA uniform buffer a draw read for a guest |
| `VITASLOP_WHICH_EXPORT` | vitaslop-runtime/src/link.rs:301 | - |
| `VITASLOP_WINDOW_CENSUS` | vitaslop-runtime/src/host.rs:3878 | `VITASLOP_WINDOW_CENSUS=1`: measure the census's `slot-blank` count exactly even under the |
| `VITASLOP_WINDOW_REREAD` | vitaslop-runtime/src/host.rs:3828 | Which windows the guest's GPU wait re-reads - see [`window_wants_reread`]. |
| `VITASLOP_WINDOW_WAIT` | vitaslop-runtime/src/host.rs:3771 | `VITASLOP_WINDOW_WAIT=1`: the draw-time window capture waits for a busy snapshot lock |
| `VITASLOP_WRITEBACK_AT_FLIP` | vitaslop-web/src/lib.rs:2230 | >>> RENDER-TARGET WRITE-BACKS GO INTO GUEST MEMORY AT THE FLIP, NOT AFTER THE PRESENT. |
| `VITASLOP_X` | vitaslop-frontend/src/settings.rs:272 | Parse a `NAME=VALUE` per line knobs box into a map. |
| `VITASLOP_XML_DUMP` | vitaslop-runtime/src/vita/sce_xml.rs:580 | `VITASLOP_XML_DUMP=<dir>`: write every document handed to `parse` into `<dir>` as |
| `VITASLOP_Y` | vitaslop-frontend/src/settings.rs:273 | Parse a `NAME=VALUE` per line knobs box into a map. |
| `VITASLOP_YIELD_ELIDE` | vitaslop-runtime/src/host.rs:8155 | Whether the `sceKernelDelayThread(0)` yield elision is on - see |
