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

410 knobs.

| knob | read in | what it does |
|---|---|---|
| `VITASLOP_A` | vitaslop-frontend/src/settings.rs:283 | - |
| `VITASLOP_AAC_DIR` | vitaslop-aac/tests/oracle.rs:69 | The `AudioSpecificConfig` an ADTS header describes: 5 bits of object type, 4 of |
| `VITASLOP_ALLOW_SOFTWARE_GPU` | vitaslop-web/src/lib.rs:1837 | Whether a run may proceed on a software rasteriser (`VITASLOP_ALLOW_SOFTWARE_GPU`). |
| `VITASLOP_AMBIENT_PROBE` | vitaslop-runtime/src/host.rs:8731 | `VITASLOP_AMBIENT_PROBE=<lo>-<hi>` (decimal display frames, or `all`) - the window the |
| `VITASLOP_ARENA_UPLOAD_PROBE` | vitaslop-platform/src/gpu.rs:23857 | `VITASLOP_ARENA_UPLOAD_PROBE=1` - time the SAME bytes through the other upload path, a |
| `VITASLOP_ARM_AT_FRAME` | vitaslop-native/src/threaded.rs:342 | Linear-memory offset of the "diagnostics armed" word, when this build was |
| `VITASLOP_ARM_FUNC` | vitaslop-native/tests/homebrew_memcpy.rs:21 | Scratch window well past the code: `[SRC, SRC+WIN)` and `[DST, DST+WIN)`. |
| `VITASLOP_AT9_DIR` | vitaslop-atrac9/tests/oracle.rs:66 | Decode a whole AT9 payload the way a superframe consumer does: for each |
| `VITASLOP_AUDIO_RAW` | vitaslop-runtime/src/vita/audio.rs:92 | Optional raw-s16le capture of the mixed output stream (env |
| `VITASLOP_B` | vitaslop-frontend/src/settings.rs:283 | - |
| `VITASLOP_BACKTRACE` | vitaslop-runtime/src/vita/mod.rs:638 | Print the guest call chain the first time a chosen NID is called from each thread |
| `VITASLOP_BLOCK_HIST` | vitaslop-native/src/recipe_runner.rs:167 | Dump the per-PC block-entry histogram gathered under `VITASLOP_BLOCK_HIST`, for |
| `VITASLOP_BLOCK_HIST_SEQ` | vitaslop-native/src/threaded.rs:1844 | Print the block-visit histogram gathered under `VITASLOP_BLOCK_HIST`: the `top` |
| `VITASLOP_BROWSER_FASTFORWARD` | vitaslop-web/src/lib.rs:1800 | Frame to fast-forward the live loop to (`VITASLOP_BROWSER_FASTFORWARD`), unpaced. |
| `VITASLOP_BROWSER_FUEL` | vitaslop-web/src/browser_sched.rs:986 | Guest work a thread may execute before the browser preempts it, in WASMTIME FUEL UNITS |
| `VITASLOP_BROWSER_HEARTBEAT_MS` | vitaslop-web/src/lib.rs:4672 | - |
| `VITASLOP_BROWSER_INSTANCE_POOL` | vitaslop-web/src/browser_sched.rs:2081 | Whether a finished thread's module instance may be REUSED by the next thread |
| `VITASLOP_BROWSER_QUANTUM_CALLS` | vitaslop-web/src/browser_sched.rs:130 | Host calls one guest thread may make before the browser preempts it |
| `VITASLOP_BROWSER_RENDER_FROM` | vitaslop-web/src/lib.rs:1820 | Frame from which a FAST-FORWARD still RENDERS (`VITASLOP_BROWSER_RENDER_FROM`), even though |
| `VITASLOP_BROWSER_SPLIT_MEMORY` | vitaslop-web/src/browser_sched.rs:1271 | >>> THE GUEST REGION INSIDE THIS MODULE'S OWN LINEAR MEMORY. |
| `VITASLOP_BROWSER_SUPERSAMPLE` | vitaslop-web/src/lib.rs:1777 | Supersample factor for the live browser render (`VITASLOP_BROWSER_SUPERSAMPLE`). |
| `VITASLOP_BUILD_FASTPATH` | vitaslop-runtime/src/render.rs:6401 | `VITASLOP_BUILD_FASTPATH=0`: the NEGATIVE CONTROL arm for what `build` stopped doing. |
| `VITASLOP_CALLSITES_WINDOW` | vitaslop-desktop/src/retail.rs:554 | Where the idle clock went SINCE `before` - the windowed reading. |
| `VITASLOP_CAPSULE_DUMP_PROGS` | vitaslop-native/examples/capsule-replay.rs:576 | - |
| `VITASLOP_CAPSULE_DUMP_SA` | vitaslop-native/examples/capsule-replay.rs:29 | `VITASLOP_CAPSULE_DUMP_SA=1`: print this draw's uniform banks - `frag_sa` with the GUEST |
| `VITASLOP_CAPSULE_DUMP_VERTS` | vitaslop-native/examples/capsule-replay.rs:485 | - |
| `VITASLOP_CAPSULE_EXTENT` | vitaslop-native/examples/capsule-replay.rs:681 | - |
| `VITASLOP_CAPSULE_TEX_DIR` | vitaslop-native/examples/capsule-replay.rs:386 | - |
| `VITASLOP_CHAIN_DRAWS` | vitaslop-platform/src/gpu.rs:22481 | - |
| `VITASLOP_CHAIN_LIMIT` | vitaslop-native/tests/gpu_rtt_gamma.rs:178 | Render a chain of `feedback` sample-and-write-back passes over the offscreen target and |
| `VITASLOP_CHAIN_SKIP` | vitaslop-native/src/wgpu_render.rs:346 | - |
| `VITASLOP_CHECK_ADDRS` | vitaslop-native/tests/retail_boot_probe.rs:43 | - |
| `VITASLOP_CLOCK_TRACE` | vitaslop-runtime/src/sched.rs:1169 | - |
| `VITASLOP_CODE_RANGE` | vitaslop-runtime/src/vita/mod.rs:621 | The guest code range scanned for the game-level caller in [`dispatch`] (env |
| `VITASLOP_COMPACT_SPARSE` | vitaslop-runtime/src/host.rs:3330 | `VITASLOP_COMPACT_SPARSE=0` is the NEGATIVE CONTROL for the compaction: the `min..=max` |
| `VITASLOP_CONSOLE` | vitaslop-web/src/logging.rs:239 | `VITASLOP_CONSOLE=1`: mirror the run's status notes - the setup summary, the adapter and |
| `VITASLOP_CPU_SHARE` | vitaslop-native/src/recipe_runner.rs:111 | Who actually got the CPU over the run, when `VITASLOP_CPU_SHARE` is set - see |
| `VITASLOP_CPU_SHARE_FROM` | vitaslop-runtime/src/sched.rs:1572 | `VITASLOP_CPU_SHARE_FROM=<frame>` - see the reset in `SchedCore::on_suspended`. |
| `VITASLOP_DBG_CALLSITES` | vitaslop-runtime/src/vita/mod.rs:579 | Diagnostic call-site profiler (`VITASLOP_DBG_CALLSITES`): counts host calls |
| `VITASLOP_DEBUG_CAPTURE` | vitaslop-frontend/src/settings.rs:187 | The `VITASLOP_*` map a run of these settings is configured with: the base set, |
| `VITASLOP_DECODE_CACHE_MB` | vitaslop-runtime/src/render.rs:6943 | Budget for the decode cache, in BYTES of decoded RGBA8, before it is cleared wholesale. |
| `VITASLOP_DEFER_GEOMETRY` | vitaslop-runtime/src/host.rs:22664 | Whether a draw's VERTEX AND INDEX BYTES are read at `sceGxmEndScene` rather than at the |
| `VITASLOP_DEFER_WINDOW_BYTES` | vitaslop-runtime/src/host.rs:3410 | `VITASLOP_GXP_ATTR_ROW=0` - the negative control for laying the interleaved row out per |
| `VITASLOP_DELAY_CENSUS` | vitaslop-runtime/src/vita/threadmgr.rs:200 | Diagnostic (`VITASLOP_DELAY_CENSUS=1`): every `sceKernelDelayThread` tallied by (call site, |
| `VITASLOP_DEVICE_BUDGET` | vitaslop-desktop/src/retail.rs:1021 | Headless self-check of the retail path (NO window): load `dir`, optionally drive a |
| `VITASLOP_DIRTY_PAGES` | vitaslop-native/src/threaded.rs:128 | Linear-memory offset of the guest-store dirty block, when this build was |
| `VITASLOP_DIRTY_RUN_MARK` | vitaslop-transpiler/src/emit.rs:847 | Turn the coalesced run-mark on or off for modules emitted on this thread after this |
| `VITASLOP_DISPATCH_ALL` | vitaslop-transpiler/src/emit.rs:1700 | The ablation arm that prices a dispatch re-entry: `VITASLOP_DISPATCH_ALL=1` sends even a |
| `VITASLOP_DRAW_ONLY` | vitaslop-runtime/src/render.rs:5473 | - |
| `VITASLOP_DRAW_RANGE` | vitaslop-platform/src/gpu.rs:755 | `VITASLOP_RTT_BG_CACHE=0` restores the OLD behaviour: a sampler bind group naming a render |
| `VITASLOP_DRAW_STATS` | vitaslop-runtime/src/render.rs:5462 | Draw one scene onto an EXISTING framebuffer and depth buffer, composing with whatever |
| `VITASLOP_DRV_KEY` | vitaslop-runtime/src/ingest/pfscrypt.rs:344 | `F00D(klicensee)` for the title, from `VITASLOP_DRV_KEY` (32 hex chars), or |
| `VITASLOP_DUMP_DIR` | vitaslop-runtime/src/ingest/pipeline.rs:824 | Diagnostic: decrypt the container and write named plaintext files out to |
| `VITASLOP_DUMP_DRAW` | vitaslop-native/tests/retail_boot_probe.rs:1286 | - |
| `VITASLOP_DUMP_DRAWS` | vitaslop-native/tests/retail_boot_probe.rs:1075 | - |
| `VITASLOP_DUMP_DRAW_GXP` | vitaslop-gxp-shader/tests/oracle.rs:653 | Correlate each captured vertex<->fragment PAIR (from a real draw run) to establish the |
| `VITASLOP_DUMP_DRAW_GXP_CAP` | vitaslop-runtime/src/host.rs:18146 | Whether `VITASLOP_DUMP_DRAW_GXP` names display frame `disp`: "all", a single frame |
| `VITASLOP_DUMP_DRAW_GXP_FULL` | vitaslop-runtime/src/host.rs:18436 | - |
| `VITASLOP_DUMP_EXPORTS` | vitaslop-runtime/src/link.rs:611 | - |
| `VITASLOP_DUMP_FILES` | vitaslop-runtime/src/ingest/pipeline.rs:826 | Diagnostic: decrypt the container and write named plaintext files out to |
| `VITASLOP_DUMP_FPROG` | vitaslop-runtime/src/host.rs:17731 | Diagnostic (VITASLOP_DUMP_FPROG): print the bound fragment program's sampler |
| `VITASLOP_DUMP_FUNC` | vitaslop-native/tests/retail_boot_probe.rs:317 | - |
| `VITASLOP_DUMP_GXP_BIN` | vitaslop-platform/src/gpu.rs:16277 | >>> A REFUSAL THAT DOES NOT HAND OVER THE EVIDENCE COSTS A PLAY SESSION. |
| `VITASLOP_DUMP_IMAGE` | vitaslop-native/tests/retail_boot_probe.rs:33 | - |
| `VITASLOP_DUMP_IMPORTS` | vitaslop-native/tests/retail_boot_probe.rs:353 | - |
| `VITASLOP_DUMP_IR` | vitaslop-native/tests/homebrew_memcpy.rs:40 | Scratch window well past the code: `[SRC, SRC+WIN)` and `[DST, DST+WIN)`. |
| `VITASLOP_DUMP_MAP` | vitaslop-native/tests/retail_boot_probe.rs:589 | - |
| `VITASLOP_DUMP_MEM` | vitaslop-native/tests/retail_boot_probe.rs:43 | - |
| `VITASLOP_DUMP_PATHS` | vitaslop-native/tests/retail_boot_probe.rs:412 | - |
| `VITASLOP_DUMP_REGION` | vitaslop-native/tests/retail_boot_probe.rs:527 | - |
| `VITASLOP_DUMP_REGION_RANGE` | vitaslop-native/tests/retail_boot_probe.rs:529 | - |
| `VITASLOP_DUMP_RENDERSCENE` | vitaslop-native/tests/retail_boot_probe.rs:1327 | - |
| `VITASLOP_DUMP_SCENES` | vitaslop-desktop/src/retail.rs:423 | Step the guest one display frame. |
| `VITASLOP_DUMP_STDOUT` | vitaslop-desktop/src/retail.rs:2024 | - |
| `VITASLOP_DUMP_STREAM_BYTES` | vitaslop-runtime/src/host.rs:22618 | `VITASLOP_DUMP_STREAM_BYTES[=<n>]`: with the per-draw dump on, print each vertex stream's |
| `VITASLOP_DUMP_STUBS` | vitaslop-native/tests/retail_boot_probe.rs:32 | - |
| `VITASLOP_DUMP_TEX` | vitaslop-native/tests/retail_boot_probe.rs:1229 | - |
| `VITASLOP_DUMP_TEX_DIR` | vitaslop-runtime/src/host.rs:18442 | - |
| `VITASLOP_DUMP_TEX_MAX_TEXELS` | vitaslop-runtime/src/host.rs:18451 | - |
| `VITASLOP_DUMP_TRIS` | vitaslop-runtime/src/render.rs:5480 | - |
| `VITASLOP_DUMP_VPROG` | vitaslop-runtime/src/host.rs:17489 | Diagnostic (VITASLOP_DUMP_VPROG): reflect the bound vertex program's parameter |
| `VITASLOP_DUMP_VUBUF` | vitaslop-runtime/src/host.rs:18252 | - |
| `VITASLOP_EAGER_FILES` | vitaslop-desktop/src/retail.rs:241 | Load, decrypt, link, transpile, and instantiate the title in `dir` for live |
| `VITASLOP_FAST_IMPORT_CURATED` | vitaslop-runtime/src/vita/mod.rs:190 | Whether `func_nid`'s handler can only ever CONTINUE, so the transpiler may route the |
| `VITASLOP_FIND_WORD` | vitaslop-native/tests/retail_boot_probe.rs:63 | The span `VITASLOP_FIND_WORD` searches: from the image base up through the guest heap. |
| `VITASLOP_FLAGS_WIDE_C` | vitaslop-transpiler/src/emit.rs:1673 | The A/B arm for [`emit_flags_add`]'s carry and overflow forms: `VITASLOP_FLAGS_WIDE_C=1` |
| `VITASLOP_FLAG_POISON` | vitaslop-transpiler/src/emit.rs:1631 | `VITASLOP_FLAG_POISON=0/1` - the FALSIFIER for the flag-liveness pass |
| `VITASLOP_FORCE_READY` | vitaslop-native/tests/retail_boot_probe.rs:735 | - |
| `VITASLOP_FORCE_READY_V2` | vitaslop-native/tests/retail_boot_probe.rs:758 | - |
| `VITASLOP_FORCE_RET` | vitaslop-transpiler/src/emit.rs:2006 | Diagnostic forced return. |
| `VITASLOP_FRAME_DIGEST` | vitaslop-native/src/recipe_runner.rs:352 | - |
| `VITASLOP_FRAME_TOPUP` | vitaslop-runtime/src/host.rs:7878 | The per-flip top-up ([`VitaState::advance_time_frame`]), which is OPT-IN: |
| `VITASLOP_FUEL` | vitaslop-native/src/threaded.rs:147 | This thread's SOFTWARE fuel counter (`abi::FUEL_EXPORT`), present only when the |
| `VITASLOP_GAME_DIR` | vitaslop-runtime/src/ingest/mod.rs:140 | Test-fixture access. |
| `VITASLOP_GAME_ID` | vitaslop-gamerun-recipes/tests/conformance.rs:30 | - |
| `VITASLOP_GAME_PKG` | vitaslop-runtime/src/ingest/pipeline.rs:520 | Diagnostic: dump the pkg header and the extracted file tree so a new |
| `VITASLOP_GAME_WORK` | vitaslop-runtime/src/ingest/pipeline.rs:697 | The pkg + work.bin chain over a privately-supplied two-file dump: extract |
| `VITASLOP_GAME_ZIP` | vitaslop-runtime/src/ingest/mod.rs:120 | - |
| `VITASLOP_GAP_CAP` | vitaslop-native/tests/retail_boot_probe.rs:266 | - |
| `VITASLOP_GESTURE_EVENT_KIND` | vitaslop-runtime/src/vita/gesture.rs:814 | `VITASLOP_GESTURE_EVENT_KIND`: write this byte at [`EVENT_KIND_OFF`]. |
| `VITASLOP_GESTURE_EVENT_STATE` | vitaslop-runtime/src/vita/gesture.rs:395 | The bits written into [`EVENT_STATE_OFF`]. |
| `VITASLOP_GESTURE_PRIMITIVE_STATE` | vitaslop-runtime/src/vita/gesture.rs:796 | `VITASLOP_GESTURE_PRIMITIVE_STATE`: write this halfword at [`PRIMITIVE_STATE_OFF`]. |
| `VITASLOP_GESTURE_TAP_ON_RELEASE` | vitaslop-runtime/src/vita/gesture.rs:862 | `VITASLOP_GESTURE_TAP_ON_RELEASE`: report a type-1 recognizer's event on the frame the |
| `VITASLOP_GESTURE_TYPE_MASK` | vitaslop-runtime/src/vita/gesture.rs:414 | Recognizer types allowed to report events (`VITASLOP_GESTURE_TYPE_MASK`, a bitmask |
| `VITASLOP_GPU` | vitaslop-native/tests/retail_boot_probe.rs:1390 | - |
| `VITASLOP_GPU_CHAIN_DIR` | vitaslop-native/src/wgpu_render.rs:579 | `VITASLOP_GPU_CHAIN_DIR=<dir>`: write every offscreen target of the frame just |
| `VITASLOP_GPU_QUEUE_DEPTH` | vitaslop-web/src/lib.rs:920 | How many submits may be in flight before a present declines to make another. |
| `VITASLOP_GUARD_REG` | vitaslop-transpiler/src/emit.rs:1860 | Diagnostic callee-saved-register guard. |
| `VITASLOP_GUEST_CORES` | vitaslop-runtime/src/host.rs:20871 | CPU cores a Vita gives a GAME. |
| `VITASLOP_GXM` | vitaslop-native/examples/capsule-replay.rs:595 | - |
| `VITASLOP_GXM_ARENA_FLOOR_KB` | vitaslop-platform/src/gpu.rs:844 | `VITASLOP_GXM_DEST_SPLIT_AB=<n>`: alternate the destination-colour pass SPLIT on and off every |
| `VITASLOP_GXM_ARENA_POOL` | vitaslop-platform/src/gpu.rs:13724 | `VITASLOP_GXM_ARENA_POOL=1` pools the six per-pass staging arenas. |
| `VITASLOP_GXM_ARENA_REPEAT` | vitaslop-platform/src/gpu.rs:3989 | >>> REPACKS OF GEOMETRY THIS RUN HAD ALREADY EVICTED - the cache THRASHING. |
| `VITASLOP_GXM_BUFFER_PREINIT` | vitaslop-platform/src/gpu.rs:18243 | Create through the pool: a hit is free, a miss is the allocation that was going to |
| `VITASLOP_GXM_DEPTH_ENC` | vitaslop-platform/src/gpu.rs:3422 | Which value a later pass reads out of a render target's depth |
| `VITASLOP_GXM_DEST_SPLIT_AB` | vitaslop-platform/src/gpu.rs:819 | `VITASLOP_GXM_DEST_SPLIT_AB=<n>`: alternate the destination-colour pass SPLIT on and off every |
| `VITASLOP_GXM_DRAW_COVERAGE` | vitaslop-gxp-shader/src/link.rs:4589 | `1` - WHERE DID THIS DRAW'S VERTICES GO? Draw the mesh at CLAMPED normalised coordinates and |
| `VITASLOP_GXM_DRAW_PROBE` | vitaslop-platform/src/gpu.rs:3482 | >>> TEMPORARY TELEMETRY (`VITASLOP_GXM_DRAW_PROBE=<keyspec>`). |
| `VITASLOP_GXM_NO_MULTISAMPLE` | vitaslop-platform/src/gpu.rs:7927 | A/B instrument: force every pass to ONE sample, whatever the guest asked for. |
| `VITASLOP_GXM_PVS_BULK_TABLE` | vitaslop-runtime/src/vita/gxm.rs:728 | `VITASLOP_GXM_PVS_BULK_TABLE=1` - restore the VERTEX bind's old WHOLESALE table copy, zeros |
| `VITASLOP_GXM_RTT_CLEAR_EVERY_FRAME` | vitaslop-platform/src/gpu.rs:23829 | `VITASLOP_GXM_RTT_CLEAR_EVERY_FRAME=1` - restore the pre-2026-09-12 behaviour, in which the |
| `VITASLOP_GXM_RTT_WRITEBACK` | vitaslop-native/src/wgpu_render.rs:483 | The rendered pixels of every offscreen target small enough to hand back to the GUEST, |
| `VITASLOP_GXM_STAGING` | vitaslop-platform/src/gpu.rs:970 | Whether the arena upload goes through the STAGING BELT (default) rather than |
| `VITASLOP_GXM_STALE_UNIFORMS` | vitaslop-desktop/src/retail.rs:1035 | Headless self-check of the retail path (NO window): load `dir`, optionally drive a |
| `VITASLOP_GXM_STATE_DEDUP` | vitaslop-platform/src/gpu.rs:7843 | What a render pass already has bound, so a draw that changes nothing rebinds nothing. |
| `VITASLOP_GXM_SWEEP_EVERY` | vitaslop-platform/src/gpu.rs:23880 | `VITASLOP_GXM_SWEEP_EVERY=<n>` - how many frames apart the packed-geometry budget check and |
| `VITASLOP_GXM_TEX_UNWRITTEN` | vitaslop-runtime/src/render.rs:6905 | `VITASLOP_GXM_TEX_UNWRITTEN=0` - the arm back for reading ANY uniform 4-byte fill as |
| `VITASLOP_GXM_UNIFORM_POISON` | vitaslop-gxp-shader/src/module.rs:371 | `:bits=<hex>` - paint a lane 1.0 when that register's RAW BITS equal this word, 0.0 |
| `VITASLOP_GXM_VIEWPORT` | vitaslop-platform/src/gpu.rs:2235 | `VITASLOP_GXM_VIEWPORT=0` - the NEGATIVE CONTROL for the guest viewport reaching the |
| `VITASLOP_GXP` | vitaslop-native/examples/capsule-replay.rs:595 | - |
| `VITASLOP_GXP_` | vitaslop-native/examples/capsule-replay.rs:14 | - |
| `VITASLOP_GXP_ALIGN_ROW` | vitaslop-runtime/src/host.rs:2519 | ONE COPY the gather makes: `len` bytes from byte `src` of stream `stream`'s row into byte |
| `VITASLOP_GXP_ALLOW_FIXED_FUNCTION` | vitaslop-platform/src/gpu.rs:16344 | >>> A PAIR THIS RECOMPILER CANNOT TRANSLATE DROPS ITS DRAWS. |
| `VITASLOP_GXP_ARENA_RING` | vitaslop-platform/src/gpu.rs:981 | `VITASLOP_GXP_ARENA_RING=<n>`: how many COPIES of each render pass's recompiled-path |
| `VITASLOP_GXP_ATTR_ALL` | vitaslop-gxp-shader/tests/corpus.rs:4973 | What [`vitaslop_gxp_shader::attrflow`] decides for every attribute of every pair that links, |
| `VITASLOP_GXP_ATTR_FILL` | vitaslop-gxp-shader/src/module.rs:2089 | Bytes the driver adds to the DEFAULT uniform buffer's bound address before writing it into |
| `VITASLOP_GXP_ATTR_ROW` | vitaslop-runtime/src/host.rs:3408 | `VITASLOP_GXP_ATTR_ROW=0` - the negative control for laying the interleaved row out per |
| `VITASLOP_GXP_BIND_TRACE` | vitaslop-platform/src/gpu.rs:767 | Diagnostic (`VITASLOP_GXP_BIND_TRACE=1`): say, once per pair and plan shape, WHICH view each |
| `VITASLOP_GXP_BLOB` | vitaslop-gxp-shader/tests/corpus.rs:689 | Print one named blob's recompiled WGSL body and its container reflection. |
| `VITASLOP_GXP_CAPSULE` | vitaslop-runtime/src/capsule.rs:653 | Diagnostic (`VITASLOP_GXP_CAPSULE=<vprog-hash>[,<vprog-hash>]:<dir>[:N]`): write the first |
| `VITASLOP_GXP_CAPSULE_MIN_INDICES` | vitaslop-runtime/src/capsule.rs:666 | Diagnostic (`VITASLOP_GXP_CAPSULE=<vprog-hash>[,<vprog-hash>]:<dir>[:N]`): write the first |
| `VITASLOP_GXP_CAPSULE_SKIP` | vitaslop-runtime/src/capsule.rs:686 | Diagnostic (`VITASLOP_GXP_CAPSULE_SKIP=<n>`): ignore the first `n` matching submissions |
| `VITASLOP_GXP_CASES_OUT` | vitaslop-gxp-shader/tests/conformance.rs:2860 | Write every case out for the GPU runner, with the INTENT as the expectation. |
| `VITASLOP_GXP_CASE_TEX` | vitaslop-gxp-shader/src/wgsl.rs:2111 | >>> DOES THE STAND-IN TEXTURE VARY WITH THE COORDINATE? |
| `VITASLOP_GXP_CLAIM_WIDTH` | vitaslop-gxp-shader/src/link.rs:1491 | Re-order a convention-placed layout so that every forwarded attribute's usage STARTS at the |
| `VITASLOP_GXP_CLIP_DRAWS` | vitaslop-platform/src/gpu.rs:3497 | >>> TEMPORARY TELEMETRY (`VITASLOP_GXP_CLIP_DRAWS=<keyspec>`). |
| `VITASLOP_GXP_CLIP_DUMP_SA` | vitaslop-platform/src/gpu.rs:14786 | - |
| `VITASLOP_GXP_CLIP_RETRY_EVERY_FRAME` | vitaslop-platform/src/gpu.rs:23895 | `VITASLOP_GXP_CLIP_RETRY_EVERY_FRAME=1` - re-measure a pair whose clip measurement found no |
| `VITASLOP_GXP_CMOVU8` | vitaslop-gxp-shader/src/module.rs:654 | Whether [`crate::wgsl`]'s byte-wise conditional move tests EACH BYTE of its test operand |
| `VITASLOP_GXP_CORPUS` | vitaslop-gxp-shader/tests/corpus.rs:5782 | Price the F16 emulation in CONVERSIONS PER FRAME, by weighing each pair's emitted count with |
| `VITASLOP_GXP_COVERAGE_LOG` | vitaslop-gxp-shader/tests/corpus.rs:5779 | Price the F16 emulation in CONVERSIONS PER FRAME, by weighing each pair's emitted count with |
| `VITASLOP_GXP_CULL` | vitaslop-platform/src/gpu.rs:1151 | `VITASLOP_GXP_CULL=0` restores the pre-2026-08-19b "draw both windings". |
| `VITASLOP_GXP_DEBUG` | vitaslop-platform/src/gpu.rs:17122 | - |
| `VITASLOP_GXP_DEFAULT_UNIFORM_OFFSET` | vitaslop-gxp-shader/src/module.rs:2086 | Bytes the driver adds to the DEFAULT uniform buffer's bound address before writing it into |
| `VITASLOP_GXP_DEPTH_PROBE` | vitaslop-gxp-shader/src/link.rs:3537 | Turn each stage's SA-bank marker into the reads the body actually needs. |
| `VITASLOP_GXP_DEST` | vitaslop-gxp-shader/src/module.rs:707 | Whether a fragment program that reads the destination colour is DECLARED as reading it. |
| `VITASLOP_GXP_DEST_BLEND` | vitaslop-gxp-shader/src/module.rs:603 | Whether [`lower_dest_blend`] is allowed to run. |
| `VITASLOP_GXP_DEST_POISON` | vitaslop-gxp-shader/src/module.rs:1701 | `VITASLOP_GXP_DEST_POISON=<r,g,b,a>` - the constant [`dest_color_init`] seeds the output |
| `VITASLOP_GXP_DEST_PROBE` | vitaslop-gxp-shader/src/link.rs:3537 | Turn each stage's SA-bank marker into the reads the body actually needs. |
| `VITASLOP_GXP_DISASM` | vitaslop-gxp-shader/tests/oracle.rs:860 | Compact disassembly of one blob (named by `VITASLOP_GXP_DISASM`, matched as a filename |
| `VITASLOP_GXP_DUAL_SOURCE` | vitaslop-gxp-shader/src/module.rs:724 | Whether a destination-reading program that is LINEAR in the destination may be lowered to a |
| `VITASLOP_GXP_DUAL_TRACE` | vitaslop-gxp-shader/tests/corpus.rs:6349 | Print the DUAL-SOURCE plan (or the reason there is none) for every destination reader in |
| `VITASLOP_GXP_DUMP` | vitaslop-platform/src/gpu.rs:8268 | Diagnostic (`VITASLOP_GXP_KEYS=<hex>,<hex>`): recompile ONLY these shader-pair keys |
| `VITASLOP_GXP_DUMPS` | vitaslop-gxp-shader/tests/oracle.rs:145 | Histogram the raw values of named fields across every instruction of a given opcode1 |
| `VITASLOP_GXP_ELIDE_NOOP` | vitaslop-platform/src/gpu.rs:777 | `VITASLOP_GXP_ELIDE_NOOP=0` - the negative control for dropping the colour work of a draw |
| `VITASLOP_GXP_EXCLUDE` | vitaslop-platform/src/gpu.rs:8272 | Pairs forced down the fixed-function path (`VITASLOP_GXP_EXCLUDE`). |
| `VITASLOP_GXP_F16_RTE` | vitaslop-gxp-shader/src/link.rs:4565 | >>> HOW AN f32 NARROWS TO AN f16 - THE NEGATIVE CONTROL FOR THE ROUNDING FIX. |
| `VITASLOP_GXP_FMEM` | vitaslop-platform/src/gpu.rs:11999 | Diagnostic (`VITASLOP_GXP_FMEM=<lane>=<value>` / `<lane>*<factor>`, comma separated): |
| `VITASLOP_GXP_FORCE` | vitaslop-platform/src/gpu.rs:8239 | Diagnostic (`VITASLOP_GXP_FORCE`): bind a neutral fallback texture for a sampler |
| `VITASLOP_GXP_FRAG` | vitaslop-gxp-shader/tests/corpus.rs:5025 | The complete LINKED WGSL for one pair, selected by `VITASLOP_GXP_VERT` + `VITASLOP_GXP_FRAG` |
| `VITASLOP_GXP_GROUP` | vitaslop-gxp-shader/tests/corpus.rs:3329 | Every distinct word of one opcode group across the corpus, with the programs it appears in. |
| `VITASLOP_GXP_GUEST_ATTRS` | vitaslop-platform/src/gpu.rs:13589 | `VITASLOP_GXP_GUEST_ATTRS=0` is the negative control for TELLING THE LINK WHAT THE GUEST |
| `VITASLOP_GXP_HALF_REGS` | vitaslop-gxp-shader/src/link.rs:3665 | Give every register the program only ever uses as a PACKED F16 PAIR an UNPACKED home, so |
| `VITASLOP_GXP_IDX_MUL` | vitaslop-gxp-shader/src/usse/mod.rs:678 | Fill in each ORDINARY-REGISTER index load's `stride` - how far apart two consecutive index |
| `VITASLOP_GXP_IDX_REGDEST` | vitaslop-gxp-shader/src/link.rs:4639 | `0` sends EVERY group-0x14 index load to the index register, which is what this decoder did |
| `VITASLOP_GXP_IDX_REPEAT` | vitaslop-gxp-shader/src/usse/decode.rs:4232 | Whether a load-index word's bits 46:44 repeat it (`VITASLOP_GXP_IDX_REPEAT=0` is the arm |
| `VITASLOP_GXP_IDX_SCALE` | vitaslop-gxp-shader/src/module.rs:665 | How many REGISTERS one count of an index register spans - see [`crate::wgsl`]'s |
| `VITASLOP_GXP_INDEX16` | vitaslop-runtime/src/render.rs:1278 | `VITASLOP_GXP_INDEX16=0` restores the OLD behaviour: every index widened to u32 whatever |
| `VITASLOP_GXP_INPUTS` | vitaslop-gxp-shader/src/usse/mod.rs:659 | Fill in each ORDINARY-REGISTER index load's `stride` - how far apart two consecutive index |
| `VITASLOP_GXP_INPUTS_DIR` | vitaslop-platform/src/gpu.rs:3578 | Whether the once-per-pair `gxp pair <key>: vprog hash ..., fprog hash ...` INDEX should be |
| `VITASLOP_GXP_INPUTS_ORDER` | vitaslop-platform/src/gpu.rs:145 | The output of a diagnostic whose own KNOB is already the gate. |
| `VITASLOP_GXP_INPUTS_SETS` | vitaslop-platform/src/gpu.rs:12258 | - |
| `VITASLOP_GXP_INPUTS_VERTS` | vitaslop-platform/src/gpu.rs:12114 | Diagnostic (`VITASLOP_GXP_INPUTS=<hex-key>[,<hex-key>]` or `=all`): print, ONCE per |
| `VITASLOP_GXP_INTERP` | vitaslop-platform/src/gpu.rs:17561 | - |
| `VITASLOP_GXP_INT_PAIRS` | vitaslop-gxp-shader/tests/corpus.rs:864 | >>> EVERY CORPUS PAIR, RE-LINKED WITH EVERY ATTRIBUTE DECLARED A PLAIN INTEGER, VALIDATES. |
| `VITASLOP_GXP_KEYCOLOR` | vitaslop-platform/src/gpu.rs:3587 | Whether the once-per-pair `gxp pair <key>: vprog hash ..., fprog hash ...` INDEX should be |
| `VITASLOP_GXP_KEYS` | vitaslop-platform/src/gpu.rs:8267 | Diagnostic (`VITASLOP_GXP_KEYS=<hex>,<hex>`): recompile ONLY these shader-pair keys |
| `VITASLOP_GXP_LIVE` | vitaslop-platform/src/gpu.rs:1984 | The guest's real vertex+fragment shaders + their draw inputs, for the GXP->WGSL |
| `VITASLOP_GXP_MAD_MASK16` | vitaslop-gxp-shader/src/usse/decode.rs:1707 | The mad-group destination write mask: a BITMASK over the destination's REGISTER LANES. |
| `VITASLOP_GXP_MEM_OFFSET16` | vitaslop-gxp-shader/tests/conformance.rs:1395 | **A MEMORY LOAD READS CONSECUTIVE GUEST WORDS FROM `pointer + offsets`, AN IMMEDIATE OFFSET |
| `VITASLOP_GXP_MEM_PEEK` | vitaslop-runtime/src/host.rs:16804 | `VITASLOP_MEM_DUMP=<hex addr>:<bytes>[,...]`: write those guest byte ranges to |
| `VITASLOP_GXP_MIPS` | vitaslop-platform/src/gpu.rs:3858 | Whether a chain is built for a seam, ignoring the per-texture exception above. |
| `VITASLOP_GXP_NEGW` | vitaslop-platform/src/gpu.rs:8423 | How to choose the clip-`w` sign correction (`VITASLOP_GXP_NEGW`). |
| `VITASLOP_GXP_NOBLEND` | vitaslop-platform/src/gpu.rs:8257 | Diagnostic (`VITASLOP_GXP_NOBLEND`): force every recompiled pipeline to REPLACE with |
| `VITASLOP_GXP_NODEPTH` | vitaslop-platform/src/gpu.rs:8249 | Diagnostic (`VITASLOP_GXP_NODEPTH`): every recompiled draw keeps its real shading and |
| `VITASLOP_GXP_NO_STENCIL` | vitaslop-conformance-harness/tests/vita_gxmconf.rs:552 | **SCENE 7 - A STENCIL MASK CONFINES A LATER DRAW TO WHAT AN INVISIBLE DRAW MARKED.** |
| `VITASLOP_GXP_ONLY` | vitaslop-platform/src/gpu.rs:8229 | Render ONLY recompiled draws, skipping the fixed-function draw for any call that |
| `VITASLOP_GXP_PACK_COMP0` | vitaslop-gxp-shader/src/link.rs:4645 | `0` restores bit 1 as comp0's high selector bit for a 16-bit PACK source, which is what this |
| `VITASLOP_GXP_PACK_DEST_SLOT` | vitaslop-gxp-shader/src/usse/decode.rs:4209 | The SMLSI slot a repeating 0x40 VPCK's DESTINATION steps under: slot 0, the DEST byte |
| `VITASLOP_GXP_PACK_INTERNAL` | vitaslop-gxp-shader/src/usse/decode.rs:4195 | Where each operand of `word` sits for repeat purposes: the destination, then each source in |
| `VITASLOP_GXP_PAIR` | vitaslop-gxp-shader/tests/corpus.rs:1014 | Link one named (vertex, fragment) pair and print the COMPLETE WGSL module both stages become. |
| `VITASLOP_GXP_PAIRS` | vitaslop-gxp-shader/tests/corpus.rs:1767 | >>> WHY THE PAIRS A RUN ACTUALLY DRAWS FAIL TO LINK, AND WHAT THEIR TWO PROGRAMS DECLARE. |
| `VITASLOP_GXP_PAIR_CORPUS` | vitaslop-gxp-shader/tests/corpus.rs:5553 | Rank a pair corpus by the F16 EMULATION it emits: the `gxp_h*` stores and `unpack2x16float`. |
| `VITASLOP_GXP_PAIR_DUAL` | vitaslop-gxp-shader/tests/corpus.rs:1035 | Link one named (vertex, fragment) pair and print the COMPLETE WGSL module both stages become. |
| `VITASLOP_GXP_PASS_SPLIT_EVERY` | vitaslop-platform/src/gpu.rs:796 | `VITASLOP_GXP_PASS_SPLIT_EVERY=<n>` cuts a render pass every `n` draws, with no shader |
| `VITASLOP_GXP_POSPROBE` | vitaslop-gxp-shader/src/link.rs:4600 | `1` - WHERE DID THIS DRAW'S VERTICES GO? Draw the mesh at CLAMPED normalised coordinates and |
| `VITASLOP_GXP_POSPROBE_DIV` | vitaslop-gxp-shader/src/link.rs:4614 | >>> HOW FAR OFF-SCREEN, NOT ONLY THAT IT IS OFF-SCREEN. |
| `VITASLOP_GXP_PRECOMPILE` | vitaslop-platform/src/gpu.rs:1160 | Whether a shader pair the guest's patcher names is compiled AHEAD of the draw that binds it |
| `VITASLOP_GXP_PRECOMPILE_CROSS` | vitaslop-runtime/src/host.rs:17191 | `VITASLOP_GXP_PRECOMPILE_CROSS`: for a title whose `sceGxmShaderPatcherCreateFragmentProgram` |
| `VITASLOP_GXP_PREFETCH_CLAIM` | vitaslop-gxp-shader/src/link.rs:1896 | >>> A PREFETCH COORDINATE THAT LANDS ON LANES THE VERTEX NEVER WRITES IS RE-POINTED TO THE |
| `VITASLOP_GXP_PREFETCH_PROJECTIVE` | vitaslop-gxp-shader/src/link.rs:1865 | Whether a projective prefetch divides by its `w` - see `container::PrefetchLookup::Projective`. |
| `VITASLOP_GXP_PREFETCH_UNFED` | vitaslop-gxp-shader/src/link.rs:879 | >>> THE VERTEX PROGRAM PRODUCES NO SUCH TEXCOORD AT ALL, so the coordinate is the |
| `VITASLOP_GXP_PROBE` | vitaslop-gxp-shader/src/module.rs:344 | Diagnostic (`VITASLOP_GXP_PROBE=<bank><idx>[@<instr>][:f32/:bits=<hex>]`, e.g. |
| `VITASLOP_GXP_PROBE_SCALE` | vitaslop-gxp-shader/src/link.rs:4583 | Divide a probed value before it is written, so an HDR term reads back under the |
| `VITASLOP_GXP_QUADS` | vitaslop-platform/src/gpu.rs:12111 | Diagnostic (`VITASLOP_GXP_INPUTS=<hex-key>[,<hex-key>]` or `=all`): print, ONCE per |
| `VITASLOP_GXP_REAL_PAIRS` | vitaslop-gxp-shader/tests/corpus.rs:1766 | >>> WHY THE PAIRS A RUN ACTUALLY DRAWS FAIL TO LINK, AND WHAT THEIR TWO PROGRAMS DECLARE. |
| `VITASLOP_GXP_RECOMPILE` | vitaslop-runtime/src/host.rs:18489 | - |
| `VITASLOP_GXP_SA` | vitaslop-platform/src/gpu.rs:12003 | Diagnostic (`VITASLOP_GXP_FMEM=<lane>=<value>` / `<lane>*<factor>`, comma separated): |
| `VITASLOP_GXP_SA_DIRECT` | vitaslop-gxp-shader/src/link.rs:4544 | `0` restores the SA copy loop, `unroll` the constant-subscript copy - see [`resolve_sa_init`]. |
| `VITASLOP_GXP_SA_LITERAL_ALWAYS` | vitaslop-gxp-shader/src/link.rs:2623 | Is a container literal laid down for EVERY read that names its register |
| `VITASLOP_GXP_SA_UNCLAIMED` | vitaslop-gxp-shader/src/link.rs:2577 | Validate that every SA register a stage reads is either inside its default uniform buffer, |
| `VITASLOP_GXP_SIZE_BANKS` | vitaslop-gxp-shader/src/link.rs:4443 | `VITASLOP_GXP_SIZE_BANKS=0` restores the pre-2026-08-20b emission - every register bank |
| `VITASLOP_GXP_SOLID` | vitaslop-platform/src/gpu.rs:8245 | Diagnostic (`VITASLOP_GXP_SOLID`): every recompiled draw outputs solid magenta with |
| `VITASLOP_GXP_STRICT` | vitaslop-platform/src/gpu.rs:16357 | >>> A PAIR THIS RECOMPILER CANNOT TRANSLATE DROPS ITS DRAWS. |
| `VITASLOP_GXP_UNCLIPPED_DEPTH` | vitaslop-platform/src/gpu.rs:416 | - |
| `VITASLOP_GXP_VARYING_LAYOUT` | vitaslop-gxp-shader/src/link.rs:1760 | Diagnostic (`VITASLOP_GXP_VARYING_LAYOUT=<vhash>:<usage>@<lane>x<comps>,...`): plan ONE |
| `VITASLOP_GXP_VARYING_ORDER` | vitaslop-gxp-shader/src/link.rs:1214 | The vertex lane order the paired FRAGMENT's declaration implies, or `None` when the two |
| `VITASLOP_GXP_VARYING_RESOLVE` | vitaslop-gxp-shader/tests/corpus.rs:5100 | Which VERTEX programs the forwarding resolver's lane RESERVATION moves, and where to. |
| `VITASLOP_GXP_VERT` | vitaslop-gxp-shader/tests/corpus.rs:5025 | The complete LINKED WGSL for one pair, selected by `VITASLOP_GXP_VERT` + `VITASLOP_GXP_FRAG` |
| `VITASLOP_GXP_VERTEX_PASSTHROUGH` | vitaslop-platform/src/gpu.rs:1133 | `VITASLOP_GXP_VERTEX_PASSTHROUGH=0` makes every recompiled draw repack its vertex stream into |
| `VITASLOP_GXP_VPROBE` | vitaslop-gxp-shader/src/link.rs:4595 | `1` - WHERE DID THIS DRAW'S VERTICES GO? Draw the mesh at CLAMPED normalised coordinates and |
| `VITASLOP_GXP_VP_TRACE` | vitaslop-conformance-harness/tests/vita_gxmconf.rs:71 | WHERE a scene painted, in one line: the count in each quadrant-edge half plus the pixel at |
| `VITASLOP_GXP_WGSL_DIR` | vitaslop-gxp-shader/tests/corpus.rs:5028 | The complete LINKED WGSL for one pair, selected by `VITASLOP_GXP_VERT` + `VITASLOP_GXP_FRAG` |
| `VITASLOP_GXP_WGSL_OUT` | vitaslop-gxp-shader/tests/corpus.rs:7576 | Write ONE linked WGSL module per FRAGMENT blob to `VITASLOP_GXP_WGSL_OUT` - the first vertex |
| `VITASLOP_GXP_YFLIP` | vitaslop-platform/src/gpu.rs:8235 | Flip clip Y (`VITASLOP_GXP_YFLIP`, default off). |
| `VITASLOP_GXP_ZFIX` | vitaslop-platform/src/gpu.rs:8233 | Apply the GXM (GL-style, NDC z in [-1,1]) -> WebGPU (z in [0,1]) clip-depth remap |
| `VITASLOP_HB_CMP` | vitaslop-native/tests/homebrew_qsort.rs:115 | Block trace for one run: set `VITASLOP_TRACE_BLOCKS=<lo>-<hi>` (emit-time) and |
| `VITASLOP_HB_CMP_BIT` | vitaslop-native/tests/homebrew_qsort.rs:67 | - |
| `VITASLOP_HB_DUMP` | vitaslop-native/tests/homebrew_qsort.rs:48 | - |
| `VITASLOP_HB_IMAGE` | vitaslop-native/tests/homebrew_strings.rs:36 | A VM over the whole image (the routines read past string ends in aligned words, so |
| `VITASLOP_HB_N` | vitaslop-native/tests/homebrew_qsort.rs:99 | Block trace for one run: set `VITASLOP_TRACE_BLOCKS=<lo>-<hi>` (emit-time) and |
| `VITASLOP_HB_QSORT` | vitaslop-native/tests/homebrew_qsort.rs:115 | Block trace for one run: set `VITASLOP_TRACE_BLOCKS=<lo>-<hi>` (emit-time) and |
| `VITASLOP_HB_STRCMP` | vitaslop-native/tests/homebrew_strings.rs:8 | - |
| `VITASLOP_HB_STRLEN` | vitaslop-native/tests/homebrew_strings.rs:35 | A VM over the whole image (the routines read past string ends in aligned words, so |
| `VITASLOP_HEADLESS_FRAMES` | vitaslop-desktop/src/retail.rs:986 | Headless self-check of the retail path (NO window): load `dir`, optionally drive a |
| `VITASLOP_HEADLESS_NO_TAPS` | vitaslop-desktop/src/retail.rs:992 | Headless self-check of the retail path (NO window): load `dir`, optionally drive a |
| `VITASLOP_HEADLESS_RECIPE` | vitaslop-desktop/src/retail.rs:989 | Headless self-check of the retail path (NO window): load `dir`, optionally drive a |
| `VITASLOP_HEADLESS_RENDER_FROM` | vitaslop-desktop/src/retail.rs:1085 | - `VITASLOP_HEADLESS_SHOT_EVERY` - also write `<shot_dir>/fNNNNNN.png` every N display |
| `VITASLOP_HEADLESS_SHOT_EVERY` | vitaslop-desktop/src/retail.rs:1078 | - `VITASLOP_HEADLESS_SHOT_EVERY` - also write `<shot_dir>/fNNNNNN.png` every N display |
| `VITASLOP_HEADLESS_SHOT_FROM` | vitaslop-desktop/src/retail.rs:1080 | - `VITASLOP_HEADLESS_SHOT_EVERY` - also write `<shot_dir>/fNNNNNN.png` every N display |
| `VITASLOP_HEADLESS_SHOT_TO` | vitaslop-desktop/src/retail.rs:1080 | - `VITASLOP_HEADLESS_SHOT_EVERY` - also write `<shot_dir>/fNNNNNN.png` every N display |
| `VITASLOP_HEADLESS_TIMING` | vitaslop-desktop/src/retail.rs:993 | Headless self-check of the retail path (NO window): load `dir`, optionally drive a |
| `VITASLOP_HEAP_TRACE` | vitaslop-platform/src/heap.rs:51 | # The large-allocation ledger (`VITASLOP_HEAP_TRACE=<min MB>`, native only) |
| `VITASLOP_HOLD_BUTTONS` | vitaslop-native/tests/retail_boot_probe.rs:96 | A minimal host world: a monotonic clock advancing one 60Hz tick per poll, no |
| `VITASLOP_HOLD_FROM` | vitaslop-native/tests/retail_boot_probe.rs:97 | A minimal host world: a monotonic clock advancing one 60Hz tick per poll, no |
| `VITASLOP_HOLD_MEM` | vitaslop-native/tests/retail_boot_probe.rs:714 | - |
| `VITASLOP_HOLD_TOUCH` | vitaslop-native/tests/retail_boot_probe.rs:114 | - |
| `VITASLOP_HOME` | vitaslop-desktop/src/library.rs:12 | - |
| `VITASLOP_HOSTCALL_WATCH` | vitaslop-runtime/src/vita/mod.rs:692 | `VITASLOP_HOSTCALL_WATCH=<hex addr>[,<hex addr>...]` - print every host call that passes one |
| `VITASLOP_HOST_WRITE_WATCH` | vitaslop-runtime/src/host.rs:781 | `VITASLOP_HOST_WRITE_WATCH=<hex addr>[,...]`: report every write a HOST CALL makes to one |
| `VITASLOP_INGEST_DEBUG` | vitaslop-runtime/src/ingest/filesdb.rs:172 | Resolve every non-directory node to its full '/'-separated path (no |
| `VITASLOP_INPUT_RECIPE` | vitaslop-native/tests/retail_boot_probe.rs:388 | - |
| `VITASLOP_INSTANCED_MEMO` | vitaslop-runtime/src/host.rs:3400 | Whether a draw decodes only the texture units its fragment program DECLARES - see |
| `VITASLOP_IO_BANDWIDTH_KIBPS` | vitaslop-runtime/src/vita/iofilemgr.rs:22 | Modelled sequential read bandwidth, in KiB per second |
| `VITASLOP_IO_PARK_THRESHOLD_US` | vitaslop-runtime/src/vita/iofilemgr.rs:88 | >>> THROUGH THE KNOB SEAM, NOT `std::env` - THE BROWSER HAS NO ENVIRONMENT. |
| `VITASLOP_IO_REQUEST_US` | vitaslop-runtime/src/vita/iofilemgr.rs:81 | Fixed per-request cost in microseconds (`VITASLOP_IO_REQUEST_US`): the command |
| `VITASLOP_JPEG_BENCH` | vitaslop-runtime/src/vita/jpeg.rs:532 | What this decoder costs, on a real image, so "is it fast enough" is a number. |
| `VITASLOP_LOG` | vitaslop-desktop/src/log.rs:30 | Whether captured events are also written to stderr. |
| `VITASLOP_MAX_FRAMES` | vitaslop-native/tests/retail_boot_probe.rs:511 | - |
| `VITASLOP_MAX_ROUNDS` | vitaslop-native/tests/retail_boot_probe.rs:516 | - |
| `VITASLOP_MEM_DUMP` | vitaslop-runtime/src/host.rs:16796 | `VITASLOP_MEM_DUMP=<hex addr>:<bytes>[,...]`: write those guest byte ranges to |
| `VITASLOP_MEM_DUMP_AT` | vitaslop-runtime/src/host.rs:16798 | `VITASLOP_MEM_DUMP=<hex addr>:<bytes>[,...]`: write those guest byte ranges to |
| `VITASLOP_MOVIE` | vitaslop-runtime/src/vita/video.rs:1990 | A track whose codec this engine does not decode is not offered at all: the title's |
| `VITASLOP_MOVIE_DUMP_DIR` | vitaslop-runtime/src/vita/avcdec.rs:1190 | >>> AND WHAT THE PICTURE ACTUALLY LOOKS LIKE, because "a picture arrived" and "the movie |
| `VITASLOP_MOVIE_DUMP_EVERY` | vitaslop-runtime/src/vita/avcdec.rs:1190 | >>> AND WHAT THE PICTURE ACTUALLY LOOKS LIKE, because "a picture arrived" and "the movie |
| `VITASLOP_MOVIE_PICTURE_HASH` | vitaslop-runtime/src/vita/avcdec.rs:168 | Pictures handed to the guest so far, which is what `VITASLOP_MOVIE_PICTURE_HASH` |
| `VITASLOP_MOVIE_SUBSTITUTE` | vitaslop-runtime/src/vita/video.rs:150 | >>> OPEN A DIFFERENT MOVIE THAN THE TITLE ASKED FOR |
| `VITASLOP_MP4_AUDIO` | vitaslop-runtime/src/vita/video.rs:1215 | The tracks this engine will hand units for, as cursors, in the order they appear in the |
| `VITASLOP_MP4_UNITS` | vitaslop-runtime/src/vita/video.rs:1321 | `VITASLOP_MP4_UNITS=none`: never return an access unit. |
| `VITASLOP_NAME` | vitaslop-desktop/src/shell.rs:735 | - |
| `VITASLOP_NEON_CACHE` | vitaslop-transpiler/src/emit.rs:3101 | Whether emitted modules hold the low NEON bank in locals across a run of vector |
| `VITASLOP_NGS_NEG_LOOP` | vitaslop-runtime/src/vita/at9.rs:1425 | A NEGATIVE `nLoopCount` means "play this buffer once and move on", not "repeat it |
| `VITASLOP_NGS_VOICE_HANDLE_MEMO` | vitaslop-runtime/src/vita/ngs.rs:406 | SceInt32 sceNgsRackGetVoiceHandle(SceNgsHRack rack, SceUInt32 index, SceNgsHVoice *handle) |
| `VITASLOP_NGS_ZERO_LEVEL` | vitaslop-runtime/src/vita/at9.rs:1431 | A NEGATIVE `nLoopCount` means "play this buffer once and move on", not "repeat it |
| `VITASLOP_NID_DIGEST` | vitaslop-runtime/src/host.rs:20042 | The cross-engine host-call digest for the frame in progress - see [`NidDigest`]. |
| `VITASLOP_NO_BC` | vitaslop-runtime/src/render.rs:2297 | Decode a whole BC1/BC2/BC3 block to its sixteen RGBA8 texels at once. |
| `VITASLOP_NO_FAST_IMPORT` | vitaslop-runtime/src/vita/mod.rs:188 | Whether `func_nid`'s handler can only ever CONTINUE, so the transpiler may route the |
| `VITASLOP_NO_INLINE_CLIB` | vitaslop-runtime/src/vita/mod.rs:376 | `VITASLOP_NO_INLINE_CLIB`: route `sceClibMemcpy`, `sceClibMemset` and `sceClibMemcmp` |
| `VITASLOP_NO_INLINE_DELAY` | vitaslop-runtime/src/vita/mod.rs:126 | `VITASLOP_NO_INLINE_DELAY`: route every `sceKernelDelayThread` through the host, leaving |
| `VITASLOP_NO_INLINE_IMPORTS` | vitaslop-runtime/src/host.rs:8154 | >>> WHO HAS WRITTEN THE CONTEXT'S TEXTURE SLOTS, counted for the failure report above. |
| `VITASLOP_NO_INLINE_LWMUTEX` | vitaslop-runtime/src/vita/mod.rs:478 | `VITASLOP_NO_INLINE_LWMUTEX`: route the lightweight-mutex lock and unlock through the |
| `VITASLOP_NO_INLINE_MUTEX` | vitaslop-runtime/src/vita/mod.rs:498 | `VITASLOP_NO_INLINE_MUTEX`: route `sceKernelLockMutex`/`sceKernelUnlockMutex` through the |
| `VITASLOP_NO_INLINE_RESERVE` | vitaslop-runtime/src/vita/mod.rs:413 | `VITASLOP_NO_INLINE_RESERVE`: route `sceGxmReserve{Vertex,Fragment}DefaultUniformBuffer` |
| `VITASLOP_NO_INLINE_STUBS` | vitaslop-runtime/src/vita/mod.rs:350 | `VITASLOP_NO_INLINE_STUBS`: route the constant-return stubs through the host, leaving |
| `VITASLOP_NO_INLINE_TEXTURE` | vitaslop-runtime/src/host.rs:8154 | >>> WHO HAS WRITTEN THE CONTEXT'S TEXTURE SLOTS, counted for the failure report above. |
| `VITASLOP_NO_INLINE_UNIFORM_DATA` | vitaslop-runtime/src/vita/mod.rs:447 | `VITASLOP_NO_INLINE_UNIFORM_DATA`: route `sceGxmSetUniformDataF` through the host, |
| `VITASLOP_NO_NGS_MIX` | vitaslop-runtime/src/vita/audio.rs:311 | `VITASLOP_NO_NGS_MIX`: skip the NGS decode-and-mix entirely, leaving the guest's |
| `VITASLOP_PACKED_CACHE_MB` | vitaslop-platform/src/gpu.rs:3900 | >>> AND A CAP IN ENTRIES IS NOT A BOUND ON MEMORY. |
| `VITASLOP_PATCH_STUBS` | vitaslop-native/tests/retail_boot_probe.rs:466 | - |
| `VITASLOP_PAUSE_ON_BLUR` | vitaslop-desktop/src/retail.rs:2230 | Run the retail title in `dir` in a live window until the window closes or the guest |
| `VITASLOP_PEEK` | vitaslop-desktop/src/retail.rs:674 | Guest memory at `addr`, for `VITASLOP_PEEK`. |
| `VITASLOP_PERF` | vitaslop-native/src/perf.rs:43 | Is perf accounting on (`VITASLOP_PERF` set)? Read once and cached. |
| `VITASLOP_PERF_CONSOLE` | vitaslop-web/src/lib.rs:1850 | Whether the per-window performance report is also written to the browser CONSOLE |
| `VITASLOP_PIXEL_TRACE` | vitaslop-runtime/src/render.rs:5454 | Draw one scene onto an EXISTING framebuffer and depth buffer, composing with whatever |
| `VITASLOP_PKG` | vitaslop-runtime/src/ingest/stream.rs:1003 | A pkg's item table, read over a source that only ever hands out RANGES |
| `VITASLOP_POISON_UNRESOLVED_VARS` | vitaslop-runtime/src/link.rs:435 | - |
| `VITASLOP_POKE` | vitaslop-native/tests/retail_boot_probe.rs:682 | - |
| `VITASLOP_POLL_ADDR` | vitaslop-native/src/threaded.rs:1935 | Guest address to sample after each host call, from `VITASLOP_POLL_ADDR` (hex). |
| `VITASLOP_PREPARE_SPLIT` | vitaslop-platform/src/gpu.rs:7032 | Where the milliseconds INSIDE one `prepare` go, plus the bytes each phase moved. |
| `VITASLOP_PREPOKE` | vitaslop-native/tests/retail_boot_probe.rs:489 | - |
| `VITASLOP_PRESENT_PROBE` | vitaslop-web/src/lib.rs:863 | Reads back WHAT WE PRESENTED, when `VITASLOP_PRESENT_PROBE` asks for it. |
| `VITASLOP_PROBE_SAMPLE` | vitaslop-platform/src/gpu.rs:878 | >>> TEMPORARY SCAFFOLDING (`VITASLOP_PROBE_SAMPLE=<hex>+<hex>...`). |
| `VITASLOP_PROMOTE_POISON` | vitaslop-transpiler/src/emit.rs:3145 | `VITASLOP_PROMOTE_POISON=<n>` - the FALSIFIER for register promotion. |
| `VITASLOP_PROMOTE_REGS` | vitaslop-native/src/threaded.rs:2485 | A concise trap description (kind + message), matching the sync `Vm`'s detail. |
| `VITASLOP_PSARC` | vitaslop-runtime/src/psarc.rs:430 | Read a REAL archive: `VITASLOP_PSARC=<path to a .psarc>`, optionally |
| `VITASLOP_PSARC_FILE` | vitaslop-runtime/src/psarc.rs:431 | Read a REAL archive: `VITASLOP_PSARC=<path to a .psarc>`, optionally |
| `VITASLOP_PVRTC_DECODE` | vitaslop-runtime/src/render.rs:7046 | Whether PVRTC decodes a whole face at a time (the default) or one texel at a time. |
| `VITASLOP_QUANTUM_CPU_US` | vitaslop-runtime/src/host.rs:20826 | Game-clock time charged for one [`QUANTUM_ARM`] of guest execution, in microseconds. |
| `VITASLOP_QUANTUM_FUEL` | vitaslop-native/tests/retail_boot_probe.rs:425 | - |
| `VITASLOP_REGION_CLIP_SCENE` | vitaslop-runtime/src/vita/gxm.rs:1773 | - |
| `VITASLOP_REGTRACE` | vitaslop-native/src/threaded.rs:1535 | `VITASLOP_REGTRACE=<lo>-<hi>:<path>` - append the reg+flag file per block entry in |
| `VITASLOP_REGTRACE_MAX` | vitaslop-native/src/threaded.rs:1700 | `VITASLOP_REGTRACE_MAX=<n>` caps the register trace at `n` lines (0 = unbounded). |
| `VITASLOP_REGTRACE_VFP` | vitaslop-native/src/threaded.rs:1739 | Append one `pc r0..r15 n z c v [mADDR=VAL...] tTHID` line (all hex, flags 0/1) to the |
| `VITASLOP_REGTRACE_WATCH` | vitaslop-native/src/threaded.rs:1460 | The `VITASLOP_REGTRACE_WATCH` words, formatted as ` mADDR=VALUE` fields ready to append |
| `VITASLOP_RESIDENT_GEOM` | vitaslop-platform/src/gpu.rs:8520 | Repacked vertices and expanded indices that have not changed since the renderer first |
| `VITASLOP_RESIDENT_GEOM_MB` | vitaslop-platform/src/gpu.rs:8563 | The byte budget for each of the two heaps (`VITASLOP_RESIDENT_GEOM_MB`, per heap). |
| `VITASLOP_ROUNDS_PER_FRAME` | vitaslop-native/tests/retail_boot_probe.rs:660 | - |
| `VITASLOP_RTT_BG_CACHE` | vitaslop-platform/src/gpu.rs:750 | `VITASLOP_RTT_BG_CACHE=0` restores the OLD behaviour: a sampler bind group naming a render |
| `VITASLOP_RTT_CLEAR_PROBE` | vitaslop-native/src/wgpu_render.rs:681 | - |
| `VITASLOP_RTT_PROBE_LOG` | vitaslop-runtime/src/rtt_writeback.rs:85 | TEMPORARY TELEMETRY (`VITASLOP_RTT_PROBE_LOG=1`): print every written-back target's |
| `VITASLOP_SAMPLER_NARROW` | vitaslop-runtime/src/host.rs:3396 | Whether a draw decodes only the texture units its fragment program DECLARES - see |
| `VITASLOP_SCAN_WORD` | vitaslop-native/tests/retail_boot_probe.rs:962 | - |
| `VITASLOP_SCENE_LIMIT` | vitaslop-native/tests/retail_boot_probe.rs:446 | - |
| `VITASLOP_SCHED_CORES` | vitaslop-runtime/src/sched.rs:638 | `VITASLOP_SCHED_CORES=<n>`: cap the baton to the top `n` runnable PRIORITIES, as the |
| `VITASLOP_SCHED_RR` | vitaslop-runtime/src/sched.rs:654 | `VITASLOP_SCHED_RR=1`: round-robin every runnable thread, ignoring priority. |
| `VITASLOP_SCHED_TRACE` | vitaslop-runtime/src/sched.rs:664 | `VITASLOP_SCHED_TRACE=<from>-<to>` (display frames, inclusive): print one line per |
| `VITASLOP_SEMA_TRAIL` | vitaslop-runtime/src/vita/sync.rs:378 | TEMPORARY DIAGNOSTIC. |
| `VITASLOP_SET_EVF` | vitaslop-native/tests/retail_boot_probe.rs:692 | - |
| `VITASLOP_SHOT_DIR` | vitaslop-native/tests/retail_boot_probe.rs:207 | Read and format one watched value from current guest memory. |
| `VITASLOP_SHOT_LAST` | vitaslop-native/tests/retail_boot_probe.rs:444 | - |
| `VITASLOP_SIGNATURE` | vitaslop-native/src/recipe_runner.rs:101 | The determinism signature over the observable output (render stream + egress), |
| `VITASLOP_SIGNATURE_EVERY` | vitaslop-native/src/recipe_runner.rs:483 | `VITASLOP_SIGNATURE_EVERY=<n>`: print the RUNNING determinism signature every `n` stepped |
| `VITASLOP_SLOW_FRAME_US` | vitaslop-web/src/lib.rs:4636 | - |
| `VITASLOP_SNAPSHOT` | vitaslop-native/src/threaded.rs:1521 | `VITASLOP_SNAPSHOT=<hexpc>:<path>` - dump full state on first entry to block `hexpc`. |
| `VITASLOP_SNAPSHOT_BUDGET_MB` | vitaslop-runtime/src/host.rs:4043 | Byte budget for retained texture snapshots, scaled to the device |
| `VITASLOP_SNAPSHOT_DENSE` | vitaslop-native/src/threaded.rs:1637 | Dump the full guest state (all non-zero pages + r0..r15 + NZCV) to `path`, in the |
| `VITASLOP_SNAPSHOT_SKIP` | vitaslop-native/src/threaded.rs:1589 | `VITASLOP_SNAPSHOT_SKIP=<n>` - skip the first `n` entries to the snapshot block before |
| `VITASLOP_SOFTWARE` | vitaslop-desktop/src/retail.rs:2115 | - |
| `VITASLOP_SSAA` | vitaslop-platform/src/gpu.rs:18818 | Set the supersample factor: 1 (default) renders the scene straight into the caller's |
| `VITASLOP_STALL_CHUNK` | vitaslop-native/tests/retail_boot_probe.rs:543 | - |
| `VITASLOP_STALL_WAKE` | vitaslop-native/tests/retail_boot_probe.rs:542 | - |
| `VITASLOP_STALL_WATCHDOG` | vitaslop-native/src/watchdog.rs:99 | The configured stall budget in seconds, from `VITASLOP_STALL_WATCHDOG`. |
| `VITASLOP_STALL_WAVES` | vitaslop-native/tests/retail_boot_probe.rs:546 | - |
| `VITASLOP_STRICT_DRAWS` | vitaslop-runtime/src/render.rs:6718 | Why [`RenderSceneBuilder::build`] discarded draws from a captured scene. |
| `VITASLOP_SWITCH_WHY` | vitaslop-transpiler/src/lower.rs:1225 | Whether the table-branch diagnostic is on for this address |
| `VITASLOP_SW_CHAIN` | vitaslop-native/src/wgpu_render.rs:582 | `VITASLOP_GPU_CHAIN_DIR=<dir>`: write every offscreen target of the frame just |
| `VITASLOP_SW_CHAIN_DIR` | vitaslop-runtime/src/render.rs:5259 | - |
| `VITASLOP_SW_POST` | vitaslop-runtime/src/render.rs:5306 | - |
| `VITASLOP_SYSTEM_FONT` | vitaslop-runtime/src/font/system.rs:75 | The resolved substitute: its bytes and a human-readable account of where they came from. |
| `VITASLOP_TEXTURE_CHECK` | vitaslop-runtime/src/host.rs:3776 | How a retained texture snapshot is re-validated (`VITASLOP_TEXTURE_CHECK`): `scene` |
| `VITASLOP_TEX_CACHE_MB` | vitaslop-platform/src/gpu.rs:658 | The texture-cache budget in bytes: [`GAME_RESIDENT_CEILING_MB`] unless |
| `VITASLOP_TEX_COMPRESS` | vitaslop-runtime/src/render.rs:1563 | Whether compressed textures reach the GPU compressed at all. |
| `VITASLOP_TEX_DIRTY_CENSUS` | vitaslop-runtime/src/host.rs:203 | >>> WHICH PARTS of `[off, off + len)` the guest may have stored into since `stamp`, |
| `VITASLOP_TEX_ENCODE_BUDGET_MS` | vitaslop-runtime/src/render.rs:1141 | >>> HOW MUCH OF ONE FRAME THE INLINE BLOCK ENCODE MAY SPEND, in milliseconds. |
| `VITASLOP_TEX_ENCODE_RESUME` | vitaslop-runtime/src/render.rs:1270 | `VITASLOP_TEX_ENCODE_RESUME=0` restores the OLD behaviour: an encode that starts runs to |
| `VITASLOP_TEX_MEMO_PER_SCENE` | vitaslop-runtime/src/host.rs:3043 | A whole DRAW's worth of snapshotted textures, by the bindings that produced it - kept |
| `VITASLOP_TEX_PAGE_READ` | vitaslop-runtime/src/host.rs:4386 | Record that this entry's bytes are current as of THIS SCENE, so a later |
| `VITASLOP_TEX_RETAIN_MB` | vitaslop-platform/src/gpu.rs:685 | How many bytes of GPU texture the view cache may RETAIN, as opposed to how many the |
| `VITASLOP_TRACE_BLOCKS` | vitaslop-native/src/threaded.rs:1435 | `VITASLOP_TRACE_FRAMES=<from>-<to>` (decimal display frames, inclusive) - print the |
| `VITASLOP_TRACE_EXIT` | vitaslop-native/tests/retail_boot_probe.rs:37 | - |
| `VITASLOP_TRACE_FILE` | vitaslop-runtime/src/vita/libkernel.rs:47 | Diagnostic (`RUST_LOG=vitaslop::exit=debug`): when the guest calls |
| `VITASLOP_TRACE_FRAMES` | vitaslop-native/src/threaded.rs:1432 | `VITASLOP_TRACE_FRAMES=<from>-<to>` (decimal display frames, inclusive) - print the |
| `VITASLOP_TRACE_FUNCS` | vitaslop-native/src/threaded.rs:1301 | Bind `env.svc`. |
| `VITASLOP_TRACE_INDIRECT` | vitaslop-transpiler/src/emit.rs:1904 | Diagnostic indirect-call tracer. |
| `VITASLOP_TRACE_IO` | vitaslop-native/tests/retail_boot_probe.rs:34 | - |
| `VITASLOP_TRACE_ORDER` | vitaslop-runtime/src/vita/mod.rs:676 | Ordered-timeline trace (env `VITASLOP_TRACE_ORDER`): print every *meaningful* |
| `VITASLOP_TRACE_ORDER_FULL` | vitaslop-runtime/src/vita/mod.rs:1047 | - |
| `VITASLOP_TRACK_PC` | vitaslop-transpiler/src/abi.rs:200 | Exported name of the diagnostic guest-PC tracker global. |
| `VITASLOP_TRANSPILE_REPORT` | vitaslop-native/src/threaded.rs:920 | - |
| `VITASLOP_TRAP_HALT` | vitaslop-transpiler/src/emit.rs:2032 | When `VITASLOP_TRAP_HALT` is set, a `Term::Halt` (a block that ran off the end of decoded |
| `VITASLOP_UBIND_TRACE` | vitaslop-runtime/src/host.rs:14855 | Apply `sceGxmSetPrecomputedVertexState(context, state)`: replace the context's |
| `VITASLOP_UNIFORM_WATCH` | vitaslop-runtime/src/vita/gxm.rs:2207 | `VITASLOP_UNIFORM_WATCH=<hex address>/<parameter name substring>[,...]`: report every |
| `VITASLOP_UV_DEBUG` | vitaslop-runtime/src/render.rs:5468 | Draw one scene onto an EXISTING framebuffer and depth buffer, composing with whatever |
| `VITASLOP_VBLANK_PARK` | vitaslop-runtime/src/vita/display.rs:201 | Whether an inlined `sceDisplayGetVcount` carries the spin guard (`VITASLOP_VBLANK_PARK`, |
| `VITASLOP_VERTEX_INTERN` | vitaslop-runtime/src/host.rs:3370 | A cheap, allocation-free fingerprint of a vertex stream, for [`TextureSnapshots:: |
| `VITASLOP_VERTEX_INTERN_USE` | vitaslop-runtime/src/host.rs:3372 | A cheap, allocation-free fingerprint of a vertex stream, for [`TextureSnapshots:: |
| `VITASLOP_VPK` | vitaslop-runtime/src/ingest/stream.rs:1057 | A homebrew VPK (`VITASLOP_VPK=<file>`): probed as `vpk`, imported as files + a |
| `VITASLOP_WASM_INDICES` | vitaslop-native/src/threaded.rs:2574 | Rewrite `<wasm function N>` in a trap backtrace to name the GUEST function it is. |
| `VITASLOP_WASM_NAMES` | vitaslop-transpiler/src/emit.rs:1470 | When `VITASLOP_WASM_NAMES` is set, emit a wasm `name` custom section labelling |
| `VITASLOP_WATCH_` | vitaslop-transpiler/src/emit.rs:1816 | Number of matching store-watchpoint hits to skip before trapping (`VITASLOP_WATCH_ |
| `VITASLOP_WATCH_FROM` | vitaslop-native/tests/retail_boot_probe.rs:672 | - |
| `VITASLOP_WATCH_MEM` | vitaslop-native/tests/retail_boot_probe.rs:139 | Parse `VITASLOP_WATCH_MEM=addr:type:label,addr:type:label,...` into watches. |
| `VITASLOP_WATCH_READ` | vitaslop-transpiler/src/emit.rs:1408 | Diagnostic read watchpoint. |
| `VITASLOP_WATCH_READ_` | vitaslop-transpiler/src/emit.rs:1842 | Optional guest-PC EXCLUDE window for the read watchpoint (`VITASLOP_WATCH_READ_ |
| `VITASLOP_WATCH_READ_NZ` | vitaslop-transpiler/src/emit.rs:3543 | Emit the read-watchpoint trap check. |
| `VITASLOP_WATCH_READ_PC_EXCL` | vitaslop-transpiler/src/emit.rs:1852 | Optional guest-PC EXCLUDE window for the read watchpoint (`VITASLOP_WATCH_READ_ |
| `VITASLOP_WATCH_READ_SKIP` | vitaslop-transpiler/src/emit.rs:1556 | WASM global index of the read-watchpoint match counter, appended after the guest-PC |
| `VITASLOP_WATCH_STORE` | vitaslop-native/src/threaded.rs:1556 | `VITASLOP_REGTRACE_WATCH=<hex guest addr>[,<hex guest addr>...]` - append the WORD |
| `VITASLOP_WATCH_STORE_ARM` | vitaslop-transpiler/src/emit.rs:1203 | The emit-time knobs [`set_emit_knob`] accepts, so a caller that forwards a whole table |
| `VITASLOP_WATCH_STORE_LOG` | vitaslop-runtime/src/capture.rs:543 | GUEST ADDRESS the bytes above were read from, or 0 when there is no bound buffer. |
| `VITASLOP_WATCH_STORE_MODE` | vitaslop-transpiler/src/emit.rs:2047 | Store-watchpoint mode, from `VITASLOP_WATCH_STORE_MODE` (default `any`): |
| `VITASLOP_WATCH_STORE_NZ` | vitaslop-transpiler/src/emit.rs:2070 | `VITASLOP_WATCH_STORE_LOG` - LOG each store to the watched address (the storing |
| `VITASLOP_WATCH_STORE_SKIP` | vitaslop-transpiler/src/emit.rs:637 | Linear-memory byte offset of the store-watchpoint MATCH COUNTER, or 0 when this |
| `VITASLOP_WHICH_EXPORT` | vitaslop-runtime/src/link.rs:261 | - |
| `VITASLOP_WINDOW_CENSUS` | vitaslop-runtime/src/host.rs:3516 | `VITASLOP_WINDOW_CENSUS=1`: measure the census's `slot-blank` count exactly even under the |
| `VITASLOP_WINDOW_REREAD` | vitaslop-runtime/src/host.rs:3466 | Which windows the guest's GPU wait re-reads - see [`window_wants_reread`]. |
| `VITASLOP_X` | vitaslop-frontend/src/settings.rs:263 | Parse a `NAME=VALUE` per line knobs box into a map. |
| `VITASLOP_XML_DUMP` | vitaslop-runtime/src/vita/sce_xml.rs:580 | `VITASLOP_XML_DUMP=<dir>`: write every document handed to `parse` into `<dir>` as |
| `VITASLOP_Y` | vitaslop-frontend/src/settings.rs:264 | Parse a `NAME=VALUE` per line knobs box into a map. |
| `VITASLOP_YIELD_ELIDE` | vitaslop-runtime/src/host.rs:7766 | Whether the `sceKernelDelayThread(0)` yield elision is on - see |
