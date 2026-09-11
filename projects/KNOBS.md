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

338 knobs.

| knob | read in | what it does |
|---|---|---|
| `VITASLOP_A` | vitaslop-frontend/src/settings.rs:263 | - |
| `VITASLOP_AAC_DIR` | vitaslop-aac/tests/oracle.rs:69 | The `AudioSpecificConfig` an ADTS header describes: 5 bits of object type, 4 of |
| `VITASLOP_ALLOW_SOFTWARE_GPU` | vitaslop-web/src/lib.rs:1682 | Whether a run may proceed on a software rasteriser (`VITASLOP_ALLOW_SOFTWARE_GPU`). |
| `VITASLOP_ARM_AT_FRAME` | vitaslop-native/src/threaded.rs:323 | Linear-memory offset of the "diagnostics armed" word, when this build was |
| `VITASLOP_ARM_FUNC` | vitaslop-native/tests/homebrew_memcpy.rs:21 | Scratch window well past the code: `[SRC, SRC+WIN)` and `[DST, DST+WIN)`. |
| `VITASLOP_AT9_DIR` | vitaslop-atrac9/tests/oracle.rs:66 | Decode a whole AT9 payload the way a superframe consumer does: for each |
| `VITASLOP_AUDIO_RAW` | vitaslop-runtime/src/vita/audio.rs:92 | Optional raw-s16le capture of the mixed output stream (env |
| `VITASLOP_B` | vitaslop-frontend/src/settings.rs:263 | - |
| `VITASLOP_BACKTRACE` | vitaslop-runtime/src/vita/mod.rs:532 | Print the guest call chain the first time a chosen NID is called from each thread |
| `VITASLOP_BLOCK_HIST` | vitaslop-native/src/recipe_runner.rs:167 | Dump the per-PC block-entry histogram gathered under `VITASLOP_BLOCK_HIST`, for |
| `VITASLOP_BLOCK_HIST_SEQ` | vitaslop-native/src/threaded.rs:1714 | Print the block-visit histogram gathered under `VITASLOP_BLOCK_HIST`: the `top` |
| `VITASLOP_BROWSER_FASTFORWARD` | vitaslop-web/src/lib.rs:1662 | Frame to fast-forward the live loop to (`VITASLOP_BROWSER_FASTFORWARD`), unpaced. |
| `VITASLOP_BROWSER_FUEL` | vitaslop-web/src/browser_sched.rs:920 | Guest work a thread may execute before the browser preempts it, in WASMTIME FUEL UNITS |
| `VITASLOP_BROWSER_HEARTBEAT_MS` | vitaslop-web/src/lib.rs:4336 | - |
| `VITASLOP_BROWSER_INSTANCE_POOL` | vitaslop-web/src/browser_sched.rs:1599 | Whether a finished thread's module instance may be REUSED by the next thread |
| `VITASLOP_BROWSER_QUANTUM_CALLS` | vitaslop-web/src/browser_sched.rs:121 | Host calls one guest thread may make before the browser preempts it |
| `VITASLOP_BROWSER_SUPERSAMPLE` | vitaslop-web/src/lib.rs:1639 | Supersample factor for the live browser render (`VITASLOP_BROWSER_SUPERSAMPLE`). |
| `VITASLOP_CALLSITES_WINDOW` | vitaslop-desktop/src/retail.rs:539 | Where the idle clock went SINCE `before` - the windowed reading. |
| `VITASLOP_CAPSULE_DUMP_PROGS` | vitaslop-native/examples/capsule-replay.rs:527 | - |
| `VITASLOP_CAPSULE_DUMP_SA` | vitaslop-native/examples/capsule-replay.rs:29 | `VITASLOP_CAPSULE_DUMP_SA=1`: print this draw's uniform banks - `frag_sa` with the GUEST |
| `VITASLOP_CAPSULE_DUMP_VERTS` | vitaslop-native/examples/capsule-replay.rs:436 | - |
| `VITASLOP_CAPSULE_TEX_DIR` | vitaslop-native/examples/capsule-replay.rs:338 | - |
| `VITASLOP_CHAIN_DRAWS` | vitaslop-platform/src/gpu.rs:18216 | - |
| `VITASLOP_CHAIN_LIMIT` | vitaslop-native/tests/gpu_rtt_gamma.rs:178 | Render a chain of `feedback` sample-and-write-back passes over the offscreen target and |
| `VITASLOP_CHAIN_SKIP` | vitaslop-native/src/wgpu_render.rs:346 | - |
| `VITASLOP_CHECK_ADDRS` | vitaslop-native/tests/retail_boot_probe.rs:43 | - |
| `VITASLOP_CLOCK_TRACE` | vitaslop-runtime/src/sched.rs:1120 | - |
| `VITASLOP_CODE_RANGE` | vitaslop-runtime/src/vita/mod.rs:515 | The guest code range scanned for the game-level caller in [`dispatch`] (env |
| `VITASLOP_CONSOLE` | vitaslop-web/src/logging.rs:218 | `VITASLOP_CONSOLE=1`: mirror the run's status notes - the setup summary, the adapter and |
| `VITASLOP_CPU_SHARE` | vitaslop-native/src/recipe_runner.rs:111 | Who actually got the CPU over the run, when `VITASLOP_CPU_SHARE` is set - see |
| `VITASLOP_DBG_CALLSITES` | vitaslop-runtime/src/vita/mod.rs:473 | Diagnostic call-site profiler (`VITASLOP_DBG_CALLSITES`): counts host calls |
| `VITASLOP_DEBUG_CAPTURE` | vitaslop-frontend/src/settings.rs:167 | The `VITASLOP_*` map a run of these settings is configured with: the base set, |
| `VITASLOP_DECODE_CACHE_MB` | vitaslop-runtime/src/render.rs:6328 | Budget for the decode cache, in BYTES of decoded RGBA8, before it is cleared wholesale. |
| `VITASLOP_DEFER_GEOMETRY` | vitaslop-runtime/src/host.rs:19355 | Whether a draw's VERTEX AND INDEX BYTES are read at `sceGxmEndScene` rather than at the |
| `VITASLOP_DELAY_CENSUS` | vitaslop-runtime/src/vita/threadmgr.rs:155 | Diagnostic (`VITASLOP_DELAY_CENSUS=1`): every `sceKernelDelayThread` tallied by (call site, |
| `VITASLOP_DIRTY_PAGES` | vitaslop-native/src/threaded.rs:128 | Linear-memory offset of the guest-store dirty block, when this build was |
| `VITASLOP_DISPATCH_ALL` | vitaslop-transpiler/src/emit.rs:1369 | The ablation arm that prices a dispatch re-entry: `VITASLOP_DISPATCH_ALL=1` sends even a |
| `VITASLOP_DRAW_ONLY` | vitaslop-runtime/src/render.rs:5099 | Draw one scene onto an EXISTING framebuffer and depth buffer, composing with whatever |
| `VITASLOP_DRAW_RANGE` | vitaslop-platform/src/gpu.rs:597 | `VITASLOP_RTT_BG_CACHE=0` restores the OLD behaviour: a sampler bind group naming a render |
| `VITASLOP_DRAW_STATS` | vitaslop-runtime/src/render.rs:5088 | Draw one scene onto an EXISTING framebuffer and depth buffer, composing with whatever |
| `VITASLOP_DRV_KEY` | vitaslop-runtime/src/ingest/pfscrypt.rs:344 | `F00D(klicensee)` for the title, from `VITASLOP_DRV_KEY` (32 hex chars), or |
| `VITASLOP_DUMP_DIR` | vitaslop-runtime/src/ingest/pipeline.rs:824 | Diagnostic: decrypt the container and write named plaintext files out to |
| `VITASLOP_DUMP_DRAW` | vitaslop-native/tests/retail_boot_probe.rs:1286 | - |
| `VITASLOP_DUMP_DRAWS` | vitaslop-native/tests/retail_boot_probe.rs:1075 | - |
| `VITASLOP_DUMP_DRAW_GXP` | vitaslop-gxp-shader/tests/oracle.rs:653 | Correlate each captured vertex<->fragment PAIR (from a real draw run) to establish the |
| `VITASLOP_DUMP_DRAW_GXP_CAP` | vitaslop-runtime/src/host.rs:15000 | Re-read every window [`STREAM_WATCH`] holds and report the ones whose bytes CHANGED |
| `VITASLOP_DUMP_DRAW_GXP_FULL` | vitaslop-runtime/src/host.rs:15217 | - |
| `VITASLOP_DUMP_EXPORTS` | vitaslop-runtime/src/link.rs:494 | - |
| `VITASLOP_DUMP_FILES` | vitaslop-runtime/src/ingest/pipeline.rs:826 | Diagnostic: decrypt the container and write named plaintext files out to |
| `VITASLOP_DUMP_FPROG` | vitaslop-runtime/src/host.rs:14808 | Diagnostic (VITASLOP_DUMP_FPROG): print the bound fragment program's sampler |
| `VITASLOP_DUMP_FUNC` | vitaslop-native/tests/retail_boot_probe.rs:317 | - |
| `VITASLOP_DUMP_GXP_BIN` | vitaslop-platform/src/gpu.rs:13385 | >>> A REFUSAL THAT DOES NOT HAND OVER THE EVIDENCE COSTS A PLAY SESSION. |
| `VITASLOP_DUMP_IMAGE` | vitaslop-native/tests/retail_boot_probe.rs:33 | - |
| `VITASLOP_DUMP_IMPORTS` | vitaslop-native/tests/retail_boot_probe.rs:353 | - |
| `VITASLOP_DUMP_IR` | vitaslop-native/tests/homebrew_memcpy.rs:40 | Scratch window well past the code: `[SRC, SRC+WIN)` and `[DST, DST+WIN)`. |
| `VITASLOP_DUMP_MAP` | vitaslop-native/tests/retail_boot_probe.rs:589 | - |
| `VITASLOP_DUMP_MEM` | vitaslop-native/tests/retail_boot_probe.rs:43 | - |
| `VITASLOP_DUMP_PATHS` | vitaslop-native/tests/retail_boot_probe.rs:412 | - |
| `VITASLOP_DUMP_REGION` | vitaslop-native/tests/retail_boot_probe.rs:527 | - |
| `VITASLOP_DUMP_REGION_RANGE` | vitaslop-native/tests/retail_boot_probe.rs:529 | - |
| `VITASLOP_DUMP_RENDERSCENE` | vitaslop-native/tests/retail_boot_probe.rs:1327 | - |
| `VITASLOP_DUMP_SCENES` | vitaslop-desktop/src/retail.rs:417 | Step the guest one display frame. |
| `VITASLOP_DUMP_STDOUT` | vitaslop-desktop/src/retail.rs:1686 | - |
| `VITASLOP_DUMP_STREAM_BYTES` | vitaslop-runtime/src/host.rs:19316 | `VITASLOP_DUMP_STREAM_BYTES[=<n>]`: with the per-draw dump on, print each vertex stream's |
| `VITASLOP_DUMP_STUBS` | vitaslop-native/tests/retail_boot_probe.rs:32 | - |
| `VITASLOP_DUMP_TEX` | vitaslop-native/tests/retail_boot_probe.rs:1229 | - |
| `VITASLOP_DUMP_TEX_DIR` | vitaslop-runtime/src/host.rs:15223 | - |
| `VITASLOP_DUMP_TEX_MAX_TEXELS` | vitaslop-runtime/src/host.rs:15232 | - |
| `VITASLOP_DUMP_TRIS` | vitaslop-runtime/src/render.rs:5106 | Draw one scene onto an EXISTING framebuffer and depth buffer, composing with whatever |
| `VITASLOP_DUMP_VPROG` | vitaslop-runtime/src/host.rs:14616 | Diagnostic (VITASLOP_DUMP_VPROG): reflect the bound vertex program's parameter |
| `VITASLOP_EAGER_FILES` | vitaslop-desktop/src/retail.rs:241 | Load, decrypt, link, transpile, and instantiate the title in `dir` for live |
| `VITASLOP_FIND_WORD` | vitaslop-native/tests/retail_boot_probe.rs:63 | The span `VITASLOP_FIND_WORD` searches: from the image base up through the guest heap. |
| `VITASLOP_FLAGS_WIDE_C` | vitaslop-transpiler/src/emit.rs:1342 | The A/B arm for [`emit_flags_add`]'s carry and overflow forms: `VITASLOP_FLAGS_WIDE_C=1` |
| `VITASLOP_FLAG_POISON` | vitaslop-transpiler/src/emit.rs:1300 | `VITASLOP_FLAG_POISON=0/1` - the FALSIFIER for the flag-liveness pass |
| `VITASLOP_FORCE_READY` | vitaslop-native/tests/retail_boot_probe.rs:735 | - |
| `VITASLOP_FORCE_READY_V2` | vitaslop-native/tests/retail_boot_probe.rs:758 | - |
| `VITASLOP_FORCE_RET` | vitaslop-transpiler/src/emit.rs:1660 | Diagnostic forced return. |
| `VITASLOP_FRAME_DIGEST` | vitaslop-native/src/recipe_runner.rs:344 | - |
| `VITASLOP_FRAME_TOPUP` | vitaslop-runtime/src/host.rs:5919 | The per-flip top-up ([`VitaState::advance_time_frame`]), which is OPT-IN: |
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
| `VITASLOP_GPU_CHAIN_DIR` | vitaslop-native/src/wgpu_render.rs:558 | `VITASLOP_GPU_CHAIN_DIR=<dir>`: write every offscreen target of the frame just |
| `VITASLOP_GPU_QUEUE_DEPTH` | vitaslop-web/src/lib.rs:805 | How many submits may be in flight before a present declines to make another. |
| `VITASLOP_GUARD_REG` | vitaslop-transpiler/src/emit.rs:1529 | Diagnostic callee-saved-register guard. |
| `VITASLOP_GUEST_CORES` | vitaslop-runtime/src/host.rs:17573 | CPU cores a Vita gives a GAME. |
| `VITASLOP_GXM` | vitaslop-native/examples/capsule-replay.rs:546 | - |
| `VITASLOP_GXM_DEPTH_ENC` | vitaslop-platform/src/gpu.rs:2748 | Which value a later pass reads out of a render target's depth |
| `VITASLOP_GXM_DEST_SPLIT_AB` | vitaslop-platform/src/gpu.rs:637 | `VITASLOP_GXM_DEST_SPLIT_AB=<n>`: alternate the destination-colour pass SPLIT on and off every |
| `VITASLOP_GXM_DRAW_COVERAGE` | vitaslop-platform/src/gpu.rs:4320 | `VITASLOP_GXM_DRAW_COVERAGE=1`: how many SAMPLES each draw of the frame actually wrote, |
| `VITASLOP_GXM_NO_MULTISAMPLE` | vitaslop-platform/src/gpu.rs:6265 | A/B instrument: force every pass to ONE sample, whatever the guest asked for. |
| `VITASLOP_GXM_RTT_WRITEBACK` | vitaslop-native/src/wgpu_render.rs:468 | The rendered pixels of every offscreen target small enough to hand back to the GUEST, |
| `VITASLOP_GXM_STALE_UNIFORMS` | vitaslop-runtime/src/host.rs:12974 | Is the bound default uniform buffer for `stage` left over from a DIFFERENT program |
| `VITASLOP_GXM_TEX_UNWRITTEN` | vitaslop-runtime/src/render.rs:6307 | `VITASLOP_GXM_TEX_UNWRITTEN=0` - the arm back for reading ANY uniform 4-byte fill as |
| `VITASLOP_GXM_UNIFORM_POISON` | vitaslop-gxp-shader/src/module.rs:386 | `:bits=<hex>` - paint a lane 1.0 when that register's RAW BITS equal this word, 0.0 |
| `VITASLOP_GXP` | vitaslop-native/examples/capsule-replay.rs:546 | - |
| `VITASLOP_GXP_` | vitaslop-native/examples/capsule-replay.rs:14 | - |
| `VITASLOP_GXP_ALLOW_FIXED_FUNCTION` | vitaslop-platform/src/gpu.rs:13452 | >>> A PAIR THIS RECOMPILER CANNOT TRANSLATE DROPS ITS DRAWS. |
| `VITASLOP_GXP_ATTR_ALL` | vitaslop-gxp-shader/tests/corpus.rs:4496 | What [`vitaslop_gxp_shader::attrflow`] decides for every attribute of every pair that links, |
| `VITASLOP_GXP_ATTR_FILL` | vitaslop-gxp-shader/src/module.rs:1944 | Bytes the driver adds to the DEFAULT uniform buffer's bound address before writing it into |
| `VITASLOP_GXP_BLOB` | vitaslop-gxp-shader/tests/corpus.rs:688 | Print one named blob's recompiled WGSL body and its container reflection. |
| `VITASLOP_GXP_CAPSULE` | vitaslop-runtime/src/capsule.rs:653 | Diagnostic (`VITASLOP_GXP_CAPSULE=<vprog-hash>[,<vprog-hash>]:<dir>[:N]`): write the first |
| `VITASLOP_GXP_CAPSULE_MIN_INDICES` | vitaslop-runtime/src/capsule.rs:666 | Diagnostic (`VITASLOP_GXP_CAPSULE=<vprog-hash>[,<vprog-hash>]:<dir>[:N]`): write the first |
| `VITASLOP_GXP_CAPSULE_SKIP` | vitaslop-runtime/src/capsule.rs:686 | Diagnostic (`VITASLOP_GXP_CAPSULE_SKIP=<n>`): ignore the first `n` matching submissions |
| `VITASLOP_GXP_CLAIM_WIDTH` | vitaslop-gxp-shader/src/link.rs:1298 | Re-order a convention-placed layout so that every forwarded attribute's usage STARTS at the |
| `VITASLOP_GXP_CLIP_DUMP_SA` | vitaslop-platform/src/gpu.rs:11952 | - |
| `VITASLOP_GXP_CORPUS` | vitaslop-gxp-shader/tests/corpus.rs:5068 | Census the RAW prefetch-bearing words of every fragment descriptor, unreduced to flags. |
| `VITASLOP_GXP_CULL` | vitaslop-platform/src/gpu.rs:779 | `VITASLOP_GXP_CULL=0` restores the pre-2026-08-19b "draw both windings". |
| `VITASLOP_GXP_DEBUG` | vitaslop-platform/src/gpu.rs:14011 | Report - once per case - a GXM blend value with no exact wgpu equivalent, so the |
| `VITASLOP_GXP_DEFAULT_UNIFORM_OFFSET` | vitaslop-gxp-shader/src/module.rs:1941 | Bytes the driver adds to the DEFAULT uniform buffer's bound address before writing it into |
| `VITASLOP_GXP_DEPTH_PROBE` | vitaslop-gxp-shader/src/link.rs:3349 | Diagnostic (`VITASLOP_GXP_DEPTH_PROBE=<lo>:<scale>`): EVERY fragment returns its own window |
| `VITASLOP_GXP_DEST` | vitaslop-gxp-shader/src/module.rs:690 | Whether a fragment program that reads the destination colour is DECLARED as reading it. |
| `VITASLOP_GXP_DEST_BLEND` | vitaslop-gxp-shader/src/module.rs:616 | Whether [`lower_dest_blend`] is allowed to run. |
| `VITASLOP_GXP_DEST_POISON` | vitaslop-gxp-shader/src/module.rs:1612 | `VITASLOP_GXP_DEST_POISON=<r,g,b,a>` - the constant [`dest_color_init`] seeds the output |
| `VITASLOP_GXP_DEST_PROBE` | vitaslop-gxp-shader/src/link.rs:3370 | Diagnostic (`VITASLOP_GXP_DEST_PROBE=1/opaque/mark`): a fragment program that reads the |
| `VITASLOP_GXP_DISASM` | vitaslop-gxp-shader/tests/oracle.rs:860 | Compact disassembly of one blob (named by `VITASLOP_GXP_DISASM`, matched as a filename |
| `VITASLOP_GXP_DUAL_SOURCE` | vitaslop-gxp-shader/src/module.rs:707 | Whether a destination-reading program that is LINEAR in the destination may be lowered to a |
| `VITASLOP_GXP_DUAL_TRACE` | vitaslop-gxp-shader/tests/corpus.rs:5490 | Print the DUAL-SOURCE plan (or the reason there is none) for every destination reader in |
| `VITASLOP_GXP_DUMP` | vitaslop-platform/src/gpu.rs:6586 | Diagnostic (`VITASLOP_GXP_KEYS=<hex>,<hex>`): recompile ONLY these shader-pair keys |
| `VITASLOP_GXP_DUMPS` | vitaslop-gxp-shader/tests/oracle.rs:145 | Histogram the raw values of named fields across every instruction of a given opcode1 |
| `VITASLOP_GXP_EXCLUDE` | vitaslop-platform/src/gpu.rs:6590 | Pairs forced down the fixed-function path (`VITASLOP_GXP_EXCLUDE`). |
| `VITASLOP_GXP_FMEM` | vitaslop-platform/src/gpu.rs:9676 | Diagnostic (`VITASLOP_GXP_FMEM=<lane>=<value>` / `<lane>*<factor>`, comma separated): |
| `VITASLOP_GXP_FORCE` | vitaslop-platform/src/gpu.rs:6557 | Diagnostic (`VITASLOP_GXP_FORCE`): bind a neutral fallback texture for a sampler |
| `VITASLOP_GXP_FRAG` | vitaslop-gxp-shader/tests/corpus.rs:4548 | The complete LINKED WGSL for one pair, selected by `VITASLOP_GXP_VERT` + `VITASLOP_GXP_FRAG` |
| `VITASLOP_GXP_GROUP` | vitaslop-gxp-shader/tests/corpus.rs:2865 | Every distinct word of one opcode group across the corpus, with the programs it appears in. |
| `VITASLOP_GXP_IDX_SCALE` | vitaslop-gxp-shader/src/module.rs:667 | How many REGISTERS one count of an index register spans - see [`crate::wgsl`]'s |
| `VITASLOP_GXP_INPUTS` | vitaslop-platform/src/gpu.rs:9775 | Diagnostic (`VITASLOP_GXP_INPUTS=<hex-key>[,<hex-key>]` or `=all`): print, ONCE per |
| `VITASLOP_GXP_INPUTS_DIR` | vitaslop-platform/src/gpu.rs:2864 | Whether the once-per-pair `gxp pair <key>: vprog hash ..., fprog hash ...` INDEX should be |
| `VITASLOP_GXP_INPUTS_ORDER` | vitaslop-platform/src/gpu.rs:145 | The output of a diagnostic whose own KNOB is already the gate. |
| `VITASLOP_GXP_INPUTS_SETS` | vitaslop-platform/src/gpu.rs:9933 | - |
| `VITASLOP_GXP_INPUTS_VERTS` | vitaslop-platform/src/gpu.rs:9789 | Diagnostic (`VITASLOP_GXP_INPUTS=<hex-key>[,<hex-key>]` or `=all`): print, ONCE per |
| `VITASLOP_GXP_INTERP` | vitaslop-platform/src/gpu.rs:14342 | - |
| `VITASLOP_GXP_KEYCOLOR` | vitaslop-platform/src/gpu.rs:2873 | Whether the once-per-pair `gxp pair <key>: vprog hash ..., fprog hash ...` INDEX should be |
| `VITASLOP_GXP_KEYS` | vitaslop-platform/src/gpu.rs:6585 | Diagnostic (`VITASLOP_GXP_KEYS=<hex>,<hex>`): recompile ONLY these shader-pair keys |
| `VITASLOP_GXP_LIVE` | vitaslop-platform/src/gpu.rs:1564 | The guest's real vertex+fragment shaders + their draw inputs, for the GXP->WGSL |
| `VITASLOP_GXP_MAD_MASK16` | vitaslop-gxp-shader/src/usse/decode.rs:1597 | The mad-group destination write mask: a BITMASK over the destination's REGISTER LANES. |
| `VITASLOP_GXP_MEM_OFFSET16` | vitaslop-gxp-shader/src/link.rs:3458 | `0` reads a memory load's REGISTER offset full-width instead of as 16 bits - see |
| `VITASLOP_GXP_MEM_PEEK` | vitaslop-runtime/src/host.rs:14064 | `VITASLOP_MEM_DUMP=<hex addr>:<bytes>[,...]`: write those guest byte ranges to |
| `VITASLOP_GXP_MIPS` | vitaslop-platform/src/gpu.rs:3144 | Whether a chain is built for a seam, ignoring the per-texture exception above. |
| `VITASLOP_GXP_NEGW` | vitaslop-platform/src/gpu.rs:6741 | How to choose the clip-`w` sign correction (`VITASLOP_GXP_NEGW`). |
| `VITASLOP_GXP_NOBLEND` | vitaslop-platform/src/gpu.rs:6575 | Diagnostic (`VITASLOP_GXP_NOBLEND`): force every recompiled pipeline to REPLACE with |
| `VITASLOP_GXP_NODEPTH` | vitaslop-platform/src/gpu.rs:6567 | Diagnostic (`VITASLOP_GXP_NODEPTH`): every recompiled draw keeps its real shading and |
| `VITASLOP_GXP_ONLY` | vitaslop-platform/src/gpu.rs:6547 | Render ONLY recompiled draws, skipping the fixed-function draw for any call that |
| `VITASLOP_GXP_PACK_INTERNAL` | vitaslop-gxp-shader/src/usse/decode.rs:3628 | Where each operand of `word` sits for repeat purposes: the destination, then each source in |
| `VITASLOP_GXP_PAIR` | vitaslop-gxp-shader/tests/corpus.rs:839 | Link one named (vertex, fragment) pair and print the COMPLETE WGSL module both stages become. |
| `VITASLOP_GXP_PAIRS` | vitaslop-gxp-shader/tests/corpus.rs:2169 | >>> THE REFUTATION ABOVE WAS MEASURED OVER THE WRONG POPULATION. |
| `VITASLOP_GXP_PAIR_CORPUS` | vitaslop-gxp-shader/tests/corpus.rs:4986 | Link every `<key>.vert.gxp` / `<key>.frag.gxp` PAIR in a directory and rank what stops them. |
| `VITASLOP_GXP_PASS_SPLIT_EVERY` | vitaslop-platform/src/gpu.rs:614 | `VITASLOP_GXP_PASS_SPLIT_EVERY=<n>` cuts a render pass every `n` draws, with no shader |
| `VITASLOP_GXP_PRECOMPILE` | vitaslop-platform/src/gpu.rs:788 | Whether a shader pair the guest's patcher names is compiled AHEAD of the draw that binds it |
| `VITASLOP_GXP_PRECOMPILE_CROSS` | vitaslop-runtime/src/host.rs:14318 | `VITASLOP_GXP_PRECOMPILE_CROSS`: for a title whose `sceGxmShaderPatcherCreateFragmentProgram` |
| `VITASLOP_GXP_PREFETCH_CLAIM` | vitaslop-gxp-shader/src/link.rs:1696 | >>> A PREFETCH COORDINATE THAT LANDS ON LANES THE VERTEX NEVER WRITES IS RE-POINTED TO THE |
| `VITASLOP_GXP_PROBE` | vitaslop-gxp-shader/src/module.rs:359 | Diagnostic (`VITASLOP_GXP_PROBE=<bank><idx>[@<instr>][:f32/:bits=<hex>]`, e.g. |
| `VITASLOP_GXP_PROBE_SCALE` | vitaslop-gxp-shader/src/module.rs:518 | The `vec4<f32>` expression that reads the final colour out of register-file array `bank`, |
| `VITASLOP_GXP_QUADS` | vitaslop-platform/src/gpu.rs:9786 | Diagnostic (`VITASLOP_GXP_INPUTS=<hex-key>[,<hex-key>]` or `=all`): print, ONCE per |
| `VITASLOP_GXP_REAL_PAIRS` | vitaslop-gxp-shader/tests/corpus.rs:2167 | >>> THE REFUTATION ABOVE WAS MEASURED OVER THE WRONG POPULATION. |
| `VITASLOP_GXP_RECOMPILE` | vitaslop-runtime/src/host.rs:15261 | - |
| `VITASLOP_GXP_SA` | vitaslop-platform/src/gpu.rs:9680 | Diagnostic (`VITASLOP_GXP_FMEM=<lane>=<value>` / `<lane>*<factor>`, comma separated): |
| `VITASLOP_GXP_SA_DIRECT` | vitaslop-gxp-shader/src/link.rs:3455 | `0` restores the SA copy loop, `unroll` the constant-subscript copy - see [`resolve_sa_init`]. |
| `VITASLOP_GXP_SA_LITERAL_ALWAYS` | vitaslop-gxp-shader/src/link.rs:2344 | Is a container literal laid down for EVERY read that names its register |
| `VITASLOP_GXP_SA_UNCLAIMED` | vitaslop-gxp-shader/src/link.rs:2298 | Validate that every SA register a stage reads is either inside its default uniform buffer, |
| `VITASLOP_GXP_SIZE_BANKS` | vitaslop-gxp-shader/src/link.rs:3394 | `VITASLOP_GXP_SIZE_BANKS=0` restores the pre-2026-08-20b emission - every register bank |
| `VITASLOP_GXP_SOLID` | vitaslop-platform/src/gpu.rs:6563 | Diagnostic (`VITASLOP_GXP_SOLID`): every recompiled draw outputs solid magenta with |
| `VITASLOP_GXP_STRICT` | vitaslop-platform/src/gpu.rs:13465 | >>> A PAIR THIS RECOMPILER CANNOT TRANSLATE DROPS ITS DRAWS. |
| `VITASLOP_GXP_VARYING_LAYOUT` | vitaslop-gxp-shader/src/link.rs:1567 | Diagnostic (`VITASLOP_GXP_VARYING_LAYOUT=<vhash>:<usage>@<lane>x<comps>,...`): plan ONE |
| `VITASLOP_GXP_VARYING_ORDER` | vitaslop-gxp-shader/src/link.rs:1021 | The vertex lane order the paired FRAGMENT's declaration implies, or `None` when the two |
| `VITASLOP_GXP_VARYING_RESOLVE` | vitaslop-gxp-shader/tests/corpus.rs:4623 | Which VERTEX programs the forwarding resolver's lane RESERVATION moves, and where to. |
| `VITASLOP_GXP_VERT` | vitaslop-gxp-shader/tests/corpus.rs:4548 | The complete LINKED WGSL for one pair, selected by `VITASLOP_GXP_VERT` + `VITASLOP_GXP_FRAG` |
| `VITASLOP_GXP_VERTEX_PASSTHROUGH` | vitaslop-platform/src/gpu.rs:761 | `VITASLOP_GXP_VERTEX_PASSTHROUGH=0` makes every recompiled draw repack its vertex stream into |
| `VITASLOP_GXP_VPROBE` | vitaslop-gxp-shader/src/module.rs:497 | The `vec4<f32>` expression that reads the final colour out of register-file array `bank`, |
| `VITASLOP_GXP_VP_TRACE` | vitaslop-platform/src/gpu.rs:686 | Diagnostic (`VITASLOP_GXP_VP_TRACE`): report the viewport rectangle every recompiled draw is |
| `VITASLOP_GXP_WGSL_DIR` | vitaslop-gxp-shader/tests/corpus.rs:4551 | The complete LINKED WGSL for one pair, selected by `VITASLOP_GXP_VERT` + `VITASLOP_GXP_FRAG` |
| `VITASLOP_GXP_YFLIP` | vitaslop-platform/src/gpu.rs:6553 | Flip clip Y (`VITASLOP_GXP_YFLIP`, default off). |
| `VITASLOP_GXP_ZFIX` | vitaslop-platform/src/gpu.rs:6551 | Apply the GXM (GL-style, NDC z in [-1,1]) -> WebGPU (z in [0,1]) clip-depth remap |
| `VITASLOP_HB_CMP` | vitaslop-native/tests/homebrew_qsort.rs:114 | Block trace for one run: set `VITASLOP_TRACE_BLOCKS=<lo>-<hi>` (emit-time) and |
| `VITASLOP_HB_CMP_BIT` | vitaslop-native/tests/homebrew_qsort.rs:66 | - |
| `VITASLOP_HB_DUMP` | vitaslop-native/tests/homebrew_qsort.rs:47 | - |
| `VITASLOP_HB_IMAGE` | vitaslop-native/tests/homebrew_strings.rs:36 | A VM over the whole image (the routines read past string ends in aligned words, so |
| `VITASLOP_HB_N` | vitaslop-native/tests/homebrew_qsort.rs:98 | Block trace for one run: set `VITASLOP_TRACE_BLOCKS=<lo>-<hi>` (emit-time) and |
| `VITASLOP_HB_QSORT` | vitaslop-native/tests/homebrew_qsort.rs:114 | Block trace for one run: set `VITASLOP_TRACE_BLOCKS=<lo>-<hi>` (emit-time) and |
| `VITASLOP_HB_STRCMP` | vitaslop-native/tests/homebrew_strings.rs:8 | - |
| `VITASLOP_HB_STRLEN` | vitaslop-native/tests/homebrew_strings.rs:35 | A VM over the whole image (the routines read past string ends in aligned words, so |
| `VITASLOP_HEADLESS_FRAMES` | vitaslop-desktop/src/retail.rs:971 | Headless self-check of the retail path (NO window): load `dir`, optionally drive a |
| `VITASLOP_HEADLESS_NO_TAPS` | vitaslop-desktop/src/retail.rs:977 | Headless self-check of the retail path (NO window): load `dir`, optionally drive a |
| `VITASLOP_HEADLESS_RECIPE` | vitaslop-desktop/src/retail.rs:974 | Headless self-check of the retail path (NO window): load `dir`, optionally drive a |
| `VITASLOP_HEADLESS_SHOT_EVERY` | vitaslop-desktop/src/retail.rs:980 | Headless self-check of the retail path (NO window): load `dir`, optionally drive a |
| `VITASLOP_HEADLESS_SHOT_FROM` | vitaslop-desktop/src/retail.rs:982 | Headless self-check of the retail path (NO window): load `dir`, optionally drive a |
| `VITASLOP_HEADLESS_SHOT_TO` | vitaslop-desktop/src/retail.rs:982 | Headless self-check of the retail path (NO window): load `dir`, optionally drive a |
| `VITASLOP_HEADLESS_TIMING` | vitaslop-desktop/src/retail.rs:978 | Headless self-check of the retail path (NO window): load `dir`, optionally drive a |
| `VITASLOP_HOLD_BUTTONS` | vitaslop-native/tests/retail_boot_probe.rs:96 | A minimal host world: a monotonic clock advancing one 60Hz tick per poll, no |
| `VITASLOP_HOLD_FROM` | vitaslop-native/tests/retail_boot_probe.rs:97 | A minimal host world: a monotonic clock advancing one 60Hz tick per poll, no |
| `VITASLOP_HOLD_MEM` | vitaslop-native/tests/retail_boot_probe.rs:714 | - |
| `VITASLOP_HOLD_TOUCH` | vitaslop-native/tests/retail_boot_probe.rs:114 | - |
| `VITASLOP_HOME` | vitaslop-desktop/src/library.rs:12 | - |
| `VITASLOP_HOSTCALL_WATCH` | vitaslop-runtime/src/vita/mod.rs:565 | `VITASLOP_HOSTCALL_WATCH=<hex addr>[,<hex addr>...]` - print every host call that passes one |
| `VITASLOP_HOST_WRITE_WATCH` | vitaslop-runtime/src/host.rs:573 | `VITASLOP_HOST_WRITE_WATCH=<hex addr>[,...]`: report every write a HOST CALL makes to one |
| `VITASLOP_INGEST_DEBUG` | vitaslop-runtime/src/ingest/filesdb.rs:172 | Resolve every non-directory node to its full '/'-separated path (no |
| `VITASLOP_INPUT_RECIPE` | vitaslop-native/tests/retail_boot_probe.rs:388 | - |
| `VITASLOP_IO_BANDWIDTH_KIBPS` | vitaslop-runtime/src/vita/iofilemgr.rs:22 | Modelled sequential read bandwidth, in KiB per second |
| `VITASLOP_IO_PARK_THRESHOLD_US` | vitaslop-runtime/src/vita/iofilemgr.rs:110 | Smallest debt worth a context switch, in microseconds |
| `VITASLOP_IO_REQUEST_US` | vitaslop-runtime/src/vita/iofilemgr.rs:81 | Fixed per-request cost in microseconds (`VITASLOP_IO_REQUEST_US`): the command |
| `VITASLOP_JPEG_BENCH` | vitaslop-runtime/src/vita/jpeg.rs:532 | What this decoder costs, on a real image, so "is it fast enough" is a number. |
| `VITASLOP_LOG` | vitaslop-desktop/src/log.rs:30 | Whether captured events are also written to stderr. |
| `VITASLOP_MAX_FRAMES` | vitaslop-native/tests/retail_boot_probe.rs:511 | - |
| `VITASLOP_MAX_ROUNDS` | vitaslop-native/tests/retail_boot_probe.rs:516 | - |
| `VITASLOP_MEM_DUMP` | vitaslop-runtime/src/host.rs:14057 | `VITASLOP_MEM_DUMP=<hex addr>:<bytes>[,...]`: write those guest byte ranges to |
| `VITASLOP_MEM_DUMP_AT` | vitaslop-runtime/src/host.rs:14059 | `VITASLOP_MEM_DUMP=<hex addr>:<bytes>[,...]`: write those guest byte ranges to |
| `VITASLOP_MOVIE` | vitaslop-runtime/src/vita/video.rs:1964 | A track whose codec this engine does not decode is not offered at all: the title's |
| `VITASLOP_MOVIE_DUMP_DIR` | vitaslop-runtime/src/vita/avcdec.rs:1190 | >>> AND WHAT THE PICTURE ACTUALLY LOOKS LIKE, because "a picture arrived" and "the movie |
| `VITASLOP_MOVIE_DUMP_EVERY` | vitaslop-runtime/src/vita/avcdec.rs:1190 | >>> AND WHAT THE PICTURE ACTUALLY LOOKS LIKE, because "a picture arrived" and "the movie |
| `VITASLOP_MOVIE_PICTURE_HASH` | vitaslop-runtime/src/vita/avcdec.rs:168 | Pictures handed to the guest so far, which is what `VITASLOP_MOVIE_PICTURE_HASH` |
| `VITASLOP_MOVIE_SUBSTITUTE` | vitaslop-runtime/src/vita/video.rs:150 | >>> OPEN A DIFFERENT MOVIE THAN THE TITLE ASKED FOR |
| `VITASLOP_MP4_AUDIO` | vitaslop-runtime/src/vita/video.rs:1215 | The tracks this engine will hand units for, as cursors, in the order they appear in the |
| `VITASLOP_MP4_UNITS` | vitaslop-runtime/src/vita/video.rs:1321 | `VITASLOP_MP4_UNITS=none`: never return an access unit. |
| `VITASLOP_NAME` | vitaslop-desktop/src/shell.rs:735 | - |
| `VITASLOP_NEON_CACHE` | vitaslop-transpiler/src/emit.rs:2654 | Whether emitted modules hold the low NEON bank in locals across a run of vector |
| `VITASLOP_NGS_NEG_LOOP` | vitaslop-runtime/src/vita/at9.rs:1339 | A NEGATIVE `nLoopCount` means "play this buffer once and move on", not "repeat it |
| `VITASLOP_NGS_VOICE_HANDLE_MEMO` | vitaslop-runtime/src/vita/ngs.rs:406 | SceInt32 sceNgsRackGetVoiceHandle(SceNgsHRack rack, SceUInt32 index, SceNgsHVoice *handle) |
| `VITASLOP_NGS_ZERO_LEVEL` | vitaslop-runtime/src/vita/at9.rs:1345 | A NEGATIVE `nLoopCount` means "play this buffer once and move on", not "repeat it |
| `VITASLOP_NID_DIGEST` | vitaslop-runtime/src/host.rs:16757 | The cross-engine host-call digest for the frame in progress - see [`NidDigest`]. |
| `VITASLOP_NO_BC` | vitaslop-runtime/src/render.rs:1971 | Decode a whole BC1/BC2/BC3 block to its sixteen RGBA8 texels at once. |
| `VITASLOP_NO_FAST_IMPORT` | vitaslop-runtime/src/vita/mod.rs:141 | Whether `func_nid`'s handler can only ever CONTINUE, so the transpiler may route the |
| `VITASLOP_NO_INLINE_CLIB` | vitaslop-runtime/src/vita/mod.rs:270 | `VITASLOP_NO_INLINE_CLIB`: route `sceClibMemcpy`, `sceClibMemset` and `sceClibMemcmp` |
| `VITASLOP_NO_INLINE_IMPORTS` | vitaslop-runtime/src/host.rs:6182 | >>> WHO HAS WRITTEN THE CONTEXT'S TEXTURE SLOTS, counted for the failure report above. |
| `VITASLOP_NO_INLINE_LWMUTEX` | vitaslop-runtime/src/vita/mod.rs:372 | `VITASLOP_NO_INLINE_LWMUTEX`: route the lightweight-mutex lock and unlock through the |
| `VITASLOP_NO_INLINE_MUTEX` | vitaslop-runtime/src/vita/mod.rs:392 | `VITASLOP_NO_INLINE_MUTEX`: route `sceKernelLockMutex`/`sceKernelUnlockMutex` through the |
| `VITASLOP_NO_INLINE_RESERVE` | vitaslop-runtime/src/vita/mod.rs:307 | `VITASLOP_NO_INLINE_RESERVE`: route `sceGxmReserve{Vertex,Fragment}DefaultUniformBuffer` |
| `VITASLOP_NO_INLINE_STUBS` | vitaslop-runtime/src/vita/mod.rs:244 | `VITASLOP_NO_INLINE_STUBS`: route the constant-return stubs through the host, leaving |
| `VITASLOP_NO_INLINE_TEXTURE` | vitaslop-runtime/src/host.rs:6182 | >>> WHO HAS WRITTEN THE CONTEXT'S TEXTURE SLOTS, counted for the failure report above. |
| `VITASLOP_NO_INLINE_UNIFORM_DATA` | vitaslop-runtime/src/vita/mod.rs:341 | `VITASLOP_NO_INLINE_UNIFORM_DATA`: route `sceGxmSetUniformDataF` through the host, |
| `VITASLOP_NO_NGS_MIX` | vitaslop-runtime/src/vita/audio.rs:291 | `VITASLOP_NO_NGS_MIX`: skip the NGS decode-and-mix entirely, leaving the guest's |
| `VITASLOP_PACKED_CACHE_MB` | vitaslop-platform/src/gpu.rs:3186 | >>> AND A CAP IN ENTRIES IS NOT A BOUND ON MEMORY. |
| `VITASLOP_PATCH_STUBS` | vitaslop-native/tests/retail_boot_probe.rs:466 | - |
| `VITASLOP_PAUSE_ON_BLUR` | vitaslop-desktop/src/retail.rs:1854 | Run the retail title in `dir` in a live window until the window closes or the guest |
| `VITASLOP_PEEK` | vitaslop-desktop/src/retail.rs:659 | Guest memory at `addr`, for `VITASLOP_PEEK`. |
| `VITASLOP_PERF` | vitaslop-native/src/perf.rs:43 | Is perf accounting on (`VITASLOP_PERF` set)? Read once and cached. |
| `VITASLOP_PERF_CONSOLE` | vitaslop-web/src/lib.rs:1695 | Whether the per-window performance report is also written to the browser CONSOLE |
| `VITASLOP_PIXEL_TRACE` | vitaslop-runtime/src/render.rs:5080 | Draw one scene onto an EXISTING framebuffer and depth buffer, composing with whatever |
| `VITASLOP_PKG` | vitaslop-runtime/src/ingest/stream.rs:1003 | A pkg's item table, read over a source that only ever hands out RANGES |
| `VITASLOP_POISON_UNRESOLVED_VARS` | vitaslop-runtime/src/link.rs:318 | - |
| `VITASLOP_POKE` | vitaslop-native/tests/retail_boot_probe.rs:682 | - |
| `VITASLOP_POLL_ADDR` | vitaslop-native/src/threaded.rs:1805 | Guest address to sample after each host call, from `VITASLOP_POLL_ADDR` (hex). |
| `VITASLOP_PREPARE_SPLIT` | vitaslop-platform/src/gpu.rs:5561 | Where the milliseconds INSIDE one `prepare` go, plus the bytes each phase moved. |
| `VITASLOP_PREPOKE` | vitaslop-native/tests/retail_boot_probe.rs:489 | - |
| `VITASLOP_PRESENT_PROBE` | vitaslop-web/src/lib.rs:748 | Reads back WHAT WE PRESENTED, when `VITASLOP_PRESENT_PROBE` asks for it. |
| `VITASLOP_PROMOTE_POISON` | vitaslop-transpiler/src/emit.rs:2698 | `VITASLOP_PROMOTE_POISON=<n>` - the FALSIFIER for register promotion. |
| `VITASLOP_PROMOTE_REGS` | vitaslop-native/src/threaded.rs:2373 | A concise trap description (kind + message), matching the sync `Vm`'s detail. |
| `VITASLOP_PSARC` | vitaslop-runtime/src/psarc.rs:430 | Read a REAL archive: `VITASLOP_PSARC=<path to a .psarc>`, optionally |
| `VITASLOP_PSARC_FILE` | vitaslop-runtime/src/psarc.rs:431 | Read a REAL archive: `VITASLOP_PSARC=<path to a .psarc>`, optionally |
| `VITASLOP_PVRTC_DECODE` | vitaslop-runtime/src/render.rs:6413 | Whether PVRTC decodes a whole face at a time (the default) or one texel at a time. |
| `VITASLOP_QUANTUM_CPU_US` | vitaslop-runtime/src/host.rs:17528 | Game-clock time charged for one [`QUANTUM_ARM`] of guest execution, in microseconds. |
| `VITASLOP_QUANTUM_FUEL` | vitaslop-native/tests/retail_boot_probe.rs:425 | - |
| `VITASLOP_REGION_CLIP_SCENE` | vitaslop-runtime/src/vita/gxm.rs:1641 | - |
| `VITASLOP_REGTRACE` | vitaslop-native/src/threaded.rs:1447 | `VITASLOP_REGTRACE=<lo>-<hi>:<path>` - append the reg+flag file per block entry in |
| `VITASLOP_REGTRACE_MAX` | vitaslop-native/src/threaded.rs:1606 | `VITASLOP_REGTRACE_MAX=<n>` caps the register trace at `n` lines (0 = unbounded). |
| `VITASLOP_REGTRACE_WATCH` | vitaslop-native/src/threaded.rs:1384 | The `VITASLOP_REGTRACE_WATCH` words, formatted as ` mADDR=VALUE` fields ready to append |
| `VITASLOP_RESIDENT_GEOM` | vitaslop-platform/src/gpu.rs:6813 | Repacked vertices and expanded indices that have not changed since the renderer first |
| `VITASLOP_RESIDENT_GEOM_MB` | vitaslop-platform/src/gpu.rs:6856 | The byte budget for each of the two heaps (`VITASLOP_RESIDENT_GEOM_MB`, per heap). |
| `VITASLOP_ROUNDS_PER_FRAME` | vitaslop-native/tests/retail_boot_probe.rs:660 | - |
| `VITASLOP_RTT_BG_CACHE` | vitaslop-platform/src/gpu.rs:592 | `VITASLOP_RTT_BG_CACHE=0` restores the OLD behaviour: a sampler bind group naming a render |
| `VITASLOP_SAMPLER_NARROW` | vitaslop-runtime/src/host.rs:2920 | Whether a draw decodes only the texture units its fragment program DECLARES - see |
| `VITASLOP_SCAN_WORD` | vitaslop-native/tests/retail_boot_probe.rs:962 | - |
| `VITASLOP_SCENE_LIMIT` | vitaslop-native/tests/retail_boot_probe.rs:446 | - |
| `VITASLOP_SCHED_CORES` | vitaslop-runtime/src/sched.rs:627 | `VITASLOP_SCHED_CORES=<n>`: cap the baton to the top `n` runnable PRIORITIES, as the |
| `VITASLOP_SCHED_RR` | vitaslop-runtime/src/sched.rs:643 | `VITASLOP_SCHED_RR=1`: round-robin every runnable thread, ignoring priority. |
| `VITASLOP_SCHED_TRACE` | vitaslop-runtime/src/sched.rs:653 | `VITASLOP_SCHED_TRACE=<from>-<to>` (display frames, inclusive): print one line per |
| `VITASLOP_SET_EVF` | vitaslop-native/tests/retail_boot_probe.rs:692 | - |
| `VITASLOP_SHOT_DIR` | vitaslop-native/tests/retail_boot_probe.rs:207 | Read and format one watched value from current guest memory. |
| `VITASLOP_SHOT_LAST` | vitaslop-native/tests/retail_boot_probe.rs:444 | - |
| `VITASLOP_SIGNATURE` | vitaslop-native/src/recipe_runner.rs:101 | The determinism signature over the observable output (render stream + egress), |
| `VITASLOP_SIGNATURE_EVERY` | vitaslop-native/src/recipe_runner.rs:475 | `VITASLOP_SIGNATURE_EVERY=<n>`: print the RUNNING determinism signature every `n` stepped |
| `VITASLOP_SLOW_FRAME_US` | vitaslop-web/src/lib.rs:4304 | - |
| `VITASLOP_SNAPSHOT` | vitaslop-native/src/threaded.rs:1433 | `VITASLOP_SNAPSHOT=<hexpc>:<path>` - dump full state on first entry to block `hexpc`. |
| `VITASLOP_SNAPSHOT_BUDGET_MB` | vitaslop-runtime/src/host.rs:3377 | Byte budget for retained texture snapshots, scaled to the device |
| `VITASLOP_SNAPSHOT_DENSE` | vitaslop-native/src/threaded.rs:1543 | Dump the full guest state (all non-zero pages + r0..r15 + NZCV) to `path`, in the |
| `VITASLOP_SNAPSHOT_SKIP` | vitaslop-native/src/threaded.rs:1495 | `VITASLOP_SNAPSHOT_SKIP=<n>` - skip the first `n` entries to the snapshot block before |
| `VITASLOP_SOFTWARE` | vitaslop-desktop/src/retail.rs:1739 | - |
| `VITASLOP_SSAA` | vitaslop-platform/src/gpu.rs:15499 | Set the supersample factor: 1 (default) renders the scene straight into the caller's |
| `VITASLOP_STALL_CHUNK` | vitaslop-native/tests/retail_boot_probe.rs:543 | - |
| `VITASLOP_STALL_WAKE` | vitaslop-native/tests/retail_boot_probe.rs:542 | - |
| `VITASLOP_STALL_WATCHDOG` | vitaslop-native/src/watchdog.rs:99 | The configured stall budget in seconds, from `VITASLOP_STALL_WATCHDOG`. |
| `VITASLOP_STALL_WAVES` | vitaslop-native/tests/retail_boot_probe.rs:546 | - |
| `VITASLOP_STRICT_DRAWS` | vitaslop-runtime/src/render.rs:6120 | Why [`RenderSceneBuilder::build`] discarded draws from a captured scene. |
| `VITASLOP_SWITCH_WHY` | vitaslop-transpiler/src/lower.rs:1069 | Whether the table-branch diagnostic is on for this address |
| `VITASLOP_SW_CHAIN` | vitaslop-native/src/wgpu_render.rs:561 | `VITASLOP_GPU_CHAIN_DIR=<dir>`: write every offscreen target of the frame just |
| `VITASLOP_SW_CHAIN_DIR` | vitaslop-runtime/src/render.rs:4897 | - |
| `VITASLOP_SW_POST` | vitaslop-runtime/src/render.rs:4944 | - |
| `VITASLOP_SYSTEM_FONT` | vitaslop-runtime/src/font/system.rs:75 | The resolved substitute: its bytes and a human-readable account of where they came from. |
| `VITASLOP_TEXTURE_CHECK` | vitaslop-runtime/src/host.rs:3128 | How a retained texture snapshot is re-validated (`VITASLOP_TEXTURE_CHECK`): `scene` |
| `VITASLOP_TEX_CACHE_MB` | vitaslop-platform/src/gpu.rs:566 | The texture-cache budget in bytes: [`GAME_RESIDENT_CEILING_MB`] unless |
| `VITASLOP_TEX_COMPRESS` | vitaslop-runtime/src/render.rs:1347 | Whether compressed textures reach the GPU compressed at all. |
| `VITASLOP_TEX_DIRTY_CENSUS` | vitaslop-runtime/src/host.rs:172 | >>> WHICH PARTS of `[off, off + len)` the guest may have stored into since `stamp`, |
| `VITASLOP_TEX_MEMO_PER_SCENE` | vitaslop-runtime/src/host.rs:2683 | A whole DRAW's worth of snapshotted textures, by the bindings that produced it - kept |
| `VITASLOP_TEX_PAGE_READ` | vitaslop-runtime/src/host.rs:3720 | Record that this entry's bytes are current as of THIS SCENE, so a later |
| `VITASLOP_TRACE_BLOCKS` | vitaslop-native/src/threaded.rs:1359 | `VITASLOP_TRACE_FRAMES=<from>-<to>` (decimal display frames, inclusive) - print the |
| `VITASLOP_TRACE_EXIT` | vitaslop-native/tests/retail_boot_probe.rs:37 | - |
| `VITASLOP_TRACE_FILE` | vitaslop-runtime/src/vita/libkernel.rs:47 | Diagnostic (`RUST_LOG=vitaslop::exit=debug`): when the guest calls |
| `VITASLOP_TRACE_FRAMES` | vitaslop-native/src/threaded.rs:1356 | `VITASLOP_TRACE_FRAMES=<from>-<to>` (decimal display frames, inclusive) - print the |
| `VITASLOP_TRACE_FUNCS` | vitaslop-native/src/threaded.rs:1273 | Bind `env.svc`. |
| `VITASLOP_TRACE_INDIRECT` | vitaslop-transpiler/src/emit.rs:1573 | Diagnostic indirect-call tracer. |
| `VITASLOP_TRACE_IO` | vitaslop-native/tests/retail_boot_probe.rs:34 | - |
| `VITASLOP_TRACE_ORDER` | vitaslop-runtime/src/vita/mod.rs:549 | Ordered-timeline trace (env `VITASLOP_TRACE_ORDER`): print every *meaningful* |
| `VITASLOP_TRACK_PC` | vitaslop-transpiler/src/abi.rs:200 | Exported name of the diagnostic guest-PC tracker global. |
| `VITASLOP_TRANSPILE_REPORT` | vitaslop-native/src/threaded.rs:885 | - |
| `VITASLOP_TRAP_HALT` | vitaslop-transpiler/src/emit.rs:1686 | When `VITASLOP_TRAP_HALT` is set, a `Term::Halt` (a block that ran off the end of decoded |
| `VITASLOP_UNIFORM_WATCH` | vitaslop-runtime/src/vita/gxm.rs:2015 | `VITASLOP_UNIFORM_WATCH=<hex address>/<parameter name substring>[,...]`: report every |
| `VITASLOP_UV_DEBUG` | vitaslop-runtime/src/render.rs:5094 | Draw one scene onto an EXISTING framebuffer and depth buffer, composing with whatever |
| `VITASLOP_VBLANK_PARK` | vitaslop-runtime/src/vita/display.rs:201 | Whether an inlined `sceDisplayGetVcount` carries the spin guard (`VITASLOP_VBLANK_PARK`, |
| `VITASLOP_VERTEX_INTERN` | vitaslop-runtime/src/host.rs:2905 | A cheap, allocation-free fingerprint of a vertex stream, for [`TextureSnapshots:: |
| `VITASLOP_VPK` | vitaslop-runtime/src/ingest/stream.rs:1057 | A homebrew VPK (`VITASLOP_VPK=<file>`): probed as `vpk`, imported as files + a |
| `VITASLOP_WASM_INDICES` | vitaslop-native/src/threaded.rs:2427 | Rewrite `<wasm function N>` in a trap backtrace to name the GUEST function it is. |
| `VITASLOP_WASM_NAMES` | vitaslop-transpiler/src/emit.rs:1139 | When `VITASLOP_WASM_NAMES` is set, emit a wasm `name` custom section labelling |
| `VITASLOP_WATCH_` | vitaslop-transpiler/src/emit.rs:1485 | Number of matching store-watchpoint hits to skip before trapping (`VITASLOP_WATCH_ |
| `VITASLOP_WATCH_FROM` | vitaslop-native/tests/retail_boot_probe.rs:672 | - |
| `VITASLOP_WATCH_MEM` | vitaslop-native/tests/retail_boot_probe.rs:139 | Parse `VITASLOP_WATCH_MEM=addr:type:label,addr:type:label,...` into watches. |
| `VITASLOP_WATCH_READ` | vitaslop-transpiler/src/emit.rs:1111 | Diagnostic read watchpoint. |
| `VITASLOP_WATCH_READ_` | vitaslop-transpiler/src/emit.rs:1511 | Optional guest-PC EXCLUDE window for the read watchpoint (`VITASLOP_WATCH_READ_ |
| `VITASLOP_WATCH_READ_NZ` | vitaslop-transpiler/src/emit.rs:3090 | Emit the read-watchpoint trap check. |
| `VITASLOP_WATCH_READ_PC_EXCL` | vitaslop-transpiler/src/emit.rs:1521 | Optional guest-PC EXCLUDE window for the read watchpoint (`VITASLOP_WATCH_READ_ |
| `VITASLOP_WATCH_READ_SKIP` | vitaslop-transpiler/src/emit.rs:1225 | WASM global index of the read-watchpoint match counter, appended after the guest-PC |
| `VITASLOP_WATCH_STORE` | vitaslop-native/src/threaded.rs:1468 | `VITASLOP_REGTRACE_WATCH=<hex guest addr>[,<hex guest addr>...]` - append the WORD |
| `VITASLOP_WATCH_STORE_ARM` | vitaslop-transpiler/src/emit.rs:908 | The emit-time knobs [`set_emit_knob`] accepts, so a caller that forwards a whole table |
| `VITASLOP_WATCH_STORE_LOG` | vitaslop-runtime/src/capture.rs:528 | GUEST ADDRESS the bytes above were read from, or 0 when there is no bound buffer. |
| `VITASLOP_WATCH_STORE_MODE` | vitaslop-transpiler/src/emit.rs:1701 | Store-watchpoint mode, from `VITASLOP_WATCH_STORE_MODE` (default `any`): |
| `VITASLOP_WATCH_STORE_NZ` | vitaslop-transpiler/src/emit.rs:1724 | `VITASLOP_WATCH_STORE_LOG` - LOG each store to the watched address (the storing |
| `VITASLOP_WATCH_STORE_SKIP` | vitaslop-transpiler/src/emit.rs:602 | Linear-memory byte offset of the store-watchpoint MATCH COUNTER, or 0 when this |
| `VITASLOP_X` | vitaslop-frontend/src/settings.rs:243 | Parse a `NAME=VALUE` per line knobs box into a map. |
| `VITASLOP_XML_DUMP` | vitaslop-runtime/src/vita/sce_xml.rs:580 | `VITASLOP_XML_DUMP=<dir>`: write every document handed to `parse` into `<dir>` as |
| `VITASLOP_Y` | vitaslop-frontend/src/settings.rs:244 | Parse a `NAME=VALUE` per line knobs box into a map. |
