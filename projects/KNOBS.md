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

270 knobs.

| knob | read in | what it does |
|---|---|---|
| `VITASLOP_AAC_DIR` | vitaslop-aac/tests/oracle.rs:69 | The `AudioSpecificConfig` an ADTS header describes: 5 bits of object type, 4 of |
| `VITASLOP_ALLOW_SOFTWARE_GPU` | vitaslop-web/src/lib.rs:1147 | Whether a run may proceed on a software rasteriser (`VITASLOP_ALLOW_SOFTWARE_GPU`). |
| `VITASLOP_ARM_AT_FRAME` | vitaslop-native/src/threaded.rs:323 | Linear-memory offset of the "diagnostics armed" word, when this build was |
| `VITASLOP_AT9_DIR` | vitaslop-atrac9/tests/oracle.rs:66 | Decode a whole AT9 payload the way a superframe consumer does: for each |
| `VITASLOP_AUDIO_RAW` | vitaslop-runtime/src/vita/audio.rs:80 | Optional raw-s16le capture of the mixed output stream (env |
| `VITASLOP_BACKTRACE` | vitaslop-runtime/src/vita/mod.rs:491 | Print the guest call chain the first time a chosen NID is called from each thread |
| `VITASLOP_BLOCK_HIST` | vitaslop-native/src/recipe_runner.rs:167 | Dump the per-PC block-entry histogram gathered under `VITASLOP_BLOCK_HIST`, for |
| `VITASLOP_BLOCK_HIST_SEQ` | vitaslop-native/src/threaded.rs:1661 | Print the block-visit histogram gathered under `VITASLOP_BLOCK_HIST`: the `top` |
| `VITASLOP_BROWSER_FASTFORWARD` | vitaslop-web/src/lib.rs:1127 | Frame to fast-forward the live loop to (`VITASLOP_BROWSER_FASTFORWARD`), unpaced. |
| `VITASLOP_BROWSER_FUEL` | vitaslop-web/src/browser_sched.rs:920 | Guest work a thread may execute before the browser preempts it, in WASMTIME FUEL UNITS |
| `VITASLOP_BROWSER_HEARTBEAT_MS` | vitaslop-web/src/lib.rs:3588 | - |
| `VITASLOP_BROWSER_INSTANCE_POOL` | vitaslop-web/src/browser_sched.rs:1569 | Whether a finished thread's module instance may be REUSED by the next thread |
| `VITASLOP_BROWSER_QUANTUM_CALLS` | vitaslop-web/src/browser_sched.rs:121 | Host calls one guest thread may make before the browser preempts it |
| `VITASLOP_BROWSER_SUPERSAMPLE` | vitaslop-web/src/lib.rs:1104 | Supersample factor for the live browser render (`VITASLOP_BROWSER_SUPERSAMPLE`). |
| `VITASLOP_CALLSITES_WINDOW` | vitaslop-desktop/src/retail.rs:461 | Where the idle clock went SINCE `before` - the windowed reading. |
| `VITASLOP_CAPSULE_DUMP_SA` | vitaslop-native/examples/capsule-replay.rs:29 | `VITASLOP_CAPSULE_DUMP_SA=1`: print this draw's uniform banks - `frag_sa` with the GUEST |
| `VITASLOP_CHAIN_DRAWS` | vitaslop-platform/src/gpu.rs:15183 | - |
| `VITASLOP_CHAIN_LIMIT` | vitaslop-native/tests/gpu_rtt_gamma.rs:177 | Render a chain of `feedback` sample-and-write-back passes over the offscreen target and |
| `VITASLOP_CHAIN_SKIP` | vitaslop-native/src/wgpu_render.rs:332 | - |
| `VITASLOP_CHECK_ADDRS` | vitaslop-native/tests/retail_boot_probe.rs:43 | - |
| `VITASLOP_CLOCK_TRACE` | vitaslop-runtime/src/sched.rs:1089 | - |
| `VITASLOP_CODE_RANGE` | vitaslop-runtime/src/vita/mod.rs:474 | The guest code range scanned for the game-level caller in [`dispatch`] (env |
| `VITASLOP_CONSOLE` | vitaslop-web/src/logging.rs:309 | `VITASLOP_CONSOLE=1`: mirror the run's status notes - the setup summary, the adapter and |
| `VITASLOP_CPU_SHARE` | vitaslop-native/src/recipe_runner.rs:111 | Who actually got the CPU over the run, when `VITASLOP_CPU_SHARE` is set - see |
| `VITASLOP_DBG_CALLSITES` | vitaslop-runtime/src/vita/mod.rs:432 | Diagnostic call-site profiler (`VITASLOP_DBG_CALLSITES`): counts host calls |
| `VITASLOP_DEBUG_CAPTURE` | vitaslop-web/src/lib.rs:3650 | - |
| `VITASLOP_DECODE_CACHE_MB` | vitaslop-runtime/src/render.rs:6303 | Budget for the decode cache, in BYTES of decoded RGBA8, before it is cleared wholesale. |
| `VITASLOP_DIRTY_PAGES` | vitaslop-native/src/threaded.rs:128 | Linear-memory offset of the guest-store dirty block, when this build was |
| `VITASLOP_DISPATCH_ALL` | vitaslop-transpiler/src/emit.rs:1300 | The ablation arm that prices a dispatch re-entry: `VITASLOP_DISPATCH_ALL=1` sends even a |
| `VITASLOP_DRAW_ONLY` | vitaslop-runtime/src/render.rs:5080 | Draw one scene onto an EXISTING framebuffer and depth buffer, composing with whatever |
| `VITASLOP_DRAW_RANGE` | vitaslop-platform/src/gpu.rs:500 | `VITASLOP_RTT_BG_CACHE=0` restores the OLD behaviour: a sampler bind group naming a render |
| `VITASLOP_DRAW_STATS` | vitaslop-runtime/src/render.rs:5069 | Draw one scene onto an EXISTING framebuffer and depth buffer, composing with whatever |
| `VITASLOP_DRV_KEY` | vitaslop-runtime/src/ingest/pfscrypt.rs:264 | `F00D(klicensee)` for the title, from `VITASLOP_DRV_KEY` (32 hex chars), or |
| `VITASLOP_DUMP_DIR` | vitaslop-runtime/src/ingest/pipeline.rs:827 | Diagnostic: decrypt the container and write named plaintext files out to |
| `VITASLOP_DUMP_DRAW` | vitaslop-native/tests/retail_boot_probe.rs:1292 | - |
| `VITASLOP_DUMP_DRAWS` | vitaslop-native/tests/retail_boot_probe.rs:1080 | - |
| `VITASLOP_DUMP_DRAW_GXP` | vitaslop-gxp-shader/tests/oracle.rs:657 | Correlate each captured vertex<->fragment PAIR (from a real draw run) to establish the |
| `VITASLOP_DUMP_DRAW_GXP_CAP` | vitaslop-runtime/src/host.rs:13803 | - |
| `VITASLOP_DUMP_DRAW_GXP_FULL` | vitaslop-runtime/src/host.rs:13909 | - |
| `VITASLOP_DUMP_EXPORTS` | vitaslop-runtime/src/link.rs:460 | - |
| `VITASLOP_DUMP_FILES` | vitaslop-runtime/src/ingest/pipeline.rs:829 | Diagnostic: decrypt the container and write named plaintext files out to |
| `VITASLOP_DUMP_FPROG` | vitaslop-runtime/src/host.rs:13691 | Diagnostic (VITASLOP_DUMP_FPROG): print the bound fragment program's sampler |
| `VITASLOP_DUMP_FUNC` | vitaslop-native/tests/retail_boot_probe.rs:319 | - |
| `VITASLOP_DUMP_GXP_BIN` | vitaslop-platform/src/gpu.rs:11368 | >>> A REFUSAL THAT DOES NOT HAND OVER THE EVIDENCE COSTS A PLAY SESSION. |
| `VITASLOP_DUMP_IMAGE` | vitaslop-native/tests/retail_boot_probe.rs:33 | - |
| `VITASLOP_DUMP_IMPORTS` | vitaslop-native/tests/retail_boot_probe.rs:355 | - |
| `VITASLOP_DUMP_MAP` | vitaslop-native/tests/retail_boot_probe.rs:591 | - |
| `VITASLOP_DUMP_MEM` | vitaslop-native/tests/retail_boot_probe.rs:43 | - |
| `VITASLOP_DUMP_PATHS` | vitaslop-native/tests/retail_boot_probe.rs:414 | - |
| `VITASLOP_DUMP_REGION` | vitaslop-native/tests/retail_boot_probe.rs:529 | - |
| `VITASLOP_DUMP_REGION_RANGE` | vitaslop-native/tests/retail_boot_probe.rs:531 | - |
| `VITASLOP_DUMP_RENDERSCENE` | vitaslop-native/tests/retail_boot_probe.rs:1334 | - |
| `VITASLOP_DUMP_SCENES` | vitaslop-desktop/src/retail.rs:339 | Step the guest one display frame. |
| `VITASLOP_DUMP_STDOUT` | vitaslop-desktop/src/retail.rs:1464 | - |
| `VITASLOP_DUMP_STUBS` | vitaslop-native/tests/retail_boot_probe.rs:32 | - |
| `VITASLOP_DUMP_TEX` | vitaslop-native/tests/retail_boot_probe.rs:1234 | - |
| `VITASLOP_DUMP_TEX_DIR` | vitaslop-runtime/src/host.rs:13915 | - |
| `VITASLOP_DUMP_TEX_MAX_TEXELS` | vitaslop-runtime/src/host.rs:13924 | - |
| `VITASLOP_DUMP_TRIS` | vitaslop-runtime/src/render.rs:5087 | Draw one scene onto an EXISTING framebuffer and depth buffer, composing with whatever |
| `VITASLOP_DUMP_VPROG` | vitaslop-runtime/src/host.rs:13498 | Diagnostic (VITASLOP_DUMP_VPROG): reflect the bound vertex program's parameter |
| `VITASLOP_FIND_WORD` | vitaslop-native/tests/retail_boot_probe.rs:63 | The span `VITASLOP_FIND_WORD` searches: from the image base up through the guest heap. |
| `VITASLOP_FLAGS_WIDE_C` | vitaslop-transpiler/src/emit.rs:1273 | The A/B arm for [`emit_flags_add`]'s carry and overflow forms: `VITASLOP_FLAGS_WIDE_C=1` |
| `VITASLOP_FLAG_POISON` | vitaslop-transpiler/src/emit.rs:1231 | `VITASLOP_FLAG_POISON=0/1` - the FALSIFIER for the flag-liveness pass |
| `VITASLOP_FORCE_READY` | vitaslop-native/tests/retail_boot_probe.rs:738 | - |
| `VITASLOP_FORCE_READY_V2` | vitaslop-native/tests/retail_boot_probe.rs:761 | - |
| `VITASLOP_FORCE_RET` | vitaslop-transpiler/src/emit.rs:1567 | Diagnostic forced return. |
| `VITASLOP_FRAME_DIGEST` | vitaslop-native/src/recipe_runner.rs:334 | - |
| `VITASLOP_FRAME_TOPUP` | vitaslop-runtime/src/host.rs:5821 | The per-flip top-up ([`VitaState::advance_time_frame`]), which is OPT-IN: |
| `VITASLOP_FUEL` | vitaslop-native/src/threaded.rs:147 | This thread's SOFTWARE fuel counter (`abi::FUEL_EXPORT`), present only when the |
| `VITASLOP_GAME_DIR` | vitaslop-runtime/src/ingest/mod.rs:139 | Test-fixture access. |
| `VITASLOP_GAME_ID` | vitaslop-gamerun-recipes/tests/conformance.rs:30 | - |
| `VITASLOP_GAME_PKG` | vitaslop-runtime/src/ingest/pipeline.rs:523 | Diagnostic: dump the pkg header and the extracted file tree so a new |
| `VITASLOP_GAME_WORK` | vitaslop-runtime/src/ingest/pipeline.rs:700 | The pkg + work.bin chain over a privately-supplied two-file dump: extract |
| `VITASLOP_GAME_ZIP` | vitaslop-runtime/src/ingest/mod.rs:119 | - |
| `VITASLOP_GAP_CAP` | vitaslop-native/tests/retail_boot_probe.rs:268 | - |
| `VITASLOP_GESTURE_EVENT_KIND` | vitaslop-runtime/src/vita/gesture.rs:771 | `VITASLOP_GESTURE_EVENT_KIND`: write this byte at [`EVENT_KIND_OFF`]. |
| `VITASLOP_GESTURE_EVENT_STATE` | vitaslop-runtime/src/vita/gesture.rs:395 | The bits written into [`EVENT_STATE_OFF`]. |
| `VITASLOP_GESTURE_PRIMITIVE_STATE` | vitaslop-runtime/src/vita/gesture.rs:753 | `VITASLOP_GESTURE_PRIMITIVE_STATE`: write this halfword at [`PRIMITIVE_STATE_OFF`]. |
| `VITASLOP_GESTURE_TAP_ON_RELEASE` | vitaslop-runtime/src/vita/gesture.rs:819 | `VITASLOP_GESTURE_TAP_ON_RELEASE`: report a type-1 recognizer's event on the frame the |
| `VITASLOP_GESTURE_TYPE_MASK` | vitaslop-runtime/src/vita/gesture.rs:414 | Recognizer types allowed to report events (`VITASLOP_GESTURE_TYPE_MASK`, a bitmask |
| `VITASLOP_GPU` | vitaslop-native/tests/retail_boot_probe.rs:1398 | - |
| `VITASLOP_GPU_CHAIN_DIR` | vitaslop-native/src/wgpu_render.rs:424 | `VITASLOP_GPU_CHAIN_DIR=<dir>`: write every offscreen target of the frame just |
| `VITASLOP_GUARD_REG` | vitaslop-transpiler/src/emit.rs:1460 | Diagnostic callee-saved-register guard. |
| `VITASLOP_GUEST_CORES` | vitaslop-runtime/src/host.rs:16104 | CPU cores a Vita gives a GAME. |
| `VITASLOP_GXM` | vitaslop-native/examples/capsule-replay.rs:196 | - |
| `VITASLOP_GXM_DEPTH_ENC` | vitaslop-platform/src/gpu.rs:2384 | Which value a later pass reads out of a render target's depth |
| `VITASLOP_GXM_NO_MULTISAMPLE` | vitaslop-platform/src/gpu.rs:5407 | A/B instrument: force every pass to ONE sample, whatever the guest asked for. |
| `VITASLOP_GXM_UNIFORM_POISON` | vitaslop-gxp-shader/src/module.rs:317 | `:bits=<hex>` - paint a lane 1.0 when that register's RAW BITS equal this word, 0.0 |
| `VITASLOP_GXP` | vitaslop-native/examples/capsule-replay.rs:196 | - |
| `VITASLOP_GXP_` | vitaslop-native/examples/capsule-replay.rs:14 | - |
| `VITASLOP_GXP_ALLOW_FIXED_FUNCTION` | vitaslop-platform/src/gpu.rs:11452 | Whether a shader pair the recompiler cannot translate may be drawn by the |
| `VITASLOP_GXP_ATTR_FILL` | vitaslop-gxp-shader/src/module.rs:771 | Bytes the driver adds to the DEFAULT uniform buffer's bound address before writing it into |
| `VITASLOP_GXP_BLOB` | vitaslop-gxp-shader/tests/corpus.rs:641 | Print one named blob's recompiled WGSL body and its container reflection. |
| `VITASLOP_GXP_CAPSULE` | vitaslop-runtime/src/capsule.rs:632 | Diagnostic (`VITASLOP_GXP_CAPSULE=<vprog-hash>[,<vprog-hash>]:<dir>[:N]`): write the first |
| `VITASLOP_GXP_CAPSULE_MIN_INDICES` | vitaslop-runtime/src/capsule.rs:645 | Diagnostic (`VITASLOP_GXP_CAPSULE=<vprog-hash>[,<vprog-hash>]:<dir>[:N]`): write the first |
| `VITASLOP_GXP_CAPSULE_SKIP` | vitaslop-runtime/src/capsule.rs:665 | Diagnostic (`VITASLOP_GXP_CAPSULE_SKIP=<n>`): ignore the first `n` matching submissions |
| `VITASLOP_GXP_CORPUS` | vitaslop-gxp-shader/tests/corpus.rs:68 | Print every blob's content hash beside its file name, so a `gxp pair` line from a live run |
| `VITASLOP_GXP_CULL` | vitaslop-platform/src/gpu.rs:518 | `VITASLOP_GXP_CULL=0` restores the pre-2026-08-19b "draw both windings". |
| `VITASLOP_GXP_DEBUG` | vitaslop-platform/src/gpu.rs:11796 | Report - once per case - a GXM blend value with no exact wgpu equivalent, so the |
| `VITASLOP_GXP_DEFAULT_UNIFORM_OFFSET` | vitaslop-gxp-shader/src/module.rs:768 | Bytes the driver adds to the DEFAULT uniform buffer's bound address before writing it into |
| `VITASLOP_GXP_DISASM` | vitaslop-gxp-shader/tests/oracle.rs:865 | Compact disassembly of one blob (named by `VITASLOP_GXP_DISASM`, matched as a filename |
| `VITASLOP_GXP_DUMP` | vitaslop-platform/src/gpu.rs:5676 | Diagnostic (`VITASLOP_GXP_KEYS=<hex>,<hex>`): recompile ONLY these shader-pair keys |
| `VITASLOP_GXP_DUMPS` | vitaslop-gxp-shader/tests/oracle.rs:145 | Histogram the raw values of named fields across every instruction of a given opcode1 |
| `VITASLOP_GXP_EXCLUDE` | vitaslop-platform/src/gpu.rs:5680 | Pairs forced down the fixed-function path (`VITASLOP_GXP_EXCLUDE`). |
| `VITASLOP_GXP_FORCE` | vitaslop-platform/src/gpu.rs:5652 | Diagnostic (`VITASLOP_GXP_FORCE`): bind a neutral fallback texture for a sampler |
| `VITASLOP_GXP_GROUP` | vitaslop-gxp-shader/tests/corpus.rs:2728 | Every distinct word of one opcode group across the corpus, with the programs it appears in. |
| `VITASLOP_GXP_INPUTS` | vitaslop-platform/src/gpu.rs:8648 | Diagnostic (`VITASLOP_GXP_INPUTS=<hex-key>[,<hex-key>]` or `=all`): print, ONCE per |
| `VITASLOP_GXP_INPUTS_DIR` | vitaslop-platform/src/gpu.rs:2500 | Whether the once-per-pair `gxp pair <key>: vprog hash ..., fprog hash ...` INDEX should be |
| `VITASLOP_GXP_INPUTS_ORDER` | vitaslop-platform/src/gpu.rs:144 | The output of a diagnostic whose own KNOB is already the gate. |
| `VITASLOP_GXP_INPUTS_VERTS` | vitaslop-platform/src/gpu.rs:8662 | Diagnostic (`VITASLOP_GXP_INPUTS=<hex-key>[,<hex-key>]` or `=all`): print, ONCE per |
| `VITASLOP_GXP_INTERP` | vitaslop-platform/src/gpu.rs:11872 | - |
| `VITASLOP_GXP_KEYCOLOR` | vitaslop-platform/src/gpu.rs:2509 | Whether the once-per-pair `gxp pair <key>: vprog hash ..., fprog hash ...` INDEX should be |
| `VITASLOP_GXP_KEYS` | vitaslop-platform/src/gpu.rs:5675 | Diagnostic (`VITASLOP_GXP_KEYS=<hex>,<hex>`): recompile ONLY these shader-pair keys |
| `VITASLOP_GXP_LIVE` | vitaslop-platform/src/gpu.rs:1303 | The guest's real vertex+fragment shaders + their draw inputs, for the GXP->WGSL |
| `VITASLOP_GXP_MIPS` | vitaslop-platform/src/gpu.rs:2781 | Whether a chain is built for a seam, ignoring the per-texture exception above. |
| `VITASLOP_GXP_NEGW` | vitaslop-platform/src/gpu.rs:5831 | How to choose the clip-`w` sign correction (`VITASLOP_GXP_NEGW`). |
| `VITASLOP_GXP_NOBLEND` | vitaslop-platform/src/gpu.rs:5670 | Diagnostic (`VITASLOP_GXP_NOBLEND`): force every recompiled pipeline to REPLACE with |
| `VITASLOP_GXP_NODEPTH` | vitaslop-platform/src/gpu.rs:5662 | Diagnostic (`VITASLOP_GXP_NODEPTH`): every recompiled draw keeps its real shading and |
| `VITASLOP_GXP_ONLY` | vitaslop-platform/src/gpu.rs:5642 | Render ONLY recompiled draws, skipping the fixed-function draw for any call that |
| `VITASLOP_GXP_PAIR` | vitaslop-gxp-shader/tests/corpus.rs:786 | Link one named (vertex, fragment) pair and print the COMPLETE WGSL module both stages become. |
| `VITASLOP_GXP_PAIRS` | vitaslop-gxp-shader/tests/corpus.rs:2117 | >>> THE REFUTATION ABOVE WAS MEASURED OVER THE WRONG POPULATION. |
| `VITASLOP_GXP_PRECOMPILE` | vitaslop-platform/src/gpu.rs:527 | Whether a shader pair the guest's patcher names is compiled AHEAD of the draw that binds it |
| `VITASLOP_GXP_PRECOMPILE_CROSS` | vitaslop-runtime/src/host.rs:13216 | `VITASLOP_GXP_PRECOMPILE_CROSS`: for a title whose `sceGxmShaderPatcherCreateFragmentProgram` |
| `VITASLOP_GXP_PROBE` | vitaslop-gxp-shader/src/module.rs:290 | Diagnostic (`VITASLOP_GXP_PROBE=<bank><idx>[@<instr>][:f32/:bits=<hex>]`, e.g. |
| `VITASLOP_GXP_QUADS` | vitaslop-platform/src/gpu.rs:8659 | Diagnostic (`VITASLOP_GXP_INPUTS=<hex-key>[,<hex-key>]` or `=all`): print, ONCE per |
| `VITASLOP_GXP_REAL_PAIRS` | vitaslop-gxp-shader/tests/corpus.rs:2115 | >>> THE REFUTATION ABOVE WAS MEASURED OVER THE WRONG POPULATION. |
| `VITASLOP_GXP_RECOMPILE` | vitaslop-runtime/src/host.rs:13953 | - |
| `VITASLOP_GXP_SA` | vitaslop-platform/src/gpu.rs:9308 | Diagnostic (`VITASLOP_GXP_SA=<key>:<v/f>:<reg>=<hexword>[,...]`): replace a default-uniform |
| `VITASLOP_GXP_SA_DIRECT` | vitaslop-gxp-shader/src/link.rs:2429 | `0` restores the SA copy loop, `unroll` the constant-subscript copy - see [`resolve_sa_init`]. |
| `VITASLOP_GXP_SIZE_BANKS` | vitaslop-gxp-shader/src/link.rs:2368 | `VITASLOP_GXP_SIZE_BANKS=0` restores the pre-2026-08-20b emission - every register bank |
| `VITASLOP_GXP_SOLID` | vitaslop-platform/src/gpu.rs:5658 | Diagnostic (`VITASLOP_GXP_SOLID`): every recompiled draw outputs solid magenta with |
| `VITASLOP_GXP_VARYING_LAYOUT` | vitaslop-gxp-shader/src/link.rs:1201 | Diagnostic (`VITASLOP_GXP_VARYING_LAYOUT=<vhash>:<usage>@<lane>x<comps>,...`): plan ONE |
| `VITASLOP_GXP_VARYING_ORDER` | vitaslop-gxp-shader/src/link.rs:931 | The vertex lane order the paired FRAGMENT's declaration implies, or `None` when the two |
| `VITASLOP_GXP_VARYING_RESOLVE` | vitaslop-gxp-shader/src/link.rs:1293 | - |
| `VITASLOP_GXP_VPROBE` | vitaslop-gxp-shader/src/module.rs:428 | The `vec4<f32>` expression that reads the final colour out of register-file array `bank`, |
| `VITASLOP_GXP_WGSL_DIR` | vitaslop-gxp-shader/tests/oracle.rs:784 | Link each captured vertex<->fragment PAIR into a single WGSL module and prove the linked |
| `VITASLOP_GXP_YFLIP` | vitaslop-platform/src/gpu.rs:5648 | Flip clip Y (`VITASLOP_GXP_YFLIP`, default off). |
| `VITASLOP_GXP_ZFIX` | vitaslop-platform/src/gpu.rs:5646 | Apply the GXM (GL-style, NDC z in [-1,1]) -> WebGPU (z in [0,1]) clip-depth remap |
| `VITASLOP_HEADLESS_FRAMES` | vitaslop-desktop/src/retail.rs:834 | Headless self-check of the retail path (NO window): load `dir`, optionally drive a |
| `VITASLOP_HEADLESS_NO_TAPS` | vitaslop-desktop/src/retail.rs:840 | Headless self-check of the retail path (NO window): load `dir`, optionally drive a |
| `VITASLOP_HEADLESS_RECIPE` | vitaslop-desktop/src/retail.rs:837 | Headless self-check of the retail path (NO window): load `dir`, optionally drive a |
| `VITASLOP_HEADLESS_SHOT_EVERY` | vitaslop-desktop/src/retail.rs:843 | Headless self-check of the retail path (NO window): load `dir`, optionally drive a |
| `VITASLOP_HEADLESS_SHOT_FROM` | vitaslop-desktop/src/retail.rs:845 | Headless self-check of the retail path (NO window): load `dir`, optionally drive a |
| `VITASLOP_HEADLESS_SHOT_TO` | vitaslop-desktop/src/retail.rs:845 | Headless self-check of the retail path (NO window): load `dir`, optionally drive a |
| `VITASLOP_HEADLESS_TIMING` | vitaslop-desktop/src/retail.rs:841 | Headless self-check of the retail path (NO window): load `dir`, optionally drive a |
| `VITASLOP_HOLD_BUTTONS` | vitaslop-native/tests/retail_boot_probe.rs:96 | A minimal host world: a monotonic clock advancing one 60Hz tick per poll, no |
| `VITASLOP_HOLD_FROM` | vitaslop-native/tests/retail_boot_probe.rs:97 | A minimal host world: a monotonic clock advancing one 60Hz tick per poll, no |
| `VITASLOP_HOLD_MEM` | vitaslop-native/tests/retail_boot_probe.rs:717 | - |
| `VITASLOP_HOLD_TOUCH` | vitaslop-native/tests/retail_boot_probe.rs:114 | - |
| `VITASLOP_HOSTCALL_WATCH` | vitaslop-runtime/src/vita/mod.rs:524 | `VITASLOP_HOSTCALL_WATCH=<hex addr>[,<hex addr>...]` - print every host call that passes one |
| `VITASLOP_HOST_WRITE_WATCH` | vitaslop-runtime/src/host.rs:575 | `VITASLOP_HOST_WRITE_WATCH=<hex addr>[,...]`: report every write a HOST CALL makes to one |
| `VITASLOP_INGEST_DEBUG` | vitaslop-runtime/src/ingest/filesdb.rs:172 | Resolve every non-directory node to its full '/'-separated path (no |
| `VITASLOP_INPUT_RECIPE` | vitaslop-native/tests/retail_boot_probe.rs:390 | - |
| `VITASLOP_IO_BANDWIDTH_KIBPS` | vitaslop-runtime/src/vita/iofilemgr.rs:22 | Modelled sequential read bandwidth, in KiB per second |
| `VITASLOP_IO_PARK_THRESHOLD_US` | vitaslop-runtime/src/vita/iofilemgr.rs:110 | Smallest debt worth a context switch, in microseconds |
| `VITASLOP_IO_REQUEST_US` | vitaslop-runtime/src/vita/iofilemgr.rs:81 | Fixed per-request cost in microseconds (`VITASLOP_IO_REQUEST_US`): the command |
| `VITASLOP_LOG` | vitaslop-platform/src/gpu.rs:24 | A renderer diagnostic, at `debug` on the `vitaslop::gxm` target. |
| `VITASLOP_MAX_FRAMES` | vitaslop-native/tests/retail_boot_probe.rs:513 | - |
| `VITASLOP_MAX_ROUNDS` | vitaslop-native/tests/retail_boot_probe.rs:518 | - |
| `VITASLOP_MOVIE` | vitaslop-runtime/src/vita/video.rs:1970 | A track whose codec this engine does not decode is not offered at all: the title's |
| `VITASLOP_MOVIE_DUMP_DIR` | vitaslop-runtime/src/vita/avcdec.rs:1066 | >>> AND WHAT THE PICTURE ACTUALLY LOOKS LIKE, because "a picture arrived" and "the movie |
| `VITASLOP_MOVIE_DUMP_EVERY` | vitaslop-runtime/src/vita/avcdec.rs:1066 | >>> AND WHAT THE PICTURE ACTUALLY LOOKS LIKE, because "a picture arrived" and "the movie |
| `VITASLOP_MOVIE_PICTURE_HASH` | vitaslop-runtime/src/vita/avcdec.rs:160 | Pictures handed to the guest so far, which is what `VITASLOP_MOVIE_PICTURE_HASH` |
| `VITASLOP_MOVIE_SUBSTITUTE` | vitaslop-runtime/src/vita/video.rs:150 | >>> OPEN A DIFFERENT MOVIE THAN THE TITLE ASKED FOR |
| `VITASLOP_MP4_AUDIO` | vitaslop-runtime/src/vita/video.rs:1218 | The tracks this engine will hand units for, as cursors, in the order they appear in the |
| `VITASLOP_MP4_UNITS` | vitaslop-runtime/src/vita/video.rs:1325 | `VITASLOP_MP4_UNITS=none`: never return an access unit. |
| `VITASLOP_NEON_CACHE` | vitaslop-transpiler/src/emit.rs:2529 | Whether emitted modules hold the low NEON bank in locals across a run of vector |
| `VITASLOP_NGS_VOICE_HANDLE_MEMO` | vitaslop-runtime/src/vita/ngs.rs:406 | SceInt32 sceNgsRackGetVoiceHandle(SceNgsHRack rack, SceUInt32 index, SceNgsHVoice *handle) |
| `VITASLOP_NO_BC` | vitaslop-runtime/src/render.rs:1953 | Decode a whole BC1/BC2/BC3 block to its sixteen RGBA8 texels at once. |
| `VITASLOP_NO_FAST_IMPORT` | vitaslop-runtime/src/vita/mod.rs:115 | Whether `func_nid`'s handler can only ever CONTINUE, so the transpiler may route the |
| `VITASLOP_NO_INLINE_CLIB` | vitaslop-runtime/src/vita/mod.rs:243 | `VITASLOP_NO_INLINE_CLIB`: route `sceClibMemcpy`, `sceClibMemset` and `sceClibMemcmp` |
| `VITASLOP_NO_INLINE_IMPORTS` | vitaslop-runtime/src/host.rs:6040 | >>> WHO HAS WRITTEN THE CONTEXT'S TEXTURE SLOTS, counted for the failure report above. |
| `VITASLOP_NO_INLINE_LWMUTEX` | vitaslop-runtime/src/vita/mod.rs:345 | `VITASLOP_NO_INLINE_LWMUTEX`: route the lightweight-mutex lock and unlock through the |
| `VITASLOP_NO_INLINE_RESERVE` | vitaslop-runtime/src/vita/mod.rs:280 | `VITASLOP_NO_INLINE_RESERVE`: route `sceGxmReserve{Vertex,Fragment}DefaultUniformBuffer` |
| `VITASLOP_NO_INLINE_STUBS` | vitaslop-runtime/src/vita/mod.rs:217 | `VITASLOP_NO_INLINE_STUBS`: route the constant-return stubs through the host, leaving |
| `VITASLOP_NO_INLINE_TEXTURE` | vitaslop-runtime/src/host.rs:6040 | >>> WHO HAS WRITTEN THE CONTEXT'S TEXTURE SLOTS, counted for the failure report above. |
| `VITASLOP_NO_INLINE_UNIFORM_DATA` | vitaslop-runtime/src/vita/mod.rs:314 | `VITASLOP_NO_INLINE_UNIFORM_DATA`: route `sceGxmSetUniformDataF` through the host, |
| `VITASLOP_NO_NGS_MIX` | vitaslop-runtime/src/vita/audio.rs:279 | `VITASLOP_NO_NGS_MIX`: skip the NGS decode-and-mix entirely, leaving the guest's |
| `VITASLOP_PATCH_STUBS` | vitaslop-native/tests/retail_boot_probe.rs:468 | - |
| `VITASLOP_PAUSE_ON_BLUR` | vitaslop-desktop/src/retail.rs:1645 | >>> THE HARD PAUSE: the window lost focus and `VITASLOP_PAUSE_ON_BLUR` (default on) |
| `VITASLOP_PEEK` | vitaslop-desktop/src/retail.rs:558 | Guest memory at `addr`, for `VITASLOP_PEEK`. |
| `VITASLOP_PERF` | vitaslop-native/src/perf.rs:43 | Is perf accounting on (`VITASLOP_PERF` set)? Read once and cached. |
| `VITASLOP_PERF_CONSOLE` | vitaslop-web/src/lib.rs:1160 | Whether the per-window performance report is also written to the browser CONSOLE |
| `VITASLOP_PIXEL_TRACE` | vitaslop-runtime/src/render.rs:5061 | Draw one scene onto an EXISTING framebuffer and depth buffer, composing with whatever |
| `VITASLOP_POISON_UNRESOLVED_VARS` | vitaslop-runtime/src/link.rs:318 | - |
| `VITASLOP_POKE` | vitaslop-native/tests/retail_boot_probe.rs:685 | - |
| `VITASLOP_POLL_ADDR` | vitaslop-native/src/threaded.rs:1752 | Guest address to sample after each host call, from `VITASLOP_POLL_ADDR` (hex). |
| `VITASLOP_PREPARE_SPLIT` | vitaslop-platform/src/gpu.rs:4733 | Where the milliseconds INSIDE one `prepare` go, plus the bytes each phase moved. |
| `VITASLOP_PREPOKE` | vitaslop-native/tests/retail_boot_probe.rs:491 | - |
| `VITASLOP_PRESENT_PROBE` | vitaslop-web/src/lib.rs:558 | Reads back WHAT WE PRESENTED, when `VITASLOP_PRESENT_PROBE` asks for it. |
| `VITASLOP_PROMOTE_POISON` | vitaslop-transpiler/src/emit.rs:2573 | `VITASLOP_PROMOTE_POISON=<n>` - the FALSIFIER for register promotion. |
| `VITASLOP_PROMOTE_REGS` | vitaslop-native/src/threaded.rs:2315 | A concise trap description (kind + message), matching the sync `Vm`'s detail. |
| `VITASLOP_PSARC` | vitaslop-runtime/src/psarc.rs:430 | Read a REAL archive: `VITASLOP_PSARC=<path to a .psarc>`, optionally |
| `VITASLOP_PSARC_FILE` | vitaslop-runtime/src/psarc.rs:431 | Read a REAL archive: `VITASLOP_PSARC=<path to a .psarc>`, optionally |
| `VITASLOP_PVRTC_DECODE` | vitaslop-runtime/src/render.rs:6323 | Whether PVRTC decodes a whole face at a time (the default) or one texel at a time. |
| `VITASLOP_QUANTUM_CPU_US` | vitaslop-runtime/src/host.rs:16059 | Game-clock time charged for one [`QUANTUM_ARM`] of guest execution, in microseconds. |
| `VITASLOP_QUANTUM_FUEL` | vitaslop-native/tests/retail_boot_probe.rs:427 | - |
| `VITASLOP_REGTRACE` | vitaslop-native/src/threaded.rs:1395 | `VITASLOP_REGTRACE=<lo>-<hi>:<path>` - append the reg+flag file per block entry in |
| `VITASLOP_REGTRACE_MAX` | vitaslop-native/src/threaded.rs:1555 | `VITASLOP_REGTRACE_MAX=<n>` caps the register trace at `n` lines (0 = unbounded). |
| `VITASLOP_REGTRACE_WATCH` | vitaslop-native/src/threaded.rs:1332 | The `VITASLOP_REGTRACE_WATCH` words, formatted as ` mADDR=VALUE` fields ready to append |
| `VITASLOP_RESIDENT_GEOM` | vitaslop-platform/src/gpu.rs:5903 | Repacked vertices and expanded indices that have not changed since the renderer first |
| `VITASLOP_RESIDENT_GEOM_MB` | vitaslop-platform/src/gpu.rs:5946 | The byte budget for each of the two heaps (`VITASLOP_RESIDENT_GEOM_MB`, per heap). |
| `VITASLOP_ROUNDS_PER_FRAME` | vitaslop-native/tests/retail_boot_probe.rs:662 | - |
| `VITASLOP_RTT_BG_CACHE` | vitaslop-platform/src/gpu.rs:495 | `VITASLOP_RTT_BG_CACHE=0` restores the OLD behaviour: a sampler bind group naming a render |
| `VITASLOP_SAMPLER_NARROW` | vitaslop-runtime/src/host.rs:2833 | Whether a draw decodes only the texture units its fragment program DECLARES - see |
| `VITASLOP_SCAN_WORD` | vitaslop-native/tests/retail_boot_probe.rs:967 | - |
| `VITASLOP_SCENE_LIMIT` | vitaslop-native/tests/retail_boot_probe.rs:448 | - |
| `VITASLOP_SCHED_CORES` | vitaslop-runtime/src/sched.rs:603 | `VITASLOP_SCHED_CORES=<n>`: cap the baton to the top `n` runnable PRIORITIES, as the |
| `VITASLOP_SCHED_RR` | vitaslop-runtime/src/sched.rs:619 | `VITASLOP_SCHED_RR=1`: round-robin every runnable thread, ignoring priority. |
| `VITASLOP_SCHED_TRACE` | vitaslop-runtime/src/sched.rs:629 | `VITASLOP_SCHED_TRACE=<from>-<to>` (display frames, inclusive): print one line per |
| `VITASLOP_SET_EVF` | vitaslop-native/tests/retail_boot_probe.rs:695 | - |
| `VITASLOP_SHOT_DIR` | vitaslop-native/tests/retail_boot_probe.rs:209 | Read and format one watched value from current guest memory. |
| `VITASLOP_SHOT_LAST` | vitaslop-native/tests/retail_boot_probe.rs:446 | - |
| `VITASLOP_SIGNATURE` | vitaslop-native/src/recipe_runner.rs:101 | The determinism signature over the observable output (render stream + egress), |
| `VITASLOP_SIGNATURE_EVERY` | vitaslop-native/src/recipe_runner.rs:465 | `VITASLOP_SIGNATURE_EVERY=<n>`: print the RUNNING determinism signature every `n` stepped |
| `VITASLOP_SLOW_FRAME_US` | vitaslop-web/src/lib.rs:3556 | - |
| `VITASLOP_SNAPSHOT` | vitaslop-native/src/threaded.rs:1381 | `VITASLOP_SNAPSHOT=<hexpc>:<path>` - dump full state on first entry to block `hexpc`. |
| `VITASLOP_SNAPSHOT_BUDGET_MB` | vitaslop-runtime/src/host.rs:3289 | Byte budget for retained texture snapshots, scaled to the device |
| `VITASLOP_SNAPSHOT_DENSE` | vitaslop-native/src/threaded.rs:1491 | Dump the full guest state (all non-zero pages + r0..r15 + NZCV) to `path`, in the |
| `VITASLOP_SNAPSHOT_SKIP` | vitaslop-native/src/threaded.rs:1443 | `VITASLOP_SNAPSHOT_SKIP=<n>` - skip the first `n` entries to the snapshot block before |
| `VITASLOP_SOFTWARE` | vitaslop-desktop/src/retail.rs:1517 | - |
| `VITASLOP_SSAA` | vitaslop-platform/src/gpu.rs:12880 | Set the supersample factor: 1 (default) renders the scene straight into the caller's |
| `VITASLOP_STALL_CHUNK` | vitaslop-native/tests/retail_boot_probe.rs:545 | - |
| `VITASLOP_STALL_WAKE` | vitaslop-native/tests/retail_boot_probe.rs:544 | - |
| `VITASLOP_STALL_WATCHDOG` | vitaslop-native/src/watchdog.rs:99 | The configured stall budget in seconds, from `VITASLOP_STALL_WATCHDOG`. |
| `VITASLOP_STALL_WAVES` | vitaslop-native/tests/retail_boot_probe.rs:548 | - |
| `VITASLOP_STRICT_DRAWS` | vitaslop-runtime/src/render.rs:6102 | Why [`RenderSceneBuilder::build`] discarded draws from a captured scene. |
| `VITASLOP_SWITCH_WHY` | vitaslop-transpiler/src/lower.rs:916 | Whether the table-branch diagnostic is on for this address |
| `VITASLOP_SW_CHAIN` | vitaslop-native/src/wgpu_render.rs:427 | `VITASLOP_GPU_CHAIN_DIR=<dir>`: write every offscreen target of the frame just |
| `VITASLOP_SW_CHAIN_DIR` | vitaslop-runtime/src/render.rs:4878 | - |
| `VITASLOP_SW_POST` | vitaslop-runtime/src/render.rs:4925 | - |
| `VITASLOP_SYSTEM_FONT` | vitaslop-runtime/src/font/system.rs:75 | The resolved substitute: its bytes and a human-readable account of where they came from. |
| `VITASLOP_TEXTURE_CHECK` | vitaslop-runtime/src/host.rs:3040 | How a retained texture snapshot is re-validated (`VITASLOP_TEXTURE_CHECK`): `scene` |
| `VITASLOP_TEX_CACHE_MB` | vitaslop-platform/src/gpu.rs:469 | The texture-cache budget in bytes: [`GAME_RESIDENT_CEILING_MB`] unless |
| `VITASLOP_TEX_COMPRESS` | vitaslop-runtime/src/render.rs:1345 | Whether compressed textures reach the GPU compressed at all. |
| `VITASLOP_TEX_DIRTY_CENSUS` | vitaslop-runtime/src/host.rs:172 | >>> WHICH PARTS of `[off, off + len)` the guest may have stored into since `stamp`, |
| `VITASLOP_TEX_MEMO_PER_SCENE` | vitaslop-runtime/src/host.rs:2596 | A whole DRAW's worth of snapshotted textures, by the bindings that produced it - kept |
| `VITASLOP_TEX_PAGE_READ` | vitaslop-runtime/src/host.rs:3633 | Record that this entry's bytes are current as of THIS SCENE, so a later |
| `VITASLOP_TRACE_BLOCKS` | vitaslop-native/src/threaded.rs:1307 | `VITASLOP_TRACE_FRAMES=<from>-<to>` (decimal display frames, inclusive) - print the |
| `VITASLOP_TRACE_EXIT` | vitaslop-native/tests/retail_boot_probe.rs:37 | - |
| `VITASLOP_TRACE_FILE` | vitaslop-runtime/src/vita/libkernel.rs:47 | Diagnostic (`RUST_LOG=vitaslop::exit=debug`): when the guest calls |
| `VITASLOP_TRACE_FRAMES` | vitaslop-native/src/threaded.rs:1304 | `VITASLOP_TRACE_FRAMES=<from>-<to>` (decimal display frames, inclusive) - print the |
| `VITASLOP_TRACE_FUNCS` | vitaslop-native/src/threaded.rs:1227 | Bind `env.svc`. |
| `VITASLOP_TRACE_INDIRECT` | vitaslop-transpiler/src/emit.rs:1504 | Diagnostic indirect-call tracer. |
| `VITASLOP_TRACE_IO` | vitaslop-native/tests/retail_boot_probe.rs:34 | - |
| `VITASLOP_TRACE_ORDER` | vitaslop-runtime/src/vita/mod.rs:508 | Ordered-timeline trace (env `VITASLOP_TRACE_ORDER`): print every *meaningful* |
| `VITASLOP_TRACK_PC` | vitaslop-transpiler/src/abi.rs:200 | Exported name of the diagnostic guest-PC tracker global. |
| `VITASLOP_TRANSPILE_REPORT` | vitaslop-native/src/threaded.rs:843 | - |
| `VITASLOP_TRAP_HALT` | vitaslop-transpiler/src/emit.rs:1593 | When `VITASLOP_TRAP_HALT` is set, a `Term::Halt` (a block that ran off the end of decoded |
| `VITASLOP_UNIFORM_WATCH` | vitaslop-runtime/src/vita/gxm.rs:1810 | `VITASLOP_UNIFORM_WATCH=<hex address>/<parameter name substring>[,...]`: report every |
| `VITASLOP_UV_DEBUG` | vitaslop-runtime/src/render.rs:5075 | Draw one scene onto an EXISTING framebuffer and depth buffer, composing with whatever |
| `VITASLOP_VBLANK_PARK` | vitaslop-runtime/src/vita/display.rs:201 | Whether an inlined `sceDisplayGetVcount` carries the spin guard (`VITASLOP_VBLANK_PARK`, |
| `VITASLOP_VERTEX_INTERN` | vitaslop-runtime/src/host.rs:2818 | A cheap, allocation-free fingerprint of a vertex stream, for [`TextureSnapshots:: |
| `VITASLOP_WASM_INDICES` | vitaslop-native/src/threaded.rs:2369 | Rewrite `<wasm function N>` in a trap backtrace to name the GUEST function it is. |
| `VITASLOP_WASM_NAMES` | vitaslop-transpiler/src/emit.rs:1083 | When `VITASLOP_WASM_NAMES` is set, emit a wasm `name` custom section labelling |
| `VITASLOP_WATCH_` | vitaslop-transpiler/src/emit.rs:1416 | Number of matching store-watchpoint hits to skip before trapping (`VITASLOP_WATCH_ |
| `VITASLOP_WATCH_FROM` | vitaslop-native/tests/retail_boot_probe.rs:674 | - |
| `VITASLOP_WATCH_MEM` | vitaslop-native/tests/retail_boot_probe.rs:141 | Parse `VITASLOP_WATCH_MEM=addr:type:label,addr:type:label,...` into watches. |
| `VITASLOP_WATCH_READ` | vitaslop-transpiler/src/emit.rs:1055 | Diagnostic read watchpoint. |
| `VITASLOP_WATCH_READ_` | vitaslop-transpiler/src/emit.rs:1442 | Optional guest-PC EXCLUDE window for the read watchpoint (`VITASLOP_WATCH_READ_ |
| `VITASLOP_WATCH_READ_NZ` | vitaslop-transpiler/src/emit.rs:2940 | Emit the read-watchpoint trap check. |
| `VITASLOP_WATCH_READ_PC_EXCL` | vitaslop-transpiler/src/emit.rs:1452 | Optional guest-PC EXCLUDE window for the read watchpoint (`VITASLOP_WATCH_READ_ |
| `VITASLOP_WATCH_READ_SKIP` | vitaslop-transpiler/src/emit.rs:1156 | WASM global index of the read-watchpoint match counter, appended after the guest-PC |
| `VITASLOP_WATCH_STORE` | vitaslop-native/src/threaded.rs:1416 | `VITASLOP_REGTRACE_WATCH=<hex guest addr>[,<hex guest addr>...]` - append the WORD |
| `VITASLOP_WATCH_STORE_ARM` | vitaslop-transpiler/src/emit.rs:1615 | Store-watchpoint mode, from `VITASLOP_WATCH_STORE_MODE` (default `any`): |
| `VITASLOP_WATCH_STORE_LOG` | vitaslop-runtime/src/capture.rs:523 | GUEST ADDRESS the bytes above were read from, or 0 when there is no bound buffer. |
| `VITASLOP_WATCH_STORE_MODE` | vitaslop-transpiler/src/emit.rs:1608 | Store-watchpoint mode, from `VITASLOP_WATCH_STORE_MODE` (default `any`): |
| `VITASLOP_WATCH_STORE_NZ` | vitaslop-transpiler/src/emit.rs:1631 | `VITASLOP_WATCH_STORE_LOG` - LOG each store to the watched address (the storing |
| `VITASLOP_WATCH_STORE_SKIP` | vitaslop-transpiler/src/emit.rs:599 | Linear-memory byte offset of the store-watchpoint MATCH COUNTER, or 0 when this |
| `VITASLOP_XML_DUMP` | vitaslop-runtime/src/vita/sce_xml.rs:580 | `VITASLOP_XML_DUMP=<dir>`: write every document handed to `parse` into `<dir>` as |
