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

200 knobs.

| knob | read in | what it does |
|---|---|---|
| `VITASLOP_` | vitaslop-runtime/src/vita/mod.rs:179 | `VITASLOP_NO_INLINE_IMPORTS`: route every host call through the host, even the |
| `VITASLOP_ALLOW_SOFTWARE_GPU` | vitaslop-web/src/lib.rs:1014 | Whether a run may proceed on a software rasteriser (`VITASLOP_ALLOW_SOFTWARE_GPU`). |
| `VITASLOP_ARM_AT_FRAME` | vitaslop-native/src/threaded.rs:264 | Linear-memory offset of the "diagnostics armed" word, when this build was |
| `VITASLOP_AT9_DIR` | vitaslop-atrac9/tests/oracle.rs:66 | Decode a whole AT9 payload the way a superframe consumer does: for each |
| `VITASLOP_AUDIO_RAW` | vitaslop-runtime/src/vita/audio.rs:48 | Optional raw-s16le capture of the mixed output stream (env |
| `VITASLOP_BACKTRACE` | vitaslop-runtime/src/vita/mod.rs:239 | Print the guest call chain the first time a chosen NID is called from each thread |
| `VITASLOP_BLOCK_HIST` | vitaslop-native/src/recipe_runner.rs:150 | Dump the per-PC block-entry histogram gathered under `VITASLOP_BLOCK_HIST`, for |
| `VITASLOP_BLOCK_HIST_SEQ` | vitaslop-native/src/threaded.rs:1201 | Print the block-visit histogram gathered under `VITASLOP_BLOCK_HIST`: the `top` |
| `VITASLOP_BROWSER_FASTFORWARD` | vitaslop-web/src/lib.rs:994 | Frame to fast-forward the live loop to (`VITASLOP_BROWSER_FASTFORWARD`), unpaced. |
| `VITASLOP_BROWSER_FUEL` | vitaslop-web/src/browser_sched.rs:558 | Guest work a thread may execute before the browser preempts it, in WASMTIME FUEL UNITS |
| `VITASLOP_BROWSER_HEARTBEAT_MS` | vitaslop-web/src/lib.rs:2382 | - |
| `VITASLOP_BROWSER_INSTANCE_POOL` | vitaslop-web/src/browser_sched.rs:879 | Whether a finished thread's module instance may be REUSED by the next thread |
| `VITASLOP_BROWSER_QUANTUM_CALLS` | vitaslop-web/src/browser_sched.rs:105 | Host calls one guest thread may make before the browser preempts it |
| `VITASLOP_BROWSER_SUPERSAMPLE` | vitaslop-web/src/lib.rs:971 | Supersample factor for the live browser render (`VITASLOP_BROWSER_SUPERSAMPLE`). |
| `VITASLOP_CALLSITES_WINDOW` | vitaslop-desktop/src/retail.rs:788 | - |
| `VITASLOP_CHAIN_DRAWS` | vitaslop-platform/src/gpu.rs:8177 | - |
| `VITASLOP_CHAIN_LIMIT` | vitaslop-native/tests/gpu_rtt_gamma.rs:173 | Render a chain of `feedback` sample-and-write-back passes over the offscreen target and |
| `VITASLOP_CHAIN_SKIP` | vitaslop-native/src/wgpu_render.rs:281 | - |
| `VITASLOP_CHECK_ADDRS` | vitaslop-native/tests/retail_boot_probe.rs:43 | - |
| `VITASLOP_CLOCK_TRACE` | vitaslop-runtime/src/sched.rs:725 | Called when [`pick_next`](Self::pick_next) found nothing runnable. |
| `VITASLOP_CODE_RANGE` | vitaslop-runtime/src/vita/mod.rs:222 | The guest code range scanned for the game-level caller in [`dispatch`] (env |
| `VITASLOP_CPU_SHARE` | vitaslop-native/src/recipe_runner.rs:98 | Who actually got the CPU over the run, when `VITASLOP_CPU_SHARE` is set - see |
| `VITASLOP_DBG_CALLSITES` | vitaslop-runtime/src/vita/mod.rs:187 | Diagnostic call-site profiler (`VITASLOP_DBG_CALLSITES`): counts host calls |
| `VITASLOP_DEBUG_CAPTURE` | vitaslop-web/src/lib.rs:2404 | - |
| `VITASLOP_DECODE_CACHE_MB` | vitaslop-runtime/src/render.rs:4780 | Budget for the decode cache, in BYTES of decoded RGBA8, before it is cleared wholesale. |
| `VITASLOP_DIRTY_PAGES` | vitaslop-native/src/threaded.rs:123 | Linear-memory offset of the guest-store dirty block, when this build was |
| `VITASLOP_DRAW_ONLY` | vitaslop-runtime/src/render.rs:3788 | Draw one scene onto an EXISTING framebuffer and depth buffer, composing with whatever |
| `VITASLOP_DRAW_STATS` | vitaslop-runtime/src/render.rs:3777 | Draw one scene onto an EXISTING framebuffer and depth buffer, composing with whatever |
| `VITASLOP_DRV_KEY` | vitaslop-runtime/src/ingest/pfscrypt.rs:264 | `F00D(klicensee)` for the title, from `VITASLOP_DRV_KEY` (32 hex chars), or |
| `VITASLOP_DUMP_DIR` | vitaslop-runtime/src/ingest/pipeline.rs:827 | Diagnostic: decrypt the container and write named plaintext files out to |
| `VITASLOP_DUMP_DRAW` | vitaslop-native/tests/retail_boot_probe.rs:1292 | - |
| `VITASLOP_DUMP_DRAWS` | vitaslop-native/tests/retail_boot_probe.rs:1080 | - |
| `VITASLOP_DUMP_DRAW_GXP` | vitaslop-gxp-shader/tests/oracle.rs:657 | Correlate each captured vertex<->fragment PAIR (from a real draw run) to establish the |
| `VITASLOP_DUMP_DRAW_GXP_CAP` | vitaslop-runtime/src/host.rs:7603 | - |
| `VITASLOP_DUMP_DRAW_GXP_FULL` | vitaslop-runtime/src/host.rs:7704 | - |
| `VITASLOP_DUMP_EXPORTS` | vitaslop-runtime/src/link.rs:414 | - |
| `VITASLOP_DUMP_FILES` | vitaslop-runtime/src/ingest/pipeline.rs:829 | Diagnostic: decrypt the container and write named plaintext files out to |
| `VITASLOP_DUMP_FPROG` | vitaslop-runtime/src/host.rs:7491 | Diagnostic (VITASLOP_DUMP_FPROG): print the bound fragment program's sampler |
| `VITASLOP_DUMP_FUNC` | vitaslop-native/tests/retail_boot_probe.rs:319 | - |
| `VITASLOP_DUMP_GXP_BIN` | vitaslop-runtime/src/host.rs:7535 | `VITASLOP_DUMP_GXP_BIN=<dir>`: write the raw `SceGxmProgram` blobs (the whole container - |
| `VITASLOP_DUMP_IMAGE` | vitaslop-native/tests/retail_boot_probe.rs:33 | - |
| `VITASLOP_DUMP_IMPORTS` | vitaslop-native/tests/retail_boot_probe.rs:355 | - |
| `VITASLOP_DUMP_MAP` | vitaslop-native/tests/retail_boot_probe.rs:591 | - |
| `VITASLOP_DUMP_MEM` | vitaslop-native/tests/retail_boot_probe.rs:43 | - |
| `VITASLOP_DUMP_PATHS` | vitaslop-native/tests/retail_boot_probe.rs:414 | - |
| `VITASLOP_DUMP_REGION` | vitaslop-native/tests/retail_boot_probe.rs:529 | - |
| `VITASLOP_DUMP_REGION_RANGE` | vitaslop-native/tests/retail_boot_probe.rs:531 | - |
| `VITASLOP_DUMP_RENDERSCENE` | vitaslop-native/tests/retail_boot_probe.rs:1334 | - |
| `VITASLOP_DUMP_SCENES` | vitaslop-desktop/src/retail.rs:259 | Step the guest one display frame. |
| `VITASLOP_DUMP_STDOUT` | vitaslop-desktop/src/retail.rs:946 | - |
| `VITASLOP_DUMP_STUBS` | vitaslop-native/tests/retail_boot_probe.rs:32 | - |
| `VITASLOP_DUMP_TEX` | vitaslop-native/tests/retail_boot_probe.rs:1234 | - |
| `VITASLOP_DUMP_TEX_DIR` | vitaslop-runtime/src/host.rs:7710 | - |
| `VITASLOP_DUMP_TEX_MAX_TEXELS` | vitaslop-runtime/src/host.rs:7719 | - |
| `VITASLOP_DUMP_TRIS` | vitaslop-runtime/src/render.rs:3795 | Draw one scene onto an EXISTING framebuffer and depth buffer, composing with whatever |
| `VITASLOP_DUMP_VPROG` | vitaslop-runtime/src/host.rs:7330 | Diagnostic (VITASLOP_DUMP_VPROG): reflect the bound vertex program's parameter |
| `VITASLOP_FIND_WORD` | vitaslop-native/tests/retail_boot_probe.rs:63 | The span `VITASLOP_FIND_WORD` searches: from the image base up through the guest heap. |
| `VITASLOP_FORCE_READY` | vitaslop-native/tests/retail_boot_probe.rs:738 | - |
| `VITASLOP_FORCE_READY_V2` | vitaslop-native/tests/retail_boot_probe.rs:761 | - |
| `VITASLOP_FORCE_RET` | vitaslop-transpiler/src/emit.rs:789 | Diagnostic forced return. |
| `VITASLOP_FRAME_TOPUP` | vitaslop-runtime/src/host.rs:2278 | The per-flip top-up ([`VitaState::advance_time_frame`]), off under |
| `VITASLOP_FUEL` | vitaslop-native/src/threaded.rs:142 | This thread's SOFTWARE fuel counter (`abi::FUEL_EXPORT`), present only when the |
| `VITASLOP_GAME_DIR` | vitaslop-runtime/src/ingest/mod.rs:138 | Test-fixture access. |
| `VITASLOP_GAME_ID` | vitaslop-gamerun-recipes/tests/conformance.rs:30 | - |
| `VITASLOP_GAME_PKG` | vitaslop-runtime/src/ingest/pipeline.rs:523 | Diagnostic: dump the pkg header and the extracted file tree so a new |
| `VITASLOP_GAME_WORK` | vitaslop-runtime/src/ingest/pipeline.rs:700 | The pkg + work.bin chain over a privately-supplied two-file dump: extract |
| `VITASLOP_GAME_ZIP` | vitaslop-runtime/src/ingest/zip.rs:113 | Scan backward from EOF for the EOCD signature. |
| `VITASLOP_GAP_CAP` | vitaslop-native/tests/retail_boot_probe.rs:268 | - |
| `VITASLOP_GESTURE_TYPE_MASK` | vitaslop-runtime/src/vita/gesture.rs:308 | Recognizer types allowed to report events (`VITASLOP_GESTURE_TYPE_MASK`, a bitmask |
| `VITASLOP_GPU` | vitaslop-native/tests/retail_boot_probe.rs:1398 | - |
| `VITASLOP_GPU_CHAIN_DIR` | vitaslop-native/src/wgpu_render.rs:369 | `VITASLOP_GPU_CHAIN_DIR=<dir>`: write every offscreen target of the frame just |
| `VITASLOP_GUARD_REG` | vitaslop-transpiler/src/emit.rs:702 | Diagnostic callee-saved-register guard. |
| `VITASLOP_GUEST_CORES` | vitaslop-runtime/src/host.rs:9117 | CPU cores a Vita gives a GAME. |
| `VITASLOP_GXM_DEPTH_ENC` | vitaslop-platform/src/gpu.rs:1342 | Which value a later pass reads out of a render target's depth |
| `VITASLOP_GXM_NO_MULTISAMPLE` | vitaslop-platform/src/gpu.rs:2403 | A/B instrument: force every pass to ONE sample, whatever the guest asked for. |
| `VITASLOP_GXM_UNIFORM_POISON` | vitaslop-runtime/src/host.rs:8114 | Diagnostic (`VITASLOP_GXM_UNIFORM_POISON=1`): fill a freshly reserved default uniform buffer |
| `VITASLOP_GXP_ALLOW_FIXED_FUNCTION` | vitaslop-platform/src/gpu.rs:5953 | Whether a shader pair the recompiler cannot translate may be drawn by the |
| `VITASLOP_GXP_BLOB` | vitaslop-gxp-shader/tests/corpus.rs:266 | Print one named blob's recompiled WGSL body and its container reflection. |
| `VITASLOP_GXP_CORPUS` | vitaslop-gxp-shader/tests/corpus.rs:66 | Print every blob's content hash beside its file name, so a `gxp pair` line from a live run |
| `VITASLOP_GXP_DEBUG` | vitaslop-platform/src/gpu.rs:6212 | Report - once per case - a GXM blend value with no exact wgpu equivalent, so the |
| `VITASLOP_GXP_DISASM` | vitaslop-gxp-shader/tests/oracle.rs:865 | Compact disassembly of one blob (named by `VITASLOP_GXP_DISASM`, matched as a filename |
| `VITASLOP_GXP_DUMP` | vitaslop-platform/src/gpu.rs:2636 | Diagnostic (`VITASLOP_GXP_KEYS=<hex>,<hex>`): recompile ONLY these shader-pair keys |
| `VITASLOP_GXP_DUMPS` | vitaslop-gxp-shader/tests/oracle.rs:145 | Histogram the raw values of named fields across every instruction of a given opcode1 |
| `VITASLOP_GXP_EXCLUDE` | vitaslop-platform/src/gpu.rs:2640 | Pairs forced down the fixed-function path (`VITASLOP_GXP_EXCLUDE`). |
| `VITASLOP_GXP_FORCE` | vitaslop-platform/src/gpu.rs:2612 | Diagnostic (`VITASLOP_GXP_FORCE`): bind a neutral fallback texture for a sampler |
| `VITASLOP_GXP_GROUP` | vitaslop-gxp-shader/tests/oracle.rs:801 | Every raw instruction word of one opcode GROUP across the whole corpus, plus a per-bit |
| `VITASLOP_GXP_INPUTS` | vitaslop-platform/src/gpu.rs:4161 | Diagnostic (`VITASLOP_GXP_INPUTS=<hex-key>[,<hex-key>]` or `=all`): print, ONCE per |
| `VITASLOP_GXP_INPUTS_DIR` | vitaslop-platform/src/gpu.rs:1439 | Whether the once-per-pair `gxp pair <key>: vprog hash ..., fprog hash ...` INDEX should be |
| `VITASLOP_GXP_INPUTS_ORDER` | vitaslop-platform/src/gpu.rs:41 | The output of a diagnostic whose own KNOB is already the gate. |
| `VITASLOP_GXP_INPUTS_VERTS` | vitaslop-platform/src/gpu.rs:1438 | Whether the once-per-pair `gxp pair <key>: vprog hash ..., fprog hash ...` INDEX should be |
| `VITASLOP_GXP_INTERP` | vitaslop-platform/src/gpu.rs:6258 | - |
| `VITASLOP_GXP_KEYCOLOR` | vitaslop-platform/src/gpu.rs:5360 | - |
| `VITASLOP_GXP_KEYS` | vitaslop-platform/src/gpu.rs:2635 | Diagnostic (`VITASLOP_GXP_KEYS=<hex>,<hex>`): recompile ONLY these shader-pair keys |
| `VITASLOP_GXP_LIVE` | vitaslop-platform/src/gpu.rs:505 | The guest's real vertex+fragment shaders + their draw inputs, for the GXP->WGSL |
| `VITASLOP_GXP_MIPS` | vitaslop-platform/src/gpu.rs:1605 | Whether [`upload_gxp_texture`] builds a mip chain for this seam. |
| `VITASLOP_GXP_NEGW` | vitaslop-platform/src/gpu.rs:2736 | How to choose the clip-`w` sign correction (`VITASLOP_GXP_NEGW`). |
| `VITASLOP_GXP_NOBLEND` | vitaslop-platform/src/gpu.rs:2630 | Diagnostic (`VITASLOP_GXP_NOBLEND`): force every recompiled pipeline to REPLACE with |
| `VITASLOP_GXP_NODEPTH` | vitaslop-platform/src/gpu.rs:2622 | Diagnostic (`VITASLOP_GXP_NODEPTH`): every recompiled draw keeps its real shading and |
| `VITASLOP_GXP_ONLY` | vitaslop-platform/src/gpu.rs:2602 | Render ONLY recompiled draws, skipping the fixed-function draw for any call that |
| `VITASLOP_GXP_PAIR` | vitaslop-gxp-shader/tests/corpus.rs:362 | Link one named (vertex, fragment) pair and print the COMPLETE WGSL module both stages become. |
| `VITASLOP_GXP_PAIRS` | vitaslop-platform/src/gpu.rs:1425 | Whether the once-per-pair `gxp pair <key>: vprog hash ..., fprog hash ...` INDEX should be |
| `VITASLOP_GXP_PROBE` | vitaslop-gxp-shader/src/module.rs:243 | The `vec4<f32>` expression that reads the final colour out of register-file array `bank`, |
| `VITASLOP_GXP_RECOMPILE` | vitaslop-runtime/src/host.rs:7748 | - |
| `VITASLOP_GXP_SA` | vitaslop-platform/src/gpu.rs:4582 | Diagnostic (`VITASLOP_GXP_SA=<key>:<v/f>:<reg>=<hexword>[,...]`): replace a default-uniform |
| `VITASLOP_GXP_SOLID` | vitaslop-platform/src/gpu.rs:2618 | Diagnostic (`VITASLOP_GXP_SOLID`): every recompiled draw outputs solid magenta with |
| `VITASLOP_GXP_VPROBE` | vitaslop-gxp-shader/src/module.rs:248 | The `vec4<f32>` expression that reads the final colour out of register-file array `bank`, |
| `VITASLOP_GXP_WGSL_DIR` | vitaslop-gxp-shader/tests/oracle.rs:784 | Link each captured vertex<->fragment PAIR into a single WGSL module and prove the linked |
| `VITASLOP_GXP_YFLIP` | vitaslop-platform/src/gpu.rs:2608 | Flip clip Y (`VITASLOP_GXP_YFLIP`, default off). |
| `VITASLOP_GXP_ZFIX` | vitaslop-platform/src/gpu.rs:2606 | Apply the GXM (GL-style, NDC z in [-1,1]) -> WebGPU (z in [0,1]) clip-depth remap |
| `VITASLOP_HEADLESS_FRAMES` | vitaslop-desktop/src/retail.rs:573 | Headless self-check of the retail path (NO window): load `dir`, optionally drive a |
| `VITASLOP_HEADLESS_NO_TAPS` | vitaslop-desktop/src/retail.rs:579 | Headless self-check of the retail path (NO window): load `dir`, optionally drive a |
| `VITASLOP_HEADLESS_RECIPE` | vitaslop-desktop/src/retail.rs:576 | Headless self-check of the retail path (NO window): load `dir`, optionally drive a |
| `VITASLOP_HEADLESS_SHOT_EVERY` | vitaslop-desktop/src/retail.rs:582 | Headless self-check of the retail path (NO window): load `dir`, optionally drive a |
| `VITASLOP_HEADLESS_SHOT_FROM` | vitaslop-desktop/src/retail.rs:584 | Headless self-check of the retail path (NO window): load `dir`, optionally drive a |
| `VITASLOP_HEADLESS_SHOT_TO` | vitaslop-desktop/src/retail.rs:584 | Headless self-check of the retail path (NO window): load `dir`, optionally drive a |
| `VITASLOP_HEADLESS_TIMING` | vitaslop-desktop/src/retail.rs:580 | Headless self-check of the retail path (NO window): load `dir`, optionally drive a |
| `VITASLOP_HOLD_BUTTONS` | vitaslop-native/tests/retail_boot_probe.rs:96 | A minimal host world: a monotonic clock advancing one 60Hz tick per poll, no |
| `VITASLOP_HOLD_FROM` | vitaslop-native/tests/retail_boot_probe.rs:97 | A minimal host world: a monotonic clock advancing one 60Hz tick per poll, no |
| `VITASLOP_HOLD_MEM` | vitaslop-native/tests/retail_boot_probe.rs:717 | - |
| `VITASLOP_HOLD_TOUCH` | vitaslop-native/tests/retail_boot_probe.rs:114 | - |
| `VITASLOP_HOSTCALL_WATCH` | vitaslop-runtime/src/vita/mod.rs:272 | `VITASLOP_HOSTCALL_WATCH=<hex addr>[,<hex addr>...]` - print every host call that passes one |
| `VITASLOP_HOST_WRITE_WATCH` | vitaslop-runtime/src/host.rs:410 | `VITASLOP_HOST_WRITE_WATCH=<hex addr>[,...]`: report every write a HOST CALL makes to one |
| `VITASLOP_INGEST_DEBUG` | vitaslop-runtime/src/ingest/filesdb.rs:172 | Resolve every non-directory node to its full '/'-separated path (no |
| `VITASLOP_INPUT_RECIPE` | vitaslop-native/tests/retail_boot_probe.rs:390 | - |
| `VITASLOP_IO_BANDWIDTH_KIBPS` | vitaslop-runtime/src/vita/iofilemgr.rs:22 | Modelled sequential read bandwidth, in KiB per second |
| `VITASLOP_IO_PARK_THRESHOLD_US` | vitaslop-runtime/src/vita/iofilemgr.rs:110 | Smallest debt worth a context switch, in microseconds |
| `VITASLOP_IO_REQUEST_US` | vitaslop-runtime/src/vita/iofilemgr.rs:81 | Fixed per-request cost in microseconds (`VITASLOP_IO_REQUEST_US`): the command |
| `VITASLOP_LOG` | vitaslop-platform/src/gpu.rs:24 | A renderer diagnostic, at `debug` on the `vitaslop::gxm` target. |
| `VITASLOP_MAX_FRAMES` | vitaslop-native/tests/retail_boot_probe.rs:513 | - |
| `VITASLOP_MAX_ROUNDS` | vitaslop-native/tests/retail_boot_probe.rs:518 | - |
| `VITASLOP_NO_INLINE_CLIB` | vitaslop-runtime/src/vita/mod.rs:89 | `VITASLOP_NO_INLINE_CLIB`: route `sceClibMemcpy`, `sceClibMemset` and `sceClibMemcmp` |
| `VITASLOP_NO_INLINE_IMPORTS` | vitaslop-runtime/src/vita/lwsync.rs:215 | The inline form of a lightweight-sync host import: the two halves of an UNCONTENDED |
| `VITASLOP_NO_INLINE_LWMUTEX` | vitaslop-runtime/src/vita/mod.rs:117 | `VITASLOP_NO_INLINE_LWMUTEX`: route the lightweight-mutex lock and unlock through the |
| `VITASLOP_NO_INLINE_TEXTURE` | vitaslop-runtime/src/vita/mod.rs:137 | `VITASLOP_NO_INLINE_TEXTURE`: route `sceGxmSetFragmentTexture` through the host, |
| `VITASLOP_PATCH_STUBS` | vitaslop-native/tests/retail_boot_probe.rs:468 | - |
| `VITASLOP_PEEK` | vitaslop-desktop/src/retail.rs:354 | Guest memory at `addr`, for `VITASLOP_PEEK`. |
| `VITASLOP_PERF` | vitaslop-native/src/perf.rs:43 | Is perf accounting on (`VITASLOP_PERF` set)? Read once and cached. |
| `VITASLOP_PIXEL_TRACE` | vitaslop-runtime/src/render.rs:3769 | Draw one scene onto an EXISTING framebuffer and depth buffer, composing with whatever |
| `VITASLOP_POISON_UNRESOLVED_VARS` | vitaslop-runtime/src/link.rs:272 | - |
| `VITASLOP_POKE` | vitaslop-native/tests/retail_boot_probe.rs:685 | - |
| `VITASLOP_POLL_ADDR` | vitaslop-native/src/threaded.rs:1292 | Guest address to sample after each host call, from `VITASLOP_POLL_ADDR` (hex). |
| `VITASLOP_PREPOKE` | vitaslop-native/tests/retail_boot_probe.rs:491 | - |
| `VITASLOP_PRESENT_PROBE` | vitaslop-web/src/lib.rs:489 | Reads back WHAT WE PRESENTED, when `VITASLOP_PRESENT_PROBE` asks for it. |
| `VITASLOP_PVRTC_DECODE` | vitaslop-runtime/src/render.rs:4796 | Whether PVRTC decodes a whole face at a time (the default) or one texel at a time. |
| `VITASLOP_QUANTUM_CPU_US` | vitaslop-runtime/src/host.rs:9094 | Game-clock time charged for one full [`QUANTUM_FUEL`] of guest execution, in |
| `VITASLOP_QUANTUM_FUEL` | vitaslop-native/tests/retail_boot_probe.rs:427 | - |
| `VITASLOP_REGTRACE` | vitaslop-native/src/threaded.rs:916 | `VITASLOP_REGTRACE=<lo>-<hi>:<path>` - append the reg+flag file per block entry in |
| `VITASLOP_REGTRACE_MAX` | vitaslop-native/src/threaded.rs:1076 | `VITASLOP_REGTRACE_MAX=<n>` caps the register trace at `n` lines (0 = unbounded). |
| `VITASLOP_REGTRACE_WATCH` | vitaslop-native/src/threaded.rs:932 | `VITASLOP_REGTRACE_WATCH=<hex guest addr>[,<hex guest addr>...]` - append the WORD |
| `VITASLOP_ROUNDS_PER_FRAME` | vitaslop-native/tests/retail_boot_probe.rs:662 | - |
| `VITASLOP_SCAN_WORD` | vitaslop-native/tests/retail_boot_probe.rs:967 | - |
| `VITASLOP_SCENE_LIMIT` | vitaslop-native/tests/retail_boot_probe.rs:448 | - |
| `VITASLOP_SET_EVF` | vitaslop-native/tests/retail_boot_probe.rs:695 | - |
| `VITASLOP_SHOT_DIR` | vitaslop-native/tests/retail_boot_probe.rs:209 | Read and format one watched value from current guest memory. |
| `VITASLOP_SHOT_LAST` | vitaslop-native/tests/retail_boot_probe.rs:446 | - |
| `VITASLOP_SNAPSHOT` | vitaslop-native/src/threaded.rs:902 | `VITASLOP_SNAPSHOT=<hexpc>:<path>` - dump full state on first entry to block `hexpc`. |
| `VITASLOP_SNAPSHOT_BUDGET_MB` | vitaslop-runtime/src/host.rs:1949 | Byte budget for retained texture snapshots. |
| `VITASLOP_SNAPSHOT_DENSE` | vitaslop-native/src/threaded.rs:1012 | Dump the full guest state (all non-zero pages + r0..r15 + NZCV) to `path`, in the |
| `VITASLOP_SNAPSHOT_SKIP` | vitaslop-native/src/threaded.rs:964 | `VITASLOP_SNAPSHOT_SKIP=<n>` - skip the first `n` entries to the snapshot block before |
| `VITASLOP_SOFTWARE` | vitaslop-desktop/src/retail.rs:984 | - |
| `VITASLOP_SSAA` | vitaslop-platform/src/gpu.rs:7062 | Set the supersample factor: 1 (default) renders the scene straight into the caller's |
| `VITASLOP_STALL_CHUNK` | vitaslop-native/tests/retail_boot_probe.rs:545 | - |
| `VITASLOP_STALL_WAKE` | vitaslop-native/tests/retail_boot_probe.rs:544 | - |
| `VITASLOP_STALL_WAVES` | vitaslop-native/tests/retail_boot_probe.rs:548 | - |
| `VITASLOP_STRICT_DRAWS` | vitaslop-runtime/src/render.rs:4607 | Why [`RenderSceneBuilder::build`] discarded draws from a captured scene. |
| `VITASLOP_SWITCH_WHY` | vitaslop-transpiler/src/lower.rs:895 | Whether the table-branch diagnostic is on for this address |
| `VITASLOP_SW_CHAIN` | vitaslop-native/src/wgpu_render.rs:372 | `VITASLOP_GPU_CHAIN_DIR=<dir>`: write every offscreen target of the frame just |
| `VITASLOP_SW_CHAIN_DIR` | vitaslop-runtime/src/render.rs:3590 | - |
| `VITASLOP_SW_POST` | vitaslop-runtime/src/render.rs:3637 | - |
| `VITASLOP_TEXTURE_CHECK` | vitaslop-runtime/src/host.rs:1888 | How a retained texture snapshot is re-validated (`VITASLOP_TEXTURE_CHECK`): `scene` |
| `VITASLOP_TEX_CACHE_MB` | vitaslop-platform/src/gpu.rs:1491 | Upper bound on the cross-frame texture caches before they are cleared wholesale |
| `VITASLOP_TEX_COMPRESS` | vitaslop-runtime/src/render.rs:952 | Whether compressed textures reach the GPU compressed at all. |
| `VITASLOP_TRACE_BLOCKS` | vitaslop-transpiler/src/emit.rs:769 | Diagnostic per-basic-block execution tracer. |
| `VITASLOP_TRACE_EXIT` | vitaslop-native/tests/retail_boot_probe.rs:37 | - |
| `VITASLOP_TRACE_FILE` | vitaslop-runtime/src/vita/libkernel.rs:47 | Diagnostic (`RUST_LOG=vitaslop::exit=debug`): when the guest calls |
| `VITASLOP_TRACE_FUNCS` | vitaslop-native/src/threaded.rs:824 | Bind `env.svc`. |
| `VITASLOP_TRACE_INDIRECT` | vitaslop-transpiler/src/emit.rs:746 | Diagnostic indirect-call tracer. |
| `VITASLOP_TRACE_IO` | vitaslop-native/tests/retail_boot_probe.rs:34 | - |
| `VITASLOP_TRACE_ORDER` | vitaslop-runtime/src/vita/mod.rs:256 | Ordered-timeline trace (env `VITASLOP_TRACE_ORDER`): print every *meaningful* |
| `VITASLOP_TRACK_PC` | vitaslop-transpiler/src/abi.rs:192 | Exported name of the diagnostic guest-PC tracker global. |
| `VITASLOP_TRANSPILE_REPORT` | vitaslop-native/tests/retail_boot_probe.rs:30 | - |
| `VITASLOP_TRAP_HALT` | vitaslop-transpiler/src/emit.rs:815 | When `VITASLOP_TRAP_HALT` is set, a `Term::Halt` (a block that ran off the end of decoded |
| `VITASLOP_UNIFORM_WATCH` | vitaslop-runtime/src/vita/gxm.rs:1433 | `VITASLOP_UNIFORM_WATCH=<hex address>/<parameter name substring>[,...]`: report every |
| `VITASLOP_UV_DEBUG` | vitaslop-runtime/src/render.rs:3783 | Draw one scene onto an EXISTING framebuffer and depth buffer, composing with whatever |
| `VITASLOP_WASM_INDICES` | vitaslop-native/tests/retail_boot_probe.rs:44 | - |
| `VITASLOP_WASM_NAMES` | vitaslop-transpiler/src/emit.rs:472 | When `VITASLOP_WASM_NAMES` is set, emit a wasm `name` custom section labelling |
| `VITASLOP_WATCH_` | vitaslop-transpiler/src/emit.rs:658 | Number of matching store-watchpoint hits to skip before trapping (`VITASLOP_WATCH_ |
| `VITASLOP_WATCH_FROM` | vitaslop-native/tests/retail_boot_probe.rs:674 | - |
| `VITASLOP_WATCH_MEM` | vitaslop-native/tests/retail_boot_probe.rs:141 | Parse `VITASLOP_WATCH_MEM=addr:type:label,addr:type:label,...` into watches. |
| `VITASLOP_WATCH_READ` | vitaslop-transpiler/src/emit.rs:444 | Diagnostic read watchpoint. |
| `VITASLOP_WATCH_READ_` | vitaslop-transpiler/src/emit.rs:684 | Optional guest-PC EXCLUDE window for the read watchpoint (`VITASLOP_WATCH_READ_ |
| `VITASLOP_WATCH_READ_NZ` | vitaslop-transpiler/src/emit.rs:1735 | Emit the read-watchpoint trap check. |
| `VITASLOP_WATCH_READ_PC_EXCL` | vitaslop-transpiler/src/emit.rs:694 | Optional guest-PC EXCLUDE window for the read watchpoint (`VITASLOP_WATCH_READ_ |
| `VITASLOP_WATCH_READ_SKIP` | vitaslop-transpiler/src/emit.rs:520 | WASM global index of the read-watchpoint match counter, appended after the guest-PC |
| `VITASLOP_WATCH_STORE` | vitaslop-native/src/threaded.rs:937 | `VITASLOP_REGTRACE_WATCH=<hex guest addr>[,<hex guest addr>...]` - append the WORD |
| `VITASLOP_WATCH_STORE_ARM` | vitaslop-transpiler/src/emit.rs:837 | Store-watchpoint mode, from `VITASLOP_WATCH_STORE_MODE` (default `any`): |
| `VITASLOP_WATCH_STORE_LOG` | vitaslop-runtime/src/capture.rs:414 | GUEST ADDRESS the bytes above were read from, or 0 when there is no bound buffer. |
| `VITASLOP_WATCH_STORE_MODE` | vitaslop-transpiler/src/emit.rs:830 | Store-watchpoint mode, from `VITASLOP_WATCH_STORE_MODE` (default `any`): |
| `VITASLOP_WATCH_STORE_NZ` | vitaslop-transpiler/src/emit.rs:853 | `VITASLOP_WATCH_STORE_LOG` - LOG each store to the watched address (the storing |
| `VITASLOP_WATCH_STORE_SKIP` | vitaslop-transpiler/src/emit.rs:526 | WASM global index of the store-watchpoint match counter (appended after `TP_GLOBAL`). |
