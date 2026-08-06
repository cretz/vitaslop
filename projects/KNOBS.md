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

171 knobs.

| knob | read in | what it does |
|---|---|---|
| `VITASLOP_` | vitaslop-gamerun-recipes/src/bin/session.rs:78 | How often the session looks for a new command. |
| `VITASLOP_ARM_AT_FRAME` | vitaslop-native/src/threaded.rs:197 | Linear-memory offset of the "diagnostics armed" word, when this build was |
| `VITASLOP_AT9_DIR` | vitaslop-atrac9/tests/oracle.rs:66 | Decode a whole AT9 payload the way a superframe consumer does: for each |
| `VITASLOP_AUDIO_RAW` | vitaslop-runtime/src/vita/audio.rs:48 | Optional raw-s16le capture of the mixed output stream (env |
| `VITASLOP_BACKTRACE` | vitaslop-runtime/src/vita/mod.rs:100 | Print the guest call chain the first time a chosen NID is called from each thread |
| `VITASLOP_BLOCK_HIST` | vitaslop-native/src/recipe_runner.rs:161 | Dump the per-PC block-entry histogram gathered under `VITASLOP_BLOCK_HIST`, for |
| `VITASLOP_BLOCK_HIST_SEQ` | vitaslop-native/src/threaded.rs:1080 | Print the block-visit histogram gathered under `VITASLOP_BLOCK_HIST`: the `top` |
| `VITASLOP_CHAIN_DRAWS` | vitaslop-platform/src/gpu.rs:4675 | - |
| `VITASLOP_CHAIN_LIMIT` | vitaslop-platform/src/gpu.rs:5006 | Every offscreen target this renderer holds, as `(guest address, texture, w, h)`. |
| `VITASLOP_CHAIN_SKIP` | vitaslop-native/src/wgpu_render.rs:271 | - |
| `VITASLOP_CHECK_ADDRS` | vitaslop-native/tests/retail_boot_probe.rs:43 | - |
| `VITASLOP_CLOCK_TRACE` | vitaslop-runtime/src/sched.rs:455 | Called when [`pick_next`](Self::pick_next) found nothing runnable. |
| `VITASLOP_CODE_RANGE` | vitaslop-runtime/src/vita/mod.rs:83 | The guest code range scanned for the game-level caller in [`dispatch`] (env |
| `VITASLOP_CPU_SHARE` | vitaslop-native/src/recipe_runner.rs:109 | Who actually got the CPU over the run, when `VITASLOP_CPU_SHARE` is set - see |
| `VITASLOP_DBG_CALLSITES` | vitaslop-runtime/src/vita/mod.rs:76 | Diagnostic call-site profiler (env `VITASLOP_DBG_CALLSITES`): counts host calls |
| `VITASLOP_DRAW_ONLY` | vitaslop-runtime/src/render.rs:3045 | Draw one scene onto an EXISTING framebuffer and depth buffer, composing with whatever |
| `VITASLOP_DRAW_STATS` | vitaslop-runtime/src/render.rs:3034 | Draw one scene onto an EXISTING framebuffer and depth buffer, composing with whatever |
| `VITASLOP_DRV_KEY` | vitaslop-runtime/src/ingest/pfscrypt.rs:264 | `F00D(klicensee)` for the title, from `VITASLOP_DRV_KEY` (32 hex chars), or |
| `VITASLOP_DUMP_DIR` | vitaslop-runtime/src/ingest/pipeline.rs:760 | Diagnostic: decrypt the container and write named plaintext files out to |
| `VITASLOP_DUMP_DRAW` | vitaslop-native/tests/retail_boot_probe.rs:1288 | - |
| `VITASLOP_DUMP_DRAWS` | vitaslop-native/tests/retail_boot_probe.rs:1076 | - |
| `VITASLOP_DUMP_DRAW_GXP` | vitaslop-gxp-shader/tests/oracle.rs:657 | Correlate each captured vertex<->fragment PAIR (from a real draw run) to establish the |
| `VITASLOP_DUMP_DRAW_GXP_CAP` | vitaslop-runtime/src/host.rs:6506 | - |
| `VITASLOP_DUMP_DRAW_GXP_FULL` | vitaslop-runtime/src/host.rs:6607 | - |
| `VITASLOP_DUMP_EXPORTS` | vitaslop-runtime/src/link.rs:414 | - |
| `VITASLOP_DUMP_FILES` | vitaslop-runtime/src/ingest/pipeline.rs:762 | Diagnostic: decrypt the container and write named plaintext files out to |
| `VITASLOP_DUMP_FPROG` | vitaslop-runtime/src/host.rs:6394 | Diagnostic (VITASLOP_DUMP_FPROG): print the bound fragment program's sampler |
| `VITASLOP_DUMP_FUNC` | vitaslop-native/tests/retail_boot_probe.rs:318 | - |
| `VITASLOP_DUMP_GXP_BIN` | vitaslop-runtime/src/host.rs:6438 | `VITASLOP_DUMP_GXP_BIN=<dir>`: write the raw `SceGxmProgram` blobs (the whole container - |
| `VITASLOP_DUMP_IMAGE` | vitaslop-native/tests/retail_boot_probe.rs:33 | - |
| `VITASLOP_DUMP_IMPORTS` | vitaslop-native/tests/retail_boot_probe.rs:354 | - |
| `VITASLOP_DUMP_MAP` | vitaslop-native/tests/retail_boot_probe.rs:587 | - |
| `VITASLOP_DUMP_MEM` | vitaslop-native/tests/retail_boot_probe.rs:43 | - |
| `VITASLOP_DUMP_PATHS` | vitaslop-native/tests/retail_boot_probe.rs:413 | - |
| `VITASLOP_DUMP_REGION` | vitaslop-native/tests/retail_boot_probe.rs:528 | - |
| `VITASLOP_DUMP_REGION_RANGE` | vitaslop-native/tests/retail_boot_probe.rs:530 | - |
| `VITASLOP_DUMP_RENDERSCENE` | vitaslop-native/tests/retail_boot_probe.rs:1330 | - |
| `VITASLOP_DUMP_SCENES` | vitaslop-desktop/src/retail.rs:259 | Step the guest one display frame. |
| `VITASLOP_DUMP_STDOUT` | vitaslop-desktop/src/retail.rs:692 | - |
| `VITASLOP_DUMP_STUBS` | vitaslop-native/tests/retail_boot_probe.rs:32 | - |
| `VITASLOP_DUMP_TEX` | vitaslop-native/tests/retail_boot_probe.rs:1230 | - |
| `VITASLOP_DUMP_TEX_DIR` | vitaslop-runtime/src/host.rs:6613 | - |
| `VITASLOP_DUMP_TEX_MAX_TEXELS` | vitaslop-runtime/src/host.rs:6622 | - |
| `VITASLOP_DUMP_TRIS` | vitaslop-runtime/src/render.rs:3052 | Draw one scene onto an EXISTING framebuffer and depth buffer, composing with whatever |
| `VITASLOP_DUMP_VPROG` | vitaslop-runtime/src/host.rs:6233 | Diagnostic (VITASLOP_DUMP_VPROG): reflect the bound vertex program's parameter |
| `VITASLOP_FIND_WORD` | vitaslop-native/tests/retail_boot_probe.rs:63 | The span `VITASLOP_FIND_WORD` searches: from the image base up through the guest heap. |
| `VITASLOP_FORCE_READY` | vitaslop-native/tests/retail_boot_probe.rs:734 | - |
| `VITASLOP_FORCE_READY_V2` | vitaslop-native/tests/retail_boot_probe.rs:757 | - |
| `VITASLOP_FORCE_RET` | vitaslop-transpiler/src/emit.rs:328 | Diagnostic forced return. |
| `VITASLOP_FRAME_TOPUP` | vitaslop-runtime/src/host.rs:3879 | One display flip happened: bring the game clock up to one full frame of progress, |
| `VITASLOP_GAME_DIR` | vitaslop-runtime/src/ingest/mod.rs:138 | Test-fixture access. |
| `VITASLOP_GAME_ID` | vitaslop-gamerun-recipes/tests/conformance.rs:30 | - |
| `VITASLOP_GAME_PKG` | vitaslop-runtime/src/ingest/pipeline.rs:456 | Diagnostic: dump the pkg header and the extracted file tree so a new |
| `VITASLOP_GAME_WORK` | vitaslop-runtime/src/ingest/pipeline.rs:633 | The pkg + work.bin chain over a privately-supplied two-file dump: extract |
| `VITASLOP_GAME_ZIP` | vitaslop-runtime/src/ingest/zip.rs:113 | Scan backward from EOF for the EOCD signature. |
| `VITASLOP_GAP_CAP` | vitaslop-native/tests/retail_boot_probe.rs:267 | - |
| `VITASLOP_GESTURE_TYPE_MASK` | vitaslop-runtime/src/vita/gesture.rs:308 | Recognizer types allowed to report events (`VITASLOP_GESTURE_TYPE_MASK`, a bitmask |
| `VITASLOP_GPU` | vitaslop-native/tests/retail_boot_probe.rs:1394 | - |
| `VITASLOP_GPU_CHAIN_DIR` | vitaslop-native/src/wgpu_render.rs:354 | `VITASLOP_GPU_CHAIN_DIR=<dir>`: write every offscreen target of the frame just |
| `VITASLOP_GUARD_REG` | vitaslop-transpiler/src/emit.rs:241 | Diagnostic callee-saved-register guard. |
| `VITASLOP_GXM_DEPTH_ENC` | vitaslop-platform/src/gpu.rs:857 | Which value a later pass reads out of a render target's depth |
| `VITASLOP_GXM_UNIFORM_POISON` | vitaslop-runtime/src/host.rs:6992 | Diagnostic (`VITASLOP_GXM_UNIFORM_POISON=1`): fill a freshly reserved default uniform buffer |
| `VITASLOP_GXP_ALLOW_FIXED_FUNCTION` | vitaslop-platform/src/gpu.rs:3023 | Whether a shader pair the recompiler cannot translate may be drawn by the |
| `VITASLOP_GXP_BLOB` | vitaslop-gxp-shader/tests/corpus.rs:174 | Print one named blob's recompiled WGSL body and its container reflection. |
| `VITASLOP_GXP_CORPUS` | vitaslop-gxp-shader/tests/corpus.rs:66 | Print every blob's content hash beside its file name, so a `gxp pair` line from a live run |
| `VITASLOP_GXP_DEBUG` | vitaslop-platform/src/gpu.rs:3218 | Report - once per case - a GXM blend value with no exact wgpu equivalent, so the |
| `VITASLOP_GXP_DISASM` | vitaslop-gxp-shader/tests/oracle.rs:865 | Compact disassembly of one blob (named by `VITASLOP_GXP_DISASM`, matched as a filename |
| `VITASLOP_GXP_DUMP` | vitaslop-platform/src/gpu.rs:1242 | Diagnostic (`VITASLOP_GXP_KEYS=<hex>,<hex>`): recompile ONLY these shader-pair keys |
| `VITASLOP_GXP_DUMPS` | vitaslop-gxp-shader/tests/oracle.rs:145 | Histogram the raw values of named fields across every instruction of a given opcode1 |
| `VITASLOP_GXP_EXCLUDE` | vitaslop-platform/src/gpu.rs:1246 | Pairs forced down the fixed-function path (`VITASLOP_GXP_EXCLUDE`). |
| `VITASLOP_GXP_FORCE` | vitaslop-platform/src/gpu.rs:1218 | Diagnostic (`VITASLOP_GXP_FORCE`): bind a neutral fallback texture for a sampler |
| `VITASLOP_GXP_GROUP` | vitaslop-gxp-shader/tests/oracle.rs:801 | Every raw instruction word of one opcode GROUP across the whole corpus, plus a per-bit |
| `VITASLOP_GXP_INPUTS` | vitaslop-platform/src/gpu.rs:2080 | Diagnostic (`VITASLOP_GXP_INPUTS=<hex-key>[,<hex-key>]` or `=all`): print, ONCE per |
| `VITASLOP_GXP_INTERP` | vitaslop-platform/src/gpu.rs:3264 | - |
| `VITASLOP_GXP_KEYCOLOR` | vitaslop-platform/src/gpu.rs:2682 | - |
| `VITASLOP_GXP_KEYS` | vitaslop-platform/src/gpu.rs:1241 | Diagnostic (`VITASLOP_GXP_KEYS=<hex>,<hex>`): recompile ONLY these shader-pair keys |
| `VITASLOP_GXP_LIVE` | vitaslop-platform/src/gpu.rs:191 | The guest's real vertex+fragment shaders + their draw inputs, for the GXP->WGSL |
| `VITASLOP_GXP_MIPS` | vitaslop-platform/src/gpu.rs:1990 | Upload a decoded [`GxmTexture`] (linear RGBA8) to a GPU texture for the recompiler path. |
| `VITASLOP_GXP_NEGW` | vitaslop-platform/src/gpu.rs:1264 | How to choose the clip-`w` sign correction (`VITASLOP_GXP_NEGW`). |
| `VITASLOP_GXP_NOBLEND` | vitaslop-platform/src/gpu.rs:1236 | Diagnostic (`VITASLOP_GXP_NOBLEND`): force every recompiled pipeline to REPLACE with |
| `VITASLOP_GXP_NODEPTH` | vitaslop-platform/src/gpu.rs:1228 | Diagnostic (`VITASLOP_GXP_NODEPTH`): every recompiled draw keeps its real shading and |
| `VITASLOP_GXP_ONLY` | vitaslop-platform/src/gpu.rs:1208 | Render ONLY recompiled draws, skipping the fixed-function draw for any call that |
| `VITASLOP_GXP_PAIR` | vitaslop-gxp-shader/tests/corpus.rs:249 | Link one named (vertex, fragment) pair and print the COMPLETE WGSL module both stages become. |
| `VITASLOP_GXP_PAIRS` | vitaslop-gxp-shader/tests/oracle.rs:660 | Correlate each captured vertex<->fragment PAIR (from a real draw run) to establish the |
| `VITASLOP_GXP_PROBE` | vitaslop-gxp-shader/src/module.rs:243 | The `vec4<f32>` expression that reads the final colour out of register-file array `bank`, |
| `VITASLOP_GXP_RECOMPILE` | vitaslop-runtime/src/host.rs:6651 | - |
| `VITASLOP_GXP_SA` | vitaslop-platform/src/gpu.rs:2272 | Diagnostic (`VITASLOP_GXP_SA=<key>:<v/f>:<reg>=<hexword>[,...]`): replace a default-uniform |
| `VITASLOP_GXP_SOLID` | vitaslop-platform/src/gpu.rs:1224 | Diagnostic (`VITASLOP_GXP_SOLID`): every recompiled draw outputs solid magenta with |
| `VITASLOP_GXP_VPROBE` | vitaslop-gxp-shader/src/module.rs:248 | The `vec4<f32>` expression that reads the final colour out of register-file array `bank`, |
| `VITASLOP_GXP_WGSL_DIR` | vitaslop-gxp-shader/tests/oracle.rs:784 | Link each captured vertex<->fragment PAIR into a single WGSL module and prove the linked |
| `VITASLOP_GXP_YFLIP` | vitaslop-platform/src/gpu.rs:1214 | Flip clip Y (`VITASLOP_GXP_YFLIP`, default off). |
| `VITASLOP_GXP_ZFIX` | vitaslop-platform/src/gpu.rs:1212 | Apply the GXM (GL-style, NDC z in [-1,1]) -> WebGPU (z in [0,1]) clip-depth remap |
| `VITASLOP_HEADLESS_FRAMES` | vitaslop-desktop/src/retail.rs:503 | Headless self-check of the retail path (NO window): load `dir`, optionally drive a |
| `VITASLOP_HEADLESS_NO_TAPS` | vitaslop-desktop/src/retail.rs:509 | Headless self-check of the retail path (NO window): load `dir`, optionally drive a |
| `VITASLOP_HEADLESS_RECIPE` | vitaslop-desktop/src/retail.rs:506 | Headless self-check of the retail path (NO window): load `dir`, optionally drive a |
| `VITASLOP_HEADLESS_SHOT_EVERY` | vitaslop-desktop/src/retail.rs:512 | Headless self-check of the retail path (NO window): load `dir`, optionally drive a |
| `VITASLOP_HEADLESS_TIMING` | vitaslop-desktop/src/retail.rs:510 | Headless self-check of the retail path (NO window): load `dir`, optionally drive a |
| `VITASLOP_HOLD_BUTTONS` | vitaslop-native/tests/retail_boot_probe.rs:96 | A minimal host world: a monotonic clock advancing one 60Hz tick per poll, no |
| `VITASLOP_HOLD_FROM` | vitaslop-native/tests/retail_boot_probe.rs:97 | A minimal host world: a monotonic clock advancing one 60Hz tick per poll, no |
| `VITASLOP_HOLD_MEM` | vitaslop-native/tests/retail_boot_probe.rs:713 | - |
| `VITASLOP_HOLD_TOUCH` | vitaslop-native/tests/retail_boot_probe.rs:114 | - |
| `VITASLOP_HOSTCALL_WATCH` | vitaslop-runtime/src/vita/mod.rs:133 | `VITASLOP_HOSTCALL_WATCH=<hex addr>[,<hex addr>...]` - print every host call that passes one |
| `VITASLOP_HOST_WRITE_WATCH` | vitaslop-runtime/src/host.rs:319 | `VITASLOP_HOST_WRITE_WATCH=<hex addr>[,...]`: report every write a HOST CALL makes to one |
| `VITASLOP_INGEST_DEBUG` | vitaslop-runtime/src/ingest/filesdb.rs:172 | Resolve every non-directory node to its full '/'-separated path (no |
| `VITASLOP_INPUT_RECIPE` | vitaslop-native/tests/retail_boot_probe.rs:389 | - |
| `VITASLOP_IO_BANDWIDTH_KIBPS` | vitaslop-runtime/src/vita/iofilemgr.rs:22 | Modelled sequential read bandwidth, in KiB per second |
| `VITASLOP_IO_PARK_THRESHOLD_US` | vitaslop-runtime/src/vita/iofilemgr.rs:110 | Smallest debt worth a context switch, in microseconds |
| `VITASLOP_IO_REQUEST_US` | vitaslop-runtime/src/vita/iofilemgr.rs:81 | Fixed per-request cost in microseconds (`VITASLOP_IO_REQUEST_US`): the command |
| `VITASLOP_MAX_FRAMES` | vitaslop-native/tests/retail_boot_probe.rs:512 | - |
| `VITASLOP_MAX_ROUNDS` | vitaslop-native/tests/retail_boot_probe.rs:517 | - |
| `VITASLOP_NO_INLINE_IMPORTS` | vitaslop-runtime/src/vita/mod.rs:61 | `VITASLOP_NO_INLINE_IMPORTS`: route every host call through the host, even the |
| `VITASLOP_PATCH_STUBS` | vitaslop-native/tests/retail_boot_probe.rs:467 | - |
| `VITASLOP_PEEK` | vitaslop-desktop/src/retail.rs:334 | Guest memory at `addr`, for `VITASLOP_PEEK`. |
| `VITASLOP_PERF` | vitaslop-native/src/perf.rs:43 | Is perf accounting on (`VITASLOP_PERF` set)? Read once and cached. |
| `VITASLOP_PIXEL_TRACE` | vitaslop-runtime/src/render.rs:3026 | Draw one scene onto an EXISTING framebuffer and depth buffer, composing with whatever |
| `VITASLOP_POISON_UNRESOLVED_VARS` | vitaslop-runtime/src/link.rs:272 | - |
| `VITASLOP_POKE` | vitaslop-native/tests/retail_boot_probe.rs:681 | - |
| `VITASLOP_POLL_ADDR` | vitaslop-native/src/threaded.rs:1166 | Guest address to sample after each host call, from `VITASLOP_POLL_ADDR` (hex). |
| `VITASLOP_PREPOKE` | vitaslop-native/tests/retail_boot_probe.rs:490 | - |
| `VITASLOP_QUANTUM_CPU_US` | vitaslop-runtime/src/host.rs:7708 | Game-clock time charged for one scheduler quantum of guest execution, in |
| `VITASLOP_QUANTUM_FUEL` | vitaslop-native/tests/retail_boot_probe.rs:426 | - |
| `VITASLOP_REGTRACE` | vitaslop-native/src/threaded.rs:795 | `VITASLOP_REGTRACE=<lo>-<hi>:<path>` - append the reg+flag file per block entry in |
| `VITASLOP_REGTRACE_MAX` | vitaslop-native/src/threaded.rs:955 | `VITASLOP_REGTRACE_MAX=<n>` caps the register trace at `n` lines (0 = unbounded). |
| `VITASLOP_REGTRACE_WATCH` | vitaslop-native/src/threaded.rs:811 | `VITASLOP_REGTRACE_WATCH=<hex guest addr>[,<hex guest addr>...]` - append the WORD |
| `VITASLOP_ROUNDS_PER_FRAME` | vitaslop-native/tests/retail_boot_probe.rs:658 | - |
| `VITASLOP_SCAN_WORD` | vitaslop-native/tests/retail_boot_probe.rs:963 | - |
| `VITASLOP_SCENE_LIMIT` | vitaslop-native/tests/retail_boot_probe.rs:447 | - |
| `VITASLOP_SET_EVF` | vitaslop-native/tests/retail_boot_probe.rs:691 | - |
| `VITASLOP_SHOT_DIR` | vitaslop-native/tests/retail_boot_probe.rs:208 | Read and format one watched value from current guest memory. |
| `VITASLOP_SHOT_LAST` | vitaslop-native/tests/retail_boot_probe.rs:445 | - |
| `VITASLOP_SNAPSHOT` | vitaslop-native/src/threaded.rs:781 | `VITASLOP_SNAPSHOT=<hexpc>:<path>` - dump full state on first entry to block `hexpc`. |
| `VITASLOP_SNAPSHOT_DENSE` | vitaslop-native/src/threaded.rs:891 | Dump the full guest state (all non-zero pages + r0..r15 + NZCV) to `path`, in the |
| `VITASLOP_SNAPSHOT_SKIP` | vitaslop-native/src/threaded.rs:843 | `VITASLOP_SNAPSHOT_SKIP=<n>` - skip the first `n` entries to the snapshot block before |
| `VITASLOP_SOFTWARE` | vitaslop-desktop/src/retail.rs:730 | - |
| `VITASLOP_SSAA` | vitaslop-platform/src/gpu.rs:3990 | Set the supersample factor: 1 (default) renders the scene straight into the caller's |
| `VITASLOP_STALL_CHUNK` | vitaslop-native/tests/retail_boot_probe.rs:544 | - |
| `VITASLOP_STALL_WAKE` | vitaslop-native/tests/retail_boot_probe.rs:543 | - |
| `VITASLOP_STALL_WAVES` | vitaslop-native/tests/retail_boot_probe.rs:547 | - |
| `VITASLOP_STRICT_DRAWS` | vitaslop-runtime/src/render.rs:3506 | Why [`RenderSceneBuilder::build`] discarded draws from a captured scene. |
| `VITASLOP_SWITCH_WHY` | vitaslop-transpiler/src/lower.rs:895 | Whether the table-branch diagnostic is on for this address |
| `VITASLOP_SW_CHAIN` | vitaslop-native/src/wgpu_render.rs:357 | `VITASLOP_GPU_CHAIN_DIR=<dir>`: write every offscreen target of the frame just |
| `VITASLOP_SW_CHAIN_DIR` | vitaslop-runtime/src/render.rs:2850 | - |
| `VITASLOP_SW_POST` | vitaslop-runtime/src/render.rs:2897 | - |
| `VITASLOP_TRACE_BLOCKS` | vitaslop-transpiler/src/emit.rs:308 | Diagnostic per-basic-block execution tracer. |
| `VITASLOP_TRACE_EXIT` | vitaslop-native/tests/retail_boot_probe.rs:37 | - |
| `VITASLOP_TRACE_FILE` | vitaslop-runtime/src/vita/libkernel.rs:47 | Diagnostic (`RUST_LOG=vitaslop::exit=debug`): when the guest calls |
| `VITASLOP_TRACE_FUNCS` | vitaslop-native/src/threaded.rs:703 | Bind `env.svc`. |
| `VITASLOP_TRACE_INDIRECT` | vitaslop-transpiler/src/emit.rs:285 | Diagnostic indirect-call tracer. |
| `VITASLOP_TRACE_IO` | vitaslop-native/tests/retail_boot_probe.rs:34 | - |
| `VITASLOP_TRACE_ORDER` | vitaslop-runtime/src/vita/mod.rs:117 | Ordered-timeline trace (env `VITASLOP_TRACE_ORDER`): print every *meaningful* |
| `VITASLOP_TRACK_PC` | vitaslop-transpiler/src/abi.rs:192 | Exported name of the diagnostic guest-PC tracker global. |
| `VITASLOP_TRANSPILE_REPORT` | vitaslop-native/tests/retail_boot_probe.rs:30 | - |
| `VITASLOP_TRAP_HALT` | vitaslop-transpiler/src/emit.rs:354 | When `VITASLOP_TRAP_HALT` is set, a `Term::Halt` (a block that ran off the end of decoded |
| `VITASLOP_UNIFORM_WATCH` | vitaslop-runtime/src/vita/gxm.rs:1170 | `VITASLOP_UNIFORM_WATCH=<hex address>/<parameter name substring>[,...]`: report every |
| `VITASLOP_UV_DEBUG` | vitaslop-runtime/src/render.rs:3040 | Draw one scene onto an EXISTING framebuffer and depth buffer, composing with whatever |
| `VITASLOP_WASM_INDICES` | vitaslop-native/tests/retail_boot_probe.rs:44 | - |
| `VITASLOP_WASM_NAMES` | vitaslop-transpiler/src/emit.rs:139 | When `VITASLOP_WASM_NAMES` is set, emit a wasm `name` custom section labelling |
| `VITASLOP_WATCH_` | vitaslop-transpiler/src/emit.rs:197 | Number of matching store-watchpoint hits to skip before trapping (`VITASLOP_WATCH_ |
| `VITASLOP_WATCH_FROM` | vitaslop-native/tests/retail_boot_probe.rs:670 | - |
| `VITASLOP_WATCH_MEM` | vitaslop-native/tests/retail_boot_probe.rs:141 | Parse `VITASLOP_WATCH_MEM=addr:type:label,addr:type:label,...` into watches. |
| `VITASLOP_WATCH_READ` | vitaslop-transpiler/src/emit.rs:111 | Diagnostic read watchpoint. |
| `VITASLOP_WATCH_READ_` | vitaslop-transpiler/src/emit.rs:223 | Optional guest-PC EXCLUDE window for the read watchpoint (`VITASLOP_WATCH_READ_ |
| `VITASLOP_WATCH_READ_NZ` | vitaslop-transpiler/src/emit.rs:1111 | Emit the read-watchpoint trap check. |
| `VITASLOP_WATCH_READ_PC_EXCL` | vitaslop-transpiler/src/emit.rs:233 | Optional guest-PC EXCLUDE window for the read watchpoint (`VITASLOP_WATCH_READ_ |
| `VITASLOP_WATCH_READ_SKIP` | vitaslop-transpiler/src/emit.rs:187 | WASM global index of the read-watchpoint match counter, appended after the guest-PC |
| `VITASLOP_WATCH_STORE` | vitaslop-native/src/threaded.rs:816 | `VITASLOP_REGTRACE_WATCH=<hex guest addr>[,<hex guest addr>...]` - append the WORD |
| `VITASLOP_WATCH_STORE_ARM` | vitaslop-transpiler/src/emit.rs:376 | Store-watchpoint mode, from `VITASLOP_WATCH_STORE_MODE` (default `any`): |
| `VITASLOP_WATCH_STORE_LOG` | vitaslop-transpiler/src/emit.rs:379 | `VITASLOP_WATCH_STORE_LOG` - LOG each store to the watched address (the storing |
| `VITASLOP_WATCH_STORE_MODE` | vitaslop-transpiler/src/emit.rs:369 | Store-watchpoint mode, from `VITASLOP_WATCH_STORE_MODE` (default `any`): |
| `VITASLOP_WATCH_STORE_NZ` | vitaslop-transpiler/src/emit.rs:392 | `VITASLOP_WATCH_STORE_LOG` - LOG each store to the watched address (the storing |
| `VITASLOP_WATCH_STORE_SKIP` | vitaslop-transpiler/src/emit.rs:193 | WASM global index of the store-watchpoint match counter (appended after `TP_GLOBAL`). |
