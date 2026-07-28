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

147 knobs.

| knob | read in | what it does |
|---|---|---|
| `VITASLOP_` | vitaslop-gamerun-recipes/src/bin/session.rs:78 | How often the session looks for a new command. |
| `VITASLOP_ARM_AT_FRAME` | vitaslop-native/src/threaded.rs:197 | Linear-memory offset of the "diagnostics armed" word, when this build was |
| `VITASLOP_AT9_DIR` | vitaslop-atrac9/tests/oracle.rs:66 | Decode a whole AT9 payload the way a superframe consumer does: for each |
| `VITASLOP_AUDIO_RAW` | vitaslop-runtime/src/vita/audio.rs:48 | Optional raw-s16le capture of the mixed output stream (env |
| `VITASLOP_BACKTRACE` | vitaslop-runtime/src/vita/mod.rs:60 | Print the guest call chain the first time a chosen NID is called from each thread |
| `VITASLOP_BLOCK_HIST` | vitaslop-native/src/threaded.rs:1053 | Print the block-visit histogram gathered under `VITASLOP_BLOCK_HIST`: the `top` |
| `VITASLOP_BLOCK_HIST_SEQ` | vitaslop-native/src/threaded.rs:1062 | Print the block-visit histogram gathered under `VITASLOP_BLOCK_HIST`: the `top` |
| `VITASLOP_CHAIN_DRAWS` | vitaslop-platform/src/gpu.rs:2710 | One scene into one target. |
| `VITASLOP_CHAIN_LIMIT` | vitaslop-native/src/wgpu_render.rs:261 | Render a whole captured FRAME - every scene the guest submitted between flips, in |
| `VITASLOP_CHAIN_SKIP` | vitaslop-native/src/wgpu_render.rs:271 | - |
| `VITASLOP_CHECK_ADDRS` | vitaslop-native/tests/retail_boot_probe.rs:43 | - |
| `VITASLOP_CLOCK_TRACE` | vitaslop-runtime/src/sched.rs:358 | Called when [`pick_next`](Self::pick_next) found nothing runnable. |
| `VITASLOP_CODE_RANGE` | vitaslop-runtime/src/vita/mod.rs:43 | The guest code range scanned for the game-level caller in [`dispatch`] (env |
| `VITASLOP_DBG_CALLSITES` | vitaslop-runtime/src/vita/mod.rs:36 | Diagnostic call-site profiler (env `VITASLOP_DBG_CALLSITES`): counts host calls |
| `VITASLOP_DRAW_ONLY` | vitaslop-runtime/src/render.rs:2779 | Draw one scene onto an EXISTING framebuffer and depth buffer, composing with whatever |
| `VITASLOP_DRAW_STATS` | vitaslop-runtime/src/render.rs:2768 | Draw one scene onto an EXISTING framebuffer and depth buffer, composing with whatever |
| `VITASLOP_DRV_KEY` | vitaslop-runtime/src/ingest/pfscrypt.rs:264 | `F00D(klicensee)` for the title, from `VITASLOP_DRV_KEY` (32 hex chars), or |
| `VITASLOP_DUMP_DIR` | vitaslop-runtime/src/ingest/pipeline.rs:760 | Diagnostic: decrypt the container and write named plaintext files out to |
| `VITASLOP_DUMP_DRAW` | vitaslop-native/tests/retail_boot_probe.rs:1226 | - |
| `VITASLOP_DUMP_DRAWS` | vitaslop-native/tests/retail_boot_probe.rs:1014 | - |
| `VITASLOP_DUMP_DRAW_GXP` | vitaslop-gxp-shader/tests/oracle.rs:657 | Correlate each captured vertex<->fragment PAIR (from a real draw run) to establish the |
| `VITASLOP_DUMP_DRAW_GXP_CAP` | vitaslop-runtime/src/host.rs:4519 | - |
| `VITASLOP_DUMP_DRAW_GXP_FULL` | vitaslop-runtime/src/host.rs:4604 | - |
| `VITASLOP_DUMP_EXPORTS` | vitaslop-runtime/src/link.rs:382 | - |
| `VITASLOP_DUMP_FILES` | vitaslop-runtime/src/ingest/pipeline.rs:762 | Diagnostic: decrypt the container and write named plaintext files out to |
| `VITASLOP_DUMP_FPROG` | vitaslop-runtime/src/host.rs:4407 | Diagnostic (VITASLOP_DUMP_FPROG): print the bound fragment program's sampler |
| `VITASLOP_DUMP_FUNC` | vitaslop-native/tests/retail_boot_probe.rs:318 | - |
| `VITASLOP_DUMP_GXP_BIN` | vitaslop-runtime/src/host.rs:4451 | `VITASLOP_DUMP_GXP_BIN=<dir>`: write the raw `SceGxmProgram` blobs (the whole container - |
| `VITASLOP_DUMP_IMAGE` | vitaslop-native/tests/retail_boot_probe.rs:33 | - |
| `VITASLOP_DUMP_IMPORTS` | vitaslop-native/tests/retail_boot_probe.rs:354 | - |
| `VITASLOP_DUMP_MAP` | vitaslop-native/tests/retail_boot_probe.rs:556 | - |
| `VITASLOP_DUMP_MEM` | vitaslop-native/tests/retail_boot_probe.rs:43 | - |
| `VITASLOP_DUMP_PATHS` | vitaslop-native/tests/retail_boot_probe.rs:412 | - |
| `VITASLOP_DUMP_REGION` | vitaslop-native/tests/retail_boot_probe.rs:497 | - |
| `VITASLOP_DUMP_REGION_RANGE` | vitaslop-native/tests/retail_boot_probe.rs:499 | - |
| `VITASLOP_DUMP_RENDERSCENE` | vitaslop-native/tests/retail_boot_probe.rs:1268 | - |
| `VITASLOP_DUMP_SCENES` | vitaslop-desktop/src/retail.rs:258 | Step the guest one display frame. |
| `VITASLOP_DUMP_STDOUT` | vitaslop-desktop/src/retail.rs:649 | - |
| `VITASLOP_DUMP_STUBS` | vitaslop-native/tests/retail_boot_probe.rs:32 | - |
| `VITASLOP_DUMP_TEX` | vitaslop-native/tests/retail_boot_probe.rs:1168 | - |
| `VITASLOP_DUMP_TEX_DIR` | vitaslop-runtime/src/host.rs:4610 | - |
| `VITASLOP_DUMP_TEX_MAX_TEXELS` | vitaslop-runtime/src/host.rs:4619 | - |
| `VITASLOP_DUMP_TRIS` | vitaslop-runtime/src/render.rs:2786 | Draw one scene onto an EXISTING framebuffer and depth buffer, composing with whatever |
| `VITASLOP_DUMP_VPROG` | vitaslop-runtime/src/host.rs:4281 | Diagnostic (VITASLOP_DUMP_VPROG): reflect the bound vertex program's parameter |
| `VITASLOP_FIND_WORD` | vitaslop-native/tests/retail_boot_probe.rs:63 | The span `VITASLOP_FIND_WORD` searches: from the image base up through the guest heap. |
| `VITASLOP_FORCE_READY` | vitaslop-native/tests/retail_boot_probe.rs:703 | - |
| `VITASLOP_FORCE_READY_V2` | vitaslop-native/tests/retail_boot_probe.rs:726 | - |
| `VITASLOP_FORCE_RET` | vitaslop-transpiler/src/emit.rs:328 | Diagnostic forced return. |
| `VITASLOP_GAME_DIR` | vitaslop-runtime/src/ingest/mod.rs:138 | Test-fixture access. |
| `VITASLOP_GAME_ID` | vitaslop-gamerun-recipes/tests/conformance.rs:30 | - |
| `VITASLOP_GAME_PKG` | vitaslop-runtime/src/ingest/pipeline.rs:456 | Diagnostic: dump the pkg header and the extracted file tree so a new |
| `VITASLOP_GAME_WORK` | vitaslop-runtime/src/ingest/pipeline.rs:633 | The pkg + work.bin chain over a privately-supplied two-file dump: extract |
| `VITASLOP_GAME_ZIP` | vitaslop-runtime/src/ingest/zip.rs:113 | Scan backward from EOF for the EOCD signature. |
| `VITASLOP_GAP_CAP` | vitaslop-native/tests/retail_boot_probe.rs:267 | - |
| `VITASLOP_GPU` | vitaslop-native/tests/retail_boot_probe.rs:1332 | - |
| `VITASLOP_GUARD_REG` | vitaslop-transpiler/src/emit.rs:241 | Diagnostic callee-saved-register guard. |
| `VITASLOP_GXM_UNIFORM_POISON` | vitaslop-runtime/src/host.rs:4968 | Diagnostic (`VITASLOP_GXM_UNIFORM_POISON=1`): fill a freshly reserved default uniform buffer |
| `VITASLOP_GXP_DEBUG` | vitaslop-platform/src/gpu.rs:1676 | Link a guest shader pair and build its two pipeline variants + bind-group layouts. |
| `VITASLOP_GXP_DISASM` | vitaslop-gxp-shader/tests/oracle.rs:798 | Compact disassembly of one blob (named by `VITASLOP_GXP_DISASM`, matched as a filename |
| `VITASLOP_GXP_DUMP` | vitaslop-platform/src/gpu.rs:1031 | Diagnostic (`VITASLOP_GXP_KEYS=<hex>,<hex>`): recompile ONLY these shader-pair keys |
| `VITASLOP_GXP_DUMPS` | vitaslop-gxp-shader/tests/oracle.rs:145 | Histogram the raw values of named fields across every instruction of a given opcode1 |
| `VITASLOP_GXP_EXCLUDE` | vitaslop-platform/src/gpu.rs:1035 | Pairs forced down the fixed-function path (`VITASLOP_GXP_EXCLUDE`). |
| `VITASLOP_GXP_FORCE` | vitaslop-platform/src/gpu.rs:1012 | Diagnostic (`VITASLOP_GXP_FORCE`): bind a neutral fallback texture for a sampler |
| `VITASLOP_GXP_INTERP` | vitaslop-platform/src/gpu.rs:1722 | - |
| `VITASLOP_GXP_KEYS` | vitaslop-platform/src/gpu.rs:1030 | Diagnostic (`VITASLOP_GXP_KEYS=<hex>,<hex>`): recompile ONLY these shader-pair keys |
| `VITASLOP_GXP_LIVE` | vitaslop-platform/src/gpu.rs:177 | The guest's real vertex+fragment shaders + their draw inputs, for the GXP->WGSL |
| `VITASLOP_GXP_NODEPTH` | vitaslop-platform/src/gpu.rs:1022 | Diagnostic (`VITASLOP_GXP_NODEPTH`): every recompiled draw keeps its real shading and |
| `VITASLOP_GXP_ONLY` | vitaslop-platform/src/gpu.rs:1002 | Render ONLY recompiled draws, skipping the fixed-function draw for any call that |
| `VITASLOP_GXP_PAIRS` | vitaslop-gxp-shader/tests/oracle.rs:660 | Correlate each captured vertex<->fragment PAIR (from a real draw run) to establish the |
| `VITASLOP_GXP_PROBE` | vitaslop-gxp-shader/src/module.rs:237 | The `vec4<f32>` expression that reads the final colour out of register-file array `bank`, |
| `VITASLOP_GXP_RECOMPILE` | vitaslop-runtime/src/host.rs:4648 | - |
| `VITASLOP_GXP_SOLID` | vitaslop-platform/src/gpu.rs:1018 | Diagnostic (`VITASLOP_GXP_SOLID`): every recompiled draw outputs solid magenta with |
| `VITASLOP_GXP_WGSL_DIR` | vitaslop-gxp-shader/tests/oracle.rs:784 | Link each captured vertex<->fragment PAIR into a single WGSL module and prove the linked |
| `VITASLOP_GXP_YFLIP` | vitaslop-platform/src/gpu.rs:1008 | Flip clip Y (`VITASLOP_GXP_YFLIP`, default off). |
| `VITASLOP_GXP_ZFIX` | vitaslop-platform/src/gpu.rs:1006 | Apply the GXM (GL-style, NDC z in [-1,1]) -> WebGPU (z in [0,1]) clip-depth remap |
| `VITASLOP_HEADLESS_FRAMES` | vitaslop-desktop/src/retail.rs:464 | Headless self-check of the retail path (NO window): load `dir`, optionally drive a |
| `VITASLOP_HEADLESS_NO_TAPS` | vitaslop-desktop/src/retail.rs:470 | Headless self-check of the retail path (NO window): load `dir`, optionally drive a |
| `VITASLOP_HEADLESS_RECIPE` | vitaslop-desktop/src/retail.rs:467 | Headless self-check of the retail path (NO window): load `dir`, optionally drive a |
| `VITASLOP_HEADLESS_SHOT_EVERY` | vitaslop-desktop/src/retail.rs:473 | Headless self-check of the retail path (NO window): load `dir`, optionally drive a |
| `VITASLOP_HEADLESS_TIMING` | vitaslop-desktop/src/retail.rs:471 | Headless self-check of the retail path (NO window): load `dir`, optionally drive a |
| `VITASLOP_HOLD_BUTTONS` | vitaslop-native/tests/retail_boot_probe.rs:96 | A minimal host world: a monotonic clock advancing one 60Hz tick per poll, no |
| `VITASLOP_HOLD_FROM` | vitaslop-native/tests/retail_boot_probe.rs:97 | A minimal host world: a monotonic clock advancing one 60Hz tick per poll, no |
| `VITASLOP_HOLD_MEM` | vitaslop-native/tests/retail_boot_probe.rs:682 | - |
| `VITASLOP_HOLD_TOUCH` | vitaslop-native/tests/retail_boot_probe.rs:114 | - |
| `VITASLOP_INGEST_DEBUG` | vitaslop-runtime/src/ingest/filesdb.rs:172 | Resolve every non-directory node to its full '/'-separated path (no |
| `VITASLOP_INPUT_RECIPE` | vitaslop-native/tests/retail_boot_probe.rs:389 | - |
| `VITASLOP_IO_BANDWIDTH_KIBPS` | vitaslop-runtime/src/vita/iofilemgr.rs:22 | Modelled sequential read bandwidth, in KiB per second |
| `VITASLOP_IO_PARK_THRESHOLD_US` | vitaslop-runtime/src/vita/iofilemgr.rs:74 | Smallest debt worth a context switch, in microseconds |
| `VITASLOP_IO_REQUEST_US` | vitaslop-runtime/src/vita/iofilemgr.rs:45 | Fixed per-request cost in microseconds (`VITASLOP_IO_REQUEST_US`): the command |
| `VITASLOP_MAX_FRAMES` | vitaslop-native/tests/retail_boot_probe.rs:481 | - |
| `VITASLOP_MAX_ROUNDS` | vitaslop-native/tests/retail_boot_probe.rs:486 | - |
| `VITASLOP_NO_INLINE_IMPORTS` | vitaslop-runtime/src/vita/gxm.rs:91 | `VITASLOP_NO_INLINE_IMPORTS`: route every host call through the host, even the |
| `VITASLOP_PATCH_STUBS` | vitaslop-native/tests/retail_boot_probe.rs:436 | - |
| `VITASLOP_PERF` | vitaslop-native/src/perf.rs:43 | Is perf accounting on (`VITASLOP_PERF` set)? Read once and cached. |
| `VITASLOP_PIXEL_TRACE` | vitaslop-runtime/src/render.rs:2760 | Draw one scene onto an EXISTING framebuffer and depth buffer, composing with whatever |
| `VITASLOP_POISON_UNRESOLVED_VARS` | vitaslop-runtime/src/link.rs:244 | - |
| `VITASLOP_POKE` | vitaslop-native/tests/retail_boot_probe.rs:650 | - |
| `VITASLOP_POLL_ADDR` | vitaslop-native/src/threaded.rs:1148 | Guest address to sample after each host call, from `VITASLOP_POLL_ADDR` (hex). |
| `VITASLOP_PREPOKE` | vitaslop-native/tests/retail_boot_probe.rs:459 | - |
| `VITASLOP_QUANTUM_FUEL` | vitaslop-native/tests/retail_boot_probe.rs:425 | - |
| `VITASLOP_REGTRACE` | vitaslop-native/src/threaded.rs:777 | `VITASLOP_REGTRACE=<lo>-<hi>:<path>` - append the reg+flag file per block entry in |
| `VITASLOP_REGTRACE_MAX` | vitaslop-native/src/threaded.rs:937 | `VITASLOP_REGTRACE_MAX=<n>` caps the register trace at `n` lines (0 = unbounded). |
| `VITASLOP_REGTRACE_WATCH` | vitaslop-native/src/threaded.rs:793 | `VITASLOP_REGTRACE_WATCH=<hex guest addr>[,<hex guest addr>...]` - append the WORD |
| `VITASLOP_ROUNDS_PER_FRAME` | vitaslop-native/tests/retail_boot_probe.rs:627 | - |
| `VITASLOP_SCAN_WORD` | vitaslop-native/tests/retail_boot_probe.rs:932 | - |
| `VITASLOP_SET_EVF` | vitaslop-native/tests/retail_boot_probe.rs:660 | - |
| `VITASLOP_SHOT_DIR` | vitaslop-native/tests/retail_boot_probe.rs:208 | Read and format one watched value from current guest memory. |
| `VITASLOP_SHOT_LAST` | vitaslop-native/tests/retail_boot_probe.rs:1212 | - |
| `VITASLOP_SNAPSHOT` | vitaslop-native/src/threaded.rs:763 | `VITASLOP_SNAPSHOT=<hexpc>:<path>` - dump full state on first entry to block `hexpc`. |
| `VITASLOP_SNAPSHOT_DENSE` | vitaslop-native/src/threaded.rs:873 | Dump the full guest state (all non-zero pages + r0..r15 + NZCV) to `path`, in the |
| `VITASLOP_SNAPSHOT_SKIP` | vitaslop-native/src/threaded.rs:825 | `VITASLOP_SNAPSHOT_SKIP=<n>` - skip the first `n` entries to the snapshot block before |
| `VITASLOP_SOFTWARE` | vitaslop-desktop/src/retail.rs:687 | - |
| `VITASLOP_SSAA` | vitaslop-platform/src/gpu.rs:2232 | Set the supersample factor: 1 (default) renders the scene straight into the caller's |
| `VITASLOP_STALL_CHUNK` | vitaslop-native/tests/retail_boot_probe.rs:513 | - |
| `VITASLOP_STALL_WAKE` | vitaslop-native/tests/retail_boot_probe.rs:512 | - |
| `VITASLOP_STALL_WAVES` | vitaslop-native/tests/retail_boot_probe.rs:516 | - |
| `VITASLOP_STRICT_DRAWS` | vitaslop-runtime/src/render.rs:3231 | Why [`RenderSceneBuilder::build`] discarded draws from a captured scene. |
| `VITASLOP_SW_CHAIN` | vitaslop-runtime/src/render.rs:2549 | Rasterize a whole FRAME - every scene the guest submitted between two display flips, |
| `VITASLOP_SW_CHAIN_DIR` | vitaslop-runtime/src/render.rs:2600 | - |
| `VITASLOP_SW_POST` | vitaslop-runtime/src/render.rs:2635 | - |
| `VITASLOP_TRACE_BLOCKS` | vitaslop-transpiler/src/emit.rs:308 | Diagnostic per-basic-block execution tracer. |
| `VITASLOP_TRACE_EXIT` | vitaslop-native/tests/retail_boot_probe.rs:37 | - |
| `VITASLOP_TRACE_FILE` | vitaslop-runtime/src/vita/libkernel.rs:47 | Diagnostic (`RUST_LOG=vitaslop::exit=debug`): when the guest calls |
| `VITASLOP_TRACE_FUNCS` | vitaslop-native/src/threaded.rs:685 | Bind `env.svc`. |
| `VITASLOP_TRACE_INDIRECT` | vitaslop-transpiler/src/emit.rs:285 | Diagnostic indirect-call tracer. |
| `VITASLOP_TRACE_IO` | vitaslop-native/tests/retail_boot_probe.rs:34 | - |
| `VITASLOP_TRACE_ORDER` | vitaslop-runtime/src/vita/mod.rs:77 | Ordered-timeline trace (env `VITASLOP_TRACE_ORDER`): print every *meaningful* |
| `VITASLOP_TRACK_PC` | vitaslop-transpiler/src/abi.rs:192 | Exported name of the diagnostic guest-PC tracker global. |
| `VITASLOP_TRANSPILE_REPORT` | vitaslop-native/tests/retail_boot_probe.rs:30 | - |
| `VITASLOP_TRAP_HALT` | vitaslop-transpiler/src/emit.rs:354 | When `VITASLOP_TRAP_HALT` is set, a `Term::Halt` (a block that ran off the end of decoded |
| `VITASLOP_UV_DEBUG` | vitaslop-runtime/src/render.rs:2774 | Draw one scene onto an EXISTING framebuffer and depth buffer, composing with whatever |
| `VITASLOP_WASM_INDICES` | vitaslop-native/tests/retail_boot_probe.rs:44 | - |
| `VITASLOP_WASM_NAMES` | vitaslop-transpiler/src/emit.rs:139 | When `VITASLOP_WASM_NAMES` is set, emit a wasm `name` custom section labelling |
| `VITASLOP_WATCH_` | vitaslop-transpiler/src/emit.rs:197 | Number of matching store-watchpoint hits to skip before trapping (`VITASLOP_WATCH_ |
| `VITASLOP_WATCH_FROM` | vitaslop-native/tests/retail_boot_probe.rs:639 | - |
| `VITASLOP_WATCH_MEM` | vitaslop-native/tests/retail_boot_probe.rs:141 | Parse `VITASLOP_WATCH_MEM=addr:type:label,addr:type:label,...` into watches. |
| `VITASLOP_WATCH_READ` | vitaslop-transpiler/src/emit.rs:111 | Diagnostic read watchpoint. |
| `VITASLOP_WATCH_READ_` | vitaslop-transpiler/src/emit.rs:223 | Optional guest-PC EXCLUDE window for the read watchpoint (`VITASLOP_WATCH_READ_ |
| `VITASLOP_WATCH_READ_NZ` | vitaslop-transpiler/src/emit.rs:1081 | Emit the read-watchpoint trap check. |
| `VITASLOP_WATCH_READ_PC_EXCL` | vitaslop-transpiler/src/emit.rs:233 | Optional guest-PC EXCLUDE window for the read watchpoint (`VITASLOP_WATCH_READ_ |
| `VITASLOP_WATCH_READ_SKIP` | vitaslop-transpiler/src/emit.rs:187 | WASM global index of the read-watchpoint match counter, appended after the guest-PC |
| `VITASLOP_WATCH_STORE` | vitaslop-native/src/threaded.rs:798 | `VITASLOP_REGTRACE_WATCH=<hex guest addr>[,<hex guest addr>...]` - append the WORD |
| `VITASLOP_WATCH_STORE_ARM` | vitaslop-transpiler/src/emit.rs:376 | Store-watchpoint mode, from `VITASLOP_WATCH_STORE_MODE` (default `any`): |
| `VITASLOP_WATCH_STORE_LOG` | vitaslop-transpiler/src/emit.rs:379 | `VITASLOP_WATCH_STORE_LOG` - LOG each store to the watched address (the storing |
| `VITASLOP_WATCH_STORE_MODE` | vitaslop-transpiler/src/emit.rs:369 | Store-watchpoint mode, from `VITASLOP_WATCH_STORE_MODE` (default `any`): |
| `VITASLOP_WATCH_STORE_NZ` | vitaslop-transpiler/src/emit.rs:392 | `VITASLOP_WATCH_STORE_LOG` - LOG each store to the watched address (the storing |
| `VITASLOP_WATCH_STORE_SKIP` | vitaslop-transpiler/src/emit.rs:193 | WASM global index of the store-watchpoint match counter (appended after `TP_GLOBAL`). |
