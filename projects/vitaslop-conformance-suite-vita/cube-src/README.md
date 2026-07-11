# cube-src: the GXM spinning-cube corpus artifact

`cube.c` is a minimal GXM spinning cube, authored clean-room from the MIT
`vita-headers` API and the publicly documented GXM init sequence. It is NOT
derived from the (unlicensed) vitasdk gxm sample.

It exists to drive the emulator bring-up **work-backwards**: a real,
compiler-produced ARM binary with real NID imports is a far better forcing
function for the loader, the transpiler CFG buildout, and the host-module ABI
than any hand-written stub. See `working-area/agent-notes.md` for the staged
milestones.

## Artifacts (committed fixtures)
- `cube.c`      - the source.
- `build.sh`    - reproducible recipe (needs `$VITASDK`).
- `cube.elf`    - linked ARM ELF (`-Wl,-q`, relocations kept).
- `cube.velf`   - Vita executable: the ELF with NID import tables encoded by
                  `vita-elf-create`. This is the loader's input. No crypto - a
                  velf is the decrypted form a SELF wraps; we own the loader so
                  we skip the SELF/fself layer.

The binary is 100% our `-nostdlib` code plus Sony NID import metadata (generated
from the MIT `vita-headers` db), so it is license-clean to commit. Rebuild with
`VITASDK=$HOME/vitasdk bash build.sh`; the output is deterministic.

## What it exercises
- 38 function imports across 4 libraries: `SceGxm`, `SceDisplayUser`, `SceCtrl`,
  `SceSysmem`.
- The full GXM frame: initialize, ring buffers + context, render target, double-
  buffered color surfaces + sync objects, depth/stencil, shader patcher, vertex/
  fragment programs, vertex/index buffers, and a per-frame begin/draw/end/swap
  loop with a rotating MVP matrix (VFP float math).

## Shader caveat (milestone 5)
The two `SceGxmProgram` blobs in `cube.c` are PLACEHOLDERS (four-byte "GXP\0"
magic, empty USSE payload). Compiling real GXM shaders needs Sony's `libshacccg`
(a blob we refuse) or hand-authored precompiled `.gxp`. Until we actually
rasterize, the shader bytes are opaque data the CPU path never interprets; our
host stubs accept them and record their use. Replace with real precompiled
`.gxp` when the wgpu/software rasterizer lands.
