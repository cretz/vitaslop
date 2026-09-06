# vitaslop-conformance-suite-vita

The Vita conformance corpus: real Vita executables (velf) that drive the loader,
transpiler, and host-module ABI over the actual NID import mechanism, not the
arm-level `svc` trap.

Unlike the arm corpus there is no simple offline oracle (no qemu equivalent for a
NID program), so each case runs as a hand-written end-to-end test in the harness
(`vitaslop-conformance-harness/tests/vita_*.rs`), which loads the velf, transpiles
and runs it, and asserts the captured output or GXM stream. The `cases/*.toml`
files are the human-readable spec (description + golden) alongside each test.

## Artifacts (`*-src/`)

Each is clean-room from the MIT vita-headers API and built `-nostdlib` with a tiny
self-contained runtime, so the committed binary is license-clean and its import
surface is only Sony NID stubs. Reproduce with the sibling `build.sh` (needs a
Vita toolchain via `$VITASDK`).

- **`hello-src/`** (`vita_hello`): sceClibPrintf-heavy hello world. Exercises the
  variadic host call - the C formatter and the AAPCS variadic argument walk (core
  registers then stack, doubles promoted and 8-byte aligned, never the VFP file).
- **`clib-src/`** (`vita_clib`): the SceLibKernel clib memory/string calls
  (memcpy/memset/memcmp/strnlen/strncpy/strcmp/strncmp/snprintf) run on real guest
  memory, results printed and checked.
- **`thread-src/`** (`vita_thread`): create/start/wait threading. Proves two hard
  mechanisms - code-pointer discovery (the address-taken thread entry the
  transpiler finds via movw/movt, not the direct-call closure) and synchronous
  guest re-entry (the host runs the worker's own code and returns its value).
- **`kernel-src/`** (`vita_kernel`): sync + timing primitives - mutex, semaphore,
  event flag (set/wait/read), and the wide system clock. Forced the transpiler to
  lift adc/sbc (the 64-bit compare), now qemu-certified in the arm suite.
- **`compute-src/`** (`vita_compute`): a CPU-core probe with volatile inputs -
  64-bit multiply (UMULL/SMULL), count-leading-zeros (CLZ), wide add/sub, shifts.
  Surfaced UMULL/SMULL/CLZ as transpiler gaps; now lifted and qemu-certified.
- **`compute2-src/`** (`vita_compute2`): a second probe - byte reverse (REV/REV16),
  sign/zero extend (SXTB/UXTB/SXTH), multiply-accumulate (MLA), bitfield extract
  (UBFX). Surfaced and lifted all of them (+ MLS/UXTH/SBFX), qemu-certified.
- **`cube-src/`** (`cube_*`): the minimal GXM spinning cube - the graphics north
  star. See its README; bring-up was work-backwards over this artifact.
- **`pvf-src/`** (`vita_pvf`): the ScePvf vector-font engine end to end - create a
  library, open a real font file through the host filesystem (the first case to
  read a data file the way a title does), configure size/resolution, and query
  metrics + rasterize a glyph. Ships the public-domain Ahem font (predictable
  geometry, so the advance/coverage assertions are exact). See its README.

## Status

`hello`, `clib`, `thread`, `kernel`, `compute`, `compute2`, `cube`, `io`, and
`pvf` are all RUNNABLE and green. Remaining north-star rungs (SELF/ELF loading of
a signed title, richer module surface, Chocolate Doom) are staged in the project
notes.
