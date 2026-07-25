# vitaslop-gxp-shader

A clean-room recompiler that translates a Vita GXP shader (`SceGxmProgram`) into
WGSL, so the guest title's own shading runs on WebGPU instead of a fixed-function
approximation of it.

A Vita fragment or vertex shader is a `SceGxmProgram` container wrapping PowerVR
SGX543 "USSE" bytecode. Reproducing a title's look faithfully (multi-layer paint,
decals, lighting, fog - the things a fixed-function stand-in cannot express) means
running that bytecode, not guessing at it. This crate does that end to end.

## Pipeline

1. **`container`** - parses the `SceGxmProgram` header, the parameter table (the
   resource-binding plan: attributes, uniforms, samplers), and the location of the
   USSE instruction stream.
2. **`usse::decode`** - decodes the fixed 64-bit USSE instructions into the `ir`:
   operation, banked operands, per-channel swizzles, source modifiers, write mask,
   predicate.
3. **`wgsl`** - emits WGSL for shaders it can translate faithfully. USSE registers
   are 32-bit scalars, so the emitter *scalarises*: one statement per written
   destination channel, reading `bank[base + lane]`. It exposes `tex_units()` (the
   sampler bindings the body references) and `wrap_module()` (wraps an emitted body
   into a standalone, compilable module - used to validate the emit and as the
   skeleton the renderer's pipeline builder binds real textures/uniforms into).
4. **`interp`** - a reference interpreter over the same operations, evaluating them
   numerically. It validates that the *meaning* the emitter claims is correct, and is
   the foundation for behavioral validation against a captured framebuffer.

## Integrity contract

The recompiler emits WGSL only for shaders composed entirely of operations whose
encoding and semantics are established facts. **It never guesses and never emits an
approximation.** Anything it cannot translate exactly - an operation not yet wired,
or an instruction carrying an operand feature whose layout is not established - is a
HARD FAILURE that names the exact instruction and opcode to implement next (an opcode
grind, like the NID dispatcher's hard-fail on an unimplemented NID). You implement the
named opcode and re-run. A wrong translation can never paint a pixel.

Callers (the renderer) treat a `RecompileError` as the signal to fall back to the
fixed-function path, so adding each opcode strictly improves fidelity and never
regresses.

## Sourcing

Every fact used here - the SGX543 USSE instruction encoding (bit layouts, operand
banks, swizzle and write-mask tables, operation semantics), the container layout, and
the parameter table - comes from the public hardware instruction-set encoding and from
the vitasdk / psdevwiki definitions. These are permissive, fact-only sources. No
copyleft or proprietary code is read, linked, or derived from. The crate ships only
the decoder and emitter; it contains no game data.

## Status

Emit covers the arithmetic core plus the transcendental, move, pack (float<->float),
integer bitwise/shift, and texture-sample groups, and treats the phase-declaration and
no-op control words as no-ops - about 98% of the instructions in the captured shader
corpus, with most fragment shaders recompiling whole. Predicate-driven control
(compare-to-predicate, predicated writes, conditional move) and branch reconstruction
are the remaining features; each is wired the same way - from the ISA facts, with a
hard-fail until it is.

## Tests

```text
cargo test -p vitaslop-gxp-shader
```

runs the unit tests plus `tests/wgsl_valid.rs`, which compiles every emittable op
through naga (the same WGSL front-end wgpu uses) to prove the output is real,
validated shader code rather than plausible strings.

`tests/oracle.rs` is an ignored, opt-in harness that validates the parser and decoder
against a directory of real captured `SceGxmProgram` blobs and prints coverage
statistics. Those blobs are game-derived and are never committed; the test skips
cleanly when the `VITASLOP_GXP_DUMPS` directory is unset, so the suite is green with no
fixture and CI never sees game data.
