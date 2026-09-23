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

## The three oracles, and what each can prove

They answer different questions, and the difference matters - the strongest of the
three cannot say what a program *means*.

- **Structural** (`tests/corpus.rs`, `tests/wgsl_valid.rs`): does a blob parse, link,
  and compile under naga and Tint? Every graphics defect this project has chased was a
  module that did all four perfectly and computed the wrong number.
- **Differential** (`tests/execcases.rs` + `vitaslop-web/e2e/gxpexec.mjs`): run the
  emitted module on the real GPU and diff it against the reference interpreter, over the
  whole corpus, with no title and no frame. Far stronger - it found five defects in the
  reference in one session - but both halves are our own reading of the ISA, so **where
  both are wrong in the same way, they agree**.
- **Conformance** (`tests/conformance.rs`): programs this project *authored*, with the
  answer stated in Rust before anything runs. Three parties must agree - the intent, the
  reference, and the GPU through Tint - so a divergence names which side is wrong rather
  than only that they differ. It also reaches features no shipped title happens to use,
  which a corpus by construction cannot. Its completeness is itself a test:
  `every_emittable_operation_kind_has_an_authored_case` fails unless every operation kind the
  emitter translates - each texture LOD builtin, and the F16 form of every kind that has one -
  is exercised by an authored case.
- **Facing wiring** (`tests/facing_wiring.rs`): the one thing both case rigs pin rather than
  check. An authored pair is linked by the shipped linker and drawn in both windings on a real
  GPU, and which winding reports the facing bit is asserted.

The conformance cases are assembled by `usse::asm` (a USSE assembler built over the
decoder's own field and swizzle tables, round-tripping every word through `decode`
before returning it) and wrapped by `gxpwrite` (a `SceGxmProgram` container writer). A
case therefore travels the same path a captured blob does: container parse, USSE decode,
link, WGSL emission, Tint, the GPU.

What conformance still cannot prove is fidelity to the real SGX543 - the program was
assembled through this project's own reading of the ISA, so a field we encode wrong we
also decode wrong. That gap belongs to the GXM conformance app
(`vitaslop-conformance-suite-vita/gxmconf-src`), which drives real libgxm, and
ultimately to a blob a real `SceShaccCg` produced.

```text
cargo test -p vitaslop-gxp-shader --test conformance
VITASLOP_GXP_CASES_OUT=<dir> cargo test -p vitaslop-gxp-shader --test conformance -- --ignored
node ../vitaslop-web/e2e/gxpexec.mjs <dir>
```
