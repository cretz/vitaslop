# vitaslop-conformance-harness

> Keep this README terse: sectioned and bulleted, not large prose. Explain
> concepts and the why, not exact type names. Update it as the code changes so it
> never goes stale.

Dev-only test runner: the one crate that runs the conformance corpora against the
real transpile-plus-execute path. Supplies the engine and the corpus's host ABI,
and exercises the emitted WASM (not an interpreter).

## What it runs

- **Generic ARM/Thumb corpus** (from the arm suite crate): per-case seed state
  plus assembled bytes plus expected output. Cases cover flags, loops, IT blocks,
  push/pop, shifts, and the arithmetic/logic/memory core.
- **Real cube tests**: slices of the committed conformance cube (memcpy, memset,
  a gpu-alloc path with IT blocks plus host imports) and a whole-cube
  transpile-plus-instantiate check, plus a run that records the NID call sequence.

## Oracles (why we trust the expected values)

- Register/memory outputs for the arm corpus are generated from an `as` + `qemu`
  oracle, not hand-computed. See the arm suite crate for regen.
- Decode is certified separately against capstone in the yaxpeax-arm fork. The
  harness assumes a correct decoder and checks execution semantics.

## What is checked

- Final register file and any declared memory outputs.
- NZCV flags are checked too (goldens record them).
- Cube runs assert the demanded NID sequence, which is the blob-free signal that
  guest code reached the expected host boundary.

## Notes

- Run from the workspace root. Thumb entry stubs must be marked as Thumb or the
  oracle enters ARM mode and faults; the regen step handles this.
- The suite crates carry no engine dependency; only this harness does.
