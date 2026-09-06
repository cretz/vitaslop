# vitaslop-hostcall

> Keep this README terse: sectioned and bulleted, not large prose. Explain
> concepts and the why, not exact type names. Update it as the code changes so it
> never goes stale.

One proc macro, `#[hostcall]`: write a Vita host call as an ordinary typed Rust
function and let it generate the calling-convention plumbing.

## Why

- A guest call arrives as a register file, not as arguments. Every handler
  otherwise repeats the same shuffle: integer args from `r0..r3` then the stack,
  float args from the VFP registers (the Vita is hardfloat), return to `r0` or
  `s0`/`d0`.
- Getting that wrong is **silent** - an argument read from the wrong register is
  a plausible number, so the handler runs and the failure surfaces elsewhere.

## What it covers

- Value params and returns: `u32`, `i32`, `bool`, `Ptr` (integer class), `f32`,
  `f64` (float class). `()` writes nothing. Params consumed left to right.
- The host state and the raw call context are threaded, not marshalled, so a
  handler can reach guest memory directly for out-params, structs and strings.

## Boundary

- **Generates the shuffle, never behavior.** Semantics stay hand-written in
  `vitaslop-runtime` because that is the reverse-engineered half, and the half
  that can be wrong.
- No stub or default expansion. An unimplemented host call hard-fails by name so
  it gets implemented, rather than returning a plausible zero.
