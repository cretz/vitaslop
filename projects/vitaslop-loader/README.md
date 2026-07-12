# vitaslop-loader

> Keep this README terse: sectioned and bulleted, not large prose. Explain
> concepts and the why, not exact type names. Update it as the code changes so it
> never goes stale.

Parses a Vita executable into a Module the transpiler can consume: base address,
entry point, segments, and resolved NID imports. Pure bounds-checked byte
reading, no dependencies, wasm-clean.

## Input format: velf (ET_SCE_RELEXEC)

- A velf is an ELF with the Sony relocatable-executable type plus NID import
  tables. No crypto is involved (SELF/fself encryption is skipped).
- Struct layouts are taken from the MIT vita-toolchain source, not reverse
  engineered from binaries.

## What it extracts, and the pitfalls

- **Entry point is not e_entry.** e_entry locates a module-info structure
  (encoded as segment-index in the high bits plus an offset). The real entry is
  the module_start field inside that structure.
- **Segment-relative vs absolute.** The module-info top/end fields are offsets
  relative to their segment. The pointers inside the import tables are absolute
  virtual addresses. Mixing these up is the classic loader bug.
- **Import walk.** Step through the import region entry by entry (a long and a
  short form both exist), and for each library read its NID plus the parallel
  arrays of function NIDs and their stub addresses.

## Output: the Module

- Yields module name, module NID, base address, entry, segments, and a list of
  imports. Each import is a library NID + function NID + stub address.
- This feeds the transpiler two things: the entry point(s) to start discovery
  from, and the stub addresses so a call to a stub becomes a host import trap
  carrying that import's dense index.

## State and TODO

- Validated against the committed conformance cube: import count, entry, and
  segments match.
- Relocations are not yet applied. Fine while the main module pins its fixed load
  address; needed for a non-fixed base.
- Vita path (SELF/ELF plus real NID resolution) comes later; today the focus is
  the velf the conformance artifact ships.
