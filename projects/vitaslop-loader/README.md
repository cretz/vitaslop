# vitaslop-loader

> Keep this README terse: sectioned and bulleted, not large prose. Explain
> concepts and the why, not exact type names. Update it as the code changes so it
> never goes stale.

Parses a Vita executable into a Module the transpiler can consume: base address,
entry point, segments, and resolved NID imports. Pure bounds-checked byte
reading, no dependencies, wasm-clean.

## Input formats: velf and SELF/fSELF

- `load` accepts either a bare velf or a SELF/fSELF container (`eboot.bin`, the
  form a shipped title actually is). It sniffs the `"SCE\0"` magic and unwraps
  the container to the inner velf first, then parses that.
- A velf is an ELF with the Sony relocatable-executable type plus NID import
  tables. No crypto or relocation is involved.
- Struct layouts (velf and SELF) are taken from the MIT vita-toolchain source
  (`sce-elf-defs.h`, `self.h`, `vita-make-fself.c`), not reverse engineered.

### SELF/fSELF unwrap (`self.rs`)

- A SELF is a small SCE header + a copy of the program headers + a per-segment
  `segment_info` table + control info, then the segment payloads. We rebuild the
  inner ELF from the segment table (handles both the verbatim uncompressed
  layout and, in principle, scattered segments).
- **Unencrypted fSELF is loadable, compressed or not** - both `vita-make-fself`
  homebrew forms. Compressed segments are inflated by the loader's own
  dependency-free zlib/DEFLATE (`inflate.rs`), so no zlib crate and still
  wasm-clean. Encrypted segments (retail titles) return `EncryptedSelf`; a
  segment that fails to inflate returns `CompressedSelf`. Retail decryption is
  out of scope (no keys, license posture).
- `vita-make-fself` patches `module_nid` (sha256 of the velf) into the copy, so
  an unwrapped fSELF carries the real NID where the bare velf had a 0
  placeholder - the one field that differs from the input velf.

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
  segments match. The same cube wrapped as an fSELF (`cube.eboot.bin`) unwraps to
  the identical module (modulo `module_nid`) and runs the whole north-star path
  end to end (`conformance-harness/tests/cube_eboot.rs`).
- Relocations are not yet applied. Fine while the main module pins its fixed load
  address; needed for a non-fixed base.
- The built-in inflate (`inflate.rs`) is validated by round-trip tests against
  real zlib output (stored/fixed/dynamic Huffman) and end to end by
  `cube.eboot_c.bin`.
- Encrypted retail SELF is intentionally out of scope (no keys, license posture).
