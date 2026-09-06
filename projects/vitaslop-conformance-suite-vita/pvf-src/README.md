# pvf-src (`vita_pvf`)

Clean-room conformance coverage for the ScePvf vector-font engine: create a
library, open a real font file through the host virtual filesystem, configure
size/resolution, and query metrics + rasterize a glyph, asserting the results.

This is the first conformance case to read a real data file the way a retail title
does: the harness preloads `ahem.ttf` into the guest filesystem with
`env.state.add_file("app0:/ahem.ttf", ...)`, and the guest opens it with
`scePvfOpenUserFile("app0:/ahem.ttf", ...)` (which reads it back through the same
`SceIoFilemgr` file table the `io` case exercises).

## Font: Ahem

`ahem.ttf` is the Ahem test font (W3C / web-platform-tests). It is used here
because its geometry is deliberately predictable, which makes the assertions exact
rather than fuzzy:

- every glyph advance is exactly 1 em,
- most glyphs render as a solid filled box spanning 0.8 em above the baseline to
  0.2 em below (so at a 16 px em the advance is 16 px and the box is a known solid
  rectangle of ~full coverage).

**License: public domain / CC0.** The font's own name table declares: "The Ahem
font belongs to the public domain. In jurisdictions that do not recognize public
domain ownership of these files, the following Creative Commons Zero declaration
applies: http://labs.creativecommons.org/licenses/zero-waive/1.0/us/legalcode".
It is therefore freely redistributable and safe to commit alongside the source.
Upstream: https://github.com/web-platform-tests/wpt (`fonts/Ahem.ttf`).

## Reproduce

`ahem.ttf` is committed. Rebuild `pvf.elf`/`pvf.velf` from `pvf.c` with a Vita
toolchain (`VITASDK=$HOME/vitasdk bash build.sh`).
