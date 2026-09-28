# vitaslop-frontend

> Keep this README terse: sectioned and bulleted, not large prose. Explain
> concepts and the why, not exact type names. Update it as the code changes so it
> never goes stale.

What the browser and the desktop front ends SHARE. No UI, no engine - just the
records both of them read and write, so the two cannot drift apart.

## Why it exists

- The same settings, the same button names and the same library metadata were
  being defined twice, once per front end. Two definitions of one record is how a
  setting comes to mean something slightly different depending on where it was
  set.
- Everything here is `serde` JSON, because JSON is what crosses to the page and
  what the desktop writes beside its library.

## What is in it

- **settings** - the one settings record and its defaults, plus the MERGE that
  turns a global record and a per-title patch into the settings a run actually
  uses. Also the knob map a run is launched with, so a per-title knob override is
  the same mechanism on both hosts.
- **input** - the Vita's buttons by name and bit, and the default keyboard and
  gamepad maps, expressed in the W3C vocabularies both platforms can speak.
- **meta** - the record kept per imported title (what the library lists and
  searches on).

## Shape

- Pure data and pure functions: no I/O, no window, no renderer. Builds for both
  native and wasm.
- Holds no game data and names no title: a title appears here only as the id its
  own dump carries.
