# vitaslop-atrac9

> Keep this README terse: sectioned and bulleted, not large prose. Explain
> concepts and the why, not exact type names. Update it as the code changes so it
> never goes stale.

Decodes ATRAC9, the codec Vita titles ship music and voice in and the one NGS
decodes at playback. Build a decoder from the 4-byte config word in an AT9
stream's RIFF `fmt ` chunk, then decode a frame at a time into interleaved s16.

## Provenance

- **A faithful port of the MIT-licensed LibAtrac9** (Alex Barney), not clean-room
  and not claimed as such. Same MIT terms, which is why it can ship here.
- Clean-room effort is reserved for parts where the only references are copyleft.
  This one already has a permissive implementation.
- **Validated bit-exact against the C original.** A perceptual codec that is
  nearly right sounds fine and is still a different decoder, so "sounds correct"
  is not a test.

## Shape

- Pure compute: no deps, no I/O. Builds identically for native and wasm.
- Errors mirror the LibAtrac9 status codes; every one means a malformed or
  unsupported bitstream. No lossy fallback - a bad frame reports, it does not
  quietly become noise.
- Modules follow the decode order: config, bit reader, Huffman codebooks
  (generated tables kept separate so they are never hand-edited), frame decode,
  band extension, inverse MDCT.

## State

- The path NGS uses is complete and validated. Wired into the NGS mix.
- Making a title audible is runtime work (voice setup, host calls), not a gap
  here. Encoding is out of scope.
