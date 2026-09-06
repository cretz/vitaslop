# vitaslop-h264

> Keep this README terse: sectioned and bulleted, not large prose. Explain
> concepts and the why, not exact type names. Update it as the code changes so it
> never goes stale.

H.264 video decoding through the decoder the platform already has: Media
Foundation on Windows, Video Toolbox on macOS, VA-API on Linux, WebCodecs in the
browser. One API over the four, plus the bitstream work all four leave to the
caller.

## Not a Vita crate

- It carries the workspace prefix because it lives here; it knows nothing about
  the Vita and **depends on no other crate in this workspace**. Keep it that way:
  it is the one crate here that could be lifted into a standalone public repo
  unchanged.
- Anything Vita-shaped - `sceAvcdec`'s frame headers, guest memory, the layout a
  guest expects a decoded frame in - belongs in the caller.

## Why not decode it ourselves

- A software H.264 decoder is a year of work and then loses anyway: every one of
  these platforms has a fixed-function video block, and reaching it is the
  difference between 1080p at a few percent of a core and not at all.
- So this crate's job is the part the system decoders do NOT do, which each of
  them leaves to the caller in a slightly different way.

## What it actually implements

- **Bitstream** (`bitstream/`): NAL framing and emulation-prevention, SPS
  (including VUI and scaling lists), PPS, slice headers, picture order counts,
  the avcC record, and the split of a stream into access units. No residual
  decoding ever happens here - only what is needed to DRIVE a decoder.
- **Access unit boundaries**: the full 7.4.1.2.4 condition list, because most
  streams carry no access unit delimiters and "frame_num changed" is not enough.
- **Presentation order**: frames come out in presentation order on every
  platform. Video Toolbox emits in decode order, so the common layer reorders it
  with the stream's own `max_num_reorder_frames`.
- **The DPB, on Linux only**: VA-API's H.264 entry point is stateless, so
  reference list construction (8.2.4) and reference marking (8.2.5) are ours.
  The other three platforms do that internally.
- **MP4 reading** (`mp4`, default feature): one H.264 track out of a
  non-fragmented file - avcC, sample table, timestamps. Enough that "decode this
  .mp4" needs no demuxer dependency.

## Rules it keeps

- **Unsupported is an error, never an approximation.** 4:2:2 and 4:4:4, 10-bit,
  MVC/SVC, FMO, and interlaced coding on the VA-API path all report rather than
  decode into something plausible.
- **A missing decoder is a normal outcome.** No libva, an N edition of Windows,
  a browser without WebCodecs: `Error::NoDecoder`, which a caller falls back on.
  libva is `dlopen`ed for exactly this reason - linking it would stop the binary
  starting on a machine that lacks it.
- **The platform's strides are kept, not normalised.** A frame describes its
  planes; a caller uploading to a GPU uses them directly, and one that wants a
  packed buffer asks for the copy explicitly.

## Testing

- The conformance test decodes a stream built entirely of `I_PCM` macroblocks
  (the `synth` module, on by default so a plain `cargo test` runs it). `I_PCM` carries raw samples with no transform, no
  prediction, and a QP of zero - which also disables deblocking on its edges - so
  a conforming decoder MUST return exactly the bytes that went in. That is what
  makes a byte-equality assertion legitimate against four different decoders, with
  no reference implementation and no checked-in sample file.
- Covered: byte-exact pixels on the HARDWARE path and on the software one,
  presentation order, caller timestamps, reset, cropping (coded 320x240 vs visible
  312x232 - the shape every 1080p stream has), HD, the avcC / length-prefixed
  path, and the refusal of a picture too large for a hardware bitstream buffer.
- NOT covered by any run: the `IMF2DBuffer2` branch (this machine's software MFT
  returns a plain contiguous buffer), and everything outside Windows.
- Run: `cargo test -p vitaslop-h264`.

## State per platform

- **Windows / Media Foundation**: verified, on hardware. Enumeration alone gets a
  synchronous MFT that decodes on the CPU; what reaches the fixed-function decoder
  is giving it a **D3D11 device manager**, after which output arrives as GPU
  textures and is read back through a staging texture. `acceleration()` reports
  `Hardware` only once a decoded sample has actually come back as a texture - it
  is proof, not intent.
- **The DXVA picture-size limit**, measured here rather than assumed: a coded
  picture must fit the driver's bitstream buffer. At 588 KB it was byte-exact; at
  633 KB the first 336 rows were correct and the picture then repeated from its
  top, with no error from Media Foundation at any layer. Real encoded pictures run
  10-100 KB, so this only bites pathological input - but since it is silent, the
  backend refuses any picture over `max_hardware_picture_bytes` (512 KiB default)
  instead of returning it, and the error says to use `hardware: Some(false)`.
- **macOS / Video Toolbox**, **Linux / VA-API**, **browser / WebCodecs**: written
  against the platform APIs and compile-checked for their targets
  (`cargo check --target ...`), but not yet run on those platforms. The same
  conformance test is what they should be checked with; it needs no assets.

## Asynchrony

- The three native backends produce output inside the call. WebCodecs does not:
  frames arrive on a callback and their pixels become readable when a second
  promise settles.
- So the browser path has `receive_async`, which waits on those callbacks - not
  on a timer, because a worker's `setTimeout(0)` is clamped to 4 ms once nested.
  On native it is the synchronous path with an `async` signature, so one caller
  works everywhere.
