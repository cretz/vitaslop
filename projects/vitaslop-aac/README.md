# vitaslop-aac

> Keep this README terse: sectioned and bulleted, not large prose. Explain
> concepts and the why, not exact type names. Update it as the code changes so it
> never goes stale.

Decodes AAC through the decoder the platform already has - Media Foundation on
Windows, WebCodecs in the browser - behind one API. A movie's sound track is AAC
and the Vita decodes it in `SceAudiodec`, so this is what that host call reaches.

## Why not decode it ourselves

- Same reasoning as [vitaslop-h264](../vitaslop-h264/): every host already ships
  an AAC decoder, and a from-scratch one is work that buys nothing.
- What is left to the caller is the part this crate does: reading the
  `AudioSpecificConfig` out of the container, and turning a submit/poll decoder
  into the synchronous answer the guest expects.

## Submit and poll, not decode

- In a browser the decoded audio arrives on a CALLBACK that only runs once the
  worker returns to the event loop, so no call here can promise "the PCM for this
  access unit" on return. The seam is therefore SUBMIT then POLL, the same shape
  the video path uses.
- A caller that needs PCM synchronously - `sceAudiodecDecode` does - gets it by
  running AHEAD of the guest, which it can, because the stream comes out of a
  container it is already demultiplexing.
- Output is interleaved s16, which is what the guest's own audio path wants.

## Shape

- Backends under `backend/` (`windows`, `web`); one is selected per target.
- No dependency on any other crate here, and nothing Vita-shaped: the guest's
  frame headers and memory layout belong to the caller.
- A decode that fails REPORTS. Silence that a caller mistakes for a working
  decoder is the failure mode this path is prone to, so there is no lossy
  fallback.

## State

- Carries the movie sound track end to end. A starved poll (nothing decoded yet
  for that unit) is served as silence and counted, so a gap is visible in the
  panel rather than being heard and guessed at.
