// The page's half of browser audio: an AudioContext, an AudioWorklet, and the shared
// ring the emulator worker fills. See `audio-worklet.js` for the consumer and
// `src/audio.rs` for the producer; the three share one control-block layout and any
// change to it is a change in all three.
//
// The page owns this rather than the worker because Web Audio is a MAIN-THREAD API -
// there is no AudioContext in a Worker. Nothing but the SharedArrayBuffer crosses back,
// so no PCM is ever copied between threads and the main thread does no per-grain work at
// all: the audio render thread reads the ring itself.
//
// A browser will not start an AudioContext without a user gesture, and a headless
// harness has no gestures. `startAudio` therefore resumes the context and REPORTS the
// resulting state rather than assuming it - a suspended context accepts every call and
// plays nothing, which is indistinguishable from an emulator producing silence unless
// somebody says which it is.

const CTL_WRITE = 0;
const CTL_READ = 1;
const CTL_UNDERRUN = 2;
const CTL_OVERRUN = 3;
const CTL_CAPACITY = 4;
const CTL_CHANNELS = 5;
const CTL_SAMPLE_RATE = 6;
const CTL_HEADER_BYTES = 32;

/// Ring depth. Half a second is far more than a real backend needs, and that is
/// deliberate: the emulator does not yet run at real time, so the producer arrives in
/// bursts separated by long gaps and a shallow ring would underrun on every gap even
/// when the average rate is fine. Deep enough to ride out a stall, still only 192 KB.
const RING_SECONDS = 0.5;

/// The Vita's `sceAudioOut` MAIN/BGM ports run at 48 kHz. Asking the AudioContext for
/// the same rate is what lets the sink hand samples straight through: a mismatch means
/// resampling, which the sink will do and REPORT rather than silently detune the game.
const VITA_SAMPLE_RATE = 48000;

/**
 * Create the audio output and its shared ring.
 *
 * Returns `{ ring, context, node, stats() }`. Post `ring` to the emulator worker; it is
 * a SharedArrayBuffer, so it is shared, not copied, and needs no transfer list.
 */
export async function startAudio() {
  const channels = 2;
  const context = new AudioContext({ sampleRate: VITA_SAMPLE_RATE, latencyHint: "playback" });
  await context.audioWorklet.addModule("./audio-worklet.js");

  const capacity = Math.floor(context.sampleRate * RING_SECONDS);
  const ring = new SharedArrayBuffer(CTL_HEADER_BYTES + capacity * channels * 4);
  const ctl = new Int32Array(ring, 0, CTL_HEADER_BYTES / 4);
  Atomics.store(ctl, CTL_CAPACITY, capacity);
  Atomics.store(ctl, CTL_CHANNELS, channels);
  Atomics.store(ctl, CTL_SAMPLE_RATE, context.sampleRate);

  const node = new AudioWorkletNode(context, "vitaslop-audio", {
    numberOfInputs: 0,
    outputChannelCount: [channels],
    processorOptions: { ring },
  });
  node.connect(context.destination);

  // Autoplay policy: without a user gesture the context stays "suspended" and plays
  // nothing. Say which happened - see the note at the top.
  await context.resume().catch(() => {});
  console.log(
    `[audio] context ${context.state} at ${context.sampleRate} Hz, ` +
      `ring ${capacity} frames (${RING_SECONDS}s)` +
      (context.state === "suspended"
        ? " - SUSPENDED: the browser is waiting for a user gesture, so nothing will be audible"
        : "")
  );

  // A gesture anywhere on the page is enough to release a suspended context, so arm one
  // rather than leaving the page silently mute until someone thinks to click.
  if (context.state !== "running") {
    const wake = () => {
      context.resume().then(() => console.log(`[audio] context ${context.state} after a gesture`));
      window.removeEventListener("pointerdown", wake);
      window.removeEventListener("keydown", wake);
    };
    window.addEventListener("pointerdown", wake);
    window.addEventListener("keydown", wake);
  }

  /// What the ring has actually carried: frames in, frames out, and the two failure
  /// counts. `underrun` is the emulator falling behind the device; `overrun` is the
  /// emulator running ahead of it. They mean opposite things and are never merged.
  const stats = () => ({
    state: context.state,
    written: Atomics.load(ctl, CTL_WRITE),
    read: Atomics.load(ctl, CTL_READ),
    underrun: Atomics.load(ctl, CTL_UNDERRUN),
    overrun: Atomics.load(ctl, CTL_OVERRUN),
    sampleRate: context.sampleRate,
  });

  return { ring, context, node, stats };
}
