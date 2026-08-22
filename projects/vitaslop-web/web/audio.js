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
/// Loudest sample seen this run, as |sample| * 32767, written by the producer. The ring
/// holds only the last half second, so this is the only whole-run evidence that anything
/// was ever audible - see the note in `src/audio.rs`.
const CTL_PEAK = 7;
/// Frames the consumer skipped past to keep the backlog under its latency cap - see
/// `audio-worklet.js`. Non-zero means the emulator produced audio faster than the device
/// consumed it, and the OLDEST audio was dropped to keep sounds in time with the game.
const CTL_LATENCY_SKIP = 8;
/// Frames of backlog at the last block: the live output latency, in frames.
const CTL_FILL = 9;
const CTL_HEADER_BYTES = 64;

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
    /// Loudest sample of the whole run, 0..1. Zero means the run was SILENT no matter
    /// what `written` says.
    peak: Atomics.load(ctl, CTL_PEAK) / 32767,
    /// Frames dropped to keep output latency bounded, and the live backlog in frames.
    /// `fill` IS the latency a player hears between a sound being produced and heard.
    latencySkip: Atomics.load(ctl, CTL_LATENCY_SKIP),
    fill: Atomics.load(ctl, CTL_FILL),
  });

  /// A copy of the PCM currently sitting in the ring, as interleaved f32.
  ///
  /// >>> THIS IS THE ONLY THING THAT CAN TELL WORKING AUDIO FROM SILENT AUDIO.
  /// `stats()` counts FRAMES, and a frame of zeroes counts exactly like a frame of
  /// music: a defect that fed the ring perfectly-paced digital silence ran for a long
  /// time with `written` and `read` both climbing and nothing audible. Anything
  /// asserting that audio WORKS has to look at sample values, so this hands them over.
  ///
  /// The ring is half a second deep and circular, so this is the last ~0.5s and no
  /// more; it is a probe, not a recording. Read in ring order (oldest first) so the
  /// samples come out as a contiguous waveform - a raw copy of the backing store would
  /// be spliced at the write cursor, which destroys exactly the sample-to-sample
  /// continuity a caller is likely measuring.
  const samples = () => {
    const write = Atomics.load(ctl, CTL_WRITE);
    const have = Math.min(write, capacity);
    const out = new Float32Array(have * channels);
    const data = new Float32Array(ring, CTL_HEADER_BYTES, capacity * channels);
    const start = (write - have) % capacity;
    for (let i = 0; i < have; i++) {
      const src = ((start + i) % capacity) * channels;
      for (let c = 0; c < channels; c++) out[i * channels + c] = data[src + c];
    }
    return out;
  };

  return { ring, context, node, stats, samples };
}
