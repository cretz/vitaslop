// The audio CONSUMER: an AudioWorkletProcessor that reads interleaved stereo f32 out of
// the shared ring the emulator worker fills, and hands it to the device.
//
// Why a worklet and a shared ring, rather than posting each grain to the page and
// scheduling an AudioBufferSourceNode:
//
//   * The emulator runs in a Web Worker and Web Audio lives on the main thread, so the
//     PCM has to cross a thread boundary either way. A SharedArrayBuffer crosses it
//     without copying and without waking the main thread at all - the audio render
//     thread reads it directly, which is the only thread whose deadline actually matters.
//   * `sceAudioOutOutput` submits a grain every few milliseconds and must not block (see
//     `vitaslop_runtime::audio::AudioSink`). A postMessage per grain puts the guest's
//     audio thread behind the main thread's event queue, which is where the jank comes
//     from on a page that is also rendering.
//
// The ring is single-producer / single-consumer with monotonic frame counters, so neither
// side ever needs a lock: the producer only advances `write`, this only advances `read`,
// and each is published with an atomic store after the data it describes is in place.
//
// UNDERRUN IS COUNTED, NEVER HIDDEN. An emulator that is not running at real time
// produces audio slower than the device consumes it, and the honest symptom of that is a
// gap. Filling it with silence and saying nothing would make "the emulator is too slow"
// look like "the audio backend is broken", which are opposite fixes.
//
// >>> TWO THINGS THIS DOES BEYOND COPYING SAMPLES, BOTH MEASURED FROM REAL COMPLAINTS.
//
// LATENCY (`MAX_LEAD_SECONDS`). The ring is half a second deep so a bursty producer does
// not starve it. But depth that is FULL is latency: a menu sound whose samples sit behind
// 0.4s of backlog is heard 0.4s after the button that caused it, which is exactly what a
// player notices first. So the backlog is CAPPED here: anything beyond the cap is skipped
// past, and the skip is counted. Dropping the OLDEST audio is what keeps the game
// responsive - dropping the newest (which is what a full ring does to the producer) keeps
// stale audio and throws away the sound that just happened.
//
// CLICKS (`RAMP_FRAMES`). A gap emitted as an abrupt zero is a step discontinuity, and a
// step is a click - which is what "static" turns out to be when a run underruns thousands
// of times. The gap is still a gap and still counted; it is just entered and left over a
// millisecond instead of instantaneously. This hides nothing: the underrun counter is
// untouched, and a run that crackles still says so in its numbers.

// Control-block layout, in Int32 slots. Mirrored in `web/audio.js` and `src/audio.rs` -
// all three must agree, so any change is a change in three places by design.
const CTL_WRITE = 0; // frames the producer has published, monotonic
const CTL_READ = 1; // frames this processor has consumed, monotonic
const CTL_UNDERRUN = 2; // frames of silence emitted for want of data
const CTL_OVERRUN = 3; // frames the producer dropped because the ring was full
const CTL_CAPACITY = 4; // ring size in frames
const CTL_CHANNELS = 5;
const CTL_SAMPLE_RATE = 6;
const CTL_PEAK = 7; // loudest sample this run, |s| * 32767
const CTL_LATENCY_SKIP = 8; // frames skipped past to keep the backlog under the cap
const CTL_FILL = 9; // frames of backlog at the last block - the live latency
const CTL_HEADER_BYTES = 64;

/// How far the ring is allowed to run ahead of the device before old audio is skipped.
/// This is the emulator's audio latency, straight up. Small enough that a UI sound
/// lands with its button press; large enough to ride out the jitter of a frame that
/// takes longer than its budget.
const MAX_LEAD_SECONDS = 0.1;

/// Frames spent ramping in or out of a gap. ~1.3ms at 48 kHz: long enough to remove the
/// step, short enough that no one perceives it as a fade.
const RAMP_FRAMES = 64;

/// Once starved, how much audio must be buffered before playback resumes.
///
/// >>> WITHOUT THIS, A RING THAT IS CHRONICALLY NEARLY-EMPTY SOUNDS WORSE THAN THE CLICKS
/// >>> IT REPLACED. An emulator running below real time delivers slightly less than one
/// block per block, so playback would start, starve, ramp down, restart, ramp up - every
/// 128 frames. That is amplitude modulation at a few hundred hertz: a warble, which is
/// far more objectionable than the occasional gap. Waiting for a real buffer turns a
/// continuous stutter into few, longer gaps, which is both less unpleasant and a more
/// honest signal of what is actually wrong.
const RESUME_SECONDS = 0.03;

class VitaslopAudio extends AudioWorkletProcessor {
  constructor(options) {
    super();
    const sab = options.processorOptions.ring;
    this.ctl = new Int32Array(sab, 0, CTL_HEADER_BYTES / 4);
    this.data = new Float32Array(sab, CTL_HEADER_BYTES);
    this.capacity = Atomics.load(this.ctl, CTL_CAPACITY);
    this.channels = Atomics.load(this.ctl, CTL_CHANNELS);
    const rate = Atomics.load(this.ctl, CTL_SAMPLE_RATE) || 48000;
    this.maxLead = Math.max(1, Math.floor(rate * MAX_LEAD_SECONDS));
    this.resumeAt = Math.max(1, Math.floor(rate * RESUME_SECONDS));
    // Envelope carried ACROSS blocks: a gap that starts at a block boundary must not
    // restart the ramp, or the click comes back at the seam.
    this.env = 0;
    // True while waiting for `resumeAt` frames to accumulate after a starvation.
    this.starved = true;
    this.stopped = false;
    this.port.onmessage = (e) => {
      if (e.data === "stop") this.stopped = true;
    };
  }

  process(_inputs, outputs) {
    const out = outputs[0];
    const frames = out[0].length;
    const write = Atomics.load(this.ctl, CTL_WRITE);
    let read = Atomics.load(this.ctl, CTL_READ);

    // Cap the backlog. See the LATENCY note at the top: everything past the cap is
    // stale, and playing it would put every later sound behind by that much.
    let available = write - read;
    if (available > this.maxLead) {
      const skip = available - this.maxLead;
      read += skip;
      available -= skip;
      Atomics.add(this.ctl, CTL_LATENCY_SKIP, skip);
    }
    Atomics.store(this.ctl, CTL_FILL, available);

    // Hysteresis: having starved, hold silent until a real buffer exists again rather
    // than restarting on the first frame that arrives. See RESUME_SECONDS.
    if (this.starved) {
      if (available >= this.resumeAt) this.starved = false;
    } else if (available === 0) {
      this.starved = true;
    }
    const take = this.starved ? 0 : Math.min(frames, Math.max(0, available));
    const step = 1 / RAMP_FRAMES;

    for (let i = 0; i < frames; i++) {
      // Ramp toward "playing" while there is data for this frame, toward silence once
      // there is not. Both directions are gradual, so neither edge is a step.
      const target = i < take ? 1 : 0;
      this.env = target > this.env ? Math.min(1, this.env + step) : Math.max(0, this.env - step);
      if (i < take) {
        const base = ((read + i) % this.capacity) * this.channels;
        for (let c = 0; c < out.length; c++) {
          // A mono ring feeds both output channels; a stereo one feeds them in order.
          out[c][i] = this.data[base + (c % this.channels)] * this.env;
        }
      } else {
        // The shortfall is silence, and it is COUNTED below. See the note at the top.
        for (let c = 0; c < out.length; c++) out[c][i] = 0;
      }
    }

    if (take < frames) {
      Atomics.add(this.ctl, CTL_UNDERRUN, frames - take);
    }
    if (take > 0) {
      read += take;
    }
    Atomics.store(this.ctl, CTL_READ, read);
    return !this.stopped;
  }
}

registerProcessor("vitaslop-audio", VitaslopAudio);
