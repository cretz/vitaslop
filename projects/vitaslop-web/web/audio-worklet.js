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
// >>> AND THE CAP IS READ OFF THE TROUGH OF THE BACKLOG, NOT OFF ITS INSTANTANEOUS VALUE.
//
// This producer is bursty BY CONSTRUCTION. The emulator runs a whole guest frame and then
// sleeps until the next one is due, so every grain that frame generated arrives in one go
// and is drained over the ~16 ms that follow; a frame that ran long delivers several
// frames' worth at once. Comparing that PEAK against the cap trims the burst the instant
// it lands, which is not latency being removed - it is the buffer that was about to be
// consumed. MEASURED on a phone's race: `23.43s skipped to keep it bounded` beside
// `UNDERRUN 43.0%` and a live backlog of 11 ms. The ring was being shaved to nothing on
// every burst and then starved by the next hitch, and each splice is itself a click.
//
// The latency a player actually experiences is the backlog the ring NEVER drops below -
// the trough, not the peak. So the skip is computed from a rolling minimum over
// `LEAD_WINDOW_SECONDS`, and only the amount by which that minimum exceeds the cap is
// skipped. A burst consumed again inside the window is left alone; a producer genuinely
// running ahead is still trimmed, because its trough rises too.
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
///
/// Compared against the TROUGH of the backlog (see the LATENCY note at the top), so this
/// is the SUSTAINED lead and not the height of a burst.
const MAX_LEAD_SECONDS = 0.12;

/// The window the rolling minimum backlog is taken over.
///
/// It has to be comfortably longer than the producer's burst period - one guest frame,
/// 16.7 ms, and several times that when a frame overruns - or the minimum is the
/// instantaneous value again and the trimming is back to shaving every burst. A second
/// covers the frame hitches a phone actually produces while still bringing the latency
/// down within a second of a title going quiet.
const LEAD_WINDOW_SECONDS = 1.0;

/// How many buckets that window is divided into. The trough is quantised to one bucket of
/// staleness, which at 16 buckets over a second is 62 ms - well inside the cap it feeds.
const LEAD_BUCKETS = 16;

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
const RESUME_SECONDS = 0.04;

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
    // The rolling trough, as buckets of per-block minima. Buckets rather than every
    // block's value so the minimum is a fixed 16-element scan on the audio render thread
    // with no monotonic-deque bookkeeping: each bucket holds the smallest fill seen while
    // it was current, and the window minimum is the smallest bucket.
    this.leadBuckets = new Array(LEAD_BUCKETS).fill(Infinity);
    this.leadBucket = 0;
    this.leadBucketBlocks = 0;
    // Blocks per bucket, derived on the first block from the size this processor is
    // actually called with (128 today, but the spec does not promise it).
    this.blocksPerBucket = 0;
    this.rate = rate;
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

    let available = write - read;

    // The SUSTAINED backlog: the smallest this ring has been over the last
    // `LEAD_WINDOW_SECONDS`. See the LATENCY note at the top for why the trough, and not
    // the peak, is the number that means "latency".
    if (this.blocksPerBucket === 0) {
      this.blocksPerBucket = Math.max(
        1,
        Math.round((this.rate * LEAD_WINDOW_SECONDS) / LEAD_BUCKETS / Math.max(1, frames)),
      );
    }
    this.leadBuckets[this.leadBucket] = Math.min(this.leadBuckets[this.leadBucket], available);
    if (++this.leadBucketBlocks >= this.blocksPerBucket) {
      this.leadBucketBlocks = 0;
      this.leadBucket = (this.leadBucket + 1) % LEAD_BUCKETS;
      this.leadBuckets[this.leadBucket] = Infinity;
    }
    let sustained = Infinity;
    for (let i = 0; i < LEAD_BUCKETS; i++) sustained = Math.min(sustained, this.leadBuckets[i]);

    // Cap the backlog. Everything past the cap is stale, and playing it would put every
    // later sound behind by that much - but only what the ring never drops below is past
    // the cap. A window that has not seen a full sweep yet trims nothing.
    if (Number.isFinite(sustained) && sustained > this.maxLead) {
      const skip = sustained - this.maxLead;
      read += skip;
      available -= skip;
      Atomics.add(this.ctl, CTL_LATENCY_SKIP, skip);
      // Those frames are gone from the window too, or the next block sees the same excess
      // and skips it a second time - which is how a trim turns back into a shave.
      for (let i = 0; i < LEAD_BUCKETS; i++) {
        if (Number.isFinite(this.leadBuckets[i])) this.leadBuckets[i] -= skip;
      }
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

    // >>> BEFORE THE FIRST GRAIN EVER WRITTEN THERE IS NOTHING TO UNDERRUN.
    // The context starts with the page and the guest boots for seconds before it opens an
    // audio port at all, so counting those blocks charged the boot to the audio path: one
    // measured 1,600-frame run reported 4.06 s of underrun of which the whole boot was part,
    // and "underrun is a PERFORMANCE number" then reads a performance problem that is not
    // there. Silence is still emitted and every block after the first write is still counted.
    if (take < frames && write > 0) {
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
