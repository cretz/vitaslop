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

// Control-block layout, in Int32 slots. Mirrored in `web/audio.js` and `src/audio.rs` -
// all three must agree, so any change is a change in three places by design.
const CTL_WRITE = 0; // frames the producer has published, monotonic
const CTL_READ = 1; // frames this processor has consumed, monotonic
const CTL_UNDERRUN = 2; // frames of silence emitted for want of data
const CTL_OVERRUN = 3; // frames the producer dropped because the ring was full
const CTL_CAPACITY = 4; // ring size in frames
const CTL_CHANNELS = 5;
const CTL_HEADER_BYTES = 32;

class VitaslopAudio extends AudioWorkletProcessor {
  constructor(options) {
    super();
    const sab = options.processorOptions.ring;
    this.ctl = new Int32Array(sab, 0, CTL_HEADER_BYTES / 4);
    this.data = new Float32Array(sab, CTL_HEADER_BYTES);
    this.capacity = Atomics.load(this.ctl, CTL_CAPACITY);
    this.channels = Atomics.load(this.ctl, CTL_CHANNELS);
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
    const available = write - read;
    const take = Math.min(frames, Math.max(0, available));

    for (let i = 0; i < take; i++) {
      const base = ((read + i) % this.capacity) * this.channels;
      for (let c = 0; c < out.length; c++) {
        // A mono ring feeds both output channels; a stereo one feeds them in order.
        out[c][i] = this.data[base + (c % this.channels)];
      }
    }
    // The shortfall is silence, and it is COUNTED. See the note at the top.
    for (let i = take; i < frames; i++) {
      for (let c = 0; c < out.length; c++) out[c][i] = 0;
    }
    if (take < frames) {
      Atomics.add(this.ctl, CTL_UNDERRUN, frames - take);
    }
    if (take > 0) {
      read += take;
      Atomics.store(this.ctl, CTL_READ, read);
    }
    return !this.stopped;
  }
}

registerProcessor("vitaslop-audio", VitaslopAudio);
