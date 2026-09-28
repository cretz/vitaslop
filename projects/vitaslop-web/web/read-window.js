// Reading a picked file for the import, at the size the device wants rather than the
// size the ingest asks for.
//
// A PICKED FILE IS NOT A LOCAL FILE. On Android it comes through the platform's file
// provider and every `FileReaderSync` slice is a round trip: measured on a phone, about
// 140 ms of fixed cost plus 5 ms per MB, WHATEVER the size. The ingest reads a megabyte
// at a time (`CHUNK` in `stream.rs`), so on that phone the reads alone capped the import
// at 6.9 MB/s, against 71.9 MB/s at 16 MB per read. This desktop's spread over the same
// range is 1.7x, which is why none of it showed up until a device was measured.
// (`web/debug/import-speed.html` is that measurement.)
//
// So reads are cached. Two things about the ingest's access pattern decide the shape,
// and both were measured, not assumed (`web/debug/import-bench-worker.js` prints the
// offset of every refill):
//
//   IT IS NOT ONE STREAM. Refills on a real title land at 0.0, 1151.4, 0.0, 0.1, 58.4,
//   0.0, 1151.4, 262.9 MB and so on: the pkg head, the pkg tail and the file tables are
//   re-read constantly, interleaved with the file data. A single window is evicted by
//   the next metadata read and re-reads the data it just dropped. Hence WAYS: a handful
//   of independent windows, least-recently-used evicted, so the metadata localities and
//   the data stream can coexist.
//
//   THE DATA IS NOT ONE RUN EITHER. A title's files are not stored in the order the
//   ingest walks them, so a fixed 16 MB read-ahead spends 16 MB to serve a 200 KB file.
//   Hence the ADAPTIVE span: a window opens small, and only doubles when the next read
//   continues exactly where the last one ended, which is what a big file does. A jump
//   resets it. Big files end up reading at the maximum, small scattered ones never pay
//   for read-ahead they will not use.
// MEASURED ON THE PHONE, not chosen. At 2 MB opening / 16 MB cap that device's real
// import ran at 8.6 MB/s taking 28 provider reads for a 128 MB slice; at 4 / 32 it ran
// at 9.4 MB/s taking 21. Every read avoided is 140 ms, which is why bigger wins even
// though it pulls more bytes (158 MB -> 196 MB for the same 128 MB written).
//
// The cost is memory: four ways at the cap is 128 MB of JS buffers in the worst case,
// though in practice only the data window grows - the metadata windows keep missing and
// so keep reopening at MIN_SPAN, for around 44 MB. Do not raise these without measuring
// on a device: past here the read amplification grows faster than the round trips shrink.
export const MIN_SPAN = 4 * 1024 * 1024;
export const MAX_SPAN = 32 * 1024 * 1024;
export const WAYS = 4;

/// A `ByteSource` for the Rust ingest over the picked files.
///
/// `entries` is `[{ path, file }]`, keyed by the path the ingest will name. `onRead` is
/// called with the bytes pulled from the PROVIDER (not the bytes served), which is what
/// makes it a measure of real progress rather than of cache hits. `onFill` reports the
/// offset of each refill, for the bench. `span: 0` disables caching entirely, which is
/// the old behaviour and exists so the bench can measure against it.
export function makeFileSource(
  entries,
  { onRead = () => {}, onFill = () => {}, span = MIN_SPAN, maxSpan = MAX_SPAN, ways = WAYS } = {}
) {
  const byPath = new Map(entries.map((e) => [e.path, e.file]));
  const reader = new FileReaderSync();
  const cache = []; // { path, start, bytes, used }
  let clock = 0;
  // The sequential detector: where the last refill ended, and how big it was.
  let last = { path: null, end: -1, span };

  const pull = (file, path, at, len) => {
    const bytes = new Uint8Array(reader.readAsArrayBuffer(file.slice(at, Math.min(file.size, at + len))));
    onRead(bytes.length, path);
    return bytes;
  };

  const fill = (file, path, at) => {
    onFill(at, path);
    // Continuing exactly where the last refill ended is a file being streamed: double
    // the span, up to the cap. Anything else starts over small.
    const next = path === last.path && at === last.end ? Math.min(last.span * 2, maxSpan) : span;
    const bytes = pull(file, path, at, next);
    last = { path, end: at + bytes.length, span: next };
    const entry = { path, start: at, bytes, used: ++clock };
    if (cache.length < ways) cache.push(entry);
    else {
      let lru = 0;
      for (let i = 1; i < cache.length; i++) if (cache[i].used < cache[lru].used) lru = i;
      cache[lru] = entry;
    }
    return entry;
  };

  const hit = (path, at) => {
    for (const e of cache) {
      if (e.path === path && at >= e.start && at < e.start + e.bytes.length) {
        e.used = ++clock;
        return e;
      }
    }
    return null;
  };

  return {
    list: () => [...byPath.keys()],
    size: (path) => {
      const f = byPath.get(path);
      return f ? f.size : undefined;
    },
    // `ByteSource::read_at` may return short ONLY at the end of the file, so this fills
    // the caller's buffer across as many windows as it takes.
    readAt: (path, off, buf) => {
      const file = byPath.get(path);
      if (!file || off >= file.size) return 0;
      const want = Math.min(buf.length, file.size - off);
      if (!span) {
        const bytes = pull(file, path, off, want);
        buf.set(bytes);
        return bytes.length;
      }
      let done = 0;
      while (done < want) {
        const at = off + done;
        // A read at least as big as the cap is already one big provider call: serving it
        // through a window would only add a copy.
        if (want - done >= maxSpan) {
          const bytes = pull(file, path, at, want - done);
          if (!bytes.length) break;
          buf.set(bytes, done);
          done += bytes.length;
          last = { path, end: at + bytes.length, span: maxSpan };
          continue;
        }
        const e = hit(path, at) || fill(file, path, at);
        if (!e.bytes.length) break;
        const from = at - e.start;
        const n = Math.min(want - done, e.bytes.length - from);
        buf.set(e.bytes.subarray(from, from + n), done);
        done += n;
      }
      return done;
    },
  };
}
