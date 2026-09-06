//! Print what an MP4 container holds, track by track.
//!
//! `cargo run -p vitaslop-runtime --example dump_mp4 -- movie.mp4 [more.mp4 ...]`
//!
//! A title's movie playback is driven entirely by what the demuxer reports - codec, size,
//! timescale, sample count, and the setup data a decoder has to be configured with - so
//! when playback goes wrong the first question is always what is actually in the file. This
//! answers it without booting anything, and it uses the engine's OWN demuxer, so it also
//! says whether that demuxer can read the file at all.
fn main() {
    let paths: Vec<String> = std::env::args().skip(1).collect();
    if paths.is_empty() {
        eprintln!("usage: dump_mp4 <file.mp4> [...]");
        std::process::exit(2);
    }
    for path in &paths {
        let data = match std::fs::read(path) {
            Ok(d) => d,
            Err(e) => {
                println!("{path}: cannot read ({e})");
                continue;
            }
        };
        let mp4 = match vitaslop_runtime::mp4::Mp4::parse(&data) {
            Ok(m) => m,
            Err(e) => {
                println!("{path}: not readable as MP4 ({e:?})");
                continue;
            }
        };
        println!(
            "{path}: {} bytes, {} track(s), {:.2}s",
            data.len(),
            mp4.tracks.len(),
            mp4.duration_us() as f64 / 1e6
        );
        for t in &mp4.tracks {
            println!(
                "  track {} {:?} codec {} {}x{} timescale {} {} samples {:.2}s setup {} bytes",
                t.id,
                t.kind,
                String::from_utf8_lossy(&t.codec),
                t.width,
                t.height,
                t.timescale,
                t.samples.len(),
                t.duration_us() as f64 / 1e6,
                t.codec_config.len(),
            );
            if !t.codec_config.is_empty() {
                let head: Vec<String> =
                    t.codec_config.iter().take(24).map(|b| format!("{b:02x}")).collect();
                println!("    setup: {}", head.join(" "));
            }
            if let Some(first) = t.samples.first() {
                let biggest = t.samples.iter().map(|s| s.size).max().unwrap_or(0);
                println!(
                    "    first sample {} bytes at {:#x}, largest {biggest} bytes, {} sync",
                    first.size,
                    first.offset,
                    t.samples.iter().filter(|s| s.sync).count()
                );
            }
        }
    }
}
