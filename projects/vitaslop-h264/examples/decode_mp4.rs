//! Decode an H.264 track out of an MP4 and report what came back.
//!
//! ```text
//! cargo run -p vitaslop-h264 --example decode_mp4 -- movie.mp4 [--frames N] [--ppm out.ppm]
//! ```
//!
//! This is the crate's own smoke test against REAL video, which the conformance suite
//! deliberately cannot be: that suite decodes a synthetic all-I_PCM stream so it can assert
//! exact pixels, which means it never exercises CABAC, inter prediction, B-frames, multiple
//! reference frames, or reordering. A real file exercises all of them - it just cannot say
//! what the right pixels are, so what this checks is that decoding SUCCEEDS, that every
//! sample yields a frame, and that timestamps come back in order.

use std::time::Instant;

use vitaslop_h264::{Decoder, DecoderConfig, InputFormat, Packet, mp4};

fn main() {
    let mut args = std::env::args().skip(1);
    let path = match args.next() {
        Some(p) => p,
        None => {
            eprintln!("usage: decode_mp4 <file.mp4> [--frames N] [--ppm out.ppm]");
            std::process::exit(2);
        }
    };
    let mut limit = usize::MAX;
    let mut ppm: Option<String> = None;
    let mut low_latency = false;
    while let Some(flag) = args.next() {
        match flag.as_str() {
            "--frames" => limit = args.next().and_then(|v| v.parse().ok()).unwrap_or(usize::MAX),
            "--ppm" => ppm = args.next(),
            // The question a caller standing in for a fixed-function decoder has to answer:
            // how many access units go in before the FIRST picture comes out.
            "--low-latency" => low_latency = true,
            other => {
                eprintln!("unknown argument {other}");
                std::process::exit(2);
            }
        }
    }

    let file = std::fs::read(&path).unwrap_or_else(|e| {
        eprintln!("{path}: {e}");
        std::process::exit(1);
    });
    let track = mp4::read_h264_track(&file).unwrap_or_else(|e| {
        eprintln!("{path}: {e}");
        std::process::exit(1);
    });
    println!(
        "track: {}x{}, {} samples, timescale {}, avcC profile {} level {}",
        track.width,
        track.height,
        track.samples.len(),
        track.timescale,
        track.avcc.profile_idc,
        track.avcc.level_idc
    );

    let config = DecoderConfig {
        input: InputFormat::LengthPrefixed { length_size: track.avcc.length_size },
        extradata: Some(track.avcc_bytes.clone()),
        packets_are_access_units: Some(true),
        low_latency,
        ..DecoderConfig::default()
    };
    let mut decoder = Decoder::new(config).unwrap_or_else(|e| {
        eprintln!("no decoder: {e}");
        std::process::exit(1);
    });

    let mut decoded = 0usize;
    let mut first_frame_after: Option<usize> = None;
    let mut last_pts = i64::MIN;
    let mut out_of_order = 0usize;
    let mut first_frame: Option<vitaslop_h264::Frame> = None;
    let started = Instant::now();

    let wanted = limit.min(track.samples.len());
    for index in 0..wanted {
        let sample = track.sample_data(&file, index).expect("sample inside the file");
        let pts = track.to_micros(track.samples[index].pts);
        if let Err(e) = decoder.send(Packet::with_pts(sample, pts)) {
            eprintln!("sample {index}: {e}");
            std::process::exit(1);
        }
        loop {
            match decoder.receive() {
                Ok(Some(frame)) => {
                    if frame.pts < last_pts {
                        out_of_order += 1;
                    }
                    last_pts = frame.pts;
                    decoded += 1;
                    first_frame_after.get_or_insert(index + 1);
                    // Keep the LAST frame decoded, so `--frames N` picks the moment to
                    // look at: frame 0 of a movie is usually a black fade-in.
                    if ppm.is_some() {
                        if let Some(old) = first_frame.replace(frame) {
                            decoder.recycle(old);
                        }
                    } else {
                        decoder.recycle(frame);
                    }
                }
                Ok(None) => break,
                Err(e) => {
                    eprintln!("frame {decoded}: {e}");
                    std::process::exit(1);
                }
            }
        }
    }
    decoder.finish().expect("flush");
    while let Some(frame) = decoder.receive().expect("drain") {
        if frame.pts < last_pts {
            out_of_order += 1;
        }
        last_pts = frame.pts;
        decoded += 1;
        decoder.recycle(frame);
    }

    let elapsed = started.elapsed();
    let info = decoder.stream_info().expect("a decoded stream has parameter sets");
    println!(
        "stream: {}x{} profile {} level {} reorder depth {} sar {:?} timing {:?}",
        info.width,
        info.height,
        info.profile_idc,
        info.level_idc,
        info.max_reorder_frames,
        info.sample_aspect_ratio,
        info.timing
    );
    println!(
        "first picture came out after {:?} access units (low_latency={low_latency})",
        first_frame_after
    );
    println!(
        "decoded {decoded}/{wanted} frames in {:.2}s ({:.0} fps) on {} [{:?}], {out_of_order} out of order",
        elapsed.as_secs_f64(),
        decoded as f64 / elapsed.as_secs_f64(),
        decoder.backend_name(),
        decoder.acceleration(),
    );

    if let (Some(path), Some(frame)) = (ppm, first_frame) {
        let mut rgba = Vec::new();
        frame.copy_to_rgba(&mut rgba);
        let mut out = format!("P6\n{} {}\n255\n", frame.width, frame.height).into_bytes();
        out.extend(rgba.chunks_exact(4).flat_map(|p| [p[0], p[1], p[2]]));
        std::fs::write(&path, out).expect("write the ppm");
        println!("wrote {path}");
    }
}
