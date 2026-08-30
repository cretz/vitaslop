//! Cross-platform conformance: decode a stream whose output is known exactly, and compare
//! every byte.
//!
//! The stream comes from [`vitaslop_h264::synth`], which builds it out of `I_PCM`
//! macroblocks - so the correct decoded picture is knowable from the bitstream itself, on
//! any conforming decoder, without a reference implementation and without a checked-in
//! sample file. That is what makes an exact comparison legitimate here: this is not
//! "the same as last time", it is "what the spec says".
//!
//! On a machine with no video decoder at all the test reports that and returns, because
//! "this CI container has no GPU" is not a decoder bug. Anything else - a decoder that
//! exists and produces the wrong pixels, the wrong count, or the wrong order - fails.

use vitaslop_h264::{Acceleration, Decoder, DecoderConfig, Error, Frame, Packet, synth};

/// Compare a decoded frame against the expected picture, reporting WHERE it went wrong
/// rather than dumping two megapixels into the test log.
fn assert_same(actual: &[u8], expected: &[u8], width: u32, height: u32, what: &str) {
    assert_eq!(actual.len(), expected.len(), "{what}: wrong size");
    if actual == expected {
        return;
    }
    let luma = (width * height) as usize;
    let chroma = luma / 4;
    let differing = actual.iter().zip(expected).filter(|(a, b)| a != b).count();
    let first = actual.iter().zip(expected).position(|(a, b)| a != b).unwrap();
    let plane = if first < luma {
        format!("Y row {} col {}", first / width as usize, first % width as usize)
    } else if first < luma + chroma {
        let at = first - luma;
        format!("Cb row {} col {}", at / (width as usize / 2), at % (width as usize / 2))
    } else {
        let at = first - luma - chroma;
        format!("Cr row {} col {}", at / (width as usize / 2), at % (width as usize / 2))
    };
    panic!(
        "{what}: {differing} of {} samples differ; first at offset {first} ({plane}),          got {} expected {}",
        expected.len(),
        actual[first],
        expected[first]
    );
}

/// Decode a whole stream, returning the frames in the order they came out.
fn decode_all(stream: &[u8]) -> Result<Option<(Vec<Frame>, &'static str)>, Error> {
    decode_all_with(stream, DecoderConfig::default())
}

/// [`decode_all`] with a chosen configuration.
fn decode_all_with(
    stream: &[u8],
    config: DecoderConfig,
) -> Result<Option<(Vec<Frame>, &'static str)>, Error> {
    let mut decoder = match Decoder::new(config) {
        Ok(d) => d,
        Err(e) if e.is_missing_decoder() => {
            eprintln!("no platform H.264 decoder on this machine, skipping: {e}");
            return Ok(None);
        }
        Err(e) => return Err(e),
    };
    let name = decoder.backend_name();

    let mut frames = Vec::new();
    decoder.send(Packet::new(stream))?;
    while let Some(frame) = decoder.receive()? {
        frames.push(frame);
    }
    decoder.finish()?;
    while let Some(frame) = decoder.receive()? {
        frames.push(frame);
    }
    Ok(Some((frames, name)))
}

/// The SHARED conformance body - the same one the browser runs.
///
/// The checks below it are native-only by nature (a Windows bitstream-buffer limit, an
/// HD-sized picture, the hardware path); this is the part that must hold on every backend,
/// and having it here as well means a native `cargo test` and a browser run cannot drift
/// apart into two different suites.
#[test]
fn the_shared_suite_passes() {
    let report = pollster_lite(vitaslop_h264::conformance::run()).expect("the suite must run");
    eprintln!("{}", report.text());
    assert!(report.ok(), "{}", report.failures().join("
"));
}

/// Drive a future to completion on this thread. The shared suite is `async` so the browser
/// can await WebCodecs; natively every await is already ready, so this never parks.
fn pollster_lite<F: std::future::Future>(mut future: F) -> F::Output {
    use std::pin::Pin;
    use std::task::{Context, Poll, RawWaker, RawWakerVTable, Waker};
    fn noop(_: *const ()) {}
    fn clone(p: *const ()) -> RawWaker {
        RawWaker::new(p, &VTABLE)
    }
    static VTABLE: RawWakerVTable = RawWakerVTable::new(clone, noop, noop, noop);
    // SAFETY: the vtable's operations are all no-ops over a null data pointer, which is
    // the standard shape for a waker that is never used to wake anything.
    let waker = unsafe { Waker::from_raw(RawWaker::new(std::ptr::null(), &VTABLE)) };
    let mut cx = Context::from_waker(&waker);
    // SAFETY: `future` is owned here and never moved again.
    let mut future = unsafe { Pin::new_unchecked(&mut future) };
    loop {
        match future.as_mut().poll(&mut cx) {
            Poll::Ready(v) => return v,
            Poll::Pending => panic!(
                "the shared conformance suite parked on a native backend, where every                  await is supposed to be ready already"
            ),
        }
    }
}

#[test]
fn decodes_a_synthetic_stream_byte_exactly() {
    // 128x96, five pictures: big enough that no decoder refuses the size, small enough that
    // the whole stream is a few hundred kilobytes of raw samples.
    let stream = synth::synthesize(8, 6, 5);
    let Some((frames, backend)) = decode_all(&stream.annex_b).expect("decoding must not fail")
    else {
        return;
    };

    assert_eq!(
        frames.len(),
        stream.frames.len(),
        "{backend} produced {} frames for a {}-picture stream",
        frames.len(),
        stream.frames.len()
    );

    let mut actual = Vec::new();
    for (index, frame) in frames.iter().enumerate() {
        assert_eq!((frame.width, frame.height), (stream.width, stream.height));
        frame.copy_to_i420(&mut actual);
        let expected = &stream.frames[index];
        assert_eq!(actual.len(), expected.len(), "frame {index} is the wrong size");
        if actual != *expected {
            let differing = actual.iter().zip(expected).filter(|(a, b)| a != b).count();
            let first = actual.iter().zip(expected).position(|(a, b)| a != b).unwrap();
            panic!(
                "{backend} decoded frame {index} wrongly: {differing} of {} samples differ, \
                 first at offset {first} (got {}, expected {})",
                expected.len(),
                actual[first],
                expected[first]
            );
        }
    }

    // Presentation order: the timestamps a caller sees must increase.
    let order: Vec<i64> = frames.iter().map(|f| f.pts).collect();
    let mut sorted = order.clone();
    sorted.sort_unstable();
    assert_eq!(order, sorted, "{backend} emitted frames out of presentation order: {order:?}");
}

#[test]
fn caller_timestamps_survive_the_round_trip() {
    let stream = synth::synthesize(4, 4, 3);
    let mut decoder = match Decoder::new(DecoderConfig::default()) {
        Ok(d) => d,
        Err(e) if e.is_missing_decoder() => return,
        Err(e) => panic!("decoder creation failed: {e}"),
    };

    // Feed one access unit per packet, each with a timestamp of the caller's choosing.
    let mut splitter = vitaslop_h264::bitstream::AuSplitter::new();
    let mut units = Vec::new();
    splitter.push_annex_b(&stream.annex_b, &mut units).unwrap();
    splitter.finish(&mut units).unwrap();
    assert_eq!(units.len(), 3);

    let stamps = [1_000_i64, 2_500, 9_999];
    let mut got = Vec::new();
    for (unit, &pts) in units.iter().zip(stamps.iter()) {
        decoder.send(Packet::with_pts(&unit.data, pts)).unwrap();
        while let Some(frame) = decoder.receive().unwrap() {
            got.push(frame.pts);
        }
    }
    decoder.finish().unwrap();
    while let Some(frame) = decoder.receive().unwrap() {
        got.push(frame.pts);
    }
    assert_eq!(got, stamps.to_vec(), "the caller's own timestamps must come back unchanged");
}

#[test]
fn a_stream_survives_a_reset() {
    let stream = synth::synthesize(4, 4, 3);
    let mut decoder = match Decoder::new(DecoderConfig::default()) {
        Ok(d) => d,
        Err(e) if e.is_missing_decoder() => return,
        Err(e) => panic!("decoder creation failed: {e}"),
    };

    decoder.send(Packet::new(&stream.annex_b)).unwrap();
    while decoder.receive().unwrap().is_some() {}
    decoder.reset().unwrap();

    // Decoding the same stream again after a reset must give the same pictures back.
    decoder.send(Packet::new(&stream.annex_b)).unwrap();
    let mut frames = Vec::new();
    while let Some(frame) = decoder.receive().unwrap() {
        frames.push(frame);
    }
    decoder.finish().unwrap();
    while let Some(frame) = decoder.receive().unwrap() {
        frames.push(frame);
    }
    assert_eq!(frames.len(), 3);

    let mut actual = Vec::new();
    for (index, frame) in frames.iter().enumerate() {
        frame.copy_to_i420(&mut actual);
        assert_eq!(actual, stream.frames[index], "frame {index} differs after a reset");
    }
}

#[test]
fn stream_info_reports_what_the_sps_said() {
    let stream = synth::synthesize(5, 3, 1);
    // One packet, one picture: telling the decoder so is what lets a single-picture stream
    // be submitted without waiting for a following packet to prove the picture ended.
    let config = DecoderConfig { packets_are_access_units: Some(true), ..DecoderConfig::default() };
    let mut decoder = match Decoder::new(config) {
        Ok(d) => d,
        Err(e) if e.is_missing_decoder() => return,
        Err(e) => panic!("decoder creation failed: {e}"),
    };
    decoder.send(Packet::new(&stream.annex_b)).unwrap();
    let info = decoder.stream_info().expect("the parameter sets have been seen by now");
    assert_eq!((info.width, info.height), (80, 48));
    assert_eq!(info.profile_idc, 66);
    assert_eq!(info.level_idc, 51);

    decoder.finish().unwrap();
    let frame = decoder.receive().unwrap().expect("the one picture decodes");
    let mut actual = Vec::new();
    frame.copy_to_i420(&mut actual);
    assert_eq!(actual, stream.frames[0]);
}

#[test]
fn cropping_is_applied_to_what_the_caller_sees() {
    // 320x240 coded, 312x232 visible: the decoder's own buffer is bigger than the picture,
    // which is the shape every 1080p stream has (1088 coded rows, 1080 visible).
    let stream = synth::synthesize_cropped(20, 15, 2, 8, 8);
    assert_eq!(stream.coded, (320, 240));
    assert_eq!((stream.width, stream.height), (312, 232));

    let Some((frames, backend)) = decode_all(&stream.annex_b).expect("decoding must not fail")
    else {
        return;
    };
    assert_eq!(frames.len(), 2);
    let mut actual = Vec::new();
    for (index, frame) in frames.iter().enumerate() {
        assert_eq!(
            (frame.width, frame.height),
            (stream.width, stream.height),
            "{backend} reported the coded size instead of the visible one"
        );
        frame.copy_to_i420(&mut actual);
        assert_eq!(actual, stream.frames[index], "{backend} cropped frame {index} wrongly");
    }
}

#[test]
fn decodes_at_hd_resolution() {
    // 1280x720, decoded in software. This picture is all-I_PCM and therefore 1.35 MB of
    // coded data - far larger than any encoder emits, and larger than a hardware decoder's
    // bitstream buffer, which is what the next test is about.
    let stream = synth::synthesize(80, 45, 2);
    let config = DecoderConfig { hardware: Some(false), ..DecoderConfig::default() };
    let Some((frames, backend)) =
        decode_all_with(&stream.annex_b, config).expect("decoding must not fail")
    else {
        return;
    };
    assert_eq!(frames.len(), 2);
    let mut actual = Vec::new();
    for (index, frame) in frames.iter().enumerate() {
        assert_eq!((frame.width, frame.height), (1280, 720));
        frame.copy_to_i420(&mut actual);
        assert_same(
            &actual,
            &stream.frames[index],
            stream.width,
            stream.height,
            &format!("{backend} HD frame {index}"),
        );
    }
}

#[test]
fn an_oversized_picture_is_refused_rather_than_half_decoded() {
    // A hardware decoder is handed pictures through a driver-sized bitstream buffer. One
    // that does not fit comes back PARTLY decoded with no error from the platform at all
    // (measured on this machine: exact through a 588 KB coded picture, and from 633 KB the
    // first 336 rows were right and the picture then repeated from its top). Silently
    // returning that frame is the one thing this crate must not do.
    let stream = synth::synthesize(80, 45, 1); // 1.35 MB coded picture
    let config = DecoderConfig { hardware: Some(true), ..DecoderConfig::default() };
    let mut decoder = match Decoder::new(config) {
        Ok(d) => d,
        Err(e) if e.is_missing_decoder() => return,
        Err(e) => panic!("decoder creation failed: {e}"),
    };
    let result = decoder.send(Packet::new(&stream.annex_b)).and_then(|()| decoder.finish());

    match decoder.acceleration() {
        // Only the hardware path has this limit, and only it must refuse.
        Acceleration::Software(_) => {}
        _ => {
            let error = result.expect_err("an oversized picture must not decode silently");
            assert!(
                matches!(error, Error::Unsupported(_)),
                "expected an Unsupported error, got {error}"
            );
            let text = error.to_string();
            assert!(text.contains("hardware: Some(false)"), "the error must say what to do: {text}");
        }
    }
}

#[test]
fn decodes_length_prefixed_samples_with_out_of_band_parameter_sets() {
    // The MP4 / WebCodecs shape: parameter sets in an avcC record, each sample a
    // length-prefixed access unit with no start codes anywhere.
    let stream = synth::synthesize(6, 4, 4);

    let mut splitter = vitaslop_h264::bitstream::AuSplitter::new();
    let mut units = Vec::new();
    splitter.push_annex_b(&stream.annex_b, &mut units).unwrap();
    splitter.finish(&mut units).unwrap();

    let record = vitaslop_h264::bitstream::avcc::AvcC::from_parameter_sets(
        splitter.sets.sps_nals(),
        splitter.sets.pps_nals(),
        4,
    )
    .unwrap();

    let mut samples = Vec::new();
    for unit in &units {
        let mut sample = Vec::new();
        vitaslop_h264::bitstream::avcc::annex_b_to_length_prefixed(&unit.data, 4, &mut sample);
        samples.push(sample);
    }

    let config = DecoderConfig {
        input: vitaslop_h264::InputFormat::LengthPrefixed { length_size: 4 },
        extradata: Some(record.to_bytes()),
        ..DecoderConfig::default()
    };
    let mut decoder = match Decoder::new(config) {
        Ok(d) => d,
        Err(e) if e.is_missing_decoder() => return,
        Err(e) => panic!("decoder creation failed: {e}"),
    };

    let mut frames = Vec::new();
    for (index, sample) in samples.iter().enumerate() {
        decoder.send(Packet::with_pts(sample, index as i64 * 33_367)).unwrap();
        while let Some(frame) = decoder.receive().unwrap() {
            frames.push(frame);
        }
    }
    decoder.finish().unwrap();
    while let Some(frame) = decoder.receive().unwrap() {
        frames.push(frame);
    }

    assert_eq!(frames.len(), 4);
    let mut actual = Vec::new();
    for (index, frame) in frames.iter().enumerate() {
        assert_eq!(frame.pts, index as i64 * 33_367);
        frame.copy_to_i420(&mut actual);
        assert_eq!(actual, stream.frames[index], "frame {index} differs on the avcC path");
    }
}

#[test]
fn ordinary_pictures_go_through_the_video_hardware() {
    // The whole point of the platform backends. A picture of a size a real encoder produces
    // must decode on the hardware path AND be byte-exact there - this is the test that
    // would have caught the mangled-picture behaviour above if it applied at normal sizes.
    let stream = synth::synthesize(40, 30, 3); // 640x480, 450 KB per coded picture
    let config = DecoderConfig { hardware: Some(true), ..DecoderConfig::default() };
    let Some((frames, backend)) =
        decode_all_with(&stream.annex_b, config).expect("decoding must not fail")
    else {
        return;
    };
    assert_eq!(frames.len(), 3);
    let mut actual = Vec::new();
    for (index, frame) in frames.iter().enumerate() {
        frame.copy_to_i420(&mut actual);
        assert_same(
            &actual,
            &stream.frames[index],
            stream.width,
            stream.height,
            &format!("{backend} hardware frame {index}"),
        );
    }
}
