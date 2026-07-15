//! Bit-exactness test against the upstream LibAtrac9 C decoder.
//!
//! The C reference is compiled separately (see
//! `working-area/scripts/build-at9-oracle.sh`) into `name.oracle.s16` files next to
//! the decrypted `name.at9` inputs. Point `VITASLOP_AT9_DIR` at that directory and
//! run `cargo test -p vitaslop-atrac9 --test oracle -- --ignored --nocapture`. The
//! test skips when the env var is unset, so `cargo test --workspace` stays green
//! without the (gitignored, game-derived) fixtures.

use std::path::PathBuf;

use vitaslop_atrac9::Atrac9Decoder;

fn rd32(p: &[u8]) -> u32 {
    u32::from_le_bytes([p[0], p[1], p[2], p[3]])
}

/// Extract the 4-byte ATRAC9 config word and the AT9 payload from a RIFF WAVE.
fn demux(buf: &[u8]) -> ([u8; 4], &[u8]) {
    let mut fmt_off = None;
    let mut data = None;
    let mut p = 12; // skip "RIFF"____"WAVE"
    while p + 8 <= buf.len() {
        let id = &buf[p..p + 4];
        let csz = rd32(&buf[p + 4..]) as usize;
        let body = p + 8;
        if id == b"fmt " {
            fmt_off = Some(body);
        } else if id == b"data" {
            data = Some(&buf[body..body + csz]);
        }
        p = body + csz + (csz & 1);
    }
    let fmt = fmt_off.expect("fmt chunk");
    // WAVEFORMATEXTENSIBLE(18) + samples(2) + channelMask(4) + guid(16) + version(4)
    let cfg_off = fmt + 18 + 2 + 4 + 16 + 4;
    let mut config = [0u8; 4];
    config.copy_from_slice(&buf[cfg_off..cfg_off + 4]);
    (config, data.expect("data chunk"))
}

/// Decode a whole AT9 payload the way a superframe consumer does: for each
/// superframe, decode every packed frame, advancing by each frame's byte count.
fn decode_all(config: [u8; 4], data: &[u8]) -> Vec<i16> {
    let mut dec = Atrac9Decoder::new(config).expect("init decoder");
    let superframe = dec.superframe_bytes();
    let frames = dec.frames_per_superframe();
    let frame_shorts = dec.frame_samples() * dec.channels();
    let mut out = Vec::new();
    let mut pcm = vec![0i16; frame_shorts];

    let mut off = 0;
    while off + superframe <= data.len() {
        let mut inner = off;
        for _ in 0..frames {
            let used = dec.decode_frame(&data[inner..], &mut pcm).expect("decode frame");
            out.extend_from_slice(&pcm);
            inner += used;
        }
        off += superframe;
    }
    out
}

#[test]
#[ignore = "needs VITASLOP_AT9_DIR with name.at9 + name.oracle.s16 pairs"]
fn matches_reference_oracle() {
    let Some(dir) = std::env::var_os("VITASLOP_AT9_DIR") else {
        eprintln!("VITASLOP_AT9_DIR unset; skipping");
        return;
    };
    let dir = PathBuf::from(dir);

    let mut checked = 0;
    for entry in std::fs::read_dir(&dir).expect("read at9 dir") {
        let path = entry.unwrap().path();
        if path.extension().and_then(|e| e.to_str()) != Some("at9") {
            continue;
        }
        let stem = path.file_stem().unwrap().to_str().unwrap().to_string();
        let oracle_path = dir.join(format!("{stem}.oracle.s16"));
        if !oracle_path.exists() {
            eprintln!("no oracle for {stem}; skipping");
            continue;
        }

        let at9 = std::fs::read(&path).unwrap();
        let (config, data) = demux(&at9);
        let mine = decode_all(config, data);

        let oracle_bytes = std::fs::read(&oracle_path).unwrap();
        let oracle: Vec<i16> = oracle_bytes
            .chunks_exact(2)
            .map(|c| i16::from_le_bytes([c[0], c[1]]))
            .collect();

        assert_eq!(
            mine.len(),
            oracle.len(),
            "{stem}: sample count {} vs oracle {}",
            mine.len(),
            oracle.len()
        );

        let mut mismatches = 0;
        let mut first = None;
        for (i, (m, o)) in mine.iter().zip(oracle.iter()).enumerate() {
            if m != o {
                mismatches += 1;
                if first.is_none() {
                    first = Some((i, *m, *o));
                }
            }
        }
        assert_eq!(
            mismatches, 0,
            "{stem}: {mismatches}/{} samples differ, first {:?}",
            mine.len(),
            first
        );
        eprintln!("{stem}: {} samples bit-exact vs oracle", mine.len());
        checked += 1;
    }
    assert!(checked > 0, "no at9/oracle pairs found in {}", dir.display());
}
