//! Tests are driven by the RIFF/oracle harness in `tests/`. This module holds only
//! small unit checks that need crate internals.

use crate::Atrac9Decoder;

#[test]
fn parses_mono_48k_config() {
    // board_break_rnd_01.at9 config word: 48 kHz mono, 64-byte frames, 4/superframe.
    let dec = Atrac9Decoder::new([0xFE, 0x70, 0x07, 0xF0]).expect("valid config");
    let info = dec.info();
    assert_eq!(info.channels, 1);
    assert_eq!(info.sample_rate, 48000);
    assert_eq!(info.frame_samples, 256);
    assert_eq!(info.frames_in_superframe, 4);
    assert_eq!(info.superframe_size, 256);
}

#[test]
fn rejects_bad_header() {
    assert!(Atrac9Decoder::new([0x00, 0x00, 0x00, 0x00]).is_err());
}

#[test]
fn bit_reader_reads_big_endian_fields() {
    use crate::bit_reader::BitReader;
    // 0b1010_0111 0b0011_0000 -> read 3,5,8 bits.
    let bytes = [0xA7u8, 0x30];
    let mut br = BitReader::new(&bytes);
    assert_eq!(br.read_int(3), 0b101);
    assert_eq!(br.read_int(5), 0b00111);
    assert_eq!(br.read_int(8), 0x30);
    assert_eq!(br.position, 16);
}

#[test]
fn bit_reader_sign_extends() {
    use crate::bit_reader::sign_extend32;
    assert_eq!(sign_extend32(0b111, 3), -1);
    assert_eq!(sign_extend32(0b011, 3), 3);
    assert_eq!(sign_extend32(0b1000, 4), -8);
}

#[test]
fn gradient_curves_span_the_base_curve() {
    // The full-length curve is exactly the base curve; every curve is a
    // downsample of it, so it starts at the base curve's first value and stays
    // within its value range.
    let curves = crate::decoder::generate_gradient_curves();
    assert_eq!(curves[47][0], 1);
    assert_eq!(curves[47][47], 30);
    for len in 1..=48usize {
        assert_eq!(curves[len - 1][0], 1);
        for &v in &curves[len - 1][..len] {
            assert!((1..=30).contains(&v));
        }
    }
}

#[test]
fn imdct_window_is_generated() {
    // The synthesis window is finite and positive across a frame; a cheap guard
    // that the trig/window generation ran and did not divide by zero.
    let w = crate::tables::Windows::generate();
    for &v in &w.imdct[2][..256] {
        assert!(v.is_finite() && v > 0.0);
    }
}
