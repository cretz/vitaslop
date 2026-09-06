//! A synthetic H.264 stream whose decoded pixels are known EXACTLY.
//!
//! Testing a decoder needs an oracle, and for a platform decoder the usual oracles are
//! unavailable: there is no reference decoder to compare against, and a checked-in sample
//! file only proves that the file decodes the way it decoded when the expected output was
//! recorded. This module gives a third answer - a stream whose correct output is knowable
//! from the bitstream by construction.
//!
//! The trick is `I_PCM`. An H.264 macroblock coded as `I_PCM` carries its 384 samples
//! (256 luma, 2 x 64 chroma) as raw bytes, with no transform, no prediction, and no
//! quantisation. The spec sets `QP'Y` to zero for such macroblocks, which makes the
//! deblocking filter's thresholds zero, so the filter cannot alter them either. A picture
//! made entirely of `I_PCM` macroblocks therefore decodes, on any conforming decoder, to
//! exactly the bytes that were written into it.
//!
//! That makes byte-equality a legitimate assertion across four different platform decoders,
//! which is what [`crate`]'s conformance test rests on. It costs nothing to ship: the
//! stream is generated at run time, so there is no test asset in the repository.

use crate::bitstream::nal;

/// A generated stream and the pictures it must decode to.
#[derive(Debug, Clone)]
pub struct SyntheticStream {
    /// Coded luma size, before cropping - what the decoder allocates internally.
    pub coded: (u32, u32),
    /// The stream in Annex B form.
    pub annex_b: Vec<u8>,
    /// Expected pictures, in presentation order, each tightly packed I420.
    pub frames: Vec<Vec<u8>>,
    /// Luma width.
    pub width: u32,
    /// Luma height.
    pub height: u32,
}

impl SyntheticStream {
    /// Bytes of one expected I420 picture.
    pub fn frame_size(&self) -> usize {
        let (w, h) = (self.width as usize, self.height as usize);
        w * h + 2 * (w / 2) * (h / 2)
    }
}

/// Generate a stream of `frames` pictures, `width_mbs` x `height_mbs` macroblocks each.
///
/// The first picture is an IDR; the rest are non-reference I pictures with increasing
/// picture order counts, which exercises the POC derivation and the presentation ordering
/// without needing inter prediction (which `I_PCM` cannot express).
///
/// Content is a deterministic pattern with no flat regions, so a decoder that returned a
/// mostly-correct picture - a wrong stride, a plane swapped, chroma taken from the previous
/// frame - fails the comparison rather than passing it by luck.
pub fn synthesize(width_mbs: u32, height_mbs: u32, frames: usize) -> SyntheticStream {
    synthesize_cropped(width_mbs, height_mbs, frames, 0, 0)
}

/// [`synthesize`], with `crop_right` and `crop_bottom` luma samples cropped away.
///
/// Cropping is how every 1080p stream in the world is coded - 1080 is not a multiple of 16,
/// so the picture is coded as 1088 rows and eight are cropped - and it is the one place a
/// backend can quietly hand back the wrong thing: the platform decoder's buffer is the
/// CODED size, and the visible size is only in the SPS. Both values must be even, because
/// 4:2:0 crops in chroma-sample units.
pub fn synthesize_cropped(
    width_mbs: u32,
    height_mbs: u32,
    frames: usize,
    crop_right: u32,
    crop_bottom: u32,
) -> SyntheticStream {
    assert!(width_mbs > 0 && height_mbs > 0, "a picture needs at least one macroblock");
    assert!(frames > 0, "a stream needs at least one picture");
    assert!(crop_right.is_multiple_of(2) && crop_bottom.is_multiple_of(2), "4:2:0 crops in chroma units");
    assert!(crop_right < width_mbs * 16 && crop_bottom < height_mbs * 16, "crop leaves nothing");
    let coded_width = width_mbs * 16;
    let coded_height = height_mbs * 16;
    let width = coded_width - crop_right;
    let height = coded_height - crop_bottom;

    let mut stream = Vec::new();
    let mut writer = BitWriter::new();
    write_sps(&mut writer, width_mbs, height_mbs, crop_right, crop_bottom);
    emit_nal(3, nal::kind::SPS, writer.finish_rbsp(), &mut stream);

    let mut writer = BitWriter::new();
    write_pps(&mut writer);
    emit_nal(3, nal::kind::PPS, writer.finish_rbsp(), &mut stream);

    // Macroblocks per slice. An all-I_PCM picture is enormous by coded-video standards -
    // 1.4 MB for one 720p frame - and a single slice that size is not something any encoder
    // emits, nor something hardware is built for: a DXVA decoder's bitstream buffer is a
    // megabyte, and a slice past it decodes partially and silently (measured: correct
    // through luma row 464 of 720, garbage after). Slicing the picture the way a real
    // encoder does keeps every slice comfortably inside that, and it exercises the
    // multi-slice access unit path as a side effect.
    let mbs_per_slice = (MAX_SLICE_BYTES / 384).max(1);

    let mut expected = Vec::with_capacity(frames);
    for index in 0..frames {
        let picture = picture_samples(coded_width, coded_height, index);
        let idr = index == 0;
        let (ref_idc, kind) = if idr { (3, nal::kind::IDR) } else { (0, nal::kind::SLICE) };
        for (slice, macroblocks) in picture.macroblocks.chunks(mbs_per_slice).enumerate() {
            let mut writer = BitWriter::new();
            write_slice(&mut writer, idr, index, slice * mbs_per_slice, macroblocks);
            emit_nal(ref_idc, kind, writer.finish_rbsp(), &mut stream);
        }
        let full = picture.into_i420(coded_width, coded_height);
        expected.push(crop_i420(&full, coded_width, coded_height, width, height));
    }

    SyntheticStream {
        annex_b: stream,
        frames: expected,
        width,
        height,
        coded: (coded_width, coded_height),
    }
}

/// Cut a tightly packed I420 picture down to its visible region.
fn crop_i420(full: &[u8], coded_w: u32, coded_h: u32, width: u32, height: u32) -> Vec<u8> {
    let (cw, ch) = (coded_w as usize, coded_h as usize);
    let (w, h) = (width as usize, height as usize);
    if (cw, ch) == (w, h) {
        return full.to_vec();
    }
    let (coded_cw, coded_ch) = (cw / 2, ch / 2);
    let (out_cw, out_ch) = (w / 2, h / 2);
    let mut out = Vec::with_capacity(w * h + 2 * out_cw * out_ch);
    for row in 0..h {
        out.extend_from_slice(&full[row * cw..row * cw + w]);
    }
    for plane in 0..2 {
        let base = cw * ch + plane * coded_cw * coded_ch;
        for row in 0..out_ch {
            let from = base + row * coded_cw;
            out.extend_from_slice(&full[from..from + out_cw]);
        }
    }
    out
}

/// Target size of one coded slice. Well under the megabyte a DXVA bitstream buffer holds,
/// and in the range a real encoder produces.
const MAX_SLICE_BYTES: usize = 192 * 1024;

/// One picture's samples, in macroblock order (which is how `I_PCM` codes them).
struct Picture {
    /// Per macroblock: 256 luma, then 64 Cb, then 64 Cr.
    macroblocks: Vec<[u8; 384]>,
    width_mbs: usize,
}

impl Picture {
    /// Rearrange into a tightly packed I420 picture, which is what a decoder produces.
    fn into_i420(self, width: u32, height: u32) -> Vec<u8> {
        let (w, h) = (width as usize, height as usize);
        let (cw, ch) = (w / 2, h / 2);
        let mut out = vec![0u8; w * h + 2 * cw * ch];
        let (y_plane, rest) = out.split_at_mut(w * h);
        let (u_plane, v_plane) = rest.split_at_mut(cw * ch);

        for (index, mb) in self.macroblocks.iter().enumerate() {
            let mb_x = (index % self.width_mbs) * 16;
            let mb_y = (index / self.width_mbs) * 16;
            for row in 0..16 {
                let from = row * 16;
                let to = (mb_y + row) * w + mb_x;
                y_plane[to..to + 16].copy_from_slice(&mb[from..from + 16]);
            }
            for row in 0..8 {
                let cb_from = 256 + row * 8;
                let cr_from = 320 + row * 8;
                let to = (mb_y / 2 + row) * cw + mb_x / 2;
                u_plane[to..to + 8].copy_from_slice(&mb[cb_from..cb_from + 8]);
                v_plane[to..to + 8].copy_from_slice(&mb[cr_from..cr_from + 8]);
            }
        }
        out
    }
}

/// The sample pattern.
///
/// A 32-bit integer hash of `(x, y, frame)` per component, not an arithmetic pattern. The
/// first version of this multiplied and XORed small constants, which looked varied and was
/// PERIODIC: `y * 13` repeats every 256 rows, so row 336 and row 80 held identical bytes.
/// That is a broken oracle - a decoder returning a vertically shifted picture compared
/// EQUAL to the expected one - and it cost a debugging session before it was noticed. A
/// hash with avalanche has no such period: every row, column and frame differs.
fn sample_at(x: usize, y: usize, frame: usize, component: u32) -> u8 {
    let mut h = (x as u32).wrapping_mul(0x9e37_79b1)
        ^ (y as u32).wrapping_mul(0x85eb_ca6b)
        ^ (frame as u32).wrapping_mul(0xc2b2_ae35)
        ^ component.wrapping_mul(0x27d4_eb2f);
    h ^= h >> 15;
    h = h.wrapping_mul(0x2545_f491);
    h ^= h >> 13;
    (h >> 8) as u8
}

/// One picture's macroblocks, filled with [`sample_at`].
fn picture_samples(width: u32, height: u32, frame: usize) -> Picture {
    let width_mbs = (width / 16) as usize;
    let height_mbs = (height / 16) as usize;
    let mut macroblocks = Vec::with_capacity(width_mbs * height_mbs);
    for mb in 0..width_mbs * height_mbs {
        let mut block = [0u8; 384];
        let (mb_x, mb_y) = (mb % width_mbs, mb / width_mbs);
        for row in 0..16 {
            for col in 0..16 {
                block[row * 16 + col] = sample_at(mb_x * 16 + col, mb_y * 16 + row, frame, 0);
            }
        }
        for row in 0..8 {
            for col in 0..8 {
                let (x, y) = (mb_x * 8 + col, mb_y * 8 + row);
                block[256 + row * 8 + col] = sample_at(x, y, frame, 1);
                block[320 + row * 8 + col] = sample_at(x, y, frame, 2);
            }
        }
        macroblocks.push(block);
    }
    Picture { macroblocks, width_mbs }
}

/// 7.3.2.1: a constrained-baseline SPS, no VUI.
fn write_sps(w: &mut BitWriter, width_mbs: u32, height_mbs: u32, crop_right: u32, crop_bottom: u32) {
    w.bits(66, 8); // profile_idc: baseline
    w.bits(0b1100_0000, 8); // constraint_set0 and constraint_set1: constrained baseline
    // Level 5.1, not something modest. An all-I_PCM picture is FAR larger than any encoded
    // one - 1.4 MB for a single 720p frame, against level 3.0's 1.25 Mbit coded picture
    // buffer - and a level that does not admit it is a real conformance violation, not a
    // technicality: a hardware decoder sized to the signalled level decodes the start of the
    // picture and drops the rest, which is exactly what DXVA did here (correct through luma
    // row 464 of 720, garbage after).
    w.bits(51, 8); // level_idc 5.1
    w.ue(0); // seq_parameter_set_id
    w.ue(0); // log2_max_frame_num_minus4 -> 4-bit frame_num
    w.ue(0); // pic_order_cnt_type
    w.ue(2); // log2_max_pic_order_cnt_lsb_minus4 -> 6-bit POC lsb, room for 32 pictures
    w.ue(1); // max_num_ref_frames
    w.bit(false); // gaps_in_frame_num_value_allowed_flag
    w.ue(width_mbs - 1);
    w.ue(height_mbs - 1);
    w.bit(true); // frame_mbs_only_flag
    w.bit(true); // direct_8x8_inference_flag
    if crop_right == 0 && crop_bottom == 0 {
        w.bit(false); // frame_cropping_flag
    } else {
        w.bit(true);
        w.ue(0); // frame_crop_left_offset
        w.ue(crop_right / 2); // frame_crop_right_offset, in chroma units
        w.ue(0); // frame_crop_top_offset
        w.ue(crop_bottom / 2); // frame_crop_bottom_offset
    }
    w.bit(false); // vui_parameters_present_flag
}

/// 7.3.2.2: CAVLC, one slice group, deblocking control present so it can be switched off.
fn write_pps(w: &mut BitWriter) {
    w.ue(0); // pic_parameter_set_id
    w.ue(0); // seq_parameter_set_id
    w.bit(false); // entropy_coding_mode_flag: CAVLC
    w.bit(false); // bottom_field_pic_order_in_frame_present_flag
    w.ue(0); // num_slice_groups_minus1
    w.ue(0); // num_ref_idx_l0_default_active_minus1
    w.ue(0); // num_ref_idx_l1_default_active_minus1
    w.bit(false); // weighted_pred_flag
    w.bits(0, 2); // weighted_bipred_idc
    w.se(0); // pic_init_qp_minus26
    w.se(0); // pic_init_qs_minus26
    w.se(0); // chroma_qp_index_offset
    w.bit(true); // deblocking_filter_control_present_flag
    w.bit(false); // constrained_intra_pred_flag
    w.bit(false); // redundant_pic_cnt_present_flag
}

/// 7.3.3 plus 7.3.4: one I slice, every macroblock `I_PCM`.
///
/// `first_mb` is where in the picture this slice starts; every slice of one picture repeats
/// the same header values apart from that, which is what makes them one access unit.
fn write_slice(w: &mut BitWriter, idr: bool, index: usize, first_mb: usize, macroblocks: &[[u8; 384]]) {
    w.ue(first_mb as u32); // first_mb_in_slice
    w.ue(7); // slice_type: I, and every slice in the picture is I
    w.ue(0); // pic_parameter_set_id
    w.bits(0, 4); // frame_num: only the IDR is a reference picture, so this never advances
    if idr {
        w.ue(0); // idr_pic_id
    }
    w.bits((index as u32 * 2) % 64, 6); // pic_order_cnt_lsb
    if idr {
        w.bit(false); // no_output_of_prior_pics_flag
        w.bit(false); // long_term_reference_flag
    }
    w.se(0); // slice_qp_delta: QP 26, which I_PCM ignores
    w.ue(1); // disable_deblocking_filter_idc: off (and it could not alter I_PCM anyway)

    for mb in macroblocks {
        w.ue(25); // mb_type 25 in an I slice is I_PCM
        w.align_to_byte(); // pcm_alignment_zero_bit
        w.bytes(mb);
    }
}

/// Wrap an RBSP in a NAL header and a start code, appending to `out`.
fn emit_nal(ref_idc: u8, kind: u8, rbsp: Vec<u8>, out: &mut Vec<u8>) {
    let mut payload = Vec::with_capacity(rbsp.len() + 8);
    payload.push((ref_idc << 5) | kind);
    nal::escape_rbsp(&rbsp, &mut payload);
    nal::write_annex_b(&payload, out);
}

/// Writes the bit-oriented syntax H.264 is built from.
struct BitWriter {
    bytes: Vec<u8>,
    /// Bits used in the byte being built, 0..8.
    used: u32,
    partial: u8,
}

impl BitWriter {
    fn new() -> BitWriter {
        BitWriter { bytes: Vec::new(), used: 0, partial: 0 }
    }

    fn bit(&mut self, value: bool) {
        self.partial = (self.partial << 1) | value as u8;
        self.used += 1;
        if self.used == 8 {
            self.bytes.push(self.partial);
            self.partial = 0;
            self.used = 0;
        }
    }

    fn bits(&mut self, value: u32, count: u32) {
        for i in (0..count).rev() {
            self.bit((value >> i) & 1 == 1);
        }
    }

    /// Append whole bytes. Only legal when the writer is byte-aligned, which is exactly
    /// where `I_PCM` sample data goes - and it makes writing a megabyte of PCM a memcpy
    /// rather than eight million shifts.
    fn bytes(&mut self, data: &[u8]) {
        debug_assert_eq!(self.used, 0, "byte writes need an aligned writer");
        self.bytes.extend_from_slice(data);
    }

    /// `ue(v)`: `k+1` written in binary, preceded by that many zeros minus one.
    fn ue(&mut self, value: u32) {
        let code = value + 1;
        let bits = 32 - code.leading_zeros();
        for _ in 0..bits - 1 {
            self.bit(false);
        }
        self.bits(code, bits);
    }

    /// `se(v)`.
    fn se(&mut self, value: i32) {
        let mapped = if value <= 0 { (-value as u32) * 2 } else { (value as u32) * 2 - 1 };
        self.ue(mapped);
    }

    /// Pad with zero bits to the next byte boundary.
    fn align_to_byte(&mut self) {
        while self.used != 0 {
            self.bit(false);
        }
    }

    /// Append `rbsp_trailing_bits()` and return the finished RBSP.
    fn finish_rbsp(mut self) -> Vec<u8> {
        self.bit(true); // rbsp_stop_one_bit
        self.align_to_byte();
        self.bytes
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::bitstream::reader::BitReader;
    use crate::bitstream::sps::Sps;
    use crate::bitstream::{AuSplitter, ParameterSets};

    #[test]
    fn exp_golomb_writes_what_the_reader_reads() {
        let mut w = BitWriter::new();
        for v in [0u32, 1, 2, 3, 8, 100, 65535] {
            w.ue(v);
        }
        for v in [0i32, 1, -1, 7, -7] {
            w.se(v);
        }
        let data = w.finish_rbsp();
        let mut r = BitReader::new(&data);
        for v in [0u32, 1, 2, 3, 8, 100, 65535] {
            assert_eq!(r.ue().unwrap(), v);
        }
        for v in [0i32, 1, -1, 7, -7] {
            assert_eq!(r.se().unwrap(), v);
        }
    }

    #[test]
    fn the_generated_sps_parses_back_to_the_size_asked_for() {
        let stream = synthesize(4, 3, 1);
        let mut sets = ParameterSets::new();
        let mut scratch = Vec::new();
        for raw in crate::bitstream::nal::split_annex_b(&stream.annex_b) {
            let n = crate::bitstream::nal::Nal::parse(raw).unwrap();
            if n.kind == crate::bitstream::nal::kind::SPS
                || n.kind == crate::bitstream::nal::kind::PPS
            {
                sets.add_nal(&n, &mut scratch).unwrap();
            }
        }
        let sps: &Sps = sets.sps(0).expect("the stream carries an SPS");
        assert_eq!(sps.width(), 64);
        assert_eq!(sps.height(), 48);
        assert_eq!(sps.profile_idc, 66);
        assert!(sets.pps(0).is_some());
    }

    #[test]
    fn every_picture_becomes_one_access_unit() {
        let stream = synthesize(2, 2, 5);
        let mut splitter = AuSplitter::new();
        let mut units = Vec::new();
        splitter.push_annex_b(&stream.annex_b, &mut units).unwrap();
        splitter.finish(&mut units).unwrap();
        assert_eq!(units.len(), 5);
        assert!(units[0].idr, "the first picture is an IDR");
        assert!(!units[1].idr);
        // Picture order counts advance by two per picture, and the splitter reads them.
        let order: Vec<i32> = units.iter().map(|u| u.order()).collect();
        assert_eq!(order, vec![0, 2, 4, 6, 8]);
        assert_eq!(units[0].slices.len(), 1);
    }

    #[test]
    fn expected_pixels_are_laid_out_as_i420() {
        let stream = synthesize(1, 1, 1);
        assert_eq!(stream.frames[0].len(), stream.frame_size());
        assert_eq!(stream.frame_size(), 16 * 16 + 2 * 8 * 8);
    }

    #[test]
    fn the_pattern_has_no_vertical_period() {
        // An earlier arithmetic pattern repeated exactly every 256 rows, which made a
        // vertically shifted picture compare EQUAL to a correct one - an oracle that could
        // not see the very thing it was there to catch.
        let stream = synthesize(2, 20, 1);
        let width = stream.width as usize;
        let row = |y: usize| &stream.frames[0][y * width..(y + 1) * width];
        assert_ne!(row(0), row(256));
        assert_ne!(row(1), row(257));
    }
}
