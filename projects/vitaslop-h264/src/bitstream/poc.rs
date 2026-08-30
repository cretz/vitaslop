//! Picture order count (ITU-T H.264 8.2.1).
//!
//! POC is the stream's own presentation order, and it is the only ordering information a
//! bare H.264 elementary stream carries - a `.h264` file has no timestamps at all. Two
//! things here depend on it: frames get a presentation-order `pts` when the caller supplied
//! none, and the VA-API backend needs the top/bottom field order counts in its picture
//! parameter buffer.

use super::slice::SliceHeader;
use super::sps::Sps;
use crate::error::{Error, Result};

/// Running state for POC derivation. One per decoding sequence; reset on a seek.
#[derive(Debug, Clone, Default)]
pub struct PocState {
    prev_pic_order_cnt_msb: i32,
    prev_pic_order_cnt_lsb: i32,
    prev_frame_num: u32,
    prev_frame_num_offset: i32,
    prev_has_mmco5: bool,
}

/// The order counts of one picture.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Default)]
pub struct Poc {
    /// `TopFieldOrderCnt`.
    pub top: i32,
    /// `BottomFieldOrderCnt`.
    pub bottom: i32,
}

impl Poc {
    /// The picture's order count: the smaller of the two field counts for a frame, or the
    /// coded field's own count. This is the value to sort presentation order by.
    pub fn value(&self, field_pic: bool, bottom_field: bool) -> i32 {
        if field_pic {
            if bottom_field { self.bottom } else { self.top }
        } else {
            self.top.min(self.bottom)
        }
    }
}

impl PocState {
    /// Forget everything: used on an IDR with `no_output_of_prior_pics`, and on a seek.
    pub fn reset(&mut self) {
        *self = PocState::default();
    }

    /// Derive the POC of the picture `h` starts, advancing the state.
    ///
    /// `has_mmco5` is whether this picture's marking carries a memory-management control
    /// operation 5 ("mark everything unused"), which resets the count as if the picture
    /// were an IDR - the one piece of state that cannot be read off the header alone.
    pub fn advance(&mut self, h: &SliceHeader, sps: &Sps) -> Result<Poc> {
        let has_mmco5 =
            h.marking.as_ref().is_some_and(|m| m.mmco.iter().any(|c| c.op == 5));
        let poc = match sps.pic_order_cnt_type {
            0 => self.type0(h, sps),
            1 => self.type1(h, sps)?,
            2 => self.type2(h, sps),
            other => return Err(Error::bitstream(format!("pic_order_cnt_type {other}"))),
        };
        self.prev_has_mmco5 = has_mmco5;
        // 8.2.1: after an MMCO 5 the picture's own count is rebased to zero, and the next
        // picture derives from that rebased value.
        if has_mmco5 {
            let top = if h.field_pic && h.bottom_field { 0 } else { poc.top - poc.value(h.field_pic, h.bottom_field) };
            let bottom = if h.field_pic && !h.bottom_field { 0 } else { poc.bottom - poc.value(h.field_pic, h.bottom_field) };
            self.prev_pic_order_cnt_msb = 0;
            self.prev_pic_order_cnt_lsb = top.max(0);
            self.prev_frame_num = 0;
            self.prev_frame_num_offset = 0;
            return Ok(Poc { top, bottom });
        }
        Ok(poc)
    }

    /// 8.2.1.1.
    fn type0(&mut self, h: &SliceHeader, sps: &Sps) -> Poc {
        let max_lsb = 1i32 << sps.log2_max_pic_order_cnt_lsb;
        let lsb = h.pic_order_cnt_lsb as i32;
        let (prev_msb, prev_lsb) = if h.idr { (0, 0) } else {
            (self.prev_pic_order_cnt_msb, self.prev_pic_order_cnt_lsb)
        };
        let msb = if lsb < prev_lsb && (prev_lsb - lsb) >= max_lsb / 2 {
            prev_msb + max_lsb
        } else if lsb > prev_lsb && (lsb - prev_lsb) > max_lsb / 2 {
            prev_msb - max_lsb
        } else {
            prev_msb
        };
        let top = msb + lsb;
        let bottom = if h.field_pic {
            // A coded field has only its own count; the other is unused. Keeping them equal
            // makes `Poc::value` correct for either parity.
            top
        } else {
            top + h.delta_pic_order_cnt_bottom
        };
        // Only reference pictures update the running state (8.2.1.1).
        if h.nal_ref_idc != 0 {
            self.prev_pic_order_cnt_msb = msb;
            self.prev_pic_order_cnt_lsb = lsb;
        }
        Poc { top, bottom }
    }

    /// 8.2.1.2.
    fn type1(&mut self, h: &SliceHeader, sps: &Sps) -> Result<Poc> {
        let max_frame_num = 1i64 << sps.log2_max_frame_num;
        let frame_num = h.frame_num as i64;
        let prev_offset = if h.idr { 0 } else { self.prev_frame_num_offset as i64 };
        let frame_num_offset = if h.idr {
            0
        } else if (self.prev_frame_num as i64) > frame_num {
            prev_offset + max_frame_num
        } else {
            prev_offset
        };

        let cycle_len = sps.offset_for_ref_frame.len() as i64;
        let expected_delta: i64 = sps.offset_for_ref_frame.iter().map(|&v| v as i64).sum();

        let abs_frame_num = if cycle_len != 0 { frame_num_offset + frame_num } else { 0 };
        let abs_frame_num =
            if h.nal_ref_idc == 0 && abs_frame_num > 0 { abs_frame_num - 1 } else { abs_frame_num };

        let mut expected_poc = 0i64;
        if abs_frame_num > 0 {
            let cycle_cnt = (abs_frame_num - 1) / cycle_len;
            let frame_in_cycle = (abs_frame_num - 1) % cycle_len;
            expected_poc = cycle_cnt * expected_delta;
            for i in 0..=frame_in_cycle {
                expected_poc += sps.offset_for_ref_frame[i as usize] as i64;
            }
        }
        if h.nal_ref_idc == 0 {
            expected_poc += sps.offset_for_non_ref_pic as i64;
        }

        let d0 = h.delta_pic_order_cnt[0] as i64;
        let d1 = h.delta_pic_order_cnt[1] as i64;
        let top = expected_poc + d0;
        let bottom = top + sps.offset_for_top_to_bottom_field as i64 + d1;

        self.prev_frame_num = h.frame_num;
        self.prev_frame_num_offset = i32::try_from(frame_num_offset)
            .map_err(|_| Error::bitstream("frame_num_offset overflowed"))?;
        Ok(Poc {
            top: i32::try_from(top).map_err(|_| Error::bitstream("POC overflowed"))?,
            bottom: i32::try_from(bottom).map_err(|_| Error::bitstream("POC overflowed"))?,
        })
    }

    /// 8.2.1.3 - decode order IS presentation order.
    fn type2(&mut self, h: &SliceHeader, sps: &Sps) -> Poc {
        let max_frame_num = 1i32 << sps.log2_max_frame_num;
        let frame_num = h.frame_num as i32;
        let frame_num_offset = if h.idr {
            0
        } else if self.prev_frame_num as i32 > frame_num {
            self.prev_frame_num_offset + max_frame_num
        } else {
            self.prev_frame_num_offset
        };
        let tmp = if h.idr {
            0
        } else if h.nal_ref_idc == 0 {
            2 * (frame_num_offset + frame_num) - 1
        } else {
            2 * (frame_num_offset + frame_num)
        };
        self.prev_frame_num = h.frame_num;
        self.prev_frame_num_offset = frame_num_offset;
        Poc { top: tmp, bottom: tmp }
    }
}
