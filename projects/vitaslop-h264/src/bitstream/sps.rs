//! Sequence parameter set (ITU-T H.264 7.3.2.1).
//!
//! Parsed in full, including the VUI, because three separate things downstream need it:
//! the visible frame size (cropping is signalled here and nowhere else), the reorder depth
//! that turns decode order into presentation order, and - on the VA-API path, where we run
//! the DPB ourselves - every field of the picture parameter buffer.

use super::reader::BitReader;
use crate::error::{Error, Result};
use crate::frame::ColorInfo;

/// Scaling lists, in the spec's flat form: six 4x4 lists then six 8x8 lists.
#[derive(Debug, Clone, PartialEq, Eq)]
pub struct ScalingLists {
    /// Six 16-entry 4x4 lists, in zig-zag order.
    pub list4x4: [[u8; 16]; 6],
    /// Six 64-entry 8x8 lists, in zig-zag order. Only the first two are used outside of
    /// 4:4:4 profiles.
    pub list8x8: [[u8; 64]; 6],
}

/// Flat 16 (4x4) - what "no scaling" means.
pub const FLAT_4X4: [u8; 16] = [16; 16];
/// Flat 16 (8x8).
pub const FLAT_8X8: [u8; 64] = [16; 64];

/// Table 7-3: Default_4x4_Intra.
const DEFAULT_4X4_INTRA: [u8; 16] =
    [6, 13, 13, 20, 20, 20, 28, 28, 28, 28, 32, 32, 32, 37, 37, 42];
/// Table 7-3: Default_4x4_Inter.
const DEFAULT_4X4_INTER: [u8; 16] =
    [10, 14, 14, 20, 20, 20, 24, 24, 24, 24, 27, 27, 27, 30, 30, 34];
/// Table 7-4: Default_8x8_Intra.
const DEFAULT_8X8_INTRA: [u8; 64] = [
    6, 10, 10, 13, 11, 13, 16, 16, 16, 16, 18, 18, 18, 18, 18, 23, 23, 23, 23, 23, 23, 25, 25,
    25, 25, 25, 25, 25, 27, 27, 27, 27, 27, 27, 27, 27, 29, 29, 29, 29, 29, 29, 29, 31, 31, 31,
    31, 31, 31, 33, 33, 33, 33, 33, 36, 36, 36, 36, 38, 38, 38, 40, 40, 42,
];
/// Table 7-4: Default_8x8_Inter.
const DEFAULT_8X8_INTER: [u8; 64] = [
    9, 13, 13, 15, 13, 15, 17, 17, 17, 17, 19, 19, 19, 19, 19, 21, 21, 21, 21, 21, 21, 22, 22,
    22, 22, 22, 22, 22, 24, 24, 24, 24, 24, 24, 24, 24, 25, 25, 25, 25, 25, 25, 25, 27, 27, 27,
    27, 27, 27, 28, 28, 28, 28, 28, 30, 30, 30, 30, 32, 32, 32, 33, 33, 35,
];

impl Default for ScalingLists {
    fn default() -> Self {
        ScalingLists { list4x4: [FLAT_4X4; 6], list8x8: [FLAT_8X8; 6] }
    }
}

/// VUI fields this crate reads. The rest of the VUI is skipped, not stored.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Default)]
pub struct Vui {
    /// Colour signalling, forwarded onto every decoded frame.
    pub color: ColorInfo,
    /// `max_num_reorder_frames`, when `bitstream_restriction_flag` said so.
    pub max_num_reorder_frames: Option<u32>,
    /// `max_dec_frame_buffering`, when signalled.
    pub max_dec_frame_buffering: Option<u32>,
    /// `(num_units_in_tick, time_scale)` when `timing_info_present_flag`. The frame rate is
    /// `time_scale / (2 * num_units_in_tick)` for progressive streams.
    pub timing: Option<(u32, u32)>,
    /// Sample aspect ratio as `(width, height)` when signalled and known.
    pub sar: Option<(u32, u32)>,
}

/// A parsed sequence parameter set.
#[derive(Debug, Clone, PartialEq, Eq)]
pub struct Sps {
    /// `seq_parameter_set_id`.
    pub id: u32,
    /// `profile_idc`.
    pub profile_idc: u8,
    /// The six constraint_setN_flags, bit N = flag N.
    pub constraint_flags: u8,
    /// `level_idc`.
    pub level_idc: u8,
    /// `chroma_format_idc`: 1 = 4:2:0, the only value this crate decodes.
    pub chroma_format_idc: u32,
    /// `separate_colour_plane_flag` (4:4:4 only).
    pub separate_colour_plane: bool,
    /// Luma bit depth (8 for everything this crate decodes).
    pub bit_depth_luma: u32,
    /// Chroma bit depth.
    pub bit_depth_chroma: u32,
    /// `qpprime_y_zero_transform_bypass_flag`.
    pub qpprime_y_zero_transform_bypass: bool,
    /// Scaling lists, already resolved through the "fall back on the default/previous list"
    /// rules of Table 7-2, so a consumer never has to reimplement them.
    pub scaling: ScalingLists,
    /// True when the SPS carried its own scaling matrix (VA-API needs to know).
    pub seq_scaling_matrix_present: bool,
    /// `log2_max_frame_num_minus4 + 4`.
    pub log2_max_frame_num: u32,
    /// `pic_order_cnt_type` (0, 1 or 2).
    pub pic_order_cnt_type: u32,
    /// `log2_max_pic_order_cnt_lsb_minus4 + 4` (type 0 only).
    pub log2_max_pic_order_cnt_lsb: u32,
    /// `delta_pic_order_always_zero_flag` (type 1 only).
    pub delta_pic_order_always_zero: bool,
    /// `offset_for_non_ref_pic` (type 1 only).
    pub offset_for_non_ref_pic: i32,
    /// `offset_for_top_to_bottom_field` (type 1 only).
    pub offset_for_top_to_bottom_field: i32,
    /// `offset_for_ref_frame[]` (type 1 only).
    pub offset_for_ref_frame: Vec<i32>,
    /// `max_num_ref_frames`.
    pub max_num_ref_frames: u32,
    /// `gaps_in_frame_num_value_allowed_flag`.
    pub gaps_in_frame_num_value_allowed: bool,
    /// `pic_width_in_mbs_minus1 + 1`.
    pub pic_width_in_mbs: u32,
    /// `pic_height_in_map_units_minus1 + 1`.
    pub pic_height_in_map_units: u32,
    /// `frame_mbs_only_flag`: false means the stream may carry fields.
    pub frame_mbs_only: bool,
    /// `mb_adaptive_frame_field_flag`.
    pub mb_adaptive_frame_field: bool,
    /// `direct_8x8_inference_flag`.
    pub direct_8x8_inference: bool,
    /// Frame cropping in CROP UNITS, exactly as coded: left, right, top, bottom.
    pub crop: (u32, u32, u32, u32),
    /// The VUI, if present.
    pub vui: Option<Vui>,
}

impl Sps {
    /// Coded (uncropped) luma width in samples.
    pub fn coded_width(&self) -> u32 {
        self.pic_width_in_mbs * 16
    }

    /// Coded (uncropped) luma height in samples. A field-coded stream doubles the map units.
    pub fn coded_height(&self) -> u32 {
        self.pic_height_in_map_units * 16 * if self.frame_mbs_only { 1 } else { 2 }
    }

    /// Visible width after cropping.
    pub fn width(&self) -> u32 {
        let (l, r, _, _) = self.crop;
        self.coded_width().saturating_sub((l + r) * self.crop_unit_x())
    }

    /// Visible height after cropping.
    pub fn height(&self) -> u32 {
        let (_, _, t, b) = self.crop;
        self.coded_height().saturating_sub((t + b) * self.crop_unit_y())
    }

    /// Horizontal crop unit (7-19): 2 for 4:2:0/4:2:2, 1 otherwise.
    fn crop_unit_x(&self) -> u32 {
        match self.chroma_format_idc {
            1 | 2 if !self.separate_colour_plane => 2,
            _ => 1,
        }
    }

    /// Vertical crop unit (7-20).
    fn crop_unit_y(&self) -> u32 {
        let sub_height = if self.chroma_format_idc == 1 && !self.separate_colour_plane { 2 } else { 1 };
        sub_height * if self.frame_mbs_only { 1 } else { 2 }
    }

    /// How many frames the DPB must hold, from Annex A's `MaxDpbMbs` table and the picture
    /// size (A.3.1 h). This is what bounds the reorder delay when the VUI does not say.
    pub fn max_dpb_frames(&self) -> u32 {
        // Table A-1, MaxDpbMbs by level_idc. `constraint_set3_flag` distinguishes level 1b
        // from 1.1 for profiles that share the idc.
        let level = self.level_idc as u32;
        let is_1b = level == 11 && (self.constraint_flags & (1 << 3)) != 0;
        let max_dpb_mbs: u32 = match if is_1b { 9 } else { level } {
            0..=9 => 396,
            10 => 396,
            11 => 900,
            12 | 13 | 20 => 2376,
            21 => 4752,
            22 | 30 => 8100,
            31 => 18000,
            32 => 20480,
            40 | 41 => 32768,
            42 => 34816,
            50 => 110400,
            51 => 184320,
            _ => 184320,
        };
        let mbs = self.pic_width_in_mbs * self.pic_height_in_map_units.max(1);
        (max_dpb_mbs / mbs.max(1)).clamp(1, 16)
    }

    /// Pictures that may have to be held back before output, i.e. the reorder depth.
    ///
    /// Prefers the VUI's own `max_num_reorder_frames`; falls back to the DPB size, which is
    /// the spec's own "unknown" answer and is always safe (it only costs latency).
    pub fn max_reorder_frames(&self) -> u32 {
        if let Some(v) = self.vui.as_ref().and_then(|v| v.max_num_reorder_frames) {
            return v;
        }
        // Annex A: for these profiles with constraint_set3, reordering is forbidden outright.
        if matches!(self.profile_idc, 44 | 86 | 100 | 110 | 122 | 244)
            && (self.constraint_flags & (1 << 3)) != 0
        {
            return 0;
        }
        self.max_dpb_frames()
    }

    /// Parse an SPS from an RBSP (NAL header byte already removed and unescaped).
    pub fn parse(rbsp: &[u8]) -> Result<Sps> {
        let mut r = BitReader::new(rbsp);
        let profile_idc = r.bits(8)? as u8;
        let constraint_flags = r.bits(8)? as u8; // 6 flags + 2 reserved bits
        let level_idc = r.bits(8)? as u8;
        let id = r.ue_max(31, "seq_parameter_set_id")?;

        let mut sps = Sps {
            id,
            profile_idc,
            constraint_flags,
            level_idc,
            chroma_format_idc: 1,
            separate_colour_plane: false,
            bit_depth_luma: 8,
            bit_depth_chroma: 8,
            qpprime_y_zero_transform_bypass: false,
            scaling: ScalingLists::default(),
            seq_scaling_matrix_present: false,
            log2_max_frame_num: 4,
            pic_order_cnt_type: 0,
            log2_max_pic_order_cnt_lsb: 4,
            delta_pic_order_always_zero: false,
            offset_for_non_ref_pic: 0,
            offset_for_top_to_bottom_field: 0,
            offset_for_ref_frame: Vec::new(),
            max_num_ref_frames: 0,
            gaps_in_frame_num_value_allowed: false,
            pic_width_in_mbs: 0,
            pic_height_in_map_units: 0,
            frame_mbs_only: true,
            mb_adaptive_frame_field: false,
            direct_8x8_inference: true,
            crop: (0, 0, 0, 0),
            vui: None,
        };

        // The "high profile" block. The list of idcs is the spec's own (7.3.2.1.1).
        if matches!(profile_idc, 100 | 110 | 122 | 244 | 44 | 83 | 86 | 118 | 128 | 138 | 139 | 134 | 135)
        {
            sps.chroma_format_idc = r.ue_max(3, "chroma_format_idc")?;
            if sps.chroma_format_idc == 3 {
                sps.separate_colour_plane = r.flag()?;
            }
            sps.bit_depth_luma = r.ue_max(6, "bit_depth_luma_minus8")? + 8;
            sps.bit_depth_chroma = r.ue_max(6, "bit_depth_chroma_minus8")? + 8;
            sps.qpprime_y_zero_transform_bypass = r.flag()?;
            sps.seq_scaling_matrix_present = r.flag()?;
            if sps.seq_scaling_matrix_present {
                let count = if sps.chroma_format_idc != 3 { 8 } else { 12 };
                parse_scaling_matrix(&mut r, count, &mut sps.scaling, true)?;
            }
        }

        sps.log2_max_frame_num = r.ue_max(12, "log2_max_frame_num_minus4")? + 4;
        sps.pic_order_cnt_type = r.ue_max(2, "pic_order_cnt_type")?;
        match sps.pic_order_cnt_type {
            0 => sps.log2_max_pic_order_cnt_lsb = r.ue_max(12, "log2_max_poc_lsb_minus4")? + 4,
            1 => {
                sps.delta_pic_order_always_zero = r.flag()?;
                sps.offset_for_non_ref_pic = r.se()?;
                sps.offset_for_top_to_bottom_field = r.se()?;
                let n = r.ue_max(255, "num_ref_frames_in_pic_order_cnt_cycle")?;
                sps.offset_for_ref_frame.reserve(n as usize);
                for _ in 0..n {
                    sps.offset_for_ref_frame.push(r.se()?);
                }
            }
            _ => {}
        }

        sps.max_num_ref_frames = r.ue_max(16, "max_num_ref_frames")?;
        sps.gaps_in_frame_num_value_allowed = r.flag()?;
        sps.pic_width_in_mbs = r.ue_max(u32::MAX / 16 - 1, "pic_width_in_mbs_minus1")? + 1;
        sps.pic_height_in_map_units = r.ue_max(u32::MAX / 32 - 1, "pic_height_in_map_units_minus1")? + 1;
        sps.frame_mbs_only = r.flag()?;
        if !sps.frame_mbs_only {
            sps.mb_adaptive_frame_field = r.flag()?;
        }
        sps.direct_8x8_inference = r.flag()?;
        if r.flag()? {
            sps.crop = (r.ue()?, r.ue()?, r.ue()?, r.ue()?);
        }
        if r.flag()? {
            sps.vui = Some(parse_vui(&mut r)?);
        }

        if sps.pic_width_in_mbs == 0 || sps.pic_height_in_map_units == 0 {
            return Err(Error::bitstream("SPS codes a zero-sized picture"));
        }
        Ok(sps)
    }
}

/// 7.3.2.1.1.1 - the six (or twelve) scaling lists, with the fall-back rules applied.
fn parse_scaling_matrix(
    r: &mut BitReader<'_>,
    count: usize,
    out: &mut ScalingLists,
    is_sps: bool,
) -> Result<()> {
    for i in 0..count {
        let present = r.flag()?;
        if i < 6 {
            let fallback: [u8; 16] = match (present, i, is_sps) {
                (true, _, _) => [0; 16], // filled below
                (false, 0, true) => DEFAULT_4X4_INTRA,
                (false, 3, true) => DEFAULT_4X4_INTER,
                (false, 0, false) | (false, 3, false) => out.list4x4[i],
                (false, _, _) => out.list4x4[i - 1],
            };
            if present {
                let mut list = FLAT_4X4;
                let use_default = parse_scaling_list(r, &mut list)?;
                out.list4x4[i] =
                    if use_default {
                        if i < 3 { DEFAULT_4X4_INTRA } else { DEFAULT_4X4_INTER }
                    } else {
                        list
                    };
            } else {
                out.list4x4[i] = fallback;
            }
        } else {
            let j = i - 6;
            let fallback: [u8; 64] = match (present, j, is_sps) {
                (true, _, _) => [0; 64],
                (false, 0, true) => DEFAULT_8X8_INTRA,
                (false, 1, true) => DEFAULT_8X8_INTER,
                (false, 0, false) | (false, 1, false) => out.list8x8[j],
                (false, _, _) => out.list8x8[j - 2],
            };
            if present {
                let mut list = FLAT_8X8;
                let use_default = parse_scaling_list(r, &mut list)?;
                out.list8x8[j] = if use_default {
                    if j % 2 == 0 { DEFAULT_8X8_INTRA } else { DEFAULT_8X8_INTER }
                } else {
                    list
                };
            } else {
                out.list8x8[j] = fallback;
            }
        }
    }
    Ok(())
}

/// The PPS's scaling matrix (7.3.2.2), whose fall-back is the SPS's list rather than the
/// spec default - which is why the two share one implementation and a flag.
pub(crate) fn parse_pic_scaling_matrix(
    r: &mut BitReader<'_>,
    count: usize,
    out: &mut ScalingLists,
) -> Result<()> {
    parse_scaling_matrix(r, count, out, false)
}

/// 7.3.2.1.1.1 scaling_list(): returns `use_default_scaling_matrix_flag`.
fn parse_scaling_list(r: &mut BitReader<'_>, list: &mut [u8]) -> Result<bool> {
    let mut last_scale: i32 = 8;
    let mut next_scale: i32 = 8;
    let mut use_default = false;
    for (j, slot) in list.iter_mut().enumerate() {
        if next_scale != 0 {
            let delta = r.se()?;
            next_scale = (last_scale + delta + 256) % 256;
            if j == 0 && next_scale == 0 {
                use_default = true;
            }
        }
        *slot = if next_scale == 0 { last_scale as u8 } else { next_scale as u8 };
        last_scale = *slot as i32;
    }
    Ok(use_default)
}

/// Annex E VUI. Everything not stored is still parsed, because the fields that follow it
/// depend on the exact bit count of what came before.
fn parse_vui(r: &mut BitReader<'_>) -> Result<Vui> {
    let mut vui = Vui { color: ColorInfo::UNSPECIFIED, ..Vui::default() };

    if r.flag()? {
        // aspect_ratio_info_present_flag
        let idc = r.bits(8)?;
        vui.sar = match idc {
            255 => {
                let w = r.bits(16)?;
                let h = r.bits(16)?;
                if w != 0 && h != 0 { Some((w, h)) } else { None }
            }
            // Table E-1.
            1..=16 => Some(SAR_TABLE[idc as usize - 1]),
            _ => None,
        };
    }
    if r.flag()? {
        // overscan_info_present_flag
        let _overscan_appropriate = r.flag()?;
    }
    if r.flag()? {
        // video_signal_type_present_flag
        let _video_format = r.bits(3)?;
        vui.color.full_range = r.flag()?;
        if r.flag()? {
            // colour_description_present_flag
            vui.color.primaries = r.bits(8)? as u8;
            vui.color.transfer = r.bits(8)? as u8;
            vui.color.matrix = r.bits(8)? as u8;
        }
    }
    if r.flag()? {
        // chroma_loc_info_present_flag
        let _top = r.ue_max(5, "chroma_sample_loc_type_top_field")?;
        let _bottom = r.ue_max(5, "chroma_sample_loc_type_bottom_field")?;
    }
    if r.flag()? {
        // timing_info_present_flag
        let num_units_in_tick = r.bits(32)?;
        let time_scale = r.bits(32)?;
        let _fixed_frame_rate = r.flag()?;
        if num_units_in_tick != 0 && time_scale != 0 {
            vui.timing = Some((num_units_in_tick, time_scale));
        }
    }
    let nal_hrd = r.flag()?;
    if nal_hrd {
        parse_hrd(r)?;
    }
    let vcl_hrd = r.flag()?;
    if vcl_hrd {
        parse_hrd(r)?;
    }
    if nal_hrd || vcl_hrd {
        let _low_delay_hrd = r.flag()?;
    }
    let _pic_struct_present = r.flag()?;
    if r.flag()? {
        // bitstream_restriction_flag
        let _motion_vectors_over_pic_boundaries = r.flag()?;
        let _max_bytes_per_pic_denom = r.ue()?;
        let _max_bits_per_mb_denom = r.ue()?;
        let _log2_max_mv_length_horizontal = r.ue()?;
        let _log2_max_mv_length_vertical = r.ue()?;
        vui.max_num_reorder_frames = Some(r.ue_max(16, "max_num_reorder_frames")?);
        vui.max_dec_frame_buffering = Some(r.ue_max(16, "max_dec_frame_buffering")?);
    }
    Ok(vui)
}

/// Annex E.1.2 hrd_parameters(): skipped, but its length depends on its own contents.
fn parse_hrd(r: &mut BitReader<'_>) -> Result<()> {
    let cpb_cnt = r.ue_max(31, "cpb_cnt_minus1")? + 1;
    let _bit_rate_scale = r.bits(4)?;
    let _cpb_size_scale = r.bits(4)?;
    for _ in 0..cpb_cnt {
        let _bit_rate_value = r.ue()?;
        let _cpb_size_value = r.ue()?;
        let _cbr_flag = r.flag()?;
    }
    r.skip(5 + 5 + 5 + 5)?; // initial_cpb_removal_delay_length, cpb_removal, dpb_output, time_offset
    Ok(())
}

/// Table E-1 entries 1..=16.
const SAR_TABLE: [(u32, u32); 16] = [
    (1, 1),
    (12, 11),
    (10, 11),
    (16, 11),
    (40, 33),
    (24, 11),
    (20, 11),
    (32, 11),
    (80, 33),
    (18, 11),
    (15, 11),
    (64, 33),
    (160, 99),
    (4, 3),
    (3, 2),
    (2, 1),
];
