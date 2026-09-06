//! Slice header (ITU-T H.264 7.3.3), including the reference-list modification,
//! prediction-weight and reference-marking sub-syntax.
//!
//! Two very different consumers need this. Every backend needs enough of it to tell where
//! one picture ends and the next begins (7.4.1.2.4), because the platform decoders are fed
//! whole access units. The VA-API backend, where we drive the hardware directly, needs
//! ALL of it: the weight tables and the marking commands go straight into VA buffers, and
//! the bit position where slice data starts is a field of the slice parameter buffer.

use super::pps::Pps;
use super::reader::BitReader;
use super::sps::Sps;
use crate::error::{Error, Result};

/// Slice coding type, with the "all slices in this picture are this type" distinction the
/// spec draws between values 0-4 and 5-9 kept as a separate flag.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum SliceType {
    /// Predictive.
    P,
    /// Bi-predictive.
    B,
    /// Intra.
    I,
    /// Switching P.
    Sp,
    /// Switching I.
    Si,
}

impl SliceType {
    fn from_code(code: u32) -> Result<(SliceType, bool)> {
        let all = code >= 5;
        let t = match code % 5 {
            0 => SliceType::P,
            1 => SliceType::B,
            2 => SliceType::I,
            3 => SliceType::Sp,
            4 => SliceType::Si,
            _ => unreachable!(),
        };
        if code > 9 {
            return Err(Error::bitstream(format!("slice_type {code} out of range")));
        }
        Ok((t, all))
    }

    /// True for slice types that reference a list 0.
    pub fn uses_list0(self) -> bool {
        matches!(self, SliceType::P | SliceType::Sp | SliceType::B)
    }

    /// True for slice types that reference a list 1.
    pub fn uses_list1(self) -> bool {
        matches!(self, SliceType::B)
    }
}

/// One `ref_pic_list_modification` command (7.3.3.1).
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub struct RefPicListMod {
    /// `modification_of_pic_nums_idc`: 0/1 short-term subtract/add, 2 long-term.
    pub idc: u32,
    /// `abs_diff_pic_num_minus1` for idc 0/1, `long_term_pic_num` for idc 2.
    pub value: u32,
}

/// A single list's weights for one component.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Default)]
pub struct Weight {
    /// Multiplicative weight.
    pub weight: i32,
    /// Additive offset.
    pub offset: i32,
}

/// `pred_weight_table` (7.3.3.2).
#[derive(Debug, Clone, PartialEq, Eq)]
pub struct PredWeightTable {
    /// `luma_log2_weight_denom`.
    pub luma_log2_denom: u32,
    /// `chroma_log2_weight_denom`.
    pub chroma_log2_denom: u32,
    /// Luma weights, `[list][ref_idx]`.
    pub luma: [Vec<Weight>; 2],
    /// Chroma weights, `[list][ref_idx][cb=0, cr=1]`.
    pub chroma: [Vec<[Weight; 2]>; 2],
}

/// One `dec_ref_pic_marking` MMCO command (7.3.3.3).
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub struct Mmco {
    /// `memory_management_control_operation`, 1..=6.
    pub op: u32,
    /// `difference_of_pic_nums_minus1` (ops 1, 3).
    pub difference_of_pic_nums_minus1: u32,
    /// `long_term_pic_num` (op 2).
    pub long_term_pic_num: u32,
    /// `long_term_frame_idx` (ops 3, 6).
    pub long_term_frame_idx: u32,
    /// `max_long_term_frame_idx_plus1` (op 4).
    pub max_long_term_frame_idx_plus1: u32,
}

/// `dec_ref_pic_marking` (7.3.3.3), for a reference picture.
#[derive(Debug, Clone, PartialEq, Eq, Default)]
pub struct RefPicMarking {
    /// IDR only: `no_output_of_prior_pics_flag`.
    pub no_output_of_prior_pics: bool,
    /// IDR only: `long_term_reference_flag`.
    pub long_term_reference: bool,
    /// `adaptive_ref_pic_marking_mode_flag`.
    pub adaptive: bool,
    /// The MMCO commands, in order, when `adaptive`.
    pub mmco: Vec<Mmco>,
}

/// A parsed slice header.
#[derive(Debug, Clone, PartialEq, Eq)]
pub struct SliceHeader {
    /// `first_mb_in_slice`.
    pub first_mb_in_slice: u32,
    /// Slice coding type.
    pub slice_type: SliceType,
    /// True when the coded value was 5..9 ("all slices in the picture are this type").
    pub slice_type_all: bool,
    /// `pic_parameter_set_id`.
    pub pps_id: u32,
    /// `colour_plane_id` (4:4:4 with separate planes only).
    pub colour_plane_id: u32,
    /// `frame_num`.
    pub frame_num: u32,
    /// `field_pic_flag`.
    pub field_pic: bool,
    /// `bottom_field_flag`.
    pub bottom_field: bool,
    /// True when this slice belongs to an IDR picture (from the NAL type, not the header).
    pub idr: bool,
    /// `nal_ref_idc` of the NAL this header came from.
    pub nal_ref_idc: u8,
    /// `idr_pic_id` (IDR only).
    pub idr_pic_id: u32,
    /// `pic_order_cnt_lsb` (POC type 0).
    pub pic_order_cnt_lsb: u32,
    /// `delta_pic_order_cnt_bottom` (POC type 0 with bottom-field POC present).
    pub delta_pic_order_cnt_bottom: i32,
    /// `delta_pic_order_cnt[0..2]` (POC type 1).
    pub delta_pic_order_cnt: [i32; 2],
    /// `redundant_pic_cnt`.
    pub redundant_pic_cnt: u32,
    /// `direct_spatial_mv_pred_flag` (B slices).
    pub direct_spatial_mv_pred: bool,
    /// Active reference count for list 0 (override applied).
    pub num_ref_idx_l0_active: u32,
    /// Active reference count for list 1 (override applied).
    pub num_ref_idx_l1_active: u32,
    /// True when the header overrode the PPS defaults.
    pub num_ref_idx_active_override: bool,
    /// Reference list modification commands per list.
    pub ref_pic_list_mod: [Vec<RefPicListMod>; 2],
    /// Prediction weights, when the PPS asked for explicit weighting.
    pub pred_weight: Option<PredWeightTable>,
    /// Reference marking, present only when `nal_ref_idc != 0`.
    pub marking: Option<RefPicMarking>,
    /// `cabac_init_idc`.
    pub cabac_init_idc: u32,
    /// `slice_qp_delta`.
    pub slice_qp_delta: i32,
    /// `sp_for_switch_flag` (SP slices).
    pub sp_for_switch: bool,
    /// `slice_qs_delta` (SP/SI slices).
    pub slice_qs_delta: i32,
    /// `disable_deblocking_filter_idc`.
    pub disable_deblocking_filter_idc: u32,
    /// `slice_alpha_c0_offset_div2`.
    pub slice_alpha_c0_offset_div2: i32,
    /// `slice_beta_offset_div2`.
    pub slice_beta_offset_div2: i32,
    /// Bit position, within the RBSP, of the first bit of `slice_data()`. VA-API's
    /// `slice_data_bit_offset` is this value counted from the start of the NAL, i.e. this
    /// plus the eight bits of the NAL header.
    pub slice_data_bit_offset: usize,
}

impl SliceHeader {
    /// Parse a slice header out of a slice NAL's RBSP.
    ///
    /// `nal_type` and `nal_ref_idc` come from the NAL header byte; `sets` must already hold
    /// the SPS and PPS this slice refers to.
    pub fn parse(
        rbsp: &[u8],
        nal_type: u8,
        nal_ref_idc: u8,
        sets: &super::ParameterSets,
    ) -> Result<(SliceHeader, Sps, Pps)> {
        let mut r = BitReader::new(rbsp);
        let idr = nal_type == super::nal::kind::IDR;

        let first_mb_in_slice = r.ue()?;
        let (slice_type, slice_type_all) = SliceType::from_code(r.ue()?)?;
        let pps_id = r.ue_max(255, "pic_parameter_set_id")?;
        let pps = sets
            .pps(pps_id)
            .ok_or_else(|| Error::bitstream(format!("slice references unknown PPS {pps_id}")))?
            .clone();
        let sps = sets
            .sps(pps.sps_id)
            .ok_or_else(|| Error::bitstream(format!("PPS {pps_id} references a missing SPS")))?
            .clone();

        let mut h = SliceHeader {
            first_mb_in_slice,
            slice_type,
            slice_type_all,
            pps_id,
            colour_plane_id: 0,
            frame_num: 0,
            field_pic: false,
            bottom_field: false,
            idr,
            nal_ref_idc,
            idr_pic_id: 0,
            pic_order_cnt_lsb: 0,
            delta_pic_order_cnt_bottom: 0,
            delta_pic_order_cnt: [0, 0],
            redundant_pic_cnt: 0,
            direct_spatial_mv_pred: false,
            num_ref_idx_l0_active: pps.num_ref_idx_l0_default_active,
            num_ref_idx_l1_active: pps.num_ref_idx_l1_default_active,
            num_ref_idx_active_override: false,
            ref_pic_list_mod: [Vec::new(), Vec::new()],
            pred_weight: None,
            marking: None,
            cabac_init_idc: 0,
            slice_qp_delta: 0,
            sp_for_switch: false,
            slice_qs_delta: 0,
            disable_deblocking_filter_idc: 0,
            slice_alpha_c0_offset_div2: 0,
            slice_beta_offset_div2: 0,
            slice_data_bit_offset: 0,
        };

        if sps.separate_colour_plane {
            h.colour_plane_id = r.bits(2)?;
        }
        h.frame_num = r.bits(sps.log2_max_frame_num)?;
        if !sps.frame_mbs_only {
            h.field_pic = r.flag()?;
            if h.field_pic {
                h.bottom_field = r.flag()?;
            }
        }
        if idr {
            h.idr_pic_id = r.ue_max(65535, "idr_pic_id")?;
        }
        match sps.pic_order_cnt_type {
            0 => {
                h.pic_order_cnt_lsb = r.bits(sps.log2_max_pic_order_cnt_lsb)?;
                if pps.bottom_field_pic_order_in_frame_present && !h.field_pic {
                    h.delta_pic_order_cnt_bottom = r.se()?;
                }
            }
            1 if !sps.delta_pic_order_always_zero => {
                h.delta_pic_order_cnt[0] = r.se()?;
                if pps.bottom_field_pic_order_in_frame_present && !h.field_pic {
                    h.delta_pic_order_cnt[1] = r.se()?;
                }
            }
            _ => {}
        }
        if pps.redundant_pic_cnt_present {
            h.redundant_pic_cnt = r.ue_max(127, "redundant_pic_cnt")?;
        }
        if h.slice_type == SliceType::B {
            h.direct_spatial_mv_pred = r.flag()?;
        }
        if h.slice_type.uses_list0() {
            h.num_ref_idx_active_override = r.flag()?;
            if h.num_ref_idx_active_override {
                h.num_ref_idx_l0_active = r.ue_max(31, "num_ref_idx_l0_active_minus1")? + 1;
                if h.slice_type.uses_list1() {
                    h.num_ref_idx_l1_active = r.ue_max(31, "num_ref_idx_l1_active_minus1")? + 1;
                }
            }
        }
        if nal_type == super::nal::kind::SLICE_EXT {
            return Err(Error::unsupported("MVC/SVC slice extension"));
        }
        parse_ref_pic_list_modification(&mut r, &mut h)?;

        // 7.3.3: `weighted_pred_flag` gates the weight table for P and SP slices ONLY - a B
        // slice takes its weighting from `weighted_bipred_idc`. Reading a table for a B
        // slice because `weighted_pred_flag` happened to be set puts every following field
        // of the header at the wrong bit. Found on a retail stream that sets
        // `weighted_pred_flag` with `weighted_bipred_idc` = 0: its P slices carry a table
        // and its B slices do not.
        let explicit_p = pps.weighted_pred && matches!(h.slice_type, SliceType::P | SliceType::Sp);
        let explicit_bi = pps.weighted_bipred_idc == 1 && h.slice_type == SliceType::B;
        if explicit_p || explicit_bi {
            h.pred_weight = Some(parse_pred_weight_table(&mut r, &h, &sps)?);
        }
        if nal_ref_idc != 0 {
            h.marking = Some(parse_dec_ref_pic_marking(&mut r, idr)?);
        }
        if pps.cabac && h.slice_type != SliceType::I && h.slice_type != SliceType::Si {
            h.cabac_init_idc = r.ue_max(2, "cabac_init_idc")?;
        }
        h.slice_qp_delta = r.se()?;
        if matches!(h.slice_type, SliceType::Sp | SliceType::Si) {
            if h.slice_type == SliceType::Sp {
                h.sp_for_switch = r.flag()?;
            }
            h.slice_qs_delta = r.se()?;
        }
        if pps.deblocking_filter_control_present {
            h.disable_deblocking_filter_idc = r.ue_max(2, "disable_deblocking_filter_idc")?;
            if h.disable_deblocking_filter_idc != 1 {
                h.slice_alpha_c0_offset_div2 = r.se()?;
                h.slice_beta_offset_div2 = r.se()?;
            }
        }
        // num_slice_groups > 1 would carry slice_group_change_cycle here; the PPS parser
        // has already refused that case, so there is nothing left to read.

        h.slice_data_bit_offset = r.bit_pos();
        Ok((h, sps, pps))
    }

    /// 7.4.1.2.4: does `self` start a new coded picture relative to `prev`?
    ///
    /// This is what splits a NAL stream into access units when the stream carries no access
    /// unit delimiters, which most do not. Getting it wrong merges two pictures into one
    /// decode call, so the full condition list is implemented rather than the usual
    /// "frame_num changed" shortcut.
    pub fn starts_new_picture(&self, prev: &SliceHeader, sps: &Sps, pps: &Pps) -> bool {
        if self.frame_num != prev.frame_num
            || self.pps_id != prev.pps_id
            || self.field_pic != prev.field_pic
            || (self.field_pic && self.bottom_field != prev.bottom_field)
            || self.idr != prev.idr
            || (self.nal_ref_idc == 0) != (prev.nal_ref_idc == 0)
        {
            return true;
        }
        if self.idr && prev.idr && self.idr_pic_id != prev.idr_pic_id {
            return true;
        }
        if sps.pic_order_cnt_type == 0
            && (self.pic_order_cnt_lsb != prev.pic_order_cnt_lsb
                || (pps.bottom_field_pic_order_in_frame_present
                    && self.delta_pic_order_cnt_bottom != prev.delta_pic_order_cnt_bottom))
        {
            return true;
        }
        if sps.pic_order_cnt_type == 1
            && self.delta_pic_order_cnt != prev.delta_pic_order_cnt
        {
            return true;
        }
        // A first_mb_in_slice that does not advance means the picture restarted. This also
        // covers the ordinary "next picture, first slice" case.
        self.first_mb_in_slice <= prev.first_mb_in_slice
    }
}

/// 7.3.3.1.
fn parse_ref_pic_list_modification(r: &mut BitReader<'_>, h: &mut SliceHeader) -> Result<()> {
    let lists: usize = match h.slice_type {
        SliceType::I | SliceType::Si => 0,
        SliceType::B => 2,
        _ => 1,
    };
    for list in 0..lists {
        if !r.flag()? {
            continue;
        }
        loop {
            let idc = r.ue_max(3, "modification_of_pic_nums_idc")?;
            if idc == 3 {
                break;
            }
            let value = r.ue()?;
            h.ref_pic_list_mod[list].push(RefPicListMod { idc, value });
            if h.ref_pic_list_mod[list].len() > 64 {
                return Err(Error::bitstream("ref_pic_list_modification never terminates"));
            }
        }
    }
    Ok(())
}

/// 7.3.3.2.
fn parse_pred_weight_table(
    r: &mut BitReader<'_>,
    h: &SliceHeader,
    sps: &Sps,
) -> Result<PredWeightTable> {
    let luma_log2_denom = r.ue_max(7, "luma_log2_weight_denom")?;
    let has_chroma = sps.chroma_format_idc != 0 && !sps.separate_colour_plane;
    let chroma_log2_denom =
        if has_chroma { r.ue_max(7, "chroma_log2_weight_denom")? } else { 0 };

    let mut table = PredWeightTable {
        luma_log2_denom,
        chroma_log2_denom,
        luma: [Vec::new(), Vec::new()],
        chroma: [Vec::new(), Vec::new()],
    };
    let counts = [h.num_ref_idx_l0_active, h.num_ref_idx_l1_active];
    let lists = if h.slice_type == SliceType::B { 2 } else { 1 };
    for (list, &count) in counts.iter().enumerate().take(lists) {
        for _ in 0..count {
            let mut luma = Weight { weight: 1 << luma_log2_denom, offset: 0 };
            if r.flag()? {
                luma.weight = r.se()?;
                luma.offset = r.se()?;
            }
            table.luma[list].push(luma);
            let mut chroma = [Weight { weight: 1 << chroma_log2_denom, offset: 0 }; 2];
            if has_chroma && r.flag()? {
                for c in chroma.iter_mut() {
                    c.weight = r.se()?;
                    c.offset = r.se()?;
                }
            }
            table.chroma[list].push(chroma);
        }
    }
    Ok(table)
}

/// 7.3.3.3.
fn parse_dec_ref_pic_marking(r: &mut BitReader<'_>, idr: bool) -> Result<RefPicMarking> {
    let mut m = RefPicMarking::default();
    if idr {
        m.no_output_of_prior_pics = r.flag()?;
        m.long_term_reference = r.flag()?;
        return Ok(m);
    }
    m.adaptive = r.flag()?;
    if !m.adaptive {
        return Ok(m);
    }
    loop {
        let op = r.ue_max(6, "memory_management_control_operation")?;
        if op == 0 {
            break;
        }
        let mut c = Mmco {
            op,
            difference_of_pic_nums_minus1: 0,
            long_term_pic_num: 0,
            long_term_frame_idx: 0,
            max_long_term_frame_idx_plus1: 0,
        };
        match op {
            1 | 3 => c.difference_of_pic_nums_minus1 = r.ue()?,
            2 => c.long_term_pic_num = r.ue()?,
            4 => c.max_long_term_frame_idx_plus1 = r.ue()?,
            6 => c.long_term_frame_idx = r.ue()?,
            _ => {}
        }
        if op == 3 {
            c.long_term_frame_idx = r.ue()?;
        }
        m.mmco.push(c);
        if m.mmco.len() > 64 {
            return Err(Error::bitstream("dec_ref_pic_marking never terminates"));
        }
    }
    Ok(m)
}
