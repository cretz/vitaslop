//! Picture parameter set (ITU-T H.264 7.3.2.2).

use super::ParameterSets;
use super::reader::BitReader;
use super::sps::ScalingLists;
use crate::error::{Error, Result};

/// A parsed picture parameter set.
#[derive(Debug, Clone, PartialEq, Eq)]
pub struct Pps {
    /// `pic_parameter_set_id`.
    pub id: u32,
    /// `seq_parameter_set_id` this PPS refers to.
    pub sps_id: u32,
    /// `entropy_coding_mode_flag`: true = CABAC.
    pub cabac: bool,
    /// `bottom_field_pic_order_in_frame_present_flag` (the spec's old `pic_order_present`).
    pub bottom_field_pic_order_in_frame_present: bool,
    /// `num_slice_groups_minus1 + 1`. Anything but 1 is FMO, which this crate rejects on
    /// the VA-API path (the platform decoders handle it themselves).
    pub num_slice_groups: u32,
    /// `num_ref_idx_l0_default_active_minus1 + 1`.
    pub num_ref_idx_l0_default_active: u32,
    /// `num_ref_idx_l1_default_active_minus1 + 1`.
    pub num_ref_idx_l1_default_active: u32,
    /// `weighted_pred_flag`.
    pub weighted_pred: bool,
    /// `weighted_bipred_idc`.
    pub weighted_bipred_idc: u32,
    /// `pic_init_qp_minus26 + 26`.
    pub pic_init_qp: i32,
    /// `pic_init_qs_minus26 + 26`.
    pub pic_init_qs: i32,
    /// `chroma_qp_index_offset`.
    pub chroma_qp_index_offset: i32,
    /// `second_chroma_qp_index_offset`, defaulting to `chroma_qp_index_offset`.
    pub second_chroma_qp_index_offset: i32,
    /// `deblocking_filter_control_present_flag`.
    pub deblocking_filter_control_present: bool,
    /// `constrained_intra_pred_flag`.
    pub constrained_intra_pred: bool,
    /// `redundant_pic_cnt_present_flag`.
    pub redundant_pic_cnt_present: bool,
    /// `transform_8x8_mode_flag`.
    pub transform_8x8_mode: bool,
    /// Scaling lists in force for pictures using this PPS: the SPS's, overridden by this
    /// PPS's own matrix where it has one. Resolved here so a consumer never has to.
    pub scaling: ScalingLists,
    /// True when the PPS carried its own scaling matrix.
    pub pic_scaling_matrix_present: bool,
}

impl Pps {
    /// Parse a PPS RBSP. The SPS it references must already be known: the number of scaling
    /// lists and the fall-back rules both depend on it.
    pub fn parse(rbsp: &[u8], sets: &ParameterSets) -> Result<Pps> {
        let mut r = BitReader::new(rbsp);
        let id = r.ue_max(255, "pic_parameter_set_id")?;
        let sps_id = r.ue_max(31, "seq_parameter_set_id")?;
        let sps = sets.sps(sps_id).ok_or_else(|| {
            Error::bitstream(format!("PPS {id} references unknown SPS {sps_id}"))
        })?;

        let cabac = r.flag()?;
        let bottom_field_pic_order_in_frame_present = r.flag()?;
        let num_slice_groups = r.ue_max(7, "num_slice_groups_minus1")? + 1;
        if num_slice_groups > 1 {
            // FMO: the map itself is parsed only far enough to reject it, because nothing
            // downstream of here can act on it and a half-parsed map would be worse.
            return Err(Error::unsupported(
                "flexible macroblock ordering (num_slice_groups > 1)",
            ));
        }
        let num_ref_idx_l0_default_active = r.ue_max(31, "num_ref_idx_l0_default_active_minus1")? + 1;
        let num_ref_idx_l1_default_active = r.ue_max(31, "num_ref_idx_l1_default_active_minus1")? + 1;
        let weighted_pred = r.flag()?;
        let weighted_bipred_idc = r.bits(2)?;
        let pic_init_qp = r.se()? + 26;
        let pic_init_qs = r.se()? + 26;
        let chroma_qp_index_offset = r.se()?;
        let deblocking_filter_control_present = r.flag()?;
        let constrained_intra_pred = r.flag()?;
        let redundant_pic_cnt_present = r.flag()?;

        let mut pps = Pps {
            id,
            sps_id,
            cabac,
            bottom_field_pic_order_in_frame_present,
            num_slice_groups,
            num_ref_idx_l0_default_active,
            num_ref_idx_l1_default_active,
            weighted_pred,
            weighted_bipred_idc,
            pic_init_qp,
            pic_init_qs,
            chroma_qp_index_offset,
            second_chroma_qp_index_offset: chroma_qp_index_offset,
            deblocking_filter_control_present,
            constrained_intra_pred,
            redundant_pic_cnt_present,
            transform_8x8_mode: false,
            scaling: sps.scaling.clone(),
            pic_scaling_matrix_present: false,
        };

        // The trailing block is optional in the syntax - a PPS may simply end here.
        if r.more_rbsp_data() {
            pps.transform_8x8_mode = r.flag()?;
            pps.pic_scaling_matrix_present = r.flag()?;
            if pps.pic_scaling_matrix_present {
                let count = 6 + if pps.transform_8x8_mode {
                    if sps.chroma_format_idc != 3 { 2 } else { 6 }
                } else {
                    0
                };
                super::sps::parse_pic_scaling_matrix(&mut r, count, &mut pps.scaling)?;
            }
            pps.second_chroma_qp_index_offset = r.se()?;
        }
        Ok(pps)
    }
}
