use vitaslop_h264::bitstream::{nal, ParameterSets, sps::Sps, slice::SliceHeader};
use vitaslop_h264::{mp4, bitstream::avcc};

fn main() {
    let path = std::env::args().nth(1).unwrap();
    let file = std::fs::read(&path).unwrap();
    let track = mp4::read_h264_track(&file).unwrap();
    let mut sets = ParameterSets::new();
    let mut scratch = Vec::new();
    for raw in track.avcc.sps.iter().chain(track.avcc.pps.iter()) {
        let n = nal::Nal::parse(raw).unwrap();
        sets.add_nal(&n, &mut scratch).unwrap();
    }
    let sps: &Sps = sets.sps(0).unwrap();
    println!("SPS: profile {} level {} chroma {} frame_mbs_only {} mbaff {} poc_type {} refs {} {}x{}",
        sps.profile_idc, sps.level_idc, sps.chroma_format_idc, sps.frame_mbs_only,
        sps.mb_adaptive_frame_field, sps.pic_order_cnt_type, sps.max_num_ref_frames,
        sps.width(), sps.height());
    let pps = sets.pps(0).unwrap();
    println!("PPS: cabac {} wpred {} wbipred {} l0 {} l1 {} deblock_ctrl {} poc_present {}",
        pps.cabac, pps.weighted_pred, pps.weighted_bipred_idc,
        pps.num_ref_idx_l0_default_active, pps.num_ref_idx_l1_default_active,
        pps.deblocking_filter_control_present, pps.bottom_field_pic_order_in_frame_present);

    for index in 0..4usize {
        let sample = track.sample_data(&file, index).unwrap();
        let mut annex = Vec::new();
        avcc::length_prefixed_to_annex_b(sample, track.avcc.length_size, &mut annex).unwrap();
        for (i, raw) in nal::split_annex_b(&annex).enumerate() {
            let n = nal::Nal::parse(raw).unwrap();
            if !n.is_slice() { println!("sample {index} nal {i}: type {}", n.kind); continue; }
            nal::rbsp_into(n.payload, &mut scratch);
            match SliceHeader::parse(&scratch, n.kind, n.ref_idc, &sets) {
                Ok((h, _, _)) => println!(
                    "sample {index} nal {i}: type {:?} all {} field {} ref_idc {} first_mb {} l0 {} l1 {} override {} mods {}/{} weights {} qp_delta {}",
                    h.slice_type, h.slice_type_all, h.field_pic, n.ref_idc, h.first_mb_in_slice,
                    h.num_ref_idx_l0_active, h.num_ref_idx_l1_active, h.num_ref_idx_active_override,
                    h.ref_pic_list_mod[0].len(), h.ref_pic_list_mod[1].len(),
                    h.pred_weight.is_some(), h.slice_qp_delta),
                Err(e) => println!("sample {index} nal {i}: PARSE FAILED: {e}"),
            }
        }
    }
}
