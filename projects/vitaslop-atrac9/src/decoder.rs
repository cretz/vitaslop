//! The frame decoder: data structures (frame/block/channel) and the full per-frame
//! pipeline, ported from LibAtrac9 `unpack.c`, `scale_factors.c`, `bit_allocation.c`,
//! `quantization.c`, and `decoder.c`. Parent pointers in the C are replaced by
//! explicit context: the immutable [`DecodeCtx`] (config, codebooks, tables) and a
//! mutable [`Frame`]. Cross-channel reads (stereo, shared scale factors) are done by
//! copying the small arrays involved rather than aliasing.

use crate::bandext::{apply_band_extension_channel, Rng, BEX_ENCODED_VALUE_COUNTS, BEX_DATA_LENGTHS, BEX_GROUP_INFO};
use crate::bit_reader::BitReader;
use crate::config::Config;
use crate::huffman::{decode_huffman_values, read_huffman_value, Codebooks};
use crate::mdct::run_imdct;
use crate::tables::{
    Trig, Windows, BAND_TO_QUANT_UNIT_COUNT, MAX_BAND_COUNT, MAX_EXTENSION_BAND, MAX_HUFF_PRECISION,
    MIN_BAND_COUNT, QUANTIZER_FINE_STEP_SIZE, QUANTIZER_STEP_SIZE, QUANT_UNIT_TO_CODEBOOK_INDEX,
    QUANT_UNIT_TO_COEFF_COUNT, QUANT_UNIT_TO_COEFF_INDEX, SCALE_FACTOR_WEIGHTS, SPECTRUM_SCALE,
};
use crate::Error;

/// Block channel layout (`BlockType`). Discriminants match the config table.
#[derive(Clone, Copy, PartialEq, Eq, Debug)]
pub(crate) enum BlockType {
    Mono = 0,
    Stereo = 1,
    Lfe = 2,
}

const MAX_FRAME_SAMPLES: usize = 256;

/// Precomputed gradient curves, `[length-1][i]` (LibAtrac9 `GradientCurves`).
pub(crate) type GradientCurves = [[u8; 48]; 48];

/// Immutable per-decode context threaded through the pipeline.
pub(crate) struct DecodeCtx<'a> {
    pub config: &'a Config,
    pub codebooks: &'a Codebooks,
    pub trig: &'a Trig,
    pub windows: &'a Windows,
    pub gradient_curves: &'a GradientCurves,
}

/// One decoded channel's persistent and per-frame state.
pub(crate) struct Channel {
    pub channel_index: i32,
    pub mdct_bits: i32,
    /// MDCT overlap history; persists across frames.
    pub imdct_previous: [f64; MAX_FRAME_SAMPLES],
    pub pcm: [f64; MAX_FRAME_SAMPLES],
    pub spectra: [f64; MAX_FRAME_SAMPLES],
    pub coded_quant_units: i32,
    pub scale_factor_coding_mode: i32,
    pub scale_factors: [i32; 31],
    /// Scale factors from the previous frame (delta coding baseline); persists.
    pub scale_factors_prev: [i32; 31],
    pub precisions: [i32; 30],
    pub precisions_fine: [i32; 30],
    pub precision_mask: [i32; 30],
    pub codebook_set: [i32; 30],
    pub quantized_spectra: [i32; MAX_FRAME_SAMPLES],
    pub quantized_spectra_fine: [i32; MAX_FRAME_SAMPLES],
    pub bex_mode: i32,
    pub bex_value_count: i32,
    pub bex_values: [i32; 4],
    /// Noise RNG; seeded once then persists across frames.
    pub rng: Rng,
}

impl Channel {
    fn new(channel_index: i32, mdct_bits: i32) -> Channel {
        Channel {
            channel_index,
            mdct_bits,
            imdct_previous: [0.0; MAX_FRAME_SAMPLES],
            pcm: [0.0; MAX_FRAME_SAMPLES],
            spectra: [0.0; MAX_FRAME_SAMPLES],
            coded_quant_units: 0,
            scale_factor_coding_mode: 0,
            scale_factors: [0; 31],
            scale_factors_prev: [0; 31],
            precisions: [0; 30],
            precisions_fine: [0; 30],
            precision_mask: [0; 30],
            codebook_set: [0; 30],
            quantized_spectra: [0; MAX_FRAME_SAMPLES],
            quantized_spectra_fine: [0; MAX_FRAME_SAMPLES],
            bex_mode: 0,
            bex_value_count: 0,
            bex_values: [0; 4],
            rng: Rng::default(),
        }
    }
}

/// One block (1 or 2 channels) within a frame.
pub(crate) struct Block {
    pub block_type: BlockType,
    /// Position of this block in the frame; kept for fidelity with the reference.
    #[allow(dead_code)]
    pub block_index: i32,
    pub channel_count: i32,
    pub first_in_superframe: bool,
    pub reuse_band_params: bool,
    pub band_count: i32,
    pub stereo_band: i32,
    pub extension_band: i32,
    pub quantization_unit_count: i32,
    pub stereo_quantization_unit: i32,
    pub extension_unit: i32,
    pub quantization_units_prev: i32,
    /// The gradient curve, one entry per quantisation unit.
    ///
    /// >>> 48, NOT 31, AND THE DIFFERENCE CRASHED THE EMULATOR. `create_gradient`
    /// writes `gradient[i]` for `i` up to `gradient_end_unit`, and the unpack validation
    /// deliberately admits `gradient_end_unit` anywhere in `0..48` - the same 48 the
    /// gradient CURVES are sized to. At 31 a perfectly legal stream indexed one past the
    /// end and panicked, which in the browser takes the whole run worker with it.
    /// Measured on a retail title the moment its AT9 voices first decoded for real.
    pub gradient: [i32; 48],
    pub gradient_mode: i32,
    pub gradient_start_unit: i32,
    pub gradient_start_value: i32,
    pub gradient_end_unit: i32,
    pub gradient_end_value: i32,
    pub gradient_boundary: i32,
    pub primary_channel_index: i32,
    pub has_joint_stereo_signs: bool,
    pub joint_stereo_signs: [i32; 30],
    pub band_extension_enabled: bool,
    pub has_extension_data: bool,
    pub bex_data_length: i32,
    pub bex_mode: i32,
    pub channels: [Channel; 2],
}

impl Block {
    fn new(config: &Config, block_index: i32) -> Block {
        let block_type = config.channel_config.types[block_index as usize];
        let channel_count = block_type_to_channel_count(block_type);
        let mdct_bits = config.frame_samples_power;
        Block {
            block_type,
            block_index,
            channel_count,
            first_in_superframe: false,
            reuse_band_params: false,
            band_count: 0,
            stereo_band: 0,
            extension_band: 0,
            quantization_unit_count: 0,
            stereo_quantization_unit: 0,
            extension_unit: 0,
            quantization_units_prev: 0,
            gradient: [0; 48],
            gradient_mode: 0,
            gradient_start_unit: 0,
            gradient_start_value: 0,
            gradient_end_unit: 0,
            gradient_end_value: 0,
            gradient_boundary: 0,
            primary_channel_index: 0,
            has_joint_stereo_signs: false,
            joint_stereo_signs: [0; 30],
            band_extension_enabled: false,
            has_extension_data: false,
            bex_data_length: 0,
            bex_mode: 0,
            channels: [Channel::new(0, mdct_bits), Channel::new(1, mdct_bits)],
        }
    }
}

/// A frame: a set of blocks whose channels together make the output frame.
pub(crate) struct Frame {
    pub index_in_superframe: i32,
    pub blocks: Vec<Block>,
}

impl Frame {
    pub(crate) fn new(config: &Config) -> Frame {
        let block_count = config.channel_config.block_count as i32;
        let blocks = (0..block_count).map(|i| Block::new(config, i)).collect();
        Frame { index_in_superframe: 0, blocks }
    }

    /// The (block, channel) index pairs in interleave order, matching the C
    /// `Frame.Channels` flattened pointer array.
    fn channel_order(&self) -> Vec<(usize, usize)> {
        let mut order = Vec::new();
        for (bi, block) in self.blocks.iter().enumerate() {
            for ci in 0..block.channel_count as usize {
                order.push((bi, ci));
            }
        }
        order
    }
}

fn block_type_to_channel_count(block_type: BlockType) -> i32 {
    match block_type {
        BlockType::Mono => 1,
        BlockType::Stereo => 2,
        BlockType::Lfe => 1,
    }
}

// --- top-level frame decode (decoder.c) ------------------------------------------

/// Decode one frame from `br` into the channels' `pcm`.
pub(crate) fn decode_frame(ctx: &DecodeCtx, frame: &mut Frame, br: &mut BitReader) -> Result<(), Error> {
    unpack_frame(ctx, frame, br)?;

    let block_count = ctx.config.channel_config.block_count as usize;
    for i in 0..block_count {
        let block = &mut frame.blocks[i];
        dequantize_spectra(block);
        apply_intensity_stereo(block);
        scale_spectrum_block(block);
        apply_band_extension(block);
        imdct_block(ctx, block);
    }
    Ok(())
}

/// Interleave the channels' `pcm` into signed-16 output (`PcmFloatToShort`).
pub(crate) fn pcm_float_to_short(config: &Config, frame: &Frame, out: &mut [i16]) {
    let channel_count = config.channel_count as usize;
    let sample_count = config.frame_samples as usize;
    let order = frame.channel_order();
    let mut i = 0;
    for smpl in 0..sample_count {
        for &(bi, ci) in order.iter().take(channel_count) {
            out[i] = clamp16(round_f(frame.blocks[bi].channels[ci].pcm[smpl]));
            i += 1;
        }
    }
}

fn dequantize_spectra(block: &mut Block) {
    for i in 0..block.channel_count as usize {
        let channel = &mut block.channels[i];
        channel.spectra = [0.0; MAX_FRAME_SAMPLES];
        for j in 0..channel.coded_quant_units {
            dequantize_quant_unit(channel, j);
        }
    }
}

fn dequantize_quant_unit(channel: &mut Channel, band: i32) {
    let sub_band_index = QUANT_UNIT_TO_COEFF_INDEX[band as usize];
    let sub_band_count = QUANT_UNIT_TO_COEFF_COUNT[band as usize];
    let step_size = QUANTIZER_STEP_SIZE[channel.precisions[band as usize] as usize];
    let step_size_fine = QUANTIZER_FINE_STEP_SIZE[channel.precisions_fine[band as usize] as usize];
    for sb in 0..sub_band_count {
        let idx = (sub_band_index + sb) as usize;
        let coarse = channel.quantized_spectra[idx] as f64 * step_size;
        let fine = channel.quantized_spectra_fine[idx] as f64 * step_size_fine;
        channel.spectra[idx] = coarse + fine;
    }
}

fn apply_intensity_stereo(block: &mut Block) {
    if block.block_type != BlockType::Stereo {
        return;
    }
    let total_units = block.quantization_unit_count;
    let stereo_units = block.stereo_quantization_unit;
    if stereo_units >= total_units {
        return;
    }
    let (source_idx, dest_idx) = if block.primary_channel_index == 0 { (0, 1) } else { (1, 0) };
    let (a, b) = block.channels.split_at_mut(1);
    let (source, dest): (&Channel, &mut Channel) = if source_idx == 0 {
        (&a[0], &mut b[0])
    } else {
        // source is index 1, dest index 0
        (&b[0], &mut a[0])
    };
    let _ = dest_idx;
    for i in stereo_units..total_units {
        let sign = block.joint_stereo_signs[i as usize];
        for sb in QUANT_UNIT_TO_COEFF_INDEX[i as usize]..QUANT_UNIT_TO_COEFF_INDEX[(i + 1) as usize] {
            let sb = sb as usize;
            if sign > 0 {
                dest.spectra[sb] = -source.spectra[sb];
            } else {
                dest.spectra[sb] = source.spectra[sb];
            }
        }
    }
}

fn scale_spectrum_block(block: &mut Block) {
    let quant_unit_count = block.quantization_unit_count;
    for i in 0..block.channel_count as usize {
        let channel = &mut block.channels[i];
        for u in 0..quant_unit_count {
            let scale = SPECTRUM_SCALE[channel.scale_factors[u as usize] as usize];
            for sb in QUANT_UNIT_TO_COEFF_INDEX[u as usize]..QUANT_UNIT_TO_COEFF_INDEX[(u + 1) as usize] {
                channel.spectra[sb as usize] *= scale;
            }
        }
    }
}

fn apply_band_extension(block: &mut Block) {
    if !block.band_extension_enabled || !block.has_extension_data {
        return;
    }
    let quant_unit_count = block.quantization_unit_count;
    for i in 0..block.channel_count as usize {
        apply_band_extension_channel(quant_unit_count, &mut block.channels[i]);
    }
}

fn imdct_block(ctx: &DecodeCtx, block: &mut Block) {
    for i in 0..block.channel_count as usize {
        let ch = &mut block.channels[i];
        run_imdct(
            ch.mdct_bits,
            ctx.trig,
            ctx.windows,
            &ch.spectra,
            &mut ch.pcm,
            &mut ch.imdct_previous,
        );
    }
}

// --- unpack (unpack.c) -----------------------------------------------------------

fn unpack_frame(ctx: &DecodeCtx, frame: &mut Frame, br: &mut BitReader) -> Result<(), Error> {
    let block_count = ctx.config.channel_config.block_count as usize;
    let frame_index = frame.index_in_superframe;

    for i in 0..block_count {
        unpack_block(ctx, &mut frame.blocks[i], br)?;
        if frame.blocks[i].first_in_superframe && frame_index != 0 {
            return Err(Error::UnpackSuperframeFlagInvalid);
        }
    }

    frame.index_in_superframe += 1;
    if frame.index_in_superframe == ctx.config.frames_per_superframe {
        frame.index_in_superframe = 0;
    }
    Ok(())
}

fn unpack_block(ctx: &DecodeCtx, block: &mut Block, br: &mut BitReader) -> Result<(), Error> {
    read_block_header(block, br)?;
    if block.block_type == BlockType::Lfe {
        unpack_lfe_block(block, br);
    } else {
        unpack_standard_block(ctx, block, br)?;
    }
    br.align_position(8);
    Ok(())
}

fn read_block_header(block: &mut Block, br: &mut BitReader) -> Result<(), Error> {
    block.first_in_superframe = br.read_int(1) == 0;
    block.reuse_band_params = br.read_int(1) != 0;
    if block.first_in_superframe && block.reuse_band_params && block.block_type != BlockType::Lfe {
        return Err(Error::UnpackReuseBandParamsInvalid);
    }
    Ok(())
}

fn unpack_standard_block(ctx: &DecodeCtx, block: &mut Block, br: &mut BitReader) -> Result<(), Error> {
    let config = ctx.config;
    if !block.reuse_band_params {
        read_band_params(config, block, br)?;
    }
    read_gradient_params(block, br)?;
    create_gradient(ctx.gradient_curves, block);
    read_stereo_params(block, br);
    read_extension_params(block, br)?;

    for i in 0..block.channel_count as usize {
        let sibling_sf: Option<[i32; 31]> =
            if i > 0 { Some(block.channels[0].scale_factors) } else { None };

        update_coded_units(
            block.primary_channel_index,
            block.quantization_unit_count,
            block.stereo_quantization_unit,
            &mut block.channels[i],
        );

        read_scale_factors(
            &mut block.channels[i],
            br,
            ctx.codebooks,
            block.first_in_superframe,
            block.quantization_units_prev,
            block.extension_unit,
            sibling_sf,
        )?;
        calculate_mask(block.quantization_unit_count, &mut block.channels[i]);
        calculate_precisions(
            block.gradient_mode,
            block.quantization_unit_count,
            &block.gradient,
            block.gradient_boundary,
            &mut block.channels[i],
        );
        calculate_spectrum_codebook_index(config.high_sample_rate, &mut block.channels[i]);
        read_spectra(&mut block.channels[i], br, ctx.codebooks, config.high_sample_rate)?;
        read_spectra_fine(&mut block.channels[i], br);
    }

    block.quantization_units_prev =
        if block.band_extension_enabled { block.extension_unit } else { block.quantization_unit_count };
    Ok(())
}

fn read_band_params(config: &Config, block: &mut Block, br: &mut BitReader) -> Result<(), Error> {
    let min_band_count = MIN_BAND_COUNT[config.high_sample_rate as usize];
    let max_extension_band = MAX_EXTENSION_BAND[config.high_sample_rate as usize];
    block.band_count = br.read_int(4) + min_band_count;
    block.quantization_unit_count = BAND_TO_QUANT_UNIT_COUNT[block.band_count as usize];

    if block.band_count > MAX_BAND_COUNT[config.sample_rate_index as usize] {
        return Err(Error::UnpackBandParamsInvalid);
    }

    if block.block_type == BlockType::Stereo {
        block.stereo_band = br.read_int(4);
        block.stereo_band += min_band_count;
        block.stereo_quantization_unit = BAND_TO_QUANT_UNIT_COUNT[block.stereo_band as usize];
    } else {
        block.stereo_band = block.band_count;
    }

    if block.stereo_band > block.band_count {
        return Err(Error::UnpackBandParamsInvalid);
    }

    block.band_extension_enabled = br.read_int(1) != 0;
    if block.band_extension_enabled {
        block.extension_band = br.read_int(4);
        block.extension_band += min_band_count;
        if block.extension_band < block.band_count || block.extension_band > max_extension_band {
            return Err(Error::UnpackBandParamsInvalid);
        }
        block.extension_unit = BAND_TO_QUANT_UNIT_COUNT[block.extension_band as usize];
    } else {
        block.extension_band = block.band_count;
        block.extension_unit = block.quantization_unit_count;
    }
    Ok(())
}

fn read_gradient_params(block: &mut Block, br: &mut BitReader) -> Result<(), Error> {
    block.gradient_mode = br.read_int(2);
    if block.gradient_mode > 0 {
        block.gradient_end_unit = 31;
        block.gradient_end_value = 31;
        block.gradient_start_unit = br.read_int(5);
        block.gradient_start_value = br.read_int(5);
    } else {
        block.gradient_start_unit = br.read_int(6);
        block.gradient_end_unit = br.read_int(6) + 1;
        block.gradient_start_value = br.read_int(5);
        block.gradient_end_value = br.read_int(5);
    }
    block.gradient_boundary = br.read_int(4);

    if block.gradient_boundary > block.quantization_unit_count {
        return Err(Error::UnpackGradBoundaryInvalid);
    }
    if block.gradient_start_unit < 0 || block.gradient_start_unit >= 48 {
        return Err(Error::UnpackGradStartUnitOob);
    }
    if block.gradient_end_unit < 0 || block.gradient_end_unit >= 48 {
        return Err(Error::UnpackGradEndUnitOob);
    }
    if block.gradient_start_unit > block.gradient_end_unit {
        return Err(Error::UnpackGradEndUnitInvalid);
    }
    if block.gradient_start_value < 0 || block.gradient_start_value >= 32 {
        return Err(Error::UnpackGradStartValueOob);
    }
    if block.gradient_end_value < 0 || block.gradient_end_value >= 32 {
        return Err(Error::UnpackGradEndValueOob);
    }
    Ok(())
}

fn read_stereo_params(block: &mut Block, br: &mut BitReader) {
    if block.block_type != BlockType::Stereo {
        return;
    }
    block.primary_channel_index = br.read_int(1);
    block.has_joint_stereo_signs = br.read_int(1) != 0;
    if block.has_joint_stereo_signs {
        for i in block.stereo_quantization_unit..block.quantization_unit_count {
            block.joint_stereo_signs[i as usize] = br.read_int(1);
        }
    } else {
        block.joint_stereo_signs = [0; 30];
    }
}

fn bex_read_header(channel: &mut Channel, br: &mut BitReader, bex_band: i32) {
    let bex_mode = br.read_int(2);
    channel.bex_mode = if bex_band > 2 { bex_mode } else { 4 };
    channel.bex_value_count = BEX_ENCODED_VALUE_COUNTS[channel.bex_mode as usize][bex_band as usize];
}

fn bex_read_data(channel: &mut Channel, br: &mut BitReader, bex_band: i32) {
    for i in 0..channel.bex_value_count as usize {
        let data_length = BEX_DATA_LENGTHS[channel.bex_mode as usize][bex_band as usize][i];
        channel.bex_values[i] = br.read_int(data_length);
    }
}

fn read_extension_params(block: &mut Block, br: &mut BitReader) -> Result<(), Error> {
    let mut bex_band = 0;
    if block.band_extension_enabled {
        bex_band = BEX_GROUP_INFO[(block.quantization_unit_count - 13) as usize].band_count;
        if block.block_type == BlockType::Stereo {
            bex_read_header(&mut block.channels[1], br, bex_band);
        } else {
            br.position += 1;
        }
    }
    block.has_extension_data = br.read_int(1) != 0;

    if !block.has_extension_data {
        return Ok(());
    }
    if !block.band_extension_enabled {
        block.bex_mode = br.read_int(2);
        block.bex_data_length = br.read_int(5);
        br.position += block.bex_data_length as usize;
        return Ok(());
    }

    bex_read_header(&mut block.channels[0], br, bex_band);

    block.bex_data_length = br.read_int(5);
    if block.bex_data_length == 0 {
        return Ok(());
    }
    let bex_data_end = br.position + block.bex_data_length as usize;

    bex_read_data(&mut block.channels[0], br, bex_band);
    if block.block_type == BlockType::Stereo {
        bex_read_data(&mut block.channels[1], br, bex_band);
    }

    if br.position > bex_data_end {
        return Err(Error::UnpackExtensionDataInvalid);
    }
    Ok(())
}

fn update_coded_units(
    primary_channel_index: i32,
    quantization_unit_count: i32,
    stereo_quantization_unit: i32,
    channel: &mut Channel,
) {
    channel.coded_quant_units = if primary_channel_index == channel.channel_index {
        quantization_unit_count
    } else {
        stereo_quantization_unit
    };
}

fn calculate_spectrum_codebook_index(high_sample_rate: bool, channel: &mut Channel) {
    channel.codebook_set = [0; 30];
    let quant_units = channel.coded_quant_units;
    if quant_units <= 1 {
        return;
    }
    if high_sample_rate {
        return;
    }

    let sf = &mut channel.scale_factors;
    // Temporarily make the last value non-special.
    let original_scale_tmp = sf[quant_units as usize];
    sf[quant_units as usize] = sf[(quant_units - 1) as usize];

    let mut avg = 0;
    if quant_units > 12 {
        for i in 0..12 {
            avg += sf[i];
        }
        avg = (avg + 6) / 12;
    }

    for i in 8..quant_units as usize {
        let prev_sf = sf[i - 1];
        let next_sf = sf[i + 1];
        let min_sf = prev_sf.min(next_sf);
        if sf[i] - min_sf >= 3 || sf[i] - prev_sf + sf[i] - next_sf >= 3 {
            channel.codebook_set[i] = 1;
        }
    }

    for i in 12..quant_units as usize {
        if channel.codebook_set[i] == 0 {
            let min_sf = channel.scale_factors[i - 1].min(channel.scale_factors[i + 1]);
            let adj = if QUANT_UNIT_TO_COEFF_COUNT[i] == 16 { 1 } else { 0 };
            if channel.scale_factors[i] - min_sf >= 2 && channel.scale_factors[i] >= avg - adj {
                channel.codebook_set[i] = 1;
            }
        }
    }

    channel.scale_factors[quant_units as usize] = original_scale_tmp;
}

fn read_spectra(
    channel: &mut Channel,
    br: &mut BitReader,
    codebooks: &Codebooks,
    high_sample_rate: bool,
) -> Result<(), Error> {
    let mut values = [0i32; 16];
    channel.quantized_spectra = [0; MAX_FRAME_SAMPLES];
    let max_huff_precision = MAX_HUFF_PRECISION[high_sample_rate as usize];

    for i in 0..channel.coded_quant_units as usize {
        let subband_count = QUANT_UNIT_TO_COEFF_COUNT[i];
        let precision = channel.precisions[i] + 1;
        if precision <= max_huff_precision {
            let set = channel.codebook_set[i] as usize;
            let cb_index = QUANT_UNIT_TO_CODEBOOK_INDEX[i] as usize;
            let huff = codebooks.spectrum[set][precision as usize][cb_index]
                .as_ref()
                .expect("spectrum codebook present");
            let group_count = subband_count >> huff.value_count_power;
            for j in 0..group_count as usize {
                values[j] = read_huffman_value(huff, br, false);
            }
            decode_huffman_values(
                &mut channel.quantized_spectra,
                QUANT_UNIT_TO_COEFF_INDEX[i] as usize,
                subband_count,
                huff,
                &values,
            );
        } else {
            let subband_index = QUANT_UNIT_TO_COEFF_INDEX[i];
            for j in subband_index..QUANT_UNIT_TO_COEFF_INDEX[i + 1] {
                channel.quantized_spectra[j as usize] = br.read_signed_int(precision);
            }
        }
    }
    Ok(())
}

fn read_spectra_fine(channel: &mut Channel, br: &mut BitReader) {
    channel.quantized_spectra_fine = [0; MAX_FRAME_SAMPLES];
    for i in 0..channel.coded_quant_units as usize {
        if channel.precisions_fine[i] > 0 {
            let overflow_bits = channel.precisions_fine[i] + 1;
            let start_subband = QUANT_UNIT_TO_COEFF_INDEX[i];
            let end_subband = QUANT_UNIT_TO_COEFF_INDEX[i + 1];
            for j in start_subband..end_subband {
                channel.quantized_spectra_fine[j as usize] = br.read_signed_int(overflow_bits);
            }
        }
    }
}

// --- LFE block (unpack.c) --------------------------------------------------------

fn unpack_lfe_block(block: &mut Block, br: &mut BitReader) {
    block.quantization_unit_count = 2;
    let reuse = block.reuse_band_params;
    let channel = &mut block.channels[0];

    // scale factors
    channel.scale_factors = [0; 31];
    for i in 0..2 {
        channel.scale_factors[i] = br.read_int(5);
    }
    // precision
    let precision = if reuse { 8 } else { 4 };
    for i in 0..2 {
        channel.precisions[i] = precision;
        channel.precisions_fine[i] = 0;
    }
    channel.coded_quant_units = block.quantization_unit_count;

    // spectra
    channel.quantized_spectra = [0; MAX_FRAME_SAMPLES];
    for i in 0..channel.coded_quant_units as usize {
        if channel.precisions[i] <= 0 {
            continue;
        }
        let precision = channel.precisions[i] + 1;
        for j in QUANT_UNIT_TO_COEFF_INDEX[i]..QUANT_UNIT_TO_COEFF_INDEX[i + 1] {
            channel.quantized_spectra[j as usize] = br.read_signed_int(precision);
        }
    }
}

// --- scale factors (scale_factors.c) ---------------------------------------------

fn read_scale_factors(
    channel: &mut Channel,
    br: &mut BitReader,
    codebooks: &Codebooks,
    first_in_superframe: bool,
    quantization_units_prev: i32,
    extension_unit: i32,
    sibling_sf: Option<[i32; 31]>,
) -> Result<(), Error> {
    channel.scale_factors = [0; 31];
    channel.scale_factor_coding_mode = br.read_int(2);
    let mode = channel.scale_factor_coding_mode;

    if channel.channel_index == 0 {
        match mode {
            0 => read_vlc_delta_offset(channel, br, codebooks, extension_unit),
            1 => read_clc_offset(channel, br, extension_unit),
            2 => {
                if first_in_superframe {
                    return Err(Error::UnpackScaleFactorModeInvalid);
                }
                let base = channel.scale_factors_prev;
                read_vlc_distance_to_baseline(channel, br, codebooks, extension_unit, &base, quantization_units_prev);
            }
            3 => {
                if first_in_superframe {
                    return Err(Error::UnpackScaleFactorModeInvalid);
                }
                let base = channel.scale_factors_prev;
                read_vlc_delta_offset_with_baseline(channel, br, codebooks, extension_unit, &base, quantization_units_prev);
            }
            _ => {}
        }
    } else {
        let sibling = sibling_sf.expect("second channel needs sibling scale factors");
        match mode {
            0 => read_vlc_delta_offset(channel, br, codebooks, extension_unit),
            1 => read_vlc_distance_to_baseline(channel, br, codebooks, extension_unit, &sibling, extension_unit),
            2 => read_vlc_delta_offset_with_baseline(channel, br, codebooks, extension_unit, &sibling, extension_unit),
            3 => {
                if first_in_superframe {
                    return Err(Error::UnpackScaleFactorModeInvalid);
                }
                let base = channel.scale_factors_prev;
                read_vlc_distance_to_baseline(channel, br, codebooks, extension_unit, &base, quantization_units_prev);
            }
            _ => {}
        }
    }

    for i in 0..extension_unit as usize {
        if channel.scale_factors[i] < 0 || channel.scale_factors[i] > 31 {
            return Err(Error::UnpackScaleFactorOob);
        }
    }

    channel.scale_factors_prev = channel.scale_factors;
    Ok(())
}

fn read_clc_offset(channel: &mut Channel, br: &mut BitReader, extension_unit: i32) {
    let max_bits = 5;
    let bit_length = br.read_int(2) + 2;
    let base_value = if bit_length < max_bits { br.read_int(max_bits) } else { 0 };
    for i in 0..extension_unit as usize {
        channel.scale_factors[i] = br.read_int(bit_length) + base_value;
    }
}

fn read_vlc_delta_offset(channel: &mut Channel, br: &mut BitReader, codebooks: &Codebooks, extension_unit: i32) {
    let weight_index = br.read_int(3);
    let weights = &SCALE_FACTOR_WEIGHTS[weight_index as usize];
    let base_value = br.read_int(5);
    let bit_length = br.read_int(2) + 3;
    let codebook = codebooks.sf_unsigned[bit_length as usize].as_ref().expect("sf codebook");

    channel.scale_factors[0] = br.read_int(bit_length);
    for i in 1..extension_unit as usize {
        let delta = read_huffman_value(codebook, br, false);
        channel.scale_factors[i] = (channel.scale_factors[i - 1] + delta) & (codebook.value_max - 1);
    }
    for i in 0..extension_unit as usize {
        channel.scale_factors[i] += base_value - weights[i] as i32;
    }
}

fn read_vlc_distance_to_baseline(
    channel: &mut Channel,
    br: &mut BitReader,
    codebooks: &Codebooks,
    extension_unit: i32,
    baseline: &[i32],
    baseline_length: i32,
) {
    let bit_length = br.read_int(2) + 2;
    let codebook = codebooks.sf_signed[bit_length as usize].as_ref().expect("sf codebook");
    let unit_count = extension_unit.min(baseline_length);
    for i in 0..unit_count as usize {
        let distance = read_huffman_value(codebook, br, true);
        channel.scale_factors[i] = (baseline[i] + distance) & 31;
    }
    for i in unit_count as usize..extension_unit as usize {
        channel.scale_factors[i] = br.read_int(5);
    }
}

fn read_vlc_delta_offset_with_baseline(
    channel: &mut Channel,
    br: &mut BitReader,
    codebooks: &Codebooks,
    extension_unit: i32,
    baseline: &[i32],
    baseline_length: i32,
) {
    let base_value = br.read_offset_binary(5);
    let bit_length = br.read_int(2) + 1;
    let codebook = codebooks.sf_unsigned[bit_length as usize].as_ref().expect("sf codebook");
    let unit_count = extension_unit.min(baseline_length);

    channel.scale_factors[0] = br.read_int(bit_length);
    for i in 1..unit_count as usize {
        let delta = read_huffman_value(codebook, br, false);
        channel.scale_factors[i] = (channel.scale_factors[i - 1] + delta) & (codebook.value_max - 1);
    }
    for i in 0..unit_count as usize {
        channel.scale_factors[i] += base_value + baseline[i];
    }
    for i in unit_count as usize..extension_unit as usize {
        channel.scale_factors[i] = br.read_int(5);
    }
}

// --- bit allocation (bit_allocation.c) -------------------------------------------

const BASE_CURVE: [u8; 48] = [
    1, 1, 1, 1, 2, 2, 2, 2, 3, 3, 3, 4, 4, 5, 5, 6, 7, 8, 9, 10, 11, 12, 13, 15, 16, 18, 19, 20, 21,
    22, 23, 24, 25, 26, 26, 27, 27, 28, 28, 28, 29, 29, 29, 29, 30, 30, 30, 30,
];

pub(crate) fn generate_gradient_curves() -> Box<GradientCurves> {
    let base_length = BASE_CURVE.len() as i32;
    let mut curves = Box::new([[0u8; 48]; 48]);
    for length in 1..=base_length {
        for i in 0..length {
            curves[(length - 1) as usize][i as usize] =
                BASE_CURVE[(i * base_length / length) as usize];
        }
    }
    curves
}

fn create_gradient(gradient_curves: &GradientCurves, block: &mut Block) {
    let value_count = block.gradient_end_value - block.gradient_start_value;
    let unit_count = block.gradient_end_unit - block.gradient_start_unit;

    for i in 0..block.gradient_end_unit {
        block.gradient[i as usize] = block.gradient_start_value;
    }
    for i in block.gradient_end_unit..=block.quantization_unit_count {
        block.gradient[i as usize] = block.gradient_end_value;
    }
    if unit_count <= 0 {
        return;
    }
    if value_count == 0 {
        return;
    }

    let curve = &gradient_curves[(unit_count - 1) as usize];
    if value_count <= 0 {
        let scale = (-value_count - 1) as f64 / 31.0;
        let base_val = block.gradient_start_value - 1;
        for i in block.gradient_start_unit..block.gradient_end_unit {
            block.gradient[i as usize] =
                base_val - (curve[(i - block.gradient_start_unit) as usize] as f64 * scale) as i32;
        }
    } else {
        let scale = (value_count - 1) as f64 / 31.0;
        let base_val = block.gradient_start_value + 1;
        for i in block.gradient_start_unit..block.gradient_end_unit {
            block.gradient[i as usize] =
                base_val + (curve[(i - block.gradient_start_unit) as usize] as f64 * scale) as i32;
        }
    }
}

fn calculate_mask(quantization_unit_count: i32, channel: &mut Channel) {
    channel.precision_mask = [0; 30];
    for i in 1..quantization_unit_count as usize {
        let delta = channel.scale_factors[i] - channel.scale_factors[i - 1];
        if delta > 1 {
            channel.precision_mask[i] += (delta - 1).min(5);
        } else if delta < -1 {
            channel.precision_mask[i - 1] += (delta * -1 - 1).min(5);
        }
    }
}

fn calculate_precisions(
    gradient_mode: i32,
    quantization_unit_count: i32,
    gradient: &[i32; 48],
    gradient_boundary: i32,
    channel: &mut Channel,
) {
    if gradient_mode != 0 {
        for i in 0..quantization_unit_count as usize {
            channel.precisions[i] =
                channel.scale_factors[i] + channel.precision_mask[i] - gradient[i];
            if channel.precisions[i] > 0 {
                match gradient_mode {
                    1 => channel.precisions[i] /= 2,
                    2 => channel.precisions[i] = 3 * channel.precisions[i] / 8,
                    3 => channel.precisions[i] /= 4,
                    _ => {}
                }
            }
        }
    } else {
        for i in 0..quantization_unit_count as usize {
            channel.precisions[i] = channel.scale_factors[i] - gradient[i];
        }
    }

    for i in 0..quantization_unit_count as usize {
        if channel.precisions[i] < 1 {
            channel.precisions[i] = 1;
        }
    }
    for i in 0..gradient_boundary as usize {
        channel.precisions[i] += 1;
    }
    for i in 0..quantization_unit_count as usize {
        channel.precisions_fine[i] = 0;
        if channel.precisions[i] > 15 {
            channel.precisions_fine[i] = channel.precisions[i] - 15;
            channel.precisions[i] = 15;
        }
    }
}

// --- output helpers (utility.c) --------------------------------------------------

fn round_f(x: f64) -> i32 {
    let x = x + 0.5;
    let t = x as i32;
    t - if x < t as f64 { 1 } else { 0 }
}

fn clamp16(value: i32) -> i16 {
    value.clamp(-32768, 32767) as i16
}
