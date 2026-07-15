//! Inverse MDCT, ported from LibAtrac9 `imdct.c`. A windowed overlap-add around a
//! decimation-in-time DCT-IV. The trig and shuffle tables come from [`Trig`], the
//! synthesis window from [`Windows`], and each channel keeps its own overlap
//! history in `previous`.

use crate::tables::{Trig, Windows};

const MAX_FRAME_SAMPLES: usize = 256;

/// Run one inverse MDCT: transform `input` (spectra) into `output` (time-domain
/// PCM), folding in and updating the per-channel overlap `previous`.
pub(crate) fn run_imdct(
    mdct_bits: i32,
    trig: &Trig,
    windows: &Windows,
    input: &[f64],
    output: &mut [f64],
    previous: &mut [f64],
) {
    let size = 1usize << mdct_bits;
    let half = size / 2;
    let mut dct_out = [0.0f64; MAX_FRAME_SAMPLES];
    let window = &windows.imdct[(mdct_bits - 6) as usize];

    dct4(mdct_bits, trig, input, &mut dct_out);

    for i in 0..half {
        output[i] = window[i] * dct_out[i + half] + previous[i];
        output[i + half] = window[i + half] * -dct_out[size - 1 - i] - previous[i + half];
        previous[i] = window[size - 1 - i] * -dct_out[half - i - 1];
        previous[i + half] = window[half - i - 1] * dct_out[i];
    }
}

fn dct4(mdct_bits: i32, trig: &Trig, input: &[f64], output: &mut [f64]) {
    let mdct_size = 1usize << mdct_bits;
    let shuffle_table = &trig.shuffle[mdct_bits as usize];
    let mut sin_table = &trig.sin[mdct_bits as usize];
    let mut cos_table = &trig.cos[mdct_bits as usize];
    let mut dct_temp = [0.0f64; MAX_FRAME_SAMPLES];

    let size = mdct_size;
    let last_index = size - 1;
    let half_size = size / 2;

    for i in 0..half_size {
        let i2 = i * 2;
        let a = input[i2];
        let b = input[last_index - i2];
        let sin = sin_table[i];
        let cos = cos_table[i];
        dct_temp[i2] = a * cos + b * sin;
        dct_temp[i2 + 1] = a * sin - b * cos;
    }
    let stage_count = mdct_bits - 1;

    for stage in 0..stage_count {
        let block_count = 1usize << stage;
        let block_size_bits = stage_count - stage;
        let block_half_size_bits = block_size_bits - 1;
        let block_size = 1usize << block_size_bits;
        let block_half_size = 1usize << block_half_size_bits;
        sin_table = &trig.sin[block_half_size_bits as usize];
        cos_table = &trig.cos[block_half_size_bits as usize];

        for block in 0..block_count {
            for i in 0..block_half_size {
                let front_pos = (block * block_size + i) * 2;
                let back_pos = front_pos + block_size;
                let a = dct_temp[front_pos] - dct_temp[back_pos];
                let b = dct_temp[front_pos + 1] - dct_temp[back_pos + 1];
                let sin = sin_table[i];
                let cos = cos_table[i];
                dct_temp[front_pos] += dct_temp[back_pos];
                dct_temp[front_pos + 1] += dct_temp[back_pos + 1];
                dct_temp[back_pos] = a * cos + b * sin;
                dct_temp[back_pos + 1] = a * sin - b * cos;
            }
        }
    }

    for i in 0..mdct_size {
        output[i] = dct_temp[shuffle_table[i]];
    }
}
