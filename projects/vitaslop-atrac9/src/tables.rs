//! Constant tables from LibAtrac9 `tables.c`, plus the trig/window/shuffle tables
//! the C generates into globals at init. In Rust those generated tables live in
//! [`Trig`] and [`Windows`] owned by the decoder rather than mutable statics, so
//! the decoder is self-contained and thread-safe.

use crate::decoder::BlockType;

/// One channel configuration: how a frame's channels split into blocks.
#[derive(Clone, Copy)]
pub(crate) struct ChannelConfig {
    pub block_count: u8,
    pub channel_count: u8,
    pub types: [BlockType; 5],
}

pub(crate) const CHANNEL_CONFIGS: [ChannelConfig; 6] = {
    use BlockType::{Lfe, Mono, Stereo};
    [
        ChannelConfig { block_count: 1, channel_count: 1, types: [Mono, Mono, Mono, Mono, Mono] },
        ChannelConfig { block_count: 2, channel_count: 2, types: [Mono, Mono, Mono, Mono, Mono] },
        ChannelConfig { block_count: 1, channel_count: 2, types: [Stereo, Mono, Mono, Mono, Mono] },
        ChannelConfig { block_count: 4, channel_count: 6, types: [Stereo, Mono, Lfe, Stereo, Mono] },
        ChannelConfig { block_count: 5, channel_count: 8, types: [Stereo, Mono, Lfe, Stereo, Stereo] },
        ChannelConfig { block_count: 2, channel_count: 4, types: [Stereo, Stereo, Mono, Mono, Mono] },
    ]
};

pub(crate) const MAX_HUFF_PRECISION: [i32; 2] = [7, 1];
pub(crate) const MIN_BAND_COUNT: [i32; 2] = [3, 1];
pub(crate) const MAX_EXTENSION_BAND: [i32; 2] = [18, 16];

pub(crate) const SAMPLING_RATE_INDEX_TO_FRAME_SAMPLES_POWER: [i32; 16] =
    [6, 6, 7, 7, 7, 8, 8, 8, 6, 6, 7, 7, 7, 8, 8, 8];

pub(crate) const MAX_BAND_COUNT: [i32; 16] =
    [8, 8, 12, 12, 12, 18, 18, 18, 8, 8, 12, 12, 12, 16, 16, 16];

pub(crate) const BAND_TO_QUANT_UNIT_COUNT: [i32; 19] =
    [0, 4, 8, 10, 12, 13, 14, 15, 16, 18, 20, 21, 22, 23, 24, 25, 26, 28, 30];

pub(crate) const QUANT_UNIT_TO_COEFF_COUNT: [i32; 30] = [
    2, 2, 2, 2, 2, 2, 2, 2, 4, 4, 4, 4, 8, 8, 8, 8, 8, 8, 8, 8, 16, 16, 16, 16, 16, 16, 16, 16, 16,
    16,
];

pub(crate) const QUANT_UNIT_TO_COEFF_INDEX: [i32; 31] = [
    0, 2, 4, 6, 8, 10, 12, 14, 16, 20, 24, 28, 32, 40, 48, 56, 64, 72, 80, 88, 96, 112, 128, 144,
    160, 176, 192, 208, 224, 240, 256,
];

pub(crate) const QUANT_UNIT_TO_CODEBOOK_INDEX: [i32; 30] = [
    0, 0, 0, 0, 0, 0, 0, 0, 1, 1, 1, 1, 2, 2, 2, 2, 2, 2, 2, 2, 3, 3, 3, 3, 3, 3, 3, 3, 3, 3,
];

pub(crate) const SAMPLE_RATES: [i32; 16] = [
    11025, 12000, 16000, 22050, 24000, 32000, 44100, 48000, 44100, 48000, 64000, 88200, 96000,
    128000, 176400, 192000,
];

pub(crate) const SCALE_FACTOR_WEIGHTS: [[u8; 32]; 8] = [
    [0, 0, 0, 1, 1, 2, 2, 2, 2, 2, 2, 3, 2, 3, 3, 4, 4, 4, 4, 4, 4, 5, 5, 6, 6, 7, 7, 8, 10, 12, 12, 12],
    [3, 2, 2, 1, 1, 1, 1, 1, 0, 1, 1, 1, 0, 0, 0, 1, 0, 1, 1, 1, 1, 1, 1, 2, 3, 3, 4, 5, 7, 10, 10, 10],
    [0, 2, 4, 5, 6, 6, 6, 6, 6, 6, 6, 6, 6, 6, 6, 6, 6, 6, 6, 6, 6, 6, 6, 7, 7, 7, 7, 8, 9, 12, 12, 12],
    [0, 1, 1, 2, 2, 2, 3, 3, 3, 3, 3, 4, 4, 4, 5, 5, 5, 6, 6, 6, 6, 7, 8, 8, 10, 11, 11, 12, 13, 13, 13, 13],
    [0, 2, 2, 3, 3, 4, 4, 5, 4, 5, 5, 5, 5, 6, 7, 8, 8, 8, 8, 9, 9, 9, 10, 10, 11, 12, 12, 13, 13, 14, 14, 14],
    [1, 1, 0, 0, 0, 0, 1, 0, 0, 1, 1, 1, 1, 1, 2, 2, 2, 2, 2, 3, 3, 3, 4, 4, 5, 6, 7, 7, 9, 11, 11, 11],
    [0, 5, 8, 10, 11, 11, 12, 12, 12, 13, 13, 13, 13, 13, 13, 13, 13, 13, 13, 13, 13, 13, 13, 13, 12, 12, 12, 12, 13, 15, 15, 15],
    [0, 2, 3, 4, 5, 6, 6, 7, 7, 8, 8, 8, 9, 9, 10, 10, 10, 11, 11, 11, 11, 11, 11, 12, 12, 12, 12, 13, 13, 15, 15, 15],
];

pub(crate) const SPECTRUM_SCALE: [f64; 32] = [
    3.0517578125e-5, 6.1035156250e-5, 1.2207031250e-4, 2.4414062500e-4, 4.8828125000e-4,
    9.7656250000e-4, 1.9531250000e-3, 3.9062500000e-3, 7.8125000000e-3, 1.5625000000e-2,
    3.1250000000e-2, 6.2500000000e-2, 1.2500000000e-1, 2.5000000000e-1, 5.0000000000e-1,
    1.0000000000e+0, 2.0000000000e+0, 4.0000000000e+0, 8.0000000000e+0, 1.6000000000e+1,
    3.2000000000e+1, 6.4000000000e+1, 1.2800000000e+2, 2.5600000000e+2, 5.1200000000e+2,
    1.0240000000e+3, 2.0480000000e+3, 4.0960000000e+3, 8.1920000000e+3, 1.6384000000e+4,
    3.2768000000e+4, 6.5536000000e+4,
];

pub(crate) const QUANTIZER_STEP_SIZE: [f64; 16] = [
    2.0000000000000000e+0, 6.6666666666666663e-1, 2.8571428571428570e-1, 1.3333333333333333e-1,
    6.4516129032258063e-2, 3.1746031746031744e-2, 1.5748031496062992e-2, 7.8431372549019607e-3,
    3.9138943248532287e-3, 1.9550342130987292e-3, 9.7703957010258913e-4, 4.8840048840048840e-4,
    2.4417043096081065e-4, 1.2207776353537203e-4, 6.1037018951994385e-5, 3.0518043793392844e-5,
];

pub(crate) const QUANTIZER_FINE_STEP_SIZE: [f64; 16] = [
    3.0518043793392844e-05, 1.0172681264464281e-05, 4.3597205419132631e-06, 2.0345362528928561e-06,
    9.8445302559331759e-07, 4.8441339354591809e-07, 2.4029955742829012e-07, 1.1967860311134448e-07,
    5.9722199204291275e-08, 2.9831909866464167e-08, 1.4908668194134265e-08, 7.4525137468602791e-09,
    3.7258019525568114e-09, 1.8627872668859698e-09, 9.3136520869755679e-10, 4.6567549848772173e-10,
];

/// Per-size sine/cosine tables for the DCT-IV, generated for size-bits 0..=8.
/// `sin[bits][i]` = sin(pi * (4i+1) / (4 * 2^bits)); likewise cos.
pub(crate) struct Trig {
    pub sin: [[f64; 256]; 9],
    pub cos: [[f64; 256]; 9],
    pub shuffle: [[usize; 256]; 9],
}

impl Trig {
    pub(crate) fn generate() -> Box<Trig> {
        // Boxed: three 256*9 tables are large for the stack.
        let mut t = Box::new(Trig {
            sin: [[0.0; 256]; 9],
            cos: [[0.0; 256]; 9],
            shuffle: [[0; 256]; 9],
        });
        for size_bits in 0..9 {
            let size = 1usize << size_bits;
            for i in 0..size {
                let value =
                    std::f64::consts::PI * (4 * i + 1) as f64 / (4 * size) as f64;
                t.sin[size_bits][i] = value.sin();
                t.cos[size_bits][i] = value.cos();
                t.shuffle[size_bits][i] = bit_reverse32((i ^ (i / 2)) as u32, size_bits as i32) as usize;
            }
        }
        t
    }
}

/// Forward/inverse MDCT windows for frame-size powers 6, 7, 8 (indexed `power-6`).
pub(crate) struct Windows {
    pub imdct: [[f64; 256]; 3],
}

impl Windows {
    pub(crate) fn generate() -> Box<Windows> {
        let mut mdct = [[0.0f64; 256]; 3];
        let mut w = Box::new(Windows { imdct: [[0.0; 256]; 3] });
        for power in 6..=8usize {
            let frame_size = 1usize << power;
            let idx = power - 6;
            for i in 0..frame_size {
                mdct[idx][i] = ((((i as f64 + 0.5) / frame_size as f64) - 0.5)
                    * std::f64::consts::PI)
                    .sin()
                    * 0.5
                    + 0.5;
            }
            for i in 0..frame_size {
                let m = &mdct[idx];
                w.imdct[idx][i] =
                    m[i] / (m[frame_size - 1 - i] * m[frame_size - 1 - i] + m[i] * m[i]);
            }
        }
        w
    }
}

/// Reverse the low `bit_count` bits of `value` (LibAtrac9 `BitReverse32`).
fn bit_reverse32(mut value: u32, bit_count: i32) -> u32 {
    value = ((value & 0xaaaa_aaaa) >> 1) | ((value & 0x5555_5555) << 1);
    value = ((value & 0xcccc_cccc) >> 2) | ((value & 0x3333_3333) << 2);
    value = ((value & 0xf0f0_f0f0) >> 4) | ((value & 0x0f0f_0f0f) << 4);
    value = ((value & 0xff00_ff00) >> 8) | ((value & 0x00ff_00ff) << 8);
    value = (value >> 16) | (value << 16);
    // The C relies on x86 masking the shift count to 5 bits, so a bit_count of 0
    // (size-1 tables) shifts by 0, not 32. Replicate that masking.
    value >> ((32 - bit_count) & 31)
}
