//! Huffman codebooks, ported from LibAtrac9 `huffCodes.c`. The large bit/code
//! tables live in [`crate::generated_huffman`] (a verbatim machine conversion of
//! the C arrays). Here we wire them into codebooks, build each one's flat lookup
//! table at init, and provide the two decode primitives.

use crate::bit_reader::{sign_extend32, BitReader};
use crate::generated_huffman as g;

/// One Huffman codebook plus its decode lookup. `lookup[peek(max_bit_size)]` gives
/// the decoded symbol index; `bits[symbol]` is that symbol's true code length.
pub(crate) struct HuffmanCodebook {
    bits: &'static [u8],
    /// Flat lookup of size `1 << max_bit_size`, symbol index per padded code.
    lookup: Vec<u8>,
    pub value_count: i32,
    pub value_count_power: i32,
    pub value_bits: i32,
    pub value_max: i32,
    max_bit_size: i32,
}

impl HuffmanCodebook {
    fn new(
        bits: &'static [u8],
        codes: &'static [u16],
        value_count: i32,
        value_count_power: i32,
        value_bits: i32,
        value_max: i32,
        max_bit_size: i32,
    ) -> HuffmanCodebook {
        let huff_length = bits.len();
        let mut lookup = vec![0u8; 1usize << max_bit_size];
        for i in 0..huff_length {
            if bits[i] == 0 {
                continue;
            }
            let unused_bits = max_bit_size - bits[i] as i32;
            let start = (codes[i] as usize) << unused_bits;
            let length = 1usize << unused_bits;
            for j in start..start + length {
                lookup[j] = i as u8;
            }
        }
        HuffmanCodebook {
            bits,
            lookup,
            value_count,
            value_count_power,
            value_bits,
            value_max,
            max_bit_size,
        }
    }
}

/// Read one Huffman-coded value, advancing the reader by the symbol's code length.
pub(crate) fn read_huffman_value(huff: &HuffmanCodebook, br: &mut BitReader, is_signed: bool) -> i32 {
    let code = br.peek_int(huff.max_bit_size);
    let value = huff.lookup[code as usize];
    let bits = huff.bits[value as usize];
    br.position += bits as usize;
    if is_signed {
        sign_extend32(value as i32, huff.value_bits)
    } else {
        value as i32
    }
}

/// Expand `values` (packed groups of `value_count` sub-values each `value_bits`
/// wide) into `spectrum` starting at `index`.
pub(crate) fn decode_huffman_values(
    spectrum: &mut [i32],
    mut index: usize,
    band_count: i32,
    huff: &HuffmanCodebook,
    values: &[i32],
) {
    let value_count = band_count >> huff.value_count_power;
    let mask = (1 << huff.value_bits) - 1;
    for i in 0..value_count as usize {
        let mut value = values[i];
        for _ in 0..huff.value_count {
            spectrum[index] = sign_extend32(value & mask, huff.value_bits);
            index += 1;
            value >>= huff.value_bits;
        }
    }
}

/// The full set of decoder codebooks, built once at init.
pub(crate) struct Codebooks {
    /// Unsigned scale-factor books, indexed by bit length (0 unused).
    pub sf_unsigned: [Option<HuffmanCodebook>; 7],
    /// Signed scale-factor books, indexed by bit length (0,1 unused).
    pub sf_signed: [Option<HuffmanCodebook>; 6],
    /// Spectrum books: `[codebookSet][precision][codebookIndex]`.
    pub spectrum: [[[Option<HuffmanCodebook>; 4]; 8]; 2],
}

impl Codebooks {
    pub(crate) fn new() -> Codebooks {
        // Helper for a present codebook.
        fn cb(
            bits: &'static [u8],
            codes: &'static [u16],
            vc: i32,
            vcp: i32,
            vb: i32,
            vmax: i32,
            mbs: i32,
        ) -> Option<HuffmanCodebook> {
            Some(HuffmanCodebook::new(bits, codes, vc, vcp, vb, vmax, mbs))
        }

        let sf_unsigned = [
            None,
            cb(&g::SCALE_FACTORS_A_1_BITS, &g::SCALE_FACTORS_A_1_CODES, 1, 0, 1, 2, 1),
            cb(&g::SCALE_FACTORS_A_2_BITS, &g::SCALE_FACTORS_A_2_CODES, 1, 0, 2, 4, 3),
            cb(&g::SCALE_FACTORS_A_3_BITS, &g::SCALE_FACTORS_A_3_CODES, 1, 0, 3, 8, 6),
            cb(&g::SCALE_FACTORS_A_4_BITS, &g::SCALE_FACTORS_A_4_CODES, 1, 0, 4, 16, 8),
            cb(&g::SCALE_FACTORS_A_5_BITS, &g::SCALE_FACTORS_A_5_CODES, 1, 0, 5, 32, 8),
            cb(&g::SCALE_FACTORS_A_6_BITS, &g::SCALE_FACTORS_A_6_CODES, 1, 0, 6, 64, 8),
        ];

        let sf_signed = [
            None,
            None,
            cb(&g::SCALE_FACTORS_B_2_BITS, &g::SCALE_FACTORS_B_2_CODES, 1, 0, 2, 4, 2),
            cb(&g::SCALE_FACTORS_B_3_BITS, &g::SCALE_FACTORS_B_3_CODES, 1, 0, 3, 8, 6),
            cb(&g::SCALE_FACTORS_B_4_BITS, &g::SCALE_FACTORS_B_4_CODES, 1, 0, 4, 16, 8),
            cb(&g::SCALE_FACTORS_B_5_BITS, &g::SCALE_FACTORS_B_5_CODES, 1, 0, 5, 32, 8),
        ];

        // set 0 (A books)
        let spectrum_a: [[Option<HuffmanCodebook>; 4]; 8] = [
            [None, None, None, None],
            [None, None, None, None],
            [
                cb(&g::SPECTRUM_A_21_BITS, &g::SPECTRUM_A_21_CODES, 2, 1, 2, 4, 3),
                cb(&g::SPECTRUM_A_22_BITS, &g::SPECTRUM_A_22_CODES, 4, 2, 2, 4, 8),
                cb(&g::SPECTRUM_A_23_BITS, &g::SPECTRUM_A_23_CODES, 4, 2, 2, 4, 9),
                cb(&g::SPECTRUM_A_24_BITS, &g::SPECTRUM_A_24_CODES, 4, 2, 2, 4, 10),
            ],
            [
                cb(&g::SPECTRUM_A_31_BITS, &g::SPECTRUM_A_31_CODES, 2, 1, 3, 8, 7),
                cb(&g::SPECTRUM_A_32_BITS, &g::SPECTRUM_A_32_CODES, 2, 1, 3, 8, 7),
                cb(&g::SPECTRUM_A_33_BITS, &g::SPECTRUM_A_33_CODES, 2, 1, 3, 8, 8),
                cb(&g::SPECTRUM_A_34_BITS, &g::SPECTRUM_A_34_CODES, 2, 1, 3, 8, 10),
            ],
            [
                cb(&g::SPECTRUM_A_41_BITS, &g::SPECTRUM_A_41_CODES, 2, 1, 4, 16, 9),
                cb(&g::SPECTRUM_A_42_BITS, &g::SPECTRUM_A_42_CODES, 2, 1, 4, 16, 10),
                cb(&g::SPECTRUM_A_43_BITS, &g::SPECTRUM_A_43_CODES, 2, 1, 4, 16, 10),
                cb(&g::SPECTRUM_A_44_BITS, &g::SPECTRUM_A_44_CODES, 2, 1, 4, 16, 10),
            ],
            [
                cb(&g::SPECTRUM_A_51_BITS, &g::SPECTRUM_A_51_CODES, 1, 0, 5, 32, 6),
                cb(&g::SPECTRUM_A_52_BITS, &g::SPECTRUM_A_52_CODES, 1, 0, 5, 32, 6),
                cb(&g::SPECTRUM_A_53_BITS, &g::SPECTRUM_A_53_CODES, 1, 0, 5, 32, 7),
                cb(&g::SPECTRUM_A_54_BITS, &g::SPECTRUM_A_54_CODES, 1, 0, 5, 32, 8),
            ],
            [
                cb(&g::SPECTRUM_A_61_BITS, &g::SPECTRUM_A_61_CODES, 1, 0, 6, 64, 7),
                cb(&g::SPECTRUM_A_62_BITS, &g::SPECTRUM_A_62_CODES, 1, 0, 6, 64, 7),
                cb(&g::SPECTRUM_A_63_BITS, &g::SPECTRUM_A_63_CODES, 1, 0, 6, 64, 8),
                cb(&g::SPECTRUM_A_64_BITS, &g::SPECTRUM_A_64_CODES, 1, 0, 6, 64, 9),
            ],
            [
                cb(&g::SPECTRUM_A_71_BITS, &g::SPECTRUM_A_71_CODES, 1, 0, 7, 128, 8),
                cb(&g::SPECTRUM_A_72_BITS, &g::SPECTRUM_A_72_CODES, 1, 0, 7, 128, 8),
                cb(&g::SPECTRUM_A_73_BITS, &g::SPECTRUM_A_73_CODES, 1, 0, 7, 128, 9),
                cb(&g::SPECTRUM_A_74_BITS, &g::SPECTRUM_A_74_CODES, 1, 0, 7, 128, 10),
            ],
        ];

        // set 1 (B books); each row's index-0 book is unused.
        let spectrum_b: [[Option<HuffmanCodebook>; 4]; 8] = [
            [None, None, None, None],
            [None, None, None, None],
            [
                None,
                cb(&g::SPECTRUM_B_22_BITS, &g::SPECTRUM_B_22_CODES, 4, 2, 2, 4, 10),
                cb(&g::SPECTRUM_B_23_BITS, &g::SPECTRUM_B_23_CODES, 4, 2, 2, 4, 10),
                cb(&g::SPECTRUM_B_24_BITS, &g::SPECTRUM_B_24_CODES, 4, 2, 2, 4, 10),
            ],
            [
                None,
                cb(&g::SPECTRUM_B_32_BITS, &g::SPECTRUM_B_32_CODES, 2, 1, 3, 8, 9),
                cb(&g::SPECTRUM_B_33_BITS, &g::SPECTRUM_B_33_CODES, 2, 1, 3, 8, 10),
                cb(&g::SPECTRUM_B_34_BITS, &g::SPECTRUM_B_34_CODES, 2, 1, 3, 8, 10),
            ],
            [
                None,
                cb(&g::SPECTRUM_B_42_BITS, &g::SPECTRUM_B_42_CODES, 2, 1, 4, 16, 10),
                cb(&g::SPECTRUM_B_43_BITS, &g::SPECTRUM_B_43_CODES, 2, 1, 4, 16, 10),
                cb(&g::SPECTRUM_B_44_BITS, &g::SPECTRUM_B_44_CODES, 2, 1, 4, 16, 10),
            ],
            [
                None,
                cb(&g::SPECTRUM_B_52_BITS, &g::SPECTRUM_B_52_CODES, 1, 0, 5, 32, 7),
                cb(&g::SPECTRUM_B_53_BITS, &g::SPECTRUM_B_53_CODES, 1, 0, 5, 32, 8),
                cb(&g::SPECTRUM_B_54_BITS, &g::SPECTRUM_B_54_CODES, 1, 0, 5, 32, 9),
            ],
            [
                None,
                cb(&g::SPECTRUM_B_62_BITS, &g::SPECTRUM_B_62_CODES, 1, 0, 6, 64, 8),
                cb(&g::SPECTRUM_B_63_BITS, &g::SPECTRUM_B_63_CODES, 1, 0, 6, 64, 9),
                cb(&g::SPECTRUM_B_64_BITS, &g::SPECTRUM_B_64_CODES, 1, 0, 6, 64, 10),
            ],
            [
                None,
                cb(&g::SPECTRUM_B_72_BITS, &g::SPECTRUM_B_72_CODES, 1, 0, 7, 128, 9),
                cb(&g::SPECTRUM_B_73_BITS, &g::SPECTRUM_B_73_CODES, 1, 0, 7, 128, 10),
                cb(&g::SPECTRUM_B_74_BITS, &g::SPECTRUM_B_74_CODES, 1, 0, 7, 128, 10),
            ],
        ];

        Codebooks { sf_unsigned, sf_signed, spectrum: [spectrum_a, spectrum_b] }
    }
}
