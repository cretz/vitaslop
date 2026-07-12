//! A small, self-contained zlib/DEFLATE decompressor (RFC 1950 + RFC 1951).
//!
//! `vita-make-fself -c` stores each segment as a zlib stream, and compressed
//! eboots are the common homebrew form. Rather than pull in a decompression
//! crate (which would break the loader's dependency-free, wasm-clean guarantee),
//! we carry this minimal inflate. It handles stored, fixed-Huffman, and
//! dynamic-Huffman blocks - everything a standard zlib deflate emits - and
//! verifies the trailing Adler-32.
//!
//! Correctness is pinned by round-trip tests against Python's `zlib` output and,
//! end to end, by decompressing a real compressed cube eboot to the same image
//! the uncompressed one yields.

/// Why inflate failed.
#[derive(Debug, PartialEq, Eq)]
pub enum InflateError {
    /// Ran off the end of the compressed input.
    UnexpectedEof,
    /// A zlib header that is not DEFLATE, or has a preset dictionary we do not
    /// support.
    BadZlibHeader,
    /// A malformed DEFLATE stream (bad block type, code, or length).
    BadStream(&'static str),
    /// The trailing Adler-32 checksum did not match the output.
    BadChecksum,
}

/// LSB-first bit reader over a byte slice, as DEFLATE specifies.
struct BitReader<'a> {
    bytes: &'a [u8],
    byte_pos: usize,
    /// Bit buffer, filled LSB-first from consumed bytes.
    bits: u32,
    /// Number of valid bits currently in `bits`.
    count: u32,
}

impl<'a> BitReader<'a> {
    fn new(bytes: &'a [u8]) -> Self {
        BitReader { bytes, byte_pos: 0, bits: 0, count: 0 }
    }

    /// Read `n` bits (0..=32... but we only ever need <= 16), LSB first.
    fn bits(&mut self, n: u32) -> Result<u32, InflateError> {
        while self.count < n {
            let byte = *self
                .bytes
                .get(self.byte_pos)
                .ok_or(InflateError::UnexpectedEof)?;
            self.byte_pos += 1;
            self.bits |= (byte as u32) << self.count;
            self.count += 8;
        }
        let out = self.bits & ((1u32 << n) - 1);
        self.bits >>= n;
        self.count -= n;
        Ok(out)
    }

    /// Drop any partial bits and align to the next byte boundary.
    fn align(&mut self) {
        self.bits = 0;
        self.count = 0;
    }

    /// Read `len` raw bytes (used for stored blocks, after aligning).
    fn raw(&mut self, len: usize) -> Result<&'a [u8], InflateError> {
        let out = self
            .bytes
            .get(self.byte_pos..self.byte_pos + len)
            .ok_or(InflateError::UnexpectedEof)?;
        self.byte_pos += len;
        Ok(out)
    }
}

/// A canonical Huffman decoder built from a list of code lengths.
struct Huffman {
    /// Sorted symbols, grouped by increasing code length.
    symbols: Vec<u16>,
    /// `counts[l]` = number of codes of length `l`.
    counts: [u16; MAX_BITS + 1],
}

/// Maximum Huffman code length in DEFLATE.
const MAX_BITS: usize = 15;

impl Huffman {
    /// Build from per-symbol code lengths (0 = symbol unused).
    fn new(lengths: &[u8]) -> Result<Huffman, InflateError> {
        let mut counts = [0u16; MAX_BITS + 1];
        for &l in lengths {
            counts[l as usize] += 1;
        }
        counts[0] = 0; // length-0 symbols are not in the tree.

        // Offsets of each length group in the sorted symbol table.
        let mut offsets = [0u16; MAX_BITS + 2];
        for l in 1..=MAX_BITS {
            offsets[l + 1] = offsets[l] + counts[l];
        }

        let mut symbols = vec![0u16; lengths.len()];
        for (sym, &l) in lengths.iter().enumerate() {
            if l != 0 {
                symbols[offsets[l as usize] as usize] = sym as u16;
                offsets[l as usize] += 1;
            }
        }

        Ok(Huffman { symbols, counts })
    }

    /// Decode one symbol from the stream.
    fn decode(&self, r: &mut BitReader) -> Result<u16, InflateError> {
        let mut code = 0i32;
        let mut first = 0i32;
        let mut index = 0i32;
        for len in 1..=MAX_BITS {
            code |= r.bits(1)? as i32;
            let count = self.counts[len] as i32;
            if code - first < count {
                return Ok(self.symbols[(index + (code - first)) as usize]);
            }
            index += count;
            first = (first + count) << 1;
            code <<= 1;
        }
        Err(InflateError::BadStream("invalid huffman code"))
    }
}

// Length/distance base values and extra-bit counts (RFC 1951 sections 3.2.5).
const LEN_BASE: [u16; 29] = [
    3, 4, 5, 6, 7, 8, 9, 10, 11, 13, 15, 17, 19, 23, 27, 31, 35, 43, 51, 59, 67, 83, 99, 115, 131,
    163, 195, 227, 258,
];
const LEN_EXTRA: [u8; 29] = [
    0, 0, 0, 0, 0, 0, 0, 0, 1, 1, 1, 1, 2, 2, 2, 2, 3, 3, 3, 3, 4, 4, 4, 4, 5, 5, 5, 5, 0,
];
const DIST_BASE: [u16; 30] = [
    1, 2, 3, 4, 5, 7, 9, 13, 17, 25, 33, 49, 65, 97, 129, 193, 257, 385, 513, 769, 1025, 1537,
    2049, 3073, 4097, 6145, 8193, 12289, 16385, 24577,
];
const DIST_EXTRA: [u8; 30] = [
    0, 0, 0, 0, 1, 1, 2, 2, 3, 3, 4, 4, 5, 5, 6, 6, 7, 7, 8, 8, 9, 9, 10, 10, 11, 11, 12, 12, 13,
    13,
];
/// Order code lengths for the dynamic-Huffman code-length alphabet appear in.
const CLEN_ORDER: [usize; 19] =
    [16, 17, 18, 0, 8, 7, 9, 6, 10, 5, 11, 4, 12, 3, 13, 2, 14, 1, 15];

/// Decompress a raw DEFLATE stream into `out`.
fn inflate_raw(r: &mut BitReader, out: &mut Vec<u8>) -> Result<(), InflateError> {
    loop {
        let final_block = r.bits(1)? == 1;
        let btype = r.bits(2)?;
        match btype {
            0 => inflate_stored(r, out)?,
            1 => inflate_block(r, out, &fixed_lit_tree()?, &fixed_dist_tree()?)?,
            2 => {
                let (lit, dist) = dynamic_trees(r)?;
                inflate_block(r, out, &lit, &dist)?;
            }
            _ => return Err(InflateError::BadStream("reserved block type")),
        }
        if final_block {
            return Ok(());
        }
    }
}

/// A stored (uncompressed) block: aligned LEN/NLEN then LEN raw bytes.
fn inflate_stored(r: &mut BitReader, out: &mut Vec<u8>) -> Result<(), InflateError> {
    r.align();
    let lo = *r.bytes.get(r.byte_pos).ok_or(InflateError::UnexpectedEof)? as usize;
    let hi = *r.bytes.get(r.byte_pos + 1).ok_or(InflateError::UnexpectedEof)? as usize;
    r.byte_pos += 4; // LEN (2) + NLEN (2), NLEN is the one's complement (unchecked).
    let len = lo | (hi << 8);
    out.extend_from_slice(r.raw(len)?);
    Ok(())
}

/// Decode a compressed block given its literal/length and distance trees.
fn inflate_block(
    r: &mut BitReader,
    out: &mut Vec<u8>,
    lit: &Huffman,
    dist: &Huffman,
) -> Result<(), InflateError> {
    loop {
        let sym = lit.decode(r)?;
        if sym < 256 {
            out.push(sym as u8);
        } else if sym == 256 {
            return Ok(()); // end of block
        } else {
            let li = (sym - 257) as usize;
            if li >= LEN_BASE.len() {
                return Err(InflateError::BadStream("bad length symbol"));
            }
            let length = LEN_BASE[li] as usize + r.bits(LEN_EXTRA[li] as u32)? as usize;
            let dsym = dist.decode(r)? as usize;
            if dsym >= DIST_BASE.len() {
                return Err(InflateError::BadStream("bad distance symbol"));
            }
            let distance = DIST_BASE[dsym] as usize + r.bits(DIST_EXTRA[dsym] as u32)? as usize;
            if distance > out.len() {
                return Err(InflateError::BadStream("distance past output start"));
            }
            let start = out.len() - distance;
            for i in 0..length {
                out.push(out[start + i]);
            }
        }
    }
}

/// Read a dynamic block's literal/length and distance Huffman trees.
fn dynamic_trees(r: &mut BitReader) -> Result<(Huffman, Huffman), InflateError> {
    let hlit = r.bits(5)? as usize + 257;
    let hdist = r.bits(5)? as usize + 1;
    let hclen = r.bits(4)? as usize + 4;

    // Code-length alphabet lengths, in the specified permuted order.
    let mut cl_lengths = [0u8; 19];
    for i in 0..hclen {
        cl_lengths[CLEN_ORDER[i]] = r.bits(3)? as u8;
    }
    let cl_tree = Huffman::new(&cl_lengths)?;

    // Decode the literal+distance code lengths using the code-length tree, with
    // its repeat codes (16/17/18).
    let total = hlit + hdist;
    let mut lengths = vec![0u8; total];
    let mut i = 0;
    while i < total {
        let sym = cl_tree.decode(r)?;
        match sym {
            0..=15 => {
                lengths[i] = sym as u8;
                i += 1;
            }
            16 => {
                if i == 0 {
                    return Err(InflateError::BadStream("repeat with no prev length"));
                }
                let prev = lengths[i - 1];
                let repeat = r.bits(2)? as usize + 3;
                for _ in 0..repeat {
                    if i >= total {
                        return Err(InflateError::BadStream("length repeat overflow"));
                    }
                    lengths[i] = prev;
                    i += 1;
                }
            }
            17 => {
                let repeat = r.bits(3)? as usize + 3;
                i += repeat;
            }
            18 => {
                let repeat = r.bits(7)? as usize + 11;
                i += repeat;
            }
            _ => return Err(InflateError::BadStream("bad code-length symbol")),
        }
    }
    if i != total {
        return Err(InflateError::BadStream("code-length count mismatch"));
    }

    let lit = Huffman::new(&lengths[..hlit])?;
    let dist = Huffman::new(&lengths[hlit..])?;
    Ok((lit, dist))
}

/// The fixed literal/length tree (RFC 1951 3.2.6).
fn fixed_lit_tree() -> Result<Huffman, InflateError> {
    let mut lengths = [0u8; 288];
    for (i, l) in lengths.iter_mut().enumerate() {
        *l = match i {
            0..=143 => 8,
            144..=255 => 9,
            256..=279 => 7,
            _ => 8,
        };
    }
    Huffman::new(&lengths)
}

/// The fixed distance tree: 30 symbols, all 5-bit.
fn fixed_dist_tree() -> Result<Huffman, InflateError> {
    Huffman::new(&[5u8; 30])
}

/// Adler-32 over `data` (RFC 1950).
fn adler32(data: &[u8]) -> u32 {
    const MOD: u32 = 65521;
    let mut a = 1u32;
    let mut b = 0u32;
    for &byte in data {
        a = (a + byte as u32) % MOD;
        b = (b + a) % MOD;
    }
    (b << 16) | a
}

/// Decompress a zlib stream (RFC 1950: 2-byte header, DEFLATE body, Adler-32).
/// `expected_len`, when known, presizes the output buffer.
pub fn zlib_inflate(input: &[u8], expected_len: usize) -> Result<Vec<u8>, InflateError> {
    if input.len() < 2 {
        return Err(InflateError::UnexpectedEof);
    }
    let cmf = input[0];
    let flg = input[1];
    // Compression method 8 = DEFLATE; FCHECK makes (cmf<<8|flg) a multiple of 31.
    if cmf & 0x0f != 8 {
        return Err(InflateError::BadZlibHeader);
    }
    if ((cmf as u16) << 8 | flg as u16) % 31 != 0 {
        return Err(InflateError::BadZlibHeader);
    }
    // FDICT (preset dictionary) is not used by vita-make-fself; reject it.
    if flg & 0x20 != 0 {
        return Err(InflateError::BadZlibHeader);
    }

    let mut out = Vec::with_capacity(expected_len);
    let mut r = BitReader::new(&input[2..]);
    inflate_raw(&mut r, &mut out)?;

    // Trailing Adler-32 is big-endian, byte-aligned after the DEFLATE body.
    r.align();
    let trailer = r
        .bytes
        .get(r.byte_pos..r.byte_pos + 4)
        .ok_or(InflateError::UnexpectedEof)?;
    let want = u32::from_be_bytes([trailer[0], trailer[1], trailer[2], trailer[3]]);
    if adler32(&out) != want {
        return Err(InflateError::BadChecksum);
    }

    Ok(out)
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn adler32_known_vectors() {
        // Wikipedia's canonical example.
        assert_eq!(adler32(b"Wikipedia"), 0x11E6_0398);
        assert_eq!(adler32(b""), 1);
    }

    #[test]
    fn inflates_stored_block() {
        // A hand-built zlib stream with a single stored block "hi".
        // header 78 01, stored final block (BFINAL=1,BTYPE=00) byte 0x01,
        // LEN=0002 NLEN=fffd, "hi", then adler32.
        let payload = b"hi";
        let adler = adler32(payload).to_be_bytes();
        let mut s = vec![0x78, 0x01, 0x01, 0x02, 0x00, 0xfd, 0xff];
        s.extend_from_slice(payload);
        s.extend_from_slice(&adler);
        assert_eq!(zlib_inflate(&s, 2).unwrap(), payload);
    }

    #[test]
    fn rejects_non_deflate_header() {
        assert_eq!(
            zlib_inflate(&[0x77, 0x00], 0),
            Err(InflateError::BadZlibHeader)
        );
    }

    // Real zlib output (Python `zlib.compress`) covering dynamic Huffman, a
    // short input, and heavy back-references across the 32K window.

    #[test]
    fn inflates_dynamic_huffman() {
        // "the quick brown fox jumps over the lazy dog. " x 8
        const D1_C: &[u8] = &[
            0x78, 0xda, 0x2b, 0xc9, 0x48, 0x55, 0x28, 0x2c, 0xcd, 0x4c, 0xce, 0x56, 0x48, 0x2a,
            0xca, 0x2f, 0xcf, 0x53, 0x48, 0xcb, 0xaf, 0x50, 0xc8, 0x2a, 0xcd, 0x2d, 0x28, 0x56,
            0xc8, 0x2f, 0x4b, 0x2d, 0x52, 0x28, 0x01, 0x4a, 0xe7, 0x24, 0x56, 0x55, 0x2a, 0xa4,
            0xe4, 0xa7, 0xeb, 0x81, 0x79, 0xa3, 0x8a, 0xc9, 0x52, 0x0c, 0x00, 0x2f, 0xc0, 0x82,
            0x39,
        ];
        let expected: Vec<u8> = b"the quick brown fox jumps over the lazy dog. "
            .iter()
            .copied()
            .cycle()
            .take(45 * 8)
            .collect();
        assert_eq!(zlib_inflate(D1_C, expected.len()).unwrap(), expected);
    }

    #[test]
    fn inflates_short_input() {
        const D2_C: &[u8] =
            &[0x78, 0xda, 0x4b, 0x4c, 0x4a, 0x4e, 0x84, 0x21, 0x00, 0x1d, 0xe0, 0x04, 0x99];
        assert_eq!(zlib_inflate(D2_C, 12).unwrap(), b"abcabcabcabc");
    }

    #[test]
    fn inflates_heavy_backrefs() {
        // (0..256) repeated 3x - long-distance matches across the window.
        const D3_C: &[u8] = &[
            0x78, 0x9c, 0x63, 0x60, 0x64, 0x62, 0x66, 0x61, 0x65, 0x63, 0xe7, 0xe0, 0xe4, 0xe2,
            0xe6, 0xe1, 0xe5, 0xe3, 0x17, 0x10, 0x14, 0x12, 0x16, 0x11, 0x15, 0x13, 0x97, 0x90,
            0x94, 0x92, 0x96, 0x91, 0x95, 0x93, 0x57, 0x50, 0x54, 0x52, 0x56, 0x51, 0x55, 0x53,
            0xd7, 0xd0, 0xd4, 0xd2, 0xd6, 0xd1, 0xd5, 0xd3, 0x37, 0x30, 0x34, 0x32, 0x36, 0x31,
            0x35, 0x33, 0xb7, 0xb0, 0xb4, 0xb2, 0xb6, 0xb1, 0xb5, 0xb3, 0x77, 0x70, 0x74, 0x72,
            0x76, 0x71, 0x75, 0x73, 0xf7, 0xf0, 0xf4, 0xf2, 0xf6, 0xf1, 0xf5, 0xf3, 0x0f, 0x08,
            0x0c, 0x0a, 0x0e, 0x09, 0x0d, 0x0b, 0x8f, 0x88, 0x8c, 0x8a, 0x8e, 0x89, 0x8d, 0x8b,
            0x4f, 0x48, 0x4c, 0x4a, 0x4e, 0x49, 0x4d, 0x4b, 0xcf, 0xc8, 0xcc, 0xca, 0xce, 0xc9,
            0xcd, 0xcb, 0x2f, 0x28, 0x2c, 0x2a, 0x2e, 0x29, 0x2d, 0x2b, 0xaf, 0xa8, 0xac, 0xaa,
            0xae, 0xa9, 0xad, 0xab, 0x6f, 0x68, 0x6c, 0x6a, 0x6e, 0x69, 0x6d, 0x6b, 0xef, 0xe8,
            0xec, 0xea, 0xee, 0xe9, 0xed, 0xeb, 0x9f, 0x30, 0x71, 0xd2, 0xe4, 0x29, 0x53, 0xa7,
            0x4d, 0x9f, 0x31, 0x73, 0xd6, 0xec, 0x39, 0x73, 0xe7, 0xcd, 0x5f, 0xb0, 0x70, 0xd1,
            0xe2, 0x25, 0x4b, 0x97, 0x2d, 0x5f, 0xb1, 0x72, 0xd5, 0xea, 0x35, 0x6b, 0xd7, 0xad,
            0xdf, 0xb0, 0x71, 0xd3, 0xe6, 0x2d, 0x5b, 0xb7, 0x6d, 0xdf, 0xb1, 0x73, 0xd7, 0xee,
            0x3d, 0x7b, 0xf7, 0xed, 0x3f, 0x70, 0xf0, 0xd0, 0xe1, 0x23, 0x47, 0x8f, 0x1d, 0x3f,
            0x71, 0xf2, 0xd4, 0xe9, 0x33, 0x67, 0xcf, 0x9d, 0xbf, 0x70, 0xf1, 0xd2, 0xe5, 0x2b,
            0x57, 0xaf, 0x5d, 0xbf, 0x71, 0xf3, 0xd6, 0xed, 0x3b, 0x77, 0xef, 0xdd, 0x7f, 0xf0,
            0xf0, 0xd1, 0xe3, 0x27, 0x4f, 0x9f, 0x3d, 0x7f, 0xf1, 0xf2, 0xd5, 0xeb, 0x37, 0x6f,
            0xdf, 0xbd, 0xff, 0xf0, 0xf1, 0xd3, 0xe7, 0x2f, 0x5f, 0xbf, 0x7d, 0xff, 0xf1, 0xf3,
            0xd7, 0xef, 0x3f, 0x7f, 0xff, 0xfd, 0x67, 0x18, 0xf5, 0x3f, 0xf3, 0x48, 0xf6, 0x3f,
            0x00, 0xa0, 0x62, 0x7e, 0x90,
        ];
        let expected: Vec<u8> = (0..=255u8).cycle().take(768).collect();
        assert_eq!(zlib_inflate(D3_C, 768).unwrap(), expected);
    }
}
