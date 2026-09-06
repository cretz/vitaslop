//! Big-endian bit reader, a direct port of the LibAtrac9 `BitReaderCxt`. Bits are
//! consumed most-significant-first from a byte buffer. The fast paths mirror the C
//! exactly (2/3/4-byte windowed reads) with a general fallback for wider reads; the
//! only deviation is computing through `u32` so the masking and shifts are
//! unambiguous, which is value-identical to the C `int` arithmetic here (every
//! result is non-negative and < 2^25 on the fast paths).

/// A cursor over a borrowed byte buffer, tracking a bit position.
pub(crate) struct BitReader<'a> {
    buffer: &'a [u8],
    /// Current read position, in bits from the start of the buffer.
    pub position: usize,
}

impl<'a> BitReader<'a> {
    pub(crate) fn new(buffer: &'a [u8]) -> Self {
        BitReader { buffer, position: 0 }
    }

    /// Byte at `i`, or 0 past the end. The C reads raw memory a few bytes past the
    /// live frame (into superframe padding); returning 0 there is equivalent for
    /// the masked fast paths and keeps the reader memory-safe.
    #[inline]
    fn byte(&self, i: usize) -> u32 {
        self.buffer.get(i).copied().unwrap_or(0) as u32
    }

    /// Read `bits` unsigned bits and advance.
    #[inline]
    pub(crate) fn read_int(&mut self, bits: i32) -> i32 {
        let value = self.peek_int(bits);
        self.position += bits as usize;
        value
    }

    /// Read `bits` bits and sign-extend from that width.
    #[inline]
    pub(crate) fn read_signed_int(&mut self, bits: i32) -> i32 {
        let value = self.peek_int(bits);
        self.position += bits as usize;
        sign_extend32(value, bits)
    }

    /// Read `bits` bits as an offset-binary value (subtract the mid-point).
    #[inline]
    pub(crate) fn read_offset_binary(&mut self, bits: i32) -> i32 {
        let offset = 1 << (bits - 1);
        let value = self.peek_int(bits) - offset;
        self.position += bits as usize;
        value
    }

    /// Peek `bits` bits without advancing.
    #[inline]
    pub(crate) fn peek_int(&self, bits: i32) -> i32 {
        let byte_index = self.position / 8;
        let bit_index = (self.position % 8) as u32;
        let bits = bits as u32;

        if bits <= 9 {
            let mut value = (self.byte(byte_index) << 8) | self.byte(byte_index + 1);
            value &= 0xFFFF >> bit_index;
            value >>= 16 - bits - bit_index;
            return value as i32;
        }
        if bits <= 17 {
            let mut value = (self.byte(byte_index) << 16)
                | (self.byte(byte_index + 1) << 8)
                | self.byte(byte_index + 2);
            value &= 0xFF_FFFF >> bit_index;
            value >>= 24 - bits - bit_index;
            return value as i32;
        }
        if bits <= 25 {
            let mut value = (self.byte(byte_index) << 24)
                | (self.byte(byte_index + 1) << 16)
                | (self.byte(byte_index + 2) << 8)
                | self.byte(byte_index + 3);
            value &= 0xFFFF_FFFF >> bit_index;
            value >>= 32 - bits - bit_index;
            return value as i32;
        }
        self.peek_int_fallback(bits)
    }

    fn peek_int_fallback(&self, mut bit_count: u32) -> i32 {
        let mut value: u32 = 0;
        let mut byte_index = self.position / 8;
        let mut bit_index = (self.position % 8) as u32;

        while bit_count > 0 {
            if bit_index >= 8 {
                bit_index = 0;
                byte_index += 1;
            }
            let mut bits_to_read = bit_count;
            if bits_to_read > 8 - bit_index {
                bits_to_read = 8 - bit_index;
            }
            let mask = 0xFF >> bit_index;
            let current_byte = (mask & self.byte(byte_index)) >> (8 - bit_index - bits_to_read);
            value = (value << bits_to_read) | current_byte;
            bit_index += bits_to_read;
            bit_count -= bits_to_read;
        }
        value as i32
    }

    /// Advance the position to the next multiple of `multiple` bits.
    #[inline]
    pub(crate) fn align_position(&mut self, multiple: usize) {
        let rem = self.position % multiple;
        if rem != 0 {
            self.position += multiple - rem;
        }
    }
}

/// Sign-extend the low `bits` bits of `value` to a full `i32`.
#[inline]
pub(crate) fn sign_extend32(value: i32, bits: i32) -> i32 {
    let shift = 32 - bits;
    (value << shift) >> shift
}
