//! RBSP bit reader.
//!
//! Reads the Exp-Golomb and fixed-width syntax elements of H.264 out of a byte slice that
//! has already had its emulation-prevention bytes removed (see [`super::nal::rbsp`]).
//!
//! Every read is checked. A truncated NAL is a stream error, not a panic and not a zero:
//! this crate is fed by files and by networks, and a decoder that reads past the end of a
//! short NAL and carries on is exactly how a malformed stream turns into a wrong picture.

use crate::error::{Error, Result};

/// A big-endian bit reader over an RBSP.
pub struct BitReader<'a> {
    data: &'a [u8],
    /// Next bit to read, counted from the first bit of `data`.
    pos: usize,
}

impl<'a> BitReader<'a> {
    /// Wrap an RBSP.
    pub fn new(data: &'a [u8]) -> Self {
        BitReader { data, pos: 0 }
    }

    /// Bits consumed so far. This is the value VA-API wants as
    /// `slice_data_bit_offset`, so it is public rather than an implementation detail.
    pub fn bit_pos(&self) -> usize {
        self.pos
    }

    /// Bits left unread.
    pub fn bits_left(&self) -> usize {
        self.data.len() * 8 - self.pos
    }

    /// Read one bit.
    pub fn u1(&mut self) -> Result<u32> {
        if self.pos >= self.data.len() * 8 {
            return Err(Error::bitstream("read past end of NAL"));
        }
        let byte = self.data[self.pos >> 3];
        let bit = (byte >> (7 - (self.pos & 7))) & 1;
        self.pos += 1;
        Ok(bit as u32)
    }

    /// Read one bit as a flag.
    pub fn flag(&mut self) -> Result<bool> {
        Ok(self.u1()? != 0)
    }

    /// Read `n` bits (n <= 32) as an unsigned value, MSB first.
    pub fn bits(&mut self, n: u32) -> Result<u32> {
        debug_assert!(n <= 32);
        if n == 0 {
            return Ok(0);
        }
        if self.bits_left() < n as usize {
            return Err(Error::bitstream("read past end of NAL"));
        }
        let mut v: u32 = 0;
        let mut left = n;
        while left > 0 {
            let byte = self.data[self.pos >> 3];
            let avail = 8 - (self.pos & 7) as u32;
            let take = avail.min(left);
            let shift = avail - take;
            let mask = if take == 8 { 0xffu32 } else { (1u32 << take) - 1 };
            v = (v << take) | ((byte as u32 >> shift) & mask);
            self.pos += take as usize;
            left -= take;
        }
        Ok(v)
    }

    /// Skip `n` bits.
    pub fn skip(&mut self, n: usize) -> Result<()> {
        if self.bits_left() < n {
            return Err(Error::bitstream("skip past end of NAL"));
        }
        self.pos += n;
        Ok(())
    }

    /// `ue(v)`: unsigned Exp-Golomb.
    ///
    /// The 32-bit cap is not arbitrary: the spec's own ranges make every legal `ue(v)` fit,
    /// so a longer prefix means a corrupt NAL, and reading it as a wider value would only
    /// let corruption through quietly.
    pub fn ue(&mut self) -> Result<u32> {
        let mut leading = 0u32;
        while self.u1()? == 0 {
            leading += 1;
            if leading > 31 {
                return Err(Error::bitstream("Exp-Golomb prefix longer than 32 bits"));
            }
        }
        if leading == 0 {
            return Ok(0);
        }
        let rest = self.bits(leading)?;
        Ok((1u32 << leading) - 1 + rest)
    }

    /// `se(v)`: signed Exp-Golomb.
    pub fn se(&mut self) -> Result<i32> {
        let k = self.ue()?;
        // 0 -> 0, 1 -> +1, 2 -> -1, 3 -> +2 ...
        let magnitude = k.div_ceil(2) as i32;
        Ok(if k & 1 == 1 { magnitude } else { -magnitude })
    }

    /// `ue(v)` constrained to a maximum, which every caller here has.
    pub fn ue_max(&mut self, max: u32, what: &'static str) -> Result<u32> {
        let v = self.ue()?;
        if v > max {
            return Err(Error::bitstream(format!("{what} = {v} exceeds {max}")));
        }
        Ok(v)
    }

    /// True while `more_rbsp_data()` holds: there is a bit after the trailing-one that is
    /// not part of the trailing zero padding.
    pub fn more_rbsp_data(&self) -> bool {
        let total = self.data.len() * 8;
        if self.pos >= total {
            return false;
        }
        // Find the last set bit in the whole RBSP: that is the rbsp_stop_one_bit.
        let mut last_one = None;
        for (i, &b) in self.data.iter().enumerate().rev() {
            if b != 0 {
                let bit_in_byte = 7 - b.trailing_zeros() as usize;
                last_one = Some(i * 8 + bit_in_byte);
                break;
            }
        }
        match last_one {
            Some(stop) => self.pos < stop,
            None => false,
        }
    }
}

#[cfg(test)]
// The binary literals below are grouped by Exp-Golomb codeword, not by nibble: that is what
// makes them readable as the spec's own examples.
#[allow(clippy::unusual_byte_groupings)]
mod tests {
    use super::*;

    #[test]
    fn exp_golomb_round_trips_the_spec_examples() {
        // Table 9-1: 1 -> 0, 010 -> 1, 011 -> 2, 00100 -> 3 ...
        let data = [0b1_010_011_0, 0b0100_0000];
        let mut r = BitReader::new(&data);
        assert_eq!(r.ue().unwrap(), 0);
        assert_eq!(r.ue().unwrap(), 1);
        assert_eq!(r.ue().unwrap(), 2);
        assert_eq!(r.ue().unwrap(), 3);
    }

    #[test]
    fn signed_exp_golomb_alternates() {
        let data = [0b1_010_011_0, 0b0100_0010, 0b1000_0000];
        let mut r = BitReader::new(&data);
        assert_eq!(r.se().unwrap(), 0);
        assert_eq!(r.se().unwrap(), 1);
        assert_eq!(r.se().unwrap(), -1);
        assert_eq!(r.se().unwrap(), 2);
        assert_eq!(r.se().unwrap(), -2);
    }

    #[test]
    fn truncation_is_an_error_not_a_zero() {
        let data = [0x00];
        let mut r = BitReader::new(&data);
        assert!(r.ue().is_err());
        let mut r = BitReader::new(&data);
        assert!(r.bits(9).is_err());
    }

    #[test]
    fn fixed_width_reads_cross_byte_boundaries() {
        let data = [0xde, 0xad, 0xbe, 0xef];
        let mut r = BitReader::new(&data);
        assert_eq!(r.bits(4).unwrap(), 0xd);
        assert_eq!(r.bits(16).unwrap(), 0xeadb);
        assert_eq!(r.bits(12).unwrap(), 0xeef);
        assert_eq!(r.bits_left(), 0);
    }
}
