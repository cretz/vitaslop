//! The `AVCDecoderConfigurationRecord` (ISO/IEC 14496-15 5.3.3.1) - "avcC".
//!
//! This is how MP4, WebCodecs and VideoToolbox all name a stream's parameter sets, while
//! Annex B carries them inline. Every backend needs one form or the other, so the
//! conversion lives here once.

use super::nal;
use crate::error::{Error, Result};

/// A parsed avcC record.
#[derive(Debug, Clone, PartialEq, Eq)]
pub struct AvcC {
    /// `AVCProfileIndication`.
    pub profile_idc: u8,
    /// `profile_compatibility`.
    pub profile_compat: u8,
    /// `AVCLevelIndication`.
    pub level_idc: u8,
    /// `lengthSizeMinusOne + 1`: bytes of NAL length prefix in each sample (1, 2 or 4).
    pub length_size: usize,
    /// Sequence parameter set NALs, header byte included, without start codes.
    pub sps: Vec<Vec<u8>>,
    /// Picture parameter set NALs, header byte included, without start codes.
    pub pps: Vec<Vec<u8>>,
}

impl AvcC {
    /// Parse an avcC record.
    pub fn parse(data: &[u8]) -> Result<AvcC> {
        if data.len() < 7 {
            return Err(Error::bitstream("avcC record shorter than its fixed header"));
        }
        if data[0] != 1 {
            return Err(Error::unsupported(format!(
                "avcC configurationVersion {} (only 1 is defined)",
                data[0]
            )));
        }
        let mut rec = AvcC {
            profile_idc: data[1],
            profile_compat: data[2],
            level_idc: data[3],
            length_size: (data[4] & 3) as usize + 1,
            sps: Vec::new(),
            pps: Vec::new(),
        };
        if rec.length_size == 3 {
            return Err(Error::bitstream("avcC lengthSizeMinusOne = 2 is reserved"));
        }
        let mut pos = 5usize;
        let num_sps = (data[pos] & 0x1f) as usize;
        pos += 1;
        for _ in 0..num_sps {
            rec.sps.push(take_sized_nal(data, &mut pos, "SPS")?);
        }
        if pos >= data.len() {
            return Err(Error::bitstream("avcC ends before its PPS count"));
        }
        let num_pps = data[pos] as usize;
        pos += 1;
        for _ in 0..num_pps {
            rec.pps.push(take_sized_nal(data, &mut pos, "PPS")?);
        }
        // The trailing high-profile block (chroma_format, bit depths, SPS extensions) is
        // optional and adds nothing this crate needs: the SPS itself carries all of it.
        Ok(rec)
    }

    /// Serialise back to an avcC record - what VideoToolbox and WebCodecs want as
    /// `description`, and what an MP4 writer stores.
    pub fn to_bytes(&self) -> Vec<u8> {
        let mut out = Vec::with_capacity(16 + self.sps.iter().map(|s| s.len() + 2).sum::<usize>());
        out.extend_from_slice(&[
            1,
            self.profile_idc,
            self.profile_compat,
            self.level_idc,
            0xfc | (self.length_size as u8 - 1),
            0xe0 | (self.sps.len() as u8 & 0x1f),
        ]);
        for s in &self.sps {
            out.extend_from_slice(&(s.len() as u16).to_be_bytes());
            out.extend_from_slice(s);
        }
        out.push(self.pps.len() as u8);
        for p in &self.pps {
            out.extend_from_slice(&(p.len() as u16).to_be_bytes());
            out.extend_from_slice(p);
        }
        out
    }

    /// Build a record from raw parameter set NALs, taking the profile/level out of the SPS
    /// itself (bytes 1..4 of an SPS NAL are exactly profile, constraints, level).
    pub fn from_parameter_sets(sps: Vec<Vec<u8>>, pps: Vec<Vec<u8>>, length_size: usize) -> Result<AvcC> {
        let first = sps.first().ok_or_else(|| Error::bitstream("avcC needs at least one SPS"))?;
        if first.len() < 4 {
            return Err(Error::bitstream("SPS NAL too short to carry a profile"));
        }
        Ok(AvcC {
            profile_idc: first[1],
            profile_compat: first[2],
            level_idc: first[3],
            length_size,
            sps,
            pps,
        })
    }

    /// The parameter sets as an Annex B prelude, for backends fed in that format.
    pub fn to_annex_b(&self) -> Vec<u8> {
        let mut out = Vec::new();
        for s in self.sps.iter().chain(self.pps.iter()) {
            nal::write_annex_b(s, &mut out);
        }
        out
    }
}

fn take_sized_nal(data: &[u8], pos: &mut usize, what: &'static str) -> Result<Vec<u8>> {
    if *pos + 2 > data.len() {
        return Err(Error::bitstream(format!("avcC ends inside a {what} length")));
    }
    let len = u16::from_be_bytes([data[*pos], data[*pos + 1]]) as usize;
    *pos += 2;
    if *pos + len > data.len() {
        return Err(Error::bitstream(format!("avcC {what} runs past the end of the record")));
    }
    let nal = data[*pos..*pos + len].to_vec();
    *pos += len;
    Ok(nal)
}

/// Rewrite a length-prefixed sample as Annex B, appending to `out`.
pub fn length_prefixed_to_annex_b(sample: &[u8], length_size: usize, out: &mut Vec<u8>) -> Result<()> {
    for n in nal::split_length_prefixed(sample, length_size)? {
        nal::write_annex_b(n, out);
    }
    Ok(())
}

/// Rewrite an Annex B access unit as a length-prefixed sample, appending to `out`.
pub fn annex_b_to_length_prefixed(au: &[u8], length_size: usize, out: &mut Vec<u8>) {
    for n in nal::split_annex_b(au) {
        let len = n.len();
        for i in (0..length_size).rev() {
            out.push((len >> (i * 8)) as u8);
        }
        out.extend_from_slice(n);
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn avcc_round_trips() {
        let rec = AvcC {
            profile_idc: 66,
            profile_compat: 0xc0,
            level_idc: 30,
            length_size: 4,
            sps: vec![vec![0x67, 0x42, 0xc0, 0x1e, 0xaa]],
            pps: vec![vec![0x68, 0xce, 0x3c, 0x80]],
        };
        let bytes = rec.to_bytes();
        assert_eq!(AvcC::parse(&bytes).unwrap(), rec);
    }

    #[test]
    fn sample_formats_convert_both_ways() {
        let annex_b = [0u8, 0, 0, 1, 0x65, 0x11, 0x22, 0, 0, 0, 1, 0x41, 0x33];
        let mut prefixed = Vec::new();
        annex_b_to_length_prefixed(&annex_b, 4, &mut prefixed);
        assert_eq!(prefixed, vec![0, 0, 0, 3, 0x65, 0x11, 0x22, 0, 0, 0, 2, 0x41, 0x33]);
        let mut back = Vec::new();
        length_prefixed_to_annex_b(&prefixed, 4, &mut back).unwrap();
        assert_eq!(back, annex_b);
    }

    #[test]
    fn a_truncated_record_is_an_error() {
        assert!(AvcC::parse(&[1, 66, 0, 30, 0xff, 0xe1, 0, 20, 1, 2]).is_err());
    }
}
