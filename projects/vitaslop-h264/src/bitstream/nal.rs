//! NAL units: finding them, and undoing the byte stuffing inside them.

use crate::error::{Error, Result};

/// `nal_unit_type` values this crate acts on. Everything else is passed through to the
/// platform decoder untouched, which is the right default: an SEI or a filler NAL is the
/// decoder's business, not ours.
pub mod kind {
    /// Coded slice of a non-IDR picture.
    pub const SLICE: u8 = 1;
    /// Coded slice data partition A - the only partition carrying a slice header.
    pub const DPA: u8 = 2;
    /// Coded slice of an IDR picture.
    pub const IDR: u8 = 5;
    /// Supplemental enhancement information.
    pub const SEI: u8 = 6;
    /// Sequence parameter set.
    pub const SPS: u8 = 7;
    /// Picture parameter set.
    pub const PPS: u8 = 8;
    /// Access unit delimiter.
    pub const AUD: u8 = 9;
    /// End of sequence.
    pub const END_OF_SEQ: u8 = 10;
    /// End of stream.
    pub const END_OF_STREAM: u8 = 11;
    /// Coded slice extension (MVC/SVC): this crate decodes the base view only.
    pub const SLICE_EXT: u8 = 20;
}

/// One NAL unit inside a buffer.
#[derive(Debug, Clone, Copy)]
pub struct Nal<'a> {
    /// `nal_ref_idc`: zero means the picture is not used for reference.
    pub ref_idc: u8,
    /// `nal_unit_type`.
    pub kind: u8,
    /// The bytes AFTER the one-byte header, still carrying emulation-prevention bytes.
    pub payload: &'a [u8],
    /// The whole NAL including its header byte - what a backend re-emits.
    pub raw: &'a [u8],
}

impl<'a> Nal<'a> {
    /// Parse the header byte and wrap the rest.
    pub fn parse(raw: &'a [u8]) -> Result<Nal<'a>> {
        let &first = raw.first().ok_or_else(|| Error::bitstream("empty NAL unit"))?;
        if first & 0x80 != 0 {
            return Err(Error::bitstream("forbidden_zero_bit set in NAL header"));
        }
        Ok(Nal { ref_idc: (first >> 5) & 3, kind: first & 0x1f, payload: &raw[1..], raw })
    }

    /// True for the NAL types that carry a slice header, i.e. that belong to a picture.
    pub fn is_slice(&self) -> bool {
        matches!(self.kind, kind::SLICE | kind::DPA | kind::IDR)
    }
}

/// Iterate the NAL units of an Annex B byte stream (`00 00 01` / `00 00 00 01` start codes).
///
/// Trailing zero bytes are trimmed from each NAL, per the `trailing_zero_8bits` allowance:
/// leaving them on turns a byte-exact re-emission into a different (still legal, but
/// different) stream, and a decoder that hashes its input would then see two.
pub fn split_annex_b(data: &[u8]) -> AnnexBIter<'_> {
    AnnexBIter { data, pos: 0 }
}

/// Iterator returned by [`split_annex_b`].
pub struct AnnexBIter<'a> {
    data: &'a [u8],
    pos: usize,
}

impl<'a> Iterator for AnnexBIter<'a> {
    type Item = &'a [u8];

    fn next(&mut self) -> Option<&'a [u8]> {
        let start = find_start_code(self.data, self.pos)?;
        let body = start.end;
        let next = find_start_code(self.data, body);
        let end = match &next {
            Some(sc) => sc.start,
            None => self.data.len(),
        };
        self.pos = match &next {
            Some(sc) => sc.start,
            None => self.data.len(),
        };
        let mut nal = &self.data[body..end];
        while nal.last() == Some(&0) {
            nal = &nal[..nal.len() - 1];
        }
        if nal.is_empty() { self.next() } else { Some(nal) }
    }
}

struct StartCode {
    start: usize,
    end: usize,
}

/// Find the next `00 00 01` at or after `from`, reporting where its leading zeros begin
/// (so the previous NAL can be cut before them) and where the payload starts.
fn find_start_code(data: &[u8], from: usize) -> Option<StartCode> {
    let mut i = from;
    while i + 2 < data.len() {
        if data[i] == 0 && data[i + 1] == 0 && data[i + 2] == 1 {
            let mut start = i;
            if start > from && data[start - 1] == 0 {
                start -= 1;
            }
            return Some(StartCode { start, end: i + 3 });
        }
        // The next start code cannot begin before the first zero of a `00 00 01`, so a
        // non-zero byte at i+2 lets us jump the whole triple.
        i += if data[i + 2] != 0 { 3 } else { 1 };
    }
    None
}

/// True if `data` looks like an Annex B stream (starts with a start code).
pub fn is_annex_b(data: &[u8]) -> bool {
    data.starts_with(&[0, 0, 1]) || data.starts_with(&[0, 0, 0, 1])
}

/// Strip emulation-prevention bytes from a NAL payload into `out`, giving the RBSP the
/// syntax parsers read.
///
/// `00 00 03` in a NAL always means "the 03 is stuffing"; the RBSP is the same bytes with
/// each such 03 removed.
pub fn rbsp_into(payload: &[u8], out: &mut Vec<u8>) {
    out.clear();
    out.reserve(payload.len());
    let mut zeros = 0usize;
    for &b in payload {
        if zeros >= 2 && b == 3 {
            zeros = 0;
            continue;
        }
        if b == 0 {
            zeros += 1;
        } else {
            zeros = 0;
        }
        out.push(b);
    }
}

/// [`rbsp_into`] with its own allocation, for callers that are not in a hot loop.
pub fn rbsp(payload: &[u8]) -> Vec<u8> {
    let mut out = Vec::new();
    rbsp_into(payload, &mut out);
    out
}

/// Insert emulation-prevention bytes: the inverse of [`rbsp_into`], used when building a
/// NAL (the synthetic conformance stream, and any caller re-serialising parameter sets).
pub fn escape_rbsp(rbsp: &[u8], out: &mut Vec<u8>) {
    let mut zeros = 0usize;
    for &b in rbsp {
        if zeros >= 2 && b <= 3 {
            out.push(3);
            zeros = 0;
        }
        if b == 0 {
            zeros += 1;
        } else {
            zeros = 0;
        }
        out.push(b);
    }
}

/// Append `nal` (header byte included, RBSP already escaped) to an Annex B stream.
pub fn write_annex_b(nal: &[u8], out: &mut Vec<u8>) {
    out.extend_from_slice(&[0, 0, 0, 1]);
    out.extend_from_slice(nal);
}

/// Iterate the NAL units of a length-prefixed (AVCC / ISO-BMFF sample) buffer.
///
/// `length_size` is the `lengthSizeMinusOne + 1` from the avcC record: 1, 2 or 4.
pub fn split_length_prefixed(data: &[u8], length_size: usize) -> Result<Vec<&[u8]>> {
    if !matches!(length_size, 1 | 2 | 4) {
        return Err(Error::bitstream(format!("illegal NAL length size {length_size}")));
    }
    let mut out = Vec::new();
    let mut pos = 0usize;
    while pos + length_size <= data.len() {
        let mut len = 0usize;
        for i in 0..length_size {
            len = (len << 8) | data[pos + i] as usize;
        }
        pos += length_size;
        if len == 0 {
            continue;
        }
        if pos + len > data.len() {
            return Err(Error::bitstream(format!(
                "NAL length {len} runs {} bytes past the end of the sample",
                pos + len - data.len()
            )));
        }
        out.push(&data[pos..pos + len]);
        pos += len;
    }
    if pos != data.len() {
        return Err(Error::bitstream("trailing bytes after the last length-prefixed NAL"));
    }
    Ok(out)
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn splits_on_both_start_code_lengths() {
        let s = [0, 0, 0, 1, 0x67, 0xaa, 0, 0, 1, 0x68, 0xbb, 0xcc];
        let nals: Vec<_> = split_annex_b(&s).collect();
        assert_eq!(nals, vec![&[0x67u8, 0xaa][..], &[0x68, 0xbb, 0xcc][..]]);
    }

    #[test]
    fn trailing_zeros_are_not_part_of_a_nal() {
        let s = [0, 0, 1, 0x09, 0x10, 0, 0, 0, 0, 1, 0x67, 0x42];
        let nals: Vec<_> = split_annex_b(&s).collect();
        assert_eq!(nals, vec![&[0x09u8, 0x10][..], &[0x67, 0x42][..]]);
    }

    #[test]
    fn emulation_prevention_round_trips() {
        let raw = vec![0u8, 0, 0, 1, 2, 3, 0, 0, 3, 0, 0, 2];
        let mut escaped = Vec::new();
        escape_rbsp(&raw, &mut escaped);
        assert!(escaped.windows(3).all(|w| w != [0, 0, 1]));
        let back = rbsp(&escaped);
        assert_eq!(back, raw);
    }

    #[test]
    fn length_prefixed_rejects_a_short_sample() {
        let data = [0, 0, 0, 9, 0x65, 0x88];
        assert!(split_length_prefixed(&data, 4).is_err());
    }

    #[test]
    fn header_bits_are_split_out() {
        let nal = Nal::parse(&[0x65, 0x88]).unwrap();
        assert_eq!(nal.kind, kind::IDR);
        assert_eq!(nal.ref_idc, 3);
        assert!(nal.is_slice());
    }
}
