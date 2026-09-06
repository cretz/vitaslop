//! Everything that reads the H.264 bitstream itself: NAL framing, parameter sets, slice
//! headers, picture order counts, and the avcC record.
//!
//! No platform code below this module, and nothing here allocates per macroblock: this
//! layer reads only what a decoder needs to be DRIVEN correctly, never the residual data.
//! The pixels are the platform decoder's job.

pub mod avcc;
pub mod nal;
pub mod pps;
pub mod poc;
pub mod reader;
pub mod slice;
pub mod sps;

use std::collections::HashMap;

use crate::error::{Error, Result};
use nal::Nal;
use poc::{Poc, PocState};
use pps::Pps;
use slice::SliceHeader;
use sps::Sps;

/// The parameter sets in force, keyed by id.
///
/// A stream may redefine a set mid-stream, and doing so is how resolution changes are
/// signalled. Later definitions simply replace earlier ones, which is what the spec says
/// happens at the next picture that references them.
#[derive(Debug, Clone, Default)]
pub struct ParameterSets {
    sps: HashMap<u32, Sps>,
    pps: HashMap<u32, Pps>,
    /// The parameter sets as they arrived, header byte included. Kept because avcC - which
    /// is what VideoToolbox, WebCodecs and MP4 all want - carries the NALs verbatim, and
    /// re-serialising a parsed SPS would produce a different (if equivalent) bit pattern.
    sps_nal: HashMap<u32, Vec<u8>>,
    pps_nal: HashMap<u32, Vec<u8>>,
}

impl ParameterSets {
    /// Empty.
    pub fn new() -> Self {
        Self::default()
    }

    /// The SPS with this id, if it has been seen.
    pub fn sps(&self, id: u32) -> Option<&Sps> {
        self.sps.get(&id)
    }

    /// The PPS with this id, if it has been seen.
    pub fn pps(&self, id: u32) -> Option<&Pps> {
        self.pps.get(&id)
    }

    /// Every SPS NAL seen, in id order.
    pub fn sps_nals(&self) -> Vec<Vec<u8>> {
        let mut ids: Vec<_> = self.sps_nal.keys().copied().collect();
        ids.sort_unstable();
        ids.iter().map(|id| self.sps_nal[id].clone()).collect()
    }

    /// Every PPS NAL seen, in id order.
    pub fn pps_nals(&self) -> Vec<Vec<u8>> {
        let mut ids: Vec<_> = self.pps_nal.keys().copied().collect();
        ids.sort_unstable();
        ids.iter().map(|id| self.pps_nal[id].clone()).collect()
    }

    /// True once at least one SPS and one PPS are known, i.e. a slice can be parsed.
    pub fn is_ready(&self) -> bool {
        !self.sps.is_empty() && !self.pps.is_empty()
    }

    /// Take in an SPS or PPS NAL. Returns true when this changed the active configuration
    /// (a new set, or a different definition of an existing id) - which is the signal a
    /// backend needs to reconfigure itself.
    pub fn add_nal(&mut self, n: &Nal<'_>, scratch: &mut Vec<u8>) -> Result<bool> {
        nal::rbsp_into(n.payload, scratch);
        match n.kind {
            nal::kind::SPS => {
                let sps = Sps::parse(scratch)?;
                self.sps_nal.insert(sps.id, n.raw.to_vec());
                let id = sps.id;
                Ok(self.sps.insert(id, sps.clone()) != Some(sps))
            }
            nal::kind::PPS => {
                let pps = Pps::parse(scratch, self)?;
                self.pps_nal.insert(pps.id, n.raw.to_vec());
                let id = pps.id;
                Ok(self.pps.insert(id, pps.clone()) != Some(pps))
            }
            _ => Ok(false),
        }
    }
}

/// Where one slice sits inside an [`AccessUnit`], and what its header said.
///
/// The headers are kept rather than re-parsed: the splitter has to read every one of them
/// anyway to find picture boundaries, and the VA-API backend needs all of them.
#[derive(Debug, Clone)]
pub struct SliceRef {
    /// The slice's parsed header.
    pub header: SliceHeader,
    /// Byte offset of the slice's NAL (its header byte first, no start code) within
    /// [`AccessUnit::data`].
    pub offset: usize,
    /// Length of the NAL in bytes.
    pub len: usize,
}

/// One access unit: the NALs of exactly one coded picture, plus the parameter sets and SEI
/// that precede it, as an Annex B byte stream.
#[derive(Debug, Clone)]
pub struct AccessUnit {
    /// The whole access unit in Annex B form, ready to hand to a decoder.
    pub data: Vec<u8>,
    /// The picture's slices, in coded order.
    pub slices: Vec<SliceRef>,
    /// True when the picture is an IDR - a point a decoder can start from.
    pub idr: bool,
    /// True when the picture is a reference picture.
    pub reference: bool,
    /// Picture order count, i.e. presentation order within the coded video sequence.
    pub poc: Poc,
    /// The slice header of the picture's first slice.
    pub header: SliceHeader,
    /// The SPS in force for this picture.
    pub sps: Sps,
    /// The PPS in force for this picture.
    pub pps: Pps,
    /// True when this access unit carried a parameter set that changed the configuration,
    /// e.g. a mid-stream resolution change.
    pub config_changed: bool,
    /// The timestamp of the packet the picture's FIRST slice arrived in, when the caller
    /// gave one. Attributing it to the first slice rather than to whichever packet happened
    /// to complete the picture is what keeps timestamps on the right frame when a caller
    /// feeds one access unit per packet - the picture is only known to be complete when the
    /// NEXT one starts.
    pub pts: Option<i64>,
}

impl AccessUnit {
    /// The picture's presentation-order key.
    pub fn order(&self) -> i32 {
        self.poc.value(self.header.field_pic, self.header.bottom_field)
    }
}

/// Splits a NAL stream into access units, tracking parameter sets and picture order counts
/// as it goes.
///
/// A caller feeding whole MP4 samples still goes through this: the sample boundary is a
/// hint, not a guarantee (an MP4 sample can legally hold several slices of one picture, and
/// a raw `.h264` file has no boundaries at all), and the parameter-set and POC tracking is
/// needed either way.
#[derive(Debug, Default)]
pub struct AuSplitter {
    /// Parameter sets seen so far.
    pub sets: ParameterSets,
    poc: PocState,
    /// Annex B bytes of the access unit being built.
    pending: Vec<u8>,
    /// Metadata of the picture in `pending`, once its first slice has been seen.
    started: Option<StartedPicture>,
    /// Slices written into `pending` so far.
    slices: Vec<SliceRef>,
    /// Set when a parameter set in the pending access unit changed the configuration.
    config_changed: bool,
    scratch: Vec<u8>,
}

#[derive(Debug, Clone)]
struct StartedPicture {
    header: SliceHeader,
    sps: Sps,
    pps: Pps,
    poc: Poc,
    pts: Option<i64>,
}

impl AuSplitter {
    /// A splitter with no parameter sets.
    pub fn new() -> Self {
        Self::default()
    }

    /// Drop all state but keep the parameter sets: what a seek needs.
    pub fn flush_state(&mut self) {
        self.pending.clear();
        self.slices.clear();
        self.started = None;
        self.config_changed = false;
        self.poc.reset();
    }

    /// Feed Annex B bytes, appending every completed access unit to `out`.
    ///
    /// The last access unit in a stream stays pending until [`AuSplitter::finish`] is
    /// called: nothing but the end of the stream proves that a picture is complete.
    pub fn push_annex_b(&mut self, data: &[u8], out: &mut Vec<AccessUnit>) -> Result<()> {
        self.push_annex_b_at(data, None, out)
    }

    /// [`AuSplitter::push_annex_b`], tagging pictures that start here with `pts`.
    pub fn push_annex_b_at(
        &mut self,
        data: &[u8],
        pts: Option<i64>,
        out: &mut Vec<AccessUnit>,
    ) -> Result<()> {
        for raw in nal::split_annex_b(data) {
            self.push_nal_at(raw, pts, out)?;
        }
        Ok(())
    }

    /// Feed one NAL (no start code, header byte included).
    pub fn push_nal(&mut self, raw: &[u8], out: &mut Vec<AccessUnit>) -> Result<()> {
        self.push_nal_at(raw, None, out)
    }

    /// [`AuSplitter::push_nal`], tagging a picture that starts here with `pts`.
    pub fn push_nal_at(
        &mut self,
        raw: &[u8],
        pts: Option<i64>,
        out: &mut Vec<AccessUnit>,
    ) -> Result<()> {
        let n = Nal::parse(raw)?;
        match n.kind {
            nal::kind::SPS | nal::kind::PPS => {
                if self.started.is_some() {
                    self.emit(out)?;
                }
                let changed = self.sets.add_nal(&n, &mut self.scratch)?;
                self.config_changed |= changed;
                nal::write_annex_b(raw, &mut self.pending);
            }
            nal::kind::AUD | nal::kind::SEI | nal::kind::END_OF_SEQ | nal::kind::END_OF_STREAM => {
                if self.started.is_some() {
                    self.emit(out)?;
                }
                nal::write_annex_b(raw, &mut self.pending);
            }
            _ if n.is_slice() => {
                if !self.sets.is_ready() {
                    // A stream that starts mid-GOP (a seek, a live join) has slices before
                    // any parameter set. Dropping them is right: there is nothing to decode
                    // them against, and passing them on would make the backend guess.
                    return Ok(());
                }
                nal::rbsp_into(n.payload, &mut self.scratch);
                let scratch = std::mem::take(&mut self.scratch);
                let parsed = SliceHeader::parse(&scratch, n.kind, n.ref_idc, &self.sets);
                self.scratch = scratch;
                let (header, sps, pps) = parsed?;

                let new_picture = match &self.started {
                    None => true,
                    Some(cur) => header.starts_new_picture(&cur.header, &cur.sps, &cur.pps),
                };
                if new_picture {
                    if self.started.is_some() {
                        self.emit(out)?;
                    }
                    if header.idr {
                        self.poc.reset();
                    }
                    let poc = self.poc.advance(&header, &sps)?;
                    self.started =
                        Some(StartedPicture { header: header.clone(), sps, pps, poc, pts });
                }
                // The start code is four bytes; the slice's own NAL begins after it.
                let offset = self.pending.len() + 4;
                nal::write_annex_b(raw, &mut self.pending);
                self.slices.push(SliceRef { header, offset, len: raw.len() });
            }
            // Slice data partitions B and C, filler, SPS extension, auxiliary pictures:
            // pass them through as part of the current access unit without interpreting.
            _ => nal::write_annex_b(raw, &mut self.pending),
        }
        Ok(())
    }

    /// Emit whatever picture is still pending. Call at end of stream, and before a seek.
    pub fn finish(&mut self, out: &mut Vec<AccessUnit>) -> Result<()> {
        if self.started.is_some() {
            self.emit(out)?;
        } else {
            self.pending.clear();
            self.slices.clear();
            self.config_changed = false;
        }
        Ok(())
    }

    fn emit(&mut self, out: &mut Vec<AccessUnit>) -> Result<()> {
        let started = self
            .started
            .take()
            .ok_or(Error::State("access unit emitted with no picture in it"))?;
        let pts = started.pts;
        out.push(AccessUnit {
            data: std::mem::take(&mut self.pending),
            slices: std::mem::take(&mut self.slices),
            idr: started.header.idr,
            reference: started.header.nal_ref_idc != 0,
            poc: started.poc,
            header: started.header,
            sps: started.sps,
            pps: started.pps,
            config_changed: std::mem::take(&mut self.config_changed),
            pts,
        });
        Ok(())
    }
}
