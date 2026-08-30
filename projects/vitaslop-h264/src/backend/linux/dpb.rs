//! The decoded picture buffer, reference list construction, and reference marking.
//!
//! VA-API's H.264 entry point is STATELESS: the driver decodes exactly the picture it is
//! handed, against exactly the reference list it is handed, and keeps no state of its own
//! between pictures. Everything in clauses 8.2.4 (reference picture lists) and 8.2.5
//! (decoded reference picture marking) is therefore the caller's job - this file - and
//! getting it wrong does not fail loudly: it produces a picture referencing the wrong
//! frame, which looks like corruption, not like an error.
//!
//! Frame-coded pictures only. A field-coded stream is refused by the backend before it
//! reaches here, because the field pairing rules would double every list in this file and
//! there is no way to half-implement them honestly.

use super::va::{VAPictureH264, VASurfaceID, picture_flag};
use crate::bitstream::poc::Poc;
use crate::bitstream::slice::{Mmco, RefPicListMod, RefPicMarking, SliceType};
use crate::error::{Error, Result};

/// How a decoded picture is being used for reference.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum RefKind {
    /// Not a reference picture.
    Unused,
    /// A short-term reference picture.
    Short,
    /// A long-term reference picture.
    Long,
}

/// One picture in the buffer.
#[derive(Debug, Clone)]
pub struct DpbEntry {
    /// Surface the decoded picture lives in.
    pub surface: VASurfaceID,
    /// `frame_num` as coded.
    pub frame_num: u32,
    /// `FrameNumWrap` (8.2.4.1), recomputed for every picture that uses this as reference.
    pub frame_num_wrap: i32,
    /// `PicNum` for a frame picture, which is `FrameNumWrap`.
    pub pic_num: i32,
    /// `LongTermFrameIdx`, when long-term.
    pub long_term_frame_idx: u32,
    /// `LongTermPicNum` for a frame, which is `LongTermFrameIdx`.
    pub long_term_pic_num: i32,
    /// The picture's order counts.
    pub poc: Poc,
    /// The presentation-order key this picture was submitted with.
    pub key: i64,
    /// Reference state.
    pub reference: RefKind,
    /// Still to be handed to the caller.
    pub needed_for_output: bool,
}

impl DpbEntry {
    /// The VA-API view of this picture, as a reference-list entry.
    pub fn as_va(&self) -> VAPictureH264 {
        let mut flags = 0;
        match self.reference {
            RefKind::Short => flags |= picture_flag::SHORT_TERM_REFERENCE,
            RefKind::Long => flags |= picture_flag::LONG_TERM_REFERENCE,
            RefKind::Unused => {}
        }
        VAPictureH264 {
            picture_id: self.surface,
            frame_idx: match self.reference {
                RefKind::Long => self.long_term_frame_idx,
                _ => self.frame_num,
            },
            flags,
            TopFieldOrderCnt: self.poc.top,
            BottomFieldOrderCnt: self.poc.bottom,
            ..VAPictureH264::invalid()
        }
    }

    /// Presentation order key within the sequence.
    pub fn poc_value(&self) -> i32 {
        self.poc.top.min(self.poc.bottom)
    }

    /// True once the picture is neither a reference nor waiting to be output: its surface
    /// can be handed to another picture.
    pub fn is_free(&self) -> bool {
        self.reference == RefKind::Unused && !self.needed_for_output
    }
}

/// A picture the buffer has decided to output.
#[derive(Debug, Clone, Copy)]
pub struct OutputPicture {
    /// Surface to read the pixels from.
    pub surface: VASurfaceID,
    /// The key it was submitted with, which becomes the frame's timestamp.
    pub key: i64,
}

/// The decoded picture buffer.
#[derive(Debug, Default)]
pub struct Dpb {
    /// Pictures held, in no particular order.
    pub frames: Vec<DpbEntry>,
    /// `max_dec_frame_buffering`: how many frames may be held at once.
    pub capacity: usize,
    /// How many frames may be held back for reordering before one must be output.
    pub max_reorder: usize,
    /// `max_num_ref_frames`.
    pub max_refs: usize,
    /// `MaxLongTermFrameIdx`, or -1 for "no long-term frames".
    pub max_long_term_frame_idx: i32,
}

impl Dpb {
    /// Size the buffer from a sequence parameter set.
    pub fn configure(&mut self, capacity: usize, max_reorder: usize, max_refs: usize) {
        self.capacity = capacity.max(1);
        self.max_reorder = max_reorder;
        self.max_refs = max_refs;
    }

    /// Drop everything without outputting it (a seek, or `no_output_of_prior_pics`).
    pub fn clear(&mut self) {
        self.frames.clear();
        self.max_long_term_frame_idx = -1;
    }

    /// Output everything still held, in presentation order, and empty the buffer.
    pub fn flush(&mut self, out: &mut Vec<OutputPicture>) {
        let mut pending: Vec<_> = self
            .frames
            .iter()
            .filter(|f| f.needed_for_output)
            .map(|f| (f.poc_value(), OutputPicture { surface: f.surface, key: f.key }))
            .collect();
        pending.sort_by_key(|(poc, _)| *poc);
        out.extend(pending.into_iter().map(|(_, p)| p));
        self.frames.clear();
        self.max_long_term_frame_idx = -1;
    }

    /// 8.2.4.1: recompute `PicNum`/`LongTermPicNum` relative to the picture being decoded.
    pub fn update_pic_nums(&mut self, current_frame_num: u32, max_frame_num: u32) {
        for f in &mut self.frames {
            match f.reference {
                RefKind::Short => {
                    f.frame_num_wrap = if f.frame_num > current_frame_num {
                        f.frame_num as i32 - max_frame_num as i32
                    } else {
                        f.frame_num as i32
                    };
                    f.pic_num = f.frame_num_wrap;
                }
                RefKind::Long => f.long_term_pic_num = f.long_term_frame_idx as i32,
                RefKind::Unused => {}
            }
        }
    }

    /// 8.2.4.2: the initial reference picture lists for one slice, before modification.
    ///
    /// Returns `(list0, list1)` as indices into [`Dpb::frames`].
    pub fn initial_lists(&self, slice_type: SliceType, current_poc: i32) -> (Vec<usize>, Vec<usize>) {
        let short: Vec<usize> = self
            .frames
            .iter()
            .enumerate()
            .filter(|(_, f)| f.reference == RefKind::Short)
            .map(|(i, _)| i)
            .collect();
        let mut long: Vec<usize> = self
            .frames
            .iter()
            .enumerate()
            .filter(|(_, f)| f.reference == RefKind::Long)
            .map(|(i, _)| i)
            .collect();
        long.sort_by_key(|&i| self.frames[i].long_term_pic_num);

        if slice_type != SliceType::B {
            // 8.2.4.2.1: P and SP slices - short-term by descending PicNum, then long-term.
            let mut list0 = short;
            list0.sort_by(|&a, &b| self.frames[b].pic_num.cmp(&self.frames[a].pic_num));
            list0.extend(long);
            return (list0, Vec::new());
        }

        // 8.2.4.2.3: B slices - list 0 counts down from the current picture and then up,
        // list 1 the other way round.
        let mut before: Vec<usize> =
            short.iter().copied().filter(|&i| self.frames[i].poc_value() < current_poc).collect();
        let mut after: Vec<usize> =
            short.iter().copied().filter(|&i| self.frames[i].poc_value() > current_poc).collect();
        before.sort_by(|&a, &b| self.frames[b].poc_value().cmp(&self.frames[a].poc_value()));
        after.sort_by(|&a, &b| self.frames[a].poc_value().cmp(&self.frames[b].poc_value()));

        let mut list0: Vec<usize> = before.iter().copied().chain(after.iter().copied()).collect();
        let mut list1: Vec<usize> = after.iter().copied().chain(before.iter().copied()).collect();
        list0.extend(long.iter().copied());
        list1.extend(long.iter().copied());
        // 8.2.4.2.3: when the two lists come out identical and there is more than one
        // entry, the first two of list 1 are swapped.
        if list0.len() > 1 && list0 == list1 {
            list1.swap(0, 1);
        }
        (list0, list1)
    }

    /// 8.2.4.3.1: apply one list's modification commands.
    pub fn modify_list(
        &self,
        list: &mut Vec<usize>,
        mods: &[RefPicListMod],
        num_active: usize,
        current_frame_num: u32,
        max_frame_num: u32,
    ) -> Result<()> {
        if mods.is_empty() {
            list.truncate(num_active);
            return Ok(());
        }
        let max_pic_num = max_frame_num as i32;
        let current_pic_num = current_frame_num as i32;
        let mut pred = current_pic_num;
        let mut index = 0usize;

        for command in mods {
            let picked = match command.idc {
                0 | 1 => {
                    let abs_diff = command.value as i32 + 1;
                    let mut no_wrap = if command.idc == 0 {
                        let v = pred - abs_diff;
                        if v < 0 { v + max_pic_num } else { v }
                    } else {
                        let v = pred + abs_diff;
                        if v >= max_pic_num { v - max_pic_num } else { v }
                    };
                    pred = no_wrap;
                    if no_wrap > current_pic_num {
                        no_wrap -= max_pic_num;
                    }
                    self.frames
                        .iter()
                        .position(|f| f.reference == RefKind::Short && f.pic_num == no_wrap)
                }
                2 => self.frames.iter().position(|f| {
                    f.reference == RefKind::Long && f.long_term_pic_num == command.value as i32
                }),
                other => {
                    return Err(Error::bitstream(format!(
                        "modification_of_pic_nums_idc {other} in a reference list"
                    )));
                }
            };
            let picked = picked.ok_or_else(|| {
                Error::bitstream(
                    "a reference list modification names a picture that is not in the buffer",
                )
            })?;

            // 8.2.4.3.1: put the named picture at `index`, shifting the rest down, and
            // drop the copy of it that was already somewhere else in the list.
            let at = index.min(list.len());
            list.insert(at, picked);
            index = at + 1;
            let mut cleaned = Vec::with_capacity(list.len());
            for (i, &e) in list.iter().enumerate() {
                if e == picked && i != at {
                    continue;
                }
                cleaned.push(e);
            }
            *list = cleaned;
        }
        list.truncate(num_active);
        Ok(())
    }

    /// 8.2.5: mark references after a picture has been decoded, then insert it.
    ///
    /// `current` is the freshly decoded picture; on return it is in the buffer.
    pub fn mark_and_insert(
        &mut self,
        mut current: DpbEntry,
        marking: Option<&RefPicMarking>,
        idr: bool,
        max_frame_num: u32,
        out: &mut Vec<OutputPicture>,
    ) -> Result<()> {
        if idr {
            let no_output = marking.map(|m| m.no_output_of_prior_pics).unwrap_or(false);
            if no_output {
                self.clear();
            } else {
                self.flush(out);
            }
            if marking.map(|m| m.long_term_reference).unwrap_or(false) {
                current.reference = RefKind::Long;
                current.long_term_frame_idx = 0;
                self.max_long_term_frame_idx = 0;
            } else if current.reference != RefKind::Unused {
                current.reference = RefKind::Short;
                self.max_long_term_frame_idx = -1;
            }
            self.frames.push(current);
            self.bump(out);
            return Ok(());
        }

        let adaptive = marking.map(|m| m.adaptive).unwrap_or(false);
        if adaptive {
            let commands = marking.map(|m| m.mmco.as_slice()).unwrap_or(&[]);
            self.apply_mmco(commands, &mut current, max_frame_num)?;
        } else if current.reference != RefKind::Unused {
            self.sliding_window();
        }
        self.frames.push(current);
        self.bump(out);
        Ok(())
    }

    /// 8.2.5.3: sliding window - the oldest short-term reference falls out.
    fn sliding_window(&mut self) {
        let refs = self
            .frames
            .iter()
            .filter(|f| matches!(f.reference, RefKind::Short | RefKind::Long))
            .count();
        if refs < self.max_refs.max(1) {
            return;
        }
        let oldest = self
            .frames
            .iter()
            .enumerate()
            .filter(|(_, f)| f.reference == RefKind::Short)
            .min_by_key(|(_, f)| f.frame_num_wrap)
            .map(|(i, _)| i);
        if let Some(i) = oldest {
            self.frames[i].reference = RefKind::Unused;
        }
        self.drop_free();
    }

    /// 8.2.5.4: the adaptive marking commands.
    fn apply_mmco(
        &mut self,
        commands: &[Mmco],
        current: &mut DpbEntry,
        max_frame_num: u32,
    ) -> Result<()> {
        let current_pic_num = current.frame_num as i32;
        for c in commands {
            match c.op {
                1 => {
                    // Mark a short-term picture unused.
                    let pic_num = current_pic_num - (c.difference_of_pic_nums_minus1 as i32 + 1);
                    let pic_num =
                        if pic_num < 0 { pic_num + max_frame_num as i32 } else { pic_num };
                    self.unmark_short(pic_num);
                }
                2 => self.unmark_long(c.long_term_pic_num as i32),
                3 => {
                    // Turn a short-term picture into a long-term one.
                    let pic_num = current_pic_num - (c.difference_of_pic_nums_minus1 as i32 + 1);
                    let pic_num =
                        if pic_num < 0 { pic_num + max_frame_num as i32 } else { pic_num };
                    self.unmark_long_idx(c.long_term_frame_idx);
                    if let Some(f) = self
                        .frames
                        .iter_mut()
                        .find(|f| f.reference == RefKind::Short && f.pic_num == pic_num)
                    {
                        f.reference = RefKind::Long;
                        f.long_term_frame_idx = c.long_term_frame_idx;
                        f.long_term_pic_num = c.long_term_frame_idx as i32;
                    }
                }
                4 => {
                    self.max_long_term_frame_idx = c.max_long_term_frame_idx_plus1 as i32 - 1;
                    let limit = self.max_long_term_frame_idx;
                    for f in &mut self.frames {
                        if f.reference == RefKind::Long && f.long_term_frame_idx as i32 > limit {
                            f.reference = RefKind::Unused;
                        }
                    }
                }
                5 => {
                    // "Mark everything unused": the sequence restarts as if from an IDR,
                    // except that the current picture stays.
                    for f in &mut self.frames {
                        f.reference = RefKind::Unused;
                    }
                    self.max_long_term_frame_idx = -1;
                    current.frame_num = 0;
                    current.pic_num = 0;
                    current.frame_num_wrap = 0;
                }
                6 => {
                    self.unmark_long_idx(c.long_term_frame_idx);
                    current.reference = RefKind::Long;
                    current.long_term_frame_idx = c.long_term_frame_idx;
                    current.long_term_pic_num = c.long_term_frame_idx as i32;
                }
                other => {
                    return Err(Error::bitstream(format!(
                        "memory_management_control_operation {other}"
                    )));
                }
            }
        }
        self.drop_free();
        Ok(())
    }

    fn unmark_short(&mut self, pic_num: i32) {
        if let Some(f) =
            self.frames.iter_mut().find(|f| f.reference == RefKind::Short && f.pic_num == pic_num)
        {
            f.reference = RefKind::Unused;
        }
    }

    fn unmark_long(&mut self, long_term_pic_num: i32) {
        if let Some(f) = self
            .frames
            .iter_mut()
            .find(|f| f.reference == RefKind::Long && f.long_term_pic_num == long_term_pic_num)
        {
            f.reference = RefKind::Unused;
        }
    }

    fn unmark_long_idx(&mut self, idx: u32) {
        for f in &mut self.frames {
            if f.reference == RefKind::Long && f.long_term_frame_idx == idx {
                f.reference = RefKind::Unused;
            }
        }
    }

    /// C.4.5.3 "bumping": output the smallest-POC picture while the buffer holds more than
    /// it may.
    pub fn bump(&mut self, out: &mut Vec<OutputPicture>) {
        loop {
            let waiting = self.frames.iter().filter(|f| f.needed_for_output).count();
            let over_reorder = waiting > self.max_reorder;
            let over_capacity = self.frames.len() > self.capacity;
            if !over_reorder && !over_capacity {
                break;
            }
            let next = self
                .frames
                .iter()
                .enumerate()
                .filter(|(_, f)| f.needed_for_output)
                .min_by_key(|(_, f)| f.poc_value())
                .map(|(i, _)| i);
            let Some(i) = next else {
                // Nothing left to output but the buffer is still over capacity: every entry
                // is a reference picture. Dropping one here would corrupt later pictures,
                // so the buffer is allowed to run one over instead.
                break;
            };
            self.frames[i].needed_for_output = false;
            out.push(OutputPicture { surface: self.frames[i].surface, key: self.frames[i].key });
            self.drop_free();
        }
    }

    /// Forget entries that are neither referenced nor waiting for output.
    pub fn drop_free(&mut self) {
        self.frames.retain(|f| !f.is_free());
    }

    /// Surfaces the buffer is holding, so the backend does not hand one out twice.
    pub fn holds(&self, surface: VASurfaceID) -> bool {
        self.frames.iter().any(|f| f.surface == surface)
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    fn entry(surface: u32, frame_num: u32, poc: i32, reference: RefKind) -> DpbEntry {
        DpbEntry {
            surface,
            frame_num,
            frame_num_wrap: frame_num as i32,
            pic_num: frame_num as i32,
            long_term_frame_idx: 0,
            long_term_pic_num: 0,
            poc: Poc { top: poc, bottom: poc },
            key: poc as i64,
            reference,
            needed_for_output: true,
        }
    }

    #[test]
    fn p_list_counts_down_from_the_current_picture() {
        let mut dpb = Dpb::default();
        dpb.configure(4, 2, 4);
        dpb.frames.push(entry(1, 1, 2, RefKind::Short));
        dpb.frames.push(entry(2, 3, 6, RefKind::Short));
        dpb.frames.push(entry(3, 2, 4, RefKind::Short));
        let (list0, list1) = dpb.initial_lists(SliceType::P, 8);
        let nums: Vec<i32> = list0.iter().map(|&i| dpb.frames[i].pic_num).collect();
        assert_eq!(nums, vec![3, 2, 1]);
        assert!(list1.is_empty());
    }

    #[test]
    fn b_lists_are_mirror_images() {
        let mut dpb = Dpb::default();
        dpb.configure(4, 2, 4);
        dpb.frames.push(entry(1, 0, 0, RefKind::Short));
        dpb.frames.push(entry(2, 1, 8, RefKind::Short));
        let (list0, list1) = dpb.initial_lists(SliceType::B, 4);
        assert_eq!(dpb.frames[list0[0]].poc_value(), 0);
        assert_eq!(dpb.frames[list0[1]].poc_value(), 8);
        assert_eq!(dpb.frames[list1[0]].poc_value(), 8);
        assert_eq!(dpb.frames[list1[1]].poc_value(), 0);
    }

    #[test]
    fn sliding_window_drops_the_oldest_short_term_reference() {
        let mut dpb = Dpb::default();
        dpb.configure(4, 0, 2);
        dpb.frames.push(entry(1, 0, 0, RefKind::Short));
        dpb.frames.push(entry(2, 1, 2, RefKind::Short));
        for f in &mut dpb.frames {
            f.needed_for_output = false;
        }
        let mut out = Vec::new();
        let current = entry(3, 2, 4, RefKind::Short);
        dpb.mark_and_insert(current, None, false, 16, &mut out).unwrap();
        let surfaces: Vec<u32> = dpb.frames.iter().map(|f| f.surface).collect();
        assert!(!surfaces.contains(&1), "the oldest reference should have fallen out");
        assert!(surfaces.contains(&2) && surfaces.contains(&3));
    }

    #[test]
    fn an_idr_outputs_everything_it_replaces() {
        let mut dpb = Dpb::default();
        dpb.configure(4, 2, 2);
        dpb.frames.push(entry(1, 0, 0, RefKind::Short));
        dpb.frames.push(entry(2, 1, 2, RefKind::Short));
        let mut out = Vec::new();
        let marking = RefPicMarking::default();
        dpb.mark_and_insert(entry(3, 0, 0, RefKind::Short), Some(&marking), true, 16, &mut out)
            .unwrap();
        assert_eq!(out.len(), 2, "both held pictures are output before the IDR");
        assert_eq!(out[0].surface, 1);
        assert_eq!(dpb.frames.len(), 1);
    }

    #[test]
    fn bumping_emits_in_presentation_order() {
        let mut dpb = Dpb::default();
        dpb.configure(4, 0, 4);
        let mut out = Vec::new();
        for (surface, poc) in [(1u32, 4i32), (2, 0), (3, 2)] {
            let mut e = entry(surface, surface, poc, RefKind::Unused);
            e.needed_for_output = true;
            dpb.frames.push(e);
        }
        dpb.bump(&mut out);
        let order: Vec<u32> = out.iter().map(|p| p.surface).collect();
        assert_eq!(order, vec![2, 3, 1]);
    }
}
