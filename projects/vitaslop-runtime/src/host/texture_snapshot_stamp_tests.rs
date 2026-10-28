use super::*;
use std::cell::Cell;
use std::rc::Rc;

const PAGE: usize = 1 << 12;

/// Shared counters, readable while `GuestCtx` holds the memory mutably borrowed.
#[derive(Clone, Default)]
struct Counters {
    /// Times the epoch was advanced. The number this whole exercise is about.
    bumps: Rc<Cell<u32>>,
    /// Times the snapshot cache borrowed bytes to COMPARE them.
    borrows: Rc<Cell<u32>>,
    /// Times the epoch ran out and the map had to be wiped.
    wraps: Rc<Cell<u32>>,
}

/// Guest memory with a working dirty map: one byte per page holding the epoch of the last
/// store into it, and a one-byte epoch - the same shape, and the same `>=` comparison, as
/// `vitaslop-native`'s `SharedView` and the browser's.
struct StampedMemory {
    bytes: Vec<u8>,
    map: Vec<Cell<u8>>,
    epoch: Cell<u8>,
    c: Counters,
}

impl StampedMemory {
    fn new(len: usize, c: Counters) -> Self {
        StampedMemory {
            bytes: vec![0u8; len],
            map: (0..len.div_ceil(PAGE) + 1).map(|_| Cell::new(0)).collect(),
            epoch: Cell::new(1),
            c,
        }
    }

    /// A GUEST store: change the bytes and stamp the page, exactly as emitted code does.
    fn guest_store(&mut self, off: usize, value: u8) {
        self.bytes[off] = value;
        self.map[off >> 12].set(self.epoch.get());
    }
}

impl GuestMemory for StampedMemory {
    fn len(&self) -> usize {
        self.bytes.len()
    }
    fn stamp_dirty_pages(&self, off: usize, len: usize) {
        if len == 0 {
            return;
        }
        let shift = vitaslop_transpiler::DIRTY_SHIFT;
        let last = ((off + len - 1) >> shift).min(self.map.len().saturating_sub(1));
        for s in &self.map[(off >> shift).min(last)..=last] {
            s.set(self.epoch.get());
        }
    }
    fn read(&self, off: usize, buf: &mut [u8]) {
        buf.copy_from_slice(&self.bytes[off..off + buf.len()]);
    }
    fn write(&mut self, off: usize, bytes: &[u8]) {
        self.bytes[off..off + bytes.len()].copy_from_slice(bytes);
    }
    fn borrow(&self, off: usize, len: usize) -> Option<&[u8]> {
        self.c.borrows.set(self.c.borrows.get() + 1);
        self.bytes.get(off..off.checked_add(len)?)
    }
    fn dirty_runs_since(
        &self,
        off: usize,
        len: usize,
        stamp: u8,
        out: &mut Vec<(usize, usize)>,
    ) -> Option<()> {
        if len == 0 {
            return Some(());
        }
        let first = off >> 12;
        let last = (off + len - 1) >> 12;
        if last >= self.map.len() {
            return None;
        }
        let overhang = first > 0 && self.map[first - 1].get() >= stamp;
        let mut run: Option<(usize, usize)> = None;
        for page in first..=last {
            let dirty = self.map[page].get() >= stamp || (page == first && overhang);
            if !dirty {
                if let Some(r) = run.take() {
                    out.push(r);
                }
                continue;
            }
            let start = (page << 12).max(off) - off;
            let end = ((page << 12) + PAGE).min(off + len) - off;
            match run.as_mut() {
                Some(r) => r.1 = end,
                None => run = Some((start, end)),
            }
        }
        if let Some(r) = run {
            out.push(r);
        }
        Some(())
    }
    fn dirty_since(&self, off: usize, len: usize, stamp: u8) -> Option<bool> {
        if len == 0 {
            return Some(false);
        }
        // One page below too - see `GuestMemory::dirty_since`'s overhang note.
        let first = (off >> 12).saturating_sub(1);
        let last = ((off + len - 1) >> 12).min(self.map.len() - 1).max(first);
        Some(self.map[first..=last].iter().any(|s| s.get() >= stamp))
    }
    fn dirty_epoch(&self) -> Option<u8> {
        Some(self.epoch.get())
    }
    fn rebase_dirty_epoch(&self, floor: u8) -> Option<u8> {
        for s in &self.map {
            let v = s.get();
            s.set(if v >= floor { v - floor + 1 } else { 0 });
        }
        let cur = self.epoch.get();
        let next = if cur >= floor { cur - floor + 1 } else { 1 };
        self.epoch.set(next);
        Some(next)
    }
    fn bump_dirty_epoch(&self) -> Option<(u8, bool)> {
        self.c.bumps.set(self.c.bumps.get() + 1);
        let next = self.epoch.get().wrapping_add(1);
        if next == 0 || next == u8::MAX {
            self.c.wraps.set(self.c.wraps.get() + 1);
            for s in &self.map {
                s.set(0);
            }
            self.epoch.set(1);
            return Some((1, true));
        }
        self.epoch.set(next);
        Some((next, false))
    }
}

fn ctx_over<'a>(
    regs: &'a mut [u32; REG_COUNT],
    vfp: &'a mut [u32; VFP_ARG_COUNT],
    mem: &'a mut StampedMemory,
) -> GuestCtx<'a> {
    GuestCtx::new(regs, vfp, mem, 0)
}

/// >>> A SPARSE DRAW READS THE ROWS IT REFERENCES AND NOTHING ELSE, AND DRAWS THE SAME.
///
/// Five indices spanning 4,000 rows of a 4-byte stream: the window read would take 16 KB
/// for four distinct rows. Compacted, the draw's vertex buffer is those four rows in
/// ascending guest order, the indices name them, and every index still lands on the row
/// it named in the guest - which is the whole claim, and the reason this is a cost change
/// and not a picture change. A dense draw (every row used) stays a window: `gather` is
/// `None` and the indices are rebased exactly as before.
#[test]
fn a_sparse_draw_is_compacted_to_the_rows_it_references() {
    let c = Counters::default();
    let mut mem = StampedMemory::new(64 * PAGE, c.clone());
    let (mut regs, mut vfp) = ([0u32; REG_COUNT], [0u32; VFP_ARG_COUNT]);
    let mut snaps = TextureSnapshots::new();
    // Row v of the stream (page 12, stride 4) holds v as a little-endian u32.
    let v0 = 12 * PAGE as u32;
    for v in 0..4096u32 {
        mem.bytes[12 * PAGE + v as usize * 4..][..4].copy_from_slice(&v.to_le_bytes());
    }
    // u16 indices at page 10.
    let i0 = 10 * PAGE as u32;
    let sparse = [5u16, 3000, 5, 4000, 3001];
    for (k, i) in sparse.iter().enumerate() {
        mem.bytes[10 * PAGE + k * 2..][..2].copy_from_slice(&i.to_le_bytes());
    }
    let streams = [VertexStreamInfo {
        stride: 4,
        column: 4,
        per_instance: false,
    }];
    let base = [0u32];
    let (idx, rows) = {
        let ctx = ctx_over(&mut regs, &mut vfp, &mut mem);
        let idx = snaps.get_or_read_indices(&ctx, i0, sparse.len() * 2, 2);
        let rows = snaps.read_draw_vertices(
            &ctx,
            &idx,
            true,
            &streams,
            &[v0],
            &base,
            &RowLayout::columns(&streams, &base),
            4,
        );
        (idx, rows)
    };
    assert_eq!(
        idx.gather.as_deref(),
        Some(&[5u32, 3000, 3001, 4000][..]),
        "sorted unique rows"
    );
    assert_eq!((idx.first_vertex, idx.vertex_count), (0, 4));
    let rewritten: Vec<u16> = idx
        .bytes
        .chunks(2)
        .map(|c| u16::from_le_bytes([c[0], c[1]]))
        .collect();
    assert_eq!(
        rewritten,
        [0, 1, 0, 3, 2],
        "each index names its row's compact position"
    );
    assert_eq!(rows.len(), 16, "four rows of four bytes");
    for (k, i) in rewritten.iter().enumerate() {
        let row = u32::from_le_bytes(rows[*i as usize * 4..][..4].try_into().unwrap());
        assert_eq!(
            row, sparse[k] as u32,
            "index {k} still lands on guest row {}",
            sparse[k]
        );
    }
    // The same five rows named densely (0..5) is a window, not a compaction.
    let dense = [0u16, 1, 2, 3, 4];
    for (k, i) in dense.iter().enumerate() {
        mem.bytes[11 * PAGE + k * 2..][..2].copy_from_slice(&i.to_le_bytes());
    }
    let idx = {
        let ctx = ctx_over(&mut regs, &mut vfp, &mut mem);
        snaps.get_or_read_indices(&ctx, 11 * PAGE as u32, dense.len() * 2, 2)
    };
    assert!(idx.gather.is_none(), "a dense draw keeps its window");
    assert_eq!((idx.first_vertex, idx.vertex_count), (0, 5));
}

/// `sceGxmDrawInstanced` on the shape a baseball title's crowd has: a four-row quad
/// (per-vertex stream, u32 rows holding the row number), an index buffer whose first
/// `indexWrap` = 6 entries are the quad and whose remaining entries are stale, and a
/// per-instance stream of u16 rows. Three instances: 18 indices. The expansion is one
/// row per (instance, vertex) - instance k's rows carry per-instance row k and the quad's
/// own rows - and the stale indices are never read.
#[test]
fn an_instanced_draw_expands_one_row_per_instance_and_vertex() {
    let c = Counters::default();
    let mut mem = StampedMemory::new(64 * PAGE, c.clone());
    let (mut regs, mut vfp) = ([0u32; REG_COUNT], [0u32; VFP_ARG_COUNT]);
    let mut snaps = TextureSnapshots::new();
    let v0 = 12 * PAGE as u32;
    for v in 0..4u32 {
        mem.bytes[12 * PAGE + v as usize * 4..][..4].copy_from_slice(&(100 + v).to_le_bytes());
    }
    // Per-instance rows at page 13: 7000 + k.
    let p0 = 13 * PAGE as u32;
    for k in 0..3u16 {
        mem.bytes[13 * PAGE + k as usize * 2..][..2].copy_from_slice(&(7000 + k).to_le_bytes());
    }
    // Indices at page 10: the quad, then stale entries that would index far past it.
    let i0 = 10 * PAGE as u32;
    let raw = [
        0u16, 1, 2, 2, 3, 0, 0, 500, 9, 124, 0, 587, 3, 3, 700, 0, 12, 99,
    ];
    for (k, i) in raw.iter().enumerate() {
        mem.bytes[10 * PAGE + k * 2..][..2].copy_from_slice(&i.to_le_bytes());
    }
    let streams = [
        VertexStreamInfo {
            stride: 4,
            column: 4,
            per_instance: false,
        },
        VertexStreamInfo {
            stride: 2,
            column: 2,
            per_instance: true,
        },
    ];
    let base = [0u32, 4];
    let (idx, rows) = {
        let ctx = ctx_over(&mut regs, &mut vfp, &mut mem);
        snaps
            .read_instanced_draw(
                &ctx,
                i0,
                raw.len() * 2,
                2,
                6,
                &streams,
                &[v0, p0],
                &RowLayout::columns(&streams, &base),
                6,
            )
            .expect("twelve indices at a wrap of six is an instanced draw")
    };
    assert_eq!(idx.vertex_count, 12, "3 instances x 4 quad rows");
    assert_eq!(rows.len(), 12 * 6);
    let rewritten: Vec<u16> = idx
        .bytes
        .chunks(2)
        .map(|c| u16::from_le_bytes([c[0], c[1]]))
        .collect();
    assert_eq!(
        rewritten,
        [0, 1, 2, 2, 3, 0, 4, 5, 6, 6, 7, 4, 8, 9, 10, 10, 11, 8],
        "the wrapped quad, offset by four rows per instance"
    );
    for k in 0..3usize {
        for v in 0..4usize {
            let r = &rows[(k * 4 + v) * 6..][..6];
            assert_eq!(
                u32::from_le_bytes(r[..4].try_into().unwrap()),
                100 + v as u32,
                "instance {k} vertex {v} quad row"
            );
            assert_eq!(
                u16::from_le_bytes([r[4], r[5]]),
                7000 + k as u16,
                "instance {k} vertex {v} per-instance row"
            );
        }
    }
    // A wrap of 0, or no more indices than one wrap, is not an instanced draw.
    let ctx = ctx_over(&mut regs, &mut vfp, &mut mem);
    assert!(
        snaps
            .read_instanced_draw(
                &ctx,
                i0,
                raw.len() * 2,
                2,
                0,
                &streams,
                &[v0, p0],
                &RowLayout::columns(&streams, &base),
                6
            )
            .is_none()
    );
    assert!(
        snaps
            .read_instanced_draw(
                &ctx,
                i0,
                6 * 2,
                2,
                6,
                &streams,
                &[v0, p0],
                &RowLayout::columns(&streams, &base),
                6
            )
            .is_none()
    );
}

/// >>> A VERTEX BUFFER ON A GUEST STACK IS RE-READ AFTER A STORE THAT CARRIED NO MARK.
///
/// The transpiler leaves SP-relative stores unmarked and the host stamps the stacks at every
/// epoch instead (`GuestStacks`). So: bytes changed WITHOUT a stamp, as such a store does.
/// On a registered stack page the next scene's read must carry the new byte; the same
/// unstamped change on an ordinary page stays served from the snapshot - which is the
/// control that proves the first half is the stamping and not a cache that never hits.
#[test]
fn a_vertex_buffer_on_a_stack_is_reread_after_an_unmarked_store() {
    let c = Counters::default();
    let mut mem = StampedMemory::new(64 * PAGE, c.clone());
    let (mut regs, mut vfp) = ([0u32; REG_COUNT], [0u32; VFP_ARG_COUNT]);
    let stacks = GuestStacks::default();
    stacks.note(20 * PAGE as u32, 24 * PAGE as u32);
    let mut snaps = TextureSnapshots::with_stacks(stacks);
    let (on_stack, off_stack) = (21 * PAGE as u32 + 64, 30 * PAGE as u32 + 64);
    // Different contents, so the content index cannot share one buffer between them.
    mem.bytes[on_stack as usize..][..16].fill(7);
    mem.bytes[off_stack as usize..][..16].fill(8);
    let mut read = |snaps: &mut TextureSnapshots, mem: &mut StampedMemory, addr: u32| {
        snaps.begin_scene();
        let ctx = ctx_over(&mut regs, &mut vfp, mem);
        let v = snaps.get_or_read_vertices(&ctx, addr, 16);
        snaps.end_scene_for_test();
        v[0]
    };
    for (a, v) in [(on_stack, 7), (off_stack, 8)] {
        assert_eq!(read(&mut snaps, &mut mem, a), v);
        assert_eq!(read(&mut snaps, &mut mem, a), v, "established");
    }
    // An unmarked store (no `guest_store`): bytes only.
    mem.bytes[on_stack as usize] = 9;
    mem.bytes[off_stack as usize] = 9;
    assert_eq!(
        read(&mut snaps, &mut mem, on_stack),
        9,
        "a stack page reads as stored-into every scene"
    );
    assert_eq!(
        read(&mut snaps, &mut mem, off_stack),
        8,
        "an unstamped ordinary page is served from the snapshot - so the line above is the stamping"
    );
}

/// The stack list keeps ranges sorted and merges any that overlap or touch, so a title that
/// spawns a callback (a fresh stack each time) every frame does not grow the stamping.
#[test]
fn guest_stacks_coalesce_touching_ranges() {
    let s = GuestStacks::default();
    s.note(0x3000, 0x4000);
    s.note(0x1000, 0x2000);
    s.note(0x2000, 0x3000); // touches both: one range
    s.note(0x8000, 0x9000);
    s.note(0x8800, 0x8900); // inside
    s.note(0x5000, 0x5000); // empty
    assert_eq!(s.snapshot(), vec![(0x1000, 0x4000), (0x8000, 0x9000)]);
}

/// >>> THE GATHER MEMO FILES ON THE SECOND SIGHTING, ANSWERS ON THE THIRD, AND A GUEST
/// STORE INTO ANY STREAM UNFILES IT.
///
/// Frame one gathers and interns (a miss: nothing filed). Frame two gathers again, the
/// intern hits, and the memo is filed with this scene's stamps. Frame three hands back
/// the SAME buffer without gathering. A store into the second stream then makes the memo
/// invalid and the fourth read is a fresh gather carrying the stored byte.
#[test]
fn the_gather_memo_answers_from_the_third_sighting_and_a_store_unfiles_it() {
    let c = Counters::default();
    let mut mem = StampedMemory::new(64 * PAGE, c.clone());
    let (mut regs, mut vfp) = ([0u32; REG_COUNT], [0u32; VFP_ARG_COUNT]);
    let mut snaps = TextureSnapshots::new();
    let (a0, a1) = (12 * PAGE as u32, 13 * PAGE as u32);
    for v in 0..64usize {
        for b in 0..4 {
            mem.bytes[12 * PAGE + v * 4 + b] = (0x10 + v * 4 + b) as u8;
        }
        for b in 0..2 {
            mem.bytes[13 * PAGE + v * 2 + b] = (0xa0 + v * 2 + b) as u8;
        }
    }
    let streams = [
        VertexStreamInfo {
            stride: 4,
            column: 4,
            per_instance: false,
        },
        VertexStreamInfo {
            stride: 2,
            column: 2,
            per_instance: false,
        },
    ];
    let base = [0u32, 4];
    let idx = DrawIndices {
        bytes: Arc::from(&[][..]),
        first_vertex: 0,
        vertex_count: 64,
        gather: None,
    };
    let mut read = |snaps: &mut TextureSnapshots, mem: &mut StampedMemory| {
        snaps.begin_scene();
        let ctx = ctx_over(&mut regs, &mut vfp, mem);
        let v = snaps.read_draw_vertices(
            &ctx,
            &idx,
            false,
            &streams,
            &[a0, a1],
            &base,
            &RowLayout::columns(&streams, &base),
            6,
        );
        snaps.end_scene_for_test();
        v
    };
    let first = read(&mut snaps, &mut mem);
    assert_eq!(first.len(), 64 * 6);
    assert!(
        snaps.gather_memo.is_empty(),
        "a first sighting files nothing"
    );
    let second = read(&mut snaps, &mut mem);
    assert!(
        Arc::ptr_eq(&first, &second),
        "the intern hands back the held buffer"
    );
    assert_eq!(
        snaps.gather_memo.len(),
        1,
        "the second sighting files the memo"
    );
    // A byte changed BEHIND the dirty map (no store recorded): a gather would see it, the
    // memo trusts the map and must not - which is how the third read proves it never
    // gathered. (A real guest cannot write without the map seeing it; this is the probe.)
    mem.bytes[13 * PAGE + 10] = 0xEE;
    let third = read(&mut snaps, &mut mem);
    assert!(
        Arc::ptr_eq(&first, &third),
        "the third sighting is the memo's buffer, ungathered"
    );
    assert_ne!(third[5 * 6 + 4], 0xEE, "which the unrecorded byte proves");
    // Now the same store, RECORDED.
    mem.guest_store(13 * PAGE + 10, 0xEE);
    let fourth = read(&mut snaps, &mut mem);
    assert!(
        !Arc::ptr_eq(&first, &fourth),
        "a written stream is a fresh gather"
    );
    assert_eq!(fourth[5 * 6 + 4], 0xEE, "carrying the stored byte");
    assert_eq!(&fourth[..4], &first[..4], "and every untouched row");
}

/// The INSTANCED expansion memo, on the same terms as the gather memo above: nothing is
/// filed on a first sighting, the second files it, a third answers WITHOUT reading (proved
/// by a byte changed behind the dirty map), and a RECORDED store unfiles it.
///
/// The index words are in the proof as well as the streams, because `min`, the rows per
/// instance and the rebased index list are all functions of them - a memo that watched
/// only the streams would keep serving one wrap after the guest changed the quad.
#[test]
fn the_instanced_memo_answers_from_the_third_sighting_and_a_store_unfiles_it() {
    let c = Counters::default();
    let mut mem = StampedMemory::new(64 * PAGE, c.clone());
    let (mut regs, mut vfp) = ([0u32; REG_COUNT], [0u32; VFP_ARG_COUNT]);
    let mut snaps = TextureSnapshots::new();
    // Index words at page 11, the per-vertex stream at 12, the per-instance one at 13.
    let (ia, a0, a1) = (11 * PAGE as u32, 12 * PAGE as u32, 13 * PAGE as u32);
    for (i, w) in [0u16, 1, 2, 2, 3, 0].iter().enumerate() {
        mem.bytes[11 * PAGE + i * 2..][..2].copy_from_slice(&w.to_le_bytes());
    }
    for v in 0..16usize {
        mem.bytes[12 * PAGE + v * 4..][..4].copy_from_slice(&[
            (0x10 + v) as u8,
            0x11,
            0x12,
            0x13,
        ]);
        mem.bytes[13 * PAGE + v * 2..][..2].copy_from_slice(&[(0xa0 + v) as u8, 0xa1]);
    }
    let streams = [
        VertexStreamInfo {
            stride: 4,
            column: 4,
            per_instance: false,
        },
        VertexStreamInfo {
            stride: 2,
            column: 2,
            per_instance: true,
        },
    ];
    let base = [0u32, 4];
    // Six indices a wrap, 18 in all, so three instances of a four-vertex quad.
    let mut read = |snaps: &mut TextureSnapshots, mem: &mut StampedMemory| {
        snaps.begin_scene();
        let ctx = ctx_over(&mut regs, &mut vfp, mem);
        let out = snaps
            .read_instanced_draw(
                &ctx,
                ia,
                36,
                2,
                6,
                &streams,
                &[a0, a1],
                &RowLayout::columns(&streams, &base),
                6,
            )
            .expect("three instances of a four-vertex quad expand");
        snaps.end_scene_for_test();
        out
    };
    let (i1, v1) = read(&mut snaps, &mut mem);
    assert_eq!(
        v1.len(),
        3 * 4 * 6,
        "three instances of four rows at stride 6"
    );
    assert_eq!(i1.vertex_count, 12);
    assert!(
        snaps.instanced_memo.is_empty(),
        "a first sighting files nothing"
    );
    let (_, v2) = read(&mut snaps, &mut mem);
    assert!(
        Arc::ptr_eq(&v1, &v2),
        "the intern hands back the held buffer"
    );
    assert_eq!(
        snaps.instanced_memo.len(),
        1,
        "the second sighting files the memo"
    );
    // A byte changed BEHIND the dirty map: an expansion would see it, the memo trusts the
    // map and must not - which is how the third read proves it never read anything.
    mem.bytes[13 * PAGE + 2] = 0xEE;
    let (_, v3) = read(&mut snaps, &mut mem);
    assert!(
        Arc::ptr_eq(&v1, &v3),
        "the third sighting is the memo's buffer, unexpanded"
    );
    assert_ne!(v3[4 * 6 + 4], 0xEE, "which the unrecorded byte proves");
    // Now the same store, RECORDED.
    mem.guest_store(13 * PAGE + 2, 0xEE);
    let (_, v4) = read(&mut snaps, &mut mem);
    assert!(
        !Arc::ptr_eq(&v1, &v4),
        "a written stream is a fresh expansion"
    );
    assert_eq!(
        v4[4 * 6 + 4],
        0xEE,
        "carrying the stored byte into instance 1"
    );
    assert_eq!(&v4[..4], &v1[..4], "and every untouched row");
    // ...and the INDEX words are in the proof too.
    let (_, v5) = read(&mut snaps, &mut mem);
    assert!(Arc::ptr_eq(&v4, &v5), "and it re-files");
    mem.guest_store(11 * PAGE + 2, 0x03);
    let (i6, _) = read(&mut snaps, &mut mem);
    assert_eq!(
        i6.vertex_count, 12,
        "the wrap still names four rows an instance"
    );
    assert!(
        snaps.instanced_memo.len() <= 1,
        "a stored index word unfiles the entry rather than leaving two"
    );
}

/// A PER-ATTRIBUTE row: a stream laid out `U8x3` at 0 then a `u16` pair at 3 - the layout
/// that makes a pipeline unbindable, because the second attribute sits at 3 modulo 4. Each
/// attribute gets its own 4-aligned slot, the guest's bytes land in it, and the byte the
/// widened `uint8x4` fetch reads past a 3-byte attribute is DEFINED (zero) rather than
/// whatever the reused scratch last held - every cache downstream hashes the whole row.
#[test]
fn a_per_attribute_row_puts_each_attribute_on_its_own_aligned_slot() {
    let c = Counters::default();
    let mut mem = StampedMemory::new(64 * PAGE, c.clone());
    let (mut regs, mut vfp) = ([0u32; REG_COUNT], [0u32; VFP_ARG_COUNT]);
    let mut snaps = TextureSnapshots::new();
    // One stream, 8-byte rows: bytes 0..3 are the U8x3, bytes 3..7 the u16 pair.
    let a0 = 12 * PAGE as u32;
    for v in 0..4usize {
        for b in 0..8 {
            mem.bytes[12 * PAGE + v * 8 + b] = (0x10 * (v + 1) + b) as u8;
        }
    }
    // Two per-vertex streams so the draw actually GATHERS (a lone per-vertex stream binds
    // the guest's own buffer and is never laid out by us).
    let a1 = 13 * PAGE as u32;
    for v in 0..4usize {
        mem.bytes[13 * PAGE + v * 2] = (0xc0 + v) as u8;
        mem.bytes[13 * PAGE + v * 2 + 1] = 0xc9;
    }
    let streams = [
        VertexStreamInfo {
            stride: 8,
            column: 7,
            per_instance: false,
        },
        VertexStreamInfo {
            stride: 2,
            column: 2,
            per_instance: false,
        },
    ];
    // The row `set_vertex_program` builds for that: U8x3 at 0, the pair at 4, stream 1 at 8.
    let layout = RowLayout::new(vec![
        ScatterPiece {
            stream: 0,
            src: 0,
            dst: 0,
            len: 3,
        },
        ScatterPiece {
            stream: 0,
            src: 3,
            dst: 4,
            len: 4,
        },
        ScatterPiece {
            stream: 1,
            src: 0,
            dst: 8,
            len: 2,
        },
    ]);
    let _base = [0u32, 8];
    let stride = 12u32;
    // Dirty the scratch first, so a byte this scatter does not write is visibly stale.
    snaps.interleave_scratch = vec![0xEE; 4 * stride as usize];
    let rows = {
        let ctx = ctx_over(&mut regs, &mut vfp, &mut mem);
        snaps.gather_into_scratch(&ctx, &streams, &[a0, a1], &layout, stride, 0, 4)
    };
    assert_eq!(rows.len(), 4 * 12);
    for v in 0..4usize {
        let r = &rows[v * 12..v * 12 + 12];
        let g = |b: usize| (0x10 * (v + 1) + b) as u8;
        assert_eq!(
            &r[0..3],
            &[g(0), g(1), g(2)],
            "the U8x3 at its own slot, row {v}"
        );
        assert_eq!(
            r[3], 0,
            "the byte a widened uint8x4 fetch reads past it is DEFINED"
        );
        assert_eq!(
            &r[4..8],
            &[g(3), g(4), g(5), g(6)],
            "the pair moved off offset 3, row {v}"
        );
        assert_eq!(
            &r[8..10],
            &[(0xc0 + v) as u8, 0xc9],
            "the second stream, row {v}"
        );
        assert_eq!(&r[10..12], &[0, 0], "and the tail pad is DEFINED too");
    }
}

/// A PER-INSTANCE stream whose attribute sits DEEP inside its row - golf's background
/// quad, which is a 4-vertex clip-space quad whose only colour comes from byte 64 of a
/// 68-byte per-instance record. The packed row is 20 bytes (a 16-byte position column and
/// a 4-byte colour slot), so the piece reads src 64 of a row the gather must read WHOLE.
///
/// It is a test because the failure is invisible: the colour lands on whatever the reused
/// scratch last held, which is a plausible colour that CHANGES between frames.
#[test]
fn a_per_instance_attribute_past_the_packed_row_is_still_read_from_its_own_record() {
    let c = Counters::default();
    let mut mem = StampedMemory::new(64 * PAGE, c.clone());
    let (mut regs, mut vfp) = ([0u32; REG_COUNT], [0u32; VFP_ARG_COUNT]);
    let mut snaps = TextureSnapshots::new();
    // Stream 0: four 16-byte vertex rows. Stream 1: one 68-byte per-instance record whose
    // last four bytes are the colour.
    let (a0, a1) = (12 * PAGE as u32, 13 * PAGE as u32);
    for v in 0..4usize {
        for b in 0..16 {
            mem.bytes[12 * PAGE + v * 16 + b] = (0x10 * (v + 1) + b) as u8;
        }
    }
    for b in 0..68 {
        mem.bytes[13 * PAGE + b] = b as u8;
    }
    mem.bytes[13 * PAGE + 64..][..4].copy_from_slice(&[0x5d, 0x81, 0x4f, 0xff]);
    let streams = [
        VertexStreamInfo {
            stride: 16,
            column: 16,
            per_instance: false,
        },
        VertexStreamInfo {
            stride: 68,
            column: 68,
            per_instance: true,
        },
    ];
    let layout = RowLayout::new(vec![
        ScatterPiece {
            stream: 0,
            src: 0,
            dst: 0,
            len: 16,
        },
        ScatterPiece {
            stream: 1,
            src: 64,
            dst: 16,
            len: 4,
        },
    ]);
    let _base = [0u32, 16];
    let stride = 20u32;
    // Dirty the scratch, so a colour this scatter fails to write is visibly stale rather
    // than conveniently zero.
    snaps.interleave_scratch = vec![0xEE; 4 * stride as usize];
    let rows = {
        let ctx = ctx_over(&mut regs, &mut vfp, &mut mem);
        snaps.gather_into_scratch(&ctx, &streams, &[a0, a1], &layout, stride, 0, 4)
    };
    assert_eq!(rows.len(), 4 * 20);
    for v in 0..4usize {
        let r = &rows[v * 20..v * 20 + 20];
        assert_eq!(r[0], (0x10 * (v + 1)) as u8, "the position column, row {v}");
        assert_eq!(
            &r[16..20],
            &[0x5d, 0x81, 0x4f, 0xff],
            "the per-instance record's own colour, row {v}"
        );
    }
}

/// The compacted gather coalesces nearby rows into one read and splits far ones - and
/// either way every row is the guest's, for a two-stream draw with a per-instance stream.
#[test]
fn a_compacted_gather_reads_runs_and_repeats_the_instance_row() {
    let c = Counters::default();
    let mut mem = StampedMemory::new(64 * PAGE, c.clone());
    let (mut regs, mut vfp) = ([0u32; REG_COUNT], [0u32; VFP_ARG_COUNT]);
    let mut snaps = TextureSnapshots::new();
    // Stream 0: 4-byte rows at page 12 (row v = v); stream 1: a per-instance 2-byte row.
    let (a0, a1) = (12 * PAGE as u32, 13 * PAGE as u32);
    for v in 0..8192u32 {
        mem.bytes[12 * PAGE + v as usize * 4..][..4].copy_from_slice(&v.to_le_bytes());
    }
    mem.bytes[13 * PAGE..][..2].copy_from_slice(&[0xab, 0xcd]);
    let streams = [
        VertexStreamInfo {
            stride: 4,
            column: 4,
            per_instance: false,
        },
        VertexStreamInfo {
            stride: 2,
            column: 2,
            per_instance: true,
        },
    ];
    let base = [0u32, 4];
    // Rows 7 and 9 are one run (8 bytes apart); 5000 is a second read.
    let gather = [7u32, 9, 5000];
    let idx = DrawIndices {
        bytes: Arc::from(&[][..]),
        first_vertex: 0,
        vertex_count: 3,
        gather: Some(Arc::from(&gather[..])),
    };
    let rows = {
        let ctx = ctx_over(&mut regs, &mut vfp, &mut mem);
        snaps.read_draw_vertices(
            &ctx,
            &idx,
            false,
            &streams,
            &[a0, a1],
            &base,
            &RowLayout::columns(&streams, &base),
            6,
        )
    };
    assert_eq!(rows.len(), 18);
    for (k, g) in gather.iter().enumerate() {
        assert_eq!(
            u32::from_le_bytes(rows[k * 6..][..4].try_into().unwrap()),
            *g,
            "row {k}"
        );
        assert_eq!(
            &rows[k * 6 + 4..k * 6 + 6],
            &[0xab, 0xcd],
            "instance row repeats into row {k}"
        );
    }
}

/// >>> A RE-READ THAT TOUCHES ONLY THE DIRTY PAGES MUST ANSWER WHAT A FULL RE-READ WOULD.
///
/// `get_or_read` now asks the engine WHICH pages of a texture were stored into and reads
/// only those, because a browser census found 7.1% of the pages of a re-read range dirty
/// (under 2% for the large ones) - the rest of every such read was re-establishing bytes
/// the map already knew. That is only admissible if the answer is identical, so this
/// checks the three outcomes that matter on a multi-page texture:
///
/// - a page written with the SAME bytes: unchanged, and the caller keeps the buffer
///   IDENTITY its decode and upload caches are keyed on;
/// - a page written with different bytes: the new contents, with every untouched page
///   carried over exactly;
/// - nothing written at all: the existing stamps shortcut, with no read.
#[test]
fn a_dirty_page_reread_answers_what_a_full_reread_would() {
    let c = Counters::default();
    // Three pages, so there is something to leave alone.
    let len = 3 * PAGE;
    let mut mem = StampedMemory::new(len + PAGE, c.clone());
    for (i, b) in mem.bytes.iter_mut().enumerate() {
        *b = (i % 251) as u8;
    }
    let mut regs = [0u32; REG_COUNT];
    let mut vfp = [0u32; VFP_ARG_COUNT];
    let mut snaps = TextureSnapshots::new();

    // Scene 1: establish the texture.
    let first = {
        let ctx = ctx_over(&mut regs, &mut vfp, &mut mem);
        snaps.get_or_read(&ctx, 0, len)
    };
    assert_eq!(first.len(), len);

    // Scene 2: the guest rewrites ONE page with the bytes it already held.
    snaps.begin_scene();
    snaps.end_scene_for_test();
    let same = mem.bytes[PAGE];
    mem.guest_store(PAGE, same);
    let second = {
        let ctx = ctx_over(&mut regs, &mut vfp, &mut mem);
        snaps.get_or_read(&ctx, 0, len)
    };
    assert!(
        Arc::ptr_eq(&first, &second),
        "a page rewritten with the same bytes must not produce a new buffer - every \
         decode and upload cache downstream is keyed on this identity"
    );

    // Scene 3: the guest changes one byte of the middle page.
    snaps.begin_scene();
    snaps.end_scene_for_test();
    mem.guest_store(PAGE + 7, 0xAB);
    let third = {
        let ctx = ctx_over(&mut regs, &mut vfp, &mut mem);
        snaps.get_or_read(&ctx, 0, len)
    };
    assert!(
        !Arc::ptr_eq(&first, &third),
        "changed bytes are a new buffer"
    );
    assert_eq!(third[PAGE + 7], 0xAB, "the changed byte is the guest's");
    assert_eq!(
        &third[..PAGE],
        &mem.bytes[..PAGE],
        "the untouched page is carried over"
    );
    assert_eq!(
        &third[2 * PAGE..len],
        &mem.bytes[2 * PAGE..len],
        "and so is the page after it"
    );
    // The whole buffer is exactly guest memory, which is the only claim that matters.
    assert_eq!(&third[..], &mem.bytes[..len]);

    // Scene 4: nothing written. The stamps shortcut answers with no read at all.
    snaps.begin_scene();
    snaps.end_scene_for_test();
    let fourth = {
        let ctx = ctx_over(&mut regs, &mut vfp, &mut mem);
        snaps.get_or_read(&ctx, 0, len)
    };
    assert!(
        Arc::ptr_eq(&third, &fourth),
        "an untouched texture is the same buffer"
    );
}

/// >>> THE REGRESSION THIS EXISTS FOR: one epoch per SCENE, not one per snapshot.
///
/// The epoch is a single byte compared with `>=`, so it cannot wrap silently - a wrap zeroes
/// the whole map and invalidates every stamp taken before it. Advancing it once per
/// re-stamped texture spent all 254 usable values in HALF A FRAME of a real race (572
/// re-stamps a frame), so no stamp ever survived to prove anything and every texture fell
/// through to the full byte compare. The mechanism was, in effect, off.
#[test]
fn the_epoch_advances_once_per_scene_not_once_per_snapshot() {
    let c = Counters::default();
    let mut mem = StampedMemory::new(400 * PAGE, c.clone());
    let (mut regs, mut vfp) = ([0u32; REG_COUNT], [0u32; VFP_ARG_COUNT]);
    let mut snaps = TextureSnapshots::new();
    let ctx = ctx_over(&mut regs, &mut vfp, &mut mem);

    // 300 distinct textures a scene, over 4 scenes. Under the old rule that is 1,200
    // epoch values against the 254 that exist, so the map would be wiped four times over.
    for _scene in 0..4 {
        snaps.begin_scene();
        for t in 0..300u32 {
            snaps.get_or_read(&ctx, t * PAGE as u32, 64);
        }
    }
    assert_eq!(c.bumps.get(), 4, "one epoch per scene, four scenes");
    assert_eq!(
        c.wraps.get(),
        0,
        "254 values must comfortably outlast four scenes"
    );
}

/// An unchanged texture must be proved unchanged WITHOUT its bytes being compared, and the
/// caller must get back the very same buffer.
///
/// Buffer IDENTITY is not incidental: the renderer's decode and upload caches are keyed on
/// it, so handing back an equal-but-different `Arc` re-decodes and re-uploads a texture that
/// did not change.
#[test]
fn an_untouched_texture_is_proved_clean_without_a_compare() {
    let c = Counters::default();
    let mut mem = StampedMemory::new(64 * PAGE, c.clone());
    let (mut regs, mut vfp) = ([0u32; REG_COUNT], [0u32; VFP_ARG_COUNT]);
    let mut snaps = TextureSnapshots::new();
    let ctx = ctx_over(&mut regs, &mut vfp, &mut mem);

    snaps.begin_scene();
    let first = snaps.get_or_read(&ctx, 8 * PAGE as u32, 2048);
    let after_read = c.borrows.get();

    // Three more scenes with no guest store at all.
    for _ in 0..3 {
        snaps.begin_scene();
        let again = snaps.get_or_read(&ctx, 8 * PAGE as u32, 2048);
        assert!(
            Arc::ptr_eq(&first, &again),
            "an unchanged texture must be the SAME buffer"
        );
    }
    assert_eq!(
        c.borrows.get(),
        after_read,
        "the stamps must prove it clean - borrowing the bytes again IS the compare"
    );
}

/// And the shortcut must never hide a real change: a guest store into the texture's page
/// has to produce fresh bytes on the next scene.
///
/// This is the half that makes the test above meaningful. An implementation that always
/// answered "clean" would pass the first test perfectly and serve stale pixels forever.
#[test]
fn a_guest_store_is_never_missed() {
    let c = Counters::default();
    let mut mem = StampedMemory::new(64 * PAGE, c.clone());
    let (mut regs, mut vfp) = ([0u32; REG_COUNT], [0u32; VFP_ARG_COUNT]);
    let mut snaps = TextureSnapshots::new();

    let addr = 8 * PAGE as u32;
    let first = {
        let ctx = ctx_over(&mut regs, &mut vfp, &mut mem);
        snaps.begin_scene();
        snaps.get_or_read(&ctx, addr, 2048)
    };
    assert_eq!(first[7], 0);

    // The guest writes one byte of it, stamping its page with the current epoch.
    mem.guest_store(8 * PAGE + 7, 0xAB);

    let ctx = ctx_over(&mut regs, &mut vfp, &mut mem);
    snaps.begin_scene();
    let again = snaps.get_or_read(&ctx, addr, 2048);
    assert!(
        !Arc::ptr_eq(&first, &again),
        "a written texture must be re-read"
    );
    assert_eq!(again[7], 0xAB, "and it must carry the new byte");

    // Having been re-read, it must go back to being provably clean rather than paying the
    // compare for ever after.
    let settled = c.borrows.get();
    for _ in 0..3 {
        snaps.begin_scene();
        let stable = snaps.get_or_read(&ctx, addr, 2048);
        assert!(Arc::ptr_eq(&again, &stable));
    }
    assert_eq!(
        c.borrows.get(),
        settled,
        "it must re-stamp clean after the re-read"
    );
}

/// Static geometry must be read ONCE and then proved unchanged, handing back the same buffer.
#[test]
fn static_vertices_are_read_once_and_then_proved_clean() {
    let c = Counters::default();
    let mut mem = StampedMemory::new(64 * PAGE, c.clone());
    let (mut regs, mut vfp) = ([0u32; REG_COUNT], [0u32; VFP_ARG_COUNT]);
    let mut snaps = TextureSnapshots::new();
    let ctx = ctx_over(&mut regs, &mut vfp, &mut mem);
    let addr = 12 * PAGE as u32;

    snaps.begin_scene();
    let first = snaps.get_or_read_vertices(&ctx, addr, 4096);
    assert_eq!(first.len(), 4096);

    // Fifty draws across five scenes from the same untouched mesh.
    for _ in 0..5 {
        snaps.begin_scene();
        for _ in 0..10 {
            let again = snaps.get_or_read_vertices(&ctx, addr, 4096);
            assert!(
                Arc::ptr_eq(&first, &again),
                "static geometry must be the SAME buffer"
            );
        }
    }
}

/// >>> THE PROPERTY THAT DIFFERS FROM TEXTURES: geometry rewritten WITHIN a scene must be
/// seen, on the very next draw.
///
/// A texture bound by many draws is compared once a scene, on the argument that a title does
/// not rewrite a texture while drawing with it. Writing vertices and drawing them repeatedly
/// inside one scene is exactly how dynamic geometry works, so the vertex path deliberately
/// takes no once-a-scene shortcut. If it ever did, UI text and particles would render one
/// draw stale and the symptom would be geometry a frame behind itself.
/// >>> THE MULTI-STREAM GATHER, PINNED AGAINST THE THREE THINGS ITS REWRITE RELIES ON.
///
/// `gather_into_scratch` no longer zeroes the whole interleaved buffer before scattering
/// into it, on the ground that the streams TILE the packed row and `read_into` defines
/// every source byte. That is the largest single item in the capture (4.5 ms/frame,
/// 53% of `record_draw`, on a retail sports title), and getting it wrong shows up as
/// GARBAGE GEOMETRY from a previous draw rather than as an error - so the properties are
/// asserted rather than argued:
///
///   1. two per-vertex streams interleave into the packed row in `stream_base` order;
///   2. a PER-INSTANCE stream repeats its row 0 into every vertex;
///   3. a stream whose guest buffer runs SHORT contributes ZEROS past the end - and does
///      so over a REUSED scratch that is still holding a longer previous draw's bytes,
///      which is exactly the case a missing fill would leak.
#[test]
fn the_multi_stream_gather_tiles_the_packed_row_and_zero_fills_a_short_stream() {
    let c = Counters::default();
    let mut mem = StampedMemory::new(64 * PAGE, c.clone());
    let (mut regs, mut vfp) = ([0u32; REG_COUNT], [0u32; VFP_ARG_COUNT]);
    let mut snaps = TextureSnapshots::new();
    // Stream 0: 4 bytes a vertex, at page 12. Stream 1: 2 bytes a vertex, at page 13.
    let (a0, a1) = (12 * PAGE as u32, 13 * PAGE as u32);
    for v in 0..8usize {
        for b in 0..4 {
            mem.bytes[12 * PAGE + v * 4 + b] = (0x10 + v * 4 + b) as u8;
        }
        for b in 0..2 {
            mem.bytes[13 * PAGE + v * 2 + b] = (0xa0 + v * 2 + b) as u8;
        }
    }
    let streams = [
        VertexStreamInfo {
            stride: 4,
            column: 4,
            per_instance: false,
        },
        VertexStreamInfo {
            stride: 2,
            column: 2,
            per_instance: false,
        },
    ];
    let base = [0u32, 4];
    // A LONG draw first, so the scratch is left holding its bytes for the short one below.
    let long = {
        let ctx = ctx_over(&mut regs, &mut vfp, &mut mem);
        snaps.gather_into_scratch(
            &ctx,
            &streams,
            &[a0, a1],
            &RowLayout::columns(&streams, &base),
            6,
            0,
            8,
        )
    };
    assert_eq!(long.len(), 48);
    let expected = long.clone();
    for v in 0..8usize {
        assert_eq!(
            &long[v * 6..v * 6 + 4],
            &mem.bytes[12 * PAGE + v * 4..][..4],
            "row {v} stream 0"
        );
        assert_eq!(
            &long[v * 6 + 4..v * 6 + 6],
            &mem.bytes[13 * PAGE + v * 2..][..2],
            "row {v} stream 1"
        );
    }
    snaps.interleave_scratch = long;

    // PER-INSTANCE: stream 1 now steps by instance, so every vertex reads its row 0.
    let per_instance = [
        VertexStreamInfo {
            stride: 4,
            column: 4,
            per_instance: false,
        },
        VertexStreamInfo {
            stride: 2,
            column: 2,
            per_instance: true,
        },
    ];
    let inst = {
        let ctx = ctx_over(&mut regs, &mut vfp, &mut mem);
        snaps.gather_into_scratch(
            &ctx,
            &per_instance,
            &[a0, a1],
            &RowLayout::columns(&per_instance, &base),
            6,
            0,
            4,
        )
    };
    for v in 0..4usize {
        assert_eq!(
            &inst[v * 6 + 4..v * 6 + 6],
            &mem.bytes[13 * PAGE..][..2],
            "row {v} is instance 0"
        );
    }
    snaps.interleave_scratch = inst;

    // SHORT: stream 1 is bound past the end of guest memory, so its column must read as
    // zeros for every row - not as whatever the two longer draws above left in the scratch.
    let past_end = mem.bytes.len() as u32;
    let short = {
        let ctx = ctx_over(&mut regs, &mut vfp, &mut mem);
        snaps.gather_into_scratch(
            &ctx,
            &streams,
            &[a0, past_end],
            &RowLayout::columns(&streams, &base),
            6,
            0,
            4,
        )
    };
    for v in 0..4usize {
        assert_eq!(
            &short[v * 6..v * 6 + 4],
            &mem.bytes[12 * PAGE + v * 4..][..4],
            "row {v} stream 0"
        );
        assert_eq!(
            &short[v * 6 + 4..v * 6 + 6],
            &[0, 0],
            "row {v}: an unreadable stream must contribute ZEROS, not the last draw's bytes"
        );
    }
    snaps.interleave_scratch = short;

    // >>> AND THE MERGED READ AGREES WITH THE SEPARATE ONE.
    //
    // The three draws above are a PAGE apart, so `gather_span` declines them and they
    // exercise the per-stream path. A title that rebuilds its geometry into a rotating
    // arena writes its streams back to back, which is the case the merge exists for - so
    // the same two streams are laid out adjacently here and asserted to produce the same
    // interleaving. Without this the merge could be silently dead, or silently wrong.
    let adj0 = 20 * PAGE as u32;
    let adj1 = adj0 + 8 * 4; // immediately after stream 0's eight 4-byte rows
    for v in 0..8usize {
        for b in 0..4 {
            mem.bytes[20 * PAGE + v * 4 + b] = (0x10 + v * 4 + b) as u8;
        }
        for b in 0..2 {
            mem.bytes[20 * PAGE + 32 + v * 2 + b] = (0xa0 + v * 2 + b) as u8;
        }
    }
    assert_eq!(
        snaps.gather_span(
            &ctx_over(&mut regs, &mut vfp, &mut mem),
            &streams,
            &[adj0, adj1],
            0,
            8
        ),
        Some((adj0, 48)),
        "adjacent streams are exactly the case the merged read is for"
    );
    let merged = {
        let ctx = ctx_over(&mut regs, &mut vfp, &mut mem);
        snaps.gather_into_scratch(
            &ctx,
            &streams,
            &[adj0, adj1],
            &RowLayout::columns(&streams, &base),
            6,
            0,
            8,
        )
    };
    assert_eq!(
        merged, expected,
        "one read of the union must interleave exactly as two reads do"
    );
}

#[test]
fn vertices_rewritten_within_one_scene_are_seen_immediately() {
    let c = Counters::default();
    let mut mem = StampedMemory::new(64 * PAGE, c.clone());
    let (mut regs, mut vfp) = ([0u32; REG_COUNT], [0u32; VFP_ARG_COUNT]);
    let mut snaps = TextureSnapshots::new();
    let addr = 12 * PAGE as u32;

    {
        let ctx = ctx_over(&mut regs, &mut vfp, &mut mem);
        snaps.begin_scene();
        let v = snaps.get_or_read_vertices(&ctx, addr, 1024);
        assert_eq!(v[5], 0);
    }
    // NO new scene - the guest writes and draws again, as dynamic geometry does.
    for round in 1..=4u8 {
        mem.guest_store(12 * PAGE + 5, round);
        let ctx = ctx_over(&mut regs, &mut vfp, &mut mem);
        let v = snaps.get_or_read_vertices(&ctx, addr, 1024);
        assert_eq!(
            v[5], round,
            "a mid-scene rewrite must be visible to the next draw"
        );
    }
}

/// A rewrite that lands on IDENTICAL bytes must hand back the same buffer, not an equal one.
///
/// A title that rebuilds a buffer every frame from unchanged inputs is common, and the
/// renderer's packed-vertex cache keys on this buffer's identity - so a fresh `Arc` there
/// would miss a cache holding the byte-for-byte same geometry.
#[test]
fn an_identical_rewrite_keeps_the_same_buffer() {
    let c = Counters::default();
    let mut mem = StampedMemory::new(64 * PAGE, c.clone());
    let (mut regs, mut vfp) = ([0u32; REG_COUNT], [0u32; VFP_ARG_COUNT]);
    let mut snaps = TextureSnapshots::new();
    let addr = 12 * PAGE as u32;

    let first = {
        let ctx = ctx_over(&mut regs, &mut vfp, &mut mem);
        snaps.begin_scene();
        snaps.get_or_read_vertices(&ctx, addr, 1024)
    };
    // Rewrite the SAME value: the dirty map says "touched", the bytes say "identical".
    mem.guest_store(12 * PAGE + 5, 0);
    let ctx = ctx_over(&mut regs, &mut vfp, &mut mem);
    snaps.begin_scene();
    let again = snaps.get_or_read_vertices(&ctx, addr, 1024);
    assert!(
        Arc::ptr_eq(&first, &again),
        "identical bytes must keep the same buffer"
    );
}

/// >>> AN UNCHANGED PALETTED TEXTURE MUST KEEP ONE EXPANSION AND ONE IDENTITY.
///
/// A P8 texture is expanded through its palette into an ordinary RGBA8 image before
/// anything downstream sees it, and that buffer is host-built, so it cannot inherit the
/// snapshot layer's identity. It used to MINT a fresh one on every decode - which is a
/// cache key that changes when nothing has: measured on this title's menu, 20 textures
/// re-expanded and re-uploaded every frame over bytes the snapshot layer had just proved
/// unchanged.
///
/// The three properties, all here because they are one mechanism: unchanged indices and an
/// unchanged palette keep the SAME buffer and the same id; a rewritten PALETTE is different
/// texels and must produce a new one; and rewritten INDICES must too.
#[test]
fn an_unchanged_paletted_texture_keeps_its_expansion() {
    let c = Counters::default();
    let mut mem = StampedMemory::new(64 * PAGE, c.clone());
    let (mut regs, mut vfp) = ([0u32; REG_COUNT], [0u32; VFP_ARG_COUNT]);
    let mut snaps = TextureSnapshots::new();
    // A 4x4 P8 image (16 indices) and its 256-entry table, in separate pages.
    let (tex, pal) = (12 * PAGE as u32, 13 * PAGE as u32);
    for i in 0..16usize {
        mem.guest_store(tex as usize + i, (i % 4) as u8);
    }
    for e in 0..4usize {
        for (b, v) in [(0u8, 10u8), (1, 20), (2, 30), (3, 255)] {
            mem.guest_store(pal as usize + e * 4 + b as usize, v.wrapping_add(e as u8));
        }
    }
    // LINEAR (tex_type 3), 4x4, data at `tex`, palette in word 3 as the 64-byte-aligned
    // field vitasdk declares. The exact format is handed in, which is what the runtime does
    // whenever the guest's own `sceGxmTextureInitP8` call was seen.
    let words = [0u32, (3 << 29) | (3 << 12) | 3, tex, pal >> 6];
    let binding = TextureBinding {
        unit: 0,
        addr: 0x1000,
        words,
        from_precomputed: false,
    };
    let fmt = Some(crate::render::P8 << 24);

    let decode = |snaps: &mut TextureSnapshots,
                  mem: &mut StampedMemory,
                  regs: &mut [u32; REG_COUNT],
                  vfp: &mut [u32; VFP_ARG_COUNT]| {
        let ctx = ctx_over(regs, vfp, mem);
        snaps.begin_scene();
        decode_texture(&ctx, snaps, &binding, fmt, None, 0).expect("a P8 texture decodes")
    };

    let first = decode(&mut snaps, &mut mem, &mut regs, &mut vfp);
    let again = decode(&mut snaps, &mut mem, &mut regs, &mut vfp);
    assert_eq!(
        first.base_format,
        crate::render::U8U8U8U8,
        "expanded to RGBA8"
    );
    assert_eq!(
        first.pixels_id, again.pixels_id,
        "an unchanged expansion keeps its identity"
    );
    assert!(Arc::ptr_eq(&first.pixels, &again.pixels), "and its buffer");

    // The PALETTE moves under a texture whose own bytes never do.
    mem.guest_store(pal as usize + 1, 99);
    let repalette = decode(&mut snaps, &mut mem, &mut regs, &mut vfp);
    assert_ne!(
        first.pixels_id, repalette.pixels_id,
        "new colours are a new buffer"
    );
    assert_ne!(
        &first.pixels[..],
        &repalette.pixels[..],
        "and they are actually different"
    );

    // And the INDICES move.
    mem.guest_store(tex as usize + 5, 3);
    let reindex = decode(&mut snaps, &mut mem, &mut regs, &mut vfp);
    assert_ne!(
        repalette.pixels_id, reindex.pixels_id,
        "new indices are a new buffer"
    );
}

/// Freed guest memory must not keep serving its old geometry.
#[test]
fn a_freed_range_drops_its_vertex_snapshot() {
    let c = Counters::default();
    let mut mem = StampedMemory::new(64 * PAGE, c.clone());
    let (mut regs, mut vfp) = ([0u32; REG_COUNT], [0u32; VFP_ARG_COUNT]);
    let mut snaps = TextureSnapshots::new();
    let addr = 12 * PAGE as u32;

    let first = {
        let ctx = ctx_over(&mut regs, &mut vfp, &mut mem);
        snaps.begin_scene();
        snaps.get_or_read_vertices(&ctx, addr, 1024)
    };
    snaps.invalidate_range(addr, 1024);
    let ctx = ctx_over(&mut regs, &mut vfp, &mut mem);
    let again = snaps.get_or_read_vertices(&ctx, addr, 1024);
    assert!(
        !Arc::ptr_eq(&first, &again),
        "a freed range must be re-read"
    );
}

/// An address whose new bytes are recognised through the CONTENT index (they equal a buffer
/// held under another address) must not keep its OLD snapshot: the next read with no store
/// in between has to see the new bytes, not what the address held before.
#[test]
fn a_content_hit_drops_the_addresss_stale_vertex_snapshot() {
    let c = Counters::default();
    let mut mem = StampedMemory::new(64 * PAGE, c.clone());
    let (mut regs, mut vfp) = ([0u32; REG_COUNT], [0u32; VFP_ARG_COUNT]);
    let mut snaps = TextureSnapshots::new();
    let (a, b) = (12 * PAGE, 20 * PAGE);
    for i in 0..64 {
        mem.guest_store(b + i, i as u8 + 1);
    }
    {
        let ctx = ctx_over(&mut regs, &mut vfp, &mut mem);
        snaps.begin_scene();
        assert!(
            snaps
                .get_or_read_vertices(&ctx, a as u32, 64)
                .iter()
                .all(|&v| v == 0)
        );
        snaps.get_or_read_vertices(&ctx, b as u32, 64);
    }
    for i in 0..64 {
        mem.guest_store(a + i, i as u8 + 1);
    }
    {
        let ctx = ctx_over(&mut regs, &mut vfp, &mut mem);
        snaps.begin_scene();
        assert_eq!(
            snaps.get_or_read_vertices(&ctx, a as u32, 64)[5],
            6,
            "the new bytes, via the content index"
        );
    }
    let ctx = ctx_over(&mut regs, &mut vfp, &mut mem);
    snaps.begin_scene();
    let again = snaps.get_or_read_vertices(&ctx, a as u32, 64);
    assert_eq!(
        again[5], 6,
        "a later read with no store between must not hand back the address's OLD snapshot"
    );
}

/// A texture whose page is written EVERY scene must be re-read every scene - the mechanism
/// must not become sticky in the other direction.
#[test]
fn a_texture_written_every_scene_is_re_read_every_scene() {
    let c = Counters::default();
    let mut mem = StampedMemory::new(64 * PAGE, c.clone());
    let (mut regs, mut vfp) = ([0u32; REG_COUNT], [0u32; VFP_ARG_COUNT]);
    let mut snaps = TextureSnapshots::new();
    let addr = 8 * PAGE as u32;

    let mut last: Option<Arc<[u8]>> = None;
    for scene in 0..6u8 {
        mem.guest_store(8 * PAGE + 3, scene + 1);
        let ctx = ctx_over(&mut regs, &mut vfp, &mut mem);
        snaps.begin_scene();
        let got = snaps.get_or_read(&ctx, addr, 2048);
        assert_eq!(got[3], scene + 1, "scene {scene} must see its own write");
        if let Some(prev) = last {
            assert!(!Arc::ptr_eq(&prev, &got));
        }
        last = Some(got);
    }
}