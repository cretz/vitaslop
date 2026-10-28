use super::*;

/// The `[start, end)` byte ranges of a `len`-byte window that `cover` (sorted, disjoint, relative
/// to the window) leaves out.
fn uncovered(len: usize, cover: &[(u32, u32)]) -> impl Iterator<Item = (usize, usize)> + '_ {
    let mut from = 0usize;
    cover
        .iter()
        .map(move |&(s, t)| ((s as usize).min(len), (t as usize).min(len)))
        .chain(std::iter::once((len, len)))
        .filter_map(move |(s, t)| {
            let gap = (s > from).then_some((from, s));
            from = from.max(t);
            gap
        })
}

impl TextureSnapshots {
    /// A cache that knows no stacks - the unit tests' form; the engine builds one with its
    /// stacks ([`Self::with_stacks`]).
    #[cfg(test)]
    pub(super) fn new() -> Self {
        Self::default()
    }

    /// A cache that stamps `stacks` - the engine's own list, shared with [`VitaState`].
    pub(super) fn with_stacks(stacks: GuestStacks) -> Self {
        TextureSnapshots {
            stacks,
            ..Self::default()
        }
    }

    /// Stamp every known stack with the epoch just handed out, so a stack page reads as
    /// stored-into for every stamp held - see [`GuestMemory::stamp_dirty_pages`]. Called after
    /// EVERY successful [`GuestCtx::bump_dirty_epoch`] in this cache, wrapped or not: a wrap
    /// zeroed the map, and these pages must not read as clean against the new stamps either.
    pub(super) fn stamp_stacks(&self, ctx: &GuestCtx) {
        for (lo, hi) in self.stacks.snapshot() {
            ctx.stamp_dirty_pages(lo, (hi - lo) as usize);
        }
    }

    /// The current bytes of the texture of `len` bytes at guest address `addr`,
    /// reusing the retained snapshot when guest memory still holds the same bytes.
    pub(super) fn get_or_read(&mut self, ctx: &GuestCtx, addr: u32, len: usize) -> Arc<[u8]> {
        let _owner = vitaslop_platform::heap::owner_scope(vitaslop_platform::heap::Owner::TextureSnapshot);
        // One epoch per scene, taken at the scene's first snapshot - see `restamp` for why this
        // is not per texture, and what it cost when it was.
        self.arm_scene_stamp(ctx);
        // >>> A USE, RECORDED BEFORE ANY OF THE LADDER BELOW. This is what makes eviction
        // possible at all: without it the only orders available are insertion and hash order,
        // and neither says which textures the title is still drawing. Every exit from this
        // function is a use of `(addr, len)` - the shortcut returns as much as the re-read - so
        // the stamp goes at the top rather than being repeated down each of the six paths.
        self.used.insert((addr, len), self.frame_seq);
        // The retained buffer, kept past the borrow below so the re-read at the bottom can
        // hand back the SAME `Arc` when the bytes turn out to be identical - see there.
        let mut retained: Option<Arc<[u8]>> = None;
        // CLONED out of the map rather than borrowed from it: the page-granular re-read below
        // needs `&mut self` (it keeps a scratch buffer and re-stamps), and an `Arc` clone is a
        // refcount where holding the borrow is a restructuring of everything under it.
        let existing = self.entries.get(&(addr, len)).cloned();
        if let Some(p) = existing {
            let p = &p;
            retained = Some(p.clone());
            // Already compared this scene: the guest cannot have run since (a host
            // call is not preemptible), so nothing can have changed.
            if !self.checked_this_scene.insert((addr, len)) {
                return p.clone();
            }
            // >>> OPT-IN ONLY (`VITASLOP_TEXTURE_CHECK=frame`): compare a texture never
            // seen to change once per FRAME instead of once per SCENE.
            //
            // MEASURED on a live race with `bench --at`: the compare below is 40.2% of the
            // whole frame - 1332 ms of a 3.31 s window - moving 116.8 MB per frame at
            // 10.5 GB/s. It already runs at memory bandwidth, so the only thing left to
            // change is how often it runs, and a frame here is eleven scenes. Doing it per
            // frame took the title from 36.8 to 47.8 fps.
            //
            // It is NOT the default, because it is not exact and this emulator's job is to
            // be exact. A texture that changes mid-frame for the FIRST time is seen one
            // scene late (per-scene from then on, since a change puts it in `mutable`
            // permanently) - one wrong frame, which is a visible artefact and not a
            // rounding error. The strict cadence is what a claim of hardware accuracy
            // rests on, so the fast path has to be asked for, and says so when taken.
            if texture_check_per_frame()
                && !self.mutable.contains(&(addr, len))
                && !self.checked_this_frame.insert((addr, len))
            {
                report_texture_check_per_frame();
                return p.clone();
            }
            // >>> THE EXACT SHORTCUT: ask the engine whether the guest stored here at all.
            //
            // On an engine that stamps its stores ([`GuestMemory::dirty_since`]), a
            // texture no store has touched since these bytes were read CANNOT have
            // changed. That is a proof, not a sample - the same guarantee the memcmp
            // below gives - for the cost of reading one byte per 4 KB page instead of
            // every byte of the texture twice. A missing stamp or a `None` answer is
            // NOT "clean": both fall through to the compare.
            //
            // Charged to the compare phase, because it IS the compare on that engine -
            // filing it elsewhere would make the phase look solved rather than moved.
            let untouched = texture_check() != TextureCheck::Bytes && {
                let _t = crate::perf::scope(crate::perf::Phase::DrawTextureCompare);
                match self.stamps.get(&(addr, len)) {
                    Some(&s) => ctx.dirty_since(addr, len, s) == Some(false),
                    None => false,
                }
            };
            // `VITASLOP_STAMP_VERIFY=1`: the stamps' FALSIFIER, naming what they missed. When the
            // stamps say "untouched", compare anyway and report - once per range - every range
            // whose bytes changed regardless: a guest store that carried no mark.
            if untouched
                && stamp_verify()
                && let Some(p) = retained.as_ref()
                && ctx.borrow_bytes(addr, len).is_some_and(|live| live != &p[..])
            {
                report_unstamped_change(ctx, addr, len, p);
            }
            if untouched {
                // >>> RE-STAMP, EVEN THOUGH NOTHING WAS READ. The stamps have just PROVED
                // these bytes current as of this scene, so recording that costs a hash write
                // and makes the entry's stamp track the epoch instead of standing still at
                // whatever it was established at.
                //
                // Without it a texture that is never written keeps its original stamp for the
                // life of the run, which is what made `rebase_epoch_if_exhausted` find a floor
                // of 1 and decline to renumber every time - MEASURED: 3 wraps and 0
                // renumberings per 30-present window, with the whole mechanism inert.
                let current = retained.clone().expect("set whenever an entry exists");
                self.restamp(addr, len);
                return current;
            }
            dirty_page_census(ctx, addr, len, &self.stamps);
            // >>> RE-READ ONLY THE PAGES THE GUEST WROTE, when the engine can name them.
            //
            // The map has just said this range was written; it can also say WHERE, and the
            // pages it does not name provably still hold the bytes in `retained`. CENSUSED
            // in a browser over 1,600 re-read ranges: 7.1% of their pages were dirty, under
            // 2% for the large textures. So the whole-texture copy below was re-establishing
            // bytes the map already knew, and in a browser that copy crosses the guest
            // boundary.
            //
            // The answer is the same answer: pages outside the runs cannot have changed, so
            // comparing the runs decides the whole texture, and a texture that HAS changed is
            // rebuilt as `retained` with those runs overwritten - which is what guest memory
            // now holds, byte for byte.
            if let Some(current) = self.reread_dirty_runs(ctx, addr, len, retained.as_ref()) {
                match current {
                    Reread::Unchanged(p) => {
                        self.restamp(addr, len);
                        return p;
                    }
                    Reread::Changed(bytes) => {
                        self.mutable.insert((addr, len));
                        crate::perf::note_bytes(crate::perf::Phase::DrawTextures, bytes.len());
                        self.insert(addr, len, bytes.clone());
                        self.restamp(addr, len);
                        return bytes;
                    }
                }
            }
            // Timed and COUNTED separately (see `Phase::DrawTextureCompare`): this is a
            // memcmp of the whole texture, and it is charged to the snapshot phase while
            // moving no bytes through `note_bytes` - so the phase reported "0.0 MB/frame"
            // and read as pure per-draw overhead. The bytes compared are the size of the
            // problem; the bytes re-read are only what actually changed.
            // `Some(false)` unchanged, `Some(true)` changed, `None` a backing that cannot
            // lend its bytes - which is not evidence either way, so it falls through to the
            // re-read without being recorded as mutable.
            let changed = {
                let _t = crate::perf::scope(crate::perf::Phase::DrawTextureCompare);
                ctx.borrow_bytes(addr, len).map(|live| {
                    crate::perf::note_bytes(crate::perf::Phase::DrawTextureCompare, len);
                    live != &p[..]
                })
            };
            match changed {
                Some(false) => {
                    // Through `retained`, not `p`: stamping needs `&mut self` and `p`
                    // borrows the entry table. They are the same `Arc`.
                    let current = retained.clone().expect("set whenever an entry exists");
                    // The compare has just re-established that these bytes are current,
                    // so stamp them: without this the shortcut above could never take
                    // hold for a texture whose stamp was dropped (a wrapped epoch, a
                    // cleared cache), and it would pay the full compare for ever.
                    self.restamp(addr, len);
                    return current;
                }
                // Seen to change ONCE is enough to keep it on the per-scene cadence for
                // the rest of the run. This is the set the frame-cadence shortcut above
                // deliberately does not apply to.
                Some(true) => {
                    self.mutable.insert((addr, len));
                    // Already proved different - the re-read below must not compare again.
                    retained = None;
                }
                None => {}
            }
        }
        // >>> COUNTED, BECAUSE ON THE BROWSER THIS *IS* THE COMPARE AND IT LOOKED FREE.
        //
        // An engine that can LEND its bytes compares them in place above and charges the
        // volume to `DrawTextureCompare`. An engine that cannot - the browser, permanently -
        // reaches here every scene a texture is not proven clean, copies the whole texture
        // across the guest boundary purely to find out whether it changed, and then usually
        // hands the retained buffer back. That copy was charged to NOTHING unless the bytes
        // had actually changed, so the one engine that pays it reported 0.0 MB for it.
        // [[vitaslop-count-bytes-when-there-is-no-clock]]: a volume only one engine pays is
        // exactly the volume that needs a counter.
        // Straight into an `Arc` - see `read_bytes_arc` for the pass this saves on a path
        // that, on the browser, copies whole textures across the guest boundary every scene.
        let raw = ctx.read_bytes_arc(addr, len);
        if retained.is_some() {
            crate::perf::note_bytes(crate::perf::Phase::DrawTextureCompare, raw.len());
        }
        // A backing that cannot LEND its bytes (`borrow_bytes` returned `None`) reaches here
        // every scene, whatever the texture is doing, so the copy above is the only way to
        // find out whether anything changed. Compare it: consumers key their decode and
        // upload caches on this buffer's IDENTITY, so handing back a fresh `Arc` holding
        // byte-identical pixels would re-decode and re-upload an unchanged texture every
        // scene. Same answer, same bytes, one buffer.
        if let Some(p) = retained
            && bytes_eq(&raw, &p)
        {
            self.restamp(addr, len);
            return p;
        }
        let bytes: Arc<[u8]> = raw;
        crate::perf::note_bytes(crate::perf::Phase::DrawTextures, bytes.len());
        self.insert(addr, len, bytes.clone());
        self.restamp(addr, len);
        bytes
    }

    /// Record that this entry's bytes are current as of THIS SCENE, so a later
    /// [`GuestMemory::dirty_since`] can prove nothing has touched them.
    ///
    /// # >>> THE EPOCH IS ONE BYTE, AND ADVANCING IT PER SNAPSHOT SPENT IT IN HALF A FRAME
    /// This used to call [`GuestCtx::bump_dirty_epoch`] itself, once per re-stamped texture.
    /// The epoch is a single byte, compared with `>=`, so it may not wrap silently - on wrap the
    /// whole dirty map is zeroed and every stamp taken before it is dropped as a lie.
    ///
    /// MEASURED on a live race with `bench --at 7400`: hundreds of re-stamps a frame against
    /// **254** usable epoch values. The map was therefore wiped repeatedly and almost no stamp
    /// survived long enough to prove anything, so the shortcut this mechanism exists for barely
    /// fired and nearly every texture fell through to the full memcmp. Half the guest CPU was
    /// spent proving, the hard way, something the stamps already knew.
    ///
    /// **The A/B, same build, same window, dirty map ON in both arms** - which is the BROWSER's
    /// configuration, and the browser is the only engine that runs with it (native leaves the
    /// map off because wasmtime bills the marks and they would speed the guest clock up):
    ///
    /// | arm | `texture snapshot compare` | bytes/frame | frame |
    /// |---|---|---|---|
    /// | epoch per snapshot | 35.9% of the window | 76.2 MB | ~22.2 ms (45 fps) |
    /// | one epoch per scene | **2.8%** | **2.9 MB** | **10.9 ms (92 fps)** |
    ///
    /// A 26x cut in bytes moved and 2.05x on guest CPU, from advancing a counter less often.
    ///
    /// The epoch does not need to advance per texture. It needs to advance whenever the GUEST
    /// may have run since the last batch of stamps, and within a scene every snapshot is taken
    /// from host calls that cannot be preempted. So it advances ONCE per scene ([`Self::
    /// scene_stamp`]), which is eleven times a frame on this title instead of five hundred, and
    /// the byte now lasts about twenty-three frames rather than half of one.
    ///
    /// A store made during this scene writes the current epoch, which EQUALS the stamp recorded
    /// here, and `>=` reports it dirty - so sharing one stamp across a scene stays exact in the
    /// direction that matters. It is conservative only for a page written earlier in the same
    /// scene, which costs one compare and then re-stamps clean.
    /// Re-establish a retained texture by reading only the pages the guest stored into.
    /// `None` when the engine cannot name them (no dirty map, or a range it will not answer
    /// for), which leaves the caller on the whole-texture path exactly as before.
    ///
    /// See the call site for the census that motivates it. The bytes charged to
    /// `DrawTextureCompare` are the ones actually moved, which is the point.
    pub(super) fn reread_dirty_runs(
        &mut self,
        ctx: &GuestCtx,
        addr: u32,
        len: usize,
        retained: Option<&Arc<[u8]>>,
    ) -> Option<Reread> {
        let retained = retained?;
        if retained.len() != len {
            return None;
        }
        // >>> THE A/B ARM. `VITASLOP_TEX_PAGE_READ=0` puts every re-read back on the whole-
        // texture path, which is the one thing that can separate "the picture is wrong because
        // a page was believed clean" from any other cause - on a device, where a bisect costs a
        // run each. The exactness argument is a proof and there is a test for it, so this is
        // not a default anyone is expected to change; it is the falsifier.
        static ON: std::sync::OnceLock<bool> = std::sync::OnceLock::new();
        if !*ON.get_or_init(|| {
            !matches!(
                crate::knobs::var("VITASLOP_TEX_PAGE_READ").as_deref(),
                Ok("0")
            )
        }) {
            return None;
        }
        let mut runs = std::mem::take(&mut self.run_scratch);
        runs.clear();
        let answered = {
            let _t = crate::perf::scope(crate::perf::Phase::DrawTextureCompare);
            ctx.dirty_runs_since(addr, len, *self.stamps.get(&(addr, len))?, &mut runs)
        };
        if answered.is_none() {
            self.run_scratch = runs;
            return None;
        }
        // >>> A RANGE THE GUEST WROTE ALMOST ALL OF GOES BACK ON THE WHOLE-TEXTURE PATH.
        //
        // When the content HAS changed, this path owes a copy of the retained buffer to build
        // the new one on, so a texture that is dirty end to end would read everything AND
        // clone it - more traffic than the plain re-read, not less. The census says that is
        // rare (large textures came back under 2% dirty), and this makes it cost nothing when
        // it happens instead of costing double.
        let dirty_bytes: usize = runs.iter().map(|&(a, b)| b - a).sum();
        if dirty_bytes * 4 > len * 3 {
            self.run_scratch = runs;
            return None;
        }
        // Every page clean: the shortcut above should have caught this, but a map that says
        // so here says the same thing, and the answer is the retained buffer.
        let mut changed = false;
        let mut fresh: Vec<u8> = Vec::new();
        for &(start, end) in runs.iter() {
            let _t = crate::perf::scope(crate::perf::Phase::DrawTextureCompare);
            let piece = ctx.read_bytes(addr + start as u32, end - start);
            crate::perf::note_bytes(crate::perf::Phase::DrawTextureCompare, piece.len());
            if piece.len() != end - start {
                // A short read means the range is not fully backed; fall back to the whole
                // path rather than inventing the tail.
                self.run_scratch = runs;
                return None;
            }
            if !changed && piece[..] != retained[start..end] {
                changed = true;
                fresh = retained.to_vec();
            }
            if changed {
                fresh[start..end].copy_from_slice(&piece);
            }
        }
        self.run_scratch = runs;
        Some(if changed {
            Reread::Changed(fresh.into())
        } else {
            Reread::Unchanged(retained.clone())
        })
    }

    pub(super) fn restamp(&mut self, addr: u32, len: usize) {
        if self.scene_stamp != 0 {
            self.stamps.insert((addr, len), self.scene_stamp);
        }
    }

    /// >>> THE HOST JUST WROTE THIS TEXTURE, SO NOBODY HAS TO FIND OUT WHETHER IT CHANGED.
    ///
    /// A decoded video picture is the one texture in a frame whose bytes are authored by the
    /// ENGINE rather than by the guest: [`crate::vita::avcdec`] converts a decoder's output
    /// into the layout the title asked for and writes it into the title's own buffer. Without
    /// this the draw that binds it then goes the whole way round the ordinary ladder to
    /// rediscover what the engine already knows - the dirty map says "written", the compare
    /// runs over the whole picture, the browser (which cannot lend its bytes) copies the
    /// picture across the guest boundary purely to compare it, and a third buffer is
    /// allocated to hold the answer.
    ///
    /// MEASURED on a phone during the movie: `draw: snapshot textures (miss path) 0.82 MB` +
    /// `draw: texture snapshot compare 2.24 MB` per frame, or ~3 MB/frame of memory traffic
    /// spent establishing that a video frame changed.
    ///
    /// So the writer hands the bytes it wrote straight to the cache. `bytes` MUST be exactly
    /// what guest memory now holds at `[addr, addr + bytes.len())` - it is handed to draws as
    /// the texture's contents, so anything else is a lie about guest memory, not a cache
    /// policy. The one epoch spent here is per PICTURE (~30 a second) against the one per
    /// scene the ordinary path spends (~330 a second), and taking it AFTER the write is what
    /// makes the stamp exact: our own stores carry the older epoch, a later guest store
    /// carries this one, and `dirty_since` tells them apart.
    pub(super) fn author(&mut self, ctx: &GuestCtx, addr: u32, bytes: Arc<[u8]>) {
        let len = bytes.len();
        if len == 0 {
            return;
        }
        self.insert(addr, len, bytes);
        // It demonstrably changes, so it must stay on the per-scene cadence: the opt-in
        // per-FRAME shortcut in `get_or_read` deliberately does not apply to a mutable
        // texture, and a video picture is the definition of one.
        self.mutable.insert((addr, len));
        // The set and per-binding memos hold the PREVIOUS picture and are re-proven by
        // pointer identity against `entries`, so they reject themselves on their next use.
        // Nothing to clear here.
        // `None` is an engine that tracks no stores: it has no fast path to arm, the entry is
        // still the right bytes, and the ordinary compare will confirm it and cost one pass.
        if let Some((stamp, wrapped)) = ctx.bump_dirty_epoch() {
            self.stamp_stacks(ctx);
            if wrapped {
                crate::perf::note_epoch_wrap();
                self.stamps.clear();
                self.vertex_stamps.clear();
                self.index_stamps.clear();
            }
            self.stamps.insert((addr, len), stamp);
        }
    }

    /// The current bytes of the VERTEX buffer of `len` bytes at guest address `addr`, reusing
    /// the retained snapshot when the guest provably has not stored there since.
    ///
    /// # This is the texture shortcut, aimed at the next biggest thing the capture moves
    /// Every draw used to `read_bytes` its whole vertex range - one allocation and one full copy
    /// out of guest memory - and then convert that `Vec` into an `Arc`, which is a SECOND
    /// allocation and copy. Twice the buffer, twice, per draw. In the browser each read is also a
    /// crossing into JS.
    ///
    /// The dirty map answers "has the guest written here since I read it?" for the cost of one
    /// page-range read, and static geometry answers no for the whole run. When it answers yes the
    /// bytes are re-read, which is exactly the old behaviour, so dynamic geometry is unaffected.
    ///
    /// Returning the SAME `Arc` matters beyond the copy: the renderer's packed-vertex cache keys
    /// on this buffer's identity, so an equal-but-different buffer misses it
    /// ([[vitaslop-content-hash-cache-must-verify]]).
    /// The shared buffer holding exactly `bytes`, for a vertex stream this engine BUILT rather
    /// than read - the interleave of a multi-stream draw.
    ///
    /// # Why the built buffer needs an identity too
    /// `get_or_read_vertices` gives a single-stream draw a stable `Arc` across frames, and every
    /// cache downstream keys on that identity. A MULTI-stream draw had none: its rows are
    /// gathered from several guest buffers into a fresh allocation, so byte-identical geometry
    /// arrived as a different buffer every frame.
    ///
    /// MEASURED on a golf title, whose vertex programs are multi-stream throughout: the
    /// renderer's allocation-keyed packed-vertex cache missed on essentially every draw and fell
    /// through to hashing each stream in full - **7.03 MB hashed per frame**, then a full memcmp
    /// to verify the hit, together 8.9% of all busy time in a V8 profile of the worker. The
    /// hashes were finding what this interning establishes once: 463 of 671 streams were
    /// byte-identical to one already held.
    ///
    /// The comparison is what decides, exactly as everywhere else here - the fingerprint only
    /// picks the candidate ([[vitaslop-content-hash-cache-must-verify]]). A miss costs one
    /// allocation, which is what the caller would have paid unconditionally before.
    /// Unused today - every caller wants [`Self::intern_vertices_noting_hit`], because the HIT
    /// is what admits geometry to the gather memo. Kept as the plain question's name.
    #[allow(dead_code)]
    pub(super) fn intern_vertices(&mut self, bytes: &[u8]) -> Arc<[u8]> {
        self.intern_vertices_noting_hit(bytes).0
    }

    /// [`Self::intern_vertices`], also saying whether the bytes were ALREADY held: a hit is
    /// the second sighting of this geometry, which is what admits it to the gather memo.
    pub(super) fn intern_vertices_noting_hit(&mut self, bytes: &[u8]) -> (Arc<[u8]>, bool) {
        // `VITASLOP_VERTEX_INTERN=0` is the NEGATIVE CONTROL: a fresh buffer per draw, which is
        // what this path did before. An A/B arm rather than a mode - the two arms must differ in
        // one thing and be reachable from ONE build, or the comparison is between two binaries
        // ([[vitaslop-browser-ab-needs-a-negative-control]]).
        if !vertex_intern_enabled() {
            return (Arc::from(bytes), false);
        }
        let fp = vertex_fingerprint(bytes);
        if let Some(p) = self.vertex_by_content.get(&(bytes.len(), fp))
            && bytes_eq(bytes, p)
        {
            // A HIT pays a comparison; a MISS pays a copy. See `Phase::DrawVertexInternHit`.
            crate::perf::note_hit(crate::perf::Phase::DrawVertexInternHit);
            let p = p.clone();
            // A hit is a use - it is what keeps this entry out of the eviction below.
            self.vertex_content_used
                .insert((bytes.len(), fp), self.frame_seq);
            return (p, true);
        }
        // A MISS on a key the index already holds is a fingerprint collision, not new
        // geometry, and it costs both meshes their identity every frame. See
        // `Phase::DrawVertexInternCollide`.
        if self.vertex_by_content.contains_key(&(bytes.len(), fp)) {
            crate::perf::note_hit(crate::perf::Phase::DrawVertexInternCollide);
        }
        let made: Arc<[u8]> = Arc::from(bytes);
        // Bounded by BYTES as well as entries: an interned buffer is held by this map alone,
        // where an entry that is also in `vertex_entries` is covered by that map's budget.
        if self.vertex_content_bytes + made.len() > vertex_snapshot_budget()
            || self.vertex_by_content.len() >= VERTEX_CONTENT_INDEX_CAP
        {
            self.evict_interned_vertices(made.len());
        }
        self.vertex_content_used
            .insert((made.len(), fp), self.frame_seq);
        if let Some(old) = self
            .vertex_by_content
            .insert((made.len(), fp), made.clone())
        {
            self.vertex_content_bytes = self.vertex_content_bytes.saturating_sub(old.len());
        }
        self.vertex_content_bytes += made.len();
        (made, false)
    }

    /// The spans a gather reads, for the memo key and its stamps: each per-vertex stream's
    /// `[first row .. last row + column]`, and each per-instance or stride-0 stream's single
    /// row. `None` for a draw that reads nothing.
    ///
    /// >>> FILLS THE CALLER'S VEC RATHER THAN RETURNING ONE. The only callers build a
    /// [`GatherKey`] to PROBE the gather memo with, and the probe is the common outcome: a
    /// returned `Vec` is an allocation and a free for every multi-stream draw (422 a frame on a
    /// baseball title's gameplay) to ask a question, not to keep an answer. The key is filed
    /// only on the rarer path that actually memoises, and clones itself there.
    pub(super) fn gather_spans_into(
        out: &mut Vec<(u32, usize, u32, u32)>,
        streams: &[VertexStreamInfo],
        bound_streams: &[u32],
        base: &[u32],
        rows: (u32, u32),
    ) {
        let (first, last) = rows;
        out.clear();
        out.extend(
            streams
                .iter()
                .enumerate()
                .filter(|(_, st)| st.column != 0)
                .map(|(si, st)| {
                    let buf = bound_streams.get(si).copied().unwrap_or(0);
                    let dst = base.get(si).copied().unwrap_or(0);
                    if st.per_instance || st.stride == 0 {
                        (buf, st.column as usize, dst, st.column)
                    } else {
                        (
                            buf.wrapping_add(first.wrapping_mul(st.stride)),
                            (last - first) as usize * st.stride as usize + st.column as usize,
                            dst,
                            st.column,
                        )
                    }
                }),
        );
    }

    /// The bytes of the uniform-buffer window at `addr`, from the snapshot when the dirty map
    /// proves the guest has not written it since, else re-read - handing back the SAME buffer
    /// when the bytes are unchanged, so everything keyed on the window's identity downstream
    /// (the renderer's per-pass push memo) keeps hitting.
    pub(super) fn get_or_read_window(&mut self, ctx: &GuestCtx, addr: u32, len: usize) -> Arc<[u8]> {
        if len == 0 {
            return Arc::from(&[][..]);
        }
        self.arm_scene_stamp(ctx);
        let key = (addr, len);
        if let Some((bytes, stamp)) = self.window_entries.get(&key)
            && *stamp != 0
            && ctx.dirty_since(addr, len, *stamp) == Some(false)
        {
            crate::perf::note_hit(crate::perf::Phase::WindowHitStamp);
            return bytes.clone();
        }
        // Compared in place first - see the same step in `get_or_read_vertices`.
        if inplace_compare()
            && let Some((held, st)) = self.window_entries.get_mut(&key)
            && let Some(live) = ctx.borrow_bytes(addr, len)
            && live.len() == held.len()
            && bytes_eq(live, held)
        {
            *st = self.scene_stamp;
            crate::perf::note_hit(crate::perf::Phase::WindowHitCompare);
            return held.clone();
        }
        crate::perf::note_hit(crate::perf::Phase::WindowMiss);
        crate::perf::note_bytes(crate::perf::Phase::WindowMiss, len);
        let raw = ctx.read_bytes_arc(addr, len);
        let stamp = self.scene_stamp;
        if let Some((held, st)) = self.window_entries.get_mut(&key) {
            if bytes_eq(&raw, held) {
                *st = stamp;
                return held.clone();
            }
            *held = raw.clone();
            *st = stamp;
            return raw;
        }
        // Bounded by the vertex budget, shed whole: windows are kilobytes and a clear costs
        // one re-read per window, never an answer.
        if self.window_bytes + len > vertex_snapshot_budget() {
            self.window_entries.clear();
            self.window_bytes = 0;
        }
        self.window_bytes += len;
        self.window_entries.insert(key, (raw.clone(), stamp));
        raw
    }

    /// Whether `bytes` IS the shared placeholder [`deferred_window_placeholder`] handed out for
    /// its length - by pointer, so a placeholder is recognised as blank WITHOUT scanning it.
    pub(super) fn is_window_placeholder(&self, bytes: &Arc<[u8]>) -> bool {
        self.window_placeholders
            .get(&bytes.len())
            .is_some_and(|p| Arc::ptr_eq(p, bytes))
    }

    /// A stand-in for a guest-memory WINDOW that is going to be re-snapshotted at the end of
    /// this scene anyway - see [`VitaState::capture_mem_windows`]. Carries the right LENGTH and
    /// nothing else; one allocation per distinct length for the life of the run, an `Arc` clone
    /// thereafter.
    pub(super) fn deferred_window_placeholder(&mut self, len: usize) -> Arc<[u8]> {
        // A handful of distinct window sizes per title; the cap is there so a title that sizes
        // a window per draw cannot grow this without bound, and losing an entry costs one
        // allocation.
        const CAP: usize = 256;
        if let Some(a) = self.window_placeholders.get(&len) {
            return a.clone();
        }
        if self.window_placeholders.len() >= CAP {
            self.window_placeholders.clear();
        }
        let a: Arc<[u8]> = vec![0u8; len].into();
        self.window_placeholders.insert(len, a.clone());
        a
    }

    /// >>> A WINDOW OVER A RAW TARGET THIS FRAME RENDERED: ONLY THE BYTES THE GPU FILL LEAVES.
    ///
    /// The renderer copies a raw 64-bit target's texels over every byte of a window that lies
    /// on it (`fill_windows_from_raw_targets`), so the guest's bytes there are read, compared,
    /// carried and uploaded only to be overwritten. `cover` is those bytes, relative to `addr`
    /// (`vitaslop_platform::gpu::raw_window_fill_cover`, the fill's own plan over the span the
    /// renderer places); they are ZERO here and everything else is the guest's. MEASURED on a
    /// shooter's stairwell: four ~4 MiB windows over its G-buffer, read, scanned and compared at
    /// every draw and again at the resolve, were ~9 s of a 68 s phase.
    ///
    /// Held per `(addr, len)` and handed back as the SAME buffer while the cover and the
    /// uncovered bytes are unchanged, so a repeat costs a compare of the uncovered bytes - none,
    /// for a window wholly on the target.
    pub(super) fn get_or_read_covered_window(&mut self, ctx: &GuestCtx, addr: u32, len: usize, cover: &[(u32, u32)]) -> Arc<[u8]> {
        if len == 0 {
            return Arc::from(&[][..]);
        }
        if let Some(e) = self.covered_windows.get(&(addr, len))
            && e.cover == cover
            && uncovered(len, cover).all(|(s, t)| match ctx.borrow_bytes(addr.wrapping_add(s as u32), t - s) {
                Some(live) => live == &e.bytes[s..t],
                None => ctx.read_bytes(addr.wrapping_add(s as u32), t - s) == e.bytes[s..t],
            })
        {
            return e.bytes.clone();
        }
        let bytes = Self::read_covered_window(ctx, addr, len, cover);
        let mut first: Option<[u8; 4]> = None;
        let blank = uncovered(len, cover).all(|(s, t)| {
            bytes[s..t].chunks(4).all(|w| {
                let mut word = [0u8; 4];
                word[..w.len()].copy_from_slice(w);
                *first.get_or_insert(word) == word
            })
        }) && first.is_some();
        // Bounded and shed whole: a handful of these a frame, and losing one costs a read.
        if self.covered_windows.len() >= 64 {
            self.covered_windows.clear();
            self.covered_by_ptr.clear();
        }
        if let Some(old) = self.covered_windows.insert(
            (addr, len),
            CoveredWindow { cover: cover.to_vec(), bytes: bytes.clone(), blank },
        ) {
            self.covered_by_ptr.remove(&(old.bytes.as_ptr() as usize));
        }
        self.covered_by_ptr.insert(bytes.as_ptr() as usize, (addr, len));
        bytes
    }

    /// [`Self::get_or_read_covered_window`]'s read, without the memo: `cover` zero, the rest
    /// read from guest memory.
    pub(super) fn read_covered_window(ctx: &GuestCtx, addr: u32, len: usize, cover: &[(u32, u32)]) -> Arc<[u8]> {
        let mut arc = unsafe {
            // SAFETY: a zeroed byte is an initialised byte.
            Arc::<[std::mem::MaybeUninit<u8>]>::assume_init(Arc::new_zeroed_slice(len))
        };
        let buf = Arc::get_mut(&mut arc).expect("freshly created, uniquely owned");
        for (s, t) in uncovered(len, cover) {
            ctx.read_into(addr.wrapping_add(s as u32), &mut buf[s..t]);
        }
        arc
    }

    /// Whether `bytes` came from [`Self::get_or_read_covered_window`] - by address, no scan.
    pub(super) fn is_covered_window(&self, bytes: &Arc<[u8]>) -> bool {
        self.covered_window_entry(bytes).is_some()
    }

    fn covered_window_entry(&self, bytes: &Arc<[u8]>) -> Option<(u32, usize)> {
        let key = *self.covered_by_ptr.get(&(bytes.as_ptr() as usize))?;
        self.covered_windows
            .get(&key)
            .is_some_and(|e| Arc::ptr_eq(&e.bytes, bytes))
            .then_some(key)
    }

    /// The resolve's re-read of a covered window: its uncovered bytes again when they were
    /// blank at the draw (the rule [`window_wants_reread`] applies to every window), else the
    /// same buffer. `None` for a window that is not a covered one.
    pub(super) fn reread_covered_window(&mut self, ctx: &GuestCtx, bytes: &Arc<[u8]>) -> Option<Arc<[u8]>> {
        let (addr, len) = self.covered_window_entry(bytes)?;
        let e = self.covered_windows.get(&(addr, len))?;
        if !e.blank {
            return Some(bytes.clone());
        }
        let cover = e.cover.clone();
        Some(self.get_or_read_covered_window(ctx, addr, len, &cover))
    }

    /// The memoised gather for `key`, if every span it read is provably unwritten since.
    pub(super) fn gather_memo_hit(&mut self, ctx: &GuestCtx, key: &GatherKey) -> Option<Arc<[u8]>> {
        let e = self.gather_memo.get_mut(key)?;
        let clean = e
            .stamps
            .iter()
            .all(|&(addr, len, stamp)| ctx.dirty_since(addr, len, stamp) == Some(false));
        if !clean {
            let old = self.gather_memo.remove(key).map_or(0, |e| e.result.len());
            self.gather_memo_bytes = self.gather_memo_bytes.saturating_sub(old);
            return None;
        }
        e.used = self.frame_seq;
        crate::perf::note_hit(crate::perf::Phase::DrawVertexGatherMemoHit);
        Some(e.result.clone())
    }

    /// File a gather under `key`, stamped with the current scene epoch, evicting the coldest
    /// entries past the vertex budget. Nothing is filed without a scene stamp: without the
    /// dirty map there is no proof to offer later.
    pub(super) fn gather_memo_file(&mut self, key: GatherKey, gather: Option<Arc<[u32]>>, result: Arc<[u8]>) {
        if self.scene_stamp == 0 {
            return;
        }
        let budget = vertex_snapshot_budget();
        if self.gather_memo_bytes + result.len() > budget {
            let target = budget.saturating_sub(result.len() + budget / 8);
            let mut by_age: Vec<(u64, GatherKey, usize)> = self
                .gather_memo
                .iter()
                .filter(|(_, e)| e.used != self.frame_seq)
                .map(|(k, e)| (e.used, k.clone(), e.result.len()))
                .collect();
            by_age.sort_by_key(|(u, ..)| *u);
            for (_, k, n) in by_age {
                if self.gather_memo_bytes <= target {
                    break;
                }
                self.gather_memo.remove(&k);
                self.gather_memo_bytes = self.gather_memo_bytes.saturating_sub(n);
            }
        }
        let stamp = self.scene_stamp;
        let stamps = key
            .spans
            .iter()
            .map(|&(a, l, _, _)| (a, l, stamp))
            .collect();
        if let Some(old) = self.gather_memo.insert(
            key,
            GatherMemo {
                result: result.clone(),
                stamps,
                used: self.frame_seq,
                gather,
            },
        ) {
            self.gather_memo_bytes = self.gather_memo_bytes.saturating_sub(old.result.len());
        }
        self.gather_memo_bytes += result.len();
    }

    /// The most [`Self::gather_src`] keeps between draws.
    ///
    /// A single draw's stream is kilobytes; this is generous by orders of magnitude so a steady
    /// frame never reallocates, and small enough that a one-off giant read cannot leave
    /// megabytes pinned in a heap that has no way to return them.
    pub(super) const GATHER_SCRATCH_CAP: usize = 1 << 20;

    /// How many bytes the union of a draw's bound streams may exceed their sum by before the
    /// merged read stops being worth the crossing it saves.
    ///
    /// A boundary crossing is on the order of a microsecond; a `Uint8Array.set` moves several
    /// gigabytes a second, so a microsecond buys a few kilobytes. This is that break-even,
    /// rounded down - the merge exists to avoid a CALL, not to read a megabyte to save one.
    pub(super) const GATHER_SPAN_SLACK: usize = 4096;

    /// The single guest range covering every bound stream of a draw, when reading it whole
    /// beats reading the streams one at a time. `None` = read them separately.
    ///
    /// See the call site in [`Self::gather_into_scratch`] for why this is exactly equivalent
    /// and what the two guards are for.
    pub(super) fn gather_span(
        &self,
        ctx: &GuestCtx,
        streams: &[VertexStreamInfo],
        bound_streams: &[u32],
        first_vertex: u32,
        vertex_count: u32,
    ) -> Option<(u32, usize)> {
        let (mut lo, mut hi, mut sum) = (u32::MAX, 0u32, 0usize);
        let mut live = 0usize;
        for (si, st) in streams.iter().enumerate() {
            // The same rule the scatter uses: an empty COLUMN is nothing to read, and a
            // stride-0 stream is one row repeated - see `VertexStreamInfo::column`. These two
            // must agree, or a stream the scatter reads is missing from the span it slices
            // from and its offset underflows.
            if st.column == 0 {
                continue;
            }
            let buf = bound_streams.get(si).copied().unwrap_or(0);
            let (addr, want) = if st.per_instance || st.stride == 0 {
                (buf, st.column as usize)
            } else {
                (
                    buf.checked_add(first_vertex.checked_mul(st.stride)?)?,
                    (vertex_count as usize - 1)
                        .checked_mul(st.stride as usize)?
                        .checked_add(st.column as usize)?,
                )
            };
            let end = addr.checked_add(u32::try_from(want).ok()?)?;
            lo = lo.min(addr);
            hi = hi.max(end);
            sum = sum.checked_add(want)?;
            live += 1;
        }
        // One stream is already one read, and none is no read at all.
        if live < 2 {
            return None;
        }
        let span = (hi - lo) as usize;
        // The union has to be worth it, and its base has to map - see the call site.
        (span <= sum + Self::GATHER_SPAN_SLACK && ctx.maps(lo)).then_some((lo, span))
    }

    /// Read every bound stream and scatter its rows into the interleaved scratch. Returns
    /// the scratch by value; the caller puts it back (see [`Self::interleave_scratch`]).
    #[allow(clippy::too_many_arguments)]
    /// A draw's vertex buffer, as its (already rewritten) indices address it - see
    /// [`DrawIndices`]. One implementation for the draw-time and the end-of-scene read.
    #[allow(clippy::too_many_arguments)]
    pub(super) fn read_draw_vertices(
        &mut self,
        ctx: &GuestCtx,
        idx: &DrawIndices,
        single_stream: bool,
        streams: &[VertexStreamInfo],
        bound_streams: &[u32],
        base: &[u32],
        layout: &RowLayout,
        stride: u32,
    ) -> Arc<[u8]> {
        let _owner = vitaslop_platform::heap::owner_scope(vitaslop_platform::heap::Owner::GeometrySnapshot);
        // >>> THE GATHER MEMO, in front of both multi-stream paths.
        //
        // A gather is guest reads (one wasm-boundary crossing per stream), a scatter into the
        // packed row, and a content intern that hashes and compares the whole result. MEASURED
        // in a browser on a baseball title's stadium: 2.42 ms of a 13 ms frame, over 420 draws
        // - and 409 of those 420 came back from the intern as bytes it ALREADY HELD. So 97%
        // of that work re-derived a buffer whose every input page was unwritten, which the
        // dirty map can say for the cost of a few page lookups.
        //
        // A memo over this gather was tried before and REFUTED on a title that rebuilds its
        // geometry into a rotating arena: it never hit and its misses cost 4 ms a window. The
        // gate that answers that: nothing is FILED until the content intern has hit - the
        // geometry's second sighting at the same spans. An arena title never hits the intern,
        // never files, and pays one failed map probe per draw; a stable-address title files
        // on its second frame and skips the gather from its third.
        // The memo's proofs are dirty-map stamps, which need this scene's epoch armed - the
        // index read normally did that already; a draw reaching here first arms it now.
        self.arm_scene_stamp(ctx);
        if let Some(gather) = idx.gather.as_ref() {
            let _gather = crate::perf::scope(crate::perf::Phase::DrawVertexGather);
            let rows = (gather[0], gather[gather.len() - 1]);
            // The probe key is a SCRATCH key held across draws - see `gather_spans_into`. Taken
            // out of `self` and put back because the memo probe below borrows `self` mutably;
            // the guest cannot run inside a host call, so nothing can observe the gap.
            let mut key = std::mem::take(&mut self.gather_key_scratch);
            Self::gather_spans_into(&mut key.spans, streams, bound_streams, base, rows);
            key.stride = stride;
            key.layout = layout.key;
            key.gather = (gather.as_ptr() as usize, gather.len());
            let hit = self.gather_memo_hit(ctx, &key);
            if let Some(hit) = hit {
                self.gather_key_scratch = key;
                crate::perf::note_bytes(crate::perf::Phase::DrawVertices, hit.len());
                return hit;
            }
            let vertices =
                self.gather_rows_into_scratch(ctx, streams, bound_streams, layout, stride, gather);
            crate::perf::note_bytes(crate::perf::Phase::DrawVertices, vertices.len());
            let (interned, seen_before) =
                crate::perf::time(crate::perf::Phase::DrawVertexGatherIntern, || {
                    self.intern_vertices_noting_hit(&vertices)
                });
            self.interleave_scratch = vertices;
            if seen_before {
                self.gather_memo_file(key.clone(), Some(gather.clone()), interned.clone());
            }
            self.gather_key_scratch = key;
            return interned;
        }
        if single_stream {
            // The overwhelmingly common case is one per-vertex stream, whose rows are already
            // contiguous: take them in one read rather than one per vertex (this path runs for
            // every draw of every frame, over meshes of thousands of vertices).
            //
            // ...and better than one read: a SNAPSHOT, so a mesh the guest has not touched
            // since the last draw is not read at all. See `get_or_read_vertices`.
            return self.get_or_read_vertices(
                ctx,
                bound_streams[0].wrapping_add(idx.first_vertex * stride),
                (idx.vertex_count * stride) as usize,
            );
        }
        let _gather = crate::perf::scope(crate::perf::Phase::DrawVertexGather);
        if idx.vertex_count == 0 {
            return Arc::from(&[][..]);
        }
        let rows = (idx.first_vertex, idx.first_vertex + idx.vertex_count - 1);
        let mut key = std::mem::take(&mut self.gather_key_scratch);
        Self::gather_spans_into(&mut key.spans, streams, bound_streams, base, rows);
        key.stride = stride;
        key.layout = layout.key;
        key.gather = (0, 0);
        let hit = self.gather_memo_hit(ctx, &key);
        if let Some(hit) = hit {
            self.gather_key_scratch = key;
            crate::perf::note_bytes(crate::perf::Phase::DrawVertices, hit.len());
            return hit;
        }
        let vertices = self.gather_into_scratch(
            ctx,
            streams,
            bound_streams,
            layout,
            stride,
            idx.first_vertex,
            idx.vertex_count,
        );
        crate::perf::note_bytes(crate::perf::Phase::DrawVertices, vertices.len());
        // Only the RESULT's identity is shared with the byte-identical buffer already held,
        // so the caches downstream that key on identity can hit. See `intern_vertices`.
        //
        // TIMED SEPARATELY from the gather that produced the bytes: this is a content
        // fingerprint plus a full comparison, and the gather is guest reads plus a scatter.
        // See `Phase::DrawVertexGatherRead`.
        let (interned, seen_before) =
            crate::perf::time(crate::perf::Phase::DrawVertexGatherIntern, || {
                self.intern_vertices_noting_hit(&vertices)
            });
        self.interleave_scratch = vertices;
        if seen_before {
            self.gather_memo_file(key.clone(), None, interned.clone());
        }
        self.gather_key_scratch = key;
        interned
    }

    /// >>> AN INSTANCED DRAW IS EXPANDED ON THE CPU, ONE ROW PER (INSTANCE, VERTEX).
    ///
    /// `sceGxmDrawInstanced(.., indexCount, indexWrap)`: the hardware reads the FIRST
    /// `indexWrap` indices of the buffer and replays them `indexCount / indexWrap` times,
    /// stepping the instance index once per replay. A stream declared with an INSTANCE index
    /// source (`VertexStreamInfo::per_instance`) is fetched at the instance index; every
    /// other stream at the index value. MEASURED on a baseball title's crowd (capsule caps35b):
    /// `indexCount` 846, `indexWrap` 6, and the index buffer holds `0 1 2 2 3 0` followed by
    /// 840 entries of whatever the arena held before - so a read of all 846 indices, which is
    /// what this engine did, drew 141 quads out of stale rows 4..587 of a FOUR-row quad
    /// buffer, and three of those "sections" tiled the whole frame with the crowd atlas.
    ///
    /// The result is a plain interleaved mesh in the layout `set_vertex_program` packed
    /// (`base`/`stride`), which every consumer draws unchanged: row `k * V + v` is per-vertex
    /// row `min_index + v` beside per-instance row `k`, and the indices are the wrapped
    /// six rebased onto the window and offset by `k * V` per instance. `None` means the
    /// draw is not instanced (wrap 0, or fewer indices than one wrap) and the plain path
    /// applies. A draw whose expansion would not fit its 16-bit index format is reported and
    /// left to the plain path - visible as ONE instance, not silently wrong.
    #[allow(clippy::too_many_arguments)]
    pub(crate) fn read_instanced_draw(
        &mut self,
        ctx: &GuestCtx,
        index_addr: u32,
        index_len: usize,
        index_elem: usize,
        index_wrap: u32,
        streams: &[VertexStreamInfo],
        bound_streams: &[u32],
        layout: &RowLayout,
        stride: u32,
    ) -> Option<(DrawIndices, Arc<[u8]>)> {
        let n = index_len / index_elem.max(1);
        let wrap = index_wrap as usize;
        if wrap == 0 || n <= wrap || stride == 0 {
            return None;
        }
        let instances = n / wrap;
        // >>> THE PARENT IS OPENED HERE, BESIDE ITS CHILD, AND NOT AT THE CALL SITE.
        //
        // This function charges `DrawVertexGather`, but both of its callers sit OUTSIDE the
        // `DrawVertices` timer the non-instanced vertex path is wrapped in - so an instanced
        // draw's gather was billed to the child and to nothing above it. MEASURED in the browser
        // on a baseball title, whose crowd is instanced: the gather read **1.06 ms/frame against its own
        // parent's 0.99**, which is impossible for a nested phase and is how much of that
        // subtree was unattributed.
        //
        // Opened HERE rather than around the call because every draw CALLS this and only an
        // instanced one does any work in it: wrapping the call added a scope to all 2.19 M
        // draws of a run, doubled the parent's entry count and put 12.5% on DRAW TOTAL in
        // timer overhead alone. Past the early returns, the entry count is the instanced-draw
        // count, which is what the row should say.
        let _vertices = crate::perf::scope(crate::perf::Phase::DrawVertices);
        let _gather = crate::perf::scope(crate::perf::Phase::DrawVertexGather);
        // >>> THE MEMO, IN FRONT OF EVERY READ - see `instanced_memo`. The stamps it proves
        // cover the index words as well as the streams, because `min`/`per` and the rebased
        // index list are functions of those words too.
        self.arm_scene_stamp(ctx);
        let memo_on = instanced_memo_enabled();
        let mut key = std::mem::take(&mut self.instanced_key_scratch);
        key.index = (index_addr, wrap * index_elem, index_elem);
        key.wrap = index_wrap;
        key.stride = stride;
        key.layout = layout.key;
        key.streams.clear();
        key.streams
            .extend(streams.iter().enumerate().map(|(si, st)| {
                (
                    bound_streams.get(si).copied().unwrap_or(0),
                    st.stride,
                    st.column,
                    st.per_instance,
                )
            }));
        if memo_on && self.instanced_memo.contains_key(&key) {
            let clean = self.instanced_memo[&key]
                .stamps
                .iter()
                .all(|&(addr, len, stamp)| ctx.dirty_since(addr, len, stamp) == Some(false));
            if clean {
                let e = self.instanced_memo.get_mut(&key).expect("just probed");
                e.used = self.frame_seq;
                let out = (
                    DrawIndices {
                        bytes: e.indices.clone(),
                        first_vertex: 0,
                        vertex_count: e.rows,
                        gather: None,
                    },
                    e.vertices.clone(),
                );
                self.instanced_key_scratch = key;
                crate::perf::note_hit(crate::perf::Phase::DrawVertexGatherMemoHit);
                return Some(out);
            }
            let old = self
                .instanced_memo
                .remove(&key)
                .map_or(0, |e| e.vertices.len() + e.indices.len());
            self.instanced_memo_bytes = self.instanced_memo_bytes.saturating_sub(old);
        }
        // Every span this expansion reads, stamped at the scene epoch armed above, accumulated
        // as the reads happen so the memo can never claim a range it did not actually read.
        let stamp = self.scene_stamp;
        let mut stamps: Vec<(u32, usize, u8)> = vec![(index_addr, wrap * index_elem, stamp)];
        let raw = crate::perf::time(crate::perf::Phase::DrawIndices, || {
            ctx.read_bytes(index_addr, wrap * index_elem)
        });
        crate::perf::note_bytes(crate::perf::Phase::DrawIndices, raw.len());
        let index_of = |c: &[u8]| match index_elem {
            2 => u16::from_le_bytes([c[0], c[1]]) as u32,
            _ => u32::from_le_bytes([c[0], c[1], c[2], c[3]]),
        };
        let wrapped: Vec<u32> = raw.chunks(index_elem).map(index_of).collect();
        let min = wrapped.iter().copied().min().unwrap_or(0);
        let max = wrapped.iter().copied().max().unwrap_or(0);
        let per = (max - min + 1) as usize;
        let rows = instances * per;
        if index_elem == 2 && rows > u16::MAX as usize + 1 {
            tracing::warn!(
                target: "vitaslop::gxm",
                instances, per_instance_vertices = per, index_wrap,
                "instanced draw: {rows} expanded rows do not fit a 16-bit index buffer - only its FIRST instance is drawn"
            );
            self.instanced_key_scratch = key;
            return None;
        }
        let stride_us = stride as usize;
        let mut vertices = vec![0u8; rows * stride_us];
        for (si, st) in streams.iter().enumerate() {
            if st.column == 0 {
                continue;
            }
            let buf = bound_streams.get(si).copied().unwrap_or(0);
            let src_step = st.stride as usize;
            let mine = Self::pieces_of(layout.pieces(), si);
            // >>> WHAT A STREAM CONTRIBUTES TO THE ROW IS ITS PIECES, NOT ITS COLUMN.
            //
            // This used to skip the stream ENTIRELY when `base[si] + column` overflowed the
            // packed row. That is the COLUMN form's invariant - there every stream occupied a
            // column as wide as its stride, laid end to end, so the sum was the row. Under the
            // PER-ATTRIBUTE row the packed row holds only the bytes attributes NAME, so a
            // stream can contribute a 4-byte colour slot out of a 68-byte record, and
            // `base[si] + column` is not a bound on anything.
            //
            // That is golf's red trees. The background quad's only colour is byte 64 of a
            // 68-byte PER-INSTANCE record; `16 + 68 > 20` sent every draw of that shape down
            // the `continue`, so the colour slot kept the zeros this buffer is allocated with
            // and the quad painted with whatever the fragment made of (0,0,0). The pieces are
            // built by `set_vertex_program` to fit the row it also sizes - they are the bound,
            // and the source rows must be long enough to cover them.
            let row_len = mine
                .iter()
                .map(|p| p.src as usize + p.len as usize)
                .max()
                .unwrap_or(0)
                .max(st.column as usize);
            let put = |dst: &mut [u8], row: &[u8]| {
                for pc in mine {
                    let (d, sv, l) = (pc.dst as usize, pc.src as usize, pc.len as usize);
                    dst[d..d + l].copy_from_slice(&row[sv..sv + l]);
                }
            };
            if st.stride == 0 {
                stamps.push((buf, row_len, stamp));
                let src = ctx.read_bytes(buf, row_len);
                for dst in vertices.chunks_exact_mut(stride_us) {
                    put(dst, &src);
                }
            } else if st.per_instance {
                let want = (instances - 1) * src_step + row_len;
                stamps.push((buf, want, stamp));
                let src = ctx.read_bytes(buf, want);
                for (r, dst) in vertices.chunks_exact_mut(stride_us).enumerate() {
                    let k = r / per;
                    put(dst, &src[k * src_step..k * src_step + row_len]);
                }
            } else {
                let at = buf.wrapping_add(min.wrapping_mul(st.stride));
                let want = (per - 1) * src_step + row_len;
                stamps.push((at, want, stamp));
                let src = ctx.read_bytes(at, want);
                for (r, dst) in vertices.chunks_exact_mut(stride_us).enumerate() {
                    let v = r % per;
                    put(dst, &src[v * src_step..v * src_step + row_len]);
                }
            }
        }
        crate::perf::note_bytes(crate::perf::Phase::DrawVertices, vertices.len());
        let mut indices = Vec::with_capacity(n * index_elem);
        for k in 0..instances {
            for &i in &wrapped {
                let e = (i - min) + (k * per) as u32;
                match index_elem {
                    2 => indices.extend_from_slice(&(e as u16).to_le_bytes()),
                    _ => indices.extend_from_slice(&e.to_le_bytes()),
                }
            }
        }
        let indices: Arc<[u8]> = Arc::from(indices);
        let idx = DrawIndices {
            bytes: indices.clone(),
            first_vertex: 0,
            vertex_count: rows as u32,
            gather: None,
        };
        // The identity share the plain path gets, so a static crowd section is one upload.
        let (interned, seen_before) = self.intern_vertices_noting_hit(&vertices);
        // >>> FILED ONLY ON THE SECOND SIGHTING, which is the gate the gather memo rests its
        // whole cost argument on: a title that rebuilds its instances into a rotating arena
        // never interns the same bytes twice, so it never files and pays one failed probe a
        // draw rather than a memo that can never hit. Nothing is filed without a scene stamp
        // either - without the dirty map there is no proof to offer later.
        if memo_on && seen_before && stamp != 0 {
            let cost = interned.len() + indices.len();
            // A QUARTER of the vertex snapshot budget, not the whole of it: this memo is a
            // SECOND hold on buffers the gather memo's budget already sized, and a frame's
            // instanced expansions are a handful of draws against its hundreds of gathers.
            // Sharing the full figure would let the two together take twice what either was
            // sized for, in a wasm heap that never gives a page back.
            let budget = vertex_snapshot_budget() / 4;
            if self.instanced_memo_bytes + cost > budget {
                let target = budget.saturating_sub(cost + budget / 8);
                let mut by_age: Vec<(u64, InstancedKey, usize)> = self
                    .instanced_memo
                    .iter()
                    .filter(|(_, e)| e.used != self.frame_seq)
                    .map(|(k, e)| (e.used, k.clone(), e.vertices.len() + e.indices.len()))
                    .collect();
                by_age.sort_by_key(|(u, ..)| *u);
                for (_, k, c) in by_age {
                    if self.instanced_memo_bytes <= target {
                        break;
                    }
                    self.instanced_memo.remove(&k);
                    self.instanced_memo_bytes = self.instanced_memo_bytes.saturating_sub(c);
                }
            }
            let prev = self.instanced_memo.insert(
                key.clone(),
                InstancedMemo {
                    indices: indices.clone(),
                    rows: rows as u32,
                    vertices: interned.clone(),
                    stamps,
                    used: self.frame_seq,
                },
            );
            if let Some(old) = prev {
                self.instanced_memo_bytes = self
                    .instanced_memo_bytes
                    .saturating_sub(old.vertices.len() + old.indices.len());
            }
            self.instanced_memo_bytes += cost;
        }
        self.instanced_key_scratch = key;
        Some((idx, interned))
    }

    /// The compacted twin of [`Self::gather_into_scratch`]: packed row `i` is guest row
    /// `gather[i]` of every per-vertex stream (`gather` sorted ascending). Rows close together
    /// are read as one run (`GATHER_RUN_GAP_BYTES`); a per-instance or stride-0 stream is one
    /// row repeated, exactly as the window gather treats it.
    #[allow(clippy::too_many_arguments)]
    pub(super) fn gather_rows_into_scratch(
        &mut self,
        ctx: &GuestCtx,
        streams: &[VertexStreamInfo],
        bound_streams: &[u32],
        layout: &RowLayout,
        stride: u32,
        gather: &[u32],
    ) -> Vec<u8> {
        let mut vertices = std::mem::take(&mut self.interleave_scratch);
        let stride_us = stride as usize;
        if stride_us == 0 || gather.is_empty() {
            vertices.clear();
            return vertices;
        }
        let need = gather.len() * stride_us;
        if vertices.len() < need {
            vertices.resize(need, 0);
        } else {
            vertices.truncate(need);
        }
        // The gaps the aligned pieces leave, zeroed before the scatter - see the same pass in
        // `gather_into_scratch` for why a pad byte nothing fetches still has to be defined.
        let pads = Self::row_pads(layout.pieces(), stride_us);
        if !pads.is_empty() {
            for dst in vertices.chunks_exact_mut(stride_us) {
                for &(a, b) in &pads {
                    dst[a..b].fill(0);
                }
            }
        }
        let mut src = std::mem::take(&mut self.gather_src);
        for (si, st) in streams.iter().enumerate() {
            if st.column == 0 {
                continue;
            }
            // The bound is this stream's PIECES, not its column - see `read_instanced_draw`,
            // where the column form of the same rule was a live `continue` that dropped a whole
            // per-instance stream under the per-attribute row. Here it was only ever an assert,
            // so the release path was already right and the debug one fired on a legal layout.
            let mine_pieces = Self::pieces_of(layout.pieces(), si);
            let row_len = mine_pieces
                .iter()
                .map(|p| p.src as usize + p.len as usize)
                .max()
                .unwrap_or(0)
                .max(st.column as usize);
            debug_assert!(
                mine_pieces
                    .iter()
                    .all(|p| p.dst as usize + p.len as usize <= stride_us),
                "stream {si} has a piece outside the packed row of {stride_us}"
            );
            let buf = bound_streams.get(si).copied().unwrap_or(0);
            let mine = Self::pieces_of(layout.pieces(), si);
            if st.per_instance || st.stride == 0 {
                if src.len() < row_len {
                    src.resize(row_len, 0);
                }
                crate::perf::time(crate::perf::Phase::DrawVertexGatherRead, || {
                    ctx.read_into(buf, &mut src[..row_len])
                });
                let row = &src[..row_len];
                for dst in vertices.chunks_exact_mut(stride_us) {
                    for pc in mine {
                        let (d, sv, l) = (pc.dst as usize, pc.src as usize, pc.len as usize);
                        dst[d..d + l].copy_from_slice(&row[sv..sv + l]);
                    }
                }
                continue;
            }
            let src_step = st.stride as usize;
            let mut k = 0usize;
            while k < gather.len() {
                // The run: consecutive referenced rows whose gaps are cheaper to read than
                // to skip with another crossing.
                let mut end = k;
                while end + 1 < gather.len()
                    && (gather[end + 1] - gather[end]) as usize * src_step <= GATHER_RUN_GAP_BYTES
                {
                    end += 1;
                }
                let first = gather[k];
                let addr = buf.wrapping_add(first.wrapping_mul(st.stride));
                let want = (gather[end] - first) as usize * src_step + row_len;
                if src.len() < want {
                    src.resize(want, 0);
                }
                crate::perf::time(crate::perf::Phase::DrawVertexGatherRead, || {
                    ctx.read_into(addr, &mut src[..want])
                });
                for j in k..=end {
                    let from = (gather[j] - first) as usize * src_step;
                    for pc in mine {
                        let (d, sv, l) = (pc.dst as usize, pc.src as usize, pc.len as usize);
                        let at = j * stride_us + d;
                        vertices[at..at + l].copy_from_slice(&src[from + sv..from + sv + l]);
                    }
                }
                k = end + 1;
            }
        }
        if src.len() > Self::GATHER_SCRATCH_CAP {
            src = Vec::new();
        }
        self.gather_src = src;
        vertices
    }

    /// The byte ranges of one packed row that NO stream column covers - see the call site.
    ///
    /// Returned by value and tiny: a row has at most one gap per stream plus a tail, and the
    /// list is built from the same `base`/`column` pair the scatter walks, so the two cannot
    /// disagree about where a column is. Empty for a tightly-tiled row, which is what the
    /// `VITASLOP_GXP_ALIGN_ROW=0` arm produces - so that arm pays nothing for this.
    /// The pieces belonging to stream `si`. The list is built in stream order, so this is a
    /// contiguous run - found by a linear scan over the handful of pieces a program has rather
    /// than by an index nothing else would use.
    pub(super) fn pieces_of(pieces: &[ScatterPiece], si: usize) -> &[ScatterPiece] {
        let start = pieces.iter().position(|p| p.stream as usize == si);
        match start {
            Some(a) => {
                let n = pieces[a..]
                    .iter()
                    .take_while(|p| p.stream as usize == si)
                    .count();
                &pieces[a..a + n]
            }
            None => &[],
        }
    }

    pub(super) fn row_pads(pieces: &[ScatterPiece], stride: usize) -> Vec<(usize, usize)> {
        let mut spans: Vec<(usize, usize)> = pieces
            .iter()
            .map(|p| {
                let b = p.dst as usize;
                (b, (b + p.len as usize).min(stride))
            })
            .collect();
        spans.sort_unstable();
        let mut pads = Vec::new();
        let mut at = 0usize;
        for (b, e) in spans {
            if b > at {
                pads.push((at, b));
            }
            at = at.max(e);
        }
        if at < stride {
            pads.push((at, stride));
        }
        pads
    }

    #[allow(clippy::too_many_arguments)]
    pub(super) fn gather_into_scratch(
        &mut self,
        ctx: &GuestCtx,
        streams: &[VertexStreamInfo],
        bound_streams: &[u32],
        layout: &RowLayout,
        stride: u32,
        first_vertex: u32,
        vertex_count: u32,
    ) -> Vec<u8> {
        let mut vertices = std::mem::take(&mut self.interleave_scratch);
        let stride_us = stride as usize;
        if stride_us == 0 {
            // Every used stream has stride 0, so there is nothing to gather and no row to
            // gather it into. Guarded because `chunks_exact_mut(0)` panics.
            vertices.clear();
            return vertices;
        }
        let need = vertex_count as usize * stride_us;
        // >>> NO WHOLE-BUFFER MEMSET, AND THAT IS AN INVARIANT RATHER THAN A GAMBLE.
        //
        // This was `clear()` then `resize(need, 0)`, which zeroes EVERY byte - several
        // megabytes a frame on a title that gathers 671 draws - immediately before the scatter
        // below writes every one of them again. Two properties make the fill dead work, and
        // both are enforced elsewhere in this file rather than assumed here:
        //
        //   * the streams TILE the packed row exactly. `set_vertex_program` builds
        //     `stream_base` as the running sum of the used streams' strides and `packed_stride`
        //     as their total, so the columns abut with no gap and no overlap.
        //   * `read_into` DEFINES every byte of its buffer - what guest memory does not cover
        //     it zeroes - so a short (or entirely out-of-range) guest buffer scatters zeros
        //     rather than leaving its rows untouched. That is the same output the old
        //     whole-buffer fill produced for that case, which is why this is a rewrite of the
        //     cost and not of the answer.
        //
        // So growth is the only case that needs a fill, and `truncate` writes nothing at all.
        if vertices.len() < need {
            vertices.resize(need, 0);
        } else {
            vertices.truncate(need);
        }
        // >>> THE PAD BYTES BETWEEN THE COLUMNS, WHICH NOTHING ELSE WRITES.
        //
        // The note above rests on the streams TILING the row exactly, and they no longer do:
        // `set_vertex_program` 4-aligns each column's base so the guest's own row can be BOUND
        // rather than copied, which leaves up to three bytes between columns and at the end.
        // Nothing FETCHES a pad byte - no attribute covers one - but everything downstream
        // HASHES the row and COMPARES it: the packed-vertex cache, the content intern, the
        // residency map and the determinism signature all read the whole buffer. Left at
        // whatever the reused scratch last held, they would differ frame to frame for geometry
        // that did not change, and every one of those caches would miss for ever.
        //
        // One pass, before the scatter, over the gaps only. A row has at most one gap per
        // stream and each is under four bytes, so this is single-digit bytes per vertex against
        // a scatter that moves the whole row.
        let pads = Self::row_pads(layout.pieces(), stride_us);
        if !pads.is_empty() {
            for dst in vertices.chunks_exact_mut(stride_us) {
                for &(a, b) in &pads {
                    dst[a..b].fill(0);
                }
            }
        }
        let mut src = std::mem::take(&mut self.gather_src);
        // >>> ONE CROSSING FOR THE WHOLE DRAW WHEN THE STREAMS SIT TOGETHER.
        //
        // Every guest read here is a call OUT of wasm and into JS: the browser's guest memory
        // is a `SharedArrayBuffer` the module cannot borrow, so `read` is
        // `dst.set(guest.subarray(..))` across the boundary. The BYTES are a memcpy and cheap;
        // the CALL is not [[vitaslop-count-calls-not-bytes-across-the-guest-boundary]], and a
        // two- or three-stream draw paid one per stream - ~1,500 crossings a frame on a retail
        // sports title that gathers 671 draws.
        //
        // A title that rebuilds its geometry into a rotating arena writes those streams next to
        // each other, so their union is usually barely larger than their sum. Reading the union
        // ONCE and slicing each stream out of it is exactly equivalent - the guest address space
        // is one contiguous mapping (`GuestCtx::offset` is `addr - base`), so a byte at a given
        // address lands at the same relative place either way, and `read_into` zero-fills
        // whatever falls past the end for the span exactly as it would per stream.
        //
        // Guarded on both sides of the trade. It is only taken when the union costs less than
        // the crossing it saves (`GATHER_SPAN_SLACK`, a few KB - the point is to avoid a call,
        // never to read a megabyte to save one), and only when the LOWEST address actually
        // maps: below the base a merged read is all zeros where the per-stream reads would
        // have found data, which is the one way this could differ.
        let span = self.gather_span(ctx, streams, bound_streams, first_vertex, vertex_count);
        if let Some((span_start, span_len)) = span {
            // Counted, because a merge that never fires changes nothing observable and would
            // look exactly like one that always does. See `Phase::DrawVertexGatherMerged`.
            crate::perf::note_hit(crate::perf::Phase::DrawVertexGatherMerged);
            if src.len() < span_len {
                src.resize(span_len, 0);
            }
            // The merged crossing, charged to the same phase as the per-stream ones so the two
            // are comparable - see `Phase::DrawVertexGatherRead`.
            crate::perf::time(crate::perf::Phase::DrawVertexGatherRead, || {
                ctx.read_into(span_start, &mut src[..span_len])
            });
        }
        for (si, st) in streams.iter().enumerate() {
            // A COLUMN of zero is the empty stream; a STRIDE of zero is a stream every vertex
            // reads the same row of - see `VertexStreamInfo::column`.
            if st.column == 0 {
                continue;
            }
            // The bound is this stream's PIECES, not its column - see `read_instanced_draw`,
            // where the column form of the same rule was a live `continue` that dropped a whole
            // per-instance stream under the per-attribute row. Here it was only ever an assert,
            // so the release path was already right and the debug one fired on a legal layout.
            let mine_pieces = Self::pieces_of(layout.pieces(), si);
            let row_len = mine_pieces
                .iter()
                .map(|p| p.src as usize + p.len as usize)
                .max()
                .unwrap_or(0)
                .max(st.column as usize);
            debug_assert!(
                mine_pieces
                    .iter()
                    .all(|p| p.dst as usize + p.len as usize <= stride_us),
                "stream {si} has a piece outside the packed row of {stride_us}"
            );
            let buf = bound_streams.get(si).copied().unwrap_or(0);
            // A per-instance stream is stepped by instance, not by vertex, so instance 0
            // reads row 0 for every vertex; a per-vertex stream's rows are contiguous.
            // Either way this is ONE guest read, then a scatter into the interleaved buffer.
            // A per-instance stream is stepped by instance, and a stride-0 stream is not
            // stepped at all: both read ONE row and repeat it into every packed vertex.
            let (addr, want, repeat) = if st.per_instance || st.stride == 0 {
                (buf, row_len, true)
            } else {
                (
                    buf.wrapping_add(first_vertex * st.stride),
                    (vertex_count as usize - 1) * st.stride as usize + row_len,
                    false,
                )
            };
            let rows = match span {
                // Already in hand: this stream's bytes are at its offset within the span.
                Some((span_start, _)) => {
                    let at = addr.wrapping_sub(span_start) as usize;
                    &src[at..at + want]
                }
                None => {
                    // High-water: grown when a bigger stream turns up and never shrunk, so a
                    // steady frame reads into the same allocation every time.
                    if src.len() < want {
                        src.resize(want, 0);
                    }
                    // The CROSSING, timed on its own - see `Phase::DrawVertexGatherRead`. The
                    // scatter below is a plain memcpy loop inside wasm and is a different cost
                    // with a different fix.
                    crate::perf::time(crate::perf::Phase::DrawVertexGatherRead, || {
                        ctx.read_into(addr, &mut src[..want])
                    });
                    &src[..want]
                }
            };
            // The row index is walked by `chunks_exact_mut` rather than recomputed per
            // vertex, and the source bound is checked once (by `want`) rather than per row.
            //
            // >>> ONE COPY PER PIECE, and under the column form there is exactly one piece per
            // stream, which is the copy this used to make - see [`ScatterPiece`].
            let mine = Self::pieces_of(layout.pieces(), si);
            if repeat {
                let row = &rows[..row_len];
                for dst in vertices.chunks_exact_mut(stride_us) {
                    for pc in mine {
                        let (d, sv, l) = (pc.dst as usize, pc.src as usize, pc.len as usize);
                        dst[d..d + l].copy_from_slice(&row[sv..sv + l]);
                    }
                }
            } else {
                // The SOURCE steps by the stream's own stride; each piece is a window inside
                // that row. The two are equal for an ordinary stream and differ whenever a
                // stream's attributes reach past its stride.
                let src_step = st.stride as usize;
                for (v, dst) in vertices.chunks_exact_mut(stride_us).enumerate() {
                    let from = v * src_step;
                    for pc in mine {
                        let (d, sv, l) = (pc.dst as usize, pc.src as usize, pc.len as usize);
                        dst[d..d + l].copy_from_slice(&rows[from + sv..from + sv + l]);
                    }
                }
            }
        }
        // Give an outsized read's buffer back rather than pinning it for the rest of the run -
        // see the field's own note. A steady frame's streams are far below the cap, so the
        // ordinary path keeps its allocation and this never fires.
        if src.capacity() > Self::GATHER_SCRATCH_CAP {
            src = Vec::new();
        }
        self.gather_src = src;
        vertices
    }

    pub(super) fn get_or_read_vertices(&mut self, ctx: &GuestCtx, addr: u32, len: usize) -> Arc<[u8]> {
        let _owner = vitaslop_platform::heap::owner_scope(vitaslop_platform::heap::Owner::GeometrySnapshot);
        if len == 0 {
            return Arc::from(&[][..]);
        }
        self.arm_scene_stamp(ctx);
        // A use - see the same stamp in `get_or_read`.
        self.vertex_used.insert((addr, len), self.frame_seq);
        // NO once-a-scene shortcut here - see `vertex_entries` for why geometry cannot take one.
        if let Some(p) = self.vertex_entries.get(&(addr, len)) {
            let clean = match self.vertex_stamps.get(&(addr, len)) {
                Some(&s) => ctx.dirty_since(addr, len, s) == Some(false),
                None => false,
            };
            if clean {
                return p.clone();
            }
        }
        // >>> COMPARED IN PLACE FIRST, where the memory can lend its bytes (the browser's
        // in-module region, the native engine). A dirty page is very often rewritten with the
        // SAME bytes (a static mesh rebuilt every frame), and the path below allocated and
        // copied the whole buffer only to find that out and throw the copy away. The answer is
        // the same buffer either way; this just skips the allocation and the copy when it is.
        // A baseball title moves ~4 MB/frame (desktop) - ~9 on a phone - through this read, on the game's
        // render thread, which is the phone's critical path (ovl26b).
        if inplace_compare()
            && let Some(p) = self.vertex_entries.get(&(addr, len))
            && let Some(live) = ctx.borrow_bytes(addr, len)
            && live.len() == p.len()
            && bytes_eq(live, p)
        {
            let same = p.clone();
            self.stamp_vertices(addr, len);
            return same;
        }
        // Straight into an `Arc` - see `read_bytes_arc` for the pass this saves on a path
        // that moves megabytes a frame.
        let raw = ctx.read_bytes_arc(addr, len);
        crate::perf::note_bytes(crate::perf::Phase::DrawVertices, raw.len());
        // Same bytes as last time? Hand back the SAME buffer. The guest may have rewritten the
        // page with identical contents (a rebuilt-every-frame buffer whose contents are static
        // is common), and a fresh `Arc` there would miss every downstream cache keyed on
        // identity while being byte-for-byte the thing they already hold.
        if let Some(p) = self.vertex_entries.get(&(addr, len))
            && bytes_eq(&raw, p)
        {
            let same = p.clone();
            self.stamp_vertices(addr, len);
            return same;
        }
        // ...and the same bytes held under a DIFFERENT address, which is what a title that
        // rebuilds its geometry into a rotating arena produces every frame. See
        // `vertex_by_content`. Proven by comparison, never by the fingerprint.
        //
        // >>> BEHIND THE SAME ARM AS THE INTERNING, because it changes the same thing: whether
        // a buffer's IDENTITY may outlive the address it was read from. A freed range must not
        // keep its old identity - `a_freed_range_drops_its_vertex_snapshot` pins that, and it
        // is what stops a later allocation at the same address inheriting the previous
        // geometry's downstream cache entries. The content index is sound about CONTENT and
        // says nothing about that, so it is off with the rest of the feature.
        let fp = vertex_fingerprint(&raw);
        if vertex_intern_enabled()
            && let Some(p) = self.vertex_by_content.get(&(len, fp))
            && bytes_eq(&raw, p)
        {
            let same = p.clone();
            // >>> A HIT IS A USE, AND THIS ONE WAS NOT RECORDED AS ONE.
            //
            // `vertex_content_used` is what `lru_victims` reads to decide which entries are
            // COLD, and only `intern_vertices_noting_hit` was stamping it. Every mesh
            // recognised through THIS path - the single-stream snapshot path, which is most
            // draws of most frames - was served from the index and then looked untouched to
            // the very next eviction. So the index evicted exactly the entries it was
            // answering from: MEASURED on one baseball-title run, **96,281 entries and 2,770 MB shed**,
            // ending at 485 entries against an 8,192 cap and 3 MB against a 64 MB budget,
            // while ~95 draws a frame arrived in a fresh allocation whose bytes the
            // renderer's packed cache still held. `VITASLOP_VERTEX_INTERN_USE=0` is the OFF
            // arm. [[vitaslop-an-intern-eviction-had-no-counter]]
            if vertex_intern_use_stamps() {
                self.vertex_content_used.insert((len, fp), self.frame_seq);
            }
            // NOT also recorded in `vertex_entries`: that map's byte accounting is per
            // ENTRY, and a rotating arena would file one shared buffer under a new address
            // every frame - counting the same megabytes N times and clearing the budget
            // early. The address entry buys nothing here anyway, because the next frame's
            // address is different again.
            //
            // >>> AND AN ENTRY THIS ADDRESS ALREADY HAD IS NOW WRONG, so it goes - it is not
            // re-stamped. This branch is reached only when that entry did NOT match the bytes
            // just read (the comparisons above), so it holds what the address held BEFORE.
            // Stamping the address here used to certify those old bytes as current, and the
            // next read of the address with no store in between - the end-of-scene re-read of
            // the same draw, or the next draw through it - handed them back. MEASURED on a
            // beach title's title screen: its marquee quads are rewritten into one of three
            // rotating buffers after the draw, the draw-time bytes matched a buffer held
            // elsewhere, and on every third frame the marquee was drawn from that buffer's
            // previous quad - mirrored and stretched across the top of the screen.
            if let Some(old) = self.vertex_entries.remove(&(addr, len)) {
                self.vertex_bytes -= old.len();
            }
            self.vertex_stamps.remove(&(addr, len));
            return same;
        }
        let bytes: Arc<[u8]> = raw;
        // A budget, and past it the COLDEST meshes are shed rather than all of them. The keys
        // are (address, length) rather than content, so an eviction costs a re-read and never
        // correctness - and geometry does turn over with the level, which is exactly what an
        // eviction by last use sheds and what a wholesale clear could only approximate by
        // throwing away the current level too. See [`lru_victims`].
        if self.vertex_bytes + bytes.len() > vertex_snapshot_budget() {
            self.evict_vertices_to_budget(bytes.len());
        }
        if let Some(old) = self.vertex_entries.insert((addr, len), bytes.clone()) {
            self.vertex_bytes -= old.len();
        }
        self.vertex_bytes += bytes.len();
        // Index it by content as well, so the next frame recognises it wherever the guest
        // rebuilds it. The key is derived from the bytes and every hit is proven by
        // comparison, so an eviction costs a repack and can never change what is answered.
        //
        // >>> COUNTED IN THE INDEX'S OWN BYTE BUDGET, exactly as `intern_vertices` counts what
        // it files. This used to file the buffer UNCOUNTED, on the reasoning that the address
        // map's budget covered it - but the address map evicts by ITS budget and leaves the
        // content entry behind, holding the buffer with no budget at all. On a title whose
        // gameplay writes ~36 MB of fresh geometry a frame the index reached its 8,192-entry
        // cap holding 2,049 MB of one-shot meshes (the heap ledger named the allocation site),
        // shed a quarter, and climbed again: the shape a device reports as a tab that dies.
        if vertex_intern_enabled() {
            if self.vertex_content_bytes + bytes.len() > vertex_snapshot_budget()
                || self.vertex_by_content.len() >= VERTEX_CONTENT_INDEX_CAP
            {
                self.evict_interned_vertices(bytes.len());
            }
            self.vertex_content_used.insert((len, fp), self.frame_seq);
            if let Some(old) = self.vertex_by_content.insert((len, fp), bytes.clone()) {
                self.vertex_content_bytes = self.vertex_content_bytes.saturating_sub(old.len());
            }
            self.vertex_content_bytes += bytes.len();
        }
        self.stamp_vertices(addr, len);
        bytes
    }

    /// The draw's index buffer, ALREADY scanned and rebased: `(indices, first_vertex,
    /// vertex_count)`.
    ///
    /// # What is cached is the answer, not the input
    /// A draw's index work is four passes: read the bytes out of guest memory, fold them for
    /// their min and max, rewrite every element to rebase it onto the `min..=max` vertex window,
    /// and convert the `Vec` into an `Arc` (a second allocation and a second full copy). All
    /// three outputs are a pure function of the source bytes, so a buffer the guest provably has
    /// not written since is worth exactly one dirty-map query and an `Arc` clone.
    ///
    /// Static geometry - a track, a hull, a UI mesh built once at load - answers "unchanged" for
    /// the whole run, which is the case this exists for. Dynamic geometry re-reads, which is
    /// precisely the old behaviour.
    ///
    /// # The window is part of the value, and it must be
    /// `first_vertex`/`vertex_count` are derived from the SAME bytes as the rebased buffer and
    /// are what the vertex read is sized from. Caching the buffer without them would hand a draw
    /// rebased indices alongside a freshly-computed window, and if those ever disagreed the mesh
    /// would index off the end of its own vertices - a prefix of the geometry, silently.
    pub(super) fn get_or_read_indices(
        &mut self,
        ctx: &GuestCtx,
        addr: u32,
        len: usize,
        elem: usize,
    ) -> DrawIndices {
        let _owner = vitaslop_platform::heap::owner_scope(vitaslop_platform::heap::Owner::GeometrySnapshot);
        if len == 0 {
            return DrawIndices {
                bytes: Arc::from(&[][..]),
                first_vertex: 0,
                vertex_count: 0,
                gather: None,
            };
        }
        self.arm_scene_stamp(ctx);
        let key = (addr, len, elem);
        // A use - see the same stamp in `get_or_read`.
        self.index_used.insert(key, self.frame_seq);
        // NO once-a-scene shortcut, for the reason `vertex_entries` gives: an index buffer is
        // geometry, and write-then-draw inside one scene is how dynamic meshes work.
        if let Some(v) = self.index_entries.get(&key) {
            let clean = match self.index_stamps.get(&key) {
                Some(&s) => ctx.dirty_since(addr, len, s) == Some(false),
                None => false,
            };
            if clean {
                return v.clone();
            }
        }
        let mut raw = crate::perf::time(crate::perf::Phase::DrawIndices, || {
            ctx.read_bytes(addr, len)
        });
        crate::perf::note_bytes(crate::perf::Phase::DrawIndices, raw.len());
        let index_of = |c: &[u8]| match elem {
            2 => u16::from_le_bytes([c[0], c[1]]) as u32,
            _ => u32::from_le_bytes([c[0], c[1], c[2], c[3]]),
        };
        let (first_vertex, vertex_count, gather) =
            crate::perf::time(crate::perf::Phase::DrawIndexScan, || {
                let (min_index, max_index) =
                    raw.chunks(elem).fold((u32::MAX, 0u32), |(lo, hi), c| {
                        let i = index_of(c);
                        (lo.min(i), hi.max(i))
                    });
                let (first_vertex, vertex_count) = if min_index > max_index {
                    (0, 0) // no indices at all
                } else {
                    (min_index, max_index - min_index + 1)
                };
                let write = |c: &mut [u8], i: u32| match elem {
                    2 => c[..2].copy_from_slice(&(i as u16).to_le_bytes()),
                    _ => c[..4].copy_from_slice(&i.to_le_bytes()),
                };
                // >>> A SPARSE DRAW IS COMPACTED: its vertices become the rows it references
                // and nothing else. The `min..=max` window is exact for a mesh drawn whole,
                // and it is the wrong shape for a draw that picks a few hundred vertices out
                // of a shared pool: MEASURED on a baseball title's stadium, ~30 draws a frame
                // with 18-1,700 indices each spanning 46K-65K rows of a 24-byte pool, 47 MB
                // of vertex bytes read, interleaved, hashed and uploaded per frame for well
                // under 1 MB of rows actually fetched. The renderer sees the same draw either
                // way - a vertex buffer and indices into it - so this changes no pixel.
                let n_idx = raw.len() / elem;
                if compact_sparse_draws()
                    && vertex_count as usize >= COMPACT_MIN_RANGE_ROWS
                    && vertex_count as usize > COMPACT_SPARSITY * n_idx
                {
                    let mut uniq: Vec<u32> = raw.chunks(elem).map(index_of).collect();
                    uniq.sort_unstable();
                    uniq.dedup();
                    for c in raw.chunks_mut(elem) {
                        let pos = uniq
                            .binary_search(&index_of(c))
                            .expect("every index is in its own unique set");
                        write(c, pos as u32);
                    }
                    let count = uniq.len() as u32;
                    return (0, count, Some(Arc::<[u32]>::from(uniq)));
                }
                if first_vertex > 0 {
                    for c in raw.chunks_mut(elem) {
                        let rebased = index_of(c) - first_vertex;
                        write(c, rebased);
                    }
                }
                (first_vertex, vertex_count, None)
            });
        let value = DrawIndices {
            bytes: Arc::<[u8]>::from(raw),
            first_vertex,
            vertex_count,
            gather,
        };
        // The same budget the vertex snapshots use, and shed the same way - coldest first. The
        // keys are (address, length, element size) rather than content, so an eviction costs a
        // re-read and can never cost correctness.
        if self.index_bytes + value.held_bytes() > vertex_snapshot_budget() {
            self.evict_indices_to_budget(value.held_bytes());
        }
        if let Some(old) = self.index_entries.insert(key, value.clone()) {
            self.index_bytes -= old.held_bytes();
        }
        self.index_bytes += value.held_bytes();
        if self.scene_stamp != 0 {
            self.index_stamps.insert(key, self.scene_stamp);
        }
        value
    }

    /// Would [`Self::get_or_read_indices`] answer this key from its cache without a read?
    /// The SAME test that function makes, with no side effect - so the flip-time prefetch can
    /// leave a range out of its table when the read it would feed is never going to happen.
    /// `ctx.dirty_since` here must be the OVERLAY's (answered from the stamp copy), or every
    /// probe is the crossing the table exists to avoid.
    pub(super) fn indices_cached(&self, ctx: &GuestCtx, addr: u32, len: usize, elem: usize) -> bool {
        let key = (addr, len, elem);
        self.index_entries.contains_key(&key)
            && matches!(self.index_stamps.get(&key), Some(&s) if ctx.dirty_since(addr, len, s) == Some(false))
    }

    /// The same for [`Self::get_or_read_vertices`].
    pub(super) fn vertices_cached(&self, ctx: &GuestCtx, addr: u32, len: usize) -> bool {
        self.vertex_entries.contains_key(&(addr, len))
            && matches!(self.vertex_stamps.get(&(addr, len)), Some(&s) if ctx.dirty_since(addr, len, s) == Some(false))
    }

    /// The same for [`Self::get_or_read_window`].
    pub(super) fn window_cached(&self, ctx: &GuestCtx, addr: u32, len: usize) -> bool {
        matches!(self.window_entries.get(&(addr, len)), Some((_, s)) if *s != 0 && ctx.dirty_since(addr, len, *s) == Some(false))
    }

    /// Would [`Self::gather_memo_hit`] answer `key`? The same proof, without the eviction a
    /// failed probe performs and without touching the use stamp - a probe, not a use.
    pub(super) fn gather_memo_cached(&self, ctx: &GuestCtx, key: &GatherKey) -> bool {
        self.gather_memo.get(key).is_some_and(|e| {
            e.stamps
                .iter()
                .all(|&(addr, len, stamp)| ctx.dirty_since(addr, len, stamp) == Some(false))
        })
    }

    /// Record that a vertex snapshot's bytes are current as of this scene. See [`Self::restamp`].
    pub(super) fn stamp_vertices(&mut self, addr: u32, len: usize) {
        if self.scene_stamp != 0 {
            self.vertex_stamps.insert((addr, len), self.scene_stamp);
        }
    }

    /// Advance the guest-store epoch for a new scene, at the FIRST snapshot the scene takes.
    ///
    /// Done lazily here rather than in [`Self::begin_scene`] because that has no [`GuestCtx`] to
    /// reach the map through, and because a scene that snapshots nothing should not spend an
    /// epoch value at all - a title with many texture-free passes would otherwise burn the byte
    /// on scenes that never ask a question of it.
    pub(super) fn arm_scene_stamp(&mut self, ctx: &GuestCtx) {
        if self.scene_stamp != 0 {
            return;
        }
        // Reclaim the epoch range before spending the last of it, rather than letting the
        // wrap disown every snapshot this cache holds. See `Self::rebase_epoch_if_exhausted`.
        self.rebase_epoch_if_exhausted(ctx);
        let Some((stamp, wrapped)) = ctx.bump_dirty_epoch() else {
            return;
        };
        self.stamp_stacks(ctx);
        // A wrapped epoch zeroed the map, so every stamp taken before it describes a page whose
        // record is gone. They are dropped, and each entry pays one compare to earn a new one.
        // BOTH sets of stamps, because both rest on the same map.
        if wrapped {
            crate::perf::note_epoch_wrap();
            self.stamps.clear();
            self.vertex_stamps.clear();
            self.gather_memo.clear();
            self.gather_memo_bytes = 0;
            // The instanced expansion memo rests on the SAME map and the same stamps - see
            // `instanced_memo`. Left behind here it would answer from stamps describing pages
            // whose record is gone, which is a stale mesh with nothing to say why.
            self.instanced_memo.clear();
            self.instanced_memo_bytes = 0;
            self.window_entries.clear();
            self.window_bytes = 0;
            self.index_stamps.clear();
        }
        self.scene_stamp = stamp;
    }

    /// >>> RENUMBER THE EPOCH RATHER THAN WRAP IT, when there is room to.
    ///
    /// The epoch is one byte and this cache spends one per SCENE, so a race frame burns
    /// eleven and the 253 usable values are gone in about ten presented frames. A wrap zeroes
    /// the map and drops every stamp, and the next use of each retained snapshot then copies
    /// the whole texture across the guest boundary to compare it - MEASURED at **5.40 MB per
    /// presented frame** on a racing title's on-track frame in the browser, every byte of it
    /// identical.
    ///
    /// The live stamps are not spread across the range: every snapshot proved current is
    /// re-stamped, so they cluster just under the epoch. Everything below the lowest live one
    /// is free, and [`GuestMemory::rebase_dirty_epoch`] hands it back - the map, the epoch and
    /// the stamps here all shift down by the same amount, which leaves the `page >= stamp`
    /// predicate exactly as it was.
    ///
    /// Not attempted when there is little to reclaim: renumbering costs a pass over the map,
    /// and doing it for a handful of values would just repeat next scene. Below the threshold
    /// the wrap happens exactly as it always did.
    pub(super) fn rebase_epoch_if_exhausted(&mut self, ctx: &GuestCtx) {
        /// How close to the ceiling the epoch has to be before renumbering is worth a pass
        /// over the map. `bump_dirty_epoch` wraps at 255.
        const NEAR_CEILING: u8 = 250;
        /// How much of the range live stamps keep. An entry whose stamp is older than this
        /// is DROPPED rather than renumbered: without a stamp it falls through to one compare
        /// the next time it is used, which is the cost of one texture - against the whole
        /// working set, which is what a wrap costs. Everything below the floor is what the
        /// renumbering hands back.
        const RETAIN: u8 = 128;
        let Some(epoch) = ctx.dirty_epoch() else {
            return;
        };
        if epoch < NEAR_CEILING {
            return;
        }
        // The floor is CHOSEN, not measured. Taking the lowest live stamp instead sounds
        // safer and is useless: one entry established just after the last wrap and proven
        // clean ever since pins it at 1, and then there is nothing to reclaim - which is
        // exactly what happened, measured, before this was a window rather than a minimum.
        let floor = epoch.saturating_sub(RETAIN);
        if floor < 2 {
            return;
        }
        // An entry older than the window loses its stamp. That is not a correctness question:
        // no stamp means "no answer", and `get_or_read` already falls through to the compare
        // for one ([`GuestMemory::dirty_since`] is emphatic that `None` is not "clean").
        self.stamps.retain(|_, s| *s >= floor);
        self.vertex_stamps.retain(|_, s| *s >= floor);
        self.index_stamps.retain(|_, s| *s >= floor);
        // The gather memo's stamps rest on the same map and are rebased with it - an entry
        // proven at an epoch below the floor has no proof left and goes; the rest shift. A
        // memo left un-rebased here answered from stale proofs after every rebase (a golf
        // title's picture moved on 7 of 12 shots), exactly the failure a stamp is for.
        self.gather_memo
            .retain(|_, e| e.stamps.iter().all(|&(_, _, st)| st >= floor));
        self.gather_memo_bytes = self.gather_memo.values().map(|e| e.result.len()).sum();
        // ...and the INSTANCED expansion memo, on the same terms and for the same reason.
        self.instanced_memo
            .retain(|_, e| e.stamps.iter().all(|&(_, _, st)| st >= floor));
        self.instanced_memo_bytes = self
            .instanced_memo
            .values()
            .map(|e| e.vertices.len() + e.indices.len())
            .sum();
        self.window_entries.retain(|_, (_, st)| *st >= floor);
        self.window_bytes = self.window_entries.values().map(|(b, _)| b.len()).sum();
        let Some(_new_epoch) = ctx.rebase_dirty_epoch(floor) else {
            return;
        };
        // Every stamp owes the same subtraction the map just took. `floor` is the minimum, so
        // none of these can underflow.
        let shift = floor - 1;
        for s in self.stamps.values_mut() {
            *s -= shift;
        }
        for s in self.vertex_stamps.values_mut() {
            *s -= shift;
        }
        for s in self.index_stamps.values_mut() {
            *s -= shift;
        }
        for e in self.gather_memo.values_mut() {
            for st in e.stamps.iter_mut() {
                st.2 -= shift;
            }
        }
        for e in self.instanced_memo.values_mut() {
            for st in e.stamps.iter_mut() {
                st.2 -= shift;
            }
        }
        for (_, st) in self.window_entries.values_mut() {
            *st -= shift;
        }
        crate::perf::note_epoch_rebase();
    }

    /// >>> SHED THE COLDEST SNAPSHOTS UNTIL THE CACHE FITS, INSTEAD OF DROPPING ALL OF THEM.
    ///
    /// This replaces a `clear()` whose own comment priced it at **2.1 seconds of `build`**, and
    /// which the user's device reported doing over and over at `held_mb=191 entries=3264` - hard
    /// against a 192 MB cap, so it re-fired as fast as the working set could be rebuilt. See
    /// [`lru_victims`] for the four dumps that show what that costs across a session.
    ///
    /// # The two maps that would have made this buy nothing
    /// `snapshot_sets` and `decoded` hold `Arc` clones of the very buffers `entries` holds, so
    /// dropping an `entries` key while a memo still names it reclaims NO memory - it only makes
    /// the byte count lie. Both are re-derivable and both already carry their own recency
    /// (`valid_scene`), so the entries that are not current are dropped FIRST, before anything is
    /// weighed: an entry the current scene has not proven is one the next use would re-prove
    /// anyway, so this costs a re-derivation and never an answer.
    ///
    /// `want` is the incoming buffer's size, so the eviction makes room for it rather than
    /// merely getting back under the line and immediately crossing it again.
    pub(super) fn evict_textures_to_budget(&mut self, want: usize) {
        let budget = texture_snapshot_budget();
        // >>> THE MEMO SWEEP IS ONCE PER FRAME, NOT ONCE PER INSERT.
        //
        // A title sitting hard against its budget - which is exactly what the device reported,
        // `held_mb=191` of 192 - calls this on EVERY changed texture. Both maps are bounded at
        // 16,384 entries, so a `retain` over them per insert would be tens of thousands of
        // visits per frame: a cost that grows with cache size, imposed by the code that exists
        // to stop costs growing with cache size. Once a frame is enough, because that is the
        // cadence at which `valid_scene` can have changed for anything.
        if self.memo_swept_frame != self.frame_seq {
            self.memo_swept_frame = self.frame_seq;
            // The memos pin the very buffers `entries` holds, so sweeping them BEFORE weighing
            // anything is what makes the eviction below reclaim memory rather than accounting.
            let scene = self.scene_seq;
            self.snapshot_sets.retain(|_, e| e.valid_scene == scene);
            self.decoded.retain(|_, e| e.valid_scene == scene);
        }
        // >>> EVICTED BELOW THE LINE, NOT TO IT. Freeing exactly enough for the incoming buffer
        // means the next insert is over budget again, so every insert would pay a sort of the
        // whole cache. An eighth of the budget of headroom turns that into one eviction pass per
        // many inserts, which is the same hysteresis the renderer's own caches use.
        let target = budget.saturating_sub(want + budget / 8);
        let victims = lru_victims(
            &self.entries,
            &self.used,
            self.bytes,
            target,
            self.frame_seq,
            |b: &Arc<[u8]>| b.len(),
        );
        let mut freed = 0usize;
        for k in &victims {
            if let Some(old) = self.entries.remove(k) {
                freed += old.len();
                self.bytes = self.bytes.saturating_sub(old.len());
            }
            // Everything keyed on this entry goes with it. A stamp or a cadence flag left
            // behind describes a buffer that is no longer here, and `pixel_ids` in particular
            // is an IDENTITY - a re-read mints a new one, so a stale entry could hand a fresh
            // buffer its predecessor's id and every cache keyed on that id would hit wrongly.
            self.pixel_ids.remove(k);
            self.stamps.remove(k);
            self.used.remove(k);
            self.mutable.remove(k);
            self.checked_this_scene.remove(k);
            self.checked_this_frame.remove(k);
            // The expansion of a paletted texture is keyed by the same guest range and is four
            // times its size; it cannot outlive the indices it was expanded from.
            if let Some(old) = self.palette_expanded.remove(k) {
                self.palette_bytes = self.palette_bytes.saturating_sub(old.pixels.len());
            }
        }
        snapcounts::note_evicted(victims.len() as u64, freed as u64);
        if victims.is_empty() && self.bytes + want > budget {
            // Nothing was evictable: every retained buffer belongs to the frame being built,
            // so the working set genuinely does not fit and the frame has to finish anyway.
            // REPORTED rather than silently done - this is the state the budget is wrong for,
            // and on a phone it is also how a worker gets killed with no error at all.
            report_snapshot_working_set_over_budget(self.bytes >> 20, self.entries.len());
        }
    }

    /// The vertex twin of [`Self::evict_textures_to_budget`]: shed the coldest meshes until the
    /// snapshots fit, with room for the incoming one.
    ///
    /// The CONTENT index holds `Arc` clones of exactly these buffers, so an eviction that did not
    /// reach it would drop the accounting and not the memory - it is swept on the same clock,
    /// just below.
    pub(super) fn evict_vertices_to_budget(&mut self, want: usize) {
        let budget = vertex_snapshot_budget();
        let victims = lru_victims(
            &self.vertex_entries,
            &self.vertex_used,
            self.vertex_bytes,
            // Below the line rather than to it, so this is one pass per many inserts - see
            // `evict_textures_to_budget` for why that matters on a cache under steady pressure.
            budget.saturating_sub(want + budget / 8),
            self.frame_seq,
            |b: &Arc<[u8]>| b.len(),
        );
        for k in &victims {
            if let Some(old) = self.vertex_entries.remove(k) {
                self.vertex_bytes = self.vertex_bytes.saturating_sub(old.len());
            }
            self.vertex_stamps.remove(k);
            self.vertex_used.remove(k);
        }
        use std::sync::atomic::Ordering::Relaxed;
        snapcounts::VERTEX_EVICTED.fetch_add(victims.len() as u64, Relaxed);
        // The interned copies of what was just dropped are what would keep it alive. Swept to
        // the same budget rather than cleared, for the same reason everything else here is.
        self.evict_interned_vertices(want);
    }

    /// Shed the coldest INTERNED vertex buffers - the content-addressed index that recognises a
    /// mesh the guest rebuilt at a new address.
    ///
    /// Bounded twice, by bytes and by entries, because an entry here is held by this map alone
    /// and its size is the mesh's. A drop costs one re-intern and can never change an answer:
    /// every hit is proven by a full comparison, so the index is a pure accelerator.
    pub(super) fn evict_interned_vertices(&mut self, want: usize) {
        let budget = vertex_snapshot_budget();
        let mut victims = lru_victims(
            &self.vertex_by_content,
            &self.vertex_content_used,
            self.vertex_content_bytes,
            budget.saturating_sub(want + budget / 8),
            self.frame_seq,
            |b: &Arc<[u8]>| b.len(),
        );
        // The ENTRY cap is the other half of the bound, and bytes alone will not satisfy it: a
        // map full of tiny meshes reaches the count long before the budget.
        if self.vertex_by_content.len() >= VERTEX_CONTENT_INDEX_CAP {
            let keep = VERTEX_CONTENT_INDEX_CAP - VERTEX_CONTENT_INDEX_CAP / 4;
            if self.vertex_by_content.len().saturating_sub(victims.len()) > keep {
                let mut by_age: Vec<((usize, u64), u64)> = self
                    .vertex_by_content
                    .keys()
                    .map(|k| (*k, self.vertex_content_used.get(k).copied().unwrap_or(0)))
                    .filter(|(_, u)| *u != self.frame_seq)
                    .collect();
                by_age.sort_by_key(|(_, u)| *u);
                let want_n = self.vertex_by_content.len() - keep;
                victims = by_age.into_iter().take(want_n).map(|(k, _)| k).collect();
            }
        }
        use std::sync::atomic::Ordering::Relaxed;
        snapcounts::VERTEX_INTERN_EVICTED.fetch_add(victims.len() as u64, Relaxed);
        for k in &victims {
            if let Some(old) = self.vertex_by_content.remove(k) {
                self.vertex_content_bytes = self.vertex_content_bytes.saturating_sub(old.len());
                snapcounts::VERTEX_INTERN_EVICTED_BYTES.fetch_add(old.len() as u64, Relaxed);
            }
            self.vertex_content_used.remove(k);
        }
    }

    /// The index twin of [`Self::evict_vertices_to_budget`].
    pub(super) fn evict_indices_to_budget(&mut self, want: usize) {
        let budget = vertex_snapshot_budget();
        let victims = lru_victims(
            &self.index_entries,
            &self.index_used,
            self.index_bytes,
            budget.saturating_sub(want + budget / 8),
            self.frame_seq,
            |v: &DrawIndices| v.held_bytes(),
        );
        for k in &victims {
            if let Some(old) = self.index_entries.remove(k) {
                self.index_bytes = self.index_bytes.saturating_sub(old.held_bytes());
            }
            self.index_stamps.remove(k);
            self.index_used.remove(k);
        }
        use std::sync::atomic::Ordering::Relaxed;
        snapcounts::INDEX_EVICTED.fetch_add(victims.len() as u64, Relaxed);
    }

    pub(super) fn insert(&mut self, addr: u32, len: usize, bytes: Arc<[u8]>) {
        if self.bytes + bytes.len() > texture_snapshot_budget() {
            self.evict_textures_to_budget(bytes.len());
        }
        if let Some(old) = self.entries.insert((addr, len), bytes.clone()) {
            self.bytes -= old.len();
        }
        // >>> ESTABLISHING AN ENTRY IS USING IT, and stamping here rather than only in
        // `get_or_read` is what makes that true on every path. `author` - the decoded video
        // picture, written by the ENGINE rather than read from the guest - reaches this function
        // without going through the read ladder, so without this line a picture would arrive
        // unstamped and be the FIRST thing the next eviction chose, every frame.
        self.used.insert((addr, len), self.frame_seq);
        // A NEW buffer is a new identity, and this is the only place one is established - see
        // `BoundTexture::pixels_id`. `insert` is reached exactly when the bytes were proven to
        // have CHANGED (or were read for the first time); the unchanged paths return the
        // retained `Arc` without coming here, so an unchanged texture keeps its id and every
        // cache keyed on it keeps hitting.
        self.pixel_ids
            .insert((addr, len), crate::capture::next_pixels_id());
        self.bytes += bytes.len();
    }

    /// >>> THE EXPANSION OF A PALETTED TEXTURE, AND THE IDENTITY THAT GOES WITH IT.
    ///
    /// A P8/P4 texture is turned into an ordinary `U8U8U8U8` image before anything downstream
    /// sees it (see `expand_paletted_texture`), and that expansion is a HOST-BUILT buffer, so
    /// it cannot inherit the snapshot layer's identity - the same indices under a different
    /// palette are different texels. It used to MINT a fresh id on every decode instead.
    ///
    /// That is a cache key that changes when nothing has changed. MEASURED on this title's
    /// menu, one browser frame: **20 textures re-expanded and re-uploaded every frame, 8.90 MB
    /// of them**, against `draw: snapshot textures (miss path)` of 0.00 MB - the snapshot layer
    /// was proving every one of them unchanged and the expansion was throwing that proof away.
    /// [[vitaslop-an-address-is-not-an-identity]] is the same lesson from the other side: an
    /// identity must be a function of the CONTENTS, and a fresh number is not.
    ///
    /// So the expansion is remembered against the two things it is a function of - the identity
    /// of the index buffer it read (`src`, which the snapshot layer already mints exactly) and
    /// the palette's own bytes (`pal`, hashed; it is at most 1 KB and it can change under a
    /// texture that did not). Same inputs, same buffer, same id, and every cache keyed on that
    /// id keeps hitting.
    pub(super) fn expanded_palette(
        &self,
        addr: u32,
        len: usize,
        src: u64,
        pal: u64,
    ) -> Option<&PaletteExpansion> {
        self.palette_expanded
            .get(&(addr, len))
            .filter(|e| e.src == src && e.pal == pal)
    }

    /// Record an expansion against its inputs - see [`Self::expanded_palette`]. One entry per
    /// paletted texture: a re-expansion replaces the entry it supersedes, so this is bounded by
    /// the working set rather than by the run.
    pub(super) fn remember_palette(&mut self, addr: u32, len: usize, e: PaletteExpansion) {
        // >>> BOUNDED, because this map PINS buffers nothing else would keep. An expansion is
        // four times its indices, one entry per paletted texture, and a title that reaches new
        // paletted art all run would otherwise grow it to the run.
        //
        // It used to clear WHOLE, and the device reported it doing so at `held_mb=95
        // entries=169`. Every entry is re-derivable, so that was safe - and it cost a
        // re-expansion AND a re-upload of every paletted texture in the next frame, which is
        // precisely the work this memo exists to avoid, repeated as fast as the cap could be
        // reached. The coldest are shed instead, on the same recency as the snapshots they are
        // keyed alongside (`used`, stamped by `get_or_read` - a paletted texture's expansion is
        // used exactly when its indices are).
        if self.palette_bytes + e.pixels.len() > palette_memo_budget() {
            let budget = palette_memo_budget();
            let victims = lru_victims(
                &self.palette_expanded,
                &self.used,
                self.palette_bytes,
                budget.saturating_sub(e.pixels.len() + budget / 8),
                self.frame_seq,
                |x: &PaletteExpansion| x.pixels.len(),
            );
            for k in &victims {
                if let Some(old) = self.palette_expanded.remove(k) {
                    self.palette_bytes = self.palette_bytes.saturating_sub(old.pixels.len());
                }
            }
        }
        self.palette_bytes += e.pixels.len();
        if let Some(old) = self.palette_expanded.insert((addr, len), e) {
            self.palette_bytes = self.palette_bytes.saturating_sub(old.pixels.len());
        }
    }

    /// The identity of the buffer currently held for `(addr, len)` - see
    /// [`crate::capture::BoundTexture::pixels_id`]. `0` when nothing is held, which is the
    /// same id the shared zero-texel buffer carries and is correct for both: neither names a
    /// distinct snapshot.
    pub(super) fn pixels_id(&self, addr: u32, len: usize) -> u64 {
        self.pixel_ids.get(&(addr, len)).copied().unwrap_or(0)
    }

    /// Clear the per-scene stamp so the next snapshot takes a fresh epoch, which is what
    /// `begin_scene` plus the next scene's first snapshot do together. Tests only.
    #[cfg(test)]
    pub(super) fn end_scene_for_test(&mut self) {
        self.scene_stamp = 0;
    }

    /// Start a new scene: every retained entry becomes due for one comparison again, and the
    /// next snapshot takes a fresh guest-store epoch.
    pub(super) fn begin_scene(&mut self) {
        self.checked_this_scene.clear();
        // The finished-list and per-binding memos are NOT cleared: each entry carries the
        // scene it was last proven in, and its first use in a new scene re-proves it against
        // the guest-store epoch before it is believed (see [`SetEntry::deps`]). The proof is
        // the same `get_or_read` ladder a rebuild would run, so keeping the entry changes
        // what is REDERIVED, never what is answered. `VITASLOP_TEX_MEMO_PER_SCENE` is the
        // A/B arm: the old cadence, cleared whole.
        if tex_memo_per_scene() {
            self.snapshot_sets.clear();
            self.decoded.clear();
        }
        self.scene_seq += 1;
        // Zero means "this scene has not taken one yet" - `arm_scene_stamp` fills it in on the
        // first snapshot. It is not a valid stamp: the epoch counter restarts at 1 after a wrap
        // and never returns to 0, so a stale 0 can never be mistaken for a real one.
        self.scene_stamp = 0;
    }

    /// Start a new FRAME: even a texture never seen to change is due for one comparison.
    /// See [`Self::get_or_read`] for why the two cadences exist.
    pub(super) fn begin_frame(&mut self) {
        self.checked_this_frame.clear();
        // The recency clock for every eviction in this struct. It is a FRAME rather than a
        // scene because a frame is what a working set is a property of: a title's passes bind
        // different subsets of the same textures, and an entry unused by one scene of a frame
        // is not therefore cold.
        self.frame_seq = self.frame_seq.wrapping_add(1);
        // Occupancy, for the panel line - see [`snapshot_cache_report`]. Five relaxed stores a
        // frame, against a symptom that previously needed a 48,000-frame session to see.
        {
            use std::sync::atomic::Ordering::Relaxed;
            snapcounts::HELD_BYTES.store(self.bytes as u64, Relaxed);
            snapcounts::ENTRIES.store(self.entries.len() as u64, Relaxed);
            snapcounts::VERTEX_BYTES.store(self.vertex_bytes as u64, Relaxed);
            snapcounts::VERTEX_CONTENT_BYTES.store(self.vertex_content_bytes as u64, Relaxed);
            snapcounts::VERTEX_CONTENT_ENTRIES.store(self.vertex_by_content.len() as u64, Relaxed);
            snapcounts::INDEX_BYTES.store(self.index_bytes as u64, Relaxed);
        }
    }

    /// The memoised decode for `key`, PROVEN current for this scene - or `None` when there is
    /// no entry or the entry could not be proven (it is dropped here and the caller rebuilds).
    ///
    /// The proof: the pixels came out of [`Self::get_or_read`] under `snap`, so asking it
    /// again this scene and getting the SAME buffer back (`Arc::ptr_eq`) re-establishes every
    /// byte through the same stamps-then-compare ladder a rebuild would use. A changed texture
    /// returns a fresh buffer, the pointers differ, and the entry dies.
    /// Bound on distinct per-binding decodes held across a long run, now that the map survives
    /// scenes. Past it the entries proven LONGEST AGO go - re-derivation, never correctness -
    /// like [`TEXTURE_TEMPLATE_CAP`] and the set cache's own cap.
    pub(super) const DECODED_CAP: usize = 16_384;

    pub(super) fn decoded_validated(
        &mut self,
        ctx: &GuestCtx,
        key: (u32, [u32; 4]),
    ) -> Option<Result<crate::capture::BoundTexture, u8>> {
        // >>> THE ALREADY-PROVEN CASE TAKES ONE PROBE AND TOUCHES NOTHING ELSE.
        //
        // This used to read the entry, clone the pixel `Arc` out of it, compare the scene, and
        // then probe the map a SECOND time to clone the result - so the overwhelmingly common
        // answer ("proven earlier in this scene") paid two hashes and a refcount pair it had no
        // use for. It is called once per dependency of every set re-proven in a scene, which is
        // ~1,150 times a frame on a race.
        let (snap, degraded) = match self.decoded.get(&key) {
            None => return None,
            Some(e) if e.valid_scene == self.scene_seq => return Some(e.res.clone()),
            Some(e) => (e.snap, e.degraded),
        };
        let pixels = self
            .decoded
            .get(&key)
            .and_then(|e| e.res.as_ref().ok())
            .map(|t| t.pixels.clone());
        // A dropped or degraded decode is re-attempted each new scene, exactly as it was
        // when this memo died with its scene: a texture that becomes readable is picked up
        // at the next scene, not never.
        let Some(held) = pixels.filter(|_| !degraded) else {
            self.decoded.remove(&key);
            return None;
        };
        let (addr, len) = snap.expect("an Ok entry always records its snapshot key");
        let current = self.get_or_read(ctx, addr, len);
        if Arc::ptr_eq(&current, &held) {
            // Re-fetched rather than held across the `get_or_read`: that call can clear
            // this very map (its insert enforces the snapshot budget), and then there is
            // nothing left to stamp - which falls through to the rebuild, correctly.
            if let Some(e) = self.decoded.get_mut(&key) {
                e.valid_scene = self.scene_seq;
                return Some(e.res.clone());
            }
        }
        self.decoded.remove(&key);
        None
    }

    /// The finished list for this exact set of bindings, PROVEN current for this scene.
    ///
    /// `tag` and `bindings` are the exact key material; `key` is their 64-bit fold, and the
    /// fold alone is never trusted - a hit whose stored bindings differ is a collision and is
    /// treated as a miss ([[vitaslop-content-hash-cache-must-verify]]).
    pub(super) fn set_validated(
        &mut self,
        ctx: &GuestCtx,
        key: u64,
        tag: u64,
        bindings: &[TextureBinding],
    ) -> Option<Arc<[crate::capture::BoundTexture]>> {
        let entry = self.snapshot_sets.get(&key)?;
        if entry.tag != tag
            || entry.bindings.len() != bindings.len()
            || !entry
                .bindings
                .iter()
                .zip(bindings)
                .all(|(&(u, a, w), b)| u == b.unit && a == b.addr && w == b.words)
        {
            report_set_key_collision();
            return None;
        }
        if entry.valid_scene == self.scene_seq {
            return Some(entry.list.clone());
        }
        let _reproof = crate::perf::scope(crate::perf::Phase::DrawTexBindReproof);
        // >>> THE DEPENDENCY LIST IS BORROWED OUT AND PUT BACK, NOT CLONED.
        //
        // The loop below needs `self` borrowed mutably while it walks the deps, and the old
        // way to get that was to CLONE the list - an allocation plus a refcount pair per
        // dependency, on a path that runs once per set per scene (~290 times a frame on a
        // race, ~1,150 dependencies). Taking the `Vec` leaves an empty one behind, which is
        // not a state anything can observe: the guest cannot run inside a host call, and every
        // exit below either restores it or removes the entry outright.
        let Some(mut deps) = self.snapshot_sets.get_mut(&key).and_then(|e| e.deps.take()) else {
            // Built with a dropped or degraded unit: it dies with its scene, so the drop is
            // counted - and the read retried - exactly as often as before.
            self.snapshot_sets.remove(&key);
            return None;
        };
        let e = self.snapshot_sets.get(&key)?;
        let (list, null_drops) = (e.list.clone(), e.null_drops);
        let mut proven = true;
        for (dep_key, held) in deps.iter() {
            // The pointer compare against the SET's own buffer is what makes this exact
            // even when the per-binding entry was independently rebuilt this scene: a
            // "valid" decode holding DIFFERENT bytes than this list must kill the list.
            if !self.decoded_pixels_are(ctx, *dep_key, held) {
                proven = false;
                break;
            }
        }
        if !proven {
            self.snapshot_sets.remove(&key);
            return None;
        }
        // The null-handle units' per-build drop notes, re-issued so keeping the set does
        // not quiet the counter that reports them.
        for _ in 0..null_drops {
            note_texture_drop(0);
        }
        // `get_or_read` inside the loop can clear this whole map (its insert enforces the
        // snapshot budget), so the entry may be gone - in which case there is nothing to
        // restore and nothing to stamp, and the list this proved is still the right answer
        // for THIS draw.
        if let Some(e) = self.snapshot_sets.get_mut(&key) {
            e.deps = Some(std::mem::take(&mut deps));
            e.valid_scene = self.scene_seq;
        }
        Some(list)
    }

    /// Just the PIXELS of a memoised decode, proven current for this scene.
    ///
    /// The set re-proof compares pixel buffers by pointer and wants nothing else, so it does
    /// not need [`Self::decoded_validated`]'s full [`crate::capture::BoundTexture`] clone -
    /// which copies a dozen fields and a second `Arc` per dependency, ~1,150 times a frame.
    pub(super) fn decoded_pixels_are(
        &mut self,
        ctx: &GuestCtx,
        key: (u32, [u32; 4]),
        held: &Arc<[u8]>,
    ) -> bool {
        // >>> THE ANSWER IS A POINTER COMPARE, SO THE BUFFER IS NEVER CLONED OUT.
        //
        // The only caller is the set re-proof, which took the `Arc` out, compared it by
        // pointer and dropped it again - a refcount pair, both of them ATOMIC on this target,
        // once per dependency of every set re-proven in a scene (MEASURED on a baseball
        // title's gameplay: ~1,260 a frame, and `set_validated` is 2.4% of the busy worker).
        // Nothing ever wanted the bytes.
        if let Some(e) = self.decoded.get(&key)
            && e.valid_scene == self.scene_seq
        {
            return matches!(e.res.as_ref(), Ok(t) if Arc::ptr_eq(&t.pixels, held));
        }
        matches!(self.decoded_validated(ctx, key), Some(Ok(t)) if Arc::ptr_eq(&t.pixels, held))
    }

    /// The PREVIOUS draw's finished fragment list, if this draw's sampler block is byte
    /// identical to the one it was decoded from and the two draws are in the same scene and
    /// bound the same fragment program. See [`Self::last_set`] for why that is exact.
    pub(super) fn set_from_previous_draw(
        &self,
        span: &[u8],
        fheader: u32,
    ) -> Option<Arc<[crate::capture::BoundTexture]>> {
        let last = self.last_set.as_ref()?;
        if last.fheader != fheader || &*last.span != span {
            return None;
        }
        // >>> THE ENTRY IS RE-PROBED, AND THAT IS WHAT MAKES THIS SAFE TO ADD.
        //
        // The list is only believed while the set cache still holds it, still under the same
        // key, still proven for THIS scene, and still the same buffer. Every invalidation
        // this cache has - a new scene, a freed range, a re-initialised texture format, a
        // budget clear, the `VITASLOP_TEX_MEMO_PER_SCENE` arm - drops or ages that entry, so
        // each of them refuses this shortcut too, with nothing to remember to do here. A memo
        // that carried its own scene number instead would have to be cleared at seven separate
        // sites and silently serve a stale list the first time a new one was added.
        let entry = self.snapshot_sets.get(&last.key)?;
        (entry.valid_scene == self.scene_seq && Arc::ptr_eq(&entry.list, &last.list))
            .then(|| last.list.clone())
    }

    /// Remember this draw's finished fragment list against the sampler bytes it came from.
    pub(super) fn remember_last_set(
        &mut self,
        span: &[u8],
        fheader: u32,
        key: u64,
        list: Arc<[crate::capture::BoundTexture]>,
    ) {
        match &mut self.last_set {
            // Reuse the box rather than allocate one per draw: the span is a fixed size, so
            // this is a copy into memory that is already the right shape.
            Some(last) if last.span.len() == span.len() => {
                last.span.copy_from_slice(span);
                last.fheader = fheader;
                last.key = key;
                last.list = list;
            }
            slot => {
                *slot = Some(LastSet {
                    span: span.to_vec().into_boxed_slice(),
                    fheader,
                    key,
                    list,
                })
            }
        }
    }

    /// The PREVIOUS draw's finished VERTEX-stage list, if nothing has rebound a vertex
    /// sampler since and the set-cache entry it came from is still current for this scene.
    pub(super) fn vertex_set_from_previous_draw(
        &self,
        generation: u64,
    ) -> Option<Arc<[crate::capture::BoundTexture]>> {
        let last = self.last_vertex_set.as_ref()?;
        if last.generation != generation {
            return None;
        }
        let entry = self.snapshot_sets.get(&last.key)?;
        (entry.valid_scene == self.scene_seq && Arc::ptr_eq(&entry.list, &last.list))
            .then(|| last.list.clone())
    }

    /// Remember this draw's finished VERTEX-stage list against the generation it came from.
    pub(super) fn remember_last_vertex_set(
        &mut self,
        generation: u64,
        key: u64,
        list: Arc<[crate::capture::BoundTexture]>,
    ) {
        self.last_vertex_set = Some(LastVertexSet {
            generation,
            key,
            list,
        });
    }

    /// Remember a finished list under its fold, with the exact key material and the
    /// per-binding facts a later scene's re-proof will check. See [`SetEntry`].
    pub(super) fn set_insert(
        &mut self,
        key: u64,
        tag: u64,
        bindings: &[TextureBinding],
        list: Arc<[crate::capture::BoundTexture]>,
    ) {
        let mut null_drops = 0u32;
        let mut deps = Some(Vec::with_capacity(bindings.len()));
        for b in bindings {
            // Mirrors `decode_texture`'s early arms exactly. `addr == 0` returns `None`
            // there - the unit is silently absent from the list, noting nothing - so it
            // contributes no dep and no re-issued note here. A null HANDLE (readable
            // address, all-zero control words) binds the constant zero texel and notes
            // drop 0 once per set build; constants need no re-proof, only the re-note.
            if b.addr == 0 {
                continue;
            }
            if b.is_null() {
                null_drops += 1;
                continue;
            }
            let dep_key = (b.addr, b.words);
            // The buffer recorded here is the one `list` holds for this unit: the list's
            // items are clones of the entry being probed, taken within this same host call,
            // so nothing can have rebuilt it in between. A unit whose decode dropped or
            // degraded (or whose entry the snapshot budget just evicted) makes the whole
            // set per-scene.
            let pixels = self
                .decoded
                .get(&dep_key)
                .and_then(|e| match (&e.res, e.degraded) {
                    (Ok(t), false) => Some(t.pixels.clone()),
                    _ => None,
                });
            match (deps.as_mut(), pixels) {
                (Some(d), Some(p)) => d.push((dep_key, p)),
                _ => deps = None,
            }
        }
        // A bound on distinct sets held across a long run, in the spirit of
        // [`TEXTURE_TEMPLATE_CAP`]. Shedding costs re-derivation and never correctness - the
        // entries re-prove themselves from live guest state - so it sheds the ones proven
        // LONGEST AGO, which each entry already records, rather than all of them.
        if self.snapshot_sets.len() >= SET_CAP {
            let scene = self.scene_seq;
            for k in count_victims(&self.snapshot_sets, SET_CAP - SET_CAP / 4, scene, |e| {
                e.valid_scene
            }) {
                self.snapshot_sets.remove(&k);
                snapcounts::SET_EVICTED.fetch_add(1, std::sync::atomic::Ordering::Relaxed);
            }
        }
        self.snapshot_sets.insert(
            key,
            SetEntry {
                list,
                tag,
                bindings: bindings.iter().map(|b| (b.unit, b.addr, b.words)).collect(),
                deps,
                null_drops,
                valid_scene: self.scene_seq,
            },
        );
        // A GAUGE, not a tally: published here because this is the only place the map grows.
        // See `snapcounts::SET_LIVE`.
        snapcounts::SET_LIVE.store(
            self.snapshot_sets.len() as u64,
            std::sync::atomic::Ordering::Relaxed,
        );
    }

    /// The mean colour of `t`'s pixels, computed once per distinct byte buffer.
    /// `None` for a texture whose format cannot be sampled, exactly as
    /// [`crate::render::texture_mean_rgb`] reports it.
    pub(super) fn mean_rgb(&mut self, t: &crate::capture::BoundTexture) -> Option<[f32; 3]> {
        let key = Arc::as_ptr(&t.pixels) as *const u8 as usize;
        if let Some((m, _)) = self.means.get(&key) {
            return Some(*m);
        }
        let m = crate::render::texture_mean_rgb(t)?;
        // Bounded by the snapshot cache it shadows: a buffer that is dropped there
        // would keep its entry alive here, so drop entries whose buffer this cache is
        // now the only owner of.
        self.means.retain(|_, (_, buf)| Arc::strong_count(buf) > 1);
        self.means.insert(key, (m, t.pixels.clone()));
        Some(m)
    }

    /// Drop every snapshot overlapping `[addr, addr+len)`, for memory the guest has
    /// released. Not needed for correctness (the comparison covers content changes),
    /// but a snapshot of freed memory is dead weight the cache should not hold.
    pub(super) fn invalidate_range(&mut self, addr: u32, len: usize) {
        // The per-scene lists hold `Arc`s to snapshots this is about to drop. Keeping them would
        // hand a later draw of the same scene a texture whose memory the guest has released,
        // which is the one thing this function exists to stop.
        self.snapshot_sets.clear();
        self.decoded.clear();
        // The content index is keyed by BYTES, not by address, so nothing in it can be located
        // by the range being freed - and a buffer it still holds would keep handing its
        // identity to a later allocation that happens to contain the same geometry. Dropped
        // whole: it is a pure accelerator, so losing it costs a re-read and never an answer.
        self.vertex_by_content.clear();
        self.vertex_content_bytes = 0;
        // The gather memo is keyed on guest spans and a freed range may be reallocated under a
        // memoised span with new bytes the dirty map has not seen; dropped whole, like the
        // content index, for the same reason: it is a pure accelerator.
        self.gather_memo.clear();
        self.gather_memo_bytes = 0;
        // The instanced expansion memo is keyed on guest spans too, and is dropped here for
        // exactly the same reason - see `instanced_memo`.
        self.instanced_memo.clear();
        self.instanced_memo_bytes = 0;
        self.window_entries.clear();
        self.window_bytes = 0;
        let end = addr as u64 + len as u64;
        let mut freed = 0usize;
        self.entries.retain(|&(a, l), bytes| {
            let overlaps = (a as u64) < end && (addr as u64) < (a as u64 + l as u64);
            if overlaps {
                freed += bytes.len();
            }
            !overlaps
        });
        // Keep the cadence bookkeeping in step with the entries it describes. A stale key
        // here is harmless (the `entries` lookup misses first and the texture is re-read),
        // but leaving it means the sets grow with every address the run ever touched.
        let overlapping =
            |&(a, l): &(u32, usize)| (a as u64) < end && (addr as u64) < (a as u64 + l as u64);
        self.mutable.retain(|k| !overlapping(k));
        self.checked_this_scene.retain(|k| !overlapping(k));
        self.checked_this_frame.retain(|k| !overlapping(k));
        self.bytes -= freed;

        // The same for VERTEX snapshots. Correctness does not rest on this either - freed memory
        // that comes back as a new mesh is WRITTEN by the guest, which stamps its pages, so the
        // dirty map reports it changed - but a vertex snapshot has no byte comparison behind it,
        // only the stamps, so leaving a freed range cached is a claim resting on one mechanism
        // where the texture path has two. Dropping it costs a re-read and removes the question.
        let mut vfreed = 0usize;
        self.vertex_entries.retain(|k, bytes| {
            if overlapping(k) {
                vfreed += bytes.len();
                return false;
            }
            true
        });
        self.vertex_stamps.retain(|k, _| !overlapping(k));
        self.vertex_bytes -= vfreed;
    }
}

#[cfg(test)]
mod covered_window_tests {
    use super::uncovered;

    /// The gaps a cover leaves are exactly the window's other bytes - before, between and after
    /// its ranges - and a cover past the window's end is clipped to it.
    #[test]
    fn uncovered_is_the_complement_of_the_cover() {
        let gaps = |len, cover: &[(u32, u32)]| uncovered(len, cover).collect::<Vec<_>>();
        assert_eq!(gaps(100, &[]), vec![(0, 100)]);
        assert_eq!(gaps(100, &[(0, 100)]), vec![]);
        assert_eq!(gaps(100, &[(8, 16), (40, 48)]), vec![(0, 8), (16, 40), (48, 100)]);
        assert_eq!(gaps(100, &[(0, 60), (60, 200)]), vec![]);
        assert_eq!(gaps(100, &[(20, 200)]), vec![(0, 20)]);
    }
}
