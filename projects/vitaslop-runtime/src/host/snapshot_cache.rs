use super::*;

/// Snapshots of guest texture bytes, keyed by (guest data address, byte length).
///
/// # The problem
/// The capture renderer needs a texture's PIXELS, and the only place they exist is
/// guest memory, so a draw that binds a texture has to read them out. Re-reading
/// every bound texture every frame is ~15 MB of allocate-and-copy per frame on a real
/// 3D title - measured at a QUARTER of the whole frame budget, the single most
/// expensive thing the capture did.
///
/// Almost none of that is a re-read of something that changed: a title uploads its
/// textures at load and samples them for the rest of the level.
///
/// # Why this compares instead of tracking invalidation
/// The obvious fix is to keep the snapshot until something invalidates it - a render
/// into that memory, a `sceGxmTextureInit*` over it, an unmap. That was tried, and a
/// verifier caught a texture this title writes DIRECTLY with the CPU, through no GXM
/// call at all. There is no call to hook, so no invalidation list can be complete,
/// and an incomplete one shows up as a stale texture: a silent visual bug.
///
/// So the snapshot is CHECKED rather than trusted. On each first use in a scene the
/// retained bytes are compared against guest memory borrowed in place
/// ([`GuestMemory::borrow`]) - no second copy - and only a texture that actually
/// changed is re-read. That is exactly as correct as re-reading everything, because
/// the compared bytes are the same bytes the copy would have taken; it just replaces
/// an allocate-plus-copy with a `memcmp` that early-exits on the first difference.
///
/// A backing that cannot lend its bytes (`borrow` returns `None`) falls back to
/// re-reading every scene - slower, still correct.
///
/// # And why the compare is skipped where the engine can prove it pointless
/// The compare is exact and it is enormous: measured on a live race, 116.8 MB a frame,
/// 40% of the whole frame, RE-READING 0.0 MB - memory bandwidth spent proving nothing
/// changed. An engine that stamps its guest stores
/// ([`GuestMemory::dirty_since`]) can answer the same question by reading one byte per
/// 4 KB page, and that is a proof, not a sample: it is the same guarantee, not a
/// weaker one. Where the engine offers no stamps, the `memcmp` still runs. Neither
/// path approximates, which is why there is no knob here.
#[derive(Default)]
pub(super) struct TextureSnapshots {
    pub(super) entries: FxHashMap<(u32, usize), Arc<[u8]>>,
    /// The identity minted for each entry's buffer - see
    /// [`crate::capture::BoundTexture::pixels_id`] for why an address will not do. Written
    /// only by [`Self::insert`], which is reached only when a NEW buffer is established.
    pub(super) pixel_ids: FxHashMap<(u32, usize), u64>,
    /// The RGBA8 a paletted texture expanded to, and the identity minted for THAT buffer -
    /// see [`Self::expanded_palette`]. Keyed by the index buffer's guest range, so one
    /// paletted texture holds one expansion.
    pub(super) palette_expanded: FxHashMap<(u32, usize), PaletteExpansion>,
    /// Bytes held by `palette_expanded`, against [`palette_memo_budget`]. An expansion is four
    /// times its indices and this map PINS it, so it needs a bound of its own - the decode
    /// cache's budget does not reach it.
    pub(super) palette_bytes: usize,
    /// The guest-store epoch each entry's bytes were established at, on an engine that
    /// stamps stores. Absent for an entry means "no proof available" and costs a
    /// compare, which is why a wrapped epoch can simply drop the whole map.
    pub(super) stamps: FxHashMap<(u32, usize), u8>,
    /// Total bytes held, so the cache cannot grow without bound across a long run.
    pub(super) bytes: usize,
    /// >>> FRAMES SINCE THE RUN STARTED, and the recency every eviction here orders by.
    ///
    /// Bumped by [`Self::begin_frame`]. An entry stamped with the current value is one THIS
    /// frame has already used and is never an eviction candidate - see [`lru_victims`].
    pub(super) frame_seq: u64,
    /// The frame each TEXTURE snapshot entry was last used in.
    pub(super) used: FxHashMap<(u32, usize), u64>,
    /// The frame each VERTEX snapshot entry was last used in.
    pub(super) vertex_used: FxHashMap<(u32, usize), u64>,
    /// The frame each INDEX snapshot entry was last used in.
    pub(super) index_used: FxHashMap<(u32, usize, usize), u64>,
    /// The frame each INTERNED vertex buffer was last recognised in, keyed like
    /// [`Self::vertex_by_content`] itself.
    pub(super) vertex_content_used: FxHashMap<(usize, u64), u64>,
    /// The frame [`Self::evict_textures_to_budget`] last swept the two per-scene memos in. They
    /// are bounded by ENTRY COUNT and a sweep visits all of them, so it happens once a frame
    /// rather than once per insert - see there.
    pub(super) memo_swept_frame: u64,
    /// Entries already checked this scene, so a texture bound by fifty draws is
    /// compared once, not fifty times. The cache's claim is "these bytes are
    /// unchanged since the last frame", and once per scene is what tests it.
    pub(super) checked_this_scene: FxHashSet<(u32, usize)>,
    /// The guest-store epoch every snapshot taken in the CURRENT scene is stamped with, or 0
    /// before this scene has taken one. See [`Self::restamp`] for why a scene shares one.
    pub(super) scene_stamp: u8,
    /// Every guest STACK the host has handed out ([`GuestStacks`]), stamped with every epoch
    /// this cache advances to ([`Self::stamp_stacks`]), because the transpiler leaves
    /// SP-relative stores unmarked - see [`GuestMemory::stamp_dirty_pages`].
    pub(super) stacks: GuestStacks,
    /// VERTEX buffer snapshots, by `(guest address, byte length)`.
    ///
    /// # Why vertices live in the texture cache, and why they are checked DIFFERENTLY
    /// They are here because they share the one thing that makes either work: the guest-store
    /// epoch, which is global guest state. Two caches arming their own scene stamps would spend
    /// the one-byte epoch twice as fast, which is the exact failure this whole mechanism just
    /// came back from.
    ///
    /// They are checked differently because the risk is not the same. A texture bound by fifty
    /// draws is compared once a scene, on the argument that a title does not rewrite a texture
    /// while drawing with it. That argument does NOT hold for geometry: writing vertices and
    /// drawing them, repeatedly, within one scene is exactly how dynamic geometry works - UI
    /// text, particles, anything skinned on the CPU. So a vertex read never takes the
    /// once-a-scene shortcut and always asks the dirty map, which is one page-range read against
    /// re-copying the whole buffer.
    ///
    /// The payoff is that STATIC geometry - a track, a ship hull, written once at load - has a
    /// page stamp far below the current epoch and is proved unchanged without being read at all.
    /// MEASURED on the device before this existed: `draw: read+interleave vertices` moved
    /// **6.06 MB every frame**, as a fresh allocation and a full copy per draw, 953 draws a
    /// frame, each one a boundary crossing in the browser.
    pub(super) vertex_entries: FxHashMap<(u32, usize), Arc<[u8]>>,
    /// Epoch each vertex snapshot's bytes were established at. Same contract as `stamps`.
    pub(super) vertex_stamps: FxHashMap<(u32, usize), u8>,
    /// >>> THE SAME GEOMETRY, REBUILT AT A NEW ADDRESS, RECOGNISED AS THE BUFFER ALREADY HELD.
    ///
    /// `vertex_entries` is keyed by ADDRESS, which is the right key for a title that keeps its
    /// meshes where it built them. It is the wrong key for one that writes its geometry into a
    /// rotating per-frame arena: the address is new every frame, so the entry above never
    /// matches, a fresh `Arc` is handed out for byte-identical geometry, and every cache
    /// downstream that keys on that buffer's IDENTITY misses.
    ///
    /// MEASURED on a golf title's gameplay frame, which does exactly that: the renderer's
    /// allocation-keyed packed-vertex cache served **1 draw in 492**, so essentially every draw
    /// fell through to `GxpLive::pack_vertices` and hashed its whole stream to find the packed
    /// bytes by CONTENT - **7.03 MB hashed per frame**, and then a full memcmp to verify the
    /// hit. A V8 profile of the worker put `fnv64` at 2.7% and that verify's `memcmp` at 6.2%
    /// of all busy time, for geometry that had not changed since the level loaded.
    ///
    /// So the identity is re-established HERE, once, where the bytes have just been read
    /// anyway: a cheap fingerprint finds the candidate and a full comparison proves it, exactly
    /// as the same-address path above does. Downstream then sees the buffer it already holds
    /// and every identity-keyed cache hits - including the resident geometry heap, which is the
    /// one that stops the bytes being uploaded again.
    ///
    /// The fingerprint is NEVER trusted on its own ([[vitaslop-content-hash-cache-must-verify]]):
    /// a collision costs a comparison that fails and a fresh entry, never a wrong mesh.
    pub(super) vertex_by_content: FxHashMap<(usize, u64), Arc<[u8]>>,
    /// Bytes held by `vertex_by_content` alone - the interned buffers that are NOT also in
    /// `vertex_entries` are held by nothing else, so they need their own budget.
    pub(super) vertex_content_bytes: usize,
    /// >>> THE GATHER MEMO: a multi-stream draw's interleaved buffer, keyed on the guest spans
    /// it was gathered from, valid while the dirty map proves none of those spans has been
    /// written since. See `read_draw_vertices`.
    pub(super) gather_memo: FxHashMap<GatherKey, GatherMemo>,
    pub(super) gather_memo_bytes: usize,
    /// >>> THE SAME MEMO FOR AN INSTANCED DRAW, which the one above cannot serve.
    ///
    /// [`Self::read_instanced_draw`] does not gather rows, it EXPANDS them: one packed row per
    /// (instance, vertex), from a wrapped index list plus a per-instance stream. Its result was
    /// interned, so its identity was already stable - but every frame re-read the index words,
    /// re-read every stream, re-scattered `instances * per` rows and then hashed the whole
    /// result to discover it was the same buffer. MEASURED on a baseball title's crowd in the
    /// desktop browser: **2.2% of the busy worker thread for 32 draws a frame**.
    ///
    /// Proven exactly as the gather memo is - dirty-map stamps over every span it read,
    /// INCLUDING the index words, since the expansion is a function of those too - and filed
    /// under the same discipline: only once the intern has already seen these bytes, so a title
    /// that rebuilds its instances into a rotating arena never files and pays one failed probe.
    pub(super) instanced_memo: FxHashMap<InstancedKey, InstancedMemo>,
    pub(super) instanced_memo_bytes: usize,
    /// The key [`Self::instanced_memo`] is PROBED with, reused across draws - see
    /// `gather_key_scratch`, which exists for the same reason.
    pub(super) instanced_key_scratch: InstancedKey,
    /// Uniform-buffer WINDOW snapshots, keyed by (address, bytes): the buffer and the scene
    /// stamp it was proven current at. A lean twin of `vertex_entries` - no content index,
    /// no per-use recency map - because a draw binds ~8 windows and pays every lookup:
    /// MEASURED at 1.9 us of a 3.9 us `sceGxmDraw` through the vertex path's bookkeeping.
    pub(super) window_entries: FxHashMap<(u32, usize), (Arc<[u8]>, u8)>,
    pub(super) window_bytes: usize,
    /// The interleave scratch, reused across draws. A multi-stream draw builds its packed
    /// vertex rows here before they are interned, so the common case - the same geometry as
    /// last frame - costs no allocation at all. See [`Self::intern_vertices`].
    pub(super) interleave_scratch: Vec<u8>,
    /// The [`GatherKey`] the gather memo is PROBED with, reused across draws so a probe costs
    /// no allocation. Its `spans` `Vec` is refilled per draw ([`Self::gather_spans_into`]) and
    /// the key is CLONED on the rare path that files an entry.
    pub(super) gather_key_scratch: GatherKey,
    /// Zero-filled `Arc`s by LENGTH, one per distinct length ever asked for - see
    /// [`Self::deferred_window_placeholder`].
    pub(super) window_placeholders: FxHashMap<usize, Arc<[u8]>>,
    /// Windows a raw 64-bit target this frame rendered COVERS, by `(address, length)` - see
    /// [`Self::get_or_read_covered_window`] - and, by the address of each one's bytes, which
    /// entry handed them out, so the resolve recognises one without scanning it.
    pub(super) covered_windows: FxHashMap<(u32, usize), CoveredWindow>,
    pub(super) covered_by_ptr: FxHashMap<usize, (u32, usize)>,
    /// The scratch ONE bound stream is read into before it is scattered into
    /// [`Self::interleave_scratch`]. Reused across streams and across draws.
    ///
    /// # Why this is not a `read_bytes` per stream
    /// It was, and on the title this path costs the most on that is **~1,500 fresh heap
    /// allocations a frame** (671 multi-stream draws, two or three bound streams each) whose
    /// contents are consumed and dropped a few microseconds later. Worse, `read_bytes`
    /// allocates a ZEROED buffer and then reads over every byte of it, and the browser's guest
    /// memory cannot be borrowed ([`GuestMemory::borrow`] is `None` on a `SharedArrayBuffer`),
    /// so the read is already a full copy across the JS boundary - the memset is a second full
    /// pass over megabytes a frame to produce bytes that are immediately overwritten.
    /// A high-water buffer read into with [`GuestCtx::read_into`] pays neither.
    ///
    /// >>> AND IT IS BOUNDED, which is not optional here. A pooled buffer that only ever grows
    /// > > > is how the last render-side arena change went wrong: the wasm heap is SHARED with the
    /// > > > guest and can never give a page back, so one loading frame's peak stays resident
    /// > > > forever and tightens allocation for everything, guest included - a cost no allocator
    /// > > > profile and no pixel oracle can see. See [`GATHER_SCRATCH_CAP`].
    pub(super) gather_src: Vec<u8>,
    /// Bytes held by `vertex_entries`, against [`vertex_snapshot_budget`].
    pub(super) vertex_bytes: usize,
    /// Entries already checked this FRAME. See [`Self::get_or_read`]: a texture that has
    /// never been observed to change is compared once per frame rather than once per
    /// scene, which is where nearly all of the compare bandwidth went.
    pub(super) checked_this_frame: FxHashSet<(u32, usize)>,
    /// Entries whose bytes HAVE been seen to change at least once. These keep the
    /// per-scene cadence, because they are the ones the cadence exists for.
    pub(super) mutable: FxHashSet<(u32, usize)>,
    /// Mean colour of a snapshot, keyed by the identity of the byte buffer itself and
    /// holding a strong reference to it.
    ///
    /// The flat ambient term of every lit draw is the average of the bound irradiance
    /// map ([`crate::render::texture_mean_rgb`]), which samples a thousand texels and
    /// decodes each - per DRAW, for a value that is a property of the texture. Keyed
    /// by the buffer's address rather than the texture's guest address because that is
    /// what actually determines the answer: a re-read allocates a new buffer, so a
    /// changed texture cannot hit a stale mean. The strong reference is what makes the
    /// address a valid key - without it a freed buffer's address could be reused.
    pub(super) means: FxHashMap<usize, ([f32; 3], Arc<[u8]>)>,
    /// INDEX buffer snapshots, already scanned and REBASED, by `(guest address, byte length,
    /// element size)`.
    ///
    /// # Why the derived value and not the bytes
    /// Every draw read its index buffer out of guest memory, folded the whole thing for its
    /// min/max, rewrote every element to rebase it onto the vertex window, and then converted the
    /// `Vec` into an `Arc` - a second allocation and a second full copy. Four passes over the
    /// indices per draw, ~650 draws a frame, and in the browser the read is also a crossing into
    /// JS. MEASURED on the browser-like configuration of `bench --at 7400`: `draw: read indices`
    /// 1.4% and `draw: scan/rebase indices` 2.3% of the window, on top of the allocation churn
    /// neither of those timers can see.
    ///
    /// All three outputs are a pure function of the source bytes, so the cache holds the ANSWER
    /// rather than the input: a hit costs one dirty-map query and an `Arc` clone, and does none
    /// of the four passes.
    ///
    /// Checked like VERTICES, not like textures - it always asks the dirty map and never takes
    /// the once-a-scene shortcut. Geometry written and drawn within one scene is exactly how
    /// dynamic meshes work, and an index buffer is geometry.
    pub(super) index_entries: FxHashMap<(u32, usize, usize), DrawIndices>,
    /// Epoch each index snapshot was established at. Same contract as `stamps`.
    pub(super) index_stamps: FxHashMap<(u32, usize, usize), u8>,
    /// Bytes held by `index_entries`, against [`vertex_snapshot_budget`] - the same budget,
    /// because they are the same kind of thing and a level's geometry turns over together.
    pub(super) index_bytes: usize,
    /// Everything [`decode_texture`] derives from a binding's four CONTROL WORDS, memoised by
    /// those words. `None` is a binding the decode drops (an unsizeable format), which is worth
    /// remembering for exactly the same reason.
    ///
    /// # Why this is the largest capture phase left, and why a memo is the answer
    /// `draw: snapshot textures` MEASURED at **11.7% of a browser-like race window, moving
    /// 0.0 MB** - so it is not volume, it is per-draw arithmetic repeated on bindings that do
    /// not change. Every draw re-derived, per bound unit: the recorded format (a hash lookup),
    /// the nearby-handle diagnostic (another), the base format and swizzle, the size fields, the
    /// block geometry, the mip-level clamp and the whole chain's byte extent
    /// ([`crate::render::level_offset`] walks every level). A race frame does that ~2,000 times
    /// for a handful of distinct textures.
    ///
    /// All of it is a pure function of the control words, which is exactly what a GXM binding
    /// IS: the words are copied BY VALUE at bind time ([[vitaslop-texture-binding-by-value]]),
    /// so two bindings with the same words describe the same texture. What is NOT cached is the
    /// PIXELS - those still go through [`Self::get_or_read`] every time, so a texture the guest
    /// rewrites is still seen to change. The memo covers the description; the snapshot covers
    /// the contents.
    ///
    /// Cleared when a recorded format changes (see `set_texture_format`), because that is the
    /// one input that is not in the key.
    /// Each entry carries the SCENE it was last used in, so the cap below sheds the control-word
    /// sets this title has stopped binding rather than all of them.
    pub(super) templates: FxHashMap<[u32; 4], (Option<TextureTemplate>, u64)>,
    /// A whole DRAW's worth of snapshotted textures, by the bindings that produced it - kept
    /// ACROSS scenes and RE-PROVEN on first use in each new scene.
    ///
    /// # Why a cache of the whole answer is exactly equivalent
    /// Within a scene: [`Self::get_or_read`] compares a texture at most once per scene; the
    /// first draw of the scene that binds it establishes the buffer, and every later draw in
    /// that scene gets the SAME `Arc` back by construction ("the guest cannot have run since -
    /// a host call is not preemptible"). So for a fixed set of bindings, every draw of a scene
    /// was already guaranteed to produce a bitwise identical list; this just stops rebuilding
    /// it.
    ///
    /// Across scenes: the guest HAS run, so nothing is believed until it is re-proven. A kept
    /// entry carries the [`Self::decoded`] keys and pixel buffers it was built from
    /// ([`SetEntry::deps`]); the first hit of a new scene runs each through
    /// [`Self::get_or_read`] - the stamps-then-compare ladder a rebuild would use - and the
    /// entry survives only if every buffer comes back POINTER-identical. That is the same
    /// proof, so keeping the entry changes what is rederived, never what is answered. This
    /// used to be cleared whole at every scene start, and a race frame is ~11 scenes binding
    /// the same sets, so ~45% of draws paid a full set rebuild for an answer the stamps
    /// already knew. `VITASLOP_TEX_MEMO_PER_SCENE=1` restores that cadence as the A/B arm.
    ///
    /// A race frame binds a handful of distinct texture sets across ~650 draws, so the hit rate
    /// is essentially the draw count. What a hit avoids is the whole of `decode_texture` per
    /// unit - the template probe, the recorded-format probe, the snapshot probe and the
    /// `Vec<BoundTexture>` the draw would otherwise allocate and fill.
    ///
    /// The key folds the fragment program header as well as the bindings, because the ALBEDO
    /// REORDER below is a function of the program's reflection - two draws binding the same
    /// textures through different fragment programs want them in different orders, and a shared
    /// entry would hand the second one the first one's albedo. The fold is 64 bits and a fold
    /// is not a proof, so every hit also verifies the EXACT key material stored beside the
    /// list ([`SetEntry::bindings`], [[vitaslop-content-hash-cache-must-verify]]).
    pub(super) snapshot_sets: FxHashMap<u64, SetEntry>,
    /// One finished [`crate::capture::BoundTexture`] per BINDING - the per-unit counterpart
    /// of [`Self::snapshot_sets`], and the one that actually carries a race frame. See the
    /// memo in `decode_texture` for why the set cache does not: the combinations of bindings
    /// a scene draws are many, the bindings themselves are few, and a set miss re-read every
    /// one of its units.
    ///
    /// Kept across scenes under the same re-proof discipline as the set cache - see
    /// [`DecodedEntry`] and [`Self::decoded_validated`]. An `Err` result is a binding the
    /// decode drops, with its drop code; it is memoised within a scene (so a loss is counted
    /// per draw without being re-derived per draw) and re-attempted each new scene, exactly
    /// as it was when this map died with the scene.
    pub(super) decoded: FxHashMap<(u32, [u32; 4]), DecodedEntry>,
    /// Which scene the run is in, monotonically. What makes the two memos above safe to KEEP
    /// across scenes: an entry proven current in scene N says nothing about scene N+1 (the
    /// guest ran in between), so each carries the scene it was last proven in, and a stale one
    /// is re-proven - one [`Self::get_or_read`] per underlying snapshot and a pointer compare -
    /// before it is believed. See [`Self::begin_scene`].
    pub(super) scene_seq: u64,
    /// >>> THE PREVIOUS DRAW'S ANSWER, KEYED ON THE BYTES IT WAS DERIVED FROM.
    ///
    /// [`Self::snapshot_sets`] already stops the LIST being rebuilt, but every draw still
    /// paid the gate in front of it: decode sixteen sampler slots out of the context
    /// snapshot, fold them into a 64-bit key, probe the map, and verify the exact key
    /// material the fold cannot prove. MEASURED in the browser at `draw: decode texture
    /// bindings (EVERY draw)` **0.54 ms of a 9.7 ms race frame over 465 draws** - 1.16 us a
    /// draw to re-derive an answer that, inside a batch of draws sharing a material, is the
    /// same one every time.
    ///
    /// Consecutive draws are exactly where that redundancy lives, so this holds ONE entry:
    /// the finished list and the raw sampler-block bytes behind it. A hit is a 384-byte
    /// `memcmp` and an `Arc` clone.
    ///
    /// It is EXACT, not a heuristic, and its two guards are what make it so:
    /// - the sampler bytes are the whole of what the bindings are (GXM copies a texture's
    ///   control words BY VALUE at bind time, [[vitaslop-texture-binding-by-value]]), and
    ///   the fragment program header is the other half of the key because the albedo
    ///   REORDER is a function of its reflection;
    /// - the entry is only believed inside the SCENE it was proven in. Across a scene the
    ///   guest has run and the pixels behind the list have to be re-proven, which is
    ///   [`Self::set_validated`]'s job; within one, `get_or_read` compares a texture at most
    ///   once per scene, so the previous draw's proof is this draw's proof by construction.
    pub(super) last_set: Option<LastSet>,
    /// The same shortcut for the VERTEX stage, keyed on [`VitaState::vertex_texture_gen`]
    /// rather than on bytes: that list is host-side with one mutator, so a counter answers
    /// "unchanged since the last draw" exactly and in one comparison.
    pub(super) last_vertex_set: Option<LastVertexSet>,
    /// Reused buffer for the dirty-page runs of one re-read - see
    /// [`TextureSnapshots::reread_dirty_runs`], which runs per texture per scene.
    pub(super) run_scratch: Vec<(usize, usize)>,
}

/// What a page-granular re-read established. See [`TextureSnapshots::reread_dirty_runs`].
pub(super) enum Reread {
    /// Every dirty page still holds the bytes the retained buffer has, so the texture is
    /// unchanged and consumers keep the buffer identity they cached against.
    Unchanged(Arc<[u8]>),
    /// The current contents: the retained buffer with the dirty pages replaced.
    Changed(Arc<[u8]>),
}

/// The previous draw's finished VERTEX-stage texture list. See
/// [`TextureSnapshots::last_vertex_set`].
pub(super) struct LastVertexSet {
    /// The binding-list generation the answer was decoded at.
    pub(super) generation: u64,
    /// The [`TextureSnapshots::snapshot_sets`] entry it came from, re-probed on every hit for
    /// the same reason [`LastSet::key`] is.
    pub(super) key: u64,
    pub(super) list: Arc<[crate::capture::BoundTexture]>,
}

/// The previous draw's finished fragment texture list. See [`TextureSnapshots::last_set`].
pub(super) struct LastSet {
    /// The raw sampler block the list was decoded from - the exact key material.
    pub(super) span: Box<[u8]>,
    /// The bound fragment program's header, which decides the albedo reorder.
    pub(super) fheader: u32,
    /// The [`TextureSnapshots::snapshot_sets`] entry this answer came from. A hit re-probes
    /// it, which is what makes this memo inherit every invalidation the set cache has rather
    /// than needing its own copy of them - see [`TextureSnapshots::set_from_previous_draw`].
    pub(super) key: u64,
    /// The answer.
    pub(super) list: Arc<[crate::capture::BoundTexture]>,
}

/// One decoded binding, kept across scenes. See [`TextureSnapshots::decoded`].
#[derive(Clone)]
pub(super) struct DecodedEntry {
    /// The decode's answer: the finished texture, or the drop code the decode reported.
    pub(super) res: Result<crate::capture::BoundTexture, u8>,
    /// The `(addr, len)` snapshot the pixels came from - the [`TextureSnapshots::get_or_read`]
    /// key a re-proof asks about. `None` for a dropped binding, which has no pixels.
    pub(super) snap: Option<(u32, usize)>,
    /// The decode fell back to level 0 (mip chain unreadable). Kept on the per-scene cadence -
    /// dropped when its scene goes stale - so the retry semantics are unchanged: the fuller
    /// chain is re-attempted each scene exactly as before the memo survived scenes.
    pub(super) degraded: bool,
    /// Scene this entry was last PROVEN current in ([`TextureSnapshots::scene_seq`]).
    pub(super) valid_scene: u64,
}

/// A whole draw's finished texture list, kept across scenes. See
/// [`TextureSnapshots::snapshot_sets`].
pub(super) struct SetEntry {
    pub(super) list: Arc<[crate::capture::BoundTexture]>,
    /// The stage tag the map key folded first: the fragment program header for the fragment
    /// stage, [`VERTEX_STAGE_TAG`] for the vertex stage. Verified on every hit.
    pub(super) tag: u64,
    /// The EXACT bindings (`unit, addr, control words`) behind the map key's 64-bit fold,
    /// verified on every hit - a fold is not a proof
    /// ([[vitaslop-content-hash-cache-must-verify]]).
    pub(super) bindings: Vec<(u32, u32, [u32; 4])>,
    /// The [`TextureSnapshots::decoded`] key AND the pixel buffer `list` holds for each
    /// non-null unit, for the per-scene re-proof: the entry survives a scene change only if
    /// every one of these re-validates to the SAME buffer. `None` marks a set that must not
    /// outlive its scene (a unit was dropped or degraded), so building it - and counting its
    /// drops - stays a per-scene event.
    pub(super) deps: Option<Vec<((u32, [u32; 4]), Arc<[u8]>)>>,
    /// How many null-handle units `list` carries. Their per-set-build drop-0 notes are
    /// re-issued when a kept set is re-proven, so the counter keeps its per-scene cadence.
    pub(super) null_drops: u32,
    /// Scene this entry was last PROVEN current in.
    pub(super) valid_scene: u64,
}

/// The stage tag [`SetEntry`] and the vertex-stage set key carry, so a vertex list and a
/// fragment list of identical bindings cannot collide in the shared map ("VERTEX").
pub(super) const VERTEX_STAGE_TAG: u64 = 0x5645_5254_5845_5300;

/// What a binding's control words say, once, so no draw has to work it out again. See
/// [`TextureSnapshots::templates`].
#[derive(Clone, Copy)]
pub(super) struct TextureTemplate {
    pub(super) base_format: u32,
    /// Guest address of a PALETTED texture's colour table, from control word 3's own field
    /// (bits 25:0, scaled by 64 - the palette is 64-byte aligned). Zero for every other format,
    /// and zero for a paletted texture whose guest never pointed it at one.
    pub(super) palette_addr: u32,
    pub(super) swizzle: u32,
    pub(super) tex_type: u32,
    pub(super) width: u32,
    pub(super) height: u32,
    pub(super) stride: u32,
    pub(super) faces: u32,
    /// Bytes one face occupies including `levels` mip levels, and the whole snapshot's length.
    pub(super) face_bytes: u32,
    pub(super) read_len: u32,
    pub(super) levels: u32,
    /// The same two if the chain read comes up short and the texture falls back to level 0 -
    /// precomputed here so the fallback costs no arithmetic either.
    pub(super) level0_bytes: u32,
    pub(super) level0_read_len: u32,
    pub(super) data_addr: u32,
    pub(super) u_addr_mode: u32,
    pub(super) v_addr_mode: u32,
    pub(super) lod_bias: u32,
    pub(super) lod_min: u32,
    pub(super) min_filter: u32,
    pub(super) mag_filter: u32,
    pub(super) gamma: u32,
    pub(super) mip_filter: u32,
}

/// Distinct control-word sets the template memo will hold. A title binds a few hundred distinct
/// textures; this is a bound on a pathological case, not a working limit. Past it the coldest
/// quarter goes, by the scene each entry was last used in, which costs re-derivation and never
/// correctness - the key IS the input.
pub(super) const TEXTURE_TEMPLATE_CAP: usize = 4096;

/// Distinct vertex streams the content index will hold - see
/// [`TextureSnapshots::vertex_by_content`]. A frame binds under a thousand; this bounds a title
/// that streams genuinely new geometry every frame, where the index cannot help and must not be
/// allowed to grow. Past it the least recently recognised go: the key is derived from the bytes
/// and every hit is proven by comparison, so shedding one costs a re-read and never
/// correctness.
pub(super) const VERTEX_CONTENT_INDEX_CAP: usize = 8192;

/// What a multi-stream gather READ: the per-vertex streams' spans (address, bytes) in stream
/// order, the packed stride, and - for a compacted draw - the identity of its gather list.
/// Two draws with equal keys interleave the same guest bytes into the same rows.
#[derive(Clone, Default, PartialEq, Eq, Hash)]
pub(super) struct GatherKey {
    /// The LAYOUT this row is scattered under - see [`RowLayout::key`]. A content fingerprint,
    /// not an address: two programs binding the same streams at the same stride can place their
    /// attributes differently, and a memo keyed on the spans alone would hand one program the
    /// other's rows.
    pub(super) layout: u64,
    /// Per stream: the guest span read, the column it lands in and that column's width. The
    /// LAYOUT is part of the key: two programs binding the same streams at the same packed
    /// stride can interleave them into different columns, and a memo keyed on the spans
    /// alone handed one program the other's rows (a golf title's picture moved on 7 of 12
    /// shots before this was in the key).
    pub(super) spans: Vec<(u32, usize, u32, u32)>,
    pub(super) stride: u32,
    /// A compacted draw's gather list, by identity - see `GatherMemo::gather`, which pins it.
    pub(super) gather: (usize, usize),
}

/// A memoised gather: the interleaved buffer and the dirty-map stamp each span was proven
/// current at. Filed only for geometry the content intern has ALREADY seen twice - see
/// `read_draw_vertices` for why that gate is the whole cost argument.
pub(super) struct GatherMemo {
    pub(super) result: Arc<[u8]>,
    pub(super) stamps: Vec<(u32, usize, u8)>,
    pub(super) used: u64,
    /// The gather list the key's identity names, held so its allocation cannot be handed to
    /// another list while this entry can still match it.
    ///
    /// Never READ, and that is the point: the field exists to own an allocation for as long as
    /// this entry can match it, so that an address cannot be reused under a key that still
    /// names it [[vitaslop-an-address-is-not-an-identity]]. Allowed rather than deleted,
    /// because deleting it is the bug.
    #[allow(dead_code)]
    pub(super) gather: Option<Arc<[u32]>>,
}

/// What an INSTANCED expansion read: the wrapped index words, the wrap and stride, and every
/// bound stream's span. Two draws with equal keys expand to the same rows.
/// See [`TextureSnapshots::instanced_memo`].
#[derive(Clone, Default, PartialEq, Eq, Hash)]
pub(super) struct InstancedKey {
    pub(super) index: (u32, usize, usize),
    pub(super) wrap: u32,
    pub(super) stride: u32,
    /// The row layout, by content fingerprint - see [`GatherKey::layout`].
    pub(super) layout: u64,
    /// Per stream, in stream order: the bound address, its stride, its column and whether it
    /// steps per instance. The ADDRESSES are the spans; the rest is the layout, which is part
    /// of the key for the same reason it is part of [`GatherKey`]'s.
    pub(super) streams: Vec<(u32, u32, u32, bool)>,
}

/// A memoised instanced expansion, with the dirty-map stamp each span was proven current at.
pub(super) struct InstancedMemo {
    pub(super) indices: Arc<[u8]>,
    pub(super) rows: u32,
    pub(super) vertices: Arc<[u8]>,
    pub(super) stamps: Vec<(u32, usize, u8)>,
    pub(super) used: u64,
}

/// A draw's indices as the renderer consumes them, and how its vertex rows are found.
///
/// The index bytes have been rewritten so that index `i` names row `i` of the draw's vertex
/// buffer. For a draw taken as a contiguous window that buffer is guest rows
/// `first_vertex .. first_vertex + vertex_count`; for a COMPACTED draw (`gather` is `Some`)
/// row `i` is guest row `gather[i]`, and `gather` is sorted ascending so the rows are read as
/// coalesced runs. See `TextureSnapshots::get_or_read_indices`.
#[derive(Clone)]
pub struct DrawIndices {
    pub bytes: Arc<[u8]>,
    pub first_vertex: u32,
    pub vertex_count: u32,
    pub gather: Option<Arc<[u32]>>,
}

impl DrawIndices {
    /// What the index cache pays to hold this entry.
    pub(super) fn held_bytes(&self) -> usize {
        self.bytes.len() + self.gather.as_ref().map_or(0, |g| g.len() * 4)
    }
}

/// A draw is compacted (see `DrawIndices::gather`) when its index range spans at least this
/// many rows and more than `COMPACT_SPARSITY` times its index count. Below the range floor the
/// window read is cheap whatever its shape; below the sparsity the window is mostly used.
pub(super) const COMPACT_MIN_RANGE_ROWS: usize = 1024;
pub(super) const COMPACT_SPARSITY: usize = 4;

/// When gathering compacted rows, two referenced rows fewer than this many BYTES apart are
/// read in one guest read: on the browser a read is a call across the wasm boundary, and one
/// costs about what copying a few kilobytes does.
pub(super) const GATHER_RUN_GAP_BYTES: usize = 4096;

/// `VITASLOP_COMPACT_SPARSE=0` is the NEGATIVE CONTROL for the compaction: the `min..=max`
/// window for every draw, which is what this path did before. One build, two arms.
pub(super) fn compact_sparse_draws() -> bool {
    use std::sync::OnceLock;
    static ON: OnceLock<bool> = OnceLock::new();
    *ON.get_or_init(|| crate::knobs::var("VITASLOP_COMPACT_SPARSE").ok().as_deref() != Some("0"))
}

/// A cheap, allocation-free fingerprint of a vertex stream, for [`TextureSnapshots::
/// vertex_by_content`]'s lookup ONLY.
///
/// # It reads a sample, not the stream, and that is the whole point
/// The thing it replaces read every byte: the renderer hashed each stream in full to key its
/// packed-vertex cache, 7.03 MB a frame on a golf title's gameplay frame, and then compared the
/// hit in full as well. Two passes over the same megabytes to answer one question.
///
/// A fingerprint may be as weak as it likes here because it decides only WHICH candidate to
/// compare - the comparison is what decides. So this reads at most 64 evenly spaced words plus
/// both ends, which is O(1) in the stream's size, and a collision costs one failed comparison.
///
/// The ends are included deliberately: a vertex stream's first and last vertices are the two
/// most likely to differ between two meshes that share a middle (an index-rebased window of one
/// larger buffer), and a purely strided sample can step straight over both.
/// Whether a built vertex stream is given the identity of the byte-identical buffer already
/// held - see [`TextureSnapshots::intern_vertices`].
///
/// >>> IT WAS MEASURED AS A LOSS, AND THAT VERDICT WAS AN ARTEFACT OF TWO BROKEN KEYS.
///
/// The first reading (`render 17.1/17.5` ON against `13.3/11.2` OFF) was taken while the
/// renderer's texture cache was keyed on an allocator ADDRESS: interning moves allocations, so
/// the whole apparent cost was that cache reacting to addresses it should never have depended
/// on. With that key exact and the paletted expansion no longer minting a fresh identity per
/// decode, the SAME A/B on the SAME screen reverses - one build, four alternated arms, golf
/// gameplay, `NO_BC=1`:
///
/// | arm | render | prepare | vertices hashed |
/// |---|---|---|---|
/// | intern ON | **7.7 / 7.0 ms** | **5.0 / 4.6 ms** | **1.02 MB** |
/// | intern OFF | 10.3 / 9.7 ms | 7.7 / 7.3 ms | 7.03 MB |
///
/// The frame is BIT-IDENTICAL in every arm. `VITASLOP_VERTEX_INTERN=0` is the arm back.
/// Read once: per-draw path.
/// `VITASLOP_VERTEX_INTERN_USE=0` restores the OLD behaviour: a content-index hit taken on the
/// single-stream snapshot path does NOT count as a use, so the entry that answered it is a
/// candidate for the very next eviction. An A/B ARM, so it is VALUE-sensitive rather than
/// presence-only [[vitaslop-knob-is-the-gate-not-the-level]]. See the hit site for what the
/// missing stamp cost.
pub(super) fn vertex_intern_use_stamps() -> bool {
    use std::sync::OnceLock;
    static ON: OnceLock<bool> = OnceLock::new();
    *ON.get_or_init(|| {
        crate::knobs::var("VITASLOP_VERTEX_INTERN_USE")
            .ok()
            .as_deref()
            != Some("0")
    })
}

pub(super) fn vertex_intern_enabled() -> bool {
    use std::sync::OnceLock;
    static ON: OnceLock<bool> = OnceLock::new();
    *ON.get_or_init(|| crate::knobs::var("VITASLOP_VERTEX_INTERN").ok().as_deref() != Some("0"))
}

/// Whether a draw decodes only the texture units its fragment program DECLARES - see
/// [`ProgramReflection::sampler_units`].
///
/// MEASURED and now the DEFAULT, one build, two arms, golf's menu in a headless browser on the
/// real GPU: texture resolutions **2,063 -> 1,293 a frame** over the same 258 draws, `cpu 6.1 ->
/// 5.7 ms`. The count is the number that matters - it is what a slower device pays more for -
/// and the frame is BIT-IDENTICAL, which is what says the narrowed units were never sampled.
/// `VITASLOP_SAMPLER_NARROW=0` is the arm back to every bound slot.
/// Whether the interleaved vertex row 4-ALIGNS each stream's column and its own stride - see
/// the call site in `set_vertex_program`. `VITASLOP_GXP_ALIGN_ROW=0` is the tight row it
/// replaces.
/// `VITASLOP_INSTANCED_MEMO=0` - the negative control for the instanced expansion memo, which
/// re-reads and re-expands every frame the way it did before. See
/// [`TextureSnapshots::instanced_memo`].
pub(super) fn instanced_memo_enabled() -> bool {
    use std::sync::OnceLock;
    static ON: OnceLock<bool> = OnceLock::new();
    *ON.get_or_init(|| crate::knobs::var("VITASLOP_INSTANCED_MEMO").ok().as_deref() != Some("0"))
}
/// `VITASLOP_GXP_ATTR_ROW=0` - the negative control for laying the interleaved row out per
/// ATTRIBUTE rather than per stream column. See [`ScatterPiece`].
/// `VITASLOP_DEFER_WINDOW_BYTES=1` - the arm BACK to skipping the draw-time snapshot of a
/// guest-memory window and letting the end-of-scene resolve supply its bytes outright.
///
/// >>> IT IS NO LONGER THE DEFAULT, BECAUSE THE END-OF-SCENE BYTES ARE NOT ALWAYS THE DRAW'S.
///
/// A window is re-read at the guest's GPU wait so a buffer the title fills AFTER the draw that
/// names it is read once it holds something (the baseball crowd's skinning matrices - see the
/// resolve site). A scene that names the SAME buffer from many draws and rewrites it between
/// them is the other half of that, and the blanket re-read hands every one of those draws the
/// LAST version.
///
/// MEASURED on the golf title's tree-impostor bake: 202 offscreen scenes in one frame, every
/// one of them a handful of draws through the same 136-byte vertex window at `0x882c4200`.
/// Captured by the resolve, every draw read `(1.0, 0.0, 1.0)` as its ambient colour - magenta -
/// while the guest's own bytes at the frame's first window-bearing draw were
/// `(1.0, 0.505, 0.835)`. The ambient's GREEN lane was exactly zero on every impostor, which
/// put a magenta wash over the whole bake; the grass and reed billboards, which have no strong
/// diffuse term to hide behind, came out solid magenta against an olive-green source texture.
///
/// So the draw-time snapshot is taken again, and [`crate::rtt_writeback::nothing_written_here`]
/// decides which of the two a given window wants - see the resolve sites.
/// >>> THE HOST'S REAL-TIME CLOCK, ONE ORIGIN FOR EVERY WORKER, in microseconds.
///
/// The guest's clocks are virtual and `World::wall_us` may be recorded or replayed, so a host
/// that has a genuine monotonic clock shared by all its threads installs it here (the browser:
/// `timeOrigin + performance.now()`, which agrees across workers). Consumers that must follow
/// real time - the audio port's play-out (`vita::audio::out_output`) - read it; with none
/// installed they keep the virtual clock.
pub(super) static HOST_WALL_MS: std::sync::OnceLock<fn() -> f64> = std::sync::OnceLock::new();

/// Install the host's real-time clock (milliseconds, one origin across threads).
pub fn set_host_wall_clock(f: fn() -> f64) {
    let _ = HOST_WALL_MS.set(f);
}

/// The earliest wall-park deadline (0 = none), readable without the host lock so an idle worker
/// can time its wait to it - see [`VitaState::wall_park`].
pub(super) static NEXT_WALL_PARK_US: std::sync::atomic::AtomicU64 = std::sync::atomic::AtomicU64::new(0);
pub(super) static WALL_PARKS_SERVED: std::sync::atomic::AtomicBool = std::sync::atomic::AtomicBool::new(false);

pub(super) fn note_next_wall_park(waiters: &[(i32, u64)]) {
    let next = waiters.iter().map(|w| w.1).min().unwrap_or(0);
    NEXT_WALL_PARK_US.store(next, std::sync::atomic::Ordering::Release);
}

/// The earliest wall-park deadline, if any thread is wall-parked.
pub fn next_wall_park_us() -> Option<u64> {
    match NEXT_WALL_PARK_US.load(std::sync::atomic::Ordering::Acquire) {
        0 => None,
        t => Some(t),
    }
}

/// Declare that this engine expires wall parks (drains call `expire_wall_parks`, idle workers
/// wake for them). Without it nothing would ever wake a wall-parked thread.
pub fn set_wall_parks_served(on: bool) {
    WALL_PARKS_SERVED.store(on, std::sync::atomic::Ordering::Release);
}

/// Whether wall parks are served - see [`set_wall_parks_served`].
pub fn wall_parks_served() -> bool {
    WALL_PARKS_SERVED.load(std::sync::atomic::Ordering::Acquire)
}

/// The host's real-time clock in microseconds, when one is installed.
pub fn host_wall_us() -> Option<u64> {
    HOST_WALL_MS.get().map(|f| (f() * 1000.0) as u64)
}

/// `VITASLOP_WINDOW_WAIT=1`: the draw-time window capture waits for a busy snapshot lock
/// instead of reading the window uncached - see `capture_mem_windows`.
pub(super) fn window_wait() -> bool {
    static ON: std::sync::OnceLock<bool> = std::sync::OnceLock::new();
    *ON.get_or_init(|| crate::knobs::flag("VITASLOP_WINDOW_WAIT"))
}

/// The wall-clock floor's per-tick clamp - see [`VitaState::wall_floor_tick`].
pub(super) fn wall_step_ms() -> f64 {
    static V: std::sync::OnceLock<f64> = std::sync::OnceLock::new();
    *V.get_or_init(|| {
        crate::knobs::var("VITASLOP_CLOCK_WALL_STEP_MS")
            .ok()
            .and_then(|v| v.trim().parse().ok())
            .unwrap_or(12.0)
    })
}

/// `VITASLOP_IO_WALL_FLOOR` (default on): the wall-clock floor that advances the game clock
/// advances the storage clock too - see `VitaState::wall_floor_tick`.
pub(super) fn io_wall_floor() -> bool {
    static ON: std::sync::OnceLock<bool> = std::sync::OnceLock::new();
    *ON.get_or_init(|| crate::knobs::var("VITASLOP_IO_WALL_FLOOR").ok().as_deref() != Some("0"))
}

pub(super) fn defer_window_bytes() -> bool {
    use std::sync::OnceLock;
    static ON: OnceLock<bool> = OnceLock::new();
    *ON.get_or_init(|| {
        crate::knobs::var("VITASLOP_DEFER_WINDOW_BYTES")
            .ok()
            .as_deref()
            == Some("1")
    })
}

/// Which windows the guest's GPU wait re-reads - see [`window_wants_reread`].
///
/// `blank` (the DEFAULT) is the standing rule: a window that is ONE REPEATED WORD over its whole
/// length. `slot` additionally takes one with any unwritten 16-byte row, and `all` re-reads
/// every window. `all` is expected to be WRONG on a title that rewrites one buffer between draws
/// of a scene - see [`defer_window_bytes`] for the golf bake that established that.
///
/// >>> `slot` IS REFUTED AS A FIX FOR A MESH THAT RENDERS ONLY PART OF ITSELF, and the census
/// >>> below is how. It was written for a football title, whose players render from the waist up: the
/// reasoning was that a bone palette partly filled at the draw keeps zero rows, and a vertex
/// weighted to a zero matrix collapses. The rule fires there in quantity - 631,134 of 3,290,724
/// windows tested in one run are partly unwritten - and the frame is **bit-identical** to the
/// `blank` arm at the same native frame. So the re-read returns the bytes the draw already had:
/// those rows are still zero at the guest's own GPU wait, and nothing fills them late.
///
/// The same census carries the fact that sent this the wrong way: **`blank` fires ZERO times on
/// this title**. The late-fill mechanism is entirely inert for that title, so whatever empties its
/// legs is not this path, and a reader should not spend a second run here.
/// [[vitaslop-a-refuted-diagnosis-left-in-place-reads-as-a-finding]]
#[derive(Clone, Copy, PartialEq)]
pub(super) enum WindowReread {
    Blank,
    Slot,
    All,
}

pub(super) fn window_reread_rule() -> WindowReread {
    use std::sync::OnceLock;
    static R: OnceLock<WindowReread> = OnceLock::new();
    *R.get_or_init(
        || match crate::knobs::var("VITASLOP_WINDOW_REREAD").ok().as_deref() {
            Some("slot") => WindowReread::Slot,
            Some("all") => WindowReread::All,
            _ => WindowReread::Blank,
        },
    )
}

/// Whether this window's draw-time bytes should be replaced by a fresh read at the guest's own
/// GPU wait.
///
/// The draw-time snapshot is the DRAW'S OWN bytes and is right whenever the guest had finished
/// filling the buffer before it issued the draw. It is wrong when a job thread is still writing,
/// and the two are told apart by what the snapshot CONTAINS: a slot nothing has written into
/// cannot be what the draw meant to read, because a shader that indexes it reads zero.
/// One window a raw target this frame rendered covers - see
/// [`TextureSnapshots::get_or_read_covered_window`].
pub(super) struct CoveredWindow {
    /// The covered byte ranges, relative to the window, sorted and disjoint. The bytes there are
    /// ZERO here: the renderer copies the target's texels over them.
    pub(super) cover: Vec<(u32, u32)>,
    pub(super) bytes: Arc<[u8]>,
    /// The uncovered bytes are one repeated word - nothing written there yet - so the resolve
    /// reads them again, as it reads any blank window again ([`window_wants_reread`]).
    pub(super) blank: bool,
}

pub(super) fn window_wants_reread(bytes: &[u8]) -> bool {
    if bytes.is_empty() {
        return false;
    }
    let whole = crate::rtt_writeback::nothing_written_here(bytes);
    // The per-16-byte-slot scan reads the WHOLE window, so it runs only where its answer is
    // used: the `slot` rule, or `VITASLOP_WINDOW_CENSUS=1` asking for the census count. Under
    // the default `blank` rule it was a full scan of every non-blank window every frame for a
    // counter nobody read.
    let rule = window_reread_rule();
    let slot = !whole
        && (matches!(rule, WindowReread::Slot) || window_census_exact())
        && crate::rtt_writeback::any_slot_unwritten(bytes);
    // >>> THE SEAM, COUNTED - which arm each window falls in, so "the rule does not fire" and
    // "the rule fires and does not help" are different answers instead of one guess.
    WINDOWS_SEEN.fetch_add(1, std::sync::atomic::Ordering::Relaxed);
    if whole {
        WINDOWS_WHOLE_BLANK.fetch_add(1, std::sync::atomic::Ordering::Relaxed);
    } else if slot {
        WINDOWS_SLOT_BLANK.fetch_add(1, std::sync::atomic::Ordering::Relaxed);
    }
    match rule {
        WindowReread::Blank => whole,
        WindowReread::Slot => whole || slot,
        WindowReread::All => true,
    }
}

/// A deferred window recognised as the placeholder (blank by construction) and re-read
/// without a scan - counted with the census so its totals still add up.
pub(super) fn note_placeholder_window() {
    use std::sync::atomic::Ordering::Relaxed;
    WINDOWS_SEEN.fetch_add(1, Relaxed);
    WINDOWS_WHOLE_BLANK.fetch_add(1, Relaxed);
}

/// `VITASLOP_WINDOW_CENSUS=1`: measure the census's `slot-blank` count exactly even under the
/// default `blank` rule, which otherwise skips the per-slot scan it needs.
pub(super) fn window_census_exact() -> bool {
    use std::sync::OnceLock;
    static ON: OnceLock<bool> = OnceLock::new();
    *ON.get_or_init(|| crate::knobs::flag("VITASLOP_WINDOW_CENSUS"))
}

pub(super) static WINDOWS_SEEN: std::sync::atomic::AtomicU64 = std::sync::atomic::AtomicU64::new(0);
pub(super) static WINDOWS_WHOLE_BLANK: std::sync::atomic::AtomicU64 = std::sync::atomic::AtomicU64::new(0);
pub(super) static WINDOWS_SLOT_BLANK: std::sync::atomic::AtomicU64 = std::sync::atomic::AtomicU64::new(0);

/// The window re-read census, printed once per run at the frame the caller names.
///
/// `slot-blank` is the population this rule exists for: a window holding real values in some
/// 16-byte rows and nothing in others. At ZERO it does not fire and a picture that did not
/// change is telling you so rather than refuting the rule.
pub fn report_window_reread_census(frame: u64) {
    use std::sync::atomic::Ordering::Relaxed;
    let (seen, whole, slot) = (
        WINDOWS_SEEN.load(Relaxed),
        WINDOWS_WHOLE_BLANK.load(Relaxed),
        WINDOWS_SLOT_BLANK.load(Relaxed),
    );
    tracing::warn!(
        target: "vitaslop::gxm",
        frame,
        seen,
        whole_blank = whole,
        slot_blank = slot,
        rule = match window_reread_rule() {
            WindowReread::Blank => "blank",
            WindowReread::Slot => "slot",
            WindowReread::All => "all",
        },
        "gxm window re-read census: {seen} window(s) tested at the GPU wait - {whole} entirely \
         unwritten, {slot} PARTLY unwritten (some 16-byte row blank, others holding values), \
         the rest fully written and keeping their draw-time bytes"
    );
}

pub(super) fn attr_packed_row() -> bool {
    use std::sync::OnceLock;
    static ON: OnceLock<bool> = OnceLock::new();
    *ON.get_or_init(|| crate::knobs::var("VITASLOP_GXP_ATTR_ROW").ok().as_deref() != Some("0"))
}

pub(super) fn align_packed_row() -> bool {
    use std::sync::OnceLock;
    static ON: OnceLock<bool> = OnceLock::new();
    *ON.get_or_init(|| crate::knobs::var("VITASLOP_GXP_ALIGN_ROW").ok().as_deref() != Some("0"))
}

pub(super) fn sampler_narrow_enabled() -> bool {
    use std::sync::OnceLock;
    static ON: OnceLock<bool> = OnceLock::new();
    *ON.get_or_init(|| crate::knobs::var("VITASLOP_SAMPLER_NARROW").ok().as_deref() != Some("0"))
}

/// >>> BYTE-SLICE EQUALITY THAT DOES NOT GO THROUGH `memcmp`.
///
/// `a == b` on `[u8]` lowers to a `memcmp` LIBCALL, and on `wasm32-unknown-unknown` that is
/// compiler-builtins' implementation: a byte-at-a-time loop. `-C target-feature=+simd128` does
/// not change it, because a call into a prebuilt library function is not code LLVM can
/// vectorise - the flag only helps code it can SEE
/// [[vitaslop-wasm-simd-must-be-asked-for]].
///
/// That matters here because this comparison runs over the whole interleaved vertex buffer of
/// every multi-stream draw. MEASURED before this existed: the content intern was **2.63 ms of a
/// 8.20 ms `DRAW TOTAL`** - 32% of all draw work - at an **86.5% HIT RATE**, and a hit is
/// exactly this comparison. 6.4 MB a frame, one byte per iteration.
///
/// Eight bytes at a time instead, with a scalar tail. Identical answer, and the loop is now
/// something the optimiser can widen further on its own.
/// # >>> AND IT IS wasm-ONLY, BECAUSE NATIVE `memcmp` IS ALREADY BETTER THAN THIS
/// On a native target `a == b` calls the platform's `memcmp`, which is hand-written SIMD and
/// beats an eight-byte Rust loop comfortably. Using this everywhere would have made the desktop
/// build - the pixel oracle, the headless replay, every test - measurably SLOWER to buy a win
/// that only exists where the libcall is a byte loop. The `#[cfg]` is the whole point of the
/// function, not a detail of it.
#[inline]
pub(super) fn bytes_eq(a: &[u8], b: &[u8]) -> bool {
    #[cfg(not(target_arch = "wasm32"))]
    {
        a == b
    }
    #[cfg(target_arch = "wasm32")]
    {
        bytes_eq_wide(a, b)
    }
}

/// The eight-at-a-time comparison itself, compiled on every target so the test below actually
/// exercises it rather than testing `==` against itself on the host.
#[inline]
#[allow(dead_code)]
pub(super) fn bytes_eq_wide(a: &[u8], b: &[u8]) -> bool {
    if a.len() != b.len() {
        return false;
    }
    let n = a.len();
    let mut i = 0;
    while i + 8 <= n {
        // `try_into` on a fixed-size window compiles to a plain unaligned load; the
        // `unwrap` cannot fire because the slice is exactly 8 long.
        let x = u64::from_ne_bytes(a[i..i + 8].try_into().unwrap());
        let y = u64::from_ne_bytes(b[i..i + 8].try_into().unwrap());
        if x != y {
            return false;
        }
        i += 8;
    }
    while i < n {
        if a[i] != b[i] {
            return false;
        }
        i += 1;
    }
    true
}

#[cfg(test)]
mod bytes_eq_tests {
    // The WIDE one by name: `bytes_eq` delegates to `==` off wasm, so testing it on the host
    // would compare `==` with itself and pass no matter what the eight-byte loop did.
    use super::bytes_eq_wide as bytes_eq;

    /// The whole point is that it answers exactly what `==` answers, including at every
    /// boundary of the 8-byte stride and in the scalar tail.
    #[test]
    fn it_agrees_with_slice_equality_at_every_length_and_difference() {
        for len in 0..40usize {
            let base: Vec<u8> = (0..len).map(|i| (i * 7 + 1) as u8).collect();
            assert!(bytes_eq(&base, &base.clone()), "len {len}: equal slices");
            if len > 0 {
                // At len 0 the "shorter" slice is the same empty slice, so there is nothing to
                // differ - the case only exists from one byte up.
                assert!(
                    !bytes_eq(&base, &base[..len - 1]),
                    "len {len}: length differs"
                );
            }
            for at in 0..len {
                let mut other = base.clone();
                other[at] ^= 0xff;
                assert!(
                    !bytes_eq(&base, &other),
                    "len {len}: a difference at byte {at} was missed - a comparison that reads \
                     eight bytes at a time must still find a single changed byte anywhere"
                );
            }
        }
    }
}

pub(super) fn vertex_fingerprint(bytes: &[u8]) -> u64 {
    const SAMPLES: usize = 64;
    let mut h = 0xcbf2_9ce4_8422_2325u64;
    let mut mix = |v: u64| {
        h ^= v;
        h = h.wrapping_mul(0x0000_0100_0000_01b3);
    };
    mix(bytes.len() as u64);
    let word = |at: usize| -> u64 {
        let mut w = [0u8; 8];
        let n = 8.min(bytes.len().saturating_sub(at));
        w[..n].copy_from_slice(&bytes[at..at + n]);
        u64::from_le_bytes(w)
    };
    if bytes.len() <= SAMPLES * 8 {
        // Small enough that the "sample" would be the whole thing anyway - read it all, so a
        // short stream's fingerprint is exact and its candidate list is never crowded.
        for at in (0..bytes.len()).step_by(8) {
            mix(word(at));
        }
        return h;
    }
    let last = bytes.len() - 8;
    let step = last / (SAMPLES - 1);
    for i in 0..SAMPLES {
        mix(word((i * step).min(last)));
    }
    h
}

/// Is the snapshot check allowed to run once per FRAME instead of once per SCENE? The one
/// inexact cadence, and the only one that has to be asked for - see [`texture_check`].
pub(super) fn texture_check_per_frame() -> bool {
    texture_check() == TextureCheck::Frame
}

/// `VITASLOP_TEX_MEMO_PER_SCENE`: clear the per-binding and per-set texture memos at every
/// scene start (the pre-cross-scene cadence) instead of keeping them and re-proving each
/// entry on first use per scene. The A/B arm for the cross-scene memo; BOTH arms are exact -
/// the difference is only what gets rederived, never what is answered.
pub(super) fn tex_memo_per_scene() -> bool {
    static ON: std::sync::OnceLock<bool> = std::sync::OnceLock::new();
    *ON.get_or_init(|| crate::knobs::flag("VITASLOP_TEX_MEMO_PER_SCENE"))
}

/// Say that a set-cache hit's stored bindings did not match the probe's. Treated as a miss, so
/// it costs a rebuild and never an answer.
///
/// # >>> THIS IS PROBABLY NOT A HASH COLLISION, AND THE REPORT SAID IT WAS
/// It fired on a 78-second phone run, which for a genuine 64-bit collision over a few thousand
/// binding sets is not credible. Look at what the two sides actually use:
///
///   * the FOLD (see the `set_key` construction at the probe) mixes the fragment header, and
///     each binding's `unit` and four control `words`;
///   * the VERIFY additionally requires `addr` - the guest `SceGxmTexture*` the binding came
///     from - to match.
///
/// So two bindings with identical control words reached through DIFFERENT descriptors fold to
/// the same key and fail the verify. That is not a collision, it is a key that omits a field its
/// own equality test depends on, and the cost is a forced rebuild every time a title binds one
/// texture through two descriptors - which is an ordinary thing for a title to do.
///
/// # Why this is a REPORT and not yet a fix
/// The fix is one line - mix `addr` into the fold, which only ever makes keys MORE distinct and
/// so can never serve a wrong list. What is not known is whether it is worth it: `addr` may be a
/// stack descriptor at a fresh address every frame, and then keying on it turns a cheap forced
/// miss into an entry per frame in `snapshot_sets`. That is a memory question, and the run that
/// raised this could not answer it because the report FIRED ONCE - "vanishingly rare" and "every
/// draw of every frame" printed exactly the same line. So it now reports on a DOUBLING cadence:
/// the last line in a dump carries the order of magnitude, at a cost that stays logarithmic.
/// [[vitaslop-a-count-needs-its-window]] [[vitaslop-an-address-is-not-an-identity]]
pub(super) fn report_set_key_collision() {
    static N: std::sync::atomic::AtomicU64 = std::sync::atomic::AtomicU64::new(0);
    let n = N.fetch_add(1, std::sync::atomic::Ordering::Relaxed) + 1;
    // 1, 2, 4, 8, ... - every line is the last one anybody needs to read.
    if !n.is_power_of_two() {
        return;
    }
    tracing::warn!(
        target: "vitaslop::gxm",
        "texture set-cache key collision ({n} so far this run): a set-cache hit's stored \
         bindings did not match the probe's. Handled exactly (treated as a miss), so the \
         picture is unaffected and the cost is a rebuild. NOTE a large count here is almost \
         certainly NOT a 64-bit hash collision: the key fold omits each binding's descriptor \
         address while the equality test requires it, so one texture bound through two \
         descriptors lands here every time. See `report_set_key_collision`."
    );
}

/// How often, and by what means, a retained texture snapshot is re-validated.
#[derive(Clone, Copy, PartialEq, Eq)]
pub(super) enum TextureCheck {
    /// The default and the exact one: once per scene, by the guest-store stamps where
    /// the engine has them and by a byte compare where it does not.
    Scene,
    /// Once per FRAME for a texture never yet seen to change. Faster, NOT exact.
    Frame,
    /// Once per scene, by a byte compare ALWAYS - the stamps are ignored even where
    /// the engine has them.
    ///
    /// Exact, and slower than `scene` on every engine, so it is not a performance
    /// option: it exists to falsify the stamps. Two runs of one input script that differ
    /// only in this must be BIT-IDENTICAL, and if they are not, the dirty map missed a
    /// store - which is a silent stale-texture bug and the only failure mode the
    /// mechanism has.
    Bytes,
}

/// How a retained texture snapshot is re-validated (`VITASLOP_TEXTURE_CHECK`): `scene`
/// (default, exact), `frame` (faster, not exact) or `bytes` (exact, slower, ignores the
/// guest-store stamps - the A/B that falsifies them). A value that is none of those is an
/// ERROR rather than a silent fallback: a run configured by a typo would otherwise quietly
/// publish frames at the accuracy nobody asked for.
pub(super) fn texture_check() -> TextureCheck {
    static ON: std::sync::OnceLock<TextureCheck> = std::sync::OnceLock::new();
    *ON.get_or_init(|| match crate::knobs::var("VITASLOP_TEXTURE_CHECK") {
        Err(_) => TextureCheck::Scene,
        Ok(v) => match v.trim() {
            "scene" => TextureCheck::Scene,
            "frame" => TextureCheck::Frame,
            "bytes" => TextureCheck::Bytes,
            other => panic!(
                "VITASLOP_TEXTURE_CHECK={other:?} is not `scene` (exact, the default), \
                 `frame` (faster, one scene of staleness the first time a texture changes) \
                 or `bytes` (exact, slower, ignores the guest-store stamps - the A/B that \
                 falsifies them)"
            ),
        },
    })
}

/// Say - once - that the per-frame texture check is in effect. It trades exactness for
/// speed, and an approximation nobody can see in the log is one that gets quoted as a
/// measurement of the real thing. [[vitaslop-fallback-must-report]]
pub(super) fn report_texture_check_per_frame() {
    static SAID: std::sync::Once = std::sync::Once::new();
    SAID.call_once(|| {
        tracing::warn!(
            target: "vitaslop::gxm",
            "VITASLOP_TEXTURE_CHECK=frame: a texture never yet seen to change is compared \
             against guest memory ONCE PER FRAME, not once per scene. Faster, and NOT exact - \
             a texture the guest rewrites mid-frame for the first time renders stale for the \
             rest of that frame. Unset this for the exact per-scene check."
        );
    });
}

/// Byte budget for retained VERTEX snapshots, separate from the texture one so a level's
/// geometry and its textures cannot evict each other.
///
/// A race frame of this project's heaviest title reads about 6 MB of vertices, and a level's
/// whole static geometry is a small multiple of that, so 64 MB holds it comfortably while staying
/// a bound rather than a licence to grow with the run.
///
/// Deliberately a CONSTANT and not a knob. A knob here would be a way of not deciding, and the
/// engine that pays for this runs whatever is defaulted ([[vitaslop-pick-the-default-dont-add-a-knob]]).
/// If it ever needs changing, the number to change it against is the `draw: read+interleave
/// vertices` figure in the device panel's bytes line.
pub(super) fn vertex_snapshot_budget() -> usize {
    // Scaled to the DEVICE - see [`crate::knobs::memory_scale`]. A desktop, a headless run and
    // anything that never reported its memory get exactly this number.
    crate::knobs::scale_budget(64 << 20)
}

/// What the snapshot caches are holding and what their budgets have cost, as process-wide
/// counters so the panel can carry them without threading a borrow of the emulator through the
/// browser's diagnostics path.
///
/// These exist because the question they answer - "is this session slowing down because a cache
/// keeps being emptied" - previously took four device dumps and a 48,000-frame play session to
/// ask ONCE, and even then could only be answered by inference. Two atomic stores a frame.
/// Upper bound on [`TextureSnapshots::snapshot_sets`], in binding sets. At module scope so the
/// diagnostics can print the cache's occupancy AGAINST it - a live count with no cap beside it
/// says nothing about whether the bound is being hit.
pub(super) const SET_CAP: usize = 16_384;

pub(super) mod snapcounts {
    use std::sync::atomic::{AtomicU64, Ordering::Relaxed};
    pub static EVICT_PASSES: AtomicU64 = AtomicU64::new(0);
    pub static EVICTED: AtomicU64 = AtomicU64::new(0);
    pub static FREED: AtomicU64 = AtomicU64::new(0);
    pub static HELD_BYTES: AtomicU64 = AtomicU64::new(0);
    pub static ENTRIES: AtomicU64 = AtomicU64::new(0);
    pub static VERTEX_BYTES: AtomicU64 = AtomicU64::new(0);
    /// Bytes the vertex CONTENT index (`vertex_by_content`) holds, and its entry count.
    pub static VERTEX_CONTENT_BYTES: AtomicU64 = AtomicU64::new(0);
    pub static VERTEX_CONTENT_ENTRIES: AtomicU64 = AtomicU64::new(0);
    pub static INDEX_BYTES: AtomicU64 = AtomicU64::new(0);
    pub static VERTEX_EVICTED: AtomicU64 = AtomicU64::new(0);
    /// Entries and bytes shed from the CONTENT index, which had no counter at all.
    ///
    /// >>> AN EVICTION HERE IS INVISIBLE EVERYWHERE ELSE, AND IT IS EXPENSIVE. The address map
    /// reports its own evictions and this map's were reported as part of that number, i.e. not
    /// at all: a run could shed the content index every frame and the panel would read `0
    /// vertex`. What it costs is not a re-read - it is a mesh LOSING ITS IDENTITY, so the
    /// renderer's allocation-keyed packed cache misses and hashes and compares the whole
    /// stream again for geometry that never changed.
    pub static VERTEX_INTERN_EVICTED: AtomicU64 = AtomicU64::new(0);
    pub static VERTEX_INTERN_EVICTED_BYTES: AtomicU64 = AtomicU64::new(0);
    pub static INDEX_EVICTED: AtomicU64 = AtomicU64::new(0);
    /// >>> LIVE ENTRIES IN THE BINDING-SET CACHE, AND WHAT ITS CAP HAS SHED.
    ///
    /// `snapshot_sets` is bounded at `SET_CAP` and nothing reported either number, so the cache
    /// could sit one entry under its cap - re-deriving a quarter of itself every time it touched
    /// it - and look exactly like a cache with room to spare. It is also the number that decides
    /// whether the set KEY may safely be made more specific: adding a field to the fold turns
    /// forced misses into distinct entries, which is only an improvement while those entries
    /// fit. See `report_set_key_collision`.
    pub static SET_LIVE: AtomicU64 = AtomicU64::new(0);
    pub static SET_EVICTED: AtomicU64 = AtomicU64::new(0);

    pub fn note_evicted(n: u64, bytes: u64) {
        if n == 0 {
            return;
        }
        EVICT_PASSES.fetch_add(1, Relaxed);
        EVICTED.fetch_add(n, Relaxed);
        FREED.fetch_add(bytes, Relaxed);
    }
}

/// One line for the diagnostics panel: what the guest-memory snapshot caches hold, against their
/// budgets, and what their bounds have actually cost this run.
///
/// A budget that never fires and a budget that fires every few seconds look identical in a frame
/// timing, and they are the difference between a session that holds its frame rate and one that
/// reaches single digits - see [`lru_victims`].
pub fn snapshot_cache_report() -> String {
    use std::sync::atomic::Ordering::Relaxed;
    let mb = |b: u64| b / (1024 * 1024);
    format!(
        "textures {} MB / {} MB budget over {} entries, vertices {} MB by address + {} MB by \
         content over {} entries, indices {} MB \
         (budget {} MB each) | EVICTED: {} passes, {} texture entries, {} MB reclaimed, \
         {} vertex (+ {} CONTENT-index entries, {} MB - a mesh shed here LOSES ITS IDENTITY,          which costs the renderer a full hash and compare of it, not a re-read), {} index |          binding sets {} live of {} cap ({} shed)",
        mb(snapcounts::HELD_BYTES.load(Relaxed)),
        texture_snapshot_budget() / (1024 * 1024),
        snapcounts::ENTRIES.load(Relaxed),
        mb(snapcounts::VERTEX_BYTES.load(Relaxed)),
        mb(snapcounts::VERTEX_CONTENT_BYTES.load(Relaxed)),
        snapcounts::VERTEX_CONTENT_ENTRIES.load(Relaxed),
        mb(snapcounts::INDEX_BYTES.load(Relaxed)),
        vertex_snapshot_budget() / (1024 * 1024),
        snapcounts::EVICT_PASSES.load(Relaxed),
        snapcounts::EVICTED.load(Relaxed),
        mb(snapcounts::FREED.load(Relaxed)),
        snapcounts::VERTEX_EVICTED.load(Relaxed),
        snapcounts::VERTEX_INTERN_EVICTED.load(Relaxed),
        mb(snapcounts::VERTEX_INTERN_EVICTED_BYTES.load(Relaxed)),
        snapcounts::INDEX_EVICTED.load(Relaxed),
        snapcounts::SET_LIVE.load(Relaxed),
        SET_CAP,
        snapcounts::SET_EVICTED.load(Relaxed),
    )
}

/// The state the budget is genuinely wrong for: nothing was evictable because every retained
/// buffer belongs to the frame being built. Reported once - it is a property of the title's
/// working set, not an event, so repeating it would bury everything else.
pub(super) fn report_snapshot_working_set_over_budget(held_mb: usize, entries: usize) {
    use std::sync::atomic::{AtomicBool, Ordering::Relaxed};
    static SAID: AtomicBool = AtomicBool::new(false);
    if SAID.swap(true, Relaxed) {
        return;
    }
    tracing::warn!(
        target: "vitaslop::gxm",
        held_mb,
        entries,
        "texture snapshot cache is over budget and NOTHING is evictable - every retained \
         buffer belongs to the frame being built, so this title's one-frame working set does \
         not fit VITASLOP_SNAPSHOT_BUDGET_MB. The frame is finished anyway; raise the budget \
         if this title is the target. Reported once."
    );
}

/// >>> WHICH ENTRIES TO SHED TO GET A BYTE-BUDGETED CACHE BACK UNDER ITS BUDGET.
///
/// Least recently used first, and NEVER an entry the current frame has already used - that one
/// is exactly what the rest of the frame is about to ask for again, so evicting it would be the
/// disease as the cure.
///
/// # Why this exists, and why it is worth more than any amount of tuning the budgets
/// Every cache in this file used to `clear()` at its cap. That is not "a rebuild" once: it is a
/// rebuild of the whole working set inside the NEXT frame, and it then repeats forever, because
/// whatever filled the cache the first time fills it again at the same rate. MEASURED across four
/// device dumps of one title, same code, ordered by how long the session had run: `prepare` 9.4 /
/// 25.6 / 36.8 / 67.3 us per draw at 30 / 30 / 15 / 8 fps, with every counter the frame prints
/// flat or FALLING. No hot path moved; what changed was how often these caps were hit. The
/// snapshot cache's own clear is priced in its comment at 2.1 seconds of `build`, and the
/// device's slowest-frame list is a cluster of exactly that which gets DENSER with run length.
/// [[vitaslop-caches-that-clear-whole-are-the-long-run-degradation]]
///
/// This is the shape [`RenderSceneBuilder::texture`] already used for the decode cache. It is
/// lifted here rather than reinvented, because the argument for it is the same one.
///
/// # It can return too few, deliberately
/// If everything in the cache belongs to the current frame there is nothing to shed and the
/// caller goes over budget for that frame. That is bounded by ONE FRAME's working set rather than
/// by the run, which is the property that matters, and the caller reports it.
/// The COUNT-bounded twin of [`lru_victims`], for the memos whose bound is a number of entries
/// rather than a number of bytes: which keys to shed to get back to `keep`, oldest use first.
///
/// `current` is the value a stamp has when the entry is in use RIGHT NOW - the current scene for
/// the per-scene memos - and those are never candidates, for the reason [`lru_victims`] gives.
/// If they are all current the caller is handed nothing and stays over its cap for one more
/// scene, which is bounded by the scene's own working set.
pub(super) fn count_victims<K: Copy + Eq + std::hash::Hash, V>(
    map: &FxHashMap<K, V>,
    keep: usize,
    current: u64,
    stamp: impl Fn(&V) -> u64,
) -> Vec<K> {
    if map.len() <= keep {
        return Vec::new();
    }
    let mut old: Vec<(K, u64)> = map
        .iter()
        .map(|(k, v)| (*k, stamp(v)))
        .filter(|(_, s)| *s != current)
        .collect();
    old.sort_by_key(|(_, s)| *s);
    old.truncate(map.len() - keep);
    old.into_iter().map(|(k, _)| k).collect()
}

pub(super) fn lru_victims<K: Copy + Eq + std::hash::Hash, V>(
    entries: &FxHashMap<K, V>,
    used: &FxHashMap<K, u64>,
    held: usize,
    budget: usize,
    frame: u64,
    size: impl Fn(&V) -> usize,
) -> Vec<K> {
    // An entry with no stamp at all is older than any stamped one - it has not been used since
    // whatever dropped its stamp - so it sorts first rather than being excluded.
    let mut stale: Vec<(K, u64)> = entries
        .keys()
        .map(|k| (*k, used.get(k).copied().unwrap_or(0)))
        .filter(|(_, u)| *u != frame)
        .collect();
    stale.sort_by_key(|(_, u)| *u);
    let mut freed = 0usize;
    let mut out = Vec::new();
    for (k, _) in stale {
        if held.saturating_sub(freed) < budget {
            break;
        }
        freed += entries.get(&k).map_or(0, &size);
        out.push(k);
    }
    out
}

/// Byte budget for retained texture snapshots, scaled to the device
/// ([`crate::knobs::memory_scale`]). Past it the LEAST RECENTLY USED entries are shed until it
/// fits - see [`TextureSnapshots::evict_textures_to_budget`].
///
/// # What the wholesale clear this replaced actually cost, measured
/// It is kept here because it is the argument for the eviction, and because the same reasoning
/// applies to any future cache added beside this one.
/// "Cleared whole" is not a local event. Every retained entry is an `Arc` that the
/// RENDERER's decode cache is keyed on by IDENTITY, so dropping them all means the next
/// scene re-reads every texture into a fresh buffer, misses the decode cache on all of
/// them, and re-decodes the whole working set inside one frame. MEASURED in the browser
/// mid-race on a 703-draw frame: **263 textures decoded, 178 MB of RGBA8 produced, and
/// `build` took 2,128 ms** - against 1.7 ms for a 430-draw frame that decoded nothing.
///
/// # The budget was NOT the cause of that burst, and the default is unchanged
/// Raising it to 512 MB was tried and MEASURED: the clear never fired at either setting,
/// and the burst frame was identical (2,145 ms against 2,128 ms, 263 decodes either way).
/// The burst is the first PRESENTED frame after a fast-forward, which presents nothing and
/// so leaves every cache cold - a property of the harness, not of this budget. The default
/// was left at 192 MB then because nothing measured asked for more, and a constant raised on a
/// refuted theory is a constant fitted to noise. (It is now LOWER, for memory - see below.)
///
/// What the investigation DID leave behind is the report below, which is now a `warn`:
/// when this does fire it costs a full re-decode, and it used to say so only at `info`.
/// `VITASLOP_SNAPSHOT_BUDGET_MB` overrides it; `=192` is the arm back to the old default.
///
/// # >>> 96 MB, BECAUSE THE WASM HEAP IS WHAT KILLS A PHONE TAB AND IT NEVER SHRINKS
/// A first-person shooter crashed the phone's tab ("Aw, Snap!"). On the desktop browser its
/// heap owner ledger (`VITASLOP_HEAP_OWNERS`) put **232 MB** of the Rust heap in texture
/// snapshots, against a frame that binds ~54-90 MB of texture. MEASURED with this at 96 and the
/// renderer's decode cache at 96 (`VITASLOP_DECODE_CACHE_MB`), one build, those two knobs the
/// only difference: Rust heap live 1,248 -> 940 MB, wasm heap 1,398 -> 1,111 MB, renderer
/// working set peak 2.71 -> 2.33 GB, snapshots 232 -> 83 MB, `RE-decoded after eviction` 0.0 a
/// frame in the steady window, speed 105% -> 97% (within this rig's run-to-run spread of
/// 94-105%). The floor at one frame's working set is untouched, so a frame never re-reads its
/// own textures; what a smaller budget costs is a re-read of a texture idle long enough to age
/// out, and no texel - the snapshot is CHECKED against guest memory either way.
pub(super) fn texture_snapshot_budget() -> usize {
    static CELL: std::sync::OnceLock<usize> = std::sync::OnceLock::new();
    let base = *CELL.get_or_init(|| {
        crate::knobs::var("VITASLOP_SNAPSHOT_BUDGET_MB")
            .ok()
            .and_then(|s| s.trim().parse::<usize>().ok())
            .unwrap_or(96)
            << 20
    });
    // >>> SCALED AT EVERY CALL, NOT AT THE ONCE-LOCK. The knob is parsed once; the DEVICE scale
    // is applied per call, because the frontend reports the device's memory during startup and a
    // value folded into the `OnceLock` would be whatever was known the first time a texture was
    // bound. See [`crate::knobs::memory_scale`] - it is one relaxed load and a multiply.
    crate::knobs::scale_budget(base)
}

/// `VITASLOP_STAMP_VERIFY=1` - see the check in `TextureSnapshots::get_or_read`.
pub(super) fn stamp_verify() -> bool {
    use std::sync::OnceLock;
    static ON: OnceLock<bool> = OnceLock::new();
    *ON.get_or_init(|| matches!(crate::knobs::var("VITASLOP_STAMP_VERIFY").as_deref(), Ok("1")))
}

/// Once per range: the dirty stamps said these bytes were untouched and they had changed.
pub(super) fn report_unstamped_change(ctx: &GuestCtx, addr: u32, len: usize, old: &[u8]) {
    use std::collections::HashSet;
    use std::sync::Mutex;
    static SEEN: Mutex<Option<HashSet<(u32, usize)>>> = Mutex::new(None);
    if !SEEN.lock().unwrap_or_else(|e| e.into_inner()).get_or_insert_with(HashSet::new).insert((addr, len)) {
        return;
    }
    let Some(live) = ctx.borrow_bytes(addr, len) else { return };
    let first = live.iter().zip(old).position(|(a, b)| a != b).unwrap_or(0);
    let last = live.iter().zip(old).rposition(|(a, b)| a != b).unwrap_or(0);
    let changed = live.iter().zip(old).filter(|(a, b)| a != b).count();
    tracing::warn!(
        target: "vitaslop::warning",
        "STAMP VERIFY: {addr:#010x}+{len:#x} CHANGED with no stamp - {changed} byte(s) differ,          first at {:#010x}, last at {:#010x}",
        addr as usize + first,
        addr as usize + last
    );
}

/// What [`TextureSnapshots::palette_expanded`] may hold, in bytes.
///
/// Half the snapshot budget: an expansion is four times the indices it came from, and the
/// working set of paletted art a title has on screen is a fraction of the textures it holds. A
/// title whose paletted set is genuinely larger than this pays a re-expansion rather than a
/// wrong picture, and the clear reports itself.
///
/// Pinned at half the OLD 192 MB snapshot default rather than following the snapshot budget
/// down to 96: that cut was measured on a title with no paletted art, and a halved palette memo
/// is a re-expansion cost nobody measured.
pub(super) fn palette_memo_budget() -> usize {
    crate::knobs::scale_budget(96 << 20)
}

/// >>> HOW MUCH OF A TEXTURE THE GUEST ACTUALLY TOUCHED, when the dirty map says it touched
/// > > > SOMETHING (`VITASLOP_TEX_DIRTY_CENSUS=1`).
///
/// The map answers one bit for a whole range, so a texture is re-read in full whenever any
/// page of it was written - and in a browser that read is a copy of the whole texture across
/// the guest boundary. MEASURED on a movie screen in Chrome: `draw: texture snapshot compare`
/// still moves 0.5 MB a frame while nothing new is uploaded, i.e. the bytes come back
/// IDENTICAL every time.
///
/// Two explanations fit that, and they call for opposite things: the guest rewrites the whole
/// texture with the same bytes (nothing to save - the read is the price of exactness), or one
/// or two pages of it are hot, or merely ADJACENT (the map is read with a one-page overhang,
/// so a store below the texture marks it), in which case re-reading only the dirty pages would
/// answer the same question for a fraction of the bytes. This counts pages so the next change
/// is chosen rather than guessed.
///
/// It is a diagnostic and it is expensive - one map probe per 4 KB page - so it is off unless
/// asked for, and it reports on a ladder rather than per texture.
pub(super) fn dirty_page_census(ctx: &GuestCtx, addr: u32, len: usize, stamps: &FxHashMap<(u32, usize), u8>) {
    use std::sync::atomic::{AtomicU64, Ordering::Relaxed};
    static ON: std::sync::OnceLock<bool> = std::sync::OnceLock::new();
    if !*ON.get_or_init(|| crate::knobs::flag("VITASLOP_TEX_DIRTY_CENSUS")) {
        return;
    }
    let Some(&stamp) = stamps.get(&(addr, len)) else {
        return;
    };
    const PAGE: usize = 4096;
    let pages = len.div_ceil(PAGE);
    let mut dirty = 0usize;
    for p in 0..pages {
        let at = addr.wrapping_add((p * PAGE) as u32);
        // One page at a time, and only the page itself: `dirty_since` widens by one page
        // BELOW its range, so asking about page p alone still includes p-1 - which is
        // exactly the overhang this census exists to expose, and it is counted as dirty
        // for page p because that is what the whole-range answer does too.
        if ctx.dirty_since(at, 1, stamp) != Some(false) {
            dirty += 1;
        }
    }
    static RANGES: AtomicU64 = AtomicU64::new(0);
    static PAGES: AtomicU64 = AtomicU64::new(0);
    static DIRTY: AtomicU64 = AtomicU64::new(0);
    let n = RANGES.fetch_add(1, Relaxed) + 1;
    let total = PAGES.fetch_add(pages as u64, Relaxed) + pages as u64;
    let hot = DIRTY.fetch_add(dirty as u64, Relaxed) + dirty as u64;
    if n.is_multiple_of(200) {
        tracing::warn!(
            target: "vitaslop::gxm",
            ranges = n,
            pages = total,
            dirty_pages = hot,
            share = format_args!("{:.1}%", hot as f64 * 100.0 / total.max(1) as f64),
            "texture compare: how much of a re-read range the guest had actually written"
        );
    }
}

/// One paletted texture's expansion, against the inputs it is a function of - see
/// [`TextureSnapshots::expanded_palette`].
pub(super) struct PaletteExpansion {
    /// The identity of the INDEX buffer that was expanded, as the snapshot layer minted it.
    pub(super) src: u64,
    /// A hash of the palette's bytes. The palette is a second guest allocation that can be
    /// rewritten under a texture whose own bytes never move, so it belongs in the key.
    pub(super) pal: u64,
    /// The expanded RGBA8, and the identity minted for IT, once.
    pub(super) pixels: Arc<[u8]>,
    pub(super) id: u64,
    /// The shape the expansion produced, which the caller would otherwise have to re-derive.
    pub(super) stride: u32,
    pub(super) face_bytes: u32,
}
