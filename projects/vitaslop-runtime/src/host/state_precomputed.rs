use super::*;

impl VitaState {
    /// Record `sceGxmPrecomputedDrawInit(precomputedDraw, vertexProgram, memBlock)`:
    /// start a precomputed-draw record keyed by the guest block address, bound to the
    /// given vertex program (its handle). Any prior record at that address is replaced.
    pub fn precomputed_draw_init(
        &mut self,
        ctx: &mut GuestCtx,
        precomputed: u32,
        vertex_program: u32,
    ) {
        for w in 0..pdraw::WORDS {
            ctx.write_u32(precomputed + w * 4, 0);
        }
        ctx.write_u32(precomputed + pdraw::OFF_MAGIC, pdraw::MAGIC);
        ctx.write_u32(precomputed + pdraw::OFF_VERTEX_PROGRAM, vertex_program);
        self.shadow_pdraw(ctx, precomputed);
    }

    /// Record the block as our setters just left it - see [`Self::note_pdraw_guest_edits`].
    pub(super) fn shadow_pdraw(&mut self, ctx: &GuestCtx, precomputed: u32) {
        let (words, _) = Self::precomputed_draw_read(ctx, precomputed);
        self.precomputed.shadow_draw(precomputed, words);
    }

    /// >>> A PRECOMPUTED-DRAW WORD THE GUEST CHANGED WITHOUT CALLING A SETTER.
    ///
    /// The block's layout is OURS (see [`pdraw`]), not the driver's, so a title that patches
    /// a field in place - a relocated vertex buffer's new address written straight into the
    /// draw it belongs to - lands its store on a word that means something else here, and
    /// the draw keeps the old value. Reported once per (block, word), with both values, so a
    /// title that does this names the word and what it wrote.
    ///
    /// `words` is the block as [`Self::precomputed_draw_read`] just read it, so the check costs
    /// no guest read of its own on the path every precomputed draw takes.
    #[inline]
    pub(super) fn note_pdraw_guest_edits(
        &mut self,
        ctx: &GuestCtx,
        precomputed: u32,
        words: &[u32; pdraw::WORDS as usize],
    ) {
        self.precomputed
            .note_guest_edits(precomputed, words, ctx.regs[14]);
    }

    /// Record `sceGxmPrecomputedDrawSetVertexStream(precomputedDraw, streamIndex, data)`.
    pub fn precomputed_draw_set_stream(
        &mut self,
        ctx: &mut GuestCtx,
        precomputed: u32,
        stream_index: u32,
        data: u32,
    ) {
        match pdraw::OFF_STREAM.get(stream_index as usize) {
            Some(&off) => {
                ctx.write_u32(precomputed + off, data);
                self.shadow_pdraw(ctx, precomputed);
            }
            None => tracing::warn!(
                target: "vitaslop::gxm",
                stream_index,
                data = format_args!("{data:#x}"),
                "precomputedDrawSetVertexStream beyond SCE_GXM_MAX_VERTEX_STREAMS - DROPPED"
            ),
        }
    }

    /// Record `sceGxmPrecomputedDrawSetParams(precomputedDraw, prim, indexType,
    /// indexData, indexCount)`.
    pub fn precomputed_draw_set_params(
        &mut self,
        ctx: &mut GuestCtx,
        precomputed: u32,
        primitive: u32,
        index_format: u32,
        index_addr: u32,
        index_count: u32,
        index_wrap: u32,
    ) {
        ctx.write_u32(precomputed + pdraw::OFF_PRIMITIVE, primitive);
        ctx.write_u32(precomputed + pdraw::OFF_INDEX_FORMAT, index_format);
        ctx.write_u32(precomputed + pdraw::OFF_INDEX_ADDR, index_addr);
        ctx.write_u32(precomputed + pdraw::OFF_INDEX_COUNT, index_count);
        ctx.write_u32(precomputed + pdraw::OFF_INDEX_WRAP, index_wrap);
        self.shadow_pdraw(ctx, precomputed);
    }

    /// Read back a precomputed draw from its guest block, or `None` when the block does
    /// not carry the initialised tag.
    /// >>> ONE BULK READ, NOT TEN WORD READS, and the block is the most-read structure in a
    /// > > > race frame.
    ///
    /// A `dyn GuestMemory` word read is a bounds check plus a virtual call, and in the browser
    /// it crosses into a `SharedArrayBuffer` view; the bytes are never the cost, the CALLS are
    /// ([[vitaslop-count-calls-not-bytes-across-the-guest-boundary]]). This is called once per
    /// `sceGxmDrawPrecomputed`, which a measured race grid issues **411 times a frame** - so the word
    /// form was ~4,100 crossings a frame to move 1.8 kB that sit CONSECUTIVELY in guest memory.
    /// The whole block is [`pdraw::WORDS`] words with no gaps worth skipping.
    pub(super) fn precomputed_draw_read(
        ctx: &GuestCtx,
        precomputed: u32,
    ) -> ([u32; pdraw::WORDS as usize], Option<PrecomputedDraw>) {
        let mut buf = [0u8; pdraw::WORDS as usize * 4];
        ctx.read_into(precomputed, &mut buf);
        let words: [u32; pdraw::WORDS as usize] = std::array::from_fn(|i| {
            u32::from_le_bytes([buf[4 * i], buf[4 * i + 1], buf[4 * i + 2], buf[4 * i + 3]])
        });
        let word = |off: u32| words[off as usize / 4];
        if word(pdraw::OFF_MAGIC) != pdraw::MAGIC {
            return (words, None);
        }
        (
            words,
            Some(PrecomputedDraw {
                vertex_program: word(pdraw::OFF_VERTEX_PROGRAM),
                streams: std::array::from_fn(|i| word(pdraw::OFF_STREAM[i])),
                primitive: word(pdraw::OFF_PRIMITIVE),
                index_format: word(pdraw::OFF_INDEX_FORMAT),
                index_addr: word(pdraw::OFF_INDEX_ADDR),
                index_count: word(pdraw::OFF_INDEX_COUNT),
                index_wrap: word(pdraw::OFF_INDEX_WRAP),
            }),
        )
    }

    /// Replay `sceGxmDrawPrecomputed(context, precomputedDraw)`: bind the precomputed
    /// draw's vertex program + stream-0 buffer and record it into the current scene,
    /// exactly as a `sceGxmDraw` would. The bound textures and reserved uniform buffer
    /// are whatever the guest set on the context around this call (sticky GXM state).
    pub fn draw_precomputed(&mut self, ctx: &mut GuestCtx, precomputed: u32) {
        let (words, d) = Self::precomputed_draw_read(ctx, precomputed);
        self.note_pdraw_guest_edits(ctx, precomputed, &words);
        let Some(d) = d else {
            // A block with no initialised tag: the draw would be LOST, which shows up in the
            // frame only as missing geometry. Say so rather than return quietly.
            tracing::debug!(
                target: "vitaslop::gxm",
                precomputed = format_args!("{precomputed:#x}"),
                "drawPrecomputed on an uninitialised block - draw DROPPED"
            );
            return;
        };
        // The precomputed block's bindings are used FOR THIS DRAW: `record_draw` lays them over
        // its one snapshot of the context block, so every reader of that snapshot sees them
        // where it sees every draw's.
        //
        // >>> A PRECOMPUTED DRAW DOES NOT CHANGE THE CONTEXT'S STICKY VERTEX STATE. It carries
        // its own vertex program and streams, and an immediate draw after it still sees what
        // `sceGxmSetVertexProgram` / `sceGxmSetVertexStream` last bound. This used to LEAVE the
        // block's bindings in the context, which a title that filters redundant binds cannot
        // survive: a football title binds a mesh's stream, draws it, issues precomputed draws,
        // then draws the SAME mesh again through another vertex program without re-binding
        // (it knows the stream is still bound) - and got the last precomputed draw's stream, a
        // different mesh's F32 buffer read as F16 positions: shards across its whole main menu.
        //
        // >>> AND SO THE CONTEXT IS NEVER WRITTEN AT ALL. This used to write the bindings into
        // the guest's context block, draw, and write the saved ones back: five single-word reads
        // and four writes per call, each a JS boundary crossing in the browser, to move five
        // words `record_draw` then re-read with the rest of the block. MEASURED on a football
        // title's crowd camera: ~330 precomputed draws a frame, 5.0 single-word reads each -
        // every single-word read the draw path made. The overlay leaves the context untouched
        // by construction, so the rule above needs no restore to keep.
        self.pdraw_binding = Some((d.vertex_program, d.streams));
        // Which CALL issued the draw, for the empty-bindings report. The two draw entry points
        // read the same sticky state today, so this changes nothing about the draw - it exists
        // because "an immediate draw lost its texture" and "a precomputed draw was handed a
        // state that has none" are different defects with different fixes, and the report
        // cannot tell them apart from the state alone.
        self.last_draw_was_precomputed = true;
        self.last_precomputed_block = precomputed;
        self.record_draw(
            ctx,
            d.primitive,
            d.index_format,
            d.index_addr,
            d.index_count,
            d.index_wrap,
        );
        self.last_draw_was_precomputed = false;
        self.last_precomputed_block = 0;
        self.pdraw_binding = None;
    }

    // --- Precomputed vertex/fragment state ----------------------------------
    //
    // A precomputed state bundles the uniform buffer + textures for one shader stage
    // into a guest struct the game builds once and binds each draw. We record the
    // guest-set pointers per state address, then `bind_precomputed_*_state` copies them
    // into the live bind state so `record_draw` snapshots the same bytes it would on the
    // direct `sceGxmSetUniformDataF`/`sceGxmSetFragmentTexture` path.

    /// The guest state struct at `state`, ENSURED: stamped `magic` with an arrays block of
    /// `block_bytes` behind it (see [`crate::vita::gxmstate`]). Returns the block address.
    ///
    /// A struct that already carries the stamp keeps its block - `Init` re-zeroes what it
    /// must - and one that does not (never initialised, or a setter arrived first, which
    /// the old table served through `entry().or_default()`) gets a fresh zeroed block from
    /// the guest heap. The title's own `memBlock` is deliberately NOT used: its size is the
    /// real driver's, which this engine does not define.
    ///
    /// >>> A BLOCK POINTER READ OUT OF GUEST MEMORY IS TRUSTED ONLY IF THIS ENGINE MADE IT.
    ///
    /// The struct is the title's memory, and titles recycle it: a state freed back to the
    /// title's allocator keeps our magic in its first word while the rest is reused for
    /// something else. MEASURED on a tennis title's court load (~7,000 state inits in 500
    /// frames, on recycled memory): such a struct carried the magic and, in the BLOCK slot, a
    /// C++ vtable pointer into the READ-ONLY text segment - and `Init` zeroed 440 bytes there,
    /// so a later virtual call through that vtable dispatched to address 0 and the run died
    /// (intermittently: it needs that exact reuse, ~1 run in 3 in the browser). So the block
    /// is reused only when [`PrecomputedRecords`] says this engine allocated it; anything else is a
    /// struct this engine never stamped, and gets a fresh block.
    ///
    /// `for_init` asks for more: the block must have been allocated for THIS struct. An `Init`
    /// is a new state - on hardware it gets the `memBlock` the title just handed it - so a
    /// struct that is a by-value COPY of another (which shares its block, as a copied state
    /// shares driver memory) is given its own block rather than zeroing the original's.
    pub(super) fn ensure_state_block(
        &mut self,
        ctx: &mut GuestCtx,
        state: u32,
        magic: u32,
        block_bytes: u32,
        for_init: bool,
    ) -> u32 {
        use crate::vita::gxmstate::off;
        if ctx.read_u32(state.wrapping_add(off::MAGIC)) == magic {
            let block = ctx.read_u32(state.wrapping_add(off::BLOCK));
            let ours = self.precomputed.block_owner(block);
            if block != 0 && ours.is_some_and(|owner| !for_init || owner == state) {
                return block;
            }
        }
        let block = self.galloc(block_bytes, 16);
        self.precomputed.claim_block(block, state);
        // ONE range write each, not a word loop: every guest write from the host is a
        // boundary crossing in the browser, and the fragment block is 100+ words. MEASURED on a
        // football title's live frame: `sceGxmPrecomputedFragmentStateInit` at 52 us a call,
        // ~3 a frame, was the most expensive call it makes after the draw itself - and on a
        // phone a crossing is ~20x the desktop's.
        ctx.write_bytes(state, &[0u8; off::BYTES as usize]);
        ctx.write_bytes(block, &vec![0u8; block_bytes as usize]);
        ctx.write_u32(state.wrapping_add(off::BLOCK), block);
        ctx.write_u32(state.wrapping_add(off::MAGIC), magic);
        block
    }

    /// `sceGxmPrecomputedVertexStateInit(state, vertexProgram, memBlock)`: stamp the guest
    /// struct, attach its arrays block, and memoise the program facts the bind needs -
    /// including the stage's uniform SIZE, which is fixed the moment the program exists
    /// (the same move the inlined reserve rests on). Replaces any prior state there.
    pub fn precomputed_vertex_state_init(
        &mut self,
        ctx: &mut GuestCtx,
        state: u32,
        vertex_program: u32,
    ) {
        use crate::vita::gxmstate::{self, off};
        let program_header = self.vertex_program_header(vertex_program);
        let block = self.ensure_state_block(
            ctx,
            state,
            gxmstate::MAGIC_VERTEX,
            gxmstate::VERTEX_BLOCK_BYTES,
            true,
        );
        // Re-init REPLACES: the arrays block and the mutable fields go back to zero. One range
        // write, not a word loop - see `ensure_state_block`.
        ctx.write_bytes(block, &[0u8; gxmstate::VERTEX_BLOCK_BYTES as usize]);
        // What the vertex bind's record carries: the reflected extent, never smaller than
        // the header's own declared default-uniform size - computed here once, exactly as
        // the bind used to compute it per call. Both inputs are immutable while the
        // program is registered (see `ProgramReflection`).
        let size = self
            .reflected_uniform_size_bytes(ctx, program_header)
            .max(default_uniform_buffer_bytes(ctx, program_header));
        ctx.write_u32(state.wrapping_add(off::HANDLE), vertex_program);
        ctx.write_u32(state.wrapping_add(off::HEADER), program_header);
        ctx.write_u32(state.wrapping_add(off::BUF), 0);
        ctx.write_u32(state.wrapping_add(off::SIZE), size);
        self.cross_precomputed_state_programs(ctx, program_header, true);
        self.report_precomputed_state_programs();
    }

    /// `sceGxmPrecomputedFragmentStateInit(state, fragmentProgram, memBlock)`.
    pub fn precomputed_fragment_state_init(
        &mut self,
        ctx: &mut GuestCtx,
        state: u32,
        fragment_program: u32,
    ) {
        use crate::vita::gxmstate::{self, off};
        let program_header = self.fragment_program_header(fragment_program);
        let block = self.ensure_state_block(
            ctx,
            state,
            gxmstate::MAGIC_FRAGMENT,
            gxmstate::FRAGMENT_BLOCK_BYTES,
            true,
        );
        // One range write, not a word loop - see `ensure_state_block`.
        ctx.write_bytes(block, &[0u8; gxmstate::FRAGMENT_BLOCK_BYTES as usize]);
        // The fragment bind's record uses the reflected size alone, exactly as the bind
        // used to compute it per call.
        let size = self.uniform_size_bytes(ctx, program_header);
        ctx.write_u32(state.wrapping_add(off::HANDLE), fragment_program);
        ctx.write_u32(state.wrapping_add(off::HEADER), program_header);
        ctx.write_u32(state.wrapping_add(off::BUF), 0);
        ctx.write_u32(state.wrapping_add(off::SIZE), size);
        self.cross_precomputed_state_programs(ctx, program_header, false);
        self.report_precomputed_state_programs();
    }

    /// Offer the shader pairs a title's PRECOMPUTED STATES imply, so the renderer can compile
    /// them while the loading screen that declares them is still up.
    ///
    /// # Why this list, when the patcher's cross product is refuted
    /// A title that calls `sceGxmShaderPatcherCreateFragmentProgram` with `vertexProgram = NULL`
    /// names no pair, and pairing everything it ever created with everything else is hopeless -
    /// MEASURED on a retail title at 865 x 375 = 324,375 candidates, and tried, and refused
    /// ([[vitaslop-precompile-cross-product-refuted]]). A precomputed state is a much stronger
    /// statement: the title is declaring, ahead of any draw, that this exact program is one it
    /// intends to draw with. The same title names **16 distinct vertex + 16 distinct fragment
    /// programs** that way, all of them by frame 8,821 with its race live at 9,600 - 256
    /// candidates and 779 frames of loading screen to compile them in.
    ///
    /// Only the pairs a NEW header adds are offered, so this is 16 + 15 + ... rather than 256
    /// per call, and `push_precompile_pair` drops any pair already queued. The renderer keeps
    /// the ones that LINK; a candidate that does not link costs a parse and nothing more.
    pub(super) fn cross_precomputed_state_programs(&mut self, ctx: &GuestCtx, header: u32, vertex: bool) {
        let Some(others) = self.precompile.note_state_program(header, vertex) else {
            return;
        };
        for o in others {
            let (v, f) = if vertex { (header, o) } else { (o, header) };
            self.push_precompile_pair(ctx, v, f);
        }
    }

    /// Say, on a bounded ladder, how many DISTINCT programs the title's PRECOMPUTED STATES
    /// name, and by what frame.
    ///
    /// # Why this number is the one that decides the loading-screen hitch
    /// A title that creates its fragment programs with `vertexProgram = NULL` never names a
    /// shader PAIR, so nothing can be prepared from the patcher - and the cross product over
    /// every program it ever created is hopeless: MEASURED on a retail title, **865 vertex x 375
    /// fragment = 324,375 candidates**, against a 4,096 cap the old experiment silently hit
    /// (which is the whole reason that experiment saved 43 ms and was written up as a
    /// refutation of the idea rather than of its truncation).
    ///
    /// A PRECOMPUTED STATE is a different and much smaller declaration: it is the title
    /// saying, ahead of time, "this program is one I will draw with". If a race's states name
    /// a few dozen programs between them, the cross product over THOSE is a few hundred
    /// candidates rather than a third of a million - and it is knowable while the loading
    /// screen is still up. This line is what says whether that is true, per title.
    ///
    /// Printed rather than knob-gated, for the same reason as
    /// [`Self::report_program_creation_frame`]: a handful of lines a run, answering a question
    /// every session on this hitch has to ask first.
    pub(super) fn report_precomputed_state_programs(&mut self) {
        // The distinct-header lists ARE the two counts: `cross_precomputed_state_programs`
        // pushes each non-zero header once.
        //
        // NOT every 50th state: a football title re-initialises ~50 states EVERY FRAME in
        // play (50,900 by frame 12,000), so that rung printed a line a frame - and on the
        // browser a status line is a console crossing. A NEW program is what this line is
        // for; a re-init of a known one only moves the power-of-two ladder.
        // Counts this state init too (`report_*` runs once per init, after the cross).
        let distinct = self.precompile.state_program_counts();
        let Some(n) = self.precomputed.note_state_init(distinct) else {
            return;
        };
        tracing::info!(
            target: "vitaslop::status",
            "gxm precompile: precomputed states name {} distinct vertex + {} distinct fragment \
             programs ({} states) by frame {}",
            distinct.0,
            distinct.1,
            n,
            self.cur_frame
        );
    }

    /// `sceGxmPrecomputed{Vertex,Fragment}StateSetDefaultUniformBuffer(state, buffer)`:
    /// store the guest pointer the game will write this stage's uniforms into. The record
    /// is created lazily so a setter before `Init` (unexpected) still lands.
    pub fn precomputed_vertex_state_set_uniform_buffer(
        &mut self,
        ctx: &mut GuestCtx,
        state: u32,
        buffer: u32,
    ) {
        use crate::vita::gxmstate::{self, off};
        self.ensure_state_block(
            ctx,
            state,
            gxmstate::MAGIC_VERTEX,
            gxmstate::VERTEX_BLOCK_BYTES,
            false,
        );
        ctx.write_u32(state.wrapping_add(off::BUF), buffer);
    }
    pub fn precomputed_fragment_state_set_uniform_buffer(
        &mut self,
        ctx: &mut GuestCtx,
        state: u32,
        buffer: u32,
    ) {
        use crate::vita::gxmstate::{self, off};
        self.ensure_state_block(
            ctx,
            state,
            gxmstate::MAGIC_FRAGMENT,
            gxmstate::FRAGMENT_BLOCK_BYTES,
            false,
        );
        ctx.write_u32(state.wrapping_add(off::BUF), buffer);
    }

    /// `sceGxmPrecomputed{Vertex,Fragment}StateSetUniformBuffer(state, bufferIndex,
    /// bufferData)`: store the NON-default uniform buffer binding, applied to the context
    /// block's table when the state is bound (the same table the direct
    /// `sceGxmSet{Vertex,Fragment}UniformBuffer` writes).
    pub fn precomputed_state_set_nondefault_uniform_buffer(
        &mut self,
        ctx: &mut GuestCtx,
        state: u32,
        stage: &str,
        index: u32,
        buffer: u32,
    ) {
        use crate::vita::gxmstate::{self};
        if (index as usize) >= crate::vita::gxmctx::MAX_UNIFORM_BUFFERS {
            return;
        }
        let (magic, bytes, table) = if stage == "vertex" {
            (gxmstate::MAGIC_VERTEX, gxmstate::VERTEX_BLOCK_BYTES, 0)
        } else {
            (
                gxmstate::MAGIC_FRAGMENT,
                gxmstate::FRAGMENT_BLOCK_BYTES,
                gxmstate::FRAGMENT_BLOCK_UNIFORM_BUFFERS,
            )
        };
        let block = self.ensure_state_block(ctx, state, magic, bytes, false);
        ctx.write_u32(block.wrapping_add(table + index * 4), buffer);
    }

    /// The ALL form of the setter above: bind every non-default uniform buffer index from
    /// the guest's pointer ARRAY in one go.
    ///
    /// >>> THIS IS THE SAME WRITES IN TWO CROSSINGS INSTEAD OF EIGHTY-FOUR, and it is the
    /// > > > largest single block of boundary traffic a football-title frame made.
    ///
    /// Per-index it used to be a `read_u32` of the array (one crossing), an
    /// `ensure_state_block` that re-read the state's magic AND block words (two more) and a
    /// `write_u32` into the table (three, counting the dirty-map stamp) - eighty-four
    /// crossings for fourteen pointers, on a title that calls this ONCE PER DRAW, 1,026 times
    /// a frame. It is where 41,333 of that frame's single-word guest reads came from, against
    /// a baseball title's 1,436 in total.
    ///
    /// The guest cannot run during a host call and the two tables are fourteen CONTIGUOUS
    /// words in either block, so the array is read whole, the block is resolved once, and the
    /// table is written whole. Byte for byte the same state; see
    /// [[vitaslop-count-calls-not-bytes-across-the-guest-boundary]].
    pub fn precomputed_state_set_all_nondefault_uniform_buffers(
        &mut self,
        ctx: &mut GuestCtx,
        state: u32,
        stage: &str,
        array: u32,
    ) {
        use crate::vita::gxmstate::{self};
        const TABLE_BYTES: usize = crate::vita::gxmctx::MAX_UNIFORM_BUFFERS * 4;
        let (magic, bytes, table) = if stage == "vertex" {
            (gxmstate::MAGIC_VERTEX, gxmstate::VERTEX_BLOCK_BYTES, 0)
        } else {
            (
                gxmstate::MAGIC_FRAGMENT,
                gxmstate::FRAGMENT_BLOCK_BYTES,
                gxmstate::FRAGMENT_BLOCK_UNIFORM_BUFFERS,
            )
        };
        let block = self.ensure_state_block(ctx, state, magic, bytes, false);
        // Out-of-range tail reads as zeros, exactly as the per-word `read_u32` answered for
        // it - so a short array binds nothing where it used to bind nothing.
        let mut ptrs = [0u8; TABLE_BYTES];
        ctx.read_into(array, &mut ptrs);
        ctx.write_bytes(block.wrapping_add(table), &ptrs);
    }

    /// `sceGxmPrecomputed{Vertex,Fragment}StateGetDefaultUniformBuffer(state)`: the pointer
    /// last set (0 if never set), so a Set/Get round-trips faithfully.
    pub fn precomputed_vertex_state_uniform_buffer(&self, ctx: &GuestCtx, state: u32) -> u32 {
        use crate::vita::gxmstate::{self, off};
        if ctx.read_u32(state.wrapping_add(off::MAGIC)) == gxmstate::MAGIC_VERTEX {
            ctx.read_u32(state.wrapping_add(off::BUF))
        } else {
            0
        }
    }
    pub fn precomputed_fragment_state_uniform_buffer(&self, ctx: &GuestCtx, state: u32) -> u32 {
        use crate::vita::gxmstate::{self, off};
        if ctx.read_u32(state.wrapping_add(off::MAGIC)) == gxmstate::MAGIC_FRAGMENT {
            ctx.read_u32(state.wrapping_add(off::BUF))
        } else {
            0
        }
    }

    /// `sceGxmPrecomputed{Vertex,Fragment}StateSetTexture(state, index, texture)`: bind a
    /// `SceGxmTexture*` to this stage's sampler `index` (0 unbinds), replacing any prior
    /// binding at that index - written into the state's guest texture array, in the
    /// context block's own slot layout, so the fragment bind is one wholesale copy.
    pub fn precomputed_vertex_state_set_texture(
        &mut self,
        ctx: &mut GuestCtx,
        state: u32,
        index: u32,
        texture: u32,
    ) {
        use crate::vita::gxmstate::{self};
        if index as usize >= crate::vita::gxmctx::MAX_TEXTURE_UNITS {
            self.report_precomputed_texture_unit(index, texture);
            return;
        }
        let b = TextureBinding::read(ctx, index, texture, true);
        let block = self.ensure_state_block(
            ctx,
            state,
            gxmstate::MAGIC_VERTEX,
            gxmstate::VERTEX_BLOCK_BYTES,
            false,
        );
        gxmstate::write_texture_slot(
            ctx,
            block.wrapping_add(gxmstate::VERTEX_BLOCK_TEXTURES),
            index,
            b.addr,
            b.words,
        );
    }
    pub fn precomputed_fragment_state_set_texture(
        &mut self,
        ctx: &mut GuestCtx,
        state: u32,
        index: u32,
        texture: u32,
    ) {
        use crate::vita::gxmstate::{self};
        if index as usize >= crate::vita::gxmctx::MAX_TEXTURE_UNITS {
            self.report_precomputed_texture_unit(index, texture);
            return;
        }
        self.texture_writes.state_texture_sets += 1;
        let b = TextureBinding::read(ctx, index, texture, true);
        let block = self.ensure_state_block(
            ctx,
            state,
            gxmstate::MAGIC_FRAGMENT,
            gxmstate::FRAGMENT_BLOCK_BYTES,
            false,
        );
        gxmstate::write_texture_slot(ctx, block, index, b.addr, b.words);
    }

    /// Say that a precomputed-state texture setter named a unit past the array. The old
    /// path recorded any index and dropped it at BIND time (`set_texture_binding`'s own
    /// check); with the array written at SET time the check moves here, same outcome.
    pub(super) fn report_precomputed_texture_unit(&self, unit: u32, texture: u32) {
        tracing::warn!(
            target: "vitaslop::gxm",
            unit,
            texture = format_args!("{texture:#x}"),
            "precomputed state SetTexture on a unit beyond SCE_GXM_MAX_TEXTURE_UNITS - DROPPED"
        );
    }

    /// `sceGxmPrecomputedDrawSetAllVertexStreams(precomputedDraw, streamDataArray)`:
    /// the array holds one pointer per `SceGxmVertexStream` the draw's vertex program
    /// was created with, in stream order. The count comes from the program (a caller
    /// passes no length), so a draw whose vertex program we never saw sets nothing -
    /// and says so, because silently binding no streams renders empty geometry.
    pub fn precomputed_draw_set_all_streams(
        &mut self,
        ctx: &mut GuestCtx,
        precomputed: u32,
        array: u32,
    ) {
        let handle = ctx.read_u32(precomputed + pdraw::OFF_VERTEX_PROGRAM);
        let count = match self.shader_patcher.vertex_program(handle) {
            Some(info) => info.streams.len(),
            None => {
                tracing::warn!(
                    target: "vitaslop::gxm",
                    precomputed = format_args!("{precomputed:#x}"),
                    vertex_program = format_args!("{handle:#x}"),
                    "precomputedDrawSetAllVertexStreams for an unknown vertex program - \
                     stream count unknown, NO streams bound"
                );
                0
            }
        };
        for i in 0..count as u32 {
            let data = ctx.read_u32(array.wrapping_add(i * 4));
            self.precomputed_draw_set_stream(ctx, precomputed, i, data);
        }
        // The stream array is usually a stack local built from the caller's own objects, so the
        // pointer it carries is only traceable back through the CALLER - named by its return
        // address (the callee-saved registers would say more, but the browser does not marshal
        // them into a host call). At debug, for walking a wrong buffer back to its source.
        tracing::debug!(
            target: "vitaslop::gxm",
            precomputed = format_args!("{precomputed:#x}"),
            stream0 = format_args!("{:#x}", ctx.read_u32(array)),
            lr = format_args!("{:#010x}", ctx.regs[14]),
            "precomputedDrawSetAllVertexStreams"
        );
    }

    /// `sceGxmPrecomputed{Vertex,Fragment}StateSetAllTextures(state, textureArray)`.
    ///
    /// Note the array is of `SceGxmTexture` STRUCTS, not pointers: element `i` lives at
    /// `textureArray + i*16`, and that address is exactly what the per-index setter takes.
    /// The length is the program's texture-unit count, since the caller passes none.
    pub fn precomputed_vertex_state_set_all_textures(
        &mut self,
        ctx: &mut GuestCtx,
        state: u32,
        array: u32,
    ) {
        use crate::vita::gxmstate::{self, off};
        let header = if ctx.read_u32(state.wrapping_add(off::MAGIC)) == gxmstate::MAGIC_VERTEX {
            ctx.read_u32(state.wrapping_add(off::HEADER))
        } else {
            0
        };
        let n = self.reflect_program(ctx, header).texture_unit_count;
        for i in 0..n {
            self.precomputed_vertex_state_set_texture(ctx, state, i, array.wrapping_add(i * 16));
        }
    }

    pub fn precomputed_fragment_state_set_all_textures(
        &mut self,
        ctx: &mut GuestCtx,
        state: u32,
        array: u32,
    ) {
        use crate::vita::gxmstate::{self, off};
        let header = if ctx.read_u32(state.wrapping_add(off::MAGIC)) == gxmstate::MAGIC_FRAGMENT {
            ctx.read_u32(state.wrapping_add(off::HEADER))
        } else {
            0
        };
        let n = self.reflect_program(ctx, header).texture_unit_count;
        for i in 0..n {
            self.precomputed_fragment_state_set_texture(ctx, state, i, array.wrapping_add(i * 16));
        }
    }

    /// Apply `sceGxmSetPrecomputedVertexState(context, state)`: replace the context's
    /// non-default uniform-buffer table with the state's, wholesale, and bind the state's
    /// default uniform buffer record. A state that was never initialised (no magic) clears
    /// both, exactly as the old table's miss arm did.
    ///
    /// This is the HOST side of `InlineOp::BindPrecomputedState` - the fallback for a
    /// pointer or magic the emitted guard declines, and the definition the inline form is
    /// held to. It writes exactly the words the inline form writes.
    pub fn bind_precomputed_vertex_state(&mut self, ctx: &mut GuestCtx, state: u32) {
        use crate::vita::{gxmctx, gxmstate};
        let inited = state != 0
            && ctx.read_u32(state.wrapping_add(gxmstate::off::MAGIC)) == gxmstate::MAGIC_VERTEX;
        let (buf, size, header, block) = if inited {
            (
                ctx.read_u32(state.wrapping_add(gxmstate::off::BUF)),
                ctx.read_u32(state.wrapping_add(gxmstate::off::SIZE)),
                ctx.read_u32(state.wrapping_add(gxmstate::off::HEADER)),
                ctx.read_u32(state.wrapping_add(gxmstate::off::BLOCK)),
            )
        } else {
            (0, 0, 0, 0)
        };
        // >>> A ZERO ENTRY HERE IS THIS ENGINE'S OWN FILL TOO, AND COPYING IT ERASED LIVE
        // >>> BINDINGS - the same defect the fragment bind's TEXTURE copy was fixed for, and
        // >>> this table was explicitly exempted from that fix on a premise now refuted.
        //
        // The exemption read: this table is written by the guest's own
        // `sceGxmPrecomputedVertexStateSetUniformBuffer`, so a zero entry is the guest saying
        // "no buffer", where the texture array's zeros are this engine's fill in a block it
        // allocates. But the BLOCK is the same host-allocated, host-zeroed block in both cases
        // (`ensure_state_block`), and a state that never received a
        // `SetAllUniformBuffers` carries fourteen zeros that no guest call put there.
        //
        // MEASURED on a football title, at the draws whose guest-memory windows are withheld:
        // one draw's context table reads `0=0x953ffe10 2=0x8d599a30` - the binding path plainly
        // works and real addresses do arrive - while the draws that DROP read `none`. A binding
        // that exists at one draw and is gone at the next, with a state bind in between, is a
        // binding something erased [[vitaslop-poison-separates-a-guest-zero-from-an-unwritten-one]].
        //
        // So the copy is now PER SLOT and skips empty ones, exactly as the fragment side does.
        // The emitted form must do the same or the two leave different state, which is the
        // whole contract the inline forms rest on - see `BindStateLayout::copy_slot_stride`,
        // set to one word for this stage.
        //
        // The cost of being wrong the other way is bounded and strictly smaller: if a guest
        // really does pass a zero for an index in its `SetAllUniformBuffers` array, a stale
        // binding survives where it would have been cleared - and nothing READS a buffer index
        // its program does not declare, so the stale word is inert. The old behaviour's cost
        // was a dropped draw for the life of the title.
        //
        // Still ONE read of the state's table: fourteen CONTIGUOUS words, and per-word this was
        // fifty-six boundary crossings for a bind a title makes once per draw. Only the WRITE
        // is now per-slot, and only for the slots that carry something.
        const TABLE_BYTES: usize = gxmctx::MAX_UNIFORM_BUFFERS * 4;
        let mut table = [0u8; TABLE_BYTES];
        if block != 0 {
            ctx.read_into(block, &mut table);
        }
        if self.gxm_context != 0 {
            for i in 0..gxmctx::MAX_UNIFORM_BUFFERS {
                let word = u32::from_le_bytes([
                    table[i * 4],
                    table[i * 4 + 1],
                    table[i * 4 + 2],
                    table[i * 4 + 3],
                ]);
                if word == 0 {
                    continue;
                }
                ctx.write_u32(
                    self.gxm_context
                        .wrapping_add(gxmctx::off::VERTEX_UNIFORM_BUFFERS + i as u32 * 4),
                    word,
                );
            }
        }
        // >>> AND THE STATE'S TEXTURES, WHICH THIS BIND USED NEVER TO APPLY. Binding a
        // precomputed vertex state binds its vertex textures exactly as binding a fragment
        // state binds its fragment textures; `sceGxmPrecomputedVertexStateSetTexture` recorded
        // them into the block and nothing read them back, so a vertex program that samples
        // one found its unit empty and the draw was dropped. Per slot, skipping empty ones,
        // for the reason the fragment bind and the table above do: the block is this engine's
        // own zero-filled allocation, so a zero slot is "never set", not "unbind". The emitted
        // form makes the same copy - see `BindStateLayout::table_slot_stride`.
        if block != 0 && self.gxm_context != 0 {
            let mut slots = [0u8; gxmctx::TEXTURE_ARRAY_BYTES as usize];
            ctx.read_into(block.wrapping_add(gxmstate::VERTEX_BLOCK_TEXTURES), &mut slots);
            for (unit, slot) in slots.chunks_exact(gxmctx::TEXTURE_STRIDE as usize).enumerate() {
                if slot[..4] == [0; 4] {
                    continue;
                }
                ctx.write_bytes(
                    self.gxm_context.wrapping_add(
                        gxmctx::off::VERTEX_TEXTURES + unit as u32 * gxmctx::TEXTURE_STRIDE,
                    ),
                    slot,
                );
            }
        }
        self.bind_uniform_buffer(ctx, gxmctx::off::VERTEX_UNIFORM, buf, size, header);
    }

    /// Apply `sceGxmSetPrecomputedFragmentState(context, state)`: bind this stage's
    /// textures to the context sampler units, exactly as a sequence of
    /// `sceGxmSetFragmentTexture` calls would, so `record_draw` snapshots them.
    pub fn bind_precomputed_fragment_state(&mut self, ctx: &mut GuestCtx, state: u32) {
        use crate::vita::{gxmctx, gxmstate};
        // A state never initialised through our Init has no magic; the old table's miss
        // arm returned without touching anything, so this does too.
        if state == 0
            || ctx.read_u32(state.wrapping_add(gxmstate::off::MAGIC)) != gxmstate::MAGIC_FRAGMENT
        {
            return;
        }
        let context = self.gxm_context;
        if context == 0 {
            self.report_bind_without_context();
            return;
        }
        // The state's texture array lands over the context's SLOT BY SLOT - the array is kept
        // in the context block's own slot layout, so each unit the state carries is one copy at
        // the same offset.
        //
        // It USED to land wholesale, on the reading that binding a precomputed state replaces
        // the whole array. That reading is REFUTED - see the loop below for the measurement -
        // and the difference is only ever visible for a slot the state does not carry.
        //
        // This is the HOST side of `InlineOp::BindPrecomputedState` - the fallback for a
        // pointer or magic the emitted guard declines - and it writes exactly the words
        // the inline form writes.
        let block = ctx.read_u32(state.wrapping_add(gxmstate::off::BLOCK));
        // Counted before the copy, while the source is still distinguishable from what it is
        // about to overwrite: an array whose unit 0 is empty CLEARS that unit, and the zeros it
        // leaves are indistinguishable from a unit nothing ever bound. See `TextureSlotWrites`.
        self.texture_writes.state_binds += 1;
        if block == 0 || ctx.read_u32(block) == 0 {
            self.texture_writes.state_binds_clearing_unit0 += 1;
        }
        // >>> ONLY THE UNITS THE STATE ACTUALLY CARRIES. A SLOT OF ZEROS IS OUR OWN DEFAULT,
        // >>> NOT AN UNBIND THE GUEST ASKED FOR.
        //
        // This block is allocated and ZEROED by `ensure_state_block`, and the only thing that
        // ever writes a slot in it is `precomputed_fragment_state_set_texture`. So a zero slot
        // means "no setter ever named this unit" - an unwritten value, not guest data
        // [[vitaslop-poison-separates-a-guest-zero-from-an-unwritten-one]] - and copying it
        // over the context DESTROYS a binding the guest made directly.
        //
        // MEASURED on a role-playing title, which is what found this: the title makes 19,603 direct
        // `sceGxmSetFragmentTexture` binds, puts a texture into a precomputed state **0 times**
        // (a complete count - those setters are never inlined), and binds ~1,286 states a
        // frame. A store watchpoint on unit 0's slot caught the sequence exactly: the guest
        // binds its sprite texture to unit 0, twenty state binds follow, and the immediate
        // `sceGxmDraw` after them reaches the renderer with all sixteen slots zero - the pair
        // then cannot be recompiled ("sampler unit 0 wants Two but the bound units are []").
        // The title's whole title-screen art hung on this.
        //
        // The replace-not-merge rule this had is still right for a unit the state DOES carry:
        // that slot is guest data and it wins, exactly as before. What changes is only the
        // meaning of an empty one.
        for unit in 0..gxmctx::MAX_TEXTURE_UNITS as u32 {
            let at = unit * gxmctx::TEXTURE_STRIDE;
            if block == 0 || ctx.read_u32(block.wrapping_add(at)) == 0 {
                continue;
            }
            for w in 0..gxmctx::TEXTURE_STRIDE / 4 {
                let v = ctx.read_u32(block.wrapping_add(at + w * 4));
                gxmctx::set(ctx, context, gxmctx::off::TEXTURES + at + w * 4, v);
            }
        }
        // The non-default uniform-buffer table, replace-not-merge for the same reason the
        // texture array is: a buffer bound by an earlier direct call must not survive into a
        // state that does not declare it. Its destination is nowhere near the texture block,
        // so it is its own loop rather than part of the copy above.
        // ONE crossing each way: fourteen contiguous words out of the state's block and into
        // the context's table - see `bind_precomputed_vertex_state` for the same move.
        const TABLE_BYTES: usize = gxmctx::MAX_UNIFORM_BUFFERS * 4;
        let mut table = [0u8; TABLE_BYTES];
        if block != 0 {
            ctx.read_into(
                block.wrapping_add(gxmstate::FRAGMENT_BLOCK_UNIFORM_BUFFERS),
                &mut table,
            );
        }
        ctx.write_bytes(
            context.wrapping_add(gxmctx::off::FRAGMENT_UNIFORM_BUFFERS),
            &table,
        );
        // Binding a precomputed fragment state leaves the context bound to its program, so it
        // goes where `sceGxmSetFragmentProgram` puts it. The blend equation follows from the
        // handle and needs no separate record.
        let handle = ctx.read_u32(state.wrapping_add(gxmstate::off::HANDLE));
        self.set_gxm_state_word(ctx, gxmctx::off::FRAGMENT_PROGRAM, handle);
        // Bind this stage's default uniform buffer (pointer + the size memoised at Init) so
        // the draw reads the per-material fragment uniforms (tint / light / fog) from guest
        // memory, exactly as the precomputed vertex path binds the vertex uniform buffer.
        let buf = ctx.read_u32(state.wrapping_add(gxmstate::off::BUF));
        let size = ctx.read_u32(state.wrapping_add(gxmstate::off::SIZE));
        let header = ctx.read_u32(state.wrapping_add(gxmstate::off::HEADER));
        self.bind_uniform_buffer(ctx, gxmctx::off::FRAGMENT_UNIFORM, buf, size, header);
    }
}
