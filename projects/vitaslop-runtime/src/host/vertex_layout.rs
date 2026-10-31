use super::*;

/// Per-draw vertex program layout captured at create time, keyed by the vertex
/// program handle so a later Draw knows how to snapshot the vertex buffer.
pub(super) struct VertexProgramInfo {
    // Captured at create time and kept beside the streams it belongs with, though the snapshot
    // path reads the streams only.
    #[allow(dead_code)]
    pub(super) attributes: Vec<crate::capture::VertexAttribute>,
    /// >>> THE INTERLEAVED LAYOUT A DRAW ACTUALLY CAPTURES INTO, BUILT ONCE HERE.
    ///
    /// `record_draw` interleaves every stream an attribute names into ONE packed buffer and
    /// rewrites the attributes onto it. What that rewrite produces - the packed row stride,
    /// each stream's byte base within a row, whether the mesh is a single per-vertex stream,
    /// and the rebased attribute list - depends only on this program's declared attributes and
    /// its streams' strides, all of which are fixed when the vertex program is CREATED.
    ///
    /// It was recomputed on every draw, at the cost of cloning the attribute list, collecting a
    /// stream vector, building a base vector and looping over the attributes to add the bases -
    /// four allocations and two loops per draw, 983 draws a presented frame, all to arrive at
    /// the same constant every time. Doing it at registration makes a draw a refcount bump and
    /// two field reads. Rebuilt whenever the handle is registered again, which is the only
    /// moment any of it can change.
    pub(super) packed_attributes: std::sync::Arc<[crate::capture::VertexAttribute]>,
    /// The streams an attribute actually names (`0..=max stream_index`), in stream order.
    pub(super) used_streams: std::sync::Arc<[VertexStreamInfo]>,
    /// Byte offset of each used stream's row within a packed row - the prefix sum of the
    /// strides above.
    pub(super) stream_base: std::sync::Arc<[u32]>,
    /// Bytes per packed vertex: the sum of the used streams' strides.
    pub(super) packed_stride: u32,
    /// How a draw of this program scatters the guest's streams into that row - see
    /// [`RowLayout`]. One piece per stream under the column form, one per ATTRIBUTE under the
    /// aligned form.
    pub(super) layout: RowLayout,
    /// Exactly one used stream, stepped per VERTEX - the overwhelmingly common case, whose
    /// rows are already contiguous in guest memory and can be taken in one read.
    pub(super) single_stream: bool,
    /// One entry per `SceGxmVertexStream` the program was created with, in stream order.
    /// A draw's attributes name a stream each, and only this tells us how wide that
    /// stream's rows are and whether it is stepped per vertex or per instance.
    pub(super) streams: Vec<VertexStreamInfo>,
    /// The `SceGxmProgram*` this vertex program was created from, so a precomputed
    /// vertex state (which references the vertex program) can size its default uniform
    /// buffer from the program header (+0x2C). 0 if it could not be resolved.
    pub(super) program_header: u32,
}

impl VertexProgramInfo {
    /// Stream `i`'s layout, or a zero-stride per-vertex stream if the program declared
    /// fewer streams than an attribute references (which a well-formed program cannot do,
    /// and which then contributes no bytes rather than reading somewhere arbitrary).
    // Kept: the doc states what a program that under-declares its streams does, which is a
    // property of the format rather than of this caller.
    #[allow(dead_code)]
    pub(super) fn stream(&self, i: usize) -> VertexStreamInfo {
        self.streams.get(i).copied().unwrap_or_default()
    }
}

/// One `SceGxmVertexStream`: `{ uint16_t stride; uint16_t indexSource; }`.
#[derive(Clone, Copy, Default, PartialEq, Eq)]
pub(super) struct VertexStreamInfo {
    pub(super) stride: u32,
    /// Bytes this stream occupies in the PACKED row, which is not always its stride.
    ///
    /// # A stride of ZERO is a real GXM layout, and it is not an empty column
    /// A vertex stream with `stride == 0` makes every vertex read the SAME row - the idiom for
    /// passing per-draw constants (an instance transform, a colour) through a stream rather
    /// than a uniform. Sizing its column by the stride gives it NO room, so `stream_base`
    /// pushes the attributes that reference it past the end of the packed row and every fetch
    /// lands in the following vertices' bytes.
    ///
    /// MEASURED on a fighting title's instanced particle program (`vert_90ba45c0`, strides
    /// `[12, 12, 4, 0]`): its six `Input.Instance*` attributes sit on stream 3 at offsets
    /// 0..64, were rebased to 28..92 inside a 28-byte packed row, and alternate vertices came
    /// out at position (0,0,0). On screen that is a full-screen woven lattice of stretched
    /// strips - the title's long-standing grey "slats" and stray flat shapes.
    ///
    /// So the column is `max(stride, the extent of the attributes that name this stream)`.
    pub(super) column: u32,
    /// True for `SCE_GXM_INDEX_SOURCE_INSTANCE_{16,32}BIT` (2 and 3): the stream is
    /// stepped by the INSTANCE number, not the vertex index, so every vertex of one
    /// instance reads the same row. Getting this wrong feeds per-instance data (a
    /// particle's world transform, a decal's placement) to the shader indexed by vertex,
    /// which scatters one object's geometry across whatever the neighbouring rows hold.
    pub(super) per_instance: bool,
}

/// ONE COPY the gather makes: `len` bytes from byte `src` of stream `stream`'s row into byte
/// `dst` of the packed row.
///
/// >>> WHY THE ROW IS DESCRIBED PER ATTRIBUTE AND NOT PER STREAM.
///
/// The interleaved row is OURS - the gather tiles the guest's bound streams into it - and
/// `VITASLOP_GXP_ALIGN_ROW` already 4-aligns each stream's COLUMN so the row can be bound
/// rather than copied. That is not enough: an attribute at an odd offset INSIDE its stream
/// stays at that offset whatever the column's base is, and WebGPU refuses a vertex attribute
/// whose offset is not a multiple of `min(4, the format's size)`. A stream laid out `U8x3` at 0
/// then anything at 3 therefore makes its pipeline unbindable on its own.
///
/// MEASURED on a baseball title's gameplay: **102 of 194 pipelines still copied**, first reason
/// `@location 1 sits at guest offset 3, not a multiple of 4`, and the draws behind them cost
/// `repack 0.47 ms` a frame plus a packed-geometry cache holding ~190 MB - a second copy of
/// every mesh, in a wasm heap that never hands a page back.
///
/// So each attribute gets its own 4-aligned slot and the scatter is described piece by piece.
/// A stream whose attributes already tile its column produces exactly the copies the column
/// form did; nothing else in the pipeline changes, because the renderer reads the attribute
/// offsets out of this same layout.
#[derive(Clone, Copy, Debug, PartialEq, Eq, Hash, Default)]
pub(crate) struct ScatterPiece {
    pub(super) stream: u16,
    pub(super) src: u16,
    pub(super) dst: u16,
    pub(super) len: u16,
}

/// How a program's draws scatter the guest's bound streams into one packed row.
#[derive(Clone, Default)]
pub(crate) struct RowLayout {
    pub(super) pieces: std::sync::Arc<[ScatterPiece]>,
    /// A CONTENT fingerprint of `pieces`, because the gather memo is keyed on the layout and an
    /// `Arc` address can be recycled by a later program - the same hazard `GatherKey::gather`
    /// holds its list to avoid, answered here without holding anything.
    pub(super) key: u64,
}

impl RowLayout {
    /// The COLUMN form: one whole-column piece per non-empty stream, which is what
    /// `set_vertex_program` builds when the per-attribute row is off. Used by the tests, which
    /// describe their streams directly rather than through a vertex program.
    #[cfg(test)]
    pub(super) fn columns(streams: &[VertexStreamInfo], base: &[u32]) -> Self {
        Self::new(
            streams
                .iter()
                .enumerate()
                .filter(|(_, st)| st.column != 0)
                .map(|(si, st)| ScatterPiece {
                    stream: si as u16,
                    src: 0,
                    dst: base.get(si).copied().unwrap_or(0) as u16,
                    len: st.column as u16,
                })
                .collect(),
        )
    }

    /// Merge every run of pieces that is contiguous on BOTH sides into one copy.
    ///
    /// >>> THE PER-ATTRIBUTE ROW IS A RENDER WIN PAID FOR ON THE CAPTURE, AND THIS IS THE
    /// >>> REFUND. MEASURED in the desktop browser on a baseball title's gameplay: describing
    /// the row per attribute took `render` from 4.40 to 3.50 ms a frame (the repack went to
    /// ZERO and the packed-geometry cache from 173 MB to none) and put 0.6 ms BACK on `cpu`,
    /// because a scatter that was one `copy_from_slice` per stream became one per attribute.
    ///
    /// Most of those splits are not real: a stream of `f32x3`, `f32x2`, `f32x4` gets slots at
    /// exactly the guest's own offsets, so the pieces abut in the source AND in the
    /// destination and the whole column is one copy again. Only a column that actually had to
    /// MOVE an attribute stays split, which is the one the alignment exists for.
    pub(super) fn merged(mut pieces: Vec<ScatterPiece>) -> Self {
        let mut out: Vec<ScatterPiece> = Vec::with_capacity(pieces.len());
        for p in pieces.drain(..) {
            match out.last_mut() {
                Some(q)
                    if q.stream == p.stream && q.src + q.len == p.src && q.dst + q.len == p.dst =>
                {
                    q.len += p.len;
                }
                _ => out.push(p),
            }
        }
        Self::new(out)
    }

    pub(super) fn new(pieces: Vec<ScatterPiece>) -> Self {
        let mut key: u64 = 0xcbf2_9ce4_8422_2325;
        for p in &pieces {
            for v in [p.stream, p.src, p.dst, p.len] {
                key ^= v as u64;
                key = key.wrapping_mul(0x0000_0100_0000_01b3);
            }
        }
        Self {
            pieces: pieces.into(),
            key,
        }
    }
    pub(super) fn pieces(&self) -> &[ScatterPiece] {
        &self.pieces
    }
}

/// A precomputed vertex- or fragment-state object (`sceGxmPrecomputed{Vertex,Fragment}
/// State*`): the default uniform buffer the guest writes its uniforms into, the bound
/// fragment/vertex textures, and the `SceGxmProgram*` (for uniform-buffer sizing).
/// Keyed by the guest state-struct address, applied to the live bind state when the
/// guest issues `sceGxmSetPrecomputed{Vertex,Fragment}State`, exactly as the individual
/// `sceGxmSetUniformDataF`/`sceGxmSetFragmentTexture` calls would be on the direct path.
/// One sampler unit's bound texture, captured the way the hardware captures it: BY VALUE.
///
/// `sceGxmSetFragmentTexture` and `sceGxmPrecomputedFragmentStateSetTexture` both COPY the
/// 16-byte `SceGxmTexture` into driver-owned memory at the moment of the call - the caller's
/// struct is `const` and is free to be a stack temporary or a slot in a recycled scratch pool.
/// Keeping only the POINTER and re-reading it at draw time therefore reads whatever the guest
/// has since put there, which on this title is zeros: a race frame's post-process chain bound
/// its three inputs through a precomputed state whose texture array had already been recycled,
/// every one of them decoded as a null data pointer, and the light, bloom and composite passes
/// all rendered pure black. The frame still looked like a frame, only far too dark.
///
/// The ADDRESS is kept beside the words because the remaining side table
/// (the recorded formats in [`TextureRegistry`]) is keyed by it - see
/// `vitaslop-host-call-reference-semantics` for why identity and value both have to survive.
/// The sampler wrap modes, filters, LOD bias, gamma and mip count used to be a third such
/// table and are now read out of `words` itself, which is what lets them survive the by-value
/// copy an address-keyed table cannot follow.
// `PartialEq` so a rebind of the SAME texture can be told from a real change - see
// `VitaState::vertex_texture_gen`.
#[derive(Clone, Copy, Debug, Default, PartialEq, Eq)]
pub(super) struct TextureBinding {
    pub(super) unit: u32,
    /// The guest `SceGxmTexture*` the binding came from. Identity only - never re-read for
    /// control words.
    pub(super) addr: u32,
    /// The four control words as they read AT BIND TIME.
    pub(super) words: [u32; 4],
    /// True when this binding arrived through a precomputed fragment state. Per-binding, not a
    /// global "last path used" flag: the two paths are live at the same time, and a global one
    /// mislabels whichever binding did not happen last - which is exactly how a precomputed
    /// binding spent a session being investigated as a direct one.
    pub(super) from_precomputed: bool,
}

/// A tally of the HOST-ROUTED writers of the context block's fragment texture slots, kept for
/// [`VitaState::report_draw_without_bound_textures`] and printed only by it. See the field's
/// own doc on `VitaState::texture_writes` for why the counts are host-routed only and which
/// knob makes them complete.
#[derive(Default)]
pub(super) struct TextureSlotWrites {
    /// `sceGxmSetFragmentTexture` with a non-null handle.
    pub(super) binds: u64,
    /// ...and with a null one, which UNBINDS the unit.
    pub(super) unbinds: u64,
    /// `sceGxmSetPrecomputedFragmentState`, which replaces the whole 16-unit array.
    pub(super) state_binds: u64,
    /// `sceGxmPrecomputedFragmentStateSetTexture` / `...SetAllTextures` - the calls that put a
    /// texture INTO a state. Never inlined, so this count is always complete, and it decides
    /// what a state's empty array MEANS: a title that never makes these calls is not asking for
    /// sixteen empty units, it is using the state for its uniforms alone.
    pub(super) state_texture_sets: u64,
    /// ...of which this many landed an array whose unit 0 is EMPTY, clearing whatever a direct
    /// bind had left there. That is the difference between "never bound" and "bound, then
    /// wiped", and it is invisible in the slots themselves.
    pub(super) state_binds_clearing_unit0: u64,
    /// The last non-null direct bind: `(frame, unit, handle)`.
    pub(super) last_bind: Option<(u64, u32, u32)>,
    /// >>> THE CONTEXT POINTER THE GUEST PASSED on the last direct bind.
    ///
    /// The two writers of a texture slot do not agree on where the slot IS. The inlined copy
    /// form writes through `r0` - the pointer the guest handed the call - while the handler
    /// ignores `r0` and writes into `VitaState::gxm_context`, the block adopted at
    /// `sceGxmCreateContext`. Those are the same address for a conforming title and nobody had
    /// ever checked, so the report prints both: one number decides whether a bind that the host
    /// counts is landing where the draw reads.
    pub(super) last_bind_context: u32,
}

impl TextureBinding {
    /// Snapshot the 16 control words at `addr` NOW - the moment the guest handed the texture to
    /// GXM. A null handle yields an all-zero binding, which the caller treats as an unbind.
    pub(super) fn read(ctx: &GuestCtx, unit: u32, addr: u32, from_precomputed: bool) -> Self {
        let words = if addr == 0 {
            [0; 4]
        } else {
            [
                ctx.read_u32(addr),
                ctx.read_u32(addr + 4),
                ctx.read_u32(addr + 8),
                ctx.read_u32(addr + 12),
            ]
        };
        TextureBinding {
            unit,
            addr,
            words,
            from_precomputed,
        }
    }

    /// True when the captured control words are all zero - a handle that was never a texture.
    pub(super) fn is_null(&self) -> bool {
        self.words.iter().all(|&w| w == 0)
    }
}

/// A precomputed draw: the vertex program, stream-0 vertex buffer, and draw
/// parameters the guest bundled with `sceGxmPrecomputedDraw{Init,SetVertexStream,
/// SetParams}`, replayed as a normal draw when the guest issues
/// `sceGxmDrawPrecomputed`.
///
/// The state lives IN the guest block (see [`pdraw`]), not in a host-side table keyed
/// by the block's address: a `SceGxmPrecomputedDraw` is a plain 11-word POD the guest
/// owns, and a title is free to build one and then `memcpy` it - or an array of them -
/// to the addresses it actually draws from. An address-keyed table cannot follow such a
/// by-value copy and silently loses every draw made through the copy.
#[derive(Clone, Copy, Default)]
pub(super) struct PrecomputedDraw {
    pub(super) vertex_program: u32,
    pub(super) streams: [u32; pdraw::STREAM_SLOTS],
    pub(super) primitive: u32,
    pub(super) index_format: u32,
    pub(super) index_addr: u32,
    pub(super) index_count: u32,
    /// `sceGxmPrecomputedDrawSetParamsInstanced`'s wrap; 0 for a non-instanced bundle.
    pub(super) index_wrap: u32,
}
