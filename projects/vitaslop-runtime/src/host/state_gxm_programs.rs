use super::*;

impl VitaState {
    /// Mint a fresh opaque GXM handle.
    pub fn new_handle(&mut self) -> u32 {
        let h = self.next_handle;
        self.next_handle += 1;
        h
    }

    /// A `SceGxmVertexProgram *` / `SceGxmFragmentProgram *` handle, as a real GUEST
    /// structure carrying the two facts a reserve needs (see [`crate::vita::gxmprog`]).
    ///
    /// This is what the shader patcher does on hardware - the handle it returns points into
    /// memory the guest gave it - and the reason to do it here is that a counter has nowhere
    /// to keep a memoised size, which is what makes
    /// `sceGxmReserve*DefaultUniformBuffer` (1,189 crossings a frame on one title) a bump
    /// instead of a boundary crossing.
    ///
    /// Falls back to an opaque counter if the guest heap cannot supply sixteen bytes: the
    /// handle is still unique and every map keyed by it still works, and the reserve simply
    /// stays on the host for that program - which is exactly what the inline form's
    /// identity-stamp guard is for.
    pub fn new_program_handle(&mut self, ctx: &mut GuestCtx, header: u32) -> u32 {
        let size = self.uniform_size_bytes(ctx, header);
        let block = self.galloc(crate::vita::gxmprog::BYTES, 4);
        if block == 0 {
            return self.new_handle();
        }
        crate::vita::gxmprog::init(ctx, block, size, header);
        block
    }

    /// Record a vertex program's attribute layout and per-stream strides.
    pub fn set_vertex_program(
        &mut self,
        handle: u32,
        attributes: Vec<crate::capture::VertexAttribute>,
        streams: Vec<(u32, bool)>,
        program_header: u32,
    ) {
        // >>> THE DECLARED STRIDE IS RIGHT. A 48-BYTE READING WAS TRIED AND IS WORSE.
        //
        // One title's particle draw declares `strides=[4, 52]` while the first few hundred
        // bytes of the buffer it binds read as 48-BYTE records - four corners of a textured
        // quad each, boundary at byte 0, the pattern holding for 33 records. It is a false
        // lead: rendering the whole run with every 52 remapped to 48 makes the picture
        // DRAMATICALLY worse (90-99% of pixels differ on six frames, magenta and white sheets
        // over the whole stage), so 52 is what the mesh is actually laid out at and the 48-byte
        // reading is a coincidence of that buffer's contents. Do not spend another session on
        // it. [`dump_stream_bytes`] prints the raw window if it needs re-checking.
        let streams: Vec<VertexStreamInfo> = streams
            .into_iter()
            .map(|(stride, per_instance)| VertexStreamInfo {
                stride,
                column: stride,
                per_instance,
            })
            .collect();
        // The packed layout every draw of this program will capture into - see
        // `VertexProgramInfo::packed_attributes` for why it belongs here and not in the draw.
        let used = attributes
            .iter()
            .map(|a| a.stream_index as usize)
            .max()
            .map(|m| m + 1)
            .unwrap_or(0);
        let used_streams: std::sync::Arc<[VertexStreamInfo]> = (0..used)
            .map(|i| streams.get(i).copied().unwrap_or_default())
            .collect();
        // The extent each stream's own attributes need, so a stride-0 stream still gets a
        // column - see `VertexStreamInfo::column`.
        let extent_of = |si: usize| -> u32 {
            attributes
                .iter()
                .filter(|a| a.stream_index as usize == si)
                .map(|a| {
                    let comp = match a.format {
                        0 | 1 | 4 | 5 => 1u32,
                        2 | 3 | 6 | 7 | 8 => 2,
                        _ => 4,
                    };
                    u32::from(a.offset) + comp * u32::from(a.component_count).max(1)
                })
                .max()
                .unwrap_or(0)
        };
        // >>> ONLY A STRIDE-0 STREAM TAKES ITS COLUMN FROM ITS ATTRIBUTES.
        //
        // This was `max(stride, extent)` for every stream, and the extent can EXCEED a real
        // stride: an `UNTYPED x2` attribute fetches eight bytes whatever the program reads of
        // them, so one placed four bytes before the end of its row overhangs into the next
        // vertex - exactly as the hardware's fetch does, harmlessly, because the program reads
        // only the first word. Widening the column to cover it changed the STRIDE a
        // single-stream draw steps the guest's own buffer by (`packed_stride` - see
        // `read_draw_vertices`), so every vertex after the first was read from the wrong place.
        //
        // MEASURED on one title's gameplay G-buffer pass: its world programs declare
        // `Position_U` F32x4 at 0, `Normal_V` and `Tangent_BinormalScale` as UNTYPED x2 at 16
        // and 24, and `AmbientOcclusion_UV` (an F16x2 parameter) as UNTYPED x2 at 32 over a
        // stream of stride 36. The buffer is laid out at 36 - each record's tangent `w` (the
        // binormal sign, 1.0) recurs every 36 bytes, and record 1's position starts at byte 36
        // - while the draw stepped it by 40: sheared walls, tangent frames assembled from
        // neighbouring records, and over a third of the G-buffer's normals saturated to
        // `(+-1, +-1, +-1)`; re-stepped at 36 the replayed normals are unit length and the room,
        // door and axe come out in place. The overhang itself is served by the gather's row length (which already
        // covers every piece) and, on the single-stream path, by the renderer's repack, which
        // zero-fills an attribute the captured rows end inside.
        // `VITASLOP_GXP_STREAM_COLUMN_MAX=1` is the arm back to the widened column.
        let widen = crate::knobs::var("VITASLOP_GXP_STREAM_COLUMN_MAX").ok().as_deref() == Some("1");
        let used_streams: std::sync::Arc<[VertexStreamInfo]> = used_streams
            .iter()
            .enumerate()
            .map(|(i, s)| VertexStreamInfo {
                column: if s.stride == 0 || widen { s.stride.max(extent_of(i)) } else { s.stride },
                ..*s
            })
            .collect();
        // >>> EVERY STREAM'S COLUMN STARTS ON A FOUR-BYTE BOUNDARY, AND THE ROW ENDS ON ONE.
        //
        // The bases were the running sum of the columns, which is the tightest row possible and
        // the wrong one. WebGPU will not accept a vertex attribute whose offset is not a
        // multiple of `min(4, the format's size)` - four for every format wider than two bytes -
        // so a stream whose column is, say, THREE bytes puts every attribute of every later
        // stream at an offset three modulo four, and the whole row becomes unbindable. The
        // renderer then has to COPY it into a row it lays out itself, for every draw of every
        // frame, and the guest's own buffer - which is otherwise exactly the shape a vertex
        // buffer wants - is never bound.
        //
        // MEASURED on a baseball title's gameplay, from the vertex-plan census: **154 pipelines
        // of 185 copied rather than bound, and the first reason was `@location 1 sits at guest
        // offset 3, not a multiple of 4`.** The row is OURS - the gather tiles the guest's bound
        // streams into it - so the alignment is ours to give it.
        //
        // The STRIDE is aligned too, for the same reason from the other end: `array_stride` has
        // to be a multiple of four, and a row whose columns sum to an odd number is refused
        // whatever its offsets are.
        //
        // `VITASLOP_GXP_ALIGN_ROW=0` is the negative control - the tight row this replaces.
        // >>> ONLY WHERE THERE IS ACTUALLY A ROW OF OURS TO ALIGN.
        //
        // A draw with ONE per-vertex stream never gathers: `read_draw_vertices` binds the
        // guest's own buffer and steps it by this very `packed_stride`, so a value that is not
        // the stream's own stride would read every vertex from the wrong place. That row is the
        // GUEST's and its offsets are the guest's; only an interleaved row is ours to lay out.
        //
        // >>> AND THE COLUMN IS STILL NOT ENOUGH, WHICH IS WHAT `ScatterPiece` IS FOR.
        //
        // An aligned column does not align the attributes INSIDE it: a stream carrying `U8x3`
        // at 0 and anything at 3 puts its second attribute at 3 modulo 4 whatever base the
        // column is given, and one such attribute makes the whole pipeline unbindable. Under
        // `attr_packed_row` each attribute gets its own 4-aligned slot instead, and the gather
        // is described piece by piece rather than column by column. `VITASLOP_GXP_ATTR_ROW=0`
        // is that arm's negative control and falls back to the column form below.
        let one_stream = matches!(&*used_streams, [s] if !s.per_instance);
        let align_row = align_packed_row() && !one_stream;
        let attr_row = attr_packed_row() && align_row;
        // The byte extent one attribute occupies in its stream's row - the same arithmetic
        // `extent_of` above uses to size a column.
        let attr_bytes = |a: &crate::capture::VertexAttribute| -> u32 {
            let comp = match a.format {
                0 | 1 | 4 | 5 => 1u32,
                2 | 3 | 6 | 7 | 8 => 2,
                _ => 4,
            };
            comp * u32::from(a.component_count).max(1)
        };
        let mut base = Vec::with_capacity(used_streams.len());
        let mut pieces: Vec<ScatterPiece> = Vec::new();
        // Where each attribute ENDED UP, by its index in `attributes`. Empty under the column
        // form, where the offset is the guest's own plus the stream base.
        let mut moved: Vec<u32> = Vec::new();
        let mut packed_stride = 0u32;
        if attr_row {
            moved = vec![0u32; attributes.len()];
            let mut mine: Vec<usize> = Vec::new();
            for (si, st) in used_streams.iter().enumerate() {
                packed_stride = packed_stride.next_multiple_of(4);
                base.push(packed_stride);
                if st.column == 0 {
                    continue;
                }
                mine.clear();
                mine.extend(
                    (0..attributes.len()).filter(|&k| attributes[k].stream_index as usize == si),
                );
                // A stream nothing names still has a column (its stride), and it is copied
                // whole exactly as it always was: nothing fetches those bytes, but every cache
                // downstream HASHES the row, so dropping them would be a change of content for
                // no reason.
                if mine.is_empty() {
                    pieces.push(ScatterPiece {
                        stream: si as u16,
                        src: 0,
                        dst: packed_stride as u16,
                        len: st.column as u16,
                    });
                    packed_stride += st.column;
                    continue;
                }
                // In GUEST offset order, so a row stays in the order the title wrote it and a
                // diff of two layouts reads sensibly. Two attributes may OVERLAP in the guest's
                // row (an aliased colour and its alpha, say); each gets its own slot and its own
                // copy, which is what the fetch wants and what the column form did implicitly.
                mine.sort_by_key(|&k| attributes[k].offset);
                for &k in mine.iter() {
                    let a = attributes[k];
                    let len = attr_bytes(&a);
                    // The SLOT is the length rounded up to four, not the length: a 3-byte
                    // `U8x3` is fetched by WebGPU as `uint8x4`, which reads one byte PAST the
                    // three the guest has, and that read has to land inside the row.
                    let slot = len.next_multiple_of(4).max(4);
                    moved[k] = packed_stride;
                    // >>> THE COPY FILLS THE WHOLE SLOT WHEN THE GUEST'S ROW HAS THE BYTES.
                    //
                    // A 3-byte attribute in a 4-byte slot leaves a pad, and a pad is not free:
                    // nothing FETCHES it, but every cache downstream hashes the row, so it has
                    // to be DEFINED - which is a per-vertex fill, and it also breaks the run
                    // that `RowLayout::merged` would otherwise fold back into one copy. Taking
                    // the guest's own next byte instead defines it, costs nothing (it is inside
                    // the row already being read), and is exactly the byte a widened `uint8x4`
                    // fetch would have read - a lane a 3-component attribute never uses, since
                    // the linker bakes the surplus lane as a constant.
                    let copy = slot
                        .min(st.column.saturating_sub(u32::from(a.offset)))
                        .max(len);
                    pieces.push(ScatterPiece {
                        stream: si as u16,
                        src: a.offset,
                        dst: packed_stride as u16,
                        len: copy as u16,
                    });
                    packed_stride += slot;
                }
            }
            packed_stride = packed_stride.next_multiple_of(4).max(4);
        } else {
            for (si, s) in used_streams.iter().enumerate() {
                if align_row {
                    packed_stride = packed_stride.next_multiple_of(4);
                }
                base.push(packed_stride);
                if s.column != 0 {
                    pieces.push(ScatterPiece {
                        stream: si as u16,
                        src: 0,
                        dst: packed_stride as u16,
                        len: s.column as u16,
                    });
                }
                packed_stride += s.column;
            }
            if align_row {
                packed_stride = packed_stride.next_multiple_of(4);
            }
        }
        let layout = RowLayout::merged(pieces);
        // >>> AN ATTRIBUTE WHOSE STREAM HAS NO STRIDE CANNOT BE FETCHED, AND THE FAILURE IS
        // >>> SILENT AND CATASTROPHIC.
        //
        // `used_streams` fills a stream the create call did not describe with
        // `VertexStreamInfo::default()` - stride 0. The rebasing above still gives that
        // stream's attributes a `stream_base` (the running sum), so their offsets move into a
        // packed row that `packed_stride` never made room for: every fetch past the first
        // stream's stride reads the FOLLOWING vertices' bytes.
        //
        // MEASURED on a fighting title's instanced particle program (`Input.InstanceOffset`,
        // `InstanceXAxis/Y/Z`, `InstanceColor`, `InstanceSubUVParams` - six per-instance
        // attributes over two per-vertex ones): attribute offsets 28..92 rebased into a
        // `packed_stride` of 28, so alternate vertices read position (0,0,0) and the mesh
        // smeared into a full-screen woven lattice of stretched strips. That is the title's
        // long-standing "stray flat shapes"/grey slats.
        for a in attributes.iter() {
            let si = a.stream_index as usize;
            let stride = used_streams.get(si).map(|s| s.stride).unwrap_or(0);
            if stride == 0 {
                tracing::warn!(
                    target: "vitaslop::gxm",
                    stream = si,
                    reg = a.reg_index,
                    offset = a.offset,
                    streams_described = streams.len(),
                    streams_used = used,
                    packed_stride,
                    program = format_args!("{program_header:#x}"),
                    per_instance = used_streams.get(si).map(|s| s.per_instance).unwrap_or(false),
                    all_strides = format_args!("{:?}", used_streams.iter().map(|s| s.stride).collect::<Vec<_>>()),
                    all_instance = format_args!("{:?}", used_streams.iter().map(|s| s.per_instance).collect::<Vec<_>>()),
                    "a vertex attribute is sourced from a STREAM WITH NO STRIDE - the create call described {} stream(s) and the attributes reference {}. Its column is rebased into a packed row with no room for it, so every fetch past the first stream reads the NEXT vertices' bytes and the mesh comes out smeared",
                    streams.len(),
                    used
                );
                break;
            }
        }
        let stream_base: std::sync::Arc<[u32]> = base.into();
        let packed_attributes: std::sync::Arc<[crate::capture::VertexAttribute]> = attributes
            .iter()
            .enumerate()
            .map(|(k, a)| {
                let mut a = *a;
                // Under the per-attribute row the offset is the SLOT this attribute was given;
                // under the column form it is the guest's own plus its stream's base.
                a.offset = match moved.get(k) {
                    Some(&at) => at as u16,
                    None => {
                        a.offset
                            + stream_base
                                .get(a.stream_index as usize)
                                .copied()
                                .unwrap_or(0) as u16
                    }
                };
                a.stream_index = 0;
                a
            })
            .collect();
        let single_stream = matches!(&*used_streams, [s] if !s.per_instance);
        // >>> A RECYCLED HANDLE THAT CHANGES A PROGRAM'S LAYOUT IS A SILENT MESH CORRUPTION.
        //
        // The handle is a `galloc` block, and `galloc` REUSES a freed block before it moves the
        // bump cursor. So a title that releases a vertex program and creates another can be
        // handed the same address - and this map is keyed on it. Everything downstream (the
        // gather's strides, the packed offsets, the linker's attribute order) then describes
        // the LAST program created at that address, while the context may still have an older
        // one bound. The draw reads its own buffer through someone else's layout, which is
        // geometry smeared across the frame and nothing in any other panel to say why.
        if let Some(prev) = self.shader_patcher.vertex_program(handle)
            && (prev.attributes != attributes || prev.streams != streams)
        {
            tracing::warn!(
                target: "vitaslop::gxm",
                handle = format_args!("{handle:#x}"),
                was_header = format_args!("{:#x}", prev.program_header),
                now_header = format_args!("{program_header:#x}"),
                was_attrs = prev.attributes.len(),
                now_attrs = attributes.len(),
                was_strides = format_args!("{:?}", prev.streams.iter().map(|s| s.stride).collect::<Vec<_>>()),
                now_strides = format_args!("{:?}", streams.iter().map(|s| s.stride).collect::<Vec<_>>()),
                "a VERTEX PROGRAM HANDLE was reused for a program with a DIFFERENT vertex layout - every draw still holding the old one now fetches through the new one's strides and offsets"
            );
        }
        self.shader_patcher.set_vertex_program(
            handle,
            VertexProgramInfo {
                attributes,
                streams,
                program_header,
                packed_attributes,
                used_streams,
                stream_base,
                packed_stride,
                layout,
                single_stream,
            },
        );
    }

    /// The `SceGxmProgram*` a vertex program handle was created from, if recorded.
    pub(crate) fn vertex_program_header(&self, handle: u32) -> u32 {
        self.shader_patcher
            .vertex_program(handle)
            .map(|i| i.program_header)
            .unwrap_or(0)
    }

    /// Record `sceGxmShaderPatcherCreateFragmentProgram`'s handle -> (`SceGxmProgram*`, the
    /// blend equation it was created with).
    pub fn set_fragment_program(
        &mut self,
        handle: u32,
        program_header: u32,
        blend: crate::capture::BlendState,
        patched_vertex_header: u32,
    ) {
        self.shader_patcher
            .set_fragment_program(handle, program_header, blend, patched_vertex_header);
    }

    pub(crate) fn fragment_program_header(&self, handle: u32) -> u32 {
        self.shader_patcher
            .fragment_program(handle)
            .map(|(h, ..)| *h)
            .unwrap_or(0)
    }

    /// The vertex `SceGxmProgram*` a fragment program handle was PATCHED against at create
    /// (0 when the title passed NULL or the handle is unknown).
    pub(super) fn fragment_program_patched_vertex(&self, handle: u32) -> u32 {
        self.shader_patcher
            .fragment_program(handle)
            .map(|(_, _, v)| *v)
            .unwrap_or(0)
    }

    /// The blend equation a fragment program handle was created with. An unknown handle
    /// yields the GXM default (no blending), which is what a NULL `blendInfo` means.
    pub(super) fn fragment_program_blend(&self, handle: u32) -> crate::capture::BlendState {
        self.shader_patcher
            .fragment_program(handle)
            .map(|(_, b, _)| *b)
            .unwrap_or_default()
    }

    /// Record a color surface, keyed by its guest struct address - reporting, once per pair,
    /// any other surface whose pixel bytes it overlaps (see [`ColorSurfaces`]).
    #[inline]
    pub fn set_color_surface(&mut self, addr: u32, surface: crate::capture::ColorSurface) {
        self.color_surfaces.set(addr, surface);
    }

    // --- scene assembly (used by the gxm handlers) ---

    pub fn begin_scene(
        &mut self,
        ctx: &mut GuestCtx,
        color: Option<crate::capture::ColorSurface>,
        depth: Option<crate::capture::DepthSurface>,
        // The render target's `SceGxmMultisampleMode` - see `capture::Scene::multisample`.
        multisample: u32,
        // The render target's own extent - see `capture::Scene::target_extent`. `None` when the
        // target was created before capture began.
        target_extent: Option<(u32, u32)>,
    ) {
        // Texture snapshots deliberately SURVIVE the scene - see `TextureSnapshots`
        // for what invalidates them instead. Only the verifier is re-armed here - unless the
        // resolver proves the textures, which then re-arms it per scene it reads.
        if !defer_textures() {
            snaps!(self).begin_scene();
        }
        self.tex_scene_seq += 1;
        // A scene that never reached `sceGxmEndScene` leaves its descriptions behind, and they
        // would pair with the NEXT scene's draws - each one reading another draw's bindings.
        // Dropped here rather than carried: an abandoned scene has no draws to feed.
        self.deferred_geometry.clear();
        // Recycle the default-uniform arena for the new scene. Every draw's uniforms
        // are snapshotted into the draw at record time, so last scene's buffers are
        // dead by now; see [`Self::alloc_default_uniform_buffer`] for what happens
        // when this is NOT recycled.
        if self.gxm_context != 0 {
            crate::vita::gxmctx::rewind_uniform_ring(ctx, self.gxm_context);
        }
        self.scene = Some(crate::capture::Scene {
            completed_early: false,
            notifications: [None; 2],
            deferred_id: 0,
            color,
            depth,
            multisample,
            target_extent,
            draws: Vec::new(),
            precompile: Default::default(),
        });
        self.clear_sa_bank(ctx);
    }

    // --- The sticky GXM context state ---------------------------------------
    //
    // It lives in the guest's context block, not here - see [`crate::vita::gxmctx`] for
    // why. These are the read side: every one resolves through `self.gxm_context`, so
    // there is exactly one home for each fact and no host copy to fall out of step with
    // an inlined setter that never reaches the host at all.

    // --- Deferred contexts and command lists -----------------------------------
    //
    // A deferred context records draws into a command list that an immediate context later
    // splices into its own scene. Its sticky state lives in its own block, exactly like the
    // immediate one's (`gxmctx::init` lays it down in the title's hostMem), so every setter -
    // host or inlined - already lands in the right place. What does not is the handful of
    // host paths that act on "the" context (`self.gxm_context`): those are pointed at the
    // deferred block for the duration of any GXM call that names it (`vita::dispatch`).
    //
    // A recorded draw is its arguments plus a COPY of the deferred block at that moment, because
    // the list must replay the state each draw saw, not the state the context ended up in.
    // Replay ([`Self::replay_command_list`]) puts each copy back in the block and runs the
    // ordinary draw path against it, inside the immediate context's open scene.

    /// Whether `context` is a deferred context this run created.
    pub fn is_deferred_context(&self, context: u32) -> bool {
        !self.deferred_contexts.is_empty() && self.deferred_contexts.contains_key(&context)
    }

    /// Make `context` "the" GXM context, answering the one it replaces.
    pub fn swap_gxm_context(&mut self, context: u32) -> u32 {
        std::mem::replace(&mut self.gxm_context, context)
    }

    /// Whether a scene is open on the immediate context.
    pub fn scene_open(&self) -> bool {
        self.scene.is_some()
    }

    /// `sceGxmMidSceneFlush`'s split: end the open scene where it stands and open its
    /// CONTINUATION on the same colour surface, depth surface, render target and multisample
    /// mode. `false` when no scene is open.
    ///
    /// A mid-scene flush promises the title that the vertex work of every draw so far is DONE,
    /// so it may rewrite those draws' vertex memory. Here a draw's geometry is read at the
    /// guest's GPU wait or flip (see [`Self::resolve_deferred_geometry`]), and only an ENDED
    /// scene's draws are filed for that read - so the half already drawn is ended into the
    /// capture, where the caller resolves it before signalling, and the rest of the scene
    /// draws into a continuation of it.
    ///
    /// The split must not be visible in the picture. Colour is the surface's own memory either
    /// side of it. Depth/stencil is not, so the first half is made to STORE it and the
    /// continuation to LOAD it, whatever the surface's own force bits say - otherwise the
    /// second half would start from the background depth. And none of [`Self::begin_scene`]'s
    /// per-scene resets run: the default-uniform ring, the SA bank and the guest's viewport and
    /// region clip all belong to the ONE scene the title began, and are left exactly as its
    /// draws so far left them. Only the texture verifier is re-armed, as for any scene.
    pub fn split_open_scene(&mut self) -> bool {
        /// `zlsControl`'s FORCE_LOAD / FORCE_STORE bits - see `vita::gxm`'s `DS_FORCE_*_MASK`.
        const ZLS_FORCE_LOAD: u32 = 0x2;
        const ZLS_FORCE_STORE: u32 = 0x4;
        let Some(open) = self.scene.as_mut() else {
            return false;
        };
        let (color, mut depth, multisample, target_extent) = (
            open.color,
            open.depth,
            open.multisample,
            open.target_extent,
        );
        if let Some(d) = open.depth.as_mut() {
            d.zls_control |= ZLS_FORCE_STORE;
        }
        self.end_scene();
        if let Some(d) = depth.as_mut() {
            d.zls_control |= ZLS_FORCE_LOAD;
        }
        if !defer_textures() {
            snaps!(self).begin_scene();
        }
        self.tex_scene_seq += 1;
        self.scene = Some(crate::capture::Scene {
            completed_early: false,
            notifications: [None; 2],
            deferred_id: 0,
            color,
            depth,
            multisample,
            target_extent,
            draws: Vec::new(),
            precompile: Default::default(),
        });
        true
    }

    /// Record a draw made on deferred `context`: the call, its arguments, and the context's
    /// block as it stands. `false` when the context is not recording.
    pub fn record_deferred_draw(
        &mut self,
        ctx: &GuestCtx,
        context: u32,
        draw: DeferredDraw,
    ) -> bool {
        let block = ctx.read_bytes(context, crate::vita::gxmctx::BYTES as usize);
        let pdraw = self.pdraw_binding;
        match self
            .deferred_contexts
            .get_mut(&context)
            .and_then(|d| d.recording.as_mut())
        {
            Some(list) => {
                list.push(RecordedDraw {
                    context,
                    draw,
                    block,
                    pdraw,
                });
                true
            }
            None => false,
        }
    }

    /// Replay a recorded command list into the open scene: for each draw, put the block it was
    /// recorded against back (with the SCENE's viewport and region clip - a list does not carry
    /// them), point the draw path at it, draw, and put everything back as it was.
    pub fn replay_command_list(&mut self, ctx: &mut GuestCtx, id: u32) -> bool {
        use crate::vita::gxmctx::off;
        let Some(draws) = self.command_lists.get(id as usize).cloned() else {
            return false;
        };
        let immediate = self.gxm_context;
        // The scene's viewport and region clip, from VIEWPORT_ENABLE up to the visibility state.
        let scene_view = (immediate != 0).then(|| {
            ctx.read_bytes(
                immediate + off::VIEWPORT_ENABLE,
                (off::FRONT_VISIBILITY_TEST_ENABLE - off::VIEWPORT_ENABLE) as usize,
            )
        });
        for d in &draws {
            let live = ctx.read_bytes(d.context, d.block.len());
            ctx.write_bytes(d.context, &d.block);
            if let Some(v) = &scene_view {
                ctx.write_bytes(d.context + off::VIEWPORT_ENABLE, v);
            }
            let prev = self.swap_gxm_context(d.context);
            let pdraw = std::mem::replace(&mut self.pdraw_binding, d.pdraw);
            match d.draw {
                DeferredDraw::Indexed {
                    primitive,
                    index_format,
                    index_data,
                    index_count,
                    index_wrap,
                } => self.record_draw(
                    ctx,
                    primitive,
                    index_format,
                    index_data,
                    index_count,
                    index_wrap,
                ),
                DeferredDraw::Precomputed { precomputed } => {
                    self.draw_precomputed(ctx, precomputed)
                }
            }
            self.pdraw_binding = pdraw;
            self.swap_gxm_context(prev);
            ctx.write_bytes(d.context, &live);
        }
        true
    }

    /// Adopt the context `sceGxmCreateContext` just built.
    ///
    /// The capture keeps ONE scene and ONE draw stream, so a second context would
    /// interleave two guests' draws into one frame. That is a real limitation and it says
    /// so rather than quietly serving the newer context.
    pub fn adopt_gxm_context(&mut self, context: u32) {
        if self.gxm_context != 0 && self.gxm_context != context {
            tracing::warn!(
                target: "vitaslop::gxm",
                previous = format_args!("{:#x}", self.gxm_context),
                context = format_args!("{context:#x}"),
                "a SECOND sceGxmContext was created - the capture has one scene and one draw \
                 stream, so draws from both will interleave into one frame"
            );
        }
        self.gxm_context = context;
    }

    /// The bound `SceGxmVertexProgram *` handle.
    pub fn bound_vertex_program(&self, ctx: &GuestCtx) -> u32 {
        self.gxm_state_word(ctx, crate::vita::gxmctx::off::VERTEX_PROGRAM)
    }

    /// The `SceGxmProgram *` header of the bound FRAGMENT program, resolved from the
    /// handle in the context block. 0 when nothing usable is bound.
    ///
    /// Resolved on read rather than at bind time: the bind is a store the transpiler can
    /// inline, and a lookup that happens once per draw instead of once per bind is cheaper
    /// anyway on a title that rebinds the same program repeatedly.
    pub fn bound_fragment_program_header(&self, ctx: &GuestCtx) -> u32 {
        self.fragment_program_header(self.bound_fragment_program(ctx))
    }

    /// The blend equation of the bound fragment program. Baked in at
    /// `sceGxmShaderPatcherCreateFragmentProgram`, so it is a pure function of the handle.
    pub fn bound_fragment_blend(&self, ctx: &GuestCtx) -> crate::capture::BlendState {
        self.fragment_program_blend(self.bound_fragment_program(ctx))
    }

    /// The bound `SceGxmFragmentProgram *` handle.
    pub(super) fn bound_fragment_program(&self, ctx: &GuestCtx) -> u32 {
        self.gxm_state_word(ctx, crate::vita::gxmctx::off::FRAGMENT_PROGRAM)
    }
}
