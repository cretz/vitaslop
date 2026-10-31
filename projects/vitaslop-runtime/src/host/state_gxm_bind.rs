use super::*;

impl VitaState {

    /// The guest vertex buffer bound to each stream index by `sceGxmSetVertexStream`.
    /// GXM allows up to [`MAX_VERTEX_STREAMS`]; a mesh that splits per-vertex data from
    /// per-instance data (particles, decals, instanced props) uses more than one, and
    /// capturing only stream 0 decodes those attributes out of the wrong buffer.
    pub fn bound_streams(&self, ctx: &GuestCtx) -> [u32; MAX_VERTEX_STREAMS] {
        match self.gxm_context {
            0 => [0; MAX_VERTEX_STREAMS],
            c => crate::vita::gxmctx::streams(ctx, c),
        }
    }

    /// The live GXM fixed-function pipeline state (cull/depth/stencil/viewport/...),
    /// snapshotted into each recorded draw. Sticky across scenes, exactly like the real
    /// GXM context - because it IS the real GXM context.
    pub fn render_state(&self, ctx: &GuestCtx) -> crate::capture::RenderState {
        match self.gxm_context {
            0 => crate::capture::RenderState::default(),
            c => crate::vita::gxmctx::load(ctx, c),
        }
    }

    /// One word of the context block, or 0 when there is no context.
    pub(super) fn gxm_state_word(&self, ctx: &GuestCtx, offset: u32) -> u32 {
        match self.gxm_context {
            0 => 0,
            c => crate::vita::gxmctx::get(ctx, c, offset),
        }
    }

    /// Write one word of the context block. The precomputed paths use this: they replay a
    /// prebuilt state object as the equivalent sequence of setters, so what they change is
    /// the same context state a `sceGxmSet*` would.
    pub(super) fn set_gxm_state_word(&self, ctx: &mut GuestCtx, offset: u32, value: u32) {
        if self.gxm_context != 0 {
            crate::vita::gxmctx::set(ctx, self.gxm_context, offset, value);
        }
    }

    /// Record `sceGxmSetFragmentTexture(context, unit, texture)`: copy the texture's control
    /// words into sampler `unit`'s slot of the context block. A zero address unbinds it.
    ///
    /// This is the FALLBACK half of the call. The transpiler emits the copy inline for the
    /// ordinary case (`InlineOp::CopyArgIndexed`), so what reaches here is a null texture, an
    /// out-of-range unit, or a pointer outside guest memory - the cases the inline form hands
    /// back precisely because this side defines them.
    ///
    /// Decode a list of texture bindings into snapshotted [`crate::capture::BoundTexture`]s.
    ///
    /// Shared by the fragment and vertex stages so the two can never decode a texture
    /// differently: the control words, the recorded exact format, the sampler state and the
    /// nearby-format search all mean the same thing whichever stage bound it.
    ///
    /// Takes the bindings as a SLICE the caller has moved out of `self` (`mem::take`, put
    /// back after), because the per-unit control state has to be read through a borrow of
    /// `self` while the snapshot cache is borrowed mutably for the decode. It used to take
    /// them by value, which cloned both binding lists on EVERY draw - hundreds of times a
    /// frame - for lists the caller owns and does not mutate here.
    pub(super) fn snapshot_bound_textures(
        &mut self,
        ctx: &GuestCtx,
        bindings: &[TextureBinding],
    ) -> Vec<crate::capture::BoundTexture> {
        // >>> THE NEARBY-FORMAT SEARCH IS FOR NULL HANDLES ONLY, AND IT IS MEMOISED.
        //
        // It is read in exactly one place: `decode_texture` hands it to
        // `report_zero_texture_handle` inside the `binding.is_null()` arm, and nowhere
        // else. It used to be computed whenever the address had no RECORDED format, which
        // is the ordinary case for a texture bound by VALUE
        // [[vitaslop-texture-binding-by-value]] rather than an error - so it ran for most
        // units of most draws. And it is a LINEAR SCAN of every texture the title has ever
        // initialised.
        //
        // MEASURED with `bench --at` on a live race: "draw: snapshot textures" was 46% of
        // the entire frame, 17.9 us per draw, while moving 0.0 MB - a diagnostic costing
        // half the frame rate and producing no data. Both engines paid it; this is shared
        // runtime code.
        //
        // Memoised because a null handle rebound every frame would otherwise pay the scan
        // every frame. The report fires once per (unit, address), so an answer that later
        // goes stale can only change a message that will never be printed again.
        for b in bindings {
            if b.is_null() {
                self.textures.memo_nearby(b.addr, 4096);
            }
        }
        // Only the recorded FORMAT and the nearby-handle diagnostic still come from host state.
        // The sampler wrap modes, filters, LOD bias and gamma now come out of the binding's own
        // control words, which `decode_texture` already holds - so they cannot go stale, and they
        // follow a by-value copy of the struct the way the hardware's do.
        // Taken out and put back rather than allocated: this runs on every draw, and the buffer
        // it needs is the same one every time. See `texture_unit_scratch`.
        let mut unit_state = std::mem::take(&mut self.texture_unit_scratch);
        unit_state.clear();
        unit_state.extend(bindings.iter().map(|&b| {
            let format = self.texture_format(b.addr);
            // The explicit row pitch `sceGxmTextureInitLinearStrided` was given - see
            // `build_texture_template`.
            let byte_stride = Self::linear_strided_stride_of(b.words);
            // Only for a handle with no format of its own: is there an initialised texture
            // NEARBY? A struct the guest inits at one address and binds at another (off by a
            // fixed member offset, or copied by value) is a completely different bug from one
            // it never initialised at all, and the two are indistinguishable from the zero
            // control words alone. Searching a window answers it in the run that hit it.
            // Resolved in the pre-pass above, for null handles only - see it for why.
            let nearby = self.textures.nearby(b.addr);
            (b, format, nearby, byte_stride)
        }));
        let yuv_profiles = self.yuv_profiles;
        let mut snapshots_guard = snaps!(self);
        let snapshots = &mut *snapshots_guard;
        let out = unit_state
            .iter()
            .filter_map(|(binding, format, nearby, byte_stride)| {
                decode_texture(ctx, snapshots, binding, *format, *nearby, *byte_stride)
            })
            .map(|mut t| {
                // The YUV conversion slot the texture names (swizzle bit 13 for the two-plane
                // formats: `YUV_CSC1`/`YVU_CSC1`) takes the profile the context has for it NOW.
                // Per draw, outside the memoised decode, because it is context state.
                if t.base_format == crate::render::YUV420P2 {
                    t.yuv_profile = yuv_profiles[((t.swizzle >> 13) & 1) as usize];
                }
                t
            })
            .collect();
        self.texture_unit_scratch = unit_state;
        out
    }

    /// `_sceGxmSetVertexTexture`: bind a texture to a VERTEX-stage sampler unit.
    ///
    /// Kept in its own array rather than merged with the fragment units, because the two stages
    /// have independent unit numbering and a shader can sample the same unit number in both
    /// with different textures. The array is in the CONTEXT BLOCK
    /// ([`crate::vita::gxmctx::off::VERTEX_TEXTURES`]), where the inlined precomputed-state bind
    /// also writes it; the draw reads it back through [`Self::sync_vertex_textures`].
    pub fn bind_vertex_texture(&mut self, ctx: &mut GuestCtx, unit: u32, texture_addr: u32) {
        let context = self.gxm_context;
        if context == 0 {
            self.report_bind_without_context();
            return;
        }
        let b = TextureBinding::read(ctx, unit, texture_addr, false);
        crate::vita::gxmctx::set_vertex_texture_binding(
            ctx,
            context,
            unit,
            crate::vita::gxmctx::TexBinding {
                addr: b.addr,
                words: b.words,
                from_precomputed: false,
            },
        );
    }

    /// Bring [`Self::bound_vertex_textures`] up to date with the context block's vertex sampler
    /// array, as this draw's snapshot `blk` holds it.
    ///
    /// The list and [`Self::vertex_texture_gen`] are a host-side CACHE of that array now: the
    /// array has two writers (`sceGxmSetVertexTexture` and the precomputed vertex-state bind,
    /// usually inlined), and neither can bump a host counter. So the draw compares the array's
    /// bytes against the copy it last parsed - a 384-byte `memcmp` - and re-parses and bumps the
    /// generation only on a real change, which keeps every generation-keyed shortcut downstream
    /// exactly as cheap as it was when the list had one host mutator.
    pub(super) fn sync_vertex_textures(&mut self, blk: &crate::vita::gxmctx::Block) {
        use crate::vita::gxmctx::{off, TEXTURE_ARRAY_BYTES};
        let span = blk.span(off::VERTEX_TEXTURES, TEXTURE_ARRAY_BYTES as usize);
        if self.vertex_texture_span[..] == *span {
            return;
        }
        self.vertex_texture_span.copy_from_slice(span);
        let mut parsed = Vec::new();
        crate::vita::gxmctx::texture_bindings_in_span(span, &mut parsed);
        self.bound_vertex_textures.clear();
        self.bound_vertex_textures.extend(parsed.into_iter().map(|(unit, b)| TextureBinding {
            unit,
            addr: b.addr,
            words: b.words,
            from_precomputed: b.from_precomputed,
        }));
        self.vertex_texture_gen += 1;
    }

    pub fn bind_fragment_texture(&mut self, ctx: &mut GuestCtx, unit: u32, texture_addr: u32) {
        // Into the CONTEXT BLOCK, which is where the inlined form of this call writes. The
        // handler and the emitted code must leave byte-identical state, because a draw reads
        // one array and cannot tell which path filled a slot - and after 1,275 binds a frame
        // go inline, this handler runs only for the cases the inline form declines (a null
        // texture, an out-of-range unit), which are exactly the ones a divergence would hide
        // in. See `crate::vita::gxmctx::TexBinding`.
        let context = self.gxm_context;
        if context == 0 {
            self.report_bind_without_context();
            return;
        }
        let words = if texture_addr == 0 {
            [0; 4]
        } else {
            [
                ctx.read_u32(texture_addr),
                ctx.read_u32(texture_addr + 4),
                ctx.read_u32(texture_addr + 8),
                ctx.read_u32(texture_addr + 12),
            ]
        };
        self.note_texture_bind(unit, texture_addr);
        crate::vita::gxmctx::set_texture_binding(
            ctx,
            context,
            unit,
            crate::vita::gxmctx::TexBinding {
                addr: texture_addr,
                words,
                from_precomputed: false,
            },
        );
    }

    /// Report - once - a texture bind that arrived before `sceGxmCreateContext`, so its state
    /// has nowhere to live.
    ///
    /// Silently dropping it would be a draw missing a texture thousands of frames later, with
    /// nothing pointing back at the bind. This cannot happen through a conforming title (GXM
    /// takes the context as its first argument), which is why it is a report and not a case.
    pub(super) fn report_bind_without_context(&mut self) {
        if !self.reported_no_gxm_context {
            self.reported_no_gxm_context = true;
            tracing::warn!(
                target: "vitaslop::gxm",
                "a texture was bound before any sceGxmContext was created, so the binding has \
                 nowhere to live and is DROPPED"
            );
        }
    }

    /// Note the context pointer a host-routed GXM call was handed, for [`TextureSlotWrites`].
    pub fn note_bind_context(&mut self, context: u32) {
        self.texture_writes.last_bind_context = context;
    }

    /// Note a host-routed `sceGxmSetFragmentTexture`, for [`TextureSlotWrites`].
    pub(super) fn note_texture_bind(&mut self, unit: u32, handle: u32) {
        if handle == 0 {
            self.texture_writes.unbinds += 1;
        } else {
            self.texture_writes.binds += 1;
            self.texture_writes.last_bind = Some((self.cur_frame, unit, handle));
        }
    }

    /// Report - ONCE - a draw whose fragment program DECLARES samplers and whose texture
    /// binding list came out EMPTY.
    ///
    /// # Why this exists, and why it dumps the block rather than naming a cause
    /// Downstream this draw produces `gxp pair ...: sampler unit N wants Two but the bound
    /// units are []`, and the recompiler then REFUSES it. That message is the renderer's view:
    /// it says the list is empty and cannot say why. The two candidate causes are opposite
    /// fixes - the narrowing mask in `record_draw` dropped the unit, or nothing ever wrote the
    /// context block's slot - and both were guessed at, in that order, before either was
    /// looked at. `VITASLOP_SAMPLER_NARROW=0` (the mask arm) and `VITASLOP_NO_INLINE_TEXTURE=1`
    /// (the inline-writer arm) each cleared their own suspect while the draw kept failing, so
    /// the remaining question is only answerable by the BYTES.
    ///
    /// So this prints what the block actually holds at the failing draw: every non-zero slot,
    /// and the count of zero ones. A block that is entirely zero says no writer ran (or one
    /// WIPED it - `bind_precomputed_fragment_state` copies a state's array wholesale, and a
    /// state whose block pointer is null writes zeros over all sixteen units). A block with the
    /// texture in a DIFFERENT unit than the program declares says the two disagree about
    /// numbering, which is a different bug with a different fix.
    ///
    /// Fires once for the run: the failing pair repeats every frame, and a per-draw line would
    /// bury the rest of the log [[vitaslop-a-diagnostic-can-bury-the-findings]].
    pub(super) fn report_draw_without_bound_textures(
        &mut self,
        blk: &crate::vita::gxmctx::Block,
        fheader: u32,
        want: u32,
    ) {
        use crate::vita::gxmctx::{self, MAX_TEXTURE_UNITS, TEXTURE_STRIDE, off};
        if self.reported_draw_without_textures {
            return;
        }
        self.reported_draw_without_textures = true;
        let mut bound = String::new();
        let mut zero = 0u32;
        for unit in 0..MAX_TEXTURE_UNITS {
            let at = off::TEXTURES + unit as u32 * TEXTURE_STRIDE;
            let addr = blk.word(at);
            if addr == 0 {
                zero += 1;
                continue;
            }
            let pre = blk.word(at + 4 + gxmctx::TEXTURE_CONTROL_WORDS * 4) != 0;
            bound.push_str(&format!(
                " [unit {unit}: handle {addr:#x} w0 {:#010x} w2 {:#010x}{}]",
                blk.word(at + 4),
                blk.word(at + 12),
                if pre { " from-precomputed" } else { "" }
            ));
        }
        if bound.is_empty() {
            bound.push_str(" NONE - all sixteen slots are zero");
        }
        // The SLOT ADDRESSES, so the next question is answerable with an existing instrument
        // rather than another new one: the context block is guest memory, every writer of it
        // (inlined or host-routed) goes through a guest store, and
        // `VITASLOP_WATCH_STORE=<addr> VITASLOP_WATCH_STORE_LOG=1` therefore enumerates them
        // all in one run [[vitaslop-watch-store-log]]. An empty slot cannot be traced without
        // its address, and the address is only in hand here.
        let unit0 = self.gxm_context.wrapping_add(off::TEXTURES);
        eprintln!(
            "gxm texture: frame {}: this draw came through {}. Fragment program {fheader:#x} DECLARES sampler units \
             {want:#06x}, but this draw's context block binds:{bound} ({zero} of {} slots \
             zero). The draw reaches the renderer with no texture, and a recompiled pair that \
             samples one of those units cannot be built. Narrowing is {}. The context block is \
             at {:#x} and unit 0's slot at {unit0:#x} - watch that address to enumerate every \
             writer (VITASLOP_WATCH_STORE={unit0:#x} VITASLOP_WATCH_STORE_LOG=1). Host-routed \
             writers so far: {} binds, {} unbinds, {} precomputed-state binds of which {} \
             landed an array with unit 0 EMPTY (so they CLEARED it); last direct bind {}. \
             This title has put a texture INTO a precomputed state {} times (a count that is \
             always complete - those setters are never inlined). \
             {}",
            self.cur_frame,
            if self.last_draw_was_precomputed {
                "sceGxmDrawPrecomputed"
            } else {
                "sceGxmDraw"
            },
            MAX_TEXTURE_UNITS,
            if sampler_narrow_enabled() {
                "ON"
            } else {
                "OFF (VITASLOP_SAMPLER_NARROW=0)"
            },
            self.gxm_context,
            self.texture_writes.binds,
            self.texture_writes.unbinds,
            self.texture_writes.state_binds,
            self.texture_writes.state_binds_clearing_unit0,
            match self.texture_writes.last_bind {
                Some((f, u, h)) => format!(
                    "was unit {u} handle {h:#x} at frame {f}, through context {:#x}{}",
                    self.texture_writes.last_bind_context,
                    if self.texture_writes.last_bind_context == self.gxm_context {
                        " (the SAME block this draw reads)"
                    } else {
                        " >>> WHICH IS NOT THE BLOCK THIS DRAW READS: the inlined form writes \
                         through the pointer the guest passes and the handler writes through the \
                         adopted context, so the two paths are writing DIFFERENT memory"
                    }
                ),
                None => "NEVER HAPPENED".to_string(),
            },
            self.texture_writes.state_texture_sets,
            match crate::vita::texture_slot_writers_are_host_routed() {
                (true, true) =>
                    "Both writers are host-routed in this run, so both counts are \
                                 COMPLETE.",
                (true, false) =>
                    "The direct-bind count is COMPLETE, but an INLINED \
                                  sceGxmSetPrecomputedFragmentState replaces all sixteen slots \
                                  without reaching the host - so a precomputed-state count of 0 \
                                  here means UNCOUNTED, not none. Re-run with \
                                  VITASLOP_NO_INLINE_IMPORTS=1 for both.",
                (false, true) =>
                    "The precomputed-state count is COMPLETE; the direct-bind count \
                                  is a lower bound (the inlined copy form does not reach the \
                                  host).",
                (false, false) =>
                    "BOTH counts are lower bounds: the inlined copy form and the \
                                   inlined state bind each write the slots without reaching the \
                                   host. Re-run with VITASLOP_NO_INLINE_IMPORTS=1.",
            },
        );
    }

    /// Record the exact `SceGxmTextureFormat` set on a `SceGxmTexture*` (by
    /// `sceGxmTextureInit*`/`SetFormat`), so a later decode recovers the exact
    /// channel swizzle rather than the lossy 3-bit control-word field.
    pub fn set_texture_format(&mut self, texture_addr: u32, format: u32) {
        let changed = self.textures.set_format(texture_addr, format);
        TEXTURE_INITS.fetch_add(1, std::sync::atomic::Ordering::Relaxed);
        // The per-binding memo derives its answer partly from this map
        // ([`TextureSnapshots::templates`]), so a recorded format that CHANGES invalidates it.
        // Dropped wholesale rather than by key: the memo is keyed on control words and this is
        // keyed on an address, so there is no key to remove - and a texture re-initialised at a
        // live address is a load-screen event, not a per-frame one.
        if changed {
            snaps!(self).templates.clear();
            // ...and the per-scene lists BUILT from those templates, for the same reason. A set
            // cached earlier in this scene describes the texture as it was before the guest
            // re-initialised it, and the old code re-derived it on the very next draw.
            snaps!(self).snapshot_sets.clear();
            snaps!(self).decoded.clear();
        }
    }

    #[inline]
    pub fn texture_format(&self, texture_addr: u32) -> Option<u32> {
        self.textures.format(texture_addr)
    }

    /// The initialised texture nearest `addr` within `+-window` bytes, as `(signed byte delta,
    /// its format)`. Diagnostic only.
    ///
    /// A `sceGxmSetFragmentTexture` of an address we have no format for is ambiguous: the guest
    /// may never have initialised a texture there, or it may have initialised one a few bytes
    /// away and bound an interior pointer or a by-value copy. Those need opposite fixes and look
    /// identical at the binding, so the answer has to come from the neighbourhood.
    #[inline]
    pub fn nearest_recorded_texture(&self, addr: u32, window: u32) -> Option<(i64, u32)> {
        self.textures.nearest(addr, window)
    }

    /// The color surface recorded for `addr` (its `SceGxmColorSurface*` struct
    /// address), if the guest initialized one there.
    #[inline]
    pub fn color_surface(&self, addr: u32) -> Option<crate::capture::ColorSurface> {
        self.color_surfaces.get(addr)
    }

    /// The explicit byte stride of the `LINEAR_STRIDED` texture at `texture_addr`, or 0 for one
    /// that is not strided (derive the pitch from width x bytes-per-pixel).
    ///
    /// Read out of the struct's own word 0, where init packed it (`vita::gxm::pack_linear_stride`),
    /// so a by-value copy - a material holding its own `SceGxmTexture` - samples at the same
    /// pitch. Held beside the struct instead, a copy sampled a render target at width x 4 instead
    /// of its real pitch, every row shifting by the difference into slanted stripes.
    pub fn linear_strided_stride(&self, ctx: &GuestCtx, texture_addr: u32) -> u32 {
        let words = [0, 4, 8, 12].map(|o| ctx.read_u32(texture_addr.wrapping_add(o)));
        Self::linear_strided_stride_of(words)
    }

    /// [`Self::linear_strided_stride`] for a binding whose control words were already read (at
    /// bind time - the struct may since have been reused, so its memory is not re-read).
    pub(super) fn linear_strided_stride_of(words: [u32; 4]) -> u32 {
        if (words[1] >> 29) & 0x7 != crate::vita::gxm::TYPE_LINEAR_STRIDED {
            return 0;
        }
        crate::vita::gxm::linear_stride_from_word0(words[0])
    }

    /// Report the first `sceGxmTextureSetGammaMode` of a run.
    ///
    /// The mode itself is stored in the guest's control word 0 by the handler; this only says so
    /// once, because a gamma texture is sampled through an sRGB format and that is a real change
    /// in how its texels are decoded - the counterpart of [`Self::set_color_surface_gamma`].
    pub fn note_texture_gamma(&mut self, texture_addr: u32, gamma: u32) {
        if gamma != 0 {
            static SAID: std::sync::atomic::AtomicBool = std::sync::atomic::AtomicBool::new(false);
            if !SAID.swap(true, std::sync::atomic::Ordering::Relaxed) {
                tracing::info!(
                    target: "vitaslop::status",
                    "gxm texture: sceGxmTextureSetGammaMode({texture_addr:#x}, {gamma:#x}) - \
                     this texture is sampled through an sRGB format, so its texels are decoded \
                     on fetch as the hardware does"
                );
            }
        }
    }

    /// Record a color surface's gamma-correction mode (`sceGxmColorSurfaceSetGammaMode`).
    ///
    /// On hardware a non-`NONE` mode makes the ROP sRGB-ENCODE on write, so a shader writing
    /// linear values lands gamma-encoded in memory and every later pass reads them that way.
    /// The mode reaches the renderer through the scene's target (`RttTarget::gamma`), which
    /// renders such a surface through an sRGB VIEW of the same texture - the encode then
    /// happens after blending, exactly where the hardware does it.
    pub fn set_color_surface_gamma(&mut self, surface_addr: u32, data_addr: u32, gamma: u32) {
        // Once per (surface, mode) for the whole run: a title sets this every frame, and
        // 20,000 copies of one fact bury every other line in the log.
        if gamma != 0
            && crate::rtt_writeback::report_once(
                0x6a00_0000_0000_0000 ^ ((surface_addr as u64) << 16) ^ gamma as u64,
            )
        {
            {
                tracing::info!(
                    target: "vitaslop::status",
                    "gxm surface: sceGxmColorSurfaceSetGammaMode({surface_addr:#x}, {gamma:#x}) - \
                     writes to this surface are sRGB-encoded after blending, as the hardware \
                     does; the renderer reports which view it actually rendered through."
                );
            }
        }
        self.color_surfaces.set_gamma(surface_addr, data_addr, gamma);
    }

    /// The `SceGxmColorSurfaceGammaMode` recorded for the surface struct at `addr`, or 0.
    ///
    /// The struct pointer is tried first and answers exactly. Only when it is unknown does the
    /// buffer address decide - see [`ColorSurfaces`] for why a title reaches this path at all
    /// and why the order matters.
    #[inline]
    pub fn color_surface_gamma_mode(&self, addr: u32, data_addr: u32) -> u32 {
        self.color_surfaces.gamma_mode(addr, data_addr)
    }

    /// Record the clip rectangle `sceGxmColorSurfaceSetClip` set on the surface at `surface`.
    #[inline]
    pub fn set_color_surface_clip(&mut self, surface: u32, rect: (u32, u32, u32, u32)) {
        self.color_surfaces.set_clip(surface, rect);
    }

    /// The clip rectangle last set on the surface at `surface`, if one was.
    #[inline]
    pub fn color_surface_clip(&self, surface: u32) -> Option<(u32, u32, u32, u32)> {
        self.color_surfaces.clip(surface)
    }

    /// Record the dither mode `sceGxmColorSurfaceSetDitherMode` set on the surface at `surface`.
    #[inline]
    pub fn set_color_surface_dither(&mut self, surface: u32, mode: u32) {
        self.color_surfaces.set_dither(surface, mode);
    }

    /// The dither mode last set on the surface at `surface`, 0 (disabled) if none was.
    #[inline]
    pub fn color_surface_dither(&self, surface: u32) -> u32 {
        self.color_surfaces.dither(surface)
    }

    /// The GPU notification region, allocating it on first use. Returns a guest
    /// pointer to `SCE_GXM_NOTIFICATION_COUNT` (512) u32 slots.
    pub fn notification_region(&mut self) -> u32 {
        if self.notification_region == 0 {
            self.notification_region = self.galloc(512 * 4, 16);
        }
        self.notification_region
    }

    // --- GPU memory mappings ------------------------------------------------
    //
    // Mapping is a no-op here (the guest's pages already ARE the memory the capture
    // reads), but UNmapping is not: after it the guest may hand those pages to its own
    // allocator, and anything we cached against those addresses is then a snapshot of
    // somebody else's data. GXM gives the size only on the map call, so the range has to
    // be remembered to be able to invalidate on the unmap.

    /// `sceGxmMapMemory(base, size, attr)` / `sceGxmMapVertexUsseMemory` /
    /// `sceGxmMapFragmentUsseMemory`: remember the range.
    /// Is `addr` inside a GXM mapping the guest still holds?
    ///
    /// >>> A DEFERRED WRITE INTO GUEST MEMORY HAS TO ASK THIS, AND IT DID NOT.
    ///
    /// The browser's render-target write-back is ASYNCHRONOUS - the GPU copy is encoded on one
    /// frame and the mapped bytes land one or two frames later - so by the time the pixels are
    /// written the guest may have unmapped that memory and its allocator may have handed the
    /// block to something that is not an image. Writing pixels over it then corrupts whatever
    /// now lives there, and the failure arrives much later and looks nothing like a renderer
    /// bug: a football title dies at frame 1263 with `pc=0x00000000`, a call through a pointer that
    /// used to be a pointer. The desktop writes back SYNCHRONOUSLY inside the same frame and
    /// does not have the window, which is why the same run survives there.
    ///
    /// `sceGxmUnmapMemory` is the guest saying that memory is no longer GPU memory, so this is
    /// the guest's own statement about the destination rather than a guess about its allocator.
    pub fn gxm_memory_is_mapped(&self, addr: u32) -> bool {
        self.gxm_mappings
            .range(..=addr)
            .next_back()
            .is_some_and(|(base, size)| addr < base.saturating_add(*size))
    }

    pub fn gxm_map(&mut self, base: u32, size: u32) {
        if base != 0 && size != 0 {
            self.gxm_mappings.insert(base, size);
        }
    }

    /// `sceGxmUnmapMemory(base)` and the two USSE variants. Drops the mapping and every
    /// texture snapshot taken from it. An unmap of a base that was never mapped is the
    /// guest's error, not ours, and is reported rather than ignored.
    pub fn gxm_unmap(&mut self, base: u32) -> i32 {
        match self.gxm_mappings.remove(&base) {
            Some(size) => {
                snaps!(self).invalidate_range(base, size as usize);
                0
            }
            None => {
                tracing::warn!(
                    target: "vitaslop::gxm",
                    base = format_args!("{base:#x}"),
                    "gxmUnmap of an address that was never mapped"
                );
                // SCE_GXM_ERROR_INVALID_VALUE.
                0x8021_0000u32 as i32
            }
        }
    }

    // --- Occlusion queries --------------------------------------------------

    /// `sceGxmSetVisibilityBuffer(context, bufferBase, stridePerCore)`.
    #[inline]
    pub fn set_visibility_buffer(&mut self, base: u32, stride_per_core: u32) {
        self.visibility.set_buffer(base, stride_per_core)
    }

    /// `sceGxmSetBackVisibilityTest{Enable,Index,Op}`: one field (0 enable, 1 index, 2 op) of
    /// `context`'s back-face test, which lives on the host - see
    /// `vita::gxm::set_back_visibility_test`.
    #[inline]
    pub(crate) fn set_back_visibility_test(&mut self, context: u32, field: usize, value: u32) {
        self.visibility.set_back_test(context, field, value)
    }

    /// Called once per recorded draw: if the front-face visibility test is enabled,
    /// add this draw's vertex count to the slot it names.
    ///
    /// The real GPU counts SAMPLES THAT PASSED the depth test. Nothing here rasterizes
    /// at capture time, so the count is the drawn vertex count instead: a real,
    /// deterministic quantity that rises with the amount of geometry submitted under
    /// the query, and is nonzero exactly when the guest drew something. What it does NOT
    /// model is occlusion - geometry behind other geometry still counts - so a title
    /// using the query to hide a sun flare behind scenery will show the flare. That is
    /// an approximation, so it says so, once.
    ///
    /// # Why this is NOT fixed with a real GPU occlusion query, deliberately
    /// WebGPU has occlusion queries and wiring one up is a morning's work. It is the wrong
    /// move, and the reason is worth stating so nobody spends that morning:
    ///
    /// - The result would enter GUEST MEMORY. The visibility buffer is read by guest code,
    ///   which branches on it, so the value decides what the title does next - not merely how
    ///   it looks. Feeding it a GPU-measured number makes guest execution depend on the GPU.
    /// - Two engines would then diverge on purpose. The desktop oracle and the browser use
    ///   different adapters with different rasterisation at the sample level, so the same
    ///   frame could take different branches on each, and the pixel oracle would stop being an
    ///   oracle for anything downstream of a query.
    /// - It is also ASYNCHRONOUS. A query result is available a frame or more after the pass
    ///   that produced it, while the guest reads the buffer when the scene ends. Serving it
    ///   would mean either stalling on the GPU every scene or handing the guest last frame's
    ///   answer - and the second is a different wrong number, arrived at more expensively.
    ///
    /// So the approximation stays, and the trade is: a flare that should be hidden is visible
    /// (a bounded, visible, reported error) instead of a title whose control flow depends on
    /// which GPU is present (an unbounded, invisible, unreproducible one). Change this only
    /// with a determinism story, not because the warning is annoying.
    pub(super) fn accumulate_visibility(&mut self, blk: &crate::vita::gxmctx::Block<'_>, index_count: u32) {
        use crate::vita::gxmctx::off;
        let front = (
            blk.word(off::FRONT_VISIBILITY_TEST_ENABLE) != 0,
            blk.word(off::FRONT_VISIBILITY_TEST_INDEX),
        );
        self.visibility
            .accumulate(self.gxm_context, front, index_count);
    }

    /// Flush the scene's accumulated occlusion counts into the guest's visibility
    /// buffer, which is when the GPU would have written them (at scene end).
    #[inline]
    pub fn flush_visibility(&mut self, ctx: &mut GuestCtx) {
        self.visibility.flush(ctx)
    }

    // --- Render target driver memblock --------------------------------------

    /// Remember the `driverMemBlock` UID from a render target's params so
    /// `sceGxmRenderTargetGetDriverMemBlock` returns exactly what the guest supplied.
    #[inline]
    pub fn set_render_target_mem_block(&mut self, render_target: u32, mem_block: u32) {
        self.render_targets.set_mem_block(render_target, mem_block)
    }

    /// The `driverMemBlock` of `render_target`, or `SCE_UID_INVALID_UID` (-1) when the
    /// target was created asking sceGxm to allocate its own - which is exactly what the
    /// guest passed in that case.
    #[inline]
    pub fn render_target_mem_block(&self, render_target: u32) -> u32 {
        self.render_targets.mem_block(render_target)
    }

    /// Remember the extent a render target was created with
    /// (`SceGxmRenderTargetParams::width`/`height`).
    #[inline]
    pub fn set_render_target_extent(&mut self, render_target: u32, width: u32, height: u32) {
        self.render_targets.set_extent(render_target, width, height)
    }

    /// The `(width, height)` of `render_target`, or `None` if it was never created
    /// here. This, not the colour surface, is what a scene rasterizes into: a title is
    /// free to hand `sceGxmBeginScene` a colour surface whose own width/height fields
    /// are meaningless (a render-to-texture pass that only writes depth, or one whose
    /// engine fills the struct from a template), and this title does exactly that -
    /// its map pass carries 20,160 triangles into a surface initialised 1x1.
    #[inline]
    pub fn render_target_extent(&self, render_target: u32) -> Option<(u32, u32)> {
        self.render_targets.extent(render_target)
    }

    // --- Paletted textures ---------------------------------------------------

    /// `sceGxmTextureSetPalette(texture, paletteData)`.
    #[inline]
    pub fn set_texture_palette(&mut self, texture: u32, palette: u32) {
        self.textures.set_palette(texture, palette);
    }

    /// The palette bound to `texture`, or 0.
    #[inline]
    pub fn texture_palette(&self, texture: u32) -> u32 {
        self.textures.palette(texture)
    }
}
