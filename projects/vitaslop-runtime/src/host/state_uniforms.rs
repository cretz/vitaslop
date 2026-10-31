use super::*;

impl VitaState {
    /// `sceGxmReserveVertexDefaultUniformBuffer(context, void **uniformBuffer)`: hand
    /// back a fresh guest buffer sized to the bound vertex program's default uniform
    /// block, and bind it as the vertex uniform source so `record_draw` reads whatever
    /// the guest writes into it (this is the direct-path counterpart of a precomputed
    /// vertex state's default uniform buffer). `sceGxmSetUniformDataF` also copies into
    /// this same buffer, so reading it captures both ways the guest sets uniforms. The
    /// size is read from the program header (+0x2C) and clamped so an unresolved header
    /// cannot request an absurd allocation; a program with no default uniforms yields
    /// size 0, and the draw falls back to the `sceGxmSetUniformDataF` capture.
    /// Hand out `size` bytes of default-uniform space from a RECYCLED per-scene arena.
    ///
    /// A title reserves a default uniform buffer once per draw per stage - hundreds of
    /// times a frame, tens of thousands of times a second. Bump-allocating a fresh
    /// block for each (which is what this did until 2026-07-25) leaks the whole arena
    /// in minutes: the cursor marches up through guest memory until it reaches the
    /// region the MAIN THREAD'S STACK descends through, and then the guest's own
    /// `memcpy` of its uniforms into "its" buffer quietly overwrites live stack frames.
    /// The damage surfaces far away and much later - as a callee-saved register that
    /// comes back from a `POP` holding a float, and a crash in a function that did
    /// nothing wrong. On real hardware the default uniform buffer IS a ring the driver
    /// recycles, so recycling is also the faithful behaviour.
    ///
    /// The cursor resets per scene ([`Self::begin_scene`]), so within one frame every
    /// reserve still gets a distinct address exactly as before - only the growth across
    /// frames is gone.
    /// This is the FALLBACK half of the bump: the transpiler emits it inline for the
    /// ordinary case (`InlineOp::ReserveUniformBuffer`), so what reaches here is a context
    /// with no ring attached yet, a scene that has overrun the ring, or a bound program we
    /// did not create. The two paths compute the same address from the same three words,
    /// which is what lets a draw be unable to tell which of them ran.
    pub(super) fn alloc_default_uniform_buffer(&mut self, ctx: &mut GuestCtx, size: u32) -> u32 {
        let size = size.max(crate::vita::gxmctx::UNIFORM_MIN_ALLOC);
        let context = self.gxm_context;
        if context == 0 {
            // Nothing to bump and nowhere to record. A draw in this state already reads the
            // GXM defaults for every piece of sticky state and says so, so this only has to
            // hand back memory the guest can write into.
            return self.galloc(size, crate::vita::gxmctx::UNIFORM_ALIGN);
        }
        let (mut base, mut end, cursor) = crate::vita::gxmctx::uniform_ring(ctx, context);
        if base == 0 {
            self.attach_uniform_ring(ctx, context);
            (base, end, _) = crate::vita::gxmctx::uniform_ring(ctx, context);
        }
        if base == 0 {
            // The arena could not supply a ring at all; fall back to a direct
            // allocation so the call still returns something usable.
            return self.galloc(size, crate::vita::gxmctx::UNIFORM_ALIGN);
        }
        let align = crate::vita::gxmctx::UNIFORM_ALIGN;
        let at = (cursor.max(base).wrapping_add(align - 1)) & !(align - 1);
        if at < base || at > end || size > end - at {
            // One scene wanted more than the ring holds. Wrapping aliases two live
            // buffers, which is a real (if rare) fidelity loss, so say so once rather
            // than silently returning overlapping memory.
            if !self.uniform_ring_wrapped {
                self.uniform_ring_wrapped = true;
                tracing::warn!(
                    target: "vitaslop::gxm",
                    ring = Self::UNIFORM_RING_BYTES,
                    wanted = size,
                    "default-uniform ring wrapped WITHIN one scene - buffers now alias; raise UNIFORM_RING_BYTES"
                );
            }
            crate::vita::gxmctx::set(
                ctx,
                context,
                crate::vita::gxmctx::off::UNIFORM_RING_CURSOR,
                base.wrapping_add(size),
            );
            return base;
        }
        crate::vita::gxmctx::set(
            ctx,
            context,
            crate::vita::gxmctx::off::UNIFORM_RING_CURSOR,
            at.wrapping_add(size),
        );
        at
    }

    /// Give `context` its default-uniform ring, once.
    ///
    /// Separate from the bump because only the HOST can allocate guest memory: the emitted
    /// reserve reads a ring that is already there and hands the call back when it is not,
    /// so this runs at most once per context (at `sceGxmCreateContext`, and again from the
    /// fallback if that allocation had failed).
    pub fn attach_uniform_ring(&mut self, ctx: &mut GuestCtx, context: u32) {
        if context == 0 || crate::vita::gxmctx::uniform_ring(ctx, context).0 != 0 {
            return;
        }
        let base = self.galloc(Self::UNIFORM_RING_BYTES, crate::vita::gxmctx::UNIFORM_ALIGN);
        if base == 0 {
            return;
        }
        crate::vita::gxmctx::set_uniform_ring(ctx, context, base, Self::UNIFORM_RING_BYTES);
    }

    /// Bytes of recycled default-uniform space. One scene of this title's main screen
    /// uses ~340 draws x 2 stages x <=4 KiB worst case; 8 MiB leaves generous headroom
    /// while staying a rounding error against the guest region.
    pub(super) const UNIFORM_RING_BYTES: u32 = 8 * 1024 * 1024;

    pub fn reserve_vertex_uniform_buffer(&mut self, ctx: &mut GuestCtx) -> u32 {
        let handle = self.bound_vertex_program(ctx);
        let (size, header) = self.memoised_uniform_size(ctx, handle, ProgramStage::Vertex);
        let buf = self.alloc_default_uniform_buffer(ctx, size);
        poison_uniform_buffer(ctx, buf, size);
        self.bind_uniform_buffer(
            ctx,
            crate::vita::gxmctx::off::VERTEX_UNIFORM,
            buf,
            size,
            header,
        );
        tracing::trace!(
            target: "vitaslop::gxm",
            program = format_args!("{handle:#x}"),
            header = format_args!("{header:#x}"),
            size,
            buffer = format_args!("{buf:#x}"),
            "reserveVertexDefaultUniformBuffer"
        );
        buf
    }

    /// The default-uniform size a program handle carries, and the `SceGxmProgram *` it was
    /// created from.
    ///
    /// **The handle's own words are the definition**, not a cache of one: they are stamped
    /// at create ([`crate::vita::gxmprog`]), the emitted inline reserve reads exactly them,
    /// and this reads them too, so there is no arrangement in which the two paths size a
    /// buffer differently. A handle that predates that stamp - or one from a title that
    /// somehow passes a pointer we never created - falls back to reflecting the program
    /// here, which is what this call always did.
    pub(super) fn memoised_uniform_size(
        &mut self,
        ctx: &GuestCtx,
        handle: u32,
        stage: ProgramStage,
    ) -> (u32, u32) {
        match crate::vita::gxmprog::program(ctx, handle) {
            Some((size, _alloc, header)) => (size, header),
            None => {
                let header = match stage {
                    ProgramStage::Vertex => self.vertex_program_header(handle),
                    ProgramStage::Fragment => self.fragment_program_header(handle),
                };
                (self.uniform_size_bytes(ctx, header), header)
            }
        }
    }

    /// How many bytes of default uniform buffer a program declares: the greater of its
    /// REFLECTED extent and the header's own count field, clamped so an unresolved header
    /// cannot ask for an absurd allocation.
    ///
    /// The one definition of that number. [`crate::vita::gxmprog::init`] memoises it into
    /// the handle at create time and the reserve reads it back from there, so this runs
    /// once per program rather than once per draw - but it is the same arithmetic either
    /// way, which is the point of having it in one place.
    pub fn uniform_size_bytes(&mut self, ctx: &GuestCtx, header: u32) -> u32 {
        self.reflected_uniform_size_bytes(ctx, header)
            .max(default_uniform_buffer_bytes(ctx, header))
            .min(MAX_DEFAULT_UNIFORM_REGS)
    }

    /// Record what a stage's default uniform buffer is, where the draw reads it from.
    pub(super) fn bind_uniform_buffer(
        &mut self,
        ctx: &mut GuestCtx,
        record: u32,
        buf: u32,
        size: u32,
        header: u32,
    ) {
        if self.gxm_context == 0 {
            if !self.reported_reserve_without_context {
                self.reported_reserve_without_context = true;
                tracing::warn!(
                    target: "vitaslop::gxm",
                    "a default uniform buffer was reserved before sceGxmCreateContext - there \
                     is no context block to record it in, so the draws that follow read their \
                     uniforms from the sceGxmSetUniformDataF capture instead"
                );
            }
            return;
        }
        crate::vita::gxmctx::set_uniform_binding(
            ctx,
            self.gxm_context,
            record,
            crate::vita::gxmctx::UniformBinding { buf, size, header },
        );
    }

    /// The vertex stage's bound default uniform buffer, as the context block holds it.
    pub(super) fn vertex_uniform(&self, ctx: &GuestCtx) -> crate::vita::gxmctx::UniformBinding {
        self.uniform_binding(ctx, crate::vita::gxmctx::off::VERTEX_UNIFORM)
    }

    /// The fragment stage's.
    // Kept beside `vertex_uniform`: a pair where only one is currently read is a pair a
    // reader should still see whole.
    #[allow(dead_code)]
    pub(super) fn fragment_uniform(&self, ctx: &GuestCtx) -> crate::vita::gxmctx::UniformBinding {
        self.uniform_binding(ctx, crate::vita::gxmctx::off::FRAGMENT_UNIFORM)
    }

    pub(super) fn uniform_binding(&self, ctx: &GuestCtx, record: u32) -> crate::vita::gxmctx::UniformBinding {
        match self.gxm_context {
            0 => crate::vita::gxmctx::UniformBinding::default(),
            c => crate::vita::gxmctx::uniform_binding(ctx, c, record),
        }
    }

    /// `sceGxmReserveFragmentDefaultUniformBuffer`: the fragment-stage counterpart of
    /// [`Self::reserve_vertex_uniform_buffer`]. Hand back a guest buffer sized to the bound
    /// fragment program's default uniform block and bind it as the fragment uniform source,
    /// so a title that writes its per-material uniforms (tint / light / fog) directly into
    /// this buffer has them captured into the draw's material.
    /// `sceGxmSet{Vertex,Fragment}DefaultUniformBuffer(context, bufferData)`: the TITLE'S OWN
    /// buffer becomes the stage's default uniforms - the reserve path without the allocation,
    /// bound exactly as [`Self::reserve_vertex_uniform_buffer`] binds its own, sized by the
    /// program bound at the call (the reserve's rule). First called by a 2011 fighting title,
    /// which never reserves.
    /// `sceGxmSet{Vertex,Fragment}DefaultUniformBuffer`: the title names its OWN buffer as the
    /// stage's default uniform buffer.
    ///
    /// >>> CONTEXT STATE, NOT A RESERVATION FOR ONE PROGRAM. A reserve is sized for the program
    /// bound when it is made, and a draw by another program reading it is reading another
    /// object's uniforms ([`Self::stale_uniforms`]). This binding is the title's own pointer and
    /// stays in effect until it is set again, across program changes - a title sets the buffer
    /// and THEN the program, and draws. Recorded with the program bound at set time, every such
    /// draw was judged stale and read the empty `sceGxmSetUniformDataF` bank instead: MEASURED on
    /// a 2D title, its full-screen clear took `clearDepth` from that empty bank, wrote depth 0
    /// over its half-resolution layer target, and every layer drawn after it failed the depth
    /// test - the blurred background never rendered and only the sharp multi-tap copies of the
    /// foreground were left, as stacked bands. So it is marked [`USER_UNIFORM_BUFFER`], and a
    /// draw reads as many bytes as the program DRAWING declares.
    pub(crate) fn set_default_uniform_buffer(
        &mut self,
        ctx: &mut GuestCtx,
        stage: ProgramStage,
        buf: u32,
    ) {
        let off = match stage {
            ProgramStage::Vertex => crate::vita::gxmctx::off::VERTEX_UNIFORM,
            ProgramStage::Fragment => crate::vita::gxmctx::off::FRAGMENT_UNIFORM,
        };
        self.bind_uniform_buffer(ctx, off, buf, 0, USER_UNIFORM_BUFFER);
    }

    pub fn reserve_fragment_uniform_buffer(&mut self, ctx: &mut GuestCtx) -> u32 {
        let handle = self.bound_fragment_program(ctx);
        let (size, header) = self.memoised_uniform_size(ctx, handle, ProgramStage::Fragment);
        let buf = self.alloc_default_uniform_buffer(ctx, size);
        self.bind_uniform_buffer(
            ctx,
            crate::vita::gxmctx::off::FRAGMENT_UNIFORM,
            buf,
            size,
            header,
        );
        buf
    }

    /// Record a `sceGxmSetUniformDataF` write into the fallback SA bank - the one a draw uses
    /// when no default uniform buffer is bound.
    ///
    /// `at` is the register the write STARTS at: the parameter's own `resource_index` plus the
    /// call's `componentOffset`. Placing the values there rather than at the top is the same
    /// fact the buffer copy in `vita::gxm::set_uniform_data_f` rests on, and the two must agree
    /// - a program that sets two uniforms would otherwise have the second land on the first here
    /// while landing correctly in the buffer, so which of the two a draw read would depend on
    /// whether the title happened to reserve a buffer.
    ///
    /// Writes ACCUMULATE within a scene (the bank is cleared in `begin_scene`), because that is
    /// what the guest's own buffer does: two calls setting different uniforms leave both set.
    /// This is the FALLBACK half: the transpiler emits the same two writes inline
    /// (`InlineOp::SetUniformData`), so what reaches here is an F16 parameter, a record we
    /// could not read, or a write past the ceiling.
    pub fn set_uniforms(&mut self, ctx: &mut GuestCtx, at: u32, values: &[f32]) {
        let end = at as usize + values.len();
        // Same ceiling the reserved buffers are clamped to: a register offset past it is a
        // record we misread, not a uniform, and growing the bank to match it would turn a bad
        // read into an allocation.
        if end > MAX_DEFAULT_UNIFORM_REGS as usize {
            return;
        }
        let bank = self.ensure_sa_bank();
        if bank == 0 {
            return;
        }
        for (i, v) in values.iter().enumerate() {
            ctx.write_u32(
                bank + SA_BANK_DATA + (at as usize + i) as u32 * 4,
                v.to_bits(),
            );
        }
        // The gap between the old high-water mark and `at` is already zero - `clear_sa_bank`
        // left it that way for this scene - which is what the `Vec`'s `resize(end, 0.0)` used
        // to provide.
        if (ctx.read_u32(bank) as usize) < end {
            ctx.write_u32(bank, end as u32);
        }
    }

    /// The half-precision counterpart of [`Self::set_uniforms`]: record a
    /// `sceGxmSetUniformDataF` write to an F16-declared uniform, which packs TWO components
    /// per register.
    ///
    /// `at_half` counts HALVES from the start of the buffer, so register `n` holds halves
    /// `2n` and `2n+1`. The bank stores raw register words (a lane's f32 bit pattern IS the
    /// register), so a partial write read-modify-writes the neighbouring half rather than
    /// clearing it - the same thing the guest's own buffer does, and the same thing
    /// `vita::gxm::set_uniform_data_f` does to the reserved buffer. The two must agree, or
    /// which one a draw reads decides what it renders.
    pub fn set_uniform_halves(&mut self, ctx: &mut GuestCtx, at_half: u32, values: &[f32]) {
        let end_reg = (at_half as usize + values.len()).div_ceil(2);
        if end_reg > MAX_DEFAULT_UNIFORM_REGS as usize {
            return;
        }
        let bank = self.ensure_sa_bank();
        if bank == 0 {
            return;
        }
        for (i, v) in values.iter().enumerate() {
            let component = at_half as usize + i;
            let addr = bank + SA_BANK_DATA + (component / 2) as u32 * 4;
            let word = ctx.read_u32(addr);
            let h = u32::from(crate::render::f32_to_half(*v));
            ctx.write_u32(
                addr,
                if component.is_multiple_of(2) {
                    (word & 0xffff_0000) | h
                } else {
                    (word & 0x0000_ffff) | (h << 16)
                },
            );
        }
        if (ctx.read_u32(bank) as usize) < end_reg {
            ctx.write_u32(bank, end_reg as u32);
        }
    }
}
