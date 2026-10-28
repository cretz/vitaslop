use super::*;

impl GxmRenderer {
    /// Ensure the kept DISPLAY IMAGE for `addr` exists at `(w, h)` - see
    /// [`GxmRenderer::display_images`] for why it is kept at all and why it is not an
    /// `rtt` entry.
    ///
    /// Rebuilt only when the extent changes, which is a window resize and not a per-frame
    /// event. One image per display buffer a title rotates: at 960x544 RGBA8 that is
    /// ~2 MB each, so a title with six of them pays ~12 MB to make its own finished frames
    /// readable.
    /// Answers whether the image was CREATED by this call, which is the only moment it
    /// holds nothing and therefore the only moment a clear is owed - see the display arm.
    pub(super) fn ensure_display_image(&mut self, device: &wgpu::Device, addr: u32, w: u32, h: u32) -> bool {
        let (w, h) = (w.max(1), h.max(1));
        if let Some(d) = self.display_images.get(&addr)
            && d.tex.width() == w && d.tex.height() == h {
                return false;
            }
        // The sRGB twin is DECLARED whether or not this title ever asks for gamma mode,
        // exactly as `ensure_rtt` declares it for every render target and for the same
        // reason: a view format cannot be added to a texture after creation, and the mode
        // is sticky guest state that can be turned on under a surface already in use, so
        // deciding here would mean rebuilding the display image mid-run.
        //
        // `compat_mode` gates it for the same reason it gates `ensure_rtt`'s: on an adapter
        // that refuses the declaration the cost is not a worse picture but NO picture - the
        // texture itself becomes invalid and every view, bind group and pass built on it
        // fails with it. A gamma surface in compat mode therefore keeps the old, darker
        // behaviour, and `report_gamma_display_surface` says so rather than leaving it
        // silent as it was.
        let srgb_fmt =
            if crate::gpu::compat_mode() { None } else { srgb_twin(self.color_format) };
        let view_formats: Vec<wgpu::TextureFormat> = srgb_fmt.into_iter().collect();
        let tex = device.create_texture(&wgpu::TextureDescriptor {
            label: Some("gxm-display-image"),
            size: wgpu::Extent3d { width: w, height: h, depth_or_array_layers: 1 },
            mip_level_count: 1,
            sample_count: 1,
            dimension: wgpu::TextureDimension::D2,
            // The caller's own format. The display pass already encodes with
            // `self.color_format` (see the display arm), so this is the same target it
            // always wrote to, moved one step earlier.
            format: self.color_format,
            // COPY_SRC because a draw whose fragment program reads the DESTINATION colour
            // needs this image copied into `GxmRenderer::dest_color` mid-pass.
            usage: wgpu::TextureUsages::RENDER_ATTACHMENT
                | wgpu::TextureUsages::TEXTURE_BINDING
                | wgpu::TextureUsages::COPY_SRC,
            view_formats: &view_formats,
        });
        let view = tex.create_view(&Default::default());
        let view_srgb = srgb_fmt.map(|format| {
            tex.create_view(&wgpu::TextureViewDescriptor {
                format: Some(format),
                ..Default::default()
            })
        });
        // >>> AND ITS OWN DEPTH, AT ITS OWN EXTENT.
        // A render pass requires every attachment to share one size. This image is the
        // GUEST surface's extent, which is NOT the caller's framebuffer extent - on the
        // browser the canvas is 960x544 while the guest surface is 640x368 - so borrowing
        // the caller's depth made `BeginRenderPass` invalid and took the WHOLE command
        // buffer down with it. MEASURED on a phone: 636 frames dropped to a validation
        // error, i.e. the title rendered nothing at all. It never fired on the desktop
        // headless path because there the two extents happen to be equal, which is exactly
        // the kind of agreement a browser stops honouring.
        let depth = device.create_texture(&wgpu::TextureDescriptor {
            label: Some("gxm-display-image-depth"),
            size: wgpu::Extent3d { width: w, height: h, depth_or_array_layers: 1 },
            mip_level_count: 1,
            sample_count: 1,
            dimension: wgpu::TextureDimension::D2,
            format: depth_format(),
            // COPY_*: a display scene whose surface FORCE_LOADs / FORCE_STOREs a stencil plane
            // a fragment program stored to moves it through a store mirror (`store_mirror`).
            usage: wgpu::TextureUsages::RENDER_ATTACHMENT | wgpu::TextureUsages::COPY_SRC | wgpu::TextureUsages::COPY_DST,
            view_formats: &[],
        });
        let depth_view = depth.create_view(&Default::default());
        // A freshly created image holds nothing yet; the display arm sets this when it
        // renders into it.
        let entry = DisplayImage { tex, view, view_srgb, gamma: false, depth, depth_view, msaa: None, shadow: None };
        if let Some(old) = self.display_images.insert(addr, entry) {
            old.tex.destroy();
            old.depth.destroy();
            if let Some((s, _, _)) = old.shadow {
                s.destroy();
            }
            if let Some(m) = old.msaa {
                m.color.destroy();
                m.depth.destroy();
            }
        }
        true
    }

    /// Give the display image at `addr` multisampled attachments at `samples`, matching its
    /// extent and format, unless it already has them.
    ///
    /// >>> THE DISPLAY BUFFER CAN BE THE MULTISAMPLED ONE. The display arm assumed its
    /// render target is always `SCE_GXM_MULTISAMPLE_NONE`; a football title renders its
    /// whole world straight into a 2X display target (with an MSAA_DOWNSCALE colour
    /// surface), so the match was drawn single-sampled - the one pass where the guest's
    /// antialiasing shows most. The offscreen arm's attachments (`ensure_rtt`), reused here:
    /// the pass rasterises into these and names the kept image as its resolve target.
    pub(super) fn ensure_display_msaa(&mut self, device: &wgpu::Device, addr: u32, samples: u32) {
        let Some(d) = self.display_images.get_mut(&addr) else { return };
        if d.msaa.as_ref().is_some_and(|m| m.samples == samples) {
            return;
        }
        if let Some(m) = d.msaa.take() {
            m.color.destroy();
            m.depth.destroy();
        }
        let size = d.tex.size();
        let srgb_fmt = if crate::gpu::compat_mode() { None } else { srgb_twin(self.color_format) };
        let view_formats: Vec<wgpu::TextureFormat> = srgb_fmt.into_iter().collect();
        let mc = device.create_texture(&wgpu::TextureDescriptor {
            label: Some("gxm-display-msaa-color"),
            size,
            mip_level_count: 1,
            sample_count: samples,
            dimension: wgpu::TextureDimension::D2,
            format: self.color_format,
            // TEXTURE_BINDING: a destination read's per-sample copy loads it.
            usage: wgpu::TextureUsages::RENDER_ATTACHMENT | wgpu::TextureUsages::TEXTURE_BINDING,
            view_formats: &view_formats,
        });
        let md = device.create_texture(&wgpu::TextureDescriptor {
            label: Some("gxm-display-msaa-depth"),
            size,
            mip_level_count: 1,
            sample_count: samples,
            dimension: wgpu::TextureDimension::D2,
            format: depth_format(),
            usage: wgpu::TextureUsages::RENDER_ATTACHMENT,
            view_formats: &[],
        });
        d.msaa = Some(MsaaAttachments {
            color_view: mc.create_view(&Default::default()),
            depth_view: md.create_view(&Default::default()),
            color: mc,
            depth: md,
            samples,
        });
        report_multisample_granted(addr, size.width, size.height, 1, samples);
    }

    /// Ensure the offscreen supersample target exists at `scale * (surf_w, surf_h)`,
    /// rebuilding it if the scale or size changed. Also refreshes the scale uniform.
    pub(super) fn ensure_ss_target(&mut self, device: &wgpu::Device, queue: &wgpu::Queue, surf_w: u32, surf_h: u32) {
        let scale = self.ss_scale;
        let (w, h) = (surf_w * scale, surf_h * scale);
        let stale = self.ss_target.as_ref().map(|t| t.scale != scale || t.width != w || t.height != h).unwrap_or(true);
        if !stale {
            return;
        }
        let color = device.create_texture(&wgpu::TextureDescriptor {
            label: Some("gxm-ss-color"),
            size: wgpu::Extent3d { width: w.max(1), height: h.max(1), depth_or_array_layers: 1 },
            mip_level_count: 1,
            sample_count: 1,
            dimension: wgpu::TextureDimension::D2,
            format: self.color_format,
            usage: wgpu::TextureUsages::RENDER_ATTACHMENT | wgpu::TextureUsages::TEXTURE_BINDING,
            view_formats: &[],
        });
        let color_view = color.create_view(&Default::default());
        let depth = device.create_texture(&wgpu::TextureDescriptor {
            label: Some("gxm-ss-depth"),
            size: wgpu::Extent3d { width: w.max(1), height: h.max(1), depth_or_array_layers: 1 },
            mip_level_count: 1,
            sample_count: 1,
            dimension: wgpu::TextureDimension::D2,
            format: depth_format(),
            usage: wgpu::TextureUsages::RENDER_ATTACHMENT,
            view_formats: &[],
        });
        let depth_view = depth.create_view(&Default::default());
        queue.write_buffer(&self.resolve_scale_buf, 0, &[scale, 0, 0, 0].iter().flat_map(|v: &u32| v.to_le_bytes()).collect::<Vec<u8>>());
        let resolve_bind = device.create_bind_group(&wgpu::BindGroupDescriptor {
            label: Some("gxm-resolve-bind"),
            layout: &self.resolve_layout,
            entries: &[
                wgpu::BindGroupEntry { binding: 0, resource: wgpu::BindingResource::TextureView(&color_view) },
                wgpu::BindGroupEntry { binding: 1, resource: self.resolve_scale_buf.as_entire_binding() },
            ],
        });
        self.ss_target = Some(SsTarget { scale, width: w, height: h, _color: color, color_view, depth_view, resolve_bind });
    }

    /// Ensure a decoded texture is uploaded (cached by content key) and that a bind
    /// group exists for it at the requested filter. A cache hit does no GPU work.
    pub(super) fn ensure_texture(&mut self, device: &wgpu::Device, queue: &wgpu::Queue, t: &crate::gpu::GxmTexture) {
        // Bound the caches BY BYTES: on pathological churn, clear wholesale and
        // re-upload (keys are content fingerprints, so correctness is unaffected).
        // See `tex_cache_budget_bytes` for why an entry COUNT bounded nothing.
        if self.views.contains_key(&t.key) {
            enc(&ENC.tex_view_cached, 1);
        } else {
            self.views_bytes += texture_bytes(t.width, t.height);
            if self.views_bytes >= tex_cache_budget_bytes() {
                enc(&ENC.tex_view_wholesale_clears, 1);
                // >>> NOT THE TEXTURES THIS PASS ALREADY HANDED OUT. A draw prepared earlier
                // in the pass holds `BindKey::Tex` for its texture and looks the bind group up
                // when the pass is ENCODED; clearing everything here, on a later draw's
                // texture, left that lookup with nothing - a panic in `bind_for` ("no entry
                // found for key") on a volleyball title's character select. The pass's own
                // textures stay, and the budget counts them.
                let keep = &self.tex_pass_keys;
                self.views.retain(|k, _| keep.contains_key(k));
                self.tex_binds.retain(|(k, _), _| keep.contains_key(k));
                self.views_bytes = keep.values().sum::<usize>() + texture_bytes(t.width, t.height);
            }
        }
        self.tex_pass_keys.insert(t.key, texture_bytes(t.width, t.height));
        self.views.entry(t.key).or_insert_with(|| {
            enc(&ENC.tex_uploaded, 1);
            enc_tex_upload(t.rgba.len() as u64);
            // >>> THE ROW PITCH IS PADDED TO 256, AND A TIGHT ONE IS A VALIDATION ERROR
            // >>> THAT LOSES THE WHOLE FRAME.
            //
            // A multi-row copy requires `bytes_per_row` to be a multiple of
            // `COPY_BYTES_PER_ROW_ALIGNMENT` (256), and `width * 4` is only that when the
            // width is a multiple of 64. A narrower texture - a gradient strip, a UI
            // sliver - made the SUBMIT a validation error, and a failed submit loses every
            // draw in the command buffer, not just this upload: MEASURED on a baseball
            // title in a live game, the picture was BLACK while the guest submitted 540
            // draws a frame. `create_texture_with_data` is NOT the fix - it records the
            // same tight-pitch copy and made it fire on every texture instead of a few.
            let tex = device.create_texture(&wgpu::TextureDescriptor {
                label: Some("gxm-tex"),
                size: wgpu::Extent3d {
                    width: t.width,
                    height: t.height,
                    depth_or_array_layers: 1,
                },
                mip_level_count: 1,
                sample_count: 1,
                dimension: wgpu::TextureDimension::D2,
                format: wgpu::TextureFormat::Rgba8Unorm,
                usage: wgpu::TextureUsages::TEXTURE_BINDING | wgpu::TextureUsages::COPY_DST,
                view_formats: &[],
            });
            const ROW_ALIGN: u32 = wgpu::COPY_BYTES_PER_ROW_ALIGNMENT;
            let tight = t.width * 4;
            let pitch = tight.div_ceil(ROW_ALIGN) * ROW_ALIGN;
            // Only when it is actually needed: on a 64-aligned width (which is most of
            // them) the guest's bytes go straight in with no copy at all.
            let padded: Option<Vec<u8>> = (pitch != tight).then(|| {
                let mut v = vec![0u8; pitch as usize * t.height as usize];
                for y in 0..t.height as usize {
                    let src = y * tight as usize;
                    let end = (src + tight as usize).min(t.rgba.len());
                    if src >= end {
                        break;
                    }
                    let dst = y * pitch as usize;
                    v[dst..dst + (end - src)].copy_from_slice(&t.rgba[src..end]);
                }
                v
            });
            // >>> TIMED, FOR THE SAME REASON THE BUFFER WRITES ARE, AND IT IS THE GAP THAT
            // KEPT THE LOADING STALL UNATTRIBUTED. Every stall this project has caught was
            // caught on a `write_buffer`, and the one that opens a scene reads `behind 215
            // textures / 31305 KB` - a 31 MB burst of `write_texture` in the same frame,
            // none of it timed. A buffer write that blocks AFTER that burst may simply be
            // inheriting an upload path the TEXTURES exhausted, and no counter here could
            // tell those apart: one says spread the texture burst over frames, the other
            // says flush the staging ring between writes. See `report_write_buffer_stall`,
            // which both now report through.
            let t_tex = Stopwatch::start();
            queue.write_texture(
                wgpu::TexelCopyTextureInfo {
                    texture: &tex,
                    mip_level: 0,
                    origin: wgpu::Origin3d::ZERO,
                    aspect: wgpu::TextureAspect::All,
                },
                padded.as_deref().unwrap_or(&t.rgba),
                wgpu::TexelCopyBufferLayout {
                    offset: 0,
                    bytes_per_row: Some(pitch),
                    rows_per_image: Some(t.height),
                },
                wgpu::Extent3d { width: t.width, height: t.height, depth_or_array_layers: 1 },
            );
            report_upload_stall(
                UploadKind::Texture,
                (t_tex.ms() * 1000.0) as u64,
                padded.as_deref().unwrap_or(&t.rgba).len() as u64,
                TEX_UPLOADS_THIS_FRAME.load(std::sync::atomic::Ordering::Relaxed),
            );
            tex.create_view(&Default::default())
        });
        let bind_key = (t.key, t.filter_linear);
        if self.tex_binds.contains_key(&bind_key) {
            enc(&ENC.bind_groups_reused, 1);
        } else {
            enc(&ENC.bind_groups_built, 1);
        }
        if !self.tex_binds.contains_key(&bind_key) {
            let view = &self.views[&t.key];
            let samp = if t.filter_linear { &self.sampler_linear } else { &self.sampler_point };
            let bind = device.create_bind_group(&wgpu::BindGroupDescriptor {
                label: Some("gxm-tex-bind"),
                layout: &self.texture_layout,
                entries: &[
                    wgpu::BindGroupEntry {
                        binding: 0,
                        resource: wgpu::BindingResource::TextureView(view),
                    },
                    wgpu::BindGroupEntry {
                        binding: 1,
                        resource: wgpu::BindingResource::Sampler(samp),
                    },
                ],
            });
            self.tex_binds.insert(bind_key, bind);
        }
    }

    /// Grow (or first-create) a per-frame arena buffer so it holds at least `need`
    /// bytes. Returns true if the buffer was (re)created (the caller rebinds if so).
    ///
    /// Shared with [`GxmRenderer::ensure_gxp_arena`] so both arena pools grow on the same
    /// schedule: geometrically, so a steadily-larger frame does not reallocate every time.
    ///
    /// # >>> A BUFFER'S FIRST WRITE IS THE EXPENSIVE ONE, and this is where that is recorded
    /// Measured in the browser, each slow write against its own control taken immediately
    /// after it, on the same buffer, on the same thread:
    ///
    /// | first write | 4 bytes after it | THE SAME BYTES AGAIN |
    /// |---|---|---|
    /// | 248 KB in 243 ms | 0.0 ms | **0.1 ms** |
    /// | 635 KB in 232 ms | 0.0 ms | **0.3 ms** |
    ///
    /// The 4-byte control says the thread was running throughout, so the stall is not a
    /// scheduling artefact; the repeat says the cost belongs to the DESTINATION and is paid
    /// once per buffer. On the user's phone those stalls are 25.7 s of one session, in
    /// blocks of up to 6 s, each freezing the whole browser.
    ///
    /// The obvious inference - that the 4 KB floor makes an arena climb a LADDER of
    /// destinations (4 KB, 8 KB, ... 256 KB) and pay that price at every rung - is MEASURED
    /// AND NULL on this machine; see `gpu::arena_floor_bytes` for the paired run and for
    /// why the arm is kept anyway. The floor stays where the measurement is.
    pub(super) fn cap_for(need: u64) -> u64 {
        need.max(4).next_power_of_two().max(crate::gpu::arena_floor_bytes())
    }

    pub(super) fn ensure_buffer(
        device: &wgpu::Device,
        queue: &wgpu::Queue,
        buf: &mut Option<wgpu::Buffer>,
        cap: &mut u64,
        need: u64,
        usage: wgpu::BufferUsages,
        label: &str,
        pool: &mut BufferPool,
    ) -> bool {
        let need = need.max(4);
        if buf.is_some() && *cap >= need {
            return false;
        }
        let new_cap = Self::cap_for(need);
        enc_buffer_created();
        // Through the pool: an arena that grew last frame and shrank back this one gets its
        // old buffer, already backed, instead of an allocation the next write pays for.
        // The size is a power of two (`cap_for`), which is what makes a hit likely at all.
        *buf = Some(pool.create(
            device,
            queue,
            &wgpu::BufferDescriptor {
                label: Some(label),
                size: new_cap,
                usage: usage | wgpu::BufferUsages::COPY_DST,
                mapped_at_creation: false,
            },
        ));
        *cap = new_cap;
        true
    }

    /// Make slot `slot` of the recompiled path's arena pool hold this pass's vertex,
    /// index and uniform data, creating or growing its three buffers only when the data
    /// no longer fits, and uploading with `write_buffer`.
    ///
    /// The slot is addressed by PASS ORDINAL, so it is written at most once per frame and
    /// the write is queue-ordered behind the previous frame's submit. A buffer that has to
    /// GROW is handed to the graveyard rather than dropped, for the same reason every
    /// other retired buffer is: dropping a `wgpu::Buffer` on the web backend only makes it
    /// collectable, and the previous frame's commands still name it until its submit
    /// retires. See [`GxmRenderer::gxp_arenas`] and `retired_buffers`.
    #[allow(clippy::too_many_arguments)]
    pub(super) fn ensure_gxp_arena(
        device: &wgpu::Device,
        queue: &wgpu::Queue,
        // The belt and the encoder the arena upload rides on. `None` is the
        // `VITASLOP_GXM_STAGING=0` arm, which puts it back on `queue.write_buffer` - see
        // `GxmRenderer::gxp_staging`.
        belt: &mut Option<wgpu::util::StagingBelt>,
        encoder: &mut wgpu::CommandEncoder,
        arenas: &mut Vec<GxpArenaSlot>,
        retired: &mut Vec<wgpu::Buffer>,
        slot: usize,
        vdata: &[u8],
        idata: &[u8],
        udata: &[u8],
        // Sorted, disjoint `[start, end)` ranges of `udata` NOT to upload - see
        // `GxpLive::udata_holes`.
        uholes: &[(usize, usize)],
        create_ms: &mut f64,
        write_ms: &mut f64,
    ) {
        // `write_buffer` copies whole 4-byte units, and a packed vertex stream need not
        // land on one. Padding the LENGTH is safe where padding the arena would not be:
        // every draw addresses its own byte range, so the tail past the last draw is
        // never read.
        let padded = |n: usize| (n.max(4) as u64).next_multiple_of(wgpu::COPY_BUFFER_ALIGNMENT);
        let (vneed, ineed, uneed) = (padded(vdata.len()), padded(idata.len()), padded(udata.len()));

        while arenas.len() <= slot {
            let n = arenas.len();
            let t_create = Stopwatch::start();
            let mk = |need: u64, usage: wgpu::BufferUsages, label: &str| {
                enc_buffer_created();
                new_buffer(
                    device,
                    queue,
                    &wgpu::BufferDescriptor {
                        label: Some(label),
                        size: Self::cap_for(need),
                        usage: usage | wgpu::BufferUsages::COPY_DST,
                        mapped_at_creation: false,
                    },
                )
            };
            // A slot below `slot` that is being created only to fill the gap is sized to
            // the minimum: it belongs to a pass that did not run this frame, and it will
            // grow on the frame that first uses it.
            let (v, i, u) = if n == slot { (vneed, ineed, uneed) } else { (4, 4, 4) };
            arenas.push(GxpArenaSlot {
                vbo: mk(v, wgpu::BufferUsages::VERTEX, "gxp-vbo"),
                ibo: mk(i, wgpu::BufferUsages::INDEX, "gxp-ibo"),
                ubo: mk(u, arena_ubo_usage(), "gxp-ubo"),
                vcap: Self::cap_for(v),
                icap: Self::cap_for(i),
                ucap: Self::cap_for(u),
                generation: next_arena_generation(),
                last_used: 0,
            });
            *create_ms += t_create.ms();
        }

        let a = &mut arenas[slot];
        let grow = |buf: &mut wgpu::Buffer,
                        cap: &mut u64,
                        need: u64,
                        usage: wgpu::BufferUsages,
                        label: &str,
                        retired: &mut Vec<wgpu::Buffer>|
         -> bool {
            if *cap >= need {
                return false;
            }
            enc_buffer_created();
            let new = new_buffer(device, queue, &wgpu::BufferDescriptor {
                label: Some(label),
                size: Self::cap_for(need),
                usage: usage | wgpu::BufferUsages::COPY_DST,
                mapped_at_creation: false,
            });
            *cap = Self::cap_for(need);
            retired.push(std::mem::replace(buf, new));
            true
        };
        let t_grow = Stopwatch::start();
        grow(&mut a.vbo, &mut a.vcap, vneed, wgpu::BufferUsages::VERTEX, "gxp-vbo", retired);
        grow(&mut a.ibo, &mut a.icap, ineed, wgpu::BufferUsages::INDEX, "gxp-ibo", retired);
        // Only the UNIFORM arena's identity is baked into a bind group, so only its
        // re-creation has to invalidate one.
        //
        // An arena made before the first storage-bound window existed cannot back that
        // binding: forget its capacity so the grow re-creates it with the usage it now needs.
        let uusage = arena_ubo_usage();
        if !a.ubo.usage().contains(uusage) {
            a.ucap = 0;
        }
        if grow(&mut a.ubo, &mut a.ucap, uneed, uusage, "gxp-ubo", retired) {
            a.generation = next_arena_generation();
        }
        *create_ms += t_grow.ms();
        let t_write = Stopwatch::start();

        // One `write_buffer` per NON-EMPTY arena, over the padded length.
        //
        // >>> AN EMPTY ARENA IS NOT WRITTEN AT ALL, and that is a call saved rather than
        // four bytes. `queue.write_buffer` is a crossing into the browser's WebGPU
        // implementation and its cost is dominated by the CALL, not the payload: the
        // measured worst frame on a retail golf title's course load is **627 writes,
        // 27.5 MB, 1,848 ms** - about 3 ms a call, which no byte count explains. A pass
        // whose draws are all resident-geometry writes no vertices, one with no
        // SA-resident uniforms writes none, and a frame that opens 209 passes was making
        // three calls for each of them regardless. Nothing can read what is not written:
        // a draw addresses an arena only through an offset the same pass produced, so an
        // empty arena has no reader, and a newly created WebGPU buffer is already zeroed.
        // `which` names the arena (0 vertices, 1 indices, 2 uniforms) for the repeat probe
        // below; it costs nothing when the probe is off.
        let write = |belt: &mut Option<wgpu::util::StagingBelt>,
                     encoder: &mut wgpu::CommandEncoder,
                     buf: &wgpu::Buffer,
                     data: &[u8],
                     need: u64,
                     which: u8,
                     // Where in `buf` the bytes land - non-zero only for a piece of the
                     // uniform arena between two fill holes (see `udata_holes` below).
                     at: u64| {
            if data.is_empty() {
                return;
            }
            enc(&ENC.buffer_bytes, need);
            enc(&ENC.buffer_writes, 1);
            // >>> VOLUME, COUNTED BEFORE THE TRANSPORT IS CHOSEN - and that placement is
            // >>> the whole point of it.
            //
            // `BUFFER_WRITE_LARGEST_BYTES` below is on the `queue.write_buffer` arm, so the
            // belt drives it to zero. That is the correct reading of "does this build still
            // make the call that blocks on the device" and a WRONG reading of "how much is
            // this build uploading" - the same megabytes still move, they just move by
            // `copy_buffer_to_buffer` instead. Without a row on this side, doubling the
            // geometry would leave the gate reporting a serene 0 KB.
            //
            // The user asked the question that found this: *"I hope it's nothing game
            // specific?"* - and the honest answer needs both numbers, because "we stopped
            // calling that" and "we stopped sending that" are different claims and only
            // the first one is true.
            UPLOAD_BYTES_RUN.fetch_add(need, std::sync::atomic::Ordering::Relaxed);
            UPLOAD_LARGEST_BYTES.fetch_max(need, std::sync::atomic::Ordering::Relaxed);
            // >>> THE STAGING BELT TAKES IT, AND THAT IS THE WHOLE FIX. See
            // `GxmRenderer::gxp_staging` for the device measurement that puts it here:
            // this exact call, at this exact size, blocked for 208 ms and then 551.9 ms on
            // the user's phone with nothing in flight.
            //
            // The belt hands back a MAPPED slice of a pooled chunk; filling it is a plain
            // memcpy into memory this process already owns, and the transfer becomes a
            // `copy_buffer_to_buffer` the GPU does on its own time instead of a call that
            // parks the emulator thread. The stall instruments below stay on the
            // `queue.write_buffer` arm, which is now the A/B arm rather than the default -
            // they measure a call this path no longer makes.
            if let Some(belt) = belt.as_mut() {
                let Some(size) = wgpu::BufferSize::new(need) else {
                    return;
                };
                // >>> ON THE WEB, STRAIGHT INTO THE BROWSER'S MAPPING.
                //
                // wgpu's web `BufferViewMut` is a COPY: the first write pulls the whole
                // mapped range into the wasm heap (`Uint8Array::to_vec`), and the drop
                // `set`s it all back - three passes over every uploaded byte, MEASURED at
                // 1.4% + 0.5% of the busy worker on a baseball title. The read view's `Uint8Array` is
                // the mapped `ArrayBuffer` itself, and WebGPU does not care which kind of
                // view a mapped range is written through, so one `set` from a view of
                // `data` does the whole upload. Same allocation, same copy command as
                // `StagingBelt::write_buffer` issues.
                #[cfg(target_arch = "wasm32")]
                {
                    let four = const { wgpu::BufferSize::new(wgpu::COPY_BUFFER_ALIGNMENT).unwrap() };
                    let s = belt.allocate(size, four);
                    encoder.copy_buffer_to_buffer(s.buffer(), s.offset(), buf, at, size.get());
                    let view = s.get_mapped_range().expect("staging belt chunk is mapped");
                    // SAFETY: nothing allocates between `view` and `set`, so the wasm memory
                    // the view borrows cannot move under it.
                    unsafe {
                        view.as_uint8array().set(&js_sys::Uint8Array::view(data), 0);
                    }
                    return;
                }
                #[allow(unreachable_code)]
                let mut view = belt.write_buffer(encoder, buf, at, size);
                // `need` is `data.len()` rounded UP to the copy alignment, so the source is
                // never longer than the view. The tail past the data is padding no draw
                // addresses - see `padded` above - so it is left as the chunk found it
                // rather than zeroed, which would be a second pass over every byte.
                //
                // `slice(..)` rather than an index: a mapped staging range may be
                // WRITE-COMBINING memory, so wgpu deliberately refuses to hand out an
                // ordinary `&mut [u8]` for it and gives a write-only view instead. Reading
                // one back would be correct-looking and glacially slow.
                view.slice(..data.len()).copy_from_slice(data);
                return;
            }
            // >>> `VITASLOP_GXM_ARENA_REPEAT=1`: HOW MANY OF THESE BYTES ARE THE SAME BYTES
            // >>> WE UPLOADED LAST FRAME. The question that decides whether residency is
            // >>> worth rebuilding, asked before building anything.
            //
            // The worker profile puts `writeBuffer` at 7.6% of the browser (plus 1.5% for
            // the copy into JS) and the counters show ONE call of 2,448 KB a frame - the
            // world pass's vertex arena, 1.6-2.4 ms of a phone frame in a single call. If
            // those bytes are identical frame to frame they should be resident and not
            // re-uploaded; if they are not, residency cannot help and the fix is elsewhere.
            //
            // THIS IS A KNOB AND NOT A DEFAULT because it hashes every uploaded byte, which
            // is the cost it is measuring. Keyed on (slot, which buffer) so a rotating arena
            // phase compares against the same destination it wrote last time.
            if crate::gpu::arena_repeat_probe() {
                let h = fnv64(0xcbf2_9ce4_8422_2325, data);
                let mut g = ARENA_REPEAT.lock().unwrap_or_else(|e| e.into_inner());
                let a = g.get_or_insert_with(ArenaRepeat::default);
                if a.seen.insert((slot as u32, which), h) == Some(h) {
                    a.repeated += need;
                }
                a.total += need;
            }
            // >>> TIMED PER CALL, because the total cannot say what this is waiting on.
            // See `buffer_write_max_us`. A `Stopwatch` here is two clock reads on a path
            // that runs about a dozen times a frame.
            let t_one = Stopwatch::start();
            if data.len() as u64 == need {
                queue.write_buffer(buf, at, data);
            } else {
                let mut padded = vec![0u8; need as usize];
                padded[..data.len()].copy_from_slice(data);
                queue.write_buffer(buf, at, &padded);
            }
            let us = (t_one.ms() * 1000.0) as u64;
            // >>> AND THE SAME BYTES THROUGH THE OTHER PATH, WHEN ASKED. See
            // `arena_upload_probe`.
            if crate::gpu::arena_upload_probe() && need >= 64 * 1024 {
                static LEFT: std::sync::atomic::AtomicU32 = std::sync::atomic::AtomicU32::new(8);
                if LEFT.fetch_update(std::sync::atomic::Ordering::Relaxed, std::sync::atomic::Ordering::Relaxed, |n| n.checked_sub(1)).is_ok() {
                    let t_map = Stopwatch::start();
                    let staging = device.create_buffer(&wgpu::BufferDescriptor {
                        label: Some("arena-upload-probe"),
                        size: need,
                        usage: wgpu::BufferUsages::COPY_SRC,
                        mapped_at_creation: true,
                    });
                    let t_created = t_map.ms();
                    {
                        let mut view = staging
                            .slice(..)
                            .get_mapped_range_mut()
                            .expect("a buffer created mapped_at_creation is mapped");
                        // Write-only by construction on this target - a mapped range may be
                        // uncached memory, so wgpu refuses to hand out a readable slice.
                        view.slice(..data.len()).copy_from_slice(data);
                    }
                    let t_filled = t_map.ms();
                    staging.unmap();
                    let t_total = t_map.ms();
                    tracing::warn!(
                        target: "vitaslop::gxm",
                        "ARENA UPLOAD PROBE, {} KB: queue.write_buffer {:.2} ms against a MAPPED staging buffer at {:.2} ms (create {:.2} + fill {:.2} + unmap {:.2}). The staging arm is the CPU half only - no copy_buffer_to_buffer, which is GPU-side - so it is a LOWER BOUND on the alternative, and the comparison is only worth acting on if it is much the smaller of the two.",
                        need / 1024,
                        us as f64 / 1000.0,
                        t_total,
                        t_created,
                        t_filled - t_created,
                        t_total - t_filled,
                    );
                }
            }
            // >>> AND THE NEGATIVE CONTROL, TAKEN THE INSTANT A WRITE LOOKS SLOW.
            //
            // A stopwatch around one call blames THAT CALL for every microsecond the thread
            // did not have. This worker shares a process with twenty-odd emulator threads
            // and a decoder, so "the write blocked" and "this thread was not running" are
            // indistinguishable from one span - and five separate GPU-side mechanisms have
            // now been timed and cleared while the span survived every one of them, which
            // is exactly what an instrument measuring the wrong thing looks like.
            //
            // So when a write reads slow, four bytes go into a buffer that is ALREADY
            // written - no allocation, no zero-init, no staging growth, nothing the theories
            // above name - and that is timed too. Fast control beside a slow write means the
            // write really is the cost; a control that is ALSO slow means the thread was
            // stopped and the write is a bystander.
            if us >= 100_000 {
                // FOUR BYTES: the floor for "a write_buffer call on this thread, now".
                let t_ctl = Stopwatch::start();
                queue.write_buffer(buf, at, &[0u8; 4]);
                let tiny_us = (t_ctl.ms() * 1000.0) as u64;
                // >>> AND THE SAME NUMBER OF BYTES AGAIN, INTO THE SAME BUFFER, which is
                // what separates the two survivors. The tiny control being instant says the
                // thread was running, not which property of the slow call was expensive:
                //
                //  * if this FULL-SIZE repeat is fast, the cost belonged to the DESTINATION
                //    - a buffer being touched for the first time - and it is paid once per
                //    buffer, so the fix is to create fewer of them;
                //  * if it is slow too, the cost is the SIZE - a staging path that pays per
                //    megabyte every time - and creating fewer buffers would change nothing.
                //
                // Writing the same bytes to the same place twice is harmless: the second
                // write lands on the queue after the first with identical content.
                let t_same = Stopwatch::start();
                if data.len() as u64 == need {
                    queue.write_buffer(buf, at, data);
                }
                let same_us = (t_same.ms() * 1000.0) as u64;
                report_write_stall_control(tiny_us, same_us, us, need);
            }
            // A MAX, so it is published with a compare-and-set rather than an add - the
            // frame's worst call is the one being looked for, not their sum.
            //
            // >>> ONE PACKED WORD, so the time and the bytes are THE SAME CALL. Maximising
            // two words independently pairs the slowest call's milliseconds with the
            // fattest call's bytes, and the whole question here is whether the slow call
            // was carrying anything. `us` in the high half orders the comparison; the
            // clamps keep a pathological frame from carrying into the other field.
            let packed = (us.min(u32::MAX as u64) << 32) | need.min(u32::MAX as u64);
            ENC.buffer_write_worst.fetch_max(packed, std::sync::atomic::Ordering::Relaxed);
            // >>> AND SAY SO, THE MOMENT IT HAPPENS. A single write blocking for SECONDS is
            // not a slow copy, and the panel's maxima can only report it after the fact.
            // See `report_write_buffer_stall`.
            report_upload_stall(
                UploadKind::Buffer,
                us,
                need,
                BUFFER_WRITES_THIS_FRAME.fetch_add(1, std::sync::atomic::Ordering::Relaxed),
            );
            // >>> AND THE SAME THING FOR THE WHOLE RUN, which is the only one a person can
            // actually read. See `BUFFER_WRITE_WORST_RUN`.
            BUFFER_WRITE_WORST_RUN.fetch_max(packed, std::sync::atomic::Ordering::Relaxed);
            // >>> AND THE SIZE ALONE, which is the half a desktop run can be held to. The
            // packed register above is keyed on the worst TIME, so on this workstation -
            // where nothing blocks - it reports whichever call happened to be slowest by
            // microseconds and says nothing about how fat the fattest one was. See
            // `BUFFER_WRITE_LARGEST_BYTES`.
            BUFFER_WRITE_LARGEST_BYTES
                .fetch_max(need, std::sync::atomic::Ordering::Relaxed);
            // >>> AND THE ONE THE QUEUE-DEPTH BOUND ARMS ITSELF FROM. Microseconds only -
            // the bytes are a diagnostic, and what arms a throttle must be the single
            // number that says "this call BLOCKED". See `BUFFER_WRITE_WORST_SINCE_READ`.
            BUFFER_WRITE_WORST_SINCE_READ
                .fetch_max(us, std::sync::atomic::Ordering::Relaxed);
        };
        write(belt, encoder, &a.vbo, vdata, vneed, 0, 0);
        write(belt, encoder, &a.ibo, idata, ineed, 1, 0);
        // The uniform arena goes up AROUND its fill holes: the ranges a raw target's texels
        // are copied over after this upload (`push_mem_windows_per_draw`). Each piece starts
        // and ends on a hole's edge, which is word-aligned, so it is a legal copy on its own.
        let mut from = 0usize;
        for &(hs, he) in uholes {
            let (hs, he) = (hs.min(udata.len()), he.min(udata.len()));
            if hs > from {
                write(belt, encoder, &a.ubo, &udata[from..hs], (hs - from) as u64, 2, from as u64);
            }
            from = from.max(he);
        }
        if from == 0 {
            write(belt, encoder, &a.ubo, udata, uneed, 2, 0);
        } else if udata.len() > from {
            write(belt, encoder, &a.ubo, &udata[from..], padded(udata.len() - from), 2, from as u64);
        }
        *write_ms += t_write.ms();
    }

    /// The group-1 bind group for an item (the shared white fallback, or a cached
    /// texture bind ensured earlier this frame).
    pub(super) fn bind_for(&self, key: BindKey) -> &wgpu::BindGroup {
        match key {
            BindKey::White => &self.white_bind,
            BindKey::Tex(k, l) => &self.tex_binds[&(k, l)],
            BindKey::Rtt(a, l, s) => &self.rtt_binds[&(a, l, s)],
        }
    }

    /// The offscreen target for the colour surface at guest address `addr`, created (or
    /// re-created at a new size) on demand.
    ///
    /// `sample_depth` says some later pass in this frame reads this target's DEPTH, which
    /// costs the depth texture a `TEXTURE_BINDING` usage and a converted R32Float
    /// companion. It is off for every target nothing samples, so an ordinary pass pays
    /// nothing for the feature.
    /// The RAW 64-bit target at `addr`, created or recreated at `width` x `height` - see
    /// [`Self::rtt_raw`]. Single-sampled, no gamma twin, no depth companion: an integer
    /// attachment has no sRGB encoding and cannot be multisample-resolved, and no title
    /// measured samples such a pass's depth.
    /// Unpack the RAW 64-bit surface `key` into its `Rgba16Float` companion, on the frame's
    /// encoder right after the pass that wrote it.
    ///
    /// # Why a sampler needs this
    /// A 64-bit surface is stored as the fragment's raw register pair, which is exactly what
    /// the hardware writes. A later program that declares an F16F16F16F16 TEXTURE over the
    /// same memory samples those words as four halves - FILTERED, through its own sample
    /// instruction - and an `Rg32Uint` texture can be neither filtered nor sampled.
    /// MEASURED on a fighting title: its image-based lighting renders 12x72 and
    /// 24x144 F16F16F16F16 intermediates and filters them into the lighting cubes; with
    /// only the raw view the cube pass was refused and every character was unlit.
    pub(super) fn convert_raw_to_float(
        &mut self,
        device: &wgpu::Device,
        encoder: &mut wgpu::CommandEncoder,
        key: (u32, u32, u32),
    ) {
        if self.raw_to_float.is_none() {
            let layout = device.create_bind_group_layout(&wgpu::BindGroupLayoutDescriptor {
                label: Some("gxm-raw64-f16-layout"),
                entries: &[wgpu::BindGroupLayoutEntry {
                    binding: 0,
                    visibility: wgpu::ShaderStages::FRAGMENT,
                    ty: wgpu::BindingType::Texture {
                        sample_type: wgpu::TextureSampleType::Uint,
                        view_dimension: wgpu::TextureViewDimension::D2,
                        multisampled: false,
                    },
                    count: None,
                }],
            });
            let module = device.create_shader_module(wgpu::ShaderModuleDescriptor {
                label: Some("gxm-raw64-f16"),
                source: wgpu::ShaderSource::Wgsl(
                    "@group(0) @binding(0) var raw: texture_2d<u32>;\n\
                     @vertex fn vs(@builtin(vertex_index) i: u32) -> @builtin(position) vec4<f32> {\n\
                     \x20 let p = vec2<f32>(f32((i << 1u) & 2u), f32(i & 2u));\n\
                     \x20 return vec4<f32>(p * 2.0 - 1.0, 0.0, 1.0);\n\
                     }\n\
                     @fragment fn fs(@builtin(position) at: vec4<f32>) -> @location(0) vec4<f32> {\n\
                     \x20 let w = textureLoad(raw, vec2<i32>(at.xy), 0).xy;\n\
                     \x20 return vec4<f32>(unpack2x16float(w.x), unpack2x16float(w.y));\n\
                     }\n"
                        .into(),
                ),
            });
            let pl = device.create_pipeline_layout(&wgpu::PipelineLayoutDescriptor {
                label: Some("gxm-raw64-f16"),
                bind_group_layouts: &[Some(&layout)],
                immediate_size: 0,
            });
            let pipeline = device.create_render_pipeline(&wgpu::RenderPipelineDescriptor {
                label: Some("gxm-raw64-f16"),
                layout: Some(&pl),
                vertex: wgpu::VertexState {
                    module: &module,
                    entry_point: Some("vs"),
                    compilation_options: Default::default(),
                    buffers: &[],
                },
                fragment: Some(wgpu::FragmentState {
                    module: &module,
                    entry_point: Some("fs"),
                    compilation_options: Default::default(),
                    targets: &[Some(wgpu::TextureFormat::Rgba16Float.into())],
                }),
                primitive: Default::default(),
                depth_stencil: None,
                multisample: Default::default(),
                multiview_mask: None,
                cache: None,
            });
            self.raw_to_float = Some((pipeline, layout));
        }
        let (Some((pipeline, layout)), Some(surface)) = (self.raw_to_float.as_ref(), self.rtt_raw.get(&key)) else {
            return;
        };
        let bind = device.create_bind_group(&wgpu::BindGroupDescriptor {
            label: Some("gxm-raw64-f16"),
            layout,
            entries: &[wgpu::BindGroupEntry {
                binding: 0,
                resource: wgpu::BindingResource::TextureView(&surface.color_view),
            }],
        });
        let mut pass = encoder.begin_render_pass(&wgpu::RenderPassDescriptor {
            label: Some("gxm-raw64-f16"),
            color_attachments: &[Some(wgpu::RenderPassColorAttachment {
                view: &surface.float_view,
                resolve_target: None,
                depth_slice: None,
                ops: wgpu::Operations { load: wgpu::LoadOp::Clear(wgpu::Color::TRANSPARENT), store: wgpu::StoreOp::Store },
            })],
            depth_stencil_attachment: None,
            timestamp_writes: None,
            occlusion_query_set: None,
            multiview_mask: None,
        });
        pass.set_pipeline(pipeline);
        pass.set_bind_group(0, &bind, &[]);
        pass.draw(0..3, 0..1);
    }

    pub(super) fn ensure_rtt_raw(&mut self, device: &wgpu::Device, addr: u32, width: u32, height: u32, sample_depth: bool) {
        // Keyed by address AND extent: a title draws one address as two 64-bit surfaces of
        // different sizes in one frame, and a sampler names the one whose extent it
        // declares. One entry per shape, so neither recreates the other every frame.
        if let Some(s) = self.rtt_raw.get_mut(&(addr, width, height)) {
            // A depth reader is added to an existing surface rather than rebuilding it: the
            // words and the depth attachment it converts are both kept.
            if sample_depth && s.gxm_depth.is_none() {
                let size = wgpu::Extent3d { width: width.max(1), height: height.max(1), depth_or_array_layers: 1 };
                s.gxm_depth = build_rtt_gxm_depth(device, size, true, &None, &s.depth);
                self.rtt_epoch += 1;
            }
            return;
        }
        self.rtt_epoch += 1;
        enc(&ENC.rtt_created, 1);
        let size = wgpu::Extent3d { width: width.max(1), height: height.max(1), depth_or_array_layers: 1 };
        let color = device.create_texture(&wgpu::TextureDescriptor {
            label: Some("gxm-rtt-raw64"),
            size,
            mip_level_count: 1,
            sample_count: 1,
            dimension: wgpu::TextureDimension::D2,
            format: RAW64_FORMAT,
            usage: wgpu::TextureUsages::RENDER_ATTACHMENT
                | wgpu::TextureUsages::TEXTURE_BINDING
                | wgpu::TextureUsages::COPY_SRC,
            view_formats: &[],
        });
        let depth = device.create_texture(&wgpu::TextureDescriptor {
            label: Some("gxm-rtt-raw64-depth"),
            size,
            mip_level_count: 1,
            sample_count: 1,
            dimension: wgpu::TextureDimension::D2,
            format: depth_format(),
            // COPY_*: a FORCE_STORE / FORCE_LOAD surface's depth moves through `depth_saved`.
            // TEXTURE_BINDING: a later pass may sample this scene's stencil plane
            // (`convert_gxm_stencil`). Raw targets are few, so it is not made conditional.
            usage: wgpu::TextureUsages::RENDER_ATTACHMENT
                | wgpu::TextureUsages::TEXTURE_BINDING
                | wgpu::TextureUsages::COPY_SRC
                | wgpu::TextureUsages::COPY_DST,
            view_formats: &[],
        });
        let color_view = color.create_view(&Default::default());
        let depth_view = depth.create_view(&Default::default());
        // The depth TEXTURE is kept now, for one reason: `depth_saved` copies it (a view
        // cannot be a copy source). It is the attachment `depth_view` renders into.
        let gxm_depth = build_rtt_gxm_depth(device, size, sample_depth, &None, &depth);
        let float_view = device
            .create_texture(&wgpu::TextureDescriptor {
                label: Some("gxm-rtt-raw64-f16"),
                size,
                mip_level_count: 1,
                sample_count: 1,
                dimension: wgpu::TextureDimension::D2,
                format: wgpu::TextureFormat::Rgba16Float,
                usage: wgpu::TextureUsages::RENDER_ATTACHMENT | wgpu::TextureUsages::TEXTURE_BINDING,
                view_formats: &[],
            })
            .create_view(&Default::default());
        self.rtt_raw.insert(
            (addr, width, height),
            RawSurface { width, height, color, color_view, depth, depth_view, float_view, gxm_depth },
        );
    }
}
