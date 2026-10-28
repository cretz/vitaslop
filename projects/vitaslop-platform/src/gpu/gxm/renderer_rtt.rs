use super::*;

impl GxmRenderer {
    /// >>> A MEMORY WINDOW OVER A 64-BIT TARGET THIS FRAME RENDERED READS THAT TARGET.
    ///
    /// A fragment program's memory loads read guest memory as it stands when the draw runs -
    /// and on the hardware a pass earlier in the frame has already STORED its tiles there. A
    /// 64-bit surface is rendered here as raw register pairs into an `Rg32Uint` target
    /// (`rtt_raw`) that never reaches guest memory within the frame, so a window over it
    /// carried whatever the guest's bytes were: zero in every capsule. MEASURED on a shooter's
    /// stairwell: its depth-reconstruct pass loads 4x4 blocks of its F16F16F16F16 world
    /// target through four windows one row apart, read all zeroes, and the passes built on
    /// its output darkened the whole frame.
    ///
    /// The raw words ARE the memory - two little-endian words per pixel, the bytes a store of
    /// the tile writes - so the window is filled by copying the target's rows into the arena
    /// at the window's own offset, after the arena upload and before the pass. Only whole
    /// texels inside the surface's width are copied; a row's pitch padding and anything past
    /// the surface keep the guest's bytes, which is what memory holds there. A window that
    /// does not sit on a texel boundary, or a target that is the pass's own attachment, is
    /// refused by name and keeps the guest's bytes. `VITASLOP_MEM_WINDOW_FROM_TARGET=0` is the
    /// arm back.
    pub(super) fn fill_windows_from_raw_targets(
        &self,
        encoder: &mut wgpu::CommandEncoder,
        slot: usize,
        prepared: &[GxpPrepared],
        current_target: Option<u32>,
    ) {
        if prepared.iter().all(|p| p.fmem_ranges.is_empty()) {
            return;
        }
        let arena = &self.gxp_arenas[slot].ubo;
        let mut done: HashSet<u64> = HashSet::default();
        for &(addr, len, aoff) in prepared.iter().flat_map(|p| p.fmem_ranges.iter()) {
            if !done.insert(aoff) {
                continue;
            }
            for &key @ (base, tw, th) in self.rtt_rendered_raw.keys() {
                let pitch = self.rtt_pitch.get(&key).map_or(tw, |&(p, _)| p);
                let plan = match raw_window_fill_plan(addr, len, base, tw, th, pitch) {
                    None => continue,
                    Some(_) if Some(base) == current_target => {
                        Err("the target is this pass's own attachment")
                    }
                    Some(plan) => plan,
                };
                let Some(raw) = self.rtt_raw.get(&key) else { continue };
                crate::gpu::report_window_from_target(addr, len, base, tw, th, plan.as_ref().err().copied());
                for c in plan.iter().flatten() {
                    encoder.copy_texture_to_buffer(
                        wgpu::TexelCopyTextureInfo {
                            texture: &raw.color,
                            mip_level: 0,
                            origin: wgpu::Origin3d { x: c.x, y: c.y, z: 0 },
                            aspect: wgpu::TextureAspect::All,
                        },
                        wgpu::TexelCopyBufferInfo {
                            buffer: arena,
                            layout: wgpu::TexelCopyBufferLayout {
                                offset: aoff + c.window_offset,
                                bytes_per_row: c.bytes_per_row,
                                rows_per_image: c.bytes_per_row.map(|_| c.rows),
                            },
                        },
                        wgpu::Extent3d { width: c.width, height: c.rows, depth_or_array_layers: 1 },
                    );
                }
            }
        }
    }

    /// The fixed-function pipeline for a MASKED draw, built on first sight and returned by
    /// its cache key (see `fixed_stencil`). The reference is not in the key: it is set on the
    /// pass per draw.
    pub(super) fn ensure_fixed_stencil_pipeline(
        &mut self,
        device: &wgpu::Device,
        opaque: bool,
        format: wgpu::TextureFormat,
        samples: u32,
        words: &[u32; 14],
        two_sided: bool,
    ) -> u64 {
        let mut h: u64 = 0xcbf2_9ce4_8422_2325;
        for (i, v) in words.iter().enumerate() {
            if i != 6 && i != 13 {
                h ^= u64::from(*v);
                h = h.wrapping_mul(0x0000_0100_0000_01b3);
            }
        }
        h ^= two_sided as u64;
        let key = (opaque, format, samples, h);
        if !self.fixed_stencil.contains_key(&key) {
            let pipe = build_fixed_pipeline(
                device,
                &self.fixed_shader,
                &self.fixed_layout,
                opaque,
                format,
                samples,
                gxm_stencil_from_words(words, two_sided),
            );
            self.fixed_stencil.insert(key, pipe);
        }
        h
    }

    pub(super) fn ensure_rtt(
        &mut self,
        device: &wgpu::Device,
        addr: u32,
        width: u32,
        height: u32,
        sample_depth: bool,
        // The render target's `SceGxmMultisampleMode` - see `RttTarget::multisample`.
        multisample: u32,
        // The colour surface asks for `SCE_GXM_COLOR_SURFACE_SCALE_MSAA_DOWNSCALE`, i.e. it
        // stores the RESOLVE. Not part of the sample-count decision - that is the render
        // target's business - but checked against it, see the report below.
        msaa_downscale: bool,
        // The colour surface is in GAMMA-CORRECT mode. Passed in rather than decided at
        // encode time so a gamma target never ALLOCATES multisampled attachments it is
        // then refused the use of - which is both wasted memory and a log that says
        // "granted" and "REFUSED" about the same target in the same frame.
        gamma: bool,
        // The format to create the colour attachment in, or `None` for the renderer's
        // default. Every caller passes `None` today: the only wide colour formats a
        // measured title renders into are the 64-bit pair, and `raw64_color_format` takes
        // those down a different path entirely (`Rg32Uint` raw words) before this is
        // reached. The parameter exists because the RECREATE test below needs a format to
        // compare against, and a target whose attachment format changed must be rebuilt.
        want_format: Option<wgpu::TextureFormat>,
    ) {
        // The target's IDENTITY: the colour surface's address AND the extent this render
        // target rasterises at. See `rtt` for why the extent belongs here - a baseball
        // title runs two render targets (256x256 and 512x1024) through the one colour
        // address 0x8e20b030 every frame, and with the address alone each pass rebuilt the
        // other's target and threw its draws away.
        let key = (addr, width, height);
        let want_samples =
            if no_multisample() || gamma { 1 } else { gxm_sample_count(multisample) };
        // Gaining a depth reader is a reason to rebuild: the depth texture it already has
        // was created without `TEXTURE_BINDING` and cannot be sampled. Gaining, losing or
        // CHANGING the sample count is the same kind of event - the attachments themselves
        // are a different texture. The REASON is kept and reported
        // (`report_rtt_recreated`): a target rebuilt every frame is a per-frame allocation
        // plus a discarded image, and the counter alone cannot say which of these did it.
        //
        // >>> "SIZE" IS NO LONGER ONE OF THE REASONS, AND ITS ABSENCE IS THE WHOLE FIX.
        //
        // A different extent at a known address is a DIFFERENT TARGET, so it is a different
        // key and lands on the `None` arm as a first sighting - the entry already there is
        // left alone with its pixels. The same key cannot disagree about its extent (the
        // extent IS the key), so the old `t.width != width` arm could only ever fire for
        // the case it got wrong.
        let want_format = want_format.unwrap_or(self.color_format);
        let why: Option<&'static str> = match self.rtt.get(&key) {
            Some(t) if sample_depth && t.gxm_depth.is_none() => Some("depth companion"),
            // A target whose guest colour format changed is a different attachment: the
            // pipelines that write it are keyed on the format, so the texture has to agree.
            Some(t) if t.color.format() != want_format => Some("colour format"),
            Some(t)
                if t.samples_now
                    != if msaa_refused_for_depth(sample_depth) { 1 } else { want_samples } =>
            {
                Some("sample count")
            }
            Some(_) => None,
            None => Some("first sighting"),
        };
        let stale = why.is_some();
        // >>> A SAMPLE-COUNT CHANGE KEEPS THE PICTURE. The resolved image lives in `color`,
        // which a change of sample count does not touch: the guest's memory at this address
        // holds that image whatever samples the NEXT scene rasterises with. MEASURED on a pool
        // title: the 3D hall is a 4x scene into 0x8e340000 (MSAA_DOWNSCALE), the HUD a
        // single-sample scene into the same address - and rebuilding the whole target between
        // them discarded the hall, so the HUD was drawn over black. Only the multisampled
        // attachments (and the depth reader built over them) are replaced.
        if why == Some("sample count")
            && let Some(t) = self.rtt.get_mut(&key)
        {
            let size = wgpu::Extent3d { width: width.max(1), height: height.max(1), depth_or_array_layers: 1 };
            let want = if msaa_refused_for_depth(sample_depth) { 1 } else { want_samples };
            // Dropping to one sample KEEPS the multisampled pair for the next multisampled
            // scene; only a count the pair does not have builds a new one.
            if want > 1 && t.msaa.as_ref().is_none_or(|m| m.samples != want) {
                let view_formats: Vec<wgpu::TextureFormat> = t.color_view_srgb.as_ref().and_then(|_| srgb_twin(self.color_format)).into_iter().collect();
                let msaa = build_rtt_msaa(device, size, want, want_format, &view_formats, sample_depth);
                if let Some(old) = std::mem::replace(&mut t.msaa, msaa) {
                    self.retired_textures.extend([old.color, old.depth]);
                }
            }
            t.samples_now = if t.msaa.is_some() { want } else { 1 };
            // The depth reader converts the attachment the pass rasterises into, so it
            // follows the active one.
            if t.gxm_depth.is_some() || sample_depth {
                let active = if t.samples_now > 1 { &t.msaa } else { &None };
                let gxm_depth = build_rtt_gxm_depth(device, size, sample_depth || t.gxm_depth.is_some(), active, &t.depth);
                if let Some(old) = std::mem::replace(&mut t.gxm_depth, gxm_depth) {
                    self.retired_textures.push(old.tex);
                }
            }
            report_rtt_samples_swapped(addr, t.samples_now);
            self.rtt_epoch += 1;
            self.rtt_binds.retain(|&(a, _, _), _| a != addr);
            return;
        }
        if let Some(why) = why {
            report_rtt_recreated(addr, why);
        }
        // >>> GAMMA MODE IS STICKY GUEST STATE AND IT CAN CHANGE UNDER AN EXISTING TARGET.
        //
        // `sceGxmColorSurfaceSetGammaMode` is set on the SURFACE, not on the render target,
        // so a title may turn it on or off for a buffer it has been using all along. The
        // texture does not need rebuilding for that - both views already exist, which is
        // why the sRGB twin is declared on every target - but the recorded answer does,
        // and it was previously written once at creation and never again.
        //
        // What that costs is a renderer that DISAGREES WITH ITSELF: `encode_chain` renders
        // through the mode the guest is asking for RIGHT NOW, while everything that SAMPLES
        // the target - `sample_views` and `snapshot_rtt` - reads the stale one. One end
        // encodes and the other does not decode, and on a feedback path that error
        // compounds every iteration, running the image to white or to black depending on
        // which way round the two ended up. Measured on a 3-pass feedback chain with a
        // stale-true flag over a linear target: 128 -> 55 -> 10 -> 1 -> 0.
        if let Some(t) = self.rtt.get_mut(&key)
            && t.gamma != gamma {
                t.gamma = gamma;
                report_gamma_mode_changed(addr, gamma);
                // The cached fixed-function bind groups name a view chosen under the old
                // answer, so they are stale in exactly the same way. By ADDRESS - see
                // `rtt_binds`, which is keyed on the address a draw sampled.
                self.rtt_binds.retain(|&(a, _, _), _| a != addr);
            }
        if !stale {
            return;
        }
        // >>> AND IT HAS NO PREVIOUS CONTENTS TO LOAD. A recreated target is a NEW texture;
        // the pixels the last frame left are in the one being dropped. See
        // `rtt_ever_rendered` - its frame's first pass clears again, once. Only for THIS
        // key: another target at the same address is untouched and keeps its accumulation.
        self.rtt_ever_rendered.remove(&key);
        // Every view of the target about to be replaced dies here, so every cached bind
        // group naming ANY target is invalidated. See `rtt_epoch`.
        self.rtt_epoch += 1;
        enc(&ENC.rtt_created, 1);
        let size = wgpu::Extent3d { width: width.max(1), height: height.max(1), depth_or_array_layers: 1 };
        // Declare the sRGB twin as an allowed view format on EVERY target, not only the
        // ones currently in gamma mode: `sceGxmColorSurfaceSetGammaMode` is sticky state a
        // title may set at any point, and a texture's view formats are fixed at creation.
        // Declaring it costs nothing on any backend that matters and removes a whole class
        // of "the mode arrived after the target existed" bug.
        // >>> COMPATIBILITY MODE FORBIDS A VIEW OF ANOTHER FORMAT, so the sRGB twin is not
        // declared there and no gamma view is built. The cost is real and is reported by
        // `report_gamma_surface`: a gamma-correct surface then stores LINEAR values where the
        // hardware would sRGB-encode them, so it and everything sampling it read darker than
        // the title intends. That is a colour error on some passes. The alternative is not a
        // better picture, it is NO picture: declaring it makes the target itself invalid, and
        // every view, bind group and pass built on it fails with it.
        // >>> AND A FLOAT TARGET HAS NO sRGB TWIN AND NEEDS NONE. `srgb_twin` answers for
        // the 8-bit UNORM formats; a `Rgba16Float` attachment already stores linear values
        // at a range and precision the encoding exists to fake, so asking for a twin of it
        // would be asking for a view format that does not exist.
        let srgb_fmt = if crate::gpu::compat_mode() || want_format != self.color_format {
            None
        } else {
            srgb_twin(self.color_format)
        };
        let view_formats: Vec<wgpu::TextureFormat> = srgb_fmt.into_iter().collect();
        let color = device.create_texture(&wgpu::TextureDescriptor {
            label: Some("gxm-rtt-color"),
            size,
            mip_level_count: 1,
            sample_count: 1,
            dimension: wgpu::TextureDimension::D2,
            format: want_format,
            usage: wgpu::TextureUsages::RENDER_ATTACHMENT
                | wgpu::TextureUsages::TEXTURE_BINDING
                | wgpu::TextureUsages::COPY_SRC,
            view_formats: &view_formats,
        });
        let depth = device.create_texture(&wgpu::TextureDescriptor {
            label: Some("gxm-rtt-depth"),
            size,
            mip_level_count: 1,
            sample_count: 1,
            dimension: wgpu::TextureDimension::D2,
            format: depth_format(),
            // COPY_*: a FORCE_STORE / FORCE_LOAD surface's depth moves through `depth_saved`.
            usage: if sample_depth {
                wgpu::TextureUsages::RENDER_ATTACHMENT | wgpu::TextureUsages::TEXTURE_BINDING
            } else {
                wgpu::TextureUsages::RENDER_ATTACHMENT
            } | wgpu::TextureUsages::COPY_SRC
                | wgpu::TextureUsages::COPY_DST,
            view_formats: &[],
        });
        let color_view = color.create_view(&Default::default());
        let color_view_srgb = srgb_fmt.map(|f| {
            color.create_view(&wgpu::TextureViewDescriptor {
                label: Some("gxm-rtt-color-srgb"),
                format: Some(f),
                ..Default::default()
            })
        });
        let depth_view = depth.create_view(&Default::default());

        // The multisampled attachments the guest asked for. A target whose depth a later
        // pass SAMPLES gets them too, except on a compatibility-mode adapter - see
        // `msaa_refused_for_depth`: its depth conversion then reads sample 0 of the
        // multisampled depth (`GxmDepthTarget::multisampled`).
        //
        // NOTE the scale mode is NOT consulted here. It describes the surface's STORED
        // form (it holds a resolved image), which is already true of `color` below; the
        // sample count is the render target's business and comes from the guest's
        // `multisampleMode`. Conflating the two is what made this a 2x2 supersample.
        let samples = want_samples;
        let msaa = build_rtt_msaa(device, size, samples, want_format, &view_formats, sample_depth);
        let gxm_depth = build_rtt_gxm_depth(device, size, sample_depth, &msaa, &depth);
        // A multisampled target whose colour surface does NOT ask for the downscale resolve
        // expects its own memory to hold the raw SAMPLES, not a resolved image - and we
        // hand it a resolved one. No target on any title measured here is in that state
        // (every multisampled one is scale mode 1), which is exactly why it would go
        // unnoticed if one ever were.
        if gxm_sample_count(multisample) > 1 && !msaa_downscale {
            report_unresolved_multisample_surface(addr, width, height);
        }
        if msaa.is_some() {
            report_multisample_granted(addr, width, height, multisample, samples);
        } else if gxm_sample_count(multisample) > 1 && !no_multisample() {
            // The guest asked and did not get it. `gamma` and `sample_depth` are the two
            // reasons, and they are reported here - at the one place that knows both -
            // rather than half here and half at encode time.
            report_multisample_refused(addr, width, height, gamma);
        }
        // A bind group over the new view is stale by construction; drop any cached ones.
        // By ADDRESS, which also drops the groups naming the OTHER target at this address
        // when there is one - conservative, and it costs one rebuild rather than risking a
        // group over a texture that has just been retired.
        self.rtt_binds.retain(|&(a, _, _), _| a != addr);
        // Hand the OLD target's allocations back before the new one is installed. `insert`
        // would otherwise drop it, which on a native backend frees it and in the browser
        // does not - see `RttSurface::destroy`. Counted beside the creations so a standing
        // gap between the two is visible rather than inferred.
        if let Some(old) = self.rtt.insert(
            key,
            RttSurface {
                width,
                height,
                color,
                color_view,
                color_view_srgb,
                gamma,
                depth,
                depth_view,
                shadow: None,
                gxm_depth,
                samples_now: msaa.as_ref().map_or(1, |m| m.samples),
                msaa_stale: false,
                msaa,
            },
        ) {
            // >>> RETIRED, NOT DESTROYED. This runs while a frame is being encoded, and an
            // earlier pass in the same command buffer may still name the attachments this
            // target owns - destroying them there invalidates the WHOLE submit. See
            // `retired_textures`.
            self.retired_textures.extend(old.into_textures());
            enc(&ENC.rtt_destroyed, 1);
        }
    }

    /// Every extent at which a render target is currently held at this guest ADDRESS,
    /// sorted, smallest first.
    ///
    /// >>> A SCAN BY ADDRESS OVER A MAP KEYED BY `(address, width, height)`, and it is here
    /// >>> because some questions genuinely have no extent to offer.
    ///
    /// The callers are the three that only ever hold an address: the sampled-extent
    /// diagnostic (which is asking "is this address a target at all" before it has looked
    /// up any extent), the alias test (which is asking whether a BOUND extent matches any
    /// target here), and the cube-face stamp (a cube names face 0's address and nothing
    /// else). `rtt` holds a handful of entries - the reclamation's whole design note says
    /// so, and `RTT_KEEP_FREELY` is the gate that keeps it that way - so a linear scan of it
    /// is cheaper than a second index that would have to be kept in step.
    pub(super) fn rtt_extents_at(&self, addr: u32) -> Vec<(u32, u32)> {
        let mut v: Vec<(u32, u32)> =
            self.rtt.keys().filter(|(a, _, _)| *a == addr).map(|(_, w, h)| (*w, *h)).collect();
        v.sort_unstable();
        v
    }

    /// Stamp the frame `now` onto the target(s) this frame touched at `addr` - see
    /// `rtt_used`, and `encode_chain` for what counts as a touch.
    ///
    /// `want` is the extent the caller can honestly name (a pass knows its render target's;
    /// a draw that samples one declares its own). When it names a target held here, only
    /// that target is stamped - which is the point of the rekeying: two targets at one
    /// address age independently, and one being used no longer keeps the other alive for
    /// ever. When it names none, or when the caller has no extent to offer at all, EVERY
    /// target at the address is stamped: the sampler path resolves by address and by range
    /// (`rendered_alias` binds a 960x544 target through a 1920x1088 descriptor), so any of
    /// them may be the one that draw is reading, and stamping is the conservative half -
    /// the worst it costs is a target held for another `RTT_STALE_FRAMES`.
    ///
    /// An address that holds no target at all is ignored, which is the rule the old
    /// closure stated: `sampled` is every texture address any draw binds - hundreds a frame
    /// - and stamping all of them would give this map the unbounded growth it exists to fix.
    pub(super) fn stamp_rtt_used(&mut self, addr: u32, want: Option<(u32, u32)>, now: u64) {
        if let Some((w, h)) = want
            && self.rtt.contains_key(&(addr, w, h)) {
                self.rtt_used.insert((addr, w, h), now);
                return;
            }
        for (w, h) in self.rtt_extents_at(addr) {
            self.rtt_used.insert((addr, w, h), now);
        }
    }

    /// Run the depth-conversion pass for the target `key` names: read its depth attachment
    /// and write the guest's own depth encoding into its R32Float companion.
    ///
    /// Takes the full `(address, width, height)` key rather than the colour address,
    /// because two targets can share that address and only one of them holds the depth
    /// this pass just filled - see `rtt`.
    ///
    /// Called right after the pass that filled the depth, because the next pass into the
    /// same target clears it again.
    pub(super) fn convert_gxm_depth(
        &mut self,
        device: &wgpu::Device,
        queue: &wgpu::Queue,
        encoder: &mut wgpu::CommandEncoder,
        key: (u32, u32, u32),
        depth_min: f32,
        depth_scale: f32,
    ) {
        let Some(t) = self.rtt.get(&key) else { return };
        let Some(gd) = t.gxm_depth.as_ref() else { return };
        let (src_view, view, multisampled) = (gd.src_view.clone(), gd.view.clone(), gd.multisampled);
        self.encode_depth_convert(queue, device, encoder, key.0, &src_view, &view, multisampled, depth_min, depth_scale);
    }

    /// [`Self::convert_gxm_depth`] for a 64-bit RAW surface's depth (`RawSurface::gxm_depth`).
    pub(super) fn convert_raw_gxm_depth(
        &mut self,
        device: &wgpu::Device,
        queue: &wgpu::Queue,
        encoder: &mut wgpu::CommandEncoder,
        key: (u32, u32, u32),
        depth_min: f32,
        depth_scale: f32,
    ) {
        let Some(gd) = self.rtt_raw.get(&key).and_then(|s| s.gxm_depth.as_ref()) else { return };
        let (src_view, view, multisampled) = (gd.src_view.clone(), gd.view.clone(), gd.multisampled);
        self.encode_depth_convert(queue, device, encoder, key.0, &src_view, &view, multisampled, depth_min, depth_scale);
    }

    /// The conversion pass itself: `src_view` (a depth attachment) into `view` (its
    /// R32Float guest-depth reader), in the encoding `gxm_depth_encoding` names, with the
    /// window-depth fit measured for the pass at colour address `fit_addr`.
    #[allow(clippy::too_many_arguments)]
    pub(super) fn encode_depth_convert(
        &mut self,
        queue: &wgpu::Queue,
        device: &wgpu::Device,
        encoder: &mut wgpu::CommandEncoder,
        fit_addr: u32,
        src_view: &wgpu::TextureView,
        view: &wgpu::TextureView,
        multisampled: bool,
        depth_min: f32,
        depth_scale: f32,
    ) {
        let (mode, konst) = gxm_depth_encoding();
        // The SAME fit the pass's own fragments use for their window POSITION - the two are
        // one quantity or every comparison between them is meaningless. `fit_by_addr` is
        // the recompiler's own map and is keyed by the target ADDRESS, so this hands it the
        // address half of the key; giving that map an extent is a separate change.
        let (fit_a, fit_c) = self.gxp.depth_fit_for(fit_addr);
        let mut u = Vec::with_capacity(GXM_DEPTH_UNIFORM_BYTES as usize);
        u.extend_from_slice(&depth_min.to_le_bytes());
        u.extend_from_slice(&depth_scale.to_le_bytes());
        u.extend_from_slice(&mode.to_le_bytes());
        u.extend_from_slice(&konst.to_le_bytes());
        u.extend_from_slice(&fit_a.to_le_bytes());
        u.extend_from_slice(&fit_c.to_le_bytes());
        u.extend_from_slice(&0f32.to_le_bytes());
        u.extend_from_slice(&0f32.to_le_bytes());
        enc(&ENC.depth_converts, 1);
        enc(&ENC.buffer_bytes, u.len() as u64);
        // A whole extra pass, its own bind group, its own draw - all of it per converted
        // depth surface per frame, and none of it predicted by the frame's draw count.
        enc(&ENC.bind_groups_built, 1);
        enc(&ENC.passes, 1);
        enc(&ENC.pipeline_sets, 1);
        enc(&ENC.bind_group_sets, 1);
        enc(&ENC.draw_calls, 1);
        queue.write_buffer(&self.gxm_depth_uniform, 0, &u);
        // A multisampled source is read by the `textureLoad` twin, which takes no sampler.
        // `ensure_rtt` grants a depth-sampled target its multisampled attachments only where
        // that twin exists, so a multisampled source without it cannot occur.
        let (layout, pipe) = match (&self.gxm_depth_ms, multisampled) {
            (Some((l, p)), true) => (l, p),
            _ => (&self.gxm_depth_layout, &self.gxm_depth_pipe),
        };
        let point = wgpu::BindGroupEntry { binding: 2, resource: wgpu::BindingResource::Sampler(&self.sampler_point) };
        let entries = [
            wgpu::BindGroupEntry { binding: 0, resource: wgpu::BindingResource::TextureView(src_view) },
            wgpu::BindGroupEntry { binding: 1, resource: self.gxm_depth_uniform.as_entire_binding() },
            point,
        ];
        let bind = device.create_bind_group(&wgpu::BindGroupDescriptor {
            label: Some("gxm-depth-convert-bind"),
            layout,
            entries: if multisampled { &entries[..2] } else { &entries[..] },
        });
        let pipe = pipe.clone();
        let view = view.clone();
        {
            let ts = self.ts_pair(|| "depth-convert".to_string());
            let mut pass = encoder.begin_render_pass(&wgpu::RenderPassDescriptor {
                label: Some("gxm-depth-convert"),
                color_attachments: &[Some(wgpu::RenderPassColorAttachment {
                    view: &view,
                    depth_slice: None,
                    resolve_target: None,
                    ops: wgpu::Operations {
                        load: wgpu::LoadOp::Clear(wgpu::Color::TRANSPARENT),
                        store: wgpu::StoreOp::Store,
                    },
                })],
                depth_stencil_attachment: None,
                timestamp_writes: Self::ts_writes(&ts),
                occlusion_query_set: None,
                multiview_mask: None,
            });
            pass.set_pipeline(&pipe);
            pass.set_bind_group(0, &bind, &[]);
            pass.draw(0..3, 0..1);
        }
        report_depth_conversion(fit_addr, mode, konst, depth_min, depth_scale);
    }

    /// >>> A SEPARATE STENCIL PLANE A LATER PASS SAMPLES IS CONVERTED AFTER THE PASS.
    ///
    /// A title can read its scene's stencil back as an ordinary U8 texture at the plane's
    /// address - a post pass that treats what the scene MARKED differently from the rest.
    /// The stencil only ever lives in our depth-stencil attachment, so without this the
    /// sampler decoded the guest bytes at that address, which no pass writes. MEASURED on
    /// an action title: its final tone-map samples the world's 720x408 stencil plane
    /// (`1RRR`) and read all zeros, and the frame came out blown out with a black sky.
    ///
    /// The stencil aspect is loaded as an integer and written as the RGBA8 the CPU decode of
    /// those same bytes would give (`render::swizzle1`, high-to-low), so the sampler sees
    /// exactly what it would have read from memory. Single-sampled attachments only: a
    /// multisampled one is reported and left to the guest bytes.
    pub(super) fn convert_gxm_stencil(
        &mut self,
        device: &wgpu::Device,
        encoder: &mut wgpu::CommandEncoder,
        depth: &wgpu::Texture,
        stencil_addr: u32,
    ) {
        let Some(&selector) = self.stencil_sampled.get(&stencil_addr) else { return };
        if !matches!(depth.format(), wgpu::TextureFormat::Depth24PlusStencil8 | wgpu::TextureFormat::Depth32FloatStencil8) {
            return;
        }
        if depth.sample_count() > 1 {
            report_stencil_sample_unserved(stencil_addr, "the pass that writes it is MULTISAMPLED and a per-sample stencil conversion is not established");
            return;
        }
        // A bind group over an unbindable view fails VALIDATION, and an invalid group poisons
        // the whole submit - a black frame for one conversion. Every caller creates the
        // attachment bindable when its stencil is sampled; this is the guard if one does not.
        if !depth.usage().contains(wgpu::TextureUsages::TEXTURE_BINDING) {
            report_stencil_sample_unserved(stencil_addr, "its depth-stencil attachment was created without TEXTURE_BINDING");
            return;
        }
        let (w, h) = (depth.width(), depth.height());
        if self.stencil_planes.get(&stencil_addr).is_none_or(|p| (p.2, p.3) != (w, h)) {
            if let Some((old, ..)) = self.stencil_planes.remove(&stencil_addr) {
                self.retired_textures.push(old);
            }
            let tex = device.create_texture(&wgpu::TextureDescriptor {
                label: Some("gxm-stencil-plane"),
                size: wgpu::Extent3d { width: w, height: h, depth_or_array_layers: 1 },
                mip_level_count: 1,
                sample_count: 1,
                dimension: wgpu::TextureDimension::D2,
                format: wgpu::TextureFormat::Rgba8Unorm,
                usage: wgpu::TextureUsages::RENDER_ATTACHMENT
                    | wgpu::TextureUsages::TEXTURE_BINDING
                    | wgpu::TextureUsages::COPY_SRC,
                view_formats: &[],
            });
            let view = tex.create_view(&Default::default());
            self.stencil_planes.insert(stencil_addr, (tex, view, w, h));
        }
        let (layout, pipe) = self
            .stencil_convert
            .entry(selector)
            .or_insert_with(|| stencil_convert_pipeline(device, selector));
        let src = depth.create_view(&wgpu::TextureViewDescriptor {
            label: Some("gxm-stencil-sampled"),
            aspect: wgpu::TextureAspect::StencilOnly,
            ..Default::default()
        });
        let bind = device.create_bind_group(&wgpu::BindGroupDescriptor {
            label: Some("gxm-stencil-convert-bind"),
            layout,
            entries: &[wgpu::BindGroupEntry { binding: 0, resource: wgpu::BindingResource::TextureView(&src) }],
        });
        let pipe = pipe.clone();
        let view = self.stencil_planes[&stencil_addr].1.clone();
        enc(&ENC.passes, 1);
        enc(&ENC.bind_groups_built, 1);
        enc(&ENC.draw_calls, 1);
        {
            let mut pass = encoder.begin_render_pass(&wgpu::RenderPassDescriptor {
                label: Some("gxm-stencil-convert"),
                color_attachments: &[Some(wgpu::RenderPassColorAttachment {
                    view: &view,
                    depth_slice: None,
                    resolve_target: None,
                    ops: wgpu::Operations { load: wgpu::LoadOp::Clear(wgpu::Color::TRANSPARENT), store: wgpu::StoreOp::Store },
                })],
                depth_stencil_attachment: None,
                timestamp_writes: None,
                occlusion_query_set: None,
                multiview_mask: None,
            });
            pass.set_pipeline(&pipe);
            pass.set_bind_group(0, &bind, &[]);
            pass.draw(0..3, 0..1);
        }
        self.rtt_depth_rendered.insert(stencil_addr, view);
        report_stencil_plane_sampled(stencil_addr, w, h, selector);
    }

    /// Copy the rendered target `key` names into a side texture and return a view of it.
    ///
    /// A pass is entitled to render into a buffer it also samples - a post-process
    /// stage reading the previous stage in place is the common shape - but a texture
    /// Make sure [`Self::dest_color`] is a texture this pass can copy its colour attachment
    /// into: same extent, same format. Recreated (and the generation bumped, which
    /// invalidates every cached group-3 bind group naming the old view) whenever any of
    /// those change.
    pub(super) fn ensure_dest_color(
        &mut self,
        device: &wgpu::Device,
        width: u32,
        height: u32,
        format: wgpu::TextureFormat,
        samples: u32,
    ) {
        if samples > 1 {
            self.dest_ms_copy.entry((format, samples)).or_insert_with(|| dest_ms_copy_pipeline(device, format, samples));
        }
        if self.dest_colors.contains_key(&(width, height, format, samples)) {
            return;
        }
        // Bounded, and DESTROYED on the way out. One entry per attachment SHAPE that
        // carries a self-blending draw is a handful on every title measured here, but a
        // title that resizes its targets would otherwise grow this for the life of the run,
        // and on the browser a dropped texture is memory the collector releases whenever it
        // likes [[vitaslop-browser-gpu-needs-destroy]]. Cleared wholesale rather than by
        // age: every entry rebuilds byte-identical from its key, and the generation bump
        // below is what keeps a cached bind group from naming one of the dead views.
        const DEST_COLOR_CAP: usize = 8;
        if self.dest_colors.len() >= DEST_COLOR_CAP {
            for (_, d) in self.dest_colors.drain() {
                d.texture.destroy();
            }
        }
        let texture = device.create_texture(&wgpu::TextureDescriptor {
            label: Some("gxp-dest-color"),
            size: wgpu::Extent3d { width, height, depth_or_array_layers: 1 },
            mip_level_count: 1,
            sample_count: samples,
            dimension: wgpu::TextureDimension::D2,
            format,
            // COPY_DST for the mid-pass copy, TEXTURE_BINDING for the fragment read. A
            // multisampled one is RENDERED into instead (see `copy_attachment_to_dest`).
            usage: if samples > 1 {
                wgpu::TextureUsages::RENDER_ATTACHMENT | wgpu::TextureUsages::TEXTURE_BINDING
            } else {
                wgpu::TextureUsages::COPY_DST | wgpu::TextureUsages::TEXTURE_BINDING
            },
            view_formats: &[],
        });
        let view = texture.create_view(&wgpu::TextureViewDescriptor::default());
        self.dest_gen += 1;
        let generation = self.dest_gen;
        self.dest_colors
            .insert((width, height, format, samples), DestColor { texture, view, width, height, samples, generation });
        report_status!(
            "gxp dest colour: a fragment program in this pass READS THE DESTINATION COLOUR (it blends for itself in ALU rather than through a patched-in blend equation), so a {width}x{height} {format:?} x{samples} copy of the attachment is taken before each such draw. WebGPU has no framebuffer fetch; every copy is a render-pass split."
        );
    }

    pub(super) fn copy_attachment_to_dest(
        &self,
        device: &wgpu::Device,
        encoder: &mut wgpu::CommandEncoder,
        source: &wgpu::Texture,
        shape: (u32, u32, wgpu::TextureFormat, u32),
        // `Some([x0, y0, x1, y1))`: only this rectangle is stale in the copy (see
        // `DestDirty`), so only it is copied - an empty one copies nothing. `None`: everything.
        // Ignored for a multisampled copy, which is always whole.
        region: Option<[u32; 4]>,
    ) {
        let Some(d) = self.dest_colors.get(&shape) else { return };
        // >>> A MULTISAMPLED ATTACHMENT IS COPIED SAMPLE FOR SAMPLE BY A DRAW. A texture
        // copy of a multisampled source is not portable (Metal has none), and a resolve is
        // not what the ROP feeds back - each sample's program reads that SAMPLE's colour.
        // One fullscreen triangle shading per sample (`sample_index`) loads the source's
        // matching sample, so the copy is exact.
        if d.samples > 1 {
            let (format, samples) = (shape.2, shape.3);
            let dst_view = d.view.clone();
            // Built by `ensure_dest_color`, before the pass borrows the renderer.
            let Some((pipeline, layout)) = self.dest_ms_copy.get(&(format, samples)) else { return };
            let src_view = source.create_view(&wgpu::TextureViewDescriptor { format: Some(format), ..Default::default() });
            let bg = device.create_bind_group(&wgpu::BindGroupDescriptor {
                label: Some("gxp-dest-ms-copy"),
                layout,
                entries: &[wgpu::BindGroupEntry { binding: 0, resource: wgpu::BindingResource::TextureView(&src_view) }],
            });
            let mut pass = encoder.begin_render_pass(&wgpu::RenderPassDescriptor {
                label: Some("gxp-dest-ms-copy"),
                color_attachments: &[Some(wgpu::RenderPassColorAttachment {
                    view: &dst_view,
                    depth_slice: None,
                    resolve_target: None,
                    ops: wgpu::Operations { load: wgpu::LoadOp::Load, store: wgpu::StoreOp::Store },
                })],
                depth_stencil_attachment: None,
                timestamp_writes: None,
                occlusion_query_set: None,
                multiview_mask: None,
            });
            pass.set_pipeline(pipeline);
            pass.set_bind_group(0, &bg, &[]);
            pass.draw(0..3, 0..1);
            return;
        }
        let [x0, y0, x1, y1] = match region {
            Some([x0, y0, x1, y1]) => [x0.min(d.width), y0.min(d.height), x1.min(d.width), y1.min(d.height)],
            None => [0, 0, d.width, d.height],
        };
        if x0 >= x1 || y0 >= y1 {
            return;
        }        let origin = wgpu::Origin3d { x: x0, y: y0, z: 0 };
        encoder.copy_texture_to_texture(
            wgpu::TexelCopyTextureInfo { texture: source, mip_level: 0, origin, aspect: wgpu::TextureAspect::All },
            wgpu::TexelCopyTextureInfo { texture: &d.texture, mip_level: 0, origin, aspect: wgpu::TextureAspect::All },
            wgpu::Extent3d { width: x1 - x0, height: y1 - y0, depth_or_array_layers: 1 },
        );
    }

    /// >>> AN F11F11F10 SURFACE HAS NO ALPHA, SO WHAT SAMPLES IT READS 1.0.
    ///
    /// It is held in an `Rgba16Float` attachment (for its range - see `float_color_format`),
    /// which KEEPS whatever alpha the fragment wrote. A title whose shader then samples the
    /// surface and uses that alpha sees a value the console never shows it. So after every
    /// pass into such a target its stored alpha is set to 1.0, which is what the console's
    /// sampler returns for the format.
    pub(super) fn fill_alpha_one(&mut self, device: &wgpu::Device, encoder: &mut wgpu::CommandEncoder, key: (u32, u32, u32)) {
        let Some(t) = self.rtt.get(&key) else { return };
        let format = t.color.format();
        let view = t.color_view.clone();
        let pipeline = self.alpha_fill.entry(format).or_insert_with(|| alpha_fill_pipeline(device, format));
        let mut pass = encoder.begin_render_pass(&wgpu::RenderPassDescriptor {
            label: Some("gxm-alpha-fill"),
            color_attachments: &[Some(wgpu::RenderPassColorAttachment {
                view: &view,
                depth_slice: None,
                resolve_target: None,
                ops: wgpu::Operations { load: wgpu::LoadOp::Load, store: wgpu::StoreOp::Store },
            })],
            depth_stencil_attachment: None,
            timestamp_writes: None,
            occlusion_query_set: None,
            multiview_mask: None,
        });
        pass.set_pipeline(pipeline);
        pass.draw(0..3, 0..1);
    }

    /// >>> A MULTISAMPLED PASS THAT LOADS STARTS FROM THE TARGET'S MEMORY, NOT FROM ITS OLD
    /// >>> SAMPLES.
    ///
    /// The tiler loads the colour surface at `sceGxmBeginScene`, and for an MSAA_DOWNSCALE
    /// surface that memory is the RESOLVED image - including whatever a single-sample scene
    /// drew into the same address since. A multisampled attachment kept across such a scene
    /// (see `RttSurface::samples_now`) still holds its own last samples; loading them instead
    /// accumulates every blended layer frame after frame. MEASURED on a pool title: a 4x
    /// world scene and a 1x HUD scene share one target, and the cue ball's blended
    /// reflection layers stacked up over stale samples into a blown-out, blocky white ball.
    /// So before such a pass, every sample is filled from `color`.
    pub(super) fn seed_msaa_from_color(&mut self, device: &wgpu::Device, encoder: &mut wgpu::CommandEncoder, key: (u32, u32, u32)) {
        let Some(t) = self.rtt.get(&key) else { return };
        let Some(m) = t.msaa.as_ref() else { return };
        let format = t.color.format();
        let samples = m.samples;
        let src_view = t.color_view.clone();
        let dst_view = m.color_view.clone();
        let (pipeline, layout) = self
            .msaa_seed
            .entry((format, samples))
            .or_insert_with(|| msaa_seed_pipeline(device, format, samples));
        let bg = device.create_bind_group(&wgpu::BindGroupDescriptor {
            label: Some("gxm-msaa-seed"),
            layout,
            entries: &[wgpu::BindGroupEntry { binding: 0, resource: wgpu::BindingResource::TextureView(&src_view) }],
        });
        let mut pass = encoder.begin_render_pass(&wgpu::RenderPassDescriptor {
            label: Some("gxm-msaa-seed"),
            color_attachments: &[Some(wgpu::RenderPassColorAttachment {
                view: &dst_view,
                depth_slice: None,
                resolve_target: None,
                ops: wgpu::Operations { load: wgpu::LoadOp::Load, store: wgpu::StoreOp::Store },
            })],
            depth_stencil_attachment: None,
            timestamp_writes: None,
            occlusion_query_set: None,
            multiview_mask: None,
        });
        pass.set_pipeline(pipeline);
        pass.set_bind_group(0, &bg, &[]);
        pass.draw(0..3, 0..1);
    }

    /// >>> THE ON-CHIP PIXEL DATA OF AN 8-BIT SCENE, AT THE PRECISION ITS PROGRAMS LEAVE THERE.
    ///
    /// A GXM scene does not render into its colour surface. Its pixels live in the tile's
    /// OUTPUT REGISTERS until the tile is written out, and only then does the pixel back end
    /// pack them into the surface's format (sRGB-encoding them on the way, for a gamma
    /// surface). A program compiled for an 8-bit surface usually packs its colour itself
    /// (the `pack.unorm8` / `mov.fx8` epilogue) and then the register already holds the
    /// surface's own form. One that leaves HALVES there and READS THE REGISTER BACK - blends
    /// for itself - sees the halves the previous draw left, not an 8-bit rounding of them.
    ///
    /// Drawing such a scene straight into its 8-bit attachment rounds after every draw
    /// instead. MEASURED on a shooter's light glows: four soft sprites summed into a bloom
    /// buffer at linear values below 0.1, each rounded to 1/255, left the glow 26 levels deep
    /// per channel; a later pass sRGB-encodes it and the tone map pushes it through a steep
    /// LUT, and the glow came out as hard concentric bands, red outermost then orange, yellow
    /// and white, stair-stepped along the bloom buffer's texels. On the console it is smooth.
    ///
    /// So such a scene is rendered into this `Rgba16Float` texture - seeded from the surface
    /// (or cleared) as the tile is, blended into at half precision by every draw, copied to a
    /// destination texture for the self-blending draws - and resolved into the real attachment
    /// once, at the end, where the pack and the sRGB encode belong. One per attachment extent.
    pub(super) fn tile_shadow(&mut self, device: &wgpu::Device, width: u32, height: u32) -> (wgpu::Texture, wgpu::TextureView) {
        if let Some((t, v)) = self.tile_shadows.get(&(width, height)) {
            return (t.clone(), v.clone());
        }
        // Bounded: one extent per qualifying target is a handful. An evicted texture may still
        // be named by a pass this frame's encoder holds, so it is RETIRED, not destroyed - see
        // `retired_textures`.
        const TILE_SHADOW_CAP: usize = 4;
        if self.tile_shadows.len() >= TILE_SHADOW_CAP {
            let old: Vec<wgpu::Texture> = self.tile_shadows.drain().map(|(_, (t, _))| t).collect();
            self.retired_textures.extend(old);
        }
        let texture = device.create_texture(&wgpu::TextureDescriptor {
            label: Some("gxm-tile-shadow"),
            size: wgpu::Extent3d { width, height, depth_or_array_layers: 1 },
            mip_level_count: 1,
            sample_count: 1,
            dimension: wgpu::TextureDimension::D2,
            format: FLOAT_TARGET_FORMAT,
            // RENDER_ATTACHMENT for the scene, TEXTURE_BINDING for the resolve, COPY_SRC for
            // the destination copies of the self-blending draws.
            usage: wgpu::TextureUsages::RENDER_ATTACHMENT
                | wgpu::TextureUsages::TEXTURE_BINDING
                | wgpu::TextureUsages::COPY_SRC,
            view_formats: &[],
        });
        let view = texture.create_view(&wgpu::TextureViewDescriptor::default());
        self.tile_shadows.insert((width, height), (texture.clone(), view.clone()));
        report_status!(
            "gxm tile shadow: an 8-bit scene's programs blend at HALF precision, so a {width}x{height} {FLOAT_TARGET_FORMAT:?} copy of its on-chip pixel data carries the scene and is packed into the surface once, at the end"
        );
        (texture, view)
    }

    /// Build the [`GxmRenderer::texel_blit`] pipeline for `dst_format`, ahead of the draw loop
    /// that has the renderer borrowed.
    pub(super) fn ensure_texel_blit(&mut self, device: &wgpu::Device, dst_format: wgpu::TextureFormat) {
        self.msaa_seed.entry((dst_format, 1)).or_insert_with(|| msaa_seed_pipeline(device, dst_format, 1));
    }

    /// Write `src` texel for texel into `dst` (a full-attachment draw through `dst`'s format:
    /// an 8-bit `dst` rounds, an sRGB view encodes, an sRGB `src` view decodes). The seed and
    /// the resolve of [`GxmRenderer::tile_shadow`]. Needs [`GxmRenderer::ensure_texel_blit`]
    /// first; without it nothing is written, which the caller's `ensure` rules out.
    pub(super) fn texel_blit(
        &self,
        device: &wgpu::Device,
        encoder: &mut wgpu::CommandEncoder,
        src: &wgpu::TextureView,
        dst: &wgpu::TextureView,
        dst_format: wgpu::TextureFormat,
    ) {
        let Some((pipeline, layout)) = self.msaa_seed.get(&(dst_format, 1)) else { return };
        let bg = device.create_bind_group(&wgpu::BindGroupDescriptor {
            label: Some("gxm-tile-blit"),
            layout,
            entries: &[wgpu::BindGroupEntry { binding: 0, resource: wgpu::BindingResource::TextureView(src) }],
        });
        let mut pass = encoder.begin_render_pass(&wgpu::RenderPassDescriptor {
            label: Some("gxm-tile-blit"),
            color_attachments: &[Some(wgpu::RenderPassColorAttachment {
                view: dst,
                depth_slice: None,
                resolve_target: None,
                ops: wgpu::Operations { load: wgpu::LoadOp::Load, store: wgpu::StoreOp::Store },
            })],
            depth_stencil_attachment: None,
            timestamp_writes: None,
            occlusion_query_set: None,
            multiview_mask: None,
        });
        pass.set_pipeline(pipeline);
        pass.set_bind_group(0, &bg, &[]);
        pass.draw(0..3, 0..1);
    }

    /// cannot be a colour target and a sampled resource in the same pass (wgpu rejects
    /// it, and the hardware read would be undefined). The snapshot is the buffer as it
    /// stood BEFORE this pass, which is exactly what such a pass means to read.
    pub(super) fn snapshot_rtt(
        &mut self,
        device: &wgpu::Device,
        encoder: &mut wgpu::CommandEncoder,
        // The full `(address, width, height)` key: the shadow is a texel-for-texel copy of
        // ONE target's colour attachment, and two targets can share an address - see `rtt`.
        key: (u32, u32, u32),
    ) -> Option<wgpu::TextureView> {
        // The shadow is a copy of the target, so it has the TARGET's format - a float target
        // is not the renderer's default format, and a copy between two formats is refused.
        let color_format = self.rtt.get(&key).map(|t| t.color.format()).unwrap_or(self.color_format);
        // The shadow texture is created ONCE per target and copied into thereafter, so a
        // view of it is stable across frames exactly as the target's own view is - but its
        // creation is still a new view, and every cached group has to hear about it.
        let fresh_shadow = self.rtt.get(&key).is_some_and(|t| t.shadow.is_none());
        if fresh_shadow {
            self.rtt_epoch += 1;
        }
        let t = self.rtt.get_mut(&key)?;
        let size = wgpu::Extent3d { width: t.width.max(1), height: t.height.max(1), depth_or_array_layers: 1 };
        if t.shadow.is_none() {
            // The sRGB twin is declared on the snapshot for exactly the reason it is
            // declared on `color`: gamma mode is sticky guest state that can arrive after
            // the texture exists, and view formats are fixed at creation.
            // Same rule as the target's own view: compatibility mode refuses a view of
            // another format, and the snapshot is a copy of that target.
            let srgb_fmt = if crate::gpu::compat_mode() || color_format != self.color_format {
                None
            } else {
                srgb_twin(color_format)
            };
            let view_formats: Vec<wgpu::TextureFormat> = srgb_fmt.into_iter().collect();
            let tex = device.create_texture(&wgpu::TextureDescriptor {
                label: Some("gxm-rtt-shadow"),
                size,
                mip_level_count: 1,
                sample_count: 1,
                dimension: wgpu::TextureDimension::D2,
                format: color_format,
                usage: wgpu::TextureUsages::TEXTURE_BINDING | wgpu::TextureUsages::COPY_DST,
                view_formats: &view_formats,
            });
            let view = tex.create_view(&Default::default());
            let view_srgb = srgb_fmt.map(|f| {
                tex.create_view(&wgpu::TextureViewDescriptor {
                    label: Some("gxm-rtt-shadow-srgb"),
                    format: Some(f),
                    ..Default::default()
                })
            });
            t.shadow = Some((tex, view, view_srgb));
        }
        let gamma = t.gamma;
        let (shadow_tex, shadow_view, shadow_view_srgb) = t.shadow.as_ref()?;
        // The view the pass that RENDERED this target would hand out. `encode_chain` picks
        // the sRGB view for a gamma target and `sample_views` mirrors it; a snapshot of the
        // same bytes has to make the same choice or the two disagree about what the bytes
        // mean.
        let shadow_view = match (gamma, shadow_view_srgb.as_ref()) {
            (true, Some(v)) => v,
            _ => shadow_view,
        };
        enc(&ENC.rtt_snapshots, 1);
        enc(&ENC.rtt_snapshot_bytes, texture_bytes(size.width, size.height) as u64);
        encoder.copy_texture_to_texture(
            wgpu::TexelCopyTextureInfo {
                texture: &t.color,
                mip_level: 0,
                origin: wgpu::Origin3d::ZERO,
                aspect: wgpu::TextureAspect::All,
            },
            wgpu::TexelCopyTextureInfo {
                texture: shadow_tex,
                mip_level: 0,
                origin: wgpu::Origin3d::ZERO,
                aspect: wgpu::TextureAspect::All,
            },
            size,
        );
        Some(shadow_view.clone())
    }

    /// Copy the kept DISPLAY IMAGE at `addr` into its snapshot and hand back the view a
    /// sampler should bind - the display arm's [`Self::snapshot_rtt`].
    ///
    /// >>> A DISPLAY PASS THAT SAMPLES ITS OWN BUFFER READ GUEST BYTES THE GPU NEVER WROTE.
    ///
    /// The sampler map never offers the attachment a pass is writing (binding it as a
    /// texture in the same pass is invalid), and the display image had no snapshot to offer
    /// instead, so the read fell through to the guest's memory - zero, because the GPU drew
    /// that picture. MEASURED on a football title: a frame whose world scene and a second
    /// scene both draw into the display buffer, the second opening with a full-screen copy
    /// (`src ONE, dst ZERO`) OF THAT BUFFER - the world rendered correctly and the copy
    /// painted it black, HUD on top (native capsule f17640). Its instant replay draws the
    /// same way every frame, so the whole replay was black behind its panel.
    pub(super) fn snapshot_display(
        &mut self,
        device: &wgpu::Device,
        encoder: &mut wgpu::CommandEncoder,
        addr: u32,
    ) -> Option<wgpu::TextureView> {
        let srgb_fmt = if crate::gpu::compat_mode() { None } else { srgb_twin(self.color_format) };
        let color_format = self.color_format;
        let d = self.display_images.get_mut(&addr)?;
        let size = d.tex.size();
        if d.shadow.as_ref().is_none_or(|(s, _, _)| s.size() != size) {
            if let Some((s, _, _)) = d.shadow.take() {
                s.destroy();
            }
            // The image's own format and the same declared sRGB twin, so the snapshot
            // decodes exactly as the image would - see `DisplayImage::gamma`.
            let view_formats: Vec<wgpu::TextureFormat> = srgb_fmt.into_iter().collect();
            let tex = device.create_texture(&wgpu::TextureDescriptor {
                label: Some("gxm-display-shadow"),
                size,
                mip_level_count: 1,
                sample_count: 1,
                dimension: wgpu::TextureDimension::D2,
                format: color_format,
                usage: wgpu::TextureUsages::TEXTURE_BINDING | wgpu::TextureUsages::COPY_DST,
                view_formats: &view_formats,
            });
            let view = tex.create_view(&Default::default());
            let view_srgb = srgb_fmt.map(|format| {
                tex.create_view(&wgpu::TextureViewDescriptor {
                    label: Some("gxm-display-shadow-srgb"),
                    format: Some(format),
                    ..Default::default()
                })
            });
            d.shadow = Some((tex, view, view_srgb));
        }
        let (shadow_tex, shadow_view, shadow_view_srgb) = d.shadow.as_ref()?;
        let view = match (d.gamma, shadow_view_srgb.as_ref()) {
            (true, Some(v)) => v,
            _ => shadow_view,
        };
        enc(&ENC.rtt_snapshots, 1);
        enc(&ENC.rtt_snapshot_bytes, texture_bytes(size.width, size.height) as u64);
        encoder.copy_texture_to_texture(
            wgpu::TexelCopyTextureInfo {
                texture: &d.tex,
                mip_level: 0,
                origin: wgpu::Origin3d::ZERO,
                aspect: wgpu::TextureAspect::All,
            },
            wgpu::TexelCopyTextureInfo {
                texture: shadow_tex,
                mip_level: 0,
                origin: wgpu::Origin3d::ZERO,
                aspect: wgpu::TextureAspect::All,
            },
            size,
        );
        Some(view.clone())
    }

    /// Ensure a fixed-function bind group exists for sampling `view` - the rendered
    /// target at `addr`, or its snapshot - with the given filter.
    /// Assemble, into `self.rtt_cubes`, every cube map whose six faces this frame has now
    /// rendered - copying each face's target into the matching array layer of one cube
    /// texture, on the caller's encoder.
    ///
    /// # Why the six faces are found by ARITHMETIC PROGRESSION and not by byte arithmetic
    /// A cube's faces sit back to back from the texture's `data_addr`, so face `k` is at
    /// `base + k*stride` where `stride` is one face's size in GUEST bytes. Deriving that
    /// stride needs the guest surface's bytes-per-texel, which an `RttSurface` does not
    /// record - and getting it wrong would silently assemble a cube out of six unrelated
    /// buffers. The render set answers it directly instead: the smallest `stride` for which
    /// ALL SIX of `base + k*stride` were rendered this frame, with all six the same size and
    /// format. Six independent addresses agreeing on one spacing is not a coincidence any
    /// unrelated set of targets produces, and requiring all six is what makes it safe -
    /// five faces and a near miss assembles nothing.
    ///
    /// Called BETWEEN passes, so the copies land after the passes that drew the faces and
    /// before any pass that samples the cube. Recorded on the frame's own encoder for the
    /// same reason: a separately submitted copy would run before the faces were drawn.
    pub(super) fn assemble_rendered_cubes(
        &mut self,
        device: &wgpu::Device,
        encoder: &mut wgpu::CommandEncoder,
        bases: &HashSet<u32>,
    ) {
        for &base in bases {
            if self.assemble_stacked_cube(device, encoder, base) {
                continue;
            }
            // >>> THE FACE EXTENTS RENDERED AT THIS ADDRESS OVER THE RUN, LARGEST FIRST.
            //
            // OVER THE RUN - `rtt_ever_rendered` - and NOT this frame's `rtt_rendered`.
            // MEASURED on the baseball title's reflection probe (`0x8d0f0d00`, six 128x128
            // faces 0x10000 apart): the six faces are rendered TOGETHER exactly once, at
            // boot, empty, and after that ONE FACE PER FRAME at scene load with the stadium
            // in them - never six in one frame again. Detecting the cube from a per-frame
            // set assembled it once, at boot, and never refreshed it, so the guest's own
            // auto-exposure read a black probe for the whole game and raised its exposure
            // 3.3x - the close-up wash-out. The guest's memory holds a face as soon as its
            // pass completes, and the cube here has to track that face by face.
            //
            // `rtt_rendered` is keyed by `(address, width, height)` now - see `rtt` for the
            // baseball title that runs two render targets of different extents through one
            // colour address - so "was face 0 rendered" no longer has a single answer. A
            // cube face is SQUARE by construction, so anything else here is not the layout
            // this is looking for and is left alone rather than guessed at; each remaining
            // candidate is tried in turn and the first that finds all six faces wins. No
            // title measured renders two square targets at one address, so this runs at
            // most once in practice - it exists so the answer cannot depend on hash order
            // if one ever does.
            let mut sizes: Vec<u32> = self
                .rtt_ever_rendered
                .iter()
                .filter(|(a, w, h)| *a == base && *w == *h && *w != 0)
                .map(|(_, w, _)| *w)
                .collect();
            sizes.sort_unstable_by(|a, b| b.cmp(a));
            for size in sizes {
            let Some(f0) = self.rtt.get(&(base, size, size)) else { continue };
            let format = f0.color.format();
            // >>> WHETHER THE FACES HOLD sRGB-ENCODED BYTES, which decides how the cube is
            // VIEWED. `copy_texture_to_texture` moves bytes, so a copy of a gamma-correct
            // target carries the ROP's encoding with it and a sampler has to decode on the
            // way back in - exactly the rule `sample_views` states for the 2D path, and
            // exactly the mistake that made an accumulation buffer skew itself to white
            // there. Reading these as linear would return every reflection too bright.
            let gamma = f0.gamma;
            // Candidate spacings, smallest first: every target ever rendered AT THIS FACE
            // EXTENT above the base. The extent is part of the key, so a target that cannot
            // be a face of this cube is rejected here rather than in the test below.
            let mut strides: Vec<u32> = self
                .rtt_ever_rendered
                .iter()
                .filter(|(a, w, h)| *a > base && *w == size && *h == size)
                .map(|(a, _, _)| a - base)
                .collect();
            strides.sort_unstable();
            let Some(stride) = strides.into_iter().find(|&s| {
                (1..CUBE_FACES).all(|k| {
                    let a = base.wrapping_add(s * k);
                    self.rtt_ever_rendered.contains(&(a, size, size))
                        && self.rtt.get(&(a, size, size)).is_some_and(|f| {
                            f.color.format() == format && f.gamma == gamma
                        })
                })
            }) else {
                continue;
            };
            // >>> AND ITS SMALLER MIP LEVELS, WHICH THE GUEST RENDERS TOO. A cube declared
            // with mips is sampled at a level-of-detail - a rough surface reads a small, blurred
            // level - and a cube built with mip 0 alone answers every such lookup with the
            // sharp face, highlights and all. MEASURED on a pool title: its reflection cube is
            // six 64x64 faces 0x5800 apart, each followed in its block by a 32x32 and a 16x16
            // render; with only mip 0 assembled every ball reflected the room's 8.0 strip
            // lights at full strength and rendered washed-out. Level `m` is the `size >> m`
            // target found inside face 0's block, at an offset all six faces share.
            let mut mips: Vec<u32> = Vec::new();
            let mut lsize = size >> 1;
            while lsize >= 1 {
                let found = self
                    .rtt_ever_rendered
                    .iter()
                    .filter(|(a, w, h)| *w == lsize && *h == lsize && *a > base && *a < base + stride)
                    .map(|(a, _, _)| a - base)
                    .filter(|off| {
                        (1..CUBE_FACES).all(|k| {
                            let a = base.wrapping_add(stride * k + off);
                            self.rtt.get(&(a, lsize, lsize)).is_some_and(|f| f.color.format() == format && f.gamma == gamma)
                        })
                    })
                    .min();
                let Some(off) = found else { break };
                mips.push(off);
                lsize >>= 1;
            }
            // Rebuild when the shape changed (or on the first sight of this cube); a copy
            // requires the extents and formats to match exactly.
            let stale = self.rtt_cubes.get(&base).is_none_or(|c| {
                c.size != size || c.format != format || c.gamma != gamma || c.mips != mips
            });
            if stale {
                // The sRGB twin, when the guest put these faces in gamma-correct mode. A
                // texture's view formats are fixed at creation, so it is declared here.
                let srgb = if gamma && !crate::gpu::compat_mode() { srgb_twin(format) } else { None };
                let view_formats: Vec<wgpu::TextureFormat> = srgb.into_iter().collect();
                let view_format = srgb;
                let tex = device.create_texture(&wgpu::TextureDescriptor {
                    label: Some("gxm-rtt-cube"),
                    size: wgpu::Extent3d {
                        width: size,
                        height: size,
                        depth_or_array_layers: CUBE_FACES,
                    },
                    mip_level_count: 1 + mips.len() as u32,
                    sample_count: 1,
                    dimension: wgpu::TextureDimension::D2,
                    format,
                    usage: wgpu::TextureUsages::COPY_DST | wgpu::TextureUsages::TEXTURE_BINDING,
                    view_formats: &view_formats,
                });
                let view = tex.create_view(&wgpu::TextureViewDescriptor {
                    label: Some("gxm-rtt-cube"),
                    format: view_format,
                    dimension: Some(wgpu::TextureViewDimension::Cube),
                    ..Default::default()
                });
                if let Some(old) = self
                    .rtt_cubes
                    .insert(base, CubeFromRenders { tex, view, size, format, gamma, stride, mips: mips.clone(), stacked: false, stack_height: 0 })
                {
                    // >>> THE EPOCH MUST MOVE BEFORE THE OLD TEXTURE DIES. A sampler bind
                    // group naming this cube is CACHED in `sampler_bgs` under a key that
                    // folds `rtt_epoch`, and it holds a view of the texture about to be
                    // destroyed. Without the bump the cache would answer a later draw with
                    // a group over destroyed memory - which is exactly what the epoch
                    // exists for ("bumped when a target or its snapshot texture is created,
                    // i.e. when views die"). Only on a REBUILD: refreshing a cube's
                    // CONTENTS reuses the same texture and invalidates nothing.
                    self.rtt_epoch += 1;
                    old.tex.destroy();
                }
                report_rendered_cube(base, size, stride, format);
            }
            let Some(cube) = self.rtt_cubes.get(&base) else { continue };
            // >>> WHICH FACES TO COPY. A freshly built cube takes all six from the face
            // targets, which persist with their last contents exactly as the guest's
            // memory does. After that, a face is copied in the frame it is RENDERED, once,
            // at the first assembly point after its pass - so a cube sampled every frame
            // and refreshed one face per frame tracks the guest face by face, and a cube
            // nobody re-renders costs nothing per frame.
            for level in 0..=cube.mips.len() as u32 {
                let lsize = size >> level;
                let off = if level == 0 { 0 } else { cube.mips[level as usize - 1] };
                for k in 0..CUBE_FACES {
                    let a = base.wrapping_add(stride * k + off);
                    let key = (a, lsize, lsize);
                    // A face is copied when the cube is (re)built, and otherwise whenever it
                    // has been rendered since its last copy - on THIS frame or on one that
                    // sampled no cube (MEASURED: a pool title's training RESTART re-renders its
                    // reflection probe behind the loading screen, and the balls reflected the
                    // pre-restart levels, washed out, until the end of the event).
                    if !stale && !self.rtt_rendered_uncopied.contains(&key) {
                        continue;
                    }
                    let Some(face) = self.rtt.get(&key) else { continue };
                    encoder.copy_texture_to_texture(
                        wgpu::TexelCopyTextureInfo {
                            texture: &face.color,
                            mip_level: 0,
                            origin: wgpu::Origin3d::ZERO,
                            aspect: wgpu::TextureAspect::All,
                        },
                        wgpu::TexelCopyTextureInfo {
                            texture: &cube.tex,
                            mip_level: level,
                            origin: wgpu::Origin3d { x: 0, y: 0, z: k },
                            aspect: wgpu::TextureAspect::All,
                        },
                        wgpu::Extent3d {
                            width: lsize,
                            height: lsize,
                            depth_or_array_layers: 1,
                        },
                    );
                    self.cube_faces_copied.insert(key);
                    self.rtt_rendered_uncopied.remove(&key);
                }
            }
            // This face extent answered; no other candidate at this address is tried.
            break;
            }
        }
    }

    /// A cube map whose six faces the guest rendered into ONE target, stacked: `w x 6w`,
    /// face `k` in rows `k*w..(k+1)*w`. Assembles it into `self.rtt_cubes` the way
    /// [`Self::assemble_rendered_cubes`] does for six separate targets, and returns whether
    /// `base` is such a cube.
    ///
    /// # Why the stacked target IS the cube, byte for byte
    /// A linear cube of `w x w` texels sits in guest memory as six faces back to back, each
    /// `w*w` texels. A linear `w x 6w` colour surface with a pitch of `w` is the same bytes:
    /// its rows `k*w..(k+1)*w` are exactly face `k`. So the guest draws all six faces in one
    /// scene (a viewport per face) and binds the address as a cube, and on hardware nothing
    /// converts anything.
    ///
    /// MEASURED on a fighting title: at stage load it renders its image-based
    /// lighting into three such targets (`0x93379000` 16x96, `0x9337a800` 16x96,
    /// `0x9337c000` 32x192) with one pair, and every character draw afterwards samples them
    /// as 16x16 / 32x32 CUBES - its diffuse and specular light. Nothing assembled them, the
    /// lookup fell to guest memory (all zero - the GPU wrote the faces, not the guest), and
    /// every character was lit by black.
    pub(super) fn assemble_stacked_cube(
        &mut self,
        device: &wgpu::Device,
        encoder: &mut wgpu::CommandEncoder,
        base: u32,
    ) -> bool {
        // The widest stacked extent ever rendered at this address. AT LEAST six faces tall,
        // not exactly: the title above declares each of these as a 16x128 SURFACE (eight
        // face-heights) and renders the six faces into its first 96 rows, and the target is
        // held at the surface's extent.
        let Some((size, height)) = self
            .rtt_ever_rendered
            .iter()
            .filter(|(a, w, h)| *a == base && *w != 0 && *h >= *w * CUBE_FACES)
            .map(|(_, w, h)| (*w, *h))
            .max()
        else {
            return false;
        };
        let key = (base, size, height);
        let Some(target) = self.rtt.get(&key) else { return false };
        let format = target.color.format();
        let gamma = target.gamma;
        let stale = self.rtt_cubes.get(&base).is_none_or(|c| {
            !c.stacked || c.size != size || c.stack_height != height || c.format != format || c.gamma != gamma
        });
        if stale {
            let srgb = if gamma && !crate::gpu::compat_mode() { srgb_twin(format) } else { None };
            let view_formats: Vec<wgpu::TextureFormat> = srgb.into_iter().collect();
            let tex = device.create_texture(&wgpu::TextureDescriptor {
                label: Some("gxm-rtt-cube-stacked"),
                size: wgpu::Extent3d { width: size, height: size, depth_or_array_layers: CUBE_FACES },
                mip_level_count: 1,
                sample_count: 1,
                dimension: wgpu::TextureDimension::D2,
                format,
                usage: wgpu::TextureUsages::COPY_DST | wgpu::TextureUsages::TEXTURE_BINDING,
                view_formats: &view_formats,
            });
            let view = tex.create_view(&wgpu::TextureViewDescriptor {
                label: Some("gxm-rtt-cube-stacked"),
                format: srgb,
                dimension: Some(wgpu::TextureViewDimension::Cube),
                ..Default::default()
            });
            if let Some(old) = self.rtt_cubes.insert(
                base,
                CubeFromRenders { tex, view, size, format, gamma, stride: 0, mips: Vec::new(), stacked: true, stack_height: height },
            ) {
                // See `assemble_rendered_cubes`: cached sampler groups hold the old view.
                self.rtt_epoch += 1;
                old.tex.destroy();
            }
            report_stacked_cube(base, size, height, format);
        }
        // Copy on first sight, then once per frame the target is re-rendered.
        if !stale && !self.rtt_rendered_uncopied.contains(&key) {
            return true;
        }
        let (Some(cube), Some(target)) = (self.rtt_cubes.get(&base), self.rtt.get(&key)) else {
            return true;
        };
        for k in 0..CUBE_FACES {
            encoder.copy_texture_to_texture(
                wgpu::TexelCopyTextureInfo {
                    texture: &target.color,
                    mip_level: 0,
                    origin: wgpu::Origin3d { x: 0, y: k * size, z: 0 },
                    aspect: wgpu::TextureAspect::All,
                },
                wgpu::TexelCopyTextureInfo {
                    texture: &cube.tex,
                    mip_level: 0,
                    origin: wgpu::Origin3d { x: 0, y: 0, z: k },
                    aspect: wgpu::TextureAspect::All,
                },
                wgpu::Extent3d { width: size, height: size, depth_or_array_layers: 1 },
            );
        }
        self.cube_faces_copied.insert(key);
        self.rtt_rendered_uncopied.remove(&key);
        true
    }

    pub(super) fn ensure_rtt_bind(
        &mut self,
        device: &wgpu::Device,
        addr: u32,
        linear: bool,
        snapshot: bool,
        view: &wgpu::TextureView,
    ) {
        if self.rtt_binds.contains_key(&(addr, linear, snapshot)) {
            enc(&ENC.bind_groups_reused, 1);
            return;
        }
        enc(&ENC.bind_groups_built, 1);
        let samp = if linear { &self.sampler_linear } else { &self.sampler_point };
        let bind = device.create_bind_group(&wgpu::BindGroupDescriptor {
            label: Some("gxm-rtt-bind"),
            layout: &self.texture_layout,
            entries: &[
                wgpu::BindGroupEntry { binding: 0, resource: wgpu::BindingResource::TextureView(view) },
                wgpu::BindGroupEntry { binding: 1, resource: wgpu::BindingResource::Sampler(samp) },
            ],
        });
        self.rtt_binds.insert((addr, linear, snapshot), bind);
    }

    /// Encode a scene that has DEPTH but no colour surface, into a target keyed by its
    /// depth address, and register the converted depth so a later pass can sample it.
    ///
    /// # Why this pass exists at all
    /// A depth prepass, a shadow map or an occlusion pass renders no colour: the whole
    /// product is the depth buffer, and a later pass in the same frame binds it as a
    /// texture. Such a scene has no colour surface, so it used to be reported and dropped -
    /// and the later pass then sampled the depth ADDRESS and found guest bytes the GPU
    /// never wrote, i.e. whatever the CPU last left there. That is not a missing pass on
    /// screen; it is a pass whose absence is laundered into a plausible-looking image by
    /// the pass that reads it, which is the hardest kind of defect to see.
    ///
    /// The colour attachment here is a throwaway sized to the pass, so that the pipelines
    /// built for ordinary passes encode this one unchanged. Its contents are never read:
    /// this address is registered in `rtt_depth_rendered` (the DEPTH map), never in
    /// `rtt_rendered` (the colour one), so no sampler can reach it by accident.
    #[allow(clippy::too_many_arguments)]
    pub(super) fn encode_depth_only_pass(
        &mut self,
        device: &wgpu::Device,
        queue: &wgpu::Queue,
        encoder: &mut wgpu::CommandEncoder,
        scene: &RenderScene,
        width: u32,
        height: u32,
    ) {
        let addr = scene.depth_addr;
        // Keyed by the DEPTH address. A depth surface and a colour surface are allocated
        // near each other but are never the same address, so this cannot collide with a
        // colour target - and `sample_depth` is true by construction, since this pass is
        // only encoded when something samples it.
        //
        // One sample per pixel, unconditionally: a multisampled depth attachment cannot be
        // fed to the conversion pass that matches the guest's own depth buffer, which is
        // the same rule `ensure_rtt` applies to any target whose depth is read.
        self.ensure_rtt(device, addr, width, height, true, 0, false, false, None);
        // The key this pass's target lives under - the depth address AND this pass's
        // extent, exactly as `ensure_rtt` just filed it.
        let key = (addr, width, height);
        let (cv, dv, ctex) = {
            let s = &self.rtt[&key];
            (s.color_view.clone(), s.depth_view.clone(), s.color.clone())
        };
        self.rtt_reads_snapshot.clear();
        self.keep_depth = true;
        self.encode_pass(
            device,
            queue,
            encoder,
            &cv,
            &dv,
            self.color_format,
            scene,
            width,
            height,
            // The attachment is the target: `ensure_rtt` created it at exactly this size.
            width,
            height,
            Some(wgpu::Color::TRANSPARENT),
            1,
            None,
            Some(&ctex),
        );
        self.keep_depth = false;
        self.convert_gxm_depth(device, queue, encoder, key, scene.depth_min, scene.depth_scale);
        if let Some(v) = self.rtt.get(&key).and_then(|s| s.gxm_depth.as_ref()) {
            self.rtt_depth_rendered.insert(addr, v.view.clone());
            // The one case where the depth address IS the address half of the `rtt` key -
            // which is what `depth_only` in `encode_pass` is testing for, so the comparison
            // there stays on the ADDRESS.
            self.rtt_depth_addrs.insert(addr, key);
        }
        // Collected only when the pass is already known to disagree: on every other pass
        // this is dead work, and a depth-only pass can carry hundreds of draws.
        let mut viewports: Vec<(([f32; 6], (i32, i32)), usize)> = Vec::new();
        if scene.depth_extent_ambiguous {
            for v in scene.draws.iter().filter_map(|d| d.gxp.as_ref().map(|g| (g.viewport, g.depth_bias))) {
                match viewports.iter_mut().find(|((p, b), _)| *b == v.1 && p.iter().zip(v.0.iter()).all(|(a, b)| a == b)) {
                    Some((_, n)) => *n += 1,
                    None => viewports.push((v, 1)),
                }
            }
            viewports.sort_by_key(|v| std::cmp::Reverse(v.1));
        }
        report_depth_only_pass(
            addr,
            width,
            height,
            scene.draws.len(),
            scene.depth_extent_ambiguous,
            &viewports,
        );
    }
}

// The plan lives beside the knob in `crate::gpu`: the capture computes the same cover to skip
// reading the bytes the fill overwrites.
use crate::gpu::raw_window_fill_plan;

#[cfg(test)]
mod raw_window_fill_tests {
    use crate::gpu::{WindowCopy, raw_window_fill_plan};

    fn copy(x: u32, y: u32, width: u32, rows: u32, window_offset: u64, bytes_per_row: Option<u32>) -> WindowCopy {
        WindowCopy { x, y, width, rows, window_offset, bytes_per_row }
    }

    /// Four windows one surface row apart, each longer than the rest of the surface - the shape
    /// a depth-reconstruct pass reads a 960x544 F16F16F16F16 target through. Every window is
    /// filled with the rows it covers, as ONE copy at the surface's own pitch, and nothing past
    /// the surface's last row is written.
    #[test]
    fn row_offset_windows_take_every_row_they_cover_in_one_copy() {
        let (base, w, h) = (0x9307_8000u32, 960u32, 544u32);
        let row = w * 8;
        for k in 0..4u32 {
            let plan = raw_window_fill_plan(base + k * row, 2 << 20, base, w, h, w).unwrap().unwrap();
            // 273 whole rows fit the 2 MiB window, and its last 512 bytes are 64 texels of the next.
            let rows = (2u32 << 20) / row;
            let tail = ((2u32 << 20) - rows * row) / 8;
            assert_eq!((rows, tail), (273, 64));
            assert_eq!(plan, vec![copy(0, k, w, rows, 0, Some(row)), copy(0, k + rows, tail, 1, u64::from(rows * row), None)]);
        }
        // A window longer than the surface stops at its last row.
        let plan = raw_window_fill_plan(base, 16 << 20, base, w, h, w).unwrap().unwrap();
        assert_eq!(plan, vec![copy(0, 0, w, h, 0, Some(row))]);
    }

    /// The plan's row-pitch test reads the device constant through a copy the capture can see.
    #[test]
    fn the_row_alignment_is_the_devices() {
        assert_eq!(crate::gpu::WINDOW_COPY_ROW_ALIGN, u64::from(wgpu::COPY_BYTES_PER_ROW_ALIGNMENT));
    }

    /// The cover is exactly the bytes the plan copies: whole rows merge into one range, a
    /// padded pitch leaves each row's padding out, the bytes past the surface stay uncovered,
    /// and a span that splits a texel - which the fill refuses - covers nothing.
    #[test]
    fn the_cover_is_the_bytes_the_fill_writes() {
        let (base, w, h) = (0x9307_8000u32, 960u32, 544u32);
        let row = w * 8;
        let mut out = Vec::new();
        crate::gpu::raw_window_fill_cover(base + row, row * h, &[(base, w, h, w)], &mut out);
        assert_eq!(out, vec![(0, row * (h - 1))]);
        // A pitch of 1024 texels: each row's 960 texels, then 64 texels of padding.
        let prow = 1024 * 8;
        crate::gpu::raw_window_fill_cover(base, prow * 2, &[(base, w, 2, 1024)], &mut out);
        assert_eq!(out, vec![(0, row), (prow, prow + row)]);
        crate::gpu::raw_window_fill_cover(base + 4, row, &[(base, w, h, w)], &mut out);
        assert!(out.is_empty());
    }

    /// A window that starts inside a row takes that row's tail, then whole rows, then the head of
    /// the row it ends in; a pitch wider than the surface leaves the padding unwritten.
    #[test]
    fn a_window_inside_a_padded_surface_skips_the_padding() {
        // 30 texels wide, a 32-texel (256-byte) pitch, 4 rows.
        let plan = raw_window_fill_plan(0x1000 + 10 * 8, 2 * 256 + 5 * 8, 0x1000, 30, 4, 32).unwrap().unwrap();
        assert_eq!(
            plan,
            vec![copy(10, 0, 20, 1, 0, None), copy(0, 1, 30, 1, 22 * 8, None), copy(0, 2, 15, 1, 22 * 8 + 256, None)]
        );
    }

    /// A pitch that is not a legal `bytes_per_row` is copied one row at a time, and a window that
    /// starts before the surface is filled from its first texel at the right offset.
    #[test]
    fn an_unaligned_pitch_goes_row_by_row() {
        let plan = raw_window_fill_plan(0x0ff0, 16 + 3 * 80, 0x1000, 10, 3, 10).unwrap().unwrap();
        assert_eq!(plan, vec![copy(0, 0, 10, 1, 16, None), copy(0, 1, 10, 1, 96, None), copy(0, 2, 10, 1, 176, None)]);
    }

    /// No overlap is no plan; an overlap that splits a texel is refused by name.
    #[test]
    fn disjoint_windows_have_no_plan_and_split_texels_are_refused() {
        assert!(raw_window_fill_plan(0x0, 0x100, 0x1000, 8, 8, 8).is_none());
        assert!(raw_window_fill_plan(0x1000 + 8 * 8 * 8, 0x100, 0x1000, 8, 8, 8).is_none());
        assert!(raw_window_fill_plan(0x1004, 0x100, 0x1000, 8, 8, 8).unwrap().is_err());
        assert!(raw_window_fill_plan(0x0ffc, 0x100, 0x1000, 8, 8, 8).unwrap().is_err());
    }
}
