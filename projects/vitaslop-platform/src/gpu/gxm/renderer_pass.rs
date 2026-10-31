use super::*;

impl GxmRenderer {
    pub(super) fn encode_pass(
        &mut self,
        device: &wgpu::Device,
        queue: &wgpu::Queue,
        encoder: &mut wgpu::CommandEncoder,
        color_view: &wgpu::TextureView,
        depth_view: &wgpu::TextureView,
        // The FORMAT of `color_view`. Usually the renderer's `color_format`, but the sRGB
        // twin when this pass renders into a gamma-correct surface - and a pipeline is
        // bound to the format of the attachment it writes, so every pipeline this pass
        // uses has to be built for this format, not for the renderer's default.
        target_format: wgpu::TextureFormat,
        scene: &RenderScene,
        surf_w: u32,
        surf_h: u32,
        // >>> THE ATTACHMENT'S OWN EXTENT, IN TEXELS. NOT DERIVED FROM `surf_w/surf_h`.
        //
        // Every rectangle the guest states - the viewport, the region clip - is in TARGET
        // pixels, i.e. in `surf_w x surf_h`. The attachment they have to be expressed
        // against is this, and the two are NOT always the same shape:
        //
        //   * an offscreen or depth-only pass renders at the target's own size, so they
        //     are equal;
        //   * the SUPERSAMPLED display pass renders `ss_scale` times larger in each axis;
        //   * the DISPLAY pass renders into the caller's FRAMEBUFFER, which is the panel -
        //     and a title may declare a display buffer SMALLER than the panel and let the
        //     display controller stretch it ([[vitaslop-display-buffer-can-be-smaller-than-
        //     the-panel]]). The golf title's front end declares 640x368 against a 960x544
        //     canvas.
        //
        // This used to be an integer `attach_scale` and the attachment extent was computed
        // as `surf * attach_scale`, which is true for the first two cases and FALSE for the
        // third. The cost of that was not subtle and it was not a corner case: on the
        // browser every UI draw that carries a guest viewport - the whole front end, menus,
        // HUD, the text marquee - was rasterised into the top-left 640x368 of a 960x544
        // frame, at two thirds scale, while the 3D scene behind it (which sets no viewport,
        // and so fell through to wgpu's default of the whole attachment) filled the frame.
        // The guest's region clip was cornered the same way, which cut UI panels off along
        // a hard edge two thirds across. It was invisible on every desktop path because
        // there the attachment IS the target size, so `surf * 1` happened to be right.
        att_w: u32,
        att_h: u32,
        clear: Option<wgpu::Color>,
        // Samples per pixel of `color_view`/`depth_view`. A pipeline is bound to its
        // attachments' sample count exactly as it is bound to their format, so this
        // travels with `target_format` and belongs in the pipeline cache key.
        samples: u32,
        // Where a multisampled pass resolves to: the stored-size texture everything
        // downstream samples. `None` when `samples == 1` and there is nothing to resolve.
        resolve: Option<&wgpu::TextureView>,
        // The TEXTURE behind `color_view`, when the renderer owns one. Only needed to copy
        // the attachment into `GxmRenderer::dest_color` for a draw that reads the
        // destination colour; `None` (a view the host handed us, with no texture behind it
        // we can name) means such a draw cannot be served and is dropped with a report.
        color_texture: Option<&wgpu::Texture>,
    ) {
        // THIS PASS'S WALL TIME, accumulated onto the frame - see
        // `EncodePhases::pass_wall_ms`. It brackets the whole function, including the work
        // BEFORE `t_prepare` (the dest-blend scan over every draw, `decide_scene_negw`),
        // which is the part the three per-pass spans never covered.
        let t_pass_wall = Stopwatch::start();
        // >>> THE SCENE'S ON-CHIP PIXEL DATA, WHEN AN 8-BIT ATTACHMENT CANNOT HOLD IT.
        //
        // A program that blends for itself at HALF precision over an 8-bit surface reads back
        // the halves the previous draw left in the tile, not an 8-bit rounding of them - see
        // `GxmRenderer::tile_shadow`. Such a pass draws into a half-float shadow instead, and
        // the shadow is packed into the real attachment once, at the end. Everything below
        // sees the SHADOW as its attachment (`color_view`, `target_format`); the real one is
        // `real_view`/`real_format`, and `color_texture` stays the REAL texture, which is what
        // the self-window copies read (they stand for MEMORY, i.e. the packed bytes).
        let any_wide_blender = self.gxp.enabled
            && crate::gpu::tile_shadow_enabled()
            && scene.draws.iter().any(|d| d.gxp.as_ref().is_some_and(|g| fragment_blends_wide(&g.fprog)));
        let tile = tile_shadow_applies(
            target_format,
            samples,
            resolve.is_some(),
            color_texture.map(|t| (t.format(), t.sample_count(), t.usage(), t.width(), t.height())),
            (att_w, att_h),
            scene.target.is_some_and(|t| alpha_single_color_format(t.format)),
            any_wide_blender,
        )
        .then(|| self.tile_shadow(device, att_w, att_h));
        let (real_view, real_format) = (color_view, target_format);
        let (color_view, target_format) = match tile.as_ref() {
            Some((_, v)) => (v, FLOAT_TARGET_FORMAT),
            None => (color_view, target_format),
        };
        // What a destination copy reads: the attachment the draws are actually written into.
        let dest_source: Option<&wgpu::Texture> = tile.as_ref().map(|(t, _)| t).or(color_texture);
        if tile.is_some() {
            self.ensure_texel_blit(device, FLOAT_TARGET_FORMAT);
            self.ensure_texel_blit(device, real_format);
            // The tile starts as the surface holds it, exactly as a scene that does not
            // clear LOADS its tiles. A clearing pass clears the shadow instead (below).
            if clear.is_none() {
                self.texel_blit(device, encoder, real_view, color_view, FLOAT_TARGET_FORMAT);
            }
        }
        // 1. Walk the scene once: pack vertex/index/uniform bytes into per-frame
        //    arenas and ensure each draw's texture upload + bind group exist.
        // Before any draw is prepared: does THIS pass's projection put clip `w` negative in
        // front of the camera? It is one answer for the whole pass and every draw's
        // `@group(3)` block carries it, so it has to be settled first.
        // TIMED, and the interpretations COUNTED - see `EncodePhases::negw_ms`.
        let t_negw = Stopwatch::start();
        let clip_measures = self.gxp.decide_scene_negw(scene);
        let negw_ms = t_negw.ms();
        // >>> DOES ANY DRAW IN THIS PASS BLEND FOR ITSELF? Asked BEFORE anything is
        // >>> prepared, because the answer decides how the pass is CUT.
        //
        // A fragment program can read its own output registers, which the hardware seeds
        // with the framebuffer colour, and blend in ordinary ALU. WebGPU has no framebuffer
        // fetch, so the only faithful mechanism is to end the render pass at such a draw,
        // copy the attachment into a texture and bind it - and a render pass cannot be
        // split once it has begun. See `GxmRenderer::dest_color`.
        // Settled before the scan, not only before the first pipeline build: the scan asks
        // the same question the recompiler will, and it has to ask it with the same arm set.
        let _ = crate::gpu::gxp_dest_blend_lowering();
        // The scan runs in BOTH arms of `VITASLOP_GXP_DEST` and the knob is applied after
        // it, so the two arms do the same work per frame. Short-circuiting on the knob made
        // the arms differ in CPU cost, and this title's guest clock is derived from work -
        // so the A/B compared two different TIMELINES and reported a picture change on
        // titles the knob does not touch at all.
        // A RAW 64-bit pass is never dual-source (an integer attachment cannot blend - the
        // link turns it off), so there a would-be dual-source draw reads the destination.
        let raw_pass = target_format == RAW64_FORMAT;
        let any_dest_reader = self.gxp.enabled
            && scene.draws.iter().any(|d| {
                d.gxp
                    .as_ref()
                    .is_some_and(|g| fragment_reads_dest(&g.fprog) && (raw_pass || !draw_dual_source(g)))
            });
        let wants_dest = any_dest_reader && crate::gpu::gxp_dest_enabled();
        // A MULTISAMPLED attachment cannot be the source of a texture-to-texture copy, and
        // resolving mid-pass is not the same image. Refuse rather than blend against
        // something else: the draw is dropped by `prepare` and says so.
        let dest_ok = wants_dest && color_texture.is_some();
        if wants_dest && !dest_ok {
            report_dest_needs_single_sample(samples, color_texture.is_some());
        }
        if dest_ok {
            self.ensure_dest_color(device, att_w, att_h, target_format, samples);
        }
        // Cloned (a view is a cheap handle) so the prepare walk below can borrow the rest
        // of the renderer mutably.
        let dest_shape = (att_w, att_h, target_format, samples);
        let dest_entry = dest_ok.then(|| self.dest_colors.get(&dest_shape)).flatten();
        let dest_gen = dest_entry.map(|d| d.generation).unwrap_or(0);
        let dest_view = dest_entry.map(|d| d.view.clone());
        let dest_binding = dest_view.as_ref().map(|v| (v, dest_gen));
        let t_prepare = Stopwatch::start();
        let stride = self.uniform_stride as usize;
        // >>> ONE `Vec::new()` PER PASS, AND THAT IS DELIBERATE AGAIN. See
        // [`GxmRenderer::arenas`] for the pooling attempt that was REVERTED and why.
        // >>> POOLED, WITH THEIR CAPACITY BOUNDED - see `GxmRenderer::pass_arenas` and the
        // >>> note above it for the version of this that was reverted and why.
        //
        // Taken out rather than borrowed: the walk below borrows the rest of the renderer
        // mutably, which is the same reason `rtt_rendered` is taken. They go back at the
        // bottom of this function, shrunk if they grew past the retention bound.
        let [mut vdata, mut idata, mut udata, mut gvdata, mut gidata, mut gudata] =
            if arena_pool_enabled() {
                let mut a = std::mem::take(&mut self.pass_arenas);
                for v in a.iter_mut() {
                    v.clear();
                }
                a
            } else {
                [const { Vec::new() }; 6]
            };
        // A fresh arena: every memoised window offset named the previous pass's.
        self.gxp.mem_push_memo.clear();
        // What this pass's window fill will copy from - the same targets, pitches and
        // exclusion `fill_windows_from_raw_targets` walks - so a per-draw window block can
        // leave those bytes out of its copy and its upload.
        self.gxp.udata_holes.clear();
        self.gxp.raw_fill.clear();
        if crate::gpu::mem_window_skip_covered() {
            let own = scene.target.map(|t| t.data_addr);
            for &key @ (base, tw, th) in self.rtt_rendered_raw.keys() {
                if Some(base) != own && self.rtt_raw.contains_key(&key) {
                    let pitch = self.rtt_pitch.get(&key).map_or(tw, |&(p, _)| p);
                    self.gxp.raw_fill.push((base, tw, th, pitch));
                }
            }
        }
        // See `uniform_align`: asked of the device once, at construction, not per pass.
        let ubo_align = self.uniform_align;
        let mut items: Vec<Item> = Vec::with_capacity(scene.draws.len());
        self.tex_pass_keys.clear();
        // The live recompiler's per-draw resources + a submission-order plan interleaving
        // recompiled and fixed-function draws (so they share one depth-tested pass).
        let gxp_enabled = self.gxp.enabled;
        let gxp_only = self.gxp.only;
        // The attachment this pass writes, NOT the renderer's default - see the parameter.
        let color_format = target_format;
        // A single-channel ALPHA surface - see `alpha_single_color_format`.
        let alpha_single = scene.target.is_some_and(|t| alpha_single_color_format(t.format));
        // What `warm_pipelines` predicts the NEXT frame's passes into this target from.
        {
            let shapes = self.pass_shapes.entry(Self::pass_id(scene)).or_default();
            let shape = (color_format, samples, alpha_single);
            if !shapes.contains(&shape) {
                // Bounded: a target is encoded under one or two shapes in practice.
                if shapes.len() >= 4 {
                    shapes.remove(0);
                }
                shapes.push(shape);
            }
            if let Some(t) = scene.target {
                let by_fmt = self.pass_shapes_by_format.entry(t.format).or_default();
                if !by_fmt.contains(&shape) {
                    if by_fmt.len() >= 4 {
                        by_fmt.remove(0);
                    }
                    by_fmt.push(shape);
                }
            }
        }
        let mut gxp_prepared: Vec<GxpPrepared> = Vec::new();
        // The scene draw each entry of `gxp_prepared` came from, in the same positions - the
        // destination-copy coalescing reads that draw's vertices (`draw_pixel_bounds`).
        let mut gxp_draw_of: Vec<usize> = Vec::new();
        let mut order: Vec<Enc> = Vec::with_capacity(scene.draws.len());
        // The guest's SCISSOR for each entry of `order`, in the same positions. It rides
        // beside `order` rather than inside `GxpPrepared`/`Item` because it is state both
        // paths share and neither owns - the fixed-function path has no per-draw state
        // struct of its own that a recompiled draw also passes through.
        let mut clips: Vec<RegionClip> = Vec::with_capacity(scene.draws.len());
        // The guest's VIEWPORT for each entry of `order`, in the same positions and for
        // the same reason as `clips`: `(SceGxmViewportMode, [xOff,xScale,yOff,yScale,
        // zOff,zScale])`. Riding beside `order` is what lets the draw loop compute the
        // rect ONCE for both arms - the recompiled one used to read it out of
        // `GxpPrepared` while the fixed-function one passed `full` unconditionally, and
        // two arms deciding a shared piece of guest state separately is how the
        // fixed-function path came to ignore the viewport entirely.
        let mut vports: Vec<(u32, [f32; 6])> = Vec::with_capacity(scene.draws.len());        // Taken out for the walk so the render-target views can be read while the
        // texture caches next to them are written; restored below.
        let rendered = std::mem::take(&mut self.rtt_rendered);
        let rendered_raw = std::mem::take(&mut self.rtt_rendered_raw);
        let mut depth_rendered = std::mem::take(&mut self.rtt_depth_rendered);
        // Read out for the same reason: `prepare` borrows `self.gxp` mutably.
        let reads_snapshot = std::mem::take(&mut self.rtt_reads_snapshot);
        // What this pass may SAMPLE: the targets this frame has rendered, plus every
        // target still RESIDENT from an earlier frame.
        //
        // A render target is guest memory, and guest memory keeps what was last written
        // into it until something overwrites it. `rtt_rendered` is cleared every frame, so
        // without this a pass sampling a buffer drawn on an EARLIER frame falls through to
        // decoding the guest bytes behind the pointer - and the GPU, not the guest, wrote
        // those pixels, so they decode to black.
        //
        // MEASURED on a retail title: at the title-to-menu transition the guest stops
        // rendering its background into 0x89204aa0 and starts blurring it instead. The root
        // of that blur chain samples 0x89204aa0, which is resident from the previous frame
        // and absent from `rtt_rendered`, so it read black - and every pass downstream of
        // it inherited the black, taking 91% of the frame with it in one flip.
        //
        // This is deliberately SEPARATE from `rtt_rendered`, which still decides two other
        // things that must not change: which pass clears a target first (a resident target
        // is still cleared on this frame's first pass, or a post-process would compose onto
        // a stale image forever) and which reads need a snapshot.
        let current_target = scene.target.map(|t| t.data_addr);
        // The cube maps assembled from rendered faces, as plain views for the sampler path.
        // Built from `rtt_cubes`, which OUTLIVES the frame: the title re-renders its faces
        // periodically and samples the cube every frame, so a map rebuilt per frame from
        // this frame's renders would serve the refresh frames and fall back to stale guest
        // memory for the rest. See the field's doc comment.
        let rendered_cubes: HashMap<u32, wgpu::TextureView> =
            self.rtt_cubes.iter().map(|(a, c)| (*a, c.view.clone())).collect();
        // >>> THE ONE PLACE `(address, width, height)` IS FLATTENED BACK TO AN ADDRESS, AND
        // >>> IT IS FLATTENED BECAUSE THE SAMPLER PATH RESOLVES BY ADDRESS AND BY RANGE.
        //
        // `rtt` and `rtt_rendered` are keyed by the target's extent as well as its address
        // (see `rtt`: one colour address, two live render targets, every frame). A DRAW has
        // only the address its texture descriptor names - and that descriptor does not have
        // to agree with the target's extent even when it means that exact target:
        // `rendered_alias` exists because a title samples a 960x544 buffer through a
        // 1920x1088 descriptor 256 bytes in, and a lookup keyed on the extent would miss it
        // completely and read guest bytes the GPU never wrote. So the map handed to
        // `prepare` stays keyed by address, and the CHOICE of which target answers for an
        // address is made here, once, on the evidence the draws themselves supply.
        //
        // The ranking, strongest first:
        //   1. the extent some draw of THIS PASS declares at that address - the guest's own
        //      statement of which of the two it is sampling;
        //   2. rendered in THIS frame, ahead of resident from an earlier one - which is
        //      exactly the precedence the old `rendered.clone()` + `or_insert` had;
        //   3. the larger image, then the wider, so the answer cannot depend on hash order.
        //
        // Rule 1 can only ever fire where an address holds more than one target, which is
        // why `declared` below is not even collected otherwise: with one target per address
        // - every title measured before the baseball title - every candidate scores identically on
        // it and this produces precisely the map it always did.
        let ambiguous: HashSet<u32> = {
            // The DISTINCT keys of the two maps, unioned first. A rendered target is
            // also an `rtt` entry under the same key, so counting the two maps' keys
            // without the union would report every rendered address as ambiguous and
            // walk the draws below on every pass of every title.
            let mut keys: HashSet<(u32, u32, u32)> = self.rtt.keys().copied().collect();
            keys.extend(rendered.keys().copied());
            let mut seen: HashSet<u32> = HashSet::default();
            let mut twice: HashSet<u32> = HashSet::default();
            for (a, _, _) in keys {
                if !seen.insert(a) {
                    twice.insert(a);
                }
            }
            twice
        };
        // The extent each draw of this pass declares at an AMBIGUOUS address. Walked only
        // when there is an ambiguity to resolve: this is a per-draw, per-sampler-unit walk
        // (a title's pass carries ~500 draws and up to seven units each), and paying it on
        // every pass of every title to answer a question that has one answer is the cost
        // this file keeps finding in the code it replaces
        // [[vitaslop-a-diagnostic-can-bury-the-findings]].
        //
        // Two draws of one pass declaring DIFFERENT extents at one ambiguous address
        // leaves the LAST of them - in draw order, so the answer is deterministic even
        // though it is arbitrary. Nothing better is available: one map per pass can hold
        // one answer per address, and a draw-by-draw resolution would mean rebuilding
        // the sampler map per draw. No title measured declares two.
        let declared: HashMap<u32, (u32, u32)> = if ambiguous.is_empty() {
            HashMap::default()
        } else {
            scene
                .draws
                .iter()
                .flat_map(|d| {
                    d.texture
                        .iter()
                        .map(|t| (t.data_addr, (t.width, t.height)))
                        .chain(d.gxp.iter().flat_map(|g| {
                            g.textures
                                .iter()
                                .chain(g.vertex_textures.iter())
                                .map(|t| (t.tex.data_addr, (t.tex.width, t.tex.height)))
                        }))
                })
                .filter(|(a, _)| ambiguous.contains(a))
                .collect()
        };
        // `(declared extent, rendered this frame, area, width)` - see the ranking above.
        type ViewRank = (bool, bool, u64, u32);
        let rank = |addr: u32, w: u32, h: u32, this_frame: bool| -> ViewRank {
            (declared.get(&addr) == Some(&(w, h)), this_frame, w as u64 * h as u64, w)
        };
        // Each offer carries the target's LINEAR view too when the offered one is its sRGB
        // view, so a sampler whose texture sets no gamma mode can read the stored bytes as
        // they are - see `Bind::RenderedLinear`. And a LINEAR target's sRGB view the other way
        // round (the flag says which twin it is), for a texture that sets a gamma mode over
        // it - see `Bind::RenderedSrgb`.
        type Offered = (ViewRank, wgpu::TextureView, Option<(wgpu::TextureView, bool)>);
        let offer = |best: &mut HashMap<u32, Offered>,
                     addr: u32,
                     r: ViewRank,
                     view: &wgpu::TextureView,
                     twin: Option<(&wgpu::TextureView, bool)>| {
            match best.get(&addr) {
                Some((have, _, _)) if *have >= r => {}
                _ => {
                    best.insert(addr, (r, view.clone(), twin.map(|(v, s)| (v.clone(), s))));
                }
            }
        };
        fn twin_of(t: &RttSurface) -> Option<(&wgpu::TextureView, bool)> {
            match (t.gamma, t.color_view_srgb.as_ref()) {
                (true, Some(_)) => Some((&t.color_view, false)),
                (false, Some(s)) => Some((s, true)),
                _ => None,
            }
        }
        let mut best: HashMap<u32, Offered> = HashMap::default();
        for (key @ (addr, w, h), view) in rendered.iter() {
            // This frame's targets are offered through the view their pass rendered with,
            // which for a gamma-correct one is its sRGB view.
            let twin = self.rtt.get(key).and_then(twin_of);
            offer(&mut best, *addr, rank(*addr, *w, *h, true), view, twin);
        }
        // >>> THE EXTENTS THIS FRAME HAS WRITTEN AT EACH ADDRESS - rendered by an earlier pass,
        // >>> or being rendered by this one.
        //
        // A target held from an EARLIER frame at such an address with a DIFFERENT extent is
        // stale: the memory it stood for has since been rewritten as the other surface, so
        // its image is no longer what those bytes hold. MEASURED on a fighting title:
        // its bloom builds a mip chain inside ONE 512x128 surface and reads
        // level 0 back through a 256x128 texture at the surface's own address; the extent
        // match handed that draw a 256x128 target left over from the character-select
        // screen - a red-only image - and every bloom level after it, and the final frame,
        // carried a red band. Not offering it lets the draw read the guest's bytes.
        let mut live_extents: HashMap<u32, HashSet<(u32, u32)>> = HashMap::default();
        for (addr, w, h) in rendered.keys() {
            live_extents.entry(*addr).or_default().insert((*w, *h));
        }
        if let Some(t) = scene.target {
            live_extents.entry(t.data_addr).or_default().insert((t.width, t.height));
        }
        for ((addr, w, h), t) in self.rtt.iter() {
            if live_extents.get(addr).is_some_and(|e| !e.contains(&(*w, *h))) && crate::gpu::stale_extent_guard() {
                continue;
            }
            // >>> AN ADDRESS THE GUEST HAS REUSED IS NOT THIS TARGET ANY MORE.
            //
            // The guest frees a render target and allocates an ordinary texture over it.
            // While the entry lives, offering it here hands that draw the OLD TARGET's
            // pixels instead of the texture the guest put there - a surface wearing
            // something else's image. `rtt_alias_block` is by ADDRESS, so this blocks every
            // target held there, which is the conservative direction: the draw falls through
            // to the guest's own bytes, which is what the guest wrote.
            //
            // CONFIRMED on the device, not inferred: the report below fired for about twenty
            // distinct addresses in one session, and the disagreements are not subtle -
            // `192x192 target` against a `1024x1024` bound texture, `64x64` against
            // `192x192`. A title sampling its own target through an odd descriptor does not
            // look like that; a recycled allocation does.
            //
            // So the extent decides. On a mismatch the entry is not offered, the draw falls
            // through to the ordinary path and decodes the guest's texture, and the target
            // goes unstamped so the reclamation takes it within its TTL.
            // [[vitaslop-an-address-is-not-an-identity]]
            if self.rtt_alias_block.contains(addr) {
                continue;
            }
            // A display pass has written this address since this target was drawn - the
            // display image is the newer picture. See `rtt_superseded`.
            if self.rtt_superseded.contains(addr) {
                continue;
            }
            // NEVER the attachment this pass is writing: binding it as a sampler at the
            // same time is a use-after-alias the driver rejects. Within-frame feedback is
            // what `rtt_reads_snapshot` is for, and it only applies once a pass has
            // already rendered into the target - which puts it in `rendered` above.
            if Some(*addr) == current_target {
                continue;
            }
            // >>> THE SAME VIEW THE PASS THAT RENDERED IT WOULD HAND OUT, WHICH FOR A
            // GAMMA-CORRECT TARGET IS THE sRGB ONE.
            //
            // A gamma-correct surface holds sRGB-ENCODED bytes: the ROP encodes every store,
            // and `color_view_srgb` exists precisely so a sampler DECODES them on the way
            // back in ("what the hardware does at both ends", per its own doc). The
            // within-frame path above obeys that - `rtt_rendered` gets the sRGB view. This
            // path did not, and handed out the linear view instead.
            //
            // Reading encoded bytes as linear returns them too BRIGHT (0.5 stored as 0.73
            // reads back as 0.73). On its own that is a one-off error. In a FEEDBACK chain -
            // a target sampled and written back into itself every frame, which is what an
            // accumulation/blur buffer is - it compounds: 0.5 -> 0.73 -> 0.88 -> 0.95 -> 1.
            //
            // That is the "renders fine for one frame, then slowly skews itself to white"
            // the user reported for days, and the shape is exactly right: a restore
            // re-creates the targets, so the first frame after it has nothing accumulated
            // and is CORRECT, and every frame after re-applies the extra encode. It reaches
            // only the cross-frame path, which is why a single-frame desktop headless shot
            // could never show it.
            let view = match (t.gamma, t.color_view_srgb.as_ref()) {
                (true, Some(v)) => v,
                _ => &t.color_view,
            };
            offer(&mut best, *addr, rank(*addr, *w, *h, false), view, twin_of(t));
        }
        // The extent of the target each address was answered with above, read back out of its
        // rank - what the sub-rectangle block below compares a texture's own extent against
        // when the texture names a target's BASE address.
        let offered_extent: HashMap<u32, (u32, u32)> = best
            .iter()
            .map(|(a, ((_, _, area, w), _, _))| (*a, (*w, (*area / u64::from((*w).max(1))) as u32)))
            .collect();
        // The linear twins of the gamma-correct views offered - see `Bind::RenderedLinear`.
        let mut rendered_linear: HashMap<u32, wgpu::TextureView> =
            best.iter().filter_map(|(a, (_, _, l))| l.as_ref().filter(|l| !l.1).map(|l| (*a, l.0.clone()))).collect();
        // ...and the sRGB twins of the LINEAR ones - see `Bind::RenderedSrgb`.
        let mut rendered_srgb: HashMap<u32, wgpu::TextureView> =
            best.iter().filter_map(|(a, (_, _, l))| l.as_ref().filter(|l| l.1).map(|l| (*a, l.0.clone()))).collect();
        let mut sample_views: HashMap<u32, wgpu::TextureView> =
            best.into_iter().map(|(a, (_, v, _))| (a, v)).collect();
        // >>> AND A DISPLAY BUFFER THIS PROCESS HAS FINISHED, for a title that samples its
        // >>> own previous frame. `or_insert` and AFTER the loop above on purpose: this
        // >>> only ever FILLS A GAP. An address `rtt` already answers for keeps that
        // >>> answer, so no title's existing behaviour moves - the reads this reaches are
        // >>> exactly the ones that used to fall through to guest bytes the GPU never
        // >>> wrote. See `display_images`.
        // >>> AND THE SAME sRGB RULE ON THE WAY BACK IN. A gamma display image holds
        // ENCODED bytes now that the pass above writes them encoded, so a title sampling
        // its own previous frame has to DECODE them - which is what binding the sRGB view
        // does. This is the rtt path's rule (see `sample_views` above, where handing out
        // the linear view of a gamma target compounded a feedback chain to white); it has
        // to hold here for the same reason and in the same direction.
        for (addr, view, linear) in self.display_images.iter().map(|(a, d)| match (d.gamma, d.view_srgb.as_ref()) {
            (true, Some(v)) => (a, v, Some(&d.view)),
            _ => (a, &d.view, None),
        }) {
            if Some(*addr) == current_target {
                // The attachment itself never; its snapshot when this pass reads it - see
                // `snapshot_display`.
                if let Some((a, snap)) = self.display_reads_snapshot.as_ref()
                    && a == addr
                {
                    sample_views.insert(*addr, snap.clone());
                }
                continue;
            }
            if !sample_views.contains_key(addr) {
                sample_views.insert(*addr, view.clone());
                if let Some(l) = linear {
                    rendered_linear.insert(*addr, l.clone());
                } else if let Some(s) = self.display_images.get(addr).and_then(|d| d.view_srgb.as_ref()) {
                    rendered_srgb.insert(*addr, s.clone());
                }
            }
        }
        // ...and the F16F16F16F16 reading of every RAW 64-bit surface. Held from an EARLIER
        // frame it only fills a gap, the same way. See `convert_raw_to_float`.
        //
        // >>> BUT ONE DRAWN THIS FRAME IS THE NEWEST PICTURE AT ITS ADDRESS, and wins over a
        // >>> target or display image an earlier frame left there. MEASURED on a shooter: its
        // loading screen draws the backdrop into an ordinary 960x544 target at the address
        // its world later renders as a 64-bit F16F16F16F16 surface, and the composite that
        // samples that address kept getting the held loading-screen target - same extent, so
        // the stale-extent guard could not tell - for every frame of the mission: the HUD over
        // the loading backdrop, while a fresh renderer (a frame replay) showed the world. An
        // ordinary target THIS frame rendered at the address keeps its answer.
        for (&key @ (addr, _, _), raw) in self.rtt_raw.iter() {
            if Some(addr) == current_target {
                continue;
            }
            if rendered_raw.contains_key(&key) && !rendered.keys().any(|&(a, _, _)| a == addr) {
                sample_views.insert(addr, raw.float_view.clone());
                rendered_linear.remove(&addr);
                rendered_srgb.remove(&addr);
            } else {
                sample_views.entry(addr).or_insert_with(|| raw.float_view.clone());
            }
        }
        // >>> ONE 32-BIT WORD OF A RAW 64-BIT SURFACE, READ AS A DOUBLE-WIDTH TEXTURE.
        //
        // A 64-bit pixel is two words in memory, so a title can sample the SAME surface as a
        // 32-bit texture twice as wide, with the surface's own row pitch: at the base address
        // every even texel is a pixel's first word, at base+4 its second. MEASURED on an
        // action title: its deferred lighting reads its F32F32 G-buffer
        // (0x8ac5fec0, 720x408) as a 1440x408 U8U8U8U8 texture (albedo) and again at +4 as
        // S8S8S8S8 (normals). Neither matched the held raw surface, and the address's entry
        // in `sample_views` was LAST frame's final RGBA8 image, which the frame writes over the
        // G-buffer's memory later on - so the lighting lit the previous picture instead of the
        // G-buffer, fed back into itself, and the daylit ruins came out as a near-black scene
        // that barely moved. The words are copied out through a buffer - a byte
        // reinterpretation, which is what the memory is - into a texture of the sampled
        // format, and bound at the address the guest named. `VITASLOP_RAW_WORD_VIEW=0` is the
        // arm back.
        if crate::gpu::raw_word_view() {
            let wants: Vec<(u32, u32, u32, u32, u32)> = scene
                .draws
                .iter()
                .flat_map(|d| d.gxp.iter().flat_map(|g| g.textures.iter().chain(g.vertex_textures.iter())))
                .map(|t| (t.tex.data_addr, t.tex.width, t.tex.height, t.tex.row_bytes, t.tex.base_format))
                .collect();
            let mut done: HashSet<u32> = HashSet::default();
            for (addr, w, h, row_bytes, base_format) in wants {
                if addr == 0 || !done.insert(addr) {
                    continue;
                }
                let format = match base_format {
                    0x0c => wgpu::TextureFormat::Rgba8Unorm,
                    0x0d => wgpu::TextureFormat::Rgba8Snorm,
                    _ => continue,
                };
                // >>> THE ROW STRIDE IS THE SURFACE'S PITCH, NOT ITS WIDTH. MEASURED on one
                // title's gameplay frame: its 720x408 G-buffer has a 1024-pixel pitch, so the
                // lighting's 1440x408 U8U8U8U8 view of it declares 8192 bytes a row, and a
                // match on `720 * 8` refused every one of them - the ambient and light passes
                // then sampled the F16 reinterpretation of the words as normals and albedo,
                // and lit the whole G-buffer white. The pitch comes from the pass that
                // rendered the surface (`rtt_pitch`), the width when it gave none.
                //
                // >>> AND THE VIEW MAY BE A TOP-LEFT WINDOW OF THE SURFACE. Row `y` of the view is
                // row `y` of the surface at its own pitch whatever the view's extent, so a view
                // narrower or shorter than the surface reads exactly its top-left texels.
                // MEASURED on an action title that renders at a reduced resolution: its G-buffer
                // pass draws 704x576 into a 960x800 render target with a 704-pixel pitch, and the
                // lighting reads 1408x576 word views. Matched only at the full `2 x 960` by `800`,
                // every view was refused, the lighting sampled guest bytes the GPU never wrote,
                // and the whole world came out as red and magenta noise behind a correct HUD.
                let hit = rendered_raw.keys().copied().find(|&(base, rw, rh)| {
                    let pitch = self.rtt_pitch.get(&(base, rw, rh)).map_or(rw, |&(p, _)| p);
                    (addr == base || addr == base.wrapping_add(4))
                        && w % 2 == 0
                        && w <= rw * 2
                        && h <= rh
                        && row_bytes == pitch * 8
                        && Some(base) != current_target
                });
                let Some(key3) = hit else { continue };
                let Some(raw) = self.rtt_raw.get(&key3) else { continue };
                let (rw, rh) = (key3.1, key3.2);
                let off = u64::from(addr - key3.0);
                let bpr = (rw * 8).next_multiple_of(wgpu::COPY_BYTES_PER_ROW_ALIGNMENT);
                let vkey = (addr, w, h, format);
                if !self.word_views.contains_key(&vkey) {
                    // Bounded: a title has a handful of these; a runaway set is dropped whole.
                    if self.word_views.len() >= 16 {
                        self.word_views.clear();
                    }
                    let staging = device.create_buffer(&wgpu::BufferDescriptor {
                        label: Some("gxm-rtt-raw64-words"),
                        size: u64::from(bpr) * u64::from(rh) + 8,
                        usage: wgpu::BufferUsages::COPY_SRC | wgpu::BufferUsages::COPY_DST,
                        mapped_at_creation: false,
                    });
                    let tex = device.create_texture(&wgpu::TextureDescriptor {
                        label: Some("gxm-rtt-raw64-wordview"),
                        size: wgpu::Extent3d { width: w, height: h, depth_or_array_layers: 1 },
                        mip_level_count: 1,
                        sample_count: 1,
                        dimension: wgpu::TextureDimension::D2,
                        format,
                        usage: wgpu::TextureUsages::TEXTURE_BINDING | wgpu::TextureUsages::COPY_DST,
                        view_formats: &[],
                    });
                    let view = tex.create_view(&Default::default());
                    self.word_views.insert(vkey, (staging, tex, view));
                }
                let (staging, view_tex, view) = {
                    let (b, t, v) = &self.word_views[&vkey];
                    (b.clone(), t.clone(), v.clone())
                };
                encoder.copy_texture_to_buffer(
                    raw.color.as_image_copy(),
                    wgpu::TexelCopyBufferInfo {
                        buffer: &staging,
                        layout: wgpu::TexelCopyBufferLayout { offset: 0, bytes_per_row: Some(bpr), rows_per_image: Some(rh) },
                    },
                    wgpu::Extent3d { width: rw, height: rh, depth_or_array_layers: 1 },
                );
                // At +4 a row's LAST texel is the next row's first word, as the memory
                // runs; the padded staging rows hold that in the next row's first bytes,
                // so every row but the last is copied whole and then that one texel is
                // taken from the following row. The surface's final texel lies past its
                // end and stays zero.
                let (whole_w, fix_last) = if off == 0 { (w, false) } else { (w - 1, true) };
                encoder.copy_buffer_to_texture(
                    wgpu::TexelCopyBufferInfo {
                        buffer: &staging,
                        layout: wgpu::TexelCopyBufferLayout { offset: off, bytes_per_row: Some(bpr), rows_per_image: Some(rh) },
                    },
                    view_tex.as_image_copy(),
                    wgpu::Extent3d { width: whole_w, height: h, depth_or_array_layers: 1 },
                );
                if fix_last && h > 1 {
                    encoder.copy_buffer_to_texture(
                        wgpu::TexelCopyBufferInfo {
                            buffer: &staging,
                            layout: wgpu::TexelCopyBufferLayout {
                                offset: u64::from(bpr),
                                bytes_per_row: Some(bpr),
                                rows_per_image: Some(rh - 1),
                            },
                        },
                        wgpu::TexelCopyTextureInfo {
                            texture: &view_tex,
                            mip_level: 0,
                            origin: wgpu::Origin3d { x: w - 1, y: 0, z: 0 },
                            aspect: wgpu::TextureAspect::All,
                        },
                        wgpu::Extent3d { width: 1, height: h - 1, depth_or_array_layers: 1 },
                    );
                }
                sample_views.insert(addr, view);
                // A word view is no surface's colour, so neither encoding's twin stands for it.
                rendered_linear.remove(&addr);
                rendered_srgb.remove(&addr);
                crate::gpu::report_raw_word_view(addr, key3.0, rw, rh, base_format);
            }
        }
        // >>> A TEXTURE THAT NAMES A SUB-RECTANGLE OF A TARGET THIS FRAME RENDERED.
        //
        // Its address lies INSIDE the target and its row stride IS the target's pitch, so on
        // the hardware it reads exactly the texels at `(offset / 4) % pitch`,
        // `(offset / 4) / pitch` onward - a window onto the surface, clamped at its own
        // edges. MEASURED on a fighting title: its bloom builds a mip chain
        // inside ONE 512x128 surface and its final composite samples level 2 as a 128x64
        // texture 1536 bytes in. The range alias (`rendered_alias`) bound the WHOLE surface
        // at its origin, so the composite stretched all three levels across the screen and
        // the level-1 region painted a red band a quarter of the frame wide. The window is
        // copied into its own texture before the pass and bound at the address the guest
        // named; a texture with ANOTHER pitch is a reinterpretation of the memory and keeps
        // the range alias. `VITASLOP_RTT_SUBRECT=0` is the arm back.
        //
        // >>> AND A WINDOW AT THE TARGET'S OWN ADDRESS, AND ONE OF A TARGET HELD FROM AN
        // >>> EARLIER FRAME. The memory rule is the same for both: the texture reads its
        // `width x height` texels from where it starts, at the pitch it declares. A texture at
        // the BASE address narrower or shorter than the surface is its top-left window, not a
        // scaled copy of the whole; and a surface an earlier frame rendered holds its texels
        // until something overwrites them. MEASURED on an action title: its bloom packs every
        // level into one 768x384 surface at the END of the frame, and the NEXT frame's sky
        // pass samples the top-left 256x128 of it (a texture at the surface's own address and
        // pitch). Handed the whole held surface, that pass painted every level - downscaled
        // copies of the frame - over the sky, and the bloom shrank them into the next frame's
        // atlas again: a checkerboard of ever smaller copies of the scene below the island.
        // A held surface is offered on the same terms the cross-frame path above offers one.
        // `VITASLOP_RTT_SUBRECT_HELD=0` is the arm back to interior windows of this frame's
        // targets alone.
        if crate::gpu::subrect_sampling() {
            let wide = crate::gpu::subrect_held_sampling();
            // A held target is a candidate only where the cross-frame path above would offer
            // it: not stale by extent, not an address reused or superseded, not this pass's
            // attachment (the hit test below refuses that for every candidate).
            let held_ok = |&(a, tw, th): &(u32, u32, u32)| {
                wide && !rendered.contains_key(&(a, tw, th))
                    && !(crate::gpu::stale_extent_guard() && live_extents.get(&a).is_some_and(|e| !e.contains(&(tw, th))))
                    && !self.rtt_alias_block.contains(&a)
                    && !self.rtt_superseded.contains(&a)
            };
            let wants: Vec<(u32, u32, u32, u32)> = scene
                .draws
                .iter()
                .flat_map(|d| d.gxp.iter().flat_map(|g| g.textures.iter().chain(g.vertex_textures.iter())))
                .map(|t| (t.tex.data_addr, t.tex.width, t.tex.height, t.tex.row_bytes))
                .filter(|&(a, w, h, rb)| {
                    a != 0
                        && rb != 0
                        && (!sample_views.contains_key(&a)
                            || (wide && offered_extent.get(&a).is_some_and(|&e| e != (w, h))))
                })
                .collect();
            let mut windowed: HashSet<u32> = HashSet::default();
            for (addr, w, h, row_bytes) in wants {
                if windowed.contains(&addr) {
                    continue;
                }
                // At the base address the window replaces the whole-surface answer the
                // cross-frame path gave, so it applies only while that answer stands: the
                // surface it was offered as, with no raw 64-bit surface claiming the address.
                let at_base_ok = |base: u32, tw: u32, th: u32| {
                    wide && offered_extent.get(&base) == Some(&(tw, th))
                        && (w, h) != (tw, th)
                        && !self.rtt_raw.keys().any(|k| k.0 == base)
                };
                if sample_views.contains_key(&addr) && !rendered.keys().chain(self.rtt.keys()).any(|&(b, tw, th)| b == addr && at_base_ok(b, tw, th)) {
                    continue;
                }
                // The target this frame rendered that holds `addr`, with the same pitch - or,
                // failing that, one held from an earlier frame.
                let candidates = rendered.keys().copied().chain(self.rtt.keys().copied().filter(held_ok));
                let hit = candidates.into_iter().find_map(|(base, tw, th)| {
                    let (pitch, bpp) = *self.rtt_pitch.get(&(base, tw, th))?;
                    if addr < base || row_bytes != pitch * bpp || Some(base) == current_target {
                        return None;
                    }
                    if addr == base && !at_base_ok(base, tw, th) {
                        return None;
                    }
                    // An INTERIOR window of a target held from an earlier frame waits on
                    // `subrect_held_interior` - see there.
                    if addr != base && !rendered.contains_key(&(base, tw, th)) && !crate::gpu::subrect_held_interior() {
                        return None;
                    }
                    let off = addr - base;
                    if off % bpp != 0 || off >= pitch * bpp * th {
                        return None;
                    }
                    let (x0, y0) = ((off / bpp) % pitch, (off / bpp) / pitch);
                    (x0 + w <= tw && y0 + h <= th).then_some(((base, tw, th), x0, y0))
                });
                let Some((key3, x0, y0)) = hit else {
                    // Say why an address INSIDE a rendered target was not given a window -
                    // the pitch disagreeing is a reinterpretation, anything else is a gap.
                    if let Some((base, tw, th)) =
                        rendered.keys().copied().find(|&(b, _, _)| addr > b && addr - b < 0x0100_0000)
                    {
                        crate::gpu::report_subrect_refused(
                            addr,
                            base,
                            format!(
                                "texture {w}x{h} row_bytes {row_bytes}; target {tw}x{th} pitch {:?}px; current target {:?}",
                                self.rtt_pitch.get(&(base, tw, th)),
                                current_target
                            ),
                        );
                    }
                    continue;
                };
                let Some(src) = self.rtt.get(&key3) else {
                    crate::gpu::report_subrect_refused(addr, key3.0, "the target holds no texture".into());
                    continue;
                };
                // The window is located in GUEST texels (the pitch test above already pinned
                // the surface at 32 bits a texel) and copied texel for texel into a texture of
                // the target's own HOST format, so the host format only has to be copyable.
                // It need not be 4 bytes: a 32-bit guest surface held as RGBA16F for its
                // range was refused here, fell back to the range alias, and the composite
                // stretched the WHOLE bloom surface - level 0 included - over the left half
                // of the screen (MEASURED on a beach title: a ghosted copy of the scene and a
                // seam at the centre of every frame).
                // A GAMMA target holds sRGB-ENCODED bytes, and every other path that hands one
                // out binds its sRGB view so the read decodes them (see `sample_views` above);
                // its window is the same bytes and takes the same view. MEASURED on a shooter's
                // front end: its whole menu is drawn into a gamma 960x544 surface and the
                // display pass samples the centred 800x480 window of it - refused, the screen
                // stayed black under a menu that was fully rendered.
                let srgb = src.color.format().add_srgb_suffix();
                if (src.gamma && srgb == src.color.format())
                    || src.color.format().block_copy_size(None).is_none()
                    || src.color.sample_count() != 1
                {
                    crate::gpu::report_subrect_refused(
                        addr,
                        key3.0,
                        format!(
                            "target gamma {} host format {:?} samples {}",
                            src.gamma,
                            src.color.format(),
                            src.color.sample_count()
                        ),
                    );
                    continue;
                }
                // The target texture may be rasterised larger than the guest extent.
                let (sx, sy) = (src.color.width() / key3.1.max(1), src.color.height() / key3.2.max(1));
                if sx == 0 || sy == 0 || src.color.width() != key3.1 * sx || src.color.height() != key3.2 * sy {
                    crate::gpu::report_subrect_refused(
                        addr,
                        key3.0,
                        format!(
                            "target texture {}x{} is not a whole multiple of its guest {}x{}",
                            src.color.width(),
                            src.color.height(),
                            key3.1,
                            key3.2
                        ),
                    );
                    continue;
                }
                let size = wgpu::Extent3d { width: w * sx, height: h * sy, depth_or_array_layers: 1 };
                let reused = self
                    .subrect_free
                    .get_mut(&(size.width, size.height, src.color.format()))
                    .and_then(Vec::pop);
                let sub = reused.unwrap_or_else(|| {
                    device.create_texture(&wgpu::TextureDescriptor {
                        label: Some("gxm-rtt-subrect"),
                        size,
                        mip_level_count: 1,
                        sample_count: 1,
                        dimension: wgpu::TextureDimension::D2,
                        format: src.color.format(),
                        usage: wgpu::TextureUsages::TEXTURE_BINDING | wgpu::TextureUsages::COPY_DST,
                        // Every window can take the sRGB view, so one pooled texture of a
                        // format serves gamma and linear targets alike.
                        view_formats: if srgb == src.color.format() { &[] } else { std::slice::from_ref(&srgb) },
                    })
                });
                encoder.copy_texture_to_texture(
                    wgpu::TexelCopyTextureInfo {
                        texture: &src.color,
                        mip_level: 0,
                        origin: wgpu::Origin3d { x: x0 * sx, y: y0 * sy, z: 0 },
                        aspect: wgpu::TextureAspect::All,
                    },
                    wgpu::TexelCopyTextureInfo {
                        texture: &sub,
                        mip_level: 0,
                        origin: wgpu::Origin3d::ZERO,
                        aspect: wgpu::TextureAspect::All,
                    },
                    size,
                );
                let view = if src.gamma {
                    sub.create_view(&wgpu::TextureViewDescriptor { format: Some(srgb), ..Default::default() })
                } else {
                    sub.create_view(&Default::default())
                };
                sample_views.insert(addr, view);
                // The window's twin is its OWN view in the other encoding - never the whole
                // surface's, which a base-address window would otherwise leave behind: the
                // linear view of a gamma window for a texture that sets no gamma mode, the
                // sRGB view of a linear window for one that does (see `Bind::RenderedSrgb`).
                rendered_linear.remove(&addr);
                rendered_srgb.remove(&addr);
                if src.gamma {
                    rendered_linear.insert(addr, sub.create_view(&Default::default()));
                } else if srgb != src.color.format() {
                    rendered_srgb.insert(addr, sub.create_view(&wgpu::TextureViewDescriptor { format: Some(srgb), ..Default::default() }));
                }
                windowed.insert(addr);
                self.subrect_keep.push(sub);
                crate::gpu::report_subrect(addr, key3.0, x0, y0, w, h);
            }
        }
        // >>> A WINDOW OF THE TARGET THIS PASS IS WRITING, READ BY A LATER DRAW OF THE PASS.
        //
        // A title can build a whole mip chain inside ONE scene: draw 0 shrinks the world into
        // the left of a 512x128 surface, draw 1 samples that region (a texture at the
        // surface's own address and pitch) into the next, draw 2 samples THAT one. MEASURED
        // on a beach title's bloom. Nothing above answers those addresses - the attachment
        // can never be bound while it is written - so they decoded guest bytes the GPU never
        // wrote, and every bloom level after the first was black.
        //
        // Each such draw sees what the draws before it rendered: the pass is cut in front of
        // it (exactly as for a destination reader) and the window is copied out of the live
        // attachment into its own texture, bound at the address the guest named. Only an
        // address nothing else answers for, only a single-sampled attachment the renderer
        // owns. `VITASLOP_RTT_SELF_WINDOW=0` is the arm back.
        let mut self_windows: HashMap<usize, Vec<SelfCopy>> = HashMap::default();
        // The windows read under their reader's own pixels, as the attachment held them before the
        // pass's first draw: `(x, y, width, height)` in attachment texels -> the snapshot, copied
        // before segment 0. See `SelfCopy::from`.
        let mut scene_start: HashMap<(u32, u32, u32, u32), wgpu::Texture> = HashMap::default();
        // Not on a RAW 64-bit pass: its attachment is handed in only for the destination
        // copy, and a window of raw words would be bound where a converted colour is read.
        if crate::gpu::self_window_sampling()
            && samples == 1
            && color_format != RAW64_FORMAT
            && let (Some(t), Some(att)) = (scene.target, color_texture)
            && t.stride_px != 0
            && att.format().block_copy_size(None).is_some()
            && (!t.gamma || att.format().add_srgb_suffix() != att.format())
            && let Some(bpp) = crate::gpu::gxm_color_bytes_per_pixel(t.format)
        {
            let pitch = t.stride_px;
            // A GAMMA attachment holds encoded bytes; its windows are read through the sRGB
            // view, as the sub-rectangle windows of an earlier gamma pass are.
            let self_window_srgb = [att.format().add_srgb_suffix()];
            let (sx, sy) = (att.width() / t.width.max(1), att.height() / t.height.max(1));
            // One texture per window address per pass, re-filled before every draw reading it.
            let mut windows: HashMap<u32, (wgpu::Texture, u32, u32)> = HashMap::default();
            // The rectangles the pass's earlier draws can have written, in target pixels: a
            // draw cannot write outside its viewport. See the override below.
            let mut drawn: Vec<(f32, f32, f32, f32)> = Vec::new();
            let full = (0.0, 0.0, t.width as f32, t.height as f32);
            for (di, d) in scene.draws.iter().enumerate() {
                let Some(g) = &d.gxp else { continue };
                // >>> ONE ADDRESS, TWO STORES: THE NEWEST WINS. An address another target also
                // answers for (a title can alias a shadow atlas over rows of its blur surface)
                // is still THIS pass's window once an earlier draw of the pass may have
                // written it - that write is newer than any earlier pass's. MEASURED on a
                // platformer's depth-of-field chain: its blur passes sampled the atlas
                // rendered at the same address one scene before, and the frame went dark.
                let written_here = |x0: u32, y0: u32, w: u32, h: u32| {
                    let (a0, b0, a1, b1) = (x0 as f32, y0 as f32, (x0 + w) as f32, (y0 + h) as f32);
                    drawn.iter().any(|&(x0, y0, x1, y1)| a0 < x1 && x0 < a1 && b0 < y1 && y0 < b1)
                };
                let v = g.viewport;
                let vp = if g.viewport_enable == crate::gpu::SCE_GXM_VIEWPORT_DISABLED || v.iter().all(|c| *c == 0.0) {
                    full
                } else {
                    (v[0] - v[1].abs(), v[2] - v[3].abs(), v[0] + v[1].abs(), v[2] + v[3].abs())
                };
                let own = own_pixels(vp, d.region_clip, t.width, t.height);
                let mut copies = Vec::new();
                for tx in g.textures.iter().chain(g.vertex_textures.iter()).map(|tx| &tx.tex) {
                    let (addr, w, h) = (tx.data_addr, tx.width, tx.height);
                    if sx == 0 || sy == 0 || addr < t.data_addr || tx.row_bytes != pitch * bpp {
                        continue;
                    }
                    let off = addr - t.data_addr;
                    let (x0, y0) = ((off / bpp) % pitch, (off / bpp) / pitch);
                    if off % bpp != 0 || x0 + w > t.width || y0 + h > t.height {
                        continue;
                    }
                    // >>> A WINDOW UNDER THE DRAW'S OWN PIXELS READS THE SURFACE AS THE SCENE FOUND IT.
                    //
                    // The GPU is tile-based: a tile's colour lives on chip while every draw of
                    // the scene runs over it, and reaches memory only when the tile is stored.
                    // A texel the reading draw is itself covering is therefore still in a tile
                    // that has not been stored, and the sample reads the memory from before
                    // the scene - never what an earlier draw of the scene put there. MEASURED
                    // on a shooter: its reflection pass clears its whole target white (draw 0),
                    // then draws 1-2 blend the frame into a history strip of the same target by
                    // reading that strip under their own pixels. Handed the live attachment
                    // they read draw 0's white, the history never survived a frame, and the
                    // reflections it feeds went white wherever they were near. Such a window
                    // is filled from a snapshot of the attachment taken before the pass's
                    // first draw. A window AWAY from the draw (a mip level built by an earlier
                    // draw and read by a later one) keeps the rule above.
                    let under_own = window_under(own, x0, y0, w, h);
                    let origin = wgpu::Origin3d { x: x0 * sx, y: y0 * sy, z: 0 };
                    let size = wgpu::Extent3d { width: w * sx, height: h * sy, depth_or_array_layers: 1 };
                    let from = if under_own && crate::gpu::self_window_scene_start() {
                        // Its own textures, never the window pool's: a snapshot is a copy source,
                        // which no pooled window is.
                        // One per window of the pass, so two windows of a size are two textures.
                        let slot = scene_start.keys().filter(|k| (k.2, k.3) == (size.width, size.height)).count() as u32;
                        let snap = scene_start.entry((origin.x, origin.y, size.width, size.height)).or_insert_with(|| {
                            let key = (size.width, size.height, att.format(), slot);
                            self.self_window_start
                                .entry(key)
                                .or_insert_with(|| {
                                    device.create_texture(&wgpu::TextureDescriptor {
                                        label: Some("gxm-rtt-self-window-start"),
                                        size,
                                        mip_level_count: 1,
                                        sample_count: 1,
                                        dimension: wgpu::TextureDimension::D2,
                                        format: att.format(),
                                        usage: wgpu::TextureUsages::COPY_SRC | wgpu::TextureUsages::COPY_DST,
                                        view_formats: &[],
                                    })
                                })
                                .clone()
                        });
                        Some(snap.clone())
                    } else {
                        None
                    };
                    let tex = match windows.get(&addr) {
                        Some((tex, ww, wh)) if (*ww, *wh) == (w, h) => tex.clone(),
                        Some(_) => continue,
                        // Under the draw the window is always this pass's: the bindings are the
                        // pass's, so a later draw reading the same address away from itself
                        // shares the texture, and this draw must have it filled.
                        None if sample_views.contains_key(&addr) && from.is_none() && !written_here(x0, y0, w, h) => continue,
                        None => {
                            let size = wgpu::Extent3d { width: w * sx, height: h * sy, depth_or_array_layers: 1 };
                            let tex = self
                                .subrect_free
                                .get_mut(&(size.width, size.height, att.format()))
                                .and_then(Vec::pop)
                                .unwrap_or_else(|| {
                                    device.create_texture(&wgpu::TextureDescriptor {
                                        label: Some("gxm-rtt-self-window"),
                                        size,
                                        mip_level_count: 1,
                                        sample_count: 1,
                                        dimension: wgpu::TextureDimension::D2,
                                        format: att.format(),
                                        usage: wgpu::TextureUsages::TEXTURE_BINDING | wgpu::TextureUsages::COPY_DST,
                                        // The pool is shared with the sub-rectangle windows,
                                        // which may be handed out through their sRGB view.
                                        view_formats: if att.format().add_srgb_suffix() == att.format() {
                                            &[]
                                        } else {
                                            &self_window_srgb
                                        },
                                    })
                                });
                            let view = if t.gamma {
                                tex.create_view(&wgpu::TextureViewDescriptor {
                                    format: Some(self_window_srgb[0]),
                                    ..Default::default()
                                })
                            } else {
                                tex.create_view(&Default::default())
                            };
                            sample_views.insert(addr, view);
                            // Its twin in the other encoding, as for a sub-rectangle window.
                            rendered_linear.remove(&addr);
                            rendered_srgb.remove(&addr);
                            if t.gamma {
                                rendered_linear.insert(addr, tex.create_view(&Default::default()));
                            } else if att.format().add_srgb_suffix() != att.format() {
                                rendered_srgb.insert(
                                    addr,
                                    tex.create_view(&wgpu::TextureViewDescriptor {
                                        format: Some(self_window_srgb[0]),
                                        ..Default::default()
                                    }),
                                );
                            }
                            self.subrect_keep.push(tex.clone());
                            crate::gpu::report_self_window(addr, t.data_addr, x0, y0, w, h);
                            windows.insert(addr, (tex.clone(), w, h));
                            tex
                        }
                    };
                    copies.push(SelfCopy { dst: tex, origin, size, from });
                }
                if !copies.is_empty() {
                    self_windows.insert(di, copies);
                }
                drawn.push(vp);
            }
        }
        // Order index (the draw's place in `order`) -> its window copies. Filled as the
        // draws are prepared; the pass is cut at every one of them.
        let mut self_cuts: HashMap<usize, Vec<SelfCopy>> = HashMap::default();
        // >>> AND THE SAME FOR DEPTH, WHICH IS WHAT A SHADOW MAP IS.
        //
        // `rtt_depth_rendered` is cleared every frame, exactly as `rtt_rendered` is, so a
        // pass sampling a depth buffer some EARLIER frame filled found nothing and fell
        // through to decoding the guest bytes behind the pointer - and those bytes were
        // written by the GPU, so they decode to nonsense. The colour side of this was
        // fixed above; the depth side was not, and a shadow map is the case that makes it
        // matter, because a title is free to redraw its shadow map less often than it
        // draws the world.
        //
        // MEASURED on the golf title, in the course: its frame is
        // `[NO-COLOUR:81, <world>:132, <offscreen>:1, <hud>:116]` - the colour-less first
        // pass renders the 1536x1536 shadow map into depth `0x89366530`, which every
        // terrain and character draw then samples. During a SWING the guest submits a
        // single scene and no shadow pass at all, so on those frames the shadow term came
        // from stale bytes and the whole course rendered white; a `VITASLOP_CHAIN_DRAWS`
        // trace shows `0x89366530(1536x1536)` with no `*` and no `~` against it.
        //
        // Rebuilt from `self.rtt` rather than by keeping last frame's map, for the reason
        // the colour path gives: a target that has been recreated since must hand out the
        // LIVE view, not the one that named the texture it replaced. The converted
        // `gxm_depth` texture is a separate resource from the depth ATTACHMENT, so binding
        // it is not an alias of anything this pass is writing.
        //
        // Keyed THROUGH `rtt_depth_addrs`, which remembers which target holds a given
        // depth surface. Iterating `self.rtt` and using its own key instead offers every
        // target's depth under that target's COLOUR address, and because the depth path is
        // consulted before the colour one, the next pass that samples that colour target
        // reads a distance instead of the image - see `rtt_depth_addrs`.
        for (depth_addr, target) in self.rtt_depth_addrs.iter() {
            if let Some(v) = self.rtt.get(target).and_then(|t| t.gxm_depth.as_ref()) {
                depth_rendered.entry(*depth_addr).or_insert_with(|| v.view.clone());
            }
        }
        // A depth-only pass is exactly the case where the two addresses coincide, which is
        // how the sampler path tells "correctly in both maps" from "wrongly in both".
        // Compared against the ADDRESS half of the value: `rtt_depth_addrs` maps a depth
        // address onto the full `(address, width, height)` key of the target that holds it,
        // and a depth-ONLY pass is the case where that address IS the depth address - see
        // `encode_depth_only_pass`.
        let depth_only: HashSet<u32> =
            self.rtt_depth_addrs.iter().filter(|(d, t)| **d == t.0).map(|(d, _)| *d).collect();
        self.sampled_addrs.clear();
        // The previous-draw sampler answer belongs to the pass that produced it: the maps a
        // unit resolves through are rebuilt HERE and nowhere else, so this is the one place
        // the fingerprint could go stale. See `GxpLive::last_sampler_bg`.
        self.gxp.last_sampler_bg = None;
        self.gxp.sampler_pre.clear();
        // `VITASLOP_CHAIN_DRAWS=1`: describe every draw in this pass that samples a
        // target the frame rendered. A composite that shows none of the world has
        // either no such draw, or one whose blend/space/geometry throws it away, and
        // the finished (black) frame cannot tell those apart.
        static CHAIN_DRAWS: std::sync::OnceLock<Option<String>> = std::sync::OnceLock::new();
        let chain_draws = CHAIN_DRAWS.get_or_init(|| crate::knobs::var("VITASLOP_CHAIN_DRAWS").ok());
        // `[all:]<from>-<to>` limits it to those GUEST frames (see [`set_guest_frame`]) - a
        // defect that shows for a few seconds of a long run (a football title's replays)
        // cannot be traced every frame, least of all through a browser console.
        let (chain_spec, chain_window) = match chain_draws.as_deref() {
            Some(s) => {
                let (head, win) = match s.rsplit_once(':') {
                    Some((h, w)) => (Some(h), Some(w)),
                    None if s.contains('-') => (None, Some(s)),
                    None => (Some(s), None),
                };
                let window = win.and_then(|w| {
                    let (a, b) = w.split_once('-')?;
                    Some((a.trim().parse::<u64>().ok()?, b.trim().parse::<u64>().ok()?))
                });
                (head, window)
            }
            None => (None, None),
        };
        let in_window = chain_window.is_none_or(|(a, b)| (a..=b).contains(&crate::gpu::guest_frame()));
        let trace_draws = chain_draws.is_some() && in_window;
        // `=all` describes EVERY draw of the pass, not only the ones sampling a
        // rendered target - what is needed when the question is "which draw was
        // supposed to put the world on screen" rather than "did this one bind right".
        let trace_all = chain_spec == Some("all");
        // `key=<hex>[+<hex>..]` traces EVERY draw of those pairs and nothing else: a
        // whole-pass trace of a long window was 222k lines and took the browser rig down
        // before the replay it was waiting for arrived, when the question was one pair's
        // samples across that window.
        let trace_keys: Option<Vec<u64>> = chain_spec.and_then(|s| s.strip_prefix("key=")).map(|ks| {
            ks.split('+').filter_map(|k| u64::from_str_radix(k.trim().trim_start_matches("0x"), 16).ok()).collect()
        });
        // Fallback draws of THIS pass, by reason - see `fallback_reasons`.
        let mut fb_reasons: HashMap<String, usize> = HashMap::default();
        for (di, d) in scene.draws.iter().enumerate() {
            let keyed = trace_keys.as_ref().is_none_or(|ks| {
                d.gxp.as_ref().is_some_and(|g| ks.contains(&GxpLive::key(g)))
            });
            if trace_draws && keyed {
                let trace_all = trace_all || trace_keys.is_some();
                // The sampled texture's own dimensions matter as much as its address: a
                // title may bind a render target through a texture describing a
                // different extent, and the UVs the shader computes are for THAT extent.
                let hits: Vec<String> = d
                    .texture
                    .iter()
                    .map(|t| (t.data_addr, t.width, t.height))
                    .chain(d.gxp.iter().flat_map(|g| {
                        g.textures
                            .iter()
                            .chain(g.vertex_textures.iter())
                            .map(|t| (t.tex.data_addr, t.tex.width, t.tex.height))
                    }))
                    .filter(|(a, _, _)| trace_all || sample_views.contains_key(a))
                    // `*` this frame rendered it, `~` it is RESIDENT from an earlier frame,
                    // nothing at all means the sample decodes guest bytes. The difference
                    // between the first two used to be invisible and it was the whole of
                    // the black-background bug - the chain root read a target
                    // the frame had not redrawn, and an unmarked address looked identical
                    // to an ordinary texture.
                    .map(|(a, w, h)| {
                        // A SCAN BY ADDRESS: this is the per-draw trace, and a draw has
                        // only the address it named - the same reason `sample_views`
                        // above is flattened onto one.
                        let mark = if rendered.keys().any(|(ra, _, _)| *ra == a) {
                            "*"
                        } else if sample_views.contains_key(&a) {
                            "~"
                        } else {
                            ""
                        };
                        format!("{a:#x}({w}x{h}){mark}")
                    })
                    .collect();
                if !hits.is_empty() || trace_all {
                    report_knob!(
                        // "carries a payload" is NOT "is recompiled" - a payload that
                        // fails to link falls back, and labelling that `recompiled=true`
                        // is how a composite draw got read as working when it was not.
                        // The index count is the RECOMPILED one when there is a payload.
                        // The fixed-function count is zero for such a draw by design (the
                        // builder does not produce a representation the renderer will not
                        // use), and printing that made every draw of a recompiled pass
                        // read as empty geometry - a diagnostic saying exactly the wrong
                        // thing about the question it exists to answer.
                        // >>> THE DEPTH STATE IS ON THIS LINE BECAUSE A DRAW THAT SHOULD
                        // >>> NOT BE ON SCREEN IS AS OFTEN DEPTH-REJECTED AS BLENDED AWAY.
                        //
                        // "which draw put this on screen" and "which draw should have been
                        // thrown away" are the same question, and a frame in which one
                        // opaque quad covers everything cannot distinguish a wrong blend
                        // from a depth test the guest expected to fail. The guest's own
                        // func/write and the VIEWPORT's depth map (zScale/zOffset - what
                        // turns a guest z of 65535 into a clip-space one) are what decide
                        // it, and neither appears anywhere else a knob can reach.
                        // >>> WHICH TARGET THIS DRAW PAINTS INTO, on every line.
                        //
                        // The chain DUMP names its files by target address, and this
                        // trace named none, so correlating "this atlas came out magenta"
                        // with "this pair drew it" was not possible from the two together
                        // - and a title that bakes two hundred small targets from the same
                        // two pairs is exactly where that correlation is the question.
                        "chain draw #{di}: into {} samples {:?} key={:?} has_payload={} blend={:?} opaque={} space={:?} idx={} depth=(func {:#x}, write {}, zScale {:?}, zOffset {:?}, bias {:?}) blend_state={:?}",
                        scene
                            .target
                            .as_ref()
                            .map(|t| format!(
                                "{:#010x}({}x{},fmt {:#x})",
                                t.data_addr, t.width, t.height, t.format
                            ))
                            .unwrap_or_else(|| "<display>".into()),
                        hits,
                        d.gxp.as_ref().map(|g| format!("{:x}", GxpLive::key(g))),
                        d.gxp.is_some(),
                        d.gxp.as_ref().map(|g| g.blend),
                        d.opaque,
                        d.space,
                        d.gxp.as_ref().map(|g| g.index_count).unwrap_or(d.index_count),
                        d.gxp.as_ref().map(|g| g.depth_func).unwrap_or(0),
                        d.gxp.as_ref().map(|g| g.depth_write).unwrap_or(false),
                        d.gxp.as_ref().map(|g| g.viewport[5]),
                        d.gxp.as_ref().map(|g| g.viewport[4]),
                        d.gxp.as_ref().map(|g| g.depth_bias),
                        // The GUEST'S OWN blend equation, raw (`[mask, colour func, alpha
                        // func, colour src, colour dst, alpha src, alpha dst]`). `blend`
                        // above is the fixed-function heuristic and says nothing about it;
                        // a full-screen quad that should contribute nothing and a
                        // full-screen quad that paints white differ only here.
                        d.gxp.as_ref().map(|g| g.blend_state),
                    );
                    // A keyed trace follows one pair across a window of frames, and a pair
                    // that turns black between two of them changed in what it READ - so
                    // print what it read: the fragment uniforms in full (a pair's are a few
                    // dozen floats), the vertex ones by hash, every memory window by hash.
                    if let (Some(_), Some(g)) = (&trace_keys, &d.gxp) {
                        let floats = |b: &[u8]| -> Vec<f32> {
                            b.chunks_exact(4).map(|c| f32::from_le_bytes([c[0], c[1], c[2], c[3]])).collect()
                        };
                        let h = |b: &[u8]| {
                            b.iter().fold(0xcbf2_9ce4_8422_2325u64, |a, x| (a ^ *x as u64).wrapping_mul(0x100_0000_01b3))
                        };
                        report_knob!(
                            "chain key f{} draw #{di}: frag_sa {:?} vert_sa {}B {:016x} vwin {:?} fwin {:?}",
                            crate::gpu::guest_frame(),
                            floats(&g.frag_sa),
                            g.vert_sa.len(),
                            h(&g.vert_sa),
                            g.mem_windows.iter().map(|(a, b)| format!("{a:#x}+{}={:016x}", b.len(), h(b))).collect::<Vec<_>>(),
                            g.frag_mem_windows.iter().map(|(a, b)| format!("{a:#x}+{}={:016x}", b.len(), h(b))).collect::<Vec<_>>(),
                        );
                    }
                }
            }
            // >>> RENDER ONLY A RANGE OF THE PASS'S DRAWS (`VITASLOP_DRAW_RANGE=lo-hi`).
            //
            // `VITASLOP_CHAIN_LIMIT` bisects a frame by PASS, and a title whose whole frame
            // is one pass cannot be bisected by it at all. The question this answers is the
            // same one at draw granularity: a movie quad that contributes NO pixels is
            // either covered by a later draw, degenerate, or blended away, and the finished
            // frame cannot tell those apart - MEASURED on a retail golf title whose opening
            // renders the movie as draw 1 of 258 and shows none of it, byte-identical to a
            // run with the movie switched off entirely.
            //
            // A diagnostic, never a mode: it renders a frame the guest never asked for.
            if let Some((lo, hi)) = crate::gpu::draw_range()
                && (di < lo || di > hi) {
                    continue;
                }
            if let Some(t) = &d.texture {
                self.sampled_addrs.insert(t.data_addr);
            }
            if let Some(g) = &d.gxp {
                // Vertex samplers count too: `final-pass-sampled` is the list a reader
                // checks a missing target against, and a target read only by a vertex
                // program was invisible in it.
                self.sampled_addrs.extend(
                    g.textures.iter().chain(g.vertex_textures.iter()).map(|t| t.tex.data_addr),
                );
            }
            // Live GXP path: draw with the guest's real shaders when the pair links. On a
            // link/format failure fall through to the fixed-function packing below (unless
            // isolate mode, which renders only the recompiled draws).
            if gxp_enabled {
                if let Some(g) = &d.gxp {
                    self.rtt_hits += g
                        .textures
                        .iter()
                        .chain(g.vertex_textures.iter())
                        .filter(|t| sample_views.contains_key(&t.tex.data_addr))
                        .count();
                    if let Some(mut prep) =
                        {
                            // Lent for the prepare so a transcode can time its passes -
                            // see `lent_ts_pair`. Reclaimed before anything else here can
                            // reach for `self.ts`.
                            lend_ts(self.ts.take());
                            let r = self.gxp.prepare(device, queue, color_format, alpha_single, samples, g, [scene.depth_min, scene.depth_scale], (surf_w, surf_h), (att_w, att_h), &sample_views, &rendered_linear, &rendered_srgb, &depth_rendered, &depth_only, &rendered_cubes, &rendered_raw, &reads_snapshot, &mut gvdata, &mut gidata, &mut gudata, ubo_align, dest_binding);
                            self.ts = reclaim_ts();
                            r
                        }
                    {
                        if self.gxp.solid {
                            prep.blend = false; // REPLACE + depth-Always variant (see make)
                        }
                        if let Some(c) = self_windows.remove(&di) {
                            self_cuts.insert(order.len(), c);
                        }
                        order.push(Enc::Gxp(gxp_prepared.len()));
                        clips.push(d.region_clip);
                        vports.push((d.viewport_enable, d.viewport));
                        gxp_prepared.push(prep);
                        gxp_draw_of.push(di);
                        continue;
                    }
                    // Prepared failed: this draw is one of the pass's fallbacks. Tally it
                    // against its pair's reason so the summary can rank causes by draws.
                    *fb_reasons.entry(fallback_reason_of(self.gxp.pair_key(g))).or_insert(0) += 1;
                    // >>> AND UNLESS THE APPROXIMATION WAS ASKED FOR, THE DRAW IS DROPPED.
                    // The fixed-function path does not run the guest's shader, so drawing
                    // through it here would put a plausible wrong picture on screen with
                    // nothing to say so - see `report_fallback` for the three choices and
                    // why dropping is the honest one. The tally above is what the perf
                    // panel ranks by cause, so a dropped draw is counted and named.
                    if gxp_only || !allow_fixed_function() {
                        continue;
                    }
                } else if gxp_only {
                    continue;
                }
            }
            // A shader-only draw has no colour source the fixed-function packing can
            // honour, so reaching here means the recompiler could not draw it and the
            // alternative would be an opaque white rectangle. Skip it - but never
            // silently: this is a draw the guest asked for that the frame will not show.
            if d.shader_only {
                crate::gpu::report_shader_only_skip(d, gxp_enabled);
                continue;
            }
            if d.index_count == 0 || d.vertices.is_empty() {
                continue;
            }
            // A RAW 64-bit pass has no fixed-function pipelines (they are built for the
            // UNORM formats only) - a draw that fell to this path there is dropped, by name.
            if color_format == RAW64_FORMAT {
                report_raw64_fixed_dropped(scene.target.map_or(0, |t| t.data_addr));
                continue;
            }
            let bind = match &d.texture {
                None => BindKey::White,
                // A texture whose data pointer is a target THIS frame already rendered
                // is that render, not the guest bytes behind the pointer (which the GPU,
                // not the guest, was supposed to have written).
                Some(t) if sample_views.contains_key(&t.data_addr) => {
                    self.rtt_hits += 1;
                    let snapshot = self.rtt_reads_snapshot.contains(&t.data_addr);
                    let view = sample_views[&t.data_addr].clone();
                    self.ensure_rtt_bind(device, t.data_addr, t.filter_linear, snapshot, &view);
                    BindKey::Rtt(t.data_addr, t.filter_linear, snapshot)
                }
                Some(t) => {
                    self.ensure_texture(device, queue, t);
                    BindKey::Tex(t.key, t.filter_linear)
                }
            };
            let (mode, mvp) = match d.space {
                DrawSpace::Mvp(m) => (0u32, m),
                DrawSpace::Ndc => (1u32, [0f32; 16]),
                DrawSpace::Pixel => (2u32, [0f32; 16]),
            };
            // Uniform block (must match the WGSL `U` struct), then pad to the
            // dynamic-offset stride. A vec4 in WGSL is 16-byte aligned, so each material
            // vec3 is written as 4 floats (xyz + a pad lane).
            let uniform_offset = udata.len() as u32;
            for v in &mvp {
                udata.extend_from_slice(&v.to_le_bytes());
            }
            udata.extend_from_slice(&mode.to_le_bytes());
            udata.extend_from_slice(&(surf_w as f32).to_le_bytes());
            udata.extend_from_slice(&(surf_h as f32).to_le_bytes());
            udata.extend_from_slice(&(d.texture.is_some() as u32).to_le_bytes());
            udata.extend_from_slice(&d.exposure.to_le_bytes());
            udata.extend_from_slice(&scene.depth_min.to_le_bytes());
            udata.extend_from_slice(&scene.depth_scale.to_le_bytes());
            udata.extend_from_slice(&[0u8; 4]); // pad2
            let vec4 = |udata: &mut Vec<u8>, v: [f32; 3]| {
                for c in v {
                    udata.extend_from_slice(&c.to_le_bytes());
                }
                udata.extend_from_slice(&[0u8; 4]); // vec4 pad lane
            };
            vec4(&mut udata, d.material.tint);
            vec4(&mut udata, d.material.light_dir);
            vec4(&mut udata, d.material.light_col);
            vec4(&mut udata, d.material.ambient);
            udata.resize(uniform_offset as usize + stride, 0);

            let v_off = vdata.len() as u64;
            vdata.extend_from_slice(&d.vertices);
            let i_off = idata.len() as u64;
            idata.extend_from_slice(&d.indices);

            // A MASKED fixed-function draw gets its own pipeline variant, built here (the
            // encoder only reads the cache). The gxmconf stencil scene is exactly this: a
            // placeholder-shader app whose draws take this path.
            let stencil = (!crate::gpu::stencil_disabled()
                && !stencil_words_are_default(&d.stencil, d.two_sided))
                .then(|| self.ensure_fixed_stencil_pipeline(device, d.opaque, color_format, samples, &d.stencil, d.two_sided));
            items.push(Item {
                v_off,
                v_len: d.vertices.len() as u64,
                i_off,
                i_len: d.indices.len() as u64,
                index_count: d.index_count,
                uniform_offset,
                opaque: d.opaque,
                bind,
                stencil,
                stencil_ref: d.stencil[6] & 0xff,
            });
            order.push(Enc::Fixed(items.len() - 1));
            clips.push(d.region_clip);
            vports.push((d.viewport_enable, d.viewport));
        }
        self.rtt_rendered = rendered;
        self.rtt_rendered_raw = rendered_raw;
        self.rtt_depth_rendered = depth_rendered;
        self.rtt_reads_snapshot = reads_snapshot;
        if gxp_enabled {
            let with_payload = scene.draws.iter().filter(|d| d.gxp.is_some()).count();
            let summary = (scene.draws.len(), with_payload, gxp_prepared.len(), items.len());
            // FAILURE-ONLY. Every number agreeing with no fixed-function item is the HEALTHY
            // case, and printing it buried the disagreements this line exists to catch under
            // 64 copies of the all-clear: a race's draw count drifts frame to frame, so each
            // frame minted a fresh tuple and the dedup never converged. What is diagnostic is
            // the GAP - a draw that carried a payload but was not prepared vanished with no
            // fallback recorded, the shape that hid a composite for a whole session - and any
            // fixed-function item at all, which recompile-or-fail should have refused.
            //
            // `draws != with_payload` is deliberately NOT a trigger: a scene legitimately
            // holds draws that carry no shader payload, and those are already counted by
            // `items`. Triggering on it would reopen the same flood from the other side.
            let interesting = gxp_prepared.len() != with_payload || !items.is_empty();
            // Said ONCE, so "no tally lines" is distinguishable from "the tally never ran".
            if !interesting && !self.gxp_summary_all_clear {
                self.gxp_summary_all_clear = true;
                report_status!(
                    "gxp: scene tallies are live and CLEAN ({} draws, every payload prepared, \
                     no fixed-function items). Only DISAGREEMENTS are printed from here.",
                    summary.0,
                );
            }
            let fresh = interesting && !self.gxp_summaries_seen.contains(&summary);
            if fresh && self.gxp_summaries_seen.len() >= GXP_SUMMARY_CAP {
                if !self.gxp_summaries_capped {
                    self.gxp_summaries_capped = true;
                    report_warn!(
                        "gxp: {GXP_SUMMARY_CAP} distinct scene tallies reported; further NEW ones will not be printed."
                    );
                }
            } else if fresh {
                self.gxp_summaries_seen.insert(summary);
                // At `warn`, and deduped on the whole tuple, so a stable pass costs one
                // line: "carries a payload" and "was prepared" are different numbers, and
                // the gap between them is a draw that vanished with no fallback recorded -
                // which is precisely the shape that hid a composite for a whole session.
                report_warn!(
                    "gxp: scene has {} draws, {} carry a shader payload, {} recompiled+prepared, {} fixed-function items",
                    summary.0, summary.1, summary.2, summary.3,
                );
                // Rank the causes by the draws they cost, biggest first. Without this the
                // only ranking available is the printed pair list, which counts pairs.
                let mut ranked: Vec<_> = fb_reasons.iter().collect();
                ranked.sort_by(|a, b| b.1.cmp(a.1).then(a.0.cmp(b.0)));
                for (reason, n) in ranked {
                    report_warn!("gxp:   {n} fallback draws - {reason}");
                }
            }
        }

        self.last_phases = EncodePhases {
            prepare_ms: t_prepare.ms(),
            upload_ms: 0.0,
            pass_ms: 0.0,
            arena_ms: 0.0,
            arena_create_ms: 0.0,
            arena_write_ms: 0.0,
            ubo_bg_ms: 0.0,
            // A PASS has no chain-level work by definition - these belong to the frame and
            // are set on `chain_phases` directly, so folding a pass must not touch them.
            precompile_ms: 0.0,
            retire_ms: 0.0,
            resident_ms: 0.0,
            // ...and so are the four wall times: a PASS is inside the chain, not a chain.
            // `pass_wall_ms` is accumulated onto `chain_phases` at the bottom of this
            // function, after the fold, for exactly that reason.
            chain_head_ms: 0.0,
            scene_loop_ms: 0.0,
            pass_wall_ms: 0.0,
            chain_tail_ms: 0.0,
            negw_ms,
            clip_measures,
            head_sweep_ms: 0.0,
            head_scan_ms: 0.0,
            head_rtt_ms: 0.0,
            gxp_draws: gxp_prepared.len(),
            fixed_draws: items.len(),
        };

        // 2. Size the arenas and upload. Rebuild the uniform bind group if the uniform
        //    buffer was (re)created.
        let t_upload = Stopwatch::start();
        if !items.is_empty() {
            Self::ensure_buffer(device, queue, &mut self.vbo, &mut self.vbo_cap, vdata.len() as u64, wgpu::BufferUsages::VERTEX, "gxm-vbo", &mut self.buffer_pool);
            Self::ensure_buffer(device, queue, &mut self.ibo, &mut self.ibo_cap, idata.len() as u64, wgpu::BufferUsages::INDEX, "gxm-ibo", &mut self.buffer_pool);
            let ubo_new = Self::ensure_buffer(device, queue, &mut self.ubo, &mut self.ubo_cap, udata.len() as u64, wgpu::BufferUsages::UNIFORM, "gxm-ubo", &mut self.buffer_pool);
            if ubo_new || self.ubo_bind.is_none() {
                enc(&ENC.bind_groups_built, 1);
                let ubo = self.ubo.as_ref().unwrap();
                self.ubo_bind = Some(device.create_bind_group(&wgpu::BindGroupDescriptor {
                    label: Some("gxm-ubo-bind"),
                    layout: &self.uniform_layout,
                    entries: &[wgpu::BindGroupEntry {
                        binding: 0,
                        resource: wgpu::BindingResource::Buffer(wgpu::BufferBinding {
                            buffer: ubo,
                            offset: 0,
                            size: wgpu::BufferSize::new(UNIFORM_BYTES),
                        }),
                    }],
                }));
            }
            enc(&ENC.buffer_bytes, (vdata.len() + idata.len() + udata.len()) as u64);
            queue.write_buffer(self.vbo.as_ref().unwrap(), 0, &vdata);
            queue.write_buffer(self.ibo.as_ref().unwrap(), 0, &idata);
            queue.write_buffer(self.ubo.as_ref().unwrap(), 0, &udata);
        }
        // The recompiled path's arenas: THREE buffers for the whole pass, however many
        // draws it carries, instead of four per draw. This pass takes the slot for its
        // ORDINAL in the chain and keeps it across frames - see `gxp_arenas` - so the
        // steady state creates no buffer at all, and every buffer named by this frame's
        // commands stays alive through submit because the renderer owns it outright.
        let t_arena = Stopwatch::start();
        let (mut arena_create_ms, mut arena_write_ms) = (0.0f64, 0.0f64);
        let gxp_slot = (!gxp_prepared.is_empty()).then(|| {
            // The pass ORDINAL picks the family; the frame's phase picks which copy of it
            // this frame writes. Everything downstream - the arena handles and the uniform
            // bind-group cache key - takes this composite index, so a bind group is built
            // once per (pass, phase) and reused every `ring` frames rather than rebuilt.
            let slot = self.gxp_arena_slot * self.gxp_arena_ring + self.gxp_arena_phase;
            self.gxp_arena_slot += 1;
            Self::ensure_gxp_arena(
                device,
                queue,
                &mut self.gxp_staging,
                encoder,
                &mut self.gxp_arenas,
                &mut self.retired_buffers,
                slot,
                &gvdata,
                &gidata,
                &gudata,
                &self.gxp.udata_holes,
                &mut arena_create_ms,
                &mut arena_write_ms,
            );
            self.gxp_arenas[slot].last_used = self.chain_frames_seen;
            slot
        });
        self.last_phases.arena_ms = t_arena.ms();
        self.last_phases.arena_create_ms = arena_create_ms;
        self.last_phases.arena_write_ms = arena_write_ms;
        // After the arena's upload is recorded and before the pass that reads it: the window
        // bytes a raw target of this frame covers are overwritten with that target's words.
        if let Some(slot) = gxp_slot {
            self.fill_windows_from_raw_targets(encoder, slot, &gxp_prepared, current_target);
        }
        // ...and a store mirror this pass's draws enlarged takes the bytes of the mirrors it
        // replaced, in frame order, before any of them stores - see `store_mirror`.
        self.gxp.store_mirrors.flush(encoder, &mut self.retired_buffers);
        // >>> BACK ON THE RENDERER, AND SHRUNK IF THEY GREW PAST THE BOUND. `shrink_to`
        // keeps the length (already zero next pass, after the `clear` above) and gives the
        // rest of the allocation back, which is the half the reverted version did not do.
        if arena_pool_enabled() {
            self.pass_arenas = [vdata, idata, udata, gvdata, gidata, gudata];
            for v in self.pass_arenas.iter_mut() {
                v.clear();
                if v.capacity() > ARENA_RETAIN_BYTES {
                    v.shrink_to(ARENA_RETAIN_BYTES);
                }
            }
        }
        let t_ubo_bg = Stopwatch::start();
        if let Some(slot) = gxp_slot {
            let generation = self.gxp_arenas[slot].generation;
            let used: Vec<(u64, wgpu::TextureFormat, u32, u32, u64, u64, u32)> =
                gxp_prepared.iter().map(|p| (p.key, p.format, p.samples, p.cull, p.layout, p.raster, p.fmem_bytes)).collect();
            self.gxp.ensure_ubo_bgs(device, &self.gxp_arenas[slot].ubo, slot, generation, &used);
        }
        self.last_phases.ubo_bg_ms = t_ubo_bg.ms();

        self.last_phases.upload_ms = t_upload.ms();

        // 3. One render pass over the whole scene. When supersampling, the scene is drawn
        //    into the offscreen `scale x` target (built here) and a resolve pass below
        //    box-downsamples it into the caller's view; otherwise it is drawn straight in.
        let t_pass = Stopwatch::start();
        enc(&ENC.passes, 1);
        {
            // >>> WHERE THIS PASS HAS TO BE CUT, and it is the only thing that can cut it.
            //
            // A draw whose fragment program reads the DESTINATION colour needs a copy of
            // the attachment as it stands IMMEDIATELY BEFORE it, and a copy cannot be
            // encoded inside a render pass. So the pass becomes a sequence of segments:
            // segment 0 carries the clear and everything up to the first such draw, and
            // every later segment is preceded by the copy and loads what the previous one
            // stored. See `GxmRenderer::dest_color`.
            //
            // On a title with no such draw - which is every title measured here but one -
            // `bounds` is `[0, order.len()]`, the loop runs once, and every byte of this is
            // the single pass it always was.
            // ATTACHMENT texels per TARGET pixel, per axis. One on an offscreen pass, the
            // supersample factor on a supersampled display pass, and the panel-over-buffer
            // ratio on a display pass whose title declared a buffer smaller than the panel
            // - which is the case that was being computed wrong. See `att_w`'s doc.
            let sx = att_w as f32 / surf_w.max(1) as f32;
            let sy = att_h as f32 / surf_h.max(1) as f32;
            let full = (0.0f32, 0.0f32, att_w as f32, att_h as f32);
            // The guest's REGION CLIP, tracked exactly as the viewport is and for the
            // same reason: `set_scissor_rect` is sticky within a pass, so a draw that
            // wants the whole attachment after a scissored one has to say so. The pass
            // starts at wgpu's default (the whole attachment), and a title that never
            // sets a region clip therefore issues no scissor call at all.
            let full_sc = (0u32, 0u32, att_w, att_h);
            // Every draw's applied viewport and scissor, in ATTACHMENT texels, worked out
            // ONCE, before the cuts are placed: the draw loop applies exactly these, and the
            // destination copies below are bounded by exactly these, so the two cannot drift.
            let applied: Vec<((f32, f32, f32, f32), (u32, u32, u32, u32))> = order
                .iter()
                .enumerate()
                .map(|(oi, e)| {
                    // BOTH arms, from the one piece of guest state, because the viewport is
                    // the guest's and not a property of how we chose to draw it. The
                    // fixed-function arm used to pass `full` here on the reasoning that it
                    // "packs its own screen-space geometry and has never carried a
                    // viewport" - it does not pack it: an `Ndc`/`Mvp` draw reaches the
                    // shader in clip space, so a guest asking for half the target got all
                    // of it, on the arm every dropped pair falls back to.
                    let (vp_en, vp) = vports[oi];
                    // The negative control (`VITASLOP_GXM_VIEWPORT=0`) narrows to the arm
                    // that CHANGED: the recompiled arm keeps the viewport it always had.
                    let reaches =
                        matches!(e, Enc::Gxp(_)) || crate::gpu::viewport_reaches_fixed_function();
                    let want = match reaches
                        .then(|| gxm_viewport_rect(vp_en, &vp, surf_w, surf_h))
                        .flatten()
                    {
                        Some(r) => (r.0 * sx, r.1 * sy, r.2 * sx, r.3 * sy),
                        None => full,
                    };
                    // The guest's hardware scissor. `rect_in` works in ATTACHMENT texels, so
                    // the clip is scaled with the pass exactly as the viewport is.
                    // Scaled by the EDGES, not by (origin, extent): scaling a width
                    // independently of its origin lets rounding move the far edge by a texel,
                    // which on a clip that is meant to reach the edge of the frame leaves a
                    // seam. Both edges are clamped to the attachment, because a guest rectangle
                    // that reaches the target's last pixel must still be inside it after the
                    // ratio is applied, and wgpu rejects a scissor that leaves the attachment.
                    let want_sc = clips[oi]
                        .rect_in(surf_w, surf_h)
                        .map(|(x, y, w, h)| {
                            let x0 = ((x as f32 * sx).floor() as u32).min(att_w);
                            let y0 = ((y as f32 * sy).floor() as u32).min(att_h);
                            let x1 = (((x + w) as f32 * sx).ceil() as u32).clamp(x0, att_w);
                            let y1 = (((y + h) as f32 * sy).ceil() as u32).clamp(y0, att_h);
                            (x0, y0, x1 - x0, y1 - y0)
                        })
                        .unwrap_or(full_sc);
                    (want, want_sc)
                })
                .collect();
            let mut bounds: Vec<usize> = vec![0];
            // Which PAIR asked for each split. The count alone says the cost; this says
            // where to spend the fix, because a pair drawn twelve times pays twelve splits
            // and is worth lowering to a fixed-function blend, while one drawn once is not.
            let mut split_pairs: Vec<u64> = Vec::new();
            // Diagnostic (`VITASLOP_GXP_PASS_SPLIT_EVERY=<n>`): cut the pass every `n`
            // draws as well, with no shader changed. A picture that goes wrong when the
            // DESTINATION lowering is turned off has two possible causes - the ALU form
            // the recompiler then emits, and the render-pass CUT that form forces - and
            // nothing separates them, because the knob that reveals one always brings the
            // other. This brings the cut ALONE: same shaders, same draws, same order.
            let split_every = crate::gpu::forced_pass_split();
            // `VITASLOP_GXM_DEST_SPLIT_AB`: in the OFF arm the destination readers keep their
            // shaders and their place in the order and simply do not cut the pass, so the
            // frame's ONLY difference is the splits and their copies. The picture is wrong
            // for those draws while it is off; that is the price of the stopwatch.
            let ab_armed = crate::gpu::dest_split_armed();
            // >>> A READER THAT TOUCHES NOTHING WRITTEN SINCE THE LAST COPY SHARES IT - see
            // >>> `DestDirty`. Only where every cut really takes the copy (the attachment has
            // a texture and this pass made its copy texture); elsewhere every reader cuts, as
            // before. `VITASLOP_GXM_DEST_COALESCE=0` is the arm back: a cut and a whole copy
            // in front of every reader.
            let coalesce = crate::gpu::dest_coalesce_enabled() && dest_ok && dest_source.is_some();
            let mut dirty = DestDirty::new();
            // The region each cut's destination copy covers, by segment (index 0 unused - the
            // first segment takes no copy): `None` the whole attachment, `Some` a rectangle
            // (possibly empty, when nothing was written since the last copy).
            let mut copy_regions: Vec<Option<[u32; 4]>> = vec![None];
            // Readers that shared the copy before them instead of cutting - the saving.
            let mut shared_reads = 0u64;
            // >>> EACH DRAW'S FOOTPRINT IS WHERE ITS VERTICES PROJECT, inside its viewport and
            // >>> scissor - see `draw_pixel_bounds` - ASKED ONLY WHERE THE ANSWER DECIDES A CUT.
            //
            // Every draw starts from its COARSE footprint (viewport and scissor, free). A reader
            // interprets its own vertices only when that coarse footprint meets something written
            // since the last copy, and then the writers it meets, newest first, stopping at the
            // first that still overlaps (`DestDirty::needs_copy_refining`). So the pass's first
            // reader (it always takes a whole copy), every draw after its last reader, and a
            // writer behind one that already forced the cut are never interpreted. On top of
            // that the pass must EARN it (`DestGeomGate`): MEASURED on a platformer's 4x
            // multisampled world pass, its skinned readers overlap one another anyway, and
            // interpreting 112 of its 122 draws cost 33 ms of a phone frame to share 4 copies.
            // `VITASLOP_GXM_DEST_GEOM=0` is the arm back to the coarse footprint alone.
            let readers = order.iter().filter(|e| matches!(e, Enc::Gxp(idx) if gxp_prepared[*idx].needs_dest)).count();
            // Priced per target AND per order of magnitude of readers: a title encodes one target
            // several times a frame (a one-reader pass beside a 130-reader one), and their
            // verdicts must not overwrite each other.
            let gate_key = (Self::pass_id(scene), readers.next_power_of_two().trailing_zeros());
            let geom_budget = (coalesce && ab_armed && readers > 0 && crate::gpu::dest_geom_bounds_enabled())
                .then(|| dest_geom_gate_begin(gate_key))
                .flatten();
            let coarse: Vec<[u32; 4]> =
                (0..order.len()).map(|i| draw_footprint(applied[i].0, applied[i].1, att_w, att_h)).collect();
            let yflip = self.gxp.yflip;
            let (mut geom_bounded, mut geom_refused, mut geom_us) = (0u64, 0u64, 0f64);
            // Each draw's refined footprint, worked out at most once: `refine(i, false)` only
            // says whether it is known yet, `refine(i, true)` works it out (or, past the budget
            // or for a draw it cannot bound, settles on the coarse one).
            let mut fine: Vec<Option<[u32; 4]>> = vec![None; order.len()];
            let mut refine = |i: usize, compute: bool| -> Option<[u32; 4]> {
                if fine[i].is_some() || !compute {
                    return fine[i];
                }
                let fp = coarse[i];
                let bounded = match (geom_budget, &order[i]) {
                    (Some(budget), Enc::Gxp(idx)) if geom_us < budget => {
                        scene.draws[gxp_draw_of[*idx]].gxp.as_ref().map(|g| {
                            let t = Stopwatch::start();
                            let b = super::clip::draw_pixel_bounds(g, gxp_prepared[*idx].clip_sign, yflip, applied[i].0, DEST_GEOM_MAX_VERTS);
                            geom_us += t.ms() * 1000.0;
                            b
                        })
                    }
                    _ => None,
                };
                let f = match bounded {
                    Some(Some(b)) => {
                        geom_bounded += 1;
                        let clamp = |v: f32, max: u32, up: bool| -> u32 {
                            let v = if up { v.ceil() } else { v.floor() };
                            v.clamp(0.0, max as f32) as u32
                        };
                        let gx0 = clamp(b[0], att_w, false).max(fp[0]);
                        let gy0 = clamp(b[1], att_h, false).max(fp[1]);
                        let gx1 = clamp(b[2], att_w, true).min(fp[2]);
                        let gy1 = clamp(b[3], att_h, true).min(fp[3]);
                        // Empty when it misses the viewport: the draw rasterises nothing.
                        if gx0 >= gx1 || gy0 >= gy1 { [0; 4] } else { [gx0, gy0, gx1, gy1] }
                    }
                    Some(None) => {
                        geom_refused += 1;
                        fp
                    }
                    None => fp,
                };
                fine[i] = Some(f);
                Some(f)
            };
            // The same pass with COARSE footprints only, run beside it: the cuts the vertices
            // saved are the difference, and that is what the gate prices.
            let mut coarse_dirty = DestDirty::new();
            let (mut cuts, mut coarse_cuts) = (0u64, 0u64);
            for (i, e) in order.iter().enumerate() {
                let fp = coarse[i];
                let forced = split_every.is_some_and(|n| i > 0 && i % n == 0);
                if geom_budget.is_some() {
                    let self_cut = matches!(e, Enc::Gxp(_)) && i > 0 && self_cuts.contains_key(&i);
                    let reader = matches!(e, Enc::Gxp(idx) if gxp_prepared[*idx].needs_dest);
                    if forced || self_cut || (reader && coarse_dirty.needs_copy(fp)) {
                        coarse_cuts += 1;
                        coarse_dirty.take_copy();
                    }
                    coarse_dirty.wrote(fp);
                }
                let cut = if forced {
                    Some(0)
                } else if let Enc::Gxp(idx) = e
                    && i > 0
                    && self_cuts.contains_key(&i)
                {
                    // A draw reading a window of this pass's own target - see `self_cuts`.
                    Some(gxp_prepared[*idx].key)
                } else if let Enc::Gxp(idx) = e
                    && gxp_prepared[*idx].needs_dest
                    && ab_armed
                {
                    let need = if !coalesce {
                        true
                    } else if geom_budget.is_some() {
                        dirty.needs_copy_refining(i, fp, &mut |d| refine(d, true).unwrap_or(coarse[d]))
                    } else {
                        dirty.needs_copy(fp)
                    };
                    if need {
                        Some(gxp_prepared[*idx].key)
                    } else {
                        shared_reads += 1;
                        None
                    }
                } else {
                    None
                };
                if let Some(k) = cut {
                    cuts += 1;
                    bounds.push(i);
                    split_pairs.push(k);
                    // Every cut takes the destination copy (see the segment loop), so every
                    // cut restarts the tracking. A multisampled copy is a draw over the whole
                    // attachment and is never narrowed.
                    let region = dirty.take_copy();
                    copy_regions.push(if coalesce && samples == 1 { region } else { None });
                }
                if geom_budget.is_none() {
                    dirty.wrote(fp);
                } else if let Some(f) = refine(i, false) {
                    dirty.wrote(f);
                } else {
                    dirty.wrote_coarse(fp, i);
                }
            }
            if shared_reads > 0 {
                enc(&ENC.dest_reads_shared, shared_reads);
            }
            if geom_budget.is_some() {
                enc(&ENC.dest_geom_bounded, geom_bounded);
                enc(&ENC.dest_geom_refused, geom_refused);
                enc(&ENC.dest_geom_us, geom_us as u64);
                dest_geom_gate_end(gate_key, coarse_cuts.saturating_sub(cuts), geom_us);
            }
            bounds.push(order.len());
            let mut ts = self.ts_pair(|| match scene.target.as_ref() {
                Some(t) => format!("{}x{} {} draws", t.width, t.height, order.len()),
                None => format!("depth-only {} draws", order.len()),
            });
            // Under `VITASLOP_GXP_PASS_SPLIT_EVERY` EVERY segment is timed, not only the first:
            // that is the instrument's whole point - the GPU cost of each n-draw slice, so a
            // pass's time can be put on the draws that spend it. Taken here, before the draw
            // loop borrows the renderer's buffers.
            let mut seg_pairs: Vec<Option<(std::sync::Arc<wgpu::QuerySet>, u32)>> =
                if split_every.is_some() && ts.is_some() {
                    (1..bounds.len() - 1)
                        .map(|s| self.ts_pair(|| format!("seg draws {}..{}", bounds[s], bounds[s + 1])))
                        .collect()
                } else {
                    Vec::new()
                };
            // Draw in submission order, switching between the fixed-function arenas and the
            // recompiled per-draw resources. The fixed-function handles are unwrapped only
            // inside a Fixed arm, where `items` is non-empty so the arenas were uploaded.
            let ubo_bind = self.ubo_bind.as_ref();
            let vbo = self.vbo.as_ref();
            let ibo = self.ibo.as_ref();
            // Unwrapped only inside a `Gxp` arm, where `gxp_prepared` is non-empty so this
            // pass took an arena slot above.
            let gxp_arena = gxp_slot.map(|s| &self.gxp_arenas[s]);
            // The guest's viewport is per-draw state, and `set_viewport` is sticky, so a
            // draw that wants the whole target after one that did not must SAY so. The
            // pass starts at the full rect (wgpu's default), and the tracker below issues
            // a change only when the requested rect actually differs - which on a title
            // whose every pass is fullscreen means it never issues one at all. The rects
            // themselves are `applied`, worked out above.
            // The clip the previous draw carried, so the report below runs on a CHANGE
            // rather than per draw. `report_region_clip_applied` takes a mutex and hits a
            // hash set; at ~500 draws a frame that would make the instrument a measurable
            // part of the `pass` phase it sits in, which is the one thing a diagnostic in
            // this loop must not be.
            // >>> REDUNDANT-STATE ELIMINATION WAS BUILT HERE AND REMOVED. Do not re-add it
            // without measuring the PHASE first.
            //
            // Skipping a `set_pipeline`/`set_bind_group` whose value is already bound is
            // safe (pass state is sticky within a pass) and it demonstrably works: MEASURED
            // on the user's phone, a race frame went from 975 pipeline sets to 458, and a
            // static screen came out BIT-IDENTICAL (0.00% of pixels, max channel delta 0).
            // It was still deleted, because it bought NOTHING - render 22.9 ms with it
            // against 22.7 ms without, same device, same screen, same build.
            //
            // The reason is in the phase split, and it is the part worth keeping: these
            // calls are counted in the `pass` phase, and `pass` is 1.1 ms of a 19.2 ms
            // encode, against `prepare` 10.1 ms and `upload` 7.6 ms. Halving the call count
            // of the smallest phase cannot move a frame however large the count looks next
            // to the draw count. RANK BY PHASE FIRST, THEN BY COUNT: counting alone picked
            // the one item in the frame that could not pay.
            //
            // Counted in LOCALS and folded once after the loop, not per call: these are the
            // hottest lines in the renderer and an atomic per `set_bind_group` would make
            // the instrument a measurable part of what it measures.
            let (mut n_vp, mut n_pipe, mut n_bg, mut n_vb, mut n_draw) = (0u64, 0u64, 0u64, 0u64, 0u64);
            // Draws the colour-no-op elision did not issue - see the arm that counts it.
            let mut n_elided = 0u64;
            let mut n_sc = 0u64;
            let mut splits_taken = 0u64;
            // `VITASLOP_GXM_DRAW_COVERAGE`: this pass's ordinal in the frame, and the query
            // set the pass descriptor names. The set is cloned out of `self` BEFORE the pass
            // exists because the descriptor holds a reference to it for the pass's whole
            // lifetime, while the draw loop keeps borrowing `self` for pipelines and bind
            // groups - the same reason `ts_writes` takes an `Arc` rather than a field.
            let cov_pass = self.coverage.as_mut().map_or(0, |c| {
                c.pass += 1;
                c.pass - 1
            });
            let cov_set = self.coverage.as_ref().map(|c| c.set.clone());
            for seg in 0..bounds.len() - 1 {
                let (start, end) = (bounds[seg], bounds[seg + 1]);
                let last_seg = seg + 2 == bounds.len();
                // The copy that makes the next draw's destination read real. Before the
                // segment, outside any render pass, which is the whole reason the pass is
                // cut here at all.
                // The windows read under their reader's own pixels, taken once, before any draw:
                // the surface as the pass found it.
                if seg == 0
                    && let Some(tex) = color_texture
                {
                    for (&(x, y, width, height), snap) in &scene_start {
                        encoder.copy_texture_to_texture(
                            wgpu::TexelCopyTextureInfo {
                                texture: tex,
                                mip_level: 0,
                                origin: wgpu::Origin3d { x, y, z: 0 },
                                aspect: wgpu::TextureAspect::All,
                            },
                            snap.as_image_copy(),
                            wgpu::Extent3d { width, height, depth_or_array_layers: 1 },
                        );
                    }
                }
                // The windows of this pass's own target the segment's first draw reads, as
                // the draws before it left them (before segment 0: as the pass found it).
                if let (Some(copies), Some(tex)) = (self_cuts.get(&start), color_texture) {
                    // Through a tile shadow, the window stands for MEMORY - the packed form of
                    // what the draws before it left - so the shadow is packed into the real
                    // attachment first. Not before segment 0: the shadow has not been cleared
                    // yet, and the real attachment still holds the surface as the pass found it.
                    if seg > 0 && tile.is_some() {
                        self.texel_blit(device, encoder, color_view, real_view, real_format);
                    }
                    for c in copies {
                        let (texture, origin) = match &c.from {
                            Some(snap) => (snap, wgpu::Origin3d::ZERO),
                            None => (tex, c.origin),
                        };
                        encoder.copy_texture_to_texture(
                            wgpu::TexelCopyTextureInfo { texture, mip_level: 0, origin, aspect: wgpu::TextureAspect::All },
                            c.dst.as_image_copy(),
                            c.size,
                        );
                    }
                }
                if seg > 0 {
                    let region = copy_regions[seg];
                    if let Some(tex) = dest_source {
                        self.copy_attachment_to_dest(device, encoder, tex, dest_shape, region);
                    }
                    splits_taken += 1;
                    enc(&ENC.dest_splits, 1);
                    enc(
                        &ENC.dest_split_bytes,
                        match region {
                            Some([x0, y0, x1, y1]) => texture_bytes(x1.saturating_sub(x0), y1.saturating_sub(y0)),
                            None => texture_bytes(att_w, att_h),
                        } as u64,
                    );
                }
                // Timestamps belong to the pass as a whole, so the first segment carries
                // them and the rest run untimed rather than each claiming a pair.
                let seg_ts = if seg == 0 {
                    ts.take()
                } else {
                    seg_pairs.get_mut(seg - 1).and_then(Option::take)
                };
                // Viewport, scissor and the clip tracker are PASS state: a new segment
                // starts at wgpu's defaults however the previous one left them.
                let mut cur_vp = full;
                let mut cur_sc = full_sc;
                let mut last_clip: Option<RegionClip> = None;
                // What this pass already has bound. A render pass starts with nothing, which
                // is exactly what `BoundState::new` records.
                let mut bound = BoundState::new();
                let mut pass = encoder.begin_render_pass(&wgpu::RenderPassDescriptor {
                    label: Some("gxm-scene"),
                    color_attachments: &[Some(wgpu::RenderPassColorAttachment {
                        view: color_view,
                        depth_slice: None,
                        // The multisample resolve, when this pass has multisampled
                        // attachments: ending the pass writes the resolved image into the
                        // stored-size texture every later pass samples. `StoreOp::Store` below
                        // still applies to the multisampled attachment itself, which has to
                        // survive so a SECOND pass into the same target composes onto this
                        // one's samples rather than onto its resolve.
                        resolve_target: resolve,
                        ops: wgpu::Operations {
                            // Only the FIRST segment clears. A later one loads what the
                            // previous segment stored, because the whole reason it exists is
                            // that a draw in it reads those pixels.
                            load: match (seg == 0).then_some(clear).flatten() {
                                Some(c) => wgpu::LoadOp::Clear(c),
                                None => wgpu::LoadOp::Load,
                            },
                            store: wgpu::StoreOp::Store,
                        },
                    })],
                    depth_stencil_attachment: Some(wgpu::RenderPassDepthStencilAttachment {
                        view: depth_view,
                        depth_ops: Some(wgpu::Operations {
                            // Same rule as the colour: a later segment carries on with the
                            // depth the earlier ones wrote, or every draw after a split would
                            // sort against an empty buffer.
                            load: if seg == 0 && !self.depth_load_now {
                                wgpu::LoadOp::Clear(scene.depth_clear)
                            } else {
                                wgpu::LoadOp::Load
                            },
                            // Discarded by default - nothing reads a depth attachment once its
                            // pass is over. A pass whose depth a LATER pass samples has to keep
                            // it, and only that pass pays the store.
                            // A middle segment ALWAYS stores: the next one loads it. Only the
                            // last segment gets to discard, and only under the same rule as
                            // before (a pass whose depth a later pass samples keeps it).
                            store: if self.keep_depth || !last_seg || zls_store(scene) {
                                wgpu::StoreOp::Store
                            } else {
                                wgpu::StoreOp::Discard
                            },
                        }),
                        // STENCIL follows the depth rule exactly: the first segment starts
                        // at the surface's background value, a later one carries on with
                        // what the earlier ones marked - a HUD's `EQUAL` fills after a split
                        // would otherwise test against a cleared buffer.
                        // A stencil plane LOADED from a store mirror (`stencil_load_now`) loads
                        // whatever the depth does - see `load_guest_stencil`.
                        stencil_ops: (!crate::gpu::stencil_disabled()).then(|| wgpu::Operations {
                            load: if seg == 0 && !self.depth_load_now && !self.stencil_load_now {
                                wgpu::LoadOp::Clear(u32::from(scene.stencil_clear))
                            } else {
                                wgpu::LoadOp::Load
                            },
                            store: if self.keep_stencil || !last_seg || zls_store(scene) {
                                wgpu::StoreOp::Store
                            } else {
                                wgpu::StoreOp::Discard
                            },
                        }),
                    }),
                    timestamp_writes: Self::ts_writes(&seg_ts),
                    occlusion_query_set: cov_set.as_deref(),
                    multiview_mask: None,

                });
                for (oi, e) in order.iter().enumerate().take(end).skip(start) {
                    // The viewport and scissor worked out for this draw above - see `applied`.
                    let (want, want_sc) = applied[oi];
                    if want != cur_vp {
                        pass.set_viewport(want.0, want.1, want.2, want.3, 0.0, 1.0);
                        cur_vp = want;
                        n_vp += 1;
                    }
                    if last_clip != Some(clips[oi]) {
                        last_clip = Some(clips[oi]);
                        crate::gpu::report_region_clip_applied(clips[oi], surf_w, surf_h);
                    }
                    // Diagnostic (`VITASLOP_GXP_VP_TRACE`): the viewport rect this draw
                    // ACTUALLY got, beside the guest viewport it came from and the surface
                    // and attachment it was mapped through. A draw that lands in a box on an
                    // otherwise empty frame is either geometry or this, and nothing else in
                    // the renderer states the applied rect.
                    if crate::gpu::vp_trace() {
                        let gvp = vports[oi].1;
                        let k = match e {
                            Enc::Gxp(idx) => gxp_prepared[*idx].key,
                            Enc::Fixed(_) => 0,
                        };
                        report!(
                            "gxp viewport: draw #{oi} key {k:016x}: guest {gvp:?} on surface {surf_w}x{surf_h} -> attachment {att_w}x{att_h} rect {want:?} clip mode {:#x} {:?} -> scissor {want_sc:?}",
                            clips[oi].mode,
                            clips[oi].rect
                        );
                    }
                    if want_sc != cur_sc {
                        // A zero-area scissor is legal in wgpu and draws nothing, which is
                        // exactly what SCE_GXM_REGION_CLIP_ALL asks for.
                        pass.set_scissor_rect(want_sc.0, want_sc.1, want_sc.2, want_sc.3);
                        cur_sc = want_sc;
                        n_sc += 1;
                    }
                    // `VITASLOP_GXM_DRAW_COVERAGE`: claim this draw's occlusion query, before
                    // anything below borrows `self` for a pipeline or a bind group. The index
                    // is claimed even for a draw the arms below decline to issue, and such a
                    // draw opens and closes an EMPTY query - which resolves to zero, and zero
                    // is the truth about a draw that wrote nothing. Leaving the slot unwritten
                    // instead would resolve to whatever the buffer last held and read as a
                    // draw that covered something.
                    let q = self.coverage.as_mut().and_then(|c| {
                        let (key, idx) = match e {
                            Enc::Gxp(i) => (gxp_prepared[*i].key, gxp_prepared[*i].index_count),
                            Enc::Fixed(i) => (0, items[*i].index_count),
                        };
                        c.take(cov_pass, oi, key, idx)
                    });
                    if let Some(qi) = q {
                        pass.begin_occlusion_query(qi);
                    }
                    match e {
                        Enc::Fixed(i) => {
                            let it = &items[*i];
                            let (ubo_bind, vbo, ibo) = (ubo_bind.unwrap(), vbo.unwrap(), ibo.unwrap());
                            // A fixed-function draw on a gamma-correct surface needs the sRGB
                            // variant, and one in a multisampled pass needs the multisampled
                            // variant, for the same reason a recompiled one does: a pipeline is
                            // bound to its attachments' format AND sample count. The two never
                            // combine - a gamma target is refused multisampling. A FLOAT target
                            // is its own format, multisampled or not, and is chosen by the
                            // format itself: "not the default format" is not "sRGB".
                            let (op, bl) = if target_format == FLOAT_TARGET_FORMAT {
                                let (o, b) = if samples > 1 { &self.float_ms } else { &self.float };
                                (o, b)
                            } else {
                                match (&self.srgb, target_format == self.color_format) {
                                    (Some((o, b)), false) => (o, b),
                                    _ if samples > 1 => (&self.opaque_ms, &self.blend_ms),
                                    _ => (&self.opaque, &self.blend),
                                }
                            };
                            match it.stencil {
                                // A masked draw: its variant, built at item prep for this
                                // pass's format and sample count, and its reference.
                                Some(k) => {
                                    let pipe = &self.fixed_stencil[&(it.opaque, target_format, samples, k)];
                                    pass.set_pipeline(pipe);
                                    pass.set_stencil_reference(it.stencil_ref);
                                }
                                None => pass.set_pipeline(if it.opaque { op } else { bl }),
                            }
                            pass.set_bind_group(0, ubo_bind, &[it.uniform_offset]);
                            pass.set_bind_group(1, self.bind_for(it.bind), &[]);
                            pass.set_vertex_buffer(0, vbo.slice(it.v_off..it.v_off + it.v_len));
                            pass.set_index_buffer(ibo.slice(it.i_off..it.i_off + it.i_len), wgpu::IndexFormat::Uint32);
                            pass.draw_indexed(0..it.index_count, 0, 0..1);
                            // The fixed-function path binds its own everything, and what it
                            // left bound is not what `BoundState` last recorded - so the next
                            // recompiled draw must rebind rather than trust a stale record.
                            bound.invalidate();
                            n_pipe += 1;
                            n_bg += 2;
                            n_vb += 2;
                            n_draw += 1;
                        }
                        Enc::Gxp(idx) => {
                            let p = &gxp_prepared[*idx];
                            // >>> A DRAW WITH NO GEOMETRY IS THE ONE FAILURE THIS PATH CANNOT
                            // >>> SHOW. Everything else about a prepared draw is visible from
                            // outside - its pipeline, its bindings, its pass - and a draw whose
                            // index count or vertex slice came out EMPTY is encoded, submitted,
                            // and rasterises nothing, which is indistinguishable on screen from
                            // a draw that was never issued. Fires once per pair, only on the
                            // failure, so it costs nothing on a working title.
                            crate::gpu::report_empty_gxp_geometry(p.key, p.index_count, p.v_len, p.i_len);
                            // >>> A PIPELINE THE DEVICE REFUSED IS NOT BOUND, because binding it
                            // would invalidate the pass, the command buffer and therefore every
                            // OTHER draw in the frame. See `note_device_error`, which is what
                            // learns the key, and which reports it loudly the first time.
                            if crate::gpu::gxp_pair_poisoned(p.key) {
                                crate::gpu::POISONED_DRAWS
                                    .fetch_add(1, std::sync::atomic::Ordering::Relaxed);
                                // Close the query this draw opened - see the claim above.
                                if q.is_some() {
                                    pass.end_occlusion_query();
                                }
                                continue;
                            }
                            // >>> A DRAW WHOSE COLOUR IS THE DESTINATION AND WHICH WRITES
                            // >>> NO DEPTH HAS NO EFFECT, so it is not issued at all.
                            //
                            // BIT-IDENTICAL by construction rather than by comparison: the
                            // colour this draw would have written is the colour already in
                            // the attachment, proven from its own uniform bytes
                            // (`draw_colour_is_noop`), and with `depth_write` clear there is
                            // no other thing a draw does. A draw that DOES write depth still
                            // owes that write and is issued below - nothing here assumes
                            // nobody reads the depth buffer.
                            //
                            // The occlusion query is closed exactly as the poisoned arm
                            // closes it, so an elided draw reads zero samples and shows up
                            // in `VITASLOP_GXM_DRAW_COVERAGE` under COVERED NOTHING with its
                            // index count - which is how the saving is measured.
                            if p.colour_noop && !p.depth_write && !p.stencil_write {
                                n_elided += 1;
                                if q.is_some() {
                                    pass.end_occlusion_query();
                                }
                                continue;
                            }
                            let arena = gxp_arena.unwrap();
                            // Geometry the renderer has resident is bound where it LIVES; only
                            // what changed this frame comes from the pass arena. Both handles
                            // are read here, at encode time, which is why the resident heap may
                            // never be recreated mid-frame - see [`ResidentHeap`].
                            let gxp_vbo = if p.v_resident {
                                self.gxp.resident_v.buf.as_ref().expect("a resident slice implies its buffer")
                            } else {
                                &arena.vbo
                            };
                            let gxp_ibo = if p.i_resident {
                                self.gxp.resident_i.buf.as_ref().expect("a resident slice implies its buffer")
                            } else {
                                &arena.ibo
                            };
                            let slot = gxp_slot.unwrap();
                            let pipe = self.gxp.pipeline(p.key, p.format, p.samples, p.cull, p.layout, p.raster);
                            if bound.pipeline_changed(&pipe.pipeline) {
                                pass.set_pipeline(&pipe.pipeline);
                                n_pipe += 1;
                            }
                            // The stencil reference is dynamic pass state: set whenever it
                            // differs from the last one set in this pass.
                            if !crate::gpu::stencil_disabled() && (!bound.on || bound.stencil_ref != Some(p.stencil_ref)) {
                                pass.set_stencil_reference(p.stencil_ref);
                                bound.stencil_ref = Some(p.stencil_ref);
                            }
                            // group0/group1 belong to the PAIR and take this draw's byte offset
                            // into the pass's uniform arena; a stage with no uniforms has an
                            // empty bind group, which takes no dynamic offsets at all.
                            //
                            // >>> ON THE STACK, BECAUSE THIS RAN ONCE PER DRAW AND ALLOCATED.
                            //
                            // The offsets used to be built as `Vec`s (`vec![off]`, plus a
                            // `push` for the memory window), which is two heap allocations and
                            // two frees for every recompiled draw - ~1,150 a frame on a title
                            // that submits ~580. There are at most two offsets in either group,
                            // both `u32`, so a fixed array and a slice of its filled prefix says
                            // exactly the same thing to wgpu with no allocator in the loop. The
                            // browser is where that matters: the allocator is a measured 7.3% of
                            // the busy worker on this title's race
                            // (a racer, the race), and a wasm malloc
                            // is dearer than a native one. The bytes handed to `set_bind_group`
                            // are unchanged, so the frame is bit-identical.
                            //
                            // group 0's dynamic offsets in BINDING order: the SA block, then
                            // the guest-memory window when this pipeline declares one.
                            let mut g0_offs = [0u32; 2];
                            let mut g0_n = 0usize;
                            if pipe.vsa_lanes != 0 {
                                g0_offs[g0_n] = p.u_off[0];
                                g0_n += 1;
                            }
                            if pipe.mem_bind_bytes > 0 {
                                g0_offs[g0_n] = p.u_off[2];
                                g0_n += 1;
                            }
                            let bg0 = self.gxp.ubo_bg(slot, p.key, p.format, p.samples, 0, 0);
                            if bound.group_changed(0, bg0, &g0_offs[..g0_n]) {
                                pass.set_bind_group(0, bg0, &g0_offs[..g0_n]);
                                n_bg += 1;
                            }
                            // group 1 in BINDING order, like group 0: the fragment SA
                            // block, then its guest-memory window when declared.
                            let mut g1_offs = [0u32; 2];
                            let mut g1_n = 0usize;
                            if pipe.fsa_lanes != 0 {
                                g1_offs[g1_n] = p.u_off[1];
                                g1_n += 1;
                            }
                            if pipe.fmem_bind_bytes > 0 {
                                g1_offs[g1_n] = p.u_off[3];
                                g1_n += 1;
                            }
                            let bg1 = self.gxp.ubo_bg(slot, p.key, p.format, p.samples, 1, p.fmem_bytes);
                            if bound.group_changed(1, bg1, &g1_offs[..g1_n]) {
                                pass.set_bind_group(1, bg1, &g1_offs[..g1_n]);
                                n_bg += 1;
                            }
                            if bound.group_changed(2, &p.bg2, &[]) {
                                pass.set_bind_group(2, &p.bg2, &[]);
                                n_bg += 1;
                            }
                            if bound.group_changed(3, &p.bg3, &[]) {
                                pass.set_bind_group(3, &p.bg3, &[]);
                                n_bg += 1;
                            }
                            if bound.vbo_changed(gxp_vbo, p.v_off, p.v_len) {
                                pass.set_vertex_buffer(0, gxp_vbo.slice(p.v_off..p.v_off + p.v_len));
                                n_vb += 1;
                            }
                            if bound.ibo_changed(gxp_ibo, p.i_off, p.i_len, p.index_u32) {
                                pass.set_index_buffer(
                                    gxp_ibo.slice(p.i_off..p.i_off + p.i_len),
                                    if p.index_u32 {
                                        wgpu::IndexFormat::Uint32
                                    } else {
                                        wgpu::IndexFormat::Uint16
                                    },
                                );
                                n_vb += 1;
                            }
                            pass.draw_indexed(0..p.index_count, 0, 0..1);
                            n_draw += 1;
                        }
                    }
                    if q.is_some() {
                        pass.end_occlusion_query();
                    }
                }
            }
            enc(&ENC.passes, splits_taken);
            if splits_taken > 0 {
                report_dest_split_count(splits_taken, order.len(), att_w, att_h, &split_pairs);
            }
            enc(&ENC.viewport_sets, n_vp);
            enc(&ENC.scissor_sets, n_sc);
            enc(&ENC.pipeline_sets, n_pipe);
            enc(&ENC.bind_group_sets, n_bg);
            enc(&ENC.vertex_buffer_sets, n_vb);
            enc(&ENC.draw_calls, n_draw);
            enc(&ENC.draws_elided, n_elided);
        }
        // The tile emit: the shadow packed into the surface's own format, once - rounded to
        // 8 bits, and sRGB-encoded when `real_view` is the gamma view.
        if tile.is_some() {
            self.texel_blit(device, encoder, color_view, real_view, real_format);
        }
        self.last_phases.pass_ms = t_pass.ms();
        self.chain_phases.add(self.last_phases);
        // AFTER the fold, because the fold deliberately does not carry this field.
        self.chain_phases.pass_wall_ms += t_pass_wall.ms();
        // The GPU-side arenas are NOT retired here any more: this pass's slot keeps them
        // for the next frame's pass of the same ordinal. Only a slot that had to GROW
        // retires a buffer, and `ensure_gxp_arena` puts that one in the graveyard itself.
    }
}

/// Whether a pass is drawn through a [`GxmRenderer::tile_shadow`]: some draw blends for itself
/// at register precision (`any_wide_blender`), and the attachment is one an 8-bit pack stands
/// between - a single-sampled RGBA8/BGRA8 (or its sRGB view) whose texture can be drawn into
/// and sampled at the attachment's own extent.
///
/// Everything else keeps its attachment: a float or raw 64-bit target already holds the
/// halves; a multisampled pass would need a multisampled shadow and a resolve the shadow does
/// not model; a single-channel alpha surface writes through its own channel remap; and a view
/// with no texture behind it (or one that cannot be sampled) cannot be seeded or resolved.
pub(super) fn tile_shadow_applies(
    target_format: wgpu::TextureFormat,
    samples: u32,
    resolves: bool,
    // `(format, sample count, usage, width, height)` of the texture behind the attachment.
    texture: Option<(wgpu::TextureFormat, u32, wgpu::TextureUsages, u32, u32)>,
    att: (u32, u32),
    alpha_single: bool,
    any_wide_blender: bool,
) -> bool {
    use wgpu::TextureFormat as F;
    let eight_bit = matches!(target_format.remove_srgb_suffix(), F::Rgba8Unorm | F::Bgra8Unorm);
    let need = wgpu::TextureUsages::RENDER_ATTACHMENT | wgpu::TextureUsages::TEXTURE_BINDING;
    let texture_ok = texture.is_some_and(|(format, count, usage, w, h)| {
        format.remove_srgb_suffix() == target_format.remove_srgb_suffix()
            && count == 1
            && usage.contains(need)
            && (w, h) == att
    });
    any_wide_blender && eight_bit && samples == 1 && !resolves && !alpha_single && texture_ok
}

/// The most distinct vertices [`super::clip::draw_pixel_bounds`] interprets for one draw
/// before it gives up and the viewport stands. A self-blending effect is a few quads; a draw
/// past this is a mesh whose footprint is rarely small, and interpreting it every frame would
/// cost more CPU than the split it might save.
const DEST_GEOM_MAX_VERTS: usize = 256;

/// The attachment pixels a draw CAN write, as `[x0, y0, x1, y1)` (exclusive far edges): its
/// applied viewport rectangle (`(x, y, width, height)` in attachment texels, fractional) cut by
/// its applied scissor (`(x, y, width, height)`), clamped to the `att_w` x `att_h` attachment.
///
/// Conservative by construction, and it has to be - see [`DestDirty`]. A primitive is clipped
/// to the clip volume, so its window coordinates lie inside the viewport rectangle, and a pixel
/// is only covered when a sample point inside it is - so no covered pixel lies outside
/// `floor(x) .. ceil(x + width)`. The scissor is exact in whole pixels. An empty rectangle
/// (`x0 >= x1` or `y0 >= y1`) is a draw that writes nothing.
pub(super) fn draw_footprint(vp: (f32, f32, f32, f32), sc: (u32, u32, u32, u32), att_w: u32, att_h: u32) -> [u32; 4] {
    let edge = |v: f32, ceil: bool, max: u32| -> u32 {
        let v = if ceil { v.ceil() } else { v.floor() };
        // A NaN viewport cannot be bounded: treat it as reaching the far edge (and from 0).
        if v.is_nan() {
            return if ceil { max } else { 0 };
        }
        v.clamp(0.0, max as f32) as u32
    };
    let (vx0, vy0) = (edge(vp.0, false, att_w), edge(vp.1, false, att_h));
    let (vx1, vy1) = (edge(vp.0 + vp.2, true, att_w), edge(vp.1 + vp.3, true, att_h));
    let sx1 = sc.0.saturating_add(sc.2).min(att_w);
    let sy1 = sc.1.saturating_add(sc.3).min(att_h);
    [vx0.max(sc.0), vy0.max(sc.1), vx1.min(sx1), vy1.min(sy1)]
}

/// >>> WHICH PIXELS OF THE DESTINATION COPY ARE STALE, so a destination reader that touches
/// >>> none of them can share the copy the previous reader took.
///
/// A self-blending draw reads the destination at ITS OWN PIXEL only (`textureLoad` at
/// `frag_coord`), from a copy of the attachment. That copy is right for every pixel nothing
/// has written since it was taken. So a reader whose footprint ([`draw_footprint`]) meets no
/// draw issued since the last copy reads exactly what a fresh copy would hand it, and the pass
/// does not have to be cut for it - BIT-IDENTICAL by construction, not by comparison. One that
/// does meet such a draw takes a copy, as every reader used to.
///
/// The copy itself is bounded the same way: right after a copy the copy equals the attachment
/// everywhere, so the next one only has to refresh what was written in between - the bounding
/// box of the dirty rectangles. The pass's FIRST copy is whole: the copy texture is shared by
/// every pass of the attachment's shape, so on entry it holds some other pass's pixels.
pub(super) struct DestDirty {
    /// A copy has been taken in THIS pass, so the copy texture equals the attachment outside
    /// `rects`.
    copied: bool,
    /// What the draws since that copy can have written, `[x0, y0, x1, y1)` each - and, for a
    /// rectangle that is still a draw's COARSE footprint ([`Self::wrote_coarse`]), which draw,
    /// so a reader that meets it can have it refined.
    rects: Vec<([u32; 4], Option<usize>)>,
}

impl DestDirty {
    /// Rectangles kept apart before they are merged into their bounding box. Merging only ever
    /// GROWS the dirty set, which can cost a cut but never skips one.
    const CAP: usize = 64;

    pub(super) fn new() -> Self {
        Self { copied: false, rects: Vec::new() }
    }

    fn overlaps(a: &[u32; 4], b: &[u32; 4]) -> bool {
        a[0] < b[2] && b[0] < a[2] && a[1] < b[3] && b[1] < a[3]
    }

    fn is_empty(r: &[u32; 4]) -> bool {
        r[0] >= r[2] || r[1] >= r[3]
    }

    /// Whether a destination reader with footprint `f` needs a fresh copy: always before the
    /// pass's first copy, and afterwards only when something written since meets `f`.
    pub(super) fn needs_copy(&self, f: [u32; 4]) -> bool {
        if Self::is_empty(&f) {
            // Writes nothing, so it reads nothing either - but before the first copy the
            // texture is not this pass's at all, and a cut there is the old behaviour.
            return !self.copied;
        }
        !self.copied || self.rects.iter().any(|r| Self::overlaps(&r.0, &f))
    }

    /// [`Self::needs_copy`] for reader `draw` whose COARSE footprint is `coarse`, asking
    /// `refine` for a draw's exact footprint (a subset of its coarse one) only where the answer
    /// can change: nothing is refined when the coarse footprint already meets nothing, the
    /// reader is refined only then, and the coarse writers it still meets are refined newest
    /// first, stopping at the first that overlaps it - that one forces the cut on its own.
    pub(super) fn needs_copy_refining(&mut self, draw: usize, coarse: [u32; 4], refine: &mut dyn FnMut(usize) -> [u32; 4]) -> bool {
        if !self.needs_copy(coarse) {
            return false;
        }
        if !self.copied {
            return true;
        }
        let f = refine(draw);
        if Self::is_empty(&f) {
            return false;
        }
        for (r, coarse_of) in self.rects.iter_mut().rev() {
            if !Self::overlaps(r, &f) {
                continue;
            }
            if let Some(w) = coarse_of.take() {
                *r = refine(w);
                if !Self::overlaps(r, &f) {
                    continue;
                }
            }
            return true;
        }
        false
    }

    /// A copy is being taken now. Returns the rectangle it has to cover - `None` for the whole
    /// attachment (the pass's first copy) - and starts tracking afresh.
    pub(super) fn take_copy(&mut self) -> Option<[u32; 4]> {
        let region = self.copied.then(|| self.bbox());
        self.copied = true;
        self.rects.clear();
        region
    }

    /// A draw with footprint `f` has been issued.
    pub(super) fn wrote(&mut self, f: [u32; 4]) {
        self.push(f, None);
    }

    /// Draw `draw` has been issued and only its COARSE footprint `f` is known yet; a reader that
    /// meets it may refine it later ([`Self::needs_copy_refining`]).
    pub(super) fn wrote_coarse(&mut self, f: [u32; 4], draw: usize) {
        self.push(f, Some(draw));
    }

    fn push(&mut self, f: [u32; 4], draw: Option<usize>) {
        // Absorbed only by an EXACT rectangle that contains it: a coarse one may yet shrink.
        let contained = |r: &([u32; 4], Option<usize>)| {
            r.1.is_none() && r.0[0] <= f[0] && r.0[1] <= f[1] && r.0[2] >= f[2] && r.0[3] >= f[3]
        };
        if Self::is_empty(&f) || self.rects.iter().any(contained) {
            return;
        }
        self.rects.push((f, draw));
        if self.rects.len() > Self::CAP {
            let b = self.bbox();
            self.rects.clear();
            self.rects.push((b, None));
        }
    }

    /// The bounding box of everything written since the last copy; empty (`[0; 4]`) when
    /// nothing was. A rectangle refined down to nothing adds nothing.
    fn bbox(&self) -> [u32; 4] {
        self.rects
            .iter()
            .map(|r| r.0)
            .filter(|r| !Self::is_empty(r))
            .reduce(|a, b| [a[0].min(b[0]), a[1].min(b[1]), a[2].max(b[2]), a[3].max(b[3])])
            .unwrap_or([0; 4])
    }
}

/// >>> A PASS'S VERTEX FOOTPRINTS MUST PAY FOR THEMSELVES - per pass, measured every frame.
///
/// Interpreting a reader's vertices can save that reader's cut and copy, and nothing else, so
/// it is worth at most what a cut costs. The vertex programs that reach this differ tenfold in
/// cost: an action title's world effects are short programs over a few quads (about 0.6 us a
/// vertex on the desktop, and 124 of 128 cuts saved), a platformer's skinned readers are long
/// ones over meshes of a hundred vertices (5-8 us a vertex, and its readers overlap one another,
/// so 4 of 43 cuts saved for 45 ms). So each pass is priced: the cuts the vertices saved
/// (against the same pass with coarse footprints, run beside it) at [`Self::CUT_WORTH_US`]
/// each, against the CPU it spent. A pass that did not earn it skips the vertices for a
/// doubling number of encodes, then probes again with a small budget, and a pass that did
/// earn it may spend up to what it earned. Only the CPU spent is decided here: any footprint,
/// coarse or exact, gives the same picture (`DestDirty`).
pub(super) struct DestGeomGate {
    /// Encodes still to skip before the next probe.
    skip: u32,
    /// What the next unprofitable verdict skips.
    backoff: u32,
    /// The CPU, in microseconds, the next encode may spend on vertices.
    budget_us: f64,
}

impl DestGeomGate {
    /// What one cut saved is credited with, in CPU microseconds. Deliberately below what a cut
    /// costs: MEASURED on the phone, 125 fewer cuts took 45 ms off the GPU, about 360 us each.
    const CUT_WORTH_US: f64 = 100.0;
    /// The budget of a probe - a first sight of a pass, or one coming back from a skip.
    const PROBE_US: f64 = 2000.0;
    const BACKOFF_MIN: u32 = 8;
    const BACKOFF_MAX: u32 = 512;

    pub(super) fn new() -> Self {
        Self { skip: 0, backoff: Self::BACKOFF_MIN, budget_us: Self::PROBE_US }
    }

    /// The budget for this encode of the pass, or `None` to use coarse footprints only.
    pub(super) fn begin(&mut self) -> Option<f64> {
        if self.skip > 0 {
            self.skip -= 1;
            return None;
        }
        Some(self.budget_us)
    }

    /// The verdict on an encode that asked: `saved` cuts against `spent_us` of CPU.
    pub(super) fn end(&mut self, saved: u64, spent_us: f64) {
        let worth = saved as f64 * Self::CUT_WORTH_US;
        if saved == 0 && spent_us <= 0.0 {
            // Nothing asked, nothing saved: no evidence either way, and it must not reset a
            // budget an earlier encode earned.
        } else if worth >= spent_us {
            self.backoff = Self::BACKOFF_MIN;
            self.budget_us = worth.max(Self::PROBE_US);
        } else {
            self.skip = self.backoff;
            self.backoff = (self.backoff * 2).min(Self::BACKOFF_MAX);
            self.budget_us = Self::PROBE_US;
        }
    }
}

/// Which gate prices an encode: the target (`GxmRenderer::pass_id`) and the order of magnitude
/// of its reader count.
type DestGeomKey = (Option<(u32, u32, u32, u32)>, u32);

/// The CPU budget for vertex footprints in this encode of pass `pass`, or `None` for coarse
/// footprints only - see `DestGeomGate`. `VITASLOP_GXM_DEST_GEOM_GATE=0` bypasses the gate: every
/// encode may spend without limit.
fn dest_geom_gate_begin(pass: DestGeomKey) -> Option<f64> {
    if !crate::gpu::dest_geom_gate_enabled() {
        return Some(f64::INFINITY);
    }
    with_dest_geom_gate(pass, DestGeomGate::begin)
}

/// [`DestGeomGate::end`] for pass `pass`.
fn dest_geom_gate_end(pass: DestGeomKey, saved: u64, spent_us: f64) {
    if crate::gpu::dest_geom_gate_enabled() {
        with_dest_geom_gate(pass, |g| g.end(saved, spent_us));
    }
}

fn with_dest_geom_gate<R>(pass: DestGeomKey, f: impl FnOnce(&mut DestGeomGate) -> R) -> R {
    use std::sync::{Mutex, OnceLock};
    static GATES: OnceLock<Mutex<HashMap<DestGeomKey, DestGeomGate>>> = OnceLock::new();
    let mut g = GATES.get_or_init(|| Mutex::new(HashMap::default())).lock().unwrap_or_else(|e| e.into_inner());
    // Bounded: a title renders into a handful of targets; a churn of addresses is cleared, and
    // a cleared pass only probes again.
    if g.len() >= 256 && !g.contains_key(&pass) {
        g.clear();
    }
    f(g.entry(pass).or_insert_with(DestGeomGate::new))
}

/// The rectangle a draw can write, `(x0, y0, x1, y1)` in target pixels: its viewport `vp`,
/// narrowed by an OUTSIDE region clip (which discards every pixel outside its rectangle).
fn own_pixels(vp: (f32, f32, f32, f32), clip: crate::gpu::RegionClip, w: u32, h: u32) -> (f32, f32, f32, f32) {
    if clip.mode & 0xC000_0000 == crate::gpu::RegionClip::OUTSIDE
        && let Some((cx, cy, cw, ch)) = clip.rect_in(w, h)
    {
        let (cx, cy, cw, ch) = (cx as f32, cy as f32, cw as f32, ch as f32);
        return (vp.0.max(cx), vp.1.max(cy), vp.2.min(cx + cw), vp.3.min(cy + ch));
    }
    vp
}

/// Whether the `w` x `h` window at `(x0, y0)` overlaps the rectangle `own` a draw writes - a
/// window the draw itself covers, which a tile GPU reads from memory as the scene found it.
fn window_under(own: (f32, f32, f32, f32), x0: u32, y0: u32, w: u32, h: u32) -> bool {
    let (a0, b0, a1, b1) = (x0 as f32, y0 as f32, (x0 + w) as f32, (y0 + h) as f32);
    a0 < own.2 && own.0 < a1 && b0 < own.3 && own.1 < b1
}

/// One self-window refill before a draw that reads a window of its own pass's target - see the
/// self-window block in `encode_chain`.
struct SelfCopy {
    /// The window texture the draw samples.
    dst: wgpu::Texture,
    /// Where the window sits in the attachment, in attachment texels.
    origin: wgpu::Origin3d,
    size: wgpu::Extent3d,
    /// `Some(snapshot)` when the window lies under the reading draw's own pixels: it is filled
    /// from the attachment as the pass FOUND it, not as the draws before it left it.
    from: Option<wgpu::Texture>,
}

#[cfg(test)]
mod self_window_tests {
    use super::{own_pixels, window_under};
    use crate::gpu::RegionClip;

    const FULL: (f32, f32, f32, f32) = (0.0, 0.0, 384.0, 192.0);

    /// A history strip read by the draw that blends into it lies under the draw (the shooter's
    /// reflection pass: whole-target viewport, OUTSIDE clip to rows 128-191, strip 384x64 at
    /// row 128); the same strip read by a draw clipped to the top rows does not.
    #[test]
    fn a_window_the_draw_covers_is_under_it_and_one_beside_it_is_not() {
        let strip = |own| window_under(own, 0, 128, 384, 64);
        let clip = |rect| RegionClip { mode: RegionClip::OUTSIDE, rect };
        assert!(strip(own_pixels(FULL, clip([0, 128, 383, 191]), 384, 192)), "the blend into the strip");
        assert!(!strip(own_pixels(FULL, clip([256, 0, 383, 63]), 384, 192)), "a reflection draw in the top rows");
        // With no clip the viewport alone decides: a 128x64 viewport in the top-right corner.
        let none = RegionClip { mode: RegionClip::NONE, rect: [0; 4] };
        assert!(!strip(own_pixels((256.0, 0.0, 384.0, 64.0), none, 384, 192)));
    }

    /// A mip chain built inside one surface reads each level from beside the draw writing the
    /// next, so none of its reads is under the reader - the earlier-draws rule still applies.
    #[test]
    fn a_mip_level_read_beside_the_next_is_not_under_its_reader() {
        let none = RegionClip { mode: RegionClip::NONE, rect: [0; 4] };
        // Level 0 at 256x128 on the left of a 512x128 surface; level 1 drawn at x 256..384.
        let level1 = own_pixels((256.0, 0.0, 384.0, 64.0), none, 512, 128);
        assert!(!window_under(level1, 0, 0, 256, 128));
        // Touching edges are not an overlap.
        assert!(!window_under((256.0, 0.0, 384.0, 64.0), 0, 0, 256, 64));
    }
}
