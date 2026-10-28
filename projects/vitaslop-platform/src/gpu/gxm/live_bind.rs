//! [`GxpLive`]: looking up a prepared draw's pipeline and uniform groups, the raster key,
//! the deferred texture encodes, and the group2 sampler bind group.

use super::*;

impl GxpLive {
    /// The cached group0/group1 bind group for a prepared draw's shader pair (only called
    /// after `ensure_ubo_bgs` has run for this pass). `fmem_bytes` is the draw's per-draw
    /// fragment window block size (`GxpPrepared::fmem_bytes`), read for group 1 only.
    pub(super) fn ubo_bg(&self, slot: usize, key: u64, format: wgpu::TextureFormat, samples: u32, group: u8, fmem_bytes: u32) -> &wgpu::BindGroup {
        let fmem = if group == 1 { fmem_bytes } else { 0 };
        &self.ubo_bgs[&(slot, key, format, samples, group, fmem)].0
    }

    /// The cached pipeline for a prepared draw (only called after `prepare` succeeded).
    /// The per-draw RASTERISER state a pipeline bakes in that the pair, format, samples,
    /// cull and vertex layout do not already name: the guest's polygon offset.
    ///
    /// # Why this has to be in the pipeline key
    /// `sceGxmSetFrontDepthBias` is per-DRAW state, and a `wgpu::RenderPipeline` fixes it at
    /// creation. Without it here the FIRST draw of a pair decides the polygon offset for
    /// every later draw of that pair - which is the same defect shape as the vertex layout
    /// above, and it cost exactly as much.
    ///
    /// MEASURED on the golf title's shadow pass: its 234 depth-only draws carry TWO biases,
    /// `(factor 1, units 0)` on 9,625 draws and `(0, 0)` on 7,873, and whichever arrived
    /// first won. A slope-scaled offset is the standard cure for shadow-map self-shadowing,
    /// so dropping it put the whole course - and the character standing on it - into its own
    /// shadow. The map has no picture of its own, so nothing said so.
    pub(super) fn raster_key(gxp: &GxpRecompile, colour_noop: bool) -> u64 {
        let mut h: u64 = 0xcbf2_9ce4_8422_2325;
        let mut mix = |v: u64| {
            h ^= v;
            h = h.wrapping_mul(0x0000_0100_0000_01b3);
        };
        mix(gxp.depth_bias.0 as u32 as u64);
        mix(gxp.depth_bias.1 as u32 as u64);
        // The DEPTH TEST and DEPTH WRITE are per-draw state for exactly the same reason and
        // were missing for exactly as long. A title that draws one pair with `LessEqual +
        // write` and again with the test off - a depth prepass, a decal, a second-depth
        // shadow caster - got whichever variant reached `prepare` first, and the other draws
        // then tested against a rule the guest did not ask for. Blend does NOT belong here:
        // it is baked into the fragment PROGRAM and is already folded into the pair `key`.
        mix(gxp.depth_func as u64);
        mix(gxp.depth_write as u64);
        // The TOPOLOGY is baked into the pipeline too, and a title may draw one pair as
        // triangles and again as lines. Without this the second one gets whichever
        // pipeline reached `prepare` first and renders as the wrong primitive.
        mix(gxp.primitive as u64);
        // A draw whose colour is provably the destination takes a DEPTH-ONLY pipeline -
        // same vertex stage, a trivial fragment stage, the colour write mask empty. That is
        // per-draw pipeline state exactly like the depth rule above it, and it belongs in
        // this key for the same reason: the same pair is submitted both ways in one frame,
        // and whichever reached `prepare` first would otherwise serve both.
        mix(colour_noop as u64);
        // The STENCIL test, ops and masks are pipeline state for the same reason as the
        // depth rule: a HUD draws one pair both as a stencil-marking `NEVER` quad and as an
        // `EQUAL` fill. The REFERENCE is not here - it is dynamic, set per draw.
        if !crate::gpu::stencil_disabled() {
            for (i, v) in gxp.stencil.iter().enumerate() {
                if i != 6 && i != 13 {
                    mix(*v as u64);
                }
            }
            mix(gxp.two_sided as u64);
        }
        h
    }

    pub(super) fn pipeline(&self, key: u64, format: wgpu::TextureFormat, samples: u32, cull: u32, layout: u64, raster: u64) -> &GxpPipeline {
        self.pipelines
            .get(&(key, format, samples, cull, layout, raster))
            .and_then(|p| p.as_ref())
            .expect("prepared key present")
    }

    /// >>> ENCODE A SLICE OF THE DEFERRED ETC2 TEXTURES AND SWAP IN THE ONES THAT FINISHED.
    ///
    /// Called once per frame, before anything is prepared, so a swap never lands between a
    /// draw that named the interim and the submit that reads it. The finished texture's copy
    /// is on the pump's submit, which precedes this frame's, so its first sampler sees the
    /// blocks. The interim is DESTROYED, not dropped (WebGPU completes any work already
    /// submitted against it first), and every cached sampler group naming its entry is
    /// dropped exactly as an eviction drops them.
    pub(super) fn pump_deferred_encodes(&mut self, device: &wgpu::Device, queue: &wgpu::Queue) {
        use std::sync::atomic::Ordering;
        let Some(te) = self.texenc.as_ref() else { return };
        te.pump(device, queue, crate::gpu::tex_encode_defer_units());
        for (key, tex) in te.take_finished() {
            let k = (key, SamplerDim::Two);
            match self.views.get_mut(&k) {
                Some(entry) => {
                    let view = tex.create_view(&wgpu::TextureViewDescriptor {
                        dimension: Some(SamplerDim::Two.view_dimension()),
                        ..Default::default()
                    });
                    let (old, _) = std::mem::replace(entry, (tex, view));
                    note_texture_destroyed();
                    old.destroy();
                    self.sampler_bgs.retain(|_, (_, named, _)| !named.contains(&k));
                    self.last_sampler_bg = None;
                    self.sampler_pre.clear();
                    crate::gpu::DEFER_SWAPPED.fetch_add(1, Ordering::Relaxed);
                }
                // Evicted while it encoded: nothing names it, and a later bind re-uploads.
                None => {
                    note_texture_destroyed();
                    tex.destroy();
                    crate::gpu::DEFER_ORPHANED.fetch_add(1, Ordering::Relaxed);
                }
            }
        }
    }

    /// Build the group2 sampler bind group: for each declared sampler unit, upload the
    /// bound texture and bind it with the matching filter sampler. `None` (fall back) if a
    /// unit has no bound texture or needs a 3D texture (not yet mapped).
    #[allow(clippy::too_many_arguments)]
    pub(super) fn make_sampler_bg(
        device: &wgpu::Device,
        queue: &wgpu::Queue,
        layout: &wgpu::BindGroupLayout,
        stages: &[(&[(u8, SamplerDim)], &[crate::gpu::GxpTex])],
        key: u64,
        view_cache: &mut HashMap<(u64, SamplerDim), (wgpu::Texture, wgpu::TextureView)>,
        // Which uploaded texture is the CURRENT one for a given guest texture, and which
        // entries a newer upload has displaced - see `GxpLive::view_slots`.
        view_slots: &mut HashMap<(u64, SamplerDim), u64>,
        view_dead: &mut HashSet<(u64, SamplerDim)>,
        view_cache_bytes: &mut usize,
        views_used: &mut HashMap<(u64, SamplerDim), (u64, usize, u32, Residency)>,
        views_evicted: &mut HashSet<(u64, SamplerDim)>,
        views_epoch: &u64,
        views_frame_high: &usize,
        views_frame_bytes: &mut usize,
        samplers_by_mode: &mut HashMap<(bool, u32, u32), wgpu::Sampler>,
        force: bool,
        rendered: &HashMap<u32, wgpu::TextureView>,
        // See `prepare`: the linear twin of each gamma-correct target in `rendered`.
        rendered_linear: &HashMap<u32, wgpu::TextureView>,
        // ...and the sRGB twin of each LINEAR target in `rendered` - see `Bind::RenderedSrgb`.
        rendered_srgb: &HashMap<u32, wgpu::TextureView>,
        depth_rendered: &HashMap<u32, wgpu::TextureView>,
        // See `prepare`: the depth addresses a depth-ONLY pass owns, where being in both
        // maps is correct rather than a bug.
        depth_only: &HashSet<u32>,
        // Cube views of the cube maps whose faces the guest RENDERED, by the address of
        // face 0 - matched before either 2D path, because face 0's address is also an
        // ordinary render target and `rendered_alias` would bind that one face flat.
        rendered_cubes: &HashMap<u32, wgpu::TextureView>,
        // The 64-bit targets this frame rendered as RAW words, by address - the only thing
        // a `SamplerDim::Raw` slot binds. See `GxmRenderer::rtt_rendered_raw`.
        rendered_raw: &HashMap<(u32, u32, u32), wgpu::TextureView>,
        // Which addresses currently resolve to the SNAPSHOT rather than the live target:
        // the two views of one address are different textures, so which one a group named
        // has to be in its key. (The `rtt_epoch` used to ride here too and no longer does -
        // a group is keyed by the VIEW'S OWN IDENTITY now, for the reason written out at
        // the `Bind::Rendered` arm below.)
        reads_snapshot: &HashSet<u32>,
        sampler_bgs: &mut SamplerBgCache,
        // The previous draw's answer - see `GxpLive::last_sampler_bg`.
        last_sampler_bg: &mut Option<(u64, wgpu::BindGroup)>,
        // Every answer this PASS has produced, by the same fingerprint - see
        // `GxpLive::sampler_pre`.
        sampler_pre: &mut HashMap<u64, u64>,
        // Which block family this device accepts - resolved once per renderer by the caller.
        bc: BlockFamily,
        // The GPU transcoder's pipelines, built once per renderer by the caller.
        texenc: &crate::texenc::Transcoder,
    ) -> Option<wgpu::BindGroup> {
        // What this group will NAME, accumulated as the units resolve, so the finished
        // group can be looked up instead of rebuilt. See `GxpLive::sampler_bgs`.
        let mut sel: u64 = 0xcbf2_9ce4_8422_2325;
        let mut mix = |v: u64| {
            sel ^= v;
            sel = sel.wrapping_mul(0x0000_0100_0000_01b3);
        };
        // A view that belongs to a target THIS frame rendered cannot be cached across
        // frames - the texture behind it is the frame's, not the cache's.
        let mut per_frame = false;
        // Both stages' samplers share this group, in declaration order: the fragment's
        // first, the vertex's after. The layout, the WGSL and this must agree on that order
        // or a sample reads the wrong texture.
        let total: usize = stages.iter().map(|(s, _)| s.len()).sum();
        if total == 0 {
            // An EMPTY group is still a GPU object, and building one per draw is the
            // same waste as building a full one per draw. It names nothing, so one per
            // pipeline is all there can ever be.
            let (bg, _, used) = sampler_bgs.entry((key, sel)).or_insert_with(|| {
                note_sampler_bg(false);
                let bg = device.create_bind_group(&wgpu::BindGroupDescriptor {
                    label: Some("gxp-samplers-empty"),
                    layout,
                    entries: &[],
                });
                // Names nothing, so no eviction can ever invalidate it.
                (bg, Vec::new(), *views_epoch)
            });
            // A hit here is a use, like every other probe of this cache.
            *used = *views_epoch;
            return Some(bg.clone());
        }
        // >>> WHAT EACH UNIT WILL BIND, NOT THE VIEW ITSELF - and that is a measurement,
        // >>> not tidiness.
        //
        // The finished group is a pure function of what it names, so it is looked up at the
        // bottom of this function and, on a title's steady frame, FOUND: measured in a
        // browser, ~424 groups reused against ~1 built per frame. Everything this loop used
        // to produce - a `TextureView` clone per unit, and three `Vec` allocations per draw
        // - was then dropped unused on 424 draws in 425, at 4.3 us a draw and **2.10 ms a
        // frame, 42% of the whole `prepare` phase**, for a phase that was building almost
        // nothing.
        //
        // So the loop now records a PLAN: one small enum per unit saying which view to take
        // and the sampler state to take it through, in one allocation. The views are cloned
        // only after the cache has said it does not have the group. Everything the loop
        // still does eagerly is there because it has an EFFECT the cache cannot replay - the
        // texture upload, the view-cache insert, the superseded-slot bookkeeping, and the
        // `views_used` stamp that keeps a view this frame needs from being evicted.
        // >>> THE PREVIOUS DRAW'S GROUP, WHEN THIS DRAW NAMES THE SAME THINGS.
        //
        // `sampler_bgs` below already answers "have I built this group before" - but only
        // AFTER the loop has resolved every unit, and that resolution is the cost: a map
        // probe of the view cache, a `views_used` stamp and a `Residency` decision per unit
        // per draw, ~3,400 times a frame on this title, plus a `Vec` per draw. A batch of
        // draws sharing a material asks for exactly the same group each time.
        //
        // The fingerprint is what DECIDES the group - the pair, and each unit's content key
        // and sampler state, in binding order. Everything else the resolution consults is
        // constant WITHIN a pass (this frame's rendered targets, the snapshot set, the view
        // cache - a hit performs no upload and no eviction, so nothing can move between two
        // consecutive hits), and the slot is cleared at every pass boundary, which is where
        // those do change. The eager effects a hit skips were performed by the call it
        // matches, in this frame: the same `views_used` stamp, the same upload.
        //
        // MEASURED, golf gameplay, browser, `NO_BC=1`: **190.5 of 543.6 draws a frame take
        // this**, i.e. a third of the frame's draws resolve no unit at all - about a
        // thousand view-cache probes and `views_used` stamps a frame that stop happening.
        // The DESKTOP CANNOT PRICE IT (`samplers` 1.93-1.98 ms against 2.04-2.31 before,
        // inside this machine's run-to-run spread), which is the same standing as the
        // sampler narrowing: a count a slower device pays more for than this one does. The
        // frame is bit-identical.
        let pre = {
            let mut h: u64 = 0x9e37_79b9_7f4a_7c15 ^ key;
            for &(samplers, textures) in stages {
                for &(unit, want) in samplers {
                    h ^= (unit as u64) << 3 | want as u64;
                    h = h.wrapping_mul(0x0000_0100_0000_01b3);
                    let k = textures
                        .iter()
                        .find(|t| t.unit == unit)
                        .map_or(0, |gt| {
                            let (linear, u, v) = sampler_mode(&gt.tex);
                            gt.tex.key
                                ^ ((linear as u64) << 62)
                                ^ ((u as u64) << 32)
                                ^ v as u64
                        });
                    h ^= k;
                    h = h.wrapping_mul(0x0000_0100_0000_01b3);
                }
            }
            h
        };
        if let Some((last, bg)) = last_sampler_bg.as_ref()
            && *last == pre {
                note_sampler_bg_prev();
                return Some(bg.clone());
            }
        // ...and, failing that, ANY draw of this pass that decided the same thing. The
        // stored value is the CONTENT key, so the group still comes out of `sampler_bgs`
        // and a key that has been invalidated since simply misses. See
        // `GxpLive::sampler_pre`.
        if let Some(&known) = sampler_pre.get(&pre)
            && let Some((bg, _, used)) = sampler_bgs.get_mut(&(key, known)) {
                *used = *views_epoch;
                let bg = &*bg;
                note_sampler_bg_pass();
                *last_sampler_bg = Some((pre, bg.clone()));
                return Some(bg.clone());
            }
        let mut plan: Vec<(Bind, (bool, u32, u32))> = Vec::with_capacity(total);
        for (stage_i, &(samplers, textures)) in stages.iter().enumerate() {
        // Fragment first, vertex after - the order the comment above `total` pins.
        let stage_name = if stage_i == 0 { "fragment" } else { "vertex" };
        for &(unit, want) in samplers {
            let bound = textures.iter().find(|t| t.unit == unit);
            // The bound texture must actually supply the dimension the shader declared: a
            // cube sampler needs the six captured faces, a 2D sampler a single image. A
            // mismatch means the container and the guest state disagree, so bind nothing.
            // >>> A CUBE SAMPLER OVER A FLAT TEXTURE IS SAMPLED, NOT REFUSED.
            //
            // A container can declare a unit as a CUBE (`sceGxmProgramParameterIsSamplerCube`)
            // and the guest can bind a flat texture to it - which is not a capture defect:
            // `faces` comes from the guest's OWN texture-type selector, so one face means the
            // control word does not say cube. MEASURED on a football title: four pairs, units
            // 4, 5 and 6, each with an **8x8 UBC3** texture bound - a dummy, the shape a title
            // binds when the effect that unit feeds is switched off. Those four pairs were
            // DROPPED, and a dropped pair's mesh is absent from the frame.
            //
            // The image is REPLICATED into six faces (see `upload_gxp_texture`), so the
            // sample returns a real texel of the real texture rather than a fabricated
            // colour. It is not exactly what the hardware returns - a 2-D texture addressed
            // by three coordinates uses (u,v) directly, while a cube view picks a face from
            // the major axis and then its own (u,v) - so for a texture that is NOT flat the
            // texel can differ. On an 8x8 dummy that is at most one of 64 texels against a
            // whole mesh missing, and it is reported once per pair so a wrong colour later
            // has a named suspect. The exactly-faithful form is to declare the unit 2-D in
            // the module and drop the third coordinate, which is a per-draw SHADER variant
            // (the module cache is keyed by pair) - worth doing when a title binds a flat
            // texture to a cube unit it actually reads a picture from.
            let usable = bound.filter(|gt| match want {
                SamplerDim::Cube => gt.tex.faces == 6 || gt.tex.faces == 1,
                SamplerDim::Two | SamplerDim::Raw => gt.tex.faces == 1,
                // Only a texture that IS on the float seam fits an unfilterable slot.
                SamplerDim::TwoF32 => gt.tex.faces == 1 && gt.tex.texel == TexelSeam::Rg32Float,
                SamplerDim::Three => false,
            });
            if want == SamplerDim::Cube && usable.is_some_and(|gt| gt.tex.faces == 1) {
                report_flat_texture_on_cube_unit(key, unit, usable.map_or(0, |gt| gt.tex.base_format));
            }
            // >>> A RAW slot binds ONLY a 64-bit target this frame rendered as raw words -
            // that is the whole reason the pipeline was linked with the unit raw (see
            // `prepare`, which sets the bit from this same map). Nothing else can satisfy a
            // `texture_2d<u32>` binding, so a raw slot with no such target binds nothing.
            if want == SamplerDim::Raw {
                let raw_key = |gt: &crate::gpu::GxpTex| (gt.tex.data_addr, gt.tex.width, gt.tex.height);
                match usable.filter(|gt| rendered_raw.contains_key(&raw_key(gt))) {
                    Some(gt) => {
                        if crate::gpu::rtt_bg_cache() {
                            mix(0x0e0e_0000_0000_0000
                                ^ gt.tex.data_addr as u64
                                ^ rendered_raw.get(&raw_key(gt)).map_or(0, view_identity));
                        } else {
                            per_frame = true;
                        }
                        // A NON-FILTERING slot takes a nearest sampler only.
                        plan.push((
                            Bind::RenderedRaw(raw_key(gt)),
                            (false, gt.tex.addr_mode_u | RAW_SAMPLER_FLAG, gt.tex.addr_mode_v),
                        ));
                    }
                    None => {
                        report_raw_slot_unbound(key, unit);
                        return None;
                    }
                }
                continue;
            }
            // >>> A CUBE MAP THE GUEST RENDERED, matched exactly by the address of face 0.
            //
            // This used to be impossible by construction: both render-target paths below
            // are gated on `SamplerDim::Two`, on the stated assumption that "a cube face is
            // never a GXM render target", so a rendered cube fell through to an upload of
            // GUEST MEMORY - which the GPU wrote and the guest never did, so the reflection
            // sampled stale or empty bytes and nothing said so
            // ([[vitaslop-a-render-target-reads-empty-in-guest-memory]]). MEASURED on
            // a golf title: ten pairs sample a cube at unit 11 naming `0x891e6520`, whose six
            // faces the same frame renders as six 256x256 passes 0x40000 apart.
            //
            // Checked BEFORE the two 2D paths for the same reason the depth path is: an
            // address that is a cube's face 0 is also a plain render target, and
            // `rendered_alias` would happily claim it and bind ONE FACE as a 2D texture.
            let cube_hit = (want == SamplerDim::Cube)
                .then(|| usable.and_then(|gt| rendered_cubes.get(&gt.tex.data_addr).map(|_| gt.tex.data_addr)))
                .flatten();
            if let (Some(gt), Some(addr)) = (usable, cube_hit) {
                if crate::gpu::rtt_bg_cache() {
                    mix(0x0b0b_0000_0000_0000 ^ addr as u64 ^ rendered_cubes.get(&addr).map_or(0, view_identity));
                } else {
                    per_frame = true;
                }
                plan.push((
                    Bind::RenderedCube(addr),
                    sampler_mode(&gt.tex),
                ));
                continue;
            }
            // A DEPTH buffer this frame rendered, matched EXACTLY and checked FIRST.
            //
            // Order is load-bearing. A title allocates a scene's depth next to its colour
            // (one racer puts them 256 bytes apart), so `rendered_alias`, which matches by
            // range, claims the depth address for the colour target and the pass reads a
            // colour where it asked for a distance - which is why its glow, blur and
            // soft-particle passes rendered pure black. Exact-matching the depth first is
            // what tells the two apart, and the address comes from the guest's own
            // `SceGxmDepthStencilSurface`, not from a guess about the layout.
            //
            // >>> AND AN ADDRESS THE FRAME RENDERED AS *COLOUR* IS NEVER A DEPTH SURFACE.
            // The two maps are keyed by different guest addresses and a depth-ONLY pass -
            // the one case where a single address is both - is registered in the depth map
            // and DELIBERATELY not in the colour one (`encode_depth_only_pass`). So an
            // address in both means something registered a colour target as depth, and
            // because this path is consulted first the pass would sample a distance where
            // it asked for an image. That shipped once, from the cross-frame carry-forward
            // keying by the `rtt` key instead of the depth address (see `rtt_depth_addrs`),
            // and it cost a title its whole world: the composite read depth and every pixel
            // came out on one channel. Prefer the colour answer and SAY SO - a renderer
            // that silently picks the wrong one of two buffers is the hardest defect here
            // to see, because the picture it produces is plausible.
            let depth_hit = (want == SamplerDim::Two)
                .then(|| {
                    usable.and_then(|gt| {
                        let a = gt.tex.data_addr;
                        // >>> A COMPRESSED TEXTURE IS NEVER A DEPTH SURFACE. The depth map
                        // is carried across frames for as long as its target lives, so an
                        // address a title frees and re-allocates keeps answering with the
                        // old distance. Base formats 0x80 and up are block families (PVRTC,
                        // UBC, ETC) that no GPU pass writes. MEASURED on a fighting title:
                        // stage 1's 512x512 shadow depth sat at the address stage 2 loaded a
                        // 1024x512 UBC1 graffiti wall into, and the wall sampled the stale
                        // depth - a flat red slab across the background.
                        if gt.tex.base_format >= 0x80 && depth_rendered.contains_key(&a) {
                            report_depth_alias_refused_compressed(a, gt.tex.base_format);
                            return None;
                        }
                        match (depth_rendered.get(&a), rendered.contains_key(&a)) {
                            // A depth-ONLY pass owns its address: `encode_depth_only_pass`
                            // keys the target BY the depth address and its colour
                            // attachment is a throwaway, so "in both maps" is the normal
                            // state there and the depth is what a sampler naming it wants.
                            (hit @ Some(_), true) if depth_only.contains(&a) => hit,
                            (Some(_), true) => {
                                report_depth_is_also_colour(a);
                                None
                            }
                            (hit, _) => hit,
                        }
                    })
                })
                .flatten();
            let aliased = depth_hit
                .is_none()
                .then(|| {
                    (want == SamplerDim::Two)
                        .then(|| {
                            usable.and_then(|gt| {
                                rendered_alias(rendered, gt.tex.data_addr, key, unit, gt.tex.guest_bytes_unwritten)
                            })
                        })
                        .flatten()
                })
                .flatten();
            match usable {
                Some(gt) if depth_hit.is_some() => {
                    report_depth_sample_bound(key, unit, gt.tex.data_addr);
                    // A view of a persistent target, so it is cacheable - keyed by the
                    // address it names and by the epoch that says the view is still alive.
                    if crate::gpu::rtt_bg_cache() {
                        mix(0x0d0d_0000_0000_0000 ^ gt.tex.data_addr as u64 ^ depth_rendered.get(&gt.tex.data_addr).map_or(0, view_identity));
                    } else {
                        per_frame = true;
                    }
                    plan.push((
                        Bind::Depth(gt.tex.data_addr),
                        sampler_mode(&gt.tex),
                    ));
                }
                // Sampling a buffer an earlier pass in THIS frame rendered: bind that
                // render. Only 2D targets - a cube face is never a GXM render target.
                Some(gt) if aliased.is_some() => {
                    let a = aliased.unwrap();
                    // >>> THE DECODE IS THE TEXTURE'S, NOT THE SURFACE'S. A gamma-correct
                    // target holds sRGB-ENCODED bytes, and the hardware sampler decodes them
                    // only when the TEXTURE that reads them sets a gamma mode
                    // (`GxpTexture::gamma`). `rendered` hands out the decoding sRGB view, which
                    // is right for such a texture and wrong for one without: that sampler
                    // reads the encoded bytes as they are. MEASURED on one title's gameplay
                    // frame: its lit buffer `0x92cc0000` and composite `0x92ec0000` are both
                    // gamma-correct; the three samplers of the first set gamma `0x8000000`,
                    // the two of the second - the display passthrough and the post pass -
                    // set none, and decoding there squared every display value (27 stored
                    // came out as 3), the near-black frame of a lit room.
                    // `VITASLOP_RTT_GAMMA_BY_TEXTURE=0` is the arm back.
                    let linear = !gt.tex.gamma && rendered_linear.contains_key(&a) && crate::gpu::rtt_gamma_by_texture();
                    // >>> AND THE OTHER WAY ROUND: a texture that SETS a gamma mode over a
                    // LINEAR target decodes the bytes it reads just the same - the sampler does
                    // not know who stored them. MEASURED on a shooter: its display composite
                    // reads the blurred-scene region of a linear post-process atlas through a
                    // `0x8000000` (BGR) texture; bound undecoded, every mid-tone of that layer
                    // came back as its sRGB encoding and the composite washed the frame white.
                    let srgb = !linear && gt.tex.gamma && rendered_srgb.contains_key(&a) && crate::gpu::rtt_gamma_by_texture();
                    let chosen = if linear {
                        rendered_linear.get(&a)
                    } else if srgb {
                        rendered_srgb.get(&a)
                    } else {
                        rendered.get(&a)
                    };
                    // >>> KEYED ON THE VIEW'S OWN IDENTITY, NOT `rtt_epoch`. The epoch
                    // moves whenever ANY target is created, and a title that creates two
                    // small targets a frame (text caches) moved it every frame - so every
                    // group naming the shadow map or a lightmap was rebuilt every frame:
                    // 229 GPU bind groups a frame against 1,831 reused, MEASURED. A view's
                    // identity changes only when THAT target is recreated, and a cached
                    // group keeps its view alive, so a stale entry is unreachable rather
                    // than dangling.
                    if crate::gpu::rtt_bg_cache() {
                        mix(0x0c0c_0000_0000_0000
                            ^ a as u64
                            ^ chosen.map_or(0, view_identity)
                            ^ if reads_snapshot.contains(&a) { 1 << 20 } else { 0 }
                            ^ if linear { 1 << 21 } else { 0 }
                            ^ if srgb { 1 << 22 } else { 0 });
                    } else {
                        per_frame = true;
                    }
                    plan.push((
                        if linear {
                            Bind::RenderedLinear(a)
                        } else if srgb {
                            Bind::RenderedSrgb(a)
                        } else {
                            Bind::Rendered(a)
                        },
                        sampler_mode(&gt.tex),
                    ));
                }
                Some(gt) => {
                    // A texture whose data pointer is null decodes to a 1x1 ZERO texel. That
                    // is the faithful substitute (see the runtime's zero-handle report), but
                    // a recompiled shader sampling it multiplies its whole output by zero,
                    // and the pass comes out BLACK with nothing in the log tying the two
                    // together. Name the pair and the unit, once each: this is a draw
                    // silently losing its content, which is the same class of event as a
                    // fallback and gets the same unconditional report.
                    if gt.tex.data_addr == 0 {
                        report_zero_texel_sample(key, unit);
                    }
                    let cache_key = (gt.tex.key, want);
                    mix(gt.tex.key);
                    // ONE lookup, not two. This runs per sampler unit per draw - about
                    // 3,400 times a frame on a title binding seven units across 492 draws -
                    // and it used to hash the key twice in a row, once to count the hit and
                    // once to branch on it.
                    let view_cached = view_cache.contains_key(&cache_key);
                    if view_cached {
                        enc(&ENC.tex_view_cached, 1);
                    }
                    if !view_cached {
                        // Bound the cache BY BYTES, evicting ENTRY BY ENTRY and never one
                        // this frame has already used. See `GxpLive::views_used` for the
                        // measurement that killed the old wholesale clear: it turned "the
                        // working set is 30% over budget" into a 0% hit rate and a 1 fps
                        // frame, because it fired part-way through every frame and threw
                        // away exactly what the next frame was about to ask for.
                        //
                        // The keys are content fingerprints, so an eviction costs a
                        // re-upload and never correctness - that part of the old argument
                        // was right, and it is what makes any eviction policy safe here.
                        *view_cache_bytes += uploaded_texture_bytes(bc, &gt.tex);
                        // The budget, but never below what ONE FRAME needs - see
                        // `GxpLive::views_frame_high`. Those bytes have to be resident
                        // during the frame regardless, so this floor costs nothing that
                        // was avoidable and is what lets the next frame hit.
                        // >>> THE RETENTION BOUND, NOT THE TEXTURE BUDGET. Those were the
                        // same number, and the budget is also what gates the BC -> ETC2
                        // re-encode, so bounding retention with it would have traded PICTURE
                        // for memory. See `tex_retain_budget_bytes` for the split and for
                        // the 343 MB / 14 MB measurement that asked for it.
                        let budget = tex_retain_budget_bytes().max(*views_frame_high);
                        if *view_cache_bytes >= budget {
                            let epoch = *views_epoch;
                            // >>> RECENTLY USED IS NOT A RETENTION CANDIDATE - see
                            // `tex_recent_frames`. Under the retention bound only a
                            // SUPERSEDED entry or one idle for `recent` frames goes; past the
                            // texture budget itself (`over_hard`), anything this frame has
                            // not used, oldest first, exactly as before.
                            let recent = crate::gpu::tex_recent_frames();
                            let over_hard = *view_cache_bytes >= tex_cache_budget_bytes().max(budget);
                            // Oldest first, so a cache under steady pressure sheds what has
                            // gone longest unused rather than whatever the hash order
                            // happens to yield.
                            let mut stale: Vec<((u64, SamplerDim), u64)> = views_used
                                .iter()
                                .filter(|(k, (used, ..))| {
                                    *used != epoch
                                        && (over_hard
                                            || recent == 0
                                            || view_dead.contains(*k)
                                            || used.saturating_add(recent) < epoch)
                                })
                                .map(|(k, (used, ..))| (*k, *used))
                                .collect();
                            // SUPERSEDED entries first, then oldest first. A superseded
                            // entry holds contents the guest has overwritten, so shedding
                            // it costs a re-upload only if that content comes back - and
                            // everything else here is still what the guest has.
                            stale.sort_by_key(|(k, used)| (!view_dead.contains(k), *used));
                            let mut evicted = 0usize;
                            let mut just_evicted: HashSet<(u64, SamplerDim)> = HashSet::default();
                            for (k, _) in stale {
                                if *view_cache_bytes < budget {
                                    break;
                                }
                                if let Some((tex, _view)) = view_cache.remove(&k) {
                                    note_view_cache_removed();
                                    // Release the GPU memory NOW. Dropping these two handles
                                    // only makes the texture collectable in the browser, and
                                    // a cache that evicts every frame would then be handing
                                    // the collector megabytes a frame to catch up on.
                                    tex.destroy();
                                    note_texture_destroyed();
                                    let bytes = views_used.remove(&k).map_or(0, |(_, b, ..)| b);
                                    *view_cache_bytes = view_cache_bytes.saturating_sub(bytes);
                                    // Remember it, so a later upload of the same key can be
                                    // attributed as THRASH rather than as a cold upload.
                                    // Bounded by dropping: forgetting an old eviction
                                    // under-counts thrash and nothing else.
                                    if views_evicted.len() < VIEW_EVICTED_KEYS_CAP {
                                        views_evicted.insert(k);
                                    }
                                    just_evicted.insert(k);
                                    view_dead.remove(&k);
                                    evicted += 1;
                                }
                            }
                            if evicted > 0 {
                                enc(&ENC.tex_view_evict_passes, 1);
                                enc(&ENC.tex_view_evicted, evicted as u64);
                                // >>> AND INTO THE RUN-LONG EVICTION SUMMARY, which named
                                // every other cache and not this one. The device panel read
                                // `EVICTIONS: none` beside a view cache holding 343 MB, and
                                // that was TRUE of every cache the summary knew about - the
                                // one cache the line was being read to ask about was the one
                                // it did not cover.
                                note_cache_evicted("texture views", evicted);
                                // >>> DROP EXACTLY THE SAMPLER BIND GROUPS THAT WENT STALE,
                                // and not the whole cache.
                                //
                                // A cached group names specific `TextureView`s, and an
                                // evicted key can be re-uploaded later as a DIFFERENT view,
                                // so any group naming it is dead - but only those. This used
                                // to bump a global `views_gen` folded into every group's
                                // key, which was right while eviction was WHOLESALE (every
                                // group really was dead) and became far too blunt when it
                                // went per entry. The policy changed and the invalidation
                                // did not follow it, exactly as the counter NAMES did not.
                                //
                                // MEASURED on the user's device at 1.07 eviction passes a
                                // frame: 446.5 sampler bind groups REBUILT per frame against
                                // 792.6 reused - one cold entry a frame was throwing away a
                                // third of the cache that exists to stop per-draw GPU object
                                // creation ([[vitaslop-per-pass-render-arenas]]).
                                sampler_bgs.retain(|_, (_, named, _)| {
                                    !named.iter().any(|k| just_evicted.contains(k))
                                });
                            } else if over_hard || recent == 0 {
                                // Nothing was evictable: every entry is in use by the
                                // frame being encoded. Exceeding the budget is the only
                                // way to finish it, and it is REPORTED rather than
                                // silently done - a working set this size is the thing to
                                // fix, and on a phone it is also how a worker gets killed
                                // with no error. (Held over the RETENTION bound by recent
                                // use alone is not this: that is the recency rule working.)
                                report_texture_budget_exceeded(
                                    *view_cache_bytes,
                                    views_used,
                                    *views_epoch,
                                );
                            }
                        }
                        if views_evicted.remove(&cache_key) {
                            enc(&ENC.tex_reuploaded_after_evict, 1);
                        }
                        // >>> "SUPERSEDED IN PLACE" IS MADE LITERAL HERE.
                        //
                        // The cache key folds the pixel buffer's identity, so a guest
                        // texture the title REWRITES arrives as a new key every frame and
                        // used to get a brand-new GPU texture, with the previous one marked
                        // dead and evicted later. MEASURED on the user's device once the key
                        // became exact: **26.7 textures uploaded and 26.9 DESTROYED per
                        // frame, over 7.37 eviction passes** - a conveyor belt, not a cache,
                        // and creating and destroying that many textures a frame is most of
                        // what `prepare` costs there.
                        //
                        // The slot is the same guest texture bound the same way, so its
                        // previous upload is a texture of exactly the right shape and
                        // format. Writing the new contents into it is the same bytes going
                        // to the same place, minus a create, a destroy and an eviction.
                        //
                        // NOT taken when THIS frame has already bound that view: a draw
                        // earlier in the frame is reading those texels, and overwriting them
                        // would change what it drew. That is the one case where the old
                        // texture has to survive alongside the new one.
                        //
                        // OFFERED ONLY TO A TEXTURE THAT CAN TAKE IT. `t.raw` is the one
                        // path that writes into an existing texture; taking the old one out
                        // of the cache for any other format would hand it to a path that
                        // does not use it, and every handle taken out has to be released.
                        let reuse = gt
                            .tex
                            .raw
                            .as_ref()
                            .and_then(|_| view_slot_key(&gt.tex, want))
                            .and_then(|slot| view_slots.get(&slot).copied())
                            .filter(|stale| *stale != gt.tex.key)
                            .map(|stale| (stale, want))
                            .filter(|k| {
                                views_used
                                    .get(k)
                                    .is_none_or(|(used, ..)| *used != *views_epoch)
                            })
                            .and_then(|k| {
                                let (tex, _) = view_cache.remove(&k)?;
                                note_view_cache_removed();
                                // Its bytes leave the cache with it, and any bind group
                                // naming its view is now dead - the same invalidation an
                                // eviction does, for the same reason.
                                let bytes =
                                    views_used.remove(&k).map_or(0, |(_, b, ..)| b);
                                *view_cache_bytes = view_cache_bytes.saturating_sub(bytes);
                                view_dead.remove(&k);
                                sampler_bgs.retain(|_, (_, named, _)| !named.contains(&k));
                                enc(&ENC.tex_view_superseded, 1);
                                // The slot now names the CURRENT contents, so the
                                // bookkeeping below finds nothing to supersede. Without
                                // this it would mark a key that is no longer in the cache
                                // as dead - an entry only an eviction removes, and
                                // evictions never see it - and count the supersede twice.
                                if let Some(slot) = view_slot_key(&gt.tex, want) {
                                    view_slots.insert(slot, gt.tex.key);
                                }
                                Some(tex)
                            });
                        let tex = upload_gxp_texture(
                            device,
                            queue,
                            &gt.tex,
                            bc,
                            texenc,
                            reuse,
                            // A cube VIEW needs six layers whatever the guest bound - see
                            // the `usable` test above.
                            if want == SamplerDim::Cube { 6 } else { 1 },
                            // Only a plain 2D view is swapped in later; every other kind
                            // of binding encodes inline, as before.
                            (want == SamplerDim::Two).then_some(gt.tex.key),
                        );
                        let view = tex.create_view(&wgpu::TextureViewDescriptor {
                            dimension: Some(want.view_dimension()),
                            ..Default::default()
                        });
                        // >>> THE PREVIOUS UPLOAD OF THIS SAME GUEST TEXTURE IS THE
                        // FIRST THING TO SHED, and only that.
                        //
                        // The cache key folds the source bytes, so a texture the guest
                        // rewrites - a video picture, thirty times a second - arrives as a
                        // NEW key every time and the old one stays resident until the byte
                        // budget notices. On a phone that is the budget filling with dead
                        // movie frames and then evicting textures the title still needs.
                        //
                        // RELEASING it here was tried and MEASURED, and it is wrong:
                        // uploads went from 0.51 MB to 2.68 MB per frame on the same
                        // screen. Content COMES BACK - a title cycles a handful of images
                        // through one guest texture - and "those bytes are gone from guest
                        // memory" holds only until the guest writes them again. So the
                        // displaced entry is marked rather than destroyed: eviction takes
                        // marked entries before anything else, so a picture that never
                        // returns is shed under pressure while an image that cycles is
                        // still there when it comes round.
                        if let Some(slot) = view_slot_key(&gt.tex, want)
                            && let Some(stale) = view_slots.insert(slot, gt.tex.key)
                                && stale != gt.tex.key {
                                    view_dead.insert((stale, want));
                                    enc(&ENC.tex_view_superseded, 1);
                                }
                        // It is current again, whatever it was before.
                        view_dead.remove(&cache_key);
                        // >>> COUNTED, AND THE RETURNED ENTRY IS THE THING TO WATCH. This
                        // arm runs only on a MISS, so `insert` must be returning `None`; if
                        // it ever returns an entry, that entry's texture is DROPPED here and
                        // the browser leaks it. Counting the displacement separately is what
                        // makes "the ledger does not close" say WHICH of the two it is.
                        enc(&ENC.tex_view_inserted, 1);
                        LEDGER_INSERTED.fetch_add(1, std::sync::atomic::Ordering::Relaxed);
                        // >>> A DISPLACED ENTRY IS DESTROYED, NEVER DROPPED. This arm runs
                        // only on a MISS, so `insert` should always return `None` - but if
                        // it ever does not, dropping the old pair leaks a `GPUTexture` in
                        // the browser, which is the exact failure `report_texture_handle_drift`
                        // exists to catch. Costs a branch on a path that just uploaded a
                        // texture.
                        if let Some((old_tex, _)) = view_cache.insert(cache_key, (tex, view)) {
                            note_view_cache_removed();
                            note_texture_destroyed();
                            old_tex.destroy();
                        }
                    }
                    // Stamp it as used by THIS frame, hit or miss - that stamp is what
                    // makes it ineligible for eviction while the frame is still being
                    // built. The size rides along so an eviction can subtract it.
                    //
                    // The passthrough flag rides along so the working-set breakdown can say
                    // which formats are still EXPANDING, rather than repeating a static
                    // claim about what BC "would" cost. That claim was aspirational for as
                    // long as the note existed: nothing uploaded compressed, and the report
                    // said "BC, uploaded compressed" anyway.
                    let resident = match compressed_upload(bc, &gt.tex) {
                        Some(c) if c.transcoded => Residency::Transcoded,
                        Some(_) => Residency::Passthrough,
                        None => Residency::Decoded,
                    };
                    // >>> THE PRICE RECORDED WHEN THE ENTRY WAS ADDED, NOT A FRESH ONE.
                    //
                    // `view_cache_bytes` was incremented by the price computed on the MISS
                    // and eviction subtracts whatever this entry records, so re-pricing here
                    // drifts the running total away from the entries it describes - the
                    // defect the decode cache's own stamp already names in as many words.
                    // MEASURED before the fix, on a fighting title at frame 3,000: the
                    // retained figure read 309 MB while the entries summed to 276.
                    //
                    // One map lookup instead of a probe plus an insert, on a path taken once
                    // per bound texture per draw - about 3,400 times a frame.
                    let epoch = *views_epoch;
                    let (bytes, first_this_frame) = match views_used.entry(cache_key) {
                        std::collections::hash_map::Entry::Occupied(mut e) => {
                            let (used, bytes, ..) = *e.get();
                            e.insert((epoch, bytes, gt.tex.base_format, resident));
                            (bytes, used != epoch)
                        }
                        std::collections::hash_map::Entry::Vacant(e) => {
                            let bytes = uploaded_texture_bytes(bc, &gt.tex);
                            e.insert((epoch, bytes, gt.tex.base_format, resident));
                            (bytes, true)
                        }
                    };
                    // Count it toward THIS frame's working set the first time the frame
                    // touches it - a texture sampled by two hundred draws is two hundred
                    // lookups and one set of bytes, and counting per lookup would inflate
                    // the learned floor by that factor.
                    if first_this_frame {
                        *views_frame_bytes += bytes;
                    }
                    // An UNFILTERABLE slot takes a nearest sampler only (see `TwoF32`).
                    let mode = if want == SamplerDim::TwoF32 {
                        (false, gt.tex.addr_mode_u | RAW_SAMPLER_FLAG, gt.tex.addr_mode_v)
                    } else {
                        sampler_mode(&gt.tex)
                    };
                    plan.push((Bind::Cached(cache_key), mode));
                }
                // A volume sampler (not yet mapped), or a unit whose real texture we could
                // not capture/decode: strict mode falls back; force mode binds a neutral
                // fallback so geometry still renders (a diagnostic, never the default).
                None => {
                    // A fresh fallback view per draw, so this group is per-frame too.
                    per_frame = true;
                    if !force {
                        // >>> WITH THE SHAPE OF WHAT WAS BOUND, not just its face count.
                        // A single face is already the answer to one half of "wants Cube,
                        // got one": `faces` is set from the guest's own texture-type
                        // selector (`render::cube_type`, the two CUBE encodings 2 and 7), so
                        // one face means the guest's control word does NOT say cube - the
                        // capture did not lose five faces. What is left is a container that
                        // declares a cube sampler over a texture the guest bound flat, and
                        // the dimensions and format are what say which texture that is.
                        report_fallback(
                            key,
                            &format!(
                                "{stage_name} sampler unit {unit} wants {want:?} but the {stage_name} stage's bound units are \
                                 {:?} (unit, faces, WxH, base format)",
                                textures
                                    .iter()
                                    .map(|t| (
                                        t.unit,
                                        t.tex.faces,
                                        (t.tex.width, t.tex.height),
                                        t.tex.base_format
                                    ))
                                    .collect::<Vec<_>>()
                            ),
                        );
                        return None;
                    }
                    let mode = if want == SamplerDim::TwoF32 { (false, RAW_SAMPLER_FLAG, 0) } else { (false, 0, 0) };
                    plan.push((Bind::Fallback(want), mode));
                }
            }
        }
        }
        // The sampler state belongs in the key too: two draws can bind the same image
        // through different filter/wrap modes, and that is a different bind group.
        for &(_, (lin, u, v)) in &plan {
            mix((lin as u64) << 62 | (u as u64) << 32 | v as u64);
        }
        // >>> WHAT THIS DRAW ACTUALLY SAMPLES (`VITASLOP_GXP_BIND_TRACE=1`).
        //
        // Every other instrument here describes what the GUEST bound: the per-draw dump
        // prints the texture descriptor, `rendered_alias` reports only when it substitutes.
        // None of them says what the renderer ended up NAMING, and those differ by design -
        // a render target this frame drew, a depth buffer, a cube, a cached decode of guest
        // bytes, or a fallback are five different answers to one descriptor. MEASURED on
        // a baseball title: one draw's descriptor is an 8x8 texture and the pixels it paints are
        // plainly a 512x512 image, and no existing line could say which of those the
        // sampler was given. Reported at `warn` so it survives the default filter, and
        // once per (pair, plan shape) so a 500-draw pass cannot flood the log.
        if crate::gpu::bind_trace() {
            let shape: Vec<String> = plan
                .iter()
                .map(|(b, _)| match b {
                    Bind::Depth(a) => format!("depth@{a:#x}"),
                    Bind::Rendered(a) => format!("rendered@{a:#x}"),
                    Bind::RenderedLinear(a) => format!("rendered-linear@{a:#x}"),
                    Bind::RenderedSrgb(a) => format!("rendered-srgb@{a:#x}"),
                    Bind::RenderedCube(a) => format!("cube@{a:#x}"),
                    Bind::RenderedRaw(a) => format!("raw@{a:?}"),
                    Bind::Cached((k, d)) => format!("guest-decode key={k:016x} dim={d:?}"),
                    Bind::Fallback(d) => format!("FALLBACK dim={d:?}"),
                })
                .collect();
            let line = shape.join(" | ");
            use std::sync::{Mutex, OnceLock};
            static SEEN: OnceLock<Mutex<HashSet<(u64, String)>>> = OnceLock::new();
            let seen = SEEN.get_or_init(|| Mutex::new(HashSet::default()));
            if seen.lock().unwrap_or_else(|e| e.into_inner()).insert((key, line.clone())) {
                report_warn!("gxp bind: pair {key:016x} samples [{line}]");
            }
        }
        // The finished group is a pure function of what it names, so ask for it before
        // building it. Only when nothing in it belongs to THIS frame's render targets.
        if !per_frame
            && let Some((bg, _, used)) = sampler_bgs.get_mut(&(key, sel)) {
                *used = *views_epoch;
                let bg = &*bg;
                note_sampler_bg(true);
                *last_sampler_bg = Some((pre, bg.clone()));
                remember_sampler_pre(sampler_pre, pre, sel);
                return Some(bg.clone());
            }
        // Create every sampler this bind group needs FIRST, so the map is not borrowed
        // mutably while the entries hold shared references into it.
        for &(_, st) in &plan {
            samplers_by_mode.entry(st).or_insert_with(|| make_gxp_sampler(device, st.0, st.1, st.2));
        }
        // ONLY NOW are the views taken: the cache above has already said it does not have
        // this group. See the `plan` declaration for what that is worth.
        let mut views: Vec<wgpu::TextureView> = Vec::with_capacity(total);
        // The view-cache keys this group NAMES, kept so an eviction can invalidate exactly
        // the groups that went stale rather than all of them. A per-frame view contributes
        // nothing here because such a group is never cached in the first place.
        let mut named_views: Vec<(u64, SamplerDim)> = Vec::with_capacity(total);
        for &(bind, _) in &plan {
            // Cloning a `TextureView` is a refcount bump, so cached views are shared, not
            // copied. A plan entry whose view has gone (an eviction between the loop above
            // and here cannot happen, but a render target that named an address nothing
            // rendered can) declines the whole group rather than binding the wrong texture.
            let view = match bind {
                Bind::Depth(addr) => depth_rendered.get(&addr)?.clone(),
                Bind::Rendered(addr) => rendered.get(&addr)?.clone(),
                Bind::RenderedLinear(addr) => rendered_linear.get(&addr)?.clone(),
                Bind::RenderedSrgb(addr) => rendered_srgb.get(&addr)?.clone(),
                Bind::RenderedCube(addr) => rendered_cubes.get(&addr)?.clone(),
                Bind::RenderedRaw(addr) => rendered_raw.get(&addr)?.clone(),
                Bind::Cached(cache_key) => {
                    named_views.push(cache_key);
                    view_cache.get(&cache_key)?.1.clone()
                }
                // A fresh fallback view per draw, which is why such a group is per-frame.
                Bind::Fallback(want) => make_fallback_view(device, queue, want.view_dimension()),
            };
            views.push(view);
        }
        let mut entries: Vec<wgpu::BindGroupEntry> = Vec::with_capacity(total * 2);
        for (i, view) in views.iter().enumerate() {
            let samp = &samplers_by_mode[&plan[i].1];
            entries.push(wgpu::BindGroupEntry { binding: i as u32 * 2, resource: wgpu::BindingResource::TextureView(view) });
            entries.push(wgpu::BindGroupEntry { binding: i as u32 * 2 + 1, resource: wgpu::BindingResource::Sampler(samp) });
        }
        note_sampler_bg(false);
        let bg = device.create_bind_group(&wgpu::BindGroupDescriptor { label: Some("gxp-samplers"), layout, entries: &entries });
        if !per_frame {
            // Bounded the way every other content-addressed cache here is: the key is a
            // fingerprint of what the group names, so dropping it costs a rebuild and
            // never correctness.
            if sampler_bgs.len() >= SAMPLER_BG_CACHE_CAP {
                let n = evict_oldest(
                    sampler_bgs,
                    evict_target(SAMPLER_BG_CACHE_CAP),
                    *views_epoch,
                    |(_, _, used)| *used,
                );
                note_cache_evicted("sampler bind groups", n);
            }
            sampler_bgs.insert((key, sel), (bg.clone(), named_views, *views_epoch));
            *last_sampler_bg = Some((pre, bg.clone()));
            remember_sampler_pre(sampler_pre, pre, sel);
        }
        Some(bg)
    }
}
