//! Offscreen render targets: the surfaces a pass draws into, their multisampled and
//! guest-depth companions, the per-pass bound state, and the reports about where a scene went.

use super::*;

/// The offscreen supersample render target: an `ss_scale * surf` colour + depth buffer the
/// scene is rendered into, plus the bind group the resolve pass samples the colour through.
/// Kept across frames and rebuilt only when the scale or the target dimensions change.
pub(super) struct SsTarget {
    pub(super) scale: u32,
    pub(super) width: u32,
    pub(super) height: u32,
    pub(super) _color: wgpu::Texture,
    pub(super) color_view: wgpu::TextureView,
    pub(super) depth_view: wgpu::TextureView,
    pub(super) resolve_bind: wgpu::BindGroup,
}

/// A scene whose colour surface could not be resolved has nowhere to be drawn, and
/// whatever it drew is missing from the frame. Reported the first time it happens (a
/// per-frame pass would otherwise flood the log), unconditionally - a pass silently
/// vanishing is indistinguishable from the guest never submitting it, and that
/// ambiguity has cost this project whole sessions.
///
/// `depth_addr` and `depth_is_sampled` are the two facts that decide whether this
/// matters. A colour-less scene is usually a DEPTH-ONLY pass - a shadow map, a depth
/// prepass - and it is only a defect when something later in the frame BINDS that depth
/// as a texture, because then a real read is being served stale guest bytes. Without
/// them the report says a pass vanished and leaves the reader with no way to find out
/// whether anything wanted it, which is how this one has sat open with "no confirmed
/// visible symptom" for several sessions.
pub(super) fn report_unplaced_scene(
    draws: usize,
    depth_addr: u32,
    depth_is_sampled: bool,
    had_extent: bool,
    pairs: &str,
) {
    // >>> DEDUPED ON THE PAIR SET, NOT ONCE FOR THE WHOLE RUN. A once-only report fires on
    // the FIRST such scene, which on a title that boots through a loading screen is not the
    // one anybody is asking about - it named the boot screen's two pairs and stayed silent
    // for every frame after, including the frames whose picture was missing.
    use std::collections::HashSet;
    use std::sync::Mutex;
    static REPORTED: Mutex<Option<HashSet<String>>> = Mutex::new(None);
    let mut g = REPORTED.lock().unwrap_or_else(|e| e.into_inner());
    let seen = g.get_or_insert_with(HashSet::new);
    if seen.len() < 16 && seen.insert(pairs.to_string()) {
        if depth_is_sampled {
            // Reaching here now means the depth-only path could not be taken, and there is
            // exactly one reason left: no extent. A `SceGxmDepthStencilSurface` carries
            // none, so the extent comes from the draws' viewport, and a pass whose draws
            // disable the viewport (or set a degenerate one) leaves nothing to size a
            // target from. Naming that is the difference between a defect someone can act
            // on and a pass that vanished.
            report_warn!(
                "gxm: a scene with {draws} draws has no resolvable colour surface, so it \
                 renders NOWHERE - and a later pass in the same frame SAMPLES its depth at \
                 {depth_addr:#x}, which therefore reads stale guest bytes rather than this \
                 pass's depth. It could not be given a depth-only target because its extent \
                 is unknown (had_extent={had_extent}): a DepthSurface carries no extent, and \
                 this scene's draws do not supply an enabled, non-degenerate viewport to \
                 take one from."
            );
            return;
        }
        report_warn!(
            "gxm: a scene with {draws} draws has no resolvable colour surface - IT RENDERS NOWHERE. Nothing in this frame samples its depth ({depth_addr:#x}), so nothing needs its DEPTH - but its colour draws are LOST. Pairs: [{pairs}]"
        );
    }
}

/// Say - once per depth address - that a colour-less pass was placed into a depth-only
/// target, and at what size.
///
/// The size is the point. It is DERIVED from the draws' viewport rather than read out of a
/// guest struct (a `SceGxmDepthStencilSurface` has no extent), and every later pass that
/// samples this depth inherits that resolution in its own texel size and screen-space
/// bias. A derived resolution nobody can see is exactly the kind of thing that makes a
/// sampling bias look wrong somewhere else entirely.
pub(super) fn report_depth_only_pass(
    depth_addr: u32,
    width: u32,
    height: u32,
    draws: usize,
    ambiguous: bool,
    // Every distinct viewport in the pass with its draw count, most-used first. Empty
    // unless the draws disagreed - see the `ambiguous` arm below for why it is worth the
    // characters.
    viewports: &[(([f32; 6], (i32, i32)), usize)],
) {
    use std::sync::{Mutex, OnceLock};
    static SEEN: OnceLock<Mutex<HashSet<u32>>> = OnceLock::new();
    let seen = SEEN.get_or_init(|| Mutex::new(HashSet::default()));
    if !seen.lock().unwrap_or_else(|e| e.into_inner()).insert(depth_addr) {
        return;
    }
    // >>> THE LEVEL IS DECIDED BY WHETHER THE SIZE IS KNOWN, NOT BY WHETHER IT IS DERIVED.
    //
    // This used to warn unconditionally because the extent came from `draws.first()` and so
    // was, honestly, a guess. It is not a guess any more: the builder now reads the
    // viewport of EVERY draw in the pass, and `ambiguous` says whether they disagreed. When
    // they all name one rectangle that rectangle is the guest's own statement of the region
    // and there is nothing to report - the accommodation simply worked. When they disagree,
    // the target is the largest of several, every later pass sampling this depth inherits a
    // resolution no single draw asked for, and that is a real defect that says so.
    //
    // Lowering the agreed case is NOT silencing: the check that made it a guess was
    // replaced, so the thing being reported stopped existing. Silencing would have been
    // lowering it while `draws.first()` still decided the answer.
    if ambiguous {
        // >>> NAME THE RECTANGLES, because "the largest of several" is where the reader
        // stops. A pass whose draws disagree is either a shadow ATLAS (several regions,
        // each correct, and the merged extent is the atlas) or two logical passes sharing
        // one depth address (in which case one of them is writing into the other's map and
        // whatever samples it reads a depth that was never meant for it). Those two have
        // opposite fixes and the LIST tells them apart at a glance; the count alone does
        // not. The z map is in the same array, so a pass carrying two different depth
        // ranges - which no single shadow map can mean - shows up here too.
        let list: Vec<String> = viewports
            .iter()
            .map(|((v, bias), n)| {
                format!(
                    "{n} draw(s) at [xOff {} xScale {} yOff {} yScale {} zOff {} zScale {}] \
                     = {}x{}{}, depth bias (factor {}, units {})",
                    v[0],
                    v[1],
                    v[2],
                    v[3],
                    v[4],
                    v[5],
                    (2.0 * v[1].abs()) as u32,
                    (2.0 * v[3].abs()) as u32,
                    if v[3] > 0.0 { ", yScale POSITIVE (flipped)" } else { "" },
                    bias.0,
                    bias.1
                )
            })
            .collect();
        report_warn!(
            "gxm depth-only pass {depth_addr:#x}: {draws} draws with no colour surface, and \
             they DISAGREE about the viewport - the depth-only target is {width}x{height}, \
             the largest of several, so it is a size no single draw asked for and every \
             later pass sampling this depth inherits it. A DepthSurface carries no extent, \
             so there is no other source to check this against. The rectangles are: {}",
            list.join("; ")
        );
    } else {
        report!(
            "gxm depth-only pass {depth_addr:#x}: {draws} draws with no colour surface, \
             placed into a depth-only target at {width}x{height} - the extent every \
             viewport-enabled draw in the pass agrees on (a DepthSurface carries no extent)."
        );
    }
}

/// Say so - once, unconditionally - when the frame's BIGGEST pass is neither the display
/// pass nor sampled by anything, because then the thing that took the most work to draw is
/// the one thing the picture cannot contain.
///
/// The display is defined as the LAST scene's colour address, and everything else is an
/// offscreen target reachable only by a later pass that SAMPLES it. That is right for a
/// title that renders its world offscreen and composites it. It is wrong the moment a
/// title draws its world straight into one of several DISPLAY buffers and ends the frame
/// with a small pass into a different one: the world then goes to an offscreen target
/// nothing reads, and the frame is the last little pass over an empty screen.
///
/// That is not hypothetical. It is what a black world with a correct HUD looks like, and
/// the shape is invisible from the frame - a pass that renders nowhere and a pass the
/// guest never submitted produce the same picture. So this reports the addresses and lets
/// the reader see two display-sized buffers in one frame.
pub(super) fn report_world_not_on_display(
    scenes: &[RenderScene],
    display: Option<u32>,
    sampled: &HashSet<u32>,
    // The faces of every cube map assembled from renders, which ARE read - through the cube
    // texture they are copied into, under an address that is not their own. Without this
    // the biggest pass in a frame that renders a cube map is reported as "renders into a
    // target nothing reads" forever, which is how this read before cube maps were bound
    // from their renders at all: face 5 of a golf title's environment cube carries 356 draws,
    // the most of any pass in the frame, and only face 0 shares the cube's address.
    cube_faces: &HashSet<u32>,
    // The answer this function reached on the PREVIOUS frame, and it is load-bearing.
    // `cube_faces` comes from `rtt_cubes`, which is filled BETWEEN passes - i.e. after
    // this runs - so on the first frame of a scene it is empty and every cube face looks
    // orphaned. This report fires once and never again, so firing it there pins a false
    // alarm for the whole run: the golf title's course reported its 423-draw environment-cube
    // face as "MISSING from the picture" in the same dump whose next line assembled that
    // cube. Requiring the same address twice costs one frame of latency and closes the
    // window, because a cube known on frame N is known on every frame after it.
    previous: &mut Option<u32>,
) {
    use std::sync::atomic::{AtomicBool, Ordering};
    static REPORTED: AtomicBool = AtomicBool::new(false);
    // NOT named `display`: `tracing`'s macros bring their own `display()` helper into
    // scope, so a local of that name is shadowed inside the format string and the error
    // it produces names a trait bound, not a shadowing.
    let Some(display_addr) = display else {
        *previous = None;
        return;
    };
    let Some(biggest) = scenes.iter().max_by_key(|s| s.draws.len()) else {
        *previous = None;
        return;
    };
    let Some(t) = biggest.target else {
        *previous = None;
        return;
    };
    if t.data_addr == display_addr
        || sampled.contains(&t.data_addr)
        || cube_faces.contains(&t.data_addr)
        || biggest.draws.len() < 2
    {
        *previous = None;
        return;
    }
    if previous.replace(t.data_addr) != Some(t.data_addr) {
        return;
    }
    if REPORTED.swap(true, Ordering::Relaxed) {
        return;
    }
    let targets: Vec<String> = scenes
        .iter()
        .map(|s| match s.target {
            Some(t) => format!("{:#x}:{}x{}({} draws)", t.data_addr, t.width, t.height, s.draws.len()),
            None => format!("none({} draws)", s.draws.len()),
        })
        .collect();
    report_warn!(
        "gxm: the frame's BIGGEST pass ({} draws into {:#x}, {}x{}) is neither the display \
         ({:#x}) nor sampled by anything, so it renders into a target nothing reads and is \
         MISSING from the picture. The display is taken to be the LAST scene's colour \
         address; if two of these are display-sized the frame is being assembled from \
         passes that do not belong to one image. Passes: [{}]",
        biggest.draws.len(), t.data_addr, t.width, t.height, display_addr, targets.join(", ")
    );
}

/// One offscreen render target: the colour buffer a pass draws into and a matching
/// depth buffer. Held by the guest address of the colour surface, so the pass that
/// samples it later can be matched to it by the texture's data pointer.
/// A 64-bit colour surface rendered as RAW words - see [`GxmRenderer::rtt_raw`]. One
/// `Rg32Uint` colour attachment (also sampled, as `texture_2d<u32>`) and its own depth.
/// Whether a scene's depth-stencil surface asks the tiler to LOAD it from memory at the start
/// of the scene / STORE it at the end - see `GxmRenderer::depth_saved`.
pub(super) fn zls_load(scene: &RenderScene) -> bool {
    scene.depth_addr != 0 && scene.zls_control & 0x2 != 0 && crate::gpu::zls_on()
}

pub(super) fn zls_store(scene: &RenderScene) -> bool {
    scene.depth_addr != 0 && scene.zls_control & 0x4 != 0 && crate::gpu::zls_on()
}

pub(super) struct RawSurface {
    pub(super) width: u32,
    pub(super) height: u32,
    pub(super) color: wgpu::Texture,
    pub(super) color_view: wgpu::TextureView,
    /// The depth attachment behind `depth_view` - see `GxmRenderer::depth_saved`.
    pub(super) depth: wgpu::Texture,
    pub(super) depth_view: wgpu::TextureView,
    /// The same words read as F16F16F16F16 - what a sampler declaring that format over this
    /// memory reads, FILTERED. Re-filled after every raw pass into the surface
    /// (`convert_raw_to_float`); the raw words ARE the four halves, so this is a
    /// reinterpretation, not a conversion.
    pub(super) float_view: wgpu::TextureView,
    /// The guest-depth reader of `depth`, for a frame that SAMPLES this surface's depth -
    /// the same converted R32Float view an ordinary target carries (`RttSurface::gxm_depth`).
    /// Built on the first frame that samples it; see the raw arm of the scene loop.
    pub(super) gxm_depth: Option<GxmDepthTarget>,
}

pub(super) struct RttSurface {
    pub(super) width: u32,
    pub(super) height: u32,
    pub(super) color: wgpu::Texture,
    pub(super) color_view: wgpu::TextureView,
    /// An sRGB view of the SAME `color` texture, for a surface the guest put in
    /// gamma-correct mode. Rendering through it makes the ROP sRGB-encode every store
    /// after blending, and sampling through it decodes on read - which is what the
    /// hardware does at both ends. `None` when the colour format has no sRGB twin.
    pub(super) color_view_srgb: Option<wgpu::TextureView>,
    /// Whether the GUEST put this surface in gamma-correct mode, i.e. whether the bytes in
    /// `color` are sRGB-ENCODED and a sampler must decode them on the way back in.
    ///
    /// # Not the same question as "does an sRGB view exist"
    /// `color_view_srgb` is built whenever the colour format HAS an sRGB twin, whatever the
    /// guest asked for, so it cannot stand in for this. Keeping the guest's answer here is
    /// what lets a pass sampling this target on a LATER frame make the same choice the pass
    /// that rendered it made - see `encode_chain`'s `sample_views`, and the note there for
    /// what it cost when the two disagreed.
    pub(super) gamma: bool,
    /// The depth ATTACHMENT texture. Held rather than left to its view so it can be
    /// `destroy()`ed with the rest of the target - see `RttSurface::destroy`.
    pub(super) depth: wgpu::Texture,
    pub(super) depth_view: wgpu::TextureView,
    /// A copy of `color` as it stood before the pass now drawing into it, made only
    /// when that pass also SAMPLES the buffer - see `GxmRenderer::snapshot_rtt`.
    ///
    /// Two views over the one copy, for the same reason `color` carries two: the copy is
    /// made with `copy_texture_to_texture`, which moves BYTES, so a snapshot of a
    /// gamma-correct target holds sRGB-ENCODED bytes and a sampler must decode them on the
    /// way back in. Handing out the linear view there reads 0.5 back as 0.73, and this is
    /// the FEEDBACK path by construction - the snapshot exists only for a pass that samples
    /// the buffer it is drawing into - so the error re-applies every iteration and walks the
    /// image to white. That is the same defect `sample_views` documents on the cross-frame
    /// path; it lived here too, on the within-frame one.
    pub(super) shadow: Option<(wgpu::Texture, wgpu::TextureView, Option<wgpu::TextureView>)>,
    /// This target's depth, re-encoded the way the GUEST's depth buffer holds it, for a
    /// later pass that samples it. Present only when some pass in the frame actually
    /// names this scene's depth address - see `GxmRenderer::encode_chain`.
    pub(super) gxm_depth: Option<GxmDepthTarget>,
    /// Samples per pixel the CURRENT pass rasterises at. A target whose scenes alternate
    /// (a 4x 3D scene, then a 1x HUD into the same address) keeps `msaa` across the 1x
    /// scene rather than reallocating it every frame; this says which attachments a pass
    /// uses. See `ensure_rtt`.
    pub(super) samples_now: u32,
    /// A single-sample pass wrote `color` since `msaa` was last rendered, so the samples no
    /// longer hold what the target's memory does - see [`GxmRenderer::seed_msaa_from_color`].
    pub(super) msaa_stale: bool,
    /// The MULTISAMPLED attachments this target is rasterised into when the guest created
    /// its render target with `SCE_GXM_MULTISAMPLE_2X`/`_4X`, resolved into `color` by the
    /// pass itself. `None` when the guest asked for one sample, which is the common case.
    ///
    /// Everything outside the pass keeps reading `color` at the STORED size: that is what
    /// later passes sample, what guest coordinates are expressed in, and what the guest's
    /// own structs describe. The multisampled buffer exists only inside the pass.
    pub(super) msaa: Option<MsaaAttachments>,
}

impl RttSurface {
    /// Release every GPU allocation this target owns.
    ///
    /// # Drop is not enough, and only one engine can see that
    /// On a native backend dropping the handle returns the allocation. In the browser a
    /// `wgpu::Texture` IS a JavaScript `GPUTexture`, so dropping it only makes it GARBAGE
    /// and the memory comes back whenever the JS collector decides to run. That is the
    /// same defect the per-pass buffer arenas and the evicted texture views were fixed for;
    /// render targets were the third owner of GPU memory and were still relying on Drop.
    ///
    /// A render target is REBUILT whenever the guest resizes it, gains a depth reader or
    /// changes its sample count, so this is not a shutdown-only path - it runs at screen
    /// transitions, which on a memory-tight device is exactly when the headroom is needed.
    pub(super) fn destroy(&self) {
        self.color.destroy();
        self.depth.destroy();
        if let Some((tex, _, _)) = &self.shadow {
            tex.destroy();
        }
        if let Some(d) = &self.gxm_depth {
            d.tex.destroy();
        }
        if let Some(m) = &self.msaa {
            m.color.destroy();
            m.depth.destroy();
        }
    }

    /// Every texture this target owns, for the RETIREMENT queue. The counterpart of
    /// [`Self::destroy`] for a target replaced while a frame is being encoded - see
    /// `GxmRenderer::retired_textures` for the black frame that made the difference
    /// load-bearing.
    pub(super) fn into_textures(self) -> Vec<wgpu::Texture> {
        let mut v = vec![self.color, self.depth];
        if let Some((tex, _, _)) = self.shadow {
            v.push(tex);
        }
        if let Some(d) = self.gxm_depth {
            v.push(d.tex);
        }
        if let Some(m) = self.msaa {
            v.push(m.color);
            v.push(m.depth);
        }
        v
    }

    /// Roughly what this target costs in GPU memory, for the residency report and for the
    /// reclamation that reads it.
    ///
    /// Approximate on purpose: it counts four bytes a texel for each attachment it owns,
    /// which is exact for the formats actually in use here and never off by more than the
    /// depth format's own width. What it is FOR is a number that tracks run length - a
    /// hundred stale targets read as hundreds of megabytes either way, and that is the
    /// distinction nothing in this renderer could make before.
    pub(super) fn bytes(&self) -> u64 {
        let one = (self.width as u64) * (self.height as u64) * 4;
        let mut n = one * 2; // colour + depth
        if self.shadow.is_some() {
            n += one;
        }
        if self.gxm_depth.is_some() {
            n += one;
        }
        if let Some(m) = &self.msaa {
            n += one * 2 * (m.samples.max(1) as u64);
        }
        n
    }
}

/// The multisampled colour + depth a target rasterises into. There is no resolve bind
/// group and no resolve pass: a multisampled colour attachment names the stored-size
/// texture as its `resolve_target` and the hardware resolves it as part of ending the
/// pass, which is also where the hardware does it.
/// A render target's multisampled attachments at `samples` per pixel, or `None` when it
/// has none (one sample, or refused - see `msaa_refused_for_depth`).
pub(super) fn build_rtt_msaa(
    device: &wgpu::Device,
    size: wgpu::Extent3d,
    samples: u32,
    want_format: wgpu::TextureFormat,
    view_formats: &[wgpu::TextureFormat],
    sample_depth: bool,
) -> Option<MsaaAttachments> {
    let grant = samples > 1 && !msaa_refused_for_depth(sample_depth);
    grant.then(|| {
            let mc = device.create_texture(&wgpu::TextureDescriptor {
                label: Some("gxm-rtt-msaa-color"),
                size,
                mip_level_count: 1,
                sample_count: samples,
                dimension: wgpu::TextureDimension::D2,
                // The colour texture's OWN format, which a float surface overrides: a resolve
                // must land in a texture of the attachment's format. MEASURED on a racer:
                // a multisampled F16 target resolved an Rgba8Unorm attachment
                // into its Rgba16Float `color`, wgpu refused every pass, and the frame was black.
                format: want_format,
                // Downstream reads `color`, which this resolves into. TEXTURE_BINDING is for
                // a draw that reads the DESTINATION: its per-sample copy loads this image
                // (`copy_attachment_to_dest`).
                usage: wgpu::TextureUsages::RENDER_ATTACHMENT | wgpu::TextureUsages::TEXTURE_BINDING,
                // The sRGB twin, for the same reason `color` declares it: a gamma surface
                // is rendered through an sRGB view and a texture's view formats are fixed
                // at creation.
                view_formats,
            });
            let md = device.create_texture(&wgpu::TextureDescriptor {
                label: Some("gxm-rtt-msaa-depth"),
                size,
                mip_level_count: 1,
                sample_count: samples,
                dimension: wgpu::TextureDimension::D2,
                format: depth_format(),
                // Sampleable when a later pass reads this depth: the conversion then reads
                // it here, at sample 0 - see `GxmDepthTarget::multisampled`.
                usage: if sample_depth {
                    wgpu::TextureUsages::RENDER_ATTACHMENT | wgpu::TextureUsages::TEXTURE_BINDING
                } else {
                    wgpu::TextureUsages::RENDER_ATTACHMENT
                },
                view_formats: &[],
            });
            MsaaAttachments {
                color_view: mc.create_view(&Default::default()),
                depth_view: md.create_view(&Default::default()),
                color: mc,
                depth: md,
                samples,
            }
        })
}

/// The guest-depth reader of a render target: it converts the attachment the pass
/// rasterises into - the multisampled one when there is one, `depth` otherwise.
pub(super) fn build_rtt_gxm_depth(
    device: &wgpu::Device,
    size: wgpu::Extent3d,
    sample_depth: bool,
    msaa: &Option<MsaaAttachments>,
    depth: &wgpu::Texture,
) -> Option<GxmDepthTarget> {
    sample_depth.then(|| {
            let tex = device.create_texture(&wgpu::TextureDescriptor {
                label: Some("gxm-rtt-guest-depth"),
                size,
                mip_level_count: 1,
                sample_count: 1,
                dimension: wgpu::TextureDimension::D2,
                format: GXM_DEPTH_FORMAT,
                // COPY_SRC so a host that can read back (the headless chain dump) can
                // report what the conversion actually produced. A depth buffer nobody can
                // look at is a depth buffer nobody can debug.
                usage: wgpu::TextureUsages::RENDER_ATTACHMENT
                    | wgpu::TextureUsages::TEXTURE_BINDING
                    | wgpu::TextureUsages::COPY_SRC,
                view_formats: &[],
            });
            let view = tex.create_view(&Default::default());
            // The conversion reads the attachment the pass actually rasterises into: the
            // multisampled one when this target was granted it - see `GxmDepthTarget`.
            let src = msaa.as_ref().map_or(depth, |m| &m.depth);
            let src_view = src.create_view(&wgpu::TextureViewDescriptor {
                label: Some("gxm-rtt-depth-sampled"),
                aspect: wgpu::TextureAspect::DepthOnly,
                ..Default::default()
            });
            GxmDepthTarget { src_view, tex, view, multisampled: msaa.is_some() }
        })
}

pub(super) struct MsaaAttachments {
    /// The textures themselves, kept only so they can be `destroy()`ed when the target is
    /// rebuilt. In the browser a `wgpu::Texture` IS a `GPUTexture`, so dropping the Rust
    /// handle makes it GARBAGE rather than freeing it - see `RttSurface::destroy`.
    pub(super) color: wgpu::Texture,
    pub(super) depth: wgpu::Texture,
    pub(super) color_view: wgpu::TextureView,
    pub(super) depth_view: wgpu::TextureView,
    /// Samples per pixel, so a pipeline built for this pass matches its attachments.
    pub(super) samples: u32,
}

/// What a render pass already has bound, so a draw that changes nothing rebinds nothing.
///
/// Every gxp draw used to set its pipeline, all four bind groups, its vertex buffer and its
/// index buffer, unconditionally - seven state calls a draw, ~3,800 in a frame of 539 draws.
/// A scene does not change state that often: the draw-coverage report shows runs of one pair
/// repeated a dozen to thirty times in a row, and consecutive draws of one pair share their
/// pipeline and usually their uniform groups and buffers.
///
/// Identity is the wgpu handle's own (`BindGroup`, `Buffer` and `RenderPipeline` all compare
/// by the resource they name), plus the DYNAMIC OFFSETS, which are part of a binding and not
/// of the group - two draws sharing a bind group and differing in their uniform offset are a
/// different binding and must rebind. The state is per PASS, because a render pass begins
/// with nothing bound.
///
/// `VITASLOP_GXM_STATE_DEDUP=0` sets everything on every draw, as before.
#[derive(Default)]
pub(super) struct BoundState {
    pub(super) on: bool,
    pub(super) pipeline: Option<wgpu::RenderPipeline>,
    pub(super) groups: [Option<(wgpu::BindGroup, [u32; 2], usize)>; 4],
    pub(super) vbo: Option<(wgpu::Buffer, u64, u64)>,
    pub(super) ibo: Option<(wgpu::Buffer, u64, u64, bool)>,
    /// The stencil reference last set in this pass. A pass begins at reference 0.
    pub(super) stencil_ref: Option<u32>,
}

impl BoundState {
    pub(super) fn new() -> Self {
        BoundState { on: state_dedup(), ..Default::default() }
    }

    /// Forget everything: what another encode path bound is not tracked here, and assuming
    /// otherwise would skip a set that was needed.
    pub(super) fn invalidate(&mut self) {
        self.pipeline = None;
        self.groups = Default::default();
        self.vbo = None;
        self.ibo = None;
        self.stencil_ref = None;
    }

    pub(super) fn pipeline_changed(&mut self, p: &wgpu::RenderPipeline) -> bool {
        if self.on && self.pipeline.as_ref() == Some(p) {
            return false;
        }
        self.pipeline = Some(p.clone());
        true
    }

    pub(super) fn group_changed(&mut self, i: usize, g: &wgpu::BindGroup, offs: &[u32]) -> bool {
        let mut key = [0u32; 2];
        key[..offs.len()].copy_from_slice(offs);
        if self.on
            && let Some((have, have_offs, n)) = self.groups[i].as_ref()
                && have == g && *n == offs.len() && have_offs[..*n] == key[..offs.len()] {
                    return false;
                }
        self.groups[i] = Some((g.clone(), key, offs.len()));
        true
    }

    pub(super) fn vbo_changed(&mut self, b: &wgpu::Buffer, off: u64, len: u64) -> bool {
        if self.on && self.vbo.as_ref().is_some_and(|(h, o, l)| h == b && *o == off && *l == len) {
            return false;
        }
        self.vbo = Some((b.clone(), off, len));
        true
    }

    pub(super) fn ibo_changed(&mut self, b: &wgpu::Buffer, off: u64, len: u64, u32_fmt: bool) -> bool {
        if self.on
            && self.ibo.as_ref().is_some_and(|(h, o, l, f)| h == b && *o == off && *l == len && *f == u32_fmt)
        {
            return false;
        }
        self.ibo = Some((b.clone(), off, len, u32_fmt));
        true
    }
}

/// `0` rebinds every draw's whole state, which is what shipped before [`BoundState`].
pub(super) fn state_dedup() -> bool {
    use std::sync::OnceLock;
    static ON: OnceLock<bool> = OnceLock::new();
    *ON.get_or_init(|| crate::knobs::var("VITASLOP_GXM_STATE_DEDUP").ok().as_deref() != Some("0"))
}

/// A/B instrument: force every pass to ONE sample, whatever the guest asked for.
///
/// This is not a quality setting and there is no per-platform default to pick. The sample
/// count is the guest's own `SceGxmRenderTargetParams::multisampleMode` and it is honoured
/// on every engine - see [`gxm_sample_count`] for why multisampling (one shader invocation
/// per pixel) is the faithful reading of that request and supersampling is not. What this
/// exists for is the measurement: a render change that cannot be turned off cannot be
/// priced, and the phone is where the price is.
pub(super) fn no_multisample() -> bool {
    use std::sync::OnceLock;
    static OFF: OnceLock<bool> = OnceLock::new();
    *OFF.get_or_init(|| crate::knobs::flag("VITASLOP_GXM_NO_MULTISAMPLE"))
}

/// Whether a target whose DEPTH a later pass samples must be refused its multisampled
/// attachments.
///
/// Only on a compatibility-mode adapter: the depth conversion then has to read the
/// attachment through a sampler, and a multisampled depth texture can only be read by
/// `textureLoad`, which compatibility mode refuses on depth. Everywhere else the conversion
/// reads sample 0 of the multisampled attachment (`gxm_depth_ms`), so the pass keeps the
/// antialiasing the guest asked for. MEASURED on a football title: its 960x544 world target
/// is 2X and a later pass samples its depth, so the whole match was drawn single-sampled.
pub(super) fn msaa_refused_for_depth(sample_depth: bool) -> bool {
    sample_depth && crate::gpu::compat_mode()
}

/// Say - once per target - that a target the guest created MULTISAMPLED got a multisampled
/// attachment, and at how many samples. The refusals are [`report_multisample_refused`].
///
/// The guest asking and us obliging used to be indistinguishable in a log: the runtime
/// reported the request unconditionally and the renderer silently ignored it, so a reader
/// could see the ask and had no way to learn the answer.
pub(super) fn report_multisample_granted(addr: u32, w: u32, h: u32, mode: u32, samples: u32) {
    use std::collections::HashSet;
    use std::sync::Mutex;
    static SEEN: Mutex<Option<HashSet<u32>>> = Mutex::new(None);
    let mut g = SEEN.lock().unwrap_or_else(|e| e.into_inner());
    if !g.get_or_insert_with(HashSet::new).insert(addr) {
        return;
    }
    // 2X served with 4 samples is the one place this path is not exact, so it is named
    // every time rather than folded into the same sentence as an exact grant.
    let exact = if mode == 1 {
        " - the guest asked for 2X and WebGPU guarantees only 1 and 4, so this pass is \
         antialiased MORE finely than the title asked"
    } else {
        ""
    };
    report!("gxm rtt {addr:#x} ({w}x{h}): MULTISAMPLE granted - {samples} samples per pixel, resolved into the stored size at end of pass{exact}");
}

/// Say - once per target - that a target the guest created MULTISAMPLED was rasterised at
/// one sample anyway, and which reason applied. Every refusal here is deliberate and
/// narrow, which is exactly the kind of quiet exception that later reads as a bug
/// somewhere else.
pub(super) fn report_multisample_refused(addr: u32, w: u32, h: u32, gamma: bool) {
    use std::collections::HashSet;
    use std::sync::Mutex;
    static SEEN: Mutex<Option<HashSet<u32>>> = Mutex::new(None);
    let mut g = SEEN.lock().unwrap_or_else(|e| e.into_inner());
    if !g.get_or_insert_with(HashSet::new).insert(addr) {
        return;
    }
    let why = if gamma {
        "it is a GAMMA-CORRECT surface, where the ROP sRGB-encodes each store after \
         blending, and whether the resolve averages the encoded or the linear values has \
         not been measured here"
    } else {
        "a later pass samples this target's DEPTH and this is a COMPATIBILITY-MODE adapter, \
         which refuses the `textureLoad` a multisampled depth texture is read by - so the \
         depth conversion has to read a single-sampled attachment"
    };
    // `report_warn`, not `report`: this one says the output is APPROXIMATED, which is the
    // documented job of the warn-level macro, and a refusal that only shows up at
    // `vitaslop::gxm=debug` is a refusal nobody reads. The grant above stays at debug -
    // it says the renderer did what it was told, which is not news.
    // STATUS, not a warning: both refusals are this renderer's own standing decisions
    // (a sampled depth attachment, a gamma-correct resolve), taken on every run of a title
    // that has those passes. A warning is a fix owed; this is a documented trade, and it
    // still lands in the panel's STATUS section where a capture reads it.
    report_status!("gxm rtt {addr:#x} ({w}x{h}): MULTISAMPLE REFUSED - {why}. This pass stays more aliased than the title asked.");
}

/// Say - once per target - that the guest created a render target MULTISAMPLED but did NOT
/// put its colour surface in `SCE_GXM_COLOR_SURFACE_SCALE_MSAA_DOWNSCALE`.
///
/// The two together mean "rasterise at N samples, store the resolve". Multisampled WITHOUT
/// the scale mode means the surface's own memory is expected to hold the raw samples, and a
/// resolved image is then the wrong bytes at the wrong size for anything that reads it. We
/// always resolve, so this is a real divergence rather than a curiosity - and it warns
/// rather than reports, because nothing downstream would look wrong in an obvious way.
pub(super) fn report_unresolved_multisample_surface(addr: u32, w: u32, h: u32) {
    use std::collections::HashSet;
    use std::sync::Mutex;
    static SEEN: Mutex<Option<HashSet<u32>>> = Mutex::new(None);
    let mut g = SEEN.lock().unwrap_or_else(|e| e.into_inner());
    if !g.get_or_insert_with(HashSet::new).insert(addr) {
        return;
    }
    report_warn!(
        "gxm rtt {addr:#x} ({w}x{h}): the render target is MULTISAMPLED but its colour \
         surface is NOT in MSAA_DOWNSCALE, so the guest expects this memory to hold the raw \
         SAMPLES - we store the resolved image instead"
    );
}

/// A render target's depth in the guest's own encoding, plus what it takes to produce and
/// sample it.
pub(super) struct GxmDepthTarget {
    /// A sampleable view of the depth ATTACHMENT (the conversion pass's input).
    pub(super) src_view: wgpu::TextureView,
    /// `src_view` is the MULTISAMPLED depth attachment, read at sample 0 by the
    /// `gxm_depth_ms` pipeline. Depth is interpolated PER SAMPLE under MSAA, so this is the
    /// depth at sample 0's position - a sub-pixel offset from the centre a single-sampled
    /// attachment holds - and at an edge it is whichever primitive covers that sample. The
    /// hardware's own buffer for a multisampled target holds per-sample depths as well
    /// (a 2X target's depth surface is twice the colour's width), so a guest reading it
    /// reads samples, not centres.
    pub(super) multisampled: bool,
    /// The converted R32Float texture (the output), and the view a draw samples it through.
    /// The texture is held (not just its view) so it can be `destroy()`ed with the target -
    /// see `RttSurface::destroy`.
    pub(super) tex: wgpu::Texture,
    pub(super) view: wgpu::TextureView,
}

/// Round `v` up to a multiple of `align` (a power of two).
pub(super) fn align_up(v: u64, align: u64) -> u64 {
    (v + align - 1) & !(align - 1)
}
