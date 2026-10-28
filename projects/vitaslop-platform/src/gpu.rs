//! The shared cube render pipeline. The WGSL shader, vertex layout, and per-draw
//! encoding live here once so the native headless renderer (`vitaslop-native`)
//! and the browser WebGPU canvas (`vitaslop-web`) draw identically - the native
//! path stays the pixel oracle for the browser. Only the surrounding host bits
//! differ: native acquires an adapter headlessly and reads pixels back, the
//! browser acquires a device from `navigator.gpu` and presents to a canvas.
//!
//! [`DrawBatch`] is the neutral, GPU-free seam: the runtime turns its captured
//! GXM scene into a list of these, and [`CubeRenderer`] (behind the `gpu`
//! feature) draws them. Keeping `DrawBatch` free of any wgpu types lets the
//! engine-agnostic runtime produce it without pulling in a GPU stack.

/// A renderer diagnostic, at `debug` on the `vitaslop::gxm` target.
///
/// # Why these are not `eprintln!`
/// This module is the SHARED render path: the same code draws the native pixel oracle
/// and the browser canvas. On wasm32 stderr is a no-op sink, so every `eprintln!` here
/// was discarded in the browser - including the recompiler's own per-scene verdict, the
/// one line that says whether a frame's draws were really recompiled or quietly fell
/// back. The engine that is hardest to inspect was the one reporting nothing, and a
/// black frame there was indistinguishable from a correct one.
///
/// Read them with `RUST_LOG=vitaslop::gxm=debug` natively, or
/// `VITASLOP_LOG=warn,vitaslop::gxm=debug` in the browser. Anything that means the
/// renderer DEGRADED - a fallback, a dropped draw, an approximated state - goes through
/// [`report_warn`] instead, so it survives the default `warn` filter on both engines
/// without anyone having to know to ask for it.
macro_rules! report {
    ($($arg:tt)*) => { tracing::debug!(target: "vitaslop::gxm", $($arg)*) };
}

/// A renderer diagnostic that means the output is WRONG or approximated. Always visible
/// at the default filter - see [`report`].
macro_rules! report_warn {
    ($($arg:tt)*) => { tracing::warn!(target: "vitaslop::gxm", $($arg)*) };
}

/// A STATUS line: something the panel should carry, that is NOT a defect.
///
/// # The rule this exists to enforce
/// A WARNING IS A CLAIM THAT SOMETHING IS BROKEN AND OWED A FIX. Before writing
/// `report_warn!`, ask what you would FIX on reading the line; if there is no answer, it is
/// status and belongs here. The frame's pass structure, a precompile count, a texture working
/// set, "the state is live", "the tallies are clean" - none of those are defects, and filing
/// them at `warn` made the one channel a phone has unreadable: the genuine findings sat
/// between twenty copies of a trace, and a user could not tell a defect from a heartbeat.
///
/// This is NOT a way to quieten an inconvenient warning. Lowering a real warning is silencing
/// it. Status rides at `info` on its own target, which the web logging layer forces ON
/// regardless of `VITASLOP_LOG` and renders in the panel's own STATUS section - so nothing
/// that used to be visible stops being visible. It is sorted, not hidden.
#[allow(unused_macros)]
macro_rules! report_status {
    ($($arg:tt)*) => { tracing::info!(target: "vitaslop::status", $($arg)*) };
}

/// A diagnostic with ONE shape and MANY subjects: the first subject in full, the rest folded
/// into a single line that only advances a count.
///
/// # Why this exists, and it is not a style point
/// The diagnostics panel each front end shows - the only channel a phone has, and the only
/// one a shipped desktop build has - keeps DISTINCT WARN lines (`diag::RING_CAP`, 96 of them)
/// and folds two lines that differ only in
/// a trailing `count=<n>` (`vitaslop_platform::diag::dedupe_key`). A report that NAMES its
/// subject is distinct per subject, so a shape with hundreds of subjects fills the panel by
/// itself and evicts every other finding.
///
/// MEASURED, on a phone capture the user sent of one title's round: the panel read
/// `648 earlier DISTINCT line(s) dropped` and **every one of the 96 lines it kept was one of
/// two shapes** - the region-clip alignment note and the narrow-attribute fill note. A desktop
/// run of the same title emits 512 distinct rectangles and 177 distinct (pair, location)
/// pairs. The person asking why the picture is wrong got back two findings repeated 96 times
/// and nothing else. [[vitaslop-a-diagnostic-can-bury-the-findings]]
///
/// So: subject one reads exactly as it did before (a title with a single occurrence loses
/// nothing), every later subject emits its full text at `debug` and advances ONE folded
/// `warn` line. The count is the whole population, not a sample, and it is the number that
/// says whether a shape is an exception or the norm - which is itself information the
/// per-subject form never carried.
#[cfg(feature = "gpu")]
pub(crate) struct Census {
    /// Subjects seen, including the first.
    n: std::sync::atomic::AtomicU64,
    /// The first subject's name, so the folded line still points somewhere concrete.
    first: std::sync::OnceLock<String>,
}

#[cfg(feature = "gpu")]
impl Census {
    pub(crate) const fn new() -> Self {
        Self { n: std::sync::atomic::AtomicU64::new(0), first: std::sync::OnceLock::new() }
    }

    /// Report one subject. `headline` must be the SAME string for every subject of this
    /// census (it is what the folded line is keyed on); `subject` names this one.
    ///
    /// The caller is responsible for having already deduplicated repeat sightings of the same
    /// subject - a census of the same rectangle seen a thousand times counts occurrences, not
    /// subjects, and the two answer different questions.
    pub(crate) fn note(&self, headline: &str, subject: &str) {
        self.note_at(true, headline, subject);
    }

    /// The same census on the STATUS channel: for a shape that is expected and measured
    /// rather than owed a fix, so it reaches the panel's STATUS section and never the
    /// console of a clean run.
    pub(crate) fn note_status(&self, headline: &str, subject: &str) {
        self.note_at(false, headline, subject);
    }

    fn note_at(&self, warn: bool, headline: &str, subject: &str) {
        use std::sync::atomic::Ordering;
        let n = self.n.fetch_add(1, Ordering::Relaxed) + 1;
        if n == 1 {
            let _ = self.first.set(subject.to_string());
            if warn {
                report_warn!("{headline} {subject}");
            } else {
                report_status!("{headline} {subject}");
            }
            return;
        }
        // The full text still exists, one level down, for anyone chasing a specific subject.
        report!("{headline} {subject}");
        // Everything before `count=` is byte-for-byte identical on every emission, which is
        // what lets the panel fold these into one line and show the LATEST count. Nothing
        // that varies with `n` may appear before the suffix, or the fold stops working and
        // this becomes the flood it replaced.
        let first = self.first.get().map_or("?", String::as_str);
        if warn {
            report_warn!(
                "{headline} [more than one subject - this line carries the running total; the \
                 first was: {first}] count={n}"
            );
        } else {
            report_status!(
                "{headline} [more than one subject - this line carries the running total; the \
                 first was: {first}] count={n}"
            );
        }
    }
}

/// The output of a diagnostic whose own KNOB is already the gate.
///
/// Emitted at `warn`, like [`report_warn`], and for a reason that is not about severity:
/// setting `VITASLOP_GXP_INPUTS_ORDER=all` is an unambiguous request for these lines, and
/// a second, invisible gate underneath it can only turn that request into silence. These
/// were `report!` (`debug`) while every documented repro command in the project's notes
/// says `VITASLOP_LOG=warn` - so the knob was set, the run completed, and the diagnostic
/// printed nothing, which reads exactly like "this draw never happened". It cost a run
/// here and, going by the notes, a good deal more than that elsewhere.
///
/// Use this ONLY where a knob has already decided the line should exist. A diagnostic that
/// fires on every draw regardless still belongs at `debug`.
#[cfg(feature = "gpu")]
macro_rules! report_knob {
    ($($arg:tt)*) => { tracing::warn!(target: "vitaslop::gxm", $($arg)*) };
}

/// The guest frame the renderer is presenting, as the engine last said (0 if no engine does).
/// The renderer counts its own presents, which are not guest frames; a frame-windowed
/// diagnostic has to key on the number scripted input and a shot are labelled with.
static GUEST_FRAME: std::sync::atomic::AtomicU64 = std::sync::atomic::AtomicU64::new(0);

/// Tell the renderer which guest frame it is about to present - see [`GUEST_FRAME`].
pub fn set_guest_frame(frame: u64) {
    GUEST_FRAME.store(frame, std::sync::atomic::Ordering::Relaxed);
}

#[cfg(feature = "gpu")]
fn guest_frame() -> u64 {
    GUEST_FRAME.load(std::sync::atomic::Ordering::Relaxed)
}

/// One ready-to-draw batch: a model-view-projection matrix and the raw vertex and
/// index bytes, already snapshotted from guest memory. The vertex layout is the
/// cube's fixed-function format (see [`CUBE_SHADER`]): position `float32x3` at
/// offset 0, color `unorm8x4` at offset 12, stride 16.
#[derive(Clone, Debug, PartialEq)]
pub struct DrawBatch {
    /// Column-major 4x4 MVP matrix (the guest's vertex default uniform buffer).
    pub mvp: [f32; 16],
    /// Interleaved vertex bytes, stride 16.
    pub vertices: Vec<u8>,
    /// Index buffer bytes.
    pub indices: Vec<u8>,
    /// Number of indices to draw.
    pub index_count: u32,
    /// True if indices are 32-bit; false for 16-bit (the cube uses 16-bit).
    pub index_u32: bool,
}

/// The fixed-function cube shader, the blob-free equivalent of the guest's
/// placeholder GXP shaders. It transforms each vertex by the captured MVP and
/// interpolates per-vertex color. The `clip.z` remap converts GL-style clip space
/// (z in [-1, 1], which the guest's projection targets) to WebGPU clip space
/// (z in [0, 1]).
pub const CUBE_SHADER: &str = r#"
struct Uniforms { mvp: mat4x4<f32> };
@group(0) @binding(0) var<uniform> u: Uniforms;

struct VsOut {
    @builtin(position) pos: vec4<f32>,
    @location(0) color: vec4<f32>,
};

@vertex
fn vs(@location(0) position: vec3<f32>, @location(1) color: vec4<f32>) -> VsOut {
    var out: VsOut;
    var clip = u.mvp * vec4<f32>(position, 1.0);
    clip.z = (clip.z + clip.w) * 0.5;
    out.pos = clip;
    out.color = color;
    return out;
}

@fragment
fn fs(in: VsOut) -> @location(0) vec4<f32> {
    return in.color;
}
"#;

// ---------------------------------------------------------------------------
// General GXM renderer seam
// ---------------------------------------------------------------------------
//
// [`CubeRenderer`] above is the fixed-function 3D-cube path (one shader, one
// vertex layout, opaque depth-tested). A real 2D title needs more: per-draw
// coordinate spaces, textured sprites, and alpha blending in submission order.
// [`RenderScene`] is the neutral, GPU-free seam for that - the runtime decodes a
// captured GXM scene into it (reusing the exact interpretation its software
// rasterizer uses, so the two agree), and [`GxmRenderer`] (behind `gpu`) draws
// it. Keeping the decode in the runtime and only the draw here means the GPU
// renderer never touches guest memory or a GXM enum - it sees canonical vertices
// and a linear RGBA8 texture, and stays the faithful GPU twin of the software
// oracle.

/// The coordinate space a general draw's vertex positions live in, recovered by
/// the runtime from the draw's vertex layout - the same [`Space`] the software
/// rasterizer uses. Determines both the vertex transform and whether the draw is
/// depth-tested (only `Mvp` is; the 2D spaces paint in submission order).
#[derive(Clone, Copy, Debug, PartialEq)]
pub enum DrawSpace {
    /// Object space transformed by a captured column-major 4x4 MVP uniform (the 3D
    /// path): perspective divide, GL-style clip z, depth-tested opaque.
    Mvp([f32; 16]),
    /// Clip coordinates emitted directly, in [-1, 1] (a fullscreen pass).
    Ndc,
    /// Screen pixels in [0, surface] with Y down (2D sprite quads whose vertex
    /// program baked the pixel-to-clip transform).
    Pixel,
}

/// A decoded, GPU-ready texture: a tightly-packed linear `Rgba8Unorm` image plus a
/// content/identity `key` the renderer caches uploads by, so an unchanged texture
/// (the common case - a font atlas or background reused every frame) is uploaded
/// once and then only bound. `rgba` is shared (`Arc`) so re-handing the same
/// texture to the renderer each frame is a pointer copy, not a pixel copy.
/// The layout of [`GxmTexture::rgba`], and the only two the renderer uploads.
///
/// Deliberately a closed set of two rather than "whatever the guest format was". The decoders
/// converge every one of the guest's several dozen formats onto one of these, so the renderer,
/// the software rasterizer and the upload path each handle two cases and not fifty - and a
/// format added later has to pick one, rather than silently arriving as bytes nobody sizes.
#[derive(Clone, Copy, PartialEq, Eq, Debug)]
pub enum TexelSeam {
    /// Four `u8` lanes per texel, normalized. Every colour format lands here.
    Rgba8,
    /// Four IEEE binary16 lanes per texel, little-endian. Exact for the guest's own F16
    /// formats, which is the point: these carry data, and a re-encode would defeat the reason
    /// for the wider seam.
    Rgba16Float,
    /// Two IEEE binary32 lanes per texel, little-endian: the guest's F32F32, exactly. NOT
    /// filterable in core WebGPU, so a unit bound to one is laid out unfilterable with a nearest
    /// sampler ([`gxm::SamplerDim::TwoF32`]).
    Rg32Float,
}

impl TexelSeam {
    /// Bytes one texel occupies in `rgba`.
    pub fn bytes_per_texel(self) -> usize {
        match self {
            TexelSeam::Rgba8 => 4,
            TexelSeam::Rgba16Float => 8,
            TexelSeam::Rg32Float => 8,
        }
    }
}

/// The optional device features this renderer wants, of those the adapter offers.
///
/// # Every device asked for `Features::empty()`, and that was the whole reason a Vita frame
/// needed 260 MB
/// A guest frame that references 260 MB of texture is not referencing 260 MB - a Vita has
/// nowhere near it. Every compressed surface was being CPU-decoded to RGBA8 on the way in,
/// which is a 4-8x expansion, and past the cache budget the allocation fails on a phone and
/// the draw comes out WHITE. Asking for `TEXTURE_COMPRESSION_BC` is the precondition for
/// handing the guest's own blocks over instead.
///
/// It is requested only where the adapter HAS it - `request_device` fails outright otherwise -
/// so an adapter without it simply keeps the decode path, and [`CompressedUpload`] is ignored.
/// This is one function rather than four copies because four copies is how three of them come
/// to be right and one silently is not.
/// The limits every engine in this project asks a device for: the conservative downlevel floor,
/// with the resolution-derived limits AND `max_buffer_size` raised to what the adapter really
/// offers.
///
/// # Why this is one function rather than four copies
/// The floor has to be the SAME on the desktop and in the browser or the native renderer is not
/// the browser's pixel oracle - a pipeline that exceeds a limit is not a fallback the recompiler
/// reports, it is an invalid pipeline whose render pass draws nothing. That argument was already
/// written down twice, in two of the four places that request a device, and the four had drifted
/// anyway.
///
/// # Why `max_buffer_size` had to move
/// `downlevel_defaults()` caps a buffer at 256 MB. MEASURED on a retail sports title's gameplay
/// frame, on BOTH engines: one pass's repacked vertex data is **362 MB**, the arena rounds that
/// up to 512 MB, and `create_buffer` is refused -
/// `Buffer size 536870912 is greater than the maximum buffer size (268435456)`. What follows is
/// not a degraded frame, it is a LOST one: the buffer is invalid, so the `write_buffer`, the
/// `set_vertex_buffer` and the submit that names it all fail in turn
/// [[vitaslop-a-failed-submit-loses-the-whole-frame]]. The browser's own error text says the
/// adapter supports 2 GB and tells you to ask for it.
///
/// Asking for the adapter's figure costs nothing - a limit is a ceiling, not an allocation - and
/// a device that genuinely cannot go above 256 MB reports 256 MB here and is unchanged. It does
/// NOT make the 362 MB arena acceptable: that is the guest's 50 MB of geometry expanded to f32
/// by the repack, and shrinking it is a separate and larger job. This only stops the frame being
/// thrown away while that stands.
#[cfg(feature = "gpu")]
pub fn device_limits(adapter: &wgpu::Adapter) -> wgpu::Limits {
    let have = adapter.limits();
    let mut limits = wgpu::Limits::downlevel_defaults().using_resolution(have.clone());
    limits.max_buffer_size = limits.max_buffer_size.max(have.max_buffer_size);
    // >>> THE UNIFORM BINDING CEILING IS WEBGPU'S 64 KiB, NOT THE DOWNLEVEL 16 KiB. The
    // recompiler admits a guest-memory window binding up to
    // `vitaslop_gxp_shader::module::MEM_WINDOW_BINDING_MAX` (WebGPU's guaranteed
    // `maxUniformBufferBindingSize`), and `downlevel_defaults()` caps a binding at 16 KiB - so
    // a program with one 16 KiB window plus its 16-byte header was ACCEPTED and then its bind
    // group was invalid: `Buffer binding 1 range 16400 exceeds max_*_buffer_binding_size limit
    // 16384`, and every submit that named it was lost. MEASURED: one title's first mission,
    // every frame black. Every WebGPU adapter offers 64 KiB; one that offers less keeps its own.
    let want_ubo = vitaslop_gxp_shader::module::MEM_WINDOW_BINDING_MAX;
    limits.max_uniform_buffer_binding_size = limits
        .max_uniform_buffer_binding_size
        .max(want_ubo.min(have.max_uniform_buffer_binding_size));
    // A window past that binds as read-only STORAGE (`module::mem_window_storage`), which the
    // downlevel defaults already grant at WebGPU's 128 MiB in every stage that may use it.
    // Whether the VERTEX stage may is an adapter capability rather than a limit, so it is
    // noted here, where every engine's device is made, for the pipeline builder to ask.
    VERTEX_STORAGE.store(
        adapter.get_downlevel_capabilities().flags.contains(wgpu::DownlevelFlags::VERTEX_STORAGE),
        std::sync::atomic::Ordering::Relaxed,
    );
    // A fragment program that STORES to guest memory binds a WRITABLE storage buffer in the
    // fragment stage - core WebGPU, but a downlevel capability elsewhere (and absent from
    // WebGPU's compatibility mode), so it is recorded the same way.
    FRAGMENT_STORAGE.store(
        adapter
            .get_downlevel_capabilities()
            .flags
            .contains(wgpu::DownlevelFlags::FRAGMENT_WRITABLE_STORAGE),
        std::sync::atomic::Ordering::Relaxed,
    );
    limits
}

/// Whether the adapter lets a FRAGMENT shader write a storage buffer - see [`device_limits`].
#[cfg(feature = "gpu")]
static FRAGMENT_STORAGE: std::sync::atomic::AtomicBool = std::sync::atomic::AtomicBool::new(false);

/// [`FRAGMENT_STORAGE`]: `false` until a device has been made through [`device_limits`].
#[cfg(feature = "gpu")]
pub fn fragment_storage_ok() -> bool {
    FRAGMENT_STORAGE.load(std::sync::atomic::Ordering::Relaxed)
}

/// Whether the adapter the device came from lets a VERTEX shader read a storage buffer - see
/// [`device_limits`], which records it.
#[cfg(feature = "gpu")]
static VERTEX_STORAGE: std::sync::atomic::AtomicBool = std::sync::atomic::AtomicBool::new(false);

/// [`VERTEX_STORAGE`]: `false` until a device has been made through [`device_limits`].
#[cfg(feature = "gpu")]
pub fn vertex_storage_ok() -> bool {
    VERTEX_STORAGE.load(std::sync::atomic::Ordering::Relaxed)
}

#[cfg(feature = "gpu")]
pub fn wanted_features(adapter: &wgpu::Adapter) -> wgpu::Features {
    // BC AND ETC2, because the two together are what covers both halves of the hardware this
    // runs on. BC is preferred where both exist: the guest's own `UBC1/2/3` blocks ARE BC, so
    // that adapter can take them with no re-encode at all, while ETC2 always costs a lossy
    // transcode. An adapter that offers neither keeps the RGBA8 decode.
    let mut have = adapter.features();
    // >>> `VITASLOP_NO_BC=1` MAKES A DESKTOP BEHAVE LIKE THE TARGET DEVICE.
    //
    // The phone's adapter exposes `etc2, astc` and no BC, so it takes the transcode path for
    // EVERY compressed texture - which is where the GPU encoder, the ETC2 target and the whole
    // `CompressedData::Gpu` plan live. A desktop with BC never enters any of it, so all of it
    // was unexercised locally and its first real execution was on the user's phone. That has
    // already cost one shipped build.
    //
    // On this machine the Intel iGPU exposes BOTH families over Vulkan, so
    // `WGPU_ADAPTER_NAME=Arc WGPU_BACKEND=vulkan VITASLOP_NO_BC=1` is the device's texture
    // configuration, on real content, with the headless renderer.
    //
    // It only ever REMOVES a capability - the same shape as `VITASLOP_TEX_COMPRESS=0`, and the
    // reason a knob is admissible here at all ([[vitaslop-knob-is-the-gate-not-the-level]]): it
    // cannot make this build do something a real adapter would not.
    if crate::knobs::var("VITASLOP_NO_BC").ok().as_deref() == Some("1") {
        have.remove(wgpu::Features::TEXTURE_COMPRESSION_BC);
        report_warn!(
            "gxm textures: VITASLOP_NO_BC=1 - pretending this adapter has no BC, so every \
             compressed texture takes the TRANSCODE path exactly as the target device does"
        );
    }
    let mut want =
        have & (wgpu::Features::TEXTURE_COMPRESSION_BC | wgpu::Features::TEXTURE_COMPRESSION_ETC2);
    // >>> GPU TIMESTAMPS, WHERE THE ADAPTER OFFERS THEM. The work-done latency the browser
    // measures cannot separate GPU execution from the event loop's dispatch of its callback
    // ([`GpuTimestamps`] is the instrument that can), and a device that reads 44 ms of
    // latency against a 27 ms period has no other way to say whether its GPU is the bottleneck.
    // A pair of timestamps per render pass costs the GPU nothing it can measure.
    // >>> DUAL-SOURCE BLENDING, WHERE THE ADAPTER OFFERS IT. A fragment program that blends
    // for itself in ALU and is LINEAR in the destination (`out = dst*F + G`, F and G per-pixel)
    // is lowered to a two-output shader and a `src0 + dst*src1` pipeline blend instead of a
    // render-pass split with an attachment copy. MEASURED on a baseball title: 130 splits and
    // 260 MB copied PER FRAME on one menu screen, 70-102 splits per pass in gameplay, from
    // three such programs. The target device has the feature; where an adapter lacks it the
    // split path stays. See `vitaslop_gxp_shader::module::dual_source_eligible`.
    if have.contains(wgpu::Features::DUAL_SOURCE_BLENDING) {
        want |= wgpu::Features::DUAL_SOURCE_BLENDING;
    } else {
        report_status!(
            "gxm: this adapter offers no `dual-source-blending`, so a fragment program that \
             blends itself over the destination costs a render-pass split here."
        );
    }
    // >>> `shader-f16`, BECAUSE EVERY HALF-PRECISION STORE THE RECOMPILER EMITS HAS TO ROUND
    // >>> TO NEAREST EVEN AND THE CHEAP WAY TO DO THAT IS THE LANGUAGE'S OWN CONVERSION.
    //
    // A USSE 16-bit register is emulated as a `u32` holding two halves, and the narrowing used
    // to be `pack2x16float` - whose rounding mode WGSL leaves to the implementation, and which
    // this machine's adapter TRUNCATES (`probe-f16round.mjs`: 48/48 round-toward-zero, 28/48
    // round-to-nearest-even). The CPU reference rounds to nearest even, as the guest's hardware
    // does, so every f16 store disagreed with it in a direction that BIASES. Fragment programs
    // on real titles are 70-90% F16, so that was essentially all fragment arithmetic - and two
    // devices lowering the builtin differently means the same title renders differently on each.
    //
    // Both arms of the fix round the same way; the feature only decides whether the narrowing
    // is one instruction or about ten. Without it the portable bit arithmetic still ships, and
    // this says so rather than leaving the phone's cost unexplained.
    // See `vitaslop_gxp_shader::wgsl::HALF_LO_FN`.
    if have.contains(wgpu::Features::SHADER_F16) {
        want |= wgpu::Features::SHADER_F16;
    } else {
        report_status!(
            "gxp f16: this adapter offers no `shader-f16`, so every half-precision store rounds \
             through portable bit arithmetic (about ten operations) instead of one conversion \
             instruction. The NUMBERS are the same either way - only the cost differs."
        );
    }
    if have.contains(wgpu::Features::TIMESTAMP_QUERY) {
        want |= wgpu::Features::TIMESTAMP_QUERY;
    } else {
        report_status!(
            "gxm: this adapter offers no `timestamp-query`, so the GPU TIME line cannot be \
             measured on this run - only the work-done latency, which includes the event loop."
        );
    }
    // >>> `depth-clip-control`, BECAUSE POWERVR CLAMPS DEPTH WHERE WEBGPU CLIPS IT.
    //
    // A primitive that runs past the near or far plane is drawn AT that plane on the guest's
    // hardware, not thrown away. Without this feature the clip fixup has to imitate that in the
    // vertex stage, and a per-VERTEX clamp is not a clamp: it moves one corner of a triangle
    // and the rasteriser then interpolates a plane that no longer passes through the geometry.
    // With it the rasteriser clamps per FRAGMENT, which is what the hardware does.
    //
    // Reported either way, in the section people paste off a device: whether a primitive past
    // the far plane is drawn or discarded is a difference you can see, and "no line" is not an
    // answer to which of the two this run did.
    if have.contains(wgpu::Features::DEPTH_CLIP_CONTROL) {
        want |= wgpu::Features::DEPTH_CLIP_CONTROL;
        report_status!(
            "gxm depth: this adapter offers `depth-clip-control`. It is NOT taken by default - \
             it disables the near and far z clip planes TOGETHER, and without the near one a \
             primitive straddling the camera plane is drawn AT the plane instead of being cut \
             (a baseball title's post-hit wedge). `VITASLOP_GXP_UNCLIPPED_DEPTH=1` turns the clamp back on."
        );
    } else {
        report_warn!(
            "gxm depth: this adapter offers no `depth-clip-control`, so a primitive that runs \
             past the near or far plane is CLIPPED where the guest's hardware would draw it at \
             the plane. Recompiled draws still write the guest's own window depth; what is lost \
             is only the geometry outside the guest's clip volume."
        );
    }
    // >>> DEPTH IS CLIPPED AT THE NEAR PLANE, NOT CLAMPED. `unclipped_depth` (the
    // `depth-clip-control` feature) turns off the near AND far z clip planes together and
    // clamps instead, which was taken on for the FAR plane - PowerVR really does draw
    // far geometry at the plane rather than losing it. The near plane is a different story,
    // and this is what it cost:
    //
    // MEASURED on a baseball title, the post-hit "wedge" the user has reported since it was first
    // playable. It is the field-decal pair `764e083d01208aa3`, four vertices of a flat ground
    // quad (y = 0, x in [-132.6, 132.6], z in [-164.4, 100.75]) textured with the 512x512
    // top-down infield atlas `0x8ca7efc0`. Through the draw's own view-projection (read out of
    // the vertex memory window it loads through) its corners land at clip
    // **w = -2,858, -2,858, +101,609, +101,609** - it STRADDLES the camera plane. With the near
    // clip disabled the half behind the eye is drawn AT the plane instead of being cut: a
    // magnified decal covering **163,057 samples, 55% of the 720x408 world pass**, with its
    // depth clamped to the nearest value so it beats the field it should be lying on. The
    // camera only drops to ground level after a hit, which is why it appears then and grows
    // frame over frame as the camera keeps moving.
    //
    // A guest whose hardware HAS a near plane and clips at it is not served by a renderer that
    // does not. What clipping costs instead is geometry past the FAR plane, which the guest
    // would have drawn at the plane - a far smaller error than painting a decal over half the
    // frame, and one no title here has been shown to depend on.
    //
    // `VITASLOP_GXP_UNCLIPPED_DEPTH=1` restores the clamp for anyone who wants to re-measure
    // the far-plane half of this.
    // Stencil rides with depth; see [`depth_format`]. Without the feature the attachment is
    // `Depth24PlusStencil8`, which every WebGPU device has.
    // `VITASLOP_GXP_DEPTH24=1` declines the feature where it is offered, so a desktop renders
    // with the `Depth24PlusStencil8` attachment a device without it (the PowerVR phone) gets.
    let depth24 = crate::knobs::var("VITASLOP_GXP_DEPTH24").map(|v| v == "1").unwrap_or(false);
    if have.contains(wgpu::Features::DEPTH32FLOAT_STENCIL8) && !depth24 {
        want |= wgpu::Features::DEPTH32FLOAT_STENCIL8;
        DEPTH32F_STENCIL8.store(true, std::sync::atomic::Ordering::Relaxed);
    }
    report_status!("gxm depth: depth attachments are {:?}.", depth_format());
    let unclipped = want.contains(wgpu::Features::DEPTH_CLIP_CONTROL)
        && crate::knobs::var("VITASLOP_GXP_UNCLIPPED_DEPTH").map(|v| v == "1").unwrap_or(false);
    set_unclipped_depth(unclipped);
    set_block_family(if want.contains(wgpu::Features::TEXTURE_COMPRESSION_BC) {
        BlockFamily::Bc
    } else if want.contains(wgpu::Features::TEXTURE_COMPRESSION_ETC2) {
        BlockFamily::Etc2
    } else {
        BlockFamily::None
    });
    // >>> PUBLISHED AS A WARNING, ON EVERY ENGINE, BECAUSE IT DECIDES A WHOLE PIECE OF WORK.
    //
    // Which block formats an adapter takes is the single fact that says whether the compressed
    // upload is available at all, and getting it off a device has failed TWICE for different
    // reasons: the browser published it under the same `Report` id as the adapter summary, so
    // the 100 ms rate limiter dropped it; and once that was fixed it landed on a panel id the
    // diagnostics dump does not carry. It is not a per-frame diagnostic and it is not something
    // a knob should have to ask for - it is one line, once, in the section people paste.
    let f = adapter.features();
    let have: Vec<&str> = [
        (wgpu::Features::TEXTURE_COMPRESSION_BC, "bc"),
        (wgpu::Features::TEXTURE_COMPRESSION_ETC2, "etc2"),
        (wgpu::Features::TEXTURE_COMPRESSION_ASTC, "astc"),
    ]
    .iter()
    .filter(|(bit, _)| f.contains(*bit))
    .map(|(_, name)| *name)
    .collect();
    report_status!(
        "gxm textures: adapter compressed-texture support: {}. This build implements `bc` \
         (passthrough of the guest's own blocks, plus transcode) and `etc2` (transcode only, \
         4 bpp opaque / 8 bpp with alpha); taking {} for this run. A format in neither family is \
         decoded to RGBA8 at 4-8x its size.",
        if have.is_empty() { "NONE".to_string() } else { have.join(", ") },
        match block_family() {
            BlockFamily::Bc => "BC",
            BlockFamily::Etc2 => "ETC2",
            BlockFamily::None => "NOTHING",
        }
    );
    want
}

/// Whether this renderer is running on a WebGPU COMPATIBILITY-MODE adapter.
///
/// # Not a performance tier - a different validation regime
/// Compat mode is what a device gets when the full one is unavailable, and on the target device
/// that is not hypothetical: Chrome blocklisted the Imagination 25.1 driver
/// (crbug.com/520126488) and left only the compatibility adapter standing. Two of its rules bite
/// this renderer directly:
///
/// * a texture may NOT carry a view of a different format, so the sRGB twin this code declares
///   on every render target makes the target itself fail to create; and
/// * `textureLoad` is refused on a depth texture.
///
/// Neither degrades gracefully. The first produces an INVALID texture, and every view, bind
/// group, pass and submit built on it is invalid in turn - which reaches the screen as black,
/// with the cause thousands of validation errors earlier. MEASURED on the device: 5 failed
/// targets, 4,776 invalid bind groups, 1,173 invalid render passes, one black frame.
static COMPAT_MODE: std::sync::atomic::AtomicBool = std::sync::atomic::AtomicBool::new(false);

/// Record that the adapter in use is a compatibility-mode one. Set by the host before any
/// texture is created.
pub fn set_compat_mode(yes: bool) {
    COMPAT_MODE.store(yes, std::sync::atomic::Ordering::Relaxed);
}

/// Whether the adapter in use is a compatibility-mode one. See [`set_compat_mode`].
pub fn compat_mode() -> bool {
    COMPAT_MODE.load(std::sync::atomic::Ordering::Relaxed)
}

/// Which block family the GPU this process is using accepts.
/// `0` unknown, `1` BC, `2` none, `3` ETC2.
static BLOCK_COMPRESSION: std::sync::atomic::AtomicU8 = std::sync::atomic::AtomicU8::new(0);

/// Record what the adapter can take, so the CPU side does not prepare work the GPU cannot use.
///
/// Set by [`wanted_features`] at device creation, which is before any texture is decoded.
pub fn set_block_family(f: BlockFamily) {
    let v = match f {
        BlockFamily::Bc => 1,
        BlockFamily::None => 2,
        BlockFamily::Etc2 => 3,
    };
    BLOCK_COMPRESSION.store(v, std::sync::atomic::Ordering::Relaxed);
}

/// Whether this device took `depth-clip-control`, so a pipeline may ask for `unclipped_depth`.
///
/// PowerVR CLAMPS depth where an immediate-mode rasteriser CLIPS: a primitive that runs past the
/// near or far plane is drawn AT that plane rather than thrown away. WebGPU offers that only
/// through this optional feature, and the clip fixup used to imitate it in the vertex stage
/// instead - which is what [`ZFix::Viewport`] documents and what a per-vertex clamp cannot do
/// (see there).
static UNCLIPPED_DEPTH: std::sync::atomic::AtomicBool = std::sync::atomic::AtomicBool::new(false);

/// Record whether the device took `depth-clip-control`. Set by [`wanted_features`], which runs
/// before any pipeline is built.
pub fn set_unclipped_depth(yes: bool) {
    UNCLIPPED_DEPTH.store(yes, std::sync::atomic::Ordering::Relaxed);
}

/// Whether a pipeline built now may set `unclipped_depth`. See [`set_unclipped_depth`].
pub fn unclipped_depth_available() -> bool {
    UNCLIPPED_DEPTH.load(std::sync::atomic::Ordering::Relaxed)
}

/// Whether the device took `depth32float-stencil8`. Set by [`wanted_features`], before any
/// depth attachment exists. See [`depth_format`].
#[cfg(feature = "gpu")]
static DEPTH32F_STENCIL8: std::sync::atomic::AtomicBool = std::sync::atomic::AtomicBool::new(false);

/// `VITASLOP_GXP_NO_STENCIL=1`: the arm back to a depth-only attachment (`Depth32Float`) with
/// no stencil at all - what every pass had before stencil was modelled.
pub fn stencil_disabled() -> bool {
    use std::sync::OnceLock;
    static OFF: OnceLock<bool> = OnceLock::new();
    *OFF.get_or_init(|| crate::knobs::var("VITASLOP_GXP_NO_STENCIL").map(|v| v == "1").unwrap_or(false))
}

/// The depth(-stencil) format of every depth attachment this renderer, and any host handing it
/// one, creates.
///
/// >>> IT CARRIES STENCIL, because titles draw with it. A football title builds its whole HUD
/// out of stencil masks: `NEVER` quads whose fail op REPLACEs a bit into the stencil, then
/// dozens of `EQUAL ref=1` draws that may only paint inside the cut-out. With no stencil every
/// one of those painted its full quad - the white team-name boxes over the scoreboard, the white
/// play-call panel. `Depth32FloatStencil8` keeps the depth exactly as it was when the device
/// offers it; `Depth24PlusStencil8` is core WebGPU and is the fallback.
#[cfg(feature = "gpu")]
pub fn depth_format() -> wgpu::TextureFormat {
    if stencil_disabled() {
        wgpu::TextureFormat::Depth32Float
    } else if DEPTH32F_STENCIL8.load(std::sync::atomic::Ordering::Relaxed) {
        wgpu::TextureFormat::Depth32FloatStencil8
    } else {
        wgpu::TextureFormat::Depth24PlusStencil8
    }
}

/// The block family to encode for.
///
/// UNKNOWN answers `Bc` for the same reason [`block_compression_available`] answers `true`: the
/// caller is a cache-filling path where guessing wrong costs one texture's bytes, and the upload
/// asks the real device before it uses any of it.
pub fn block_family() -> BlockFamily {
    match BLOCK_COMPRESSION.load(std::sync::atomic::Ordering::Relaxed) {
        2 => BlockFamily::None,
        3 => BlockFamily::Etc2,
        _ => BlockFamily::Bc,
    }
}

/// Whether it is worth building a [`CompressedUpload`] at all.
///
/// # This exists so a device WITHOUT BC pays nothing for a feature it cannot use
/// The compressed source is built on the texture decode path, in the runtime, which has no
/// device and cannot ask. Without this it would de-swizzle and copy every BC texture's blocks
/// on every engine, and hold them in the decode cache for the life of the run, purely so the
/// uploader could decline them - and the engine that would pay for that is a phone whose GPU
/// exposes ASTC and ETC2 but not BC, i.e. exactly the device this whole line of work is for.
///
/// UNKNOWN reads as available: the only caller is a cache-filling path where being wrong costs
/// one texture's worth of bytes, while being wrong the other way would permanently disable the
/// passthrough on any engine that sets this late. The upload itself never trusts this - it asks
/// the real device (see `GxpLive::bc_supported`).
pub fn block_compression_available() -> bool {
    BLOCK_COMPRESSION.load(std::sync::atomic::Ordering::Relaxed) != 2
}

/// The last complete frame's uploaded texture working set, in bytes. `0` before the first frame.
static LAST_WORKING_SET: std::sync::atomic::AtomicUsize = std::sync::atomic::AtomicUsize::new(0);

/// Is the texture working set close enough to its budget that spending CPU to shrink it is still
/// worth something?
///
/// # Compression is a means, and this is the end it serves
/// Re-encoding a texture exists to make a frame's textures FIT. Once they fit with room to spare,
/// further compression buys memory nobody needs at a CPU cost this device cannot afford - and it
/// is the one resource it has least of. MEASURED on the target phone after the PVRTC transcode
/// landed: the race frame's working set fell from **256 MB against a 256 MB budget** to **82 MB**,
/// at which point the remaining BC textures could have been left alone entirely.
///
/// Two-thirds rather than the whole budget, because the answer has to be given BEFORE the work:
/// a texture skipped at 90% would be wanted again the moment the next screen loaded, and a
/// control that only reacts at the limit oscillates across it.
/// The most texture a game can have resident on the console, in MiB, and this renderer's default
/// cache budget: `ScePhyMemPartGame` (256) + the +109 MiB mode (109) + `ScePhyMemPartGameCdram`
/// (112). Spelled as its three partitions so it cannot quietly become a round number somebody
/// liked - the previous value was fitted to one title's MENU and cost a race 83% re-decodes on
/// the target device. See `gxm::GxpLive::views` for the full derivation.
///
/// >>> AT FILE SCOPE ON PURPOSE. It used to live inside `mod gxm`, which is `#[cfg(feature =
/// "gpu")]`, so `texture_budget_pressure` - which is not - could not reach it and carried its own
/// > > > `unwrap_or(256)` instead. That is how the two disagreed by 221 MiB while a comment said they
/// > > > could not.
pub(crate) const GAME_RESIDENT_CEILING_MB: usize = 256 + 109 + 112;

/// The texture-cache budget in bytes: [`GAME_RESIDENT_CEILING_MB`] unless
/// `VITASLOP_TEX_CACHE_MB` overrides it. Read once.
///
/// The ONE reader of that knob. Every consumer - the uploader's eviction, the pressure signal
/// above, the over-budget report - must call this rather than re-derive it.
pub(crate) fn tex_cache_budget_bytes() -> usize {
    use std::sync::OnceLock;
    static CELL: OnceLock<usize> = OnceLock::new();
    let base = *CELL.get_or_init(|| {
        crate::knobs::var("VITASLOP_TEX_CACHE_MB")
            .ok()
            .and_then(|s| s.trim().parse::<usize>().ok())
            .unwrap_or(GAME_RESIDENT_CEILING_MB)
            * 1024
            * 1024
    });
    // >>> AND SCALED TO THE DEVICE. The default above is the CONSOLE's resident ceiling, which
    // is the right number for what a title can ask for and says nothing about what the machine
    // running the emulator has. MEASURED on a 48,000-frame browser session: this cache alone held
    // **476 MB** against a one-frame working set of 106 MB, inside a renderer process at 1.53 GB.
    // A desktop reports the specification's 8 GB cap and is unchanged; a 4 GB phone halves it.
    // See [`crate::knobs::memory_scale`], and note that per-entry eviction (which this cache
    // already had, and which the rest now have too) is what makes a smaller budget cost
    // re-decodes of the coldest entries instead of a cliff.
    crate::knobs::scale_budget(base)
}

/// How many bytes of GPU texture the view cache may RETAIN, as opposed to how many the
/// re-encode gate is allowed to believe are available. `VITASLOP_TEX_RETAIN_MB` overrides it.
///
/// # >>> THE BUDGET AND THE RETENTION BOUND WERE ONE NUMBER, AND THEY ARE TWO QUESTIONS
///
/// [`tex_cache_budget_bytes`] is the CONSOLE's resident ceiling, and it has a second job that
/// has nothing to do with retention: [`texture_budget_pressure`] reads it to decide whether a
/// BC texture is re-encoded to ETC2, which is a SECOND lossy step over the guest's own
/// compression. Lowering that number to bound retention therefore buys memory with PICTURE -
/// silently, on the one engine that cannot be looked at - which is why the deliberate decision
/// not to scale budgets from `deviceMemory` was the right one and is left alone.
///
/// Retention is a different question with no quality in it at all. Every key in this cache is a
/// content fingerprint, so shedding an entry costs a re-upload and can never change a texel
/// [[vitaslop-never-trade-quality]]. Splitting the two lets the retention bound be set from what
/// a frame actually needs while the re-encode gate keeps reading the full budget and keeps
/// declaring "there is room, do not spend quality".
///
/// MEASURED on the user's device, two gameplay windows ~1,900 and ~2,100 frames in: `texture
/// views 296 (296 stamped, 156 dead, 140 slots)` holding **343 MB**, against a ONE-FRAME
/// working set of **14 MB** and a 477 MB budget the eviction pass had therefore never reached.
/// More than half the entries were superseded - contents the guest had already overwritten -
/// and nothing could ever shed them, because the only thing that sheds anything here is the
/// budget.
///
/// The bound is a FLOOR-PROTECTED one: the use site takes `max(this, views_frame_high)`, so a
/// title whose one frame genuinely needs more than this keeps every byte of it and this number
/// bounds only the cross-frame TAIL. That is what makes a figure well under the console ceiling
/// safe rather than a cliff - see `GxpLive::views_frame_high` for the measurement that killed
/// the wholesale clear.
pub(crate) fn tex_retain_budget_bytes() -> usize {
    use std::sync::OnceLock;
    static CELL: OnceLock<usize> = OnceLock::new();
    let base = *CELL.get_or_init(|| {
        crate::knobs::var("VITASLOP_TEX_RETAIN_MB")
            .ok()
            .and_then(|s| s.trim().parse::<usize>().ok())
            .unwrap_or(TEX_RETAIN_MB)
            * 1024
            * 1024
    });
    // Scaled like every other budget, and never ABOVE the texture budget itself: an operator who
    // tightens `VITASLOP_TEX_CACHE_MB` is tightening the whole cache, and a retention bound that
    // sat above it would quietly make that knob a no-op for retention.
    crate::knobs::scale_budget(base).min(tex_cache_budget_bytes())
}

/// >>> A TEXTURE IS KEPT WHILE THE GUEST STILL HOLDS ITS BYTES - THE VITA HAS NO TEXTURE CACHE.
///
/// The console's GPU samples straight out of guest memory, so a texture is "resident" for
/// exactly as long as the title keeps those bytes in place, and the title's own memory is what
/// bounds its texture set. The faithful retention rule is the same: an entry goes when the guest
/// OVERWRITES it (superseded - `view_dead`), and otherwise stays, with the texture BUDGET as the
/// only byte ceiling (past it, anything this frame did not use goes, oldest first). A copy whose
/// memory the guest reused for something else can never match again (keys are content
/// fingerprints) and is simply the oldest thing there when the budget is reached.
///
/// This returns the number of frames an entry idle-but-live is protected from the 96 MB
/// RETENTION bound: unset = 600 (see below); `VITASLOP_TEX_RECENT_FRAMES=<n>` = `n` frames, and
/// `18446744073709551615` restores the old "forever" (the rule above); `0` = the oldest rule,
/// where anything this frame did not use was a candidate.
///
/// # >>> 600 FRAMES, NOT FOREVER: "FOREVER" MEANT "UNTIL THE 477 MB BUDGET"
/// With idle entries protected for ever the retention bound shed only superseded ones, so the
/// view cache sat at the texture BUDGET, not at the working set. MEASURED on a first-person
/// shooter's level in the desktop browser, one build, this knob the only difference (with the
/// decode and snapshot caches at 96 MB in both): `recompiler views` 400 -> 176 MB, GPU census
/// textures 473 -> 250 MB live, `RE-uploaded after eviction` 0.0 a frame in the steady window,
/// speed 97% -> 98%. On the phone that is unified memory the tab shares. 600 frames is 10-20 s:
/// the golf title's two cameras (below) alternate well inside it, so its thrash is not
/// reintroduced; what goes is what has not been drawn for that long, and an eviction never
/// changes a texel.
///
/// The retention bound is floored at the largest ONE-frame working set, and that is the wrong
/// floor for a title that cycles views: MEASURED on the user's phone in a golf title
/// (2026-09-27), `recompiler views 138 MB ... ONE FRAME needs 138 MB, learned floor 139 MB`
/// against the 96 MB bound, with **3.8 textures RE-uploaded after eviction per frame** on
/// average and **108 in the worst frame** (16.5 MB, 90 of them re-encoded on the GPU, 42
/// eviction passes) - the shot camera and the player camera each fit, their union did not,
/// and each switch threw out what the other was about to use. The desktop measurement that set
/// the bound read 0.25 a frame.
///
/// An eviction here never changes a texel (every key is a content fingerprint), so this trades
/// only memory, and only up to the texture budget: past that the old oldest-first rule applies
/// to everything. SUPERSEDED entries stay evictable at any age, which is what the bound was
/// introduced for (343 MB retained, 14 MB frame, more than half superseded).
#[cfg(feature = "gpu")]
pub(crate) fn tex_recent_frames() -> u64 {
    use std::sync::OnceLock;
    static CELL: OnceLock<u64> = OnceLock::new();
    *CELL.get_or_init(|| {
        crate::knobs::var("VITASLOP_TEX_RECENT_FRAMES")
            .ok()
            .and_then(|s| s.trim().parse::<u64>().ok())
            .unwrap_or(TEX_RECENT_FRAMES)
    })
}

/// The default for [`tex_recent_frames`].
#[cfg(feature = "gpu")]
const TEX_RECENT_FRAMES: u64 = 600;

/// The default retention bound, in MiB. See [`tex_retain_budget_bytes`].
///
/// Chosen from the working set rather than from the device, and it is a FLOOR on the tail rather
/// than a ceiling on the cache - the use site never takes it below `views_frame_high`, so a
/// title whose frame is larger than this is simply not bounded by it. MEASURED on three titles,
/// each run twice off one build with this knob as the only difference:
///
/// | title, desktop headless, encode budget OFF | retained at 477 | retained at 96 | one frame | shots |
/// |---|---|---|---|---|
/// | a fighting title, 3,000 frames | 276 MB | **95 MB** | 8 MB | 12 of 12 identical |
/// | a golf title, 4,700 frames | 476 MB | **139 MB** | 130 MB | 12 of 12 identical |
/// | a racing title, 9,400 frames | 476 MB | **234 MB** | 162 MB | 12 of 12 identical |
///
/// The last two show the floor doing its work: their bound resolves to 139 and 235, not 96, and
/// the only title actually cut to this number is the one whose frame is small. The COST is the
/// re-uploads the shed entries pay for when they come back: 325 -> 1,156 GPU texture expansions
/// over the golf run (0.07 -> 0.25 a frame) and 292 -> 416 over the racing one (0.03 -> 0.04).
const TEX_RETAIN_MB: usize = 96;

/// `VITASLOP_RTT_BG_CACHE=0` restores the OLD behaviour: a sampler bind group naming a render
/// target is rebuilt every frame instead of being keyed by `rtt_epoch`. An A/B ARM, so it is
/// VALUE-sensitive rather than presence-only - a knob used as an arm that reads `NAME=0` as ON
/// has already cost this project a whole measurement
/// ([[vitaslop-knob-is-the-gate-not-the-level]]).
/// `VITASLOP_DRAW_RANGE=<lo>-<hi>`: encode only draws `lo..=hi` of every pass. See the use
/// site in `encode_chain` for what it is for and why it is not a mode.
#[cfg(feature = "gpu")]
pub(crate) fn draw_range() -> Option<(usize, usize)> {
    use std::sync::OnceLock;
    static CELL: OnceLock<Option<(usize, usize)>> = OnceLock::new();
    *CELL.get_or_init(|| {
        let v = crate::knobs::var("VITASLOP_DRAW_RANGE").ok()?;
        let (a, b) = v.split_once('-')?;
        Some((a.trim().parse().ok()?, b.trim().parse().ok()?))
    })
}

/// Diagnostic (`VITASLOP_GXP_BIND_TRACE=1`): say, once per pair and plan shape, WHICH view each
/// of a draw's sampler units is actually given - a rendered target, a depth buffer, a cube, a
/// decode of the guest's own bytes, or a fallback. See the trace site in the sampler bind group
/// builder for why the guest's descriptor does not answer this.
#[cfg(feature = "gpu")]
pub(crate) fn bind_trace() -> bool {
    use std::sync::OnceLock;
    static CELL: OnceLock<bool> = OnceLock::new();
    *CELL.get_or_init(|| crate::knobs::flag("VITASLOP_GXP_BIND_TRACE"))
}

/// `VITASLOP_GXP_ELIDE_NOOP=0` - the negative control for dropping the colour work of a draw
/// that provably writes the DESTINATION BACK UNCHANGED. Default ON.
///
/// The elision is BIT-IDENTICAL by construction, not by measurement: the colour a dropped draw
/// would have written is the colour already there, proven from the draw's own uniform bytes
/// (`draw_colour_is_noop`). What the control is for is the OTHER half - a draw also writes
/// DEPTH, and this renderer keeps that half rather than assuming nobody reads it, so the arm
/// exists to price the saving and to answer "is the picture the same" without an argument.
#[cfg(feature = "gpu")]
pub(crate) fn elide_noop_draws() -> bool {
    use std::sync::OnceLock;
    static CELL: OnceLock<bool> = OnceLock::new();
    *CELL.get_or_init(|| crate::knobs::var("VITASLOP_GXP_ELIDE_NOOP").map(|v| v.trim() != "0").unwrap_or(true))
}

#[cfg(feature = "gpu")]
pub(crate) fn rtt_bg_cache() -> bool {
    use std::sync::OnceLock;
    static CELL: OnceLock<bool> = OnceLock::new();
    *CELL.get_or_init(|| crate::knobs::var("VITASLOP_RTT_BG_CACHE").map(|v| v.trim() != "0").unwrap_or(true))
}

/// Whether a sampler over a GAMMA-CORRECT render target decodes by the TEXTURE's own gamma mode
/// (the default) rather than always - see the `Bind::RenderedLinear` arm of `make_sampler_bg`.
/// `VITASLOP_RTT_GAMMA_BY_TEXTURE=0` is the arm back to always decoding.
#[cfg(feature = "gpu")]
pub(crate) fn rtt_gamma_by_texture() -> bool {
    use std::sync::OnceLock;
    static CELL: OnceLock<bool> = OnceLock::new();
    *CELL.get_or_init(|| crate::knobs::var("VITASLOP_RTT_GAMMA_BY_TEXTURE").map(|v| v.trim() != "0").unwrap_or(true))
}
/// `VITASLOP_GXP_PASS_SPLIT_EVERY=<n>` cuts a render pass every `n` draws, with no shader
/// changed and no destination read added.
///
/// The control the destination-blend A/B never had. `VITASLOP_GXP_DEST_BLEND=0` changes TWO
/// things at once - the ALU form the recompiler then emits, and the render-pass CUT that form
/// forces - so a picture that goes wrong under it says nothing about which. This brings the cut
/// alone. **MEASURED: 49 forced splits, attachment copies included, over one title's 145-draw
/// pass are BIT-IDENTICAL to none**, which is what retired "the split path is itself broken".
///
/// Cached: this is asked per PASS, and reading an unset environment variable on Windows is not
/// free - the same reason every other per-draw knob in this file sits behind a `OnceLock`.
/// `VITASLOP_GPU_TIME_ALL=1`: the GPU TIME report lists every pass in frame order, not the
/// costliest five. Cached: asked per report.
#[cfg(feature = "gpu")]
pub(crate) fn gpu_time_all_passes() -> bool {
    use std::sync::OnceLock;
    static CELL: OnceLock<bool> = OnceLock::new();
    *CELL.get_or_init(|| crate::knobs::flag("VITASLOP_GPU_TIME_ALL"))
}

#[cfg(feature = "gpu")]
pub(crate) fn forced_pass_split() -> Option<usize> {
    use std::sync::OnceLock;
    static N: OnceLock<Option<usize>> = OnceLock::new();
    *N.get_or_init(|| {
        crate::knobs::var("VITASLOP_GXP_PASS_SPLIT_EVERY")
            .ok()
            .and_then(|v| v.trim().parse::<usize>().ok())
            .filter(|n| *n > 0)
    })
}


/// `VITASLOP_GXM_DEST_SPLIT_AB=<n>`: alternate the destination-colour pass SPLIT on and off every
/// `n` frames, and report the GPU time each arm measured. Off (0/unset) is the normal renderer.
///
/// # >>> WHY AN IN-RUN ALTERNATION AND NOT TWO RUNS
/// The question is what ~230 render-pass splits a frame cost a TILING GPU, and the only machine
/// that can answer it is a phone. But a phone's GPU time swings with what is on screen - a
/// close-up and a wide field are not the same frame - so two runs, or a run against a diagnostic
/// captured on some earlier day, compare SCENES and not arms. There is no baseline to compare a
/// new capture against, because past captures were taken when something looked wrong, not at
/// random [[vitaslop-browser-ab-needs-a-negative-control]].
///
/// Alternating INSIDE one run makes the two arms sample the same gameplay a few frames apart,
/// which is the closest thing to holding the scene fixed that a live title allows. The user
/// plays normally for a minute and one diagnostic carries both halves.
///
/// # And why this is not `VITASLOP_GXP_DEST=0`
/// That knob is the wrong instrument for this question twice over. It is a `OnceLock` pushed into
/// the recompiler, so it cannot be flipped mid-run at all; and it changes the SHADER as well as
/// the cut, which is the exact confusion that made "the split path is itself broken" a wrong
/// finding for months [[vitaslop-the-render-pass-split-is-inert]]. This changes ONE thing: the
/// pass is not cut and the attachment is not copied. Same shaders, same draws, same order.
///
/// The OFF arm's picture is WRONG - a destination reader samples whatever the dest texture last
/// held - and that is expected and temporary: it is a stopwatch, not a fix, and the flicker
/// between arms is what tells the user it is armed.
/// `VITASLOP_GXM_ARENA_FLOOR_KB=<n>`: the smallest arena buffer the renderer will create.
/// **Default 4 KB, which is what this always did.**
///
/// # >>> RAISING IT IS MEASURED AND NULL ON THIS MACHINE - the arm is kept for the DEVICE
/// The mechanism behind it is real and is recorded on `GxmRenderer::cap_for`: a buffer's FIRST
/// write costs hundreds of milliseconds where the same bytes written again to the same buffer
/// cost a tenth of one. The inference was that an arena climbing a ladder of ever-bigger
/// destinations pays that price at every rung, so starting high would skip the ladder.
///
/// A paired run of the browser rig says it does not, HERE: `512` gives 4 stalls / 1,630 ms
/// against `4` giving 3 stalls / 1,489 ms, which is a null against a same-code spread that has
/// read 114-305 ms on a single stall. The reason is in the stall lines themselves - this rig's
/// stalls are FIRST-SIGHT creations (a pass ordinal allocating its three arenas the first time
/// that pass is ever encoded), and no floor can remove a buffer that has to exist.
///
/// It is not deleted because the DEVICE's stalls look different: the user's capture shows them
/// with `4 new buffers` in steady gameplay, which is the growth shape this would remove, and
/// the phone is the only instrument that can say. Raising it costs GPU memory a small pass does
/// not use - three buffers per pass ordinal, so ~1.5 MB per ordinal at 512 - which is why the
/// DEFAULT stays where the measurement is.
#[cfg(feature = "gpu")]
pub(crate) fn arena_floor_bytes() -> u64 {
    use std::sync::OnceLock;
    static N: OnceLock<u64> = OnceLock::new();
    *N.get_or_init(|| {
        crate::knobs::var("VITASLOP_GXM_ARENA_FLOOR_KB")
            .ok()
            .and_then(|v| v.trim().parse::<u64>().ok())
            .filter(|n| *n > 0)
            .unwrap_or(4)
            .saturating_mul(1024)
            .max(4096)
    })
}

/// One staging-belt chunk, in bytes - see `GxmRenderer::gxp_staging`.
///
/// Above the largest arena write MEASURED on the user's device (2,315 KB) with room on top, so
/// an ordinary frame's three arena writes come out of one chunk and the belt allocates nothing
/// in the steady state. A write larger than this still succeeds; the belt just gives it a
/// buffer of its own, which is the case worth avoiding rather than the case worth forbidding.
#[cfg(feature = "gpu")]
pub(crate) const GXP_STAGING_CHUNK: u64 = 4 * 1024 * 1024;

/// Whether a frame whose last scene is not a flipped (display) buffer shows its last display
/// scene - or, with none, the flipped buffer's kept image - instead of that last scene
/// (default) - see `GxmRenderer::display_choice`. `VITASLOP_GXM_HOLD_FLIP=0` is the old rule.
#[cfg(feature = "gpu")]
pub(crate) fn hold_flip_enabled() -> bool {
    use std::sync::OnceLock;
    static CELL: OnceLock<bool> = OnceLock::new();
    *CELL.get_or_init(|| crate::knobs::var("VITASLOP_GXM_HOLD_FLIP").map(|v| v.trim() != "0").unwrap_or(true))
}

/// The 2026-09-28 memory bounds, each with its arm back (`=0`): `VITASLOP_SUBRECT_POOL`
/// (reuse sub-rectangle copy textures), `VITASLOP_RESIDENT_COMPACT_EARLY` (compact a resident
/// heap below its budget instead of doubling), `VITASLOP_ARENA_TRIM` (drop idle trailing pass
/// arenas), `VITASLOP_STAGING_TRIM` (destroy surplus free staging chunks).
#[cfg(feature = "gpu")]
pub(crate) fn memory_bound_on(name: &'static str) -> bool {
    use std::collections::HashMap;
    use std::sync::{Mutex, OnceLock};
    static CELL: OnceLock<Mutex<HashMap<&'static str, bool>>> = OnceLock::new();
    let mut m = CELL.get_or_init(Default::default).lock().unwrap_or_else(|e| e.into_inner());
    *m.entry(name).or_insert_with(|| crate::knobs::var(name).map(|v| v.trim() != "0").unwrap_or(true))
}

/// Free staging-belt chunks kept across the periodic trim - see the recall site in
/// `encode_chain`. A steady frame uses one or two; the rest of the reserve absorbs a burst.
#[cfg(feature = "gpu")]
const STAGING_KEEP_CHUNKS: usize = 6;

/// Whether the arena upload goes through the STAGING BELT (default) rather than
/// `queue.write_buffer`. `VITASLOP_GXM_STAGING=0` is the arm back, so the two paths can be
/// compared from ONE build.
///
/// VALUE-sensitive and read through the knob seam, not the environment: the browser is where
/// the old path costs hundreds of milliseconds and the browser has no environment.
#[cfg(feature = "gpu")]
pub(crate) fn gxm_staging_on() -> bool {
    use std::sync::OnceLock;
    static ON: OnceLock<bool> = OnceLock::new();
    *ON.get_or_init(|| !matches!(crate::knobs::var("VITASLOP_GXM_STAGING").as_deref(), Ok("0")))
}

/// `VITASLOP_GXP_ARENA_RING=<n>`: how many COPIES of each render pass's recompiled-path
/// vertex/index/uniform arenas are kept and rotated, one per frame. **Default 1 - the single
/// buffer per pass ordinal this has always used.** `n > 1` rotates.
///
/// # >>> A FALSIFIER, NOT A FIX. THE HYPOTHESIS BEHIND IT IS REFUTED; READ THIS BEFORE RAISING IT
/// It was built for one measurement: on the user's phone, a gameplay window of a retail
/// fighting title, a single `queue.write_buffer` of 298 KB BLOCKED for 309.6 ms - the whole of
/// that frame's 312 ms encode - while the GPU's own timestamp query read 13.1 ms against a
/// 34.6 ms period. The reading was "a write aimed at a buffer a pending submit still reads can
/// only be staged, so rotate the destination".
///
/// THAT CANNOT BE WHAT BLOCKS. `GPUQueue.writeBuffer` is specified as a QUEUE operation: the
/// data is taken at the call and the copy is ordered on the queue timeline, so the
/// destination's in-flight state does not gate it. Whatever blocks is upstream of the
/// destination - the staging allocator, or the wire to the GPU process - and neither is
/// addressed by having more destinations.
///
/// AND THE EVENT WAS OVER-RANKED, which its own control says: a second window of the SAME
/// title, 1,926 frames, worst write of the whole run 3.0 ms for 148 KB. The 310 ms block
/// happened ONCE, in the run that was in PORTRAIT after the device was rotated and that
/// hard-paused twice for 13.8 s. A surface reconfigure explains a one-off far better than
/// gameplay does.
///
/// It is kept because it is built and PROVEN PICTURE-NEUTRAL (24 of 24 shots bit-identical
/// across `1` and `3`, two titles, one build twice), and costs nothing at the default - so if
/// a device capture ever does implicate the destination, the arm is one knob away. What it
/// would cost: `n` times the arena bytes and `n` times the uniform bind groups, since each ring
/// entry is its own cache key (a group is BUILT once per (pass, phase), then reused every `n`
/// frames rather than rebuilt). A frame that opens two hundred passes takes a slot per ordinal
/// and the ring multiplies that, which is the reason not to raise the default on a guess.
///
/// >>> LAZY ZERO-INITIALISATION IS REFUTED AS THE CAUSE OF THE BLOCKING WRITE, and the arm that
/// tested it is gone rather than left behind as a knob that does nothing.
///
/// The theory was good: WebGPU guarantees a new buffer reads as zero, implementations defer that
/// until first use, and every stall here IS the first write into a buffer created that frame -
/// which would explain a 128 KB write costing 138 ms. Zeroing every new buffer with
/// `clear_buffer` on the queue, so the write that follows is a plain copy, changed nothing:
/// 791/508/191 ms against 653/286/138 for the same three stalls, inside a run-to-run spread that
/// has read 763/259/305 and 616/293/131 on identical code.
#[cfg(feature = "gpu")]
pub(crate) fn gxp_arena_ring() -> usize {
    use std::sync::OnceLock;
    static N: OnceLock<usize> = OnceLock::new();
    *N.get_or_init(|| {
        crate::knobs::var("VITASLOP_GXP_ARENA_RING")
            .ok()
            .and_then(|v| v.trim().parse::<usize>().ok())
            .filter(|n| *n > 0)
            .unwrap_or(1)
            .min(8)
    })
}

#[cfg(feature = "gpu")]
pub(crate) fn dest_split_ab() -> Option<u32> {
    use std::sync::OnceLock;
    static N: OnceLock<Option<u32>> = OnceLock::new();
    *N.get_or_init(|| {
        crate::knobs::var("VITASLOP_GXM_DEST_SPLIT_AB")
            .ok()
            .and_then(|v| v.trim().parse::<u32>().ok())
            .filter(|n| *n > 0)
    })
}

/// Frames encoded since the run began, for [`dest_split_ab`]. Bumped once per frame by the
/// renderer, so every pass in one frame agrees on which arm it is in - a counter bumped per PASS
/// would split a single frame across both arms and average the answer away.
#[cfg(feature = "gpu")]
pub(crate) static DEST_AB_FRAME: std::sync::atomic::AtomicU32 = std::sync::atomic::AtomicU32::new(0);

/// >>> A TEXTURE'S ETC2 ENCODE IS SPREAD OVER LATER FRAMES, AND THE FRAME THAT BINDS IT DRAWS
/// >>> THE EXACT DECODE MEANWHILE - see `texenc::Transcoder::run_deferred`.
///
/// MEASURED on the phone (a fighting title, runner jobs 078/085): screens that bind dozens of new textures
/// spent 240-955 ms of GPU in ONE frame encoding them, freezing the display for up to 1.4 s.
/// `VITASLOP_TEX_ENCODE_DEFER=0` is the arm back: every encode inline, in the frame that binds it.
#[cfg(feature = "gpu")]
pub(crate) fn tex_encode_defer() -> bool {
    static ON: std::sync::OnceLock<bool> = std::sync::OnceLock::new();
    *ON.get_or_init(|| crate::knobs::var("VITASLOP_TEX_ENCODE_DEFER").ok().as_deref() != Some("0"))
}

/// Block WORK UNITS the deferred encoder spends per frame (`VITASLOP_TEX_ENCODE_DEFER_UNITS`):
/// an RGB block is one, an RGBA block two. MEASURED on the phone (`texenc-bench`, job 084): about
/// 0.8 us of GPU per RGB block in a big dispatch, so the default is ~5 ms of the frame; a
/// texture's small levels cost a dispatch's latency (1-2 ms) whatever the budget.
#[cfg(feature = "gpu")]
pub(crate) fn tex_encode_defer_units() -> u64 {
    static N: std::sync::OnceLock<u64> = std::sync::OnceLock::new();
    *N.get_or_init(|| {
        crate::knobs::var("VITASLOP_TEX_ENCODE_DEFER_UNITS")
            .ok()
            .and_then(|v| v.trim().parse().ok())
            .filter(|n: &u64| *n > 0)
            .unwrap_or(6144)
    })
}

/// Whether `warm_pipelines` predicts a never-encoded target's pipelines from its guest colour
/// format (`GxmRenderer::pass_shapes_by_format`). `VITASLOP_WARM_BY_FORMAT=0` is the arm back.
#[cfg(feature = "gpu")]
pub(crate) fn warm_by_format() -> bool {
    static ON: std::sync::OnceLock<bool> = std::sync::OnceLock::new();
    *ON.get_or_init(|| crate::knobs::var("VITASLOP_WARM_BY_FORMAT").ok().as_deref() != Some("0"))
}

/// Whether `warm_pipelines` guesses the shape of a target with no recorded one from the
/// offscreen arm's rule (`GxmRenderer::offscreen_shape`) instead of leaving its pipelines to a
/// synchronous build in the encode. `VITASLOP_WARM_NEW_TARGETS=0` is the arm back.
#[cfg(feature = "gpu")]
pub(crate) fn warm_new_targets() -> bool {
    static ON: std::sync::OnceLock<bool> = std::sync::OnceLock::new();
    *ON.get_or_init(|| crate::knobs::var("VITASLOP_WARM_NEW_TARGETS").ok().as_deref().map(str::trim) != Some("0"))
}

/// Deferred encodes swapped into the view cache, and finished after their entry had already
/// gone (evicted) - for the report beside `texenc::defer_stats`.
#[cfg(feature = "gpu")]
static DEFER_SWAPPED: std::sync::atomic::AtomicU64 = std::sync::atomic::AtomicU64::new(0);
#[cfg(feature = "gpu")]
static DEFER_ORPHANED: std::sync::atomic::AtomicU64 = std::sync::atomic::AtomicU64::new(0);

/// The guest frame the frontend is on, set once per frame (see [`set_frame_tag`]). This crate
/// has no scheduler of its own, and a GPU measurement that cannot be named by guest frame cannot
/// be lined up with the page's display-gap and slow-frame panels, which are.
static FRAME_TAG: std::sync::atomic::AtomicU64 = std::sync::atomic::AtomicU64::new(0);

/// Tell the renderer which guest frame it is drawing, for its reports. Diagnostic only.
pub fn set_frame_tag(frame: u64) {
    FRAME_TAG.store(frame, std::sync::atomic::Ordering::Relaxed);
}

#[cfg(feature = "gpu")]
pub(crate) fn frame_tag() -> u64 {
    FRAME_TAG.load(std::sync::atomic::Ordering::Relaxed)
}

/// A timestamp pair for a texture-transcode compute pass, if the frame being prepared is
/// measured - see `gxm::lent_ts_pair`. `None` outside a frame's texture preparation.
#[cfg(feature = "gpu")]
pub(crate) fn transcode_ts_pair(
    label: impl FnOnce() -> String,
) -> Option<(std::sync::Arc<wgpu::QuerySet>, u32)> {
    gxm::lent_ts_pair(label)
}

/// Whether THIS frame is in the arm that takes splits. `true` whenever the A/B is off, so the
/// ordinary path is unchanged.
#[cfg(feature = "gpu")]
pub(crate) fn dest_split_armed() -> bool {
    let Some(n) = dest_split_ab() else { return true };
    let frame = DEST_AB_FRAME.load(std::sync::atomic::Ordering::Relaxed);
    (frame / n).is_multiple_of(2)
}

/// `VITASLOP_GXM_DEST_COALESCE=0`: cut the pass and take a WHOLE destination copy in front of
/// every destination reader, as before. On by default: a reader whose footprint meets nothing
/// written since the last copy shares that copy, and each later copy refreshes only what was
/// written since - see `gxm::DestDirty`. The off arm is the identity check (both arms must be
/// bit-identical).
#[cfg(feature = "gpu")]
pub(crate) fn dest_coalesce_enabled() -> bool {
    use std::sync::OnceLock;
    static CELL: OnceLock<bool> = OnceLock::new();
    *CELL.get_or_init(|| crate::knobs::var("VITASLOP_GXM_DEST_COALESCE").map(|v| v.trim() != "0").unwrap_or(true))
}

/// `VITASLOP_GXM_DEST_GEOM=0`: bound the destination-copy coalescing by each draw's VIEWPORT
/// only, as before, instead of by where its own vertices project - see
/// `gxm::clip::draw_pixel_bounds`. The arm back; both arms must be bit-identical.
#[cfg(feature = "gpu")]
pub(crate) fn dest_geom_bounds_enabled() -> bool {
    use std::sync::OnceLock;
    static CELL: OnceLock<bool> = OnceLock::new();
    *CELL.get_or_init(|| crate::knobs::var("VITASLOP_GXM_DEST_GEOM").map(|v| v.trim() != "0").unwrap_or(true))
}

/// `VITASLOP_GXM_DEST_GEOM_GATE=0`: ask for vertex footprints in every pass, every frame, with
/// no CPU budget - the arm back from the per-pass profitability gate (`gxm::DestGeomGate`). The
/// gate only decides how much CPU is spent proving copies shareable; every arm is bit-identical.
#[cfg(feature = "gpu")]
pub(crate) fn dest_geom_gate_enabled() -> bool {
    use std::sync::OnceLock;
    static CELL: OnceLock<bool> = OnceLock::new();
    *CELL.get_or_init(|| crate::knobs::var("VITASLOP_GXM_DEST_GEOM_GATE").map(|v| v.trim() != "0").unwrap_or(true))
}

/// Diagnostic (`VITASLOP_GXP_VP_TRACE`): report the viewport rectangle every recompiled draw is
/// actually given, beside the guest viewport it was derived from. See the trace site in
/// `encode_pass`.
#[cfg(feature = "gpu")]
pub(crate) fn vp_trace() -> bool {
    use std::sync::OnceLock;
    static CELL: OnceLock<bool> = OnceLock::new();
    *CELL.get_or_init(|| crate::knobs::flag("VITASLOP_GXP_VP_TRACE"))
}

/// `VITASLOP_GXP_DEST=0` restores the pre-destination-read behaviour: a fragment program that
/// reads its output bank reads ZEROS instead of the framebuffer colour, and no render pass is
/// split. An A/B arm - VALUE-sensitive, on by default - because the split is the one cost this
/// feature has and a measurement of it needs both halves. With it off, a title that composites
/// through its output registers goes black. See [`GxmRenderer::dest_color`].
#[cfg(feature = "gpu")]
pub(crate) fn gxp_dest_enabled() -> bool {
    use std::sync::OnceLock;
    static CELL: OnceLock<bool> = OnceLock::new();
    *CELL.get_or_init(|| {
        let on = crate::knobs::var("VITASLOP_GXP_DEST").map(|v| v.trim() != "0").unwrap_or(true);
        // Pushed into the shader crate so the module does not DECLARE a destination texture the
        // pass will not supply - see `module::DEST_COLOR_READ` for the third behaviour that
        // gating only the copy produced. The knob is READ on every build, `gpu` or not, so the
        // KNOBS.md index and the browser override table stay honest; only the push needs the
        // recompiler, which is a `gpu`-only dependency.
        #[cfg(feature = "gpu")]
        vitaslop_gxp_shader::module::set_dest_color_read(on);
        on
    })
}

/// `VITASLOP_GXP_DEST_BLEND=0` stops the recompiler recovering an ALU blend over the destination
/// as PIPELINE state, so every such program keeps its ALU form and pays a render-pass split. An
/// A/B arm - VALUE-sensitive, on by default - and the one that proves a lowering equivalent: the
/// two arms must produce the same picture, and where they differ it is only the 8-bit rounding
/// the ROP does that the shader's f32 arithmetic does not.
///
/// Pushed into the shader crate rather than read there, because that crate is compiled for the
/// browser and the browser has no environment [[vitaslop-browser-has-no-env]].
#[cfg(feature = "gpu")]
pub(crate) fn gxp_dest_blend_lowering() -> bool {
    use std::sync::OnceLock;
    static CELL: OnceLock<bool> = OnceLock::new();
    *CELL.get_or_init(|| {
        // Read on every build, `gpu` or not, so the KNOBS.md index and the browser override
        // table describe the same set of knobs whatever the crate was compiled with; only the
        // push into the recompiler needs it, and without the GPU stack nothing draws anyway.
        let spec = crate::knobs::var("VITASLOP_GXP_DEST_BLEND");
        #[cfg(not(feature = "gpu"))]
        {
            let _ = spec;
            false
        }
        #[cfg(feature = "gpu")]
        {
            use vitaslop_gxp_shader::module as m;
            let forms = match spec.as_deref().map(str::trim) {
                Err(_) | Ok("1") => m::FORM_DEFAULT,
                Ok("all") => m::FORM_ALL,
                Ok("0") => 0,
                // A comma list bisects a wrong picture to ONE shape without a rebuild.
                Ok(list) => list.split(',').fold(0u32, |acc, w| {
                    acc | match w.trim() {
                        "lerp" => m::FORM_LERP,
                        "modulate" => m::FORM_MODULATE,
                        "additive" => m::FORM_ADDITIVE,
                        "lerp-src" => m::FORM_LERP_SRC,
                        _ => 0,
                    }
                }),
            };
            m::set_dest_blend_lowering(forms);
            forms != 0
        }
    })
}

/// `VITASLOP_GXP_VERTEX_PASSTHROUGH=0` makes every recompiled draw repack its vertex stream into
/// tightly-packed f32, which is what this renderer did unconditionally until 2026-09-10d. An A/B
/// arm - VALUE-sensitive, ON by default - and the arm back for the whole passthrough path (see
/// `passthrough_vertex_format`).
///
/// The repack writes EVERY component as a 4-byte f32 padded to the SHADER's declared width, so an
/// `F16x2` texcoord (4 guest bytes) costs 8 and a `U8N` colour costs 16. MEASURED on a retail
/// sports title: 50.3 MB of guest vertex streams a frame became a 362 MB arena - about 7x - and
/// that one number is the packed-geometry cache's size, the per-frame `write_buffer` volume, and
/// most of the `prepare` stage that was 82% of a 265 ms browser render frame.
#[cfg(feature = "gpu")]
pub(crate) fn gxp_vertex_passthrough() -> bool {
    use std::sync::OnceLock;
    static CELL: OnceLock<bool> = OnceLock::new();
    *CELL.get_or_init(|| {
        crate::knobs::var("VITASLOP_GXP_VERTEX_PASSTHROUGH").map(|v| v.trim() != "0").unwrap_or(true)
    })
}

/// `VITASLOP_GXP_CULL=0` restores the pre-2026-08-19b "draw both windings". An A/B arm, so it is
/// VALUE-sensitive rather than presence-only.
#[cfg(feature = "gpu")]
pub(crate) fn gxp_cull() -> bool {
    use std::sync::OnceLock;
    static CELL: OnceLock<bool> = OnceLock::new();
    *CELL.get_or_init(|| crate::knobs::var("VITASLOP_GXP_CULL").map(|v| v.trim() != "0").unwrap_or(true))
}

/// Whether the shader pairs the guest's patcher names are TRANSLATED ahead of the draws that bind
/// them - see [`gxm::GxmRenderer::precompile_pairs`]. OFF; `VITASLOP_GXP_PRECOMPILE=1` arms it.
///
/// >>> OFF BECAUSE OF WHAT THE PATCHER NAMES, NOT BECAUSE OF THE KEY. MEASURED on the phone
/// (a football title, job 105): the patcher named 4,096 pairs (the list cap), the walk translated them in
/// 8,072 ms of render-worker time, and 43 draws used one - 1% of the work, spread as up to 6 ms
/// of every frame across the loads and the coin toss. The translation speedups and the draw-time
/// memo (`vitaslop_gxp_shader::link_programs_memo`) are unconditional and are what cut the
/// in-frame cost; what would make this worth a default is bounding the walk by DEMAND.
///
/// This replaced a warmer that was OFF for good reason: it created shader MODULES filed under
/// the two blobs alone while the draw looked them up under a key folded with the blend, raw-slot
/// and vertex-layout state, so for every skinned pair it paid the work and the draw paid it
/// again (the football title's hitches -81% worse at the worst window with it on). The translation memo keys on
/// the link's FULL inputs, and the patcher-time entry carries the vertex layout the title created
/// its program with, so the draw's lookup is the same key by construction.
#[cfg(feature = "gpu")]
pub(crate) fn gxp_precompile() -> bool {
    use std::sync::OnceLock;
    static CELL: OnceLock<bool> = OnceLock::new();
    *CELL.get_or_init(|| {
        crate::knobs::var("VITASLOP_GXP_PRECOMPILE").map(|v| v.trim() != "0").unwrap_or(false)
    })
}

/// >>> PIPELINES ARE STARTED ASYNC, AHEAD OF THE FRAME THAT DRAWS THEM - see
/// >>> `GxmRenderer::warm_pipelines`. `VITASLOP_ASYNC_PIPELINES=0` is the arm back.
///
/// MEASURED on the user's phone (runner job kind `pipeline-compile`, 100 real corpus pairs): a
/// synchronous `createRenderPipeline` compiles on Chrome's GPU MAIN thread - the compositor - so
/// after 10 of them a submit + `onSubmittedWorkDone` probe took 2,024 ms (idle: 1.7 ms) and the
/// whole browser froze (a golf title 5.6 s, a baseball title 7.8 s). With 10 `createRenderPipelineAsync` in
/// flight the same probe took 193 ms.
#[cfg(feature = "gpu")]
pub(crate) fn async_pipelines() -> bool {
    use std::sync::OnceLock;
    static CELL: OnceLock<bool> = OnceLock::new();
    *CELL.get_or_init(|| {
        crate::knobs::var("VITASLOP_ASYNC_PIPELINES").map(|v| v.trim() != "0").unwrap_or(true)
    })
}

/// Draws linked with / without the vertex program their fragment was patched against - the
/// form `GxmRenderer::premodule_pairs` builds ahead in.
#[cfg(feature = "gpu")]
static DRAWS_PATCHED: std::sync::atomic::AtomicU64 = std::sync::atomic::AtomicU64::new(0);
#[cfg(feature = "gpu")]
static DRAWS_UNPATCHED: std::sync::atomic::AtomicU64 = std::sync::atomic::AtomicU64::new(0);

/// Native: a frame's new pipelines are built on several threads ahead of the encode - see
/// `GxmRenderer::prebuild_pipelines`. `VITASLOP_PARALLEL_PIPELINES=0` is the arm back.
#[cfg(not(target_arch = "wasm32"))]
#[cfg(feature = "gpu")]
pub(crate) fn parallel_pipelines() -> bool {
    use std::sync::OnceLock;
    static CELL: OnceLock<bool> = OnceLock::new();
    *CELL.get_or_init(|| {
        std::env::var("VITASLOP_PARALLEL_PIPELINES").map(|v| v.trim() != "0").unwrap_or(true)
    })
}

/// Microseconds spent inside `create_shader_module` and `create_render_pipeline` for recompiled
/// pairs, and how many pipelines that was. See the timing site in `build_gxp_pipeline`.
static PIPE_MODULE_US: std::sync::atomic::AtomicU64 = std::sync::atomic::AtomicU64::new(0);
static PIPE_CREATE_US: std::sync::atomic::AtomicU64 = std::sync::atomic::AtomicU64::new(0);
/// Microseconds spent in OUR OWN translation before the driver sees anything: `link_programs`
/// (decoding both USSE programs, building the IR, emitting the WGSL) plus the clip fixup. It is
/// separate from [`PIPE_MODULE_US`] because "the shader build is slow" has two halves with
/// unrelated fixes - our Rust, and the driver's compiler - and one number cannot say which.
/// Counted for a REFUSED link too: a pair that fails to link still paid for the attempt.
static PIPE_LINK_US: std::sync::atomic::AtomicU64 = std::sync::atomic::AtomicU64::new(0);
/// Microseconds spent compiling WGSL AHEAD of any draw, when the guest's shader patcher named
/// the pair - see `GxmRenderer::precompile_pairs`. Reported separately from `PIPE_MODULE_US`
/// because the whole point is that this time is NOT in a gameplay frame: adding the two together
/// would hide the only thing the change is trying to move.
static PIPE_PRECOMPILE_US: std::sync::atomic::AtomicU64 = std::sync::atomic::AtomicU64::new(0);

/// Takes MILLISECONDS from `gxm::Stopwatch`, never a `std::time::Instant`.
///
/// >>> `std::time::Instant::now()` PANICS ON wasm32-unknown-unknown - "time not implemented on
/// > > > this platform" - and it panics inside `GxpLive::prepare`, i.e. on the first frame the browser
/// > > > renders anything. It shipped that way for one build and killed the run worker outright.
/// > > > `Stopwatch` exists a few hundred lines above precisely for this and is the only clock this
/// > > > module may use. [[vitaslop-count-bytes-when-there-is-no-clock]] is the same lesson from the
/// > > > other direction: the browser has no phase timer, so reach for a COUNT before a clock.
#[cfg(feature = "gpu")]
fn add_build_ms(slot: &std::sync::atomic::AtomicU64, ms: f64) {
    slot.fetch_add((ms * 1000.0).max(0.0) as u64, std::sync::atomic::Ordering::Relaxed);
}

/// `(shader-module ms, pipeline-create ms)` since the last call, and reset. The split that says
/// whether building pipelines AHEAD of the draw that needs them is worth doing, and which half
/// of it could be moved.
pub fn take_pipeline_build_split() -> (f64, f64, f64) {
    use std::sync::atomic::Ordering::Relaxed;
    let (l, m, c) = (PIPE_LINK_US.swap(0, Relaxed), PIPE_MODULE_US.swap(0, Relaxed), PIPE_CREATE_US.swap(0, Relaxed));
    RUN_LINK_US.fetch_add(l, Relaxed);
    RUN_MODULE_US.fetch_add(m, Relaxed);
    RUN_CREATE_US.fetch_add(c, Relaxed);
    (l as f64 / 1000.0, m as f64 / 1000.0, c as f64 / 1000.0)
}

/// What every `take_pipeline_build_split` has taken, for the run - so a span's difference of
/// two peeks does not read a panel window's reset as zero (see `run_encode_work`).
static RUN_LINK_US: std::sync::atomic::AtomicU64 = std::sync::atomic::AtomicU64::new(0);
static RUN_MODULE_US: std::sync::atomic::AtomicU64 = std::sync::atomic::AtomicU64::new(0);
static RUN_CREATE_US: std::sync::atomic::AtomicU64 = std::sync::atomic::AtomicU64::new(0);

/// The same three as RUN totals (taken + not yet taken) - for a caller that differences two
/// reads around one slow span while the panel keeps its own `take`; a `take` in between does
/// not move them.
pub fn peek_pipeline_build_split() -> (f64, f64, f64) {
    use std::sync::atomic::Ordering::Relaxed;
    (
        (RUN_LINK_US.load(Relaxed) + PIPE_LINK_US.load(Relaxed)) as f64 / 1000.0,
        (RUN_MODULE_US.load(Relaxed) + PIPE_MODULE_US.load(Relaxed)) as f64 / 1000.0,
        (RUN_CREATE_US.load(Relaxed) + PIPE_CREATE_US.load(Relaxed)) as f64 / 1000.0,
    )
}

/// Milliseconds spent compiling WGSL ahead of the draw, since the last call, and reset.
pub fn take_precompile_ms() -> f64 {
    PIPE_PRECOMPILE_US.swap(0, std::sync::atomic::Ordering::Relaxed) as f64 / 1000.0
}

/// Shader pairs whose PIPELINE the device itself refused, keyed the way every other report
/// here keys a pair.
fn poisoned_pairs() -> &'static std::sync::Mutex<std::collections::HashSet<u64>> {
    static POISONED: std::sync::OnceLock<std::sync::Mutex<std::collections::HashSet<u64>>> =
        std::sync::OnceLock::new();
    POISONED.get_or_init(|| std::sync::Mutex::new(std::collections::HashSet::new()))
}

/// Learn from a WebGPU uncaptured error which shader pair the DEVICE could not build, so the
/// next frame can leave that one draw out instead of losing the whole picture.
///
/// # >>> ONE REFUSED PIPELINE BLANKS THE ENTIRE FRAME, AND THAT IS THE FAILURE THIS PREVENTS
/// A `setPipeline` with an invalid pipeline invalidates the render pass, which invalidates the
/// command buffer, which makes `queue.submit` a no-op. So a device that refuses FOUR of a
/// title's shader pairs does not lose four objects - it loses **every** draw of every frame,
/// and the screen is black with no visible relationship to the four errors scrolling past. That
/// is exactly how an Android PowerVR device presented a race: 1,554 draws prepared, 0 pixels.
///
/// The device's own message names the pipeline by its label, which is the pair key
/// (`[Invalid RenderPipeline "gxp:873eb144f958a48b"]`), so the text IS the diagnosis. Parsing it
/// back is not elegant, but the alternative - `pop_error_scope` - is asynchronous and the render
/// path is not, and no amount of elegance is worth the frame.
///
/// Returns how many distinct pairs this message newly poisoned.
pub fn note_device_error(kind: &str, msg: &str) -> usize {
    let mut newly = 0usize;
    let mut rest = msg;
    while let Some(at) = rest.find("gxp:") {
        rest = &rest[at + "gxp:".len()..];
        let hex = rest.len() - rest.trim_start_matches(|c: char| c.is_ascii_hexdigit()).len();
        let Ok(key) = u64::from_str_radix(&rest[..hex], 16) else { continue };
        rest = &rest[hex..];
        if poisoned_pairs().lock().unwrap_or_else(|e| e.into_inner()).insert(key) {
            ANY_POISONED.store(true, std::sync::atomic::Ordering::Relaxed);
            newly += 1;
            report_warn!(
                "gxp pair {key:016x}: THE DEVICE REFUSED THIS PIPELINE [{kind}] and every draw \
                 that uses it is being DROPPED from here on. One invalid pipeline invalidates \
                 the whole command buffer, so leaving it in loses the entire frame - this trades \
                 one object for every other object in it. THE PICTURE IS NOW INCOMPLETE and this \
                 is not a fix: the pair is named so it can be dumped with \
                 `VITASLOP_GXP_WGSL_DIR` and taken back to its two blobs by content hash. The \
                 device said: {msg}"
            );
        }
    }
    newly
}

/// Say - once per pair - that a prepared GXP draw was encoded with NO geometry.
///
/// A draw whose index count or vertex/index slice is empty is submitted and rasterises
/// nothing, and from the finished frame that is identical to a draw the renderer never
/// issued, to one the device refused, and to one that shaded transparent. Those are four
/// different bugs and only this one is silent.
pub fn report_empty_gxp_geometry(key: u64, index_count: u32, v_len: u64, i_len: u64) {
    if index_count != 0 && v_len != 0 && i_len != 0 {
        return;
    }
    use std::collections::HashSet;
    use std::sync::Mutex;
    static SEEN: Mutex<Option<HashSet<u64>>> = Mutex::new(None);
    let mut g = SEEN.lock().unwrap_or_else(|e| e.into_inner());
    if !g.get_or_insert_with(HashSet::new).insert(key) {
        return;
    }
    report_warn!(
        "gxp pair {key:016x}: encoded with EMPTY geometry - {index_count} indices, {v_len} vertex bytes, {i_len} index bytes. The draw is submitted and rasterises nothing, which on screen is indistinguishable from a draw that was never issued."
    );
}

/// Whether [`note_device_error`] has seen the device refuse this pair's pipeline.
///
/// The atomic is checked FIRST and is the whole point of it: this runs once per recompiled
/// draw, a race frame here submits ~560 of them, and taking a mutex per draw to ask a question
/// whose answer is "no" on every device that works would put the damage control in the hot
/// path of the case it exists to protect.
pub fn gxp_pair_poisoned(key: u64) -> bool {
    if !ANY_POISONED.load(std::sync::atomic::Ordering::Relaxed) {
        return false;
    }
    poisoned_pairs().lock().unwrap_or_else(|e| e.into_inner()).contains(&key)
}

/// Set once [`note_device_error`] poisons anything. See [`gxp_pair_poisoned`].
static ANY_POISONED: std::sync::atomic::AtomicBool = std::sync::atomic::AtomicBool::new(false);

/// How many draws the poison list has cost this frame, since the last call, and reset.
pub fn take_poisoned_draws() -> u32 {
    POISONED_DRAWS.swap(0, std::sync::atomic::Ordering::Relaxed)
}

static POISONED_DRAWS: std::sync::atomic::AtomicU32 = std::sync::atomic::AtomicU32::new(0);

/// The LAST frame's texture working set in bytes - what [`texture_budget_pressure`] judges,
/// published so a running heartbeat can carry it.
///
/// The high-water report (`report_texture_working_set`) only speaks on a new 32 MB high, so a
/// working set that climbs and then stays there is invisible after its first frame. A run that
/// gets slower as it goes needs the CURRENT value on every heartbeat, not the record.
pub fn texture_working_set_bytes() -> usize {
    LAST_WORKING_SET.load(std::sync::atomic::Ordering::Relaxed)
}

/// The texture-cache budget actually in force, in bytes - what the panel should print beside the
/// working set.
///
/// >>> BECAUSE "scaled x1.00" IS NOT THE SAME STATEMENT AS "THE BUDGET IS 477 MB", AND THE
/// >>> DIFFERENCE IS THE WHOLE POINT ON A PHONE.
///
/// The default is the CONSOLE's resident ceiling (`GAME_RESIDENT_CEILING_MB`), scaled by
/// `navigator.deviceMemory`. MEASURED on the user's device: it reports **8 GB**, which is the
/// value the specification caps at, so the scale is 1.00 and the phone is handed a 477 MB
/// texture budget - the same one a workstation gets. The scaling added for exactly this device
/// is therefore INERT on it, and the panel said `scaled x1.00` without ever saying what it was
/// scaling, so the gap was invisible.
///
/// Printing the number is not the fix; it is what makes the fix arguable. `deviceMemory` cannot
/// tell a phone from a desktop, and the ADAPTER can - `vendor=img-tec` is a PowerVR part and
/// nothing else ships one.
pub fn tex_cache_budget_now() -> usize {
    tex_cache_budget_bytes()
}

/// The RETENTION bound - what the view cache may hold ACROSS frames, as opposed to the budget
/// above, which is also the re-encode gate. See [`tex_retain_budget_bytes`]. Reported beside the
/// budget because a panel that prints only the budget is what let 343 MB sit behind a line
/// saying `14 MB of a 477 MB budget` and read as healthy.
pub fn tex_retain_budget_now() -> usize {
    tex_retain_budget_bytes()
}

pub fn texture_budget_pressure() -> bool {
    let ws = LAST_WORKING_SET.load(std::sync::atomic::Ordering::Relaxed);
    // Unknown (no frame finished yet) counts as PRESSURE: the first frames of a screen are
    // exactly when the working set is being built, and guessing "plenty of room" there would
    // skip the compression that stops the budget being blown in the first place.
    // >>> THE SAME FUNCTION the uploader enforces, not a second copy of its arithmetic.
    //
    // This used to re-read the knob here with its own `unwrap_or(256)` while
    // `gxm::tex_cache_budget_bytes` defaulted to the console's 477 MiB ceiling - so the comment
    // that used to sit here, "read the same way, so the two cannot disagree about what tight
    // means", was false, and they disagreed by 221 MiB. Pressure therefore declared itself at
    // 171 MiB where the uploader's own threshold is 318, and every screen spent its first frames
    // transcoding textures to save memory that was not short. Setting `VITASLOP_TEX_CACHE_MB`
    // hid it, because that moves BOTH readers at once - which is exactly how a duplicated
    // default stays invisible.
    let budget = tex_cache_budget_bytes();
    ws == 0 || ws * 3 >= budget * 2
}

/// The decoded texels of a [`GxmTexture`] - produced the FIRST time something reads them, and
/// never produced at all when nothing does.
///
/// # >>> THE DECODE WAS UNCONDITIONAL, AND THE PATH THAT SHIPS THROWS IT AWAY
/// Every bound texture was expanded to RGBA8 the moment it was captured, and the result was
/// then held for the life of the run in the decode cache. That is exactly right for the
/// fixed-function renderer, which samples those bytes. It is pure waste for the configuration
/// this project actually ships: the recompiler path hands the GPU the guest's own compressed
/// BLOCKS ([`CompressedUpload`]), so `upload_gxp_texture` returns before it ever looks at
/// `rgba`, and the expansion is a PVRTC or BC decode of the whole image whose only consumer is
/// the allocator that frees it.
///
/// It cost twice over. Once in CPU - the decode is the single most expensive thing a screen
/// transition does, and it ran for textures whose blocks were about to be handed over untouched.
/// And once in MEMORY - eight bytes of RGBA8 per byte of guest texture, resident in the decode
/// cache, which is what pushed a phone's working set past its budget and started the eviction
/// thrash that the compressed upload had just been written to prevent. A transcode paid it
/// TWICE: `transcoded_source` decodes each level itself, so the eager decode was a second,
/// discarded copy of level 0.
///
/// Laziness rather than a flag, deliberately: every consumer that genuinely needs texels
/// (the software rasterizer, the fixed-function upload, the vertex-texture clip probe, the
/// diagnostics) keeps working unchanged and simply pays for what it reads. There is no
/// configuration in which this returns different bytes from the eager decode - it returns the
/// same bytes, or nobody asks.
pub struct Texels(std::sync::Arc<TexelsInner>);

struct TexelsInner {
    ready: std::sync::OnceLock<Vec<u8>>,
    /// The decode itself. `None` is an already-materialised buffer (see [`Texels::ready`]).
    make: Option<Box<dyn Fn() -> Vec<u8> + Send + Sync>>,
}

impl Texels {
    /// Texels that already exist. No decode is deferred and [`Texels::resident`] is true from
    /// the start.
    pub fn ready(v: Vec<u8>) -> Self {
        let cell = std::sync::OnceLock::new();
        let _ = cell.set(v);
        Texels(std::sync::Arc::new(TexelsInner { ready: cell, make: None }))
    }

    /// Texels that will be produced by `f` if anything reads them. `f` runs AT MOST once, and
    /// every clone of this handle shares the one result.
    pub fn lazy(f: impl Fn() -> Vec<u8> + Send + Sync + 'static) -> Self {
        Texels(std::sync::Arc::new(TexelsInner {
            ready: std::sync::OnceLock::new(),
            make: Some(Box::new(f)),
        }))
    }

    /// Have these texels actually been produced? Used by accounting that must not FORCE the
    /// decode just to price it - reading `len()` there would defeat the whole mechanism.
    pub fn resident(&self) -> bool {
        self.0.ready.get().is_some()
    }

    /// Bytes currently held, WITHOUT forcing the decode: 0 while it has not run.
    pub fn resident_len(&self) -> usize {
        self.0.ready.get().map_or(0, |v| v.len())
    }
}

impl std::ops::Deref for Texels {
    type Target = Vec<u8>;

    /// Reading the texels IS the decode. Every existing `t.rgba.len()` / `&t.rgba` /
    /// `t.rgba.get(..)` call site keeps its meaning; what changes is when the work happens.
    fn deref(&self) -> &Vec<u8> {
        self.0.ready.get_or_init(|| match &self.0.make {
            Some(f) => f(),
            // Unreachable: `ready` sets the cell, `lazy` sets `make`. An empty buffer is the
            // only honest answer that is not a panic inside a `Deref`.
            None => Vec::new(),
        })
    }
}

impl Clone for Texels {
    fn clone(&self) -> Self {
        Texels(self.0.clone())
    }
}

impl std::fmt::Debug for Texels {
    /// Deliberately does NOT force the decode. A `{:?}` of a texture is a diagnostic, and a
    /// diagnostic that performs the expensive work it is reporting on is an instrument whose
    /// failure imitates its subject.
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        match self.0.ready.get() {
            Some(v) => write!(f, "Texels({} bytes)", v.len()),
            None => write!(f, "Texels(not decoded)"),
        }
    }
}

#[derive(Clone, Debug)]
pub struct GxmTexture {
    pub key: u64,
    /// Guest address of the texture's pixel data. Identity, not content: a title that
    /// renders a pass into an offscreen buffer and samples it in a later pass binds a
    /// texture whose data pointer IS that pass's colour surface, and the guest bytes
    /// behind it are stale (the GPU wrote the real pixels, not the guest). The renderer
    /// matches this against the render targets it has drawn this frame and binds the
    /// rendered texture instead - see `GxmRenderer::encode_chain`.
    pub data_addr: u32,
    pub width: u32,
    pub height: u32,
    /// Number of `width x height` images `rgba` holds back to back: 1 normally, 6 for a
    /// cube map (in +X, -X, +Y, -Y, +Z, -Z order, which is WebGPU's array-layer order).
    pub faces: u32,
    /// The decoded texels, in the layout [`Self::texel`] names - produced on first READ, so a
    /// texture the GPU takes compressed is never expanded at all. See [`Texels`].
    pub rgba: Texels,
    /// Which layout `rgba` is in - four bytes a texel, or four HALVES a texel.
    ///
    /// # Why this seam is not always RGBA8
    /// It was, and for an IMAGE that is a defensible trade. It is not defensible for a texture
    /// whose texels are DATA, and the guest's 64-bit half-float formats are how a title stores
    /// data in a texture. Crushing a 16-bit lane to 8 bits leaves 256 distinct values per
    /// channel, and if those values are coordinates rather than colours, 256 is the number of
    /// distinct places the shader can address.
    ///
    /// MEASURED: this title's campaign map draws its labels through a "vector canvas" - a
    /// 512x128 `F16F16F16F16` (base format `0x1b`) lookup texture, sampled BY THE VERTEX
    /// PROGRAM, whose R and G lanes are each glyph's origin in a 2048x2048 atlas. Through an
    /// 8-bit seam one step of that origin is 2048/255 = 8 texels, so every glyph snapped to an
    /// 8-texel grid and the label rendered as skewed fragments. No amount of shader or sampler
    /// work could have fixed it: the information was gone at decode.
    pub texel: TexelSeam,
    /// The `SceGxmTextureFormat` base format and channel swizzle `rgba` was decoded THROUGH.
    /// Diagnostic only - the pixels are already RGBA8 by the time the renderer sees them - but
    /// it is the one fact that separates "this channel is genuinely zero in the asset" from "we
    /// decoded the asset through the wrong layout", and those need opposite fixes.
    pub base_format: u32,
    pub swizzle: u32,
    /// >>> WHETHER THE GUEST'S OWN BYTES AT [`Self::data_addr`] ARE ALL ZERO, as snapshotted at
    /// >>> bind time.
    ///
    /// This is the witness that tells a LIVE render target being sampled from a RECYCLED
    /// allocation, and it is the only one that does. A target's pixels live on the GPU, so its
    /// guest memory reads empty [[vitaslop-a-render-target-reads-empty-in-guest-memory]]; a
    /// texture the guest allocated over a freed target has the guest's real bytes in it. Both
    /// bind a descriptor whose extent can disagree with the target's, so the extent alone cannot
    /// separate them - see `encode_chain`'s `rtt_alias_block`.
    ///
    /// Computed where the bytes already are (the capture's snapshot) rather than here: the
    /// renderer holds only [`Self::rgba`], which is produced lazily and on purpose, and forcing
    /// a decode to answer this would expand every texture the GPU was about to take compressed.
    pub guest_bytes_unwritten: bool,
    /// Bytes per row of the guest's texture memory (its STRIDE), 0 when unknown. What says
    /// whether a texture whose address lies inside a render target is a SUB-RECTANGLE of it
    /// (the same row pitch) or a reinterpretation of its memory (another pitch) - see the
    /// sub-rectangle rule in `encode_pass`.
    pub row_bytes: u32,
    /// True if the guest set this texture's magnification filter to LINEAR
    /// (`SceGxmTextureFilter` == 1); the renderer then bilinear-samples it, matching
    /// the software rasterizer's `sample_texture_bilinear`. False = POINT/nearest.
    pub filter_linear: bool,
    /// The guest's `SceGxmTextureAddrMode` for U and V
    /// (`sceGxmTextureSet{U,V}AddrMode`, 0 = REPEAT). The recompiled path used to hardcode
    /// REPEAT for every sampler, which is invisible while every coordinate stays inside
    /// [0,1] and catastrophic the moment one does not: a full-screen pass reading one texel
    /// past the edge wraps to the OPPOSITE edge instead of clamping, and a title's composite
    /// showed the world tiled diagonally across the display rather than clipped.
    pub addr_mode_u: u32,
    pub addr_mode_v: u32,
    /// The finest mip level the guest lets this texture be sampled at (`sceGxmTextureSetLodMin`):
    /// a whole-level lower clamp on the LOD, after the bias - the sampler's `lod_min_clamp`. A
    /// depth-of-field layer sampled with it pinned to a blurry level reads as rows of sharp
    /// copies when it is dropped.
    pub lod_min: u32,
    /// The guest set `sceGxmTextureSetGammaMode` on this texture, so the hardware sampler
    /// sRGB-DECODES every texel it fetches. Uploading through an sRGB texture format does the
    /// same decode in the same place, before filtering.
    ///
    /// The counterpart of [`RttTarget::gamma`], which is the write half of the same state.
    pub gamma: bool,
    /// Mip levels the GUEST'S texture declares, counting level 0, and whether it asked the
    /// hardware to FILTER between them (`sceGxmTextureSetMipFilter`, 0 = disabled).
    ///
    /// # These are carried so `mips_for_texture` can READ the guest instead of guessing
    /// The decoded path builds a chain for every RGBA8 texture, and that is right whenever the
    /// guest HAS a chain the hardware samples: we decode level 0 only, so without one a minified
    /// surface is point-sampled from the full-size image and a distant road reads as white
    /// speckle ([[vitaslop-textures-need-mips]]).
    ///
    /// It is wrong for a texture the guest gave ONE level and told the hardware not to filter
    /// between levels - that is a surface the DEVICE samples from its base level alone, so a
    /// generated chain is filtering the Vita never did. The compressed passthrough already
    /// applies exactly this rule and says so in as many words ("what would be lost is a chain we
    /// invented, which the Vita never had"); the decoded path could not, because these two facts
    /// stopped at the runtime and never reached here.
    ///
    /// MEASURED on a golf title's character select: the 1024x512 system-font atlas at `0x8f8b4c00`
    /// is written by the guest at one level, and the club-bar label minifies it several times
    /// over - with an invented chain that is a soft grey smear where the device draws small
    /// sharp text.
    pub levels: u32,
    pub mip_filter: u32,
    /// The guest's own block-compressed bytes, when this texture may be handed to the GPU
    /// WITHOUT being decoded - see [`CompressedUpload`]. `None` is the ordinary case and
    /// costs nothing: `rgba` is still the decode, and the uploader uses it.
    pub compressed: Option<CompressedUpload>,
    /// The guest's own bytes and layout when this texture's whole decode is a PERMUTATION and
    /// the GPU can do it - see [`GpuRawExpand`]. `None` is the ordinary case and costs nothing:
    /// `rgba` is still the decode and the uploader uses it.
    pub raw: Option<GpuRawExpand>,
    /// The guest's own 4:2:0 planes, when this texture is a decoded VIDEO frame.
    ///
    /// Carried instead of relying on `rgba` because that decode is the thing worth avoiding:
    /// a video frame changes every frame, so its conversion is paid per frame, and doing it
    /// on the GPU also shrinks what crosses to the driver from RGBA to the guest's own 4:2:0
    /// bytes. `rgba` remains the fallback and produces the same picture.
    pub planar_yuv: Option<PlanarYuvSource>,
}

/// The guest's bytes and layout for a two-plane 4:2:0 texture - see [`GxmTexture::planar_yuv`].
#[derive(Clone, Debug)]
pub struct PlanarYuvSource {
    pub width: u32,
    pub height: u32,
    pub luma_stride: u32,
    pub chroma_stride: u32,
    pub chroma_offset: u32,
    pub swap_chroma: bool,
    /// The `SceGxmYuvProfile` (0..3) the context had for the texture's conversion slot.
    pub profile: u32,
    pub data: std::sync::Arc<[u8]>,
}

/// A block-compressed format WebGPU can be handed directly, and which of the guest's base
/// formats it IS.
///
/// The guest's `UBC1`/`UBC2`/`UBC3` (base formats `0x85`/`0x86`/`0x87`) are BC1/BC2/BC3 - the
/// same blocks, the same 4x4 geometry, the same bits. Nothing is transcoded here; the only
/// work between guest memory and the GPU is undoing the Morton BLOCK order on a swizzled
/// texture, which is a permutation of whole blocks and touches no bit inside one.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub enum BlockFormat {
    Bc1,
    Bc2,
    Bc3,
    /// ETC2 RGB8: 4 bits per texel, alpha discarded. The rate the guest's PVRTC was stored at.
    Etc2Rgb8,
    /// ETC2 RGBA8: the 64-bit EAC alpha block followed by the 64-bit colour block, 8 bpp.
    Etc2Rgba8,
}

impl BlockFormat {
    /// Bytes in one 4x4 block.
    pub fn block_bytes(self) -> u32 {
        match self {
            BlockFormat::Bc1 | BlockFormat::Etc2Rgb8 => 8,
            BlockFormat::Bc2 | BlockFormat::Bc3 | BlockFormat::Etc2Rgba8 => 16,
        }
    }

    /// The adapter feature this format needs.
    pub fn family(self) -> BlockFamily {
        match self {
            BlockFormat::Bc1 | BlockFormat::Bc2 | BlockFormat::Bc3 => BlockFamily::Bc,
            BlockFormat::Etc2Rgb8 | BlockFormat::Etc2Rgba8 => BlockFamily::Etc2,
        }
    }
}

/// Which family of block-compressed formats the GPU in use accepts.
///
/// # This is not a boolean, and treating it as one is what made the whole feature desktop-only
/// The first version of this recorded "does the adapter take BC". Every desktop said yes and the
/// device this work exists for said no, so the phone kept decoding every compressed texture to
/// RGBA8 at 4-8x - a 354 MB working set against a 256 MB budget - while the desktop reported a
/// 274 -> 57 MB win. The fact that decides the whole feature is WHICH family, not whether.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub enum BlockFamily {
    None,
    Bc,
    Etc2,
}

/// The guest's compressed bytes, relaid into the order a WebGPU upload wants.
///
/// # Why this exists rather than a flag on the decode
/// A race frame of one retail title needs 260 MB of texture against a 256 MB budget, and the
/// reason is not the number of textures - it is that every compressed format is CPU-decoded to
/// RGBA8 on the way in. Past the budget the allocation fails on a phone and the draw comes out
/// WHITE, which is exactly the reported symptom, on exactly the screens that carry big
/// textures. Handing the blocks over unchanged removes the expansion instead of managing it.
#[derive(Clone, Debug)]
pub struct CompressedUpload {
    pub format: BlockFormat,
    /// The dimensions `data` actually describes.
    ///
    /// # >>> THESE ARE ALWAYS THE GUEST'S OWN SIZE TODAY, AND THAT IS A RULE, NOT A COINCIDENCE
    /// This field once carried a TIERED transcode: build the first version of a texture from the
    /// guest's small mip levels and replace it with larger tiers over the following frames, so a
    /// screen transition went soft for a moment instead of freezing. **That mechanism is
    /// DELETED** - it is the progressive-residency trade written up in
    /// [[vitaslop-never-trade-quality]], where the device never got past 128 texels a side and a
    /// 2048x2048 atlas rendered at a sixteenth of its resolution per axis for a whole run.
    ///
    /// The field stays because it is what the uploader must declare the texture at, and because
    /// declaring `t.width`/`t.height` over a smaller buffer is a validation error at best and a
    /// read past the buffer at worst. Anything that ever makes these differ from the guest's own
    /// size is shipping a lower-resolution picture and needs the argument that goes with that.
    pub width: u32,
    pub height: u32,
    /// The blocks themselves, or the instructions for the GPU to make them. See [`CompressedData`].
    pub data: CompressedData,
    /// Mip levels `data` holds per face, counting level 0. These are the GUEST'S levels: there
    /// is no box filter for a compressed block, so a passthrough either carries the chain the
    /// hardware itself samples or ships level 0 alone, and level-0-alone is the "distant road
    /// reads as white speckle" failure ([[vitaslop-textures-need-mips]]).
    pub levels: u32,
    /// True if these blocks were RE-ENCODED from decoded texels rather than being the guest's
    /// own - a second lossy step, not a handover.
    ///
    /// # Carried, not inferred from the format
    /// The first version of the working-set report worked this out from the guest base format:
    /// PVRTC has no WebGPU format, so PVRTC + compressed must mean transcoded. That held until
    /// a refused UBC2 texture started going through the transcode too, at which point the one
    /// report anyone reads to find out where the megabytes went described 45 MB of re-encoded
    /// blocks as "guest blocks, guest mips". A provenance that can be inferred today is a
    /// provenance that will be wrong later.
    pub transcoded: bool,
}

impl CompressedUpload {
    /// Bytes this upload will occupy on the GPU, whichever side produced them.
    ///
    /// A GPU-built chain is priced from its geometry rather than from a buffer, because the
    /// buffer does not exist yet on the CPU side and never will - which is the entire point.
    /// The CPU-built blocks, or `None` when the GPU will build them.
    ///
    /// Every caller of this is asserting something about the BYTES, which only exist on the CPU
    /// path. A GPU plan has no bytes to assert about by construction - that is what it is for -
    /// so the `Option` is the honest signature rather than an empty slice that would let a test
    /// pass by measuring nothing.
    pub fn cpu_bytes(&self) -> Option<&[u8]> {
        match &self.data {
            CompressedData::Cpu(v) => Some(v),
            CompressedData::Gpu(_) => None,
        }
    }

    pub fn byte_len(&self) -> usize {
        match &self.data {
            CompressedData::Cpu(v) => v.len(),
            CompressedData::Gpu(_) => {
                let bb = self.format.block_bytes() as usize;
                (0..self.levels)
                    .map(|l| {
                        let w = (self.width >> l).max(1);
                        let h = (self.height >> l).max(1);
                        (w.div_ceil(4) * h.div_ceil(4)) as usize * bb
                    })
                    .sum()
            }
        }
    }
}

/// Where a [`CompressedUpload`]'s blocks come from.
///
/// # >>> THE ENCODER IS THE MOST EXPENSIVE THING THIS EMULATOR DOES, AND IT RAN ON THE IDLE SIDE
/// A screen transition on the target device binds a hundred textures at once, every one of them
/// PVRTC, which no WebGPU adapter can be handed compressed. Each has to be decoded and re-encoded,
/// and the device's CPU encoder runs at about 1 Mtexel/s - so a single 1024x1024 atlas with its
/// chain is well over a second of frozen guest, and a transition frame MEASURED at
/// **BUILD 21,182 ms**. Over the same frame the GPU was doing essentially nothing: `pass 3.3 ms`
/// against 128 ms of CPU.
///
/// Block encoding is embarrassingly parallel over independent 4x4 blocks and needs no guest state
/// at all, so it belongs on the processor that is idle. [`CompressedData::Gpu`] carries the guest's
/// own bytes and a description of how to read them; the decode, the mip chain and the block encode
/// then happen in compute shaders and the result is copied straight into the compressed texture.
/// **Nothing crosses back to the CPU**, so there is no stall to schedule around and no readback.
///
/// It is not a quality trade in either direction: the GPU runs the same algorithm over the same
/// texels at the guest's own resolution. `gpu_etc2_matches_the_cpu_encoder` is what says so,
/// against the CPU encoder this replaces, over the same corpus its own error ceilings are written
/// from.
#[derive(Clone, Debug)]
pub enum CompressedData {
    /// Blocks already built on the CPU: level 0 first, then each smaller level, block-packed with
    /// no row padding, layer-major over the texture's faces - which is precisely what
    /// `TextureDataOrder::LayerMajor` reads, level by level, at each level's block-rounded
    /// PHYSICAL size.
    Cpu(std::sync::Arc<Vec<u8>>),
    /// The guest's raw bytes, to be decoded, mipped and encoded on the GPU.
    Gpu(GpuTranscode),
}

/// Everything the GPU needs to build a texture's blocks from the guest's own bytes.
///
/// The layout arithmetic (which level starts where, how many blocks, Morton or linear) is done
/// ONCE on the CPU, where `level_layout` and `morton_index` already live and are already tested,
/// and handed over as a table. A second implementation of that addressing in WGSL is exactly the
/// kind of duplicate that drifts, and its failure mode is a texture that decodes plausibly out of
/// the wrong bytes.
#[derive(Clone, Debug)]
pub struct GpuTranscode {
    /// The guest's pixel bytes for face 0. Single-face only - a cube map's six chains are not
    /// established, and the CPU path refuses them for the same reason.
    pub src: std::sync::Arc<[u8]>,
    /// How to read `src`.
    pub codec: SourceCodec,
    /// Texel dimensions of level 0.
    pub width: u32,
    pub height: u32,
    /// Levels the FINISHED texture will have: a full chain down to 1x1. Levels past
    /// `src_levels.len()` are box-filtered on the GPU from the level above, exactly as the CPU
    /// transcode does in RGBA8.
    pub levels: u32,
    /// The guest's own levels present in `src`, largest first.
    pub src_levels: Vec<SrcLevel>,
}

/// Note that a GPU texture was CREATED, for the handle-drift invariant. Called from every
/// site that creates one, including [`crate::texenc`], which is why it is crate-visible rather
/// than living beside its counter.
/// Note that a GPU texture was CREATED, for the handle-drift invariant. Every site that creates
/// one calls this, including [] - which is why it is re-exported here rather than
/// left inside the renderer module its counter lives in.
#[cfg(feature = "gpu")]
pub(crate) use gxm::{note_texture_created, note_texture_destroyed};

/// A guest texture whose "decode" is a PERMUTATION, described so the GPU can do it.
///
/// # Why this exists beside [`GpuTranscode`] rather than inside it
/// That one turns lossy blocks into other lossy blocks, and its whole design question is
/// whether the result matches the CPU closely enough. This is the opposite case: the guest's
/// texel is already four 8-bit channels, so the only work is undoing the Morton interleave that
/// decides WHERE a texel lives and applying the SWIZZLE4 channel order. There is no arithmetic
/// and nothing to lose, so it is not a trade at all - and it is the largest single item on the
/// target device, whose own report reads `texture decode by format: 2988.8 MB total - 0x0c raw
/// 2964.5 MB`.
///
/// What the CPU pays today for one of these is the per-texel un-swizzle AND a `writeTexture` of
/// the expanded RGBA8. What it pays through this is one buffer write of the guest's own bytes.
#[derive(Clone, Debug)]
pub struct GpuRawExpand {
    /// The guest's pixel bytes for face 0. Single-face only, like the transcode: a cube map's
    /// six chains are not established.
    pub src: std::sync::Arc<[u8]>,
    /// Texel dimensions of level 0.
    pub width: u32,
    pub height: u32,
    /// Levels the FINISHED texture will have. Levels past `src_levels.len()` are box-filtered
    /// on the GPU from the level above, by the same `halve` shader the transcode uses.
    pub levels: u32,
    /// The SWIZZLE4 selector - bits 12..14 of the guest's `SceGxmTextureFormat`, naming the
    /// channel order. The shader's arms are `render::swizzle4`'s, one for one.
    pub swizzle: u32,
    /// The guest's own levels present in `src`, largest first. `blocks_x` carries the level's
    /// ROW STRIDE IN TEXELS (the guest stride over four) for a linear level, and is unread for
    /// a swizzled one, which addresses through `padded_x`/`padded_y`.
    pub src_levels: Vec<SrcLevel>,
    /// The BLOCK codec the source levels are in, or `None` for an uncompressed source.
    ///
    /// >>> THIS IS WHAT TAKES A PHONE'S BC DECODE OFF ITS CPU, AND IT COSTS NO QUALITY.
    ///
    /// A GPU with no BC support has to expand every compressed guest texture to RGBA8, and
    /// `transcoded_source` deliberately declines to re-encode one that already fits the budget
    /// - re-encoding to ETC2 is a second LOSSY step and buys only megabytes. So those textures
    /// took the CPU decode: MEASURED on the user's device, one frame's working set is 130 MB
    /// with `0x85 BC -> RGBA8 x136 (98.1 MB)` in it, and the run's slowest frames are half a
    /// second each at ordinary host-call counts.
    ///
    /// That reasoning is sound and its premise was incomplete: the alternative is not only the
    /// lossy re-encode. The `decode_bc` compute pipeline already exists as stage ONE of the
    /// BC->ETC2 transcode, so the blocks can be expanded to RGBA8 on the GPU and STOPPED there
    /// - the same bytes the CPU decode produces, at the RGBA8 memory cost the policy already
    /// accepts. `gpu_bc_expand_matches_the_cpu_decoder` is what says "the same bytes".
    pub codec: Option<SourceCodec>,
}

/// One guest mip level inside [`GpuTranscode::src`].
#[derive(Clone, Copy, Debug)]
pub struct SrcLevel {
    /// Byte offset of this level from the start of `src`.
    pub byte_offset: u32,
    pub width: u32,
    pub height: u32,
    /// Blocks needed to cover the level, in the SOURCE codec's block geometry (4x4 for PVRTC
    /// 4bpp and for BC, 8x4 for PVRTC 2bpp).
    pub blocks_x: u32,
    pub blocks_y: u32,
    /// The power-of-two-padded grid a SWIZZLED level's Morton addressing runs over. Equal to
    /// `blocks_x`/`blocks_y` when the level is stored linearly.
    pub padded_x: u32,
    pub padded_y: u32,
    /// Block rows are Morton-ordered over the padded grid rather than laid out linearly.
    pub swizzled: bool,
}

/// The wgpu format a [`BlockFormat`] uploads through, in its plain or sRGB twin.
///
/// The renderer's own `block_wgpu_format` is the definition; this forwards to it so the GPU
/// transcoder ([`crate::texenc`]) creates its texture through exactly the same mapping the
/// uploader would have. Two spellings of "which wgpu format is this" is how a gamma decode ends
/// up applied on one path and not the other.
#[cfg(feature = "gpu")]
pub fn block_wgpu_format_pub(f: BlockFormat, gamma: bool) -> wgpu::TextureFormat {
    gxm::block_wgpu_format_pub(f, gamma)
}

/// The compressed format the GUEST stored a texture in, as the GPU decoder needs to read it.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub enum SourceCodec {
    /// PVRTC1 (`two = false`) or PVRTC2, at 2 or 4 bits per texel.
    Pvrtc { two: bool, four_bpp: bool },
    /// The guest's `UBC1`/`UBC2`/`UBC3` (base formats `0x85`/`0x86`/`0x87`), which ARE BC1/2/3.
    ///
    /// On a desktop these pass through untouched and this never runs. It exists for an adapter
    /// with no BC at all - the target device - where the alternatives are an RGBA8 expansion at
    /// four times the size or a CPU decode plus a CPU re-encode, which is the single most
    /// expensive path in the texture pipeline.
    Bc { base_format: u32 },
}

impl SourceCodec {
    /// Bytes in one SOURCE block. PVRTC is always 8; BC1 is 8 and BC2/BC3 are 16.
    pub fn block_bytes(self) -> u32 {
        match self {
            SourceCodec::Pvrtc { .. } | SourceCodec::Bc { base_format: 0x85 } => 8,
            SourceCodec::Bc { .. } => 16,
        }
    }
}

/// The per-material forward-lighting inputs the GPU `fs_opaque` needs to reproduce the
/// software rasterizer's `shade_lit`: a base-colour `tint`, one directional light
/// (`light_dir` world-space direction-to-light + `light_col`), and a flat `ambient`. The
/// runtime reflects these from the fragment program; neutral values (tint 1, light col 1,
/// modest ambient) make it a pass-through for a 2D/UI draw. Kept engine-neutral here so
/// `vitaslop-platform` stays free of any runtime dependency.
#[derive(Clone, Copy, Debug)]
pub struct GxmMaterial {
    pub tint: [f32; 3],
    pub light_dir: [f32; 3],
    pub light_col: [f32; 3],
    pub ambient: [f32; 3],
}

impl Default for GxmMaterial {
    fn default() -> Self {
        GxmMaterial {
            tint: [1.0, 1.0, 1.0],
            light_dir: [0.0, 1.0, 0.0],
            light_col: [1.0, 1.0, 1.0],
            ambient: [0.35, 0.35, 0.35],
        }
    }
}

/// One ready-to-draw call, CPU-decoded by the runtime into a canonical vertex
/// layout so the GPU never decodes a GXM attribute format. The vertex layout is
/// fixed: position `float32x3` at offset 0, texcoord `float32x2` at offset 12
/// (already divided by the draw's uv scale), color `unorm8x4` at offset 20, and the
/// world-space normal `float32x3` at offset 24 (baked by the runtime from the object
/// normal and the model-to-world matrix, for the opaque lighting term), stride 36.
/// Indices are always 32-bit. `texture` is `None` for an untextured draw (the renderer
/// binds a 1x1 white texel so one shader path serves both).
#[derive(Clone, Debug)]
pub struct GxmDraw {
    pub space: DrawSpace,
    pub vertices: Vec<u8>,
    pub indices: Vec<u8>,
    pub index_count: u32,
    pub texture: Option<GxmTexture>,
    /// The per-material lighting inputs for the opaque path (ignored when `opaque` is false).
    pub material: GxmMaterial,
    /// True for genuinely opaque 3D geometry: an MVP-space draw that also writes depth
    /// (`front_depth_write != SCE_GXM_DEPTH_WRITE_DISABLED`). Drives BOTH the pipeline
    /// (depth-tested opaque replace) AND the fragment combine (the albedo texel is taken
    /// straight, tone-mapped by `exposure`, alpha forced to 1). This is the software
    /// rasterizer's `depth_test` decision, carried here rather than re-derived from
    /// `space` alone - an MVP draw with depth writes DISABLED is a 2D alpha-blended
    /// overlay, not opaque, so keying opacity off MVP-space would wrongly replace and
    /// tonemap it. False = 2D overlay: `vertex_color * texel` modulate, alpha-blended in
    /// submission order.
    pub opaque: bool,
    /// Scene exposure (linear multiplier from `vsCoarseExposureReg`) applied to opaque
    /// draws before a Reinhard tonemap. 1.0 is a no-op (2D/UI and any shader with no
    /// exposure uniform), so the tonemap is skipped and the color passes through.
    pub exposure: f32,
    /// The guest's real vertex+fragment shaders + their draw inputs, for the GXP->WGSL
    /// recompiler (live guest-shader) path. `Some` only when the runtime captured it
    /// (`VITASLOP_GXP_LIVE`); the renderer links + caches a pipeline and ALWAYS falls back
    /// to the fixed-function fields above on any link/format error. See [`GxpRecompile`].
    pub gxp: Option<GxpRecompile>,
    /// This draw has NO valid fixed-function representation: its geometry carries neither a
    /// texcoord nor a per-vertex colour, so the fixed-function fields above have no colour
    /// source and would paint opaque white. Its real colour lives in the guest's fragment
    /// shader (typically a uniform), so it can only be drawn through `gxp`.
    ///
    /// The scene builder used to DISCARD such a draw outright, silently. That is what made a
    /// retail title's solid-colour UI fills - a button's interior, a panel backing - simply
    /// absent, indistinguishable from the guest never drawing them, and it survived being
    /// stared at in screenshots. Keeping the draw and marking it lets the recompiler render it
    /// exactly; if no recompiled pipeline is available the renderer skips it and SAYS SO,
    /// rather than either dropping it quietly or painting a white rectangle.
    pub shader_only: bool,
    /// The guest's `sceGxmSetRegionClip` for this draw - GXM's hardware SCISSOR. See
    /// [`RegionClip`].
    pub region_clip: RegionClip,
    /// The guest's `sceGxmSetViewport` for this draw, `[xOffset,xScale,yOffset,yScale,
    /// zOffset,zScale]`, with its `SceGxmViewportMode` - see [`viewport_rect`].
    ///
    /// The fixed-function path used to carry no viewport at all and pass the whole target,
    /// on the reasoning that it "packs its own screen-space geometry". It does not: an `Ndc`
    /// or `Mvp` draw arrives in clip space and is mapped onto the attachment here, so a
    /// guest that asked for half of it got twice the rectangle it wanted. That is the
    /// conformance app's viewport scene, which BOTH backends were failing.
    pub viewport: [f32; 6],
    pub viewport_enable: u32,
    /// The guest's stencil state for this draw, as [`GxpRecompile::stencil`] carries it - here
    /// for the FIXED-FUNCTION path, whose draws have no recompile payload to carry it in.
    pub stencil: [u32; 14],
    /// See [`GxpRecompile::two_sided`].
    pub two_sided: bool,
}

/// GXM's region clip (`sceGxmSetRegionClip`), which is the hardware SCISSOR, captured per
/// draw because it is per-draw state exactly as the viewport is.
///
/// # What the two enabled modes mean, and how that was settled
/// `SceGxmRegionClipMode` has four values in the top two bits: `NONE` (0), `ALL`
/// (`0x40000000`), `OUTSIDE` (`0x80000000`) and `INSIDE` (`0xC0000000`). The vitasdk
/// header's prose is ambiguous about which of the two enabled modes keeps the inside of
/// the rectangle and which keeps the outside, and reading it either way makes one of two
/// retail titles blank its own first scene. The titles settle it between them:
///
/// - A retail racer issues `INSIDE` with `0,0 .. 959,543` as the first call of its first
///   scene (a whole-target default), then a run of arbitrary rectangles - `9,0..959,543`,
///   then `0,0..719,543`, `539`, `404`, `302`, `226`, `169`, `127`, `95`, `71`, `53` -
///   which is a closing WIPE. **None of those is a multiple of 32.**
/// - A retail futuristic racer issues `OUTSIDE` with a whole-target rectangle and with
///   `0,0..127,63`, `0,64..63,95`, `0,96..31,127`, `32,96..63,127`, `64,64..95,95`,
///   `64,96..95,127`, `96,64..127,95`, `96,96..127,127` - a disjoint TILING of one
///   128x128 atlas, which only makes sense if each draw is confined to its rectangle.
///   **Every one of those is a multiple of 32.**
///
/// So BOTH enabled modes keep the INSIDE of the rectangle - an inverse reading of either
/// one blanks that title's whole-target default. That half stands.
///
/// **The GRANULARITY half is REFUTED, and by the check that was written to test it.** The
/// reading used to be that `OUTSIDE` clips whole 32-pixel tiles and `INSIDE` clips exactly,
/// on the evidence that every `OUTSIDE` rectangle those two titles issued was 32-aligned and
/// every `INSIDE` one was not. A third title - a retail sports title - issues **512 distinct
/// `OUTSIDE` rectangles in one round**, nearly all unaligned, and they are centred bands
/// moving ONE pixel a frame across a mip chain (`0,33..191,157`, `0,31..191,159`,
/// `0,38..191,152`, ... on 192/160/128/96/64-pixel targets). A scissor that quantised to 32
/// pixels could not animate that at all. Both enabled modes are pixel-exact, which is what
/// [`RegionClip::rect_in`] has always done, so no pixel changes - what changes is that the
/// alignment note is no longer a suspicion. See [`report_region_clip_not_tile_aligned`].
///
/// `ALL` is "clip everything" and is reported rather than guessed at: it has never been
/// observed, and a mode that draws nothing is not something to infer from silence.
#[derive(Clone, Copy, Debug, Default, PartialEq, Eq)]
pub struct RegionClip {
    /// The raw `SceGxmRegionClipMode` word (the enum lives in the top two bits).
    pub mode: u32,
    /// `[xMin, yMin, xMax, yMax]`, INCLUSIVE, in target pixels - GXM's own convention.
    pub rect: [u32; 4],
}

/// The widest and tallest render target GXM can address. A region-clip bound at or above it
/// cannot be describing any target, so it is a wrapped coordinate rather than a rectangle
/// written for a different surface - see [`RegionClip::rect_in`] for why the two must be told
/// apart and what it cost when they were not.
pub const MAX_SURFACE_EXTENT: u32 = 4096;

impl RegionClip {
    pub const NONE: u32 = 0x0000_0000;
    pub const ALL: u32 = 0x4000_0000;
    pub const OUTSIDE: u32 = 0x8000_0000;
    pub const INSIDE: u32 = 0xC000_0000;
    /// Tile side in pixels, for the alignment CHECK only. Nothing rounds by it: the check
    /// exists so that the one case where tile-granularity and pixel-granularity differ
    /// cannot pass silently.
    #[cfg(feature = "gpu")]
    const TILE: u32 = 32;

    /// This draw's scissor rectangle in pixels of a `w` x `h` attachment, as
    /// `(x, y, width, height)`, or `None` for "the whole attachment".
    ///
    /// The GXM rectangle is INCLUSIVE at both ends, so the width is `xMax - xMin + 1`.
    /// Clamped into the attachment: wgpu rejects a scissor that leaves it, and a rejected
    /// pass loses every draw in it, which is a far worse failure than a clamped rectangle.
    #[cfg(feature = "gpu")]
    fn rect_in(&self, w: u32, h: u32) -> Option<(u32, u32, u32, u32)> {
        match self.mode & 0xC000_0000 {
            Self::NONE => None,
            Self::ALL => Some((0, 0, 0, 0)),
            // >>> A RECTANGLE THAT DOES NOT FIT THIS TARGET WAS NOT WRITTEN FOR IT.
            //
            // The guest's region clip is CONTEXT state, not scene state, so a rectangle set for
            // one render target is still in the context when the next scene begins. One retail
            // title paginates a 1024x512 atlas through region clips and then draws its whole
            // FIGHT into a 640x368 display surface, and on some runs the atlas's last rectangle
            // - `0,0 .. 1023,127` - is what the display pass inherits. Applied, it scissors the
            // frame to its top 128 rows: the world, the HUD and the composite all stop at a
            // hard horizontal line two thirds up, which is exactly the picture that arrives.
            //
            // A rectangle whose far edge is outside the target cannot be a statement about the
            // target, so it is not honoured. Clamping instead keeps the narrowing axis and the
            // broken frame with it; ignoring costs nothing where the clip was meant to cover
            // everything anyway, which is the only other way a rectangle overflows.
            // >>> BUT A COORDINATE THAT NAMES NO SURFACE AT ALL IS NOT A FOREIGN RECTANGLE.
            //
            // `sceGxmSetRegionClip` takes its four bounds as `unsigned int` and this engine
            // stores them as the guest passed them. A title that computes `xMax` from a
            // subtraction that goes negative therefore hands over a WRAPPED value, and one
            // does: MEASURED on a retail golf title, 698 clips of the shape
            // `0,139 .. 4294966581,304` - `xMax` is -715. That is not a rectangle written for
            // some other render target, it is the guest's own arithmetic underflowing, and the
            // hardware truncates it to its control word's field width and clips to the surface.
            // Refusing it as foreign un-scissored a horizontal band the title's menu draws
            // across the full width, and changed the picture ([[vitaslop-a-region-clip-outlives-
            // its-scene]] is about the OTHER case and stays).
            //
            // The two are separable without a guess: no GXM render target exceeds
            // [`MAX_SURFACE_EXTENT`], so a bound at or above it cannot be describing a target,
            // while `1023` on a 640-wide one plainly can. Below the cap and outside the target,
            // the rectangle belongs to a different scene and is refused; at or above it, the
            // coordinate is garbage and is CLAMPED, which is what the hardware does with it.
            _ if (self.rect[2] >= w || self.rect[3] >= h)
                && self.rect[2] < MAX_SURFACE_EXTENT
                && self.rect[3] < MAX_SURFACE_EXTENT =>
            {
                report_region_clip_foreign(*self, w, h);
                None
            }
            _ => {
                let [x0, y0, x1, y1] = self.rect;
                let x = x0.min(w);
                let y = y0.min(h);
                // `x1`/`y1` are inclusive; `+1` cannot overflow a sane rectangle, and a
                // saturating add keeps a garbage one from wrapping to an empty scissor.
                let right = x1.saturating_add(1).min(w);
                let bottom = y1.saturating_add(1).min(h);
                Some((x, y, right.saturating_sub(x), bottom.saturating_sub(y)))
            }
        }
    }
}

/// `SCE_GXM_VIEWPORT_DISABLED` - the guest asked for NO viewport transform, and the draw
/// covers the whole render target whatever the viewport registers still hold.
///
/// `SCE_GXM_VIEWPORT_ENABLED` is **0**, which is also the context default, so a title that
/// never calls `sceGxmSetViewportEnable` is enabled throughout - the same shape as
/// `SCE_GXM_FRAGMENT_PROGRAM_ENABLED`.
pub const SCE_GXM_VIEWPORT_DISABLED: u32 = 0x0001_0000;

/// The guest's `sceGxmSetViewport` as a RECTANGLE of a `w` x `h` render target -
/// `(x, y, width, height)` in pixels, or `None` for "the whole target".
///
/// # Why this is one function and not one per backend
/// GXM states a viewport as an OFFSET and a SCALE per axis, in pixels, where the scale is a
/// HALF-extent: `screen = offset + scale * ndc`. Read the other way round - scale as a full
/// extent - a draw covers twice its rectangle, which on a half-width viewport is the whole
/// target and looks like a viewport that was never applied at all. That is exactly what the
/// conformance app's viewport scene was built to catch, and BOTH backends were failing it: the
/// software rasteriser mapped an `Ndc` vertex straight onto the full target, and the GPU
/// fixed-function arm passed `full` because "the fixed-function path has never carried a
/// viewport". A second transcription of the convention is how the two would drift again, so the
/// software rasteriser, the GPU recompiled arm and the GPU fixed-function arm all read the
/// rectangle from HERE.
///
/// `enable` is the raw `SceGxmViewportMode` word: [`SCE_GXM_VIEWPORT_DISABLED`] means the guest
/// turned the transform off and the registers are stale state, not a request.
///
/// The Y flip is NOT in the rectangle: GXM's `yScale` is normally NEGATIVE (ndc +1 is the TOP
/// of the screen) and a wgpu viewport cannot express a vertical flip, so the rectangle carries
/// `|yScale|` and each backend applies the sense itself - the recompiled arm through
/// `inject_clip_fixup`, the software rasteriser directly in its own projection, which is why
/// only the software path is exact for a POSITIVE `yScale`.
pub fn viewport_rect(
    enable: u32,
    vp: &[f32; 6],
    w: u32,
    h: u32,
) -> Option<(f32, f32, f32, f32)> {
    let [xo, xs, yo, ys, _, _] = *vp;
    if enable == SCE_GXM_VIEWPORT_DISABLED {
        return None;
    }
    // All-zero is the sentinel for "the guest never called sceGxmSetViewport", not a
    // request for a zero-area viewport. A zero SCALE with a nonzero offset would be a
    // degenerate viewport and is reported below rather than silently ignored.
    if vp.iter().all(|v| *v == 0.0) {
        return None;
    }
    if !vp.iter().all(|v| v.is_finite()) {
        report_viewport_problem(vp, "it contains a non-finite component");
        return None;
    }
    if ys > 0.0 {
        // ndc +1 lands at the BOTTOM of the rect. wgpu's viewport cannot express a vertical
        // flip (it requires a positive height), so on the recompiled arm the CLIP FIXUP does
        // it instead - see `inject_clip_fixup`, which negates clip y for exactly these draws.
        // The rect below takes `|yScale|`, which is then the right extent.
        //
        // Still reported, because "this pass is mirrored relative to a wgpu viewport" is a
        // fact about the guest worth seeing once, and because if the correction is ever
        // disabled this line is the only thing that names the passes it was carrying.
        report_viewport_problem(vp, "yScale is POSITIVE - ndc +1 is the BOTTOM of this rect, which a wgpu viewport cannot express; the clip fixup negates y for these draws instead");
    }
    let (x, y) = (xo - xs.abs(), yo - ys.abs());
    let (vw, vh) = (2.0 * xs.abs(), 2.0 * ys.abs());
    if vw <= 0.0 || vh <= 0.0 {
        report_viewport_problem(vp, "it has a zero-area rectangle");
        return None;
    }
    // Already the whole target: setting it is a no-op and skipping it keeps the common
    // fullscreen case free of state changes.
    if x == 0.0 && y == 0.0 && vw == w as f32 && vh == h as f32 {
        return None;
    }
    // wgpu rejects a viewport that leaves the attachment. Clamp INTO the target and say
    // so: rendering the pass at the wrong rect is wrong, but dropping the viewport
    // entirely is wrong in a way that looks like nothing happened.
    let (cx, cy) = (x.max(0.0), y.max(0.0));
    let (cw, ch) = ((vw + x.min(0.0)).min(w as f32 - cx), (vh + y.min(0.0)).min(h as f32 - cy));
    if (cx, cy, cw, ch) != (x, y, vw, vh) {
        report_viewport_problem(
            vp,
            &format!(
                "it asks for ({x}, {y}, {vw}, {vh}) which leaves a {w}x{h} target - \
                 CLAMPED to ({cx}, {cy}, {cw}, {ch})"
            ),
        );
    }
    if cw <= 0.0 || ch <= 0.0 {
        return None;
    }
    Some((cx, cy, cw, ch))
}

/// `VITASLOP_GXM_VIEWPORT=0` - the NEGATIVE CONTROL for the guest viewport reaching the
/// FIXED-FUNCTION GPU arm and the SOFTWARE rasteriser, which is exactly what this project
/// changed when the GXM pipeline-conformance app's viewport scene failed on both.
///
/// It is scoped to those two paths on purpose, and that scoping is the whole point: the
/// RECOMPILED arm has always applied the viewport, so a flag that also switched that off would
/// not be a control for the change - it would be a third behaviour nothing ever shipped, and a
/// picture A/B against it could not attribute a single pixel.
///
/// A rendering change that cannot be turned off cannot be measured, and this one moves where
/// every clip-space draw of every title lands. Not a quality setting: the guest's viewport is
/// the guest's, and the DEFAULT is to honour it everywhere.
pub fn viewport_reaches_fixed_function() -> bool {
    use std::sync::OnceLock;
    static ON: OnceLock<bool> = OnceLock::new();
    *ON.get_or_init(|| crate::knobs::var("VITASLOP_GXM_VIEWPORT").ok().as_deref() != Some("0"))
}

/// Say so, once per distinct (viewport, complaint), that a guest viewport could not be applied
/// as written. See [`viewport_rect`].
fn report_viewport_problem(vp: &[f32; 6], what: &str) {
    use std::collections::HashSet;
    use std::sync::{Mutex, OnceLock};
    static SEEN: OnceLock<Mutex<HashSet<(String, String)>>> = OnceLock::new();
    let seen = SEEN.get_or_init(|| Mutex::new(HashSet::new()));
    let mut seen = seen.lock().unwrap_or_else(|e| e.into_inner());
    let vps = format!("{vp:?}");
    if seen.insert((vps.clone(), what.to_string())) {
        report!("gxm viewport {vps}: {what}");
    }
}

/// Report - once per distinct rectangle and target - that a region clip was NOT honoured because
/// its rectangle lies outside the render target it reached. See [`RegionClip::rect_in`].
///
/// This is a finding, not a heartbeat: it means the guest's context still holds a rectangle from
/// an earlier scene, and every draw of this pass would have been scissored by a rectangle
/// written for a differently-sized target.
#[cfg(feature = "gpu")]
fn report_region_clip_foreign(clip: RegionClip, w: u32, h: u32) {
    use std::collections::HashSet;
    use std::sync::Mutex;
    static SEEN: Mutex<Option<HashSet<(u32, [u32; 4], u32, u32)>>> = Mutex::new(None);
    let mut g = SEEN.lock().unwrap_or_else(|e| e.into_inner());
    if !g.get_or_insert_with(HashSet::new).insert((clip.mode, clip.rect, w, h)) {
        return;
    }
    report_warn!(
        "gxm region clip: mode {:#x} over {},{} .. {},{} reached a {w}x{h} target - its far edge is OUTSIDE that target, so the rectangle was written for a different one and is NOT honoured. The guest's region clip is context state and survives a scene change; applying this one scissors the whole pass to a rectangle it never meant.",
        clip.mode, clip.rect[0], clip.rect[1], clip.rect[2], clip.rect[3]
    );
}

/// Name each distinct region clip that reaches a draw, and say whether it actually narrows
/// anything.
///
/// A scissor that equals the whole target issues no `set_scissor_rect` at all (the pass
/// already starts there), so `scissor_sets` reads zero for a title that sets a whole-target
/// default and never narrows it - which is indistinguishable from a scissor that never
/// reached the renderer. One line per distinct rectangle tells those apart, and it is the
/// only way to know the capture is carrying the state at all.
#[cfg(feature = "gpu")]
fn report_region_clip_applied(clip: RegionClip, w: u32, h: u32) {
    use std::collections::HashSet;
    use std::sync::Mutex;
    let (mode, rect) = (clip.mode, clip.rect);
    if mode & 0xC000_0000 == RegionClip::NONE {
        return;
    }
    static SEEN: Mutex<Option<HashSet<(u32, [u32; 4])>>> = Mutex::new(None);
    let mut g = SEEN.lock().unwrap_or_else(|e| e.into_inner());
    if !g.get_or_insert_with(HashSet::new).insert((mode, rect)) {
        return;
    }
    let whole = rect[0] == 0 && rect[1] == 0 && rect[2] + 1 >= w && rect[3] + 1 >= h;
    // ONE line for the whole run, not one per rectangle. The stated purpose is to prove the
    // capture is carrying scissor state at all - `scissor_sets` reads zero both for a title
    // that only ever sets a whole-target default and for a scissor that never reached the
    // renderer, and one line tells those apart. That purpose is served by the FIRST rectangle;
    // the rest are normal operation. Deduping per (mode, rect) instead was written expecting
    // "a title with a stable scissor pays one line", which a real title refutes: one paginates
    // a 128x128 atlas through eight distinct rectangles and paid eight.
    //
    // The two rectangles that are NOT merely informational still report every time they are
    // seen, below - narrowing this line does not narrow those.
    static ANNOUNCED: Mutex<bool> = Mutex::new(false);
    let mut announced = ANNOUNCED.lock().unwrap_or_else(|e| e.into_inner());
    if !*announced {
        *announced = true;
        report_status!(
            "gxm region clip: state is LIVE - first rectangle mode {mode:#x} over {},{} .. {},{} \
             on a {w}x{h} target - {}. Further rectangles are normal operation and are not \
             printed; an ALL or a non-tile-aligned OUTSIDE still reports.",
            rect[0], rect[1], rect[2], rect[3],
            if whole { "the WHOLE target, so no scissor is issued" } else { "SCISSORED" }
        );
    }
    drop(announced);
    // The two cases that are not merely informational, checked here rather than in
    // `RegionClip::rect_in` so that the pure geometry stays pure and neither check costs a
    // lock on the per-draw path.
    if mode & 0xC000_0000 == RegionClip::ALL {
        report_region_clip_all(rect);
    }
    if mode & 0xC000_0000 == RegionClip::OUTSIDE
        && (rect[0] % RegionClip::TILE != 0
            || rect[1] % RegionClip::TILE != 0
            || (rect[2] + 1) % RegionClip::TILE != 0
            || (rect[3] + 1) % RegionClip::TILE != 0)
    {
        report_region_clip_not_tile_aligned(rect);
    }
}

/// Say so that an `OUTSIDE` region clip arrived with an edge off a 32-pixel boundary.
///
/// # What this check was for, and what it has now ANSWERED
/// [`RegionClip`]'s doc records the inference this was built to test: that the two enabled
/// modes both keep the inside of the rectangle and differ in GRANULARITY, because every
/// `OUTSIDE` rectangle measured on the two titles available at the time was a multiple of 32
/// while the `INSIDE` ones were arbitrary. If that inference held, an unaligned `OUTSIDE`
/// rectangle would be the one case where "clip whole tiles" and "clip exactly" paint
/// different pixels, so it had to fire rather than pass silently.
///
/// **A third title refutes the granularity half of it.** One retail sports title issues
/// **512 distinct `OUTSIDE` rectangles in a single round**, nearly all unaligned, and their
/// shape names what they are: centred bands that shrink and grow by ONE pixel a frame over a
/// mip chain of 192, 160, 128, 96 and 64-pixel targets - `0,33..191,157`, `0,31..191,159`,
/// `0,38..191,152`, and so on. Nobody authors a smooth per-pixel animation through a scissor
/// that quantises to 32 pixels; the title plainly expects, and gets, an exact rectangle. So
/// `OUTSIDE` is a pixel-exact scissor, this renderer already applies it as one, and the
/// picture is right.
///
/// The check is KEPT because the census it now produces is still worth having (it is how the
/// above was measured at all), but it is no longer one warning per rectangle: see [`Census`]
/// for what 512 of those did to the only diagnostics channel a phone has.
#[cfg(feature = "gpu")]
fn report_region_clip_not_tile_aligned(rect: [u32; 4]) {
    use std::collections::HashSet;
    use std::sync::Mutex;
    static SEEN: Mutex<Option<HashSet<[u32; 4]>>> = Mutex::new(None);
    let mut g = SEEN.lock().unwrap_or_else(|e| e.into_inner());
    if !g.get_or_insert_with(HashSet::new).insert(rect) {
        return;
    }
    drop(g);
    static CENSUS: Census = Census::new();
    CENSUS.note(
        "gxm region clip: an OUTSIDE rectangle is not a multiple of 32, so tile-granular and \
         pixel-exact clipping would differ here - it is applied EXACTLY, which is what a \
         title animating a scissor one pixel at a time asks for. Rectangle:",
        &format!("{},{} .. {},{}", rect[0], rect[1], rect[2], rect[3]),
    );
}

/// Say so, once, that a draw asked for `SCE_GXM_REGION_CLIP_ALL`. It clips inside AND
/// outside the region, so nothing rasterises - which is indistinguishable from a bug
/// unless it is announced.
#[cfg(feature = "gpu")]
fn report_region_clip_all(rect: [u32; 4]) {
    use std::sync::atomic::{AtomicBool, Ordering};
    static SAID: AtomicBool = AtomicBool::new(false);
    if SAID.swap(true, Ordering::Relaxed) {
        return;
    }
    report_warn!(
        "gxm region clip: SCE_GXM_REGION_CLIP_ALL over {},{} .. {},{} - it clips both inside \
         and outside the region, so every draw under it rasterises NOTHING. That is what the \
         guest asked for; it is reported because an empty pass otherwise reads as a defect.",
        rect[0], rect[1], rect[2], rect[3]
    );
}

/// Report, once per run, that a draw with no fixed-function representation could not be drawn
/// by the recompiler either - so it is missing from the frame. Suppressed after the first so a
/// per-frame occurrence cannot flood the log; the first one carries what matters.
#[cfg(feature = "gpu")]
pub(crate) fn report_shader_only_skip(d: &GxmDraw, gxp_enabled: bool) {
    use std::sync::atomic::{AtomicBool, Ordering};
    static SAID: AtomicBool = AtomicBool::new(false);
    if SAID.swap(true, Ordering::Relaxed) {
        return;
    }
    // At WARN, and this one is not a style point. A title's FINAL COMPOSITE is a fullscreen
    // blit whose colour lives entirely in its fragment shader, so it is shader-only: without
    // `VITASLOP_GXP_LIVE` the whole visible frame is this one skipped draw, the headless shot
    // comes back as the flat CLEAR COLOUR, and at `debug` nothing said so. Every offscreen
    // pass still renders, so `VITASLOP_GPU_CHAIN_DIR` shows a correct world target beside a
    // blank presented frame - which reads as "the composite is broken" and is really "the
    // recompiler was never switched on". The message already names the knob; it just could
    // not be seen at the filter every documented repro command uses.
    report_warn!(
        "render: a draw with NO fixed-function representation (position-only geometry, its \
         colour lives in the guest fragment shader) is MISSING from the frame - {}. \
         {} indices, {} vertex bytes.",
        if !gxp_enabled {
            "the GXP recompiler is not enabled (set VITASLOP_GXP_LIVE)"
        } else if d.gxp.is_none() {
            "the runtime captured no shader payload for it"
        } else {
            "its shader pair did not link (the fallback above names the reason)"
        },
        d.index_count,
        d.vertices.len(),
    );
}

/// Everything the GXP->WGSL recompiler needs to draw one call with the guest's real
/// vertex+fragment shaders, snapshotted by the runtime and carried as plain data (no GPU or
/// runtime dependency). The renderer links the pair on first sight, caches the pipeline by
/// shader identity, and ALWAYS falls back to the fixed-function path on any link/format
/// error - a wrong translation never paints a pixel.
#[derive(Clone, Debug)]
pub struct GxpRecompile {
    /// The vertex `SceGxmProgram` container bytes. Shared, not owned: the capture reads each
    /// container out of guest memory once and every draw bound to it carries the same `Arc`,
    /// so building a frame's draw list does not copy a few kilobytes per draw.
    pub vprog: std::sync::Arc<[u8]>,
    /// The fragment `SceGxmProgram` container bytes. Shared for the same reason as `vprog`.
    pub fprog: std::sync::Arc<[u8]>,
    /// The vertex program the fragment was PATCHED against, when it is not `vprog`; empty
    /// otherwise. See `capture::Draw::fprog_patched_vprog` and `LinkOptions::patched_against`.
    pub fprog_patched_vprog: std::sync::Arc<[u8]>,
    /// Raw vertex default-uniform-buffer (SA bank) bytes, as the guest wrote them.
    /// Shared, not owned - see `capture::Draw::vert_sa` for the measurement behind that.
    pub vert_sa: std::sync::Arc<[u8]>,
    /// Raw fragment default-uniform-buffer (SA bank) bytes, as the guest wrote them. Shared.
    pub frag_sa: std::sync::Arc<[u8]>,
    /// Guest address `frag_sa` was read from (0 = none). See `capture::Draw::frag_sa_addr`:
    /// the bytes say what the draw got, the address is what a store watch can be pointed at.
    pub frag_sa_addr: u32,
    /// The guest-memory WINDOWS the recompiled vertex shader's 0xE8 memory loads read
    /// through, in the order the shader's `gxp_mem` binding lays them out:
    /// `(window guest base address, the window's bytes at draw time)` each. EMPTY for a
    /// program that loads no memory; a draw whose PIPELINE declares windows but carries none
    /// here is DROPPED with a report rather than fed fabricated bytes.
    pub mem_windows: Vec<(u32, std::sync::Arc<[u8]>)>,
    /// The same, for the FRAGMENT stage's own `gxp_fmem` binding, snapshotted from the
    /// FRAGMENT uniform-buffer table. Empty for the overwhelming majority of programs.
    pub frag_mem_windows: Vec<(u32, std::sync::Arc<[u8]>)>,
    /// The buffers the FRAGMENT program's guest-memory STORES write through, in the order its
    /// store binding lays them out (`vitaslop_gxp_shader::module::resolve_store_windows`):
    /// `(guest base address, the guest bytes there at draw time)` each - the bytes SEED the
    /// renderer's writable mirror of that memory (see `GxmRenderer` `store_mirrors`). Empty for
    /// every program that does not store.
    pub frag_store_windows: Vec<(u32, std::sync::Arc<[u8]>)>,
    /// Raw guest vertex stream bytes (stream 0) exactly as bound.
    ///
    /// Shared with the capture that snapshotted it rather than copied: this is the whole
    /// mesh of one draw, and copying it per draw per frame measured 2.4-3.2 MB of
    /// allocate-and-free every frame on a 500-700 draw scene. Sharing also gives the
    /// buffer an identity anything derived from it can be cached against.
    pub vertices: std::sync::Arc<[u8]>,
    /// Byte stride of one guest vertex within `vertices`.
    pub vertex_stride: u32,
    /// Guest vertex attributes: stream byte offset + raw GXM format + component count, keyed
    /// to the recompiler's vertex-input `@location` by `reg_index` (the attribute base lane).
    /// SHARED, not owned - and the sharing is with the vertex PROGRAM. The list is a pure
    /// function of the guest's declared attributes, which are fixed when the vertex program is
    /// created, so building a fresh `Vec` per draw allocated and copied a CONSTANT hundreds of
    /// times a frame. See `RenderSceneBuilder::gxp_attributes`.
    pub attributes: std::sync::Arc<[GxpAttr]>,
    /// The draw's index buffer, already expanded to a flat winding-normalized triangle-LIST
    /// of `u32`s. Shared, and CACHED by the builder against the guest index buffer it was
    /// expanded from: the guest's own index bytes do not change from frame to frame for
    /// static geometry, so expanding them again every frame is pure repetition.
    pub indices: std::sync::Arc<[u8]>,
    /// Number of indices.
    pub index_count: u32,
    /// True = 32-bit indices, false = 16-bit (GXM index format 0).
    pub index_u32: bool,
    /// GXM primitive type word (drives the pipeline topology).
    pub primitive: u32,
    /// Decoded textures bound per fragment sampler unit.
    /// SHARED, not owned, for the same reason [`crate::gpu::GxmDraw`]'s capture-side list is
    /// (`capture::Draw::textures`): for a fixed set of bindings every draw of a scene produces
    /// a bitwise identical list, and the capture already hands those draws ONE `Arc`. Deriving
    /// a fresh `Vec` from it per draw threw that away - **3,782 decode-cache lookups and 672
    /// allocations a frame** on a retail sports title, to rebuild a list the previous draw had.
    /// See `RenderSceneBuilder::gxp_textures`.
    pub textures: std::sync::Arc<[GxpTex]>,
    /// Decoded textures bound per VERTEX sampler unit. Separate list, because the two stages
    /// number their units independently - and a vertex program that samples is building its
    /// geometry from what it reads, so binding the fragment's texture here draws a wrong mesh
    /// rather than shading a surface wrongly.
    /// Shared for the same reason [`Self::textures`] is, and out of the same per-set cache.
    pub vertex_textures: std::sync::Arc<[GxpTex]>,
    /// Depth write enabled for this draw (GXM `front_depth_write != DISABLED`).
    pub depth_write: bool,
    /// GXM depth-compare function word (`SceGxmDepthFunc`).
    pub depth_func: u32,
    /// GXM `sceGxmSetFrontDepthBias(factor, units)`: the polygon offset for this draw.
    /// `(0, 0)` - the default - is no bias, which is what most draws carry.
    pub depth_bias: (i32, i32),
    /// GXM cull-mode word (`SceGxmCullMode`).
    pub cull_mode: u32,
    /// Whether the guest left the FRAGMENT PROGRAM enabled for this draw
    /// (`sceGxmSetFrontFragmentProgramEnable`). `false` is a legal and common GXM state: the
    /// draw rasterises and updates DEPTH and STENCIL and writes no colour at all. It is how a
    /// title writes a depth prepass, and how one title clears its depth buffer mid-scene -
    /// a fullscreen triangle with `DEPTH_FUNC_ALWAYS`, depth write on and the fragment program
    /// off. Ignoring it painted that clear's shader output (opaque black) over the finished
    /// world every frame.
    pub fragment_program_enabled: bool,
    /// The bound fragment program's `SceGxmProgram*`. Diagnostic only - it names which
    /// `SceGxmFragmentProgram` a draw used, which the shader bytes cannot: a title can register
    /// one fragment shader twice with different blend equations (one title's 2D primitive-render
    /// program is registered both with a NULL `blendInfo` and with straight-alpha src-over), and
    /// the two render completely differently from identical bytes.
    pub fprog_header: u32,
    /// Whether this draw is alpha-blended (a 2D/overlay draw, not opaque geometry).
    ///
    /// A HEURISTIC read off the geometry, kept only for the fixed-function path. The
    /// recompiler path uses `blend_state`, which is the guest's own answer.
    pub blend: bool,
    /// The blend equation the guest baked into this draw's fragment program, as raw GXM enum
    /// values: `[color_mask, color_func, alpha_func, color_src, color_dst, alpha_src,
    /// alpha_dst]`.
    ///
    /// Carried as a plain array rather than the capture type so this crate stays independent
    /// of the capture crate, exactly as `depth_func` and `cull_mode` are carried.
    pub blend_state: [u8; 7],
    /// GXM viewport `[xOffset,xScale,yOffset,yScale,zOffset,zScale]` mapping the guest clip
    /// output to the framebuffer. All-zero means the guest left the default (fullscreen).
    pub viewport: [f32; 6],
    /// The guest's `SceGxmViewportMode` - see [`SCE_GXM_VIEWPORT_DISABLED`]. Carried beside
    /// `viewport` because the two are only meaningful together: the registers keep their last
    /// values after `sceGxmSetViewportEnable(DISABLED)`, so a rectangle applied without
    /// consulting this is a transform the guest switched off.
    pub viewport_enable: u32,
    /// The draw's STENCIL state, straight from the guest's `sceGxmSet{Front,Back}StencilFunc`
    /// and `...StencilRef`: `[func, fail_op, depth_fail_op, depth_pass_op, compare_mask,
    /// write_mask, ref]` for the FRONT face, then the same seven for the BACK face. Raw GXM
    /// enums; see [`gxm_stencil_state`].
    pub stencil: [u32; 14],
    /// `SCE_GXM_TWO_SIDED_ENABLED`: with it off the hardware applies the FRONT stencil state
    /// to both faces, and the back words above are recorded but not consulted.
    pub two_sided: bool,
}

/// One guest vertex attribute for the recompiler path: where it sits in the stream and its
/// raw GXM format, so the pipeline builds a matching `wgpu` vertex layout.
#[derive(Clone, Copy, Debug)]
pub struct GxpAttr {
    /// The vertex program's attribute resource index = the recompiler's `@location` base lane.
    pub reg_index: u16,
    /// Byte offset of this attribute within a vertex.
    pub offset: u16,
    /// Raw `SceGxmAttributeFormat` value.
    pub gxm_format: u8,
    /// Component count (1..4).
    pub components: u8,
}

/// A decoded texture bound to a specific fragment sampler unit for the recompiler path.
#[derive(Clone, Debug)]
pub struct GxpTex {
    /// The sampler unit the fragment program samples this from (`t{unit}`/`s{unit}`).
    pub unit: u8,
    /// The decoded, GPU-ready texture.
    pub tex: GxmTexture,
    /// The texture's own LOD bias, in mip levels, which every implicit-LOD fragment sample of
    /// this unit adds (the shader reads it from the per-draw `@group(3)` block). Per BINDING, not
    /// on [`GxmTexture`]: one decoded image is bound at many biases, and the decode cache shares it.
    pub lod_bias: f32,
}

/// Stride of the canonical [`GxmDraw`] vertex: pos(12) + uv(8) + color(4) + world-normal(12).
pub const GXM_VERTEX_STRIDE: u64 = 36;

/// Where a scene's pixels land: the guest address of its colour surface and that
/// surface's size. A frame is usually several scenes - offscreen passes that render
/// shadow maps, reflections, the 3D world - followed by one that composites them onto
/// the display buffer, and the composite samples the earlier ones as TEXTURES. Matching
/// a sampled texture's data pointer against these is what lets the renderer bind what it
/// actually drew instead of the guest bytes, which the GPU never wrote.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub struct RttTarget {
    pub data_addr: u32,
    pub width: u32,
    pub height: u32,
    /// The guest's `SceGxmColorFormat` of the surface, verbatim. Decides whether the pass is
    /// rendered through the ordinary 8-bit UNORM attachment or, for a 64-bit format, as RAW
    /// words into an `Rg32Uint` one - see [`raw64_color_format`].
    pub format: u32,
    /// The guest asked for GAMMA-CORRECT writes on this surface
    /// (`sceGxmColorSurfaceSetGammaMode`), so the hardware sRGB-encodes every value the ROP
    /// stores. Rendering through an sRGB view of the same texture reproduces that exactly -
    /// including doing it AFTER blending, which is where the hardware does it too.
    pub gamma: bool,
    /// The guest asked for `SCE_GXM_COLOR_SURFACE_SCALE_MSAA_DOWNSCALE` on this surface, which
    /// means the surface STORES the resolved image: its stored size is the sampled size, and
    /// the pass behind it was rasterised with more samples than that per pixel. How many is
    /// [`Self::multisample`] - the two are halves of one setting and neither is readable alone.
    pub msaa_downscale: bool,
    /// The `SceGxmMultisampleMode` of the render target this pass rasterises through:
    /// 0 = NONE, 1 = 2X, 2 = 4X. See [`gxm_sample_count`] for the mapping to a GPU sample
    /// count and for why 2X cannot be reproduced exactly.
    pub multisample: u32,
    /// The colour surface's row pitch in PIXELS (`strideInPixels`), 0 when unknown.
    pub stride_px: u32,
}

/// The GPU sample count that reproduces a `SceGxmMultisampleMode`.
///
/// # This is MSAA, and the difference from supersampling is the whole point
/// Multisampling rasterises COVERAGE and DEPTH at N samples per pixel but invokes the
/// fragment shader ONCE per pixel, at the pixel centre, and gives every covered sample that
/// one result. Supersampling invokes it once per SAMPLE. On an ordinary colour image the two
/// look nearly identical, which is why rendering a 2x2 image and box-averaging it passed for
/// MSAA here for a while. They are not the same program:
///
/// - Inside a triangle, MSAA's N samples are N copies of one value, so the resolve returns it
///   EXACTLY. Supersampling's N samples are the shader evaluated at N different positions, so
///   the resolve returns their mean - which equals the centre value only when the shader is
///   linear in position.
/// - MEASURED, and it is why the 2x2 raster had to be left off: this title's 1024x1024 shadow
///   map is a fragment program writing a DEPTH into a colour surface. Averaging four depths
///   sampled at four positions gives a depth belonging to none of them, so the shadow
///   comparison biased towards occluded and the track's dappled sunlight flattened into
///   uniform shade over 17.6% of the frame. Under MSAA that surface's interior is bit-exact,
///   because there is one shaded value per pixel to begin with.
///
/// So the guest's request is reproduced by a real multisampled attachment, not by a finer
/// raster. That also makes it CHEAPER than the 2x2 path it replaces: one shader invocation
/// per pixel instead of four.
///
/// # 2X is an approximation and says so at the call site
/// WebGPU guarantees sample counts 1 and 4 only, so `SCE_GXM_MULTISAMPLE_2X` is served with 4
/// samples: more antialiasing than the title asked for rather than less, and never a
/// different image where the shader is what MSAA promises it is (one invocation per pixel).
/// Whether a guest `SceGxmColorFormat` is a 64-bit surface, which this renderer stores as the
/// fragment's RAW register pair in an `Rg32Uint` attachment rather than as a converted colour.
///
/// The hardware stores a fragment's first two output registers into such a surface as they
/// are - a program writing packed halves gets an F16F16F16F16 image, one writing packed bytes
/// gets those bytes - and a sampler declared over the same memory reads whatever bits are
/// there. MEASURED on a baseball title: its crowd impostor atlas is rendered into an
/// F16F16F16F16 surface by a program that packs RGBA8 pairs into its colour registers, and the
/// crowd reads it back as a U32U32 texture. Through an 8-bit UNORM attachment those bits
/// were float-converted and clamped, the atlas held nothing usable, and the stands were empty.
/// The 32-bit floating formats (F16F16, F32, F11F11F10, U2F10F10F10...) are NOT this: they
/// are converted colours on an `Rgba16Float` attachment - see [`float_color_format`].
pub fn raw64_color_format(format: u32) -> bool {
    // F16F16F16F16 and F32F32 - the two 64-bit members of the colour-format enum.
    matches!(format & 0xff80_0000, 0x0100_0000 | 0x1100_0000)
}

/// Whether a guest `SceGxmColorFormat` is F11F11F10, the packed FLOATING-POINT format a title
/// renders HDR light and exposure into. It is rendered into an `Rgba16Float` attachment
/// ([`FLOAT_TARGET_FORMAT`]) so values above 1.0 survive; `VITASLOP_GXM_FLOAT_TARGETS=0` is the
/// arm back to the 8-bit UNORM attachment every other 32-bit format uses.
///
/// MEASURED on a fighting title: its exposure chain renders a luminance reduction through
/// 16x96 and 32x192 F11F11F10 targets. Clamped to 8 bits there, the average it measured was
/// wrong by whatever lay above 1.0 and the tone-map it drove blew the stage out to flat cyan.
///
/// One difference remains, named: the format has no alpha channel, so on the console a sample
/// of it reads alpha 1.0, while this attachment keeps whatever alpha the fragment wrote.
///
/// # The other 32-bit float formats take the same attachment
/// `U2F10F10F10`, `F16`, `F16F16` and `F32` are HDR surfaces too, and on the UNORM attachment
/// they clamped exactly as F11F11F10 did. MEASURED on an action title: it renders
/// its whole world into a 720x408 `U2F10F10F10` target and its exposure through 32x32 `F32`
/// ones; clamped, the tone-map read a world capped at 1.0 against a clamped average and the
/// night prologue came out near-black. The 10-bit unsigned floats and F16 are EXACT in f16 (5
/// exponent bits, fewer mantissa bits); F32 keeps its range but rounds to f16's 11-bit
/// significand - a named difference from the console, far smaller than a clamp at 1.0. As
/// with F11F11F10, `U2F10F10F10`'s 2-bit alpha is kept at the fragment's full precision.
/// `VITASLOP_GXM_FLOAT_TARGETS_WIDE=0` is the arm back to F11F11F10 alone.
pub fn float_color_format(format: u32) -> bool {
    use std::sync::OnceLock;
    static ON: OnceLock<bool> = OnceLock::new();
    static WIDE: OnceLock<bool> = OnceLock::new();
    let member = match format & 0xff80_0000 {
        0x2100_0000 => true,
        0x4100_0000 | 0xF000_0000 | 0x0080_0000 | 0x1080_0000 => *WIDE.get_or_init(|| {
            crate::knobs::var("VITASLOP_GXM_FLOAT_TARGETS_WIDE").map(|v| v.trim() != "0").unwrap_or(true)
        }),
        _ => false,
    };
    member && *ON.get_or_init(|| crate::knobs::var("VITASLOP_GXM_FLOAT_TARGETS").map(|v| v.trim() != "0").unwrap_or(true))
}

/// Whether a guest `SceGxmColorFormat` is F11F11F10 held in a float attachment - a surface with
/// NO ALPHA CHANNEL, which on the console samples as alpha 1.0 whatever the fragment wrote. See
/// `GxmRenderer::fill_alpha_one`.
pub fn no_alpha_float_format(format: u32) -> bool {
    format & 0xff80_0000 == 0x2100_0000 && float_color_format(format)
}

/// Whether a guest `SceGxmColorFormat` is a SINGLE-CHANNEL surface that stores the fragment's
/// ALPHA (`U8_A`, `S8_A`: the U8/S8 base with `SWIZZLE1_A`, bit 20).
///
/// The renderer holds every 32-bit-or-narrower target as RGBA8 and treats the single-channel
/// family's one channel as RED - that is what its write-back encodes and what a U8 texture
/// reads back (see `rtt_writeback::encode_row`). So for an `_A` surface the fragment's alpha has
/// to be the value that lands in red: the link splats it (`LinkOptions::alpha_to_red`), the
/// pipeline blends that channel by the guest's ALPHA equation, and the guest's alpha write bit
/// gates it. MEASURED on a fighting title: its fight renders four draws a frame into
/// a 128x256 `U8_A` target (0x80900000), which stored red instead.
/// `VITASLOP_GXM_ALPHA_SINGLE=0` is the arm back to storing red.
/// The bytes one texel of a guest `SceGxmColorFormat` occupies in memory - what turns a
/// texture address INSIDE a render target into the texel it names. `None` for a base this
/// renderer has not established (packed U8U8U8), which then gets no window.
///
/// MEASURED: a platformer renders its ambient-occlusion atlas into a `U16` surface (2 bytes a
/// texel) and samples a 384x384 quadrant of it at `base + 768` bytes. Read at four bytes a
/// texel that quadrant was "not a window" (its row bytes were half the pitch's), the sampler
/// decoded guest bytes the GPU never wrote, and every lit character rendered black.
pub fn gxm_color_bytes_per_pixel(format: u32) -> Option<u32> {
    Some(match format & 0xF180_0000 {
        0x0000_0000 | 0x0080_0000 | 0x1080_0000 | 0x3080_0000 | 0x5080_0000 | 0x6080_0000 | 0xD080_0000
        | 0xE080_0000 | 0x2100_0000 | 0x3100_0000 | 0x4100_0000 => 4,
        0x3000_0000 | 0x4000_0000 | 0x5000_0000 | 0x6000_0000 | 0xF000_0000 | 0x2080_0000 | 0x4080_0000
        | 0xA080_0000 | 0xB080_0000 | 0xC080_0000 => 2,
        0x8080_0000 | 0x9080_0000 => 1,
        0x0100_0000 | 0x1100_0000 => 8,
        _ => return None,
    })
}

pub fn alpha_single_color_format(format: u32) -> bool {
    use std::sync::OnceLock;
    static ON: OnceLock<bool> = OnceLock::new();
    matches!(format & 0xF180_0000, 0x8080_0000 | 0x9080_0000)
        && format & 0x0010_0000 != 0
        && *ON.get_or_init(|| crate::knobs::var("VITASLOP_GXM_ALPHA_SINGLE").map(|v| v.trim() != "0").unwrap_or(true))
}

/// Whether a held render target is refused as a sampler source when this frame has written its
/// address at a DIFFERENT extent (see the `live_extents` rule in the pass encoder).
/// `VITASLOP_RTT_STALE_EXTENT=0` is the arm back to offering it.
#[cfg(feature = "gpu")]
pub(crate) fn stale_extent_guard() -> bool {
    use std::sync::OnceLock;
    static ON: OnceLock<bool> = OnceLock::new();
    *ON.get_or_init(|| crate::knobs::var("VITASLOP_RTT_STALE_EXTENT").map(|v| v.trim() != "0").unwrap_or(true))
}

/// Whether a sampler naming a SUB-RECTANGLE of a rendered target (an interior address with the
/// target's own pitch) gets that window rather than the whole target - see `encode_pass`.
/// `VITASLOP_RTT_SUBRECT=0` is the arm back.
#[cfg(feature = "gpu")]
pub(crate) fn subrect_sampling() -> bool {
    use std::sync::OnceLock;
    static ON: OnceLock<bool> = OnceLock::new();
    *ON.get_or_init(|| crate::knobs::var("VITASLOP_RTT_SUBRECT").map(|v| v.trim() != "0").unwrap_or(true))
}

/// Whether the sub-rectangle rule also covers a texture at a target's BASE address that is
/// smaller than the target, and a target held from an EARLIER frame - see `encode_pass`.
/// `VITASLOP_RTT_SUBRECT_HELD=0` is the arm back to interior windows of this frame's targets.
#[cfg(feature = "gpu")]
pub(crate) fn subrect_held_sampling() -> bool {
    use std::sync::OnceLock;
    static ON: OnceLock<bool> = OnceLock::new();
    *ON.get_or_init(|| crate::knobs::var("VITASLOP_RTT_SUBRECT_HELD").map(|v| v.trim() != "0").unwrap_or(true))
}

/// Whether an INTERIOR window of a render target held from an earlier frame is sampled from that
/// target (`VITASLOP_RTT_SUBRECT_HELD_INTERIOR=1`), rather than left to the guest's bytes.
///
/// >>> OFF BY DEFAULT, AND WHY. It is the same memory rule as the base-address window
/// [`subrect_held_sampling`] turns on, but the only reads it reaches today (MEASURED: a shooter
/// whose first pass samples three windows of the previous frame's post-process atlas into a
/// temporal history strip) feed a chain that is not yet right live - with them on the first-person
/// weapon bleaches white with a bright fringe. Turn it on by default once that chain renders right.
pub(crate) fn subrect_held_interior() -> bool {
    use std::sync::OnceLock;
    static ON: OnceLock<bool> = OnceLock::new();
    *ON.get_or_init(|| crate::knobs::var("VITASLOP_RTT_SUBRECT_HELD_INTERIOR").is_ok_and(|v| v.trim() == "1"))
}

/// `VITASLOP_GXM_TILE_SHADOW=0` is the arm back for the half-precision tile shadow of an 8-bit
/// scene whose programs blend at register precision - see `GxmRenderer::tile_shadow`.
#[cfg(feature = "gpu")]
pub(crate) fn tile_shadow_enabled() -> bool {
    use std::sync::OnceLock;
    static ON: OnceLock<bool> = OnceLock::new();
    *ON.get_or_init(|| crate::knobs::var("VITASLOP_GXM_TILE_SHADOW").map(|v| v.trim() != "0").unwrap_or(true))
}

/// `VITASLOP_RTT_SELF_WINDOW=0` is the arm back for a draw sampling a window of the target its
/// own pass is writing - see the self-window block in `encode_chain`.
#[cfg(feature = "gpu")]
pub(crate) fn self_window_sampling() -> bool {
    use std::sync::OnceLock;
    static ON: OnceLock<bool> = OnceLock::new();
    *ON.get_or_init(|| crate::knobs::var("VITASLOP_RTT_SELF_WINDOW").map(|v| v.trim() != "0").unwrap_or(true))
}

/// Say, once per address, why a sampler naming an address inside a rendered target was NOT
/// given a sub-rectangle window.
#[cfg(feature = "gpu")]
pub(crate) fn report_subrect_refused(addr: u32, base: u32, why: String) {
    use std::sync::{Mutex, OnceLock};
    static SEEN: OnceLock<Mutex<std::collections::HashSet<u32>>> = OnceLock::new();
    let seen = SEEN.get_or_init(|| Mutex::new(std::collections::HashSet::new()));
    if seen.lock().unwrap_or_else(|e| e.into_inner()).insert(addr) {
        tracing::info!(
            target: "vitaslop::render",
            "gxm rtt subrect: a sampler at {addr:#x} lies inside the target at {base:#x} but gets no window: {why}"
        );
    }
}

/// Say, once per address, that a sampler was served a sub-rectangle of a rendered target.
#[cfg(feature = "gpu")]
pub(crate) fn report_subrect(addr: u32, base: u32, x0: u32, y0: u32, w: u32, h: u32) {
    use std::sync::{Mutex, OnceLock};
    static SEEN: OnceLock<Mutex<std::collections::HashSet<u32>>> = OnceLock::new();
    let seen = SEEN.get_or_init(|| Mutex::new(std::collections::HashSet::new()));
    if seen.lock().unwrap_or_else(|e| e.into_inner()).insert(addr) {
        tracing::info!(
            target: "vitaslop::render",
            "gxm rtt subrect: a sampler at {addr:#x} reads the {w}x{h} window at ({x0}, {y0}) of the target at {base:#x} (same pitch) - bound as that window, not the whole target"
        );
    }
}

/// Say, once per address, that a draw reads a window of the target its OWN pass is writing, so
/// the pass is cut in front of it and the window copied out of the attachment.
#[cfg(feature = "gpu")]
pub(crate) fn report_self_window(addr: u32, base: u32, x0: u32, y0: u32, w: u32, h: u32) {
    use std::sync::{Mutex, OnceLock};
    static SEEN: OnceLock<Mutex<std::collections::HashSet<u32>>> = OnceLock::new();
    let seen = SEEN.get_or_init(|| Mutex::new(std::collections::HashSet::new()));
    if seen.lock().unwrap_or_else(|e| e.into_inner()).insert(addr) {
        tracing::info!(
            target: "vitaslop::render",
            "gxm rtt self-window: a sampler at {addr:#x} reads the {w}x{h} window at ({x0}, {y0}) of {base:#x}, the target its own pass is writing - the pass is cut before each such draw and the window copied from the attachment"
        );
    }
}

/// Whether a sampler reading one 32-bit word of a raw 64-bit surface as a double-width texture
/// is bound to that word view (see the pass encoder). `VITASLOP_RAW_WORD_VIEW=0` is the arm back.
/// Whether a depth-stencil surface's FORCE_LOAD / FORCE_STORE bits carry its depth between passes
/// (see `GxmRenderer::depth_saved`). `VITASLOP_GXM_ZLS=0` is the arm back.
#[cfg(feature = "gpu")]
pub(crate) fn zls_on() -> bool {
    use std::sync::OnceLock;
    static ON: OnceLock<bool> = OnceLock::new();
    *ON.get_or_init(|| crate::knobs::var("VITASLOP_GXM_ZLS").map(|v| v.trim() != "0").unwrap_or(true))
}

/// Whether a scene whose depth-stencil surface FORCE_LOADs a separate STENCIL plane that a
/// fragment program's guest-memory stores wrote loads that plane from the renderer's mirror of
/// the memory, and whether a FORCE_STORE of such a plane writes the mirror back (see
/// `GxmRenderer::load_guest_stencil`). `VITASLOP_GXM_GUEST_STENCIL=0` is the arm back: the
/// stencil starts at the surface's background value, as every build did before.
#[cfg(feature = "gpu")]
pub(crate) fn guest_stencil_on() -> bool {
    use std::sync::OnceLock;
    static ON: OnceLock<bool> = OnceLock::new();
    *ON.get_or_init(|| crate::knobs::var("VITASLOP_GXM_GUEST_STENCIL").map(|v| v.trim() != "0").unwrap_or(true))
}

/// `VITASLOP_DISPLAY_SNAPSHOT=0`: the arm back to a display pass that samples its own buffer
/// reading the guest's bytes there - see `GxmRenderer::snapshot_display`.
#[cfg(feature = "gpu")]
pub(crate) fn display_snapshot() -> bool {
    use std::sync::OnceLock;
    static ON: OnceLock<bool> = OnceLock::new();
    *ON.get_or_init(|| crate::knobs::var("VITASLOP_DISPLAY_SNAPSHOT").map(|v| v.trim() != "0").unwrap_or(true))
}

/// Say, once per depth address, that a FORCE_LOAD pass could not be given the stored depth (a
/// multisampled attachment, or a stored copy of another extent or format) and started cleared.
#[cfg(feature = "gpu")]
pub(crate) fn report_zls_unserved(addr: u32, w: u32, h: u32) {
    use std::sync::{Mutex, OnceLock};
    static SEEN: OnceLock<Mutex<std::collections::HashSet<u32>>> = OnceLock::new();
    let seen = SEEN.get_or_init(|| Mutex::new(std::collections::HashSet::new()));
    if seen.lock().unwrap_or_else(|e| e.into_inner()).insert(addr) {
        tracing::warn!(
            target: "vitaslop::render",
            "gxm depth FORCE_LOAD at {addr:#x} ({w}x{h}): a pass asks to start from the depth a FORCE_STORE pass left there, but its attachment is multisampled or of another size/format than the stored copy - it starts CLEARED instead, and its depth tests run against nothing"
        );
    }
}

#[cfg(feature = "gpu")]
pub(crate) fn raw_word_view() -> bool {
    use std::sync::OnceLock;
    static ON: OnceLock<bool> = OnceLock::new();
    *ON.get_or_init(|| crate::knobs::var("VITASLOP_RAW_WORD_VIEW").map(|v| v.trim() != "0").unwrap_or(true))
}

/// Say, once per address, that a sampler was bound to a word view of a raw 64-bit surface.
#[cfg(feature = "gpu")]
pub(crate) fn report_raw_word_view(addr: u32, base: u32, w: u32, h: u32, base_format: u32) {
    use std::sync::{Mutex, OnceLock};
    static SEEN: OnceLock<Mutex<std::collections::HashSet<u32>>> = OnceLock::new();
    let seen = SEEN.get_or_init(|| Mutex::new(std::collections::HashSet::new()));
    if seen.lock().unwrap_or_else(|e| e.into_inner()).insert(addr) {
        tracing::info!(
            target: "vitaslop::render",
            "gxm rtt raw word view: a sampler at {addr:#x} (base format {base_format:#x}) reads word {} of the raw 64-bit surface at {base:#x} ({w}x{h}) as a {}x{h} texture - bound to that word view",
            (addr - base) / 4,
            w * 2
        );
    }
}

/// `VITASLOP_MEM_WINDOW_FROM_TARGET=0`: the arm back to a shader memory window that reads the
/// guest's bytes even where this frame rendered a 64-bit target - see
/// `GxmRenderer::fill_windows_from_raw_targets`. Read by the capture too, which skips reading
/// the bytes that fill overwrites (see [`raw_window_fill_cover`]).
pub fn mem_window_from_target() -> bool {
    use std::sync::OnceLock;
    static ON: OnceLock<bool> = OnceLock::new();
    *ON.get_or_init(|| crate::knobs::var("VITASLOP_MEM_WINDOW_FROM_TARGET").map(|v| v.trim() != "0").unwrap_or(true))
}

/// `VITASLOP_MEM_WINDOW_SKIP_COVERED=0`: the arm back to reading, copying and uploading every
/// byte of a memory window the GPU then fills from a raw target (see
/// [`raw_window_fill_cover`]). On by default: those bytes are overwritten before any shader
/// reads them, so skipping them changes no byte the shader sees.
pub fn mem_window_skip_covered() -> bool {
    use std::sync::OnceLock;
    static ON: OnceLock<bool> = OnceLock::new();
    *ON.get_or_init(|| {
        mem_window_from_target()
            && crate::knobs::var("VITASLOP_MEM_WINDOW_SKIP_COVERED").map(|v| v.trim() != "0").unwrap_or(true)
    })
}

/// The row-pitch alignment a multi-row texture-to-buffer copy needs -
/// `wgpu::COPY_BYTES_PER_ROW_ALIGNMENT`, spelled out so [`raw_window_fill_plan`] builds without
/// the `gpu` feature: the capture asks it too.
pub const WINDOW_COPY_ROW_ALIGN: u64 = 256;

/// One texture-to-buffer copy of [`raw_window_fill_plan`]: `width x rows` texels of a raw 64-bit
/// target from `(x, y)`, landing `window_offset` bytes into the window. `bytes_per_row` is set
/// only for a multi-row copy, whose rows are the surface's own pitch apart in the window.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub struct WindowCopy {
    pub x: u32,
    pub y: u32,
    pub width: u32,
    pub rows: u32,
    pub window_offset: u64,
    pub bytes_per_row: Option<u32>,
}

/// The copies that put a raw 64-bit surface's texels into a memory window over the same guest
/// bytes - see `GxmRenderer::fill_windows_from_raw_targets`. `None` when the window
/// `[addr, addr + len)` does not touch the surface (`width x height` texels from `base`, `pitch`
/// texels of eight bytes a row); `Err` names why one that does cannot be filled.
///
/// The overlap is walked in MEMORY order, so a row's pitch padding and the bytes past the
/// surface's last row are skipped rather than written: those keep the guest's bytes, which is
/// what memory holds there. Whole rows go as one copy when the pitch is a legal
/// `bytes_per_row`; otherwise, and for a row the window holds only part of, one row per copy.
pub fn raw_window_fill_plan(
    addr: u32,
    len: u32,
    base: u32,
    width: u32,
    height: u32,
    pitch: u32,
) -> Option<Result<Vec<WindowCopy>, &'static str>> {
    let row = u64::from(pitch.max(width)) * 8;
    let (lo, hi) = (u64::from(addr), u64::from(addr) + u64::from(len));
    let (sb, se) = (u64::from(base), u64::from(base) + row * u64::from(height));
    if hi <= sb || lo >= se || row == 0 {
        return None;
    }
    let start = lo.max(sb);
    // A copy into a buffer lands on a multiple of the texel size, and a window that splits a
    // texel holds half of one: neither can be expressed as a texel copy.
    if (start - sb) % 8 != 0 || (start - lo) % 8 != 0 {
        return Some(Err("the window does not start on a texel boundary of the surface"));
    }
    let bulk = row % WINDOW_COPY_ROW_ALIGN == 0;
    let mut out = Vec::new();
    // `at` is the window byte being filled, `(x, y)` the texel memory holds there.
    let mut at = start - lo;
    let mut y = ((start - sb) / row) as u32;
    let mut x = (((start - sb) % row) / 8) as u32;
    while at < u64::from(len) && y < height {
        let left = u64::from(len) - at;
        let rows = if x == 0 && bulk { (left / row).min(u64::from(height - y)) as u32 } else { 0 };
        if rows > 1 {
            out.push(WindowCopy { x: 0, y, width, rows, window_offset: at, bytes_per_row: Some(row as u32) });
            at += row * u64::from(rows);
            y += rows;
            continue;
        }
        let w = u64::from(width.saturating_sub(x)).min(left / 8) as u32;
        if w > 0 {
            out.push(WindowCopy { x, y, width: w, rows: 1, window_offset: at, bytes_per_row: None });
        }
        at += row - u64::from(x) * 8;
        y += 1;
        x = 0;
    }
    Some(Ok(out))
}

/// A raw 64-bit target a frame has rendered, as the window fill reads it:
/// `(base, width, height, pitch in texels)`.
pub type RawFillTarget = (u32, u32, u32, u32);

/// >>> THE BYTES OF A WINDOW SPAN THE GPU FILL OVERWRITES, so nobody reads or uploads them first.
///
/// Every byte [`raw_window_fill_plan`] copies a target's texels over, for the guest span
/// `[addr, addr + len)` and every target of `targets` (the ones the fill would copy from: a
/// target the pass renders into is not one). Written to `out` as sorted, disjoint `[start, end)`
/// offsets relative to `addr`; a target the plan refuses contributes nothing, exactly as the fill
/// copies nothing from it.
///
/// The capture and the renderer both ask this of the SAME span - the one
/// [`per_draw_window_spans`] places - because the plan's texel-boundary test depends on where the
/// span starts: a window asked on its own could be "covered" where the span the renderer fills
/// is refused, and its bytes would then be zero where the guest's belong.
pub fn raw_window_fill_cover(addr: u32, len: u32, targets: &[RawFillTarget], out: &mut Vec<(u32, u32)>) {
    out.clear();
    for &(base, w, h, pitch) in targets {
        let Some(Ok(plan)) = raw_window_fill_plan(addr, len, base, w, h, pitch) else {
            continue;
        };
        for c in plan {
            let texels = u64::from(c.width) * 8;
            let rows = c.bytes_per_row.map_or(1, |_| u64::from(c.rows));
            let step = c.bytes_per_row.map_or(0, u64::from);
            for r in 0..rows {
                let s = c.window_offset + r * step;
                out.push((s as u32, (s + texels) as u32));
            }
        }
    }
    out.sort_unstable();
    out.dedup_by(|next, kept| {
        if next.0 <= kept.1 {
            kept.1 = kept.1.max(next.1);
            true
        } else {
            false
        }
    });
}

/// How a PER-DRAW window set is placed in its block: windows that overlap in guest memory, and
/// sit a whole number of words apart, share ONE span holding their union. Returns each span's
/// guest `[start, end)` in address order and, per window, the index of its span. `extents` is
/// each window's reach for the draw; a missing one is 0.
pub fn per_draw_window_spans(addrs: &[u32], extents: &[u32]) -> (Vec<(u64, u64)>, Vec<usize>) {
    let n = addrs.len();
    let mut order: Vec<usize> = (0..n).collect();
    order.sort_by_key(|&i| addrs[i]);
    let mut spans: Vec<(u64, u64)> = Vec::new();
    let mut span_of = vec![0usize; n];
    for &i in &order {
        let (a, e) = (u64::from(addrs[i]), u64::from(extents.get(i).copied().unwrap_or(0)));
        match spans.last_mut() {
            Some(s) if a <= s.1 && (a - s.0) % 4 == 0 => s.1 = s.1.max(a + e),
            _ => spans.push((a, a + e)),
        }
        span_of[i] = spans.len() - 1;
    }
    (spans, span_of)
}

/// Say, once per window address, that a memory window was filled from a rendered target - or,
/// with `refused`, that it overlaps one and could not be, and why.
#[cfg(feature = "gpu")]
pub(crate) fn report_window_from_target(addr: u32, len: u32, base: u32, w: u32, h: u32, refused: Option<&str>) {
    use std::sync::{Mutex, OnceLock};
    static SEEN: OnceLock<Mutex<std::collections::HashSet<u32>>> = OnceLock::new();
    let seen = SEEN.get_or_init(|| Mutex::new(std::collections::HashSet::new()));
    if !seen.lock().unwrap_or_else(|e| e.into_inner()).insert(addr) {
        return;
    }
    match refused {
        None => tracing::info!(
            target: "vitaslop::render",
            "gxm memory window {addr:#x}+{len} lies on the raw 64-bit surface at {base:#x} ({w}x{h}) this frame rendered - its bytes are copied from that target on the GPU, not read from guest memory the GPU never wrote back"
        ),
        Some(why) => tracing::warn!(
            target: "vitaslop::render",
            "gxm memory window {addr:#x}+{len} lies on the raw 64-bit surface at {base:#x} ({w}x{h}) this frame rendered, and is NOT filled from it ({why}) - the shader reads the guest's bytes there, which the GPU never wrote"
        ),
    }
}

/// The attachment format a [`float_color_format`] surface is rendered into.
#[cfg(feature = "gpu")]
pub const FLOAT_TARGET_FORMAT: wgpu::TextureFormat = wgpu::TextureFormat::Rgba16Float;

/// The attachment format a [`raw64_color_format`] surface is rendered into.
#[cfg(feature = "gpu")]
pub const RAW64_FORMAT: wgpu::TextureFormat = wgpu::TextureFormat::Rg32Uint;

pub fn gxm_sample_count(multisample_mode: u32) -> u32 {
    match multisample_mode {
        // SCE_GXM_MULTISAMPLE_2X and _4X. Anything else is NONE: the enum has no other
        // members, and an unrecognised value is not a licence to invent a sample count.
        1 | 2 => MSAA_SAMPLES,
        _ => 1,
    }
}

/// The one multisampled count this renderer builds pipelines for.
///
/// WebGPU guarantees sample counts 1 and 4 and nothing between, so honouring the guest's
/// modes takes exactly two pipeline variants rather than a family - which is what lets the
/// fixed-function pair be built eagerly instead of discovered mid-frame.
pub const MSAA_SAMPLES: u32 = 4;

/// A whole scene reduced to general draws, in submission order. The runtime builds
/// it from a captured [`Scene`](vitaslop_runtime-side); [`GxmRenderer`] draws it.
#[derive(Clone, Debug, Default)]
pub struct RenderScene {
    /// Shader pairs the guest's patcher has named, for [`GxmRenderer::encode_chain`] to prepare
    /// before it encodes anything. `(vertex container bytes, fragment container bytes)`.
    ///
    /// See `capture::Scene::precompile`: the device patches pre-compiled USSE code at
    /// `sceGxmShaderPatcherCreateFragmentProgram`, which titles call behind a loading screen,
    /// while this recompiler has to produce WGSL and have a driver compile it. Doing that at the
    /// first DRAW is what puts 50-100 ms of pipeline building inside gameplay frames.
    pub precompile: std::sync::Arc<Vec<vitaslop_gxp_shader::PatcherPair>>,
    pub draws: Vec<GxmDraw>,
    /// This scene's render target, when the guest's colour surface was resolvable.
    pub target: Option<RttTarget>,
    /// Linear depth-normalization for the opaque (Mvp) draws: the shader maps each
    /// fragment's post-divide depth `d = c.z/c.w` to `clamp((d - depth_min) * depth_scale,
    /// 0, 1)` before the depth test. The guest's captured matrices produce `d` in an
    /// arbitrary, often huge range (hundreds to thousands, not a normalized [0,1]); the
    /// builder scans the visible opaque geometry and picks `depth_min`/`depth_scale` so the
    /// whole range maps linearly across the [0,1] depth buffer at full f32 precision. This
    /// preserves the software oracle's exact depth ORDERING (a monotonic map) while keeping
    /// enough resolution to separate, e.g., a vehicle from the ground it sits on - which a
    /// saturating squash loses when both sit at a large depth. `depth_scale == 0` means no
    /// opaque geometry (or a single depth plane): every opaque fragment maps to 0.
    pub depth_min: f32,
    pub depth_scale: f32,
    /// Where the guest's `SceGxmDepthStencilSurface` for this scene puts its depth samples,
    /// or 0 when the scene had none.
    ///
    /// A later pass that reads this scene's depth - a soft-particle fade, fog, SSAO - binds a
    /// texture at exactly this address. It is the ONLY thing that tells such a sample apart
    /// from one of the colour target, because a title allocates the two next to each other:
    /// on one retail racer the world's colour is at `0x89204aa0` and its depth 256 bytes later
    /// at `0x89204ba0`, so an address-RANGE match against the colour target claims it first
    /// and the pass reads a colour where it wanted a distance.
    pub depth_addr: u32,
    /// Where the guest's `SceGxmDepthStencilSurface` puts its STENCIL plane, when that is a
    /// separate plane (not the depth address); 0 otherwise. A later pass can sample it as a U8
    /// texture - a post pass masking by what the scene marked - and this address is the only
    /// thing that names it. See `GxmRenderer::convert_gxm_stencil`.
    pub stencil_addr: u32,
    /// Where a FORCE_LOAD scene loads its depth from: [`Self::depth_addr`] unless
    /// `sceGxmBeginSceneEx` named a separate load surface. The scene still STORES to
    /// `depth_addr`. See `GxmRenderer::load_saved_depth`.
    pub depth_load_addr: u32,
    /// The stencil value this scene's depth-stencil surface clears to - the low byte of its
    /// `backgroundControl`. Every pass starts its stencil there, as its depth starts at
    /// [`Self::depth_clear`].
    pub stencil_clear: u8,
    /// The depth this scene's depth-stencil surface clears to - its `backgroundDepth`
    /// (`sceGxmDepthStencilSurfaceSetBackgroundDepth`, 1.0 after `...Init`).
    ///
    /// MEASURED on a fighting title: every scene sets it to 0.0 and draws its world
    /// with `GREATER_EQUAL` - a REVERSED depth range. Cleared to a fixed 1.0, no surface could
    /// pass its test, the world target held only the always-pass light shafts, and the frame
    /// tone-mapped those into a flat cyan. `VITASLOP_GXM_BACKGROUND_DEPTH=0` is the arm back to
    /// the fixed 1.0.
    pub depth_clear: f32,
    /// The pixel extent a DEPTH-ONLY pass rasterises at, when this scene has no colour
    /// surface to take one from.
    ///
    /// A `SceGxmDepthStencilSurface` carries no width or height - it is a pointer, a stride
    /// and a format - so a scene that renders depth and no colour has nothing on it that
    /// says how big the pass is. Without a size such a pass cannot be given a target at all,
    /// which is why it used to be reported and skipped; and skipping it is only harmless
    /// while nothing samples the depth it was supposed to write.
    ///
    /// The extent comes from the draws' VIEWPORT, which is the guest's own statement of the
    /// pixel region it is drawing into. That is the same fallback
    /// [`Scene::adopt_viewport_extent`](vitaslop_runtime) already uses for a degenerate
    /// colour surface, and it is sound for the same reason - but it is still a derived
    /// number rather than one the guest wrote down, so a pass placed this way says so.
    pub depth_extent: Option<(u32, u32)>,
    /// The viewport-enabled draws of a depth-only pass did NOT all name the same rectangle, so
    /// [`Self::depth_extent`] is the largest of several rather than the one the guest stated.
    ///
    /// Carried separately because it is the only part of that extent worth reporting: an
    /// agreed extent is a measurement and needs no comment, while a disagreement means every
    /// later pass sampling this depth inherits a resolution no single draw asked for.
    pub depth_extent_ambiguous: bool,
    /// The depth-stencil surface's `zlsControl` word, 0 when the scene has none. Its
    /// FORCE_LOAD bit (0x2) makes the tiler LOAD the surface's depth and stencil from memory at
    /// the start of the scene, and FORCE_STORE (0x4) STORE them at the end - which is how a
    /// deferred renderer carries the depth its geometry pass wrote into a later pass that
    /// draws into a DIFFERENT colour target. See `GxmRenderer::depth_saved`.
    pub zls_control: u32,
}

#[cfg(feature = "gpu")]
pub use render::CubeRenderer;

#[cfg(feature = "gpu")]
pub use gxm::{
    buffer_write_largest_bytes, buffer_write_worst_run_us_kb, device_budget_rows, upload_volume,
    upload_transport,
    BudgetRow, WRITE_BUFFER_BUDGET_BYTES,
    take_worst_write_us,
    write_stall_census,
    GPU_SUBMITS_IN_FLIGHT,
    take_encode_work, peek_encode_work, run_encode_work, take_prepare_split, take_sampler_bg_counts, take_sampler_bg_pass,
    take_sampler_bg_prev,
    wasm_clock_installed,
    EncodePhases, EncodeWork, PrepareSplit,
    GxmRenderer, PipeFuture, PipelineWarm, PipelinesReady, pipe_rebuilds,
    vertex_plan_census_line,
};

/// See [`gxm::set_wasm_clock`] - the browser installs its `performance.now()` here so the
/// renderer's own phase split is measured there too, not structurally zero.
#[cfg(all(feature = "gpu", target_arch = "wasm32"))]
pub use gxm::set_wasm_clock;
// Gated like every other `gxm` re-export above. Without this the crate does not COMPILE with
// the `gpu` feature off, which is how `cargo test -p vitaslop-runtime --lib` builds it - so the
// generated knob index could not be regenerated at all. Feature unification hides it from a
// workspace check, which is why it survived.
#[cfg(feature = "gpu")]
pub use gxm::take_arena_repeat;

/// A monotonic millisecond clock on EITHER target, and available whether or not the `gpu`
/// feature is on.
///
/// Here rather than inside `gxm` because the SCENE BUILD lives in another crate and is not part
/// of the renderer: a build phase that can only be timed natively is a build phase whose cost is
/// unknown on the one engine that is the product
/// [[vitaslop-web-is-the-product-not-the-tool]]. On wasm it reads whatever clock the frontend
/// installed with [`set_wasm_clock`], and 0.0 until one is - a structural zero, which the
/// caller's own line has to be readable as.
/// The patcher-time translation figures - see `vitaslop_gxp_shader::prelink_stats`.
pub use vitaslop_gxp_shader::prelink_stats;

pub fn now_ms() -> f64 {
    #[cfg(not(target_arch = "wasm32"))]
    {
        use std::sync::OnceLock;
        static EPOCH: OnceLock<std::time::Instant> = OnceLock::new();
        EPOCH.get_or_init(std::time::Instant::now).elapsed().as_secs_f64() * 1000.0
    }
    #[cfg(all(target_arch = "wasm32", feature = "gpu"))]
    {
        gxm::now_ms()
    }
    #[cfg(all(target_arch = "wasm32", not(feature = "gpu")))]
    {
        0.0
    }
}

#[cfg(feature = "gpu")]
mod render;

/// `SCE_GXM_TEXTURE_BASE_FORMAT_YUV420P2 >> 24`: 4:2:0 in two planes - a full-resolution
/// luma plane followed by one of interleaved chroma at half resolution in both axes. It is
/// what a video decoder writes and what a title binds to put a movie on screen.
///
/// It lives in this crate rather than beside the rest of the guest's format handling because
/// BOTH sides need it: the runtime decodes it, and this crate's uploader treats it
/// differently from every other RGBA8 texture (no mip chain - see `mips_for_texture`).
pub const GXM_BASE_FORMAT_YUV420P2: u32 = 0x90;

/// The general GXM renderer: the GPU twin of the runtime's software rasterizer,
/// drawing a [`RenderScene`] of textured, alpha-blended, multi-space draws.
#[cfg(feature = "gpu")]
mod gxm;

/// `VITASLOP_GXM_RTT_CLEAR_EVERY_FRAME=1` - restore the pre-2026-09-12 behaviour, in which the
/// frame's FIRST pass into an offscreen render target cleared it to transparent black even
/// though earlier frames had rendered into it.
///
/// The default is now to LOAD, which is what a GXM tiler does at `sceGxmBeginScene`. This knob
/// exists because the two failure modes - a target that wrongly accumulates and one that wrongly
/// loses its accumulation - are indistinguishable in a picture, so any title whose frame moves
/// needs the A/B rather than an argument.
pub fn rtt_clear_every_frame() -> bool {
    static ON: std::sync::OnceLock<bool> = std::sync::OnceLock::new();
    *ON.get_or_init(|| matches!(crate::knobs::var("VITASLOP_GXM_RTT_CLEAR_EVERY_FRAME").as_deref(), Ok("1")))
}

/// `VITASLOP_GXM_ARENA_REPEAT=1` - count how many of the bytes uploaded into the per-pass
/// geometry arenas are BYTE-IDENTICAL to what the same arena was given last frame.
///
/// Off by default because it hashes every uploaded byte, which is the cost it is measuring. It
/// exists to decide one question before any machinery is built for it: the worker profile puts
/// `writeBuffer` at 7.6% of the browser and the counters show a single 2,448 KB call a frame,
/// so IF those bytes repeat they should be resident geometry and not an upload - and if they do
/// not repeat, residency cannot help and the answer is somewhere else. The previous attempt at
/// this (keying residency on the packed content) was built, measured and REFUTED; this is the
/// measurement that should have come first.
pub fn arena_repeat_probe() -> bool {
    static ON: std::sync::OnceLock<bool> = std::sync::OnceLock::new();
    *ON.get_or_init(|| matches!(crate::knobs::var("VITASLOP_GXM_ARENA_REPEAT").as_deref(), Ok("1")))
}

/// `VITASLOP_ARENA_UPLOAD_PROBE=1` - time the SAME bytes through the other upload path, a
/// handful of times, and say which is cheaper on THIS engine.
///
/// >>> THE QUESTION IT ANSWERS, WHICH NOTHING ELSE CAN. A V8 worker profile of a baseball
/// title's live gameplay puts `queue.writeBuffer` at **8.2% of the busy worker thread**, 2,735
/// of its 2,780 samples under `ensure_gxp_arena`, for 1.76 MB a frame in ten calls - about
/// 1.3 GB/s, which is an order of magnitude under a memcpy. The bytes themselves are not
/// reducible (they are the frame's genuinely changed geometry and its skinning windows), so
/// the only lever left is the MECHANISM, and there is exactly one alternative: a staging buffer
/// created MAPPED, filled through its mapped range, unmapped, and copied on the GPU.
///
/// So this times both, on the same bytes, back to back, and reports the pair. It is a knob and
/// not a default because the alternative allocates a buffer per call - the shape that has
/// already cost this project a regression once
/// [[vitaslop-mapped-at-creation-exhausts-a-staging-arena]] - and nothing should be built on it
/// until the two numbers are side by side. The staging buffer is DROPPED without being copied
/// from: this measures the CPU cost of getting the bytes to the GPU's side of the boundary,
/// which is the whole of what `write_buffer` is being charged for.
pub fn arena_upload_probe() -> bool {
    static ON: std::sync::OnceLock<bool> = std::sync::OnceLock::new();
    *ON.get_or_init(|| matches!(crate::knobs::var("VITASLOP_ARENA_UPLOAD_PROBE").as_deref(), Ok("1")))
}

/// `VITASLOP_GXM_SWEEP_EVERY=<n>` - how many frames apart the packed-geometry budget check and
/// the promotion-map dead sweep run. Default 8; `1` restores the per-frame behaviour and is the
/// OFF arm of that change. Both are MEMORY BOUNDS, not correctness invariants - see the site in
/// `encode_chain`, and the device number that prompted it (1.1-1.4 ms of a 35 ms frame).
pub fn sweep_every() -> u64 {
    static N: std::sync::OnceLock<u64> = std::sync::OnceLock::new();
    *N.get_or_init(|| {
        crate::knobs::var("VITASLOP_GXM_SWEEP_EVERY")
            .ok()
            .and_then(|v| v.trim().parse::<u64>().ok())
            .filter(|n| *n > 0)
            .unwrap_or(8)
    })
}

/// `VITASLOP_GXP_CLIP_RETRY_EVERY_FRAME=1` - re-measure a pair whose clip measurement found no
/// geometry on EVERY appearance, as this did before the retry was put on a backoff. See
/// `GxpLive::negw_retry` for what that cost: 5.02 ms of a 6.62 ms encode on a baseball title's
/// gameplay frame, for ever, over five pairs that can never produce evidence.
///
/// The OFF arm of the identity check, and VALUE-sensitive like the knobs beside it, so one build
/// answers both arms.
pub fn gxp_clip_retry_every_frame() -> bool {
    static ON: std::sync::OnceLock<bool> = std::sync::OnceLock::new();
    *ON.get_or_init(|| matches!(crate::knobs::var("VITASLOP_GXP_CLIP_RETRY_EVERY_FRAME").as_deref(), Ok("1")))
}
