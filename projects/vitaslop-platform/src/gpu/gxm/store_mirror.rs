//! WRITABLE MIRRORS of the guest memory fragment programs STORE to, and the depth-stencil
//! FORCE_LOAD / FORCE_STORE of a stencil plane that lives in that memory.
//!
//! # Why a mirror
//! A fragment program's 0xF0 store writes GUEST memory (see
//! `vitaslop_gxp_shader::module::resolve_store_windows`). WGSL has no pointers, so the stored-to
//! range is a storage buffer: the union of a draw's store windows, SEEDED from the guest bytes
//! the capture read at draw time and written by the shader. The mirror then IS that memory as
//! far as the GPU is concerned - newer than the guest's RAM, which our GPU never writes - so the
//! later readers of the memory read the mirror:
//!
//! * a scene whose depth-stencil surface FORCE_LOADs a separate stencil plane inside a mirror
//!   loads its stencil from it ([`GxmRenderer::load_guest_stencil`]); and
//! * a FORCE_STORE of such a plane writes the stencil back into the mirror
//!   ([`GxmRenderer::store_guest_stencil`]), as the tiler stores it to memory.
//!
//! Everything else keeps its current path: a scene with no mirror over its stencil plane keeps
//! the renderer-held depth-stencil (`depth_saved`) or its background clear exactly as before.
//!
//! # What is relied on
//! * ORDER. The mirror's GPU writes (a storing pass, a stencil store) and reads (a stencil load)
//!   are encoder commands in frame order. A re-seed from a changed guest snapshot is a
//!   `write_buffer`, which the queue runs BEFORE the frame's commands - so it is taken only
//!   while nothing earlier in the frame touched the mirror, and is otherwise reported and
//!   skipped rather than run out of order.
//! * WHAT A CHANGED SNAPSHOT MEANS. Our GPU never writes guest RAM, so the guest bytes are what
//!   the CPU last wrote there. A snapshot that differs from the one the mirror was seeded from
//!   means the CPU wrote the range since; one that does not is taken as "the CPU did not write
//!   it", and the mirror keeps the GPU's newer bytes. A CPU that rewrites the range with the SAME
//!   bytes every frame is indistinguishable from one that does not write it at all; for a range
//!   a store draw rewrites whole (the one program that stores), the two agree.
//! * CPU READS of the stored bytes are not delivered: the render-target writeback covers small
//!   colour surfaces only, and a store mirror is not one (the one that exists is 525,120 bytes,
//!   past that writeback's whole cap). Reported once per mirror.

use super::*;

/// One writable mirror of a guest byte range.
pub(super) struct GuestMirror {
    /// The guest byte address of the mirror's first byte (word aligned) and its length.
    pub(super) lo: u32,
    pub(super) bytes: u32,
    pub(super) buf: wgpu::Buffer,
    /// Identity for bind-group caches (a mirror replaced by a larger one gets a new id).
    pub(super) id: u64,
    /// Fingerprint of the guest bytes the mirror was last seeded from.
    seed_fp: u64,
    /// The [`StoreMirrors::frame`] in which a GPU command last read or wrote the mirror.
    used_frame: u64,
}

impl GuestMirror {
    /// Whether the mirror holds the whole guest range `[lo, hi)`.
    pub(super) fn covers(&self, lo: u32, hi: u64) -> bool {
        self.lo <= lo && hi <= u64::from(self.lo) + u64::from(self.bytes)
    }
}

/// The renderer's mirrors - held by [`GxpLive`], which builds the bind groups naming them.
#[derive(Default)]
pub(super) struct StoreMirrors {
    pub(super) mirrors: Vec<GuestMirror>,
    next_id: u64,
    /// Counts frames, for [`GuestMirror::used_frame`]. Advanced by `encode_chain`.
    pub(super) frame: u64,
    /// Copies a mirror that REPLACED smaller ones owes from them, `(from, to mirror id, to
    /// offset, bytes)`, encoded before the next pass that may use it ([`Self::flush`]).
    pending: Vec<(wgpu::Buffer, u64, u64, u64)>,
    /// Replaced mirrors' buffers, destroyed after the submit that last names them.
    retired: Vec<wgpu::Buffer>,
}

/// The largest mirror this renderer makes. A store window is a buffer a program declares, and
/// the one that exists is half a megabyte; a span past this is refused by name rather than
/// allocated.
const MIRROR_MAX_BYTES: u64 = 64 << 20;

/// A fingerprint of a draw's store-window snapshots over their UNION, each guest byte once
/// (windows overlap - the four of the program that stores are one surface's row phases).
fn seed_fingerprint(windows: &[(u32, std::sync::Arc<[u8]>)]) -> u64 {
    let mut order: Vec<usize> = (0..windows.len()).collect();
    order.sort_by_key(|&i| windows[i].0);
    let mut h: u64 = 0xcbf2_9ce4_8422_2325;
    let mut mix = |v: u64| h = (h ^ v).wrapping_mul(0x0000_0100_0000_01b3).rotate_left(5);
    let mut covered: u64 = 0;
    for i in order {
        let (addr, bytes) = &windows[i];
        let start = u64::from(*addr);
        let end = start + bytes.len() as u64;
        let from = covered.max(start);
        if from >= end {
            continue;
        }
        mix(from);
        let tail = &bytes[(from - start) as usize..];
        let mut chunks = tail.chunks_exact(8);
        for c in &mut chunks {
            mix(u64::from_le_bytes([c[0], c[1], c[2], c[3], c[4], c[5], c[6], c[7]]));
        }
        for &b in chunks.remainder() {
            mix(u64::from(b));
        }
        covered = end;
    }
    h
}

impl StoreMirrors {
    /// The mirror a draw with these store windows writes, created, enlarged or re-seeded as
    /// needed - its index in [`Self::mirrors`]. `Err` names why the draw cannot be given one.
    pub(super) fn ensure(
        &mut self,
        device: &wgpu::Device,
        queue: &wgpu::Queue,
        windows: &[(u32, std::sync::Arc<[u8]>)],
    ) -> Result<usize, &'static str> {
        if windows.iter().any(|(a, b)| a % 4 != 0 || b.len() % 4 != 0) {
            return Err("a buffer this draw's fragment program stores through is not word aligned");
        }
        let lo = windows.iter().map(|(a, _)| *a).min().ok_or("no store window")?;
        let hi = windows.iter().map(|(a, b)| u64::from(*a) + b.len() as u64).max().unwrap_or(0).div_ceil(4) * 4;
        let fp = seed_fingerprint(windows);
        if let Some(i) = self.mirrors.iter().position(|m| m.covers(lo, hi)) {
            let frame = self.frame;
            let m = &mut self.mirrors[i];
            if m.seed_fp != fp {
                if m.used_frame == frame {
                    report_mirror_reseed_skipped(m.lo, m.bytes);
                } else {
                    for (addr, bytes) in windows {
                        queue.write_buffer(&m.buf, u64::from(addr - m.lo), bytes);
                    }
                    m.seed_fp = fp;
                }
            }
            m.used_frame = frame;
            return Ok(i);
        }
        // A new mirror over the union of this span and every mirror it overlaps.
        let overlaps = |m: &GuestMirror| u64::from(m.lo) < hi && u64::from(lo) < u64::from(m.lo) + u64::from(m.bytes);
        let new_lo = self.mirrors.iter().filter(|m| overlaps(m)).map(|m| m.lo).fold(lo, u32::min);
        let new_hi = self
            .mirrors
            .iter()
            .filter(|m| overlaps(m))
            .map(|m| u64::from(m.lo) + u64::from(m.bytes))
            .fold(hi, u64::max);
        let size = new_hi - u64::from(new_lo);
        if size > MIRROR_MAX_BYTES || size > device.limits().max_storage_buffer_binding_size {
            return Err("the guest range this draw's stores reach is larger than a storage binding this renderer makes");
        }
        let buf = device.create_buffer(&wgpu::BufferDescriptor {
            label: Some("gxm-store-mirror"),
            size,
            usage: wgpu::BufferUsages::STORAGE | wgpu::BufferUsages::COPY_SRC | wgpu::BufferUsages::COPY_DST,
            mapped_at_creation: false,
        });
        for (addr, bytes) in windows {
            queue.write_buffer(&buf, u64::from(addr - new_lo), bytes);
        }
        self.next_id += 1;
        let id = self.next_id;
        // The replaced mirrors' bytes are NEWER than the snapshot (the GPU wrote them), so
        // they are copied over the seed, in frame order, before the next pass.
        let mut keep = Vec::new();
        for m in self.mirrors.drain(..) {
            if overlaps(&m) {
                self.pending.push((m.buf.clone(), id, u64::from(m.lo - new_lo), u64::from(m.bytes)));
                self.retired.push(m.buf);
            } else {
                keep.push(m);
            }
        }
        self.mirrors = keep;
        report_mirror_created(new_lo, size);
        self.mirrors.push(GuestMirror { lo: new_lo, bytes: size as u32, buf, id, seed_fp: fp, used_frame: self.frame });
        Ok(self.mirrors.len() - 1)
    }

    /// Encode the copies an enlarged mirror owes, and hand the replaced buffers to the
    /// renderer's graveyard (`retired`), which destroys them after the submit.
    pub(super) fn flush(&mut self, encoder: &mut wgpu::CommandEncoder, retired: &mut Vec<wgpu::Buffer>) {
        for (from, to, at, bytes) in self.pending.drain(..) {
            if let Some(m) = self.mirrors.iter().find(|m| m.id == to) {
                encoder.copy_buffer_to_buffer(&from, 0, &m.buf, at, bytes);
            }
        }
        retired.append(&mut self.retired);
    }

    /// The mirror holding the whole guest range `[lo, hi)`, marked used this frame.
    pub(super) fn covering(&mut self, lo: u32, hi: u64) -> Option<&mut GuestMirror> {
        let frame = self.frame;
        let m = self.mirrors.iter_mut().find(|m| m.covers(lo, hi))?;
        m.used_frame = frame;
        Some(m)
    }
}

/// A scene's separate STENCIL plane as GXM lays it out: its guest address, row pitch in bytes
/// and extent - `None` when the surface's stencil is not a plane this renderer can address
/// (no stencil address, a packed depth-stencil format, a tiled layout).
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub(super) struct StencilPlane {
    pub(super) addr: u32,
    pub(super) pitch: u32,
    pub(super) width: u32,
    pub(super) height: u32,
}

impl StencilPlane {
    /// From `zlsControl`, as `sceGxmDepthStencilSurfaceInit` encodes it: the stride in samples
    /// at bits 10:3 (`stride / 32 - 1`), the surface TYPE at 0x11000 (0 = linear), the FORMAT at
    /// 0x07EEE000. A separate 8-bit stencil plane is one byte per sample, so its row pitch is the
    /// stride. Formats with one: `S8` (0x22000), `DF32_S8` (0x66000), `DF32M_S8` (0xEE000).
    pub(super) fn of(scene: &RenderScene, width: u32, height: u32) -> Result<StencilPlane, &'static str> {
        let zls = scene.zls_control;
        if scene.stencil_addr == 0 {
            return Err("the surface has no separate stencil plane");
        }
        if !matches!(zls & 0x07EE_E000, 0x0002_2000 | 0x0006_6000 | 0x000E_E000) {
            return Err("the surface's format carries no separate 8-bit stencil plane");
        }
        if zls & 0x0001_1000 != 0 {
            return Err("the stencil plane is TILED, and the tiled depth-stencil layout is not established");
        }
        let pitch = (((zls >> 3) & 0xff) + 1) * 32;
        if width > pitch {
            return Err("the pass is wider than the stencil plane's own stride");
        }
        Ok(StencilPlane { addr: scene.stencil_addr, pitch, width, height })
    }

    /// One past the last guest byte the plane's `width x height` pixels occupy.
    pub(super) fn end(&self) -> u64 {
        u64::from(self.addr) + u64::from(self.pitch) * u64::from(self.height.saturating_sub(1)) + u64::from(self.width)
    }
}

/// The word-row copy between a mirror (rows at the guest's pitch) and a staging buffer (rows at
/// the 256-byte pitch a buffer-texture copy needs) - one compute pipeline serving both
/// directions. Params: `(src_off, src_pitch, dst_off, dst_pitch)` and `(row_words, rows,
/// src_len, dst_len)`, all in 32-bit words.
const ROW_COPY_WGSL: &str = "struct P { a: vec4<u32>, b: vec4<u32> };
@group(0) @binding(0) var<storage, read> src: array<u32>;
@group(0) @binding(1) var<storage, read_write> dst: array<u32>;
@group(0) @binding(2) var<uniform> p: P;
@compute @workgroup_size(64, 1, 1)
fn main(@builtin(global_invocation_id) id: vec3<u32>) {
  if (id.x >= p.b.x || id.y >= p.b.y) { return; }
  let s = p.a.x + id.y * p.a.y + id.x;
  let d = p.a.z + id.y * p.a.w + id.x;
  if (s < p.b.z && d < p.b.w) { dst[d] = src[s]; }
}
";

/// The pipeline and layout behind [`ROW_COPY_WGSL`].
pub(super) fn row_copy_pipeline(device: &wgpu::Device) -> (wgpu::BindGroupLayout, wgpu::ComputePipeline) {
    let module = device.create_shader_module(wgpu::ShaderModuleDescriptor {
        label: Some("gxm-stencil-row-copy"),
        source: wgpu::ShaderSource::Wgsl(ROW_COPY_WGSL.into()),
    });
    let storage = |binding: u32, read_only: bool| wgpu::BindGroupLayoutEntry {
        binding,
        visibility: wgpu::ShaderStages::COMPUTE,
        ty: wgpu::BindingType::Buffer {
            ty: wgpu::BufferBindingType::Storage { read_only },
            has_dynamic_offset: false,
            min_binding_size: None,
        },
        count: None,
    };
    let layout = device.create_bind_group_layout(&wgpu::BindGroupLayoutDescriptor {
        label: Some("gxm-stencil-row-copy"),
        entries: &[
            storage(0, true),
            storage(1, false),
            wgpu::BindGroupLayoutEntry {
                binding: 2,
                visibility: wgpu::ShaderStages::COMPUTE,
                ty: wgpu::BindingType::Buffer {
                    ty: wgpu::BufferBindingType::Uniform,
                    has_dynamic_offset: false,
                    min_binding_size: None,
                },
                count: None,
            },
        ],
    });
    let pl = device.create_pipeline_layout(&wgpu::PipelineLayoutDescriptor {
        label: Some("gxm-stencil-row-copy"),
        bind_group_layouts: &[Some(&layout)],
        immediate_size: 0,
    });
    let pipe = device.create_compute_pipeline(&wgpu::ComputePipelineDescriptor {
        label: Some("gxm-stencil-row-copy"),
        layout: Some(&pl),
        module: &module,
        entry_point: Some("main"),
        compilation_options: Default::default(),
        cache: None,
    });
    (layout, pipe)
}

/// The staging pitch a buffer-texture copy of a `width`-byte stencil row needs.
fn staging_pitch(width: u32) -> u32 {
    width.div_ceil(wgpu::COPY_BYTES_PER_ROW_ALIGNMENT) * wgpu::COPY_BYTES_PER_ROW_ALIGNMENT
}

/// Encode one word-row copy (see [`ROW_COPY_WGSL`]) from `src` to `dst`. Returns the copy's
/// parameter block, which must stay alive until the submit that runs it.
#[allow(clippy::too_many_arguments)]
fn encode_row_copy(
    device: &wgpu::Device,
    queue: &wgpu::Queue,
    encoder: &mut wgpu::CommandEncoder,
    (layout, pipe): &(wgpu::BindGroupLayout, wgpu::ComputePipeline),
    src: &wgpu::Buffer,
    src_off_pitch: (u32, u32),
    dst: &wgpu::Buffer,
    dst_off_pitch: (u32, u32),
    row_words: u32,
    rows: u32,
    pool: &mut BufferPool,
) -> wgpu::Buffer {
    let words: [u32; 8] = [
        src_off_pitch.0,
        src_off_pitch.1,
        dst_off_pitch.0,
        dst_off_pitch.1,
        row_words,
        rows,
        (src.size() / 4) as u32,
        (dst.size() / 4) as u32,
    ];
    // A separate 32-byte block per copy: one shared block rewritten by `write_buffer` would hand
    // every copy in the frame the LAST copy's parameters.
    //
    // >>> FROM THE POOL, because the block goes back there after the submit (the caller hands
    // it to `retired_buffers`). Created fresh every time, the pool only ever FILLED with them:
    // MEASURED in the desktop browser's GPU census on a shooter's level, 3,326 of these live
    // and climbing ~20 a second, each a real GPU allocation however small its 32 bytes.
    let usage = wgpu::BufferUsages::UNIFORM | wgpu::BufferUsages::COPY_DST;
    let pooled = crate::gpu::memory_bound_on("VITASLOP_POOL_SMALL_BUFFERS").then(|| pool.take(32, usage)).flatten();
    let params = pooled.unwrap_or_else(|| {
        device.create_buffer(&wgpu::BufferDescriptor {
            label: Some("gxm-stencil-row-copy-params"),
            size: 32,
            usage,
            mapped_at_creation: false,
        })
    });
    queue.write_buffer(&params, 0, &words.iter().flat_map(|w| w.to_le_bytes()).collect::<Vec<u8>>());
    let bind = device.create_bind_group(&wgpu::BindGroupDescriptor {
        label: Some("gxm-stencil-row-copy"),
        layout,
        entries: &[
            wgpu::BindGroupEntry { binding: 0, resource: src.as_entire_binding() },
            wgpu::BindGroupEntry { binding: 1, resource: dst.as_entire_binding() },
            wgpu::BindGroupEntry { binding: 2, resource: params.as_entire_binding() },
        ],
    });
    {
        let mut pass = encoder.begin_compute_pass(&wgpu::ComputePassDescriptor {
            label: Some("gxm-stencil-row-copy"),
            timestamp_writes: None,
        });
        pass.set_pipeline(pipe);
        pass.set_bind_group(0, &bind, &[]);
        pass.dispatch_workgroups(row_words.div_ceil(64), rows, 1);
    }
    params
}

/// The stencil LOAD of a plane: its rows out of `mirror` (whose first byte is guest address
/// `mirror_lo`) into `staging` at a buffer-texture copy's pitch, then into the stencil aspect
/// of `dst`. Returns the row copy's parameter block (see [`encode_row_copy`]).
#[allow(clippy::too_many_arguments)]
fn encode_plane_load(
    device: &wgpu::Device,
    queue: &wgpu::Queue,
    encoder: &mut wgpu::CommandEncoder,
    pipe: &(wgpu::BindGroupLayout, wgpu::ComputePipeline),
    mirror: &wgpu::Buffer,
    mirror_lo: u32,
    plane: &StencilPlane,
    staging: &wgpu::Buffer,
    dst: &wgpu::Texture,
    pool: &mut BufferPool,
) -> wgpu::Buffer {
    let pitch = staging_pitch(plane.width);
    let params = encode_row_copy(
        device,
        queue,
        encoder,
        pipe,
        mirror,
        ((plane.addr - mirror_lo) / 4, plane.pitch / 4),
        staging,
        (0, pitch / 4),
        plane.width.div_ceil(4),
        plane.height,
        pool,
    );
    encoder.copy_buffer_to_texture(
        wgpu::TexelCopyBufferInfo {
            buffer: staging,
            layout: wgpu::TexelCopyBufferLayout { offset: 0, bytes_per_row: Some(pitch), rows_per_image: Some(plane.height) },
        },
        wgpu::TexelCopyTextureInfo { texture: dst, mip_level: 0, origin: wgpu::Origin3d::ZERO, aspect: wgpu::TextureAspect::StencilOnly },
        wgpu::Extent3d { width: plane.width, height: plane.height, depth_or_array_layers: 1 },
    );
    params
}

/// The stencil STORE of a plane: the stencil aspect of `src` into `staging`, then its rows into
/// `mirror` at the plane's guest pitch. Whole words only: a plane whose width is not a multiple
/// of four would carry the texture copy's row padding into the mirror's next bytes, so its last
/// partial word is left as the mirror held it. Returns the row copy's parameter block, if any.
#[allow(clippy::too_many_arguments)]
fn encode_plane_store(
    device: &wgpu::Device,
    queue: &wgpu::Queue,
    encoder: &mut wgpu::CommandEncoder,
    pipe: &(wgpu::BindGroupLayout, wgpu::ComputePipeline),
    src: &wgpu::Texture,
    plane: &StencilPlane,
    staging: &wgpu::Buffer,
    mirror: &wgpu::Buffer,
    mirror_lo: u32,
    pool: &mut BufferPool,
) -> Option<wgpu::Buffer> {
    let pitch = staging_pitch(plane.width);
    encoder.copy_texture_to_buffer(
        wgpu::TexelCopyTextureInfo { texture: src, mip_level: 0, origin: wgpu::Origin3d::ZERO, aspect: wgpu::TextureAspect::StencilOnly },
        wgpu::TexelCopyBufferInfo {
            buffer: staging,
            layout: wgpu::TexelCopyBufferLayout { offset: 0, bytes_per_row: Some(pitch), rows_per_image: Some(plane.height) },
        },
        wgpu::Extent3d { width: plane.width, height: plane.height, depth_or_array_layers: 1 },
    );
    let row_words = plane.width / 4;
    (row_words > 0).then(|| {
        encode_row_copy(
            device,
            queue,
            encoder,
            pipe,
            staging,
            (0, pitch / 4),
            mirror,
            ((plane.addr - mirror_lo) / 4, plane.pitch / 4),
            row_words,
            plane.height,
            pool,
        )
    })
}

impl GxmRenderer {
    /// A staging buffer of at least `bytes`, usable both ways round.
    fn stencil_staging(&mut self, device: &wgpu::Device, bytes: u64) -> wgpu::Buffer {
        if self.stencil_staging.as_ref().is_none_or(|b| b.size() < bytes) {
            if let Some(old) = self.stencil_staging.take() {
                self.retired_buffers.push(old);
            }
            self.stencil_staging = Some(device.create_buffer(&wgpu::BufferDescriptor {
                label: Some("gxm-stencil-staging"),
                size: bytes,
                usage: wgpu::BufferUsages::STORAGE | wgpu::BufferUsages::COPY_SRC | wgpu::BufferUsages::COPY_DST,
                mapped_at_creation: false,
            }));
        }
        self.stencil_staging.clone().expect("just ensured")
    }

    /// The mirror and plane a scene's stencil load or store would use, or `None` when the scene
    /// keeps its current path. Reports, once per address, a plane that lies in a mirror but
    /// cannot be served.
    fn guest_stencil_target(
        &mut self,
        scene: &RenderScene,
        dst: &wgpu::Texture,
        usage: wgpu::TextureUsages,
        // The guest surface's extent in pixels - what the plane's rows hold.
        surf: (u32, u32),
    ) -> Option<(StencilPlane, wgpu::Buffer, u32)> {
        if !crate::gpu::guest_stencil_on() || scene.stencil_addr == 0 || crate::gpu::stencil_disabled() {
            return None;
        }
        let (w, h) = surf;
        let plane = match StencilPlane::of(scene, w, h) {
            Ok(p) => p,
            Err(why) => {
                // Only worth saying when the plane is in memory a fragment program wrote.
                if self.gxp.store_mirrors.mirrors.iter().any(|m| m.covers(scene.stencil_addr, u64::from(scene.stencil_addr) + 1)) {
                    report_guest_stencil_unserved(scene.stencil_addr, why);
                }
                return None;
            }
        };
        let m = self.gxp.store_mirrors.covering(plane.addr, plane.end())?;
        let (buf, lo) = (m.buf.clone(), m.lo);
        let why = if dst.sample_count() != 1 {
            Some("the pass's depth-stencil attachment is MULTISAMPLED, and a buffer copy cannot address its samples")
        } else if (dst.width(), dst.height()) != surf {
            Some("the pass rasterises at another extent than the guest surface, so the plane's pixels do not map one to one")
        } else if !matches!(dst.format(), wgpu::TextureFormat::Depth24PlusStencil8 | wgpu::TextureFormat::Depth32FloatStencil8) {
            Some("the pass's depth attachment carries no stencil aspect")
        } else if !dst.usage().contains(usage) {
            Some("the pass's depth-stencil attachment was created without the copy usage the load or store needs")
        } else if (plane.addr - lo) % 4 != 0 || plane.pitch % 4 != 0 {
            Some("the stencil plane's rows are not word aligned inside the mirror")
        } else {
            None
        };
        if let Some(why) = why {
            report_guest_stencil_unserved(plane.addr, why);
            return None;
        }
        Some((plane, buf, lo))
    }

    /// Before a pass: when its depth-stencil surface FORCE_LOADs a separate stencil plane that
    /// lies inside a store mirror, load the plane's bytes into the stencil aspect of `dst` and
    /// have the pass LOAD its stencil (`stencil_load_now`). The depth aspect is untouched -
    /// it loads or clears by the surface's own rule.
    ///
    /// The plane's rows sit at the guest's pitch in the mirror; a buffer-texture copy needs a
    /// 256-byte pitch, so a compute pass repacks them into staging first.
    pub(super) fn load_guest_stencil(
        &mut self,
        device: &wgpu::Device,
        queue: &wgpu::Queue,
        encoder: &mut wgpu::CommandEncoder,
        scene: &RenderScene,
        dst: &wgpu::Texture,
        surf: (u32, u32),
    ) {
        self.stencil_load_now = false;
        if scene.zls_control & 0x2 == 0 {
            return;
        }
        self.gxp.store_mirrors.flush(encoder, &mut self.retired_buffers);
        let Some((plane, mirror, lo)) = self.guest_stencil_target(scene, dst, wgpu::TextureUsages::COPY_DST, surf) else {
            return;
        };
        let pitch = staging_pitch(plane.width);
        let staging = self.stencil_staging(device, u64::from(pitch) * u64::from(plane.height));
        let pipe = self.stencil_row_copy.get_or_insert_with(|| row_copy_pipeline(device)).clone();
        let params =
            encode_plane_load(device, queue, encoder, &pipe, &mirror, lo, &plane, &staging, dst, &mut self.buffer_pool);
        self.retired_buffers.push(params);
        self.stencil_load_now = true;
        report_guest_stencil_loaded(plane.addr, plane.width, plane.height, plane.pitch);
    }

    /// The pass-start half of a scene's guest stencil, for an arm of `encode_chain`: clear the
    /// load flag, load the plane when it applies, and say whether the pass's stencil must be
    /// STORED for [`Self::store_guest_stencil`] afterwards. `dtex` is the single-sample
    /// depth-stencil texture the pass rasterises into, `None` when it has none to copy.
    pub(super) fn begin_guest_stencil(
        &mut self,
        device: &wgpu::Device,
        queue: &wgpu::Queue,
        encoder: &mut wgpu::CommandEncoder,
        scene: &RenderScene,
        dtex: Option<&wgpu::Texture>,
        surf: (u32, u32),
    ) -> bool {
        self.stencil_load_now = false;
        let Some(d) = dtex else { return false };
        self.load_guest_stencil(device, queue, encoder, scene, d, surf);
        self.guest_stencil_stores(scene, d, surf)
    }

    /// Whether [`Self::store_guest_stencil`] will write this scene's stencil back, so the pass
    /// must STORE its stencil aspect rather than discard it.
    pub(super) fn guest_stencil_stores(&mut self, scene: &RenderScene, dst: &wgpu::Texture, surf: (u32, u32)) -> bool {
        scene.zls_control & 0x4 != 0 && self.guest_stencil_target(scene, dst, wgpu::TextureUsages::COPY_SRC, surf).is_some()
    }

    /// After a pass: when its surface FORCE_STOREs a separate stencil plane inside a store
    /// mirror, write the pass's stencil back into the mirror at the plane's rows - the tiler's
    /// store to memory, which every later reader of that memory then sees.
    pub(super) fn store_guest_stencil(
        &mut self,
        device: &wgpu::Device,
        queue: &wgpu::Queue,
        encoder: &mut wgpu::CommandEncoder,
        scene: &RenderScene,
        src: &wgpu::Texture,
        surf: (u32, u32),
    ) {
        if scene.zls_control & 0x4 == 0 {
            return;
        }
        let Some((plane, mirror, lo)) = self.guest_stencil_target(scene, src, wgpu::TextureUsages::COPY_SRC, surf) else {
            return;
        };
        let pitch = staging_pitch(plane.width);
        let staging = self.stencil_staging(device, u64::from(pitch) * u64::from(plane.height));
        let pipe = self.stencil_row_copy.get_or_insert_with(|| row_copy_pipeline(device)).clone();
        if let Some(params) =
            encode_plane_store(device, queue, encoder, &pipe, src, &plane, &staging, &mirror, lo, &mut self.buffer_pool)
        {
            self.retired_buffers.push(params);
        }
        report_guest_stencil_stored(plane.addr, plane.width, plane.height);
    }
}

fn once_per(seen: &'static std::sync::OnceLock<std::sync::Mutex<HashSet<u32>>>, addr: u32) -> bool {
    seen.get_or_init(Default::default).lock().unwrap_or_else(|e| e.into_inner()).insert(addr)
}

fn report_mirror_created(lo: u32, bytes: u64) {
    static SEEN: std::sync::OnceLock<std::sync::Mutex<HashSet<u32>>> = std::sync::OnceLock::new();
    if once_per(&SEEN, lo) {
        report_status!(
            "gxp stores: a fragment program STORES to guest memory - {bytes} bytes at {lo:#x} are mirrored on the GPU, seeded from the guest's bytes and written by the shader; a FORCE_LOADed stencil plane in that range loads from the mirror. CPU reads of the stored bytes are not delivered (the render-target writeback covers small colour surfaces only)"
        );
    }
}

fn report_mirror_reseed_skipped(lo: u32, bytes: u32) {
    static SEEN: std::sync::OnceLock<std::sync::Mutex<HashSet<u32>>> = std::sync::OnceLock::new();
    if once_per(&SEEN, lo) {
        report_warn!(
            "gxp stores: the guest bytes behind the {bytes}-byte store mirror at {lo:#x} changed AFTER a GPU command this frame already used the mirror - the CPU's new bytes cannot be ordered behind that command, so the mirror keeps the GPU's"
        );
    }
}

fn report_guest_stencil_loaded(addr: u32, w: u32, h: u32, pitch: u32) {
    static SEEN: std::sync::OnceLock<std::sync::Mutex<HashSet<u32>>> = std::sync::OnceLock::new();
    if once_per(&SEEN, addr) {
        report_status!(
            "gxm stencil FORCE_LOAD: the {w}x{h} stencil plane at {addr:#x} (pitch {pitch}) lies in memory a fragment program STORED to, and is loaded from that store's mirror instead of starting at the surface's background value"
        );
    }
}

fn report_guest_stencil_stored(addr: u32, w: u32, h: u32) {
    static SEEN: std::sync::OnceLock<std::sync::Mutex<HashSet<u32>>> = std::sync::OnceLock::new();
    if once_per(&SEEN, addr) {
        report_status!(
            "gxm stencil FORCE_STORE: the {w}x{h} stencil plane at {addr:#x} is written back into the store mirror over that memory, as the tiler stores it"
        );
    }
}

fn report_guest_stencil_unserved(addr: u32, why: &str) {
    static SEEN: std::sync::OnceLock<std::sync::Mutex<HashSet<u32>>> = std::sync::OnceLock::new();
    if once_per(&SEEN, addr) {
        report_warn!(
            "gxm stencil: the stencil plane at {addr:#x} lies in memory a fragment program STORED to, but {why} - the pass keeps the renderer-held stencil (or the surface's background value) instead of the stored bytes"
        );
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    fn scene(zls: u32, stencil: u32) -> RenderScene {
        RenderScene { zls_control: zls, stencil_addr: stencil, ..Default::default() }
    }

    /// The one surface the stores feed: `S8`, linear, stride 960 (`zls 0x1220ef`).
    #[test]
    fn the_stencil_plane_reads_its_pitch_format_and_layout_from_zls() {
        let p = StencilPlane::of(&scene(0x0012_20ef, 0x924c_0000), 960, 544).unwrap();
        assert_eq!(p, StencilPlane { addr: 0x924c_0000, pitch: 960, width: 960, height: 544 });
        assert_eq!(p.end(), 0x924c_0000 + 960 * 544);
        // A tiled plane, a packed S8D24 surface and a surface without a plane are refused.
        assert!(StencilPlane::of(&scene(0x0012_20ef | 0x0001_1000, 0x924c_0000), 960, 544).is_err());
        assert!(StencilPlane::of(&scene((0x0012_20ef & !0x07EE_E000) | 0x0126_6000, 0x924c_0000), 960, 544).is_err());
        assert!(StencilPlane::of(&scene(0x0012_20ef, 0), 960, 544).is_err());
        // Wider than the stride is not a plane the pass can map one to one.
        assert!(StencilPlane::of(&scene(0x0012_20ef, 0x924c_0000), 992, 544).is_err());
    }

    /// A headless device, or `None` (the GPU tests skip).
    fn device() -> Option<(wgpu::Device, wgpu::Queue)> {
        let instance = wgpu::Instance::default();
        let adapter = pollster::block_on(instance.request_adapter(&wgpu::RequestAdapterOptions::default())).ok()?;
        pollster::block_on(adapter.request_device(&wgpu::DeviceDescriptor {
            label: Some("store-mirror-test"),
            required_features: wgpu::Features::empty(),
            required_limits: wgpu::Limits::downlevel_defaults(),
            experimental_features: wgpu::ExperimentalFeatures::disabled(),
            memory_hints: wgpu::MemoryHints::default(),
            trace: wgpu::Trace::Off,
        }))
        .ok()
    }

    fn read_back(device: &wgpu::Device, buf: &wgpu::Buffer) -> Vec<u8> {
        buf.slice(..).map_async(wgpu::MapMode::Read, |_| {});
        let _ = device.poll(wgpu::PollType::wait_indefinitely());
        buf.slice(..).get_mapped_range().expect("mapped").to_vec()
    }

    /// A stencil plane LOADED from a mirror lands in the stencil aspect byte for byte, and a
    /// STORE puts the attachment's stencil back at the plane's guest pitch, touching nothing
    /// between the rows - a round trip through a real depth-stencil attachment. The width is
    /// not a multiple of four, so the store's whole-word rule leaves the row's last two bytes
    /// as the mirror held them. Content-free; skips with no adapter.
    #[test]
    fn a_stencil_plane_round_trips_through_the_attachment() {
        let Some((device, queue)) = device() else {
            eprintln!("no GPU adapter - skipping the stencil plane copy test");
            return;
        };
        let (w, h, pitch, lead) = (22u32, 7u32, 32u32, 64u32);
        let lo = 0x0010_0000u32;
        let plane = StencilPlane { addr: lo + lead, pitch, width: w, height: h };
        let size = u64::from(lead + pitch * h + 64);
        let value = |x: u32, y: u32| (x * 7 + y * 13 + 1) as u8;
        let mut seed = vec![0xeeu8; size as usize];
        for y in 0..h {
            for x in 0..w {
                seed[(lead + y * pitch + x) as usize] = value(x, y);
            }
        }
        let buffer = |usage: wgpu::BufferUsages, size: u64| {
            device.create_buffer(&wgpu::BufferDescriptor { label: None, size, usage, mapped_at_creation: false })
        };
        let rw = wgpu::BufferUsages::STORAGE | wgpu::BufferUsages::COPY_SRC | wgpu::BufferUsages::COPY_DST;
        let readable = wgpu::BufferUsages::MAP_READ | wgpu::BufferUsages::COPY_DST;
        let mirror = buffer(rw, size);
        queue.write_buffer(&mirror, 0, &seed);
        let back = buffer(rw, size);
        queue.write_buffer(&back, 0, &vec![0x55u8; size as usize]);
        let tex = device.create_texture(&wgpu::TextureDescriptor {
            label: None,
            size: wgpu::Extent3d { width: w, height: h, depth_or_array_layers: 1 },
            mip_level_count: 1,
            sample_count: 1,
            dimension: wgpu::TextureDimension::D2,
            format: wgpu::TextureFormat::Depth24PlusStencil8,
            usage: wgpu::TextureUsages::RENDER_ATTACHMENT | wgpu::TextureUsages::COPY_SRC | wgpu::TextureUsages::COPY_DST,
            view_formats: &[],
        });
        let spitch = staging_pitch(w);
        let staging = buffer(rw, u64::from(spitch * h));
        let pipe = row_copy_pipeline(&device);
        let mut enc = device.create_command_encoder(&Default::default());
        let mut pool = BufferPool::default();
        let _load = encode_plane_load(&device, &queue, &mut enc, &pipe, &mirror, lo, &plane, &staging, &tex, &mut pool);
        let stencil = buffer(readable, u64::from(spitch * h));
        enc.copy_texture_to_buffer(
            wgpu::TexelCopyTextureInfo { texture: &tex, mip_level: 0, origin: wgpu::Origin3d::ZERO, aspect: wgpu::TextureAspect::StencilOnly },
            wgpu::TexelCopyBufferInfo {
                buffer: &stencil,
                layout: wgpu::TexelCopyBufferLayout { offset: 0, bytes_per_row: Some(spitch), rows_per_image: Some(h) },
            },
            wgpu::Extent3d { width: w, height: h, depth_or_array_layers: 1 },
        );
        let _store = encode_plane_store(&device, &queue, &mut enc, &pipe, &tex, &plane, &staging, &back, lo, &mut pool);
        let stored = buffer(readable, size);
        enc.copy_buffer_to_buffer(&back, 0, &stored, 0, size);
        queue.submit([enc.finish()]);
        let stencil = read_back(&device, &stencil);
        let stored = read_back(&device, &stored);
        for y in 0..h {
            for x in 0..w {
                assert_eq!(stencil[(y * spitch + x) as usize], value(x, y), "loaded stencil at ({x}, {y})");
            }
        }
        for (i, &b) in stored.iter().enumerate() {
            let at = i as u32;
            let in_rows = at >= lead && at < lead + pitch * h && (at - lead) % pitch < w / 4 * 4;
            let want = if in_rows { value((at - lead) % pitch, (at - lead) / pitch) } else { 0x55 };
            assert_eq!(b, want, "stored mirror byte {i}");
        }
    }

    /// The four edge-mask windows are one surface's row phases; the union is fingerprinted
    /// once per byte, so the same bytes give the same print whatever the window order.
    #[test]
    fn the_seed_fingerprint_covers_the_union_once() {
        let surface: Vec<u8> = (0..4000u32).map(|i| (i * 7) as u8).collect();
        let w = |k: u32| (0x1000 + 960 * k, std::sync::Arc::<[u8]>::from(&surface[(960 * k) as usize..(960 * k + 1000) as usize]));
        let a = vec![w(0), w(1), w(2)];
        let b = vec![w(2), w(0), w(1)];
        assert_eq!(seed_fingerprint(&a), seed_fingerprint(&b));
        let mut changed = surface.clone();
        changed[1500] ^= 1;
        let c = vec![w(0), (0x1000 + 960, std::sync::Arc::<[u8]>::from(&changed[960..1960])), w(2)];
        assert_ne!(seed_fingerprint(&a), seed_fingerprint(&c));
    }
}
