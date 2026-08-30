//! Guest texture -> compressed blocks, on the GPU.
//!
//! # Why this exists
//! Re-encoding a texture is the most expensive thing this emulator does, and until now it ran on
//! the CPU - the one resource the target device has least of. MEASURED on that device: a screen
//! transition spends **BUILD 21,182 ms** decoding and re-encoding the textures the new screen
//! binds, while over the same frame the GPU is idle at `pass 3.3 ms` against 128 ms of CPU. The
//! encoder runs at roughly 1 Mtexel/s there, so a single 1024x1024 atlas with its mip chain is
//! well over a second of frozen guest, and a transition binds a hundred textures at once.
//!
//! Block encoding is embarrassingly parallel over independent 4x4 blocks and touches no guest
//! state at all. It belongs on the processor that is doing nothing.
//!
//! # The shape, and why there is no readback
//! Three compute passes over one scratch buffer - decode the guest's blocks to RGBA8, box-filter
//! the mip chain, encode the target's blocks - then `copyBufferToTexture` straight into the
//! finished compressed texture. **Nothing crosses back to the CPU at any point**, so there is no
//! stall to schedule around, no fence to wait on, and no frame that has to be held back for a
//! map. The one thing the CPU still decides is whether the texture carries alpha, and that is
//! read from the source blocks' own opacity flags rather than from decoded texels - see
//! `vitaslop_runtime::render`.
//!
//! # It is not a quality trade in either direction
//! The shaders are ports of the CPU implementations they replace, arithmetic for arithmetic, at
//! the guest's own resolution. `gpu_etc2_matches_the_cpu_encoder` and
//! `gpu_pvrtc_matches_the_cpu_decoder` are what hold them to that, on a real device, against the
//! CPU code as the oracle.

use crate::gpu::{BlockFormat, CompressedUpload, GpuTranscode, SourceCodec};

/// Rows of a compressed level are padded to this, because `copyBufferToTexture` requires a
/// `bytes_per_row` that is a multiple of it and a packed block row is not.
const COPY_ROW_ALIGN: u32 = 256;

/// Uniform slots are spaced by this so a dynamic offset is always legal. 256 is the WebGPU
/// default `min_uniform_buffer_offset_alignment` and the maximum any adapter reports, so using it
/// unconditionally avoids a per-device query for a buffer that is a few kilobytes at most.
const UNIFORM_SLOT: u64 = 256;

/// The largest RGBA8 scratch chain this will allocate on the GPU. Past it the texture takes the
/// CPU path, which is slow but bounded, rather than failing an allocation on a phone.
///
/// 96 MB covers a 4096x4096 chain. The default `max_storage_buffer_binding_size` is 128 MB and
/// every adapter reports at least that, so this stays inside the guaranteed limit with room for
/// the driver's own rounding.
const MAX_SCRATCH_BYTES: u64 = 96 << 20;

/// >>> WHAT THE RGBA8 EXPANSION MAY HOLD RESIDENT, ACROSS ALL THREE OF ITS BUFFERS.
///
/// [`MAX_SCRATCH_BYTES`] bounds a TRANSIENT allocation: `run` sizes its buffers per texture and
/// destroys them, so 96 MB is a peak that exists for one submit. [`RawScratch`] is the opposite
/// - it is KEPT, grown to the largest texture ever seen and never shrunk - so the same number
/// there is 96 MB of RGBA scratch plus as much again of output plus the source, held for the
/// life of the run.
///
/// On a desktop that is invisible. On the target phone it is the whole device: a run took
/// **three GPU buffers sized for the largest texture of the session and then failed a
/// FORTY-EIGHT BYTE `createBuffer`**, which is what a WebGPU device with nothing left looks
/// like - and the panic named an innocent caller that happened to allocate next.
///
/// A texture that does not fit is REFUSED and takes the CPU decode, which is slower and
/// completely correct. That is the trade this codebase already makes everywhere else on this
/// path: the picture is never the variable, the cost is.
const RAW_SCRATCH_BUDGET: u64 = 24 << 20;

/// The compute pipelines, built once per device.
pub struct Transcoder {
    layout: wgpu::BindGroupLayout,
    /// The YUV path's own layout and pipeline: it writes its texels straight into a storage
    /// TEXTURE, where every other entry point writes into a storage buffer that is then copied
    /// into one. See [`Transcoder::convert_yuv420p2`].
    yuv_layout: wgpu::BindGroupLayout,
    yuv_to_texture: wgpu::ComputePipeline,
    /// The buffers the YUV path reuses across pictures - see [`YuvScratch`].
    yuv_scratch: std::cell::RefCell<Option<YuvScratch>>,
    /// The buffers the RGBA8 expansion reuses across textures - see [`RawScratch`].
    raw_scratch: std::cell::RefCell<Option<RawScratch>>,
    decode_pvrtc: wgpu::ComputePipeline,
    decode_bc: wgpu::ComputePipeline,
    /// The UNCOMPRESSED path: a Morton un-swizzle plus a channel permutation, which is the
    /// whole "decode" for the guest's 32-bit four-channel formats and the largest single item
    /// in the target device's texture work. See `decode_raw` in the shader.
    decode_raw: wgpu::ComputePipeline,
    /// Packed RGBA8 rows -> the 256-byte-aligned rows `copyBufferToTexture` requires.
    copy_rows: wgpu::ComputePipeline,
    halve: wgpu::ComputePipeline,
    encode_etc2: wgpu::ComputePipeline,
    convert_yuv: wgpu::ComputePipeline,
}

/// A two-plane 4:2:0 surface in the GUEST's own bytes: a full-resolution luma plane, then one
/// of interleaved chroma at half resolution in both axes.
///
/// The strides are the guest's, not ours - a texture is laid out however the title's decoder
/// wrote it, and normalising here would mean a copy of exactly the size this path exists to
/// avoid.
pub struct PlanarYuv<'a> {
    /// Visible size in texels.
    pub width: u32,
    pub height: u32,
    /// Bytes per row of the luma plane.
    pub luma_stride: u32,
    /// Bytes per row of the interleaved chroma plane.
    pub chroma_stride: u32,
    /// Byte offset of the chroma plane within `data`.
    pub chroma_offset: u32,
    /// The format's swizzle says the pair is Cr,Cb rather than Cb,Cr.
    pub swap_chroma: bool,
    /// The guest's bytes, both planes.
    pub data: &'a [u8],
}

/// The two buffers the video path keeps between pictures: the picture itself, and the 64
/// bytes of parameters that describe it.
///
/// A movie's pictures are all one shape, so these are allocated on the first one and reused
/// for the rest of the film. The source buffer only ever grows, because a shrink would mean
/// re-allocating on a size that came back.
struct YuvScratch {
    /// Capacity, in bytes, of [`YuvScratch::src`].
    src_bytes: u64,
    src: wgpu::Buffer,
    params: wgpu::Buffer,
}

impl YuvScratch {
    fn new(device: &wgpu::Device, src_bytes: u64) -> Self {
        YuvScratch {
            src_bytes,
            src: device.create_buffer(&wgpu::BufferDescriptor {
                label: Some("yuv-src"),
                size: src_bytes,
                usage: wgpu::BufferUsages::STORAGE | wgpu::BufferUsages::COPY_DST,
                mapped_at_creation: false,
            }),
            params: device.create_buffer(&wgpu::BufferDescriptor {
                label: Some("yuv-params"),
                size: UNIFORM_SLOT,
                usage: wgpu::BufferUsages::UNIFORM | wgpu::BufferUsages::COPY_DST,
                mapped_at_creation: false,
            }),
        }
    }
}

/// The buffers [`Transcoder::expand_rgba8`] reuses across textures and across frames.
///
/// # Why this is not a micro-optimisation on this path
/// The expansion runs for every guest texture the title rewrites - MEASURED on the user's
/// device at **26.7 a frame**. Without this each one created four `GPUBuffer`s and a bind
/// group, then destroyed them, and copied the guest's bytes into a temporary `Vec` purely to
/// pad the length to a multiple of four. That is over a hundred GPU object lifetimes and
/// megabytes of pointless `memcpy` per frame, on the engine where allocation is most expensive.
///
/// Grown, never shrunk, exactly as [`YuvScratch`] is: a title's texture sizes settle within the
/// first frames of a screen, and a re-allocation here is the thing this exists to avoid.
struct RawScratch {
    src_bytes: u64,
    rgba_bytes: u64,
    out_bytes: u64,
    src: wgpu::Buffer,
    rgba: wgpu::Buffer,
    out: wgpu::Buffer,
    params: wgpu::Buffer,
    /// Built once per set of buffers rather than per texture: the bind group names the four
    /// buffers and nothing else, and the per-level parameters ride in through a dynamic offset.
    bind: wgpu::BindGroup,
}

impl RawScratch {
    /// Bytes of parameter slots: two dispatches per level, and a chain is at most 14 levels
    /// (a 8192-texel side). Fixed rather than grown - it is a few kilobytes.
    const PARAM_BYTES: u64 = UNIFORM_SLOT * 32;

    fn new(
        device: &wgpu::Device,
        layout: &wgpu::BindGroupLayout,
        src_bytes: u64,
        rgba_bytes: u64,
        out_bytes: u64,
    ) -> Self {
        let mk = |size: u64, usage: wgpu::BufferUsages, label: &str| {
            device.create_buffer(&wgpu::BufferDescriptor {
                label: Some(label),
                size: size.max(4),
                usage,
                mapped_at_creation: false,
            })
        };
        let src = mk(
            src_bytes,
            wgpu::BufferUsages::STORAGE | wgpu::BufferUsages::COPY_DST,
            "texraw-src",
        );
        let rgba = mk(rgba_bytes, wgpu::BufferUsages::STORAGE, "texraw-rgba");
        let out = mk(
            out_bytes,
            wgpu::BufferUsages::STORAGE | wgpu::BufferUsages::COPY_SRC,
            "texraw-out",
        );
        let params = mk(
            Self::PARAM_BYTES,
            wgpu::BufferUsages::UNIFORM | wgpu::BufferUsages::COPY_DST,
            "texraw-params",
        );
        let bind = device.create_bind_group(&wgpu::BindGroupDescriptor {
            label: Some("texraw-bg"),
            layout,
            entries: &[
                wgpu::BindGroupEntry {
                    binding: 0,
                    resource: wgpu::BindingResource::Buffer(wgpu::BufferBinding {
                        buffer: &params,
                        offset: 0,
                        size: wgpu::BufferSize::new(Params::BYTES as u64),
                    }),
                },
                wgpu::BindGroupEntry { binding: 1, resource: src.as_entire_binding() },
                wgpu::BindGroupEntry { binding: 2, resource: rgba.as_entire_binding() },
                wgpu::BindGroupEntry { binding: 3, resource: out.as_entire_binding() },
            ],
        });
        RawScratch { src_bytes, rgba_bytes, out_bytes, src, rgba, out, params, bind }
    }

    /// Release the GPU memory NOW, rather than leaving four `GPUBuffer`s to the JavaScript
    /// collector - see `expand_rgba8`'s note on what dropping a handle in a browser does. Called
    /// when the scratch is REPLACED by a larger one, which is the only time it is discarded.
    fn destroy(self) {
        self.src.destroy();
        self.rgba.destroy();
        self.out.destroy();
        self.params.destroy();
    }
}

/// One level of the finished chain, and where its bytes live in the scratch buffers.
#[derive(Debug)]
struct Level {
    width: u32,
    height: u32,
    /// Word offset of this level's RGBA8 texels in the scratch buffer.
    rgba_word: u32,
    /// Byte offset of this level's blocks in the output buffer. A multiple of
    /// [`COPY_ROW_ALIGN`], which is what makes it a legal copy source offset.
    out_byte: u32,
    /// Padded bytes per block row - what `copyBufferToTexture` is given.
    out_row_bytes: u32,
    blocks_x: u32,
    blocks_y: u32,
}

/// One RGBA8 level of an [`Transcoder::expand_rgba8`] chain: where its texels sit in the packed
/// scratch, and where its ALIGNED rows sit in the buffer the copy reads.
///
/// Separate from [`Level`] because that one counts BLOCKS - its copy extent is `blocks * 4` -
/// and an RGBA8 level's extent is its texels. Sharing the struct would mean a field that means
/// two different things depending on the caller, which is how a copy extent reached a phone
/// wrong once already.
struct RgbaLevel {
    width: u32,
    height: u32,
    /// Word offset of this level's texels in the packed scratch buffer.
    rgba_word: u32,
    /// Byte offset of this level's aligned rows in the output buffer.
    out_byte: u32,
    /// Padded bytes per row - what `copyBufferToTexture` is given.
    out_row_bytes: u32,
}

#[derive(Clone, Copy, Default)]
struct Params {
    width: u32,
    height: u32,
    blocks_x: u32,
    blocks_y: u32,
    padded_x: u32,
    padded_y: u32,
    src_word: u32,
    rgba_word: u32,
    out_word: u32,
    src_width: u32,
    src_height: u32,
    out_row_words: u32,
    flags: u32,
    src_format: u32,
    src_block_words: u32,
    _pad: u32,
}

impl Params {
    /// The 64 bytes the shader's `Params` struct reads. Written field by field rather than
    /// transmuted: a `repr(C)` reinterpretation would silently depend on Rust's layout matching
    /// WGSL's, and a mismatch there is a texture decoded out of the wrong offsets rather than an
    /// error.
    const BYTES: usize = 64;

    fn to_bytes(self) -> [u8; Self::BYTES] {
        let words = [
            self.width,
            self.height,
            self.blocks_x,
            self.blocks_y,
            self.padded_x,
            self.padded_y,
            self.src_word,
            self.rgba_word,
            self.out_word,
            self.src_width,
            self.src_height,
            self.out_row_words,
            self.flags,
            self.src_format,
            self.src_block_words,
            self._pad,
        ];
        let mut out = [0u8; Self::BYTES];
        for (i, w) in words.iter().enumerate() {
            out[i * 4..i * 4 + 4].copy_from_slice(&w.to_le_bytes());
        }
        out
    }
}

const FLAG_SWIZZLED: u32 = 1;
const FLAG_PVRTC2: u32 = 2;
const FLAG_4BPP: u32 = 4;
const FLAG_ALPHA: u32 = 8;
/// `convert_yuv420p2` only: the format's swizzle says the chroma pair is Cr,Cb not Cb,Cr.
const FLAG_SWAP_CHROMA: u32 = 16;

fn align_up(v: u32, to: u32) -> u32 {
    v.div_ceil(to) * to
}

impl Transcoder {
    pub fn new(device: &wgpu::Device) -> Self {
        let module = device.create_shader_module(wgpu::ShaderModuleDescriptor {
            label: Some("texenc"),
            source: wgpu::ShaderSource::Wgsl(include_str!("texenc.wgsl").into()),
        });
        let layout = device.create_bind_group_layout(&wgpu::BindGroupLayoutDescriptor {
            label: Some("texenc-bgl"),
            entries: &[
                wgpu::BindGroupLayoutEntry {
                    binding: 0,
                    visibility: wgpu::ShaderStages::COMPUTE,
                    ty: wgpu::BindingType::Buffer {
                        ty: wgpu::BufferBindingType::Uniform,
                        has_dynamic_offset: true,
                        min_binding_size: wgpu::BufferSize::new(
                            Params::BYTES as u64
                        ),
                    },
                    count: None,
                },
                storage_entry(1, true),
                storage_entry(2, false),
                storage_entry(3, false),
            ],
        });
        let pipe_layout = device.create_pipeline_layout(&wgpu::PipelineLayoutDescriptor {
            label: Some("texenc-pl"),
            bind_group_layouts: &[Some(&layout)],
            immediate_size: 0,
        });
        let make = |entry: &str| {
            device.create_compute_pipeline(&wgpu::ComputePipelineDescriptor {
                label: Some(entry),
                layout: Some(&pipe_layout),
                module: &module,
                entry_point: Some(entry),
                compilation_options: Default::default(),
                cache: None,
            })
        };
        let yuv_layout = device.create_bind_group_layout(&wgpu::BindGroupLayoutDescriptor {
            label: Some("texenc-yuv-bgl"),
            entries: &[
                wgpu::BindGroupLayoutEntry {
                    binding: 0,
                    visibility: wgpu::ShaderStages::COMPUTE,
                    ty: wgpu::BindingType::Buffer {
                        ty: wgpu::BufferBindingType::Uniform,
                        has_dynamic_offset: true,
                        min_binding_size: wgpu::BufferSize::new(Params::BYTES as u64),
                    },
                    count: None,
                },
                storage_entry(1, true),
                wgpu::BindGroupLayoutEntry {
                    binding: 4,
                    visibility: wgpu::ShaderStages::COMPUTE,
                    ty: wgpu::BindingType::StorageTexture {
                        access: wgpu::StorageTextureAccess::WriteOnly,
                        format: wgpu::TextureFormat::Rgba8Unorm,
                        view_dimension: wgpu::TextureViewDimension::D2,
                    },
                    count: None,
                },
            ],
        });
        let yuv_pl = device.create_pipeline_layout(&wgpu::PipelineLayoutDescriptor {
            label: Some("texenc-yuv-pl"),
            bind_group_layouts: &[Some(&yuv_layout)],
            immediate_size: 0,
        });
        let yuv_to_texture = device.create_compute_pipeline(&wgpu::ComputePipelineDescriptor {
            label: Some("convert_yuv420p2_tex"),
            layout: Some(&yuv_pl),
            module: &module,
            entry_point: Some("convert_yuv420p2_tex"),
            compilation_options: Default::default(),
            cache: None,
        });
        Self {
            yuv_layout,
            yuv_to_texture,
            yuv_scratch: std::cell::RefCell::new(None),
            raw_scratch: std::cell::RefCell::new(None),
            decode_pvrtc: make("decode_pvrtc"),
            decode_bc: make("decode_bc"),
            decode_raw: make("decode_raw"),
            copy_rows: make("copy_rows"),
            halve: make("halve"),
            encode_etc2: make("encode_etc2"),
            convert_yuv: make("convert_yuv420p2"),
            layout,
        }
    }

    /// Convert a two-plane 4:2:0 (video) surface to an RGBA texture ON THE GPU.
    ///
    /// `None` when the shape is outside what the shader covers, in which case the caller
    /// falls back to the CPU conversion, which produces the same picture and only costs
    /// more. See the entry point in `texenc.wgsl` for why this is worth doing at all: a
    /// decoded video frame is the one texture whose content changes every frame, so its
    /// conversion is paid per frame and no cache can help.
    ///
    /// # >>> WHAT A PER-FRAME PATH MUST NOT DO, and this one used to do all of it
    ///
    /// Everything here happens thirty times a second for as long as a movie runs, so an
    /// allocation is not a one-off cost, it is a rate. The first version copied the picture
    /// (`to_vec`, to pad it to a multiple of four), created a storage buffer from that copy,
    /// created a second buffer the size of the RGBA output, created a third for a binding the
    /// entry point never reads, created a fourth for 64 bytes of parameters, and then copied
    /// the output buffer into a texture - so a 0.75 MB picture cost two CPU copies of itself,
    /// four buffer allocations, and 2 MB of GPU-side copying, per frame.
    ///
    /// Now: the source is written straight into a buffer kept from the last picture
    /// ([`YuvScratch`]), the parameters into another, and the shader writes its texels
    /// directly into the destination texture through a write-only storage binding, which is
    /// what removes the output buffer and the copy after it. Nothing is allocated per picture
    /// except the destination texture itself, which is what the caller asked for.
    pub fn convert_yuv420p2(
        &self,
        device: &wgpu::Device,
        queue: &wgpu::Queue,
        planes: &PlanarYuv,
    ) -> Option<wgpu::Texture> {
        let (w, h) = (planes.width, planes.height);
        if w == 0 || h == 0 {
            return None;
        }
        if w as u64 * h as u64 * 4 > MAX_SCRATCH_BYTES {
            return None;
        }
        // The chroma plane's last row must be inside the bytes we were given, or the shader
        // would address past the buffer. Refusing is not a fallback for a defect - a short
        // snapshot is a legitimate outcome when a guest allocation ends early.
        let need = planes.chroma_offset as u64
            + planes.chroma_stride as u64 * (h.div_ceil(2) - 1) as u64
            + w.div_ceil(2) as u64 * 2;
        if (planes.data.len() as u64) < need {
            return None;
        }

        let src_bytes = align_up(planes.data.len() as u32, 4) as u64;
        let mut held = self.yuv_scratch.borrow_mut();
        let scratch = match held.as_ref() {
            // Grown, never shrunk: a movie's pictures are all one size, and a re-allocation
            // here is the thing this exists to avoid.
            Some(s) if s.src_bytes >= src_bytes => held.as_ref().expect("just matched"),
            _ => {
                *held = Some(YuvScratch::new(device, src_bytes));
                held.as_ref().expect("just set")
            }
        };

        // >>> THE PICTURE IS WRITTEN, NOT COPIED THEN WRITTEN. `write_buffer` wants a length
        // that is a multiple of four; the last few bytes of an odd-length picture go through a
        // padded tail rather than through a copy of the whole thing.
        let whole = planes.data.len() & !3;
        queue.write_buffer(&scratch.src, 0, &planes.data[..whole]);
        if whole < planes.data.len() {
            let mut tail = [0u8; 4];
            tail[..planes.data.len() - whole].copy_from_slice(&planes.data[whole..]);
            queue.write_buffer(&scratch.src, whole as u64, &tail);
        }
        let params = Params {
            width: w,
            height: h,
            src_word: planes.luma_stride,
            rgba_word: planes.chroma_stride,
            out_word: planes.chroma_offset,
            out_row_words: w,
            flags: if planes.swap_chroma { FLAG_SWAP_CHROMA } else { 0 },
            ..Params::default()
        };
        queue.write_buffer(&scratch.params, 0, &params.to_bytes());

        crate::gpu::note_texture_created();
        let texture = device.create_texture(&wgpu::TextureDescriptor {
            label: Some("gxp-tex-yuv"),
            size: wgpu::Extent3d { width: w, height: h, depth_or_array_layers: 1 },
            mip_level_count: 1,
            sample_count: 1,
            dimension: wgpu::TextureDimension::D2,
            format: wgpu::TextureFormat::Rgba8Unorm,
            // COPY_SRC so the picture can be read back: `gpu_yuv_conversion_matches_the_cpu`
            // is the only thing standing between this path and another silently black movie.
            usage: wgpu::TextureUsages::TEXTURE_BINDING
                | wgpu::TextureUsages::STORAGE_BINDING
                | wgpu::TextureUsages::COPY_DST
                | wgpu::TextureUsages::COPY_SRC,
            view_formats: &[],
        });
        let view = texture.create_view(&wgpu::TextureViewDescriptor::default());
        let bg = device.create_bind_group(&wgpu::BindGroupDescriptor {
            label: Some("yuv-bg"),
            layout: &self.yuv_layout,
            entries: &[
                wgpu::BindGroupEntry {
                    binding: 0,
                    resource: wgpu::BindingResource::Buffer(wgpu::BufferBinding {
                        buffer: &scratch.params,
                        offset: 0,
                        size: wgpu::BufferSize::new(Params::BYTES as u64),
                    }),
                },
                wgpu::BindGroupEntry { binding: 1, resource: scratch.src.as_entire_binding() },
                wgpu::BindGroupEntry { binding: 4, resource: wgpu::BindingResource::TextureView(&view) },
            ],
        });

        let mut enc = device.create_command_encoder(&wgpu::CommandEncoderDescriptor {
            label: Some("yuv-convert"),
        });
        {
            let mut pass = enc.begin_compute_pass(&wgpu::ComputePassDescriptor {
                label: Some("yuv-convert"),
                timestamp_writes: None,
            });
            pass.set_pipeline(&self.yuv_to_texture);
            pass.set_bind_group(0, &bg, &[0]);
            pass.dispatch_workgroups(w.div_ceil(2).div_ceil(8), h.div_ceil(2).div_ceil(8), 1);
        }
        queue.submit([enc.finish()]);
        Some(texture)
    }


    /// Build the finished compressed texture for `plan`, or `None` if this adapter or this
    /// texture is outside what the shaders cover - in which case the caller falls back to the
    /// CPU decode, which is slow but always available.
    ///
    /// `None` is never a silent failure: every rejection here is a shape the CPU path handles
    /// correctly, so the picture is the same either way and only the cost differs.
    pub fn run(
        &self,
        device: &wgpu::Device,
        queue: &wgpu::Queue,
        upload: &CompressedUpload,
        plan: &GpuTranscode,
        gamma: bool,
    ) -> Option<wgpu::Texture> {
        // ETC2 is the only target implemented here. BC stays on the CPU deliberately: it runs at
        // 95 Mtexel/s, so it was never the bottleneck, and the desktop's headless render is the
        // determinism oracle every capture in this project is compared against. Moving the
        // encoder that produces its blocks would retire that comparison to fix a cost that does
        // not exist on that engine.
        let alpha = match upload.format {
            BlockFormat::Etc2Rgb8 => false,
            BlockFormat::Etc2Rgba8 => true,
            BlockFormat::Bc1 | BlockFormat::Bc2 | BlockFormat::Bc3 => return None,
        };
        // >>> ASK THE DEVICE, HERE, RATHER THAN TRUST THE CALLER.
        //
        // The caller does resolve the family from `device.features()` and filters the upload by
        // it, so this is redundant TODAY. It is asked anyway because of what the redundancy is
        // insurance against: creating a texture with a format the device lacks is not a soft
        // failure. The texture comes back INVALID, and every view, bind group, pass and submit
        // built on it is invalid in turn - which reaches the screen as black, with the cause
        // thousands of validation messages earlier. That is the exact shape the compatibility-mode
        // work already had to diagnose once (5 failed targets, 4,776 invalid bind groups, one
        // black frame).
        //
        // The invariant currently lives three layers up from the `create_texture` below. One
        // feature-bit test, cached by the caller's own `bc_supported` pattern, keeps a future
        // restructure of that call chain to a SLOW frame rather than a black one.
        if !device.features().contains(wgpu::Features::TEXTURE_COMPRESSION_ETC2) {
            return None;
        }
        let levels = self.plan_levels(plan, upload.format)?;
        let rgba_words: u64 = levels
            .iter()
            .map(|l| l.width as u64 * l.height as u64)
            .sum();
        if rgba_words * 4 > MAX_SCRATCH_BYTES {
            return None;
        }
        let out_bytes: u64 = levels
            .last()
            .map(|l| l.out_byte as u64 + l.out_row_bytes as u64 * l.blocks_y as u64)?;

        // The guest's own bytes, padded to a whole number of words so the shader can read them
        // as `array<u32>`. The pad is zeros, and the shader never addresses past a block it was
        // told exists.
        let mut src_bytes = plan.src.to_vec();
        while src_bytes.len() % 4 != 0 {
            src_bytes.push(0);
        }
        // `create_buffer_init` is `mappedAtCreation` on the web backend and every call takes a
        // renderer-side staging region; this path runs per transcoded texture, which on a screen
        // transition is a hundred of them. See the note in `gxm`'s depth cache for the crash
        // that shape produces and why the message blames the wrong allocation.
        let src_buf = device.create_buffer(&wgpu::BufferDescriptor {
            label: Some("texenc-src"),
            size: src_bytes.len().max(4) as u64,
            usage: wgpu::BufferUsages::STORAGE | wgpu::BufferUsages::COPY_DST,
            mapped_at_creation: false,
        });
        queue.write_buffer(&src_buf, 0, &src_bytes);
        let rgba_buf = device.create_buffer(&wgpu::BufferDescriptor {
            label: Some("texenc-rgba"),
            size: (rgba_words * 4).max(4),
            usage: wgpu::BufferUsages::STORAGE,
            mapped_at_creation: false,
        });
        let out_buf = device.create_buffer(&wgpu::BufferDescriptor {
            label: Some("texenc-out"),
            size: out_bytes.max(4),
            usage: wgpu::BufferUsages::STORAGE | wgpu::BufferUsages::COPY_SRC,
            mapped_at_creation: false,
        });

        // Every dispatch's parameters, in one buffer, addressed by dynamic offset. Building them
        // up front is what keeps this to a single bind group for the whole texture.
        let base_flags = codec_flags(plan.codec) | (u32::from(alpha) * FLAG_ALPHA);
        let mut slots: Vec<u8> = Vec::new();
        let mut push = |p: Params| {
            slots.extend_from_slice(&p.to_bytes());
            slots.resize(align_up(slots.len() as u32, UNIFORM_SLOT as u32) as usize, 0);
        };

        // Phase 1: one dispatch per level, decoding the guest's own levels and box-filtering the
        // rest. Dispatches inside one compute pass are ordered and their writes are visible to
        // the next, which is what lets the filter read the level the dispatch before it wrote.
        for (i, l) in levels.iter().enumerate() {
            match plan.src_levels.get(i) {
                Some(s) => push(Params {
                    width: l.width,
                    height: l.height,
                    blocks_x: s.blocks_x,
                    blocks_y: s.blocks_y,
                    padded_x: s.padded_x,
                    padded_y: s.padded_y,
                    src_word: s.byte_offset / 4,
                    rgba_word: l.rgba_word,
                    flags: base_flags | (u32::from(s.swizzled) * FLAG_SWIZZLED),
                    src_format: codec_format(plan.codec),
                    src_block_words: plan.codec.block_bytes() / 4,
                    ..Default::default()
                }),
                None => {
                    let prev = &levels[i - 1];
                    push(Params {
                        width: l.width,
                        height: l.height,
                        rgba_word: l.rgba_word,
                        src_word: prev.rgba_word,
                        src_width: prev.width,
                        src_height: prev.height,
                        flags: base_flags,
                        ..Default::default()
                    })
                }
            }
        }
        // Phase 2: one dispatch per level, encoding blocks.
        for l in &levels {
            push(Params {
                width: l.width,
                height: l.height,
                rgba_word: l.rgba_word,
                out_word: l.out_byte / 4,
                out_row_words: l.out_row_bytes / 4,
                flags: base_flags,
                ..Default::default()
            });
        }
        let params_buf = device.create_buffer(&wgpu::BufferDescriptor {
            label: Some("texenc-params"),
            size: slots.len().max(4) as u64,
            usage: wgpu::BufferUsages::UNIFORM | wgpu::BufferUsages::COPY_DST,
            mapped_at_creation: false,
        });
        queue.write_buffer(&params_buf, 0, &slots);
        let bg = device.create_bind_group(&wgpu::BindGroupDescriptor {
            label: Some("texenc-bg"),
            layout: &self.layout,
            entries: &[
                wgpu::BindGroupEntry {
                    binding: 0,
                    resource: wgpu::BindingResource::Buffer(wgpu::BufferBinding {
                        buffer: &params_buf,
                        offset: 0,
                        size: wgpu::BufferSize::new(Params::BYTES as u64),
                    }),
                },
                wgpu::BindGroupEntry { binding: 1, resource: src_buf.as_entire_binding() },
                wgpu::BindGroupEntry { binding: 2, resource: rgba_buf.as_entire_binding() },
                wgpu::BindGroupEntry { binding: 3, resource: out_buf.as_entire_binding() },
            ],
        });

        crate::gpu::note_texture_created();
        let texture = device.create_texture(&wgpu::TextureDescriptor {
            label: Some("gxp-tex-gpu"),
            size: wgpu::Extent3d {
                width: plan.width.max(1),
                height: plan.height.max(1),
                depth_or_array_layers: 1,
            },
            mip_level_count: levels.len() as u32,
            sample_count: 1,
            dimension: wgpu::TextureDimension::D2,
            format: crate::gpu::block_wgpu_format_pub(upload.format, gamma),
            // >>> COPY_SRC IS HERE SO THE FINISHED TEXTURE CAN BE READ BACK AND CHECKED.
            //
            // Nothing in the renderer copies out of it. It is here because the alternative was a
            // test-only code path that builds its own texture, and "the verification runs
            // different code from the thing that ships" is precisely how the copy extent reached
            // a phone wrong: the shaders were each verified through their own readback helpers
            // while `run` itself - the buffer sizes, the dynamic offsets, the copies - was never
            // executed anywhere. `gpu_transcode_round_trips_through_a_real_etc2_texture` drives
            // THIS function and reads THIS texture.
            //
            // The cost is a flag on a sampled texture. It can stop a driver electing a
            // compressed-in-memory layout for a render target; this is neither.
            usage: wgpu::TextureUsages::TEXTURE_BINDING
                | wgpu::TextureUsages::COPY_DST
                | wgpu::TextureUsages::COPY_SRC,
            view_formats: &[],
        });

        let mut enc = device.create_command_encoder(&wgpu::CommandEncoderDescriptor {
            label: Some("texenc"),
        });
        {
            let mut pass = enc.begin_compute_pass(&wgpu::ComputePassDescriptor {
                label: Some("texenc-pass"),
                timestamp_writes: None,
            });
            for (i, l) in levels.iter().enumerate() {
                let off = (i as u64) * UNIFORM_SLOT;
                match plan.src_levels.get(i) {
                    Some(s) => {
                        pass.set_pipeline(self.decoder(plan.codec));
                        pass.set_bind_group(0, &bg, &[off as u32]);
                        pass.dispatch_workgroups(s.blocks_x.div_ceil(8), s.blocks_y.div_ceil(8), 1);
                    }
                    None => {
                        pass.set_pipeline(&self.halve);
                        pass.set_bind_group(0, &bg, &[off as u32]);
                        pass.dispatch_workgroups(l.width.div_ceil(8), l.height.div_ceil(8), 1);
                    }
                }
            }
            pass.set_pipeline(&self.encode_etc2);
            for (i, l) in levels.iter().enumerate() {
                let off = ((levels.len() + i) as u64) * UNIFORM_SLOT;
                pass.set_bind_group(0, &bg, &[off as u32]);
                pass.dispatch_workgroups(l.blocks_x.div_ceil(8), l.blocks_y.div_ceil(8), 1);
            }
        }
        record_copies(&mut enc, &out_buf, &texture, &levels);
        queue.submit([enc.finish()]);

        // >>> DESTROYED, NOT DROPPED. In the browser a `wgpu::Buffer` is a `GPUBuffer` living in
        // JavaScript and dropping the Rust handle only makes it GARBAGE; the GPU memory behind it
        // comes back whenever the JS collector next feels like it, which is not a schedule a path
        // that allocates tens of megabytes per texture can rely on. WebGPU defines `destroy()` on
        // a buffer with work in flight as completing that work before releasing the memory, and
        // the submit above is that work.
        src_buf.destroy();
        rgba_buf.destroy();
        out_buf.destroy();
        params_buf.destroy();
        Some(texture)
    }

    /// Build a finished RGBA8 texture, with its whole mip chain, WITHOUT the CPU touching a
    /// texel - for the guest's uncompressed 32-bit four-channel formats.
    ///
    /// # What this replaces, and why it is the biggest item on the target device
    /// The CPU path expands one of these texel by texel (`render::decode_uncompressed_at` plus
    /// the Morton tables), box-filters a chain in RGBA8, and hands the whole expansion to
    /// `writeTexture`. MEASURED on the user's device over a run: **2,964 MB of its 2,989 MB of
    /// texture decode is this one format**, and on a gameplay frame the decode and the upload
    /// together are 17 ms of a 25 ms render.
    ///
    /// Through here the CPU writes the guest's own bytes into a buffer and issues the
    /// dispatches. Nothing crosses back, so there is no stall to schedule around.
    ///
    /// # It is not a quality question, unlike every other shader in this file
    /// The others reconstruct texels from a lossy block format, so "does it match the CPU" is a
    /// real question with a real answer. This is a PERMUTATION - the Morton un-interleave and
    /// the SWIZZLE4 channel order - so equality is a property of the code. There is a test
    /// against the CPU decoder as the oracle anyway, because "it cannot differ" is exactly the
    /// kind of claim that turns out to differ.
    ///
    /// `None` is never a failure: every shape declined here is one the CPU decode handles
    /// correctly, so the picture is the same either way and only the cost differs.
    ///
    /// `into` is a texture of the SAME shape and format to write into instead of creating one -
    /// see the caller for why a guest texture the title rewrites must not get a new GPU object
    /// every frame.
    pub fn expand_rgba8(
        &self,
        device: &wgpu::Device,
        queue: &wgpu::Queue,
        plan: &crate::gpu::GpuRawExpand,
        gamma: bool,
        into: Option<wgpu::Texture>,
    ) -> Option<wgpu::Texture> {
        // >>> ANYTHING HANDED IN AND NOT USED IS DESTROYED, NEVER DROPPED.
        //
        // The caller has already taken this texture OUT of its cache, so this function owns the
        // only handle. In the browser a `wgpu::Texture` is a `GPUTexture` living in JavaScript
        // and dropping the Rust handle only makes it COLLECTABLE - the GPU memory comes back
        // whenever the collector next feels like it. On a path that supersedes twenty-odd
        // textures a frame that is not a schedule anything can rely on: it is a leak measured in
        // megabytes per second, and it ends as a device with nothing left, where even a 48-byte
        // `createBuffer` fails and the worker dies with no frame of its own to blame.
        let mut spare = into;
        let mut give_up = |spare: Option<wgpu::Texture>| {
            if let Some(t) = spare {
                t.destroy();
            }
            None::<wgpu::Texture>
        };
        let (w0, h0) = (plan.width.max(1), plan.height.max(1));
        if plan.src_levels.is_empty() || plan.levels == 0 {
            return give_up(spare);
        }
        let mut levels: Vec<RgbaLevel> = Vec::with_capacity(plan.levels as usize);
        let (mut rgba_word, mut out_byte) = (0u32, 0u32);
        for i in 0..plan.levels {
            let width = (w0 >> i).max(1);
            let height = (h0 >> i).max(1);
            let out_row_bytes = align_up(width * 4, COPY_ROW_ALIGN);
            levels.push(RgbaLevel { width, height, rgba_word, out_byte, out_row_bytes });
            let Some(rw) = rgba_word.checked_add(width.checked_mul(height).unwrap_or(u32::MAX))
            else {
                return give_up(spare);
            };
            let Some(ob) =
                out_byte.checked_add(out_row_bytes.checked_mul(height).unwrap_or(u32::MAX))
            else {
                return give_up(spare);
            };
            rgba_word = rw;
            out_byte = ob;
        }
        // Judged against what may be KEPT, not against the transient peak - see
        // `RAW_SCRATCH_BUDGET`. Checked here so an oversized texture is refused before any
        // buffer is sized for it.
        if (rgba_word as u64) * 4 > RAW_SCRATCH_BUDGET {
            return give_up(spare);
        }
        // Every guest level must be fully inside the bytes we were given, or a dispatch would
        // address past the buffer. A short snapshot is a legitimate outcome (an allocation that
        // ends early), so this refuses rather than reads.
        for (i, sl) in plan.src_levels.iter().enumerate() {
            let Some(l) = levels.get(i) else {
                return give_up(spare);
            };
            let texels = if sl.swizzled {
                (sl.padded_x as u64) * (sl.padded_y as u64)
            } else {
                (sl.blocks_x as u64) * (l.height as u64)
            };
            if sl.byte_offset as u64 + texels * 4 > plan.src.len() as u64 {
                return give_up(spare);
            }
        }

        // The scratch, grown to fit and then kept - see `RawScratch`.
        let need_src = align_up(plan.src.len() as u32, 4) as u64;
        let (need_rgba, need_out) = ((rgba_word as u64) * 4, out_byte as u64);
        let mut held = self.raw_scratch.borrow_mut();
        let fits = held.as_ref().is_some_and(|k| {
            k.src_bytes >= need_src && k.rgba_bytes >= need_rgba && k.out_bytes >= need_out
        });
        if !fits {
            // Grown to the LARGEST seen so far in every axis, so a small texture after a large
            // one does not re-allocate its way back down and then up again.
            let (s0, r0, o0) = held
                .as_ref()
                .map_or((0, 0, 0), |k| (k.src_bytes, k.rgba_bytes, k.out_bytes));
            let (want_src, want_rgba, want_out) =
                (need_src.max(s0), need_rgba.max(r0), need_out.max(o0));
            // The kept scratch is bounded, and the bound is over the SUM - three buffers each
            // individually reasonable still add up to a device.
            if want_src + want_rgba + want_out > RAW_SCRATCH_BUDGET {
                return give_up(spare);
            }
            // DESTROYED, not dropped: the scratch being replaced holds four `GPUBuffer`s, and
            // leaving them to the collector leaks the previous size every time this grows.
            if let Some(old) = held.take() {
                old.destroy();
            }
            *held = Some(RawScratch::new(device, &self.layout, want_src, want_rgba, want_out));
        }
        let scratch = held.as_ref().expect("just set");

        // >>> WRITTEN, NOT COPIED THEN WRITTEN. `write_buffer` wants a length that is a
        // multiple of four; the last few bytes of an odd-length texture go through a padded
        // tail rather than through a copy of the whole thing - which for this path is
        // megabytes a frame.
        let whole = plan.src.len() & !3;
        queue.write_buffer(&scratch.src, 0, &plan.src[..whole]);
        if whole < plan.src.len() {
            let mut tail = [0u8; 4];
            tail[..plan.src.len() - whole].copy_from_slice(&plan.src[whole..]);
            queue.write_buffer(&scratch.src, whole as u64, &tail);
        }

        let mut slots: Vec<u8> = Vec::new();
        let mut push = |p: Params| {
            slots.extend_from_slice(&p.to_bytes());
            slots.resize(align_up(slots.len() as u32, UNIFORM_SLOT as u32) as usize, 0);
        };
        // Phase 1: decode the guest's own levels, box-filter the rest. Dispatches in one pass
        // are ordered and their writes are visible to the next, which is what lets the filter
        // read the level the dispatch before it wrote.
        for (i, l) in levels.iter().enumerate() {
            match plan.src_levels.get(i) {
                Some(sl) => push(Params {
                    width: l.width,
                    height: l.height,
                    blocks_x: sl.blocks_x,
                    blocks_y: sl.blocks_y,
                    padded_x: sl.padded_x,
                    padded_y: sl.padded_y,
                    src_word: sl.byte_offset / 4,
                    rgba_word: l.rgba_word,
                    flags: u32::from(sl.swizzled) * FLAG_SWIZZLED,
                    // The shader reads the SWIZZLE4 selector here; there is no block format.
                    src_format: plan.swizzle,
                    ..Default::default()
                }),
                None => {
                    let prev = &levels[i - 1];
                    push(Params {
                        width: l.width,
                        height: l.height,
                        rgba_word: l.rgba_word,
                        src_word: prev.rgba_word,
                        src_width: prev.width,
                        src_height: prev.height,
                        ..Default::default()
                    })
                }
            }
        }
        // Phase 2: lay each level out with the aligned rows `copyBufferToTexture` requires.
        for l in &levels {
            push(Params {
                width: l.width,
                height: l.height,
                rgba_word: l.rgba_word,
                out_word: l.out_byte / 4,
                out_row_words: l.out_row_bytes / 4,
                ..Default::default()
            });
        }
        // A chain deeper than the fixed parameter buffer holds is refused rather than
        // truncated: the CPU decode handles it correctly and only costs more.
        if slots.len() as u64 > RawScratch::PARAM_BYTES {
            return give_up(spare);
        }
        queue.write_buffer(&scratch.params, 0, &slots);
        let bg = &scratch.bind;

        let format = if gamma {
            wgpu::TextureFormat::Rgba8UnormSrgb
        } else {
            wgpu::TextureFormat::Rgba8Unorm
        };
        // >>> WRITTEN INTO, NOT REPLACED, when the caller has the right texture already.
        //
        // The shape is re-checked here rather than trusted: a copy into a texture of the wrong
        // extent or mip count is a validation failure, and the caller's own test is a hash of
        // guest state rather than a look at the GPU object.
        let reusable = spare.as_ref().is_some_and(|t| {
            t.width() == w0
                && t.height() == h0
                && t.mip_level_count() == levels.len() as u32
                && t.format() == format
        });
        let texture = match spare.take() {
            Some(t) if reusable => t,
            other => {
                // The wrong shape is still OUR handle to release.
                if let Some(t) = other {
                    t.destroy();
                }
                crate::gpu::note_texture_created();
                device.create_texture(&wgpu::TextureDescriptor {
            label: Some("gxp-tex-raw"),
            size: wgpu::Extent3d { width: w0, height: h0, depth_or_array_layers: 1 },
            mip_level_count: levels.len() as u32,
            sample_count: 1,
            dimension: wgpu::TextureDimension::D2,
            format,
            // COPY_SRC for the reason the transcode's texture carries it: the test that holds
            // this to the CPU decoder drives THIS function and reads THIS texture, rather than a
            // test-only path that would not be the thing that ships.
            usage: wgpu::TextureUsages::TEXTURE_BINDING
                | wgpu::TextureUsages::COPY_DST
                | wgpu::TextureUsages::COPY_SRC,
            view_formats: &[],
                })
            }
        };

        let mut enc = device.create_command_encoder(&wgpu::CommandEncoderDescriptor {
            label: Some("texraw"),
        });
        {
            let mut pass = enc.begin_compute_pass(&wgpu::ComputePassDescriptor {
                label: Some("texraw-pass"),
                timestamp_writes: None,
            });
            for (i, l) in levels.iter().enumerate() {
                let off = ((i as u64) * UNIFORM_SLOT) as u32;
                if plan.src_levels.get(i).is_some() {
                    pass.set_pipeline(&self.decode_raw);
                } else {
                    pass.set_pipeline(&self.halve);
                }
                pass.set_bind_group(0, bg, &[off]);
                pass.dispatch_workgroups(l.width.div_ceil(8), l.height.div_ceil(8), 1);
            }
            pass.set_pipeline(&self.copy_rows);
            for (i, l) in levels.iter().enumerate() {
                let off = (((levels.len() + i) as u64) * UNIFORM_SLOT) as u32;
                pass.set_bind_group(0, bg, &[off]);
                pass.dispatch_workgroups(l.width.div_ceil(8), l.height.div_ceil(8), 1);
            }
        }
        for (i, l) in levels.iter().enumerate() {
            enc.copy_buffer_to_texture(
                wgpu::TexelCopyBufferInfo {
                    buffer: &scratch.out,
                    layout: wgpu::TexelCopyBufferLayout {
                        offset: l.out_byte as u64,
                        bytes_per_row: Some(l.out_row_bytes),
                        rows_per_image: Some(l.height),
                    },
                },
                wgpu::TexelCopyTextureInfo {
                    texture: &texture,
                    mip_level: i as u32,
                    origin: wgpu::Origin3d::ZERO,
                    aspect: wgpu::TextureAspect::All,
                },
                wgpu::Extent3d { width: l.width, height: l.height, depth_or_array_layers: 1 },
            );
        }
        queue.submit([enc.finish()]);
        // Nothing to destroy: the buffers are the scratch and outlive this call. That is the
        // point - `run` has to destroy its own because it sizes them per texture, and doing
        // that here was over a hundred GPU object lifetimes a frame.
        Some(texture)
    }

    /// Run the ETC2 encoder over one RGBA8 image and hand the blocks BACK, packed with no row
    /// padding - exactly the layout `etcenc::encode_etc2_rgb8`/`_rgba8` produce.
    ///
    /// # This is the only place the GPU path reads back, and it is deliberately not the fast one
    /// The shipped path never maps a buffer: the blocks go straight into a texture and the CPU
    /// never sees them, which is what makes the whole thing free of stalls. But an encoder whose
    /// output nothing can look at is an encoder nobody can check, and a WGSL port of a 600-line
    /// integer search is not something to take on faith. So this exists for
    /// `gpu_etc2_matches_the_cpu_encoder`, which runs both encoders over the same corpus the CPU
    /// one's own error ceilings were written from and compares them block for block.
    ///
    /// Not `#[cfg(test)]`: the test lives in another crate (it needs a real adapter), and a
    /// verification path that only compiles under `cfg(test)` in the crate under test is one the
    /// shipped build never type-checks.
    pub fn encode_rgba8_readback(
        &self,
        device: &wgpu::Device,
        queue: &wgpu::Queue,
        width: u32,
        height: u32,
        rgba: &[u8],
        alpha: bool,
    ) -> Vec<u8> {
        let flags = u32::from(alpha) * FLAG_ALPHA;
        let bb: u32 = if alpha { 16 } else { 8 };
        let blocks_x = width.div_ceil(4);
        let blocks_y = height.div_ceil(4);
        let row_bytes = align_up(blocks_x * bb, COPY_ROW_ALIGN);
        let out_size = (row_bytes * blocks_y) as u64;

        let rgba_buf = wgpu::util::DeviceExt::create_buffer_init(
            device,
            &wgpu::util::BufferInitDescriptor {
                label: Some("texenc-test-rgba"),
                contents: rgba,
                usage: wgpu::BufferUsages::STORAGE,
            },
        );
        let src_buf = device.create_buffer(&wgpu::BufferDescriptor {
            label: Some("texenc-test-src"),
            size: 4,
            usage: wgpu::BufferUsages::STORAGE,
            mapped_at_creation: false,
        });
        let out_buf = device.create_buffer(&wgpu::BufferDescriptor {
            label: Some("texenc-test-out"),
            size: out_size,
            usage: wgpu::BufferUsages::STORAGE | wgpu::BufferUsages::COPY_SRC,
            mapped_at_creation: false,
        });
        let read_buf = device.create_buffer(&wgpu::BufferDescriptor {
            label: Some("texenc-test-read"),
            size: out_size,
            usage: wgpu::BufferUsages::COPY_DST | wgpu::BufferUsages::MAP_READ,
            mapped_at_creation: false,
        });
        let mut slot = Params {
            width,
            height,
            out_row_words: row_bytes / 4,
            flags,
            ..Default::default()
        }
        .to_bytes()
        .to_vec();
        slot.resize(UNIFORM_SLOT as usize, 0);
        let params_buf = wgpu::util::DeviceExt::create_buffer_init(
            device,
            &wgpu::util::BufferInitDescriptor {
                label: Some("texenc-test-params"),
                contents: &slot,
                usage: wgpu::BufferUsages::UNIFORM,
            },
        );
        let bg = device.create_bind_group(&wgpu::BindGroupDescriptor {
            label: Some("texenc-test-bg"),
            layout: &self.layout,
            entries: &[
                wgpu::BindGroupEntry {
                    binding: 0,
                    resource: wgpu::BindingResource::Buffer(wgpu::BufferBinding {
                        buffer: &params_buf,
                        offset: 0,
                        size: wgpu::BufferSize::new(Params::BYTES as u64),
                    }),
                },
                wgpu::BindGroupEntry { binding: 1, resource: src_buf.as_entire_binding() },
                wgpu::BindGroupEntry { binding: 2, resource: rgba_buf.as_entire_binding() },
                wgpu::BindGroupEntry { binding: 3, resource: out_buf.as_entire_binding() },
            ],
        });
        let mut enc = device.create_command_encoder(&Default::default());
        {
            let mut pass = enc.begin_compute_pass(&Default::default());
            pass.set_pipeline(&self.encode_etc2);
            pass.set_bind_group(0, &bg, &[0]);
            pass.dispatch_workgroups(blocks_x.div_ceil(8), blocks_y.div_ceil(8), 1);
        }
        enc.copy_buffer_to_buffer(&out_buf, 0, &read_buf, 0, out_size);
        queue.submit([enc.finish()]);

        let slice = read_buf.slice(..);
        slice.map_async(wgpu::MapMode::Read, |_| {});
        let _ = device.poll(wgpu::PollType::Wait { submission_index: None, timeout: None });
        let view = slice.get_mapped_range().expect("the poll above waited for the map");
        // Strip the row padding, so the caller compares against the CPU encoder's own packing
        // rather than against this function's copy convention.
        let mut out = Vec::with_capacity((blocks_x * blocks_y * bb) as usize);
        for by in 0..blocks_y {
            let start = (by * row_bytes) as usize;
            out.extend_from_slice(&view[start..start + (blocks_x * bb) as usize]);
        }
        drop(view);
        read_buf.unmap();
        out
    }

    /// Run the DECODE and MIP half of a plan and hand back every level's RGBA8, for
    /// `gpu_pvrtc_matches_the_cpu_decoder`.
    ///
    /// Same reasoning as [`Self::encode_rgba8_readback`]: the shipped path never maps a buffer,
    /// and an intermediate nothing can look at is an intermediate nobody can check. A PVRTC texel
    /// is not block-local - it reads the four blocks whose centres surround it, and the format
    /// wraps at the edges - so this is exactly the kind of addressing that goes plausibly wrong.
    pub fn decode_chain_readback(
        &self,
        device: &wgpu::Device,
        queue: &wgpu::Queue,
        plan: &GpuTranscode,
    ) -> Vec<(u32, u32, Vec<u8>)> {
        let levels = self
            .plan_levels(plan, BlockFormat::Etc2Rgb8)
            .expect("the caller built this plan from the same layout arithmetic");
        let rgba_words: u64 = levels.iter().map(|l| l.width as u64 * l.height as u64).sum();
        let size = rgba_words * 4;

        let mut src_bytes = plan.src.to_vec();
        while src_bytes.len() % 4 != 0 {
            src_bytes.push(0);
        }
        let src_buf = wgpu::util::DeviceExt::create_buffer_init(
            device,
            &wgpu::util::BufferInitDescriptor {
                label: Some("texenc-test-src"),
                contents: &src_bytes,
                usage: wgpu::BufferUsages::STORAGE,
            },
        );
        let rgba_buf = device.create_buffer(&wgpu::BufferDescriptor {
            label: Some("texenc-test-rgba"),
            size,
            usage: wgpu::BufferUsages::STORAGE | wgpu::BufferUsages::COPY_SRC,
            mapped_at_creation: false,
        });
        let out_buf = device.create_buffer(&wgpu::BufferDescriptor {
            label: Some("texenc-test-out"),
            size: 4,
            usage: wgpu::BufferUsages::STORAGE,
            mapped_at_creation: false,
        });
        let read_buf = device.create_buffer(&wgpu::BufferDescriptor {
            label: Some("texenc-test-read"),
            size,
            usage: wgpu::BufferUsages::COPY_DST | wgpu::BufferUsages::MAP_READ,
            mapped_at_creation: false,
        });
        let base_flags = codec_flags(plan.codec);
        let mut slots: Vec<u8> = Vec::new();
        for (i, l) in levels.iter().enumerate() {
            let p = match plan.src_levels.get(i) {
                Some(s) => Params {
                    width: l.width,
                    height: l.height,
                    blocks_x: s.blocks_x,
                    blocks_y: s.blocks_y,
                    padded_x: s.padded_x,
                    padded_y: s.padded_y,
                    src_word: s.byte_offset / 4,
                    rgba_word: l.rgba_word,
                    flags: base_flags | (u32::from(s.swizzled) * FLAG_SWIZZLED),
                    src_format: codec_format(plan.codec),
                    src_block_words: plan.codec.block_bytes() / 4,
                    ..Default::default()
                },
                None => {
                    let prev = &levels[i - 1];
                    Params {
                        width: l.width,
                        height: l.height,
                        rgba_word: l.rgba_word,
                        src_word: prev.rgba_word,
                        src_width: prev.width,
                        src_height: prev.height,
                        flags: base_flags,
                        ..Default::default()
                    }
                }
            };
            slots.extend_from_slice(&p.to_bytes());
            slots.resize(align_up(slots.len() as u32, UNIFORM_SLOT as u32) as usize, 0);
        }
        let params_buf = wgpu::util::DeviceExt::create_buffer_init(
            device,
            &wgpu::util::BufferInitDescriptor {
                label: Some("texenc-test-params"),
                contents: &slots,
                usage: wgpu::BufferUsages::UNIFORM,
            },
        );
        let bg = device.create_bind_group(&wgpu::BindGroupDescriptor {
            label: Some("texenc-test-bg"),
            layout: &self.layout,
            entries: &[
                wgpu::BindGroupEntry {
                    binding: 0,
                    resource: wgpu::BindingResource::Buffer(wgpu::BufferBinding {
                        buffer: &params_buf,
                        offset: 0,
                        size: wgpu::BufferSize::new(Params::BYTES as u64),
                    }),
                },
                wgpu::BindGroupEntry { binding: 1, resource: src_buf.as_entire_binding() },
                wgpu::BindGroupEntry { binding: 2, resource: rgba_buf.as_entire_binding() },
                wgpu::BindGroupEntry { binding: 3, resource: out_buf.as_entire_binding() },
            ],
        });
        let mut enc = device.create_command_encoder(&Default::default());
        {
            let mut pass = enc.begin_compute_pass(&Default::default());
            for (i, l) in levels.iter().enumerate() {
                let off = (i as u64) * UNIFORM_SLOT;
                match plan.src_levels.get(i) {
                    Some(s) => {
                        pass.set_pipeline(self.decoder(plan.codec));
                        pass.set_bind_group(0, &bg, &[off as u32]);
                        pass.dispatch_workgroups(s.blocks_x.div_ceil(8), s.blocks_y.div_ceil(8), 1);
                    }
                    None => {
                        pass.set_pipeline(&self.halve);
                        pass.set_bind_group(0, &bg, &[off as u32]);
                        pass.dispatch_workgroups(l.width.div_ceil(8), l.height.div_ceil(8), 1);
                    }
                }
            }
        }
        enc.copy_buffer_to_buffer(&rgba_buf, 0, &read_buf, 0, size);
        queue.submit([enc.finish()]);
        let slice = read_buf.slice(..);
        slice.map_async(wgpu::MapMode::Read, |_| {});
        let _ = device.poll(wgpu::PollType::Wait { submission_index: None, timeout: None });
        let view = slice.get_mapped_range().expect("the poll above waited for the map");
        let out = levels
            .iter()
            .map(|l| {
                let start = l.rgba_word as usize * 4;
                let n = (l.width * l.height * 4) as usize;
                (l.width, l.height, view[start..start + n].to_vec())
            })
            .collect();
        drop(view);
        read_buf.unmap();
        out
    }

    /// Record and submit the REAL per-level copies for a chain of `width x height` in `format`,
    /// into a texture this device actually accepts.
    ///
    /// # Why a self-test exists for the copy and not for the shaders
    /// The shaders are verified against the CPU by reading their output back, which needs no
    /// texture at all - that runs on any adapter. The COPY is different: it needs a texture in a
    /// compressed format, and the machine this was written on exposes only BC, so the ETC2 copy
    /// path was declared untestable here and shipped UNEXERCISED. It was then wrong in the most
    /// basic way available - the copy extent was each level's LOGICAL size instead of its
    /// physical one - and it reached the user's phone, where it invalidated the whole command
    /// buffer and left every transcoded texture created and never written.
    ///
    /// The mistake in that reasoning was treating the copy as ETC2-specific. **It is not.** Every
    /// rule it has to satisfy - extent a multiple of the block size, offset block-aligned,
    /// `bytes_per_row` a multiple of 256, the buffer long enough for the last row of the last
    /// level - is a property of the BLOCK GEOMETRY, and BC1 and `Etc2Rgb8` have the same 4x4
    /// blocks at the same 8 bytes per block. So the copy can be exercised on ANY adapter using
    /// whichever block format it does support, and the bug that shipped would have failed this on
    /// the machine that wrote it.
    ///
    /// Native wgpu's default uncaptured-error handler panics, so a validation failure surfaces as
    /// a failed test rather than as a silently invalid command buffer.
    pub fn copy_geometry_selftest(
        &self,
        device: &wgpu::Device,
        queue: &wgpu::Queue,
        width: u32,
        height: u32,
        format: BlockFormat,
        wgpu_format: wgpu::TextureFormat,
    ) {
        let plan = GpuTranscode {
            src: std::sync::Arc::from(&[][..]),
            codec: SourceCodec::Pvrtc { two: false, four_bpp: true },
            width,
            height,
            levels: 32 - width.max(height).max(1).leading_zeros(),
            src_levels: Vec::new(),
        };
        let levels = plan_levels(&plan, format).expect("a plain chain plans");
        let size = levels
            .last()
            .map(|l| l.out_byte as u64 + l.out_row_bytes as u64 * l.blocks_y as u64)
            .expect("a chain has levels");
        let out_buf = device.create_buffer(&wgpu::BufferDescriptor {
            label: Some("texenc-selftest-out"),
            size,
            usage: wgpu::BufferUsages::COPY_SRC | wgpu::BufferUsages::COPY_DST,
            mapped_at_creation: false,
        });
        let texture = device.create_texture(&wgpu::TextureDescriptor {
            label: Some("texenc-selftest-tex"),
            size: wgpu::Extent3d { width, height, depth_or_array_layers: 1 },
            mip_level_count: levels.len() as u32,
            sample_count: 1,
            dimension: wgpu::TextureDimension::D2,
            format: wgpu_format,
            usage: wgpu::TextureUsages::TEXTURE_BINDING | wgpu::TextureUsages::COPY_DST,
            view_formats: &[],
        });
        let mut enc = device.create_command_encoder(&Default::default());
        record_copies(&mut enc, &out_buf, &texture, &levels);
        queue.submit([enc.finish()]);
        let _ = device.poll(wgpu::PollType::Wait { submission_index: None, timeout: None });
        out_buf.destroy();
        texture.destroy();
    }

    fn plan_levels(&self, plan: &GpuTranscode, format: BlockFormat) -> Option<Vec<Level>> {
        plan_levels(plan, format)
    }

    /// The decode shader for a source codec.
    fn decoder(&self, codec: SourceCodec) -> &wgpu::ComputePipeline {
        match codec {
            SourceCodec::Pvrtc { .. } => &self.decode_pvrtc,
            SourceCodec::Bc { .. } => &self.decode_bc,
        }
    }
}

/// Record the buffer-to-texture copy for every level of a finished chain.
///
/// # >>> THE COPY EXTENT IS THE LEVEL'S PHYSICAL SIZE, NOT ITS LOGICAL ONE.
/// This is the whole content of this function and it shipped wrong. WebGPU validates a
/// compressed copy against the level's PHYSICAL extent - the logical size rounded UP to whole
/// blocks - and requires `copySize` to be a multiple of the block size. A 2x2 mip level of a
/// 4x4-block format is logically 2x2 and physically 4x4: it occupies one whole block, and the
/// copy has to say 4x4.
///
/// Passing the logical size instead was rejected with `copySize.height (2) is not a multiple of
/// compressed texture format block height (4)`, and the failure is not local: the offending copy
/// invalidates the whole COMMAND BUFFER, so the submit does nothing, so every texture the encoder
/// built was created and never written. On the device that read as "no textures are right" - not
/// as a bad encode, which is where an hour goes if the error message is not in front of you.
///
/// The block-aligned extent is also exactly what the output buffer was sized for: `plan_levels`
/// allocates `blocks_x * blocks_y` blocks a level, and `blocks_x` is already `div_ceil(width, 4)`.
/// So the bytes were always there; only the copy's description of them was wrong.
fn record_copies(
    enc: &mut wgpu::CommandEncoder,
    out_buf: &wgpu::Buffer,
    texture: &wgpu::Texture,
    levels: &[Level],
) {
    for (i, l) in levels.iter().enumerate() {
        enc.copy_buffer_to_texture(
            wgpu::TexelCopyBufferInfo {
                buffer: out_buf,
                layout: wgpu::TexelCopyBufferLayout {
                    offset: l.out_byte as u64,
                    bytes_per_row: Some(l.out_row_bytes),
                    rows_per_image: Some(l.blocks_y),
                },
            },
            wgpu::TexelCopyTextureInfo {
                texture,
                mip_level: i as u32,
                origin: wgpu::Origin3d::ZERO,
                aspect: wgpu::TextureAspect::All,
            },
            wgpu::Extent3d {
                width: l.blocks_x * 4,
                height: l.blocks_y * 4,
                depth_or_array_layers: 1,
            },
        );
    }
}

/// The shader flag bits a source codec sets.
fn codec_flags(codec: SourceCodec) -> u32 {
    match codec {
        SourceCodec::Pvrtc { two, four_bpp } => {
            (u32::from(two) * FLAG_PVRTC2) | (u32::from(four_bpp) * FLAG_4BPP)
        }
        SourceCodec::Bc { .. } => 0,
    }
}

/// The guest base format the BC decoder branches on. Zero for PVRTC, which branches on flags.
fn codec_format(codec: SourceCodec) -> u32 {
    match codec {
        SourceCodec::Pvrtc { .. } => 0,
        SourceCodec::Bc { base_format } => base_format,
    }
}

/// Where every level of the finished chain lives in the two scratch buffers.
///
/// A free function, and deliberately: it decides the offsets and row pitches that
/// `copyBufferToTexture` validates, and those rules are arithmetic rather than anything about a
/// device. This machine's adapter exposes only BC, so the copy itself cannot be exercised here at
/// all - which makes a device-free test of the geometry the difference between "checked" and
/// "we will find out on the phone". See `the_copy_geometry_is_legal`.
fn plan_levels(plan: &GpuTranscode, format: BlockFormat) -> Option<Vec<Level>> {
    {
        let bb = format.block_bytes();
        let mut out = Vec::with_capacity(plan.levels as usize);
        let mut rgba_word = 0u32;
        let mut out_byte = 0u32;
        for l in 0..plan.levels {
            let width = (plan.width >> l).max(1);
            let height = (plan.height >> l).max(1);
            let blocks_x = width.div_ceil(4);
            let blocks_y = height.div_ceil(4);
            let out_row_bytes = align_up(blocks_x * bb, COPY_ROW_ALIGN);
            out.push(Level { width, height, rgba_word, out_byte, out_row_bytes, blocks_x, blocks_y });
            rgba_word = rgba_word.checked_add(width.checked_mul(height)?)?;
            out_byte = out_byte.checked_add(out_row_bytes.checked_mul(blocks_y)?)?;
        }
        // A guest level the plan claims but does not describe would leave a level of the chain
        // holding whatever the scratch buffer had in it. The runtime builds `src_levels` from the
        // same `level_layout` the CPU path uses, so this is a statement of that rather than a
        // condition anything is expected to hit.
        if plan.src_levels.len() > out.len() {
            return None;
        }
        Some(out)
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::gpu::{SourceCodec, SrcLevel};

    fn plan(width: u32, height: u32) -> GpuTranscode {
        GpuTranscode {
            src: std::sync::Arc::from(&[][..]),
            codec: SourceCodec::Pvrtc { two: false, four_bpp: true },
            width,
            height,
            levels: 32 - width.max(height).max(1).leading_zeros(),
            src_levels: vec![SrcLevel {
                byte_offset: 0,
                width,
                height,
                blocks_x: width.div_ceil(4),
                blocks_y: height.div_ceil(4),
                padded_x: width.div_ceil(4),
                padded_y: height.div_ceil(4),
                swizzled: false,
            }],
        }
    }

    /// Every level's copy must satisfy what `copyBufferToTexture` requires, at every size the
    /// chain passes through.
    ///
    /// # Why this is asserted rather than left to the device
    /// The rules are unforgiving and they all bite at the SMALL end of a mip chain, which is
    /// exactly where nobody looks: `bytes_per_row` must be a multiple of 256 and a packed block
    /// row of a 16-texel-wide level is 32 bytes; the source offset must be block-aligned; and the
    /// buffer has to be long enough for the last row of the last level. Breaking any of them is a
    /// validation error, and in the browser a validation error on a texture upload surfaces as a
    /// black draw thousands of messages later.
    ///
    /// It matters more than usual that this is checked without a GPU: the adapter on the machine
    /// this was written on exposes only BC, so the ETC2 copy path cannot run here at all. Without
    /// this the first execution of that arithmetic would be on the user's phone.
    #[test]
    fn the_copy_geometry_is_legal() {
        for (w, h) in [(2048u32, 2048u32), (1024, 512), (64, 64), (24, 24), (8, 4), (4, 4)] {
            for format in [BlockFormat::Etc2Rgb8, BlockFormat::Etc2Rgba8] {
                let p = plan(w, h);
                let levels = plan_levels(&p, format).expect("a plain chain plans");
                assert_eq!(levels.len(), p.levels as usize, "{w}x{h}: level count");
                let bb = format.block_bytes();
                let mut prev_end = 0u32;
                for (i, l) in levels.iter().enumerate() {
                    assert_eq!(l.out_byte % COPY_ROW_ALIGN, 0, "{w}x{h} level {i}: copy offset");
                    assert_eq!(l.out_row_bytes % COPY_ROW_ALIGN, 0, "{w}x{h} level {i}: row pitch");
                    assert!(
                        l.out_row_bytes >= l.blocks_x * bb,
                        "{w}x{h} level {i}: row pitch is shorter than the blocks it carries"
                    );
                    assert!(l.out_byte >= prev_end, "{w}x{h} level {i}: levels overlap");
                    prev_end = l.out_byte + l.out_row_bytes * l.blocks_y;
                    // The level's own dimensions, which is what the copy extent is set to. A
                    // compressed copy may be smaller than a block only when it IS the level.
                    assert_eq!(l.width, (w >> i).max(1), "{w}x{h} level {i}: width");
                    assert_eq!(l.height, (h >> i).max(1), "{w}x{h} level {i}: height");
                    assert_eq!(l.blocks_x, l.width.div_ceil(4));
                    assert_eq!(l.blocks_y, l.height.div_ceil(4));
                }
                // The RGBA scratch offsets must tile the buffer exactly, or one level reads
                // another's texels - which is a picture, just not this texture's.
                let mut word = 0u32;
                for l in &levels {
                    assert_eq!(l.rgba_word, word, "{w}x{h}: rgba offsets are not contiguous");
                    word += l.width * l.height;
                }
            }
        }
    }

    /// The chain always reaches 1x1, whatever the aspect ratio - the same rule
    /// `max_mip_levels` states, restated here because the uploader declares
    /// `mip_level_count` from this list and a short chain is a texture whose smallest levels
    /// sample whatever the driver left there.
    #[test]
    fn the_chain_reaches_one_by_one() {
        for (w, h) in [(2048u32, 1u32), (1, 512), (24, 24), (4096, 2048)] {
            let levels = plan_levels(&plan(w, h), BlockFormat::Etc2Rgb8).unwrap();
            let last = levels.last().unwrap();
            assert_eq!((last.width, last.height), (1, 1), "{w}x{h} stopped at {last:?}");
        }
    }
}

fn storage_entry(binding: u32, read_only: bool) -> wgpu::BindGroupLayoutEntry {
    wgpu::BindGroupLayoutEntry {
        binding,
        visibility: wgpu::ShaderStages::COMPUTE,
        ty: wgpu::BindingType::Buffer {
            ty: wgpu::BufferBindingType::Storage { read_only },
            has_dynamic_offset: false,
            min_binding_size: None,
        },
        count: None,
    }
}
