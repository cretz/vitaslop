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

/// The compute pipelines, built once per device.
pub struct Transcoder {
    layout: wgpu::BindGroupLayout,
    decode_pvrtc: wgpu::ComputePipeline,
    decode_bc: wgpu::ComputePipeline,
    halve: wgpu::ComputePipeline,
    encode_etc2: wgpu::ComputePipeline,
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
        Self {
            decode_pvrtc: make("decode_pvrtc"),
            decode_bc: make("decode_bc"),
            halve: make("halve"),
            encode_etc2: make("encode_etc2"),
            layout,
        }
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
        let src_buf = wgpu::util::DeviceExt::create_buffer_init(
            device,
            &wgpu::util::BufferInitDescriptor {
                label: Some("texenc-src"),
                contents: &src_bytes,
                usage: wgpu::BufferUsages::STORAGE,
            },
        );
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
        let params_buf = wgpu::util::DeviceExt::create_buffer_init(
            device,
            &wgpu::util::BufferInitDescriptor {
                label: Some("texenc-params"),
                contents: &slots,
                usage: wgpu::BufferUsages::UNIFORM,
            },
        );
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
