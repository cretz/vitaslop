use super::*;

impl GxmRenderer {
    /// The next timestamp pair for a pass being begun, if this frame is measured.
    pub(super) fn ts_pair(
        &mut self,
        label: impl FnOnce() -> String,
    ) -> Option<(std::sync::Arc<wgpu::QuerySet>, u32)> {
        self.ts.as_mut().and_then(|t| t.take(label))
    }

    /// A pass descriptor's `timestamp_writes` for a pair from [`Self::ts_pair`].
    pub(super) fn ts_writes(
        pair: &Option<(std::sync::Arc<wgpu::QuerySet>, u32)>,
    ) -> Option<wgpu::RenderPassTimestampWrites<'_>> {
        pair.as_ref().map(|(set, i)| wgpu::RenderPassTimestampWrites {
            query_set: set,
            beginning_of_pass_write_index: Some(i * 2),
            end_of_pass_write_index: Some(i * 2 + 1),
        })
    }

    /// After [`Self::encode_chain`] and before the encoder is finished: resolve this
    /// frame's timestamps onto the same submit. See [`GpuTimestamps`].
    pub fn ts_finish_chain(&mut self, encoder: &mut wgpu::CommandEncoder) {
        if let Some(ts) = self.ts.as_mut() {
            ts.finish_chain(encoder);
        }
        if let Some(c) = self.coverage.as_mut() {
            c.finish_chain(encoder);
        }
    }

    /// After the submit that carried [`Self::ts_finish_chain`]'s copy: report every draw's
    /// occlusion count. Does nothing unless `VITASLOP_GXM_DRAW_COVERAGE` is set, and BLOCKS
    /// on the GPU when it is - see [`DrawCoverage`].
    pub fn coverage_report_blocking(&mut self, device: &wgpu::Device) {
        if let Some(c) = self.coverage.as_mut() {
            c.report_blocking(device);
        }
    }

    /// After the submit that carried [`Self::ts_finish_chain`]'s copy: ask for the map.
    pub fn ts_map_after_submit(&mut self) {
        if let Some(ts) = self.ts.as_mut() {
            ts.map_after_submit();
        }
    }

    /// Before a chain: collect a readback that has completed since the last present.
    pub fn ts_poll(&mut self) {
        if let Some(ts) = self.ts.as_mut() {
            ts.poll();
        }
    }

    /// The newest measured frame's GPU ms (summed over its passes) and the run's count of
    /// measured frames, or `None` on an adapter without `timestamp-query`. The count moving
    /// is what says the figure is fresh: under a deep queue the readback itself stalls, and
    /// a reader must be able to see that no new measurement has arrived.
    pub fn ts_latest(&self) -> Option<(f64, u64)> {
        self.ts.as_ref().map(|ts| (ts.latest_ms, ts.latest_seq))
    }

    /// Whether a timestamp readback is IN FLIGHT - encoded into a submitted frame and not yet
    /// collected. A measurement can only go stale while one is: with none in flight there is
    /// nothing to wait for, and the next present asks again.
    pub fn ts_pending(&self) -> bool {
        self.ts.as_ref().is_some_and(|ts| ts.in_flight.is_some())
    }

    /// >>> A TEST RIG: `iterations` of arithmetic per invocation over 64K invocations, in a
    /// >>> compute pass that takes a timestamp pair like any render pass.
    ///
    /// The phone's frames cost it 27-32 ms of GPU against a 33-38 ms period, and whether the
    /// page keeps its queue bounded when the GPU cannot keep up is a question the desktop
    /// cannot ask while its GPU sits 8% busy. `VITASLOP_GPU_BURN=<n>` makes it ask: the
    /// work is real GPU time on the device's own clock, timed and summed with the frame's
    /// passes exactly as the phone's fragment cost is. Encoded after the chain, so it adds
    /// load without changing anything a pass draws.
    pub fn gpu_burn(&mut self, device: &wgpu::Device, encoder: &mut wgpu::CommandEncoder, iterations: u32) {
        if iterations == 0 {
            return;
        }
        if self.burn.as_ref().is_none_or(|b| b.0 != iterations) {
            let src = format!(
                "@group(0) @binding(0) var<storage, read_write> sink: array<f32>;
                     @compute @workgroup_size(64) fn main(@builtin(global_invocation_id) id: vec3<u32>) {{
                       var x = f32(id.x) * 0.001;
                       for (var i = 0u; i < {iterations}u; i = i + 1u) {{ x = fract(sin(x) * 43758.5453 + 0.1); }}
                       sink[id.x & 1023u] = x;
                     }}
"
            );
            let module = device.create_shader_module(wgpu::ShaderModuleDescriptor {
                label: Some("gpu-burn"),
                source: wgpu::ShaderSource::Wgsl(src.into()),
            });
            let pipeline = device.create_compute_pipeline(&wgpu::ComputePipelineDescriptor {
                label: Some("gpu-burn"),
                layout: None,
                module: &module,
                entry_point: Some("main"),
                compilation_options: Default::default(),
                cache: None,
            });
            let sink = device.create_buffer(&wgpu::BufferDescriptor {
                label: Some("gpu-burn-sink"),
                size: 4096,
                usage: wgpu::BufferUsages::STORAGE,
                mapped_at_creation: false,
            });
            let bg = device.create_bind_group(&wgpu::BindGroupDescriptor {
                label: Some("gpu-burn"),
                layout: &pipeline.get_bind_group_layout(0),
                entries: &[wgpu::BindGroupEntry { binding: 0, resource: sink.as_entire_binding() }],
            });
            self.burn = Some((iterations, pipeline, bg));
        }
        let pair = self.ts_pair(|| format!("gpu-burn x{iterations}"));
        let Some((_, pipeline, bg)) = self.burn.as_ref() else { return };
        let mut pass = encoder.begin_compute_pass(&wgpu::ComputePassDescriptor {
            label: Some("gpu-burn"),
            timestamp_writes: pair.as_ref().map(|(set, i)| wgpu::ComputePassTimestampWrites {
                query_set: set,
                beginning_of_pass_write_index: Some(i * 2),
                end_of_pass_write_index: Some(i * 2 + 1),
            }),
        });
        pass.set_pipeline(pipeline);
        pass.set_bind_group(0, bg, &[]);
        pass.dispatch_workgroups(1024, 1, 1);
    }

    /// The mean GPU ms per frame of the window the last [`Self::take_gpu_time_report`]
    /// described, or 0 where nothing was measured. Read it AFTER that call.
    pub fn last_gpu_ms_per_frame(&self) -> f64 {
        self.ts.as_ref().map_or(0.0, |ts| ts.last_mean_ms)
    }

    /// The window's GPU-time report, or why there is none. See [`GpuTimestamps`].
    pub fn take_gpu_time_report(&mut self) -> String {
        match self.ts.as_mut() {
            None => "not measurable: this adapter offers no `timestamp-query`, so only the \
                     work-done latency above is available, and that includes the event loop"
                .to_string(),
            Some(ts) => ts.take_report().unwrap_or_else(|| {
                "no frame was measured this window (a readback is collected on a later \
                 present; a window with no presents measures nothing)"
                    .to_string()
            }),
        }
    }

    /// Build both pipelines for a `color_format` render target. `queue` is used
    /// once here to upload the 1x1 white fallback texel.
    pub fn new(
        device: &wgpu::Device,
        queue: &wgpu::Queue,
        color_format: wgpu::TextureFormat,
    ) -> Self {
        // Whether a destination-reading fragment program may be lowered to a DUAL-SOURCE
        // blend on THIS device - pushed into the recompiler before any program is
        // classified, because `fragment_reads_dest` caches its answer per blob and the
        // pass-split decision is made from that cache. `VITASLOP_GXP_DUAL_SOURCE=0` is
        // the arm back to the split for every such program.
        {
            let have = device.features().contains(wgpu::Features::DUAL_SOURCE_BLENDING);
            let want = crate::knobs::var("VITASLOP_GXP_DUAL_SOURCE")
                .map(|v| v.trim() != "0")
                .unwrap_or(true);
            vitaslop_gxp_shader::module::set_dual_source_blend(have && want);
            // >>> AND WHICH WAY THIS DEVICE'S f16 STORES ROUND, for the same reason and in
            // the same place: the emitted BODY calls the store helpers and the module
            // PREAMBLE defines them, so the decision has to be made before any program is
            // recompiled. See `vitaslop_gxp_shader::wgsl::HALF_LO_FN`.
            // `VITASLOP_GXP_NATIVE_F16=0` takes the portable narrowing on a device that has
            // the feature - the A/B for a device whose own `f16` is suspect.
            let f16 = device.features().contains(wgpu::Features::SHADER_F16)
                && crate::knobs::var("VITASLOP_GXP_NATIVE_F16").map(|v| v.trim() != "0").unwrap_or(true);
            vitaslop_gxp_shader::wgsl::set_native_f16(f16);
            report_status!(
                "gxp f16 stores: round-to-nearest-even {}, through the {} narrowing. The \
                 guest's hardware and the CPU reference both round this way; \
                 `VITASLOP_GXP_F16_RTE=0` is the arm back to the device's own \
                 implementation-defined mode (this adapter TRUNCATES), which is what every \
                 build before 2026-09-21c shipped.",
                if vitaslop_gxp_shader::wgsl::f16_round_to_nearest() { "ON" } else { "OFF (knob)" },
                if f16 { "native `f16()`" } else { "portable bit-arithmetic" },
            );
            report_status!(
                "gxp dual-source blend: {} - a fragment program that blends itself over the \
                 destination and is linear in it {}",
                if have && want { "ON" } else if have { "OFF (knob)" } else { "OFF (no feature)" },
                if have && want {
                    "is lowered to a two-output shader and a pipeline blend, with no pass split"
                } else {
                    "costs a render-pass split and an attachment copy"
                }
            );
        }
        let shader = device.create_shader_module(wgpu::ShaderModuleDescriptor {
            label: Some("gxm"),
            source: wgpu::ShaderSource::Wgsl(GXM_SHADER.into()),
        });

        // Group 0: the per-draw uniform, addressed by dynamic offset into one shared
        // buffer (so a whole frame needs a single uniform bind group). `min_binding_size`
        // pins the shader-visible window to the `U` struct so validation is tight.
        let uniform_layout = device.create_bind_group_layout(&wgpu::BindGroupLayoutDescriptor {
            label: Some("gxm-uniform"),
            entries: &[wgpu::BindGroupLayoutEntry {
                binding: 0,
                visibility: wgpu::ShaderStages::VERTEX_FRAGMENT,
                ty: wgpu::BindingType::Buffer {
                    ty: wgpu::BufferBindingType::Uniform,
                    has_dynamic_offset: true,
                    min_binding_size: wgpu::BufferSize::new(UNIFORM_BYTES),
                },
                count: None,
            }],
        });

        // Group 1: the sampled texture + its sampler. `filterable` / `Filtering` so a
        // LINEAR-magnified texture (UI/font atlas) can bilinear-sample on the GPU; a
        // Nearest sampler is still valid in a filtering slot, so the same layout serves
        // both filter modes (the sampler is chosen per draw, see `bind_for`).
        let texture_layout = device.create_bind_group_layout(&wgpu::BindGroupLayoutDescriptor {
            label: Some("gxm-texture"),
            entries: &[
                wgpu::BindGroupLayoutEntry {
                    binding: 0,
                    visibility: wgpu::ShaderStages::FRAGMENT,
                    ty: wgpu::BindingType::Texture {
                        sample_type: wgpu::TextureSampleType::Float { filterable: true },
                        view_dimension: wgpu::TextureViewDimension::D2,
                        multisampled: false,
                    },
                    count: None,
                },
                wgpu::BindGroupLayoutEntry {
                    binding: 1,
                    visibility: wgpu::ShaderStages::FRAGMENT,
                    ty: wgpu::BindingType::Sampler(wgpu::SamplerBindingType::Filtering),
                    count: None,
                },
            ],
        });

        let pipeline_layout = device.create_pipeline_layout(&wgpu::PipelineLayoutDescriptor {
            label: Some("gxm-layout"),
            bind_group_layouts: &[Some(&uniform_layout), Some(&texture_layout)],
            immediate_size: 0,
        });

        // The fixed-function pipeline pair is built by [`build_fixed_pipeline`], which the
        // STENCIL variants (see `ensure_fixed_stencil_pipeline`) reuse with the same shader
        // and layout - so the default pair and a masked draw's pipeline cannot drift.
        let make = |opaque: bool, target_format: wgpu::TextureFormat, samples: u32| {
            build_fixed_pipeline(
                device,
                &shader,
                &pipeline_layout,
                opaque,
                target_format,
                samples,
                wgpu::StencilState::default(),
            )
        };

        let opaque = make(true, color_format, 1);
        let blend = make(false, color_format, 1);
        // The same two pipelines for a MULTISAMPLED pass. A fixed-function draw and a
        // recompiled one share one render pass, so when the guest's render target asks for
        // multisampling BOTH have to be built for it - a pipeline may only be bound in a
        // pass whose sample count matches. Built eagerly for the same reason the sRGB pair
        // below is: it is two pipelines, not a family (`gxm_sample_count` yields 1 or
        // MSAA_SAMPLES and nothing else), and the alternative is discovering mid-frame that
        // a fixed-function draw landed on a multisampled target with nothing to draw it
        // with.
        let opaque_ms = make(true, color_format, MSAA_SAMPLES);
        let blend_ms = make(false, color_format, MSAA_SAMPLES);
        // The same two pipelines against the sRGB view of the same texture, for a pass
        // whose colour surface the guest put in GAMMA-CORRECT mode. Built eagerly because
        // it is two pipelines, not a family: the alternative is discovering mid-frame that
        // a fixed-function draw landed on a gamma surface and having nothing to draw it
        // with. No multisampled variant: a gamma target is refused multisampling (see
        // `report_multisample_refused`), so the pair would never be bound.
        let srgb = srgb_twin(color_format)
            .map(|f| (make(true, f, 1), make(false, f, 1)));
        let float = (make(true, FLOAT_TARGET_FORMAT, 1), make(false, FLOAT_TARGET_FORMAT, 1));
        let float_ms = (make(true, FLOAT_TARGET_FORMAT, MSAA_SAMPLES), make(false, FLOAT_TARGET_FORMAT, MSAA_SAMPLES));

        let sampler = |linear: bool| {
            let f = if linear {
                wgpu::FilterMode::Linear
            } else {
                wgpu::FilterMode::Nearest
            };
            device.create_sampler(&wgpu::SamplerDescriptor {
                label: Some("gxm-sampler"),
                address_mode_u: wgpu::AddressMode::Repeat,
                address_mode_v: wgpu::AddressMode::Repeat,
                address_mode_w: wgpu::AddressMode::Repeat,
                mag_filter: f,
                min_filter: f,
                mipmap_filter: wgpu::MipmapFilterMode::Nearest,
                ..Default::default()
            })
        };
        let sampler_point = sampler(false);
        let sampler_linear = sampler(true);

        // A 1x1 opaque-white texel: the fallback an untextured draw binds so the one
        // shader path serves both textured and vertex-color draws.
        let white_tex = device.create_texture_with_data(
            queue,
            &wgpu::TextureDescriptor {
                label: Some("gxm-white"),
                size: wgpu::Extent3d { width: 1, height: 1, depth_or_array_layers: 1 },
                mip_level_count: 1,
                sample_count: 1,
                dimension: wgpu::TextureDimension::D2,
                format: wgpu::TextureFormat::Rgba8Unorm,
                usage: wgpu::TextureUsages::TEXTURE_BINDING,
                view_formats: &[],
            },
            wgpu::util::TextureDataOrder::LayerMajor,
            &[255, 255, 255, 255],
        );
        let white = white_tex.create_view(&Default::default());
        let white_bind = device.create_bind_group(&wgpu::BindGroupDescriptor {
            label: Some("gxm-white-bind"),
            layout: &texture_layout,
            entries: &[
                wgpu::BindGroupEntry {
                    binding: 0,
                    resource: wgpu::BindingResource::TextureView(&white),
                },
                wgpu::BindGroupEntry {
                    binding: 1,
                    resource: wgpu::BindingResource::Sampler(&sampler_point),
                },
            ],
        });

        let uniform_align = device.limits().min_uniform_buffer_offset_alignment as u64;
        let uniform_stride = align_up(UNIFORM_BYTES, uniform_align.max(1));

        // The supersample resolve: a fullscreen triangle that box-averages each
        // `scale x scale` block of the offscreen colour target into one output pixel, the
        // exact integer box `Framebuffer::downsampled` applies on the software oracle so the
        // two AA'd frames stay in lockstep. `textureLoad` (no sampler) does the exact taps.
        let resolve_shader = device.create_shader_module(wgpu::ShaderModuleDescriptor {
            label: Some("gxm-resolve"),
            source: wgpu::ShaderSource::Wgsl(RESOLVE_SHADER.into()),
        });
        let resolve_layout = device.create_bind_group_layout(&wgpu::BindGroupLayoutDescriptor {
            label: Some("gxm-resolve-layout"),
            entries: &[
                wgpu::BindGroupLayoutEntry {
                    binding: 0,
                    visibility: wgpu::ShaderStages::FRAGMENT,
                    ty: wgpu::BindingType::Texture {
                        sample_type: wgpu::TextureSampleType::Float { filterable: false },
                        view_dimension: wgpu::TextureViewDimension::D2,
                        multisampled: false,
                    },
                    count: None,
                },
                wgpu::BindGroupLayoutEntry {
                    binding: 1,
                    visibility: wgpu::ShaderStages::FRAGMENT,
                    ty: wgpu::BindingType::Buffer {
                        ty: wgpu::BufferBindingType::Uniform,
                        has_dynamic_offset: false,
                        min_binding_size: wgpu::BufferSize::new(16),
                    },
                    count: None,
                },
            ],
        });
        let resolve_pl = device.create_pipeline_layout(&wgpu::PipelineLayoutDescriptor {
            label: Some("gxm-resolve-pl"),
            bind_group_layouts: &[Some(&resolve_layout)],
            immediate_size: 0,
        });
        let resolve_pipeline = device.create_render_pipeline(&wgpu::RenderPipelineDescriptor {
            label: Some("gxm-resolve-pipe"),
            layout: Some(&resolve_pl),
            vertex: wgpu::VertexState {
                module: &resolve_shader,
                entry_point: Some("vres"),
                buffers: &[],
                compilation_options: Default::default(),
            },
            primitive: wgpu::PrimitiveState::default(),
            depth_stencil: None,
            multisample: wgpu::MultisampleState::default(),
            fragment: Some(wgpu::FragmentState {
                module: &resolve_shader,
                entry_point: Some("fres"),
                targets: &[Some(wgpu::ColorTargetState {
                    format: color_format,
                    blend: None,
                    write_mask: wgpu::ColorWrites::ALL,
                })],
                compilation_options: Default::default(),
            }),
            multiview_mask: None,
            cache: None,
        });
        let resolve_scale_buf = device.create_buffer(&wgpu::BufferDescriptor {
            label: Some("gxm-resolve-scale"),
            size: 16,
            usage: wgpu::BufferUsages::UNIFORM | wgpu::BufferUsages::COPY_DST,
            mapped_at_creation: false,
        });
        // The guest-depth conversion pass. Built unconditionally (it is one small
        // pipeline) but only ever ENCODED for a target whose depth some pass samples.
        let gxm_depth_shader = device.create_shader_module(wgpu::ShaderModuleDescriptor {
            label: Some("gxm-depth-convert"),
            source: wgpu::ShaderSource::Wgsl(GXM_DEPTH_SHADER.into()),
        });
        let gxm_depth_layout = device.create_bind_group_layout(&wgpu::BindGroupLayoutDescriptor {
            label: Some("gxm-depth-convert-layout"),
            entries: &[
                wgpu::BindGroupLayoutEntry {
                    binding: 0,
                    visibility: wgpu::ShaderStages::FRAGMENT,
                    ty: wgpu::BindingType::Texture {
                        sample_type: wgpu::TextureSampleType::Depth,
                        view_dimension: wgpu::TextureViewDimension::D2,
                        multisampled: false,
                    },
                    count: None,
                },
                wgpu::BindGroupLayoutEntry {
                    binding: 1,
                    visibility: wgpu::ShaderStages::FRAGMENT,
                    ty: wgpu::BindingType::Buffer {
                        ty: wgpu::BufferBindingType::Uniform,
                        has_dynamic_offset: false,
                        min_binding_size: wgpu::BufferSize::new(GXM_DEPTH_UNIFORM_BYTES),
                    },
                    count: None,
                },
                wgpu::BindGroupLayoutEntry {
                    binding: 2,
                    visibility: wgpu::ShaderStages::FRAGMENT,
                    // NON-filtering: a depth buffer is data, and this pass must return the
                    // stored texel rather than a blend of its neighbours.
                    ty: wgpu::BindingType::Sampler(wgpu::SamplerBindingType::NonFiltering),
                    count: None,
                },
            ],
        });
        let gxm_depth_pl = device.create_pipeline_layout(&wgpu::PipelineLayoutDescriptor {
            label: Some("gxm-depth-convert-pl"),
            bind_group_layouts: &[Some(&gxm_depth_layout)],
            immediate_size: 0,
        });
        let gxm_depth_pipe = device.create_render_pipeline(&wgpu::RenderPipelineDescriptor {
            label: Some("gxm-depth-convert-pipe"),
            layout: Some(&gxm_depth_pl),
            vertex: wgpu::VertexState {
                module: &gxm_depth_shader,
                entry_point: Some("vdep"),
                buffers: &[],
                compilation_options: Default::default(),
            },
            primitive: wgpu::PrimitiveState::default(),
            depth_stencil: None,
            multisample: wgpu::MultisampleState::default(),
            fragment: Some(wgpu::FragmentState {
                module: &gxm_depth_shader,
                entry_point: Some("fdep"),
                targets: &[Some(wgpu::ColorTargetState {
                    format: GXM_DEPTH_FORMAT,
                    blend: None,
                    write_mask: wgpu::ColorWrites::ALL,
                })],
                compilation_options: Default::default(),
            }),
            multiview_mask: None,
            cache: None,
        });
        // The multisampled twin: the same shader with its source declared multisampled and
        // read by `textureLoad` at sample 0, which compatibility mode refuses on depth.
        let gxm_depth_ms = (!crate::gpu::compat_mode()).then(|| {
            let src = GXM_DEPTH_SHADER
                .replace("var srcDepth: texture_depth_2d;", "var srcDepth: texture_depth_multisampled_2d;")
                .replace(
                    "textureSampleLevel(srcDepth, srcSamp, pos.xy / dims, 0)",
                    "textureLoad(srcDepth, vec2<i32>(pos.xy), 0)",
                );
            let module = device.create_shader_module(wgpu::ShaderModuleDescriptor {
                label: Some("gxm-depth-convert-ms"),
                source: wgpu::ShaderSource::Wgsl(src.into()),
            });
            let layout = device.create_bind_group_layout(&wgpu::BindGroupLayoutDescriptor {
                label: Some("gxm-depth-convert-ms-layout"),
                entries: &[
                    wgpu::BindGroupLayoutEntry {
                        binding: 0,
                        visibility: wgpu::ShaderStages::FRAGMENT,
                        ty: wgpu::BindingType::Texture {
                            sample_type: wgpu::TextureSampleType::Depth,
                            view_dimension: wgpu::TextureViewDimension::D2,
                            multisampled: true,
                        },
                        count: None,
                    },
                    wgpu::BindGroupLayoutEntry {
                        binding: 1,
                        visibility: wgpu::ShaderStages::FRAGMENT,
                        ty: wgpu::BindingType::Buffer {
                            ty: wgpu::BufferBindingType::Uniform,
                            has_dynamic_offset: false,
                            min_binding_size: wgpu::BufferSize::new(GXM_DEPTH_UNIFORM_BYTES),
                        },
                        count: None,
                    },
                ],
            });
            let pl = device.create_pipeline_layout(&wgpu::PipelineLayoutDescriptor {
                label: Some("gxm-depth-convert-ms-pl"),
                bind_group_layouts: &[Some(&layout)],
                immediate_size: 0,
            });
            let pipe = device.create_render_pipeline(&wgpu::RenderPipelineDescriptor {
                label: Some("gxm-depth-convert-ms-pipe"),
                layout: Some(&pl),
                vertex: wgpu::VertexState {
                    module: &module,
                    entry_point: Some("vdep"),
                    buffers: &[],
                    compilation_options: Default::default(),
                },
                primitive: wgpu::PrimitiveState::default(),
                depth_stencil: None,
                multisample: wgpu::MultisampleState::default(),
                fragment: Some(wgpu::FragmentState {
                    module: &module,
                    entry_point: Some("fdep"),
                    targets: &[Some(wgpu::ColorTargetState {
                        format: GXM_DEPTH_FORMAT,
                        blend: None,
                        write_mask: wgpu::ColorWrites::ALL,
                    })],
                    compilation_options: Default::default(),
                }),
                multiview_mask: None,
                cache: None,
            });
            (layout, pipe)
        });
        let gxm_depth_uniform = device.create_buffer(&wgpu::BufferDescriptor {
            label: Some("gxm-depth-convert-uniform"),
            size: GXM_DEPTH_UNIFORM_BYTES,
            usage: wgpu::BufferUsages::UNIFORM | wgpu::BufferUsages::COPY_DST,
            mapped_at_creation: false,
        });

        GxmRenderer {
            srgb,
            fixed_shader: shader,
            fixed_layout: pipeline_layout,
            fixed_stencil: HashMap::default(),
            gxm_depth_pipe,
            gxm_depth_layout,
            gxm_depth_uniform,
            gxm_depth_ms,
            opaque,
            blend,
            opaque_ms,
            blend_ms,
            float,
            float_ms,
            uniform_layout,
            texture_layout,
            sampler_point,
            sampler_linear,
            white_bind,
            views: HashMap::default(),
            views_bytes: 0,
            tex_binds: HashMap::default(),
            tex_pass_keys: HashMap::default(),
            vbo: None,
            ibo: None,
            ubo: None,
            ubo_bind: None,
            vbo_cap: 0,
            ibo_cap: 0,
            ubo_cap: 0,
            gxp_arenas: Vec::new(),
            // See the field's doc comment. Built here rather than lazily so the first
            // gameplay frame does not pay the pool's creation on top of everything else a
            // first frame already pays.
            gxp_staging: crate::gpu::gxm_staging_on().then(|| {
                wgpu::util::StagingBelt::new(device.clone(), crate::gpu::GXP_STAGING_CHUNK)
            }),
            pass_arenas: [const { Vec::new() }; 6],
            gxp_arena_ring: crate::gpu::gxp_arena_ring(),
            gxp_arena_phase: 0,
            gxp_arena_slot: 0,
            ts: GpuTimestamps::new(device, queue),
            burn: None,
            coverage: {
                let want = crate::knobs::var("VITASLOP_GXM_DRAW_COVERAGE")
                    .map(|v| v.trim() != "0")
                    .unwrap_or(false);
                // NATIVE ONLY, and it says so rather than encoding queries nobody reads.
                // The readback blocks on `Device::poll`, which the browser does not have -
                // there the frame's counts would have to come back two frames later through
                // the asynchronous machinery `GpuTimestamps` uses, and a diagnostic that is
                // silently inert on the engine the product ships on is worse than one that
                // refuses by name.
                if want && cfg!(target_arch = "wasm32") {
                    report_warn!(
                        "VITASLOP_GXM_DRAW_COVERAGE is set, but the per-draw occlusion readback blocks on Device::poll and is therefore NATIVE ONLY - this run measures nothing. Take the measurement on the desktop build."
                    );
                }
                (want && !cfg!(target_arch = "wasm32")).then(|| DrawCoverage::new(device))
            },
            uniform_stride,
            uniform_align,
            color_format,
            ss_scale: 1,
            gxp_summaries_seen: HashSet::default(),
            gxp_summaries_capped: false,
            gxp_summary_all_clear: false,
            last_phases: EncodePhases::default(),
            chain_phases: EncodePhases::default(),
            resolve_pipeline,
            resolve_layout,
            resolve_scale_buf,
            ss_target: None,
            rtt: HashMap::default(),
            rtt_epoch: 0,
            rtt_rendered: HashMap::default(),
            rtt_pitch: HashMap::default(),
            subrect_keep: Vec::new(),
            subrect_free: HashMap::default(),
            self_window_start: HashMap::default(),
            rtt_ever_rendered: HashSet::default(),
            rtt_raw: HashMap::default(),
            pass_shapes: HashMap::default(),
            pass_shapes_by_format: HashMap::default(),
            raw_to_float: None,
            rtt_rendered_raw: HashMap::default(),
            display_images: HashMap::default(),
            display_reads_snapshot: None,
            #[cfg(not(target_arch = "wasm32"))]
            premodules: std::sync::Arc::default(),
            #[cfg(not(target_arch = "wasm32"))]
            premodule_seen: HashSet::default(),
            rtt_binds: HashMap::default(),
            offscreen_only: false,
            rtt_cubes: HashMap::default(),
            orphan_candidate: None,
            presented: Vec::new(),
            presented_matched: false,
            flipped_ever: Vec::new(),
            last_flipped: None,
            rtt_used: HashMap::default(),
            rtt_alias_block: HashSet::default(),
            rtt_superseded: HashSet::default(),
            cube_faces_copied: HashSet::default(),
            rtt_rendered_uncopied: HashSet::default(),
            rtt_reads_snapshot: HashSet::default(),
            dest_colors: HashMap::default(),
            dest_gen: 0,
            dest_ms_copy: HashMap::default(),
            msaa_seed: HashMap::default(),
            tile_shadows: HashMap::default(),
            alpha_fill: HashMap::default(),
            rtt_depth_rendered: HashMap::default(),
            rtt_depth_addrs: HashMap::default(),
            retired_buffers: Vec::new(),
            buffer_pool: BufferPool::default(),
            retired_textures: Vec::new(),
            keep_depth: false,
            keep_stencil: false,
            stencil_sampled: HashMap::default(),
            stencil_planes: HashMap::default(),
            stencil_convert: HashMap::default(),
            depth_load_now: false,
            stencil_load_now: false,
            stencil_row_copy: None,
            stencil_staging: None,
            depth_saved: HashMap::default(),
            word_views: HashMap::default(),
            rtt_hits: 0,
            chain_shapes_seen: HashSet::default(),
            chain_shapes_capped: false,
            frame_sampled: HashSet::default(),
            chain_frames_seen: 0,
            sampled_addrs: HashSet::default(),
            gxp: {
                // >>> THE GPU TRANSCODER IS COMPILED HERE, WITH EVERY OTHER ENGINE SHADER.
                //
                // It is generic WGSL - it has no dependence on the title - and it used to be
                // built on the first recompiled DRAW instead, which put its compile inside a
                // gameplay frame. MEASURED on this desktop: 9.4-10.9 ms, and a desktop that
                // takes the guest's blocks verbatim is the floor, not the case that matters:
                // the device with no BC is the one that transcodes
                // [[vitaslop-phone-gpu-has-no-bc]].
                //
                // Only when the recompiler is ON, which is what the field's doc always
                // intended and what the draw-site build did not deliver: that site ran for
                // every run that drew a recompiled pair, so "a run that never transcodes
                // should not pay for it" was never true. A run with the recompiler off
                // reaches no `GxpLive::prepare` at all and now genuinely pays nothing.
                let mut gxp = GxpLive::from_env();
                if gxp.enabled {
                    let t = Stopwatch::start();
                    gxp.texenc = Some(crate::texenc::Transcoder::new(device));
                    report_status!(
                        "gxp texenc: compiled the GPU transcoder's compute pipelines in \
                         {:.1} ms, at renderer construction - generic engine WGSL, kept out \
                         of the frame that first needs it",
                        t.ms()
                    );
                }
                gxp
            },
        }
    }

    /// Set the supersample factor: 1 (default) renders the scene straight into the caller's
    /// target; `scale > 1` renders it at `scale x` the target dimensions into an offscreen
    /// buffer and box-downsamples that into the caller's view on resolve. This antialiases
    /// the geometric aliasing of a heavily-tessellated distant mesh (many sub-pixel triangles
    /// per final pixel) and coincident-panel z-fighting - the vehicle-speckle a 1x render
    /// shows - and mirrors the software oracle's `VITASLOP_SSAA` / `Framebuffer::downsampled`
    /// so both paths stay in lockstep. Cost is `scale^2` fill; 2x is the quality/perf
    /// default a caller opts into. A no-op if `scale` is unchanged.
    pub fn set_supersample(&mut self, scale: u32) {
        let scale = scale.max(1);
        if scale != self.ss_scale {
            self.ss_scale = scale;
            self.ss_target = None; // force a rebuild at the new scale
        }
    }
}
