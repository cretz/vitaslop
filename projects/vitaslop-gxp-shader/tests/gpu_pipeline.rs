//! Real-GPU pipeline-creation validation for LINKED vertex+fragment modules.
//!
//! [`crate::link::link_programs`] emits one WGSL module carrying both stages with a matched
//! `@location` varying interface and a three-group binding namespace. The `oracle` harness
//! proves that module is naga-valid (the WGSL front-end wgpu uses). This test goes one step
//! further and the exact step the live `GxmRenderer` wiring must take: it builds a REAL
//! `wgpu::RenderPipeline` from the linked module - a shader module, three bind-group layouts
//! (vertex uniform @group0, fragment uniform @group1, samplers @group2), and a vertex-buffer
//! layout derived from the linked program's attribute plan. If wgpu accepts it, the binding
//! scheme + the guest vertex layout are pipeline-valid, not just parseable text - so the
//! renderer wiring can bind the draw's real resources to it with confidence.
//!
//! Game-derived shader bytes are a private oracle: this reads them from `VITASLOP_GXP_DUMPS`
//! and the pairings from `VITASLOP_GXP_PAIRS`, and SKIPS cleanly when either is unset (so CI
//! stays green with no fixture). It also skips cleanly when no GPU adapter is present (a
//! headless box), so it never breaks a GPU-less build.

use std::fs;
use std::path::PathBuf;

use vitaslop_gxp_shader::{link_programs, LinkedProgram};

mod common;

/// Acquire a headless GPU device, or `None` when no adapter is available (skip the test).
fn device() -> Option<(wgpu::Device, wgpu::Queue, String)> {
    let instance = wgpu::Instance::default();
    let adapter = pollster::block_on(instance.request_adapter(&wgpu::RequestAdapterOptions {
        power_preference: wgpu::PowerPreference::HighPerformance,
        force_fallback_adapter: false,
        compatible_surface: None,
        apply_limit_buckets: false,
    }))
    .ok()?;
    let name = adapter.get_info().name;
    let (device, queue) = pollster::block_on(adapter.request_device(&wgpu::DeviceDescriptor {
        label: Some("gxp-pipeline-test"),
        required_features: wgpu::Features::empty(),
        required_limits: wgpu::Limits::downlevel_defaults(),
        experimental_features: wgpu::ExperimentalFeatures::disabled(),
        memory_hints: wgpu::MemoryHints::default(),
        trace: wgpu::Trace::Off,
    }))
    .ok()?;
    Some((device, queue, name))
}

/// Build the three bind-group layouts a linked module declares: vertex uniform @group0,
/// fragment uniform @group1, samplers @group2. Unused groups get an empty layout so the
/// pipeline layout still covers every group index the shader may reference.
fn bind_group_layouts(device: &wgpu::Device, linked: &LinkedProgram) -> [wgpu::BindGroupLayout; 3] {
    let uniform = |visibility: wgpu::ShaderStages| wgpu::BindGroupLayoutEntry {
        binding: 0,
        visibility,
        ty: wgpu::BindingType::Buffer {
            ty: wgpu::BufferBindingType::Uniform,
            has_dynamic_offset: false,
            min_binding_size: None,
        },
        count: None,
    };

    // group 0: vertex default-uniform buffer (present only when the vertex reads uniforms).
    let g0_entries: Vec<wgpu::BindGroupLayoutEntry> = if linked.vertex_bindings.sa_lane_count > 0 {
        vec![uniform(wgpu::ShaderStages::VERTEX)]
    } else {
        vec![]
    };
    // group 1: fragment default-uniform buffer.
    let g1_entries: Vec<wgpu::BindGroupLayoutEntry> = if linked.fragment_bindings.sa_lane_count > 0 {
        vec![uniform(wgpu::ShaderStages::FRAGMENT)]
    } else {
        vec![]
    };
    // group 2: one texture+sampler pair per referenced sampler unit (t = 2*i, s = 2*i+1).
    let mut g2_entries: Vec<wgpu::BindGroupLayoutEntry> = Vec::new();
    for (i, b) in linked.fragment_bindings.samplers.iter().enumerate() {
        let view_dimension = if b.coords >= 3 {
            wgpu::TextureViewDimension::D3
        } else {
            wgpu::TextureViewDimension::D2
        };
        g2_entries.push(wgpu::BindGroupLayoutEntry {
            binding: i as u32 * 2,
            visibility: wgpu::ShaderStages::FRAGMENT,
            ty: wgpu::BindingType::Texture {
                sample_type: wgpu::TextureSampleType::Float { filterable: true },
                view_dimension,
                multisampled: false,
            },
            count: None,
        });
        g2_entries.push(wgpu::BindGroupLayoutEntry {
            binding: i as u32 * 2 + 1,
            visibility: wgpu::ShaderStages::FRAGMENT,
            ty: wgpu::BindingType::Sampler(wgpu::SamplerBindingType::Filtering),
            count: None,
        });
    }

    let make = |label: &str, entries: &[wgpu::BindGroupLayoutEntry]| {
        device.create_bind_group_layout(&wgpu::BindGroupLayoutDescriptor { label: Some(label), entries })
    };
    [
        make("gxp-g0-vertex-uniform", &g0_entries),
        make("gxp-g1-fragment-uniform", &g1_entries),
        make("gxp-g2-samplers", &g2_entries),
    ]
}

/// The vertex-buffer layout the linked module's `@location` inputs require, derived from the
/// linked program's attribute plan: each attribute is a float vector of its component count,
/// placed at a sequential offset. (The live renderer supplies the guest's real stride/offsets;
/// this proves the shader's declared inputs form a pipeline-valid layout.)
fn vertex_attributes(linked: &LinkedProgram) -> (Vec<wgpu::VertexAttribute>, u64) {
    let mut attrs = Vec::new();
    let mut offset = 0u64;
    for a in &linked.vertex_bindings.attributes {
        let format = match a.components {
            1 => wgpu::VertexFormat::Float32,
            2 => wgpu::VertexFormat::Float32x2,
            3 => wgpu::VertexFormat::Float32x3,
            _ => wgpu::VertexFormat::Float32x4,
        };
        attrs.push(wgpu::VertexAttribute { format, offset, shader_location: a.location });
        offset += format.size();
    }
    (attrs, offset.max(4))
}

/// Build a real render pipeline from a linked module, panicking with a clear message if wgpu
/// rejects it (that is a genuine failure the renderer would hit). Returns nothing on success.
fn build_pipeline(device: &wgpu::Device, name: &str, linked: &LinkedProgram) {
    let scope = device.push_error_scope(wgpu::ErrorFilter::Validation);

    let module = device.create_shader_module(wgpu::ShaderModuleDescriptor {
        label: Some(name),
        source: wgpu::ShaderSource::Wgsl(linked.wgsl.clone().into()),
    });
    let layouts = bind_group_layouts(device, linked);
    let pipeline_layout = device.create_pipeline_layout(&wgpu::PipelineLayoutDescriptor {
        label: Some("gxp-pipeline-layout"),
        bind_group_layouts: &[Some(&layouts[0]), Some(&layouts[1]), Some(&layouts[2])],
        immediate_size: 0,
    });
    let (attrs, stride) = vertex_attributes(linked);
    let vertex_layout = wgpu::VertexBufferLayout {
        array_stride: stride,
        step_mode: wgpu::VertexStepMode::Vertex,
        attributes: &attrs,
    };
    let buffers: Vec<Option<wgpu::VertexBufferLayout>> =
        if attrs.is_empty() { vec![] } else { vec![Some(vertex_layout)] };
    let _pipeline = device.create_render_pipeline(&wgpu::RenderPipelineDescriptor {
        label: Some(name),
        layout: Some(&pipeline_layout),
        vertex: wgpu::VertexState {
            module: &module,
            entry_point: Some("vs_main"),
            buffers: &buffers,
            compilation_options: Default::default(),
        },
        fragment: Some(wgpu::FragmentState {
            module: &module,
            entry_point: Some("fs_main"),
            targets: &[Some(wgpu::ColorTargetState {
                format: wgpu::TextureFormat::Rgba8Unorm,
                blend: None,
                write_mask: wgpu::ColorWrites::ALL,
            })],
            compilation_options: Default::default(),
        }),
        primitive: wgpu::PrimitiveState {
            topology: wgpu::PrimitiveTopology::TriangleList,
            cull_mode: None,
            ..Default::default()
        },
        depth_stencil: None,
        multisample: Default::default(),
        multiview_mask: None,
        cache: None,
    });

    // Surface any deferred validation error (pipeline creation can report async).
    if let Some(err) = pollster::block_on(scope.pop()) {
        panic!("{name}: wgpu rejected the linked pipeline: {err}");
    }
}

fn dump_dir() -> Option<PathBuf> {
    let v = std::env::var("VITASLOP_GXP_DUMPS").ok()?;
    if v.trim().is_empty() { None } else { Some(PathBuf::from(v)) }
}

#[test]
#[ignore = "requires VITASLOP_GXP_DUMPS + VITASLOP_GXP_PAIRS + a GPU adapter; run explicitly"]
fn linked_pairs_build_real_pipelines() {
    let Some(dir) = dump_dir() else {
        eprintln!("VITASLOP_GXP_DUMPS unset - skipping GPU pipeline test");
        return;
    };
    let Ok(spec) = std::env::var("VITASLOP_GXP_PAIRS") else {
        eprintln!("set VITASLOP_GXP_PAIRS=\"vh:fh,...\" to build pipelines from real pairs");
        return;
    };
    let Some((device, _queue, adapter)) = device() else {
        eprintln!("no GPU adapter - skipping GPU pipeline test");
        return;
    };
    eprintln!("building linked pipelines on: {adapter}");

    let pairs: Vec<(String, String)> = spec
        .split(',')
        .filter_map(|p| p.split_once(':').map(|(v, f)| (v.trim().to_string(), f.trim().to_string())))
        .collect();

    let mut built = 0u32;
    let mut fell_back = 0u32;
    for (vh, fh) in &pairs {
        let (Ok(vb), Ok(fb)) = (
            fs::read(dir.join(format!("vert_{vh}.gxp"))),
            fs::read(dir.join(format!("frag_{fh}.gxp"))),
        ) else {
            continue;
        };
        match link_programs(&vb, &fb) {
            Ok(linked) => {
                let name = format!("vert_{vh}+frag_{fh}");
                build_pipeline(&device, &name, &linked);
                built += 1;
                eprintln!(
                    "  {name}  PIPELINE OK (attrs={} varyings v={}/f={} samplers={})",
                    linked.vertex_bindings.attributes.len(),
                    linked.vertex_varyings,
                    linked.fragment_varyings,
                    linked.fragment_bindings.samplers.len(),
                );
            }
            Err(_) => fell_back += 1,
        }
    }
    eprintln!("=> {built} linked pairs built a real wgpu pipeline; {fell_back} fell back (unlinkable)");
    assert!(built > 0, "no linkable pairs built a pipeline - check VITASLOP_GXP_PAIRS");
}

/// A program whose guest-memory windows outgrow a uniform binding declares them as a
/// read-only STORAGE array, and that module builds a real pipeline against a layout whose
/// binding 1 is `Storage { read_only: true }` with a dynamic offset - the layout the renderer
/// builds for it - and a bind group over a 1 MiB storage range binds to it. Authored, so it
/// needs no corpus; skips with no adapter.
#[test]
fn a_storage_window_module_builds_a_real_pipeline() {
    use vitaslop_gxp_shader::gxpwrite::{self, ContainerSpec, ParamSpec, ProgramSpec, VertexOutputs};
    use vitaslop_gxp_shader::ir::Bank;
    use vitaslop_gxp_shader::usse::asm::{self, Dest, IntSrc};
    let t = Bank::Temp;
    let code = vec![
        asm::imad_step(false, false, Dest::new(t, 8), (Bank::PrimaryAttr, 0), IntSrc::Imm(16), IntSrc::Reg(Bank::SecondaryAttr, 20))
            .unwrap(),
        asm::ldmem(Dest::new(t, 0), 4, (t, 8), IntSrc::Imm(0), IntSrc::Imm(0)).unwrap(),
    ];
    let blob = gxpwrite::write(
        &ProgramSpec::vertex(code)
            .with_parameters(vec![
                ParamSpec::attribute("IN.position", 0, 4),
                ParamSpec::uniform_buffer("g_Grid", 0, 4 << 20),
            ])
            .with_default_uniform_regs(16)
            .with_containers(vec![
                ContainerSpec { index: 14, base_sa: 0, size_regs: 16 },
                ContainerSpec { index: 19, base_sa: 20, size_regs: 4 },
            ])
            .with_ub_bindings(vec![(0, 0)])
            .with_registers(8, 24, 16)
            .with_outputs(VertexOutputs { color0: false, texcoords: vec![(0, 4)] }),
    );
    let (_, module) = vitaslop_gxp_shader::recompile_vertex_module(&blob).expect("recompiles");
    let windows = &module.bindings.mem_windows;
    assert!(vitaslop_gxp_shader::module::mem_window_storage(windows));
    let Some((device, _queue, adapter)) = device() else {
        eprintln!("no GPU adapter - skipping the storage-window pipeline test");
        return;
    };
    let bind_bytes = u64::from(vitaslop_gxp_shader::module::mem_window_vec4_count(windows)) * 16;
    let scope = device.push_error_scope(wgpu::ErrorFilter::Validation);
    let shader = device.create_shader_module(wgpu::ShaderModuleDescriptor {
        label: Some("storage-window"),
        source: wgpu::ShaderSource::Wgsl(module.wgsl.clone().into()),
    });
    let mut entries = Vec::new();
    if module.bindings.sa_lane_count > 0 {
        entries.push(wgpu::BindGroupLayoutEntry {
            binding: 0,
            visibility: wgpu::ShaderStages::VERTEX,
            ty: wgpu::BindingType::Buffer {
                ty: wgpu::BufferBindingType::Uniform,
                has_dynamic_offset: true,
                min_binding_size: wgpu::BufferSize::new(u64::from(module.bindings.sa_vec4_count()) * 16),
            },
            count: None,
        });
    }
    entries.push(wgpu::BindGroupLayoutEntry {
        binding: 1,
        visibility: wgpu::ShaderStages::VERTEX,
        ty: wgpu::BindingType::Buffer {
            ty: wgpu::BufferBindingType::Storage { read_only: true },
            has_dynamic_offset: true,
            min_binding_size: wgpu::BufferSize::new(bind_bytes),
        },
        count: None,
    });
    let bgl = device.create_bind_group_layout(&wgpu::BindGroupLayoutDescriptor { label: None, entries: &entries });
    let empty = device.create_bind_group_layout(&wgpu::BindGroupLayoutDescriptor { label: None, entries: &[] });
    let layout = device.create_pipeline_layout(&wgpu::PipelineLayoutDescriptor {
        label: None,
        bind_group_layouts: &[Some(&bgl), Some(&empty)],
        immediate_size: 0,
    });
    let attrs = [wgpu::VertexAttribute { format: wgpu::VertexFormat::Float32x4, offset: 0, shader_location: 0 }];
    let _pipeline = device.create_render_pipeline(&wgpu::RenderPipelineDescriptor {
        label: Some("storage-window"),
        layout: Some(&layout),
        vertex: wgpu::VertexState {
            module: &shader,
            entry_point: Some("vs_main"),
            buffers: &[Some(wgpu::VertexBufferLayout {
                array_stride: 16,
                step_mode: wgpu::VertexStepMode::Vertex,
                attributes: &attrs,
            })],
            compilation_options: Default::default(),
        },
        // Vertex-only: a depth-only pass, the shape of the draw that found this.
        fragment: None,
        primitive: Default::default(),
        depth_stencil: Some(wgpu::DepthStencilState {
            format: wgpu::TextureFormat::Depth32Float,
            depth_write_enabled: Some(true),
            depth_compare: Some(wgpu::CompareFunction::Less),
            stencil: Default::default(),
            bias: Default::default(),
        }),
        multisample: Default::default(),
        multiview_mask: None,
        cache: None,
    });
    // The arena the renderer binds it from: UNIFORM | STORAGE, the window at a 256-aligned
    // dynamic offset past the SA block.
    let arena = device.create_buffer(&wgpu::BufferDescriptor {
        label: None,
        size: 256 + bind_bytes,
        usage: wgpu::BufferUsages::UNIFORM | wgpu::BufferUsages::STORAGE | wgpu::BufferUsages::COPY_DST,
        mapped_at_creation: false,
    });
    let mut bg_entries = Vec::new();
    if module.bindings.sa_lane_count > 0 {
        bg_entries.push(wgpu::BindGroupEntry {
            binding: 0,
            resource: wgpu::BindingResource::Buffer(wgpu::BufferBinding {
                buffer: &arena,
                offset: 0,
                size: wgpu::BufferSize::new(u64::from(module.bindings.sa_vec4_count()) * 16),
            }),
        });
    }
    bg_entries.push(wgpu::BindGroupEntry {
        binding: 1,
        resource: wgpu::BindingResource::Buffer(wgpu::BufferBinding {
            buffer: &arena,
            offset: 0,
            size: wgpu::BufferSize::new(bind_bytes),
        }),
    });
    let _bg = device.create_bind_group(&wgpu::BindGroupDescriptor { label: None, layout: &bgl, entries: &bg_entries });
    if let Some(err) = pollster::block_on(scope.pop()) {
        panic!("{adapter}: wgpu rejected the storage-window pipeline: {err}\n{}", module.wgsl);
    }
}

/// **A FRAGMENT PROGRAM THAT STORES TO GUEST MEMORY, END TO END ON A REAL GPU.**
///
/// The authored edge-mask arithmetic (`common::tile_store_pair`) addresses each fragment's store
/// through the TILE registers and writes the word's own index. Linked, built against the group-3
/// layout the renderer gives a storing pipeline (the depth block at binding 0, the writable
/// mirror at 2, the window headers at 3), and drawn over a 40x36 target - which crosses the
/// 16-pixel boundary of both registers' split twice in each axis - every pixel's word must hold
/// its own index, and every word past a row's 40 pixels (the stride is 48) must be untouched.
/// That is the emitted split, the store helper, the pointer seeded from the header and the
/// bindings, all at once. Authored, so it needs no corpus; skips with no adapter.
#[test]
fn a_storing_fragment_program_writes_each_pixel_through_a_real_pipeline() {
    const BASE: u32 = 0x8100_0000;
    const UNTOUCHED: u32 = 0xaaaa_aaaa;
    let (stride, w, h) = (48u32, 40u32, 36u32);
    let (vblob, fblob) = common::tile_store_pair(stride, h);
    let linked = link_programs(&vblob, &fblob).expect("the storing pair links");
    assert_eq!(linked.fragment_bindings.store_windows.len(), 1);
    let Some((device, queue, adapter)) = device() else {
        eprintln!("no GPU adapter - skipping the storing-pipeline test");
        return;
    };
    let scope = device.push_error_scope(wgpu::ErrorFilter::Validation);
    let module = device.create_shader_module(wgpu::ShaderModuleDescriptor {
        label: Some("storing-pair"),
        source: wgpu::ShaderSource::Wgsl(linked.wgsl.clone().into()),
    });
    let uniform = |binding: u32, vis: wgpu::ShaderStages| wgpu::BindGroupLayoutEntry {
        binding,
        visibility: vis,
        ty: wgpu::BindingType::Buffer { ty: wgpu::BufferBindingType::Uniform, has_dynamic_offset: false, min_binding_size: None },
        count: None,
    };
    let frag = wgpu::ShaderStages::FRAGMENT;
    let layout = |entries: &[wgpu::BindGroupLayoutEntry]| device.create_bind_group_layout(&wgpu::BindGroupLayoutDescriptor { label: None, entries });
    assert_eq!(linked.vertex_bindings.sa_lane_count, 0, "the pass-through vertex program reads no uniforms");
    let g0 = layout(&[]);
    let g1 = layout(&[uniform(0, frag)]);
    let g2 = layout(&[]);
    let g3 = layout(&[
        uniform(0, wgpu::ShaderStages::VERTEX_FRAGMENT),
        wgpu::BindGroupLayoutEntry {
            binding: 2,
            visibility: frag,
            ty: wgpu::BindingType::Buffer {
                ty: wgpu::BufferBindingType::Storage { read_only: false },
                has_dynamic_offset: false,
                min_binding_size: None,
            },
            count: None,
        },
        uniform(3, frag),
    ]);
    let pl = device.create_pipeline_layout(&wgpu::PipelineLayoutDescriptor {
        label: None,
        bind_group_layouts: &[Some(&g0), Some(&g1), Some(&g2), Some(&g3)],
        immediate_size: 0,
    });
    let (attrs, vstride) = vertex_attributes(&linked);
    let pipeline = device.create_render_pipeline(&wgpu::RenderPipelineDescriptor {
        label: Some("storing-pair"),
        layout: Some(&pl),
        vertex: wgpu::VertexState {
            module: &module,
            entry_point: Some("vs_main"),
            buffers: &[Some(wgpu::VertexBufferLayout { array_stride: vstride, step_mode: wgpu::VertexStepMode::Vertex, attributes: &attrs })],
            compilation_options: Default::default(),
        },
        fragment: Some(wgpu::FragmentState {
            module: &module,
            entry_point: Some("fs_main"),
            targets: &[Some(wgpu::ColorTargetState { format: wgpu::TextureFormat::Rgba8Unorm, blend: None, write_mask: wgpu::ColorWrites::ALL })],
            compilation_options: Default::default(),
        }),
        primitive: Default::default(),
        depth_stencil: None,
        multisample: Default::default(),
        multiview_mask: None,
        cache: None,
    });
    let buffer = |usage: wgpu::BufferUsages, bytes: &[u8]| {
        let b = device.create_buffer(&wgpu::BufferDescriptor { label: None, size: bytes.len() as u64, usage: usage | wgpu::BufferUsages::COPY_DST, mapped_at_creation: false });
        queue.write_buffer(&b, 0, bytes);
        b
    };
    let words = |w: &[u32]| w.iter().flat_map(|v| v.to_le_bytes()).collect::<Vec<u8>>();
    // A full-screen quad, clip position straight through.
    let quad: Vec<f32> = [(-1.0, -1.0), (1.0, -1.0), (1.0, 1.0), (-1.0, -1.0), (1.0, 1.0), (-1.0, 1.0)]
        .iter()
        .flat_map(|&(x, y)| [x, y, 0.5f32, 1.0])
        .collect();
    let vbuf = buffer(wgpu::BufferUsages::VERTEX, &quad.iter().flat_map(|v| v.to_le_bytes()).collect::<Vec<u8>>());
    // The fragment SA bank: `sa[0]` the stride as a raw integer, padded to the binding's vec4s.
    let sa_vec4s = linked.fragment_bindings.sa_vec4_count().max(1) as usize;
    let mut sa = vec![0u32; sa_vec4s * 4];
    sa[common::TILE_STORE_STRIDE_SA] = stride;
    let sa_buf = buffer(wgpu::BufferUsages::UNIFORM, &words(&sa));
    let depth_buf = buffer(wgpu::BufferUsages::UNIFORM, &[0u8; 112]);
    let mirror = buffer(wgpu::BufferUsages::STORAGE | wgpu::BufferUsages::COPY_SRC, &words(&vec![UNTOUCHED; (stride * h) as usize]));
    let header = buffer(wgpu::BufferUsages::UNIFORM, &words(&[BASE, stride * h, 0, 0]));
    let bind = |layout: &wgpu::BindGroupLayout, entries: &[wgpu::BindGroupEntry]| {
        device.create_bind_group(&wgpu::BindGroupDescriptor { label: None, layout, entries })
    };
    let bg0 = bind(&g0, &[]);
    let bg1 = bind(&g1, &[wgpu::BindGroupEntry { binding: 0, resource: sa_buf.as_entire_binding() }]);
    let bg2 = bind(&g2, &[]);
    let bg3 = bind(
        &g3,
        &[
            wgpu::BindGroupEntry { binding: 0, resource: depth_buf.as_entire_binding() },
            wgpu::BindGroupEntry { binding: 2, resource: mirror.as_entire_binding() },
            wgpu::BindGroupEntry { binding: 3, resource: header.as_entire_binding() },
        ],
    );
    let target = device.create_texture(&wgpu::TextureDescriptor {
        label: None,
        size: wgpu::Extent3d { width: w, height: h, depth_or_array_layers: 1 },
        mip_level_count: 1,
        sample_count: 1,
        dimension: wgpu::TextureDimension::D2,
        format: wgpu::TextureFormat::Rgba8Unorm,
        usage: wgpu::TextureUsages::RENDER_ATTACHMENT,
        view_formats: &[],
    });
    let view = target.create_view(&Default::default());
    let readback = device.create_buffer(&wgpu::BufferDescriptor {
        label: None,
        size: mirror.size(),
        usage: wgpu::BufferUsages::MAP_READ | wgpu::BufferUsages::COPY_DST,
        mapped_at_creation: false,
    });
    let mut enc = device.create_command_encoder(&Default::default());
    {
        let mut pass = enc.begin_render_pass(&wgpu::RenderPassDescriptor {
            label: None,
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
        pass.set_pipeline(&pipeline);
        for (i, bg) in [&bg0, &bg1, &bg2, &bg3].into_iter().enumerate() {
            pass.set_bind_group(i as u32, bg, &[]);
        }
        pass.set_vertex_buffer(0, vbuf.slice(..));
        pass.draw(0..6, 0..1);
    }
    enc.copy_buffer_to_buffer(&mirror, 0, &readback, 0, mirror.size());
    queue.submit([enc.finish()]);
    readback.slice(..).map_async(wgpu::MapMode::Read, |_| {});
    let _ = device.poll(wgpu::PollType::wait_indefinitely());
    if let Some(err) = pollster::block_on(scope.pop()) {
        panic!("{adapter}: wgpu rejected the storing pipeline: {err}\n{}", linked.wgsl);
    }
    let got: Vec<u32> = readback
        .slice(..)
        .get_mapped_range()
        .expect("mapped")
        .chunks_exact(4)
        .map(|c| u32::from_le_bytes([c[0], c[1], c[2], c[3]]))
        .collect();
    for y in 0..h {
        for x in 0..stride {
            let i = y * stride + x;
            let want = if x < w { i } else { UNTOUCHED };
            assert_eq!(got[i as usize], want, "{adapter}: the word for pixel ({x}, {y})");
        }
    }
}
