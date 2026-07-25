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
