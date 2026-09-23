//! **THE SHIPPED FACING WIRING, ON A REAL GPU: an authored pair, linked by the shipped linker,
//! drawn in both windings.**
//!
//! Both case rigs PIN `gxp_front_facing` to a constant, because neither side of the differential
//! models a rasteriser - so they check what a program does WITH the facing bit and nothing about
//! where the bit comes from. That wiring is one line of the linked module
//! (`let gxp_front_facing: bool = in.front_facing;`) plus the pipeline's `front_face`, and a
//! title's two-sided lighting turns on it: the wrong sense paints every visible surface of a
//! car-body livery black.
//!
//! So this renders it. The fragment program is the corpus's own facing idiom -
//! `p0 = (GLOBAL[16] & 1) != 0`, then a write guarded by `p0` - and it paints WHITE where the
//! bit is set and black (alpha one) where it is clear. Two triangles, one wound each way in clip
//! space, through the pipeline state the renderer builds (`FrontFace::Ccw`, no culling). What
//! each reports is MEASURED and pinned below, not derived from any spec's winding convention:
//! the renderer's own cull mapping records that the obvious derivation of that convention came
//! out backwards once already.
//!
//! Skips cleanly when no GPU adapter is present.

use vitaslop_gxp_shader::gxpwrite::{self, ProgramSpec, VertexOutputs};
use vitaslop_gxp_shader::ir::{Bank, Op, Predicate};
use vitaslop_gxp_shader::link_programs;
use vitaslop_gxp_shader::usse::asm::{self, Dest, Src};

const W: u32 = 8;
const H: u32 = 4;

fn device() -> Option<(wgpu::Device, wgpu::Queue)> {
    let instance = wgpu::Instance::default();
    let adapter = pollster::block_on(instance.request_adapter(&wgpu::RequestAdapterOptions {
        power_preference: wgpu::PowerPreference::HighPerformance,
        force_fallback_adapter: false,
        compatible_surface: None,
        apply_limit_buckets: false,
    }))
    .ok()?;
    pollster::block_on(adapter.request_device(&wgpu::DeviceDescriptor {
        label: Some("gxp-facing-test"),
        required_features: wgpu::Features::empty(),
        required_limits: wgpu::Limits::downlevel_defaults(),
        experimental_features: wgpu::ExperimentalFeatures::disabled(),
        memory_hints: wgpu::MemoryHints::default(),
        trace: wgpu::Trace::Off,
    }))
    .ok()
}

/// The pair: a vertex program that passes its position attribute through as the clip position,
/// and a fragment program that paints the facing bit.
fn pair() -> (Vec<u8>, Vec<u8>) {
    let zero = Src::cnst(0).swz([4, 4, 4, 4]);
    let vertex = ProgramSpec::vertex(vec![asm::alu(
        Op::Add,
        false,
        Dest::new(Bank::Output, 0),
        [true; 4],
        zero,
        Src::reg(Bank::PrimaryAttr, 0),
    )
    .unwrap()])
    .with_parameters(vec![gxpwrite::ParamSpec::attribute("IN.position", 0, 4)])
    .with_registers(4, 0, 4)
    .with_outputs(VertexOutputs::default());
    // o0 = (0, 0, 0, 1); then, where p0 holds, o0 = (1, 1, 1, 1). `r[0]` is never written, so it
    // is the zero each add needs as its register operand.
    let fragment = ProgramSpec::fragment(vec![
        asm::vtst_global_bit(0, 16, 1).unwrap(),
        asm::alu(Op::Add, false, Dest::new(Bank::Output, 0), [true; 4], Src::cnst(0).swz([4, 4, 4, 5]), Src::reg(Bank::Temp, 0))
            .unwrap(),
        asm::alu_pred(
            Op::Add,
            false,
            Predicate::IfP(0),
            Dest::new(Bank::Output, 0),
            [true; 4],
            Src::cnst(0).swz([5, 5, 5, 5]),
            Src::reg(Bank::Temp, 0),
        )
        .unwrap(),
    ])
    .with_registers(0, 0, 4);
    (gxpwrite::write(&vertex), gxpwrite::write(&fragment))
}

/// Draw `tri` (three clip-space xy pairs) over a cleared BLUE target and return the RGBA of every
/// pixel, row-major.
fn render(device: &wgpu::Device, queue: &wgpu::Queue, wgsl: &str, tris: &[[[f32; 2]; 3]]) -> Vec<[u8; 4]> {
    let scope = device.push_error_scope(wgpu::ErrorFilter::Validation);
    let module = device.create_shader_module(wgpu::ShaderModuleDescriptor {
        label: Some("facing"),
        source: wgpu::ShaderSource::Wgsl(wgsl.into()),
    });
    let empty = device.create_bind_group_layout(&wgpu::BindGroupLayoutDescriptor { label: None, entries: &[] });
    let layout = device.create_pipeline_layout(&wgpu::PipelineLayoutDescriptor {
        label: None,
        bind_group_layouts: &[Some(&empty), Some(&empty), Some(&empty)],
        immediate_size: 0,
    });
    let attrs = [wgpu::VertexAttribute { format: wgpu::VertexFormat::Float32x4, offset: 0, shader_location: 0 }];
    let pipeline = device.create_render_pipeline(&wgpu::RenderPipelineDescriptor {
        label: Some("facing"),
        layout: Some(&layout),
        vertex: wgpu::VertexState {
            module: &module,
            entry_point: Some("vs_main"),
            buffers: &[Some(wgpu::VertexBufferLayout {
                array_stride: 16,
                step_mode: wgpu::VertexStepMode::Vertex,
                attributes: &attrs,
            })],
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
        // The renderer's own primitive state: counter-clockwise is front, nothing culled.
        primitive: wgpu::PrimitiveState {
            topology: wgpu::PrimitiveTopology::TriangleList,
            front_face: wgpu::FrontFace::Ccw,
            cull_mode: None,
            ..Default::default()
        },
        depth_stencil: None,
        multisample: Default::default(),
        multiview_mask: None,
        cache: None,
    });
    let mut verts: Vec<u8> = Vec::new();
    for tri in tris {
        for p in tri {
            for v in [p[0], p[1], 0.5, 1.0] {
                verts.extend_from_slice(&v.to_le_bytes());
            }
        }
    }
    let vbuf = device.create_buffer(&wgpu::BufferDescriptor {
        label: None,
        size: verts.len() as u64,
        usage: wgpu::BufferUsages::VERTEX | wgpu::BufferUsages::COPY_DST,
        mapped_at_creation: false,
    });
    queue.write_buffer(&vbuf, 0, &verts);
    let target = device.create_texture(&wgpu::TextureDescriptor {
        label: None,
        size: wgpu::Extent3d { width: W, height: H, depth_or_array_layers: 1 },
        mip_level_count: 1,
        sample_count: 1,
        dimension: wgpu::TextureDimension::D2,
        format: wgpu::TextureFormat::Rgba8Unorm,
        usage: wgpu::TextureUsages::RENDER_ATTACHMENT | wgpu::TextureUsages::COPY_SRC,
        view_formats: &[],
    });
    let view = target.create_view(&Default::default());
    let empty_group = device.create_bind_group(&wgpu::BindGroupDescriptor { label: None, layout: &empty, entries: &[] });
    let mut enc = device.create_command_encoder(&Default::default());
    {
        let mut pass = enc.begin_render_pass(&wgpu::RenderPassDescriptor {
            label: None,
            color_attachments: &[Some(wgpu::RenderPassColorAttachment {
                view: &view,
                depth_slice: None,
                resolve_target: None,
                ops: wgpu::Operations { load: wgpu::LoadOp::Clear(wgpu::Color::BLUE), store: wgpu::StoreOp::Store },
            })],
            depth_stencil_attachment: None,
            timestamp_writes: None,
            occlusion_query_set: None,
            multiview_mask: None,
        });
        pass.set_pipeline(&pipeline);
        for g in 0..3 {
            pass.set_bind_group(g, &empty_group, &[]);
        }
        pass.set_vertex_buffer(0, vbuf.slice(..));
        pass.draw(0..(tris.len() * 3) as u32, 0..1);
    }
    let row = (W * 4).div_ceil(256) * 256;
    let buf = device.create_buffer(&wgpu::BufferDescriptor {
        label: None,
        size: u64::from(row * H),
        usage: wgpu::BufferUsages::COPY_DST | wgpu::BufferUsages::MAP_READ,
        mapped_at_creation: false,
    });
    enc.copy_texture_to_buffer(
        wgpu::TexelCopyTextureInfo {
            texture: &target,
            mip_level: 0,
            origin: wgpu::Origin3d::ZERO,
            aspect: wgpu::TextureAspect::All,
        },
        wgpu::TexelCopyBufferInfo {
            buffer: &buf,
            layout: wgpu::TexelCopyBufferLayout { offset: 0, bytes_per_row: Some(row), rows_per_image: Some(H) },
        },
        wgpu::Extent3d { width: W, height: H, depth_or_array_layers: 1 },
    );
    queue.submit([enc.finish()]);
    if let Some(err) = pollster::block_on(scope.pop()) {
        panic!("wgpu rejected the linked facing pipeline: {err}\n{wgsl}");
    }
    let slice = buf.slice(..);
    slice.map_async(wgpu::MapMode::Read, |_| {});
    let _ = device.poll(wgpu::PollType::Wait { submission_index: None, timeout: None });
    let data = slice.get_mapped_range().expect("the poll above waited for the map");
    let mut out = Vec::new();
    for y in 0..H {
        for x in 0..W {
            let o = (y * row + x * 4) as usize;
            out.push([data[o], data[o + 1], data[o + 2], data[o + 3]]);
        }
    }
    out
}

/// The pixel at clip-space `(x, y)` (y up), in the row-major readback.
fn at(px: &[[u8; 4]], x: f32, y: f32) -> [u8; 4] {
    let col = (((x + 1.0) * 0.5) * W as f32) as u32;
    let row = (((1.0 - y) * 0.5) * H as f32) as u32;
    px[(row.min(H - 1) * W + col.min(W - 1)) as usize]
}

#[test]
fn the_linked_module_reports_facing_by_winding() {
    let Some((device, queue)) = device() else {
        eprintln!("no GPU adapter - skipping");
        return;
    };
    let (vbytes, fbytes) = pair();
    let linked = link_programs(&vbytes, &fbytes).expect("the authored facing pair links");
    // COUNTER-CLOCKWISE in clip space (y up) on the left half, CLOCKWISE on the right. Each
    // covers its half's lower-left corner, where the sample points below sit.
    let ccw = [[-1.0, -1.0], [0.0, -1.0], [-1.0, 1.0]];
    let cw = [[0.0, -1.0], [0.0, 1.0], [1.0, -1.0]];
    let px = render(&device, &queue, &linked.wgsl, &[ccw, cw]);
    let (left, right) = (at(&px, -0.875, -0.75), at(&px, 0.125, -0.75));
    const WHITE: [u8; 4] = [255, 255, 255, 255];
    const BLACK: [u8; 4] = [0, 0, 0, 255];
    println!("facing: CCW triangle paints {left:?}, CW triangle paints {right:?}");
    // Both drew: neither is the BLUE clear. A pair that failed to rasterise would otherwise read
    // as "the bit is clear" on both sides.
    assert!(left != [0, 0, 255, 255] && right != [0, 0, 255, 255], "a triangle did not draw: {left:?} {right:?}");
    // >>> MEASURED: the counter-clockwise triangle reports the bit SET and the clockwise one CLEAR.
    // Under `FrontFace::Ccw` that is `GLOBAL[16] bit 0 == front_facing`, the sense the renderer's
    // two-sided lighting was pinned to on a title's liveries. A module that inverted the bit, or
    // pinned it to a constant, paints both triangles alike.
    assert_eq!((left, right), (WHITE, BLACK), "the linked module's facing bit, by winding");
}
