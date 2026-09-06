//! Prove the emitter produces REAL, compilable WGSL - not merely plausible strings.
//!
//! Each emittable op is emitted, wrapped into a complete module, and run through naga's WGSL
//! front-end + validator (the same shader compiler wgpu uses on the shipped path). A syntax
//! or type error in the emit surfaces here as a hard failure, so the emittable op set is
//! guaranteed to be valid shader code, not just structurally shaped text. (The oracle
//! harness additionally validates the COMPLETE bindable modules of every real recompilable
//! shader; this file pins the per-op emit in isolation.)

use vitaslop_gxp_shader::ir::{
    Bank, BitwiseKind, CompareMethod, Instr, Op, Operand, Predicate, Shader, TestAlu, TestCmp,
    TestReduce, TexLod,
};
use vitaslop_gxp_shader::container::ProgramKind;
use vitaslop_gxp_shader::wgsl::{emit_body, emit_fragment, tex_units, wrap_module, wrap_vertex_module, TexBinding};

fn instr(op: Op, dest: Operand, srcs: Vec<Operand>, mask: [bool; 4]) -> Instr {
    Instr {
        op,
        pred: Predicate::Always,
        dest: Some(dest),
        write_mask: mask,
        srcs,
        half_precision: false,
        raw: 0,
        group: 0,
        blocked: None,
    }
}

fn validate(body: &str) {
    validate_with(body, &[]);
}

fn validate_src(module_src: &str) {
    let module = naga::front::wgsl::parse_str(module_src)
        .unwrap_or_else(|e| panic!("emitted WGSL failed to parse:\n{module_src}\n\nerror: {e:?}"));
    let mut validator = naga::valid::Validator::new(
        naga::valid::ValidationFlags::all(),
        naga::valid::Capabilities::all(),
    );
    validator
        .validate(&module)
        .unwrap_or_else(|e| panic!("emitted WGSL failed validation:\n{module_src}\n\nerror: {e:?}"));
}

fn validate_with(body: &str, units: &[TexBinding]) {
    let module_src = wrap_module(body, units, ProgramKind::Fragment);
    let module = naga::front::wgsl::parse_str(&module_src)
        .unwrap_or_else(|e| panic!("emitted WGSL failed to parse:\n{module_src}\n\nerror: {e:?}"));
    let mut validator = naga::valid::Validator::new(
        naga::valid::ValidationFlags::all(),
        naga::valid::Capabilities::all(),
    );
    validator
        .validate(&module)
        .unwrap_or_else(|e| panic!("emitted WGSL failed validation:\n{module_src}\n\nerror: {e:?}"));
}

#[test]
fn every_emittable_op_produces_valid_wgsl() {
    let d = |idx| Operand::plain(Bank::Output, idx, 1);
    let r = |idx| Operand::plain(Bank::Temp, idx, 0);
    let sa = |idx| Operand::plain(Bank::SecondaryAttr, idx, 3);
    let pa = |idx| Operand::plain(Bank::PrimaryAttr, idx, 2);
    let k = |sel| Operand::plain(Bank::Constant, sel, 0);
    let full = [true; 4];

    let cases: Vec<(&str, Shader)> = vec![
        ("mul", Shader { kind: ProgramKind::Fragment, instrs: vec![instr(Op::Mul, d(0), vec![r(4), sa(8)], full)] }),
        ("add", Shader { kind: ProgramKind::Fragment, instrs: vec![instr(Op::Add, d(0), vec![pa(4), r(8)], full)] }),
        ("mad", Shader { kind: ProgramKind::Fragment, instrs: vec![instr(Op::Mad, d(0), vec![r(4), sa(8), pa(0)], full)] }),
        ("min", Shader { kind: ProgramKind::Fragment, instrs: vec![instr(Op::Min, d(0), vec![r(4), r(8)], full)] }),
        ("max", Shader { kind: ProgramKind::Fragment, instrs: vec![instr(Op::Max, d(0), vec![r(4), r(8)], full)] }),
        ("frc", Shader { kind: ProgramKind::Fragment, instrs: vec![instr(Op::Frc, d(0), vec![r(4)], full)] }),
        ("dsx", Shader { kind: ProgramKind::Fragment, instrs: vec![instr(Op::Dsx, d(0), vec![r(4)], full)] }),
        ("dot4", Shader { kind: ProgramKind::Fragment, instrs: vec![instr(Op::Dot { components: 4 }, d(0), vec![r(4), sa(8)], full)] }),
        // Group 0x30 transcendentals (source broadcasts one component) + group 0x38 move.
        ("rcp", Shader { kind: ProgramKind::Fragment, instrs: vec![instr(Op::Rcp, d(0), vec![sa(4)], full)] }),
        ("rsq", Shader { kind: ProgramKind::Fragment, instrs: vec![instr(Op::Rsq, d(0), vec![pa(4)], full)] }),
        ("log", Shader { kind: ProgramKind::Fragment, instrs: vec![instr(Op::Log, d(0), vec![r(4)], full)] }),
        ("exp", Shader { kind: ProgramKind::Fragment, instrs: vec![instr(Op::Exp, d(0), vec![r(4)], full)] }),
        ("mov", Shader { kind: ProgramKind::Fragment, instrs: vec![instr(Op::Mov, d(0), vec![sa(8)], full)] }),
        ("cmov", Shader { kind: ProgramKind::Fragment, instrs: vec![instr(Op::Cmov { test: CompareMethod::LtZero }, d(0), vec![r(4), sa(8), pa(0)], full)] }),
        ("predicated", Shader { kind: ProgramKind::Fragment, instrs: vec![{
            let mut i = instr(Op::Add, d(0), vec![r(4), r(8)], full);
            i.pred = Predicate::IfP(2);
            i
        }] }),
        ("pack", Shader { kind: ProgramKind::Fragment, instrs: vec![instr(Op::Pack { src_half: false }, d(0), vec![r(4)], full)] }),
        ("and-imm", Shader { kind: ProgramKind::Fragment, instrs: vec![instr(Op::Bitwise { kind: BitwiseKind::And, imm: Some(0xFF), lane_bits: 32 }, d(0), vec![r(4)], [true, false, false, false])] }),
        ("shr-reg", Shader { kind: ProgramKind::Fragment, instrs: vec![instr(Op::Bitwise { kind: BitwiseKind::Shr, imm: None, lane_bits: 32 }, d(0), vec![r(4), r(8)], [true, false, false, false])] }),
        ("asr-reg", Shader { kind: ProgramKind::Fragment, instrs: vec![instr(Op::Bitwise { kind: BitwiseKind::Asr, imm: None, lane_bits: 32 }, d(0), vec![r(4), r(8)], [true, false, false, false])] }),
        ("const", Shader { kind: ProgramKind::Fragment, instrs: vec![instr(Op::Mul, d(0), vec![r(4), k(2)], full)] }),
        // The NaN constant table entries must also materialise as valid WGSL (via bitcast).
        ("const-nan", Shader { kind: ProgramKind::Fragment, instrs: vec![instr(Op::Mul, d(0), vec![r(4), k(0x38)], full)] }),
        // VTST in both families. The BITWISE form emits an integer expression rather than a
        // float one, and WGSL binds `!=` tighter than `&`, so an unparenthesised AND compiles
        // to `u32 & bool` and fails validation - which is exactly how it shipped until a real
        // shader first reached emit. Both families are covered here so neither can regress.
        ("test-float", Shader { kind: ProgramKind::Fragment, instrs: vec![instr(
            Op::Test { alu: TestAlu::Sub, cmp: TestCmp::Lt, reduce: TestReduce::Channel(0), pdst: 1, write_back: false },
            d(0), vec![r(4), sa(8)], full)] }),
        ("test-bitand", Shader { kind: ProgramKind::Fragment, instrs: vec![instr(
            Op::Test { alu: TestAlu::BitAnd, cmp: TestCmp::Ne, reduce: TestReduce::Channel(0), pdst: 0, write_back: false },
            d(0), vec![r(4), Operand::plain(Bank::Immediate, 1, 2)], full)] }),
        // The established facing GLOBAL, in the shape the real shaders use it.
        ("test-facing", Shader { kind: ProgramKind::Fragment, instrs: vec![instr(
            Op::Test { alu: TestAlu::BitAnd, cmp: TestCmp::Ne, reduce: TestReduce::Channel(0), pdst: 0, write_back: false },
            d(0), vec![Operand::plain(Bank::Global, 16, 1), Operand::plain(Bank::Immediate, 1, 2)], full)] }),
        // A fragment DEPTH write (0xF8 DEPTHF): no register destination, one scalar source,
        // and it stores through the entry point's `@builtin(frag_depth)`.
        ("depthf", Shader { kind: ProgramKind::Fragment, instrs: vec![{
            let mut i = instr(Op::DepthF, d(0), vec![r(0)], full);
            i.dest = None;
            i.write_mask = [false; 4];
            i
        }] }),
    ];

    for (name, sh) in &cases {
        let body = emit_fragment(sh).unwrap_or_else(|e| panic!("{name}: emit failed: {e}"));
        validate_with(&body, &tex_units(sh, |_| false));
    }
}

#[test]
fn texture_sample_produces_valid_wgsl() {
    // A 2D texture sample of sampler unit 3 from coord in pa registers, result to output.
    let coord = Operand::plain(Bank::PrimaryAttr, 4, 2);
    let sh = Shader {
        kind: ProgramKind::Fragment,
        instrs: vec![instr(Op::Tex { unit: 3, coords: 2, coord_half: false, lod: TexLod::Implicit }, Operand::plain(Bank::Output, 0, 1), vec![coord], [true; 4])],
    };
    let body = emit_fragment(&sh).unwrap();
    assert!(body.contains("textureSample(t3, s3,"), "tex emit:\n{body}");
    assert_eq!(tex_units(&sh, |_| false), vec![TexBinding { unit: 3, coords: 2, cube: false }]);
    validate_with(&body, &tex_units(&sh, |_| false));

    // A 3-component sample validates as a texture_3d binding when the container does not mark
    // the sampler a cube, and as a texture_cube when it does. The coordinate emit is the same
    // vec3 either way; only the declared type differs.
    let sh3 = Shader {
        kind: ProgramKind::Fragment,
        instrs: vec![instr(Op::Tex { unit: 6, coords: 3, coord_half: false, lod: TexLod::Implicit }, Operand::plain(Bank::Output, 0, 1), vec![Operand::plain(Bank::PrimaryAttr, 8, 2)], [true; 4])],
    };
    let body3 = emit_fragment(&sh3).unwrap();
    assert!(body3.contains("vec3<f32>"), "3D tex emits vec3 coord:\n{body3}");
    validate_with(&body3, &tex_units(&sh3, |_| false));
    let cube = tex_units(&sh3, |_| true);
    assert_eq!(cube, vec![TexBinding { unit: 6, coords: 3, cube: true }]);
    assert_eq!(cube[0].wgsl_type(), "texture_cube<f32>");
    validate_with(&body3, &cube);
}

#[test]
fn vertex_module_produces_valid_wgsl() {
    // A vertex program: clip position from a mad over an attribute + uniform, and two varying
    // groups written to o[6] and o[12]. The standalone vertex wrapper must be valid WGSL with a
    // @builtin(position) + @location varyings interface.
    let sh = Shader {
        kind: ProgramKind::Vertex,
        instrs: vec![
            instr(Op::Mad, Operand::plain(Bank::Output, 0, 1),
                vec![Operand::plain(Bank::PrimaryAttr, 0, 2), Operand::plain(Bank::SecondaryAttr, 0, 3), Operand::plain(Bank::Constant, 2, 0)], [true; 4]),
            instr(Op::Mov, Operand::plain(Bank::Output, 6, 1), vec![Operand::plain(Bank::PrimaryAttr, 4, 2)], [true; 4]),
            instr(Op::Mov, Operand::plain(Bank::Output, 12, 1), vec![Operand::plain(Bank::SecondaryAttr, 8, 3)], [true; 4]),
        ],
    };
    let body = emit_body(&sh).unwrap();
    // Output extent 16 -> ceil((16-4)/4) = 3 varying vec4s.
    validate_src(&wrap_vertex_module(&body, 3));
    // Position-only (no varyings) must also validate.
    let pos_only = Shader {
        kind: ProgramKind::Vertex,
        instrs: vec![instr(Op::Mov, Operand::plain(Bank::Output, 0, 1), vec![Operand::plain(Bank::PrimaryAttr, 0, 2)], [true; 4])],
    };
    validate_src(&wrap_vertex_module(&emit_body(&pos_only).unwrap(), 0));
}

#[test]
fn multi_instruction_shader_validates() {
    // A small straight-line arithmetic program: r0 = pa0*sa0; r1 = frac(r0); o = dot(r0..,sa..).
    let sh = Shader {
        kind: ProgramKind::Fragment,
        instrs: vec![
            instr(Op::Mad, Operand::plain(Bank::Temp, 0, 0),
                vec![Operand::plain(Bank::PrimaryAttr, 0, 2), Operand::plain(Bank::SecondaryAttr, 0, 3), Operand::plain(Bank::Constant, 0, 0)], [true; 4]),
            instr(Op::Frc, Operand::plain(Bank::Temp, 4, 0), vec![Operand::plain(Bank::Temp, 0, 0)], [true, true, false, false]),
            instr(Op::Dot { components: 3 }, Operand::plain(Bank::Output, 0, 1),
                vec![Operand::plain(Bank::Temp, 0, 0), Operand::plain(Bank::SecondaryAttr, 4, 3)], [true, false, false, false]),
        ],
    };
    let body = emit_fragment(&sh).unwrap();
    validate(&body);
}
