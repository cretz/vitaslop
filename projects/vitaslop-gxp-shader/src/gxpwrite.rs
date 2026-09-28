//! Build a real `SceGxmProgram` container - the inverse of [`crate::container::Program::parse`].
//!
//! # Why a WRITER
//!
//! Every blob this project has ever looked at was captured from a title. That makes the corpus
//! an accident of what seven compilers happened to emit: a container field nothing ships is
//! never exercised, and a program shape nothing ships cannot be tested at all. Worse, a captured
//! blob states no INTENT - so the corpus differential can only ask whether our emitter and our
//! reference agree, never whether either is right.
//!
//! With a writer, a test can say "here is a program that multiplies its input by two" and then
//! require the whole shipped path - parse, decode, link, emit, run on the GPU - to produce
//! exactly that. What the assembled blob asserts is the pipeline's fidelity to a stated meaning,
//! which is the link the corpus cannot supply.
//!
//! # What it does NOT establish
//!
//! The blob is written through THIS project's reading of the container format, so a field we
//! read wrong we also write wrong, and the round trip agrees with itself. It is a check on
//! everything downstream of the container - the decoder, the linker, the emitter, the GPU - and
//! not on the container reading itself. Only a blob a real `SceShaccCg` produced, or one a real
//! Vita executes, can do that; see the conformance app.
//!
//! # Layout
//!
//! Every table is placed BEFORE the code and the code is last, because the parser recovers the
//! code's extent as `[asm_offset .. min(literal_table, parameter_table, end))`. With the tables
//! ahead of it, that minimum is the end of the blob and the instruction stream is exactly what
//! was written - an ordering property, not a convention, and one a blob that violated it would
//! silently truncate.

use crate::container::{ParamCategory, ParamType, ProgramKind};

/// One entry of the parameter table.
#[derive(Debug, Clone)]
pub struct ParamSpec {
    pub name: String,
    pub category: ParamCategory,
    pub ptype: ParamType,
    /// Components (1..4).
    pub components: u8,
    /// Which container this parameter's storage belongs to (14 = the default uniform buffer).
    pub container_index: u8,
    pub array_size: u32,
    /// For an attribute the PA register it loads into; for a uniform its offset in the default
    /// uniform buffer; for a sampler the texture UNIT.
    pub resource_index: i32,
    pub semantic: u8,
    pub semantic_index: u8,
    pub sampler_cube: bool,
}

impl ParamSpec {
    /// A vertex input attribute at PA register `pa`.
    pub fn attribute(name: &str, pa: i32, components: u8) -> ParamSpec {
        ParamSpec {
            name: name.to_string(),
            category: ParamCategory::Attribute,
            ptype: ParamType::F32,
            components,
            container_index: 0,
            array_size: 1,
            resource_index: pa,
            semantic: 0,
            semantic_index: 0,
            sampler_cube: false,
        }
    }

    /// A default-uniform-buffer entry at four-byte-register offset `offset`.
    pub fn uniform(name: &str, offset: i32, components: u8) -> ParamSpec {
        ParamSpec {
            name: name.to_string(),
            category: ParamCategory::Uniform,
            ptype: ParamType::F32,
            components,
            container_index: 14,
            array_size: 1,
            resource_index: offset,
            semantic: 0,
            semantic_index: 0,
            sampler_cube: false,
        }
    }

    /// A texture sampler bound to `unit`.
    pub fn sampler(name: &str, unit: i32) -> ParamSpec {
        ParamSpec {
            name: name.to_string(),
            category: ParamCategory::Sampler,
            ptype: ParamType::F32,
            components: 4,
            container_index: 0,
            array_size: 1,
            resource_index: unit,
            semantic: 0,
            semantic_index: 0,
            sampler_cube: false,
        }
    }

    /// A non-default UNIFORM BUFFER at GXM index `index`, `bytes` long - the parameter a program
    /// that chases the buffer's pointer with memory loads declares. `array_size` is the byte
    /// count, which is what the window resolution reads as the buffer's extent.
    pub fn uniform_buffer(name: &str, index: i32, bytes: u32) -> ParamSpec {
        ParamSpec {
            name: name.to_string(),
            category: ParamCategory::UniformBuffer,
            ptype: ParamType::Aggregate,
            components: 1,
            container_index: 0,
            array_size: bytes,
            resource_index: index,
            semantic: 0,
            semantic_index: 0,
            sampler_cube: false,
        }
    }

    /// This parameter with an array size (what an indexed uniform READ addresses into).
    pub fn array(mut self, n: u32) -> ParamSpec {
        self.array_size = n;
        self
    }

    /// This parameter with a semantic and semantic index - the pair the vertex output order is
    /// established from when the varyings block cannot state it.
    pub fn with_semantic(mut self, semantic: u8, index: u8) -> ParamSpec {
        self.semantic = semantic;
        self.semantic_index = index;
        self
    }

    /// The packed `type` word of the parameter record.
    fn packed(&self) -> u32 {
        let category = match self.category {
            ParamCategory::Attribute => 0u32,
            ParamCategory::Uniform => 1,
            ParamCategory::Sampler => 2,
            ParamCategory::AuxSurface => 3,
            ParamCategory::UniformBuffer => 4,
            ParamCategory::Unknown(v) => u32::from(v),
        };
        let ptype = match self.ptype {
            ParamType::F32 => 0u32,
            ParamType::F16 => 1,
            ParamType::C10 => 2,
            ParamType::U32 => 3,
            ParamType::S32 => 4,
            ParamType::U16 => 5,
            ParamType::S16 => 6,
            ParamType::U8 => 7,
            ParamType::S8 => 8,
            ParamType::Aggregate => 9,
            ParamType::Unknown(v) => u32::from(v),
        };
        (category & 0xf)
            | ((ptype & 0xf) << 4)
            | ((u32::from(self.components) & 0xf) << 8)
            | ((u32::from(self.container_index) & 0xf) << 12)
            | ((u32::from(self.semantic)) << 16)
            | ((u32::from(self.semantic_index)) << 24)
            | if self.sampler_cube { 0x1000_0000 } else { 0 }
    }
}

/// One entry of the container table: which SA registers a storage block occupies.
#[derive(Debug, Clone, Copy)]
pub struct ContainerSpec {
    /// 0..13 ordinary uniform buffers, 14 DEFAULT uniform, 15 TEXTURE, 16 LITERAL, 19 DATA.
    pub index: u16,
    pub base_sa: u16,
    pub size_regs: u16,
}

/// What a VERTEX program declares it writes. Clip position is always present (a block without
/// it is refused by the parser, and rightly - it is what the first four output lanes ARE).
#[derive(Debug, Clone, Default)]
pub struct VertexOutputs {
    /// Declare a four-lane COLOR0 output.
    pub color0: bool,
    /// `(texcoord number, component count)` pairs, ascending.
    pub texcoords: Vec<(u8, u32)>,
}

/// One fragment interpolant declaration.
#[derive(Debug, Clone, Copy)]
pub struct InterpolantSpec {
    /// Texcoord number 0..9, or [`InterpolantSpec::COLOR0`].
    pub usage: u32,
    /// PA registers of interpolated data (1..4).
    pub registers: u8,
    /// F16 interpolation (two components per PA register) rather than F32.
    pub half: bool,
}

impl InterpolantSpec {
    /// The `attribute_info` usage nibble naming COLOR0.
    pub const COLOR0: u32 = 0xa;

    /// A texcoord interpolant spanning `registers` PA registers.
    pub fn texcoord(number: u8, registers: u8) -> InterpolantSpec {
        InterpolantSpec { usage: u32::from(number), registers, half: false }
    }

    /// The four-lane COLOR0 interpolant.
    pub fn color0() -> InterpolantSpec {
        InterpolantSpec { usage: InterpolantSpec::COLOR0, registers: 4, half: false }
    }

    /// This interpolant at F16 precision - two components to a PA register, which is how a
    /// real fragment program carries most of what it reads.
    pub fn at_f16(mut self) -> InterpolantSpec {
        self.half = true;
        self
    }

    /// The descriptor's `attribute_info` word.
    ///
    /// Its low byte is the PREFETCH SOURCE, and a descriptor with no prefetch must name
    /// [`PREFETCH_SOURCE_NONE`] there - the parser cross-checks the two independent prefetch
    /// statements and refuses the whole list if they disagree. Writing a zero here (the
    /// obvious-looking "nothing set") is not "no prefetch", it is prefetch source 0.
    fn attribute_info(&self) -> u32 {
        ((self.usage & 0xf) << 12) | PREFETCH_SOURCE_NONE | if self.half { 0x2000_0000 } else { 0 }
    }

    fn size_word(&self) -> u32 {
        (u32::from(self.registers.saturating_sub(1)) & 0x3) << 4
    }
}

/// A complete program to assemble into a container.
#[derive(Debug, Clone)]
pub struct ProgramSpec {
    pub kind: ProgramKind,
    /// The primary instruction stream.
    pub code: Vec<u64>,
    /// The SECONDARY stream, which runs once per draw and fills the SA bank.
    pub secondary_code: Vec<u64>,
    pub parameters: Vec<ParamSpec>,
    /// `(sa_register_index_within_the_literal_container, value_bits)`.
    pub literals: Vec<(u32, u32)>,
    /// `(index_within_the_DATA_container, texture_unit)` - the table a `SMP`'s sampler operand
    /// is resolved through.
    ///
    /// A sampler instruction does NOT name its texture unit: it names an SA register, and this
    /// table is what says which unit's control words live there. A program with a sample and no
    /// entry here is refused by the emitter, by name - which is correct, and is why a
    /// conformance case with a sample must declare one.
    pub texture_control: Vec<(u32, u32)>,
    pub containers: Vec<ContainerSpec>,
    /// How many SA registers the DEFAULT uniform buffer occupies.
    pub default_uniform_regs: u32,
    pub primary_reg_count: u16,
    pub secondary_reg_count: u16,
    pub temp_reg_count: u16,
    /// Vertex only.
    pub outputs: VertexOutputs,
    /// Fragment only.
    pub interpolants: Vec<InterpolantSpec>,
    /// The +0x78 table: `(buffer_index, data_slot)` - the DATA-container slot whose SA register
    /// the driver writes that buffer's bound guest ADDRESS into, for a program that loads
    /// through it.
    pub ub_bindings: Vec<(u16, u16)>,
}

impl ProgramSpec {
    /// A vertex program with the given code, nothing else declared.
    pub fn vertex(code: Vec<u64>) -> ProgramSpec {
        ProgramSpec {
            kind: ProgramKind::Vertex,
            code,
            secondary_code: Vec::new(),
            parameters: Vec::new(),
            literals: Vec::new(),
            texture_control: Vec::new(),
            containers: Vec::new(),
            default_uniform_regs: 0,
            primary_reg_count: 0,
            secondary_reg_count: 0,
            temp_reg_count: 8,
            outputs: VertexOutputs::default(),
            interpolants: Vec::new(),
            ub_bindings: Vec::new(),
        }
    }

    /// A fragment program with the given code.
    pub fn fragment(code: Vec<u64>) -> ProgramSpec {
        ProgramSpec { kind: ProgramKind::Fragment, ..ProgramSpec::vertex(code) }
    }

    pub fn with_parameters(mut self, params: Vec<ParamSpec>) -> ProgramSpec {
        self.parameters = params;
        self
    }

    pub fn with_literals(mut self, literals: Vec<(u32, u32)>) -> ProgramSpec {
        self.literals = literals;
        self
    }

    pub fn with_texture_control(mut self, entries: Vec<(u32, u32)>) -> ProgramSpec {
        self.texture_control = entries;
        self
    }

    pub fn with_containers(mut self, containers: Vec<ContainerSpec>) -> ProgramSpec {
        self.containers = containers;
        self
    }

    pub fn with_default_uniform_regs(mut self, n: u32) -> ProgramSpec {
        self.default_uniform_regs = n;
        self
    }

    pub fn with_registers(mut self, primary: u16, secondary: u16, temp: u16) -> ProgramSpec {
        self.primary_reg_count = primary;
        self.secondary_reg_count = secondary;
        self.temp_reg_count = temp;
        self
    }

    pub fn with_outputs(mut self, outputs: VertexOutputs) -> ProgramSpec {
        self.outputs = outputs;
        self
    }

    pub fn with_interpolants(mut self, interpolants: Vec<InterpolantSpec>) -> ProgramSpec {
        self.interpolants = interpolants;
        self
    }

    pub fn with_secondary_code(mut self, code: Vec<u64>) -> ProgramSpec {
        self.secondary_code = code;
        self
    }

    pub fn with_ub_bindings(mut self, bindings: Vec<(u16, u16)>) -> ProgramSpec {
        self.ub_bindings = bindings;
        self
    }
}

// Header field offsets. These are the SAME constants `container.rs` reads; they are restated
// here rather than shared because a writer that imported the reader's private table could not
// catch a field the reader places wrong - the two agreeing would then be circular. As it is, a
// disagreement shows up as a failed round trip in `a_written_container_parses_back_to_itself`.
const OFF_MAGIC: usize = 0x00;
const OFF_MAJOR: usize = 0x04;
const OFF_MINOR: usize = 0x05;
const OFF_SIZE: usize = 0x08;
const OFF_TYPE: usize = 0x14;
const OFF_PARAM_COUNT: usize = 0x24;
const OFF_PARAMS_OFFSET: usize = 0x28;
const OFF_VARYINGS_OFFSET: usize = 0x2c;
const OFF_PRIMARY_REG: usize = 0x30;
const OFF_SECONDARY_REG: usize = 0x32;
const OFF_TEMP1: usize = 0x34;
const OFF_TEMP2: usize = 0x38;
const OFF_ASM_OFFSET: usize = 0x40;
const OFF_SECONDARY_COUNT: usize = 0x44;
const OFF_SECONDARY_OFFSET: usize = 0x48;
const OFF_SECONDARY_END_OFFSET: usize = 0x4c;
const OFF_DEFAULT_UNIFORM_REGS: usize = 0x64;
const OFF_LITERAL_COUNT: usize = 0x70;
const OFF_LITERAL_OFFSET: usize = 0x74;
const OFF_UB_BINDING_COUNT: usize = 0x78;
const OFF_UB_BINDING_OFFSET: usize = 0x7c;
const OFF_TEXTURE_COUNT: usize = 0x80;
const OFF_TEXTURE_OFFSET: usize = 0x84;
const OFF_CONTAINER_COUNT: usize = 0x90;
const OFF_CONTAINER_OFFSET: usize = 0x94;

/// Bytes of fixed header before the first table. 0x98 is the last field's end; the blocks that
/// follow start at an eight-byte boundary past it.
const HEADER_BYTES: usize = 0xa0;

/// Bit 0x1000 of a vertex varyings block's first output word: a clip POSITION is present. The
/// parser refuses a block without it, because the first four output lanes are the position and
/// placing varyings from lane 4 in a block that does not declare one puts every one of them in
/// the wrong register.
const POSITION_PRESENT_BIT: u32 = 0x1000;
/// Bit 0x0800: a four-lane COLOR0 output.
const COLOR0_PRESENT_BIT: u32 = 0x0800;

/// The value a fragment varying descriptor's prefetch-SOURCE byte carries when the descriptor
/// has no PDS-prefetched sample riding along. Not zero - zero is source 0.
const PREFETCH_SOURCE_NONE: u32 = 0x0f;

/// Assemble `spec` into `SceGxmProgram` bytes.
///
/// The result is what [`crate::container::Program::parse`] reads back and what
/// [`crate::recompile_vertex`] / [`crate::recompile_fragment`] compile - the same entry points
/// the shipped renderer calls on a captured blob, with nothing test-only in the path.
pub fn write(spec: &ProgramSpec) -> Vec<u8> {
    let mut out = vec![0u8; HEADER_BYTES];

    // ---- varyings block ----
    let varyings_at = out.len();
    match spec.kind {
        ProgramKind::Vertex => {
            let mut vo1 = POSITION_PRESENT_BIT;
            let mut lanes = 4u32; // the clip position's own four
            if spec.outputs.color0 {
                vo1 |= COLOR0_PRESENT_BIT;
                lanes += 4;
            }
            let mut vo2 = 0u32;
            for &(k, components) in &spec.outputs.texcoords {
                // The three-bit field decodes to a lane count as `(v & 1) * 2 + bit1 + bit2`,
                // so the encoder inverts that rather than restating a width table: 1 -> 2
                // lanes, 2 -> 1, 3 -> 3, 7 -> 4.
                let v = match components {
                    1 => 2u32,
                    2 => 1,
                    3 => 3,
                    4 => 7,
                    other => panic!("a texcoord cannot carry {other} components"),
                };
                vo2 |= v << (u32::from(k) * 3);
                lanes += components;
            }
            // The block's own total output-lane count, which the parser validates the decoded
            // texcoord widths against.
            vo1 |= lanes << 24;
            out.resize(varyings_at + 0x18, 0);
            put_u32(&mut out, varyings_at + 0x10, vo1);
            put_u32(&mut out, varyings_at + 0x14, vo2);
        }
        ProgramKind::Fragment => {
            // A fixed block header, then a self-relative pointer at +0x10 to the descriptor
            // array, which is placed immediately after it.
            out.resize(varyings_at + 0x18, 0);
            put_u16(&mut out, varyings_at + 0x0c, spec.interpolants.len() as u16);
            let arr_field = varyings_at + 0x10;
            let arr = out.len();
            put_u32(&mut out, arr_field, (arr - arr_field) as u32);
            for it in &spec.interpolants {
                let d = out.len();
                out.resize(d + 16, 0);
                put_u32(&mut out, d, it.attribute_info());
                // `resource_index` names the prefetched texture UNIT, and must be zero on a
                // descriptor with no prefetch. It is NOT the texcoord number - that is in the
                // usage nibble of `attribute_info`.
                put_u32(&mut out, d + 4, 0);
                put_u32(&mut out, d + 8, it.size_word());
                put_u32(&mut out, d + 12, 0);
            }
        }
    }
    put_u32(&mut out, OFF_VARYINGS_OFFSET, (varyings_at - OFF_VARYINGS_OFFSET) as u32);

    // ---- container table ----
    let containers_at = out.len();
    for c in &spec.containers {
        let e = out.len();
        out.resize(e + 8, 0);
        put_u16(&mut out, e, c.index);
        put_u16(&mut out, e + 4, c.base_sa);
        put_u16(&mut out, e + 6, c.size_regs);
    }
    put_u32(&mut out, OFF_CONTAINER_COUNT, spec.containers.len() as u32);
    put_u32(&mut out, OFF_CONTAINER_OFFSET, (containers_at - OFF_CONTAINER_OFFSET) as u32);

    // ---- literal table ----
    let literals_at = out.len();
    for &(index, value) in &spec.literals {
        let e = out.len();
        out.resize(e + 8, 0);
        put_u32(&mut out, e, index);
        put_u32(&mut out, e + 4, value);
    }
    put_u32(&mut out, OFF_LITERAL_COUNT, spec.literals.len() as u32);
    put_u32(&mut out, OFF_LITERAL_OFFSET, (literals_at - OFF_LITERAL_OFFSET) as u32);

    // ---- texture-control table ----
    //
    // FOUR words per texture, of which only word 0 names the base register; the other three are
    // structural and are skipped on the way back in (`e & 3 != 0`). Writing only word 0 would
    // parse identically today, and writing all four is what a real container does - so the
    // reader's skip is exercised rather than merely present.
    let textures_at = out.len();
    for &(index, unit) in &spec.texture_control {
        for word in 0..4u32 {
            let e = out.len();
            out.resize(e + 4, 0);
            put_u32(&mut out, e, (index << 16) | (unit << 2) | word);
        }
    }
    put_u32(&mut out, OFF_TEXTURE_COUNT, (spec.texture_control.len() * 4) as u32);
    put_u32(&mut out, OFF_TEXTURE_OFFSET, (textures_at - OFF_TEXTURE_OFFSET) as u32);

    // ---- uniform-buffer binding table ----
    //
    // EIGHT-byte entries - buffer index, DATA slot, four structural bytes - the stride the reader
    // settled on a three-entry table. An empty table still points somewhere inside the blob.
    let bindings_at = out.len();
    for &(buffer_index, data_slot) in &spec.ub_bindings {
        let e = out.len();
        out.resize(e + 8, 0);
        out[e..e + 2].copy_from_slice(&buffer_index.to_le_bytes());
        out[e + 2..e + 4].copy_from_slice(&data_slot.to_le_bytes());
    }
    put_u32(&mut out, OFF_UB_BINDING_COUNT, spec.ub_bindings.len() as u32);
    put_u32(&mut out, OFF_UB_BINDING_OFFSET, (bindings_at - OFF_UB_BINDING_OFFSET) as u32);

    // ---- parameter table, then the name strings it points at ----
    let params_at = out.len();
    out.resize(params_at + spec.parameters.len() * 16, 0);
    let mut names_at = out.len();
    for (i, p) in spec.parameters.iter().enumerate() {
        let e = params_at + i * 16;
        // The name offset is SELF-RELATIVE to the parameter's own entry address.
        put_u32(&mut out, e, (names_at - e) as u32);
        put_u32(&mut out, e + 4, p.packed());
        put_u32(&mut out, e + 8, p.array_size.max(1));
        put_u32(&mut out, e + 12, p.resource_index as u32);
        out.extend_from_slice(p.name.as_bytes());
        out.push(0);
        names_at = out.len();
    }
    put_u32(&mut out, OFF_PARAM_COUNT, spec.parameters.len() as u32);
    put_u32(&mut out, OFF_PARAMS_OFFSET, (params_at - OFF_PARAMS_OFFSET) as u32);

    // ---- secondary stream, then the primary one LAST ----
    while !out.len().is_multiple_of(8) {
        out.push(0);
    }
    let secondary_at = out.len();
    for w in &spec.secondary_code {
        out.extend_from_slice(&w.to_le_bytes());
    }
    let secondary_end = out.len();
    put_u32(&mut out, OFF_SECONDARY_COUNT, spec.secondary_code.len() as u32);
    put_u32(&mut out, OFF_SECONDARY_OFFSET, (secondary_at - OFF_SECONDARY_OFFSET) as u32);
    put_u32(&mut out, OFF_SECONDARY_END_OFFSET, (secondary_end - OFF_SECONDARY_END_OFFSET) as u32);

    // The primary code is LAST so the parser's `min(literal, params, end)` extent is the end of
    // the blob. A table placed after it would truncate the stream with no error anywhere.
    let asm_at = out.len();
    for w in &spec.code {
        out.extend_from_slice(&w.to_le_bytes());
    }
    put_u32(&mut out, OFF_ASM_OFFSET, (asm_at - OFF_ASM_OFFSET) as u32);

    // ---- the fixed header fields ----
    put_u32(&mut out, OFF_MAGIC, 0x0050_5847); // "GXP\0"
    out[OFF_MAJOR] = 1;
    out[OFF_MINOR] = 4;
    put_u32(&mut out, OFF_TYPE, u32::from(spec.kind == ProgramKind::Fragment));
    put_u16(&mut out, OFF_PRIMARY_REG, spec.primary_reg_count);
    put_u16(&mut out, OFF_SECONDARY_REG, spec.secondary_reg_count);
    put_u16(&mut out, OFF_TEMP1, spec.temp_reg_count);
    put_u16(&mut out, OFF_TEMP2, spec.temp_reg_count);
    put_u32(&mut out, OFF_DEFAULT_UNIFORM_REGS, spec.default_uniform_regs);
    let size = out.len() as u32;
    put_u32(&mut out, OFF_SIZE, size);
    out
}

fn put_u32(buf: &mut [u8], at: usize, v: u32) {
    buf[at..at + 4].copy_from_slice(&v.to_le_bytes());
}

fn put_u16(buf: &mut [u8], at: usize, v: u16) {
    buf[at..at + 2].copy_from_slice(&v.to_le_bytes());
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::container::{Program, VaryingUsage};
    use crate::ir::Bank;
    use crate::usse::asm::{self, Dest, Src};

    /// Everything the writer states must come back out of the READER, field for field. This is
    /// the test that makes the writer usable as a test fixture at all: a blob whose parameter
    /// table or whose code extent came back different would silently build every later case on
    /// a program other than the one written.
    #[test]
    fn a_written_container_parses_back_to_itself() {
        let code = vec![
            asm::mov(false, Dest::new(Bank::Output, 0), [true; 4], Src::reg(Bank::PrimaryAttr, 0))
                .unwrap(),
            asm::alu(
                crate::ir::Op::Mul,
                false,
                Dest::new(Bank::Output, 4),
                [true, true, false, false],
                Src::reg(Bank::PrimaryAttr, 4),
                Src::reg(Bank::SecondaryAttr, 0),
            )
            .unwrap(),
        ];
        let spec = ProgramSpec::vertex(code.clone())
            .with_parameters(vec![
                ParamSpec::attribute("IN.position", 0, 4),
                ParamSpec::attribute("IN.uv", 4, 2),
                ParamSpec::uniform("scale", 0, 4).array(3),
            ])
            .with_literals(vec![(0, 0x3f80_0000), (1, 0x4000_0000)])
            .with_containers(vec![
                ContainerSpec { index: 14, base_sa: 0, size_regs: 12 },
                ContainerSpec { index: 16, base_sa: 12, size_regs: 2 },
                ContainerSpec { index: 19, base_sa: 14, size_regs: 2 },
            ])
            .with_default_uniform_regs(12)
            .with_registers(8, 16, 8)
            .with_outputs(VertexOutputs { color0: true, texcoords: vec![(0, 2)] });

        let bytes = write(&spec);
        let p = Program::parse(&bytes).expect("the written blob parses");

        assert_eq!(p.kind, ProgramKind::Vertex);
        assert_eq!(p.size as usize, bytes.len(), "the size field states the whole blob");
        assert_eq!(p.code, code, "the code extent is exactly the stream written");
        assert_eq!(p.primary_reg_count, 8);
        assert_eq!(p.secondary_reg_count, 16);
        assert_eq!(p.temp_reg_count, 8);
        assert_eq!(p.default_uniform_regs, 12);
        assert_eq!(p.containers.len(), 3);
        // The literal table's index is relative to the LITERAL container's own base, so a
        // literal written at index 0 lands at that base. Reading this back proves the two
        // tables were placed against each other correctly, which is what the base field is for.
        assert_eq!(p.literals, vec![(12, 0x3f80_0000), (13, 0x4000_0000)]);
        assert!(p.sa_base_from_container, "both containers were declared, so the base is stored");

        let names: Vec<&str> = p.parameters.iter().map(|x| x.name.as_str()).collect();
        assert_eq!(names, ["IN.position", "IN.uv", "scale"]);
        assert_eq!(p.parameters[0].category, ParamCategory::Attribute);
        assert_eq!(p.parameters[0].component_count, 4);
        assert_eq!(p.parameters[1].resource_index, 4);
        assert_eq!(p.parameters[2].category, ParamCategory::Uniform);
        assert_eq!(p.parameters[2].array_size, 3);
        assert_eq!(p.parameters[2].container_index, 14);

        assert_eq!(p.varyings_error, None, "the varyings block decoded");
        let usages: Vec<VaryingUsage> = p.output_varyings.iter().map(|v| v.usage).collect();
        assert!(usages.contains(&VaryingUsage::TexCoord(0)));
    }

    /// The FRAGMENT half: interpolants come back with the PA spans and precisions declared.
    #[test]
    fn a_written_fragment_container_declares_its_interpolants() {
        let code = vec![asm::mov(
            false,
            Dest::new(Bank::Output, 0),
            [true; 4],
            Src::reg(Bank::PrimaryAttr, 0),
        )
        .unwrap()];
        let spec = ProgramSpec::fragment(code)
            .with_interpolants(vec![
                InterpolantSpec::texcoord(0, 4),
                InterpolantSpec::texcoord(1, 1).at_f16(),
            ])
            .with_registers(5, 0, 4);

        let bytes = write(&spec);
        let p = Program::parse(&bytes).expect("the written fragment blob parses");
        assert_eq!(p.kind, ProgramKind::Fragment);
        assert_eq!(p.varyings_error, None);
        assert_eq!(p.interpolants.len(), 2);
        assert_eq!(p.interpolants[0].usage, VaryingUsage::TexCoord(0));
        assert_eq!(p.interpolants[0].pa_base, 0);
        assert_eq!(p.interpolants[0].register_count, 4);
        assert!(!p.interpolants[0].half);
        // The PA base ACCUMULATES across the array in declaration order - there is no explicit
        // base field - so the second interpolant starts where the first one ended.
        assert_eq!(p.interpolants[1].pa_base, 4);
        assert!(p.interpolants[1].half);
    }

    /// A SECONDARY stream states its extent three times over (a count and two offsets) and the
    /// parser refuses a blob where they disagree. The writer must satisfy all three.
    #[test]
    fn a_written_secondary_stream_satisfies_all_three_of_its_statements() {
        let secondary = vec![
            asm::mov(
                false,
                Dest::new(Bank::Temp, 0),
                [true; 4],
                Src::reg(Bank::SecondaryAttr, 0),
            )
            .unwrap(),
            asm::mov(
                false,
                Dest::new(Bank::Temp, 4),
                [true; 4],
                Src::reg(Bank::SecondaryAttr, 4),
            )
            .unwrap(),
        ];
        let spec = ProgramSpec::vertex(vec![asm::mov(
            false,
            Dest::new(Bank::Output, 0),
            [true; 4],
            Src::reg(Bank::Temp, 0),
        )
        .unwrap()])
        .with_secondary_code(secondary.clone());

        let p = Program::parse(&write(&spec)).expect("parses");
        assert_eq!(p.secondary_code, secondary);
    }
}
