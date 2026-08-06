//! GXP container parsing: the `SceGxmProgram` header, its parameter table, and the
//! location of the USSE instruction stream.
//!
//! The layout below is transcribed from the public `SceGxmProgram` structure definitions
//! and the vitasdk `gxm.h` enums, and every offset here is verified empirically against the
//! 43 captured `.gxp` blobs (magic, version 1.4, `header.size == file len`, and
//! `asm_abs < literal_abs <= params_abs` with the code region a whole number of 64-bit
//! instructions). No game bytes ship in this crate; only the parser does.
//!
//! GOTCHA that governs the whole format: every *_offset field is SELF-RELATIVE - the
//! absolute position is `field_address + stored_value`, not `blob_start + value`.

/// "GXP\0" little-endian.
pub const GXP_MAGIC: u32 = 0x0050_5847;

/// Whether the program is a vertex or fragment shader. The discriminator is bit 0 of
/// the byte at header offset 0x14 (`SceGxmProgramType`: 0 = vertex, 1 = fragment),
/// verified against the dumps (every `vert_*` blob has 0x00 there, every `frag_*` has
/// an odd value).
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum ProgramKind {
    Vertex,
    Fragment,
}

/// `SceGxmParameterCategory` (from `SceGxmTypes.h`). Selects what a parameter binds.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum ParamCategory {
    /// Vertex input attribute; `resource_index` is the PA register it loads into.
    Attribute,
    /// Uniform constant; `resource_index` is a 4-byte-register offset into the default
    /// uniform buffer (the SA bank).
    Uniform,
    /// Texture sampler; `resource_index` is the texture UNIT number.
    Sampler,
    /// Auxiliary surface (untested in the wild here).
    AuxSurface,
    /// Uniform buffer binding.
    UniformBuffer,
    /// Any category value the RE'd enum does not name.
    Unknown(u8),
}

impl ParamCategory {
    fn from_bits(v: u8) -> Self {
        match v {
            0 => ParamCategory::Attribute,
            1 => ParamCategory::Uniform,
            2 => ParamCategory::Sampler,
            3 => ParamCategory::AuxSurface,
            4 => ParamCategory::UniformBuffer,
            other => ParamCategory::Unknown(other),
        }
    }
}

/// `SceGxmParameterType` (from `SceGxmTypes.h`): the scalar component type of a uniform
/// or attribute. Fragment default-uniform packing depends on this (F16 = 2 bytes/comp,
/// F32 = 4 bytes/comp).
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum ParamType {
    F32,
    F16,
    C10,
    U32,
    S32,
    U16,
    S16,
    U8,
    S8,
    Aggregate,
    Unknown(u8),
}

impl ParamType {
    /// Decode the parameter record's type nibble. Public because the runtime decodes the SAME
    /// nibble straight out of guest memory: `sceGxmSetUniformDataF` is handed a pointer to one
    /// of these records and must know how wide a component is before it can write one - an F16
    /// uniform packs two components per register, and assuming four bytes each puts every
    /// component after the first at the wrong offset.
    pub fn from_bits(v: u8) -> Self {
        match v {
            0 => ParamType::F32,
            1 => ParamType::F16,
            2 => ParamType::C10,
            3 => ParamType::U32,
            4 => ParamType::S32,
            5 => ParamType::U16,
            6 => ParamType::S16,
            7 => ParamType::U8,
            8 => ParamType::S8,
            9 => ParamType::Aggregate,
            other => ParamType::Unknown(other),
        }
    }

    /// Bytes each scalar component occupies in the default uniform buffer (the packing
    /// the fragment material reflection already relies on). `None` for types whose
    /// width is not a fixed uniform-buffer scalar.
    pub fn component_bytes(self) -> Option<u32> {
        Some(match self {
            ParamType::F32 | ParamType::U32 | ParamType::S32 => 4,
            ParamType::F16 | ParamType::U16 | ParamType::S16 => 2,
            ParamType::U8 | ParamType::S8 => 1,
            ParamType::C10 => 2, // 10-bit fixed, stored 2 bytes here
            ParamType::Aggregate | ParamType::Unknown(_) => return None,
        })
    }
}

/// Bit of a `SceGxmProgramParameter`'s packed `type` word marking a SAMPLER as a CUBE map
/// (`sceGxmProgramParameterIsSamplerCube`). ESTABLISHED, not assumed: across every captured
/// fragment blob this bit is set on exactly two samplers - and independently, those are exactly
/// the two whose guest-bound `SceGxmTexture` carries `SCE_GXM_TEXTURE_CUBE` as its type. It is
/// clear on all seventeen others, including a one-coordinate fog lookup table, so neither the
/// coordinate count nor the sampler name can substitute for it.
const SAMPLER_CUBE_BIT: u32 = 0x1000_0000;

/// Bit of a fragment varying descriptor's `size` word marking that a PDS-PREFETCHED texture
/// sample rides along with this varying: two PA registers beyond its interpolated data hold the
/// sample's four F16 components, so the next interpolant's base is that much higher.
///
/// A PowerVR SGX fragment program does not have to issue a texture read whose coordinate is a
/// plain interpolated texcoord - the PDS can fetch it before the shader starts and leave the
/// result sitting in the primary-attribute bank, which is why such samples never appear as SMP
/// instructions in the instruction stream. See [`SamplePrefetch`] for the fields that describe
/// one, and [`INFO_PREFETCH`] / [`INFO_PREFETCH_LAST`] for the flags that confirm the decode.
///
/// `size` bit 6 is NOT one of them, and used to be treated as one. MEASURED over a 314-blob
/// corpus, tabulating every descriptor by these bits: `attribute_info & 0x100` and
/// `component_info & 0x20` are equal on EVERY descriptor - all thirty distinct combinations -
/// while `size & 0x40` differs from them on fourteen, every one of which is a genuine prefetch
/// (it names a texture unit and a source texcoord). Requiring all three to agree therefore
/// threw away those fourteen programs' entire interpolant lists, and with them a title's whole
/// post-process chain: sixteen of its seventeen draws fell back to fixed-function and its
/// bloom/tonemap targets came out black.
///
/// What `size` bit 6 does mean is NOT established. It co-occurs exactly with
/// `component_info == 0x70` (rather than `0x20`), so the two are one fact seen twice, but
/// nothing here depends on knowing which fact - so it is read as an independent field and left
/// alone rather than given a meaning it has not earned.
const _SIZE_BIT6_NOT_A_PREFETCH_FLAG: u32 = 0x40;

/// `component_info` bit a fragment varying descriptor carries exactly when it declares a
/// prefetched sample - the redundant statement of [`INFO_PREFETCH`]. The two are cross-checked
/// on parse; a program where they disagree is not decoded at all.
const COMPONENT_INFO_PREFETCH: u32 = 0x20;

/// `attribute_info` bit marking a descriptor that declares a prefetched sample.
const INFO_PREFETCH: u32 = 0x0000_0100;

/// `attribute_info` bit marking the LAST prefetched sample in the program's descriptor array
/// (the end of the PDS fetch sequence). Set on exactly one descriptor per program that has any
/// prefetch, and always the highest-indexed one - checked on all 33 prefetches in the corpus.
/// It carries no layout information; it is decoded only as a consistency check.
const INFO_PREFETCH_LAST: u32 = 0x0000_0800;

/// `attribute_info` low byte: the TEXCOORD index whose interpolated value is the prefetched
/// sample's coordinate, or [`PREFETCH_SOURCE_NONE`] when the descriptor has no prefetch.
const INFO_PREFETCH_SOURCE: u32 = 0x0000_00ff;

/// The `attribute_info` low-byte value meaning "no prefetch source". Not a usable TEXCOORD index
/// (the pipeline has TEXCOORD0..9), so it cannot be confused with one.
const PREFETCH_SOURCE_NONE: u32 = 0x0f;

/// One entry of the program's parameter table (16 bytes on disk).
#[derive(Debug, Clone, PartialEq, Eq)]
pub struct Parameter {
    pub name: String,
    pub category: ParamCategory,
    pub ptype: ParamType,
    /// Number of scalar components (float2/float3/float4 -> 2/3/4). At least 1.
    pub component_count: u8,
    pub container_index: u8,
    /// SAMPLER parameters only: this sampler is a CUBE map, so its three sample coordinates
    /// are a direction rather than a volume position. See [`SAMPLER_CUBE_BIT`].
    pub sampler_cube: bool,
    /// Array length (>= 1).
    pub array_size: u32,
    /// Register number: PA reg for attributes, SA-bank 4-byte offset for uniforms, the
    /// texture UNIT for samplers. Signed on disk.
    pub resource_index: i32,
}

/// Highest TEXCOORD index the GXM varying interface carries (TEXCOORD0..9), which is also the
/// number of 3-bit width fields the vertex varyings block's `vertex_outputs2` word holds.
pub const MAX_TEXCOORD: u8 = 9;

/// A vertex-output / fragment-input varying usage, in the fixed canonical order the GXM
/// pipeline links by (position, colours, fog, texcoords). Linkage is POSITIONAL by this
/// usage id - the m-th usage present in both the vertex and fragment program occupies
/// matching relative register offsets - never by name string.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum VaryingUsage {
    Position,
    Color0,
    Color1,
    Fog,
    /// TEXCOORD0..[`MAX_TEXCOORD`].
    TexCoord(u8),
    /// A usage the decode did not recognise. Carries the WHOLE raw `attribute_info` word,
    /// not just the semantic nibble: the nibble alone cannot say whether the input is one
    /// the rasteriser generates (bit 0x40000000) or a vertex output under a semantic we do
    /// not map, and those want opposite treatment - synthesise it, or refuse to link.
    Unknown(u32),
}

/// A texture sample the PDS performs BEFORE the fragment program runs, leaving its result in
/// the primary-attribute bank. The shader then reads those registers like any other input, so
/// nothing in the instruction stream reveals that a sample happened - the varying descriptor is
/// the only record of it.
///
/// ESTABLISHED, not assumed. Across every captured fragment blob, a descriptor carrying
/// [`SIZE_PREFETCH`] also carries [`INFO_PREFETCH`] and [`COMPONENT_INFO_PREFETCH`]; its
/// `resource_index` is always one of that program's own declared SAMPLER units (and is always 0
/// on descriptors without the flag); and its `attribute_info` low byte is always a TEXCOORD
/// index 0..9 (and always [`PREFETCH_SOURCE_NONE`] without the flag). The semantics corroborate
/// it exactly - the unit named is a shadow map fed by the light-space-position texcoord, or an
/// albedo/normal map fed by the UV texcoord, and the shader consumes those registers as such.
/// It is also what closes the register accounting: two extra registers hold four F16
/// components, which is why the spans sum to `primary_reg_count` on all 22 blobs.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub struct SamplePrefetch {
    /// GXM texture unit sampled (a declared SAMPLER parameter's `resource_index`).
    pub unit: u8,
    /// TEXCOORD index whose interpolated value is the sample coordinate. It need NOT be one of
    /// this program's own interpolants: a texcoord consumed only by the PDS costs the shader no
    /// PA registers and so is not declared at all.
    pub source_texcoord: u8,
    /// This is the last prefetch in the program's fetch sequence ([`INFO_PREFETCH_LAST`]).
    pub last: bool,
}

/// One fragment-program interpolated input, decoded from the varyings block's per-interpolant
/// descriptor array. `pa_base` is the PA (primary-attribute) register the interpolated value
/// lands in and `register_count` how many registers that data occupies; a `prefetch` sample,
/// when present, occupies the two registers immediately after it.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub struct Interpolant {
    pub usage: VaryingUsage,
    /// PA register index the varying occupies (raw register number, not a scalar lane).
    pub pa_base: u8,
    /// Number of PA registers this varying's interpolated DATA occupies (1..4).
    pub register_count: u8,
    /// Number of PA registers the varying is ALLOCATED, which is what the next interpolant's
    /// base accumulates by: `register_count`, plus two more when a [`SamplePrefetch`] rides
    /// along. Always at least `register_count`.
    pub span: u8,
    /// Whether the interpolated value arrives as packed F16 halves (two scalar components per
    /// PA register) rather than one F32 per register - the `attribute_info` precision field.
    /// This decides how many interpolated components the vertex stage must supply for it.
    pub half: bool,
    /// A texture sample the PDS leaves in the registers at `pa_base + register_count`, as
    /// packed F16 components - [`Interpolant::prefetch_regs`] of them.
    pub prefetch: Option<SamplePrefetch>,
    /// How many PA registers this descriptor's prefetched sample occupies: 2 (four packed F16
    /// components) or 1 (two).
    ///
    /// This is what `size` bit 6 means. MEASURED by the closure the whole PA layout rests on -
    /// the descriptor spans must sum to the program's own `primary_reg_count`. Reading every
    /// prefetch as two registers makes fourteen of a title's fragment programs overrun their
    /// declared count by EXACTLY ONE each, and every one of those fourteen is a descriptor
    /// with `size` bit 6 clear; reading those as one register closes all fourteen exactly.
    /// No other assignment closes them, because the miss is a constant one register per
    /// program and each has exactly one such descriptor.
    pub prefetch_regs: u8,
}

impl Interpolant {
    /// First PA register of this interpolant's prefetched sample, if it has one. The sample's
    /// four F16 components occupy this register and the next.
    pub fn prefetch_base(&self) -> Option<u8> {
        self.prefetch.map(|_| self.pa_base + self.register_count)
    }
}

/// One varying a VERTEX program produces, placed in its OUTPUT register bank. Unlike the
/// fragment side (which packs F16 varyings two per register), the vertex writes ONE float per
/// output lane, so `components` is exactly the number of output lanes this varying occupies,
/// starting at `base_lane`. The rasteriser interpolates each lane independently and only then
/// packs the result into the fragment's PA registers at the consuming precision - which is why
/// an F16 fragment varying costs two vertex lanes.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub struct OutputVarying {
    pub usage: VaryingUsage,
    /// First OUTPUT-bank scalar lane this varying occupies.
    pub base_lane: u32,
    /// Number of interpolated scalar components (= output lanes).
    pub components: u32,
}

/// A parsed GXP program: header facts, the full parameter table, and the raw USSE
/// instruction words. Owns its data so it can outlive the source bytes.
#[derive(Debug, Clone)]
pub struct Program {
    pub kind: ProgramKind,
    pub major: u8,
    pub minor: u8,
    /// Header `size` field (bytes, excluding trailing pad).
    pub size: u32,
    /// PA register count - the count of input iterators/varyings the shader reads.
    pub primary_reg_count: u16,
    /// SA register count - the size (in 4-byte registers) of the default uniform block.
    pub secondary_reg_count: u16,
    /// Temporary register high-water mark (max of the two header temp fields).
    pub temp_reg_count: u16,
    pub parameters: Vec<Parameter>,
    /// The USSE instruction stream as 64-bit little-endian words, in program order.
    pub code: Vec<u64>,
    /// The SECONDARY program's instruction stream, in program order (empty when the program
    /// has none). This runs BEFORE [`Self::code`] and its job is to fill SA-bank registers -
    /// so a register the guest's default uniform buffer never writes can still hold a real
    /// value by the time the primary program reads it. Located by header +0x44 (count) and
    /// +0x48 (offset); see [`OFF_SECONDARY_END_OFFSET`] for the redundant end offset that
    /// cross-checks both.
    pub secondary_code: Vec<u64>,
    /// Fragment interpolated inputs (the varyings this fragment program consumes), decoded
    /// from the varyings-block descriptor array. Empty for a vertex program or a fragment
    /// program with no varyings block. Gives the PA register each varying (colour/texcoord/
    /// position) lands in, which is how the recompiled fragment's `pa[]` is fed from the
    /// vertex stage (positional-by-usage linkage).
    pub interpolants: Vec<Interpolant>,
    /// Why [`Program::interpolants`] is empty, when it is empty because the varyings block
    /// could not be decoded rather than because the program declares none.
    ///
    /// The two are completely different situations and used to be the same value. A fragment
    /// that genuinely has no interpolants is fine; one whose block failed to decode reads PA
    /// registers nothing feeds, and the link then fails with "no declared interpolant covers
    /// it" - a message that describes the SYMPTOM and hides the cause, which is here.
    pub varyings_error: Option<&'static str>,
    /// Vertex interpolated OUTPUTS (the varyings this vertex program produces), decoded from
    /// the varyings block. Empty for a fragment program, and empty for a vertex program whose
    /// decoded placement does not reproduce the block's own total output-lane count - the
    /// linker then falls back rather than route varyings by an unvalidated layout.
    pub output_varyings: Vec<OutputVarying>,
    /// Size of the default uniform buffer in 32-bit SA registers (header +0x64). A UNIFORM
    /// parameter's `resource_index` is a register index into this buffer, and the buffer is
    /// loaded at SA register 0 in the main program's address space - so the shader's `sa[k]`
    /// for `k < default_uniform_regs` IS uniform-buffer register `k`.
    pub default_uniform_regs: u32,
    /// Compile-time constants the driver stores into SA registers before the shader runs:
    /// `(sa_register, raw 32-bit value)`. An F16 literal occupies the low half of its
    /// register (e.g. `0x0000_3c00` = 1.0h).
    pub literals: Vec<(u32, u32)>,
    /// Where each bound texture's control words live: `(sa_register_base, gxm_texture_unit)`,
    /// one entry per texture (the four consecutive control words share a base). A USSE `SMP`
    /// resolves its sampler through this - see [`Program::sampler_unit_at`].
    pub texture_control: Vec<(u32, u32)>,
    /// Stable content hash of the whole blob, for pipeline caching (FNV-1a).
    pub hash: u64,
}

impl Program {
    /// The GXM texture unit whose control words sit at SA register `sa_register`, i.e. the
    /// unit a `SMP` instruction addresses. The USSE `SMP` sampler field is a register number
    /// in double-register units, so the caller passes `2 * field`. Returns `None` when no
    /// texture is declared there (a shader/state mismatch - the caller must fall back rather
    /// than sample an arbitrary unit).
    pub fn sampler_unit_at(&self, sa_register: u32) -> Option<u32> {
        self.texture_control.iter().find(|(base, _)| *base == sa_register).map(|(_, unit)| *unit)
    }

    /// Whether the SAMPLER declared at GXM texture `unit` is a CUBE map (three sample
    /// coordinates naming a direction) rather than a 2D or 3D texture. Drives both the WGSL
    /// texture type the recompiled fragment declares and the view dimension the renderer binds,
    /// so the two cannot disagree. Unknown units answer `false`: a sampler the parameter table
    /// does not declare is not a cube.
    pub fn sampler_is_cube(&self, unit: u32) -> bool {
        self.sampler_at(unit).is_some_and(|p| p.sampler_cube)
    }

    /// The SAMPLER parameter declared at GXM texture `unit`, if the program declares one. A
    /// prefetched sample names its unit directly rather than through the texture-control table,
    /// so this is how the linker checks that the unit is one the program actually declares
    /// before binding a texture to it.
    pub fn sampler_at(&self, unit: u32) -> Option<&Parameter> {
        self.parameters.iter().find(|p| {
            p.category == ParamCategory::Sampler
                && p.resource_index >= 0
                && p.resource_index as u32 == unit
        })
    }
}

/// Parse the SA-resident constant/texture tables. Both are indexed in the same "table"
/// space, which the main program sees offset by the default-uniform-buffer size:
/// `sa_register = table_index + default_uniform_regs` (table indices 0 and 1 are reserved).
/// Verified by exact tiling against `secondary_reg_count` on every captured fragment blob,
/// and semantically (a fog `1.0h` literal read as `1.0 - fogcoord`).
fn parse_sa_tables(bytes: &[u8], default_uniform_regs: u32) -> (Vec<(u32, u32)>, Vec<(u32, u32)>) {
    let sa_reg = |table_index: u32| table_index.wrapping_add(default_uniform_regs);
    let mut literals = Vec::new();
    if let (Some(count), Some(rel)) = (rd_u32(bytes, OFF_LITERAL_COUNT), rd_u32(bytes, OFF_LITERAL_OFFSET)) {
        let base = OFF_LITERAL_OFFSET.wrapping_add(rel as usize);
        for i in 0..count as usize {
            let e = base.wrapping_add(i * LITERAL_ENTRY);
            match (rd_u32(bytes, e), rd_u32(bytes, e + 4)) {
                (Some(index), Some(value)) => literals.push((sa_reg(index), value)),
                _ => break,
            }
        }
    }
    // Texture control words: four entries per texture, `(index << 16) | (unit << 2) | word`.
    // Only word 0 names the base register, so the other three are structural padding here.
    let mut texture_control = Vec::new();
    if let (Some(count), Some(rel)) = (rd_u32(bytes, OFF_TEXTURE_COUNT), rd_u32(bytes, OFF_TEXTURE_OFFSET)) {
        let base = OFF_TEXTURE_OFFSET.wrapping_add(rel as usize);
        for i in 0..count as usize {
            let Some(e) = rd_u32(bytes, base.wrapping_add(i * TEXTURE_ENTRY)) else { break };
            if e & 0x3 != 0 {
                continue;
            }
            texture_control.push((sa_reg(e >> 16), (e & 0xffff) >> 2));
        }
    }
    (literals, texture_control)
}

/// Why a blob failed to parse. Parsing is fast-fail: a malformed or out-of-range field
/// is a hard error, never a silently-truncated partial parse.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum ParseError {
    /// Blob shorter than the fixed header needs.
    TooSmall,
    /// Magic was not "GXP\0".
    BadMagic(u32),
    /// Major/minor version this parser has not validated against.
    UnsupportedVersion(u8, u8),
    /// A self-relative offset (params / asm / literal) pointed outside the blob.
    OffsetOutOfRange(&'static str),
    /// The USSE code region length was not a whole number of 64-bit instructions.
    CodeMisaligned,
    /// The parameter table ran past the end of the blob.
    ParamTableOutOfRange,
    /// The secondary program's count and its two offsets disagree, or its block lies outside
    /// the blob. The three fields are redundant by construction, so a disagreement means the
    /// layout is not what this parser understands - refuse the program rather than run a
    /// misread instruction stream into the SA bank.
    SecondaryCodeInconsistent,
}

// Header field offsets (bytes) in the SceGxmProgram container.
const OFF_MAGIC: usize = 0x00;
const OFF_MAJOR: usize = 0x04;
const OFF_MINOR: usize = 0x05;
const OFF_SIZE: usize = 0x08;
const OFF_TYPE: usize = 0x14;
const OFF_PARAM_COUNT: usize = 0x24;
const OFF_PARAMS_OFFSET: usize = 0x28;
/// Self-relative offset to the varyings block (`SceGxmProgramVertexVaryings`); 0 = none.
const OFF_VARYINGS_OFFSET: usize = 0x2c;
const OFF_PRIMARY_REG: usize = 0x30;
const OFF_SECONDARY_REG: usize = 0x32;
const OFF_TEMP1: usize = 0x34;
const OFF_TEMP2: usize = 0x38;
const OFF_ASM_OFFSET: usize = 0x40;
/// Number of instructions in the SECONDARY program (see [`Program::secondary_code`]).
const OFF_SECONDARY_COUNT: usize = 0x44;
/// Self-relative offset to the SECONDARY program's first instruction.
const OFF_SECONDARY_OFFSET: usize = 0x48;
/// Self-relative offset to just past the SECONDARY program's last instruction. Its distance
/// from [`OFF_SECONDARY_OFFSET`] is exactly `8 * count` on every captured blob, which is what
/// makes the whole triple self-checking.
const OFF_SECONDARY_END_OFFSET: usize = 0x4c;
/// Default uniform buffer size, in 32-bit SA registers.
const OFF_DEFAULT_UNIFORM_REGS: usize = 0x64;
const OFF_LITERAL_COUNT: usize = 0x70;
const OFF_LITERAL_OFFSET: usize = 0x74;
const OFF_TEXTURE_COUNT: usize = 0x80;
const OFF_TEXTURE_OFFSET: usize = 0x84;
/// On-disk size of one literal entry: `(u32 table_index, u32 value)`.
const LITERAL_ENTRY: usize = 8;
/// On-disk size of one texture-control-word entry (a single packed u32).
const TEXTURE_ENTRY: usize = 4;
/// Minimum header size to safely read every fixed field above (through 0x78).
const MIN_HEADER: usize = 0x7c;
/// On-disk size of one parameter entry.
const PARAM_ENTRY: usize = 16;

#[inline]
fn rd_u16(b: &[u8], off: usize) -> Option<u16> {
    b.get(off..off + 2).map(|s| u16::from_le_bytes([s[0], s[1]]))
}

#[inline]
fn rd_u32(b: &[u8], off: usize) -> Option<u32> {
    b.get(off..off + 4).map(|s| u32::from_le_bytes([s[0], s[1], s[2], s[3]]))
}

/// FNV-1a 64 over the whole blob - a stable, wasm-safe content key for the pipeline
/// cache (identical shader bytes -> identical hash, no `Hash`/`RandomState` needed).
fn fnv1a64(b: &[u8]) -> u64 {
    let mut h: u64 = 0xcbf2_9ce4_8422_2325;
    for &byte in b {
        h ^= byte as u64;
        h = h.wrapping_mul(0x0000_0100_0000_01b3);
    }
    h
}

impl Program {
    /// Parse a complete `SceGxmProgram` blob. `bytes` must be the whole container
    /// (header + parameter table + USSE code + literals), exactly as
    /// `sceGxmProgramGetSize` reports and as the capture dumps store.
    pub fn parse(bytes: &[u8]) -> Result<Program, ParseError> {
        if bytes.len() < MIN_HEADER {
            return Err(ParseError::TooSmall);
        }
        let magic = rd_u32(bytes, OFF_MAGIC).ok_or(ParseError::TooSmall)?;
        if magic != GXP_MAGIC {
            return Err(ParseError::BadMagic(magic));
        }
        let major = bytes[OFF_MAJOR];
        let minor = bytes[OFF_MINOR];
        // Every retail Vita GXP seen is 1.x with x>=4; refuse anything older than the
        // layout this parser was verified against rather than mis-parse it.
        if major != 1 || minor < 4 {
            return Err(ParseError::UnsupportedVersion(major, minor));
        }
        let size = rd_u32(bytes, OFF_SIZE).ok_or(ParseError::TooSmall)?;

        let kind = if bytes[OFF_TYPE] & 1 == 1 {
            ProgramKind::Fragment
        } else {
            ProgramKind::Vertex
        };

        let primary_reg_count = rd_u16(bytes, OFF_PRIMARY_REG).ok_or(ParseError::TooSmall)?;
        let secondary_reg_count = rd_u16(bytes, OFF_SECONDARY_REG).ok_or(ParseError::TooSmall)?;
        let temp1 = rd_u16(bytes, OFF_TEMP1).ok_or(ParseError::TooSmall)?;
        let temp2 = rd_u16(bytes, OFF_TEMP2).ok_or(ParseError::TooSmall)?;
        let temp_reg_count = temp1.max(temp2);

        let parameters = parse_parameters(bytes)?;
        let (interpolants, varyings_error, output_varyings) = if kind == ProgramKind::Fragment {
            match parse_fragment_interpolants(bytes) {
                Ok(v) => (v, None, Vec::new()),
                Err(why) => (Vec::new(), Some(why), Vec::new()),
            }
        } else {
            match parse_vertex_output_varyings(bytes) {
                Ok(v) => (Vec::new(), None, v),
                Err(why) => (Vec::new(), Some(why), Vec::new()),
            }
        };

        // USSE code region: [asm_abs .. min(literal_abs, params_abs)]. Self-relative
        // offsets from their own field address.
        let asm_rel = rd_u32(bytes, OFF_ASM_OFFSET).ok_or(ParseError::TooSmall)?;
        let asm_abs = OFF_ASM_OFFSET
            .checked_add(asm_rel as usize)
            .ok_or(ParseError::OffsetOutOfRange("asm"))?;
        let lit_rel = rd_u32(bytes, OFF_LITERAL_OFFSET).ok_or(ParseError::TooSmall)?;
        let lit_abs = OFF_LITERAL_OFFSET
            .checked_add(lit_rel as usize)
            .ok_or(ParseError::OffsetOutOfRange("literal"))?;
        let params_abs = params_table_abs(bytes)?;

        if asm_abs > bytes.len() {
            return Err(ParseError::OffsetOutOfRange("asm"));
        }
        // The code ends at whichever of the literal block / parameter table comes first
        // after the code (both sit after the instructions in every observed blob).
        let mut end = bytes.len();
        if lit_abs > asm_abs && lit_abs <= bytes.len() {
            end = end.min(lit_abs);
        }
        if params_abs > asm_abs && params_abs <= bytes.len() {
            end = end.min(params_abs);
        }
        if end < asm_abs {
            return Err(ParseError::OffsetOutOfRange("asm"));
        }
        let code_bytes = &bytes[asm_abs..end];
        if code_bytes.len() % 8 != 0 {
            return Err(ParseError::CodeMisaligned);
        }
        let code = code_bytes
            .chunks_exact(8)
            .map(|c| u64::from_le_bytes([c[0], c[1], c[2], c[3], c[4], c[5], c[6], c[7]]))
            .collect();

        let secondary_code = parse_secondary_code(bytes)?;

        let default_uniform_regs = rd_u32(bytes, OFF_DEFAULT_UNIFORM_REGS).unwrap_or(0);
        let (literals, texture_control) = parse_sa_tables(bytes, default_uniform_regs);

        Ok(Program {
            kind,
            major,
            minor,
            size,
            default_uniform_regs,
            literals,
            texture_control,
            primary_reg_count,
            secondary_reg_count,
            temp_reg_count,
            parameters,
            code,
            secondary_code,
            interpolants,
            varyings_error,
            output_varyings,
            hash: fnv1a64(bytes),
        })
    }

    /// The samplers this fragment program declares, as (unit, name). Handy for the
    /// integration layer to bind the right captured texture to each USSE SMP.
    pub fn samplers(&self) -> impl Iterator<Item = (u32, &str)> {
        self.parameters
            .iter()
            .filter(|p| p.category == ParamCategory::Sampler)
            .map(|p| (p.resource_index.max(0) as u32, p.name.as_str()))
    }
}

/// Decode a fragment program's interpolated inputs from the varyings-block descriptor array.
///
/// The varyings block sits at the self-relative offset stored at header +0x2C. For a fragment
/// program it is a fixed header (`varyings_count` at +0x0C) followed by `varyings_count`
/// 16-byte `SceGxmProgramAttributeDescriptor` entries. Each descriptor names a usage in its
/// `attribute_info` high nibble and spans `((size>>4)&3)+1` PA registers; the PA register base
/// accumulates across the array in declaration order (positional-by-usage linkage).
///
/// Decode a fragment program's interpolated inputs from the varyings-block descriptor array.
///
/// Layout (byte-exact, validated against real blobs): the varyings block sits at the
/// self-relative offset stored at header +0x2C. For a fragment program, the u32 at
/// `block + 0x10` is itself a SELF-RELATIVE offset to the descriptor array (add it to the
/// address of that field). `varyings_count` (at block +0x0C) 16-byte descriptors follow, each
/// `[attribute_info u32, resource_index u32, size u32, component_info u32]`. Each descriptor's
/// interpolated data spans `((size>>4)&3)+1` PA registers, plus two more when it carries a
/// [`SamplePrefetch`]; the PA register base is purely accumulated across the array in
/// declaration order (positional-by-usage linkage - no explicit base field). That accumulation
/// is not a fit but a closure: the spans sum to the program's own `primary_reg_count` on every
/// captured blob.
///
/// Best-effort and never fails the whole parse: any out-of-range read yields an empty list
/// (the renderer then falls back to the fixed-function path, never a wrong binding). The
/// oracle harness cross-checks the decoded texcoord PA spans against the SMP coordinate reads.
/// The RAW varying descriptors of a fragment program, as
/// `[attribute_info, resource_index, size, component_info]` per entry.
///
/// For reverse engineering the varyings block: when [`parse_fragment_interpolants`] refuses a
/// blob, the only way forward is to look at the words it refused, and reconstructing the block
/// address by hand from a hex dump is exactly the kind of step that gets done wrong. Returns
/// an empty list when the block itself cannot be located.
pub fn raw_varying_descriptors(bytes: &[u8]) -> Vec<[u32; 4]> {
    const DESCRIPTOR_LEN: usize = 16;
    let Some(rel) = rd_u32(bytes, OFF_VARYINGS_OFFSET).filter(|r| *r != 0) else { return Vec::new() };
    let Some(block) = OFF_VARYINGS_OFFSET.checked_add(rel as usize) else { return Vec::new() };
    let Some(count) = rd_u16(bytes, block + 0x0c).map(usize::from).filter(|c| *c <= 32) else {
        return Vec::new();
    };
    let arr_field = block + 0x10;
    let Some(arr) = rd_u32(bytes, arr_field).and_then(|r| arr_field.checked_add(r as usize)) else {
        return Vec::new();
    };
    (0..count)
        .filter_map(|i| {
            let d = arr + i * DESCRIPTOR_LEN;
            Some([
                rd_u32(bytes, d)?,
                rd_u32(bytes, d + 4)?,
                rd_u32(bytes, d + 8)?,
                rd_u32(bytes, d + 12)?,
            ])
        })
        .collect()
}

fn parse_fragment_interpolants(bytes: &[u8]) -> Result<Vec<Interpolant>, &'static str> {
    const DESCRIPTOR_LEN: usize = 16;

    let Some(rel) = rd_u32(bytes, OFF_VARYINGS_OFFSET) else {
        return Err("the varyings-block offset field is outside the blob");
    };
    if rel == 0 {
        return Err("the program declares no varyings block (offset 0)");
    }
    let Some(block) = OFF_VARYINGS_OFFSET.checked_add(rel as usize) else {
        return Err("the varyings-block offset overflows");
    };
    let Some(count) = rd_u16(bytes, block + 0x0c) else {
        return Err("the varyings count is outside the blob");
    };
    let count = count as usize;
    // A sane fragment program has a handful of varyings; reject an absurd count rather than
    // walk off the blob (a sign the block offset is wrong for this blob).
    if count == 0 {
        return Err("the varyings block declares a count of 0");
    }
    if count > 32 {
        return Err("the varyings count is absurd (>32), so the block offset is wrong here");
    }
    // The descriptor array is at a self-relative offset stored at block + 0x10.
    let arr_field = block + 0x10;
    let Some(arr_rel) = rd_u32(bytes, arr_field) else {
        return Err("the descriptor-array offset field is outside the blob");
    };
    let Some(arr) = arr_field.checked_add(arr_rel as usize) else {
        return Err("the descriptor-array offset overflows");
    };

    let mut out = Vec::with_capacity(count);
    let mut pa_base: u32 = 0;
    for i in 0..count {
        let d = arr + i * DESCRIPTOR_LEN;
        let (Some(attribute_info), Some(resource_index), Some(size), Some(component_info)) =
            (rd_u32(bytes, d), rd_u32(bytes, d + 4), rd_u32(bytes, d + 8), rd_u32(bytes, d + 12))
        else {
            // Ran off the blob -> layout mismatch for this blob; bind nothing.
            return Err("a varying descriptor lies outside the blob");
        };
        let usage = varying_usage_from_attribute_info(attribute_info);
        // A descriptor whose semantic nibble is 0xF interpolates NOTHING: it exists only to
        // carry a PDS-prefetched sample. Its size field's register-count bits are 0 and its
        // precision bits are clear, because there is no iterated data for them to describe.
        //
        // Measured, on the closure the whole PA layout rests on (the descriptor spans must
        // sum to the program's own `primary_reg_count`): reading these as one data register
        // plus the prefetch pair closes on 11 of a retail racer's 18 race fragment programs;
        // reading them as prefetch-only closes on 15, and the three that still fall short
        // are short because the program allocates PA registers no descriptor covers, which
        // is allowed. The 40-blob corpus is unaffected either way - it contains no 0xF
        // descriptor at all, which is why this went unnoticed until a title composited its
        // world through one.
        let prefetch_only = matches!(usage, VaryingUsage::Unknown(info) if (info >> 12) & 0xf == 0xf);
        let register_count = if prefetch_only { 0 } else { ((size >> 4) & 0x3) as u8 + 1 };
        // Precision field (`attribute_info & 0x30100000`): observed as 0x20000000 on every F16
        // interpolant across the captured blobs and 0 on the F32 ones. Cross-checked by span:
        // an F16 varying packs two components per PA register, so a 4-register interpolant is
        // always F32 (four components is a texcoord's maximum).
        let half = attribute_info & 0x2000_0000 != 0;

        // TWO independent fields state whether a prefetched sample rides along, and they agree
        // on every descriptor of every captured blob, so a disagreement means this program's
        // varyings block is not the layout decoded here - bind nothing rather than a wrong PA
        // register map. (`size` bit 6 was once a third; see
        // `_SIZE_BIT6_NOT_A_PREFETCH_FLAG` for the measurement that removed it.)
        let source = attribute_info & INFO_PREFETCH_SOURCE;
        let flags = [
            attribute_info & INFO_PREFETCH != 0,
            // A BIT test, not equality: a retail title has a descriptor carrying 0x30 here, and
            // rejecting it threw away that whole program's interpolant list (the parse is
            // all-or-nothing) over a bit that is not the prefetch flag.
            component_info & COMPONENT_INFO_PREFETCH != 0,
        ];
        let prefetch = match flags {
            [false, false] => {
                // A descriptor with no prefetch names no unit and no source.
                if source != PREFETCH_SOURCE_NONE || resource_index != 0 {
                    return Err("a descriptor carries no prefetch flag yet names a unit or source");
                }
                None
            }
            [true, true] => {
                if source > MAX_TEXCOORD as u32 || resource_index > u8::MAX as u32 {
                    return Err("a prefetch descriptor names an out-of-range texcoord or unit");
                }
                Some(SamplePrefetch {
                    unit: resource_index as u8,
                    source_texcoord: source as u8,
                    last: attribute_info & INFO_PREFETCH_LAST != 0,
                })
            }
            _ => {
                return Err(
                    "the two prefetch flags of a descriptor disagree, so this program's \
                     varyings block is not the layout decoded here",
                )
            }
        };

        // `size` bit 6 says the prefetched sample occupies TWO PA registers (four packed F16
        // components) rather than one - see `Interpolant::prefetch_regs`.
        let prefetch_regs = if size & 0x40 != 0 { 2 } else { 1 };
        let span = register_count + if prefetch.is_some() { prefetch_regs } else { 0 };
        out.push(Interpolant {
            usage,
            pa_base: pa_base.min(u8::MAX as u32) as u8,
            register_count,
            span,
            half,
            prefetch,
            prefetch_regs,
        });
        pa_base += span as u32;
    }
    Ok(out)
}

/// The OUTPUT lanes a vertex program's clip POSITION occupies. The rasteriser consumes these;
/// they are never a varying. Every other output lane is either a reserved fog / point-size slot
/// or an interpolated varying.
const VERTEX_POSITION_LANES: u32 = 4;

/// The reserved-slot width, in output lanes, for which the FOG lane's position is established.
/// See [`parse_vertex_output_varyings`].
const FOG_RESERVED_LANES: u32 = 2;

/// The reserved-slot width, in output lanes, for which COLOR0's position is established: a
/// full vec4 filling the region. See [`parse_vertex_output_varyings`].
const COLOR0_RESERVED_LANES: u32 = 4;

/// Decode a VERTEX program's interpolated outputs from its varyings block.
///
/// The block (header +0x2C, self-relative) carries two words the vertex side needs:
/// `vertex_outputs1` at +0x10, whose top byte is the program's TOTAL output-lane count, and
/// `vertex_outputs2` at +0x14, ten 3-bit fields giving each TEXCOORD's component width
/// (`(v&1)*2 + ((v>>1)&1) + ((v>>2)&1)` = 2/3/4 components; 0 = that texcoord absent).
///
/// The texcoords sit at the TOP of the output bank, in ascending index, one lane per component.
/// So their base lane is not a constant - it is the block's own total minus the widths it
/// declares, leaving lanes `4..base` as the reserved fog / point-size region. That accounting is
/// a CHECK as much as a placement: the reserved region must be non-negative, and it reproduces
/// every captured program's actually-written output lanes exactly (including one whose texcoord
/// starts at lane 12 behind an eight-lane reserved region, and two with no reserved region at
/// all whose varyings start at lane 4). A contradictory block yields an empty list, which sends
/// the linker to its fixed-function fallback rather than route varyings by a layout the
/// container disagrees with.
///
/// FOG is declared only for the [`FOG_RESERVED_LANES`]-wide reserved region, where it occupies
/// the FIRST reserved lane as a single component. ESTABLISHED, not assumed: every captured
/// vertex program with that region ends in a byte-identical F16->F32 pack whose destination is
/// output lane 4, computing `clamp(-depth * fogScale, 0, fogMax)` from the lane its own
/// projection wrote - and the fragments that consume it declare Fog as exactly one F32
/// component.
///
/// COLOR0 fills a [`COLOR0_RESERVED_LANES`]-wide reserved region as a vec4 from the first
/// reserved lane, established the same way FOG's placement is - by reading what the programs
/// with that region actually write. One moves `Output[4] <- PrimaryAttr[4]`, whose parameter
/// table names PA register 4 as a 4-component ATTRIBUTE with the COLOR semantic; another moves
/// `Output[4] <- SecondaryAttr[0]`, a 4-component `color` UNIFORM. Both are paired with a
/// fragment declaring Color0, and region width, source component count and consumer
/// declaration all agree at 4.
///
/// HISTORY, because declaring this was once tried and reverted: on its own it made every draw
/// of a title recompile and the screen render BLACK, where the fixed-function approximation had
/// been legible. The missing piece was not the placement - it was that each of those `mov`s
/// carries a REPEAT COUNT, so a single instruction under a two-channel mask writes two or three
/// lane PAIRS, and the recompiler was executing it once. Lanes 6..7 (COLOR0's z/w) and, in the
/// textured program, lanes 8..9 (the only write of the texture coordinate anywhere in it) were
/// simply never written, so every glyph sampled the atlas at (0,0) and multiplied by a colour
/// with zero alpha. See [`crate::usse::unroll_repeats`]; with repetition modelled, the written
/// lanes reproduce the container's own total exactly and this declaration is safe.
///
/// The 8-lane region is NOT placed, and that is a measured decision rather than a gap nobody
/// looked at. What the corpus does establish is its CONTENT: a racing title's race frame has
/// seven vertex programs with an 8-lane region, and the fragment each is drawn with declares
/// **Color1** - a usage that appears next to no 4-lane region anywhere in the corpus, where every
/// fragment declares Color0 alone. Three of the seven read both colours and four read only
/// Color1.
///
/// What it does NOT establish is the LAYOUT, and both readings were tried and refuted against
/// PIXELS. Declaring COLOR0 at lane 4 and COLOR1 at lane 8 makes the Color1-only fragments read
/// the lanes their vertex fills from `In.UV1` - texture coordinates used as a colour - and they
/// paint saturated yellow over the track. Swapping them makes those fragments read a lane pair
/// the vertex fills from the literal `(0, 1, -0.01, 0)`, and they paint pure green. Neither is a
/// colour, so neither placement is right, and a third arrangement is not something to guess at:
/// the region may not be two vec4s at all.
///
/// So these pairs FALL BACK, visibly and by name, and the renderer draws its approximation -
/// which looks like scenery - instead of a recompiled shader sampling a register the vertex
/// filled with something else. Settling it needs evidence from the varyings block itself.
///
/// The same rule still covers every other width: a fragment reading a varying we did not place
/// falls back rather than sample an uninterpolated register.
fn parse_vertex_output_varyings(bytes: &[u8]) -> Result<Vec<OutputVarying>, &'static str> {
    let Some(rel) = rd_u32(bytes, OFF_VARYINGS_OFFSET) else {
        return Err("the varyings-block offset field is outside the blob");
    };
    if rel == 0 {
        // No block at all. A program with no varyings block outputs clip position and nothing
        // else, which is exactly what a depth-only (shadow/z-prepass) vertex program is.
        return Ok(Vec::new());
    }
    let Some(block) = OFF_VARYINGS_OFFSET.checked_add(rel as usize) else {
        return Err("the varyings-block offset overflowed");
    };
    let (Some(vo1), Some(vo2)) = (rd_u32(bytes, block + 0x10), rd_u32(bytes, block + 0x14)) else {
        return Err("the varyings block's two output words lie outside the blob");
    };

    let widths: Vec<(u8, u32)> = (0..=MAX_TEXCOORD as u32)
        .filter_map(|k| {
            let v = (vo2 >> (k * 3)) & 0x7;
            (v != 0).then(|| (k as u8, (v & 1) * 2 + ((v >> 1) & 1) + ((v >> 2) & 1)))
        })
        .collect();
    // No texcoords is a real layout, not a decode failure: a program can forward only COLOR0
    // (one of the two that draw a retail title's whole front-end does exactly that, declaring
    // 8 total lanes = clip position + a 4-lane reserved region). The lane accounting below is
    // what validates the result, so it does not need a texcoord to be trustworthy.
    let total_lanes = vo1 >> 24;
    let texcoord_lanes: u32 = widths.iter().map(|&(_, n)| n).sum();
    let Some(base_lane) = total_lanes.checked_sub(texcoord_lanes) else {
        return Err("the decoded texcoord widths exceed the block's own total output-lane count");
    };
    let Some(reserved) = base_lane.checked_sub(VERTEX_POSITION_LANES) else {
        return Err("the block declares fewer output lanes than the clip position occupies");
    };

    let mut out = Vec::new();
    if reserved == FOG_RESERVED_LANES {
        out.push(OutputVarying {
            usage: VaryingUsage::Fog,
            base_lane: VERTEX_POSITION_LANES,
            components: 1,
        });
    }
    if reserved == COLOR0_RESERVED_LANES {
        out.push(OutputVarying {
            usage: VaryingUsage::Color0,
            base_lane: VERTEX_POSITION_LANES,
            components: COLOR0_RESERVED_LANES,
        });
    }
    let mut lane = base_lane;
    for (k, components) in widths {
        out.push(OutputVarying { usage: VaryingUsage::TexCoord(k), base_lane: lane, components });
        lane += components;
    }
    Ok(out)
}

/// Decode a fragment interpolant descriptor's usage from its `attribute_info` semantic nibble
/// (`info & 0xF000`): texcoord 0..9 = 0x0000..0x9000, COLOR0 = 0xA000, COLOR1 = 0xB000,
/// FOG = 0xC000, POSITION = 0xD000, 0xE000/0xF000 unused. Bit 0x40000000 marks a
/// fragment-generated sprite/point coordinate (no vertex output feeds it) -> `Unknown`.
fn varying_usage_from_attribute_info(info: u32) -> VaryingUsage {
    if info & 0x4000_0000 != 0 {
        return VaryingUsage::Unknown(info);
    }
    match info & 0xf000 {
        v @ 0x0000..=0x9000 => VaryingUsage::TexCoord((v >> 12) as u8),
        0xa000 => VaryingUsage::Color0,
        0xb000 => VaryingUsage::Color1,
        0xc000 => VaryingUsage::Fog,
        0xd000 => VaryingUsage::Position,
        _ => VaryingUsage::Unknown(info),
    }
}

/// Absolute byte position of the parameter table (self-relative from 0x28).
fn params_table_abs(bytes: &[u8]) -> Result<usize, ParseError> {
    let rel = rd_u32(bytes, OFF_PARAMS_OFFSET).ok_or(ParseError::TooSmall)?;
    OFF_PARAMS_OFFSET
        .checked_add(rel as usize)
        .ok_or(ParseError::OffsetOutOfRange("params"))
}

/// Extract the SECONDARY program's instruction stream (see [`Program::secondary_code`]).
///
/// The header states it three times over - a count at +0x44 and self-relative start/end offsets
/// at +0x48/+0x4c - and on every one of the 43 captured blobs `end - start == 8 * count`
/// exactly, for counts from 0 to 15. That redundancy is the proof the fields mean what they say
/// (a coincidence would have to hold across every program of every shader family), and it is
/// checked here on every parse: a program whose three statements disagree is refused rather than
/// decoded, because the alternative is running a mis-sliced instruction stream into the SA bank
/// and silently poisoning every uniform the primary program reads.
fn parse_secondary_code(bytes: &[u8]) -> Result<Vec<u64>, ParseError> {
    let count = rd_u32(bytes, OFF_SECONDARY_COUNT).ok_or(ParseError::TooSmall)? as usize;
    // No secondary program: there is no stream to slice, so nothing can be mis-sliced and the
    // offsets are not required to agree (a program without one need not fill them in).
    if count == 0 {
        return Ok(Vec::new());
    }
    let start = OFF_SECONDARY_OFFSET
        .checked_add(rd_u32(bytes, OFF_SECONDARY_OFFSET).ok_or(ParseError::TooSmall)? as usize)
        .ok_or(ParseError::SecondaryCodeInconsistent)?;
    let end = OFF_SECONDARY_END_OFFSET
        .checked_add(rd_u32(bytes, OFF_SECONDARY_END_OFFSET).ok_or(ParseError::TooSmall)? as usize)
        .ok_or(ParseError::SecondaryCodeInconsistent)?;
    if end.checked_sub(start) != count.checked_mul(8) || end > bytes.len() {
        return Err(ParseError::SecondaryCodeInconsistent);
    }
    Ok(bytes[start..end]
        .chunks_exact(8)
        .map(|c| u64::from_le_bytes([c[0], c[1], c[2], c[3], c[4], c[5], c[6], c[7]]))
        .collect())
}

fn parse_parameters(bytes: &[u8]) -> Result<Vec<Parameter>, ParseError> {
    let count = rd_u32(bytes, OFF_PARAM_COUNT).ok_or(ParseError::TooSmall)? as usize;
    let table = params_table_abs(bytes)?;
    // Guard the whole table up front so a corrupt count cannot walk off the blob.
    let table_end = table
        .checked_add(count.checked_mul(PARAM_ENTRY).ok_or(ParseError::ParamTableOutOfRange)?)
        .ok_or(ParseError::ParamTableOutOfRange)?;
    if table_end > bytes.len() {
        return Err(ParseError::ParamTableOutOfRange);
    }
    let mut out = Vec::with_capacity(count);
    for i in 0..count {
        let p = table + i * PARAM_ENTRY;
        let name_rel = rd_u32(bytes, p).ok_or(ParseError::ParamTableOutOfRange)? as i32;
        let packed = rd_u32(bytes, p + 4).ok_or(ParseError::ParamTableOutOfRange)?;
        let array_size = rd_u32(bytes, p + 8).ok_or(ParseError::ParamTableOutOfRange)?.max(1);
        let resource_index = rd_u32(bytes, p + 12).ok_or(ParseError::ParamTableOutOfRange)? as i32;

        let category = ParamCategory::from_bits((packed & 0xf) as u8);
        let ptype = ParamType::from_bits(((packed >> 4) & 0xf) as u8);
        let component_count = (((packed >> 8) & 0xf) as u8).max(1);
        let container_index = ((packed >> 12) & 0xf) as u8;
        let sampler_cube = packed & SAMPLER_CUBE_BIT != 0;

        // Name string sits at a SELF-RELATIVE offset from this parameter's own address.
        let name = read_cstr_self_relative(bytes, p, name_rel);

        out.push(Parameter {
            name,
            category,
            ptype,
            component_count,
            container_index,
            sampler_cube,
            array_size,
            resource_index,
        });
    }
    Ok(out)
}

/// Read a NUL-terminated ASCII name whose offset is stored self-relative to `field_pos`.
/// Returns an empty string if the offset lands outside the blob (a malformed name never
/// aborts the whole parse - the register bindings, not the names, are what matter).
fn read_cstr_self_relative(bytes: &[u8], field_pos: usize, rel: i32) -> String {
    let abs = field_pos as i64 + rel as i64;
    if abs < 0 || abs as usize >= bytes.len() {
        return String::new();
    }
    let start = abs as usize;
    let end = bytes[start..]
        .iter()
        .position(|&b| b == 0)
        .map(|n| start + n)
        .unwrap_or(bytes.len());
    String::from_utf8_lossy(&bytes[start..end]).into_owned()
}

#[cfg(test)]
mod tests {
    use super::*;

    /// One parameter to bake into a synthetic blob.
    struct P {
        name: &'static str,
        category: u8,
        ptype: u8,
        comp: u8,
        container: u8,
        array: u32,
        res: i32,
    }

    /// Build a minimal but structurally valid GXP blob: fixed header, a parameter table
    /// (with names in a string pool after it), then a USSE code region of `code` words.
    /// Returns the bytes. Mirrors the real layout so the parser is exercised end to end.
    fn build_blob(kind_fragment: bool, params: &[P], code: &[u64]) -> Vec<u8> {
        let header_len = 0x80usize;
        let params_abs = header_len;
        let table_len = params.len() * PARAM_ENTRY;
        let strpool_abs = params_abs + table_len;

        // Lay out the string pool and record each name's absolute offset.
        let mut strpool = Vec::new();
        let mut name_abs = Vec::new();
        for p in params {
            name_abs.push(strpool_abs + strpool.len());
            strpool.extend_from_slice(p.name.as_bytes());
            strpool.push(0);
        }
        // Align the code region to 8 bytes after the string pool.
        let mut code_abs = strpool_abs + strpool.len();
        code_abs = (code_abs + 7) & !7;

        let total = code_abs + code.len() * 8;
        let mut b = vec![0u8; total];

        // Header.
        b[0..4].copy_from_slice(&GXP_MAGIC.to_le_bytes());
        b[OFF_MAJOR] = 1;
        b[OFF_MINOR] = 4;
        b[OFF_SIZE..OFF_SIZE + 4].copy_from_slice(&(total as u32).to_le_bytes());
        b[OFF_TYPE] = if kind_fragment { 0x01 } else { 0x00 };
        b[OFF_PARAM_COUNT..OFF_PARAM_COUNT + 4].copy_from_slice(&(params.len() as u32).to_le_bytes());
        // params_offset is self-relative from OFF_PARAMS_OFFSET.
        let params_rel = (params_abs - OFF_PARAMS_OFFSET) as u32;
        b[OFF_PARAMS_OFFSET..OFF_PARAMS_OFFSET + 4].copy_from_slice(&params_rel.to_le_bytes());
        b[OFF_PRIMARY_REG..OFF_PRIMARY_REG + 2].copy_from_slice(&7u16.to_le_bytes());
        b[OFF_SECONDARY_REG..OFF_SECONDARY_REG + 2].copy_from_slice(&12u16.to_le_bytes());
        b[OFF_TEMP1..OFF_TEMP1 + 2].copy_from_slice(&5u16.to_le_bytes());
        b[OFF_TEMP2..OFF_TEMP2 + 2].copy_from_slice(&3u16.to_le_bytes());
        // asm_offset self-relative from OFF_ASM_OFFSET.
        let asm_rel = (code_abs - OFF_ASM_OFFSET) as u32;
        b[OFF_ASM_OFFSET..OFF_ASM_OFFSET + 4].copy_from_slice(&asm_rel.to_le_bytes());
        // literal_offset points at end-of-code (no literals) self-relative from 0x74.
        let lit_rel = (total - OFF_LITERAL_OFFSET) as u32;
        b[OFF_LITERAL_OFFSET..OFF_LITERAL_OFFSET + 4].copy_from_slice(&lit_rel.to_le_bytes());

        // Parameter table.
        for (i, p) in params.iter().enumerate() {
            let off = params_abs + i * PARAM_ENTRY;
            let name_rel = (name_abs[i] as i64 - off as i64) as i32;
            b[off..off + 4].copy_from_slice(&name_rel.to_le_bytes());
            let packed = (p.category as u32 & 0xf)
                | ((p.ptype as u32 & 0xf) << 4)
                | ((p.comp as u32 & 0xf) << 8)
                | ((p.container as u32 & 0xf) << 12);
            b[off + 4..off + 8].copy_from_slice(&packed.to_le_bytes());
            b[off + 8..off + 12].copy_from_slice(&p.array.to_le_bytes());
            b[off + 12..off + 16].copy_from_slice(&p.res.to_le_bytes());
        }
        // String pool.
        b[strpool_abs..strpool_abs + strpool.len()].copy_from_slice(&strpool);
        // Code.
        for (i, w) in code.iter().enumerate() {
            let off = code_abs + i * 8;
            b[off..off + 8].copy_from_slice(&w.to_le_bytes());
        }
        b
    }

    /// The secondary program is stated three times over - a count and a start/end offset pair -
    /// and the parser only accepts a blob where all three agree, because the alternative is
    /// slicing an instruction stream out of the wrong bytes and running it into the SA bank.
    #[test]
    fn secondary_program_is_sliced_from_the_agreeing_count_and_offsets() {
        let code = [0x3880_0502_8204_0800u64, 0x4081_0d46_a680_2200u64, 0x1000_0000_0000_0001u64];
        let base = build_blob(false, &[], &code);
        let code_abs = base.len() - code.len() * 8;

        // Designate the first TWO code words as the secondary program.
        let patched = |count: u32, end_abs: usize| {
            let mut b = base.clone();
            b[OFF_SECONDARY_COUNT..OFF_SECONDARY_COUNT + 4].copy_from_slice(&count.to_le_bytes());
            let start_rel = (code_abs - OFF_SECONDARY_OFFSET) as u32;
            b[OFF_SECONDARY_OFFSET..OFF_SECONDARY_OFFSET + 4]
                .copy_from_slice(&start_rel.to_le_bytes());
            let end_rel = (end_abs - OFF_SECONDARY_END_OFFSET) as u32;
            b[OFF_SECONDARY_END_OFFSET..OFF_SECONDARY_END_OFFSET + 4]
                .copy_from_slice(&end_rel.to_le_bytes());
            b
        };

        let p = Program::parse(&patched(2, code_abs + 16)).expect("parse");
        assert_eq!(p.secondary_code, code[..2]);

        // A count that does not match the offset span is a layout this parser does not
        // understand - refuse the program rather than decode a mis-sliced stream.
        assert_eq!(
            Program::parse(&patched(3, code_abs + 16)).unwrap_err(),
            ParseError::SecondaryCodeInconsistent
        );
        // ... and an end offset past the blob is likewise refused.
        assert_eq!(
            Program::parse(&patched(2, base.len() + 8)).unwrap_err(),
            ParseError::SecondaryCodeInconsistent
        );
    }

    #[test]
    fn parses_header_params_and_code() {
        let params = [
            P { name: "AlbedoTexture", category: 2, ptype: 0, comp: 1, container: 0, array: 1, res: 3 },
            P { name: "Primarytint", category: 1, ptype: 1, comp: 3, container: 0, array: 1, res: 8 },
            P { name: "vPosition", category: 0, ptype: 0, comp: 4, container: 0, array: 1, res: 0 },
        ];
        let code = [0x1800_0000_dead_beefu64, 0x1000_0000_0000_0001u64];
        let blob = build_blob(true, &params, &code);

        let p = Program::parse(&blob).expect("parse");
        assert_eq!(p.kind, ProgramKind::Fragment);
        assert_eq!((p.major, p.minor), (1, 4));
        assert_eq!(p.size as usize, blob.len());
        assert_eq!(p.primary_reg_count, 7);
        assert_eq!(p.secondary_reg_count, 12);
        assert_eq!(p.temp_reg_count, 5); // max(5,3)
        assert_eq!(p.code, code);
        assert_eq!(p.parameters.len(), 3);

        assert!(p.secondary_code.is_empty(), "a count of 0 means no secondary program");

        let tex = &p.parameters[0];
        assert_eq!(tex.name, "AlbedoTexture");
        assert_eq!(tex.category, ParamCategory::Sampler);
        assert_eq!(tex.resource_index, 3);

        let tint = &p.parameters[1];
        assert_eq!(tint.name, "Primarytint");
        assert_eq!(tint.category, ParamCategory::Uniform);
        assert_eq!(tint.ptype, ParamType::F16);
        assert_eq!(tint.component_count, 3);
        assert_eq!(tint.resource_index, 8);

        assert_eq!(p.parameters[2].category, ParamCategory::Attribute);

        // Sampler helper.
        let samplers: Vec<_> = p.samplers().collect();
        assert_eq!(samplers, vec![(3u32, "AlbedoTexture")]);
    }

    #[test]
    fn vertex_kind_and_hash_stability() {
        let code = [0u64];
        let a = build_blob(false, &[], &code);
        let p = Program::parse(&a).unwrap();
        assert_eq!(p.kind, ProgramKind::Vertex);
        // Same bytes hash identically; a changed byte changes the hash.
        let p2 = Program::parse(&a).unwrap();
        assert_eq!(p.hash, p2.hash);
        let mut b = a.clone();
        *b.last_mut().unwrap() ^= 0xff;
        assert_ne!(Program::parse(&b).unwrap().hash, p.hash);
    }

    #[test]
    fn rejects_bad_magic_and_short() {
        assert_eq!(Program::parse(&[]).unwrap_err(), ParseError::TooSmall);
        let mut blob = build_blob(true, &[], &[0u64]);
        blob[0] = 0;
        assert!(matches!(Program::parse(&blob).unwrap_err(), ParseError::BadMagic(_)));
    }

    #[test]
    fn rejects_out_of_range_param_table() {
        let mut blob = build_blob(true, &[], &[0u64]);
        // Corrupt param_count to something absurd.
        blob[OFF_PARAM_COUNT..OFF_PARAM_COUNT + 4].copy_from_slice(&9999u32.to_le_bytes());
        assert_eq!(Program::parse(&blob).unwrap_err(), ParseError::ParamTableOutOfRange);
    }

    #[test]
    fn component_bytes_packing() {
        assert_eq!(ParamType::F32.component_bytes(), Some(4));
        assert_eq!(ParamType::F16.component_bytes(), Some(2));
        assert_eq!(ParamType::Aggregate.component_bytes(), None);
    }

    #[test]
    fn varying_usage_decodes_semantic_nibble() {
        // Real attribute_info values from frag_82d27fb0 (validated: TEXCOORD1/2/3/0).
        assert_eq!(varying_usage_from_attribute_info(0x2cc0_1100), VaryingUsage::TexCoord(1));
        assert_eq!(varying_usage_from_attribute_info(0x2cc0_2903), VaryingUsage::TexCoord(2));
        assert_eq!(varying_usage_from_attribute_info(0x0cc0_300f), VaryingUsage::TexCoord(3));
        assert_eq!(varying_usage_from_attribute_info(0x0ec0_000f), VaryingUsage::TexCoord(0));
        assert_eq!(varying_usage_from_attribute_info(0x0000_a000), VaryingUsage::Color0);
        assert_eq!(varying_usage_from_attribute_info(0x0000_b000), VaryingUsage::Color1);
        assert_eq!(varying_usage_from_attribute_info(0x0000_c000), VaryingUsage::Fog);
        assert_eq!(varying_usage_from_attribute_info(0x0000_d000), VaryingUsage::Position);
        // A sprite/point coordinate (bit 0x40000000) is fragment-generated, not a vertex output.
        assert_eq!(varying_usage_from_attribute_info(0x4000_1000), VaryingUsage::Unknown(0x4000_1000));
    }

    /// Build a minimal but structurally valid FRAGMENT blob whose varyings block carries a
    /// descriptor array (each entry the four raw descriptor words `(attribute_info,
    /// resource_index, size, component_info)`), so the interpolant parse is exercised end to end
    /// without the private dumps.
    fn build_frag_with_varyings(descs: &[(u32, u32, u32, u32)]) -> Vec<u8> {
        let header = 0x80usize;
        let block = header; // varyings block right after the header
        let arr = block + 0x14; // descriptor array (self-relative from block+0x10)
        let arr_end = arr + descs.len() * 16;
        let params = arr_end; // empty parameter table
        let code_off = (params + 7) & !7;
        let total = code_off + 8; // one USSE instruction
        let mut b = vec![0u8; total];

        b[0..4].copy_from_slice(&GXP_MAGIC.to_le_bytes());
        b[OFF_MAJOR] = 1;
        b[OFF_MINOR] = 4;
        b[OFF_SIZE..OFF_SIZE + 4].copy_from_slice(&(total as u32).to_le_bytes());
        b[OFF_TYPE] = 0x01; // fragment
        // param_count = 0 (zeroed). params_offset self-relative from 0x28.
        b[OFF_PARAMS_OFFSET..OFF_PARAMS_OFFSET + 4]
            .copy_from_slice(&((params - OFF_PARAMS_OFFSET) as u32).to_le_bytes());
        // varyings_offset self-relative from 0x2C.
        b[OFF_VARYINGS_OFFSET..OFF_VARYINGS_OFFSET + 4]
            .copy_from_slice(&((block - OFF_VARYINGS_OFFSET) as u32).to_le_bytes());
        // asm_offset self-relative from 0x40; literal offset -> end (no literals).
        b[OFF_ASM_OFFSET..OFF_ASM_OFFSET + 4]
            .copy_from_slice(&((code_off - OFF_ASM_OFFSET) as u32).to_le_bytes());
        b[OFF_LITERAL_OFFSET..OFF_LITERAL_OFFSET + 4]
            .copy_from_slice(&((total - OFF_LITERAL_OFFSET) as u32).to_le_bytes());

        // Varyings block: count at +0x0C, self-relative descriptor-array offset at +0x10.
        b[block + 0x0c..block + 0x0e].copy_from_slice(&(descs.len() as u16).to_le_bytes());
        b[block + 0x10..block + 0x14].copy_from_slice(&((arr - (block + 0x10)) as u32).to_le_bytes());
        for (i, &(ai, ri, sz, ci)) in descs.iter().enumerate() {
            let d = arr + i * 16;
            b[d..d + 4].copy_from_slice(&ai.to_le_bytes());
            b[d + 4..d + 8].copy_from_slice(&ri.to_le_bytes());
            b[d + 8..d + 12].copy_from_slice(&sz.to_le_bytes());
            b[d + 12..d + 16].copy_from_slice(&ci.to_le_bytes());
        }
        b
    }

    #[test]
    fn parses_fragment_interpolant_descriptor_array() {
        // The first three descriptors of frag_82d27fb0, verbatim: TEXCOORD1 with a prefetched
        // shadMap sample (unit 13) fed by TEXCOORD0, then TEXCOORD2 with a prefetched
        // LiveryAlbedo sample (unit 0) fed by TEXCOORD3 and flagged as the last of the fetch
        // sequence, then a plain F32 TEXCOORD3. PA bases accumulate by SPAN - 2 data registers
        // plus 2 for a prefetched sample - so they run 0, 4, 8.
        let b = build_frag_with_varyings(&[
            (0x2cc0_1100, 13, 0x50, 0x20),
            (0x2cc0_2903, 0, 0x50, 0x20),
            (0x0cc0_300f, 0, 0x30, 0x00),
        ]);
        let p = Program::parse(&b).expect("parse");
        assert_eq!(p.kind, ProgramKind::Fragment);
        assert_eq!(
            p.interpolants,
            vec![
                // The 0x2000_0000 precision bit marks the F16 interpolants (two packed halves
                // per PA register); the 0x0c.. entry is F32.
                Interpolant {
                    usage: VaryingUsage::TexCoord(1),
                    pa_base: 0,
                    register_count: 2,
                    span: 4,
                    half: true,
                    prefetch: Some(SamplePrefetch { unit: 13, source_texcoord: 0, last: false }),
                    prefetch_regs: 2,
                },
                Interpolant {
                    usage: VaryingUsage::TexCoord(2),
                    pa_base: 4,
                    register_count: 2,
                    span: 4,
                    half: true,
                    prefetch: Some(SamplePrefetch { unit: 0, source_texcoord: 3, last: true }),
                    prefetch_regs: 2,
                },
                Interpolant {
                    usage: VaryingUsage::TexCoord(3),
                    pa_base: 8,
                    register_count: 4,
                    span: 4,
                    half: false,
                    prefetch: None,
                    // No prefetch rides along, so this only reflects `size` bit 6 (clear here).
                    prefetch_regs: 1,
                },
            ]
        );
        // The prefetched samples land immediately after each interpolant's own data.
        assert_eq!(p.interpolants[0].prefetch_base(), Some(2));
        assert_eq!(p.interpolants[1].prefetch_base(), Some(6));
        assert_eq!(p.interpolants[2].prefetch_base(), None);
    }

    #[test]
    fn descriptors_whose_prefetch_flags_disagree_are_not_decoded() {
        // TWO independent fields state whether a prefetched sample rides along, and they agree
        // on every descriptor of every captured blob. A descriptor where they do not is not the
        // layout decoded here, so the whole block yields nothing and the renderer falls back -
        // rather than shift every later interpolant's PA base by two and silently feed the
        // shader the wrong registers.
        //
        // `size` bit 6 is deliberately NOT one of them: it disagrees with the other two on
        // fourteen real descriptors that are unambiguously prefetches, and demanding it agree
        // discarded those programs entirely. The third case below pins that - the same
        // descriptor that used to be rejected for lacking bit 6 must now decode.
        for bad in [
            (0x2cc0_1100u32, 13u32, 0x50u32, 0x00u32), // attribute_info, no component_info
            (0x2cc0_1000, 13, 0x50, 0x20),             // component_info, no attribute_info
        ] {
            assert!(
                Program::parse(&build_frag_with_varyings(&[bad])).unwrap().interpolants.is_empty(),
                "{bad:x?} should not decode"
            );
        }
        // A prefetch descriptor WITHOUT `size` bit 6 decodes - it is a real prefetch, and the
        // corpus has fourteen of them. This is the regression guard for the class of program
        // that was being thrown away whole.
        let no_size_bit6 = build_frag_with_varyings(&[(0x2cc0_1100, 13, 0x10, 0x20)]);
        let decoded = Program::parse(&no_size_bit6).unwrap();
        assert_eq!(decoded.interpolants.len(), 1, "a prefetch without size bit 6 must decode");
        assert!(decoded.interpolants[0].prefetch.is_some());

        // A descriptor with no prefetch names neither a source texcoord nor a texture unit.
        let unit_without_prefetch = build_frag_with_varyings(&[(0x0cc0_300f, 13, 0x30, 0x00)]);
        assert!(Program::parse(&unit_without_prefetch).unwrap().interpolants.is_empty());
        let source_without_prefetch = build_frag_with_varyings(&[(0x0cc0_3003, 0, 0x30, 0x00)]);
        assert!(Program::parse(&source_without_prefetch).unwrap().interpolants.is_empty());
    }

    #[test]
    fn vertex_program_has_no_interpolants() {
        // A vertex blob (no varyings_offset set) yields no fragment interpolants.
        let b = build_blob(false, &[], &[0u64]);
        assert!(Program::parse(&b).unwrap().interpolants.is_empty());
    }
}
