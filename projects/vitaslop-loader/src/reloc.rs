//! SCE module relocations (the `PT_SCE_RELA`, `p_type == 0x6000_0000`, segment).
//!
//! A velf is `ET_SCE_RELEXEC`: its loadable segments are laid out relative to a
//! nominal link base (every Vita module links at `0x8100_0000`), and the OS
//! loader is expected to place the module wherever it likes and apply a table of
//! relocations to fix up every absolute address the code and data embed. A single
//! self-contained module can be loaded at its link base with the fixups applied
//! in place; several modules that must coexist (a game's `eboot.bin` plus its
//! `*.suprx`) each get a distinct base, so relocation is mandatory to place them.
//!
//! The entry format is the SCE variable-length encoding. It is recovered
//! clean-room from the MIT-licensed vita-toolchain `SCE_Rel` union (the
//! authoritative producer of this exact byte layout): a 4-bit tag in the low
//! nibble of the first word selects a short (8-byte) or long (12-byte) entry.
//! Each entry names a *symbol segment* (whose runtime base is the relocation's
//! `S`), a *data segment* and offset (where the fixup is written, `P`), an ARM
//! relocation `code`, and an `addend`. A long entry can additionally piggyback a
//! second fixup (`code2`/`dist2`), which is how a `MOVW`/`MOVT` immediate pair is
//! encoded in one entry.

/// The `p_type` of the SCE relocation segment.
pub const PT_SCE_RELA: u32 = 0x6000_0000;

/// ARM relocation codes that appear in Vita modules. Values are the standard
/// `R_ARM_*` ELF constants (the `code` field is 8-bit).
pub mod code {
    pub const NONE: u8 = 0;
    pub const ABS32: u8 = 2;
    pub const REL32: u8 = 3;
    pub const THM_CALL: u8 = 10;
    pub const CALL: u8 = 28;
    pub const JUMP24: u8 = 29;
    pub const THM_JUMP24: u8 = 30;
    pub const TARGET1: u8 = 38;
    pub const V4BX: u8 = 40;
    pub const TARGET2: u8 = 41;
    pub const PREL31: u8 = 42;
    pub const MOVW_ABS_NC: u8 = 43;
    pub const MOVT_ABS: u8 = 44;
    pub const THM_MOVW_ABS_NC: u8 = 47;
    pub const THM_MOVT_ABS: u8 = 48;
    pub const THM_JUMP11: u8 = 102;
    pub const THM_JUMP8: u8 = 103;
}

/// One decoded relocation fixup: write, at `sym_seg`-plus-`code`-computed value,
/// into `data_seg` at `offset`, with `addend`.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub struct Reloc {
    /// ARM relocation code (`R_ARM_*`).
    pub code: u8,
    /// Segment index whose runtime base is the symbol value `S`.
    pub sym_seg: u8,
    /// Segment index the fixup is written into.
    pub data_seg: u8,
    /// Byte offset of the fixup within `data_seg`.
    pub offset: u32,
    /// Constant added to `S` to form the target address.
    pub addend: u32,
}

/// Decode every relocation entry out of a raw SCE relocation blob (the bytes of a
/// `PT_SCE_RELA` segment).
///
/// The blob is a tight sequence of variable-length entries. A short entry is 8
/// bytes; a long entry is 12 and may carry a second piggybacked fixup. An entry
/// whose format nibble we do not model is a HARD ERROR ([`crate::Error::UnknownRelocFormat`]):
/// returning the fixups decoded so far would silently drop every following
/// relocation, leaving zeroed dangling pointers that only fault much later (a
/// wrong-answer far from its cause). Fail loudly at the offending entry instead.
pub fn decode(blob: &[u8]) -> Result<Vec<Reloc>, crate::Error> {
    let mut out = Vec::new();
    let mut o = 0usize;
    let rd = |b: &[u8], at: usize| u32::from_le_bytes([b[at], b[at + 1], b[at + 2], b[at + 3]]);
    while o + 8 <= blob.len() {
        let w0 = rd(blob, o);
        let format = w0 & 0xF;
        match format {
            // Short entry (8 bytes): the common case.
            1 => {
                let w1 = rd(blob, o + 4);
                let sym_seg = ((w0 >> 4) & 0xF) as u8;
                let code = ((w0 >> 8) & 0xFF) as u8;
                let data_seg = ((w0 >> 16) & 0xF) as u8;
                let offset_lo = (w0 >> 20) & 0xFFF; // 12 bits
                let offset_hi = w1 & 0xF_FFFF; // 20 bits
                let addend = (w1 >> 20) & 0xFFF; // 12 bits
                out.push(Reloc {
                    code,
                    sym_seg,
                    data_seg,
                    offset: offset_lo | (offset_hi << 12),
                    addend,
                });
                o += 8;
            }
            // Long entry (12 bytes): full 32-bit offset and addend, and an
            // optional second fixup (`code2` at `offset + dist2 * 2`) that shares
            // the same symbol segment and addend - the MOVW/MOVT pairing.
            0 => {
                if o + 12 > blob.len() {
                    break;
                }
                let w1 = rd(blob, o + 4);
                let w2 = rd(blob, o + 8);
                let sym_seg = ((w0 >> 4) & 0xF) as u8;
                let code = ((w0 >> 8) & 0xFF) as u8;
                let data_seg = ((w0 >> 16) & 0xF) as u8;
                let code2 = ((w0 >> 20) & 0xFF) as u8;
                let dist2 = (w0 >> 28) & 0xF;
                let addend = w1;
                let offset = w2;
                out.push(Reloc { code, sym_seg, data_seg, offset, addend });
                if code2 != code::NONE {
                    out.push(Reloc {
                        code: code2,
                        sym_seg,
                        data_seg,
                        offset: offset.wrapping_add(dist2 * 2),
                        addend,
                    });
                }
                o += 12;
            }
            // Any other tag is a format we do not model. Fail loudly - dropping the
            // rest of the blob would corrupt the image with dangling pointers.
            _ => return Err(crate::Error::UnknownRelocFormat(format as u8, o)),
        }
    }
    Ok(out)
}
