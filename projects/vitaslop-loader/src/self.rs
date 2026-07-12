//! SELF/fSELF container unwrapping: turns an `eboot.bin` (the wrapper a real
//! Vita title ships as) back into the plain ELF/velf the [`load`](crate::load)
//! path already understands.
//!
//! A SELF is a small SCE header plus a copy of the program headers, a
//! per-segment `segment_info` table, and control info, followed by the segment
//! payloads. `vita-make-fself` produces the homebrew form (fSELF): unencrypted,
//! and by default uncompressed, in which case the whole original ELF is copied
//! verbatim after the header. We reconstruct the inner ELF from the segment
//! table so both the verbatim and the (future) compressed layout are handled by
//! the same code.
//!
//! We deliberately do NOT carry a crypto dependency: a real retail eboot has
//! encrypted segments we cannot (and by license posture do not want to)
//! decrypt, so those return a clear error. Compressed segments (the common
//! `vita-make-fself -c` homebrew form) ARE handled, via the loader's own
//! dependency-free [`inflate`](crate::inflate) - no zlib crate, so the loader
//! stays wasm-clean.
//!
//! Layout facts are from the MIT `vita-toolchain` (`self.h`,
//! `vita-make-fself.c`), not from any copyleft source.

use crate::inflate;
use crate::{Error, Reader};

/// SCE container magic: the bytes `"SCE\0"`.
const SCE_MAGIC: &[u8; 4] = b"SCE\0";
/// ELF32 program-header entry size.
const PHDR_SIZE: usize = 0x20;
/// `segment_info` entry size (offset/length/compression/encryption, all u64).
const SEGMENT_INFO_SIZE: usize = 0x20;
/// ELF32 header size (the inner ELF's header we copy out).
const EHDR_SIZE: usize = 52;

/// `segment_info.compression`: 1 = stored, 2 = zlib-compressed.
const COMPRESSION_NONE: u64 = 1;
const COMPRESSION_ZLIB: u64 = 2;
/// `segment_info.encryption`: 2 = plain (fSELF); 1 would be encrypted (retail).
const ENCRYPTION_PLAIN: u64 = 2;

/// True if `bytes` begins with the SCE container magic.
pub fn is_self(bytes: &[u8]) -> bool {
    bytes.len() >= 4 && &bytes[0..4] == SCE_MAGIC
}

/// Reconstruct the inner ELF/velf image from a SELF/fSELF container.
///
/// Handles the unencrypted fSELF that `vita-make-fself` emits, compressed or
/// not. Returns [`Error::EncryptedSelf`] for encrypted segments (retail titles)
/// and [`Error::CompressedSelf`] if a compressed segment fails to inflate.
pub fn unwrap_self(bytes: &[u8]) -> Result<Vec<u8>, Error> {
    let r = Reader { bytes };
    if !is_self(bytes) {
        return Err(Error::NotElf);
    }

    // SCE_header (packed): the u64 offsets we need to find the inner ELF, its
    // program headers, and the per-segment payload table.
    let elf_offset = r.u64(64)? as usize;
    let phdr_offset = r.u64(72)? as usize;
    let section_info_offset = r.u64(88)? as usize;

    // Inner ELF header: e_phoff/e_phnum place the program headers within the
    // reconstructed image; e_type is validated later by `load`.
    if bytes.get(elf_offset..elf_offset + EHDR_SIZE).is_none() {
        return Err(Error::OutOfBounds("self elf header"));
    }
    let e_phoff = r.u32(elf_offset + 28)? as usize;
    let e_phnum = r.u16(elf_offset + 44)? as usize;

    // First pass: size the output image = the furthest byte any segment, or the
    // header block, reaches. Segment file bytes land at their ELF p_offset.
    let mut out_size = e_phoff + e_phnum * PHDR_SIZE;
    out_size = out_size.max(EHDR_SIZE);
    for i in 0..e_phnum {
        let ph = phdr_offset + i * PHDR_SIZE;
        let p_offset = r.u32(ph + 4)? as usize;
        let p_filesz = r.u32(ph + 16)? as usize;
        out_size = out_size.max(p_offset + p_filesz);
    }

    let mut out = vec![0u8; out_size];

    // Copy each segment's payload into place, decoding the segment table.
    for i in 0..e_phnum {
        let ph = phdr_offset + i * PHDR_SIZE;
        let p_offset = r.u32(ph + 4)? as usize;
        let p_filesz = r.u32(ph + 16)? as usize;

        let si = section_info_offset + i * SEGMENT_INFO_SIZE;
        let seg_off = r.u64(si)? as usize;
        let seg_len = r.u64(si + 8)? as usize;
        let compression = r.u64(si + 16)?;
        let encryption = r.u64(si + 24)?;

        if seg_len == 0 {
            continue; // .bss-style segment, no file backing.
        }
        if encryption != ENCRYPTION_PLAIN {
            return Err(Error::EncryptedSelf);
        }

        let src = bytes
            .get(seg_off..seg_off + seg_len)
            .ok_or(Error::OutOfBounds("self segment"))?;

        match compression {
            COMPRESSION_NONE => {
                out.get_mut(p_offset..p_offset + seg_len)
                    .ok_or(Error::OutOfBounds("self segment dest"))?
                    .copy_from_slice(src);
            }
            COMPRESSION_ZLIB => {
                // The decompressed segment is exactly p_filesz bytes.
                let seg = inflate::zlib_inflate(src, p_filesz)
                    .map_err(|_| Error::CompressedSelf)?;
                if seg.len() != p_filesz {
                    return Err(Error::CompressedSelf);
                }
                out.get_mut(p_offset..p_offset + p_filesz)
                    .ok_or(Error::OutOfBounds("self segment dest"))?
                    .copy_from_slice(&seg);
            }
            _ => return Err(Error::CompressedSelf),
        }
    }

    // Overlay the ELF header and program headers last, so the header block is
    // always the reconstructed one even if a segment nominally covers offset 0.
    out[0..EHDR_SIZE].copy_from_slice(&bytes[elf_offset..elf_offset + EHDR_SIZE]);
    let phdrs = bytes
        .get(phdr_offset..phdr_offset + e_phnum * PHDR_SIZE)
        .ok_or(Error::OutOfBounds("self phdrs"))?;
    out.get_mut(e_phoff..e_phoff + e_phnum * PHDR_SIZE)
        .ok_or(Error::OutOfBounds("self phdrs dest"))?
        .copy_from_slice(phdrs);

    Ok(out)
}
