//! SELF -> ELF for encrypted retail application containers ("self2elf").
//!
//! The plaintext `eboot.bin` / `*.suprx` that come out of the outer container are
//! still SCE SELF wrappers: an SCE header, a copy of the ELF program headers, a
//! per-segment table, encrypted control/metadata, and AES-128-CTR-encrypted
//! (optionally zlib-compressed) code segments. This module unwraps that to the
//! plain velf the loader parses.
//!
//! The plaintext fSELF that homebrew ships (`vita-make-fself`) is handled without
//! crypto by [`vitaslop_loader::self_`]; this is the retail counterpart, which
//! needs the NPDRM klicensee and the published metadata keys.
//!
//! Algorithm ported from the MIT-licensed `sceutils` (`self2elf.py`,
//! `sceutils.py`); the metadata/NPDRM key constants are the published psdevwiki
//! values (see [`super::keys`]). Scope: retail NPDRM **application** SELFs
//! (`self_type` APP) with ARM segments - the only kind a game ships.

use super::{Error, Reader};
use aes::cipher::{BlockDecryptMut, KeyIvInit, StreamCipher, block_padding::NoPadding};

type Aes128CbcDec = cbc::Decryptor<aes::Aes128>;
type Aes256CbcDec = cbc::Decryptor<aes::Aes256>;
type Aes128Ctr = ctr::Ctr128BE<aes::Aes128>;

/// SCE container magic `"SCE\0"` as a little-endian u32.
const SCE_MAGIC: u32 = 0x0045_4353;
/// `AppInfoHeader.self_type` for an application.
const SELF_TYPE_APP: u32 = 0x08;
/// `SegmentInfo.plaintext` / `.compressed` `SecureBool` values.
const SECURE_NO: u32 = 1;
const SECURE_YES: u32 = 2;
/// `MetadataSection.encryption` == AES-128-CTR.
const ENC_AES128CTR: u32 = 3;

const ELF_HDR_SIZE: usize = 52;
const PHDR_SIZE: usize = 32;
const SEGINFO_SIZE: usize = 32;
const METADATA_SIG_SKIP: usize = 48;
const METADATA_INFO_SIZE: usize = 64;
const METADATA_HDR_SIZE: usize = 32;
const METADATA_SEC_SIZE: usize = 48;

/// Unwrap an encrypted application SELF into its inner ELF/velf image.
///
/// `klicensee` is the 16-byte license secret (from the RIF). Returns the
/// reconstructed ELF bytes, ready for [`vitaslop_loader::load`].
pub fn self2elf(bytes: &[u8], klicensee: &[u8; 16]) -> Result<Vec<u8>, Error> {
    let r = Reader::new(bytes);
    if r.u32(0)? != SCE_MAGIC {
        return Err(Error::BadMagic("SELF SCE magic"));
    }
    let key_revision = bytes.get(9).copied().ok_or(Error::OutOfBounds("key_rev"))? as usize;
    let metadata_offset = r.u32(12)? as usize;
    let header_length = r.u64(16)? as usize;

    // SelfHeader at file offset 32; its fields are offsets from that base.
    const SH: usize = 32;
    let appinfo_offset = r.u64(SH + 24)? as usize;
    let elf_offset = r.u64(SH + 32)? as usize;
    let phdr_offset = r.u64(SH + 40)? as usize;
    let segment_info_offset = r.u64(SH + 56)? as usize;

    // AppInfoHeader: only self_type matters for the path we support.
    let self_type = r.u32(appinfo_offset + 12)?;
    if self_type != SELF_TYPE_APP {
        return Err(Error::BadMagic("SELF not an application"));
    }

    // ELF header (copied verbatim) gives the program-header count.
    if bytes.get(elf_offset..elf_offset + ELF_HDR_SIZE).is_none() {
        return Err(Error::OutOfBounds("SELF elf header"));
    }
    let e_phnum = r.u16(elf_offset + 44)? as usize;

    // Per-segment: copy the program header, read the segment-info entry, and note
    // whether any segment is encrypted.
    let mut phdrs = Vec::with_capacity(e_phnum);
    let mut seginfos = Vec::with_capacity(e_phnum);
    let mut encrypted = false;
    for i in 0..e_phnum {
        let ph = phdr_offset + i * PHDR_SIZE;
        let p_offset = r.u32(ph + 4)? as usize;
        let p_filesz = r.u32(ph + 16)? as usize;
        phdrs.push((p_offset, p_filesz));

        let si = segment_info_offset + i * SEGINFO_SIZE;
        let offset = r.u64(si)? as usize;
        let size = r.u64(si + 8)? as usize;
        let compressed = r.u32(si + 16)?;
        let plaintext = r.u32(si + 24)?;
        if plaintext == SECURE_NO {
            encrypted = true;
        }
        seginfos.push((offset, size, compressed, plaintext));
    }

    // Decrypt the metadata to recover per-segment AES-CTR key/iv, when encrypted.
    let seg_keys = if encrypted {
        Some(decrypt_metadata(
            bytes,
            metadata_offset,
            header_length,
            key_revision,
            klicensee,
        )?)
    } else {
        None
    };

    // Reassemble like sceutils: write the ELF header and program headers at the
    // start, then place each segment's decrypted payload AT its `p_offset`
    // (seek-and-write - segments need not be ordered, and may sit anywhere past
    // the header). Gaps are left zero.
    let mut out: Vec<u8> = Vec::new();
    out.extend_from_slice(&bytes[elf_offset..elf_offset + ELF_HDR_SIZE]);
    let phdr_block = bytes
        .get(phdr_offset..phdr_offset + e_phnum * PHDR_SIZE)
        .ok_or(Error::OutOfBounds("SELF phdrs"))?;
    out.extend_from_slice(phdr_block);

    for i in 0..e_phnum {
        let (p_offset, p_filesz) = phdrs[i];
        if p_filesz == 0 {
            continue;
        }
        let (offset, size, compressed, plaintext) = seginfos[i];
        let mut dat = bytes
            .get(offset..offset + size)
            .ok_or(Error::OutOfBounds("SELF segment payload"))?
            .to_vec();

        if plaintext == SECURE_NO {
            let keys = seg_keys.as_ref().ok_or(Error::MissingKey("segment key"))?;
            let (key, iv) = keys
                .get(&i)
                .ok_or(Error::IntegrityCheck("no key for encrypted segment"))?;
            let mut c = Aes128Ctr::new(key.into(), iv.into());
            c.apply_keystream(&mut dat);
        }
        if compressed == SECURE_YES {
            dat = vitaslop_loader::inflate::zlib_inflate(&dat, p_filesz)
                .map_err(|_| Error::IntegrityCheck("SELF segment inflate"))?;
        }

        let end = p_offset + dat.len();
        if out.len() < end {
            out.resize(end, 0);
        }
        out[p_offset..end].copy_from_slice(&dat);
    }

    Ok(out)
}

/// Decrypt the SELF metadata blob and return the per-segment AES-CTR key+iv,
/// keyed by segment index.
fn decrypt_metadata(
    bytes: &[u8],
    metadata_offset: usize,
    header_length: usize,
    key_revision: usize,
    klicensee: &[u8; 16],
) -> Result<std::collections::HashMap<usize, ([u8; 16], [u8; 16])>, Error> {
    use super::keys;

    let (md_key, md_iv) =
        keys::metadata_app(key_revision).ok_or(Error::MissingKey("metadata key revision"))?;
    let np_key = keys::SELF_NPDRM_APP[keys::self_npdrm_row(key_revision as u8)];
    let zero_iv = [0u8; 16];

    // The metadata blob follows the signature area at metadata_offset+48 and runs
    // to header_length. blob[0:64] is the encrypted MetadataInfo; the rest is the
    // encrypted header + sections + key vault.
    let blob_start = metadata_offset + METADATA_SIG_SKIP;
    let blob = bytes
        .get(blob_start..header_length)
        .ok_or(Error::OutOfBounds("SELF metadata blob"))?;
    if blob.len() < METADATA_INFO_SIZE {
        return Err(Error::OutOfBounds("SELF metadata info"));
    }

    // NPDRM: predecrypt the klicensee (AES-128-CBC, zero IV), then use the result
    // as an AES-128 key to CBC-decrypt the MetadataInfo, then the metadata key
    // (AES-256-CBC) finishes it.
    let predec = aes128_cbc(&np_key, &zero_iv, klicensee)?;
    let predec: [u8; 16] = predec[..16].try_into().unwrap();
    let dec_in = aes128_cbc(&predec, &zero_iv, &blob[..METADATA_INFO_SIZE])?;
    let info = aes256_cbc(&md_key, &md_iv, &dec_in)?;

    // MetadataInfo: key at 0, iv at 32; the two 16-byte gaps must be zero, which
    // is the "right key" check.
    if info[16..32] != [0u8; 16] || info[48..64] != [0u8; 16] {
        return Err(Error::IntegrityCheck("SELF metadata info padding"));
    }
    let mi_key: [u8; 16] = info[0..16].try_into().unwrap();
    let mi_iv: [u8; 16] = info[32..48].try_into().unwrap();

    // The rest of the blob, decrypted with the recovered key.
    let body = aes128_cbc(&mi_key, &mi_iv, &blob[METADATA_INFO_SIZE..])?;
    let br = Reader::new(&body);
    let section_count = br.u32(12)? as usize;
    let key_count = br.u32(16)? as usize;

    let vault_start = METADATA_HDR_SIZE + section_count * METADATA_SEC_SIZE;
    let vault = |x: i32| -> Result<[u8; 16], Error> {
        if x < 0 || x as usize >= key_count {
            return Err(Error::IntegrityCheck("SELF key vault index"));
        }
        let at = vault_start + x as usize * 16;
        body.get(at..at + 16)
            .ok_or(Error::OutOfBounds("SELF key vault"))
            .map(|s| s.try_into().unwrap())
    };

    let mut out = std::collections::HashMap::new();
    for s in 0..section_count {
        let base = METADATA_HDR_SIZE + s * METADATA_SEC_SIZE;
        let seg_idx = br.u32(base + 20)? as i32;
        let encryption = br.u32(base + 32)?;
        let key_idx = br.u32(base + 36)? as i32;
        let iv_idx = br.u32(base + 40)? as i32;
        if encryption == ENC_AES128CTR {
            out.insert(seg_idx.max(0) as usize, (vault(key_idx)?, vault(iv_idx)?));
        }
    }
    Ok(out)
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::ingest::filesdb::FilesDb;
    use crate::ingest::pfs::{PfsCrypto, PfsImage};
    use crate::ingest::pfscrypt::GameData;
    use crate::ingest::rif::Rif;
    use crate::ingest::testfix;
    use crate::ingest::unicv::UnicvDb;

    /// Full retail chain, offline: PFS-decrypt `eboot.bin` to its SELF, then
    /// self2elf to a plain ELF. Skips without the fixture.
    #[test]
    fn eboot_pfs_then_self2elf_to_elf() {
        let (Some(fdb), Some(ucv), Some(work), Some(eboot_ct)) = (
            testfix::read("sce_pfs/files.db"),
            testfix::read("sce_pfs/unicv.db"),
            testfix::read("sce_sys/package/work.bin"),
            testfix::read("eboot.bin"),
        ) else {
            return;
        };
        let img = PfsImage::new(
            FilesDb::parse(&fdb).unwrap(),
            UnicvDb::parse(&ucv).unwrap(),
        )
        .unwrap();
        let rif = Rif::parse(&work).unwrap();

        // PFS layer -> SELF.
        let ctx = img.file_ctx("eboot.bin", &rif.key).expect("eboot ctx");
        let self_bytes = GameData::from_klicensee(&rif.key)
            .decrypt_file(&ctx, &eboot_ct)
            .expect("pfs decrypt");
        assert_eq!(&self_bytes[..4], b"SCE\0");

        // SELF layer -> ELF.
        let elf = self2elf(&self_bytes, &rif.key).expect("self2elf");
        assert_eq!(&elf[..4], b"\x7fELF", "self2elf did not yield an ELF");
    }

    /// Diagnostic: run the decrypted eboot ELF through the loader and report what
    /// the real OlliOlli module needs (name, entry, imports by library). Ignored;
    /// run with `--ignored --nocapture`.
    #[test]
    #[ignore = "diagnostic: needs fixture"]
    fn probe_eboot_loader() {
        let (Some(fdb), Some(ucv), Some(work), Some(eboot_ct)) = (
            testfix::read("sce_pfs/files.db"),
            testfix::read("sce_pfs/unicv.db"),
            testfix::read("sce_sys/package/work.bin"),
            testfix::read("eboot.bin"),
        ) else {
            return;
        };
        let img = PfsImage::new(FilesDb::parse(&fdb).unwrap(), UnicvDb::parse(&ucv).unwrap())
            .unwrap();
        let rif = Rif::parse(&work).unwrap();
        let ctx = img.file_ctx("eboot.bin", &rif.key).unwrap();
        let self_bytes = GameData::from_klicensee(&rif.key)
            .decrypt_file(&ctx, &eboot_ct)
            .unwrap();
        let elf = self2elf(&self_bytes, &rif.key).unwrap();

        match vitaslop_loader::load(&elf) {
            Ok(m) => {
                eprintln!(
                    "loaded module '{}' nid={:#x} base={:#x} entry={:#x} segs={} imports={} init_ptrs={}",
                    m.name, m.module_nid, m.base, m.entry, m.segments.len(), m.imports.len(), m.init_pointers.len()
                );
                use std::collections::BTreeMap;
                let mut by_lib: BTreeMap<u32, usize> = BTreeMap::new();
                for imp in &m.imports {
                    *by_lib.entry(imp.library_nid).or_default() += 1;
                }
                for (lib, n) in by_lib {
                    eprintln!("  library {lib:#010x}: {n} imports");
                }
            }
            Err(e) => eprintln!("loader error: {e:?}"),
        }
    }
}

/// AES-128-CBC decrypt of block-aligned `data` (no padding).
fn aes128_cbc(key: &[u8; 16], iv: &[u8; 16], data: &[u8]) -> Result<Vec<u8>, Error> {
    let mut buf = data.to_vec();
    Aes128CbcDec::new(key.into(), iv.into())
        .decrypt_padded_mut::<NoPadding>(&mut buf)
        .map_err(|_| Error::IntegrityCheck("aes128-cbc length"))?;
    Ok(buf)
}

/// AES-256-CBC decrypt of block-aligned `data` (no padding).
fn aes256_cbc(key: &[u8; 32], iv: &[u8; 16], data: &[u8]) -> Result<Vec<u8>, Error> {
    let mut buf = data.to_vec();
    Aes256CbcDec::new(key.into(), iv.into())
        .decrypt_padded_mut::<NoPadding>(&mut buf)
        .map_err(|_| Error::IntegrityCheck("aes256-cbc length"))?;
    Ok(buf)
}
