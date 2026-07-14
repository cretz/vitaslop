//! PFS gamedata byte decryption: the concrete [`PfsCrypto`] for a read-only
//! (unicv.db) application image.
//!
//! The construction (validated end-to-end against a real title, fully offline):
//! - `drv_key` (16 bytes) = `F00D(klicensee)` = `AES-128-ECB-decrypt(klicensee,`
//!   [`keys::PFS_F00D_CONTRACT`]`)` - the secure-coprocessor keygen, reproduced in
//!   software from a public constant (see [`f00d_drv_key`] /
//!   [`GameData::from_klicensee`]).
//! - `tweak_key` (16 bytes) = `HMAC-SHA1(`[`keys::PFS_TWEAK_BASE`]`, dbseed)[..16]`
//!   - the per-file sector-IV mask (files with a `dbseed`, i.e. icv_version > 1).
//! - `secret` (20 bytes) = AES-128-CBC-CTS-encrypt of
//!   `HMAC-SHA1(`[`keys::PFS_INTEGRITY_BASE`]`, salt)` under `drv_key` with IV
//!   [`keys::PFS_FIXED_CBC_IV`], where `salt` = `files_salt||icv_salt` (LE, 8
//!   bytes; or `icv_salt` alone when `files_salt == 0`). This is the per-sector
//!   HMAC key.
//!
//! Each `0x8000` sector is AES-128-CBC-CTS-decrypted under `drv_key` with
//! `IV = LE128(block_size * (sector_base + i)) XOR tweak_key`; the file's last
//! (partial) sector uses the CTS tail rule. Per-sector integrity is
//! `HMAC-SHA1(HMAC-SHA1(secret, LE32(sector_index)), encrypted_sector)`.

use super::keys;
use super::pfs::{FileCtx, PfsCrypto};
use super::Error;
use aes::cipher::generic_array::GenericArray;
use aes::cipher::{BlockDecrypt, BlockEncrypt, KeyInit};
use aes::Aes128;
use hmac::{Hmac, Mac};
use sha1::Sha1;

type HmacSha1 = Hmac<Sha1>;

/// HMAC-SHA1 with `key`.
pub(crate) fn hmac_sha1(key: &[u8], data: &[u8]) -> [u8; 20] {
    let mut mac = <HmacSha1 as Mac>::new_from_slice(key).expect("hmac accepts any key length");
    mac.update(data);
    let out = mac.finalize().into_bytes();
    let mut sig = [0u8; 20];
    sig.copy_from_slice(&out);
    sig
}

/// AES-128-CBC-CTS decrypt in place semantics: full 16-byte blocks are CBC-
/// decrypted; a 1..15-byte tail is recovered CFB-style from the trailing chaining
/// value. `iv` is the sector IV.
fn cbc_cts_decrypt(key: &[u8; 16], iv: &[u8; 16], ct: &[u8]) -> Vec<u8> {
    let cipher = Aes128::new(GenericArray::from_slice(key));
    let full = ct.len() & !0xF;
    let tail = ct.len() & 0xF;
    let mut out = vec![0u8; ct.len()];
    let mut prev = *iv;
    let mut i = 0;
    while i < full {
        let mut block = GenericArray::clone_from_slice(&ct[i..i + 16]);
        cipher.decrypt_block(&mut block);
        for j in 0..16 {
            out[i + j] = block[j] ^ prev[j];
        }
        prev.copy_from_slice(&ct[i..i + 16]);
        i += 16;
    }
    if tail != 0 {
        let mut ks = GenericArray::clone_from_slice(&prev);
        cipher.encrypt_block(&mut ks);
        for j in 0..tail {
            out[full + j] = ct[full + j] ^ ks[j];
        }
    }
    out
}

/// AES-128-CBC-CTS encrypt (used only to build the 20-byte `secret`: one full
/// block plus a 4-byte tail).
fn cbc_cts_encrypt(key: &[u8; 16], iv: &[u8; 16], pt: &[u8]) -> Vec<u8> {
    let cipher = Aes128::new(GenericArray::from_slice(key));
    let full = pt.len() & !0xF;
    let tail = pt.len() & 0xF;
    let mut out = vec![0u8; pt.len()];
    let mut prev = *iv;
    let mut i = 0;
    while i < full {
        let mut b = [0u8; 16];
        for j in 0..16 {
            b[j] = pt[i + j] ^ prev[j];
        }
        let mut blk = GenericArray::clone_from_slice(&b);
        cipher.encrypt_block(&mut blk);
        out[i..i + 16].copy_from_slice(&blk);
        prev.copy_from_slice(&blk);
        i += 16;
    }
    if tail != 0 {
        let mut ks = GenericArray::clone_from_slice(&prev);
        cipher.encrypt_block(&mut ks);
        for j in 0..tail {
            out[full + j] = pt[full + j] ^ ks[j];
        }
    }
    out
}

/// The salt buffer that keys the integrity secret: `files_salt||icv_salt` (LE),
/// or `icv_salt` alone when `files_salt` is zero.
fn secret_salt(files_salt: u32, icv_salt: u32) -> Vec<u8> {
    if files_salt != 0 {
        let mut v = files_salt.to_le_bytes().to_vec();
        v.extend_from_slice(&icv_salt.to_le_bytes());
        v
    } else {
        icv_salt.to_le_bytes().to_vec()
    }
}

/// The 20-byte integrity secret (per-sector HMAC key).
fn integrity_secret(drv_key: &[u8; 16], files_salt: u32, icv_salt: u32) -> [u8; 20] {
    let combo = hmac_sha1(&keys::PFS_INTEGRITY_BASE, &secret_salt(files_salt, icv_salt));
    let wrapped = cbc_cts_encrypt(drv_key, &keys::PFS_FIXED_CBC_IV, &combo);
    let mut out = [0u8; 20];
    out.copy_from_slice(&wrapped);
    out
}

/// Compute `drv_key = F00D(klicensee)` in software: AES-128-ECB-decrypt of the
/// klicensee under the fixed [`keys::PFS_F00D_CONTRACT`]. This is the value the
/// secure coprocessor keygen returns, computable offline from a public constant.
pub fn f00d_drv_key(klicensee: &[u8; 16]) -> [u8; 16] {
    let cipher = Aes128::new(GenericArray::from_slice(&keys::PFS_F00D_CONTRACT));
    let mut b = GenericArray::clone_from_slice(klicensee);
    cipher.decrypt_block(&mut b);
    let mut out = [0u8; 16];
    out.copy_from_slice(&b);
    out
}

/// The 16-byte sector-IV mask, from the file's `dbseed`.
fn tweak_key(dbseed: &[u8; 20]) -> [u8; 16] {
    let h = hmac_sha1(&keys::PFS_TWEAK_BASE, dbseed);
    let mut out = [0u8; 16];
    out.copy_from_slice(&h[..16]);
    out
}

/// PFS crypto for a read-only gamedata image, parameterised by the title's
/// `drv_key` (`F00D(klicensee)`).
pub struct GameData {
    /// `F00D(klicensee)` for this title - the sector AES key and the wrap key for
    /// the integrity secret.
    pub drv_key: [u8; 16],
    /// Fail decryption on a per-sector integrity mismatch.
    pub verify: bool,
}

impl GameData {
    /// Build from a title's `drv_key` (the coprocessor keygen of its klicensee).
    pub fn new(drv_key: [u8; 16]) -> GameData {
        GameData {
            drv_key,
            verify: true,
        }
    }

    /// Build from a title's raw klicensee, computing `drv_key` offline via the
    /// software F00D keygen ([`f00d_drv_key`]).
    pub fn from_klicensee(klicensee: &[u8; 16]) -> GameData {
        GameData::new(f00d_drv_key(klicensee))
    }
}

impl PfsCrypto for GameData {
    fn decrypt_file(&self, ctx: &FileCtx, ciphertext: &[u8]) -> Result<Vec<u8>, Error> {
        let secret = integrity_secret(&self.drv_key, ctx.files_salt, ctx.icv_salt);
        let tweak = tweak_key(ctx.iv_seed);
        let page_size = ctx.page_size.max(1) as usize;
        let mut out = Vec::with_capacity(ciphertext.len());

        for (idx, chunk) in ciphertext.chunks(page_size).enumerate() {
            if self.verify {
                if let Some(expect) = ctx.signatures.get(idx) {
                    let subkey = hmac_sha1(&secret, &(idx as u32).to_le_bytes());
                    let got = hmac_sha1(&subkey, chunk);
                    if &got != expect {
                        return Err(Error::IntegrityCheck("pfs sector hmac mismatch"));
                    }
                }
            }
            let offset = (page_size as u64).wrapping_mul(idx as u64);
            let mut iv = [0u8; 16];
            iv[..8].copy_from_slice(&offset.to_le_bytes());
            for j in 0..16 {
                iv[j] ^= tweak[j];
            }
            out.extend_from_slice(&cbc_cts_decrypt(&self.drv_key, &iv, chunk));
        }

        out.truncate(ctx.plaintext_size);
        Ok(out)
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::ingest::filesdb::FilesDb;
    use crate::ingest::pfs::PfsImage;
    use crate::ingest::rif::Rif;
    use crate::ingest::testfix;
    use crate::ingest::unicv::UnicvDb;

    fn aes_ecb(key: &[u8; 16], block: &[u8; 16], decrypt: bool) -> [u8; 16] {
        let cipher = Aes128::new(GenericArray::from_slice(key));
        let mut b = GenericArray::clone_from_slice(block);
        if decrypt {
            cipher.decrypt_block(&mut b);
        } else {
            cipher.encrypt_block(&mut b);
        }
        let mut o = [0u8; 16];
        o.copy_from_slice(&b);
        o
    }

    fn load() -> Option<(PfsImage, Rif)> {
        let fdb = testfix::read("sce_pfs/files.db")?;
        let ucv = testfix::read("sce_pfs/unicv.db")?;
        let work = testfix::read("sce_sys/package/work.bin")?;
        let files_db = FilesDb::parse(&fdb).ok()?;
        let unicv = UnicvDb::parse(&ucv).ok()?;
        let img = PfsImage::new(files_db, unicv).ok()?;
        let rif = Rif::parse(&work).ok()?;
        Some((img, rif))
    }

    /// `F00D(klicensee)` for the title, from `VITASLOP_DRV_KEY` (32 hex chars), or
    /// `None` if unset. This is the one value with no public offline formula.
    fn env_drv_key() -> Option<[u8; 16]> {
        let s = std::env::var("VITASLOP_DRV_KEY").ok()?;
        let s = s.trim();
        if s.len() != 32 {
            return None;
        }
        let mut k = [0u8; 16];
        for i in 0..16 {
            k[i] = u8::from_str_radix(&s[i * 2..i * 2 + 2], 16).ok()?;
        }
        Some(k)
    }

    /// Primitive known-answer tests (FIPS-197 AES-128, RFC-2202 HMAC-SHA1).
    #[test]
    fn primitive_kats() {
        let key = [
            0x00, 0x01, 0x02, 0x03, 0x04, 0x05, 0x06, 0x07, 0x08, 0x09, 0x0a, 0x0b, 0x0c, 0x0d,
            0x0e, 0x0f,
        ];
        let pt = [
            0x00, 0x11, 0x22, 0x33, 0x44, 0x55, 0x66, 0x77, 0x88, 0x99, 0xaa, 0xbb, 0xcc, 0xdd,
            0xee, 0xff,
        ];
        let ct = aes_ecb(&key, &pt, false);
        assert_eq!(
            ct,
            [
                0x69, 0xc4, 0xe0, 0xd8, 0x6a, 0x7b, 0x04, 0x30, 0xd8, 0xcd, 0xb7, 0x80, 0x70, 0xb4,
                0xc5, 0x5a
            ]
        );
        assert_eq!(aes_ecb(&key, &ct, true), pt);
        assert_eq!(
            hmac_sha1(&[0x0b; 20], b"Hi There"),
            [
                0xb6, 0x17, 0x31, 0x86, 0x55, 0x05, 0x72, 0x64, 0xe2, 0x8b, 0xc0, 0xb6, 0xfb, 0x37,
                0x8c, 0x8e, 0xf1, 0x46, 0xbe, 0x00
            ]
        );
    }

    /// CBC-CTS encrypt and decrypt are inverses across block-aligned and tailed
    /// lengths (the tail rule must round-trip for a real sector to decrypt).
    #[test]
    fn cbc_cts_roundtrips() {
        let key = [
            0x2b, 0x7e, 0x15, 0x16, 0x28, 0xae, 0xd2, 0xa6, 0xab, 0xf7, 0x15, 0x88, 0x09, 0xcf,
            0x4f, 0x3c,
        ];
        let iv = [
            0x11, 0x22, 0x33, 0x44, 0x55, 0x66, 0x77, 0x88, 0x99, 0xaa, 0xbb, 0xcc, 0xdd, 0xee,
            0xff, 0x00,
        ];
        for &len in &[16usize, 20, 32, 0x8000, 0x8000 + 5] {
            let pt: Vec<u8> = (0..len).map(|i| (i * 31 + 7) as u8).collect();
            let ct = cbc_cts_encrypt(&key, &iv, &pt);
            assert_eq!(ct.len(), len);
            assert_eq!(cbc_cts_decrypt(&key, &iv, &ct), pt, "len {len}");
        }
    }

    /// The node<->table correlation `table.page_no == node.id` must pair every
    /// file with a table of the right sector count (page_no is a bijection).
    #[test]
    fn correlation_is_page_no_eq_node_id() {
        let Some((img, _)) = load() else {
            return;
        };
        use std::collections::HashMap;
        let by_page: HashMap<u32, &crate::ingest::unicv::FileTable> =
            img.unicv.tables.iter().map(|t| (t.page_no, t)).collect();
        let expected = |sz: u32| if sz == 0 { 0 } else { sz.div_ceil(0x8000) };
        for n in &img.files_db.nodes {
            if n.is_dir() {
                continue;
            }
            let t = by_page
                .get(&n.id)
                .unwrap_or_else(|| panic!("no table at page_no == node id {}", n.id));
            assert_eq!(t.n_sectors, expected(n.size), "sector count for node {}", n.id);
        }
        let distinct: std::collections::HashSet<u32> =
            img.unicv.tables.iter().map(|t| t.page_no).collect();
        assert_eq!(distinct.len(), img.unicv.tables.len(), "page_no not a bijection");
    }

    /// End-to-end acid test, fully offline: `drv_key` is computed from the raw
    /// klicensee via the software F00D keygen (no console, no service). The
    /// files.db header ICV must reproduce and `eboot.bin` must decrypt (integrity
    /// verified) to an `SCE\0` SELF header. Skips without the fixture, so
    /// `cargo test` stays green. `VITASLOP_DRV_KEY` overrides the computed key.
    #[test]
    fn decrypts_eboot_offline() {
        let Some((img, rif)) = load() else {
            return;
        };
        let drv = env_drv_key().unwrap_or_else(|| f00d_drv_key(&rif.key));
        let fdb = testfix::read("sce_pfs/files.db").unwrap();

        // 1. files.db header ICV validates the integrity-secret keygen (icv_salt=0),
        // independent of the content cipher.
        let secret = integrity_secret(&drv, img.files_db.header.seed, 0);
        let mut hdr = fdb[..0x160].to_vec();
        for b in &mut hdr[0x4c..0x160] {
            *b = 0;
        }
        assert_eq!(
            &hmac_sha1(&secret, &hdr)[..],
            &fdb[0x4c..0x60],
            "files.db header ICV mismatch - keygen/drv_key wrong"
        );

        // 2. eboot decrypts (with per-sector integrity on) to a SELF header.
        let eboot_ct = testfix::read("eboot.bin").unwrap();
        let ctx = img.file_ctx("eboot.bin", &rif.key).expect("eboot ctx");
        let pt = GameData::new(drv)
            .decrypt_file(&ctx, &eboot_ct)
            .expect("decrypt eboot");
        assert_eq!(&pt[..4], b"SCE\0", "eboot did not decrypt to an SCE header");
    }

    /// Generality check: several files decrypt (integrity-verified) to their true
    /// magic - a multi-page module (SELF), an image (PNG), across different node
    /// ids / dbseeds / page counts. Fully offline. Skips without the fixture.
    #[test]
    fn decrypts_assets_offline() {
        let Some((img, rif)) = load() else {
            return;
        };
        let gd = GameData::from_klicensee(&rif.key);
        let cases: &[(&str, &[u8])] = &[
            ("sce_module/libc.suprx", b"SCE\0"),
            ("sce_module/libfios2.suprx", b"SCE\0"),
            ("sce_sys/icon0.png", b"\x89PNG"),
        ];
        for (path, magic) in cases {
            let Some(ct) = testfix::read(path) else {
                continue;
            };
            let ctx = img.file_ctx(path, &rif.key).expect("file ctx");
            let pt = gd
                .decrypt_file(&ctx, &ct)
                .unwrap_or_else(|e| panic!("decrypt {path}: {e}"));
            assert_eq!(
                &pt[..magic.len()],
                *magic,
                "{path} decrypted to wrong magic {:02x?}",
                &pt[..magic.len().min(pt.len())]
            );
        }
    }
}
