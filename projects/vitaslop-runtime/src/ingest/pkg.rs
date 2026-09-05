//! `.pkg` transport layer: the AES-CTR outer wrapper a NoNpDrm package ships in.
//!
//! A Vita `.pkg` is a header, an item (file) table, and a data blob, the last two
//! encrypted as a single AES-128-CTR stream. The per-package session key is
//! `AES-128-ECB(pkg_key, riv)` where `riv` is the 0x10-byte value at header
//! offset 0x70 and `pkg_key` is chosen by `header[0xe7] & 7` (Type 2 for retail
//! Vita apps). The CTR counter starts at `riv` and advances one block per 16
//! bytes across the whole data section, so a byte at absolute offset `off`
//! (>= data_offset) uses counter `riv + (off - data_offset)/16`.
//!
//! All header/table integers are BIG-endian (unlike everything else here).
//!
//! Note on layering: decrypting the pkg gives the app files as they sit inside,
//! which for a retail app are still PFS-encrypted (the raw dump is exactly this
//! post-transport form). So the pkg layer feeds the PFS layer, it does not
//! replace it. Algorithm from the public psdevwiki pkg description; key is the
//! published Type-2 constant.

use super::vfs::MemVfs;
use super::{keys, Error, Reader};
use aes::cipher::{
    BlockEncrypt, KeyInit, KeyIvInit, StreamCipher, StreamCipherSeek, generic_array::GenericArray,
};

type Aes128Ctr = ctr::Ctr128BE<aes::Aes128>;

/// pkg magic (`"\x7fPKG"`).
const MAGIC: &[u8; 4] = b"\x7fPKG";

/// A parsed pkg header (big-endian fields).
#[derive(Debug, Clone)]
pub struct PkgHeader {
    pub item_count: u32,
    pub data_offset: u64,
    pub data_size: u64,
    pub content_id: String,
    /// The 0x10-byte package IV (header 0x70) - the CTR base counter.
    pub riv: [u8; 16],
    /// `header[0xe7] & 7` - selects the pkg key class.
    pub key_type: u8,
}

/// One decrypted item-table entry.
#[derive(Debug, Clone)]
pub struct PkgItem {
    pub name: String,
    /// Absolute file offset of the item's data within the pkg.
    pub data_offset: u64,
    pub data_size: u64,
    pub flags: u32,
}

/// Whether an item-table entry is a directory (no file data) rather than a file,
/// per the low byte of its `flags`. A pkg records directories as their own
/// entries; types 4 and 18 are the directory codes, everything else is a file.
pub(crate) fn is_directory(flags: u32) -> bool {
    matches!(flags & 0xff, 4 | 18)
}

/// Big-endian reads (the pkg header/table are big-endian).
fn be_u32(b: &[u8], at: usize) -> Result<u32, Error> {
    let s = b.get(at..at + 4).ok_or(Error::OutOfBounds("pkg u32"))?;
    Ok(u32::from_be_bytes([s[0], s[1], s[2], s[3]]))
}
fn be_u64(b: &[u8], at: usize) -> Result<u64, Error> {
    let s = b.get(at..at + 8).ok_or(Error::OutOfBounds("pkg u64"))?;
    Ok(u64::from_be_bytes([
        s[0], s[1], s[2], s[3], s[4], s[5], s[6], s[7],
    ]))
}

impl PkgHeader {
    pub fn parse(bytes: &[u8]) -> Result<PkgHeader, Error> {
        if bytes.get(0..4) != Some(MAGIC.as_slice()) {
            return Err(Error::BadMagic("pkg magic"));
        }
        let content_id = {
            let r = Reader::new(bytes);
            let cid = r.bytes::<0x30>(0x30)?;
            let n = cid.iter().position(|&b| b == 0).unwrap_or(cid.len());
            String::from_utf8_lossy(&cid[..n]).into_owned()
        };
        let riv = Reader::new(bytes).bytes::<16>(0x70)?;
        let key_type = *bytes.get(0xe7).ok_or(Error::OutOfBounds("pkg key_type"))? & 7;
        Ok(PkgHeader {
            item_count: be_u32(bytes, 0x14)?,
            data_offset: be_u64(bytes, 0x20)?,
            data_size: be_u64(bytes, 0x28)?,
            content_id,
            riv,
            key_type,
        })
    }

    /// The per-package AES-128-CTR session key: `AES-128-ECB-encrypt(pkg_key, riv)`.
    pub(crate) fn session_key(&self) -> Result<[u8; 16], Error> {
        let pkg_key = match self.key_type {
            2 => keys::PKG_TYPE2,
            _ => return Err(Error::MissingKey("pkg key type")),
        };
        let cipher = aes::Aes128::new(GenericArray::from_slice(&pkg_key));
        let mut block = GenericArray::clone_from_slice(&self.riv);
        cipher.encrypt_block(&mut block);
        Ok(block.into())
    }
}

/// A decryptor over a pkg's CTR stream.
pub struct Pkg<'a> {
    bytes: &'a [u8],
    header: PkgHeader,
    session_key: [u8; 16],
}

impl<'a> Pkg<'a> {
    pub fn open(bytes: &'a [u8]) -> Result<Pkg<'a>, Error> {
        let header = PkgHeader::parse(bytes)?;
        let session_key = header.session_key()?;
        Ok(Pkg {
            bytes,
            header,
            session_key,
        })
    }

    pub fn header(&self) -> &PkgHeader {
        &self.header
    }

    /// Decrypt `len` bytes of the CTR stream starting at absolute offset `off`
    /// (which must be >= data_offset).
    pub fn decrypt_at(&self, off: u64, len: usize) -> Result<Vec<u8>, Error> {
        let start = off as usize;
        let src = self
            .bytes
            .get(start..start + len)
            .ok_or(Error::OutOfBounds("pkg stream range"))?;
        let mut buf = src.to_vec();
        let mut c = Aes128Ctr::new(
            GenericArray::from_slice(&self.session_key),
            GenericArray::from_slice(&self.header.riv),
        );
        // Seek to this offset within the single data-section CTR stream.
        let rel = off
            .checked_sub(self.header.data_offset)
            .ok_or(Error::OutOfBounds("pkg offset before data"))?;
        c.seek(rel);
        c.apply_keystream(&mut buf);
        Ok(buf)
    }

    /// Extract every file entry into a [`MemVfs`], CTR-decrypting each item's
    /// data. Directory entries carry no data and only imply structure, so they are
    /// skipped (the tree is implicit in the '/'-separated file paths). The result
    /// is the app's on-disk file tree - for a retail app that is still the
    /// PFS-encrypted "raw dump" form the PFS layer then decrypts.
    pub fn extract(&self) -> Result<MemVfs, Error> {
        let mut vfs = MemVfs::new();
        for item in self.items()? {
            if is_directory(item.flags) {
                continue;
            }
            let data = self.decrypt_at(item.data_offset, item.data_size as usize)?;
            vfs.insert(item.name, data);
        }
        Ok(vfs)
    }

    /// Decrypt and parse the item (file) table.
    pub fn items(&self) -> Result<Vec<PkgItem>, Error> {
        // The table sits at the start of the data section; each entry is 0x20
        // bytes, and item names live inline in the data section too.
        let table_len = self.header.item_count as usize * 0x20;
        let table = self.decrypt_at(self.header.data_offset, table_len)?;
        let mut out = Vec::with_capacity(self.header.item_count as usize);
        for i in 0..self.header.item_count as usize {
            let e = i * 0x20;
            let name_offset = be_u32(&table, e)? as u64;
            let name_size = be_u32(&table, e + 4)? as usize;
            let data_offset = be_u64(&table, e + 8)?;
            let data_size = be_u64(&table, e + 0x10)?;
            let flags = be_u32(&table, e + 0x18)?;
            // Names are CTR-encrypted at data_offset + name_offset.
            let name_abs = self.header.data_offset + name_offset;
            let name_bytes = self.decrypt_at(name_abs, name_size)?;
            let name = String::from_utf8_lossy(&name_bytes).into_owned();
            out.push(PkgItem {
                name,
                data_offset: self.header.data_offset + data_offset,
                data_size,
                flags,
            });
        }
        Ok(out)
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::ingest::testfix;

    #[test]
    fn decrypts_pkg_item_table_from_head_bin() {
        // head.bin is the pkg header + item table (the data blob is truncated),
        // so we can validate the CTR primitive by decrypting the file names.
        let Some(head) = testfix::read("sce_sys/package/head.bin") else {
            return;
        };
        let pkg = Pkg::open(&head).expect("open pkg header");
        assert_eq!(pkg.header().key_type, 2);
        assert!(!pkg.header().content_id.is_empty()); // value is title-specific

        let items = pkg.items().expect("decrypt item table");
        // The table names must decrypt to sane paths.
        let names: Vec<&str> = items.iter().map(|i| i.name.as_str()).collect();
        assert!(
            names.iter().any(|n| *n == "eboot.bin"),
            "eboot.bin not among {} decrypted names (first few: {:?})",
            names.len(),
            &names[..names.len().min(6)]
        );
        assert!(names.iter().any(|n| n.contains("sce_sys/param.sfo")));
    }
}
