//! `sce_pfs/unicv.db` - the read-only PFS integrity database (`SCEIRODB`).
//!
//! unicv.db holds, per file, a table of per-page HMAC-SHA1 signatures used to
//! verify decrypted data, plus a per-file IV seed. It is a sequence of
//! `block_size` (0x400) blocks: a leading `SCEIRODB` header, then one `SCEIFTBL`
//! table per filesystem node, in the same order files.db enumerates them.
//!
//! Layout facts (henkaku `Bugs.wiki` "icv mismatch" diagram + original RE):
//!
//! ```text
//! [SCEIRODB header]
//! [SCEIFTBL for node 0] [SCEIFTBL for node 1] ...
//! ```
//! For files.db v5 / unicv.db v2 each SCEIFTBL is:
//! ```text
//! [main header        0x20]
//! [hash-of-hashblocks 0x14]   (HMAC-SHA1 over the signature region)
//! [iv seed            0x14]
//! [page signatures ...    ]   (0x14 HMAC-SHA1 each, one per 0x8000 data page)
//! ```
//! The data page size (0x8000) is read from the table header. Signatures for a
//! file larger than one block continue into the following block(s).
//!
//! We verify pages with these signatures once decryption produces plaintext; the
//! signatures are not needed to derive keys, so this parser stays crypto-free.

use super::{Error, Reader};

const RODB_MAGIC: &[u8; 8] = b"SCEIRODB";
const IFTBL_MAGIC: &[u8; 8] = b"SCEIFTBL";
/// SCEIFTBL main-header size for the v2 (read-only) layout.
const IFTBL_HDR: usize = 0x20;
/// Size of one page signature (HMAC-SHA1).
pub const SIG_LEN: usize = 0x14;

/// The `SCEIRODB` database header.
#[derive(Debug, Clone, Copy)]
pub struct RoDbHeader {
    pub version: u32,
    pub block_size: u32,
    /// Total unicv.db byte length declared in the header.
    pub total_size: u64,
}

/// One `SCEIFTBL` per-file signature table.
#[derive(Debug, Clone)]
pub struct FileTable {
    /// The 0x400-page index (within unicv.db) where this SCEIFTBL begins. This
    /// is the file's ICV salt in the key derivation.
    pub page_no: u32,
    /// Data page size these signatures cover (0x8000).
    pub page_size: u32,
    /// Declared number of data sectors (== number of signatures).
    pub n_sectors: u32,
    /// Per-file seed (`dbseed`, 0x14) - an input to key derivation.
    pub iv_seed: [u8; SIG_LEN],
    /// One HMAC-SHA1 signature per data sector, in sector order.
    pub signatures: Vec<[u8; SIG_LEN]>,
}

/// A parsed unicv.db: header plus every per-file table in file order.
pub struct UnicvDb {
    pub header: RoDbHeader,
    pub tables: Vec<FileTable>,
}

impl UnicvDb {
    pub fn parse(bytes: &[u8]) -> Result<UnicvDb, Error> {
        let r = Reader::new(bytes);
        if bytes.get(0..8) != Some(RODB_MAGIC.as_slice()) {
            return Err(Error::BadMagic("unicv.db magic"));
        }
        let header = RoDbHeader {
            version: r.u32(8)?,
            block_size: r.u32(0xc)?,
            total_size: r.u64(0x18)?,
        };
        let bs = header.block_size as usize;
        if bs == 0 || bs < IFTBL_HDR + 2 * SIG_LEN {
            return Err(Error::BadMagic("unicv.db block_size"));
        }

        // Each table is a SCEIFTBL header page (0x400) followed by however many
        // signature pages hold its `n_sectors` signatures. A signature page is a
        // 0x10 header (binTreeSize, sigSize, nSignatures, pad) then that many
        // `sigSize` signatures, padded to the block. Empty tables (directories,
        // `n_sectors == 0`) have no signature pages.
        let mut tables = Vec::new();
        let mut off = bs; // first block after the SCEIRODB header
        while off + bs <= bytes.len() {
            if &bytes[off..off + 8] != IFTBL_MAGIC.as_slice() {
                return Err(Error::BadMagic("expected SCEIFTBL"));
            }
            let tr = Reader::new(&bytes[off..off + bs]);
            let n_sectors = tr.u32(0x14)?;
            let page_size = tr.u32(0x18)?; // fileSectorSize (0x8000)
            let iv_seed = tr.bytes::<SIG_LEN>(0x34)?; // dbseed

            // Collect n_sectors signatures from the following signature pages.
            let mut signatures = Vec::with_capacity(n_sectors as usize);
            let mut p = off + bs;
            while (signatures.len() as u32) < n_sectors {
                if p + bs > bytes.len() {
                    return Err(Error::OutOfBounds("unicv signature page"));
                }
                let sp = Reader::new(&bytes[p..p + bs]);
                let sig_size = sp.u32(0x04)? as usize;
                let n_in_page = sp.u32(0x08)? as usize;
                if sig_size != SIG_LEN {
                    return Err(Error::BadMagic("unicv sig size"));
                }
                let base = 0x10;
                for i in 0..n_in_page {
                    let at = base + i * SIG_LEN;
                    let mut s = [0u8; SIG_LEN];
                    s.copy_from_slice(&bytes[p + at..p + at + SIG_LEN]);
                    signatures.push(s);
                }
                p += bs;
            }
            tables.push(FileTable {
                page_no: (off / bs) as u32,
                page_size,
                n_sectors,
                iv_seed,
                signatures,
            });
            off = p;
        }

        Ok(UnicvDb { header, tables })
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::ingest::testfix;

    #[test]
    fn parses_fixture_unicv() {
        // Opt-in acid test against a privately-supplied dump; universal invariants only.
        let Some(bytes) = testfix::read("sce_pfs/unicv.db") else {
            return;
        };
        let db = UnicvDb::parse(&bytes).expect("parse unicv.db");
        assert_eq!(db.header.version, 2);
        assert_eq!(db.header.block_size, 0x400);
        // total_size counts the table region after the SCEIRODB header block.
        assert_eq!(
            db.header.total_size as usize,
            bytes.len() - db.header.block_size as usize
        );
        // One table per node; every table declares the 0x8000 page size.
        assert!(!db.tables.is_empty());
        assert!(db.tables.iter().all(|t| t.page_size == 0x8000));
    }
}
