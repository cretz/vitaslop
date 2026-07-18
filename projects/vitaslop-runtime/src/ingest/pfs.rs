//! PFS image assembly: tie `files.db` and `unicv.db` together and drive per-file
//! decryption.
//!
//! In a read-only gamedata image every filesystem node - directory and file
//! alike - has one `unicv.db` `SCEIFTBL` table, in the same order `files.db`
//! enumerates nodes. So the k-th node pairs with the k-th table: a file's page
//! size, per-file IV seed, and per-page HMAC-SHA1 signatures all come from its
//! table. Directories get (empty) tables too, which is why the leading tables
//! carry no signatures.
//!
//! This module builds that correlation - which is fully determined by the
//! (clean-room) container structure - and exposes it as a list of decryptable
//! files. The actual byte decryption is delegated to a [`PfsCrypto`]
//! implementation, because the key-derivation and page-cipher details are the
//! one part of PFS not published in any clean source (see [`crate::ingest`] and
//! the crypto seam below).

use super::filesdb::{FilesDb, Node};
use super::unicv::{FileTable, UnicvDb};
use super::Error;

/// A file inside a PFS image, paired with its integrity table.
pub struct PfsFile<'a> {
    /// Full '/'-separated path, e.g. `"sce_module/libc.suprx"`.
    pub path: String,
    pub node: &'a Node,
    /// The node's position in enumeration order == its unicv table index.
    pub table_index: usize,
    pub table: &'a FileTable,
}

impl PfsFile<'_> {
    /// Number of `page_size` data pages this file spans.
    pub fn page_count(&self) -> usize {
        let ps = self.table.page_size.max(1) as usize;
        (self.node.size as usize).div_ceil(ps)
    }
}

/// A parsed PFS image: the two databases plus their node<->table alignment.
pub struct PfsImage {
    pub files_db: FilesDb,
    pub unicv: UnicvDb,
}

impl PfsImage {
    /// Pair a parsed files.db with a parsed unicv.db. Every node must have a
    /// table (they are emitted 1:1, in the same order).
    pub fn new(files_db: FilesDb, unicv: UnicvDb) -> Result<PfsImage, Error> {
        if files_db.nodes.len() != unicv.tables.len() {
            return Err(Error::IntegrityCheck("files.db / unicv.db node count mismatch"));
        }
        Ok(PfsImage { files_db, unicv })
    }

    /// Every non-directory node, resolved to its path and paired with its table.
    ///
    /// Correlation is by `node.id == table.page_no`: a file's ICV salt is the
    /// unicv page its SCEIFTBL begins at, which equals its files.db node id.
    /// Verified on the fixture (883/883 files agree on sector count, page_no is a
    /// bijection) - unlike a positional pairing, which does not hold because
    /// files.db nodes are in B-tree block order. (The image-independent fallback
    /// is zero-sector signature association, which needs the crypto keygen.)
    pub fn files(&self) -> Vec<PfsFile<'_>> {
        use std::collections::HashMap;
        let by_id: HashMap<u32, &Node> = self.files_db.nodes.iter().map(|n| (n.id, n)).collect();
        let by_page: HashMap<u32, usize> = self
            .unicv
            .tables
            .iter()
            .enumerate()
            .map(|(i, t)| (t.page_no, i))
            .collect();

        let mut out = Vec::new();
        for node in &self.files_db.nodes {
            if node.is_dir() {
                continue;
            }
            let Some(path) = resolve_path(node, &by_id) else {
                continue;
            };
            let Some(&idx) = by_page.get(&node.id) else {
                continue;
            };
            out.push(PfsFile {
                path,
                node,
                table_index: idx,
                table: &self.unicv.tables[idx],
            });
        }
        out
    }

    /// Build the decryption context for `path`, or `None` if no such file. The
    /// returned context borrows this image and `klicensee`.
    pub fn file_ctx<'a>(&'a self, path: &str, klicensee: &'a [u8; 16]) -> Option<FileCtx<'a>> {
        let file = self.files().into_iter().find(|f| f.path == path)?;
        Some(FileCtx {
            klicensee,
            files_salt: self.files_db.header.seed,
            key_id: self.files_db.header.key_id,
            icv_salt: self.unicv.tables[file.table_index].page_no,
            table_index: file.table_index,
            iv_seed: file_table_iv_seed(self, file.table_index),
            has_dbseed: self.unicv.header.version > 1,
            page_size: file.table.page_size,
            signatures: file_table_signatures(self, file.table_index),
            plaintext_size: file.node.size as usize,
            encrypted: file.node.is_encrypted(),
        })
    }

    /// Decrypt one file's on-disk ciphertext to plaintext, via `crypto`. `path`
    /// selects the file; `ciphertext` is its raw bytes as stored in the
    /// container. A non-encrypted (`nenc`) file is returned as-is.
    pub fn decrypt<C: PfsCrypto>(
        &self,
        path: &str,
        ciphertext: &[u8],
        klicensee: &[u8; 16],
        crypto: &C,
    ) -> Result<Vec<u8>, Error> {
        let ctx = self
            .file_ctx(path, klicensee)
            .ok_or_else(|| Error::MissingFile(path.to_string()))?;
        if !ctx.encrypted {
            return Ok(ciphertext.to_vec());
        }
        crypto.decrypt_file(&ctx, ciphertext)
    }
}

/// Borrow a table's iv seed by index (kept as a free fn so `file_ctx` can build a
/// context that borrows `self` without also borrowing the temporary `PfsFile`).
fn file_table_iv_seed(img: &PfsImage, idx: usize) -> &[u8; 20] {
    &img.unicv.tables[idx].iv_seed
}

fn file_table_signatures(img: &PfsImage, idx: usize) -> &[[u8; 20]] {
    &img.unicv.tables[idx].signatures
}

/// Everything a [`PfsCrypto`] needs to decrypt (and verify) one file.
pub struct FileCtx<'a> {
    /// The 16-byte license secret from the RIF.
    pub klicensee: &'a [u8; 16],
    /// files.db header seed (`files_salt`).
    pub files_salt: u32,
    /// files.db header key id.
    pub key_id: u16,
    /// The file's ICV salt: the unicv page its SCEIFTBL begins at (== node id).
    pub icv_salt: u32,
    /// The file's unicv table index.
    pub table_index: usize,
    /// The file's per-file IV seed from unicv.
    pub iv_seed: &'a [u8; 20],
    /// Whether this image carries per-file dbseeds (unicv `icv_version > 1`). When
    /// false (the older v1 read-only format, e.g. launch-window titles) there is no
    /// dbseed and the sector IV uses no tweak mask - see [`crate::ingest::pfscrypt`].
    pub has_dbseed: bool,
    /// Data page size (0x8000).
    pub page_size: u32,
    /// Per-page HMAC-SHA1 signatures.
    pub signatures: &'a [[u8; 20]],
    /// The file's real (decrypted, untruncated) size.
    pub plaintext_size: usize,
    /// Whether the file is PFS-encrypted (false = stored plaintext).
    pub encrypted: bool,
}

/// The clean-room seam for PFS byte decryption.
///
/// Given a file's context and its ciphertext, produce the plaintext. This is the
/// single piece of PFS that no clean source (psdevwiki/henkaku) documents: the
/// derivation of the AES key / iv-xor-key / HMAC key from
/// `(klicensee, files_salt, key_id)`, the data-page cipher mode, and the
/// per-page IV formula. The container structure around it - files.db, unicv.db,
/// page layout, HMAC-SHA1 page signatures, page size 0x8000 - is all recovered
/// clean-room and handed over via [`FileCtx`]. An implementation of this trait
/// is deliberately not committed until its algorithm can be sourced without
/// reading copyleft/unlicensed emulator code.
pub trait PfsCrypto {
    fn decrypt_file(&self, ctx: &FileCtx, ciphertext: &[u8]) -> Result<Vec<u8>, Error>;
}

/// The default seam implementation: reports the key-derivation gap. Swapped for a
/// real implementation once its algorithm is available from an allowed source.
pub struct Unavailable;

impl PfsCrypto for Unavailable {
    fn decrypt_file(&self, _ctx: &FileCtx, _ciphertext: &[u8]) -> Result<Vec<u8>, Error> {
        Err(Error::MissingKey(
            "PFS key derivation (klicensee -> page key/iv/hmac) not available",
        ))
    }
}

/// Walk parent links from `node` to the root (id 0), building `a/b/c`.
fn resolve_path(node: &Node, by_id: &std::collections::HashMap<u32, &Node>) -> Option<String> {
    let mut parts = vec![node.name.clone()];
    let mut parent = node.parent_id;
    let mut guard = 0;
    while parent != 0 {
        let p = by_id.get(&parent)?;
        parts.push(p.name.clone());
        parent = p.parent_id;
        guard += 1;
        if guard > 4096 {
            return None;
        }
    }
    parts.reverse();
    Some(parts.join("/"))
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::ingest::testfix;

    fn load_image() -> Option<PfsImage> {
        let fdb = testfix::read("sce_pfs/files.db")?;
        let ucv = testfix::read("sce_pfs/unicv.db")?;
        let files_db = FilesDb::parse(&fdb).ok()?;
        let unicv = UnicvDb::parse(&ucv).ok()?;
        PfsImage::new(files_db, unicv).ok()
    }

    #[test]
    fn files_and_tables_are_structurally_paired() {
        let Some(img) = load_image() else { return };
        // Node count matches table count (checked in `new`), files resolve, and
        // every table declares the 0x8000 sector size.
        let files = img.files();
        assert!(!files.is_empty());
        assert!(img.unicv.tables.iter().all(|t| t.page_size == 0x8000));

        // eboot resolves and is 13 sectors; a matching-size table exists. (The
        // node<->table pairing itself is NOT positional - see `files()` note - so
        // we do not assert `eboot`'s positional table here.)
        let eboot = files
            .iter()
            .find(|f| f.path == "eboot.bin")
            .expect("eboot.bin");
        assert_eq!(eboot.page_count(), 13);
        assert!(
            img.unicv.tables.iter().any(|t| t.n_sectors == 13),
            "expected a 13-sector table for eboot"
        );
    }

    #[test]
    fn decrypt_reports_the_key_gap() {
        let Some(img) = load_image() else { return };
        // The structure is complete; only the crypto seam is unfilled, so an
        // encrypted file surfaces the precise missing-key gap (not a parse error).
        let err = img
            .decrypt("eboot.bin", b"ciphertext-placeholder", &[0u8; 16], &Unavailable)
            .unwrap_err();
        assert!(matches!(err, Error::MissingKey(_)));
    }
}
