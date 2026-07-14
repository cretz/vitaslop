//! `sce_pfs/files.db` - the PFS directory database (`SCENGPFS`).
//!
//! files.db is a fixed-size-block database that names every directory and file
//! in a PFS image and records, per file, its size and an encryption class. In a
//! raw app dump the file *bytes* live as ordinary on-disk files at their real
//! paths, so we do not need files.db to find data - but we do need it for two
//! things: the per-file encryption class (which files are PFS-encrypted vs stored
//! plaintext), and the file ordering that lines up with the `unicv.db` hash
//! tables and the per-file crypto IV base.
//!
//! Layout (psdevwiki `Files.db` + original RE against a v5 dump):
//!
//! * The file is a sequence of `block_size` blocks (0x400 here). Block 0 is a
//!   header ([`Header`]); the rest are node blocks.
//! * Each node block begins with a 0x10 `{id, type, nfiles, unk}` header, then 9
//!   filename entries (`{parent_id:u32, name[68]}`, 72 bytes each) at 0x10, then
//!   9 file-info entries (`{id:u32, type:u16, unk:u16, size:u32, unk:u32}`, 16
//!   bytes each) at 0x298. Entry `i` in the two arrays describe the same node:
//!   its own id and attributes from file-info, its parent id and name from the
//!   filename array. `nfiles` (<= 9) entries are live.
//! * Block `type` 0 carries real file/dir entries; type 1 blocks are interior
//!   B-tree index nodes whose file-info is zeroed - we skip those.
//!
//! Node id 0 is the root directory; a filename entry's `parent_id` points at the
//! file-info id of its containing directory, so full paths resolve by walking
//! parent links.

use super::{Error, Reader};

/// files.db magic (`"SCENGPFS"`).
const MAGIC: &[u8; 8] = b"SCENGPFS";
const HEADER_BLOCK: usize = 0;
const NAMES_OFF: usize = 0x10;
const NAME_ENTRY: usize = 72;
const FINFO_OFF: usize = 0x298;
const FINFO_ENTRY: usize = 16;
const MAX_ENTRIES: usize = 9;

/// files.db block type: real file/directory entries (vs 1, an index node).
const BLOCK_TYPE_LEAF: u32 = 0;

/// File-info `type` (encryption class) values.
pub mod ftype {
    /// Ordinary PFS-encrypted file.
    pub const NORMAL: u16 = 0x0001;
    /// System file - PFS-encrypted like [`NORMAL`], different class bit.
    pub const SYS: u16 = 0x0006;
    /// Directory (no data, no encryption).
    pub const DIR: u16 = 0x8000;
    /// Non-encrypted file: stored as plaintext even inside PFS (e.g. param.sfo).
    pub const NENC: u16 = 0x4006;
}

/// files.db header block fields we use.
#[derive(Debug, Clone, Copy)]
pub struct Header {
    pub version: u32,
    pub block_size: u32,
    /// The NPDRM key id (`fsdb_np_key_id`, files.db header offset 0xE) that
    /// selects the key class for derivation.
    pub key_id: u16,
    /// The files.db seed added to the key derivation for version >= 4 (a.k.a.
    /// `files_salt`). Feeds the per-file key derivation.
    pub seed: u32,
    /// Byte length of the block region after the header block.
    pub body_size: u64,
}

/// One node (file or directory) from files.db.
#[derive(Debug, Clone)]
pub struct Node {
    /// This node's own id (root is 0).
    pub id: u32,
    /// The id of the directory containing this node.
    pub parent_id: u32,
    pub name: String,
    /// Encryption/kind class - one of [`ftype`].
    pub ftype: u16,
    /// File size in bytes (0 for directories).
    pub size: u32,
}

impl Node {
    pub fn is_dir(&self) -> bool {
        self.ftype == ftype::DIR
    }
    /// Whether this file's on-disk bytes are PFS-encrypted (vs stored plaintext).
    pub fn is_encrypted(&self) -> bool {
        matches!(self.ftype, ftype::NORMAL | ftype::SYS)
    }
}

/// A parsed files.db: its header and every live node, in block/entry order.
pub struct FilesDb {
    pub header: Header,
    pub nodes: Vec<Node>,
}

impl FilesDb {
    /// Parse a whole files.db image.
    pub fn parse(bytes: &[u8]) -> Result<FilesDb, Error> {
        let r = Reader::new(bytes);
        if bytes.get(0..8) != Some(MAGIC.as_slice()) {
            return Err(Error::BadMagic("files.db magic"));
        }
        let version = r.u32(8)?;
        let block_size = r.u32(0x10)?;
        if block_size == 0 || (block_size as usize) < FINFO_OFF + MAX_ENTRIES * FINFO_ENTRY {
            return Err(Error::BadMagic("files.db block_size"));
        }
        let header = Header {
            version,
            block_size,
            key_id: r.u16(0x0e)?,
            seed: r.u32(0x1c)?,
            body_size: r.u64(0x28)?,
        };

        let bs = block_size as usize;
        let mut nodes = Vec::new();
        // Node blocks follow the header block.
        let mut off = (HEADER_BLOCK + 1) * bs;
        while off + bs <= bytes.len() {
            let block = &bytes[off..off + bs];
            let br = Reader::new(block);
            let btype = br.u32(4)?;
            let nfiles = br.u32(8)? as usize;
            if btype == BLOCK_TYPE_LEAF && nfiles <= MAX_ENTRIES {
                for i in 0..nfiles {
                    let name_at = NAMES_OFF + i * NAME_ENTRY;
                    let parent_id = br.u32(name_at)?;
                    let name = read_cstr(block, name_at + 4, NAME_ENTRY - 4);

                    let finfo_at = FINFO_OFF + i * FINFO_ENTRY;
                    let id = br.u32(finfo_at)?;
                    let ftype = br.u16(finfo_at + 4)?;
                    let size = br.u32(finfo_at + 8)?;

                    nodes.push(Node {
                        id,
                        parent_id,
                        name,
                        ftype,
                        size,
                    });
                }
            }
            off += bs;
        }
        Ok(FilesDb { header, nodes })
    }

    /// Resolve every non-directory node to its full '/'-separated path (no
    /// leading slash), e.g. `"sce_module/libc.suprx"`. Nodes whose parent chain
    /// does not reach the root are skipped (defensive; should not happen in a
    /// well-formed db).
    pub fn file_paths(&self) -> Vec<(String, &Node)> {
        use std::collections::HashMap;
        // id -> (name, parent_id) for directory-chain walking.
        let by_id: HashMap<u32, &Node> = self.nodes.iter().map(|n| (n.id, n)).collect();

        let mut out = Vec::new();
        for node in &self.nodes {
            if node.is_dir() {
                continue;
            }
            if let Some(path) = resolve_path(node, &by_id) {
                out.push((path, node));
            }
        }
        out
    }
}

/// Walk parent links from `node` up to the root (id 0), building `a/b/c`.
fn resolve_path(node: &Node, by_id: &std::collections::HashMap<u32, &Node>) -> Option<String> {
    let mut parts = vec![node.name.clone()];
    let mut parent = node.parent_id;
    // The root directory is id 0 and is its own containing scope; bound the walk
    // so a malformed cyclic db cannot loop forever.
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

/// Read a NUL-terminated string of at most `max` bytes at `at` within `block`.
fn read_cstr(block: &[u8], at: usize, max: usize) -> String {
    let end = (at + max).min(block.len());
    let slice = &block[at.min(block.len())..end];
    let n = slice.iter().position(|&b| b == 0).unwrap_or(slice.len());
    String::from_utf8_lossy(&slice[..n]).into_owned()
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::ingest::testfix;

    #[test]
    fn parses_olliolli_filesdb() {
        let Some(bytes) = testfix::read("sce_pfs/files.db") else {
            return; // fixture absent - skip (VITASLOP_GAME_DIR unset)
        };
        let db = FilesDb::parse(&bytes).expect("parse files.db");
        assert_eq!(db.header.version, 5);
        assert_eq!(db.header.block_size, 0x400);
        assert_eq!(db.header.seed, 0x74f4_145f);

        let paths = db.file_paths();
        let by_path = |want: &str| paths.iter().find(|(p, _)| p == want).map(|(_, n)| *n);

        // eboot.bin: root-level, normal (encrypted), known size.
        let eboot = by_path("eboot.bin").expect("eboot.bin in files.db");
        assert_eq!(eboot.ftype, ftype::NORMAL);
        assert!(eboot.is_encrypted());
        assert_eq!(eboot.size, 412816);

        // A nested module resolves its full path and is encrypted.
        let libc = by_path("sce_module/libc.suprx").expect("libc.suprx path");
        assert!(libc.is_encrypted());

        // param.sfo is stored non-encrypted (nenc), matching its plaintext bytes.
        let sfo = by_path("sce_sys/param.sfo").expect("param.sfo path");
        assert_eq!(sfo.ftype, ftype::NENC);
        assert!(!sfo.is_encrypted());
    }
}
