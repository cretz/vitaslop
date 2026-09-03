//! A read-only PSARC archive reader, and the index a title's paths resolve through.
//!
//! # Why the filesystem needs to look inside an archive at all
//! One retail title ships its entire data tree as a single `PSP2/data.psarc` - 1.6 GB, no
//! loose `data/` directory anywhere in the container - and reads it ITSELF: FIOS2 is
//! statically linked into its eboot, so every asset the title loads goes through the
//! title's own archive code and lands as a plain read of `data.psarc`. That works here
//! today and needs nothing from this module.
//!
//! What does NOT work is a path a SYSTEM MODULE opens on the title's behalf.
//! `sceMp4OpenFile("data/Videos/intro.mp4")` is one: the guest hands over a path and the
//! module does the opening, so the title's own archive layer is never consulted, the file
//! is not on disk, and the open fails (`0x80010002`). On hardware that movie plays; here
//! the title showed its attract screen with no video and waited for the button that skips
//! it. Nothing was wrong with the renderer, and nothing was wrong with the path handling.
//! The bytes were simply inside the archive.
//!
//! So the filesystem mounts the archive: an entry inside a `.psarc` is visible at the path
//! the archive names it by, AFTER every resident and backed file has missed. That makes
//! the module's open resolve exactly as the device resolves it, and it cannot shadow
//! anything real, because the archive is consulted last.
//!
//! # The format, as parsed here
//! Big-endian throughout. A 32-byte header (`PSAR`, version, compression, TOC length,
//! entry size, entry count, block size, flags), then `count` fixed-size TOC entries, then
//! a table of per-block COMPRESSED sizes. Entry 0 is the manifest: the names of entries
//! 1.. , one per line. A file's bytes are a run of blocks starting at its `first_block`,
//! each holding exactly `block_size` uncompressed bytes except the last - which is what
//! makes a random-access read O(1) rather than a walk from the start of the file.
//!
//! Only zlib compression is implemented. An `lzma` archive is refused by name rather than
//! mis-parsed into plausible garbage.

use std::cell::RefCell;
use std::collections::HashMap;

/// One file inside an archive.
pub struct Entry {
    /// The path the archive names it by, in the archive's own spelling (the caller
    /// normalises for lookup and keeps this for directory listings).
    pub name: String,
    /// Index into [`Psarc::blocks`] of this file's first block.
    first_block: u32,
    /// Uncompressed length.
    pub size: u64,
    /// Byte offset of the first block in the archive file.
    offset: u64,
}

/// A mounted archive: its index, and a small cache of recently inflated blocks.
pub struct Psarc {
    /// The normalised vfs key of the archive FILE, so reads can be issued back through
    /// the filesystem that mounted it - resident on native, OPFS-backed in the browser.
    /// This module never touches storage itself.
    pub key: String,
    /// Uncompressed bytes per block; every block but a file's last holds exactly this.
    block_size: u32,
    pub entries: Vec<Entry>,
    /// Compressed size of each block. ZERO means a full, uncompressed `block_size` block -
    /// the format's way of spelling "this block did not compress".
    blocks: Vec<u32>,
    /// `cum[i]` is the archive-file distance from block 0 to block `i`, so the offset of a
    /// file's `n`th block is `entry.offset + cum[first + n] - cum[first]`. Without it every
    /// read would sum block sizes from the start of the file.
    cum: Vec<u64>,
    /// Recently inflated blocks, by absolute block index. A movie streams sequentially and
    /// a 64 KB read straddles at most two blocks, so a handful of entries turns a whole
    /// playback into one inflate per block rather than one per read.
    cache: RefCell<Cache>,
}

/// Inflated blocks, newest last, bounded by count.
#[derive(Default)]
struct Cache {
    /// Shared, not owned: a hit used to CLONE the whole inflated block for every read,
    /// so a stream read in 4 KB pieces copied each 64 KB block sixteen times over.
    map: HashMap<u32, std::sync::Arc<[u8]>>,
    order: Vec<u32>,
}

/// How many inflated blocks to keep. Eight 64 KB blocks is half a megabyte, which is
/// nothing beside the archive and enough to cover a straddling read plus read-ahead.
const CACHE_BLOCKS: usize = 8;

/// The header is fixed at 32 bytes; a TOC shorter than that is not one.
const HEADER_LEN: usize = 32;

impl Psarc {
    /// Parse the header, TOC and block table of the archive at `key`, reading through
    /// `read` (`(offset, len) -> bytes`, short at end of file).
    ///
    /// Returns `Err` with a reason for anything that is not a zlib PSARC this can serve.
    /// A refusal here leaves the file exactly as it was - an ordinary opaque asset the
    /// title reads itself - so a mis-detected `.psarc` costs a log line and nothing else.
    pub fn parse(
        key: &str,
        read: &dyn Fn(usize, usize) -> Option<Vec<u8>>,
    ) -> Result<Psarc, String> {
        let head = read(0, HEADER_LEN).unwrap_or_default();
        if head.len() < HEADER_LEN {
            return Err(format!("{key}: {} bytes, too short for a PSARC header", head.len()));
        }
        let be32 = |at: usize| u32::from_be_bytes([head[at], head[at + 1], head[at + 2], head[at + 3]]);
        if &head[0..4] != b"PSAR" {
            return Err(format!("{key}: no PSAR magic"));
        }
        let comp = String::from_utf8_lossy(&head[8..12]).to_string();
        if comp != "zlib" {
            return Err(format!("{key}: {comp} compression is not implemented (only zlib)"));
        }
        let toc_len = be32(12) as usize;
        let entry_size = be32(16) as usize;
        let entry_count = be32(20) as usize;
        let block_size = be32(24);
        if toc_len < HEADER_LEN || entry_size < 30 || entry_count == 0 || block_size == 0 {
            return Err(format!(
                "{key}: implausible TOC (length {toc_len}, {entry_count} entries of \
                 {entry_size} bytes, {block_size}-byte blocks)"
            ));
        }
        let toc = read(HEADER_LEN, toc_len - HEADER_LEN).unwrap_or_default();
        if toc.len() != toc_len - HEADER_LEN {
            return Err(format!("{key}: read {} of its {toc_len}-byte TOC", toc.len() + HEADER_LEN));
        }
        // A size or offset is 40 bits: one byte then a big-endian u32.
        let u40 = |b: &[u8], at: usize| -> u64 {
            (b[at] as u64) << 32
                | u32::from_be_bytes([b[at + 1], b[at + 2], b[at + 3], b[at + 4]]) as u64
        };
        let table_at = entry_count * entry_size;
        if table_at > toc.len() {
            return Err(format!("{key}: {entry_count} TOC entries do not fit in {toc_len} bytes"));
        }
        let mut raw = Vec::with_capacity(entry_count);
        for i in 0..entry_count {
            let o = i * entry_size;
            // The 16-byte MD5 of the name leads each entry and is not needed: names come
            // from the manifest, in the same order.
            raw.push((
                u32::from_be_bytes([toc[o + 16], toc[o + 17], toc[o + 18], toc[o + 19]]),
                u40(&toc, o + 20),
                u40(&toc, o + 25),
            ));
        }
        // The block table's width follows the block size: two bytes up to 64 KB, three up
        // to 16 MB, four beyond. It is the only field in the format whose SIZE is implied.
        let width = if block_size > 0x100_0000 {
            4
        } else if block_size > 0x1_0000 {
            3
        } else {
            2
        };
        let mut blocks = Vec::with_capacity((toc.len() - table_at) / width);
        let mut at = table_at;
        while at + width <= toc.len() {
            let mut v = 0u32;
            for b in 0..width {
                v = (v << 8) | toc[at + b] as u32;
            }
            blocks.push(v);
            at += width;
        }
        let mut cum = Vec::with_capacity(blocks.len() + 1);
        let mut running = 0u64;
        cum.push(0);
        for &b in &blocks {
            running += if b == 0 { block_size as u64 } else { b as u64 };
            cum.push(running);
        }

        let mut psarc = Psarc {
            key: key.to_string(),
            block_size,
            entries: Vec::new(),
            blocks,
            cum,
            cache: RefCell::new(Cache::default()),
        };
        // Entry 0 is the manifest and is read through the very machinery it is about to
        // name: it is an ordinary entry whose blocks are already indexed.
        let (fb, size, offset) = raw[0];
        psarc.entries.push(Entry { name: String::new(), first_block: fb, size, offset });
        let mut manifest = vec![0u8; size as usize];
        let got = psarc.read_entry(0, 0, &mut manifest, read);
        if got != manifest.len() {
            return Err(format!("{key}: read {got} of its {size}-byte manifest"));
        }
        let names: Vec<&str> = std::str::from_utf8(&manifest)
            .map_err(|_| format!("{key}: its manifest is not UTF-8"))?
            .split('\n')
            .map(|l| l.trim_end_matches('\r'))
            .filter(|l| !l.is_empty())
            .collect();
        if names.len() != entry_count - 1 {
            return Err(format!(
                "{key}: its manifest names {} files but the TOC holds {}",
                names.len(),
                entry_count - 1
            ));
        }
        psarc.entries.clear();
        for (i, name) in names.iter().enumerate() {
            let (first_block, size, offset) = raw[i + 1];
            psarc.entries.push(Entry {
                name: name.trim_start_matches('/').to_string(),
                first_block,
                size,
                offset,
            });
        }
        Ok(psarc)
    }

    /// Read `[off, off + buf.len())` of entry `idx` into `buf`, returning the count -
    /// short only at the end of the file. `read` fetches raw archive bytes.
    pub fn read_entry(
        &self,
        idx: usize,
        off: usize,
        buf: &mut [u8],
        read: &dyn Fn(usize, usize) -> Option<Vec<u8>>,
    ) -> usize {
        let Some(entry) = self.entries.get(idx) else { return 0 };
        let bs = self.block_size as usize;
        let mut done = 0usize;
        while done < buf.len() {
            let at = off + done;
            if at as u64 >= entry.size {
                break;
            }
            let nth = at / bs;
            let within = at % bs;
            let block = entry.first_block as usize + nth;
            let Some(bytes) = self.block(block, entry, nth, read) else { break };
            if within >= bytes.len() {
                break;
            }
            let take = (bytes.len() - within).min(buf.len() - done);
            buf[done..done + take].copy_from_slice(&bytes[within..within + take]);
            done += take;
            // A short block is the file's last one; anything past it does not exist.
            if bytes.len() < bs {
                break;
            }
        }
        done
    }

    /// The inflated bytes of absolute block `block` (the `nth` block of `entry`), from
    /// the cache or by reading and inflating it.
    fn block(
        &self,
        block: usize,
        entry: &Entry,
        nth: usize,
        read: &dyn Fn(usize, usize) -> Option<Vec<u8>>,
    ) -> Option<std::sync::Arc<[u8]>> {
        if let Some(hit) = self.cache.borrow().map.get(&(block as u32)) {
            return Some(hit.clone());
        }
        let csz = match self.blocks.get(block).copied() {
            // Zero is the format's "stored, and exactly one block long".
            Some(0) => self.block_size as usize,
            Some(n) => n as usize,
            None => return None,
        };
        let at = entry.offset + self.cum[entry.first_block as usize + nth]
            - self.cum[entry.first_block as usize];
        let raw = read(at as usize, csz)?;
        if raw.is_empty() {
            return None;
        }
        // A block that did not compress is stored verbatim. zlib's own first byte (0x78
        // for every window size a deflater emits here) is what tells the two apart, and a
        // failed inflate falls back to verbatim rather than to nothing: a stored block
        // whose first byte happens to be 0x78 is rare but not impossible.
        let out = if raw[0] == 0x78 {
            miniz_oxide::inflate::decompress_to_vec_zlib_with_limit(&raw, self.block_size as usize)
                .unwrap_or(raw)
        } else {
            raw
        };
        let out: std::sync::Arc<[u8]> = out.into();
        let mut cache = self.cache.borrow_mut();
        cache.map.insert(block as u32, out.clone());
        cache.order.push(block as u32);
        if cache.order.len() > CACHE_BLOCKS {
            let old = cache.order.remove(0);
            cache.map.remove(&old);
        }
        Some(out)
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    /// Build a PSARC the way the format specifies, so the reader is tested against bytes
    /// rather than against itself. Two files, one of them longer than a block, and a
    /// deliberately incompressible one so the STORED-block path is exercised too.
    fn synthesise(block_size: u32, files: &[(&str, Vec<u8>)]) -> Vec<u8> {
        let manifest: Vec<u8> =
            files.iter().map(|(n, _)| *n).collect::<Vec<_>>().join("\n").into_bytes();
        // Entry 0 is the manifest; the rest follow in order.
        let payloads: Vec<&[u8]> =
            std::iter::once(manifest.as_slice()).chain(files.iter().map(|(_, b)| b.as_slice())).collect();

        let mut blocks: Vec<u32> = Vec::new();
        let mut data: Vec<u8> = Vec::new();
        let mut toc: Vec<(u32, u64, u64)> = Vec::new();
        for payload in &payloads {
            let first = blocks.len() as u32;
            let offset = data.len() as u64;
            for chunk in payload.chunks(block_size as usize) {
                let packed = miniz_oxide::deflate::compress_to_vec_zlib(chunk, 6);
                // The format stores a block verbatim when compressing it did not pay.
                if packed.len() < chunk.len() {
                    blocks.push(packed.len() as u32);
                    data.extend_from_slice(&packed);
                } else {
                    blocks.push(if chunk.len() == block_size as usize { 0 } else { chunk.len() as u32 });
                    data.extend_from_slice(chunk);
                }
            }
            toc.push((first, payload.len() as u64, offset));
        }

        let entry_size = 30usize;
        let toc_len = 32 + toc.len() * entry_size + blocks.len() * 2;
        let mut out = Vec::new();
        out.extend_from_slice(b"PSAR");
        out.extend_from_slice(&1u16.to_be_bytes());
        out.extend_from_slice(&4u16.to_be_bytes());
        out.extend_from_slice(b"zlib");
        out.extend_from_slice(&(toc_len as u32).to_be_bytes());
        out.extend_from_slice(&(entry_size as u32).to_be_bytes());
        out.extend_from_slice(&(toc.len() as u32).to_be_bytes());
        out.extend_from_slice(&block_size.to_be_bytes());
        out.extend_from_slice(&1u32.to_be_bytes());
        for (first, size, offset) in &toc {
            out.extend_from_slice(&[0u8; 16]);
            out.extend_from_slice(&first.to_be_bytes());
            out.push((size >> 32) as u8);
            out.extend_from_slice(&(*size as u32).to_be_bytes());
            out.push((offset >> 32) as u8);
            out.extend_from_slice(&(*offset as u32).to_be_bytes());
        }
        for b in &blocks {
            out.extend_from_slice(&(*b as u16).to_be_bytes());
        }
        // Entry offsets were recorded relative to the start of the data area.
        let base = out.len() as u64;
        let fixed_toc = toc.iter().map(|(f, s, o)| (*f, *s, o + base)).collect::<Vec<_>>();
        let mut at = 32;
        for (first, size, offset) in &fixed_toc {
            out[at + 16..at + 20].copy_from_slice(&first.to_be_bytes());
            out[at + 20] = (size >> 32) as u8;
            out[at + 21..at + 25].copy_from_slice(&(*size as u32).to_be_bytes());
            out[at + 25] = (offset >> 32) as u8;
            out[at + 26..at + 30].copy_from_slice(&(*offset as u32).to_be_bytes());
            at += entry_size;
        }
        out.extend_from_slice(&data);
        out
    }

    fn reader(bytes: Vec<u8>) -> impl Fn(usize, usize) -> Option<Vec<u8>> {
        move |off: usize, len: usize| {
            let start = off.min(bytes.len());
            let end = (start + len).min(bytes.len());
            Some(bytes[start..end].to_vec())
        }
    }

    #[test]
    fn reads_every_file_at_every_offset() {
        // 64 bytes a block, so the long file spans several and the reader's block
        // arithmetic is exercised rather than trivially satisfied.
        let compressible: Vec<u8> = std::iter::repeat(b'a').take(300).collect();
        let mut incompressible: Vec<u8> = Vec::new();
        let mut x = 0x1234_5678u32;
        for _ in 0..200 {
            x = x.wrapping_mul(1_664_525).wrapping_add(1_013_904_223);
            incompressible.push((x >> 24) as u8);
        }
        let files = vec![
            ("data/Videos/intro.mp4", compressible.clone()),
            ("data/noise.bin", incompressible.clone()),
        ];
        let read = reader(synthesise(64, &files));
        let arc = Psarc::parse("psp2/data.psarc", &read).expect("parses");
        assert_eq!(arc.entries.len(), 2);
        assert_eq!(arc.entries[0].name, "data/Videos/intro.mp4");
        assert_eq!(arc.entries[1].size, incompressible.len() as u64);

        for (idx, want) in [(0usize, &compressible), (1, &incompressible)] {
            // Whole file.
            let mut all = vec![0u8; want.len()];
            assert_eq!(arc.read_entry(idx, 0, &mut all, &read), want.len());
            assert_eq!(&all, want.as_slice());
            // Every offset, in reads that straddle block boundaries.
            for off in 0..want.len() {
                let mut buf = vec![0u8; 70];
                let got = arc.read_entry(idx, off, &mut buf, &read);
                assert_eq!(got, (want.len() - off).min(70), "short read at {off}");
                assert_eq!(&buf[..got], &want[off..off + got], "wrong bytes at {off}");
            }
            // Past the end reads nothing rather than wrapping or panicking.
            assert_eq!(arc.read_entry(idx, want.len(), &mut [0u8; 8], &read), 0);
        }
    }

    #[test]
    fn refuses_what_it_cannot_serve() {
        // `Psarc` holds a cache and is deliberately not `Debug`, so the error is matched
        // out rather than unwrapped.
        let why = |r: Result<Psarc, String>| match r {
            Ok(_) => panic!("parsed something that is not a servable archive"),
            Err(e) => e,
        };
        let read = reader(b"not an archive at all, not even close".to_vec());
        assert!(why(Psarc::parse("x.psarc", &read)).contains("no PSAR magic"));
        let mut lzma = synthesise(64, &[("a", vec![1, 2, 3])]);
        lzma[8..12].copy_from_slice(b"lzma");
        assert!(why(Psarc::parse("x.psarc", &reader(lzma))).contains("not implemented"));
    }

    /// Read a REAL archive: `VITASLOP_PSARC=<path to a .psarc>`, optionally
    /// `VITASLOP_PSARC_FILE=<a path inside it>` to dump the head of one entry.
    ///
    /// Ignored by default because it needs a title's own container, which is not in the
    /// repo. It is how the synthetic test above is kept honest about the real format.
    #[test]
    #[ignore]
    fn read_a_real_archive() {
        let path = std::env::var("VITASLOP_PSARC").expect("set VITASLOP_PSARC");
        // Seek + read through a `RefCell` rather than a platform positioned-read, so this
        // test compiles everywhere the crate does.
        use std::io::{Read, Seek, SeekFrom};
        let file = RefCell::new(std::fs::File::open(&path).expect("open"));
        let read = |off: usize, len: usize| {
            let mut f = file.borrow_mut();
            f.seek(SeekFrom::Start(off as u64)).ok()?;
            let mut buf = vec![0u8; len];
            let mut done = 0;
            while done < len {
                match f.read(&mut buf[done..]) {
                    Ok(0) | Err(_) => break,
                    Ok(n) => done += n,
                }
            }
            buf.truncate(done);
            Some(buf)
        };
        let arc = Psarc::parse("real.psarc", &read).expect("parses");
        println!("{}: {} files", path, arc.entries.len());
        let want = std::env::var("VITASLOP_PSARC_FILE").unwrap_or_default();
        if want.is_empty() {
            for e in arc.entries.iter().take(10) {
                println!("  {} ({} bytes)", e.name, e.size);
            }
            return;
        }
        let idx = arc
            .entries
            .iter()
            .position(|e| e.name.eq_ignore_ascii_case(&want))
            .unwrap_or_else(|| panic!("{want} is not in the archive"));
        let mut head = vec![0u8; 64];
        let got = arc.read_entry(idx, 0, &mut head, &read);
        println!(
            "{want}: {} bytes, first {got}: {:02x?}\n  as ascii: {}",
            arc.entries[idx].size,
            &head[..got],
            String::from_utf8_lossy(&head[..got].iter().map(|b| if b.is_ascii_graphic() { *b } else { b'.' }).collect::<Vec<_>>())
        );
    }
}
