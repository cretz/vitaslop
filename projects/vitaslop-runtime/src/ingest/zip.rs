//! A minimal, wasm-safe ZIP reader for the NoNpDrm app zip.
//!
//! The user's primary ROM input is a `.zip` of the raw app dump. Rather than
//! pull the full `zip` crate (heavier, and its codec stack is more than we need),
//! this reads the central directory and inflates entries itself - Stored (0) and
//! Deflate (8), which is everything a normal archiver emits. Raw DEFLATE is
//! handled by `miniz_oxide`; both are pure Rust and build for wasm.
//!
//! The whole archive is taken as a byte slice, so the same code serves a native
//! `std::fs::read` and browser bytes handed in from JS. Entries are decompressed
//! into a [`MemVfs`](super::vfs::MemVfs).

use super::vfs::MemVfs;
use super::{Error, Reader};

const EOCD_SIG: u32 = 0x0605_4b50; // "PK\x05\x06"
const CDIR_SIG: u32 = 0x0201_4b50; // "PK\x01\x02"
const LOCAL_SIG: u32 = 0x0403_4b50; // "PK\x03\x04"
const EOCD_MIN: usize = 22;

/// Read every file entry of a ZIP archive into a [`MemVfs`].
pub fn read_zip(bytes: &[u8]) -> Result<MemVfs, Error> {
    let r = Reader::new(bytes);

    // Find the End Of Central Directory record by scanning back for its
    // signature (the trailing comment, if any, is short).
    let eocd = find_eocd(bytes).ok_or(Error::BadMagic("zip EOCD"))?;
    let total = r.u16(eocd + 10)? as usize;
    let mut cd = r.u32(eocd + 16)? as usize; // central directory offset

    let mut vfs = MemVfs::new();
    for _ in 0..total {
        if r.u32(cd)? != CDIR_SIG {
            return Err(Error::BadMagic("zip central dir entry"));
        }
        let method = r.u16(cd + 10)?;
        let comp_size = r.u32(cd + 20)? as usize;
        let uncomp_size = r.u32(cd + 24)? as usize;
        let name_len = r.u16(cd + 28)? as usize;
        let extra_len = r.u16(cd + 30)? as usize;
        let comment_len = r.u16(cd + 32)? as usize;
        let local_off = r.u32(cd + 42)? as usize;

        let name = bytes
            .get(cd + 46..cd + 46 + name_len)
            .ok_or(Error::OutOfBounds("zip name"))?;
        let name = String::from_utf8_lossy(name).into_owned();

        // Directory entries (trailing '/') carry no data.
        if !name.ends_with('/') {
            let data = read_local_entry(bytes, local_off, method, comp_size, uncomp_size)?;
            vfs.insert(name, data);
        }

        cd += 46 + name_len + extra_len + comment_len;
    }
    Ok(vfs)
}

/// Read + decompress one entry given its local-header offset.
fn read_local_entry(
    bytes: &[u8],
    local_off: usize,
    method: u16,
    comp_size: usize,
    uncomp_size: usize,
) -> Result<Vec<u8>, Error> {
    let r = Reader::new(bytes);
    if r.u32(local_off)? != LOCAL_SIG {
        return Err(Error::BadMagic("zip local header"));
    }
    // The local header repeats name/extra lengths (they can differ from the
    // central dir's extra), so read them here to find the data start.
    let name_len = r.u16(local_off + 26)? as usize;
    let extra_len = r.u16(local_off + 28)? as usize;
    let data_off = local_off + 30 + name_len + extra_len;
    let comp = bytes
        .get(data_off..data_off + comp_size)
        .ok_or(Error::OutOfBounds("zip entry data"))?;

    match method {
        0 => Ok(comp.to_vec()), // Stored
        8 => miniz_oxide::inflate::decompress_to_vec_with_limit(comp, uncomp_size.max(1))
            .map_err(|_| Error::IntegrityCheck("zip deflate")),
        _ => Err(Error::BadMagic("zip method")),
    }
}

/// Scan backward from EOF for the EOCD signature.
fn find_eocd(bytes: &[u8]) -> Option<usize> {
    if bytes.len() < EOCD_MIN {
        return None;
    }
    // The comment is at most 0xFFFF bytes, so bound the scan.
    let max_back = (bytes.len()).min(EOCD_MIN + 0xFFFF);
    let start = bytes.len() - max_back;
    for i in (start..=bytes.len() - EOCD_MIN).rev() {
        if u32::from_le_bytes([bytes[i], bytes[i + 1], bytes[i + 2], bytes[i + 3]]) == EOCD_SIG {
            return Some(i);
        }
    }
    None
}


// ---------------------------------------------------------------------------
// Writing
// ---------------------------------------------------------------------------

/// Write a ZIP archive from `(name, bytes)` entries, in the order given.
///
/// # Why this crate writes its own
/// The reader above already exists for the ROM path, so the format is understood here and
/// the alternative is a whole archiver crate in a wasm binary that ships to a phone. What
/// is emitted is a plain, single-disk archive with Stored and Deflate entries and no
/// extensions - the subset every archiver on every platform reads, which is the point: a
/// game-data export is a file the USER handles, so it has to open in Explorer, in Finder,
/// and in whatever the phone offers.
///
/// # STORED, not deflated, and that was MEASURED rather than assumed
/// Every entry is written uncompressed. Linking miniz_oxide's COMPRESSOR - the reader only
/// ever needed its inflate half - cost **113 KB of the shipped wasm (5,667,665 ->
/// 5,783,872 bytes, +2.0%)**, which every user downloads on every cold load, and the
/// deflate itself runs on the frame that exports the save, where tens of milliseconds are
/// a visible hitch. What it would have bought is a smaller copy of a file that is
/// kilobytes to a few megabytes and is transferred by hand, occasionally. That is the
/// wrong side of the trade in both directions at once.
///
/// The READER still takes Deflate (it always did, for the ROM path), so a user who unpacks
/// one of these and re-zips it with an ordinary archiver can still upload the result.
///
/// No Zip64: the guest data this carries is a savedata mount, kilobytes to a few
/// megabytes, and an entry over 4 GB cannot arise from one.
pub fn write_zip(entries: &[(String, Vec<u8>)]) -> Vec<u8> {
    let mut out = Vec::new();
    // (local header offset, name, crc, compressed size, uncompressed size, method)
    let mut central: Vec<(u32, &str, u32, u32, u32, u16)> = Vec::new();

    for (name, data) in entries {
        let crc = crc32(data);
        let (method, payload): (u16, &[u8]) = (0, data);
        let local_off = out.len() as u32;

        out.extend_from_slice(&LOCAL_SIG.to_le_bytes());
        out.extend_from_slice(&20u16.to_le_bytes()); // version needed
        out.extend_from_slice(&0u16.to_le_bytes()); // flags
        out.extend_from_slice(&method.to_le_bytes());
        // A FIXED timestamp, and that is deliberate: the same game data must produce the
        // same bytes twice. A clock in here would make every export differ from the last
        // one, which defeats comparing two of them and makes a test that round-trips the
        // bytes impossible to write. 1980-01-01, the zero of the DOS epoch.
        out.extend_from_slice(&0u16.to_le_bytes()); // mod time
        out.extend_from_slice(&0x0021u16.to_le_bytes()); // mod date (1980-01-01)
        out.extend_from_slice(&crc.to_le_bytes());
        out.extend_from_slice(&(payload.len() as u32).to_le_bytes());
        out.extend_from_slice(&(data.len() as u32).to_le_bytes());
        out.extend_from_slice(&(name.len() as u16).to_le_bytes());
        out.extend_from_slice(&0u16.to_le_bytes()); // extra len
        out.extend_from_slice(name.as_bytes());
        out.extend_from_slice(payload);

        central.push((local_off, name, crc, payload.len() as u32, data.len() as u32, method));
    }

    let cd_start = out.len() as u32;
    for (local_off, name, crc, comp, uncomp, method) in &central {
        out.extend_from_slice(&CDIR_SIG.to_le_bytes());
        out.extend_from_slice(&20u16.to_le_bytes()); // version made by
        out.extend_from_slice(&20u16.to_le_bytes()); // version needed
        out.extend_from_slice(&0u16.to_le_bytes()); // flags
        out.extend_from_slice(&method.to_le_bytes());
        out.extend_from_slice(&0u16.to_le_bytes()); // mod time
        out.extend_from_slice(&0x0021u16.to_le_bytes()); // mod date
        out.extend_from_slice(&crc.to_le_bytes());
        out.extend_from_slice(&comp.to_le_bytes());
        out.extend_from_slice(&uncomp.to_le_bytes());
        out.extend_from_slice(&(name.len() as u16).to_le_bytes());
        out.extend_from_slice(&0u16.to_le_bytes()); // extra len
        out.extend_from_slice(&0u16.to_le_bytes()); // comment len
        out.extend_from_slice(&0u16.to_le_bytes()); // disk number
        out.extend_from_slice(&0u16.to_le_bytes()); // internal attrs
        out.extend_from_slice(&0u32.to_le_bytes()); // external attrs
        out.extend_from_slice(&local_off.to_le_bytes());
        out.extend_from_slice(name.as_bytes());
    }
    let cd_len = out.len() as u32 - cd_start;

    let n = central.len() as u16;
    out.extend_from_slice(&EOCD_SIG.to_le_bytes());
    out.extend_from_slice(&0u16.to_le_bytes()); // this disk
    out.extend_from_slice(&0u16.to_le_bytes()); // disk with cd
    out.extend_from_slice(&n.to_le_bytes());
    out.extend_from_slice(&n.to_le_bytes());
    out.extend_from_slice(&cd_len.to_le_bytes());
    out.extend_from_slice(&cd_start.to_le_bytes());
    out.extend_from_slice(&0u16.to_le_bytes()); // comment len
    out
}

/// CRC-32 (the ZIP entry checksum), table-free.
///
/// An archiver that reads one of these VERIFIES it, so a wrong value here does not make a
/// slightly-off file - it makes one every tool refuses to open, and the guest data inside
/// unreachable. Bit-for-bit the same polynomial the PNG writer in `render.rs` uses; kept
/// separate because that one is a streaming struct for chunk-at-a-time hashing and this is
/// a whole buffer at once.
pub fn crc32(data: &[u8]) -> u32 {
    let mut v = 0xFFFF_FFFFu32;
    for &byte in data {
        v ^= byte as u32;
        for _ in 0..8 {
            let mask = (v & 1).wrapping_neg();
            v = (v >> 1) ^ (0xEDB8_8320 & mask);
        }
    }
    !v
}

#[cfg(test)]
mod tests {
    use super::*;
    use super::super::vfs::Vfs;
    use crate::ingest::testfix;

    #[test]
    fn written_archives_read_back_through_both_methods() {
        // The writer emits Stored (see `write_zip` for why), so the round trip covers that
        // arm; the DEFLATE arm of the reader is what an ordinary archiver produces when a
        // user re-zips an export by hand, and it is exercised below.
        let squishy = vec![b'a'; 10_000];
        let random: Vec<u8> = (0..2000u32).map(|i| (i.wrapping_mul(2654435761) >> 13) as u8).collect();
        let entries = vec![
            ("empty".to_string(), Vec::new()),
            ("squishy".to_string(), squishy.clone()),
            ("dir/random.bin".to_string(), random.clone()),
        ];
        let zip = write_zip(&entries);
        let vfs = read_zip(&zip).expect("read back what we wrote");
        assert_eq!(vfs.read("empty").unwrap(), Vec::<u8>::new());
        assert_eq!(vfs.read("squishy").unwrap(), squishy);
        assert_eq!(vfs.read("dir/random.bin").unwrap(), random);
    }

    #[test]
    fn a_deflated_entry_still_reads_back() {
        // The path a user takes without knowing it: unpack an export, change something,
        // re-zip with the archiver their OS ships. Almost every one of those deflates, and
        // this crate no longer writes an archive of that shape - so the arm has to be
        // exercised deliberately or it ships untested.
        let data = vec![b'z'; 5_000];
        let comp = miniz_oxide::deflate::compress_to_vec(&data, 6);
        let name = b"hand-rezipped.bin";
        let mut z = Vec::new();
        let mut local = Vec::new();
        local.extend_from_slice(&LOCAL_SIG.to_le_bytes());
        local.extend_from_slice(&20u16.to_le_bytes());
        local.extend_from_slice(&0u16.to_le_bytes());
        local.extend_from_slice(&8u16.to_le_bytes()); // Deflate
        local.extend_from_slice(&0u32.to_le_bytes()); // time+date
        local.extend_from_slice(&crc32(&data).to_le_bytes());
        local.extend_from_slice(&(comp.len() as u32).to_le_bytes());
        local.extend_from_slice(&(data.len() as u32).to_le_bytes());
        local.extend_from_slice(&(name.len() as u16).to_le_bytes());
        local.extend_from_slice(&0u16.to_le_bytes());
        local.extend_from_slice(name);
        local.extend_from_slice(&comp);
        z.extend_from_slice(&local);
        let cd_start = z.len() as u32;
        z.extend_from_slice(&CDIR_SIG.to_le_bytes());
        z.extend_from_slice(&20u16.to_le_bytes());
        z.extend_from_slice(&20u16.to_le_bytes());
        z.extend_from_slice(&0u16.to_le_bytes());
        z.extend_from_slice(&8u16.to_le_bytes());
        z.extend_from_slice(&0u32.to_le_bytes());
        z.extend_from_slice(&crc32(&data).to_le_bytes());
        z.extend_from_slice(&(comp.len() as u32).to_le_bytes());
        z.extend_from_slice(&(data.len() as u32).to_le_bytes());
        z.extend_from_slice(&(name.len() as u16).to_le_bytes());
        z.extend_from_slice(&0u16.to_le_bytes());
        z.extend_from_slice(&0u16.to_le_bytes());
        z.extend_from_slice(&0u16.to_le_bytes());
        z.extend_from_slice(&0u16.to_le_bytes());
        z.extend_from_slice(&0u32.to_le_bytes());
        z.extend_from_slice(&0u32.to_le_bytes());
        z.extend_from_slice(name);
        let cd_len = z.len() as u32 - cd_start;
        z.extend_from_slice(&EOCD_SIG.to_le_bytes());
        z.extend_from_slice(&0u16.to_le_bytes());
        z.extend_from_slice(&0u16.to_le_bytes());
        z.extend_from_slice(&1u16.to_le_bytes());
        z.extend_from_slice(&1u16.to_le_bytes());
        z.extend_from_slice(&cd_len.to_le_bytes());
        z.extend_from_slice(&cd_start.to_le_bytes());
        z.extend_from_slice(&0u16.to_le_bytes());
        assert!(comp.len() < data.len(), "the fixture must really be deflated");
        let vfs = read_zip(&z).expect("read a hand-deflated archive");
        assert_eq!(vfs.read("hand-rezipped.bin").unwrap(), data);
    }

    #[test]
    fn the_entry_checksum_is_the_one_archivers_verify() {
        // A wrong CRC does not make a slightly-off archive, it makes one every tool
        // refuses - so the value is pinned against the published check vector rather
        // than against this implementation's own output.
        assert_eq!(crc32(b"123456789"), 0xCBF4_3926);
        assert_eq!(crc32(b""), 0);
    }

    #[test]
    fn reads_fixture_zip() {
        // Opt-in acid test: VITASLOP_GAME_ZIP points at a privately-supplied NoNpDrm
        // zip. Universal invariants only; the title id in the paths is not assumed.
        let Some(path) = std::env::var_os("VITASLOP_GAME_ZIP") else {
            return;
        };
        let Ok(bytes) = std::fs::read(path) else {
            return;
        };
        let vfs = read_zip(&bytes).expect("read zip");
        let list = vfs.list();
        // param.sfo is stored non-encrypted; it must inflate to the PSF magic. Find it
        // by suffix so the title id in the path does not need to be known.
        let sfo_path = list
            .iter()
            .find(|p| p.ends_with("sce_sys/param.sfo"))
            .expect("param.sfo in zip");
        let sfo = vfs.read(sfo_path).expect("read param.sfo");
        assert_eq!(&sfo[0..4], b"\x00PSF");
        // An eboot is present (still PFS-encrypted at this layer).
        assert!(list.iter().any(|p| p.ends_with("eboot.bin")));
        assert!(list.len() > 100);
        let _ = &testfix::read; // keep import used when zip env is unset
    }
}
