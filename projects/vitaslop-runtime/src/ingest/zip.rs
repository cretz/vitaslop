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

#[cfg(test)]
mod tests {
    use super::*;
    use super::super::vfs::Vfs;
    use crate::ingest::testfix;

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
