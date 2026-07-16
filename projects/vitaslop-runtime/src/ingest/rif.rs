//! `sce_sys/package/work.bin` - the NPDRM RIF (Rights Information File) that
//! carries the license key.
//!
//! For a game we own, the RIF holds the `klicensee` - the 16-byte secret that
//! seeds PFS decryption of gamedata. A NoNpDrm dump writes a "fake" RIF with a
//! fixed account id (`0x0123456789ABCDEF`); the 16 bytes at offset 0x50 are the
//! klicensee used verbatim (no decrypt - a NoNpDrm/zRIF license carries the key
//! in the clear, unlike a console-bound retail RIF).
//!
//! Layout: version/type flags at 0x00, the account id at 0x08, the 0x30-byte
//! content id at 0x10, and the klicensee at 0x50.

use super::{Error, Reader};

const AID_OFF: usize = 0x08;
const CONTENT_ID_OFF: usize = 0x10;
const CONTENT_ID_LEN: usize = 0x30;
const KEY_OFF: usize = 0x50;
const KEY_LEN: usize = 0x10;
/// The account id NoNpDrm stamps into a fake RIF.
pub const NONPDRM_FAKE_AID: u64 = 0x0123_4567_89AB_CDEF;

/// A parsed RIF.
#[derive(Debug, Clone)]
pub struct Rif {
    /// Account id (`NONPDRM_FAKE_AID` for a NoNpDrm fake RIF).
    pub account_id: u64,
    /// Content id, e.g. `"XXYYYY-ABCD00001_00-0123456789ABCDEF"`.
    pub content_id: String,
    /// The 16-byte klicensee at 0x50 (used verbatim for a NoNpDrm license).
    pub key: [u8; KEY_LEN],
}

impl Rif {
    pub fn parse(bytes: &[u8]) -> Result<Rif, Error> {
        let r = Reader::new(bytes);
        let account_id = r.u64(AID_OFF)?;
        let cid = r
            .bytes::<CONTENT_ID_LEN>(CONTENT_ID_OFF)?;
        let n = cid.iter().position(|&b| b == 0).unwrap_or(cid.len());
        let content_id = String::from_utf8_lossy(&cid[..n]).into_owned();
        let key = r.bytes::<KEY_LEN>(KEY_OFF)?;
        Ok(Rif {
            account_id,
            content_id,
            key,
        })
    }

    /// True if this looks like a NoNpDrm fake RIF (fixed account id).
    pub fn is_fake(&self) -> bool {
        self.account_id == NONPDRM_FAKE_AID
    }

    /// The 16-byte klicensee - the key field, used as-is (no decrypt).
    pub fn klicensee(&self) -> [u8; 16] {
        self.key
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::ingest::testfix;

    #[test]
    fn parses_fixture_rif() {
        // Opt-in acid test against a privately-supplied dump (skips if absent).
        // Asserts only universal NoNpDrm RIF invariants, no title-specific values.
        let Some(bytes) = testfix::read("sce_sys/package/work.bin") else {
            return;
        };
        let rif = Rif::parse(&bytes).expect("parse work.bin");
        assert_eq!(rif.account_id, NONPDRM_FAKE_AID);
        assert!(rif.is_fake());
        // A content id is present and printable ASCII (format is title-independent).
        assert!(!rif.content_id.is_empty());
        assert!(rif.content_id.bytes().all(|b| b.is_ascii_graphic() || b == b'-' || b == b'_'));
        // The key field is present and non-zero.
        assert_ne!(rif.key, [0u8; 16]);
    }
}
