//! Embedded key material, as bare constants.
//!
//! These are fixed, published console keys the ingest layer needs to decrypt a
//! container we own. Each is named for what it unlocks; nothing records where a
//! value came from.

/// Decode a hex string literal to a byte array at compile time.
pub(crate) const fn hex<const N: usize>(s: &str) -> [u8; N] {
    let b = s.as_bytes();
    assert!(b.len() == N * 2, "hex length mismatch");
    let mut out = [0u8; N];
    let mut i = 0;
    while i < N {
        out[i] = (nib(b[i * 2]) << 4) | nib(b[i * 2 + 1]);
        i += 1;
    }
    out
}

const fn nib(c: u8) -> u8 {
    match c {
        b'0'..=b'9' => c - b'0',
        b'a'..=b'f' => c - b'a' + 10,
        b'A'..=b'F' => c - b'A' + 10,
        _ => panic!("bad hex digit"),
    }
}

/// AES-128-CTR key for PS Vita NPDRM `.pkg` transport decryption (Type 2). The
/// per-pkg session key is `AES-ECB(this, iv)`.
pub const PKG_TYPE2: [u8; 16] = hex("E31A70C9CE1DD72BF3C0622963F2ECCB");

/// AES-128 key that decrypts the sealed savedata secret (pfsSKKey EncKey).
pub const PFS_SK_ENC: [u8; 16] = hex("00298CDF4428E72C8785DAE0923C60BD");
/// HMAC key that authenticates the sealed savedata secret (pfsSKKey Secret).
pub const PFS_SK_SECRET: [u8; 16] = hex("8C5D3A4B9D9BF4B453BCE6CDC34331D8");

/// HMAC-SHA1 key that produces the PFS integrity `secret` (keys the salted digest
/// of files_salt/icv_salt that is then CBC-CTS-wrapped under the F00D drv_key).
pub const PFS_INTEGRITY_BASE: [u8; 20] = hex("AFE656BB3C17256A3C809F6E9BF19FDD5A388543");
/// HMAC-SHA1 key that produces the PFS content tweak (sector IV mask) key from a
/// file's `dbseed`.
pub const PFS_TWEAK_BASE: [u8; 20] = hex("E462258B1F3121560745DB62B1436723D2BF80FE");
/// Fixed AES-128-CBC IV used when CBC-CTS-wrapping the integrity secret.
pub const PFS_FIXED_CBC_IV: [u8; 16] = hex("74D20CC39881C213EE770B1010E4BEA7");

/// The F00D keygen constant: a title's `drv_key` is
/// `AES-128-ECB-decrypt(klicensee, this)`. This is the one value that makes PFS
/// gamedata decryption computable offline.
pub const PFS_F00D_CONTRACT: [u8; 16] = hex("E12213B48016B0E99AB81F8EC02AD4A2");

/// HMAC-SHA256 key verifying the keystone digest.
pub const KEYSTONE_KS_SECRET: [u8; 32] =
    hex("310C2F2D70A62226F4582B4FF03E24196EEF01EF73A8981F2504BD50549A478F");

/// AES-256-CBC key + IV that decrypt an application SELF's MetadataInfo, indexed
/// by the SCE header's `key_revision` (0..=5). Retail (external) key rows.
pub const SELF_METADATA_APP: [([u8; 32], [u8; 16]); 6] = [
    (
        hex("5661E5FB20CFD1D1DFF50C1E59A6EA977D0AA5C5770F53B9CDD4E9451FFF55CB"),
        hex("23D02FF79BF430E2D123869BF0CACAA0"),
    ),
    (
        hex("4181B2DF5F5D94D3C80B7D86EACF1928533A49BA58EDE2B43CDEE7E572568BD4"),
        hex("B1678C0543B6C1997B63A6F4F3C8FD33"),
    ),
    (
        hex("5282582F17F068F89A260AAFB71C58928F45A8D08C681376B07FF9EAB1114226"),
        hex("29672DF43E426F41AF46D42E8437D449"),
    ),
    (
        hex("270CBA370061B87077672ADB5142D18844AAED352A9CCEE63602B0D740594334"),
        hex("1CF2454FBF47D76221B91AFC3B608C28"),
    ),
    (
        hex("A782BC5A9EDDFC49A513FF3E592C4677A8C8920F23C9F11F2558FB9D99A43868"),
        hex("559B5E658559EB65EBF892C274E098A9"),
    ),
    (
        hex("12D64D0172495226010A687DE245A73DE028B3561E25E69BABC325636F3CAE0A"),
        hex("F149EED1757E5A915B24309795BFC380"),
    ),
];

/// AES-128-CBC NPDRM key (IV is zero) that unwraps the klicensee predecrypt for
/// an application SELF. Row 0 applies to `key_revision` 0..=2, row 1 to 3..=5.
pub const SELF_NPDRM_APP: [[u8; 16]; 2] = [
    hex("C10368BF3D2943BC6E5BD05E46A9A7B6"),
    hex("16419DD3BFBE8BDC596929B72CE237CD"),
];

/// Select the NPDRM key row for a SELF `key_revision`. The psdevwiki groups
/// NPDRM key 0 with application key revisions 0..=2 and key 1 with 3..=5.
pub const fn self_npdrm_row(key_revision: u8) -> usize {
    if key_revision <= 2 { 0 } else { 1 }
}

/// The AES-256-CBC metadata key+iv for an application SELF `key_revision`, or
/// `None` if out of range.
pub fn metadata_app(key_revision: usize) -> Option<([u8; 32], [u8; 16])> {
    SELF_METADATA_APP.get(key_revision).copied()
}

#[cfg(test)]
mod tests {
    use super::*;
    #[test]
    fn hex_decodes() {
        assert_eq!(PKG_TYPE2[0], 0xE3);
        assert_eq!(PKG_TYPE2[15], 0xCB);
        assert_eq!(PFS_SK_ENC[0], 0x00);
        assert_eq!(KEYSTONE_KS_SECRET.len(), 32);
    }
}
