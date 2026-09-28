//! What the ingest's per-byte crypto costs, on the device that is running it.
//!
//! WASM ONLY, deliberately. On x86_64 and aarch64 the `aes` crate compiles to the
//! CPU's AES instructions and this question does not arise; on `wasm32` there are no
//! such instructions and it falls back to the constant-time fixslice software
//! backend, which is the slow path this measures. The desktop binary does not contain
//! any of this.
//!
//! WHY IT EXISTS. A phone import spends about three quarters of its time inside this
//! crypto (measured: `web/debug/import-speed.html`), and the browser can do the same
//! primitives in hardware through `crypto.subtle` at 116 MB/s. Whether that is worth
//! restructuring the ingest for depends on how much of the three quarters is the
//! ciphers themselves and how much is everything around them - the per-page `Vec`,
//! the copies, the wasm-bindgen crossings. WebCrypto replaces only the first. So each
//! primitive is timed on its own, through the REAL functions the import calls, and
//! the caller compares them against its own WebCrypto numbers.
//!
//! The clock comes from the caller (`now`, milliseconds) because wasm has no
//! `std::time::Instant`.

use super::keys;
use super::pfs::FileCtx;
use super::pfscrypt::{hmac_sha1, GameData};
use aes::cipher::{KeyIvInit, StreamCipher};

type Aes128Ctr = ctr::Ctr128BE<aes::Aes128>;

/// Megabytes per second for each stage of the per-byte chain.
pub struct Rates {
    /// The pkg layer: one AES-128-CTR pass over the whole data region.
    pub pkg_ctr: f64,
    /// The PFS integrity check: HMAC-SHA1 over each 0x8000 page of ciphertext.
    pub pfs_hmac: f64,
    /// The PFS page cipher: AES-128-CBC-CTS per page, verification off.
    pub pfs_cbc: f64,
    /// All three, as the import actually runs them: `decrypt_page` with verification
    /// on, over the same pages, after a CTR pass. This is the number to beat.
    pub chain: f64,
}

/// The page size every read-only title uses, and so the unit of PFS work.
const PAGE: usize = 0x8000;

/// Time `f` over `budget_ms` and return MB/s. `f` returns the bytes it processed.
fn rate(budget_ms: f64, now: &dyn Fn() -> f64, mut f: impl FnMut() -> usize) -> f64 {
    let t0 = now();
    let mut bytes = 0usize;
    while now() - t0 < budget_ms {
        bytes += f();
    }
    let secs = (now() - t0) / 1000.0;
    if secs <= 0.0 {
        return 0.0;
    }
    (bytes as f64 / (1024.0 * 1024.0)) / secs
}

/// Measure the chain. `budget_ms` is spent on EACH of the four stages.
pub fn crypto_rates(budget_ms: f64, now: &dyn Fn() -> f64) -> Rates {
    // A megabyte of nothing in particular: AES and SHA1 cost the same whatever the
    // bytes are, and a fixed pattern keeps the measurement repeatable.
    let mut buf = vec![0u8; 1024 * 1024];
    for (i, b) in buf.iter_mut().enumerate() {
        *b = (i * 31) as u8;
    }
    let key = [0x5au8; 16];
    let iv = [0x3cu8; 16];

    let pkg_ctr = rate(budget_ms, now, || {
        let mut c = Aes128Ctr::new((&key).into(), (&iv).into());
        c.apply_keystream(&mut buf);
        buf.len()
    });

    let pfs_hmac = rate(budget_ms, now, || {
        let mut n = 0;
        for page in buf.chunks(PAGE) {
            // Both hashes the import does per page: the per-page subkey, then the page.
            let subkey = hmac_sha1(&key, &0u32.to_le_bytes());
            let _ = hmac_sha1(&subkey, page);
            n += page.len();
        }
        n
    });

    // The real page decrypt, through the real type. `signatures` empty means
    // `decrypt_page` skips the integrity check, which is how the two halves are
    // separated: this stage is the cipher alone.
    let seed = [0u8; 20];
    let ctx = FileCtx {
        klicensee: &key,
        files_salt: 0,
        key_id: 0,
        icv_salt: 1,
        table_index: 0,
        iv_seed: &seed,
        has_dbseed: true,
        page_size: PAGE as u32,
        signatures: &[],
        plaintext_size: buf.len(),
        encrypted: true,
    };
    let gd = GameData::new(keys::PFS_F00D_CONTRACT);
    let page_keys = gd.page_keys(&ctx);
    let pfs_cbc = rate(budget_ms, now, || {
        let mut n = 0;
        for (i, page) in buf.chunks(PAGE).enumerate() {
            let _ = gd.decrypt_page(&ctx, &page_keys, i, page);
            n += page.len();
        }
        n
    });

    // Everything, in the order the import does it.
    let chain = rate(budget_ms, now, || {
        let mut c = Aes128Ctr::new((&key).into(), (&iv).into());
        c.apply_keystream(&mut buf);
        let mut n = 0;
        for (i, page) in buf.chunks(PAGE).enumerate() {
            let subkey = hmac_sha1(&key, &(i as u32).to_le_bytes());
            let _ = hmac_sha1(&subkey, page);
            let _ = gd.decrypt_page(&ctx, &page_keys, i, page);
            n += page.len();
        }
        n
    });

    Rates { pkg_ctr, pfs_hmac, pfs_cbc, chain }
}
