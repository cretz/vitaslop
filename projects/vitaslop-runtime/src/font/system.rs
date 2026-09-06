//! Where the SYSTEM font comes from, when there is one.
//!
//! # The problem this exists for
//! `sceFontOpen` / `scePvfOpen` open one of the console's own installed fonts by INDEX. Those
//! fonts are the vendor's assets and are not shipped here, so both calls used to refuse - and
//! refuse SILENTLY. A title that renders its strings through the system font therefore drew
//! every one of them from an all-zero glyph atlas, which reaches the screen as blank or BLACK
//! areas where dynamic text belongs. MEASURED on PCSA00009: one `sceFontOpen(0)` at boot, a
//! 1024x512 8-bit atlas at `0x8f8b4c00` that the guest zeroes and nothing ever fills, and two
//! UI quads sampling it as a flat black rectangle over the club list.
//!
//! # What this does, and what it does NOT do
//! It resolves the bytes of a font to STAND IN for the console's, from three places in order:
//!
//! 1. bytes handed in by the host embedder ([`set_bytes`]) - how the browser build supplies one,
//!    since it has no filesystem to probe and no environment to read;
//! 2. `VITASLOP_SYSTEM_FONT=<path>`, an explicit choice;
//! 3. on a native host only, the first of a short list of well-known host font paths.
//!
//! It does not ship a font, and it does not pretend a substitute is the real thing. The glyph
//! SHAPES will differ from the console's, and so will the metrics, so text may wrap or centre
//! slightly differently - which is why [`describe`] exists and why the callers report it once.
//! A substitute that says so is better than a black rectangle that says nothing; a substitute
//! that claims to be the console's font would be worse than both.
//!
//! # Why bytes and not a path all the way down
//! The browser is the target this project is for, and it has neither `std::env` nor a readable
//! host filesystem. Anything that resolved to a PATH would work on the desktop and be dead code
//! where it matters, so the interface below the knob is bytes, and the desktop's file read is
//! just one of the three ways to get them.

use std::sync::{Arc, OnceLock, RwLock};

/// The pixels-per-em the stand-in is rasterized at.
///
/// ScePgf has no set-char-size call at all - a PGF font's size is intrinsic to the file - so a
/// substitute has to be rasterized at SOMETHING, and this is it. 16 is the Vita system font's
/// working size for UI text; a title that drives ScePvf sets its own size over this immediately
/// and never sees it.
///
/// The number matters much less than it looks: a title sizes its glyph CELLS from the maxima
/// `sceFontGetFontInfo` reports, so text scales with this rather than overflowing or shrinking.
/// What made the text tiny was the maxima, not the size - see [`super::FontLibrary::face_metrics`].
pub const SUBSTITUTE_PX: f32 = 16.0;

/// Font bytes injected by the embedder, if any. Set before the guest runs.
static INJECTED: RwLock<Option<Arc<Vec<u8>>>> = RwLock::new(None);

/// Supply the system-font substitute's bytes directly.
///
/// This is the browser's route: `vitaslop-web` fetches a font alongside the page and hands the
/// buffer over before `run_game`, because there is no path to probe and no environment to read
/// there. Calling it after a font has already been opened has no effect on that open - the
/// resolution below is cached for the life of the process, so the answer cannot change halfway
/// through a run and leave two fonts in one frame.
pub fn set_bytes(bytes: Vec<u8>) {
    if let Ok(mut slot) = INJECTED.write() {
        *slot = Some(Arc::new(bytes));
    }
}

/// The resolved substitute: its bytes and a human-readable account of where they came from.
struct Resolved {
    bytes: Arc<Vec<u8>>,
    source: String,
}

fn resolve() -> &'static Option<Resolved> {
    static CELL: OnceLock<Option<Resolved>> = OnceLock::new();
    CELL.get_or_init(|| {
        if let Some(bytes) = INJECTED.read().ok().and_then(|s| s.clone()) {
            return Some(Resolved { bytes, source: "supplied by the host embedder".to_string() });
        }
        // An explicit path wins over any probe: someone who names a font means that font.
        if let Some(path) = crate::knobs::var("VITASLOP_SYSTEM_FONT").ok().filter(|p| !p.is_empty())
        {
            return match read_file(&path) {
                Some(bytes) => Some(Resolved { bytes: Arc::new(bytes), source: path }),
                None => {
                    // A named path that does not load is a MISTAKE, not a fallback: silently
                    // probing past it would hide the typo and leave the run looking as though
                    // no font was ever asked for.
                    tracing::warn!(
                        target: "vitaslop::cb",
                        path,
                        "VITASLOP_SYSTEM_FONT names a file that could not be read - no system \
                         font is available and the probe below is NOT tried, because a path \
                         given explicitly is a choice, not a hint"
                    );
                    None
                }
            };
        }
        probe_host_fonts()
    })
}

#[cfg(not(target_arch = "wasm32"))]
fn read_file(path: &str) -> Option<Vec<u8>> {
    std::fs::read(path).ok()
}

/// The browser has no host filesystem, so a path can only ever have come from a knob someone
/// set by hand - and there is nothing there to open. [`set_bytes`] is the route that works.
#[cfg(target_arch = "wasm32")]
fn read_file(_path: &str) -> Option<Vec<u8>> {
    None
}

/// The first readable font from a short list of standard host locations.
///
/// Deliberately SHORT and deliberately generic faces. The point is to have letterforms at all,
/// not to guess at a match for the console's own face; a longer list would only make which font
/// a run used less predictable. Nothing here is installed by this project, so on a host with
/// none of them the answer is `None` and the caller reports the refusal.
#[cfg(not(target_arch = "wasm32"))]
fn probe_host_fonts() -> Option<Resolved> {
    const CANDIDATES: &[&str] = &[
        // Windows
        "C:/Windows/Fonts/segoeui.ttf",
        "C:/Windows/Fonts/arial.ttf",
        "C:/Windows/Fonts/tahoma.ttf",
        // macOS
        "/System/Library/Fonts/Helvetica.ttc",
        "/Library/Fonts/Arial.ttf",
        // Linux
        "/usr/share/fonts/truetype/dejavu/DejaVuSans.ttf",
        "/usr/share/fonts/truetype/liberation/LiberationSans-Regular.ttf",
        "/usr/share/fonts/TTF/DejaVuSans.ttf",
        "/usr/share/fonts/noto/NotoSans-Regular.ttf",
    ];
    for path in CANDIDATES {
        if let Some(bytes) = read_file(path) {
            return Some(Resolved { bytes: Arc::new(bytes), source: (*path).to_string() });
        }
    }
    None
}

#[cfg(target_arch = "wasm32")]
fn probe_host_fonts() -> Option<Resolved> {
    None
}

/// The substitute's bytes, or `None` when this host has nothing to stand in with.
pub fn bytes() -> Option<Arc<Vec<u8>>> {
    resolve().as_ref().map(|r| r.bytes.clone())
}

/// Where the substitute came from, for the report the callers make once.
pub fn describe() -> Option<&'static str> {
    resolve().as_ref().map(|r| r.source.as_str())
}
