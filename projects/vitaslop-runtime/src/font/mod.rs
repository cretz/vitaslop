//! Font rasterization behind a swappable backend seam.
//!
//! ScePvf (the Vita's vector-font engine) is modeled here rather than in `vita/`
//! so the actual scaling/hinting/rasterization is a pluggable backend: the guest
//! ScePvf NID handlers ([`crate::vita::pvf`]) marshal Vita structs and call into a
//! [`FontLibrary`], which owns the backend, the open lib/font handles with their
//! size configuration, and a glyph cache.
//!
//! The default backend ([`skrifa_backend::SkrifaBackend`]) is pure Rust (skrifa for
//! table parsing + scaling + hinting, zeno for coverage rasterization), so it works
//! identically on native and `wasm32`. A native-only FreeType backend can implement
//! the same [`FontBackend`] trait later without touching the guest layer.
//!
//! Backend outputs are backend-neutral ([`GlyphMetrics`], [`GlyphBitmap`],
//! [`FaceMetrics`]) in fractional pixels; the ScePvf layer converts those into the
//! engine's 26.6 fixed-point (`*64`) struct fields.

pub mod skrifa_backend;

use std::collections::HashMap;

/// A backend-owned parsed font face, referred to by this opaque id.
pub type FaceId = u32;

/// The rasterization size, in pixels-per-em. ScePvf's `scePvfSetCharSize` allows an
/// independent horizontal and vertical size (anisotropic text); backends scale the
/// vertical em to `v` and stretch x by `h / v`.
#[derive(Clone, Copy, Debug, PartialEq)]
pub struct PixelSize {
    pub h: f32,
    pub v: f32,
}

impl PixelSize {
    /// A stable hashable key for the glyph cache (raw float bits).
    fn key(self) -> (u32, u32) {
        (self.h.to_bits(), self.v.to_bits())
    }
}

/// Per-glyph metrics in fractional pixels (y-up, baseline-relative), backend-neutral.
/// The ScePvf marshaller turns these into 26.6 fixed-point struct fields.
#[derive(Clone, Copy, Debug, Default)]
pub struct GlyphMetrics {
    /// Horizontal advance (pen movement) after drawing this glyph.
    pub h_advance: f32,
    /// Vertical advance (for vertical layout).
    pub v_advance: f32,
    /// Left side bearing: x from the pen origin to the glyph's left edge.
    pub h_bearing_x: f32,
    /// Top bearing: y from the baseline up to the glyph's top edge.
    pub h_bearing_y: f32,
    /// Vertical-layout bearings (approximated for horizontal-only titles).
    pub v_bearing_x: f32,
    pub v_bearing_y: f32,
    /// Glyph bounding-box extent.
    pub width: f32,
    pub height: f32,
    /// Top/bottom of the bounding box relative to the baseline (y-up).
    pub ascender: f32,
    pub descender: f32,
    /// Rasterized bitmap placement, in whole pixels: `left` from the pen origin,
    /// `top` from the baseline up to the first bitmap row.
    pub bitmap_left: i32,
    pub bitmap_top: i32,
    pub bitmap_width: u32,
    pub bitmap_height: u32,
}

/// An 8-bit coverage bitmap (row-major, one byte per pixel, top row first).
#[derive(Clone, Default)]
pub struct GlyphBitmap {
    pub width: u32,
    pub height: u32,
    pub coverage: Vec<u8>,
}

/// Face-wide metrics at a given size, in fractional pixels.
#[derive(Clone, Copy, Debug, Default)]
pub struct FaceMetrics {
    pub ascender: f32,
    pub descender: f32,
    /// Line height (ascender - descender + line gap).
    pub height: f32,
    /// Maximum horizontal advance across the face.
    pub max_advance: f32,
    pub num_glyphs: u32,
}

/// The swappable rasterization backend. Object-safe so a `#[cfg]`-gated native
/// backend (e.g. FreeType) can be boxed in alongside the default pure-Rust one.
/// `ch` is a Unicode scalar value (ScePvf passes UCS-2 char codes).
pub trait FontBackend: Send {
    /// Parse a font from raw file bytes; returns its face id, or `None` if the bytes
    /// are not a font the backend can read.
    fn load_face(&mut self, bytes: &[u8]) -> Option<FaceId>;
    /// Whether the face has a glyph for this character.
    fn has_glyph(&self, face: FaceId, ch: u32) -> bool;
    /// Face-wide metrics at a size.
    fn face_metrics(&self, face: FaceId, size: PixelSize) -> Option<FaceMetrics>;
    /// Rasterize a glyph, returning its coverage bitmap and metrics. `None` if the
    /// face has no such glyph. A whitespace glyph yields an empty (0x0) bitmap with
    /// valid advance metrics. Takes `&mut self` so a backend can memoize expensive
    /// per-size setup (the skrifa backend caches its hinting instances).
    fn rasterize(&mut self, face: FaceId, size: PixelSize, ch: u32) -> Option<(GlyphBitmap, GlyphMetrics)>;
}

/// A ScePvf library instance: unit configuration shared by the fonts opened under it.
struct LibState {
    /// EM value set by `scePvfSetEM` (informational for the current pixel-size model).
    em: f32,
    /// Horizontal/vertical resolution (DPI) for pixel<->point conversion.
    h_res: f32,
    v_res: f32,
}

impl Default for LibState {
    fn default() -> Self {
        // 72 DPI makes point == pixel until the title sets a resolution, and a 0 EM
        // reads as "unset".
        LibState { em: 0.0, h_res: 72.0, v_res: 72.0 }
    }
}

/// An open ScePvf font: the backend face plus the current rasterization size.
struct FontState {
    lib: u32,
    face: FaceId,
    size: PixelSize,
}

/// Cached rasterization keyed by face + size + character.
#[derive(Clone, Copy, PartialEq, Eq, Hash)]
struct GlyphKey {
    face: FaceId,
    size: (u32, u32),
    ch: u32,
}

/// A memoized glyph: its coverage bitmap and metrics.
struct CachedGlyph {
    bitmap: GlyphBitmap,
    metrics: GlyphMetrics,
}

/// The stateful ScePvf model: owns the backend, the open lib/font handles, and the
/// glyph cache. Lives in [`crate::host::VitaState`]. Glyph caching sits above the
/// backend so every backend benefits and a title re-rendering the same UI text each
/// frame only rasterizes each glyph once.
pub struct FontLibrary {
    backend: Box<dyn FontBackend>,
    libs: HashMap<u32, LibState>,
    fonts: HashMap<u32, FontState>,
    cache: HashMap<GlyphKey, CachedGlyph>,
    /// Monotonic handle counter shared by lib and font ids so the two id spaces never
    /// collide (a stale font id can never be mistaken for a live lib id).
    next_handle: u32,
}

impl Default for FontLibrary {
    fn default() -> Self {
        FontLibrary::new(Box::new(skrifa_backend::SkrifaBackend::default()))
    }
}

impl FontLibrary {
    pub fn new(backend: Box<dyn FontBackend>) -> Self {
        FontLibrary {
            backend,
            libs: HashMap::new(),
            fonts: HashMap::new(),
            cache: HashMap::new(),
            // Start above zero (a null handle is the ScePvf error sentinel) and in a
            // high range so an opaque handle never looks like a small array index.
            next_handle: 0x0001_0000,
        }
    }

    fn alloc_handle(&mut self) -> u32 {
        let h = self.next_handle;
        self.next_handle = self.next_handle.wrapping_add(1);
        h
    }

    // --- library lifecycle / units --------------------------------------------

    /// `scePvfNewLib`: create a library instance, returning its handle.
    pub fn new_lib(&mut self) -> u32 {
        let h = self.alloc_handle();
        self.libs.insert(h, LibState::default());
        h
    }

    /// `scePvfDoneLib`: destroy a library and every font opened under it.
    pub fn done_lib(&mut self, lib: u32) -> bool {
        if self.libs.remove(&lib).is_none() {
            return false;
        }
        self.fonts.retain(|_, f| f.lib != lib);
        true
    }

    pub fn lib_exists(&self, lib: u32) -> bool {
        self.libs.contains_key(&lib)
    }

    /// `scePvfSetEM`.
    pub fn set_em(&mut self, lib: u32, em: f32) -> bool {
        match self.libs.get_mut(&lib) {
            Some(l) => {
                l.em = em;
                true
            }
            None => false,
        }
    }

    /// `scePvfSetResolution`.
    pub fn set_resolution(&mut self, lib: u32, h_res: f32, v_res: f32) -> bool {
        match self.libs.get_mut(&lib) {
            Some(l) => {
                l.h_res = h_res;
                l.v_res = v_res;
                true
            }
            None => false,
        }
    }

    /// `scePvfPixelToPointH/V`: point = pixel * 72 / resolution. A missing lib or a
    /// non-positive resolution yields 0.
    pub fn pixel_to_point(&self, lib: u32, pixel: f32, vertical: bool) -> f32 {
        let Some(l) = self.libs.get(&lib) else { return 0.0 };
        let res = if vertical { l.v_res } else { l.h_res };
        if res > 0.0 {
            pixel * 72.0 / res
        } else {
            0.0
        }
    }

    // --- font lifecycle -------------------------------------------------------

    /// `scePvfOpenUserFile`: parse the font bytes and open a font under `lib`.
    /// Returns the font handle, or `None` if the lib is unknown or the bytes do not
    /// parse as a font.
    pub fn open_user_file(&mut self, lib: u32, bytes: &[u8]) -> Option<u32> {
        if !self.libs.contains_key(&lib) {
            return None;
        }
        let face = self.backend.load_face(bytes)?;
        let h = self.alloc_handle();
        // Default to a 16px em until the title sets a size, so an immediate metrics
        // query does not divide by zero.
        self.fonts.insert(h, FontState { lib, face, size: PixelSize { h: 16.0, v: 16.0 } });
        Some(h)
    }

    /// `scePvfOpenUserMemory`: open a font from bytes the title already holds in GUEST
    /// MEMORY, rather than from a file.
    ///
    /// Identical to [`Self::open_user_file`] once the bytes are in hand - the difference
    /// is entirely on the caller's side (a guest pointer and a length instead of a path),
    /// so they share everything below the read. Titles use this for a font they have
    /// already loaded into their own heap, or one packed inside an archive they unpack
    /// themselves, which a path-based open cannot reach at all.
    pub fn open_user_memory(&mut self, lib: u32, bytes: &[u8]) -> Option<u32> {
        self.open_user_file(lib, bytes)
    }

    /// `scePvfClose`: drop a font handle.
    ///
    /// Returns whether the handle was one this library had open, so the caller can report
    /// `SCE_PVF_ERROR_ARG` for a handle that was never issued (or was closed twice) rather
    /// than succeeding - a double close that reads as success hides a title's own
    /// use-after-free from it.
    ///
    /// The glyph cache is keyed by `(face, size, char)`, so the closed font's entries
    /// become unreachable the moment its face goes; they are dropped here rather than left
    /// to accumulate for the life of a run that opens and closes fonts per screen.
    ///
    /// The face is only dropped when NO other open font is using it, so closing one of two
    /// handles on the same face does not pull the cache out from under the other.
    pub fn close(&mut self, font: u32) -> bool {
        let Some(state) = self.fonts.remove(&font) else { return false };
        let face = state.face;
        if self.fonts.values().any(|f| f.face == face) {
            return true;
        }
        self.cache.retain(|k, _| k.face != face);
        true
    }

    pub fn font_exists(&self, font: u32) -> bool {
        self.fonts.contains_key(&font)
    }

    /// `scePvfSetCharSize`: set the pixel em size for a font. Invalidating any cached
    /// glyphs is unnecessary - the cache is keyed by size, so a new size simply misses.
    pub fn set_char_size(&mut self, font: u32, h: f32, v: f32) -> bool {
        match self.fonts.get_mut(&font) {
            Some(f) => {
                f.size = PixelSize { h, v };
                true
            }
            None => false,
        }
    }

    // --- glyph queries (cached) -----------------------------------------------

    /// `scePvfIsElement`: whether the font can render this character.
    pub fn has_glyph(&self, font: u32, ch: u32) -> bool {
        match self.fonts.get(&font) {
            Some(f) => self.backend.has_glyph(f.face, ch),
            None => false,
        }
    }

    /// `scePvfGetFontInfo`: face-wide metrics at the font's current size.
    pub fn face_metrics(&self, font: u32) -> Option<FaceMetrics> {
        let f = self.fonts.get(&font)?;
        self.backend.face_metrics(f.face, f.size)
    }

    /// The cached (rasterizing on first use) glyph for `scePvfGetCharInfo`,
    /// `scePvfGetCharImageRect`, and `scePvfGetCharGlyphImage`. Returns the metrics
    /// and coverage bitmap, or `None` if the font handle or glyph is unknown.
    pub fn glyph(&mut self, font: u32, ch: u32) -> Option<(&GlyphBitmap, &GlyphMetrics)> {
        // Copy the face/size out so the fonts borrow ends before the mutable backend
        // call (the backend memoizes per-size hinting state).
        let (face, size) = {
            let f = self.fonts.get(&font)?;
            (f.face, f.size)
        };
        let key = GlyphKey { face, size: size.key(), ch };
        if !self.cache.contains_key(&key) {
            let (bitmap, metrics) = self.backend.rasterize(face, size, ch)?;
            self.cache.insert(key, CachedGlyph { bitmap, metrics });
        }
        let g = self.cache.get(&key)?;
        Some((&g.bitmap, &g.metrics))
    }
}

#[cfg(test)]
mod tests {
    //! The backend-agnostic library logic: handle lifecycle, unit conversion, and the
    //! glyph cache (rasterize-once). Backed by a mock so the tests do not depend on a
    //! font file - the real skrifa backend is covered end to end by the `vita_pvf`
    //! conformance case.
    use super::*;
    use std::sync::atomic::{AtomicU32, Ordering};
    use std::sync::Arc;

    /// A stand-in backend: every char < 128 is a glyph, and each `rasterize` bumps a
    /// shared counter so a test can assert the cache prevents repeat work.
    struct MockBackend {
        raster_calls: Arc<AtomicU32>,
    }

    impl FontBackend for MockBackend {
        fn load_face(&mut self, _bytes: &[u8]) -> Option<FaceId> {
            Some(7)
        }
        fn has_glyph(&self, _face: FaceId, ch: u32) -> bool {
            ch < 128
        }
        fn face_metrics(&self, _face: FaceId, _size: PixelSize) -> Option<FaceMetrics> {
            Some(FaceMetrics { ascender: 12.0, descender: -3.0, height: 16.0, max_advance: 16.0, num_glyphs: 128 })
        }
        fn rasterize(&mut self, _face: FaceId, size: PixelSize, _ch: u32) -> Option<(GlyphBitmap, GlyphMetrics)> {
            self.raster_calls.fetch_add(1, Ordering::Relaxed);
            let m = GlyphMetrics { h_advance: size.v, ..GlyphMetrics::default() };
            Some((GlyphBitmap { width: 2, height: 2, coverage: vec![255; 4] }, m))
        }
    }

    fn lib_with_counter() -> (FontLibrary, Arc<AtomicU32>) {
        let calls = Arc::new(AtomicU32::new(0));
        let lib = FontLibrary::new(Box::new(MockBackend { raster_calls: calls.clone() }));
        (lib, calls)
    }

    #[test]
    fn lib_and_font_lifecycle() {
        let (mut f, _) = lib_with_counter();
        let lib = f.new_lib();
        assert!(lib != 0 && f.lib_exists(lib));
        let font = f.open_user_file(lib, b"anything").expect("open");
        assert!(f.font_exists(font));
        // Opening under an unknown lib fails.
        assert!(f.open_user_file(lib + 999, b"x").is_none());
        // DoneLib drops the lib and its fonts.
        assert!(f.done_lib(lib));
        assert!(!f.lib_exists(lib) && !f.font_exists(font));
        assert!(!f.done_lib(lib)); // already gone
    }

    #[test]
    fn set_calls_reject_unknown_handles() {
        let (mut f, _) = lib_with_counter();
        assert!(!f.set_em(1, 72.0));
        assert!(!f.set_resolution(1, 72.0, 72.0));
        assert!(!f.set_char_size(1, 16.0, 16.0));
        let lib = f.new_lib();
        assert!(f.set_em(lib, 72.0) && f.set_resolution(lib, 72.0, 72.0));
    }

    #[test]
    fn pixel_to_point_uses_resolution() {
        let (mut f, _) = lib_with_counter();
        let lib = f.new_lib();
        // Default 72 dpi: point == pixel.
        assert_eq!(f.pixel_to_point(lib, 16.0, false), 16.0);
        // 144 dpi halves the point value.
        f.set_resolution(lib, 144.0, 36.0);
        assert_eq!(f.pixel_to_point(lib, 16.0, false), 8.0);
        assert_eq!(f.pixel_to_point(lib, 16.0, true), 32.0);
        // Unknown lib yields 0.
        assert_eq!(f.pixel_to_point(lib + 1, 16.0, false), 0.0);
    }

    #[test]
    fn glyph_cache_rasterizes_once_per_key() {
        let (mut f, calls) = lib_with_counter();
        let lib = f.new_lib();
        let font = f.open_user_file(lib, b"x").unwrap();
        f.set_char_size(font, 16.0, 16.0);
        assert!(f.glyph(font, u32::from('A')).is_some());
        assert!(f.glyph(font, u32::from('A')).is_some()); // cache hit
        assert_eq!(calls.load(Ordering::Relaxed), 1, "second query must hit the cache");
        // A different char rasterizes again.
        assert!(f.glyph(font, u32::from('B')).is_some());
        assert_eq!(calls.load(Ordering::Relaxed), 2);
        // A different size is a distinct key.
        f.set_char_size(font, 24.0, 24.0);
        assert!(f.glyph(font, u32::from('A')).is_some());
        assert_eq!(calls.load(Ordering::Relaxed), 3);
    }
}
