//! The default pure-Rust font backend: skrifa for table parsing + scaling +
//! hinting, zeno for coverage rasterization. Compiles identically for native and
//! `wasm32`, so the browser and desktop render text the same way.
//!
//! Hinting uses skrifa's default [`HintingOptions`] (auto-fallback engine, smooth
//! normal target), which mirrors FreeType's behavior - the TrueType bytecode
//! interpreter when the font carries reliable instructions, otherwise the autohinter.
//! The hinted advance/left-side-bearing from [`AdjustedMetrics`] are used verbatim, so
//! metrics match what a FreeType-based libpvf would report.

use std::collections::HashMap;
use std::sync::Arc;

use skrifa::instance::{LocationRef, Size};
use skrifa::outline::{DrawSettings, HintingInstance, HintingOptions, OutlinePen};
use skrifa::{FontRef, GlyphId, MetadataProvider};
use zeno::{Command, Format, Mask, Point};

use super::{FaceId, FaceMetrics, FontBackend, GlyphBitmap, GlyphMetrics, PixelSize};

/// A pure-Rust backend. Owns the raw font bytes (a `FontRef` is rebuilt per call -
/// that is only a table-directory parse) and caches one [`HintingInstance`] per
/// (face, size) since building the hinter is the costly part; the actual glyph
/// bitmaps are cached one level up in [`super::FontLibrary`].
#[derive(Default)]
pub struct SkrifaBackend {
    /// Font bytes behind an `Arc` so `rasterize` can cheaply take an owned handle,
    /// releasing the borrow of `self` before it mutates the hinter cache.
    faces: Vec<Arc<[u8]>>,
    hinters: HashMap<(FaceId, (u32, u32)), HintingInstance>,
}

impl SkrifaBackend {
    fn bytes(&self, face: FaceId) -> Option<Arc<[u8]>> {
        self.faces.get(face as usize).cloned()
    }
}

/// An [`OutlinePen`] that records the drawn contour as zeno commands, flipping y
/// (fonts are y-up, zeno rasterizes y-down) and stretching x for anisotropic sizes.
struct ZenoPen {
    cmds: Vec<Command>,
    x_scale: f32,
}

impl ZenoPen {
    fn pt(&self, x: f32, y: f32) -> Point {
        Point::new(x * self.x_scale, -y)
    }
}

impl OutlinePen for ZenoPen {
    fn move_to(&mut self, x: f32, y: f32) {
        let p = self.pt(x, y);
        self.cmds.push(Command::MoveTo(p));
    }
    fn line_to(&mut self, x: f32, y: f32) {
        let p = self.pt(x, y);
        self.cmds.push(Command::LineTo(p));
    }
    fn quad_to(&mut self, cx0: f32, cy0: f32, x: f32, y: f32) {
        let c = self.pt(cx0, cy0);
        let p = self.pt(x, y);
        self.cmds.push(Command::QuadTo(c, p));
    }
    fn curve_to(&mut self, cx0: f32, cy0: f32, cx1: f32, cy1: f32, x: f32, y: f32) {
        let c0 = self.pt(cx0, cy0);
        let c1 = self.pt(cx1, cy1);
        let p = self.pt(x, y);
        self.cmds.push(Command::CurveTo(c0, c1, p));
    }
    fn close(&mut self) {
        self.cmds.push(Command::Close);
    }
}

impl FontBackend for SkrifaBackend {
    fn load_face(&mut self, bytes: &[u8]) -> Option<FaceId> {
        // Validate it parses before taking ownership; reject non-font bytes.
        FontRef::new(bytes).ok()?;
        let id = self.faces.len() as FaceId;
        self.faces.push(Arc::from(bytes));
        Some(id)
    }

    fn has_glyph(&self, face: FaceId, ch: u32) -> bool {
        let Some(bytes) = self.bytes(face) else { return false };
        let Ok(font) = FontRef::new(&bytes) else { return false };
        font.charmap().map(ch).is_some()
    }

    fn face_metrics(&self, face: FaceId, size: PixelSize) -> Option<FaceMetrics> {
        let bytes = self.bytes(face)?;
        let font = FontRef::new(&bytes).ok()?;
        let m = font.metrics(Size::new(size.v), LocationRef::default());
        let x_scale = if size.v > 0.0 { size.h / size.v } else { 1.0 };
        Some(FaceMetrics {
            ascender: m.ascent,
            descender: m.descent,
            height: m.ascent - m.descent + m.leading,
            max_advance: m.max_width.unwrap_or(m.ascent - m.descent) * x_scale,
            num_glyphs: m.glyph_count as u32,
        })
    }

    fn rasterize(&mut self, face: FaceId, size: PixelSize, ch: u32) -> Option<(GlyphBitmap, GlyphMetrics)> {
        let bytes = self.bytes(face)?;
        let font = FontRef::new(&bytes).ok()?;
        let gid: GlyphId = font.charmap().map(ch)?;
        let px = Size::new(size.v);
        let x_scale = if size.v > 0.0 { size.h / size.v } else { 1.0 };

        // Linear (unhinted) metrics as the baseline; the hinted values from draw()
        // override advance/LSB below to match FreeType.
        let gm = font.glyph_metrics(px, LocationRef::default());
        let linear_advance = gm.advance_width(gid).unwrap_or(0.0);
        let bounds = gm.bounds(gid);

        let outlines = font.outline_glyphs();
        let outline = outlines.get(gid)?;

        // One hinter per (face, size); reused across glyphs.
        let key = (face, (size.v.to_bits(), size.h.to_bits()));
        if !self.hinters.contains_key(&key) {
            let hinter = HintingInstance::new(&outlines, px, LocationRef::default(), HintingOptions::default()).ok()?;
            self.hinters.insert(key, hinter);
        }
        let hinter = self.hinters.get(&key)?;

        let mut pen = ZenoPen { cmds: Vec::new(), x_scale };
        let adjusted = outline.draw(DrawSettings::hinted(hinter, false), &mut pen).ok()?;

        let (coverage, placement) = Mask::new(&pen.cmds[..]).format(Format::Alpha).render();

        // Hinted advance/LSB (FreeType-equivalent) where the scaler produced them.
        let h_advance = adjusted.advance_width.unwrap_or(linear_advance) * x_scale;
        let h_bearing_x = adjusted.lsb.unwrap_or_else(|| bounds.map(|b| b.x_min).unwrap_or(0.0)) * x_scale;
        let (bb_w, bb_h, asc, desc) = match bounds {
            Some(b) => ((b.x_max - b.x_min) * x_scale, b.y_max - b.y_min, b.y_max, b.y_min),
            None => (0.0, 0.0, 0.0, 0.0),
        };

        let metrics = GlyphMetrics {
            h_advance,
            // No vertical-layout data from the font here; a plausible advance keeps
            // any vertical-text query defined (horizontal titles never read it).
            v_advance: asc - desc,
            h_bearing_x,
            h_bearing_y: asc,
            v_bearing_x: -(bb_w / 2.0),
            v_bearing_y: asc,
            width: bb_w,
            height: bb_h,
            ascender: asc,
            descender: desc,
            bitmap_left: placement.left,
            // zeno's TopLeft placement.top is the (negative) offset from the baseline
            // to the top row; FreeType's bitmap_top is that distance measured upward.
            bitmap_top: -placement.top,
            bitmap_width: placement.width,
            bitmap_height: placement.height,
        };

        let bitmap = GlyphBitmap {
            width: placement.width,
            height: placement.height,
            coverage,
        };
        Some((bitmap, metrics))
    }
}
