//! What the host remembers about each guest `SceGxmTexture*` beyond its control words: the
//! exact format the guest initialised it with, the palette it set, and the memoised answers
//! of the nearby-texture diagnostic.
//!
//! [`TextureRegistry`] owns all three, keyed by the guest struct address. The per-binding
//! decode memos those answers feed live in the snapshot cache, so invalidating them when a
//! format changes stays with `VitaState::set_texture_format`.

use crate::fasthash::FxHashMap;
use std::collections::HashMap;

/// Per-texture host state. See the module docs.
#[derive(Default)]
pub(super) struct TextureRegistry {
    /// Exact `SceGxmTextureFormat` last set on a guest `SceGxmTexture*` via
    /// `sceGxmTextureInit*`/`SetFormat`. The 16-byte control words alone lose the
    /// channel swizzle (only a 3-bit field survives), so we keep the full 32-bit
    /// format the guest passed for an exact decode; a texture the guest fills
    /// directly (a `.gxt` blob) is absent here and falls back to control-word parse.
    ///
    /// Maps, not lists: these are looked up once per BOUND TEXTURE UNIT per DRAW, while
    /// they grow with every distinct texture the title has ever touched. A linear scan
    /// made the capture cost of a frame grow with the size of the level.
    formats: FxHashMap<u32, u32>,
    /// Memoised answers from [`TextureRegistry::nearest`], keyed by the NULL handle address
    /// that asked. That search is a linear scan of `formats`, and it feeds a report that
    /// fires once per (unit, address) - see `VitaState::snapshot_bound_textures`, where
    /// running it on the ordinary by-value binding path cost 46% of a race frame.
    nearby: FxHashMap<u32, Option<(i64, u32)>>,
    /// `SceGxmTexture*` -> the palette pointer set by `sceGxmTextureSetPalette`.
    /// Kept beside the control words (rather than packed into word 0) because the
    /// palette field's bit layout in the control words is not published, and a wrong
    /// packing would corrupt fields that ARE understood.
    palettes: HashMap<u32, u32>,
}

impl TextureRegistry {
    /// Record the format set on the texture at `addr`; true when it differs from the one
    /// recorded before (or none was).
    pub(super) fn set_format(&mut self, addr: u32, format: u32) -> bool {
        self.formats.insert(addr, format) != Some(format)
    }

    #[inline]
    pub(super) fn format(&self, addr: u32) -> Option<u32> {
        self.formats.get(&addr).copied()
    }

    /// The initialised texture nearest `addr` within `+-window` bytes, as `(signed byte delta,
    /// its format)`. A linear scan - see [`TextureRegistry::memo_nearby`] for the memo.
    pub(super) fn nearest(&self, addr: u32, window: u32) -> Option<(i64, u32)> {
        self.formats
            .iter()
            .filter_map(|(&a, &f)| {
                let d = i64::from(a) - i64::from(addr);
                (d != 0 && d.unsigned_abs() <= u64::from(window)).then_some((d, f))
            })
            .min_by_key(|(d, _)| d.abs())
    }

    /// Resolve and remember the nearby-texture answer for a NULL handle at `addr` that has no
    /// format of its own, unless one is already remembered.
    pub(super) fn memo_nearby(&mut self, addr: u32, window: u32) {
        if self.format(addr).is_none() && !self.nearby.contains_key(&addr) {
            let found = self.nearest(addr, window);
            self.nearby.insert(addr, found);
        }
    }

    /// The remembered nearby-texture answer for `addr`; `None` when none was resolved.
    #[inline]
    pub(super) fn nearby(&self, addr: u32) -> Option<(i64, u32)> {
        self.nearby.get(&addr).copied().flatten()
    }

    pub(super) fn set_palette(&mut self, texture: u32, palette: u32) {
        self.palettes.insert(texture, palette);
    }

    pub(super) fn palette(&self, texture: u32) -> u32 {
        self.palettes.get(&texture).copied().unwrap_or(0)
    }
}
