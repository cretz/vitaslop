//! The guest's colour surfaces: the `SceGxmColorSurface` structs it initialised, and the state
//! the surface API keeps beside them - gamma mode, clip rectangle, dither mode.
//!
//! [`ColorSurfaces`] owns all of it, keyed by the guest struct address. It needs nothing from
//! the rest of the host; `VitaState` forwards to it from `state_gxm_programs.rs` and
//! `state_gxm_bind.rs`.

use crate::capture::ColorSurface;
use crate::fasthash::FxHashMap;
use std::collections::{HashMap, HashSet};

/// The colour surfaces and their per-surface API state. See the module docs.
#[derive(Default)]
pub(super) struct ColorSurfaces {
    /// Keyed by the guest STRUCT address; a re-init replaces. A `Vec` with a linear find
    /// grew by two entries a frame on a title that re-describes its small targets every
    /// frame, and every init then scanned all of them for the overlap report: 42 us per
    /// `sceGxmColorSurfaceSetGammaMode` and rising, MEASURED at frame 11,800.
    surfaces: FxHashMap<u32, ColorSurface>,
    /// Per-color-surface gamma-correction mode set by `sceGxmColorSurfaceSetGammaMode`, as
    /// `(SceGxmColorSurface*, data address, mode)`. Absent = SCE_GXM_COLOR_SURFACE_GAMMA_NONE.
    ///
    /// # WHY THE BUFFER ADDRESS IS RECORDED BESIDE THE STRUCT POINTER
    /// The mode is set on a `SceGxmColorSurface` and read back when a scene names its colour
    /// surface - and a title is free to pass a DIFFERENT pointer to the same surface. The
    /// struct is 32 plain bytes the guest owns: it can be copied into a frame's own scratch,
    /// held per swap-buffer, or rebuilt by `sceGxmColorSurfaceInit` before each scene. Every
    /// one of those keeps the same DATA address and changes the struct address, and a table
    /// keyed only on the pointer answers 0 for all of them - which silently reverts the
    /// surface to linear and darkens the whole frame.
    ///
    /// So the buffer the surface writes is recorded too, and [`ColorSurfaces::gamma_mode`]
    /// falls back to it. Copying a surface copies its data address, which is what makes that
    /// the identity worth keying on here; the struct pointer stays the FIRST key so an exact
    /// match still wins, and so a data address the guest has recycled under a fresh surface
    /// cannot outvote what that surface itself says.
    gamma: FxHashMap<u32, (u32, u32)>,
    /// The latest gamma mode set for a surface writing each DATA address - the fallback
    /// [`ColorSurfaces::gamma_mode`] takes when the struct address is unknown.
    gamma_by_data: FxHashMap<u32, u32>,
    /// Per colour-surface POINTER: the clip rectangle `sceGxmColorSurfaceSetClip` set
    /// `(xMin, yMin, xMax, yMax)` and the dither mode `...SetDitherMode` set - kept only so
    /// their getters read back what the title wrote (the renderer consumes neither).
    clip: HashMap<u32, (u32, u32, u32, u32)>,
    dither: HashMap<u32, u32>,
    /// Colour-surface pairs already reported as overlapping in guest memory, so the report
    /// fires once per pair rather than once per `sceGxmColorSurfaceInit`.
    reported_overlaps: HashSet<(u32, u32)>,
}

impl ColorSurfaces {
    /// Record `surface` at struct address `addr`, reporting any overlap with another first.
    pub(super) fn set(&mut self, addr: u32, surface: ColorSurface) {
        self.report_overlaps(&surface);
        self.surfaces.insert(addr, surface);
    }

    pub(super) fn get(&self, addr: u32) -> Option<ColorSurface> {
        self.surfaces.get(&addr).copied()
    }

    /// Report - once per colliding pair - two colour surfaces whose PIXEL BYTES overlap.
    ///
    /// Two render targets cannot both hold their contents in the same bytes, so an overlap is
    /// one of exactly two things and both matter: the guest is aliasing buffers it knows are
    /// never live together (legitimate, and it means a later pass sampling the earlier target
    /// is reading whatever overwrote it), or the data pointer we recorded is not the pixel
    /// pointer at all. The second is the dangerous one, because every downstream identity -
    /// which render a sampled texture resolves to, which target a pass draws into - is keyed
    /// off that address, and a wrong-but-consistent key looks exactly like a right one.
    ///
    /// MEASURED on a title whose six targets sit 16 KiB apart while each needs 2 MB.
    fn report_overlaps(&mut self, new: &ColorSurface) {
        // ONE byte per pixel: the smallest any `SceGxmColorFormat` can be. The real formats
        // here are wider, so this is a hard LOWER bound on the memory a surface occupies and
        // an overlap it finds is an overlap under every format. A diagnostic that guesses the
        // width would report pairs that do not really collide, and a report that cries wolf is
        // worse than none - this one is only allowed to be silent, never wrong.
        let bytes = |s: &ColorSurface| -> u64 {
            u64::from(s.stride_pixels.max(s.width)) * u64::from(s.height)
        };
        let (lo, hi) = (
            u64::from(new.data_addr),
            u64::from(new.data_addr) + bytes(new),
        );
        if new.data_addr == 0 || hi == lo {
            return;
        }
        for old in self.surfaces.values() {
            let (olo, ohi) = (
                u64::from(old.data_addr),
                u64::from(old.data_addr) + bytes(old),
            );
            if old.data_addr == 0 || old.data_addr == new.data_addr || ohi <= lo || hi <= olo {
                continue;
            }
            let key = (
                old.data_addr.min(new.data_addr),
                old.data_addr.max(new.data_addr),
            );
            if self.reported_overlaps.insert(key) {
                tracing::warn!(
                    target: "vitaslop::gxm",
                    "gxm surface: colour surfaces {:#x} ({}x{} stride {}, >= {} bytes) and {:#x} \
                     ({}x{} stride {}, >= {} bytes) OVERLAP in guest memory even at ONE byte per \
                     pixel - they cannot both hold their pixels, so either the guest aliases them \
                     or one of these data pointers is not the pixel pointer",
                    old.data_addr, old.width, old.height, old.stride_pixels, bytes(old),
                    new.data_addr, new.width, new.height, new.stride_pixels, bytes(new),
                );
            }
        }
    }

    /// Record the gamma mode set on the surface struct at `surface_addr`, writing `data_addr`.
    pub(super) fn set_gamma(&mut self, surface_addr: u32, data_addr: u32, gamma: u32) {
        self.gamma.insert(surface_addr, (data_addr, gamma));
        if data_addr != 0 {
            self.gamma_by_data.insert(data_addr, gamma);
        }
    }

    /// The gamma mode recorded for the surface struct at `addr`, else - when that pointer is
    /// unknown - for a surface writing `data_addr`; 0 when neither is known.
    pub(super) fn gamma_mode(&self, addr: u32, data_addr: u32) -> u32 {
        if let Some((_, g)) = self.gamma.get(&addr) {
            return *g;
        }
        // A zero data address is "the surface names no buffer", which is not an identity and
        // must not match the entries of one that does.
        if data_addr == 0 {
            return 0;
        }
        self.gamma_by_data.get(&data_addr).copied().unwrap_or(0)
    }

    pub(super) fn set_clip(&mut self, surface: u32, rect: (u32, u32, u32, u32)) {
        self.clip.insert(surface, rect);
    }

    pub(super) fn clip(&self, surface: u32) -> Option<(u32, u32, u32, u32)> {
        self.clip.get(&surface).copied()
    }

    pub(super) fn set_dither(&mut self, surface: u32, mode: u32) {
        self.dither.insert(surface, mode);
    }

    pub(super) fn dither(&self, surface: u32) -> u32 {
        self.dither.get(&surface).copied().unwrap_or(0)
    }
}
