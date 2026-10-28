//! FIOS2 path overlays: the registered records, the threads that have switched them off,
//! and the path-prefix matching resolution is built on.
//!
//! [`FiosOverlays`] owns the registry. Resolving a path through it needs the filesystem (an
//! overlay type can ask whether a candidate exists), so that walk stays with
//! `VitaState::fios_resolve`.

/// One registered FIOS2 path overlay: `src` stands in for `dst` under the policy
/// `kind` selects, at priority `order`. Mirrors the guest's `SceFiosOverlay`.
///
/// This is real, observable state - a title adds an overlay and then opens files
/// through it - so it is held as a decoded record rather than as the raw guest bytes.
#[derive(Clone)]
pub struct FiosOverlay {
    pub id: i32,
    /// `SceFiosOverlayType`: 0 opaque, 1 translucent, 2 newer, 3 writable.
    pub kind: u8,
    /// Resolution priority. Overlays apply lowest `order` first.
    pub order: u8,
    pub pid: i32,
    /// The path prefix being overlaid.
    pub dst: String,
    /// The path prefix that stands in for it.
    pub src: String,
}

/// `SceFiosOverlayType` values (vitasdk `psp2common/fios2.h`).
pub const SCE_FIOS_OVERLAY_TYPE_OPAQUE: u8 = 0;
pub const SCE_FIOS_OVERLAY_TYPE_TRANSLUCENT: u8 = 1;
pub const SCE_FIOS_OVERLAY_TYPE_NEWER: u8 = 2;
pub const SCE_FIOS_OVERLAY_TYPE_WRITABLE: u8 = 3;

/// Base of the SceUID range FIOS2 overlay ids come from, kept clear of thread and
/// sync-object ids for the same reason module ids are.
const FIOS_OVERLAY_ID_BASE: i32 = 0x0200_0000;

/// The registered overlays and the threads that have opted out of them. See the module docs.
#[derive(Default)]
pub(super) struct FiosOverlays {
    /// Registered FIOS2 path overlays, kept sorted by `order` (see
    /// [`crate::vita::fios2`]). Path resolution walks them in that order, which is
    /// what `order` is for, so sorting on insert keeps every resolve a plain scan.
    overlays: Vec<FiosOverlay>,
    /// Threads that have switched overlay resolution off for themselves
    /// (`_sceFiosKernelOverlayThreadSetDisabled`). Per-thread, because that is the
    /// point of the call: a loader thread turns overlays off to reach the real file
    /// underneath one it has just overlaid.
    disabled: std::collections::HashSet<i32>,
}

impl FiosOverlays {
    /// Register an overlay, returning its id. Insertion keeps the list sorted by
    /// `order` so resolution is a straight scan in application order.
    pub(super) fn add(&mut self, mut overlay: FiosOverlay) -> i32 {
        let id = FIOS_OVERLAY_ID_BASE + self.overlays.len() as i32;
        overlay.id = id;
        // SAY THAT A PATH REMAP EXISTS. Overlays are consulted ONLY by the FIOS2 resolve NIDs;
        // a raw `sceIoOpen` does not apply them. So a title that registers one and then opens a
        // remapped path through some OTHER system module gets a miss here that hardware would
        // resolve - and the miss is silent everywhere except in whatever that module reports.
        // One title's intro movie opens `data/Videos/intro.mp4` through SceMp4 and fails
        // (`0x80010002`) while all of its other assets load, which is exactly that shape.
        // Listing the overlays makes it checkable instead of theoretical.
        tracing::info!(
            target: "vitaslop::status",
            id,
            order = overlay.order,
            kind = overlay.kind,
            dst = %overlay.dst,
            src = %overlay.src,
            "fios overlay registered (a path REMAP; raw sceIoOpen does NOT apply these)"
        );
        let at = self
            .overlays
            .iter()
            .position(|o| o.order > overlay.order)
            .unwrap_or(self.overlays.len());
        self.overlays.insert(at, overlay);
        id
    }

    /// The overlay with this id, if it is still registered.
    pub(super) fn get(&self, id: i32) -> Option<&FiosOverlay> {
        self.overlays.iter().find(|o| o.id == id)
    }

    /// Replace a registered overlay's configuration, keeping its id. Returns whether
    /// the id resolved. The list is re-sorted because `order` may have changed.
    pub(super) fn modify(&mut self, id: i32, mut new_value: FiosOverlay) -> bool {
        let Some(i) = self.overlays.iter().position(|o| o.id == id) else {
            return false;
        };
        new_value.id = id;
        self.overlays[i] = new_value;
        self.overlays.sort_by_key(|o| o.order);
        true
    }

    /// Unregister an overlay. Returns whether the id resolved.
    pub(super) fn remove(&mut self, id: i32) -> bool {
        let before = self.overlays.len();
        self.overlays.retain(|o| o.id != id);
        self.overlays.len() != before
    }

    /// Every overlay whose `order` falls in `[min_order, max_order]`, in application order.
    pub(super) fn in_orders(
        &self,
        min_order: u8,
        max_order: u8,
    ) -> impl Iterator<Item = &FiosOverlay> {
        self.overlays
            .iter()
            .filter(move |o| o.order >= min_order && o.order <= max_order)
    }

    /// Whether thread `thid` has switched overlay resolution off for itself.
    pub(super) fn disabled_for(&self, thid: i32) -> bool {
        self.disabled.contains(&thid)
    }

    /// Switch overlay resolution on or off for thread `thid`. Returns the previous setting.
    pub(super) fn set_disabled_for(&mut self, thid: i32, disabled: bool) -> bool {
        if disabled {
            !self.disabled.insert(thid)
        } else {
            self.disabled.remove(&thid)
        }
    }
}

/// Strip `prefix` from `path` when it is a genuine PATH prefix, returning the rest.
///
/// "Genuine" means the prefix ends at a component boundary: `ux0:data` must match
/// `ux0:data/save` and `ux0:data` itself, but not `ux0:database`. A plain
/// `starts_with` would redirect the second, which is a silent misdirection of a file
/// the overlay was never meant to cover. Matching is case-insensitive, as the rest of
/// this filesystem's lookups are.
///
/// A device's root is spelled both ways - `app0:game_data` and `app0:/game_data` name the
/// same directory - and a title can register an overlay in one spelling and open through
/// it in the other (one does exactly that for its whole asset tree), so both sides are
/// compared with the slashes after the device colon dropped.
pub(super) fn strip_path_prefix(path: &str, prefix: &str) -> Option<String> {
    let path = without_root_slash(path);
    let prefix = without_root_slash(prefix);
    let prefix = prefix.trim_end_matches('/');
    if prefix.is_empty() {
        return Some(path.to_string());
    }
    let (lp, lx) = (path.to_ascii_lowercase(), prefix.to_ascii_lowercase());
    let rest = lp.strip_prefix(&lx)?;
    if rest.is_empty() || rest.starts_with('/') {
        Some(path[prefix.len()..].to_string())
    } else if prefix.ends_with(':') {
        // A bare device prefix: the rest is a path from its root.
        Some(format!("/{}", &path[prefix.len()..]))
    } else {
        None
    }
}

/// `dev:/a/b` -> `dev:a/b`; a path without a device, or without the slash, is unchanged.
fn without_root_slash(path: &str) -> std::borrow::Cow<'_, str> {
    match path.find(':') {
        Some(i) if path[i + 1..].starts_with('/') => {
            format!("{}{}", &path[..=i], path[i + 1..].trim_start_matches('/')).into()
        }
        _ => path.into(),
    }
}

#[cfg(test)]
mod strip_path_prefix_tests {
    use super::strip_path_prefix;

    #[test]
    fn a_prefix_matches_only_at_a_component_boundary() {
        assert_eq!(
            strip_path_prefix("ux0:data/save", "ux0:data").as_deref(),
            Some("/save")
        );
        assert_eq!(
            strip_path_prefix("ux0:data", "ux0:data/").as_deref(),
            Some("")
        );
        assert_eq!(strip_path_prefix("ux0:database", "ux0:data"), None);
        assert_eq!(
            strip_path_prefix("UX0:Data/x", "ux0:data").as_deref(),
            Some("/x")
        );
    }

    #[test]
    fn the_device_root_slash_is_spelled_either_way() {
        let (dst, path) = (
            "app0:game_data/cache/lumps",
            "app0:/game_data/cache/lumps/a.core",
        );
        assert_eq!(strip_path_prefix(path, dst).as_deref(), Some("/a.core"));
        assert_eq!(
            strip_path_prefix("app0:game_data/x", "app0:/game_data").as_deref(),
            Some("/x")
        );
        assert_eq!(
            strip_path_prefix("app0:/x/y", "app0:").as_deref(),
            Some("/x/y")
        );
        assert_eq!(strip_path_prefix("app0:x", "app0:/").as_deref(), Some("/x"));
        assert_eq!(strip_path_prefix("app1:/x", "app0:"), None);
    }
}
