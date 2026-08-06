//! SceFios2Kernel: the kernel-side path OVERLAY layer under FIOS2.
//!
//! FIOS2 is Sony's asynchronous file IO library, and titles ship its user-mode half
//! (`sce_module/libfios2.suprx`) inside their own package. That module runs as guest
//! code here like any other, and where it needs the kernel it calls into this
//! surface - so these are not "library" calls we could route elsewhere, they are the
//! floor the shipped library stands on.
//!
//! What the layer does is remap paths. A title registers overlays, each saying "`src`
//! stands in for `dst`", with a policy (opaque / translucent / newer / writable) and
//! an `order` that decides which applies first. Everything else here - resolve,
//! get-info, get-list, the directory-handle family - is built on that registry, which
//! lives in [`VitaState`] because it is state the guest can observe.
//!
//! Two struct layouts this file needs are NOT published in vita-headers or on the
//! wiki: `SceFiosNativeStat` and `SceFiosNativeDirEntry` (vitasdk declares both as
//! "missing structs"). They are implemented here as the platform's own `SceIoStat`
//! and `SceIoDirent`, which is what "native" denotes throughout FIOS2 - the layer
//! below FIOS2 is SceIo, and these calls are documented as passing its results
//! straight through. If that reading is wrong the failure is loud rather than subtle
//! (the caller reads a size or a name out of the wrong offset), which is why it is
//! worth taking rather than refusing the calls.

use crate::host::{FiosOverlay, GuestCtx};
use crate::hostcall;

use super::iofilemgr::read_cstr;

/// `SCE_FIOS2_OVERLAY_PATH_SIZE` - the fixed path buffer inside `SceFiosOverlay`.
const OVERLAY_PATH_SIZE: u32 = 292;

/// Byte size of a guest `SceFiosOverlay` (vitasdk asserts 0x258), and the offsets
/// within it: type(u8), order(u8), dst_len(u16), src_len(u16), unk2(u16), pid(s32),
/// id(s32), dst[292], src[292].
const OVERLAY_SIZE: u32 = 0x258;
const OVERLAY_PID: u32 = 8;
const OVERLAY_ID: u32 = 12;
const OVERLAY_DST: u32 = 16;
const OVERLAY_SRC: u32 = OVERLAY_DST + OVERLAY_PATH_SIZE;

/// The whole overlay order range, for a resolve that does not name one.
const ORDER_MIN: u8 = 0;
const ORDER_MAX: u8 = u8::MAX;

/// `SCE_ERROR_ERRNO_EINVAL` / `ENOENT` / `EBADF`, the errnos this surface returns.
const EINVAL: i32 = 0x8001_0016u32 as i32;
const ENOENT: i32 = 0x8001_0002u32 as i32;
const EBADF: i32 = 0x8001_0009u32 as i32;

/// Read a guest `SceFiosOverlay` into the decoded record the registry holds.
///
/// The two path fields are fixed 292-byte buffers with an explicit length beside
/// them; the length is trusted when it is in range and the NUL used otherwise, which
/// covers a caller that fills the buffer but leaves the length at zero.
fn read_overlay(ctx: &GuestCtx, addr: u32) -> FiosOverlay {
    let path_at = |off: u32, len: u16| -> String {
        let bytes = ctx.read_bytes(addr + off, OVERLAY_PATH_SIZE as usize);
        let n = if len > 0 && (len as u32) < OVERLAY_PATH_SIZE {
            len as usize
        } else {
            bytes.iter().position(|&b| b == 0).unwrap_or(bytes.len())
        };
        String::from_utf8_lossy(&bytes[..n]).into_owned()
    };
    let header = ctx.read_u32(addr);
    let lens = ctx.read_u32(addr + 4);
    FiosOverlay {
        id: ctx.read_u32(addr + OVERLAY_ID) as i32,
        kind: header as u8,
        order: (header >> 8) as u8,
        pid: ctx.read_u32(addr + OVERLAY_PID) as i32,
        dst: path_at(OVERLAY_DST, (header >> 16) as u16),
        src: path_at(OVERLAY_SRC, lens as u16),
    }
}

/// Write a registry record back out as a guest `SceFiosOverlay`. The whole struct is
/// written (zeroed then filled) so the caller never reads its own stale buffer back
/// as overlay configuration.
fn write_overlay(ctx: &mut GuestCtx, addr: u32, o: &FiosOverlay) {
    let mut buf = vec![0u8; OVERLAY_SIZE as usize];
    let (dst, src) = (o.dst.as_bytes(), o.src.as_bytes());
    let dst_n = dst.len().min(OVERLAY_PATH_SIZE as usize - 1);
    let src_n = src.len().min(OVERLAY_PATH_SIZE as usize - 1);
    buf[0] = o.kind;
    buf[1] = o.order;
    buf[2..4].copy_from_slice(&(dst_n as u16).to_le_bytes());
    buf[4..6].copy_from_slice(&(src_n as u16).to_le_bytes());
    buf[OVERLAY_PID as usize..OVERLAY_PID as usize + 4].copy_from_slice(&o.pid.to_le_bytes());
    buf[OVERLAY_ID as usize..OVERLAY_ID as usize + 4].copy_from_slice(&o.id.to_le_bytes());
    let d = OVERLAY_DST as usize;
    buf[d..d + dst_n].copy_from_slice(&dst[..dst_n]);
    let s = OVERLAY_SRC as usize;
    buf[s..s + src_n].copy_from_slice(&src[..src_n]);
    ctx.write_bytes(addr, &buf);
}

/// int _sceFiosKernelOverlayAdd(const SceFiosKernelOverlay *overlay,
///                              SceFiosKernelOverlayID *out_id)
#[hostcall]
pub(super) fn overlay_add(ctx: &mut GuestCtx, st: &mut VitaState, overlay: Ptr, out_id: Ptr) -> i32 {
    if overlay.is_null() {
        EINVAL
    } else {
        let rec = read_overlay(ctx, overlay.addr());
        tracing::debug!(
            target: "vitaslop::io",
            kind = rec.kind, order = rec.order, dst = %rec.dst, src = %rec.src,
            "fios2 overlay add"
        );
        let id = st.fios_overlay_add(rec);
        if !out_id.is_null() {
            ctx.write_u32(out_id.addr(), id as u32);
        }
        0
    }
}

/// int _sceFiosKernelOverlayAddForProcess(SceUID target_process,
///     const SceFiosKernelOverlay *overlay, SceFiosKernelOverlayID *out_id)
///
/// The cross-process form. One process exists here, so the only target that can be
/// named is this one; naming another is an error rather than a silent add to ours,
/// because an overlay applied to the wrong process is not an overlay at all.
#[hostcall]
pub(super) fn overlay_add_for_process(
    ctx: &mut GuestCtx,
    st: &mut VitaState,
    target: i32,
    overlay: Ptr,
    out_id: Ptr,
) -> i32 {
    if !is_own_process(target) {
        EINVAL
    } else if overlay.is_null() {
        EINVAL
    } else {
        let rec = read_overlay(ctx, overlay.addr());
        let id = st.fios_overlay_add(rec);
        if !out_id.is_null() {
            ctx.write_u32(out_id.addr(), id as u32);
        }
        0
    }
}

/// The single process this kernel runs (matching `sceKernelGetProcessId`). `0` and
/// `-1` are the usual "the calling process" spellings.
const PROCESS_ID: i32 = 0x1000;
fn is_own_process(pid: i32) -> bool {
    pid == 0 || pid == -1 || pid == PROCESS_ID
}

/// int _sceFiosKernelOverlayGetInfo(SceFiosKernelOverlayID id,
///                                  SceFiosKernelOverlay *out_overlay)
#[hostcall]
pub(super) fn overlay_get_info(ctx: &mut GuestCtx, st: &mut VitaState, id: i32, out: Ptr) -> i32 {
    match st.fios_overlay(id) {
        Some(o) if !out.is_null() => {
            let o = o.clone();
            write_overlay(ctx, out.addr(), &o);
            0
        }
        Some(_) => EINVAL,
        None => ENOENT,
    }
}

/// int _sceFiosKernelOverlayGetInfoForProcess(SceUID target_process,
///     SceFiosKernelOverlayID id, SceFiosKernelOverlay *out_overlay)
#[hostcall]
pub(super) fn overlay_get_info_for_process(
    ctx: &mut GuestCtx,
    st: &mut VitaState,
    target: i32,
    id: i32,
    out: Ptr,
) -> i32 {
    if !is_own_process(target) {
        EINVAL
    } else {
        match st.fios_overlay(id) {
            Some(o) if !out.is_null() => {
                let o = o.clone();
                write_overlay(ctx, out.addr(), &o);
                0
            }
            Some(_) => EINVAL,
            None => ENOENT,
        }
    }
}

/// int _sceFiosKernelOverlayModify(SceFiosKernelOverlayID id,
///                                 const SceFiosKernelOverlay *new_value)
#[hostcall]
pub(super) fn overlay_modify(ctx: &mut GuestCtx, st: &mut VitaState, id: i32, new_value: Ptr) -> i32 {
    if new_value.is_null() {
        EINVAL
    } else {
        let rec = read_overlay(ctx, new_value.addr());
        if st.fios_overlay_modify(id, rec) {
            0
        } else {
            ENOENT
        }
    }
}

/// int _sceFiosKernelOverlayModifyForProcess(SceUID target_process,
///     SceFiosKernelOverlayID id, const SceFiosKernelOverlay *new_value)
#[hostcall]
pub(super) fn overlay_modify_for_process(
    ctx: &mut GuestCtx,
    st: &mut VitaState,
    target: i32,
    id: i32,
    new_value: Ptr,
) -> i32 {
    if !is_own_process(target) || new_value.is_null() {
        EINVAL
    } else if st.fios_overlay_modify(id, read_overlay(ctx, new_value.addr())) {
        0
    } else {
        ENOENT
    }
}

/// int _sceFiosKernelOverlayRemove(SceFiosKernelOverlayID id)
#[hostcall]
pub(super) fn overlay_remove(st: &mut VitaState, id: i32) -> i32 {
    if st.fios_overlay_remove(id) {
        0
    } else {
        ENOENT
    }
}

/// int _sceFiosKernelOverlayRemoveForProcess(SceUID target_process,
///                                           SceFiosKernelOverlayID id)
#[hostcall]
pub(super) fn overlay_remove_for_process(st: &mut VitaState, target: i32, id: i32) -> i32 {
    if !is_own_process(target) {
        EINVAL
    } else if st.fios_overlay_remove(id) {
        0
    } else {
        ENOENT
    }
}

/// Offsets within `SceFiosGetListSyscallArgs` (vitasdk asserts 0x18): the out-id
/// array pointer, then five words whose meanings are not published. The one that
/// matters is `data_0x0C`, a `SceSize` - the only size-typed field, so it is the
/// capacity of the caller's array.
const GET_LIST_OUT_IDS: u32 = 0;
const GET_LIST_CAPACITY: u32 = 0x0C;

/// int _sceFiosKernelOverlayGetList(SceUID pid, SceUInt8 min_order,
///     SceUInt8 max_order, SceFiosGetListSyscallArgs *args)
///
/// Fill the caller's array with the ids of the overlays in the order range, and
/// return how many there are. Bounded by the caller's capacity, which is what stops
/// an overlay-heavy title overrunning its own buffer.
#[hostcall]
pub(super) fn overlay_get_list(
    ctx: &mut GuestCtx,
    st: &mut VitaState,
    pid: i32,
    min_order: u32,
    max_order: u32,
    args: Ptr,
) -> i32 {
    if !is_own_process(pid) || args.is_null() {
        EINVAL
    } else {
        let ids = st.fios_overlay_ids(min_order as u8, max_order as u8);
        let out = ctx.read_u32(args.addr() + GET_LIST_OUT_IDS);
        let capacity = ctx.read_u32(args.addr() + GET_LIST_CAPACITY) as usize;
        if out != 0 {
            for (i, id) in ids.iter().take(capacity).enumerate() {
                ctx.write_u32(out + i as u32 * 4, *id as u32);
            }
        }
        ids.len() as i32
    }
}

/// Offsets within `SceFiosResolveSyncSyscallArgs` / the with-range variant (both
/// 0x18): the output path buffer, then its length. The range form additionally
/// carries the two order bounds as bytes at 0x08 and 0x09.
const RESOLVE_OUT_PATH: u32 = 0;
const RESOLVE_OUT_LEN: u32 = 4;
const RESOLVE_RANGE_ORDERS: u32 = 8;

/// Write a resolved path into the caller's buffer, NUL-terminated and bounded.
/// Returns the errno, or 0.
fn write_resolved(ctx: &mut GuestCtx, args: u32, resolved: &str) -> i32 {
    let out = ctx.read_u32(args + RESOLVE_OUT_PATH);
    let max = ctx.read_u32(args + RESOLVE_OUT_LEN) as usize;
    // A caller that passes no length still passes a buffer; the overlay path size is
    // the interface's own bound and the only defensible cap in that case.
    let max = if max == 0 { OVERLAY_PATH_SIZE as usize } else { max };
    if out == 0 {
        EINVAL
    } else {
        let bytes = resolved.as_bytes();
        let n = bytes.len().min(max - 1);
        let mut buf = bytes[..n].to_vec();
        buf.push(0);
        ctx.write_bytes(out, &buf);
        0
    }
}

/// int _sceFiosKernelOverlayResolveSync(SceUID pid, int resolve_flag,
///     const char *in_path, SceFiosResolveSyncSyscallArgs *args)
///
/// Run a path through every registered overlay and hand back what the filesystem
/// should actually see. This is the call the whole layer exists for: FIOS2's user
/// half resolves here and then opens the result with ordinary `sceIo*`.
///
/// `resolve_flag`'s bit meanings are not published. Resolution here does not consult
/// it, and that is a deliberate limit rather than an oversight: the flag can only
/// select among behaviours (a read-vs-write resolve is the obvious candidate), and
/// every overlay type resolves reads and writes to the same place except `WRITABLE`,
/// whose difference shows up at open time and not here.
#[hostcall]
pub(super) fn overlay_resolve_sync(
    ctx: &mut GuestCtx,
    st: &mut VitaState,
    pid: i32,
    _resolve_flag: i32,
    in_path: Ptr,
    args: Ptr,
) -> i32 {
    if !is_own_process(pid) || in_path.is_null() || args.is_null() {
        EINVAL
    } else {
        let path = read_cstr(ctx, in_path.addr());
        let resolved = st.fios_resolve(&path, ORDER_MIN, ORDER_MAX);
        tracing::trace!(target: "vitaslop::io", path, resolved, "fios2 resolve");
        write_resolved(ctx, args.addr(), &resolved)
    }
}

/// int _sceFiosKernelOverlayResolveWithRangeSync(SceUID pid, int resolve_flag,
///     const char *in_path, SceFiosResolveWithRangeSyncSyscallArgs *args)
///
/// The same resolve restricted to an `order` window, which is how a caller asks
/// "what would this path be with only the overlays below mine applied" - the question
/// an overlay's own implementation has to ask to reach the layer beneath it. The two
/// bounds are the byte fields at 0x08 and 0x09 of the args block.
#[hostcall]
pub(super) fn overlay_resolve_with_range_sync(
    ctx: &mut GuestCtx,
    st: &mut VitaState,
    pid: i32,
    _resolve_flag: i32,
    in_path: Ptr,
    args: Ptr,
) -> i32 {
    if !is_own_process(pid) || in_path.is_null() || args.is_null() {
        EINVAL
    } else {
        let orders = ctx.read_u32(args.addr() + RESOLVE_RANGE_ORDERS);
        let (min_order, max_order) = (orders as u8, (orders >> 8) as u8);
        let path = read_cstr(ctx, in_path.addr());
        let resolved = st.fios_resolve(&path, min_order, max_order);
        write_resolved(ctx, args.addr(), &resolved)
    }
}

/// int _sceFiosKernelOverlayThreadIsDisabled(void)
/// int _sceFiosKernelOverlayThreadSetDisabled(SceInt32 disabled)
///
/// Overlay resolution can be switched off PER THREAD, which is how a loader reaches
/// the real file underneath a path it has itself overlaid. Per-thread is the whole
/// point - a process-wide flag would disable it for every other thread mid-load - so
/// the flag is keyed by the calling thread.
#[hostcall]
pub(super) fn overlay_thread_is_disabled(st: &mut VitaState) -> i32 {
    i32::from(st.fios_overlay_disabled())
}

/// Returns the PREVIOUS setting, which is what a caller saves to restore.
#[hostcall]
pub(super) fn overlay_thread_set_disabled(st: &mut VitaState, disabled: i32) -> i32 {
    i32::from(st.fios_overlay_set_disabled(disabled != 0))
}

/// int _sceFiosKernelOverlayGetRecommendedScheduler(int avail,
///     const char *partially_resolved_path, SceUInt64 *a3)
///
/// Ask which IO scheduler suits a path - the answer on hardware depends on which
/// physical device the path lands on (game card, internal storage, memory card),
/// because their seek behaviour differs enough to want different queueing.
///
/// Every path here is served from one in-memory store with a single modelled
/// bandwidth (see `vita::iofilemgr`), so there is exactly one device and therefore
/// exactly one scheduler to recommend. The out-parameter is written rather than left
/// alone so the caller does not read its own stack as a recommendation.
#[hostcall]
pub(super) fn overlay_get_recommended_scheduler(
    ctx: &mut GuestCtx,
    _st: &mut VitaState,
    _avail: i32,
    _path: Ptr,
    out: Ptr,
) -> i32 {
    if !out.is_null() {
        ctx.write_u32(out.addr(), 0);
        ctx.write_u32(out.addr() + 4, 0);
    }
    0
}

// --- the directory-handle family ------------------------------------------------
//
// A `SceFiosKernelOverlayDH` enumerates a directory through the overlaid namespace.
// It is backed by the ordinary directory descriptor `sceIoDopen` returns, opened on
// the RESOLVED path - which is what makes it an overlay-aware handle rather than a
// second directory implementation.

/// int _sceFiosKernelOverlayDHOpenSync(SceFiosKernelOverlayDH *out_dh,
///     const char *path, SceUInt8 from_order, SceFiosDHOpenSyncSyscallArgs *args)
///
/// `from_order` and the args block's `to_order` bound which overlays apply, exactly
/// as in the with-range resolve.
#[hostcall]
pub(super) fn dh_open_sync(
    ctx: &mut GuestCtx,
    st: &mut VitaState,
    out_dh: Ptr,
    path: Ptr,
    from_order: u32,
    args: Ptr,
) -> i32 {
    if path.is_null() {
        EINVAL
    } else {
        let to_order = if args.is_null() { ORDER_MAX } else { ctx.read_u32(args.addr()) as u8 };
        let raw = read_cstr(ctx, path.addr());
        let resolved = st.fios_resolve(&raw, from_order as u8, to_order);
        let dh = st.io_dopen(&resolved);
        if dh < 0 {
            dh
        } else {
            if !out_dh.is_null() {
                ctx.write_u32(out_dh.addr(), dh as u32);
            }
            0
        }
    }
}

/// int _sceFiosKernelOverlayDHReadSync(SceFiosKernelOverlayDH dh,
///                                     SceFiosNativeDirEntry *out_entry)
///
/// Read the next entry. Returns 1 for an entry, 0 at the end of the listing, or an
/// errno. `SceFiosNativeDirEntry` is written as the platform's `SceIoDirent` - see
/// this module's header for why.
#[hostcall]
pub(super) fn dh_read_sync(ctx: &mut GuestCtx, st: &mut VitaState, dh: i32, out_entry: Ptr) -> i32 {
    super::iofilemgr::write_dirent(ctx, st, dh, out_entry.addr())
}

/// int _sceFiosKernelOverlayDHStatSync(SceFiosKernelOverlayDH dh,
///                                     SceFiosNativeStat *out_stat)
///
/// Stat the directory the handle is open on. `SceFiosNativeStat` is written as the
/// platform's `SceIoStat` - see this module's header.
#[hostcall]
pub(super) fn dh_stat_sync(ctx: &mut GuestCtx, st: &mut VitaState, dh: i32, out_stat: Ptr) -> i32 {
    if !st.io_dir_is_open(dh) {
        EBADF
    } else if out_stat.is_null() {
        EINVAL
    } else {
        super::iofilemgr::write_dir_stat(ctx, out_stat.addr());
        0
    }
}

/// int _sceFiosKernelOverlayDHChstatSync(SceFiosKernelOverlayDH dh,
///     const SceFiosNativeStat *new_stat, unsigned int cbit)
///
/// Change the status of the directory a handle is open on. The handle names a
/// directory descriptor, and this filesystem records status overrides by PATH, so
/// the descriptor's path is what the change is applied to.
#[hostcall]
pub(super) fn dh_chstat_sync(ctx: &mut GuestCtx, st: &mut VitaState, dh: i32, new_stat: Ptr, cbit: u32) -> i32 {
    match st.io_dir_path(dh) {
        None => EBADF,
        Some(_) if new_stat.is_null() => EINVAL,
        Some(path) => {
            let over = super::iofilemgr::read_stat_override(ctx, new_stat.addr(), cbit as i32);
            st.io_chstat(&path, over)
        }
    }
}

/// int _sceFiosKernelOverlayDHSyncSync(SceFiosKernelOverlayDH dh, int flag)
///
/// Flush the directory's pending changes. Nothing is buffered behind this filesystem
/// (see `iofilemgr::io_sync`), so the only work is validating the handle - which
/// still matters, because syncing a closed handle is an error the caller wants back.
#[hostcall]
pub(super) fn dh_sync_sync(st: &mut VitaState, dh: i32, _flag: i32) -> i32 {
    if st.io_dir_is_open(dh) {
        0
    } else {
        EBADF
    }
}

/// int _sceFiosKernelOverlayDHCloseSync(SceFiosKernelOverlayDH dh)
#[hostcall]
pub(super) fn dh_close_sync(st: &mut VitaState, dh: i32) -> i32 {
    st.io_dclose(dh)
}
