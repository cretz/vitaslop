//! SceIoFilemgr: file IO over a host virtual filesystem.
//!
//! This is the path a real title's asset reads take, and - via `sceIoWrite` to
//! fd 1 - the sink newlib's stdout resolves to. The backing store is a host-only
//! path->bytes map ([`VitaState`] `fs`): read files are preloaded by the harness,
//! write opens create/truncate entries, and fd 1/2 are the captured console
//! streams. No real filesystem is touched, so runs stay deterministic.
//!
//! Most handlers use `#[hostcall]` for the AAPCS marshalling; `sceIoLseek` is
//! hand-written because its 64-bit `SceOff` offset and return do not fit the
//! macro's 32-bit value model (the offset lands in the r2:r3 even-aligned pair,
//! the return in r0:r1).

use crate::host::{GuestCtx, VitaState};
use crate::hostcall;
use crate::SvcOutcome;

/// Bound on a path string read from guest memory.
const MAX_PATH: usize = 1024;

/// Modelled sequential read bandwidth, in KiB per second
/// (`VITASLOP_IO_BANDWIDTH_KIBPS`; `0` disables the model entirely).
///
/// Our reads are served from a host `path -> bytes` map, so without this they complete in
/// zero guest time - and that is not a harmless speed-up, it changes ORDER. A title that
/// posts an asynchronous load job and then touches the thing being loaded expects the job
/// to still be in flight; with instant reads the job finishes in the frame it was posted,
/// its resource is released, and the consumer reads a dangling handle. That is the exact
/// shape of the PCSE00001 race-load crash: 4 MB of car + course data that takes about a
/// second off a real game card completed inside two frames here.
///
/// 10 MiB/s is a defensible PS Vita game-card figure. Deterministic: the delay is a pure
/// function of the byte count, so a run is as reproducible as it was with instant reads.
///
/// A first attempt at this parked the reader on the GAME clock (`sleep_park`) and
/// livelocked: `virtual_us` advances a frame per display FLIP and otherwise only on the
/// scheduler's nothing-is-runnable idle path, so a title waiting for its load produced no
/// flips, a sibling polling with a zero-length delay kept the idle path finding a deadline
/// that bought no time, and the clock stood still at ANY bandwidth including 1 GiB/s. The
/// park is now charged against a separate storage clock (`VitaState::io_park`) that
/// advances on flips, on scheduler quanta and on the idle path, so it cannot depend on the
/// thing it is blocking.
const DEFAULT_BANDWIDTH_KIBPS: u64 = 10 * 1024;

/// Fixed per-request cost in microseconds (`VITASLOP_IO_REQUEST_US`): the command
/// round-trip a read pays whatever its size, so a stream of small reads is not free either.
const DEFAULT_REQUEST_US: u64 = 200;

fn env_u64(name: &str, default: u64) -> u64 {
    std::env::var(name).ok().and_then(|s| s.trim().parse().ok()).unwrap_or(default)
}

/// The modelled time a read of `bytes` bytes takes, in microseconds, or `None` when the
/// model is switched off (bandwidth 0) or the transfer is empty.
fn read_latency_us(bytes: usize) -> Option<u64> {
    use std::sync::OnceLock;
    static CFG: OnceLock<(u64, u64)> = OnceLock::new();
    let &(kibps, request_us) = CFG.get_or_init(|| {
        (
            env_u64("VITASLOP_IO_BANDWIDTH_KIBPS", DEFAULT_BANDWIDTH_KIBPS),
            env_u64("VITASLOP_IO_REQUEST_US", DEFAULT_REQUEST_US),
        )
    });
    if kibps == 0 || bytes == 0 {
        return None;
    }
    // bytes / (kibps * 1024) seconds, in microseconds, rounded up so a sub-microsecond
    // read still costs something.
    let transfer_us = (bytes as u64 * 1_000_000).div_ceil(kibps * 1024);
    Some(request_us + transfer_us)
}

/// Smallest debt worth a context switch, in microseconds
/// (`VITASLOP_IO_PARK_THRESHOLD_US`). See [`charge_read`].
const DEFAULT_PARK_THRESHOLD_US: u64 = 2_000;

/// Charge the modelled cost of having transferred `bytes` to the calling thread, parking it
/// once the accrued debt is worth a context switch.
///
/// Parking on EVERY read is what a first attempt did, and it was unusable: a title's boot
/// issues an enormous number of small reads (a text parser reading a few bytes at a time),
/// and each park costs a full block/idle/clock-advance/resume round trip through the
/// scheduler. Booting stopped making progress at ANY bandwidth, including 1 GiB/s - which is
/// the measurement that shows the cost is the park COUNT, not the delay size.
///
/// Accumulating instead keeps the aggregate rate exactly as modelled (no time is discarded -
/// the debt is only deferred, then paid in full) while making the number of parks
/// proportional to bytes moved rather than to reads issued.
fn charge_read(st: &mut VitaState, bytes: usize) -> SvcOutcome {
    let Some(us) = read_latency_us(bytes) else { return SvcOutcome::Continue };
    if !st.is_preemptive() {
        // Cooperative hosts cannot park a thread at all; keep instant reads there.
        return SvcOutcome::Continue;
    }
    let debt = st.add_io_debt_us(us);
    use std::sync::OnceLock;
    static THRESHOLD: OnceLock<u64> = OnceLock::new();
    let threshold =
        *THRESHOLD.get_or_init(|| env_u64("VITASLOP_IO_PARK_THRESHOLD_US", DEFAULT_PARK_THRESHOLD_US));
    if debt < threshold {
        return SvcOutcome::Continue;
    }
    st.take_io_debt_us();
    tracing::debug!(target: "vitaslop::io", thid = st.current_thread(), debt_us = debt, "storage park");
    st.io_park(debt);
    SvcOutcome::Block
}

/// Offset of `st_size` (SceOff, 64-bit) within SceIoStat: after `st_mode` (u32)
/// and `st_attr` (u32).
const STAT_SIZE_OFFSET: u32 = 8;

/// SceIoStat `st_mode` for a readable regular file: SCE_S_IFREG plus user/system
/// read permission (octal 020000 | 0400 | 04). A title that checks `SCE_S_ISREG`
/// before trusting the size needs the format bits set, not a bare zero.
const STAT_MODE_REGULAR_READABLE: u32 = 0o20000 | 0o400 | 0o4;
/// SceIoStat `st_attr` for a regular file: SCE_SO_IFREG.
const STAT_ATTR_REGULAR: u32 = 0x0020;

/// Fill the size-relevant fields of a guest SceIoStat: mark it a readable regular
/// file and write the 64-bit size. The remaining fields (timestamps, private) are
/// left untouched - a size query is all a title needs from these handlers, and the
/// guest zeroes its stat buffer before the call. Shared by the by-path and by-fd
/// stat handlers so both report a file identically.
fn write_file_stat(ctx: &mut GuestCtx, stat_addr: u32, size: u64) {
    if stat_addr == 0 {
        return;
    }
    ctx.write_u32(stat_addr, STAT_MODE_REGULAR_READABLE); // st_mode
    ctx.write_u32(stat_addr + 4, STAT_ATTR_REGULAR); // st_attr
    ctx.write_u32(stat_addr + STAT_SIZE_OFFSET, size as u32);
    ctx.write_u32(stat_addr + STAT_SIZE_OFFSET + 4, (size >> 32) as u32);
}

/// SceUID sceIoOpen(const char *file, int flags, SceMode mode)
#[hostcall]
pub(super) fn io_open(ctx: &mut GuestCtx, st: &mut VitaState, file: Ptr, flags: u32, _mode: u32) -> i32 {
    let path = read_cstr(ctx, file.addr());
    tracing::trace!(target: "vitaslop::io", path, flags = format_args!("{flags:#x}"), lr = format_args!("{:#x}", ctx.regs[14]), "open_from");
    st.io_open(&path, flags)
}

/// int sceIoRead(SceUID fd, void *buf, SceSize size)
/// Returns the number of bytes read, or a negative errno on a bad descriptor.
///
/// Marshalled by hand rather than through `#[hostcall]` because a read PARKS the calling
/// thread for the transfer's modelled duration (see [`read_latency_us`]), so it has to be
/// able to return [`SvcOutcome::Block`].
pub(super) fn io_read(ctx: &mut GuestCtx, st: &mut VitaState) -> SvcOutcome {
    let fd = ctx.arg(0) as i32;
    let buf = ctx.arg(1);
    let size = ctx.arg(2);
    let (ret, got) = match st.io_read(fd, size as usize) {
        Some(data) => {
            ctx.write_bytes(buf, &data);
            (data.len() as i32, data.len())
        }
        None => (EBADF, 0),
    };
    ctx.ret(ret as u32);
    charge_read(st, got)
}

/// int sceIoWrite(SceUID fd, const void *buf, SceSize size)
/// fd 1/2 go to the captured console; other descriptors to the vfs.
#[hostcall]
pub(super) fn io_write(ctx: &mut GuestCtx, st: &mut VitaState, fd: i32, buf: Ptr, size: u32) -> i32 {
    let bytes = ctx.read_bytes(buf.addr(), size as usize);
    match st.io_write(fd, &bytes) {
        Some(n) => n as i32,
        None => EBADF,
    }
}

/// int sceIoLseek32(SceUID fd, int offset, int whence)
/// The 32-bit seek: everything fits in core registers and the return.
#[hostcall]
pub(super) fn io_lseek32(st: &mut VitaState, fd: i32, offset: i32, whence: i32) -> i32 {
    st.io_lseek(fd, offset as i64, whence) as i32
}

/// SceOff sceIoLseek(SceUID fd, SceOff offset, int whence)
///
/// `SceOff` is 64-bit: with `fd` in r0, the offset takes the even-aligned r2:r3
/// pair (r1 is padding) and `whence` spills to the stack; the return goes in
/// r0:r1. The macro cannot express this, so it is marshalled by hand.
pub(super) fn io_lseek(ctx: &mut GuestCtx, st: &mut VitaState) {
    let fd = ctx.arg(0) as i32;
    let offset = (ctx.arg(2) as u64 | (ctx.arg(3) as u64) << 32) as i64;
    let whence = ctx.arg(4) as i32;
    let pos = st.io_lseek(fd, offset, whence);
    ctx.regs[0] = pos as u32;
    ctx.regs[1] = (pos >> 32) as u32;
}

/// int sceIoPread(SceUID fd, void *data, SceSize size, SceOff offset)
///
/// A positioned read that does not move the descriptor's cursor. `SceOff` is
/// 64-bit: with `fd`, `data`, `size` filling r0-r2, r3 is the alignment pad and the
/// 64-bit offset spills to the even-aligned first stack slot (`arg(4):arg(5)`), like
/// `sceIoLseek`. This is the path an AT9 music streamer uses to read the
/// file header and successive chunks from a shared descriptor.
pub(super) fn io_pread(ctx: &mut GuestCtx, st: &mut VitaState) -> SvcOutcome {
    let fd = ctx.arg(0) as i32;
    let buf = ctx.arg(1);
    let size = ctx.arg(2);
    let offset = ctx.arg(4) as u64 | (ctx.arg(5) as u64) << 32;
    let (ret, got) = match st.io_pread(fd, offset, size as usize) {
        Some(data) => {
            ctx.write_bytes(buf, &data);
            (data.len() as i32, data.len())
        }
        None => (EBADF, 0),
    };
    tracing::trace!(target: "vitaslop::io", fd, offset, size, ret, "pread");
    ctx.ret(ret as u32);
    charge_read(st, got)
}

/// int sceIoPwrite(SceUID fd, const void *data, SceSize size, SceOff offset)
/// The positioned-write counterpart of [`io_pread`]; same register layout.
pub(super) fn io_pwrite(ctx: &mut GuestCtx, st: &mut VitaState) {
    let fd = ctx.arg(0) as i32;
    let buf = ctx.arg(1);
    let size = ctx.arg(2);
    let offset = ctx.arg(4) as u64 | (ctx.arg(5) as u64) << 32;
    let bytes = ctx.read_bytes(buf, size as usize);
    let ret = match st.io_pwrite(fd, offset, &bytes) {
        Some(n) => n as i32,
        None => EBADF,
    };
    ctx.ret(ret as u32);
}

/// int sceIoClose(SceUID fd)
#[hostcall]
pub(super) fn io_close(st: &mut VitaState, fd: i32) -> i32 {
    st.io_close(fd)
}

/// int sceIoGetstat(const char *file, SceIoStat *stat)
/// Fills what a size query needs: the regular-file mode/attr and the 64-bit size
/// (see [`write_file_stat`]). Returns a negative errno if the path does not exist.
#[hostcall]
pub(super) fn io_getstat(ctx: &mut GuestCtx, st: &mut VitaState, file: Ptr, stat: Ptr) -> i32 {
    let path = read_cstr(ctx, file.addr());
    match st.io_size(&path) {
        Some(size) => {
            write_file_stat(ctx, stat.addr(), size);
            0
        }
        None => ENOENT,
    }
}

/// int sceIoGetstatByFd(SceUID fd, SceIoStat *stat)
/// The open-descriptor counterpart of [`io_getstat`]: a title that has already
/// opened a file often stats it by fd to size a read buffer before reading. Fills
/// the same regular-file stat from the descriptor's backing file.
#[hostcall]
pub(super) fn io_getstat_by_fd(ctx: &mut GuestCtx, st: &mut VitaState, fd: i32, stat: Ptr) -> i32 {
    match st.io_size_fd(fd) {
        Some(size) => {
            write_file_stat(ctx, stat.addr(), size);
            0
        }
        None => EBADF,
    }
}

/// Size of a guest SceIoStat: st_mode (4) + st_attr (4) + st_size (8) + three
/// SceDateTime (16 each) + st_private[6] (24).
const STAT_SIZE: usize = 88;
/// Size of a guest SceIoDirent: SceIoStat d_stat + char d_name[256] + void
/// *d_private + int dummy.
const DIRENT_SIZE: usize = STAT_SIZE + 256 + 4 + 4;

/// SceIoStat `st_mode` for a traversable directory: SCE_S_IFDIR plus user/system
/// read+exec permission (octal 010000 | 0500 | 05).
const STAT_MODE_DIR: u32 = 0o10000 | 0o500 | 0o5;
/// SceIoStat `st_attr` for a directory: SCE_SO_IFDIR.
const STAT_ATTR_DIR: u32 = 0x0010;

/// SceUID sceIoDopen(const char *dirname)
#[hostcall]
pub(super) fn io_dopen(ctx: &mut GuestCtx, st: &mut VitaState, dirname: Ptr) -> i32 {
    let path = read_cstr(ctx, dirname.addr());
    st.io_dopen(&path)
}

/// int sceIoDread(SceUID fd, SceIoDirent *dirent)
/// Returns >0 with an entry filled in, 0 at the end of the listing, or a negative
/// errno on a bad descriptor. The whole dirent is written (zeroed then filled) so a
/// title reading name or stat fields sees no stale guest memory.
#[hostcall]
pub(super) fn io_dread(ctx: &mut GuestCtx, st: &mut VitaState, fd: i32, dirent: Ptr) -> i32 {
    match st.io_dread(fd) {
        Some(Some(entry)) => {
            let mut buf = [0u8; DIRENT_SIZE];
            let (mode, attr) = if entry.is_dir {
                (STAT_MODE_DIR, STAT_ATTR_DIR)
            } else {
                (STAT_MODE_REGULAR_READABLE, STAT_ATTR_REGULAR)
            };
            buf[0..4].copy_from_slice(&mode.to_le_bytes());
            buf[4..8].copy_from_slice(&attr.to_le_bytes());
            buf[8..16].copy_from_slice(&entry.size.to_le_bytes());
            let name = entry.name.as_bytes();
            let n = name.len().min(255); // keep the trailing NUL
            buf[STAT_SIZE..STAT_SIZE + n].copy_from_slice(&name[..n]);
            ctx.write_bytes(dirent.addr(), &buf);
            1
        }
        Some(None) => 0,
        None => EBADF,
    }
}

/// int sceIoDclose(SceUID fd)
#[hostcall]
pub(super) fn io_dclose(st: &mut VitaState, fd: i32) -> i32 {
    st.io_dclose(fd)
}

/// int sceIoMkdir(const char *dir, SceMode mode)
/// The vfs is flat (paths are opaque keys), so directories are a no-op success.
#[hostcall]
pub(super) fn io_mkdir(_st: &mut VitaState, _dir: Ptr, _mode: u32) -> i32 {
    0
}

/// int sceIoRemove(const char *file)
/// No-op success: the vfs does not track a deleted state (a rewrite goes through
/// SCE_O_TRUNC on open instead). Faithful enough for bring-up.
#[hostcall]
pub(super) fn io_remove(_st: &mut VitaState, _file: Ptr) -> i32 {
    0
}

/// Bad file descriptor / no such file, as SCE returns negative on IO failure.
const EBADF: i32 = 0x8001_0009u32 as i32;
const ENOENT: i32 = 0x8001_0002u32 as i32;

/// Read a bounded NUL-terminated path string from guest memory.
pub(super) fn read_cstr(ctx: &GuestCtx, addr: u32) -> String {
    let bytes = ctx.read_bytes(addr, MAX_PATH);
    let n = bytes.iter().position(|&b| b == 0).unwrap_or(bytes.len());
    String::from_utf8_lossy(&bytes[..n]).into_owned()
}
