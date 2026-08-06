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
/// shape of a retail racer's race-load crash: 4 MB of car + course data that takes about a
/// second off a real game card completed inside two frames here. That failure is the
/// thing this constant is chosen against - see the table below.
///
/// # Why this is 50 MiB/s, which is FASTER than the hardware
/// The model exists for ORDERING, not for fidelity to a particular device, and the
/// hardware is not the target: measured retail PS Vita memory cards read at about
/// 6-8 MB/s, so even the 10 MiB/s this defaulted to first was already faster than a
/// real console. Being slower than we need to be costs real time on every run - a title
/// whose front-end streams tens of megabytes spends tens of thousands of frames on a
/// loading screen - and buys nothing, because nothing here is trying to reproduce a
/// console's loading times.
///
/// So the figure is chosen as the FASTEST rate that still preserves ordering, and that
/// boundary is measured rather than guessed. Sweeping a retail racer's gameplay recipe
/// (2500 frames, the title whose race-load crash this model was built for):
///
/// | `VITASLOP_IO_BANDWIDTH_KIBPS` | determinism signature |
/// |---|---|
/// | 10240 (10 MiB/s) | `0x24440860369ae8c9` |
/// | **25600, 51200 (25, 50 MiB/s)** | **`0xe5686dc22877ba86` - one plateau** |
/// | 102400 (100 MiB/s) | `0xc0182fa9dba87ebd` |
/// | 204800 (200 MiB/s) | `0xa061166e6b3304bb` |
/// | 0 (model OFF, instant reads) | `0xa061166e6b3304bb` - **the same** |
///
/// The last row is the finding that fixes the number: at 200 MiB/s the title behaves
/// EXACTLY as it does with no model at all, so by then the model has stopped doing its
/// job. 25 and 50 MiB/s produce bit-identical behaviour, so 50 is the top of a plateau
/// where the title cannot tell the difference from the conservative setting - with a 4x
/// margin below the point where the model stops mattering.
///
/// A second title (a retail skater with a real `@assert`) was swept the same way and is
/// bit-identical at EVERY value including `0` - it is small enough not to be I/O-bound
/// at all. That is a no-regression check, not a second boundary measurement, and it is
/// recorded as such: only a title that streams can locate this number.
///
/// Raise this further only with the same measurement, on a title that streams: a value
/// that reproduces the `0` row is not a fast model, it is no model.
///
/// Deterministic: the delay is a pure function of the byte count, so a run is as
/// reproducible as it was with instant reads.
///
/// A first attempt at this parked the reader on the GAME clock (`sleep_park`) and
/// livelocked: `virtual_us` advances a frame per display FLIP and otherwise only on the
/// scheduler's nothing-is-runnable idle path, so a title waiting for its load produced no
/// flips, a sibling polling with a zero-length delay kept the idle path finding a deadline
/// that bought no time, and the clock stood still at ANY bandwidth including 1 GiB/s. The
/// park is now charged against a separate storage clock (`VitaState::io_park`) that
/// advances on flips, on scheduler quanta and on the idle path, so it cannot depend on the
/// thing it is blocking.
const DEFAULT_BANDWIDTH_KIBPS: u64 = 50 * 1024;

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
/// Apply any `sceIoChstat` overrides recorded for `path` on top of the synthesized
/// stat already written at `stat_addr`. Read-after-write on file status is the whole
/// point of chstat, so what was set has to come back out here.
fn apply_stat_override(ctx: &mut GuestCtx, st: &VitaState, path: &str, stat_addr: u32) {
    let Some(over) = st.io_stat_override(path) else { return };
    let (mode, attr, times) = (over.mode, over.attr, over.times);
    if let Some(mode) = mode {
        ctx.write_u32(stat_addr, mode);
    }
    if let Some(attr) = attr {
        ctx.write_u32(stat_addr + 4, attr);
    }
    for (i, t) in times.iter().enumerate() {
        if let Some(t) = t {
            ctx.write_bytes(stat_addr + STAT_CTIME_OFFSET + i as u32 * 16, t);
        }
    }
}

#[hostcall]
pub(super) fn io_getstat(ctx: &mut GuestCtx, st: &mut VitaState, file: Ptr, stat: Ptr) -> i32 {
    let path = read_cstr(ctx, file.addr());
    match st.io_size(&path) {
        Some(size) => {
            write_file_stat(ctx, stat.addr(), size);
            apply_stat_override(ctx, st, &path, stat.addr());
            0
        }
        // Not a file - but a DIRECTORY still stats, and a title checking whether its
        // save directory exists asks exactly this way. Reporting ENOENT for a real
        // directory sends it down the "create everything" path every boot.
        None if st.io_is_dir(&path) => {
            let addr = stat.addr();
            if addr != 0 {
                ctx.write_u32(addr, STAT_MODE_DIR);
                ctx.write_u32(addr + 4, STAT_ATTR_DIR);
                ctx.write_u32(addr + STAT_SIZE_OFFSET, 0);
                ctx.write_u32(addr + STAT_SIZE_OFFSET + 4, 0);
                apply_stat_override(ctx, st, &path, addr);
            }
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
    write_dirent(ctx, st, fd, dirent.addr())
}

/// Read the next entry of directory descriptor `fd` into the guest `SceIoDirent` at
/// `dirent`. Shared with `SceFios2Kernel`'s directory-handle family, whose
/// `SceFiosNativeDirEntry` is this same native structure (see `vita::fios2`).
///
/// Takes a raw guest address rather than a `Ptr` because `Ptr` exists only inside a
/// `#[hostcall]` body.
pub(super) fn write_dirent(ctx: &mut GuestCtx, st: &mut VitaState, fd: i32, dirent: u32) -> i32 {
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
            ctx.write_bytes(dirent, &buf);
            1
        }
        Some(None) => 0,
        None => EBADF,
    }
}

/// Write a directory's `SceIoStat` (mode/attr plus a zero size) at `addr`. Shared
/// with `SceFios2Kernel`'s `SceFiosNativeStat`, which is this same structure.
pub(super) fn write_dir_stat(ctx: &mut GuestCtx, addr: u32) {
    ctx.write_u32(addr, STAT_MODE_DIR);
    ctx.write_u32(addr + 4, STAT_ATTR_DIR);
    ctx.write_u32(addr + STAT_SIZE_OFFSET, 0);
    ctx.write_u32(addr + STAT_SIZE_OFFSET + 4, 0);
}

/// Read the `bits`-selected fields of a guest `SceIoStat` at `addr` into the override
/// record the filesystem stores. Shared with `SceFios2Kernel`'s DH chstat.
pub(super) fn read_stat_override(ctx: &GuestCtx, addr: u32, bits: i32) -> crate::host::FileStatOverride {
    let mut over = crate::host::FileStatOverride::default();
    if bits & SCE_CST_MODE != 0 {
        over.mode = Some(ctx.read_u32(addr));
    }
    if bits & SCE_CST_ATTR != 0 {
        over.attr = Some(ctx.read_u32(addr + 4));
    }
    for (i, sel) in [SCE_CST_CT, SCE_CST_AT, SCE_CST_MT].into_iter().enumerate() {
        if bits & sel != 0 {
            let raw = ctx.read_bytes(addr + STAT_CTIME_OFFSET + i as u32 * 16, 16);
            let mut t = [0u8; 16];
            t.copy_from_slice(&raw);
            over.times[i] = Some(t);
        }
    }
    over
}

/// int sceIoDclose(SceUID fd)
#[hostcall]
pub(super) fn io_dclose(st: &mut VitaState, fd: i32) -> i32 {
    st.io_dclose(fd)
}

/// int sceIoMkdir(const char *dir, SceMode mode)
/// Really creates the directory, so a title can list, stat and remove what it made.
/// Reports EEXIST when something already occupies the path.
#[hostcall]
pub(super) fn io_mkdir(ctx: &mut GuestCtx, st: &mut VitaState, dir: Ptr, _mode: u32) -> i32 {
    let path = read_cstr(ctx, dir.addr());
    st.io_mkdir(&path)
}

/// int sceIoRmdir(const char *path)
/// Removes an EMPTY directory; a non-empty one is refused with ENOTEMPTY, as the
/// kernel refuses it.
#[hostcall]
pub(super) fn io_rmdir(ctx: &mut GuestCtx, st: &mut VitaState, path: Ptr) -> i32 {
    let path = read_cstr(ctx, path.addr());
    st.io_rmdir(&path)
}

/// int sceIoRemove(const char *file)
/// Really deletes the file: a later open without SCE_O_CREAT reports ENOENT and a
/// listing no longer shows it. A directory is refused (that is `sceIoRmdir`'s job).
#[hostcall]
pub(super) fn io_remove(ctx: &mut GuestCtx, st: &mut VitaState, file: Ptr) -> i32 {
    let path = read_cstr(ctx, file.addr());
    st.io_remove(&path)
}

/// int sceIoRename(const char *oldname, const char *newname)
/// Moves a file, or a whole directory subtree with its contents. Refuses to
/// overwrite an existing destination.
#[hostcall]
pub(super) fn io_rename(ctx: &mut GuestCtx, st: &mut VitaState, old: Ptr, new: Ptr) -> i32 {
    let (old, new) = (read_cstr(ctx, old.addr()), read_cstr(ctx, new.addr()));
    st.io_rename(&old, &new)
}

/// `SceIoStat` field offsets past the 8-byte size: the three `SceDateTime`s
/// (st_ctime, st_atime, st_mtime), 16 bytes each.
const STAT_CTIME_OFFSET: u32 = 16;

/// `sceIoChstat` `bits` selectors (SCE_CST_*), from the vitasdk io/stat header.
const SCE_CST_MODE: i32 = 0x0001;
const SCE_CST_AT: i32 = 0x0008;
const SCE_CST_MT: i32 = 0x0010;
const SCE_CST_CT: i32 = 0x0020;
/// SCE_CST_ATTR - the Vita-specific attribute word beside the POSIX mode.
const SCE_CST_ATTR: i32 = 0x0002;

/// int sceIoChstat(const char *name, SceIoStat *stat, int bits)
///
/// Apply the selected fields of `stat` to a path. `bits` says WHICH fields, so a call
/// that sets only the modification time must not also overwrite the mode - each field
/// is recorded independently and read back by `sceIoGetstat`. A chstat whose effect is
/// invisible is worse than an error: the title believes the change took.
#[hostcall]
pub(super) fn io_chstat(ctx: &mut GuestCtx, st: &mut VitaState, name: Ptr, stat: Ptr, bits: i32) -> i32 {
    let path = read_cstr(ctx, name.addr());
    if stat.addr() == 0 {
        ENOENT
    } else {
        let over = read_stat_override(ctx, stat.addr(), bits);
        st.io_chstat(&path, over)
    }
}

/// int sceIoDevctl(const char *devname, int cmd, void *arg, SceSize arglen,
///                 void *bufp, SceSize buflen)
///
/// Send a device-specific control command to a mounted device. Unlike the rest of
/// this module there is no single behaviour to implement: `cmd` selects among a
/// per-device command set, and the user-mode command numbers are not published in
/// vita-headers or the wiki.
///
/// So this is a real dispatch with no commands in it yet, and an unknown command
/// STOPS THE RUN naming the device and command rather than returning a plausible
/// success. That is the same rule the unimplemented-NID path follows and for the same
/// reason: a fabricated answer to "how much space is free" or "is a card inserted"
/// sends the title down a branch on a false premise, and the failure then surfaces
/// somewhere with no connection to this call.
pub(super) fn io_devctl(ctx: &mut GuestCtx, _st: &mut VitaState) -> SvcOutcome {
    let dev = read_cstr(ctx, ctx.arg(0));
    let cmd = ctx.arg(1);
    SvcOutcome::Fatal(format!(
        "sceIoDevctl: device {dev:?} command {cmd:#x} is not implemented - no user-mode \
         devctl command set is published, so implement this command against what the \
         title does with the result (no silent stub)"
    ))
}

/// int sceIoIoctl(SceUID fd, int cmd, void *argp, SceSize arglen, void *bufp,
///                SceSize buflen)
///
/// The by-descriptor counterpart of [`io_devctl`], with the same reasoning: a real
/// dispatch, no commands published, and an unknown one stops the run naming the
/// descriptor's file and the command.
pub(super) fn io_ioctl(ctx: &mut GuestCtx, st: &mut VitaState) -> SvcOutcome {
    let fd = ctx.arg(0) as i32;
    let cmd = ctx.arg(1);
    let known = st.io_size_fd(fd).is_some();
    SvcOutcome::Fatal(format!(
        "sceIoIoctl: fd {fd} ({}) command {cmd:#x} is not implemented - implement this \
         command against what the title does with the result (no silent stub)",
        if known { "open" } else { "not open" }
    ))
}

/// int sceIoSync(const char *devname, unsigned int flag)
/// int sceIoSyncByFd(SceUID fd, int flag)
///
/// Flush a device's (or a descriptor's) pending writes to storage. This vfs holds
/// file bytes directly with no write-back cache in front of them, so every write is
/// already durable at the instant it is made and there is genuinely nothing queued to
/// push - the sync is complete when it is asked for. That is a property of the
/// backing store, not a shortcut: were a cache ever added, this is the call that
/// would have to drain it.
#[hostcall]
pub(super) fn io_sync(_st: &mut VitaState, _devname: Ptr, _flag: u32) -> i32 {
    0
}

/// int sceIoSyncByFd(SceUID fd, int flag) - see [`io_sync`]. Validates the
/// descriptor, because syncing a closed fd is an error the caller should hear about.
#[hostcall]
pub(super) fn io_sync_by_fd(st: &mut VitaState, fd: i32, _flag: i32) -> i32 {
    if st.io_size_fd(fd).is_some() {
        0
    } else {
        EBADF
    }
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
