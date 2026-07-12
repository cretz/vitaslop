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
use crate::nid::iofilemgr as nid;
use crate::SvcOutcome;

/// Bound on a path string read from guest memory.
const MAX_PATH: usize = 1024;

/// Offset of `st_size` (SceOff, 64-bit) within SceIoStat: after `st_mode` (u32)
/// and `st_attr` (u32).
const STAT_SIZE_OFFSET: u32 = 8;

pub fn try_dispatch(func_nid: u32, ctx: &mut GuestCtx, st: &mut VitaState) -> Option<SvcOutcome> {
    match func_nid {
        nid::IO_OPEN => io_open(ctx, st),
        nid::IO_CLOSE => io_close(ctx, st),
        nid::IO_READ => io_read(ctx, st),
        nid::IO_WRITE => io_write(ctx, st),
        nid::IO_LSEEK32 => io_lseek32(ctx, st),
        nid::IO_LSEEK => io_lseek(ctx, st), // hand-written 64-bit
        nid::IO_GETSTAT => io_getstat(ctx, st),
        nid::IO_MKDIR => io_mkdir(ctx, st),
        nid::IO_REMOVE => io_remove(ctx, st),
        _ => return None,
    }
    Some(SvcOutcome::Continue)
}

/// SceUID sceIoOpen(const char *file, int flags, SceMode mode)
#[hostcall]
fn io_open(ctx: &mut GuestCtx, st: &mut VitaState, file: Ptr, flags: u32, _mode: u32) -> i32 {
    let path = read_cstr(ctx, file.addr());
    st.io_open(&path, flags)
}

/// int sceIoRead(SceUID fd, void *buf, SceSize size)
/// Returns the number of bytes read, or a negative errno on a bad descriptor.
#[hostcall]
fn io_read(ctx: &mut GuestCtx, st: &mut VitaState, fd: i32, buf: Ptr, size: u32) -> i32 {
    match st.io_read(fd, size as usize) {
        Some(data) => {
            ctx.write_bytes(buf.addr(), &data);
            data.len() as i32
        }
        None => EBADF,
    }
}

/// int sceIoWrite(SceUID fd, const void *buf, SceSize size)
/// fd 1/2 go to the captured console; other descriptors to the vfs.
#[hostcall]
fn io_write(ctx: &mut GuestCtx, st: &mut VitaState, fd: i32, buf: Ptr, size: u32) -> i32 {
    let bytes = ctx.read_bytes(buf.addr(), size as usize);
    match st.io_write(fd, &bytes) {
        Some(n) => n as i32,
        None => EBADF,
    }
}

/// int sceIoLseek32(SceUID fd, int offset, int whence)
/// The 32-bit seek: everything fits in core registers and the return.
#[hostcall]
fn io_lseek32(st: &mut VitaState, fd: i32, offset: i32, whence: i32) -> i32 {
    st.io_lseek(fd, offset as i64, whence) as i32
}

/// SceOff sceIoLseek(SceUID fd, SceOff offset, int whence)
///
/// `SceOff` is 64-bit: with `fd` in r0, the offset takes the even-aligned r2:r3
/// pair (r1 is padding) and `whence` spills to the stack; the return goes in
/// r0:r1. The macro cannot express this, so it is marshalled by hand.
fn io_lseek(ctx: &mut GuestCtx, st: &mut VitaState) {
    let fd = ctx.arg(0) as i32;
    let offset = (ctx.arg(2) as u64 | (ctx.arg(3) as u64) << 32) as i64;
    let whence = ctx.arg(4) as i32;
    let pos = st.io_lseek(fd, offset, whence);
    ctx.regs[0] = pos as u32;
    ctx.regs[1] = (pos >> 32) as u32;
}

/// int sceIoClose(SceUID fd)
#[hostcall]
fn io_close(st: &mut VitaState, fd: i32) -> i32 {
    st.io_close(fd)
}

/// int sceIoGetstat(const char *file, SceIoStat *stat)
/// Fills only what a size query needs: zero the mode/attr words and write the
/// 64-bit size. Returns a negative errno if the path does not exist.
#[hostcall]
fn io_getstat(ctx: &mut GuestCtx, st: &mut VitaState, file: Ptr, stat: Ptr) -> i32 {
    let path = read_cstr(ctx, file.addr());
    match st.io_size(&path) {
        Some(size) => {
            if !stat.is_null() {
                ctx.write_u32(stat.addr(), 0); // st_mode
                ctx.write_u32(stat.addr() + 4, 0); // st_attr
                ctx.write_u32(stat.addr() + STAT_SIZE_OFFSET, size as u32);
                ctx.write_u32(stat.addr() + STAT_SIZE_OFFSET + 4, (size >> 32) as u32);
            }
            0
        }
        None => ENOENT,
    }
}

/// int sceIoMkdir(const char *dir, SceMode mode)
/// The vfs is flat (paths are opaque keys), so directories are a no-op success.
#[hostcall]
fn io_mkdir(_st: &mut VitaState, _dir: Ptr, _mode: u32) -> i32 {
    0
}

/// int sceIoRemove(const char *file)
/// No-op success: the vfs does not track a deleted state (a rewrite goes through
/// SCE_O_TRUNC on open instead). Faithful enough for bring-up.
#[hostcall]
fn io_remove(_st: &mut VitaState, _file: Ptr) -> i32 {
    0
}

/// Bad file descriptor / no such file, as SCE returns negative on IO failure.
const EBADF: i32 = 0x8001_0009u32 as i32;
const ENOENT: i32 = 0x8001_0002u32 as i32;

/// Read a bounded NUL-terminated path string from guest memory.
fn read_cstr(ctx: &GuestCtx, addr: u32) -> String {
    let bytes = ctx.read_bytes(addr, MAX_PATH);
    let n = bytes.iter().position(|&b| b == 0).unwrap_or(bytes.len());
    String::from_utf8_lossy(&bytes[..n]).into_owned()
}
