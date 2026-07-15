//! The host-call boundary: how a guest NID import trap becomes a typed Rust
//! handler. `GuestCtx` marshals AAPCS arguments (r0..r3 then stack) and guest
//! memory in and out; `VitaEnv` owns the per-run state (allocator, handles,
//! capture, world) and routes a dense import index to a per-module handler.
//! See `projects/vitaslop-runtime/README.md`.

use vitaslop_transpiler::abi::{REG_COUNT, SP};

use crate::capture::Capture;
use crate::world::{DeterministicWorld, World};
use crate::{vita, SvcOutcome};

/// The number of VFP single-precision registers (s0..s15) that carry floating-
/// point arguments and the float return under AAPCS-VFP (the Vita is hardfloat).
/// The host reads these from the guest at a NID trap alongside the core registers
/// so [`GuestCtx`] can marshal float args and returns.
pub const VFP_ARG_COUNT: usize = 16;

/// A guest address crossing the host-call boundary (an in- or out-parameter). A
/// newtype so `#[hostcall]` classifies it as integer class (core register / stack)
/// while keeping the intent - "this is a pointer" - visible in the signature. The
/// handler dereferences it through [`GuestCtx`] (`read_u32`, `write_bytes`, ...).
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub struct Ptr(pub u32);

impl Ptr {
    /// The raw guest address.
    pub fn addr(self) -> u32 {
        self.0
    }
    /// True for a null guest pointer.
    pub fn is_null(self) -> bool {
        self.0 == 0
    }
}

/// Random-access to the guest's linear memory during a host call, rebased so
/// guest address `A` is byte `A - base`. The abstraction exists because the two
/// engines reach guest memory differently: on native wasmtime the runtime and
/// guest share one address space, so a plain `&mut [u8]` slice works and is
/// zero-copy; in the browser the guest runs as its own `WebAssembly` instance
/// with a linear memory the runtime's (wasm-bindgen) module cannot borrow as a
/// Rust slice, so its impl copies through the guest `ArrayBuffer`. Host calls
/// happen only at kernel/GXM boundaries (tens per frame), never in the hot CPU
/// loop, so the per-access indirection here costs nothing on the fast path.
///
/// Callers (via [`GuestCtx`]) validate `off` against [`len`](GuestMemory::len)
/// and clamp before calling, so implementors may assume `off` and
/// `off + buf.len()` are in range.
pub trait GuestMemory {
    /// The size of the provisioned guest region, in bytes from `base`.
    fn len(&self) -> usize;
    /// True when no guest memory is provisioned.
    fn is_empty(&self) -> bool {
        self.len() == 0
    }
    /// Read `buf.len()` bytes starting at rebased offset `off` into `buf`.
    fn read(&self, off: usize, buf: &mut [u8]);
    /// Write `bytes` at rebased offset `off`.
    fn write(&mut self, off: usize, bytes: &[u8]);
}

/// The native/zero-copy backing: guest memory the runtime can borrow directly
/// (wasmtime linear memory, or any in-process buffer). A `Sized` newtype so it
/// can coerce to `&mut dyn GuestMemory` (an unsized `[u8]` cannot back a trait
/// object). Wrap the engine's rebased slice: `SliceMemory(bytes)`.
pub struct SliceMemory<'a>(pub &'a mut [u8]);

impl GuestMemory for SliceMemory<'_> {
    fn len(&self) -> usize {
        self.0.len()
    }
    fn read(&self, off: usize, buf: &mut [u8]) {
        buf.copy_from_slice(&self.0[off..off + buf.len()]);
    }
    fn write(&mut self, off: usize, bytes: &[u8]) {
        self.0[off..off + bytes.len()].copy_from_slice(bytes);
    }
}

/// A borrowed view of guest state for the duration of one host call: the
/// register file, guest memory (rebased so guest address `A` is byte `A - base`),
/// and a sequential AAPCS argument cursor.
pub struct GuestCtx<'a> {
    pub regs: &'a mut [u32; REG_COUNT],
    /// Raw bits of the VFP single-precision registers s0..s15 (the float argument
    /// and return file). d`n` is (s[2n], s[2n+1]).
    pub vfp: &'a mut [u32; VFP_ARG_COUNT],
    pub mem: &'a mut dyn GuestMemory,
    pub base: u32,
    /// Next core (integer/pointer) argument slot: args 0..3 are r0..r3, args >=4
    /// are on the stack at sp + (n-4)*4.
    next_core: usize,
    /// Next VFP single-register slot for a float argument (AAPCS-VFP keeps a
    /// separate counter from the core one; doubles align to an even slot).
    next_vfp: usize,
}

impl<'a> GuestCtx<'a> {
    pub(crate) fn new(
        regs: &'a mut [u32; REG_COUNT],
        vfp: &'a mut [u32; VFP_ARG_COUNT],
        mem: &'a mut dyn GuestMemory,
        base: u32,
    ) -> Self {
        GuestCtx { regs, vfp, mem, base, next_core: 0, next_vfp: 0 }
    }

    /// Read positional integer/pointer argument `n` (AAPCS: r0..r3 then stack).
    pub fn arg(&self, n: usize) -> u32 {
        if n < 4 {
            self.regs[n]
        } else {
            let sp = self.regs[SP];
            self.read_u32(sp.wrapping_add(((n - 4) * 4) as u32))
        }
    }

    /// Read the next core integer/pointer argument and advance the core cursor.
    pub fn next_u32(&mut self) -> u32 {
        let v = self.arg(self.next_core);
        self.next_core += 1;
        v
    }

    /// Read the next float argument (a single VFP register) and advance the VFP
    /// cursor. Used by `#[hostcall]` for `f32` parameters.
    pub fn next_f32(&mut self) -> f32 {
        let i = self.next_vfp;
        self.next_vfp += 1;
        let bits = self.vfp.get(i).copied().unwrap_or(0);
        f32::from_bits(bits)
    }

    /// Read the next double argument (an even-aligned VFP register pair) and
    /// advance the VFP cursor past it. Used by `#[hostcall]` for `f64` parameters.
    pub fn next_f64(&mut self) -> f64 {
        // Doubles occupy an even-aligned single-register pair (d`k` = s[2k],
        // s[2k+1]); align up to the next even slot, no back-fill.
        let i = (self.next_vfp + 1) & !1;
        self.next_vfp = i + 2;
        let lo = self.vfp.get(i).copied().unwrap_or(0) as u64;
        let hi = self.vfp.get(i + 1).copied().unwrap_or(0) as u64;
        f64::from_bits(lo | (hi << 32))
    }

    /// Set the call's float return value in s0.
    pub fn ret_f32(&mut self, v: f32) {
        self.vfp[0] = v.to_bits();
    }

    /// Set the call's double return value in d0 (s0 low, s1 high).
    pub fn ret_f64(&mut self, v: f64) {
        let bits = v.to_bits();
        self.vfp[0] = bits as u32;
        self.vfp[1] = (bits >> 32) as u32;
    }

    /// The offset into guest memory for guest address `addr`, or None if outside
    /// the provisioned region.
    fn offset(&self, addr: u32) -> Option<usize> {
        let off = addr.checked_sub(self.base)? as usize;
        if off <= self.mem.len() {
            Some(off)
        } else {
            None
        }
    }

    /// Read a little-endian u32 at guest address `addr` (0 if out of range).
    pub fn read_u32(&self, addr: u32) -> u32 {
        match self.offset(addr) {
            Some(o) if o + 4 <= self.mem.len() => {
                let mut b = [0u8; 4];
                self.mem.read(o, &mut b);
                u32::from_le_bytes(b)
            }
            _ => 0,
        }
    }

    /// Write a little-endian u32 at guest address `addr` (ignored if out of range).
    pub fn write_u32(&mut self, addr: u32, v: u32) {
        if let Some(o) = self.offset(addr) {
            if o + 4 <= self.mem.len() {
                self.mem.write(o, &v.to_le_bytes());
            }
        }
    }

    /// Read `len` bytes at guest address `addr` (short read clamped to range).
    pub fn read_bytes(&self, addr: u32, len: usize) -> Vec<u8> {
        match self.offset(addr) {
            Some(o) => {
                let end = (o + len).min(self.mem.len());
                let mut buf = vec![0u8; end - o];
                self.mem.read(o, &mut buf);
                buf
            }
            None => Vec::new(),
        }
    }

    /// Write `bytes` at guest address `addr` (clamped to range).
    pub fn write_bytes(&mut self, addr: u32, bytes: &[u8]) {
        if let Some(o) = self.offset(addr) {
            let end = (o + bytes.len()).min(self.mem.len());
            self.mem.write(o, &bytes[..end - o]);
        }
    }

    /// Read a little-endian f32 at guest address `addr`.
    pub fn read_f32(&self, addr: u32) -> f32 {
        f32::from_bits(self.read_u32(addr))
    }

    /// Read a NUL-terminated ASCII string at guest address `addr` (bounded).
    pub fn read_cstr(&self, addr: u32, max: usize) -> String {
        let bytes = self.read_bytes(addr, max);
        let n = bytes.iter().position(|&b| b == 0).unwrap_or(bytes.len());
        String::from_utf8_lossy(&bytes[..n]).into_owned()
    }

    /// Set the call's return value (r0).
    pub fn ret(&mut self, v: u32) {
        self.regs[0] = v;
    }
}

/// A GXM opaque handle we hand back to the guest. Kept below the guest image base
/// so it never collides with a real guest address, and nonzero so guest NULL
/// checks pass. The guest treats these as opaque and never dereferences them.
pub const HANDLE_BASE: u32 = 0x4000_0000;

/// A GPU memory block the guest allocated: its SceUID, guest base address, and
/// size. Backed by real guest memory so the guest can read and write it.
pub struct MemBlock {
    pub uid: i32,
    pub base: u32,
    pub size: u32,
}

/// A guest thread the program created: its SceUID, entry address (Thumb bit
/// cleared, so it names the transpiled `f_<addr>` export), its own stack top, and
/// its return value once it has run.
struct ThreadRec {
    uid: i32,
    entry: u32,
    stack_top: u32,
    exit_code: Option<u32>,
    /// SceKernel thread priority (lower number = higher priority). The scheduler
    /// runs the highest-priority runnable thread, matching the real kernel.
    priority: i32,
}

/// The default SceKernel user-thread priority (`SCE_KERNEL_DEFAULT_PRIORITY_USER`),
/// used for the initial (main) thread, which is not created via `create_thread`.
pub const DEFAULT_THREAD_PRIORITY: i32 = 0x1000_0100;

/// A recursive mutex's state (preemptive mode only; the single-thread model needs
/// none). `owner` is the holding thread's id (None if free), `count` the recursion
/// depth, `waiters` the threads parked in `sceKernelLockMutex` in FIFO order.
struct MutexRec {
    uid: i32,
    owner: Option<i32>,
    count: i32,
    waiters: Vec<i32>,
}

/// A condition variable's state (preemptive mode only). `mutex` is the associated
/// mutex it releases on wait and re-acquires on wake; `waiters` are the threads
/// parked in `sceKernelWaitCond` in FIFO order. A condition variable has no
/// memory: a signal with no waiter is lost.
struct CondRec {
    uid: i32,
    mutex: i32,
    waiters: Vec<i32>,
}

/// A thread parked in `sceKernelWaitSema`: which semaphore, which thread, and how
/// many signals it still needs. It is released (and the count consumed) when a
/// signal makes `need` available.
struct SemaWaiter {
    uid: i32,
    thid: i32,
    need: i32,
}

/// A request to synchronously run guest code (a thread entry) that a host call
/// raised. The engine-agnostic runtime cannot itself re-enter the wasm engine, so
/// it records this intent and the engine host ([`vitaslop_native::Vm`]) executes
/// it - seeding r0/r1 and sp, calling the guest function, and reporting the
/// result back through [`ImportDispatch::set_thread_exit`]. The register file is
/// saved and restored around the call so the re-entry is transparent to the
/// interrupted thread.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub struct Reentry {
    /// Guest function to run (Thumb bit cleared; names the `f_<addr>` export).
    pub entry: u32,
    /// r0 for the entry: the argument-block length (`SceSize args`).
    pub arg_len: u32,
    /// r1 for the entry: the argument-block pointer (`void *argp`).
    pub arg_ptr: u32,
    /// sp for the entry: the top of the thread's own stack.
    pub stack_top: u32,
    /// The thread whose exit code the result becomes.
    pub thid: i32,
    /// The new thread's SceKernel priority (lower = higher), so the scheduler can
    /// place it correctly on the run queue.
    pub priority: i32,
}

/// One open file descriptor: which path it refers to, the read/write cursor, and
/// the access mode decoded from the open flags.
struct OpenFile {
    path: String,
    cursor: usize,
    readable: bool,
    writable: bool,
}

/// A minimal virtual filesystem backing SceIoFilemgr: a path -> bytes map plus a
/// table of open descriptors. Read files are preloaded by the harness (e.g. a
/// game's data files); write opens create or truncate entries here. The console
/// streams (fd 1/2) are handled directly in the IO handlers and are not tracked
/// here. Deterministic and host-only, so a run touches no real filesystem.
#[derive(Default)]
pub struct FileTable {
    files: std::collections::HashMap<String, Vec<u8>>,
    open: std::collections::HashMap<i32, OpenFile>,
    next_fd: i32,
}

// Vita open flags (SCE_O_*), from the MIT vita-headers.
const SCE_O_RDONLY: u32 = 0x0001;
const SCE_O_WRONLY: u32 = 0x0002;
const SCE_O_RDWR: u32 = 0x0003;
const SCE_O_APPEND: u32 = 0x0100;
const SCE_O_CREAT: u32 = 0x0200;
const SCE_O_TRUNC: u32 = 0x0400;

// Seek origins.
const SCE_SEEK_SET: i32 = 0;
const SCE_SEEK_CUR: i32 = 1;
const SCE_SEEK_END: i32 = 2;

/// A generic "no such file/dir" errno, as SCE returns negative on IO failure.
/// The guest only ever tests fd < 0, so the exact value only needs to be
/// negative; this is the real `SCE_ERROR_ERRNO_ENOENT`.
const SCE_ERROR_ERRNO_ENOENT: i32 = 0x8001_0002u32 as i32;
/// Bad file descriptor.
const SCE_ERROR_ERRNO_EBADF: i32 = 0x8001_0009u32 as i32;

/// Reserved descriptors: 0 stdin, 1 stdout, 2 stderr. Real fds start at 3.
const FIRST_FD: i32 = 3;
pub const FD_STDOUT: i32 = 1;
pub const FD_STDERR: i32 = 2;

/// Resolve a guest file path to a VFS key.
///
/// Two Vita filesystem facts are folded in here so a title finds its data files:
/// - **Mount point.** A Vita mounts the running app's own read-only data at `app0:`,
///   and the decrypted game files are stored under their paths relative to that mount,
///   so an `app0:` prefix (with or without the separator) is stripped. Other mounts -
///   `ux0:`, `savedata0:` - keep their full path, a distinct, writable space.
/// - **Case-insensitivity.** The Vita's FAT/exFAT filesystem is case-insensitive, so
///   the key is lowercased. Titles routinely request an asset with different casing
///   than the shipped filename (e.g. open `Neondecals.gxt` for the file
///   `neondecals.gxt`); a case-sensitive map would miss it.
///
/// Without this a title that opens `app0:/settings/foo.ini` or `.../Foo.GXT` would
/// miss the file stored as `settings/foo.ini` and take its file-missing (often fatal)
/// path.
fn vfs_key(path: &str) -> String {
    path.strip_prefix("app0:/")
        .or_else(|| path.strip_prefix("app0:"))
        .unwrap_or(path)
        .to_lowercase()
}

impl FileTable {
    fn new() -> Self {
        FileTable { next_fd: FIRST_FD, ..Default::default() }
    }

    /// Open `path` per the SCE_O_* `flags`; returns a new fd or a negative errno.
    fn open(&mut self, path: &str, flags: u32) -> i32 {
        let path = vfs_key(path);
        let readable = flags & SCE_O_RDWR == SCE_O_RDONLY || flags & SCE_O_RDWR == SCE_O_RDWR;
        let writable = flags & SCE_O_WRONLY != 0;
        let exists = self.files.contains_key(&path);

        if !exists {
            if flags & SCE_O_CREAT != 0 {
                self.files.insert(path.clone(), Vec::new());
            } else {
                return SCE_ERROR_ERRNO_ENOENT;
            }
        } else if flags & SCE_O_TRUNC != 0 {
            self.files.insert(path.clone(), Vec::new());
        }

        // Append seeks to end; every other open starts at the beginning.
        let cursor = if flags & SCE_O_APPEND != 0 {
            self.files.get(&path).map(|d| d.len()).unwrap_or(0)
        } else {
            0
        };

        let fd = self.next_fd;
        self.next_fd += 1;
        self.open.insert(fd, OpenFile { path, cursor, readable, writable });
        fd
    }

    /// Read up to `len` bytes from `fd` at its cursor; advances it. Returns the
    /// bytes, or None on a bad/unreadable descriptor.
    fn read(&mut self, fd: i32, len: usize) -> Option<Vec<u8>> {
        let of = self.open.get_mut(&fd)?;
        if !of.readable {
            return None;
        }
        let data = self.files.get(&of.path)?;
        let start = of.cursor.min(data.len());
        let end = (start + len).min(data.len());
        let out = data[start..end].to_vec();
        of.cursor = end;
        Some(out)
    }

    /// Read up to `len` bytes from `fd` starting at absolute `offset` WITHOUT moving
    /// the descriptor's cursor (sceIoPread). Returns the bytes, or None on a
    /// bad/unreadable descriptor. This is how a streaming reader (e.g. the AT9 music
    /// streamer) pulls the file header and successive chunks from a shared fd.
    fn pread(&self, fd: i32, offset: u64, len: usize) -> Option<Vec<u8>> {
        let of = self.open.get(&fd)?;
        if !of.readable {
            return None;
        }
        let data = self.files.get(&of.path)?;
        let start = (offset as usize).min(data.len());
        let end = (start + len).min(data.len());
        Some(data[start..end].to_vec())
    }

    /// Write `bytes` to `fd` at absolute `offset` WITHOUT moving the cursor
    /// (sceIoPwrite), extending the file with zeros if needed. Returns the count, or
    /// None on a bad/unwritable descriptor.
    fn pwrite(&mut self, fd: i32, offset: u64, bytes: &[u8]) -> Option<usize> {
        let of = self.open.get(&fd)?;
        if !of.writable {
            return None;
        }
        let path = of.path.clone();
        let data = self.files.entry(path).or_default();
        let end = offset as usize + bytes.len();
        if end > data.len() {
            data.resize(end, 0);
        }
        data[offset as usize..end].copy_from_slice(bytes);
        Some(bytes.len())
    }

    /// Write `bytes` to `fd` at its cursor (extending the file); advances it.
    /// Returns the count written, or None on a bad/unwritable descriptor.
    fn write(&mut self, fd: i32, bytes: &[u8]) -> Option<usize> {
        let of = self.open.get_mut(&fd)?;
        if !of.writable {
            return None;
        }
        let data = self.files.entry(of.path.clone()).or_default();
        if of.cursor > data.len() {
            data.resize(of.cursor, 0);
        }
        let end = of.cursor + bytes.len();
        if end > data.len() {
            data.resize(end, 0);
        }
        data[of.cursor..end].copy_from_slice(bytes);
        of.cursor = end;
        Some(bytes.len())
    }

    /// Seek `fd` to `offset` from `whence`; returns the new absolute position or a
    /// negative errno.
    fn lseek(&mut self, fd: i32, offset: i64, whence: i32) -> i64 {
        let Some(of) = self.open.get_mut(&fd) else {
            return SCE_ERROR_ERRNO_EBADF as i64;
        };
        let size = self.files.get(&of.path).map(|d| d.len()).unwrap_or(0) as i64;
        let base = match whence {
            SCE_SEEK_SET => 0,
            SCE_SEEK_CUR => of.cursor as i64,
            SCE_SEEK_END => size,
            _ => return SCE_ERROR_ERRNO_EBADF as i64,
        };
        let pos = base + offset;
        if pos < 0 {
            return SCE_ERROR_ERRNO_EBADF as i64;
        }
        of.cursor = pos as usize;
        pos
    }

    /// Close `fd`; returns 0 or a negative errno.
    fn close(&mut self, fd: i32) -> i32 {
        if self.open.remove(&fd).is_some() {
            0
        } else {
            SCE_ERROR_ERRNO_EBADF
        }
    }

    /// The size of `path` if it exists (for sceIoGetstat).
    fn size_of(&self, path: &str) -> Option<u64> {
        self.files.get(&vfs_key(path)).map(|d| d.len() as u64)
    }
}

/// Per-draw vertex program layout captured at create time, keyed by the vertex
/// program handle so a later Draw knows how to snapshot the vertex buffer.
struct VertexProgramInfo {
    attributes: Vec<crate::capture::VertexAttribute>,
    stride: u32,
}

/// All host state for one run: the guest allocator, handle tables, the capture
/// stream, the world (determinism seam), and the in-progress scene state.
pub struct VitaState {
    pub base: u32,
    pub mem_bytes: u32,
    alloc_cursor: u32,
    next_handle: u32,
    next_uid: i32,
    memblocks: Vec<MemBlock>,
    vertex_programs: Vec<(u32, VertexProgramInfo)>,
    color_surfaces: Vec<(u32, crate::capture::ColorSurface)>,
    /// Guest address of the displayQueue callback from sceGxmInitialize, and its
    /// data size. Recorded for faithfulness; the present address is captured from
    /// the display queue directly so the callback need not run yet.
    pub display_queue_cb: u32,
    pub display_queue_cb_data_size: u32,
    // In-progress scene (BeginScene..EndScene).
    scene: Option<crate::capture::Scene>,
    bound_vertex_program: u32,
    bound_stream0: u32,
    pending_uniforms: Vec<f32>,
    // Threads the program created, and any pending synchronous thread run raised
    // by sceKernelStartThread (drained by the engine host after the call).
    threads: Vec<ThreadRec>,
    pending_reentry: Option<Reentry>,
    // Synchronization objects. Bring-up model: one thread of control (workers run
    // synchronously to completion), so nothing ever actually blocks; a semaphore's
    // count and an event flag's bit pattern are still tracked so their observable
    // state is faithful for single-thread use (guarding data, wait-then-read).
    semaphores: Vec<(i32, i32)>,
    event_flags: Vec<(i32, u32)>,
    // Preemptive-mode state (unused in the single-thread model). When `preemptive`
    // is set, blocking primitives actually park the calling thread (`current`) and
    // are woken by another thread's signal/unlock/thread-end; the scheduler drains
    // `pending_spawns` (threads to start as their own fibers) and `pending_wakes`
    // (parked threads made runnable) after each host call. See the runtime README
    // concurrency model and `vitaslop_native::ThreadedScheduler`.
    preemptive: bool,
    current: i32,
    mutexes: Vec<MutexRec>,
    conds: Vec<CondRec>,
    sema_waiters: Vec<SemaWaiter>,
    /// (waiter thread id, target thread id, stat pointer) for threads parked in a
    /// join. `stat` is the guest `int *` the joiner passed to `sceKernelWaitThreadEnd`
    /// (0 = NULL); the target's exit code is written there when the join completes.
    join_waiters: Vec<(i32, i32, u32)>,
    pending_spawns: Vec<Reentry>,
    pending_wakes: Vec<i32>,
    /// Guest memory writes to apply when a blocked joiner is woken: `(stat_ptr,
    /// exit_code)`. `sceKernelWaitThreadEnd`'s handler cannot re-run at wake time, so
    /// the exit code it owes the joiner's `stat` out-parameter is queued here and
    /// applied by the scheduler (which has memory access) before the joiner resumes.
    pending_stat_writes: Vec<(u32, u32)>,
    // Virtual filesystem backing SceIoFilemgr (open/read/write/lseek/close).
    fs: FileTable,
    pub capture: Capture,
    // `Send` so a `VitaEnv` can be the data of a wasmtime async Store (the
    // cooperative scheduler runs the guest on a fiber, which wasmtime may resume
    // on any thread). Everything stays single-threaded in practice.
    pub world: Box<dyn World + Send>,
    /// The audio-output sink every `sceAudioOut` port feeds. A `NullSink` by default
    /// (silent), replaced by a host that wants real playback (native device / Web
    /// Audio). See [`crate::audio`].
    pub audio: Box<dyn crate::audio::AudioSink + Send>,
    /// Opened NGS/audio port and handle bookkeeping (see `vita::ngs` / `vita::audio`).
    pub(crate) audio_state: crate::vita::audio::AudioState,
    /// Bring-up aid: halt the run when the guest calls sceGxmTerminate. The cube
    /// entry is `_start`, which spins forever after `main` returns (there is no OS
    /// to exit to yet), so terminate is the clean stopping point after teardown.
    pub halt_on_terminate: bool,
    /// Guest address of the main module's `SceProcessParam`, returned verbatim by
    /// `sceKernelGetProcessParam`. libc's crt reads the `SceLibcParam` it points to
    /// for the heap configuration, so this must be a real address (0 would fault).
    process_param: u32,
    /// Per-`(thread, key)` thread-local storage slots handed out by
    /// `sceKernelGetTLSAddr`. Each is a distinct zero-initialized guest block whose
    /// address is stable across calls, so a thread stores and reads back its own
    /// TLS pointer faithfully.
    tls_slots: Vec<((i32, u32), u32)>,
    /// `(shaderPatcherId handle, SceGxmProgram* header)` from every
    /// `sceGxmShaderPatcherRegisterProgram`, so `sceGxmShaderPatcherGetProgramFromId`
    /// can hand back the real program pointer the guest registered.
    shader_programs: Vec<(u32, u32)>,
    /// Fragment textures currently bound by `sceGxmSetFragmentTexture`, keyed by
    /// sampler unit -> guest `SceGxmTexture*`. Bindings persist across draws until
    /// rebound (GXM state is sticky), so this is read - not cleared - at each draw.
    bound_textures: Vec<(u32, u32)>,
    /// Exact `SceGxmTextureFormat` last set on a guest `SceGxmTexture*` via
    /// `sceGxmTextureInit*`/`SetFormat`. The 16-byte control words alone lose the
    /// channel swizzle (only a 3-bit field survives), so we keep the full 32-bit
    /// format the guest passed for an exact decode; a texture the guest fills
    /// directly (a `.gxt` blob) is absent here and falls back to control-word parse.
    texture_formats: Vec<(u32, u32)>,
    /// Threads parked in `sceKernelWaitLwCond`, as `(thread id, cond work address,
    /// wake deadline)`. `deadline` is `Some(virtual_us)` for a timed wait (woken by
    /// a signal or when the clock reaches it) or `None` for an infinite wait (only a
    /// signal wakes it). Keyed by the cond's guest work pointer, since lightweight
    /// objects have no kernel handle. See the scheduler's idle-time advance.
    lwcond_waiters: Vec<(i32, u32, Option<u64>)>,
    /// Threads sleeping until a virtual-clock deadline, as `(thread id, deadline_us)`.
    /// Unlike a lwcond waiter these are woken purely by time - `sceKernelDelayThread`
    /// and, crucially, `sceAudioOutOutput` pacing: an audio mixer thread hands one
    /// grain per call and on hardware blocks until it drains, which paces it to real
    /// time. Parking it here means the audio thread stops busy-spinning and, because
    /// it now yields, the scheduler goes idle and can advance the clock - so a title's
    /// time-based loading wait actually progresses instead of being starved.
    sleep_waiters: Vec<(i32, u64)>,
    /// The virtual monotonic clock in microseconds, read by
    /// `sceKernelGetProcessTimeWide`/`GetSystemTimeWide`. The scheduler advances it
    /// (jumping to the earliest pending deadline when every thread is parked), so a
    /// timed wait costs one round instead of millions of busy-poll iterations.
    virtual_us: u64,
}

impl VitaState {
    /// New state for a run over `[base, base + mem_bytes)`. Allocations start
    /// above the image (at base + 1 MiB) and grow up, well below the stack that
    /// starts at the top of the region.
    pub fn new(base: u32, mem_bytes: u32, world: Box<dyn World + Send>) -> Self {
        VitaState {
            base,
            mem_bytes,
            alloc_cursor: base + 0x0010_0000,
            next_handle: HANDLE_BASE + 1,
            next_uid: 0x100,
            memblocks: Vec::new(),
            vertex_programs: Vec::new(),
            color_surfaces: Vec::new(),
            display_queue_cb: 0,
            display_queue_cb_data_size: 0,
            scene: None,
            bound_vertex_program: 0,
            bound_stream0: 0,
            pending_uniforms: Vec::new(),
            threads: Vec::new(),
            pending_reentry: None,
            semaphores: Vec::new(),
            event_flags: Vec::new(),
            preemptive: false,
            current: 0,
            mutexes: Vec::new(),
            conds: Vec::new(),
            sema_waiters: Vec::new(),
            join_waiters: Vec::new(),
            pending_spawns: Vec::new(),
            pending_wakes: Vec::new(),
            pending_stat_writes: Vec::new(),
            fs: FileTable::new(),
            capture: Capture::new(),
            world,
            audio: Box::new(crate::audio::NullSink::default()),
            audio_state: crate::vita::audio::AudioState::default(),
            halt_on_terminate: false,
            process_param: 0,
            tls_slots: Vec::new(),
            shader_programs: Vec::new(),
            bound_textures: Vec::new(),
            texture_formats: Vec::new(),
            lwcond_waiters: Vec::new(),
            sleep_waiters: Vec::new(),
            virtual_us: 0,
        }
    }

    /// Record a registered shader program (its guest `SceGxmProgram*` header) and
    /// return the opaque `shaderPatcherId` handle the guest will pass around.
    pub fn register_shader_program(&mut self, program_header: u32) -> u32 {
        let handle = self.new_handle();
        self.shader_programs.push((handle, program_header));
        handle
    }

    /// The `SceGxmProgram*` header for a `shaderPatcherId`, or 0 if unknown.
    pub fn shader_program(&self, id: u32) -> u32 {
        self.shader_programs
            .iter()
            .find(|&&(h, _)| h == id)
            .map(|&(_, p)| p)
            .unwrap_or(0)
    }

    /// Record the guest address of the main module's `SceProcessParam` (from
    /// [`crate::link::LinkedProgram::process_param`]) so `sceKernelGetProcessParam`
    /// can hand it to libc. Set once before the run.
    pub fn set_process_param(&mut self, addr: u32) {
        self.process_param = addr;
    }

    /// The `SceProcessParam` address (0 if the title carries none).
    pub fn process_param(&self) -> u32 {
        self.process_param
    }

    /// The stable guest address of TLS slot `key` for the currently running thread,
    /// allocating a fresh zero-initialized 4-byte pointer slot on first use. Guest
    /// memory starts zeroed, so a never-written slot reads back as NULL.
    pub fn tls_addr(&mut self, key: u32) -> u32 {
        let thread = self.current;
        if let Some(&(_, addr)) = self.tls_slots.iter().find(|&&(k, _)| k == (thread, key)) {
            return addr;
        }
        // A pointer-sized slot per key is what the low-level TLS API hands out; the
        // caller stores its own per-thread pointer there.
        let addr = self.galloc(4, 4);
        self.tls_slots.push(((thread, key), addr));
        addr
    }

    /// Move the heap allocation cursor to `addr`. A multi-module linked title
    /// (see [`crate::link`]) fills far more than the default 1 MiB below the heap,
    /// so the host must set this above the whole image (`LinkedProgram::alloc_base`)
    /// before the run, or allocations would overwrite guest code.
    pub fn set_alloc_base(&mut self, addr: u32) {
        self.alloc_cursor = addr;
    }

    // --- SceIoFilemgr virtual filesystem ---

    /// Preload a read-only file into the virtual filesystem before a run (e.g. a
    /// title's data file). The guest can then `sceIoOpen`/`sceIoRead` it.
    pub fn add_file(&mut self, path: &str, bytes: Vec<u8>) {
        self.fs.files.insert(vfs_key(path), bytes);
    }

    /// Read back a file's current bytes (a write target after the run, for tests).
    pub fn file_bytes(&self, path: &str) -> Option<&[u8]> {
        self.fs.files.get(&vfs_key(path)).map(|v| v.as_slice())
    }

    /// sceIoOpen: returns a new fd or a negative errno.
    pub fn io_open(&mut self, path: &str, flags: u32) -> i32 {
        let fd = self.fs.open(path, flags);
        if std::env::var("VITASLOP_TRACE_IO").is_ok() {
            eprintln!("[io] open({path:?}, flags={flags:#x}) -> {fd}");
        }
        fd
    }

    /// sceIoRead: read up to `len` bytes; None on a bad/unreadable fd.
    pub fn io_read(&mut self, fd: i32, len: usize) -> Option<Vec<u8>> {
        self.fs.read(fd, len)
    }

    /// sceIoPread: positioned read at `offset` that leaves the cursor untouched.
    pub fn io_pread(&mut self, fd: i32, offset: u64, len: usize) -> Option<Vec<u8>> {
        self.fs.pread(fd, offset, len)
    }

    /// sceIoPwrite: positioned write at `offset` that leaves the cursor untouched.
    pub fn io_pwrite(&mut self, fd: i32, offset: u64, bytes: &[u8]) -> Option<usize> {
        self.fs.pwrite(fd, offset, bytes)
    }

    /// sceIoWrite: fd 1/2 go to the captured console; other fds to the vfs.
    /// Returns the byte count written (always the full length for the console).
    pub fn io_write(&mut self, fd: i32, bytes: &[u8]) -> Option<usize> {
        match fd {
            FD_STDOUT => {
                self.capture.stdout.extend_from_slice(bytes);
                Some(bytes.len())
            }
            FD_STDERR => {
                self.capture.stderr.extend_from_slice(bytes);
                Some(bytes.len())
            }
            _ => self.fs.write(fd, bytes),
        }
    }

    /// sceIoLseek/sceIoLseek32: new absolute position or a negative errno.
    pub fn io_lseek(&mut self, fd: i32, offset: i64, whence: i32) -> i64 {
        self.fs.lseek(fd, offset, whence)
    }

    /// sceIoClose: 0 or a negative errno.
    pub fn io_close(&mut self, fd: i32) -> i32 {
        self.fs.close(fd)
    }

    /// File size for sceIoGetstat, or None if the path does not exist.
    pub fn io_size(&self, path: &str) -> Option<u64> {
        self.fs.size_of(path)
    }

    /// Enable preemptive multithreading: blocking primitives now actually park the
    /// calling thread instead of succeeding uncontended. Set once, before the run,
    /// by the [`ThreadedScheduler`](vitaslop_native::ThreadedScheduler); the single-
    /// worker hosts leave it off and keep the run-to-completion behavior.
    /// (Scheduler: `vitaslop_native::ThreadedScheduler`.)
    pub fn set_preemptive(&mut self, on: bool) {
        self.preemptive = on;
    }

    /// True when running under the preemptive scheduler.
    pub fn is_preemptive(&self) -> bool {
        self.preemptive
    }

    /// The id of the thread currently executing a host call (preemptive mode). Set
    /// by the scheduler before each dispatch via `ImportDispatch::set_current_thread`.
    pub fn current_thread(&self) -> i32 {
        self.current
    }

    /// Record which thread is running (scheduler hook).
    pub fn set_current(&mut self, thid: i32) {
        self.current = thid;
    }

    /// Take the threads a host call asked to start (scheduler hook).
    pub fn take_spawns(&mut self) -> Vec<Reentry> {
        std::mem::take(&mut self.pending_spawns)
    }

    /// Take the parked threads a host call just made runnable (scheduler hook).
    pub fn take_wakes(&mut self) -> Vec<i32> {
        std::mem::take(&mut self.pending_wakes)
    }

    /// Create a thread: allocate its own stack and record it, returning its
    /// SceUID. The entry's Thumb bit is cleared so it names the transpiled export.
    pub fn create_thread(&mut self, entry: u32, stack_size: u32, priority: i32) -> i32 {
        let size = stack_size.max(0x1000);
        let stack = self.galloc(size, 16);
        // The stack grows down from an 8-byte-aligned top (AAPCS at a public call).
        let stack_top = stack.wrapping_add(size) & !0xF;
        let uid = self.next_uid;
        self.next_uid += 1;
        self.threads.push(ThreadRec { uid, entry: entry & !1, stack_top, exit_code: None, priority });
        uid
    }

    /// The priority the current thread is running at (lower = higher priority).
    pub fn current_priority(&self) -> i32 {
        self.threads
            .iter()
            .find(|t| t.uid == self.current)
            .map_or(DEFAULT_THREAD_PRIORITY, |t| t.priority)
    }

    /// Start a thread. In the single-thread model this raises a *synchronous*
    /// re-entry (the engine host runs the entry to completion before the guest sees
    /// anything past the start call). Under the preemptive scheduler it instead
    /// queues a *spawn*: the entry becomes its own concurrent fiber, and the
    /// starting thread keeps running.
    /// Returns true if the started thread outranks the current one (lower priority
    /// number), so the caller should reschedule at once - the real kernel preempts
    /// the starter and runs the higher-priority thread until it blocks. This is the
    /// ordering guarantee titles rely on when a worker must initialize (e.g. create
    /// and store a semaphore the starter then waits on) before the starter proceeds.
    pub fn start_thread(&mut self, thid: i32, arg_len: u32, arg_ptr: u32) -> bool {
        let Some(t) = self.threads.iter().find(|t| t.uid == thid) else { return false };
        let new_priority = t.priority;
        let req = Reentry {
            entry: t.entry,
            arg_len,
            arg_ptr,
            stack_top: t.stack_top,
            thid,
            priority: new_priority,
        };
        if self.preemptive {
            self.pending_spawns.push(req);
            new_priority < self.current_priority()
        } else {
            self.pending_reentry = Some(req);
            false
        }
    }

    /// The exit code of a finished thread, if it has run.
    pub fn thread_exit_code(&self, thid: i32) -> Option<u32> {
        self.threads.iter().find(|t| t.uid == thid).and_then(|t| t.exit_code)
    }

    /// Whether the thread with id `thid` has finished.
    pub fn thread_finished(&self, thid: i32) -> bool {
        self.threads.iter().any(|t| t.uid == thid && t.exit_code.is_some())
    }

    /// Take the pending thread-run request, if any (drained by the engine host).
    pub fn take_reentry(&mut self) -> Option<Reentry> {
        self.pending_reentry.take()
    }

    /// Record a finished thread's return value (set by the engine host after the
    /// thread ends). In preemptive mode this also wakes any thread parked joining
    /// it (`sceKernelWaitThreadEnd`).
    pub fn set_thread_exit(&mut self, thid: i32, code: u32) {
        if let Some(t) = self.threads.iter_mut().find(|t| t.uid == thid) {
            t.exit_code = Some(code);
        }
        if self.preemptive {
            let mut i = 0;
            while i < self.join_waiters.len() {
                if self.join_waiters[i].1 == thid {
                    let (waiter, _, stat) = self.join_waiters.remove(i);
                    // Deliver the exit code to the joiner's `stat` out-parameter now
                    // that it is known; the wait handler cannot write it at wake time.
                    if std::env::var("VITASLOP_TRACE_EXIT").is_ok() {
                        eprintln!("[join] target {thid:#x} exited code={code:#x}; waking {waiter:#x}, stat={stat:#x}");
                    }
                    if stat != 0 {
                        self.pending_stat_writes.push((stat, code));
                    }
                    self.pending_wakes.push(waiter);
                } else {
                    i += 1;
                }
            }
        }
    }

    /// Park the current thread joining `target`, unless it has already finished.
    /// Returns true if `target` is already done (the caller continues) or false if
    /// the caller was parked (return [`SvcOutcome::Block`]). `stat` is the joiner's
    /// `int *` out-parameter (0 = NULL), written with the target's exit code when the
    /// join completes (at wake time, via [`take_stat_writes`](Self::take_stat_writes)).
    pub fn join_block(&mut self, target: i32, stat: u32) -> bool {
        if self.thread_finished(target) {
            true
        } else {
            self.join_waiters.push((self.current, target, stat));
            false
        }
    }

    /// Take the queued joiner `stat` writes (`(stat_ptr, exit_code)`) so the scheduler
    /// can apply them to guest memory before the woken joiners resume. Drained after
    /// each dispatch alongside spawns and wakes.
    pub fn take_stat_writes(&mut self) -> Vec<(u32, u32)> {
        std::mem::take(&mut self.pending_stat_writes)
    }

    /// Whether a semaphore with this uid currently exists.
    pub fn sema_exists(&self, uid: i32) -> bool {
        self.semaphores.iter().any(|(u, _)| *u == uid)
    }

    /// Try to take `need` from semaphore `uid` without blocking. Returns true if
    /// the count was available (and consumed), false otherwise.
    pub fn sema_try_acquire(&mut self, uid: i32, need: i32) -> bool {
        if let Some((_, c)) = self.semaphores.iter_mut().find(|(u, _)| *u == uid) {
            if *c >= need {
                *c -= need;
                return true;
            }
        }
        false
    }

    /// Park the current thread waiting for `need` signals on semaphore `uid`.
    pub fn sema_block(&mut self, uid: i32, need: i32) {
        self.sema_waiters.push(SemaWaiter { uid, thid: self.current, need });
    }

    /// Signal semaphore `uid` by `n`, then release every parked waiter the new
    /// count can satisfy (in FIFO order), consuming the count for each. Preemptive
    /// counterpart of [`sema_signal`](Self::sema_signal).
    pub fn sema_signal_wake(&mut self, uid: i32, n: i32) {
        self.sema_signal(uid, n);
        loop {
            let count = self
                .semaphores
                .iter()
                .find(|(u, _)| *u == uid)
                .map(|(_, c)| *c)
                .unwrap_or(0);
            // The first waiter on this semaphore whose need the count can meet.
            let next = self
                .sema_waiters
                .iter()
                .position(|w| w.uid == uid && w.need <= count);
            let Some(idx) = next else { break };
            let w = self.sema_waiters.remove(idx);
            self.sema_signal(uid, -w.need); // consume the count for the woken waiter
            self.pending_wakes.push(w.thid);
        }
    }

    /// Create a recursive mutex, recording its state (preemptive ownership
    /// tracking), and return its SceUID.
    pub fn create_mutex(&mut self) -> i32 {
        let uid = self.new_uid();
        self.mutexes.push(MutexRec { uid, owner: None, count: 0, waiters: Vec::new() });
        uid
    }

    /// Lock mutex `uid` for the current thread. Returns true if acquired (free, or
    /// already held by this thread - recursive), false if the caller was parked
    /// behind the current owner (return [`SvcOutcome::Block`]).
    pub fn mutex_lock(&mut self, uid: i32) -> bool {
        let cur = self.current;
        if let Some(m) = self.mutexes.iter_mut().find(|m| m.uid == uid) {
            match m.owner {
                None => {
                    m.owner = Some(cur);
                    m.count = 1;
                    true
                }
                Some(o) if o == cur => {
                    m.count += 1;
                    true
                }
                Some(_) => {
                    m.waiters.push(cur);
                    false
                }
            }
        } else {
            // Unknown mutex: treat as uncontended success.
            true
        }
    }

    /// Whether locking mutex `uid` right now would contend (another thread owns
    /// it). Used by `sceKernelTryLockMutex`, which fails rather than blocks.
    pub fn mutex_contended(&self, uid: i32) -> bool {
        let cur = self.current;
        self.mutexes
            .iter()
            .find(|m| m.uid == uid)
            .map(|m| matches!(m.owner, Some(o) if o != cur))
            .unwrap_or(false)
    }

    /// Unlock mutex `uid`. When the recursion count reaches zero, hand ownership to
    /// the next parked waiter (FIFO) and wake it.
    pub fn mutex_unlock(&mut self, uid: i32) {
        let mut wake = None;
        if let Some(m) = self.mutexes.iter_mut().find(|m| m.uid == uid) {
            if m.count > 0 {
                m.count -= 1;
            }
            if m.count == 0 {
                if m.waiters.is_empty() {
                    m.owner = None;
                } else {
                    let next = m.waiters.remove(0);
                    m.owner = Some(next);
                    m.count = 1;
                    wake = Some(next);
                }
            }
        }
        if let Some(next) = wake {
            self.pending_wakes.push(next);
        }
    }

    /// Acquire mutex `uid` on behalf of thread `thid` (not the current thread).
    /// Used when a condition-variable signal transfers a waiter to its mutex: if
    /// the mutex is free the thread takes it and is woken now; otherwise it joins
    /// the mutex wait queue and is woken when the owner unlocks. An unknown mutex
    /// just wakes the thread.
    fn mutex_acquire_for(&mut self, uid: i32, thid: i32) {
        let mut wake = true;
        if let Some(m) = self.mutexes.iter_mut().find(|m| m.uid == uid) {
            match m.owner {
                None => {
                    m.owner = Some(thid);
                    m.count = 1;
                }
                Some(o) if o == thid => m.count += 1,
                Some(_) => {
                    m.waiters.push(thid);
                    wake = false; // woken later, when the owner unlocks
                }
            }
        }
        if wake {
            self.pending_wakes.push(thid);
        }
    }

    // --- condition variables (preemptive mode) ---

    /// Create a condition variable bound to mutex `mutex_uid`.
    pub fn create_cond(&mut self, mutex_uid: i32) -> i32 {
        let uid = self.new_uid();
        self.conds.push(CondRec { uid, mutex: mutex_uid, waiters: Vec::new() });
        uid
    }

    /// `sceKernelWaitCond`: release the associated mutex (handing it to any waiter)
    /// and park the current thread on the condition variable. The caller returns
    /// [`SvcOutcome::Block`]; on wake it has re-acquired the mutex (see
    /// [`cond_signal`](Self::cond_signal)) and the wait returns 0.
    pub fn cond_wait(&mut self, uid: i32) {
        let Some(mutex) = self.conds.iter().find(|c| c.uid == uid).map(|c| c.mutex) else {
            return;
        };
        self.mutex_unlock(mutex);
        let cur = self.current;
        if let Some(c) = self.conds.iter_mut().find(|c| c.uid == uid) {
            c.waiters.push(cur);
        }
    }

    /// `sceKernelSignalCond`/`SignalCondAll`: wake one (or all) parked waiter. Each
    /// woken thread must re-acquire the condition's mutex before it runs, so it is
    /// handed to the mutex (taken now if free, else queued behind the owner).
    pub fn cond_signal(&mut self, uid: i32, all: bool) {
        let (mutex, woken) = {
            let Some(c) = self.conds.iter_mut().find(|c| c.uid == uid) else {
                return;
            };
            let woken: Vec<i32> = if all {
                std::mem::take(&mut c.waiters)
            } else if c.waiters.is_empty() {
                Vec::new()
            } else {
                vec![c.waiters.remove(0)]
            };
            (c.mutex, woken)
        };
        for thid in woken {
            self.mutex_acquire_for(mutex, thid);
        }
    }

    // --- lightweight condition variables + virtual clock -------------------
    //
    // Lightweight conds (`sceKernelWaitLwCond`/`SignalLwCond`) are identified by a
    // guest work pointer, not a kernel handle, so they park by work address here
    // rather than through `CondRec`. A parked wait yields the fiber (the caller
    // returns `Block`), which lets the scheduler run the thread that will set the
    // awaited condition and signal it - the whole point, since a non-yielding wait
    // starves the producer and busy-spins.

    /// The virtual monotonic clock in microseconds.
    pub fn now_us(&self) -> u64 {
        self.virtual_us
    }

    /// Park the current thread in `sceKernelWaitLwCond` on cond `work`. `timeout_us`
    /// of 0 is an infinite wait (only a signal wakes it); non-zero sets a deadline
    /// at `now + timeout_us` so the scheduler can time it out even if no signal comes.
    pub fn lwcond_wait(&mut self, work: u32, timeout_us: u32) {
        let deadline = (timeout_us != 0).then(|| self.virtual_us + timeout_us as u64);
        self.lwcond_waiters.push((self.current, work, deadline));
    }

    /// `sceKernelSignalLwCond`/`SignalLwCondAll`: wake one (or all) threads parked on
    /// cond `work`, moving them onto the scheduler's wake list.
    pub fn lwcond_signal(&mut self, work: u32, all: bool) {
        let mut woke_one = false;
        self.lwcond_waiters.retain(|&(thid, w, _)| {
            if w == work && (all || !woke_one) {
                self.pending_wakes.push(thid);
                woke_one = true;
                false
            } else {
                true
            }
        });
    }

    /// Park the current thread until `now + us` on the virtual clock, woken only by
    /// time. Used for `sceKernelDelayThread` and `sceAudioOutOutput` grain pacing.
    pub fn sleep_park(&mut self, us: u64) {
        self.sleep_waiters.push((self.current, self.virtual_us.wrapping_add(us)));
    }

    /// The earliest pending timed-wake deadline across lightweight-cond waits and
    /// pure sleeps. The scheduler uses this to jump the clock forward when every
    /// thread is blocked, instead of declaring a deadlock.
    pub fn earliest_lwcond_deadline(&self) -> Option<u64> {
        let lw = self.lwcond_waiters.iter().filter_map(|&(_, _, d)| d);
        let sl = self.sleep_waiters.iter().map(|&(_, d)| d);
        lw.chain(sl).min()
    }

    /// Advance the virtual clock to at least `to_us` and wake every timed wait -
    /// lightweight cond or pure sleep - whose deadline has now passed. Called by the
    /// scheduler when no thread is runnable but a timed wait can still fire.
    pub fn advance_time_to(&mut self, to_us: u64) {
        self.virtual_us = self.virtual_us.max(to_us);
        let now = self.virtual_us;
        self.lwcond_waiters.retain(|&(thid, _, deadline)| match deadline {
            Some(d) if d <= now => {
                self.pending_wakes.push(thid);
                false
            }
            _ => true,
        });
        self.sleep_waiters.retain(|&(thid, deadline)| {
            if deadline <= now {
                self.pending_wakes.push(thid);
                false
            } else {
                true
            }
        });
    }

    /// Mint a fresh SceUID (for a mutex, semaphore, event flag, ...).
    pub fn new_uid(&mut self) -> i32 {
        let uid = self.next_uid;
        self.next_uid += 1;
        uid
    }

    /// Create a semaphore with an initial count, returning its SceUID.
    pub fn create_sema(&mut self, init: i32) -> i32 {
        let uid = self.new_uid();
        self.semaphores.push((uid, init));
        uid
    }

    /// Wait on a semaphore: take `n` from its count (never blocks in the single-
    /// thread model; the count floors at 0).
    pub fn sema_wait(&mut self, uid: i32, n: i32) {
        if let Some((_, c)) = self.semaphores.iter_mut().find(|(u, _)| *u == uid) {
            *c = (*c - n).max(0);
        }
    }

    /// Signal a semaphore: add `n` to its count.
    pub fn sema_signal(&mut self, uid: i32, n: i32) {
        if let Some((_, c)) = self.semaphores.iter_mut().find(|(u, _)| *u == uid) {
            *c += n;
        }
    }

    /// Create an event flag with an initial bit pattern, returning its SceUID.
    pub fn create_event_flag(&mut self, init: u32) -> i32 {
        let uid = self.new_uid();
        self.event_flags.push((uid, init));
        uid
    }

    /// Set (OR in) bits on an event flag.
    pub fn event_set(&mut self, uid: i32, bits: u32) {
        if let Some((_, p)) = self.event_flags.iter_mut().find(|(u, _)| *u == uid) {
            *p |= bits;
        }
    }

    /// Clear an event flag's bits: keep only the bits also set in `bits` (the
    /// sceKernelClearEventFlag semantics, `pattern &= bits`).
    pub fn event_clear(&mut self, uid: i32, bits: u32) {
        if let Some((_, p)) = self.event_flags.iter_mut().find(|(u, _)| *u == uid) {
            *p &= bits;
        }
    }

    /// The current bit pattern of an event flag.
    pub fn event_pattern(&self, uid: i32) -> u32 {
        self.event_flags.iter().find(|(u, _)| *u == uid).map(|(_, p)| *p).unwrap_or(0)
    }

    /// Allocate `size` bytes of real guest memory aligned to `align`, returning
    /// the guest address. Deterministic: a pure function of the allocation order.
    pub fn galloc(&mut self, size: u32, align: u32) -> u32 {
        let a = align.max(4);
        self.alloc_cursor = (self.alloc_cursor + a - 1) & !(a - 1);
        let p = self.alloc_cursor;
        self.alloc_cursor = self.alloc_cursor.wrapping_add(size.max(4));
        p
    }

    /// Mint a fresh opaque GXM handle.
    pub fn new_handle(&mut self) -> u32 {
        let h = self.next_handle;
        self.next_handle += 1;
        h
    }

    /// Allocate a memory block of `size` and record it, returning its SceUID.
    pub fn alloc_memblock(&mut self, size: u32, align: u32) -> i32 {
        let base = self.galloc(size, align);
        let uid = self.next_uid;
        self.next_uid += 1;
        self.memblocks.push(MemBlock { uid, base, size });
        uid
    }

    /// The base address of the block with SceUID `uid`, if known.
    pub fn memblock_base(&self, uid: i32) -> Option<u32> {
        self.memblocks.iter().find(|b| b.uid == uid).map(|b| b.base)
    }

    /// Record a vertex program's attribute layout and stream stride.
    pub fn set_vertex_program(
        &mut self,
        handle: u32,
        attributes: Vec<crate::capture::VertexAttribute>,
        stride: u32,
    ) {
        self.vertex_programs.push((handle, VertexProgramInfo { attributes, stride }));
    }

    /// Record a color surface, keyed by its guest struct address.
    pub fn set_color_surface(&mut self, addr: u32, surface: crate::capture::ColorSurface) {
        self.color_surfaces.push((addr, surface));
    }

    // --- scene assembly (used by the gxm handlers) ---

    pub fn begin_scene(&mut self, color_surface_addr: u32) {
        let color = self
            .color_surfaces
            .iter()
            .find(|(a, _)| *a == color_surface_addr)
            .map(|(_, s)| *s);
        self.scene = Some(crate::capture::Scene { color, draws: Vec::new() });
        self.pending_uniforms.clear();
    }

    pub fn bind_vertex_program(&mut self, handle: u32) {
        self.bound_vertex_program = handle;
    }

    pub fn bind_stream0(&mut self, addr: u32) {
        self.bound_stream0 = addr;
    }

    /// Record `sceGxmSetFragmentTexture(unit, texture)`: remember which guest
    /// `SceGxmTexture*` is bound to sampler `unit`. A zero address unbinds it.
    pub fn bind_fragment_texture(&mut self, unit: u32, texture_addr: u32) {
        self.bound_textures.retain(|(u, _)| *u != unit);
        if texture_addr != 0 {
            self.bound_textures.push((unit, texture_addr));
        }
    }

    /// Record the exact `SceGxmTextureFormat` set on a `SceGxmTexture*` (by
    /// `sceGxmTextureInit*`/`SetFormat`), so a later decode recovers the exact
    /// channel swizzle rather than the lossy 3-bit control-word field.
    pub fn set_texture_format(&mut self, texture_addr: u32, format: u32) {
        self.texture_formats.retain(|(a, _)| *a != texture_addr);
        self.texture_formats.push((texture_addr, format));
    }

    pub fn texture_format(&self, texture_addr: u32) -> Option<u32> {
        self.texture_formats
            .iter()
            .find(|(a, _)| *a == texture_addr)
            .map(|(_, f)| *f)
    }

    pub fn set_uniforms(&mut self, values: Vec<f32>) {
        self.pending_uniforms = values;
    }

    /// The bound vertex program's layout, if recorded.
    fn bound_layout(&self) -> Option<&VertexProgramInfo> {
        self.vertex_programs
            .iter()
            .find(|(h, _)| *h == self.bound_vertex_program)
            .map(|(_, info)| info)
    }

    /// Append a draw to the current scene, snapshotting vertex and index bytes.
    pub fn record_draw(
        &mut self,
        ctx: &GuestCtx,
        primitive: u32,
        index_format: u32,
        index_addr: u32,
        index_count: u32,
    ) {
        let (attributes, stride) = match self.bound_layout() {
            Some(info) => (info.attributes.clone(), info.stride),
            None => (Vec::new(), 0),
        };
        // Index element size: U16 (0) is 2 bytes, U32 is 4.
        let index_elem = if index_format == 0 { 2 } else { 4 };
        let indices = ctx.read_bytes(index_addr, index_count as usize * index_elem);
        // Highest index referenced, so we snapshot exactly the vertices used.
        let max_index = indices
            .chunks(index_elem)
            .map(|c| match index_elem {
                2 => u16::from_le_bytes([c[0], c[1]]) as u32,
                _ => u32::from_le_bytes([c[0], c[1], c[2], c[3]]),
            })
            .max()
            .unwrap_or(0);
        let vertex_bytes = if stride > 0 {
            (max_index + 1) * stride
        } else {
            0
        };
        let vertices = ctx.read_bytes(self.bound_stream0, vertex_bytes as usize);
        // Snapshot every bound fragment texture (decoded from its control words),
        // sorted by unit so unit 0 is first.
        let mut units: Vec<(u32, u32)> = self.bound_textures.clone();
        units.sort_by_key(|(u, _)| *u);
        let textures: Vec<crate::capture::BoundTexture> = units
            .iter()
            .filter_map(|&(unit, addr)| decode_texture(ctx, unit, addr, self.texture_format(addr)))
            .collect();
        let draw = crate::capture::Draw {
            primitive,
            index_format,
            index_count,
            vertices,
            vertex_stride: stride,
            attributes,
            indices,
            uniforms: self.pending_uniforms.clone(),
            textures,
        };
        if let Some(scene) = self.scene.as_mut() {
            scene.draws.push(draw);
        }
    }

    pub fn end_scene(&mut self) {
        if let Some(scene) = self.scene.take() {
            self.capture.scenes.push(scene);
        }
    }

    pub fn present(&mut self, buffer_addr: u32) {
        self.capture.presents.push(buffer_addr);
    }

    /// Append bytes the guest printed to the debug console (sceClibPrintf).
    pub fn write_stdout(&mut self, bytes: &[u8]) {
        self.capture.stdout.extend_from_slice(bytes);
    }
}

/// Round `v` up to the next multiple of `align` (a power of two).
fn align_up(v: u32, align: u32) -> u32 {
    (v + align - 1) & !(align - 1)
}

/// Decode a bound `SceGxmTexture` (16 bytes, 4 control words) from guest memory
/// and snapshot its pixel bytes. Returns `None` for a null/unreadable handle or a
/// format whose byte size we do not know yet. The layout is the public GXM texture
/// control-word format (vitasdk `gxm.h` `struct SceGxmTexture`): word 1 holds
/// width/height (stored as size-1) and the base format, word 2 the data address.
fn decode_texture(
    ctx: &GuestCtx,
    unit: u32,
    addr: u32,
    exact_format: Option<u32>,
) -> Option<crate::capture::BoundTexture> {
    if addr == 0 {
        return None;
    }
    let w0 = ctx.read_u32(addr);
    let w1 = ctx.read_u32(addr + 4);
    let w2 = ctx.read_u32(addr + 8);
    let w3 = ctx.read_u32(addr + 12);

    let tex_type = (w1 >> 29) & 0x7;
    // generic2 layout (non-swizzled/non-cube): width/height are 12-bit size-1.
    let width = ((w1 >> 12) & 0xfff) + 1;
    let height = (w1 & 0xfff) + 1;
    let data_addr = w2 & 0xffff_fffc;
    // Prefer the exact format the guest set (full high byte keeps 24-bit/paletted
    // formats and the channel swizzle intact); otherwise reconstruct the high byte
    // from the 5-bit field plus the format0 extension bit (word 0 bit 31), and take
    // the 3-bit control-word swizzle.
    let (base_format, swizzle) = match exact_format {
        Some(f) => ((f >> 24) & 0xff, f & 0x00ff_ffff),
        None => (((w1 >> 24) & 0x1f) | (((w0 >> 31) & 1) << 7), (w3 >> 29) & 0x7),
    };

    // Block geometry: uncompressed formats are 1x1 texel "blocks"; BC/DXT are 4x4.
    let (block_w, block_h, block_bytes) = crate::render::block_layout(base_format)?;
    let blocks_x = width.div_ceil(block_w);
    let blocks_y = height.div_ceil(block_h);
    // `stride` is the bytes per block-row we snapshot, and `total` the bytes to read.
    // A SWIZZLED (Morton) texture is stored over a power-of-two-padded block grid; a
    // LINEAR one is row-major, with uncompressed rows padded to a multiple of 8
    // texels (the GXM linear alignment) and compressed rows block-packed.
    let (stride, total) = if crate::render::swizzled_type(tex_type) {
        let padded_x = blocks_x.next_power_of_two();
        let padded_y = blocks_y.next_power_of_two();
        (padded_x * block_bytes, padded_x * padded_y * block_bytes)
    } else {
        let row_blocks = if block_w == 1 { align_up(width, 8) } else { blocks_x };
        (row_blocks * block_bytes, row_blocks * block_bytes * blocks_y)
    };
    let pixels = ctx.read_bytes(data_addr, total as usize);
    if pixels.is_empty() {
        return None;
    }
    Some(crate::capture::BoundTexture {
        unit,
        base_format,
        swizzle,
        tex_type,
        width,
        height,
        stride,
        data_addr,
        pixels,
    })
}

/// The Vita host environment: the NID import table plus per-run state. Implements
/// [`ImportDispatch`] so an engine host (native wasmtime or the browser) can
/// route the `env.import` trap here.
pub struct VitaEnv {
    /// (library_nid, func_nid) per dense import index, in loader order.
    imports: Vec<(u32, u32)>,
    pub state: VitaState,
}

impl VitaEnv {
    /// Build an environment for a module whose imports are `(library_nid,
    /// func_nid)` in dense index order, over guest memory `[base, base+mem_bytes)`.
    pub fn new(
        imports: Vec<(u32, u32)>,
        base: u32,
        mem_bytes: u32,
        world: Box<dyn World + Send>,
    ) -> Self {
        VitaEnv { imports, state: VitaState::new(base, mem_bytes, world) }
    }

    /// Convenience constructor with the default deterministic world.
    pub fn with_default_world(imports: Vec<(u32, u32)>, base: u32, mem_bytes: u32) -> Self {
        VitaEnv::new(imports, base, mem_bytes, Box::new(DeterministicWorld::default()))
    }
}

/// The seam an engine host implements to deliver an ARM `svc` trap to a host
/// convention (e.g. the Linux-EABI the ARM conformance corpus uses). The parallel
/// of [`ImportDispatch`] for the `env.svc` import instead of `env.import`. The
/// host passes the `svc` immediate, the register file, and rebased guest memory;
/// any register the handler changes is written back.
pub trait SvcDispatch {
    fn svc(
        &mut self,
        imm: u32,
        regs: &mut [u32; REG_COUNT],
        mem: &mut dyn GuestMemory,
        base: u32,
    ) -> SvcOutcome;
}

/// Drive a `svc` handler behind a shared handle, so a caller can keep a clone to
/// read accumulated state (e.g. captured output) back after the run.
impl<T: SvcDispatch> SvcDispatch for std::rc::Rc<std::cell::RefCell<T>> {
    fn svc(
        &mut self,
        imm: u32,
        regs: &mut [u32; REG_COUNT],
        mem: &mut dyn GuestMemory,
        base: u32,
    ) -> SvcOutcome {
        self.borrow_mut().svc(imm, regs, mem, base)
    }
}

/// The seam an engine host implements to deliver a NID import trap to the Vita
/// host. Engine-agnostic: the host passes the raw register file and rebased guest
/// memory, and any register the handler changes is written back.
pub trait ImportDispatch {
    fn dispatch(
        &mut self,
        index: u32,
        regs: &mut [u32; REG_COUNT],
        vfp: &mut [u32; VFP_ARG_COUNT],
        mem: &mut dyn GuestMemory,
        base: u32,
    ) -> SvcOutcome;

    /// Take a synchronous guest re-entry the last dispatch raised (a thread
    /// start), if any. The engine host calls this after each dispatch and runs
    /// the returned entry, then reports the result via [`set_thread_exit`]. A host
    /// with no engine to re-enter (default) never has one.
    ///
    /// [`set_thread_exit`]: ImportDispatch::set_thread_exit
    fn take_reentry(&mut self) -> Option<Reentry> {
        None
    }

    /// Record a re-entered thread's return value.
    fn set_thread_exit(&mut self, _thid: i32, _code: u32) {}

    // --- preemptive scheduler hooks (default no-ops) -----------------------
    //
    // The single-worker `Vm`/`Scheduler` never call these; only the preemptive
    // `ThreadedScheduler` does. They let a shared host and the scheduler agree on
    // which thread is running, which new threads to spawn as their own fibers, and
    // which parked threads a signal just made runnable - the state the scheduler
    // cannot see (it lives in the host's sync objects) and the host cannot act on
    // (only the scheduler owns the fibers).

    /// Tell the host which guest thread is about to run a host call, so a blocking
    /// primitive knows whom to park and a wake knows whom to release. The scheduler
    /// sets this before each dispatch.
    fn set_current_thread(&mut self, _thid: i32) {}

    /// Take the threads the last dispatch asked to *start* (each becomes its own
    /// fiber sharing the guest address space), if any. The preemptive counterpart
    /// of [`take_reentry`](Self::take_reentry): instead of running the entry
    /// synchronously to completion, the scheduler creates a concurrent thread.
    fn take_spawns(&mut self) -> Vec<Reentry> {
        Vec::new()
    }

    /// Take the parked threads the last dispatch just made runnable (a signal, an
    /// unlock, an event-flag set that satisfied a waiter). The scheduler drains
    /// this after each dispatch and moves those threads back onto the run queue;
    /// each resumes inside its blocking call, which then returns success.
    fn take_wakes(&mut self) -> Vec<i32> {
        Vec::new()
    }

    /// Take the guest memory writes (`(addr, value)`) the last dispatch queued for
    /// woken joiners - the exit code owed to a `sceKernelWaitThreadEnd` `stat`
    /// out-parameter. The scheduler applies these to guest memory before the woken
    /// joiner resumes (the wait handler cannot write them at wake time).
    fn take_stat_writes(&mut self) -> Vec<(u32, u32)> {
        Vec::new()
    }

    /// The earliest pending timed-wait deadline (virtual microseconds), if a thread
    /// is parked on a timed wait. When no thread is runnable, the scheduler jumps the
    /// clock to this instead of declaring a deadlock (a busy loop's timed wait).
    fn earliest_deadline(&self) -> Option<u64> {
        None
    }

    /// Advance the virtual clock to `to_us`, waking any timed waits that expire; the
    /// woken thread ids then surface through [`take_wakes`](Self::take_wakes).
    fn advance_time_to(&mut self, _to_us: u64) {}

    /// A display frame just flipped (`frame` flips so far). Lets a frame-keyed input
    /// source - a scripted TAS recipe - advance in lockstep with the render loop.
    /// The default ignores it.
    fn on_frame_boundary(&mut self, _frame: u64) {}
}

impl ImportDispatch for VitaEnv {
    fn dispatch(
        &mut self,
        index: u32,
        regs: &mut [u32; REG_COUNT],
        vfp: &mut [u32; VFP_ARG_COUNT],
        mem: &mut dyn GuestMemory,
        base: u32,
    ) -> SvcOutcome {
        self.state.capture.call_count += 1;
        let (library_nid, func_nid) = self
            .imports
            .get(index as usize)
            .copied()
            .unwrap_or((0, 0));
        self.state.capture.trace.push(func_nid);
        self.state.capture.trace_thid.push(self.state.current);
        let mut ctx = GuestCtx::new(regs, vfp, mem, base);
        vita::dispatch(library_nid, func_nid, &mut ctx, &mut self.state)
    }

    fn take_reentry(&mut self) -> Option<Reentry> {
        self.state.take_reentry()
    }

    fn set_thread_exit(&mut self, thid: i32, code: u32) {
        self.state.set_thread_exit(thid, code);
    }

    fn set_current_thread(&mut self, thid: i32) {
        self.state.set_current(thid);
    }

    fn take_spawns(&mut self) -> Vec<Reentry> {
        self.state.take_spawns()
    }

    fn take_wakes(&mut self) -> Vec<i32> {
        self.state.take_wakes()
    }

    fn take_stat_writes(&mut self) -> Vec<(u32, u32)> {
        self.state.take_stat_writes()
    }

    fn earliest_deadline(&self) -> Option<u64> {
        self.state.earliest_lwcond_deadline()
    }

    fn advance_time_to(&mut self, to_us: u64) {
        self.state.advance_time_to(to_us);
    }

    fn on_frame_boundary(&mut self, frame: u64) {
        self.state.world.set_frame(frame);
        // Advance the virtual clock by one full 60 Hz frame per display flip: a
        // rendered frame represents ~16.6 ms of wall time passing, so the game's
        // animation and dialog timers progress at the right rate. `advance_time_to`
        // wakes every timed wait (cond timeout, sleep, audio grain park) whose
        // deadline falls within the frame, so those still fire - just at frame
        // granularity. This advance is essential: while a title renders continuously
        // its main thread never blocks, so the scheduler's "nothing runnable -> jump
        // the clock" idle path never runs and only this drives wall time.
        //
        // It must NOT be clamped down to the earliest pending deadline: a thread that
        // perpetually re-waits with a near-zero timeout (a 1 us cond poll) would then
        // pin the per-frame advance to ~1 us and freeze the game clock, so every
        // time-based state (a dialog's fade-in / input-enable timer) would never
        // elapse and the title would sit frozen and input-inert.
        const FRAME_US: u64 = 1_000_000 / 60;
        let target = self.state.now_us() + FRAME_US;
        self.state.advance_time_to(target);
    }
}

/// A shared handle to a `VitaEnv`: attach one clone as the engine's import
/// environment and keep another to read the capture back after the run. Single
/// threaded (the guest CPU worker), so `Rc`/`RefCell` are the right tools.
impl ImportDispatch for std::rc::Rc<std::cell::RefCell<VitaEnv>> {
    fn dispatch(
        &mut self,
        index: u32,
        regs: &mut [u32; REG_COUNT],
        vfp: &mut [u32; VFP_ARG_COUNT],
        mem: &mut dyn GuestMemory,
        base: u32,
    ) -> SvcOutcome {
        self.borrow_mut().dispatch(index, regs, vfp, mem, base)
    }

    fn take_reentry(&mut self) -> Option<Reentry> {
        self.borrow_mut().take_reentry()
    }

    fn set_thread_exit(&mut self, thid: i32, code: u32) {
        self.borrow_mut().set_thread_exit(thid, code);
    }

    fn set_current_thread(&mut self, thid: i32) {
        self.borrow_mut().set_current_thread(thid);
    }

    fn take_spawns(&mut self) -> Vec<Reentry> {
        self.borrow_mut().take_spawns()
    }

    fn take_wakes(&mut self) -> Vec<i32> {
        self.borrow_mut().take_wakes()
    }

    fn take_stat_writes(&mut self) -> Vec<(u32, u32)> {
        self.borrow_mut().take_stat_writes()
    }

    fn earliest_deadline(&self) -> Option<u64> {
        self.borrow().earliest_deadline()
    }

    fn advance_time_to(&mut self, to_us: u64) {
        self.borrow_mut().advance_time_to(to_us);
    }
}

#[cfg(test)]
mod hostcall_tests {
    //! Verify the `#[hostcall]` marshalling directly (the cube exercises only
    //! integer/pointer handlers, so these cover the hardfloat classification and
    //! the float returns end to end against a hand-built `GuestCtx`).
    use super::*;
    use crate::hostcall;
    use crate::world::DeterministicWorld;

    fn dummy_state() -> VitaState {
        VitaState::new(0, 64, Box::new(DeterministicWorld::default()))
    }

    // Mixed integer and float args with a float return: `a` and `b` come from the
    // core registers (r0, r1), `x` and `y` from the VFP file (s0 and d1), and the
    // f32 result goes to s0.
    #[hostcall]
    fn mix(st: &mut VitaState, a: u32, x: f32, b: i32, y: f64) -> f32 {
        a as f32 + x + b as f32 + y as f32 + st.base as f32
    }

    // A pure float handler: double the d0 argument back into d0.
    #[hostcall]
    fn dbl(v: f64) -> f64 {
        v * 2.0
    }

    // A pointer out-param handler: write the sum of two ints through the pointer.
    #[hostcall]
    fn sum_out(ctx: &mut GuestCtx, a: u32, b: u32, out: Ptr) -> i32 {
        ctx.write_u32(out.addr(), a.wrapping_add(b));
        0
    }

    #[test]
    fn classifies_int_and_float_args_and_returns_float() {
        let mut regs = [0u32; REG_COUNT];
        let mut vfp = [0u32; VFP_ARG_COUNT];
        let mut bytes = vec![0u8; 64];
        regs[0] = 5; // a
        regs[1] = (-3i32) as u32; // b
        vfp[0] = 2.5f32.to_bits(); // x -> s0
        let y = 1.5f64.to_bits(); // y -> d1 (s2 low, s3 high)
        vfp[2] = y as u32;
        vfp[3] = (y >> 32) as u32;

        let mut st = dummy_state();
        let mut mem = SliceMemory(&mut bytes);
        {
            let mut ctx = GuestCtx::new(&mut regs, &mut vfp, &mut mem, 0);
            mix(&mut ctx, &mut st);
        }
        // 5 + 2.5 + (-3) + 1.5 + 0 = 6.0, in s0.
        assert_eq!(f32::from_bits(vfp[0]), 6.0);
    }

    #[test]
    fn marshals_double_arg_and_return() {
        let mut regs = [0u32; REG_COUNT];
        let mut vfp = [0u32; VFP_ARG_COUNT];
        let mut bytes = vec![0u8; 8];
        let v = 3.0f64.to_bits(); // d0 = s0 (low), s1 (high)
        vfp[0] = v as u32;
        vfp[1] = (v >> 32) as u32;

        let mut st = dummy_state();
        let mut mem = SliceMemory(&mut bytes);
        {
            let mut ctx = GuestCtx::new(&mut regs, &mut vfp, &mut mem, 0);
            dbl(&mut ctx, &mut st);
        }
        let out = (vfp[0] as u64) | ((vfp[1] as u64) << 32);
        assert_eq!(f64::from_bits(out), 6.0);
    }

    #[test]
    fn writes_through_pointer_out_param() {
        let mut regs = [0u32; REG_COUNT];
        let mut vfp = [0u32; VFP_ARG_COUNT];
        let mut bytes = vec![0u8; 64];
        regs[0] = 7; // a
        regs[1] = 35; // b
        regs[2] = 16; // out pointer (guest addr, base 0 -> offset 16)

        let mut st = dummy_state();
        let mut mem = SliceMemory(&mut bytes);
        {
            let mut ctx = GuestCtx::new(&mut regs, &mut vfp, &mut mem, 0);
            sum_out(&mut ctx, &mut st);
        }
        assert_eq!(u32::from_le_bytes([bytes[16], bytes[17], bytes[18], bytes[19]]), 42);
    }
}

#[cfg(test)]
mod preemptive_tests {
    //! The preemptive block/wake bookkeeping in `VitaState`, exercised directly
    //! (no wasm engine): a parked thread and its release for each primitive. The
    //! engine-level proof that these outcomes drive real fibers lives in
    //! `vitaslop-native/tests/threaded_scheduler.rs`.
    use super::*;
    use crate::world::DeterministicWorld;

    fn state() -> VitaState {
        let mut st = VitaState::new(0x1000, 0x10000, Box::new(DeterministicWorld::default()));
        st.set_preemptive(true);
        st
    }

    #[test]
    fn semaphore_parks_then_a_signal_wakes_and_consumes() {
        let mut st = state();
        let sem = st.create_sema(0);
        // Thread 1 wants 1 but the count is 0: it cannot acquire and parks.
        st.set_current(1);
        assert!(!st.sema_try_acquire(sem, 1));
        st.sema_block(sem, 1);
        assert!(st.take_wakes().is_empty());
        // Thread 2 signals: thread 1 is released and the token is consumed for it.
        st.set_current(2);
        st.sema_signal_wake(sem, 1);
        assert_eq!(st.take_wakes(), vec![1]);
        // The count was consumed by the wake, so a fresh acquire would block again.
        assert!(!st.sema_try_acquire(sem, 1));
    }

    #[test]
    fn mutex_hands_ownership_to_the_next_waiter_on_unlock() {
        let mut st = state();
        let m = st.create_mutex();
        // Thread 1 acquires; thread 2 contends and parks.
        st.set_current(1);
        assert!(st.mutex_lock(m));
        st.set_current(2);
        assert!(st.mutex_contended(m));
        assert!(!st.mutex_lock(m));
        assert!(st.take_wakes().is_empty());
        // Thread 1 unlocks: ownership passes to thread 2, which is woken.
        st.set_current(1);
        st.mutex_unlock(m);
        assert_eq!(st.take_wakes(), vec![2]);
        // Thread 2 now owns it, so thread 3 would contend.
        st.set_current(3);
        assert!(st.mutex_contended(m));
    }

    #[test]
    fn mutex_is_recursive_for_the_owner() {
        let mut st = state();
        let m = st.create_mutex();
        st.set_current(1);
        assert!(st.mutex_lock(m)); // count 1
        assert!(st.mutex_lock(m)); // count 2 (recursive, same owner)
        st.mutex_unlock(m); // count 1, still owned
        st.set_current(2);
        assert!(st.mutex_contended(m), "still held after one of two unlocks");
        st.set_current(1);
        st.mutex_unlock(m); // count 0, released
        st.set_current(2);
        assert!(!st.mutex_contended(m), "free after the matching unlock");
    }

    #[test]
    fn cond_wait_releases_the_mutex_and_signal_hands_it_back() {
        let mut st = state();
        let m = st.create_mutex();
        let cv = st.create_cond(m);

        // Thread 1 holds the mutex, then waits on the condition: the wait releases
        // the mutex (so another thread can take it) and parks thread 1.
        st.set_current(1);
        assert!(st.mutex_lock(m));
        st.cond_wait(cv);
        assert!(st.take_wakes().is_empty(), "waiter is parked, not runnable");
        st.set_current(2);
        assert!(!st.mutex_contended(m), "mutex was released by the wait");

        // Thread 2 takes the mutex, then signals: the waiter cannot run yet (thread
        // 2 still owns the mutex), so it is queued behind the owner, not woken.
        assert!(st.mutex_lock(m));
        st.cond_signal(cv, false);
        assert!(st.take_wakes().is_empty(), "waiter must re-acquire the mutex first");

        // Thread 2 unlocks: ownership passes to the waiter, which is finally woken.
        st.mutex_unlock(m);
        assert_eq!(st.take_wakes(), vec![1]);
        // Thread 1 now owns the mutex again (the wait re-acquired it).
        st.set_current(3);
        assert!(st.mutex_contended(m));
    }

    #[test]
    fn cond_signal_with_free_mutex_wakes_immediately() {
        let mut st = state();
        let m = st.create_mutex();
        let cv = st.create_cond(m);
        st.set_current(1);
        assert!(st.mutex_lock(m));
        st.cond_wait(cv); // releases m, parks thread 1
        // The signaller does not hold the mutex, so the woken waiter takes it now.
        st.set_current(2);
        st.cond_signal(cv, false);
        assert_eq!(st.take_wakes(), vec![1]);
        st.set_current(3);
        assert!(st.mutex_contended(m), "waiter re-acquired the free mutex");
    }

    #[test]
    fn cond_signal_all_wakes_every_waiter() {
        let mut st = state();
        let m = st.create_mutex();
        let cv = st.create_cond(m);
        // Two threads wait (neither holds the mutex at wait's end - a plain park).
        for t in [1, 2] {
            st.set_current(t);
            st.mutex_lock(m);
            st.cond_wait(cv);
            st.mutex_unlock(m); // no-op: wait already released it
        }
        st.set_current(3);
        st.cond_signal(cv, true);
        // All waiters are released from the condition, but the mutex serializes
        // them: only the first can hold it and wake now; the second queues behind
        // it and is woken when the first unlocks.
        assert_eq!(st.take_wakes(), vec![1]);
        st.set_current(1);
        st.mutex_unlock(m);
        assert_eq!(st.take_wakes(), vec![2]);
    }

    #[test]
    fn join_parks_until_the_target_thread_exits() {
        let mut st = state();
        let worker = st.create_thread(0x2000, 0x1000, DEFAULT_THREAD_PRIORITY);
        // Main (thread 0) joins the not-yet-finished worker and parks, passing a
        // `stat` out-parameter at guest address 0x5000.
        st.set_current(0);
        assert!(!st.join_block(worker, 0x5000));
        assert!(st.take_wakes().is_empty());
        assert!(st.take_stat_writes().is_empty(), "no exit code to deliver yet");
        // The worker ends: the joiner is woken, the exit code is recorded, AND queued
        // for delivery to the joiner's `stat` pointer (the wait handler cannot write
        // it at wake time).
        st.set_thread_exit(worker, 7);
        assert_eq!(st.take_wakes(), vec![0]);
        assert_eq!(st.thread_exit_code(worker), Some(7));
        assert_eq!(st.take_stat_writes(), vec![(0x5000, 7)]);
        // A join after the fact does not park (and a NULL stat queues no write).
        assert!(st.join_block(worker, 0));
        assert!(st.take_stat_writes().is_empty());
    }

    #[test]
    fn start_thread_queues_a_spawn_not_a_synchronous_reentry() {
        let mut st = state();
        let worker = st.create_thread(0x2000, 0x1000, DEFAULT_THREAD_PRIORITY);
        st.start_thread(worker, 4, 0x1234);
        assert!(st.take_reentry().is_none(), "preemptive start does not re-enter synchronously");
        let spawns = st.take_spawns();
        assert_eq!(spawns.len(), 1);
        assert_eq!(spawns[0].thid, worker);
        assert_eq!(spawns[0].entry, 0x2000);
        assert_eq!(spawns[0].arg_len, 4);
        assert_eq!(spawns[0].arg_ptr, 0x1234);
    }
}

#[cfg(test)]
mod filesystem_tests {
    //! The SceIoFilemgr virtual filesystem in `VitaState`, exercised directly via
    //! the public io_* methods. The end-to-end proof on a real velf is
    //! `vitaslop-conformance-harness/tests/vita_io.rs`; these cover the paths that
    //! artifact does not (SEEK_CUR/END, append, truncate, bad fd, stderr).
    use super::*;
    use crate::world::DeterministicWorld;

    fn state() -> VitaState {
        VitaState::new(0, 64, Box::new(DeterministicWorld::default()))
    }

    #[test]
    fn write_then_read_back_via_the_vfs() {
        let mut st = state();
        let fd = st.io_open("ux0:/a", SCE_O_WRONLY | SCE_O_CREAT);
        assert!(fd >= 3);
        assert_eq!(st.io_write(fd, b"abcdef"), Some(6));
        assert_eq!(st.io_close(fd), 0);
        assert_eq!(st.io_size("ux0:/a"), Some(6));

        let r = st.io_open("ux0:/a", SCE_O_RDONLY);
        assert_eq!(st.io_lseek(r, 2, SCE_SEEK_SET), 2);
        assert_eq!(st.io_read(r, 100).as_deref(), Some(&b"cdef"[..]));
        // At EOF a further read is empty, not an error.
        assert_eq!(st.io_read(r, 100).as_deref(), Some(&b""[..]));
        st.io_close(r);
    }

    #[test]
    fn seek_cur_end_and_negative_is_error() {
        let mut st = state();
        st.add_file("app0:/f", b"0123456789".to_vec());
        let fd = st.io_open("app0:/f", SCE_O_RDONLY);
        assert_eq!(st.io_lseek(fd, 3, SCE_SEEK_SET), 3);
        assert_eq!(st.io_lseek(fd, 2, SCE_SEEK_CUR), 5);
        assert_eq!(st.io_lseek(fd, -1, SCE_SEEK_END), 9);
        assert_eq!(st.io_read(fd, 4).as_deref(), Some(&b"9"[..]));
        // Seeking before the start is a negative errno, leaving the cursor put.
        assert!(st.io_lseek(fd, -100, SCE_SEEK_SET) < 0);
    }

    #[test]
    fn append_seeks_to_end_and_truncate_clears() {
        let mut st = state();
        st.add_file("ux0:/log", b"head".to_vec());
        let a = st.io_open("ux0:/log", SCE_O_WRONLY | SCE_O_APPEND);
        assert_eq!(st.io_write(a, b"-tail"), Some(5));
        st.io_close(a);
        assert_eq!(st.file_bytes("ux0:/log"), Some(&b"head-tail"[..]));

        // Opening with TRUNC discards the old contents.
        let t = st.io_open("ux0:/log", SCE_O_WRONLY | SCE_O_TRUNC);
        assert_eq!(st.io_write(t, b"new"), Some(3));
        st.io_close(t);
        assert_eq!(st.file_bytes("ux0:/log"), Some(&b"new"[..]));
    }

    #[test]
    fn missing_file_and_bad_fd_are_errors() {
        let mut st = state();
        assert!(st.io_open("app0:/nope", SCE_O_RDONLY) < 0);
        assert!(st.io_read(999, 4).is_none());
        assert!(st.io_write(999, b"x").is_none());
        assert!(st.io_lseek(999, 0, SCE_SEEK_SET) < 0);
        assert!(st.io_close(999) < 0);
    }

    #[test]
    fn fd_one_and_two_route_to_captured_console() {
        let mut st = state();
        assert_eq!(st.io_write(FD_STDOUT, b"out"), Some(3));
        assert_eq!(st.io_write(FD_STDERR, b"err"), Some(3));
        assert_eq!(st.capture.stdout, b"out");
        assert_eq!(st.capture.stderr, b"err");
    }

    #[test]
    fn pread_is_positioned_and_leaves_the_cursor_put() {
        // sceIoPread reads at an absolute offset without disturbing the descriptor
        // cursor - the AT9 music streamer relies on this to pull the header and then
        // interleave chunk reads on one shared fd.
        let mut st = state();
        st.add_file("app0:/song.at9", b"RIFF....fmt ....data".to_vec());
        let fd = st.io_open("app0:/song.at9", SCE_O_RDONLY);
        // A normal read advances the cursor to 4.
        assert_eq!(st.io_read(fd, 4).as_deref(), Some(&b"RIFF"[..]));
        // pread at offset 8 does not move the cursor...
        assert_eq!(st.io_pread(fd, 8, 4).as_deref(), Some(&b"fmt "[..]));
        // ...so the next sequential read continues from 4.
        assert_eq!(st.io_read(fd, 4).as_deref(), Some(&b"...."[..]));
        // Reading past the end clamps to the available bytes.
        assert_eq!(st.io_pread(fd, 16, 100).as_deref(), Some(&b"data"[..]));
        // A bad fd yields None (mapped to EBADF by the handler).
        assert!(st.io_pread(999, 0, 4).is_none());
    }

    #[test]
    fn pwrite_is_positioned_and_extends_with_zeros() {
        let mut st = state();
        st.add_file("ux0:/save.bin", vec![0u8; 4]);
        let fd = st.io_open("ux0:/save.bin", SCE_O_RDWR);
        // Positioned write past the current end zero-fills the gap.
        assert_eq!(st.io_pwrite(fd, 6, b"AB"), Some(2));
        assert_eq!(st.file_bytes("ux0:/save.bin"), Some(&[0, 0, 0, 0, 0, 0, b'A', b'B'][..]));
        // The cursor is untouched, so a sequential read still starts at 0.
        assert_eq!(st.io_read(fd, 2).as_deref(), Some(&[0, 0][..]));
    }
}
