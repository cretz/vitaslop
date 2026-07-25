//! The host-call boundary: how a guest NID import trap becomes a typed Rust
//! handler. `GuestCtx` marshals AAPCS arguments (r0..r3 then stack) and guest
//! memory in and out; `VitaEnv` owns the per-run state (allocator, handles,
//! capture, world) and routes a dense import index to a per-module handler.
//! See `projects/vitaslop-runtime/README.md`.

use std::collections::HashMap;
use std::sync::Arc;

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

    /// Read a little-endian u16 at guest address `addr` (0 if out of range).
    pub fn read_u16(&self, addr: u32) -> u16 {
        match self.offset(addr) {
            Some(o) if o + 2 <= self.mem.len() => {
                let mut b = [0u8; 2];
                self.mem.read(o, &mut b);
                u16::from_le_bytes(b)
            }
            _ => 0,
        }
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

/// `SCE_KERNEL_DEFAULT_PRIORITY` - the *sentinel* a title passes to
/// `sceKernelCreateThread` to mean "the default priority", and the base for
/// relative priorities (`base + delta`). It is NOT an actual runnable priority;
/// [`resolve_priority`] maps it (and the relative range around it) to a concrete
/// value before it ever reaches the scheduler.
pub const SCE_KERNEL_DEFAULT_PRIORITY: i32 = 0x1000_0100;

/// The concrete default user-thread priority (`SCE_KERNEL_DEFAULT_PRIORITY_USER`,
/// 0xA0). Absolute user priorities run 0x40 (highest) .. 0xBF (lowest); 0xA0 is the
/// middle default the initial (main) thread and a defaulted worker resolve to.
pub const DEFAULT_THREAD_PRIORITY: i32 = 0xA0;

/// `SCE_KERNEL_ERROR_WAIT_TIMEOUT` - the value a *timed* blocking wait
/// (`sceKernelWaitSema`/`WaitCond`/`WaitLwCond`/`WaitEventFlag` with a non-null
/// timeout) returns when its deadline passes before the wait is satisfied. Delivered
/// to the woken thread's `r0` through the resume-code channel (see
/// [`VitaState::take_resume_code`]); a wait satisfied by a signal returns 0 instead.
pub const SCE_KERNEL_ERROR_WAIT_TIMEOUT: u32 = 0x8002_8005;

/// Resolve a `sceKernelCreateThread` priority argument to a concrete scheduler
/// priority. Absolute user priorities (small numbers, ~0x40..0xBF) pass through;
/// the relative range around [`SCE_KERNEL_DEFAULT_PRIORITY`] (e.g. the sentinel
/// itself, or `default - 1`) is rebased onto [`DEFAULT_THREAD_PRIORITY`]. Without
/// this the sentinel (a huge number) would read as the lowest possible priority,
/// so any concrete-priority worker would outrank - and starve - the main thread.
pub fn resolve_priority(prio: i32) -> i32 {
    if prio >= 0x1000_0000 {
        DEFAULT_THREAD_PRIORITY + (prio - SCE_KERNEL_DEFAULT_PRIORITY)
    } else {
        prio
    }
}

/// A recursive mutex's state (preemptive mode only; the single-thread model needs
/// none). `owner` is the holding thread's id (None if free), `count` the recursion
/// depth, `waiters` the threads parked in `sceKernelLockMutex` in FIFO order.
struct MutexRec {
    uid: i32,
    owner: Option<i32>,
    count: i32,
    waiters: Vec<i32>,
}

/// A lightweight mutex's state (preemptive mode only), keyed by its guest work-area
/// address rather than a kernel handle - a lightweight object's state lives in memory
/// the title owns, not a kernel id. Otherwise identical to [`MutexRec`]: `owner` (None
/// if free), recursion `count`, and the FIFO `waiters` parked in `sceKernelLockLwMutex`.
struct LwMutexRec {
    work: u32,
    owner: Option<i32>,
    count: i32,
    waiters: Vec<i32>,
}

/// A condition variable's state (preemptive mode only). `mutex` is the associated
/// mutex it releases on wait and re-acquires on wake; `waiters` are the threads
/// parked in `sceKernelWaitCond` in FIFO order, each with an optional virtual-clock
/// `deadline` for a timed wait (`None` = wait forever). A condition variable has no
/// memory: a signal with no waiter is lost.
struct CondRec {
    uid: i32,
    mutex: i32,
    waiters: Vec<CondWaiter>,
}

/// A thread parked in `sceKernelWaitCond`: which thread and an optional virtual-clock
/// deadline (`Some` for a timed wait, woken by a signal or when the clock reaches it;
/// `None` for an infinite wait). A timed-out cond wait still re-acquires the mutex
/// before it resumes (see [`VitaState::advance_time_to`]).
struct CondWaiter {
    thid: i32,
    deadline: Option<u64>,
}

/// A thread parked in `sceKernelWaitSema`: which semaphore, which thread, how many
/// signals it still needs, and an optional virtual-clock `deadline` for a timed wait
/// (`None` = wait forever). It is released (and the count consumed) when a signal
/// makes `need` available, or woken with `SCE_KERNEL_ERROR_WAIT_TIMEOUT` when the
/// deadline passes.
struct SemaWaiter {
    uid: i32,
    thid: i32,
    need: i32,
    deadline: Option<u64>,
}

/// A thread parked in `sceKernelWaitEventFlag`: which flag, which thread, the bit
/// pattern and wait mode it asked for, the guest address of its `outBits`
/// out-parameter (0 = NULL), and an optional virtual-clock deadline for a timed
/// wait. Released when a `sceKernelSetEventFlag` satisfies the pattern (the match
/// pattern is then written through `out_addr` via the pending stat-write channel)
/// or the deadline passes.
struct EvfWaiter {
    uid: i32,
    thid: i32,
    bits: u32,
    mode: u32,
    out_addr: u32,
    deadline: Option<u64>,
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
    /// r2 for the entry. Zero for a `sceKernelStartThread` entry (whose ABI is only
    /// `(SceSize args, void *argp)`); nonzero only for a host-delivered callback
    /// whose ABI passes a third register (e.g. an NP service-state callback thunk
    /// that takes its `this`/userdata in r2).
    pub r2: u32,
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
    /// Original (as-added) spelling per lowercased key, so a directory listing can
    /// return real mixed-case names for a title's own glob matching. Lookup stays
    /// case-insensitive through the lowercased `files` key.
    originals: std::collections::HashMap<String, String>,
    open: std::collections::HashMap<i32, OpenFile>,
    open_dirs: std::collections::HashMap<i32, OpenDir>,
    next_fd: i32,
}

/// In-memory savedata slot store: the metadata layer SceAppUtil exposes on top of a
/// title's savedata mount. Real hardware persists each slot's SceAppUtilSaveDataSlotParam
/// (the localized title/subtitle/detail and bookkeeping) to the mounted savedata
/// partition; this keeps the raw param blob per (mount, slot) in memory so a slot the
/// title creates is visible to a later get - faithful read-after-write - without touching
/// a real disk. Deterministic and host-only, like [`FileTable`]. A future backend (native
/// disk, browser OPFS) can persist these blobs; the offline oracle keeps them in RAM.
#[derive(Default)]
pub struct SaveDataStore {
    /// (mount name, slot id) -> the exact SceAppUtilSaveDataSlotParam bytes the title
    /// wrote at create/set time, echoed back verbatim on get.
    slots: std::collections::HashMap<(String, u32), Vec<u8>>,
}

impl SaveDataStore {
    fn key(mount: &str, slot_id: u32) -> (String, u32) {
        (mount.to_string(), slot_id)
    }
    /// Whether a slot exists under this mount.
    pub fn contains(&self, mount: &str, slot_id: u32) -> bool {
        self.slots.contains_key(&Self::key(mount, slot_id))
    }
    /// Store (create or overwrite) a slot's param blob.
    pub fn put(&mut self, mount: &str, slot_id: u32, param: Vec<u8>) {
        self.slots.insert(Self::key(mount, slot_id), param);
    }
    /// The stored param blob for a slot, or `None` if it was never created.
    pub fn get(&self, mount: &str, slot_id: u32) -> Option<&[u8]> {
        self.slots.get(&Self::key(mount, slot_id)).map(Vec::as_slice)
    }
    /// Remove a slot; returns whether it existed.
    pub fn remove(&mut self, mount: &str, slot_id: u32) -> bool {
        self.slots.remove(&Self::key(mount, slot_id)).is_some()
    }
}

/// One entry a directory descriptor yields: the child's original-case name, whether
/// it is a subdirectory (synthesized from deeper paths in the flat map), and the
/// file size (0 for a directory).
#[derive(Clone, Debug, PartialEq, Eq)]
pub struct DirEntry {
    pub name: String,
    pub is_dir: bool,
    pub size: u64,
}

/// An open directory descriptor: the listing snapshotted at `sceIoDopen` time and
/// a read cursor advanced by each `sceIoDread`.
struct OpenDir {
    entries: Vec<DirEntry>,
    cursor: usize,
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
    // Normalize like a real Vita FS before matching: collapse repeated separators,
    // drop `.` and empty segments, and resolve `..`. Titles build paths by joining a
    // directory (often with a trailing `/`) to a subpath (often with a leading `/`),
    // yielding runs like `usrdir//ui/fonts//x.ttf`; an exact-match store would miss
    // those. Both the store side (`add_file`) and every lookup route through here, so
    // the normalization stays symmetric. Backslashes are folded to `/` for the rare
    // title that uses them. Lowercased last (see the case note above).
    let stripped = strip_app0(path).replace('\\', "/");
    let mut segs: Vec<&str> = Vec::new();
    for seg in stripped.split('/') {
        match seg {
            "" | "." => {}
            ".." => {
                segs.pop();
            }
            s => segs.push(s),
        }
    }
    segs.join("/").to_ascii_lowercase()
}

/// Strip an `app0:` mount prefix (with or without the separator), case-insensitively:
/// titles spell the mount both `app0:` and `APP0:`. The rest of the path keeps its
/// case - [`vfs_key`] lowercases for lookup, while the original spelling is preserved
/// for directory listings (a title pattern-matches `sceIoDread` names against
/// mixed-case globs like `Foo_*`).
fn strip_app0(path: &str) -> &str {
    let b = path.as_bytes();
    if b.len() >= 5 && b[..5].eq_ignore_ascii_case(b"app0:") {
        let rest = &path[5..];
        rest.strip_prefix('/').unwrap_or(rest)
    } else {
        path
    }
}

/// Whether a (already-lowercased [`vfs_key`]) path lives on a writable, persisted
/// Vita mount - `savedata0:` (the title's own save slot) or `ux0:` (the memory
/// card). A write-then-close on one of these is game-authored persistent state, so
/// the egress ledger records it. `app0:` (read-only game data) never qualifies, and
/// is stripped by `vfs_key` anyway.
fn is_persisted_mount(key: &str) -> bool {
    key.starts_with("savedata") || key.starts_with("ux0:")
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

    /// Open a directory listing (sceIoDopen): returns a new descriptor or a negative
    /// errno if nothing exists under the path.
    ///
    /// The vfs is a flat path->bytes map, so the listing is synthesized: every stored
    /// key under `path/` contributes its next path component, deduplicated - a
    /// component with deeper structure is a subdirectory, a terminal one a file.
    /// Names are returned in their original (as-added) case because titles glob
    /// listings against mixed-case patterns, while dedup/order use the lowercased
    /// form to match the case-insensitive lookup rules. The listing is snapshotted at
    /// open, ordered by lowercased name, and consumed by [`Self::dread`].
    fn dopen(&mut self, path: &str) -> i32 {
        let key = vfs_key(path);
        let key = key.trim_end_matches('/');
        let prefix = if key.is_empty() { String::new() } else { format!("{key}/") };
        // Child components sit at this index when a full key is split on '/'.
        let comp_index = prefix.matches('/').count();
        let mut children: std::collections::BTreeMap<String, DirEntry> =
            std::collections::BTreeMap::new();
        for (k, data) in &self.files {
            let Some(rest) = k.strip_prefix(&prefix) else { continue };
            if rest.is_empty() {
                continue;
            }
            let (comp, is_dir) = match rest.split_once('/') {
                Some((first, _)) => (first, true),
                None => (rest, false),
            };
            let entry = children.entry(comp.to_string()).or_insert_with(|| {
                // Recover the child's original spelling from the as-added path; the
                // lowercased and original paths split identically ('/' survives
                // ASCII lowercasing), so the component index lines up.
                let name = self
                    .originals
                    .get(k)
                    .and_then(|orig| orig.split('/').nth(comp_index))
                    .unwrap_or(comp)
                    .to_string();
                DirEntry { name, is_dir, size: 0 }
            });
            if !is_dir {
                entry.size = data.len() as u64;
            }
        }
        if children.is_empty() {
            return SCE_ERROR_ERRNO_ENOENT;
        }
        let fd = self.next_fd;
        self.next_fd += 1;
        self.open_dirs.insert(fd, OpenDir { entries: children.into_values().collect(), cursor: 0 });
        fd
    }

    /// Read the next entry from a directory descriptor (sceIoDread). Returns
    /// `None` on a bad descriptor, `Some(None)` when the listing is exhausted, and
    /// `Some(Some(entry))` otherwise (advancing the cursor).
    fn dread(&mut self, fd: i32) -> Option<Option<DirEntry>> {
        let od = self.open_dirs.get_mut(&fd)?;
        let entry = od.entries.get(od.cursor).cloned();
        if entry.is_some() {
            od.cursor += 1;
        }
        Some(entry)
    }

    /// Close a directory descriptor (sceIoDclose); returns 0 or a negative errno.
    fn dclose(&mut self, fd: i32) -> i32 {
        if self.open_dirs.remove(&fd).is_some() {
            0
        } else {
            SCE_ERROR_ERRNO_EBADF
        }
    }

    /// The size of `path` if it exists (for sceIoGetstat).
    fn size_of(&self, path: &str) -> Option<u64> {
        self.files.get(&vfs_key(path)).map(|d| d.len() as u64)
    }

    /// The whole contents of `path` if it exists, cloned. For consumers that need a
    /// file's bytes in one shot without managing a descriptor (e.g. loading a font).
    fn read_all(&self, path: &str) -> Option<Vec<u8>> {
        self.files.get(&vfs_key(path)).cloned()
    }

    /// The size of the file behind an open descriptor (for sceIoGetstatByFd).
    /// Returns None on a bad descriptor - the path is already the vfs key.
    fn size_of_fd(&self, fd: i32) -> Option<u64> {
        let of = self.open.get(&fd)?;
        self.files.get(&of.path).map(|d| d.len() as u64)
    }
}

/// Per-draw vertex program layout captured at create time, keyed by the vertex
/// program handle so a later Draw knows how to snapshot the vertex buffer.
struct VertexProgramInfo {
    attributes: Vec<crate::capture::VertexAttribute>,
    stride: u32,
    /// The `SceGxmProgram*` this vertex program was created from, so a precomputed
    /// vertex state (which references the vertex program) can size its default uniform
    /// buffer from the program header (+0x2C). 0 if it could not be resolved.
    program_header: u32,
}

/// A precomputed vertex- or fragment-state object (`sceGxmPrecomputed{Vertex,Fragment}
/// State*`): the default uniform buffer the guest writes its uniforms into, the bound
/// fragment/vertex textures, and the `SceGxmProgram*` (for uniform-buffer sizing).
/// Keyed by the guest state-struct address, applied to the live bind state when the
/// guest issues `sceGxmSetPrecomputed{Vertex,Fragment}State`, exactly as the individual
/// `sceGxmSetUniformDataF`/`sceGxmSetFragmentTexture` calls would be on the direct path.
#[derive(Clone, Default)]
struct PrecomputedState {
    program_header: u32,
    default_uniform_buffer: u32,
    /// (textureIndex, `SceGxmTexture*` addr).
    textures: Vec<(u32, u32)>,
}

/// Extra sticky per-texture state the `sceGxmTextureGet*` getters read back. The
/// hardware packs these into the 16-byte control words; we keep them in a shadow
/// (like `texture_samplers`) so a setter never risks corrupting a struct the guest
/// re-reads, and a getter returns exactly what the guest set. Absent = GXM defaults.
#[derive(Clone, Copy)]
struct TextureExtra {
    /// Mip-map level count (1-based), from the `mipCount` argument to a non-strided
    /// `sceGxmTextureInit*`. `SetData`/`SetFormat` leave it unchanged.
    mip_count: u32,
    /// Explicit byte stride for a `LINEAR_STRIDED` texture (0 = derive from width x
    /// bytes-per-pixel, as the driver does for every other layout).
    byte_stride: u32,
    /// Minification/magnification/mip filters (`SceGxmTextureFilter`).
    min_filter: u32,
    mag_filter: u32,
    mip_filter: u32,
    /// Gamma-correction mode (`SceGxmTextureGammaMode`, the raw enum word).
    gamma: u32,
}

impl Default for TextureExtra {
    fn default() -> Self {
        // GXM defaults: one mip level, driver-derived stride, POINT filtering, no gamma.
        TextureExtra { mip_count: 1, byte_stride: 0, min_filter: 0, mag_filter: 0, mip_filter: 0, gamma: 0 }
    }
}

/// A precomputed draw: the vertex program, stream-0 vertex buffer, and draw
/// parameters the guest bundled with `sceGxmPrecomputedDraw{Init,SetVertexStream,
/// SetParams}`, replayed as a normal draw when the guest issues
/// `sceGxmDrawPrecomputed`.
///
/// The state lives IN the guest block (see [`pdraw`]), not in a host-side table keyed
/// by the block's address: a `SceGxmPrecomputedDraw` is a plain 11-word POD the guest
/// owns, and a title is free to build one and then `memcpy` it - or an array of them -
/// to the addresses it actually draws from. An address-keyed table cannot follow such a
/// by-value copy and silently loses every draw made through the copy.
#[derive(Clone, Copy, Default)]
struct PrecomputedDraw {
    vertex_program: u32,
    stream0: u32,
    primitive: u32,
    index_format: u32,
    index_addr: u32,
    index_count: u32,
}

/// Byte layout of the guest `SceGxmPrecomputedDraw` work area. Eleven 32-bit words is
/// what `sceGxmGetPrecomputedDrawSize` reports and what the SDK struct declares
/// (`uint32_t reserved[11]`), and its contents are opaque to the guest, so the fields
/// below are ours to define. Word 0 is a tag: a block that does not carry it was never
/// initialised through our `Init` (or was copied from uninitialised memory), which is a
/// hard error rather than a silently skipped draw.
/// Per-scene texture-byte snapshots, keyed by (guest data address, byte length).
type TextureSnapshots = HashMap<(u32, usize), Arc<[u8]>>;

mod pdraw {
    /// "PDRW" - the initialised-block tag in word 0.
    pub const MAGIC: u32 = 0x5744_5250;
    pub const OFF_MAGIC: u32 = 0;
    pub const OFF_VERTEX_PROGRAM: u32 = 4;
    pub const OFF_STREAM0: u32 = 8;
    pub const OFF_PRIMITIVE: u32 = 12;
    pub const OFF_INDEX_FORMAT: u32 = 16;
    pub const OFF_INDEX_ADDR: u32 = 20;
    pub const OFF_INDEX_COUNT: u32 = 24;
    /// Words 7..10 are unused; kept so the block is exactly the reported size.
    pub const WORDS: u32 = 11;
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
    // The `SceGxmProgram*` of the fragment program bound by the last precomputed
    // fragment state, so a draw can reflect which sampler unit is the albedo (base
    // colour) and prefer it over a normal/spec/env map when picking the one texture
    // the capture renderer samples. 0 when no fragment program is bound.
    bound_fragment_program_header: u32,
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
    /// One bit per open SceCommonDialog family (see `vita::services::DialogFamily`):
    /// set by `*DialogInit`, read by `*DialogGetStatus` (open reports FINISHED -
    /// dialogs complete instantly offline), cleared by `*DialogTerm`.
    pub(crate) open_dialogs: u32,
    // Preemptive-mode state (unused in the single-thread model). When `preemptive`
    // is set, blocking primitives actually park the calling thread (`current`) and
    // are woken by another thread's signal/unlock/thread-end; the scheduler drains
    // `pending_spawns` (threads to start as their own fibers) and `pending_wakes`
    // (parked threads made runnable) after each host call. See the runtime README
    // concurrency model and `vitaslop_native::ThreadedScheduler`.
    preemptive: bool,
    current: i32,
    mutexes: Vec<MutexRec>,
    lwmutexes: Vec<LwMutexRec>,
    conds: Vec<CondRec>,
    sema_waiters: Vec<SemaWaiter>,
    evf_waiters: Vec<EvfWaiter>,
    // Display-queue callback execution (preemptive mode; see
    // [`Self::enqueue_display_callback`]): the dedicated guest stack and
    // callback-data slot ring (lazily allocated), entries waiting to run, and the
    // in-flight callback's thread id. The queue is serial - one callback at a
    // time, in submission order - matching the real display queue's single
    // internal thread.
    display_cb_stack: u32,
    display_cb_slots: u32,
    display_cb_next_slot: u32,
    display_cb_queue: std::collections::VecDeque<u32>,
    display_cb_running: Option<i32>,
    // Registered service-state callbacks (preemptive mode). A title registers
    // these at boot and pumps them every frame (`sce*CheckCallback`), waiting for
    // the one-time state notification (offline: signed-out / disconnected) before
    // it leaves its boot screen. Each is `Some((entry, userdata))` once registered
    // and `delivered` once its callback has been run. See
    // [`Self::pump_service_callbacks`].
    np_service_cb: Option<(u32, u32)>,
    np_cb_delivered: bool,
    net_inet_cb: Option<(u32, u32)>,
    net_cb_delivered: bool,
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
    /// The `r0` value to hand a parked thread when it resumes, keyed by thread id:
    /// `(thid, code)`. A blocking wait sets its return value (`ctx.ret(0)`) *before* it
    /// parks, so a wake that must return something other than 0 - a timed wait that
    /// expired, returning `SCE_KERNEL_ERROR_WAIT_TIMEOUT` - cannot write it then.
    /// Instead the timeout is queued here and the engine applies it to the woken
    /// thread's `r0` at the point it resumes (see the native/browser schedulers).
    /// Absent (the common case, a signal wake) leaves the pre-park `r0` (0) in place.
    pending_resume_codes: Vec<(i32, u32)>,
    // Virtual filesystem backing SceIoFilemgr (open/read/write/lseek/close).
    fs: FileTable,
    // In-memory savedata slot metadata (SceAppUtil), so a created slot round-trips on get.
    savedata: SaveDataStore,
    /// ScePvf font engine: open lib/font handles, size config, glyph cache. Public so
    /// the ScePvf NID handlers in `vita/pvf.rs` can drive it.
    pub fonts: crate::font::FontLibrary,
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
    /// The main executable's TLS template `(init image address, initialized byte
    /// count, full block size)`, from [`crate::link::LinkedProgram::tls_template`].
    /// Each thread gets a private block of `memsz` bytes; the compiler reaches
    /// `__thread` variables at `thread_pointer + offset`. `(0, 0, 0)` = no TLS.
    tls_template: (u32, u32, u32),
    /// Per-thread thread-pointer (`TPIDRURO`) block base, keyed by thread id, from
    /// [`Self::ensure_tls_block`]. The engine reads it via [`Self::thread_tls_base`]
    /// when it instantiates a thread and seeds the `tp` register from it.
    tls_bases: Vec<(i32, u32)>,
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
    /// Sampler wrap/LOD state per guest `SceGxmTexture*`, as `(addr, (u_addr_mode,
    /// v_addr_mode, lod_bias))`, set by `sceGxmTextureSet{U,V}AddrMode[Safe]` /
    /// `SetLodBias`. Kept beside `texture_formats` (rather than mutated into the
    /// guest control words) so recording the state never risks corrupting a struct
    /// the guest re-reads. Absent = GXM defaults (REPEAT/REPEAT/0).
    texture_samplers: Vec<(u32, (u32, u32, u32))>,
    /// Extra sticky per-texture state (`mipCount`, byte stride, min/mag/mip filter,
    /// gamma) the `sceGxmTextureGet*` getters read back, keyed by `SceGxmTexture*`.
    texture_extra: Vec<(u32, TextureExtra)>,
    /// Per-color-surface gamma-correction mode set by `sceGxmColorSurfaceSetGammaMode`,
    /// keyed by `SceGxmColorSurface*`. Absent = SCE_GXM_COLOR_SURFACE_GAMMA_NONE.
    color_surface_gamma: Vec<(u32, u32)>,
    /// Per-scene texture-byte snapshots, keyed by (guest data address, byte length), so a
    /// texture bound by hundreds of draws is read from guest memory once and shared. Cleared
    /// at `beginScene` - see the note in `decode_texture` for why that is the right scope.
    texture_snapshots: TextureSnapshots,
    /// Precomputed vertex/fragment states (`sceGxmPrecomputed{Vertex,Fragment}StateInit`
    /// + setters), keyed by the guest state-struct address, applied to the live bind
    /// state by `sceGxmSetPrecomputed{Vertex,Fragment}State`. A HashMap (not the Vec the
    /// other GXM tables use) because the bind lookup runs once per draw - thousands of
    /// times per frame - so the lookup must be O(1), not a linear scan over every state.
    precomputed_vertex_states: std::collections::HashMap<u32, PrecomputedState>,
    precomputed_fragment_states: std::collections::HashMap<u32, PrecomputedState>,
    /// `SceGxmFragmentProgram*` handle -> its `SceGxmProgram*`, recorded at
    /// `sceGxmShaderPatcherCreateFragmentProgram` so a precomputed fragment state can
    /// size its default uniform buffer. (Vertex programs carry this in `VertexProgramInfo`.)
    fragment_programs: Vec<(u32, u32)>,
    /// The vertex default uniform buffer bound for the next draw (guest ptr, byte size),
    /// from `sceGxmSetPrecomputedVertexState`. Read into the draw's uniforms at record
    /// time; 0 = fall back to the `sceGxmSetUniformDataF` path (`pending_uniforms`).
    bound_vertex_uniform_buf: u32,
    bound_vertex_uniform_size: u32,
    /// The FRAGMENT default uniform buffer bound for the next draw (guest ptr, byte size),
    /// from `sceGxmSetPrecomputedFragmentState` or `sceGxmReserveFragmentDefaultUniformBuffer`.
    /// Holds the per-material fragment uniforms - base-colour tint (`AlbedoColour`/
    /// `Primarytint`), the directional light direction/colour, fog colour - which the real
    /// fragment program multiplies the sampled albedo by. Read into the draw's material at
    /// record time so the capture renderer can reproduce the lit colour instead of the raw
    /// albedo texel (which, for e.g. a tyre whose albedo texture is near-white detail scaled
    /// by a dark `AlbedoColour`, is why unlit wheels rendered as white rings). 0 = unbound.
    bound_fragment_uniform_buf: u32,
    bound_fragment_uniform_size: u32,
    /// The GPU notification region: a guest buffer of `SCE_GXM_NOTIFICATION_COUNT`
    /// u32 slots handed out by `sceGxmGetNotificationRegion`, lazily allocated on
    /// first use (0 = not yet allocated). Scenes complete synchronously here, so a
    /// notification the guest waits on is treated as already signalled.
    notification_region: u32,
    /// The live GXM fixed-function pipeline state (cull/depth/stencil/viewport/...),
    /// mutated by the `sceGxmSet*` setters and snapshotted into each recorded draw.
    /// Sticky across scenes, exactly like the real GXM context.
    render_state: crate::capture::RenderState,
    /// Threads parked in `sceKernelWaitLwCond`, as `(thread id, cond work address,
    /// wake deadline)`. `deadline` is `Some(virtual_us)` for a timed wait (woken by
    /// a signal or when the clock reaches it) or `None` for an infinite wait (only a
    /// signal wakes it). Keyed by the cond's guest work pointer, since lightweight
    /// objects have no kernel handle. See the scheduler's idle-time advance.
    lwcond_waiters: Vec<(i32, u32, Option<u64>)>,
    /// Each lightweight condition variable's associated lightweight mutex, as `(cond
    /// work address, mutex work address)`, recorded at `sceKernelCreateLwCond` (which
    /// is handed the `SceKernelLwMutexWork*` it binds to). `sceKernelWaitLwCond`
    /// atomically releases this mutex as it parks and re-acquires it before the woken
    /// thread runs - exactly like the heavyweight cond/mutex pair. Without it a thread
    /// would hold the mutex across the wait and deadlock any thread that then locks it.
    lwcond_mutex: Vec<(u32, u32)>,
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
    /// The current display-frame index, updated each flip by the scheduler
    /// (`on_frame_boundary`). Frame-tags egress-ledger events so a recipe can assert
    /// roughly when a milestone occurred, not only that it did.
    cur_frame: u64,
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
            bound_fragment_program_header: 0,
            bound_stream0: 0,
            pending_uniforms: Vec::new(),
            threads: Vec::new(),
            pending_reentry: None,
            semaphores: Vec::new(),
            event_flags: Vec::new(),
            open_dialogs: 0,
            preemptive: false,
            current: 0,
            mutexes: Vec::new(),
            lwmutexes: Vec::new(),
            conds: Vec::new(),
            sema_waiters: Vec::new(),
            evf_waiters: Vec::new(),
            display_cb_stack: 0,
            display_cb_slots: 0,
            display_cb_next_slot: 0,
            display_cb_queue: std::collections::VecDeque::new(),
            display_cb_running: None,
            np_service_cb: None,
            np_cb_delivered: false,
            net_inet_cb: None,
            net_cb_delivered: false,
            join_waiters: Vec::new(),
            pending_spawns: Vec::new(),
            pending_wakes: Vec::new(),
            pending_stat_writes: Vec::new(),
            pending_resume_codes: Vec::new(),
            fs: FileTable::new(),
            savedata: SaveDataStore::default(),
            fonts: crate::font::FontLibrary::default(),
            capture: Capture::new(),
            world,
            audio: Box::new(crate::audio::NullSink::default()),
            audio_state: crate::vita::audio::AudioState::default(),
            halt_on_terminate: false,
            process_param: 0,
            tls_slots: Vec::new(),
            tls_template: (0, 0, 0),
            tls_bases: Vec::new(),
            shader_programs: Vec::new(),
            bound_textures: Vec::new(),
            texture_formats: Vec::new(),
            texture_samplers: Vec::new(),
            texture_extra: Vec::new(),
            color_surface_gamma: Vec::new(),
            texture_snapshots: TextureSnapshots::new(),
            precomputed_vertex_states: std::collections::HashMap::new(),
            precomputed_fragment_states: std::collections::HashMap::new(),
            fragment_programs: Vec::new(),
            bound_vertex_uniform_buf: 0,
            bound_vertex_uniform_size: 0,
            bound_fragment_uniform_buf: 0,
            bound_fragment_uniform_size: 0,
            notification_region: 0,
            render_state: crate::capture::RenderState::default(),
            lwcond_waiters: Vec::new(),
            lwcond_mutex: Vec::new(),
            sleep_waiters: Vec::new(),
            virtual_us: 0,
            cur_frame: 0,
        }
    }

    /// The current display-frame index (set by the scheduler each flip). Read by the
    /// egress ledger to frame-tag events and by an observation harness sampling per
    /// frame.
    pub fn cur_frame(&self) -> u64 {
        self.cur_frame
    }

    /// Set the current display-frame index. Called by the scheduler's frame boundary
    /// so egress events and per-frame samples carry the right frame number.
    pub fn set_cur_frame(&mut self, frame: u64) {
        self.cur_frame = frame;
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

    /// Record the main executable's TLS template (from
    /// [`crate::link::LinkedProgram::tls_template`]) so each thread can be given its
    /// own thread-local-storage block. Set once before the run.
    pub fn set_tls_template(&mut self, template: (u32, u32, u32)) {
        self.tls_template = template;
    }

    /// The thread pointer (`TPIDRURO`) for thread `thid`: the base of its private TLS
    /// block, allocating the block on first request. The block is `memsz` bytes; its
    /// `.tbss` tail is zero (guest memory starts zeroed) and its `.tdata` head is
    /// copied from the template by the engine at thread instantiation (it owns guest
    /// memory). Returns 0 when the title has no TLS - the compiler then never reads
    /// the thread pointer, so 0 is never dereferenced.
    pub fn ensure_tls_block(&mut self, thid: i32) -> u32 {
        let (_, _, memsz) = self.tls_template;
        if memsz == 0 {
            return 0;
        }
        if let Some(&(_, base)) = self.tls_bases.iter().find(|&&(t, _)| t == thid) {
            return base;
        }
        // 8-byte aligned so a `__thread` double or 64-bit value in the block is aligned.
        let base = self.galloc(memsz, 8);
        self.tls_bases.push((thid, base));
        base
    }

    /// The TLS init image `(source address, initialized byte count)` for the template,
    /// so the engine can copy `.tdata` into a freshly allocated block. `(_, 0)` when
    /// there is nothing to copy (a pure `.tbss` template stays zero).
    pub fn tls_init_image(&self) -> (u32, u32) {
        (self.tls_template.0, self.tls_template.1)
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
        let key = vfs_key(path);
        self.fs.originals.insert(key.clone(), strip_app0(path).trim_start_matches('/').to_string());
        self.fs.files.insert(key, bytes);
    }

    /// Read back a file's current bytes (a write target after the run, for tests).
    pub fn file_bytes(&self, path: &str) -> Option<&[u8]> {
        self.fs.files.get(&vfs_key(path)).map(|v| v.as_slice())
    }

    /// sceIoOpen: returns a new fd or a negative errno.
    pub fn io_open(&mut self, path: &str, flags: u32) -> i32 {
        let fd = self.fs.open(path, flags);
        tracing::trace!(target: "vitaslop::io", path, flags = format_args!("{flags:#x}"), fd, "open");
        fd
    }

    /// sceIoRead: read up to `len` bytes; None on a bad/unreadable fd.
    pub fn io_read(&mut self, fd: i32, len: usize) -> Option<Vec<u8>> {
        let r = self.fs.read(fd, len);
        let path = self.fs.open.get(&fd).map(|o| o.path.clone()).unwrap_or_default();
        tracing::trace!(target: "vitaslop::io", fd, len, path, got = r.as_ref().map(|d| d.len()).unwrap_or(0), "read");
        r
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
    ///
    /// Egress ledger: if the descriptor being closed was a writable file under a
    /// persisted mount (`savedata0:`/`ux0:`), record a [`SaveWrite`](crate::capture::EgressKind::SaveWrite)
    /// with the path, final size, and an ASCII preview - the title-agnostic,
    /// human-readable "the game persisted state" signal a conformance recipe asserts
    /// on. Close is the natural commit point (the whole file is present by then).
    pub fn io_close(&mut self, fd: i32) -> i32 {
        if let Some(of) = self.fs.open.get(&fd) {
            if of.writable && is_persisted_mount(&of.path) {
                let path = of.path.clone();
                if let Some(data) = self.fs.files.get(&path) {
                    let ev = crate::capture::EgressEvent {
                        frame: self.cur_frame,
                        kind: crate::capture::EgressKind::SaveWrite {
                            path,
                            bytes: data.len(),
                            ascii: crate::capture::ascii_preview(data, 96),
                        },
                    };
                    self.capture.egress.push(ev);
                }
            }
        }
        self.fs.close(fd)
    }

    /// sceIoDopen: open a directory listing; a new fd or a negative errno.
    pub fn io_dopen(&mut self, path: &str) -> i32 {
        let fd = self.fs.dopen(path);
        tracing::trace!(target: "vitaslop::io", path, fd, "dopen");
        fd
    }

    /// sceIoDread: the next entry of an open directory. `None` = bad descriptor,
    /// `Some(None)` = end of listing.
    pub fn io_dread(&mut self, fd: i32) -> Option<Option<DirEntry>> {
        let r = self.fs.dread(fd);
        let name = match &r { Some(Some(e)) => e.name.clone(), Some(None) => "<end>".into(), None => "<badfd>".into() };
        tracing::trace!(target: "vitaslop::io", fd, name, "dread");
        r
    }

    /// sceIoDclose: 0 or a negative errno.
    pub fn io_dclose(&mut self, fd: i32) -> i32 {
        self.fs.dclose(fd)
    }

    /// File size for sceIoGetstat, or None if the path does not exist.
    pub fn io_size(&self, path: &str) -> Option<u64> {
        let r = self.fs.size_of(path);
        tracing::trace!(target: "vitaslop::io", path, size = r.unwrap_or(u64::MAX), found = r.is_some(), "getstat");
        r
    }

    /// The size of the file behind an open descriptor (for sceIoGetstatByFd).
    pub fn io_size_fd(&self, fd: i32) -> Option<u64> {
        self.fs.size_of_fd(fd)
    }

    /// The whole contents of a guest file by path (for one-shot loads like a font
    /// file), or `None` if it does not exist.
    pub fn read_file(&self, path: &str) -> Option<Vec<u8>> {
        self.fs.read_all(path)
    }

    /// Whether a savedata slot exists (SceAppUtil metadata layer).
    pub fn savedata_slot_exists(&self, mount: &str, slot_id: u32) -> bool {
        self.savedata.contains(mount, slot_id)
    }

    /// Create or overwrite a savedata slot's param blob.
    pub fn savedata_slot_put(&mut self, mount: &str, slot_id: u32, param: Vec<u8>) {
        self.savedata.put(mount, slot_id, param);
    }

    /// The stored param blob for a savedata slot, cloned for the caller to write back
    /// into guest memory (`None` if the slot was never created).
    pub fn savedata_slot_get(&self, mount: &str, slot_id: u32) -> Option<Vec<u8>> {
        self.savedata.get(mount, slot_id).map(<[u8]>::to_vec)
    }

    /// Remove a savedata slot; returns whether it existed.
    pub fn savedata_slot_remove(&mut self, mount: &str, slot_id: u32) -> bool {
        self.savedata.remove(mount, slot_id)
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
        // Resolve the sentinel/relative priority to a concrete value now, so every
        // scheduler comparison against it (and against the main thread's default)
        // is meaningful.
        let priority = resolve_priority(priority);
        tracing::trace!(
            target: "vitaslop::thread",
            uid,
            entry = format_args!("{:#x}", entry & !1),
            priority = format_args!("{priority:#x}"),
            main_priority = format_args!("{DEFAULT_THREAD_PRIORITY:#x}"),
            "create"
        );
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
            r2: 0,
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
                    tracing::debug!(
                        target: "vitaslop::exit",
                        target_thid = format_args!("{thid:#x}"),
                        code = format_args!("{code:#x}"),
                        waking = format_args!("{waiter:#x}"),
                        stat = format_args!("{stat:#x}"),
                        "join delivered"
                    );
                    if stat != 0 {
                        self.pending_stat_writes.push((stat, code));
                    }
                    self.pending_wakes.push(waiter);
                } else {
                    i += 1;
                }
            }
            // A finished display callback unblocks the next queued one (the
            // display queue runs its callbacks serially, in submission order).
            if self.display_cb_running == Some(thid) {
                self.display_cb_running = None;
                self.pump_display_callback();
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
    /// `timeout_us` of 0 is an infinite wait; non-zero arms a deadline at
    /// `now + timeout_us`, after which the wait is woken with
    /// `SCE_KERNEL_ERROR_WAIT_TIMEOUT` even if no signal arrives.
    pub fn sema_block(&mut self, uid: i32, need: i32, timeout_us: u32) {
        let deadline = (timeout_us != 0).then(|| self.virtual_us + timeout_us as u64);
        self.sema_waiters.push(SemaWaiter { uid, thid: self.current, need, deadline });
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

    // --- lightweight mutexes (preemptive mode), keyed by guest work address -----
    //
    // A lightweight mutex has no kernel handle; its state lives in the caller's work
    // area, so the host tracks ownership by that guest address. These mirror the
    // heavyweight [`mutex_lock`]/[`mutex_contended`]/[`mutex_unlock`] exactly, so a
    // `sceKernelLockLwMutex` genuinely blocks on contention and enforces mutual
    // exclusion (the old "always succeed" stub did neither, so two threads could hold
    // the same lightweight mutex across a yield and race the data it guards).

    /// The record for the lightweight mutex at guest work address `work`, created
    /// (unlocked) on first use - a title may lock a work area it zero-initialized
    /// itself without a distinct create call.
    fn lwmutex_rec(&mut self, work: u32) -> &mut LwMutexRec {
        if !self.lwmutexes.iter().any(|m| m.work == work) {
            self.lwmutexes.push(LwMutexRec { work, owner: None, count: 0, waiters: Vec::new() });
        }
        self.lwmutexes.iter_mut().find(|m| m.work == work).expect("just inserted")
    }

    /// Register the lightweight mutex at `work` (its canonical work-area address) at
    /// `sceKernelCreateLwMutex`, so a later lock/unlock on a *copy* of the work area can
    /// be resolved back to it by the identity stamped in the work area (see the lwsync
    /// `resolve_mutex`). Idempotent - a re-create just keeps the existing record.
    pub fn lwmutex_register(&mut self, work: u32) {
        let _ = self.lwmutex_rec(work);
    }

    /// Whether `work` is a lightweight mutex we have a record for (created, or already
    /// locked at least once). Used to resolve a possibly-copied work pointer.
    pub fn lwmutex_is_known(&self, work: u32) -> bool {
        self.lwmutexes.iter().any(|m| m.work == work)
    }

    /// Lock the lightweight mutex at `work` for the current thread. Returns true if
    /// acquired (free, or already held by this thread - recursive), false if the caller
    /// was parked behind the owner (return [`SvcOutcome::Block`]).
    pub fn lwmutex_lock(&mut self, work: u32) -> bool {
        let cur = self.current;
        let m = self.lwmutex_rec(work);
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
    }

    /// Whether locking the lightweight mutex at `work` now would contend (another
    /// thread owns it). Used by `sceKernelTryLockLwMutex`, which fails rather than blocks.
    pub fn lwmutex_contended(&self, work: u32) -> bool {
        let cur = self.current;
        self.lwmutexes
            .iter()
            .find(|m| m.work == work)
            .map(|m| matches!(m.owner, Some(o) if o != cur))
            .unwrap_or(false)
    }

    /// Unlock the lightweight mutex at `work`. On full release, hand ownership to the
    /// next parked waiter (FIFO) and wake it.
    pub fn lwmutex_unlock(&mut self, work: u32) {
        let mut wake = None;
        if let Some(m) = self.lwmutexes.iter_mut().find(|m| m.work == work) {
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

    /// Forget the lightweight mutex at `work` (`sceKernelDeleteLwMutex`).
    pub fn lwmutex_delete(&mut self, work: u32) {
        self.lwmutexes.retain(|m| m.work != work);
    }

    /// Acquire the lightweight mutex at `work` on behalf of thread `thid` (not the
    /// current thread). Used when a lightweight-cond signal/timeout transfers a waiter
    /// back to its mutex: if free the thread takes it and is woken now, else it queues
    /// behind the owner and is woken when the owner unlocks. Work-keyed twin of
    /// [`mutex_acquire_for`](Self::mutex_acquire_for).
    fn lwmutex_acquire_for(&mut self, work: u32, thid: i32) {
        let mut wake = true;
        let m = self.lwmutex_rec(work);
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
    /// [`cond_signal`](Self::cond_signal)) and the wait returns 0. `timeout_us` of 0
    /// is an infinite wait; non-zero arms a deadline after which the wait times out
    /// (still re-acquiring the mutex) and returns `SCE_KERNEL_ERROR_WAIT_TIMEOUT`.
    pub fn cond_wait(&mut self, uid: i32, timeout_us: u32) {
        let Some(mutex) = self.conds.iter().find(|c| c.uid == uid).map(|c| c.mutex) else {
            return;
        };
        self.mutex_unlock(mutex);
        let cur = self.current;
        let deadline = (timeout_us != 0).then(|| self.virtual_us + timeout_us as u64);
        if let Some(c) = self.conds.iter_mut().find(|c| c.uid == uid) {
            c.waiters.push(CondWaiter { thid: cur, deadline });
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
                std::mem::take(&mut c.waiters).into_iter().map(|w| w.thid).collect()
            } else if c.waiters.is_empty() {
                Vec::new()
            } else {
                vec![c.waiters.remove(0).thid]
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

    /// Record a lightweight condition variable's associated lightweight mutex at
    /// `sceKernelCreateLwCond(cond_work, .., mutex_work, ..)`, so a later
    /// `sceKernelWaitLwCond` on `cond_work` knows which mutex to release and re-acquire.
    pub fn lwcond_bind_mutex(&mut self, cond_work: u32, mutex_work: u32) {
        self.lwcond_mutex.retain(|&(c, _)| c != cond_work);
        self.lwcond_mutex.push((cond_work, mutex_work));
    }

    /// The lightweight mutex work address bound to lightweight cond `cond_work`, if
    /// its `CreateLwCond` was seen.
    fn lwcond_mutex_of(&self, cond_work: u32) -> Option<u32> {
        self.lwcond_mutex.iter().find(|&&(c, _)| c == cond_work).map(|&(_, m)| m)
    }

    /// Whether `cond_work` is a lightweight cond we recorded at `sceKernelCreateLwCond`
    /// (every created cond binds a mutex, so a recorded binding is its existence).
    pub fn lwcond_is_known(&self, cond_work: u32) -> bool {
        self.lwcond_mutex_of(cond_work).is_some()
    }

    /// Park the current thread in `sceKernelWaitLwCond` on cond `work`, atomically
    /// releasing its bound lightweight mutex (handing it to any waiter) - exactly like
    /// the heavyweight [`cond_wait`](Self::cond_wait). `timeout_us` of 0 is an infinite
    /// wait (only a signal wakes it); non-zero sets a deadline at `now + timeout_us`.
    /// On wake (signal or timeout) the thread re-acquires the mutex before it runs.
    ///
    /// Returns `false` if `work` is not a cond we recorded a `CreateLwCond` for: the
    /// kernel rejects an unknown lightweight cond (`SCE_KERNEL_ERROR_UNKNOWN_LW_COND_ID`)
    /// rather than releasing a mutex or blocking, so the caller must surface that error
    /// and the thread must NOT park. A cond with no binding reaching here means its
    /// creation was never observed upstream (the real bug lives there); faithful
    /// behavior here refuses to invent a binding.
    #[must_use]
    pub fn lwcond_wait(&mut self, work: u32, timeout_us: u32) -> bool {
        let Some(mutex_work) = self.lwcond_mutex_of(work) else {
            return false;
        };
        self.lwmutex_unlock(mutex_work);
        let deadline = (timeout_us != 0).then(|| self.virtual_us + timeout_us as u64);
        self.lwcond_waiters.push((self.current, work, deadline));
        true
    }


    /// `sceKernelSignalLwCond`/`SignalLwCondAll`: wake one (or all) threads parked on
    /// cond `work`. Each woken thread must re-acquire the cond's bound lightweight mutex
    /// before it runs (taken now if free, else queued behind the owner), mirroring the
    /// heavyweight [`cond_signal`](Self::cond_signal).
    pub fn lwcond_signal(&mut self, work: u32, all: bool) {
        let mut woke_one = false;
        let mut woken: Vec<i32> = Vec::new();
        self.lwcond_waiters.retain(|&(thid, w, _)| {
            if w == work && (all || !woke_one) {
                woken.push(thid);
                woke_one = true;
                false
            } else {
                true
            }
        });
        match self.lwcond_mutex_of(work) {
            Some(mutex_work) => {
                for thid in woken {
                    self.lwmutex_acquire_for(mutex_work, thid);
                }
            }
            // No bound mutex recorded (a bare cond): just make the waiters runnable.
            None => self.pending_wakes.extend(woken),
        }
    }

    /// Park the current thread until `now + us` on the virtual clock, woken only by
    /// time. Used for `sceKernelDelayThread` and `sceAudioOutOutput` grain pacing.
    pub fn sleep_park(&mut self, us: u64) {
        self.sleep_waiters.push((self.current, self.virtual_us.wrapping_add(us)));
    }

    /// Guest stack for the display-callback thread. One callback runs at a time,
    /// so a single stack serves every invocation.
    const DISPLAY_CB_STACK_BYTES: u32 = 0x1_0000;
    /// Callback-data copy slots. The queue is drained one entry per callback run;
    /// the ring only needs to cover entries submitted while earlier ones are still
    /// pending (the display queue itself is at most a few frames deep).
    const DISPLAY_CB_SLOT_COUNT: u32 = 8;

    /// Queue the registered sceGxmInitialize display callback to run as guest code
    /// with this entry's `callback_data` (preemptive mode only).
    ///
    /// On hardware every `sceGxmDisplayQueueAddEntry` eventually runs the
    /// callback - typically `sceDisplaySetFrameBuf` plus the game's own buffer
    /// bookkeeping - on the display queue's internal thread. A double-buffered
    /// title waits (often a bare memory-poll loop, no host calls) for that
    /// bookkeeping to release the older buffer, so a host that never runs the
    /// callback stalls it after exactly two frames. GXM copies `callback_data`
    /// into the queue at AddEntry time; mirrored here via a slot ring so the
    /// caller can immediately reuse its buffer.
    pub fn enqueue_display_callback(&mut self, ctx: &mut GuestCtx, callback_data: u32) {
        if !self.preemptive || self.display_queue_cb == 0 {
            return;
        }
        let size = self.display_queue_cb_data_size.max(4);
        if self.display_cb_slots == 0 {
            self.display_cb_slots = self.galloc(size * Self::DISPLAY_CB_SLOT_COUNT, 8);
            self.display_cb_stack = self.galloc(Self::DISPLAY_CB_STACK_BYTES, 16);
            if self.display_cb_slots == 0 || self.display_cb_stack == 0 {
                self.display_queue_cb = 0; // OOM: disable rather than corrupt
                return;
            }
        }
        let slot =
            self.display_cb_slots + (self.display_cb_next_slot % Self::DISPLAY_CB_SLOT_COUNT) * size;
        self.display_cb_next_slot = self.display_cb_next_slot.wrapping_add(1);
        if callback_data != 0 {
            let bytes = ctx.read_bytes(callback_data, size as usize);
            ctx.write_bytes(slot, &bytes);
        }
        self.display_cb_queue.push_back(slot);
        self.pump_display_callback();
    }

    /// Start the next queued display callback if none is in flight. Chained from
    /// [`Self::set_thread_exit`] when the in-flight one ends, preserving
    /// submission order.
    fn pump_display_callback(&mut self) {
        if self.display_cb_running.is_some() {
            return;
        }
        let Some(data) = self.display_cb_queue.pop_front() else { return };
        let thid = self.new_uid();
        self.pending_spawns.push(Reentry {
            entry: self.display_queue_cb & !1,
            arg_len: data, // r0: the callback's `const void *callbackData`
            arg_ptr: 0,
            r2: 0,
            stack_top: self.display_cb_stack + Self::DISPLAY_CB_STACK_BYTES,
            thid,
            priority: DEFAULT_THREAD_PRIORITY - 0x10, // above game threads: presents promptly
        });
        self.display_cb_running = Some(thid);
    }

    /// Guest stack bytes for a one-shot service-state callback delivery.
    const SERVICE_CB_STACK_BYTES: u32 = 0x8000;

    /// Record the SceNpManager service-state callback a title registered
    /// (`sceNpRegisterServiceStateCallback`); pumped by
    /// [`Self::pump_service_callbacks`].
    pub fn set_np_service_callback(&mut self, entry: u32, userdata: u32) {
        self.np_service_cb = (entry != 0).then_some((entry, userdata));
        self.np_cb_delivered = false;
    }

    /// Record the SceNetCtl inet-state callback a title registered
    /// (`sceNetCtlInetRegisterCallback`).
    pub fn set_net_inet_callback(&mut self, entry: u32, userdata: u32) {
        self.net_inet_cb = (entry != 0).then_some((entry, userdata));
        self.net_cb_delivered = false;
    }

    /// Spawn a one-shot guest callback `entry(r0, r1, r2)` on its own fresh stack
    /// (preemptive mode). Used to deliver the service-state notifications a title
    /// waits for; the callback typically just records the state into a global the
    /// boot state machine polls. Returns false if not spawned (single-thread mode
    /// or heap exhausted).
    fn spawn_oneshot_callback(&mut self, entry: u32, r0: u32, r1: u32, r2: u32) -> bool {
        if !self.preemptive || entry == 0 {
            return false;
        }
        let stack = self.galloc(Self::SERVICE_CB_STACK_BYTES, 16);
        if stack == 0 {
            return false;
        }
        let thid = self.new_uid();
        self.pending_spawns.push(Reentry {
            entry: entry & !1,
            arg_len: r0,
            arg_ptr: r1,
            r2,
            stack_top: stack + Self::SERVICE_CB_STACK_BYTES,
            thid,
            priority: DEFAULT_THREAD_PRIORITY,
        });
        true
    }

    /// Deliver any registered-but-undelivered service-state callback (called from
    /// the per-frame `sce*CheckCallback` pumps). Each fires exactly once, with the
    /// offline state the title expects: NP signed-out, net disconnected. Without
    /// this a title that gates its boot on the notification waits forever.
    ///
    /// `state` is the `SceNpServiceState` enum to deliver (signed-out offline).
    pub fn pump_np_callback(&mut self, state: u32) {
        if self.np_cb_delivered {
            return;
        }
        if let Some((entry, userdata)) = self.np_service_cb {
            // The registered NP callback is a C++ member-function thunk taking its
            // `this`/userdata in r2, the service-state enum in r0, and a second
            // reserved argument in r1 (verified by disassembling the game's handler,
            // which does `switch(r0)` on the state): callback(state, 0, userdata).
            if self.spawn_oneshot_callback(entry, state, 0, userdata) {
                self.np_cb_delivered = true;
            }
        }
    }

    /// Deliver the net inet-state callback once with `event` (a `SceNetCtlState`).
    /// The SceNetCtl callback ABI is `cb(int event_type, void *arg)`: r0=event,
    /// r1=arg (the registered userdata), r2 unused.
    pub fn pump_net_callback(&mut self, event: u32) {
        if self.net_cb_delivered {
            return;
        }
        if let Some((entry, userdata)) = self.net_inet_cb {
            if self.spawn_oneshot_callback(entry, event, userdata, 0) {
                self.net_cb_delivered = true;
            }
        }
    }

    /// The earliest pending timed-wake deadline across every timed blocking wait -
    /// lightweight-cond, heavyweight cond, semaphore, event flag - and pure sleeps.
    /// The scheduler uses this to jump the clock forward when every thread is blocked,
    /// instead of declaring a deadlock. A timed wait omitted here would be turned into
    /// an infinite park (it could never wake by timeout), so every timed-wait kind
    /// must contribute its deadline.
    pub fn earliest_lwcond_deadline(&self) -> Option<u64> {
        let lw = self.lwcond_waiters.iter().filter_map(|&(_, _, d)| d);
        let sl = self.sleep_waiters.iter().map(|&(_, d)| d);
        let ev = self.evf_waiters.iter().filter_map(|w| w.deadline);
        let sem = self.sema_waiters.iter().filter_map(|w| w.deadline);
        let cnd = self.conds.iter().flat_map(|c| c.waiters.iter().filter_map(|w| w.deadline));
        lw.chain(sl).chain(ev).chain(sem).chain(cnd).min()
    }

    /// Take the `r0` value owed to thread `thid` when it resumes from a block (a timed
    /// wait that expired), if any. Drained once, by the engine, at the point the thread
    /// resumes; `None` (a signal wake) leaves the thread's pre-park return value intact.
    pub fn take_resume_code(&mut self, thid: i32) -> Option<u32> {
        let idx = self.pending_resume_codes.iter().position(|&(t, _)| t == thid)?;
        Some(self.pending_resume_codes.remove(idx).1)
    }

    /// Advance the virtual clock to at least `to_us` and wake every timed wait whose
    /// deadline has now passed. Called by the scheduler when no thread is runnable but
    /// a timed wait can still fire. Every kind of timed *wait* (lightweight cond,
    /// heavyweight cond, semaphore, event flag) is woken with
    /// `SCE_KERNEL_ERROR_WAIT_TIMEOUT` delivered through the resume-code channel; a
    /// pure `sceKernelDelayThread` sleep instead completes with success (0), since a
    /// delay elapsing is not a timeout. A timed-out cond wait additionally re-acquires
    /// its mutex before resuming (it may re-block on the mutex first).
    pub fn advance_time_to(&mut self, to_us: u64) {
        self.virtual_us = self.virtual_us.max(to_us);
        let now = self.virtual_us;
        // Timed lightweight-cond waits: like the heavyweight cond, a timed-out
        // WaitLwCond re-acquires its bound mutex before resuming, so collect the
        // expirees, then hand each to its mutex (or wake directly if none is bound).
        let mut expired_lw: Vec<(i32, u32)> = Vec::new(); // (thid, cond work address)
        self.lwcond_waiters.retain(|&(thid, work, deadline)| match deadline {
            Some(d) if d <= now => {
                expired_lw.push((thid, work));
                false
            }
            _ => true,
        });
        for (thid, work) in expired_lw {
            self.pending_resume_codes.push((thid, SCE_KERNEL_ERROR_WAIT_TIMEOUT));
            match self.lwcond_mutex_of(work) {
                Some(mutex_work) => self.lwmutex_acquire_for(mutex_work, thid),
                None => self.pending_wakes.push(thid),
            }
        }
        // A pure sleep (sceKernelDelayThread / audio grain pacing) that elapses is a
        // successful completion, not a timed-out wait: wake it with its return value
        // (0) unchanged - no resume code.
        self.sleep_waiters.retain(|&(thid, deadline)| {
            if deadline <= now {
                self.pending_wakes.push(thid);
                false
            } else {
                true
            }
        });
        // Timed semaphore waits whose deadline passed: wake with WAIT_TIMEOUT. The
        // count is untouched (nothing was available to consume).
        self.sema_waiters.retain(|w| match w.deadline {
            Some(d) if d <= now => {
                self.pending_wakes.push(w.thid);
                self.pending_resume_codes.push((w.thid, SCE_KERNEL_ERROR_WAIT_TIMEOUT));
                false
            }
            _ => true,
        });
        // Timed heavyweight cond waits whose deadline passed: a timed-out WaitCond must
        // re-acquire its mutex before returning, so collect the expirees first, then
        // hand each to its mutex (taken now if free, else queued behind the owner). The
        // WAIT_TIMEOUT code is delivered whenever the thread ultimately resumes.
        let mut expired_cond: Vec<(i32, i32)> = Vec::new(); // (thid, mutex uid)
        for c in self.conds.iter_mut() {
            let mutex = c.mutex;
            c.waiters.retain(|w| match w.deadline {
                Some(d) if d <= now => {
                    expired_cond.push((w.thid, mutex));
                    false
                }
                _ => true,
            });
        }
        for (thid, mutex) in expired_cond {
            self.pending_resume_codes.push((thid, SCE_KERNEL_ERROR_WAIT_TIMEOUT));
            self.mutex_acquire_for(mutex, thid);
        }
        // Timed event flag waits whose deadline passed: wake with WAIT_TIMEOUT and the
        // CURRENT pattern written through outBits (the caller reads the pattern back
        // and re-checks; see `vita::sync::wait_event_flag`).
        let patterns: Vec<(i32, u32)> = self.event_flags.clone();
        self.evf_waiters.retain(|w| match w.deadline {
            Some(d) if d <= now => {
                if w.out_addr != 0 {
                    let p = patterns.iter().find(|(u, _)| *u == w.uid).map(|(_, p)| *p).unwrap_or(0);
                    self.pending_stat_writes.push((w.out_addr, p));
                }
                self.pending_wakes.push(w.thid);
                self.pending_resume_codes.push((w.thid, SCE_KERNEL_ERROR_WAIT_TIMEOUT));
                false
            }
            _ => true,
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

    /// SceEventFlagWaitTypes: OR (any requested bit) vs the default AND (all of
    /// them), plus the two clear-on-match ops.
    const EVF_WAITOR: u32 = 1;
    const EVF_WAITCLEAR: u32 = 2;
    const EVF_WAITCLEAR_PAT: u32 = 4;

    /// Whether `pattern` satisfies a wait for `bits` under `mode`.
    fn evf_satisfied(pattern: u32, bits: u32, mode: u32) -> bool {
        if mode & Self::EVF_WAITOR != 0 {
            pattern & bits != 0
        } else {
            pattern & bits == bits
        }
    }

    /// Try to satisfy an event flag wait without blocking. On a match, applies the
    /// mode's clear op and returns the pattern AT the match (what `outBits` should
    /// report); `None` means the caller must park ([`Self::evf_block`]).
    pub fn evf_try_wait(&mut self, uid: i32, bits: u32, mode: u32) -> Option<u32> {
        let (_, p) = self.event_flags.iter_mut().find(|(u, _)| *u == uid)?;
        if !Self::evf_satisfied(*p, bits, mode) {
            return None;
        }
        let at_match = *p;
        if mode & Self::EVF_WAITCLEAR != 0 {
            *p = 0;
        } else if mode & Self::EVF_WAITCLEAR_PAT != 0 {
            *p &= !bits;
        }
        Some(at_match)
    }

    /// Park the current thread on event flag `uid` until `bits` is satisfied under
    /// `mode` (or `timeout_us` passes; 0 = wait forever). `out_addr` is the guest
    /// `outBits` pointer the wake will write the match pattern through.
    pub fn evf_block(&mut self, uid: i32, bits: u32, mode: u32, out_addr: u32, timeout_us: u32) {
        let deadline = (timeout_us != 0).then(|| self.virtual_us + timeout_us as u64);
        self.evf_waiters.push(EvfWaiter { uid, thid: self.current, bits, mode, out_addr, deadline });
    }

    /// Set bits on an event flag, then release every parked waiter the new pattern
    /// satisfies (in FIFO order; each match applies its clear op before the next
    /// waiter is evaluated, matching kernel release order). The preemptive
    /// counterpart of [`event_set`](Self::event_set).
    pub fn event_set_wake(&mut self, uid: i32, bits: u32) {
        self.event_set(uid, bits);
        loop {
            let pattern = self.event_pattern(uid);
            let next = self
                .evf_waiters
                .iter()
                .position(|w| w.uid == uid && Self::evf_satisfied(pattern, w.bits, w.mode));
            let Some(idx) = next else { break };
            let w = self.evf_waiters.remove(idx);
            let at_match = self
                .evf_try_wait(uid, w.bits, w.mode)
                .expect("matched waiter satisfies its own condition");
            if w.out_addr != 0 {
                self.pending_stat_writes.push((w.out_addr, at_match));
            }
            self.pending_wakes.push(w.thid);
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
    /// the guest address (0 on exhaustion). Deterministic: a pure function of the
    /// allocation order. The bump cursor is capped at the guest region ceiling
    /// (`base + mem_bytes`): the indirect-dispatch address table lives immediately
    /// above that, so an unbounded heap must never be handed an address that reaches
    /// it (a stray guest/host write there would corrupt the table and fault every
    /// later indirect call). An allocation past the ceiling is a real
    /// out-of-memory: return 0 rather than a colliding pointer.
    pub fn galloc(&mut self, size: u32, align: u32) -> u32 {
        let a = align.max(4);
        let p = (self.alloc_cursor + a - 1) & !(a - 1);
        let ceiling = self.base.wrapping_add(self.mem_bytes);
        let end = p.wrapping_add(size.max(4));
        if end > ceiling || end < p {
            return 0; // heap exhausted; do not encroach on the dispatch table
        }
        self.alloc_cursor = end;
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
        program_header: u32,
    ) {
        self.vertex_programs.push((handle, VertexProgramInfo { attributes, stride, program_header }));
    }

    /// The `SceGxmProgram*` a vertex program handle was created from, if recorded.
    fn vertex_program_header(&self, handle: u32) -> u32 {
        self.vertex_programs.iter().find(|(h, _)| *h == handle).map(|(_, i)| i.program_header).unwrap_or(0)
    }

    /// Record `sceGxmShaderPatcherCreateFragmentProgram`'s handle -> `SceGxmProgram*`.
    pub fn set_fragment_program(&mut self, handle: u32, program_header: u32) {
        self.fragment_programs.push((handle, program_header));
    }

    fn fragment_program_header(&self, handle: u32) -> u32 {
        self.fragment_programs.iter().find(|(h, _)| *h == handle).map(|(_, p)| *p).unwrap_or(0)
    }

    /// Record a color surface, keyed by its guest struct address.
    pub fn set_color_surface(&mut self, addr: u32, surface: crate::capture::ColorSurface) {
        self.color_surfaces.push((addr, surface));
    }

    // --- scene assembly (used by the gxm handlers) ---

    pub fn begin_scene(&mut self, color_surface_addr: u32) {
        self.texture_snapshots.clear();
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

    /// `sceGxmSetFragmentProgram`: resolve the bound fragment program handle to its
    /// `SceGxmProgram*` and remember it, so a draw can reflect its samplers to choose the
    /// albedo texture. A null/unknown handle clears the binding (header 0).
    pub fn bind_fragment_program(&mut self, handle: u32) {
        self.bound_fragment_program_header = self.fragment_program_header(handle);
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

    /// Mutable access to the live GXM fixed-function state, so a `sceGxmSet*` setter
    /// updates the single field it owns (the state is sticky and snapshotted per draw).
    pub fn render_state_mut(&mut self) -> &mut crate::capture::RenderState {
        &mut self.render_state
    }

    /// The recorded sampler wrap/LOD state for a `SceGxmTexture*`, defaulting to the
    /// GXM defaults (REPEAT/REPEAT/0) when the guest never set it.
    fn texture_sampler(&self, texture_addr: u32) -> (u32, u32, u32) {
        self.texture_samplers
            .iter()
            .find(|(a, _)| *a == texture_addr)
            .map(|(_, s)| *s)
            .unwrap_or((0, 0, 0))
    }

    /// Update one component of a texture's sampler state, keyed by its guest address.
    /// `which`: 0 = U addr mode, 1 = V addr mode, 2 = LOD bias.
    pub fn set_texture_sampler(&mut self, texture_addr: u32, which: u8, value: u32) {
        let slot = match self.texture_samplers.iter_mut().find(|(a, _)| *a == texture_addr) {
            Some((_, s)) => s,
            None => {
                self.texture_samplers.push((texture_addr, (0, 0, 0)));
                &mut self.texture_samplers.last_mut().unwrap().1
            }
        };
        match which {
            0 => slot.0 = value,
            1 => slot.1 = value,
            _ => slot.2 = value,
        }
    }

    /// The color surface recorded for `addr` (its `SceGxmColorSurface*` struct
    /// address), if the guest initialized one there.
    pub fn color_surface(&self, addr: u32) -> Option<crate::capture::ColorSurface> {
        self.color_surfaces.iter().find(|(a, _)| *a == addr).map(|(_, s)| *s)
    }

    /// The sticky extra state for a `SceGxmTexture*`, or GXM defaults if never set.
    fn texture_extra(&self, texture_addr: u32) -> TextureExtra {
        self.texture_extra
            .iter()
            .find(|(a, _)| *a == texture_addr)
            .map(|(_, e)| *e)
            .unwrap_or_default()
    }

    /// Mutable slot for a texture's extra state, inserting GXM defaults on first touch.
    fn texture_extra_mut(&mut self, texture_addr: u32) -> &mut TextureExtra {
        if let Some(i) = self.texture_extra.iter().position(|(a, _)| *a == texture_addr) {
            &mut self.texture_extra[i].1
        } else {
            self.texture_extra.push((texture_addr, TextureExtra::default()));
            &mut self.texture_extra.last_mut().unwrap().1
        }
    }

    /// Record the mip count / explicit byte stride a `sceGxmTextureInit*` established
    /// (the 6th argument is `mipCount` for every layout except `LINEAR_STRIDED`, where
    /// it is the byte stride). Called from the texture-init handler.
    pub fn set_texture_init_extra(&mut self, texture_addr: u32, mip_count: u32, byte_stride: u32) {
        let e = self.texture_extra_mut(texture_addr);
        e.mip_count = mip_count.max(1);
        e.byte_stride = byte_stride;
    }

    /// Record a texture filter. `which`: 0 = min, 1 = mag, 2 = mip.
    pub fn set_texture_filter(&mut self, texture_addr: u32, which: u8, value: u32) {
        let e = self.texture_extra_mut(texture_addr);
        match which {
            0 => e.min_filter = value,
            1 => e.mag_filter = value,
            _ => e.mip_filter = value,
        }
    }

    /// Record a texture's gamma-correction mode (`sceGxmTextureSetGammaMode`).
    pub fn set_texture_gamma(&mut self, texture_addr: u32, gamma: u32) {
        self.texture_extra_mut(texture_addr).gamma = gamma;
    }

    /// `sceGxmTextureGetMipmapCountUnsafe`.
    pub fn texture_mip_count(&self, texture_addr: u32) -> u32 {
        self.texture_extra(texture_addr).mip_count
    }

    /// `sceGxmTextureGetLodBias`.
    pub fn texture_lod_bias(&self, texture_addr: u32) -> u32 {
        self.texture_sampler(texture_addr).2
    }

    /// `sceGxmTextureGet{U,V}AddrModeSafe`. `which`: 0 = U, 1 = V.
    pub fn texture_addr_mode(&self, texture_addr: u32, which: u8) -> u32 {
        let s = self.texture_sampler(texture_addr);
        if which == 0 { s.0 } else { s.1 }
    }

    /// `sceGxmTextureGet{Min,Mag}Filter` / gamma. `which`: 0 = min, 1 = mag, 2 = gamma.
    pub fn texture_filter(&self, texture_addr: u32, which: u8) -> u32 {
        let e = self.texture_extra(texture_addr);
        match which {
            0 => e.min_filter,
            1 => e.mag_filter,
            _ => e.gamma,
        }
    }

    /// `sceGxmTextureGetStride`: the row pitch in bytes. Returns the explicit byte
    /// stride for a `LINEAR_STRIDED` texture, otherwise the driver-derived linear
    /// stride: uncompressed rows are padded to a multiple of 8 texels (the GXM linear
    /// alignment), compressed rows are block-packed - the same formula the capture
    /// decoder uses (`decode_texture`).
    pub fn texture_stride(&self, ctx: &GuestCtx, texture_addr: u32) -> u32 {
        let e = self.texture_extra(texture_addr);
        if e.byte_stride != 0 {
            return e.byte_stride;
        }
        let w0 = ctx.read_u32(texture_addr);
        let w1 = ctx.read_u32(texture_addr.wrapping_add(4));
        let width = ((w1 >> 12) & 0xfff) + 1;
        // Reconstruct the full base-format high byte exactly as decode_texture does:
        // prefer the exact 32-bit format the guest set, else the 5-bit control-word
        // field plus the format0 extension bit (word 0 bit 31).
        let base_format = match self.texture_format(texture_addr) {
            Some(f) => (f >> 24) & 0xff,
            None => ((w1 >> 24) & 0x1f) | (((w0 >> 31) & 1) << 7),
        };
        match crate::render::block_layout(base_format) {
            Some((block_w, _block_h, block_bytes)) => {
                if block_w == 1 {
                    align_up(width, 8) * block_bytes
                } else {
                    width.div_ceil(block_w) * block_bytes
                }
            }
            None => width * 4,
        }
    }

    /// Record a color surface's gamma-correction mode (`sceGxmColorSurfaceSetGammaMode`).
    pub fn set_color_surface_gamma(&mut self, surface_addr: u32, gamma: u32) {
        self.color_surface_gamma.retain(|(a, _)| *a != surface_addr);
        self.color_surface_gamma.push((surface_addr, gamma));
    }

    /// The GPU notification region, allocating it on first use. Returns a guest
    /// pointer to `SCE_GXM_NOTIFICATION_COUNT` (512) u32 slots.
    pub fn notification_region(&mut self) -> u32 {
        if self.notification_region == 0 {
            self.notification_region = self.galloc(512 * 4, 16);
        }
        self.notification_region
    }

    /// Record `sceGxmPrecomputedDrawInit(precomputedDraw, vertexProgram, memBlock)`:
    /// start a precomputed-draw record keyed by the guest block address, bound to the
    /// given vertex program (its handle). Any prior record at that address is replaced.
    pub fn precomputed_draw_init(&mut self, ctx: &mut GuestCtx, precomputed: u32, vertex_program: u32) {
        for w in 0..pdraw::WORDS {
            ctx.write_u32(precomputed + w * 4, 0);
        }
        ctx.write_u32(precomputed + pdraw::OFF_MAGIC, pdraw::MAGIC);
        ctx.write_u32(precomputed + pdraw::OFF_VERTEX_PROGRAM, vertex_program);
    }

    /// Record `sceGxmPrecomputedDrawSetVertexStream(precomputedDraw, streamIndex, data)`.
    pub fn precomputed_draw_set_stream(
        &mut self,
        ctx: &mut GuestCtx,
        precomputed: u32,
        stream_index: u32,
        data: u32,
    ) {
        if stream_index == 0 {
            ctx.write_u32(precomputed + pdraw::OFF_STREAM0, data);
        }
    }

    /// Record `sceGxmPrecomputedDrawSetParams(precomputedDraw, prim, indexType,
    /// indexData, indexCount)`.
    pub fn precomputed_draw_set_params(
        &mut self,
        ctx: &mut GuestCtx,
        precomputed: u32,
        primitive: u32,
        index_format: u32,
        index_addr: u32,
        index_count: u32,
    ) {
        ctx.write_u32(precomputed + pdraw::OFF_PRIMITIVE, primitive);
        ctx.write_u32(precomputed + pdraw::OFF_INDEX_FORMAT, index_format);
        ctx.write_u32(precomputed + pdraw::OFF_INDEX_ADDR, index_addr);
        ctx.write_u32(precomputed + pdraw::OFF_INDEX_COUNT, index_count);
    }

    /// Read back a precomputed draw from its guest block, or `None` when the block does
    /// not carry the initialised tag.
    fn precomputed_draw_read(ctx: &GuestCtx, precomputed: u32) -> Option<PrecomputedDraw> {
        if ctx.read_u32(precomputed + pdraw::OFF_MAGIC) != pdraw::MAGIC {
            return None;
        }
        Some(PrecomputedDraw {
            vertex_program: ctx.read_u32(precomputed + pdraw::OFF_VERTEX_PROGRAM),
            stream0: ctx.read_u32(precomputed + pdraw::OFF_STREAM0),
            primitive: ctx.read_u32(precomputed + pdraw::OFF_PRIMITIVE),
            index_format: ctx.read_u32(precomputed + pdraw::OFF_INDEX_FORMAT),
            index_addr: ctx.read_u32(precomputed + pdraw::OFF_INDEX_ADDR),
            index_count: ctx.read_u32(precomputed + pdraw::OFF_INDEX_COUNT),
        })
    }

    /// Replay `sceGxmDrawPrecomputed(context, precomputedDraw)`: bind the precomputed
    /// draw's vertex program + stream-0 buffer and record it into the current scene,
    /// exactly as a `sceGxmDraw` would. The bound textures and reserved uniform buffer
    /// are whatever the guest set on the context around this call (sticky GXM state).
    pub fn draw_precomputed(&mut self, ctx: &GuestCtx, precomputed: u32) {
        let Some(d) = Self::precomputed_draw_read(ctx, precomputed) else {
            // A block with no initialised tag: the draw would be LOST, which shows up in the
            // frame only as missing geometry. Say so rather than return quietly.
            tracing::debug!(
                target: "vitaslop::gxm",
                precomputed = format_args!("{precomputed:#x}"),
                "drawPrecomputed on an uninitialised block - draw DROPPED"
            );
            return;
        };
        self.bound_vertex_program = d.vertex_program;
        self.bound_stream0 = d.stream0;
        self.record_draw(ctx, d.primitive, d.index_format, d.index_addr, d.index_count);
    }

    // --- Precomputed vertex/fragment state ----------------------------------
    //
    // A precomputed state bundles the uniform buffer + textures for one shader stage
    // into a guest struct the game builds once and binds each draw. We record the
    // guest-set pointers per state address, then `bind_precomputed_*_state` copies them
    // into the live bind state so `record_draw` snapshots the same bytes it would on the
    // direct `sceGxmSetUniformDataF`/`sceGxmSetFragmentTexture` path.

    /// `sceGxmPrecomputedVertexStateInit(state, vertexProgram, memBlock)`: start a fresh
    /// vertex-state record bound to the vertex program (resolved to its `SceGxmProgram*`
    /// for later uniform-buffer sizing). Replaces any prior record at that address.
    pub fn precomputed_vertex_state_init(&mut self, state: u32, vertex_program: u32) {
        let program_header = self.vertex_program_header(vertex_program);
        self.precomputed_vertex_states
            .insert(state, PrecomputedState { program_header, ..PrecomputedState::default() });
    }

    /// `sceGxmPrecomputedFragmentStateInit(state, fragmentProgram, memBlock)`.
    pub fn precomputed_fragment_state_init(&mut self, state: u32, fragment_program: u32) {
        let program_header = self.fragment_program_header(fragment_program);
        self.precomputed_fragment_states
            .insert(state, PrecomputedState { program_header, ..PrecomputedState::default() });
    }

    /// `sceGxmPrecomputed{Vertex,Fragment}StateSetDefaultUniformBuffer(state, buffer)`:
    /// store the guest pointer the game will write this stage's uniforms into. The record
    /// is created lazily so a setter before `Init` (unexpected) still lands.
    pub fn precomputed_vertex_state_set_uniform_buffer(&mut self, state: u32, buffer: u32) {
        self.precomputed_vertex_states.entry(state).or_default().default_uniform_buffer = buffer;
    }
    pub fn precomputed_fragment_state_set_uniform_buffer(&mut self, state: u32, buffer: u32) {
        self.precomputed_fragment_states.entry(state).or_default().default_uniform_buffer = buffer;
    }

    /// `sceGxmPrecomputed{Vertex,Fragment}StateGetDefaultUniformBuffer(state)`: the pointer
    /// last set (0 if never set), so a Set/Get round-trips faithfully.
    pub fn precomputed_vertex_state_uniform_buffer(&self, state: u32) -> u32 {
        self.precomputed_vertex_states.get(&state).map(|s| s.default_uniform_buffer).unwrap_or(0)
    }
    pub fn precomputed_fragment_state_uniform_buffer(&self, state: u32) -> u32 {
        self.precomputed_fragment_states.get(&state).map(|s| s.default_uniform_buffer).unwrap_or(0)
    }

    /// `sceGxmPrecomputed{Vertex,Fragment}StateSetTexture(state, index, texture)`: bind a
    /// `SceGxmTexture*` to this stage's sampler `index` (0 unbinds), replacing any prior
    /// binding at that index. Textures are kept sorted by index so the bound order is
    /// stable when the state is applied.
    pub fn precomputed_vertex_state_set_texture(&mut self, state: u32, index: u32, texture: u32) {
        Self::state_set_texture(self.precomputed_vertex_states.entry(state).or_default(), index, texture);
    }
    pub fn precomputed_fragment_state_set_texture(&mut self, state: u32, index: u32, texture: u32) {
        Self::state_set_texture(self.precomputed_fragment_states.entry(state).or_default(), index, texture);
    }

    fn state_set_texture(s: &mut PrecomputedState, index: u32, texture: u32) {
        s.textures.retain(|(i, _)| *i != index);
        if texture != 0 {
            s.textures.push((index, texture));
            s.textures.sort_by_key(|(i, _)| *i);
        }
    }

    /// Apply `sceGxmSetPrecomputedVertexState(context, state)`: bind this stage's default
    /// uniform buffer (pointer + size, sized from the vertex program header at +0x2C) so
    /// the next `record_draw` reads its uniforms from guest memory. A state that was never
    /// built (or a null bind) clears the binding, restoring the direct uniform path.
    pub fn bind_precomputed_vertex_state(&mut self, ctx: &GuestCtx, state: u32) {
        match self.precomputed_vertex_states.get(&state) {
            Some(s) => {
                self.bound_vertex_uniform_buf = s.default_uniform_buffer;
                self.bound_vertex_uniform_size = self
                    .reflected_uniform_size_bytes(ctx, s.program_header)
                    .max(default_uniform_buffer_bytes(ctx, s.program_header));
                tracing::trace!(
                    target: "vitaslop::gxm",
                    buffer = format_args!("{:#x}", self.bound_vertex_uniform_buf),
                    size = self.bound_vertex_uniform_size,
                    header = format_args!("{:#x}", s.program_header),
                    "bindPrecomputedVertexState"
                );
            }
            None => {
                self.bound_vertex_uniform_buf = 0;
                self.bound_vertex_uniform_size = 0;
            }
        }
    }

    /// Apply `sceGxmSetPrecomputedFragmentState(context, state)`: bind this stage's
    /// textures to the context sampler units, exactly as a sequence of
    /// `sceGxmSetFragmentTexture` calls would, so `record_draw` snapshots them.
    pub fn bind_precomputed_fragment_state(&mut self, ctx: &GuestCtx, state: u32) {
        let (textures, header, uniform_buf) = match self.precomputed_fragment_states.get(&state) {
            Some(s) => (s.textures.clone(), s.program_header, s.default_uniform_buffer),
            None => return,
        };
        self.bound_textures = textures;
        self.bound_fragment_program_header = header;
        // Bind this stage's default uniform buffer (pointer + reflected size) so the draw
        // reads the per-material fragment uniforms (tint / light / fog) from guest memory,
        // exactly as the precomputed vertex path binds the vertex uniform buffer.
        self.bound_fragment_uniform_buf = uniform_buf;
        self.bound_fragment_uniform_size = self
            .reflected_uniform_size_bytes(ctx, header)
            .max(default_uniform_buffer_bytes(ctx, header))
            .min(4096);
    }

    /// `sceGxmReserveVertexDefaultUniformBuffer(context, void **uniformBuffer)`: hand
    /// back a fresh guest buffer sized to the bound vertex program's default uniform
    /// block, and bind it as the vertex uniform source so `record_draw` reads whatever
    /// the guest writes into it (this is the direct-path counterpart of a precomputed
    /// vertex state's default uniform buffer). `sceGxmSetUniformDataF` also copies into
    /// this same buffer, so reading it captures both ways the guest sets uniforms. The
    /// size is read from the program header (+0x2C) and clamped so an unresolved header
    /// cannot request an absurd allocation; a program with no default uniforms yields
    /// size 0, and the draw falls back to the `sceGxmSetUniformDataF` capture.
    pub fn reserve_vertex_uniform_buffer(&mut self, ctx: &mut GuestCtx) -> u32 {
        let header = self.vertex_program_header(self.bound_vertex_program);
        let header_size = default_uniform_buffer_bytes(ctx, header);
        let size = self.reflected_uniform_size_bytes(ctx, header).max(header_size).min(4096);
        let buf = self.galloc(size.max(256), 16);
        poison_uniform_buffer(ctx, buf, size);
        self.bound_vertex_uniform_buf = buf;
        self.bound_vertex_uniform_size = size;
        tracing::trace!(
            target: "vitaslop::gxm",
            program = format_args!("{:#x}", self.bound_vertex_program),
            header = format_args!("{header:#x}"),
            header_size,
            size,
            buffer = format_args!("{buf:#x}"),
            "reserveVertexDefaultUniformBuffer"
        );
        buf
    }

    /// `sceGxmReserveFragmentDefaultUniformBuffer`: the fragment-stage counterpart of
    /// [`Self::reserve_vertex_uniform_buffer`]. Hand back a guest buffer sized to the bound
    /// fragment program's default uniform block and bind it as the fragment uniform source,
    /// so a title that writes its per-material uniforms (tint / light / fog) directly into
    /// this buffer has them captured into the draw's material.
    pub fn reserve_fragment_uniform_buffer(&mut self, ctx: &GuestCtx) -> u32 {
        let header = self.bound_fragment_program_header;
        let header_size = default_uniform_buffer_bytes(ctx, header);
        let size = self.reflected_uniform_size_bytes(ctx, header).max(header_size).min(4096);
        let buf = self.galloc(size.max(256), 16);
        self.bound_fragment_uniform_buf = buf;
        self.bound_fragment_uniform_size = size;
        buf
    }

    /// Release a memory block by SceUID (`sceKernelFreeMemBlock`). Returns true if a
    /// block was registered under `uid`. The deterministic bump allocation itself is
    /// not reclaimed (the arena only grows), but the registry entry is removed so a
    /// later `sceKernelGetMemBlockBase(uid)` no longer resolves it, matching the guest-
    /// visible contract that the id is now invalid.
    pub fn free_memblock(&mut self, uid: i32) -> bool {
        let before = self.memblocks.len();
        self.memblocks.retain(|b| b.uid != uid);
        self.memblocks.len() != before
    }

    pub fn set_uniforms(&mut self, values: Vec<f32>) {
        self.pending_uniforms = values;
    }

    /// Human-readable dump of every preemptive sync primitive's ownership/waiter state
    /// plus the sleep/timed-wait queues, for diagnosing a boot that stalls without
    /// reaching a frame. The decisive question it answers: is a given thread parked in
    /// some waiter list (blocked, and on what) or absent from all of them (running -
    /// i.e. spinning in pure guest compute)? Content-free (ids/addresses only).
    pub fn debug_sync_dump(&self) -> String {
        use std::fmt::Write;
        let mut s = String::new();
        let _ = writeln!(s, "current thread = {:#x}, virtual_us = {}", self.current, self.virtual_us);
        let _ = writeln!(s, "lwmutexes ({}):", self.lwmutexes.len());
        for m in &self.lwmutexes {
            let _ = writeln!(
                s, "  work={:#010x} owner={:?} count={} waiters={:x?}",
                m.work, m.owner, m.count, m.waiters
            );
        }
        let _ = writeln!(s, "mutexes ({}):", self.mutexes.len());
        for m in &self.mutexes {
            let _ = writeln!(
                s, "  uid={:#x} owner={:?} count={} waiters={:x?}",
                m.uid, m.owner, m.count, m.waiters
            );
        }
        let _ = writeln!(s, "semaphores ({}):", self.semaphores.len());
        for (uid, count) in &self.semaphores {
            let waiters: Vec<(i32, i32)> =
                self.sema_waiters.iter().filter(|w| w.uid == *uid).map(|w| (w.thid, w.need)).collect();
            let _ = writeln!(s, "  uid={uid:#x} count={count} waiters(thid,need)={waiters:x?}");
        }
        let _ = writeln!(s, "cond waiters:");
        for c in &self.conds {
            let w: Vec<i32> = c.waiters.iter().map(|x| x.thid).collect();
            if !w.is_empty() {
                let _ = writeln!(s, "  cond uid={:#x} mutex={:#x} waiters={:x?}", c.uid, c.mutex, w);
            }
        }
        let _ = writeln!(s, "lwcond waiters (thid, cond_work, deadline): {:x?}", self.lwcond_waiters);
        let _ = writeln!(s, "lwcond->mutex bindings (cond_work, mutex_work): {:x?}", self.lwcond_mutex);
        let _ = writeln!(s, "sleep waiters (thid, deadline_us): {:x?}", self.sleep_waiters);
        s
    }

    /// The bound vertex program's layout, if recorded.
    fn bound_layout(&self) -> Option<&VertexProgramInfo> {
        self.vertex_programs
            .iter()
            .find(|(h, _)| *h == self.bound_vertex_program)
            .map(|(_, info)| info)
    }

    /// The vertex uniforms in effect for the next draw. On the precomputed path
    /// (`bound_vertex_uniform_buf` set by `sceGxmSetPrecomputedVertexState`) read the
    /// default uniform buffer straight from guest memory, sized by the program's default
    /// uniform buffer size. Otherwise use the `sceGxmSetUniformDataF` capture.
    fn current_vertex_uniforms(&self, ctx: &GuestCtx) -> Vec<f32> {
        if self.bound_vertex_uniform_buf != 0 && self.bound_vertex_uniform_size >= 4 {
            let count = (self.bound_vertex_uniform_size / 4) as usize;
            (0..count)
                .map(|i| ctx.read_f32(self.bound_vertex_uniform_buf + (i as u32) * 4))
                .collect()
        } else {
            self.pending_uniforms.clone()
        }
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
        let mut indices = ctx.read_bytes(index_addr, index_count as usize * index_elem);
        let index_of = |c: &[u8]| match index_elem {
            2 => u16::from_le_bytes([c[0], c[1]]) as u32,
            _ => u32::from_le_bytes([c[0], c[1], c[2], c[3]]),
        };
        // Snapshot exactly the vertices this draw REFERENCES, not the whole prefix of the
        // stream. A chunked world mesh draws a few hundred vertices out of a shared buffer of
        // tens of thousands, so copying `0..=max_index` per draw costs hundreds of megabytes a
        // frame (and reads far past what the draw can touch). Take the `min..=max` window and
        // rebase the indices onto it, which leaves every consumer's indexing unchanged.
        let (min_index, max_index) = indices.chunks(index_elem).fold((u32::MAX, 0u32), |(lo, hi), c| {
            let i = index_of(c);
            (lo.min(i), hi.max(i))
        });
        let (first_vertex, vertex_count) = if min_index > max_index {
            (0, 0) // no indices at all
        } else {
            (min_index, max_index - min_index + 1)
        };
        if first_vertex > 0 {
            for c in indices.chunks_mut(index_elem) {
                let rebased = index_of(c) - first_vertex;
                match index_elem {
                    2 => c[..2].copy_from_slice(&(rebased as u16).to_le_bytes()),
                    _ => c[..4].copy_from_slice(&rebased.to_le_bytes()),
                }
            }
        }
        let vertices = ctx.read_bytes(
            self.bound_stream0.wrapping_add(first_vertex * stride),
            (vertex_count * stride) as usize,
        );
        // Snapshot every bound fragment texture (decoded from its control words),
        // sorted by unit so unit 0 is first.
        let mut units: Vec<(u32, u32)> = self.bound_textures.clone();
        units.sort_by_key(|(u, _)| *u);
        // Read the per-unit control state first, so the snapshot cache can be borrowed
        // mutably for the decode loop without also holding a shared borrow of `self`.
        let unit_state: Vec<(u32, u32, Option<u32>, (u32, u32, u32), (u32, u32))> = units
            .iter()
            .map(|&(unit, addr)| {
                let e = self.texture_extra(addr);
                (unit, addr, self.texture_format(addr), self.texture_sampler(addr), (e.min_filter, e.mag_filter))
            })
            .collect();
        let snapshots = &mut self.texture_snapshots;
        let mut textures: Vec<crate::capture::BoundTexture> = unit_state
            .into_iter()
            .filter_map(|(unit, addr, format, sampler, filters)| {
                decode_texture(ctx, snapshots, unit, addr, format, sampler, filters)
            })
            .collect();
        // The capture renderer samples a single texture (`textures.first()`). This title
        // binds the NORMAL map at unit 0, so pick the albedo sampler by fragment-program
        // reflection and move it to the front; without this every surface is tinted by a
        // normal map (flat blue/purple). Falls back to the unit-0 order when reflection
        // finds no albedo or that unit is not currently bound.
        // Failing name reflection, index 0 must still be a plausible SURFACE texture: a
        // one-dimensional lookup table (a fog ramp) or a cube map (an irradiance probe) can sort
        // ahead of the real albedo by unit number, and neither is indexed by surface UV. See
        // `Draw::albedo`, which drops a leading non-surface texture rather than stretch it.
        if let Some(pos) =
            textures.iter().position(|t| t.height > 1 && t.faces <= 1).filter(|&p| p > 0)
        {
            textures[..=pos].rotate_right(1);
        }
        if let Some(unit) = self.fragment_albedo_unit(ctx) {
            if let Some(pos) = textures.iter().position(|t| t.unit == unit) {
                textures[..=pos].rotate_right(1);
            }
        }
        if std::env::var("VITASLOP_DUMP_FPROG").is_ok() {
            self.dump_fragment_program_samplers(ctx);
        }
        self.dump_draw_gxp(ctx, &textures, &attributes, primitive, index_count, stride);
        // Vertex uniforms: on the precomputed path the game wrote them into a default
        // uniform buffer bound by `sceGxmSetPrecomputedVertexState`, so read that guest
        // buffer now (its contents are current at draw time). On the direct path the
        // buffer is 0 and we fall back to the `sceGxmSetUniformDataF` capture.
        let mut uniforms = self.current_vertex_uniforms(ctx);
        // The RAW vertex default-uniform (SA bank) as the guest wrote it, BEFORE the composed
        // MVP is stamped over lanes 0..16 below. This is what the recompiled vertex shader
        // needs (it recomputes its own clip transform from the guest matrices), and unlike the
        // raw `bound_vertex_uniform_buf` it also covers the direct `sceGxmSetUniformDataF`
        // path (where that pointer is 0). Only materialised when the recompiler is enabled.
        let vert_sa_raw: Vec<u8> = if gxp_live_capture() {
            uniforms.iter().flat_map(|f| f.to_le_bytes()).collect()
        } else {
            Vec::new()
        };
        // The model-to-world matrix (for bringing the vertex normal into world space for
        // lighting). Read from the ORIGINAL uniforms, before `composed_mvp` below overwrites
        // uniforms[0..16] (which is exactly where `vsModelToWorldMatrix` usually sits).
        let world = self.reflected_world_matrix(ctx, &uniforms);
        // Recover the true clip-space transform from the vertex program's reflected
        // uniforms. A 3D shader keeps a per-object model->world matrix and a shared
        // world->projection matrix as separate uniforms (named e.g. `vsModelToWorldMatrix`
        // / `vsWorldToProjectionMatrix`), and multiplies them in the shader. The capture
        // renderer has no shader, so compose that product here and stamp it over the
        // first 16 floats, which the software/GPU paths read as the MVP. A shader with a
        // single combined transform (2D UI: `vsPrimRenderTransform` at offset 0) has no
        // projection matrix, so this leaves its MVP untouched.
        if let Some(mvp) = self.composed_mvp(ctx, &uniforms) {
            if uniforms.len() >= 16 {
                uniforms[..16].copy_from_slice(&mvp);
            }
        }
        if std::env::var("VITASLOP_DUMP_VPROG").is_ok() {
            self.dump_vertex_program_params(ctx);
        }
        let exposure = self.reflected_exposure(ctx, &uniforms);
        // The per-material fragment inputs (tint / directional light / ambient), reflected
        // from the fragment program's uniforms so the renderer reproduces the LIT colour.
        let material = self.reflect_fragment_material(ctx, &textures);
        // Snapshot the raw shader blobs + SA uniform bytes for the GXP->WGSL recompiler
        // path, but only when it is enabled (the reads + per-draw clones are pure cost on
        // the default fixed-function path). The blob size is the container total-length
        // field at header+0x08, the same idiom the GXP-bin dump uses.
        let (vprog, fprog, vert_sa, frag_sa) = if gxp_live_capture() {
            let read_blob = |hdr: u32| -> Vec<u8> {
                if hdr == 0 {
                    return Vec::new();
                }
                let sz = ctx.read_u32(hdr.wrapping_add(0x08)).clamp(0x40, 0x40000) as usize;
                ctx.read_bytes(hdr, sz)
            };
            let vprog = read_blob(self.vertex_program_header(self.bound_vertex_program));
            let fprog = read_blob(self.bound_fragment_program_header);
            // The vertex SA is the pre-stamp raw uniforms captured above (covers both the bound
            // buffer and the direct sceGxmSetUniformDataF path).
            let vert_sa = vert_sa_raw;
            let frag_sa = if self.bound_fragment_uniform_buf != 0 {
                ctx.read_bytes(self.bound_fragment_uniform_buf, self.bound_fragment_uniform_size as usize)
            } else {
                Vec::new()
            };
            (vprog, fprog, vert_sa, frag_sa)
        } else {
            (Vec::new(), Vec::new(), Vec::new(), Vec::new())
        };
        let draw = crate::capture::Draw {
            primitive,
            index_format,
            index_count,
            vertices,
            vertex_stride: stride,
            attributes,
            indices,
            uniforms,
            textures,
            render_state: self.render_state,
            exposure,
            material,
            world,
            vprog,
            fprog,
            vert_sa,
            frag_sa,
        };
        match self.scene.as_mut() {
            Some(scene) => scene.draws.push(draw),
            // A draw outside begin/endScene has nowhere to go. That is a real hole in the
            // frame, so it is logged rather than dropped in silence.
            None => tracing::debug!(target: "vitaslop::gxm", index_count, "draw outside a scene - DROPPED"),
        }
    }

    /// Size in bytes of a program's default uniform buffer, computed from its
    /// reflected parameter table: the maximum `resource_index + component_count *
    /// array_size` (in floats) over the uniform (`category == 1`) parameters, times 4.
    /// The program header's own size field (+0x2C) under-reports for shaders with a
    /// large uniform block (e.g. a world matrix at float 0 plus a world-to-projection
    /// matrix at float 16 plus lighting/fog), truncating the captured buffer and
    /// dropping the view-projection - so the reflected extent is the reliable size.
    fn reflected_uniform_size_bytes(&self, ctx: &GuestCtx, header: u32) -> u32 {
        if header == 0 {
            return 0;
        }
        let count = ctx.read_u32(header.wrapping_add(0x24));
        let base = header.wrapping_add(0x28).wrapping_add(ctx.read_u32(header.wrapping_add(0x28)));
        let mut max_floats = 0u32;
        for i in 0..count.min(256) {
            let p = base.wrapping_add(i.wrapping_mul(16));
            let word = ctx.read_u32(p.wrapping_add(4)) & 0xffff;
            if word & 0xf != 1 {
                continue; // category 1 == uniform (0 is a vertex attribute)
            }
            let comp = ((word >> 8) & 0xf).max(1);
            let array = ctx.read_u32(p.wrapping_add(8)).max(1);
            let res = ctx.read_u32(p.wrapping_add(0xc));
            max_floats = max_floats.max(res.wrapping_add(comp.wrapping_mul(array)));
        }
        max_floats.wrapping_mul(4)
    }

    /// Compose the model->projection MVP for the bound vertex program from its captured
    /// `uniforms`, using reflection to locate the model->world and world->projection
    /// matrices by their declared names. Returns `None` when the shader has no separate
    /// projection matrix (a single-transform 2D/UI shader), so the caller keeps the
    /// offset-0 matrix as-is. Both matrices are column-major 4x4 float blocks at their
    /// reflected `resource_index` (in floats); the result is `projection * world`.
    fn composed_mvp(&self, ctx: &GuestCtx, uniforms: &[f32]) -> Option<[f32; 16]> {
        let (world, proj) = self.reflected_world_proj(ctx, uniforms)?;
        // Column-major 4x4 multiply: out = proj * world.
        let mut out = [0f32; 16];
        for col in 0..4 {
            for row in 0..4 {
                let mut s = 0f32;
                for k in 0..4 {
                    s += proj[k * 4 + row] * world[col * 4 + k];
                }
                out[col * 4 + row] = s;
            }
        }
        Some(out)
    }

    /// The model-to-world matrix reflected from the vertex program's `vsModelToWorldMatrix`
    /// (column-major 4x4), or identity if the shader declares no world matrix. Used to bring
    /// the object-space vertex normal into world space for lighting.
    fn reflected_world_matrix(&self, ctx: &GuestCtx, uniforms: &[f32]) -> [f32; 16] {
        const IDENTITY: [f32; 16] =
            [1.0, 0.0, 0.0, 0.0, 0.0, 1.0, 0.0, 0.0, 0.0, 0.0, 1.0, 0.0, 0.0, 0.0, 0.0, 1.0];
        self.reflected_world_proj(ctx, uniforms).map(|(w, _)| w).unwrap_or(IDENTITY)
    }

    /// Reflect the vertex program's model->world and world->projection 4x4 matrices from its
    /// captured `uniforms`, located by their declared names. Returns `(world, proj)` or `None`
    /// when the shader has no separate projection matrix (a single-transform 2D/UI shader).
    fn reflected_world_proj(&self, ctx: &GuestCtx, uniforms: &[f32]) -> Option<([f32; 16], [f32; 16])> {
        let header = self.vertex_program_header(self.bound_vertex_program);
        if header == 0 {
            return None;
        }
        let count = ctx.read_u32(header.wrapping_add(0x24));
        let base = header.wrapping_add(0x28).wrapping_add(ctx.read_u32(header.wrapping_add(0x28)));
        let (mut world_off, mut proj_off) = (None, None);
        for i in 0..count.min(256) {
            let p = base.wrapping_add(i.wrapping_mul(16));
            let word = ctx.read_u32(p.wrapping_add(4)) & 0xffff;
            // Category 1 (uniform), a 4x4 matrix: component_count 4 and array_size 4.
            if word & 0xf != 1 || (word >> 8) & 0xf != 4 || ctx.read_u32(p.wrapping_add(8)) != 4 {
                continue;
            }
            let res = ctx.read_u32(p.wrapping_add(0xc)) as usize;
            let name_addr = (p as i64 + ctx.read_u32(p) as i32 as i64) as u32;
            let raw = ctx.read_bytes(name_addr, 48);
            let name: String = raw.iter().take_while(|&&b| b != 0).map(|&b| (b as char).to_ascii_lowercase()).collect();
            if name.contains("toprojection") || name.contains("worldtoclip") || name.contains("viewprojection") {
                proj_off = Some(res);
            } else if name.contains("modeltoworld") {
                world_off = Some(res);
            }
        }
        let (wo, po) = (world_off?, proj_off?);
        let read4x4 = |off: usize| -> Option<[f32; 16]> {
            let mut m = [0f32; 16];
            m.copy_from_slice(uniforms.get(off..off + 16)?);
            Some(m)
        };
        Some((read4x4(wo)?, read4x4(po)?))
    }

    /// Recover the scene exposure from the bound vertex program's reflected
    /// `vsCoarseExposureReg` uniform (a float4 whose first component is the linear
    /// exposure scale the shaders multiply lit albedo by before tone-mapping). Returns
    /// `1.0` when the shader declares no exposure uniform (2D/UI shaders), or when the
    /// value is not a sane positive number, so it is a safe no-op there.
    fn reflected_exposure(&self, ctx: &GuestCtx, uniforms: &[f32]) -> f32 {
        let header = self.vertex_program_header(self.bound_vertex_program);
        if header == 0 {
            return 1.0;
        }
        let count = ctx.read_u32(header.wrapping_add(0x24));
        let base = header.wrapping_add(0x28).wrapping_add(ctx.read_u32(header.wrapping_add(0x28)));
        for i in 0..count.min(256) {
            let p = base.wrapping_add(i.wrapping_mul(16));
            let word = ctx.read_u32(p.wrapping_add(4)) & 0xffff;
            if word & 0xf != 1 {
                continue; // category 1 == uniform
            }
            let name_addr = (p as i64 + ctx.read_u32(p) as i32 as i64) as u32;
            let raw = ctx.read_bytes(name_addr, 48);
            let name: String =
                raw.iter().take_while(|&&b| b != 0).map(|&b| (b as char).to_ascii_lowercase()).collect();
            if name.contains("coarseexposure") {
                let res = ctx.read_u32(p.wrapping_add(0xc)) as usize;
                if let Some(&e) = uniforms.get(res) {
                    if e.is_finite() && e > 0.0 {
                        return e;
                    }
                }
            }
        }
        1.0
    }

    /// Diagnostic (VITASLOP_DUMP_VPROG): reflect the bound vertex program's parameter
    /// table (name / category / type / component_count / container / array_size /
    /// resource_index) once per unique program, so the uniform-buffer layout (which
    /// slots are the world matrix vs the shared view-projection) is known by name.
    fn dump_vertex_program_params(&self, ctx: &GuestCtx) {
        use std::sync::Mutex;
        static SEEN: Mutex<Vec<u32>> = Mutex::new(Vec::new());
        let ph = self.vertex_program_header(self.bound_vertex_program);
        if ph == 0 {
            return;
        }
        {
            let mut seen = SEEN.lock().unwrap();
            if seen.contains(&ph) {
                return;
            }
            seen.push(ph);
        }
        let count = ctx.read_u32(ph.wrapping_add(0x24));
        let base = ph.wrapping_add(0x28).wrapping_add(ctx.read_u32(ph.wrapping_add(0x28)));
        eprintln!("VPROG header={ph:#x} params={count}");
        for i in 0..count.min(64) {
            let p = base.wrapping_add(i.wrapping_mul(16));
            let name_off = ctx.read_u32(p) as i32;
            let word = ctx.read_u32(p.wrapping_add(4)) & 0xffff;
            let array_size = ctx.read_u32(p.wrapping_add(8));
            let res_index = ctx.read_u32(p.wrapping_add(0xc));
            let name_addr = (p as i64 + name_off as i64) as u32;
            let raw = ctx.read_bytes(name_addr, 48);
            let name: String = raw.iter().take_while(|&&b| b != 0).map(|&b| b as char).collect();
            eprintln!(
                "  param[{i}] name={name:?} cat={} type={} comp={} container={} array={array_size} res_index={res_index}",
                word & 0xf, (word >> 4) & 0xf, (word >> 8) & 0xf, (word >> 12) & 0xf,
            );
        }
    }

    /// Reflect the bound fragment program's SAMPLER parameters (parameter category 2)
    /// and return the texture unit (`resource_index`) of the one that is the base-colour
    /// / albedo map, so the capture renderer - which samples a single texture - picks the
    /// diffuse colour rather than whatever happens to sit at unit 0 (this title binds the
    /// NORMAL map at unit 0, so a naive unit-0 pick paints every surface flat blue).
    ///
    /// Selection is by the sampler's declared name: a diffuse/albedo/base-colour name
    /// wins; names that are clearly a different map role (normal, specular, environment/
    /// cube, reflection, gloss, light, shadow, detail, bump, height, mask, ao) are
    /// rejected. Returns `None` when there is no fragment program or no sampler reads as
    /// an albedo, leaving the caller on its default unit-0 pick.
    fn fragment_albedo_unit(&self, ctx: &GuestCtx) -> Option<u32> {
        let header = self.bound_fragment_program_header;
        if header == 0 {
            return None;
        }
        let count = ctx.read_u32(header.wrapping_add(0x24));
        let base = header.wrapping_add(0x28).wrapping_add(ctx.read_u32(header.wrapping_add(0x28)));
        let mut best: Option<(i32, u32)> = None; // (score, unit)
        for i in 0..count.min(256) {
            let p = base.wrapping_add(i.wrapping_mul(16));
            let word = ctx.read_u32(p.wrapping_add(4)) & 0xffff;
            if word & 0xf != 2 {
                continue; // category 2 == sampler
            }
            let unit = ctx.read_u32(p.wrapping_add(0xc));
            let name_addr = (p as i64 + ctx.read_u32(p) as i32 as i64) as u32;
            let raw = ctx.read_bytes(name_addr, 64);
            let name: String =
                raw.iter().take_while(|&&b| b != 0).map(|&b| (b as char).to_ascii_lowercase()).collect();
            let score = albedo_name_score(&name);
            if score > 0 && best.map(|(s, _)| score > s).unwrap_or(true) {
                best = Some((score, unit));
            }
        }
        best.map(|(_, u)| u)
    }

    /// Reflect the bound fragment program's per-material inputs (base-colour tint, the
    /// directional light, and a flat ambient term) from its parameter table read against the
    /// captured fragment default uniform buffer. The material model is uniform across this
    /// engine's shaders (verified empirically): a `*tint*`/`*albedocolour*` float3 multiplies
    /// the sampled albedo, and `directionalLight0Colour` * saturate(N.L) + ambient lights it.
    ///
    /// Uniform packing (also verified empirically against the real shaders): a parameter's
    /// `resource_index` is a 4-byte-register offset into the default uniform buffer; each of
    /// its `component_count` components is an F16 (`type == 1`, 2 bytes) or an F32 (`type ==
    /// 0`, 4 bytes). `textures` is the draw's bound textures (albedo reordered to the front),
    /// used to resolve the ambient colour from the tiny `diffuseAmbientMap` irradiance map.
    fn reflect_fragment_material(&self, ctx: &GuestCtx, textures: &[crate::capture::BoundTexture]) -> crate::capture::FragmentMaterial {
        let mut m = crate::capture::FragmentMaterial::default();
        let header = self.bound_fragment_program_header;
        let buf = self.bound_fragment_uniform_buf;
        if header != 0 && buf != 0 {
            let count = ctx.read_u32(header.wrapping_add(0x24));
            let base = header.wrapping_add(0x28).wrapping_add(ctx.read_u32(header.wrapping_add(0x28)));
            // Read `comp` scalar components of a uniform parameter from the fragment default
            // uniform buffer at its register offset, honouring the F16/F32 component type.
            let read_vals = |res_index: u32, comp: usize, is_f16: bool| -> Vec<f32> {
                let byte_off = buf.wrapping_add(res_index.wrapping_mul(4));
                (0..comp)
                    .map(|i| {
                        if is_f16 {
                            crate::render::half_to_f32(ctx.read_u16(byte_off.wrapping_add((i as u32) * 2)))
                        } else {
                            ctx.read_f32(byte_off.wrapping_add((i as u32) * 4))
                        }
                    })
                    .collect()
            };
            for i in 0..count.min(256) {
                let p = base.wrapping_add(i.wrapping_mul(16));
                let word = ctx.read_u32(p.wrapping_add(4)) & 0xffff;
                if word & 0xf != 1 {
                    continue; // category 1 == uniform
                }
                let ptype = (word >> 4) & 0xf;
                let comp = ((word >> 8) & 0xf).max(1) as usize;
                let res = ctx.read_u32(p.wrapping_add(0xc));
                let name_addr = (p as i64 + ctx.read_u32(p) as i32 as i64) as u32;
                let raw = ctx.read_bytes(name_addr, 64);
                let name: String =
                    raw.iter().take_while(|&&b| b != 0).map(|&b| (b as char).to_ascii_lowercase()).collect();
                let is_f16 = ptype == 1;
                let take3 = |v: &[f32]| [*v.first().unwrap_or(&0.0), *v.get(1).unwrap_or(&0.0), *v.get(2).unwrap_or(&0.0)];
                // The base-colour tint: the primary layer's tint (or a wheel's AlbedoColour).
                // A "secondary" tint belongs to a second material layer we do not composite,
                // so only the primary/albedo tint drives the base colour.
                if (name.contains("albedocolour") || name.contains("albedocolor") || name.contains("primarytint"))
                    && comp >= 3
                {
                    let t = take3(&read_vals(res, 3, is_f16));
                    if t.iter().all(|c| c.is_finite() && *c >= 0.0 && *c <= 8.0) {
                        m.tint = t;
                    }
                } else if name.contains("light0direction") && comp >= 3 {
                    let d = take3(&read_vals(res, 3, is_f16));
                    if d.iter().all(|c| c.is_finite()) && d.iter().any(|c| *c != 0.0) {
                        m.light_dir = d;
                        m.has_light = true;
                    }
                } else if name.contains("light0colour") || name.contains("light0color") {
                    if comp >= 3 {
                        let c = take3(&read_vals(res, 3, is_f16));
                        if c.iter().all(|v| v.is_finite() && *v >= 0.0 && *v <= 16.0) {
                            m.light_col = c;
                        }
                    }
                }
            }
        }
        // Ambient: the average colour of the small `diffuseAmbientMap` irradiance texture, if
        // one is bound. It is a coarse (16x16 / 128x128) light probe, so its mean is a good
        // flat ambient for a renderer that does not sample it per-normal. Leaves the default
        // grey ambient when no such map is present.
        if let Some(amb) = textures.iter().find(|t| t.unit == 15) {
            if let Some(mean) = crate::render::texture_mean_rgb(amb) {
                m.ambient = mean;
            }
        }
        m
    }

    /// Diagnostic (VITASLOP_DUMP_FPROG): print the bound fragment program's sampler
    /// parameters (name + unit) once per unique program, so the albedo-selection name
    /// matching can be grounded in the real sampler names a title declares.
    fn dump_fragment_program_samplers(&self, ctx: &GuestCtx) {
        use std::sync::Mutex;
        static SEEN: Mutex<Vec<u32>> = Mutex::new(Vec::new());
        let header = self.bound_fragment_program_header;
        if header == 0 {
            return;
        }
        {
            let mut seen = SEEN.lock().unwrap();
            if seen.contains(&header) {
                return;
            }
            seen.push(header);
        }
        let count = ctx.read_u32(header.wrapping_add(0x24));
        let base = header.wrapping_add(0x28).wrapping_add(ctx.read_u32(header.wrapping_add(0x28)));
        eprintln!("FPROG header={header:#x} params={count}");
        for i in 0..count.min(256) {
            let p = base.wrapping_add(i.wrapping_mul(16));
            let word = ctx.read_u32(p.wrapping_add(4)) & 0xffff;
            if word & 0xf != 2 {
                continue;
            }
            let unit = ctx.read_u32(p.wrapping_add(0xc));
            let name_addr = (p as i64 + ctx.read_u32(p) as i32 as i64) as u32;
            let raw = ctx.read_bytes(name_addr, 64);
            let name: String = raw.iter().take_while(|&&b| b != 0).map(|&b| b as char).collect();
            eprintln!("  sampler name={name:?} unit={unit} score={}", albedo_name_score(&name.to_ascii_lowercase()));
        }
    }

    /// Diagnostic (VITASLOP_DUMP_DRAW_GXP=<frame>): dump, per draw of the matching
    /// display frame, everything needed to reason about the fragment-program compositing
    /// the capture renderer cannot yet reproduce - the vertex attribute layout (how many
    /// texcoord SETS the mesh carries), the bound fragment program's FULL parameter table
    /// (samplers with their unit, and the fragment INPUT varyings / category-0 attributes
    /// that tell which TEXCOORD each stage reads), and the bound textures with the sampler
    /// name at each unit. This is the empirical instrument for the wheel-ring artifact: it
    /// shows whether a draw feeds one UV set to several samplers (a composite we must
    /// reproduce) or picks the wrong UV set for a sampler (a binding we can correct).
    fn dump_draw_gxp(&self, ctx: &GuestCtx, textures: &[crate::capture::BoundTexture], attributes: &[crate::capture::VertexAttribute], primitive: u32, index_count: u32, stride: u32) {
        let want = match std::env::var("VITASLOP_DUMP_DRAW_GXP").ok() {
            Some(s) => s,
            None => return,
        };
        // The frame the dump keys on is the scheduler frame (`cur_frame`, the display-flip /
        // yield count set at each `on_frame_boundary`), which is exactly the `fNNNN` a shot is
        // labelled with in the recipe runner. (`presents.len()` is NOT it: this title renders
        // through the GXM display queue and calls the raw `present` path only a couple of times.)
        let disp = self.cur_frame;
        // "all", a single frame "N", or an inclusive range "LO-HI".
        let match_frame = want == "all"
            || match want.split_once('-') {
                Some((lo, hi)) => match (lo.parse::<u64>(), hi.parse::<u64>()) {
                    (Ok(lo), Ok(hi)) => (lo..=hi).contains(&disp),
                    _ => false,
                },
                None => want.parse::<u64>().map(|f| f == disp).unwrap_or(false),
            };
        if !match_frame {
            return;
        }
        // Bound the total lines dumped (a wide window over a 90-draws/frame scene is a lot of
        // output); `VITASLOP_DUMP_DRAW_GXP_CAP` raises it when a full multi-frame trace is wanted.
        let cap = std::env::var("VITASLOP_DUMP_DRAW_GXP_CAP")
            .ok()
            .and_then(|s| s.parse::<u32>().ok())
            .unwrap_or(400);
        use std::sync::atomic::{AtomicU32, Ordering};
        static DUMPED: AtomicU32 = AtomicU32::new(0);
        if DUMPED.fetch_add(1, Ordering::Relaxed) >= cap {
            return;
        }
        let seq = self.scene.as_ref().map(|s| s.draws.len()).unwrap_or(0);
        let vh = self.vertex_program_header(self.bound_vertex_program);
        let fh = self.bound_fragment_program_header;
        let mat = self.reflect_fragment_material(ctx, textures);
        eprintln!(
            "DRAW frame={disp} seq={seq} prim={primitive:#010x} idx={index_count} stride={stride} vprog={vh:#x} fprog={fh:#x} fubuf={:#x}",
            self.bound_fragment_uniform_buf
        );
        eprintln!(
            "  MATERIAL tint=[{:.3},{:.3},{:.3}] has_light={} light_dir=[{:.3},{:.3},{:.3}] light_col=[{:.3},{:.3},{:.3}] ambient=[{:.3},{:.3},{:.3}]",
            mat.tint[0], mat.tint[1], mat.tint[2], mat.has_light,
            mat.light_dir[0], mat.light_dir[1], mat.light_dir[2],
            mat.light_col[0], mat.light_col[1], mat.light_col[2],
            mat.ambient[0], mat.ambient[1], mat.ambient[2],
        );
        // Vertex stream attributes (bytes -> shader register): the count of distinct UV-ish
        // float2/float4 lanes is the number of texcoord sets the mesh actually carries.
        for a in attributes {
            eprintln!("  attr stream={} off={} fmt={} comp={} reg={}", a.stream_index, a.offset, a.format, a.component_count, a.reg_index);
        }
        // Sampler name at a given unit, by reflecting the fragment program.
        let sampler_name = |unit: u32| -> String { self.gxp_sampler_name(ctx, fh, unit) };
        for t in textures {
            // Fraction of the snapshotted bytes that are non-zero. A render-target texture the
            // engine never rendered into (a shadow map, a reflection probe) reads as an all-zero
            // buffer, and a shader multiplying by it paints only its ambient term - which looks
            // like a lighting bug rather than a missing render pass unless this is measured.
            let nonzero = if t.pixels.is_empty() {
                0.0
            } else {
                t.pixels.iter().filter(|&&b| b != 0).count() as f32 / t.pixels.len() as f32
            };
            eprintln!(
                "  tex unit={} {}x{} base_fmt={:#x} swizzle={:#x} type={} nonzero={nonzero:.3} sampler={:?}",
                t.unit, t.width, t.height, t.base_format, t.swizzle, t.tex_type, sampler_name(t.unit)
            );
        }
        if std::env::var("VITASLOP_DUMP_DRAW_GXP_FULL").is_ok() {
            eprintln!("  -- fragment program params --");
            self.dump_all_params(ctx, fh);
            eprintln!("  -- vertex program params --");
            self.dump_all_params(ctx, vh);
        }
        // VITASLOP_DUMP_TEX_DIR=<dir>: also write each bound texture, decoded to RGBA8 via the
        // exact sampler decode, as a PNG named `f<frame>_seq<seq>_u<unit>_<sampler>.png`. Lets a
        // human see the ACTUAL texel content a draw sampled (is the sampled albedo the paint, or
        // a decal atlas whose UVs the shader would remap?) - the instrument for deciding whether
        // a render artifact is a decode/pick bug or the untranslated fragment-program composite.
        if let Ok(dir) = std::env::var("VITASLOP_DUMP_TEX_DIR") {
            use std::sync::Mutex;
            // De-dup by guest data address: the shared shadow/ambient/atlas maps are bound in
            // every draw, so writing them per-draw would emit gigabytes. Each unique texture is
            // written once. Maps above `VITASLOP_DUMP_TEX_MAX_TEXELS` (default 1 Mtexel) are
            // skipped - the material inputs usually worth eyeballing are the small per-part
            // albedo/normal sheets, but raising the cap is how you inspect a render-target map
            // such as the 4096x2048 shadMap when the question is whether it holds real content.
            static SEEN: Mutex<Vec<u32>> = Mutex::new(Vec::new());
            let max_texels = std::env::var("VITASLOP_DUMP_TEX_MAX_TEXELS")
                .ok()
                .and_then(|v| v.parse::<u32>().ok())
                .unwrap_or(1 << 20);
            let _ = std::fs::create_dir_all(&dir);
            for t in textures {
                if t.width == 0 || t.height == 0 || t.width * t.height > max_texels {
                    continue;
                }
                {
                    let mut seen = SEEN.lock().unwrap();
                    if seen.contains(&t.data_addr) {
                        continue;
                    }
                    seen.push(t.data_addr);
                }
                let (w, h, rgba) = crate::render::decode_texture_rgba8(t);
                let samp = self.gxp_sampler_name(ctx, fh, t.unit);
                let safe: String = samp.chars().filter(|c| c.is_ascii_alphanumeric()).collect();
                let path = std::path::Path::new(&dir)
                    .join(format!("f{disp}_seq{seq}_u{}_{}.png", t.unit, safe));
                let _ = std::fs::write(path, crate::render::rgba_to_png(w, h, &rgba));
            }
        }
        // VITASLOP_DUMP_GXP_BIN=<dir>: write the raw SceGxmProgram blobs (the whole container -
        // header + parameter table + USSE bytecode) for the bound fragment and vertex programs,
        // named `<type>_<header-addr>.gxp`, deduped by address. These are the durable artifacts
        // the clean-room GXP->WGSL shader recompiler decodes. `SceGxmProgram.size` is the u32 at
        // header+0x08 (the container's total byte length; clamped defensively).
        if let Ok(dir) = std::env::var("VITASLOP_DUMP_GXP_BIN") {
            use std::sync::Mutex;
            static SEEN: Mutex<Vec<u32>> = Mutex::new(Vec::new());
            let _ = std::fs::create_dir_all(&dir);
            for (kind, header) in [("frag", fh), ("vert", vh)] {
                if header == 0 {
                    continue;
                }
                {
                    let mut seen = SEEN.lock().unwrap();
                    if seen.contains(&header) {
                        continue;
                    }
                    seen.push(header);
                }
                let size = ctx.read_u32(header.wrapping_add(0x08)).clamp(0x40, 0x40000);
                let bytes = ctx.read_bytes(header, size as usize);
                let path = std::path::Path::new(&dir).join(format!("{kind}_{header:08x}.gxp"));
                let _ = std::fs::write(path, &bytes);
            }
        }

        // VITASLOP_GXP_RECOMPILE=1: the explicit clean-room shader-recompiler GRIND mode.
        // For each unique bound fragment AND vertex program, read its container from guest
        // memory and recompile the guest USSE to a complete, bindable WGSL module. This is NOT
        // a silent fixed-function fallback: on any unsupported opcode / decode failure it
        // HARD-FAILS (panics) naming the exact instruction + opcode to implement next, exactly
        // like the NID grind. The ordinary renderer runs unchanged when this is unset (the
        // recompiler is simply not engaged - a separate default renderer, not a fallback here).
        if std::env::var_os("VITASLOP_GXP_RECOMPILE").is_some() {
            use std::sync::Mutex;
            static SEEN: Mutex<Vec<u32>> = Mutex::new(Vec::new());
            for (kind, header) in [("frag", fh), ("vert", vh)] {
                if header == 0 {
                    continue;
                }
                let first = {
                    let mut seen = SEEN.lock().unwrap();
                    if seen.contains(&header) {
                        false
                    } else {
                        seen.push(header);
                        true
                    }
                };
                if !first {
                    continue;
                }
                let size = ctx.read_u32(header.wrapping_add(0x08)).clamp(0x40, 0x40000);
                let bytes = ctx.read_bytes(header, size as usize);
                match kind {
                    "frag" => match vitaslop_gxp_shader::recompile_fragment_module(&bytes) {
                        Ok((r, m)) => tracing::info!(
                            target: "vitaslop::gxp",
                            header = format_args!("{header:#010x}"),
                            instrs = r.shader.instrs.len(),
                            pa = m.bindings.pa_lane_count,
                            sa = m.bindings.sa_lane_count,
                            samplers = m.bindings.samplers.len(),
                            color = format_args!("{:?}", m.bindings.color),
                            "recompiled FRAGMENT program to a bindable WGSL module ({} chars)",
                            m.wgsl.len(),
                        ),
                        Err(e) => panic!("VITASLOP_GXP_RECOMPILE: fragment program {header:#010x}: {e}"),
                    },
                    _ => match vitaslop_gxp_shader::recompile_vertex_module(&bytes) {
                        Ok((r, m)) => tracing::info!(
                            target: "vitaslop::gxp",
                            header = format_args!("{header:#010x}"),
                            instrs = r.shader.instrs.len(),
                            attributes = m.bindings.attributes.len(),
                            sa = m.bindings.sa_lane_count,
                            varyings = m.bindings.varying_vec4s,
                            "recompiled VERTEX program to a bindable WGSL module ({} chars)",
                            m.wgsl.len(),
                        ),
                        Err(e) => panic!("VITASLOP_GXP_RECOMPILE: vertex program {header:#010x}: {e}"),
                    },
                }
            }
        }
    }

    /// Reflect the name of the fragment sampler bound to texture `unit`, or "" if none.
    fn gxp_sampler_name(&self, ctx: &GuestCtx, header: u32, unit: u32) -> String {
        if header == 0 {
            return String::new();
        }
        let count = ctx.read_u32(header.wrapping_add(0x24));
        let base = header.wrapping_add(0x28).wrapping_add(ctx.read_u32(header.wrapping_add(0x28)));
        for i in 0..count.min(256) {
            let p = base.wrapping_add(i.wrapping_mul(16));
            let word = ctx.read_u32(p.wrapping_add(4)) & 0xffff;
            if word & 0xf != 2 {
                continue;
            }
            if ctx.read_u32(p.wrapping_add(0xc)) == unit {
                let name_addr = (p as i64 + ctx.read_u32(p) as i32 as i64) as u32;
                let raw = ctx.read_bytes(name_addr, 64);
                return raw.iter().take_while(|&&b| b != 0).map(|&b| b as char).collect();
            }
        }
        String::new()
    }

    /// Dump every parameter (all categories) of a GXP program: category / type /
    /// component_count / container_index / array_size / resource_index / name. Category
    /// 0 = attribute (a fragment program's INPUT varyings, e.g. TEXCOORD0/1), 1 = uniform,
    /// 2 = sampler, 4 = uniform buffer.
    fn dump_all_params(&self, ctx: &GuestCtx, header: u32) {
        if header == 0 {
            eprintln!("    (no program)");
            return;
        }
        let count = ctx.read_u32(header.wrapping_add(0x24));
        let base = header.wrapping_add(0x28).wrapping_add(ctx.read_u32(header.wrapping_add(0x28)));
        for i in 0..count.min(128) {
            let p = base.wrapping_add(i.wrapping_mul(16));
            let word = ctx.read_u32(p.wrapping_add(4)) & 0xffff;
            let array_size = ctx.read_u32(p.wrapping_add(8));
            let res_index = ctx.read_u32(p.wrapping_add(0xc));
            let name_addr = (p as i64 + ctx.read_u32(p) as i32 as i64) as u32;
            let raw = ctx.read_bytes(name_addr, 64);
            let name: String = raw.iter().take_while(|&&b| b != 0).map(|&b| b as char).collect();
            let cat = match word & 0xf {
                0 => "attr",
                1 => "unif",
                2 => "samp",
                4 => "ubuf",
                _ => "?",
            };
            eprintln!(
                "    [{i}] {cat} type={} comp={} container={} array={array_size} res={res_index} name={name:?}",
                (word >> 4) & 0xf, (word >> 8) & 0xf, (word >> 12) & 0xf,
            );
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

/// Score a fragment-sampler name (already lowercased) for how strongly it reads as the
/// albedo / base-colour map. Positive = a diffuse colour source (higher is a better
/// match); 0 = not an albedo, or a name that clearly names a different map role
/// (normal/spec/env/etc). Used to pick the one texture the capture renderer samples.
fn albedo_name_score(name: &str) -> i32 {
    // A name that clearly identifies a non-colour map role is never the albedo, even if
    // it also contains a generic "map"/"tex" token.
    const NON_ALBEDO: [&str; 14] = [
        "normal", "spec", "gloss", "rough", "env", "cube", "reflect", "light", "shadow",
        "detail", "bump", "height", "mask", "emiss",
    ];
    if NON_ALBEDO.iter().any(|k| name.contains(k)) {
        return 0;
    }
    // Explicit albedo names rank highest; generic colour names next; a bare "diffuse"
    // fragment (some engines name the sole colour sampler just "tex"/"map") lowest.
    //
    // NOTE (empirical, this title): an `AlbedoTexture`/`LiveryAlbedo` sampler is this
    // engine's base colour for a body panel (the blue livery) BUT is also the livery/decal
    // SOURCE sheet on other parts (a gauge/number sheet that, sampled whole at a tiling UV,
    // paints white rings on the wheels). Both use the identical sampler name and role - the
    // only difference is the mesh's UV mapping, which the (untranslated) fragment program
    // resolves. So this name-based pick CANNOT separate them; ranking diffuse over albedo
    // to kill the wheel rings also flattens the body's livery (a worse regression). The
    // rings are left as a known artifact until the GXP fragment/vertex texcoord binding is
    // reflected (or the fragment program is translated) - do not "fix" it by name score.
    for (kw, score) in [
        ("albedo", 100),
        ("basecolor", 95),
        ("basecolour", 95),
        ("diffuse", 90),
        ("diff", 70),
        ("colour", 60),
        ("color", 60),
        ("base", 40),
    ] {
        if name.contains(kw) {
            return score;
        }
    }
    0
}

/// Size in bytes of a `SceGxmProgram`'s default uniform buffer, from the container's
/// `default_uniform_buffer_count` field (header +0x64), which counts 32-bit SA registers.
///
/// The field at +0x2C that this previously read is the varyings-block offset, not a size: it
/// reads as a fixed 108 on every program in a title, so it neither bounded the buffer nor
/// tracked its real extent. Clamped so a header we failed to resolve cannot request an absurd
/// allocation.
///
/// This is also what `sceGxmProgramGetDefaultUniformBufferSize` must hand the GUEST: a title
/// uses that size as the length of the block it writes into the buffer it reserved, so a
/// wrong answer truncates its own uniform upload (see [`crate::vita::gxm`]).
pub(crate) fn default_uniform_buffer_bytes(ctx: &GuestCtx, header: u32) -> u32 {
    if header == 0 {
        return 0;
    }
    ctx.read_u32(header.wrapping_add(0x64)).min(4096).wrapping_mul(4)
}

/// Diagnostic (`VITASLOP_GXM_UNIFORM_POISON=1`): fill a freshly reserved default uniform buffer
/// with a recognisable bit pattern instead of leaving it zeroed.
///
/// A zeroed buffer cannot distinguish "the guest wrote 0.0 here" from "the guest never wrote
/// here at all", and those have completely different causes: the first is real data, the second
/// means the value reaches the shader by a path we do not model. The poison is a quiet NaN
/// (`0x7fc0dead`), so any lane still holding it at draw time was never written - and a shader
/// consuming it produces NaN rather than a plausible-looking black, which is itself the signal.
fn poison_uniform_buffer(ctx: &mut GuestCtx, buf: u32, size: u32) {
    use std::sync::OnceLock;
    static ON: OnceLock<bool> = OnceLock::new();
    if buf == 0 || !*ON.get_or_init(|| std::env::var_os("VITASLOP_GXM_UNIFORM_POISON").is_some()) {
        return;
    }
    for i in 0..size / 4 {
        ctx.write_u32(buf + i * 4, 0x7fc0_dead);
    }
}

/// Whether the GXP->WGSL recompiler capture path is enabled (env `VITASLOP_GXP_LIVE`).
/// Checked once and cached, so the per-draw `record_draw` gate is a cheap load rather than
/// an environment lookup per draw.
fn gxp_live_capture() -> bool {
    use std::sync::OnceLock;
    static ENABLED: OnceLock<bool> = OnceLock::new();
    *ENABLED.get_or_init(|| std::env::var_os("VITASLOP_GXP_LIVE").is_some())
}

/// Decode a bound `SceGxmTexture` (16 bytes, 4 control words) from guest memory
/// and snapshot its pixel bytes. Returns `None` for a null/unreadable handle or a
/// format whose byte size we do not know yet. The layout is the public GXM texture
/// control-word format (vitasdk `gxm.h` `struct SceGxmTexture`): word 1 holds
/// width/height (stored as size-1) and the base format, word 2 the data address.
fn decode_texture(
    ctx: &GuestCtx,
    cache: &mut TextureSnapshots,
    unit: u32,
    addr: u32,
    exact_format: Option<u32>,
    sampler: (u32, u32, u32),
    filters: (u32, u32),
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

    // Block geometry: uncompressed formats are 1x1 texel "blocks"; BC/DXT are 4x4. A format we
    // cannot size is dropped rather than guessed, but never silently: an undecoded unit shows up
    // downstream only as a missing sampler binding, which is far harder to trace back than the
    // format that caused it.
    let Some((block_w, block_h, block_bytes)) = crate::render::block_layout(base_format) else {
        tracing::debug!(
            target: "vitaslop::render",
            unit,
            base_format = format_args!("{base_format:#04x}"),
            tex_type,
            width,
            height,
            "texture format not sized - unit left unbound",
        );
        return None;
    };
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
    // A CUBE texture stores its six faces back to back, each laid out exactly like a standalone
    // texture of the same size - so one face is `total` bytes and the snapshot is six of them.
    let faces = if crate::render::cube_type(tex_type) { 6 } else { 1 };
    // Read each distinct (address, size) once per scene and share the bytes with every
    // draw that binds it. A scene binds a handful of textures across hundreds of draws, so
    // re-reading them per draw is the difference between megabytes and gigabytes. The cache
    // is scene-scoped because a render target is written in one scene and sampled in a
    // later one - within a single scene the guest cannot rewrite a texture the GPU is
    // consuming, so the snapshot cannot go stale.
    let len = (total * faces) as usize;
    let pixels = match cache.get(&(data_addr, len)) {
        Some(p) => p.clone(),
        None => {
            let bytes: Arc<[u8]> = ctx.read_bytes(data_addr, len).into();
            cache.insert((data_addr, len), bytes.clone());
            bytes
        }
    };
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
        faces,
        face_bytes: total,
        data_addr,
        pixels,
        u_addr_mode: sampler.0,
        v_addr_mode: sampler.1,
        lod_bias: sampler.2,
        min_filter: filters.0,
        mag_filter: filters.1,
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

    /// The thread pointer (`TPIDRURO`) for thread `thid`: the base of its private
    /// thread-local-storage block, allocated on first request. The engine reads it
    /// when it instantiates the thread and seeds the guest `tp` register, so a
    /// `MRC p15,0,Rt,c13,c0,3` reads a per-thread pointer. Also returns the TLS init
    /// image `(source, len)` to copy into the block's `.tdata` head (`len == 0` for a
    /// pure-`.tbss` template). Default `(0, 0, 0)`: a host with no TLS model.
    fn thread_tls_base(&mut self, _thid: i32) -> (u32, u32, u32) {
        (0, 0, 0)
    }

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

    /// The `r0` value owed to thread `thid` as it resumes from a block (a timed wait
    /// that expired, returning `SCE_KERNEL_ERROR_WAIT_TIMEOUT`), if any. The engine
    /// calls this at the point the woken thread resumes and, on `Some`, overwrites its
    /// `r0`; `None` (a signal wake) leaves the pre-park return value (0) in place.
    fn take_resume_code(&mut self, _thid: i32) -> Option<u32> {
        None
    }

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
        let (library_nid, func_nid) = self
            .imports
            .get(index as usize)
            .copied()
            .unwrap_or((0, 0));
        self.state.capture.record_call(func_nid, self.state.current);
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

    fn thread_tls_base(&mut self, thid: i32) -> (u32, u32, u32) {
        let base = self.state.ensure_tls_block(thid);
        let (src, len) = self.state.tls_init_image();
        (base, src, len)
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

    fn take_resume_code(&mut self, thid: i32) -> Option<u32> {
        self.state.take_resume_code(thid)
    }

    fn on_frame_boundary(&mut self, frame: u64) {
        self.state.world.set_frame(frame);
        self.state.set_cur_frame(frame);
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

    fn thread_tls_base(&mut self, thid: i32) -> (u32, u32, u32) {
        self.borrow_mut().thread_tls_base(thid)
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

    fn take_resume_code(&mut self, thid: i32) -> Option<u32> {
        self.borrow_mut().take_resume_code(thid)
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
mod savedata_tests {
    //! The in-memory savedata slot store: a created slot must round-trip on get, be
    //! namespaced per mount, and delete cleanly - the read-after-write faithfulness
    //! the SceAppUtil slot API depends on.
    use super::*;
    use crate::world::DeterministicWorld;

    fn dummy_state() -> VitaState {
        VitaState::new(0, 64, Box::new(DeterministicWorld::default()))
    }

    #[test]
    fn create_then_get_round_trips() {
        let mut st = dummy_state();
        assert!(!st.savedata_slot_exists("savedata0:", 0));
        assert_eq!(st.savedata_slot_get("savedata0:", 0), None);

        let param = vec![0xABu8; 0x34C];
        st.savedata_slot_put("savedata0:", 0, param.clone());

        assert!(st.savedata_slot_exists("savedata0:", 0));
        assert_eq!(st.savedata_slot_get("savedata0:", 0).as_deref(), Some(param.as_slice()));
    }

    #[test]
    fn slots_are_namespaced_by_mount_and_id() {
        let mut st = dummy_state();
        st.savedata_slot_put("savedata0:", 0, vec![1]);
        // A different slot id or mount is a distinct, still-empty slot.
        assert!(!st.savedata_slot_exists("savedata0:", 1));
        assert!(!st.savedata_slot_exists("savedata1:", 0));
        assert_eq!(st.savedata_slot_get("savedata0:", 0).as_deref(), Some([1].as_slice()));
    }

    #[test]
    fn remove_reports_prior_existence() {
        let mut st = dummy_state();
        st.savedata_slot_put("", 3, vec![9, 9]);
        assert!(st.savedata_slot_remove("", 3));
        assert!(!st.savedata_slot_remove("", 3)); // already gone
        assert!(!st.savedata_slot_exists("", 3));
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
        st.sema_block(sem, 1, 0);
        assert!(st.take_wakes().is_empty());
        // Thread 2 signals: thread 1 is released and the token is consumed for it.
        st.set_current(2);
        st.sema_signal_wake(sem, 1);
        assert_eq!(st.take_wakes(), vec![1]);
        // The count was consumed by the wake, so a fresh acquire would block again.
        assert!(!st.sema_try_acquire(sem, 1));
    }

    #[test]
    fn event_flag_parks_then_a_set_wakes_with_the_match_pattern() {
        let mut st = state();
        let ev = st.create_event_flag(0);
        // Thread 1 waits (AND) for bits 0x3: unsatisfied, parks.
        st.set_current(1);
        assert_eq!(st.evf_try_wait(ev, 0x3, 0), None);
        st.evf_block(ev, 0x3, 0, 0x2000, 0);
        assert!(st.take_wakes().is_empty());
        // Setting only bit 0 does not satisfy an AND wait for 0x3.
        st.set_current(2);
        st.event_set_wake(ev, 0x1);
        assert!(st.take_wakes().is_empty());
        // Completing the pattern wakes thread 1 and delivers 0x3 to its outBits.
        st.event_set_wake(ev, 0x2);
        assert_eq!(st.take_wakes(), vec![1]);
        assert_eq!(st.take_stat_writes(), vec![(0x2000, 0x3)]);
    }

    #[test]
    fn event_flag_wait_modes_or_and_clear() {
        let mut st = state();
        let ev = st.create_event_flag(0b110);
        // OR: any requested bit satisfies; no clear op leaves the pattern intact.
        assert_eq!(st.evf_try_wait(ev, 0b010, 1), Some(0b110));
        assert_eq!(st.event_pattern(ev), 0b110);
        // WAITCLEAR_PAT clears only the requested bits at match.
        assert_eq!(st.evf_try_wait(ev, 0b100, 1 | 4), Some(0b110));
        assert_eq!(st.event_pattern(ev), 0b010);
        // WAITCLEAR zeroes the whole pattern at match.
        assert_eq!(st.evf_try_wait(ev, 0b010, 1 | 2), Some(0b010));
        assert_eq!(st.event_pattern(ev), 0);
        // AND on a partial pattern does not match.
        st.event_set(ev, 0b01);
        assert_eq!(st.evf_try_wait(ev, 0b11, 0), None);
    }

    #[test]
    fn timed_event_flag_wait_wakes_at_its_deadline() {
        let mut st = state();
        let ev = st.create_event_flag(0);
        st.set_current(1);
        st.evf_block(ev, 0x1, 0, 0x3000, 500);
        assert_eq!(st.earliest_lwcond_deadline(), Some(500));
        // Short of the deadline: still parked.
        st.advance_time_to(499);
        assert!(st.take_wakes().is_empty());
        // At the deadline: woken with the (still unset) current pattern delivered.
        st.advance_time_to(500);
        assert_eq!(st.take_wakes(), vec![1]);
        assert_eq!(st.take_stat_writes(), vec![(0x3000, 0)]);
        // The expired waiter is gone: a later set wakes no one.
        st.event_set_wake(ev, 0x1);
        assert!(st.take_wakes().is_empty());
    }

    #[test]
    fn multiple_semaphore_waiters_release_fifo() {
        let mut st = state();
        let sem = st.create_sema(0);
        // Threads 1, 2, 3 park in order, each needing one permit.
        for t in [1, 2, 3] {
            st.set_current(t);
            assert!(!st.sema_try_acquire(sem, 1));
            st.sema_block(sem, 1, 0);
        }
        // Three single posts release them in FIFO (block) order, one permit each.
        st.set_current(4);
        st.sema_signal_wake(sem, 1);
        st.sema_signal_wake(sem, 1);
        st.sema_signal_wake(sem, 1);
        assert_eq!(st.take_wakes(), vec![1, 2, 3]);
    }

    #[test]
    fn semaphore_need_greater_than_one_waits_for_accumulation() {
        let mut st = state();
        let sem = st.create_sema(0);
        st.set_current(1);
        st.sema_block(sem, 3, 0); // needs 3
        st.set_current(2);
        st.sema_signal_wake(sem, 2); // count 2 < 3: not released
        assert!(st.take_wakes().is_empty());
        st.sema_signal_wake(sem, 1); // count 3: released, consuming all 3
        assert_eq!(st.take_wakes(), vec![1]);
        assert!(!st.sema_try_acquire(sem, 1), "the wake consumed the whole count");
    }

    #[test]
    fn timed_semaphore_wait_times_out_with_wait_timeout_code() {
        let mut st = state();
        let sem = st.create_sema(0);
        st.set_current(1);
        assert!(!st.sema_try_acquire(sem, 1));
        st.sema_block(sem, 1, 500);
        // The semaphore deadline participates in the scheduler's clock jump.
        assert_eq!(st.earliest_lwcond_deadline(), Some(500));
        // Short of the deadline: still parked, nothing owed.
        st.advance_time_to(499);
        assert!(st.take_wakes().is_empty());
        assert_eq!(st.take_resume_code(1), None);
        // At the deadline: woken and owed SCE_KERNEL_ERROR_WAIT_TIMEOUT for its r0.
        st.advance_time_to(500);
        assert_eq!(st.take_wakes(), vec![1]);
        assert_eq!(st.take_resume_code(1), Some(SCE_KERNEL_ERROR_WAIT_TIMEOUT));
        assert_eq!(st.take_resume_code(1), None, "the code is drained once");
    }

    #[test]
    fn timed_semaphore_wait_satisfied_before_deadline_returns_zero() {
        let mut st = state();
        let sem = st.create_sema(0);
        st.set_current(1);
        st.sema_block(sem, 1, 500);
        // A signal before the deadline releases it with NO resume code (r0 stays 0).
        st.set_current(2);
        st.sema_signal_wake(sem, 1);
        assert_eq!(st.take_wakes(), vec![1]);
        assert_eq!(st.take_resume_code(1), None);
        // Its deadline no longer holds the scheduler's clock.
        assert_eq!(st.earliest_lwcond_deadline(), None);
    }

    #[test]
    fn timed_cond_wait_times_out_reacquires_free_mutex_and_returns_code() {
        let mut st = state();
        let m = st.create_mutex();
        let cv = st.create_cond(m);
        // Thread 1 holds the mutex, then waits with a 500 us timeout: the wait releases
        // the mutex, parks, and registers a deadline.
        st.set_current(1);
        assert!(st.mutex_lock(m));
        st.cond_wait(cv, 500);
        assert_eq!(st.earliest_lwcond_deadline(), Some(500));
        assert!(st.take_wakes().is_empty());
        // No signaller comes: at the deadline the wait times out, re-acquires the (now
        // free) mutex, is woken, and is owed WAIT_TIMEOUT.
        st.advance_time_to(500);
        assert_eq!(st.take_wakes(), vec![1]);
        assert_eq!(st.take_resume_code(1), Some(SCE_KERNEL_ERROR_WAIT_TIMEOUT));
        st.set_current(2);
        assert!(st.mutex_contended(m), "the timed-out cond wait re-acquired its mutex");
    }

    #[test]
    fn timed_cond_wait_timeout_queues_behind_a_held_mutex() {
        let mut st = state();
        let m = st.create_mutex();
        let cv = st.create_cond(m);
        // Thread 1 waits with a timeout (releasing the mutex as it parks).
        st.set_current(1);
        assert!(st.mutex_lock(m));
        st.cond_wait(cv, 500);
        // Thread 2 grabs the mutex before the deadline.
        st.set_current(2);
        assert!(st.mutex_lock(m));
        // At the deadline the wait times out, but the mutex is held: thread 1 queues
        // behind thread 2 (not woken yet) while its WAIT_TIMEOUT code is already owed.
        st.advance_time_to(500);
        assert!(st.take_wakes().is_empty(), "the timed-out waiter queues behind the owner");
        // Thread 2 unlocks: thread 1 finally gets the mutex, is woken, and the owed
        // timeout code is delivered when it resumes.
        st.mutex_unlock(m);
        assert_eq!(st.take_wakes(), vec![1]);
        assert_eq!(st.take_resume_code(1), Some(SCE_KERNEL_ERROR_WAIT_TIMEOUT));
    }

    #[test]
    fn timed_lwcond_wait_times_out_with_code_but_a_signal_returns_zero() {
        let mut st = state();
        // A timed lightweight-cond wait that expires is owed WAIT_TIMEOUT. A cond
        // always has a bound mutex (set at CreateLwCond) - bind one so the wait parks.
        st.lwcond_bind_mutex(0x9000, 0x9100);
        st.set_current(1);
        assert!(st.lwcond_wait(0x9000, 250)); // work-area address, 250 us timeout
        assert_eq!(st.earliest_lwcond_deadline(), Some(250));
        st.advance_time_to(250);
        assert_eq!(st.take_wakes(), vec![1]);
        assert_eq!(st.take_resume_code(1), Some(SCE_KERNEL_ERROR_WAIT_TIMEOUT));
        // A timed wait released by a signal instead returns 0 (no resume code).
        st.set_current(2);
        assert!(st.lwcond_wait(0x9000, 250));
        st.set_current(3);
        st.lwcond_signal(0x9000, false);
        assert_eq!(st.take_wakes(), vec![2]);
        assert_eq!(st.take_resume_code(2), None);
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
    fn lightweight_mutex_blocks_and_hands_over_keyed_by_work_address() {
        let mut st = state();
        let work = 0x8000; // guest work-area address (no kernel handle)
        // Thread 1 locks the lightweight mutex; thread 2 contends and parks.
        st.set_current(1);
        assert!(st.lwmutex_lock(work));
        st.set_current(2);
        assert!(st.lwmutex_contended(work));
        assert!(!st.lwmutex_lock(work), "contender must block, not silently succeed");
        assert!(st.take_wakes().is_empty());
        // Thread 1 unlocks: ownership passes to thread 2, which is woken.
        st.set_current(1);
        st.lwmutex_unlock(work);
        assert_eq!(st.take_wakes(), vec![2]);
        // A different work address is an independent lock (thread 3 takes it freely).
        st.set_current(3);
        assert!(st.lwmutex_lock(0x9000));
        assert!(st.lwmutex_contended(work), "the first lock is still held by thread 2");
    }

    #[test]
    fn lightweight_mutex_is_recursive_for_the_owner() {
        let mut st = state();
        let work = 0x8000;
        st.set_current(1);
        assert!(st.lwmutex_lock(work)); // count 1
        assert!(st.lwmutex_lock(work)); // count 2 (recursive, same owner)
        st.lwmutex_unlock(work); // count 1, still owned
        st.set_current(2);
        assert!(st.lwmutex_contended(work), "still held after one of two unlocks");
        st.set_current(1);
        st.lwmutex_unlock(work); // count 0, released
        st.set_current(2);
        assert!(!st.lwmutex_contended(work), "free after the matching unlock");
    }

    #[test]
    fn lightweight_cond_wait_releases_and_reacquires_its_bound_mutex() {
        let mut st = state();
        let mutex_work = 0x8000;
        let cond_work = 0x8100;
        st.lwcond_bind_mutex(cond_work, mutex_work);
        // Thread 1 holds the lwmutex, then waits on the lwcond: the wait releases the
        // mutex (so a sibling can take it) and parks thread 1.
        st.set_current(1);
        assert!(st.lwmutex_lock(mutex_work));
        assert!(st.lwcond_wait(cond_work, 0));
        assert!(st.take_wakes().is_empty(), "waiter is parked, not runnable");
        st.set_current(2);
        assert!(!st.lwmutex_contended(mutex_work), "the lwmutex was released by the wait");
        // Thread 2 takes the lwmutex, then signals: the waiter must re-acquire the
        // mutex first, so it queues behind thread 2 (not woken yet).
        assert!(st.lwmutex_lock(mutex_work));
        st.lwcond_signal(cond_work, false);
        assert!(st.take_wakes().is_empty(), "waiter must re-acquire the lwmutex first");
        // Thread 2 unlocks: ownership passes to the waiter, which is finally woken.
        st.lwmutex_unlock(mutex_work);
        assert_eq!(st.take_wakes(), vec![1]);
        st.set_current(3);
        assert!(st.lwmutex_contended(mutex_work), "the woken waiter re-acquired the lwmutex");
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
        st.cond_wait(cv, 0);
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
        st.cond_wait(cv, 0); // releases m, parks thread 1
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
            st.cond_wait(cv, 0);
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
    fn doubled_slashes_and_dot_segments_resolve_to_the_same_file() {
        // A title that joins a dir with a trailing `/` to a subpath with a leading `/`
        // produces `//`; a real Vita FS collapses it. The font-load path
        // `app0:/usrdir//ui/fonts//x.ttf` regressed a whole title before this
        // normalization landed (font missed -> every glyph "missing" -> null-glyph copy).
        let mut st = state();
        st.add_file("app0:/usrdir/ui/fonts/x.ttf", b"FONTDATA".to_vec());
        assert_eq!(
            st.read_file("app0:/usrdir//ui/fonts//x.ttf").as_deref(),
            Some(&b"FONTDATA"[..])
        );
        // `.` and `..` segments resolve too, and open() shares the same key.
        assert_eq!(
            st.read_file("app0:/usrdir/./ui/bogus/../fonts/x.ttf").as_deref(),
            Some(&b"FONTDATA"[..])
        );
        assert!(st.io_open("app0:/usrdir//ui/fonts//x.ttf", SCE_O_RDONLY) >= 3);
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
    fn uppercase_app0_mount_finds_the_same_file() {
        // Titles spell the mount both ways; both must resolve to the same key.
        let mut st = state();
        st.add_file("Disc/Data/File.bin", b"x".to_vec());
        assert!(st.io_open("APP0:Disc/Data/File.bin", SCE_O_RDONLY) >= 3);
        assert!(st.io_open("app0:/disc/data/file.bin", SCE_O_RDONLY) >= 3);
        assert_eq!(st.io_size("APP0:Disc/Data/File.bin"), Some(1));
    }

    #[test]
    fn dopen_lists_children_with_original_case() {
        let mut st = state();
        st.add_file("Disc/Course/Course_601/a.dat", vec![0; 3]);
        st.add_file("Disc/Course/Course_601/b.dat", vec![0; 4]);
        st.add_file("Disc/Course/Course_700/a.dat", vec![0; 5]);
        st.add_file("Disc/Course/Readme.txt", vec![0; 7]);

        let fd = st.io_dopen("APP0:Disc/Course");
        assert!(fd >= 3);
        // Ordered by lowercased name; subdirectories deduplicated; names keep the
        // original mixed case a title's glob (e.g. `Course_*`) expects.
        let e1 = st.io_dread(fd).unwrap().unwrap();
        assert_eq!((e1.name.as_str(), e1.is_dir, e1.size), ("Course_601", true, 0));
        let e2 = st.io_dread(fd).unwrap().unwrap();
        assert_eq!((e2.name.as_str(), e2.is_dir, e2.size), ("Course_700", true, 0));
        let e3 = st.io_dread(fd).unwrap().unwrap();
        assert_eq!((e3.name.as_str(), e3.is_dir, e3.size), ("Readme.txt", false, 7));
        // Exhausted: end-of-listing, repeatably.
        assert_eq!(st.io_dread(fd), Some(None));
        assert_eq!(st.io_dread(fd), Some(None));
        assert_eq!(st.io_dclose(fd), 0);
        // Closed (and never-opened) descriptors are bad.
        assert!(st.io_dread(fd).is_none());
        assert!(st.io_dclose(fd) < 0);
    }

    #[test]
    fn dopen_of_a_subdir_and_missing_dir() {
        let mut st = state();
        st.add_file("Disc/Course/Course_601/Deep/x.bin", vec![0; 2]);
        // A trailing slash and mixed-case path still resolve.
        let fd = st.io_dopen("app0:Disc/COURSE/Course_601/");
        assert!(fd >= 3);
        let e = st.io_dread(fd).unwrap().unwrap();
        assert_eq!((e.name.as_str(), e.is_dir), ("Deep", true));
        assert_eq!(st.io_dread(fd), Some(None));
        st.io_dclose(fd);
        // Nothing under the path: a negative errno.
        assert!(st.io_dopen("app0:Disc/Nope") < 0);
        // A file fd is not a directory fd and vice versa.
        let ffd = st.io_open("app0:disc/course/course_601/deep/x.bin", SCE_O_RDONLY);
        assert!(st.io_dread(ffd).is_none());
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
