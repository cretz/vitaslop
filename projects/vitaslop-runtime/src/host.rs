//! The host-call boundary: how a guest NID import trap becomes a typed Rust
//! handler. `GuestCtx` marshals AAPCS arguments (r0..r3 then stack) and guest
//! memory in and out; `VitaEnv` owns the per-run state (allocator, handles,
//! capture, world) and routes a dense import index to a per-module handler.
//! See `projects/vitaslop-runtime/README.md`.

// The per-draw maps below are hashed with FxHash rather than SipHash. See `crate::fasthash` for
// why, and for what that trade actually is.
use crate::fasthash::{FxHashMap, FxHashSet};
use std::sync::Arc;

use vitaslop_transpiler::abi::{REG_COUNT, SP};

use crate::capture::Capture;
use crate::world::{DeterministicWorld, World};
use crate::vita::lwwork;
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
/// Read and write single guest WORDS by guest address, for state that lives in guest
/// memory and must be reachable from outside a host call.
///
/// [`GuestCtx`] is the ordinary way to touch guest memory, but it exists only for the
/// duration of a dispatch. Some state the guest owns has to be maintained at points where
/// no guest call is in flight - a lightweight mutex handed to a waiter whose cond wait
/// TIMED OUT is decided in `advance_time_to`, on the scheduler's idle path, with no ctx
/// anywhere. This is the narrow accessor those paths take instead, so the state can have
/// exactly one home (see [`crate::vita::lwwork`]) rather than a guest copy and a host copy
/// that agree until they do not.
///
/// An address outside the provisioned region reads zero and swallows its write, matching
/// what [`GuestCtx`] does. That is a real answer for the callers here: a work area outside
/// guest memory is not a mutex, and every fast path tests its identity stamp first.
pub trait GuestWords {
    fn word(&self, addr: u32) -> u32;
    fn set_word(&mut self, addr: u32, value: u32);
}

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
    /// Read the little-endian u32 at rebased offset `off`, which the caller has already
    /// bounds-checked.
    ///
    /// # Why a backing gets to override this
    /// The default is [`read`](Self::read) into a four-byte buffer, which is right for an
    /// in-process backing where a block read is a `memcpy`. It is NOT right for the
    /// browser, where guest memory is a `SharedArrayBuffer` reached through a typed array:
    /// there a block read is a `subarray` (a boundary crossing that ALLOCATES a JS object)
    /// followed by a `copy_to` (a second crossing), where an aligned word is one
    /// `get_index` into a `Uint32Array` over the same buffer - one crossing, no allocation.
    ///
    /// That matters because word reads are not rare. MEASURED on a racing title's on-track
    /// frame in the browser: ~22,000 single-word reads per presented frame, after the four largest
    /// per-word readers had already been converted to bulk reads
    /// ([[vitaslop-count-calls-not-bytes-across-the-guest-boundary]]). Converting a reader
    /// to a bulk read removes its words; this makes the words that REMAIN - in every
    /// handler, on every path, including ones nobody has profiled - cost a third as much.
    fn read_u32(&self, off: usize) -> u32 {
        let mut b = [0u8; 4];
        self.read(off, &mut b);
        u32::from_le_bytes(b)
    }

    /// Read the little-endian u16 at rebased offset `off`. See [`read_u32`](Self::read_u32).
    fn read_u16(&self, off: usize) -> u16 {
        let mut b = [0u8; 2];
        self.read(off, &mut b);
        u16::from_le_bytes(b)
    }

    /// Write the little-endian u32 `v` at rebased offset `off`. See
    /// [`read_u32`](Self::read_u32) - a write pays the same two crossings a read does, and
    /// the scalar GXM setters and the scheduler's guest-visible mirror are all writes.
    ///
    /// An overriding implementation owes the same side effects [`write`](Self::write) has,
    /// which for the browser means stamping the guest-store dirty map.
    fn write_u32(&mut self, off: usize, v: u32) {
        self.write(off, &v.to_le_bytes());
    }

    /// Borrow `len` bytes at rebased offset `off` directly, without copying, if this
    /// backing can lend them.
    ///
    /// Every in-process host can: guest memory is a byte buffer it owns. The point is
    /// to COMPARE a large region against a host-side copy without materialising a
    /// second one - which is how the texture-snapshot cache stays exact (it re-reads
    /// only what actually changed) instead of re-reading everything every frame.
    ///
    /// The default is `None`, so a backing that cannot lend simply loses the
    /// optimisation rather than the correctness.
    fn borrow(&self, off: usize, len: usize) -> Option<&[u8]> {
        let _ = (off, len);
        None
    }

    /// Has the guest STORED anywhere in rebased `[off, off + len)` at or after epoch
    /// `stamp`? `Some(true)` if it may have, `Some(false)` if it provably did not,
    /// `None` if this backing tracks no stores at all.
    ///
    /// `None` is not "clean" - it is "no answer", and a caller that cannot tell the
    /// difference would serve stale pixels. Every caller must fall back to reading the
    /// bytes.
    ///
    /// The map is a byte per 4 KB page holding the epoch of the last store into it; the
    /// transpiler writes it (`emit_dirty_mark`, which explains why it is an epoch and
    /// not a flag) and this reads it. Nothing is ever cleared, so two callers asking
    /// about overlapping pages cannot spoil each other's answer.
    ///
    /// # The one-page overhang
    /// The transpiler stamps the page a store STARTS in, not every page it touches -
    /// widening it costs as much again on the hottest path in the module, and the
    /// reader gets it for free. The largest translated store is 8 bytes, so a store
    /// reaching into a page can only have started in that page or the one below it.
    /// An implementation therefore examines one page BELOW `off` as well, which makes
    /// the answer exact.
    fn dirty_since(&self, off: usize, len: usize, stamp: u8) -> Option<bool> {
        let _ = (off, len, stamp);
        None
    }

    /// Advance the guest-store epoch and return `(the stamp to record, whether the
    /// epoch WRAPPED)`. `None` when this backing tracks no stores.
    ///
    /// A caller records the returned stamp against bytes it has just read out of guest
    /// memory; a later [`dirty_since`](Self::dirty_since) with that stamp then answers
    /// "has the guest written here since I read it?". Advancing BEFORE the caller
    /// records is what makes it exact: the guest cannot run during a host call, so no
    /// store can already carry the new epoch.
    ///
    /// The epoch is one byte, so it runs out; on wrap the map is zeroed and `true` is
    /// returned, and every stamp recorded before that moment is meaningless. A caller
    /// that sees `true` must discard its stamps and re-establish them the slow way.
    ///
    /// # Why `&self` for a method that writes
    /// The map is not guest state and not Rust-owned memory: the only backing that has
    /// one reaches it through a JS typed array over a `SharedArrayBuffer`, where no
    /// `&mut [u8]` exists to alias. Taking `&mut self` would instead force
    /// [`GuestCtx::borrow_bytes`]'s caller to hold a mutable borrow of guest memory
    /// across the very compare it is trying to avoid.
    fn bump_dirty_epoch(&self) -> Option<(u8, bool)> {
        None
    }

    /// The epoch a store made RIGHT NOW would be stamped with, or `None` from a backing
    /// that tracks no stores. The number [`rebase_dirty_epoch`](Self::rebase_dirty_epoch)
    /// exists to keep away from its ceiling.
    fn dirty_epoch(&self) -> Option<u8> {
        None
    }

    /// >>> RENUMBER THE EPOCH INSTEAD OF SPENDING IT, so a wrap does not throw the whole
    /// texture working set away.
    ///
    /// The epoch is ONE BYTE and advances once per SCENE. A race frame is eleven scenes and
    /// the browser runs more than one guest frame per present, so the 253 usable values are
    /// gone in about ten presented frames - and [`bump_dirty_epoch`](Self::bump_dirty_epoch)
    /// answers that by zeroing the map, which invalidates every stamp its caller holds.
    /// MEASURED on a racing title's on-track frame in the browser: **5.40 MB per present**
    /// copied
    /// across the guest boundary and compared, purely to re-establish snapshots a wrap had
    /// just disowned - and every one of those compares found the bytes IDENTICAL.
    ///
    /// The values in use are not spread over the range: a snapshot is re-stamped whenever it
    /// is proved current, so live stamps cluster just below the epoch and everything below
    /// `floor` is free. Subtracting `floor - 1` from the map, from the epoch and from every
    /// stamp the caller holds reclaims all of it and preserves the `page >= stamp` predicate
    /// exactly - a page below `floor` becomes 0, which is below every renumbered stamp, and it
    /// was below every live stamp before.
    ///
    /// `floor` MUST be the lowest stamp the caller still holds; a stamp below it would be
    /// renumbered into nonsense. Returns the new current epoch, and `None` from a backing
    /// with no map (whose caller then keeps the wrap it always had).
    fn rebase_dirty_epoch(&self, floor: u8) -> Option<u8> {
        let _ = floor;
        None
    }
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
    fn borrow(&self, off: usize, len: usize) -> Option<&[u8]> {
        self.0.get(off..off.checked_add(len)?)
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
        crate::perf::note_word_read();
        match self.offset(addr) {
            Some(o) if o + 4 <= self.mem.len() => self.mem.read_u32(o),
            _ => 0,
        }
    }

    /// Write a little-endian u32 at guest address `addr` (ignored if out of range).
    pub fn write_u32(&mut self, addr: u32, v: u32) {
        report_host_write(self, addr, 4, &v.to_le_bytes());
        if let Some(o) = self.offset(addr) {
            if o + 4 <= self.mem.len() {
                self.mem.write_u32(o, v);
            }
        }
    }

    /// Read `len` bytes at guest address `addr` (short read clamped to range).
    pub fn read_bytes(&self, addr: u32, len: usize) -> Vec<u8> {
        crate::perf::note_bulk_read();
        match self.offset(addr) {
            Some(o) => {
                let n = (o + len).min(self.mem.len()) - o;
                // Borrow-and-clone where the backing allows it: `to_vec` allocates
                // and copies once, where allocating a zeroed buffer and reading into
                // it writes every byte TWICE. This path moves megabytes a frame
                // (vertex, index and texture snapshots), so the second pass is real.
                if let Some(s) = self.mem.borrow(o, n) {
                    return s.to_vec();
                }
                let mut buf = vec![0u8; n];
                self.mem.read(o, &mut buf);
                buf
            }
            None => Vec::new(),
        }
    }

    /// Read guest memory at `addr` into `buf`, filling the tail with zeros if the
    /// range runs past the provisioned region. The allocation-free counterpart of
    /// [`read_bytes`](Self::read_bytes), for a caller with a buffer to reuse - which is
    /// what a per-draw or per-parameter read must be, since the allocator is otherwise
    /// the dominant cost of reading a few dozen bytes.
    pub fn read_into(&self, addr: u32, buf: &mut [u8]) {
        crate::perf::note_bulk_read();
        let n = match self.offset(addr) {
            Some(o) => {
                let n = buf.len().min(self.mem.len().saturating_sub(o));
                self.mem.read(o, &mut buf[..n]);
                n
            }
            None => 0,
        };
        // Only the TAIL that guest memory did not cover is zeroed - zeroing the whole
        // buffer first would write every byte twice on the common in-range path.
        buf[n..].fill(0);
    }

    /// Read `count` consecutive little-endian f32s starting at `addr`.
    ///
    /// One bounds check and one block copy, rather than `count` calls through the
    /// `dyn GuestMemory` vtable. A default uniform buffer is hundreds of floats and is
    /// read on every draw, so the per-element version showed up as a per-draw cost.
    pub fn read_f32s(&self, addr: u32, count: usize) -> Vec<f32> {
        // Borrow and convert straight into the result: no staging buffer, and the
        // result is written once. A default uniform buffer is hundreds of floats and
        // is read on every draw.
        if let Some(o) = self.offset(addr) {
            if let Some(s) = self.mem.borrow(o, count * 4) {
                return s
                    .chunks_exact(4)
                    .map(|c| f32::from_le_bytes([c[0], c[1], c[2], c[3]]))
                    .collect();
            }
        }
        let mut raw = vec![0u8; count * 4];
        self.read_into(addr, &mut raw);
        raw.chunks_exact(4).map(|c| f32::from_le_bytes([c[0], c[1], c[2], c[3]])).collect()
    }

    /// Borrow `len` bytes of guest memory at `addr` in place, without copying.
    /// `None` when the range is out of bounds or the backing cannot lend
    /// ([`GuestMemory::borrow`]).
    pub fn borrow_bytes(&self, addr: u32, len: usize) -> Option<&[u8]> {
        crate::perf::note_bulk_read();
        self.mem.borrow(self.offset(addr)?, len)
    }

    /// Has the guest stored anywhere in `[addr, addr + len)` at or after epoch
    /// `stamp`? `None` when the engine tracks no stores, which is not an answer -
    /// see [`GuestMemory::dirty_since`], which this forwards to.
    pub fn dirty_since(&self, addr: u32, len: usize, stamp: u8) -> Option<bool> {
        self.mem.dirty_since(self.offset(addr)?, len, stamp)
    }

    /// Advance the guest-store epoch - see [`GuestMemory::bump_dirty_epoch`].
    pub fn bump_dirty_epoch(&self) -> Option<(u8, bool)> {
        self.mem.bump_dirty_epoch()
    }

    /// The epoch a store made now would carry - see [`GuestMemory::dirty_epoch`].
    pub fn dirty_epoch(&self) -> Option<u8> {
        self.mem.dirty_epoch()
    }

    /// Renumber the epoch against `floor` - see [`GuestMemory::rebase_dirty_epoch`]. The
    /// caller owes the same subtraction on every stamp it holds.
    pub fn rebase_dirty_epoch(&self, floor: u8) -> Option<u8> {
        self.mem.rebase_dirty_epoch(floor)
    }

    /// Write `bytes` at guest address `addr` (clamped to range).
    pub fn write_bytes(&mut self, addr: u32, bytes: &[u8]) {
        report_host_write(self, addr, bytes.len() as u32, bytes);
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
        crate::perf::note_word_read();
        match self.offset(addr) {
            Some(o) if o + 2 <= self.mem.len() => self.mem.read_u16(o),
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

/// The ordinary way to reach guest-resident state: through the call in flight, so the
/// write watches and the range clamp all apply exactly as they do to any other host write.
impl GuestWords for GuestCtx<'_> {
    fn word(&self, addr: u32) -> u32 {
        self.read_u32(addr)
    }
    fn set_word(&mut self, addr: u32, value: u32) {
        self.write_u32(addr, value);
    }
}

/// `VITASLOP_HOST_WRITE_WATCH=<hex addr>[,...]`: report every write a HOST CALL makes to one
/// of those guest addresses, with the bytes and the guest return address that asked for it.
///
/// This is the counterpart `VITASLOP_WATCH_STORE` cannot be. That watchpoint traps the guest's
/// own stores; a host call writing a guest buffer on the guest's behalf - a file read, an
/// out-parameter, `sceGxmSetUniformDataF` into a game-owned block - leaves no store to trap.
/// "No guest store ever writes that address" then reads as "it must be static data", and a
/// whole session can go looking for a file that does not exist. Both watchpoints exist because
/// each is blind exactly where the other sees (memory `vitaslop-host-call-reference-semantics`).
/// An entry may also be `v:<hex word>`, which matches on the VALUE written anywhere rather
/// than on an address. When the question is "where did this number come from" the address is
/// often the thing you do not have: a material's uniform block is heap, so its address moves
/// between runs, while the suspicious 32-bit pattern in it is the same every time.
fn report_host_write(ctx: &GuestCtx, addr: u32, len: u32, bytes: &[u8]) {
    use std::sync::OnceLock;
    static WATCH: OnceLock<(Vec<u32>, Vec<u32>)> = OnceLock::new();
    let (addrs, vals) = WATCH.get_or_init(|| {
        let (mut a, mut v) = (Vec::new(), Vec::new());
        for t in std::env::var("VITASLOP_HOST_WRITE_WATCH").unwrap_or_default().split(',') {
            let t = t.trim();
            let (list, hex) = match t.strip_prefix("v:") {
                Some(rest) => (&mut v, rest),
                None => (&mut a, t),
            };
            if let Ok(n) = u32::from_str_radix(hex.trim_start_matches("0x"), 16) {
                list.push(n);
            }
        }
        (a, v)
    });
    if addrs.is_empty() && vals.is_empty() {
        return;
    }
    let end = addr.wrapping_add(len.max(1)).wrapping_sub(1);
    let hit = addrs.iter().copied().find(|&a| a >= addr && a <= end).or_else(|| {
        (!vals.is_empty())
            .then(|| {
                bytes.chunks_exact(4).position(|c| {
                    vals.contains(&u32::from_le_bytes([c[0], c[1], c[2], c[3]]))
                })
            })
            .flatten()
            .map(|i| addr + i as u32 * 4)
    });
    let Some(hit) = hit else { return };
    // The bytes AT the watched word, not the whole (possibly megabyte) write: what the question
    // is about is the value that ends up there.
    let off = (hit.wrapping_sub(addr)) as usize & !3;
    let word = bytes
        .get(off..off + 4)
        .map(|b| u32::from_le_bytes([b[0], b[1], b[2], b[3]]))
        .unwrap_or(0);
    eprintln!(
        "host write watch: a HOST CALL wrote {len} bytes at {addr:#x} covering {hit:#x}, leaving \
         {word:#010x} there, with the guest at lr={:#010x} pc={:#010x}",
        ctx.regs[14], ctx.regs[15]
    );
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
/// Strip `prefix` from `path` when it is a genuine PATH prefix, returning the rest.
///
/// "Genuine" means the prefix ends at a component boundary: `ux0:data` must match
/// `ux0:data/save` and `ux0:data` itself, but not `ux0:database`. A plain
/// `starts_with` would redirect the second, which is a silent misdirection of a file
/// the overlay was never meant to cover. Matching is case-insensitive, as the rest of
/// this filesystem's lookups are.
fn strip_path_prefix(path: &str, prefix: &str) -> Option<String> {
    let prefix = prefix.trim_end_matches('/');
    if prefix.is_empty() {
        return Some(path.to_string());
    }
    let (lp, lx) = (path.to_ascii_lowercase(), prefix.to_ascii_lowercase());
    let rest = lp.strip_prefix(&lx)?;
    if rest.is_empty() || rest.starts_with('/') {
        Some(path[prefix.len()..].to_string())
    } else {
        None
    }
}

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

/// Base of the SceUID range module ids come from. Modules are placed once at link
/// time and never unloaded, so a module's load-order index is a stable identity; the
/// base keeps those ids clear of the small ones threads and sync objects mint.
const MODULE_UID_BASE: i32 = 0x0100_0000;

/// What `sceKernelGetThreadInfo` reports about one thread. A borrowed view rather
/// than a copy: the caller writes it straight into the guest's struct.
pub struct ThreadInfo<'a> {
    pub name: &'a str,
    pub attr: u32,
    pub status: u32,
    pub entry: u32,
    pub stack_base: u32,
    pub stack_size: i32,
    pub init_priority: i32,
    pub current_priority: i32,
    pub cpu_affinity: i32,
    pub exit_status: i32,
}

/// `SceThreadStatus` bits (vitasdk `psp2common/kernel/threadmgr.h`).
const SCE_THREAD_RUNNING: u32 = 1;
const SCE_THREAD_READY: u32 = 2;
const SCE_THREAD_DORMANT: u32 = 16;
const SCE_THREAD_DEAD: u32 = 32;

/// The `SceThreadStatus` of a thread record. A created-but-never-started thread and a
/// finished one are distinguished the same way [`VitaState::delete_thread`] does it:
/// neither has an exit code until it has actually run.
fn thread_status(t: &ThreadRec, current: i32) -> u32 {
    if t.exit_code.is_some() {
        SCE_THREAD_DEAD
    } else if !t.started {
        SCE_THREAD_DORMANT
    } else if t.uid == current {
        SCE_THREAD_RUNNING
    } else {
        SCE_THREAD_READY
    }
}

/// A live semaphore. The count is what the primitive runs on; the rest is what
/// `sceKernelGetSemaInfo` reports and what `sceKernelOpenSema` matches on, so it is
/// recorded at create time rather than reconstructed. `init`/`max` are the values the
/// guest asked for, not clamps we apply - the count is floored at 0 by the waiters and
/// the kernel does not police `max` on signal either.
struct SemaRec {
    uid: i32,
    /// The name passed to `sceKernelCreateSema`. Named semaphores are what
    /// `sceKernelOpenSema` resolves, so this is a lookup key, not just a label.
    name: String,
    attr: u32,
    init: i32,
    max: i32,
    count: i32,
}

struct ThreadRec {
    uid: i32,
    /// The name the guest passed to `sceKernelCreateThread`, if any. A worker's own
    /// name ("threadFile", "NU::File::RequestManager", "auto_save") is the fastest
    /// way to read a stalled thread dump, so it is kept rather than only logged.
    name: String,
    entry: u32,
    stack_top: u32,
    /// The stack allocation this thread owns: `(base, size)`. Kept so
    /// [`VitaState::delete_thread`] can hand it back for reuse - the guest allocator here
    /// is a bump allocator with no free, so a title that churns threads would otherwise
    /// march through the arena until it collided with something else.
    stack: (u32, u32),
    /// Whether the thread has ever been started. A created-but-never-started thread is
    /// DORMANT and may be deleted; a running one may not, and both have no exit code, so
    /// the exit code alone cannot tell them apart.
    started: bool,
    exit_code: Option<u32>,
    /// SceKernel thread priority (lower number = higher priority). The scheduler
    /// runs the highest-priority runnable thread, matching the real kernel.
    priority: i32,
    /// The priority the thread was CREATED with, before any
    /// `sceKernelChangeThreadPriority`. Reported separately from the current one by
    /// `sceKernelGetThreadInfo`, and the value a title restores when it temporarily
    /// boosts a worker.
    init_priority: i32,
    /// The `attr` and `cpuAffinityMask` from `sceKernelCreateThread`. Neither steers
    /// this scheduler (one core, and the attribute bits select stack/VFP options we
    /// always provide), but both are reported verbatim by `sceKernelGetThreadInfo`,
    /// so they are kept rather than dropped at the seam.
    attr: u32,
    cpu_affinity: i32,
    /// Signals delivered by `sceKernelSendSignal` and not yet consumed by
    /// `sceKernelWaitSignal`. A counter, not a flag: the kernel counts sends, and a
    /// producer that signals twice before the consumer runs must not lose one.
    signals: u32,
}

/// One socket in the offline SceNet table. Everything recorded here is LOCAL state -
/// what the guest itself set - because nothing else exists to record.
struct NetSocket {
    id: i32,
    /// The debug name SceNet takes at creation (it names sockets, unlike BSD).
    name: String,
    domain: i32,
    ty: i32,
    protocol: i32,
    /// The address `sceNetBind` recorded, as `(network-order ip, host-order port)`.
    local: (u32, u16),
    listening: bool,
    /// `(level, optname, value)` for every option the guest set, so a Get round-trips.
    options: Vec<(i32, i32, u32)>,
    closed: bool,
}

/// Socket, resolver and epoll id ranges. Deliberately disjoint from each other and well
/// above the file-descriptor range, so an id used with the wrong API is rejected rather
/// than silently naming somebody else's object.
const NET_FD_BASE: i32 = 0x1000;
const NET_RESOLVER_BASE: i32 = 0x2000;
const NET_EPOLL_BASE: i32 = 0x3000;

/// The stack pointer a fiber starts at: the top of its context buffer, 8-byte aligned
/// as AAPCS requires at a public call. A fiber with no context yet yields 0, which the
/// callers refuse to run rather than execute on a null stack.
fn fiber_stack_top(context_addr: u32, context_size: u32) -> u32 {
    if context_addr == 0 {
        0
    } else {
        context_addr.wrapping_add(context_size) & !0xF
    }
}

/// One `SceFiber` the guest has initialised.
///
/// A fiber is cooperative: exactly one member of a fiber chain runs at a time, and a
/// switch is explicit. That is precisely what the preemptive scheduler already
/// provides through park/wake, so a fiber is backed by an ordinary guest thread that
/// is runnable ONLY while it holds the baton. Nothing about the guest's own stack
/// changes: `Reentry` already carries an explicit `stack_top`, so the fiber's thread
/// runs on the context buffer the guest supplied, exactly as the hardware does.
///
/// Modelling a switched-away fiber by nesting a re-entry cannot work - its stack has
/// to survive with live frames on it until it is switched back to - and that survival
/// is exactly what a parked scheduler thread gives for free.
struct FiberRec {
    /// The guest `SceFiber*`. This IS the fiber's identity: every API call names it,
    /// and the guest may hold it anywhere, so nothing about a fiber is keyed on
    /// anything else.
    addr: u32,
    name: String,
    entry: u32,
    arg_on_initialize: u32,
    /// The guest-supplied context (the fiber's stack) as `(base, size)`. Both zero
    /// until `_sceFiberAttachContextAndSwitch` supplies one - a fiber may be
    /// initialised without a context and given one at its first switch.
    context: (u32, u32),
    /// The scheduler thread backing this fiber. Created at initialize, STARTED only
    /// on the first run/switch: a fiber that is never run must never execute.
    thid: i32,
    started: bool,
    /// Set by `sceFiberFinalize`; the record is kept so a later call naming a
    /// finalized fiber is refused rather than silently resurrecting it.
    finalized: bool,
    /// The THREAD that ran this fiber's chain - what `sceFiberReturnToThread`
    /// returns to. 0 when the fiber is not currently running.
    runner: i32,
    /// Where to write the value this fiber is handed when it is next resumed: the
    /// `argOnRun` out-pointer of the `sceFiberSwitch`/`ReturnToThread` it is parked
    /// in. 0 when it has none (or has not run yet, where the value arrives in r1).
    resume_out: u32,
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

/// The SceUID of the initial ("main") thread. It is the one thread with no
/// [`ThreadRec`]: `create_thread` records what the GUEST creates, and the loader
/// starts this one before any guest code runs. Anything that has to name it uses
/// this rather than a bare 0.
pub const MAIN_THID: i32 = 0;

/// `SCE_KERNEL_ERROR_WAIT_TIMEOUT` - the value a *timed* blocking wait
/// (`sceKernelWaitSema`/`WaitCond`/`WaitLwCond`/`WaitEventFlag` with a non-null
/// timeout) returns when its deadline passes before the wait is satisfied. Delivered
/// to the woken thread's `r0` through the resume-code channel (see
/// [`VitaState::take_resume_code`]); a wait satisfied by a signal returns 0 instead.
pub const SCE_KERNEL_ERROR_WAIT_TIMEOUT: u32 = 0x8002_8005;

/// `SCE_KERNEL_ERROR_NOT_DORMANT` - a thread operation that requires the thread to be
/// stopped was asked of a running one.
///
/// From vitasdk `psp2/kernel/error.h`, NOT the psdevwiki error-code table: that table
/// disagrees with the header on this whole block (it lists NOT_DORMANT at 0x80028002 and
/// WAIT_TIMEOUT at 0x80028008, where the header has 0x80028028 and 0x80028005). The header
/// is the one that matches the value this engine already returns for a timed-out wait, so
/// the table's rows appear to be shifted. Prefer the header for SceKernel error numbers.
pub const SCE_KERNEL_ERROR_NOT_DORMANT: u32 = 0x8002_8028;
/// `SCE_KERNEL_ERROR_UNKNOWN_THREAD_ID` - no thread has that SceUID (same header).
pub const SCE_KERNEL_ERROR_UNKNOWN_THREAD_ID: u32 = 0x8002_8021;

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

/// What the HOST still keeps for a lightweight mutex: the threads parked on it, in FIFO
/// order, and the work-area address that names it.
///
/// The ownership itself - identity, owner, recursion count - is NOT here. It lives in the
/// guest's own work area ([`crate::vita::lwwork`]), which is where the hardware keeps it
/// and what makes the uncontended take inlinable. Only the parked QUEUE stays host-side,
/// because a list of thread ids in arrival order is not something guest memory holds
/// usefully; its LENGTH is published into the work area so guest code can tell the case it
/// may serve itself from the case only the host can.
struct LwMutexRec {
    work: u32,
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

/// Serves file bytes the host never loaded into memory.
///
/// # Why a title's files cannot simply be bytes
/// The in-memory map below is the right model for a test fixture and for native, where
/// the host has an address space to spare. It is the wrong model for the browser: a
/// retail container is over a gigabyte, and the emulator's wasm32 heap tops out at four.
/// Measured on a 1719 MB title: Chrome peaked at 8.01 GB during ingest and the worker was
/// killed mid-boot. So the browser stores the title in OPFS and serves reads from there,
/// and only what a guest actually asks for is ever resident.
///
/// # Reads are synchronous, and that is a requirement rather than a preference
/// A guest file read happens inside a host call, on a suspended guest stack that cannot
/// await. OPFS is the only browser storage offering synchronous reads
/// (`FileSystemSyncAccessHandle`, Workers only), which is why it is the backing and
/// Cache Storage / IndexedDB are not.
/// # The key an implementation is asked for is NORMALISED
/// [`keys`](FileBacking::keys) returns paths as the storage spells them, but
/// [`len`](FileBacking::len) and [`read_at`](FileBacking::read_at) are called with the
/// result of [`vfs_key`] - collapsed separators, resolved `.`/`..`, and LOWERCASED,
/// because the Vita filesystem matches case-insensitively. An implementation that maps
/// the key it receives straight back onto its own storage will therefore miss every
/// mixed-case path.
///
/// That is not a hypothetical: it is how this first shipped. The file OPENED (existence
/// is checked against the same normalised key, so that matched) and then every read
/// returned zero bytes, and what surfaced was a guest memory-access trap 31,000 host
/// calls later with nothing pointing at the filesystem. Build the normalised-to-stored
/// map with [`vfs_key`] itself, so the two sides cannot drift.
///
/// `Send` because the native preemptive scheduler moves the whole host between OS
/// threads, and that guarantee is real there - it must not be weakened for every backing
/// just because the browser's cannot honour it. The browser's OPFS backing holds JS
/// handles, which are `!Send`, and asserts `Send` for itself on the grounds that a wasm
/// worker is single-threaded by construction; that assertion is stated where it is made,
/// not hidden here.
pub trait FileBacking: Send {
    /// Length of `key`, or `None` if this backing does not serve it.
    fn len(&self, key: &str) -> Option<usize>;
    /// Read into `buf` starting at `off`; returns the count, short at end of file.
    fn read_at(&self, key: &str, off: usize, buf: &mut [u8]) -> usize;
    /// Every key served, as STORAGE spells them, so existence checks and directory
    /// listings see them all with their real names.
    fn keys(&self) -> Vec<String>;
}

/// A minimal virtual filesystem backing SceIoFilemgr: a path -> bytes map plus a
/// table of open descriptors. Read files are preloaded by the harness (e.g. a
/// game's data files) or served lazily by a [`FileBacking`]; write opens create or
/// truncate entries here. The console streams (fd 1/2) are handled directly in the IO
/// handlers and are not tracked here. Deterministic and host-only, so a run touches no
/// real filesystem.
#[derive(Default)]
pub struct FileTable {
    files: std::collections::HashMap<String, Vec<u8>>,
    /// Files served on demand by [`backing`](FileTable::backing), as key -> length.
    /// A key here is NOT in `files` until something writes to it, at which point it is
    /// faulted in and becomes an ordinary resident file - so a guest write to a game
    /// asset behaves exactly as it always did, at the cost of materialising that one
    /// file.
    backed: std::collections::HashMap<String, usize>,
    backing: Option<Box<dyn FileBacking>>,
    /// Original (as-added) spelling per lowercased key, so a directory listing can
    /// return real mixed-case names for a title's own glob matching. Lookup stays
    /// case-insensitive through the lowercased `files` key.
    originals: std::collections::HashMap<String, String>,
    open: std::collections::HashMap<i32, OpenFile>,
    open_dirs: std::collections::HashMap<i32, OpenDir>,
    next_fd: i32,
    /// Directories that exist but hold nothing. The map is flat, so a directory is
    /// normally IMPLIED by the keys beneath it - which makes an empty one
    /// unrepresentable, and `sceIoMkdir` followed by `sceIoRmdir` (or by a listing)
    /// unable to see its own work. Recording explicitly-created directories fixes
    /// that without giving up the flat map: a directory exists if it is in here OR
    /// some key lies under it.
    dirs: std::collections::HashSet<String>,
    /// Per-path status overrides written by `sceIoChstat`. Absent means "as
    /// synthesized" (a plain readable regular file); present means the guest set it,
    /// and a later `sceIoGetstat` must report what was set.
    stats: std::collections::HashMap<String, FileStatOverride>,
}

/// The `sceIoChstat`-settable parts of a file's status. Each is `Option` because
/// chstat takes a bit mask of WHICH fields to apply, so a call that sets only the
/// times must not also reset the mode.
#[derive(Clone, Default)]
pub struct FileStatOverride {
    pub mode: Option<u32>,
    pub attr: Option<u32>,
    /// `(ctime, atime, mtime)`, each the raw 16-byte guest `SceDateTime`.
    pub times: [Option<[u8; 16]>; 3],
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
    /// The (vfs-keyed) path the descriptor was opened on. Kept because a descriptor
    /// can be the subject of a path operation - `_sceFiosKernelOverlayDHChstatSync`
    /// changes the status of the directory a HANDLE names - and status is recorded
    /// per path here.
    path: String,
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
/// The path already exists (mkdir over something, rename onto something).
const SCE_ERROR_ERRNO_EEXIST: i32 = 0x8001_0011u32 as i32;
/// A file operation was asked of a directory.
const SCE_ERROR_ERRNO_EISDIR: i32 = 0x8001_0015u32 as i32;
/// `sceIoRmdir` on a directory that still holds entries.
const SCE_ERROR_ERRNO_ENOTEMPTY: i32 = 0x8001_005Au32 as i32;

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
pub fn vfs_key(path: &str) -> String {
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

    /// Install the lazy backing and register every key it serves. Called once at setup,
    /// before the guest runs.
    ///
    /// Registering the keys up front (rather than probing the backing on each miss) is
    /// what keeps existence checks, directory listings and `sceIoGetstat` answering from
    /// a plain map lookup - so a backed title behaves identically to a resident one
    /// everywhere except the read itself.
    pub fn set_backing(&mut self, backing: Box<dyn FileBacking>) {
        for path in backing.keys() {
            // NORMALISE FIRST. `len` and `read_at` take the same key the filesystem will
            // later look up with, not the storage spelling - see `FileBacking`. Asking
            // with the raw path is how this first shipped, and because a missing length
            // merely skipped the file, EVERY file was skipped and the title booted with an
            // empty filesystem. A silent skip is what turned a one-line mistake into a
            // guest memory trap 30,000 host calls away.
            let key = vfs_key(&path);
            let Some(n) = backing.len(&key) else {
                panic!(
                    "file backing lists {path:?} but has no length for its normalised key \
                     {key:?} - the two sides disagree on how a path is spelled, and every \
                     read of this file would silently return nothing"
                )
            };
            // Keep the as-supplied spelling too: a directory listing hands names back to
            // the title's own glob matching, and a lowercased name would not match.
            self.originals
                .insert(key.clone(), strip_app0(&path).trim_start_matches('/').to_string());
            self.backed.insert(key, n);
        }
        self.backing = Some(backing);
    }

    /// Whether `key` exists at all - resident or served by the backing.
    fn has(&self, key: &str) -> bool {
        self.files.contains_key(key) || self.backed.contains_key(key)
    }

    /// Length of `key`, wherever it lives.
    fn byte_len(&self, key: &str) -> Option<usize> {
        self.files.get(key).map(|d| d.len()).or_else(|| self.backed.get(key).copied())
    }

    /// Read `[start, start+len)` of `key`, clamped to its end. Serves a resident file
    /// from the map and a backed one straight from storage, so a backed read never
    /// materialises anything but the bytes asked for.
    fn read_range(&self, key: &str, start: usize, len: usize) -> Option<Vec<u8>> {
        if let Some(data) = self.files.get(key) {
            let start = start.min(data.len());
            let end = (start + len).min(data.len());
            return Some(data[start..end].to_vec());
        }
        let size = *self.backed.get(key)?;
        let backing = self.backing.as_ref()?;
        let start = start.min(size);
        let end = (start + len).min(size);
        let mut out = vec![0u8; end - start];
        let got = backing.read_at(key, start, &mut out);
        out.truncate(got);
        Some(out)
    }

    /// Backed keys equal to `key` or beneath it as a directory prefix. Used where a key
    /// is about to MOVE, which the backing cannot follow.
    fn backed_under(&self, key: &str) -> Vec<String> {
        let prefix = format!("{key}/");
        self.backed.keys().filter(|k| *k == key || k.starts_with(&prefix)).cloned().collect()
    }

    /// Make `key` resident so it can be written, reading it out of the backing first if
    /// that is where it lives. A no-op for a file that is already resident.
    ///
    /// This is the one place a backed file becomes an ordinary one. It is deliberately
    /// eager and whole-file: a partial write to a lazily-backed file would otherwise need
    /// a copy-on-write overlay, and the case does not arise in practice (titles write
    /// savedata, which they create, not the assets they shipped).
    fn make_resident(&mut self, key: &str) -> &mut Vec<u8> {
        if !self.files.contains_key(key) {
            let bytes = if self.backed.contains_key(key) {
                let n = self.backed[key];
                self.read_range(key, 0, n).unwrap_or_default()
            } else {
                Vec::new()
            };
            self.backed.remove(key);
            self.files.insert(key.to_string(), bytes);
        }
        self.files.get_mut(key).expect("just inserted")
    }

    /// Open `path` per the SCE_O_* `flags`; returns a new fd or a negative errno.
    fn open(&mut self, path: &str, flags: u32) -> i32 {
        let path = vfs_key(path);
        let readable = flags & SCE_O_RDWR == SCE_O_RDONLY || flags & SCE_O_RDWR == SCE_O_RDWR;
        let writable = flags & SCE_O_WRONLY != 0;
        let exists = self.has(&path);

        if !exists {
            if flags & SCE_O_CREAT != 0 {
                self.files.insert(path.clone(), Vec::new());
            } else {
                return SCE_ERROR_ERRNO_ENOENT;
            }
        } else if flags & SCE_O_TRUNC != 0 {
            // Truncation discards the contents, so a backed file needs no fault-in -
            // just drop the backing's claim on the key and start empty.
            self.backed.remove(&path);
            self.files.insert(path.clone(), Vec::new());
        }

        // Append seeks to end; every other open starts at the beginning.
        let cursor = if flags & SCE_O_APPEND != 0 { self.byte_len(&path).unwrap_or(0) } else { 0 };

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
        let (path, cursor) = (of.path.clone(), of.cursor);
        let out = self.read_range(&path, cursor, len)?;
        // Advance by what was actually delivered, so a short read at end of file leaves
        // the cursor at the end rather than past it.
        let end = cursor.min(self.byte_len(&path).unwrap_or(0)) + out.len();
        self.open.get_mut(&fd)?.cursor = end;
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
        self.read_range(&of.path.clone(), offset as usize, len)
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
        let data = self.make_resident(&path);
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
        let (path, cursor) = (of.path.clone(), of.cursor);
        let data = self.make_resident(&path);
        if cursor > data.len() {
            data.resize(cursor, 0);
        }
        let end = cursor + bytes.len();
        if end > data.len() {
            data.resize(end, 0);
        }
        data[cursor..end].copy_from_slice(bytes);
        self.open.get_mut(&fd)?.cursor = end;
        Some(bytes.len())
    }

    /// Seek `fd` to `offset` from `whence`; returns the new absolute position or a
    /// negative errno.
    fn lseek(&mut self, fd: i32, offset: i64, whence: i32) -> i64 {
        // The size lookup borrows `self`, so read the descriptor's path first and let
        // that borrow end before taking the mutable one back for the cursor update.
        let Some(path) = self.open.get(&fd).map(|of| of.path.clone()) else {
            return SCE_ERROR_ERRNO_EBADF as i64;
        };
        let size = self.byte_len(&path).unwrap_or(0) as i64;
        let Some(of) = self.open.get_mut(&fd) else {
            return SCE_ERROR_ERRNO_EBADF as i64;
        };
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
        // Resident and backed files both appear in a listing: a title that stored its
        // assets lazily must still see them when it enumerates a directory, and the
        // sizes have to be right because a title sizes its own read buffers from them.
        let entries: Vec<(&String, usize)> = self
            .files
            .iter()
            .map(|(k, d)| (k, d.len()))
            .chain(
                self.backed.iter().filter(|(k, _)| !self.files.contains_key(*k)).map(|(k, n)| (k, *n)),
            )
            .collect();
        for (k, len) in entries {
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
                entry.size = len as u64;
            }
        }
        // Explicitly-created directories are not implied by any file key, so they have
        // to be folded in separately or a title cannot see the directory it just made.
        for d in &self.dirs {
            let Some(rest) = d.strip_prefix(&prefix) else { continue };
            if rest.is_empty() {
                continue;
            }
            let comp = rest.split('/').next().unwrap_or(rest);
            children.entry(comp.to_string()).or_insert_with(|| {
                let name = self
                    .originals
                    .get(d)
                    .and_then(|orig| orig.split('/').nth(comp_index))
                    .unwrap_or(comp)
                    .to_string();
                DirEntry { name, is_dir: true, size: 0 }
            });
        }
        // An empty listing is only ENOENT when the directory itself does not exist; a
        // real, explicitly-created empty directory opens fine and reads zero entries.
        if children.is_empty() && !self.dirs.contains(key) {
            return SCE_ERROR_ERRNO_ENOENT;
        }
        let fd = self.next_fd;
        self.next_fd += 1;
        self.open_dirs.insert(
            fd,
            OpenDir { entries: children.into_values().collect(), cursor: 0, path: key.to_string() },
        );
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

    /// Whether any stored key lies under `key` (i.e. the directory has content).
    fn has_children(&self, key: &str) -> bool {
        let prefix = format!("{key}/");
        self.files.keys().any(|k| k.starts_with(&prefix))
            || self.backed.keys().any(|k| k.starts_with(&prefix))
            || self.dirs.iter().any(|d| d.starts_with(&prefix))
    }

    /// Whether `key` names a directory: one explicitly created, or one implied by a
    /// file stored beneath it.
    fn is_dir(&self, key: &str) -> bool {
        self.dirs.contains(key) || self.has_children(key)
    }

    /// int sceIoMkdir(const char *dir, SceMode mode): create a directory.
    ///
    /// Recorded rather than ignored, so the title can see the directory it just made
    /// (list it, stat it, remove it). The parent is not required to exist - the map is
    /// flat and the kernel's own mkdir is likewise a single-level operation whose
    /// failure mode a title checks by return value, which it now gets truthfully.
    fn mkdir(&mut self, path: &str) -> i32 {
        let key = vfs_key(path);
        let key = key.trim_end_matches('/').to_string();
        if key.is_empty() {
            return SCE_ERROR_ERRNO_EEXIST;
        }
        if self.has(&key) || self.is_dir(&key) {
            return SCE_ERROR_ERRNO_EEXIST;
        }
        self.originals.insert(key.clone(), path.trim_end_matches('/').to_string());
        self.dirs.insert(key);
        0
    }

    /// int sceIoRmdir(const char *path): remove an EMPTY directory.
    ///
    /// A non-empty one is refused (ENOTEMPTY) exactly as the kernel refuses it -
    /// silently succeeding would tell the title its tree is gone while every file under
    /// it is still readable.
    fn rmdir(&mut self, path: &str) -> i32 {
        let key = vfs_key(path);
        let key = key.trim_end_matches('/').to_string();
        if !self.is_dir(&key) {
            return SCE_ERROR_ERRNO_ENOENT;
        }
        if self.has_children(&key) {
            return SCE_ERROR_ERRNO_ENOTEMPTY;
        }
        self.dirs.remove(&key);
        self.originals.remove(&key);
        self.stats.remove(&key);
        0
    }

    /// int sceIoRemove(const char *file): delete a file.
    ///
    /// Really deletes it: the entry goes, so a later open without SCE_O_CREAT reports
    /// ENOENT and a listing no longer shows it. Refuses a directory (the kernel has
    /// `sceIoRmdir` for that) and an unknown path.
    fn remove(&mut self, path: &str) -> i32 {
        let key = vfs_key(path);
        if !self.has(&key) {
            return if self.is_dir(&key) { SCE_ERROR_ERRNO_EISDIR } else { SCE_ERROR_ERRNO_ENOENT };
        }
        self.files.remove(&key);
        // A backed file is deleted by forgetting the key: the bytes stay in storage but
        // nothing can reach them, which is what the guest asked for. Storage is the
        // user's installed title and is not ours to erase on a guest unlink.
        self.backed.remove(&key);
        self.originals.remove(&key);
        self.stats.remove(&key);
        0
    }

    /// int sceIoRename(const char *oldname, const char *newname): move a file or a
    /// whole directory subtree.
    ///
    /// A directory rename has to carry its contents, and in a flat map that means
    /// re-keying every entry under the old prefix - which is the entire reason this
    /// cannot be a two-line map swap. Refuses to overwrite an existing destination,
    /// as the kernel does.
    fn rename(&mut self, old: &str, new: &str) -> i32 {
        let (from, to) = (vfs_key(old), vfs_key(new));
        let (from, to) = (from.trim_end_matches('/').to_string(), to.trim_end_matches('/').to_string());
        if from == to {
            return 0;
        }
        if self.has(&to) || self.is_dir(&to) {
            return SCE_ERROR_ERRNO_EEXIST;
        }
        let new_orig = new.trim_end_matches('/').to_string();

        // A rename re-keys the file, and the backing only knows its ORIGINAL key - so a
        // backed file has to be faulted in first or its new name would read as empty.
        // Deliberately the expensive path: renaming a shipped asset does not happen in
        // practice, and a silently unreadable file would be far worse than a copy.
        for key in self.backed_under(&from) {
            self.make_resident(&key);
        }

        if let Some(data) = self.files.remove(&from) {
            self.files.insert(to.clone(), data);
            self.originals.remove(&from);
            self.originals.insert(to.clone(), new_orig);
            if let Some(s) = self.stats.remove(&from) {
                self.stats.insert(to, s);
            }
            return 0;
        }
        if !self.is_dir(&from) {
            return SCE_ERROR_ERRNO_ENOENT;
        }
        // A directory: re-key the directory itself and everything beneath it.
        let old_prefix = format!("{from}/");
        let rekey = |k: &str| format!("{to}{}", &k[from.len()..]);
        let moved: Vec<String> =
            self.files.keys().filter(|k| k.starts_with(&old_prefix)).cloned().collect();
        for k in moved {
            let data = self.files.remove(&k).unwrap_or_default();
            let nk = rekey(&k);
            // The original spelling's tail is preserved; only the renamed prefix changes.
            let orig_tail = self.originals.remove(&k).map(|o| o[from.len().min(o.len())..].to_string());
            self.files.insert(nk.clone(), data);
            if let Some(tail) = orig_tail {
                self.originals.insert(nk.clone(), format!("{new_orig}{tail}"));
            }
            if let Some(s) = self.stats.remove(&k) {
                self.stats.insert(nk, s);
            }
        }
        let moved_dirs: Vec<String> = self
            .dirs
            .iter()
            .filter(|d| **d == from || d.starts_with(&old_prefix))
            .cloned()
            .collect();
        for d in moved_dirs {
            self.dirs.remove(&d);
            self.dirs.insert(rekey(&d));
        }
        self.originals.remove(&from);
        self.originals.insert(to, new_orig);
        0
    }

    /// int sceIoChstat(const char *name, SceIoStat *stat, int bits): apply the
    /// selected fields of `stat` to a path. Stored so a later `sceIoGetstat` reports
    /// them back - a chstat whose effect is invisible is worse than an error, because
    /// the title believes the change took.
    fn chstat(&mut self, path: &str, over: FileStatOverride) -> i32 {
        let key = vfs_key(path);
        if !self.has(&key) && !self.is_dir(&key) {
            return SCE_ERROR_ERRNO_ENOENT;
        }
        let e = self.stats.entry(key).or_default();
        if over.mode.is_some() {
            e.mode = over.mode;
        }
        if over.attr.is_some() {
            e.attr = over.attr;
        }
        for (i, t) in over.times.iter().enumerate() {
            if t.is_some() {
                e.times[i] = *t;
            }
        }
        0
    }

    /// The chstat overrides recorded for a path, if any.
    fn stat_override(&self, path: &str) -> Option<&FileStatOverride> {
        self.stats.get(&vfs_key(path))
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
        self.byte_len(&vfs_key(path)).map(|n| n as u64)
    }

    /// The whole contents of `path` if it exists, cloned. For consumers that need a
    /// file's bytes in one shot without managing a descriptor (e.g. loading a font).
    fn read_all(&self, path: &str) -> Option<Vec<u8>> {
        let key = vfs_key(path);
        let n = self.byte_len(&key)?;
        self.read_range(&key, 0, n)
    }

    /// The size of the file behind an open descriptor (for sceIoGetstatByFd).
    /// Returns None on a bad descriptor - the path is already the vfs key.
    fn size_of_fd(&self, fd: i32) -> Option<u64> {
        let of = self.open.get(&fd)?;
        self.byte_len(&of.path).map(|n| n as u64)
    }
}

/// Per-draw vertex program layout captured at create time, keyed by the vertex
/// program handle so a later Draw knows how to snapshot the vertex buffer.
struct VertexProgramInfo {
    attributes: Vec<crate::capture::VertexAttribute>,
    /// >>> THE INTERLEAVED LAYOUT A DRAW ACTUALLY CAPTURES INTO, BUILT ONCE HERE.
    ///
    /// `record_draw` interleaves every stream an attribute names into ONE packed buffer and
    /// rewrites the attributes onto it. What that rewrite produces - the packed row stride,
    /// each stream's byte base within a row, whether the mesh is a single per-vertex stream,
    /// and the rebased attribute list - depends only on this program's declared attributes and
    /// its streams' strides, all of which are fixed when the vertex program is CREATED.
    ///
    /// It was recomputed on every draw, at the cost of cloning the attribute list, collecting a
    /// stream vector, building a base vector and looping over the attributes to add the bases -
    /// four allocations and two loops per draw, 983 draws a presented frame, all to arrive at
    /// the same constant every time. Doing it at registration makes a draw a refcount bump and
    /// two field reads. Rebuilt whenever the handle is registered again, which is the only
    /// moment any of it can change.
    packed_attributes: std::sync::Arc<[crate::capture::VertexAttribute]>,
    /// The streams an attribute actually names (`0..=max stream_index`), in stream order.
    used_streams: std::sync::Arc<[VertexStreamInfo]>,
    /// Byte offset of each used stream's row within a packed row - the prefix sum of the
    /// strides above.
    stream_base: std::sync::Arc<[u32]>,
    /// Bytes per packed vertex: the sum of the used streams' strides.
    packed_stride: u32,
    /// Exactly one used stream, stepped per VERTEX - the overwhelmingly common case, whose
    /// rows are already contiguous in guest memory and can be taken in one read.
    single_stream: bool,
    /// One entry per `SceGxmVertexStream` the program was created with, in stream order.
    /// A draw's attributes name a stream each, and only this tells us how wide that
    /// stream's rows are and whether it is stepped per vertex or per instance.
    streams: Vec<VertexStreamInfo>,
    /// The `SceGxmProgram*` this vertex program was created from, so a precomputed
    /// vertex state (which references the vertex program) can size its default uniform
    /// buffer from the program header (+0x2C). 0 if it could not be resolved.
    program_header: u32,
}

impl VertexProgramInfo {
    /// Stream `i`'s layout, or a zero-stride per-vertex stream if the program declared
    /// fewer streams than an attribute references (which a well-formed program cannot do,
    /// and which then contributes no bytes rather than reading somewhere arbitrary).
    fn stream(&self, i: usize) -> VertexStreamInfo {
        self.streams.get(i).copied().unwrap_or_default()
    }
}

/// One `SceGxmVertexStream`: `{ uint16_t stride; uint16_t indexSource; }`.
#[derive(Clone, Copy, Default)]
struct VertexStreamInfo {
    stride: u32,
    /// True for `SCE_GXM_INDEX_SOURCE_INSTANCE_{16,32}BIT` (2 and 3): the stream is
    /// stepped by the INSTANCE number, not the vertex index, so every vertex of one
    /// instance reads the same row. Getting this wrong feeds per-instance data (a
    /// particle's world transform, a decal's placement) to the shader indexed by vertex,
    /// which scatters one object's geometry across whatever the neighbouring rows hold.
    per_instance: bool,
}

/// A precomputed vertex- or fragment-state object (`sceGxmPrecomputed{Vertex,Fragment}
/// State*`): the default uniform buffer the guest writes its uniforms into, the bound
/// fragment/vertex textures, and the `SceGxmProgram*` (for uniform-buffer sizing).
/// Keyed by the guest state-struct address, applied to the live bind state when the
/// guest issues `sceGxmSetPrecomputed{Vertex,Fragment}State`, exactly as the individual
/// `sceGxmSetUniformDataF`/`sceGxmSetFragmentTexture` calls would be on the direct path.
/// One sampler unit's bound texture, captured the way the hardware captures it: BY VALUE.
///
/// `sceGxmSetFragmentTexture` and `sceGxmPrecomputedFragmentStateSetTexture` both COPY the
/// 16-byte `SceGxmTexture` into driver-owned memory at the moment of the call - the caller's
/// struct is `const` and is free to be a stack temporary or a slot in a recycled scratch pool.
/// Keeping only the POINTER and re-reading it at draw time therefore reads whatever the guest
/// has since put there, which on this title is zeros: a race frame's post-process chain bound
/// its three inputs through a precomputed state whose texture array had already been recycled,
/// every one of them decoded as a null data pointer, and the light, bloom and composite passes
/// all rendered pure black. The frame still looked like a frame, only far too dark.
///
/// The ADDRESS is kept beside the words because the two remaining side tables
/// (`texture_formats`, `texture_extra`) are keyed by it - see
/// `vitaslop-host-call-reference-semantics` for why identity and value both have to survive.
/// The sampler wrap modes, filters, LOD bias, gamma and mip count used to be a third such
/// table and are now read out of `words` itself, which is what lets them survive the by-value
/// copy an address-keyed table cannot follow.
// `PartialEq` so a rebind of the SAME texture can be told from a real change - see
// `VitaState::vertex_texture_gen`.
#[derive(Clone, Copy, Debug, Default, PartialEq, Eq)]
struct TextureBinding {
    unit: u32,
    /// The guest `SceGxmTexture*` the binding came from. Identity only - never re-read for
    /// control words.
    addr: u32,
    /// The four control words as they read AT BIND TIME.
    words: [u32; 4],
    /// True when this binding arrived through a precomputed fragment state. Per-binding, not a
    /// global "last path used" flag: the two paths are live at the same time, and a global one
    /// mislabels whichever binding did not happen last - which is exactly how a precomputed
    /// binding spent a session being investigated as a direct one.
    from_precomputed: bool,
}

impl TextureBinding {
    /// Snapshot the 16 control words at `addr` NOW - the moment the guest handed the texture to
    /// GXM. A null handle yields an all-zero binding, which the caller treats as an unbind.
    fn read(ctx: &GuestCtx, unit: u32, addr: u32, from_precomputed: bool) -> Self {
        let words = if addr == 0 {
            [0; 4]
        } else {
            [ctx.read_u32(addr), ctx.read_u32(addr + 4), ctx.read_u32(addr + 8), ctx.read_u32(addr + 12)]
        };
        TextureBinding { unit, addr, words, from_precomputed }
    }

    /// True when the captured control words are all zero - a handle that was never a texture.
    fn is_null(&self) -> bool {
        self.words.iter().all(|&w| w == 0)
    }
}


/// The ONE piece of per-texture state that cannot live in the guest's own control words.
///
/// Everything else this used to hold - the wrap modes, the three filters, the LOD bias, the
/// gamma mode and the mip count - is now written into and read out of the guest's 16-byte
/// `SceGxmTexture`, where the hardware keeps it (see `vita::gxm::texword0`). This shadow was
/// address-keyed, and a `SceGxmTexture` is a POD the guest copies freely, so a copied texture
/// silently lost all of it - the identical defect the doc on [`PrecomputedDraw`] below describes
/// for precomputed draws, which this codebase had already fixed there.
///
/// The byte stride stays, and the reason is the refuse-to-guess rule rather than convenience: a
/// `LINEAR_STRIDED` texture spreads its stride across word 0's `stride_ext`, `stride_low` and
/// `stride` fields, and the vitasdk header documents their WIDTHS but not how they COMPOSE
/// ("stride extension", "internal stride lower bits"). Packing it would mean inventing that, and
/// a wrong packing would corrupt the neighbouring fields that ARE understood. So this holds what
/// the guest passed to `sceGxmTextureInitLinearStrided`, and a by-value copy of a strided texture
/// still loses its stride - which is a known, bounded gap, not a solved one.
#[derive(Clone, Copy, Default)]
struct TextureExtra {
    /// Explicit byte stride for a `LINEAR_STRIDED` texture (0 = derive from width x
    /// bytes-per-pixel, as the driver does for every other layout).
    byte_stride: u32,
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
    streams: [u32; pdraw::STREAM_SLOTS],
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
/// Snapshots of guest texture bytes, keyed by (guest data address, byte length).
///
/// # The problem
/// The capture renderer needs a texture's PIXELS, and the only place they exist is
/// guest memory, so a draw that binds a texture has to read them out. Re-reading
/// every bound texture every frame is ~15 MB of allocate-and-copy per frame on a real
/// 3D title - measured at a QUARTER of the whole frame budget, the single most
/// expensive thing the capture did.
///
/// Almost none of that is a re-read of something that changed: a title uploads its
/// textures at load and samples them for the rest of the level.
///
/// # Why this compares instead of tracking invalidation
/// The obvious fix is to keep the snapshot until something invalidates it - a render
/// into that memory, a `sceGxmTextureInit*` over it, an unmap. That was tried, and a
/// verifier caught a texture this title writes DIRECTLY with the CPU, through no GXM
/// call at all. There is no call to hook, so no invalidation list can be complete,
/// and an incomplete one shows up as a stale texture: a silent visual bug.
///
/// So the snapshot is CHECKED rather than trusted. On each first use in a scene the
/// retained bytes are compared against guest memory borrowed in place
/// ([`GuestMemory::borrow`]) - no second copy - and only a texture that actually
/// changed is re-read. That is exactly as correct as re-reading everything, because
/// the compared bytes are the same bytes the copy would have taken; it just replaces
/// an allocate-plus-copy with a `memcmp` that early-exits on the first difference.
///
/// A backing that cannot lend its bytes (`borrow` returns `None`) falls back to
/// re-reading every scene - slower, still correct.
///
/// # And why the compare is skipped where the engine can prove it pointless
/// The compare is exact and it is enormous: measured on a live race, 116.8 MB a frame,
/// 40% of the whole frame, RE-READING 0.0 MB - memory bandwidth spent proving nothing
/// changed. An engine that stamps its guest stores
/// ([`GuestMemory::dirty_since`]) can answer the same question by reading one byte per
/// 4 KB page, and that is a proof, not a sample: it is the same guarantee, not a
/// weaker one. Where the engine offers no stamps, the `memcmp` still runs. Neither
/// path approximates, which is why there is no knob here.
#[derive(Default)]
struct TextureSnapshots {
    entries: FxHashMap<(u32, usize), Arc<[u8]>>,
    /// The guest-store epoch each entry's bytes were established at, on an engine that
    /// stamps stores. Absent for an entry means "no proof available" and costs a
    /// compare, which is why a wrapped epoch can simply drop the whole map.
    stamps: FxHashMap<(u32, usize), u8>,
    /// Total bytes held, so the cache cannot grow without bound across a long run.
    bytes: usize,
    /// Entries already checked this scene, so a texture bound by fifty draws is
    /// compared once, not fifty times. The cache's claim is "these bytes are
    /// unchanged since the last frame", and once per scene is what tests it.
    checked_this_scene: FxHashSet<(u32, usize)>,
    /// The guest-store epoch every snapshot taken in the CURRENT scene is stamped with, or 0
    /// before this scene has taken one. See [`Self::restamp`] for why a scene shares one.
    scene_stamp: u8,
    /// VERTEX buffer snapshots, by `(guest address, byte length)`.
    ///
    /// # Why vertices live in the texture cache, and why they are checked DIFFERENTLY
    /// They are here because they share the one thing that makes either work: the guest-store
    /// epoch, which is global guest state. Two caches arming their own scene stamps would spend
    /// the one-byte epoch twice as fast, which is the exact failure this whole mechanism just
    /// came back from.
    ///
    /// They are checked differently because the risk is not the same. A texture bound by fifty
    /// draws is compared once a scene, on the argument that a title does not rewrite a texture
    /// while drawing with it. That argument does NOT hold for geometry: writing vertices and
    /// drawing them, repeatedly, within one scene is exactly how dynamic geometry works - UI
    /// text, particles, anything skinned on the CPU. So a vertex read never takes the
    /// once-a-scene shortcut and always asks the dirty map, which is one page-range read against
    /// re-copying the whole buffer.
    ///
    /// The payoff is that STATIC geometry - a track, a ship hull, written once at load - has a
    /// page stamp far below the current epoch and is proved unchanged without being read at all.
    /// MEASURED on the device before this existed: `draw: read+interleave vertices` moved
    /// **6.06 MB every frame**, as a fresh allocation and a full copy per draw, 953 draws a
    /// frame, each one a boundary crossing in the browser.
    vertex_entries: FxHashMap<(u32, usize), Arc<[u8]>>,
    /// Epoch each vertex snapshot's bytes were established at. Same contract as `stamps`.
    vertex_stamps: FxHashMap<(u32, usize), u8>,
    /// Bytes held by `vertex_entries`, against [`vertex_snapshot_budget`].
    vertex_bytes: usize,
    /// Entries already checked this FRAME. See [`Self::get_or_read`]: a texture that has
    /// never been observed to change is compared once per frame rather than once per
    /// scene, which is where nearly all of the compare bandwidth went.
    checked_this_frame: FxHashSet<(u32, usize)>,
    /// Entries whose bytes HAVE been seen to change at least once. These keep the
    /// per-scene cadence, because they are the ones the cadence exists for.
    mutable: FxHashSet<(u32, usize)>,
    /// Mean colour of a snapshot, keyed by the identity of the byte buffer itself and
    /// holding a strong reference to it.
    ///
    /// The flat ambient term of every lit draw is the average of the bound irradiance
    /// map ([`crate::render::texture_mean_rgb`]), which samples a thousand texels and
    /// decodes each - per DRAW, for a value that is a property of the texture. Keyed
    /// by the buffer's address rather than the texture's guest address because that is
    /// what actually determines the answer: a re-read allocates a new buffer, so a
    /// changed texture cannot hit a stale mean. The strong reference is what makes the
    /// address a valid key - without it a freed buffer's address could be reused.
    means: FxHashMap<usize, ([f32; 3], Arc<[u8]>)>,
    /// INDEX buffer snapshots, already scanned and REBASED, by `(guest address, byte length,
    /// element size)`.
    ///
    /// # Why the derived value and not the bytes
    /// Every draw read its index buffer out of guest memory, folded the whole thing for its
    /// min/max, rewrote every element to rebase it onto the vertex window, and then converted the
    /// `Vec` into an `Arc` - a second allocation and a second full copy. Four passes over the
    /// indices per draw, ~650 draws a frame, and in the browser the read is also a crossing into
    /// JS. MEASURED on the browser-like configuration of `bench --at 7400`: `draw: read indices`
    /// 1.4% and `draw: scan/rebase indices` 2.3% of the window, on top of the allocation churn
    /// neither of those timers can see.
    ///
    /// All three outputs are a pure function of the source bytes, so the cache holds the ANSWER
    /// rather than the input: a hit costs one dirty-map query and an `Arc` clone, and does none
    /// of the four passes.
    ///
    /// Checked like VERTICES, not like textures - it always asks the dirty map and never takes
    /// the once-a-scene shortcut. Geometry written and drawn within one scene is exactly how
    /// dynamic meshes work, and an index buffer is geometry.
    index_entries: FxHashMap<(u32, usize, usize), (Arc<[u8]>, u32, u32)>,
    /// Epoch each index snapshot was established at. Same contract as `stamps`.
    index_stamps: FxHashMap<(u32, usize, usize), u8>,
    /// Bytes held by `index_entries`, against [`vertex_snapshot_budget`] - the same budget,
    /// because they are the same kind of thing and a level's geometry turns over together.
    index_bytes: usize,
    /// Everything [`decode_texture`] derives from a binding's four CONTROL WORDS, memoised by
    /// those words. `None` is a binding the decode drops (an unsizeable format), which is worth
    /// remembering for exactly the same reason.
    ///
    /// # Why this is the largest capture phase left, and why a memo is the answer
    /// `draw: snapshot textures` MEASURED at **11.7% of a browser-like race window, moving
    /// 0.0 MB** - so it is not volume, it is per-draw arithmetic repeated on bindings that do
    /// not change. Every draw re-derived, per bound unit: the recorded format (a hash lookup),
    /// the nearby-handle diagnostic (another), the base format and swizzle, the size fields, the
    /// block geometry, the mip-level clamp and the whole chain's byte extent
    /// ([`crate::render::level_offset`] walks every level). A race frame does that ~2,000 times
    /// for a handful of distinct textures.
    ///
    /// All of it is a pure function of the control words, which is exactly what a GXM binding
    /// IS: the words are copied BY VALUE at bind time ([[vitaslop-texture-binding-by-value]]),
    /// so two bindings with the same words describe the same texture. What is NOT cached is the
    /// PIXELS - those still go through [`Self::get_or_read`] every time, so a texture the guest
    /// rewrites is still seen to change. The memo covers the description; the snapshot covers
    /// the contents.
    ///
    /// Cleared when a recorded format changes (see `set_texture_format`), because that is the
    /// one input that is not in the key.
    templates: FxHashMap<[u32; 4], Option<TextureTemplate>>,
    /// A whole DRAW's worth of snapshotted textures, by the bindings that produced it - kept
    /// ACROSS scenes and RE-PROVEN on first use in each new scene.
    ///
    /// # Why a cache of the whole answer is exactly equivalent
    /// Within a scene: [`Self::get_or_read`] compares a texture at most once per scene; the
    /// first draw of the scene that binds it establishes the buffer, and every later draw in
    /// that scene gets the SAME `Arc` back by construction ("the guest cannot have run since -
    /// a host call is not preemptible"). So for a fixed set of bindings, every draw of a scene
    /// was already guaranteed to produce a bitwise identical list; this just stops rebuilding
    /// it.
    ///
    /// Across scenes: the guest HAS run, so nothing is believed until it is re-proven. A kept
    /// entry carries the [`Self::decoded`] keys and pixel buffers it was built from
    /// ([`SetEntry::deps`]); the first hit of a new scene runs each through
    /// [`Self::get_or_read`] - the stamps-then-compare ladder a rebuild would use - and the
    /// entry survives only if every buffer comes back POINTER-identical. That is the same
    /// proof, so keeping the entry changes what is rederived, never what is answered. This
    /// used to be cleared whole at every scene start, and a race frame is ~11 scenes binding
    /// the same sets, so ~45% of draws paid a full set rebuild for an answer the stamps
    /// already knew. `VITASLOP_TEX_MEMO_PER_SCENE=1` restores that cadence as the A/B arm.
    ///
    /// A race frame binds a handful of distinct texture sets across ~650 draws, so the hit rate
    /// is essentially the draw count. What a hit avoids is the whole of `decode_texture` per
    /// unit - the template probe, the recorded-format probe, the snapshot probe and the
    /// `Vec<BoundTexture>` the draw would otherwise allocate and fill.
    ///
    /// The key folds the fragment program header as well as the bindings, because the ALBEDO
    /// REORDER below is a function of the program's reflection - two draws binding the same
    /// textures through different fragment programs want them in different orders, and a shared
    /// entry would hand the second one the first one's albedo. The fold is 64 bits and a fold
    /// is not a proof, so every hit also verifies the EXACT key material stored beside the
    /// list ([`SetEntry::bindings`], [[vitaslop-content-hash-cache-must-verify]]).
    snapshot_sets: FxHashMap<u64, SetEntry>,
    /// One finished [`crate::capture::BoundTexture`] per BINDING - the per-unit counterpart
    /// of [`Self::snapshot_sets`], and the one that actually carries a race frame. See the
    /// memo in `decode_texture` for why the set cache does not: the combinations of bindings
    /// a scene draws are many, the bindings themselves are few, and a set miss re-read every
    /// one of its units.
    ///
    /// Kept across scenes under the same re-proof discipline as the set cache - see
    /// [`DecodedEntry`] and [`Self::decoded_validated`]. An `Err` result is a binding the
    /// decode drops, with its drop code; it is memoised within a scene (so a loss is counted
    /// per draw without being re-derived per draw) and re-attempted each new scene, exactly
    /// as it was when this map died with the scene.
    decoded: FxHashMap<(u32, [u32; 4]), DecodedEntry>,
    /// Which scene the run is in, monotonically. What makes the two memos above safe to KEEP
    /// across scenes: an entry proven current in scene N says nothing about scene N+1 (the
    /// guest ran in between), so each carries the scene it was last proven in, and a stale one
    /// is re-proven - one [`Self::get_or_read`] per underlying snapshot and a pointer compare -
    /// before it is believed. See [`Self::begin_scene`].
    scene_seq: u64,
    /// >>> THE PREVIOUS DRAW'S ANSWER, KEYED ON THE BYTES IT WAS DERIVED FROM.
    ///
    /// [`Self::snapshot_sets`] already stops the LIST being rebuilt, but every draw still
    /// paid the gate in front of it: decode sixteen sampler slots out of the context
    /// snapshot, fold them into a 64-bit key, probe the map, and verify the exact key
    /// material the fold cannot prove. MEASURED in the browser at `draw: decode texture
    /// bindings (EVERY draw)` **0.54 ms of a 9.7 ms race frame over 465 draws** - 1.16 us a
    /// draw to re-derive an answer that, inside a batch of draws sharing a material, is the
    /// same one every time.
    ///
    /// Consecutive draws are exactly where that redundancy lives, so this holds ONE entry:
    /// the finished list and the raw sampler-block bytes behind it. A hit is a 384-byte
    /// `memcmp` and an `Arc` clone.
    ///
    /// It is EXACT, not a heuristic, and its two guards are what make it so:
    /// - the sampler bytes are the whole of what the bindings are (GXM copies a texture's
    ///   control words BY VALUE at bind time, [[vitaslop-texture-binding-by-value]]), and
    ///   the fragment program header is the other half of the key because the albedo
    ///   REORDER is a function of its reflection;
    /// - the entry is only believed inside the SCENE it was proven in. Across a scene the
    ///   guest has run and the pixels behind the list have to be re-proven, which is
    ///   [`Self::set_validated`]'s job; within one, `get_or_read` compares a texture at most
    ///   once per scene, so the previous draw's proof is this draw's proof by construction.
    last_set: Option<LastSet>,
    /// The same shortcut for the VERTEX stage, keyed on [`VitaState::vertex_texture_gen`]
    /// rather than on bytes: that list is host-side with one mutator, so a counter answers
    /// "unchanged since the last draw" exactly and in one comparison.
    last_vertex_set: Option<LastVertexSet>,
}

/// The previous draw's finished VERTEX-stage texture list. See
/// [`TextureSnapshots::last_vertex_set`].
struct LastVertexSet {
    /// The binding-list generation the answer was decoded at.
    generation: u64,
    /// The [`TextureSnapshots::snapshot_sets`] entry it came from, re-probed on every hit for
    /// the same reason [`LastSet::key`] is.
    key: u64,
    list: Arc<[crate::capture::BoundTexture]>,
}

/// The previous draw's finished fragment texture list. See [`TextureSnapshots::last_set`].
struct LastSet {
    /// The raw sampler block the list was decoded from - the exact key material.
    span: Box<[u8]>,
    /// The bound fragment program's header, which decides the albedo reorder.
    fheader: u32,
    /// The [`TextureSnapshots::snapshot_sets`] entry this answer came from. A hit re-probes
    /// it, which is what makes this memo inherit every invalidation the set cache has rather
    /// than needing its own copy of them - see [`TextureSnapshots::set_from_previous_draw`].
    key: u64,
    /// The answer.
    list: Arc<[crate::capture::BoundTexture]>,
}

/// One decoded binding, kept across scenes. See [`TextureSnapshots::decoded`].
#[derive(Clone)]
struct DecodedEntry {
    /// The decode's answer: the finished texture, or the drop code the decode reported.
    res: Result<crate::capture::BoundTexture, u8>,
    /// The `(addr, len)` snapshot the pixels came from - the [`TextureSnapshots::get_or_read`]
    /// key a re-proof asks about. `None` for a dropped binding, which has no pixels.
    snap: Option<(u32, usize)>,
    /// The decode fell back to level 0 (mip chain unreadable). Kept on the per-scene cadence -
    /// dropped when its scene goes stale - so the retry semantics are unchanged: the fuller
    /// chain is re-attempted each scene exactly as before the memo survived scenes.
    degraded: bool,
    /// Scene this entry was last PROVEN current in ([`TextureSnapshots::scene_seq`]).
    valid_scene: u64,
}

/// A whole draw's finished texture list, kept across scenes. See
/// [`TextureSnapshots::snapshot_sets`].
struct SetEntry {
    list: Arc<[crate::capture::BoundTexture]>,
    /// The stage tag the map key folded first: the fragment program header for the fragment
    /// stage, [`VERTEX_STAGE_TAG`] for the vertex stage. Verified on every hit.
    tag: u64,
    /// The EXACT bindings (`unit, addr, control words`) behind the map key's 64-bit fold,
    /// verified on every hit - a fold is not a proof
    /// ([[vitaslop-content-hash-cache-must-verify]]).
    bindings: Vec<(u32, u32, [u32; 4])>,
    /// The [`TextureSnapshots::decoded`] key AND the pixel buffer `list` holds for each
    /// non-null unit, for the per-scene re-proof: the entry survives a scene change only if
    /// every one of these re-validates to the SAME buffer. `None` marks a set that must not
    /// outlive its scene (a unit was dropped or degraded), so building it - and counting its
    /// drops - stays a per-scene event.
    deps: Option<Vec<((u32, [u32; 4]), Arc<[u8]>)>>,
    /// How many null-handle units `list` carries. Their per-set-build drop-0 notes are
    /// re-issued when a kept set is re-proven, so the counter keeps its per-scene cadence.
    null_drops: u32,
    /// Scene this entry was last PROVEN current in.
    valid_scene: u64,
}

/// The stage tag [`SetEntry`] and the vertex-stage set key carry, so a vertex list and a
/// fragment list of identical bindings cannot collide in the shared map ("VERTEX").
const VERTEX_STAGE_TAG: u64 = 0x5645_5254_5845_5300;

/// What a binding's control words say, once, so no draw has to work it out again. See
/// [`TextureSnapshots::templates`].
#[derive(Clone, Copy)]
struct TextureTemplate {
    base_format: u32,
    swizzle: u32,
    tex_type: u32,
    width: u32,
    height: u32,
    stride: u32,
    faces: u32,
    /// Bytes one face occupies including `levels` mip levels, and the whole snapshot's length.
    face_bytes: u32,
    read_len: u32,
    levels: u32,
    /// The same two if the chain read comes up short and the texture falls back to level 0 -
    /// precomputed here so the fallback costs no arithmetic either.
    level0_bytes: u32,
    level0_read_len: u32,
    data_addr: u32,
    u_addr_mode: u32,
    v_addr_mode: u32,
    lod_bias: u32,
    min_filter: u32,
    mag_filter: u32,
    gamma: u32,
    mip_filter: u32,
}

/// Distinct control-word sets the template memo will hold. A title binds a few hundred distinct
/// textures; this is a bound on a pathological case, not a working limit. Cleared whole when hit,
/// which costs re-derivation and never correctness - the key IS the input.
const TEXTURE_TEMPLATE_CAP: usize = 4096;

/// Is the snapshot check allowed to run once per FRAME instead of once per SCENE? The one
/// inexact cadence, and the only one that has to be asked for - see [`texture_check`].
fn texture_check_per_frame() -> bool {
    texture_check() == TextureCheck::Frame
}

/// `VITASLOP_TEX_MEMO_PER_SCENE`: clear the per-binding and per-set texture memos at every
/// scene start (the pre-cross-scene cadence) instead of keeping them and re-proving each
/// entry on first use per scene. The A/B arm for the cross-scene memo; BOTH arms are exact -
/// the difference is only what gets rederived, never what is answered.
fn tex_memo_per_scene() -> bool {
    static ON: std::sync::OnceLock<bool> = std::sync::OnceLock::new();
    *ON.get_or_init(|| crate::knobs::flag("VITASLOP_TEX_MEMO_PER_SCENE"))
}

/// Say - once - that a set-cache hit's stored bindings did not match the probe's, i.e. the
/// 64-bit fold collided. Treated as a miss, so it costs a rebuild and never an answer; said
/// aloud because a collision here is rare enough that one occurring is worth a line.
fn report_set_key_collision() {
    static ONCE: std::sync::Once = std::sync::Once::new();
    ONCE.call_once(|| {
        tracing::warn!(
            target: "vitaslop::gxm",
            "texture set-cache key collision: two distinct binding sets folded to one 64-bit \
             key. Handled exactly (the exact-match check treated it as a miss); noted because \
             it should be vanishingly rare."
        );
    });
}

/// How often, and by what means, a retained texture snapshot is re-validated.
#[derive(Clone, Copy, PartialEq, Eq)]
enum TextureCheck {
    /// The default and the exact one: once per scene, by the guest-store stamps where
    /// the engine has them and by a byte compare where it does not.
    Scene,
    /// Once per FRAME for a texture never yet seen to change. Faster, NOT exact.
    Frame,
    /// Once per scene, by a byte compare ALWAYS - the stamps are ignored even where
    /// the engine has them.
    ///
    /// Exact, and slower than `scene` on every engine, so it is not a performance
    /// option: it exists to falsify the stamps. Two runs of one recipe that differ
    /// only in this must be BIT-IDENTICAL, and if they are not, the dirty map missed a
    /// store - which is a silent stale-texture bug and the only failure mode the
    /// mechanism has.
    Bytes,
}

/// How a retained texture snapshot is re-validated (`VITASLOP_TEXTURE_CHECK`): `scene`
/// (default, exact), `frame` (faster, not exact) or `bytes` (exact, slower, ignores the
/// guest-store stamps - the A/B that falsifies them). A value that is none of those is an
/// ERROR rather than a silent fallback: a run configured by a typo would otherwise quietly
/// publish frames at the accuracy nobody asked for.
fn texture_check() -> TextureCheck {
    static ON: std::sync::OnceLock<TextureCheck> = std::sync::OnceLock::new();
    *ON.get_or_init(|| match crate::knobs::var("VITASLOP_TEXTURE_CHECK") {
        Err(_) => TextureCheck::Scene,
        Ok(v) => match v.trim() {
            "scene" => TextureCheck::Scene,
            "frame" => TextureCheck::Frame,
            "bytes" => TextureCheck::Bytes,
            other => panic!(
                "VITASLOP_TEXTURE_CHECK={other:?} is not `scene` (exact, the default), \
                 `frame` (faster, one scene of staleness the first time a texture changes) \
                 or `bytes` (exact, slower, ignores the guest-store stamps - the A/B that \
                 falsifies them)"
            ),
        },
    })
}

/// Say - once - that the per-frame texture check is in effect. It trades exactness for
/// speed, and an approximation nobody can see in the log is one that gets quoted as a
/// measurement of the real thing. [[vitaslop-fallback-must-report]]
fn report_texture_check_per_frame() {
    static SAID: std::sync::Once = std::sync::Once::new();
    SAID.call_once(|| {
        tracing::warn!(
            target: "vitaslop::gxm",
            "VITASLOP_TEXTURE_CHECK=frame: a texture never yet seen to change is compared \
             against guest memory ONCE PER FRAME, not once per scene. Faster, and NOT exact - \
             a texture the guest rewrites mid-frame for the first time renders stale for the \
             rest of that frame. Unset this for the exact per-scene check."
        );
    });
}

/// Byte budget for retained VERTEX snapshots, separate from the texture one so a level's
/// geometry and its textures cannot evict each other.
///
/// A race frame of this project's heaviest title reads about 6 MB of vertices, and a level's
/// whole static geometry is a small multiple of that, so 64 MB holds it comfortably while staying
/// a bound rather than a licence to grow with the run.
///
/// Deliberately a CONSTANT and not a knob. A knob here would be a way of not deciding, and the
/// engine that pays for this runs whatever is defaulted ([[vitaslop-pick-the-default-dont-add-a-knob]]).
/// If it ever needs changing, the number to change it against is the `draw: read+interleave
/// vertices` figure in the device panel's bytes line.
const fn vertex_snapshot_budget() -> usize {
    64 << 20
}

/// Byte budget for retained texture snapshots. Past it the cache is cleared whole
/// (and says so): a level's working set is what it holds, and a title that exceeds
/// this is telling us something worth seeing rather than something to silently trim.
///
/// # What clearing it actually costs, measured
/// "Cleared whole" is not a local event. Every retained entry is an `Arc` that the
/// RENDERER's decode cache is keyed on by IDENTITY, so dropping them all means the next
/// scene re-reads every texture into a fresh buffer, misses the decode cache on all of
/// them, and re-decodes the whole working set inside one frame. MEASURED in the browser
/// mid-race on a 703-draw frame: **263 textures decoded, 178 MB of RGBA8 produced, and
/// `build` took 2,128 ms** - against 1.7 ms for a 430-draw frame that decoded nothing.
///
/// # The budget was NOT the cause of that burst, and the default is unchanged
/// Raising it to 512 MB was tried and MEASURED: the clear never fired at either setting,
/// and the burst frame was identical (2,145 ms against 2,128 ms, 263 decodes either way).
/// The burst is the first PRESENTED frame after a fast-forward, which presents nothing and
/// so leaves every cache cold - a property of the harness, not of this budget. The default
/// stays at 192 MB because nothing measured asked for more, and a constant raised on a
/// refuted theory is a constant fitted to noise.
///
/// What the investigation DID leave behind is the report below, which is now a `warn`:
/// when this does fire it costs a full re-decode, and it used to say so only at `info`.
/// `VITASLOP_SNAPSHOT_BUDGET_MB` overrides it - a mobile target will want it smaller.
fn texture_snapshot_budget() -> usize {
    static CELL: std::sync::OnceLock<usize> = std::sync::OnceLock::new();
    *CELL.get_or_init(|| {
        crate::knobs::var("VITASLOP_SNAPSHOT_BUDGET_MB")
            .ok()
            .and_then(|s| s.trim().parse::<usize>().ok())
            .unwrap_or(192)
            << 20
    })
}

impl TextureSnapshots {
    fn new() -> Self {
        Self::default()
    }

    /// The current bytes of the texture of `len` bytes at guest address `addr`,
    /// reusing the retained snapshot when guest memory still holds the same bytes.
    fn get_or_read(&mut self, ctx: &GuestCtx, addr: u32, len: usize) -> Arc<[u8]> {
        // One epoch per scene, taken at the scene's first snapshot - see `restamp` for why this
        // is not per texture, and what it cost when it was.
        self.arm_scene_stamp(ctx);
        // The retained buffer, kept past the borrow below so the re-read at the bottom can
        // hand back the SAME `Arc` when the bytes turn out to be identical - see there.
        let mut retained: Option<Arc<[u8]>> = None;
        if let Some(p) = self.entries.get(&(addr, len)) {
            retained = Some(p.clone());
            // Already compared this scene: the guest cannot have run since (a host
            // call is not preemptible), so nothing can have changed.
            if !self.checked_this_scene.insert((addr, len)) {
                return p.clone();
            }
            // >>> OPT-IN ONLY (`VITASLOP_TEXTURE_CHECK=frame`): compare a texture never
            // seen to change once per FRAME instead of once per SCENE.
            //
            // MEASURED on a live race with `bench --at`: the compare below is 40.2% of the
            // whole frame - 1332 ms of a 3.31 s window - moving 116.8 MB per frame at
            // 10.5 GB/s. It already runs at memory bandwidth, so the only thing left to
            // change is how often it runs, and a frame here is eleven scenes. Doing it per
            // frame took the title from 36.8 to 47.8 fps.
            //
            // It is NOT the default, because it is not exact and this emulator's job is to
            // be exact. A texture that changes mid-frame for the FIRST time is seen one
            // scene late (per-scene from then on, since a change puts it in `mutable`
            // permanently) - one wrong frame, which is a visible artefact and not a
            // rounding error. The strict cadence is what a claim of hardware accuracy
            // rests on, so the fast path has to be asked for, and says so when taken.
            if texture_check_per_frame()
                && !self.mutable.contains(&(addr, len))
                && !self.checked_this_frame.insert((addr, len))
            {
                report_texture_check_per_frame();
                return p.clone();
            }
            // >>> THE EXACT SHORTCUT: ask the engine whether the guest stored here at all.
            //
            // On an engine that stamps its stores ([`GuestMemory::dirty_since`]), a
            // texture no store has touched since these bytes were read CANNOT have
            // changed. That is a proof, not a sample - the same guarantee the memcmp
            // below gives - for the cost of reading one byte per 4 KB page instead of
            // every byte of the texture twice. A missing stamp or a `None` answer is
            // NOT "clean": both fall through to the compare.
            //
            // Charged to the compare phase, because it IS the compare on that engine -
            // filing it elsewhere would make the phase look solved rather than moved.
            let untouched = texture_check() != TextureCheck::Bytes && {
                let _t = crate::perf::scope(crate::perf::Phase::DrawTextureCompare);
                match self.stamps.get(&(addr, len)) {
                    Some(&s) => ctx.dirty_since(addr, len, s) == Some(false),
                    None => false,
                }
            };
            if untouched {
                // >>> RE-STAMP, EVEN THOUGH NOTHING WAS READ. The stamps have just PROVED
                // these bytes current as of this scene, so recording that costs a hash write
                // and makes the entry's stamp track the epoch instead of standing still at
                // whatever it was established at.
                //
                // Without it a texture that is never written keeps its original stamp for the
                // life of the run, which is what made `rebase_epoch_if_exhausted` find a floor
                // of 1 and decline to renumber every time - MEASURED: 3 wraps and 0
                // renumberings per 30-present window, with the whole mechanism inert.
                let current = retained.clone().expect("set whenever an entry exists");
                self.restamp(addr, len);
                return current;
            }
            // Timed and COUNTED separately (see `Phase::DrawTextureCompare`): this is a
            // memcmp of the whole texture, and it is charged to the snapshot phase while
            // moving no bytes through `note_bytes` - so the phase reported "0.0 MB/frame"
            // and read as pure per-draw overhead. The bytes compared are the size of the
            // problem; the bytes re-read are only what actually changed.
            // `Some(false)` unchanged, `Some(true)` changed, `None` a backing that cannot
            // lend its bytes - which is not evidence either way, so it falls through to the
            // re-read without being recorded as mutable.
            let changed = {
                let _t = crate::perf::scope(crate::perf::Phase::DrawTextureCompare);
                ctx.borrow_bytes(addr, len).map(|live| {
                    crate::perf::note_bytes(crate::perf::Phase::DrawTextureCompare, len);
                    live != &p[..]
                })
            };
            match changed {
                Some(false) => {
                    // Through `retained`, not `p`: stamping needs `&mut self` and `p`
                    // borrows the entry table. They are the same `Arc`.
                    let current = retained.clone().expect("set whenever an entry exists");
                    // The compare has just re-established that these bytes are current,
                    // so stamp them: without this the shortcut above could never take
                    // hold for a texture whose stamp was dropped (a wrapped epoch, a
                    // cleared cache), and it would pay the full compare for ever.
                    self.restamp(addr, len);
                    return current;
                }
                // Seen to change ONCE is enough to keep it on the per-scene cadence for
                // the rest of the run. This is the set the frame-cadence shortcut above
                // deliberately does not apply to.
                Some(true) => {
                    self.mutable.insert((addr, len));
                    // Already proved different - the re-read below must not compare again.
                    retained = None;
                }
                None => {}
            }
        }
        // >>> COUNTED, BECAUSE ON THE BROWSER THIS *IS* THE COMPARE AND IT LOOKED FREE.
        //
        // An engine that can LEND its bytes compares them in place above and charges the
        // volume to `DrawTextureCompare`. An engine that cannot - the browser, permanently -
        // reaches here every scene a texture is not proven clean, copies the whole texture
        // across the guest boundary purely to find out whether it changed, and then usually
        // hands the retained buffer back. That copy was charged to NOTHING unless the bytes
        // had actually changed, so the one engine that pays it reported 0.0 MB for it.
        // [[vitaslop-count-bytes-when-there-is-no-clock]]: a volume only one engine pays is
        // exactly the volume that needs a counter.
        let raw = ctx.read_bytes(addr, len);
        if retained.is_some() {
            crate::perf::note_bytes(crate::perf::Phase::DrawTextureCompare, raw.len());
        }
        // A backing that cannot LEND its bytes (`borrow_bytes` returned `None`) reaches here
        // every scene, whatever the texture is doing, so the copy above is the only way to
        // find out whether anything changed. Compare it: consumers key their decode and
        // upload caches on this buffer's IDENTITY, so handing back a fresh `Arc` holding
        // byte-identical pixels would re-decode and re-upload an unchanged texture every
        // scene. Same answer, same bytes, one buffer.
        if let Some(p) = retained {
            if raw[..] == p[..] {
                self.restamp(addr, len);
                return p;
            }
        }
        let bytes: Arc<[u8]> = raw.into();
        crate::perf::note_bytes(crate::perf::Phase::DrawTextures, bytes.len());
        self.insert(addr, len, bytes.clone());
        self.restamp(addr, len);
        bytes
    }

    /// Record that this entry's bytes are current as of THIS SCENE, so a later
    /// [`GuestMemory::dirty_since`] can prove nothing has touched them.
    ///
    /// # >>> THE EPOCH IS ONE BYTE, AND ADVANCING IT PER SNAPSHOT SPENT IT IN HALF A FRAME
    /// This used to call [`GuestCtx::bump_dirty_epoch`] itself, once per re-stamped texture.
    /// The epoch is a single byte, compared with `>=`, so it may not wrap silently - on wrap the
    /// whole dirty map is zeroed and every stamp taken before it is dropped as a lie.
    ///
    /// MEASURED on a live race with `bench --at 7400`: hundreds of re-stamps a frame against
    /// **254** usable epoch values. The map was therefore wiped repeatedly and almost no stamp
    /// survived long enough to prove anything, so the shortcut this mechanism exists for barely
    /// fired and nearly every texture fell through to the full memcmp. Half the guest CPU was
    /// spent proving, the hard way, something the stamps already knew.
    ///
    /// **The A/B, same build, same window, dirty map ON in both arms** - which is the BROWSER's
    /// configuration, and the browser is the only engine that runs with it (native leaves the
    /// map off because wasmtime bills the marks and they would speed the guest clock up):
    ///
    /// | arm | `texture snapshot compare` | bytes/frame | frame |
    /// |---|---|---|---|
    /// | epoch per snapshot | 35.9% of the window | 76.2 MB | ~22.2 ms (45 fps) |
    /// | one epoch per scene | **2.8%** | **2.9 MB** | **10.9 ms (92 fps)** |
    ///
    /// A 26x cut in bytes moved and 2.05x on guest CPU, from advancing a counter less often.
    ///
    /// The epoch does not need to advance per texture. It needs to advance whenever the GUEST
    /// may have run since the last batch of stamps, and within a scene every snapshot is taken
    /// from host calls that cannot be preempted. So it advances ONCE per scene ([`Self::
    /// scene_stamp`]), which is eleven times a frame on this title instead of five hundred, and
    /// the byte now lasts about twenty-three frames rather than half of one.
    ///
    /// A store made during this scene writes the current epoch, which EQUALS the stamp recorded
    /// here, and `>=` reports it dirty - so sharing one stamp across a scene stays exact in the
    /// direction that matters. It is conservative only for a page written earlier in the same
    /// scene, which costs one compare and then re-stamps clean.
    fn restamp(&mut self, addr: u32, len: usize) {
        if self.scene_stamp != 0 {
            self.stamps.insert((addr, len), self.scene_stamp);
        }
    }

    /// The current bytes of the VERTEX buffer of `len` bytes at guest address `addr`, reusing
    /// the retained snapshot when the guest provably has not stored there since.
    ///
    /// # This is the texture shortcut, aimed at the next biggest thing the capture moves
    /// Every draw used to `read_bytes` its whole vertex range - one allocation and one full copy
    /// out of guest memory - and then convert that `Vec` into an `Arc`, which is a SECOND
    /// allocation and copy. Twice the buffer, twice, per draw. In the browser each read is also a
    /// crossing into JS.
    ///
    /// The dirty map answers "has the guest written here since I read it?" for the cost of one
    /// page-range read, and static geometry answers no for the whole run. When it answers yes the
    /// bytes are re-read, which is exactly the old behaviour, so dynamic geometry is unaffected.
    ///
    /// Returning the SAME `Arc` matters beyond the copy: the renderer's packed-vertex cache keys
    /// on this buffer's identity, so an equal-but-different buffer misses it
    /// ([[vitaslop-content-hash-cache-must-verify]]).
    fn get_or_read_vertices(&mut self, ctx: &GuestCtx, addr: u32, len: usize) -> Arc<[u8]> {
        if len == 0 {
            return Arc::from(&[][..]);
        }
        self.arm_scene_stamp(ctx);
        // NO once-a-scene shortcut here - see `vertex_entries` for why geometry cannot take one.
        if let Some(p) = self.vertex_entries.get(&(addr, len)) {
            let clean = match self.vertex_stamps.get(&(addr, len)) {
                Some(&s) => ctx.dirty_since(addr, len, s) == Some(false),
                None => false,
            };
            if clean {
                return p.clone();
            }
        }
        let raw = ctx.read_bytes(addr, len);
        crate::perf::note_bytes(crate::perf::Phase::DrawVertices, raw.len());
        // Same bytes as last time? Hand back the SAME buffer. The guest may have rewritten the
        // page with identical contents (a rebuilt-every-frame buffer whose contents are static
        // is common), and a fresh `Arc` there would miss every downstream cache keyed on
        // identity while being byte-for-byte the thing they already hold.
        if let Some(p) = self.vertex_entries.get(&(addr, len)) {
            if raw[..] == p[..] {
                let same = p.clone();
                self.stamp_vertices(addr, len);
                return same;
            }
        }
        let bytes: Arc<[u8]> = raw.into();
        // A budget, cleared whole when exceeded. The keys are (address, length) rather than
        // content, so a clear costs re-reads and never correctness - and geometry turns over
        // with the level, so a level change should not leave the previous one's meshes resident.
        if self.vertex_bytes + bytes.len() > vertex_snapshot_budget() {
            self.vertex_entries.clear();
            self.vertex_stamps.clear();
            self.vertex_bytes = 0;
        }
        if let Some(old) = self.vertex_entries.insert((addr, len), bytes.clone()) {
            self.vertex_bytes -= old.len();
        }
        self.vertex_bytes += bytes.len();
        self.stamp_vertices(addr, len);
        bytes
    }

    /// The draw's index buffer, ALREADY scanned and rebased: `(indices, first_vertex,
    /// vertex_count)`.
    ///
    /// # What is cached is the answer, not the input
    /// A draw's index work is four passes: read the bytes out of guest memory, fold them for
    /// their min and max, rewrite every element to rebase it onto the `min..=max` vertex window,
    /// and convert the `Vec` into an `Arc` (a second allocation and a second full copy). All
    /// three outputs are a pure function of the source bytes, so a buffer the guest provably has
    /// not written since is worth exactly one dirty-map query and an `Arc` clone.
    ///
    /// Static geometry - a track, a hull, a UI mesh built once at load - answers "unchanged" for
    /// the whole run, which is the case this exists for. Dynamic geometry re-reads, which is
    /// precisely the old behaviour.
    ///
    /// # The window is part of the value, and it must be
    /// `first_vertex`/`vertex_count` are derived from the SAME bytes as the rebased buffer and
    /// are what the vertex read is sized from. Caching the buffer without them would hand a draw
    /// rebased indices alongside a freshly-computed window, and if those ever disagreed the mesh
    /// would index off the end of its own vertices - a prefix of the geometry, silently.
    fn get_or_read_indices(
        &mut self,
        ctx: &GuestCtx,
        addr: u32,
        len: usize,
        elem: usize,
    ) -> (Arc<[u8]>, u32, u32) {
        if len == 0 {
            return (Arc::from(&[][..]), 0, 0);
        }
        self.arm_scene_stamp(ctx);
        let key = (addr, len, elem);
        // NO once-a-scene shortcut, for the reason `vertex_entries` gives: an index buffer is
        // geometry, and write-then-draw inside one scene is how dynamic meshes work.
        if let Some(v) = self.index_entries.get(&key) {
            let clean = match self.index_stamps.get(&key) {
                Some(&s) => ctx.dirty_since(addr, len, s) == Some(false),
                None => false,
            };
            if clean {
                return v.clone();
            }
        }
        let mut raw = crate::perf::time(crate::perf::Phase::DrawIndices, || {
            ctx.read_bytes(addr, len)
        });
        crate::perf::note_bytes(crate::perf::Phase::DrawIndices, raw.len());
        let index_of = |c: &[u8]| match elem {
            2 => u16::from_le_bytes([c[0], c[1]]) as u32,
            _ => u32::from_le_bytes([c[0], c[1], c[2], c[3]]),
        };
        let (first_vertex, vertex_count) =
            crate::perf::time(crate::perf::Phase::DrawIndexScan, || {
                let (min_index, max_index) =
                    raw.chunks(elem).fold((u32::MAX, 0u32), |(lo, hi), c| {
                        let i = index_of(c);
                        (lo.min(i), hi.max(i))
                    });
                let (first_vertex, vertex_count) = if min_index > max_index {
                    (0, 0) // no indices at all
                } else {
                    (min_index, max_index - min_index + 1)
                };
                if first_vertex > 0 {
                    for c in raw.chunks_mut(elem) {
                        let rebased = index_of(c) - first_vertex;
                        match elem {
                            2 => c[..2].copy_from_slice(&(rebased as u16).to_le_bytes()),
                            _ => c[..4].copy_from_slice(&rebased.to_le_bytes()),
                        }
                    }
                }
                (first_vertex, vertex_count)
            });
        let value = (Arc::<[u8]>::from(raw), first_vertex, vertex_count);
        // The same budget the vertex snapshots use, and cleared the same way: the keys are
        // (address, length, element size) rather than content, so a clear costs re-reads and can
        // never cost correctness.
        if self.index_bytes + value.0.len() > vertex_snapshot_budget() {
            self.index_entries.clear();
            self.index_stamps.clear();
            self.index_bytes = 0;
        }
        if let Some((old, ..)) = self.index_entries.insert(key, value.clone()) {
            self.index_bytes -= old.len();
        }
        self.index_bytes += value.0.len();
        if self.scene_stamp != 0 {
            self.index_stamps.insert(key, self.scene_stamp);
        }
        value
    }

    /// Record that a vertex snapshot's bytes are current as of this scene. See [`Self::restamp`].
    fn stamp_vertices(&mut self, addr: u32, len: usize) {
        if self.scene_stamp != 0 {
            self.vertex_stamps.insert((addr, len), self.scene_stamp);
        }
    }

    /// Advance the guest-store epoch for a new scene, at the FIRST snapshot the scene takes.
    ///
    /// Done lazily here rather than in [`Self::begin_scene`] because that has no [`GuestCtx`] to
    /// reach the map through, and because a scene that snapshots nothing should not spend an
    /// epoch value at all - a title with many texture-free passes would otherwise burn the byte
    /// on scenes that never ask a question of it.
    fn arm_scene_stamp(&mut self, ctx: &GuestCtx) {
        if self.scene_stamp != 0 {
            return;
        }
        // Reclaim the epoch range before spending the last of it, rather than letting the
        // wrap disown every snapshot this cache holds. See `Self::rebase_epoch_if_exhausted`.
        self.rebase_epoch_if_exhausted(ctx);
        let Some((stamp, wrapped)) = ctx.bump_dirty_epoch() else { return };
        // A wrapped epoch zeroed the map, so every stamp taken before it describes a page whose
        // record is gone. They are dropped, and each entry pays one compare to earn a new one.
        // BOTH sets of stamps, because both rest on the same map.
        if wrapped {
            crate::perf::note_epoch_wrap();
            self.stamps.clear();
            self.vertex_stamps.clear();
            self.index_stamps.clear();
        }
        self.scene_stamp = stamp;
    }

    /// >>> RENUMBER THE EPOCH RATHER THAN WRAP IT, when there is room to.
    ///
    /// The epoch is one byte and this cache spends one per SCENE, so a race frame burns
    /// eleven and the 253 usable values are gone in about ten presented frames. A wrap zeroes
    /// the map and drops every stamp, and the next use of each retained snapshot then copies
    /// the whole texture across the guest boundary to compare it - MEASURED at **5.40 MB per
    /// presented frame** on a racing title's on-track frame in the browser, every byte of it
    /// identical.
    ///
    /// The live stamps are not spread across the range: every snapshot proved current is
    /// re-stamped, so they cluster just under the epoch. Everything below the lowest live one
    /// is free, and [`GuestMemory::rebase_dirty_epoch`] hands it back - the map, the epoch and
    /// the stamps here all shift down by the same amount, which leaves the `page >= stamp`
    /// predicate exactly as it was.
    ///
    /// Not attempted when there is little to reclaim: renumbering costs a pass over the map,
    /// and doing it for a handful of values would just repeat next scene. Below the threshold
    /// the wrap happens exactly as it always did.
    fn rebase_epoch_if_exhausted(&mut self, ctx: &GuestCtx) {
        /// How close to the ceiling the epoch has to be before renumbering is worth a pass
        /// over the map. `bump_dirty_epoch` wraps at 255.
        const NEAR_CEILING: u8 = 250;
        /// How much of the range live stamps keep. An entry whose stamp is older than this
        /// is DROPPED rather than renumbered: without a stamp it falls through to one compare
        /// the next time it is used, which is the cost of one texture - against the whole
        /// working set, which is what a wrap costs. Everything below the floor is what the
        /// renumbering hands back.
        const RETAIN: u8 = 128;
        let Some(epoch) = ctx.dirty_epoch() else { return };
        if epoch < NEAR_CEILING {
            return;
        }
        // The floor is CHOSEN, not measured. Taking the lowest live stamp instead sounds
        // safer and is useless: one entry established just after the last wrap and proven
        // clean ever since pins it at 1, and then there is nothing to reclaim - which is
        // exactly what happened, measured, before this was a window rather than a minimum.
        let floor = epoch.saturating_sub(RETAIN);
        if floor < 2 {
            return;
        }
        // An entry older than the window loses its stamp. That is not a correctness question:
        // no stamp means "no answer", and `get_or_read` already falls through to the compare
        // for one ([`GuestMemory::dirty_since`] is emphatic that `None` is not "clean").
        self.stamps.retain(|_, s| *s >= floor);
        self.vertex_stamps.retain(|_, s| *s >= floor);
        self.index_stamps.retain(|_, s| *s >= floor);
        let Some(_new_epoch) = ctx.rebase_dirty_epoch(floor) else { return };
        // Every stamp owes the same subtraction the map just took. `floor` is the minimum, so
        // none of these can underflow.
        let shift = floor - 1;
        for s in self.stamps.values_mut() {
            *s -= shift;
        }
        for s in self.vertex_stamps.values_mut() {
            *s -= shift;
        }
        for s in self.index_stamps.values_mut() {
            *s -= shift;
        }
        crate::perf::note_epoch_rebase();
    }

    fn insert(&mut self, addr: u32, len: usize, bytes: Arc<[u8]>) {
        if self.bytes + bytes.len() > texture_snapshot_budget() {
            // A WARNING, not an info: this drops every buffer the renderer's decode cache
            // is keyed on, so it costs a full re-decode of the working set inside the next
            // frame - measured at 2.1 seconds of `build` in the browser. It was `info`,
            // which the default `warn` filter discards, so the most expensive event in the
            // frame was also the only one nobody could see. [[vitaslop-fallback-must-report]]
            tracing::warn!(
                target: "vitaslop::gxm",
                held_mb = self.bytes >> 20,
                entries = self.entries.len(),
                "texture snapshot cache over budget - CLEARED WHOLE. Every retained buffer \
                 is dropped, so the next frame re-reads and RE-DECODES the entire working \
                 set. Raise VITASLOP_SNAPSHOT_BUDGET_MB if this repeats."
            );
            self.entries.clear();
            // The cadence sets and the stamps describe entries that no longer exist.
            self.mutable.clear();
            self.checked_this_scene.clear();
            self.checked_this_frame.clear();
            self.stamps.clear();
            // The per-scene decodes hold `Arc`s into the buffers just dropped.
            self.snapshot_sets.clear();
            self.decoded.clear();
            self.bytes = 0;
            // NOT `scene_stamp`: the epoch describes guest memory, not this cache, and the
            // scene is still the same one. Clearing it here would take a second epoch value for
            // one scene, which is exactly the spending this rewrite exists to stop.
        }
        if let Some(old) = self.entries.insert((addr, len), bytes.clone()) {
            self.bytes -= old.len();
        }
        self.bytes += bytes.len();
    }

    /// Start a new scene: every retained entry becomes due for one comparison again, and the
    /// next snapshot takes a fresh guest-store epoch.
    fn begin_scene(&mut self) {
        self.checked_this_scene.clear();
        // The finished-list and per-binding memos are NOT cleared: each entry carries the
        // scene it was last proven in, and its first use in a new scene re-proves it against
        // the guest-store epoch before it is believed (see [`SetEntry::deps`]). The proof is
        // the same `get_or_read` ladder a rebuild would run, so keeping the entry changes
        // what is REDERIVED, never what is answered. `VITASLOP_TEX_MEMO_PER_SCENE` is the
        // A/B arm: the old cadence, cleared whole.
        if tex_memo_per_scene() {
            self.snapshot_sets.clear();
            self.decoded.clear();
        }
        self.scene_seq += 1;
        // Zero means "this scene has not taken one yet" - `arm_scene_stamp` fills it in on the
        // first snapshot. It is not a valid stamp: the epoch counter restarts at 1 after a wrap
        // and never returns to 0, so a stale 0 can never be mistaken for a real one.
        self.scene_stamp = 0;
    }

    /// Start a new FRAME: even a texture never seen to change is due for one comparison.
    /// See [`Self::get_or_read`] for why the two cadences exist.
    fn begin_frame(&mut self) {
        self.checked_this_frame.clear();
    }

    /// The memoised decode for `key`, PROVEN current for this scene - or `None` when there is
    /// no entry or the entry could not be proven (it is dropped here and the caller rebuilds).
    ///
    /// The proof: the pixels came out of [`Self::get_or_read`] under `snap`, so asking it
    /// again this scene and getting the SAME buffer back (`Arc::ptr_eq`) re-establishes every
    /// byte through the same stamps-then-compare ladder a rebuild would use. A changed texture
    /// returns a fresh buffer, the pointers differ, and the entry dies.
    /// Bound on distinct per-binding decodes held across a long run, now that the map
    /// survives scenes. Cleared whole when hit - re-derivation, never correctness - like
    /// [`TEXTURE_TEMPLATE_CAP`] and the set cache's own cap.
    const DECODED_CAP: usize = 16_384;

    fn decoded_validated(
        &mut self,
        ctx: &GuestCtx,
        key: (u32, [u32; 4]),
    ) -> Option<Result<crate::capture::BoundTexture, u8>> {
        // >>> THE ALREADY-PROVEN CASE TAKES ONE PROBE AND TOUCHES NOTHING ELSE.
        //
        // This used to read the entry, clone the pixel `Arc` out of it, compare the scene, and
        // then probe the map a SECOND time to clone the result - so the overwhelmingly common
        // answer ("proven earlier in this scene") paid two hashes and a refcount pair it had no
        // use for. It is called once per dependency of every set re-proven in a scene, which is
        // ~1,150 times a frame on a race.
        let (snap, degraded) = match self.decoded.get(&key) {
            None => return None,
            Some(e) if e.valid_scene == self.scene_seq => return Some(e.res.clone()),
            Some(e) => (e.snap, e.degraded),
        };
        let pixels = self.decoded.get(&key).and_then(|e| e.res.as_ref().ok()).map(|t| t.pixels.clone());
        // A dropped or degraded decode is re-attempted each new scene, exactly as it was
        // when this memo died with its scene: a texture that becomes readable is picked up
        // at the next scene, not never.
        let Some(held) = pixels.filter(|_| !degraded) else {
            self.decoded.remove(&key);
            return None;
        };
        let (addr, len) = snap.expect("an Ok entry always records its snapshot key");
        let current = self.get_or_read(ctx, addr, len);
        if Arc::ptr_eq(&current, &held) {
            // Re-fetched rather than held across the `get_or_read`: that call can clear
            // this very map (its insert enforces the snapshot budget), and then there is
            // nothing left to stamp - which falls through to the rebuild, correctly.
            if let Some(e) = self.decoded.get_mut(&key) {
                e.valid_scene = self.scene_seq;
                return Some(e.res.clone());
            }
        }
        self.decoded.remove(&key);
        None
    }

    /// The finished list for this exact set of bindings, PROVEN current for this scene.
    ///
    /// `tag` and `bindings` are the exact key material; `key` is their 64-bit fold, and the
    /// fold alone is never trusted - a hit whose stored bindings differ is a collision and is
    /// treated as a miss ([[vitaslop-content-hash-cache-must-verify]]).
    fn set_validated(
        &mut self,
        ctx: &GuestCtx,
        key: u64,
        tag: u64,
        bindings: &[TextureBinding],
    ) -> Option<Arc<[crate::capture::BoundTexture]>> {
        let entry = self.snapshot_sets.get(&key)?;
        if entry.tag != tag
            || entry.bindings.len() != bindings.len()
            || !entry
                .bindings
                .iter()
                .zip(bindings)
                .all(|(&(u, a, w), b)| u == b.unit && a == b.addr && w == b.words)
        {
            report_set_key_collision();
            return None;
        }
        if entry.valid_scene == self.scene_seq {
            return Some(entry.list.clone());
        }
        let _reproof = crate::perf::scope(crate::perf::Phase::DrawTexBindReproof);
        // >>> THE DEPENDENCY LIST IS BORROWED OUT AND PUT BACK, NOT CLONED.
        //
        // The loop below needs `self` borrowed mutably while it walks the deps, and the old
        // way to get that was to CLONE the list - an allocation plus a refcount pair per
        // dependency, on a path that runs once per set per scene (~290 times a frame on a
        // race, ~1,150 dependencies). Taking the `Vec` leaves an empty one behind, which is
        // not a state anything can observe: the guest cannot run inside a host call, and every
        // exit below either restores it or removes the entry outright.
        let Some(mut deps) = self.snapshot_sets.get_mut(&key).and_then(|e| e.deps.take()) else {
            // Built with a dropped or degraded unit: it dies with its scene, so the drop is
            // counted - and the read retried - exactly as often as before.
            self.snapshot_sets.remove(&key);
            return None;
        };
        let (list, null_drops) = match self.snapshot_sets.get(&key) {
            Some(e) => (e.list.clone(), e.null_drops),
            None => return None,
        };
        let mut proven = true;
        for (dep_key, held) in deps.iter() {
            // The pointer compare against the SET's own buffer is what makes this exact
            // even when the per-binding entry was independently rebuilt this scene: a
            // "valid" decode holding DIFFERENT bytes than this list must kill the list.
            match self.decoded_pixels_validated(ctx, *dep_key) {
                Some(p) if Arc::ptr_eq(&p, held) => {}
                _ => {
                    proven = false;
                    break;
                }
            }
        }
        if !proven {
            self.snapshot_sets.remove(&key);
            return None;
        }
        // The null-handle units' per-build drop notes, re-issued so keeping the set does
        // not quiet the counter that reports them.
        for _ in 0..null_drops {
            note_texture_drop(0);
        }
        // `get_or_read` inside the loop can clear this whole map (its insert enforces the
        // snapshot budget), so the entry may be gone - in which case there is nothing to
        // restore and nothing to stamp, and the list this proved is still the right answer
        // for THIS draw.
        if let Some(e) = self.snapshot_sets.get_mut(&key) {
            e.deps = Some(std::mem::take(&mut deps));
            e.valid_scene = self.scene_seq;
        }
        Some(list)
    }

    /// Just the PIXELS of a memoised decode, proven current for this scene.
    ///
    /// The set re-proof compares pixel buffers by pointer and wants nothing else, so it does
    /// not need [`Self::decoded_validated`]'s full [`crate::capture::BoundTexture`] clone -
    /// which copies a dozen fields and a second `Arc` per dependency, ~1,150 times a frame.
    fn decoded_pixels_validated(&mut self, ctx: &GuestCtx, key: (u32, [u32; 4])) -> Option<Arc<[u8]>> {
        if let Some(e) = self.decoded.get(&key) {
            if e.valid_scene == self.scene_seq {
                return e.res.as_ref().ok().map(|t| t.pixels.clone());
            }
        }
        match self.decoded_validated(ctx, key) {
            Some(Ok(t)) => Some(t.pixels),
            _ => None,
        }
    }

    /// The PREVIOUS draw's finished fragment list, if this draw's sampler block is byte
    /// identical to the one it was decoded from and the two draws are in the same scene and
    /// bound the same fragment program. See [`Self::last_set`] for why that is exact.
    fn set_from_previous_draw(
        &self,
        span: &[u8],
        fheader: u32,
    ) -> Option<Arc<[crate::capture::BoundTexture]>> {
        let last = self.last_set.as_ref()?;
        if last.fheader != fheader || &*last.span != span {
            return None;
        }
        // >>> THE ENTRY IS RE-PROBED, AND THAT IS WHAT MAKES THIS SAFE TO ADD.
        //
        // The list is only believed while the set cache still holds it, still under the same
        // key, still proven for THIS scene, and still the same buffer. Every invalidation
        // this cache has - a new scene, a freed range, a re-initialised texture format, a
        // budget clear, the `VITASLOP_TEX_MEMO_PER_SCENE` arm - drops or ages that entry, so
        // each of them refuses this shortcut too, with nothing to remember to do here. A memo
        // that carried its own scene number instead would have to be cleared at seven separate
        // sites and silently serve a stale list the first time a new one was added.
        let entry = self.snapshot_sets.get(&last.key)?;
        (entry.valid_scene == self.scene_seq && Arc::ptr_eq(&entry.list, &last.list))
            .then(|| last.list.clone())
    }

    /// Remember this draw's finished fragment list against the sampler bytes it came from.
    fn remember_last_set(
        &mut self,
        span: &[u8],
        fheader: u32,
        key: u64,
        list: Arc<[crate::capture::BoundTexture]>,
    ) {
        match &mut self.last_set {
            // Reuse the box rather than allocate one per draw: the span is a fixed size, so
            // this is a copy into memory that is already the right shape.
            Some(last) if last.span.len() == span.len() => {
                last.span.copy_from_slice(span);
                last.fheader = fheader;
                last.key = key;
                last.list = list;
            }
            slot => {
                *slot = Some(LastSet {
                    span: span.to_vec().into_boxed_slice(),
                    fheader,
                    key,
                    list,
                })
            }
        }
    }

    /// The PREVIOUS draw's finished VERTEX-stage list, if nothing has rebound a vertex
    /// sampler since and the set-cache entry it came from is still current for this scene.
    fn vertex_set_from_previous_draw(
        &self,
        generation: u64,
    ) -> Option<Arc<[crate::capture::BoundTexture]>> {
        let last = self.last_vertex_set.as_ref()?;
        if last.generation != generation {
            return None;
        }
        let entry = self.snapshot_sets.get(&last.key)?;
        (entry.valid_scene == self.scene_seq && Arc::ptr_eq(&entry.list, &last.list))
            .then(|| last.list.clone())
    }

    /// Remember this draw's finished VERTEX-stage list against the generation it came from.
    fn remember_last_vertex_set(
        &mut self,
        generation: u64,
        key: u64,
        list: Arc<[crate::capture::BoundTexture]>,
    ) {
        self.last_vertex_set = Some(LastVertexSet { generation, key, list });
    }

    /// Remember a finished list under its fold, with the exact key material and the
    /// per-binding facts a later scene's re-proof will check. See [`SetEntry`].
    fn set_insert(
        &mut self,
        key: u64,
        tag: u64,
        bindings: &[TextureBinding],
        list: Arc<[crate::capture::BoundTexture]>,
    ) {
        let mut null_drops = 0u32;
        let mut deps = Some(Vec::with_capacity(bindings.len()));
        for b in bindings {
            // Mirrors `decode_texture`'s early arms exactly. `addr == 0` returns `None`
            // there - the unit is silently absent from the list, noting nothing - so it
            // contributes no dep and no re-issued note here. A null HANDLE (readable
            // address, all-zero control words) binds the constant zero texel and notes
            // drop 0 once per set build; constants need no re-proof, only the re-note.
            if b.addr == 0 {
                continue;
            }
            if b.is_null() {
                null_drops += 1;
                continue;
            }
            let dep_key = (b.addr, b.words);
            // The buffer recorded here is the one `list` holds for this unit: the list's
            // items are clones of the entry being probed, taken within this same host call,
            // so nothing can have rebuilt it in between. A unit whose decode dropped or
            // degraded (or whose entry the snapshot budget just evicted) makes the whole
            // set per-scene.
            let pixels = self.decoded.get(&dep_key).and_then(|e| match (&e.res, e.degraded) {
                (Ok(t), false) => Some(t.pixels.clone()),
                _ => None,
            });
            match (deps.as_mut(), pixels) {
                (Some(d), Some(p)) => d.push((dep_key, p)),
                _ => deps = None,
            }
        }
        // A bound on distinct sets held across a long run, in the spirit of
        // [`TEXTURE_TEMPLATE_CAP`]: cleared whole when hit, which costs re-derivation and
        // never correctness - the entries re-prove themselves from live guest state.
        const SET_CAP: usize = 16_384;
        if self.snapshot_sets.len() >= SET_CAP {
            self.snapshot_sets.clear();
        }
        self.snapshot_sets.insert(
            key,
            SetEntry {
                list,
                tag,
                bindings: bindings.iter().map(|b| (b.unit, b.addr, b.words)).collect(),
                deps,
                null_drops,
                valid_scene: self.scene_seq,
            },
        );
    }

    /// The mean colour of `t`'s pixels, computed once per distinct byte buffer.
    /// `None` for a texture whose format cannot be sampled, exactly as
    /// [`crate::render::texture_mean_rgb`] reports it.
    fn mean_rgb(&mut self, t: &crate::capture::BoundTexture) -> Option<[f32; 3]> {
        let key = Arc::as_ptr(&t.pixels) as *const u8 as usize;
        if let Some((m, _)) = self.means.get(&key) {
            return Some(*m);
        }
        let m = crate::render::texture_mean_rgb(t)?;
        // Bounded by the snapshot cache it shadows: a buffer that is dropped there
        // would keep its entry alive here, so drop entries whose buffer this cache is
        // now the only owner of.
        self.means.retain(|_, (_, buf)| Arc::strong_count(buf) > 1);
        self.means.insert(key, (m, t.pixels.clone()));
        Some(m)
    }

    /// Drop every snapshot overlapping `[addr, addr+len)`, for memory the guest has
    /// released. Not needed for correctness (the comparison covers content changes),
    /// but a snapshot of freed memory is dead weight the cache should not hold.
    fn invalidate_range(&mut self, addr: u32, len: usize) {
        // The per-scene lists hold `Arc`s to snapshots this is about to drop. Keeping them would
        // hand a later draw of the same scene a texture whose memory the guest has released,
        // which is the one thing this function exists to stop.
        self.snapshot_sets.clear();
        self.decoded.clear();
        let end = addr as u64 + len as u64;
        let mut freed = 0usize;
        self.entries.retain(|&(a, l), bytes| {
            let overlaps = (a as u64) < end && (addr as u64) < (a as u64 + l as u64);
            if overlaps {
                freed += bytes.len();
            }
            !overlaps
        });
        // Keep the cadence bookkeeping in step with the entries it describes. A stale key
        // here is harmless (the `entries` lookup misses first and the texture is re-read),
        // but leaving it means the sets grow with every address the run ever touched.
        let overlapping = |&(a, l): &(u32, usize)| {
            (a as u64) < end && (addr as u64) < (a as u64 + l as u64)
        };
        self.mutable.retain(|k| !overlapping(k));
        self.checked_this_scene.retain(|k| !overlapping(k));
        self.checked_this_frame.retain(|k| !overlapping(k));
        self.bytes -= freed;

        // The same for VERTEX snapshots. Correctness does not rest on this either - freed memory
        // that comes back as a new mesh is WRITTEN by the guest, which stamps its pages, so the
        // dirty map reports it changed - but a vertex snapshot has no byte comparison behind it,
        // only the stamps, so leaving a freed range cached is a claim resting on one mechanism
        // where the texture path has two. Dropping it costs a re-read and removes the question.
        let mut vfreed = 0usize;
        self.vertex_entries.retain(|k, bytes| {
            if overlapping(k) {
                vfreed += bytes.len();
                return false;
            }
            true
        });
        self.vertex_stamps.retain(|k, _| !overlapping(k));
        self.vertex_bytes -= vfreed;
    }
}

/// The guest-store stamp mechanism, exercised against a memory that implements the dirty map
/// the way both real engines do.
///
/// # Why this is worth its length
/// The stamps are the difference between proving a texture unchanged for the cost of one byte
/// per 4 KB page and proving it by comparing every byte of it twice. MEASURED on a live race:
/// the compare is **44% of the whole frame and 105.8 MB moved per frame**. And the mechanism
/// fails SILENTLY - when the stamps stop working the picture is still perfectly correct, just
/// half the speed, so nothing in a capture points at it. It was in fact broken for its entire
/// life on the only engine that runs it, and the symptom was "the phone is slow".
///
/// The native engine does not stamp by default (wasmtime bills every operator, so the marks
/// would speed the guest clock up), which is exactly why these are unit tests over a fake
/// backing rather than something the desktop's own runs would have caught.
#[cfg(test)]
mod texture_snapshot_stamp_tests {
    use super::*;
    use std::cell::Cell;
    use std::rc::Rc;

    const PAGE: usize = 1 << 12;

    /// Shared counters, readable while `GuestCtx` holds the memory mutably borrowed.
    #[derive(Clone, Default)]
    struct Counters {
        /// Times the epoch was advanced. The number this whole exercise is about.
        bumps: Rc<Cell<u32>>,
        /// Times the snapshot cache borrowed bytes to COMPARE them.
        borrows: Rc<Cell<u32>>,
        /// Times the epoch ran out and the map had to be wiped.
        wraps: Rc<Cell<u32>>,
    }

    /// Guest memory with a working dirty map: one byte per page holding the epoch of the last
    /// store into it, and a one-byte epoch - the same shape, and the same `>=` comparison, as
    /// `vitaslop-native`'s `SharedView` and the browser's.
    struct StampedMemory {
        bytes: Vec<u8>,
        map: Vec<Cell<u8>>,
        epoch: Cell<u8>,
        c: Counters,
    }

    impl StampedMemory {
        fn new(len: usize, c: Counters) -> Self {
            StampedMemory {
                bytes: vec![0u8; len],
                map: (0..len.div_ceil(PAGE) + 1).map(|_| Cell::new(0)).collect(),
                epoch: Cell::new(1),
                c,
            }
        }

        /// A GUEST store: change the bytes and stamp the page, exactly as emitted code does.
        fn guest_store(&mut self, off: usize, value: u8) {
            self.bytes[off] = value;
            self.map[off >> 12].set(self.epoch.get());
        }
    }

    impl GuestMemory for StampedMemory {
        fn len(&self) -> usize {
            self.bytes.len()
        }
        fn read(&self, off: usize, buf: &mut [u8]) {
            buf.copy_from_slice(&self.bytes[off..off + buf.len()]);
        }
        fn write(&mut self, off: usize, bytes: &[u8]) {
            self.bytes[off..off + bytes.len()].copy_from_slice(bytes);
        }
        fn borrow(&self, off: usize, len: usize) -> Option<&[u8]> {
            self.c.borrows.set(self.c.borrows.get() + 1);
            self.bytes.get(off..off.checked_add(len)?)
        }
        fn dirty_since(&self, off: usize, len: usize, stamp: u8) -> Option<bool> {
            if len == 0 {
                return Some(false);
            }
            // One page below too - see `GuestMemory::dirty_since`'s overhang note.
            let first = (off >> 12).saturating_sub(1);
            let last = ((off + len - 1) >> 12).min(self.map.len() - 1).max(first);
            Some(self.map[first..=last].iter().any(|s| s.get() >= stamp))
        }
        fn dirty_epoch(&self) -> Option<u8> {
            Some(self.epoch.get())
        }
        fn rebase_dirty_epoch(&self, floor: u8) -> Option<u8> {
            for s in &self.map {
                let v = s.get();
                s.set(if v >= floor { v - floor + 1 } else { 0 });
            }
            let cur = self.epoch.get();
            let next = if cur >= floor { cur - floor + 1 } else { 1 };
            self.epoch.set(next);
            Some(next)
        }
        fn bump_dirty_epoch(&self) -> Option<(u8, bool)> {
            self.c.bumps.set(self.c.bumps.get() + 1);
            let next = self.epoch.get().wrapping_add(1);
            if next == 0 || next == u8::MAX {
                self.c.wraps.set(self.c.wraps.get() + 1);
                for s in &self.map {
                    s.set(0);
                }
                self.epoch.set(1);
                return Some((1, true));
            }
            self.epoch.set(next);
            Some((next, false))
        }
    }

    fn ctx_over<'a>(
        regs: &'a mut [u32; REG_COUNT],
        vfp: &'a mut [u32; VFP_ARG_COUNT],
        mem: &'a mut StampedMemory,
    ) -> GuestCtx<'a> {
        GuestCtx::new(regs, vfp, mem, 0)
    }

    /// >>> THE REGRESSION THIS EXISTS FOR: one epoch per SCENE, not one per snapshot.
    ///
    /// The epoch is a single byte compared with `>=`, so it cannot wrap silently - a wrap zeroes
    /// the whole map and invalidates every stamp taken before it. Advancing it once per
    /// re-stamped texture spent all 254 usable values in HALF A FRAME of a real race (572
    /// re-stamps a frame), so no stamp ever survived to prove anything and every texture fell
    /// through to the full byte compare. The mechanism was, in effect, off.
    #[test]
    fn the_epoch_advances_once_per_scene_not_once_per_snapshot() {
        let c = Counters::default();
        let mut mem = StampedMemory::new(400 * PAGE, c.clone());
        let (mut regs, mut vfp) = ([0u32; REG_COUNT], [0u32; VFP_ARG_COUNT]);
        let mut snaps = TextureSnapshots::new();
        let ctx = ctx_over(&mut regs, &mut vfp, &mut mem);

        // 300 distinct textures a scene, over 4 scenes. Under the old rule that is 1,200
        // epoch values against the 254 that exist, so the map would be wiped four times over.
        for _scene in 0..4 {
            snaps.begin_scene();
            for t in 0..300u32 {
                snaps.get_or_read(&ctx, t * PAGE as u32, 64);
            }
        }
        assert_eq!(c.bumps.get(), 4, "one epoch per scene, four scenes");
        assert_eq!(c.wraps.get(), 0, "254 values must comfortably outlast four scenes");
    }

    /// An unchanged texture must be proved unchanged WITHOUT its bytes being compared, and the
    /// caller must get back the very same buffer.
    ///
    /// Buffer IDENTITY is not incidental: the renderer's decode and upload caches are keyed on
    /// it, so handing back an equal-but-different `Arc` re-decodes and re-uploads a texture that
    /// did not change.
    #[test]
    fn an_untouched_texture_is_proved_clean_without_a_compare() {
        let c = Counters::default();
        let mut mem = StampedMemory::new(64 * PAGE, c.clone());
        let (mut regs, mut vfp) = ([0u32; REG_COUNT], [0u32; VFP_ARG_COUNT]);
        let mut snaps = TextureSnapshots::new();
        let ctx = ctx_over(&mut regs, &mut vfp, &mut mem);

        snaps.begin_scene();
        let first = snaps.get_or_read(&ctx, 8 * PAGE as u32, 2048);
        let after_read = c.borrows.get();

        // Three more scenes with no guest store at all.
        for _ in 0..3 {
            snaps.begin_scene();
            let again = snaps.get_or_read(&ctx, 8 * PAGE as u32, 2048);
            assert!(Arc::ptr_eq(&first, &again), "an unchanged texture must be the SAME buffer");
        }
        assert_eq!(
            c.borrows.get(),
            after_read,
            "the stamps must prove it clean - borrowing the bytes again IS the compare"
        );
    }

    /// And the shortcut must never hide a real change: a guest store into the texture's page
    /// has to produce fresh bytes on the next scene.
    ///
    /// This is the half that makes the test above meaningful. An implementation that always
    /// answered "clean" would pass the first test perfectly and serve stale pixels forever.
    #[test]
    fn a_guest_store_is_never_missed() {
        let c = Counters::default();
        let mut mem = StampedMemory::new(64 * PAGE, c.clone());
        let (mut regs, mut vfp) = ([0u32; REG_COUNT], [0u32; VFP_ARG_COUNT]);
        let mut snaps = TextureSnapshots::new();

        let addr = 8 * PAGE as u32;
        let first = {
            let ctx = ctx_over(&mut regs, &mut vfp, &mut mem);
            snaps.begin_scene();
            snaps.get_or_read(&ctx, addr, 2048)
        };
        assert_eq!(first[7], 0);

        // The guest writes one byte of it, stamping its page with the current epoch.
        mem.guest_store(8 * PAGE + 7, 0xAB);

        let ctx = ctx_over(&mut regs, &mut vfp, &mut mem);
        snaps.begin_scene();
        let again = snaps.get_or_read(&ctx, addr, 2048);
        assert!(!Arc::ptr_eq(&first, &again), "a written texture must be re-read");
        assert_eq!(again[7], 0xAB, "and it must carry the new byte");

        // Having been re-read, it must go back to being provably clean rather than paying the
        // compare for ever after.
        let settled = c.borrows.get();
        for _ in 0..3 {
            snaps.begin_scene();
            let stable = snaps.get_or_read(&ctx, addr, 2048);
            assert!(Arc::ptr_eq(&again, &stable));
        }
        assert_eq!(c.borrows.get(), settled, "it must re-stamp clean after the re-read");
    }

    /// Static geometry must be read ONCE and then proved unchanged, handing back the same buffer.
    #[test]
    fn static_vertices_are_read_once_and_then_proved_clean() {
        let c = Counters::default();
        let mut mem = StampedMemory::new(64 * PAGE, c.clone());
        let (mut regs, mut vfp) = ([0u32; REG_COUNT], [0u32; VFP_ARG_COUNT]);
        let mut snaps = TextureSnapshots::new();
        let ctx = ctx_over(&mut regs, &mut vfp, &mut mem);
        let addr = 12 * PAGE as u32;

        snaps.begin_scene();
        let first = snaps.get_or_read_vertices(&ctx, addr, 4096);
        assert_eq!(first.len(), 4096);

        // Fifty draws across five scenes from the same untouched mesh.
        for _ in 0..5 {
            snaps.begin_scene();
            for _ in 0..10 {
                let again = snaps.get_or_read_vertices(&ctx, addr, 4096);
                assert!(Arc::ptr_eq(&first, &again), "static geometry must be the SAME buffer");
            }
        }
    }

    /// >>> THE PROPERTY THAT DIFFERS FROM TEXTURES: geometry rewritten WITHIN a scene must be
    /// seen, on the very next draw.
    ///
    /// A texture bound by many draws is compared once a scene, on the argument that a title does
    /// not rewrite a texture while drawing with it. Writing vertices and drawing them repeatedly
    /// inside one scene is exactly how dynamic geometry works, so the vertex path deliberately
    /// takes no once-a-scene shortcut. If it ever did, UI text and particles would render one
    /// draw stale and the symptom would be geometry a frame behind itself.
    #[test]
    fn vertices_rewritten_within_one_scene_are_seen_immediately() {
        let c = Counters::default();
        let mut mem = StampedMemory::new(64 * PAGE, c.clone());
        let (mut regs, mut vfp) = ([0u32; REG_COUNT], [0u32; VFP_ARG_COUNT]);
        let mut snaps = TextureSnapshots::new();
        let addr = 12 * PAGE as u32;

        {
            let ctx = ctx_over(&mut regs, &mut vfp, &mut mem);
            snaps.begin_scene();
            let v = snaps.get_or_read_vertices(&ctx, addr, 1024);
            assert_eq!(v[5], 0);
        }
        // NO new scene - the guest writes and draws again, as dynamic geometry does.
        for round in 1..=4u8 {
            mem.guest_store(12 * PAGE + 5, round);
            let ctx = ctx_over(&mut regs, &mut vfp, &mut mem);
            let v = snaps.get_or_read_vertices(&ctx, addr, 1024);
            assert_eq!(v[5], round, "a mid-scene rewrite must be visible to the next draw");
        }
    }

    /// A rewrite that lands on IDENTICAL bytes must hand back the same buffer, not an equal one.
    ///
    /// A title that rebuilds a buffer every frame from unchanged inputs is common, and the
    /// renderer's packed-vertex cache keys on this buffer's identity - so a fresh `Arc` there
    /// would miss a cache holding the byte-for-byte same geometry.
    #[test]
    fn an_identical_rewrite_keeps_the_same_buffer() {
        let c = Counters::default();
        let mut mem = StampedMemory::new(64 * PAGE, c.clone());
        let (mut regs, mut vfp) = ([0u32; REG_COUNT], [0u32; VFP_ARG_COUNT]);
        let mut snaps = TextureSnapshots::new();
        let addr = 12 * PAGE as u32;

        let first = {
            let ctx = ctx_over(&mut regs, &mut vfp, &mut mem);
            snaps.begin_scene();
            snaps.get_or_read_vertices(&ctx, addr, 1024)
        };
        // Rewrite the SAME value: the dirty map says "touched", the bytes say "identical".
        mem.guest_store(12 * PAGE + 5, 0);
        let ctx = ctx_over(&mut regs, &mut vfp, &mut mem);
        snaps.begin_scene();
        let again = snaps.get_or_read_vertices(&ctx, addr, 1024);
        assert!(Arc::ptr_eq(&first, &again), "identical bytes must keep the same buffer");
    }

    /// Freed guest memory must not keep serving its old geometry.
    #[test]
    fn a_freed_range_drops_its_vertex_snapshot() {
        let c = Counters::default();
        let mut mem = StampedMemory::new(64 * PAGE, c.clone());
        let (mut regs, mut vfp) = ([0u32; REG_COUNT], [0u32; VFP_ARG_COUNT]);
        let mut snaps = TextureSnapshots::new();
        let addr = 12 * PAGE as u32;

        let first = {
            let ctx = ctx_over(&mut regs, &mut vfp, &mut mem);
            snaps.begin_scene();
            snaps.get_or_read_vertices(&ctx, addr, 1024)
        };
        snaps.invalidate_range(addr, 1024);
        let ctx = ctx_over(&mut regs, &mut vfp, &mut mem);
        let again = snaps.get_or_read_vertices(&ctx, addr, 1024);
        assert!(!Arc::ptr_eq(&first, &again), "a freed range must be re-read");
    }

    /// A texture whose page is written EVERY scene must be re-read every scene - the mechanism
    /// must not become sticky in the other direction.
    #[test]
    fn a_texture_written_every_scene_is_re_read_every_scene() {
        let c = Counters::default();
        let mut mem = StampedMemory::new(64 * PAGE, c.clone());
        let (mut regs, mut vfp) = ([0u32; REG_COUNT], [0u32; VFP_ARG_COUNT]);
        let mut snaps = TextureSnapshots::new();
        let addr = 8 * PAGE as u32;

        let mut last: Option<Arc<[u8]>> = None;
        for scene in 0..6u8 {
            mem.guest_store(8 * PAGE + 3, scene + 1);
            let ctx = ctx_over(&mut regs, &mut vfp, &mut mem);
            snaps.begin_scene();
            let got = snaps.get_or_read(&ctx, addr, 2048);
            assert_eq!(got[3], scene + 1, "scene {scene} must see its own write");
            if let Some(prev) = last {
                assert!(!Arc::ptr_eq(&prev, &got));
            }
            last = Some(got);
        }
    }
}

mod pdraw {
    /// "PDRW" - the initialised-block tag in word 0.
    pub const MAGIC: u32 = 0x5744_5250;
    pub const OFF_MAGIC: u32 = 0;
    pub const OFF_VERTEX_PROGRAM: u32 = 4;
    /// Stream `i`'s buffer pointer, one word each: words 2, 7, 8, 9. Stream 0 keeps word 2
    /// (where it has always been) and the rest use the words that were spare.
    ///
    /// FOUR, not [`super::MAX_VERTEX_STREAMS`]: `SCE_GXM_PRECOMPUTED_DRAW_WORD_COUNT` is 11
    /// and the other seven words are spoken for, so this encoding can carry four stream
    /// pointers and no more. That is a limit of OUR packing (real GXM packs the block
    /// differently), which is why a draw that needs a fifth stream is reported by
    /// `precomputed_draw_set_stream` rather than silently dropping the buffer.
    pub const OFF_STREAM: [u32; STREAM_SLOTS] = [8, 28, 32, 36];
    /// How many stream pointers [`OFF_STREAM`] has room for.
    pub const STREAM_SLOTS: usize = 4;
    pub const OFF_PRIMITIVE: u32 = 12;
    pub const OFF_INDEX_FORMAT: u32 = 16;
    pub const OFF_INDEX_ADDR: u32 = 20;
    pub const OFF_INDEX_COUNT: u32 = 24;
    /// Word 10 is unused; kept so the block is exactly the reported size.
    pub const WORDS: u32 = 11;
}

/// `SCE_GXM_MAX_VERTEX_STREAMS`. A vertex program may source its attributes from up to
/// this many separate guest buffers.
///
/// One definition, in [`crate::vita::gxmctx`], because the context block reserves a slot per
/// stream and a count that disagreed with the block's layout would read a neighbour's word.
/// It used to be 4 here - the number of stream pointers a `SceGxmPrecomputedDraw` can carry
/// in our packing, which is a different quantity entirely and is now `pdraw::STREAM_SLOTS`.
pub(crate) use crate::vita::gxmctx::MAX_VERTEX_STREAMS;

/// One reflected uniform parameter: where its components sit in the program's default
/// uniform buffer and how they are packed. `res` is a 4-byte-register offset; each of
/// the `comp` components is an F16 (2 bytes) when `f16`, else an F32.
#[derive(Clone, Copy, Debug, PartialEq)]
struct ParamRef {
    res: u32,
    comp: u8,
    f16: bool,
}

/// Everything the capture needs to know about ONE `SceGxmProgram`, reflected out of
/// its parameter table.
///
/// # Why this is cached
/// A program's parameter table is immutable for the life of the registration: it is
/// part of the compiled shader blob the guest hands to the shader patcher. Every field
/// here is therefore a constant of the program, while the VALUES at these offsets
/// change every draw.
///
/// Reflecting per draw meant five independent walks of the table (world/projection,
/// exposure, shader-expansion, albedo sampler, material) each allocating a `String`
/// per parameter to lowercase and substring-match its name. At a couple of hundred
/// draws a frame that was tens of thousands of allocations per frame and the single
/// most expensive thing a `sceGxmDraw` did. One walk, once per program, replaces it.
///
/// Every field is `Copy`, so a draw takes the whole reflection out of the cache
/// without touching the allocator.
#[derive(Clone, Copy, Debug, Default, PartialEq)]
struct ProgramReflection {
    /// Float offset of the model->world 4x4 (`vsModelToWorldMatrix`).
    world_off: Option<u32>,
    /// Float offset of the world->projection 4x4.
    proj_off: Option<u32>,
    /// Float offset of a SINGLE combined model->clip 4x4 (`worldViewProj`), when the
    /// shader keeps one instead of a model/projection pair. Takes precedence over
    /// [`Self::world_off`]/[`Self::proj_off`] because it is already the whole transform.
    mvp_off: Option<u32>,
    /// Float offset of the scene exposure scalar (`vsCoarseExposureReg`).
    exposure_off: Option<u32>,
    /// Does this vertex program SYNTHESIZE its primitive (point sprite / billboard)?
    shader_expanded: bool,
    /// Texture unit of the sampler that reads as the albedo / base-colour map.
    albedo_unit: Option<u32>,
    /// The base-colour tint multiplying the sampled albedo.
    tint: Option<ParamRef>,
    /// Direction and colour of the single directional light the material model uses.
    light_dir: Option<ParamRef>,
    light_col: Option<ParamRef>,
    /// Size of the program's default uniform buffer, from the table's extent in 32-bit
    /// REGISTERS (each parameter's components packed at its own type's width, not one
    /// register per component). See [`VitaState::reflected_uniform_size_bytes`].
    uniform_size_bytes: u32,
    /// The program header's OWN default-uniform-buffer size field, clamped and scaled -
    /// what [`default_uniform_buffer_bytes`] reads, memoised here.
    ///
    /// It is a word of the container, which is immutable while the program is registered
    /// (the same fact `program_blobs` and this whole table rest on), and the staleness check
    /// in `record_draw` read it FRESH on every draw of every frame. MEASURED in the browser:
    /// two single-word guest reads per draw, ~2,000 a presented frame, which is where the
    /// draw path's whole remaining word-read count came from. A word read is a bounds check
    /// and a virtual call into a typed array there [[vitaslop-browser-scalar-reads-are-a-typed-array]].
    default_uniform_bytes: u32,
    /// Texture units the program's samplers occupy (one past the highest sampler
    /// resource index). This is the length of the texture array the whole-array
    /// setter `sceGxmPrecomputed*StateSetAllTextures` is given.
    texture_unit_count: u32,
}

/// All host state for one run: the guest allocator, handle tables, the capture
/// stream, the world (determinism seam), and the in-progress scene state.
/// Which path is advancing the virtual game clock, for [`VitaState::clock_sources`].
#[derive(Clone, Copy, PartialEq, Eq)]
enum ClockSource {
    /// A scheduler quantum of guest execution ([`VitaState::charge_cpu_quantum`]).
    Quantum,
    /// The per-flip top-up ([`VitaState::advance_time_frame`]), off under
    /// `VITASLOP_FRAME_TOPUP=0`.
    FrameTopup,
    /// The scheduler's idle path jumping to the earliest pending deadline - and the
    /// default, so an advance from an untagged path is attributed rather than lost.
    Idle,
}

pub struct VitaState {
    pub base: u32,
    pub mem_bytes: u32,
    alloc_cursor: u32,
    next_handle: u32,
    next_uid: i32,
    memblocks: Vec<MemBlock>,
    /// Accrued modelled storage-transfer time not yet paid by parking a thread.
    io_debt_us: u64,
    /// The modelled storage device's own clock, in microseconds. Deliberately NOT the
    /// game clock (`virtual_us`): a park charged against `virtual_us` livelocks, because
    /// that clock only advances on a display flip or on the scheduler's
    /// nothing-is-runnable idle path, and a title waiting for a load produces neither.
    /// This clock instead advances on any evidence that time passed at all - see
    /// [`VitaState::advance_io_by`] - so a modelled transfer always completes.
    io_us: u64,
    /// `(thid, io_us deadline)` for each thread parked paying its storage-latency debt.
    io_waiters: Vec<(i32, u64)>,
    /// Storage-clock time already charged by quantum boundaries since the last display
    /// flip, so the per-flip advance can top up to exactly one frame rather than
    /// double-count. See [`VitaState::advance_io_frame`].
    io_charged_since_flip: u64,
    /// Game-clock time already charged by quantum boundaries since the last display
    /// flip, so the per-flip advance can top the clock up to exactly one frame rather
    /// than double-count. Exactly the role `io_charged_since_flip` plays for the
    /// storage clock. See [`VitaState::charge_cpu_quantum`].
    cpu_charged_since_flip: u64,
    /// Sub-microsecond remainders of the two per-fuel charges, `(game clock, storage
    /// clock)`, carried from one charge to the next.
    ///
    /// # Without these the clock STOPS on a title that yields in small bursts
    /// A charge is `per_quantum * fuel / QUANTUM_FUEL`, and `QUANTUM_FUEL` is five
    /// million. A thread that suspends after a few hundred fuel units - a spin whose loop
    /// body makes one cheap host call - therefore rounds to **zero microseconds**, every
    /// time, forever. That is not a rounding error, it is a stopped clock: measured on a
    /// retail title in the browser, one thread was resumed 198,968 times at frame 2 while
    /// the game clock stood still at 3.44 s, because the storage transfer it was spinning on
    /// could only complete when time passed and no time ever did.
    ///
    /// Carrying the remainder makes the charge exact in the long run - a hundred yields of
    /// a tenth of a microsecond now cost ten microseconds instead of nothing - which is
    /// what lets such a spin terminate. It is the same argument the CPU charge itself
    /// rests on: work the guest really did has to cost time, or something waiting on time
    /// waits for ever.
    charge_rem: (u64, u64),
    /// Scheduler quanta and display flips observed, for the calibration diagnostic that
    /// sets `QUANTUM_CPU_US` (a quantum is worth one frame divided by the quanta a
    /// rendering frame actually takes - see [`VitaState::charge_cpu_quantum`]).
    quantum_count: u64,
    flip_count: u64,
    /// Earliest game-clock time the display may latch the NEXT frame - the vsync floor.
    /// See [`VitaState::pace_flip`].
    next_flip_us: u64,
    /// The `sync` argument of the most recent `sceDisplaySetFrameBuf`, which is what
    /// decides whether a present waits for the scanout. See
    /// [`VitaState::set_display_sync`].
    display_sync: u32,
    /// Whether the guest has called `sceDisplaySetFrameBuf` at all, so the diagnostic can
    /// tell "asked for the default" from "never asked".
    display_sync_seen: bool,
    /// `(base, size)` of every released memory block, in release order, available for
    /// reuse by [`VitaState::alloc_memblock`]. Not coalesced: adjacency in the arena is
    /// not adjacency in usefulness here (blocks are whole buffers a title allocates and
    /// releases as units), and leaving the list in release order keeps the choice of hole
    /// a pure function of the guest's own allocation sequence.
    freed_memblocks: Vec<(u32, u32)>,
    /// Vertex program handle -> its attribute layout. A map, not a list: every draw
    /// resolves the bound handle several times, and a title registers hundreds of
    /// programs, so a linear scan here was per-draw cost that grew with the level.
    /// >>> AND AN INTEGER HASHER, LIKE EVERY OTHER PER-DRAW MAP HERE. The key is a guest
    /// address; SipHash on it is tens of nanoseconds per probe, and a draw probes this and the
    /// two below several times each - about 4,000 probes a presented frame. See
    /// [`crate::fasthash`], which the texture maps have used since the same measurement.
    vertex_programs: FxHashMap<u32, VertexProgramInfo>,
    /// Reflected constants of each `SceGxmProgram`, keyed by its header address. See
    /// [`ProgramReflection`]; cleared whenever a program is registered or unregistered,
    /// which is the only moment a header address can come to mean a different program.
    program_reflection: FxHashMap<u32, ProgramReflection>,
    /// Raw `SceGxmProgram` container bytes by header address - see [`VitaState::program_blob`].
    /// Cleared alongside `program_reflection`, and for the same reason.
    program_blobs: FxHashMap<u32, std::sync::Arc<[u8]>>,
    /// Whether (and how) each VERTEX program's 0xE8 memory loads need a guest-memory window
    /// snapshotted at draw time - `vitaslop_gxp_shader::mem_window_for_vertex_blob`, memoised
    /// by header address. `None` for the overwhelming majority of programs, so the per-draw
    /// cost of this feature on every other title is one map lookup. Cleared alongside
    /// `program_reflection`, and for the same reason.
    mem_window_specs: FxHashMap<u32, Option<vitaslop_gxp_shader::MemWindow>>,
    /// Shader PAIRS the guest's patcher has named but the renderer has not been told about yet
    /// - see [`VitaState::queue_shader_precompile`]. Drained onto the next scene handed to the
    /// renderer, which compiles them before it encodes anything.
    /// Held behind one `Arc` so handing it to a scene costs a single refcount - see
    /// [`crate::capture::Scene::precompile`]. Appending clones the list once (`Arc::make_mut`),
    /// which happens while a loading screen names pairs and never in a gameplay frame.
    pending_precompile: std::sync::Arc<Vec<(std::sync::Arc<[u8]>, std::sync::Arc<[u8]>)>>,
    /// Pairs already queued, so a title that creates the same fragment program repeatedly - or
    /// re-creates one after a patcher reset - does not re-queue work the renderer has cached.
    precompiled_pairs: std::collections::HashSet<(u32, u32)>,
    /// Every `SceGxmProgram *` the patcher has created a VERTEX program from, and every one it
    /// has created a FRAGMENT program from with a NULL `vertexProgram`. Kept in creation order
    /// and only used by [`VitaState::cross_precompile`] - a title that names its pairs properly
    /// fills the first list and never reads either.
    created_vertex_headers: Vec<u32>,
    null_fragment_headers: Vec<u32>,
    /// The DISTINCT programs the title's PRECOMPUTED STATES name, in the order it named them -
    /// see [`VitaState::cross_precomputed_state_programs`]. A different and far smaller list
    /// than the two above: a precomputed state is the title declaring "this program is one I
    /// will draw with", so the cross product over these is a few hundred candidates rather
    /// than a third of a million.
    precomputed_state_vertex_headers: Vec<u32>,
    precomputed_state_fragment_headers: Vec<u32>,
    color_surfaces: Vec<(u32, crate::capture::ColorSurface)>,
    /// Guest address of the displayQueue callback from sceGxmInitialize, and its
    /// data size. Recorded for faithfulness; the present address is captured from
    /// the display queue directly so the callback need not run yet.
    pub display_queue_cb: u32,
    pub display_queue_cb_data_size: u32,
    // In-progress scene (BeginScene..EndScene).
    scene: Option<crate::capture::Scene>,
    /// The guest `SceGxmContext *` the sticky draw state lives in - the `hostMem` the guest
    /// handed `sceGxmCreateContext`, as on hardware. See [`crate::vita::gxmctx`].
    ///
    /// This is the only GXM context state left on the host, and it is an ADDRESS, not a
    /// value: the bound vertex/fragment program, the vertex streams and the whole
    /// [`crate::capture::RenderState`] are read back out of guest memory on demand
    /// ([`Self::bound_vertex_program`] and friends). That is what lets their setters be
    /// inlined into guest code instead of crossing the host boundary once per call.
    ///
    /// Zero until a context exists. A draw before then is reported, not guessed at.
    gxm_context: u32,
    /// Whether a draw with no context has already been reported, so the report is once.
    reported_no_gxm_context: bool,
    /// Guest address of the fallback SA bank - the uniforms a draw reads when no default
    /// uniform buffer is bound for its stage. Placed once in [`Self::set_alloc_base`],
    /// before any guest code runs, and published to the host-mirror block so an inlined
    /// `sceGxmSetUniformDataF` can reach it; see [`SA_BANK_DATA`] for the layout and for
    /// why it is in guest memory at all. Zero if the arena could not place one.
    sa_bank: u32,
    /// Scratch for [`crate::vita::gxmctx::texture_bindings`], kept so the hottest path in the
    /// engine does not allocate a `Vec` per draw. Never read between draws.
    bound_binding_scratch: Vec<(u32, crate::vita::gxmctx::TexBinding)>,
    // Threads the program created, and any pending synchronous thread run raised
    // by sceKernelStartThread (drained by the engine host after the call).
    threads: Vec<ThreadRec>,
    /// Stacks belonging to deleted threads, as `(base, size)`. NOT a free list - see
    /// [`VitaState::create_thread`] for why they are not recycled. Kept so the leak a
    /// thread-churning title causes is a number someone can look at rather than an
    /// invisible drift toward an allocator collision.
    free_stacks: Vec<(u32, u32)>,
    /// Whether the runaway-thread-creation warning has been emitted (once per run).
    runaway_threads_reported: bool,
    /// Whether "video playback is not implemented" has been reported (once per run).
    /// See `vita::video`.
    pub(crate) reported_no_video: bool,
    pending_reentry: Option<Reentry>,
    // Synchronization objects. Bring-up model: one thread of control (workers run
    // synchronously to completion), so nothing ever actually blocks; a semaphore's
    // count and an event flag's bit pattern are still tracked so their observable
    // state is faithful for single-thread use (guarding data, wait-then-read).
    semaphores: Vec<SemaRec>,
    event_flags: Vec<(i32, u32)>,
    /// The guest's own name for each event flag, for REPORTS only (see
    /// [`create_event_flag`](Self::create_event_flag)). Kept beside `event_flags` rather
    /// than in it so no waiter/set path pays for a string it never reads.
    event_flag_names: std::collections::BTreeMap<i32, String>,
    /// One bit per open SceCommonDialog family (see `vita::services::DialogFamily`):
    /// set by `*DialogInit`, read by `*DialogGetStatus` (open reports FINISHED -
    /// dialogs complete instantly offline), cleared by `*DialogTerm`.
    pub(crate) open_dialogs: u32,
    /// Registered FIOS2 path overlays, kept sorted by `order` (see
    /// [`crate::vita::fios2`]). Path resolution walks them in that order, which is
    /// what `order` is for, so sorting on insert keeps every resolve a plain scan.
    fios_overlays: Vec<FiosOverlay>,
    /// Threads that have switched overlay resolution off for themselves
    /// (`_sceFiosKernelOverlayThreadSetDisabled`). Per-thread, because that is the
    /// point of the call: a loader thread turns overlays off to reach the real file
    /// underneath one it has just overlaid.
    fios_overlay_disabled: std::collections::HashSet<i32>,
    /// The debug GPIO output register (`sceKernelSetGPO`). On a development unit its
    /// low bits drive the board's diagnostic LEDs; retail hardware has none wired, so
    /// nothing observable comes of a write. It is still a register and the value is
    /// still held, so it is held here rather than discarded - a title that toggles it
    /// as a progress marker through boot leaves its last marker readable.
    pub(crate) gpo: u32,
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
    /// Lightweight-mutex handoffs decided where no guest memory was reachable, settled by
    /// [`resolve_deferred_lwmutex`](VitaState::resolve_deferred_lwmutex) before the next
    /// resume. `(work area, thread to give it to)`.
    pending_lwmutex_acquires: Vec<(u32, i32)>,
    conds: Vec<CondRec>,
    sema_waiters: Vec<SemaWaiter>,
    evf_waiters: Vec<EvfWaiter>,
    /// Threads parked in `sceKernelWaitSignal`. A plain id list: the wait has no
    /// count or pattern to match, only "a signal arrived for me".
    signal_waiters: Vec<i32>,
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
    /// SceNpTrophy: the sets read out of the title's own TRP plus this run's unlock
    /// ledger. Public to the NID handlers in `vita/services.rs`.
    pub(crate) trophies: crate::trophy::TrophyStore,
    /// `sceClibMspace*`: general allocators over blocks of the title's own memory.
    pub(crate) mspaces: crate::mspace::MspaceStore,
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
    /// SceLibLocation handles and permission-dialog state (see `vita::location`). The
    /// position itself is never stored here - it is read from [`World`] at each call, so
    /// a fix cannot go stale behind the guest's back.
    pub(crate) location: crate::vita::location::LocationState,
    /// Bring-up aid: halt the run when the guest calls sceGxmTerminate. The cube
    /// entry is `_start`, which spins forever after `main` returns (there is no OS
    /// to exit to yet), so terminate is the clean stopping point after teardown.
    pub halt_on_terminate: bool,
    /// Guest address of the main module's `SceProcessParam`, returned verbatim by
    /// `sceKernelGetProcessParam`. libc's crt reads the `SceLibcParam` it points to
    /// for the heap configuration, so this must be a real address (0 would fault).
    process_param: u32,
    /// The modules in the linked image (from [`crate::link::LinkedProgram`]), which
    /// is what the kernel's module queries answer from. Set once before the run.
    modules: Vec<crate::link::LoadedModule>,
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
    bound_textures: Vec<TextureBinding>,
    /// Textures bound to VERTEX-stage sampler units (`_sceGxmSetVertexTexture`). Kept sorted by
    /// unit, like [`Self::bound_textures`], and separate from it because the two stages number
    /// their units independently.
    bound_vertex_textures: Vec<TextureBinding>,
    /// How many times [`Self::bind_vertex_texture`] has CHANGED the list above.
    ///
    /// The fragment stage's bindings live in guest memory, so "did they change?" is a byte
    /// compare of the sampler block. This list lives on the host and has exactly one mutator,
    /// so the same question is one counter - and it needs asking, because this title binds a
    /// vertex sampler on every one of ~626 draws a frame and the answer is almost always no.
    vertex_texture_gen: u64,
    /// The previous draw's parsed fixed-function render state, with the two context-block
    /// spans it was parsed from.
    ///
    /// The state is ~50 fields read one word at a time out of the snapshot and ~240 bytes of
    /// the `Draw` it lands in, and a run of draws inside one pass carries the identical state.
    /// Two `memcmp`s (132 + 24 bytes, the two contiguous regions the state occupies) answer
    /// "the same as last time?", and a hit is an `Arc` clone instead of a parse and a copy.
    render_state_memo: Option<(Box<[u8]>, Box<[u8]>, std::sync::Arc<crate::capture::RenderState>)>,
    /// Scratch for decoding a draw's vertex uniform bank into floats.
    ///
    /// The bank is read as BYTES (that is what the recompiled shader wants) and the reflection
    /// below wants floats, so every draw converted the whole thing into a fresh `Vec<f32>` -
    /// an allocation and a free per draw, ~626 a frame, for a buffer whose only surviving
    /// consumer on the recompiled path is the first sixteen lanes. Reused here and handed back
    /// at the end of the draw.
    uniform_float_scratch: Vec<f32>,
    /// Exact `SceGxmTextureFormat` last set on a guest `SceGxmTexture*` via
    /// `sceGxmTextureInit*`/`SetFormat`. The 16-byte control words alone lose the
    /// channel swizzle (only a 3-bit field survives), so we keep the full 32-bit
    /// format the guest passed for an exact decode; a texture the guest fills
    /// directly (a `.gxt` blob) is absent here and falls back to control-word parse.
    ///
    /// Maps, not lists: these three are looked up once per BOUND TEXTURE UNIT per
    /// DRAW, while they grow with every distinct texture the title has ever touched.
    /// A linear scan made the capture cost of a frame grow with the size of the level.
    texture_formats: crate::fasthash::FxHashMap<u32, u32>,
    /// Memoised answers from [`Self::nearest_recorded_texture`], keyed by the NULL handle
    /// address that asked. That search is a linear scan of `texture_formats`, and it feeds
    /// a report that fires once per (unit, address) - see `snapshot_bound_textures`, where
    /// running it on the ordinary by-value binding path cost 46% of a race frame.
    nearby_texture_cache: FxHashMap<u32, Option<(i64, u32)>>,
    /// Scratch for [`Self::snapshot_bound_textures`]'s per-unit pass, kept so the hottest
    /// function in the capture does not allocate a `Vec` on every draw.
    ///
    /// It exists at all because the per-unit control state must be read through a shared borrow
    /// of `self` while the snapshot cache is borrowed mutably; `mem::take` on a field is what
    /// lets that be one reused buffer rather than a fresh allocation per draw. On this title's
    /// race that is ~644 draws a frame, and in the browser an allocation is a good deal dearer
    /// than it is here.
    texture_unit_scratch: Vec<(TextureBinding, Option<u32>, Option<(i64, u32)>)>,
    /// The one piece of per-texture state that cannot be packed into the guest's own control
    /// words - see [`TextureExtra`]. Everything the sampler getters used to read from here now
    /// lives in the guest's `SceGxmTexture`, where the hardware keeps it.
    texture_extra: std::collections::HashMap<u32, TextureExtra>,
    /// Per-color-surface gamma-correction mode set by `sceGxmColorSurfaceSetGammaMode`,
    /// keyed by `SceGxmColorSurface*`. Absent = SCE_GXM_COLOR_SURFACE_GAMMA_NONE.
    color_surface_gamma: Vec<(u32, u32)>,
    /// Per-scene texture-byte snapshots, keyed by (guest data address, byte length), so a
    /// texture bound by hundreds of draws is read from guest memory once and shared. Cleared
    /// at `beginScene` - see the note in `decode_texture` for why that is the right scope.
    texture_snapshots: TextureSnapshots,
    /// How many precomputed states have been INITIALISED, for
    /// [`Self::report_precomputed_state_programs`]'s ladder. The states themselves live in
    /// GUEST memory now (`vita::gxmstate`) - the struct plus an arrays block this engine
    /// allocates from the guest heap - which is what lets the per-draw binds be inlined
    /// and lets a state the guest `memcpy`s keep working (the same by-value fix the
    /// precomputed-DRAW family got, see [`pdraw`]).
    precomputed_state_count: u32,
    // Two host-side texture diagnostics used to live here - "did the current bindings come
    // from a precomputed state" and "which handles were live when they were bound". Both are
    // now properties of the BINDING, in the context block: `from_precomputed` is per unit
    // (a global one mislabels whichever path did not happen last), and "live at bind" is
    // exactly "the copied control words are not all zero", because the copy happens AT the
    // bind. Keeping host mirrors of them would have been worse than redundant once the bind
    // went inline: the handler stops running, so they would have read as false for every
    // binding in the run while looking like measurements.
    /// `SceGxmFragmentProgram*` handle -> (its `SceGxmProgram*`, the blend equation it was
    /// created with), recorded at `sceGxmShaderPatcherCreateFragmentProgram` so a precomputed
    /// fragment state can size its default uniform buffer and every draw can carry its real
    /// blend mode. (Vertex programs carry their header in `VertexProgramInfo`.)
    fragment_programs: std::collections::HashMap<u32, (u32, crate::capture::BlendState)>,
    /// Set once, when a single scene has asked for more default-uniform bytes than the
    /// ring holds, so the wrap is reported exactly once instead of every frame.
    ///
    /// The RING ITSELF is not here: it lives in the guest's context block
    /// ([`crate::vita::gxmctx::off::UNIFORM_RING_BASE`]), which is where the hardware keeps
    /// it and what makes `sceGxmReserve*DefaultUniformBuffer` inlinable. Only this
    /// once-per-run report is host state, because "have I already said this" is not a fact
    /// about the guest.
    uniform_ring_wrapped: bool,
    /// Said once, if a reserve ever arrives with no context block to record into.
    reported_reserve_without_context: bool,
    /// How many times each `(stage, bound-for program, drawing program)` triple has been
    /// seen by [`Self::stale_uniforms`]. A COUNT, not a set: the warning fires on powers of
    /// ten, so a run says whether this happened to four draws or to four hundred thousand
    /// without printing hundreds of lines a frame. "Once per pair" hides the scale, and the
    /// scale is the whole question when the suspicion is that a stage is being starved of its
    /// uniforms across a title.
    reported_stale_uniforms: std::collections::HashMap<(&'static str, u32, u32), u64>,
    /// Colour-surface pairs already reported as overlapping in guest memory, so the report
    /// fires once per pair rather than once per `sceGxmColorSurfaceInit`.
    reported_surface_overlaps: std::collections::HashSet<(u32, u32)>,
    /// The GPU notification region: a guest buffer of `SCE_GXM_NOTIFICATION_COUNT`
    /// u32 slots handed out by `sceGxmGetNotificationRegion`, lazily allocated on
    /// first use (0 = not yet allocated). Scenes complete synchronously here, so a
    /// notification the guest waits on is treated as already signalled.
    notification_region: u32,
    /// `sceGxmSetVisibilityBuffer(context, bufferBase, stridePerCore)`: where the GPU
    /// writes occlusion-query results, and the per-core stride. 0 = no buffer bound.
    visibility_buffer: u32,
    visibility_stride: u32,
    /// Per visibility-test index, the sample count accumulated over the current scene
    /// (see [`Self::accumulate_visibility`]). Flushed into the visibility buffer when
    /// the scene ends, because that is when the GPU would have written it.
    visibility_counts: std::collections::BTreeMap<u32, u32>,
    /// GPU memory mappings from `sceGxmMapMemory(base, size, attr)`, as base -> size.
    /// Needed only so `sceGxmUnmapMemory(base)` - which is given no size - can drop the
    /// texture snapshots taken from that range: after an unmap the guest is free to
    /// reuse the pages, and a snapshot cached against those addresses would then be
    /// sampled as though it were still the texture that lived there.
    gxm_mappings: std::collections::BTreeMap<u32, u32>,
    /// `SceGxmRenderTarget` handle -> the `driverMemBlock` UID its params carried, so
    /// `sceGxmRenderTargetGetDriverMemBlock` hands back exactly what the guest gave.
    render_target_mem_blocks: std::collections::HashMap<u32, u32>,
    /// `SceGxmRenderTarget` handle -> the `(width, height)` its params declared. This
    /// is the authoritative extent of every scene begun on that target: GXM
    /// rasterizes the render target's region, and the colour surface only describes
    /// where the pixels land (its data pointer, format and stride). See
    /// [`Self::render_target_extent`].
    render_target_extents: std::collections::HashMap<u32, (u32, u32)>,
    /// How many images `scePhotoExportFromData` has written, so each export gets its
    /// own path instead of overwriting the last.
    photo_exports: u32,
    /// The offline SceNet socket table, its resolver and epoll handles, and the
    /// per-thread errno slots. See `vita::net` for the model.
    net_sockets: Vec<NetSocket>,
    net_resolvers: Vec<(i32, i32)>,
    net_epolls: Vec<(i32, Vec<i32>)>,
    net_errno: Vec<(i32, u32)>,
    /// Every `SceFiber` the guest has initialised (see [`FiberRec`]).
    fibers: Vec<FiberRec>,
    /// Per THREAD that called `sceFiberRun`, the `argOnRun` out-pointer to fill when
    /// the fiber chain it started returns to it. Keyed by thread id, not by fiber,
    /// because it belongs to the run call the thread is parked in.
    fiber_run_out: std::collections::HashMap<i32, u32>,
    /// `SceGxmTexture*` -> the palette pointer set by `sceGxmTextureSetPalette`.
    /// Kept beside the control words (rather than packed into word 0) because the
    /// palette field's bit layout in the control words is not published, and a wrong
    /// packing would corrupt fields that ARE understood.
    texture_palettes: std::collections::HashMap<u32, u32>,
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
    /// Which caller is currently advancing the clock, for the [`clock_sources`] split.
    ///
    /// `Idle` is the DEFAULT rather than an `Option`, so an advance made from a path
    /// nobody tagged is attributed to the scheduler's idle jump instead of disappearing.
    /// A bucket that silently drops time would make the split add up to less than the
    /// clock and quietly invite the wrong conclusion.
    clock_source: ClockSource,
    clock_from_quanta_us: u64,
    clock_from_topup_us: u64,
    clock_from_idle_us: u64,
    /// The current display-frame index, updated each flip by the scheduler
    /// (`on_frame_boundary`). Frame-tags egress-ledger events so a recipe can assert
    /// roughly when a milestone occurred, not only that it did.
    cur_frame: u64,
}

/// The ceiling on a default uniform buffer, in 4-byte registers. A program header or parameter
/// record that asks for more than this is one we are misreading, and the clamp keeps a bad read
/// from turning into an allocation.
pub(crate) const MAX_DEFAULT_UNIFORM_REGS: u32 = 4096;

/// Byte offset of the first float in the fallback SA bank. The word before it is the
/// high-water float count.
///
/// The bank lives in GUEST memory, and that is what makes `sceGxmSetUniformDataF` -
/// **1,106 calls a frame on one title, 58% of everything it still calls** - an inlinable
/// pair of `memory.copy`s instead of a boundary crossing. It is engine scratch rather than
/// device state (hardware has no such bank: a title writes into the buffer it names, full
/// stop), but the rule that decides where it goes is the same one `gxmctx` rests on - a
/// fact a hot call needs must live where guest code can reach it, or the call cannot be
/// inlined at all. [[vitaslop-guest-state-is-what-makes-a-call-inlinable]]
pub(crate) const SA_BANK_DATA: u32 = 4;

/// Total bytes the bank occupies: the high-water word plus a float per register the bank
/// admits ([`MAX_DEFAULT_UNIFORM_REGS`], which is also the ceiling `set_uniforms` refuses
/// past).
const SA_BANK_BYTES: u32 = SA_BANK_DATA + MAX_DEFAULT_UNIFORM_REGS * 4;

/// Which of the two shader stages a program handle belongs to. The stages differ in one
/// thing only - which host map resolves a handle to its `SceGxmProgram *` - so the code
/// that reserves a default uniform buffer is written once and told which map to ask.
#[derive(Clone, Copy, PartialEq, Eq)]
enum ProgramStage {
    Vertex,
    Fragment,
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
            io_debt_us: 0,
            io_us: 0,
            io_waiters: Vec::new(),
            io_charged_since_flip: 0,
            charge_rem: (0, 0),
            cpu_charged_since_flip: 0,
            next_flip_us: 0,
            display_sync: Self::SETBUF_NEXTFRAME,
            display_sync_seen: false,
            quantum_count: 0,
            flip_count: 0,
            freed_memblocks: Vec::new(),
            vertex_programs: FxHashMap::default(),
            program_reflection: FxHashMap::default(),
            program_blobs: FxHashMap::default(),
            mem_window_specs: FxHashMap::default(),
            pending_precompile: Default::default(),
            precompiled_pairs: std::collections::HashSet::new(),
            created_vertex_headers: Vec::new(),
            null_fragment_headers: Vec::new(),
            precomputed_state_vertex_headers: Vec::new(),
            precomputed_state_fragment_headers: Vec::new(),
            color_surfaces: Vec::new(),
            display_queue_cb: 0,
            display_queue_cb_data_size: 0,
            scene: None,
            gxm_context: 0,
            reported_no_gxm_context: false,
            sa_bank: 0,
            bound_binding_scratch: Vec::new(),
            threads: Vec::new(),
            free_stacks: Vec::new(),
            runaway_threads_reported: false,
            reported_no_video: false,
            pending_reentry: None,
            semaphores: Vec::new(),
            event_flags: Vec::new(),
            event_flag_names: std::collections::BTreeMap::new(),
            open_dialogs: 0,
            fios_overlays: Vec::new(),
            fios_overlay_disabled: std::collections::HashSet::new(),
            gpo: 0,
            preemptive: false,
            current: 0,
            mutexes: Vec::new(),
            lwmutexes: Vec::new(),
            pending_lwmutex_acquires: Vec::new(),
            conds: Vec::new(),
            sema_waiters: Vec::new(),
            signal_waiters: Vec::new(),
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
            trophies: crate::trophy::TrophyStore::default(),
            mspaces: crate::mspace::MspaceStore::default(),
            fonts: crate::font::FontLibrary::default(),
            capture: Capture::new(),
            world,
            audio: Box::new(crate::audio::NullSink::default()),
            audio_state: crate::vita::audio::AudioState::default(),
            location: crate::vita::location::LocationState::default(),
            halt_on_terminate: false,
            process_param: 0,
            modules: Vec::new(),
            tls_slots: Vec::new(),
            tls_template: (0, 0, 0),
            tls_bases: Vec::new(),
            shader_programs: Vec::new(),
            bound_textures: Vec::new(),
            bound_vertex_textures: Vec::new(),
            vertex_texture_gen: 0,
            render_state_memo: None,
            uniform_float_scratch: Vec::new(),
            texture_formats: Default::default(),
            nearby_texture_cache: FxHashMap::default(),
            texture_unit_scratch: Vec::new(),
            texture_extra: std::collections::HashMap::new(),
            color_surface_gamma: Vec::new(),
            texture_snapshots: TextureSnapshots::new(),
            precomputed_state_count: 0,
            fragment_programs: std::collections::HashMap::new(),
            uniform_ring_wrapped: false,
            reported_reserve_without_context: false,
            reported_stale_uniforms: std::collections::HashMap::new(),
            reported_surface_overlaps: std::collections::HashSet::new(),
            notification_region: 0,
            visibility_buffer: 0,
            visibility_stride: 0,
            visibility_counts: std::collections::BTreeMap::new(),
            gxm_mappings: std::collections::BTreeMap::new(),
            render_target_mem_blocks: std::collections::HashMap::new(),
            render_target_extents: std::collections::HashMap::new(),
            texture_palettes: std::collections::HashMap::new(),
            photo_exports: 0,
            net_sockets: Vec::new(),
            net_resolvers: Vec::new(),
            net_epolls: Vec::new(),
            net_errno: Vec::new(),
            fibers: Vec::new(),
            fiber_run_out: std::collections::HashMap::new(),
            lwcond_waiters: Vec::new(),
            lwcond_mutex: Vec::new(),
            sleep_waiters: Vec::new(),
            virtual_us: 0,
            clock_source: ClockSource::Idle,
            clock_from_quanta_us: 0,
            clock_from_topup_us: 0,
            clock_from_idle_us: 0,
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
        // A registration is the one moment a header address can begin to mean a
        // different program (the guest may have freed and reallocated the blob), so
        // nothing reflected earlier may be trusted past it.
        self.invalidate_program_reflection();
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

    /// Record the linked image's modules so the kernel's module queries can answer
    /// from them. Set once before the run, alongside [`Self::set_process_param`].
    pub fn set_modules(&mut self, modules: Vec<crate::link::LoadedModule>) {
        self.modules = modules;
    }

    /// The module whose segments contain `addr`, and its SceUID, or `None` if the
    /// address is in no loaded module (guest heap, a stack, a memory block).
    ///
    /// The UID is the module's index plus a base: modules are placed once at link
    /// time and never unloaded here, so the index is a stable identity, and offsetting
    /// it keeps a module id from colliding with the small ids threads and sync objects
    /// use.
    pub fn module_by_addr(&self, addr: u32) -> Option<(i32, &crate::link::LoadedModule)> {
        self.modules
            .iter()
            .position(|m| m.contains(addr))
            .map(|i| (MODULE_UID_BASE + i as i32, &self.modules[i]))
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
        // A fiber shares its runner's thread-local storage: on hardware it runs ON that
        // thread, so a slot it reads must be the same slot the runner writes.
        let thread = self.logical_thread(self.current);
        if let Some(&(_, addr)) = self.tls_slots.iter().find(|&&(k, _)| k == (thread, key)) {
            return addr;
        }
        // A pointer-sized slot per key is what the low-level TLS API hands out; the
        // caller stores its own per-thread pointer there.
        let addr = self.galloc(4, 4);
        self.tls_slots.push(((thread, key), addr));
        addr
    }

    /// Delete a file from the virtual filesystem, reporting whether it was there.
    /// Used by `sceAppUtilSaveDataDataRemove`, whose whole job is that the deletion is
    /// real: a title that removes a save slot and re-reads it must see it gone.
    pub fn remove_file(&mut self, path: &str) -> bool {
        let key = vfs_key(path);
        self.fs.originals.remove(&key);
        self.fs.files.remove(&key).is_some()
    }

    /// Total bytes the virtual filesystem holds under a mount (e.g. `savedata0:`),
    /// which is what a quota query has to report as USED.
    pub fn mount_used_bytes(&self, mount: &str) -> usize {
        let prefix = vfs_key(mount.trim_end_matches('/'));
        self.fs
            .files
            .iter()
            .filter(|(k, _)| k.starts_with(&prefix))
            .map(|(_, v)| v.len())
            .sum()
    }

    /// A fresh index for the next `scePhotoExportFromData`, so successive exports land
    /// in distinct files instead of overwriting one another.
    pub fn next_photo_export_index(&mut self) -> u32 {
        let i = self.photo_exports;
        self.photo_exports += 1;
        i
    }

    /// Move the heap allocation cursor to `addr`. A multi-module linked title
    /// (see [`crate::link`]) fills far more than the default 1 MiB below the heap,
    /// so the host must set this above the whole image (`LinkedProgram::alloc_base`)
    /// before the run, or allocations would overwrite guest code.
    pub fn set_alloc_base(&mut self, addr: u32) {
        self.alloc_cursor = addr;
        // The fallback SA bank is placed HERE, before a single guest instruction runs, and
        // never moves. That is what lets its address live in the host-mirror block: the
        // mirror's contract is that a slot cannot change while guest code is running, and a
        // pointer fixed before the run trivially satisfies it. Allocating it lazily at the
        // first `sceGxmSetUniformDataF` would put a host call inside that contract for no
        // gain. See [`Self::sa_bank`].
        self.ensure_sa_bank();
    }

    /// Place the fallback SA bank if it is not placed yet, and return it.
    ///
    /// Called from [`Self::set_alloc_base`], which every real host calls before the guest
    /// runs - so on a real run the bank is a constant from before frame zero, which is what
    /// the mirror slot's contract wants. The lazy arm is for a harness that stands a
    /// `VitaState` up without an arena base and then uses it (the conformance cube does);
    /// there the slot goes 0 -> address once, and a resume that read the stale 0 simply
    /// ran the handler, which is this same code.
    fn ensure_sa_bank(&mut self) -> u32 {
        if self.sa_bank == 0 {
            self.sa_bank = self.galloc(SA_BANK_BYTES, 16);
        }
        self.sa_bank
    }

    /// Guest address of the fallback SA bank, or 0 if the arena could not place one.
    ///
    /// Read by [`crate::vita::mirror::snapshot`] into the slot the inlined
    /// `sceGxmSetUniformDataF` reads it from.
    pub fn sa_bank(&self) -> u32 {
        self.sa_bank
    }

    /// Clear the bank for a new scene: zero the prefix anything wrote and reset the
    /// high-water mark.
    ///
    /// Only the prefix, because that is all a reader can see and all a later write can
    /// leave a hole in - `set_uniforms` fills the gap between the old high-water mark and
    /// its own start by RELYING on this being zero, which is the same thing the `Vec`
    /// this replaced got from `resize(end, 0.0)`.
    fn clear_sa_bank(&mut self, ctx: &mut GuestCtx) {
        let bank = self.sa_bank;
        if bank == 0 {
            return;
        }
        let len = ctx.read_u32(bank);
        for i in 0..len.min(MAX_DEFAULT_UNIFORM_REGS) {
            ctx.write_u32(bank + SA_BANK_DATA + i * 4, 0);
        }
        ctx.write_u32(bank, 0);
    }

    /// The floats the bank holds, up to its high-water mark - what a draw with no default
    /// uniform buffer bound reads as its SA bank.
    fn sa_bank_floats(&self, ctx: &GuestCtx) -> Vec<f32> {
        let bank = self.sa_bank;
        if bank == 0 {
            return Vec::new();
        }
        let len = ctx.read_u32(bank).min(MAX_DEFAULT_UNIFORM_REGS) as usize;
        ctx.read_f32s(bank + SA_BANK_DATA, len)
    }

    /// The same bank as BYTES - see [`Self::current_vertex_uniform_bytes`] for why both forms
    /// exist and why neither is derived from the other.
    fn sa_bank_bytes(&self, ctx: &GuestCtx) -> Vec<u8> {
        let bank = self.sa_bank;
        if bank == 0 {
            return Vec::new();
        }
        let len = ctx.read_u32(bank).min(MAX_DEFAULT_UNIFORM_REGS) as usize;
        ctx.read_bytes(bank + SA_BANK_DATA, len * 4)
    }

    // --- SceIoFilemgr virtual filesystem ---

    /// Preload a read-only file into the virtual filesystem before a run (e.g. a
    /// title's data file). The guest can then `sceIoOpen`/`sceIoRead` it.
    pub fn add_file(&mut self, path: &str, bytes: Vec<u8>) {
        let key = vfs_key(path);
        self.fs.originals.insert(key.clone(), strip_app0(path).trim_start_matches('/').to_string());
        self.fs.files.insert(key, bytes);
    }

    /// Serve the guest's read-only files from `backing` instead of loading them.
    ///
    /// The browser installs OPFS here: a retail container is over a gigabyte and the
    /// emulator's wasm32 heap tops out at four, so holding a title's assets in memory is
    /// what put a browser run over the edge. Native leaves this unset and keeps every
    /// file resident, which is what an in-process host should do.
    ///
    /// Call before the guest runs. Anything already added with [`add_file`](Self::add_file)
    /// stays resident and wins over a backed key of the same name.
    pub fn set_file_backing(&mut self, backing: Box<dyn FileBacking>) {
        self.fs.set_backing(backing);
    }

    /// Read back a file's current bytes (a write target after the run, for tests).
    ///
    /// Resident files only, by design: this exists to inspect what a run WROTE, and a
    /// write always makes its file resident. A borrowed slice cannot be served from a
    /// backing without materialising the file, and doing that silently on an inspection
    /// call would hide the very residency this seam exists to avoid.
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
    // --- FIOS2 path overlays -------------------------------------------------

    /// Register an overlay, returning its id. Insertion keeps the list sorted by
    /// `order` so resolution is a straight scan in application order.
    pub fn fios_overlay_add(&mut self, mut overlay: FiosOverlay) -> i32 {
        let id = FIOS_OVERLAY_ID_BASE + self.fios_overlays.len() as i32;
        overlay.id = id;
        let at = self
            .fios_overlays
            .iter()
            .position(|o| o.order > overlay.order)
            .unwrap_or(self.fios_overlays.len());
        self.fios_overlays.insert(at, overlay);
        id
    }

    /// The overlay with this id, if it is still registered.
    pub fn fios_overlay(&self, id: i32) -> Option<&FiosOverlay> {
        self.fios_overlays.iter().find(|o| o.id == id)
    }

    /// Replace a registered overlay's configuration, keeping its id. Returns whether
    /// the id resolved. The list is re-sorted because `order` may have changed.
    pub fn fios_overlay_modify(&mut self, id: i32, mut new_value: FiosOverlay) -> bool {
        let Some(i) = self.fios_overlays.iter().position(|o| o.id == id) else { return false };
        new_value.id = id;
        self.fios_overlays[i] = new_value;
        self.fios_overlays.sort_by_key(|o| o.order);
        true
    }

    /// Unregister an overlay. Returns whether the id resolved.
    pub fn fios_overlay_remove(&mut self, id: i32) -> bool {
        let before = self.fios_overlays.len();
        self.fios_overlays.retain(|o| o.id != id);
        self.fios_overlays.len() != before
    }

    /// The ids of every overlay whose `order` falls in `[min_order, max_order]`, in
    /// application order.
    pub fn fios_overlay_ids(&self, min_order: u8, max_order: u8) -> Vec<i32> {
        self.fios_overlays
            .iter()
            .filter(|o| o.order >= min_order && o.order <= max_order)
            .map(|o| o.id)
            .collect()
    }

    /// Whether the calling thread has switched overlay resolution off for itself.
    pub fn fios_overlay_disabled(&self) -> bool {
        self.fios_overlay_disabled.contains(&self.current)
    }

    /// Switch overlay resolution on or off for the calling thread. Returns the
    /// previous setting, which is what a caller saves to restore.
    pub fn fios_overlay_set_disabled(&mut self, disabled: bool) -> bool {
        let was = self.fios_overlay_disabled.contains(&self.current);
        if disabled {
            self.fios_overlay_disabled.insert(self.current);
        } else {
            self.fios_overlay_disabled.remove(&self.current);
        }
        was
    }

    /// Resolve `path` through the registered overlays whose `order` is in
    /// `[min_order, max_order]`, returning the path the filesystem should actually
    /// see.
    ///
    /// Each overlay whose `dst` is a path prefix of the current path can rewrite that
    /// prefix to its `src`. What decides whether it does is the overlay TYPE:
    ///
    /// - `OPAQUE` replaces unconditionally: `src` stands in for `dst` whether or not
    ///   anything is there, which is what makes it opaque - the original is hidden.
    /// - `TRANSLUCENT` and `WRITABLE` check `src` first and fall back to `dst`, so the
    ///   rewrite only takes when the file is really present in the overlay. (They
    ///   differ in where WRITES land, which is a property of the open, not of the
    ///   resolve.)
    /// - `NEWER` picks whichever copy has the later modification time. Nothing in
    ///   this filesystem carries a modification time unless a title set one with
    ///   `sceIoChstat`, and the type's own documented tie-break is "if both have the
    ///   same modification time, dst is used" - which is exactly the untimed case, so
    ///   an untimed pair resolves to `dst` unless only `src` exists.
    ///
    /// Overlays compose: a rewritten path is fed to the next overlay in order, so a
    /// chain of them behaves as a stack.
    pub fn fios_resolve(&self, path: &str, min_order: u8, max_order: u8) -> String {
        if self.fios_overlay_disabled() {
            return path.to_string();
        }
        let mut cur = path.to_string();
        for o in self.fios_overlays.iter().filter(|o| o.order >= min_order && o.order <= max_order) {
            let Some(rest) = strip_path_prefix(&cur, &o.dst) else { continue };
            let candidate = format!("{}{rest}", o.src);
            let exists = |p: &str| self.io_size(p).is_some() || self.io_is_dir(p);
            cur = match o.kind {
                SCE_FIOS_OVERLAY_TYPE_OPAQUE => candidate,
                SCE_FIOS_OVERLAY_TYPE_TRANSLUCENT | SCE_FIOS_OVERLAY_TYPE_WRITABLE => {
                    if exists(&candidate) {
                        candidate
                    } else {
                        cur
                    }
                }
                SCE_FIOS_OVERLAY_TYPE_NEWER => {
                    if exists(&cur) {
                        cur
                    } else if exists(&candidate) {
                        candidate
                    } else {
                        cur
                    }
                }
                // An overlay type the console does not define. Leaving the path alone
                // would silently ignore a registered overlay, so say so once and move
                // on rather than pretending it applied.
                other => {
                    tracing::error!(
                        target: "vitaslop::err",
                        kind = other, id = o.id, dst = %o.dst, src = %o.src,
                        "FIOS2 overlay has an undefined type - it is being ignored, so \
                         paths under its dst are NOT being redirected"
                    );
                    cur
                }
            };
        }
        cur
    }

    /// int sceIoMkdir: create a directory (see [`FileTable::mkdir`]).
    pub fn io_mkdir(&mut self, path: &str) -> i32 {
        self.fs.mkdir(path)
    }

    /// int sceIoRmdir: remove an empty directory (see [`FileTable::rmdir`]).
    pub fn io_rmdir(&mut self, path: &str) -> i32 {
        self.fs.rmdir(path)
    }

    /// int sceIoRemove: delete a file (see [`FileTable::remove`]).
    pub fn io_remove(&mut self, path: &str) -> i32 {
        self.fs.remove(path)
    }

    /// int sceIoRename: move a file or directory subtree (see [`FileTable::rename`]).
    pub fn io_rename(&mut self, old: &str, new: &str) -> i32 {
        self.fs.rename(old, new)
    }

    /// int sceIoChstat: apply the selected status fields to a path.
    pub fn io_chstat(&mut self, path: &str, over: FileStatOverride) -> i32 {
        self.fs.chstat(path, over)
    }

    /// The chstat overrides recorded for a path, if any.
    pub fn io_stat_override(&self, path: &str) -> Option<&FileStatOverride> {
        self.fs.stat_override(path)
    }

    /// Whether a path names a directory (explicit or implied by its contents).
    pub fn io_is_dir(&self, path: &str) -> bool {
        self.fs.is_dir(&vfs_key(path))
    }

    /// Whether `fd` is a live DIRECTORY descriptor.
    pub fn io_dir_is_open(&self, fd: i32) -> bool {
        self.fs.open_dirs.contains_key(&fd)
    }

    /// The path a live directory descriptor was opened on.
    pub fn io_dir_path(&self, fd: i32) -> Option<String> {
        self.fs.open_dirs.get(&fd).map(|d| d.path.clone())
    }

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

    /// The entries directly under `path`, for a host call that must DISCOVER a name
    /// rather than being handed one - a system service resolving the running title's
    /// own data, where the console knows the name from the installed package and we
    /// only have the package's layout. Empty if the directory does not exist.
    pub fn list_dir(&mut self, path: &str) -> Vec<DirEntry> {
        let fd = self.fs.dopen(path);
        if fd < 0 {
            return Vec::new();
        }
        let mut out = Vec::new();
        while let Some(Some(e)) = self.fs.dread(fd) {
            out.push(e);
        }
        self.fs.dclose(fd);
        out
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

    /// How many live thread records is so many that the guest must be in a retry loop.
    ///
    /// No real title holds thousands of threads at once; a title that reaches this is
    /// creating and abandoning them. Left unchecked, each one takes a stack off a bump
    /// allocator that cannot free, so the run ends in a heap exhaustion whose crash site is
    /// unrelated to the cause - an out-of-bounds access with a stack pointer near zero, in
    /// whichever thread happened to start last. Naming the entry point that is repeating
    /// turns that into a one-line diagnosis. Measured against a real case: a trophy-info
    /// thread respawning on an error it could not act on reached ~74,000.
    const RUNAWAY_THREAD_LIMIT: usize = 2048;

    /// Create a thread: allocate its own stack and record it, returning its
    /// SceUID. The entry's Thumb bit is cleared so it names the transpiled export.
    pub fn create_thread(
        &mut self,
        entry: u32,
        stack_size: u32,
        priority: i32,
        attr: u32,
        cpu_affinity: i32,
    ) -> i32 {
        if self.threads.len() >= Self::RUNAWAY_THREAD_LIMIT {
            // Report ONCE, with the entry that dominates, then keep going: the heap
            // exhaustion below will stop the run loudly anyway, and this is the line that
            // explains it.
            if !self.runaway_threads_reported {
                self.runaway_threads_reported = true;
                let mut counts: std::collections::HashMap<u32, usize> =
                    std::collections::HashMap::new();
                for t in &self.threads {
                    *counts.entry(t.entry).or_insert(0) += 1;
                }
                let worst = counts.iter().max_by_key(|(_, n)| **n);
                tracing::error!(
                    target: "vitaslop::err",
                    live = self.threads.len(),
                    entry = format_args!("{entry:#x}"),
                    repeated_entry = format_args!("{:#x}", worst.map(|(e, _)| *e).unwrap_or(0)),
                    repeated_count = worst.map(|(_, n)| *n).unwrap_or(0),
                    "RUNAWAY thread creation - the guest is spawning threads it never \
                     reaps, which will exhaust the guest heap and crash somewhere \
                     unrelated. Almost always a host call returning something the guest \
                     treats as retryable: look at what the repeated entry point calls."
                );
            }
        }
        let size = stack_size.max(0x1000);
        // A deleted thread's stack is deliberately NOT recycled here. It looks like free
        // memory - the guest allocator is a bump allocator with no free, so recycling is
        // tempting - but "the guest deleted the thread" and "the host has finished with
        // that fiber" are different events: an exited thread's stack can still be live
        // under the scheduler when the guest deletes its record. Handing it to a new
        // thread then corrupts two threads' stacks at once, which surfaces as an
        // out-of-bounds access with a nonsense stack pointer nowhere near the cause.
        // Leaking it is the safe direction; `free_stacks` records the leak so it is
        // measurable rather than invisible.
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
        self.threads.push(ThreadRec {
            uid,
            name: String::new(),
            entry: entry & !1,
            stack_top,
            stack: (stack, size),
            started: false,
            exit_code: None,
            priority,
            init_priority: priority,
            attr,
            cpu_affinity,
            signals: 0,
        });
        uid
    }

    // --- SceFiber -----------------------------------------------------------
    //
    // See [`FiberRec`] for why a fiber is backed by a scheduler thread. Everything
    // below is bookkeeping around one invariant: within a fiber chain exactly ONE of
    // {the running thread, one fiber} is runnable, and every transition hands the
    // baton over in a single step (wake the target, park the caller).

    /// `SCE_FIBER_ERROR_*`, from vitasdk `fiber.h`.
    pub const FIBER_ERROR_NULL: i32 = 0x8059_0001u32 as i32;
    pub const FIBER_ERROR_INVALID: i32 = 0x8059_0004u32 as i32;
    pub const FIBER_ERROR_PERMISSION: i32 = 0x8059_0005u32 as i32;
    pub const FIBER_ERROR_STATE: i32 = 0x8059_0006u32 as i32;

    fn fiber_index(&self, addr: u32) -> Option<usize> {
        self.fibers.iter().position(|f| f.addr == addr && !f.finalized)
    }

    /// The fiber the CURRENT thread is executing, if it is executing one.
    fn current_fiber_index(&self) -> Option<usize> {
        let cur = self.current;
        self.fibers.iter().position(|f| f.thid == cur && f.started && !f.finalized)
    }

    /// The guest `SceFiber*` the current thread is running, or 0 for a plain thread.
    /// This is `sceFiberGetSelf`.
    pub fn current_fiber(&self) -> u32 {
        self.current_fiber_index().map(|i| self.fibers[i].addr).unwrap_or(0)
    }

    /// The thread a title's own code would call itself: for a fiber's backing thread,
    /// the THREAD that ran the chain, because on hardware a fiber runs on its runner's
    /// thread and shares its identity and its thread-local storage. Everything keyed by
    /// "which thread am I" - `sceKernelGetThreadId`, `sceKernelGetTLSAddr` - has to ask
    /// this rather than the raw scheduler id, or a fiber-based job system sees a
    /// different worker identity (and a different TLS block) each time it switches.
    pub fn logical_thread(&self, thid: i32) -> i32 {
        match self.fibers.iter().find(|f| f.thid == thid && f.started && !f.finalized) {
            Some(f) if f.runner != 0 => self.logical_thread(f.runner),
            _ => thid,
        }
    }

    /// `_sceFiberInitializeImpl(fiber, name, entry, argOnInitialize, addrContext,
    /// sizeContext, params)`: record the fiber and create - but do not start - the
    /// thread that will run it.
    ///
    /// `addrContext` may be null (a fiber initialised without a stack), in which case
    /// `_sceFiberAttachContextAndSwitch` supplies one before it first runs.
    #[allow(clippy::too_many_arguments)]
    pub fn fiber_initialize(
        &mut self,
        addr: u32,
        name: String,
        entry: u32,
        arg_on_initialize: u32,
        context_addr: u32,
        context_size: u32,
    ) -> i32 {
        if addr == 0 || entry == 0 {
            return Self::FIBER_ERROR_NULL;
        }
        // Re-initialising a live fiber would strand its parked thread on a stack the
        // guest is about to reuse. That is a guest error the kernel refuses, so refuse
        // it here rather than leaving a thread parked forever.
        if let Some(i) = self.fiber_index(addr) {
            if self.fibers[i].started && self.fibers[i].runner != 0 {
                return Self::FIBER_ERROR_STATE;
            }
            self.fibers.swap_remove(i);
        }
        let uid = self.next_uid;
        self.next_uid += 1;
        // The fiber runs on the GUEST's context buffer, so this thread record owns no
        // stack of its own - `stack: (0, 0)` marks that there is nothing to reclaim
        // when the record goes away.
        self.threads.push(ThreadRec {
            uid,
            name: name.clone(),
            entry: entry & !1,
            stack_top: fiber_stack_top(context_addr, context_size),
            stack: (0, 0),
            started: false,
            exit_code: None,
            priority: DEFAULT_THREAD_PRIORITY,
            init_priority: DEFAULT_THREAD_PRIORITY,
            attr: 0,
            cpu_affinity: 0,
            signals: 0,
        });
        self.fibers.push(FiberRec {
            addr,
            name,
            entry: entry & !1,
            arg_on_initialize,
            context: (context_addr, context_size),
            thid: uid,
            started: false,
            finalized: false,
            runner: 0,
            resume_out: 0,
        });
        0
    }

    /// Give a fiber the baton: start its thread on first use, or wake it and deliver
    /// `arg_on_run` through the out-pointer of the call it is parked in.
    fn fiber_dispatch(&mut self, i: usize, runner: i32, arg_on_run: u32) {
        self.fibers[i].runner = runner;
        if self.fibers[i].started {
            let out = self.fibers[i].resume_out;
            if out != 0 {
                self.pending_stat_writes.push((out, arg_on_run));
            }
            self.pending_wakes.push(self.fibers[i].thid);
            return;
        }
        self.fibers[i].started = true;
        let (thid, entry, arg0, stack_top) = {
            let f = &self.fibers[i];
            (f.thid, f.entry, f.arg_on_initialize, fiber_stack_top(f.context.0, f.context.1))
        };
        // A fiber shares its runner's thread-local storage, because on hardware it IS
        // that thread. Bind the block before the scheduler instantiates the fiber (it
        // reads the base then), so a `__thread` variable the fiber touches is the same
        // storage the runner sees.
        let runner_tls = self.ensure_tls_block(self.logical_thread(runner));
        if runner_tls != 0 {
            self.tls_bases.push((thid, runner_tls));
        }
        if let Some(t) = self.threads.iter_mut().find(|t| t.uid == thid) {
            t.started = true;
            t.stack_top = stack_top;
        }
        self.pending_spawns.push(Reentry {
            entry,
            // `void entry(SceUInt32 argOnInitialize, SceUInt32 argOnRun)`.
            arg_len: arg0,
            arg_ptr: arg_on_run,
            r2: 0,
            stack_top,
            thid,
            priority: DEFAULT_THREAD_PRIORITY,
        });
    }

    /// `sceFiberRun(fiber, argOnRunTo, argOnRun)` called by a plain THREAD: hand the
    /// fiber the baton and park the caller until the chain returns to it. Returns
    /// `Ok(())` when the caller must park, or the errno to return instead.
    pub fn fiber_run(&mut self, addr: u32, arg_on_run_to: u32, arg_on_run_out: u32) -> Result<(), i32> {
        if self.current_fiber_index().is_some() {
            // A fiber must Switch, not Run: running nests a second chain on one thread,
            // which the baton model (and the hardware) does not have.
            return Err(Self::FIBER_ERROR_PERMISSION);
        }
        let Some(i) = self.fiber_index(addr) else { return Err(Self::FIBER_ERROR_INVALID) };
        if self.fibers[i].runner != 0 {
            return Err(Self::FIBER_ERROR_STATE);
        }
        if self.fibers[i].context.0 == 0 {
            // No stack: it would run on address 0. Refuse rather than fault later.
            return Err(Self::FIBER_ERROR_NULL);
        }
        let runner = self.current;
        self.fiber_run_out.insert(runner, arg_on_run_out);
        self.fiber_dispatch(i, runner, arg_on_run_to);
        Ok(())
    }

    /// `sceFiberSwitch(fiber, argOnRunTo, argOnRun)` called by a FIBER: pass the baton
    /// sideways, keeping the same runner. `context` optionally attaches a stack first
    /// (`_sceFiberAttachContextAndSwitch`).
    pub fn fiber_switch(
        &mut self,
        addr: u32,
        arg_on_run_to: u32,
        arg_on_run_out: u32,
        context: Option<(u32, u32)>,
    ) -> Result<(), i32> {
        let Some(from) = self.current_fiber_index() else { return Err(Self::FIBER_ERROR_PERMISSION) };
        let Some(to) = self.fiber_index(addr) else { return Err(Self::FIBER_ERROR_INVALID) };
        if to == from || self.fibers[to].runner != 0 {
            return Err(Self::FIBER_ERROR_STATE);
        }
        if let Some(c) = context {
            // Attaching a context to a fiber that is already running on one would move
            // the stack out from under live frames.
            if self.fibers[to].started {
                return Err(Self::FIBER_ERROR_STATE);
            }
            self.fibers[to].context = c;
        }
        if self.fibers[to].context.0 == 0 {
            return Err(Self::FIBER_ERROR_NULL);
        }
        let runner = self.fibers[from].runner;
        self.fibers[from].runner = 0;
        self.fibers[from].resume_out = arg_on_run_out;
        self.fiber_dispatch(to, runner, arg_on_run_to);
        Ok(())
    }

    /// `sceFiberReturnToThread(argOnReturn, argOnRun)`: give the baton back to the
    /// thread that ran this chain and park the fiber where it stands.
    pub fn fiber_return_to_thread(&mut self, arg_on_return: u32, arg_on_run_out: u32) -> Result<(), i32> {
        let Some(i) = self.current_fiber_index() else { return Err(Self::FIBER_ERROR_PERMISSION) };
        let runner = self.fibers[i].runner;
        if runner == 0 {
            return Err(Self::FIBER_ERROR_STATE);
        }
        self.fibers[i].runner = 0;
        self.fibers[i].resume_out = arg_on_run_out;
        if let Some(&out) = self.fiber_run_out.get(&runner) {
            if out != 0 {
                self.pending_stat_writes.push((out, arg_on_return));
            }
        }
        self.fiber_run_out.remove(&runner);
        self.pending_wakes.push(runner);
        Ok(())
    }

    /// `sceFiberFinalize(fiber)`: retire a fiber that is not running.
    pub fn fiber_finalize(&mut self, addr: u32) -> i32 {
        let Some(i) = self.fiber_index(addr) else { return Self::FIBER_ERROR_INVALID };
        if self.fibers[i].runner != 0 {
            return Self::FIBER_ERROR_STATE;
        }
        // A started fiber that has not run to completion is parked mid-switch: its
        // thread would never be resumed again. The kernel calls that a state error, and
        // saying so is far better than leaking a thread the scheduler still counts.
        if self.fibers[i].started && !self.thread_finished(self.fibers[i].thid) {
            tracing::warn!(
                target: "vitaslop::thread",
                fiber = format_args!("{addr:#x}"),
                "sceFiberFinalize on a fiber that is switched away, not finished - its \
                 backing thread stays parked; the guest is finalizing a live fiber"
            );
            return Self::FIBER_ERROR_STATE;
        }
        self.fibers[i].finalized = true;
        0
    }

    /// `sceFiberGetInfo(fiber, SceFiberInfo *out)`: entry, argOnInitialize, context and
    /// name, at the offsets vitasdk's `SceFiberInfo` publishes.
    pub fn fiber_info(&self, addr: u32) -> Option<(u32, u32, u32, u32, &str)> {
        let i = self.fiber_index(addr)?;
        let f = &self.fibers[i];
        Some((f.entry, f.arg_on_initialize, f.context.0, f.context.1, f.name.as_str()))
    }

    /// int sceKernelChangeThreadPriority(SceUID thid, int priority): retarget a
    /// thread's scheduler priority. `thid` 0 means the calling thread. Returns the
    /// PREVIOUS priority on success (what the kernel returns, and what a title saves
    /// to restore after a temporary boost), or an error for an unknown id.
    ///
    /// This genuinely moves the thread in the run order - the scheduler picks the
    /// highest-priority runnable thread - so a title raising a loader above the main
    /// thread gets the ordering it asked for.
    pub fn change_thread_priority(&mut self, thid: i32, priority: i32) -> Result<i32, u32> {
        let thid = if thid == 0 { self.current } else { thid };
        let priority = resolve_priority(priority);
        match self.threads.iter_mut().find(|t| t.uid == thid) {
            Some(t) => {
                let previous = t.priority;
                t.priority = priority;
                Ok(previous)
            }
            None => Err(SCE_KERNEL_ERROR_UNKNOWN_THREAD_ID),
        }
    }

    /// Everything `sceKernelGetThreadInfo` reports that this kernel models:
    /// `(name, attr, status, entry, stack base, stack size, init priority, current
    /// priority, cpu affinity, exit status)`. `thid` 0 means the calling thread.
    pub fn thread_info(&self, thid: i32) -> Option<ThreadInfo<'_>> {
        let thid = if thid == 0 { self.current } else { thid };
        let t = self.threads.iter().find(|t| t.uid == thid)?;
        Some(ThreadInfo {
            name: t.name.as_str(),
            attr: t.attr,
            status: thread_status(t, self.current),
            entry: t.entry,
            stack_base: t.stack.0,
            stack_size: t.stack.1 as i32,
            init_priority: t.init_priority,
            current_priority: t.priority,
            cpu_affinity: t.cpu_affinity,
            exit_status: t.exit_code.unwrap_or(0) as i32,
        })
    }

    /// int sceKernelSendSignal(SceUID thid): deliver one signal to a thread, waking it
    /// if it is parked in `sceKernelWaitSignal`. Counted, so a send that arrives before
    /// the wait is not lost - that ordering is the whole point of the primitive.
    pub fn send_signal(&mut self, thid: i32) -> Result<(), u32> {
        match self.threads.iter_mut().find(|t| t.uid == thid) {
            Some(t) => {
                t.signals += 1;
                if self.preemptive && self.signal_waiters.contains(&thid) {
                    self.signal_waiters.retain(|w| *w != thid);
                    self.take_signal(thid);
                    self.pending_wakes.push(thid);
                }
                Ok(())
            }
            None => Err(SCE_KERNEL_ERROR_UNKNOWN_THREAD_ID),
        }
    }

    /// Consume one pending signal from a thread, if it has one.
    pub fn take_signal(&mut self, thid: i32) -> bool {
        match self.threads.iter_mut().find(|t| t.uid == thid && t.signals > 0) {
            Some(t) => {
                t.signals -= 1;
                true
            }
            None => false,
        }
    }

    /// Park the current thread until a signal is sent to it (`sceKernelWaitSignal`).
    pub fn signal_block(&mut self) {
        self.signal_waiters.push(self.current);
    }

    /// Record the guest's own name for a thread (from `sceKernelCreateThread`).
    pub fn set_thread_name(&mut self, thid: i32, name: &str) {
        if let Some(t) = self.threads.iter_mut().find(|t| t.uid == thid) {
            t.name = name.to_string();
        }
    }

    /// int sceKernelDeleteThread(SceUID thid): drop a DORMANT thread's record and hand its
    /// stack back for reuse.
    ///
    /// Deleting a thread that is still running is refused with `SCE_KERNEL_ERROR_NOT_DORMANT`,
    /// as the kernel does - accepting it would invalidate the id of a thread the scheduler
    /// is about to resume. A thread that was created and never started is dormant too, which
    /// is why [`ThreadRec::started`] exists: neither it nor a running thread has an exit code.
    pub fn delete_thread(&mut self, thid: i32) -> Result<(), u32> {
        let Some(i) = self.threads.iter().position(|t| t.uid == thid) else {
            return Err(SCE_KERNEL_ERROR_UNKNOWN_THREAD_ID);
        };
        let t = &self.threads[i];
        if t.started && t.exit_code.is_none() {
            return Err(SCE_KERNEL_ERROR_NOT_DORMANT);
        }
        let rec = self.threads.swap_remove(i);
        self.free_stacks.push(rec.stack);
        tracing::trace!(target: "vitaslop::thread", uid = thid, "delete");
        Ok(())
    }

    /// Record a thread's requested CPU affinity mask (`sceKernelChangeThreadCpuAffinityMask`).
    ///
    /// Stored, not obeyed: this scheduler interleaves guest threads on one baton, so there
    /// is no per-core placement to honour. What matters is that the getter agrees with the
    /// setter - a title that sets an affinity and reads it back must not be contradicted.
    /// The `cpu_affinity` field is the same one `sceKernelCreateThread` fills and
    /// `sceKernelGetThreadInfo` already reports, so all three now tell one story.
    pub fn set_thread_cpu_affinity(&mut self, thid: i32, mask: i32) -> i32 {
        // 0 addresses the CALLING thread throughout the threadmgr API.
        let target = if thid == 0 { self.current } else { thid };
        match self.threads.iter_mut().find(|t| t.uid == target) {
            Some(t) => {
                t.cpu_affinity = mask;
                0
            }
            None => SCE_KERNEL_ERROR_UNKNOWN_THREAD_ID as i32,
        }
    }

    /// A thread's CPU affinity mask, or a negative error. See
    /// [`Self::set_thread_cpu_affinity`].
    pub fn thread_cpu_affinity(&self, thid: i32) -> i32 {
        let target = if thid == 0 { self.current } else { thid };
        match self.threads.iter().find(|t| t.uid == target) {
            // A thread created with 0 ("inherit") has never named a set, and the kernel
            // answers with the real one it may run on rather than the sentinel.
            Some(t) if t.cpu_affinity == 0 => crate::vita::threadmgr::CPU_MASK_USER_ALL,
            Some(t) => t.cpu_affinity,
            None => SCE_KERNEL_ERROR_UNKNOWN_THREAD_ID as i32,
        }
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
        let Some(t) = self.threads.iter_mut().find(|t| t.uid == thid) else { return false };
        t.started = true;
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
        let flip = self.flip_count;
        if let Some(t) = self.threads.iter_mut().find(|t| t.uid == thid) {
            // A thread ENDING is reported (`RUST_LOG=vitaslop::thread=debug`) for the
            // same reason its creation is: a title whose game logic has quietly gone
            // away looks exactly like one that is running and drawing, because the
            // render thread keeps resubmitting the last command buffer. The only
            // record of the difference is this line and the frame it happened on.
            tracing::debug!(
                target: "vitaslop::thread",
                thid,
                name = %t.name,
                code = format_args!("{code:#x}"),
                flip,
                "threadEnded"
            );
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
        self.semaphores.iter().any(|s| s.uid == uid)
    }

    /// The SceUID of the live semaphore created with this name, if any.
    /// `sceKernelOpenSema` resolves a name to the existing object rather than making
    /// a new one, so this is the lookup behind it. First match wins, which matches a
    /// kernel that refuses a duplicate openable name.
    pub fn sema_by_name(&self, name: &str) -> Option<i32> {
        self.semaphores.iter().find(|s| s.name == name).map(|s| s.uid)
    }

    /// Everything `sceKernelGetSemaInfo` reports: `(name, attr, init, current, max,
    /// waiting threads)`. The waiter count is derived from the park queue rather than
    /// stored, so it cannot drift from the threads actually blocked.
    pub fn sema_info(&self, uid: i32) -> Option<(&str, u32, i32, i32, i32, i32)> {
        let s = self.semaphores.iter().find(|s| s.uid == uid)?;
        let waiting = self.sema_waiters.iter().filter(|w| w.uid == uid).count() as i32;
        Some((s.name.as_str(), s.attr, s.init, s.count, s.max, waiting))
    }

    /// Try to take `need` from semaphore `uid` without blocking. Returns true if
    /// the count was available (and consumed), false otherwise.
    pub fn sema_try_acquire(&mut self, uid: i32, need: i32) -> bool {
        if let Some(s) = self.semaphores.iter_mut().find(|s| s.uid == uid) {
            if s.count >= need {
                s.count -= need;
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
                .find(|s| s.uid == uid)
                .map(|s| s.count)
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
            // Traced HERE and not at the lock site, because this is where a parked thread
            // actually becomes the owner. `lock_mutex` only ever sees the immediate acquire, so
            // counting acquisitions there undercounts every handoff - which on a two-thread
            // producer/consumer mutex means the log shows thousands of unlocks against a
            // handful of takes and reads like a thread unlocking something it never held.
            tracing::trace!(target: "vitaslop::sema", id = uid, thread = next, "mutex GRANTED on wake");
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

    // --- lightweight mutexes, whose state lives in the GUEST WORK AREA -------------
    //
    // A lightweight mutex has no kernel handle. Identity, owner and recursion count are
    // four words of the caller's own `SceKernelLwMutexWork` (see `crate::vita::lwwork`),
    // which is where the hardware keeps them and what lets the uncontended take be
    // emitted straight into guest code. Everything below is the CONTENDED half plus the
    // parked queue - the part a userspace CAS cannot do, and the part that on the device
    // is a syscall too.
    //
    // These still mirror the heavyweight [`mutex_lock`]/[`mutex_contended`]/[`mutex_unlock`]
    // in behaviour, so a `sceKernelLockLwMutex` genuinely blocks on contention and enforces
    // mutual exclusion.

    /// The record for the lightweight mutex at guest work address `work`, created (with no
    /// waiters) on first use - a title may lock a work area it zero-initialized itself
    /// without a distinct create call.
    fn lwmutex_rec(&mut self, work: u32) -> &mut LwMutexRec {
        if !self.lwmutexes.iter().any(|m| m.work == work) {
            self.lwmutexes.push(LwMutexRec { work, waiters: Vec::new() });
        }
        self.lwmutexes.iter_mut().find(|m| m.work == work).expect("just inserted")
    }

    /// Register the lightweight mutex at `work` and lay its state out there
    /// (`sceKernelCreateLwMutex`). Idempotent as a record; the work area is re-initialized,
    /// which is what a re-create means.
    pub fn lwmutex_register(&mut self, w: &mut dyn GuestWords, work: u32) {
        let _ = self.lwmutex_rec(work);
        lwwork::init(w, work);
    }

    /// Whether `work` is a lightweight mutex we have a record for (created, or already
    /// locked at least once). Used to resolve a possibly-copied work pointer.
    pub fn lwmutex_is_known(&self, work: u32) -> bool {
        self.lwmutexes.iter().any(|m| m.work == work)
    }

    /// Adopt a work area no create was ever seen for, so it can be locked - and, from the
    /// next lock on, be taken inline.
    ///
    /// A zero identity stamp is unambiguous: a byte COPY of a created mutex carries the
    /// ORIGINAL's address there, so the only work area that can read zero is one no
    /// `sceKernelCreateLwMutex` ever touched. Statically-initialized mutexes are ordinary
    /// in libc, and this is the same lazy-by-address behaviour the host record has always
    /// had - just written down where the guest can see it.
    pub fn lwmutex_adopt(&mut self, w: &mut dyn GuestWords, work: u32) {
        self.lwmutex_register(w, work);
    }

    /// Lock the lightweight mutex at `work` for the current thread. Returns true if
    /// acquired (free, or already held by this thread - recursive), false if the caller
    /// was parked behind the owner (return [`SvcOutcome::Block`]).
    ///
    /// The uncontended answer comes from [`lwwork::fast_lock`], which is the same function
    /// the inline form is compiled from - so the two paths cannot decide differently about
    /// the same four words.
    pub fn lwmutex_lock(&mut self, w: &mut dyn GuestWords, work: u32) -> bool {
        let cur = self.current;
        if lwwork::fast_lock(w, work, cur, 1) {
            return true;
        }
        // Contended, or a work area the fast path will not serve. Take it anyway if it is
        // free or ours - the fast path also refuses on a parked waiter, and the host is
        // exactly the side allowed to barge past that.
        let held = lwwork::count(w, work);
        if held == 0 || lwwork::owner(w, work) == cur {
            lwwork::set_owner_count(w, work, cur, held + 1);
            return true;
        }
        let m = self.lwmutex_rec(work);
        m.waiters.push(cur);
        let parked = m.waiters.len();
        lwwork::set_waiters(w, work, parked);
        false
    }

    /// Whether locking the lightweight mutex at `work` now would contend (another
    /// thread owns it). Used by `sceKernelTryLockLwMutex`, which fails rather than blocks.
    pub fn lwmutex_contended(&self, w: &dyn GuestWords, work: u32) -> bool {
        lwwork::count(w, work) != 0 && lwwork::owner(w, work) != self.current
    }

    /// Unlock the lightweight mutex at `work`. On full release, hand ownership to the
    /// next parked waiter (FIFO) and wake it.
    pub fn lwmutex_unlock(&mut self, w: &mut dyn GuestWords, work: u32) {
        let held = lwwork::count(w, work);
        if held == 0 {
            // An unlock with nothing held. The old host-side record silently ignored this
            // and so does this, but say so: it is either a title bug or a mutex we failed
            // to resolve, and both are worth seeing when a deadlock is being read.
            tracing::debug!(
                target: "vitaslop::sema",
                work = format_args!("{work:#010x}"),
                thread = format_args!("{:#x}", self.current),
                "sceKernelUnlockLwMutex on a mutex nothing holds"
            );
            return;
        }
        let remaining = held - 1;
        if remaining != 0 {
            lwwork::set_count(w, work, remaining);
            return;
        }
        // Fully released. Hand it straight to the next parked thread rather than freeing
        // it - a woken waiter that had to race for it could lose to a barging inline take
        // and park again, forever.
        let next = {
            let m = self.lwmutex_rec(work);
            let next = (!m.waiters.is_empty()).then(|| m.waiters.remove(0));
            lwwork::set_waiters(w, work, m.waiters.len());
            next
        };
        match next {
            Some(thid) => {
                lwwork::set_owner_count(w, work, thid, 1);
                self.pending_wakes.push(thid);
            }
            None => lwwork::set_count(w, work, 0),
        }
    }

    /// Forget the lightweight mutex at `work` (`sceKernelDeleteLwMutex`), clearing the
    /// identity stamp so a guest that kept the pointer cannot take the stale work area
    /// inline.
    pub fn lwmutex_delete(&mut self, w: &mut dyn GuestWords, work: u32) {
        lwwork::clear(w, work);
        self.lwmutexes.retain(|m| m.work != work);
    }

    /// Acquire the lightweight mutex at `work` on behalf of thread `thid` (not the
    /// current thread). Used when a lightweight-cond signal/timeout transfers a waiter
    /// back to its mutex: if free the thread takes it and is woken now, else it queues
    /// behind the owner and is woken when the owner unlocks. Work-keyed twin of
    /// [`mutex_acquire_for`](Self::mutex_acquire_for).
    fn lwmutex_acquire_for(&mut self, w: &mut dyn GuestWords, work: u32, thid: i32) {
        let held = lwwork::count(w, work);
        if held == 0 || lwwork::owner(w, work) == thid {
            lwwork::set_owner_count(w, work, thid, held + 1);
            self.pending_wakes.push(thid);
            return;
        }
        let m = self.lwmutex_rec(work);
        m.waiters.push(thid);
        let parked = m.waiters.len();
        lwwork::set_waiters(w, work, parked);
        // Woken later, when the owner unlocks.
    }

    /// Resolve the deferred lightweight-mutex handoffs queued where no guest memory was
    /// reachable, and report how many were applied.
    ///
    /// A lightweight-cond wait that TIMES OUT must re-acquire its bound mutex before it
    /// runs, and that expiry is decided in [`advance_time_to`](Self::advance_time_to) - on
    /// the scheduler's idle path, with no host call in flight and so no [`GuestCtx`]
    /// anywhere. The decision needs to READ the work area (is it free?), which is why it
    /// cannot simply be queued as a write. So it is queued as an INTENT and settled here,
    /// by the scheduler, before anything is resumed.
    ///
    /// That is exact rather than nearly exact: no guest code runs between the expiry and
    /// this drain, so there is no moment at which a guest could observe the difference.
    pub fn resolve_deferred_lwmutex(&mut self, w: &mut dyn GuestWords) -> usize {
        let queued = std::mem::take(&mut self.pending_lwmutex_acquires);
        for &(work, thid) in &queued {
            self.lwmutex_acquire_for(w, work, thid);
        }
        queued.len()
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
    pub fn lwcond_wait(&mut self, w: &mut dyn GuestWords, work: u32, timeout_us: u32) -> bool {
        let Some(mutex_work) = self.lwcond_mutex_of(work) else {
            return false;
        };
        self.lwmutex_unlock(w, mutex_work);
        let deadline = (timeout_us != 0).then(|| self.virtual_us + timeout_us as u64);
        self.lwcond_waiters.push((self.current, work, deadline));
        true
    }


    /// `sceKernelSignalLwCond`/`SignalLwCondAll`: wake one (or all) threads parked on
    /// cond `work`. Each woken thread must re-acquire the cond's bound lightweight mutex
    /// before it runs (taken now if free, else queued behind the owner), mirroring the
    /// heavyweight [`cond_signal`](Self::cond_signal).
    pub fn lwcond_signal(&mut self, w: &mut dyn GuestWords, work: u32, all: bool) {
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
                    self.lwmutex_acquire_for(w, mutex_work, thid);
                }
            }
            // No bound mutex recorded (a bare cond): just make the waiters runnable.
            None => self.pending_wakes.extend(woken),
        }
    }

    /// Park the current thread until `now + us` on the virtual clock, woken only by
    /// time. Used for `sceKernelDelayThread` and `sceAudioOutOutput` grain pacing.
    /// Add `us` to the accrued storage-latency debt and return the new total. See
    /// `vita::iofilemgr::charge_read`: the debt exists so the modelled transfer time is
    /// charged in full without a scheduler round trip per read.
    pub fn add_io_debt_us(&mut self, us: u64) -> u64 {
        self.io_debt_us = self.io_debt_us.saturating_add(us);
        self.io_debt_us
    }

    /// Clear the accrued storage-latency debt (the caller is about to park for it).
    pub fn take_io_debt_us(&mut self) -> u64 {
        core::mem::take(&mut self.io_debt_us)
    }

    /// Park the current thread on the STORAGE clock for `us` of modelled transfer time.
    ///
    /// Not [`sleep_park`](Self::sleep_park): a sleep's deadline is in `virtual_us`, which
    /// only moves on a display flip or on the scheduler's idle path. A title blocked
    /// waiting for the load produces no flips, and a sibling thread polling with a
    /// zero-length delay keeps the idle path finding a deadline that buys no time, so the
    /// game clock stands still and the load can never finish - the measured livelock. The
    /// storage clock has no such dependency.
    pub fn io_park(&mut self, us: u64) {
        let deadline = self.io_us.saturating_add(us);
        self.io_waiters.push((self.current, deadline));
    }

    /// Advance the storage clock by `us` and wake every transfer that has now completed.
    fn advance_io_by(&mut self, us: u64) {
        self.io_us = self.io_us.saturating_add(us);
        let now = self.io_us;
        let mut woken: Vec<i32> = Vec::new();
        self.io_waiters.retain(|&(thid, deadline)| {
            if deadline <= now {
                woken.push(thid);
                false
            } else {
                true
            }
        });
        self.pending_wakes.extend(woken);
    }

    /// One display flip happened: bring the storage clock up to one full frame of
    /// progress, minus whatever the quantum charges already contributed since the last
    /// flip. While a title renders, this is the whole model - the device streams one
    /// frame's worth of bytes per rendered frame, which is what a loading screen's
    /// spinner is telling the player.
    pub fn advance_io_frame(&mut self, frame_us: u64) {
        let owed = frame_us.saturating_sub(core::mem::take(&mut self.io_charged_since_flip));
        self.advance_io_by(owed);
    }

    /// A thread burned a whole scheduler quantum (or yielded) without a flip. Time
    /// passed on the device too. This is what keeps a title that spins in guest code
    /// waiting for a load - never blocking, never flipping - from waiting forever;
    /// [`advance_io_frame`](Self::advance_io_frame) nets these out so a normally
    /// rendering title still advances exactly one frame of storage time per frame.
    pub fn charge_io_quantum(&mut self, us: u64) {
        self.io_charged_since_flip = self.io_charged_since_flip.saturating_add(us);
        self.advance_io_by(us);
    }

    /// A thread burned a whole scheduler quantum of guest execution. **Game time passed
    /// while it did**, and this is the only thing that says so when the title is neither
    /// blocking nor flipping.
    ///
    /// Without this the game clock advances only on a display flip or on the scheduler's
    /// nothing-is-runnable idle path, so a guest busy-wait ON THE CLOCK ITSELF - the
    /// `do { v = sceDisplayGetVcount(); } while (v == last);` vblank spin that is
    /// ordinary, correct guest code - can never be satisfied: the thread never blocks,
    /// so the scheduler is never idle, and no flip can happen because the flip is what
    /// the spin is waiting for. Two such threads livelock the whole title. That is not a
    /// slow boot, it is a stopped clock, and it is what made this title's boot take
    /// hours while the clock advanced 60 ms.
    ///
    /// [`advance_time_frame`](Self::advance_time_frame) nets these charges out, so a
    /// title that renders normally still advances exactly one frame of game time per
    /// rendered frame and its own timers keep 60 Hz - the charges only become visible
    /// when the guest burns more CPU than a frame's worth, which is the honest answer.
    pub fn charge_cpu_quantum(&mut self, us: u64) {
        self.quantum_count = self.quantum_count.saturating_add(1);
        self.cpu_charged_since_flip = self.cpu_charged_since_flip.saturating_add(us);
        let target = self.virtual_us.saturating_add(us);
        self.clock_source = ClockSource::Quantum;
        self.advance_time_to(target);
    }

    /// `(from quanta, from the frame top-up, from idle jumps)` microseconds of game clock
    /// so far.
    ///
    /// The split, not the total. Two engines sharing this scheduler can reach very
    /// different clocks from the same guest, and the fix depends entirely on which
    /// bucket differs: a smaller `quanta` bucket means the preemption rate is wrong,
    /// while a smaller `idle` bucket means the run never goes idle - some thread is
    /// still runnable - and no amount of re-calibrating the quantum will touch it.
    pub fn clock_sources(&self) -> (u64, u64, u64) {
        (self.clock_from_quanta_us, self.clock_from_topup_us, self.clock_from_idle_us)
    }

    /// One display flip happened: bring the game clock up to one full frame of progress,
    /// minus whatever [`charge_cpu_quantum`](Self::charge_cpu_quantum) already
    /// contributed since the last flip. The storage clock's
    /// [`advance_io_frame`](Self::advance_io_frame) with the game clock's units.
    ///
    /// # The top-up hands out time the guest was never given the CPU to use
    /// `VITASLOP_FRAME_TOPUP=0` switches it off, and that is not a micro-optimisation -
    /// it changes what a displayed frame MEANS. With the top-up on, a flip advances the
    /// clock a whole frame however little guest code ran, so the render thread's next
    /// vblank wait is already satisfied and it flips again at once. Measured on a
    /// retail title's loading screen: **3120 fps, 8 thread resumes a frame, barely one
    /// whole scheduler quantum per frame** - the guest gets about a TENTH of the CPU a
    /// console frame carries, so a load that costs hardware 3,000 frames costs us
    /// 30,000+, and the wall time goes on per-frame overhead instead of on the loader.
    ///
    /// With it off, a flip advances the clock by nothing it did not earn. The render
    /// thread's vblank wait is then a real wait: its spin burns quanta, every other
    /// runnable thread is scheduled between them, and the frame ends when a frame's
    /// worth of guest execution has actually happened. Neither of the two ways the
    /// clock can stall applies - a spinning thread charges quanta, and a title that
    /// blocks everything reaches the scheduler's idle path, which advances the clock to
    /// the earliest deadline.
    ///
    /// It is a knob rather than the default because it moves every title's timing and
    /// every determinism signature.
    pub fn advance_time_frame(&mut self, frame_us: u64) {
        self.flip_count = self.flip_count.saturating_add(1);
        let charged = core::mem::take(&mut self.cpu_charged_since_flip);
        if !frame_topup() {
            return;
        }
        let owed = frame_us.saturating_sub(charged);
        let target = self.virtual_us.saturating_add(owed);
        self.clock_source = ClockSource::FrameTopup;
        self.advance_time_to(target);
    }

    /// Scheduler quanta and display flips seen so far. `QUANTUM_CPU_US` is calibrated
    /// from their ratio on a title that is rendering steadily: one quantum is worth one
    /// frame divided by the quanta a frame actually takes.
    /// Display flips completed so far - the frame number a recipe and a `@shot` count
    /// in. Diagnostics stamp it so an event can be placed on the timeline.
    pub fn flip_count(&self) -> u64 {
        self.flip_count
    }

    pub fn quantum_flip_counts(&self) -> (u64, u64) {
        (self.quantum_count, self.flip_count)
    }

    /// How much more storage time the EARLIEST outstanding transfer still owes, or
    /// `None` when nothing is in flight.
    ///
    /// This is the number the idle path compares against the next timed wait. Without it
    /// the scheduler could only ask "is a transfer outstanding", and the answer to that
    /// question is the same for a transfer with 200 us left and one with 40 ms left -
    /// which is how a modelled read came to cost no display frames at all whatever its
    /// size. See [`crate::sched`]'s idle path.
    pub fn earliest_io_remaining_us(&self) -> Option<u64> {
        self.io_waiters.iter().map(|&(_, d)| d.saturating_sub(self.io_us)).min()
    }

    /// Nothing is runnable and the game clock is being jumped to `us` ahead: the device
    /// was reading for that whole idle interval too, so credit it.
    ///
    /// Charged exactly like a quantum - and netted out of the next flip for the same
    /// reason - so a title that renders normally still advances exactly one frame of
    /// storage time per rendered frame however its idle fell.
    pub fn charge_io_idle(&mut self, us: u64) {
        self.charge_io_quantum(us);
    }

    /// Nothing is runnable and no timed wait completes sooner: complete the earliest
    /// outstanding transfer. Returns whether anything was released.
    ///
    /// **The caller must have established that no timed wait expires first.** The
    /// earlier rule here was "an outstanding transfer comes FIRST, before any timed
    /// wait", justified as "no guest code can run until something completes, so nothing
    /// observable is lost". That reasoning is wrong, and the thing it loses is the
    /// display. The render thread's vblank wait IS a timed wait, so completing the
    /// transfer ahead of it means a transfer always finishes before the next frame flips,
    /// whatever its modelled size - and the storage clock only advances on flips and
    /// quanta, so the size then buys nothing at all. Measured on a retail racer's course
    /// load: the modelled cost of a 1,996,800-byte read moved from 4,554 us to 25,020 us
    /// across a 250x bandwidth sweep and the load landed on the SAME guest frame every
    /// time, because the idle path handed it back its time immediately. The transfer and
    /// the timed wait are now honoured in time order.
    pub fn release_earliest_io(&mut self) -> bool {
        let Some(deadline) = self.io_waiters.iter().map(|&(_, d)| d).min() else {
            return false;
        };
        self.advance_io_by(deadline.saturating_sub(self.io_us));
        true
    }

    /// Whether any thread is parked paying storage latency.
    pub fn has_io_waiters(&self) -> bool {
        !self.io_waiters.is_empty()
    }

    pub fn sleep_park(&mut self, us: u64) {
        self.sleep_waiters.push((self.current, self.virtual_us.wrapping_add(us)));
    }

    /// Pace a display flip to the scanout: park the calling thread until the display can
    /// actually latch the frame it just queued, and reserve the vblank after that for the
    /// one to come. `frame_us` is one display period.
    ///
    /// # Vsync is a FLOOR on when a flip may complete, not a grant of free time
    /// Neither of the two behaviours this replaces was physical, and they failed in
    /// opposite directions.
    ///
    /// With the per-flip top-up ON, a flip advanced the clock a whole frame however
    /// little guest code had run. The render thread's next vblank wait was therefore
    /// already satisfied and it flipped again at once: measured on a retail loading
    /// screen, **3120 fps and barely one scheduler quantum per frame**, so the guest got
    /// about a TENTH of the CPU a console frame carries and a load that costs hardware
    /// 3,000 frames cost 30,000+.
    ///
    /// With it OFF, a flip advanced the clock by nothing it had not earned - but nothing
    /// stopped the guest flipping again immediately either, so the frame rate became
    /// "however fast the guest can draw". Measured on this title's front end: **4.64 ms
    /// of game clock per flip, 216 fps**, against a 60 Hz panel. Every wall-clock wait in
    /// the title then cost 3.6x the flips hardware needs, and since a frame number is
    /// what a recipe, a screenshot and `--max-frames` are keyed to, the whole timeline
    /// stretched by that factor. The race was unaffected (33.5 ms/flip, a true 30 fps),
    /// which is what makes this a pacing bug and not a speed one.
    ///
    /// The floor fixes both. A flip may not complete before the next vblank, so a cheap
    /// frame WAITS - and it waits by blocking, not by being handed time: while this
    /// thread is parked the scheduler runs every other runnable thread (charging real
    /// quanta) and only jumps the clock when nothing at all is runnable. The guest
    /// therefore gets a whole frame's worth of CPU per frame, which is the half the
    /// top-up never supplied.
    ///
    /// # A latch happens ON a vblank, not one period after the request
    /// The floor above is a duration; the scanout is a GRID. A frame that takes 17 ms
    /// of a 16.67 ms period does not latch at 17 ms - it misses the vblank and waits
    /// for the next one, so it costs 33.3 ms and the title runs at 30 Hz. Rounding the
    /// latch up to the grid is what produces the console's characteristic 60/30/20 Hz
    /// quantisation instead of a continuum, and it is also what lets a title that
    /// renders in 5 ms and then waits for vblank stay PHASE-LOCKED rather than drifting
    /// by its own render time every frame. `sceDisplayGetVcount` is already defined off
    /// this same grid ([`crate::vita::display::vcount`]), so the two agree by
    /// construction.
    ///
    /// Returns the microseconds parked, for the caller's diagnostics.
    pub fn pace_flip(&mut self, frame_us: u64) -> u64 {
        // SCE_DISPLAY_SETBUF_IMMEDIATE: the title asked for the buffer change to take
        // effect at once rather than at a vblank, which on hardware tears and does NOT
        // pace. Honouring it means no floor at all - and clearing the reservation too,
        // or a floor left over from an earlier NEXTFRAME phase would go on pacing a
        // title that has explicitly asked not to be. See [`Self::set_display_sync`].
        // A queue depth of one vblank: the caller may not run ahead of the scanout. GXM's
        // `displayQueueMaxPendingCount` would allow a title to be a frame or two ahead,
        // which raises throughput but cannot raise the LATCH rate above the panel - and
        // the latch rate is what the frame count and the game clock are made of.
        let latch = Self::at_or_after_vblank(self.next_flip_us.max(self.virtual_us), frame_us);
        self.next_flip_us = latch.saturating_add(frame_us);
        let park = latch.saturating_sub(self.virtual_us);
        self.sleep_park(park);
        park
    }

    /// `SceDisplaySetBufSync`: the buffer change takes effect immediately (and tears).
    pub const SETBUF_IMMEDIATE: u32 = 0;
    /// `SceDisplaySetBufSync`: the buffer change takes effect at the next vblank.
    pub const SETBUF_NEXTFRAME: u32 = 1;

    /// Record the `sync` argument of a `sceDisplaySetFrameBuf`.
    ///
    /// # It is RECORDED and REPORTED, and deliberately not acted on
    /// `SceDisplaySetBufSync` has exactly two values, IMMEDIATE and NEXTFRAME, and the
    /// difference between them is WHEN the scanout's framebuffer pointer changes - mid-
    /// scan, which tears, or at a vblank, which does not. It is not a swap interval, it
    /// cannot ask for 30 Hz, and it does not change how fast a title can present: the
    /// panel still scans at 60 Hz and the guest still has to draw each frame. We present
    /// whole frames, so we do not model tearing, and there is nothing left for the
    /// argument to change.
    ///
    /// Both behaviours it seemed to imply were TRIED and both were wrong, on a retail
    /// title that really does ask for IMMEDIATE:
    /// - dropping [`pace_flip`](Self::pace_flip)'s vblank floor for it re-created exactly
    ///   the bug the floor exists to fix, and
    /// - letting `sceDisplayWaitSetFrameBuf` return at once (nothing to wait for, if the
    ///   buffer changed when it was asked for) LIVELOCKED the title: it spins on that
    ///   call, and the run reached **frame 3** with **34.3 million thread resumes**, one
    ///   thread taking 25.7 million of them and burning 699 whole quanta. On hardware
    ///   that call waits for the DISPLAY to update, which is a vblank event whichever
    ///   sync mode set the buffer.
    ///
    /// So it is kept as a diagnostic: it says what the title asked for, which is worth
    /// knowing and was previously invisible.
    pub fn set_display_sync(&mut self, sync: u32) {
        // Report the first call as well as every change, not changes alone. A run that
        // never logs is otherwise ambiguous between "the title asks for the default" and
        // "the title never called this at all", and those are different worlds: the
        // second means the presents are reaching the display down some other path and
        // the sync mode being honoured here is honouring nothing.
        if !self.display_sync_seen || self.display_sync != sync {
            tracing::info!(
                target: "vitaslop::gxm",
                sync,
                first = !self.display_sync_seen,
                "display: sceDisplaySetFrameBuf sync = {} ({})",
                sync,
                if sync == Self::SETBUF_IMMEDIATE {
                    "IMMEDIATE - the buffer change takes effect at once, so presents do not wait for vblank"
                } else {
                    "NEXTFRAME - presents latch at a vblank"
                },
            );
            self.display_sync_seen = true;
            self.display_sync = sync;
        }
    }

    /// The `sync` mode the guest last asked for. See [`Self::set_display_sync`].
    pub fn display_sync(&self) -> u32 {
        self.display_sync
    }

    /// The first vblank edge at or after `t`, on a grid of `period` starting at 0.
    fn at_or_after_vblank(t: u64, period: u64) -> u64 {
        if period == 0 {
            return t;
        }
        t.div_ceil(period).saturating_mul(period)
    }

    /// Park the calling thread until the `n`th vblank edge STRICTLY after now, and
    /// return the microseconds parked. `n` of 0 parks for nothing.
    ///
    /// The distinction from `sleep_park(n * period)` is the whole point: a vblank is a
    /// scanout heartbeat the guest joins, not a stopwatch it starts. Parking a full
    /// period from wherever the guest happened to call over-waits by half a period on
    /// average and can never phase-lock, so a title that renders in 5 ms and waits for
    /// vblank ran at 5 + 16.67 ms (46 fps) instead of the 60 the hardware gives it.
    /// Count one vblank-spin park - see [`report_vblank_spin_parks`].
    pub fn note_vblank_spin_park(&mut self) {
        VBLANK_SPIN_PARKS.fetch_add(1, std::sync::atomic::Ordering::Relaxed);
    }

    pub fn vblank_park(&mut self, n: u64, period: u64) -> u64 {
        if n == 0 {
            return 0;
        }
        // STRICTLY after: a wait issued exactly on an edge waits for the next one, or a
        // loop of them would return instantly forever and never advance the clock.
        let first = Self::at_or_after_vblank(self.virtual_us.saturating_add(1), period);
        let edge = first.saturating_add((n - 1).saturating_mul(period));
        let park = edge.saturating_sub(self.virtual_us);
        self.sleep_park(park);
        park
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
        // WHERE the game clock's time came from, cumulatively. See
        // [`clock_sources`](Self::clock_sources): a total alone cannot say whether one
        // engine's clock runs slower because it charges less per quantum or because it
        // never reaches the idle path that JUMPS the clock to the next deadline, and
        // those have nothing in common as bugs.
        let gained = to_us.saturating_sub(self.virtual_us);
        match self.clock_source {
            ClockSource::Quantum => self.clock_from_quanta_us += gained,
            ClockSource::FrameTopup => self.clock_from_topup_us += gained,
            ClockSource::Idle => self.clock_from_idle_us += gained,
        }
        // Back to the default immediately: the tag describes THIS call only, and letting
        // it persist would credit the next untagged advance to whichever path happened to
        // run last - which is precisely the kind of quiet misattribution this split exists
        // to prevent.
        self.clock_source = ClockSource::Idle;
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
                // No guest memory is reachable here (see `resolve_deferred_lwmutex`), and
                // deciding this needs to READ the work area, so queue the intent and let
                // the scheduler settle it before anything resumes.
                Some(mutex_work) => self.pending_lwmutex_acquires.push((mutex_work, thid)),
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

    /// Create a semaphore, returning its SceUID. `name`, `attr` and `max` are kept
    /// because they are observable: `sceKernelOpenSema` resolves the name and
    /// `sceKernelGetSemaInfo` reports all three.
    pub fn create_sema(&mut self, name: &str, attr: u32, init: i32, max: i32) -> i32 {
        let uid = self.new_uid();
        self.semaphores.push(SemaRec { uid, name: name.to_string(), attr, init, max, count: init });
        uid
    }

    /// Wait on a semaphore: take `n` from its count (never blocks in the single-
    /// thread model; the count floors at 0).
    pub fn sema_wait(&mut self, uid: i32, n: i32) {
        if let Some(s) = self.semaphores.iter_mut().find(|s| s.uid == uid) {
            s.count = (s.count - n).max(0);
        }
    }

    /// Signal a semaphore: add `n` to its count.
    pub fn sema_signal(&mut self, uid: i32, n: i32) {
        if let Some(s) = self.semaphores.iter_mut().find(|s| s.uid == uid) {
            s.count += n;
        }
    }

    /// Create an event flag with an initial bit pattern, returning its SceUID.
    pub fn create_event_flag(&mut self, init: u32) -> i32 {
        let uid = self.new_uid();
        self.event_flags.push((uid, init));
        uid
    }

    /// Record the guest's name for an event flag, so every report that names the uid can
    /// name the subsystem too. Empty names are not stored.
    pub fn name_event_flag(&mut self, uid: i32, name: &str) {
        if !name.is_empty() {
            self.event_flag_names.insert(uid, name.to_string());
        }
    }

    /// The guest's name for an event flag, or `""`.
    pub fn event_flag_name(&self, uid: i32) -> &str {
        self.event_flag_names.get(&uid).map(String::as_str).unwrap_or("")
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

    /// Guest-region bytes reserved at the top for the MAIN THREAD'S STACK, which the
    /// heap may never hand out.
    ///
    /// The main thread's stack pointer starts near the top of the region and grows
    /// DOWN, while this bump allocator grows UP; nothing but this reservation keeps
    /// them apart. Capping only at the region ceiling is not enough, and the failure is
    /// vicious: the heap marches into live stack frames and the next perfectly ordinary
    /// guest write - a `memcpy` into a buffer we just handed out - silently overwrites a
    /// caller's saved registers. The crash then lands in an innocent function, frames
    /// and often minutes later, with a callee-saved register holding a float.
    ///
    /// The reserve covers the host's startup headroom above the initial SP plus room
    /// for real stack depth beneath it.
    const MAIN_STACK_RESERVE_BYTES: u32 = 5 * 1024 * 1024;

    /// Allocate `size` bytes of real guest memory aligned to `align`, returning
    /// the guest address (0 on exhaustion). Deterministic: a pure function of the
    /// allocation order. The bump cursor is capped below the guest region ceiling
    /// (`base + mem_bytes`) by [`Self::MAIN_STACK_RESERVE_BYTES`), for two reasons: the
    /// indirect-dispatch address table lives immediately above the ceiling (a stray
    /// write there corrupts the table and faults every later indirect call), and the
    /// main thread's stack descends through the top of the region itself. An allocation
    /// that would reach either is a real out-of-memory: return 0 - loudly - rather than
    /// a pointer that aliases something live.
    pub fn galloc(&mut self, size: u32, align: u32) -> u32 {
        let a = align.max(4);
        let p = (self.alloc_cursor + a - 1) & !(a - 1);
        let ceiling = self
            .base
            .wrapping_add(self.mem_bytes)
            .wrapping_sub(Self::MAIN_STACK_RESERVE_BYTES);
        let end = p.wrapping_add(size.max(4));
        if end > ceiling || end < p {
            // Silence here is what turns a leak into memory corruption somewhere else
            // entirely, so an exhausted heap always says so.
            tracing::error!(
                target: "vitaslop::err",
                size,
                align,
                cursor = format_args!("{:#x}", self.alloc_cursor),
                ceiling = format_args!("{ceiling:#x}"),
                "guest heap EXHAUSTED - returning a null allocation rather than encroaching \
                 on the main thread stack or the dispatch table"
            );
            return 0;
        }
        self.alloc_cursor = end;
        p
    }

    /// The UID of the memory block that fully contains `[addr, addr + size)`, or
    /// `None`. A zero `size` asks about the single byte at `addr`, which is how a
    /// caller with only a pointer phrases the question.
    pub fn memblock_containing(&self, addr: u32, size: u32) -> Option<i32> {
        let end = addr.wrapping_add(size.max(1));
        self.memblocks
            .iter()
            .find(|b| addr >= b.base && end <= b.base.wrapping_add(b.size))
            .map(|b| b.uid)
    }

    // --- SceNet: the offline socket table ------------------------------------
    //
    // Real descriptors and real local state, no host sockets. See `vita::net` for what
    // is modelled and why. Everything here is a pure function of the guest's own call
    // sequence, so two runs of the same recipe produce identical socket ids.

    /// `sceNetSocket`: allocate a descriptor. Ids start above the file-descriptor range
    /// so a socket id mistakenly passed to `sceIoClose` cannot close a file.
    pub fn net_socket(&mut self, name: &str, domain: i32, ty: i32, protocol: i32) -> i32 {
        let id = NET_FD_BASE + self.net_sockets.len() as i32;
        self.net_sockets.push(NetSocket {
            id,
            name: name.to_string(),
            domain,
            ty,
            protocol,
            local: (0, 0),
            listening: false,
            options: Vec::new(),
            closed: false,
        });
        id
    }

    fn net_socket_mut(&mut self, id: i32) -> Option<&mut NetSocket> {
        self.net_sockets.iter_mut().find(|s| s.id == id && !s.closed)
    }

    pub fn net_socket_exists(&self, id: i32) -> bool {
        self.net_sockets.iter().any(|s| s.id == id && !s.closed)
    }

    pub fn net_close(&mut self, id: i32) -> bool {
        match self.net_socket_mut(id) {
            Some(s) => {
                s.closed = true;
                true
            }
            None => false,
        }
    }

    pub fn net_bind(&mut self, id: i32, ip: u32, port: u16) -> bool {
        match self.net_socket_mut(id) {
            Some(s) => {
                s.local = (ip, port);
                true
            }
            None => false,
        }
    }

    pub fn net_listen(&mut self, id: i32) -> bool {
        match self.net_socket_mut(id) {
            Some(s) => {
                s.listening = true;
                true
            }
            None => false,
        }
    }

    pub fn net_local_addr(&self, id: i32) -> Option<(u32, u16)> {
        self.net_sockets.iter().find(|s| s.id == id && !s.closed).map(|s| s.local)
    }

    pub fn net_set_opt(&mut self, id: i32, level: i32, name: i32, value: u32) -> bool {
        match self.net_socket_mut(id) {
            Some(s) => {
                s.options.retain(|&(l, n, _)| (l, n) != (level, name));
                s.options.push((level, name, value));
                true
            }
            None => false,
        }
    }

    /// An option never set reads back as 0, the kernel's default for every option a
    /// title queries here. `None` means the DESCRIPTOR is bad, which is a different
    /// answer from "the option is zero" and must not be conflated.
    pub fn net_get_opt(&self, id: i32, level: i32, name: i32) -> Option<u32> {
        let s = self.net_sockets.iter().find(|s| s.id == id && !s.closed)?;
        Some(s.options.iter().find(|&&(l, n, _)| (l, n) == (level, name)).map(|&(_, _, v)| v).unwrap_or(0))
    }

    /// `sceNetShowNetstat`'s output: the live socket table, one line each.
    pub fn net_netstat(&self) -> String {
        use std::fmt::Write;
        let mut s = String::from("netstat: offline model, no host sockets\n");
        for k in self.net_sockets.iter().filter(|s| !s.closed) {
            let (ip, port) = k.local;
            let b = ip.to_le_bytes();
            let _ = writeln!(
                s,
                "  fd={} name={:?} domain={} type={} proto={} local={}.{}.{}.{}:{}{}",
                k.id, k.name, k.domain, k.ty, k.protocol, b[0], b[1], b[2], b[3], port,
                if k.listening { " LISTEN" } else { "" }
            );
        }
        s
    }

    /// The guest address of the CALLING thread's `sceNetErrnoLoc` slot, allocated on
    /// first use. Per thread, because two workers failing at once must not overwrite
    /// each other's reason - which is why the API hands back a location, not a value.
    pub fn net_errno_addr(&mut self) -> u32 {
        let thid = self.logical_thread(self.current);
        if let Some(&(_, addr)) = self.net_errno.iter().find(|&&(t, _)| t == thid) {
            return addr;
        }
        let addr = self.galloc(4, 4);
        self.net_errno.push((thid, addr));
        addr
    }

    /// Record why the last SceNet call failed, where `sceNetErrnoLoc` will read it.
    pub fn net_set_errno(&mut self, code: i32) {
        let addr = self.net_errno_addr();
        if addr != 0 {
            self.pending_stat_writes.push((addr, code as u32));
        }
    }

    pub fn net_resolver_create(&mut self) -> i32 {
        let id = NET_RESOLVER_BASE + self.net_resolvers.len() as i32;
        self.net_resolvers.push((id, 0));
        id
    }

    pub fn net_resolver_destroy(&mut self, rid: i32) -> bool {
        let before = self.net_resolvers.len();
        self.net_resolvers.retain(|&(id, _)| id != rid);
        self.net_resolvers.len() != before
    }

    pub fn net_resolver_set_error(&mut self, rid: i32, code: i32) {
        if let Some(e) = self.net_resolvers.iter_mut().find(|(id, _)| *id == rid) {
            e.1 = code;
        }
    }

    pub fn net_resolver_error(&self, rid: i32) -> Option<i32> {
        self.net_resolvers.iter().find(|(id, _)| *id == rid).map(|&(_, e)| e)
    }

    pub fn net_epoll_create(&mut self) -> i32 {
        let id = NET_EPOLL_BASE + self.net_epolls.len() as i32;
        self.net_epolls.push((id, Vec::new()));
        id
    }

    pub fn net_epoll_exists(&self, eid: i32) -> bool {
        self.net_epolls.iter().any(|(id, _)| *id == eid)
    }

    pub fn net_epoll_destroy(&mut self, eid: i32) -> bool {
        let before = self.net_epolls.len();
        self.net_epolls.retain(|(id, _)| *id != eid);
        self.net_epolls.len() != before
    }

    /// `SCE_NET_EPOLL_CTL_ADD` (1) / `MOD` (2) / `DEL` (3) over the registered set.
    pub fn net_epoll_control(&mut self, eid: i32, op: i32, id: i32) -> bool {
        match self.net_epolls.iter_mut().find(|(e, _)| *e == eid) {
            Some((_, set)) => {
                match op {
                    3 => set.retain(|&s| s != id),
                    _ => {
                        if !set.contains(&id) {
                            set.push(id);
                        }
                    }
                }
                true
            }
            None => false,
        }
    }

    /// Mint a fresh opaque GXM handle.
    pub fn new_handle(&mut self) -> u32 {
        let h = self.next_handle;
        self.next_handle += 1;
        h
    }

    /// A `SceGxmVertexProgram *` / `SceGxmFragmentProgram *` handle, as a real GUEST
    /// structure carrying the two facts a reserve needs (see [`crate::vita::gxmprog`]).
    ///
    /// This is what the shader patcher does on hardware - the handle it returns points into
    /// memory the guest gave it - and the reason to do it here is that a counter has nowhere
    /// to keep a memoised size, which is what makes
    /// `sceGxmReserve*DefaultUniformBuffer` (1,189 crossings a frame on one title) a bump
    /// instead of a boundary crossing.
    ///
    /// Falls back to an opaque counter if the guest heap cannot supply sixteen bytes: the
    /// handle is still unique and every map keyed by it still works, and the reserve simply
    /// stays on the host for that program - which is exactly what the inline form's
    /// identity-stamp guard is for.
    pub fn new_program_handle(&mut self, ctx: &mut GuestCtx, header: u32) -> u32 {
        let size = self.uniform_size_bytes(ctx, header);
        let block = self.galloc(crate::vita::gxmprog::BYTES, 4);
        if block == 0 {
            return self.new_handle();
        }
        crate::vita::gxmprog::init(ctx, block, size, header);
        block
    }

    /// Allocate a memory block of `size` and record it, returning its SceUID, or 0 when
    /// the arena cannot satisfy it.
    ///
    /// A freed block is reused before the bump cursor moves (first fit over the
    /// release-ordered free list, so the choice is deterministic). Without that, a title
    /// that cycles screens - freeing one screen's buffers and allocating the next one's -
    /// grows the arena forever and eventually gets a null allocation for a request the
    /// console would have satisfied from the memory it just handed back.
    ///
    /// Returning 0 on exhaustion matters as much as the reuse: the caller turns it into a
    /// real `sceKernelAllocMemBlock` error, where handing back a live SceUID whose base is
    /// 0 is a hollow success the guest cannot detect until it writes through the pointer.
    pub fn alloc_memblock(&mut self, size: u32, align: u32) -> i32 {
        let want = size.max(4);
        let a = align.max(4);
        let reused = self.freed_memblocks.iter().position(|&(base, sz)| {
            sz >= want && base & (a - 1) == 0
        });
        let base = match reused {
            Some(i) => {
                let (base, sz) = self.freed_memblocks.remove(i);
                // Keep the remainder available rather than rounding the whole hole up to
                // this request; a 4 KiB reuse of a 2 MiB hole must not lose 2 MiB.
                let rest = sz - want;
                if rest >= a {
                    let split = base + want;
                    let aligned = (split + a - 1) & !(a - 1);
                    if aligned < base + sz {
                        self.freed_memblocks.push((aligned, base + sz - aligned));
                    }
                }
                base
            }
            None => self.galloc(size, align),
        };
        if base == 0 {
            return 0;
        }
        let uid = self.next_uid;
        self.next_uid += 1;
        self.memblocks.push(MemBlock { uid, base, size });
        uid
    }

    /// The base address of the block with SceUID `uid`, if known.
    pub fn memblock_base(&self, uid: i32) -> Option<u32> {
        self.memblocks.iter().find(|b| b.uid == uid).map(|b| b.base)
    }

    /// Record a vertex program's attribute layout and per-stream strides.
    pub fn set_vertex_program(
        &mut self,
        handle: u32,
        attributes: Vec<crate::capture::VertexAttribute>,
        streams: Vec<(u32, bool)>,
        program_header: u32,
    ) {
        let streams: Vec<VertexStreamInfo> = streams
            .into_iter()
            .map(|(stride, per_instance)| VertexStreamInfo { stride, per_instance })
            .collect();
        // The packed layout every draw of this program will capture into - see
        // `VertexProgramInfo::packed_attributes` for why it belongs here and not in the draw.
        let used = attributes
            .iter()
            .map(|a| a.stream_index as usize)
            .max()
            .map(|m| m + 1)
            .unwrap_or(0);
        let used_streams: std::sync::Arc<[VertexStreamInfo]> =
            (0..used).map(|i| streams.get(i).copied().unwrap_or_default()).collect();
        let mut base = Vec::with_capacity(used_streams.len());
        let mut packed_stride = 0u32;
        for s in used_streams.iter() {
            base.push(packed_stride);
            packed_stride += s.stride;
        }
        let stream_base: std::sync::Arc<[u32]> = base.into();
        let packed_attributes: std::sync::Arc<[crate::capture::VertexAttribute]> = attributes
            .iter()
            .map(|a| {
                let mut a = *a;
                a.offset += stream_base.get(a.stream_index as usize).copied().unwrap_or(0) as u16;
                a.stream_index = 0;
                a
            })
            .collect();
        let single_stream = matches!(&*used_streams, [s] if !s.per_instance);
        self.vertex_programs.insert(
            handle,
            VertexProgramInfo {
                attributes,
                streams,
                program_header,
                packed_attributes,
                used_streams,
                stream_base,
                packed_stride,
                single_stream,
            },
        );
    }

    /// The `SceGxmProgram*` a vertex program handle was created from, if recorded.
    fn vertex_program_header(&self, handle: u32) -> u32 {
        self.vertex_programs.get(&handle).map(|i| i.program_header).unwrap_or(0)
    }

    /// Record `sceGxmShaderPatcherCreateFragmentProgram`'s handle -> (`SceGxmProgram*`, the
    /// blend equation it was created with).
    pub fn set_fragment_program(
        &mut self,
        handle: u32,
        program_header: u32,
        blend: crate::capture::BlendState,
    ) {
        self.fragment_programs.insert(handle, (program_header, blend));
    }

    fn fragment_program_header(&self, handle: u32) -> u32 {
        self.fragment_programs.get(&handle).map(|(h, _)| *h).unwrap_or(0)
    }

    /// The blend equation a fragment program handle was created with. An unknown handle
    /// yields the GXM default (no blending), which is what a NULL `blendInfo` means.
    fn fragment_program_blend(&self, handle: u32) -> crate::capture::BlendState {
        self.fragment_programs.get(&handle).map(|(_, b)| *b).unwrap_or_default()
    }

    /// Record a color surface, keyed by its guest struct address.
    pub fn set_color_surface(&mut self, addr: u32, surface: crate::capture::ColorSurface) {
        self.report_overlapping_color_surfaces(&surface);
        self.color_surfaces.push((addr, surface));
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
    fn report_overlapping_color_surfaces(&mut self, new: &crate::capture::ColorSurface) {
        // ONE byte per pixel: the smallest any `SceGxmColorFormat` can be. The real formats
        // here are wider, so this is a hard LOWER bound on the memory a surface occupies and
        // an overlap it finds is an overlap under every format. A diagnostic that guesses the
        // width would report pairs that do not really collide, and a report that cries wolf is
        // worse than none - this one is only allowed to be silent, never wrong.
        let bytes = |s: &crate::capture::ColorSurface| -> u64 {
            u64::from(s.stride_pixels.max(s.width)) * u64::from(s.height)
        };
        let (lo, hi) = (u64::from(new.data_addr), u64::from(new.data_addr) + bytes(new));
        if new.data_addr == 0 || hi == lo {
            return;
        }
        for (_, old) in &self.color_surfaces {
            let (olo, ohi) = (u64::from(old.data_addr), u64::from(old.data_addr) + bytes(old));
            if old.data_addr == 0 || old.data_addr == new.data_addr || ohi <= lo || hi <= olo {
                continue;
            }
            let key = (old.data_addr.min(new.data_addr), old.data_addr.max(new.data_addr));
            if self.reported_surface_overlaps.insert(key) {
                eprintln!(
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

    // --- scene assembly (used by the gxm handlers) ---

    pub fn begin_scene(
        &mut self,
        ctx: &mut GuestCtx,
        color: Option<crate::capture::ColorSurface>,
        depth: Option<crate::capture::DepthSurface>,
        // The render target's `SceGxmMultisampleMode` - see `capture::Scene::multisample`.
        multisample: u32,
    ) {
        // Texture snapshots deliberately SURVIVE the scene - see `TextureSnapshots`
        // for what invalidates them instead. Only the verifier is re-armed here.
        self.texture_snapshots.begin_scene();
        // Recycle the default-uniform arena for the new scene. Every draw's uniforms
        // are snapshotted into the draw at record time, so last scene's buffers are
        // dead by now; see [`Self::alloc_default_uniform_buffer`] for what happens
        // when this is NOT recycled.
        if self.gxm_context != 0 {
            crate::vita::gxmctx::rewind_uniform_ring(ctx, self.gxm_context);
        }
        self.scene =
            Some(crate::capture::Scene { color, depth, multisample, draws: Vec::new(), precompile: Default::default() });
        self.clear_sa_bank(ctx);
    }

    // --- The sticky GXM context state ---------------------------------------
    //
    // It lives in the guest's context block, not here - see [`crate::vita::gxmctx`] for
    // why. These are the read side: every one resolves through `self.gxm_context`, so
    // there is exactly one home for each fact and no host copy to fall out of step with
    // an inlined setter that never reaches the host at all.

    /// Adopt the context `sceGxmCreateContext` just built.
    ///
    /// The capture keeps ONE scene and ONE draw stream, so a second context would
    /// interleave two guests' draws into one frame. That is a real limitation and it says
    /// so rather than quietly serving the newer context.
    pub fn adopt_gxm_context(&mut self, context: u32) {
        if self.gxm_context != 0 && self.gxm_context != context {
            tracing::warn!(
                target: "vitaslop::gxm",
                previous = format_args!("{:#x}", self.gxm_context),
                context = format_args!("{context:#x}"),
                "a SECOND sceGxmContext was created - the capture has one scene and one draw \
                 stream, so draws from both will interleave into one frame"
            );
        }
        self.gxm_context = context;
    }

    /// The bound `SceGxmVertexProgram *` handle.
    pub fn bound_vertex_program(&self, ctx: &GuestCtx) -> u32 {
        self.gxm_state_word(ctx, crate::vita::gxmctx::off::VERTEX_PROGRAM)
    }

    /// The `SceGxmProgram *` header of the bound FRAGMENT program, resolved from the
    /// handle in the context block. 0 when nothing usable is bound.
    ///
    /// Resolved on read rather than at bind time: the bind is a store the transpiler can
    /// inline, and a lookup that happens once per draw instead of once per bind is cheaper
    /// anyway on a title that rebinds the same program repeatedly.
    pub fn bound_fragment_program_header(&self, ctx: &GuestCtx) -> u32 {
        self.fragment_program_header(self.bound_fragment_program(ctx))
    }

    /// The blend equation of the bound fragment program. Baked in at
    /// `sceGxmShaderPatcherCreateFragmentProgram`, so it is a pure function of the handle.
    pub fn bound_fragment_blend(&self, ctx: &GuestCtx) -> crate::capture::BlendState {
        self.fragment_program_blend(self.bound_fragment_program(ctx))
    }

    /// The bound `SceGxmFragmentProgram *` handle.
    fn bound_fragment_program(&self, ctx: &GuestCtx) -> u32 {
        self.gxm_state_word(ctx, crate::vita::gxmctx::off::FRAGMENT_PROGRAM)
    }

    /// The guest vertex buffer bound to each stream index by `sceGxmSetVertexStream`.
    /// GXM allows up to [`MAX_VERTEX_STREAMS`]; a mesh that splits per-vertex data from
    /// per-instance data (particles, decals, instanced props) uses more than one, and
    /// capturing only stream 0 decodes those attributes out of the wrong buffer.
    pub fn bound_streams(&self, ctx: &GuestCtx) -> [u32; MAX_VERTEX_STREAMS] {
        match self.gxm_context {
            0 => [0; MAX_VERTEX_STREAMS],
            c => crate::vita::gxmctx::streams(ctx, c),
        }
    }

    /// The live GXM fixed-function pipeline state (cull/depth/stencil/viewport/...),
    /// snapshotted into each recorded draw. Sticky across scenes, exactly like the real
    /// GXM context - because it IS the real GXM context.
    pub fn render_state(&self, ctx: &GuestCtx) -> crate::capture::RenderState {
        match self.gxm_context {
            0 => crate::capture::RenderState::default(),
            c => crate::vita::gxmctx::load(ctx, c),
        }
    }

    /// One word of the context block, or 0 when there is no context.
    fn gxm_state_word(&self, ctx: &GuestCtx, offset: u32) -> u32 {
        match self.gxm_context {
            0 => 0,
            c => crate::vita::gxmctx::get(ctx, c, offset),
        }
    }

    /// Write one word of the context block. The precomputed paths use this: they replay a
    /// prebuilt state object as the equivalent sequence of setters, so what they change is
    /// the same context state a `sceGxmSet*` would.
    fn set_gxm_state_word(&self, ctx: &mut GuestCtx, offset: u32, value: u32) {
        if self.gxm_context != 0 {
            crate::vita::gxmctx::set(ctx, self.gxm_context, offset, value);
        }
    }

    /// Record `sceGxmSetFragmentTexture(context, unit, texture)`: copy the texture's control
    /// words into sampler `unit`'s slot of the context block. A zero address unbinds it.
    ///
    /// This is the FALLBACK half of the call. The transpiler emits the copy inline for the
    /// ordinary case (`InlineOp::CopyArgIndexed`), so what reaches here is a null texture, an
    /// out-of-range unit, or a pointer outside guest memory - the cases the inline form hands
    /// back precisely because this side defines them.

    /// Decode a list of texture bindings into snapshotted [`crate::capture::BoundTexture`]s.
    ///
    /// Shared by the fragment and vertex stages so the two can never decode a texture
    /// differently: the control words, the recorded exact format, the sampler state and the
    /// nearby-format search all mean the same thing whichever stage bound it.
    ///
    /// Takes the bindings as a SLICE the caller has moved out of `self` (`mem::take`, put
    /// back after), because the per-unit control state has to be read through a borrow of
    /// `self` while the snapshot cache is borrowed mutably for the decode. It used to take
    /// them by value, which cloned both binding lists on EVERY draw - hundreds of times a
    /// frame - for lists the caller owns and does not mutate here.
    fn snapshot_bound_textures(
        &mut self,
        ctx: &GuestCtx,
        bindings: &[TextureBinding],
    ) -> Vec<crate::capture::BoundTexture> {
        // >>> THE NEARBY-FORMAT SEARCH IS FOR NULL HANDLES ONLY, AND IT IS MEMOISED.
        //
        // It is read in exactly one place: `decode_texture` hands it to
        // `report_zero_texture_handle` inside the `binding.is_null()` arm, and nowhere
        // else. It used to be computed whenever the address had no RECORDED format, which
        // is the ordinary case for a texture bound by VALUE
        // [[vitaslop-texture-binding-by-value]] rather than an error - so it ran for most
        // units of most draws. And it is a LINEAR SCAN of every texture the title has ever
        // initialised.
        //
        // MEASURED with `bench --at` on a live race: "draw: snapshot textures" was 46% of
        // the entire frame, 17.9 us per draw, while moving 0.0 MB - a diagnostic costing
        // half the frame rate and producing no data. Both engines paid it; this is shared
        // runtime code.
        //
        // Memoised because a null handle rebound every frame would otherwise pay the scan
        // every frame. The report fires once per (unit, address), so an answer that later
        // goes stale can only change a message that will never be printed again.
        for b in bindings {
            if b.is_null()
                && self.texture_format(b.addr).is_none()
                && !self.nearby_texture_cache.contains_key(&b.addr)
            {
                let found = self.nearest_recorded_texture(b.addr, 4096);
                self.nearby_texture_cache.insert(b.addr, found);
            }
        }
        // Only the recorded FORMAT and the nearby-handle diagnostic still come from host state.
        // The sampler wrap modes, filters, LOD bias and gamma now come out of the binding's own
        // control words, which `decode_texture` already holds - so they cannot go stale, and they
        // follow a by-value copy of the struct the way the hardware's do.
        // Taken out and put back rather than allocated: this runs on every draw, and the buffer
        // it needs is the same one every time. See `texture_unit_scratch`.
        let mut unit_state = std::mem::take(&mut self.texture_unit_scratch);
        unit_state.clear();
        unit_state.extend(bindings.iter().map(|&b| {
            let format = self.texture_format(b.addr);
            // Only for a handle with no format of its own: is there an initialised texture
            // NEARBY? A struct the guest inits at one address and binds at another (off by a
            // fixed member offset, or copied by value) is a completely different bug from one
            // it never initialised at all, and the two are indistinguishable from the zero
            // control words alone. Searching a window answers it in the run that hit it.
            // Resolved in the pre-pass above, for null handles only - see it for why.
            let nearby = self.nearby_texture_cache.get(&b.addr).copied().flatten();
            (b, format, nearby)
        }));
        let snapshots = &mut self.texture_snapshots;
        let out = unit_state
            .iter()
            .filter_map(|(binding, format, nearby)| {
                decode_texture(ctx, snapshots, binding, *format, *nearby)
            })
            .collect();
        self.texture_unit_scratch = unit_state;
        out
    }

    /// `_sceGxmSetVertexTexture`: bind a texture to a VERTEX-stage sampler unit.
    ///
    /// Kept in its own list rather than merged with the fragment units, because the two stages
    /// have independent unit numbering and a shader can sample the same unit number in both
    /// with different textures.
    pub fn bind_vertex_texture(&mut self, ctx: &GuestCtx, unit: u32, texture_addr: u32) {
        let binding = TextureBinding::read(ctx, unit, texture_addr, false);
        // Bumped only on a REAL change, so a title that rebinds the same texture keeps the
        // draw-side shortcut. See [`Self::vertex_texture_gen`].
        let mut changed = true;
        match self.bound_vertex_textures.binary_search_by_key(&unit, |b| b.unit) {
            Ok(i) => {
                if texture_addr == 0 {
                    self.bound_vertex_textures.remove(i);
                } else if self.bound_vertex_textures[i] == binding {
                    changed = false;
                } else {
                    self.bound_vertex_textures[i] = binding;
                }
            }
            Err(i) => {
                if texture_addr != 0 {
                    self.bound_vertex_textures.insert(i, binding);
                } else {
                    changed = false;
                }
            }
        }
        if changed {
            self.vertex_texture_gen += 1;
        }
    }

    pub fn bind_fragment_texture(&mut self, ctx: &mut GuestCtx, unit: u32, texture_addr: u32) {
        // Into the CONTEXT BLOCK, which is where the inlined form of this call writes. The
        // handler and the emitted code must leave byte-identical state, because a draw reads
        // one array and cannot tell which path filled a slot - and after 1,275 binds a frame
        // go inline, this handler runs only for the cases the inline form declines (a null
        // texture, an out-of-range unit), which are exactly the ones a divergence would hide
        // in. See `crate::vita::gxmctx::TexBinding`.
        let context = self.gxm_context;
        if context == 0 {
            self.report_bind_without_context();
            return;
        }
        let words = if texture_addr == 0 {
            [0; 4]
        } else {
            [
                ctx.read_u32(texture_addr),
                ctx.read_u32(texture_addr + 4),
                ctx.read_u32(texture_addr + 8),
                ctx.read_u32(texture_addr + 12),
            ]
        };
        crate::vita::gxmctx::set_texture_binding(
            ctx,
            context,
            unit,
            crate::vita::gxmctx::TexBinding { addr: texture_addr, words, from_precomputed: false },
        );
    }

    /// Report - once - a texture bind that arrived before `sceGxmCreateContext`, so its state
    /// has nowhere to live.
    ///
    /// Silently dropping it would be a draw missing a texture thousands of frames later, with
    /// nothing pointing back at the bind. This cannot happen through a conforming title (GXM
    /// takes the context as its first argument), which is why it is a report and not a case.
    fn report_bind_without_context(&mut self) {
        if !self.reported_no_gxm_context {
            self.reported_no_gxm_context = true;
            tracing::warn!(
                target: "vitaslop::gxm",
                "a texture was bound before any sceGxmContext was created, so the binding has \
                 nowhere to live and is DROPPED"
            );
        }
    }

    /// Record the exact `SceGxmTextureFormat` set on a `SceGxmTexture*` (by
    /// `sceGxmTextureInit*`/`SetFormat`), so a later decode recovers the exact
    /// channel swizzle rather than the lossy 3-bit control-word field.
    pub fn set_texture_format(&mut self, texture_addr: u32, format: u32) {
        let changed = self.texture_formats.insert(texture_addr, format) != Some(format);
        TEXTURE_INITS.fetch_add(1, std::sync::atomic::Ordering::Relaxed);
        // The per-binding memo derives its answer partly from this map
        // ([`TextureSnapshots::templates`]), so a recorded format that CHANGES invalidates it.
        // Dropped wholesale rather than by key: the memo is keyed on control words and this is
        // keyed on an address, so there is no key to remove - and a texture re-initialised at a
        // live address is a load-screen event, not a per-frame one.
        if changed {
            self.texture_snapshots.templates.clear();
            // ...and the per-scene lists BUILT from those templates, for the same reason. A set
            // cached earlier in this scene describes the texture as it was before the guest
            // re-initialised it, and the old code re-derived it on the very next draw.
            self.texture_snapshots.snapshot_sets.clear();
            self.texture_snapshots.decoded.clear();
        }
    }


    pub fn texture_format(&self, texture_addr: u32) -> Option<u32> {
        self.texture_formats.get(&texture_addr).copied()
    }

    /// The initialised texture nearest `addr` within `+-window` bytes, as `(signed byte delta,
    /// its format)`. Diagnostic only.
    ///
    /// A `sceGxmSetFragmentTexture` of an address we have no format for is ambiguous: the guest
    /// may never have initialised a texture there, or it may have initialised one a few bytes
    /// away and bound an interior pointer or a by-value copy. Those need opposite fixes and look
    /// identical at the binding, so the answer has to come from the neighbourhood.
    pub fn nearest_recorded_texture(&self, addr: u32, window: u32) -> Option<(i64, u32)> {
        self.texture_formats
            .iter()
            .filter_map(|(&a, &f)| {
                let d = i64::from(a) - i64::from(addr);
                (d != 0 && d.unsigned_abs() <= u64::from(window)).then_some((d, f))
            })
            .min_by_key(|(d, _)| d.abs())
    }

    /// The color surface recorded for `addr` (its `SceGxmColorSurface*` struct
    /// address), if the guest initialized one there.
    pub fn color_surface(&self, addr: u32) -> Option<crate::capture::ColorSurface> {
        self.color_surfaces.iter().find(|(a, _)| *a == addr).map(|(_, s)| *s)
    }

    /// The sticky extra state for a `SceGxmTexture*`, or GXM defaults if never set.
    fn texture_extra(&self, texture_addr: u32) -> TextureExtra {
        self.texture_extra.get(&texture_addr).copied().unwrap_or_default()
    }

    /// Mutable slot for a texture's extra state, inserting GXM defaults on first touch.
    fn texture_extra_mut(&mut self, texture_addr: u32) -> &mut TextureExtra {
        self.texture_extra.entry(texture_addr).or_default()
    }

    /// Record the explicit byte stride a `sceGxmTextureInitLinearStrided` established - the one
    /// field of word 0 that cannot be packed into the guest's words. See [`TextureExtra`].
    pub fn set_texture_stride(&mut self, texture_addr: u32, byte_stride: u32) {
        self.texture_extra_mut(texture_addr).byte_stride = byte_stride;
    }

    /// Report the first `sceGxmTextureSetGammaMode` of a run.
    ///
    /// The mode itself is stored in the guest's control word 0 by the handler; this only says so
    /// once, because a gamma texture is sampled through an sRGB format and that is a real change
    /// in how its texels are decoded - the counterpart of [`Self::set_color_surface_gamma`].
    pub fn note_texture_gamma(&mut self, texture_addr: u32, gamma: u32) {
        if gamma != 0 {
            static SAID: std::sync::atomic::AtomicBool = std::sync::atomic::AtomicBool::new(false);
            if !SAID.swap(true, std::sync::atomic::Ordering::Relaxed) {
                eprintln!(
                    "gxm texture: sceGxmTextureSetGammaMode({texture_addr:#x}, {gamma:#x}) - \
                     this texture is sampled through an sRGB format, so its texels are decoded \
                     on fetch as the hardware does"
                );
            }
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
    ///
    /// On hardware a non-`NONE` mode makes the ROP sRGB-ENCODE on write, so a shader writing
    /// linear values lands gamma-encoded in memory and every later pass reads them that way.
    /// The mode reaches the renderer through the scene's target (`RttTarget::gamma`), which
    /// renders such a surface through an sRGB VIEW of the same texture - the encode then
    /// happens after blending, exactly where the hardware does it.
    pub fn set_color_surface_gamma(&mut self, surface_addr: u32, gamma: u32) {
        if gamma != 0 && self.color_surface_gamma.iter().all(|(a, _)| *a != surface_addr) {
            {
                eprintln!(
                    "gxm surface: sceGxmColorSurfaceSetGammaMode({surface_addr:#x}, {gamma:#x}) - \
                     writes to this surface are sRGB-encoded after blending, as the hardware \
                     does; the renderer reports which view it actually rendered through."
                );
            }
        }
        self.color_surface_gamma.retain(|(a, _)| *a != surface_addr);
        self.color_surface_gamma.push((surface_addr, gamma));
    }

    /// The `SceGxmColorSurfaceGammaMode` recorded for the surface struct at `addr`, or 0.
    pub fn color_surface_gamma_mode(&self, addr: u32) -> u32 {
        self.color_surface_gamma.iter().rev().find(|(a, _)| *a == addr).map(|(_, g)| *g).unwrap_or(0)
    }

    /// The GPU notification region, allocating it on first use. Returns a guest
    /// pointer to `SCE_GXM_NOTIFICATION_COUNT` (512) u32 slots.
    pub fn notification_region(&mut self) -> u32 {
        if self.notification_region == 0 {
            self.notification_region = self.galloc(512 * 4, 16);
        }
        self.notification_region
    }

    // --- GPU memory mappings ------------------------------------------------
    //
    // Mapping is a no-op here (the guest's pages already ARE the memory the capture
    // reads), but UNmapping is not: after it the guest may hand those pages to its own
    // allocator, and anything we cached against those addresses is then a snapshot of
    // somebody else's data. GXM gives the size only on the map call, so the range has to
    // be remembered to be able to invalidate on the unmap.

    /// `sceGxmMapMemory(base, size, attr)` / `sceGxmMapVertexUsseMemory` /
    /// `sceGxmMapFragmentUsseMemory`: remember the range.
    pub fn gxm_map(&mut self, base: u32, size: u32) {
        if base != 0 && size != 0 {
            self.gxm_mappings.insert(base, size);
        }
    }

    /// `sceGxmUnmapMemory(base)` and the two USSE variants. Drops the mapping and every
    /// texture snapshot taken from it. An unmap of a base that was never mapped is the
    /// guest's error, not ours, and is reported rather than ignored.
    pub fn gxm_unmap(&mut self, base: u32) -> i32 {
        match self.gxm_mappings.remove(&base) {
            Some(size) => {
                self.texture_snapshots.invalidate_range(base, size as usize);
                0
            }
            None => {
                tracing::warn!(
                    target: "vitaslop::gxm",
                    base = format_args!("{base:#x}"),
                    "gxmUnmap of an address that was never mapped"
                );
                // SCE_GXM_ERROR_INVALID_VALUE.
                0x8021_0000u32 as i32
            }
        }
    }

    // --- Occlusion queries --------------------------------------------------

    /// `sceGxmSetVisibilityBuffer(context, bufferBase, stridePerCore)`.
    pub fn set_visibility_buffer(&mut self, base: u32, stride_per_core: u32) {
        self.visibility_buffer = base;
        self.visibility_stride = stride_per_core;
        self.visibility_counts.clear();
    }

    /// Called once per recorded draw: if the front-face visibility test is enabled,
    /// add this draw's vertex count to the slot it names.
    ///
    /// The real GPU counts SAMPLES THAT PASSED the depth test. Nothing here rasterizes
    /// at capture time, so the count is the drawn vertex count instead: a real,
    /// deterministic quantity that rises with the amount of geometry submitted under
    /// the query, and is nonzero exactly when the guest drew something. What it does NOT
    /// model is occlusion - geometry behind other geometry still counts - so a title
    /// using the query to hide a sun flare behind scenery will show the flare. That is
    /// an approximation, so it says so, once.
    ///
    /// # Why this is NOT fixed with a real GPU occlusion query, deliberately
    /// WebGPU has occlusion queries and wiring one up is a morning's work. It is the wrong
    /// move, and the reason is worth stating so nobody spends that morning:
    ///
    /// - The result would enter GUEST MEMORY. The visibility buffer is read by guest code,
    ///   which branches on it, so the value decides what the title does next - not merely how
    ///   it looks. Feeding it a GPU-measured number makes guest execution depend on the GPU.
    /// - Two engines would then diverge on purpose. The desktop oracle and the browser use
    ///   different adapters with different rasterisation at the sample level, so the same
    ///   frame could take different branches on each, and the pixel oracle would stop being an
    ///   oracle for anything downstream of a query.
    /// - It is also ASYNCHRONOUS. A query result is available a frame or more after the pass
    ///   that produced it, while the guest reads the buffer when the scene ends. Serving it
    ///   would mean either stalling on the GPU every scene or handing the guest last frame's
    ///   answer - and the second is a different wrong number, arrived at more expensively.
    ///
    /// So the approximation stays, and the trade is: a flare that should be hidden is visible
    /// (a bounded, visible, reported error) instead of a title whose control flow depends on
    /// which GPU is present (an unbounded, invisible, unreproducible one). Change this only
    /// with a determinism story, not because the warning is annoying.
    fn accumulate_visibility(&mut self, blk: &crate::vita::gxmctx::Block<'_>, index_count: u32) {
        use crate::vita::gxmctx::off;
        if blk.word(off::FRONT_VISIBILITY_TEST_ENABLE) == 0 || self.visibility_buffer == 0 {
            return;
        }
        static REPORTED: std::sync::atomic::AtomicBool = std::sync::atomic::AtomicBool::new(false);
        if !REPORTED.swap(true, std::sync::atomic::Ordering::Relaxed) {
            // >>> STAYS AT `warn`. It was briefly moved to `debug` on the argument that a
            // deliberate decision is not a defect, and that argument is wrong: the OUTPUT is
            // still incorrect. Occluded geometry reports visible, guest code branches on it,
            // and a title can therefore behave differently here than on hardware. "We chose
            // this" describes why it has not been fixed, not whether it is broken - and a
            // warning that is quiet because the cause is known is a warning that has been
            // silenced. It fires ONCE per run, which is the right volume for a standing
            // incorrect result. Fix it (a real query plus a determinism story) or leave it
            // loud; do not lower it.
            tracing::warn!(
                target: "vitaslop::gxm",
                "an occlusion query is live; the capture does not rasterize, so the visibility \
                 buffer receives the SUBMITTED vertex count, not the count of samples that \
                 passed - occluded geometry still reports visible (e.g. a flare that should be \
                 hidden behind scenery is not). This is a DELIBERATE approximation: the result \
                 is read by guest code and branched on, so a real GPU query would make guest \
                 control flow depend on which adapter is present. See `accumulate_visibility`."
            );
        }
        let slot = blk.word(off::FRONT_VISIBILITY_TEST_INDEX);
        *self.visibility_counts.entry(slot).or_insert(0) += index_count;
    }

    /// Flush the scene's accumulated occlusion counts into the guest's visibility
    /// buffer, which is when the GPU would have written them (at scene end).
    pub fn flush_visibility(&mut self, ctx: &mut GuestCtx) {
        if self.visibility_buffer == 0 {
            return;
        }
        for (&slot, &count) in &self.visibility_counts {
            ctx.write_u32(self.visibility_buffer.wrapping_add(slot * 4), count);
        }
        self.visibility_counts.clear();
    }

    // --- Render target driver memblock --------------------------------------

    /// Remember the `driverMemBlock` UID from a render target's params so
    /// `sceGxmRenderTargetGetDriverMemBlock` returns exactly what the guest supplied.
    pub fn set_render_target_mem_block(&mut self, render_target: u32, mem_block: u32) {
        self.render_target_mem_blocks.insert(render_target, mem_block);
    }

    /// The `driverMemBlock` of `render_target`, or `SCE_UID_INVALID_UID` (-1) when the
    /// target was created asking sceGxm to allocate its own - which is exactly what the
    /// guest passed in that case.
    pub fn render_target_mem_block(&self, render_target: u32) -> u32 {
        self.render_target_mem_blocks.get(&render_target).copied().unwrap_or(0xffff_ffff)
    }

    /// Remember the extent a render target was created with
    /// (`SceGxmRenderTargetParams::width`/`height`).
    pub fn set_render_target_extent(&mut self, render_target: u32, width: u32, height: u32) {
        self.render_target_extents.insert(render_target, (width, height));
    }

    /// The `(width, height)` of `render_target`, or `None` if it was never created
    /// here. This, not the colour surface, is what a scene rasterizes into: a title is
    /// free to hand `sceGxmBeginScene` a colour surface whose own width/height fields
    /// are meaningless (a render-to-texture pass that only writes depth, or one whose
    /// engine fills the struct from a template), and this title does exactly that -
    /// its map pass carries 20,160 triangles into a surface initialised 1x1.
    pub fn render_target_extent(&self, render_target: u32) -> Option<(u32, u32)> {
        self.render_target_extents.get(&render_target).copied()
    }

    // --- Paletted textures ---------------------------------------------------

    /// `sceGxmTextureSetPalette(texture, paletteData)`.
    pub fn set_texture_palette(&mut self, texture: u32, palette: u32) {
        self.texture_palettes.insert(texture, palette);
    }

    /// The palette bound to `texture`, or 0.
    pub fn texture_palette(&self, texture: u32) -> u32 {
        self.texture_palettes.get(&texture).copied().unwrap_or(0)
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
        match pdraw::OFF_STREAM.get(stream_index as usize) {
            Some(&off) => ctx.write_u32(precomputed + off, data),
            None => tracing::warn!(
                target: "vitaslop::gxm",
                stream_index,
                data = format_args!("{data:#x}"),
                "precomputedDrawSetVertexStream beyond SCE_GXM_MAX_VERTEX_STREAMS - DROPPED"
            ),
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
    /// >>> ONE BULK READ, NOT TEN WORD READS, and the block is the most-read structure in a
    /// race frame.
    ///
    /// A `dyn GuestMemory` word read is a bounds check plus a virtual call, and in the browser
    /// it crosses into a `SharedArrayBuffer` view; the bytes are never the cost, the CALLS are
    /// ([[vitaslop-count-calls-not-bytes-across-the-guest-boundary]]). This is called once per
    /// `sceGxmDrawPrecomputed`, which a measured race grid issues **411 times a frame** - so the word
    /// form was ~4,100 crossings a frame to move 1.8 kB that sit CONSECUTIVELY in guest memory.
    /// The whole block is [`pdraw::WORDS`] words with no gaps worth skipping.
    fn precomputed_draw_read(ctx: &GuestCtx, precomputed: u32) -> Option<PrecomputedDraw> {
        let mut buf = [0u8; pdraw::WORDS as usize * 4];
        ctx.read_into(precomputed, &mut buf);
        let word = |off: u32| {
            let i = off as usize;
            u32::from_le_bytes([buf[i], buf[i + 1], buf[i + 2], buf[i + 3]])
        };
        if word(pdraw::OFF_MAGIC) != pdraw::MAGIC {
            return None;
        }
        Some(PrecomputedDraw {
            vertex_program: word(pdraw::OFF_VERTEX_PROGRAM),
            streams: std::array::from_fn(|i| word(pdraw::OFF_STREAM[i])),
            primitive: word(pdraw::OFF_PRIMITIVE),
            index_format: word(pdraw::OFF_INDEX_FORMAT),
            index_addr: word(pdraw::OFF_INDEX_ADDR),
            index_count: word(pdraw::OFF_INDEX_COUNT),
        })
    }

    /// Replay `sceGxmDrawPrecomputed(context, precomputedDraw)`: bind the precomputed
    /// draw's vertex program + stream-0 buffer and record it into the current scene,
    /// exactly as a `sceGxmDraw` would. The bound textures and reserved uniform buffer
    /// are whatever the guest set on the context around this call (sticky GXM state).
    pub fn draw_precomputed(&mut self, ctx: &mut GuestCtx, precomputed: u32) {
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
        // The precomputed block's bindings become the CONTEXT's, exactly as the equivalent
        // sequence of `sceGxmSetVertexProgram` / `sceGxmSetVertexStream` calls would leave
        // them - so they go where those setters put them, and not into a second copy here.
        use crate::vita::gxmctx::off;
        self.set_gxm_state_word(ctx, off::VERTEX_PROGRAM, d.vertex_program);
        for (i, &addr) in d.streams.iter().enumerate() {
            self.set_gxm_state_word(ctx, off::STREAMS + i as u32 * 4, addr);
        }
        self.record_draw(ctx, d.primitive, d.index_format, d.index_addr, d.index_count);
    }

    // --- Precomputed vertex/fragment state ----------------------------------
    //
    // A precomputed state bundles the uniform buffer + textures for one shader stage
    // into a guest struct the game builds once and binds each draw. We record the
    // guest-set pointers per state address, then `bind_precomputed_*_state` copies them
    // into the live bind state so `record_draw` snapshots the same bytes it would on the
    // direct `sceGxmSetUniformDataF`/`sceGxmSetFragmentTexture` path.

    /// The guest state struct at `state`, ENSURED: stamped `magic` with an arrays block of
    /// `block_bytes` behind it (see [`crate::vita::gxmstate`]). Returns the block address.
    ///
    /// A struct that already carries the stamp keeps its block - `Init` re-zeroes what it
    /// must - and one that does not (never initialised, or a setter arrived first, which
    /// the old table served through `entry().or_default()`) gets a fresh zeroed block from
    /// the guest heap. The title's own `memBlock` is deliberately NOT used: its size is the
    /// real driver's, which this engine does not define.
    fn ensure_state_block(
        &mut self,
        ctx: &mut GuestCtx,
        state: u32,
        magic: u32,
        block_bytes: u32,
    ) -> u32 {
        use crate::vita::gxmstate::off;
        if ctx.read_u32(state.wrapping_add(off::MAGIC)) == magic {
            let block = ctx.read_u32(state.wrapping_add(off::BLOCK));
            if block != 0 {
                return block;
            }
        }
        let block = self.galloc(block_bytes, 16);
        for w in 0..off::BYTES / 4 {
            ctx.write_u32(state.wrapping_add(w * 4), 0);
        }
        for w in 0..block_bytes / 4 {
            ctx.write_u32(block.wrapping_add(w * 4), 0);
        }
        ctx.write_u32(state.wrapping_add(off::BLOCK), block);
        ctx.write_u32(state.wrapping_add(off::MAGIC), magic);
        block
    }

    /// `sceGxmPrecomputedVertexStateInit(state, vertexProgram, memBlock)`: stamp the guest
    /// struct, attach its arrays block, and memoise the program facts the bind needs -
    /// including the stage's uniform SIZE, which is fixed the moment the program exists
    /// (the same move the inlined reserve rests on). Replaces any prior state there.
    pub fn precomputed_vertex_state_init(&mut self, ctx: &mut GuestCtx, state: u32, vertex_program: u32) {
        use crate::vita::gxmstate::{self, off};
        let program_header = self.vertex_program_header(vertex_program);
        let block =
            self.ensure_state_block(ctx, state, gxmstate::MAGIC_VERTEX, gxmstate::VERTEX_BLOCK_BYTES);
        // Re-init REPLACES: the arrays block and the mutable fields go back to zero.
        for w in 0..gxmstate::VERTEX_BLOCK_BYTES / 4 {
            ctx.write_u32(block.wrapping_add(w * 4), 0);
        }
        // What the vertex bind's record carries: the reflected extent, never smaller than
        // the header's own declared default-uniform size - computed here once, exactly as
        // the bind used to compute it per call. Both inputs are immutable while the
        // program is registered (see `ProgramReflection`).
        let size = self
            .reflected_uniform_size_bytes(ctx, program_header)
            .max(default_uniform_buffer_bytes(ctx, program_header));
        ctx.write_u32(state.wrapping_add(off::HANDLE), vertex_program);
        ctx.write_u32(state.wrapping_add(off::HEADER), program_header);
        ctx.write_u32(state.wrapping_add(off::BUF), 0);
        ctx.write_u32(state.wrapping_add(off::SIZE), size);
        self.precomputed_state_count += 1;
        self.cross_precomputed_state_programs(ctx, program_header, true);
        self.report_precomputed_state_programs();
    }

    /// `sceGxmPrecomputedFragmentStateInit(state, fragmentProgram, memBlock)`.
    pub fn precomputed_fragment_state_init(&mut self, ctx: &mut GuestCtx, state: u32, fragment_program: u32) {
        use crate::vita::gxmstate::{self, off};
        let program_header = self.fragment_program_header(fragment_program);
        let block = self.ensure_state_block(
            ctx,
            state,
            gxmstate::MAGIC_FRAGMENT,
            gxmstate::FRAGMENT_BLOCK_BYTES,
        );
        for w in 0..gxmstate::FRAGMENT_BLOCK_BYTES / 4 {
            ctx.write_u32(block.wrapping_add(w * 4), 0);
        }
        // The fragment bind's record uses the reflected size alone, exactly as the bind
        // used to compute it per call.
        let size = self.uniform_size_bytes(ctx, program_header);
        ctx.write_u32(state.wrapping_add(off::HANDLE), fragment_program);
        ctx.write_u32(state.wrapping_add(off::HEADER), program_header);
        ctx.write_u32(state.wrapping_add(off::BUF), 0);
        ctx.write_u32(state.wrapping_add(off::SIZE), size);
        self.precomputed_state_count += 1;
        self.cross_precomputed_state_programs(ctx, program_header, false);
        self.report_precomputed_state_programs();
    }

    /// Offer the shader pairs a title's PRECOMPUTED STATES imply, so the renderer can compile
    /// them while the loading screen that declares them is still up.
    ///
    /// # Why this list, when the patcher's cross product is refuted
    /// A title that calls `sceGxmShaderPatcherCreateFragmentProgram` with `vertexProgram = NULL`
    /// names no pair, and pairing everything it ever created with everything else is hopeless -
    /// MEASURED on a retail title at 865 x 375 = 324,375 candidates, and tried, and refused
    /// ([[vitaslop-precompile-cross-product-refuted]]). A precomputed state is a much stronger
    /// statement: the title is declaring, ahead of any draw, that this exact program is one it
    /// intends to draw with. The same title names **16 distinct vertex + 16 distinct fragment
    /// programs** that way, all of them by frame 8,821 with its race live at 9,600 - 256
    /// candidates and 779 frames of loading screen to compile them in.
    ///
    /// Only the pairs a NEW header adds are offered, so this is 16 + 15 + ... rather than 256
    /// per call, and `push_precompile_pair` drops any pair already queued. The renderer keeps
    /// the ones that LINK; a candidate that does not link costs a parse and nothing more.
    fn cross_precomputed_state_programs(&mut self, ctx: &GuestCtx, header: u32, vertex: bool) {
        if header == 0 {
            return;
        }
        let (own, other) = if vertex {
            (&mut self.precomputed_state_vertex_headers, &self.precomputed_state_fragment_headers)
        } else {
            (&mut self.precomputed_state_fragment_headers, &self.precomputed_state_vertex_headers)
        };
        if own.contains(&header) {
            return;
        }
        own.push(header);
        let others = other.clone();
        for o in others {
            let (v, f) = if vertex { (header, o) } else { (o, header) };
            self.push_precompile_pair(ctx, v, f);
        }
    }

    /// Say, on a bounded ladder, how many DISTINCT programs the title's PRECOMPUTED STATES
    /// name, and by what frame.
    ///
    /// # Why this number is the one that decides the loading-screen hitch
    /// A title that creates its fragment programs with `vertexProgram = NULL` never names a
    /// shader PAIR, so nothing can be prepared from the patcher - and the cross product over
    /// every program it ever created is hopeless: MEASURED on a retail title, **865 vertex x 375
    /// fragment = 324,375 candidates**, against a 4,096 cap the old experiment silently hit
    /// (which is the whole reason that experiment saved 43 ms and was written up as a
    /// refutation of the idea rather than of its truncation).
    ///
    /// A PRECOMPUTED STATE is a different and much smaller declaration: it is the title
    /// saying, ahead of time, "this program is one I will draw with". If a race's states name
    /// a few dozen programs between them, the cross product over THOSE is a few hundred
    /// candidates rather than a third of a million - and it is knowable while the loading
    /// screen is still up. This line is what says whether that is true, per title.
    ///
    /// Printed rather than knob-gated, for the same reason as
    /// [`Self::report_program_creation_frame`]: a handful of lines a run, answering a question
    /// every session on this hitch has to ask first.
    fn report_precomputed_state_programs(&self) {
        // The distinct-header lists ARE the two counts: `cross_precomputed_state_programs`
        // pushes each non-zero header once.
        let n = self.precomputed_state_count;
        if n > 4 && !n.is_power_of_two() && n % 50 != 0 {
            return;
        }
        eprintln!(
            "gxm precompile: precomputed states name {} distinct vertex + {} distinct fragment \
             programs ({} states) by frame {}",
            self.precomputed_state_vertex_headers.len(),
            self.precomputed_state_fragment_headers.len(),
            n,
            self.cur_frame
        );
    }

    /// `sceGxmPrecomputed{Vertex,Fragment}StateSetDefaultUniformBuffer(state, buffer)`:
    /// store the guest pointer the game will write this stage's uniforms into. The record
    /// is created lazily so a setter before `Init` (unexpected) still lands.
    pub fn precomputed_vertex_state_set_uniform_buffer(&mut self, ctx: &mut GuestCtx, state: u32, buffer: u32) {
        use crate::vita::gxmstate::{self, off};
        self.ensure_state_block(ctx, state, gxmstate::MAGIC_VERTEX, gxmstate::VERTEX_BLOCK_BYTES);
        ctx.write_u32(state.wrapping_add(off::BUF), buffer);
    }
    pub fn precomputed_fragment_state_set_uniform_buffer(&mut self, ctx: &mut GuestCtx, state: u32, buffer: u32) {
        use crate::vita::gxmstate::{self, off};
        self.ensure_state_block(ctx, state, gxmstate::MAGIC_FRAGMENT, gxmstate::FRAGMENT_BLOCK_BYTES);
        ctx.write_u32(state.wrapping_add(off::BUF), buffer);
    }

    /// `sceGxmPrecomputedVertexStateSetUniformBuffer(state, bufferIndex, bufferData)`: store
    /// the NON-default uniform buffer binding, applied to the context block's table when the
    /// state is bound (the same table the direct `sceGxmSetVertexUniformBuffer` writes).
    pub fn precomputed_vertex_state_set_nondefault_uniform_buffer(
        &mut self,
        ctx: &mut GuestCtx,
        state: u32,
        index: u32,
        buffer: u32,
    ) {
        use crate::vita::gxmstate::{self};
        if (index as usize) < crate::vita::gxmctx::MAX_UNIFORM_BUFFERS {
            let block =
                self.ensure_state_block(ctx, state, gxmstate::MAGIC_VERTEX, gxmstate::VERTEX_BLOCK_BYTES);
            ctx.write_u32(block.wrapping_add(index * 4), buffer);
        }
    }

    /// `sceGxmPrecomputed{Vertex,Fragment}StateGetDefaultUniformBuffer(state)`: the pointer
    /// last set (0 if never set), so a Set/Get round-trips faithfully.
    pub fn precomputed_vertex_state_uniform_buffer(&self, ctx: &GuestCtx, state: u32) -> u32 {
        use crate::vita::gxmstate::{self, off};
        if ctx.read_u32(state.wrapping_add(off::MAGIC)) == gxmstate::MAGIC_VERTEX {
            ctx.read_u32(state.wrapping_add(off::BUF))
        } else {
            0
        }
    }
    pub fn precomputed_fragment_state_uniform_buffer(&self, ctx: &GuestCtx, state: u32) -> u32 {
        use crate::vita::gxmstate::{self, off};
        if ctx.read_u32(state.wrapping_add(off::MAGIC)) == gxmstate::MAGIC_FRAGMENT {
            ctx.read_u32(state.wrapping_add(off::BUF))
        } else {
            0
        }
    }

    /// `sceGxmPrecomputed{Vertex,Fragment}StateSetTexture(state, index, texture)`: bind a
    /// `SceGxmTexture*` to this stage's sampler `index` (0 unbinds), replacing any prior
    /// binding at that index - written into the state's guest texture array, in the
    /// context block's own slot layout, so the fragment bind is one wholesale copy.
    pub fn precomputed_vertex_state_set_texture(&mut self, ctx: &mut GuestCtx, state: u32, index: u32, texture: u32) {
        use crate::vita::gxmstate::{self};
        if index as usize >= crate::vita::gxmctx::MAX_TEXTURE_UNITS {
            self.report_precomputed_texture_unit(index, texture);
            return;
        }
        let b = TextureBinding::read(ctx, index, texture, true);
        let block =
            self.ensure_state_block(ctx, state, gxmstate::MAGIC_VERTEX, gxmstate::VERTEX_BLOCK_BYTES);
        gxmstate::write_texture_slot(
            ctx,
            block.wrapping_add(gxmstate::VERTEX_BLOCK_TEXTURES),
            index,
            b.addr,
            b.words,
        );
    }
    pub fn precomputed_fragment_state_set_texture(&mut self, ctx: &mut GuestCtx, state: u32, index: u32, texture: u32) {
        use crate::vita::gxmstate::{self};
        if index as usize >= crate::vita::gxmctx::MAX_TEXTURE_UNITS {
            self.report_precomputed_texture_unit(index, texture);
            return;
        }
        let b = TextureBinding::read(ctx, index, texture, true);
        let block = self.ensure_state_block(
            ctx,
            state,
            gxmstate::MAGIC_FRAGMENT,
            gxmstate::FRAGMENT_BLOCK_BYTES,
        );
        gxmstate::write_texture_slot(ctx, block, index, b.addr, b.words);
    }

    /// Say that a precomputed-state texture setter named a unit past the array. The old
    /// path recorded any index and dropped it at BIND time (`set_texture_binding`'s own
    /// check); with the array written at SET time the check moves here, same outcome.
    fn report_precomputed_texture_unit(&self, unit: u32, texture: u32) {
        tracing::warn!(
            target: "vitaslop::gxm",
            unit,
            texture = format_args!("{texture:#x}"),
            "precomputed state SetTexture on a unit beyond SCE_GXM_MAX_TEXTURE_UNITS - DROPPED"
        );
    }

    /// `sceGxmPrecomputedDrawSetAllVertexStreams(precomputedDraw, streamDataArray)`:
    /// the array holds one pointer per `SceGxmVertexStream` the draw's vertex program
    /// was created with, in stream order. The count comes from the program (a caller
    /// passes no length), so a draw whose vertex program we never saw sets nothing -
    /// and says so, because silently binding no streams renders empty geometry.
    pub fn precomputed_draw_set_all_streams(&mut self, ctx: &mut GuestCtx, precomputed: u32, array: u32) {
        let handle = ctx.read_u32(precomputed + pdraw::OFF_VERTEX_PROGRAM);
        let count = match self.vertex_programs.get(&handle) {
            Some(info) => info.streams.len(),
            None => {
                tracing::warn!(
                    target: "vitaslop::gxm",
                    precomputed = format_args!("{precomputed:#x}"),
                    vertex_program = format_args!("{handle:#x}"),
                    "precomputedDrawSetAllVertexStreams for an unknown vertex program - \
                     stream count unknown, NO streams bound"
                );
                0
            }
        };
        for i in 0..count as u32 {
            let data = ctx.read_u32(array.wrapping_add(i * 4));
            self.precomputed_draw_set_stream(ctx, precomputed, i, data);
        }
    }

    /// `sceGxmPrecomputed{Vertex,Fragment}StateSetAllTextures(state, textureArray)`.
    ///
    /// Note the array is of `SceGxmTexture` STRUCTS, not pointers: element `i` lives at
    /// `textureArray + i*16`, and that address is exactly what the per-index setter takes.
    /// The length is the program's texture-unit count, since the caller passes none.
    pub fn precomputed_vertex_state_set_all_textures(&mut self, ctx: &mut GuestCtx, state: u32, array: u32) {
        use crate::vita::gxmstate::{self, off};
        let header = if ctx.read_u32(state.wrapping_add(off::MAGIC)) == gxmstate::MAGIC_VERTEX {
            ctx.read_u32(state.wrapping_add(off::HEADER))
        } else {
            0
        };
        let n = self.reflect_program(ctx, header).texture_unit_count;
        for i in 0..n {
            self.precomputed_vertex_state_set_texture(ctx, state, i, array.wrapping_add(i * 16));
        }
    }

    pub fn precomputed_fragment_state_set_all_textures(&mut self, ctx: &mut GuestCtx, state: u32, array: u32) {
        use crate::vita::gxmstate::{self, off};
        let header = if ctx.read_u32(state.wrapping_add(off::MAGIC)) == gxmstate::MAGIC_FRAGMENT {
            ctx.read_u32(state.wrapping_add(off::HEADER))
        } else {
            0
        };
        let n = self.reflect_program(ctx, header).texture_unit_count;
        for i in 0..n {
            self.precomputed_fragment_state_set_texture(ctx, state, i, array.wrapping_add(i * 16));
        }
    }

    /// Apply `sceGxmSetPrecomputedVertexState(context, state)`: replace the context's
    /// non-default uniform-buffer table with the state's, wholesale, and bind the state's
    /// default uniform buffer record. A state that was never initialised (no magic) clears
    /// both, exactly as the old table's miss arm did.
    ///
    /// This is the HOST side of `InlineOp::BindPrecomputedState` - the fallback for a
    /// pointer or magic the emitted guard declines, and the definition the inline form is
    /// held to. It writes exactly the words the inline form writes.
    pub fn bind_precomputed_vertex_state(&mut self, ctx: &mut GuestCtx, state: u32) {
        use crate::vita::{gxmctx, gxmstate};
        let inited = state != 0
            && ctx.read_u32(state.wrapping_add(gxmstate::off::MAGIC)) == gxmstate::MAGIC_VERTEX;
        let (buf, size, header, block) = if inited {
            (
                ctx.read_u32(state.wrapping_add(gxmstate::off::BUF)),
                ctx.read_u32(state.wrapping_add(gxmstate::off::SIZE)),
                ctx.read_u32(state.wrapping_add(gxmstate::off::HEADER)),
                ctx.read_u32(state.wrapping_add(gxmstate::off::BLOCK)),
            )
        } else {
            (0, 0, 0, 0)
        };
        // The whole table, zeros included - the same replace-not-merge the fragment bind's
        // texture copy performs: a buffer bound by an earlier direct call must not survive
        // into a state that does not declare it.
        let context = self.gxm_context;
        for i in 0..gxmctx::MAX_UNIFORM_BUFFERS as u32 {
            let addr = if block != 0 { ctx.read_u32(block.wrapping_add(i * 4)) } else { 0 };
            gxmctx::set_vertex_uniform_buffer(ctx, context, i, addr);
        }
        self.bind_uniform_buffer(ctx, gxmctx::off::VERTEX_UNIFORM, buf, size, header);
    }

    /// Apply `sceGxmSetPrecomputedFragmentState(context, state)`: bind this stage's
    /// textures to the context sampler units, exactly as a sequence of
    /// `sceGxmSetFragmentTexture` calls would, so `record_draw` snapshots them.
    pub fn bind_precomputed_fragment_state(&mut self, ctx: &mut GuestCtx, state: u32) {
        use crate::vita::{gxmctx, gxmstate};
        // A state never initialised through our Init has no magic; the old table's miss
        // arm returned without touching anything, so this does too.
        if state == 0
            || ctx.read_u32(state.wrapping_add(gxmstate::off::MAGIC)) != gxmstate::MAGIC_FRAGMENT
        {
            return;
        }
        let context = self.gxm_context;
        if context == 0 {
            self.report_bind_without_context();
            return;
        }
        // The state's texture array lands over the context's, WHOLESALE - the array is
        // kept in the context block's own slot layout precisely so this is one copy.
        // Binding a precomputed state REPLACES the whole array on hardware; the block's
        // unset slots are zero, which is the context's own unbound encoding, so a unit
        // bound by an earlier direct call cannot survive into a state that does not
        // declare it.
        //
        // This is the HOST side of `InlineOp::BindPrecomputedState` - the fallback for a
        // pointer or magic the emitted guard declines - and it writes exactly the words
        // the inline form writes.
        let block = ctx.read_u32(state.wrapping_add(gxmstate::off::BLOCK));
        for w in 0..gxmstate::FRAGMENT_BLOCK_BYTES / 4 {
            let v = if block != 0 { ctx.read_u32(block.wrapping_add(w * 4)) } else { 0 };
            gxmctx::set(ctx, context, gxmctx::off::TEXTURES + w * 4, v);
        }
        // Binding a precomputed fragment state leaves the context bound to its program, so it
        // goes where `sceGxmSetFragmentProgram` puts it. The blend equation follows from the
        // handle and needs no separate record.
        let handle = ctx.read_u32(state.wrapping_add(gxmstate::off::HANDLE));
        self.set_gxm_state_word(ctx, gxmctx::off::FRAGMENT_PROGRAM, handle);
        // Bind this stage's default uniform buffer (pointer + the size memoised at Init) so
        // the draw reads the per-material fragment uniforms (tint / light / fog) from guest
        // memory, exactly as the precomputed vertex path binds the vertex uniform buffer.
        let buf = ctx.read_u32(state.wrapping_add(gxmstate::off::BUF));
        let size = ctx.read_u32(state.wrapping_add(gxmstate::off::SIZE));
        let header = ctx.read_u32(state.wrapping_add(gxmstate::off::HEADER));
        self.bind_uniform_buffer(ctx, gxmctx::off::FRAGMENT_UNIFORM, buf, size, header);
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
    /// Hand out `size` bytes of default-uniform space from a RECYCLED per-scene arena.
    ///
    /// A title reserves a default uniform buffer once per draw per stage - hundreds of
    /// times a frame, tens of thousands of times a second. Bump-allocating a fresh
    /// block for each (which is what this did until 2026-07-25) leaks the whole arena
    /// in minutes: the cursor marches up through guest memory until it reaches the
    /// region the MAIN THREAD'S STACK descends through, and then the guest's own
    /// `memcpy` of its uniforms into "its" buffer quietly overwrites live stack frames.
    /// The damage surfaces far away and much later - as a callee-saved register that
    /// comes back from a `POP` holding a float, and a crash in a function that did
    /// nothing wrong. On real hardware the default uniform buffer IS a ring the driver
    /// recycles, so recycling is also the faithful behaviour.
    ///
    /// The cursor resets per scene ([`Self::begin_scene`]), so within one frame every
    /// reserve still gets a distinct address exactly as before - only the growth across
    /// frames is gone.
    /// This is the FALLBACK half of the bump: the transpiler emits it inline for the
    /// ordinary case (`InlineOp::ReserveUniformBuffer`), so what reaches here is a context
    /// with no ring attached yet, a scene that has overrun the ring, or a bound program we
    /// did not create. The two paths compute the same address from the same three words,
    /// which is what lets a draw be unable to tell which of them ran.
    fn alloc_default_uniform_buffer(&mut self, ctx: &mut GuestCtx, size: u32) -> u32 {
        let size = size.max(crate::vita::gxmctx::UNIFORM_MIN_ALLOC);
        let context = self.gxm_context;
        if context == 0 {
            // Nothing to bump and nowhere to record. A draw in this state already reads the
            // GXM defaults for every piece of sticky state and says so, so this only has to
            // hand back memory the guest can write into.
            return self.galloc(size, crate::vita::gxmctx::UNIFORM_ALIGN);
        }
        let (mut base, mut end, cursor) = crate::vita::gxmctx::uniform_ring(ctx, context);
        if base == 0 {
            self.attach_uniform_ring(ctx, context);
            (base, end, _) = crate::vita::gxmctx::uniform_ring(ctx, context);
        }
        if base == 0 {
            // The arena could not supply a ring at all; fall back to a direct
            // allocation so the call still returns something usable.
            return self.galloc(size, crate::vita::gxmctx::UNIFORM_ALIGN);
        }
        let align = crate::vita::gxmctx::UNIFORM_ALIGN;
        let at = (cursor.max(base).wrapping_add(align - 1)) & !(align - 1);
        if at < base || at > end || size > end - at {
            // One scene wanted more than the ring holds. Wrapping aliases two live
            // buffers, which is a real (if rare) fidelity loss, so say so once rather
            // than silently returning overlapping memory.
            if !self.uniform_ring_wrapped {
                self.uniform_ring_wrapped = true;
                tracing::warn!(
                    target: "vitaslop::gxm",
                    ring = Self::UNIFORM_RING_BYTES,
                    wanted = size,
                    "default-uniform ring wrapped WITHIN one scene - buffers now alias; raise UNIFORM_RING_BYTES"
                );
            }
            crate::vita::gxmctx::set(
                ctx,
                context,
                crate::vita::gxmctx::off::UNIFORM_RING_CURSOR,
                base.wrapping_add(size),
            );
            return base;
        }
        crate::vita::gxmctx::set(
            ctx,
            context,
            crate::vita::gxmctx::off::UNIFORM_RING_CURSOR,
            at.wrapping_add(size),
        );
        at
    }

    /// Give `context` its default-uniform ring, once.
    ///
    /// Separate from the bump because only the HOST can allocate guest memory: the emitted
    /// reserve reads a ring that is already there and hands the call back when it is not,
    /// so this runs at most once per context (at `sceGxmCreateContext`, and again from the
    /// fallback if that allocation had failed).
    pub fn attach_uniform_ring(&mut self, ctx: &mut GuestCtx, context: u32) {
        if context == 0 || crate::vita::gxmctx::uniform_ring(ctx, context).0 != 0 {
            return;
        }
        let base = self.galloc(Self::UNIFORM_RING_BYTES, crate::vita::gxmctx::UNIFORM_ALIGN);
        if base == 0 {
            return;
        }
        crate::vita::gxmctx::set_uniform_ring(ctx, context, base, Self::UNIFORM_RING_BYTES);
    }

    /// Bytes of recycled default-uniform space. One scene of this title's main screen
    /// uses ~340 draws x 2 stages x <=4 KiB worst case; 8 MiB leaves generous headroom
    /// while staying a rounding error against the guest region.
    const UNIFORM_RING_BYTES: u32 = 8 * 1024 * 1024;

    pub fn reserve_vertex_uniform_buffer(&mut self, ctx: &mut GuestCtx) -> u32 {
        let handle = self.bound_vertex_program(ctx);
        let (size, header) = self.memoised_uniform_size(ctx, handle, ProgramStage::Vertex);
        let buf = self.alloc_default_uniform_buffer(ctx, size);
        poison_uniform_buffer(ctx, buf, size);
        self.bind_uniform_buffer(ctx, crate::vita::gxmctx::off::VERTEX_UNIFORM, buf, size, header);
        tracing::trace!(
            target: "vitaslop::gxm",
            program = format_args!("{handle:#x}"),
            header = format_args!("{header:#x}"),
            size,
            buffer = format_args!("{buf:#x}"),
            "reserveVertexDefaultUniformBuffer"
        );
        buf
    }

    /// The default-uniform size a program handle carries, and the `SceGxmProgram *` it was
    /// created from.
    ///
    /// **The handle's own words are the definition**, not a cache of one: they are stamped
    /// at create ([`crate::vita::gxmprog`]), the emitted inline reserve reads exactly them,
    /// and this reads them too, so there is no arrangement in which the two paths size a
    /// buffer differently. A handle that predates that stamp - or one from a title that
    /// somehow passes a pointer we never created - falls back to reflecting the program
    /// here, which is what this call always did.
    fn memoised_uniform_size(
        &mut self,
        ctx: &GuestCtx,
        handle: u32,
        stage: ProgramStage,
    ) -> (u32, u32) {
        match crate::vita::gxmprog::program(ctx, handle) {
            Some((size, _alloc, header)) => (size, header),
            None => {
                let header = match stage {
                    ProgramStage::Vertex => self.vertex_program_header(handle),
                    ProgramStage::Fragment => self.fragment_program_header(handle),
                };
                (self.uniform_size_bytes(ctx, header), header)
            }
        }
    }

    /// How many bytes of default uniform buffer a program declares: the greater of its
    /// REFLECTED extent and the header's own count field, clamped so an unresolved header
    /// cannot ask for an absurd allocation.
    ///
    /// The one definition of that number. [`crate::vita::gxmprog::init`] memoises it into
    /// the handle at create time and the reserve reads it back from there, so this runs
    /// once per program rather than once per draw - but it is the same arithmetic either
    /// way, which is the point of having it in one place.
    pub fn uniform_size_bytes(&mut self, ctx: &GuestCtx, header: u32) -> u32 {
        self.reflected_uniform_size_bytes(ctx, header)
            .max(default_uniform_buffer_bytes(ctx, header))
            .min(MAX_DEFAULT_UNIFORM_REGS)
    }

    /// Record what a stage's default uniform buffer is, where the draw reads it from.
    fn bind_uniform_buffer(
        &mut self,
        ctx: &mut GuestCtx,
        record: u32,
        buf: u32,
        size: u32,
        header: u32,
    ) {
        if self.gxm_context == 0 {
            if !self.reported_reserve_without_context {
                self.reported_reserve_without_context = true;
                tracing::warn!(
                    target: "vitaslop::gxm",
                    "a default uniform buffer was reserved before sceGxmCreateContext - there \
                     is no context block to record it in, so the draws that follow read their \
                     uniforms from the sceGxmSetUniformDataF capture instead"
                );
            }
            return;
        }
        crate::vita::gxmctx::set_uniform_binding(
            ctx,
            self.gxm_context,
            record,
            crate::vita::gxmctx::UniformBinding { buf, size, header },
        );
    }

    /// The vertex stage's bound default uniform buffer, as the context block holds it.
    fn vertex_uniform(&self, ctx: &GuestCtx) -> crate::vita::gxmctx::UniformBinding {
        self.uniform_binding(ctx, crate::vita::gxmctx::off::VERTEX_UNIFORM)
    }

    /// The fragment stage's.
    fn fragment_uniform(&self, ctx: &GuestCtx) -> crate::vita::gxmctx::UniformBinding {
        self.uniform_binding(ctx, crate::vita::gxmctx::off::FRAGMENT_UNIFORM)
    }

    fn uniform_binding(&self, ctx: &GuestCtx, record: u32) -> crate::vita::gxmctx::UniformBinding {
        match self.gxm_context {
            0 => crate::vita::gxmctx::UniformBinding::default(),
            c => crate::vita::gxmctx::uniform_binding(ctx, c, record),
        }
    }

    /// `sceGxmReserveFragmentDefaultUniformBuffer`: the fragment-stage counterpart of
    /// [`Self::reserve_vertex_uniform_buffer`]. Hand back a guest buffer sized to the bound
    /// fragment program's default uniform block and bind it as the fragment uniform source,
    /// so a title that writes its per-material uniforms (tint / light / fog) directly into
    /// this buffer has them captured into the draw's material.
    pub fn reserve_fragment_uniform_buffer(&mut self, ctx: &mut GuestCtx) -> u32 {
        let handle = self.bound_fragment_program(ctx);
        let (size, header) = self.memoised_uniform_size(ctx, handle, ProgramStage::Fragment);
        let buf = self.alloc_default_uniform_buffer(ctx, size);
        self.bind_uniform_buffer(ctx, crate::vita::gxmctx::off::FRAGMENT_UNIFORM, buf, size, header);
        buf
    }

    /// Release a memory block by SceUID (`sceKernelFreeMemBlock`). Returns true if a
    /// block was registered under `uid`. The deterministic bump allocation itself is
    /// not reclaimed (the arena only grows), but the registry entry is removed so a
    /// later `sceKernelGetMemBlockBase(uid)` no longer resolves it, matching the guest-
    /// visible contract that the id is now invalid.
    pub fn free_memblock(&mut self, uid: i32) -> bool {
        let Some(i) = self.memblocks.iter().position(|b| b.uid == uid) else { return false };
        let b = self.memblocks.remove(i);
        // Drop any texture snapshot over the released memory. Now that the block's
        // address CAN come back from [`Self::alloc_memblock`], this invalidation is
        // load-bearing rather than merely tidy: a stale snapshot over reused memory
        // would render the previous screen's pixels into the next one's texture.
        self.texture_snapshots.invalidate_range(b.base, b.size as usize);
        if b.base != 0 {
            self.freed_memblocks.push((b.base, b.size.max(4)));
        }
        true
    }

    /// Record a `sceGxmSetUniformDataF` write into the fallback SA bank - the one a draw uses
    /// when no default uniform buffer is bound.
    ///
    /// `at` is the register the write STARTS at: the parameter's own `resource_index` plus the
    /// call's `componentOffset`. Placing the values there rather than at the top is the same
    /// fact the buffer copy in `vita::gxm::set_uniform_data_f` rests on, and the two must agree
    /// - a program that sets two uniforms would otherwise have the second land on the first here
    /// while landing correctly in the buffer, so which of the two a draw read would depend on
    /// whether the title happened to reserve a buffer.
    ///
    /// Writes ACCUMULATE within a scene (the bank is cleared in `begin_scene`), because that is
    /// what the guest's own buffer does: two calls setting different uniforms leave both set.
    /// This is the FALLBACK half: the transpiler emits the same two writes inline
    /// (`InlineOp::SetUniformData`), so what reaches here is an F16 parameter, a record we
    /// could not read, or a write past the ceiling.
    pub fn set_uniforms(&mut self, ctx: &mut GuestCtx, at: u32, values: &[f32]) {
        let end = at as usize + values.len();
        // Same ceiling the reserved buffers are clamped to: a register offset past it is a
        // record we misread, not a uniform, and growing the bank to match it would turn a bad
        // read into an allocation.
        if end > MAX_DEFAULT_UNIFORM_REGS as usize {
            return;
        }
        let bank = self.ensure_sa_bank();
        if bank == 0 {
            return;
        }
        for (i, v) in values.iter().enumerate() {
            ctx.write_u32(bank + SA_BANK_DATA + (at as usize + i) as u32 * 4, v.to_bits());
        }
        // The gap between the old high-water mark and `at` is already zero - `clear_sa_bank`
        // left it that way for this scene - which is what the `Vec`'s `resize(end, 0.0)` used
        // to provide.
        if (ctx.read_u32(bank) as usize) < end {
            ctx.write_u32(bank, end as u32);
        }
    }

    /// The half-precision counterpart of [`Self::set_uniforms`]: record a
    /// `sceGxmSetUniformDataF` write to an F16-declared uniform, which packs TWO components
    /// per register.
    ///
    /// `at_half` counts HALVES from the start of the buffer, so register `n` holds halves
    /// `2n` and `2n+1`. The bank stores raw register words (a lane's f32 bit pattern IS the
    /// register), so a partial write read-modify-writes the neighbouring half rather than
    /// clearing it - the same thing the guest's own buffer does, and the same thing
    /// `vita::gxm::set_uniform_data_f` does to the reserved buffer. The two must agree, or
    /// which one a draw reads decides what it renders.
    pub fn set_uniform_halves(&mut self, ctx: &mut GuestCtx, at_half: u32, values: &[f32]) {
        let end_reg = (at_half as usize + values.len()).div_ceil(2);
        if end_reg > MAX_DEFAULT_UNIFORM_REGS as usize {
            return;
        }
        let bank = self.ensure_sa_bank();
        if bank == 0 {
            return;
        }
        for (i, v) in values.iter().enumerate() {
            let component = at_half as usize + i;
            let addr = bank + SA_BANK_DATA + (component / 2) as u32 * 4;
            let word = ctx.read_u32(addr);
            let h = u32::from(crate::render::f32_to_half(*v));
            ctx.write_u32(
                addr,
                if component % 2 == 0 {
                    (word & 0xffff_0000) | h
                } else {
                    (word & 0x0000_ffff) | (h << 16)
                },
            );
        }
        if (ctx.read_u32(bank) as usize) < end_reg {
            ctx.write_u32(bank, end_reg as u32);
        }
    }

    /// What thread `thid` is parked on, as one line, or `RUNNABLE` if it appears in no
    /// waiter list at all. "Runnable" is the interesting answer as often as the blocked
    /// ones are: a title that renders but never progresses has one thread spinning on a
    /// flag in pure guest compute, and that thread is exactly the one nothing here
    /// mentions.
    pub fn thread_wait_state(&self, thid: i32) -> String {
        for m in &self.lwmutexes {
            if m.waiters.contains(&thid) {
                // No owner here on purpose: it lives in the guest's work area now, and
                // this dump has no guest memory. Naming the address is enough to go and
                // read it (`VITASLOP_PEEK=<work>:16`), and inventing a host-side echo of
                // it would print a value the inline take never updated - a wrong answer
                // in the one report that exists to explain a deadlock.
                return format!(
                    "blocked on lwmutex work={:#010x} ({} parked; owner is in the work area)",
                    m.work,
                    m.waiters.len()
                );
            }
        }
        for m in &self.mutexes {
            if m.waiters.contains(&thid) {
                return format!("blocked on mutex uid={:#x} (owner={:?})", m.uid, m.owner);
            }
        }
        if let Some(w) = self.sema_waiters.iter().find(|w| w.thid == thid) {
            return format!("blocked on sema uid={:#x} need={}", w.uid, w.need);
        }
        if let Some(w) = self.evf_waiters.iter().find(|w| w.thid == thid) {
            let pattern = self.event_flags.iter().find(|(u, _)| *u == w.uid).map(|(_, p)| *p).unwrap_or(0);
            return format!(
                "blocked on eventflag uid={:#x}{} want={:#x} mode={:#x} have={:#x}{}",
                w.uid,
                match self.event_flag_name(w.uid) {
                    "" => String::new(),
                    n => format!(" {n:?}"),
                },
                w.bits,
                w.mode,
                pattern,
                match w.deadline {
                    Some(d) => format!(" deadline_us={d}"),
                    None => String::new(),
                }
            );
        }
        for c in &self.conds {
            if c.waiters.iter().any(|w| w.thid == thid) {
                return format!("blocked on cond uid={:#x} (mutex uid={:#x})", c.uid, c.mutex);
            }
        }
        if let Some((_, work, deadline)) = self.lwcond_waiters.iter().find(|(t, _, _)| *t == thid) {
            return format!("blocked on lwcond work={work:#010x} deadline={deadline:?}");
        }
        if let Some((_, deadline)) = self.io_waiters.iter().find(|(t, _)| *t == thid) {
            return format!("awaiting storage io_us={deadline} (now {})", self.io_us);
        }
        if let Some((_, deadline)) = self.sleep_waiters.iter().find(|(t, _)| *t == thid) {
            return format!("sleeping until_us={deadline} (now {})", self.virtual_us);
        }
        if let Some((_, target, _)) = self.join_waiters.iter().find(|(t, _, _)| *t == thid) {
            return format!("joining thid={target:#x}");
        }
        "RUNNABLE".to_string()
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
        // The threads themselves come first: the primitive lists below say who is
        // waiting on each object, but only this says, for every live thread, whether
        // it is parked at all - and a thread absent from every waiter list is the
        // one actually running. A stall dump without it forces the reader to
        // cross-reference by hand and silently omits any thread blocked on a
        // primitive whose list is not printed.
        let _ = writeln!(
            s,
            "threads ({} live):",
            1 + self.threads.iter().filter(|t| t.exit_code.is_none()).count()
        );
        // The MAIN thread is listed first and by hand, because it is the one thread
        // that has no `ThreadRec`: `create_thread` records the threads the GUEST
        // creates, and the initial thread was created by the loader before any of
        // that. Omitting it made this dump read as "the game logic thread is gone"
        // for a title whose main thread was in fact alive and spinning - a wrong
        // diagnosis that a stall dump exists precisely to prevent. Its thid is 0.
        let _ = writeln!(
            s,
            "  thid={:#x} \"main\" (the initial thread - no create record) {}",
            MAIN_THID,
            self.thread_wait_state(MAIN_THID),
        );
        for t in &self.threads {
            if t.exit_code.is_some() {
                continue;
            }
            let name = if t.name.is_empty() { "-".to_string() } else { format!("{:?}", t.name) };
            let _ = writeln!(
                s,
                "  thid={:#x} {name} entry={:#010x} prio={:#x} {}{}",
                t.uid,
                t.entry,
                t.priority,
                if t.started { "" } else { "DORMANT " },
                self.thread_wait_state(t.uid),
            );
        }
        let _ = writeln!(s, "lwmutexes ({}):", self.lwmutexes.len());
        for m in &self.lwmutexes {
            // Owner and count are guest-resident (`vita::lwwork`) and this dump has no
            // guest memory; `VITASLOP_PEEK=<work>:16` reads them, in that order.
            let _ = writeln!(s, "  work={:#010x} parked={:x?}", m.work, m.waiters);
        }
        let _ = writeln!(s, "mutexes ({}):", self.mutexes.len());
        for m in &self.mutexes {
            let _ = writeln!(
                s, "  uid={:#x} owner={:?} count={} waiters={:x?}",
                m.uid, m.owner, m.count, m.waiters
            );
        }
        let _ = writeln!(s, "semaphores ({}):", self.semaphores.len());
        for sem in &self.semaphores {
            let waiters: Vec<(i32, i32)> =
                self.sema_waiters.iter().filter(|w| w.uid == sem.uid).map(|w| (w.thid, w.need)).collect();
            let _ = writeln!(
                s,
                "  uid={:#x} name={:?} count={} waiters(thid,need)={waiters:x?}",
                sem.uid, sem.name, sem.count
            );
        }
        // Event flags were missing from this dump entirely, and they are what a title's
        // own worker threads actually park on - so a stall could show every printed
        // list empty while several threads were blocked.
        let _ = writeln!(s, "event flags ({}):", self.event_flags.len());
        for (uid, pattern) in &self.event_flags {
            let waiters: Vec<(i32, u32)> =
                self.evf_waiters.iter().filter(|w| w.uid == *uid).map(|w| (w.thid, w.bits)).collect();
            if !waiters.is_empty() {
                let _ = writeln!(s, "  uid={uid:#x} pattern={pattern:#x} waiters(thid,want)={waiters:x?}");
            }
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
    fn bound_layout(&self, ctx: &GuestCtx) -> Option<&VertexProgramInfo> {
        self.vertex_programs.get(&self.bound_vertex_program(ctx))
    }

    /// 1, 10, 100, ... - the schedule a repeating diagnostic reports on when what matters is
    /// its ORDER OF MAGNITUDE and printing every occurrence would drown the log.
    fn is_power_of_ten_impl(mut n: u64) -> bool {
        if n == 0 {
            return false;
        }
        while n % 10 == 0 {
            n /= 10;
        }
        n == 1
    }

    /// Is the bound default uniform buffer for `stage` left over from a DIFFERENT program
    /// than the one about to draw?
    ///
    /// A default uniform buffer is one program's SA bank: its size and its every offset
    /// come from that program's parameter table, and GXM invalidates the reservation when
    /// the program changes (the app reserves again after each `sceGxmSetVertexProgram`).
    /// A draw that arrives with the previous program's buffer still bound therefore reads
    /// another object's uniforms through this program's layout. When lane 0 is a model
    /// matrix - which it is for every 3D vertex program in this title - the result is a
    /// mesh rendered at some other object's position, following it around the world.
    ///
    /// Reported unconditionally (once per program pair): a draw silently transformed by
    /// the wrong matrix is indistinguishable from a faithful render except by eye, and
    /// that is exactly how this class of bug survives.
    fn stale_uniforms(
        &mut self,
        stage: &'static str,
        bound_for: u32,
        drawing: u32,
        needs_bytes: u32,
    ) -> bool {
        // `drawing == 0` means the program header was never recorded (an unknown handle),
        // so there is nothing to compare against and no evidence of staleness.
        if bound_for == drawing || drawing == 0 {
            return false;
        }
        // A program with NO default uniform block reads no SA bank, so a buffer left bound
        // for a different program is not starving it of anything - there is nothing for it to
        // be starved of. Dropping the binding is still right (it is not this draw's data), but
        // it is bookkeeping, not a defect, and reporting it as one is what buried three real
        // render defects under seventy lines of noise on the race screen.
        //
        // This is the common case by construction: the direct path reserves per program, and a
        // title only omits the reserve for a program that needs no uniforms. That is why the
        // counts are in the thousands on a title whose picture is broadly correct - the warning
        // was measuring how OFTEN the guest binds a uniform-less fragment program after a
        // uniform-taking one, which is a fact about the title, not about the emulator.
        if needs_bytes == 0 {
            return true;
        }
        let n = self.reported_stale_uniforms.entry((stage, bound_for, drawing)).or_insert(0);
        *n += 1;
        // On powers of ten, so the log carries the SCALE without carrying a line per draw.
        // Four dropped draws in a run is a curiosity; four hundred thousand is a stage
        // rendering with a zeroed uniform bank, which comes out black and looks like a
        // shading bug anywhere else in the pipeline.
        if Self::is_power_of_ten_impl(*n) {
            tracing::warn!(
                target: "vitaslop::gxm",
                stage,
                bound_for = format_args!("{bound_for:#x}"),
                drawing = format_args!("{drawing:#x}"),
                needs_bytes,
                count = *n,
                "STALE default uniform buffer: bound for another program, so this draw's stage \
                 renders with NO uniform bank at all (it declares needs_bytes of them)"
            );
        }
        true
    }

    /// The vertex uniforms in effect for the next draw. On the precomputed path
    /// (`bound_vertex_uniform_buf` set by `sceGxmSetPrecomputedVertexState`) read the
    /// default uniform buffer straight from guest memory, sized by the program's default
    /// uniform buffer size. Otherwise use the `sceGxmSetUniformDataF` capture.
    ///
    /// A buffer bound for a different vertex program is NOT this draw's SA bank - see
    /// [`Self::stale_uniforms`] - so it is dropped rather than misread.
    /// WHERE this draw's vertex uniforms live: `Some((address, byte length))` for a bound
    /// default uniform buffer, `None` for "the `sceGxmSetUniformDataF` SA bank".
    ///
    /// Split out from the two readers below because the DECISION - which includes the
    /// staleness check, which reports - must be made once and identically however the bytes
    /// are then taken. `record_draw` needs the same bank as FLOATS (for the transform
    /// reflection) and as BYTES (the recompiler's `vert_sa`), and it used to read floats and
    /// then serialise them back to bytes: a third buffer per draw for data it had already had
    /// in that form. One decision, two readers, no round trip.
    fn current_vertex_uniform_src(
        &mut self,
        ctx: &GuestCtx,
        blk: &crate::vita::gxmctx::Block<'_>,
    ) -> Option<(u32, usize)> {
        let bound = blk.uniform_binding(crate::vita::gxmctx::off::VERTEX_UNIFORM);
        if bound.buf != 0 {
            let drawing =
                self.vertex_program_header(blk.word(crate::vita::gxmctx::off::VERTEX_PROGRAM));
            let refl = self.reflect_program(ctx, drawing);
            let needs = refl.uniform_size_bytes.max(refl.default_uniform_bytes);
            if self.stale_uniforms("vertex", bound.header, drawing, needs) {
                return None;
            }
        }
        (bound.size >= 4 && bound.buf != 0).then(|| (bound.buf, (bound.size / 4) as usize * 4))
    }

    /// This draw's vertex uniforms as BYTES, exactly as the guest wrote them - the form the
    /// GXP recompiler wants. See [`Self::current_vertex_uniform_src`].
    fn current_vertex_uniform_bytes(
        &mut self,
        ctx: &GuestCtx,
        blk: &crate::vita::gxmctx::Block<'_>,
    ) -> Vec<u8> {
        match self.current_vertex_uniform_src(ctx, blk) {
            Some((addr, len)) => ctx.read_bytes(addr, len),
            None => self.sa_bank_bytes(ctx),
        }
    }

    #[allow(dead_code)]
    fn current_vertex_uniforms(
        &mut self,
        ctx: &GuestCtx,
        blk: &crate::vita::gxmctx::Block<'_>,
    ) -> Vec<f32> {
        // Only a BOUND buffer can be stale, so the test belongs behind the same guard the
        // fragment stage uses in `record_draw`. With nothing bound - `vertex_uniform_header` is
        // 0 after a null or never-built precomputed state, and after a cleared reservation -
        // both arms below return the SA bank regardless, so testing staleness here
        // cannot change a pixel and can only report a drop that did not happen. It did:
        // 10,000 `bound_for=0x0` warnings in one retail run, which reads as a rendering
        // defect when the direct uniform path was serving those draws correctly.
        let bound = blk.uniform_binding(crate::vita::gxmctx::off::VERTEX_UNIFORM);
        if bound.buf != 0 {
            let drawing =
                self.vertex_program_header(blk.word(crate::vita::gxmctx::off::VERTEX_PROGRAM));
            // Both halves come out of the ONE cached reflection: the header's own size field
            // is memoised beside the reflected extent (see
            // `ProgramReflection::default_uniform_bytes`), so this is a map lookup rather
            // than a map lookup AND a guest word read on every draw.
            let refl = self.reflect_program(ctx, drawing);
            let needs = refl.uniform_size_bytes.max(refl.default_uniform_bytes);
            if self.stale_uniforms("vertex", bound.header, drawing, needs) {
                return self.sa_bank_floats(ctx);
            }
        }
        if bound.size >= 4 && bound.buf != 0 {
            let count = (bound.size / 4) as usize;
            ctx.read_f32s(bound.buf, count)
        } else {
            self.sa_bank_floats(ctx)
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
        let _all = crate::perf::scope(crate::perf::Phase::DrawTotal);
        // A draw with no context block behind it reads the GXM DEFAULTS for every piece of
        // sticky state - no bound program, no streams, cull none, depth less-equal - which
        // renders as missing geometry rather than as an error. Nothing else would report it,
        // because reading a default is not a failure anywhere downstream.
        if self.gxm_context == 0 && !self.reported_no_gxm_context {
            self.reported_no_gxm_context = true;
            tracing::warn!(
                target: "vitaslop::gxm",
                "a draw was recorded before sceGxmCreateContext - every piece of sticky GXM \
                 state reads as its DEFAULT, so this draw has no bound program and no vertex \
                 streams"
            );
        }
        // >>> ONE SNAPSHOT OF THE CONTEXT BLOCK, FOR EVERY READER BELOW. See
        // [`crate::vita::gxmctx::Block`]: a draw read that one 652-byte structure through a
        // dozen separate crossings into guest memory, and the guest cannot run during a host
        // call, so one copy answers all of them identically.
        let blk = {
            let _b = crate::perf::scope(crate::perf::Phase::DrawBlockRead);
            crate::vita::gxmctx::Block::read(ctx, self.gxm_context)
        };
        // Occlusion query: this draw contributes to whatever slot the front-face
        // visibility test currently names (no-op when no query is live).
        self.accumulate_visibility(&blk, index_count);
        // Drop a default uniform buffer still bound for a DIFFERENT program before anything
        // below reads it (see [`Self::stale_uniforms`]). The vertex stage is handled inside
        // `current_vertex_uniforms`; the fragment stage has several independent readers
        // (material reflection, the recompiler's `frag_sa`, the draw dump), so clear it once
        // here and they all see the same truth.
        let reflect_phase = crate::perf::scope(crate::perf::Phase::DrawReflect);
        let mut frag_uniform = blk.uniform_binding(crate::vita::gxmctx::off::FRAGMENT_UNIFORM);
        let fheader = self.fragment_program_header(blk.word(crate::vita::gxmctx::off::FRAGMENT_PROGRAM));
        if frag_uniform.buf != 0 {
            let drawing = fheader;
            // How much SA bank the program ABOUT TO DRAW actually declares. Zero means it
            // reads none, and then a mismatched binding starves it of nothing - see
            // `stale_uniforms`.
            // Both halves come out of the ONE cached reflection: the header's own size field
            // is memoised beside the reflected extent (see
            // `ProgramReflection::default_uniform_bytes`), so this is a map lookup rather
            // than a map lookup AND a guest word read on every draw.
            let refl = self.reflect_program(ctx, drawing);
            let needs = refl.uniform_size_bytes.max(refl.default_uniform_bytes);
            if self.stale_uniforms("fragment", frag_uniform.header, drawing, needs) {
                frag_uniform.buf = 0;
                frag_uniform.size = 0;
            }
        }
        // Reflect both bound programs ONCE, here. Every name-based lookup below reads
        // these instead of re-walking a parameter table (see `ProgramReflection`).
        let vhandle = blk.word(crate::vita::gxmctx::off::VERTEX_PROGRAM);
        let vheader = self.vertex_program_header(vhandle);
        let vref = self.reflect_program(ctx, vheader);
        let fref = self.reflect_program(ctx, fheader);
        // The bound program's attributes, and the layout of each stream they name. The
        // attributes are rewritten below onto the single interleaved buffer this draw is
        // captured into, so what goes into the `Draw` is the REPACKED layout, not the
        // guest's.
        // >>> ALL OF IT PRECOMPUTED, at the moment the vertex program was created. See
        // `VertexProgramInfo::packed_attributes`: the packed stride, the per-stream bases, the
        // single-stream test and the rebased attribute list are constants of the PROGRAM, and
        // building them per draw was four allocations and two loops on the hottest path here.
        let (attributes, streams, base, stride, single_stream) = match self.vertex_programs.get(&vhandle) {
            Some(info) => (
                info.packed_attributes.clone(),
                info.used_streams.clone(),
                info.stream_base.clone(),
                info.packed_stride,
                info.single_stream,
            ),
            None => (
                crate::capture::no_attributes(),
                std::sync::Arc::from(&[][..]),
                std::sync::Arc::from(&[][..]),
                0,
                false,
            ),
        };
        drop(reflect_phase);
        // Index element size: U16 (0) is 2 bytes, U32 is 4.
        let index_elem = if index_format == 0 { 2 } else { 4 };
        // Snapshot exactly the vertices this draw REFERENCES, not the whole prefix of the
        // stream. A chunked world mesh draws a few hundred vertices out of a shared buffer of
        // tens of thousands, so copying `0..=max_index` per draw costs hundreds of megabytes a
        // frame (and reads far past what the draw can touch). Take the `min..=max` window and
        // rebase the indices onto it, which leaves every consumer's indexing unchanged.
        //
        // Read, scanned, rebased and shared in one call - and skipped entirely for a buffer the
        // guest provably has not written since. See `get_or_read_indices`.
        let (indices, first_vertex, vertex_count) = self.texture_snapshots.get_or_read_indices(
            ctx,
            index_addr,
            index_count as usize * index_elem,
            index_elem,
        );
        // Interleave every stream this draw's attributes name into ONE buffer, and rewrite
        // the attributes onto it. A vertex here is the concatenation of its row from each
        // used stream, so the result is a plain single-stream mesh that indexes exactly as
        // the guest's did - every consumer (software raster, fixed-function GPU path, the
        // recompiler's repack) keeps working unchanged and now sees the data it was
        // previously decoding out of stream 0 by mistake.
        //
        // `instance` is 0: only the first instance of an instanced draw is captured (see
        // `sceGxmDrawInstanced`), so a per-instance stream contributes its row 0 to every
        // vertex, which is what instance 0 reads.
        let bound_streams = blk.streams();
        let snapshots = &mut self.texture_snapshots;
        let vertices: Arc<[u8]> = crate::perf::time(crate::perf::Phase::DrawVertices, || {
            if single_stream {
                // The overwhelmingly common case is one per-vertex stream, whose rows are
                // already contiguous: take them in one read rather than one per vertex (this
                // path runs for every draw of every frame, over meshes of thousands of
                // vertices).
                //
                // ...and better than one read: a SNAPSHOT, so a mesh the guest has not touched
                // since the last draw is not read at all. See `get_or_read_vertices`.
                return snapshots.get_or_read_vertices(
                    ctx,
                    bound_streams[0].wrapping_add(first_vertex * stride),
                    (vertex_count * stride) as usize,
                );
            }
            let mut vertices = vec![0u8; (vertex_count * stride) as usize];
            for (si, s) in streams.iter().enumerate() {
                if s.stride == 0 {
                    continue;
                }
                let buf = bound_streams.get(si).copied().unwrap_or(0);
                // A per-instance stream is stepped by instance, not by vertex, so instance 0
                // reads row 0 for every vertex; a per-vertex stream's rows are contiguous.
                // Either way this is ONE guest read, then a scatter into the interleaved
                // buffer.
                let (src, repeat) = if s.per_instance {
                    (ctx.read_bytes(buf, s.stride as usize), true)
                } else {
                    let start = buf.wrapping_add(first_vertex * s.stride);
                    (ctx.read_bytes(start, (vertex_count * s.stride) as usize), false)
                };
                let row_len = s.stride as usize;
                for v in 0..vertex_count as usize {
                    let from = if repeat { 0 } else { v * row_len };
                    let Some(row) = src.get(from..from + row_len) else {
                        break; // the guest buffer ended short of the indices; keep what we have
                    };
                    let dst = v * stride as usize + base[si] as usize;
                    vertices[dst..dst + row_len].copy_from_slice(row);
                }
            }
            // The multi-stream path is NOT snapshotted: its result is an interleave of several
            // guest buffers, so a snapshot key would have to fold every stream's address, length
            // and stride, and its validity would rest on all of them at once. It is also the
            // uncommon case. Left as the plain read it always was, rather than given a cache
            // whose invalidation is harder to argue than the copy it saves.
            crate::perf::note_bytes(crate::perf::Phase::DrawVertices, vertices.len());
            Arc::from(vertices)
        });
        // Snapshot every bound fragment texture (decoded from its control words),
        // sorted by unit so unit 0 is first. `bound_textures` is already kept sorted by
        // unit as it is bound, so this reads it in place rather than cloning and
        // re-sorting a fresh Vec for every draw.
        // >>> THE GATE, TIMED SEPARATELY FROM WHAT IT GATES.
        //
        // Everything down to the `snapshot_sets` lookup is paid by EVERY draw, hit or miss:
        // the sampler block comes out of the guest's context, decodes into bindings, is
        // hashed, and the hash is looked up. Only what follows is the miss path. Keeping
        // them in one phase was hiding which of the two a fix would have to land on, and
        // the two fixes have nothing in common.
        let bind_phase = crate::perf::scope(crate::perf::Phase::DrawTextureBind);
        // >>> THE PREVIOUS DRAW'S ANSWER, IF THIS DRAW BINDS THE SAME BYTES.
        //
        // Everything below - the sixteen-slot decode, the fold, the map probe and the exact
        // verify - re-derives a list that a batch of draws sharing a material produces
        // identically every time. One `memcmp` of the sampler block answers it instead. See
        // `TextureSnapshots::last_set` for why a hit is exact rather than a guess, and why
        // it is refused across a scene boundary.
        let sampler_span = blk.span(
            crate::vita::gxmctx::off::TEXTURES,
            crate::vita::gxmctx::MAX_TEXTURE_UNITS * crate::vita::gxmctx::TEXTURE_STRIDE as usize,
        );
        let previous_set = self.texture_snapshots.set_from_previous_draw(sampler_span, fheader);
        if previous_set.is_some() {
            crate::perf::note_hit(crate::perf::Phase::DrawTexSetPrev);
        }
        // The fragment bindings come from the CONTEXT BLOCK now, because that is where the
        // inlined `sceGxmSetFragmentTexture` writes them and the host no longer sees most
        // binds at all. Read in unit order, which is the order this list has always been in.
        //
        // `bound_textures` is reused as the scratch buffer rather than allocated here: it is
        // taken and put back for the same reason it always was - the decode needs `self`
        // borrowed mutably while the bindings are borrowed - and keeping the allocation
        // avoids a Vec per draw on the hottest path in the engine.
        let mut frag_binds = std::mem::take(&mut self.bound_textures);
        // On a previous-draw hit `frag_binds` is left EXACTLY as the last draw left it, which
        // is the same list these identical bytes decode to - so the decode, the fold, the map
        // probe and the exact verify are all skipped and nothing downstream can tell.
        let (set_key, cached_set) = match &previous_set {
            Some(list) => (0u64, Some(list.clone())),
            None => {
                frag_binds.clear();
                let decode_phase = crate::perf::scope(crate::perf::Phase::DrawTexBindDecode);
                let context = self.gxm_context;
                if context != 0 {
                    // ONE borrow of the whole sampler block, not forty `read_u32`s - each of
                    // those is a virtual call through `dyn GuestMemory`, and on this path (627
                    // draws a frame) the calls were the cost rather than the bytes. See
                    // `gxmctx::texture_bindings`. The scratch list is reused for the same
                    // reason `bound_textures` is.
                    let mut raw = std::mem::take(&mut self.bound_binding_scratch);
                    blk.texture_bindings(&mut raw);
                    for (unit, b) in raw.iter() {
                        frag_binds.push(TextureBinding {
                            unit: *unit,
                            addr: b.addr,
                            words: b.words,
                            from_precomputed: b.from_precomputed,
                        });
                    }
                    self.bound_binding_scratch = raw;
                }
                drop(decode_phase);
                // >>> THE WHOLE LIST IS SHARED ACROSS SCENES, RE-PROVEN PER SCENE. See
                // `TextureSnapshots::snapshot_sets`.
                //
                // The key is what the list is a function of: the bindings, and the fragment
                // program header, which decides the albedo reorder below. The reorder is
                // applied BEFORE the entry is stored, so a hit is the finished list.
                let set_key = {
                    let _f = crate::perf::scope(crate::perf::Phase::DrawTexBindFold);
                    let mut h: u64 = 0xcbf2_9ce4_8422_2325;
                    let mut mix = |v: u64| {
                        h ^= v;
                        h = h.wrapping_mul(0x0000_0100_0000_01b3);
                    };
                    mix(fheader as u64);
                    for b in &frag_binds {
                        mix(b.unit as u64);
                        for w in b.words {
                            mix(w as u64);
                        }
                    }
                    h
                };
                let _p = crate::perf::scope(crate::perf::Phase::DrawTexBindProbe);
                let cached = self.texture_snapshots.set_validated(
                    ctx,
                    set_key,
                    fheader as u64,
                    &frag_binds,
                );
                (set_key, cached)
            }
        };
        drop(bind_phase);
        let texture_phase = crate::perf::scope(crate::perf::Phase::DrawTextures);
        let mut textures = match cached_set {
            Some(_) => Vec::new(),
            None => {
                let _m = crate::perf::scope(crate::perf::Phase::DrawTexFragMiss);
                self.snapshot_bound_textures(ctx, &frag_binds)
            }
        };
        self.bound_textures = frag_binds;
        // The VERTEX stage's own samplers, decoded exactly the same way. A vertex program that
        // fetches a texture builds its geometry from it - one retail title draws its whole
        // campaign map that way - so a draw that carries none of these renders as if the fetch
        // returned nothing, which is a blank screen rather than a visible error.
        //
        // >>> SHARED WITHIN A SCENE THE SAME WAY THE FRAGMENT LIST IS, and it was not.
        // The fragment stage has had `snapshot_sets` for a while; this side re-decoded its
        // units on EVERY draw, and a decode is where the texture-data snapshot and its compare
        // live. On a title that binds a vertex texture that is the whole remaining cost of the
        // phase. Same map, so the same invalidations (scene start, freed range, re-initialised
        // texture) apply without a second cache to keep honest. The key mixes a STAGE TAG
        // first, so a vertex list and a fragment list of identical bindings cannot collide in
        // the shared map - the fragment key starts from the program header, which is not a
        // separation a vertex list could be relied on to reproduce.
        //
        // >>> SHARED, NOT COPIED, AND USUALLY NOT EVEN LOOKED UP.
        //
        // This used to `to_vec` the shared entry into a fresh `Vec` per draw because
        // `capture::Draw` held one - "the tidier follow-up" the old note here named. It is an
        // `Arc<[_]>` on both sides now, so a hit is a refcount bump. And in front of the fold
        // and the probe sits the generation check: this title binds a vertex sampler on EVERY
        // draw, so the phase ran 626 times a frame to re-derive an answer that changes only
        // when the guest actually rebinds one.
        let vert_gen = self.vertex_texture_gen;
        let vert_binds = std::mem::take(&mut self.bound_vertex_textures);
        let vertex_textures: Arc<[crate::capture::BoundTexture]> = if vert_binds.is_empty() {
            // The common case for every other title: nothing bound decodes to nothing, and an
            // empty `Arc` slice allocates nothing.
            Arc::from(&[][..])
        } else if let Some(list) = self.texture_snapshots.vertex_set_from_previous_draw(vert_gen) {
            crate::perf::note_hit(crate::perf::Phase::DrawTexSetPrev);
            list
        } else {
            let _v = crate::perf::scope(crate::perf::Phase::DrawTexVertex);
            let vkey = {
                let mut h: u64 = 0xcbf2_9ce4_8422_2325;
                let mut mix = |v: u64| {
                    h ^= v;
                    h = h.wrapping_mul(0x0000_0100_0000_01b3);
                };
                mix(VERTEX_STAGE_TAG); // "VERTEX" - the stage tag, mixed first
                for b in &vert_binds {
                    mix(b.unit as u64);
                    for w in b.words {
                        mix(w as u64);
                    }
                }
                h
            };
            let list = match self.texture_snapshots.set_validated(
                ctx,
                vkey,
                VERTEX_STAGE_TAG,
                &vert_binds,
            ) {
                Some(set) => set,
                None => {
                    let decoded = self.snapshot_bound_textures(ctx, &vert_binds);
                    let set: Arc<[crate::capture::BoundTexture]> = decoded.as_slice().into();
                    self.texture_snapshots.set_insert(vkey, VERTEX_STAGE_TAG, &vert_binds, set.clone());
                    set
                }
            };
            self.texture_snapshots.remember_last_vertex_set(vert_gen, vkey, list.clone());
            list
        };
        self.bound_vertex_textures = vert_binds;
        drop(texture_phase);
        // The capture renderer samples a single texture (`textures.first()`). This title
        // binds the NORMAL map at unit 0, so pick the albedo sampler by fragment-program
        // reflection and move it to the front; without this every surface is tinted by a
        // normal map (flat blue/purple). Falls back to the unit-0 order when reflection
        // finds no albedo or that unit is not currently bound.
        // Failing name reflection, index 0 must still be a plausible SURFACE texture: a
        // one-dimensional lookup table (a fog ramp) or a cube map (an irradiance probe) can sort
        // ahead of the real albedo by unit number, and neither is indexed by surface UV. See
        // `Draw::albedo`, which drops a leading non-surface texture rather than stretch it.
        let textures: Arc<[crate::capture::BoundTexture]> = match cached_set {
            // Already the previous draw's answer, or one just re-proven for this scene -
            // either way the memo below is current and there is nothing to remember.
            Some(set) => set,
            None => {
                if let Some(pos) =
                    textures.iter().position(|t| t.height > 1 && t.faces <= 1).filter(|&p| p > 0)
                {
                    textures[..=pos].rotate_right(1);
                }
                if let Some(unit) = Self::fragment_albedo_unit(&fref) {
                    if let Some(pos) = textures.iter().position(|t| t.unit == unit) {
                        textures[..=pos].rotate_right(1);
                    }
                }
                let set: Arc<[crate::capture::BoundTexture]> = textures.into();
                // Taken and put back: `set_insert` needs the bindings that produced the
                // list, and they were returned to `self.bound_textures` above.
                let binds = std::mem::take(&mut self.bound_textures);
                self.texture_snapshots.set_insert(set_key, fheader as u64, &binds, set.clone());
                self.bound_textures = binds;
                set
            }
        };
        // Remember it against the bytes it came from, so the NEXT draw of this batch can skip
        // the gate entirely. Not on a previous-draw hit: that entry is this one.
        if previous_set.is_none() {
            self.texture_snapshots.remember_last_set(
                sampler_span,
                fheader,
                set_key,
                textures.clone(),
            );
        }
        if dump_fprog() {
            self.dump_fragment_program_samplers(ctx);
        }
        // Vertex uniforms: on the precomputed path the game wrote them into a default
        // uniform buffer bound by `sceGxmSetPrecomputedVertexState`, so read that guest
        // buffer now (its contents are current at draw time). On the direct path the
        // buffer is 0 and we fall back to the `sceGxmSetUniformDataF` capture.
        // >>> THE FRAGMENT SA BANK, READ ONCE FOR THE TWO READERS THAT WANT IT.
        //
        // The GXP capture below needs the whole bank; the material reflection needs nine
        // floats out of it. Read here, before either, so the material's nine reads come out
        // of these bytes instead of being nine separate crossings into guest memory - and so
        // the bank is read ONCE per draw rather than once per reader. Charged to the GXP
        // phase because that is the phase that owes the read; the material's share of it was
        // never a bulk read at all.
        //
        // Empty when the recompiler is off (the fixed-function path does not want the bank,
        // and reading it there would be pure cost) or when nothing is bound - and
        // `reflect_fragment_material` falls back to the per-word read for exactly that case.
        let frag_sa: Vec<u8> = {
            let _g = crate::perf::scope(crate::perf::Phase::DrawGxpCapture);
            if gxp_live_capture() && frag_uniform.buf != 0 {
                ctx.read_bytes(frag_uniform.buf, frag_uniform.size as usize)
            } else {
                Vec::new()
            }
        };
        let uniform_phase = crate::perf::scope(crate::perf::Phase::DrawUniforms);
        // >>> THE VERTEX BANK IS READ ONCE, IN THE FORM THE GUEST WROTE IT.
        //
        // Two consumers want it: the transform reflection wants FLOATS, and the recompiler
        // wants the RAW BYTES as the guest wrote them, BEFORE the composed MVP is stamped over
        // lanes 0..16 below (`vert_sa_raw` also covers the direct `sceGxmSetUniformDataF`
        // path, where the bound pointer is 0). It used to read floats and then serialise them
        // back to bytes - a third buffer per draw holding what the read had already produced
        // and thrown away, since a guest read is bytes to begin with.
        //
        // Off the recompiler path nothing wants the bytes, and there the float read stays as
        // it was: on native it BORROWS guest memory and converts in place, so materialising a
        // byte buffer there would be a cost with no reader.
        // >>> THE WHOLE BANK IS CONVERTED, AND READING ONLY THE REFLECTED LANES WAS SLOWER.
        //
        // The recompiled shader reads the guest's BYTES, so the floats here serve only the
        // three reflections below and `interpret_draw`'s sixteen - about fifty lanes at
        // offsets the reflection already knows. Reading exactly those lanes out of the bytes
        // and skipping this conversion is the obvious saving and it MEASURED AS A LOSS: two
        // interleaved browser runs put the lane-reader at cpu p25 8.30 and 8.07 ms/frame
        // against 7.68 for this. A bank is a few dozen floats, `chunks_exact(4)` vectorises,
        // and fifty bounds-checked per-lane reads through a closure cost more than converting
        // it. Do not re-try it without a measurement.
        let want_fixed_function = fixed_function_wanted();
        let mut uniforms = std::mem::take(&mut self.uniform_float_scratch);
        uniforms.clear();
        let vert_sa_raw = if gxp_live_capture() {
            let raw = self.current_vertex_uniform_bytes(ctx, &blk);
            uniforms.extend(
                raw.chunks_exact(4).map(|c| f32::from_le_bytes([c[0], c[1], c[2], c[3]])),
            );
            raw
        } else {
            // The fixed-function path's float read BORROWS guest memory natively and converts
            // in place, so it stays as it was; the extra copy into the scratch is one path's
            // cost and that path is not the one that ships.
            uniforms.extend_from_slice(&self.current_vertex_uniforms(ctx, &blk));
            Vec::new()
        };
        let lane = |i: usize| -> Option<f32> { uniforms.get(i).copied() };
        // The model-to-world matrix (for bringing the vertex normal into world space for
        // lighting). Read from the ORIGINAL bank, before the composed MVP is stamped over
        // lanes 0..16 (which is exactly where `vsModelToWorldMatrix` usually sits).
        // >>> STILL COMPUTED ON THE RECOMPILED PATH, AND DELIBERATELY.
        // It is two 16-lane reads out of a bank already in hand, and it is not only the
        // fixed-function pipeline's: `object_locations` reads the world matrix's translation
        // and basis to say WHERE a mesh is, which is the whole of the `locate` tool and of
        // every driving controller built on it. Handing those identity would not fail, it
        // would answer the origin - see [[vitaslop-camera-not-address]].
        // >>> READ ONCE, USED TWICE. The model->world matrix and the composed MVP are both
        // functions of the SAME pair of reflected matrices, and each used to read the pair for
        // itself - 32 lanes of the bank fetched twice per draw, ~626 draws a frame, for bytes
        // the first read already had.
        const IDENTITY_4X4: [f32; 16] =
            [1.0, 0.0, 0.0, 0.0, 0.0, 1.0, 0.0, 0.0, 0.0, 0.0, 1.0, 0.0, 0.0, 0.0, 0.0, 1.0];
        let world_proj = Self::reflected_world_proj_by(&vref, lane);
        let world = world_proj.map(|(w, _)| w).unwrap_or(IDENTITY_4X4);
        // Recover the true clip-space transform from the vertex program's reflected
        // uniforms. A 3D shader keeps a per-object model->world matrix and a shared
        // world->projection matrix as separate uniforms (named e.g. `vsModelToWorldMatrix`
        // / `vsWorldToProjectionMatrix`), and multiplies them in the shader. The capture
        // renderer has no shader, so compose that product here and stamp it over the
        // first 16 floats, which the software/GPU paths read as the MVP. A shader with a
        // single combined transform (2D UI: `vsPrimRenderTransform` at offset 0) has no
        // projection matrix, so this leaves its MVP untouched.
        let composed = match vref.mvp_off {
            Some(off) => Self::lanes_by(&lane, off as usize),
            None => world_proj.map(|(w, p)| Self::compose_mvp(&w, &p)),
        };
        if dump_vprog() {
            self.dump_vertex_program_params(ctx);
        }
        // >>> WHAT THE `Draw` ACTUALLY CARRIES, WHICH ON THE RECOMPILED PATH IS SIXTEEN FLOATS.
        //
        // The whole bank is here only because the fixed-function pipeline reads it as its
        // uniform block. A recompiled draw reads `vert_sa` - the guest's own bytes - and the
        // ONLY consumer of this field is `interpret_draw`, which classifies the draw's
        // coordinate space from lanes 0..16 (the MVP stamped just above, or the shader's own
        // combined transform). Carrying the tail was an allocation and a copy per draw for
        // bytes nothing downstream reads. A bank SHORTER than sixteen floats keeps its length,
        // because that length is what says "this is not an MVP draw".
        let bank_lanes = uniforms.len();
        let mut draw_uniforms: Vec<f32> = if want_fixed_function {
            uniforms.clone()
        } else {
            uniforms[..bank_lanes.min(16)].to_vec()
        };
        if let Some(mvp) = composed {
            if draw_uniforms.len() >= 16 {
                draw_uniforms[..16].copy_from_slice(&mvp);
            }
        }
        // >>> READ AFTER THE STAMP, WHICH IS WHERE IT ALWAYS WAS.
        //
        // A shader whose reflected exposure lane sits inside 0..16 reads the composed MVP
        // there rather than its own value. That is almost certainly not what the title means,
        // but it is what this has always done and it is a fixed-function value - so the order
        // is preserved rather than quietly corrected, and the recompiled path (where nothing
        // reads it) takes the same lanes for the same reason.
        let stamped = |i: usize| -> Option<f32> {
            match &composed {
                Some(mvp) if i < 16 && bank_lanes >= 16 => Some(mvp[i]),
                _ => lane(i),
            }
        };
        let exposure = Self::reflected_exposure_by(&vref, stamped);
        // The per-material fragment inputs (tint / directional light / ambient), reflected
        // from the fragment program's uniforms so the renderer reproduces the LIT colour.
        let material = if want_fixed_function {
            self.reflect_fragment_material(ctx, &fref, &textures, frag_uniform.buf, &frag_sa)
        } else {
            // The neutral material - what this already reflects for a shader declaring none.
            crate::capture::FragmentMaterial::default()
        };
        drop(uniform_phase);
        // The per-draw diagnostic dump, after everything it reports has been computed
        // (it prints the reflected transform and material rather than recomputing them).
        self.dump_gxp_blobs(ctx, fheader, vheader);
        self.dump_draw_gxp(
            ctx, &vref, &material, &textures, &attributes, primitive, index_count, stride,
            frag_uniform.buf,
        );
        // Snapshot the raw shader blobs + SA uniform bytes for the GXP->WGSL recompiler
        // path, but only when it is enabled (the reads are pure cost on the default
        // fixed-function path). The blobs come from `program_blob`, which reads each
        // container out of guest memory ONCE and hands every later draw a shared `Arc` -
        // see there for why a per-draw read is not affordable.
        let gxp_phase = crate::perf::scope(crate::perf::Phase::DrawGxpCapture);
        let (vprog, fprog, vert_sa, frag_sa) = if gxp_live_capture() {
            let vprog = self.program_blob(ctx, vheader);
            let fprog = self.program_blob(ctx, fheader);
            // The vertex SA is the pre-stamp raw uniforms captured above (covers both the bound
            // buffer and the direct sceGxmSetUniformDataF path).
            let vert_sa = vert_sa_raw;
            // >>> THE FRAGMENT BANK IS THE ONE ALREADY READ, NOT A SECOND READ OF THE SAME
            // BYTES. It was read above for the material's nine floats, under the identical
            // condition, and the guest cannot run between the two - so this used to copy the
            // whole bank out of guest memory a SECOND time on every draw, which is a bulk read
            // and an allocation per draw on the hottest path in the engine. The comment above
            // that read already said "read once for the two readers"; this is the second
            // reader actually taking it.
            (vprog, fprog, vert_sa, frag_sa)
        } else {
            (crate::capture::no_program(), crate::capture::no_program(), Vec::new(), Vec::new())
        };
        // ...and WHERE the fragment bank came from, which the bytes cannot say. See
        // `capture::Draw::frag_sa_addr`.
        let frag_sa_addr = if gxp_live_capture() { frag_uniform.buf } else { 0 };
        // The guest-memory window the vertex program's 0xE8 loads read through, snapshotted
        // at draw time like every other guest input. One map lookup for a program without
        // loads, which is every program of every other captured title.
        let mem_window =
            if gxp_live_capture() { self.capture_mem_window(ctx, &blk, vheader) } else { None };
        drop(gxp_phase);
        let render_state = {
            let _r = crate::perf::scope(crate::perf::Phase::DrawRenderState);
            use crate::vita::gxmctx::off;
            // The two contiguous regions the fixed-function state occupies: the scalars,
            // viewport and region clip below the program handles, and the back-face stencil
            // block that sits past the sampler array.
            const FRONT_LEN: usize = (off::FRONT_VISIBILITY_TEST_OP + 4 - off::CULL_MODE) as usize;
            const BACK_LEN: usize = (off::AFTER_BACK_STENCIL - off::AFTER_TEXTURES) as usize;
            let front = blk.span(off::CULL_MODE, FRONT_LEN);
            let back = blk.span(off::AFTER_TEXTURES, BACK_LEN);
            match &self.render_state_memo {
                Some((f, b, rs)) if &**f == front && &**b == back => rs.clone(),
                _ => {
                    let rs = std::sync::Arc::new(blk.render_state());
                    self.render_state_memo = Some((
                        front.to_vec().into_boxed_slice(),
                        back.to_vec().into_boxed_slice(),
                        rs.clone(),
                    ));
                    rs
                }
            }
        };
        self.uniform_float_scratch = uniforms;
        let record_phase = crate::perf::scope(crate::perf::Phase::DrawRecord);
        let draw = crate::capture::Draw {
            primitive,
            index_format,
            index_count,
            vertices,
            vertex_stride: stride,
            attributes,
            vertex_textures,
            indices: indices.into(),
            uniforms: draw_uniforms,
            textures,
            render_state,
            blend: self.fragment_program_blend(blk.word(crate::vita::gxmctx::off::FRAGMENT_PROGRAM)),
            fragment_program_header: fheader,
            exposure,
            material,
            world,
            vprog,
            fprog,
            vert_sa,
            frag_sa,
            frag_sa_addr,
            mem_window,
            shader_expanded: Self::reflected_shader_expanded(&vref),
        };
        match self.scene.as_mut() {
            Some(scene) => scene.draws.push(draw),
            // A draw outside begin/endScene has nowhere to go. That is a real hole in the
            // frame, so it is logged rather than dropped in silence.
            None => tracing::debug!(target: "vitaslop::gxm", index_count, "draw outside a scene - DROPPED"),
        }
        drop(record_phase);
    }

    /// Size in bytes of a program's default uniform buffer, computed from its
    /// reflected parameter table: the maximum `resource_index + <registers this
    /// parameter occupies>` over the uniform (`category == 1`) parameters, times 4.
    /// The program header's own size field (+0x2C) under-reports for shaders with a
    /// large uniform block (e.g. a world matrix at float 0 plus a world-to-projection
    /// matrix at float 16 plus lighting/fog), truncating the captured buffer and
    /// dropping the view-projection - so the reflected extent is the reliable size.
    ///
    /// A parameter's registers are NOT its component count: see the width table in
    /// [`reflect_program_uncached`] for what counting them that way cost.
    fn reflected_uniform_size_bytes(&mut self, ctx: &GuestCtx, header: u32) -> u32 {
        self.reflect_program(ctx, header).uniform_size_bytes
    }

    /// Drop every cached [`ProgramReflection`]. Called when the guest registers or
    /// unregisters a program with the shader patcher - the only points at which a
    /// header address can start meaning a different program - so a reused address can
    /// never be read through the old program's layout.
    ///
    /// A whole-table clear rather than a targeted eviction: registration happens a few
    /// hundred times over a run, at load, and the next draw simply re-reflects. Being
    /// obviously correct is worth more here than evicting one entry.
    pub fn invalidate_program_reflection(&mut self) {
        self.program_reflection.clear();
        self.program_blobs.clear();
        self.mem_window_specs.clear();
    }

    /// The guest-memory window the VERTEX program at `header` needs snapshotted per draw,
    /// if any (see `vitaslop_gxp_shader::mem_window_for_vertex_blob`). Memoised: the decode
    /// behind it runs once per registered program, and the per-draw cost for the near-total
    /// majority of programs that need no window is this one map lookup.
    fn mem_window_spec(&mut self, ctx: &GuestCtx, header: u32) -> Option<vitaslop_gxp_shader::MemWindow> {
        if header == 0 {
            return None;
        }
        if let Some(spec) = self.mem_window_specs.get(&header) {
            return *spec;
        }
        let blob = self.program_blob(ctx, header);
        let spec = vitaslop_gxp_shader::mem_window_for_vertex_blob(&blob);
        self.mem_window_specs.insert(header, spec);
        spec
    }

    /// Snapshot the guest-memory window this draw's VERTEX program reads through its 0xE8
    /// memory loads: `(window guest base address, the window's bytes)`.
    ///
    /// The bound address comes from the context block's own table - written by
    /// `sceGxmSetVertexUniformBuffer` directly or by binding a precomputed vertex state -
    /// and the extent is the program's own declared buffer size (see
    /// `vitaslop_gxp_shader::MemWindow`). Snapshotted AT DRAW TIME like every other guest
    /// input the capture carries, so the renderer later reads what the guest had bound now.
    ///
    /// Returns `None` (reported, throttled to once per program per run by the caller's
    /// nature: an unbound buffer is a property of the title's call order, not of one draw)
    /// when nothing is bound or the base is not 4-aligned - the renderer then DROPS the
    /// draw with a report rather than feeding the loads fabricated bytes.
    fn capture_mem_window(
        &mut self,
        ctx: &GuestCtx,
        blk: &crate::vita::gxmctx::Block<'_>,
        vheader: u32,
    ) -> Option<(u32, Vec<u8>)> {
        let spec = self.mem_window_spec(ctx, vheader)?;
        let _g = crate::perf::scope(crate::perf::Phase::DrawGxpCapture);
        let addr = blk.vertex_uniform_buffer(spec.buffer_index);
        if addr == 0 || addr % 4 != 0 {
            static REPORTED: std::sync::atomic::AtomicBool =
                std::sync::atomic::AtomicBool::new(false);
            if !REPORTED.swap(true, std::sync::atomic::Ordering::Relaxed) {
                tracing::warn!(
                    target: "vitaslop::gxm",
                    vertex_program = format_args!("{vheader:#x}"),
                    buffer_index = spec.buffer_index,
                    addr = format_args!("{addr:#x}"),
                    "a vertex program with MEMORY LOADS has no usable uniform buffer bound \
                     (unbound, or a base the 32-bit loads cannot address) - its draws will \
                     be DROPPED, not fed fabricated bytes"
                );
            }
            return None;
        }
        Some((addr, ctx.read_bytes(addr, spec.bytes as usize)))
    }

    /// The raw `SceGxmProgram` container bytes at `header`, read out of guest memory once and
    /// shared by every draw that binds it afterwards.
    ///
    /// The recompiler path needs the whole container per draw, and a container is a few
    /// kilobytes. A race frame submits 400+ draws over a couple of dozen distinct programs, so
    /// reading it per draw copies the same bytes hundreds of times a frame - the single largest
    /// avoidable item on the capture side of a recompiled frame. The bytes are IMMUTABLE while
    /// the program is registered (GXM's shader patcher owns them; the guest does not rewrite a
    /// live program), and the cache is dropped by [`invalidate_program_reflection`] at exactly
    /// the moment a header address can come to mean a different program - the same discipline,
    /// and the same clear, that `program_reflection` already relies on.
    fn program_blob(&mut self, ctx: &GuestCtx, header: u32) -> std::sync::Arc<[u8]> {
        if header == 0 {
            return crate::capture::no_program();
        }
        if let Some(b) = self.program_blobs.get(&header) {
            return b.clone();
        }
        // The blob size is the container total-length field at header+0x08, the same idiom the
        // GXP-bin dump uses.
        let sz = ctx.read_u32(header.wrapping_add(0x08)).clamp(0x40, 0x40000) as usize;
        let blob: std::sync::Arc<[u8]> = std::sync::Arc::from(ctx.read_bytes(header, sz));
        self.program_blobs.insert(header, blob.clone());
        blob
    }

    /// Say, once, that a title creates its fragment programs with a NULL `vertexProgram` - so
    /// this call names no PAIR and nothing can be prepared from it.
    ///
    /// # This is the measurement that decides whether shader preparation is possible at all
    /// GXM's signature carries `const SceGxmProgram *vertexProgram` so the patcher can patch the
    /// varying linkage, which makes the pair knowable while a loading screen is still up. Whether
    /// a given title actually passes it is a fact about that title, and it is not visible
    /// anywhere else: a NULL simply means the preparation silently never happens and the compile
    /// lands in a gameplay frame instead. MEASURED on a retail title: **all 17 of its distinct
    /// fragment programs are created with vertexProgram = NULL**, so its 160 pairs are first
    /// known at the DRAW that binds them.
    fn report_null_vertex_program(vertex_header: u32, fragment_header: u32) {
        use std::sync::atomic::{AtomicBool, Ordering};
        static SEEN: AtomicBool = AtomicBool::new(false);
        if vertex_header != 0 || fragment_header == 0 || SEEN.swap(true, Ordering::Relaxed) {
            return;
        }
        eprintln!(
            "gxm precompile: this title creates fragment programs with a NULL vertexProgram (first \
             at frag {fragment_header:#x}), so sceGxmShaderPatcherCreateFragmentProgram names no \
             shader PAIR and none can be compiled ahead of the draw that binds it. The WGSL \
             compile stays in the frame - see the hiccup log's split."
        );
    }

    /// Queue the shader PAIR `sceGxmShaderPatcherCreateFragmentProgram` just named, so the
    /// renderer can prepare it before a draw ever asks for it.
    ///
    /// # Why the pair is knowable here at all
    /// That call takes `const SceGxmProgram *vertexProgram` alongside the fragment program,
    /// because GXM needs both to patch the varying linkage - so the pair is fully determined
    /// while the title is still on its loading screen. The device's work at this point is
    /// patching pre-compiled USSE machine code; ours is producing WGSL and having a driver
    /// compile it, which is far more expensive and was happening at the first DRAW instead.
    ///
    /// Deduplicated by `(vertex header, fragment header)`: a title creates fragment programs in
    /// bursts and re-creates them after a patcher reset, and the renderer's own module cache
    /// would absorb that anyway - but the guest-memory reads and the queue would not.
    pub fn queue_shader_precompile(&mut self, ctx: &GuestCtx, vertex_header: u32, fragment_header: u32) {
        Self::report_null_vertex_program(vertex_header, fragment_header);
        if fragment_header == 0 {
            return;
        }
        if vertex_header == 0 {
            // The title named no pair. Under `VITASLOP_GXP_PRECOMPILE_CROSS` this fragment
            // program is offered against every vertex program created SO FAR - see
            // [`Self::cross_precompile`] for what that costs and why it is not the default.
            self.cross_precompile_fragment(ctx, fragment_header);
            return;
        }
        self.push_precompile_pair(ctx, vertex_header, fragment_header);
    }

    /// `VITASLOP_GXP_PRECOMPILE_CROSS`: for a title whose `sceGxmShaderPatcherCreateFragmentProgram`
    /// passes `vertexProgram = NULL`, offer the CROSS PRODUCT of the fragment programs it creates
    /// with the vertex programs it creates, and let the renderer keep the ones that LINK.
    ///
    /// # Why this is a knob and not the default
    /// Precompiling a pair the patcher NAMED is free of guesswork: the title said those two go
    /// together. A cross product does not know that. It is speculative work paid on a loading
    /// screen, and how much of it is wasted is a per-title measurement nobody has taken - which
    /// is exactly what this knob exists to take. **MEASURED on a retail title: all 17 of its
    /// distinct fragment programs are created with a NULL vertexProgram, and its race builds
    /// 160 pipelines IN FRAME**, so it is the title the question is about.
    ///
    /// `link_programs` is our own Rust and cheap, so a candidate that does not link costs a parse
    /// and nothing more; only a pair that LINKS costs a WGSL compile. The renderer reports how
    /// many of each, which is the number that decides whether this should ever become a default.
    fn cross_precompile() -> bool {
        static ON: std::sync::OnceLock<bool> = std::sync::OnceLock::new();
        *ON.get_or_init(|| crate::knobs::flag("VITASLOP_GXP_PRECOMPILE_CROSS"))
    }

    /// Record a fragment program created with a NULL `vertexProgram`, and (under the knob) pair
    /// it with every vertex program created so far.
    fn cross_precompile_fragment(&mut self, ctx: &GuestCtx, fragment_header: u32) {
        if self.null_fragment_headers.contains(&fragment_header) {
            return;
        }
        // Recorded UNCONDITIONALLY, not just under the knob: the list is what says WHEN this
        // title's pairs first become knowable, and that question is asked by
        // `report_program_creation_frame` on every run - see there for why it decides whether
        // preparing shaders ahead of a draw is possible for this title at all.
        self.null_fragment_headers.push(fragment_header);
        self.report_program_creation_frame();
        if !Self::cross_precompile() {
            return;
        }
        let vertices = self.created_vertex_headers.clone();
        for v in vertices {
            self.push_precompile_pair(ctx, v, fragment_header);
        }
    }

    /// Say, on a bounded ladder, HOW MANY shader programs the title has created and BY WHAT
    /// FRAME.
    ///
    /// # Why this line decides an open question
    /// Preparing a shader ahead of the draw that needs it is only possible if the PAIR is
    /// knowable before that draw. For a title whose `sceGxmShaderPatcherCreateFragmentProgram`
    /// passes `vertexProgram = NULL` the pair is never named, and the only material left is the
    /// two LISTS - so the whole question becomes "were both programs created early enough". A
    /// cross product over programs the title has not created yet cannot contain the pair the
    /// race draws, and that is indistinguishable, from the compile counters alone, from a cross
    /// product whose pairs simply do not link. **MEASURED on a retail title: 1,088 of 4,096
    /// candidates linked and compiled 4,356 ms ahead of any draw, and in-frame WGSL barely
    /// moved (744 -> 701 ms)** - which is only consistent with the race's real pairs being
    /// ABSENT from the candidate set. This line is what tells the two apart.
    ///
    /// Printed rather than gated behind a knob, because it is a handful of lines over a whole
    /// run and it answers a question every future session on the loading-screen hitch asks.
    /// The ladder is powers of two plus every ten, so a burst is visible without a line per
    /// program.
    fn report_program_creation_frame(&self) {
        let v = self.created_vertex_headers.len();
        let f = self.null_fragment_headers.len();
        let n = v + f;
        if n > 4 && !n.is_power_of_two() && n % 10 != 0 {
            return;
        }
        eprintln!(
            "gxm precompile: {v} vertex + {f} NULL-paired fragment programs created by frame {}",
            self.cur_frame
        );
    }

    /// Record a vertex program the patcher created, and (under the knob) pair it with every
    /// fragment program already created with a NULL `vertexProgram`.
    ///
    /// Called unconditionally from `sceGxmShaderPatcherCreateVertexProgram`, because the LIST is
    /// what the cross product needs and a title creates its programs in whatever order it likes -
    /// a vertex program created after the fragment programs would otherwise never be paired.
    pub fn note_vertex_program_created(&mut self, ctx: &GuestCtx, vertex_header: u32) {
        if vertex_header == 0 || self.created_vertex_headers.contains(&vertex_header) {
            return;
        }
        self.created_vertex_headers.push(vertex_header);
        self.report_program_creation_frame();
        if !Self::cross_precompile() {
            return;
        }
        let fragments = self.null_fragment_headers.clone();
        for f in fragments {
            self.push_precompile_pair(ctx, vertex_header, f);
        }
    }

    /// The shared tail of both queueing paths: dedupe, bound, and read the two blobs.
    ///
    /// **The cap is REPORTED when it bites.** A cross product is the one caller that can reach
    /// it, and a silently truncated candidate list would look exactly like a title whose pairs
    /// do not link - the opposite conclusion from the same evidence.
    fn push_precompile_pair(&mut self, ctx: &GuestCtx, vertex_header: u32, fragment_header: u32) {
        const MAX_PENDING: usize = 4096;
        if self.pending_precompile.len() >= MAX_PENDING {
            static SEEN: std::sync::atomic::AtomicBool = std::sync::atomic::AtomicBool::new(false);
            if !SEEN.swap(true, std::sync::atomic::Ordering::Relaxed) {
                tracing::warn!(
                    target: "vitaslop::gxm",
                    max = MAX_PENDING,
                    "gxm precompile: the candidate list reached its cap and later pairs are being \
                     DROPPED - anything past here still compiles at its first draw"
                );
            }
            return;
        }
        if !self.precompiled_pairs.insert((vertex_header, fragment_header)) {
            return;
        }
        let vprog = self.program_blob(ctx, vertex_header);
        let fprog = self.program_blob(ctx, fragment_header);
        std::sync::Arc::make_mut(&mut self.pending_precompile).push((vprog, fprog));
    }

    /// The shader pairs the patcher has named, for the renderer to prepare.
    ///
    /// # Why this does not DRAIN
    /// It used to, and the pairs were lost. A title creates its programs during boot, so the
    /// scene that happened to close next carried them - and that scene is not necessarily one
    /// anything renders. A `--headless` run without a shot window renders exactly ONE scene, at
    /// the very end, so every pair was drained into a scene nobody looked at and the preparation
    /// silently never happened. The renderer skips a pair whose module it already has, at the
    /// cost of one hash lookup, so re-offering the whole list is cheap and cannot be missed.
    pub fn shader_precompile(
        &self,
    ) -> std::sync::Arc<Vec<(std::sync::Arc<[u8]>, std::sync::Arc<[u8]>)>> {
        self.pending_precompile.clone()
    }

    /// The reflected constants of the program at `header`, walking its parameter table
    /// on the first ask and returning the cached answer afterwards. See
    /// [`ProgramReflection`] for why this is one walk instead of five.
    ///
    /// A zero header (no program bound, or a handle never recorded) reflects to the
    /// default: no matrices, no material, identity behaviour at every call site.
    fn reflect_program(&mut self, ctx: &GuestCtx, header: u32) -> ProgramReflection {
        if header == 0 {
            return ProgramReflection::default();
        }
        if let Some(r) = self.program_reflection.get(&header) {
            return *r;
        }
        let r = reflect_program_uncached(ctx, header);
        self.program_reflection.insert(header, r);
        r
    }

    /// Compose the model->projection MVP for the bound vertex program from its captured
    /// `uniforms`, using reflection to locate the model->world and world->projection
    /// matrices by their declared names. Returns `None` when the shader has no separate
    /// projection matrix (a single-transform 2D/UI shader), so the caller keeps the
    /// offset-0 matrix as-is. Both matrices are column-major 4x4 float blocks at their
    /// reflected `resource_index` (in floats); the result is `projection * world`.
    fn composed_mvp(&self, r: &ProgramReflection, uniforms: &[f32]) -> Option<[f32; 16]> {
        Self::composed_mvp_by(r, |i| uniforms.get(i).copied())
    }

    /// >>> THE THREE REFLECTIONS, OVER ANY SOURCE OF UNIFORM LANES.
    ///
    /// The bank is bytes in guest memory. The fixed-function pipeline wants the WHOLE thing as
    /// floats and gets a `Vec`; a recompiled draw wants none of it - its shader reads the raw
    /// bytes - and yet every draw converted the entire bank to floats so that these three could
    /// index it. They read at most 33 lanes between them, all at offsets the reflection already
    /// knows, so taking a lane READER instead of a slice lets the recompiled path pull those
    /// lanes straight out of the guest's bytes and skip the conversion and the allocation.
    fn lanes_by(lane: &impl Fn(usize) -> Option<f32>, off: usize) -> Option<[f32; 16]> {
        let mut m = [0f32; 16];
        for (k, v) in m.iter_mut().enumerate() {
            *v = lane(off + k)?;
        }
        Some(m)
    }

    fn reflected_world_proj_by(
        r: &ProgramReflection,
        lane: impl Fn(usize) -> Option<f32>,
    ) -> Option<([f32; 16], [f32; 16])> {
        let (wo, po) = (r.world_off? as usize, r.proj_off? as usize);
        Some((Self::lanes_by(&lane, wo)?, Self::lanes_by(&lane, po)?))
    }

    fn composed_mvp_by(
        r: &ProgramReflection,
        lane: impl Fn(usize) -> Option<f32>,
    ) -> Option<[f32; 16]> {
        // A shader that keeps ONE combined matrix needs no composition - but it does
        // need to be found, because it is not always at offset 0 and the fallback
        // there would then read whatever else the program put first.
        if let Some(off) = r.mvp_off.map(|o| o as usize) {
            return Self::lanes_by(&lane, off);
        }
        let (world, proj) = Self::reflected_world_proj_by(r, lane)?;
        Some(Self::compose_mvp(&world, &proj))
    }

    /// Column-major 4x4 multiply: `projection * world`.
    fn compose_mvp(world: &[f32; 16], proj: &[f32; 16]) -> [f32; 16] {
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
        out
    }

    /// The model-to-world matrix reflected from the vertex program's `vsModelToWorldMatrix`
    /// (column-major 4x4), or identity if the shader declares no world matrix. Used to bring
    /// the object-space vertex normal into world space for lighting.
    fn reflected_world_matrix(&self, r: &ProgramReflection, uniforms: &[f32]) -> [f32; 16] {
        Self::reflected_world_matrix_by(r, |i| uniforms.get(i).copied())
    }

    fn reflected_world_matrix_by(
        r: &ProgramReflection,
        lane: impl Fn(usize) -> Option<f32>,
    ) -> [f32; 16] {
        const IDENTITY: [f32; 16] =
            [1.0, 0.0, 0.0, 0.0, 0.0, 1.0, 0.0, 0.0, 0.0, 0.0, 1.0, 0.0, 0.0, 0.0, 0.0, 1.0];
        Self::reflected_world_proj_by(r, lane).map(|(w, _)| w).unwrap_or(IDENTITY)
    }


    /// Pull the vertex program's model->world and world->projection 4x4 matrices out of
    /// this draw's `uniforms`, at the offsets [`ProgramReflection`] located by name.
    /// Returns `(world, proj)` or `None` when the shader has no separate projection
    /// matrix (a single-transform 2D/UI shader) or the buffer is short of either.
    fn reflected_world_proj(
        r: &ProgramReflection,
        uniforms: &[f32],
    ) -> Option<([f32; 16], [f32; 16])> {
        Self::reflected_world_proj_by(r, |i| uniforms.get(i).copied())
    }

    /// Does the bound vertex program SYNTHESIZE its primitive instead of reading it?
    ///
    /// A point-sprite/billboard shader is handed one record per sprite - a centre position
    /// plus the basis it expands into corners with - and builds the quad itself. The
    /// declared ATTRIBUTE names are what say so, and they are the guest's own reflection
    /// data, not a guess: an expansion basis is a `scale_rotation`, or an explicit
    /// right/up axis pair. Ordinary geometry declares none of these (across this title's
    /// whole shader corpus the per-vertex names are position/normal/tangent/uv/colour).
    ///
    /// Consumed by [`crate::capture::Draw::shader_expanded`]. This costs one parameter-table
    /// scan per draw, the same scan the world/projection and exposure reflection already do.
    /// Reflected once per program (category-0 attribute names only, so a UNIFORM named
    /// "rotation" cannot trip it) - see [`ProgramReflection::shader_expanded`].
    fn reflected_shader_expanded(r: &ProgramReflection) -> bool {
        r.shader_expanded
    }

    /// Recover the scene exposure from the bound vertex program's reflected
    /// `vsCoarseExposureReg` uniform (a float4 whose first component is the linear
    /// exposure scale the shaders multiply lit albedo by before tone-mapping). Returns
    /// `1.0` when the shader declares no exposure uniform (2D/UI shaders), or when the
    /// value is not a sane positive number, so it is a safe no-op there.
    fn reflected_exposure(r: &ProgramReflection, uniforms: &[f32]) -> f32 {
        Self::reflected_exposure_by(r, |i| uniforms.get(i).copied())
    }

    fn reflected_exposure_by(r: &ProgramReflection, lane: impl Fn(usize) -> Option<f32>) -> f32 {
        let Some(off) = r.exposure_off else { return 1.0 };
        match lane(off as usize) {
            Some(e) if e.is_finite() && e > 0.0 => e,
            _ => 1.0,
        }
    }

    /// Diagnostic (VITASLOP_DUMP_VPROG): reflect the bound vertex program's parameter
    /// table (name / category / type / component_count / container / array_size /
    /// resource_index) once per unique program, so the uniform-buffer layout (which
    /// slots are the world matrix vs the shared view-projection) is known by name.
    /// Print every UNIFORM (category 1) the vertex program `ph` declares, with its
    /// name, its float offset, and the values this draw supplied - a 4x4 laid out as
    /// four columns.
    ///
    /// A capture renderer has no shader, so it can only place geometry with a matrix it
    /// RECOGNISES; when a title's shader names its transform something the reflection
    /// does not know, the renderer falls back to "the first sixteen floats" and draws
    /// the mesh wherever those happen to point. That failure is silent and looks like a
    /// pass that drew nothing. This is how you find the real transform: read the names.
    fn dump_named_uniforms(&self, ctx: &GuestCtx, ph: u32, values: &[f32]) {
        if ph == 0 {
            return;
        }
        let count = ctx.read_u32(ph.wrapping_add(0x24));
        let base = ph.wrapping_add(0x28).wrapping_add(ctx.read_u32(ph.wrapping_add(0x28)));
        for i in 0..count.min(64) {
            let p = base.wrapping_add(i.wrapping_mul(16));
            let word = ctx.read_u32(p.wrapping_add(4)) & 0xffff;
            if word & 0xf != 1 {
                continue; // category 1 = uniform
            }
            let comp = ((word >> 8) & 0xf).max(1);
            let array = ctx.read_u32(p.wrapping_add(8)).max(1);
            let res = ctx.read_u32(p.wrapping_add(0xc)) as usize;
            let name_addr = (p as i64 + ctx.read_u32(p) as i32 as i64) as u32;
            let raw = ctx.read_bytes(name_addr, 48);
            let name: String = raw.iter().take_while(|&&b| b != 0).map(|&b| b as char).collect();
            let n = (comp * array) as usize;
            let vals: Vec<String> = (0..n.min(16))
                .map(|k| values.get(res + k).map_or("-".into(), |v| format!("{v:.4}")))
                .collect();
            eprintln!("  uniform {name:?} res={res} comp={comp} array={array} = [{}]", vals.join(","));
        }
    }

    fn dump_vertex_program_params(&self, ctx: &GuestCtx) {
        use std::sync::Mutex;
        static SEEN: Mutex<Vec<u32>> = Mutex::new(Vec::new());
        let ph = self.vertex_program_header(self.bound_vertex_program(ctx));
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
            // The upper half of the +4 word is the `semantic` u16 that
            // `sceGxmProgramParameterGetSemantic`/`GetSemanticIndex` split. Printed RAW as
            // well as split, because a title that builds its `SceGxmVertexAttribute` array
            // by matching semantics produces ZERO attributes when the split is wrong - and
            // an empty attribute array is silent: the draw arrives with real geometry and
            // no way to fetch it.
            let semantic = ctx.read_u32(p.wrapping_add(4)) >> 16;
            eprintln!(
                "  param[{i}] name={name:?} cat={} type={} comp={} container={} array={array_size} \
                 res_index={res_index} semantic_word={semantic:#06x}",
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
    fn fragment_albedo_unit(r: &ProgramReflection) -> Option<u32> {
        r.albedo_unit
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
    fn reflect_fragment_material(
        &mut self,
        ctx: &GuestCtx,
        r: &ProgramReflection,
        textures: &[crate::capture::BoundTexture],
        // The fragment stage's bound default uniform buffer, as the caller resolved it -
        // 0 when nothing is bound OR when the binding is stale for the program about to
        // draw. Passed in rather than re-read so every reader of this draw agrees about
        // which of those it is; see `record_draw`.
        buf: u32,
        // The fragment default uniform buffer's bytes, when the caller has ALREADY read them
        // (the GXP capture path reads the whole bank per draw). Empty when it has not.
        //
        // # Why this is passed in rather than read here
        // The three reflected parameters below are read three scalar components at a time,
        // and each of those is a `dyn GuestMemory` access - which in the browser is a
        // boundary crossing. That is up to NINE crossings per draw to fetch nine floats out
        // of a buffer the very next phase copies in one call
        // ([[vitaslop-count-calls-not-bytes-across-the-guest-boundary]]). Taking the bytes
        // the caller already has costs nothing and removes all nine.
        //
        // It is bytes rather than floats because the components can be F16.
        sa: &[u8],
    ) -> crate::capture::FragmentMaterial {
        let mut m = crate::capture::FragmentMaterial::default();
        if buf != 0 {
            // Read the first three scalar components of a reflected parameter from the
            // fragment default uniform buffer at its register offset, honouring the
            // F16/F32 component type.
            //
            // Out of `sa` when it covers the parameter, and out of guest memory when it does
            // not - a short or absent bank is not an error here (the caller may not have read
            // one at all), and falling back is the SAME read this always did.
            let read3 = |p: ParamRef| -> [f32; 3] {
                let base = p.res.wrapping_mul(4) as usize;
                let width = if p.f16 { 2 } else { 4 };
                let local = sa.get(base..base + 3 * width);
                let byte_off = buf.wrapping_add(p.res.wrapping_mul(4));
                std::array::from_fn(|i| match (local, p.f16) {
                    (Some(s), true) => {
                        crate::render::half_to_f32(u16::from_le_bytes([s[i * 2], s[i * 2 + 1]]))
                    }
                    (Some(s), false) => f32::from_le_bytes([
                        s[i * 4],
                        s[i * 4 + 1],
                        s[i * 4 + 2],
                        s[i * 4 + 3],
                    ]),
                    (None, true) => {
                        crate::render::half_to_f32(ctx.read_u16(byte_off.wrapping_add(i as u32 * 2)))
                    }
                    (None, false) => ctx.read_f32(byte_off.wrapping_add(i as u32 * 4)),
                })
            };
            // The base-colour tint: the primary layer's tint (or a wheel's AlbedoColour).
            if let Some(p) = r.tint {
                let t = read3(p);
                if t.iter().all(|c| c.is_finite() && *c >= 0.0 && *c <= 8.0) {
                    m.tint = t;
                }
            }
            if let Some(p) = r.light_dir {
                let d = read3(p);
                if d.iter().all(|c| c.is_finite()) && d.iter().any(|c| *c != 0.0) {
                    m.light_dir = d;
                    m.has_light = true;
                }
            }
            if let Some(p) = r.light_col {
                let c = read3(p);
                if c.iter().all(|v| v.is_finite() && *v >= 0.0 && *v <= 16.0) {
                    m.light_col = c;
                }
            }
        }
        // Ambient: the average colour of the small `diffuseAmbientMap` irradiance texture, if
        // one is bound. It is a coarse (16x16 / 128x128) light probe, so its mean is a good
        // flat ambient for a renderer that does not sample it per-normal. Leaves the default
        // grey ambient when no such map is present.
        if let Some(amb) = textures.iter().find(|t| t.unit == 15) {
            if let Some(mean) = self.texture_snapshots.mean_rgb(amb) {
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
        let header = self.bound_fragment_program_header(ctx);
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
    #[allow(clippy::too_many_arguments)]
    /// `VITASLOP_DUMP_GXP_BIN=<dir>`: write the raw `SceGxmProgram` blobs (the whole container -
    /// header + parameter table + USSE bytecode) for the bound fragment and vertex programs,
    /// named `<type>_<header-addr>.gxp`, deduped by header address. These are the durable
    /// artifacts the clean-room GXP->WGSL recompiler decodes, and the corpus every ISA question
    /// is settled against. `SceGxmProgram.size` is the u32 at header+0x08 (the container's total
    /// byte length; clamped defensively).
    ///
    /// This has its OWN gate rather than living inside [`Self::dump_draw_gxp`]. Capturing the
    /// corpus and reading a per-draw trace are different jobs: the trace is frame-keyed and
    /// capped because it is gigabytes of text, while a corpus wants every program the whole run
    /// ever binds. Coupling them meant the shaders reachable only late in a level could not be
    /// captured at all without also emitting the trace for every draw before them.
    fn dump_gxp_blobs(&self, ctx: &GuestCtx, fh: u32, vh: u32) {
        // Cached: this runs per draw, and reading an unset environment variable on Windows is
        // not free (see `dump_vprog`).
        static WANT: std::sync::OnceLock<Option<String>> = std::sync::OnceLock::new();
        let Some(dir) = WANT.get_or_init(|| std::env::var("VITASLOP_DUMP_GXP_BIN").ok()) else {
            return;
        };
        use std::sync::Mutex;
        static SEEN: Mutex<Vec<u32>> = Mutex::new(Vec::new());
        for (kind, header) in [("frag", fh), ("vert", vh)] {
            if header == 0 {
                continue;
            }
            {
                let mut seen = SEEN.lock().unwrap();
                if seen.contains(&header) {
                    continue;
                }
                if seen.is_empty() {
                    let _ = std::fs::create_dir_all(dir);
                }
                seen.push(header);
            }
            let size = ctx.read_u32(header.wrapping_add(0x08)).clamp(0x40, 0x40000);
            let bytes = ctx.read_bytes(header, size as usize);
            let path = std::path::Path::new(dir).join(format!("{kind}_{header:08x}.gxp"));
            let _ = std::fs::write(path, &bytes);
        }
    }

    fn dump_draw_gxp(&self, ctx: &GuestCtx, vref: &ProgramReflection, material: &crate::capture::FragmentMaterial, textures: &[crate::capture::BoundTexture], attributes: &[crate::capture::VertexAttribute], primitive: u32, index_count: u32, stride: u32, frag_buf: u32) {
        // Cached: this runs per draw, and reading an unset environment variable on
        // Windows is not free (see `dump_vprog`).
        static WANT: std::sync::OnceLock<Option<String>> = std::sync::OnceLock::new();
        let want = match WANT.get_or_init(|| std::env::var("VITASLOP_DUMP_DRAW_GXP").ok()) {
            Some(s) => s.clone(),
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
        let vh = self.vertex_program_header(self.bound_vertex_program(ctx));
        let fh = self.bound_fragment_program_header(ctx);
        // The caller has already reflected it - recomputing here would be a second
        // (and, with the ambient-mean cache, a differently-warmed) evaluation of the
        // same thing.
        let mat = material;
        eprintln!(
            "DRAW frame={disp} seq={seq} prim={primitive:#010x} idx={index_count} stride={stride} vprog={vh:#x} fprog={fh:#x} fubuf={frag_buf:#x}"
        );
        // Where this draw's transform comes from, and what it is. A mesh rendered in the
        // wrong place is always one of three things: the wrong default uniform buffer still
        // bound (`STALE-ubuf`), no reflected projection matrix so offset 0 is used raw and
        // may not be an MVP at all (`composed=no`), or a leftover `sceGxmSetUniformDataF`
        // capture (`setUniformDataF` with a buffer of 0). This line separates them, and
        // prints the MVP's translation column - the clip-space position of the object's
        // origin, which is what "it is drawn at the wrong place" actually means.
        let vbound = self.vertex_uniform(ctx);
        let vbuf = vbound.buf;
        let usable = vbuf != 0 && vbound.size >= 4 && vbound.header == vh;
        let source = match (vbuf != 0, vbound.header == vh) {
            (true, true) => "ubuf",
            (true, false) => "STALE-ubuf",
            (false, _) => "setUniformDataF",
        };
        let raw: Vec<f32> = if usable {
            (0..(vbound.size / 4) as usize)
                .map(|i| ctx.read_f32(vbuf.wrapping_add(i as u32 * 4)))
                .collect()
        } else {
            self.sa_bank_floats(ctx)
        };
        let composed = self.composed_mvp(vref, &raw);
        let eff = composed.unwrap_or_else(|| {
            let mut m = [0f32; 16];
            for (i, s) in m.iter_mut().zip(raw.iter()) {
                *i = *s;
            }
            m
        });
        eprintln!(
            "  TRANSFORM source={source} ubuf={vbuf:#x} lanes={} composed={} origin_clip=[{:.3},{:.3},{:.3},{:.3}]",
            raw.len(),
            if composed.is_some() { "yes" } else { "no" },
            eff[12], eff[13], eff[14], eff[15],
        );
        // The whole transform, column by column. `origin_clip` above is only its last
        // column, which says WHERE the origin lands and nothing about the scale or the
        // axes - and a mesh that misses the screen because its scale is wrong looks
        // exactly like one whose translation is wrong.
        eprintln!(
            "  MVP c0=[{:.4},{:.4},{:.4},{:.4}] c1=[{:.4},{:.4},{:.4},{:.4}] \
             c2=[{:.4},{:.4},{:.4},{:.4}] c3=[{:.4},{:.4},{:.4},{:.4}]",
            eff[0], eff[1], eff[2], eff[3],
            eff[4], eff[5], eff[6], eff[7],
            eff[8], eff[9], eff[10], eff[11],
            eff[12], eff[13], eff[14], eff[15],
        );
        // Every uniform lane the program declares, by NAME, so a transform that is not
        // in the first sixteen floats can be found rather than guessed at. This is the
        // dump's whole job on a shader whose transform the reflection does not know.
        self.dump_named_uniforms(ctx, vh, &raw);
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
        if let Some(mut scene) = self.scene.take() {
            scene.adopt_viewport_extent();
            // Hand the renderer whatever the shader patcher has named since the last scene, so
            // it can prepare those pairs before it encodes. Riding the scene keeps this on the
            // path the renderer already consumes - no engine has to learn a new call - and it
            // reaches the renderer at the first scene AFTER the loading screen's creates, which
            // is exactly where the spare time is.
            scene.precompile = self.shader_precompile();
            // Goes through `push_scene`, not a bare push, so a bounded-retention run
            // still folds every scene into the determinism signature.
            self.capture.push_scene(scene);
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

/// Walk one `SceGxmProgram`'s parameter table and reflect every constant the capture
/// path needs from it. This is the uncached body behind
/// [`VitaState::reflect_program`] - see [`ProgramReflection`] for the caching rationale
/// and why it must stay a pure function of the program blob.
///
/// Table layout (from the guest's own reflection API, the same offsets
/// `sceGxmProgramGetParameter` and friends use): `header+0x24` is the parameter count,
/// `header+0x28` holds the offset to the table, and each 16-byte entry is
/// `{ +0x0 name offset (signed, relative to the entry), +0x4 packed word, +0x8 array
/// size, +0xc resource index }`. The packed word's low nibble is the category (0
/// attribute, 1 uniform, 2 sampler), the next the component type (1 = F16), the next
/// the component count.
fn reflect_program_uncached(ctx: &GuestCtx, header: u32) -> ProgramReflection {
    let mut r = ProgramReflection::default();
    r.default_uniform_bytes = default_uniform_buffer_bytes(ctx, header);
    let count = ctx.read_u32(header.wrapping_add(0x24));
    let base = header.wrapping_add(0x28).wrapping_add(ctx.read_u32(header.wrapping_add(0x28)));
    let mut max_regs = 0u32;
    let mut best_albedo: Option<(i32, u32)> = None;
    // One scratch buffer for every name in the table: names are short and this walk is
    // the whole reason the per-draw version allocated.
    let mut name = String::with_capacity(64);
    for i in 0..count.min(256) {
        let p = base.wrapping_add(i.wrapping_mul(16));
        let word = ctx.read_u32(p.wrapping_add(4)) & 0xffff;
        let category = word & 0xf;
        let ptype = (word >> 4) & 0xf;
        let comp = (word >> 8) & 0xf;
        let res = ctx.read_u32(p.wrapping_add(0xc));
        let name_addr = (p as i64 + ctx.read_u32(p) as i32 as i64) as u32;
        read_lowercase_name(ctx, name_addr, &mut name);
        match category {
            // A per-vertex attribute. Only its NAME matters: an expansion basis
            // (`scale_rotation`, or an explicit right/up axis pair) says the vertex
            // program synthesizes its own primitive rather than reading triangles.
            0 => {
                if name.contains("scale_rotation")
                    || name.contains("rightvector")
                    || name.contains("upvector")
                {
                    r.shader_expanded = true;
                }
            }
            1 => {
                let array = ctx.read_u32(p.wrapping_add(8)).max(1);
                let is_f16 = ptype == 1;
                // The extent this parameter occupies, in 32-BIT REGISTERS - which is what
                // `resource_index` counts, and what the buffer is sized in.
                //
                // A component is not a register. The width comes from the type nibble, the
                // same mapping as `ParamType::component_bytes`: F16/U16/S16/C10 pack two
                // components per register and U8/S8 four, so an `F16[4]` at register 2 ends
                // at register 4, not register 6. Counting every component as a register
                // OVER-SIZES the buffer, and the over-size is not benign: the default
                // uniform buffer is a recycled per-scene ring, so the bytes past what the
                // guest wrote are the PREVIOUS draw's uniforms. That is how a fragment
                // program declaring 16 bytes came to be captured with 24, whose tail
                // "ramped smoothly" frame over frame and read exactly like a runaway
                // exposure multiplier the guest was computing. It was our own arithmetic.
                let comps = comp.max(1).wrapping_mul(array);
                let regs = match ptype {
                    1 | 2 | 5 | 6 => comps.div_ceil(2),
                    7 | 8 => comps.div_ceil(4),
                    // F32/U32/S32, and anything whose width this table does not know: one
                    // register per component, which is the widest reading and so the one
                    // that cannot truncate a buffer.
                    _ => comps,
                };
                max_regs = max_regs.max(res.wrapping_add(regs));
                let pref = ParamRef { res, comp: comp.max(1) as u8, f16: is_f16 };
                // A 4x4 matrix is declared as component_count 4, array_size 4.
                let is_matrix = comp == 4 && array == 4;
                // A single COMBINED model->clip matrix, tested first: its name contains
                // the projection substring too, and reading it as the projection half of
                // a pair would compose it with an identity world and place the mesh by a
                // matrix that is already complete.
                if is_matrix
                    && (name.contains("worldviewproj")
                        || name.contains("modelviewproj")
                        || name.contains("wvp"))
                {
                    r.mvp_off = Some(res);
                } else if is_matrix
                    && (name.contains("toprojection")
                        || name.contains("worldtoclip")
                        // `viewProjection` and the shorter `viewProj` this title's race
                        // shaders use. Its model half is named plainly `world` (below);
                        // between them they are the whole transform, and without both the
                        // renderer falls back to "the first sixteen floats" - which for
                        // these programs is the MODEL matrix alone, so the world is drawn
                        // with no camera at all and no observer can recover the eye.
                        || name.contains("viewproj"))
                {
                    r.proj_off = Some(res);
                } else if is_matrix && (name.contains("modeltoworld") || name == "world") {
                    r.world_off = Some(res);
                } else if name.contains("coarseexposure") {
                    r.exposure_off = Some(res);
                } else if (name.contains("albedocolour")
                    || name.contains("albedocolor")
                    || name.contains("primarytint"))
                    && comp >= 3
                {
                    // The primary layer's tint only: a "secondary" tint belongs to a
                    // second material layer the capture renderer does not composite.
                    r.tint = Some(pref);
                } else if name.contains("light0direction") && comp >= 3 {
                    r.light_dir = Some(pref);
                } else if (name.contains("light0colour") || name.contains("light0color"))
                    && comp >= 3
                {
                    r.light_col = Some(pref);
                }
            }
            2 => {
                // Sampler. Its resource index IS its texture unit, so the number of units
                // the program occupies is one past the highest - which is the length of
                // the array `sceGxmPrecomputed*StateSetAllTextures` is handed.
                r.texture_unit_count = r.texture_unit_count.max(res.wrapping_add(1));
                let score = albedo_name_score(&name);
                if score > 0 && best_albedo.map(|(s, _)| score > s).unwrap_or(true) {
                    best_albedo = Some((score, res));
                }
            }
            _ => {}
        }
    }
    r.albedo_unit = best_albedo.map(|(_, u)| u);
    r.uniform_size_bytes = max_regs.wrapping_mul(4);
    r
}

/// Diagnostic (`VITASLOP_DUMP_VPROG`): dump each vertex program's parameter table.
///
/// Read once and cached. These two are tested ONCE PER DRAW, and on Windows
/// `std::env::var` copies and re-encodes the whole environment block before it can
/// answer - so an unset diagnostic knob was costing more per frame than several draws.
fn dump_vprog() -> bool {
    static ON: std::sync::OnceLock<bool> = std::sync::OnceLock::new();
    *ON.get_or_init(|| std::env::var_os("VITASLOP_DUMP_VPROG").is_some())
}

/// Diagnostic (`VITASLOP_DUMP_FPROG`): dump each fragment program's samplers. See
/// [`dump_vprog`] for why this is cached.
fn dump_fprog() -> bool {
    static ON: std::sync::OnceLock<bool> = std::sync::OnceLock::new();
    *ON.get_or_init(|| std::env::var_os("VITASLOP_DUMP_FPROG").is_some())
}

/// Longest parameter name read while reflecting. The previous per-call code read 48 or
/// 64 bytes depending on the call site; 64 is the larger of the two, so nothing that
/// used to match stops matching.
const MAX_PARAM_NAME: usize = 64;

/// Read the NUL-terminated parameter name at `addr` into `out`, lowercased. Reuses the
/// caller's buffer so a whole parameter table costs one allocation, not one per name.
fn read_lowercase_name(ctx: &GuestCtx, addr: u32, out: &mut String) {
    let mut raw = [0u8; MAX_PARAM_NAME];
    ctx.read_into(addr, &mut raw);
    out.clear();
    out.extend(raw.iter().take_while(|&&b| b != 0).map(|&b| (b as char).to_ascii_lowercase()));
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
    use crate::vita::gxm::{DEFAULT_UNIFORM_BUFFER_MAX_WORDS, GXP_DEFAULT_UNIFORM_BUFFER_COUNT_OFF};
    if header == 0 {
        return 0;
    }
    ctx.read_u32(header.wrapping_add(GXP_DEFAULT_UNIFORM_BUFFER_COUNT_OFF))
        .min(DEFAULT_UNIFORM_BUFFER_MAX_WORDS)
        .wrapping_mul(4)
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
    // Through the knob table. This is the one experiment that separates "the guest wrote this"
    // from "this is the previous draw's uniforms still in the ring", and the value it has to
    // decide that about (`screenTintColour`, the white-out) only ever appears in the BROWSER -
    // which has no environment, so the knob could not be set on the only engine that shows it.
    if buf == 0 || !*ON.get_or_init(|| crate::knobs::var_os("VITASLOP_GXM_UNIFORM_POISON").is_some())
    {
        return;
    }
    for i in 0..size / 4 {
        ctx.write_u32(buf + i * 4, 0x7fc0_dead);
    }
}

/// Whether a captured draw's FIXED-FUNCTION representation will be used by anything.
///
/// It will not when the recompiler is live and fixed-function is not allowed as a fallback -
/// which is the SHIPPING configuration, and `RenderSceneBuilder::gxp_only` is the same test on
/// the other side of the capture. There the flag already skips the per-vertex walk; here it
/// skips the per-draw MATERIAL reflection - tint, directional light and ambient, the last of
/// which is the average of the bound irradiance map. A recompiled draw runs the guest's own
/// shader and derives its lighting from the SA banks, so the material is computed, stored in
/// the `Draw` and thrown away.
///
/// Two neighbours are deliberately NOT in this set. The MVP is still composed, because
/// `interpret_draw` classifies a draw's coordinate space by it and that classification decides
/// depth, culling and the opaque range on the recompiled path too. The model->world matrix is
/// still reflected, because `object_locations` - the `locate` tool and every driving controller
/// over it - reads it to say where a mesh IS, and identity there answers the origin rather than
/// failing.
fn fixed_function_wanted() -> bool {
    use std::sync::OnceLock;
    static WANTED: OnceLock<bool> = OnceLock::new();
    *WANTED.get_or_init(|| {
        !(gxp_live_capture() && !crate::knobs::flag("VITASLOP_GXP_ALLOW_FIXED_FUNCTION"))
    })
}

/// Whether the GXP->WGSL recompiler capture path is enabled (env `VITASLOP_GXP_LIVE`).
/// Checked once and cached, so the per-draw `record_draw` gate is a cheap load rather than
/// an environment lookup per draw.
fn gxp_live_capture() -> bool {
    use std::sync::OnceLock;
    static ENABLED: OnceLock<bool> = OnceLock::new();
    *ENABLED.get_or_init(|| crate::knobs::flag("VITASLOP_GXP_LIVE"))
}

/// Report - once per distinct base format, unconditionally - that a bound texture's format
/// has no known block geometry, so the sampler unit it was bound to is left EMPTY.
///
/// This has to be loud rather than a `debug!` behind a filter nobody has on. Downstream the
/// only evidence is a recompiled shader falling back with "sampler unit N wants Two but the
/// bound units are [...]", and reading that backwards to a texture format costs a session.
/// Deduped by format because a title binds the same format on hundreds of draws a frame.
/// Report - once per (base format, swizzle) - a texture resolved from its CONTROL WORDS rather
/// than from the exact format recorded at `sceGxmTextureInit*`.
///
/// This is the path a texture takes once a title copies the struct it lives in, and it is
/// lossier than the recorded one: the base format has to be reassembled from a 5-bit field plus
/// an extension bit, so every format above 0x7f (all of BC/PVRTC, and `U2F10F10F10`) decodes as
/// a different, smaller format if that bit is not carried. A UI atlas that is really BC3 read as
/// a 16-bit uncompressed format is not a dropped texture - it renders, subtly wrong.
fn report_texture_resolved_from_control_words(unit: u32, base_format: u32, swizzle: u32) {
    use std::collections::HashSet;
    use std::sync::Mutex;
    static SEEN: Mutex<Option<HashSet<(u32, u32)>>> = Mutex::new(None);
    let mut g = SEEN.lock().unwrap_or_else(|e| e.into_inner());
    if !g.get_or_insert_with(HashSet::new).insert((base_format, swizzle)) {
        return;
    }
    eprintln!(
        "gxm texture: unit {unit} bound a texture with NO recorded format - resolving it from \
         its control words alone as base format {base_format:#04x}, swizzle {swizzle}. This is a \
         copy of a texture initialised elsewhere."
    );
}

fn report_unsized_texture_format(unit: u32, base_format: u32, tex_type: u32, width: u32, height: u32) {
    use std::collections::HashSet;
    use std::sync::Mutex;
    static SEEN: Mutex<Option<HashSet<(u32, u32)>>> = Mutex::new(None);
    let mut g = SEEN.lock().unwrap_or_else(|e| e.into_inner());
    if !g.get_or_insert_with(HashSet::new).insert((unit, base_format)) {
        return;
    }
    eprintln!(
        "gxm texture: unit {unit} base format {base_format:#04x} (type {tex_type}, \
         {width}x{height}) has no known block size - EVERY unit bound to this format is left \
         unbound, and any shader sampling it falls back to fixed-function"
    );
}

/// Report - once per base format - that a CUBE map declares mip levels this path deliberately
/// does not read.
///
/// Six faces each carrying a chain can interleave two ways (chain per face, or every face's
/// level 0 then every face's level 1), and no clean source this project may read publishes
/// which. Guessing would put a face's level 1 where its level 2 belongs, which reads as a
/// correctly-shaped cube map that goes wrong only when something minifies - the kind of defect
/// that survives a whole session of looking at screenshots. So the chain is not read, and the
/// six faces get their level 0 exactly as they always have.
fn report_cube_mip_chain_skipped(unit: u32, base_format: u32, width: u32, height: u32, mip_field: u32) {
    use std::collections::HashSet;
    use std::sync::Mutex;
    static SEEN: Mutex<Option<HashSet<u32>>> = Mutex::new(None);
    let mut g = SEEN.lock().unwrap_or_else(|e| e.into_inner());
    if !g.get_or_insert_with(HashSet::new).insert(base_format) {
        return;
    }
    tracing::warn!(
        target: "vitaslop::gxm",
        "gxm texture: unit {unit} bound a CUBE map ({width}x{height}, base format \
         {base_format:#04x}) whose control word declares {mip_field} mip levels - only level 0 \
         of each face is snapshotted, because how six chains interleave in guest memory is not \
         established. A minified sample of this texture reads OUR box-filtered chain, not the \
         guest's."
    );
}

/// Report - once per base format - that the guest's mip chain could not be read, so the
/// snapshot fell back to level 0 alone.
///
/// This is expected for a texture whose data pointer is a render target's colour surface: it
/// has no mips in memory whatever its control word says, and the bytes past level 0 are
/// somebody else's or nothing at all. It is NOT expected for an ordinary asset, and there the
/// line is the difference between "the guest has no mips here" and "we computed the chain size
/// wrong", which need opposite fixes.
fn report_mip_chain_unreadable(unit: u32, base_format: u32, data_addr: u32, width: u32, height: u32, levels: u32) {
    use std::collections::HashSet;
    use std::sync::Mutex;
    static SEEN: Mutex<Option<HashSet<u32>>> = Mutex::new(None);
    let mut g = SEEN.lock().unwrap_or_else(|e| e.into_inner());
    if !g.get_or_insert_with(HashSet::new).insert(base_format) {
        return;
    }
    tracing::warn!(
        target: "vitaslop::gxm",
        "gxm texture: unit {unit} ({width}x{height}, base format {base_format:#04x} at \
         {data_addr:#010x}) declares {levels} mip levels but its memory does not extend that \
         far - falling back to level 0 alone for every texture of this format"
    );
}

/// How many sampler-unit bindings each drop cause has cost, over the whole run.
///
/// The per-cause REPORTS are deduped so they stay readable, which means they say nothing
/// about scale - and scale is what decides which cause to fix first. One unsized format on
/// four units and a zero handle on three read identically in the log while costing wildly
/// different numbers of draws. Counted here, printed by [`report_texture_drops`].
static TEXTURE_DROPS: std::sync::Mutex<[u64; 4]> = std::sync::Mutex::new([0; 4]);

/// Drop causes, in [`TEXTURE_DROPS`] order.
const DROP_CAUSES: [&str; 4] = [
    "zero control words (null data pointer - bound to a 1x1 zero texel, not dropped)",
    "unsized format",
    "pixels not readable",
    "undecodable base format",
];

/// How many times a `sceGxmTextureInit*` / `SetFormat` recorded a format for an address. A zero
/// here alongside a large drop count means the title never goes through the init API at all, and
/// builds its control words itself - which is a completely different investigation from "it
/// inits somewhere we are not looking".
static TEXTURE_INITS: std::sync::atomic::AtomicU64 = std::sync::atomic::AtomicU64::new(0);

/// Distinct `SceGxmTexture*` handles that read as all-zero control words at draw time. See
/// [`report_zero_texture_handle`].
static ZERO_TEXTURE_ADDRS: std::sync::Mutex<Option<std::collections::HashSet<u32>>> =
    std::sync::Mutex::new(None);

/// Vblank spins parked, and the guest instructions that would otherwise have been spent
/// spinning. Reported once per run rather than per park - a race parks once or twice a frame
/// and a line each would bury everything else.
static VBLANK_SPIN_PARKS: std::sync::atomic::AtomicU64 = std::sync::atomic::AtomicU64::new(0);

/// How many vblank spins have been parked so far - see [`report_vblank_spin_parks`]. The
/// browser reports it on its running status line rather than at run end, because a phone
/// run is watched while it happens and may never reach an orderly end.
pub fn vblank_spin_parks() -> u64 {
    VBLANK_SPIN_PARKS.load(std::sync::atomic::Ordering::Relaxed)
}

/// One line saying the spin guard fired, and how often. Silent when it never did - which is
/// itself the reading for a title that does not spin on the vblank counter.
///
/// It is NOT silent when it does fire, because this is a behaviour change: the guest's own
/// wait loop is being cut short, and a run where that happens should say so rather than
/// leave the next person to infer it from a frame time.
pub fn report_vblank_spin_parks() {
    let n = VBLANK_SPIN_PARKS.load(std::sync::atomic::Ordering::Relaxed);
    if n == 0 {
        return;
    }
    tracing::info!(
        target: "vitaslop::perf",
        "vblank spin: {n} wait loops PARKED on the next vblank after          {} reads of the counter inside one resume. Each one is a loop that could not have          observed a change (the mirror is frozen while guest code runs) and would otherwise          have run until its own fuel dragged the clock to the vblank edge",
        crate::vita::mirror::SPIN_BUDGET,
    );
}

fn note_texture_drop(cause: usize) {
    if let Ok(mut g) = TEXTURE_DROPS.lock() {
        g[cause] += 1;
    }
}

/// Print how many sampler-unit bindings were dropped, by cause, over the run. Call once at
/// the end of a run; silent when nothing was dropped.
pub fn report_texture_drops() {
    let counts = *TEXTURE_DROPS.lock().unwrap_or_else(|e| e.into_inner());
    if counts.iter().all(|&c| c == 0) {
        return;
    }
    let mut ranked: Vec<_> = DROP_CAUSES.iter().zip(counts).filter(|(_, c)| *c > 0).collect();
    ranked.sort_by(|a, b| b.1.cmp(&a.1));
    eprintln!("gxm texture: sampler-unit bindings DROPPED over this run, by cause:");
    for (cause, n) in ranked {
        eprintln!("gxm texture:   {n} - {cause}");
    }
    eprintln!(
        "gxm texture:   {} texture format(s) were recorded through the init API over the run",
        TEXTURE_INITS.load(std::sync::atomic::Ordering::Relaxed)
    );
    if let Ok(g) = ZERO_TEXTURE_ADDRS.lock() {
        if let Some(addrs) = g.as_ref().filter(|a| !a.is_empty()) {
            let mut list: Vec<u32> = addrs.iter().copied().collect();
            list.sort_unstable();
            let shown: Vec<String> = list.iter().take(12).map(|a| format!("{a:#x}")).collect();
            eprintln!(
                "gxm texture:   the zero control words come from {} DISTINCT handle(s): {}{}",
                list.len(),
                shown.join(" "),
                if list.len() > shown.len() { " ..." } else { "" }
            );
        }
    }
}

/// Report - once per unit - that a bound `SceGxmTexture` handle reads as all zeros at draw
/// time, naming which binding path put it there.
///
/// `from_precomputed` is the whole point of the message: a zero handle bound through
/// `sceGxmSetFragmentTexture` means the guest really bound an uninitialised texture, while
/// one that arrived with a precomputed fragment state means the state record is stale or
/// keyed wrong on OUR side. Those need opposite investigations and look identical downstream.
fn report_zero_texture_handle(
    ctx: &GuestCtx,
    binding: &TextureBinding,
    exact_format: Option<u32>,
    nearby: Option<(i64, u32)>,
) {
    let (unit, addr, from_precomputed) = (binding.unit, binding.addr, binding.from_precomputed);
    use std::collections::HashSet;
    use std::sync::Mutex;
    // Every distinct zero HANDLE is counted, so the end-of-run summary can say how many
    // addresses are involved. "2.3 million dropped bindings" reads like a catastrophe and is
    // consistent with three addresses rebound every frame; the two need different responses.
    if let Ok(mut a) = ZERO_TEXTURE_ADDRS.lock() {
        a.get_or_insert_with(HashSet::new).insert(addr);
    }
    // Once per (unit, ADDRESS), not per unit: a title binds a different uninitialised handle
    // to the same unit for each of its post-process steps, and reporting only the first tells
    // you a chain is broken without telling you how many distinct objects are involved - which
    // is the difference between one bad pointer and a whole class of them.
    static SEEN: Mutex<Option<HashSet<(u32, u32)>>> = Mutex::new(None);
    let mut g = SEEN.lock().unwrap_or_else(|e| e.into_inner());
    if !g.get_or_insert_with(HashSet::new).insert((unit, addr)) {
        return;
    }
    // The memory AROUND the handle, so the two ways this happens can be told apart: memory
    // that is zero for a long stretch means the struct was never written, while live data
    // either side of a zero window means the handle address is off by an offset.
    let window: Vec<String> = (0..32)
        .map(|i| format!("{:08x}", ctx.read_u32(addr.wrapping_sub(48).wrapping_add(i * 4))))
        .collect();
    eprintln!("gxm texture: memory around handle {addr:#x} (from -48): {}", window.join(" "));
    let via = if from_precomputed {
        "a PRECOMPUTED fragment state"
    } else {
        "sceGxmSetFragmentTexture"
    };
    // Whether a `sceGxmTextureInit*`/`SetFormat` was ever seen for THIS address splits the
    // cause in two: recorded means the guest initialised the texture somewhere else and the
    // bytes here are a copy that never got them (a by-value problem on our side); not
    // recorded means the guest never went through an init we implement at all.
    let init = match (exact_format, nearby) {
        (Some(f), _) => format!("a format ({f:#010x}) WAS recorded for this address"),
        // A texture initialised a few bytes away is the whole answer: the guest DID build a
        // texture and this binding points at the wrong place in it (an interior member, or a
        // copy that missed the init). Naming the delta turns the fix into arithmetic.
        (None, Some((d, f))) => format!(
            "NO format was recorded here, but one ({f:#010x}) WAS recorded {d} bytes away - the \
             binding is off by that much, or this is an uninitialised COPY of that texture"
        ),
        (None, None) => {
            "NO format was ever recorded for this address, nor within 4 KiB of it".to_string()
        }
    };
    eprintln!(
        "gxm texture: handle {addr:#x} read as all-zero control words AT BIND TIME (the binding          captures them then, so this is what the guest handed to GXM, not what the memory          happens to hold now)"
    );
    eprintln!(
        "gxm texture: unit {unit} handle {addr:#x} (bound via {via}) reads as ALL ZERO control \
         words, and {init} - its data pointer is null, so the unit is bound to a 1x1 ZERO \
         texel (what the hardware would read at address 0)"
    );
}

/// Report - once per (unit, format) - that a bound texture's pixel bytes could not be read
/// from guest memory, so the unit is left EMPTY.
///
/// The other half of [`report_unsized_texture_format`]. A unit can go missing because the
/// format has no size OR because the bytes behind a perfectly-understood format are not
/// readable, and downstream both look identical: a recompiled shader falling back with
/// "sampler unit N wants Two but the bound units are [...]". Only the address and length
/// distinguish them, and only here are they still in hand.
fn report_unreadable_texture(unit: u32, base_format: u32, addr: u32, len: usize, w: u32, h: u32) {
    use std::collections::HashSet;
    use std::sync::Mutex;
    static SEEN: Mutex<Option<HashSet<(u32, u32)>>> = Mutex::new(None);
    let mut g = SEEN.lock().unwrap_or_else(|e| e.into_inner());
    if !g.get_or_insert_with(HashSet::new).insert((unit, base_format)) {
        return;
    }
    eprintln!(
        "gxm texture: unit {unit} format {base_format:#04x} {w}x{h} at {addr:#x} ({len} bytes) is \
         NOT READABLE from guest memory - the unit is left unbound, and any shader sampling it \
         falls back to fixed-function"
    );
}

/// Decode a bound `SceGxmTexture` (16 bytes, 4 control words) from guest memory
/// and snapshot its pixel bytes. Returns `None` for a null/unreadable handle or a
/// format whose byte size we do not know yet. The layout is the public GXM texture
/// control-word format (vitasdk `gxm.h` `struct SceGxmTexture`): word 1 holds
/// width/height (stored as size-1) and the base format, word 2 the data address.
/// The single 1x1 zero-RGBA buffer substituted for every null texture handle.
///
/// Shared rather than allocated per call so its ADDRESS is stable: an identity-keyed
/// consumer cache (`crate::render::tex_key`) would otherwise see a different texture every
/// draw. The bytes are constant, so one buffer is as correct as a million.
fn zero_texel() -> Arc<[u8]> {
    static ZERO: std::sync::OnceLock<Arc<[u8]>> = std::sync::OnceLock::new();
    ZERO.get_or_init(|| Arc::from([0u8, 0, 0, 0].as_slice())).clone()
}

/// The sampler state a texture's CONTROL WORD 0 carries, as
/// `(u addr mode, v addr mode, lod bias, min filter, mag filter, gamma)`.
///
/// Field positions per `vita::gxm::texword0` (vitasdk `gxm.h`). The one type-dependent case is
/// `min_filter`: for a `LINEAR_STRIDED` texture those bits are part of the stride, and the
/// header states that such a texture uses its MAG filter for minification too - so reading bits
/// 10..11 there would sample by a couple of stride bits instead of a filter mode.
fn word0_sampler_state(w0: u32, tex_type: u32) -> (u32, u32, u32, u32, u32, u32, u32) {
    const TYPE_LINEAR_STRIDED: u32 = 1;
    let mag = (w0 >> 12) & 0x3;
    let min = if tex_type == TYPE_LINEAR_STRIDED { mag } else { (w0 >> 10) & 0x3 };
    (
        (w0 >> 6) & 0x7,  // uaddr_mode
        (w0 >> 3) & 0x7,  // vaddr_mode
        (w0 >> 21) & 0x3f, // lod_bias
        min,
        mag,
        w0 & 0x1800_0000, // gamma_mode, in the enum's own already-shifted encoding
        (w0 >> 9) & 0x1,  // mip_filter (`vita::gxm::texword0::MIP_FILTER`)
    )
}

fn decode_texture(
    ctx: &GuestCtx,
    cache: &mut TextureSnapshots,
    binding: &TextureBinding,
    exact_format: Option<u32>,
    nearby: Option<(i64, u32)>,
) -> Option<crate::capture::BoundTexture> {
    let (unit, addr) = (binding.unit, binding.addr);
    if addr == 0 {
        return None;
    }
    // The control words come from the BINDING, captured when the guest handed the texture to
    // GXM, not from guest memory now. See `TextureBinding`.
    //
    // All four control words zero is not a texture - it is a handle that was never
    // initialised, or one whose memory is not readable (an unmapped read yields zero). Either
    // way the unit ends up unbound, and it is a completely different bug from a format we
    // cannot size, so it gets its own report rather than being folded into "not readable".
    if binding.is_null() {
        note_texture_drop(0);
        report_zero_texture_handle(ctx, binding, exact_format, nearby);
        // Bind a 1x1 ZERO texel rather than leaving the unit empty.
        //
        // Zero is not a neutral choice, it is the FAITHFUL one: a zeroed handle has a null
        // data pointer, so the hardware samples address 0 - and the texel it reads there is
        // zeros in every format. Dropping the unit instead costs the whole draw, because the
        // recompiled shader then cannot build its bind group at all and falls back to a
        // fixed-function approximation.
        //
        // It was briefly white, on the reasoning that white is the identity for a modulate.
        // That was wrong twice over: it is not what the hardware reads, and a title's
        // post-process chain sampled one of these and came out solid white, which then
        // composited over the entire race. A substitute has to be defensible as a VALUE, not
        // as a guess about how the shader will use it.
        return Some(crate::capture::BoundTexture {
            unit,
            base_format: 0x0c, // U8U8U8U8
            swizzle: 0,        // SWIZZLE4_ABGR - the identity permutation
            tex_type: 3,       // LINEAR
            width: 1,
            height: 1,
            stride: 4,
            faces: 1,
            face_bytes: 4,
            levels: 1,
            data_addr: 0,
            // ONE buffer for every null handle of the run, not one per draw. Consumers key
            // their caches on this buffer's IDENTITY (see `crate::render::tex_key`), so a
            // fresh allocation per draw would miss every time and churn a 512-entry decode
            // cache on a unit whose contents are four constant zero bytes.
            pixels: zero_texel(),
            // A zeroed handle's control words are zero, so every one of these is the GXM
            // default by construction - which is what the hardware would sample with too.
            u_addr_mode: 0,
            v_addr_mode: 0,
            lod_bias: 0,
            min_filter: 0,
            mag_filter: 0,
            mip_filter: 0,
            gamma: 0,
        });
    }

    // >>> THE WHOLE DECODE, INCLUDING THE PIXELS, IS DONE ONCE PER BINDING - and then only
    // RE-PROVEN, not redone, on the first use of each later scene (`decoded_validated`).
    //
    // [`TextureSnapshots::snapshot_sets`] already caches a whole draw's list by the SET of
    // bindings that produced it, and its doc comment says the hit rate is "essentially the
    // draw count". MEASURED on a racing title's on-track frame in the browser: it is
    // **53%**. A race
    // scene draws a few hundred objects out of a shared pool of textures, so the individual
    // BINDINGS repeat constantly while the combinations do not - and every set miss then paid
    // a full `get_or_read` per bound unit. That was 3,631 `get_or_read` calls per presented
    // frame, 86% of the fragment miss path and the largest single item in the capture.
    //
    // Memoising per binding is exactly equivalent for the same reason the set cache is:
    // `get_or_read` compares a texture at most once per SCENE and hands every later draw of
    // that scene the same `Arc` by construction, so a second decode of the same binding in the
    // same scene could only ever produce the identical answer. Cleared wherever
    // `snapshot_sets` is, plus wherever `templates` is - a recorded format is an input here
    // too.
    //
    // Keyed by the texture's ADDRESS as well as its control words: the recorded format and the
    // nearby-handle diagnostic are looked up by address, so two bindings with identical words
    // at different addresses are not interchangeable. `unit` is NOT in the key - it is a
    // property of the binding, not of the texture - so it is stamped onto the clone.
    let key = (addr, binding.words);
    let cached = match cache.decoded_validated(ctx, key) {
        Some(d) => d,
        None => {
            let entry = decode_texture_pixels(ctx, cache, binding, exact_format);
            let d = entry.res.clone();
            // The count bound, now that the memo outlives scenes. A set whose per-binding
            // entry vanishes here simply fails its next re-proof and rebuilds.
            if cache.decoded.len() >= TextureSnapshots::DECODED_CAP {
                cache.decoded.clear();
            }
            cache.decoded.insert(key, entry);
            d
        }
    };
    return match cached {
        Ok(mut t) => {
            t.unit = unit;
            Some(t)
        }
        // Noted on EVERY draw that binds it, not once per scene: the drop counters report how
        // many draws lost a texture, and a memo must not change how often a loss is counted.
        Err(code) => {
            note_texture_drop(code as usize);
            None
        }
    };
}

/// The decode itself - everything [`decode_texture`] memoises, returned with the facts the
/// cross-scene re-proof needs ([`DecodedEntry::snap`] / [`DecodedEntry::degraded`]). An
/// `Err(code)` result is a binding the decode DROPS, carrying the drop code its caller
/// reports.
fn decode_texture_pixels(
    ctx: &GuestCtx,
    cache: &mut TextureSnapshots,
    binding: &TextureBinding,
    exact_format: Option<u32>,
) -> DecodedEntry {
    let unit = binding.unit;
    // >>> EVERYTHING BELOW IS A PURE FUNCTION OF THE FOUR CONTROL WORDS, SO IT IS DONE ONCE.
    //
    // See [`TextureSnapshots::templates`]. The PIXELS are deliberately not part of it - they
    // still go through `get_or_read` on every draw, so a texture the guest rewrites is still
    // seen to change. Only the description is remembered.
    let template = match cache.templates.get(&binding.words) {
        Some(t) => *t,
        None => {
            let t = build_texture_template(unit, binding.words, exact_format);
            if cache.templates.len() >= TEXTURE_TEMPLATE_CAP {
                cache.templates.clear();
                cache.decoded.clear();
                // The finished lists were built FROM these templates, and they now outlive
                // the scene - so they die with them.
                cache.snapshot_sets.clear();
            }
            cache.templates.insert(binding.words, t);
            t
        }
    };
    let scene = cache.scene_seq;
    let entry = move |res, snap, degraded| DecodedEntry { res, snap, degraded, valid_scene: scene };
    let Some(t) = template else {
        return entry(Err(1), None, false);
    };
    let mut levels = t.levels;
    let mut face_bytes = t.face_bytes;
    let mut snap_len = t.read_len as usize;
    let mut degraded = false;
    let mut pixels = {
        let _r = crate::perf::scope(crate::perf::Phase::DrawTexRead);
        cache.get_or_read(ctx, t.data_addr, t.read_len as usize)
    };
    // >>> THE CHAIN READ CAN FAIL, AND FALLING BACK IS NOT SILENT. A texture whose allocation
    // really does end after level 0 (a render target sampled as a texture, say - it has no
    // mips whatever its control word says) makes the fuller read unmappable. That is a fact
    // about this texture, not an error, so it degrades to level 0 and SAYS SO once per format.
    if levels > 1 && pixels.len() < t.read_len as usize {
        report_mip_chain_unreadable(unit, t.base_format, t.data_addr, t.width, t.height, levels);
        levels = 1;
        face_bytes = t.level0_bytes;
        snap_len = t.level0_read_len as usize;
        degraded = true;
        pixels = cache.get_or_read(ctx, t.data_addr, t.level0_read_len as usize);
    }
    if pixels.is_empty() {
        report_unreadable_texture(unit, t.base_format, t.data_addr, t.read_len as usize, t.width, t.height);
        return entry(Err(2), None, false);
    }
    let snap = Some((t.data_addr, snap_len));
    return entry(Ok(crate::capture::BoundTexture {
        unit,
        base_format: t.base_format,
        swizzle: t.swizzle,
        tex_type: t.tex_type,
        width: t.width,
        height: t.height,
        stride: t.stride,
        faces: t.faces,
        face_bytes,
        levels,
        data_addr: t.data_addr,
        pixels,
        u_addr_mode: t.u_addr_mode,
        v_addr_mode: t.v_addr_mode,
        lod_bias: t.lod_bias,
        min_filter: t.min_filter,
        mag_filter: t.mag_filter,
        gamma: t.gamma,
        mip_filter: t.mip_filter,
    }), snap, degraded);
}

/// Derive everything a binding's four control words say about its texture. Called once per
/// distinct control-word set; see [`TextureSnapshots::templates`].
///
/// `None` is a binding the decode DROPS - a format that cannot be sized. It is memoised too,
/// because a dropped binding is re-bound every frame like any other and re-deriving the drop is
/// the same work as re-deriving the answer.
fn build_texture_template(
    unit: u32,
    words: [u32; 4],
    exact_format: Option<u32>,
) -> Option<TextureTemplate> {
    let [w0, w1, w2, w3] = words;
    let tex_type = (w1 >> 29) & 0x7;
    // The sampler state, out of the guest's own word 0 rather than a host-side map. See
    // [`word0_sampler_state`] and `vita::gxm::texword0`.
    let sampler = word0_sampler_state(w0, tex_type);
    // generic2 layout (non-swizzled/non-cube): width/height are 12-bit size-1.
    let width = ((w1 >> 12) & 0xfff) + 1;
    let height = (w1 & 0xfff) + 1;
    let data_addr = w2 & 0xffff_fffc;
    // Prefer the exact format the guest set (full high byte keeps 24-bit/paletted
    // formats and the channel swizzle intact); otherwise reconstruct the high byte
    // from the 5-bit field plus the format0 extension bit (word 0 bit 31), and take
    // the 3-bit control-word swizzle.
    // Both arms must produce the swizzle in the FORMAT FIELD's position (bits 14:12), because
    // that is what every consumer reads it out of (`(swizzle >> 12) & 0x7`). The control words
    // store it as a bare 3-bit value in word 3, so it has to be shifted back up - without that
    // every texture resolved from its control words alone silently reads as swizzle 0, and that
    // is the path a texture takes as soon as a title COPIES the struct it lives in (a colour
    // surface's `backgroundTex` is copied every time the surface is).
    let (base_format, swizzle) = match exact_format {
        Some(f) => ((f >> 24) & 0xff, f & 0x00ff_ffff),
        None => {
            let base = ((w1 >> 24) & 0x1f) | (((w0 >> 31) & 1) << 7);
            // Bits 28:30, per vitasdk's `SceGxmTexture` - see `vita::gxm::texword3`, which
            // records why reading bit 29 here was wrong yet passed every round-trip test.
            use crate::vita::gxm::texword3;
            let swz = (w3 >> texword3::SWIZZLE_SHIFT) & texword3::SWIZZLE_MASK;
            report_texture_resolved_from_control_words(unit, base, swz);
            (base, swz << 12)
        }
    };

    // Block geometry: uncompressed formats are 1x1 texel "blocks"; BC/DXT are 4x4. A format we
    // cannot size is dropped rather than guessed, but never silently: an undecoded unit shows up
    // downstream only as a missing sampler binding, which is far harder to trace back than the
    // format that caused it.
    // `stride` is the bytes per block-row we snapshot, and `level0` the bytes ONE level
    // occupies. The layout arithmetic lives in `render::level_layout` so the chain read below
    // and the level-0 read here cannot drift apart.
    let Some(l0) = crate::render::level_layout(base_format, tex_type, width, height, 0) else {
        report_unsized_texture_format(unit, base_format, tex_type, width, height);
        return None;
    };
    let stride = l0.stride;
    let level0 = l0.bytes;
    // A CUBE texture stores its six faces back to back, each laid out exactly like a standalone
    // texture of the same size - so one face is `level0` bytes and the snapshot is six of them.
    let faces = if crate::render::cube_type(tex_type) { 6 } else { 1 };
    // >>> HOW MANY MIP LEVELS TO READ, and why the count is CLAMPED rather than trusted.
    //
    // `mip_count` is control word 0 bits 17-20 (`vita::gxm::texword0::MIP_COUNT`), and whether
    // the driver packs the count or the count minus one is NOT settled by any clean source -
    // that ambiguity is recorded at `texword0`, and it is unobservable through the API because
    // both our writer and our reader use the same shift. It is observable HERE, from the data,
    // and `report_mip_chain_probe` is the instrument that measures it.
    //
    // Reading the field AS the count is the conservative choice: if the truth is count-minus-one
    // we stop one (1x1) level short, which costs a level nothing minifies far enough to sample,
    // whereas the other error reads past the guest's allocation.
    // >>> AND NOT READ AT ALL IF NOTHING CAN USE IT.
    //
    // The chain is ~33% more guest bytes per texture, and those bytes are re-COMPARED against
    // guest memory once per scene to decide whether the snapshot is stale - a compare this
    // project has measured at 40% of a frame. The only consumer today is the compressed upload
    // path, so on a GPU that cannot take compressed textures this would be a third more of the
    // most expensive loop in the capture, bought for nothing. Level 0 is unchanged either way,
    // so this changes no picture on any engine.
    let mip_field = if vitaslop_platform::gpu::block_compression_available() {
        (w0 >> 17) & 0xf
    } else {
        1
    };
    let max_levels = crate::render::max_mip_levels(width, height);
    let want_levels = if faces > 1 {
        // Six chains, and how they interleave (chain per face, or level across all faces) is
        // not established. Do not guess at it - level 0 of each face is what this path has
        // always read, and it stays exactly that until a title makes the question answerable.
        if mip_field > 1 {
            report_cube_mip_chain_skipped(unit, base_format, width, height, mip_field);
        }
        1
    } else {
        mip_field.clamp(1, max_levels)
    };
    // The chain's byte extent, and the level-0 fallback the caller drops to when the guest's
    // allocation turns out not to reach that far. `level_offset` walks every level, which is
    // exactly the sort of per-draw arithmetic this memo exists to stop repeating.
    let face_bytes = crate::render::level_offset(base_format, tex_type, width, height, want_levels)
        .unwrap_or(level0);
    Some(TextureTemplate {
        base_format,
        swizzle,
        tex_type,
        width,
        height,
        stride,
        faces,
        face_bytes,
        read_len: face_bytes * faces,
        levels: want_levels,
        level0_bytes: level0,
        level0_read_len: level0 * faces,
        data_addr,
        // Straight out of the binding's own control word 0, so this is the state the guest
        // handed to GXM for THIS texture - including a by-value copy, which an address-keyed
        // shadow could not follow.
        u_addr_mode: sampler.0,
        v_addr_mode: sampler.1,
        lod_bias: sampler.2,
        min_filter: sampler.3,
        mag_filter: sampler.4,
        gamma: sampler.5,
        mip_filter: sampler.6,
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

    /// The `(library_nid, func_nid)` pair a dense import index dispatches to, or
    /// `None` past the end of the table. A profiler counts host calls by the index
    /// the emitted wasm passes, so this is what turns those counts back into names.
    pub fn import_at(&self, index: u32) -> Option<(u32, u32)> {
        self.imports.get(index as usize).copied()
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

    /// Tell the host which guest thread is running, so a blocking primitive knows whom to
    /// park and a wake knows whom to release.
    ///
    /// The scheduler sets this when it PICKS a thread, before anything of that thread's is
    /// refreshed or resumed, and again before each dispatch. The first of those is the one
    /// that matters and it was missing: the current thread is mirrored into guest memory
    /// for the inlined lightweight-mutex take ([`crate::vita::mirror::SLOT_CURRENT_THREAD`]),
    /// so a build that only set it at dispatch time would refresh the block with the
    /// PREVIOUS thread's id and let a resumed thread take mutexes in its name until it
    /// happened to make a host call.
    fn set_current_thread(&mut self, _thid: i32) {}

    /// Settle any host state that was decided where no guest memory was reachable, and
    /// report how many items were settled.
    ///
    /// Called by the scheduler at the top of every drain - after a resume and after a clock
    /// advance, and always before a woken thread can run. Today this is the
    /// lightweight-mutex handoff owed to a cond wait that timed out; see
    /// [`VitaState::resolve_deferred_lwmutex`] for why that one cannot be a plain queued
    /// write. The default settles nothing, which is correct for a host with no such state.
    fn resolve_deferred(&mut self, _words: &mut dyn GuestWords) -> usize {
        0
    }

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

    /// Human-readable state of every thread and sync primitive, for a scheduler that
    /// wants to report a stall from inside its own loop.
    ///
    /// It lives on the trait rather than being reached through a concrete host because
    /// both schedulers are generic over the host, and a stall that happens on one engine
    /// and not the other can only be diagnosed by comparing the two dumps - which
    /// requires both to be able to produce one. The default is empty for a host with no
    /// sync state to report.
    fn sync_dump(&self) -> String {
        String::new()
    }

    /// Whether the host has a WAIT RECORD for `thid` - it is parked in some waiter queue.
    ///
    /// # Why a scheduler needs to ask
    /// A thread the SCHEDULER has marked blocked but that the HOST is not waiting on cannot
    /// be woken by anything: no signal, no timeout, no I/O completion names it. That is a
    /// LOST WAKEUP - a bug in the emulator - and it is not the same thing as a deadlock,
    /// where the waits are real and simply cannot be satisfied. Reported as one for years:
    /// a retail headless run stops at frame 1 with eight "blocked" threads, and the eighth
    /// is its render thread, parked by nothing at all.
    ///
    /// The default is `true` (assume a record exists) so a host that cannot answer never
    /// turns a real deadlock into a spurious lost-wakeup report.
    fn thread_has_wait_record(&self, _thid: i32) -> bool {
        true
    }

    /// What `thid` is waiting ON, in one short phrase ("RUNNABLE" if nothing).
    ///
    /// The whole-machine [`sync_dump`](Self::sync_dump) answers this for every thread at
    /// one instant; this answers it for ONE thread at the instant it parks, which is what
    /// a scheduling TIMELINE needs. A timeline of picks and blocks without the reason says
    /// only that a thread stopped running - and "it was descheduled" and "it is waiting on
    /// a semaphore the other thread signals once a frame" are the two readings such a
    /// timeline exists to separate.
    fn thread_wait_reason(&self, _thid: i32) -> String {
        String::new()
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

    /// The virtual clock now, in microseconds. The scheduler compares it against
    /// [`earliest_deadline`](Self::earliest_deadline) to tell a wait that will buy time
    /// from one that will not (a zero-length delay re-armed every iteration).
    fn clock_us(&self) -> u64 {
        0
    }

    /// Nothing is runnable and no timed wait completes sooner: complete the earliest
    /// outstanding modelled storage transfer, waking its reader. Returns whether anything
    /// was released. See [`VitaState::release_earliest_io`].
    fn release_earliest_io(&mut self) -> bool {
        false
    }

    /// Storage time the earliest outstanding transfer still owes, for the idle path to
    /// compare against the next timed wait. See [`VitaState::earliest_io_remaining_us`].
    fn earliest_io_remaining_us(&self) -> Option<u64> {
        None
    }

    /// Credit the modelled storage device for an idle interval the scheduler just jumped
    /// the game clock over. See [`VitaState::charge_io_idle`].
    fn charge_io_idle(&mut self, _us: u64) {}

    /// A thread just suspended having burned `fuel` units of guest execution. Lets the
    /// host charge clocks that track executed work rather than rendered frames - the game
    /// clock and the modelled storage clock. The default ignores it.
    ///
    /// This is called for EVERY suspend, not only a preemption: a thread that blocks or
    /// flips did guest work on the way there and must be billed for it, and a thread that
    /// yielded immediately must not be billed for a quantum it never used.
    ///
    /// `runnable` is how many threads were ready to run at that moment, INCLUDING the
    /// one that just stopped. The scheduler runs one at a time; the device does not, so
    /// this is what lets the host divide the WALL time among the cores that would have
    /// been executing it in parallel. See [`VitaEnv::on_guest_work`].
    fn on_guest_work(&mut self, _runnable: usize, _fuel: u64, _retired: Option<u64>) {}

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

    /// Write the current value of every HOST MIRROR slot through `write`, which stores
    /// one word at a slot index. Returns how many slots were written.
    ///
    /// Called by the scheduler immediately before guest code resumes, which is what
    /// makes an inlined read of the block exactly equal to the host call it replaces -
    /// see [`crate::vita::mirror`] for the rule about what may live there. The default
    /// writes nothing and returns 0, which is correct for a host with no mirrored
    /// values and is DETECTED (not tolerated) for one that needs them.
    fn refresh_mirror(&mut self, _write: &mut dyn FnMut(u32, u32)) -> usize {
        0
    }
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
        // The transpiler's software fuel point, not a NID: this thread has used its
        // quantum and is asking to be rescheduled. Intercepted before the import table
        // because it is not IN the import table - it is a reserved selector above every
        // real index (see `vitaslop_transpiler::abi::FUEL_SELECTOR`).
        //
        // Deliberately NOT recorded as a call: it is host bookkeeping the guest never
        // asked for, and folding it into the per-NID histogram would put a synthetic
        // entry at the top of every profile.
        if index == vitaslop_transpiler::abi::FUEL_SELECTOR {
            return SvcOutcome::Reschedule;
        }
        // >>> THE VBLANK SPIN, PARKED. Also not a NID, and not recorded as a call.
        //
        // The emitted `sceDisplayGetVcount` has taken `mirror::SPIN_BUDGET` reads inside one
        // resume. The mirror cannot change while guest code runs, so every one of those reads
        // returned the same word: this thread is in a loop it cannot leave until the clock
        // moves, and while it is runnable the scheduler will not move it - the loop's own
        // fuel is what drags the clock to the next vblank. So do what the loop is asking for
        // and park it there, which is the same wait `sceDisplayWaitVblankStart` performs.
        //
        // The budget needs no reset here: the block is refreshed before EVERY resume, which
        // is the same contract that makes the mirror readable at all, so the thread comes
        // back with a full budget by construction. A host that broke that contract would
        // leave the counter running negative and simply stop parking - the behaviour this
        // guard replaced, which is the right way for it to fail.
        if index == vitaslop_transpiler::abi::VBLANK_PARK_SELECTOR {
            // Same guard the real vblank wait carries: in the SINGLE-THREAD model there is
            // nothing to yield to and the clock is host-driven, so a park would block on a
            // wake nothing can deliver. There the spin is left exactly as it was.
            if !self.state.is_preemptive() {
                return SvcOutcome::Continue;
            }
            self.state.note_vblank_spin_park();
            self.state.vblank_park(1, vita::display::VBLANK_US);
            return SvcOutcome::Block;
        }
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

    fn resolve_deferred(&mut self, words: &mut dyn GuestWords) -> usize {
        self.state.resolve_deferred_lwmutex(words)
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

    fn sync_dump(&self) -> String {
        self.state.debug_sync_dump()
    }

    fn thread_has_wait_record(&self, thid: i32) -> bool {
        self.state.thread_wait_state(thid) != "RUNNABLE"
    }

    fn thread_wait_reason(&self, thid: i32) -> String {
        self.state.thread_wait_state(thid)
    }

    fn earliest_deadline(&self) -> Option<u64> {
        self.state.earliest_lwcond_deadline()
    }

    fn advance_time_to(&mut self, to_us: u64) {
        self.state.advance_time_to(to_us);
    }

    fn clock_us(&self) -> u64 {
        self.state.now_us()
    }

    fn release_earliest_io(&mut self) -> bool {
        self.state.release_earliest_io()
    }

    fn earliest_io_remaining_us(&self) -> Option<u64> {
        self.state.earliest_io_remaining_us()
    }

    fn charge_io_idle(&mut self, us: u64) {
        self.state.charge_io_idle(us);
    }

    /// # A quantum of guest EXECUTION is not a quantum of WALL time
    /// The scheduler has one baton: it resumes a thread, that thread retires a quantum
    /// of guest code, and the clock is charged for it. The Vita has three CPUs for the
    /// game, so three threads retire their quanta AT ONCE and only ONE quantum of wall
    /// time passes while they do. Charging the full quantum per thread makes the game
    /// clock run at the total rate of every runnable thread, which turns a guest thread
    /// SPINNING - waiting on nothing, retiring instructions - into game time it never
    /// costs on hardware.
    ///
    /// Measured on this title's event load: the load costs ~230 s of GAME clock against
    /// **54 s of wall clock** - the frames themselves run at 2.83 ms each - and `bench`
    /// reports 18 thread resumes a frame, "one per 0 host calls". Threads burning whole
    /// quanta in guest code, waiting on nothing, and the clock billing every one of them
    /// as if it had a machine to itself.
    ///
    /// So the charge is the quantum divided by the cores that would have been busy:
    /// `runnable` of them, capped at [`GUEST_CORES`], and never fewer than one (a lone
    /// runnable thread really does have the machine to itself, and the other two cores
    /// are idle). The STORAGE clock divides by exactly the same figure and for exactly
    /// the same reason - it charges for elapsed wall time too, and the device does not
    /// get three times the bandwidth because three threads are running.
    fn on_guest_work(&mut self, runnable: usize, fuel: u64, retired: Option<u64>) {
        let cores = runnable.clamp(1, guest_cores()) as u64;
        // >>> A QUANTUM OF GUEST WORK, IN THE UNIT THE ENGINE CAN REPORT.
        //
        // GUEST INSTRUCTIONS when the engine carries the emitted per-block counter, which
        // is the whole point: an ARM instruction is the same amount of guest work whatever
        // this transpiler's codegen turns it into, so the emulated console's speed stops
        // moving every time the codegen improves. Fuel (wasm operators) is the fallback for
        // an engine with no such counter, and is exactly the pre-2026-08-16 behaviour.
        let (work, quantum) = match retired {
            Some(r) => (r, QUANTUM_ARM),
            None => (fuel, QUANTUM_FUEL),
        };
        // A quantum's worth of fuel costs a quantum's worth of time, so a thread that
        // really does burn a whole preemption slice is charged exactly what it always was.
        // What changes is everything either side of that: a voluntary yield after a
        // hundred instructions now costs a hundred instructions, and a thread that blocks
        // is billed for the work it did before blocking instead of for nothing.
        // Divide ONCE, carrying the remainder (see `VitaState::charge_rem`). Dividing
        // twice - by `QUANTUM_FUEL` and then by `cores` - truncated a small burn to zero
        // microseconds and stopped the clock outright for a title that yields in short
        // bursts, which is a livelock and not a rounding error.
        let den = quantum.saturating_mul(cores);
        let charge = |num: u64, rem: &mut u64| -> u64 {
            let total = rem.saturating_add(num);
            *rem = total % den;
            total / den
        };
        let (mut cpu_rem, mut io_rem) = self.state.charge_rem;
        let io_us = charge(QUANTUM_IO_US.saturating_mul(work), &mut io_rem);
        let cpu_us = charge(quantum_cpu_us().saturating_mul(work), &mut cpu_rem);
        self.state.charge_rem = (cpu_rem, io_rem);
        if self.state.has_io_waiters() {
            self.state.charge_io_quantum(io_us);
        }
        self.state.charge_cpu_quantum(cpu_us);
    }

    fn take_resume_code(&mut self, thid: i32) -> Option<u32> {
        self.state.take_resume_code(thid)
    }

    fn refresh_mirror(&mut self, write: &mut dyn FnMut(u32, u32)) -> usize {
        let words = vita::mirror::snapshot(&self.state);
        for (slot, value) in words.iter().enumerate() {
            write(slot as u32, *value);
        }
        words.len()
    }

    fn on_frame_boundary(&mut self, frame: u64) {
        // Re-arm the per-frame texture comparison cadence - see `TextureSnapshots`.
        self.state.texture_snapshots.begin_frame();
        self.state.world.set_frame(frame);
        self.state.set_cur_frame(frame);
        // Close the capture's frame so an observer can ask for the scenes THIS frame was
        // built from rather than for the last scene submitted, which on a multi-pass
        // title is the composite and holds no world geometry at all.
        self.state.capture.end_frame();
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
        //
        // It is a TOP-UP, not an unconditional addition: `charge_cpu_quantum` has
        // already advanced the clock for the guest CPU work this frame took, and adding
        // a whole frame on top of that would run the game clock fast (the failure mode
        // in `vitaslop-race-clock-5x`).
        const FRAME_US: u64 = 1_000_000 / 60;
        self.state.advance_time_frame(FRAME_US);
        // The modelled storage device gets one frame of progress per rendered frame,
        // net of anything the quantum charges already contributed.
        self.state.advance_io_frame(FRAME_US);
    }
}

/// The preemption quantum, in units of engine FUEL. Both engines must use this as their
/// preemption granularity, and it is the denominator every per-quantum constant below is
/// expressed against: a suspend that burned this much fuel costs exactly one quantum's
/// worth of clock, and a suspend that burned a tenth of it costs a tenth.
///
/// It lives here rather than in an engine because it is a UNIT, not a policy. The two
/// engines measure fuel the same way - it is proportional to executed wasm on both - so
/// the same guest work must cost the same game time whichever one ran it, and that only
/// holds while both divide by the same number.
///
/// **5,000,000 is what every real consumer preempts at** - the desktop's retail runner, the
/// resident session, the recipe runner, and the browser's software-fuel interval (whose own
/// knob is documented where it is read, not here: naming it in this comment makes the generated
/// knob index attribute it to a constant rather than to the code that reads it). A 1,000,000
/// default also exists on two constructors nothing production uses, and taking the calibration
/// from THAT one made the game clock five times fast; the run that caught it reported a single
/// suspend burning 5,000,000 against a stated interval of 1,000,000, which is impossible when
/// the engine preempts at the interval. The headless run's `fuel ... max` line exists for
/// exactly that check - a max above the interval is a broken reading, not a busy title.
pub const QUANTUM_FUEL: u64 = 5_000_000;

/// Storage-clock time charged for one [`QUANTUM_ARM`] of guest execution, in microseconds.
///
/// # It is the SAME elapsed time the game clock charges, and that is not a coincidence
/// The storage clock ([`VitaState::advance_io_by`]) is a second clock in real microseconds:
/// a transfer parks until `io_us` reaches a deadline derived from the device's bandwidth. So
/// "how much time passed while the guest burned a quantum" has exactly one answer, and both
/// clocks must charge it. Two clocks measuring the same elapsed time at different rates means
/// the emulated console's storage and its CPU disagree about how long a second is - which
/// shows up as loads that complete at the wrong point in a title's own timeline, with nothing
/// in a run to say so.
///
/// It was an independent 2000 us until 2026-08-17b, when the game clock stopped being fitted
/// and the two drifted apart by 1.75x. The exact figure only matters when a title spins in
/// guest code waiting for a load instead of rendering, since a rendering title's storage
/// progress is pinned to its frames by [`VitaState::advance_io_frame`] - but that case is
/// exactly a loading screen, which is where every title spends its first minute.
const QUANTUM_IO_US: u64 = QUANTUM_CPU_US;

/// The device's CPU clock. The Vita's application processor is a quad-core ARM
/// Cortex-A9 MPCore; 444 MHz is the clock a game runs at (the SoC can be driven higher,
/// but a title's own `sceKernelGetProcessTime` rate and every published figure for this
/// console are at 444).
const GUEST_CPU_HZ: u64 = 444_000_000;
/// Guest ARM instructions the device retires per 1000 CYCLES - IPC in thousandths.
///
/// # This is the one judgement in the clock, and it is a claim about the DEVICE
/// Everything else in the emulated CPU's speed is now measured: the guest's instruction
/// count comes from the emitted counter and the clock rate is 444 MHz. What remains is
/// how many of those instructions a real A9 retires per cycle, and that cannot be read off
/// a datasheet for game code.
///
/// **1.000 is the model taken: one instruction per cycle.** The Cortex-A9 is a
/// dual-issue, out-of-order core, so its ceiling is 2.0 and this is well inside what the
/// hardware can sustain rather than an overstatement of it. It is also the simplest
/// statement that can be made about the device - a 444 MHz core retiring an instruction a
/// cycle - which matters for a constant that no measurement on this machine can settle.
///
/// A slower model is one edit away and its consequences are exactly proportional: 0.75
/// gives 1521 us against this 1141, i.e. an emulated console a third slower. If a real
/// device measurement ever becomes available (a title's own frame pacing against the
/// console's, taken on hardware), this is the constant it lands in.
const GUEST_IPC_MILLI: u64 = 1_000;

/// Game-clock time charged for one [`QUANTUM_ARM`] of guest execution, in microseconds.
/// See [`VitaState::charge_cpu_quantum`] for why the game clock must advance for CPU work
/// at all.
///
/// # >>> DERIVED FROM THE DEVICE, NOT FITTED TO A RUN
/// `QUANTUM_ARM` guest instructions, a 444 MHz core, and an instruction per cycle. That is
/// the whole derivation, and every term in it is either measured or a stated claim about
/// the hardware. Nothing here is tuned to make a particular title behave.
///
/// **What it replaced, so it is not reintroduced: a hand-fitted 2780.** The clock used to
/// be billed in wasm OPERATORS, so the emulated console's speed was `fuel rate / code
/// expansion` - and the expansion is a property of this transpiler's codegen. Every
/// codegen improvement therefore made the emulated Vita faster, silently, and had to be
/// undone by hand: one session cut executed operators 28% and moved this constant 2000 ->
/// 2780 to put the speed back. That compensation could only ever be right for the
/// instruction mix it was measured on (the static factor said 1.391 where the race window
/// said 1.316), it had to be redone after every codegen change, and it left the emulated
/// console running at 182 MIPS - **2.4x slower than the device it models**.
///
/// It also had a correctness cost that was measured before it was removed. One title's
/// null-pointer dereference at a fixed frame moved with NOTHING except this constant: a
/// 33x sweep of I/O latency and a 4x sweep of the preemption quantum did not shift the
/// crash by a single frame, while 2000 and 4000 both made it disappear and only the fitted
/// 2780 reproduced it. A fitted timing constant does not merely mismodel the device, it
/// puts the guest into interleavings the device never produces.
///
/// **This is a rate, not a per-suspend price.** It was the latter until the game clock ran
/// 1.08x on one title and 4.34x on another with the same build on the same day, which is
/// the signature of billing a fixed amount for a variable thing. Charging per unit of
/// guest work is what makes one constant able to fit both.
///
/// The integer microsecond truncates 1140.96 to 1140, so the emulated core runs 0.08%
/// fast. That is three orders of magnitude inside the uncertainty on
/// [`GUEST_IPC_MILLI`], and the alternative is a finer time unit than the clock's own
/// consumers use.
///
/// Override for an experiment with `VITASLOP_QUANTUM_CPU_US`; 0 restores the old model
/// (a game clock that moves only on a flip or a scheduler idle) for an A/B.
const QUANTUM_CPU_US: u64 =
    QUANTUM_ARM * 1_000 * 1_000_000 / (GUEST_CPU_HZ * GUEST_IPC_MILLI);

/// GUEST ARM INSTRUCTIONS in one quantum of guest work - the unit [`QUANTUM_CPU_US`] and
/// [`QUANTUM_IO_US`] are priced against, on any engine that carries the emitted per-block
/// instruction counter (`abi::ARM_COUNT_GLOBAL`).
///
/// # It is a UNIT, and only the ratio to [`QUANTUM_CPU_US`] means anything
/// The pair states one rate: `QUANTUM_ARM` guest instructions cost `QUANTUM_CPU_US`
/// microseconds of game time, so the emulated core retires 444 million instructions per
/// emulated second. Scaling both by the same factor changes nothing.
///
/// The particular value is historical and harmless: it is `QUANTUM_FUEL / 9.87`, the code
/// expansion measured on the day the clock stopped being billed in wasm OPERATORS, chosen
/// so that changeover was exactly speed-neutral. What it is NOT any more is a speed - see
/// [`QUANTUM_CPU_US`], which now derives that from the device.
///
/// # Both engines carry the counter this is measured in
/// The browser has emitted the per-block instruction count since 2026-08-16 and native
/// since 2026-08-17 (`threaded::WasmtimeThread::arm_retired`). So the two engines charge
/// identical guest work identically, and neither of them moves when the codegen does -
/// which is what makes a codegen A/B measurable at all, on either engine.
const QUANTUM_ARM: u64 = 506_586;

/// [`QUANTUM_CPU_US`], overridable per-run by `VITASLOP_QUANTUM_CPU_US` (the knob exists
/// to take the calibration measurement and to A/B it; 0 restores the pre-calibration
/// behaviour of a game clock that only moves on a flip or an idle).
fn quantum_cpu_us() -> u64 {
    static CELL: std::sync::OnceLock<u64> = std::sync::OnceLock::new();
    *CELL.get_or_init(|| {
        std::env::var("VITASLOP_QUANTUM_CPU_US")
            .ok()
            .and_then(|s| s.parse().ok())
            .unwrap_or(QUANTUM_CPU_US)
    })
}

/// CPU cores a Vita gives a GAME. The console has four Cortex-A9 cores; the system
/// software reserves one, so a title's threads run three at a time. See
/// [`VitaEnv::on_quantum`], which is the only thing this number means to us: how much
/// wall time one scheduler quantum of guest execution is worth.
///
/// It is a property of the DEVICE, not a tuning constant. Override it with
/// `VITASLOP_GUEST_CORES` only to A/B the model; 1 restores the one-baton clock.
const GUEST_CORES: usize = 3;

/// [`GUEST_CORES`], overridable per-run by `VITASLOP_GUEST_CORES` for an A/B.
pub fn guest_cores() -> usize {
    static CELL: std::sync::OnceLock<usize> = std::sync::OnceLock::new();
    *CELL.get_or_init(|| {
        crate::knobs::var("VITASLOP_GUEST_CORES")
            .ok()
            .and_then(|s| s.parse().ok())
            .filter(|&n| n >= 1)
            .unwrap_or(GUEST_CORES)
    })
}

/// `VITASLOP_FRAME_TOPUP=1`: top the game clock up to a full frame at a display flip.
/// **OFF by default, and it should stay off** - see [`VitaState::advance_time_frame`].
///
/// It is superseded by [`VitaState::pace_flip`], which parks the flipping thread until
/// the scanout can latch. The two cannot both be on: the top-up runs at the frame
/// boundary, AFTER the flip's park, so it would add a second frame of clock on top of
/// the one the guest just waited out and run the game clock at double rate. Turning it
/// on also silently disables pacing - the top-up leaves the clock past the vblank floor,
/// so the next flip never has anything to wait for.
///
/// Kept as a knob only so the old behaviour can be measured against the new one.
fn frame_topup() -> bool {
    static CELL: std::sync::OnceLock<bool> = std::sync::OnceLock::new();
    *CELL.get_or_init(|| crate::knobs::var("VITASLOP_FRAME_TOPUP").as_deref() == Ok("1"))
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

    fn resolve_deferred(&mut self, words: &mut dyn GuestWords) -> usize {
        self.borrow_mut().resolve_deferred(words)
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

    fn clock_us(&self) -> u64 {
        self.borrow().clock_us()
    }

    fn release_earliest_io(&mut self) -> bool {
        self.borrow_mut().release_earliest_io()
    }

    fn earliest_io_remaining_us(&self) -> Option<u64> {
        self.borrow().earliest_io_remaining_us()
    }

    fn charge_io_idle(&mut self, us: u64) {
        self.borrow_mut().charge_io_idle(us);
    }

    fn on_guest_work(&mut self, runnable: usize, fuel: u64, retired: Option<u64>) {
        self.borrow_mut().on_guest_work(runnable, fuel, retired);
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
mod program_reflection_tests {
    //! The reflected extent of a program's default uniform buffer, in 32-bit registers.
    //!
    //! This is the size the capture reads out of the guest's reserved buffer, and the
    //! buffer is a RECYCLED per-scene ring - so an over-read does not return zeros, it
    //! returns a neighbouring draw's uniforms. That is a wrong answer wearing the shape
    //! of a real one, which is why the width table is tested per type rather than on the
    //! one program that exposed it.
    use super::*;

    /// Build a `SceGxmProgram` container just complete enough for
    /// [`reflect_program_uncached`]: the parameter count at +0x24, the table offset at
    /// +0x28, and one 16-byte record per parameter.
    ///
    /// A record is `{ +0 name offset (relative to the record), +4 packed type word,
    /// +8 array size, +0xc resource index }`, and the packed word is
    /// `category | type << 4 | components << 8`.
    fn program_with_uniforms(params: &[(u32, u32, u32, u32)]) -> Vec<u8> {
        const TABLE: u32 = 0x40;
        let mut bytes = vec![0u8; (TABLE as usize) + params.len() * 16 + 64];
        let put = |b: &mut Vec<u8>, at: u32, v: u32| {
            b[at as usize..at as usize + 4].copy_from_slice(&v.to_le_bytes());
        };
        put(&mut bytes, 0x24, params.len() as u32);
        // The table offset is stored RELATIVE to the field that holds it.
        put(&mut bytes, 0x28, TABLE - 0x28);
        for (i, &(ptype, comp, array, res)) in params.iter().enumerate() {
            let p = TABLE + i as u32 * 16;
            // Name offset 0 points the name reader at the record itself, which is a
            // zero byte - an empty name. None of the name-matched fields are under
            // test here, and an empty name cannot accidentally match one.
            put(&mut bytes, p, 0);
            put(&mut bytes, p + 4, 1 | (ptype << 4) | (comp << 8));
            put(&mut bytes, p + 8, array);
            put(&mut bytes, p + 0xc, res);
        }
        bytes
    }

    fn uniform_bytes(params: &[(u32, u32, u32, u32)]) -> u32 {
        let mut bytes = program_with_uniforms(params);
        let mut regs = [0u32; REG_COUNT];
        let mut vfp = [0u32; VFP_ARG_COUNT];
        let mut mem = SliceMemory(&mut bytes);
        let ctx = GuestCtx::new(&mut regs, &mut vfp, &mut mem, 0);
        reflect_program_uncached(&ctx, 0).uniform_size_bytes
    }

    #[test]
    fn f32_parameters_take_one_register_per_component() {
        // A 4x4 matrix is declared components 4, array 4 - sixteen registers.
        assert_eq!(uniform_bytes(&[(0, 4, 4, 0)]), 64);
        // ...and one starting at register 16 ends at 32.
        assert_eq!(uniform_bytes(&[(0, 4, 4, 0), (0, 4, 4, 16)]), 128);
    }

    #[test]
    fn f16_parameters_pack_two_components_per_register() {
        // THE CASE THAT WAS WRONG. This title's tone-map fragment program declares
        // `exposure` F16[4] at register 0 and `luminanceFactor` F16[4] at register 2,
        // and its own header says the buffer is 4 registers / 16 bytes. Counting a
        // component as a register reported 24, and the extra 8 bytes were read out of
        // the uniform RING - the previous draw's values, which drift smoothly frame
        // over frame and read exactly like a runaway multiplier the guest is computing.
        assert_eq!(uniform_bytes(&[(1, 4, 1, 0), (1, 4, 1, 2)]), 16);
        // An F16 pair is one register; an odd component count still rounds up to one.
        assert_eq!(uniform_bytes(&[(1, 2, 1, 0)]), 4);
        assert_eq!(uniform_bytes(&[(1, 1, 1, 0)]), 4);
        assert_eq!(uniform_bytes(&[(1, 3, 1, 5)]), 28);
    }

    #[test]
    fn narrow_integer_parameters_use_their_own_widths() {
        // U16/S16/C10 pack two per register, U8/S8 four - the same table as
        // `ParamType::component_bytes`.
        assert_eq!(uniform_bytes(&[(5, 4, 1, 0)]), 8);
        assert_eq!(uniform_bytes(&[(6, 4, 1, 0)]), 8);
        assert_eq!(uniform_bytes(&[(2, 4, 1, 0)]), 8);
        assert_eq!(uniform_bytes(&[(7, 4, 1, 0)]), 4);
        assert_eq!(uniform_bytes(&[(8, 8, 1, 0)]), 8);
    }

    #[test]
    fn an_unknown_type_is_sized_as_wide_as_possible() {
        // A type nibble this table does not know gets one register per component. That
        // can only ever OVER-size, and an over-size is recoverable where a truncated
        // buffer silently drops the tail of a matrix.
        assert_eq!(uniform_bytes(&[(9, 4, 1, 0)]), 16);
        assert_eq!(uniform_bytes(&[(0xf, 2, 2, 0)]), 16);
    }

    #[test]
    fn the_extent_is_the_maximum_not_the_sum() {
        // Two parameters that OVERLAP (a title may alias a block) size the buffer by
        // the furthest reach, not by adding them up.
        assert_eq!(uniform_bytes(&[(0, 4, 1, 0), (0, 4, 1, 0)]), 16);
        // ...and order in the table does not matter.
        assert_eq!(uniform_bytes(&[(0, 4, 1, 8), (0, 4, 1, 0)]), 48);
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
mod register_window_tests {
    //! >>> THE HOST-CALL REGISTER WINDOW IS A CONTRACT, AND THIS IS WHAT ENFORCES IT.
    //!
    //! The browser marshals only the registers AAPCS lets a host call reach
    //! (`browser_sched::NARROW_REGS`: r0..r3, sp, lr, pc). Reading r4..r11 through a live
    //! `GuestCtx` would therefore hand back ZERO rather than the guest's value - a silent
    //! wrong answer on the one engine that ships, and invisible on the desktop, which
    //! marshals the whole file.
    //!
    //! A comment cannot hold that. This scans the workspace for the shape that would break
    //! it, in the same way `knobs::a_knob_routed_through_this_module_is_reachable_from_the_browser`
    //! scans for an unregistered knob - and for the same reason: the failure is silent, so it
    //! has to be caught where it is written rather than where it shows.

    /// Register indices a host call may read through a `GuestCtx`.
    const WINDOW: &[usize] = &[0, 1, 2, 3, 13, 14, 15];

    #[test]
    fn a_host_handler_only_reads_registers_the_call_marshals() {
        let root = crate::knobs::workspace_root();
        let mut bad = Vec::new();
        for path in crate::knobs::rust_sources(&root) {
            let rel = path
                .strip_prefix(&root)
                .unwrap_or(&path)
                .to_string_lossy()
                .replace('\\', "/");
            // The browser scheduler owns the window and names it; the transpiler and the
            // native engine hold whole register FILES of their own, which are not this.
            if !rel.starts_with("vitaslop-runtime/src/vita/")
                && !rel.ends_with("vitaslop-runtime/src/host.rs")
            {
                continue;
            }
            let Ok(text) = std::fs::read_to_string(&path) else { continue };
            for (i, line) in text.lines().enumerate() {
                // `ctx.regs[N]` / `self.regs[N]` with a literal index. A non-literal index is
                // either the AAPCS argument cursor (bounded to 0..4 where it is written) or a
                // loop over a caller's OWN array, neither of which is a live-context read.
                for m in line.match_indices(".regs[") {
                    let rest = &line[m.0 + ".regs[".len()..];
                    let digits: String =
                        rest.chars().take_while(|c| c.is_ascii_digit()).collect();
                    let Ok(n) = digits.parse::<usize>() else { continue };
                    // A range (`regs[0..13]`) is spelt with a literal too; treat its start as
                    // the index and let the end be caught by the same rule on the next scan.
                    if !WINDOW.contains(&n) {
                        bad.push(format!("{rel}:{} reads r{n}", i + 1));
                    }
                    if let Some(end) = rest.strip_prefix(&digits).and_then(|r| r.strip_prefix(".."))
                    {
                        let e: String = end.chars().take_while(|c| c.is_ascii_digit()).collect();
                        if let Ok(e) = e.parse::<usize>() {
                            if (n..e).any(|k| !WINDOW.contains(&k)) {
                                bad.push(format!("{rel}:{} reads r{n}..r{e}", i + 1));
                            }
                        }
                    }
                }
            }
        }
        bad.sort();
        bad.dedup();
        assert!(
            bad.is_empty(),
            "these read a register the BROWSER does not marshal into a host call, so they \
             would read ZERO there and the guest's value on the desktop:\n  {}\n\
             Either use one of r{WINDOW:?}, or widen `browser_sched::NARROW_REGS` and this \
             list together.",
            bad.join("\n  ")
        );
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

    /// Guest memory for the tests that touch guest-resident state, as a sparse word map -
    /// a lightweight mutex lives in the guest's own work area now, so a test of one needs
    /// somewhere for it to live.
    #[derive(Default)]
    pub(super) struct Words(std::collections::BTreeMap<u32, u32>);

    impl GuestWords for Words {
        fn word(&self, addr: u32) -> u32 {
            self.0.get(&addr).copied().unwrap_or(0)
        }
        fn set_word(&mut self, addr: u32, value: u32) {
            self.0.insert(addr, value);
        }
    }

    /// A work area laid out as a created lightweight mutex, the way
    /// `sceKernelCreateLwMutex` leaves it.
    fn created_lwmutex(st: &mut VitaState, w: &mut Words, work: u32) {
        st.lwmutex_register(w, work);
    }

    /// The MOST RECENT park a thread registered, in microseconds from now. The last one,
    /// not the first: `sleep_waiters` accumulates within a test, and the first entry is
    /// whatever an earlier step parked on.
    fn parked_us(st: &VitaState, thid: i32) -> u64 {
        let (_, deadline) = st
            .sleep_waiters
            .iter()
            .rev()
            .find(|(t, _)| *t == thid)
            .copied()
            .expect("thread is not parked");
        deadline.saturating_sub(st.now_us())
    }

    #[test]
    fn a_flip_latches_on_a_vblank_edge_not_a_period_after_the_request() {
        const FRAME_US: u64 = 1_000_000 / 60;
        let mut st = state();
        st.set_current(1);
        // A frame whose work took 5 ms latches at the vblank, 11.67 ms away - NOT a
        // whole period later. That is what phase-locks a cheap frame to 60 Hz instead
        // of letting it drift by its own render time every frame.
        st.advance_time_to(5_000);
        assert_eq!(st.pace_flip(FRAME_US), FRAME_US - 5_000);
        assert_eq!(parked_us(&st, 1), FRAME_US - 5_000);
        // A frame that overruns its period MISSES the vblank and waits for the next
        // one, so it costs two whole periods - the console's 60/30 quantisation, not a
        // continuum. Here the guest lands 1 ms past the 2nd edge and latches at the 3rd.
        st.advance_time_to(2 * FRAME_US + 1_000);
        assert_eq!(st.pace_flip(FRAME_US), 3 * FRAME_US - (2 * FRAME_US + 1_000));
    }

    #[test]
    fn the_setbuf_sync_mode_is_recorded_and_changes_no_pacing() {
        const FRAME_US: u64 = 1_000_000 / 60;
        let mut st = state();
        st.set_current(1);
        assert_eq!(st.display_sync(), VitaState::SETBUF_NEXTFRAME, "the safe default");
        st.advance_time_to(5_000);
        let paced = st.pace_flip(FRAME_US);
        assert!(paced > 0);
        // IMMEDIATE decides whether the scanout's pointer changes MID-SCAN, which tears.
        // It is not a swap interval and it does not let a title present faster, so the
        // vblank floor is unchanged by it. Both readings that said otherwise were tried
        // and both broke the one retail title that asks for it - see `set_display_sync`.
        st.set_display_sync(VitaState::SETBUF_IMMEDIATE);
        assert_eq!(st.display_sync(), VitaState::SETBUF_IMMEDIATE, "recorded");
        // The second flip is reserved for the vblank after the first, and asking for
        // IMMEDIATE does not release it: the floor is the panel, not the mode.
        st.advance_time_to(10_000);
        assert_eq!(st.pace_flip(FRAME_US), 2 * FRAME_US - 10_000);
        let _ = paced;
    }

    #[test]
    fn a_vblank_wait_joins_the_heartbeat_and_never_returns_instantly() {
        const FRAME_US: u64 = 1_000_000 / 60;
        let mut st = state();
        st.set_current(1);
        // Called 5 ms into a frame, one vblank away is 11.67 ms - not 16.67.
        st.advance_time_to(5_000);
        assert_eq!(st.vblank_park(1, FRAME_US), FRAME_US - 5_000);
        // n counts EDGES, so two vblanks from the same point is one more period.
        assert_eq!(st.vblank_park(2, FRAME_US), 2 * FRAME_US - 5_000);
        // Exactly ON an edge the wait is for the NEXT one. A full period, never zero:
        // a loop of instantly-returning vblank waits would spin without ever advancing
        // the clock, which is the shape that livelocks this scheduler.
        st.advance_time_to(4 * FRAME_US);
        assert_eq!(st.vblank_park(1, FRAME_US), FRAME_US);
    }

    #[test]
    fn storage_park_is_paid_by_frames_not_by_the_game_clock() {
        const FRAME_US: u64 = 1_000_000 / 60;
        let mut st = state();
        // Thread 1 pays for a transfer worth two and a bit frames.
        st.set_current(1);
        st.io_park(2 * FRAME_US + 10);
        assert!(st.take_wakes().is_empty());
        // The game clock running does NOT pay for it: a transfer is the device's time,
        // and coupling the two is what livelocked (the clock only moves on flips, and
        // the title is waiting on this very load).
        st.advance_time_to(st.now_us() + 10_000_000);
        assert!(st.take_wakes().is_empty());
        // Rendered frames do. Two are not enough; the third completes it.
        st.advance_io_frame(FRAME_US);
        st.advance_io_frame(FRAME_US);
        assert!(st.take_wakes().is_empty());
        st.advance_io_frame(FRAME_US);
        assert_eq!(st.take_wakes(), vec![1]);
    }

    #[test]
    fn quantum_charges_are_netted_out_of_the_next_frame() {
        const FRAME_US: u64 = 1_000_000 / 60;
        let mut st = state();
        st.set_current(1);
        st.io_park(FRAME_US);
        // A whole frame's worth of storage time charged by quanta must not be charged
        // AGAIN by the flip that follows - a rendering title advances exactly one frame
        // of storage time per frame however its quanta fell.
        st.charge_io_quantum(FRAME_US / 2);
        st.charge_io_quantum(FRAME_US / 2);
        assert_eq!(st.take_wakes(), vec![1]);
        st.set_current(2);
        st.io_park(1);
        st.advance_io_frame(FRAME_US);
        assert!(st.take_wakes().is_empty(), "the flip owed nothing, so nothing completed");
    }

    #[test]
    fn the_idle_path_completes_the_earliest_transfer() {
        let mut st = state();
        st.set_current(1);
        st.io_park(5_000);
        st.set_current(2);
        st.io_park(1_000);
        // Nothing else can run, so the device is the only thing with work left: the
        // earliest transfer completes, and only that one.
        assert!(st.release_earliest_io());
        assert_eq!(st.take_wakes(), vec![2]);
        assert!(st.has_io_waiters());
        assert!(st.release_earliest_io());
        assert_eq!(st.take_wakes(), vec![1]);
        assert!(!st.release_earliest_io(), "nothing outstanding");
    }

    #[test]
    fn semaphore_parks_then_a_signal_wakes_and_consumes() {
        let mut st = state();
        let sem = st.create_sema("test", 0, 0, i32::MAX);
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
        let sem = st.create_sema("test", 0, 0, i32::MAX);
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
        let sem = st.create_sema("test", 0, 0, i32::MAX);
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
        let sem = st.create_sema("test", 0, 0, i32::MAX);
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
        let sem = st.create_sema("test", 0, 0, i32::MAX);
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
        let w = &mut Words::default();
        created_lwmutex(&mut st, w, 0x9100);
        st.lwcond_bind_mutex(0x9000, 0x9100);
        st.set_current(1);
        assert!(st.lwcond_wait(w, 0x9000, 250)); // work-area address, 250 us timeout
        assert_eq!(st.earliest_lwcond_deadline(), Some(250));
        st.advance_time_to(250);
        // The expiry owes thread 1 its mutex back, and `advance_time_to` has no guest
        // memory to hand it over in - so nothing is woken until the scheduler settles it.
        // Asserted rather than skipped: a wake that appeared here would mean the handoff
        // had been guessed at instead of read.
        assert!(st.take_wakes().is_empty(), "the handoff is deferred, so the wake is too");
        assert_eq!(st.resolve_deferred_lwmutex(w), 1);
        assert_eq!(st.take_wakes(), vec![1]);
        assert_eq!(st.take_resume_code(1), Some(SCE_KERNEL_ERROR_WAIT_TIMEOUT));
        // A timed wait released by a signal instead returns 0 (no resume code).
        st.set_current(2);
        assert!(st.lwcond_wait(w, 0x9000, 250));
        st.set_current(3);
        st.lwcond_signal(w, 0x9000, false);
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
        let w = &mut Words::default();
        let work = 0x8000; // guest work-area address (no kernel handle)
        created_lwmutex(&mut st, w, work);
        created_lwmutex(&mut st, w, 0x9000);
        // Thread 1 locks the lightweight mutex; thread 2 contends and parks.
        st.set_current(1);
        assert!(st.lwmutex_lock(w, work));
        st.set_current(2);
        assert!(st.lwmutex_contended(w, work));
        assert!(!st.lwmutex_lock(w, work), "contender must block, not silently succeed");
        assert!(st.take_wakes().is_empty());
        // ...and the work area SAYS a thread is parked, which is what keeps the inline
        // fast path off a mutex only the host can now release correctly.
        assert_eq!(w.word(work + lwwork::off::WAITERS), 1);
        // Thread 1 unlocks: ownership passes to thread 2, which is woken.
        st.set_current(1);
        st.lwmutex_unlock(w, work);
        assert_eq!(st.take_wakes(), vec![2]);
        assert_eq!(w.word(work + lwwork::off::WAITERS), 0, "the queue drained with the handoff");
        assert_eq!(lwwork::owner(w, work), 2, "and thread 2 holds it now");
        // A different work address is an independent lock (thread 3 takes it freely).
        st.set_current(3);
        assert!(st.lwmutex_lock(w, 0x9000));
        assert!(st.lwmutex_contended(w, work), "the first lock is still held by thread 2");
    }

    #[test]
    fn lightweight_mutex_is_recursive_for_the_owner() {
        let mut st = state();
        let w = &mut Words::default();
        let work = 0x8000;
        created_lwmutex(&mut st, w, work);
        st.set_current(1);
        assert!(st.lwmutex_lock(w, work)); // count 1
        assert!(st.lwmutex_lock(w, work)); // count 2 (recursive, same owner)
        st.lwmutex_unlock(w, work); // count 1, still owned
        st.set_current(2);
        assert!(st.lwmutex_contended(w, work), "still held after one of two unlocks");
        st.set_current(1);
        st.lwmutex_unlock(w, work); // count 0, released
        st.set_current(2);
        assert!(!st.lwmutex_contended(w, work), "free after the matching unlock");
    }

    #[test]
    fn lightweight_cond_wait_releases_and_reacquires_its_bound_mutex() {
        let mut st = state();
        let w = &mut Words::default();
        let mutex_work = 0x8000;
        let cond_work = 0x8100;
        created_lwmutex(&mut st, w, mutex_work);
        st.lwcond_bind_mutex(cond_work, mutex_work);
        // Thread 1 holds the lwmutex, then waits on the lwcond: the wait releases the
        // mutex (so a sibling can take it) and parks thread 1.
        st.set_current(1);
        assert!(st.lwmutex_lock(w, mutex_work));
        assert!(st.lwcond_wait(w, cond_work, 0));
        assert!(st.take_wakes().is_empty(), "waiter is parked, not runnable");
        st.set_current(2);
        assert!(!st.lwmutex_contended(w, mutex_work), "the lwmutex was released by the wait");
        // Thread 2 takes the lwmutex, then signals: the waiter must re-acquire the
        // mutex first, so it queues behind thread 2 (not woken yet).
        assert!(st.lwmutex_lock(w, mutex_work));
        st.lwcond_signal(w, cond_work, false);
        assert!(st.take_wakes().is_empty(), "waiter must re-acquire the lwmutex first");
        // Thread 2 unlocks: ownership passes to the waiter, which is finally woken.
        st.lwmutex_unlock(w, mutex_work);
        assert_eq!(st.take_wakes(), vec![1]);
        st.set_current(3);
        assert!(st.lwmutex_contended(w, mutex_work), "the woken waiter re-acquired the lwmutex");
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
        let worker = st.create_thread(0x2000, 0x1000, DEFAULT_THREAD_PRIORITY, 0, 0);
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
        let worker = st.create_thread(0x2000, 0x1000, DEFAULT_THREAD_PRIORITY, 0, 0);
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

    /// A backing built the way a real one must be: it lists paths in their own spelling
    /// and resolves lookups through [`vfs_key`], exactly as an OPFS or disk backing has
    /// to. Keying it by the raw path instead would make every mixed-case test pass
    /// against a backing that could never work in practice.
    struct MapBacking {
        /// Storage spelling -> bytes, in listing order.
        stored: Vec<(String, Vec<u8>)>,
        /// Normalised key -> index into `stored`.
        by_key: std::collections::HashMap<String, usize>,
    }

    impl MapBacking {
        fn new(files: &[(&str, Vec<u8>)]) -> Self {
            let stored: Vec<(String, Vec<u8>)> =
                files.iter().map(|(p, b)| (p.to_string(), b.clone())).collect();
            let by_key =
                stored.iter().enumerate().map(|(i, (p, _))| (vfs_key(p), i)).collect();
            MapBacking { stored, by_key }
        }
        fn get(&self, key: &str) -> Option<&[u8]> {
            self.by_key.get(key).map(|&i| self.stored[i].1.as_slice())
        }
    }

    impl FileBacking for MapBacking {
        fn len(&self, key: &str) -> Option<usize> {
            self.get(key).map(|v| v.len())
        }
        fn read_at(&self, key: &str, off: usize, buf: &mut [u8]) -> usize {
            let Some(v) = self.get(key) else { return 0 };
            let end = (off + buf.len()).min(v.len());
            let n = end.saturating_sub(off);
            buf[..n].copy_from_slice(&v[off..end]);
            n
        }
        fn keys(&self) -> Vec<String> {
            self.stored.iter().map(|(p, _)| p.clone()).collect()
        }
    }

    fn backed_state(path: &str, bytes: &[u8]) -> VitaState {
        let mut st = state();
        st.set_file_backing(Box::new(MapBacking::new(&[(path, bytes.to_vec())])));
        st
    }

    /// A lazily-backed file must be indistinguishable from a resident one through the
    /// whole SceIo surface. This is the property the browser's memory budget rests on,
    /// and every way it can go wrong is silent: a title that reads a truncated asset, or
    /// sizes a buffer from a zero stat, fails somewhere else entirely.
    #[test]
    fn a_backed_file_reads_seeks_and_stats_like_a_resident_one() {
        let data: Vec<u8> = (0..=255u8).collect();
        let mut st = backed_state("ux0:/data/asset.bin", &data);

        // It exists, and its size is known WITHOUT reading it.
        assert_eq!(st.io_size("ux0:/data/asset.bin"), Some(256));

        let fd = st.io_open("ux0:/data/asset.bin", SCE_O_RDONLY);
        assert!(fd >= 0, "a backed file must open for reading");

        // Sequential reads advance the cursor and deliver the real bytes.
        assert_eq!(st.io_read(fd, 4), Some(vec![0, 1, 2, 3]));
        assert_eq!(st.io_read(fd, 4), Some(vec![4, 5, 6, 7]));

        // Seek from the end, then read across it: a short read at end of file must be
        // short, not padded, and must leave the cursor AT the end.
        assert_eq!(st.io_lseek(fd, -2, SCE_SEEK_END), 254);
        assert_eq!(st.io_read(fd, 8), Some(vec![254, 255]));
        assert_eq!(st.io_read(fd, 8), Some(vec![]));

        // Positional reads do not disturb the cursor.
        assert_eq!(st.io_lseek(fd, 10, SCE_SEEK_SET), 10);
        assert_eq!(st.io_pread(fd, 100, 3), Some(vec![100, 101, 102]));
        assert_eq!(st.io_read(fd, 2), Some(vec![10, 11]));
    }

    /// A write to a backed file faults it in and then behaves exactly as it always did.
    /// Worth pinning because the fault-in is the one place the two storage models meet,
    /// and a read-after-write that returned the ORIGINAL bytes would look like a
    /// corrupted save rather than a filesystem bug.
    #[test]
    fn writing_a_backed_file_faults_it_in_and_read_after_write_is_exact() {
        let mut st = backed_state("ux0:/data/asset.bin", &[9u8; 16]);
        let fd = st.io_open("ux0:/data/asset.bin", SCE_O_RDWR);
        assert!(fd >= 0);
        assert_eq!(st.io_lseek(fd, 4, SCE_SEEK_SET), 4);
        assert_eq!(st.io_write(fd, &[1, 2, 3]), Some(3));
        assert_eq!(st.io_lseek(fd, 0, SCE_SEEK_SET), 0);
        assert_eq!(st.io_read(fd, 8), Some(vec![9, 9, 9, 9, 1, 2, 3, 9]));
        // Faulted in, so the inspection seam can now see it.
        assert_eq!(st.file_bytes("ux0:/data/asset.bin").map(|b| b.len()), Some(16));
    }

    /// A backed file whose stored name has CAPITALS must be reachable by the guest, which
    /// asks case-insensitively through an `app0:` mount.
    ///
    /// This is the exact bug that shipped and cost the most to find. The guest filesystem
    /// normalises every path (lowercased, `app0:` stripped) before it reaches a backing,
    /// while storage keeps the name as written. Get the two out of step and the file is
    /// either missing or reads as nothing - and what surfaces is a guest memory-access
    /// trap 30,000 host calls later, with nothing anywhere pointing at the filesystem. It
    /// went wrong twice in a row, in opposite directions, so both ends are pinned here.
    #[test]
    fn a_backed_file_is_reachable_through_a_mount_and_a_different_case() {
        let data: Vec<u8> = (0..64u8).collect();
        let mut st = state();
        st.set_file_backing(Box::new(MapBacking::new(&[("PSP2/Data/Blob.BIN", data.clone())])));

        // The spelling the title actually uses: an app0: mount, and whatever case it feels
        // like. Every one of these is the same file.
        for path in ["app0:/PSP2/Data/Blob.BIN", "app0:/psp2/data/blob.bin", "PSP2/Data/Blob.BIN"] {
            assert_eq!(st.io_size(path), Some(64), "size of {path}");
            let fd = st.io_open(path, SCE_O_RDONLY);
            assert!(fd >= 0, "{path} must open (got {fd})");
            assert_eq!(st.io_read(fd, 8), Some(vec![0, 1, 2, 3, 4, 5, 6, 7]), "read of {path}");
        }
    }

    /// A backed file has to show up in a directory listing with its real size, because a
    /// title enumerates its own data directory and sizes read buffers from what it finds.
    #[test]
    fn a_backed_file_appears_in_a_listing_with_its_size() {
        let mut st = backed_state("ux0:/data/Asset.BIN", &[0u8; 42]);
        let fd = st.io_dopen("ux0:/data");
        assert!(fd >= 0, "the implied directory of a backed file must open");
        let e = st.io_dread(fd).expect("a valid dir fd").expect("one entry");
        assert_eq!(e.name, "Asset.BIN", "the as-supplied spelling must survive");
        assert_eq!(e.size, 42);
        assert!(!e.is_dir);
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
