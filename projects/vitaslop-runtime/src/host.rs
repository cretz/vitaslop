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
    fn new(
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
    pub capture: Capture,
    // `Send` so a `VitaEnv` can be the data of a wasmtime async Store (the
    // cooperative scheduler runs the guest on a fiber, which wasmtime may resume
    // on any thread). Everything stays single-threaded in practice.
    pub world: Box<dyn World + Send>,
    /// Bring-up aid: halt the run when the guest calls sceGxmTerminate. The cube
    /// entry is `_start`, which spins forever after `main` returns (there is no OS
    /// to exit to yet), so terminate is the clean stopping point after teardown.
    pub halt_on_terminate: bool,
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
            capture: Capture::new(),
            world,
            halt_on_terminate: false,
        }
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
        let draw = crate::capture::Draw {
            primitive,
            index_format,
            index_count,
            vertices,
            vertex_stride: stride,
            attributes,
            indices,
            uniforms: self.pending_uniforms.clone(),
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
        let mut ctx = GuestCtx::new(regs, vfp, mem, base);
        vita::dispatch(library_nid, func_nid, &mut ctx, &mut self.state)
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
