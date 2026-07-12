//! The host-call boundary: how a guest NID import trap becomes a typed Rust
//! handler. `GuestCtx` marshals AAPCS arguments (r0..r3 then stack) and guest
//! memory in and out; `VitaEnv` owns the per-run state (allocator, handles,
//! capture, world) and routes a dense import index to a per-module handler.
//! See `projects/vitaslop-runtime/README.md`.

use vitaslop_transpiler::abi::{REG_COUNT, SP};

use crate::capture::Capture;
use crate::world::{DeterministicWorld, World};
use crate::{vita, SvcOutcome};

/// A borrowed view of guest state for the duration of one host call: the
/// register file, guest memory (rebased so guest address `A` is byte `A - base`),
/// and a sequential AAPCS argument cursor.
pub struct GuestCtx<'a> {
    pub regs: &'a mut [u32; REG_COUNT],
    pub mem: &'a mut [u8],
    pub base: u32,
    /// Next positional argument to read (0-based). Args 0..3 are r0..r3, args >=4
    /// are on the stack at sp + (n-4)*4.
    next_arg: usize,
}

impl<'a> GuestCtx<'a> {
    fn new(regs: &'a mut [u32; REG_COUNT], mem: &'a mut [u8], base: u32) -> Self {
        GuestCtx { regs, mem, base, next_arg: 0 }
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

    /// Read the next positional argument and advance the cursor.
    pub fn next_u32(&mut self) -> u32 {
        let v = self.arg(self.next_arg);
        self.next_arg += 1;
        v
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
                u32::from_le_bytes([self.mem[o], self.mem[o + 1], self.mem[o + 2], self.mem[o + 3]])
            }
            _ => 0,
        }
    }

    /// Write a little-endian u32 at guest address `addr` (ignored if out of range).
    pub fn write_u32(&mut self, addr: u32, v: u32) {
        if let Some(o) = self.offset(addr) {
            if o + 4 <= self.mem.len() {
                self.mem[o..o + 4].copy_from_slice(&v.to_le_bytes());
            }
        }
    }

    /// Read `len` bytes at guest address `addr` (short read clamped to range).
    pub fn read_bytes(&self, addr: u32, len: usize) -> Vec<u8> {
        match self.offset(addr) {
            Some(o) => {
                let end = (o + len).min(self.mem.len());
                self.mem[o..end].to_vec()
            }
            None => Vec::new(),
        }
    }

    /// Write `bytes` at guest address `addr` (clamped to range).
    pub fn write_bytes(&mut self, addr: u32, bytes: &[u8]) {
        if let Some(o) = self.offset(addr) {
            let end = (o + bytes.len()).min(self.mem.len());
            self.mem[o..end].copy_from_slice(&bytes[..end - o]);
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
    pub world: Box<dyn World>,
    /// Bring-up aid: halt the run when the guest calls sceGxmTerminate. The cube
    /// entry is `_start`, which spins forever after `main` returns (there is no OS
    /// to exit to yet), so terminate is the clean stopping point after teardown.
    pub halt_on_terminate: bool,
}

impl VitaState {
    /// New state for a run over `[base, base + mem_bytes)`. Allocations start
    /// above the image (at base + 1 MiB) and grow up, well below the stack that
    /// starts at the top of the region.
    pub fn new(base: u32, mem_bytes: u32, world: Box<dyn World>) -> Self {
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
    pub fn new(imports: Vec<(u32, u32)>, base: u32, mem_bytes: u32, world: Box<dyn World>) -> Self {
        VitaEnv { imports, state: VitaState::new(base, mem_bytes, world) }
    }

    /// Convenience constructor with the default deterministic world.
    pub fn with_default_world(imports: Vec<(u32, u32)>, base: u32, mem_bytes: u32) -> Self {
        VitaEnv::new(imports, base, mem_bytes, Box::new(DeterministicWorld::default()))
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
        mem: &mut [u8],
        base: u32,
    ) -> SvcOutcome;
}

impl ImportDispatch for VitaEnv {
    fn dispatch(
        &mut self,
        index: u32,
        regs: &mut [u32; REG_COUNT],
        mem: &mut [u8],
        base: u32,
    ) -> SvcOutcome {
        self.state.capture.call_count += 1;
        let (library_nid, func_nid) = self
            .imports
            .get(index as usize)
            .copied()
            .unwrap_or((0, 0));
        self.state.capture.trace.push(func_nid);
        let mut ctx = GuestCtx::new(regs, mem, base);
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
        mem: &mut [u8],
        base: u32,
    ) -> SvcOutcome {
        self.borrow_mut().dispatch(index, regs, mem, base)
    }
}
