//! The shader patcher's records: every program the guest registered, every vertex and
//! fragment program it created from one, and the user-data word it gave each patcher.
//!
//! [`ShaderPatcherRecords`] owns them. Handing out a fresh handle and invalidating the
//! reflection memos on a registration stay with `VitaState::register_shader_program`, which
//! is the one moment a header address can begin to mean a different program.

use super::VertexProgramInfo;
use crate::capture::BlendState;
use crate::fasthash::FxHashMap;
use std::collections::HashMap;

/// See the module docs.
#[derive(Default)]
pub(super) struct ShaderPatcherRecords {
    /// `(shaderPatcherId handle, SceGxmProgram* header)` from every
    /// `sceGxmShaderPatcherRegisterProgram`, so `sceGxmShaderPatcherGetProgramFromId`
    /// can hand back the real program pointer the guest registered.
    registered: Vec<(u32, u32)>,
    /// `(shaderPatcher handle, userData)` from `sceGxmShaderPatcherSetUserData`. One
    /// opaque word per patcher that GXM keeps and hands back - a title's own allocator
    /// context, typically, which its host-callback allocator then needs to find again.
    /// Keyed by the patcher HANDLE (not an address), so it is the identity the guest was
    /// given rather than wherever it happens to have stored it.
    user_data: Vec<(u32, u32)>,
    /// Vertex program handle -> its attribute layout. A map, not a list: every draw
    /// resolves the bound handle several times, and a title registers hundreds of
    /// programs, so a linear scan here was per-draw cost that grew with the level.
    /// >>> AND AN INTEGER HASHER, LIKE EVERY OTHER PER-DRAW MAP HERE. The key is a guest
    /// address; SipHash on it is tens of nanoseconds per probe, and a draw probes this and
    /// the fragment map several times each - about 4,000 probes a presented frame. See
    /// [`crate::fasthash`], which the texture maps have used since the same measurement.
    vertex_programs: FxHashMap<u32, VertexProgramInfo>,
    /// `SceGxmFragmentProgram*` handle -> (its `SceGxmProgram*`, the blend equation it was
    /// created with), recorded at `sceGxmShaderPatcherCreateFragmentProgram` so a precomputed
    /// fragment state can size its default uniform buffer and every draw can carry its real
    /// blend mode. (Vertex programs carry their header in `VertexProgramInfo`.) The third
    /// field is the `vertexProgram` the create call named - the program the patcher linked this
    /// fragment's varying iteration AGAINST, which is not always the one a draw binds. See
    /// `capture::Draw::fprog_patched_vprog`.
    fragment_programs: HashMap<u32, (u32, BlendState, u32)>,
}

impl ShaderPatcherRecords {
    pub(super) fn register(&mut self, handle: u32, program_header: u32) {
        self.registered.push((handle, program_header));
    }

    /// The `SceGxmProgram*` header for a `shaderPatcherId`, or 0 if unknown.
    pub(super) fn registered(&self, id: u32) -> u32 {
        self.registered
            .iter()
            .find(|&&(h, _)| h == id)
            .map(|&(_, p)| p)
            .unwrap_or(0)
    }

    /// Store the patcher's user-data word, replacing any previous one.
    pub(super) fn set_user_data(&mut self, patcher: u32, user_data: u32) {
        match self.user_data.iter_mut().find(|(h, _)| *h == patcher) {
            Some(slot) => slot.1 = user_data,
            None => self.user_data.push((patcher, user_data)),
        }
    }

    /// The patcher's user-data word; NULL for one that was never given one.
    pub(super) fn user_data(&self, patcher: u32) -> u32 {
        self.user_data
            .iter()
            .find(|&&(h, _)| h == patcher)
            .map(|&(_, d)| d)
            .unwrap_or(0)
    }

    #[inline]
    pub(super) fn vertex_program(&self, handle: u32) -> Option<&VertexProgramInfo> {
        self.vertex_programs.get(&handle)
    }

    pub(super) fn vertex_programs(&self) -> impl Iterator<Item = &VertexProgramInfo> {
        self.vertex_programs.values()
    }

    pub(super) fn set_vertex_program(&mut self, handle: u32, info: VertexProgramInfo) {
        self.vertex_programs.insert(handle, info);
    }

    pub(super) fn set_fragment_program(
        &mut self,
        handle: u32,
        program_header: u32,
        blend: BlendState,
        patched_vertex_header: u32,
    ) {
        self.fragment_programs
            .insert(handle, (program_header, blend, patched_vertex_header));
    }

    #[inline]
    pub(super) fn fragment_program(&self, handle: u32) -> Option<&(u32, BlendState, u32)> {
        self.fragment_programs.get(&handle)
    }
}
