//! Everything memoised per registered `SceGxmProgram`, keyed by its header address.
//!
//! Every table here is a pure function of the program's container bytes, which are immutable
//! while the program is registered (GXM's shader patcher owns them; the guest does not rewrite a
//! live program). So the tables share one lifetime: they fill lazily, one entry per program on
//! its first use, and they are all dropped TOGETHER by [`ProgramMemos::clear`] at the only
//! moment a header address can come to mean a different program - a register or unregister.
//! One owner is what makes "all together" a property of the code rather than of a list someone
//! has to keep in step.

use super::*;

/// The per-program memo tables. See the module docs.
#[derive(Default)]
pub(super) struct ProgramMemos {
    /// Reflected constants of each `SceGxmProgram` - see [`ProgramReflection`].
    reflection: FxHashMap<u32, ProgramReflection>,
    /// Raw `SceGxmProgram` container bytes - see `VitaState::program_blob`.
    blobs: FxHashMap<u32, std::sync::Arc<[u8]>>,
    /// The blend compiled into each fragment program's epilogue (`vitaslop_gxp_shader::rop_blend`).
    /// A title that re-creates its fragment programs every frame (a football title: ~46) paid a
    /// whole-shader parse and decode per create to re-derive the same answer.
    rop_blend: FxHashMap<u32, Option<vitaslop_gxp_shader::RopBlend>>,
    /// Whether (and how) each program's 0xE8 memory loads need guest-memory windows snapshotted
    /// at draw time, by `(header, is fragment)`. EMPTY for the overwhelming majority of
    /// programs, so the per-draw cost of this feature on every other title is one lookup.
    mem_windows: FxHashMap<(u32, bool), std::sync::Arc<[vitaslop_gxp_shader::MemWindow]>>,
    /// The buffers each FRAGMENT program's guest-memory STORES write through
    /// (`vitaslop_gxp_shader::store_windows_for_fragment_blob`), by header. Empty for every
    /// program that does not store.
    store_windows: FxHashMap<u32, std::sync::Arc<[vitaslop_gxp_shader::MemWindow]>>,
    /// Per program: the non-default uniform buffers the driver copies into the SA register
    /// file, and where the DEFAULT buffer's own block starts.
    sa_uniforms: FxHashMap<u32, std::sync::Arc<SaUniformLayout>>,
}

impl ProgramMemos {
    /// Drop every table - see the module docs for when.
    pub(super) fn clear(&mut self) {
        self.reflection.clear();
        self.blobs.clear();
        self.rop_blend.clear();
        self.mem_windows.clear();
        self.store_windows.clear();
        self.sa_uniforms.clear();
    }

    pub(super) fn store_windows(&self, header: u32) -> Option<std::sync::Arc<[vitaslop_gxp_shader::MemWindow]>> {
        self.store_windows.get(&header).cloned()
    }

    pub(super) fn set_store_windows(&mut self, header: u32, spec: std::sync::Arc<[vitaslop_gxp_shader::MemWindow]>) {
        self.store_windows.insert(header, spec);
    }

    pub(super) fn reflection(&self, header: u32) -> Option<ProgramReflection> {
        self.reflection.get(&header).copied()
    }

    pub(super) fn set_reflection(&mut self, header: u32, r: ProgramReflection) {
        self.reflection.insert(header, r);
    }

    pub(super) fn blob(&self, header: u32) -> Option<std::sync::Arc<[u8]>> {
        self.blobs.get(&header).cloned()
    }

    pub(super) fn set_blob(&mut self, header: u32, blob: std::sync::Arc<[u8]>) {
        self.blobs.insert(header, blob);
    }

    /// The memoised epilogue blend, if `header` has been asked about: `Some(None)` is a
    /// program known to carry none.
    pub(super) fn rop_blend(&self, header: u32) -> Option<Option<vitaslop_gxp_shader::RopBlend>> {
        self.rop_blend.get(&header).copied()
    }

    pub(super) fn set_rop_blend(
        &mut self,
        header: u32,
        blend: Option<vitaslop_gxp_shader::RopBlend>,
    ) {
        self.rop_blend.insert(header, blend);
    }

    pub(super) fn mem_windows(
        &self,
        key: (u32, bool),
    ) -> Option<std::sync::Arc<[vitaslop_gxp_shader::MemWindow]>> {
        self.mem_windows.get(&key).cloned()
    }

    pub(super) fn set_mem_windows(
        &mut self,
        key: (u32, bool),
        spec: std::sync::Arc<[vitaslop_gxp_shader::MemWindow]>,
    ) {
        self.mem_windows.insert(key, spec);
    }

    pub(super) fn sa_uniforms(&self, header: u32) -> Option<std::sync::Arc<SaUniformLayout>> {
        self.sa_uniforms.get(&header).cloned()
    }

    pub(super) fn set_sa_uniforms(&mut self, header: u32, layout: std::sync::Arc<SaUniformLayout>) {
        self.sa_uniforms.insert(header, layout);
    }
}
