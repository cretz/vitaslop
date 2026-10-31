//! The draw-time snapshot of the buffers a FRAGMENT program's guest-memory STORES write
//! through - see `vitaslop_gxp_shader::module::resolve_store_windows` and the renderer's
//! store mirrors, which these bytes SEED.

use super::*;

impl VitaState {
    /// The store windows of the fragment program at `header`, memoised per registered program
    /// like [`Self::mem_window_spec`]: one map lookup per draw for every program that does not
    /// store.
    fn store_window_spec(&mut self, ctx: &GuestCtx, header: u32) -> std::sync::Arc<[vitaslop_gxp_shader::MemWindow]> {
        if header == 0 {
            return std::sync::Arc::from(&[][..]);
        }
        if let Some(spec) = self.program_memos.store_windows(header) {
            return spec;
        }
        let blob = self.program_blob(ctx, header);
        let spec: std::sync::Arc<[vitaslop_gxp_shader::MemWindow]> =
            std::sync::Arc::from(vitaslop_gxp_shader::store_windows_for_fragment_blob(&blob));
        self.program_memos.set_store_windows(header, spec.clone());
        spec
    }

    /// Snapshot, AT DRAW TIME, the guest bytes of every buffer the fragment program at
    /// `fheader` stores through: `(bound address, the window's declared extent of bytes)` each,
    /// in its store binding's order. Read now rather than deferred with the draw's geometry: the
    /// bytes seed the GPU's copy of memory the draw is about to write, so they must be the
    /// memory as it stands when the draw is issued. A buffer the guest left unbound yields no
    /// windows at all, and the renderer then drops the draw by name rather than seed a mirror
    /// from nothing.
    pub(super) fn capture_store_windows(
        &mut self,
        ctx: &GuestCtx,
        blk: &crate::vita::gxmctx::Block<'_>,
        fheader: u32,
    ) -> Vec<(u32, Arc<[u8]>)> {
        let spec = self.store_window_spec(ctx, fheader);
        if spec.is_empty() {
            return Vec::new();
        }
        let mut out = Vec::with_capacity(spec.len());
        for w in spec.iter() {
            let addr = blk.fragment_uniform_buffer(w.buffer_index);
            if addr == 0 {
                report_unbound_store_window(fheader, w.buffer_index);
                return Vec::new();
            }
            let mut bytes = vec![0u8; w.bytes as usize];
            ctx.read_into(addr, &mut bytes);
            out.push((addr, Arc::from(bytes)));
        }
        out
    }
}

/// Say, once per (program, buffer), that a storing fragment program was drawn with one of its
/// stored-through buffers unbound - its draw is then dropped by the renderer.
fn report_unbound_store_window(fheader: u32, buffer_index: u32) {
    use std::sync::{Mutex, OnceLock};
    static SEEN: OnceLock<Mutex<std::collections::HashSet<(u32, u32)>>> = OnceLock::new();
    if SEEN.get_or_init(Default::default).lock().unwrap_or_else(|e| e.into_inner()).insert((fheader, buffer_index)) {
        tracing::warn!(
            target: "vitaslop::gxm",
            program = format_args!("{fheader:#x}"),
            buffer_index,
            "a fragment program that STORES to guest memory was drawn with the uniform buffer it stores through unbound - the draw carries no store windows and is dropped"
        );
    }
}
