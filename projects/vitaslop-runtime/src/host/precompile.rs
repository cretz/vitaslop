//! The shader PAIRS the renderer can prepare ahead of the draw that needs them, and the
//! program lists they are derived from.
//!
//! [`PrecompileQueue`] owns the queue (bounded, deduplicated, never drained - see
//! `VitaState::shader_precompile`) and the three sources of candidates: the pairs the patcher
//! NAMES, the vertex x NULL-paired-fragment cross product (`VITASLOP_GXP_PRECOMPILE_CROSS`), and
//! the programs the title's PRECOMPUTED STATES name. Reading the two programs' blobs for a pair
//! is `VitaState`'s, in `state_shaders.rs`; this type decides which pairs are worth reading.

use std::collections::HashSet;
use std::sync::Arc;

/// The ahead-of-draw shader pairs and their sources. See the module docs.
#[derive(Default)]
pub(super) struct PrecompileQueue {
    /// Shader PAIRS the guest's patcher has named but the renderer has not been told about yet
    /// - see `VitaState::queue_shader_precompile`. Offered to every scene handed to the
    /// renderer, which compiles them before it encodes anything.
    /// Held behind one `Arc` so handing it to a scene costs a single refcount - see
    /// [`crate::capture::Scene::precompile`]. Appending clones the list once (`Arc::make_mut`),
    /// which happens while a loading screen names pairs and never in a gameplay frame.
    pending: Arc<Vec<vitaslop_gxp_shader::PatcherPair>>,
    /// Pairs already queued, so a title that creates the same fragment program repeatedly - or
    /// re-creates one after a patcher reset - does not re-queue work the renderer has cached.
    queued: HashSet<(u32, u32)>,
    /// Every `SceGxmProgram *` the patcher has created a VERTEX program from, and every one it
    /// has created a FRAGMENT program from with a NULL `vertexProgram`. Kept in creation order
    /// and only used by the cross product - a title that names its pairs properly fills the
    /// first list and never reads either.
    created_vertex_headers: Vec<u32>,
    null_fragment_headers: Vec<u32>,
    /// The DISTINCT programs the title's PRECOMPUTED STATES name, in the order it named them -
    /// see `VitaState::cross_precomputed_state_programs`. A different and far smaller list
    /// than the two above: a precomputed state is the title declaring "this program is one I
    /// will draw with", so the cross product over these is a few hundred candidates rather
    /// than a third of a million.
    state_vertex_headers: Vec<u32>,
    state_fragment_headers: Vec<u32>,
}

impl PrecompileQueue {
    /// The most pairs the queue holds.
    const MAX_PENDING: usize = 4096;

    /// Whether the pair `(vertex, fragment)` should be read and queued now: not past the cap
    /// and not queued before. Counts it as queued when it says yes.
    ///
    /// **The cap is REPORTED when it bites.** A cross product is the one caller that can reach
    /// it, and a silently truncated candidate list would look exactly like a title whose pairs
    /// do not link - the opposite conclusion from the same evidence.
    pub(super) fn admit(&mut self, vertex_header: u32, fragment_header: u32) -> bool {
        if self.pending.len() >= Self::MAX_PENDING {
            static SEEN: std::sync::atomic::AtomicBool = std::sync::atomic::AtomicBool::new(false);
            if !SEEN.swap(true, std::sync::atomic::Ordering::Relaxed) {
                tracing::warn!(
                    target: "vitaslop::gxm",
                    max = Self::MAX_PENDING,
                    "gxm precompile: the candidate list reached its cap and later pairs are being \
                     DROPPED - anything past here still compiles at its first draw"
                );
            }
            return false;
        }
        self.queued.insert((vertex_header, fragment_header))
    }

    /// Append pairs an [`admit`](Self::admit) said yes to.
    pub(super) fn push(
        &mut self,
        pairs: impl IntoIterator<Item = vitaslop_gxp_shader::PatcherPair>,
    ) {
        Arc::make_mut(&mut self.pending).extend(pairs);
    }

    /// Every pair queued so far - see `VitaState::shader_precompile` for why this never drains.
    pub(super) fn pending(&self) -> Arc<Vec<vitaslop_gxp_shader::PatcherPair>> {
        self.pending.clone()
    }

    /// Record a fragment program created with a NULL `vertexProgram`. `false` if it was
    /// recorded before.
    pub(super) fn note_null_fragment(&mut self, fragment_header: u32) -> bool {
        if self.null_fragment_headers.contains(&fragment_header) {
            return false;
        }
        self.null_fragment_headers.push(fragment_header);
        true
    }

    /// Record a vertex program the patcher created. `false` for 0 or one recorded before.
    pub(super) fn note_vertex(&mut self, vertex_header: u32) -> bool {
        if vertex_header == 0 || self.created_vertex_headers.contains(&vertex_header) {
            return false;
        }
        self.created_vertex_headers.push(vertex_header);
        true
    }

    pub(super) fn created_vertices(&self) -> &[u32] {
        &self.created_vertex_headers
    }

    pub(super) fn null_fragments(&self) -> &[u32] {
        &self.null_fragment_headers
    }

    /// Record a program a precomputed state names; when it is NEW, the programs of the other
    /// stage named so far - the pairs it adds. `None` for 0 or one named before.
    pub(super) fn note_state_program(&mut self, header: u32, vertex: bool) -> Option<Vec<u32>> {
        if header == 0 {
            return None;
        }
        let (own, other) = if vertex {
            (&mut self.state_vertex_headers, &self.state_fragment_headers)
        } else {
            (&mut self.state_fragment_headers, &self.state_vertex_headers)
        };
        if own.contains(&header) {
            return None;
        }
        own.push(header);
        Some(other.clone())
    }

    /// How many distinct `(vertex, fragment)` programs the precomputed states have named.
    pub(super) fn state_program_counts(&self) -> (usize, usize) {
        (
            self.state_vertex_headers.len(),
            self.state_fragment_headers.len(),
        )
    }
}
