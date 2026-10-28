//! The host's records about the guest's PRECOMPUTED states and draws. The states and draws
//! themselves live in guest memory (`vita::gxmstate`, [`super::pdraw`]); what is kept here is
//! only what the guest's memory cannot vouch for: which arrays blocks this engine allocated and
//! for which struct, each draw block as our setters last left it, and the report ladder.
//!
//! [`PrecomputedRecords`] owns those; `VitaState` (`state_precomputed.rs`) does the guest reads
//! and writes and decides when to consult them.

use super::pdraw;
use std::collections::HashMap;

/// See the module docs.
#[derive(Default)]
pub(super) struct PrecomputedRecords {
    /// Every precomputed-state ARRAYS block this engine has allocated, with the state struct
    /// address it was allocated FOR - see `VitaState::ensure_state_block` for why a block
    /// pointer read out of guest memory is trusted only when it is in here.
    state_blocks: HashMap<u32, u32>,
    /// How many precomputed states have been INITIALISED, for
    /// `VitaState::report_precomputed_state_programs`'s ladder.
    state_count: u32,
    /// The distinct (vertex, fragment) header counts the ladder last printed - a new program
    /// is always worth a line, a re-init of a known one is not.
    state_reported: (usize, usize),
    /// Each precomputed-draw block as OUR setters last left it - see
    /// [`PrecomputedRecords::note_guest_edits`].
    draw_shadow: HashMap<u32, [u32; pdraw::WORDS as usize]>,
}

impl PrecomputedRecords {
    /// The state struct an arrays block at `block` was allocated for, if this engine made it.
    pub(super) fn block_owner(&self, block: u32) -> Option<u32> {
        self.state_blocks.get(&block).copied()
    }

    /// Record that `block` was allocated for the state struct at `state`.
    pub(super) fn claim_block(&mut self, block: u32, state: u32) {
        self.state_blocks.insert(block, state);
    }

    /// Count one more state init; returns `Some(count)` when the ladder should print, given the
    /// distinct header counts now - a new program always does, a re-init only on a power of two.
    pub(super) fn note_state_init(&mut self, distinct: (usize, usize)) -> Option<u32> {
        self.state_count += 1;
        let n = self.state_count;
        if n > 4 && !n.is_power_of_two() && distinct == self.state_reported {
            return None;
        }
        self.state_reported = distinct;
        Some(n)
    }

    /// Record the draw block at `precomputed` as our setters just left it.
    pub(super) fn shadow_draw(&mut self, precomputed: u32, words: [u32; pdraw::WORDS as usize]) {
        self.draw_shadow.insert(precomputed, words);
    }

    /// Compare a draw block as just read against the shadow, warning once per change for every
    /// word the GUEST changed without a setter, and adopt the new values. `lr` names the caller.
    pub(super) fn note_guest_edits(
        &mut self,
        precomputed: u32,
        words: &[u32; pdraw::WORDS as usize],
        lr: u32,
    ) {
        let Some(shadow) = self.draw_shadow.get_mut(&precomputed) else {
            return;
        };
        for (i, old) in shadow.iter_mut().enumerate() {
            let now = words[i];
            if now != *old {
                tracing::warn!(
                    target: "vitaslop::gxm",
                    precomputed = format_args!("{precomputed:#x}"),
                    word = i,
                    old = format_args!("{old:#010x}"),
                    new = format_args!("{now:#010x}"),
                    lr = format_args!("{lr:#010x}"),
                    "precomputed draw block changed by the GUEST, not by a setter"
                );
                *old = now;
            }
        }
    }
}
