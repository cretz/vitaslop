//! The HOST MIRROR block: the few system values a guest can read INLINE, without
//! trapping to the host at all.
//!
//! # What may live here, and what may not
//! A value qualifies only if it **cannot change while guest code is running**. That is
//! not a statement about how often it changes - it is a statement about *when*. The
//! virtual clock qualifies because it advances in exactly three places, all of them in
//! the scheduler with no guest thread live: a quantum charge, a display flip, and the
//! idle path ([`VitaState::charge_cpu_quantum`], [`VitaState::advance_time_frame`],
//! [`VitaState::advance_time_to`]). So a word refreshed before every resume is not an
//! approximation of the call - it is the answer the call would have given, and it stays
//! that answer for as long as the guest can observe it.
//!
//! A value that a HOST CALL can change does NOT qualify, even if calls are rare: the
//! guest can make that call and then read the mirror within the same slice.
//!
//! # Why this exists
//! `sceDisplayGetVcount` is an ordinary vblank wait's inner loop
//! (`do { v = sceDisplayGetVcount(); } while (v == last);`), and on a real title that
//! is ~12,300 of the ~14,800 host calls a frame - about half the frame spent crossing
//! the wasm/host boundary to read a counter. Inline, it is one `i32.load`.
//!
//! # The contract, which is not optional
//! [`snapshot`] must be written into the block before guest code resumes. The
//! scheduler does that at its one resume point; a host that does not is rejected when
//! the run is stood up rather than left to serve a frozen clock (which presents as a
//! livelocked vblank spin thousands of frames from the cause).

use crate::host::VitaState;

/// Slot 0: `sceDisplayGetVcount`.
pub const SLOT_VCOUNT: u32 = 0;

/// How many slots the block has. The scheduler writes exactly this many.
pub const SLOT_COUNT: usize = 1;

/// The current value of every mirror slot, in slot order.
///
/// The single definition of what the block holds. Each entry must be computed by the
/// SAME function its host handler calls, so the inline and host forms cannot drift;
/// `mirror_matches_its_handlers` holds them to that.
pub fn snapshot(st: &VitaState) -> [u32; SLOT_COUNT] {
    [super::display::vcount(st)]
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::nid::display as d;
    use crate::world::DeterministicWorld;
    use crate::{SliceMemory, VFP_ARG_COUNT};
    use vitaslop_transpiler::abi::REG_COUNT;
    use vitaslop_transpiler::InlineOp;

    /// A state whose clock is at `us`.
    fn state_at(us: u64) -> VitaState {
        let mut st = VitaState::new(0, 4096, Box::new(DeterministicWorld::default()));
        st.advance_time_to(us);
        st
    }

    /// The r0 the guest would see from the real host call, over `st`.
    fn handler_result(func_nid: u32, st: &mut VitaState) -> u32 {
        let mut regs = [0u32; REG_COUNT];
        let mut vfp = [0u32; VFP_ARG_COUNT];
        let mut bytes = vec![0u8; 4096];
        let mut mem = SliceMemory(&mut bytes);
        let mut ctx = crate::host::GuestCtx::new(&mut regs, &mut vfp, &mut mem, 0);
        super::super::dispatch(crate::nid::lib::SCE_DISPLAY_USER, func_nid, &mut ctx, st);
        regs[0]
    }

    /// Every mirrored NID must compute exactly what its host handler computes. The
    /// inline form is a SECOND implementation - the transpiler emits it into guest code
    /// and the handler never runs - so nothing else in the system would notice them
    /// drifting apart. The clock values below span a vblank boundary in both directions
    /// so an off-by-one in the division cannot pass.
    #[test]
    fn mirror_matches_its_handlers() {
        for us in [0, 1, 16_665, 16_666, 16_667, 33_333, 1_000_000, 4_000_000_000] {
            let mut st = state_at(us);
            let words = snapshot(&st);
            let op = super::super::display::inline_op(d::GET_VCOUNT)
                .expect("sceDisplayGetVcount has an inline form");
            let InlineOp::LoadMirror { slot } = op else {
                panic!("sceDisplayGetVcount must lower to a mirror read, got {op:?}");
            };
            assert_eq!(
                op.eval(words[slot as usize]),
                handler_result(d::GET_VCOUNT, &mut st),
                "inline sceDisplayGetVcount disagrees with its handler at {us} us",
            );
        }
    }

    /// Every slot the block declares is filled by [`snapshot`], and every slot an inline
    /// op names is one the block declares. A slot that no snapshot writes reads as a
    /// word that never changes, which for a clock is a spin that can never be satisfied.
    #[test]
    fn every_declared_slot_is_within_the_block() {
        assert_eq!(snapshot(&state_at(0)).len(), SLOT_COUNT, "snapshot fills every slot");
        for slot in [SLOT_VCOUNT] {
            assert!((slot as usize) < SLOT_COUNT, "slot {slot} is outside the block");
        }
    }

    /// The waits are NOT mirrorable and must stay host calls: they block, which is
    /// behaviour, and an inlined call never reaches the host at all.
    #[test]
    fn only_the_counter_is_mirrored() {
        for nid in [d::WAIT_VBLANK_START, d::WAIT_VBLANK_START_MULTI, d::SET_FRAME_BUF] {
            assert!(
                super::super::display::inline_op(nid).is_none(),
                "{} has behaviour and must not be inlined",
                crate::nid::name(nid)
            );
        }
    }
}
