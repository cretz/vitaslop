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

/// Slots 1-2: the virtual process clock in microseconds, low word then high word.
///
/// TWO CONTIGUOUS slots, and the pair forms depend on that adjacency - they read
/// `mirror[slot]` and `mirror[slot + 1]` with one base address. Inserting a slot between
/// them would compile and would serve a clock whose high word is something else entirely.
///
/// This backs the whole `sceKernelGetProcessTime` family, which after the vblank counter
/// is the most-called host function a real title makes: 56,287 calls in one profile
/// window. All three spellings are the same pure function of the clock and differ only in
/// where they put it - through a pointer, in the r0/r1 return pair, or truncated to r0.
pub const SLOT_CLOCK_LO: u32 = 1;
pub const SLOT_CLOCK_HI: u32 = 2;

/// How many slots the block has. The scheduler writes exactly this many.
pub const SLOT_COUNT: usize = 3;

/// The current value of every mirror slot, in slot order.
///
/// The single definition of what the block holds. Each entry must be computed by the
/// SAME function its host handler calls, so the inline and host forms cannot drift;
/// `mirror_matches_its_handlers` holds them to that.
pub fn snapshot(st: &VitaState) -> [u32; SLOT_COUNT] {
    let now = st.now_us();
    [super::display::vcount(st), now as u32, (now >> 32) as u32]
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
        for slot in [SLOT_VCOUNT, SLOT_CLOCK_LO, SLOT_CLOCK_HI] {
            assert!((slot as usize) < SLOT_COUNT, "slot {slot} is outside the block");
        }
    }

    /// The clock's two slots must be ADJACENT, low first. Both pair forms read
    /// `mirror[slot]` and `mirror[slot + 1]` off one base address, so this adjacency is
    /// load-bearing: separating them still compiles and serves a clock whose high word is
    /// whatever else landed next door.
    #[test]
    fn the_clock_slots_are_adjacent_and_ordered() {
        assert_eq!(SLOT_CLOCK_HI, SLOT_CLOCK_LO + 1, "the high word follows the low word");
    }

    /// The whole `sceKernelGetProcessTime` family must read the same clock its handlers
    /// do, in the right halves. A swapped pair is a clock that runs about 4295 seconds
    /// per microsecond, which a title reads as enormous elapsed time - so it is not
    /// subtle, but nothing else in the system would attribute it here.
    #[test]
    fn the_clock_mirror_matches_its_handlers() {
        use crate::nid::libkernel as lk;
        use vitaslop_transpiler::InlineOp;
        // Values that span the 32-bit boundary in both directions, so a high word that is
        // simply never written (or written from the low half) cannot pass.
        for us in [0u64, 1, 0xFFFF_FFFF, 0x1_0000_0000, 0x1_2345_6789, 4_000_000_000] {
            let st = state_at(us);
            let words = snapshot(&st);
            let lo = words[SLOT_CLOCK_LO as usize];
            let hi = words[SLOT_CLOCK_HI as usize];
            assert_eq!(
                (u64::from(hi) << 32) | u64::from(lo),
                st.now_us(),
                "the mirrored pair must reassemble the clock at {us} us"
            );

            // ...and each spelling must name the clock's LOW slot as its base, since both
            // pair forms take the high word from the slot above it.
            for nid in [lk::GET_PROCESS_TIME, lk::GET_PROCESS_TIME_WIDE, lk::GET_PROCESS_TIME_LOW] {
                let op = super::super::libkernel::inline_op(nid).expect("has an inline form");
                assert_eq!(
                    op.mirror_slot(),
                    Some(SLOT_CLOCK_LO),
                    "{} must base at the clock's low slot",
                    crate::nid::name(nid)
                );
            }
            // The truncated spelling is the low word and nothing else.
            let low = super::super::libkernel::inline_op(lk::GET_PROCESS_TIME_LOW).unwrap();
            assert_eq!(low.eval(lo), st.now_us() as u32, "the Low spelling truncates");
            assert!(
                matches!(low, InlineOp::LoadMirror { .. }),
                "the Low spelling reads ONE word, not a pair"
            );
        }
    }

    /// A blocking or state-touching kernel call must never be inlined, however pure it
    /// looks: an inlined call never reaches the host, so the state change would simply
    /// not happen and the symptom would surface far away.
    #[test]
    fn only_the_clock_is_inlined_in_libkernel() {
        use crate::nid::libkernel as lk;
        for nid in [
            crate::vita::tm_nid::DELAY_THREAD, // blocks: that IS the behaviour
            lk::CREATE_THREAD,                 // spawns
            lk::EXIT_PROCESS,                  // halts the run
            lk::GET_TLS_ADDR,                  // per-thread host state
        ] {
            assert!(
                super::super::libkernel::inline_op(nid).is_none(),
                "{} has behaviour and must stay a host call",
                crate::nid::name(nid)
            );
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
