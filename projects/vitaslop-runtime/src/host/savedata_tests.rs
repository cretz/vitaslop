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
    assert_eq!(
        st.savedata_slot_get("savedata0:", 0).as_deref(),
        Some(param.as_slice())
    );
}

#[test]
fn slots_are_namespaced_by_mount_and_id() {
    let mut st = dummy_state();
    st.savedata_slot_put("savedata0:", 0, vec![1]);
    // A different slot id or mount is a distinct, still-empty slot.
    assert!(!st.savedata_slot_exists("savedata0:", 1));
    assert!(!st.savedata_slot_exists("savedata1:", 0));
    assert_eq!(
        st.savedata_slot_get("savedata0:", 0).as_deref(),
        Some([1].as_slice())
    );
}

#[test]
fn remove_reports_prior_existence() {
    let mut st = dummy_state();
    st.savedata_slot_put("", 3, vec![9, 9]);
    assert!(st.savedata_slot_remove("", 3));
    assert!(!st.savedata_slot_remove("", 3)); // already gone
    assert!(!st.savedata_slot_exists("", 3));
}
