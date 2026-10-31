//! The key-slot layout behind `sceKernelGetTLSAddr`: callers index it like an array.
use super::*;
use crate::world::DeterministicWorld;

fn state() -> VitaState {
    VitaState::new(0x1000, 0x10000, Box::new(DeterministicWorld::default()))
}

/// Consecutive keys are consecutive words of ONE per-thread array: a caller that keeps a
/// struct in adjacent keys reads its own fields, not whatever was allocated next.
#[test]
fn consecutive_keys_are_consecutive_words_of_one_array_per_thread() {
    let mut st = state();
    let a = st.tls_addr_of(0x40, 0x8c).unwrap();
    assert_eq!(st.tls_addr_of(0x40, 0x8d), Some(a + 4));
    assert_eq!(st.tls_addr_of(0x40, 0x8e), Some(a + 8));
    assert_eq!(
        st.tls_addr_of(0x40, 0),
        Some(a - 0x8c * 4),
        "key 0 is the array base"
    );
    assert_eq!(st.tls_addr_of(0x40, 0x8c), Some(a), "stable across calls");
    let b = st.tls_addr_of(0x41, 0x8c).unwrap();
    assert_ne!(a, b, "each thread has its own array");
    assert!(
        b.abs_diff(a) >= (TLS_KEY_MAX + 1) * 4,
        "the arrays do not overlap"
    );
}

/// The last key has a slot; the one past it is refused rather than handed memory nobody owns.
#[test]
fn a_key_past_the_last_slot_is_refused() {
    let mut st = state();
    assert!(st.tls_addr_of(0x40, TLS_KEY_MAX).is_some());
    assert_eq!(st.tls_addr_of(0x40, TLS_KEY_MAX + 1), None);
}
