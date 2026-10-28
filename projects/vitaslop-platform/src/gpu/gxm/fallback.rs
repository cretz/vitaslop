//! A pair the recompiler cannot translate: dropping its draws and saying so, the evidence a
//! refusal carries, the per-draw fallback reasons, and the branch measurement.

use super::*;

/// >>> A REFUSAL THAT DOES NOT HAND OVER THE EVIDENCE COSTS A PLAY SESSION.
///
/// When the recompiler refuses a pair, the ONE thing needed to fix it is that pair's two
/// containers - and until this existed the only way to get them was to know in advance to
/// set `VITASLOP_DUMP_GXP_BIN` and then reach the same draw again. On a title where the
/// refusal is several holes into a round on one course, that is a play session per attempt,
/// and the user who first hit `7089f16e34be693f` could not be asked to repeat it on demand.
///
/// So the refusal carries the blobs with it, base64 in the panic text, which reaches the
/// browser's diagnostics panel - the thing a user can already copy in one tap. Pasting that
/// back reconstitutes the exact pair offline, where `tests/corpus.rs` answers in a second
/// what a device answers in an evening.
///
/// Only on the FATAL path (`allow_fixed_function` off) and only for the first pair, because
/// this is kilobytes of text: a warning that fires per draw would bury the panel
/// [[vitaslop-a-diagnostic-can-bury-the-findings]].
pub(super) fn base64(bytes: &[u8]) -> String {
    const A: &[u8; 64] = b"ABCDEFGHIJKLMNOPQRSTUVWXYZabcdefghijklmnopqrstuvwxyz0123456789+/";
    let mut out = String::with_capacity(bytes.len().div_ceil(3) * 4);
    for c in bytes.chunks(3) {
        let (b0, b1, b2) = (c[0] as u32, *c.get(1).unwrap_or(&0) as u32, *c.get(2).unwrap_or(&0) as u32);
        let n = (b0 << 16) | (b1 << 8) | b2;
        out.push(A[(n >> 18) as usize & 63] as char);
        out.push(A[(n >> 12) as usize & 63] as char);
        out.push(if c.len() > 1 { A[(n >> 6) as usize & 63] as char } else { '=' });
        out.push(if c.len() > 2 { A[n as usize & 63] as char } else { '=' });
    }
    out
}

/// The pair whose containers a fatal refusal should print, set by the recompile path just
/// before it may refuse. Cleared after use so only the first refusal carries the payload.
pub(super) fn pending_blobs() -> &'static std::sync::Mutex<Option<(u64, std::sync::Arc<[u8]>, std::sync::Arc<[u8]>)>> {
    static P: std::sync::OnceLock<
        std::sync::Mutex<Option<(u64, std::sync::Arc<[u8]>, std::sync::Arc<[u8]>)>>,
    > = std::sync::OnceLock::new();
    P.get_or_init(|| std::sync::Mutex::new(None))
}

/// Record the containers a subsequent `report_fallback` for `key` should carry.
pub(crate) fn arm_fallback_blobs(key: u64, vprog: std::sync::Arc<[u8]>, fprog: std::sync::Arc<[u8]>) {
    *pending_blobs().lock().unwrap_or_else(|e| e.into_inner()) = Some((key, vprog, fprog));
}

/// The base64 of the armed pair, if it is the one being refused.
pub(super) fn blob_evidence(key: u64) -> String {
    let armed = pending_blobs().lock().unwrap_or_else(|e| e.into_inner()).take();
    match armed {
        Some((k, v, f)) if k == key => format!(
            "\n\n>>> THE PAIR, so this does not have to be reproduced to be fixed. Base64 of \
             the two `SceGxmProgram` containers; decode each to a .gxp and point \
             VITASLOP_GXP_CORPUS at the directory.\nVERTEX {} bytes:\n{}\nFRAGMENT {} \
             bytes:\n{}\n",
            v.len(),
            base64(&v),
            f.len(),
            base64(&f),
        ),
        _ => String::new(),
    }
}

/// >>> A PAIR THIS RECOMPILER CANNOT TRANSLATE DROPS ITS DRAWS. It does not approximate
/// >>> them, and it no longer kills the run.
///
/// Three things can be done with a draw whose shader pair will not recompile, and only one
/// is honest at every stage of the project:
///
/// * DRAW IT WITH THE FIXED-FUNCTION APPROXIMATION. A different renderer, which never runs
///   the guest's shader; what it produces is plausible and wrong and cannot be told from a
///   faithful render by looking at the frame. It is how one title's world silently rendered
///   328 of 388 draws wrong. Still available behind `VITASLOP_GXP_ALLOW_FIXED_FUNCTION=1`,
///   still bring-up only.
/// * PANIC. What this did. For a headless development run that is right - stop at the first
///   thing the recompiler cannot translate. On a PLAYER's device it is the worst of the
///   three: the worker dies and the game with it. MEASURED - a phone reached a demo fight,
///   hit ONE blocked instruction in ONE of this title's ~70 vertex programs (`0xF8 LIMM`,
///   whose published field list contradicts itself and stays blocked on purpose), and the
///   run ended mid-match with `RuntimeError: unreachable`.
/// * DROP THE DRAW AND SAY SO. Nothing wrong is painted - the mesh is simply ABSENT, a
///   difference a player and a developer can both see - the run continues, and the pair,
///   the reason and both containers are reported. That is the default now.
///
/// The rule the panic protected is "never show an approximation and call it a render", and
/// dropping keeps it: absence is not a wrong picture. `VITASLOP_GXP_STRICT=1` restores the
/// hard stop for a run that wants to fail at the first untranslatable pair.
pub(super) fn report_fallback(key: u64, reason: &str) {
    use std::sync::{Mutex, OnceLock};
    static SEEN: OnceLock<Mutex<HashSet<u64>>> = OnceLock::new();
    let seen = SEEN.get_or_init(|| Mutex::new(HashSet::default()));
    // A poisoned lock must not lose the diagnostic - recover the set and report anyway.
    let mut seen = seen.lock().unwrap_or_else(|e| e.into_inner());
    let first = seen.insert(key);
    drop(seen);
    if first {
        if allow_fixed_function() {
            // The bring-up path. It was asked for, the picture is approximate by consent,
            // and a base64 dump of both containers per pair would bury the run that asked
            // for it - a corpus-collection pass falls back on everything by design.
            report_warn!(
                "gxp pair {key:016x}: FALLS BACK to the fixed-function approximation - {reason}"
            );
        } else {
            // The default path, and the one that costs a picture: carry the two containers
            // so the pair never has to be reproduced to be fixed.
            report_warn!(
                "gxp pair {key:016x}: cannot be recompiled, so every draw using it is DROPPED - its mesh is ABSENT from the frame rather than approximated - {reason}{}",
                blob_evidence(key)
            );
        }
    }
    fallback_reasons().lock().unwrap_or_else(|e| e.into_inner()).insert(key, reason.to_string());
    if gxp_strict() {
        panic!(
            "gxp pair {key:016x} cannot be recompiled: {reason}
                 VITASLOP_GXP_STRICT is set, so the run stops at the first pair the recompiler                  cannot translate rather than dropping its draws. Unset it to carry on with the                  mesh absent, or set VITASLOP_GXP_ALLOW_FIXED_FUNCTION=1 to approximate it                  (bring-up only: it is how a title's world silently rendered 328 of 388 draws                  wrong)."
        );
    }
}

/// `VITASLOP_GXP_STRICT=1`: stop the run at the first shader pair the recompiler cannot
/// translate, instead of dropping that pair's draws and carrying on. For a development run
/// that wants to fail loudly; see [`report_fallback`] for why it is not the default.
pub(super) fn gxp_strict() -> bool {
    use std::sync::OnceLock;
    static ON: OnceLock<bool> = OnceLock::new();
    *ON.get_or_init(|| crate::knobs::flag("VITASLOP_GXP_STRICT"))
}

/// Whether a shader pair the recompiler cannot translate may be drawn by the
/// fixed-function approximation instead (`VITASLOP_GXP_ALLOW_FIXED_FUNCTION=1`).
///
/// OFF by default, and that default is the point. The approximation reconstructs a draw
/// from captured state without running the guest's fragment program, so what it produces
/// is plausible and wrong, and indistinguishable on screen from a correct render. It
/// remains available for bring-up, where seeing a menu at all is what lets a playbook be
/// authored - but a run that does not ask for it now stops at the first pair it cannot
/// translate, naming the pair and the reason.
pub(super) fn allow_fixed_function() -> bool {
    use std::sync::OnceLock;
    static ON: OnceLock<bool> = OnceLock::new();
    *ON.get_or_init(|| crate::knobs::flag("VITASLOP_GXP_ALLOW_FIXED_FUNCTION"))
}

/// Why each pair last fell back, keyed by pair.
///
/// [`report_fallback`] PRINTS once per pair but RECORDS every time, so a per-scene tally
/// can weight reasons by how many DRAWS each cost. That is the number that says what to
/// fix next, and the printed list cannot give it: sixty pairs that fall back on one draw
/// each and one pair that falls back on three hundred read exactly the same there.
pub(super) fn fallback_reasons() -> &'static std::sync::Mutex<HashMap<u64, String>> {
    use std::sync::{Mutex, OnceLock};
    static REASONS: OnceLock<Mutex<HashMap<u64, String>>> = OnceLock::new();
    REASONS.get_or_init(|| Mutex::new(HashMap::default()))
}

/// Record why a DRAW fell back, without printing.
///
/// >>> WHY THIS EXISTS SEPARATELY FROM [`report_fallback`]. That one is for a pair the
/// recompiler cannot TRANSLATE: it prints once per pair, carries the containers, and trips
/// `VITASLOP_GXP_STRICT`. But `prepare` returns `None` on several paths that are not
/// translation failures at all - empty geometry, a knob filter, a pipeline the cache does
/// not hold - and none of them recorded anything, so the per-scene tally attributed every
/// one of those draws to `(no reason recorded)`.
///
/// MEASURED: on a football title's kickoff that placeholder was **588 fallback draws a
/// frame - the whole of that title's remaining fallback population** - with nothing in any
/// report naming a cause. The comment on [`fallback_reason_of`] said an unrecorded reason
/// "would itself be a bug worth seeing"; it was there to be seen and nothing showed it.
///
/// Silent on purpose: these are per-DRAW conditions, several of them entirely normal, and
/// printing one per pair would bury the translation failures that matter.
pub(super) fn record_fallback_reason(key: u64, reason: &str) {
    let mut g = fallback_reasons().lock().unwrap_or_else(|e| e.into_inner());
    // Never overwrite a TRANSLATION failure with a per-draw one: a pair that cannot be
    // recompiled also has no pipeline entry, so the later, blander reason would otherwise
    // replace the one that names the defect.
    g.entry(key).or_insert_with(|| reason.to_string());
}

/// The recorded fallback reason for `key`, or a placeholder if the pair fell back before
/// any reason was recorded (which would itself be a bug worth seeing).
pub(super) fn fallback_reason_of(key: u64) -> String {
    fallback_reasons()
        .lock()
        .unwrap_or_else(|e| e.into_inner())
        .get(&key)
        .cloned()
        .unwrap_or_else(|| "(no reason recorded)".to_string())
}

/// Report, once per pair, every USSE control-flow branch either program contains: the
/// branch's instruction index, its signed word delta, the target that resolves to, and the
/// program's instruction count.
///
/// This is a MEASUREMENT, not a debug aid, and that is why it is unconditional like
/// [`report_fallback`] rather than behind a flag. Branch translation rests on one fact the
/// distilled ISA reference states but the captured shader corpus cannot corroborate (it
/// contains no branches at all): that a target is `index + rel` rather than `index + 1 +
/// rel`. Those differ by exactly one instruction, so the wrong reading leaves the last
/// instruction of every conditional block running unconditionally - a plausible wrong
/// picture with no error anywhere. A branch whose target lands exactly on `total` (one past
/// the end, the shape a trailing `if` compiles to) confirms the implemented reading;
/// targets that cluster one short of it, with none ever reaching `total`, would refute it.
pub(super) fn report_branches(key: u64, gxp: &GxpRecompile) {
    use std::sync::{Mutex, OnceLock};
    static SEEN: OnceLock<Mutex<HashSet<u64>>> = OnceLock::new();
    let seen = SEEN.get_or_init(|| Mutex::new(HashSet::default()));
    let mut seen = seen.lock().unwrap_or_else(|e| e.into_inner());
    if !seen.insert(key) {
        return;
    }
    for (kind, bytes) in [("vertex", &gxp.vprog), ("fragment", &gxp.fprog)] {
        let Ok((total, sites)) = vitaslop_gxp_shader::branch_sites(bytes) else { continue };
        for s in sites {
            report!(
                "gxp pair {key:016x}: {kind} BR #{} rel={} -> target {} of {total} instrs",
                s.index, s.rel, s.target
            );
        }
    }
}
