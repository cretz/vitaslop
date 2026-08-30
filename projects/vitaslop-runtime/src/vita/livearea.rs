//! SceLiveAreaUtil: what a title can read back about its own LiveArea gate.
//!
//! # There is no published interface for this library
//! vitasdk ships no header for it, and neither wiki has a page - only the NIDs and the
//! fact that its errors live in the `0x801040XX` facility. Both entry points here were
//! REVERSE-ENGINEERED from the calling title, from the instruction stream around the
//! call, in the way [[vitaslop-re-undocumented-nid-from-callsite]] describes. The
//! evidence for each claim is recorded at the call it came from, below.
//!
//! # Where the answer comes from, and why it is not invented
//! A LiveArea gate is declared by the title's OWN package, in
//! `sce_sys/livearea/contents/template.xml`: each `<frame id="..." rev="N">` is one
//! panel of the gate. So "does frame1 exist, and what revision is it" is a question the
//! bytes we were handed already answer, and this module answers it by reading that file
//! rather than by fabricating a gate. A frame the template does not declare does not
//! exist, and the call reports so.
//!
//! The other half of a frame is its USER DATA - a blob the title asks the shell to
//! store alongside the panel, so the gate can show a last score or a save slot. Nothing
//! in this emulator runs a shell, and no title running here has ever completed a
//! LiveArea update, so there is no user data for any frame. That is the state of a
//! freshly installed title on a real console, and it is what `GetFrameUserData`
//! reports - it does NOT write a zero-filled buffer and call it a stored blob.

use crate::host::{GuestCtx, Ptr, VitaState};
use crate::hostcall;

/// The title's own gate declaration, inside its package.
const TEMPLATE_PATH: &str = "sce_sys/livearea/contents/template.xml";

/// Longest frame id worth reading from the guest. Ids in a template are short names
/// like `frame3`; this is generous.
const FRAME_NAME_MAX: usize = 64;

/// A `SceLiveAreaUtil` error meaning "this title's gate has no such frame".
///
/// **The low byte is NOT established.** The only public fact about this library's error
/// space is the facility - `0x801040XX`, from the wikis' error-code table - and no
/// source names the individual codes. So the FACILITY here is evidence and the code
/// within it is ours. It is used only where the honest answer is a failure and the
/// alternative would be a fabricated success, and [`report_unestablished`] says so on
/// stderr the first time either call takes this path, so it can never be mistaken for
/// a verified constant by a later reader of a trace.
const SCE_LIVEAREA_UTIL_ERROR_NOT_FOUND: i32 = 0x8010_4001u32 as i32;

/// Say once, out loud, that the error code above is our choice and not a documented
/// one. Unconditional (not behind a knob) for the reason in
/// [[vitaslop-fallback-must-report]]: a report nobody enabled is a report nobody reads.
fn report_unestablished(what: &str) {
    static SAID: std::sync::Once = std::sync::Once::new();
    SAID.call_once(|| {
        tracing::warn!(
            "SceLiveAreaUtil: {what} - reporting failure with {SCE_LIVEAREA_UTIL_ERROR_NOT_FOUND:#010x}. \
             The 0x801040XX FACILITY is documented; the code within it is NOT, and this one is ours"
        );
    });
}

/// Every `<frame id="..." rev="N">` the title's template declares, as `(id, rev)`.
///
/// A deliberately minimal scan rather than an XML parser: the only thing wanted is the
/// two attributes on the `frame` element, this file is authored by the title's own
/// packaging tool in a fixed shape, and pulling a real XML dependency into the runtime
/// (which compiles to wasm) to read seven short attributes would cost far more than it
/// buys. `rev` defaults to 0 when absent - the template's own default for a frame that
/// has never been revised.
fn declared_frames(st: &mut VitaState) -> Vec<(String, u64)> {
    match st.read_file(TEMPLATE_PATH) {
        Some(bytes) => parse_frames(&bytes),
        None => Vec::new(),
    }
}

/// [`declared_frames`] over the template BYTES - the whole of the parsing, split out so
/// it is testable without a filesystem.
fn parse_frames(bytes: &[u8]) -> Vec<(String, u64)> {
    let text = String::from_utf8_lossy(bytes);
    let mut out = Vec::new();
    for tag in text.split('<').filter(|t| t.starts_with("frame ")) {
        let Some(id) = attr(tag, "id") else { continue };
        let rev = attr(tag, "rev").and_then(|r| r.parse::<u64>().ok()).unwrap_or(0);
        out.push((id, rev));
    }
    out
}

/// The value of `name="..."` inside one element's text, or `None`.
fn attr(tag: &str, name: &str) -> Option<String> {
    let at = tag.find(&format!("{name}=\""))? + name.len() + 2;
    let rest = &tag[at..];
    Some(rest[..rest.find('"')?].to_string())
}

/// int sceLiveAreaGetFrameRevision(const char *frameName, SceUInt64 *revision)
///
/// # How the shape was established (title at `0x8138a20a`)
/// The caller loops seven times over frame names it builds itself with
/// `sprintf(buf, "frame%d", i)`, and calls this with `r0 = buf`, `r1 = <record>`. Only
/// `r0` and `r1` are set before the branch - `r2` still holds the value the sprintf
/// left - so it takes TWO arguments. The record it passes advances by 1032 bytes per
/// iteration and the NEXT call ([`get_frame_user_data`]) is handed `record + 8` with a
/// size of 1024, so the revision this writes is the 8 bytes at `record + 0`: a 64-bit
/// value, which is also what a monotonically-increasing revision would be.
///
/// The caller ignores the return code (the instruction after the branch sets up the
/// next call, it does not test `r0`), so the CORRECTNESS of this call is entirely in
/// what it writes - which is exactly why an unwritten out-param would be the bug here.
#[hostcall]
pub(super) fn get_frame_revision(ctx: &mut GuestCtx, st: &mut VitaState, name: Ptr, revision: Ptr) -> i32 {
    let name = if name.is_null() { String::new() } else { ctx.read_cstr(name.addr(), FRAME_NAME_MAX) };
    let rev = declared_frames(st).into_iter().find(|(id, _)| *id == name).map(|(_, rev)| rev);
    // Write the out-param on BOTH paths. The caller does not check the return, so a
    // revision left unwritten is read as whatever the buffer held - and the failing
    // case must read as "no revision", which is 0, not as a stale one.
    if !revision.is_null() {
        let v = rev.unwrap_or(0);
        ctx.write_u32(revision.addr(), v as u32);
        ctx.write_u32(revision.addr() + 4, (v >> 32) as u32);
    }
    match rev {
        Some(_) => 0,
        None => {
            report_unestablished("a frame the title's own template.xml does not declare");
            SCE_LIVEAREA_UTIL_ERROR_NOT_FOUND
        }
    }
}

/// int sceLiveAreaGetFrameUserData(const char *frameName, void *data, SceSize size)
///
/// # How the shape was established (title at `0x8138a218`)
/// Same loop, the very next call: `r0 = buf` (the same frame name), `r1 = record + 8`,
/// `r2 = 1024`. Three arguments, the third a BUFFER SIZE - it is a constant that
/// matches the gap between one 1032-byte record and the 8-byte revision at its head.
///
/// # Why this reports failure rather than writing zeros
/// User data is written by the shell when a title completes a LiveArea update. No
/// shell runs here and no update has ever completed, so no frame has user data. A
/// zero-filled buffer would be a title's stored blob that happens to be all zeros -
/// a different, and false, statement. The caller's buffer is left exactly as it found
/// it.
#[hostcall]
pub(super) fn get_frame_user_data(_ctx: &mut GuestCtx, _st: &mut VitaState) -> i32 {
    report_unestablished("no LiveArea user data exists - nothing here runs a shell that could have stored any");
    SCE_LIVEAREA_UTIL_ERROR_NOT_FOUND
}

#[cfg(test)]
mod tests {
    use super::{attr, parse_frames};

    /// The scan reads the two attributes it needs off a real template's shape, and
    /// ignores the `<livearea>` element's own `content-rev` (which is not a frame).
    #[test]
    fn frames_come_from_the_template() {
        let xml = r#"<?xml version="1.0" encoding="utf-8"?>
<livearea style="a3" format-ver="01.00" content-rev="1">
  <gate><startup-image>startup.png</startup-image></gate>
  <frame id="frame3" multi="o" autoflip="0" rev="1">
    <liveitem><image>a.png</image></liveitem>
  </frame>
  <frame id="frame5" multi="o" autoflip="0" rev="2"/>
  <frame id="frame7"/>
</livearea>"#;
        assert_eq!(
            parse_frames(xml.as_bytes()),
            vec![("frame3".into(), 1), ("frame5".into(), 2), ("frame7".into(), 0)]
        );
    }

    #[test]
    fn attr_reads_one_value() {
        assert_eq!(attr(r#"frame id="frame3" rev="4""#, "rev"), Some("4".into()));
        assert_eq!(attr(r#"frame id="frame3""#, "rev"), None);
    }
}
