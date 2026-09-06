//! SceLibDbg: the assertion and logging handlers behind the `SCE_DBG_ASSERT` /
//! `SCE_DBG_LOG` macros.
//!
//! These are the title's own diagnostics, and they are worth more here than they are
//! on hardware: a shipped title still calls them on the paths it considers
//! exceptional, so an assertion firing during bring-up names - in the title's own
//! words, with its own file and line - what it thinks went wrong. That message is
//! usually a better lead than anything the emulator can infer.
//!
//! Both are variadic, so they are hand-marshalled over the shared C formatter
//! (`vita::cfmt`) rather than written with `#[hostcall]`.

use crate::host::{GuestCtx, VitaState};
use crate::vita::cfmt;

/// Read a bounded NUL-terminated string from guest memory, or `None` for a null
/// pointer. `component` is documented as optional and callers do pass NULL.
fn opt_cstr(ctx: &GuestCtx, addr: u32) -> Option<String> {
    (addr != 0).then(|| super::iofilemgr::read_cstr(ctx, addr))
}

/// The formatted output SceLibDbg truncates at, per the header's own note.
const MAX_OUTPUT: usize = 511;

/// int sceDbgAssertionHandler(const char *file, int line, int unk,
///                            const char *component, const char *msg, ...)
///
/// Report a failed assertion. It is a LOGGING handler, not a terminator: the
/// `SCE_DBG_ASSERT` macro calls it and only then decides whether to break, so
/// halting the run here would stop titles that assert-and-continue by design.
///
/// Returns `unk` verbatim, which the header documents as this function's return
/// value.
pub(super) fn assertion_handler(ctx: &mut GuestCtx, st: &mut VitaState) {
    let file = opt_cstr(ctx, ctx.arg(0));
    let line = ctx.arg(1);
    let unk = ctx.arg(2);
    let component = opt_cstr(ctx, ctx.arg(3));
    let fmt_addr = ctx.arg(4);
    let mut out = Vec::new();
    // file, line, unk, component, msg are words 0..4; the variadic tail is word 5.
    cfmt::format_into(&mut out, ctx, fmt_addr, 5);
    out.truncate(MAX_OUTPUT);
    let msg = String::from_utf8_lossy(&out).trim_end().to_string();
    tracing::error!(
        target: "vitaslop::err",
        file = file.as_deref().unwrap_or("?"),
        line,
        component = component.as_deref().unwrap_or(""),
        "GUEST ASSERTION FAILED: {msg}"
    );
    // Also into the captured console, so it lands beside the title's own printf
    // output in the order it actually happened.
    st.write_stdout(format!("ASSERT {}:{line}: {msg}\n", file.as_deref().unwrap_or("?")).as_bytes());
    ctx.ret(unk);
}

/// int sceDbgLoggingHandler(const char *file, int line, SceDbgLogLevel logLevel,
///                          const char *component, const char *msg, ...)
///
/// The title's own log line. A line break is appended by the real handler, and the
/// output is truncated at 511 characters; the return is 0, or negative if it had to
/// truncate. Same argument layout as [`assertion_handler`], with the log level where
/// its `unk` sits.
pub(super) fn logging_handler(ctx: &mut GuestCtx, st: &mut VitaState) {
    let file = opt_cstr(ctx, ctx.arg(0));
    let line = ctx.arg(1);
    let level = ctx.arg(2);
    let component = opt_cstr(ctx, ctx.arg(3));
    let fmt_addr = ctx.arg(4);
    let mut out = Vec::new();
    cfmt::format_into(&mut out, ctx, fmt_addr, 5);
    let truncated = out.len() > MAX_OUTPUT;
    out.truncate(MAX_OUTPUT);
    let msg = String::from_utf8_lossy(&out).trim_end().to_string();
    tracing::debug!(
        target: "vitaslop::guest",
        file = file.as_deref().unwrap_or("?"),
        line,
        level,
        component = component.as_deref().unwrap_or(""),
        "{msg}"
    );
    st.write_stdout(format!("{msg}\n").as_bytes());
    ctx.ret(if truncated { -1i32 as u32 } else { 0 });
}
