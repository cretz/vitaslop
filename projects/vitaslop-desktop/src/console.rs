//! Where the binary's output goes on Windows.
//!
//! # The shell is a window, not a console program
//! A release build is linked as a GUI-subsystem program (`windows_subsystem` in `main.rs`), so
//! double-clicking `vitaslop.exe` opens the library and nothing else. Linked as a console
//! program it opened a terminal window first, which is what a player sees as "a cmd window
//! before the app".
//!
//! # Every other entry point is read from a terminal
//! `--game`, `--headless`, `import`, `list`, `serve` and the rigs print what they did, and a GUI
//! program starts with no console to print to. So when the binary is given arguments, the
//! console of whoever started it is attached - but ONLY for a stream that has nowhere to go
//! already. A rig's `> log.txt 2>&1` hands the process valid file handles, and replacing them
//! with the console would send the whole log to a terminal nobody is watching and leave the
//! file empty. Each of stdout and stderr is judged on its own: `2> err.txt` alone keeps the
//! file for stderr and attaches the console for stdout.
//!
//! A GUI program does not hold an interactive `cmd` prompt while it runs, so typed at one the
//! prompt returns at once and the output lands under it. bash, `cmd /c` and batch files wait
//! and redirect as before. Windows PowerShell does NOT: it neither waits for a GUI program nor
//! connects its `>` / `2>&1` to one (MEASURED: an empty file and no exit code), so from
//! PowerShell run it through `cmd /c "vitaslop.exe ... > log.txt 2>&1"` or `Start-Process -Wait
//! -NoNewWindow -RedirectStandardOutput ... -RedirectStandardError ...`.

/// Whether the parent's console should be attached: arguments were given (a terminal entry
/// point, not the double-clicked shell) and at least one output stream has no handle. Only
/// Windows asks; the rule's test runs everywhere.
#[cfg(any(windows, test))]
pub fn wants_console(cli: bool, stdout_ok: bool, stderr_ok: bool) -> bool {
    cli && !(stdout_ok && stderr_ok)
}

/// Attach the parent's console for the output streams that have no handle - see the module
/// docs. A no-op off Windows, for the shell, and when both streams are already redirected.
#[cfg(windows)]
pub fn attach_parent(cli: bool) {
    use windows_sys::Win32::Foundation::{GENERIC_READ, GENERIC_WRITE, HANDLE, INVALID_HANDLE_VALUE};
    use windows_sys::Win32::Storage::FileSystem::{CreateFileW, FILE_SHARE_READ, FILE_SHARE_WRITE, OPEN_EXISTING};
    use windows_sys::Win32::System::Console::{
        ATTACH_PARENT_PROCESS, AttachConsole, GetStdHandle, STD_ERROR_HANDLE, STD_HANDLE, STD_OUTPUT_HANDLE, SetStdHandle,
    };
    let valid = |h: HANDLE| !h.is_null() && h != INVALID_HANDLE_VALUE;
    // SAFETY: plain Win32 calls on this process's own standard handles, made at start-up
    // before any other thread exists or any output has been written.
    unsafe {
        let ok = |id: STD_HANDLE| valid(GetStdHandle(id));
        if !wants_console(cli, ok(STD_OUTPUT_HANDLE), ok(STD_ERROR_HANDLE)) {
            return;
        }
        let missing: Vec<STD_HANDLE> = [STD_OUTPUT_HANDLE, STD_ERROR_HANDLE].into_iter().filter(|&id| !ok(id)).collect();
        // No console to attach to (started from Explorer with arguments, or by a service):
        // the streams stay without a handle and Rust's stdio discards what is written to them.
        if AttachConsole(ATTACH_PARENT_PROCESS) == 0 {
            return;
        }
        let name: Vec<u16> = "CONOUT$\0".encode_utf16().collect();
        for id in missing {
            // Attaching may already have filled the slot; a handle that is valid now stays.
            if ok(id) {
                continue;
            }
            let h = CreateFileW(
                name.as_ptr(),
                GENERIC_READ | GENERIC_WRITE,
                FILE_SHARE_READ | FILE_SHARE_WRITE,
                std::ptr::null(),
                OPEN_EXISTING,
                0,
                std::ptr::null_mut(),
            );
            if valid(h) {
                SetStdHandle(id, h);
            }
        }
    }
}

#[cfg(not(windows))]
pub fn attach_parent(_cli: bool) {}

#[cfg(test)]
mod tests {
    use super::wants_console;

    #[test]
    fn only_a_terminal_entry_point_with_an_unredirected_stream_attaches() {
        assert!(!wants_console(false, false, false), "the double-clicked shell never opens a console");
        assert!(wants_console(true, false, false), "a rig typed at a terminal prints there");
        assert!(!wants_console(true, true, true), "`> log 2>&1` keeps both files");
        assert!(wants_console(true, true, false), "`> log` alone still shows stderr");
        assert!(wants_console(true, false, true), "`2> err` alone still shows stdout");
    }
}
