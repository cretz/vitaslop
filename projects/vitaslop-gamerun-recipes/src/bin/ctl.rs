//! `ctl` - talk to a resident `session`.
//!
//! Appends one command to the session's control file, blocks until the session
//! writes the matching reply, prints it, and exits with the command's status. That
//! indirection exists for a mundane but decisive reason: every shell command an
//! agent runs is its own process, so a session reading its own stdin would be
//! unreachable the moment the shell that launched it returned. A shared directory
//! of append-only files is reachable from anywhere, needs no OS-specific IPC, and
//! leaves a complete transcript of the playthrough behind.
//!
//! Usage:
//!   ctl --dir <session-dir> "<command>" [more commands...]
//!
//! Options:
//!   --dir <dir>       the session directory (required)
//!   --timeout <secs>  give up waiting for a reply (default 900)
//!   --quiet           print only the reply body, no framing
//!
//! Several commands in one invocation run in order and stop at the first failure,
//! so a whole experiment is one call:
//!
//!   ctl --dir s "input l ly=0" "step 120" "watches" "shot after-throttle"
//!
//! Exit code is 0 when every command succeeded, 1 otherwise.

use std::path::PathBuf;
use std::process::ExitCode;
use std::time::{Duration, Instant};

use vitaslop_native::ControlDir;

/// How often to look for the reply. The session polls at 20 ms, so anything much
/// finer than this just burns CPU while a long `step` runs.
const POLL: Duration = Duration::from_millis(15);

fn main() -> ExitCode {
    let args: Vec<String> = std::env::args().collect();
    let mut dir: Option<PathBuf> = None;
    let mut timeout = 900u64;
    let mut quiet = false;
    let mut commands: Vec<String> = Vec::new();

    let mut i = 1;
    while i < args.len() {
        match args[i].as_str() {
            "--dir" | "--session" => {
                i += 1;
                dir = args.get(i).map(PathBuf::from);
            }
            "--timeout" => {
                i += 1;
                timeout = args.get(i).and_then(|s| s.parse().ok()).unwrap_or(timeout);
            }
            "--quiet" => quiet = true,
            "-h" | "--help" => {
                eprintln!("usage: ctl --dir <session-dir> \"<command>\" [\"<command>\" ...]");
                return ExitCode::from(2);
            }
            other => commands.push(other.to_string()),
        }
        i += 1;
    }

    let Some(dir) = dir else {
        eprintln!("error: --dir <session-dir> is required");
        return ExitCode::from(2);
    };
    if commands.is_empty() {
        eprintln!("error: no command given");
        return ExitCode::from(2);
    }
    let ctl = match ControlDir::new(&dir) {
        Ok(c) => c,
        Err(e) => {
            eprintln!("error: session dir {}: {e}", dir.display());
            return ExitCode::from(2);
        }
    };
    // A session takes a while to boot and fast-forward, and the natural thing to do
    // is start one and immediately queue the first command. So wait for it rather
    // than failing: the only unrecoverable case is that nobody ever shows up.
    if !ctl.ready_path().exists() {
        eprintln!("waiting for a session on {} ...", dir.display());
        let start = Instant::now();
        while !ctl.ready_path().exists() {
            if start.elapsed().as_secs() >= timeout {
                eprintln!(
                    "error: no session came up on {} within {timeout}s (start one with `session --game ... --dir ...`)",
                    dir.display()
                );
                return ExitCode::from(2);
            }
            std::thread::sleep(POLL);
        }
    }

    for cmd in commands {
        let seq = match ctl.request(&cmd) {
            Ok(s) => s,
            Err(e) => {
                eprintln!("error: writing command: {e}");
                return ExitCode::FAILURE;
            }
        };
        let start = Instant::now();
        loop {
            match ctl.read_reply(seq) {
                Ok(Some((ok, body))) => {
                    if !quiet {
                        println!("$ {cmd}");
                    }
                    if ok {
                        print!("{body}");
                    } else {
                        print!("ERR {body}");
                    }
                    if !ok {
                        return ExitCode::FAILURE;
                    }
                    break;
                }
                Ok(None) => {}
                Err(e) => {
                    eprintln!("error: reading reply: {e}");
                    return ExitCode::FAILURE;
                }
            }
            // A session that died mid-command would otherwise hang the client until
            // the timeout; its `ready` file disappearing is the tell.
            if !ctl.ready_path().exists() {
                eprintln!("error: the session exited before answering {cmd:?}");
                return ExitCode::FAILURE;
            }
            if start.elapsed().as_secs() >= timeout {
                eprintln!("error: timed out after {timeout}s waiting for {cmd:?}");
                return ExitCode::FAILURE;
            }
            std::thread::sleep(POLL);
        }
    }
    ExitCode::SUCCESS
}
