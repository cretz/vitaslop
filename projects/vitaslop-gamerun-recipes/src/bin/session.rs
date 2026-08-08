//! `session` - a RESIDENT emulator session an agent plays a game through.
//!
//! Boot a title once, leave it running, and drive it forward with commands. The
//! batch runner (`play`) re-boots and re-plays the whole prefix for every
//! experiment, which is a minute of emulation per question asked; a session pays
//! that once and answers every later question in the frames you actually asked for.
//!
//! Usage:
//!   session --game <app-dir> --dir <session-dir> [options] &
//!   ctl --dir <session-dir> "step 60"
//!
//! Options:
//!   --game <dir>        the extracted app directory (required)
//!   --dir <dir>         the session's control directory (required); `ctl` talks to
//!                       the session through it, and it holds the transcript
//!   --recipe <file>     seed the input timeline from a recipe, so a session can
//!                       start from an authored prefix instead of frame 0
//!   --shots <dir>       where screenshots and logs go (never inside the repo)
//!   --to <frame>        fast-forward to this frame before accepting commands
//!   --commands <file>   run these commands and exit (a scripted session; the same
//!                       command set, no control directory needed)
//!   --quantum-fuel <N>  scheduler preemption quantum (default 5000000)
//!   --idle-exit <secs>  exit after this long with no command (default 3600), so a
//!                       forgotten session cannot hold a core forever
//!
//! Everything the session can do is listed by its own `help` command, which is the
//! authoritative reference (`vitaslop_native::session::HELP`).
//!
//! `RUST_LOG` selects the engine's tracing diagnostics on stderr, exactly as for
//! `play` - e.g. `RUST_LOG=vitaslop::input=trace` to see every pad sample the guest
//! actually reads.

use std::path::PathBuf;
use std::process::ExitCode;
use std::time::{Duration, Instant};

use vitaslop_native::{ControlDir, Session, SessionOpts};
use vitaslop_runtime::Recipe;

/// How often the session looks for a new command. Short enough to feel immediate,
/// long enough that an idle session costs nothing.
const POLL: Duration = Duration::from_millis(20);

fn main() -> ExitCode {
    let _ = tracing_subscriber::fmt()
        .with_env_filter(tracing_subscriber::EnvFilter::new(vitaslop_runtime::knobs::log_filter()))
        .with_writer(std::io::stderr)
        .try_init();

    let args: Vec<String> = std::env::args().collect();
    let mut game: Option<String> = None;
    let mut dir: Option<PathBuf> = None;
    let mut recipe_path: Option<String> = None;
    let mut commands: Option<String> = None;
    let mut to: Option<u64> = None;
    let mut idle_exit = 3600u64;
    let mut opts = SessionOpts::default();

    let mut i = 1;
    while i < args.len() {
        let a = args[i].as_str();
        let mut next = || {
            i += 1;
            args.get(i).cloned()
        };
        match a {
            "--game" => game = next(),
            "--dir" => dir = next().map(PathBuf::from),
            "--recipe" => recipe_path = next(),
            "--commands" => commands = next(),
            "--shots" => opts.shot_dir = next().map(PathBuf::from),
            "--to" => to = next().and_then(|s| s.parse().ok()),
            "--idle-exit" => idle_exit = next().and_then(|s| s.parse().ok()).unwrap_or(idle_exit),
            "--quantum-fuel" => {
                opts.quantum_fuel = next().and_then(|s| s.parse().ok()).unwrap_or(opts.quantum_fuel)
            }
            "--list-knobs" => {
                // Every VITASLOP_* diagnostic knob the workspace reads, generated
                // from the source that reads it, so it is never out of date. Beats
                // grepping the tree, which is how these were found until now.
                print!("{}", vitaslop_runtime::knobs::INDEX);
                return ExitCode::SUCCESS;
            }
            "-h" | "--help" => {
                eprintln!("usage: session --game <dir> --dir <session-dir> [--recipe f] [--shots d] [--to N] [--commands f]");
                eprintln!("\n{}", vitaslop_native::session::HELP);
                return ExitCode::from(2);
            }
            other => {
                eprintln!("unknown argument {other:?}");
                return ExitCode::from(2);
            }
        }
        i += 1;
    }

    let Some(game) = game else {
        eprintln!("error: --game is required");
        return ExitCode::from(2);
    };
    if dir.is_none() && commands.is_none() {
        eprintln!("error: one of --dir (resident) or --commands (scripted) is required");
        return ExitCode::from(2);
    }

    // The seed recipe supplies the input prefix and the metadata header a `save`
    // will carry forward.
    let recipe = match &recipe_path {
        Some(p) => match std::fs::read_to_string(p).map_err(|e| e.to_string()).and_then(|t| {
            Recipe::parse(&t).map_err(|e| e.to_string())
        }) {
            Ok(r) => r,
            Err(e) => {
                eprintln!("error: reading recipe {p}: {e}");
                return ExitCode::from(2);
            }
        },
        None => Recipe::default(),
    };
    for n in &recipe.notes {
        eprintln!("# {} f{}: {}", if n.todo { "TODO" } else { "NOTE" }, n.frame, n.text);
    }

    let boot_start = Instant::now();
    let mut session = match Session::boot(&game, recipe, opts) {
        Ok(s) => s,
        Err(e) => {
            eprintln!("error: boot failed: {e}");
            return ExitCode::FAILURE;
        }
    };
    eprintln!("# booted in {:.1}s", boot_start.elapsed().as_secs_f64());

    if let Some(frame) = to {
        let t = Instant::now();
        match session.execute(&format!("step {frame}")) {
            Ok(status) => eprintln!(
                "# fast-forwarded to f{} in {:.1}s: {status}",
                session.frame(),
                t.elapsed().as_secs_f64()
            ),
            Err(e) => eprintln!("# fast-forward stopped: {e}"),
        }
    }

    // Scripted mode: run a command file top to bottom, echoing each command and its
    // reply, and exit. Same command set as the resident mode - a scripted session is
    // just a session nobody is watching.
    if let Some(path) = commands {
        let text = match std::fs::read_to_string(&path) {
            Ok(t) => t,
            Err(e) => {
                eprintln!("error: reading commands {path}: {e}");
                return ExitCode::from(2);
            }
        };
        let mut failed = false;
        for line in text.lines() {
            if line.trim().is_empty() || line.trim_start().starts_with('#') {
                continue;
            }
            if line.trim() == "quit" {
                break;
            }
            println!("$ {}", line.trim());
            match session.execute(line) {
                Ok(reply) => println!("{}", reply.trim_end()),
                Err(e) => {
                    println!("ERR {e}");
                    failed = true;
                }
            }
        }
        return if failed { ExitCode::FAILURE } else { ExitCode::SUCCESS };
    }

    // Resident mode: serve the control directory until told to quit or left idle.
    let dir = dir.expect("checked above");
    let ctl = match ControlDir::new(&dir) {
        Ok(c) => c,
        Err(e) => {
            eprintln!("error: session dir {}: {e}", dir.display());
            return ExitCode::FAILURE;
        }
    };
    // Consume whatever a previous session left in the control file, so a reused
    // directory does not replay old commands into a fresh boot.
    let mut consumed = ctl.poll(0).map(|(_, n)| n).unwrap_or(0);
    let _ = std::fs::write(
        ctl.ready_path(),
        format!("pid={} frame={} game={game}\n", std::process::id(), session.frame()),
    );
    eprintln!("# ready at f{} - talk to it with: ctl --dir {} \"<command>\"", session.frame(), dir.display());

    let mut last_activity = Instant::now();
    loop {
        let (reqs, n) = match ctl.poll(consumed) {
            Ok(v) => v,
            Err(e) => {
                eprintln!("error: reading control file: {e}");
                break;
            }
        };
        consumed = n;
        if reqs.is_empty() {
            if last_activity.elapsed().as_secs() >= idle_exit {
                eprintln!("# idle for {idle_exit}s, exiting");
                break;
            }
            std::thread::sleep(POLL);
            continue;
        }
        last_activity = Instant::now();
        for req in reqs {
            if req.line.trim() == "quit" {
                let _ = ctl.reply(req.seq, true, "bye");
                let _ = std::fs::remove_file(ctl.ready_path());
                eprintln!("# quit at f{}", session.frame());
                return ExitCode::SUCCESS;
            }
            let (ok, body) = match session.execute(&req.line) {
                Ok(r) => (true, r),
                Err(e) => (false, e),
            };
            if let Err(e) = ctl.reply(req.seq, ok, &body) {
                eprintln!("error: writing reply: {e}");
            }
        }
    }
    let _ = std::fs::remove_file(ctl.ready_path());
    ExitCode::SUCCESS
}
