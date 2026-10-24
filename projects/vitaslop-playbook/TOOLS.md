# The playbook tools - how a playbook gets written

These are the native tools for finding out what a title responds to and where its state lives,
before that knowledge goes into a playbook. They need the native engine, so they build with this
crate's default `native` feature (the browser build turns it off).

## The tools

| tool | what it is for |
|---|---|
| `session` + `ctl` | a RESIDENT game session you drive forward command by command. The main tool. Boot once, then explore for as long as you like. |
| `explore` | try many candidate inputs at one frame, in parallel, and report which of them did something different. |
| `bench` | what a frame COSTS on a chosen screen, and where the time went. |
| `memdiff` | diff two runs' guest memory and see exactly what an input touched. How you find the address of a live value. |
| `sheet` | montage a run's screenshots into one labelled grid. |

```
cargo build --release -p vitaslop-playbook    # builds them all
```

The session takes `--list-knobs` to print the
generated index of `VITASLOP_*` diagnostics - see [KNOBS.md](../KNOBS.md).

## Start here: the resident session

The single biggest cost in playing a game through an emulator is not the emulator, it
is REPLAY. Reaching a decision frame deep in a title takes a minute of emulation, and
tuning one input is that minute times fifty. The prefix is not the experiment, it is
the tax on the experiment.

A session pays it once:

```
# start it in the background; it boots and fast-forwards to live gameplay
session --game /path/to/app --dir /tmp/scratch/sess \
        --tape /tmp/scratch/run/playbook.tape \
        --shots /tmp/scratch/shots --to 1800 &

# then talk to it - each call returns in the time the frames actually take
ctl --dir /tmp/scratch/sess "info"
ctl --dir /tmp/scratch/sess "press l --hold 60" "shot after-throttle" "watches"
```

`ctl` appends your command to a file in the session directory and blocks until the
session writes the reply. That indirection exists for a mundane but decisive reason:
every shell command an agent runs is its own process, so a session reading its own
stdin would be unreachable the moment the shell that launched it returned. Two
append-only files in a shared directory are reachable from anywhere, need no
OS-specific IPC, and leave a complete transcript of the playthrough behind.

Several commands in one `ctl` call run in order and stop at the first failure, so a
whole experiment is one invocation.

Run `ctl --dir <dir> help` for the full command set. The important ones:

- `step [N]`, `until <watch> <op> <value>` - advance time.
- `input <tokens>`, `press <tokens> --hold N` - the one-line input grammar
  (`vitaslop-runtime/src/timeline.rs`): buttons, `lx=`..`ry=`, `lang=`/`rang=`, `touch=X,Y`, `back=X,Y`.
- `read`, `peek`, `poke`, `watch`, `dump` - see and touch guest memory.
- `scan ...` - the in-session value finder (read the caveat below).
- `shot`, `sig`, `egress`, `section`, `note` - observe and annotate (the transcript keeps them).

### Starting deep: `--tape`

`session`, `explore` and `bench` reach their starting frame by replaying a playbook run's tape
(`--tape <run>/playbook.tape`): its input changes, at the display frames they were recorded on.
That reproduces the run where display frames repeat - the deterministic engine, one build. The
durable artifact is the playbook, not the session: what a session learns goes into a playbook
step, which waits on what the game does instead of on a frame number.

### What a session cannot do

A session runs FORWARD only. There is no rewind and no state snapshot, and that is
not an oversight: the transpiler turns each guest function into a wasm function and
each guest call into a wasm call, so a suspended guest thread's state lives partly in
a live wasm call stack, which cannot be serialized and reloaded. Restoring one would
mean making every translated function re-enterable at every call site and rebuilding
the whole stack - a project, not a feature.

What recovers the same benefit is running forward in PARALLEL: `explore` starts N
sessions at once, so N candidate experiments cost ONE prefix of wall-clock, not N.

Also: rebuilding the workspace fails while a session is running (the executable is
locked). `ctl --dir <dir> quit` first.

## Seeing where things are: `locate`

The slowest thing an agent does is look at a screenshot. A PNG tells a human where
the car is; it tells a program nothing, so every navigation decision needs a person
in the loop and a playthrough cannot run unattended.

`locate` answers the same question in numbers, and needs no reverse engineering to do
it. Every captured draw carries the model-to-world matrix its vertex program was
given, so the translation IS the object's world position and the rotation IS its
heading. Grouping the draws by that placement turns a scene of several hundred draws
into the handful of OBJECTS the frame contains:

```
ctl --dir <sess> "locate --min-tris 50"

id=0x1e6e491a97a7b935 draw301 world=(12.73,-49.96,109.25) heading=(-88.2,1.8)
                             screen=(481,268) box=(463,261)-(500,276) dist=120.4 tris=4609
```

- `id` is a hash of the object's OBJECT-SPACE geometry, so it is stable frame to
  frame. The draw index is not: the draw list is rebuilt every frame and its length
  changes as things come into view, so a delta matched on draw index reports enormous
  motion for a world that barely moved. Follow one object with `--id <hex>`.
- `heading` is the bearing of the object's local +X and +Z axes, in the SAME
  convention as the `lang=` stick directive - so a commanded bearing and a measured
  heading are directly comparable numbers.
- `--moving` shows only what changed world position since the last `locate`, which is
  how you find the player object: step with the throttle held and see what responds.
  Static scenery has a zero world delta even as the camera pans over it.

Two things this makes cheap that were not:

1. **Steering is arithmetic.** Read the heading, compute the bearing you want, set
   `lang=`. Without the rotation you have to infer facing by driving and watching
   where the car ends up - which takes long enough that it hits a wall first, and the
   answer is contaminated by the turn it made getting there.
2. **Wedged is detectable.** A car nosed into a wall reports `moved=0` with the
   throttle held. That is the difference between a loop that notices and reverses and
   an agent that stares at a PNG wondering why nothing is happening.

Objects that share one mesh (a row of identical cones) share one `id`; `--moving`
pairs them up by nearest previous position, which is right over one step but is worth
knowing when a reading looks odd.

## Finding the address of a live value

To assert anything about gameplay you need an address, and a title ships no symbols.
There are two tools here and they are good at opposite things. Read this section
before reaching for either - the obvious one is the weaker one.

### The honest version: a single run cannot tell input apart from time

The intuitive method is "hold the throttle and scan for what changed". The session
supports it (`scan new f32`, then `scan changed` / `unchanged` / `increased` / ...,
narrowing as you alternate behaviours), and it is genuinely useful when the game is
otherwise still - a paused menu, a frozen scene, a value you can make jump on demand.

But on a LIVE game it does not separate the signal from the background. Measured on a
real title, over 60 frames:

```
scan new f32 ; step 60 (parked) ; scan unchanged ; step 60 (throttle held) ; scan changed
  -> 3478 candidates
the same protocol with NO throttle in the second phase
  -> 4139 candidates
```

The throttle "found" fewer things than doing nothing did. Thousands of slots move
every frame - timers, animation cursors, particles, allocator bumps, audio phase -
and "changed over time" cannot distinguish them from "changed because of me". No
amount of extra narrowing fixes this, because every pass is measuring the same
confounded thing. If you take one lesson from this file, take that one, and always
run the control.

### The sharp instrument: diff two runs

Two RUNS remove the background completely. The emulator is deterministic, so two runs
that replay the same prefix are byte-identical in memory at the branch frame. Let
them differ only in what is held from there, dump the same region from each at the
same frame, and every differing byte is CAUSED by that input:

```
explore --game /path/to/app --tape /tmp/scratch/run/playbook.tape \
        --at 1800 --hold 60 --after 30 \
        --dump all --dump-dir /tmp/scratch/dumps \
        --variant "throttle=l" --variant "brake=r"

memdiff /tmp/scratch/dumps/baseline.bin /tmp/scratch/dumps/throttle.bin
```

That determinism is not an assumption, it is checked: run a `control=` variant (the
same empty input as the baseline) and `memdiff` must report **0 differing bytes**. It
does. Any nonzero result there is a determinism bug in the emulator and invalidates
every other reading, so run it whenever you doubt a result.

An EMPTY diff against a real variant is a first-class answer, and often the important
one: the input reached no game state whatsoever, so no amount of retiming or holding
it longer will help - the problem is elsewhere.

Shrink `--hold`/`--after` to shrink the causal cone: two frames of input diffs to the
handful of words the button IMMEDIATELY touched (its state copy), which is where the
consumer is. Widen it to see how far the effect propagates.

`session` has `dump [addr len] <file>` if you want to drive the runs yourself.

## explore - many candidates at once

```
explore --game <dir> --at <frame> --variant "<label>=<input>" ... [options]
explore --game <dir> --at <frame> --sweep "cross" --over 1200-1260
```

Two ideas do the work:

1. **The prefix runs are independent, so run them at once.** Each variant is its own
   worker process - crash-isolated, no shared state. On a 16-core machine a
   24-candidate search costs about two prefixes of wall-clock instead of 24.
2. **The determinism signature is a free equivalence oracle.** Runs are grouped by
   the signature of their observable output, so identical outcomes collapse into one
   bucket and you eyeball one screenshot per DISTINCT outcome instead of one per
   candidate.

A no-input `baseline` variant always runs, and any bucket equal to it is marked
`NO EFFECT`. If EVERY variant lands in one bucket, that is a diagnosis, not a dead
end: the parameter you are varying is not the lever. Vary something else before
tuning it further.

`--sweep "<input>" --over LO-HI` is the frame-timing search (when to press X to land
a trick): one variant per candidate frame.

### Report more than a signature

`--report "<session command>"` runs that command at the end of every variant and puts
its output in the bucket report. Paired with `locate` this is the difference between
a search you have to look at and one you can read:

```
explore --game <dir> --tape <run.tape> --at 1810 --hold 90 \
        --report "locate --id 0x1e6e491a97a7b935" \
        --variant "b000=ry=0 lang=0" --variant "b090=ry=0 lang=90" ...
```

Every candidate now reports where the car actually ended up, in world coordinates,
from an identical starting state - a measured response curve for the control scheme
instead of eight screenshots and an opinion.

### A variant can be a manoeuvre, not just a held input

Semicolons split a variant's input into PHASES, `<frames>@<input>`, played in order
from the decision frame:

```
--variant "hook-left=40@ry=0 lang=0 ; 30@ry=0 lang=315 ; 60@ry=0 lang=0"
```

A phase with no `<frames>@` runs for `--hold`; an empty phase releases everything, so
`; 20@ ;` is a deliberate pause. This matters because the interesting question in a
game with a world in it is rarely "which button" - it is "which route", and a route
is a short sequence. Phases keep the parallelism spent on the expensive part: one
prefix replay per candidate MANOEUVRE rather than per candidate button.

## sheet - judge a whole run in one look

```
sheet --out /tmp/scratch/run.png /tmp/scratch/shots --cols 5 --limit 30
```

Reading fifty screenshots one at a time is the slowest thing an agent does, and the
thing you are looking for - where the picture changes, where it stops changing, where
it goes wrong - only shows up when consecutive frames sit side by side. Cadence shots
are named `f<frame>` precisely so sorting by name sorts by frame.

## When an input seems to be ignored, look before you guess

The tools install the engine's `tracing` subscriber, so `RUST_LOG` works and writes to stderr:

```
RUST_LOG=vitaslop::input=trace session --game ... --dir ...
```

That distinguishes two very different failures behind one symptom: the guest polled
the pad and chose to do nothing (your timing or button is wrong), versus the guest
never polled it at all (it is waiting on something else, and no input tuning helps).

## Measure what a stick axis MEANS before writing playbooks against it

A stick is often not two independent axes, and assuming it is produces a title that
looks broken - but the opposite assumption is just as expensive. One retail racer's
left stick was written up here as an ABSOLUTE, camera-relative heading (`lx`/`ly` as
one vector pointing where the car should FACE), and that was WRONG: `lx` is a TURN
RATE and `ly` is not read at all, proven by two runs whose determinism signatures were
identical with `ly` set and unset. A playbook built on the absolute-heading reading
steers into every correction.

Establish the axis empirically, and prefer a measurement whose signal is bigger than
its noise: hold one axis at full deflection for enough frames to move the vehicle a
long way, and compare against a run with the axis neutral. A calibration taken over a
few frames while the vehicle is still settling measures drift, not steering.

`lang=<bearing>` remains available for a title whose stick really is an absolute
heading; do not reach for it until that has been measured. It also makes the search
over headings a one-liner, which is the form navigation actually takes:

```
explore --game <dir> --at 2340 --tape ... \
        --variant "n=ry=0 lang=0"   --variant "ne=ry=0 lang=45" \
        --variant "e=ry=0 lang=90"  --variant "se=ry=0 lang=135"
```

If a title will not respond to input, suspect the MAPPING before the engine, get the
mapping from outside the game, and then check you are using the whole of it.

The input grammar is the rustdoc of `vitaslop-runtime/src/timeline.rs`; the session is
`vitaslop-playbook/src/session.rs`.

## Hold every press for several frames

The guest samples the pad once per DISPLAY FRAME. A press that spans fewer frames
than the gap between two samples is never observed at all, and that reads on screen
as a title that ignores input. `press --hold N` defaults to 8 frames for this reason,
and a playbook should hold a press about 12. If a press stops working, check whether
the guest polled at all before you retime it.

## Determinism and the signature

The deterministic engine (`VITASLOP_SMP=0`: one guest thread at a time, switched only at host
calls and frame flips) replays identically headless, on desktop and in the browser. The run's
FNV-1a signature over the observable output (the render stream + the egress ledger) is the
"did these two runs do the same thing" check: the session's `sig` command, or
`VITASLOP_SIGNATURE=1` / `VITASLOP_SIGNATURE_EVERY=<n>` on a headless or browser run, which print
it (the latter as `sigtrace` lines, so a divergence is located to its first frame). It covers
observable output only, so it is engine-independent.

## Lessons for agents playing games

These came out of playing real titles and are worth internalizing before you grind
a new title:

1. **Get a machine-readable outcome, not just screenshots.** The slowest part of the
   loop is a human eyeballing a PNG. The game already computes its verdict and
   usually renders it. Capture it as a scalar: a watch or memory cue, the egress ledger, the
   signature. An outcome you can compare in code is what lets the loop run without a
   human in it.
2. **Always run the control.** Any measurement of "what changed" on a live game needs
   the counterfactual - the same protocol with the input removed. Two of the three
   findings in this file only became findings once the control was run, and one of
   them (the in-session scan) turned out to be measuring nothing at all.
3. **"No effect" is a result.** If perturbing the variable does not move the
   signature - or a memory diff against the baseline is empty - you are tuning the
   wrong variable. Cheap to check, saves hours.
4. **Iteration cost is prefix replay.** Use a session for anything sequential and
   `explore` for anything you want to try several ways. Neither needs a snapshot.
5. **Live-state addresses are not stable across builds.** An address found by RE in
   one build can be a stale constant in the next (the live object relocates). Prefer
   re-discovering state behaviourally over trusting a fixed address.
