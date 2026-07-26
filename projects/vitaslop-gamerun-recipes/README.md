# vitaslop game-run recipes - how an agent plays a Vita game

This crate is where playthroughs live, and where the tools an agent plays with live.
A **recipe** is a plain-text, frame-keyed script that drives a real title through the
emulator: it presses buttons, taps the touch panel, watches memory, asserts that the
expected things happen, and grabs screenshots for review. Recipes are how an agent
(or a human) plays and beats a game reproducibly, and how the conformance suite
proves a title still runs.

Recipes are the ONLY place a specific game is named in this project. The engine
crates stay game-agnostic - every title-specific fact (where its live state lives in
memory, what "level complete" looks like, when to screenshot) is data in a recipe,
interpreted by a generic runner.

## The tools

| tool | what it is for |
|---|---|
| `session` + `ctl` | a RESIDENT game session you drive forward command by command. The main tool. Boot once, then explore for as long as you like. |
| `play` | run a whole recipe start to finish and print a structured report. The batch/CI form. |
| `explore` | try many candidate inputs at one frame, in parallel, and report which of them did something different. |
| `memdiff` | diff two runs' guest memory and see exactly what an input touched. How you find the address of a live value. |
| `sheet` | montage a run's screenshots into one labelled grid. |

```
cargo build --release -p vitaslop-gamerun-recipes    # builds all five
```

Every one of them takes `--list-knobs` (well, `play` and `session` do) to print the
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
        --recipe recipes/PCSA00027/gameplay-prefix.recipe \
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
- `input <tokens>`, `press <tokens> --hold N` - the same input grammar a recipe uses.
- `read`, `peek`, `poke`, `watch`, `dump` - see and touch guest memory.
- `scan ...` - the in-session value finder (read the caveat below).
- `shot`, `sig`, `egress`, `section`, `note` - observe and annotate.
- `save <file.recipe>` - write the whole session out as a replayable recipe.

### A session IS a recipe

Input goes through the same frame-keyed timeline a recipe replays, and every command
that changes the input appends a segment to it. The played run is literally a recipe
under construction: `save` writes it out, and replaying that file reproduces the
session. Exploration and the committed artifact are the same object, which is what
stops a successful playthrough from being a thing that happened once in someone's
terminal.

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
explore --game /path/to/app --recipe recipes/PCSA00027/gameplay-prefix.recipe \
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
explore --game <dir> --recipe <prefix> --at 1810 --hold 90 \
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
are named `<section>-f<frame>` precisely so sorting by name sorts by frame.

## play - the batch form

```
cargo run --release -p vitaslop-gamerun-recipes --bin play -- \
  --game  /path/to/app/PCSE00341 \
  --recipe recipes/PCSE00341/pushing.recipe \
  --shots /tmp/scratch/olli        # screenshots + watch.csv (NEVER the repo)
```

```
RUN FramesReached(480) frames=480 sig=0x1f3a...c04
ASSERT f460  PASS    vpos ~ 60 +-6 -> actual vpos=60
SHOT   f470  pushing-rolling -> /tmp/scratch/olli/pushing-rolling.png
RESULT 1/1 asserts passed
```

A failing assertion prints actual-vs-expected, which is the feedback you act on:

```
ASSERT f480  FAIL    vpos ~ 60 +-3 -> actual vpos=-12
```

`play` exits 0 when every assertion passes, 1 otherwise, so you can gate a script on
it. `--observe-from <frame>` fast-forwards the prefix when you only care about a late
section.

**When an input seems to be ignored, look before you guess.** `play` and `session`
install the engine's `tracing` subscriber, so `RUST_LOG` works and writes to stderr
(the report stays on stdout):

```
RUST_LOG=vitaslop::input=trace play --game ... --recipe ...
```

That distinguishes two very different failures behind one symptom: the guest polled
the pad and chose to do nothing (your timing or button is wrong), versus the guest
never polled it at all (it is waiting on something else, and no input tuning helps).

## The recipe format

Header directives (no frame):

- `@title <text>` - a human label.
- `@game <TITLE_ID>` - the title this recipe targets (the recipes subdir name).
- `@watch <name> <type> <addr>` - a live memory value to sample every frame; `type`
  is `u8|u16|u32|i32|f32`, `addr` is hex. Name it so assertions refer to it.
- `@sig <hex>` - the expected determinism signature; replay fails if it differs.
- `@shot-every <N>` - auto-screenshot every N observed frames (on top of explicit
  `@shot`s), named `<section>-f<frame>`. Use a modest cadence (30 to 60 frames)
  through gameplay sections.

Timeline lines `<frame>: ...`:

- Input: buttons (`cross`/`x`, `circle`, `square`, `triangle`, `up`/`down`/`left`/
  `right`, `start`, `select`, `l`, `r`), analog (`lx=`/`ly=`/`rx=`/`ry=`, 0..255, 128
  neutral), and `touch=X,Y` (front panel, = screen*2). Input is STICKY: it holds
  until the next line changes it. A tap is `touch=` on one line then a later line
  that drops it.
- A stick in POLAR form: `lang=<deg>[,<mag>]` / `rang=` set both of that stick's axes
  from a compass bearing - `0` up-screen, increasing clockwise, optional magnitude
  `0..127` (default full). `lang=0` is exactly `lx=128 ly=1`. See below for why this
  is not sugar.

### Aim a stick with `lang=`, not with `lx=`

A stick is often not two independent axes, and assuming it is produces a title that
looks broken. PCSA00027's left stick is an ABSOLUTE, camera-relative heading: it
points where the car should FACE, so `lx` and `ly` are one vector and must be set
together. Setting only `lx` - which every early recipe here did - can aim the car
exactly screen-left or exactly screen-right and at NO diagonal, so "steer a little
toward the trail" is not expressible at all and the car drives into a wall instead.
That cost two sessions across two different depths of investigation.

`lang=<bearing>` says the thing the control legend says. It also makes the search
over headings a one-liner, which is the form navigation actually takes:

```
explore --game <dir> --at 2340 --recipe ... \
        --variant "n=ry=0 lang=0"   --variant "ne=ry=0 lang=45" \
        --variant "e=ry=0 lang=90"  --variant "se=ry=0 lang=135"
```

If a title will not respond to input, suspect the MAPPING before the engine, get the
mapping from outside the game, and then check you are using the whole of it.
- `<frame>: @section <name>` - name a region (groups shots/asserts, aids bisection).
- `<frame>: @assert <watch> <op> <value> [+-<tol>]` - `op` is `== != < <= > >= ~`.
- `<frame>: @assert egress <Kind> [field<op>value ...]` - assert the OS-egress ledger
  recorded a `SaveWrite` / `Trophy` / `ScoreSubmit` at or before this frame. This is
  the content-free "the game did the thing" surface.
- `<frame>: @shot <name>` - render this frame to `<name>.png`.
- `<frame>: @note <text>` / `<frame>: @todo <text>` - a durable note or open task.

The full grammar is the rustdoc of `vitaslop-runtime/src/recipe.rs`; the runner is
`vitaslop-native/src/recipe_runner.rs`; the session is
`vitaslop-native/src/session.rs`.

### Hold every press for several frames

The guest samples the pad once per DISPLAY FRAME. A press that spans fewer frames
than the gap between two samples is never observed at all, and that reads on screen
as a title that ignores input. `press --hold N` defaults to 8 frames for this reason,
and a recipe should hold a press about 12. If a press stops working, check whether
the guest polled at all before you retime it.

### Comment heavily - this is strongly encouraged

Recipes are read far more than they are written, and the next agent to touch one has
none of the context you have now. Treat the recipe as a lab notebook:

- Explain WHY an input is timed where it is.
- Record what you tried that did NOT work, so nobody re-walks a dead end.
- Note every RE fact you relied on (an address, a panel coord, a state value) and how
  you found it.
- Use `@note` for durable findings and `@todo` for open work. `play` and `session`
  echo them at startup, so they are the handoff.

A recipe with dense comments is one the next agent can extend. A bare list of button
presses is one nobody can safely touch.

## Determinism and @sig

The engine runs the guest deterministically (single-baton cooperative scheduling: one
guest thread at a time, switched only at host calls and frame flips), so a recipe
replays identically headless, on desktop, and in the browser. `@sig` pins the FNV-1a
signature over the observable output (the render stream + the egress ledger):

- Run the recipe, read `sig=0x...` from the report, and add `@sig 0x...`.
- Every later run recomputes it; a mismatch fails loudly, catching nondeterminism (or
  a real behaviour change) before it silently invalidates the recipe.

The signature covers observable output only (not internal RAM or thread timing), so
it is engine-independent and doubles as the cross-engine "did native and the browser
run identically" check. `session save` pins it automatically.

## Lessons for agents playing games

These came out of authoring real recipes and are worth internalizing before you grind
a new title:

1. **Get a machine-readable outcome, not just screenshots.** The slowest part of the
   loop is a human eyeballing a PNG. The game already computes its verdict and
   usually renders it. Capture it as a scalar: a `@watch`, the egress ledger, the
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

## Layout

```
games.toml                      registry: title id -> friendly name
recipes/<TITLE_ID>/*.recipe     the recipes for one title (subdir = title id)
src/lib.rs                      the registry loader (engine-free)
src/bin/play.rs                 batch: run a recipe, print a report
src/bin/session.rs              resident: boot once, drive forward
src/bin/ctl.rs                  the client that talks to a session
src/bin/explore.rs              parallel variant search
src/bin/memdiff.rs              what did this input touch?
src/bin/sheet.rs                contact sheet
tests/conformance.rs            opt-in: replay every recipe against a private dump
```

## Conformance

`tests/conformance.rs` replays every committed recipe against a privately-supplied
dump and asserts each recipe's own `@assert`s pass. It is `#[ignore]`d and skips
without `VITASLOP_GAME_DIR`, so the workspace stays green for everyone:

```
VITASLOP_GAME_DIR=/path/to/app/PCSE00341 \
  cargo test --release -p vitaslop-gamerun-recipes --test conformance -- --ignored --nocapture
```

Recipes are content-free and committed; running them needs the private game bytes,
which are never committed.
