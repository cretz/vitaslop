# vitaslop game-run recipes

This crate is where playthroughs live. A **recipe** is a plain-text, frame-keyed
script that drives a real title through the emulator: it presses buttons, taps the
touch panel, watches memory, asserts that the expected things happen, and grabs
screenshots for human review. Recipes are how an agent (or a human) plays and beats a
game reproducibly, and how the conformance suite proves a title still runs.

Recipes are the ONLY place a specific game is named in this project. The engine crates
stay game-agnostic - every title-specific fact (where its live state lives in memory,
what "level complete" looks like, when to screenshot) is data in a recipe, interpreted
by a generic runner.

## Layout

```
games.toml                      registry: title id -> friendly name
recipes/<TITLE_ID>/*.recipe     the recipes for one title (subdir = title id)
src/lib.rs                      the registry loader (engine-free)
src/bin/play.rs                 the `play` binary: run a recipe, print a report
tests/conformance.rs            opt-in: replay every recipe against a private dump
```

The recipe FORMAT (the full grammar) is documented in the rustdoc of
`vitaslop-runtime/src/recipe.rs`. The RUNNER lives in
`vitaslop-native/src/recipe_runner.rs`. This README teaches how to author recipes; the
rustdoc is the authoritative spec.

## The loop: author, run, read, adjust

The whole point is a fast feedback loop. You edit a recipe, run it, read a structured
report, and adjust - the report tells you exactly how an assertion missed, so you are
not guessing.

```
cargo run --release -p vitaslop-gamerun-recipes --bin play -- \
  --game  /path/to/app/PCSE00341 \
  --recipe recipes/PCSE00341/pushing.recipe \
  --shots /tmp/scratch/olli        # where screenshots + watch.csv go (NEVER the repo)
```

The report looks like:

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

`vpos=-12` at the frame you expected a landing means the skater is still airborne -
you pressed X too early. Nudge the X-press a few frames later and run again. That is
the loop: the numbers tell you which way to move.

`play` exits 0 when every assertion passes, 1 otherwise, so you can gate a script on
it.

## The directive vocabulary (quick reference)

Header directives (no frame):

- `@title <text>` - a human label.
- `@game <TITLE_ID>` - the title this recipe targets (the recipes subdir name).
- `@watch <name> <type> <addr>` - a live memory value to sample every frame; `type`
  is `u8|u16|u32|i32|f32`, `addr` is hex. Name it so assertions refer to it.
- `@sig <hex>` - the expected determinism signature; replay fails if it differs (see
  "Determinism" below).
- `@shot-every <N>` - auto-screenshot every N observed frames (on top of explicit
  `@shot`s), named `<section>-f<frame>`. In-game shots are how a human judges whether
  the render and gameplay actually look right, so use a modest cadence (e.g. every 30
  to 60 frames) through gameplay sections. `--shot-every N` on `play` overrides it.

Timeline lines `<frame>: ...`:

- Input: buttons (`cross`/`x`, `circle`, `square`, `triangle`, `up`/`down`/`left`/
  `right`, `start`, `select`, `l`, `r`), analog (`lx=`/`ly=`/`rx=`/`ry=`, 0..255, 128
  neutral), and `touch=X,Y` (front panel, = screen*2). Input is STICKY: it holds until
  the next line changes it. A tap is `touch=` on one line then a later line drops it.
- `<frame>: @section <name>` - name a region (groups shots/asserts, aids bisection).
- `<frame>: @assert <watch> <op> <value> [+-<tol>]` - assert a watched value; `op` is
  `== != < <= > >= ~` (`~` is approximate, pair with `+-`).
- `<frame>: @assert egress <Kind> [field<op>value ...]` - assert the OS-egress ledger
  recorded a `SaveWrite` / `Trophy` / `ScoreSubmit` at or before this frame, optionally
  matching fields (`path=...`, `ascii~substr`, `bytes>=N`, `id=N`, `board=N`,
  `score>=N`). This is the content-free "the game did the thing" surface.
- `<frame>: @shot <name>` - render this frame to `<name>.png` (needs `--shots`).
- `<frame>: @note <text>` / `<frame>: @todo <text>` - a durable note or open task.

## COMMENT HEAVILY - this is strongly encouraged

Recipes are read far more than they are written, and the next agent to touch one has
none of the context you have now. Treat the recipe as a lab notebook:

- Explain WHY an input is timed where it is ("X here lands the cone gap; two frames
  earlier bails").
- Record what you tried that did NOT work, so nobody re-walks a dead end ("holding X
  through the gap SLOPPY-lands and kills the combo; must tap on touchdown").
- Note every RE fact you relied on (an address, a panel coord, a state value) and how
  you found it.
- Use `@note` for durable findings and `@todo` for open work; use `#` comments freely
  everywhere else. `play` echoes `@note`/`@todo` at startup, so they are the handoff.

A recipe with dense comments is a recipe the next agent can extend. A bare list of
button presses is a recipe nobody can safely touch.

## Authoring tricks (how to find what a recipe needs)

- **Find live state by value-search.** To assert on a value (skater height, score, a
  level-complete flag), you first need its address. Step the run frame by frame with a
  memory region dumped each frame (the boot probe's `VITASLOP_DUMP_REGION` /
  `VITASLOP_WATCH_MEM`), do a known thing (jump, land, finish), and diff the dumps to
  find the address whose value tracks it. Then record it as `@watch`. The search is
  external, game-agnostic tooling; the recipe keeps only the result.
- **Time landings off a vertical-position watch.** For a title with airtime, watch the
  height value: it dips while airborne and returns to its grounded value at touchdown.
  Press the land button on the frame it returns. `@assert height ~ <ground> +-<tol>` at
  your intended land frame tells you if you hit it.
- **Menu taps are panel coords = screen*2.** Read a screen coordinate off a `@shot`,
  double it, and `touch=` there. A tap is a touch on one line then a drop on the next.
- **Fast-forward the boring prefix.** `--observe-from <frame>` runs everything before a
  frame in one batch (no per-frame sampling), so iterating on a late level does not
  re-step the whole tutorial each time. When you declare `@watch`es, observation starts
  at frame 0 by default so the whole watch log is captured.
- **Screenshot the outcome, not every frame.** Put a `@shot` at the moment that
  decides pass/fail (the land frame, the results screen) and eyeball just that PNG.

## Determinism and @sig

The engine runs the guest deterministically (single-baton cooperative scheduling: one
guest thread at a time, switched only at host calls and frame flips), so a recipe
replays identically headless, on desktop, and in the browser. `@sig` pins the FNV-1a
signature the runner computes over the observable output (the render stream + the
egress ledger). Add it once a recipe is stable:

- Run the recipe, read `sig=0x...` from the report, and add `@sig 0x...`.
- Every later run recomputes it; a mismatch fails loudly, catching nondeterminism (or
  a real behavior change) before it silently invalidates the recipe.

The signature covers observable output only (not internal RAM or thread timing), so it
is engine-independent and also serves as the cross-engine "did native and the browser
run identically" check.

## Conformance

`tests/conformance.rs` replays every committed recipe against a privately-supplied
dump and asserts each recipe's own `@assert`s pass. It is `#[ignore]`d and skips
without `VITASLOP_GAME_DIR`, so the workspace stays green for everyone; it only runs
for whoever holds the dump:

```
VITASLOP_GAME_DIR=/path/to/app/PCSE00341 \
  cargo test --release -p vitaslop-gamerun-recipes --test conformance -- --ignored --nocapture
```

Recipes are content-free and committed; running them needs the private game bytes,
which are never committed.
