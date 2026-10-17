# vitaslop-playbook

> Keep this README terse: sectioned and bulleted, not large prose. Explain
> concepts and the why, not exact type names. Update it as the code changes so it
> never goes stale.

Lua scripts that play a title by reacting to it, and the tapes their runs record.

- A **playbook** waits on what the game DOES (a call it makes, a value in memory),
  not on frame numbers.
- A deterministic run records a **tape**: exact input on exact input frames, which
  replays byte for byte on the same build.
- A real-speed (SMP) run adapts to whatever timing and randomness it gets.
- Holds the engine, the Lua seam, the tape format, the tools, and the playbooks
  (`playbooks/<TITLE_ID>/*.lua`). Replaces the recipes in
  [`vitaslop-gamerun-recipes`](../vitaslop-gamerun-recipes/).
- **Playbooks (and recipes) are the only place a title is named.** The Rust stays
  game-agnostic: no title names or ids in source, comments or tests.

## Why

- Frame-numbered recipes need frame N to be the same every run: one-baton on the
  guest clock only, never real speed.
- They break across builds whenever a CPU-side change shifts guest timing.
- A playbook that waits on the game works in both modes and survives timing shifts.

## Running one

Desktop headless (product GPU renderer, frame capsules and all):

```
vitaslop --game <extracted dir> --headless <out> --playbook playbooks/<TITLE_ID>/<name>.lua
vitaslop --game <extracted dir> --headless <out> --tape <out>/playbook.tape
```

| output | what |
|---|---|
| `<out>/playbook.txt` | report: verdict, phases (frames, guest/wall seconds, speed), splits, warnings |
| `<out>/playbook.tape` | the tape this run recorded |
| `<out>/<name>.png` | each `shot(name)` |
| stdout | `[playbook] input <n> f<display> <secs>s ...` per phase, split, log, trace |

- Exit code is non-zero on failure (timeout, failed `check`, Lua error).

Browser, through the worker-mode harness (`vitaslop-web/e2e`); writes the same files
into `SHOT_DIR` and ends when the playbook does:

```
WORKER=1 HEADLESS=1 PLAYBOOK=<path>/<TITLE_ID>/<name>.lua GAME_DIR=<extracted dir> \
  MAX_FRAMES=<cap> TARGET_FRAME=<cap> KNOBS='{"VITASLOP_SMP":"1"}' node e2e/game-boot.mjs
```

### Modes

- `--headless` is one-baton on the guest clock: **deterministic**. Playbook and tape
  runs also force `VITASLOP_AUDIO_WALL=0` (wall-paced audio is a wall-clock input).
- `VITASLOP_SMP=1` is **real speed**; the browser page always is.
- The tape header records the mode. Byte comparisons are for deterministic runs;
  real-speed runs are judged on reaching each phase in time, and per-phase speed.
- **Every frame is rendered** during a playbook run, so shots are faithful (a target
  painted once and sampled for a thousand frames is only right if that frame was
  rendered). `VITASLOP_HEADLESS_SHOT_EVERY` adds a shot cadence on top.

## Finding cues: `--discover`

```
vitaslop --game <dir> --headless <out> --discover [--playbook <p.lua> | --recipe <r>]
```

- Appends to the report every host call the game made, ordered by the display frame
  it FIRST appeared on: a new screen is new calls.
- Also every distinct C-string argument (file paths, movie names) with its first frame.
- Read it beside a shot cadence and pick the call that marks the screen you need.
- Memory cues come from value search: `VITASLOP_MEM_DUMP=<addr>:<bytes>` with
  `VITASLOP_MEM_DUMP_AT=<f>,<f>,...` dumps guest memory at each frame; dump twice per screen
  (or across a value's change) and keep the cells constant within a screen and distinct
  across screens.
- **Prefer a cue in the image's own data.** A heap address can move between runs, and under
  SMP it has been seen holding garbage; check every memory cue at real speed.
- **A screen with no call and no cell is answered by repetition**: press its button until the
  NEXT screen's cue happens. Fixed waits are the last resort; say so in a comment when used.

## The tools

```
cargo run -p vitaslop-playbook --bin playbook -- check            # compile every playbook
cargo run -p vitaslop-playbook --bin playbook -- show <tape>      # a tape's timeline
cargo run -p vitaslop-playbook --bin playbook -- diff <a> <b>     # how far each note moved
```

- `diff` is the cross-build regression read: two deterministic tapes of one playbook,
  every phase and split that moved, by input frames and guest seconds.

## Writing one

```lua
-- Every API call below is a global. Comment the WHY of every phase and input.
local pad   = cue.call("sceCtrlPeekBufferPositive")
local movie = cue.call("sceAvPlayerClose")            -- the intro movie ended
local title = cue.call("sceNpInit"):happened()        -- the title screen took START (a condition)
local card  = cue.u32(0x81000010)                     -- found by value search

phase("boot", { timeout = 120 }, function()
  wait(movie)
  wait(pad:after(movie))                              -- the title screen reads the pad
  repeat tap("start") until try_wait(title, 1)
end)

phase("menu", { timeout = 60 }, function()
  while card:value() ~= 2 do tap("right") end         -- step the card until it reads 2
  shot("menu")
end)
```

### Units

- **Input frame**: the first pad read after each display flip. Loads and movies that
  do not read the pad do not count, so `hold("cross", 20)` is 20 frames of game logic
  at any speed. Holds, idles and `stable` count these.
- **Guest seconds**: timeouts and `cue.seconds`.

### Cues

| declare | value | waiting on it fires |
|---|---|---|
| `cue.call(name)` | calls made so far | on the next call after the wait began |
| `cue.call(name, {arg=0, has="stage"})` | matching calls | ...whose C string in `r<arg>` contains `has` |
| `cue.u8/u16/u32/i8/i16/i32/f32(addr)` | the value | when it differs from its value at the wait |
| `cue.u32(ptr(base, off, ...))` | the value | (pointer chain: read a pointer at `base`, add `off`, repeat) |
| `c:is(v)` `:isnt` `:above` `:below` `:atleast` `:atmost` | 1/0 | while it holds (at once if it already does) |
| `c:happened()` | 1/0 | once call cue `c` has fired at all since `c` was declared (`:atleast(1)`) |
| `c:changed()` | 1/0 | on a frame the value changed |
| `c:stable(n)` | 1/0 | once the value has held `n` input frames |
| `a:after(b)` | - | on `a`'s first firing after `b` fired |
| `cue.any(a, b, ...)` | - | on the first to fire; returns its 1-based index |
| `cue.all(a, b, ...)` | - | once each has fired (each latches) |
| `cue.frames(n)` / `cue.seconds(s)` | - | `n` input frames / `s` guest seconds after the wait began |

- Call cues count from every guest thread, lock-free.
- Memory and conditions are sampled once per input frame.
- A cue name that never matched a call is a `WARN` in the report.
- **A loop that presses while it waits must wait on a CONDITION.** `wait(c)` on a call
  cue fires on the next call after the wait begins, so in
  `repeat tap("start") until try_wait(c, 1)` a call made during the tap is missed for
  good. `c:happened()` cannot miss it.
- **A call cue counts from ITS OWN declaration.** `c:happened()` on a cue declared at the
  top is already true if `c` was called during boot; to mean "since this phase began",
  declare a fresh `cue.call(...)` at the start of the phase.
- **Declare cues ONCE, outside loops.** Every `:happened()`, `:is()` etc. declares a new
  cue; a run that declares thousands fails, naming this.

### Statements

| | |
|---|---|
| phases | `phase(name, {timeout = s}, fn)` - nestable; the timeout bounds the phase and every wait in it |
| waits | `wait(c, {timeout = s, soft = true})` returns the cue's value; `try_wait(c, s)` returns nil on timeout |
| input | `press(b...)` `release(b...)` (none = all) `hold(b, n)` `tap(b, down=6, up=6)` `idle(n)` `stick("left", x, y)` `touch(x, y)` `untouch()` |
| observe | `shot(name)` `log(...)` (= `print`) `split(name)` `c:trace(label)` `c:value()` `now()` |
| assert | `check(cond_cue, msg)` `fail(msg)` |

- Buttons: `cross` `circle` `square` `triangle` `up` `down` `left` `right` `start`
  `select` `l`/`ltrigger` `r`/`rtrigger`; combine with `+` (`"start+cross"`).
- Sticks are raw 0..255, 128 centred. Touch is front-panel coordinates (screen x2).
- **Every wait has a timeout**, its own or its enclosing phases'. A wait with neither is
  an error at its line. A timeout fails the run, naming the cue and the phase path.
- Control flow is plain Lua. A loop with no wait, idle or hold fails the run instead of
  hanging it.
- Libraries: `string`, `table`, `math` (seeded with 0, runs must replay), `utf8`,
  `coroutine`. No `io`, `os` or `package`: a playbook reads the game, never the machine.

## Tapes

```
vitaslop-tape 1
playbook ABCD00001/example.lua
title ABCD00001
mode deterministic (one-baton)
input-frame first pad read after each display flip
# <input frame> <display frame> <buttons> <lx> <ly> <rx> <ry> <touch>  - a line per CHANGE
0 f312 - 128 128 128 128 -
# 41 f353 0.884s split title
41 f353 start 128 128 128 128 -
47 f359 - 128 128 128 128 -
end 2210 f4120
```

- Header: `key value` lines. Unknown keys are kept; replay ignores them all.
- Body line = a CHANGE: the input from that input frame until the next line.
  `<buttons>` is `-` or names joined by `+`; `<touch>` is `-` or `x,y`.
- `# <input> f<display> <secs>s <text>` lines are the timeline (phases, splits, logs,
  traces, verdict), read by `show` and `diff`, not replayed.
- `end <input> f<display>` is the last input frame recorded.
- Replay presses each change on its input frame and takes each recorded `shot` on its
  input frame, so a replay on a new build writes the same named shots to byte-compare.
- In deterministic mode a change landing on a different DISPLAY frame than recorded is a
  `WARN tape desync`: the run diverged.
- Measured: a 9,700-input-frame tape replayed on its own build gave 17/17 named shots
  byte-identical, no desync.

## How it works

- **The tick.** The runtime calls one observer on every host call
  ([`vitaslop_runtime::observer`](../vitaslop-runtime/src/observer.rs)).
  - Call cues are counted there.
  - On the first pad read of each display frame: sample memory cues, advance the
    blocking request, resume the playbook if it finished, latch the input every pad
    read of that frame returns.
  - The host also reports each display frame, which enforces timeouts on a game that
    has stopped reading the pad.
- **The VM never calls the host.** The prelude (`src/prelude.lua`) turns every API call
  into a JSON request yielded from the playbook's coroutine; the engine replies in JSON.
  - A Lua backend is two functions: run a chunk, call a global with a string
    ([`src/vm.rs`](src/vm.rs)). That keeps the implementation swappable.
  - Native: `mlua` with PUC-Rio Lua 5.4 from source. Browser: wasmoon, the same C Lua
    5.4 compiled to wasm.
- **Where the playbook resumes.**
  - Native: inside the tick. A request finishing on input frame N has its successor's
    input on frame N, in every mode.
  - Browser: the VM lives in JS on the run worker, which a guest worker's tick cannot
    call, so it resumes once per present and a successor lands on the next input frame.
    The browser is real-speed only, so nothing exact is lost.

