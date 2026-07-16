//! A deterministic, self-contained "game-run recipe": frame-keyed controller input
//! plus the metadata a runner needs to observe and judge the run - live memory
//! watches, assertions, screenshot points, named sections, and free-form notes. A
//! headless run replays a recipe reproducibly to navigate menus, play a level, and
//! assert that the expected things happened, all without embedding any game content.
//!
//! # Why this format exists
//! The recipe is the seam that lets an agent (or a human) drive a title with no
//! game-specific code in the engine: everything a run needs to know about a
//! particular title - where its live state lives in memory, what a "level complete"
//! looks like, when to grab a screenshot - is data in the recipe, interpreted by a
//! generic runner. The engine stays a black box that presses buttons and reports
//! what it observed.
//!
//! # Line format
//! The recipe is line based. A line is one of:
//!
//! - A blank line or a `#` comment (comments may also trail any line).
//! - A HEADER directive `@<keyword> <args>` (no frame): `@title`, `@game`, `@sig`,
//!   `@watch`.
//! - A TIMELINE line `<frame>: <rest>`, keyed to the display-flip index at which it
//!   takes effect. `<rest>` is EITHER input directives OR a single frame-keyed meta
//!   directive `@<keyword> <args>` (`@assert`, `@shot`, `@section`, `@note`,
//!   `@todo`). Keeping meta directives on their own timeline line keeps parsing
//!   unambiguous and the file readable.
//!
//! ## Input directives (sticky between timeline lines)
//! Buttons to hold this segment - `cross`/`x`, `circle`, `square`, `triangle`,
//! `up`, `down`, `left`, `right`, `start`, `select`, `l`, `r` - and analog overrides
//! `lx=`/`ly=`/`rx=`/`ry=` in `0..255` (128 neutral). A `touch=X,Y` directive holds
//! one finger on the front panel at panel coordinates `(X, Y)` (the front panel is
//! 1920x1088, so panel = screen * 2). Input state is STICKY: it holds until the next
//! timeline line changes it, so a tap is `touch=` on one line then a later line that
//! drops it, and a line with no input directives releases everything.
//!
//! ## Header directives
//! - `@title <text>` - a human label for the run.
//! - `@game <id>` - the title id this recipe targets (e.g. a subdir name); the
//!   engine never reads it, it is provenance for the registry.
//! - `@sig <hex>` - the expected determinism signature (see the runner). Replay
//!   recomputes it and fails on a mismatch, so a recipe is only valid while it stays
//!   reproducible.
//! - `@watch <name> <type> <addr>` - declare a live memory value to sample every
//!   frame, `type` in `u8|u16|u32|i32|f32`, `addr` hex. Named so assertions and the
//!   watch log refer to it by name. Discovering the address is external RE (value
//!   search); the recipe records the result.
//! - `@shot-every <N>` - auto-screenshot every N observed frames (on top of any
//!   explicit `@shot`), so a human can assess in-game quality across the whole run,
//!   not only at the decisive frames.
//!
//! ## Frame-keyed meta directives
//! - `<frame>: @section <name>` - start a named region (ends at the next section or
//!   the run end); groups shots/asserts for human review and for bisecting a
//!   divergence.
//! - `<frame>: @assert <name> <op> <value> [+-<tol>]` - assert a watched value at
//!   this frame, `op` in `== != < > <= >= ~` (`~` is approximate, pair with a `+-`
//!   tolerance).
//! - `<frame>: @assert egress <Kind> [field<op>value ...]` - assert the OS-egress
//!   ledger recorded an event of `Kind` (`SaveWrite`/`Trophy`/`ScoreSubmit`) at or
//!   before this frame, optionally matching fields (`path=...`, `ascii~substr`,
//!   `bytes>=N`, `id=N`, `board=N`, `score>=N`). This is the content-free "the game
//!   did the thing" surface.
//! - `<frame>: @shot <name>` - render the current frame to a screenshot named
//!   `<name>` for human review.
//! - `<frame>: @note <text>` / `<frame>: @todo <text>` - a durable note or an open
//!   task for whoever (agent or human) picks up the recipe next. Heavy commenting is
//!   strongly encouraged: record why an input is timed as it is and what was learned.
//!
//! ```text
//! @title Tutorial - first lesson
//! @game  ABCD00001
//! @watch vpos i32 0x81502058   # skater vertical: 60 ground, negative airborne
//!
//! 0: @section navigate
//! 12: touch=450,674            # dismiss a startup dialog
//! 19:
//! 150: cross                   # rhythmic input drives the character forward
//! 154:
//! 480: @assert vpos ~ 60 +-3   # touchdown reached
//! 480: @shot pushing-land
//! 520: @assert egress SaveWrite path=savedata0:data.bin
//! ```
//!
//! Input is keyed to the frame count the scheduler reports through
//! [`World::set_frame`], not to wall time or poll order, so the same recipe yields
//! the same run regardless of how often the guest polls the pad.

use crate::world::{CtrlFrame, TouchFrame, World};

/// The Vita `SceCtrlButtons` bit for each directive keyword (from the MIT
/// vita-headers `psp2common/ctrl.h`).
fn button_bit(name: &str) -> Option<u32> {
    Some(match name {
        "select" => 0x0000_0001,
        "l3" => 0x0000_0002,
        "r3" => 0x0000_0004,
        "start" => 0x0000_0008,
        "up" => 0x0000_0010,
        "right" => 0x0000_0020,
        "down" => 0x0000_0040,
        "left" => 0x0000_0080,
        "ltrigger" | "l" => 0x0000_0100,
        "rtrigger" | "r" => 0x0000_0200,
        "l1" => 0x0000_0400,
        "r1" => 0x0000_0800,
        "triangle" => 0x0000_1000,
        "circle" => 0x0000_2000,
        "cross" | "x" => 0x0000_4000,
        "square" => 0x0000_8000,
        _ => return None,
    })
}

/// A watched value's storage type. Sampled from guest memory each frame and widened
/// to `f64` for comparison so integer and float watches assert uniformly.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub enum ValType {
    U8,
    U16,
    U32,
    I32,
    F32,
}

impl ValType {
    /// Parse a type keyword. `None` if unrecognized.
    pub fn parse(s: &str) -> Option<ValType> {
        Some(match s {
            "u8" => ValType::U8,
            "u16" => ValType::U16,
            "u32" => ValType::U32,
            "i32" => ValType::I32,
            "f32" => ValType::F32,
            _ => return None,
        })
    }

    /// The width in bytes to read from guest memory.
    pub fn width(self) -> usize {
        match self {
            ValType::U8 => 1,
            ValType::U16 => 2,
            _ => 4,
        }
    }

    /// Decode little-endian `bytes` (at least [`width`](Self::width) long) to `f64`.
    pub fn decode(self, bytes: &[u8]) -> Option<f64> {
        Some(match self {
            ValType::U8 => bytes.first().copied()? as f64,
            ValType::U16 => u16::from_le_bytes([*bytes.first()?, *bytes.get(1)?]) as f64,
            ValType::U32 => {
                u32::from_le_bytes([bytes[0], bytes[1], bytes[2], bytes[3]]) as f64
            }
            ValType::I32 => {
                i32::from_le_bytes([bytes[0], bytes[1], bytes[2], bytes[3]]) as f64
            }
            ValType::F32 => {
                f32::from_le_bytes([bytes[0], bytes[1], bytes[2], bytes[3]]) as f64
            }
        })
    }
}

/// A comparison operator for a memory assertion.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub enum CmpOp {
    Eq,
    Ne,
    Lt,
    Le,
    Gt,
    Ge,
    /// Approximately equal, within a tolerance (paired with `+-<tol>`).
    Approx,
}

impl CmpOp {
    fn parse(s: &str) -> Option<CmpOp> {
        Some(match s {
            "==" | "=" => CmpOp::Eq,
            "!=" => CmpOp::Ne,
            "<" => CmpOp::Lt,
            "<=" => CmpOp::Le,
            ">" => CmpOp::Gt,
            ">=" => CmpOp::Ge,
            "~" => CmpOp::Approx,
            _ => return None,
        })
    }

    /// Evaluate `actual <op> value` (with `tol` for [`Approx`](Self::Approx)).
    pub fn eval(self, actual: f64, value: f64, tol: f64) -> bool {
        match self {
            CmpOp::Eq => actual == value,
            CmpOp::Ne => actual != value,
            CmpOp::Lt => actual < value,
            CmpOp::Le => actual <= value,
            CmpOp::Gt => actual > value,
            CmpOp::Ge => actual >= value,
            CmpOp::Approx => (actual - value).abs() <= tol,
        }
    }
}

/// One parsed timeline entry: the input state that takes effect at `frame` and holds
/// until the next entry.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
struct Segment {
    frame: u64,
    input: CtrlFrame,
    touch: TouchFrame,
}

/// A `@watch <name> <type> <addr>` declaration: a live memory value to sample.
#[derive(Clone, Debug, PartialEq, Eq)]
pub struct WatchDecl {
    pub name: String,
    pub ty: ValType,
    pub addr: u32,
}

/// A memory assertion: `@assert <watch> <op> <value> [+-<tol>]`.
#[derive(Clone, Debug, PartialEq)]
pub struct MemAssert {
    pub watch: String,
    pub op: CmpOp,
    pub value: f64,
    pub tol: f64,
}

/// One field matcher inside an egress assertion, e.g. `bytes>=64` or `ascii~URBAN`.
#[derive(Clone, Debug, PartialEq)]
pub struct FieldMatch {
    pub field: String,
    pub op: FieldOp,
    /// The raw right-hand side; the runner interprets it as a number or a substring
    /// per `op`.
    pub value: String,
}

/// The operator in a [`FieldMatch`].
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub enum FieldOp {
    Eq,
    Ge,
    Le,
    Gt,
    Lt,
    /// Substring containment (for the ascii preview).
    Contains,
}

/// An egress assertion: `@assert egress <Kind> [field<op>value ...]`.
#[derive(Clone, Debug, PartialEq)]
pub struct EgressAssert {
    /// `SaveWrite` | `Trophy` | `ScoreSubmit`.
    pub kind: String,
    pub fields: Vec<FieldMatch>,
}

/// What an `@assert` line checks.
#[derive(Clone, Debug, PartialEq)]
pub enum AssertKind {
    Mem(MemAssert),
    Egress(EgressAssert),
}

/// A frame-keyed assertion.
#[derive(Clone, Debug, PartialEq)]
pub struct AssertDecl {
    pub frame: u64,
    pub kind: AssertKind,
}

/// A frame-keyed screenshot request.
#[derive(Clone, Debug, PartialEq, Eq)]
pub struct ShotDecl {
    pub frame: u64,
    pub name: String,
}

/// A named region of the run, starting at `frame` (ending at the next section or the
/// run end).
#[derive(Clone, Debug, PartialEq, Eq)]
pub struct Section {
    pub frame: u64,
    pub name: String,
}

/// A note or open task attached to a frame, for handoff between whoever authors and
/// whoever picks up the recipe.
#[derive(Clone, Debug, PartialEq, Eq)]
pub struct NoteDecl {
    pub frame: u64,
    pub todo: bool,
    pub text: String,
}

/// Non-timeline provenance from the header directives.
#[derive(Clone, Debug, Default, PartialEq, Eq)]
pub struct RecipeMeta {
    pub title: Option<String>,
    pub game: Option<String>,
    /// The expected determinism signature (`@sig <hex>`), if pinned.
    pub sig: Option<u64>,
    /// `@shot-every <N>`: auto-screenshot every N observed frames (in addition to any
    /// explicit `@shot` points), so a human can assess in-game quality across a run,
    /// not just at the decisive frames. `None` = only explicit shots.
    pub shot_every: Option<u64>,
}

/// A parse failure with the 1-based line number and a reason.
#[derive(Debug, PartialEq, Eq)]
pub struct RecipeError {
    pub line: usize,
    pub reason: String,
}

impl std::fmt::Display for RecipeError {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        write!(f, "recipe line {}: {}", self.line, self.reason)
    }
}

/// A fully parsed recipe: the input timeline plus all the metadata a generic runner
/// needs to observe and judge the run. Free of game-specific code - the engine reads
/// none of this except the input timeline (via [`RecipeWorld`]); the runner reads the
/// rest.
#[derive(Clone, Debug, Default, PartialEq)]
pub struct Recipe {
    pub meta: RecipeMeta,
    pub watches: Vec<WatchDecl>,
    pub asserts: Vec<AssertDecl>,
    pub shots: Vec<ShotDecl>,
    pub sections: Vec<Section>,
    pub notes: Vec<NoteDecl>,
    /// The input timeline, frame-sorted.
    segments: Vec<Segment>,
}

impl Recipe {
    /// Parse recipe `text`. Fails with the offending 1-based line on any malformed
    /// directive.
    pub fn parse(text: &str) -> Result<Recipe, RecipeError> {
        let mut r = Recipe::default();
        for (i, raw) in text.lines().enumerate() {
            let line_no = i + 1;
            let line = raw.split('#').next().unwrap_or("").trim();
            if line.is_empty() {
                continue;
            }
            if let Some(rest) = line.strip_prefix('@') {
                r.parse_header(rest, line_no)?;
                continue;
            }
            // A timeline line: `<frame>: <rest>`.
            let (frame_str, rest) = line.split_once(':').ok_or_else(|| RecipeError {
                line: line_no,
                reason: "expected '<frame>: ...' or a '@' header directive".into(),
            })?;
            let frame: u64 = frame_str.trim().parse().map_err(|_| RecipeError {
                line: line_no,
                reason: format!("bad frame number {:?}", frame_str.trim()),
            })?;
            let rest = rest.trim();
            if let Some(meta) = rest.strip_prefix('@') {
                r.parse_frame_meta(frame, meta, line_no)?;
            } else {
                r.segments.push(parse_input_segment(frame, rest, line_no)?);
            }
        }
        // Frame-sort the input timeline; a stable sort keeps file order among entries
        // that share a frame (last wins on lookup).
        r.segments.sort_by_key(|s| s.frame);
        Ok(r)
    }

    /// Parse a header `@<keyword> <args>` (the `@` already stripped).
    fn parse_header(&mut self, rest: &str, line_no: usize) -> Result<(), RecipeError> {
        let (kw, args) = split_kw(rest);
        match kw {
            "title" => self.meta.title = Some(args.trim().to_string()),
            "game" => self.meta.game = Some(args.trim().to_string()),
            "sig" => {
                let s = args.trim().trim_start_matches("0x");
                let v = u64::from_str_radix(s, 16).map_err(|_| RecipeError {
                    line: line_no,
                    reason: format!("bad @sig hex {:?}", args.trim()),
                })?;
                self.meta.sig = Some(v);
            }
            "shot-every" => {
                let n: u64 = args.trim().parse().map_err(|_| RecipeError {
                    line: line_no,
                    reason: format!("bad @shot-every count {:?}", args.trim()),
                })?;
                if n == 0 {
                    return Err(RecipeError {
                        line: line_no,
                        reason: "@shot-every needs a positive frame count".into(),
                    });
                }
                self.meta.shot_every = Some(n);
            }
            "watch" => {
                let mut it = args.split_whitespace();
                let (Some(name), Some(ty), Some(addr)) = (it.next(), it.next(), it.next())
                else {
                    return Err(RecipeError {
                        line: line_no,
                        reason: "@watch expects '<name> <type> <addr>'".into(),
                    });
                };
                let ty = ValType::parse(ty).ok_or_else(|| RecipeError {
                    line: line_no,
                    reason: format!("@watch bad type {ty:?} (u8|u16|u32|i32|f32)"),
                })?;
                let addr = parse_hex(addr).ok_or_else(|| RecipeError {
                    line: line_no,
                    reason: format!("@watch bad address {addr:?}"),
                })?;
                self.watches.push(WatchDecl { name: name.to_string(), ty, addr });
            }
            other => {
                return Err(RecipeError {
                    line: line_no,
                    reason: format!("unknown header directive @{other}"),
                })
            }
        }
        Ok(())
    }

    /// Parse a frame-keyed meta directive `@<keyword> <args>` (the `@` stripped).
    fn parse_frame_meta(
        &mut self,
        frame: u64,
        rest: &str,
        line_no: usize,
    ) -> Result<(), RecipeError> {
        let (kw, args) = split_kw(rest);
        match kw {
            "section" => self.sections.push(Section { frame, name: args.trim().to_string() }),
            "shot" => {
                let name = args.trim();
                if name.is_empty() {
                    return Err(RecipeError { line: line_no, reason: "@shot needs a name".into() });
                }
                self.shots.push(ShotDecl { frame, name: name.to_string() });
            }
            "note" => self.notes.push(NoteDecl { frame, todo: false, text: args.trim().to_string() }),
            "todo" => self.notes.push(NoteDecl { frame, todo: true, text: args.trim().to_string() }),
            "assert" => {
                let kind = parse_assert(args, line_no)?;
                self.asserts.push(AssertDecl { frame, kind });
            }
            other => {
                return Err(RecipeError {
                    line: line_no,
                    reason: format!("unknown frame directive @{other}"),
                })
            }
        }
        Ok(())
    }

    /// A [`RecipeWorld`] that replays this recipe's input timeline.
    pub fn into_world(self) -> RecipeWorld {
        RecipeWorld::from_recipe(self)
    }

    /// The number of input timeline segments (mostly for tests).
    pub fn segment_count(&self) -> usize {
        self.segments.len()
    }
}

/// Split `s` into the first whitespace-delimited keyword and the trimmed remainder.
fn split_kw(s: &str) -> (&str, &str) {
    let s = s.trim_start();
    match s.split_once(char::is_whitespace) {
        Some((kw, rest)) => (kw, rest.trim()),
        None => (s, ""),
    }
}

/// Parse a hex address, with or without a `0x` prefix.
fn parse_hex(s: &str) -> Option<u32> {
    u32::from_str_radix(s.trim().trim_start_matches("0x"), 16).ok()
}

/// Parse the body of an `@assert` (either `egress ...` or `<watch> <op> <value>`).
fn parse_assert(args: &str, line_no: usize) -> Result<AssertKind, RecipeError> {
    let args = args.trim();
    if let Some(rest) = args.strip_prefix("egress") {
        let mut it = rest.split_whitespace();
        let kind = it.next().ok_or_else(|| RecipeError {
            line: line_no,
            reason: "@assert egress needs a Kind (SaveWrite|Trophy|ScoreSubmit)".into(),
        })?;
        let mut fields = Vec::new();
        for tok in it {
            fields.push(parse_field_match(tok, line_no)?);
        }
        return Ok(AssertKind::Egress(EgressAssert { kind: kind.to_string(), fields }));
    }
    // Memory assertion: `<watch> <op> <value> [+-<tol>]`.
    let mut it = args.split_whitespace();
    let (Some(watch), Some(op), Some(value)) = (it.next(), it.next(), it.next()) else {
        return Err(RecipeError {
            line: line_no,
            reason: "@assert expects '<watch> <op> <value>' or 'egress <Kind> ...'".into(),
        });
    };
    let op = CmpOp::parse(op).ok_or_else(|| RecipeError {
        line: line_no,
        reason: format!("bad assert op {op:?} (== != < <= > >= ~)"),
    })?;
    let value: f64 = value.parse().map_err(|_| RecipeError {
        line: line_no,
        reason: format!("bad assert value {value:?}"),
    })?;
    // Optional tolerance: `+-<tol>` (or `+/-<tol>`).
    let mut tol = 0.0;
    if let Some(t) = it.next() {
        let t = t.trim_start_matches("+-").trim_start_matches("+/-");
        tol = t.parse().map_err(|_| RecipeError {
            line: line_no,
            reason: format!("bad tolerance {t:?} (expected '+-<number>')"),
        })?;
    }
    Ok(AssertKind::Mem(MemAssert { watch: watch.to_string(), op, value, tol }))
}

/// Parse one egress field matcher like `bytes>=64`, `path=savedata0:data.bin`, or
/// `ascii~URBAN`. The operator is detected inside the token.
fn parse_field_match(tok: &str, line_no: usize) -> Result<FieldMatch, RecipeError> {
    // Order matters: check the two-char operators before the one-char ones.
    for (sym, op) in
        [(">=", FieldOp::Ge), ("<=", FieldOp::Le), (">", FieldOp::Gt), ("<", FieldOp::Lt), ("~", FieldOp::Contains), ("=", FieldOp::Eq)]
    {
        if let Some((field, value)) = tok.split_once(sym) {
            if field.is_empty() {
                break;
            }
            return Ok(FieldMatch {
                field: field.to_string(),
                op,
                value: value.to_string(),
            });
        }
    }
    Err(RecipeError {
        line: line_no,
        reason: format!("bad egress field matcher {tok:?} (want field<op>value)"),
    })
}

/// Parse the input directives of a timeline line into a [`Segment`].
fn parse_input_segment(frame: u64, rest: &str, line_no: usize) -> Result<Segment, RecipeError> {
    let mut input = CtrlFrame::default();
    let mut touch = TouchFrame::default();
    for tok in rest.split_whitespace() {
        let tok_lower = tok.to_ascii_lowercase();
        if let Some((key, val)) = tok_lower.split_once('=') {
            if key == "touch" {
                let (x, y) = val.split_once(',').ok_or_else(|| RecipeError {
                    line: line_no,
                    reason: format!("touch expects X,Y panel coords, got {val:?}"),
                })?;
                let px: u16 = x.parse().map_err(|_| RecipeError {
                    line: line_no,
                    reason: format!("bad touch X {x:?}"),
                })?;
                let py: u16 = y.parse().map_err(|_| RecipeError {
                    line: line_no,
                    reason: format!("bad touch Y {y:?}"),
                })?;
                touch = TouchFrame::single(px, py);
                continue;
            }
            let v: u8 = val.parse().map_err(|_| RecipeError {
                line: line_no,
                reason: format!("bad analog value {val:?} (expected 0..255)"),
            })?;
            match key {
                "lx" => input.lx = v,
                "ly" => input.ly = v,
                "rx" => input.rx = v,
                "ry" => input.ry = v,
                _ => {
                    return Err(RecipeError {
                        line: line_no,
                        reason: format!("unknown analog axis {key:?}"),
                    })
                }
            }
        } else {
            let bit = button_bit(&tok_lower).ok_or_else(|| RecipeError {
                line: line_no,
                reason: format!("unknown button {tok:?}"),
            })?;
            input.buttons |= bit;
        }
    }
    Ok(Segment { frame, input, touch })
}

/// A [`World`] that replays a recipe's scripted input over a virtual clock. Frame
/// advance comes from the scheduler via [`World::set_frame`].
pub struct RecipeWorld {
    segments: Vec<Segment>,
    frame: u64,
    monotonic_us: u64,
    wall_us: u64,
    rng: u64,
}

/// Microseconds per virtual frame (60Hz), used to advance the clock per frame so a
/// title reading elapsed time still sees monotonic progress.
const FRAME_US: u64 = 16_666;

impl RecipeWorld {
    /// Parse `text` into a scripted-input world. Fails with the offending line on a
    /// malformed recipe. Convenience for callers that only need the input timeline
    /// (metadata is dropped); use [`Recipe::parse`] to keep the metadata.
    pub fn parse(text: &str) -> Result<Self, RecipeError> {
        Ok(RecipeWorld::from_recipe(Recipe::parse(text)?))
    }

    /// Build a world from an already-parsed recipe.
    pub fn from_recipe(recipe: Recipe) -> Self {
        RecipeWorld {
            segments: recipe.segments,
            frame: 0,
            monotonic_us: 0,
            wall_us: 1_500_000_000_000_000,
            rng: 0x9E37_79B9_7F4A_7C15,
        }
    }

    /// The segment active at the current frame: the last one whose frame is at or
    /// before it, or `None` before any segment starts (neutral state).
    fn current(&self) -> Option<&Segment> {
        self.segments.iter().rev().find(|s| s.frame <= self.frame)
    }
}

impl World for RecipeWorld {
    fn monotonic_us(&mut self) -> u64 {
        self.monotonic_us
    }
    fn wall_us(&mut self) -> u64 {
        self.wall_us
    }
    fn poll_ctrl(&mut self, _port: u32) -> CtrlFrame {
        self.current().map(|s| s.input).unwrap_or_default()
    }
    fn poll_touch(&mut self, port: u32) -> TouchFrame {
        // Only the front panel (port 0) is scripted; the back panel stays untouched.
        if port == 0 {
            self.current().map(|s| s.touch).unwrap_or_default()
        } else {
            TouchFrame::default()
        }
    }
    fn fill_random(&mut self, buf: &mut [u8]) {
        // SplitMix64, matching DeterministicWorld: deterministic and cheap.
        for chunk in buf.chunks_mut(8) {
            self.rng = self.rng.wrapping_add(0x9E37_79B9_7F4A_7C15);
            let mut z = self.rng;
            z = (z ^ (z >> 30)).wrapping_mul(0xBF58_476D_1CE4_E5B9);
            z = (z ^ (z >> 27)).wrapping_mul(0x94D0_49BB_1331_11EB);
            z ^= z >> 31;
            for (i, b) in chunk.iter_mut().enumerate() {
                *b = (z >> (i * 8)) as u8;
            }
        }
    }
    fn set_frame(&mut self, frame: u64) {
        self.frame = frame;
        // Keep the virtual clock roughly in step with frames so a title polling
        // elapsed time still advances (the preemptive scheduler's own virtual clock
        // drives pacing; this only backstops a title that reads monotonic_us).
        self.monotonic_us = frame.wrapping_mul(FRAME_US);
        self.wall_us = 1_500_000_000_000_000u64.wrapping_add(self.monotonic_us);
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    fn segs(text: &str) -> Vec<Segment> {
        Recipe::parse(text).unwrap().segments
    }

    #[test]
    fn parses_buttons_and_analog_and_comments() {
        let r = "# header\n0:\n30: cross down  # press\n45: lx=0 ly=255\n";
        let segs = segs(r);
        assert_eq!(segs.len(), 3);
        assert_eq!(
            segs[0],
            Segment { frame: 0, input: CtrlFrame::default(), touch: TouchFrame::default() }
        );
        assert_eq!(segs[1].frame, 30);
        assert_eq!(segs[1].input.buttons, 0x4000 | 0x0040); // cross | down
        assert_eq!(segs[2].input.lx, 0);
        assert_eq!(segs[2].input.ly, 255);
        // Analog defaults stay centered where not overridden.
        assert_eq!(segs[2].input.rx, 128);
    }

    #[test]
    fn input_is_sticky_between_segments() {
        let mut w = RecipeWorld::parse("30: cross\n45:\n").unwrap();
        // Before the first segment: neutral.
        w.set_frame(0);
        assert_eq!(w.poll_ctrl(0), CtrlFrame::default());
        // From frame 30 the cross is held...
        w.set_frame(30);
        assert_eq!(w.poll_ctrl(0).buttons, 0x4000);
        w.set_frame(44);
        assert_eq!(w.poll_ctrl(0).buttons, 0x4000);
        // ...and released at 45.
        w.set_frame(45);
        assert_eq!(w.poll_ctrl(0).buttons, 0);
    }

    #[test]
    fn aliases_x_and_l_r() {
        let segs = segs("0: x l r\n");
        assert_eq!(segs[0].input.buttons, 0x4000 | 0x0100 | 0x0200);
    }

    #[test]
    fn later_line_at_same_frame_wins_full_state() {
        // Two entries at the same frame: the stable sort keeps file order, and the
        // reverse scan picks the last, so the second line's state is the one used.
        let mut w = RecipeWorld::parse("10: cross\n10: circle\n").unwrap();
        w.set_frame(10);
        assert_eq!(w.poll_ctrl(0).buttons, 0x2000); // circle only (state replaced)
    }

    #[test]
    fn touch_is_sticky_and_lifts_on_release() {
        let mut w = RecipeWorld::parse("10: touch=450,674\n20:\n").unwrap();
        w.set_frame(0);
        assert_eq!(w.poll_touch(0).count, 0);
        w.set_frame(10);
        let t = w.poll_touch(0);
        assert_eq!(t.count, 1);
        assert_eq!((t.points[0].x, t.points[0].y), (450, 674));
        w.set_frame(19);
        assert_eq!(w.poll_touch(0).count, 1);
        w.set_frame(20);
        assert_eq!(w.poll_touch(0).count, 0);
        w.set_frame(10);
        assert_eq!(w.poll_touch(1).count, 0);
    }

    #[test]
    fn touch_and_buttons_share_a_segment() {
        let mut w = RecipeWorld::parse("5: cross touch=100,200\n").unwrap();
        w.set_frame(5);
        assert_eq!(w.poll_ctrl(0).buttons, 0x4000);
        assert_eq!(w.poll_touch(0).count, 1);
    }

    #[test]
    fn rejects_malformed_touch() {
        assert_eq!(Recipe::parse("0: touch=100\n").unwrap_err().line, 1);
        assert_eq!(Recipe::parse("0: touch=x,200\n").unwrap_err().line, 1);
    }

    #[test]
    fn reports_the_offending_line_on_error() {
        assert_eq!(Recipe::parse("0:\nnope\n").unwrap_err().line, 2);
        assert_eq!(Recipe::parse("5: wiggle\n").unwrap_err().line, 1);
        assert_eq!(Recipe::parse("x: cross\n").unwrap_err().line, 1);
        assert_eq!(Recipe::parse("0: lx=999\n").unwrap_err().line, 1);
    }

    #[test]
    fn clock_advances_with_frame() {
        let mut w = RecipeWorld::parse("").unwrap();
        w.set_frame(0);
        assert_eq!(w.monotonic_us(), 0);
        w.set_frame(60);
        assert_eq!(w.monotonic_us(), 60 * FRAME_US);
    }

    #[test]
    fn parses_header_metadata() {
        let r = Recipe::parse(
            "@title My Run\n@game ABCD00001\n@sig 0x3f9a1c04\n@watch vpos i32 0x81502058\n0: cross\n",
        )
        .unwrap();
        assert_eq!(r.meta.title.as_deref(), Some("My Run"));
        assert_eq!(r.meta.game.as_deref(), Some("ABCD00001"));
        assert_eq!(r.meta.sig, Some(0x3f9a_1c04));
        assert_eq!(r.meta.shot_every, None);
        assert_eq!(r.watches.len(), 1);
        assert_eq!(r.watches[0], WatchDecl { name: "vpos".into(), ty: ValType::I32, addr: 0x8150_2058 });
        // The input timeline still parses alongside the metadata.
        assert_eq!(r.segment_count(), 1);
    }

    #[test]
    fn parses_frame_keyed_meta_directives() {
        let text = "\
0: @section navigate
480: @assert vpos ~ 60 +-3
480: @shot pushing-land
520: @assert egress SaveWrite path=savedata0:data.bin ascii~URBAN bytes>=64
900: @note landed clean after retiming X two frames later
910: @todo grind rail still bails, retune down-flick
";
        let r = Recipe::parse(text).unwrap();
        assert_eq!(r.sections, vec![Section { frame: 0, name: "navigate".into() }]);
        assert_eq!(r.shots, vec![ShotDecl { frame: 480, name: "pushing-land".into() }]);
        assert_eq!(r.asserts.len(), 2);
        assert_eq!(
            r.asserts[0],
            AssertDecl {
                frame: 480,
                kind: AssertKind::Mem(MemAssert {
                    watch: "vpos".into(),
                    op: CmpOp::Approx,
                    value: 60.0,
                    tol: 3.0
                })
            }
        );
        let AssertKind::Egress(e) = &r.asserts[1].kind else { panic!("want egress") };
        assert_eq!(e.kind, "SaveWrite");
        assert_eq!(e.fields.len(), 3);
        assert_eq!(e.fields[0], FieldMatch { field: "path".into(), op: FieldOp::Eq, value: "savedata0:data.bin".into() });
        assert_eq!(e.fields[1], FieldMatch { field: "ascii".into(), op: FieldOp::Contains, value: "URBAN".into() });
        assert_eq!(e.fields[2], FieldMatch { field: "bytes".into(), op: FieldOp::Ge, value: "64".into() });
        assert_eq!(r.notes.len(), 2);
        assert!(!r.notes[0].todo && r.notes[1].todo);
    }

    #[test]
    fn assert_ops_and_types_evaluate() {
        assert!(CmpOp::Approx.eval(58.0, 60.0, 3.0));
        assert!(!CmpOp::Approx.eval(50.0, 60.0, 3.0));
        assert!(CmpOp::Ge.eval(60.0, 60.0, 0.0));
        assert_eq!(ValType::I32.decode(&(-52i32).to_le_bytes()), Some(-52.0));
        assert_eq!(ValType::U8.decode(&[200]), Some(200.0));
    }

    #[test]
    fn parses_shot_every() {
        let r = Recipe::parse("@shot-every 30\n0: cross\n").unwrap();
        assert_eq!(r.meta.shot_every, Some(30));
        assert_eq!(Recipe::parse("@shot-every 0\n").unwrap_err().line, 1);
        assert_eq!(Recipe::parse("@shot-every abc\n").unwrap_err().line, 1);
    }

    #[test]
    fn rejects_bad_meta_directives() {
        assert_eq!(Recipe::parse("@sig zzz\n").unwrap_err().line, 1);
        assert_eq!(Recipe::parse("@watch vpos i32\n").unwrap_err().line, 1);
        assert_eq!(Recipe::parse("@bogus x\n").unwrap_err().line, 1);
        assert_eq!(Recipe::parse("0: @assert vpos !! 3\n").unwrap_err().line, 1);
        assert_eq!(Recipe::parse("0: @nope x\n").unwrap_err().line, 1);
    }
}
