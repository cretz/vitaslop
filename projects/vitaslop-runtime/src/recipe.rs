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
//! A stick can also be given in POLAR form: `lang=<deg>[,<mag>]` and `rang=` set both
//! axes of that stick at once from a compass bearing, `0` up-screen and increasing
//! CLOCKWISE (`90` right, `180` down, `270` left), with an optional magnitude `0..127`
//! that defaults to full deflection. `lang=0` is exactly `lx=128 ly=1`.
//!
//! This exists because a stick is frequently not two independent axes. A title whose
//! left stick is an absolute heading (point where the car should FACE) cannot be
//! driven diagonally by `lx=` alone - every value of `lx` on its own aims either
//! exactly screen-left or exactly screen-right - and writing the pair out by hand
//! turns every heading into arithmetic. Bearings are what the intent is actually
//! expressed in, so they are what a recipe should be able to say. Polar and cartesian
//! forms may be mixed on one line; the last directive touching an axis wins.
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
    // The canonical names come from BUTTON_NAMES (which is also what serialization
    // emits, so the two cannot drift); these are the accepted shorthands.
    let canonical = match name {
        "x" => "cross",
        "l" => "ltrigger",
        "r" => "rtrigger",
        other => other,
    };
    BUTTON_NAMES.iter().find(|(n, _)| *n == canonical).map(|(_, bit)| *bit)
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

    /// The directive keyword for this type (the inverse of [`ValType::parse`]).
    pub fn keyword(self) -> &'static str {
        match self {
            ValType::U8 => "u8",
            ValType::U16 => "u16",
            ValType::U32 => "u32",
            ValType::I32 => "i32",
            ValType::F32 => "f32",
        }
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
    /// Parse a comparison keyword. `None` if unrecognized.
    pub fn parse(s: &str) -> Option<CmpOp> {
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

    /// The directive keyword for this operator (the inverse of [`CmpOp::parse`]).
    pub fn keyword(self) -> &'static str {
        match self {
            CmpOp::Eq => "==",
            CmpOp::Ne => "!=",
            CmpOp::Lt => "<",
            CmpOp::Le => "<=",
            CmpOp::Gt => ">",
            CmpOp::Ge => ">=",
            CmpOp::Approx => "~",
        }
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

/// One timeline entry: the input state that takes effect at `frame` and holds until
/// the next entry. Public because an interactive session is nothing more than a
/// recipe whose segments are appended as they are played (see [`Timeline`]).
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub struct InputSegment {
    pub frame: u64,
    pub input: CtrlFrame,
    pub touch: TouchFrame,
}

/// The frame-keyed, sticky input timeline a [`RecipeWorld`] replays: the segment
/// active at a frame is the last one at or before it.
///
/// It is APPENDABLE at runtime, and that is deliberate. A scripted replay fills it
/// once from a parsed recipe; a live session ([`SharedTimeline`]) starts from the
/// same place and pushes a new segment each time the player changes the input. The
/// two are then the same object, which is what lets an interactive playthrough be
/// written straight back out as a committed, reproducible recipe - the session does
/// not approximate a recipe, it *is* one being authored.
#[derive(Clone, Debug, Default, PartialEq, Eq)]
pub struct Timeline {
    segments: Vec<InputSegment>,
}

/// A [`Timeline`] shared between the world the guest polls and whoever drives it.
/// One lock per pad poll (once per display frame) is free at this granularity.
pub type SharedTimeline = std::sync::Arc<std::sync::Mutex<Timeline>>;

impl Timeline {
    /// A timeline over `segments`, frame-sorted (a stable sort keeps authoring order
    /// among entries sharing a frame, so the last one written wins).
    pub fn new(mut segments: Vec<InputSegment>) -> Timeline {
        segments.sort_by_key(|s| s.frame);
        Timeline { segments }
    }

    /// The input state in effect at `frame`: the last segment at or before it, or
    /// neutral before any segment starts.
    pub fn at(&self, frame: u64) -> (CtrlFrame, TouchFrame) {
        match self.segments.iter().rev().find(|s| s.frame <= frame) {
            Some(s) => (s.input, s.touch),
            None => (CtrlFrame::default(), TouchFrame::default()),
        }
    }

    /// Append a segment, keeping the timeline frame-sorted. A segment pushed at a
    /// frame that already has one is ordered after it, so the newest write wins -
    /// exactly the "I just pressed a button" semantics a live session needs.
    pub fn push(&mut self, seg: InputSegment) {
        let at = self.segments.partition_point(|s| s.frame <= seg.frame);
        self.segments.insert(at, seg);
    }

    /// Every segment, in frame order.
    pub fn segments(&self) -> &[InputSegment] {
        &self.segments
    }
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

impl FieldOp {
    /// The symbol this operator is written with inside a field matcher.
    pub fn symbol(self) -> &'static str {
        match self {
            FieldOp::Eq => "=",
            FieldOp::Ge => ">=",
            FieldOp::Le => "<=",
            FieldOp::Gt => ">",
            FieldOp::Lt => "<",
            FieldOp::Contains => "~",
        }
    }
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
    segments: Vec<InputSegment>,
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

    /// The input timeline, frame-sorted.
    pub fn segments(&self) -> &[InputSegment] {
        &self.segments
    }

    /// Replace the input timeline (frame-sorting it). Used when writing a live
    /// session's played timeline back into the recipe it started from.
    pub fn set_segments(&mut self, segments: Vec<InputSegment>) {
        self.segments = Timeline::new(segments).segments;
    }

    /// Parse one timeline line's INPUT directives (`cross ly=0 touch=450,674`) into
    /// the state they select. Empty text releases everything. This is the same
    /// grammar the recipe file uses, exposed so an interactive session and a recipe
    /// speak one input language rather than two.
    pub fn parse_input(text: &str) -> Result<(CtrlFrame, TouchFrame), String> {
        parse_input_segment(0, text, 0)
            .map(|s| (s.input, s.touch))
            .map_err(|e| e.reason)
    }

    /// Serialize back to recipe text: the header directives, then one line per
    /// frame-keyed item (input segments, sections, asserts, shots, notes) in frame
    /// order. Round-trips through [`Recipe::parse`], which is what makes a session's
    /// played timeline directly committable.
    pub fn to_text(&self) -> String {
        let mut s = String::new();
        if let Some(t) = &self.meta.title {
            s.push_str(&format!("@title {t}\n"));
        }
        if let Some(g) = &self.meta.game {
            s.push_str(&format!("@game  {g}\n"));
        }
        if let Some(n) = self.meta.shot_every {
            s.push_str(&format!("@shot-every {n}\n"));
        }
        if let Some(sig) = self.meta.sig {
            s.push_str(&format!("@sig {sig:#018x}\n"));
        }
        for w in &self.watches {
            s.push_str(&format!("@watch {} {} {:#x}\n", w.name, w.ty.keyword(), w.addr));
        }
        s.push('\n');

        // Every frame-keyed line, tagged with its frame, emitted in frame order. A
        // stable sort keeps each kind's authoring order within one frame.
        let mut lines: Vec<(u64, String)> = Vec::new();
        for seg in &self.segments {
            lines.push((seg.frame, format_input(seg)));
        }
        for sec in &self.sections {
            lines.push((sec.frame, format!("@section {}", sec.name)));
        }
        for a in &self.asserts {
            lines.push((a.frame, format!("@assert {}", format_assert(&a.kind))));
        }
        for sh in &self.shots {
            lines.push((sh.frame, format!("@shot {}", sh.name)));
        }
        for n in &self.notes {
            lines.push((n.frame, format!("@{} {}", if n.todo { "todo" } else { "note" }, n.text)));
        }
        lines.sort_by_key(|(f, _)| *f);
        for (frame, body) in lines {
            s.push_str(&format!("{frame}: {body}\n"));
        }
        s
    }
}

/// Render one input segment's directives (the inverse of [`parse_input_segment`]).
/// An all-neutral segment renders as an empty body, which is the recipe's "release
/// everything" line.
fn format_input(seg: &InputSegment) -> String {
    let mut parts: Vec<String> = Vec::new();
    for (name, bit) in BUTTON_NAMES {
        if seg.input.buttons & bit != 0 {
            parts.push((*name).to_string());
        }
    }
    for (axis, v) in [("lx", seg.input.lx), ("ly", seg.input.ly), ("rx", seg.input.rx), ("ry", seg.input.ry)] {
        if v != 128 {
            parts.push(format!("{axis}={v}"));
        }
    }
    if let Some(p) = seg.touch.active().first() {
        parts.push(format!("touch={},{}", p.x, p.y));
    }
    parts.join(" ")
}

/// Render an assertion back to its `@assert` body.
fn format_assert(kind: &AssertKind) -> String {
    match kind {
        AssertKind::Mem(m) => {
            let op = m.op.keyword();
            if m.tol != 0.0 {
                format!("{} {op} {} +-{}", m.watch, m.value, m.tol)
            } else {
                format!("{} {op} {}", m.watch, m.value)
            }
        }
        AssertKind::Egress(e) => {
            let mut s = format!("egress {}", e.kind);
            for f in &e.fields {
                s.push_str(&format!(" {}{}{}", f.field, f.op.symbol(), f.value));
            }
            s
        }
    }
}

/// The canonical directive keyword for each button bit, in the order a serialized
/// line lists them. The single source of truth for the round trip: [`button_bit`]
/// also accepts the aliases (`x`, `l`, `r`), which never round-trip out.
const BUTTON_NAMES: &[(&str, u32)] = &[
    ("select", 0x0000_0001),
    ("l3", 0x0000_0002),
    ("r3", 0x0000_0004),
    ("start", 0x0000_0008),
    ("up", 0x0000_0010),
    ("right", 0x0000_0020),
    ("down", 0x0000_0040),
    ("left", 0x0000_0080),
    ("ltrigger", 0x0000_0100),
    ("rtrigger", 0x0000_0200),
    ("l1", 0x0000_0400),
    ("r1", 0x0000_0800),
    ("triangle", 0x0000_1000),
    ("circle", 0x0000_2000),
    ("cross", 0x0000_4000),
    ("square", 0x0000_8000),
];

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

/// Resolve a polar stick directive's `<deg>[,<mag>]` argument to the `(x, y)` pair
/// the pad actually carries.
///
/// The bearing is a compass bearing in the plane the player sees: `0` points
/// up-screen and it increases CLOCKWISE, so `90` is right and `270` is left. That
/// matches how a control legend and a human both describe a direction, and it is the
/// convention the axes already imply (a stick pushed up reads `ly=1`, pushed right
/// reads `lx=255`).
///
/// `mag` is the deflection in pad units, `0` centred to `127` fully pushed, and
/// defaults to full. An out-of-range magnitude is an ERROR rather than a clamp: a
/// recipe asking for `mag=200` has a wrong model of the pad, and silently giving it
/// 127 would hide that until someone wondered why two headings drove identically.
fn polar_stick(val: &str, line_no: usize) -> Result<(u8, u8), RecipeError> {
    let (deg_s, mag_s) = match val.split_once(',') {
        Some((d, m)) => (d, Some(m)),
        None => (val, None),
    };
    let deg: f64 = deg_s.trim().parse().map_err(|_| RecipeError {
        line: line_no,
        reason: format!("bad stick bearing {deg_s:?} (want degrees, 0 = up-screen, clockwise)"),
    })?;
    let mag: f64 = match mag_s {
        Some(m) => m.trim().parse().map_err(|_| RecipeError {
            line: line_no,
            reason: format!("bad stick magnitude {m:?} (want 0..127)"),
        })?,
        None => 127.0,
    };
    if !(0.0..=127.0).contains(&mag) {
        return Err(RecipeError {
            line: line_no,
            reason: format!("stick magnitude {mag} out of range (0..127, 127 = fully deflected)"),
        });
    }
    let rad = deg.to_radians();
    // Screen y grows DOWNWARD in pad units (up-screen is the low end), hence the
    // subtraction on the y axis and not on x.
    let x = (128.0 + mag * rad.sin()).round();
    let y = (128.0 - mag * rad.cos()).round();
    Ok((x.clamp(0.0, 255.0) as u8, y.clamp(0.0, 255.0) as u8))
}

/// Parse the input directives of a timeline line into an [`InputSegment`].
fn parse_input_segment(frame: u64, rest: &str, line_no: usize) -> Result<InputSegment, RecipeError> {
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
            if key == "lang" || key == "rang" {
                let (x, y) = polar_stick(val, line_no)?;
                if key == "lang" {
                    input.lx = x;
                    input.ly = y;
                } else {
                    input.rx = x;
                    input.ry = y;
                }
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
    Ok(InputSegment { frame, input, touch })
}

/// A [`World`] that replays a recipe's scripted input over a virtual clock. Frame
/// advance comes from the scheduler via [`World::set_frame`].
pub struct RecipeWorld {
    timeline: SharedTimeline,
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
        RecipeWorld::from_timeline(std::sync::Arc::new(std::sync::Mutex::new(Timeline::new(
            recipe.segments,
        ))))
    }

    /// Build a world over an externally-owned timeline. A live session holds the
    /// other end and appends to it as the run proceeds.
    pub fn from_timeline(timeline: SharedTimeline) -> Self {
        RecipeWorld {
            timeline,
            frame: 0,
            monotonic_us: 0,
            wall_us: 1_500_000_000_000_000,
            rng: 0x9E37_79B9_7F4A_7C15,
        }
    }

    /// A handle to the timeline this world replays, for a driver that appends to it.
    pub fn timeline(&self) -> SharedTimeline {
        self.timeline.clone()
    }

    /// The input state in effect at the current frame.
    fn current(&self) -> (CtrlFrame, TouchFrame) {
        self.timeline.lock().unwrap().at(self.frame)
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
        self.current().0
    }
    fn poll_touch(&mut self, port: u32) -> TouchFrame {
        // Only the front panel (port 0) is scripted; the back panel stays untouched.
        if port == 0 {
            self.current().1
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

    fn segs(text: &str) -> Vec<InputSegment> {
        Recipe::parse(text).unwrap().segments
    }

    #[test]
    fn parses_buttons_and_analog_and_comments() {
        let r = "# header\n0:\n30: cross down  # press\n45: lx=0 ly=255\n";
        let segs = segs(r);
        assert_eq!(segs.len(), 3);
        assert_eq!(
            segs[0],
            InputSegment { frame: 0, input: CtrlFrame::default(), touch: TouchFrame::default() }
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

    /// Serialization is only useful if what a live session writes out parses back to
    /// the same run - otherwise "save the session as a recipe" quietly loses inputs
    /// or assertions, and the reproducible artifact is a lie.
    #[test]
    fn to_text_round_trips_through_parse() {
        let src = "\
@title Round trip
@game  ABCD00001
@shot-every 30
@watch speed f32 0x81502058
@watch flag u8 0x815020ff

0: @section boot
30: cross down ly=0 touch=450,674
45:
60: @assert speed ~ 12.5 +-0.5
60: @assert flag == 1
70: @assert egress SaveWrite path=savedata0:data.bin ascii~URBAN bytes>=64
80: @shot landed
90: @note found the speed word by scan
95: @todo drive the second lap
";
        let a = Recipe::parse(src).unwrap();
        let b = Recipe::parse(&a.to_text()).unwrap();
        assert_eq!(a, b, "recipe did not survive a to_text/parse round trip");
        // And the timeline it drives is the same one, segment for segment.
        assert_eq!(a.segments(), b.segments());
    }

    #[test]
    fn a_timeline_appended_to_at_runtime_takes_effect() {
        // The live-session model: the world replays a shared timeline that its
        // driver extends as the run proceeds.
        let world = RecipeWorld::parse("0:\n").unwrap();
        let timeline = world.timeline();
        let mut w = world;
        w.set_frame(10);
        assert_eq!(w.poll_ctrl(0).buttons, 0);
        let (input, touch) = Recipe::parse_input("cross ly=0").unwrap();
        timeline.lock().unwrap().push(InputSegment { frame: 11, input, touch });
        w.set_frame(11);
        assert_eq!(w.poll_ctrl(0).buttons, 0x4000);
        assert_eq!(w.poll_ctrl(0).ly, 0);
        // A push at a frame that already has a segment supersedes it.
        let (input, touch) = Recipe::parse_input("").unwrap();
        timeline.lock().unwrap().push(InputSegment { frame: 11, input, touch });
        assert_eq!(w.poll_ctrl(0).buttons, 0);
    }

    #[test]
    fn parse_input_accepts_one_lines_directives() {
        let (input, touch) = Recipe::parse_input("l ly=0 touch=100,200").unwrap();
        assert_eq!(input.buttons, 0x0100);
        assert_eq!(input.ly, 0);
        assert_eq!(touch.count, 1);
        assert!(Recipe::parse_input("nosuchbutton").is_err());
    }

    #[test]
    fn polar_stick_directives_set_both_axes() {
        // The four cardinals, which are what a control legend names. Up-screen is
        // the LOW end of the y axis, so bearing 0 must reach ly=1, not ly=255.
        for (spec, lx, ly) in [
            ("lang=0", 128, 1),
            ("lang=90", 255, 128),
            ("lang=180", 128, 255),
            ("lang=270", 1, 128),
        ] {
            let (input, _) = Recipe::parse_input(spec).unwrap();
            assert_eq!((input.lx, input.ly), (lx, ly), "{spec}");
        }
        // The diagonal no `lx=` alone can express - the case polar form exists for.
        let (input, _) = Recipe::parse_input("lang=45").unwrap();
        assert!(input.lx > 128 && input.ly < 128, "45 deg aims up-and-right");
        // Bearings wrap, and the right stick has its own directive.
        let (a, _) = Recipe::parse_input("lang=-90").unwrap();
        let (b, _) = Recipe::parse_input("lang=270").unwrap();
        assert_eq!((a.lx, a.ly), (b.lx, b.ly));
        let (input, _) = Recipe::parse_input("rang=0").unwrap();
        assert_eq!((input.rx, input.ry, input.lx, input.ly), (128, 1, 128, 128));
        // A partial deflection, and full is the default.
        let (input, _) = Recipe::parse_input("lang=90,60").unwrap();
        assert_eq!((input.lx, input.ly), (188, 128));
        // Polar and cartesian mix on one line; the later directive wins.
        let (input, _) = Recipe::parse_input("lang=0 ly=200").unwrap();
        assert_eq!((input.lx, input.ly), (128, 200));
        // A magnitude the pad cannot reach is an error, never a silent clamp.
        assert!(Recipe::parse_input("lang=0,200").is_err());
        assert!(Recipe::parse_input("lang=up").is_err());
    }
}
