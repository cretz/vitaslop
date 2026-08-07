//! Evaluating a recipe's observations - `@watch`, `@assert`, `@sig` - independently of
//! which engine ran it.
//!
//! # Why this is not in the native runner
//!
//! A recipe is the project's oracle: it says what the title should be doing at a given
//! frame, and a run either meets it or does not. That was only ever true of NATIVE runs.
//! The browser shared the recipe's *input* timeline (`RecipeWorld`) but none of its
//! checks, because they lived in `vitaslop-native`, which a `wasm32` build cannot reach.
//! So a browser run replayed the same buttons and then verified NOTHING - the one engine
//! whose correctness is hardest to eyeball was the one with no assertions.
//!
//! Everything here depends only on the [`Capture`] and on being able to read guest
//! memory, both of which each engine already has. What stays engine-specific is
//! rendering a screenshot (a native PNG file versus a canvas) - not what is checked, only
//! where the picture goes.
use std::collections::HashMap;

use crate::capture::{Capture, EgressKind};
use crate::recipe::{
    AssertKind, CmpOp, EgressAssert, FieldMatch, FieldOp, MemAssert, Recipe, WatchDecl,
};

/// Reads guest memory. Native reads through the wasmtime store, the browser through its
/// shared `SharedArrayBuffer` view; a watch sample is the same question either way.
pub trait GuestRead {
    /// Fill `out` from guest address `addr`. False if the range is not mapped, in which
    /// case the watch reports `oob` rather than a plausible zero.
    fn read_into(&self, addr: u32, out: &mut [u8]) -> bool;
}

/// The result of one `@assert`.
#[derive(Clone, Debug, PartialEq, Eq)]
pub struct AssertOutcome {
    pub frame: u64,
    /// A human statement of what was checked.
    pub desc: String,
    pub passed: bool,
    /// What was actually seen, so a failure needs no second run to interpret.
    pub detail: String,
}

/// Sample one `@watch` from guest memory. `None` if the address is not mapped.
pub fn sample_watch(mem: &impl GuestRead, w: &WatchDecl) -> Option<f64> {
    let mut buf = [0u8; 4];
    let width = w.ty.width();
    if !mem.read_into(w.addr, &mut buf[..width]) {
        return None;
    }
    w.ty.decode(&buf[..width])
}

/// Evaluate one assertion at `frame` against the capture and this frame's watch samples.
pub fn eval_assert(
    cap: &Capture,
    frame: u64,
    kind: &AssertKind,
    watch_vals: &HashMap<String, f64>,
) -> AssertOutcome {
    let desc = describe_assert(kind);
    match kind {
        AssertKind::Mem(m) => eval_mem_assert(frame, m, watch_vals, desc),
        AssertKind::Egress(e) => eval_egress_assert(cap, frame, e, desc),
    }
}

fn eval_mem_assert(
    frame: u64,
    m: &MemAssert,
    watch_vals: &HashMap<String, f64>,
    desc: String,
) -> AssertOutcome {
    match watch_vals.get(&m.watch) {
        Some(&actual) => {
            let passed = m.op.eval(actual, m.value, m.tol);
            AssertOutcome {
                frame,
                desc,
                passed,
                detail: format!("actual {}={}", m.watch, format_f64(actual)),
            }
        }
        None => AssertOutcome {
            frame,
            desc,
            passed: false,
            detail: format!("watch {:?} not declared or not sampled", m.watch),
        },
    }
}

fn eval_egress_assert(cap: &Capture, frame: u64, e: &EgressAssert, desc: String) -> AssertOutcome {
    // An egress event at or before this frame matching the kind and every field.
    let hit = cap.egress.iter().filter(|ev| ev.frame <= frame).any(|ev| egress_matches(&ev.kind, e));
    AssertOutcome {
        frame,
        desc,
        passed: hit,
        detail: if hit {
            "matched an egress event".to_string()
        } else {
            "no matching egress event".to_string()
        },
    }
}

/// Does egress event `ev` match assertion `want` (kind plus every field matcher)?
fn egress_matches(ev: &EgressKind, want: &EgressAssert) -> bool {
    let kind_ok = match ev {
        EgressKind::SaveWrite { .. } => want.kind == "SaveWrite",
        EgressKind::Trophy { .. } => want.kind == "Trophy",
        EgressKind::ScoreSubmit { .. } => want.kind == "ScoreSubmit",
    };
    if !kind_ok {
        return false;
    }
    want.fields.iter().all(|f| field_matches(ev, f))
}

/// Evaluate one field matcher against an egress event.
fn field_matches(ev: &EgressKind, f: &FieldMatch) -> bool {
    // Resolve the field to either a string or a number, then apply the operator.
    match (ev, f.field.as_str()) {
        (EgressKind::SaveWrite { path, .. }, "path") => str_match(path, f),
        (EgressKind::SaveWrite { ascii, .. }, "ascii") => str_match(ascii, f),
        (EgressKind::SaveWrite { bytes, .. }, "bytes") => num_match(*bytes as f64, f),
        (EgressKind::Trophy { id }, "id") => num_match(*id as f64, f),
        (EgressKind::ScoreSubmit { board, .. }, "board") => num_match(*board as f64, f),
        (EgressKind::ScoreSubmit { score, .. }, "score") => num_match(*score as f64, f),
        // An unknown field for this kind never matches.
        _ => false,
    }
}

fn str_match(actual: &str, f: &FieldMatch) -> bool {
    match f.op {
        FieldOp::Eq => actual == f.value,
        FieldOp::Contains => actual.contains(&f.value),
        // Ordering ops are meaningless on strings.
        _ => false,
    }
}

fn num_match(actual: f64, f: &FieldMatch) -> bool {
    let Ok(want) = f.value.parse::<f64>() else { return false };
    match f.op {
        FieldOp::Eq => actual == want,
        FieldOp::Ge => actual >= want,
        FieldOp::Le => actual <= want,
        FieldOp::Gt => actual > want,
        FieldOp::Lt => actual < want,
        // Substring on a number is meaningless.
        FieldOp::Contains => false,
    }
}

/// A human statement of what an assertion checks.
pub fn describe_assert(kind: &AssertKind) -> String {
    match kind {
        AssertKind::Mem(m) => {
            let op = match m.op {
                CmpOp::Eq => "==",
                CmpOp::Ne => "!=",
                CmpOp::Lt => "<",
                CmpOp::Le => "<=",
                CmpOp::Gt => ">",
                CmpOp::Ge => ">=",
                CmpOp::Approx => "~",
            };
            if m.tol != 0.0 {
                format!("{} {} {} +-{}", m.watch, op, format_f64(m.value), format_f64(m.tol))
            } else {
                format!("{} {} {}", m.watch, op, format_f64(m.value))
            }
        }
        AssertKind::Egress(e) => {
            let fields: Vec<String> =
                e.fields.iter().map(|f| format!("{}{}{}", f.field, f.op.symbol(), f.value)).collect();
            format!("egress {} {}", e.kind, fields.join(" "))
        }
    }
}

/// Format an `f64` compactly: integers without a trailing `.0`.
pub fn format_f64(x: f64) -> String {
    if x.fract() == 0.0 && x.abs() < 1e15 {
        format!("{}", x as i64)
    } else {
        format!("{x}")
    }
}

/// Drives a recipe's observations frame by frame, accumulating the outcomes.
///
/// Both engines' run loops call [`on_frame`](Self::on_frame) at every display flip and
/// [`finish`](Self::finish) at the end, so the two produce the SAME verdict from the same
/// recipe file. Screenshots are the one thing left to the caller - it is handed the
/// frame's shot names and decides where a picture goes.
pub struct RecipeEval<'a> {
    recipe: &'a Recipe,
    /// Outcomes so far, in frame order.
    pub asserts: Vec<AssertOutcome>,
    /// One CSV row per observed frame: `frame,<watch>,...`, `oob` for an unmapped read.
    pub watch_csv: String,
    /// Screenshot cadence (`@shot-every`, or a caller override).
    shot_every: Option<u64>,
    /// The freshest sampled value of each watch, by name.
    ///
    /// PERSISTENT across frames, which is the long-standing native behaviour: a watch
    /// whose address reads out of bounds on one frame keeps its last good value, and an
    /// assertion on that frame is answered from it. That is preserved here rather than
    /// quietly changed, but it is a real hazard - an assertion can pass on a value from
    /// an earlier frame - so a stale answer SAYS it is stale in its detail line instead
    /// of reading like a fresh measurement.
    watch_vals: HashMap<String, f64>,
    /// Watch names sampled successfully on the CURRENT frame, to spot the stale case.
    fresh: std::collections::HashSet<String>,
}

impl<'a> RecipeEval<'a> {
    /// Start evaluating `recipe`. `shot_every` overrides the recipe's own cadence when
    /// `Some`.
    pub fn new(recipe: &'a Recipe, shot_every: Option<u64>) -> RecipeEval<'a> {
        let mut watch_csv = String::new();
        if !recipe.watches.is_empty() {
            watch_csv.push_str("frame");
            for w in &recipe.watches {
                watch_csv.push(',');
                watch_csv.push_str(&w.name);
            }
            watch_csv.push('\n');
        }
        RecipeEval {
            recipe,
            asserts: Vec::new(),
            watch_csv,
            shot_every: shot_every.or(recipe.meta.shot_every).filter(|&n| n > 0),
            watch_vals: HashMap::new(),
            fresh: std::collections::HashSet::new(),
        }
    }

    /// Observe display frame `frame`: sample every `@watch`, evaluate every `@assert` due
    /// here, and return the screenshot names this frame owes (explicit `@shot` points
    /// first, then the cadence shot named by the active section).
    pub fn on_frame(&mut self, frame: u64, mem: &impl GuestRead, cap: &Capture) -> Vec<String> {
        self.fresh.clear();
        if !self.recipe.watches.is_empty() {
            self.watch_csv.push_str(&frame.to_string());
            for w in &self.recipe.watches {
                self.watch_csv.push(',');
                match sample_watch(mem, w) {
                    Some(v) => {
                        self.watch_csv.push_str(&format_f64(v));
                        self.watch_vals.insert(w.name.clone(), v);
                        self.fresh.insert(w.name.clone());
                    }
                    // NOT a zero. A watch whose address is unmapped and a watch whose
                    // value is genuinely zero are different findings, and only one of
                    // them means the address is wrong.
                    None => self.watch_csv.push_str("oob"),
                }
            }
            self.watch_csv.push('\n');
        }

        for a in self.recipe.asserts.iter().filter(|a| a.frame == frame) {
            let mut outcome = eval_assert(cap, frame, &a.kind, &self.watch_vals);
            // Say when the answer came from an earlier frame's sample - see `watch_vals`.
            if let AssertKind::Mem(m) = &a.kind {
                if self.watch_vals.contains_key(&m.watch) && !self.fresh.contains(&m.watch) {
                    outcome.detail.push_str(" (STALE - the watch read out of bounds this frame)");
                }
            }
            self.asserts.push(outcome);
        }

        let mut shots: Vec<String> =
            self.recipe.shots.iter().filter(|s| s.frame == frame).map(|s| s.name.clone()).collect();
        if let Some(n) = self.shot_every {
            if n > 0 && frame % n == 0 {
                let section =
                    self.recipe.sections.iter().rev().find(|s| s.frame <= frame).map(|s| &s.name);
                shots.push(match section {
                    Some(sec) => format!("{sec}-f{frame:05}"),
                    None => format!("f{frame:05}"),
                });
            }
        }
        shots
    }

    /// Close the run at `frames`: record every assertion past the frame actually reached
    /// as a FAILURE, and check the recipe's pinned `@sig` against `sig`.
    ///
    /// The first half matters more than it looks. A run that stalls short of its last
    /// assertion has not passed the ones it never reached, and reporting only the
    /// assertions that ran turns "the title stopped half way" into "all checks passed".
    pub fn finish(&mut self, frames: u64, sig: u64) {
        for a in self.recipe.asserts.iter().filter(|a| a.frame > frames) {
            self.asserts.push(AssertOutcome {
                frame: a.frame,
                desc: describe_assert(&a.kind),
                passed: false,
                detail: format!("frame {} never reached (run stopped at {frames})", a.frame),
            });
        }
        if let Some(expected) = self.recipe.meta.sig {
            self.asserts.push(AssertOutcome {
                frame: frames,
                desc: "determinism @sig".to_string(),
                passed: expected == sig,
                detail: format!("expected {expected:#018x}, got {sig:#018x}"),
            });
        }
    }

    /// Every assertion passed (vacuously true for a recipe with none).
    pub fn passed(&self) -> bool {
        self.asserts.iter().all(|a| a.passed)
    }

    /// A one-line verdict: `N/M assertions passed`, plus the first failure if any.
    pub fn summary(&self) -> String {
        let total = self.asserts.len();
        let ok = self.asserts.iter().filter(|a| a.passed).count();
        match self.asserts.iter().find(|a| !a.passed) {
            Some(f) => format!(
                "{ok}/{total} assertions passed; first failure at frame {}: {} ({})",
                f.frame, f.desc, f.detail
            ),
            None => format!("{ok}/{total} assertions passed"),
        }
    }
}
