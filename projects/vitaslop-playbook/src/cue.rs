//! Cues: what a playbook can inspect and wait on.
//!
//! A cue is declared once (it gets an id) and is then either READ (its value now) or WAITED
//! on. Three families:
//!
//! - **event** - fires at a moment: a host call ([`CueDef::Call`]), a change, a duration.
//! - **state** - has a value at every input frame: guest memory ([`CueDef::Mem`]).
//! - **condition** - a truth over a state cue: [`CueDef::Cmp`], [`CueDef::Stable`].
//!
//! Waiting arms a fresh [`Armed`] tree over the cue, so arm-relative cues (durations,
//! `after`, the latches of `all`) start from the wait, while state history (how long a value
//! has been stable) is tracked continuously by the engine and is the same for every reader.
use crate::proto::CmpOp;

pub type CueId = usize;

/// A memory cue's storage type.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub enum MemTy {
    U8,
    U16,
    U32,
    I8,
    I16,
    I32,
    F32,
}

impl MemTy {
    pub fn parse(s: &str) -> Option<MemTy> {
        Some(match s {
            "u8" => MemTy::U8,
            "u16" => MemTy::U16,
            "u32" => MemTy::U32,
            "i8" => MemTy::I8,
            "i16" => MemTy::I16,
            "i32" => MemTy::I32,
            "f32" => MemTy::F32,
            _ => return None,
        })
    }

    pub fn width(self) -> usize {
        match self {
            MemTy::U8 | MemTy::I8 => 1,
            MemTy::U16 | MemTy::I16 => 2,
            _ => 4,
        }
    }

    /// Widen little-endian `b` (exactly [`Self::width`] bytes) to `f64`. A NaN float reads as
    /// `None`: it compares unequal to itself, so a "changed" cue on it would fire every frame.
    pub fn decode(self, b: &[u8]) -> Option<f64> {
        Some(match self {
            MemTy::U8 => b[0] as f64,
            MemTy::I8 => b[0] as i8 as f64,
            MemTy::U16 => u16::from_le_bytes([b[0], b[1]]) as f64,
            MemTy::I16 => i16::from_le_bytes([b[0], b[1]]) as f64,
            MemTy::U32 => u32::from_le_bytes([b[0], b[1], b[2], b[3]]) as f64,
            MemTy::I32 => i32::from_le_bytes([b[0], b[1], b[2], b[3]]) as f64,
            MemTy::F32 => {
                let f = f32::from_le_bytes([b[0], b[1], b[2], b[3]]) as f64;
                if f.is_nan() {
                    return None;
                }
                f
            }
        })
    }
}

/// A declared cue.
#[derive(Clone, Debug, PartialEq)]
pub enum CueDef {
    /// A host call by name; `filter` keeps only calls whose C-string argument in register
    /// `arg` contains `has`. Its value is how many matching calls have been made.
    Call { name: String, filter: Option<(u8, String)> },
    /// Guest memory: start at `base`, follow `offs` (read a pointer, add the offset) for each.
    Mem { ty: MemTy, base: u32, offs: Vec<u32> },
    /// `n` input frames after the wait begins.
    Frames(u64),
    /// `s` guest seconds after the wait begins.
    Seconds(f64),
    /// `src <op> value`.
    Cmp { src: CueId, op: CmpOp, value: f64 },
    /// `src`'s value changed on this input frame.
    Changed(CueId),
    /// `src`'s value has held for `n` input frames.
    Stable(CueId, u64),
    /// `a`'s first firing after `b` has fired.
    After(CueId, CueId),
    Any(Vec<CueId>),
    All(Vec<CueId>),
}

impl CueDef {
    /// The cues this one is built from.
    pub fn children(&self) -> Vec<CueId> {
        match self {
            CueDef::Cmp { src, .. } | CueDef::Changed(src) | CueDef::Stable(src, _) => vec![*src],
            CueDef::After(a, b) => vec![*a, *b],
            CueDef::Any(v) | CueDef::All(v) => v.clone(),
            _ => vec![],
        }
    }

    /// A short human description, for logs and timeout messages.
    pub fn describe(&self, label: &dyn Fn(CueId) -> String) -> String {
        match self {
            CueDef::Call { name, filter: None } => format!("call {name}"),
            CueDef::Call { name, filter: Some((arg, has)) } => format!("call {name}(r{arg} has {has:?})"),
            CueDef::Mem { ty, base, offs } => {
                let mut s = format!("{ty:?}@{base:#x}").to_lowercase();
                for o in offs {
                    s.push_str(&format!("->+{o:#x}"));
                }
                s
            }
            CueDef::Frames(n) => format!("{n} input frames"),
            CueDef::Seconds(s) => format!("{s} s"),
            CueDef::Cmp { src, op, value } => format!("{} {} {value}", label(*src), op.symbol()),
            CueDef::Changed(src) => format!("{} changed", label(*src)),
            CueDef::Stable(src, n) => format!("{} stable {n}", label(*src)),
            CueDef::After(a, b) => format!("{} after {}", label(*a), label(*b)),
            CueDef::Any(v) => format!("any({})", v.iter().map(|c| label(*c)).collect::<Vec<_>>().join(", ")),
            CueDef::All(v) => format!("all({})", v.iter().map(|c| label(*c)).collect::<Vec<_>>().join(", ")),
        }
    }
}

/// What the engine knows about every cue at the current input frame.
pub trait CueState {
    fn def(&self, id: CueId) -> &CueDef;
    /// A call cue's count, a memory cue's value, a condition's truth (1/0); `None` when there
    /// is no value (unmapped memory, a duration).
    fn value(&self, id: CueId) -> Option<f64>;
    /// Whether a state cue's value changed on this input frame.
    fn changed_now(&self, id: CueId) -> bool;
    fn input_frame(&self) -> u64;
    fn guest_us(&self) -> u64;
}

/// A wait in progress over one cue, carrying the arm-relative state its kind needs.
#[derive(Debug)]
pub enum Armed {
    /// An event counted from the arm: fires when the count passes `base`.
    Count { id: CueId, base: f64 },
    /// A bare state cue: fires when its value differs from the value at the arm.
    Change { id: CueId, base: Option<f64> },
    /// A condition: fires while it holds.
    Level { id: CueId },
    /// Fires on a change seen after the arm frame.
    Edge { id: CueId, armed_at: u64 },
    Frames { until: u64 },
    Seconds { until_us: u64 },
    After { a: CueId, b: Box<Armed>, then: Option<Box<Armed>> },
    Any(Vec<Armed>),
    All(Vec<(Armed, bool)>),
}

impl Armed {
    /// Arm a wait on `id` at the state's current input frame.
    pub fn arm(st: &dyn CueState, id: CueId) -> Armed {
        match st.def(id) {
            CueDef::Call { .. } => Armed::Count { id, base: st.value(id).unwrap_or(0.0) },
            CueDef::Mem { .. } => Armed::Change { id, base: st.value(id) },
            CueDef::Cmp { .. } | CueDef::Stable(..) => Armed::Level { id },
            CueDef::Changed(_) => Armed::Edge { id, armed_at: st.input_frame() },
            CueDef::Frames(n) => Armed::Frames { until: st.input_frame() + n },
            CueDef::Seconds(s) => Armed::Seconds { until_us: st.guest_us() + (s * 1e6) as u64 },
            CueDef::After(a, b) => Armed::After { a: *a, b: Box::new(Armed::arm(st, *b)), then: None },
            CueDef::Any(v) => Armed::Any(v.iter().map(|c| Armed::arm(st, *c)).collect()),
            CueDef::All(v) => Armed::All(v.iter().map(|c| (Armed::arm(st, *c), false)).collect()),
        }
    }

    /// Whether the wait has fired, advancing any arm-relative state. `Some(v)` is the value
    /// the wait returns: the cue's value, or for `any` the 1-based index of the child that
    /// fired.
    pub fn poll(&mut self, st: &dyn CueState) -> Option<Option<f64>> {
        match self {
            Armed::Count { id, base } => {
                let v = st.value(*id).unwrap_or(0.0);
                (v > *base).then_some(Some(v))
            }
            Armed::Change { id, base } => {
                let v = st.value(*id);
                // Armed before the cue's first sample (declared and waited on in one go):
                // that sample is the baseline, not a change.
                if base.is_none() {
                    *base = v;
                    return None;
                }
                (v != *base).then_some(v)
            }
            Armed::Level { id } => match st.value(*id) {
                Some(v) if v != 0.0 => Some(Some(v)),
                _ => None,
            },
            Armed::Edge { id, armed_at } => {
                let src = match st.def(*id) {
                    CueDef::Changed(src) => *src,
                    _ => *id,
                };
                (st.input_frame() > *armed_at && st.changed_now(src)).then(|| st.value(src))
            }
            Armed::Frames { until } => (st.input_frame() >= *until).then_some(None),
            Armed::Seconds { until_us } => (st.guest_us() >= *until_us).then_some(None),
            Armed::After { a, b, then } => {
                if then.is_none() {
                    b.poll(st)?;
                    *then = Some(Box::new(Armed::arm(st, *a)));
                    // A level `a` that already holds fires on the same frame `b` did; an event
                    // `a` needs a firing after this point.
                }
                then.as_mut().unwrap().poll(st)
            }
            Armed::Any(children) => {
                for (i, c) in children.iter_mut().enumerate() {
                    if c.poll(st).is_some() {
                        return Some(Some((i + 1) as f64));
                    }
                }
                None
            }
            Armed::All(children) => {
                for (c, latched) in children.iter_mut() {
                    if !*latched && c.poll(st).is_some() {
                        *latched = true;
                    }
                }
                children.iter().all(|(_, l)| *l).then_some(Some(1.0))
            }
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    struct St {
        defs: Vec<CueDef>,
        vals: Vec<Option<f64>>,
        changed: Vec<bool>,
        frame: u64,
        us: u64,
    }

    impl CueState for St {
        fn def(&self, id: CueId) -> &CueDef {
            &self.defs[id]
        }
        fn value(&self, id: CueId) -> Option<f64> {
            match &self.defs[id] {
                CueDef::Cmp { src, op, value } => self.vals[*src].map(|v| op.eval(v, *value) as u8 as f64),
                _ => self.vals[id],
            }
        }
        fn changed_now(&self, id: CueId) -> bool {
            self.changed[id]
        }
        fn input_frame(&self) -> u64 {
            self.frame
        }
        fn guest_us(&self) -> u64 {
            self.us
        }
    }

    fn st(defs: Vec<CueDef>) -> St {
        let n = defs.len();
        St { defs, vals: vec![None; n], changed: vec![false; n], frame: 0, us: 0 }
    }

    #[test]
    fn a_call_wait_fires_only_on_calls_made_after_the_arm() {
        let mut s = st(vec![CueDef::Call { name: "f".into(), filter: None }]);
        s.vals[0] = Some(3.0);
        let mut w = Armed::arm(&s, 0);
        assert_eq!(w.poll(&s), None, "three calls before the wait do not count");
        s.vals[0] = Some(4.0);
        assert_eq!(w.poll(&s), Some(Some(4.0)));
    }

    #[test]
    fn a_condition_that_already_holds_fires_at_once() {
        let mut s = st(vec![
            CueDef::Mem { ty: MemTy::U32, base: 0, offs: vec![] },
            CueDef::Cmp { src: 0, op: CmpOp::Eq, value: 2.0 },
        ]);
        s.vals[0] = Some(2.0);
        assert!(Armed::arm(&s, 1).poll(&s).is_some());
    }

    #[test]
    fn after_waits_for_b_then_for_a_fresh_firing() {
        let mut s = st(vec![
            CueDef::Call { name: "a".into(), filter: None },
            CueDef::Call { name: "b".into(), filter: None },
            CueDef::After(0, 1),
        ]);
        s.vals[0] = Some(0.0);
        s.vals[1] = Some(0.0);
        let mut w = Armed::arm(&s, 2);
        s.vals[0] = Some(1.0);
        assert_eq!(w.poll(&s), None, "a before b does not count");
        s.vals[1] = Some(1.0);
        assert_eq!(w.poll(&s), None, "b arms a; a must fire again");
        s.vals[0] = Some(2.0);
        assert_eq!(w.poll(&s), Some(Some(2.0)));
    }

    #[test]
    fn any_returns_the_index_of_the_child_that_fired() {
        let mut s = st(vec![
            CueDef::Call { name: "a".into(), filter: None },
            CueDef::Frames(5),
            CueDef::Any(vec![0, 1]),
        ]);
        s.vals[0] = Some(0.0);
        let mut w = Armed::arm(&s, 2);
        s.frame = 5;
        assert_eq!(w.poll(&s), Some(Some(2.0)));
    }

    #[test]
    fn all_latches_each_child() {
        let mut s = st(vec![
            CueDef::Call { name: "a".into(), filter: None },
            CueDef::Call { name: "b".into(), filter: None },
            CueDef::All(vec![0, 1]),
        ]);
        s.vals[0] = Some(0.0);
        s.vals[1] = Some(0.0);
        let mut w = Armed::arm(&s, 2);
        s.vals[0] = Some(1.0);
        assert_eq!(w.poll(&s), None);
        s.vals[1] = Some(1.0);
        assert!(w.poll(&s).is_some(), "a latched on the earlier frame");
    }
}
