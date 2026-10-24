//! The request/reply wire between the prelude and the engine: one JSON object per yield.
//!
//! The shapes are the prelude's (`prelude.lua`); this module is their only Rust reader, so a
//! new API call is one prelude function plus one [`Request`] arm.
use serde_json::{json, Map, Value};

use crate::cue::{CueDef, CueId, MemTy};

/// A comparison a condition cue applies.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub enum CmpOp {
    Eq,
    Ne,
    Lt,
    Le,
    Gt,
    Ge,
}

impl CmpOp {
    pub fn parse(s: &str) -> Option<CmpOp> {
        Some(match s {
            "==" => CmpOp::Eq,
            "!=" => CmpOp::Ne,
            "<" => CmpOp::Lt,
            "<=" => CmpOp::Le,
            ">" => CmpOp::Gt,
            ">=" => CmpOp::Ge,
            _ => return None,
        })
    }

    pub fn symbol(self) -> &'static str {
        match self {
            CmpOp::Eq => "==",
            CmpOp::Ne => "!=",
            CmpOp::Lt => "<",
            CmpOp::Le => "<=",
            CmpOp::Gt => ">",
            CmpOp::Ge => ">=",
        }
    }

    pub fn eval(self, a: f64, b: f64) -> bool {
        match self {
            CmpOp::Eq => a == b,
            CmpOp::Ne => a != b,
            CmpOp::Lt => a < b,
            CmpOp::Le => a <= b,
            CmpOp::Gt => a > b,
            CmpOp::Ge => a >= b,
        }
    }
}

/// One request from the playbook.
#[derive(Debug, PartialEq)]
pub enum Request {
    Cue(CueDef),
    Value(CueId),
    Trace { cue: CueId, label: String },
    Phase { name: String, timeout: Option<f64> },
    EndPhase { name: String },
    Wait { cue: CueId, timeout: Option<f64>, soft: bool },
    Press(Vec<String>),
    Release(Vec<String>),
    Hold { buttons: Vec<String>, frames: u64 },
    Idle(u64),
    Stick { right: bool, x: u8, y: u8 },
    /// A finger down at `at` (or lifted, `None`) on the front panel, or the rear one.
    Touch { back: bool, at: Option<(u16, u16)> },
    Shot(String),
    Log(String),
    Split(String),
    Check { cue: CueId, msg: Option<String> },
    Now,
    /// The playbook returned.
    Done,
    /// The playbook raised (message carries the location and a traceback).
    Error(String),
}

fn obj(v: &Value) -> Result<&Map<String, Value>, String> {
    v.as_object().ok_or_else(|| format!("a request must be an object, got {v}"))
}

fn str_of(o: &Map<String, Value>, k: &str) -> Result<String, String> {
    o.get(k).and_then(|v| v.as_str()).map(str::to_string).ok_or_else(|| format!("`{k}` must be a string"))
}

fn num_of(o: &Map<String, Value>, k: &str) -> Result<f64, String> {
    o.get(k).and_then(|v| v.as_f64()).ok_or_else(|| format!("`{k}` must be a number"))
}

fn opt_num(o: &Map<String, Value>, k: &str) -> Result<Option<f64>, String> {
    match o.get(k) {
        None | Some(Value::Null) => Ok(None),
        Some(v) => v.as_f64().map(Some).ok_or_else(|| format!("`{k}` must be a number")),
    }
}

fn frames_of(o: &Map<String, Value>, k: &str) -> Result<u64, String> {
    let n = num_of(o, k)?;
    if n < 0.0 || n.fract() != 0.0 {
        return Err(format!("`{k}` must be a whole number of input frames, got {n}"));
    }
    Ok(n as u64)
}

fn cue_of(o: &Map<String, Value>, k: &str) -> Result<CueId, String> {
    o.get(k)
        .and_then(|v| v.as_u64())
        .map(|v| v as CueId)
        .ok_or_else(|| format!("`{k}` must be a cue"))
}

fn names_of(o: &Map<String, Value>, k: &str) -> Result<Vec<String>, String> {
    match o.get(k) {
        None | Some(Value::Null) => Ok(vec![]),
        Some(Value::Object(m)) if m.is_empty() => Ok(vec![]),
        Some(Value::Array(a)) => a
            .iter()
            .map(|v| v.as_str().map(str::to_string).ok_or_else(|| format!("`{k}` must be names")))
            .collect(),
        Some(v) => Err(format!("`{k}` must be a list, got {v}")),
    }
}

fn cues_of(o: &Map<String, Value>, k: &str) -> Result<Vec<CueId>, String> {
    match o.get(k) {
        Some(Value::Array(a)) if !a.is_empty() => a
            .iter()
            .map(|v| v.as_u64().map(|v| v as CueId).ok_or_else(|| format!("`{k}` must be cues")))
            .collect(),
        _ => Err(format!("`{k}` must be one or more cues")),
    }
}

fn axis(o: &Map<String, Value>, k: &str) -> Result<u8, String> {
    let n = num_of(o, k)?;
    if !(0.0..=255.0).contains(&n) || n.fract() != 0.0 {
        return Err(format!("stick `{k}` must be 0..255, got {n}"));
    }
    Ok(n as u8)
}

fn parse_cue(o: &Map<String, Value>) -> Result<CueDef, String> {
    let kind = str_of(o, "kind")?;
    Ok(match kind.as_str() {
        "call" => {
            let name = str_of(o, "name")?;
            let filter = match (o.get("arg").and_then(|v| v.as_u64()), o.get("has").and_then(|v| v.as_str())) {
                (Some(a), Some(h)) if a <= 3 => Some((a as u8, h.to_string())),
                (None, None) => None,
                (Some(a), Some(_)) => return Err(format!("cue.call arg must be 0..3, got {a}")),
                _ => return Err("cue.call needs both `arg` and `has`, or neither".into()),
            };
            CueDef::Call { name, filter }
        }
        "mem" => {
            let ty = str_of(o, "ty")?;
            let ty = MemTy::parse(&ty).ok_or_else(|| format!("bad memory type {ty:?}"))?;
            let base = num_of(o, "base")?;
            if !(0.0..=u32::MAX as f64).contains(&base) {
                return Err(format!("address {base} is not a 32-bit guest address"));
            }
            let offs = match o.get("offs") {
                Some(Value::Array(a)) => a
                    .iter()
                    .map(|v| v.as_u64().map(|v| v as u32).ok_or("ptr offsets must be numbers".to_string()))
                    .collect::<Result<_, _>>()?,
                _ => vec![],
            };
            CueDef::Mem { ty, base: base as u32, offs }
        }
        "color" => {
            let dim = |k: &str, max: f64| -> Result<u32, String> {
                let n = num_of(o, k)?;
                if !(0.0..=max).contains(&n) || n.fract() != 0.0 {
                    return Err(format!("cue.color `{k}` must be a whole number 0..{max}, got {n}"));
                }
                Ok(n as u32)
            };
            let (x, y) = (dim("x", 959.0)?, dim("y", 543.0)?);
            let (w, h) = (dim("w", 960.0)?.max(1), dim("h", 544.0)?.max(1));
            if x + w > 960 || y + h > 544 {
                return Err(format!("cue.color {w}x{h} at {x},{y} runs off the 960x544 frame"));
            }
            CueDef::Color { x, y, w, h }
        }
        "pixels" | "locate" => {
            let dim = |k: &str, max: f64| -> Result<u32, String> {
                let n = num_of(o, k)?;
                if !(0.0..=max).contains(&n) || n.fract() != 0.0 {
                    return Err(format!("cue.pixels `{k}` must be a whole number 0..{max}, got {n}"));
                }
                Ok(n as u32)
            };
            let (x, y) = (dim("x", 959.0)?, dim("y", 543.0)?);
            let (w, h) = (dim("w", 960.0)?.max(1), dim("h", 544.0)?.max(1));
            if x + w > 960 || y + h > 544 {
                return Err(format!("cue.pixels {w}x{h} at {x},{y} runs off the 960x544 frame"));
            }
            let rgb = match o.get("rgb") {
                Some(Value::Array(a)) if a.len() == 3 => {
                    let mut c = [0u8; 3];
                    for (i, v) in a.iter().enumerate() {
                        c[i] = v.as_u64().filter(|v| *v <= 255).ok_or("cue.pixels `rgb` is three numbers 0..255")? as u8;
                    }
                    c
                }
                _ => return Err("cue.pixels needs `rgb = { r, g, b }`".into()),
            };
            let tol = dim("tol", 255.0)? as u8;
            if kind == "locate" {
                CueDef::Locate { x, y, w, h, rgb, tol }
            } else {
                CueDef::Pixels { x, y, w, h, rgb, tol }
            }
        }
        "frames" => CueDef::Frames(frames_of(o, "n")?),
        "seconds" => CueDef::Seconds(num_of(o, "s")?),
        "cmp" => {
            let op = str_of(o, "op")?;
            CueDef::Cmp {
                src: cue_of(o, "src")?,
                op: CmpOp::parse(&op).ok_or_else(|| format!("bad comparison {op:?}"))?,
                value: num_of(o, "value")?,
            }
        }
        "changed" => CueDef::Changed(cue_of(o, "src")?),
        "stable" => CueDef::Stable(cue_of(o, "src")?, frames_of(o, "n")?),
        "after" => CueDef::After(cue_of(o, "a")?, cue_of(o, "b")?),
        "any" => CueDef::Any(cues_of(o, "of")?),
        "all" => CueDef::All(cues_of(o, "of")?),
        other => return Err(format!("unknown cue kind {other:?}")),
    })
}

/// Parse one request. An `Err` is a bug in the prelude or a malformed argument, and is sent
/// back to the playbook as an error at the line that made the call.
pub fn parse(text: &str) -> Result<Request, String> {
    let v: Value = serde_json::from_str(text).map_err(|e| format!("unreadable request {text:?}: {e}"))?;
    let o = obj(&v)?;
    let op = str_of(o, "op")?;
    Ok(match op.as_str() {
        "cue" => Request::Cue(parse_cue(obj(o.get("def").unwrap_or(&Value::Null))?)?),
        "value" => Request::Value(cue_of(o, "cue")?),
        "trace" => Request::Trace { cue: cue_of(o, "cue")?, label: str_of(o, "label")? },
        "phase" => Request::Phase { name: str_of(o, "name")?, timeout: opt_num(o, "timeout")? },
        "end_phase" => Request::EndPhase { name: str_of(o, "name")? },
        "wait" => Request::Wait {
            cue: cue_of(o, "cue")?,
            timeout: opt_num(o, "timeout")?,
            soft: o.get("soft").and_then(|v| v.as_bool()).unwrap_or(false),
        },
        "press" => Request::Press(names_of(o, "buttons")?),
        "release" => Request::Release(names_of(o, "buttons")?),
        "hold" => Request::Hold { buttons: names_of(o, "buttons")?, frames: frames_of(o, "frames")? },
        "idle" => Request::Idle(frames_of(o, "frames")?),
        "stick" => {
            let which = str_of(o, "which")?;
            let right = match which.as_str() {
                "left" => false,
                "right" => true,
                _ => return Err(format!("stick must be \"left\" or \"right\", got {which:?}")),
            };
            Request::Stick { right, x: axis(o, "x")?, y: axis(o, "y")? }
        }
        "touch" => {
            let back = match o.get("panel").and_then(|v| v.as_str()) {
                None | Some("front") => false,
                Some("back") => true,
                Some(p) => return Err(format!("touch panel must be \"front\" or \"back\", got {p:?}")),
            };
            let at = match (opt_num(o, "x")?, opt_num(o, "y")?) {
                (Some(x), Some(y)) => Some((x as u16, y as u16)),
                _ => None,
            };
            Request::Touch { back, at }
        }
        "shot" => Request::Shot(str_of(o, "name")?),
        "log" => Request::Log(str_of(o, "text")?),
        "split" => Request::Split(str_of(o, "name")?),
        "check" => Request::Check { cue: cue_of(o, "cue")?, msg: o.get("msg").and_then(|v| v.as_str()).map(str::to_string) },
        "now" => Request::Now,
        "done" => Request::Done,
        "error" => Request::Error(str_of(o, "message")?),
        other => return Err(format!("unknown request {other:?}")),
    })
}

/// The empty reply: the request needed no answer.
pub fn ok() -> String {
    "{}".into()
}

/// A value reply (`{v = ...}`; nil when there is none).
pub fn value(v: Option<f64>) -> String {
    match v {
        Some(v) if v.fract() == 0.0 && v.abs() < 9.0e15 => json!({ "v": v as i64 }).to_string(),
        Some(v) => json!({ "v": v }).to_string(),
        None => "{}".into(),
    }
}

/// An error raised in the playbook at the line that made the request.
pub fn error(msg: &str) -> String {
    json!({ "error": msg }).to_string()
}

pub fn id(id: CueId) -> String {
    json!({ "id": id }).to_string()
}

pub fn now(input: u64, display: u64, seconds: f64) -> String {
    json!({ "input": input, "display": display, "seconds": seconds }).to_string()
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn a_pixel_box_must_lie_on_the_frame() {
        let ok = parse(r#"{"op":"cue","def":{"kind":"pixels","x":10,"y":20,"w":4,"h":5,"rgb":[255,0,1],"tol":9}}"#);
        assert_eq!(ok, Ok(Request::Cue(CueDef::Pixels { x: 10, y: 20, w: 4, h: 5, rgb: [255, 0, 1], tol: 9 })));
        let at = parse(r#"{"op":"cue","def":{"kind":"locate","x":0,"y":0,"w":960,"h":544,"rgb":[1,2,3],"tol":0}}"#);
        assert_eq!(at, Ok(Request::Cue(CueDef::Locate { x: 0, y: 0, w: 960, h: 544, rgb: [1, 2, 3], tol: 0 })));
        assert!(parse(r#"{"op":"cue","def":{"kind":"pixels","x":950,"y":0,"w":20,"h":1,"rgb":[0,0,0],"tol":0}}"#).is_err());
        assert!(parse(r#"{"op":"cue","def":{"kind":"pixels","x":0,"y":0,"w":1,"h":1,"tol":0}}"#).is_err(), "no rgb");
    }

    #[test]
    fn a_colour_box_must_lie_on_the_frame() {
        let ok = parse(r#"{"op":"cue","def":{"kind":"color","x":580,"y":410,"w":80,"h":40}}"#);
        assert_eq!(ok, Ok(Request::Cue(CueDef::Color { x: 580, y: 410, w: 80, h: 40 })));
        assert!(parse(r#"{"op":"cue","def":{"kind":"color","x":950,"y":0,"w":20,"h":1}}"#).is_err());
    }

    #[test]
    fn requests_parse_from_the_preludes_shapes() {
        assert_eq!(
            parse(r#"{"op":"cue","def":{"kind":"call","name":"sceIoOpen","arg":0,"has":"stage"}}"#),
            Ok(Request::Cue(CueDef::Call { name: "sceIoOpen".into(), filter: Some((0, "stage".into())) }))
        );
        assert_eq!(
            parse(r#"{"op":"cue","def":{"kind":"mem","ty":"u32","base":2172176264,"offs":[60]}}"#),
            Ok(Request::Cue(CueDef::Mem { ty: MemTy::U32, base: 0x8178_c788, offs: vec![60] }))
        );
        // An empty Lua table encodes as `{}`: `release()` with no buttons.
        assert_eq!(parse(r#"{"op":"release","buttons":{}}"#), Ok(Request::Release(vec![])));
        assert_eq!(parse(r#"{"op":"hold","buttons":["cross"],"frames":20}"#).unwrap(), Request::Hold { buttons: vec!["cross".into()], frames: 20 });
        assert!(parse(r#"{"op":"idle","frames":1.5}"#).is_err(), "input frames are whole");
        assert!(parse(r#"{"op":"stick","which":"up","x":1,"y":1}"#).is_err());
    }

    #[test]
    fn whole_values_reply_as_integers() {
        assert_eq!(value(Some(3.0)), r#"{"v":3}"#);
        assert_eq!(value(Some(0.5)), r#"{"v":0.5}"#);
        assert_eq!(value(None), "{}");
    }
}
