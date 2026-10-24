//! Reading typed values out of guest memory: the [`GuestRead`] seam both engines implement, and
//! the named memory WATCH a live session samples through it.

/// Reads guest memory. Native reads through the wasmtime store, the browser through its shared
/// `SharedArrayBuffer` view; a sample is the same question either way.
pub trait GuestRead {
    /// Fill `out` from guest address `addr`. False if the range is not mapped, in which case a
    /// watch reports `oob` rather than a plausible zero.
    fn read_into(&self, addr: u32, out: &mut [u8]) -> bool;
}

/// A watched value's storage type, widened to `f64` so integer and float values compare
/// uniformly.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub enum ValType {
    U8,
    U16,
    U32,
    I32,
    F32,
}

impl ValType {
    /// Parse a type keyword (`u8|u16|u32|i32|f32`). `None` if unrecognized.
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

    /// The keyword for this type (the inverse of [`ValType::parse`]).
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
        let word = || bytes.get(..4).map(|b| [b[0], b[1], b[2], b[3]]);
        Some(match self {
            ValType::U8 => *bytes.first()? as f64,
            ValType::U16 => u16::from_le_bytes([*bytes.first()?, *bytes.get(1)?]) as f64,
            ValType::U32 => u32::from_le_bytes(word()?) as f64,
            ValType::I32 => i32::from_le_bytes(word()?) as f64,
            ValType::F32 => f32::from_le_bytes(word()?) as f64,
        })
    }
}

/// A comparison operator over a sampled value.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub enum CmpOp {
    Eq,
    Ne,
    Lt,
    Le,
    Gt,
    Ge,
    /// Approximately equal, within a tolerance.
    Approx,
}

impl CmpOp {
    /// Parse `== = != < <= > >= ~`. `None` if unrecognized.
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

    /// The keyword for this operator (the inverse of [`CmpOp::parse`]).
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

/// A named live memory value: `ty` at `addr`, or - with `deref` - at the 32-bit pointer stored at
/// `addr` plus that offset. A value in a pooled or allocated block moves between runs that load
/// differently - a fighting title's health block sits in a different pool slot on the desktop and
/// in the browser, because the two load the stage at different speeds - while the pointer to it
/// does not.
#[derive(Clone, Debug, PartialEq, Eq)]
pub struct WatchDecl {
    pub name: String,
    pub ty: ValType,
    /// The value's address, or - with `deref` - the address of the pointer to it.
    pub addr: u32,
    /// `Some(offset)`: `addr` holds a pointer; the value is at `pointer + offset`.
    pub deref: Option<u32>,
}

impl WatchDecl {
    /// The address spelled `0x...` or `*0x...+0x...`.
    pub fn addr_text(&self) -> String {
        match self.deref {
            None => format!("{:#x}", self.addr),
            Some(0) => format!("*{:#x}", self.addr),
            Some(o) => format!("*{:#x}+{o:#x}", self.addr),
        }
    }
}

/// Sample one watch from guest memory. `None` if an address on the way is not mapped.
pub fn sample_watch(mem: &impl GuestRead, w: &WatchDecl) -> Option<f64> {
    let mut buf = [0u8; 4];
    let addr = match w.deref {
        None => w.addr,
        Some(off) => {
            if !mem.read_into(w.addr, &mut buf) {
                return None;
            }
            u32::from_le_bytes(buf).wrapping_add(off)
        }
    };
    let width = w.ty.width();
    if !mem.read_into(addr, &mut buf[..width]) {
        return None;
    }
    w.ty.decode(&buf[..width])
}

/// Format an `f64` compactly: integers without a trailing `.0`.
pub fn format_f64(x: f64) -> String {
    if x.fract() == 0.0 && x.abs() < 1e15 {
        format!("{}", x as i64)
    } else {
        format!("{x}")
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    struct Mem(Vec<u8>);
    impl GuestRead for Mem {
        fn read_into(&self, addr: u32, out: &mut [u8]) -> bool {
            let a = addr as usize;
            match self.0.get(a..a + out.len()) {
                Some(src) => {
                    out.copy_from_slice(src);
                    true
                }
                None => false,
            }
        }
    }

    #[test]
    fn a_watch_reads_its_type_directly_or_through_a_pointer() {
        let mut bytes = vec![0u8; 64];
        bytes[8..12].copy_from_slice(&(-5i32).to_le_bytes());
        bytes[16..20].copy_from_slice(&4u32.to_le_bytes()); // a pointer to 4...
        bytes[12..16].copy_from_slice(&1.5f32.to_le_bytes()); // ...+8 = 12
        let mem = Mem(bytes);
        let direct = WatchDecl {
            name: "a".into(),
            ty: ValType::I32,
            addr: 8,
            deref: None,
        };
        assert_eq!(sample_watch(&mem, &direct), Some(-5.0));
        let through = WatchDecl {
            name: "b".into(),
            ty: ValType::F32,
            addr: 16,
            deref: Some(8),
        };
        assert_eq!(sample_watch(&mem, &through), Some(1.5));
        assert_eq!(through.addr_text(), "*0x10+0x8");
        let oob = WatchDecl {
            name: "c".into(),
            ty: ValType::U32,
            addr: 62,
            deref: None,
        };
        assert_eq!(
            sample_watch(&mem, &oob),
            None,
            "unmapped reads as oob, not zero"
        );
    }

    #[test]
    fn comparisons_and_formatting() {
        assert!(CmpOp::parse("~").unwrap().eval(60.4, 60.0, 0.5));
        assert!(!CmpOp::parse(">=").unwrap().eval(1.0, 2.0, 0.0));
        assert_eq!(CmpOp::parse("<=").map(CmpOp::keyword), Some("<="));
        assert_eq!(
            (format_f64(3.0), format_f64(2.5)),
            ("3".to_string(), "2.5".to_string())
        );
    }
}
