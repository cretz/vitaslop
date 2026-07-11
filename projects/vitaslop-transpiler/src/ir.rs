//! The transpiler's intermediate representation: a per-function control-flow
//! graph of basic blocks, each a list of side-effect statements over a small
//! value-expression tree. Decode/lowering ([`crate::lower`]) produces it; wasm
//! emission ([`crate::emit`]) consumes it. Keeping a real IR here (rather than a
//! 1:1 decode-to-wasm emitter) is what lets optimizations - const folding, lazy
//! flags, better register allocation, a relooper - grow later without touching
//! the front or back end.

pub use yaxpeax_arm::armv7::ConditionCode;

/// Width of a memory access.
#[derive(Clone, Copy, PartialEq, Eq)]
pub enum MemSize {
    Byte,
    Half,
    Word,
}

/// A pure, side-effect-free value expression. Emitted post-order onto the wasm
/// stack. Loads are here (not statements) because ARM addressing folds a load
/// into an operand; stores, which have an effect, are statements.
#[derive(Clone)]
pub enum Value {
    /// A 32-bit constant.
    Imm(u32),
    /// The current value of guest register `r` (r0..r15).
    Reg(u8),
    /// A memory load of `size` at `addr`, zero- or sign-extended to 32 bits.
    Load {
        addr: Box<Value>,
        size: MemSize,
        signed: bool,
    },
    /// Bitwise NOT.
    Not(Box<Value>),
    /// A binary operation.
    Bin(BinOp, Box<Value>, Box<Value>),
}

/// Binary operators over 32-bit values. Shifts are the logical/arithmetic wasm
/// forms; ARM shift-amount masking is applied during lowering when needed.
#[derive(Clone, Copy)]
pub enum BinOp {
    Add,
    Sub,
    And,
    Or,
    Xor,
    Shl,
    Lsr,
    Asr,
    Mul,
}

/// One side-effecting statement within a basic block.
pub enum Stmt {
    /// `r[reg] = value`.
    SetReg(u8, Value),
    /// Store the low `size` bytes of `data` to `addr`.
    Store {
        addr: Value,
        data: Value,
        size: MemSize,
    },
    /// Set N,Z,C,V for the result of `a + b + cin`. Subtraction and compare pass
    /// `b` already bit-inverted with `cin = 1` (ARM computes `a - b` as
    /// `a + ~b + 1`), so this one primitive covers adds/subs/cmp/cmn/adc/sbc.
    FlagsAdd { a: Value, b: Value, cin: u32 },
    /// Set N,Z from `value` (logical result); set C to bit 0 of `carry` if
    /// present (the shifter carry-out); leave V unchanged.
    FlagsLogic { value: Value, carry: Option<Value> },
    /// Service an ARM `svc #imm` through the host `svc` import.
    Svc(u32),
    /// Service a Vita NID call through the host `import` import, by dense index.
    Import(u32),
    /// A direct guest call (`bl`/`blx` to translated code): call the callee's
    /// wasm function, which returns here. `lr` is set by a preceding `SetReg`.
    Call { target: u32 },
    /// Execute the inner statements only if `cond` holds (ARM predication / an
    /// `IT` block body).
    Guard(ConditionCode, Vec<Stmt>),
}

/// How a basic block hands control to the next. The not-taken side of a branch
/// and an explicit fall-through both continue into the textually-next block
/// (blocks are emitted in ascending address order, so that successor is
/// adjacent and needs no wasm branch).
pub enum Term {
    /// Continue into the next block (this block ended only because its successor
    /// is a branch target).
    Fallthrough,
    /// Unconditional direct branch to another block in this function.
    Jump(u32),
    /// Conditional branch: if `cond`, go to `taken`; else fall through.
    Branch { cond: ConditionCode, taken: u32 },
    /// `cbz`/`cbnz`: branch to `taken` if register `reg` is zero (`nonzero`
    /// false) or non-zero (`nonzero` true); else fall through. These test a
    /// register directly, not the condition flags.
    BranchZero { reg: u8, nonzero: bool, taken: u32 },
    /// Return to the caller (`bx lr`, `pop {..,pc}`, ...).
    Return,
    /// Stop running this function without returning: an infinite self-loop
    /// (`b .`), a statically-known noreturn `svc`, or undecodable tail.
    Halt,
}

/// A basic block: its start address, its statements, and how it terminates.
pub struct Block {
    pub addr: u32,
    pub stmts: Vec<Stmt>,
    pub term: Term,
}

/// A discovered guest function: one wasm function. Blocks are sorted ascending by
/// address; `blocks[0]` is the entry block.
pub struct Func {
    pub addr: u32,
    /// Decode mode this function was discovered in. Carried for the future
    /// per-function `blx` mode switch; the emitter is mode-agnostic today.
    #[allow(dead_code)]
    pub thumb: bool,
    pub blocks: Vec<Block>,
}

impl Func {
    /// Index of the block starting at `addr`, if any (for branch lowering).
    pub fn block_index(&self, addr: u32) -> Option<usize> {
        self.blocks.iter().position(|b| b.addr == addr)
    }
}
