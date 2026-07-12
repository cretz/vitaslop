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
    /// The current value (0 or 1) of a condition flag. Used as the runtime
    /// carry-in of `adc`/`sbc` (and their flag computation).
    Flag(crate::abi::Flag),
    /// Count leading zeros of the inner value (ARM `clz`, wasm `i32.clz`).
    Clz(Box<Value>),
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

/// A VFP register reference: single-precision `S` (32-bit) or double `D` (64-bit).
/// `Q`/lane forms are not modeled yet (the cube uses only S arithmetic and whole-D
/// memory moves).
#[derive(Clone, Copy)]
pub enum VfpReg {
    S(u8),
    D(u8),
}

/// A NEON vector register operand: a 128-bit quad `Q` (0..15) or a 64-bit double
/// `D` (0..31). NEON data-processing operates on these; [`crate::emit`] maps each
/// onto a wasm `v128` (a `D` uses the low 64 bits, high 64 discarded on store).
#[derive(Clone, Copy)]
pub enum NeonReg {
    Q(u8),
    D(u8),
}

/// Element data type for a NEON operation: element size in bits (8/16/32/64) plus
/// how to interpret it. `signed` matters for the widening / min-max / abs-diff /
/// pairwise-long ops; `float` selects the f32 element (only `vabs`/`vneg`).
#[derive(Clone, Copy)]
pub struct NeonType {
    pub bits: u8,
    pub signed: bool,
    pub float: bool,
}

/// The same-length elementwise NEON binary operations ([`NeonStmt::Bin`]).
#[derive(Clone, Copy)]
pub enum NeonBin {
    Add,
    Sub,
    Mul,
    Max,
    Min,
    /// Absolute difference `|a - b|` (unsigned magnitude), `vabd`.
    Abd,
}

/// A NEON data-processing operation lowered to the IR. Every operand is a vector
/// register (except the `vmov` immediate). [`crate::emit::emit_neon`] turns each
/// into wasm 128-bit SIMD - see there for the exact instruction sequences and why
/// each maps cleanly (extend/extmul/extadd-pairwise cover the widening family).
pub enum NeonStmt {
    /// Same-length elementwise: `dst = a <op> b`.
    Bin { op: NeonBin, ty: NeonType, dst: NeonReg, a: NeonReg, b: NeonReg },
    /// Same-length multiply-accumulate: `dst = dst -/+ (a * b)` (`vmls`/`vmla`).
    MulAcc { ty: NeonType, dst: NeonReg, a: NeonReg, b: NeonReg, sub: bool },
    /// Pairwise add of adjacent elements of `a` then `b` into `dst` (`vpadd`).
    PairAdd { ty: NeonType, dst: NeonReg, a: NeonReg, b: NeonReg },
    /// Widening move: `dst(Q) = widen(a(D))` (`vmovl`). `ty` is the source element.
    Widen { ty: NeonType, dst: NeonReg, a: NeonReg },
    /// Widening add/sub: `dst(Q) = a -/+ widen(b(D))`. `wide` picks the wide form
    /// (`a` is a `Q` of already-wide elements, `vaddw`/`vsubw`) over the long form
    /// (`a` is a `D`, widened too, `vaddl`/`vsubl`). `ty` is the narrow element.
    WideAddSub { sub: bool, wide: bool, ty: NeonType, dst: NeonReg, a: NeonReg, b: NeonReg },
    /// Widening multiply[-accumulate]: `dst(Q) = [dst -/+] widen(a(D)) * widen(b(D))`
    /// (`vmull`/`vmlal`/`vmlsl`). `acc` enables the accumulate, `sub` its sign.
    WideMul { acc: bool, sub: bool, ty: NeonType, dst: NeonReg, a: NeonReg, b: NeonReg },
    /// Widening absolute difference[-accumulate]: `dst(Q) = [dst +] |widen(a) - widen(b)|`
    /// (`vabdl`/`vabal`). `acc` enables the accumulate. `ty` is the narrow element.
    WideAbd { acc: bool, ty: NeonType, dst: NeonReg, a: NeonReg, b: NeonReg },
    /// Pairwise-add-long[-accumulate]: `dst = [dst +] pairwise_widen_add(a)`
    /// (`vpaddl`/`vpadal`). `acc` enables the accumulate. `ty` is the narrow element.
    PairLong { acc: bool, ty: NeonType, dst: NeonReg, a: NeonReg },
    /// Elementwise unary: `dst = |a|` or `dst = -a` (`vabs`/`vneg`), integer or f32.
    Unary { neg: bool, ty: NeonType, dst: NeonReg, a: NeonReg },
    /// Immediate broadcast: set every `ty.bits`-bit element of `dst` to `imm`
    /// (`vmov.iN`). `imm` is the per-element value.
    MovImm { ty: NeonType, dst: NeonReg, imm: u32 },
}

/// Floating-point binary operators (single precision).
#[derive(Clone, Copy)]
pub enum FBinOp {
    Add,
    Sub,
    Mul,
    Div,
}

/// A VFP data-processing / move / compare / convert operation. All single
/// precision (f32); operands name S-register numbers unless noted. Kept separate
/// from the integer [`Value`] tree because these produce/consume f32 on the wasm
/// stack rather than i32.
pub enum VfpOp {
    /// `rd = rn <op> rm` (vadd/vsub/vmul/vdiv, f32).
    Bin32 { op: FBinOp, rd: u8, rn: u8, rm: u8 },
    /// Multiply-accumulate: `rd = rd +/- (rn * rm)` (vmla `sub=false`, vmls
    /// `sub=true`), non-fused (two roundings), f32.
    MulAcc32 { rd: u8, rn: u8, rm: u8, sub: bool },
    /// `rd = -rm` (vneg, f32).
    Neg32 { rd: u8, rm: u8 },
    /// `rd = |rm|` (vabs, f32).
    Abs32 { rd: u8, rm: u8 },
    /// `rd = sqrt(rm)` (vsqrt, f32).
    Sqrt32 { rd: u8, rm: u8 },
    /// `rd = rm` raw bit copy (vmov S,S).
    Mov32 { rd: u8, rm: u8 },
    /// Copy the raw 32 bits of single-precision register S`s` into core register
    /// `rt` (`vmov Rt, Sn`).
    ScalarToCore { rt: u8, s: u8 },
    /// Copy the 32 bits of core register `rt` into single-precision register S`s`
    /// (`vmov Sn, Rt`).
    CoreToScalar { s: u8, rt: u8 },
    /// Set single register `s` to a 32-bit immediate bit pattern (the VFP
    /// `vmov.f32 sd, #imm`).
    SetImmS { s: u8, bits: u32 },
    /// Set double register `d` to a 64-bit immediate (`lo` low word, `hi` high
    /// word) - the NEON `vmov.iN` modified-immediate (per constituent D register)
    /// and the VFP `vmov.f64 dd, #imm`.
    SetImmD { d: u8, lo: u32, hi: u32 },
    /// Compare `rn` against `rm` (or `+0.0` when `rm` is `None`), setting the FP
    /// condition flags (vcmp/vcmpe).
    Cmp32 { rn: u8, rm: Option<u8> },
    /// Copy FP condition flags into the integer NZCV flags (`vmrs APSR_nzcv`).
    MrsNzcv,
    /// Convert f32 in `rm` to a 32-bit integer in `rd` (round toward zero,
    /// saturating), signed or unsigned (vcvt.s32/u32.f32).
    CvtToInt { rd: u8, rm: u8, signed: bool },
    /// Convert a 32-bit integer in `rm` to f32 in `rd` (vcvt.f32.s32/u32).
    CvtFromInt { rd: u8, rm: u8, signed: bool },
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
    FlagsAdd { a: Value, b: Value, cin: Value },
    /// Set N,Z from `value` (logical result); set C to bit 0 of `carry` if
    /// present (the shifter carry-out); leave V unchanged.
    FlagsLogic { value: Value, carry: Option<Value> },
    /// Service an ARM `svc #imm` through the host `svc` import.
    Svc(u32),
    /// Service a Vita NID call through the host `import` import, by dense index.
    Import(u32),
    /// A 64-bit widening multiply: `{rdhi:rdlo} = rn * rm`, unsigned or signed
    /// (ARM `umull`/`smull`). The two 32-bit operands are extended to 64 bits
    /// (per `signed`), multiplied, and the product's low/high halves written to
    /// `rdlo`/`rdhi`.
    MulLong { rdlo: u8, rdhi: u8, rn: Value, rm: Value, signed: bool },
    /// A direct guest call (`bl`/`blx` to translated code): call the callee's
    /// wasm function, which returns here. `lr` is set by a preceding `SetReg`.
    Call { target: u32 },
    /// Execute the inner statements only if `cond` holds (ARM predication / an
    /// `IT` block body).
    Guard(ConditionCode, Vec<Stmt>),
    /// A VFP data-processing / move / compare / convert op.
    Vfp(VfpOp),
    /// One VFP register <-> memory transfer (a single lane of vldr/vstr/vldm/
    /// vstm/vpush/vpop/vld1/vst1). `load` picks direction; `reg`'s width picks the
    /// access size (S = 4 bytes, D = 8 bytes).
    VfpMem { reg: VfpReg, addr: Value, load: bool },
    /// A NEON (Advanced SIMD) data-processing operation.
    Neon(NeonStmt),
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
