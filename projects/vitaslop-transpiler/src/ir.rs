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
    /// Rotate right (ARM `ror`): the amount is taken modulo 32, matching both
    /// wasm `i32.rotr` and ARM's register-rotate masking.
    Ror,
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
    /// Duplicate the low `ty.bits` bits of core register `rt` into every element
    /// of `dst` (`vdup.N Qd/Dd, Rt`).
    DupCore { ty: NeonType, dst: NeonReg, rt: u8 },
    /// Whole-register bitwise logical op (the 3-same logical family): `vand`,
    /// `vorr`, `veor`, `vbic`, `vorn`, and the insert/select forms `vbsl`/`vbit`/
    /// `vbif` (which also read `dst`). Element-size agnostic.
    Bitwise { op: NeonBitwise, dst: NeonReg, a: NeonReg, b: NeonReg },
    /// NEON single-element load/store to/from one lane of a D register, or a
    /// broadcast load to all its lanes (`vld1`/`vst1` single-element forms). `esize`
    /// is the element size in bits (8/16/32). This is the element-wise, deinterleave-
    /// free 1-structure case: a lane load reads `esize` bits and inserts them into
    /// lane `lane`, leaving the rest of `d` intact; a broadcast load replicates the
    /// read element across every lane; a lane store writes one lane's bits out.
    ElemMem { d: u8, esize: u8, lane: ElemLane, addr: Value, load: bool },
}

/// Which lane(s) a [`NeonStmt::ElemMem`] transfer touches.
#[derive(Clone, Copy)]
pub enum ElemLane {
    /// A single lane, by index (`{dN[i]}`).
    One(u8),
    /// Every lane (a broadcast load, `{dN[]}`); store is not a valid form.
    All,
}

/// The NEON bitwise logical operations ([`NeonStmt::Bitwise`]).
#[derive(Clone, Copy)]
pub enum NeonBitwise {
    And,
    Or,
    Xor,
    /// `vbic`: `a AND NOT b`.
    Bic,
    /// `vorn`: `a OR NOT b`.
    Orn,
    /// `vbsl`: bitwise select with `dst` as the mask.
    Bsl,
    /// `vbit`: insert `a` where `b` is set.
    Bit,
    /// `vbif`: insert `a` where `b` is clear.
    Bif,
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
    /// Multiply-accumulate: `rd = (-rd if neg else rd) +/- (rn * rm)`, non-fused
    /// (two roundings), f32. Covers vmla (`neg=false,sub=false`), vmls
    /// (`neg=false,sub=true`), vnmls (`neg=true,sub=false`), vnmla
    /// (`neg=true,sub=true`).
    MulAcc32 { rd: u8, rn: u8, rm: u8, sub: bool, neg: bool },
    /// `rd = -(rn * rm)` (vnmul, f32).
    NegMul32 { rd: u8, rn: u8, rm: u8 },
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
    /// Read the FPSCR into core register `rt` (`vmrs Rt, fpscr`): reconstruct the
    /// NZCV bits [31:28] from the FP flags; all other bits read as zero.
    MrsFpscr { rt: u8 },
    /// Convert f32 in `rm` to a 32-bit integer in `rd` (round toward zero,
    /// saturating), signed or unsigned (vcvt.s32/u32.f32).
    CvtToInt { rd: u8, rm: u8, signed: bool },
    /// Convert a 32-bit integer in `rm` to f32 in `rd` (vcvt.f32.s32/u32).
    CvtFromInt { rd: u8, rm: u8, signed: bool },

    // --- Double precision (f64). Operands name D-register numbers. ---
    /// `rd = rn <op> rm` (vadd/vsub/vmul/vdiv, f64).
    Bin64 { op: FBinOp, rd: u8, rn: u8, rm: u8 },
    /// `rd = (-rd if neg else rd) +/- (rn * rm)`, f64 (vmla/vmls/vnmls/vnmla).
    MulAcc64 { rd: u8, rn: u8, rm: u8, sub: bool, neg: bool },
    /// `rd = -(rn * rm)` (vnmul, f64).
    NegMul64 { rd: u8, rn: u8, rm: u8 },
    /// `rd = -rm` (vneg, f64).
    Neg64 { rd: u8, rm: u8 },
    /// `rd = |rm|` (vabs, f64).
    Abs64 { rd: u8, rm: u8 },
    /// `rd = sqrt(rm)` (vsqrt, f64).
    Sqrt64 { rd: u8, rm: u8 },
    /// `rd = rm` raw 64-bit copy (vmov D,D).
    Mov64 { rd: u8, rm: u8 },
    /// Compare `rn` against `rm` (or `+0.0` when `None`), setting the FP flags
    /// (vcmp/vcmpe, f64).
    Cmp64 { rn: u8, rm: Option<u8> },
    /// Convert a 32-bit integer in S`s` to f64 in D`d` (vcvt.f64.s32/u32).
    CvtF64FromInt { d: u8, s: u8, signed: bool },
    /// Convert f64 in D`d` to a 32-bit integer in S`s`, round toward zero,
    /// saturating (vcvt.s32/u32.f64).
    CvtIntFromF64 { s: u8, d: u8, signed: bool },
    /// Widen f32 in S`s` to f64 in D`d` (vcvt.f64.f32).
    CvtF64FromF32 { d: u8, s: u8 },
    /// Narrow f64 in D`d` to f32 in S`s` (vcvt.f32.f64).
    CvtF32FromF64 { s: u8, d: u8 },
    /// Widen the IEEE half-precision (f16) value in a 16-bit half of S`sm` to f32 in
    /// S`sd` (`vcvtb`/`vcvtt.f32.f16`). `top` selects the top half (`vcvtt`) over the
    /// bottom (`vcvtb`). Emitted as the branchless bit/float conversion.
    CvtF32FromHalf { sd: u8, sm: u8, top: bool },
    /// `vmov Rt, Rt2, Dm`: copy D`d`'s low 32 bits to `rt`, high 32 to `rt2`.
    DoubleToCore { rt: u8, rt2: u8, d: u8 },
    /// `vmov Dm, Rt, Rt2`: assemble D`d` from `rt` (low) and `rt2` (high).
    CoreToDouble { d: u8, rt: u8, rt2: u8 },
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
    /// Reverse the bit order of `rm` into `rd` (ARM `rbit`). No single wasm
    /// primitive; emitted as the 5-step swap network over a scratch local.
    Rbit { rd: u8, rm: Value },
    /// A 64-bit widening multiply: `{rdhi:rdlo} = rn * rm`, unsigned or signed
    /// (ARM `umull`/`smull`). The two 32-bit operands are extended to 64 bits
    /// (per `signed`), multiplied, and the product's low/high halves written to
    /// `rdlo`/`rdhi`.
    MulLong { rdlo: u8, rdhi: u8, rn: Value, rm: Value, signed: bool },
    /// A direct guest call (`bl`/`blx` to translated code): call the callee's
    /// wasm function, which returns here. `lr` is set by a preceding `SetReg`.
    Call { target: u32 },
    /// An indirect guest call (`blx rN` through a function pointer - init_array
    /// constructors, qsort comparators, C++ vtables): `addr` is the runtime target
    /// (Thumb bit set). Emission routes it through the module's dispatcher, which
    /// maps the address to the matching translated function. `set_lr` is the return
    /// address to load into `lr` for a call (`blx rN`), or `None` for a tail call
    /// through a register (`bx rN`) that leaves `lr` untouched. Emission snapshots
    /// `addr` *before* writing `lr`, so `blx lr` (a compiler using `lr` as the
    /// call-target scratch) dispatches to the real target, not the clobbered return.
    CallIndirect { addr: Value, set_lr: Option<u32> },
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
    /// A computed jump through a dense index (ARM `tbb`/`tbh` switch dispatch):
    /// branch to `targets[index]`. The jump table was read statically at discovery
    /// time, so `targets` are resolved block addresses and the runtime only needs
    /// the index register - no guest-memory table read. `index` is guaranteed in
    /// `0..targets.len()` by a preceding range-check branch (the compiler's
    /// `cmp; bhi default`), so `default` (the out-of-range block) is normally
    /// unreachable from here; it is recorded for faithfulness when known.
    Switch { index: Value, targets: Vec<u32>, default: Option<u32> },
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
    /// A placeholder for a function that could not be lowered (an unlifted
    /// instruction). Its body is a single `unreachable` trap, so the module still
    /// builds and runs; reaching this function at runtime faults loudly, revealing
    /// that the un-transpiled code is actually on the executed path. Used only by
    /// the lenient whole-program build ([`crate::transpile_lenient`]).
    pub stub: bool,
}

impl Func {
    /// Index of the block starting at `addr`, if any (for branch lowering).
    pub fn block_index(&self, addr: u32) -> Option<usize> {
        self.blocks.iter().position(|b| b.addr == addr)
    }

    /// A trapping placeholder function at `addr` (see [`Func::stub`]).
    pub fn new_stub(addr: u32) -> Self {
        Func { addr, thumb: true, blocks: Vec::new(), stub: true }
    }

    /// True if every intra-function branch target resolves to a block in this
    /// function. A tentatively-discovered function (a guessed code pointer) can
    /// decode into nonsense whose terminator branches to an address that is not a
    /// block - emitting it would panic in `goto`. Such a function was never real
    /// and must be dropped. Hard (call-graph-reachable) functions are always
    /// well-formed by construction; this guards the tentative path.
    pub fn well_formed(&self) -> bool {
        let is_block = |a: u32| self.blocks.iter().any(|b| b.addr == a);
        self.blocks.iter().all(|b| match &b.term {
            Term::Jump(t) | Term::Branch { taken: t, .. } | Term::BranchZero { taken: t, .. } => {
                is_block(*t)
            }
            Term::Switch { targets, default, .. } => {
                targets.iter().all(|&t| is_block(t)) && default.map_or(true, is_block)
            }
            Term::Fallthrough | Term::Return | Term::Halt => true,
        })
    }
}
