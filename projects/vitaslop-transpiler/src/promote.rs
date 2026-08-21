//! Promoting the ARM register file out of wasm globals and into wasm locals: the
//! policy, and the model that prices it before any code is emitted differently.
//!
//! # The question
//! [`crate::abi`] keeps r0..r15 and NZCV in mutable i32 globals, and the module's own
//! [`crate::emit::Expansion`] report says roughly a quarter of every operator emitted is
//! a move to or from one of them. A wasm engine turns a `local.get` into a machine
//! register and a `global.get` into a load from the instance, so the obvious next
//! codegen step is to hold the file in locals and write it back only where something can
//! observe it.
//!
//! # What is measured, and why it is not the 23%
//! Two facts have to be held together, and each kills a naive version of this change:
//!
//! 1. **Promotion does not remove operators - it ADDS them.** A `local.get` costs one
//!    operator exactly as a `global.get` does, and the loads and write-backs are extra.
//!    So `fuel`, the expansion factor, and every count this project usually reasons in
//!    are BLIND to the change; only wall-clock can see it. (This is also the good news:
//!    the guest clock is billed in ARM instructions, which promotion does not touch at
//!    all, so the emulated console does not change speed - see [`crate::abi::WORK_GLOBAL`].)
//! 2. **The win shrinks fast as the register-move share falls.** MEASURED on a synthetic
//!    block on this machine, best-of-five, three repeats, both engines, with the share
//!    tuned by diluting the operator stream:
//!
//!    | register moves | V8 globals -> locals | Cranelift globals -> locals |
//!    |---|---|---|
//!    | 64.7% | -45% | -46% |
//!    | 29.7% | -12% | -23% |
//!    | 19.3% | -15% | -1.5% |
//!
//!    So at this codegen's real density the ceiling is around **12-15% of the time spent
//!    in translated code on V8**, not the 46% a naive synthetic loop suggests, and
//!    Cranelift's win is not even reliably present.
//!
//! # The policy
//! A promoted register is a write-back cache with three states - the local mirrors the
//! global (`Fresh`), the local is newer (`Dirty`), or the global is newer and the local
//! holds nothing (`Stale`). The cache is valid only along a straight-line run of
//! operators; it is flushed and invalidated at every point where control can leave or
//! join, and at every call.
//!
//! **The sync points are all one shape, which is what makes this safe to do at all.**
//! Every one of them - a guest `bl`, an indirect dispatch, a host `svc`, a Vita NID
//! import, the fuel yield the scheduler suspends on - is a wasm `Call` or `CallIndirect`
//! in the emitted stream, and every join is a branch or a scope boundary. So the whole
//! policy lives at [`crate::emit::Body`]'s single instruction seam and no individual
//! emitter has to remember it. That matters more than it sounds: the fuel check emits
//! its host call through `untolled`, which bypasses the billing path entirely, and a
//! policy applied anywhere but the seam would have missed the one call the SCHEDULER
//! suspends on - the host reads the register file there.

use crate::abi;
use wasm_encoder::Instruction;

/// A register accessed more than this many times within one straight-line run is worth
/// promoting for that run.
///
/// # Why a threshold at all
/// Promotion is not free per register: the first read costs an extra operator (the value
/// has to be both used and cached, `global.get; local.tee`) and a register that was
/// written owes two more at the end of the run (`local.get; global.set`). A register
/// touched once or twice therefore pays more than it saves. Three accesses is where the
/// converted accesses start to outnumber the operators the conversion costs.
pub const PROMOTION_THRESHOLD: u32 = abi::LOCAL_PROMOTION_THRESHOLD;

/// Number of core-state globals: the 16 ARM registers followed by the 4 condition flags.
/// They occupy the first globals in that order (see [`crate::abi`]), so one range test
/// identifies a core-state move.
pub const CORE_GLOBALS: u32 = abi::REG_COUNT as u32 + abi::FLAG_COUNT as u32;

/// True if `g` is one of the core-state globals.
pub fn is_core(g: u32) -> bool {
    g < CORE_GLOBALS
}

/// What promoting this module's register file would cost and save, counted over the
/// operator stream that is actually emitted rather than estimated from the IR.
///
/// Reported alongside the expansion factor so the decision to do the work - or to stop
/// proposing it - rests on this module's own numbers. See the module docs for what the
/// numbers mean and for the measured price of a converted access.
#[derive(Clone, Copy, Debug, Default)]
pub struct PromotionModel {
    /// Straight-line runs the operator stream breaks into.
    pub runs: u64,
    /// Core-state accesses that would become LOCAL accesses under the policy - the
    /// operators that get cheaper. This is the number the win is proportional to.
    pub converted: u64,
    /// Core-state accesses left on their globals because their run did not touch them
    /// often enough to pay the conversion back.
    pub left: u64,
    /// Operators the policy ADDS: one per promoted register first read in a run, two per
    /// promoted register written in one.
    pub overhead: u64,
    /// Longest straight-line run, in core-state accesses. A run is bounded by branches,
    /// scope boundaries and calls, so this says how much room the policy has at all.
    pub longest_run: u32,
    /// Accesses left on their globals, split by WHAT ended the run they were in.
    ///
    /// This is the number that says whether a smarter policy is worth writing. The
    /// conservative rule invalidates the cache at every control-flow operator, but they
    /// are not all equally unavoidable: a `Call` genuinely can rewrite the register file
    /// behind the cache's back, whereas the `if`/`end` an ARM predicated instruction
    /// lowers to is a STRUCTURED construct whose two paths a linear emitter can reconcile.
    /// If most of the loss is the structured kind, a better policy exists; if it is calls,
    /// this is the ceiling.
    pub lost_to: [u64; ENDER_KINDS],
}

/// What ended a run, for [`PromotionModel::lost_to`].
#[derive(Clone, Copy, PartialEq, Eq, Debug)]
pub enum Ender {
    /// A call: a guest `bl`, an indirect dispatch, a host `svc`/import, or the fuel
    /// yield. The callee reaches the same globals, so this one is not negotiable.
    Call = 0,
    /// A structured scope boundary - `if`/`else`/`end`, which is what ARM predication and
    /// an `IT` block lower to. Reconcilable in principle.
    Scope = 1,
    /// A branch or branch target: the dispatch `br_table`, a `br` to it, a `block`/`loop`
    /// header. A real control-flow join.
    Branch = 2,
    /// `return` or `unreachable` - the function is leaving.
    Exit = 3,
}

/// Number of [`Ender`] kinds.
pub const ENDER_KINDS: usize = 4;

impl PromotionModel {
    /// Fold one function's model into a module-wide total.
    pub fn add(&mut self, other: &PromotionModel) {
        self.runs += other.runs;
        self.converted += other.converted;
        self.left += other.left;
        self.overhead += other.overhead;
        self.longest_run = self.longest_run.max(other.longest_run);
        for (a, b) in self.lost_to.iter_mut().zip(other.lost_to.iter()) {
            *a += b;
        }
    }

    /// Share of a module's `emitted` operators that would become local accesses.
    pub fn converted_share(&self, emitted: u64) -> f64 {
        if emitted == 0 {
            return 0.0;
        }
        100.0 * self.converted as f64 / emitted as f64
    }

    /// Net change in emitted operators, as a share of `emitted`. Positive means the
    /// module gets BIGGER, which promotion always does - the win is in what the
    /// converted accesses cost to execute, not in how many there are.
    pub fn overhead_share(&self, emitted: u64) -> f64 {
        if emitted == 0 {
            return 0.0;
        }
        100.0 * self.overhead as f64 / emitted as f64
    }
}

/// Accumulates the core-state accesses of the run currently being emitted and, when the
/// run ends, folds what the policy would have done with it into a [`PromotionModel`].
///
/// Measuring on the emitted stream rather than on the IR is deliberate: several register
/// accesses have no IR statement of their own (an inlined host call reads its arguments
/// straight out of the register globals, `Rbit` and `MulLong` stage through them, the
/// flag statements write them), and a model built from the IR would miss exactly those.
#[derive(Default)]
pub struct RunTracker {
    /// `(global, is_write)` for each core-state access since the last sync point.
    run: Vec<(u32, bool)>,
    pub model: PromotionModel,
    /// The promoted set of each completed run, which a second emission pass applies.
    pub plan: Plan,
}

impl RunTracker {
    /// Record one core-state access.
    pub fn access(&mut self, global: u32, is_write: bool) {
        self.run.push((global, is_write));
    }

    /// End the current run: control can leave or join here, or a call can observe the
    /// globals, so the cache would be flushed and invalidated. `by` is what ended it,
    /// which is what [`PromotionModel::lost_to`] attributes the unpromotable accesses to.
    pub fn sync(&mut self, by: Ender) {
        if self.run.is_empty() {
            return;
        }
        self.model.runs += 1;
        self.model.longest_run = self.model.longest_run.max(self.run.len() as u32);

        // Per core-state global: how many times touched, whether the first touch was a
        // read, and whether anything wrote it. That is everything the policy needs.
        let mut count = [0u32; CORE_GLOBALS as usize];
        let mut first_is_read = [false; CORE_GLOBALS as usize];
        let mut written = [false; CORE_GLOBALS as usize];
        for &(g, is_write) in &self.run {
            let i = g as usize;
            if count[i] == 0 {
                first_is_read[i] = !is_write;
            }
            count[i] += 1;
            written[i] |= is_write;
        }

        let mut mask = 0u32;
        for i in 0..CORE_GLOBALS as usize {
            let n = count[i];
            if n == 0 {
                continue;
            }
            if n > PROMOTION_THRESHOLD {
                mask |= 1 << i;
                self.model.converted += u64::from(n);
                // The first read has to both yield the value and cache it.
                if first_is_read[i] {
                    self.model.overhead += 1;
                }
                // A register the run wrote owes its global the new value before anything
                // can observe it.
                if written[i] {
                    self.model.overhead += 2;
                }
            } else {
                self.model.left += u64::from(n);
                self.model.lost_to[by as usize] += u64::from(n);
            }
        }
        self.plan.runs.push(mask);

        self.run.clear();
    }
}

/// The per-run promoted sets a modelling pass produced for one function: `plan[k]` is a
/// bitmask over the core globals promoted during run `k`.
///
/// # Why the plan comes from a first emission pass rather than from the IR
/// Whether a register pays its promotion back depends on how many times the EMITTED code
/// touches it in the run, and several of those touches have no IR statement of their own
/// - an inlined host call reads its arguments straight out of the register globals,
/// `Rbit` and `MulLong` stage through them, the flag statements write them. A plan
/// derived from the IR would be a guess about the emitted stream; one derived from the
/// stream is the stream.
///
/// The two passes see IDENTICAL run boundaries, which is what makes the plan applicable
/// at all: a run ends only at a control-flow operator or a call, and promotion adds
/// neither - it only ever substitutes a local access for a global one, or inserts a load
/// or a write-back.
#[derive(Clone, Debug, Default)]
pub struct Plan {
    /// One mask per non-empty run, in emission order.
    pub runs: Vec<u32>,
}

impl Plan {
    /// True if this plan promotes nothing anywhere, so the function needs no locals for
    /// it and emits exactly what it always did.
    pub fn is_empty(&self) -> bool {
        self.runs.iter().all(|&m| m == 0)
    }
}

/// Where a promoted register's current value lives.
#[derive(Clone, Copy, PartialEq, Eq, Debug)]
enum Cell {
    /// The local holds nothing usable; the global is the truth. Every register starts
    /// here at the top of a run.
    Stale,
    /// The local and the global agree.
    Fresh,
    /// The local is newer; the global owes a write-back before anything can observe it.
    Dirty,
}

/// The ARM register file held in wasm locals for the duration of a straight-line run: a
/// write-back cache over the core globals.
///
/// # The invariant, which is the whole correctness argument
/// **At every point where anything other than this run's own emitted code could read a
/// core global, that global holds the register's current value.** The cache enforces it
/// by flushing every `Dirty` cell immediately before any operator that could hand control
/// away - a call, a branch, a scope boundary, a return, a trap - and then treating every
/// cell as `Stale`, because after such an operator the local may not have been written on
/// the path that arrives here, and a call may have rewritten the global underneath.
///
/// That invariant is deliberately stronger than it needs to be at a scope boundary, where
/// the two paths could in principle be reconciled. It is the version that is obviously
/// correct; see the module docs for what the weaker one would be worth.
pub struct Cache {
    plan: Plan,
    /// Local index of promoted global `g` is `base + g`.
    base: u32,
    /// Index of the run being emitted.
    run: usize,
    /// Whether the current run has seen a core-state access yet. A run with none is not
    /// in the plan (the modelling pass skips it), so the index must not advance for it.
    touched: bool,
    cell: [Cell; CORE_GLOBALS as usize],
    /// The FALSIFIER constant, when this is a poison build. See [`Cache::poison`].
    poison: Option<i32>,
}

impl Cache {
    /// A cache applying `plan`, with the promoted locals starting at local index `base`.
    pub fn new(plan: Plan, base: u32) -> Self {
        Cache {
            plan,
            base,
            run: 0,
            touched: false,
            cell: [Cell::Stale; CORE_GLOBALS as usize],
            poison: None,
        }
    }

    /// `VITASLOP_PROMOTE_POISON=<n>` - the falsifier. While a register is held in its
    /// local, store `n` into its GLOBAL, so anything that reads the global during the run
    /// - when it should have read the local - gets a value that is wrong rather than
    /// merely stale.
    ///
    /// # Why this is needed and a code review is not
    /// A promotion bug is invisible in the ordinary way: the global holds the register's
    /// value from *some* earlier moment, so a wrong read usually returns something
    /// plausible and often the right answer by luck. Nothing crashes and no pixel moves
    /// until the one run where it does. Poison converts "stale" into "impossible".
    ///
    /// **Run the title in TWO poison builds and compare them with each other**, never a
    /// poisoned build against a plain one - the same rule the flag-liveness falsifier
    /// documents. The two differ only in one constant, so they emit the same number of
    /// operators, burn identical fuel and schedule identically; the single variable is
    /// what an incorrectly-read global would contain.
    ///
    /// The write-back has to be widened to match: a poison build flushes every cached
    /// register, not only the dirty ones, because a register that was read and never
    /// written would otherwise leave the poison sitting in its global for the next run.
    pub fn with_poison(mut self, poison: Option<i32>) -> Self {
        self.poison = poison;
        self
    }

    /// Number of locals a promoted function declares.
    pub const LOCALS: u32 = CORE_GLOBALS;

    fn promoted(&self, g: u32) -> bool {
        self.plan.runs.get(self.run).is_some_and(|m| m & (1 << g) != 0)
    }

    fn local(&self, g: u32) -> u32 {
        self.base + g
    }

    /// Offer one operator to the cache. Returns true if the cache emitted a replacement,
    /// in which case the caller must not emit the operator itself.
    ///
    /// `billed` is carried through to whatever the cache emits, so a replacement costs
    /// the emulator exactly what the operator it replaced would have.
    pub fn offer<S: Sink>(&mut self, out: &mut S, i: &Instruction, billed: bool) -> bool {
        match i {
            &Instruction::GlobalGet(g) if is_core(g) => {
                self.touched = true;
                if !self.promoted(g) {
                    return false;
                }
                if self.cell[g as usize] == Cell::Stale {
                    // First read of the run: yield the value AND cache it. This is the
                    // one operator a promotion costs up front.
                    out.raw(&Instruction::GlobalGet(g), billed);
                    out.raw(&Instruction::LocalTee(self.local(g)), billed);
                    self.cell[g as usize] = Cell::Fresh;
                    // In a falsifier build the global now holds a value nothing may read
                    // until the write-back restores it. Untolled, so the two poison arms
                    // stay identical in fuel to each other - which is what makes them
                    // comparable. See `with_poison`.
                    if let Some(p) = self.poison {
                        out.raw(&Instruction::I32Const(p), false);
                        out.raw(&Instruction::GlobalSet(g), false);
                    }
                } else {
                    out.raw(&Instruction::LocalGet(self.local(g)), billed);
                }
                true
            }
            &Instruction::GlobalSet(g) if is_core(g) => {
                self.touched = true;
                if !self.promoted(g) {
                    return false;
                }
                out.raw(&Instruction::LocalSet(self.local(g)), billed);
                self.cell[g as usize] = Cell::Dirty;
                true
            }
            _ => {
                if ends_run(i).is_some() {
                    // Everything this run wrote has to be back on its global BEFORE the
                    // operator that can hand control away.
                    //
                    // # Why inserting code here cannot break the module
                    // Each write-back is `local.get; global.set` - it pushes one value and
                    // pops it. Balanced, so whatever was already staged for the operator
                    // stays exactly where it was and at the same depth: a `br_if`'s
                    // condition, a call's arguments, and - the case that is easy to miss -
                    // the RESULT VALUE of an `if` block typed `BlockType::Result(i32)`,
                    // which this emitter does use, sitting on the stack at the `else` and
                    // the `end`. The whole real title validates as wasm with the cache on,
                    // which is the check that would catch a violation of this.
                    self.write_back(out, billed);
                    if self.touched {
                        self.run += 1;
                        self.touched = false;
                    }
                }
                false
            }
        }
    }

    /// Flush every dirty cell to its global and invalidate the whole cache.
    fn write_back<S: Sink>(&mut self, out: &mut S, billed: bool) {
        for g in 0..CORE_GLOBALS {
            // A poison build also restores a register it only READ: its global is holding
            // the poison, and leaving that there would corrupt the next run rather than
            // falsify this one.
            let owes = match self.cell[g as usize] {
                Cell::Dirty => true,
                Cell::Fresh => self.poison.is_some(),
                Cell::Stale => false,
            };
            if owes {
                out.raw(&Instruction::LocalGet(self.local(g)), billed);
                out.raw(&Instruction::GlobalSet(g), billed);
            }
            // Invalidated, not merely cleaned: control can ARRIVE here from a path that
            // never wrote this local, and a call can have rewritten the global.
            self.cell[g as usize] = Cell::Stale;
        }
    }
}

/// Where [`Cache`] puts the operators it emits. Implemented by the emitter's function
/// body; a trait so the cache can own the policy without the policy owning the encoder.
pub trait Sink {
    /// Encode one instruction WITHOUT offering it back to the cache.
    fn raw(&mut self, i: &Instruction, billed: bool);
}

/// What kind of run-ender this operator is, or `None` if it does not end a run.
///
/// `Block` and `Loop` count even though entering one is unconditional: a `Block` is a
/// branch TARGET, so control can arrive at it from a `br` that knew nothing of this run's
/// cache, and a `Loop` is the same plus a back edge.
pub fn ends_run(i: &Instruction) -> Option<Ender> {
    use Instruction as W;
    Some(match i {
        W::Call(_)
        | W::CallIndirect { .. }
        | W::ReturnCall(_)
        | W::ReturnCallIndirect { .. }
        | W::ReturnCallRef(_) => Ender::Call,
        W::If(_) | W::Else | W::End => Ender::Scope,
        W::Block(_) | W::Loop(_) | W::Br(_) | W::BrIf(_) | W::BrTable(..) | W::BrOnNull(_)
        | W::BrOnNonNull(_) => Ender::Branch,
        W::Return | W::Unreachable => Ender::Exit,
        _ => return None,
    })
}
