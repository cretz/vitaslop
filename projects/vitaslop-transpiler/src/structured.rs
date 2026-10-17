//! >>> STRUCTURED CONTROL FLOW: a guest function's blocks as wasm `block`/`loop`/`if`, not a
//! > > > `br_table` dispatch loop.
//!
//! The dispatch emitter ([`super::emit_func_body`]'s fallback) makes every transfer that is not
//! a fallthrough set `$bb` and re-enter one loop through a jump table - an INDIRECT branch, and
//! one that hides the function's real shape from the engine's optimiser. PRICED on the product
//! path (desktop browser, a football title's match, SMP): routing the fallthroughs through the
//! table as well (`VITASLOP_DISPATCH_ALL`, +31% re-entries) cost the main thread 6.8% of its
//! guest MIPS, and giving only self-loops a native `loop` gained 1.5% (+4% on a job thread).
//!
//! # The algorithm
//! Ramsey, "Beyond Relooper: recursive translation of unstructured control flow to structured
//! control flow" (ICFP 2022), on the dominator tree of a REDUCIBLE graph:
//!
//! * a node's code is emitted where its immediate dominator's is, so every forward edge either
//!   leads to a node this one dominates (whose code is then inlined in place - it has exactly
//!   one forward edge in, so it is emitted once) or to a MERGE node;
//! * a merge node (two or more forward edges in, or any switch target) is emitted right after
//!   a `block` that its dominator's code is nested in, so a forward edge to it is a `br` out;
//! * a loop header (the target of a retreating edge) wraps its code in a `loop`, so the
//!   retreating edge is a `br` back to it.
//!
//! A function this cannot express - an IRREDUCIBLE graph, or a dominator tree deeper than
//! [`MAX_DEPTH`] (the emission recurses along it) - keeps the dispatch loop, which expresses
//! anything. So does a function on a cycle of direct calls, for its STACK frames rather than
//! its graph - see [`recursive_functions`]. `VITASLOP_STRUCTURED_CF=0` puts every function
//! back on it.
//!
//! # What does NOT change
//! The per-block code is [`super::emit_block_body`], so the clock charge, the statements, the
//! dirty marks and the NEON cache are byte-for-byte what the dispatcher emits around them. The
//! fuel check sits on exactly the edges the dispatcher puts it on (target address <= source
//! address, see [`super::goto`]): every cycle in a function has such an edge, whatever its
//! layout, so every loop still preempts.

use super::*;
use std::collections::{HashMap, HashSet};

/// Dominator-tree depth past which a function keeps the dispatch loop. The emission recurses
/// about four frames per level and a browser worker's stack is small; a dominator chain this
/// long is a very long run of blocks, where the dispatch costs least anyway.
const MAX_DEPTH: usize = 400;

/// Whether functions are emitted structured. ON by default; `VITASLOP_STRUCTURED_CF=0` is the
/// arm back to the dispatch loop everywhere (value-sensitive, like the other emit arms).
pub(super) fn structured_cf() -> bool {
    use std::sync::OnceLock;
    static ON: OnceLock<bool> = OnceLock::new();
    *ON.get_or_init(|| !matches!(emit_var("VITASLOP_STRUCTURED_CF").as_deref(), Ok("0")))
}

/// The analysis the emission walks. Indices are into `func.blocks`.
pub(super) struct Cfg {
    entry: usize,
    /// Reverse-postorder number from the entry; `usize::MAX` for an unreachable block.
    rpo: Vec<usize>,
    loop_header: Vec<bool>,
    merge: Vec<bool>,
    /// Dominator-tree children that are merge nodes, highest reverse-postorder number first
    /// (the order the nested `block`s are opened in).
    merge_children: Vec<Vec<usize>>,
}

/// The successors of block `i`, in no particular order, or `None` if a target is not a block
/// of this function (the dispatcher reports that with its own message).
fn successors(func: &Func, index: &HashMap<u32, usize>, i: usize) -> Option<Vec<usize>> {
    let next = (i + 1 < func.blocks.len()).then_some(i + 1);
    let at = |a: &u32| index.get(a).copied();
    Some(match &func.blocks[i].term {
        Term::Fallthrough | Term::ReturnIf { .. } => next.into_iter().collect(),
        Term::Jump(t) => vec![at(t)?],
        Term::Branch { taken, .. } | Term::BranchZero { taken, .. } => {
            let mut v = vec![at(taken)?];
            v.extend(next);
            v
        }
        Term::Switch { targets, default, .. } => {
            let mut v = Vec::with_capacity(targets.len() + 1);
            for t in targets {
                v.push(at(t)?);
            }
            if let Some(d) = default {
                v.push(at(d)?);
            }
            v
        }
        Term::Return | Term::Halt | Term::Unreachable => Vec::new(),
    })
}

/// >>> A FUNCTION ON A CYCLE OF DIRECT CALLS KEEPS THE DISPATCH LOOP.
///
/// Every guest call is a nested wasm call, so a guest that recurses N deep holds N frames of
/// each function on the cycle, and the browser's stack is fixed. MEASURED on the phone, a
/// football title's match: with its functions structured, a four-function recursion ran the
/// worker out of stack ("Maximum call stack size exceeded", 5 of 9 runs, frames 10285-14639);
/// on the dispatch loop it never did. On the desktop with V8's stack cut to 400 KB the same
/// run overflowed structured and finished on the dispatcher - and finished structured under
/// `--liftoff-only` and under `--no-wasm-inlining` too, so it is TurboFan INLINING into the
/// structured body that grows each frame. Keeping only the cycle (five functions of 115,000)
/// on the dispatcher finished the 400 KB run. A function on no cycle appears at most once on
/// any stack, so its frame size cannot be multiplied by the guest's data.
///
/// `edges` maps each function to the functions it calls directly; a callee not in `edges` is
/// ignored. Calls through a pointer are not edges, so a cycle reachable only that way is not
/// seen. Iterative Tarjan: this runs in a browser worker, whose stack is the thing at stake.
pub(crate) fn recursive_functions(edges: &std::collections::BTreeMap<u32, Vec<u32>>) -> HashSet<u32> {
    let addrs: Vec<u32> = edges.keys().copied().collect();
    let id: HashMap<u32, usize> = addrs.iter().enumerate().map(|(i, &a)| (a, i)).collect();
    let adj: Vec<Vec<usize>> =
        edges.values().map(|callees| callees.iter().filter_map(|a| id.get(a).copied()).collect()).collect();
    let n = addrs.len();
    let mut index = vec![usize::MAX; n];
    let mut low = vec![0usize; n];
    let mut on_stack = vec![false; n];
    let mut stack: Vec<usize> = Vec::new();
    let mut next = 0usize;
    let mut out = HashSet::new();
    for root in 0..n {
        if index[root] != usize::MAX {
            continue;
        }
        index[root] = next;
        low[root] = next;
        next += 1;
        stack.push(root);
        on_stack[root] = true;
        // (node, next edge to look at) - the recursion's frames, on the heap.
        let mut walk: Vec<(usize, usize)> = vec![(root, 0)];
        while let Some(top) = walk.last_mut() {
            let v = top.0;
            if top.1 < adj[v].len() {
                let w = adj[v][top.1];
                top.1 += 1;
                if index[w] == usize::MAX {
                    index[w] = next;
                    low[w] = next;
                    next += 1;
                    stack.push(w);
                    on_stack[w] = true;
                    walk.push((w, 0));
                } else if on_stack[w] {
                    low[v] = low[v].min(index[w]);
                }
                continue;
            }
            walk.pop();
            if let Some(&(u, _)) = walk.last() {
                low[u] = low[u].min(low[v]);
            }
            if low[v] == index[v] {
                let mut component = Vec::new();
                while let Some(w) = stack.pop() {
                    on_stack[w] = false;
                    component.push(w);
                    if w == v {
                        break;
                    }
                }
                if component.len() > 1 || adj[v].contains(&v) {
                    out.extend(component.iter().map(|&w| addrs[w]));
                }
            }
        }
    }
    out
}

/// Analyse `func`, or `None` when it must keep the dispatch loop (see the module doc).
pub(super) fn analyse(func: &Func) -> Option<Cfg> {
    let n = func.blocks.len();
    // An empty function (it happens) is left to the dispatcher's form exactly as it was.
    if n == 0 {
        return None;
    }
    let index: HashMap<u32, usize> = func.blocks.iter().enumerate().map(|(i, b)| (b.addr, i)).collect();
    // The dispatcher starts at the block AT the function's address, else block 0.
    let entry = index.get(&func.addr).copied().unwrap_or(0);
    let mut succs = Vec::with_capacity(n);
    for i in 0..n {
        succs.push(successors(func, &index, i)?);
    }
    // Postorder by an explicit stack (no recursion on a long chain).
    let mut post = Vec::with_capacity(n);
    let mut seen = vec![false; n];
    let mut stack: Vec<(usize, usize)> = vec![(entry, 0)];
    seen[entry] = true;
    while let Some(&mut (v, ref mut k)) = stack.last_mut() {
        if let Some(&w) = succs[v].get(*k) {
            *k += 1;
            if !seen[w] {
                seen[w] = true;
                stack.push((w, 0));
            }
        } else {
            post.push(v);
            stack.pop();
        }
    }
    let mut rpo = vec![usize::MAX; n];
    for (k, &v) in post.iter().rev().enumerate() {
        rpo[v] = k;
    }
    let order: Vec<usize> = post.iter().rev().copied().collect();
    // Predecessors among the reachable blocks, one entry per EDGE.
    let mut preds: Vec<Vec<usize>> = vec![Vec::new(); n];
    for &v in &order {
        for &w in &succs[v] {
            preds[w].push(v);
        }
    }
    // Immediate dominators (Cooper, Harvey & Kennedy, "A Simple, Fast Dominance Algorithm").
    let undef = usize::MAX;
    let mut idom = vec![undef; n];
    idom[entry] = entry;
    let intersect = |idom: &[usize], mut a: usize, mut b: usize| {
        while a != b {
            while rpo[a] > rpo[b] {
                a = idom[a];
            }
            while rpo[b] > rpo[a] {
                b = idom[b];
            }
        }
        a
    };
    let mut changed = true;
    while changed {
        changed = false;
        for &v in order.iter().skip(1) {
            let mut new = undef;
            for &p in &preds[v] {
                if idom[p] != undef {
                    new = if new == undef { p } else { intersect(&idom, p, new) };
                }
            }
            if new != idom[v] {
                idom[v] = new;
                changed = true;
            }
        }
    }
    let dominates = |a: usize, mut b: usize| loop {
        if a == b {
            return true;
        }
        if b == entry {
            return false;
        }
        b = idom[b];
    };
    let mut loop_header = vec![false; n];
    let mut forward_in = vec![0usize; n];
    let mut merge = vec![false; n];
    for &v in &order {
        for &w in &succs[v] {
            if rpo[w] <= rpo[v] {
                // A retreating edge must be a BACK edge - its target dominating its source - or
                // the graph is irreducible and has no structured form.
                if !dominates(w, v) {
                    return None;
                }
                loop_header[w] = true;
            } else {
                forward_in[w] += 1;
            }
        }
        // A switch arm is always a `br` to a label, so every target is given one.
        if let Term::Switch { .. } = func.blocks[v].term {
            for &w in &succs[v] {
                merge[w] = true;
            }
        }
    }
    for v in 0..n {
        merge[v] |= forward_in[v] >= 2;
    }
    let mut merge_children: Vec<Vec<usize>> = vec![Vec::new(); n];
    let mut depth = vec![0usize; n];
    for &v in order.iter().skip(1) {
        // Reverse postorder visits a dominator before what it dominates.
        depth[v] = depth[idom[v]] + 1;
        if depth[v] > MAX_DEPTH {
            return None;
        }
        if merge[v] {
            merge_children[idom[v]].push(v);
        }
    }
    for c in &mut merge_children {
        c.sort_by(|a, b| rpo[*b].cmp(&rpo[*a]));
    }
    Some(Cfg { entry, rpo, loop_header, merge, merge_children })
}

/// A frame the emission has open, innermost last - what a `br` depth counts through.
#[derive(PartialEq)]
enum Frame {
    /// An `if`/`else` of a conditional terminator, or a `block` of a switch's pads.
    Plain,
    /// The `loop` of this loop header: a retreating edge to it is a `br` here.
    Loop(usize),
    /// A `block` whose `end` is followed by this merge node's code.
    Follows(usize),
}

pub(super) struct Emitter<'a> {
    pub f: &'a mut Body,
    pub func: &'a Func,
    pub cfg: &'a Cfg,
    pub func_index: &'a BTreeMap<u32, u32>,
    pub base: u32,
    pub inline: &'a InlineImports,
    frames: Vec<Frame>,
}

impl<'a> Emitter<'a> {
    pub fn new(
        f: &'a mut Body,
        func: &'a Func,
        cfg: &'a Cfg,
        func_index: &'a BTreeMap<u32, u32>,
        base: u32,
        inline: &'a InlineImports,
    ) -> Self {
        Emitter { f, func, cfg, func_index, base, inline, frames: Vec::new() }
    }

    /// The whole function body, without its final `end`.
    pub fn emit(&mut self) {
        self.tree(self.cfg.entry);
        debug_assert!(self.frames.is_empty());
    }

    fn open(&mut self, i: &W, frame: Frame) {
        self.f.instruction(i);
        self.frames.push(frame);
    }

    fn close(&mut self) {
        self.frames.pop();
        self.f.instruction(&W::End);
    }

    /// `br` depth to `frame`, which the algorithm guarantees is open.
    fn depth(&self, frame: &Frame) -> u32 {
        let at = self.frames.iter().rposition(|x| x == frame).unwrap_or_else(|| {
            panic!("structured emit: no open label for a branch in f_{:x}", self.func.addr)
        });
        (self.frames.len() - 1 - at) as u32
    }

    /// Ramsey's `doTree`: node `x` with everything it dominates.
    fn tree(&mut self, x: usize) {
        let ys = self.cfg.merge_children[x].clone();
        if self.cfg.loop_header[x] {
            self.open(&W::Loop(BlockType::Empty), Frame::Loop(x));
            self.within(x, &ys);
            self.close();
        } else {
            self.within(x, &ys);
        }
    }

    /// Ramsey's `nodeWithin`: `x`'s code nested in one `block` per merge child, the highest
    /// numbered outermost, each child's tree following its block's `end`.
    fn within(&mut self, x: usize, ys: &[usize]) {
        match ys.split_first() {
            Some((&y, rest)) => {
                self.open(&W::Block(BlockType::Empty), Frame::Follows(y));
                self.within(x, rest);
                self.close();
                self.tree(y);
            }
            None => {
                let func = self.func;
                emit_block_body(self.f, &func.blocks[x], func, self.func_index, self.base, self.inline);
                self.term(x);
            }
        }
    }

    /// Ramsey's `doBranch`, from block `src` to block `dst`.
    fn branch(&mut self, src: usize, dst: usize) {
        // Exactly the dispatcher's back-edge rule (`goto`), so preemption is where it was.
        if self.func.blocks[dst].addr <= self.func.blocks[src].addr {
            emit_fuel_check(self.f);
        }
        if self.cfg.rpo[dst] <= self.cfg.rpo[src] {
            let d = self.depth(&Frame::Loop(dst));
            self.f.instruction(&W::Br(d));
        } else if self.cfg.merge[dst] {
            let d = self.depth(&Frame::Follows(dst));
            self.f.instruction(&W::Br(d));
        } else {
            self.tree(dst);
        }
    }

    /// The fallthrough successor of `src` (the next block by index), or a return from the
    /// last one - which is what running off the dispatcher's last block does.
    fn branch_next(&mut self, src: usize) {
        if src + 1 < self.func.blocks.len() {
            self.branch(src, src + 1);
        } else {
            self.f.instruction(&W::Return);
        }
    }

    fn index(&self, addr: u32) -> usize {
        self.func.block_index(addr).expect("analyse resolved every target")
    }

    fn term(&mut self, x: usize) {
        let func = self.func;
        match &func.blocks[x].term {
            Term::Fallthrough => self.branch_next(x),
            Term::Return => {
                self.f.instruction(&W::Return);
            }
            Term::ReturnIf { cond } => {
                emit_cond(self.f, *cond);
                self.open(&W::If(BlockType::Empty), Frame::Plain);
                self.f.instruction(&W::Return);
                self.close();
                self.branch_next(x);
            }
            Term::Halt => {
                if trap_halt() {
                    self.f.instruction(&W::Unreachable);
                } else {
                    self.f.instruction(&W::Return);
                }
            }
            Term::Unreachable => {
                self.f.instruction(&W::Unreachable);
            }
            Term::Jump(t) => {
                let t = self.index(*t);
                self.branch(x, t);
            }
            Term::Branch { cond, taken } => {
                let t = self.index(*taken);
                emit_cond(self.f, *cond);
                self.conditional(x, t);
            }
            Term::BranchZero { reg, nonzero, taken } => {
                let t = self.index(*taken);
                self.f.instruction(&W::GlobalGet(abi::reg_global(*reg as usize)));
                self.f.instruction(&W::I32Eqz); // reg == 0
                if *nonzero {
                    self.f.instruction(&W::I32Eqz); // reg != 0
                }
                self.conditional(x, t);
            }
            // The dispatcher's pad layout (`emit_term`), each pad a `br` to its target's label
            // (every switch target is a merge node - see `analyse`).
            Term::Switch { index, targets, default } => {
                let n = targets.len() as u32;
                for _ in 0..=n {
                    self.open(&W::Block(BlockType::Empty), Frame::Plain);
                }
                emit_value(self.f, index, self.base);
                let table: Vec<u32> = (0..n).collect();
                self.f.instruction(&W::BrTable(table.into(), n /* default -> outer block */));
                for &t in targets {
                    self.close(); // closes T_i
                    let t = self.index(t);
                    self.branch(x, t);
                }
                self.close(); // the default (outer) block
                match default {
                    Some(d) => {
                        let d = self.index(*d);
                        self.branch(x, d);
                    }
                    None => {
                        self.f.instruction(&W::Unreachable);
                    }
                }
            }
        }
    }

    /// The condition is on the stack: `taken` when it holds, the next block when it does not.
    fn conditional(&mut self, x: usize, taken: usize) {
        self.open(&W::If(BlockType::Empty), Frame::Plain);
        self.branch(x, taken);
        self.f.instruction(&W::Else);
        self.branch_next(x);
        self.close();
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    /// A function of blocks at 0x100, 0x110, ... with these terminators.
    fn func(terms: Vec<Term>) -> Func {
        let blocks = terms
            .into_iter()
            .enumerate()
            .map(|(i, term)| Block { addr: 0x100 + 0x10 * i as u32, stmts: Vec::new(), term, arm_count: 1 })
            .collect();
        Func { addr: 0x100, thumb: true, blocks, stub: false, recursive: false }
    }

    fn a(i: u32) -> u32 {
        0x100 + 0x10 * i
    }

    /// `if (c) A else B; join`: the join has two forward edges in and is a merge node; the
    /// arms are emitted inline, so neither is.
    #[test]
    fn a_diamond_merges_at_the_join() {
        let f = func(vec![
            Term::Branch { cond: ConditionCode::EQ, taken: a(2) }, // 0 -> 2 | 1
            Term::Jump(a(3)),                                    // 1 -> 3
            Term::Fallthrough,                                   // 2 -> 3
            Term::Return,                                        // 3
        ]);
        let cfg = analyse(&f).expect("reducible");
        assert_eq!(cfg.merge, vec![false, false, false, true]);
        assert_eq!(cfg.loop_header, vec![false; 4]);
        assert_eq!(cfg.merge_children[0], vec![3], "the join is the entry's merge child");
    }

    /// A loop's header is the target of its retreating edge, and its exit a merge child.
    #[test]
    fn a_loop_has_a_header() {
        let f = func(vec![
            Term::Fallthrough,                                     // 0 -> 1
            Term::Branch { cond: ConditionCode::NE, taken: a(1) }, // 1 -> 1 | 2  (self-loop)
            Term::Return,                                          // 2
        ]);
        let cfg = analyse(&f).expect("reducible");
        assert_eq!(cfg.loop_header, vec![false, true, false]);
        assert!(!cfg.merge[2], "one forward edge in: inlined");
    }

    /// Two entries into one cycle (`0 -> 1`, `0 -> 2`, `1 <-> 2`) has no structured form, and
    /// the function keeps the dispatch loop.
    #[test]
    fn an_irreducible_graph_keeps_the_dispatcher() {
        let f = func(vec![
            Term::Branch { cond: ConditionCode::EQ, taken: a(2) }, // 0 -> 2 | 1
            Term::Jump(a(2)),                                      // 1 -> 2
            Term::Branch { cond: ConditionCode::NE, taken: a(1) }, // 2 -> 1 | 3
            Term::Return,                                          // 3
        ]);
        assert!(analyse(&f).is_none());
    }

    /// The same target on both edges of a conditional is reached by TWO edges, so it is a
    /// merge node - inlining it would emit its code twice.
    #[test]
    fn a_target_reached_twice_from_one_block_is_a_merge_node() {
        let f = func(vec![
            Term::Branch { cond: ConditionCode::EQ, taken: a(1) }, // 0 -> 1 | 1
            Term::Return,                                          // 1
        ]);
        let cfg = analyse(&f).expect("reducible");
        assert!(cfg.merge[1]);
    }

    /// The entry need not be block 0: a shared tail below the entry is reached backward, and
    /// the entry is where reverse postorder starts.
    #[test]
    fn the_entry_is_the_block_at_the_function_address() {
        let mut f = func(vec![Term::Return, Term::Branch { cond: ConditionCode::EQ, taken: a(0) }, Term::Return]);
        f.addr = a(1);
        let cfg = analyse(&f).expect("reducible");
        assert_eq!(cfg.entry, 1);
        assert_eq!(cfg.rpo[1], 0);
        assert!(!cfg.loop_header[0], "a backward ADDRESS is not a retreating edge");
    }

    fn graph(edges: &[(u32, &[u32])]) -> std::collections::BTreeMap<u32, Vec<u32>> {
        edges.iter().map(|&(f, cs)| (f, cs.to_vec())).collect()
    }

    fn sorted(s: HashSet<u32>) -> Vec<u32> {
        let mut v: Vec<u32> = s.into_iter().collect();
        v.sort_unstable();
        v
    }

    /// Only functions ON a cycle: the caller into it and the leaf it calls appear once on any
    /// stack, so they keep the structured form.
    #[test]
    fn only_the_cycle_is_recursive() {
        // 1 -> 2 -> 3 -> 4 -> 5 -> 2 (the four-function shape that overflowed), 3 -> 9 a leaf.
        let g = graph(&[(1, &[2]), (2, &[3]), (3, &[4, 9]), (4, &[5]), (5, &[2]), (9, &[])]);
        assert_eq!(sorted(recursive_functions(&g)), vec![2, 3, 4, 5]);
    }

    #[test]
    fn self_recursion_counts_and_a_chain_does_not() {
        let g = graph(&[(1, &[2]), (2, &[2, 3]), (3, &[4]), (4, &[])]);
        assert_eq!(sorted(recursive_functions(&g)), vec![2]);
        let chain = graph(&[(1, &[2, 3]), (2, &[3]), (3, &[])]);
        assert!(recursive_functions(&chain).is_empty(), "a diamond of calls is not a cycle");
    }

    /// A callee outside the map (an import, a function not lifted) closes no cycle.
    #[test]
    fn unknown_callees_are_ignored() {
        let g = graph(&[(1, &[0xdead, 2]), (2, &[0xdead])]);
        assert!(recursive_functions(&g).is_empty());
    }

    /// The walk keeps its frames on the heap: a call chain far deeper than any native stack
    /// would allow recursing along, ending in a cycle, is analysed and the cycle found.
    #[test]
    fn a_very_deep_chain_does_not_recurse_natively() {
        const N: u32 = 200_000;
        let mut g: std::collections::BTreeMap<u32, Vec<u32>> = (0..N).map(|i| (i, vec![i + 1])).collect();
        g.insert(N, vec![N - 1]);
        assert_eq!(sorted(recursive_functions(&g)), vec![N - 1, N]);
    }
}
