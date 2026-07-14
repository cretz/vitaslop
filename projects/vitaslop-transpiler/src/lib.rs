//! Vita-agnostic ARMv7-A + Thumb-2 (later NEON/VFP) to WASM transpiler.
//!
//! Input is a code image, entry-point addresses, and import/relocation facts
//! ([`Program`]). Output is a WASM module plus the guest-function-to-export map
//! ([`Artifact`]). Nothing Vita-specific belongs here.
//!
//! Pipeline: decode + discover ([`lower`]) -> per-function CFG IR ([`ir`]) ->
//! wasm emission ([`emit`]). Each guest function becomes one wasm function whose
//! body is a dispatch loop over its basic blocks; see [`emit`] and [`abi`] for
//! the control-flow and state model, and the crate README for the rationale.

pub mod abi;
mod emit;
mod ir;
mod lower;

use std::collections::BTreeMap;

pub use ir::ConditionCode;
use lower::Imports;

/// A code image to transpile: the ARM/Thumb blob, where it loads, which
/// addresses to start decoding from, how guest imports wire to host functions,
/// and the guest memory size.
pub struct Program<'a> {
    /// The ARM/Thumb code/data image.
    pub code: &'a [u8],
    /// Guest address the image loads at (and the linear-memory rebase origin).
    pub base: u32,
    /// Default decode mode for [`entries`](Self::entries) and their transitively
    /// discovered callees: true for Thumb-2, false for ARM. A title is overwhelmingly
    /// one mode (Vita user code is Thumb); the few functions in the other mode are
    /// listed in [`arm_entries`](Self::arm_entries).
    pub thumb: bool,
    /// Entry points to discover in the default [`thumb`](Self::thumb) mode (each
    /// becomes a function, transitively pulling in its direct callees).
    pub entries: &'a [u32],
    /// Entry points to discover in ARM mode regardless of [`thumb`](Self::thumb) -
    /// even code pointers a Thumb title reaches through a `blx` into an ARM stub.
    /// Discovered tentatively (a bad guess that fails to decode is dropped), like
    /// address-taken code pointers.
    pub arm_entries: &'a [u32],
    /// Extern (imported-function) wiring: each maps a guest stub address to a
    /// dense host-import index.
    pub externs: &'a [Extern],
    /// Inter-module redirects: each maps an import-stub address to the guest
    /// address of the function that satisfies it in another module of the same
    /// link. A `bl`/`b` to such a stub becomes a direct guest call to the target
    /// (no host trap), so a call across module boundaries is as cheap as one
    /// within a module. Empty for a single-module link.
    pub redirects: &'a [Redirect],
    /// Syscall numbers (guest r7) that do not return, so a `svc` with a
    /// statically-known one of them ends decoding (before trailing data).
    pub noreturn_svc: &'a [u32],
    /// Total guest memory to provision, in bytes from `base`. The host keeps all
    /// guest allocations (image, stack, heap) within `[base, base + mem_bytes)`.
    pub mem_bytes: u32,
    /// Also discover address-taken code pointers (functions materialized via
    /// `movw`/`movt` but never directly called - e.g. a thread entry passed to
    /// sceKernelCreateThread). Such candidates are transpiled tentatively: one
    /// that fails to decode is skipped, so a mis-identified constant can never
    /// break the build. Vita modules set this; the tightly controlled ARM
    /// conformance corpus leaves it off so its output stays exactly as before.
    pub discover_code_pointers: bool,
    /// Import a shared linear memory (`env.memory`) instead of defining one, so
    /// several instances of this module can share one guest address space while
    /// keeping independent register globals. Only the preemptive multi-thread
    /// scheduler (`vitaslop_native::ThreadedScheduler`) needs this; every single-
    /// instance host leaves it off and gets the original self-contained module.
    pub import_memory: bool,
}

/// A guest address that dispatches to a host import (the Vita NID mechanism): a
/// `bl`/`blx` to `addr` becomes a host call with dense index `import`.
pub struct Extern {
    pub addr: u32,
    pub import: u32,
}

/// An inter-module redirect: a `bl`/`b` to import stub `addr` is retargeted to
/// the guest function at `target` (an export of another module in the same link),
/// lowering to a direct guest call instead of a host import trap.
pub struct Redirect {
    pub addr: u32,
    pub target: u32,
}

/// The transpiler output: the WASM blob plus the map the runtime needs to enter
/// guest code by address.
pub struct Artifact {
    /// The emitted WASM module bytes.
    pub wasm: Vec<u8>,
    /// One entry per transpiled function: its guest address and wasm export name.
    pub funcs: Vec<FuncExport>,
    /// Total linear-memory pages the module declares (the guest region plus the
    /// appended indirect-dispatch address table). A host that imports a shared memory
    /// must create it with exactly this many pages; a host that lets the module define
    /// its own memory can ignore this. See [`emit::EmitOutput`].
    pub mem_pages: u32,
}

/// A transpiled function: the guest address it starts at and its wasm export.
pub struct FuncExport {
    pub addr: u32,
    pub export: String,
}

/// Why transpilation failed.
#[derive(Debug)]
pub enum Error {
    /// The decoder could not decode the bytes at `addr`.
    Decode { addr: u32 },
    /// A decoded instruction is not lifted yet.
    Unsupported {
        addr: u32,
        opcode: yaxpeax_arm::armv7::Opcode,
    },
    /// An operand had an unexpected shape for its opcode.
    Operand { addr: u32 },
}

/// The linker inserts Thumb->ARM interworking veneers so Thumb code (e.g. newlib,
/// linked without -nostdlib) can `bl` an ARM import stub. Each veneer has a fixed
/// five-word shape:
///   bx   pc          ; 0x4778     Thumb -> ARM, pc word-aligned to veneer+4
///   b.n  .-2         ; 0xe7fd     never executed (bx pc reads pc = veneer+4)
///   ldr  ip, [pc]    ; 0xe59fc000 ip = the offset word below
///   add  pc, ip, pc  ; 0xe08cf00f pc(here)+8 + ip  ->  stub = veneer+16+off
///   .word off
/// Resolve each veneer to its stub and, when the stub is a known import, return
/// `(veneer_addr, import_index)`. Aliasing the veneer's entry to that import lets
/// a `bl veneer` lower straight to the import call - no ARM lifting and no
/// computed-branch support needed. Artifacts with no veneers (our -nostdlib ones,
/// which `blx` the stub directly) yield nothing here, so this is purely additive.
fn scan_veneers(code: &[u8], base: u32, imports: &BTreeMap<u32, u32>) -> Vec<(u32, u32)> {
    let rd16 = |o: usize| u16::from_le_bytes([code[o], code[o + 1]]);
    let rd32 = |o: usize| u32::from_le_bytes([code[o], code[o + 1], code[o + 2], code[o + 3]]);
    let mut out = Vec::new();
    let mut o = 0usize;
    while o + 16 <= code.len() {
        if rd16(o) == 0x4778 && rd32(o + 4) == 0xe59f_c000 && rd32(o + 8) == 0xe08c_f00f {
            let veneer = base.wrapping_add(o as u32);
            let stub = veneer.wrapping_add(16).wrapping_add(rd32(o + 12));
            if let Some(&idx) = imports.get(&stub) {
                out.push((veneer, idx));
            }
        }
        o += 2;
    }
    out
}

/// Transpile `program` into a WASM module and its dispatch map.
/// One pending discovery target: its address, whether it is `tentative` (a guessed
/// code pointer that is dropped on a decode failure rather than erroring), and which
/// `thumb`/ARM mode to decode it in.
struct WorkItem {
    addr: u32,
    tentative: bool,
    thumb: bool,
}

impl WorkItem {
    /// A hard target (a decode failure is a real error), decoded in `thumb` mode.
    fn hard(addr: u32, thumb: bool) -> WorkItem {
        WorkItem { addr, tentative: false, thumb }
    }
    /// A tentative target (dropped on a decode failure), decoded in `thumb` mode.
    fn tentative(addr: u32, thumb: bool) -> WorkItem {
        WorkItem { addr, tentative: true, thumb }
    }
}

/// The initial discovery worklist: hard entries in the program's default mode, plus
/// tentative ARM entries (even code pointers reached through a `blx` into ARM code).
fn seed_worklist(program: &Program) -> Vec<WorkItem> {
    let mut work: Vec<WorkItem> =
        program.entries.iter().map(|&a| WorkItem::hard(a, program.thumb)).collect();
    work.extend(program.arm_entries.iter().map(|&a| WorkItem::tentative(a, false)));
    work
}

pub fn transpile(program: &Program) -> Result<Artifact, Error> {
    let mut import_map: BTreeMap<u32, u32> =
        program.externs.iter().map(|e| (e.addr, e.import)).collect();
    // Alias Thumb->ARM interworking veneers to the imports they trampoline to.
    for (veneer, idx) in scan_veneers(program.code, program.base, &import_map) {
        import_map.insert(veneer, idx);
    }
    let redirect_map: BTreeMap<u32, u32> =
        program.redirects.iter().map(|r| (r.addr, r.target)).collect();
    let imports = Imports::new(&import_map, &redirect_map);

    // Discover the transitive closure from the entries: direct callees are hard
    // (a decode failure is a real bug and propagates), while address-taken code
    // pointers are tentative (a mis-identified constant that fails to decode is
    // silently skipped, never breaking the build).
    let mut funcs: BTreeMap<u32, ir::Func> = BTreeMap::new();
    let mut work = seed_worklist(program);
    while let Some(WorkItem { addr, tentative, thumb }) = work.pop() {
        if funcs.contains_key(&addr) {
            continue;
        }
        let found = match lower::discover(
            program.code,
            program.base,
            addr,
            thumb,
            &imports,
            program.noreturn_svc,
            program.discover_code_pointers,
        ) {
            Ok(found) => found,
            // A tentative code pointer that does not decode was not a function;
            // drop it. A hard callee failure is a genuine error.
            Err(_) if tentative => continue,
            Err(e) => return Err(e),
        };
        // A tentative guess that decoded into a malformed function (a branch to a
        // non-block - common when ARM-decoding data or a Thumb address) was never a
        // real function; drop it so it never reaches emission.
        if tentative && !found.func.well_formed() {
            continue;
        }
        // Direct callees inherit this function's mode and its tentativeness (a
        // callee reached only from a guess is itself a guess); discovered code
        // pointers are tentative Thumb (materialized with the Thumb bit set).
        let callee: fn(u32, bool) -> WorkItem = if tentative { WorkItem::tentative } else { WorkItem::hard };
        work.extend(found.callees.into_iter().map(|a| callee(a, thumb)));
        work.extend(found.code_pointers.into_iter().map(|a| WorkItem::tentative(a, true)));
        funcs.insert(addr, found.func);
    }

    // Assign wasm function indices (imports occupy the low indices) in ascending
    // address order, and build the address -> index map for call lowering.
    let ordered: Vec<ir::Func> = funcs.into_values().collect();
    let func_index: BTreeMap<u32, u32> = ordered
        .iter()
        .enumerate()
        .map(|(i, f)| (f.addr, emit::IMPORT_FUNCS + i as u32))
        .collect();

    let emit::EmitOutput { wasm, mem_pages } =
        emit::emit_module(&ordered, &func_index, program.base, program.mem_bytes, program.import_memory);
    let funcs = ordered
        .iter()
        .map(|f| FuncExport {
            addr: f.addr,
            export: abi::func_export(f.addr),
        })
        .collect();
    Ok(Artifact { wasm, funcs, mem_pages })
}

/// The output of a lenient whole-program build ([`transpile_lenient`]): the module
/// plus the addresses that were emitted as trapping stubs.
pub struct LenientArtifact {
    pub artifact: Artifact,
    /// Guest addresses of functions that could not be lowered and became trapping
    /// stubs. Empty means a fully-translated program. Any of these reached at
    /// runtime faults (a real, visible signal that the code is on the hot path).
    pub stubbed: Vec<u32>,
    /// The wasm function indices of the stubs (parallel to [`stubbed`](Self::stubbed)),
    /// so a runtime `unreachable` trap backtrace ("wasm function N") can be told
    /// apart from a genuine miscompile - a stub is expected, anything else is a bug.
    pub stub_wasm_indices: Vec<u32>,
}

/// Transpile like [`transpile`], but never abort: a function that fails to lower
/// becomes a trapping stub instead of an error, so the whole program still emits a
/// valid, runnable module. This is what lets a real title boot while a handful of
/// exotic instructions remain unlifted - the boot path proceeds, and any stub it
/// actually calls faults loudly instead of silently mistranslating. Discovery from
/// a stubbed function's body stops (its callees are unknown), which is correct: we
/// could not decode past the unlifted instruction anyway.
pub fn transpile_lenient(program: &Program) -> LenientArtifact {
    let mut import_map: BTreeMap<u32, u32> =
        program.externs.iter().map(|e| (e.addr, e.import)).collect();
    for (veneer, idx) in scan_veneers(program.code, program.base, &import_map) {
        import_map.insert(veneer, idx);
    }
    let redirect_map: BTreeMap<u32, u32> =
        program.redirects.iter().map(|r| (r.addr, r.target)).collect();
    let imports = Imports::new(&import_map, &redirect_map);

    let mut funcs: BTreeMap<u32, ir::Func> = BTreeMap::new();
    let mut stubbed = Vec::new();
    let mut work = seed_worklist(program);
    while let Some(WorkItem { addr, tentative, thumb }) = work.pop() {
        if funcs.contains_key(&addr) {
            continue;
        }
        match lower::discover(
            program.code,
            program.base,
            addr,
            thumb,
            &imports,
            program.noreturn_svc,
            program.discover_code_pointers,
        ) {
            // A tentative guess that decoded into a malformed function (a branch to
            // a non-block) was never real; drop it before it can be emitted.
            Ok(found) if tentative && !found.func.well_formed() => {}
            Ok(found) => {
                let callee: fn(u32, bool) -> WorkItem =
                    if tentative { WorkItem::tentative } else { WorkItem::hard };
                work.extend(found.callees.into_iter().map(|a| callee(a, thumb)));
                work.extend(found.code_pointers.into_iter().map(|a| WorkItem::tentative(a, true)));
                funcs.insert(addr, found.func);
            }
            // A tentative code pointer that does not decode was never a function.
            Err(_) if tentative => {}
            // A hard callee we cannot lower becomes a trapping stub, so the rest of
            // the program still builds and runs.
            Err(_) => {
                stubbed.push(addr);
                funcs.insert(addr, ir::Func::new_stub(addr));
            }
        }
    }

    let ordered: Vec<ir::Func> = funcs.into_values().collect();
    let func_index: BTreeMap<u32, u32> = ordered
        .iter()
        .enumerate()
        .map(|(i, f)| (f.addr, emit::IMPORT_FUNCS + i as u32))
        .collect();
    let emit::EmitOutput { wasm, mem_pages } = emit::emit_module(
        &ordered,
        &func_index,
        program.base,
        program.mem_bytes,
        program.import_memory,
    );
    let funcs = ordered
        .iter()
        .map(|f| FuncExport { addr: f.addr, export: abi::func_export(f.addr) })
        .collect();
    stubbed.sort_unstable();
    let stub_wasm_indices = stubbed.iter().map(|a| func_index[a]).collect();
    LenientArtifact { artifact: Artifact { wasm, funcs, mem_pages }, stubbed, stub_wasm_indices }
}

/// A single function the report could not translate: the discovery root it was
/// reached from, and the error (whose `addr` pinpoints the offending instruction).
pub struct ReportFailure {
    pub root: u32,
    pub error: Error,
}

/// The result of a resilient, diagnostic-only discovery pass: how many functions
/// translated cleanly and the ones that failed.
pub struct Report {
    /// Guest addresses of functions that discovered and lowered without error.
    pub ok: Vec<u32>,
    /// Functions that failed to translate (skipped rather than aborting).
    pub failures: Vec<ReportFailure>,
}

/// Diagnostic: attempt to discover and lower every reachable function, but on a
/// failure record it and continue instead of aborting at the first one. Unlike
/// [`transpile`] this never emits a module and never returns `Err` - it is purely
/// for sizing what CPU work remains (which instructions block which functions),
/// so bring-up can see the whole gap at once rather than one instruction per run.
pub fn transpile_report(program: &Program) -> Report {
    let import_map: BTreeMap<u32, u32> =
        program.externs.iter().map(|e| (e.addr, e.import)).collect();
    let redirect_map: BTreeMap<u32, u32> =
        program.redirects.iter().map(|r| (r.addr, r.target)).collect();
    let imports = Imports::new(&import_map, &redirect_map);

    let mut done: std::collections::BTreeSet<u32> = std::collections::BTreeSet::new();
    let mut ok = Vec::new();
    let mut failures = Vec::new();
    let mut work = seed_worklist(program);
    while let Some(WorkItem { addr, tentative, thumb }) = work.pop() {
        if !done.insert(addr) {
            continue;
        }
        match lower::discover(
            program.code,
            program.base,
            addr,
            thumb,
            &imports,
            program.noreturn_svc,
            program.discover_code_pointers,
        ) {
            Ok(found) if tentative && !found.func.well_formed() => {}
            Ok(found) => {
                let callee: fn(u32, bool) -> WorkItem =
                    if tentative { WorkItem::tentative } else { WorkItem::hard };
                work.extend(found.callees.into_iter().map(|a| callee(a, thumb)));
                work.extend(found.code_pointers.into_iter().map(|a| WorkItem::tentative(a, true)));
                ok.push(addr);
            }
            // A tentative code pointer that does not decode was never a function.
            Err(_) if tentative => {}
            Err(error) => failures.push(ReportFailure { root: addr, error }),
        }
    }
    Report { ok, failures }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn transpiles_arm_hello() {
        // adr r0,msg / mov r1,#13 / svc #0 / svc #1  (base 0x10000, ARM).
        let code: [u8; 16] = [
            0x08, 0x00, 0x8f, 0xe2, // adr r0, msg
            0x0d, 0x10, 0xa0, 0xe3, // mov r1, #13
            0x00, 0x00, 0x00, 0xef, // svc #0
            0x01, 0x00, 0x00, 0xef, // svc #1
        ];
        let artifact = transpile(&Program {
            code: &code,
            base: 0x10000,
            thumb: false,
            entries: &[0x10000],
            arm_entries: &[],
            externs: &[],
            redirects: &[],
            noreturn_svc: &[],
            mem_bytes: 0x20000,
            discover_code_pointers: false,
            import_memory: false,
        })
        .expect("transpile");
        assert!(!artifact.wasm.is_empty());
        assert_eq!(artifact.funcs.len(), 1);
        assert_eq!(artifact.funcs[0].addr, 0x10000);
        assert_eq!(artifact.funcs[0].export, "f_10000");
        // The module must validate.
        wasmparser::validate(&artifact.wasm).expect("valid wasm");
    }

    #[test]
    fn transpiles_indirect_call() {
        // Thumb: `blx r0` (indirect call through a function pointer) then `bx lr`.
        // Exercises the indirect-call lowering + the module dispatcher emission.
        let code: [u8; 4] = [
            0x80, 0x47, // blx r0
            0x70, 0x47, // bx lr
        ];
        let artifact = transpile(&Program {
            code: &code,
            base: 0x10000,
            thumb: true,
            entries: &[0x10000],
            arm_entries: &[],
            externs: &[],
            redirects: &[],
            noreturn_svc: &[],
            mem_bytes: 0x20000,
            discover_code_pointers: false,
            import_memory: false,
        })
        .expect("transpile indirect");
        // One guest function plus the emitted dispatcher must produce valid wasm.
        assert_eq!(artifact.funcs.len(), 1);
        wasmparser::validate(&artifact.wasm).expect("valid wasm with dispatcher");
    }

    #[test]
    fn scan_veneers_resolves_interworking_stub() {
        // A Thumb->ARM interworking veneer trampolining to a stub 0x10 past it.
        // Layout at base: bx pc / b.n .-2 / ldr ip,[pc] / add pc,ip,pc / .word off.
        let mut code = vec![0u8; 0x40];
        let put16 = |c: &mut [u8], o: usize, v: u16| c[o..o + 2].copy_from_slice(&v.to_le_bytes());
        let put32 = |c: &mut [u8], o: usize, v: u32| c[o..o + 4].copy_from_slice(&v.to_le_bytes());
        put16(&mut code, 0x00, 0x4778); // bx pc
        put16(&mut code, 0x02, 0xe7fd); // b.n .-2
        put32(&mut code, 0x04, 0xe59f_c000); // ldr ip, [pc]
        put32(&mut code, 0x08, 0xe08c_f00f); // add pc, ip, pc
        put32(&mut code, 0x0c, 0x0000_0010); // .word 0x10 -> stub = base+16+16 = base+0x20
        let base = 0x8100_0000;
        let mut imports = BTreeMap::new();
        imports.insert(base + 0x20, 7u32); // the stub at base+0x20 -> import 7
        let found = scan_veneers(&code, base, &imports);
        assert_eq!(found, vec![(base, 7)]);
    }

    #[test]
    fn transpiles_cube_memset() {
        // The cube's Thumb memset at 0x81000948 (cbz, subs, strb[!], cmp, bne,
        // bx lr): a 4-block loop that stresses the dispatch machinery.
        let code: [u8; 18] = [
            0x32, 0xb1, 0x01, 0x3a, 0x43, 0x1e, 0x02, 0x44, 0x03, 0xf8, 0x01, 0x1f, 0x93, 0x42,
            0xfb, 0xd1, 0x70, 0x47,
        ];
        let base = 0x8100_0948;
        let artifact = transpile(&Program {
            code: &code,
            base,
            thumb: true,
            entries: &[base],
            arm_entries: &[],
            externs: &[],
            redirects: &[],
            noreturn_svc: &[],
            mem_bytes: 0x1_0000,
            discover_code_pointers: false,
            import_memory: false,
        })
        .expect("transpile memset");
        if let Err(e) = wasmparser::validate(&artifact.wasm) {
            dump_last_func(&artifact.wasm);
            panic!("invalid memset module: {e}");
        }
    }

    /// Print the operators of the last function body with a running control depth.
    fn dump_last_func(wasm: &[u8]) {
        use wasmparser::{Parser, Payload};
        let mut depth = 0i32;
        for payload in Parser::new(0).parse_all(wasm) {
            if let Ok(Payload::CodeSectionEntry(body)) = payload {
                let mut reader = body.get_operators_reader().unwrap();
                while let Ok(op) = reader.read() {
                    use wasmparser::Operator::*;
                    let before = depth;
                    match op {
                        Block { .. } | Loop { .. } | If { .. } => depth += 1,
                        End => depth -= 1,
                        _ => {}
                    }
                    eprintln!("depth {before}->{depth}: {op:?}");
                }
            }
        }
    }

    #[test]
    fn import_memory_mode_imports_shared_memory_and_validates() {
        // Same tiny ARM program, once self-contained and once importing a shared
        // memory: both must validate, and only the shared build declares the
        // `env.memory` import (shared, with a maximum).
        let code: [u8; 8] = [
            0x0d, 0x10, 0xa0, 0xe3, // mov r1, #13
            0x00, 0x00, 0x00, 0xef, // svc #0
        ];
        let mk = |import_memory| {
            transpile(&Program {
                code: &code,
                base: 0x10000,
                thumb: false,
                entries: &[0x10000],
                arm_entries: &[],
                externs: &[],
                redirects: &[],
                noreturn_svc: &[],
                mem_bytes: 0x20000,
                discover_code_pointers: false,
                import_memory,
            })
            .expect("transpile")
            .wasm
        };

        let plain = mk(false);
        let shared = mk(true);
        wasmparser::validate(&plain).expect("plain module valid");
        wasmparser::validate(&shared).expect("shared-memory module valid");

        // Walk imports: the shared build must import a shared memory named
        // `env.memory`; the plain build must not import any memory.
        let has_shared_mem_import = |wasm: &[u8]| {
            use wasmparser::{Parser, Payload, TypeRef};
            for payload in Parser::new(0).parse_all(wasm) {
                if let Ok(Payload::ImportSection(reader)) = payload {
                    for imp in reader.into_imports() {
                        let imp = imp.unwrap();
                        if let TypeRef::Memory(mt) = imp.ty {
                            assert_eq!(imp.module, "env");
                            assert_eq!(imp.name, "memory");
                            assert!(mt.shared, "imported memory must be shared");
                            assert!(mt.maximum.is_some(), "shared memory needs a maximum");
                            return true;
                        }
                    }
                }
            }
            false
        };
        assert!(has_shared_mem_import(&shared), "shared build imports env.memory");
        assert!(!has_shared_mem_import(&plain), "plain build imports no memory");
    }
}
