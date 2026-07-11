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
    /// True to decode in Thumb-2 mode, false for ARM. Whole-program for now;
    /// per-function `blx` mode switches come later.
    pub thumb: bool,
    /// Entry points to discover (each becomes a function, transitively pulling
    /// in its direct callees).
    pub entries: &'a [u32],
    /// Extern (imported-function) wiring: each maps a guest stub address to a
    /// dense host-import index.
    pub externs: &'a [Extern],
    /// Syscall numbers (guest r7) that do not return, so a `svc` with a
    /// statically-known one of them ends decoding (before trailing data).
    pub noreturn_svc: &'a [u32],
    /// Total guest memory to provision, in bytes from `base`. The host keeps all
    /// guest allocations (image, stack, heap) within `[base, base + mem_bytes)`.
    pub mem_bytes: u32,
}

/// A guest address that dispatches to a host import (the Vita NID mechanism): a
/// `bl`/`blx` to `addr` becomes a host call with dense index `import`.
pub struct Extern {
    pub addr: u32,
    pub import: u32,
}

/// The transpiler output: the WASM blob plus the map the runtime needs to enter
/// guest code by address.
pub struct Artifact {
    /// The emitted WASM module bytes.
    pub wasm: Vec<u8>,
    /// One entry per transpiled function: its guest address and wasm export name.
    pub funcs: Vec<FuncExport>,
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

/// Transpile `program` into a WASM module and its dispatch map.
pub fn transpile(program: &Program) -> Result<Artifact, Error> {
    let import_map: BTreeMap<u32, u32> =
        program.externs.iter().map(|e| (e.addr, e.import)).collect();
    let imports = Imports::new(&import_map);

    // Discover the transitive direct-call closure from the entries.
    let mut funcs: BTreeMap<u32, ir::Func> = BTreeMap::new();
    let mut work: Vec<u32> = program.entries.to_vec();
    while let Some(addr) = work.pop() {
        if funcs.contains_key(&addr) {
            continue;
        }
        let found = lower::discover(
            program.code,
            program.base,
            addr,
            program.thumb,
            &imports,
            program.noreturn_svc,
        )?;
        work.extend(found.callees);
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

    let wasm = emit::emit_module(&ordered, &func_index, program.base, program.mem_bytes);
    let funcs = ordered
        .iter()
        .map(|f| FuncExport {
            addr: f.addr,
            export: abi::func_export(f.addr),
        })
        .collect();
    Ok(Artifact { wasm, funcs })
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
            externs: &[],
            noreturn_svc: &[],
            mem_bytes: 0x20000,
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
            externs: &[],
            noreturn_svc: &[],
            mem_bytes: 0x1_0000,
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
}
