//! The multi-module linker: several loaded [`Module`]s into one unified program
//! image the transpiler compiles as a single wasm module.
//!
//! A real title is not one executable. A game is typically an `eboot.bin` plus
//! several shared libraries (`libc`, `libfios2`, `libsmart`, `libult`, `libface`
//! are common ones), and every one of them links at the same nominal base
//! (`0x8100_0000`) - they
//! overlap completely. To run them together this linker:
//!
//! 1. **Lays them out** at distinct, page-aligned bases in one shared guest
//!    address space, and [`rebase`](vitaslop_loader::Module::rebase)s each -
//!    applying its SCE relocations so its code and data point at the new base.
//! 2. **Resolves imports.** Each module imports functions by `(library, NID)`.
//!    An import another loaded module *exports* is wired as a [`Redirect`] - a
//!    direct guest-to-guest call, no host round-trip. Everything else is a system
//!    call the host services, wired as an [`Extern`] to a dense import index.
//! 3. **Assembles one image** (every segment blitted at its rebased address) and
//!    the transpiler [`Program`] inputs over it, with every module's entry point
//!    and static constructors as discovery roots.
//!
//! The whole set becomes one wasm module with one guest address space and one
//! register file - faithful to the single ARM core the Vita presents - so an
//! inter-module call is an ordinary direct call and shared globals resolve for
//! free.

use std::collections::HashMap;

use vitaslop_loader::{Module, Segment};
use vitaslop_transpiler::{Extern, Program, Redirect};

/// Guest base the first module is placed at (every Vita module's nominal link
/// base; the whole unified image begins here).
pub const IMAGE_BASE: u32 = 0x8100_0000;

/// Alignment between consecutive modules in the shared address space. A generous
/// 64 KiB gap keeps module images clearly separated (and page-aligned) so a
/// stray in-bounds decode can never wander from one module into the next.
const MODULE_ALIGN: u32 = 0x1_0000;

/// Total guest memory the unified program runs in (image + heap + stack), matching
/// the order of magnitude of a Vita game partition. Allocations begin above the
/// image (see [`LinkedProgram::alloc_base`]) and the stack descends from the top.
pub const GUEST_MEM_BYTES: u32 = 0x1000_0000; // 256 MiB

/// A linked, ready-to-transpile program: the combined image plus everything the
/// transpiler and the host environment need to compile and run it.
pub struct LinkedProgram {
    /// Base guest address of the whole image (== [`IMAGE_BASE`]).
    pub base: u32,
    /// The unified code+data image, byte `i` at guest address `base + i`.
    pub image: Vec<u8>,
    /// Guest memory to provision, in bytes from `base`.
    pub mem_bytes: u32,
    /// Discovery roots: every module entry point and static constructor.
    pub entries: Vec<u32>,
    /// ARM-mode discovery roots: even code pointers from relocated tables (see
    /// [`vitaslop_loader::Module::arm_code_pointers`]). Discovered as ARM, tentatively.
    pub arm_entries: Vec<u32>,
    /// Host-import wiring: each module import that no loaded module satisfies,
    /// mapped to a dense host-import index.
    pub externs: Vec<Extern>,
    /// Inter-module wiring: each import satisfied by another module's export,
    /// mapped to that export's address (a direct guest call).
    pub redirects: Vec<Redirect>,
    /// `(library_nid, func_nid)` per dense host-import index, for the host
    /// environment's NID dispatch (parallel to [`externs`](Self::externs)).
    pub imports: Vec<(u32, u32)>,
    /// Each module's `module_start` entry, in load order (shared libraries first,
    /// then the eboot): the host runs these before the main entry so a library's
    /// constructors and TLS run before anything calls into it.
    pub module_inits: Vec<u32>,
    /// The main executable's entry point (the eboot's `module_start`).
    pub main_entry: u32,
    /// Guest address of the main module's `SceProcessParam` (the "PSP2" block that
    /// libc's `module_start` fetches via `sceKernelGetProcessParam` to read the
    /// `SceLibcParam` heap configuration), or 0 if the module carries none.
    pub process_param: u32,
    /// Guest address above the whole image where host allocations may begin.
    pub alloc_base: u32,
    /// Imports that resolved to neither a loaded export nor - as far as the linker
    /// knows - a host handler are still wired as host imports; this records them
    /// for diagnostics (a still-missing NID surfaces here and in the capture).
    pub host_import_count: usize,
}

impl LinkedProgram {
    /// Borrow the linked image as a transpiler [`Program`]. The whole program is
    /// Thumb-2 (Vita user code, confirmed by its Thumb-only relocations); code
    /// pointers are discovered so address-taken thread entries and callbacks are
    /// translated.
    pub fn program(&self) -> Program<'_> {
        self.program_with(false)
    }

    /// Like [`program`](Self::program) but with an imported **shared** memory, for
    /// the preemptive multi-thread scheduler (every thread instance imports one
    /// shared linear memory - see [`vitaslop_transpiler::Program::import_memory`]).
    pub fn shared_program(&self) -> Program<'_> {
        self.program_with(true)
    }

    fn program_with(&self, import_memory: bool) -> Program<'_> {
        Program {
            code: &self.image,
            base: self.base,
            thumb: true,
            entries: &self.entries,
            arm_entries: &self.arm_entries,
            externs: &self.externs,
            redirects: &self.redirects,
            noreturn_svc: &[],
            mem_bytes: self.mem_bytes,
            discover_code_pointers: true,
            import_memory,
        }
    }
}

/// Round `v` up to a multiple of `align` (a power of two).
fn align_up(v: u32, align: u32) -> u32 {
    (v + align - 1) & !(align - 1)
}

/// Link `modules` (in load order - shared libraries first, the main executable
/// last) into one program image.
///
/// Each module is placed at the next free page-aligned base and relocated there.
/// Returns an error if a relocation code is unsupported or a module is malformed;
/// an unresolved import is *not* an error - it becomes a host import so the run
/// proceeds and the gap is visible in the capture.
pub fn link(mut modules: Vec<Module>) -> Result<LinkedProgram, vitaslop_loader::Error> {
    // 1. Assign each module a distinct base and relocate it there.
    let mut cursor = IMAGE_BASE;
    for m in &mut modules {
        let span = m.image_end().wrapping_sub(m.base);
        m.rebase(cursor)?;
        cursor = align_up(cursor.wrapping_add(span), MODULE_ALIGN);
    }
    let image_end = cursor;

    // 2. Build the global export table, keyed by (library, NID). A function is
    //    reached across modules by this pair; the address carries the Thumb bit.
    let mut exports: HashMap<(u32, u32), u32> = HashMap::new();
    for m in &modules {
        for e in &m.exports {
            exports.entry((e.library_nid, e.func_nid)).or_insert(e.addr);
        }
    }

    // 3. Assemble the combined image: every segment at its rebased address.
    let mut image = vec![0u8; image_end.wrapping_sub(IMAGE_BASE) as usize];
    for m in &modules {
        for s in &m.segments {
            blit(&mut image, IMAGE_BASE, s);
        }
    }

    // 4. Resolve every module's imports: an exported one becomes a direct
    //    inter-module redirect, everything else a host import.
    let mut externs = Vec::new();
    let mut redirects = Vec::new();
    let mut imports = Vec::new();
    for m in &modules {
        for imp in &m.imports {
            if let Some(&target) = exports.get(&(imp.library_nid, imp.func_nid)) {
                // The export address carries the Thumb bit (any ARM function
                // pointer does); the transpiler decodes at the even address.
                redirects.push(Redirect { addr: imp.stub_addr, target: target & !1 });
            } else {
                externs.push(Extern { addr: imp.stub_addr, import: imports.len() as u32 });
                imports.push((imp.library_nid, imp.func_nid));
            }
        }
    }

    // 5. Discovery roots: every module entry and constructor. The entry addresses
    //    are even (SCE stores module_start without the Thumb bit); decoding is
    //    Thumb regardless.
    let mut entries = Vec::new();
    let mut arm_entries = Vec::new();
    let mut module_inits = Vec::new();
    for m in &modules {
        entries.push(m.entry & !1);
        module_inits.push(m.entry & !1);
        for &p in &m.init_pointers {
            entries.push(p & !1);
        }
        // Function pointers sitting in relocated data tables (vtables, callback
        // arrays) - reached only via indirect `blx`/`bx`, so they must be seeded
        // as discovery roots or the dispatcher would trap on them at runtime.
        for &p in &m.code_pointers {
            let off = p.wrapping_sub(IMAGE_BASE);
            if (off as usize) < image.len() {
                entries.push(p);
            }
        }
        // Even code pointers are ARM-mode functions (a `blx` to an even address a
        // title reaches through a relocated table). Seeded as tentative ARM
        // discovery roots so the dispatcher can resolve them instead of trapping.
        for &p in &m.arm_code_pointers {
            let off = p.wrapping_sub(IMAGE_BASE);
            if (off as usize) < image.len() {
                arm_entries.push(p);
            }
        }
    }
    // A Thumb function's address must never be seeded as ARM: an even relocated
    // pointer that happens to equal a known Thumb entry would otherwise decode that
    // function as ARM (garbage). Thumb discovery wins - drop any such ARM candidate.
    let thumb_roots: std::collections::HashSet<u32> = entries.iter().copied().collect();
    arm_entries.retain(|a| !thumb_roots.contains(a));
    arm_entries.sort_unstable();
    arm_entries.dedup();

    // The main executable is the last module in load order (the eboot).
    let main_entry = modules.last().map(|m| m.entry & !1).unwrap_or(0);
    let process_param = modules
        .last()
        .and_then(|m| find_process_param(&image, IMAGE_BASE, m))
        .unwrap_or(0);

    let host_import_count = imports.len();
    Ok(LinkedProgram {
        base: IMAGE_BASE,
        image,
        mem_bytes: GUEST_MEM_BYTES,
        entries,
        arm_entries,
        externs,
        redirects,
        imports,
        module_inits,
        main_entry,
        alloc_base: align_up(image_end, MODULE_ALIGN),
        host_import_count,
        process_param,
    })
}

/// Magic at `SceProcessParam + 4`: the ASCII "PSP2" that identifies the block.
const PROCESS_PARAM_MAGIC: u32 = 0x3250_5350;

/// Locate the main module's `SceProcessParam` in the combined image by its magic.
///
/// The block is a fixed structure the toolchain emits into the main executable's
/// data; `sceKernelGetProcessParam` hands its address to libc so the crt can read
/// the `SceLibcParam` (heap size, malloc replacement). The magic word uniquely
/// identifies it, and the search is confined to this module's own segments (so no
/// unrelated data can collide) and validated by version and a sane size.
fn find_process_param(image: &[u8], image_base: u32, m: &Module) -> Option<u32> {
    for s in &m.segments {
        let off = s.vaddr.wrapping_sub(image_base) as usize;
        let end = (off + s.data.len()).min(image.len());
        let seg = image.get(off..end)?;
        let mut i = 0;
        while i + 0x10 <= seg.len() {
            let magic = u32::from_le_bytes([seg[i + 4], seg[i + 5], seg[i + 6], seg[i + 7]]);
            if magic == PROCESS_PARAM_MAGIC {
                let size = u32::from_le_bytes([seg[i], seg[i + 1], seg[i + 2], seg[i + 3]]);
                let ver = u32::from_le_bytes([seg[i + 8], seg[i + 9], seg[i + 10], seg[i + 11]]);
                // A real block is version 6 and a small header (0x30..=0x40 bytes).
                if ver == 6 && (0x28..=0x48).contains(&size) {
                    return Some(image_base + (off + i) as u32);
                }
            }
            i += 4; // the block is word-aligned
        }
    }
    None
}

/// Copy a segment's file-backed bytes into the combined image at its rebased
/// address (the `.bss` tail past `data.len()` stays zero).
fn blit(image: &mut [u8], image_base: u32, seg: &Segment) {
    let off = seg.vaddr.wrapping_sub(image_base) as usize;
    let end = off + seg.data.len();
    if end <= image.len() {
        image[off..end].copy_from_slice(&seg.data);
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::ingest::pipeline::decrypt_container;
    use crate::ingest::testfix;
    use crate::ingest::vfs::DirVfs;

    /// Load every module from a privately-supplied dump, link them, and check the
    /// structural invariants the whole downstream pipeline depends on. Skips without
    /// the fixture; all assertions are title-independent.
    #[test]
    fn links_fixture_modules() {
        let Some(dir) = testfix::game_dir() else {
            return;
        };
        let game = decrypt_container(&DirVfs::new(dir)).expect("decrypt");
        let modules: Vec<Module> = game
            .modules
            .iter()
            .map(|m| vitaslop_loader::load(&m.elf).expect("load module"))
            .collect();
        let n_modules = modules.len();
        let linked = link(modules).expect("link");

        // Every module got a distinct, non-overlapping, page-aligned base.
        assert_eq!(linked.base, IMAGE_BASE);
        assert_eq!(linked.module_inits.len(), n_modules);
        let mut sorted = linked.module_inits.clone();
        sorted.sort();
        for w in sorted.windows(2) {
            assert!(w[0] != w[1], "two modules share an init address");
        }

        // The inter-module set is real: a game's eboot imports SceLibc /
        // SceLibm / SceLibstdcxx (libc.suprx) and SceFios2 (libfios2.suprx), so a
        // substantial number of imports must resolve to redirects, not host
        // imports. (A real title had 64 such imports across the eboot; other modules
        // add their own cross-references.)
        assert!(
            linked.redirects.len() >= 60,
            "expected many inter-module redirects, got {}",
            linked.redirects.len()
        );

        // Host imports and redirects together cover every module's imports, and
        // the host-import table is consistent with its extern wiring.
        assert_eq!(linked.imports.len(), linked.externs.len());
        assert_eq!(linked.imports.len(), linked.host_import_count);

        // The image spans past the eboot and allocations begin above it.
        assert!(linked.alloc_base > linked.base);
        assert!(linked.image.len() as u32 <= linked.mem_bytes);
    }

    /// The whole linked program must emit one valid wasm module (with the handful of
    /// still-unlifted functions as trapping stubs). This is the emit-side counterpart
    /// to `transpiles_fixture` (which only exercises decode/lower): it proves the
    /// computed-jump, VFP-double, and NEON-logical emission all validate together on
    /// the real 800+-function image, not just on unit cases.
    #[test]
    #[ignore = "needs fixture"]
    fn fixture_emits_valid_wasm() {
        let Some(dir) = testfix::game_dir() else {
            return;
        };
        let game = decrypt_container(&DirVfs::new(dir)).expect("decrypt");
        let modules: Vec<Module> = game
            .modules
            .iter()
            .map(|m| vitaslop_loader::load(&m.elf).expect("load module"))
            .collect();
        let linked = link(modules).expect("link");
        let built = vitaslop_transpiler::transpile_lenient(&linked.program());
        eprintln!(
            "emitted {} KiB wasm, {} functions, {} trapping stubs",
            built.artifact.wasm.len() / 1024,
            built.artifact.funcs.len(),
            built.stubbed.len(),
        );
        wasmparser::validate(&built.artifact.wasm).expect("linked wasm must validate");
        // Stubs are the still-unlifted remainder (dominated by NEON structure
        // load/store); they must stay a small fraction of the whole program.
        let total = built.artifact.funcs.len();
        assert!(
            built.stubbed.len() * 20 < total,
            "too many stubs: {}/{total}",
            built.stubbed.len(),
        );
    }

    /// Diagnostic: attempt to transpile the whole linked program and report how far
    /// it gets - success, or the exact address and opcode of the first instruction the
    /// transpiler cannot yet handle. This is the signal for what CPU work (NEON
    /// decode/lift, mixed ARM/Thumb, ...) remains. Ignored; run with
    /// `--ignored --nocapture`.
    #[test]
    #[ignore = "diagnostic: needs fixture"]
    fn transpiles_fixture() {
        let Some(dir) = testfix::game_dir() else {
            return;
        };
        let game = decrypt_container(&DirVfs::new(dir)).expect("decrypt");
        let modules: Vec<Module> = game
            .modules
            .iter()
            .map(|m| vitaslop_loader::load(&m.elf).expect("load module"))
            .collect();
        let linked = link(modules).expect("link");
        eprintln!(
            "linked: image={} KiB entries={} host_imports={} redirects={}",
            linked.image.len() / 1024,
            linked.entries.len(),
            linked.imports.len(),
            linked.redirects.len(),
        );
        // Resilient pass: translate every reachable function, bucketing the ones
        // that fail so the whole remaining CPU gap is visible at once.
        use std::collections::BTreeMap;
        let report = vitaslop_transpiler::transpile_report(&linked.program());
        eprintln!(
            "report: {} functions translated, {} failed",
            report.ok.len(),
            report.failures.len()
        );
        // Bucket failures by the instruction that blocked each: error kind, the
        // coprocessor/SIMD family (the high byte of the first Thumb halfword), and
        // the exact first halfword.
        let mut by_family: BTreeMap<(&str, u8), usize> = BTreeMap::new();
        let mut by_hw1: BTreeMap<(&str, u16), (usize, u32)> = BTreeMap::new();
        for f in &report.failures {
            let (kind, addr) = match f.error {
                vitaslop_transpiler::Error::Decode { addr } => ("decode", addr),
                vitaslop_transpiler::Error::Unsupported { addr, .. } => ("unsupported", addr),
                vitaslop_transpiler::Error::Operand { addr } => ("operand", addr),
            };
            let off = (addr & !1).wrapping_sub(linked.base) as usize;
            if off + 2 > linked.image.len() {
                continue;
            }
            let hw1 = u16::from_le_bytes([linked.image[off], linked.image[off + 1]]);
            *by_family.entry((kind, (hw1 >> 8) as u8)).or_default() += 1;
            by_hw1.entry((kind, hw1)).or_insert((0, addr & !1)).0 += 1;
        }
        eprintln!("-- failures by (kind, hw1>>8 family) --");
        for ((kind, fam), n) in &by_family {
            eprintln!("  {kind:<12} family {fam:#04x}: {n}");
        }
        eprintln!("-- top distinct first-halfwords --");
        let mut rows: Vec<_> = by_hw1.iter().collect();
        rows.sort_by_key(|(_, (n, _))| std::cmp::Reverse(*n));
        for ((kind, hw1), (n, example)) in rows.iter().take(25) {
            eprintln!("  {kind:<12} hw1={hw1:#06x} x{n:<5} e.g. {example:#010x}");
        }
        // For the "unsupported" failures (decoded but not lifted), the error
        // carries the exact opcode - the precise list of what to add to the lifter.
        let mut by_op: BTreeMap<String, usize> = BTreeMap::new();
        for f in &report.failures {
            if let vitaslop_transpiler::Error::Unsupported { opcode, .. } = &f.error {
                *by_op.entry(format!("{opcode:?}")).or_default() += 1;
            }
        }
        eprintln!("-- unsupported opcodes (exact) --");
        let mut ops: Vec<_> = by_op.iter().collect();
        ops.sort_by_key(|(_, n)| std::cmp::Reverse(**n));
        for (op, n) in ops {
            eprintln!("  {op:<20} x{n}");
        }
    }
}
