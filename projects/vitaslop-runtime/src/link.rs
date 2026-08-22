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
/// the order of magnitude of a Vita game partition (the console has 512 MiB of
/// physical LPDDR2, most of which a game may map). Allocations begin above the image
/// (see [`LinkedProgram::alloc_base`]) and the stack descends from the top; the
/// indirect-dispatch address table is appended immediately above this ceiling and is
/// protected by the [`VitaState::galloc`](crate::host::VitaState::galloc) cap. A 3D
/// title can need well over 256 MiB of heap before its first frame.
pub const GUEST_MEM_BYTES: u32 = 0x2000_0000; // 512 MiB

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
    /// The subset of [`imports`](Self::imports) whose behaviour is one guest-memory
    /// read, emitted inline by the transpiler instead of trapping to the host. Derived
    /// from the import table here, because only the runtime knows what a NID means -
    /// see [`crate::vita::gxm::inline_op`].
    pub inline_imports: Vec<vitaslop_transpiler::InlineImport>,
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
    /// The main executable's thread-local-storage template: `(init image address,
    /// initialized byte count, full per-thread block size)`, taken from the eboot's
    /// `SceModuleInfo`. The host gives each thread its own copy of this block and
    /// sets the thread pointer (`TPIDRURO`, read by `MRC p15,0,Rt,c13,c0,3`) to its
    /// base, so `__thread` accesses at `thread_pointer + offset` land in per-thread
    /// memory. `(0, 0, 0)` when the main module has no TLS.
    pub tls_template: (u32, u32, u32),
    /// Guest address above the whole image where host allocations may begin.
    pub alloc_base: u32,
    /// Imports that resolved to neither a loaded export nor - as far as the linker
    /// knows - a host handler are still wired as host imports; this records them
    /// for diagnostics (a still-missing NID surfaces here and in the capture).
    pub host_import_count: usize,
    /// Variable (data-symbol) imports no loaded module exported, as `(library_nid,
    /// var_nid)`. Empty on a clean link. Unlike an unresolved function import (a host
    /// stub that traps when called), an unresolved variable silently leaves a garbage
    /// pointer/value in the image, so a consumer (probe, front-end) must surface this
    /// list - a non-empty one means some data table the guest reads is unbound.
    pub unresolved_var_imports: Vec<(u32, u32)>,
    /// Every module in the finished image, in load order, with the placed addresses
    /// its segments ended up at. The kernel's module queries (`GetModuleInfoByAddr`,
    /// `GetModuleIdByAddr`) answer from this: a guest address maps to whichever
    /// module's segments contain it. Held here because only the linker knows where a
    /// relocatable module was finally placed.
    pub loaded_modules: Vec<LoadedModule>,
}

/// One module as it sits in the linked image: what the kernel reports about it.
#[derive(Clone)]
pub struct LoadedModule {
    /// Module name from `SceModuleInfo` (e.g. `"eboot.bin"`, `"SceLibc"`).
    pub name: String,
    pub module_nid: u32,
    /// `SceModuleInfo::module_start`, carrying the Thumb bit as any ARM function
    /// pointer does.
    pub entry: u32,
    /// The placed segments: `(vaddr, mem_size, file_size, executable, writable)`.
    pub segments: Vec<(u32, u32, u32, bool, bool)>,
}

impl LoadedModule {
    /// Whether `addr` falls inside any of this module's placed segments.
    pub fn contains(&self, addr: u32) -> bool {
        self.segments.iter().any(|&(vaddr, mem_size, ..)| {
            addr >= vaddr && addr < vaddr.wrapping_add(mem_size)
        })
    }
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
            inline_imports: &self.inline_imports,
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
    // 1. Assign each module a base and relocate it there, in two passes.
    //
    //    A fixed `ET_SCE_EXEC` image (a launch-title eboot) has no relocations and
    //    absolute internal pointers, so it can only load at its native base -
    //    `IMAGE_BASE`. A relocatable `ET_SCE_RELEXEC` library can go anywhere. So
    //    we pin every fixed module at its native base first (reserving those
    //    ranges), then lay the relocatable modules into the free space above. This
    //    keeps the eboot at `0x8100_0000` while its bundled `.suprx` libraries -
    //    which the naive single cursor would have collided onto that same base -
    //    are placed after it.
    let span_of = |m: &Module| m.image_end().wrapping_sub(m.base);
    let mut cursor = IMAGE_BASE;
    for m in modules.iter_mut().filter(|m| !m.relocatable) {
        let native = m.base;
        m.rebase(native)?; // delta 0: validated, no shift of a fixed image
        cursor = cursor.max(align_up(native.wrapping_add(span_of(m)), MODULE_ALIGN));
    }
    for m in modules.iter_mut().filter(|m| m.relocatable) {
        let span = span_of(m);
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

    // 3b. Resolve variable (data-symbol) imports. Unlike a function import - a
    //     single call stub redirected to the callee - a variable import carries a
    //     fixup blob listing every code/data site that references the symbol. We
    //     bind each site to the exporting module's copy of the variable, exactly as
    //     the SCE loader does. This matters for statically-linked titles that import
    //     the C library's ctype tables (`_Tolotab`/`_Ctype`/`_Touptab`) and stdio
    //     handles as variables: unresolved, `tolower()` reads a garbage table and
    //     every case-folded path (archive names, config keys) comes out corrupt.
    //     Patch the image in place before transpilation so the decoder sees the
    //     resolved immediates.
    let mut var_exports: HashMap<(u32, u32), u32> = modules
        .iter()
        .flat_map(|m| m.var_exports.iter())
        .map(|e| ((e.library_nid, e.var_nid), e.addr))
        .collect();
    // Special case: `__sce_libcparam` (SceLibcParam library 0x5ad9c136, var
    // 0xdf084dfa). A statically-linked SceLibc imports this variable to size and
    // configure its heap (heap size, extension policy, optional malloc replacement),
    // but the main module does not export it through the normal export table - the
    // real kernel resolves it to `SceProcessParam->sce_libcparam`. Mirror that: read
    // the pointer out of the main module's SceProcessParam and register it as the
    // export so the normal fixup loop binds SceLibc's import to the real struct.
    // Unresolved, SceLibc reads a null/garbage param and falls back to a default heap
    // whose layout differs from the title's, which misplaces later allocations.
    if let Some(pp) = modules.last().and_then(|m| find_process_param(&image, IMAGE_BASE, m)) {
        let off = pp.wrapping_sub(IMAGE_BASE) as usize;
        let rd = |o: usize| -> u32 {
            image.get(o..o + 4).map(|b| u32::from_le_bytes([b[0], b[1], b[2], b[3]])).unwrap_or(0)
        };
        let ver = rd(off + 8);
        // v6+ carries the SceLibcParam pointer at +0x28; v5 (and earlier) one word
        // later at +0x2c. Pick whichever slot holds a non-zero in-image pointer.
        let (a, b) = if ver >= 6 { (rd(off + 0x28), rd(off + 0x2c)) } else { (rd(off + 0x2c), rd(off + 0x28)) };
        let libcparam = if a != 0 { a } else { b };
        if libcparam != 0 {
            var_exports.entry((0x5ad9_c136, 0xdf08_4dfa)).or_insert(libcparam);
        }
    }
    let mut var_fixups_applied = 0u32;
    let mut unresolved_var_imports: Vec<(u32, u32)> = Vec::new();
    // Opt-in: bind every unresolved variable-import site to a poison address well outside
    // the guest region, so the FIRST guest access of the missing symbol traps loudly
    // (MemoryOutOfBounds, with the recognizable 0xE000_xxxx pointer in the reg dump)
    // instead of silently reading whatever the image left in the slot. This converts the
    // exact silent-corruption class that hid the `_Tolotab`/ctype-table gap for six
    // sessions into an immediate, addressed fault. Left off by default because a title may
    // read a genuinely-unused import's value without dereferencing it; on when hunting a
    // suspected data-import corruption.
    let poison_unresolved = std::env::var("VITASLOP_POISON_UNRESOLVED_VARS").is_ok();
    for m in &modules {
        for vi in &m.var_imports {
            match var_exports.get(&(vi.library_nid, vi.var_nid)) {
                Some(&sym) => {
                    apply_var_fixups(&mut image, m.base, vi.blob_ptr, sym, &mut var_fixups_applied)?
                }
                None => {
                    if poison_unresolved {
                        // Distinct per unresolved symbol so a trap names which one.
                        let poison = 0xE000_0000u32.wrapping_add(unresolved_var_imports.len() as u32 * 4);
                        apply_var_fixups(&mut image, m.base, vi.blob_ptr, poison, &mut var_fixups_applied)?;
                    }
                    unresolved_var_imports.push((vi.library_nid, vi.var_nid));
                }
            }
        }
    }
    if !unresolved_var_imports.is_empty() {
        // A missing variable export is a real gap: unlike a function import (which becomes
        // a host stub that traps loudly when CALLED), an unresolved variable silently
        // leaves a garbage pointer/value in the image that reads with no trap. Name each
        // one - the (library, symbol) NID pair is enough to look the symbol up in the NID
        // db and see immediately what data table is unbound (e.g. `_Tolotab` -> the ctype
        // tables). Never fold this to a bare count: a count is the silent failure.
        eprintln!(
            "link: WARNING {} variable import(s) unresolved (no matching export){}:",
            unresolved_var_imports.len(),
            if poison_unresolved { " - bound to poison, first access will trap" } else { "" },
        );
        for (lib, var) in &unresolved_var_imports {
            eprintln!("  unresolved var import: library_nid={lib:#010x} var_nid={var:#010x}");
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
                redirects.push(Redirect {
                    addr: imp.stub_addr,
                    target: target & !1,
                    thumb: target & 1 == 1,
                });
            } else {
                externs.push(Extern { addr: imp.stub_addr, import: imports.len() as u32 });
                imports.push((imp.library_nid, imp.func_nid));
            }
        }
    }

    // Which host imports we have no NAME for. A function import that no handler covers
    // becomes a stub that hard-fails when CALLED, which is correct but serialises the work:
    // each run reveals exactly one missing NID, so bringing up a title costs one boot per
    // call. The imports are all known HERE, at link time, so the whole list can be reported
    // at once and implemented in one pass.
    //
    // `nid::name` is the test because a NID gets its name in the same change that gives it
    // a handler, so an unnamed import is an unimplemented one. It is a HINT, not a
    // guarantee: a named NID with no dispatch arm still hard-fails at the call, which is
    // why the call-time failure stays exactly as loud as it was.
    let unnamed: Vec<(u32, u32)> = {
        let mut v: Vec<(u32, u32)> = imports
            .iter()
            .copied()
            .filter(|(_, func)| crate::nid::name(*func) == crate::nid::UNKNOWN_NAME)
            .collect();
        v.sort_unstable();
        v.dedup();
        v
    };
    if !unnamed.is_empty() {
        eprintln!(
            "link: {} imported function NID(s) have no handler - the title will hard-fail if it \
             CALLS one. Look each up in the vitasdk NID db (db/360/*.yml) and implement it:",
            unnamed.len()
        );
        for (lib, func) in &unnamed {
            eprintln!("  unhandled import: library_nid={lib:#010x} func_nid={func:#010x}");
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
    // The main executable's TLS template drives per-thread thread-local storage. A
    // shared library's own TLS is reached through the key-based `sceKernelGetTLSAddr`
    // path instead, so only the eboot's static template is needed here.
    let tls_template = modules
        .last()
        .filter(|m| m.tls_memsz != 0)
        .map(|m| (m.tls_vaddr, m.tls_filesz, m.tls_memsz))
        .unwrap_or((0, 0, 0));

    if std::env::var("VITASLOP_DUMP_EXPORTS").is_ok() {
        for (mi, m) in modules.iter().enumerate() {
            eprintln!(
                "  module[{mi}] base={:#x} exports={} var_exports={} var_imports={}",
                m.base,
                m.exports.len(),
                m.var_exports.len(),
                m.var_imports.len()
            );
            for ve in &m.var_exports {
                eprintln!(
                    "    var_export lib={:#010x} nid={:#010x} addr={:#x}",
                    ve.library_nid, ve.var_nid, ve.addr
                );
            }
            // Distinct export library NIDs this module provides.
            let mut libs: Vec<u32> = m.exports.iter().map(|e| e.library_nid).collect();
            libs.sort_unstable();
            libs.dedup();
            eprintln!("    export libs: {:?}", libs.iter().map(|l| format!("{l:#010x}")).collect::<Vec<_>>());
        }
        // SceProcessParam+0x28 = pointer to SceLibcParam (__sce_libcparam).
        if process_param != 0 {
            let off = process_param.wrapping_sub(IMAGE_BASE) as usize;
            let rd = |o: usize| -> u32 {
                if o + 4 <= image.len() {
                    u32::from_le_bytes([image[o], image[o + 1], image[o + 2], image[o + 3]])
                } else {
                    0
                }
            };
            eprintln!(
                "  process_param={:#x} size={:#x} ver={:#x} libcparam_ptr(+0x28)={:#x} (+0x2c)={:#x}",
                process_param,
                rd(off),
                rd(off + 8),
                rd(off + 0x28),
                rd(off + 0x2c)
            );
        }
    }

    // Snapshot each module's placed identity and segments before the modules are
    // dropped. Taken here, after rebasing, so the addresses are the ones the guest
    // will actually run at.
    let loaded_modules: Vec<LoadedModule> = modules
        .iter()
        .map(|m| LoadedModule {
            name: m.name.clone(),
            module_nid: m.module_nid,
            entry: m.entry,
            segments: m
                .segments
                .iter()
                .map(|s| (s.vaddr, s.mem_size, s.data.len() as u32, s.executable, s.writable))
                .collect(),
        })
        .collect();

    let host_import_count = imports.len();
    // Which host imports the transpiler may emit inline. Derived from the finished
    // import table so the index a call site uses and the index carrying the inline op
    // are the same by construction.
    let inline_imports: Vec<vitaslop_transpiler::InlineImport> = imports
        .iter()
        .enumerate()
        .filter_map(|(i, &(_, func_nid))| {
            crate::vita::inline_op(func_nid)
                .map(|op| vitaslop_transpiler::InlineImport { import: i as u32, op })
        })
        .collect();
    // >>> WHICH CALLS WERE EMITTED AS A BARE CONSTANT, SAID ONCE, HERE.
    //
    // An inlined call never reaches the host, so it is absent from the call histogram and the
    // host-call trace - and for a constant-return STUB that is the one place anyone would look
    // to find out that a title depends on a call nothing implements. Every other inline form
    // replaces a handler that computes the same answer; these replace a handler that computes
    // nothing, which is exactly what someone auditing coverage wants to see. So the list is
    // printed rather than left to be inferred from an empty count, and
    // `VITASLOP_NO_INLINE_STUBS` puts them all back on the host.
    let stubbed: Vec<&'static str> = {
        let mut v: Vec<&'static str> = inline_imports
            .iter()
            .filter(|i| matches!(i.op, vitaslop_transpiler::InlineOp::RetConst { .. }))
            .map(|i| crate::nid::name(imports[i.import as usize].1))
            .collect();
        v.sort_unstable();
        v.dedup();
        v
    };
    if !stubbed.is_empty() {
        tracing::info!(
            target: "vitaslop::link",
            "link: {} constant-return stub(s) emitted INLINE, so they will not appear in the \
             host-call histogram or trace: {}",
            stubbed.len(),
            stubbed.join(", "),
        );
    }

    Ok(LinkedProgram {
        base: IMAGE_BASE,
        image,
        mem_bytes: GUEST_MEM_BYTES,
        entries,
        arm_entries,
        externs,
        redirects,
        imports,
        inline_imports,
        module_inits,
        main_entry,
        alloc_base: align_up(image_end, MODULE_ALIGN),
        host_import_count,
        process_param,
        tls_template,
        unresolved_var_imports,
        loaded_modules,
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
                // A real block is a small header (0x28..=0x48 bytes) and a known
                // version. The toolchain has emitted several over the years - v6 is
                // the common recent one, but older titles ship v5 (the SceLibcParam
                // pointer sits one word later, which is the guest crt's concern, not
                // ours - we only hand back the block's address). The exact "PSP2"
                // magic, confined to this module's own segments and paired with a sane
                // size, already identifies the block uniquely; the version is a final
                // sanity guard, so accept the whole real range rather than pinning v6.
                if (1..=6).contains(&ver) && (0x28..=0x48).contains(&size) {
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

/// Apply one variable import's fixup blob, binding every listed site to the
/// resolved symbol address `sym`.
///
/// The blob (at guest address `blob_ptr`, inside `image`) is a `u32` header whose
/// value is `byte_len << 4` (the length, header included), followed by
/// `{ code_word: u32, site_offset: u32 }` entries. `code_word`'s byte 1 is an
/// `R_ARM_*` relocation code; `site_offset` is the fixup site's byte offset from the
/// importing module's link base, so its runtime address is `module_base +
/// site_offset`. The addend is always zero for a variable import (the site holds no
/// prior displacement). Supports the codes Vita variable imports emit: the Thumb
/// `MOVW`/`MOVT` pair that materializes the address in a register, and `ABS32` for a
/// plain pointer word.
fn apply_var_fixups(
    image: &mut [u8],
    module_base: u32,
    blob_ptr: u32,
    sym: u32,
    applied: &mut u32,
) -> Result<(), vitaslop_loader::Error> {
    use vitaslop_loader::reloc::code;
    let rd32 = |img: &[u8], addr: u32| -> Option<u32> {
        let o = addr.checked_sub(IMAGE_BASE)? as usize;
        img.get(o..o + 4).map(|b| u32::from_le_bytes([b[0], b[1], b[2], b[3]]))
    };
    let header = rd32(image, blob_ptr).ok_or(vitaslop_loader::Error::OutOfBounds("var fixup header"))?;
    let byte_len = (header >> 4) as usize;
    if byte_len < 4 {
        return Ok(());
    }
    // Entries begin after the 4-byte header and run to the declared length.
    let mut entry_addr = blob_ptr.wrapping_add(4);
    let mut remaining = byte_len - 4;
    while remaining >= 8 {
        let code_word = rd32(image, entry_addr).ok_or(vitaslop_loader::Error::OutOfBounds("var fixup entry"))?;
        let site_offset = rd32(image, entry_addr.wrapping_add(4))
            .ok_or(vitaslop_loader::Error::OutOfBounds("var fixup entry"))?;
        let rcode = ((code_word >> 8) & 0xFF) as u8;
        let site = module_base.wrapping_add(site_offset);
        let off = site.wrapping_sub(IMAGE_BASE) as usize;
        match rcode {
            code::NONE => {}
            code::THM_MOVW_ABS_NC => vitaslop_loader::patch_thumb_mov(image, off, (sym & 0xFFFF) as u16)?,
            code::THM_MOVT_ABS => vitaslop_loader::patch_thumb_mov(image, off, (sym >> 16) as u16)?,
            code::ABS32 | code::TARGET1 => {
                let slot = image
                    .get_mut(off..off + 4)
                    .ok_or(vitaslop_loader::Error::OutOfBounds("var fixup abs32 site"))?;
                slot.copy_from_slice(&sym.to_le_bytes());
            }
            other => return Err(vitaslop_loader::Error::UnsupportedReloc(other)),
        }
        *applied += 1;
        entry_addr = entry_addr.wrapping_add(8);
        remaining -= 8;
    }
    Ok(())
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
        let game = decrypt_container(&mut DirVfs::new(dir)).expect("decrypt");
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
        let game = decrypt_container(&mut DirVfs::new(dir)).expect("decrypt");
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
        let game = decrypt_container(&mut DirVfs::new(dir)).expect("decrypt");
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
