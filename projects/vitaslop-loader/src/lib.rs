//! Vita ROM front half: parses a Vita executable (velf) - the loadable
//! segments, the entry point, and the NID function-import table - into the
//! [`Module`] the rest of the pipeline consumes.
//!
//! A velf is an ELF with `e_type` `ET_SCE_RELEXEC` (0xFE04). It is the
//! decrypted form a SELF wraps; we own the loader so we skip the SELF/fself
//! crypto layer entirely and parse the velf directly. The Vita-specific part is
//! the `SceModuleInfo` structure (found via `e_entry`), whose import table lists
//! each imported library by NID and, per function, a NID plus the guest address
//! of the stub the module calls. Resolving an import means pointing that stub at
//! a host implementation - the transpiler's [`Extern`](vitaslop_transpiler::Extern)
//! wiring.
//!
//! Parsing is pure, bounds-checked byte reading with no dependencies, so it
//! builds for wasm as cleanly as native.
//!
//! Real titles ship as a SELF/fSELF (`eboot.bin`), which wraps this velf in an
//! SCE container. [`load`] auto-detects the `"SCE\0"` magic and unwraps it (see
//! [`self_`]) before parsing the velf inside.

#[path = "self.rs"]
pub mod self_;
pub mod inflate;
pub mod reloc;

/// `e_type` of a velf: a relocatable SCE executable (position-independent, laid
/// out relative to a link base and carrying SCE relocations).
const ET_SCE_RELEXEC: u16 = 0xFE04;
/// `e_type` of a fixed-address SCE executable: its segments carry absolute load
/// vaddrs and it ships no SCE relocations, so it must load at its native base and
/// cannot be shifted. Common for launch-window titles.
const ET_SCE_EXEC: u16 = 0xFE00;
const PT_LOAD: u32 = 1;
const EHDR_SIZE: usize = 52;
const PHDR_SIZE: usize = 32;
/// `sce_module_imports` (the 0x34-byte form vita-elf-create emits).
const IMPORTS_ENTRY_SIZE: u32 = 0x34;

/// A loadable segment of the module image.
pub struct Segment {
    /// Guest virtual address the segment loads at.
    pub vaddr: u32,
    /// Total size in memory. `data.len()` bytes are file-backed; any remainder
    /// up to `mem_size` is zero-filled (`.bss`).
    pub mem_size: u32,
    /// The file-backed bytes of the segment.
    pub data: Vec<u8>,
    pub executable: bool,
    pub writable: bool,
}

/// One imported function: the library it comes from, its NID, and the guest
/// address of the stub the module branches to when it calls it.
pub struct Import {
    pub library_nid: u32,
    pub func_nid: u32,
    pub stub_addr: u32,
}

/// One exported function: the library it belongs to, its NID, and the guest
/// address other modules reach it at (the low bit carries the Thumb state, as in
/// any ARM function pointer). Parsed from the module's `sce_module_exports`
/// tables so a multi-module link can resolve an inter-module import to the real
/// callee instead of a host trap.
pub struct Export {
    pub library_nid: u32,
    pub func_nid: u32,
    pub addr: u32,
}

/// One imported *variable* (data symbol): the library and NID it comes from, and
/// the guest address of its SCE fixup blob. Unlike a function import (which patches
/// a single call stub), a variable import carries a list of code/data sites that
/// reference the symbol; resolving it applies that list, pointing every site at the
/// exporting module's copy of the variable. Vita's C library exports its ctype
/// tables (`_Ctype`, `_Tolotab`, `_Touptab`) and stdio handles (`_Stdout`,
/// `_Stderr`, ...) this way, so a statically-linked title whose variable imports go
/// unresolved reads a garbage tolower table and corrupts every case-folded path.
#[derive(Debug, Clone, Copy)]
pub struct VarImport {
    pub library_nid: u32,
    pub var_nid: u32,
    /// Guest address of the fixup blob: a `u32` header (byte length = `header >> 4`,
    /// including the header) followed by `{ code_word: u32, offset: u32 }` entries.
    /// `code_word`'s byte 1 is an `R_ARM_*` code; `offset` is the site's byte offset
    /// from the module's link base.
    pub blob_ptr: u32,
}

/// One exported *variable* (data symbol): the library, NID, and the guest address
/// of the variable itself in this module. A [`VarImport`] with the same NID binds
/// its fixup sites to this address.
#[derive(Debug, Clone, Copy)]
pub struct VarExport {
    pub library_nid: u32,
    pub var_nid: u32,
    pub addr: u32,
}

/// A parsed Vita module.
pub struct Module {
    /// Module name from `SceModuleInfo` (e.g. "cube.elf").
    pub name: String,
    pub module_nid: u32,
    /// Lowest segment vaddr - where the image begins.
    pub base: u32,
    /// Whether this image may be relocated to a different base. A `ET_SCE_RELEXEC`
    /// velf can (it carries SCE relocations); a fixed `ET_SCE_EXEC` cannot - it has
    /// absolute internal pointers and no relocations, so it must load at [`base`].
    ///
    /// [`base`]: Module::base
    pub relocatable: bool,
    /// The entry point (`SceModuleInfo::module_start`), where execution begins.
    pub entry: u32,
    pub segments: Vec<Segment>,
    /// Function imports, in import-table order.
    pub imports: Vec<Import>,
    /// Function exports (what other modules can import from this one), in
    /// export-table order. Empty for a module that exports nothing callable.
    pub exports: Vec<Export>,
    /// Variable (data-symbol) imports, in import-table order. Resolved at link time
    /// by applying each one's fixup blob against the matching [`VarExport`].
    pub var_imports: Vec<VarImport>,
    /// Variable (data-symbol) exports this module provides to others.
    pub var_exports: Vec<VarExport>,
    /// SCE relocations (`PT_SCE_RELA`), decoded but not yet applied. [`rebase`]
    /// applies them when the module is placed at a runtime base.
    ///
    /// [`rebase`]: Module::rebase
    pub relocations: Vec<reloc::Reloc>,
    /// Static constructor/destructor function pointers, read from the
    /// `.preinit_array`/`.init_array`/`.fini_array` sections (when the module
    /// keeps section headers). These are reachable only through an indirect call
    /// (`__libc_init_array` walks the table and `blx`es each), so they seed
    /// transpiler discovery that the direct-call closure alone would miss. Empty
    /// for `-nostdlib` modules and for anything with its section headers stripped.
    pub init_pointers: Vec<u32>,
    /// Absolute function pointers recovered from `R_ARM_ABS32`/`TARGET1`
    /// relocations whose resolved value has the Thumb bit set (data pointers are
    /// even, so a bit0-set pointer into code is a function pointer). These sit in
    /// data tables - vtables, callback arrays, jump tables - reached only through
    /// an indirect `blx`/`bx`, which the `movw`/`movt` code-pointer scan cannot
    /// see. Populated by [`rebase`](Module::rebase) with the Thumb bit stripped;
    /// empty until then. Seeds transpiler discovery.
    pub code_pointers: Vec<u32>,
    /// Absolute pointers (`R_ARM_ABS32`/`TARGET1`) that are *even* (Thumb bit clear)
    /// and land inside an executable segment - i.e. ARM-mode function pointers. A
    /// Thumb function pointer carries bit 0; an even pointer into code is either ARM
    /// code (e.g. a table of ARM stubs a `blx` reaches) or a stray data address, so
    /// these seed discovery *tentatively* as ARM functions (a bad guess that fails to
    /// decode is dropped). Populated by [`rebase`](Module::rebase); empty until then.
    pub arm_code_pointers: Vec<u32>,
    /// Thread-local-storage template, from `SceModuleInfo` (`tls_start`/`tls_filesz`/
    /// `tls_memsz`). `tls_vaddr` is the guest address of the init image (`.tdata`),
    /// `tls_filesz` its initialized byte count, `tls_memsz` the full per-thread block
    /// size (init data plus zero-filled `.tbss`). Each thread gets its own copy, and
    /// the compiler reaches `__thread` variables at `thread_pointer + offset` after a
    /// `MRC p15,0,Rt,c13,c0,3` read of the thread pointer. `tls_vaddr` is 0 (and the
    /// sizes 0) for a module with no TLS. [`rebase`](Module::rebase) shifts `tls_vaddr`.
    pub tls_vaddr: u32,
    pub tls_filesz: u32,
    pub tls_memsz: u32,
}

/// Why loading failed.
#[derive(Debug, PartialEq, Eq)]
pub enum Error {
    /// Not a 32-bit little-endian ELF.
    NotElf,
    /// `e_type` was not a velf we handle.
    UnsupportedType(u16),
    /// A structure referenced bytes outside the file or any segment.
    OutOfBounds(&'static str),
    /// The module-info segment index encoded in `e_entry` has no segment.
    BadModuleInfo,
    /// A SELF segment is encrypted (a real retail title); we do not decrypt.
    EncryptedSelf,
    /// A SELF segment is zlib-compressed (`vita-make-fself -c`); the loader is
    /// dependency-free, so inflate is not built in yet.
    CompressedSelf,
    /// A relocation used a code we do not model (see [`reloc::code`]).
    UnsupportedReloc(u8),
    /// The SCE relocation blob carried an entry format we do not model (the low
    /// nibble of the entry's first word). Carries `(format, byte_offset)`. Silently
    /// dropping the rest of the blob would leave dangling zeroed pointers that only
    /// fault much later, so this is a hard load failure.
    UnknownRelocFormat(u8, usize),
}

impl Module {
    /// The guest address one past the highest byte of any segment.
    pub fn image_end(&self) -> u32 {
        self.segments
            .iter()
            .map(|s| s.vaddr.wrapping_add(s.mem_size))
            .max()
            .unwrap_or(self.base)
    }

    /// Place this module at runtime base `new_base`, applying every relocation.
    ///
    /// A velf links at a nominal base (`self.base`, always `0x8100_0000`); several
    /// modules that must share one address space each need a distinct base. This
    /// rigidly shifts the whole image by `delta = new_base - self.base` and, for
    /// every relocation, patches the embedded absolute address at the fixup site
    /// so the code and data point at the module's new location. After it returns,
    /// `base`, `entry`, every segment `vaddr`, every import stub, every export
    /// address, and the init-pointer table are all expressed at the new base, and
    /// `relocations` is cleared (a module is relocated exactly once).
    ///
    /// Only the relocation codes Vita modules actually emit are handled - absolute
    /// 32-bit words and Thumb `MOVW`/`MOVT` immediate pairs; an unknown code is a
    /// hard error (silently skipping a fixup would leave a dangling pointer).
    pub fn rebase(&mut self, new_base: u32) -> Result<(), Error> {
        let delta = new_base.wrapping_sub(self.base);
        if delta == 0 && self.relocations.is_empty() {
            return Ok(());
        }
        // Runtime base of each segment after the shift, indexed as the relocation
        // `sym_seg` / `data_seg` fields index (PT_LOAD order).
        let seg_base: Vec<u32> = self
            .segments
            .iter()
            .map(|s| s.vaddr.wrapping_add(delta))
            .collect();

        // Post-shift executable-segment ranges, to classify even absolute pointers:
        // one landing in code is an ARM function pointer, one elsewhere is data.
        let exec_ranges: Vec<(u32, u32)> = self
            .segments
            .iter()
            .enumerate()
            .filter(|(_, s)| s.executable)
            .map(|(i, s)| (seg_base[i], seg_base[i].wrapping_add(s.mem_size)))
            .collect();
        let in_exec = |a: u32| exec_ranges.iter().any(|&(lo, hi)| a >= lo && a < hi);

        let mut code_pointers: Vec<u32> = Vec::new();
        let mut arm_code_pointers: Vec<u32> = Vec::new();
        for r in &self.relocations {
            let s = *seg_base
                .get(r.sym_seg as usize)
                .ok_or(Error::OutOfBounds("reloc sym_seg"))?;
            let target = s.wrapping_add(r.addend);
            let data = self
                .segments
                .get_mut(r.data_seg as usize)
                .ok_or(Error::OutOfBounds("reloc data_seg"))?;
            let off = r.offset as usize;
            match r.code {
                reloc::code::NONE => {}
                reloc::code::ABS32 | reloc::code::TARGET1 => {
                    let slot = data
                        .data
                        .get_mut(off..off + 4)
                        .ok_or(Error::OutOfBounds("reloc abs32 site"))?;
                    slot.copy_from_slice(&target.to_le_bytes());
                    // A resolved pointer with the Thumb bit set is a function
                    // pointer sitting in data - a discovery seed the direct-call /
                    // movw-movt closures cannot reach.
                    if target & 1 == 1 {
                        code_pointers.push(target & !1);
                    } else if target != 0 && in_exec(target) {
                        // Even pointer into code: an ARM-mode function pointer.
                        arm_code_pointers.push(target);
                    }
                }
                // PC-relative words. Homebrew toolchains emit these (a retail title never
                // did); `P` is the site's own post-shift address.
                reloc::code::REL32 | reloc::code::TARGET2 => {
                    let site = seg_base
                        .get(r.data_seg as usize)
                        .ok_or(Error::OutOfBounds("reloc data_seg"))?
                        .wrapping_add(r.offset);
                    let slot = data
                        .data
                        .get_mut(off..off + 4)
                        .ok_or(Error::OutOfBounds("reloc rel32 site"))?;
                    slot.copy_from_slice(&target.wrapping_sub(site).to_le_bytes());
                }
                reloc::code::PREL31 => {
                    let site = seg_base
                        .get(r.data_seg as usize)
                        .ok_or(Error::OutOfBounds("reloc data_seg"))?
                        .wrapping_add(r.offset);
                    let slot = data
                        .data
                        .get_mut(off..off + 4)
                        .ok_or(Error::OutOfBounds("reloc prel31 site"))?;
                    let old = u32::from_le_bytes([slot[0], slot[1], slot[2], slot[3]]);
                    let v = (target.wrapping_sub(site) & 0x7FFF_FFFF) | (old & 0x8000_0000);
                    slot.copy_from_slice(&v.to_le_bytes());
                }
                // PC-relative branches. Every segment moves by the same `delta`, so a
                // branch from one place in the module to another keeps its encoded
                // distance: nothing to patch. Homebrew toolchains emit these; a retail
                // title never did.
                reloc::code::THM_CALL
                | reloc::code::CALL
                | reloc::code::JUMP24
                | reloc::code::THM_JUMP24
                | reloc::code::THM_JUMP11
                | reloc::code::THM_JUMP8 => {}
                // `BX Rm` on a v4 target: nothing to patch on ARMv7.
                reloc::code::V4BX => {}
                reloc::code::THM_MOVW_ABS_NC => {
                    patch_thumb_mov(&mut data.data, off, (target & 0xFFFF) as u16)?;
                }
                reloc::code::THM_MOVT_ABS => {
                    patch_thumb_mov(&mut data.data, off, (target >> 16) as u16)?;
                }
                other => return Err(Error::UnsupportedReloc(other)),
            }
        }

        // Shift every address the module reports now that the sites are patched.
        for s in &mut self.segments {
            s.vaddr = s.vaddr.wrapping_add(delta);
        }
        for imp in &mut self.imports {
            imp.stub_addr = imp.stub_addr.wrapping_add(delta);
        }
        for ex in &mut self.exports {
            ex.addr = ex.addr.wrapping_add(delta);
        }
        // Variable-import fixup blobs and variable-export addresses are module data:
        // they move with the module. The site offsets inside each blob stay
        // link-base-relative (resolved against the module base at apply time).
        for vi in &mut self.var_imports {
            vi.blob_ptr = vi.blob_ptr.wrapping_add(delta);
        }
        for ve in &mut self.var_exports {
            ve.addr = ve.addr.wrapping_add(delta);
        }
        for p in &mut self.init_pointers {
            *p = p.wrapping_add(delta);
        }
        self.entry = self.entry.wrapping_add(delta);
        // The TLS init image moves with the module (0 = no TLS, leave it 0).
        if self.tls_vaddr != 0 {
            self.tls_vaddr = self.tls_vaddr.wrapping_add(delta);
        }
        self.base = new_base;
        self.relocations.clear();
        code_pointers.sort_unstable();
        code_pointers.dedup();
        self.code_pointers = code_pointers;
        arm_code_pointers.sort_unstable();
        arm_code_pointers.dedup();
        self.arm_code_pointers = arm_code_pointers;
        Ok(())
    }
}

/// Patch the 16-bit immediate of a 32-bit Thumb-2 `MOVW`/`MOVT` (T3 encoding) at
/// `off` in `data`, preserving the opcode and destination register. The immediate
/// splits as `imm4:i:imm3:imm8` across the two halfwords (little-endian): the
/// first halfword carries `i` (bit 10) and `imm4` (bits 3..0); the second carries
/// `imm3` (bits 14..12) and `imm8` (bits 7..0).
pub fn patch_thumb_mov(data: &mut [u8], off: usize, imm16: u16) -> Result<(), Error> {
    let site = data
        .get_mut(off..off + 4)
        .ok_or(Error::OutOfBounds("reloc thumb-mov site"))?;
    let mut hw1 = u16::from_le_bytes([site[0], site[1]]);
    let mut hw2 = u16::from_le_bytes([site[2], site[3]]);
    let imm4 = (imm16 >> 12) & 0xF;
    let i = (imm16 >> 11) & 0x1;
    let imm3 = (imm16 >> 8) & 0x7;
    let imm8 = imm16 & 0xFF;
    hw1 = (hw1 & 0xFBF0) | (i << 10) | imm4;
    hw2 = (hw2 & 0x8F00) | (imm3 << 12) | imm8;
    site[0..2].copy_from_slice(&hw1.to_le_bytes());
    site[2..4].copy_from_slice(&hw2.to_le_bytes());
    Ok(())
}

/// Owned inputs for the transpiler, derived from a [`Module`]. Holds the buffers
/// [`vitaslop_transpiler::Program`] borrows; call [`program`](Self::program) to
/// view it. This is the loader -> transpiler seam: the executable segment as the
/// code image, the entry point, and each NID import as an [`Extern`] from its
/// stub address to a dense import index (the host later maps index -> handler).
pub struct ProgramInputs {
    pub code: Vec<u8>,
    pub base: u32,
    pub entries: Vec<u32>,
    pub externs: Vec<vitaslop_transpiler::Extern>,
    /// Inter-module redirects (empty for a single-module load; the multi-module
    /// linker populates them).
    pub redirects: Vec<vitaslop_transpiler::Redirect>,
    /// Imports to emit inline rather than as a host trap. The loader does not decide
    /// this - it has no idea what a NID means - so the runtime fills it in from the
    /// import table (see `vitaslop_runtime::link`). Empty leaves every import a host
    /// call, which is what the ARM conformance corpus wants.
    pub inline_imports: Vec<vitaslop_transpiler::InlineImport>,
    /// True if the entry point is Thumb (entry had bit 0 set).
    pub thumb_entry: bool,
    /// Total guest memory to provision (image + stack + heap), in bytes from
    /// `base`. Chosen by the caller; defaults to [`DEFAULT_MEM_BYTES`].
    pub mem_bytes: u32,
    /// Emit a module that imports a shared linear memory instead of defining one,
    /// so several instances can share the guest address space (the preemptive
    /// multi-thread scheduler). Off by default; single-instance hosts leave it so.
    pub import_memory: bool,
}

/// Default guest memory for a loaded module (image + stack + host allocations).
pub const DEFAULT_MEM_BYTES: u32 = 64 * 1024 * 1024;

impl ProgramInputs {
    /// Borrow these inputs as a transpiler [`Program`](vitaslop_transpiler::Program).
    pub fn program(&self) -> vitaslop_transpiler::Program<'_> {
        vitaslop_transpiler::Program {
            code: &self.code,
            base: self.base,
            thumb: self.thumb_entry,
            entries: &self.entries,
            // Single-module loads (the conformance corpus) are whole-program Thumb;
            // the multi-module retail linker is what seeds ARM code pointers.
            arm_entries: &[],
            externs: &self.externs,
            redirects: &self.redirects,
            inline_imports: &self.inline_imports,
            // Vita dispatches host calls through NID stubs (bl/blx to a stub
            // address), not `svc`, so there is no noreturn-syscall set here.
            noreturn_svc: &[],
            mem_bytes: self.mem_bytes,
            // Vita modules take function addresses (thread entries, GXM
            // callbacks) that the direct-call closure alone would miss.
            discover_code_pointers: true,
            import_memory: self.import_memory,
        }
    }
}

impl Module {
    /// Lower this module to transpiler inputs: the executable segment becomes the
    /// code image, the entry (Thumb bit cleared) the sole entry point, and each
    /// import a stub-address -> import-index [`Extern`].
    pub fn program_inputs(&self) -> ProgramInputs {
        let exec = self
            .segments
            .iter()
            .find(|s| s.executable)
            .unwrap_or(&self.segments[0]);
        let externs = self
            .imports
            .iter()
            .enumerate()
            .map(|(i, imp)| vitaslop_transpiler::Extern {
                addr: imp.stub_addr,
                import: i as u32,
            })
            .collect();
        // The image the host loads into guest memory spans every loadable segment,
        // not just the executable one: a Vita module's initialized data (`.data`)
        // and zero-filled `.bss` live in a second RW segment above the code, and a
        // program that reads a static initializer (or any global) needs those bytes
        // present. Each segment is placed at `vaddr - base`; the gaps and each
        // segment's `.bss` tail are left zero. The transpiler decodes only the code
        // reachable from the entry, so the trailing data bytes are never mis-decoded.
        let base = exec.vaddr;
        let end = self
            .segments
            .iter()
            .map(|s| s.vaddr.wrapping_add(s.mem_size))
            .max()
            .unwrap_or(base);
        let mut code = vec![0u8; end.wrapping_sub(base) as usize];
        for seg in &self.segments {
            let off = seg.vaddr.wrapping_sub(base) as usize;
            code[off..off + seg.data.len()].copy_from_slice(&seg.data);
        }
        // Entries: the module entry plus every static constructor/destructor the
        // init/fini arrays reference (each masked to its even function address).
        // These are reached only via indirect call, so without them the closure
        // walk would never translate the constructors and the indirect-call
        // dispatcher could not resolve them.
        let in_image = |a: u32| a.wrapping_sub(base) < end.wrapping_sub(base);
        let mut entries = vec![self.entry & !1];
        for &p in &self.init_pointers {
            let a = p & !1;
            if in_image(a) && !entries.contains(&a) {
                entries.push(a);
            }
        }
        ProgramInputs {
            code,
            base,
            entries,
            externs,
            redirects: Vec::new(),
            inline_imports: Vec::new(),
            thumb_entry: self.entry & 1 != 0,
            mem_bytes: DEFAULT_MEM_BYTES,
            import_memory: false,
        }
    }
}

/// Little-endian reads out of a byte slice, each bounds-checked.
struct Reader<'a> {
    bytes: &'a [u8],
}

impl<'a> Reader<'a> {
    fn u16(&self, at: usize) -> Result<u16, Error> {
        let b = self
            .bytes
            .get(at..at + 2)
            .ok_or(Error::OutOfBounds("u16"))?;
        Ok(u16::from_le_bytes([b[0], b[1]]))
    }

    fn u32(&self, at: usize) -> Result<u32, Error> {
        let b = self
            .bytes
            .get(at..at + 4)
            .ok_or(Error::OutOfBounds("u32"))?;
        Ok(u32::from_le_bytes([b[0], b[1], b[2], b[3]]))
    }

    fn u64(&self, at: usize) -> Result<u64, Error> {
        let b = self
            .bytes
            .get(at..at + 8)
            .ok_or(Error::OutOfBounds("u64"))?;
        Ok(u64::from_le_bytes([
            b[0], b[1], b[2], b[3], b[4], b[5], b[6], b[7],
        ]))
    }
}

/// A loadable segment plus where it sits in the file, so we can map a guest
/// vaddr back to file bytes while parsing.
struct LoadSeg {
    vaddr: u32,
    file_offset: usize,
    file_size: usize,
    mem_size: u32,
    flags: u32,
}

/// Parse a Vita executable into a [`Module`].
///
/// Accepts either a bare velf or a SELF/fSELF container (`eboot.bin`): a SELF is
/// unwrapped to its inner velf first, then parsed. The returned [`Module`] owns
/// its segment bytes, so the unwrapped buffer does not need to outlive the call.
pub fn load(bytes: &[u8]) -> Result<Module, Error> {
    if self_::is_self(bytes) {
        let inner = self_::unwrap_self(bytes)?;
        return load(&inner);
    }

    let r = Reader { bytes };

    // ELF32 little-endian header check: magic + EI_CLASS=1 + EI_DATA=1.
    if bytes.len() < EHDR_SIZE
        || &bytes[0..4] != b"\x7fELF"
        || bytes[4] != 1
        || bytes[5] != 1
    {
        return Err(Error::NotElf);
    }

    let e_type = r.u16(16)?;
    // Both SCE executable shapes carry SceModuleInfo the same way and are parsed
    // identically here; they differ only in placement (a RELEXEC relocates to any
    // base, a fixed EXEC must stay at its absolute vaddrs), recorded in
    // `relocatable` so the linker never shifts a fixed image (which has no
    // relocations to patch).
    let relocatable = match e_type {
        ET_SCE_RELEXEC => true,
        ET_SCE_EXEC => false,
        other => return Err(Error::UnsupportedType(other)),
    };

    // e_entry locates SceModuleInfo: top two bits are the program-header (segment)
    // index, the rest is the offset within it. (Same encoding for both e_types.)
    let e_entry = r.u32(24)?;
    let e_phoff = r.u32(28)? as usize;
    let e_phnum = r.u16(44)? as usize;

    // Collect the loadable segments in program-header order (the module-info
    // index and every relocated pointer are resolved against this order).
    let mut load_segs: Vec<LoadSeg> = Vec::new();
    // Parallel: segment index (in phdr order) for each LoadSeg, for e_entry.
    let mut seg_phdr_index: Vec<usize> = Vec::new();
    for i in 0..e_phnum {
        let ph = e_phoff + i * PHDR_SIZE;
        if r.u32(ph)? != PT_LOAD {
            continue;
        }
        let file_offset = r.u32(ph + 4)? as usize;
        let vaddr = r.u32(ph + 8)?;
        let file_size = r.u32(ph + 16)? as usize;
        let mem_size = r.u32(ph + 20)?;
        let flags = r.u32(ph + 24)?;
        if bytes.get(file_offset..file_offset + file_size).is_none() {
            return Err(Error::OutOfBounds("segment"));
        }
        load_segs.push(LoadSeg {
            vaddr,
            file_offset,
            file_size,
            mem_size,
            flags,
        });
        seg_phdr_index.push(i);
    }
    if load_segs.is_empty() {
        return Err(Error::BadModuleInfo);
    }

    // Read a u32 at a guest vaddr by finding the segment that contains it.
    let read_vaddr = |vaddr: u32| -> Result<u32, Error> {
        for s in &load_segs {
            let end = s.vaddr.wrapping_add(s.file_size as u32);
            if vaddr >= s.vaddr && vaddr.wrapping_add(4) <= end {
                let off = s.file_offset + (vaddr - s.vaddr) as usize;
                return r.u32(off);
            }
        }
        Err(Error::OutOfBounds("vaddr"))
    };

    // Locate SceModuleInfo via e_entry (segment index + offset).
    let mi_seg_index = (e_entry >> 30) as usize;
    let mi_offset = (e_entry & 0x3FFF_FFFF) as usize;
    let mi_seg = seg_phdr_index
        .iter()
        .position(|&p| p == mi_seg_index)
        .and_then(|pos| load_segs.get(pos))
        .ok_or(Error::BadModuleInfo)?;
    let mi_file = mi_seg.file_offset + mi_offset;

    // SceModuleInfo fields we need. The top/end pointers here are SEGMENT-
    // RELATIVE offsets (unlike the absolute pointers inside the import structs).
    let name = read_cstr(bytes, mi_file + 0x04, 27);
    let export_top_off = r.u32(mi_file + 0x24)?;
    let export_end_off = r.u32(mi_file + 0x28)?;
    let import_top_off = r.u32(mi_file + 0x2c)?;
    let import_end_off = r.u32(mi_file + 0x30)?;
    let module_nid = r.u32(mi_file + 0x34)?;
    let tls_start_off = r.u32(mi_file + 0x38)?;
    let tls_filesz = r.u32(mi_file + 0x3c)?;
    let tls_memsz = r.u32(mi_file + 0x40)?;
    let module_start_off = r.u32(mi_file + 0x44)?;

    let seg_base = mi_seg.vaddr;
    let entry = seg_base.wrapping_add(module_start_off);
    // TLS template address is a segment-relative offset like the export/import
    // pointers; 0 means the module has no TLS. Keep it 0 in that case so a consumer
    // can cheaply test `tls_memsz == 0`.
    let tls_vaddr = if tls_start_off == 0 { 0 } else { seg_base.wrapping_add(tls_start_off) };
    let export_top = seg_base.wrapping_add(export_top_off);
    let export_end = seg_base.wrapping_add(export_end_off);
    let import_top = seg_base.wrapping_add(import_top_off);
    let import_end = seg_base.wrapping_add(import_end_off);

    // Walk the import table: fixed-size entries from import_top to import_end.
    let mut imports = Vec::new();
    let mut var_imports = Vec::new();
    let mut cursor = import_top;
    while cursor < import_end {
        let base = cursor;
        // Read the entry's declared size so we tolerate the 0x24 short form too.
        let size = read_vaddr_u16(&read_vaddr, base)?;
        let step = if size == 0 {
            IMPORTS_ENTRY_SIZE
        } else {
            size as u32
        };

        // 0x34 form layout (see sce_module_imports). The short 0x24 form drops
        // reserved1/reserved2, shifting the table pointers up by 8; detect it by
        // the declared size.
        let (num_funcs, library_nid, func_nid_table, func_entry_table) = if step == 0x24 {
            (
                read_vaddr_u16(&read_vaddr, base + 0x06)?,
                read_vaddr(base + 0x0c)?,
                read_vaddr(base + 0x14)?,
                read_vaddr(base + 0x18)?,
            )
        } else {
            (
                read_vaddr_u16(&read_vaddr, base + 0x06)?,
                read_vaddr(base + 0x10)?,
                read_vaddr(base + 0x1c)?,
                read_vaddr(base + 0x20)?,
            )
        };

        for i in 0..num_funcs as u32 {
            let func_nid = read_vaddr(func_nid_table + i * 4)?;
            let stub_addr = read_vaddr(func_entry_table + i * 4)?;
            imports.push(Import {
                library_nid,
                func_nid,
                stub_addr,
            });
        }

        // Variable (data) imports follow the functions in the same import entry.
        // The short (0x24) form drops the two reserved words, shifting the var
        // tables up by 8, exactly as the function tables shift.
        let (num_vars, var_nid_table, var_entry_table) = if step == 0x24 {
            (read_vaddr_u16(&read_vaddr, base + 0x08)?, read_vaddr(base + 0x1c)?, read_vaddr(base + 0x20)?)
        } else {
            (read_vaddr_u16(&read_vaddr, base + 0x08)?, read_vaddr(base + 0x24)?, read_vaddr(base + 0x28)?)
        };
        for i in 0..num_vars as u32 {
            let var_nid = read_vaddr(var_nid_table + i * 4)?;
            let blob_ptr = read_vaddr(var_entry_table + i * 4)?;
            var_imports.push(VarImport { library_nid, var_nid, blob_ptr });
        }

        cursor = cursor.wrapping_add(step);
    }

    // Walk the export table: fixed-size 0x20-byte `sce_module_exports` entries.
    // Each declares num_syms_funcs function symbols followed by num_syms_vars
    // variables; we keep the functions (what another module can call). The
    // entry addresses are absolute vaddrs carrying the Thumb bit.
    let mut exports = Vec::new();
    let mut var_exports = Vec::new();
    let mut cursor = export_top;
    while cursor < export_end {
        let base = cursor;
        let size = read_vaddr_u16(&read_vaddr, base)?;
        let step = if size == 0 { 0x20 } else { size as u32 };
        let num_funcs = read_vaddr_u16(&read_vaddr, base + 0x06)?;
        let num_vars = read_vaddr_u16(&read_vaddr, base + 0x08)?;
        let library_nid = read_vaddr(base + 0x10)?;
        let nid_table = read_vaddr(base + 0x18)?;
        let entry_table = read_vaddr(base + 0x1c)?;
        for i in 0..num_funcs as u32 {
            let func_nid = read_vaddr(nid_table + i * 4)?;
            let addr = read_vaddr(entry_table + i * 4)?;
            exports.push(Export { library_nid, func_nid, addr });
        }
        // Variable exports follow the functions in the shared nid/entry tables.
        for i in num_funcs as u32..(num_funcs as u32 + num_vars as u32) {
            let var_nid = read_vaddr(nid_table + i * 4)?;
            let addr = read_vaddr(entry_table + i * 4)?;
            var_exports.push(VarExport { library_nid, var_nid, addr });
        }
        cursor = cursor.wrapping_add(step);
    }

    // Decode every SCE relocation (PT_SCE_RELA, p_type 0x60000000). They stay
    // pending until `rebase` places the module and applies them.
    let mut relocations = Vec::new();
    for i in 0..e_phnum {
        let ph = e_phoff + i * PHDR_SIZE;
        if r.u32(ph)? != reloc::PT_SCE_RELA {
            continue;
        }
        let p_offset = r.u32(ph + 4)? as usize;
        let p_filesz = r.u32(ph + 16)? as usize;
        let blob = bytes
            .get(p_offset..p_offset + p_filesz)
            .ok_or(Error::OutOfBounds("reloc segment"))?;
        relocations.extend(reloc::decode(blob)?);
    }

    // Build the public segments (owning their file bytes, bss zero-filled).
    let mut segments = Vec::new();
    let mut base = u32::MAX;
    for s in &load_segs {
        base = base.min(s.vaddr);
        let mut data = bytes[s.file_offset..s.file_offset + s.file_size].to_vec();
        // Zero-extend file-backed data up to mem_size is left to the runtime;
        // we keep only the file bytes and report mem_size.
        data.shrink_to_fit();
        segments.push(Segment {
            vaddr: s.vaddr,
            mem_size: s.mem_size,
            data,
            executable: s.flags & 0x1 != 0,
            writable: s.flags & 0x2 != 0,
        });
    }

    let init_pointers = read_init_pointers(&r, bytes).unwrap_or_default();

    // A fixed `ET_SCE_EXEC` image ships no SCE relocations, so the code-pointer
    // seeds a RELEXEC gets from its `R_ARM_ABS32` fixups (function pointers sitting
    // in data tables - vtables, callback arrays, jump tables, reached only through
    // an indirect `blx`/`bx`) are unavailable, and the transpiler's `movw`/`movt`
    // scan cannot see a pointer that is loaded from memory rather than materialized
    // as an immediate. Recover them by scanning the image's word-aligned data for
    // values that address one of its own executable segments: an odd value is a
    // Thumb function pointer, an even one a tentative ARM pointer. Both are
    // discovery seeds the transpiler verifies (a bad guess fails to decode and is
    // dropped). A RELEXEC gets these from `rebase` instead, so only scan a fixed
    // image here.
    let (code_pointers, arm_code_pointers) = if relocatable {
        (Vec::new(), Vec::new())
    } else {
        scan_fixed_code_pointers(&segments)
    };

    Ok(Module {
        name,
        module_nid,
        base,
        relocatable,
        entry,
        segments,
        imports,
        exports,
        var_imports,
        var_exports,
        relocations,
        init_pointers,
        code_pointers,
        arm_code_pointers,
        tls_vaddr,
        tls_filesz,
        tls_memsz,
    })
}

/// Read the constructor/destructor pointer tables (`.preinit_array`,
/// `.init_array`, `.fini_array`) from the ELF section headers, if present. The
/// entries are absolute Thumb function pointers (bit 0 set); we return them
/// verbatim (masking is the transpiler's job). Any structural problem (stripped
/// section headers, a table off the end of the file) yields `None`, so a module
/// without these sections simply contributes no seeds.
fn read_init_pointers(r: &Reader, bytes: &[u8]) -> Option<Vec<u32>> {
    let e_shoff = r.u32(0x20).ok()? as usize;
    let e_shentsize = r.u16(0x2e).ok()? as usize;
    let e_shnum = r.u16(0x30).ok()? as usize;
    let e_shstrndx = r.u16(0x32).ok()? as usize;
    if e_shoff == 0 || e_shnum == 0 || e_shstrndx >= e_shnum {
        return None;
    }
    // The section-header string table gives each section's name.
    let strtab_off = r.u32(e_shoff + e_shstrndx * e_shentsize + 0x10).ok()? as usize;

    let mut out = Vec::new();
    for i in 0..e_shnum {
        let sh = e_shoff + i * e_shentsize;
        let name_off = strtab_off + r.u32(sh).ok()? as usize;
        let rest = bytes.get(name_off..)?;
        let name = &rest[..rest.iter().position(|&b| b == 0)?];
        if !matches!(name, b".preinit_array" | b".init_array" | b".fini_array") {
            continue;
        }
        let sh_offset = r.u32(sh + 0x10).ok()? as usize;
        let sh_size = r.u32(sh + 0x14).ok()? as usize;
        for w in (0..sh_size).step_by(4) {
            let p = r.u32(sh_offset + w).ok()?;
            // 0 and ~0 are the empty-array sentinels some linkers emit.
            if p != 0 && p != u32::MAX {
                out.push(p);
            }
        }
    }
    Some(out)
}

/// Scan a fixed image's segments for word-aligned code pointers into its own
/// executable segments. Returns `(thumb_pointers, arm_pointers)`: an odd value
/// addressing executable code is a Thumb function pointer (Thumb bit stripped); an
/// even one is a tentative ARM function pointer. Values are absolute (a fixed
/// image is never rebased). Used only for `ET_SCE_EXEC`, whose function-pointer
/// tables carry no relocations to recover these from.
fn scan_fixed_code_pointers(segments: &[Segment]) -> (Vec<u32>, Vec<u32>) {
    let exec: Vec<(u32, u32)> = segments
        .iter()
        .filter(|s| s.executable)
        .map(|s| (s.vaddr, s.vaddr.wrapping_add(s.mem_size)))
        .collect();
    let in_exec = |a: u32| exec.iter().any(|&(lo, hi)| a >= lo && a < hi);

    let mut thumb = Vec::new();
    let mut arm = Vec::new();
    for s in segments {
        // Word-aligned scan: a function-pointer table entry is 4-byte aligned. The
        // segment's file-backed bytes hold every stored pointer (bss is zero).
        let n = s.data.len() & !3;
        let mut i = 0;
        while i < n {
            let w = u32::from_le_bytes([s.data[i], s.data[i + 1], s.data[i + 2], s.data[i + 3]]);
            if w != 0 && in_exec(w & !1) {
                if w & 1 == 1 {
                    thumb.push(w & !1);
                } else {
                    arm.push(w);
                }
            }
            i += 4;
        }
    }
    thumb.sort_unstable();
    thumb.dedup();
    arm.sort_unstable();
    arm.dedup();
    (thumb, arm)
}

/// Read a u16 at a guest vaddr (via the u32 reader, masking).
fn read_vaddr_u16(
    read_vaddr: &impl Fn(u32) -> Result<u32, Error>,
    vaddr: u32,
) -> Result<u16, Error> {
    Ok((read_vaddr(vaddr)? & 0xFFFF) as u16)
}

/// Read a NUL-terminated string of at most `max` bytes at `at`.
fn read_cstr(bytes: &[u8], at: usize, max: usize) -> String {
    let end = (at + max).min(bytes.len());
    let slice = &bytes[at.min(bytes.len())..end];
    let n = slice.iter().position(|&b| b == 0).unwrap_or(slice.len());
    String::from_utf8_lossy(&slice[..n]).into_owned()
}

#[cfg(test)]
mod tests {
    use super::*;

    // The committed cube corpus artifact.
    const CUBE: &[u8] = include_bytes!(
        "../../vitaslop-conformance-suite-vita/cube-src/cube.velf"
    );
    // The same cube wrapped as an fSELF eboot.bin: uncompressed (loadable) and
    // zlib-compressed (exercises the clean CompressedSelf error).
    const CUBE_EBOOT: &[u8] = include_bytes!(
        "../../vitaslop-conformance-suite-vita/cube-src/cube.eboot.bin"
    );
    const CUBE_EBOOT_C: &[u8] = include_bytes!(
        "../../vitaslop-conformance-suite-vita/cube-src/cube.eboot_c.bin"
    );

    #[test]
    fn loads_cube_header() {
        let m = load(CUBE).expect("load cube.velf");
        assert_eq!(m.name, "cube.elf");
        assert_eq!(m.base, 0x8100_0000);
        // Entry is module_start, not e_entry (which points at module_info).
        assert_eq!(m.entry, 0x8100_095d);
        // Two loadable segments: R-E code/rodata and RW bss.
        assert_eq!(m.segments.len(), 2);
        assert!(m.segments[0].executable);
        assert!(m.segments[1].writable);
        // bss: memory larger than its file backing.
        assert!(m.segments[1].mem_size as usize > m.segments[1].data.len());
    }

    #[test]
    fn resolves_all_nid_imports() {
        let m = load(CUBE).expect("load cube.velf");
        // 34 SceGxm + 1 SceDisplayUser + 1 SceCtrl + 2 SceSysmem = 38.
        assert_eq!(m.imports.len(), 38);

        // Every stub address lands inside the executable segment.
        let code = &m.segments[0];
        let code_end = code.vaddr + code.data.len() as u32;
        for imp in &m.imports {
            assert!(
                imp.stub_addr >= code.vaddr && imp.stub_addr < code_end,
                "stub {:#x} outside code segment",
                imp.stub_addr
            );
        }

        // Spot-check known (library_nid, func_nid) pairs from vita-elf-create.
        let has = |lib: u32, func: u32| {
            m.imports
                .iter()
                .any(|i| i.library_nid == lib && i.func_nid == func)
        };
        // SceGxm::sceGxmInitialize
        assert!(has(0xF76B_66BD, 0xB0F1_E4EC));
        // SceGxm::sceGxmDraw
        assert!(has(0xF76B_66BD, 0xBC05_9AFC));
        // SceDisplayUser::sceDisplaySetFrameBuf
        assert!(has(0x4FAA_CD11, 0x7A41_0B64));
        // SceCtrl::sceCtrlPeekBufferPositive
        assert!(has(0xD197_E3C7, 0xA9C3_CED6));
        // SceSysmem::sceKernelAllocMemBlock
        assert!(has(0x37FE_725A, 0xB9D5_EBDE));
    }

    #[test]
    fn lowers_to_transpiler_inputs() {
        let m = load(CUBE).expect("load cube.velf");
        let inputs = m.program_inputs();

        // The memory image spans every loadable segment (each placed at vaddr - base),
        // based at 0x81000000; the executable segment sits at the front.
        assert_eq!(inputs.base, 0x8100_0000);
        let expected_len = m
            .segments
            .iter()
            .map(|s| s.vaddr + s.mem_size - inputs.base)
            .max()
            .unwrap() as usize;
        assert_eq!(inputs.code.len(), expected_len);
        assert_eq!(&inputs.code[..m.segments[0].data.len()], &m.segments[0].data[..]);

        // The Vita entry is Thumb (odd address); the entry point clears bit 0.
        assert!(inputs.thumb_entry);
        assert_eq!(inputs.entries, vec![0x8100_095c]);

        // One extern per import, mapping each stub address to a dense index.
        assert_eq!(inputs.externs.len(), 38);
        for (i, ext) in inputs.externs.iter().enumerate() {
            assert_eq!(ext.import, i as u32);
            assert_eq!(ext.addr, m.imports[i].stub_addr);
        }

        // The borrowed Program view is well-formed.
        let prog = inputs.program();
        assert_eq!(prog.base, 0x8100_0000);
        assert_eq!(prog.entries.len(), 1);
    }

    #[test]
    fn detects_self_magic() {
        assert!(self_::is_self(CUBE_EBOOT));
        assert!(!self_::is_self(CUBE));
    }

    #[test]
    fn loads_cube_from_fself() {
        // Unwrapping the fSELF must yield the same module the bare velf does.
        // vita-make-fself patches module_nid (sha256 of the input), so that one
        // field legitimately differs; everything the pipeline consumes matches.
        let m = load(CUBE_EBOOT).expect("load cube.eboot.bin");
        let velf = load(CUBE).expect("load cube.velf");

        assert_eq!(m.name, velf.name);
        assert_eq!(m.base, velf.base);
        assert_eq!(m.entry, velf.entry);
        assert_eq!(m.segments.len(), velf.segments.len());
        // Segment bytes are identical except for module_nid: vita-make-fself
        // computes it (sha256 of the velf) and patches the 4-byte field, while
        // the bare velf still carries its 0 placeholder. So the fSELF module
        // carries the real NID and the segment data differs by exactly those
        // 4 bytes and nothing else.
        assert_ne!(m.module_nid, velf.module_nid);
        assert_eq!(velf.module_nid, 0);
        let mut diff_bytes = 0usize;
        for (a, b) in m.segments.iter().zip(&velf.segments) {
            assert_eq!(a.vaddr, b.vaddr);
            assert_eq!(a.mem_size, b.mem_size);
            assert_eq!(a.executable, b.executable);
            assert_eq!(a.writable, b.writable);
            assert_eq!(a.data.len(), b.data.len());
            diff_bytes += a.data.iter().zip(&b.data).filter(|(x, y)| x != y).count();
        }
        assert_eq!(diff_bytes, 4, "only the module_nid word may differ");
        assert_eq!(m.imports.len(), velf.imports.len());
        for (a, b) in m.imports.iter().zip(&velf.imports) {
            assert_eq!(a.library_nid, b.library_nid);
            assert_eq!(a.func_nid, b.func_nid);
            assert_eq!(a.stub_addr, b.stub_addr);
        }
    }

    #[test]
    fn unwrapped_fself_transpiles_identically() {
        // The unwrapped image lowers to the same transpiler inputs as the velf,
        // so the whole downstream pipeline is unaffected by the container.
        let m = load(CUBE_EBOOT).expect("load cube.eboot.bin");
        let inputs = m.program_inputs();
        assert_eq!(inputs.base, 0x8100_0000);
        assert!(inputs.thumb_entry);
        assert_eq!(inputs.entries, vec![0x8100_095c]);
        assert_eq!(inputs.externs.len(), 38);
    }

    #[test]
    fn loads_cube_from_compressed_fself() {
        // A zlib-compressed eboot (vita-make-fself -c) inflates to the identical
        // module the uncompressed eboot yields - proving the built-in inflate.
        let c = load(CUBE_EBOOT_C).expect("load compressed cube.eboot_c.bin");
        let u = load(CUBE_EBOOT).expect("load cube.eboot.bin");

        assert_eq!(c.name, u.name);
        assert_eq!(c.base, u.base);
        assert_eq!(c.entry, u.entry);
        assert_eq!(c.module_nid, u.module_nid);
        assert_eq!(c.segments.len(), u.segments.len());
        for (a, b) in c.segments.iter().zip(&u.segments) {
            assert_eq!(a.vaddr, b.vaddr);
            assert_eq!(a.mem_size, b.mem_size);
            assert_eq!(a.data, b.data);
        }
        assert_eq!(c.imports.len(), u.imports.len());
        for (a, b) in c.imports.iter().zip(&u.imports) {
            assert_eq!(a.library_nid, b.library_nid);
            assert_eq!(a.func_nid, b.func_nid);
            assert_eq!(a.stub_addr, b.stub_addr);
        }
    }
}
