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

/// `e_type` of a velf: a relocatable SCE executable.
const ET_SCE_RELEXEC: u16 = 0xFE04;
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

/// A parsed Vita module.
pub struct Module {
    /// Module name from `SceModuleInfo` (e.g. "cube.elf").
    pub name: String,
    pub module_nid: u32,
    /// Lowest segment vaddr - where the image begins.
    pub base: u32,
    /// The entry point (`SceModuleInfo::module_start`), where execution begins.
    pub entry: u32,
    pub segments: Vec<Segment>,
    /// Function imports, in import-table order.
    pub imports: Vec<Import>,
    /// Static constructor/destructor function pointers, read from the
    /// `.preinit_array`/`.init_array`/`.fini_array` sections (when the module
    /// keeps section headers). These are reachable only through an indirect call
    /// (`__libc_init_array` walks the table and `blx`es each), so they seed
    /// transpiler discovery that the direct-call closure alone would miss. Empty
    /// for `-nostdlib` modules and for anything with its section headers stripped.
    pub init_pointers: Vec<u32>,
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
            externs: &self.externs,
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
    if e_type != ET_SCE_RELEXEC {
        return Err(Error::UnsupportedType(e_type));
    }

    // For ET_SCE_RELEXEC, e_entry locates SceModuleInfo: top two bits are the
    // program-header (segment) index, the rest is the offset within it.
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
    let import_top_off = r.u32(mi_file + 0x2c)?;
    let import_end_off = r.u32(mi_file + 0x30)?;
    let module_nid = r.u32(mi_file + 0x34)?;
    let module_start_off = r.u32(mi_file + 0x44)?;

    let seg_base = mi_seg.vaddr;
    let entry = seg_base.wrapping_add(module_start_off);
    let import_top = seg_base.wrapping_add(import_top_off);
    let import_end = seg_base.wrapping_add(import_end_off);

    // Walk the import table: fixed-size entries from import_top to import_end.
    let mut imports = Vec::new();
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

        cursor = cursor.wrapping_add(step);
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

    Ok(Module {
        name,
        module_nid,
        base,
        entry,
        segments,
        imports,
        init_pointers,
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
