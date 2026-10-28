use super::*;

impl VitaState {
    /// Size in bytes of a program's default uniform buffer, computed from its
    /// reflected parameter table: the maximum `resource_index + <registers this
    /// parameter occupies>` over the uniform (`category == 1`) parameters, times 4.
    /// The program header's own size field (+0x2C) under-reports for shaders with a
    /// large uniform block (e.g. a world matrix at float 0 plus a world-to-projection
    /// matrix at float 16 plus lighting/fog), truncating the captured buffer and
    /// dropping the view-projection - so the reflected extent is the reliable size.
    ///
    /// A parameter's registers are NOT its component count: see the width table in
    /// [`reflect_program_uncached`] for what counting them that way cost.
    pub(super) fn reflected_uniform_size_bytes(&mut self, ctx: &GuestCtx, header: u32) -> u32 {
        self.reflect_program(ctx, header).uniform_size_bytes
    }

    /// Drop every cached [`ProgramReflection`]. Called when the guest registers or
    /// unregisters a program with the shader patcher - the only points at which a
    /// header address can start meaning a different program - so a reused address can
    /// never be read through the old program's layout.
    ///
    /// A whole-table clear rather than a targeted eviction: registration happens a few
    /// hundred times over a run, at load, and the next draw simply re-reflects. Being
    /// obviously correct is worth more here than evicting one entry.
    pub fn invalidate_program_reflection(&mut self) {
        self.program_memos.clear();
    }

    /// The blend compiled into the epilogue of the fragment program at `header`
    /// (`vitaslop_gxp_shader::rop_blend`), decoded once per registered program.
    pub(crate) fn program_rop_blend(
        &mut self,
        ctx: &GuestCtx,
        header: u32,
    ) -> Option<vitaslop_gxp_shader::RopBlend> {
        if let Some(r) = self.program_memos.rop_blend(header) {
            return r;
        }
        let blob = self.program_blob(ctx, header);
        let r = vitaslop_gxp_shader::rop_blend(&blob);
        self.program_memos.set_rop_blend(header, r);
        r
    }

    /// The guest-memory windows the VERTEX program at `header` needs snapshotted per draw,
    /// if any (see `vitaslop_gxp_shader::mem_windows_for_vertex_blob`). Memoised: the decode
    /// behind it runs once per registered program, and the per-draw cost for the near-total
    /// majority of programs that need no window is this one map lookup.
    pub(super) fn mem_window_spec(
        &mut self,
        ctx: &GuestCtx,
        header: u32,
        stage: ProgramStage,
    ) -> std::sync::Arc<[vitaslop_gxp_shader::MemWindow]> {
        if header == 0 {
            return std::sync::Arc::from(&[][..]);
        }
        // The header address alone does NOT identify the spec: a fragment program and a vertex
        // program are different headers, so one map is enough, but the RESOLVER differs by
        // stage (each refuses a blob of the other kind), so the stage is part of the key.
        let key = (header, matches!(stage, ProgramStage::Fragment));
        if let Some(spec) = self.program_memos.mem_windows(key) {
            return spec;
        }
        let blob = self.program_blob(ctx, header);
        let spec: std::sync::Arc<[vitaslop_gxp_shader::MemWindow]> =
            std::sync::Arc::from(match stage {
                ProgramStage::Vertex => vitaslop_gxp_shader::mem_windows_for_vertex_blob(&blob),
                ProgramStage::Fragment => vitaslop_gxp_shader::mem_windows_for_fragment_blob(&blob),
            });
        self.program_memos.set_mem_windows(key, spec.clone());
        spec
    }

    /// The SA-register layout of one program's uniform BUFFERS - see [`SaUniformLayout`].
    /// Memoised per registered program beside [`Self::mem_window_spec`].
    pub(super) fn sa_uniform_layout(
        &mut self,
        ctx: &GuestCtx,
        header: u32,
    ) -> std::sync::Arc<SaUniformLayout> {
        if header == 0 {
            return SaUniformLayout::none();
        }
        if let Some(l) = self.program_memos.sa_uniforms(header) {
            return l;
        }
        let blob = self.program_blob(ctx, header);
        let layout = std::sync::Arc::new(match vitaslop_gxp_shader::Program::parse(&blob) {
            Ok(p) => {
                let (default_base, default_regs) = p.default_uniform_sa_span();
                let short = p.default_uniform_overflows_its_container();
                if short > 0 {
                    // ...unless the program can reach the tail by POINTER. A +0x78 binding
                    // entry naming buffer 14 is the driver leaving the default uniform
                    // buffer's ADDRESS in a DATA slot: the copy into the SA file carries the
                    // head and the program loads the rest through that pointer, which is
                    // exactly what a header declaring more than its own container holds has
                    // been describing. Reporting those as short is a false alarm, and a
                    // warning that is wrong for the ordinary case trains its reader to skip it.
                    if !p
                        .uniform_buffer_bindings
                        .iter()
                        .any(|b| b.buffer_index == 14)
                    {
                        report_default_uniform_container_short(header, default_regs, short);
                    }
                }
                SaUniformLayout {
                    default_base,
                    default_regs,
                    buffers: p.sa_uniform_buffers(),
                }
            }
            Err(_) => SaUniformLayout::default(),
        });
        self.program_memos.set_sa_uniforms(header, layout.clone());
        layout
    }

    /// Lay this draw's uniform bytes out the way the DRIVER lays them out in the SA register
    /// file, and return the whole image: the default uniform buffer at its own container's
    /// base, and every non-default buffer the program declares an SA block for at its.
    ///
    /// # This is what a recompiled shader reads, and it used to be one buffer at register 0
    /// The emitted WGSL indexes `sa[k]` by ABSOLUTE SA register, so the bytes bound to it have
    /// to BE the register file, not one of the blocks that lives in it. That distinction is
    /// invisible on every title whose container table puts the default buffer at register 0
    /// with nothing else resident - which is every title captured before this one - and it is
    /// the difference between a picture and a black frame on a title that keeps its transform
    /// in container 0 and declares a default buffer of size zero.
    ///
    /// A buffer the guest has not bound contributes nothing and leaves its registers zero,
    /// which is what they read as today. That is not silently fine, so it is REPORTED - once
    /// per program - by the caller's own diagnostic (`report_unfed_uniforms` on the renderer
    /// side names the program; this names the draw's missing bind).
    /// The guest ranges [`Self::sa_uniform_image`] will read for `header`'s SA-resident buffers,
    /// appended as `(address, length)` - so a draw can fetch them with its other uniform banks.
    /// None for the usual program (default buffer at register 0, nothing else resident).
    pub(super) fn sa_uniform_ranges(
        &mut self,
        ctx: &GuestCtx,
        blk: &crate::vita::gxmctx::Block<'_>,
        header: u32,
        stage: UniformStage,
        out: &mut Vec<(u32, usize)>,
    ) {
        let layout = self.sa_uniform_layout(ctx, header);
        for b in layout.buffers.iter() {
            let addr = match stage {
                UniformStage::Vertex => blk.vertex_uniform_buffer(b.buffer_index),
                UniformStage::Fragment => blk.fragment_uniform_buffer(b.buffer_index),
            };
            if addr != 0 {
                out.push((addr, b.size_regs as usize * 4));
            }
        }
    }

    pub(super) fn sa_uniform_image(
        &mut self,
        ctx: &GuestCtx,
        blk: &crate::vita::gxmctx::Block<'_>,
        header: u32,
        stage: UniformStage,
        default_bytes: Arc<[u8]>,
    ) -> Arc<[u8]> {
        let layout = self.sa_uniform_layout(ctx, header);
        // The overwhelming majority of programs: the default buffer at register 0 and nothing
        // else SA-resident. The bytes ARE the image, with no copy and no allocation - and now
        // no conversion either: the caller's `Arc` comes straight back, where it used to be
        // rebuilt from a `Vec` one line later.
        if layout.buffers.is_empty() && layout.default_base == 0 {
            return default_bytes;
        }
        let extent_regs = layout
            .buffers
            .iter()
            .map(|b| b.base_sa + b.size_regs)
            .fold(layout.default_base + layout.default_regs, u32::max);
        // ONE allocation, the Arc itself, and each bound buffer is read STRAIGHT INTO it.
        // This runs twice per draw (one stage each): the previous shape - a Vec, a Vec per
        // buffer from `read_bytes`, then `Vec -> Arc` - was four allocations and two extra
        // copies per stage, inside `sceGxmDraw`, on a title that draws 600 times a frame.
        let mut image: Arc<[u8]> = unsafe {
            // SAFETY: a zeroed byte is an initialised byte.
            Arc::<[std::mem::MaybeUninit<u8>]>::assume_init(Arc::new_zeroed_slice(
                extent_regs as usize * 4,
            ))
        };
        {
            let img = Arc::get_mut(&mut image).expect("freshly created, uniquely owned");
            let at = layout.default_base as usize * 4;
            let n = default_bytes.len().min(img.len().saturating_sub(at));
            img[at..at + n].copy_from_slice(&default_bytes[..n]);
            for b in layout.buffers.iter() {
                let addr = match stage {
                    UniformStage::Vertex => blk.vertex_uniform_buffer(b.buffer_index),
                    UniformStage::Fragment => blk.fragment_uniform_buffer(b.buffer_index),
                };
                if addr == 0 {
                    report_unbound_sa_uniform_buffer(header, stage, b.buffer_index);
                    continue;
                }
                let at = b.base_sa as usize * 4;
                let n = (b.size_regs as usize * 4).min(img.len().saturating_sub(at));
                ctx.read_into(addr, &mut img[at..at + n]);
                if watch_uniform_rewrites() {
                    UNIFORM_WATCH
                        .lock()
                        .unwrap_or_else(|e| e.into_inner())
                        .push((
                            addr,
                            n as u32,
                            stream_watch_hash(&img[at..at + n]),
                            header,
                            b.buffer_index,
                        ));
                }
            }
        }
        // The SA-resident path assembles a fresh image, so this one conversion is unavoidable
        // - and it is the rare arm, taken only by a program that declares SA-resident buffers.
        image
    }

    /// The buffer index a +0x78 entry gives the DEFAULT uniform buffer. Its address does not
    /// live in the context block's non-default table - it is what the stage's own
    /// `sceGxmReserve*DefaultUniformBuffer` handed back.
    pub(super) const DEFAULT_UNIFORM_BUFFER_INDEX: u32 = 14;

    /// Snapshot the guest-memory windows this draw's VERTEX program reads through its 0xE8
    /// memory loads: `(window guest base address, the window's bytes)` each, in the order the
    /// shader's `gxp_mem` binding lays them out.
    ///
    /// A bound address comes from the context block's own table - written by
    /// `sceGxmSetVertexUniformBuffer` directly or by binding a precomputed vertex state -
    /// except for the DEFAULT uniform buffer, whose address is the one the stage's reserve
    /// handed back. The extent is the program's own declared buffer size (see
    /// `vitaslop_gxp_shader::MemWindow`). Snapshotted AT DRAW TIME like every other guest
    /// input the capture carries, so the renderer later reads what the guest had bound now.
    ///
    /// Returns EMPTY (reported, throttled to once per run) when any window has nothing bound
    /// or a base that is not 4-aligned - the renderer then DROPS the draw with a report rather
    /// than feeding the loads fabricated bytes. All or nothing: a draw with some of its
    /// windows fed would read zeroes through the rest with nothing to say so.
    /// Say, once per (vertex program, buffer), WHERE a guest-memory window points and whether
    /// the bytes behind it are all zero.
    ///
    /// A vertex program that loads its transform through a pointer reads zeroes if the window
    /// is bound but empty, and a zero matrix collapses every vertex onto the origin - which
    /// looks exactly like a draw that was never issued. Nothing else in the pipeline can tell
    /// those apart: the draw is prepared, the pipeline is built, the pass encodes it, and the
    /// frame is black. [[vitaslop-poison-separates-a-guest-zero-from-an-unwritten-one]]
    ///
    /// STATUS when the window has content - that is the healthy path and it is worth being able
    /// to see the address. A WARNING when it is all zero, because a transform of zeroes is not
    /// something a title asks for.
    pub(super) fn report_mem_window(vheader: u32, buffer_index: u32, at: u32, bytes: &[u8]) {
        use std::sync::atomic::{AtomicU32, Ordering};
        // >>> CALLED ONCE PER (PROGRAM, BUFFER) - the caller keeps the set (see
        // `VitaState::mem_windows_zero_tested`) - and silent after `MAX_REPORTS`.
        //
        // It used to run for every window of every draw: a byte scan, then a SipHash of every
        // byte and a global lock to dedupe a warning that had already fired - on a baseball title's at-bat
        // that is 231 crowd draws a frame, each with a 3,776-byte skinning window that is all
        // zero at the draw (a job thread fills it later). Measured on the phone, removing the
        // hash alone did NOT move the window-snapshot phase, so this is waste, not the phase's
        // main cost; it is gone because a warning only needs saying once.
        const MAX_REPORTS: u32 = 6;
        static REPORTED: AtomicU32 = AtomicU32::new(0);
        if REPORTED.load(Ordering::Relaxed) >= MAX_REPORTS {
            return;
        }
        if !bytes.iter().all(|b| *b == 0) {
            return;
        }
        REPORTED.fetch_add(1, Ordering::Relaxed);
        {
            tracing::warn!(
                target: "vitaslop::gxm",
                vertex_program = format_args!("{vheader:#x}"),
                buffer_index,
                at = format_args!("{at:#x}"),
                len = bytes.len(),
                "a program's guest-memory WINDOW is bound but every byte behind it is ZERO. A transform loaded from it is a zero matrix, which collapses every vertex onto the origin, and a BLEND coefficient loaded from it multiplies the draw away - either way a frame that looks identical to one where the draw was never issued."
            );
        }
        // A window that carries data is the healthy case and is NOT reported: once per
        // (program, slot) it was 90 of the 96 status lines a device panel keeps, on a title
        // with a hundred programs, and the lines it displaced were the ones the panel was
        // read for. What a window holds at a given draw is `VITASLOP_DUMP_DRAW_GXP`'s job.
    }

    /// `VITASLOP_MEM_DUMP=<hex addr>:<bytes>[,...]`: write those guest byte ranges to
    /// `mem-<addr>.bin` in the working directory, once, at the first END SCENE or
    /// window-bearing draw of the display frame named by `VITASLOP_MEM_DUMP_AT` (or
    /// immediately if that is unset). `VITASLOP_MEM_DUMP_AT=<f>,<f>,...` dumps once at EACH
    /// listed frame instead, to `mem-<addr>-f<frame>.bin`.
    ///
    /// # Why a whole-range dump rather than another targeted print
    /// A wrong UNIFORM has two possible owners - the guest computed the value, or we stored it
    /// wrongly - and the only thing that separates them is the guest's own F32 source next to
    /// the F16 the shader reads. `VITASLOP_GXP_MEM_PEEK` prints eight words at two fixed
    /// offsets, which answers "is the window placed right"; it cannot answer "is the value
    /// this block was built FROM anywhere nearby". A range on disk can be searched for any
    /// bit pattern, which is what turns a value question into a grep.
    /// [`Self::dump_guest_ranges`] at the current display frame, for a caller that is not a
    /// draw. See the END SCENE site in `vita::gxm::end_scene`.
    pub fn dump_guest_ranges_now(&self, ctx: &GuestCtx) {
        Self::dump_guest_ranges(ctx, self.cur_frame);
        Self::watch_guest_range_changes(ctx, self.cur_frame, None);
        self.scan_guest_words(ctx);
    }

    /// `VITASLOP_MEM_SCAN=<frame>:<hex word>[,<hex word>...]`: at the first scene end at or
    /// after `frame`, list every 4-aligned guest address holding each word (64 at most each).
    ///
    /// The question "what points at this object" has no address to aim a watch at - the
    /// pointers are wherever the title keeps them - and a scan answers it in one run. The
    /// browser has its own (`VITASLOP_MEM_FIND`); this is the desktop's.
    pub(super) fn scan_guest_words(&self, ctx: &GuestCtx) {
        use std::sync::OnceLock;
        use std::sync::atomic::{AtomicBool, Ordering};
        static SPEC: OnceLock<Option<(u64, Vec<u32>)>> = OnceLock::new();
        static DONE: AtomicBool = AtomicBool::new(false);
        let Some((at, words)) = SPEC.get_or_init(|| {
            let v = crate::knobs::var("VITASLOP_MEM_SCAN").ok()?;
            let (f, list) = v.split_once(':')?;
            let words = list
                .split(',')
                .filter_map(|w| u32::from_str_radix(w.trim().trim_start_matches("0x"), 16).ok())
                .collect();
            Some((f.trim().parse().ok()?, words))
        }) else {
            return;
        };
        if self.cur_frame < *at || DONE.swap(true, Ordering::Relaxed) {
            return;
        }
        let mut hits: Vec<Vec<u32>> = vec![Vec::new(); words.len()];
        const CHUNK: u32 = 1 << 20;
        let mut addr = self.base;
        let end = self.base.wrapping_add(self.mem_bytes);
        while addr < end {
            let n = CHUNK.min(end - addr);
            let bytes = ctx.read_bytes(addr, n as usize);
            for (i, c) in bytes.chunks_exact(4).enumerate() {
                let v = u32::from_le_bytes([c[0], c[1], c[2], c[3]]);
                for (k, w) in words.iter().enumerate() {
                    if v == *w && hits[k].len() < 64 {
                        hits[k].push(addr + 4 * i as u32);
                    }
                }
            }
            addr = addr.wrapping_add(n);
        }
        for (w, h) in words.iter().zip(&hits) {
            let list: Vec<String> = h.iter().map(|a| format!("{a:#010x}")).collect();
            eprintln!(
                "MEM-SCAN frame {}: {w:#010x} found at {} place(s): {}",
                self.cur_frame,
                h.len(),
                list.join(" ")
            );
        }
    }

    /// `VITASLOP_MEM_CHANGE_WATCH=<hex addr>:<bytes>[,...]`: at every scene end, report each
    /// range whose bytes differ from the last look - the frame and the new head bytes.
    ///
    /// The store watchpoint covers one exact word and the host-write watch covers host calls;
    /// a range overwritten by anything else (a bulk or vector store whose start is elsewhere,
    /// a renderer write-back) is invisible to both, and "neither fired" then reads as "nothing
    /// wrote it". This answers WHEN it changed, whoever did it, so the next instrument can be
    /// aimed at that frame.
    ///
    /// Also checked at EVERY host call when armed (`at` = thread, return address, NID), which
    /// pins a change between two host calls of one thread - close enough to disassemble.
    pub(crate) fn watch_guest_range_changes(
        ctx: &GuestCtx,
        frame: u64,
        at: Option<(u32, u32, u32)>,
    ) {
        use std::sync::{Mutex, OnceLock};
        static SPEC: OnceLock<Vec<(u32, u32)>> = OnceLock::new();
        static LAST: Mutex<Vec<Option<Vec<u8>>>> = Mutex::new(Vec::new());
        let spec = SPEC.get_or_init(|| {
            let Ok(v) = crate::knobs::var("VITASLOP_MEM_CHANGE_WATCH") else {
                return Vec::new();
            };
            v.split(',')
                .filter_map(|one| {
                    let (a, n) = one.trim().split_once(':')?;
                    Some((
                        u32::from_str_radix(a.trim_start_matches("0x"), 16).ok()?,
                        n.parse().ok()?,
                    ))
                })
                .collect()
        });
        if spec.is_empty() {
            return;
        }
        // The host call BEFORE this one, any thread: one-baton, the write happened between it
        // and this call.
        static PREV: Mutex<Option<(u32, u32, u32)>> = Mutex::new(None);
        // ...and the CURRENT thread's own previous host call: the write is in this thread's code
        // between that call and this one when no other thread's call came in between.
        static PREV_OWN: Mutex<Option<std::collections::HashMap<u32, (u32, u32)>>> =
            Mutex::new(None);
        let prev = {
            let mut p = PREV.lock().unwrap_or_else(|e| e.into_inner());
            let was = *p;
            if at.is_some() {
                *p = at;
            }
            was
        };
        let prev_own = at.and_then(|(t, lr, nid)| {
            let mut m = PREV_OWN.lock().unwrap_or_else(|e| e.into_inner());
            m.get_or_insert_with(Default::default).insert(t, (lr, nid))
        });
        let mut last = LAST.lock().unwrap_or_else(|e| e.into_inner());
        last.resize(spec.len(), None);
        for (i, &(addr, len)) in spec.iter().enumerate() {
            let now = ctx.read_bytes(addr, len as usize);
            if last[i].as_ref() != Some(&now) {
                let head: String = now.iter().take(32).map(|b| format!("{b:02x}")).collect();
                let site = at.map_or_else(
                    || "scene end".to_string(),
                    |(t, lr, nid)| {
                        format!("entering {} on t{t} lr={lr:#010x}", crate::nid::name(nid))
                    },
                );
                let site = match prev {
                    Some((t, lr, nid)) => format!(
                        "{site}; previous host call {} on t{t} lr={lr:#010x}",
                        crate::nid::name(nid)
                    ),
                    None => site,
                };
                let site = match prev_own {
                    Some((lr, nid)) => format!(
                        "{site}; this thread's previous host call {} lr={lr:#010x}",
                        crate::nid::name(nid)
                    ),
                    None => site,
                };
                eprintln!(
                    "MEM-CHANGE frame {frame} ({site}): {len} bytes at {addr:#010x} now start {head}"
                );
                last[i] = Some(now);
            }
        }
    }

    pub(super) fn dump_guest_ranges(ctx: &GuestCtx, frame: u64) {
        use std::sync::OnceLock;
        static SPEC: OnceLock<Vec<(u32, u32)>> = OnceLock::new();
        static AT: OnceLock<Vec<u64>> = OnceLock::new();
        static DONE: std::sync::atomic::AtomicBool = std::sync::atomic::AtomicBool::new(false);
        static ARMED: std::sync::atomic::AtomicBool = std::sync::atomic::AtomicBool::new(false);
        let spec = SPEC.get_or_init(|| {
            let Ok(v) = crate::knobs::var("VITASLOP_MEM_DUMP") else {
                return Vec::new();
            };
            v.split(',')
                .filter_map(|one| {
                    let (a, n) = one.trim().split_once(':')?;
                    // BOTH halves accept a `0x` prefix. A length silently parsed as decimal
                    // when it was written as hex drops the whole entry and leaves the spec
                    // empty, which reads as "the knob did nothing".
                    let num = |t: &str| {
                        let t = t.trim();
                        match t.strip_prefix("0x") {
                            Some(h) => u32::from_str_radix(h, 16).ok(),
                            None => t.parse().ok(),
                        }
                    };
                    Some((
                        u32::from_str_radix(a.trim().trim_start_matches("0x"), 16).ok()?,
                        num(n)?,
                    ))
                })
                .collect()
        });
        // Called for every window-bearing draw: with the knob unset this is the whole cost - a
        // cached empty spec - and nothing below, not even the one-time "reached" line, runs.
        static KNOB_SET: std::sync::OnceLock<bool> = std::sync::OnceLock::new();
        if spec.is_empty()
            && !*KNOB_SET.get_or_init(|| crate::knobs::var("VITASLOP_MEM_DUMP").is_ok())
        {
            return;
        }
        if !ARMED.load(std::sync::atomic::Ordering::Relaxed)
            && !ARMED.swap(true, std::sync::atomic::Ordering::Relaxed)
        {
            eprintln!(
                "MEM-DUMP: reached at frame {frame}; VITASLOP_MEM_DUMP={:?}",
                crate::knobs::var("VITASLOP_MEM_DUMP").ok()
            );
        }
        if spec.is_empty() || DONE.load(std::sync::atomic::Ordering::Relaxed) {
            return;
        }

        // `VITASLOP_MEM_DUMP_AT=<frame>[,<frame>...]`: one dump at each listed frame. Several
        // frames of ONE run are what a value search needs - the same heap, the state stepped
        // between them - and two runs cannot give that once the heap layout drifts.
        let frames = AT.get_or_init(|| {
            let mut v: Vec<u64> = crate::knobs::var("VITASLOP_MEM_DUMP_AT")
                .ok()
                .map(|s| s.split(',').filter_map(|f| f.trim().parse().ok()).collect())
                .unwrap_or_default();
            v.sort_unstable();
            v
        });
        static NEXT: std::sync::atomic::AtomicUsize = std::sync::atomic::AtomicUsize::new(0);
        let next = NEXT.load(std::sync::atomic::Ordering::Relaxed);
        if frames.len() > 1 {
            if frames.get(next).is_none_or(|&w| frame < w) {
                return;
            }
            // Every listed frame this one has passed is answered by this one dump.
            let passed = frames.iter().filter(|&&w| w <= frame).count();
            NEXT.store(passed, std::sync::atomic::Ordering::Relaxed);
            if passed == frames.len() {
                DONE.store(true, std::sync::atomic::Ordering::Relaxed);
            }
            for &(addr, len) in spec {
                let bytes: Vec<u8> = ctx.read_bytes(addr, len as usize);
                let path = format!("mem-{addr:08x}-f{frame}.bin");
                match std::fs::write(&path, &bytes) {
                    Ok(()) => {
                        eprintln!("MEM-DUMP frame {frame}: {len} bytes of {addr:#010x} -> {path}")
                    }
                    Err(e) => eprintln!("MEM-DUMP {addr:#010x}: {e}"),
                }
            }
            return;
        }
        let at = frames.first().copied();
        if at.is_some_and(|w| frame < w) {
            // Not there yet on THIS counter. Re-dump periodically anyway, overwriting, so the
            // file on disk always holds the LATEST state - a gate on a frame counter the caller
            // cannot see is otherwise a dump that silently never happens, which is the failure
            // this instrument exists to avoid.
            static N: std::sync::atomic::AtomicU64 = std::sync::atomic::AtomicU64::new(0);
            if !N
                .fetch_add(1, std::sync::atomic::Ordering::Relaxed)
                .is_multiple_of(65536)
            {
                return;
            }
        } else {
            DONE.store(true, std::sync::atomic::Ordering::Relaxed);
        }
        for &(addr, len) in spec {
            let bytes: Vec<u8> = ctx.read_bytes(addr, len as usize);
            let path = format!("mem-{addr:08x}.bin");
            match std::fs::write(&path, &bytes) {
                Ok(()) => {
                    eprintln!("MEM-DUMP frame {frame}: {len} bytes of {addr:#010x} -> {path}")
                }
                Err(e) => eprintln!("MEM-DUMP {addr:#010x}: {e}"),
            }
        }
    }

    /// Print the guest words at a window's base and at `base + VITASLOP_GXP_MEM_PEEK` bytes,
    /// once per (vertex program, buffer).
    ///
    /// # Why a knob for this exists
    /// A shader can add a compile-time LITERAL to a load's address, and when that literal is
    /// bigger than the window the read lands in no bound window and reads zero. Two readings
    /// then explain the same picture - the literal is real and this renderer places the window
    /// wrong, or the literal is not a byte displacement at all - and only the guest's own
    /// memory separates them: if the bytes THERE are the data the shader wants, the window is
    /// wrong; if they are not, the addend is.
    pub(super) fn peek_past_mem_window(ctx: &GuestCtx, vheader: u32, buffer_index: u32, at: u32) {
        use std::sync::{Mutex, OnceLock};
        static WANT: OnceLock<Option<u32>> = OnceLock::new();
        let Some(delta) = *WANT.get_or_init(|| {
            std::env::var("VITASLOP_GXP_MEM_PEEK").ok().and_then(|s| {
                let s = s.trim();
                u32::from_str_radix(s.trim_start_matches("0x"), 16).ok()
            })
        }) else {
            return;
        };
        static SEEN: OnceLock<Mutex<std::collections::HashSet<(u32, u32)>>> = OnceLock::new();
        let seen = SEEN.get_or_init(|| Mutex::new(std::collections::HashSet::new()));
        if !seen
            .lock()
            .unwrap_or_else(|e| e.into_inner())
            .insert((vheader, buffer_index))
        {
            return;
        }
        let words = |a: u32| -> String {
            (0..8)
                .map(|i| {
                    let w = ctx.read_u32(a.wrapping_add(i * 4));
                    format!("{w:#010x}({:.4})", f32::from_bits(w))
                })
                .collect::<Vec<_>>()
                .join(" ")
        };
        eprintln!(
            "MEM-PEEK vprog={vheader:#x} buffer={buffer_index}
  at   {at:#x}: {}
  +{delta:#x} {:#x}: {}",
            words(at),
            at.wrapping_add(delta),
            words(at.wrapping_add(delta))
        );
    }

    /// Say that a program with MEMORY LOADS was handed a uniform-buffer base it cannot use, and
    /// say it PER (program, stage, buffer) rather than once for the whole run.
    ///
    /// # Why the key matters more than the message
    /// This report used to be a single global `AtomicBool`: the first unusable window in a run
    /// printed, and every other program, stage and buffer index after it was silent. A title
    /// dropping five hundred draws a frame across several programs therefore produced ONE line
    /// naming ONE of them, which reads as an isolated curiosity rather than as the whole of that
    /// title's fallback population. The key is the thing being counted, so the key is the thing
    /// the report has to be keyed on [[vitaslop-a-diagnostic-keyed-on-the-immediate-lr-collapses-behind-a-wrapper]].
    ///
    /// The running count goes with it: a first sighting cannot say whether this costs one draw
    /// or a frame's worth, and "how many" is exactly the question a reader arrives with.
    pub(super) fn report_unbound_mem_window(
        vheader: u32,
        stage: ProgramStage,
        buffer_index: u32,
        addr: u32,
        windows: usize,
        table: &[u32],
        default_buf: u32,
    ) {
        use std::sync::{Mutex, OnceLock};
        static SEEN: OnceLock<Mutex<std::collections::HashMap<(u32, bool, u32), u64>>> =
            OnceLock::new();
        let frag = matches!(stage, ProgramStage::Fragment);
        let seen = SEEN.get_or_init(|| Mutex::new(std::collections::HashMap::new()));
        let n = {
            let mut g = seen.lock().unwrap_or_else(|e| e.into_inner());
            let c = g.entry((vheader, frag, buffer_index)).or_insert(0);
            *c += 1;
            *c
        };
        // First sighting, then a geometric thinning: the population is what the reader needs and
        // one line per dropped draw would bury it under itself.
        if n != 1 && !n.is_multiple_of(10_000) {
            return;
        }
        let why = if addr == 0 {
            "nothing is bound to it"
        } else {
            "its base is not 4-aligned, which the 32-bit loads cannot address"
        };
        let stage_name = if frag { "fragment" } else { "vertex" };
        // Which slots of the stage's table DO hold an address, so the reader can tell "the
        // guest never binds a non-default buffer at all" from "every other slot is bound and
        // this one is not". Both writers of this table are INLINED into guest code
        // (`StoreArgIndexed` for the direct setter, `BindPrecomputedState` for the precomputed
        // one), so no Rust handler sees them and reading the table back here is the only place
        // the distinction is visible.
        let bound: Vec<String> = table
            .iter()
            .enumerate()
            .filter(|&(_, &a)| a != 0)
            .map(|(i, a)| format!("{i}={a:#x}"))
            .collect();
        let verdict = if bound.is_empty() {
            "NO slot of this stage's non-default table is bound at this draw, so the open              question is whether the guest binds one at all - not why this index is missing."
        } else {
            "OTHER slots of this stage's table ARE bound, so the binding path works and this              INDEX is the one that does not arrive."
        };
        let bound_slots = if bound.is_empty() {
            "none".to_string()
        } else {
            bound.join(" ")
        };
        tracing::warn!(
            target: "vitaslop::gxm",
            program = format_args!("{vheader:#x}"),
            stage = stage_name,
            buffer_index,
            addr = format_args!("{addr:#x}"),
            windows,
            dropped_so_far = n,
            bound_slots,
            default_uniform_buffer = format_args!("{default_buf:#x}"),
            "a {stage_name} program with MEMORY LOADS wants uniform buffer {buffer_index} and              {why} - ALL {windows} of its windows are withheld and the draw is DROPPED, rather              than fed fabricated bytes. {verdict}"
        );
    }

    /// The address the guest bound for window `w` of a `stage` program. EACH STAGE READS ITS OWN
    /// TABLE: `sceGxmSetVertexUniformBuffer` and `sceGxmSetFragmentUniformBuffer` are separate
    /// bindings at the same index, so taking the vertex one for a fragment program feeds the
    /// loads another buffer's bytes - a wrong picture with nothing to say so.
    fn bound_window_address(
        blk: &crate::vita::gxmctx::Block<'_>,
        stage: ProgramStage,
        w: &vitaslop_gxp_shader::MemWindow,
    ) -> u32 {
        match stage {
            ProgramStage::Vertex if w.buffer_index == Self::DEFAULT_UNIFORM_BUFFER_INDEX => {
                blk.uniform_binding(crate::vita::gxmctx::off::VERTEX_UNIFORM).buf
            }
            ProgramStage::Vertex => blk.vertex_uniform_buffer(w.buffer_index),
            ProgramStage::Fragment if w.buffer_index == Self::DEFAULT_UNIFORM_BUFFER_INDEX => {
                blk.uniform_binding(crate::vita::gxmctx::off::FRAGMENT_UNIFORM).buf
            }
            ProgramStage::Fragment => blk.fragment_uniform_buffer(w.buffer_index),
        }
    }

    pub(super) fn capture_mem_windows(
        &mut self,
        ctx: &GuestCtx,
        blk: &crate::vita::gxmctx::Block<'_>,
        vheader: u32,
        stage: ProgramStage,
        // Whether this draw's geometry - and with it every one of these windows - is read at
        // the end of the scene instead of here. See the snapshot below.
        deferring: bool,
        // The stage's SA uniform image for this draw (`sa_uniform_image`) - an input to a
        // per-draw window set's reach.
        sa_image: &[u8],
    ) -> Vec<(u32, Arc<[u8]>)> {
        let spec = self.mem_window_spec(ctx, vheader, stage);
        if spec.is_empty() {
            return Vec::new();
        }
        // Cheapest place to hang `VITASLOP_MEM_DUMP` off: it fires once, and this is a point
        // where a uniform block is DEMONSTRABLY live (its window is about to be snapshotted),
        // so the bytes on disk are the bytes the shader is about to read.
        Self::dump_guest_ranges(ctx, self.cur_frame);
        // >>> NO SCOPE HERE: THIS PHASE WOULD BE NESTING INSIDE ITSELF.
        //
        // Both callers of this function sit inside an already-open `DrawGxpCapture` scope (the
        // capture block opens one and calls this twice, once per stage), so a scope here bills
        // the SAME wall interval to the same phase three times over.
        //
        // MEASURED in the browser on a baseball title: `draw: gxp blob + SA bytes` read **1.03 ms/frame over
        // 1,739 entries against 459 draws** - 3.79 scopes a draw - and 26% of DRAW TOTAL, which
        // made it the largest named row in the table and the one 16d's ranking put first. A
        // phase that double-counts is worse than one that is missing: it reads as the thing to
        // fix. Nothing else calls this, so the outer scope already covers every byte of it.
        // >>> A PER-DRAW WINDOW SET IS READ AS FAR AS THIS DRAW CAN REACH, not to its declared
        // extent (`vitaslop_gxp_shader::module::MemWindow::per_draw`): the reach comes from the
        // draw's SA image, its windows' addresses and the scene's target extent - the same
        // inputs the renderer sizes the binding from, so the two agree. A depth-reconstruct
        // pass's four windows are declared 17 MiB each and reach about 4 MiB.
        let per_draw_addrs: Vec<u32> = if vitaslop_gxp_shader::module::mem_window_per_draw(&spec) {
            spec.iter()
                .map(|w| Self::bound_window_address(blk, stage, w).wrapping_add(w.base_offset))
                .collect()
        } else {
            Vec::new()
        };
        let extents = vitaslop_gxp_shader::module::mem_window_per_draw(&spec).then(|| {
            let addrs = &per_draw_addrs;
            let target = self.scene.as_ref().map_or((0, 0), |s| {
                let (cw, ch) = s.color.as_ref().map_or((0, 0), |c| (c.width, c.height));
                let (tw, th) = s.target_extent.unwrap_or((0, 0));
                (cw.max(tw), ch.max(th))
            });
            let blob = self.program_blob(ctx, vheader);
            let _r = crate::perf::scope(crate::perf::Phase::DrawWindowReach);
            let (e, hit) = vitaslop_gxp_shader::reach::per_draw_extents_memo(&blob, &spec, sa_image, addrs, target);
            if hit {
                crate::perf::note_hit(crate::perf::Phase::DrawWindowReachHit);
            }
            e
        });
        // A reach no storage binding can hold is not read at all: the renderer would refuse the
        // draw anyway, and reading it would copy up to gigabytes of guest memory first.
        if let Some(e) = extents.as_ref()
            && e.iter().map(|&b| u64::from(b)).sum::<u64>() > vitaslop_gxp_shader::module::MEM_WINDOW_STORAGE_MAX
        {
            static ONCE: std::sync::Once = std::sync::Once::new();
            ONCE.call_once(|| {
                tracing::warn!(
                    target: "vitaslop::gxm",
                    header = format_args!("{vheader:#010x}"),
                    "a program's per-draw memory windows reach {e:?} bytes this draw - past any storage binding, so its windows are withheld and the draw is DROPPED"
                );
            });
            return Vec::new();
        }
        let covers = extents
            .as_ref()
            .filter(|_| matches!(stage, ProgramStage::Fragment))
            .map(|e| self.raw_target_covers(&per_draw_addrs, e))
            .unwrap_or_default();
        let mut out = Vec::with_capacity(spec.len());
        for (i, w) in spec.iter().enumerate() {
            let len = extents.as_ref().map_or(w.bytes, |e| e[i]);
            let addr = Self::bound_window_address(blk, stage, w);
            // UNBOUND is a property of the BOUND address, tested before the offset is added -
            // adding one to zero produces a small non-zero pointer, which would turn "nothing is
            // bound" into "read guest address 124" and feed the draw exactly the fabricated
            // bytes this check exists to refuse.
            if addr == 0 || addr % 4 != 0 {
                // >>> THE WHOLE TABLE, NOT JUST THE SLOT THAT FAILED. Both paths that write
                // this table are INLINED into guest code (`StoreArgIndexed` for the direct
                // setter, `BindPrecomputedState` for the precomputed one), so neither passes
                // through a Rust handler a counter could sit in
                // [[vitaslop-inlined-host-calls-escape-both-watchpoints]]. Reading the table
                // back HERE is the one place the answer is visible: an all-zero table says
                // nothing ever bound anything, and a table with other slots filled says the
                // binding path works and this INDEX is the one that never arrives. Those are
                // different bugs and the slot alone cannot tell them apart.
                let table: Vec<u32> = (0..crate::vita::gxmctx::MAX_UNIFORM_BUFFERS as u32)
                    .map(|i| match stage {
                        ProgramStage::Vertex => blk.vertex_uniform_buffer(i),
                        ProgramStage::Fragment => blk.fragment_uniform_buffer(i),
                    })
                    .collect();
                let default_buf = blk
                    .uniform_binding(match stage {
                        ProgramStage::Vertex => crate::vita::gxmctx::off::VERTEX_UNIFORM,
                        ProgramStage::Fragment => crate::vita::gxmctx::off::FRAGMENT_UNIFORM,
                    })
                    .buf;
                Self::report_unbound_mem_window(
                    vheader,
                    stage,
                    w.buffer_index,
                    addr,
                    spec.len(),
                    &table,
                    default_buf,
                );
                return Vec::new();
            }
            // The pointer the DRIVER places, which for the default uniform buffer is NOT the
            // buffer's base - it names the first register the driver did not copy into the SA
            // file. See `MemWindow::base_offset` for the corpus closure that establishes it and
            // for the red sky it explains. Zero for every other buffer, so this is inert there.
            let at = addr.wrapping_add(w.base_offset);
            // >>> A WINDOW ON A RAW TARGET THIS FRAME RENDERED: read only what the GPU fill
            // leaves, at the draw even under deferral - see `get_or_read_covered_window`, and
            // the resolve, which knows these by address and re-reads only their blank
            // uncovered bytes.
            if let Some(cover) = covers.get(i).filter(|c| !c.is_empty()) {
                let bytes = match self.texture_snapshots.try_lock() {
                    Err(std::sync::TryLockError::WouldBlock) => {
                        crate::perf::note_hit(crate::perf::Phase::SnapsLockWait);
                        TextureSnapshots::read_covered_window(ctx, at, len as usize, cover)
                    }
                    held => {
                        drop(held);
                        snaps!(self).get_or_read_covered_window(ctx, at, len as usize, cover)
                    }
                };
                out.push((at, bytes));
                continue;
            }
            // >>> UNDER DEFERRAL THIS READ IS DEAD, AND IT WAS ALSO READING THE WRONG BYTES.
            //
            // `resolve_deferred_geometry` re-snapshots EVERY window of every draw at the
            // guest's own GPU wait, because a title may fill a uniform buffer after the draw
            // that names it - which this title does: its crowd skinning window reads ALL ZEROES
            // at the draw call and is filled by a job thread afterwards. So the draw-time
            // snapshot is overwritten before anything reads it, and the only thing the capture
            // needs from it is the window's ADDRESS and LENGTH.
            //
            // MEASURED in the desktop browser on this title's heaviest gameplay frame:
            // `capture_mem_windows` is **3.3% of the busy worker thread**, the largest single
            // item inside `sceGxmDraw`, for ~1,000 calls a frame that answer a question asked
            // again a moment later. The placeholder carries the length and costs an `Arc`
            // clone.
            //
            // The two reports go with the read: reporting the PRE-FILL bytes is what made this
            // one say a live window was all zero. They fire on the non-deferred arm, which is
            // the one whose bytes are the draw's own.
            let bytes = if deferring && defer_window_bytes() {
                snaps!(self).deferred_window_placeholder(len as usize)
            } else {
                // >>> NOT BEHIND THE RESOLVER. The cache only keeps a window's IDENTITY stable
                // for the renderer's push memo; when the resolver holds the snapshot lock for a
                // chunk of its flip read, reading the few KB straight out of guest memory costs
                // less than the wait - MEASURED on the phone (a baseball title's pitches, 048) 1.36 ms/f of the
                // render thread's frame queued here over ~4 takes. The bytes are the same either
                // way (both read guest memory now). `VITASLOP_WINDOW_WAIT=1` is the arm back.
                let bytes = match self.texture_snapshots.try_lock() {
                    Err(std::sync::TryLockError::WouldBlock) if !window_wait() => {
                        crate::perf::note_hit(crate::perf::Phase::SnapsLockWait);
                        ctx.read_bytes_arc(at, len as usize)
                    }
                    held => {
                        drop(held);
                        snaps!(self).get_or_read_window(ctx, at, len as usize)
                    }
                };
                // Tested ONCE per (program, buffer) - see `report_mem_window`. A set on the state,
                // not a global lock: this line runs for every window of every draw.
                if self
                    .mem_windows_zero_tested
                    .insert((vheader, w.buffer_index))
                {
                    Self::report_mem_window(vheader, w.buffer_index, at, &bytes);
                }
                Self::peek_past_mem_window(ctx, vheader, w.buffer_index, at);
                bytes
            };
            out.push((at, bytes));
        }
        out
    }

    /// Per window of a PER-DRAW set (`addrs`, `extents`), the bytes - relative to the window - a
    /// raw 64-bit target this frame rendered COVERS: the bytes the renderer's window fill copies
    /// the target's texels over (`fill_windows_from_raw_targets`), so reading them is wasted.
    ///
    /// Computed the way the renderer computes them: over the SPAN it places the windows in
    /// (`per_draw_window_spans`), from every raw target this frame's ended scenes rendered except
    /// the one this scene renders into (the fill refuses a pass's own attachment), with the
    /// fill's own plan. Empty when nothing is covered.
    fn raw_target_covers(&self, addrs: &[u32], extents: &[u32]) -> Vec<Vec<(u32, u32)>> {
        if self.frame_raw_targets.is_empty() || !vitaslop_platform::gpu::mem_window_skip_covered() {
            return Vec::new();
        }
        let own = self.scene.as_ref().and_then(|s| s.color.as_ref()).map(|c| c.data_addr);
        let targets: Vec<vitaslop_platform::gpu::RawFillTarget> =
            self.frame_raw_targets.iter().copied().filter(|t| Some(t.0) != own).collect();
        if targets.is_empty() {
            return Vec::new();
        }
        let (spans, span_of) = vitaslop_platform::gpu::per_draw_window_spans(addrs, extents);
        let span_cover: Vec<Vec<(u32, u32)>> = spans
            .iter()
            .map(|&(s, e)| {
                let mut c = Vec::new();
                vitaslop_platform::gpu::raw_window_fill_cover(s as u32, (e - s) as u32, &targets, &mut c);
                c
            })
            .collect();
        addrs
            .iter()
            .enumerate()
            .map(|(i, &a)| {
                let d = u64::from(a) - spans[span_of[i]].0;
                let len = u64::from(extents.get(i).copied().unwrap_or(0));
                span_cover[span_of[i]]
                    .iter()
                    .filter_map(|&(cs, ce)| {
                        let (cs, ce) = (u64::from(cs).max(d), u64::from(ce).min(d + len));
                        (ce > cs).then(|| ((cs - d) as u32, (ce - d) as u32))
                    })
                    .collect()
            })
            .collect()
    }

    /// The flip: the renderer starts its set of raw targets rendered this frame afresh, and so
    /// does [`Self::raw_target_covers`].
    pub(super) fn end_frame_raw_targets(&mut self) {
        self.frame_raw_targets.clear();
    }

    /// The raw `SceGxmProgram` container bytes at `header`, read out of guest memory once and
    /// shared by every draw that binds it afterwards.
    ///
    /// The recompiler path needs the whole container per draw, and a container is a few
    /// kilobytes. A race frame submits 400+ draws over a couple of dozen distinct programs, so
    /// reading it per draw copies the same bytes hundreds of times a frame - the single largest
    /// avoidable item on the capture side of a recompiled frame. The bytes are IMMUTABLE while
    /// the program is registered (GXM's shader patcher owns them; the guest does not rewrite a
    /// live program), and the cache is dropped by [`invalidate_program_reflection`] at exactly
    /// the moment a header address can come to mean a different program - the same discipline,
    /// and the same clear, that `program_reflection` already relies on.
    pub(crate) fn program_blob(&mut self, ctx: &GuestCtx, header: u32) -> std::sync::Arc<[u8]> {
        if header == 0 {
            return crate::capture::no_program();
        }
        if let Some(b) = self.program_memos.blob(header) {
            return b;
        }
        // The blob size is the container total-length field at header+0x08, the same idiom the
        // GXP-bin dump uses.
        let sz = ctx.read_u32(header.wrapping_add(0x08)).clamp(0x40, 0x40000) as usize;
        let blob: std::sync::Arc<[u8]> = std::sync::Arc::from(ctx.read_bytes(header, sz));
        self.program_memos.set_blob(header, blob.clone());
        blob
    }

    /// Say, once, that a title creates its fragment programs with a NULL `vertexProgram` - so
    /// this call names no PAIR and nothing can be prepared from it.
    ///
    /// # This is the measurement that decides whether shader preparation is possible at all
    /// GXM's signature carries `const SceGxmProgram *vertexProgram` so the patcher can patch the
    /// varying linkage, which makes the pair knowable while a loading screen is still up. Whether
    /// a given title actually passes it is a fact about that title, and it is not visible
    /// anywhere else: a NULL simply means the preparation silently never happens and the compile
    /// lands in a gameplay frame instead. MEASURED on a retail title: **all 17 of its distinct
    /// fragment programs are created with vertexProgram = NULL**, so its 160 pairs are first
    /// known at the DRAW that binds them.
    pub(super) fn report_null_vertex_program(vertex_header: u32, fragment_header: u32) {
        use std::sync::atomic::{AtomicBool, Ordering};
        static SEEN: AtomicBool = AtomicBool::new(false);
        if vertex_header != 0 || fragment_header == 0 || SEEN.swap(true, Ordering::Relaxed) {
            return;
        }
        tracing::info!(
            target: "vitaslop::status",
            "gxm precompile: this title creates fragment programs with a NULL vertexProgram (first \
             at frag {fragment_header:#x}), so sceGxmShaderPatcherCreateFragmentProgram names no \
             shader PAIR and none can be compiled ahead of the draw that binds it. The WGSL \
             compile stays in the frame - see the hiccup log's split."
        );
    }
}
