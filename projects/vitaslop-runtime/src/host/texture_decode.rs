use super::*;

/// Report - once per distinct base format, unconditionally - that a bound texture's format
/// has no known block geometry, so the sampler unit it was bound to is left EMPTY.
///
/// This has to be loud rather than a `debug!` behind a filter nobody has on. Downstream the
/// only evidence is a recompiled shader falling back with "sampler unit N wants Two but the
/// bound units are [...]", and reading that backwards to a texture format costs a session.
/// Deduped by format because a title binds the same format on hundreds of draws a frame.
/// Report - once per (base format, swizzle) - a texture resolved from its CONTROL WORDS rather
/// than from the exact format recorded at `sceGxmTextureInit*`.
///
/// This is the path a texture takes once a title copies the struct it lives in, and it is
/// lossier than the recorded one: the base format has to be reassembled from a 5-bit field plus
/// an extension bit, so every format above 0x7f (all of BC/PVRTC, and `U2F10F10F10`) decodes as
/// a different, smaller format if that bit is not carried. A UI atlas that is really BC3 read as
/// a 16-bit uncompressed format is not a dropped texture - it renders, subtly wrong.
/// Say - once per distinct address - that the guest PRESENTED a buffer that no GXM scene in
/// this capture ever rendered into.
///
/// # Why this is worth a line
/// The renderer takes "the display" to be the LAST scene's colour address
/// (`gpu.rs`, `let display = last.target.map(...)`) and never consults what the guest actually
/// handed `sceDisplaySetFrameBuf`. That is right for a title that composites through GXM and
/// presents what it composited, which is every title measured so far. It is NOT right for a
/// buffer GXM never wrote - the obvious one being a full-screen MOVIE, whose picture the video
/// decoder writes straight into guest memory for the display controller to scan out.
///
/// When that happens the frame is composed from GXM passes that have nothing in them and the
/// picture is BLACK, with nothing anywhere saying the guest had a buffer it wanted shown. This
/// line is that statement. It does not fix it: honouring a presented buffer is a change to how
/// every frame is composed, and what it needs first is to know which titles do this and with
/// what.
pub(super) fn report_present_no_scene_rendered(buffer_addr: u32, cap: &crate::capture::Capture, frame: u64) {
    use std::collections::HashSet;
    use std::sync::Mutex;
    // >>> ONLY THE SCENES THE RENDERER WILL ACTUALLY BE HANDED. A scene whose geometry is still
    // pending (`deferred_id != 0`) is NOT taken at this boundary - the frontend partitions it
    // out and it waits for the next one - so it is not part of the picture this present makes,
    // and `gpu.rs` never sees it. Reading `cap.scenes` whole made one frame of a baseball title
    // report "scene 9 of 41" where the renderer was composing eleven, and the pass it named as
    // the composite was a 256x256 target that frame never displayed. A diagnostic about which
    // pass reaches the screen has to reason over the same list the renderer does.
    let scenes: Vec<&crate::capture::Scene> =
        cap.scenes.iter().filter(|s| s.deferred_id == 0).collect();
    if buffer_addr == 0 || scenes.is_empty() {
        return;
    }
    // WHICH scene it is, is the question that matters: the renderer takes the display to be the
    // LAST scene's colour surface, so a present that names an EARLIER one is a frame composed
    // from the wrong pass - the picture on screen is whatever that last pass drew, and the pass
    // the guest actually wanted shown is an offscreen target nothing reads.
    if let Some(i) = scenes
        .iter()
        .position(|s| s.color.as_ref().is_some_and(|c| c.data_addr == buffer_addr))
    {
        // >>> THE TEST IS THE ADDRESS, NOT THE INDEX, and on the index it cried wolf four times
        // out of five. What the renderer composes from is the last scene's colour ADDRESS, so a
        // present naming an EARLIER scene that renders into THAT SAME ADDRESS reaches the
        // screen exactly as the guest asked. This title ends most frames with a 0-draw scene on
        // the buffer it just filled - a perfectly ordinary shape - and the index test called
        // every one of those a wrong-pass frame. A warning about picture correctness that is
        // mostly false teaches its reader to skip the one firing that is true, and there WAS
        // one true firing hiding in those five [[vitaslop-a-warning-means-we-owe-a-fix]].
        // Exactly what `gpu.rs` computes - `scenes.last().target.map(data_addr)` - including
        // the None when the final scene has no colour target at all. Anything cleverer here
        // would be a SECOND opinion about the display buffer, and the whole value of this
        // warning is that it reports the renderer's actual choice rather than a better one.
        let composed = scenes
            .last()
            .and_then(|s| s.color.as_ref())
            .map(|c| c.data_addr);
        if composed != Some(buffer_addr) {
            report_present_is_not_the_last_scene(buffer_addr, i, &scenes, frame);
        }
        return;
    }
    static SEEN: Mutex<Option<HashSet<u32>>> = Mutex::new(None);
    let mut g = SEEN.lock().unwrap_or_else(|e| e.into_inner());
    let seen = g.get_or_insert_with(HashSet::new);
    // Bounded: a title that presents from a ring of buffers GXM never wrote would otherwise
    // print one line per buffer forever.
    if seen.len() >= 8 || !seen.insert(buffer_addr) {
        return;
    }
    let targets: Vec<String> = scenes
        .iter()
        .map(|s| match s.color.as_ref() {
            Some(c) => format!("{:#x}({}x{})", c.data_addr, c.width, c.height),
            None => "none".to_string(),
        })
        .collect();
    // >>> AND SAY WHEN THE BUFFER IS MERELY DEFERRED, because the two cases want opposite
    // reactions. A scene whose geometry is still pending IS this frame's, it simply waits for
    // the next boundary - so "nothing renders it" would be wrong and would send a reader
    // hunting a missing pass that is sitting in the queue. The list above is the scenes the
    // renderer is handed NOW; this line is the rest of the truth.
    let deferred = cap.scenes.iter().any(|s| {
        s.deferred_id != 0 && s.color.as_ref().is_some_and(|c| c.data_addr == buffer_addr)
    });
    if deferred {
        tracing::warn!(
            target: "vitaslop::display",
            "display: the guest PRESENTED {buffer_addr:#x}, which THIS frame renders into but \
             whose geometry is still PENDING, so it is not in the scene list the renderer is \
             handed now - it is taken at the next boundary. The frame on screen is composed \
             from [{}] without it.",
            targets.join(", ")
        );
        return;
    }
    tracing::warn!(
        target: "vitaslop::display",
        "display: the guest PRESENTED {buffer_addr:#x}, which no GXM scene of this frame \
         rendered into - the frame's colour surfaces are [{}]. The renderer composes the \
         picture from the scenes, so whatever is in that buffer (a movie's decoded picture is \
         the usual one) does not reach the screen.",
        targets.join(", ")
    );
}

/// Say - once per distinct (scene index, scene count) shape - that the guest presented a buffer
/// that IS one of this frame's colour surfaces but is NOT the one the renderer composes from.
///
/// `gpu.rs` takes `display` to be the LAST scene's colour address, which is right for a title
/// that ends its frame on the pass it wants shown. A title that composites into an offscreen
/// target and then submits further passes (a UI overlay into a second surface, say) breaks that
/// assumption, and the symptom is the whole world missing behind a correct HUD - or, when the
/// later pass is a clear, a flat-coloured screen with every draw of the frame landing somewhere
/// nothing reads.
///
/// The caller tests the ADDRESS, not the scene index - see there. This function still prints
/// the index because it is what locates the pass in the list below it.
pub(super) fn report_present_is_not_the_last_scene(
    buffer_addr: u32,
    index: usize,
    scenes: &[&crate::capture::Scene],
    frame: u64,
) {
    use std::collections::HashSet;
    use std::sync::Mutex;
    static SEEN: Mutex<Option<HashSet<(usize, usize)>>> = Mutex::new(None);
    let mut g = SEEN.lock().unwrap_or_else(|e| e.into_inner());
    if !g
        .get_or_insert_with(HashSet::new)
        .insert((index, scenes.len()))
    {
        return;
    }
    let targets: Vec<String> = scenes
        .iter()
        .map(|s| match s.color.as_ref() {
            Some(c) => format!(
                "{:#x}({}x{}, {} draws)",
                c.data_addr,
                c.width,
                c.height,
                s.draws.len()
            ),
            None => format!("none({} draws)", s.draws.len()),
        })
        .collect();
    eprintln!(
        "display: at frame {frame} the guest PRESENTED {buffer_addr:#x}, which is scene {index} \
         of {}, but the renderer composes from the LAST scene colour surface - {} - so what reaches \
         the screen is that pass and not the one the guest asked to show. Passes: [{}]",
        scenes.len(),
        match scenes.last().and_then(|s| s.color.as_ref()) {
            Some(c) => format!("{:#x}", c.data_addr),
            None => "no colour surface at all".to_string(),
        },
        targets.join(", ")
    );
}

pub(super) fn report_texture_resolved_from_control_words(unit: u32, base_format: u32, swizzle: u32) {
    use std::collections::HashSet;
    use std::sync::Mutex;
    // Per decode of a by-value texture - the ordinary case - so a key this thread has seen
    // returns before the global lock.
    thread_local! {
        static DONE: std::cell::RefCell<crate::fasthash::FxHashSet<(u32, u32)>> = std::cell::RefCell::new(Default::default());
    }
    if !DONE.with(|d| d.borrow_mut().insert((base_format, swizzle))) {
        return;
    }
    static SEEN: Mutex<Option<HashSet<(u32, u32)>>> = Mutex::new(None);
    let mut g = SEEN.lock().unwrap_or_else(|e| e.into_inner());
    if !g
        .get_or_insert_with(HashSet::new)
        .insert((base_format, swizzle))
    {
        return;
    }
    tracing::info!(
        target: "vitaslop::status",
        "gxm texture: unit {unit} bound a texture with NO recorded format - resolving it from \
         its control words alone as base format {base_format:#04x}, swizzle {swizzle}. This is a \
         copy of a texture initialised elsewhere."
    );
}

pub(super) fn report_unsized_texture_format(
    unit: u32,
    base_format: u32,
    tex_type: u32,
    width: u32,
    height: u32,
) {
    use std::collections::HashSet;
    use std::sync::Mutex;
    static SEEN: Mutex<Option<HashSet<(u32, u32)>>> = Mutex::new(None);
    let mut g = SEEN.lock().unwrap_or_else(|e| e.into_inner());
    if !g
        .get_or_insert_with(HashSet::new)
        .insert((unit, base_format))
    {
        return;
    }
    eprintln!(
        "gxm texture: unit {unit} base format {base_format:#04x} (type {tex_type}, \
         {width}x{height}) has no known block size - EVERY unit bound to this format is left \
         unbound, and any shader sampling it falls back to fixed-function"
    );
}

/// Report - once per base format - that the guest's mip chain could not be read, so the
/// snapshot fell back to level 0 alone.
///
/// This is expected for a texture whose data pointer is a render target's colour surface: it
/// has no mips in memory whatever its control word says, and the bytes past level 0 are
/// somebody else's or nothing at all. It is NOT expected for an ordinary asset, and there the
/// line is the difference between "the guest has no mips here" and "we computed the chain size
/// wrong", which need opposite fixes.
pub(super) fn report_mip_chain_unreadable(
    unit: u32,
    base_format: u32,
    data_addr: u32,
    width: u32,
    height: u32,
    levels: u32,
) {
    use std::collections::HashSet;
    use std::sync::Mutex;
    static SEEN: Mutex<Option<HashSet<u32>>> = Mutex::new(None);
    let mut g = SEEN.lock().unwrap_or_else(|e| e.into_inner());
    if !g.get_or_insert_with(HashSet::new).insert(base_format) {
        return;
    }
    tracing::warn!(
        target: "vitaslop::gxm",
        "gxm texture: unit {unit} ({width}x{height}, base format {base_format:#04x} at \
         {data_addr:#010x}) declares {levels} mip levels but its memory does not extend that \
         far - falling back to level 0 alone for every texture of this format"
    );
}

/// How many sampler-unit bindings each drop cause has cost, over the whole run.
///
/// The per-cause REPORTS are deduped so they stay readable, which means they say nothing
/// about scale - and scale is what decides which cause to fix first. One unsized format on
/// four units and a zero handle on three read identically in the log while costing wildly
/// different numbers of draws. Counted here, printed by [`report_texture_drops`].
/// Atomics, not a locked array: this is counted per dropped BINDING, millions on some titles.
pub(super) static TEXTURE_DROPS: [std::sync::atomic::AtomicU64; 6] =
    [const { std::sync::atomic::AtomicU64::new(0) }; 6];

/// Drop causes, in [`TEXTURE_DROPS`] order.
pub(super) const DROP_CAUSES: [&str; 6] = [
    "zero control words (null data pointer - bound to a 1x1 zero texel, not dropped)",
    "unsized format",
    "pixels not readable",
    "undecodable base format",
    "a PALETTED texture with no readable colour table",
    "a PALETTED texture whose index data does not cover its declared levels",
];

/// How many times a `sceGxmTextureInit*` / `SetFormat` recorded a format for an address. A zero
/// here alongside a large drop count means the title never goes through the init API at all, and
/// builds its control words itself - which is a completely different investigation from "it
/// inits somewhere we are not looking".
pub(super) static TEXTURE_INITS: std::sync::atomic::AtomicU64 = std::sync::atomic::AtomicU64::new(0);

/// Distinct `SceGxmTexture*` handles that read as all-zero control words at draw time. See
/// [`report_zero_texture_handle`].
pub(super) static ZERO_TEXTURE_ADDRS: std::sync::Mutex<Option<std::collections::HashSet<u32>>> =
    std::sync::Mutex::new(None);

/// Vblank spins parked, and the guest instructions that would otherwise have been spent
/// spinning. Reported once per run rather than per park - a race parks once or twice a frame
/// and a line each would bury everything else.
pub(super) static VBLANK_SPIN_PARKS: std::sync::atomic::AtomicU64 = std::sync::atomic::AtomicU64::new(0);

/// How many vblank spins have been parked so far - see [`report_vblank_spin_parks`]. The
/// browser reports it on its running status line rather than at run end, because a phone
/// run is watched while it happens and may never reach an orderly end.
pub fn vblank_spin_parks() -> u64 {
    VBLANK_SPIN_PARKS.load(std::sync::atomic::Ordering::Relaxed)
}

/// One line saying the spin guard fired, and how often. Silent when it never did - which is
/// itself the reading for a title that does not spin on the vblank counter.
///
/// It is NOT silent when it does fire, because this is a behaviour change: the guest's own
/// wait loop is being cut short, and a run where that happens should say so rather than
/// leave the next person to infer it from a frame time.
/// One line per live guest memory space: how full it is, and - the reading that matters -
/// how many allocations it served against how many frees it saw. See `mspace::Mspace`.
///
/// On `VitaState` rather than a free function because the spaces are per-run state, not a
/// process-wide counter.
impl VitaState {
    pub fn mspace_report(&self) -> Vec<String> {
        self.mspaces.report()
    }
}

pub fn report_vblank_spin_parks() {
    let n = VBLANK_SPIN_PARKS.load(std::sync::atomic::Ordering::Relaxed);
    if n == 0 {
        return;
    }
    tracing::info!(
        target: "vitaslop::perf",
        "vblank spin: {n} wait loops PARKED on the next vblank after {} reads of the counter inside one resume. Each one is a loop that could not have observed a change (the mirror is frozen while guest code runs) and would otherwise have run until its own fuel dragged the clock to the vblank edge",
        crate::vita::mirror::SPIN_BUDGET,
    );
}

pub(super) fn note_texture_drop(cause: usize) {
    TEXTURE_DROPS[cause].fetch_add(1, std::sync::atomic::Ordering::Relaxed);
}

/// Print how many sampler-unit bindings were dropped, by cause, over the run. Call once at
/// the end of a run; silent when nothing was dropped.
pub fn report_texture_drops() {
    let counts: [u64; 6] =
        std::array::from_fn(|i| TEXTURE_DROPS[i].load(std::sync::atomic::Ordering::Relaxed));
    if counts.iter().all(|&c| c == 0) {
        return;
    }
    let mut ranked: Vec<_> = DROP_CAUSES
        .iter()
        .zip(counts)
        .filter(|(_, c)| *c > 0)
        .collect();
    ranked.sort_by_key(|r| std::cmp::Reverse(r.1));
    eprintln!("gxm texture: sampler-unit bindings DROPPED over this run, by cause:");
    for (cause, n) in ranked {
        eprintln!("gxm texture:   {n} - {cause}");
    }
    eprintln!(
        "gxm texture:   {} texture format(s) were recorded through the init API over the run",
        TEXTURE_INITS.load(std::sync::atomic::Ordering::Relaxed)
    );
    if let Ok(g) = ZERO_TEXTURE_ADDRS.lock()
        && let Some(addrs) = g.as_ref().filter(|a| !a.is_empty())
    {
        let mut list: Vec<u32> = addrs.iter().copied().collect();
        list.sort_unstable();
        let shown: Vec<String> = list.iter().take(12).map(|a| format!("{a:#x}")).collect();
        eprintln!(
            "gxm texture:   the zero control words come from {} DISTINCT handle(s): {}{}",
            list.len(),
            shown.join(" "),
            if list.len() > shown.len() { " ..." } else { "" }
        );
    }
}

/// Report - once per unit - that a bound `SceGxmTexture` handle reads as all zeros at draw
/// time, naming which binding path put it there.
///
/// `from_precomputed` is the whole point of the message: a zero handle bound through
/// `sceGxmSetFragmentTexture` means the guest really bound an uninitialised texture, while
/// one that arrived with a precomputed fragment state means the state record is stale or
/// keyed wrong on OUR side. Those need opposite investigations and look identical downstream.
pub(super) fn report_zero_texture_handle(
    ctx: &GuestCtx,
    binding: &TextureBinding,
    exact_format: Option<u32>,
    nearby: Option<(i64, u32)>,
) {
    let (unit, addr, from_precomputed) = (binding.unit, binding.addr, binding.from_precomputed);
    use std::collections::HashSet;
    use std::sync::Mutex;
    // A (unit, address) THIS thread has already handled has nothing left to add - both sets
    // below already hold it - so it returns before either global lock. Per dropped binding.
    thread_local! {
        static DONE: std::cell::RefCell<crate::fasthash::FxHashSet<(u32, u32)>> = std::cell::RefCell::new(Default::default());
    }
    if !DONE.with(|d| d.borrow_mut().insert((unit, addr))) {
        return;
    }
    // Every distinct zero HANDLE is counted, so the end-of-run summary can say how many
    // addresses are involved. "2.3 million dropped bindings" reads like a catastrophe and is
    // consistent with three addresses rebound every frame; the two need different responses.
    if let Ok(mut a) = ZERO_TEXTURE_ADDRS.lock() {
        a.get_or_insert_with(HashSet::new).insert(addr);
    }
    // Once per (unit, ADDRESS), not per unit: a title binds a different uninitialised handle
    // to the same unit for each of its post-process steps, and reporting only the first tells
    // you a chain is broken without telling you how many distinct objects are involved - which
    // is the difference between one bad pointer and a whole class of them.
    static SEEN: Mutex<Option<HashSet<(u32, u32)>>> = Mutex::new(None);
    let mut g = SEEN.lock().unwrap_or_else(|e| e.into_inner());
    if !g.get_or_insert_with(HashSet::new).insert((unit, addr)) {
        return;
    }
    // The memory AROUND the handle, so the two ways this happens can be told apart: memory
    // that is zero for a long stretch means the struct was never written, while live data
    // either side of a zero window means the handle address is off by an offset.
    let window: Vec<String> = (0..32)
        .map(|i| {
            format!(
                "{:08x}",
                ctx.read_u32(addr.wrapping_sub(48).wrapping_add(i * 4))
            )
        })
        .collect();
    eprintln!(
        "gxm texture: memory around handle {addr:#x} (from -48): {}",
        window.join(" ")
    );
    let via = if from_precomputed {
        "a PRECOMPUTED fragment state"
    } else {
        "sceGxmSetFragmentTexture"
    };
    // Whether a `sceGxmTextureInit*`/`SetFormat` was ever seen for THIS address splits the
    // cause in two: recorded means the guest initialised the texture somewhere else and the
    // bytes here are a copy that never got them (a by-value problem on our side); not
    // recorded means the guest never went through an init we implement at all.
    let init = match (exact_format, nearby) {
        (Some(f), _) => format!("a format ({f:#010x}) WAS recorded for this address"),
        // A texture initialised a few bytes away is the whole answer: the guest DID build a
        // texture and this binding points at the wrong place in it (an interior member, or a
        // copy that missed the init). Naming the delta turns the fix into arithmetic.
        (None, Some((d, f))) => format!(
            "NO format was recorded here, but one ({f:#010x}) WAS recorded {d} bytes away - the \
             binding is off by that much, or this is an uninitialised COPY of that texture"
        ),
        (None, None) => {
            "NO format was ever recorded for this address, nor within 4 KiB of it".to_string()
        }
    };
    eprintln!(
        "gxm texture: handle {addr:#x} read as all-zero control words AT BIND TIME (the binding captures them then, so this is what the guest handed to GXM, not what the memory happens to hold now)"
    );
    eprintln!(
        "gxm texture: unit {unit} handle {addr:#x} (bound via {via}) reads as ALL ZERO control \
         words, and {init} - its data pointer is null, so the unit is bound to a 1x1 ZERO \
         texel (what the hardware would read at address 0)"
    );
}

/// Report - once per (unit, format) - that a bound texture's pixel bytes could not be read
/// from guest memory, so the unit is left EMPTY.
///
/// The other half of [`report_unsized_texture_format`]. A unit can go missing because the
/// format has no size OR because the bytes behind a perfectly-understood format are not
/// readable, and downstream both look identical: a recompiled shader falling back with
/// "sampler unit N wants Two but the bound units are [...]". Only the address and length
/// distinguish them, and only here are they still in hand.
pub(super) fn report_unreadable_texture(unit: u32, base_format: u32, addr: u32, len: usize, w: u32, h: u32) {
    use std::collections::HashSet;
    use std::sync::Mutex;
    static SEEN: Mutex<Option<HashSet<(u32, u32)>>> = Mutex::new(None);
    let mut g = SEEN.lock().unwrap_or_else(|e| e.into_inner());
    if !g
        .get_or_insert_with(HashSet::new)
        .insert((unit, base_format))
    {
        return;
    }
    eprintln!(
        "gxm texture: unit {unit} format {base_format:#04x} {w}x{h} at {addr:#x} ({len} bytes) is \
         NOT READABLE from guest memory - the unit is left unbound, and any shader sampling it \
         falls back to fixed-function"
    );
}

/// Decode a bound `SceGxmTexture` (16 bytes, 4 control words) from guest memory
/// and snapshot its pixel bytes. Returns `None` for a null/unreadable handle or a
/// format whose byte size we do not know yet. The layout is the public GXM texture
/// control-word format (vitasdk `gxm.h` `struct SceGxmTexture`): word 1 holds
/// width/height (stored as size-1) and the base format, word 2 the data address.
/// The single 1x1 zero-RGBA buffer substituted for every null texture handle.
///
/// Shared rather than allocated per call so its ADDRESS is stable: an identity-keyed
/// consumer cache (`crate::render::tex_key`) would otherwise see a different texture every
/// draw. The bytes are constant, so one buffer is as correct as a million.
pub(super) fn zero_texel() -> Arc<[u8]> {
    static ZERO: std::sync::OnceLock<Arc<[u8]>> = std::sync::OnceLock::new();
    ZERO.get_or_init(|| Arc::from([0u8, 0, 0, 0].as_slice()))
        .clone()
}

/// The sampler state a texture's CONTROL WORD 0 carries, as
/// `(u addr mode, v addr mode, lod bias, min filter, mag filter, gamma)`.
///
/// Field positions per `vita::gxm::texword0` (vitasdk `gxm.h`). The one type-dependent case is
/// `min_filter`: for a `LINEAR_STRIDED` texture those bits are part of the stride, and the
/// header states that such a texture uses its MAG filter for minification too - so reading bits
/// 10..11 there would sample by a couple of stride bits instead of a filter mode.
pub(super) fn word0_sampler_state(w0: u32, tex_type: u32) -> (u32, u32, u32, u32, u32, u32, u32) {
    // `SCE_GXM_TEXTURE_LINEAR_STRIDED` (0xC000_0000) as the 3-bit selector `(w1 >> 29) & 7` -
    // see `vita::gxm::TYPE_LINEAR_STRIDED`. This was 1, which is no texture type at all, so a
    // strided texture read two of its stride bits as its minification filter.
    const TYPE_LINEAR_STRIDED: u32 = 6;
    let mag = (w0 >> 12) & 0x3;
    let min = if tex_type == TYPE_LINEAR_STRIDED {
        mag
    } else {
        (w0 >> 10) & 0x3
    };
    (
        (w0 >> 6) & 0x7,   // uaddr_mode
        (w0 >> 3) & 0x7,   // vaddr_mode
        (w0 >> 21) & 0x3f, // lod_bias
        min,
        mag,
        w0 & 0x1800_0000, // gamma_mode, in the enum's own already-shifted encoding
        // mip_filter (`vita::gxm::texword0::MIP_FILTER`) - a stride bit on a strided texture,
        // which has one level and nothing to filter between.
        if tex_type == TYPE_LINEAR_STRIDED {
            0
        } else {
            (w0 >> 9) & 0x1
        },
    )
}

pub(super) fn decode_texture(
    ctx: &GuestCtx,
    cache: &mut TextureSnapshots,
    binding: &TextureBinding,
    exact_format: Option<u32>,
    nearby: Option<(i64, u32)>,
    byte_stride: u32,
) -> Option<crate::capture::BoundTexture> {
    let _owner = vitaslop_platform::heap::owner_scope(vitaslop_platform::heap::Owner::TextureDecode);
    let (unit, addr) = (binding.unit, binding.addr);
    if addr == 0 {
        return None;
    }
    // The control words come from the BINDING, captured when the guest handed the texture to
    // GXM, not from guest memory now. See `TextureBinding`.
    //
    // All four control words zero is not a texture - it is a handle that was never
    // initialised, or one whose memory is not readable (an unmapped read yields zero). Either
    // way the unit ends up unbound, and it is a completely different bug from a format we
    // cannot size, so it gets its own report rather than being folded into "not readable".
    if binding.is_null() {
        note_texture_drop(0);
        report_zero_texture_handle(ctx, binding, exact_format, nearby);
        // Bind a 1x1 ZERO texel rather than leaving the unit empty.
        //
        // Zero is not a neutral choice, it is the FAITHFUL one: a zeroed handle has a null
        // data pointer, so the hardware samples address 0 - and the texel it reads there is
        // zeros in every format. Dropping the unit instead costs the whole draw, because the
        // recompiled shader then cannot build its bind group at all and falls back to a
        // fixed-function approximation.
        //
        // It was briefly white, on the reasoning that white is the identity for a modulate.
        // That was wrong twice over: it is not what the hardware reads, and a title's
        // post-process chain sampled one of these and came out solid white, which then
        // composited over the entire race. A substitute has to be defensible as a VALUE, not
        // as a guess about how the shader will use it.
        return Some(crate::capture::BoundTexture {
            unit,
            base_format: 0x0c, // U8U8U8U8
            swizzle: 0,        // SWIZZLE4_ABGR - the identity permutation
            tex_type: 3,       // LINEAR
            width: 1,
            height: 1,
            stride: 4,
            faces: 1,
            face_bytes: 4,
            levels: 1,
            data_addr: 0,
            // ONE buffer for every null handle of the run, not one per draw. Consumers key
            // their caches on this buffer's IDENTITY (see `crate::render::tex_key`), so a
            // fresh allocation per draw would miss every time and churn a 512-entry decode
            // cache on a unit whose contents are four constant zero bytes.
            pixels: zero_texel(),
            // ONE identity for the one shared zero buffer - see `next_pixels_id`.
            pixels_id: 0,
            // A zeroed handle's control words are zero, so every one of these is the GXM
            // default by construction - which is what the hardware would sample with too.
            u_addr_mode: 0,
            v_addr_mode: 0,
            lod_bias: 0,
            lod_min: 0,
            min_filter: 0,
            mag_filter: 0,
            mip_filter: 0,
            gamma: 0,
            yuv_profile: 0,
        });
    }

    // >>> THE WHOLE DECODE, INCLUDING THE PIXELS, IS DONE ONCE PER BINDING - and then only
    // RE-PROVEN, not redone, on the first use of each later scene (`decoded_validated`).
    //
    // [`TextureSnapshots::snapshot_sets`] already caches a whole draw's list by the SET of
    // bindings that produced it, and its doc comment says the hit rate is "essentially the
    // draw count". MEASURED on a racing title's on-track frame in the browser: it is
    // **53%**. A race
    // scene draws a few hundred objects out of a shared pool of textures, so the individual
    // BINDINGS repeat constantly while the combinations do not - and every set miss then paid
    // a full `get_or_read` per bound unit. That was 3,631 `get_or_read` calls per presented
    // frame, 86% of the fragment miss path and the largest single item in the capture.
    //
    // Memoising per binding is exactly equivalent for the same reason the set cache is:
    // `get_or_read` compares a texture at most once per SCENE and hands every later draw of
    // that scene the same `Arc` by construction, so a second decode of the same binding in the
    // same scene could only ever produce the identical answer. Cleared wherever
    // `snapshot_sets` is, plus wherever `templates` is - a recorded format is an input here
    // too.
    //
    // Keyed by the texture's ADDRESS as well as its control words: the recorded format and the
    // nearby-handle diagnostic are looked up by address, so two bindings with identical words
    // at different addresses are not interchangeable. `unit` is NOT in the key - it is a
    // property of the binding, not of the texture - so it is stamped onto the clone.
    let key = (addr, binding.words);
    let cached = match cache.decoded_validated(ctx, key) {
        Some(d) => d,
        None => {
            let entry = decode_texture_pixels(ctx, cache, binding, exact_format, byte_stride);
            let d = entry.res.clone();
            // The count bound, now that the memo outlives scenes. A set whose per-binding
            // entry vanishes here simply fails its next re-proof and rebuilds.
            //
            // Shed OLDEST-PROVEN FIRST rather than wholesale: every entry carries the scene it
            // was last proven in, so "least recently useful" is already recorded here and a
            // clear threw it away along with the whole working set. See [`count_victims`].
            if cache.decoded.len() >= TextureSnapshots::DECODED_CAP {
                let keep = TextureSnapshots::DECODED_CAP - TextureSnapshots::DECODED_CAP / 4;
                let scene = cache.scene_seq;
                for k in count_victims(&cache.decoded, keep, scene, |e| e.valid_scene) {
                    cache.decoded.remove(&k);
                }
            }
            cache.decoded.insert(key, entry);
            d
        }
    };
    match cached {
        Ok(mut t) => {
            t.unit = unit;
            Some(t)
        }
        // Noted on EVERY draw that binds it, not once per scene: the drop counters report how
        // many draws lost a texture, and a memo must not change how often a loss is counted.
        Err(code) => {
            note_texture_drop(code as usize);
            None
        }
    }
}

/// The decode itself - everything [`decode_texture`] memoises, returned with the facts the
/// cross-scene re-proof needs ([`DecodedEntry::snap`] / [`DecodedEntry::degraded`]). An
/// `Err(code)` result is a binding the decode DROPS, carrying the drop code its caller
/// reports.
pub(super) fn decode_texture_pixels(
    ctx: &GuestCtx,
    cache: &mut TextureSnapshots,
    binding: &TextureBinding,
    exact_format: Option<u32>,
    byte_stride: u32,
) -> DecodedEntry {
    let unit = binding.unit;
    // >>> EVERYTHING BELOW IS A PURE FUNCTION OF THE FOUR CONTROL WORDS, SO IT IS DONE ONCE.
    //
    // See [`TextureSnapshots::templates`]. The PIXELS are deliberately not part of it - they
    // still go through `get_or_read` on every draw, so a texture the guest rewrites is still
    // seen to change. Only the description is remembered.
    let scene_now = cache.scene_seq;
    let template = match cache.templates.get_mut(&binding.words) {
        Some((t, used)) => {
            *used = scene_now;
            *t
        }
        None => {
            let t = build_texture_template(unit, binding.words, exact_format, byte_stride);
            if cache.templates.len() >= TEXTURE_TEMPLATE_CAP {
                // >>> ONLY THE TEMPLATES, and only the coldest quarter of them.
                //
                // This used to clear `decoded` and `snapshot_sets` as well, on the ground that
                // the finished lists were BUILT from these templates. They were - but a template
                // is a pure function of the four control words (that is the whole basis of the
                // memo), so a rebuilt one is bit-identical to the one dropped and nothing
                // derived from it can be invalidated by dropping it. The clear that IS a
                // correctness question is the one in `set_texture_format`, where the recorded
                // format - an input NOT in this key - actually changes; that one still takes all
                // three. Here it was throwing away the frame's whole texture working set to
                // bound a map of small `Copy` structs.
                let keep = TEXTURE_TEMPLATE_CAP - TEXTURE_TEMPLATE_CAP / 4;
                for k in count_victims(&cache.templates, keep, scene_now, |(_, used)| *used) {
                    cache.templates.remove(&k);
                }
            }
            cache.templates.insert(binding.words, (t, scene_now));
            t
        }
    };
    let scene = cache.scene_seq;
    let entry = move |res, snap, degraded| DecodedEntry {
        res,
        snap,
        degraded,
        valid_scene: scene,
    };
    let Some(t) = template else {
        return entry(Err(1), None, false);
    };
    let mut levels = t.levels;
    let mut face_bytes = t.face_bytes;
    let mut snap_len = t.read_len as usize;
    let mut degraded = false;
    let mut pixels = {
        let _r = crate::perf::scope(crate::perf::Phase::DrawTexRead);
        cache.get_or_read(ctx, t.data_addr, t.read_len as usize)
    };
    // The identity of the buffer just handed back, taken from the layer that MINTED it - see
    // `BoundTexture::pixels_id`. Read here rather than derived downstream, because only this
    // layer knows whether the bytes are the ones it already held or a fresh read.
    let mut pixels_id = cache.pixels_id(t.data_addr, t.read_len as usize);
    // >>> THE CHAIN READ CAN FAIL, AND FALLING BACK IS NOT SILENT. A texture whose allocation
    // really does end after level 0 (a render target sampled as a texture, say - it has no
    // mips whatever its control word says) makes the fuller read unmappable. That is a fact
    // about this texture, not an error, so it degrades to level 0 and SAYS SO once per format.
    if levels > 1 && pixels.len() < t.read_len as usize {
        report_mip_chain_unreadable(unit, t.base_format, t.data_addr, t.width, t.height, levels);
        levels = 1;
        // A cube's faces stay where they are - only the chain read stops at level 0.
        if t.faces == 1 {
            face_bytes = t.level0_bytes;
        }
        snap_len = t.level0_read_len as usize;
        degraded = true;
        pixels = cache.get_or_read(ctx, t.data_addr, t.level0_read_len as usize);
        pixels_id = cache.pixels_id(t.data_addr, t.level0_read_len as usize);
    }
    if pixels.is_empty() {
        report_unreadable_texture(
            unit,
            t.base_format,
            t.data_addr,
            t.read_len as usize,
            t.width,
            t.height,
        );
        return entry(Err(2), None, false);
    }
    let snap = Some((t.data_addr, snap_len));
    // >>> A PALETTED TEXTURE IS EXPANDED HERE, ONCE, RATHER THAN AT EVERY TEXEL FETCH.
    //
    // Both consumers of a `BoundTexture` - the software rasteriser and the GPU uploader - read
    // texels through `render::texel_rgba_face` / `decode_face_fast`, and neither has anywhere
    // to put a colour table: it is a second guest allocation, at an address only the control
    // words know. Threading it through both would mean two more places that can disagree, on
    // the hottest loop in the engine. Turning the indices into their colours right here means
    // everything downstream sees an ordinary 32-bit texture and cannot tell the difference -
    // which is also exactly what the hardware's sampler presents.
    let (base_format, stride, face_bytes, levels, pixels) =
        match crate::render::palette_entries(t.base_format) {
            None => (t.base_format, t.stride, face_bytes, levels, pixels),
            Some(entries) => {
                let pal_len = entries as usize * 4;
                let palette = if t.palette_addr == 0 {
                    Vec::new()
                } else {
                    ctx.read_bytes(t.palette_addr, pal_len)
                };
                if palette.len() < pal_len {
                    report_missing_palette(
                        unit,
                        t.base_format,
                        t.palette_addr,
                        palette.len(),
                        pal_len,
                    );
                    return entry(Err(4), None, false);
                }
                // >>> THE EXPANSION IS REMEMBERED AGAINST ITS INPUTS, and the identity with
                // it. See `TextureSnapshots::expanded_palette` for the measurement: minting a
                // fresh id here re-expanded and re-uploaded every paletted texture in the
                // frame, every frame, over bytes the snapshot layer had just proved unchanged.
                let pal_key = {
                    let mut h: u64 = 0xcbf2_9ce4_8422_2325;
                    let mut mix = |v: u64| {
                        h ^= v;
                        h = h.wrapping_mul(0x0000_0100_0000_01b3);
                    };
                    for b in &palette {
                        mix(*b as u64);
                    }
                    // THE SHAPE TOO, because the expansion is a function of it and the entry is
                    // keyed by the index buffer's guest range: the same range bound as two
                    // different textures (a re-initialised control word, a smaller mip view)
                    // expands differently, and handing one the other's buffer would be the
                    // address-is-not-an-identity defect again in a new place.
                    for v in [
                        t.base_format as u64,
                        t.tex_type as u64,
                        (t.width as u64) << 32 | t.height as u64,
                        (t.stride as u64) << 32 | t.faces as u64,
                        (levels as u64) << 32 | face_bytes as u64,
                    ] {
                        mix(v);
                    }
                    h
                };
                let src_id = pixels_id;
                let held = cache
                    .expanded_palette(t.data_addr, pixels.len(), src_id, pal_key)
                    .map(|e| (e.pixels.clone(), e.id, e.stride, e.face_bytes));
                match held {
                    Some((px, id, stride, fb)) => {
                        pixels_id = id;
                        (crate::render::U8U8U8U8, stride, fb, levels, px)
                    }
                    None => {
                        match expand_paletted_texture(&t, &pixels, &palette, levels, face_bytes) {
                            Some((px, stride, fb)) => {
                                // A DIFFERENT buffer than the one the snapshot layer holds - the
                                // indices expanded through the palette - so it carries an identity
                                // of its own. Minted here rather than inherited: the same indices
                                // under a different palette are different texels, and inheriting
                                // would give them one cache entry.
                                let px: std::sync::Arc<[u8]> = std::sync::Arc::from(px);
                                pixels_id = crate::capture::next_pixels_id();
                                cache.remember_palette(
                                    t.data_addr,
                                    pixels.len(),
                                    PaletteExpansion {
                                        src: src_id,
                                        pal: pal_key,
                                        pixels: px.clone(),
                                        id: pixels_id,
                                        stride,
                                        face_bytes: fb,
                                    },
                                );
                                (crate::render::U8U8U8U8, stride, fb, levels, px)
                            }
                            None => {
                                report_unexpandable_palette(
                                    unit,
                                    t.base_format,
                                    t.tex_type,
                                    t.width,
                                    t.height,
                                );
                                return entry(Err(5), None, false);
                            }
                        }
                    }
                }
            }
        };
    entry(
        Ok(crate::capture::BoundTexture {
            unit,
            base_format,
            swizzle: t.swizzle,
            tex_type: t.tex_type,
            width: t.width,
            height: t.height,
            stride,
            faces: t.faces,
            face_bytes,
            levels,
            data_addr: t.data_addr,
            pixels,
            pixels_id,
            u_addr_mode: t.u_addr_mode,
            v_addr_mode: t.v_addr_mode,
            lod_bias: t.lod_bias,
            lod_min: t.lod_min,
            min_filter: t.min_filter,
            mag_filter: t.mag_filter,
            gamma: t.gamma,
            mip_filter: t.mip_filter,
            // Set per draw by the caller from the context's profiles - see
            // `VitaState::apply_yuv_profiles`; the memoised decode cannot know them.
            yuv_profile: 0,
        }),
        snap,
        degraded,
    )
}

/// Turn a paletted texture's INDICES into the colours its table names, level by level and face
/// by face, producing an ordinary `U8U8U8U8` image with the same dimensions, the same layout
/// rule (Morton or linear) and the same mip chain.
///
/// Returns `(pixels, level-0 stride, bytes per face)`, or `None` when the source or the
/// destination cannot be laid out - which is a refusal, never a partial image.
///
/// # The entry is copied VERBATIM, and that is the whole reason the swizzle needs no thought
/// A palette entry is a 32-bit texel in the texture's own declared swizzle - that is what the
/// swizzle FIELD of a paletted format describes, since the indices themselves have no channels.
/// So the four bytes go across untouched and the `BoundTexture` keeps its swizzle, and every
/// downstream reader applies it exactly as it would to any other 32-bit texel.
pub(super) fn expand_paletted_texture(
    t: &TextureTemplate,
    pixels: &[u8],
    palette: &[u8],
    levels: u32,
    src_face_bytes: u32,
) -> Option<(Vec<u8>, u32, u32)> {
    use crate::render::{P4, U8U8U8U8, level_layout};
    let bits = if t.base_format == P4 { 4u32 } else { 8 };
    let index_mask = if bits == 4 { 0x0fu32 } else { 0xff };
    let entries = crate::render::palette_entries(t.base_format)?;

    // Where each destination level starts inside one expanded face, and how big a face is.
    let mut dst_level_off = Vec::with_capacity(levels as usize);
    let mut dst_face_bytes = 0u32;
    for l in 0..levels {
        dst_level_off.push(dst_face_bytes);
        dst_face_bytes += level_layout(U8U8U8U8, t.tex_type, t.width, t.height, l)?.bytes;
    }
    let dst_stride = level_layout(U8U8U8U8, t.tex_type, t.width, t.height, 0)?.stride;

    let mut src_level_off = Vec::with_capacity(levels as usize);
    let mut acc = 0u32;
    for l in 0..levels {
        src_level_off.push(acc);
        acc += level_layout(t.base_format, t.tex_type, t.width, t.height, l)?.bytes;
    }

    let mut out = vec![0u8; (dst_face_bytes as usize).checked_mul(t.faces as usize)?];
    for face in 0..t.faces {
        let src_face = (face * src_face_bytes) as usize;
        let dst_face = (face * dst_face_bytes) as usize;
        for l in 0..levels {
            let src_l = level_layout(t.base_format, t.tex_type, t.width, t.height, l)?;
            let dst_l = level_layout(U8U8U8U8, t.tex_type, t.width, t.height, l)?;
            let src_base = src_face + src_level_off[l as usize] as usize;
            let dst_base = dst_face + dst_level_off[l as usize] as usize;
            // >>> THE ADDRESSING IS HOISTED OUT OF THE TEXEL LOOP.
            // This used to call `texel_element` TWICE per texel, and in the swizzled case that
            // recomputes two `next_power_of_two`s and runs `morton_index`'s per-bit loop - about
            // ten iterations - each time. MEASURED with a V8 sampling profile of the browser
            // worker on the golf title, whose textures are paletted: `texel_element` was 30.5%
            // of the whole thread and the iterator inside it another 31.9%, against 4.6 ms of
            // actual rendering. The output is unchanged by construction - see
            // `morton_tables`, which is asserted to be the same function as `morton_index`.
            let swizzled = crate::render::swizzled_type(t.tex_type);
            let (sxs, sys) = if swizzled {
                let (pw, ph) = (
                    src_l.width.next_power_of_two(),
                    src_l.height.next_power_of_two(),
                );
                crate::render::morton_tables(pw, ph, src_l.width, src_l.height)
            } else {
                (Vec::new(), Vec::new())
            };
            let (dxs, dys) = if swizzled {
                let (pw, ph) = (
                    dst_l.width.next_power_of_two(),
                    dst_l.height.next_power_of_two(),
                );
                crate::render::morton_tables(pw, ph, dst_l.width, dst_l.height)
            } else {
                (Vec::new(), Vec::new())
            };
            for y in 0..src_l.height {
                let srow = (y * src_l.stride) as usize;
                let drow = (y * dst_l.stride) as usize;
                for x in 0..src_l.width {
                    let (soff, nib) = if swizzled {
                        let m = (sxs[x as usize] + sys[y as usize]) as usize;
                        if bits == 4 {
                            (m / 2, (m % 2) as u32)
                        } else {
                            (m, 0)
                        }
                    } else if bits == 4 {
                        (srow + (x / 2) as usize, x % 2)
                    } else {
                        (srow + x as usize, 0)
                    };
                    let byte = *pixels.get(src_base + soff)? as u32;
                    let index = ((byte >> (nib * 4)) & index_mask).min(entries - 1) as usize;
                    let doff = if swizzled {
                        (dxs[x as usize] + dys[y as usize]) as usize * 4
                    } else {
                        drow + x as usize * 4
                    };
                    let d = dst_base + doff;
                    out.get_mut(d..d + 4)?
                        .copy_from_slice(palette.get(index * 4..index * 4 + 4)?);
                }
            }
        }
    }
    Some((out, dst_stride, dst_face_bytes))
}

/// Report - once per (unit, format) - that a paletted texture has no readable colour table, so
/// the binding is DROPPED rather than sampled as raw indices.
///
/// Sampling the indices is what this path used to do, and it is the failure shape this project
/// exists to avoid: an index of 3 in an 8-bit lane reads as a colour value of 3, so the whole
/// texture comes out very nearly black and looks like a lighting bug.
pub(super) fn report_missing_palette(unit: u32, base_format: u32, addr: u32, got: usize, want: usize) {
    use std::collections::HashSet;
    use std::sync::Mutex;
    static SEEN: Mutex<Option<HashSet<(u32, u32)>>> = Mutex::new(None);
    let mut g = SEEN.lock().unwrap_or_else(|e| e.into_inner());
    if !g
        .get_or_insert_with(HashSet::new)
        .insert((unit, base_format))
    {
        return;
    }
    tracing::warn!(
        target: "vitaslop::gxm",
        unit,
        base_format = format_args!("{base_format:#04x}"),
        palette = format_args!("{addr:#010x}"),
        got,
        want,
        "a PALETTED texture's colour table could not be read - every draw binding this format \
         to this unit is DROPPED. A zero address means the guest never pointed the texture at \
         a palette (or did it through a path this does not see); a short read means the \
         address is wrong."
    );
}

/// Report - once per (unit, format) - that a paletted texture's LAYOUT could not be walked, so
/// the binding is dropped.
pub(super) fn report_unexpandable_palette(
    unit: u32,
    base_format: u32,
    tex_type: u32,
    width: u32,
    height: u32,
) {
    use std::collections::HashSet;
    use std::sync::Mutex;
    static SEEN: Mutex<Option<HashSet<(u32, u32)>>> = Mutex::new(None);
    let mut g = SEEN.lock().unwrap_or_else(|e| e.into_inner());
    if !g
        .get_or_insert_with(HashSet::new)
        .insert((unit, base_format))
    {
        return;
    }
    tracing::warn!(
        target: "vitaslop::gxm",
        unit,
        base_format = format_args!("{base_format:#04x}"),
        tex_type,
        width,
        height,
        "a PALETTED texture could not be laid out for expansion (its index data does not \
         cover its declared levels) - the binding is DROPPED rather than half-expanded"
    );
}

/// Derive everything a binding's four control words say about its texture. Called once per
/// distinct control-word set; see [`TextureSnapshots::templates`].
///
/// `None` is a binding the decode DROPS - a format that cannot be sized. It is memoised too,
/// because a dropped binding is re-bound every frame like any other and re-deriving the drop is
/// the same work as re-deriving the answer.
pub(super) fn build_texture_template(
    unit: u32,
    words: [u32; 4],
    exact_format: Option<u32>,
    byte_stride: u32,
) -> Option<TextureTemplate> {
    let [w0, w1, w2, w3] = words;
    let tex_type = (w1 >> 29) & 0x7;
    // The sampler state, out of the guest's own word 0 rather than a host-side map. See
    // [`word0_sampler_state`] and `vita::gxm::texword0`.
    let sampler = word0_sampler_state(w0, tex_type);
    // Size-1 for most types, log2 for SWIZZLED and CUBE - see `vita::gxm::texture_dims`.
    let (width, height) = crate::vita::gxm::texture_dims(w1);
    let data_addr = w2 & 0xffff_fffc;
    // Prefer the exact format the guest set (full high byte keeps 24-bit/paletted
    // formats and the channel swizzle intact); otherwise reconstruct the high byte
    // from the 5-bit field plus the format0 extension bit (word 0 bit 31), and take
    // the 3-bit control-word swizzle.
    // Both arms must produce the swizzle in the FORMAT FIELD's position (bits 14:12), because
    // that is what every consumer reads it out of (`(swizzle >> 12) & 0x7`). The control words
    // store it as a bare 3-bit value in word 3, so it has to be shifted back up - without that
    // every texture resolved from its control words alone silently reads as swizzle 0, and that
    // is the path a texture takes as soon as a title COPIES the struct it lives in (a colour
    // surface's `backgroundTex` is copied every time the surface is).
    let (base_format, swizzle) = match exact_format {
        Some(f) => ((f >> 24) & 0xff, f & 0x00ff_ffff),
        None => {
            let base = ((w1 >> 24) & 0x1f) | (((w0 >> 31) & 1) << 7);
            // Bits 28:30, per vitasdk's `SceGxmTexture` - see `vita::gxm::texword3`, which
            // records why reading bit 29 here was wrong yet passed every round-trip test.
            use crate::vita::gxm::texword3;
            let swz = (w3 >> texword3::SWIZZLE_SHIFT) & texword3::SWIZZLE_MASK;
            report_texture_resolved_from_control_words(unit, base, swz);
            (base, swz << 12)
        }
    };

    // Block geometry: uncompressed formats are 1x1 texel "blocks"; BC/DXT are 4x4. A format we
    // cannot size is dropped rather than guessed, but never silently: an undecoded unit shows up
    // downstream only as a missing sampler binding, which is far harder to trace back than the
    // format that caused it.
    // `stride` is the bytes per block-row we snapshot, and `level0` the bytes ONE level
    // occupies. The layout arithmetic lives in `render::level_layout` so the chain read below
    // and the level-0 read here cannot drift apart.
    let Some(l0) = crate::render::level_layout(base_format, tex_type, width, height, 0) else {
        report_unsized_texture_format(unit, base_format, tex_type, width, height);
        return None;
    };
    // >>> A LINEAR_STRIDED TEXTURE'S ROWS ARE `byte_stride` APART, NOT A ROW'S WIDTH.
    //
    // The pitch is the explicit stride `sceGxmTextureInitLinearStrided` was given, packed into
    // word 0 (see `vita::gxm::pack_linear_stride`). MEASURED on a fighting title: its
    // bloom samples each mip level as a 128- or 256-wide strided window onto ONE 512-wide
    // surface. Read at the width's pitch, every row after the first came from the wrong place
    // (four texture rows to one surface row), and a renderer resolving the window against the
    // rendered surface saw a texture whose pitch did not match it.
    let strided = tex_type == 6 && byte_stride != 0;
    let stride = if strided { byte_stride } else { l0.stride };
    let level0 = if strided {
        byte_stride * height
    } else {
        l0.bytes
    };
    // A CUBE texture stores its six faces back to back, each laid out exactly like a standalone
    // texture of the same size - so one face is `level0` bytes and the snapshot is six of them.
    let faces = if crate::render::cube_type(tex_type) {
        6
    } else {
        1
    };
    // >>> HOW MANY MIP LEVELS TO READ, and why the count is CLAMPED rather than trusted.
    //
    // `mip_count` is control word 0 bits 17-20 (`vita::gxm::texword0::MIP_COUNT`), stored as
    // the count MINUS ONE (`vita::gxm::mip_count_of_field`). It is clamped to the levels the
    // size can have, because a title that builds its own words can put anything there.
    // >>> AND NOT READ AT ALL IF NOTHING CAN USE IT.
    //
    // The chain is ~33% more guest bytes per texture, and those bytes are re-COMPARED against
    // guest memory once per scene to decide whether the snapshot is stale - a compare this
    // project has measured at 40% of a frame. The only consumer today is the compressed upload
    // path, so on a GPU that cannot take compressed textures this would be a third more of the
    // most expensive loop in the capture, bought for nothing. Level 0 is unchanged either way,
    // so this changes no picture on any engine.
    let mip_field = if vitaslop_platform::gpu::block_compression_available() {
        crate::vita::gxm::mip_count_of_field((w0 >> 17) & 0xf)
    } else {
        1
    };
    let max_levels = crate::render::max_mip_levels(width, height);
    // A cube's six chains are one per face, `render::cube_face_stride` apart - read exactly like
    // a flat texture's, six times.
    let want_levels = mip_field.clamp(1, max_levels);
    // The chain's byte extent, and the level-0 fallback the caller drops to when the guest's
    // allocation turns out not to reach that far. `level_offset` walks every level, which is
    // exactly the sort of per-draw arithmetic this memo exists to stop repeating.
    // A strided texture has ONE level (its mip-count bits are stride bits) laid out at its own
    // pitch, which the mip-chain arithmetic knows nothing about.
    let want_levels = if strided { 1 } else { want_levels };
    let chain = if strided {
        level0
    } else {
        crate::render::level_offset(base_format, tex_type, width, height, want_levels)
            .unwrap_or(level0)
    };
    // >>> A CUBE'S FACES ARE A WHOLE MIP CHAIN APART, however many levels it declares. See
    // `render::cube_face_stride` for the measurement.
    let face_bytes = if faces > 1 && !strided {
        crate::render::cube_face_stride(base_format, tex_type, width, height).unwrap_or(level0)
    } else {
        chain
    };
    // A cube packed at its level-0 size has no room for a chain between its faces, whatever it
    // declares - so it is read, and sampled, as the level 0 it stores.
    let (want_levels, chain) = if face_bytes < chain {
        (1, level0)
    } else {
        (want_levels, chain)
    };
    Some(TextureTemplate {
        base_format,
        // >>> THE PALETTE POINTER IS IN THE CONTROL WORDS, and reading it there is what makes a
        // COPIED texture work. `sceGxmTextureSetPalette`'s own handler keeps a host-side map
        // keyed by the `SceGxmTexture *`, which is exactly the shape that cannot follow a
        // struct copy - and a copy is how the title that needed this binds every one of its
        // paletted textures (`report_texture_resolved_from_control_words` fires on all of
        // them). vitasdk's `SceGxmTexture` declares `palette_addr : 26` as control word 3's
        // low field, and `sceGxmTextureGetPalette` shifts it left by six; the address really is
        // 64-byte aligned in the one run that could be checked. Only ever consulted for a
        // paletted base format.
        palette_addr: crate::render::palette_entries(base_format)
            .map_or(0, |_| (w3 & 0x03ff_ffff) << 6),
        swizzle,
        tex_type,
        width,
        height,
        stride,
        faces,
        face_bytes,
        // A cube's last face ends at its own declared chain: the levels and padding after it are
        // never read, and reading them could run off the end of the guest's allocation.
        read_len: face_bytes * (faces - 1) + chain,
        levels: want_levels,
        level0_bytes: level0,
        level0_read_len: face_bytes * (faces - 1) + level0,
        data_addr,
        // Straight out of the binding's own control word 0, so this is the state the guest
        // handed to GXM for THIS texture - including a by-value copy, which an address-keyed
        // shadow could not follow.
        u_addr_mode: sampler.0,
        v_addr_mode: sampler.1,
        lod_bias: sampler.2,
        // Word 2 bits 0-1 are its HIGH bits, word 3 bits 26-27 the low ones - see `vita::gxm`'s
        // `texture_set_lod_min`. A strided texture has no LOD min (nor any level to clamp to).
        lod_min: if tex_type == crate::vita::gxm::TYPE_LINEAR_STRIDED {
            0
        } else {
            ((w2 & 0x3) << 2) | ((w3 >> 26) & 0x3)
        },
        min_filter: sampler.3,
        mag_filter: sampler.4,
        gamma: sampler.5,
        mip_filter: sampler.6,
    })
}