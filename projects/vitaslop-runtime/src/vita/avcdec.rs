//! SceVideodec / SceAvcdec: the guest's H.264 decoder, and the codec engine's memory.
//!
//! The Vita decodes H.264 in a fixed-function block, and a title reaches it through this
//! API: it asks how much memory a decoder of a given size needs, hands that memory over,
//! then feeds one access unit per call and gets pictures back into buffers of its own.
//! [`crate::vita::video`] is what produced those access units - `SceMp4` only demuxes.
//!
//! Every prototype here is DOCUMENTED (`psp2/videodec.h` in the MIT vita-headers), so
//! nothing about the call shapes is inferred. Two things are not documented and were read
//! off the title's call sites with `--example disasm`:
//!
//! - `sceVideodecQueryMemSize(codec, initInfo, out_size)` - three arguments, the third an
//!   out-parameter the caller adds to its pool arithmetic.
//! - `sceVideodecInitLibraryWithUnmapMem(codec, mem, initInfo, size)` - the plain
//!   `sceVideodecInitLibrary` with the memory the caller allocated from a codec-engine
//!   block passed in.
//!
//! # What this engine actually does with the memory
//!
//! Nothing. The host decoder ([`vitaslop_platform::video`]) has its own working memory on
//! the host side, so the guest's decoder pool is never read or written by us - we only
//! have to keep the guest's own bookkeeping consistent, which means answering the size
//! queries with a plausible number and handing back allocations that live inside the block
//! the guest itself supplied. **The sizes we report are therefore an assumption**, and
//! they say so once per run.

use std::sync::Arc;

use crate::hostcall;
use crate::host::{GuestCtx, Ptr, VitaState};
use crate::SvcOutcome;
use vitaslop_platform::video::{DecodedPicture, PictureFormat, VideoDecode};

/// `SCE_VIDEODEC_TYPE_HW_AVCDEC`, the only codec type this API defines.
const SCE_VIDEODEC_TYPE_HW_AVCDEC: u32 = 0x1001;

/// `SCE_AVCDEC_ERROR_INVALID_PARAM`.
const SCE_AVCDEC_ERROR_INVALID_PARAM: i32 = 0x8062_0002u32 as i32;
/// `SCE_AVCDEC_ERROR_INVALID_STATE`.
const SCE_AVCDEC_ERROR_INVALID_STATE: i32 = 0x8062_0004u32 as i32;
/// `SCE_VIDEODEC_ERROR_INVALID_TYPE`.
const SCE_VIDEODEC_ERROR_INVALID_TYPE: i32 = 0x8062_0801u32 as i32;
/// `SCE_AVCDEC_ERROR_INVALID_TYPE`.
const SCE_AVCDEC_ERROR_INVALID_TYPE: i32 = 0x8062_0001u32 as i32;

/// `SCE_AVCDEC_PIXELFORMAT_YUV420_RASTER`: three planes, the caller supplying two pointers.
const PIXELFORMAT_YUV420_RASTER: u32 = 0x10;
/// `SCE_AVCDEC_PIXELFORMAT_YUV420_PACKED_RASTER`: one buffer, luma then interleaved chroma.
const PIXELFORMAT_YUV420_PACKED_RASTER: u32 = 0x20;

// ---------------------------------------------------------------------------
// Field offsets, all from psp2/videodec.h
// ---------------------------------------------------------------------------

/// `SceAvcdecQueryDecoderInfo` - what a caller asks a decoder to be sized for.
mod query {
    pub const HORIZONTAL: u32 = 0x00;
    pub const VERTICAL: u32 = 0x04;
    pub const NUM_OF_REF_FRAMES: u32 = 0x08;
}

/// `SceAvcdecCtrl`: the decoder handle plus the frame buffer the caller gave it.
mod ctrl {
    pub const HANDLE: u32 = 0x00;
    pub const FRAME_BUF_PTR: u32 = 0x04;
    pub const FRAME_BUF_SIZE: u32 = 0x08;
}

/// `SceAvcdecAu`: one access unit, with its timestamps.
mod au {
    pub const PTS_UPPER: u32 = 0x00;
    pub const PTS_LOWER: u32 = 0x04;
    pub const ES_PTR: u32 = 0x10;
    pub const ES_SIZE: u32 = 0x14;
}

/// `SceAvcdecArrayPicture`: how many pictures came back, and where they go.
mod array_picture {
    pub const NUM_OF_OUTPUT: u32 = 0x00;
    pub const NUM_OF_ELM: u32 = 0x04;
    pub const P_PICTURE: u32 = 0x08;
}

/// `SceAvcdecPicture` = `size` then `SceAvcdecFrame` then `SceAvcdecInfo`.
mod picture {
    /// `SceAvcdecFrame` starts here.
    pub const FRAME: u32 = 0x04;
    /// `SceAvcdecInfo` starts here.
    pub const INFO: u32 = 0x44;
}

/// `SceAvcdecFrame`, relative to [`picture::FRAME`].
mod frame {
    pub const PIXEL_TYPE: u32 = 0x00;
    pub const FRAME_PITCH: u32 = 0x04;
    pub const FRAME_WIDTH: u32 = 0x08;
    pub const FRAME_HEIGHT: u32 = 0x0c;
    pub const HORIZONTAL_SIZE: u32 = 0x10;
    pub const VERTICAL_SIZE: u32 = 0x14;
    pub const CROP_LEFT: u32 = 0x18;
    pub const CROP_RIGHT: u32 = 0x1c;
    pub const CROP_TOP: u32 = 0x20;
    pub const CROP_BOTTOM: u32 = 0x24;
    pub const P_PICTURE_0: u32 = 0x38;
    pub const P_PICTURE_1: u32 = 0x3c;
}

/// `SceAvcdecInfo`, relative to [`picture::INFO`].
mod info {
    pub const NUM_UNITS_IN_TICK: u32 = 0x00;
    pub const TIME_SCALE: u32 = 0x04;
    pub const PTS_UPPER: u32 = 0x18;
    pub const PTS_LOWER: u32 = 0x1c;
}

// ---------------------------------------------------------------------------
// State
// ---------------------------------------------------------------------------

/// One decoder the guest created.
pub struct AvcdecSession {
    /// The handle the guest holds, in its `SceAvcdecCtrl`. Never zero.
    pub handle: u32,
    /// The host's H.264 decoder for this stream.
    pub decoder: Box<dyn VideoDecode>,
    /// Pictures decoded but not yet taken by the guest. The API hands back at most
    /// `numOfElm` per call, and the title asks for one - so a decoder that emits two for
    /// one access unit (a reordering flush) must keep the spare rather than drop it.
    pub ready: std::collections::VecDeque<DecodedPicture>,
    /// Access units submitted, and pictures handed back, for the one line a run says
    /// about how much of the movie actually decoded.
    pub submitted: u64,
    pub delivered: u64,
}

/// One block of guest memory the codec engine is lending out.
///
/// The guest allocates the block itself and hands us the pointer, so this is pure
/// bookkeeping: a bump cursor inside memory that was never ours.
pub struct CodecMemBlock {
    pub uid: i32,
    pub base: u32,
    pub size: u32,
    pub cursor: u32,
}

/// Everything the video-decode API holds between calls.
#[derive(Default)]
pub struct AvcdecState {
    pub blocks: Vec<CodecMemBlock>,
    pub next_block_uid: i32,
    pub sessions: Vec<AvcdecSession>,
    pub next_handle: u32,
    /// Whether the library has been initialised, and for what picture size.
    pub library: Option<(u32, u32)>,
    /// One-shot reports.
    pub reported_sizes: bool,
    pub reported_pixel_format: bool,
    pub reported_decoder: bool,
    /// Pictures handed to the guest so far, which is what `VITASLOP_MOVIE_PICTURE_HASH`
    /// keys its per-picture line on. See [`report_picture_hash`].
    pub pictures_written: u64,
}

impl AvcdecState {
    fn block_mut(&mut self, uid: i32) -> Option<&mut CodecMemBlock> {
        self.blocks.iter_mut().find(|b| b.uid == uid)
    }

    fn session_mut(&mut self, handle: u32) -> Option<&mut AvcdecSession> {
        self.sessions.iter_mut().find(|s| s.handle == handle)
    }
}

// ---------------------------------------------------------------------------
// SceCodecEngine: memory the guest owns, lent to a codec
// ---------------------------------------------------------------------------

/// SceUID sceCodecEngineOpenUnmapMemBlock(void *pMemBlock, SceUInt32 size)
///
/// The caller has already allocated the block; on hardware this is what un-maps it from
/// the CPU and hands it to the codec engine. Here it only starts a bump cursor over memory
/// that stays perfectly mapped - which is why nothing has to be given back at close.
fn do_codec_engine_open_unmap_mem_block(
    _ctx: &mut GuestCtx,
    st: &mut VitaState,
    block: Ptr,
    size: u32,
) -> i32 {
    if block.is_null() || size == 0 {
        return SCE_AVCDEC_ERROR_INVALID_PARAM;
    }
    st.avcdec.next_block_uid += 1;
    let uid = st.avcdec.next_block_uid;
    st.avcdec.blocks.push(CodecMemBlock {
        uid,
        base: block.addr(),
        size,
        cursor: block.addr(),
    });
    uid
}

/// SceUIntVAddr sceCodecEngineAllocMemoryFromUnmapMemBlock(SceUID uid, SceUInt32 size, SceUInt32 alignment)
///
/// Returns an ADDRESS, and zero is its failure - the call site checks `cmp r0,#0`, not a
/// sign. A bump allocator is the whole of it: the block is never freed piecewise in
/// practice (the teardown frees each allocation and then closes the block), and a codec
/// pool that outlives its allocations is what the API is for.
fn do_codec_engine_alloc_memory_from_unmap_mem_block(
    _ctx: &mut GuestCtx,
    st: &mut VitaState,
    uid: i32,
    size: u32,
    alignment: u32,
) -> i32 {
    let Some(block) = st.avcdec.block_mut(uid) else {
        return 0;
    };
    let align = alignment.max(1);
    let start = block.cursor.next_multiple_of(align);
    let Some(end) = start.checked_add(size) else {
        return 0;
    };
    if end > block.base.saturating_add(block.size) {
        tracing::warn!(
            target: "vitaslop::movie",
            uid, size, alignment,
            base = format_args!("{:#010x}", block.base),
            block_size = block.size,
            used = block.cursor - block.base,
            "sceCodecEngineAllocMemoryFromUnmapMemBlock: the block is exhausted"
        );
        return 0;
    }
    block.cursor = end;
    start as i32
}

/// SceInt32 sceCodecEngineFreeMemoryFromUnmapMemBlock(SceUID uid, SceUIntVAddr va)
///
/// A bump allocator cannot free one allocation in the middle, and does not have to: the
/// memory belongs to the guest's own block, which it frees whole. Succeeds so the
/// teardown path runs to its end.
fn do_codec_engine_free_memory_from_unmap_mem_block(
    _ctx: &mut GuestCtx,
    st: &mut VitaState,
    uid: i32,
    _va: Ptr,
) -> i32 {
    if st.avcdec.block_mut(uid).is_none() {
        return SCE_AVCDEC_ERROR_INVALID_PARAM;
    }
    0
}

/// SceInt32 sceCodecEngineCloseUnmapMemBlock(SceUID uid)
fn do_codec_engine_close_unmap_mem_block(
    _ctx: &mut GuestCtx,
    st: &mut VitaState,
    uid: i32,
) -> i32 {
    let before = st.avcdec.blocks.len();
    st.avcdec.blocks.retain(|b| b.uid != uid);
    if st.avcdec.blocks.len() == before {
        return SCE_AVCDEC_ERROR_INVALID_PARAM;
    }
    0
}

// ---------------------------------------------------------------------------
// SceVideodec: the library
// ---------------------------------------------------------------------------

/// How much memory this engine SAYS a decoder of this size needs.
///
/// It is not what the hardware needs, because nothing here uses it: the host decoder holds
/// its own working set. What the number has to be is large enough that the title's own
/// arithmetic over it stays sane and small enough not to eat the guest heap - a DPB's
/// worth of 4:2:0 frames is both, and is at least the right ORDER of magnitude for the
/// thing being modelled.
fn modelled_frame_mem(width: u32, height: u32, ref_frames: u32) -> u32 {
    let frame = width.max(16).saturating_mul(height.max(16)) * 3 / 2;
    let bytes = frame.saturating_mul(ref_frames.clamp(1, 16) + 2);
    // Rounded to a whole MEGABYTE, and that is load-bearing rather than tidy. The caller
    // sizes ONE pool for both of these and its formula is
    // `(videodec + avcdec + 0x101fff) & ~0xfffff` - a megabyte of slack, truncated - and
    // it then asks for the second allocation 1 MiB aligned. Sizes that are not multiples
    // of a megabyte spend that slack on alignment padding and the second allocation runs
    // off the end of the pool the title just made. Real decoder allocations are chunky
    // for the same reason.
    bytes.next_multiple_of(1024 * 1024)
}

/// Say, once, that the memory sizes handed back are modelled rather than measured.
fn report_sizes(st: &mut VitaState, width: u32, height: u32, bytes: u32) {
    if st.avcdec.reported_sizes {
        return;
    }
    st.avcdec.reported_sizes = true;
    tracing::info!(
        target: "vitaslop::movie",
        width, height, bytes,
        "SceVideodec: the decoder memory size reported to the title is MODELLED - the host \
         decoder holds its own working set and never touches the guest's pool. The title \
         allocates this much and it goes unused."
    );
}

/// int sceVideodecQueryMemSize(SceVideodecType codec, const SceVideodecQueryInitInfo *initInfo, SceUInt32 *outSize)
///
/// The third argument is RECOVERED, not documented: the caller passes `sp+0x10` and then
/// adds the word there to its pool size. `initInfo` is `SceVideodecQueryInitInfoHwAvcdec`
/// - size, horizontal, vertical, numOfRefFrames, numOfStreams.
fn do_videodec_query_mem_size(
    ctx: &mut GuestCtx,
    st: &mut VitaState,
    codec: u32,
    init_info: Ptr,
    out_size: Ptr,
) -> i32 {
    if codec != SCE_VIDEODEC_TYPE_HW_AVCDEC {
        return SCE_VIDEODEC_ERROR_INVALID_TYPE;
    }
    if init_info.is_null() || out_size.is_null() {
        return SCE_AVCDEC_ERROR_INVALID_PARAM;
    }
    // `SceVideodecQueryInitInfoHwAvcdec`: size, horizontal, vertical, refFrames, streams.
    let width = ctx.read_u32(init_info.addr() + 4);
    let height = ctx.read_u32(init_info.addr() + 8);
    let refs = ctx.read_u32(init_info.addr() + 12);
    let bytes = modelled_frame_mem(width, height, refs);
    ctx.write_u32(out_size.addr(), bytes);
    report_sizes(st, width, height, bytes);
    0
}

/// int sceVideodecInitLibraryWithUnmapMem(SceVideodecType codec, SceUIntVAddr mem, const SceVideodecQueryInitInfo *initInfo, SceUInt32 memSize)
///
/// The `WithUnmapMem` variant of `sceVideodecInitLibrary`: the caller supplies the memory
/// the library is to work in, which it got from [`codec_engine_alloc_memory_from_unmap_mem_block`].
/// Nothing of ours lives there, so the pointer and size are only recorded.
fn do_videodec_init_library_with_unmap_mem(
    ctx: &mut GuestCtx,
    st: &mut VitaState,
    codec: u32,
    mem: Ptr,
    init_info: Ptr,
    mem_size: u32,
) -> i32 {
    if codec != SCE_VIDEODEC_TYPE_HW_AVCDEC {
        return SCE_VIDEODEC_ERROR_INVALID_TYPE;
    }
    if init_info.is_null() {
        return SCE_AVCDEC_ERROR_INVALID_PARAM;
    }
    let width = ctx.read_u32(init_info.addr() + 4);
    let height = ctx.read_u32(init_info.addr() + 8);
    st.avcdec.library = Some((width, height));
    tracing::debug!(
        target: "vitaslop::movie",
        width, height,
        mem = format_args!("{:#010x}", mem.addr()), mem_size,
        "sceVideodecInitLibraryWithUnmapMem"
    );
    0
}

/// int sceVideodecTermLibrary(SceVideodecType codec)
fn do_videodec_term_library(_ctx: &mut GuestCtx, st: &mut VitaState, codec: u32) -> i32 {
    if codec != SCE_VIDEODEC_TYPE_HW_AVCDEC {
        return SCE_VIDEODEC_ERROR_INVALID_TYPE;
    }
    st.avcdec.library = None;
    0
}

// ---------------------------------------------------------------------------
// SceAvcdec: one decoder
// ---------------------------------------------------------------------------

/// int sceAvcdecQueryDecoderMemSize(SceVideodecType codec, const SceAvcdecQueryDecoderInfo *query, SceAvcdecDecoderInfo *decoderInfo)
fn do_avcdec_query_decoder_mem_size(
    ctx: &mut GuestCtx,
    st: &mut VitaState,
    codec: u32,
    query: Ptr,
    decoder_info: Ptr,
) -> i32 {
    if codec != SCE_VIDEODEC_TYPE_HW_AVCDEC {
        return SCE_AVCDEC_ERROR_INVALID_TYPE;
    }
    if query.is_null() || decoder_info.is_null() {
        return SCE_AVCDEC_ERROR_INVALID_PARAM;
    }
    let width = ctx.read_u32(query.addr() + query::HORIZONTAL);
    let height = ctx.read_u32(query.addr() + query::VERTICAL);
    let refs = ctx.read_u32(query.addr() + query::NUM_OF_REF_FRAMES);
    let bytes = modelled_frame_mem(width, height, refs);
    ctx.write_u32(decoder_info.addr(), bytes);
    report_sizes(st, width, height, bytes);
    0
}

/// int sceAvcdecCreateDecoder(SceVideodecType codec, SceAvcdecCtrl *decoder, const SceAvcdecQueryDecoderInfo *query)
///
/// This is where the HOST decoder is opened. It is given no parameter sets: the stream it
/// will be fed is Annex B and carries its own, which is what the demuxer produces and what
/// the hardware this models is given too.
fn do_avcdec_create_decoder(
    ctx: &mut GuestCtx,
    st: &mut VitaState,
    codec: u32,
    decoder: Ptr,
    query: Ptr,
) -> i32 {
    if codec != SCE_VIDEODEC_TYPE_HW_AVCDEC {
        return SCE_AVCDEC_ERROR_INVALID_TYPE;
    }
    if decoder.is_null() {
        return SCE_AVCDEC_ERROR_INVALID_PARAM;
    }
    let host = match st.video.open_h264_annex_b() {
        Ok(d) => d,
        Err(e) => {
            tracing::warn!(
                target: "vitaslop::movie",
                error = %e,
                "sceAvcdecCreateDecoder: this host cannot decode H.264, so the movie will \
                 not play. The title is told the decoder could not be created and is \
                 expected to carry on without it."
            );
            return SCE_AVCDEC_ERROR_INVALID_STATE;
        }
    };
    if !st.avcdec.reported_decoder {
        st.avcdec.reported_decoder = true;
        let (width, height) = if query.is_null() {
            (0, 0)
        } else {
            (
                ctx.read_u32(query.addr() + query::HORIZONTAL),
                ctx.read_u32(query.addr() + query::VERTICAL),
            )
        };
        // A milestone, not a warning: it says which decoder a movie got, which the panel's
        // STATUS section is for. A backend that cannot decode reports itself further down.
        tracing::info!(
            target: "vitaslop::status",
            width, height, backend = %host.describe(),
            "sceAvcdecCreateDecoder: decoding the movie on the host's own H.264 decoder"
        );
    }
    st.avcdec.next_handle += 1;
    let handle = st.avcdec.next_handle;
    st.avcdec.sessions.push(AvcdecSession {
        handle,
        decoder: host,
        ready: std::collections::VecDeque::new(),
        submitted: 0,
        delivered: 0,
    });
    ctx.write_u32(decoder.addr() + ctrl::HANDLE, handle);
    // The frame buffer the caller put in the control block is memory of its own; it is
    // read back here only so a mismatched teardown is visible in a log.
    tracing::debug!(
        target: "vitaslop::movie",
        handle,
        frame_buf = format_args!("{:#010x}", ctx.read_u32(decoder.addr() + ctrl::FRAME_BUF_PTR)),
        frame_buf_size = ctx.read_u32(decoder.addr() + ctrl::FRAME_BUF_SIZE),
        "sceAvcdecCreateDecoder"
    );
    0
}

/// int sceAvcdecDeleteDecoder(SceAvcdecCtrl *decoder)
fn do_avcdec_delete_decoder(ctx: &mut GuestCtx, st: &mut VitaState, decoder: Ptr) -> i32 {
    if decoder.is_null() {
        return SCE_AVCDEC_ERROR_INVALID_PARAM;
    }
    let handle = ctx.read_u32(decoder.addr() + ctrl::HANDLE);
    let Some(index) = st.avcdec.sessions.iter().position(|s| s.handle == handle) else {
        return SCE_AVCDEC_ERROR_INVALID_PARAM;
    };
    let session = st.avcdec.sessions.remove(index);
    // Whatever this decoder still owed goes with it - see `pictures_owed`.
    PICTURES_OWED.store(0, std::sync::atomic::Ordering::Relaxed);
    // >>> THE ERROR IS DECIDED HERE, AT THE END, WHERE IT CAN BE TRUE.
    //
    // It used to fire the moment 120 access units had produced nothing - and on a loaded
    // machine the host decoder then produced every picture of the movie, so a clean run
    // carried an ERROR saying "the movie will not appear" above a movie that appeared.
    // A decoder that was handed real work and delivered NOTHING by the time the title
    // closed it is the genuine failure, and it is only knowable now.
    if session.delivered == 0 && session.submitted >= PICTURES_OWED_BEFORE_REPORTING {
        tracing::error!(
            target: "vitaslop::movie",
            handle,
            access_units = session.submitted,
            backend = %session.decoder.describe(),
            "sceAvcdecDeleteDecoder: the host decoder was given every access unit of this \
             movie and produced NO pictures - the movie never appeared"
        );
    }
    tracing::info!(
        target: "vitaslop::movie",
        handle,
        access_units = session.submitted,
        pictures = session.delivered,
        "sceAvcdecDeleteDecoder: this decoder's playback is over"
    );
    0
}

/// int sceAvcdecDecodeFlush(SceAvcdecCtrl *decoder)
/// Throw away everything held: a seek, or a stream about to be restarted.
fn do_avcdec_decode_flush(ctx: &mut GuestCtx, st: &mut VitaState, decoder: Ptr) -> i32 {
    if decoder.is_null() {
        return SCE_AVCDEC_ERROR_INVALID_PARAM;
    }
    let handle = ctx.read_u32(decoder.addr() + ctrl::HANDLE);
    let Some(session) = st.avcdec.session_mut(handle) else {
        return SCE_AVCDEC_ERROR_INVALID_PARAM;
    };
    session.ready.clear();
    // A flush discards everything in flight, so nothing is owed for it any more.
    PICTURES_OWED.store(0, std::sync::atomic::Ordering::Relaxed);
    match session.decoder.reset() {
        Ok(()) => 0,
        Err(e) => {
            tracing::warn!(target: "vitaslop::movie", error = %e, "sceAvcdecDecodeFlush failed");
            SCE_AVCDEC_ERROR_INVALID_STATE
        }
    }
}

/// int sceAvcdecDecode(const SceAvcdecCtrl *decoder, const SceAvcdecAu *au, SceAvcdecArrayPicture *array)
///
/// One access unit in, zero or more pictures out. **Zero is a normal answer**: a decoder
/// is pipelined and a stream with B-frames owes its first picture for several access
/// units, which is exactly why the API reports `numOfOutput` rather than promising one.
fn do_avcdec_decode(
    ctx: &mut GuestCtx,
    st: &mut VitaState,
    decoder: Ptr,
    au_ptr: Ptr,
    array: Ptr,
) -> i32 {
    if decoder.is_null() || au_ptr.is_null() || array.is_null() {
        return SCE_AVCDEC_ERROR_INVALID_PARAM;
    }
    let handle = ctx.read_u32(decoder.addr() + ctrl::HANDLE);
    if st.avcdec.session_mut(handle).is_none() {
        return SCE_AVCDEC_ERROR_INVALID_PARAM;
    }

    let es_ptr = ctx.read_u32(au_ptr.addr() + au::ES_PTR);
    let es_size = ctx.read_u32(au_ptr.addr() + au::ES_SIZE);
    let pts = ((ctx.read_u32(au_ptr.addr() + au::PTS_UPPER) as u64) << 32)
        | ctx.read_u32(au_ptr.addr() + au::PTS_LOWER) as u64;
    if es_ptr != 0 && es_size != 0 {
        let bytes = ctx.read_bytes(es_ptr, es_size as usize);
        // What the guest actually handed over. An Annex B access unit starts with a start
        // code, so the first four bytes say at once whether the pointer and the length the
        // title derived from the demuxer's descriptor are the ones we meant it to have -
        // which is the whole of the SceMp4 field map, checked from the other side.
        tracing::debug!(
            target: "vitaslop::movie",
            es = format_args!("{es_ptr:#010x}"), es_size, pts,
            head = format_args!("{:02x?}", &bytes[..bytes.len().min(8)]),
            "sceAvcdecDecode"
        );
        let session = st.avcdec.session_mut(handle).expect("checked above");
        session.submitted += 1;
        AU_SUBMITTED.fetch_add(1, std::sync::atomic::Ordering::Relaxed);
        PICTURES_OWED.fetch_add(1, std::sync::atomic::Ordering::Relaxed);
        if let Err(e) = session.decoder.submit(&bytes, pts as i64) {
            tracing::warn!(
                target: "vitaslop::movie", error = %e, es_size,
                "sceAvcdecDecode: the host decoder refused an access unit"
            );
            return SCE_AVCDEC_ERROR_INVALID_STATE;
        }
    }
    deliver_pictures(ctx, st, handle, array.addr())
}

/// int sceAvcdecDecodeStop(const SceAvcdecCtrl *decoder, SceAvcdecArrayPicture *array)
///
/// End of stream: no more input is coming, so whatever the decoder is holding back for
/// reordering comes out now.
fn do_avcdec_decode_stop(
    ctx: &mut GuestCtx,
    st: &mut VitaState,
    decoder: Ptr,
    array: Ptr,
) -> i32 {
    if decoder.is_null() || array.is_null() {
        return SCE_AVCDEC_ERROR_INVALID_PARAM;
    }
    let handle = ctx.read_u32(decoder.addr() + ctrl::HANDLE);
    let Some(session) = st.avcdec.session_mut(handle) else {
        return SCE_AVCDEC_ERROR_INVALID_PARAM;
    };
    if let Err(e) = session.decoder.finish() {
        tracing::warn!(target: "vitaslop::movie", error = %e, "sceAvcdecDecodeStop failed");
        return SCE_AVCDEC_ERROR_INVALID_STATE;
    }
    deliver_pictures(ctx, st, handle, array.addr())
}

/// Drain what the host decoder has ready into the caller's picture array.
///
/// `numOfElm` is how many the caller has room for; anything past that stays queued for the
/// next call rather than being dropped, because a dropped picture is a frame of the movie
/// that never appears and nothing downstream would say so.
fn deliver_pictures(ctx: &mut GuestCtx, st: &mut VitaState, handle: u32, array: u32) -> i32 {
    let session = st.avcdec.session_mut(handle).expect("caller checked the handle");
    loop {
        match session.decoder.poll() {
            Ok(Some(p)) => session.ready.push_back(p),
            Ok(None) => break,
            Err(e) => {
                tracing::warn!(target: "vitaslop::movie", error = %e, "the host decoder failed");
                return SCE_AVCDEC_ERROR_INVALID_STATE;
            }
        }
    }

    let capacity = ctx.read_u32(array + array_picture::NUM_OF_ELM);
    let list = ctx.read_u32(array + array_picture::P_PICTURE);
    // >>> WHERE THE CALLER'S PICTURES GO, because "frame, black, frame" is a statement
    // about BUFFERS and every counter here was a statement about pictures.
    //
    // A call that delivers nothing leaves whatever the caller offered untouched. That is
    // correct - `numOfOutput` says so - but a title that flips its own buffer regardless
    // then shows a buffer this engine has never written, which is black the first time
    // round and stale afterwards. Whether that can happen at all depends on how many
    // distinct destination buffers the title rotates through, which nothing recorded.
    for i in 0..capacity.min(4) {
        if list == 0 {
            break;
        }
        let slot = ctx.read_u32(list + i * 4);
        if slot == 0 {
            continue;
        }
        let dest = ctx.read_u32(slot + picture::FRAME + frame::P_PICTURE_0);
        note_destination(dest);
        tracing::debug!(
            target: "vitaslop::movie",
            dest = format_args!("{dest:#010x}"), i,
            ready = st.avcdec.session_mut(handle).map_or(0, |s| s.ready.len()),
            "sceAvcdecDecode: destination offered"
        );
    }
    DECODE_CALLS.fetch_add(1, std::sync::atomic::Ordering::Relaxed);
    SLOTS_OFFERED.fetch_add(capacity as u64, std::sync::atomic::Ordering::Relaxed);
    let mut written = 0u32;
    while written < capacity {
        let session = st.avcdec.session_mut(handle).expect("caller checked the handle");
        let Some(pic) = session.ready.pop_front() else {
            break;
        };
        session.delivered += 1;
        PICTURES_DELIVERED.fetch_add(1, std::sync::atomic::Ordering::Relaxed);
        // Saturating: a decoder may hand back more pictures than the units still outstanding
        // after a flush cleared the gauge, and a wrap would leave the idle path spinning tasks
        // for the rest of the run.
        let _ = PICTURES_OWED.fetch_update(
            std::sync::atomic::Ordering::Relaxed,
            std::sync::atomic::Ordering::Relaxed,
            |v| Some(v.saturating_sub(1)),
        );
        if list == 0 {
            break;
        }
        let slot = ctx.read_u32(list + written * 4);
        if slot == 0 {
            break;
        }
        write_picture(ctx, st, handle, slot, &pic);
        written += 1;
    }
    ctx.write_u32(array + array_picture::NUM_OF_OUTPUT, written);
    if written == 0 {
        EMPTY_CALLS.fetch_add(1, std::sync::atomic::Ordering::Relaxed);
    }
    let session = st.avcdec.session_mut(handle).expect("caller checked the handle");
    if session.delivered <= 4 || session.submitted % 200 == 0 {
        tracing::debug!(
            target: "vitaslop::movie",
            written, capacity, queued = session.ready.len(),
            submitted = session.submitted, delivered = session.delivered,
            owes = session.decoder.owes_frames(),
            "sceAvcdecDecode: pictures out"
        );
    }
    // >>> A DECODER THAT TAKES EVERYTHING AND GIVES NOTHING BACK IS SILENT OTHERWISE.
    //
    // Every call here succeeds: the access unit was accepted and `numOfOutput` is a
    // legitimate zero, which is what a pipelined decoder answers at the start of a stream.
    // A decoder that NEVER answers anything else looks identical, call by call, and the
    // only visible consequence is a movie that does not appear and a title that spins
    // waiting for it - which is what a phone reported after the desktop played the same
    // movie fine. So the condition is reported once, from the count, at the point where
    // "still starting up" has stopped being a plausible reading.
    if session.delivered == 0 && session.submitted == PICTURES_OWED_BEFORE_REPORTING {
        let backend = session.decoder.describe();
        // A WARNING, worded as what is known: the decoder is late, and whether it ever
        // answers is decided at `sceAvcdecDeleteDecoder`, which is where the ERROR lives.
        tracing::warn!(
            target: "vitaslop::movie",
            submitted = PICTURES_OWED_BEFORE_REPORTING, %backend,
            "the host decoder has been given {PICTURES_OWED_BEFORE_REPORTING} access units              and has produced NO pictures yet. Every call succeeded and reported zero outputs,              which is what a decoder answers while it fills its pipeline - but not usually              this many times. If the movie never appears, this is why; if it appears late,              the host decoder was slow to start."
        );
    }
    0
}

/// How many access units a decoder may take without answering before that is reported as a
/// failure. A real pipeline is a few frames deep - the Windows decoder measured 14 without
/// a low-latency hint, 2 with one - so this is far above any legitimate start-up and far
/// below a movie.
const PICTURES_OWED_BEFORE_REPORTING: u64 = 120;

// >>> THE MOVIE'S PICTURE CADENCE, BECAUSE IT WAS ONLY EVER READABLE BY INFERENCE.
//
// A device reported the title-screen movie playing "frame, black, frame". Nothing on the
// panel said how fast pictures were arriving; the only trace of it was that the ENCODE
// counters showed `textures 0.2 UPLOADED` a frame where the same screen on this desktop
// showed 1.0 - i.e. the movie's guest buffer changed one frame in five, so the decoder was
// delivering about nine pictures a second against a thirty-frame movie. Reading a decoder's
// throughput out of a texture-upload counter is two inferences deep and needs a desktop
// capture of the same screen beside it to mean anything.
//
// These are the direct reading, and they are counters rather than a per-call log for the
// reason the rest of this file's reports are: the question is a RATE over a run, and the
// device that can answer it has no console. Both are cumulative for the run.
static AU_SUBMITTED: std::sync::atomic::AtomicU64 = std::sync::atomic::AtomicU64::new(0);
static PICTURES_DELIVERED: std::sync::atomic::AtomicU64 = std::sync::atomic::AtomicU64::new(0);
/// Access units submitted that no picture has come back for YET - a LIVE gauge, not the
/// difference of two run totals. See [`pictures_owed`], which is read by the browser
/// scheduler's idle path on every idle round, so it has to fall back to zero when playback
/// ends rather than staying at whatever the movie finished short by.
static PICTURES_OWED: std::sync::atomic::AtomicU64 = std::sync::atomic::AtomicU64::new(0);
/// How often the title asked for pictures, how many slots it offered across those calls, and
/// how many of those calls handed back nothing at all. `EMPTY_CALLS / DECODE_CALLS` is the
/// share of the movie's own frames that had no new picture to show, which is the flicker
/// stated as a fraction rather than inferred from a rate.
static DECODE_CALLS: std::sync::atomic::AtomicU64 = std::sync::atomic::AtomicU64::new(0);
static SLOTS_OFFERED: std::sync::atomic::AtomicU64 = std::sync::atomic::AtomicU64::new(0);
static EMPTY_CALLS: std::sync::atomic::AtomicU64 = std::sync::atomic::AtomicU64::new(0);
/// The distinct guest buffers the title has offered as a destination, up to four, and how
/// many it has offered in total. One buffer means a call that delivers nothing leaves the
/// LAST picture on screen (a held frame); two or more mean it can show a buffer that was
/// never written, and THAT is a black frame rather than a held one.
static DEST_BUFFERS: [std::sync::atomic::AtomicU32; 4] = [
    std::sync::atomic::AtomicU32::new(0),
    std::sync::atomic::AtomicU32::new(0),
    std::sync::atomic::AtomicU32::new(0),
    std::sync::atomic::AtomicU32::new(0),
];
static DEST_BUFFER_COUNT: std::sync::atomic::AtomicU32 = std::sync::atomic::AtomicU32::new(0);

/// Record a destination buffer the title offered, keeping the first four distinct addresses.
fn note_destination(addr: u32) {
    use std::sync::atomic::Ordering::Relaxed;
    if addr == 0 {
        return;
    }
    for slot in DEST_BUFFERS.iter() {
        match slot.compare_exchange(0, addr, Relaxed, Relaxed) {
            Ok(_) => {
                DEST_BUFFER_COUNT.fetch_add(1, Relaxed);
                return;
            }
            Err(seen) if seen == addr => return,
            Err(_) => continue,
        }
    }
    // More than four: the count keeps rising so the report can say "at least".
    DEST_BUFFER_COUNT.fetch_max(5, Relaxed);
}

/// Forget every access unit and picture counted so far.
///
/// Called when the live loop leaves a FAST-FORWARD, for the same reason it resets its own
/// frame-cost meters there: a fast-forward runs the guest unpaced and presents nothing, so
/// the worker never returns to the JS event loop and a decoder that answers on a callback
/// cannot answer at all. MEASURED on this desktop, a run that fast-forwarded 1400 frames:
/// 819 access units submitted against 160 pictures, i.e. essentially the whole backlog was
/// the fast-forward. Leaving that in the total makes paced play look like a decoder five
/// times behind whatever it is actually doing.
pub fn reset_movie_counters() {
    use std::sync::atomic::Ordering::Relaxed;
    AU_SUBMITTED.store(0, Relaxed);
    PICTURES_DELIVERED.store(0, Relaxed);
    DECODE_CALLS.store(0, Relaxed);
    SLOTS_OFFERED.store(0, Relaxed);
    EMPTY_CALLS.store(0, Relaxed);
    PICTURES_OWED.store(0, Relaxed);
}

/// The raw movie counters: access units submitted, pictures delivered, calls made, calls that
/// handed back nothing. Cumulative since the last reset.
///
/// # A cumulative ratio cannot describe a movie that starts part way through the run
/// `movie_report`'s "per displayed frame" divides by every frame since the pacing meters were
/// reset, and this title's front end plays its movie from around frame 1300 of a run that
/// started counting at 900. That reads as 0.34 pictures a frame where the title is in fact
/// asking for exactly ONE per frame - a factor of three, in the direction that makes a healthy
/// decoder look starved. A caller that wants a RATE has to difference these itself over a
/// window it knows the length of.
/// Access units the guest has submitted that no picture has come back for yet.
///
/// # What reads it, and why it is not just a diagnostic
/// A `VideoDecoder` hands its pictures back through a callback, and a callback is a TASK - so
/// a browser worker that never returns to its event loop can submit a whole movie and receive
/// none of it. The scheduler's idle path is where that turn comes from, and it hands one out
/// only after a long run of consecutive idle rounds, on the reasoning that idling must not
/// cost a task per round. This is the counter that says the reasoning does not apply right
/// now: a decoder that OWES pictures is precisely the "the emulator is waiting for something
/// only the event loop can deliver" case, and it is worth a turn immediately rather than after
/// sixty-four rounds that may never come.
///
/// MEASURED on one title's front end through the shipping page: `10 access units submitted, 0
/// pictures delivered`, `100.0%` of the guest's calls handed back nothing, and `event loop: 0
/// extra turns from the idle path` - the title has enough live threads that the scheduler
/// never reaches the sixty-fourth consecutive idle round at all, so the decoder's whole budget
/// was the single turn the tick gives per displayed frame, and a picture needs two.
pub fn pictures_owed() -> u64 {
    PICTURES_OWED.load(std::sync::atomic::Ordering::Relaxed)
}

pub fn movie_counters() -> (u64, u64, u64, u64) {
    use std::sync::atomic::Ordering::Relaxed;
    (
        AU_SUBMITTED.load(Relaxed),
        PICTURES_DELIVERED.load(Relaxed),
        DECODE_CALLS.load(Relaxed),
        EMPTY_CALLS.load(Relaxed),
    )
}

/// One line on how many pictures the movie decoder actually produced, or empty if no movie
/// was ever decoded. `frames` is the run's display-flip count, so the ratio is pictures per
/// displayed frame - which is what a stutter or a flicker is a statement about.
///
/// A 30 fps movie on a 60 Hz display is 0.50 here and a 60 fps one is 1.00; anything far
/// below the movie's own rate is the decoder not keeping up, and `owed` says whether that
/// is because it was never fed or because it never answered.
pub fn movie_report(frames: u64) -> Vec<String> {
    use std::sync::atomic::Ordering::Relaxed;
    let (sub, del) = (AU_SUBMITTED.load(Relaxed), PICTURES_DELIVERED.load(Relaxed));
    if sub == 0 {
        return Vec::new();
    }
    let per_frame = if frames > 0 { del as f64 / frames as f64 } else { 0.0 };
    let (calls, empty, slots) = (
        DECODE_CALLS.load(Relaxed),
        EMPTY_CALLS.load(Relaxed),
        SLOTS_OFFERED.load(Relaxed),
    );
    let buffers = DEST_BUFFER_COUNT.load(Relaxed);
    let empty_pct = if calls > 0 { empty as f64 * 100.0 / calls as f64 } else { 0.0 };
    let slots_per_call = if calls > 0 { slots as f64 / calls as f64 } else { 0.0 };
    vec![
        format!(
            "movie: {sub} access units submitted, {del} pictures delivered \
             ({per_frame:.2} per displayed frame, {} still owed). A 30 fps movie on a 60 Hz \
             display is 0.50 per frame; well under the movie's own rate is the decoder not \
             keeping up, and a title that shows a buffer it expected to be filled shows the \
             last one - or nothing - on the frames it is not",
            sub.saturating_sub(del),
        ),
        format!(
            "movie cadence: {calls} calls asked for pictures, offering {slots_per_call:.2} \
             slots each, and {empty} of them ({empty_pct:.1}%) handed back NOTHING. The title \
             rotates {} destination buffer(s). One buffer and an empty call HOLDS the last \
             picture; more than one and an empty call shows a buffer this engine may never \
             have written, which is the black half of a flicker",
            if buffers >= 5 { "4+".to_string() } else { buffers.to_string() },
        ),
    ]
}

/// `(num_units_in_tick, time_scale)` for the movie currently open, in H.264's VUI convention,
/// or `None` if there is no movie or its video track does not say.
///
/// The frame duration is taken from the track's samples rather than from the track duration
/// divided by the count: a container with one odd-length sample at the end would skew the
/// average, and what this describes is the cadence of the picture in hand. The first sample
/// whose duration is non-zero decides it - a variable-frame-rate movie has no single answer to
/// this question and the API has nowhere to put one.
fn picture_frame_timing(st: &VitaState) -> Option<(u32, u32)> {
    let movie = st.movie.as_ref()?;
    let track = movie.mp4.tracks.iter().find(|t| t.kind == crate::mp4::TrackKind::Video)?;
    let units = track.samples.iter().map(|s| s.duration).find(|&d| d != 0)?;
    let scale = u32::try_from(track.timescale.checked_mul(2)?).ok()?;
    Some((u32::try_from(units).ok()?, scale))
}

/// Fill one `SceAvcdecPicture` from a decoded frame, writing the pixels into the buffer
/// the CALLER supplied at `frame.pPicture[0]`.
///
/// The caller also chose the pixel format and the pitch; both are honoured rather than
/// overwritten, because the buffer was sized for them and whatever samples that buffer
/// next was configured for them too.
fn write_picture(
    ctx: &mut GuestCtx,
    st: &mut VitaState,
    handle: u32,
    picture: u32,
    pic: &DecodedPicture,
) {
    let f = picture + picture::FRAME;
    let pixel_type = ctx.read_u32(f + frame::PIXEL_TYPE);
    let dest = ctx.read_u32(f + frame::P_PICTURE_0);
    let pitch = ctx.read_u32(f + frame::FRAME_PITCH).max(pic.width);
    let height = ctx.read_u32(f + frame::FRAME_HEIGHT).max(pic.height);

    if dest != 0 {
        // The surface the engine just laid down, when it laid down the WHOLE of it. Handed
        // to the texture cache below so the draw that binds this picture does not read it
        // back out of guest memory to discover what the engine already knows.
        let authored = match pixel_type {
            PIXELFORMAT_YUV420_PACKED_RASTER => write_nv12(ctx, dest, pitch, height, pic),
            PIXELFORMAT_YUV420_RASTER => {
                // Three planes; the second pointer is where chroma starts. A caller that
                // left it null gets the packed layout, which is the only thing one pointer
                // can describe.
                let chroma = ctx.read_u32(f + frame::P_PICTURE_1);
                if chroma == 0 {
                    write_nv12(ctx, dest, pitch, height, pic)
                } else {
                    write_i420(ctx, dest, chroma, pitch, height, pic);
                    None
                }
            }
            other => {
                report_unsupported_pixel_format(st, other);
                None
            }
        };
        if let Some(bytes) = authored {
            report_picture_hash(st, &bytes, pitch, height);
            st.author_texture_bytes(ctx, dest, bytes);
        }
    }

    ctx.write_u32(f + frame::FRAME_WIDTH, pitch);
    ctx.write_u32(f + frame::FRAME_HEIGHT, height);
    ctx.write_u32(f + frame::HORIZONTAL_SIZE, pic.width);
    ctx.write_u32(f + frame::VERTICAL_SIZE, pic.height);
    ctx.write_u32(f + frame::CROP_LEFT, 0);
    ctx.write_u32(f + frame::CROP_RIGHT, pitch.saturating_sub(pic.width));
    ctx.write_u32(f + frame::CROP_TOP, 0);
    ctx.write_u32(f + frame::CROP_BOTTOM, height.saturating_sub(pic.height));

    // >>> THE PICTURE'S OWN FRAME RATE, FROM THE STREAM - IT USED TO SAY 1/60 FOR EVERY MOVIE.
    //
    // `num_units_in_tick / time_scale` is H.264's VUI frame timing and the only statement this
    // structure makes about how long a picture lasts. It was hardcoded to `1 / 60`, which is a
    // claim that every movie ever decoded is 60 fps.
    //
    // MEASURED on this title's front end, where the movie is 29.97 fps (`time_base 1/29970`,
    // 1000 ticks a frame, 1241 samples): the title submits **exactly one access unit per
    // display flip** at 60 flips a second, so 210 flips consumed 210 frames - 7.0 seconds of
    // movie in 3.5 seconds of guest time, **2.0x playback**, reproducible on this desktop. The
    // access-unit rate is the playback rate here, because the title advances its own
    // destination buffer only when a picture comes back (the `dest` sequence shows 18 calls
    // into one buffer while the decoder filled its pipeline, then a strict 3-cycle).
    //
    // The convention is `fps = time_scale / (2 * num_units_in_tick)`, so a 29.97 fps track with
    // a 29970 timescale and 1000-tick frames is `59940 / (2 * 1000)`. Taken from the container
    // the picture actually came from; the old constants remain the fallback for a stream that
    // does not say, which is the only case where a guess is the only thing available.
    let (units_in_tick, time_scale) = picture_frame_timing(st).unwrap_or((1, 60));
    let i = picture + picture::INFO;
    ctx.write_u32(i + info::NUM_UNITS_IN_TICK, units_in_tick);
    ctx.write_u32(i + info::TIME_SCALE, time_scale);
    ctx.write_u32(i + info::PTS_UPPER, (pic.pts as u64 >> 32) as u32);
    ctx.write_u32(i + info::PTS_LOWER, pic.pts as u32);

    if !st.avcdec.reported_pixel_format {
        st.avcdec.reported_pixel_format = true;
        // >>> WARN, NOT INFO, AND ON PURPOSE.
        //
        // It fires ONCE per run and it is the line that says the whole video path reached
        // its end: a picture, of a known size, in a known layout, in guest memory. The
        // alternative is a run that renders nothing and a diagnostic dump with no evidence
        // either way in it, because a device's default log level is `warn` - and asking a
        // person holding the phone to set a knob first costs them the run. It also carries
        // what the decoder turned out to BE, which in a browser is not knowable until it
        // has decoded something.
        let backend = st
            .avcdec
            .session_mut(handle)
            .map(|s| s.decoder.describe())
            .unwrap_or_default();
        // A milestone for the panel's STATUS section, not a console warning.
        tracing::info!(
            target: "vitaslop::status",
            pixel_type = format_args!("{pixel_type:#x}"),
            pitch, height,
            visible = format_args!("{}x{}", pic.width, pic.height),
            dest = format_args!("{dest:#010x}"),
            %backend,
            "the movie's FIRST PICTURE reached guest memory. `packed raster` is taken to              be luma then INTERLEAVED chroma (NV12) - one buffer is the only thing the              caller's single pointer can describe, but the chroma ORDER within it is an              assumption until the texture the title binds over this buffer says otherwise."
        );
    }
}

/// >>> THE ORACLE FOR THE MOVIE PATH, because the RENDERED FRAME IS NOT ONE.
///
/// `VITASLOP_MOVIE_PICTURE_HASH=1` prints one line per picture: which picture it is, its
/// shape, and a hash of the exact bytes the guest's buffer now holds.
///
/// # Why a screenshot cannot answer this question
/// A movie frame is delivered by a decoder running on the HOST's clock, and the guest takes
/// whatever picture is ready when it looks. MEASURED here: three runs of the same recipe -
/// two of them the same binary - produced three different `f001400.png`, while every
/// non-movie shot in the same runs was byte-identical. So "the picture did not change" cannot
/// be established by comparing a shot inside the movie, and a claim that it was is a claim
/// about scheduling luck. [[vitaslop-a-hash-is-half-an-instrument]] is why the line carries
/// the shape and the byte count as well: a hash alone cannot tell a changed picture from a
/// picture that stopped being written.
///
/// The SEQUENCE of pictures is deterministic - it is the file - so picture N's hash is
/// comparable across runs even when the frame it lands on is not.
fn report_picture_hash(st: &mut VitaState, bytes: &[u8], pitch: u32, height: u32) {
    let n = st.avcdec.pictures_written;
    st.avcdec.pictures_written += 1;
    dump_picture(n, bytes, pitch, height);
    if !crate::knobs::flag("VITASLOP_MOVIE_PICTURE_HASH") {
        return;
    }
    // FNV-1a over the surface. Not a cryptographic question: the comparison is against the
    // same run of the same file, and what it has to catch is a plane written to the wrong
    // offset or a conversion that changed.
    let mut h: u64 = 0xcbf2_9ce4_8422_2325;
    for b in bytes {
        h ^= *b as u64;
        h = h.wrapping_mul(0x0000_0100_0000_01B3);
    }
    // The MEAN LUMA beside the hash, because a hash cannot tell a black picture from a
    // bright one and "the movie is black" is the question this diagnostic gets asked.
    let luma_at = pitch as usize * height as usize;
    let mean = bytes[..luma_at.min(bytes.len())].iter().map(|&b| b as u64).sum::<u64>()
        / (luma_at.min(bytes.len()).max(1) as u64);
    tracing::info!(
        target: "vitaslop::movie",
        picture = n,
        pitch, height,
        len = bytes.len(),
        mean_luma = mean,
        hash = format_args!("{h:#018x}"),
        "movie picture written to guest memory"
    );
}

/// >>> AND WHAT THE PICTURE ACTUALLY LOOKS LIKE, because "a picture arrived" and "the movie
/// is playing" are different claims and only one of them a hash can make.
///
/// `VITASLOP_MOVIE_DUMP_DIR=<dir>` writes every `VITASLOP_MOVIE_DUMP_EVERY`th picture (default
/// 30) as `movie-<n>.png`, converted out of the surface the guest was just given - so it is
/// the guest's bytes that are looked at, not the decoder's. That distinction is the whole
/// point: a black frame on screen can be a decoder producing black, a conversion writing the
/// planes to the wrong offsets, or a draw that never sampled the texture, and only this
/// separates the first from the other two.
fn dump_picture(n: u64, bytes: &[u8], pitch: u32, height: u32) {
    let Ok(dir) = crate::knobs::var("VITASLOP_MOVIE_DUMP_DIR") else { return };
    let every: u64 = crate::knobs::var("VITASLOP_MOVIE_DUMP_EVERY")
        .ok()
        .and_then(|s| s.trim().parse().ok())
        .unwrap_or(30);
    if every == 0 || n % every != 0 {
        return;
    }
    let (w, h) = (pitch as usize, height as usize);
    let chroma_at = w * h;
    let mut rgba = vec![0u8; w * h * 4];
    for y in 0..h {
        for x in 0..w {
            let luma = bytes[y * w + x] as i32;
            let c = chroma_at + (y / 2) * w + (x / 2) * 2;
            let (cb, cr) = match (bytes.get(c), bytes.get(c + 1)) {
                (Some(&a), Some(&b)) => (a as i32, b as i32),
                _ => (128, 128),
            };
            let (yt, u, v) = ((luma - 16) * 76309, cb - 128, cr - 128);
            let at = (y * w + x) * 4;
            rgba[at] = (((yt + 104597 * v + 32768) >> 16).clamp(0, 255)) as u8;
            rgba[at + 1] = (((yt - 25675 * u - 53279 * v + 32768) >> 16).clamp(0, 255)) as u8;
            rgba[at + 2] = (((yt + 132201 * u + 32768) >> 16).clamp(0, 255)) as u8;
            rgba[at + 3] = 255;
        }
    }
    let path = std::path::Path::new(&dir).join(format!("movie-{n:05}.png"));
    if let Err(e) = std::fs::write(&path, crate::render::rgba_to_png(pitch, height, &rgba)) {
        tracing::warn!(target: "vitaslop::movie", error = %e, path = %path.display(),
            "could not write the movie picture dump");
    }
}

/// Say, once, that a pixel format this engine cannot produce was asked for.
fn report_unsupported_pixel_format(st: &mut VitaState, pixel_type: u32) {
    if st.avcdec.reported_pixel_format {
        return;
    }
    st.avcdec.reported_pixel_format = true;
    tracing::error!(
        target: "vitaslop::movie",
        pixel_type = format_args!("{pixel_type:#x}"),
        "sceAvcdecDecode: the title asked for a pixel format this engine does not produce. \
         The picture buffer is left UNTOUCHED rather than filled with something plausible, \
         so the movie shows whatever was already there."
    );
}

/// Luma plane then interleaved Cb/Cr, at the caller's pitch (NV12).
///
/// Returns the bytes it wrote when they are the WHOLE of the guest's surface, so the caller
/// can hand them to the texture cache - see [`author_picture`].
///
/// # One conversion, or none
///
/// The guest asked for NV12 and the decoder produced whatever suited it, so this converts
/// only where the two differ. That is the whole point of [`DecodedPicture`] carrying its own
/// format: an NV12 decoder feeding an NV12 guest - the ordinary case on three of the four
/// backends - is now a row copy, where it used to be NV12 -> I420 -> NV12, two conversions
/// of every pixel to arrive back where it started.
///
/// # And it composes the surface before writing it, rather than writing it in 816 pieces
///
/// The picture is laid out once, in host memory, at the guest's own pitch; the write is then
/// ONE call instead of one per row (544 luma + 272 chroma on this title's movie), and the
/// buffer that was laid out is exactly what guest memory now holds - which is what makes it
/// legitimate to hand to the texture cache instead of having a draw read it back out.
/// A surface with padding columns, or one the picture does not fill, is written row by row
/// out of the same buffer so the bytes BETWEEN the rows keep whatever the guest put there,
/// and nothing is authored, because in that case the buffer is not the surface.
/// >>> `PIXELFORMAT_YUV420_PACKED_RASTER` IS **V THEN U**, AND THE TITLES SAY SO.
///
/// The interleaved chroma plane this writes used to be laid down Cb,Cr - ordinary NV12 - and
/// the picture then reached the screen with red and blue exchanged: a golf title's opening
/// movie rendered an ORANGE sky over a BLUE golfer.
///
/// The layout is not ours to choose. The guest binds a texture over this very buffer and that
/// texture DECLARES how the bytes are ordered, so it is the record of what the console's own
/// decoder wrote. MEASURED on two independent titles, from the control words of the texture
/// each binds over its movie surface: both are `0x00003000` =
/// `SCE_GXM_TEXTURE_SWIZZLE_YVU_CSC1` (vitasdk `psp2/gxm.h`), i.e. **V first**. One title
/// could be a quirk; two agreeing is the layout.
///
/// The reader ([`vitaslop_runtime::render`]'s YUV420P2 decode) already honours the swizzle
/// exactly, which is why this shows up as a clean channel swap rather than as noise.
///
/// Still an ASSUMPTION, and a different one: both titles also ask for `CSC1` while this engine
/// converts every 4:2:0 texture with the BT.601 studio-swing profile. That is reported once per
/// swizzle by `report_yuv_profile_assumed` and is a separate question from the byte order.
const PACKED_RASTER_IS_VU: () = ();

fn write_nv12(
    ctx: &mut GuestCtx,
    dest: u32,
    pitch: u32,
    height: u32,
    pic: &DecodedPicture,
) -> Option<Arc<[u8]>> {
    let (w, h) = (pic.width as usize, pic.height as usize);
    let pitch = pitch as usize;
    let height = height as usize;
    let rows = height.min(h);
    let (cw, ch) = (w.div_ceil(2), h.div_ceil(2));
    let chroma_rows = ch.min(height / 2);
    if pitch == 0 || rows == 0 {
        return None;
    }
    let chroma_at = pitch * height;
    let mut buf = vec![0u8; chroma_at + pitch * chroma_rows];

    if pic.format == PictureFormat::Rgba {
        lay_rgba_as_nv12(&mut buf, pitch, chroma_at, rows, chroma_rows, pic);
    } else {
        // Luma is the same plane in both 4:2:0 layouts: a row copy at the caller's pitch.
        for y in 0..rows {
            let at = y * w;
            buf[y * pitch..y * pitch + w].copy_from_slice(&pic.data[at..at + w]);
        }
        match pic.format {
            // Already interleaved - but the host's NV12 is Cb,Cr and the guest reads V,U
            // (`PACKED_RASTER_IS_VU`), so the pairs are swapped on the way across rather
            // than copied.
            PictureFormat::Nv12 => {
                let base = pic.chroma_offset();
                for y in 0..chroma_rows {
                    let at = base + y * cw * 2;
                    let to = chroma_at + y * pitch;
                    for x in 0..cw.min(pitch / 2) {
                        buf[to + 2 * x] = pic.data[at + 2 * x + 1];
                        buf[to + 2 * x + 1] = pic.data[at + 2 * x];
                    }
                }
            }
            // Two quarter-size planes woven into one half-height plane, V FIRST - see
            // `PACKED_RASTER_IS_VU`.
            PictureFormat::I420 => {
                let (cb, cr) = (pic.chroma_offset(), pic.cr_offset());
                for y in 0..chroma_rows {
                    let to = chroma_at + y * pitch;
                    for x in 0..cw.min(pitch / 2) {
                        buf[to + 2 * x] = pic.data[cr + y * cw + x];
                        buf[to + 2 * x + 1] = pic.data[cb + y * cw + x];
                    }
                }
            }
            PictureFormat::Rgba => unreachable!("handled above"),
        }
    }

    // Every byte of the composed buffer is a byte of the surface: one write, and the buffer
    // can be handed on as the texture's contents.
    if pitch == w && rows == height && chroma_rows == height / 2 && cw * 2 == pitch {
        ctx.write_bytes(dest, &buf);
        return Some(Arc::from(buf));
    }
    for y in 0..rows {
        ctx.write_bytes(dest + (y * pitch) as u32, &buf[y * pitch..y * pitch + w]);
    }
    let chroma_base = dest + chroma_at as u32;
    for y in 0..chroma_rows {
        let at = chroma_at + y * pitch;
        ctx.write_bytes(chroma_base + (y * pitch) as u32, &buf[at..at + cw.min(pitch / 2) * 2]);
    }
    None
}

/// Packed RGBA straight to NV12, in ONE pass, into the composed surface.
///
/// The route that made this worth writing: a phone's browser hands frames back as RGBA, and
/// the picture then has to reach the guest as 4:2:0. Going through I420 meant converting
/// every pixel and then reading the result back to interleave it - two passes and a
/// picture-sized allocation, per frame. This produces the guest's bytes directly.
///
/// The matrix is shared with [`rgb_to_luma`] and [`rgb_to_chroma`] so the two routes into
/// 4:2:0 cannot drift into two slightly different pictures.
fn lay_rgba_as_nv12(
    buf: &mut [u8],
    pitch: usize,
    chroma_at: usize,
    rows: usize,
    chroma_rows: usize,
    pic: &DecodedPicture,
) {
    let w = pic.width as usize;
    let cw = w.div_ceil(2);
    let src = &pic.data;
    // >>> ROW SLICES, NOT PER-TEXEL ADDRESSING, and that is not a micro-optimisation here.
    //
    // This is the ONE path a PowerVR phone takes for every picture of every movie and the
    // one no desktop run exercises (this machine's browser and its Media Foundation both
    // hand back 4:2:0, where that device's browser converts to RGB first). MEASURED by
    // `rgba_conversion_cost`: the per-texel version cost 0.90 ms a picture natively, and a
    // phone's wasm is several times slower than that, thirty times a second.
    //
    // The arithmetic is unchanged - `rgb_to_luma` and `rgb_to_chroma` are still the only
    // two conversions in the engine, and `composed_surface_matches_the_row_by_row_reference`
    // compares this against the reference that calls them - so what is gone is the
    // ADDRESSING: a multiply and a clamp per texel, and a second full pass over the
    // picture for chroma.
    for y in 0..rows {
        let row = &src[y * w * 4..y * w * 4 + w * 4];
        let out = &mut buf[y * pitch..y * pitch + w];
        for (px, o) in row.chunks_exact(4).zip(out.iter_mut()) {
            *o = rgb_to_luma((px[0] as i32, px[1] as i32, px[2] as i32));
        }
    }
    for cy in 0..chroma_rows {
        // The bottom row of a 2x2 group is clamped to the last row the picture has, which
        // is what the per-texel version did through `rgb_to_chroma`.
        let y0 = cy * 2;
        let y1 = (cy * 2 + 1).min(rows.saturating_sub(1));
        let (r0, r1) = (
            &src[y0 * w * 4..y0 * w * 4 + w * 4],
            &src[y1 * w * 4..y1 * w * 4 + w * 4],
        );
        let out = &mut buf[chroma_at + cy * pitch..chroma_at + cy * pitch + cw * 2];
        for (cx, pair) in out.chunks_exact_mut(2).enumerate() {
            let x0 = cx * 2 * 4;
            let x1 = (cx * 2 + 1).min(w - 1) * 4;
            let mut sum = [0i32; 3];
            for (c, s) in sum.iter_mut().enumerate() {
                *s = r0[x0 + c] as i32 + r0[x1 + c] as i32 + r1[x0 + c] as i32 + r1[x1 + c] as i32;
            }
            let (r, g, b) = (sum[0] / 4, sum[1] / 4, sum[2] / 4);
            let cb = (-9713 * r - 19070 * g + 28784 * b + (128 << 16) + 32768) >> 16;
            let cr = (28784 * r - 24103 * g - 4681 * b + (128 << 16) + 32768) >> 16;
            // V FIRST - see `PACKED_RASTER_IS_VU`.
            pair[0] = cr.clamp(0, 255) as u8;
            pair[1] = cb.clamp(0, 255) as u8;
        }
    }
}

/// One RGBA texel of a packed picture, clamped to the last column.
fn rgba_at(pic: &DecodedPicture, x: usize, y: usize) -> (i32, i32, i32) {
    let w = pic.width as usize;
    let at = (y * w + x.min(w - 1)) * 4;
    (pic.data[at] as i32, pic.data[at + 1] as i32, pic.data[at + 2] as i32)
}

/// BT.601 studio-swing luma, in 16.16 - the inverse of the conversion the display path
/// applies, so a picture that makes the round trip comes back where it started.
fn rgb_to_luma((r, g, b): (i32, i32, i32)) -> u8 {
    ((16589 * r + 32558 * g + 6321 * b + (16 << 16) + 32768) >> 16).clamp(0, 255) as u8
}

/// The chroma pair for one 2x2 group, BOX-FILTERED: the samples being dropped are real, and
/// averaging them is what the subsampling this frame came from would have done.
fn rgb_to_chroma(pic: &DecodedPicture, cx: usize, cy: usize, rows: usize) -> (u8, u8) {
    let mut sum = (0i32, 0i32, 0i32);
    for (dx, dy) in [(0usize, 0usize), (1, 0), (0, 1), (1, 1)] {
        let sy = (cy * 2 + dy).min(rows.saturating_sub(1));
        let (r, g, b) = rgba_at(pic, cx * 2 + dx, sy);
        sum = (sum.0 + r, sum.1 + g, sum.2 + b);
    }
    let (r, g, b) = (sum.0 / 4, sum.1 / 4, sum.2 / 4);
    let cb = (-9713 * r - 19070 * g + 28784 * b + (128 << 16) + 32768) >> 16;
    let cr = (28784 * r - 24103 * g - 4681 * b + (128 << 16) + 32768) >> 16;
    (cb.clamp(0, 255) as u8, cr.clamp(0, 255) as u8)
}

/// Three separate planes: luma at `dest`, then Cb and Cr from `chroma` on.
fn write_i420(
    ctx: &mut GuestCtx,
    dest: u32,
    chroma: u32,
    pitch: u32,
    height: u32,
    pic: &DecodedPicture,
) {
    let (w, h) = (pic.width as usize, pic.height as usize);
    let pitch = pitch as usize;
    let (cw, ch) = (w.div_ceil(2), h.div_ceil(2));
    // Only a three-plane source maps straight across. No title seen yet asks the decoder for
    // this format at all, so the others route through one shared re-lay rather than growing
    // a second copy of the same arithmetic.
    let owned;
    let src = if pic.format == PictureFormat::I420 {
        pic
    } else {
        owned = to_i420(pic);
        &owned
    };
    for y in 0..(height as usize).min(h) {
        let at = y * w;
        ctx.write_bytes(dest + (y * pitch) as u32, &src.data[at..at + w]);
    }
    let (cb, cr) = (src.chroma_offset(), src.cr_offset());
    let plane = (pitch / 2 * ch) as u32;
    for y in 0..ch {
        ctx.write_bytes(chroma + (y * pitch / 2) as u32, &src.data[cb + y * cw..cb + y * cw + cw]);
        ctx.write_bytes(
            chroma + plane + (y * pitch / 2) as u32,
            &src.data[cr + y * cw..cr + y * cw + cw],
        );
    }
}

/// Re-lay a picture as three separate planes.
fn to_i420(pic: &DecodedPicture) -> DecodedPicture {
    let (w, h) = (pic.width as usize, pic.height as usize);
    let (cw, ch) = (w.div_ceil(2), h.div_ceil(2));
    let mut data = vec![0u8; w * h + 2 * cw * ch];
    match pic.format {
        PictureFormat::I420 => data.copy_from_slice(&pic.data),
        PictureFormat::Nv12 => {
            data[..w * h].copy_from_slice(&pic.data[..w * h]);
            let base = pic.chroma_offset();
            for y in 0..ch {
                for x in 0..cw {
                    data[w * h + y * cw + x] = pic.data[base + y * cw * 2 + x * 2];
                    data[w * h + cw * ch + y * cw + x] = pic.data[base + y * cw * 2 + x * 2 + 1];
                }
            }
        }
        PictureFormat::Rgba => {
            for y in 0..h {
                for x in 0..w {
                    data[y * w + x] = rgb_to_luma(rgba_at(pic, x, y));
                }
            }
            for cy in 0..ch {
                for cx in 0..cw {
                    let (cb, cr) = rgb_to_chroma(pic, cx, cy, h);
                    data[w * h + cy * cw + cx] = cb;
                    data[w * h + cw * ch + cy * cw + cx] = cr;
                }
            }
        }
    }
    DecodedPicture {
        width: pic.width,
        height: pic.height,
        pts: pic.pts,
        format: PictureFormat::I420,
        data,
    }
}

// ---------------------------------------------------------------------------
// The `#[hostcall]` entry points.
//
// The macro wraps a body in the generated marshalling function, so a `return` inside
// one would return from THAT and write nothing. Every handler above is therefore a
// plain function free to use guard clauses, and these are the thin dispatch shims.
// ---------------------------------------------------------------------------

#[hostcall]
pub(super) fn codec_engine_open_unmap_mem_block(
    _ctx: &mut GuestCtx,
    st: &mut VitaState,
    block: Ptr,
    size: u32,
) -> i32 {
    do_codec_engine_open_unmap_mem_block(_ctx, st, block, size)
}

#[hostcall]
pub(super) fn codec_engine_alloc_memory_from_unmap_mem_block(
    _ctx: &mut GuestCtx,
    st: &mut VitaState,
    uid: i32,
    size: u32,
    alignment: u32,
) -> i32 {
    do_codec_engine_alloc_memory_from_unmap_mem_block(_ctx, st, uid, size, alignment)
}

#[hostcall]
pub(super) fn codec_engine_free_memory_from_unmap_mem_block(
    _ctx: &mut GuestCtx,
    st: &mut VitaState,
    uid: i32,
    _va: Ptr,
) -> i32 {
    do_codec_engine_free_memory_from_unmap_mem_block(_ctx, st, uid, _va)
}

#[hostcall]
pub(super) fn codec_engine_close_unmap_mem_block(
    _ctx: &mut GuestCtx,
    st: &mut VitaState,
    uid: i32,
) -> i32 {
    do_codec_engine_close_unmap_mem_block(_ctx, st, uid)
}

#[hostcall]
pub(super) fn videodec_query_mem_size(
    ctx: &mut GuestCtx,
    st: &mut VitaState,
    codec: u32,
    init_info: Ptr,
    out_size: Ptr,
) -> i32 {
    do_videodec_query_mem_size(ctx, st, codec, init_info, out_size)
}

#[hostcall]
pub(super) fn videodec_init_library_with_unmap_mem(
    ctx: &mut GuestCtx,
    st: &mut VitaState,
    codec: u32,
    mem: Ptr,
    init_info: Ptr,
    mem_size: u32,
) -> i32 {
    do_videodec_init_library_with_unmap_mem(ctx, st, codec, mem, init_info, mem_size)
}

#[hostcall]
pub(super) fn avcdec_query_decoder_mem_size(
    ctx: &mut GuestCtx,
    st: &mut VitaState,
    codec: u32,
    query: Ptr,
    decoder_info: Ptr,
) -> i32 {
    do_avcdec_query_decoder_mem_size(ctx, st, codec, query, decoder_info)
}

#[hostcall]
pub(super) fn avcdec_create_decoder(
    ctx: &mut GuestCtx,
    st: &mut VitaState,
    codec: u32,
    decoder: Ptr,
    query: Ptr,
) -> i32 {
    do_avcdec_create_decoder(ctx, st, codec, decoder, query)
}

#[hostcall]
pub(super) fn avcdec_decode_call(
    ctx: &mut GuestCtx,
    st: &mut VitaState,
    decoder: Ptr,
    au_ptr: Ptr,
    array: Ptr,
) -> i32 {
    do_avcdec_decode(ctx, st, decoder, au_ptr, array)
}

/// `sceAvcdecDecode`, as the SCHEDULER sees it.
///
/// # Why this call parks the thread
///
/// On hardware a decode takes real time - a millisecond or two for a 960x544 picture -
/// and the calling thread is descheduled for it. Modelling that as an instant call was
/// wrong in the ordinary way (the guest's clock does not advance for work that happened)
/// and catastrophically wrong in ONE environment: a browser.
///
/// There the decoder is not in this process. It answers on a callback the JavaScript event
/// loop delivers, and the event loop only runs when the worker returns to it. A movie
/// thread that submits an access unit, is told "no picture yet", and immediately submits
/// the next one never returns - so the frame never ends, the event loop never runs, and
/// the pictures it is waiting for cannot arrive. MEASURED on this desktop through the page
/// that ships: 120 access units submitted in 29 ms, the first picture 4.8 SECONDS later,
/// and about ten thousand lines of the guest spinning in between. On a phone, which is
/// slower, it never got out at all - the run sat at one frame reporting no rate.
///
/// So a decode that hands nothing back, from a decoder that still owes pictures, parks the
/// caller. That is what the hardware does, and it is what gives an asynchronous decoder
/// the one thing it needs: a moment when this engine is not running.
pub(super) fn avcdec_decode(ctx: &mut GuestCtx, st: &mut VitaState) -> SvcOutcome {
    let (decoder, au_ptr, array) = (Ptr(ctx.arg(0)), Ptr(ctx.arg(1)), Ptr(ctx.arg(2)));
    let status = do_avcdec_decode(ctx, st, decoder, au_ptr, array);
    ctx.ret(status as u32);
    if status < 0 || !st.is_preemptive() {
        return SvcOutcome::Continue;
    }
    let handle = ctx.read_u32(decoder.addr() + ctrl::HANDLE);
    let owed = st
        .avcdec
        .session_mut(handle)
        .map(|s| s.ready.is_empty() && s.decoder.owes_frames())
        .unwrap_or(false);
    if !owed {
        return SvcOutcome::Continue;
    }
    st.sleep_park(DECODE_PARK_US);
    SvcOutcome::Block
}

/// How long a picture-less decode parks its caller.
///
/// It is the order of a real hardware decode of a standard-definition picture, so it is
/// what the guest's clock should be charged either way. It is also, in a browser, the
/// interval at which the decoder gets to answer - short enough that a decoder keeping up
/// costs a movie nothing, long enough that a thread waiting on one is not spinning.
const DECODE_PARK_US: u64 = 2_000;

#[hostcall]
pub(super) fn avcdec_decode_stop(
    ctx: &mut GuestCtx,
    st: &mut VitaState,
    decoder: Ptr,
    array: Ptr,
) -> i32 {
    do_avcdec_decode_stop(ctx, st, decoder, array)
}

#[hostcall]
pub(super) fn videodec_term_library(
    _ctx: &mut GuestCtx,
    st: &mut VitaState,
    codec: u32,
) -> i32 {
    do_videodec_term_library(_ctx, st, codec)
}

#[hostcall]
pub(super) fn avcdec_delete_decoder(
    ctx: &mut GuestCtx,
    st: &mut VitaState,
    decoder: Ptr,
) -> i32 {
    do_avcdec_delete_decoder(ctx, st, decoder)
}

#[hostcall]
pub(super) fn avcdec_decode_flush(
    ctx: &mut GuestCtx,
    st: &mut VitaState,
    decoder: Ptr,
) -> i32 {
    do_avcdec_decode_flush(ctx, st, decoder)
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::host::SliceMemory;

    /// The reference NV12 surface, written the obvious way: one row at a time, straight out of
    /// the source picture, with the bytes between the rows left alone. [`write_nv12`] composes
    /// the whole surface in one buffer instead, and this is what says the two agree.
    fn reference_nv12(pic: &DecodedPicture, pitch: usize, height: usize, mem: &mut [u8]) {
        let (w, h) = (pic.width as usize, pic.height as usize);
        let (cw, ch) = (w.div_ceil(2), h.div_ceil(2));
        let rows = height.min(h);
        let chroma_rows = ch.min(height / 2);
        let chroma_base = pitch * height;
        for y in 0..rows {
            for x in 0..w {
                mem[y * pitch + x] = match pic.format {
                    PictureFormat::Rgba => rgb_to_luma(rgba_at(pic, x, y)),
                    _ => pic.data[y * w + x],
                };
            }
        }
        for cy in 0..chroma_rows {
            for cx in 0..cw {
                let (cb, cr) = match pic.format {
                    PictureFormat::Nv12 => {
                        let base = pic.chroma_offset() + cy * cw * 2 + cx * 2;
                        (pic.data[base], pic.data[base + 1])
                    }
                    PictureFormat::I420 => (
                        pic.data[pic.chroma_offset() + cy * cw + cx],
                        pic.data[pic.cr_offset() + cy * cw + cx],
                    ),
                    PictureFormat::Rgba => rgb_to_chroma(pic, cx, cy, rows),
                };
                // V THEN U, the layout the guest's own texture declares - see
                // `PACKED_RASTER_IS_VU`. Written out here rather than shared with the code
                // under test, which is the whole point of a reference.
                if cx * 2 + 1 < pitch {
                    mem[chroma_base + cy * pitch + cx * 2] = cr;
                    mem[chroma_base + cy * pitch + cx * 2 + 1] = cb;
                }
            }
        }
    }

    fn picture(format: PictureFormat, w: u32, h: u32) -> DecodedPicture {
        let (cw, ch) = (w.div_ceil(2) as usize, h.div_ceil(2) as usize);
        let len = match format {
            PictureFormat::Rgba => (w * h * 4) as usize,
            _ => (w * h) as usize + 2 * cw * ch,
        };
        // Deterministic, and not a gradient in one direction only: a plane written at the
        // wrong offset has to change the bytes, so neighbouring rows must differ.
        let data = (0..len).map(|i| ((i * 37 + i / 7 * 11) % 251) as u8).collect();
        DecodedPicture { width: w, height: h, pts: 0, format, data }
    }

    fn run(format: PictureFormat, w: u32, h: u32, pitch: u32, height: u32) -> (Vec<u8>, Option<Arc<[u8]>>) {
        let pic = picture(format, w, h);
        let bytes = (pitch as usize * height as usize * 3 / 2) + pitch as usize;
        let mut mem = vec![0xa5u8; bytes];
        let mut regs = [0u32; vitaslop_transpiler::abi::REG_COUNT];
        let mut vfp = [0u32; crate::host::VFP_ARG_COUNT];
        let authored = {
            let mut slice = SliceMemory(&mut mem);
            let mut ctx = crate::host::GuestCtx::new(&mut regs, &mut vfp, &mut slice, 0);
            write_nv12(&mut ctx, 0, pitch, height, &pic)
        };
        (mem, authored)
    }

    /// Every source layout, at the guest's own pitch, is laid out where the obvious writer
    /// would have put it - and the composed buffer IS the surface, so it can be handed to the
    /// texture cache.
    #[test]
    fn composed_surface_matches_the_row_by_row_reference() {
        for format in [PictureFormat::Nv12, PictureFormat::I420, PictureFormat::Rgba] {
            let (w, h, pitch, height) = (64u32, 32u32, 64u32, 32u32);
            let (mem, authored) = run(format, w, h, pitch, height);
            let mut want = vec![0xa5u8; mem.len()];
            reference_nv12(&picture(format, w, h), pitch as usize, height as usize, &mut want);
            assert_eq!(mem, want, "{format:?} surface");
            let authored = authored.expect("a full surface is authored");
            assert_eq!(
                &authored[..],
                &mem[..pitch as usize * height as usize * 3 / 2],
                "{format:?}: the authored bytes must BE guest memory, or a draw is served a lie"
            );
        }
    }

    /// A surface with padding columns is written row by row, the padding is left as the guest
    /// had it, and nothing is authored - the composed buffer is not the surface there.
    #[test]
    fn padded_surface_keeps_its_padding_and_is_not_authored() {
        let (w, h, pitch, height) = (60u32, 32u32, 64u32, 32u32);
        let (mem, authored) = run(PictureFormat::Nv12, w, h, pitch, height);
        assert!(authored.is_none(), "a padded surface is not the buffer we composed");
        let mut want = vec![0xa5u8; mem.len()];
        reference_nv12(&picture(PictureFormat::Nv12, w, h), pitch as usize, height as usize, &mut want);
        assert_eq!(mem, want);
        assert_eq!(mem[60], 0xa5, "the padding column is the guest's, not ours");
    }

    /// >>> HOW LONG THE RGBA ROUTE ACTUALLY TAKES, because it is the one route no desktop
    /// run exercises and the one a PowerVR phone takes for every single picture.
    ///
    /// Chrome on that device hands `VideoFrame`s back as packed RGBA (no H.264 decoder
    /// produces RGB - the browser converts first), so the engine converts every pixel back
    /// to 4:2:0 to put it in the guest's buffer. This desktop gets NV12 and the whole path
    /// is a row copy, so the cost has never appeared in a measurement here.
    ///
    /// `cargo test --release -p vitaslop-runtime -- --ignored --nocapture rgba_conversion`
    #[test]
    #[ignore = "a timing, not an assertion"]
    fn rgba_conversion_cost() {
        let pic = picture(PictureFormat::Rgba, 960, 544);
        let (pitch, height) = (960usize, 544usize);
        let chroma_at = pitch * height;
        let mut buf = vec![0u8; chroma_at + pitch * (height / 2)];
        let rounds = 50;
        let start = std::time::Instant::now();
        for _ in 0..rounds {
            lay_rgba_as_nv12(&mut buf, pitch, chroma_at, height, height / 2, &pic);
        }
        let each = start.elapsed().as_secs_f64() * 1000.0 / rounds as f64;
        println!("RGBA -> NV12, 960x544: {each:.2} ms per picture on this machine");
        // A phone's wasm is several times slower than this and has to do it 30 times a
        // second, so a number here that looks small is not evidence that it is.
        assert!(each < 100.0, "a picture conversion should not take {each} ms");
    }

    /// A picture shorter than the surface fills what it reaches and no more.
    #[test]
    fn short_picture_does_not_write_past_itself() {
        let (mem, authored) = run(PictureFormat::Nv12, 64, 16, 64, 32);
        assert!(authored.is_none(), "a picture that does not fill the surface is not authored");
        assert_eq!(mem[64 * 16], 0xa5, "the row after the picture is untouched");
    }
}
