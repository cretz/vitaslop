//! macOS: Video Toolbox.
//!
//! `VTDecompressionSession` is the userland face of the same fixed-function decoder the
//! whole system uses, on Intel and on Apple silicon alike. It is fed `CMSampleBuffer`s of
//! LENGTH-PREFIXED NALs (Annex B is not accepted: the parameter sets go in separately, as a
//! `CMVideoFormatDescription`), and it hands back `CVPixelBuffer`s on a callback.
//!
//! Two things about it shape this file. The callback runs on Video Toolbox's own thread, so
//! everything it touches is behind a mutex. And frames come out in DECODE order unless the
//! session is asked to buffer, which only moves the reordering delay somewhere less
//! visible - so this backend declares [`OutputOrder::Decode`] and lets the common layer's
//! reorderer, which knows the stream's own `max_num_reorder_frames`, do it exactly.
//!
//! The bindings are hand-written C FFI. Video Toolbox, Core Media and Core Video are plain
//! C APIs, so an Objective-C runtime crate would buy nothing here.

use std::ffi::c_void;
use std::sync::{Arc, Mutex};

use super::{Backend, FramePool, OutputOrder, StreamConfig};
use crate::bitstream::AccessUnit;
use crate::bitstream::avcc;
use crate::error::{Error, Result};
use crate::frame::{Frame, PixelFormat};

type OsStatus = i32;
type CfAllocatorRef = *const c_void;
type CfTypeRef = *const c_void;
type CfStringRef = *const c_void;
type CfDictionaryRef = *const c_void;
type CfNumberRef = *const c_void;
type CmFormatDescriptionRef = *const c_void;
type CmBlockBufferRef = *const c_void;
type CmSampleBufferRef = *const c_void;
type CvImageBufferRef = *const c_void;
type VtDecompressionSessionRef = *const c_void;

/// `CMTime`. Laid out exactly as Core Media declares it; passed and returned by value.
#[repr(C)]
#[derive(Debug, Clone, Copy)]
struct CmTime {
    value: i64,
    timescale: i32,
    flags: u32,
    epoch: i64,
}

impl CmTime {
    /// `kCMTimeFlags_Valid`.
    const VALID: u32 = 1;

    fn new(value: i64, timescale: i32) -> CmTime {
        CmTime { value, timescale, flags: CmTime::VALID, epoch: 0 }
    }

    fn invalid() -> CmTime {
        CmTime { value: 0, timescale: 0, flags: 0, epoch: 0 }
    }
}

/// `CMSampleTimingInfo`.
#[repr(C)]
#[derive(Debug, Clone, Copy)]
struct CmSampleTimingInfo {
    duration: CmTime,
    presentation_time_stamp: CmTime,
    decode_time_stamp: CmTime,
}

/// `VTDecompressionOutputCallbackRecord`.
#[repr(C)]
struct VtOutputCallbackRecord {
    callback: Option<
        unsafe extern "C" fn(
            *mut c_void,
            *mut c_void,
            OsStatus,
            u32,
            CvImageBufferRef,
            CmTime,
            CmTime,
        ),
    >,
    refcon: *mut c_void,
}

#[link(name = "CoreFoundation", kind = "framework")]
unsafe extern "C" {
    fn CFRelease(cf: CfTypeRef);
    fn CFNumberCreate(allocator: CfAllocatorRef, the_type: i32, value_ptr: *const c_void) -> CfNumberRef;
    fn CFDictionaryCreate(
        allocator: CfAllocatorRef,
        keys: *const *const c_void,
        values: *const *const c_void,
        num_values: isize,
        key_callbacks: *const c_void,
        value_callbacks: *const c_void,
    ) -> CfDictionaryRef;
    static kCFTypeDictionaryKeyCallBacks: c_void;
    static kCFTypeDictionaryValueCallBacks: c_void;
}

#[link(name = "CoreMedia", kind = "framework")]
unsafe extern "C" {
    fn CMVideoFormatDescriptionCreateFromH264ParameterSets(
        allocator: CfAllocatorRef,
        parameter_set_count: usize,
        parameter_set_pointers: *const *const u8,
        parameter_set_sizes: *const usize,
        nal_unit_header_length: i32,
        format_description_out: *mut CmFormatDescriptionRef,
    ) -> OsStatus;
    fn CMBlockBufferCreateWithMemoryBlock(
        structure_allocator: CfAllocatorRef,
        memory_block: *mut c_void,
        block_length: usize,
        block_allocator: CfAllocatorRef,
        custom_block_source: *const c_void,
        offset_to_data: usize,
        data_length: usize,
        flags: u32,
        block_buffer_out: *mut CmBlockBufferRef,
    ) -> OsStatus;
    fn CMBlockBufferReplaceDataBytes(
        source_bytes: *const c_void,
        destination_buffer: CmBlockBufferRef,
        offset_into_destination: usize,
        data_length: usize,
    ) -> OsStatus;
    fn CMSampleBufferCreateReady(
        allocator: CfAllocatorRef,
        data_buffer: CmBlockBufferRef,
        format_description: CmFormatDescriptionRef,
        num_samples: isize,
        num_sample_timing_entries: isize,
        sample_timing_array: *const CmSampleTimingInfo,
        num_sample_size_entries: isize,
        sample_size_array: *const usize,
        sample_buffer_out: *mut CmSampleBufferRef,
    ) -> OsStatus;
}

#[link(name = "CoreVideo", kind = "framework")]
unsafe extern "C" {
    static kCVPixelBufferPixelFormatTypeKey: CfStringRef;
    fn CVPixelBufferLockBaseAddress(pixel_buffer: CvImageBufferRef, lock_flags: u64) -> i32;
    fn CVPixelBufferUnlockBaseAddress(pixel_buffer: CvImageBufferRef, unlock_flags: u64) -> i32;
    fn CVPixelBufferGetPixelFormatType(pixel_buffer: CvImageBufferRef) -> u32;
    fn CVPixelBufferGetWidth(pixel_buffer: CvImageBufferRef) -> usize;
    fn CVPixelBufferGetHeight(pixel_buffer: CvImageBufferRef) -> usize;
    fn CVPixelBufferGetPlaneCount(pixel_buffer: CvImageBufferRef) -> usize;
    fn CVPixelBufferGetBaseAddressOfPlane(pixel_buffer: CvImageBufferRef, plane: usize) -> *mut u8;
    fn CVPixelBufferGetBytesPerRowOfPlane(pixel_buffer: CvImageBufferRef, plane: usize) -> usize;
    fn CVPixelBufferGetHeightOfPlane(pixel_buffer: CvImageBufferRef, plane: usize) -> usize;
    fn CVPixelBufferGetWidthOfPlane(pixel_buffer: CvImageBufferRef, plane: usize) -> usize;
}

#[link(name = "VideoToolbox", kind = "framework")]
unsafe extern "C" {
    fn VTDecompressionSessionCreate(
        allocator: CfAllocatorRef,
        video_format_description: CmFormatDescriptionRef,
        video_decoder_specification: CfDictionaryRef,
        destination_image_buffer_attributes: CfDictionaryRef,
        output_callback: *const VtOutputCallbackRecord,
        decompression_session_out: *mut VtDecompressionSessionRef,
    ) -> OsStatus;
    fn VTDecompressionSessionDecodeFrame(
        session: VtDecompressionSessionRef,
        sample_buffer: CmSampleBufferRef,
        decode_flags: u32,
        source_frame_refcon: *mut c_void,
        info_flags_out: *mut u32,
    ) -> OsStatus;
    fn VTDecompressionSessionWaitForAsynchronousFrames(session: VtDecompressionSessionRef) -> OsStatus;
    fn VTDecompressionSessionFinishDelayedFrames(session: VtDecompressionSessionRef) -> OsStatus;
    fn VTDecompressionSessionInvalidate(session: VtDecompressionSessionRef);
    fn VTDecompressionSessionCanAcceptFormatDescription(
        session: VtDecompressionSessionRef,
        new_format: CmFormatDescriptionRef,
    ) -> u8;
}

/// `kCVPixelFormatType_420YpCbCr8BiPlanarVideoRange` - NV12, studio range.
const PIXEL_FORMAT_NV12_VIDEO: u32 = 0x34323076; // '420v'
/// `kCVPixelFormatType_420YpCbCr8BiPlanarFullRange`.
const PIXEL_FORMAT_NV12_FULL: u32 = 0x34323066; // '420f'
/// `kCFNumberSInt32Type`.
const CF_NUMBER_SINT32: i32 = 3;
/// `kCVPixelBufferLock_ReadOnly`.
const CV_LOCK_READ_ONLY: u64 = 1;
/// `kVTDecodeFrame_EnableAsynchronousDecompression`.
const VT_DECODE_ASYNC: u32 = 1 << 0;

/// What the output callback and the decoding thread share.
#[derive(Default)]
struct Shared {
    ready: Vec<Frame>,
    error: Option<(OsStatus, &'static str)>,
    pool: Vec<Vec<u8>>,
    /// Visible size, so the callback can crop the pixel buffer to it.
    visible: (u32, u32),
}

/// The Video Toolbox backend.
pub struct VideoToolboxBackend {
    session: VtDecompressionSessionRef,
    format: CmFormatDescriptionRef,
    shared: Arc<Mutex<Shared>>,
    /// The `Arc` reference handed to the session as its callback refcon. Owned here so it
    /// is released with the session and not one moment earlier: a callback that outlived
    /// its shared state would write through a dangling pointer.
    refcon: *mut c_void,
    scratch: Vec<u8>,
}

// SAFETY: the raw pointers held here are Core Foundation objects, which are documented as
// safe to use from any one thread at a time; this type is never shared between threads
// without an exclusive borrow, and the only concurrent access - the output callback - goes
// through `shared`'s mutex.
unsafe impl Send for VideoToolboxBackend {}

impl VideoToolboxBackend {
    /// Create a backend with no session yet: the session needs the stream's parameter sets,
    /// which arrive with the first access unit.
    pub fn new() -> Result<VideoToolboxBackend> {
        Ok(VideoToolboxBackend {
            session: std::ptr::null(),
            format: std::ptr::null(),
            shared: Arc::new(Mutex::new(Shared::default())),
            refcon: std::ptr::null_mut(),
            scratch: Vec::new(),
        })
    }

    fn tear_down(&mut self) {
        // SAFETY: both handles were created by this backend and are released exactly once.
        unsafe {
            if !self.session.is_null() {
                VTDecompressionSessionInvalidate(self.session);
                CFRelease(self.session);
                self.session = std::ptr::null();
            }
            if !self.format.is_null() {
                CFRelease(self.format);
                self.format = std::ptr::null();
            }
            if !self.refcon.is_null() {
                // SAFETY: the session is gone, so no callback can still hold this; the
                // pointer came from `Arc::into_raw`.
                drop(Arc::from_raw(self.refcon as *const Mutex<Shared>));
                self.refcon = std::ptr::null_mut();
            }
        }
    }
}

impl Backend for VideoToolboxBackend {
    fn name(&self) -> &'static str {
        "VideoToolbox"
    }

    fn output_order(&self) -> OutputOrder {
        OutputOrder::Decode
    }

    fn configure(&mut self, config: StreamConfig<'_>) -> Result<()> {
        let avcc = config.avcc;
        if avcc.sps.is_empty() || avcc.pps.is_empty() {
            return Err(Error::bitstream("VideoToolbox needs both an SPS and a PPS"));
        }
        // The parameter sets go in as pointers into the caller's own NAL bytes; Core Media
        // copies them into the format description before the call returns.
        let mut pointers: Vec<*const u8> = Vec::new();
        let mut sizes: Vec<usize> = Vec::new();
        for set in avcc.sps.iter().chain(avcc.pps.iter()) {
            pointers.push(set.as_ptr());
            sizes.push(set.len());
        }

        let mut format: CmFormatDescriptionRef = std::ptr::null();
        // SAFETY: the pointer and size arrays are the same length and outlive the call.
        let status = unsafe {
            CMVideoFormatDescriptionCreateFromH264ParameterSets(
                std::ptr::null(),
                pointers.len(),
                pointers.as_ptr(),
                sizes.as_ptr(),
                4,
                &mut format,
            )
        };
        if status != 0 || format.is_null() {
            return Err(Error::platform(
                "CMVideoFormatDescriptionCreateFromH264ParameterSets",
                status as i64,
                "the parameter sets were rejected",
            ));
        }

        // An existing session that can take the new format keeps running: tearing one down
        // on every parameter set costs a full decoder restart, and streams that repeat
        // their SPS on every keyframe are the common case, not the rare one.
        if !self.session.is_null() {
            // SAFETY: both handles are live.
            let ok = unsafe { VTDecompressionSessionCanAcceptFormatDescription(self.session, format) };
            if ok != 0 {
                // SAFETY: replacing one retained format description with another.
                unsafe { CFRelease(self.format) };
                self.format = format;
                self.shared.lock().unwrap().visible = (config.width, config.height);
                return Ok(());
            }
        }
        self.tear_down();
        self.format = format;

        // Ask for NV12 explicitly. Without this the session picks a format from the
        // decoder's preference, which on some Macs is a biplanar full-range buffer and on
        // others a 4:2:2 one - and a backend that accepts whatever it is handed is how a
        // colour shift ships.
        let pixel_format = PIXEL_FORMAT_NV12_VIDEO as i32;
        // SAFETY: a one-entry CFDictionary of CFString -> CFNumber, released below.
        let attributes = unsafe {
            let number = CFNumberCreate(
                std::ptr::null(),
                CF_NUMBER_SINT32,
                &pixel_format as *const i32 as *const c_void,
            );
            let keys = [kCVPixelBufferPixelFormatTypeKey];
            let values = [number];
            let dict = CFDictionaryCreate(
                std::ptr::null(),
                keys.as_ptr(),
                values.as_ptr(),
                1,
                &kCFTypeDictionaryKeyCallBacks as *const c_void,
                &kCFTypeDictionaryValueCallBacks as *const c_void,
            );
            CFRelease(number);
            dict
        };

        let refcon = Arc::into_raw(self.shared.clone()) as *mut c_void;
        let record = VtOutputCallbackRecord { callback: Some(output_callback), refcon };
        let mut session: VtDecompressionSessionRef = std::ptr::null();
        // SAFETY: the format description and attributes are live for the call, and the
        // callback record is copied by Video Toolbox.
        let status = unsafe {
            VTDecompressionSessionCreate(
                std::ptr::null(),
                self.format,
                std::ptr::null(),
                attributes,
                &record,
                &mut session,
            )
        };
        // SAFETY: the dictionary was created above and the session holds its own reference.
        unsafe { CFRelease(attributes) };
        if status != 0 || session.is_null() {
            // Take the refcon reference back: no session means no callback will ever run.
            // SAFETY: the pointer came from `Arc::into_raw` just above and is unused.
            unsafe { drop(Arc::from_raw(refcon as *const Mutex<Shared>)) };
            return Err(Error::platform(
                "VTDecompressionSessionCreate",
                status as i64,
                "no decoder for this stream",
            ));
        }
        self.session = session;
        self.refcon = refcon;
        self.shared.lock().unwrap().visible = (config.width, config.height);
        Ok(())
    }

    fn send(&mut self, au: &AccessUnit, timestamp: i64) -> Result<()> {
        if self.session.is_null() {
            return Err(Error::State("VideoToolbox session used before it was configured"));
        }
        // Video Toolbox takes length-prefixed NALs; the format description above declared a
        // 4-byte length.
        self.scratch.clear();
        avcc::annex_b_to_length_prefixed(&au.data, 4, &mut self.scratch);

        let mut block: CmBlockBufferRef = std::ptr::null();
        // SAFETY: Core Media allocates the block itself (memory_block = null) and we copy
        // into it immediately after.
        let status = unsafe {
            CMBlockBufferCreateWithMemoryBlock(
                std::ptr::null(),
                std::ptr::null_mut(),
                self.scratch.len(),
                std::ptr::null(),
                std::ptr::null(),
                0,
                self.scratch.len(),
                0,
                &mut block,
            )
        };
        if status != 0 {
            return Err(Error::platform("CMBlockBufferCreateWithMemoryBlock", status as i64, ""));
        }
        // SAFETY: `block` is `self.scratch.len()` bytes long, which is what is copied.
        let status = unsafe {
            CMBlockBufferReplaceDataBytes(
                self.scratch.as_ptr() as *const c_void,
                block,
                0,
                self.scratch.len(),
            )
        };
        if status != 0 {
            // SAFETY: releasing the block created just above.
            unsafe { CFRelease(block) };
            return Err(Error::platform("CMBlockBufferReplaceDataBytes", status as i64, ""));
        }

        let timing = CmSampleTimingInfo {
            duration: CmTime::new(1, 1000),
            presentation_time_stamp: CmTime::new(timestamp, 1000),
            decode_time_stamp: CmTime::invalid(),
        };
        let sizes = [self.scratch.len()];
        let mut sample: CmSampleBufferRef = std::ptr::null();
        // SAFETY: one sample, one timing entry, one size entry - all three arrays match.
        let status = unsafe {
            CMSampleBufferCreateReady(
                std::ptr::null(),
                block,
                self.format,
                1,
                1,
                &timing,
                1,
                sizes.as_ptr(),
                &mut sample,
            )
        };
        // SAFETY: the sample buffer retains the block on success; either way this backend
        // is done with its own reference.
        unsafe { CFRelease(block) };
        if status != 0 {
            return Err(Error::platform("CMSampleBufferCreateReady", status as i64, ""));
        }

        let mut info: u32 = 0;
        // SAFETY: the session and sample are both live; no source refcon is used, so the
        // callback receives null for it.
        let status = unsafe {
            VTDecompressionSessionDecodeFrame(
                self.session,
                sample,
                VT_DECODE_ASYNC,
                std::ptr::null_mut(),
                &mut info,
            )
        };
        // SAFETY: the session retains the sample for as long as it needs it.
        unsafe { CFRelease(sample) };
        if status != 0 {
            return Err(Error::platform(
                "VTDecompressionSessionDecodeFrame",
                status as i64,
                "the decoder rejected an access unit",
            ));
        }
        Ok(())
    }

    fn poll(&mut self, pool: &mut FramePool, out: &mut Vec<Frame>) -> Result<()> {
        let mut shared = self.shared.lock().unwrap();
        while shared.pool.len() < 4 {
            let buf = pool.take();
            if buf.capacity() == 0 {
                break;
            }
            shared.pool.push(buf);
        }
        if let Some((status, call)) = shared.error.take() {
            return Err(Error::platform(call, status as i64, "decode failed"));
        }
        out.append(&mut shared.ready);
        Ok(())
    }

    fn drain(&mut self, pool: &mut FramePool, out: &mut Vec<Frame>) -> Result<()> {
        if !self.session.is_null() {
            // SAFETY: the session is live; both calls block until its queue is empty.
            let status = unsafe {
                let a = VTDecompressionSessionFinishDelayedFrames(self.session);
                let b = VTDecompressionSessionWaitForAsynchronousFrames(self.session);
                if a != 0 { a } else { b }
            };
            if status != 0 {
                return Err(Error::platform(
                    "VTDecompressionSessionWaitForAsynchronousFrames",
                    status as i64,
                    "",
                ));
            }
        }
        self.poll(pool, out)
    }

    fn reset(&mut self) -> Result<()> {
        if !self.session.is_null() {
            // SAFETY: the session is live. Waiting here is what makes the callback stop
            // touching `shared` before the queues below are cleared.
            unsafe { VTDecompressionSessionWaitForAsynchronousFrames(self.session) };
        }
        let mut shared = self.shared.lock().unwrap();
        shared.ready.clear();
        shared.error = None;
        Ok(())
    }
}

impl Drop for VideoToolboxBackend {
    fn drop(&mut self) {
        if !self.session.is_null() {
            // SAFETY: no callback may be in flight once this returns, which is what makes
            // dropping the refcon's Arc reference below safe.
            unsafe { VTDecompressionSessionWaitForAsynchronousFrames(self.session) };
        }
        self.tear_down();
    }
}

/// Video Toolbox's output callback. Runs on Video Toolbox's thread, not the caller's.
///
/// # Safety
///
/// Called by Video Toolbox with the refcon this backend passed to
/// `VTDecompressionSessionCreate`, which is an `Arc<Mutex<Shared>>` that outlives the
/// session.
unsafe extern "C" fn output_callback(
    refcon: *mut c_void,
    _source_frame_refcon: *mut c_void,
    status: OsStatus,
    _info_flags: u32,
    image_buffer: CvImageBufferRef,
    presentation_time: CmTime,
    _presentation_duration: CmTime,
) {
    if refcon.is_null() {
        return;
    }
    // SAFETY: the refcon is the `Arc` pointer handed to the session, still alive because
    // the backend waits for outstanding frames before dropping it. The borrow is not
    // consumed: `ManuallyDrop`-style, the Arc is only reconstructed to be read.
    let shared = unsafe { &*(refcon as *const Mutex<Shared>) };
    let mut shared = match shared.lock() {
        Ok(guard) => guard,
        Err(_) => return,
    };
    if status != 0 {
        if shared.error.is_none() {
            shared.error = Some((status, "VTDecompressionSession output"));
        }
        return;
    }
    if image_buffer.is_null() {
        // A dropped frame: the session reports it with no image. Not an error.
        return;
    }
    match unsafe { copy_pixel_buffer(image_buffer, &mut shared, presentation_time) } {
        Ok(frame) => shared.ready.push(frame),
        Err((status, call)) => {
            if shared.error.is_none() {
                shared.error = Some((status, call));
            }
        }
    }
}

/// Copy the visible region of a `CVPixelBuffer` into a [`Frame`].
///
/// # Safety
///
/// `image_buffer` must be a live `CVPixelBufferRef`.
unsafe fn copy_pixel_buffer(
    image_buffer: CvImageBufferRef,
    shared: &mut Shared,
    time: CmTime,
) -> std::result::Result<Frame, (OsStatus, &'static str)> {
    // SAFETY: the caller guarantees a live pixel buffer; every read below happens between
    // the lock and the unlock.
    unsafe {
        let format = CVPixelBufferGetPixelFormatType(image_buffer);
        if format != PIXEL_FORMAT_NV12_VIDEO && format != PIXEL_FORMAT_NV12_FULL {
            return Err((format as OsStatus, "CVPixelBufferGetPixelFormatType (not NV12)"));
        }
        let status = CVPixelBufferLockBaseAddress(image_buffer, CV_LOCK_READ_ONLY);
        if status != 0 {
            return Err((status, "CVPixelBufferLockBaseAddress"));
        }
        let result = copy_locked(image_buffer, shared, time);
        CVPixelBufferUnlockBaseAddress(image_buffer, CV_LOCK_READ_ONLY);
        result
    }
}

/// The body of [`copy_pixel_buffer`], with the buffer already locked.
///
/// # Safety
///
/// `image_buffer` must be locked for reading.
unsafe fn copy_locked(
    image_buffer: CvImageBufferRef,
    shared: &mut Shared,
    time: CmTime,
) -> std::result::Result<Frame, (OsStatus, &'static str)> {
    // SAFETY: the buffer is locked, so every plane's base address is mapped.
    unsafe {
        if CVPixelBufferGetPlaneCount(image_buffer) < 2 {
            return Err((0, "CVPixelBufferGetPlaneCount (not biplanar)"));
        }
        // The pixel buffer is the CODED size; the visible size came from the SPS. Video
        // Toolbox does carry a clean aperture, but it is advisory and several decoders
        // leave it at the coded size - the SPS crop is the authority.
        let (mut width, mut height) = shared.visible;
        let coded_w = CVPixelBufferGetWidth(image_buffer) as u32;
        let coded_h = CVPixelBufferGetHeight(image_buffer) as u32;
        if width == 0 || height == 0 || width > coded_w || height > coded_h {
            width = coded_w;
            height = coded_h;
        }

        let buf = shared.pool.pop().unwrap_or_default();
        let mut frame = Frame::alloc(PixelFormat::Nv12, width, height, buf);
        frame.pts = if time.timescale > 0 { time.value } else { 0 };

        for plane in 0..2 {
            let src = CVPixelBufferGetBaseAddressOfPlane(image_buffer, plane);
            if src.is_null() {
                return Err((0, "CVPixelBufferGetBaseAddressOfPlane"));
            }
            let src_stride = CVPixelBufferGetBytesPerRowOfPlane(image_buffer, plane);
            let src_rows = CVPixelBufferGetHeightOfPlane(image_buffer, plane);
            let src_row_bytes = CVPixelBufferGetWidthOfPlane(image_buffer, plane)
                * if plane == 0 { 1 } else { 2 };
            let dst_plane = frame.planes[plane];
            if dst_plane.rows > src_rows || dst_plane.row_bytes > src_row_bytes {
                return Err((0, "CVPixelBuffer plane smaller than the visible picture"));
            }
            let src = std::slice::from_raw_parts(src, src_stride * src_rows);
            let dst = frame.plane_mut(plane);
            for r in 0..dst_plane.rows {
                dst[r * dst_plane.stride..r * dst_plane.stride + dst_plane.row_bytes]
                    .copy_from_slice(&src[r * src_stride..r * src_stride + dst_plane.row_bytes]);
            }
        }
        Ok(frame)
    }
}

