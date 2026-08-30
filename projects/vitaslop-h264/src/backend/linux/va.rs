//! libva bindings, loaded at run time.
//!
//! Nothing here is linked. `libva.so.2` and `libva-drm.so.2` are opened with `dlopen(3)`
//! the first time a decoder is created, so a binary built with this crate still starts on
//! a machine that has no VA-API at all - it reports [`crate::Error::NoDecoder`] and lets
//! the caller fall back. Linking libva instead would make the whole program refuse to load.
//!
//! The structures are transcribed from `va.h` and `va_dec_h264.h` (VA-API 1.x, which has
//! been ABI-stable since 2017). Bit-fields are declared as plain `u32` and packed by hand:
//! that is what the C compiler does with them on every little-endian target libva supports,
//! and doing it explicitly means the layout does not depend on how Rust chooses to
//! interpret a bit-field it cannot see.

#![allow(non_snake_case, non_camel_case_types)]

use std::ffi::{CString, c_char, c_int, c_void};

use crate::error::{Error, Result};

/// `VADisplay`.
pub type VADisplay = *mut c_void;
/// `VAStatus`.
pub type VAStatus = c_int;
/// `VAGenericID` and everything aliased to it.
pub type VAGenericID = u32;
/// `VASurfaceID`.
pub type VASurfaceID = VAGenericID;
/// `VAConfigID`.
pub type VAConfigID = VAGenericID;
/// `VAContextID`.
pub type VAContextID = VAGenericID;
/// `VABufferID`.
pub type VABufferID = VAGenericID;
/// `VAImageID`.
pub type VAImageID = VAGenericID;

/// `VA_STATUS_SUCCESS`.
pub const VA_STATUS_SUCCESS: VAStatus = 0;
/// `VA_INVALID_ID`.
pub const VA_INVALID_ID: VAGenericID = 0xffff_ffff;
/// `VA_INVALID_SURFACE`.
pub const VA_INVALID_SURFACE: VASurfaceID = VA_INVALID_ID;
/// `VA_RT_FORMAT_YUV420`.
pub const VA_RT_FORMAT_YUV420: u32 = 0x0000_0001;
/// `VA_PROGRESSIVE`, the context flag for frame-coded content.
pub const VA_PROGRESSIVE: c_int = 0x1;
/// `VA_FOURCC_NV12`.
pub const VA_FOURCC_NV12: u32 = 0x3231_564e;

/// `VAProfile` values for H.264.
pub mod profile {
    use std::ffi::c_int;
    /// `VAProfileH264Main`.
    pub const H264_MAIN: c_int = 6;
    /// `VAProfileH264High`.
    pub const H264_HIGH: c_int = 7;
    /// `VAProfileH264ConstrainedBaseline`.
    pub const H264_CONSTRAINED_BASELINE: c_int = 13;
}

/// `VAEntrypointVLD`: bitstream decoding, which is the only entry point this crate uses.
pub const VA_ENTRYPOINT_VLD: c_int = 1;

/// `VABufferType` values used here.
pub mod buffer_type {
    use std::ffi::c_int;
    /// `VAPictureParameterBufferType`.
    pub const PICTURE_PARAMETER: c_int = 0;
    /// `VAIQMatrixBufferType`.
    pub const IQ_MATRIX: c_int = 1;
    /// `VASliceParameterBufferType`.
    pub const SLICE_PARAMETER: c_int = 4;
    /// `VASliceDataBufferType`.
    pub const SLICE_DATA: c_int = 5;
}

/// Flags on a [`VAPictureH264`].
pub mod picture_flag {
    /// `VA_PICTURE_H264_INVALID`.
    pub const INVALID: u32 = 0x0000_0001;
    /// `VA_PICTURE_H264_TOP_FIELD`.
    pub const TOP_FIELD: u32 = 0x0000_0002;
    /// `VA_PICTURE_H264_BOTTOM_FIELD`.
    pub const BOTTOM_FIELD: u32 = 0x0000_0004;
    /// `VA_PICTURE_H264_SHORT_TERM_REFERENCE`.
    pub const SHORT_TERM_REFERENCE: u32 = 0x0000_0008;
    /// `VA_PICTURE_H264_LONG_TERM_REFERENCE`.
    pub const LONG_TERM_REFERENCE: u32 = 0x0000_0010;
}

/// `VA_SLICE_DATA_FLAG_ALL`: the slice buffer holds a whole slice.
pub const VA_SLICE_DATA_FLAG_ALL: u32 = 0x00;

/// `VAPictureH264`.
#[repr(C)]
#[derive(Debug, Clone, Copy)]
pub struct VAPictureH264 {
    /// Surface holding the picture, or [`VA_INVALID_SURFACE`].
    pub picture_id: VASurfaceID,
    /// `FrameNum` for a short-term reference, `LongTermFrameIdx` for a long-term one.
    pub frame_idx: u32,
    /// See [`picture_flag`].
    pub flags: u32,
    /// `TopFieldOrderCnt`.
    pub TopFieldOrderCnt: i32,
    /// `BottomFieldOrderCnt`.
    pub BottomFieldOrderCnt: i32,
    /// Reserved by libva; zero.
    pub va_reserved: [u32; 4],
}

impl VAPictureH264 {
    /// An empty slot: what an unused reference list entry has to be.
    pub fn invalid() -> VAPictureH264 {
        VAPictureH264 {
            picture_id: VA_INVALID_SURFACE,
            frame_idx: 0,
            flags: picture_flag::INVALID,
            TopFieldOrderCnt: 0,
            BottomFieldOrderCnt: 0,
            va_reserved: [0; 4],
        }
    }
}

/// `VAPictureParameterBufferH264`.
#[repr(C)]
#[derive(Debug, Clone, Copy)]
pub struct VAPictureParameterBufferH264 {
    /// The picture being decoded.
    pub CurrPic: VAPictureH264,
    /// The DPB, unused slots invalid.
    pub ReferenceFrames: [VAPictureH264; 16],
    /// `pic_width_in_mbs_minus1`.
    pub picture_width_in_mbs_minus1: u16,
    /// `pic_height_in_map_units_minus1` for frame pictures.
    pub picture_height_in_mbs_minus1: u16,
    /// `bit_depth_luma_minus8`.
    pub bit_depth_luma_minus8: u8,
    /// `bit_depth_chroma_minus8`.
    pub bit_depth_chroma_minus8: u8,
    /// `max_num_ref_frames`.
    pub num_ref_frames: u8,
    /// Packed `seq_fields`; build with [`SeqFields`].
    pub seq_fields: u32,
    /// Deprecated, kept for layout.
    pub num_slice_groups_minus1: u8,
    /// Deprecated, kept for layout.
    pub slice_group_map_type: u8,
    /// Deprecated, kept for layout.
    pub slice_group_change_rate_minus1: u16,
    /// `pic_init_qp_minus26`.
    pub pic_init_qp_minus26: i8,
    /// `pic_init_qs_minus26`.
    pub pic_init_qs_minus26: i8,
    /// `chroma_qp_index_offset`.
    pub chroma_qp_index_offset: i8,
    /// `second_chroma_qp_index_offset`.
    pub second_chroma_qp_index_offset: i8,
    /// Packed `pic_fields`; build with [`PicFields`].
    pub pic_fields: u32,
    /// `frame_num`.
    pub frame_num: u16,
    /// Reserved by libva; zero.
    pub va_reserved: [u32; 8],
}

/// Builder for `VAPictureParameterBufferH264::seq_fields`, in `va_dec_h264.h` bit order.
#[derive(Debug, Default, Clone, Copy)]
pub struct SeqFields {
    /// `chroma_format_idc`, 2 bits.
    pub chroma_format_idc: u32,
    /// `residual_colour_transform_flag`.
    pub residual_colour_transform_flag: bool,
    /// `gaps_in_frame_num_value_allowed_flag`.
    pub gaps_in_frame_num_value_allowed_flag: bool,
    /// `frame_mbs_only_flag`.
    pub frame_mbs_only_flag: bool,
    /// `mb_adaptive_frame_field_flag`.
    pub mb_adaptive_frame_field_flag: bool,
    /// `direct_8x8_inference_flag`.
    pub direct_8x8_inference_flag: bool,
    /// `MinLumaBiPredSize8x8`, derived from the level.
    pub min_luma_bipred_size8x8: bool,
    /// `log2_max_frame_num_minus4`, 4 bits.
    pub log2_max_frame_num_minus4: u32,
    /// `pic_order_cnt_type`, 2 bits.
    pub pic_order_cnt_type: u32,
    /// `log2_max_pic_order_cnt_lsb_minus4`, 4 bits.
    pub log2_max_pic_order_cnt_lsb_minus4: u32,
    /// `delta_pic_order_always_zero_flag`.
    pub delta_pic_order_always_zero_flag: bool,
}

impl SeqFields {
    /// Pack into the union's `value`.
    pub fn pack(&self) -> u32 {
        let mut v = 0u32;
        let mut at = 0u32;
        let mut put = |value: u32, bits: u32| {
            v |= (value & ((1 << bits) - 1)) << at;
            at += bits;
        };
        put(self.chroma_format_idc, 2);
        put(self.residual_colour_transform_flag as u32, 1);
        put(self.gaps_in_frame_num_value_allowed_flag as u32, 1);
        put(self.frame_mbs_only_flag as u32, 1);
        put(self.mb_adaptive_frame_field_flag as u32, 1);
        put(self.direct_8x8_inference_flag as u32, 1);
        put(self.min_luma_bipred_size8x8 as u32, 1);
        put(self.log2_max_frame_num_minus4, 4);
        put(self.pic_order_cnt_type, 2);
        put(self.log2_max_pic_order_cnt_lsb_minus4, 4);
        put(self.delta_pic_order_always_zero_flag as u32, 1);
        v
    }
}

/// Builder for `VAPictureParameterBufferH264::pic_fields`.
#[derive(Debug, Default, Clone, Copy)]
pub struct PicFields {
    /// `entropy_coding_mode_flag`.
    pub entropy_coding_mode_flag: bool,
    /// `weighted_pred_flag`.
    pub weighted_pred_flag: bool,
    /// `weighted_bipred_idc`, 2 bits.
    pub weighted_bipred_idc: u32,
    /// `transform_8x8_mode_flag`.
    pub transform_8x8_mode_flag: bool,
    /// `field_pic_flag`.
    pub field_pic_flag: bool,
    /// `constrained_intra_pred_flag`.
    pub constrained_intra_pred_flag: bool,
    /// `pic_order_present_flag`, i.e. `bottom_field_pic_order_in_frame_present_flag`.
    pub pic_order_present_flag: bool,
    /// `deblocking_filter_control_present_flag`.
    pub deblocking_filter_control_present_flag: bool,
    /// `redundant_pic_cnt_present_flag`.
    pub redundant_pic_cnt_present_flag: bool,
    /// `reference_pic_flag`: is the current picture a reference picture.
    pub reference_pic_flag: bool,
}

impl PicFields {
    /// Pack into the union's `value`.
    pub fn pack(&self) -> u32 {
        let mut v = 0u32;
        let mut at = 0u32;
        let mut put = |value: u32, bits: u32| {
            v |= (value & ((1 << bits) - 1)) << at;
            at += bits;
        };
        put(self.entropy_coding_mode_flag as u32, 1);
        put(self.weighted_pred_flag as u32, 1);
        put(self.weighted_bipred_idc, 2);
        put(self.transform_8x8_mode_flag as u32, 1);
        put(self.field_pic_flag as u32, 1);
        put(self.constrained_intra_pred_flag as u32, 1);
        put(self.pic_order_present_flag as u32, 1);
        put(self.deblocking_filter_control_present_flag as u32, 1);
        put(self.redundant_pic_cnt_present_flag as u32, 1);
        put(self.reference_pic_flag as u32, 1);
        v
    }
}

/// `VASliceParameterBufferH264`.
#[repr(C)]
#[derive(Debug, Clone, Copy)]
pub struct VASliceParameterBufferH264 {
    /// Bytes of slice data in the data buffer.
    pub slice_data_size: u32,
    /// Offset of this slice inside the data buffer.
    pub slice_data_offset: u32,
    /// [`VA_SLICE_DATA_FLAG_ALL`].
    pub slice_data_flag: u32,
    /// Bit offset, from the start of the NAL, at which slice data begins.
    pub slice_data_bit_offset: u16,
    /// `first_mb_in_slice`.
    pub first_mb_in_slice: u16,
    /// `slice_type` (0 P, 1 B, 2 I).
    pub slice_type: u8,
    /// `direct_spatial_mv_pred_flag`.
    pub direct_spatial_mv_pred_flag: u8,
    /// `num_ref_idx_l0_active_minus1`.
    pub num_ref_idx_l0_active_minus1: u8,
    /// `num_ref_idx_l1_active_minus1`.
    pub num_ref_idx_l1_active_minus1: u8,
    /// `cabac_init_idc`.
    pub cabac_init_idc: u8,
    /// `slice_qp_delta`.
    pub slice_qp_delta: i8,
    /// `disable_deblocking_filter_idc`.
    pub disable_deblocking_filter_idc: u8,
    /// `slice_alpha_c0_offset_div2`.
    pub slice_alpha_c0_offset_div2: i8,
    /// `slice_beta_offset_div2`.
    pub slice_beta_offset_div2: i8,
    /// Reference list 0.
    pub RefPicList0: [VAPictureH264; 32],
    /// Reference list 1.
    pub RefPicList1: [VAPictureH264; 32],
    /// `luma_log2_weight_denom`.
    pub luma_log2_weight_denom: u8,
    /// `chroma_log2_weight_denom`.
    pub chroma_log2_weight_denom: u8,
    /// Whether list 0 luma weights are present.
    pub luma_weight_l0_flag: u8,
    /// List 0 luma weights.
    pub luma_weight_l0: [i16; 32],
    /// List 0 luma offsets.
    pub luma_offset_l0: [i16; 32],
    /// Whether list 0 chroma weights are present.
    pub chroma_weight_l0_flag: u8,
    /// List 0 chroma weights, `[ref][cb, cr]`.
    pub chroma_weight_l0: [[i16; 2]; 32],
    /// List 0 chroma offsets.
    pub chroma_offset_l0: [[i16; 2]; 32],
    /// Whether list 1 luma weights are present.
    pub luma_weight_l1_flag: u8,
    /// List 1 luma weights.
    pub luma_weight_l1: [i16; 32],
    /// List 1 luma offsets.
    pub luma_offset_l1: [i16; 32],
    /// Whether list 1 chroma weights are present.
    pub chroma_weight_l1_flag: u8,
    /// List 1 chroma weights.
    pub chroma_weight_l1: [[i16; 2]; 32],
    /// List 1 chroma offsets.
    pub chroma_offset_l1: [[i16; 2]; 32],
    /// Reserved by libva; zero.
    pub va_reserved: [u32; 4],
}

impl Default for VASliceParameterBufferH264 {
    fn default() -> Self {
        VASliceParameterBufferH264 {
            slice_data_size: 0,
            slice_data_offset: 0,
            slice_data_flag: VA_SLICE_DATA_FLAG_ALL,
            slice_data_bit_offset: 0,
            first_mb_in_slice: 0,
            slice_type: 2,
            direct_spatial_mv_pred_flag: 0,
            num_ref_idx_l0_active_minus1: 0,
            num_ref_idx_l1_active_minus1: 0,
            cabac_init_idc: 0,
            slice_qp_delta: 0,
            disable_deblocking_filter_idc: 0,
            slice_alpha_c0_offset_div2: 0,
            slice_beta_offset_div2: 0,
            RefPicList0: [VAPictureH264::invalid(); 32],
            RefPicList1: [VAPictureH264::invalid(); 32],
            luma_log2_weight_denom: 0,
            chroma_log2_weight_denom: 0,
            luma_weight_l0_flag: 0,
            luma_weight_l0: [0; 32],
            luma_offset_l0: [0; 32],
            chroma_weight_l0_flag: 0,
            chroma_weight_l0: [[0; 2]; 32],
            chroma_offset_l0: [[0; 2]; 32],
            luma_weight_l1_flag: 0,
            luma_weight_l1: [0; 32],
            luma_offset_l1: [0; 32],
            chroma_weight_l1_flag: 0,
            chroma_weight_l1: [[0; 2]; 32],
            chroma_offset_l1: [[0; 2]; 32],
            va_reserved: [0; 4],
        }
    }
}

/// `VAIQMatrixBufferH264`.
#[repr(C)]
#[derive(Debug, Clone, Copy)]
pub struct VAIQMatrixBufferH264 {
    /// Six 4x4 scaling lists.
    pub ScalingList4x4: [[u8; 16]; 6],
    /// Two 8x8 scaling lists.
    pub ScalingList8x8: [[u8; 64]; 2],
    /// Reserved by libva; zero.
    pub va_reserved: [u32; 4],
}

impl Default for VAIQMatrixBufferH264 {
    fn default() -> Self {
        VAIQMatrixBufferH264 {
            ScalingList4x4: [[16; 16]; 6],
            ScalingList8x8: [[16; 64]; 2],
            va_reserved: [0; 4],
        }
    }
}

/// `VAImageFormat`.
#[repr(C)]
#[derive(Debug, Clone, Copy, Default)]
pub struct VAImageFormat {
    /// FourCC of the layout.
    pub fourcc: u32,
    /// Byte order.
    pub byte_order: u32,
    /// Bits per pixel.
    pub bits_per_pixel: u32,
    /// Significant bits.
    pub depth: u32,
    /// RGB masks, unused for YUV.
    pub red_mask: u32,
    /// RGB masks, unused for YUV.
    pub green_mask: u32,
    /// RGB masks, unused for YUV.
    pub blue_mask: u32,
    /// RGB masks, unused for YUV.
    pub alpha_mask: u32,
    /// Reserved by libva; zero.
    pub va_reserved: [u32; 4],
}

/// `VAImage`.
#[repr(C)]
#[derive(Debug, Clone, Copy, Default)]
pub struct VAImage {
    /// Image handle.
    pub image_id: VAImageID,
    /// Pixel layout.
    pub format: VAImageFormat,
    /// Buffer holding the pixels.
    pub buf: VABufferID,
    /// Width in pixels.
    pub width: u16,
    /// Height in pixels.
    pub height: u16,
    /// Bytes in the buffer.
    pub data_size: u32,
    /// Planes in use.
    pub num_planes: u32,
    /// Per-plane stride.
    pub pitches: [u32; 3],
    /// Per-plane byte offset.
    pub offsets: [u32; 3],
    /// Palette entries (unused for YUV).
    pub num_palette_entries: i32,
    /// Palette entry size (unused for YUV).
    pub entry_bytes: i32,
    /// Component order (unused for YUV).
    pub component_order: [i8; 4],
    /// Reserved by libva; zero.
    pub va_reserved: [u32; 4],
}

/// `VAConfigAttrib`.
#[repr(C)]
#[derive(Debug, Clone, Copy, Default)]
pub struct VAConfigAttrib {
    /// Attribute type.
    pub type_: c_int,
    /// Attribute value.
    pub value: u32,
}

/// The libva entry points this crate calls, resolved from the shared objects.
///
/// Every field is the libva function of the same name, with the signature `va.h` declares
/// for it; documenting them one by one would only restate the C header.
#[allow(missing_docs)]
pub struct Va {
    _libva: Library,
    _libva_drm: Library,
    pub vaGetDisplayDRM: unsafe extern "C" fn(c_int) -> VADisplay,
    pub vaInitialize: unsafe extern "C" fn(VADisplay, *mut c_int, *mut c_int) -> VAStatus,
    pub vaTerminate: unsafe extern "C" fn(VADisplay) -> VAStatus,
    pub vaErrorStr: unsafe extern "C" fn(VAStatus) -> *const c_char,
    pub vaQueryVendorString: unsafe extern "C" fn(VADisplay) -> *const c_char,
    pub vaCreateConfig: unsafe extern "C" fn(
        VADisplay,
        c_int,
        c_int,
        *mut VAConfigAttrib,
        c_int,
        *mut VAConfigID,
    ) -> VAStatus,
    pub vaDestroyConfig: unsafe extern "C" fn(VADisplay, VAConfigID) -> VAStatus,
    pub vaCreateSurfaces: unsafe extern "C" fn(
        VADisplay,
        u32,
        u32,
        u32,
        *mut VASurfaceID,
        u32,
        *mut c_void,
        u32,
    ) -> VAStatus,
    pub vaDestroySurfaces: unsafe extern "C" fn(VADisplay, *mut VASurfaceID, c_int) -> VAStatus,
    pub vaCreateContext: unsafe extern "C" fn(
        VADisplay,
        VAConfigID,
        c_int,
        c_int,
        c_int,
        *mut VASurfaceID,
        c_int,
        *mut VAContextID,
    ) -> VAStatus,
    pub vaDestroyContext: unsafe extern "C" fn(VADisplay, VAContextID) -> VAStatus,
    pub vaCreateBuffer: unsafe extern "C" fn(
        VADisplay,
        VAContextID,
        c_int,
        u32,
        u32,
        *mut c_void,
        *mut VABufferID,
    ) -> VAStatus,
    pub vaDestroyBuffer: unsafe extern "C" fn(VADisplay, VABufferID) -> VAStatus,
    pub vaBeginPicture: unsafe extern "C" fn(VADisplay, VAContextID, VASurfaceID) -> VAStatus,
    pub vaRenderPicture:
        unsafe extern "C" fn(VADisplay, VAContextID, *mut VABufferID, c_int) -> VAStatus,
    pub vaEndPicture: unsafe extern "C" fn(VADisplay, VAContextID) -> VAStatus,
    pub vaSyncSurface: unsafe extern "C" fn(VADisplay, VASurfaceID) -> VAStatus,
    pub vaDeriveImage: unsafe extern "C" fn(VADisplay, VASurfaceID, *mut VAImage) -> VAStatus,
    pub vaDestroyImage: unsafe extern "C" fn(VADisplay, VAImageID) -> VAStatus,
    pub vaMapBuffer: unsafe extern "C" fn(VADisplay, VABufferID, *mut *mut c_void) -> VAStatus,
    pub vaUnmapBuffer: unsafe extern "C" fn(VADisplay, VABufferID) -> VAStatus,
    pub vaCreateImage:
        unsafe extern "C" fn(VADisplay, *mut VAImageFormat, c_int, c_int, *mut VAImage) -> VAStatus,
    pub vaGetImage: unsafe extern "C" fn(
        VADisplay,
        VASurfaceID,
        c_int,
        c_int,
        u32,
        u32,
        VAImageID,
    ) -> VAStatus,
}

impl Va {
    /// Open libva and resolve everything, or say why it could not be done.
    ///
    /// Each `transmute` below turns a `dlsym` result into the field's own declared function
    /// type, which is the only type it could become; naming it a second time at every call
    /// site would just be the struct definition again, in a place where a typo is harder to
    /// see.
    #[allow(clippy::missing_transmute_annotations)]
    pub fn load() -> Result<Va> {
        let libva = Library::open("libva.so.2")?;
        let libva_drm = Library::open("libva-drm.so.2")?;
        // SAFETY: every symbol below is looked up by its documented name and given the
        // signature `va.h` declares for it. A missing symbol is an error, not a null call.
        unsafe {
            Ok(Va {
                vaGetDisplayDRM: std::mem::transmute::<*mut c_void, _>(
                    libva_drm.symbol("vaGetDisplayDRM")?,
                ),
                vaInitialize: std::mem::transmute::<*mut c_void, _>(libva.symbol("vaInitialize")?),
                vaTerminate: std::mem::transmute::<*mut c_void, _>(libva.symbol("vaTerminate")?),
                vaErrorStr: std::mem::transmute::<*mut c_void, _>(libva.symbol("vaErrorStr")?),
                vaQueryVendorString: std::mem::transmute::<*mut c_void, _>(
                    libva.symbol("vaQueryVendorString")?,
                ),
                vaCreateConfig: std::mem::transmute::<*mut c_void, _>(
                    libva.symbol("vaCreateConfig")?,
                ),
                vaDestroyConfig: std::mem::transmute::<*mut c_void, _>(
                    libva.symbol("vaDestroyConfig")?,
                ),
                vaCreateSurfaces: std::mem::transmute::<*mut c_void, _>(
                    libva.symbol("vaCreateSurfaces")?,
                ),
                vaDestroySurfaces: std::mem::transmute::<*mut c_void, _>(
                    libva.symbol("vaDestroySurfaces")?,
                ),
                vaCreateContext: std::mem::transmute::<*mut c_void, _>(
                    libva.symbol("vaCreateContext")?,
                ),
                vaDestroyContext: std::mem::transmute::<*mut c_void, _>(
                    libva.symbol("vaDestroyContext")?,
                ),
                vaCreateBuffer: std::mem::transmute::<*mut c_void, _>(
                    libva.symbol("vaCreateBuffer")?,
                ),
                vaDestroyBuffer: std::mem::transmute::<*mut c_void, _>(
                    libva.symbol("vaDestroyBuffer")?,
                ),
                vaBeginPicture: std::mem::transmute::<*mut c_void, _>(
                    libva.symbol("vaBeginPicture")?,
                ),
                vaRenderPicture: std::mem::transmute::<*mut c_void, _>(
                    libva.symbol("vaRenderPicture")?,
                ),
                vaEndPicture: std::mem::transmute::<*mut c_void, _>(libva.symbol("vaEndPicture")?),
                vaSyncSurface: std::mem::transmute::<*mut c_void, _>(
                    libva.symbol("vaSyncSurface")?,
                ),
                vaDeriveImage: std::mem::transmute::<*mut c_void, _>(
                    libva.symbol("vaDeriveImage")?,
                ),
                vaDestroyImage: std::mem::transmute::<*mut c_void, _>(
                    libva.symbol("vaDestroyImage")?,
                ),
                vaMapBuffer: std::mem::transmute::<*mut c_void, _>(libva.symbol("vaMapBuffer")?),
                vaUnmapBuffer: std::mem::transmute::<*mut c_void, _>(
                    libva.symbol("vaUnmapBuffer")?,
                ),
                vaCreateImage: std::mem::transmute::<*mut c_void, _>(
                    libva.symbol("vaCreateImage")?,
                ),
                vaGetImage: std::mem::transmute::<*mut c_void, _>(libva.symbol("vaGetImage")?),
                _libva: libva,
                _libva_drm: libva_drm,
            })
        }
    }

    /// Turn a failing `VAStatus` into an [`Error`], with libva's own message.
    pub fn check(&self, call: &'static str, status: VAStatus) -> Result<()> {
        if status == VA_STATUS_SUCCESS {
            return Ok(());
        }
        // SAFETY: `vaErrorStr` returns a static, NUL-terminated string for any status.
        let text = unsafe {
            let ptr = (self.vaErrorStr)(status);
            if ptr.is_null() {
                String::new()
            } else {
                std::ffi::CStr::from_ptr(ptr).to_string_lossy().into_owned()
            }
        };
        Err(Error::Platform { call, code: status as i64, detail: text })
    }
}

/// A `dlopen`ed shared object.
pub struct Library {
    handle: *mut c_void,
    name: &'static str,
}

impl Library {
    /// `dlopen(name, RTLD_NOW | RTLD_LOCAL)`.
    pub fn open(name: &'static str) -> Result<Library> {
        let c_name = CString::new(name).expect("library names hold no NUL");
        // SAFETY: `c_name` is a valid NUL-terminated string for the duration of the call.
        let handle = unsafe { libc::dlopen(c_name.as_ptr(), libc::RTLD_NOW | libc::RTLD_LOCAL) };
        if handle.is_null() {
            return Err(Error::no_decoder(format!("{name} could not be loaded: {}", dlerror())));
        }
        Ok(Library { handle, name })
    }

    /// Resolve a symbol.
    pub fn symbol(&self, name: &str) -> Result<*mut c_void> {
        let c_name = CString::new(name).expect("symbol names hold no NUL");
        // SAFETY: the handle is live and the name is NUL-terminated.
        let ptr = unsafe { libc::dlsym(self.handle, c_name.as_ptr()) };
        if ptr.is_null() {
            return Err(Error::no_decoder(format!(
                "{} does not export {name}: {}",
                self.name,
                dlerror()
            )));
        }
        Ok(ptr)
    }
}

impl Drop for Library {
    fn drop(&mut self) {
        // SAFETY: the handle came from `dlopen` and is closed exactly once.
        unsafe { libc::dlclose(self.handle) };
    }
}

fn dlerror() -> String {
    // SAFETY: `dlerror` returns either null or a NUL-terminated string owned by libdl.
    unsafe {
        let ptr = libc::dlerror();
        if ptr.is_null() {
            "no error reported".to_string()
        } else {
            std::ffi::CStr::from_ptr(ptr).to_string_lossy().into_owned()
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn bit_fields_pack_in_declaration_order() {
        let seq = SeqFields {
            chroma_format_idc: 1,
            frame_mbs_only_flag: true,
            direct_8x8_inference_flag: true,
            log2_max_frame_num_minus4: 2,
            pic_order_cnt_type: 0,
            log2_max_pic_order_cnt_lsb_minus4: 3,
            ..SeqFields::default()
        };
        // chroma_format_idc at bit 0, frame_mbs_only at bit 4, direct_8x8 at bit 6,
        // log2_max_frame_num at bits 8..12, poc type at 12..14, poc lsb at 14..18.
        assert_eq!(seq.pack(), 1 | (1 << 4) | (1 << 6) | (2 << 8) | (3 << 14));

        let pic = PicFields {
            entropy_coding_mode_flag: true,
            weighted_bipred_idc: 2,
            reference_pic_flag: true,
            ..PicFields::default()
        };
        assert_eq!(pic.pack(), 1 | (2 << 2) | (1 << 12));
    }

    #[test]
    fn structures_match_the_c_layout() {
        // VAPictureH264: five 4-byte fields plus four reserved words.
        assert_eq!(std::mem::size_of::<VAPictureH264>(), 36);
        // The picture parameter buffer's own fields, with the arrays it contains.
        assert_eq!(std::mem::size_of::<VAIQMatrixBufferH264>(), 6 * 16 + 2 * 64 + 16);
    }
}
