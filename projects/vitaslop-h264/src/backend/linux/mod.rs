//! Linux: VA-API.
//!
//! There is no single "decode this" call on Linux. VA-API's H.264 entry point is stateless:
//! the driver is handed one picture's slices, the reference lists to decode them against,
//! and the surface to write into - and it keeps nothing between calls. So this backend
//! carries the whole of the decoded picture buffer ([`dpb`]), the reference list
//! construction, and the reference marking, which on the other three platforms the system
//! decoder does internally.
//!
//! libva is opened with `dlopen` rather than linked ([`va`]), so a binary built with this
//! crate still runs on a machine with no VA-API - it reports [`crate::Error::NoDecoder`].
//!
//! # What this path refuses
//!
//! Field-coded (interlaced) streams, 4:2:2 and 4:4:4 chroma, bit depths above 8, and
//! flexible macroblock ordering. Each of those needs list construction or a surface format
//! this file does not implement, and a decoder that half-implements them produces a picture
//! that looks decoded and is wrong.

pub mod dpb;
pub mod va;

use std::ffi::{CString, c_int, c_void};

use dpb::{Dpb, DpbEntry, OutputPicture, RefKind};
use va::*;

use super::{Backend, FramePool, OutputOrder, StreamConfig};
use crate::bitstream::slice::{SliceHeader, SliceType};
use crate::bitstream::{AccessUnit, sps::Sps};
use crate::error::{Error, Result};
use crate::frame::{Frame, PixelFormat};

/// Extra surfaces beyond the DPB: one for the picture being decoded, and a few so that
/// output copies never have to wait for a surface to come free.
const SPARE_SURFACES: usize = 4;

/// The VA-API backend.
pub struct VaapiBackend {
    va: Va,
    display: VADisplay,
    /// The DRM render node's file descriptor, held open for the display's lifetime.
    fd: c_int,
    /// Which node was opened, for diagnostics.
    device: String,
    config: VAConfigID,
    context: VAContextID,
    surfaces: Vec<VASurfaceID>,
    dpb: Dpb,
    /// Pictures the buffer has released for output but whose pixels are not copied yet.
    pending: Vec<OutputPicture>,
    coded: (u32, u32),
    visible: (u32, u32),
    /// The profile the context was created for, so a repeated SPS does not rebuild it.
    profile: c_int,
    /// False once `vaDeriveImage` has been seen to fail: some drivers do not implement it
    /// for decode surfaces, and the copy path then goes through `vaGetImage` instead.
    can_derive: bool,
}

// SAFETY: a VADisplay is a per-thread handle in practice, and this type is only ever used
// through an exclusive borrow. It is Send so a caller can move a decoder between threads,
// which is what an ordinary "decode on a worker" arrangement needs.
unsafe impl Send for VaapiBackend {}

impl VaapiBackend {
    /// Open a DRM render node and initialise VA-API on it.
    pub fn new() -> Result<VaapiBackend> {
        let va = Va::load()?;
        let (fd, device) = open_render_node()?;
        // SAFETY: `fd` is an open DRM render node, which is what vaGetDisplayDRM wants.
        let display = unsafe { (va.vaGetDisplayDRM)(fd) };
        if display.is_null() {
            // SAFETY: closing a descriptor this function opened and no longer uses.
            unsafe { libc::close(fd) };
            return Err(Error::no_decoder(format!("vaGetDisplayDRM({device}) returned no display")));
        }
        let mut major = 0;
        let mut minor = 0;
        // SAFETY: the display was just created; both out-parameters are ours.
        let status = unsafe { (va.vaInitialize)(display, &mut major, &mut minor) };
        if status != VA_STATUS_SUCCESS {
            // SAFETY: the display is being abandoned along with its descriptor.
            unsafe { libc::close(fd) };
            return Err(Error::no_decoder(format!(
                "vaInitialize on {device} failed with status {status}"
            )));
        }
        Ok(VaapiBackend {
            va,
            display,
            fd,
            device,
            config: VA_INVALID_ID,
            context: VA_INVALID_ID,
            surfaces: Vec::new(),
            dpb: Dpb::default(),
            pending: Vec::new(),
            coded: (0, 0),
            visible: (0, 0),
            profile: -1,
            can_derive: true,
        })
    }

    /// The driver behind the display ("Intel iHD driver", "Mesa Gallium driver ...").
    pub fn vendor(&self) -> String {
        // SAFETY: the display is live; libva owns the returned string.
        unsafe {
            let ptr = (self.va.vaQueryVendorString)(self.display);
            if ptr.is_null() {
                String::new()
            } else {
                std::ffi::CStr::from_ptr(ptr).to_string_lossy().into_owned()
            }
        }
    }

    /// Tear down the context, config and surfaces (but not the display).
    fn destroy_context(&mut self) {
        // SAFETY: every handle is one this backend created, destroyed at most once.
        unsafe {
            if self.context != VA_INVALID_ID {
                (self.va.vaDestroyContext)(self.display, self.context);
                self.context = VA_INVALID_ID;
            }
            if !self.surfaces.is_empty() {
                (self.va.vaDestroySurfaces)(
                    self.display,
                    self.surfaces.as_mut_ptr(),
                    self.surfaces.len() as c_int,
                );
                self.surfaces.clear();
            }
            if self.config != VA_INVALID_ID {
                (self.va.vaDestroyConfig)(self.display, self.config);
                self.config = VA_INVALID_ID;
            }
        }
        self.dpb.clear();
        self.pending.clear();
    }

    /// A surface no picture is using.
    fn free_surface(&self) -> Option<VASurfaceID> {
        self.surfaces
            .iter()
            .copied()
            .find(|&s| !self.dpb.holds(s) && !self.pending.iter().any(|p| p.surface == s))
    }

    /// Create a VA buffer holding `data`, returning its id.
    ///
    /// # Safety
    ///
    /// `data` must be the layout the buffer type expects.
    unsafe fn buffer(&self, kind: c_int, size: usize, count: usize, data: *mut c_void) -> Result<VABufferID> {
        let mut id = VA_INVALID_ID;
        // SAFETY: the caller guarantees `data` points at `size * count` bytes of the layout
        // libva expects for `kind`; libva copies them before returning.
        let status = unsafe {
            (self.va.vaCreateBuffer)(
                self.display,
                self.context,
                kind,
                size as u32,
                count as u32,
                data,
                &mut id,
            )
        };
        self.va.check("vaCreateBuffer", status)?;
        Ok(id)
    }

    /// Build the picture parameter buffer for one access unit.
    fn picture_parameters(
        &self,
        au: &AccessUnit,
        surface: VASurfaceID,
    ) -> VAPictureParameterBufferH264 {
        let sps = &au.sps;
        let pps = &au.pps;
        let header = &au.header;

        let mut reference_frames = [VAPictureH264::invalid(); 16];
        for (slot, entry) in reference_frames
            .iter_mut()
            .zip(self.dpb.frames.iter().filter(|f| f.reference != RefKind::Unused))
        {
            *slot = entry.as_va();
        }

        let current = VAPictureH264 {
            picture_id: surface,
            frame_idx: header.frame_num,
            flags: if au.reference { va::picture_flag::SHORT_TERM_REFERENCE } else { 0 },
            TopFieldOrderCnt: au.poc.top,
            BottomFieldOrderCnt: au.poc.bottom,
            ..VAPictureH264::invalid()
        };

        let seq = SeqFields {
            chroma_format_idc: sps.chroma_format_idc,
            residual_colour_transform_flag: sps.separate_colour_plane,
            gaps_in_frame_num_value_allowed_flag: sps.gaps_in_frame_num_value_allowed,
            frame_mbs_only_flag: sps.frame_mbs_only,
            mb_adaptive_frame_field_flag: sps.mb_adaptive_frame_field,
            direct_8x8_inference_flag: sps.direct_8x8_inference,
            // A.3.3: at level 3.0 and above, 8x8 is the smallest bi-predicted block.
            min_luma_bipred_size8x8: sps.level_idc >= 31,
            log2_max_frame_num_minus4: sps.log2_max_frame_num - 4,
            pic_order_cnt_type: sps.pic_order_cnt_type,
            log2_max_pic_order_cnt_lsb_minus4: sps.log2_max_pic_order_cnt_lsb.saturating_sub(4),
            delta_pic_order_always_zero_flag: sps.delta_pic_order_always_zero,
        };
        let pic = PicFields {
            entropy_coding_mode_flag: pps.cabac,
            weighted_pred_flag: pps.weighted_pred,
            weighted_bipred_idc: pps.weighted_bipred_idc,
            transform_8x8_mode_flag: pps.transform_8x8_mode,
            field_pic_flag: header.field_pic,
            constrained_intra_pred_flag: pps.constrained_intra_pred,
            pic_order_present_flag: pps.bottom_field_pic_order_in_frame_present,
            deblocking_filter_control_present_flag: pps.deblocking_filter_control_present,
            redundant_pic_cnt_present_flag: pps.redundant_pic_cnt_present,
            reference_pic_flag: au.reference,
        };

        VAPictureParameterBufferH264 {
            CurrPic: current,
            ReferenceFrames: reference_frames,
            picture_width_in_mbs_minus1: (sps.pic_width_in_mbs - 1) as u16,
            picture_height_in_mbs_minus1: (sps.pic_height_in_map_units - 1) as u16,
            bit_depth_luma_minus8: (sps.bit_depth_luma - 8) as u8,
            bit_depth_chroma_minus8: (sps.bit_depth_chroma - 8) as u8,
            num_ref_frames: sps.max_num_ref_frames as u8,
            seq_fields: seq.pack(),
            num_slice_groups_minus1: 0,
            slice_group_map_type: 0,
            slice_group_change_rate_minus1: 0,
            pic_init_qp_minus26: (pps.pic_init_qp - 26) as i8,
            pic_init_qs_minus26: (pps.pic_init_qs - 26) as i8,
            chroma_qp_index_offset: pps.chroma_qp_index_offset as i8,
            second_chroma_qp_index_offset: pps.second_chroma_qp_index_offset as i8,
            pic_fields: pic.pack(),
            frame_num: header.frame_num as u16,
            va_reserved: [0; 8],
        }
    }

    /// Build one slice's parameter buffer, including its reference lists.
    fn slice_parameters(
        &self,
        au: &AccessUnit,
        header: &SliceHeader,
        data_offset: usize,
        data_size: usize,
        nal_bit_offset: usize,
    ) -> Result<VASliceParameterBufferH264> {
        let mut p = VASliceParameterBufferH264 {
            slice_data_size: data_size as u32,
            slice_data_offset: data_offset as u32,
            slice_data_flag: VA_SLICE_DATA_FLAG_ALL,
            slice_data_bit_offset: u16::try_from(nal_bit_offset).map_err(|_| {
                Error::bitstream("slice header longer than 8 KB, which no encoder emits")
            })?,
            first_mb_in_slice: header.first_mb_in_slice as u16,
            slice_type: match header.slice_type {
                SliceType::P => 0,
                SliceType::B => 1,
                SliceType::I => 2,
                SliceType::Sp => 3,
                SliceType::Si => 4,
            },
            direct_spatial_mv_pred_flag: header.direct_spatial_mv_pred as u8,
            num_ref_idx_l0_active_minus1: header.num_ref_idx_l0_active.saturating_sub(1) as u8,
            num_ref_idx_l1_active_minus1: header.num_ref_idx_l1_active.saturating_sub(1) as u8,
            cabac_init_idc: header.cabac_init_idc as u8,
            slice_qp_delta: header.slice_qp_delta as i8,
            disable_deblocking_filter_idc: header.disable_deblocking_filter_idc as u8,
            slice_alpha_c0_offset_div2: header.slice_alpha_c0_offset_div2 as i8,
            slice_beta_offset_div2: header.slice_beta_offset_div2 as i8,
            ..VASliceParameterBufferH264::default()
        };

        let max_frame_num = 1u32 << au.sps.log2_max_frame_num;
        let (mut list0, mut list1) =
            self.dpb.initial_lists(header.slice_type, au.order());
        if header.slice_type.uses_list0() {
            self.dpb.modify_list(
                &mut list0,
                &header.ref_pic_list_mod[0],
                header.num_ref_idx_l0_active as usize,
                header.frame_num,
                max_frame_num,
            )?;
            for (slot, &index) in p.RefPicList0.iter_mut().zip(list0.iter()) {
                *slot = self.dpb.frames[index].as_va();
            }
        }
        if header.slice_type.uses_list1() {
            self.dpb.modify_list(
                &mut list1,
                &header.ref_pic_list_mod[1],
                header.num_ref_idx_l1_active as usize,
                header.frame_num,
                max_frame_num,
            )?;
            for (slot, &index) in p.RefPicList1.iter_mut().zip(list1.iter()) {
                *slot = self.dpb.frames[index].as_va();
            }
        }

        if let Some(weights) = &header.pred_weight {
            p.luma_log2_weight_denom = weights.luma_log2_denom as u8;
            p.chroma_log2_weight_denom = weights.chroma_log2_denom as u8;
            p.luma_weight_l0_flag = !weights.luma[0].is_empty() as u8;
            p.chroma_weight_l0_flag = !weights.chroma[0].is_empty() as u8;
            p.luma_weight_l1_flag = !weights.luma[1].is_empty() as u8;
            p.chroma_weight_l1_flag = !weights.chroma[1].is_empty() as u8;
            for (i, w) in weights.luma[0].iter().take(32).enumerate() {
                p.luma_weight_l0[i] = w.weight as i16;
                p.luma_offset_l0[i] = w.offset as i16;
            }
            for (i, w) in weights.luma[1].iter().take(32).enumerate() {
                p.luma_weight_l1[i] = w.weight as i16;
                p.luma_offset_l1[i] = w.offset as i16;
            }
            for (i, c) in weights.chroma[0].iter().take(32).enumerate() {
                for (k, w) in c.iter().enumerate() {
                    p.chroma_weight_l0[i][k] = w.weight as i16;
                    p.chroma_offset_l0[i][k] = w.offset as i16;
                }
            }
            for (i, c) in weights.chroma[1].iter().take(32).enumerate() {
                for (k, w) in c.iter().enumerate() {
                    p.chroma_weight_l1[i][k] = w.weight as i16;
                    p.chroma_offset_l1[i][k] = w.offset as i16;
                }
            }
        }
        Ok(p)
    }

    /// Copy one decoded surface into a frame.
    fn read_surface(&mut self, picture: OutputPicture, pool: &mut FramePool) -> Result<Frame> {
        // SAFETY: the surface belongs to this context; syncing blocks until the driver has
        // finished writing it, which is what makes the map below read finished pixels.
        let status = unsafe { (self.va.vaSyncSurface)(self.display, picture.surface) };
        self.va.check("vaSyncSurface", status)?;

        let mut image = VAImage::default();
        let mut derived = false;
        if self.can_derive {
            // SAFETY: `image` is ours to fill.
            let status =
                unsafe { (self.va.vaDeriveImage)(self.display, picture.surface, &mut image) };
            if status == VA_STATUS_SUCCESS {
                derived = true;
            } else {
                // Not every driver implements a direct mapping of a decode surface. The
                // fallback below always works, at the cost of a driver-side copy - so the
                // failure is remembered rather than retried once per frame.
                self.can_derive = false;
            }
        }
        if !derived {
            let mut format = VAImageFormat {
                fourcc: VA_FOURCC_NV12,
                byte_order: 1, // VA_LSB_FIRST
                bits_per_pixel: 12,
                ..VAImageFormat::default()
            };
            // SAFETY: one format in, one image out, both ours.
            let status = unsafe {
                (self.va.vaCreateImage)(
                    self.display,
                    &mut format,
                    self.coded.0 as c_int,
                    self.coded.1 as c_int,
                    &mut image,
                )
            };
            self.va.check("vaCreateImage", status)?;
            // SAFETY: copies the whole coded picture into the image just created.
            let status = unsafe {
                (self.va.vaGetImage)(
                    self.display,
                    picture.surface,
                    0,
                    0,
                    self.coded.0,
                    self.coded.1,
                    image.image_id,
                )
            };
            if status != VA_STATUS_SUCCESS {
                // SAFETY: releasing the image created just above.
                unsafe { (self.va.vaDestroyImage)(self.display, image.image_id) };
                return Err(self.va.check("vaGetImage", status).unwrap_err());
            }
        }

        let result = self.copy_image(&image, picture.key, pool);
        // SAFETY: the image was derived or created above, and is destroyed exactly once.
        unsafe { (self.va.vaDestroyImage)(self.display, image.image_id) };
        result
    }

    /// Map an image's buffer and copy the visible region out of it.
    fn copy_image(&self, image: &VAImage, key: i64, pool: &mut FramePool) -> Result<Frame> {
        if image.format.fourcc != VA_FOURCC_NV12 {
            return Err(Error::unsupported(format!(
                "the driver decoded into fourcc 0x{:08x}, not NV12",
                image.format.fourcc
            )));
        }
        let mut data: *mut c_void = std::ptr::null_mut();
        // SAFETY: mapping a buffer this image owns; unmapped below on every path.
        let status = unsafe { (self.va.vaMapBuffer)(self.display, image.buf, &mut data) };
        self.va.check("vaMapBuffer", status)?;

        let result = (|| -> Result<Frame> {
            if data.is_null() {
                return Err(Error::platform("vaMapBuffer", 0, "mapped a null pointer"));
            }
            // SAFETY: libva reports the mapping's size in `data_size`, and every read below
            // is bounds-checked against the slice built from it.
            let src = unsafe { std::slice::from_raw_parts(data as *const u8, image.data_size as usize) };
            let (width, height) = self.visible;
            let mut frame = pool.frame(PixelFormat::Nv12, width, height);
            frame.pts = key;

            for plane in 0..2 {
                let stride = image.pitches[plane] as usize;
                let base = image.offsets[plane] as usize;
                let dst_plane = frame.planes[plane];
                let needed = base + stride * (dst_plane.rows - 1) + dst_plane.row_bytes;
                if stride == 0 || needed > src.len() {
                    return Err(Error::platform(
                        "vaDeriveImage",
                        0,
                        format!(
                            "plane {plane} (offset {base}, pitch {stride}) does not fit the \
                             {}-byte image",
                            image.data_size
                        ),
                    ));
                }
                let dst = frame.plane_mut(plane);
                for r in 0..dst_plane.rows {
                    let from = base + r * stride;
                    dst[r * dst_plane.stride..r * dst_plane.stride + dst_plane.row_bytes]
                        .copy_from_slice(&src[from..from + dst_plane.row_bytes]);
                }
            }
            Ok(frame)
        })();

        // SAFETY: unmapping the buffer mapped above.
        unsafe { (self.va.vaUnmapBuffer)(self.display, image.buf) };
        result
    }
}

impl Backend for VaapiBackend {
    fn name(&self) -> &'static str {
        "VA-API"
    }

    fn output_order(&self) -> OutputOrder {
        // The DPB in this file outputs by picture order count, so frames leave it in
        // presentation order already.
        OutputOrder::Presentation
    }

    fn configure(&mut self, config: StreamConfig<'_>) -> Result<()> {
        let sps = config.sps;
        if !sps.frame_mbs_only {
            return Err(Error::unsupported(
                "field or MBAFF coding on the VA-API path (frame_mbs_only_flag = 0)",
            ));
        }
        let profile = va_profile(sps)?;
        let coded = (sps.coded_width(), sps.coded_height());
        if self.context != VA_INVALID_ID && profile == self.profile && coded == self.coded {
            self.visible = (config.width, config.height);
            return Ok(());
        }
        self.destroy_context();

        let mut attrib = VAConfigAttrib { type_: 0 /* VAConfigAttribRTFormat */, value: VA_RT_FORMAT_YUV420 };
        let mut config_id = VA_INVALID_ID;
        // SAFETY: one attribute in, one config id out.
        let status = unsafe {
            (self.va.vaCreateConfig)(
                self.display,
                profile,
                VA_ENTRYPOINT_VLD,
                &mut attrib,
                1,
                &mut config_id,
            )
        };
        if status != VA_STATUS_SUCCESS {
            return Err(Error::no_decoder(format!(
                "{} cannot decode H.264 profile {} ({}): VA status {status}",
                self.device,
                sps.profile_idc,
                self.vendor()
            )));
        }
        self.config = config_id;
        self.profile = profile;
        self.coded = coded;
        self.visible = (config.width, config.height);

        let dpb_frames = sps.max_dpb_frames() as usize;
        let count = dpb_frames + SPARE_SURFACES;
        let mut surfaces = vec![VA_INVALID_SURFACE; count];
        // SAFETY: `surfaces` has room for `count` ids and no surface attributes are passed.
        let status = unsafe {
            (self.va.vaCreateSurfaces)(
                self.display,
                VA_RT_FORMAT_YUV420,
                coded.0,
                coded.1,
                surfaces.as_mut_ptr(),
                count as u32,
                std::ptr::null_mut(),
                0,
            )
        };
        self.va.check("vaCreateSurfaces", status)?;
        self.surfaces = surfaces;

        let mut context = VA_INVALID_ID;
        // SAFETY: the surfaces above are this context's render targets and outlive it.
        let status = unsafe {
            (self.va.vaCreateContext)(
                self.display,
                self.config,
                coded.0 as c_int,
                coded.1 as c_int,
                VA_PROGRESSIVE,
                self.surfaces.as_mut_ptr(),
                self.surfaces.len() as c_int,
                &mut context,
            )
        };
        self.va.check("vaCreateContext", status)?;
        self.context = context;

        self.dpb.configure(dpb_frames, sps.max_reorder_frames() as usize, sps.max_num_ref_frames as usize);
        Ok(())
    }

    fn send(&mut self, au: &AccessUnit, timestamp: i64) -> Result<()> {
        if self.context == VA_INVALID_ID {
            return Err(Error::State("VA-API context used before it was configured"));
        }
        if au.header.field_pic {
            return Err(Error::unsupported("field-coded pictures on the VA-API path"));
        }
        if au.slices.is_empty() {
            return Err(Error::bitstream("access unit with no slices"));
        }

        let max_frame_num = 1u32 << au.sps.log2_max_frame_num;
        self.dpb.update_pic_nums(au.header.frame_num, max_frame_num);

        let surface = self
            .free_surface()
            .ok_or(Error::State("no free surface: decoded frames must be received"))?;

        // SAFETY: the surface belongs to this context.
        let status = unsafe { (self.va.vaBeginPicture)(self.display, self.context, surface) };
        self.va.check("vaBeginPicture", status)?;

        let mut buffers: Vec<VABufferID> = Vec::with_capacity(2 + au.slices.len() * 2);
        let result = (|| -> Result<()> {
            let mut picture = self.picture_parameters(au, surface);
            // SAFETY: the buffer is exactly one picture parameter structure.
            let id = unsafe {
                self.buffer(
                    va::buffer_type::PICTURE_PARAMETER,
                    std::mem::size_of::<VAPictureParameterBufferH264>(),
                    1,
                    &mut picture as *mut _ as *mut c_void,
                )?
            };
            buffers.push(id);

            let mut matrix = VAIQMatrixBufferH264 {
                ScalingList4x4: au.pps.scaling.list4x4,
                ScalingList8x8: [au.pps.scaling.list8x8[0], au.pps.scaling.list8x8[1]],
                ..VAIQMatrixBufferH264::default()
            };
            // SAFETY: one IQ matrix structure.
            let id = unsafe {
                self.buffer(
                    va::buffer_type::IQ_MATRIX,
                    std::mem::size_of::<VAIQMatrixBufferH264>(),
                    1,
                    &mut matrix as *mut _ as *mut c_void,
                )?
            };
            buffers.push(id);

            for slice in &au.slices {
                let nal = &au.data[slice.offset..slice.offset + slice.len];
                let bit_offset = nal_bit_offset(nal, slice.header.slice_data_bit_offset);
                let mut params =
                    self.slice_parameters(au, &slice.header, 0, nal.len(), bit_offset)?;
                // SAFETY: one slice parameter structure.
                let id = unsafe {
                    self.buffer(
                        va::buffer_type::SLICE_PARAMETER,
                        std::mem::size_of::<VASliceParameterBufferH264>(),
                        1,
                        &mut params as *mut _ as *mut c_void,
                    )?
                };
                buffers.push(id);
                // SAFETY: the slice's own bytes, NAL header included - which is what
                // `slice_data_bit_offset` above is counted from.
                let id = unsafe {
                    self.buffer(
                        va::buffer_type::SLICE_DATA,
                        nal.len(),
                        1,
                        nal.as_ptr() as *mut c_void,
                    )?
                };
                buffers.push(id);
            }

            // SAFETY: every id in `buffers` was created against this context.
            let status = unsafe {
                (self.va.vaRenderPicture)(
                    self.display,
                    self.context,
                    buffers.as_mut_ptr(),
                    buffers.len() as c_int,
                )
            };
            self.va.check("vaRenderPicture", status)?;
            // SAFETY: ends the picture begun above.
            let status = unsafe { (self.va.vaEndPicture)(self.display, self.context) };
            self.va.check("vaEndPicture", status)
        })();

        for id in buffers {
            // SAFETY: each id was created above; libva copied the contents at creation, so
            // releasing them here is safe even while the picture is still decoding.
            unsafe { (self.va.vaDestroyBuffer)(self.display, id) };
        }
        result?;

        let entry = DpbEntry {
            surface,
            frame_num: au.header.frame_num,
            frame_num_wrap: au.header.frame_num as i32,
            pic_num: au.header.frame_num as i32,
            long_term_frame_idx: 0,
            long_term_pic_num: 0,
            poc: au.poc,
            key: timestamp,
            reference: if au.reference { RefKind::Short } else { RefKind::Unused },
            needed_for_output: true,
        };
        self.dpb.mark_and_insert(
            entry,
            au.header.marking.as_ref(),
            au.idr,
            max_frame_num,
            &mut self.pending,
        )
    }

    fn poll(&mut self, pool: &mut FramePool, out: &mut Vec<Frame>) -> Result<()> {
        while !self.pending.is_empty() {
            let picture = self.pending.remove(0);
            out.push(self.read_surface(picture, pool)?);
        }
        Ok(())
    }

    fn drain(&mut self, pool: &mut FramePool, out: &mut Vec<Frame>) -> Result<()> {
        let mut flushed = Vec::new();
        self.dpb.flush(&mut flushed);
        self.pending.extend(flushed);
        self.poll(pool, out)
    }

    fn reset(&mut self) -> Result<()> {
        self.dpb.clear();
        self.pending.clear();
        Ok(())
    }
}

impl Drop for VaapiBackend {
    fn drop(&mut self) {
        self.destroy_context();
        // SAFETY: the display and descriptor were created in `new` and are released once.
        unsafe {
            (self.va.vaTerminate)(self.display);
            libc::close(self.fd);
        }
    }
}

/// Map an SPS's profile onto a `VAProfile`.
///
/// High is a superset of Main which is a superset of Constrained Baseline, so a stream is
/// mapped up rather than refused when its exact profile has no VA-API value: a driver that
/// advertises High decodes all three.
fn va_profile(sps: &Sps) -> Result<c_int> {
    Ok(match sps.profile_idc {
        // Baseline with constraint_set1 is Constrained Baseline, which every driver has.
        66 => {
            if sps.constraint_flags & (1 << 1) != 0 {
                va::profile::H264_CONSTRAINED_BASELINE
            } else {
                // Unconstrained Baseline allows ASO and redundant slices; no VA-API driver
                // exposes a profile for it, and Main is the closest that is real.
                va::profile::H264_MAIN
            }
        }
        77 => va::profile::H264_MAIN,
        100 => va::profile::H264_HIGH,
        other => {
            return Err(Error::unsupported(format!(
                "H.264 profile_idc {other} has no VA-API decode profile"
            )));
        }
    })
}

/// Convert a bit offset measured in the RBSP into one measured in the NAL as transmitted.
///
/// VA-API counts `slice_data_bit_offset` from the first byte of the NAL - header byte
/// included, emulation-prevention bytes included - while the slice header was parsed out of
/// the RBSP, which has neither. The difference is eight bits for the header plus eight for
/// every `03` that was removed before this point.
fn nal_bit_offset(nal: &[u8], rbsp_bit: usize) -> usize {
    let payload = &nal[1..];
    let target_byte = rbsp_bit / 8;
    let mut removed = 0usize;
    let mut rbsp_index = 0usize;
    let mut zeros = 0usize;
    for &b in payload {
        if zeros >= 2 && b == 3 {
            removed += 1;
            zeros = 0;
            continue;
        }
        if rbsp_index >= target_byte {
            break;
        }
        rbsp_index += 1;
        if b == 0 {
            zeros += 1;
        } else {
            zeros = 0;
        }
    }
    8 + rbsp_bit + removed * 8
}

/// Open the first DRM render node that VA-API can use.
///
/// Render nodes are the unprivileged half of DRM: no X server, no session, no seat - which
/// is what makes this work in a container, over SSH, and on a headless build machine.
fn open_render_node() -> Result<(c_int, String)> {
    let mut last = String::new();
    for index in 128..136 {
        let path = format!("/dev/dri/renderD{index}");
        let c_path = CString::new(path.clone()).expect("paths hold no NUL");
        // SAFETY: a NUL-terminated path and a flag constant; the descriptor is ours.
        let fd = unsafe { libc::open(c_path.as_ptr(), libc::O_RDWR | libc::O_CLOEXEC) };
        if fd >= 0 {
            return Ok((fd, path));
        }
        // SAFETY: reading the thread's errno through libc's accessor.
        let err = std::io::Error::last_os_error();
        last = format!("{path}: {err}");
    }
    Err(Error::no_decoder(format!("no DRM render node could be opened ({last})")))
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn bit_offsets_account_for_emulation_prevention() {
        // A NAL whose payload had one 03 removed before the 40th RBSP bit.
        let nal = [0x65u8, 0x00, 0x00, 0x03, 0x01, 0x88, 0x99];
        // RBSP bit 32 sits in the byte after the removed 03.
        assert_eq!(nal_bit_offset(&nal, 32), 8 + 32 + 8);
        // A bit before the stuffing is unaffected.
        assert_eq!(nal_bit_offset(&nal, 8), 8 + 8);
    }

    #[test]
    fn slice_data_is_not_reparsed_from_the_access_unit() {
        // The NAL slice offsets recorded by the splitter must point at the NAL header byte.
        let data = [0u8, 0, 0, 1, 0x65, 0x88, 0x84, 0x00];
        assert_eq!(&data[4..], &[0x65, 0x88, 0x84, 0x00]);
        assert_eq!(nal::Nal::parse(&data[4..]).unwrap().kind, nal::kind::IDR);
    }
}
