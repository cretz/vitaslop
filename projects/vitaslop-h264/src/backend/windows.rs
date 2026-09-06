//! Windows: the Media Foundation H.264 decoder MFT.
//!
//! Media Foundation is COM, and the decoder is a Transform: parameter sets and slices go in
//! as `IMFSample`s of Annex B bytes, NV12 samples come out. The MFT does its own DPB and
//! emits in presentation order, so the common layer's reorderer is not used here.
//!
//! # Reaching the video hardware
//!
//! Enumerating an MFT is not enough to get hardware decoding, and this is the part that is
//! easy to get wrong quietly. Most GPU vendors expose their decoder as an ASYNCHRONOUS MFT,
//! which needs the whole event-driven `IMFMediaEventGenerator` model; only synchronous
//! transforms are accepted here, so enumeration alone lands on Microsoft's own MFT, which
//! decodes on the CPU.
//!
//! What makes that MFT use the fixed-function decoder is a D3D11 DEVICE MANAGER. Given one,
//! it decodes through DXVA and hands back frames as GPU textures; the cost is that reading
//! them needs a copy through a staging texture. Both paths are implemented: the sample's
//! buffer is asked for `IMFDXGIBuffer` first, and falls back to the system-memory layout.
//!
//! Whether it worked is reported rather than assumed - [`Backend::acceleration`] answers
//! from what the OUTPUT actually was, not from what was asked for.

use std::mem::ManuallyDrop;

use windows::Win32::Graphics::Direct3D::D3D_DRIVER_TYPE_HARDWARE;
use windows::Win32::Graphics::Direct3D11::*;
use windows::Win32::Graphics::Dxgi::Common::{DXGI_FORMAT_NV12, DXGI_SAMPLE_DESC};
use windows::Win32::Media::MediaFoundation::*;
use windows::core::{GUID, Interface};

use super::{Acceleration, Backend, BackendOptions, FramePool, OutputOrder, StreamConfig};
use crate::bitstream::AccessUnit;
use crate::error::{Error, Result};
use crate::frame::{Frame, PixelFormat};

/// Sample times are in 100ns units. One key step is one millisecond, which is far enough
/// apart that nothing inside the MFT can confuse two pictures, and small enough that no
/// stream can overflow the range.
const TICKS_PER_KEY: i64 = 10_000;

/// The Media Foundation backend.
pub struct MediaFoundationBackend {
    transform: IMFTransform,
    /// True when the MFT allocates its own output samples (hardware MFTs do).
    provides_samples: bool,
    /// Bytes the MFT wants an output sample to hold, when we allocate them.
    output_sample_size: u32,
    /// Coded (aligned) frame size of the negotiated output type - what the buffer's plane
    /// layout is built on, and NOT the visible size.
    coded: (u32, u32),
    /// Visible size, from the SPS.
    visible: (u32, u32),
    /// Default stride of the negotiated output type, used when a sample's buffer does not
    /// implement `IMF2DBuffer2`.
    default_stride: i32,
    /// The D3D11 device the MFT decodes through, when one could be given to it.
    d3d: Option<D3dContext>,
    /// Why there is no D3D11 device, when there is none.
    software_reason: Option<String>,
    /// Set once a decoded sample has come back as a GPU texture, which is the only proof
    /// that the fixed-function decoder is what produced it.
    hardware_output_seen: bool,
    /// Largest coded picture the DXVA path may be handed.
    max_hardware_picture_bytes: usize,
    /// True once streaming has begun (so a reset knows to restart it).
    streaming: bool,
    input_scratch: Vec<u8>,
}

// SAFETY: the COM interfaces held here are Media Foundation objects created in the
// process's multithreaded apartment, where a proxy is not required to move a reference
// between threads. What COM does forbid is two threads using one interface AT ONCE, and a
// decoder is owned by one caller through an exclusive borrow, so that cannot happen. The
// D3D11 device is separately marked multithread-protected, which is the same requirement
// stated the way D3D states it.
unsafe impl Send for MediaFoundationBackend {}

impl MediaFoundationBackend {
    /// Start Media Foundation and bind the best available H.264 decoder MFT.
    ///
    /// `options.hardware` asks for DXVA. It is worth being able to turn off for one
    /// specific reason: the hardware path decodes through a driver-sized bitstream buffer,
    /// and a picture larger than that buffer comes back MANGLED with nothing reported -
    /// measured on this machine as exact through a 588 KB coded picture and wrong from
    /// 633 KB, where "wrong" was the first 336 rows correct and the picture then repeating
    /// from its top. Real encoded pictures run 10-100 KB, so this is about pathological
    /// input; an oversized picture is refused by `send` rather than decoded into a lie.
    pub fn new(options: BackendOptions) -> Result<MediaFoundationBackend> {
        startup()?;
        let transform = enumerate_decoder()?;
        let mut backend = MediaFoundationBackend {
            transform,
            provides_samples: false,
            output_sample_size: 0,
            coded: (0, 0),
            visible: (0, 0),
            default_stride: 0,
            d3d: None,
            software_reason: None,
            hardware_output_seen: false,
            max_hardware_picture_bytes: options.max_hardware_picture_bytes,
            streaming: false,
            input_scratch: Vec::new(),
        };
        if options.hardware {
            backend.attach_d3d();
        } else {
            backend.software_reason = Some("the caller asked for software decoding".to_string());
        }
        if options.low_latency {
            backend.request_low_latency();
        }
        Ok(backend)
    }

    /// Ask the MFT to hand each picture over as soon as it is decoded.
    ///
    /// Two settings say it, and both are attempted because which one an MFT honours varies:
    /// the `MF_LOW_LATENCY` attribute on the transform, and `CODECAPI_AVLowLatencyMode`
    /// through `ICodecAPI`. Neither is required to exist - a transform that refuses simply
    /// keeps its pipeline, so a failure here is not reported: it costs latency, not
    /// correctness, and the caller asked for a preference rather than a guarantee.
    fn request_low_latency(&mut self) {
        // SAFETY: both are optional interfaces on the transform we own; every failure path
        // is ignored deliberately.
        unsafe {
            if let Ok(attributes) = self.transform.GetAttributes() {
                let _ = attributes.SetUINT32(&MF_LOW_LATENCY, 1);
            }
            if let Ok(codec_api) = self.transform.cast::<ICodecAPI>() {
                let value = windows::Win32::System::Variant::VARIANT::from(true);
                let _ = codec_api.SetValue(&CODECAPI_AVLowLatencyMode, &value);
            }
        }
    }

    /// Give the MFT a D3D11 device manager, so it decodes through DXVA rather than on the
    /// CPU. Failure here is not fatal: it costs speed, not correctness, and the reason is
    /// kept so [`Backend::acceleration`] can say what happened.
    fn attach_d3d(&mut self) {
        match self.try_attach_d3d() {
            Ok(context) => self.d3d = Some(context),
            Err(reason) => self.software_reason = Some(reason),
        }
    }

    fn try_attach_d3d(&mut self) -> std::result::Result<D3dContext, String> {
        // SAFETY: reading an optional attribute store off the transform.
        let aware = unsafe {
            self.transform
                .GetAttributes()
                .and_then(|attributes| attributes.GetUINT32(&MF_SA_D3D11_AWARE))
                .unwrap_or(0)
        };
        if aware == 0 {
            return Err("the decoder MFT is not D3D11-aware".to_string());
        }

        let mut device: Option<ID3D11Device> = None;
        let mut immediate: Option<ID3D11DeviceContext> = None;
        // SAFETY: a hardware device with video support; both out-parameters are ours.
        unsafe {
            D3D11CreateDevice(
                None,
                D3D_DRIVER_TYPE_HARDWARE,
                windows::Win32::Foundation::HMODULE::default(),
                D3D11_CREATE_DEVICE_VIDEO_SUPPORT,
                None,
                D3D11_SDK_VERSION,
                Some(&mut device),
                None,
                Some(&mut immediate),
            )
            .map_err(|e| format!("D3D11CreateDevice with video support failed: {e}"))?;
        }
        let device = device.ok_or("D3D11CreateDevice returned no device")?;
        let immediate = immediate.ok_or("D3D11CreateDevice returned no context")?;

        // Media Foundation uses the device from its own threads, so it must be told that.
        // Without this the driver is free to assume single-threaded access and the result
        // is corruption that only shows up under load.
        if let Ok(multithread) = device.cast::<ID3D11Multithread>() {
            // SAFETY: the interface came from this device. The returned value is the
            // PREVIOUS setting, which nothing here needs.
            let _previous = unsafe { multithread.SetMultithreadProtected(true) };
        }

        let mut token = 0u32;
        let mut manager: Option<IMFDXGIDeviceManager> = None;
        // SAFETY: both out-parameters are ours; the token pairs with the manager.
        unsafe {
            MFCreateDXGIDeviceManager(&mut token, &mut manager)
                .map_err(|e| format!("MFCreateDXGIDeviceManager failed: {e}"))?;
        }
        let manager = manager.ok_or("MFCreateDXGIDeviceManager returned no manager")?;
        // SAFETY: the manager takes a reference on the device.
        unsafe {
            manager
                .ResetDevice(&device, token)
                .map_err(|e| format!("IMFDXGIDeviceManager::ResetDevice failed: {e}"))?;
        }
        // SAFETY: the MFT takes its own reference on the manager, which this struct also
        // holds for as long as the transform lives.
        unsafe {
            self.transform
                .ProcessMessage(MFT_MESSAGE_SET_D3D_MANAGER, manager.as_raw() as usize)
                .map_err(|e| format!("MFT_MESSAGE_SET_D3D_MANAGER was refused: {e}"))?;
        }
        // The MFT decodes on its own thread, through this same device. Reading a decode
        // texture therefore has to take the device lock the manager exists to provide -
        // WITHOUT it the copy races the decoder and returns a partly written picture, which
        // is not a crash and not an error: it is a frame that is correct for the first few
        // hundred rows and untouched memory after (measured, twice, at different rows).
        let handle = unsafe { manager.OpenDeviceHandle() }
            .map_err(|e| format!("IMFDXGIDeviceManager::OpenDeviceHandle failed: {e}"))?;
        Ok(D3dContext { device, immediate, manager, handle, staging: None })
    }

    /// Negotiate the NV12 output type and read back the layout the MFT settled on.
    fn negotiate_output(&mut self) -> Result<()> {
        let mut index = 0u32;
        loop {
            let candidate = unsafe { self.transform.GetOutputAvailableType(0, index) };
            let candidate = match candidate {
                Ok(t) => t,
                Err(e) if e.code() == MF_E_NO_MORE_TYPES => {
                    return Err(Error::platform(
                        "IMFTransform::GetOutputAvailableType",
                        MF_E_NO_MORE_TYPES.0 as i64,
                        "the decoder offers no NV12 output type",
                    ));
                }
                Err(e) => return Err(mf_err("IMFTransform::GetOutputAvailableType", e)),
            };
            let subtype = unsafe { candidate.GetGUID(&MF_MT_SUBTYPE) }
                .map_err(|e| mf_err("IMFMediaType::GetGUID(MF_MT_SUBTYPE)", e))?;
            if subtype == MFVideoFormat_NV12 {
                unsafe { self.transform.SetOutputType(0, &candidate, 0) }
                    .map_err(|e| mf_err("IMFTransform::SetOutputType", e))?;
                let size = unsafe { candidate.GetUINT64(&MF_MT_FRAME_SIZE) }
                    .map_err(|e| mf_err("IMFMediaType::GetUINT64(MF_MT_FRAME_SIZE)", e))?;
                self.coded = ((size >> 32) as u32, size as u32);
                self.default_stride = unsafe { candidate.GetUINT32(&MF_MT_DEFAULT_STRIDE) }
                    .map(|v| v as i32)
                    .unwrap_or(self.coded.0 as i32);
                let info = unsafe { self.transform.GetOutputStreamInfo(0) }
                    .map_err(|e| mf_err("IMFTransform::GetOutputStreamInfo", e))?;
                let provides = MFT_OUTPUT_STREAM_PROVIDES_SAMPLES.0 as u32
                    | MFT_OUTPUT_STREAM_CAN_PROVIDE_SAMPLES.0 as u32;
                self.provides_samples = info.dwFlags & provides != 0;
                self.output_sample_size = info.cbSize;
                return Ok(());
            }
            index += 1;
        }
    }

    /// Pull one sample out of the MFT, if it has one ready.
    ///
    /// Returns `Ok(false)` for "nothing yet", which is the ordinary state of a pipelined
    /// decoder and not an error.
    fn process_output(&mut self, pool: &mut FramePool, out: &mut Vec<Frame>) -> Result<bool> {
        let sample = if self.provides_samples {
            None
        } else {
            Some(alloc_sample(self.output_sample_size)?)
        };
        let mut buffers = [MFT_OUTPUT_DATA_BUFFER {
            dwStreamID: 0,
            pSample: ManuallyDrop::new(sample.clone()),
            dwStatus: 0,
            pEvents: ManuallyDrop::new(None),
        }];
        let mut status = 0u32;
        let hr = unsafe { self.transform.ProcessOutput(0, &mut buffers, &mut status) };

        // The events collection and (for MFT-allocated samples) the sample itself are
        // out-parameters this crate owns from here; taking them out of the ManuallyDrop
        // wrappers is what releases them.
        let produced = unsafe { ManuallyDrop::take(&mut buffers[0].pSample) };
        let _events = unsafe { ManuallyDrop::take(&mut buffers[0].pEvents) };

        match hr {
            Ok(()) => {}
            Err(e) if e.code() == MF_E_TRANSFORM_NEED_MORE_INPUT => return Ok(false),
            Err(e) if e.code() == MF_E_TRANSFORM_STREAM_CHANGE => {
                // The stream's format changed under us (a new SPS with a new size). The
                // MFT holds its output until the type is renegotiated.
                self.negotiate_output()?;
                return Ok(true);
            }
            Err(e) => return Err(mf_err("IMFTransform::ProcessOutput", e)),
        }

        let produced = produced.ok_or_else(|| {
            Error::platform("IMFTransform::ProcessOutput", 0, "succeeded without a sample")
        })?;
        out.push(self.sample_to_frame(&produced, pool)?);
        Ok(true)
    }

    /// Copy one NV12 sample into a pooled frame.
    fn sample_to_frame(&mut self, sample: &IMFSample, pool: &mut FramePool) -> Result<Frame> {
        let time = unsafe { sample.GetSampleTime() }
            .map_err(|e| mf_err("IMFSample::GetSampleTime", e))?;
        // A DXGI-backed sample must NOT be flattened first: `ConvertToContiguousBuffer`
        // would pull the picture into system memory itself, which is the copy this path
        // exists to control.
        let buffer = match unsafe { sample.GetBufferCount() } {
            Ok(1) => unsafe { sample.GetBufferByIndex(0) }
                .map_err(|e| mf_err("IMFSample::GetBufferByIndex", e))?,
            _ => unsafe { sample.ConvertToContiguousBuffer() }
                .map_err(|e| mf_err("IMFSample::ConvertToContiguousBuffer", e))?,
        };

        let (visible_w, visible_h) = self.visible;
        let mut frame = pool.frame(PixelFormat::Nv12, visible_w, visible_h);
        frame.pts = time / TICKS_PER_KEY;

        // The GPU path: the decoder wrote into a video texture, so the pixels come back
        // through a staging copy rather than a mapped system buffer.
        if let Ok(dxgi) = buffer.cast::<IMFDXGIBuffer>() {
            self.hardware_output_seen = true;
            self.copy_from_texture(&dxgi, &mut frame)?;
            return Ok(frame);
        }

        // `IMF2DBuffer2` is the only way to learn a hardware sample's real pitch; the
        // contiguous fallback is the software MFT's layout, which is the type's stride.
        let two_d: Option<IMF2DBuffer2> = buffer.cast().ok();
        if let Some(two_d) = two_d {
            let mut scanline0: *mut u8 = std::ptr::null_mut();
            let mut pitch: i32 = 0;
            let mut start: *mut u8 = std::ptr::null_mut();
            let mut length: u32 = 0;
            unsafe {
                two_d
                    .Lock2DSize(
                        MF2DBuffer_LockFlags_Read,
                        &mut scanline0,
                        &mut pitch,
                        &mut start,
                        &mut length,
                    )
                    .map_err(|e| mf_err("IMF2DBuffer2::Lock2DSize", e))?;
            }
            let result = copy_nv12(&mut frame, scanline0, pitch, length as usize, self.coded.1 as usize);
            unsafe { two_d.Unlock2D().ok() };
            result?;
        } else {
            let mut data: *mut u8 = std::ptr::null_mut();
            let mut current: u32 = 0;
            unsafe {
                buffer
                    .Lock(&mut data, None, Some(&mut current))
                    .map_err(|e| mf_err("IMFMediaBuffer::Lock", e))?;
            }
            let result =
                copy_nv12(&mut frame, data, self.default_stride, current as usize, self.coded.1 as usize);
            unsafe { buffer.Unlock().ok() };
            result?;
        }
        Ok(frame)
    }

    /// Copy a decoded GPU texture into `frame` through a staging texture.
    ///
    /// A decode texture cannot be mapped directly - it is created without CPU access, and
    /// on most drivers it is also an array slice of a decoder heap - so the picture is
    /// copied into a staging texture of the same size, which is cached and reused rather
    /// than created per frame.
    fn copy_from_texture(&mut self, dxgi: &IMFDXGIBuffer, frame: &mut Frame) -> Result<()> {
        let context = self
            .d3d
            .as_mut()
            .ok_or_else(|| Error::platform("IMFDXGIBuffer", 0, "a GPU sample with no device"))?;

        // SAFETY: the resource is the texture the decoder wrote into; the subresource index
        // says which slice of it.
        let (texture, subresource) = unsafe {
            let mut raw: *mut core::ffi::c_void = std::ptr::null_mut();
            dxgi.GetResource(&ID3D11Texture2D::IID, &mut raw)
                .map_err(|e| mf_err("IMFDXGIBuffer::GetResource", e))?;
            let texture = ID3D11Texture2D::from_raw(raw);
            let index = dxgi
                .GetSubresourceIndex()
                .map_err(|e| mf_err("IMFDXGIBuffer::GetSubresourceIndex", e))?;
            (texture, index)
        };

        let mut desc = D3D11_TEXTURE2D_DESC::default();
        // SAFETY: the texture is live and the descriptor is ours to fill.
        unsafe { texture.GetDesc(&mut desc) };
        if desc.Format != DXGI_FORMAT_NV12 {
            return Err(Error::unsupported(format!(
                "the decoder produced DXGI format {:?}, not NV12",
                desc.Format
            )));
        }
        let staging = context.staging_for(desc.Width, desc.Height)?;

        // Everything from here to the unlock is mutually exclusive with the MFT's own use
        // of the device.
        let lock = DeviceLock::take(&context.manager, context.handle)?;

        // SAFETY: both textures have the same format and dimensions, and the source slice
        // exists (the decoder just wrote it).
        unsafe {
            context.immediate.CopySubresourceRegion(
                &staging,
                0,
                0,
                0,
                0,
                &texture,
                subresource,
                None,
            );
        }

        let mut mapped = D3D11_MAPPED_SUBRESOURCE::default();
        // SAFETY: a staging texture created with CPU read access; unmapped on both paths.
        unsafe {
            context
                .immediate
                .Map(&staging, 0, D3D11_MAP_READ, 0, Some(&mut mapped))
                .map_err(|e| mf_err("ID3D11DeviceContext::Map", e))?;
        }
        let pitch = mapped.RowPitch as usize;
        // An NV12 texture is one allocation: luma rows, then the chroma plane immediately
        // below them at the same pitch.
        let length = pitch * (desc.Height as usize + desc.Height as usize / 2);
        let result = copy_nv12(
            frame,
            mapped.pData as *const u8,
            mapped.RowPitch as i32,
            length,
            desc.Height as usize,
        );
        // SAFETY: unmapping the resource mapped above.
        unsafe { context.immediate.Unmap(&staging, 0) };
        drop(lock);
        result
    }

    fn message(&self, msg: MFT_MESSAGE_TYPE, param: usize) -> Result<()> {
        unsafe { self.transform.ProcessMessage(msg, param) }
            .map_err(|e| mf_err("IMFTransform::ProcessMessage", e))
    }
}

/// Holds the device manager's lock for as long as it is alive.
///
/// A guard rather than a pair of calls because every early return between the lock and the
/// unlock - and there are several, all of them error paths - would otherwise leave the
/// device locked and wedge the decoder on its next frame.
struct DeviceLock<'a> {
    manager: &'a IMFDXGIDeviceManager,
    handle: windows::Win32::Foundation::HANDLE,
}

impl<'a> DeviceLock<'a> {
    fn take(
        manager: &'a IMFDXGIDeviceManager,
        handle: windows::Win32::Foundation::HANDLE,
    ) -> Result<DeviceLock<'a>> {
        let mut device: *mut core::ffi::c_void = std::ptr::null_mut();
        // SAFETY: the handle came from this manager's OpenDeviceHandle. `fblock` = true
        // waits for the MFT rather than failing, which is what a readback wants.
        unsafe {
            manager
                .LockDevice(handle, &ID3D11Device::IID, &mut device, true)
                .map_err(|e| mf_err("IMFDXGIDeviceManager::LockDevice", e))?;
            // The device reference handed back is one this crate already owns.
            if !device.is_null() {
                drop(ID3D11Device::from_raw(device));
            }
        }
        Ok(DeviceLock { manager, handle })
    }
}

impl Drop for DeviceLock<'_> {
    fn drop(&mut self) {
        // SAFETY: paired with the LockDevice above. `fsavestate` = false: nothing here
        // changes device state worth restoring.
        unsafe {
            let _ = self.manager.UnlockDevice(self.handle, false);
        }
    }
}

/// The D3D11 device the MFT decodes through, plus the staging texture its output is read
/// back with.
struct D3dContext {
    device: ID3D11Device,
    immediate: ID3D11DeviceContext,
    /// The manager the MFT decodes through. Held for as long as the transform uses it, and
    /// used to take the device lock around every readback.
    manager: IMFDXGIDeviceManager,
    /// Device handle for `LockDevice`, opened once.
    handle: windows::Win32::Foundation::HANDLE,
    /// Cached readback texture, keyed on the size it was made for.
    staging: Option<(ID3D11Texture2D, u32, u32)>,
}

impl Drop for D3dContext {
    fn drop(&mut self) {
        // SAFETY: closing the handle opened in `try_attach_d3d`.
        unsafe {
            let _ = self.manager.CloseDeviceHandle(self.handle);
        }
    }
}

impl D3dContext {
    /// The staging texture for a `width` x `height` NV12 picture, creating it on the first
    /// frame and on any size change.
    ///
    /// Reusing it matters: creating a texture per frame is a driver allocation sixty times
    /// a second, which is most of what a naive readback path costs.
    /// The returned handle is a reference-counted clone, not a borrow: the caller needs the
    /// device context at the same time, and an `AddRef` is cheaper than the borrow dance.
    fn staging_for(&mut self, width: u32, height: u32) -> Result<ID3D11Texture2D> {
        let stale = !matches!(&self.staging, Some((_, w, h)) if *w == width && *h == height);
        if stale {
            let desc = D3D11_TEXTURE2D_DESC {
                Width: width,
                Height: height,
                MipLevels: 1,
                ArraySize: 1,
                Format: DXGI_FORMAT_NV12,
                SampleDesc: DXGI_SAMPLE_DESC { Count: 1, Quality: 0 },
                Usage: D3D11_USAGE_STAGING,
                BindFlags: 0,
                CPUAccessFlags: D3D11_CPU_ACCESS_READ.0 as u32,
                MiscFlags: 0,
            };
            let mut texture: Option<ID3D11Texture2D> = None;
            // SAFETY: a plain staging texture; no initial data is supplied.
            unsafe {
                self.device
                    .CreateTexture2D(&desc, None, Some(&mut texture))
                    .map_err(|e| mf_err("ID3D11Device::CreateTexture2D", e))?;
            }
            let texture = texture.ok_or_else(|| {
                Error::platform("ID3D11Device::CreateTexture2D", 0, "returned no texture")
            })?;
            self.staging = Some((texture, width, height));
        }
        Ok(self.staging.as_ref().expect("just created").0.clone())
    }
}

impl Backend for MediaFoundationBackend {
    fn name(&self) -> &'static str {
        "MediaFoundation"
    }

    fn output_order(&self) -> OutputOrder {
        OutputOrder::Presentation
    }

    fn configure(&mut self, config: StreamConfig<'_>) -> Result<()> {
        let input = unsafe { MFCreateMediaType() }.map_err(|e| mf_err("MFCreateMediaType", e))?;
        unsafe {
            input
                .SetGUID(&MF_MT_MAJOR_TYPE, &MFMediaType_Video)
                .and_then(|()| input.SetGUID(&MF_MT_SUBTYPE, &MFVideoFormat_H264))
                .and_then(|()| {
                    input.SetUINT64(
                        &MF_MT_FRAME_SIZE,
                        ((config.width as u64) << 32) | config.sps.coded_height() as u64,
                    )
                })
                .and_then(|()| {
                    input.SetUINT32(&MF_MT_INTERLACE_MODE, MFVideoInterlace_MixedInterlaceOrProgressive.0 as u32)
                })
                .map_err(|e| mf_err("IMFMediaType::Set*", e))?;
            self.transform
                .SetInputType(0, &input, 0)
                .map_err(|e| mf_err("IMFTransform::SetInputType", e))?;
        }
        self.visible = (config.width, config.height);
        self.negotiate_output()?;
        if !self.streaming {
            self.message(MFT_MESSAGE_NOTIFY_BEGIN_STREAMING, 0)?;
            self.message(MFT_MESSAGE_NOTIFY_START_OF_STREAM, 0)?;
            self.streaming = true;
        }
        Ok(())
    }

    fn send(&mut self, au: &AccessUnit, timestamp: i64) -> Result<()> {
        if self.d3d.is_some() && au.data.len() > self.max_hardware_picture_bytes {
            return Err(Error::unsupported(format!(
                "a {} KB coded picture exceeds what this driver's DXVA bitstream buffer holds                  ({} KB); it would decode partially and report nothing. Recreate the decoder                  with `hardware: Some(false)`, or raise `max_hardware_picture_bytes` if this                  machine is known to take more",
                au.data.len() / 1024,
                self.max_hardware_picture_bytes / 1024,
            )));
        }
        self.input_scratch.clear();
        self.input_scratch.extend_from_slice(&au.data);
        let sample = alloc_sample(self.input_scratch.len() as u32)?;
        let buffer = unsafe { sample.GetBufferByIndex(0) }
            .map_err(|e| mf_err("IMFSample::GetBufferByIndex", e))?;
        unsafe {
            let mut data: *mut u8 = std::ptr::null_mut();
            buffer
                .Lock(&mut data, None, None)
                .map_err(|e| mf_err("IMFMediaBuffer::Lock", e))?;
            // SAFETY: the buffer was created with at least this capacity just above, and
            // stays mapped until Unlock.
            std::ptr::copy_nonoverlapping(self.input_scratch.as_ptr(), data, self.input_scratch.len());
            buffer.Unlock().ok();
            buffer
                .SetCurrentLength(self.input_scratch.len() as u32)
                .map_err(|e| mf_err("IMFMediaBuffer::SetCurrentLength", e))?;
            sample
                .SetSampleTime(timestamp * TICKS_PER_KEY)
                .and_then(|()| sample.SetSampleDuration(TICKS_PER_KEY))
                .map_err(|e| mf_err("IMFSample::SetSampleTime", e))?;
            if au.idr {
                sample.SetUINT32(&SAMPLE_EXTENSION_CLEAN_POINT, 1).ok();
            }
        }

        match unsafe { self.transform.ProcessInput(0, &sample, 0) } {
            Ok(()) => Ok(()),
            Err(e) if e.code() == MF_E_NOTACCEPTING => {
                // The MFT will not take input while it holds output. This cannot happen
                // with the common layer's poll-after-send loop, but a caller that ignores
                // frames would otherwise wedge here.
                Err(Error::State("decoder output must be drained before more input"))
            }
            Err(e) => Err(mf_err("IMFTransform::ProcessInput", e)),
        }
    }

    fn poll(&mut self, pool: &mut FramePool, out: &mut Vec<Frame>) -> Result<()> {
        while self.process_output(pool, out)? {}
        Ok(())
    }

    fn drain(&mut self, pool: &mut FramePool, out: &mut Vec<Frame>) -> Result<()> {
        if !self.streaming {
            return Ok(());
        }
        self.message(MFT_MESSAGE_COMMAND_DRAIN, 0)?;
        while self.process_output(pool, out)? {}
        // A drained MFT needs a new start-of-stream before it accepts input again.
        self.message(MFT_MESSAGE_NOTIFY_START_OF_STREAM, 0)?;
        Ok(())
    }

    fn acceleration(&self) -> Acceleration {
        if self.hardware_output_seen {
            return Acceleration::Hardware;
        }
        match (&self.d3d, &self.software_reason) {
            // A device is attached but nothing has come back through it yet: the honest
            // answer is that it is not established, not that it worked.
            (Some(_), _) => Acceleration::Unknown,
            (None, Some(reason)) => Acceleration::Software(reason.clone()),
            (None, None) => Acceleration::Unknown,
        }
    }

    fn reset(&mut self) -> Result<()> {
        if self.streaming {
            self.message(MFT_MESSAGE_COMMAND_FLUSH, 0)?;
            self.message(MFT_MESSAGE_NOTIFY_START_OF_STREAM, 0)?;
        }
        Ok(())
    }
}

impl Drop for MediaFoundationBackend {
    fn drop(&mut self) {
        if self.streaming {
            let _ = self.message(MFT_MESSAGE_NOTIFY_END_STREAMING, 0);
        }
        // SAFETY: paired with the MFStartup this instance performed. Media Foundation
        // reference counts the pair, so other users in the process are unaffected.
        unsafe {
            let _ = MFShutdown();
        }
    }
}

/// `MFStartup`, once per backend instance (it is reference counted).
fn startup() -> Result<()> {
    // SAFETY: no preconditions beyond a version that matches the headers we compiled with.
    unsafe { MFStartup(MF_VERSION, MFSTARTUP_NOSOCKET) }.map_err(|e| {
        Error::NoDecoder(format!("MFStartup failed ({}): Media Foundation is not available", e))
    })
}

/// Find an H.264 decoder MFT and activate it.
fn enumerate_decoder() -> Result<IMFTransform> {
    let input_info = MFT_REGISTER_TYPE_INFO { guidMajorType: MFMediaType_Video, guidSubtype: MFVideoFormat_H264 };
    let output_info = MFT_REGISTER_TYPE_INFO { guidMajorType: MFMediaType_Video, guidSubtype: MFVideoFormat_NV12 };
    let mut activates: *mut Option<IMFActivate> = std::ptr::null_mut();
    let mut count: u32 = 0;
    // Hardware first, then anything local and synchronous. SORTANDFILTER puts the
    // preferred (and driver-blessed) transform first, which is the same order the OS
    // media pipeline itself would pick.
    let flags = MFT_ENUM_FLAG(
        MFT_ENUM_FLAG_HARDWARE.0
            | MFT_ENUM_FLAG_SYNCMFT.0
            | MFT_ENUM_FLAG_LOCALMFT.0
            | MFT_ENUM_FLAG_SORTANDFILTER.0,
    );
    // SAFETY: both type-info structs outlive the call, and the out-pointers are ours.
    unsafe {
        MFTEnumEx(
            MFT_CATEGORY_VIDEO_DECODER,
            flags,
            Some(&input_info),
            Some(&output_info),
            &mut activates,
            &mut count,
        )
        .map_err(|e| Error::NoDecoder(format!("MFTEnumEx failed: {e}")))?;
    }
    if activates.is_null() || count == 0 {
        return Err(Error::no_decoder(
            "no H.264 decoder MFT is registered (an N edition without the Media Feature Pack?)",
        ));
    }
    // SAFETY: MFTEnumEx allocated `count` entries with CoTaskMemAlloc and handed us
    // ownership of both the array and the interface references inside it.
    let list = unsafe { std::slice::from_raw_parts(activates, count as usize) };
    let mut chosen: Option<IMFTransform> = None;
    let mut last_error = String::new();
    for activate in list.iter().flatten() {
        match unsafe { activate.ActivateObject::<IMFTransform>() } {
            Ok(t) if chosen.is_none() => chosen = Some(t),
            Ok(_) => {}
            Err(e) => last_error = e.to_string(),
        }
    }
    // Release the enumeration: the interfaces we did not keep, then the array itself.
    for slot in list {
        drop(slot.clone());
    }
    unsafe { windows::Win32::System::Com::CoTaskMemFree(Some(activates as *const _)) };

    chosen.ok_or_else(|| {
        Error::no_decoder(format!("no H.264 decoder MFT could be activated: {last_error}"))
    })
}

/// An `IMFSample` holding one empty buffer of `size` bytes.
fn alloc_sample(size: u32) -> Result<IMFSample> {
    // SAFETY: plain MF factory calls; both objects are reference counted by the bindings.
    unsafe {
        let sample = MFCreateSample().map_err(|e| mf_err("MFCreateSample", e))?;
        let buffer = MFCreateMemoryBuffer(size.max(1)).map_err(|e| mf_err("MFCreateMemoryBuffer", e))?;
        sample.AddBuffer(&buffer).map_err(|e| mf_err("IMFSample::AddBuffer", e))?;
        Ok(sample)
    }
}

/// Copy the visible region of an NV12 buffer into `frame`.
///
/// `pitch` is the source stride, `base` points at its first row, and `coded_height` is how
/// many luma rows the buffer holds - which is NOT the visible height: a decoder handed
/// 1080p writes 1088 rows and puts the chroma plane below all of them.
///
/// A negative pitch means a bottom-up buffer, which the H.264 decoder never produces for
/// NV12; it is refused rather than guessed at, because guessing it wrong flips the picture.
fn copy_nv12(
    frame: &mut Frame,
    base: *const u8,
    pitch: i32,
    length: usize,
    coded_height: usize,
) -> Result<()> {
    if base.is_null() {
        return Err(Error::platform("IMFMediaBuffer::Lock", 0, "locked a null buffer"));
    }
    if pitch <= 0 {
        return Err(Error::unsupported(format!(
            "Media Foundation returned a {pitch}-byte NV12 pitch (bottom-up buffer)"
        )));
    }
let pitch = pitch as usize;
let (coded_h, visible_w, visible_h) = (coded_height, frame.width as usize, frame.height as usize);
    let chroma_rows = visible_h.div_ceil(2);
    let needed = pitch * coded_h + pitch * chroma_rows;
    if length < needed {
        return Err(Error::platform(
            "IMFMediaBuffer::Lock",
            0,
            format!("NV12 buffer is {length} bytes, needs {needed}"),
        ));
    }
    // SAFETY: `length` bytes are mapped at `base` for as long as the buffer is locked,
    // and the check above proves the rows read below all sit inside them.
    let src = unsafe { std::slice::from_raw_parts(base, length) };

    let y = frame.planes[0];
    let dst = frame.plane_mut(0);
    for r in 0..visible_h {
        dst[r * y.stride..r * y.stride + visible_w]
            .copy_from_slice(&src[r * pitch..r * pitch + visible_w]);
    }
    // The chroma plane starts after the CODED height of luma rows, not the visible one:
    // an MFT decoding 1080p hands back 1088 luma rows and puts chroma below all of them.
    let uv_base = pitch * coded_h;
    let uv = frame.planes[1];
    let uv_bytes = uv.row_bytes;
    let dst = frame.plane_mut(1);
    for r in 0..chroma_rows {
        let from = uv_base + r * pitch;
        dst[r * uv.stride..r * uv.stride + uv_bytes]
            .copy_from_slice(&src[from..from + uv_bytes]);
    }
    Ok(())
}

fn mf_err(call: &'static str, e: windows::core::Error) -> Error {
    Error::Platform { call, code: e.code().0 as i64, detail: e.message() }
}

/// Media Foundation's `MFSampleExtension_CleanPoint` attribute, which the bindings do not
/// export as a constant in every version.
const SAMPLE_EXTENSION_CLEAN_POINT: GUID =
    GUID::from_u128(0x9cdf01d8_a0f0_43ba_b077_eaa06cbd728a);
