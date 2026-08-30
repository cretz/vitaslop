//! Video decoding, as the engine sees it.
//!
//! The Vita decodes H.264 in fixed-function hardware, and so does every host this engine
//! runs on - so the engine's job is not to decode but to REACH the host's decoder.
//! [`vitaslop_h264`] does that for all four platforms; this module is the seam that lets
//! the engine use it without knowing which one it is on.
//!
//! # Why it is a seam and not a direct call
//!
//! One thing differs between the hosts and it is not cosmetic: on native, decoding happens
//! inside the call; in the browser, WebCodecs delivers frames on a callback, and a guest
//! host call runs on a suspended stack that cannot await. So the engine cannot ask for
//! "the frame for this access unit" and get it. What it can do - what this seam offers - is
//! SUBMIT and later POLL, which is exactly how the guest's own decoder behaves too: the
//! hardware takes an access unit and raises an interrupt when the picture is ready.

#[cfg(feature = "video")]
use std::collections::VecDeque;

#[cfg(feature = "video")]
pub use vitaslop_h264::{Acceleration, ColorInfo, Frame, PixelFormat};

/// The layout a [`DecodedPicture`] is in.
///
/// A decoder produces what suits it, and the three that appear here are all of them: 4:2:0
/// in one interleaved chroma plane, 4:2:0 in two separate ones, and packed RGBA (which a
/// browser's decoder is entitled to hand back, and one phone's does).
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum PictureFormat {
    /// Luma plane, then one interleaved Cb/Cr plane at half resolution in both axes.
    Nv12,
    /// Luma, Cb, Cr as three separate planes.
    I420,
    /// One packed plane of R, G, B, A.
    Rgba,
}

/// One decoded picture, in terms that carry no decoder type with them.
///
/// The engine holds video decoding through [`VideoDecode`], and the point of that trait is
/// that the engine does not link a decoder - so what crosses it cannot be the decoder
/// crate's own frame. This is a plain owned picture, tightly packed.
///
/// # It carries its own format rather than being normalised to one
///
/// Normalising here was the obvious thing and it was waste: this used to convert every
/// picture to I420, and the consumer then converted I420 to whatever the guest asked for.
/// A decoder that produced NV12 and a guest that wanted NV12 - which is the ordinary case
/// on three of the four backends - paid for TWO conversions of every pixel to arrive back
/// where it started. The consumer converts once, or not at all.
#[derive(Debug, Clone)]
pub struct DecodedPicture {
    /// Visible width in luma samples.
    pub width: u32,
    /// Visible height in luma samples.
    pub height: u32,
    /// Presentation timestamp, in whatever units were submitted with the access unit.
    pub pts: i64,
    /// What [`DecodedPicture::data`] is laid out as.
    pub format: PictureFormat,
    /// The picture, tightly packed in [`DecodedPicture::format`].
    pub data: Vec<u8>,
}

impl Default for DecodedPicture {
    fn default() -> Self {
        DecodedPicture {
            width: 0,
            height: 0,
            pts: 0,
            format: PictureFormat::I420,
            data: Vec::new(),
        }
    }
}

impl DecodedPicture {
    /// Byte offset of the chroma plane (both planes, for NV12) within the data.
    pub fn chroma_offset(&self) -> usize {
        (self.width * self.height) as usize
    }

    /// Byte offset of the Cr plane, for a three-plane picture.
    pub fn cr_offset(&self) -> usize {
        self.chroma_offset() + (self.width.div_ceil(2) * self.height.div_ceil(2)) as usize
    }
}

/// Opens decoders. The engine holds one of these rather than a decoder, because a movie
/// needs a decoder of its own and the engine must be able to make one without linking any.
///
/// The default is [`NoVideo`], which reports that this host decodes no video - a normal
/// outcome, not a failure: a title whose movie will not play should carry on to the part
/// that is a game.
pub trait VideoDecodeFactory: Send {
    /// Open a decoder for an H.264 stream described by an avcC record, whose samples
    /// arrive with `length_size`-byte length prefixes.
    fn open_h264(
        &mut self,
        avcc: &[u8],
        length_size: usize,
    ) -> Result<Box<dyn VideoDecode>, VideoError>;

    /// Open a decoder for an Annex B stream: start-code delimited NALs that carry their
    /// own parameter sets.
    ///
    /// This is what a guest video-decoder API is given - it is handed elementary stream
    /// and knows nothing about a container - so it is a separate entry point rather than
    /// an empty avcC through the one above, which would read as "no parameter sets" and
    /// be a different thing entirely.
    fn open_h264_annex_b(&mut self) -> Result<Box<dyn VideoDecode>, VideoError>;
}

/// The default factory: this host has no video decoder.
#[derive(Debug, Default, Clone, Copy)]
pub struct NoVideo;

impl VideoDecodeFactory for NoVideo {
    fn open_h264(&mut self, _avcc: &[u8], _length_size: usize) -> Result<Box<dyn VideoDecode>, VideoError> {
        Err(VideoError::NoDecoder(
            "this host was started without a video decoder".to_string(),
        ))
    }

    fn open_h264_annex_b(&mut self) -> Result<Box<dyn VideoDecode>, VideoError> {
        Err(VideoError::NoDecoder(
            "this host was started without a video decoder".to_string(),
        ))
    }
}

/// A factory backed by [`vitaslop_h264`], i.e. by whatever decoder the platform has.
#[cfg(feature = "video")]
#[derive(Debug, Default, Clone, Copy)]
pub struct H264Factory;

#[cfg(feature = "video")]
impl VideoDecodeFactory for H264Factory {
    fn open_h264(
        &mut self,
        avcc: &[u8],
        length_size: usize,
    ) -> Result<Box<dyn VideoDecode>, VideoError> {
        off_thread(VideoDecoder::from_avcc(avcc, length_size)?)
    }

    fn open_h264_annex_b(&mut self) -> Result<Box<dyn VideoDecode>, VideoError> {
        off_thread(VideoDecoder::annex_b()?)
    }
}

/// >>> THE DECODE DOES NOT BELONG ON THE GUEST'S THREAD, and on this platform it was.
///
/// MEASURED on a movie frame of one title with `bench --at`, native: `sceAvcdecDecode` was
/// **199 ms of a 302 ms window - 66% of the guest's whole frame - at 1.66 ms a call**, one
/// call per frame. That is not the guest computing anything: it is the platform decoder
/// running, and the guest's own thread standing still while it does.
///
/// On hardware the decode happens in a fixed-function block while the calling thread is
/// descheduled, and in a BROWSER it already works that way (WebCodecs answers on a
/// callback, which is why the seam is submit/poll and not decode). So the native backends
/// are the odd ones out: they decode inside `submit`. This wrapper puts them on a thread of
/// their own, which makes every host the same shape as the hardware and as each other -
/// and means the desktop is no longer measuring a cost the device does not have.
///
/// wasm gets the decoder unwrapped: there is no thread to move it to, and its backend is
/// asynchronous already.
#[cfg(all(feature = "video", not(target_arch = "wasm32")))]
fn off_thread(decoder: VideoDecoder) -> Result<Box<dyn VideoDecode>, VideoError> {
    Ok(Box::new(ThreadedDecode::spawn(decoder)))
}

#[cfg(all(feature = "video", target_arch = "wasm32"))]
fn off_thread(decoder: VideoDecoder) -> Result<Box<dyn VideoDecode>, VideoError> {
    Ok(Box::new(decoder))
}

/// Video decoding, as the ENGINE declares it.
///
/// The engine (`vitaslop-runtime`) compiles to wasm and links no decoder: it holds one of
/// these, supplied by whichever front-end started it. That is the same arrangement the
/// renderer and the audio sink use, and it is what keeps a JS binding out of the engine.
pub trait VideoDecode: Send {
    /// Submit one access unit with its presentation timestamp.
    ///
    /// The picture is NOT returned here. In the browser it does not exist yet - WebCodecs
    /// answers on a callback - and a trait that promised otherwise would work on the
    /// desktop and stall in a worker.
    fn submit(&mut self, sample: &[u8], pts: i64) -> Result<(), VideoError>;

    /// Take the next decoded picture, if one has arrived.
    fn poll(&mut self) -> Result<Option<DecodedPicture>, VideoError>;

    /// No more input is coming: decode whatever is still held back.
    fn finish(&mut self) -> Result<(), VideoError>;

    /// True while pictures are still owed for input already given. Always a "may yet
    /// arrive", never a promise - see [`VideoDecoder::owes_frames`].
    fn owes_frames(&self) -> bool;

    /// Discard everything for a seek.
    fn reset(&mut self) -> Result<(), VideoError>;

    /// Which decoder this is, and whether it reaches video hardware - for the one line a
    /// run should print about how its video was decoded.
    fn describe(&self) -> String;
}

#[cfg(feature = "video")]
impl VideoDecode for VideoDecoder {
    fn submit(&mut self, sample: &[u8], pts: i64) -> Result<(), VideoError> {
        VideoDecoder::submit(self, sample, pts)
    }

    fn poll(&mut self) -> Result<Option<DecodedPicture>, VideoError> {
        let Some(frame) = VideoDecoder::poll(self)? else {
            return Ok(None);
        };
        let mut picture = DecodedPicture {
            width: frame.width,
            height: frame.height,
            pts: frame.pts,
            format: match frame.format {
                PixelFormat::Nv12 => PictureFormat::Nv12,
                PixelFormat::I420 => PictureFormat::I420,
                PixelFormat::Rgba => PictureFormat::Rgba,
            },
            data: Vec::new(),
        };
        // The frame's OWN layout - see `DecodedPicture`. No conversion happens here.
        frame.copy_packed(&mut picture.data);
        // The frame's buffer goes straight back into the pool; the picture owns its own.
        VideoDecoder::recycle(self, frame);
        Ok(Some(picture))
    }

    fn finish(&mut self) -> Result<(), VideoError> {
        VideoDecoder::finish(self)
    }

    fn owes_frames(&self) -> bool {
        VideoDecoder::owes_frames(self)
    }

    fn reset(&mut self) -> Result<(), VideoError> {
        VideoDecoder::reset(self)
    }

    fn describe(&self) -> String {
        match self.inner.backend_detail() {
            Some(detail) => {
                format!("{} [{:?}] - {detail}", self.backend_name(), self.acceleration())
            }
            None => format!("{} [{:?}]", self.backend_name(), self.acceleration()),
        }
    }
}

/// A decoder driven from a thread of its own - see [`off_thread`].
///
/// # What it does NOT change
/// The seam is already "submit now, poll later", and every caller already treats a poll that
/// answers nothing as ordinary ([`VideoDecode::owes_frames`] is documented as "may yet
/// arrive"). So this changes WHEN a picture appears, never whether one does, and the guest
/// parks in exactly the case it was already parked in.
///
/// # >>> AND NEITHER SIDE MAY EVER BLOCK ON THE OTHER, which the first version got wrong
///
/// It used bounded channels both ways, on the reasoning that a caller which stops polling
/// should not be able to make the worker allocate a movie's worth of pictures. What that
/// actually built was a deadlock: a title that stops asking for pictures - because the movie
/// ended, or because it moved to another screen - leaves the worker blocked in `send`, which
/// leaves the work queue full, which blocks the GUEST's next `submit` for ever. MEASURED as a
/// run that sat at 24 seconds of CPU over 25 minutes of wall clock, doing nothing at all.
///
/// So the queues are unbounded and the BOUND is enforced where it costs nothing: the worker
/// keeps at most [`PICTURE_BACKLOG`] undelivered pictures and drops the OLDEST beyond that,
/// which is what a decoder whose output nobody is collecting should do. The drops are counted
/// and reported, because a silently shortened movie is exactly the kind of thing that would
/// otherwise be found on a device.
#[cfg(all(feature = "video", not(target_arch = "wasm32")))]
pub struct ThreadedDecode {
    work: Option<std::sync::mpsc::Sender<Job>>,
    shared: std::sync::Arc<std::sync::Mutex<Answers>>,
    worker: Option<std::thread::JoinHandle<()>>,
    /// Access units handed over and not yet answered. The whole of `owes_frames` - and it is
    /// deliberately maintained HERE rather than asked of the worker: a question asked across a
    /// channel is a question about the past, and this one decides whether the caller parks.
    outstanding: usize,
    /// Bumped by a reset. A picture decoded before the reset carries the old value and is
    /// dropped on arrival, which is what makes a seek exact without a round trip.
    epoch: u64,
}

/// How many decoded pictures the worker may hold for a caller that is not collecting them.
/// Deep enough that no decoder's own pipeline is ever the reason one is dropped.
#[cfg(all(feature = "video", not(target_arch = "wasm32")))]
const PICTURE_BACKLOG: usize = 8;

/// What the calling thread asks the decoder thread to do.
#[cfg(all(feature = "video", not(target_arch = "wasm32")))]
enum Job {
    Submit { sample: Vec<u8>, pts: i64, epoch: u64 },
    Finish { epoch: u64 },
    Reset { epoch: u64 },
}

/// What the worker has to hand back. Shared rather than sent so that neither side can be made
/// to wait by the other - see [`ThreadedDecode`].
#[cfg(all(feature = "video", not(target_arch = "wasm32")))]
#[derive(Default)]
struct Answers {
    /// Decoded pictures, oldest first, each with the epoch it was decoded under.
    pictures: VecDeque<(u64, DecodedPicture)>,
    /// Inputs the worker has finished with, per epoch it retired them under.
    retired: Vec<(u64, usize)>,
    /// The first error the worker reported, held until a caller asks for a picture.
    failed: Option<String>,
    /// Pictures dropped because the backlog was full - a movie whose output nobody collected.
    dropped: u64,
    /// What the decoder turned out to BE, which on some backends is not knowable until it has
    /// decoded something.
    detail: String,
}

#[cfg(all(feature = "video", not(target_arch = "wasm32")))]
impl ThreadedDecode {
    fn spawn(mut decoder: VideoDecoder) -> ThreadedDecode {
        let (work_tx, work_rx) = std::sync::mpsc::channel::<Job>();
        let shared = std::sync::Arc::new(std::sync::Mutex::new(Answers {
            detail: decoder.describe(),
            ..Answers::default()
        }));
        let worker_shared = shared.clone();
        let worker = std::thread::Builder::new()
            .name("vitaslop-video".to_string())
            .spawn(move || {
                while let Ok(job) = work_rx.recv() {
                    // Only a SUBMIT is an input the caller is owed an answer for: it is the
                    // only job that incremented `outstanding`, and retiring a `Finish` against
                    // that count would say "nothing is owed" while a picture is still coming -
                    // which is precisely the un-parked spin this whole path exists to avoid.
                    let (epoch, owed, result) = match job {
                        Job::Submit { sample, pts, epoch } => {
                            (epoch, true, VideoDecode::submit(&mut decoder, &sample, pts))
                        }
                        Job::Finish { epoch } => (epoch, false, VideoDecode::finish(&mut decoder)),
                        Job::Reset { epoch } => (epoch, false, VideoDecode::reset(&mut decoder)),
                    };
                    let mut answers = match worker_shared.lock() {
                        Ok(a) => a,
                        // The caller's side has panicked; there is nothing to hand back to.
                        Err(_) => return,
                    };
                    if let Err(e) = result {
                        answers.failed.get_or_insert(e.to_string());
                        continue;
                    }
                    drop(answers);
                    loop {
                        let picture = match VideoDecode::poll(&mut decoder) {
                            Ok(Some(p)) => p,
                            Ok(None) => break,
                            Err(e) => {
                                if let Ok(mut a) = worker_shared.lock() {
                                    a.failed.get_or_insert(e.to_string());
                                }
                                break;
                            }
                        };
                        let Ok(mut a) = worker_shared.lock() else { return };
                        a.pictures.push_back((epoch, picture));
                        while a.pictures.len() > PICTURE_BACKLOG {
                            a.pictures.pop_front();
                            a.dropped += 1;
                        }
                    }
                    let detail = VideoDecode::describe(&decoder);
                    let Ok(mut a) = worker_shared.lock() else { return };
                    a.detail = detail;
                    if owed {
                        match a.retired.iter_mut().find(|(e, _)| *e == epoch) {
                            Some((_, n)) => *n += 1,
                            None => a.retired.push((epoch, 1)),
                        }
                    }
                }
            })
            .expect("a decoder thread");
        ThreadedDecode {
            work: Some(work_tx),
            shared,
            worker: Some(worker),
            outstanding: 0,
            epoch: 0,
        }
    }

    fn send(&mut self, job: Job) -> Result<(), VideoError> {
        match self.work.as_ref() {
            Some(tx) => tx.send(job).map_err(|_| {
                VideoError::Stream("the video decoder thread has stopped".to_string())
            }),
            None => Err(VideoError::Stream("the video decoder is closed".to_string())),
        }
    }
}

#[cfg(all(feature = "video", not(target_arch = "wasm32")))]
impl VideoDecode for ThreadedDecode {
    fn submit(&mut self, sample: &[u8], pts: i64) -> Result<(), VideoError> {
        let epoch = self.epoch;
        self.send(Job::Submit { sample: sample.to_vec(), pts, epoch })?;
        self.outstanding += 1;
        Ok(())
    }

    fn poll(&mut self) -> Result<Option<DecodedPicture>, VideoError> {
        let (picture, failed) = {
            let Ok(mut a) = self.shared.lock() else {
                return Err(VideoError::Stream("the video decoder thread panicked".to_string()));
            };
            // Everything the worker retired under the current epoch is no longer owed.
            if let Some(i) = a.retired.iter().position(|(e, _)| *e == self.epoch) {
                let (_, n) = a.retired.remove(i);
                self.outstanding = self.outstanding.saturating_sub(n);
            }
            a.retired.retain(|(e, _)| *e >= self.epoch);
            // A picture from before a reset describes a stream position the guest has left.
            while a.pictures.front().is_some_and(|(e, _)| *e != self.epoch) {
                a.pictures.pop_front();
            }
            if a.dropped > 0 {
                let dropped = std::mem::take(&mut a.dropped);
                report_pictures_dropped(dropped);
            }
            (a.pictures.pop_front().map(|(_, p)| p), a.failed.take())
        };
        match (picture, failed) {
            (Some(p), _) => Ok(Some(p)),
            (None, Some(e)) => Err(VideoError::Stream(e)),
            (None, None) => Ok(None),
        }
    }

    fn finish(&mut self) -> Result<(), VideoError> {
        let epoch = self.epoch;
        self.send(Job::Finish { epoch })
    }

    fn owes_frames(&self) -> bool {
        self.outstanding > 0
    }

    fn reset(&mut self) -> Result<(), VideoError> {
        self.epoch += 1;
        self.outstanding = 0;
        if let Ok(mut a) = self.shared.lock() {
            a.pictures.clear();
            a.retired.clear();
        }
        self.send(Job::Reset { epoch: self.epoch })
    }

    fn describe(&self) -> String {
        self.shared.lock().map(|a| a.detail.clone()).unwrap_or_default()
    }
}

#[cfg(all(feature = "video", not(target_arch = "wasm32")))]
impl Drop for ThreadedDecode {
    fn drop(&mut self) {
        // Dropping the sender ends the worker's `recv` loop; the join then makes the
        // decoder's own teardown (a COM shutdown, a Video Toolbox session) happen before this
        // returns rather than at some point after the movie is gone.
        self.work = None;
        if let Some(worker) = self.worker.take() {
            let _ = worker.join();
        }
    }
}

/// Say, once and unconditionally, that decoded pictures were thrown away because nobody
/// collected them. A movie that quietly loses frames is a fallback like any other.
#[cfg(all(feature = "video", not(target_arch = "wasm32")))]
fn report_pictures_dropped(dropped: u64) {
    static ONCE: std::sync::Once = std::sync::Once::new();
    ONCE.call_once(|| {
        eprintln!(
            "vitaslop video: {dropped} decoded picture(s) were dropped because the title \
             stopped collecting them and the decoder's backlog was full. The decode is \
             correct and the movie is short by that many frames."
        );
    });
}

/// A decoder for one video stream.
///
/// Cheap to create and to drop: a title that plays several movies makes one of these per
/// movie, which is also what keeps a stream's parameter sets from leaking into the next.
#[cfg(feature = "video")]
pub struct VideoDecoder {
    inner: vitaslop_h264::Decoder,
    ready: VecDeque<Frame>,
    /// Access units submitted but not yet answered with a frame. A decoder is pipelined,
    /// so this is normally non-zero during playback - it is how [`VideoDecoder::owes_frames`]
    /// tells "still working" from "nothing to wait for".
    outstanding: usize,
    /// Set once the stream's end has been signalled, so [`VideoDecoder::poll`] drains.
    finishing: bool,
}

/// Why a decoder could not be created.
#[derive(Debug)]
pub enum VideoError {
    /// This machine has no H.264 decoder at all. The caller is expected to carry on
    /// without video rather than fail - a movie is not a game.
    NoDecoder(String),
    /// The stream, or something in it, is not decodable here.
    Stream(String),
}

impl core::fmt::Display for VideoError {
    fn fmt(&self, f: &mut core::fmt::Formatter<'_>) -> core::fmt::Result {
        match self {
            VideoError::NoDecoder(m) => write!(f, "no video decoder on this host: {m}"),
            VideoError::Stream(m) => write!(f, "video stream cannot be decoded: {m}"),
        }
    }
}

#[cfg(feature = "video")]
impl From<vitaslop_h264::Error> for VideoError {
    fn from(e: vitaslop_h264::Error) -> Self {
        if e.is_missing_decoder() {
            VideoError::NoDecoder(e.to_string())
        } else {
            VideoError::Stream(e.to_string())
        }
    }
}

#[cfg(feature = "video")]
impl VideoDecoder {
    /// Build a decoder for a stream whose parameter sets are the given avcC record, with
    /// samples arriving length-prefixed - which is what an MP4 track carries.
    pub fn from_avcc(avcc: &[u8], length_size: usize) -> Result<VideoDecoder, VideoError> {
        let config = vitaslop_h264::DecoderConfig {
            input: vitaslop_h264::InputFormat::LengthPrefixed { length_size },
            extradata: Some(avcc.to_vec()),
            // An MP4 sample is exactly one access unit, so a picture can be submitted as
            // soon as its sample arrives instead of waiting for the next one to prove it
            // ended. For a movie playing under a title's own clock that is a whole frame
            // of latency saved.
            packets_are_access_units: Some(true),
            ..vitaslop_h264::DecoderConfig::default()
        };
        Ok(VideoDecoder {
            inner: vitaslop_h264::Decoder::new(config)?,
            ready: VecDeque::new(),
            outstanding: 0,
            finishing: false,
        })
    }

    /// Build a decoder for an Annex B elementary stream: start-code delimited NALs whose
    /// parameter sets arrive in band, which is what a guest hands a video-decoder API.
    ///
    /// One packet is one access unit here for the same reason it is above: the caller is
    /// a demuxer's consumer and submits whole frames, so nothing has to wait for the next
    /// packet to prove the last one ended.
    pub fn annex_b() -> Result<VideoDecoder, VideoError> {
        let config = vitaslop_h264::DecoderConfig {
            input: vitaslop_h264::InputFormat::AnnexB,
            packets_are_access_units: Some(true),
            // A guest video-decode API is a fixed-function block's interface: the caller
            // submits one access unit and expects the picture for it, and a decoder that
            // fills a pipeline first never lets that caller start. See
            // [`vitaslop_h264::DecoderConfig::low_latency`] for what it costs.
            low_latency: true,
            ..vitaslop_h264::DecoderConfig::default()
        };
        Ok(VideoDecoder {
            inner: vitaslop_h264::Decoder::new(config)?,
            ready: VecDeque::new(),
            outstanding: 0,
            finishing: false,
        })
    }

    /// Which platform decoder is behind this, for diagnostics.
    pub fn backend_name(&self) -> &'static str {
        self.inner.backend_name()
    }

    /// Whether decoding is reaching video hardware.
    pub fn acceleration(&self) -> Acceleration {
        self.inner.acceleration()
    }

    /// Submit one access unit (one MP4 sample) with its presentation timestamp.
    ///
    /// This does NOT return the frame: on the browser the picture is not ready yet, and a
    /// seam that pretended otherwise would work on the desktop and stall in a worker.
    pub fn submit(&mut self, sample: &[u8], pts: i64) -> Result<(), VideoError> {
        self.inner.send(vitaslop_h264::Packet::with_pts(sample, pts))?;
        self.outstanding += 1;
        self.collect()
    }

    /// No more input is coming: decode whatever is still held back.
    pub fn finish(&mut self) -> Result<(), VideoError> {
        self.finishing = true;
        self.inner.finish()?;
        self.collect()
    }

    /// Take the next decoded picture, if one has arrived.
    pub fn poll(&mut self) -> Result<Option<Frame>, VideoError> {
        self.collect()?;
        Ok(self.ready.pop_front())
    }

    /// True while the decoder still owes pictures for input it has been given. A caller
    /// that must wait for a frame waits while this holds and gives up when it does not -
    /// which is what keeps a dropped or duplicate access unit from wedging a guest thread.
    ///
    /// It is a "may yet arrive", not a promise: a decoder is allowed to drop a picture it
    /// cannot decode (a stream joined mid-GOP), and then this stays true for that access
    /// unit until the next one is answered. So a caller waits on it with a bound, never
    /// forever.
    #[doc(alias = "owes")]
    pub fn owes_frames(&self) -> bool {
        self.outstanding > 0 || self.finishing
    }

    /// Hand a frame's buffer back for reuse, which is what keeps steady playback from
    /// allocating a picture-sized block per frame.
    pub fn recycle(&mut self, frame: Frame) {
        self.inner.recycle(frame);
    }

    /// Discard everything for a seek.
    pub fn reset(&mut self) -> Result<(), VideoError> {
        self.inner.reset()?;
        self.ready.clear();
        self.outstanding = 0;
        self.finishing = false;
        Ok(())
    }

    fn collect(&mut self) -> Result<(), VideoError> {
        while let Some(frame) = self.inner.receive()? {
            self.outstanding = self.outstanding.saturating_sub(1);
            self.ready.push_back(frame);
        }
        // A flush is only outstanding until everything it was flushing has arrived.
        if self.outstanding == 0 {
            self.finishing = false;
        }
        Ok(())
    }
}

#[cfg(feature = "video")]
impl core::fmt::Debug for VideoDecoder {
    fn fmt(&self, f: &mut core::fmt::Formatter<'_>) -> core::fmt::Result {
        f.debug_struct("VideoDecoder")
            .field("backend", &self.inner.backend_name())
            .field("ready", &self.ready.len())
            .field("outstanding", &self.outstanding)
            .finish()
    }
}

#[cfg(all(test, feature = "video"))]
mod tests {
    use super::*;

    /// The seam decodes a stream end to end, using the H.264 crate's own synthetic
    /// conformance stream so the expected pixels are known exactly.
    #[test]
    fn decodes_a_stream_through_the_seam() {
        let stream = vitaslop_h264::synth::synthesize(8, 6, 4);

        // Repackage the Annex B stream as an MP4 track would carry it: an avcC record plus
        // length-prefixed samples. That is the shape this seam takes.
        let mut splitter = vitaslop_h264::bitstream::AuSplitter::new();
        let mut units = Vec::new();
        splitter.push_annex_b(&stream.annex_b, &mut units).unwrap();
        splitter.finish(&mut units).unwrap();
        let record = vitaslop_h264::bitstream::avcc::AvcC::from_parameter_sets(
            splitter.sets.sps_nals(),
            splitter.sets.pps_nals(),
            4,
        )
        .unwrap();

        let mut decoder = match VideoDecoder::from_avcc(&record.to_bytes(), 4) {
            Ok(d) => d,
            Err(VideoError::NoDecoder(_)) => return, // no decoder on this machine
            Err(e) => panic!("{e}"),
        };

        let mut decoded = Vec::new();
        let mut packed = Vec::new();
        for (index, unit) in units.iter().enumerate() {
            let mut sample = Vec::new();
            vitaslop_h264::bitstream::avcc::annex_b_to_length_prefixed(&unit.data, 4, &mut sample);
            decoder.submit(&sample, index as i64).unwrap();
            while let Some(frame) = decoder.poll().unwrap() {
                frame.copy_to_i420(&mut packed);
                decoded.push(packed.clone());
                decoder.recycle(frame);
            }
        }
        decoder.finish().unwrap();
        while let Some(frame) = decoder.poll().unwrap() {
            frame.copy_to_i420(&mut packed);
            decoded.push(packed.clone());
            decoder.recycle(frame);
        }

        assert_eq!(decoded.len(), stream.frames.len());
        for (index, (got, expected)) in decoded.iter().zip(&stream.frames).enumerate() {
            assert_eq!(got, expected, "frame {index} differs through the platform seam");
        }
        assert!(!decoder.owes_frames(), "every submitted access unit was answered");
    }
}
