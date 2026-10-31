//! Queue uploads: the per-frame and per-run write registers, the device budget, and the
//! stall census a blocked `write_buffer` reports into.

use super::*;

/// >>> THE WORST `queue.write_buffer` OF THE WHOLE RUN, packed as `us << 32 | bytes`.
///
/// # A WINDOWED MAXIMUM CANNOT BE READ WHEN THE BAD FRAME HANGS THE BROWSER
/// `EncodeWork::buffer_write_worst` is a maximum over the REPORTING WINDOW, which is the
/// right thing for "what is this frame doing now" and the wrong thing for the case that
/// actually matters here. The user's report: *"often hangs/hiccups actually hang the browser
/// so you can't copy at that time"*. So the sequence is - the stall happens, the page is
/// unresponsive, it recovers, and only THEN can a dump be taken - by which point the window
/// holding the stall's worst call has rolled over and the number is gone. Asking for a dump
/// "at the moment it hitches" is asking for something the platform does not allow.
///
/// This one never resets, so a dump taken after recovery still carries the worst single
/// write the run ever made. It sits beside the `SLOWEST FRAMES, cumulative for the run`
/// list, which survives for exactly the same reason and is exactly the pairing that makes a
/// post-hoc dump worth taking. [[vitaslop-a-count-needs-its-window]]
/// >>> THE WORST `queue.write_buffer` SINCE SOMEBODY LAST ASKED, in MICROSECONDS.
///
/// # A THROTTLE MUST BE ARMED BY THE THING IT EXISTS TO PREVENT
/// The queue-depth bound ([`crate::gpu`]'s caller, `LivePlayback::gpu_in_flight`) was armed
/// unconditionally and it throttled a HEALTHY run to a third of its rate. MEASURED on the
/// user's phone, a gameplay window after the empty-pass fix: GPU execution **3.9 ms** a
/// frame against a 19.4 ms period, `arena write 0.2 ms`, worst write of the whole run
/// **0.6 ms for 288 KB** - and `908 presents DECLINED`, `0.35 presents/tick`, *"95% of the
/// frames the emulator computed were DISCARDED unpresented"*. Nothing was backed up.
///
/// What misled the bound is that `on_submitted_work_done` reads **897 ms** on that same
/// window. With a timestamp query saying the GPU did 3.9 ms of work, that number cannot be
/// GPU backlog: it is callback dispatch latency, which the panel's own text always warned
/// it could not separate. So the promise is not a usable depth signal on this device, and a
/// bound gated on it declines presents for a queue that does not exist.
///
/// This is the signal that IS the pathology: a `write_buffer` that BLOCKS. No copy of a few
/// hundred KB costs milliseconds, so a large value here means the thread is waiting on the
/// staging ring to retire - the exact failure the bound removes. Read and cleared by
/// [`take_worst_write_us`], so the caller sees the worst since its last frame.
pub(super) static BUFFER_WRITE_WORST_SINCE_READ: std::sync::atomic::AtomicU64 =
    std::sync::atomic::AtomicU64::new(0);

/// The worst `queue.write_buffer` in microseconds since the last call, and zero the
/// register. See [`BUFFER_WRITE_WORST_SINCE_READ`].
pub fn take_worst_write_us() -> u64 {
    BUFFER_WRITE_WORST_SINCE_READ.swap(0, std::sync::atomic::Ordering::Relaxed)
}

pub(super) static BUFFER_WRITE_WORST_RUN: std::sync::atomic::AtomicU64 =
    std::sync::atomic::AtomicU64::new(0);

/// >>> WHICH WRITE OF THE FRAME THIS IS - the number that separates the two live stories.
///
/// If `write_buffer` is blocking because it needs a staging chunk that only completion of
/// the PREVIOUS SUBMIT can free, the stalls land on the FIRST write of a frame and the rest
/// run at microseconds. If instead the queue is simply saturated, they land anywhere. Those
/// have different fixes - rotating the arenas over more buffers versus reducing GPU work -
/// and no total, no maximum and no byte count can tell them apart. Reset per frame at the
/// top of `encode_chain`.
pub(super) static BUFFER_WRITES_THIS_FRAME: std::sync::atomic::AtomicU64 =
    std::sync::atomic::AtomicU64::new(0);

/// >>> TEXTURE UPLOADED IN *THIS FRAME*, for the stall report.
///
/// The user's observation, which is worth more than my theory was: the stall happens on a
/// NEW thing - a new course, a new menu - which is precisely when a frame first meets
/// textures it has never uploaded. That covers EVERY texture path, not just the BC
/// expansion I tested and wrongly generalised from: `0x00`/`0x13`/`0x0c` are uncompressed
/// SWIZZLED formats that are un-swizzled and uploaded whether or not the adapter has BC, and
/// a device dump has 36.9 MB of exactly those. Switching off `NO_BC` never touched them, so
/// "expansion is refuted" was a statement about one path reported as if it were all of them.
///
/// Per FRAME rather than cumulative, because the question is what THIS frame put in the
/// queue in front of the write that stalled.
pub(super) static TEX_UPLOADS_THIS_FRAME: std::sync::atomic::AtomicU64 =
    std::sync::atomic::AtomicU64::new(0);
pub(super) static TEX_UPLOAD_BYTES_THIS_FRAME: std::sync::atomic::AtomicU64 =
    std::sync::atomic::AtomicU64::new(0);

/// >>> ARENA BUFFERS CREATED IN *THIS FRAME* - the untested candidate.
///
/// Two mechanisms are already refuted by measurement: the stalls are not the first write of
/// a frame (so not simply waiting on the previous submit), and the frame that uploaded 79
/// textures / 5.7 MB stalled for LESS time than two frames that uploaded 2 textures /
/// 601 KB (so not upload volume).
///
/// What both of those miss is that a NEW course or menu brings bigger geometry, which makes
/// a pass arena GROW: a fresh `create_buffer`, the old one destroyed, in the middle of a
/// frame. That matches the user's report of where the stall happens far better than
/// anything about steady-state volume, and nothing counts it.
pub(super) static BUFFERS_MADE_THIS_FRAME: std::sync::atomic::AtomicU64 =
    std::sync::atomic::AtomicU64::new(0);

/// Charge a texture upload to both the window tally and the current FRAME.
///
/// One function rather than two calls at six sites: the per-frame pair existing but being
/// updated at five of the six upload paths would be a counter that reads low for reasons
/// nobody could see, which is the failure this file keeps finding in its own instruments.
/// Charge one GPU buffer creation to the window tally and to the current FRAME.
///
/// Written AFTER the call sites were renamed, deliberately: doing it the other way round is
/// what turned `enc_tex_upload` into a call to itself a few minutes ago.
pub(super) fn enc_buffer_created() {
    enc(&ENC.buffers_created, 1);
    BUFFERS_MADE_THIS_FRAME.fetch_add(1, std::sync::atomic::Ordering::Relaxed);
}

/// >>> PIPELINES BUILT IN *THIS FRAME*, and it is the candidate the stall context was
/// >>> missing.
///
/// The stalls that matter have `0 submit(s) in flight` and one of them blocked 158 ms
/// writing ONE KILOBYTE. Neither bandwidth nor GPU backlog can produce that: an empty queue
/// and a kilobyte of data mean the client is waiting on the GPU PROCESS, not on the GPU.
/// The thing that makes a GPU process unresponsive for hundreds of milliseconds is a SHADER
/// COMPILE, and a frame that opens a new scene builds pipelines and creates buffers and
/// writes arenas all at once - so every stall so far has had a compile it could have been
/// queued behind, and nothing counted them.
pub(super) static PIPELINES_BUILT_THIS_FRAME: std::sync::atomic::AtomicU64 =
    std::sync::atomic::AtomicU64::new(0);

/// >>> GPU RESOURCES *DESTROYED* IN THIS FRAME, AND WHAT THAT COST ON THE CLOCK.
///
/// The stall that survives every other explanation is this one: an EMPTY queue (`0 submit(s)
/// in flight`, and that counter goes up at submit and only down on a callback, so it
/// over-counts if it is wrong), no textures uploaded, no pipeline built, one new buffer -
/// and a ONE KILOBYTE write blocks for 259 ms while the whole browser hitches, which is the
/// user's own description of it. A browser-wide freeze is the GPU PROCESS, and a kilobyte
/// is not why.
///
/// `destroy()` is the traffic nothing here has ever timed. Every frame retires last frame's
/// buffers and textures, and freeing GPU memory is exactly the kind of work that makes a
/// GPU process stop answering. These are counted and TIMED per frame so the stall line can
/// say whether the frame that blocked had just freed a pile of memory.
pub(super) static RESOURCES_DESTROYED_THIS_FRAME: std::sync::atomic::AtomicU64 =
    std::sync::atomic::AtomicU64::new(0);

/// Microseconds spent in `destroy()` this frame. See [`RESOURCES_DESTROYED_THIS_FRAME`].
pub(super) static DESTROY_US_THIS_FRAME: std::sync::atomic::AtomicU64 =
    std::sync::atomic::AtomicU64::new(0);

/// >>> MICROSECONDS SPENT IN `create_buffer` THIS FRAME, AND THE LAST SUSPECT STANDING.
///
/// Everything else the stalling frames do has now been timed and cleared: `destroy()` is
/// 0 ms, no textures are uploaded on two of them, and one has no pipeline build. What every
/// stall DOES have is a new buffer - 1, 17, 47 of them - and the worst of them is the FIRST
/// upload of its frame, immediately after one buffer was created.
///
/// A GPU buffer is allocated LAZILY: `create_buffer` records a descriptor, and the pages
/// are found when something first writes to it. That makes `write_buffer` the call that
/// pays for the allocation, which explains the one reading nothing else can - a ONE
/// KILOBYTE write blocking for hundreds of milliseconds - and it explains why the whole
/// browser hitches, because GPU memory is allocated in the GPU process and that process is
/// shared with every other tab.
///
/// If this lands near zero while the write still blocks, the allocation is happening inside
/// the write and the fix is to create FEWER buffers, not to move the creation.
pub(super) static CREATE_US_THIS_FRAME: std::sync::atomic::AtomicU64 =
    std::sync::atomic::AtomicU64::new(0);

/// Charge one `create_buffer` call's wall clock to this frame. See
/// [`CREATE_US_THIS_FRAME`].
pub(super) fn enc_create_us(us: u64) {
    CREATE_US_THIS_FRAME.fetch_add(us, std::sync::atomic::Ordering::Relaxed);
}

/// >>> THE WASM HEAP AT THE START OF THIS FRAME, IN PAGES, AND THE SUSPECT THAT IS LEFT.
///
/// Four GPU-side mechanisms have now been timed and refuted for the blocking write - the
/// texture burst, `destroy`, `create_buffer` and the lazy zero-init - and the queue is
/// EMPTY at every one of them. What has never been looked at is that the freeze the user
/// describes is *browser-wide*, which the RENDERER process explains as well as the GPU
/// process does: growing wasm linear memory by tens of megabytes is a single synchronous
/// event that stops that process dead, and it lands on whichever call happens to be running
/// - which on a load frame is the big arena write.
///
/// This is the page count at frame start; the stall reports the GROWTH since then. If a
/// stalling frame grew the heap and a healthy one did not, the write was never the problem
/// and every fix aimed at the GPU was aimed at the wrong process.
pub(super) static HEAP_PAGES_AT_FRAME_START: std::sync::atomic::AtomicU64 =
    std::sync::atomic::AtomicU64::new(0);

/// Wasm linear-memory pages (64 KB each), or 0 where there is no linear memory to ask about.
pub(super) fn heap_pages_now() -> u64 {
    #[cfg(target_arch = "wasm32")]
    {
        core::arch::wasm32::memory_size(0) as u64
    }
    #[cfg(not(target_arch = "wasm32"))]
    {
        0
    }
}

pub(super) fn enc_tex_upload(bytes: u64) {
    use std::sync::atomic::Ordering::Relaxed;
    // NOT `enc_tex_upload` - a bulk rename rewrote this line into a call to its own
    // function, and the worker died with `Maximum call stack size exceeded` naming nothing
    // useful. A rename that matches the body of the thing being renamed is how that happens.
    enc(&ENC.tex_upload_bytes, bytes);
    TEX_UPLOADS_THIS_FRAME.fetch_add(1, Relaxed);
    TEX_UPLOAD_BYTES_THIS_FRAME.fetch_add(bytes, Relaxed);
}

/// `(milliseconds, KB)` of the worst single `queue.write_buffer` of the run - the pair that
/// survives a hang. See [`BUFFER_WRITE_WORST_RUN`].
pub fn buffer_write_worst_run_us_kb() -> (f64, f64) {
    let v = BUFFER_WRITE_WORST_RUN.load(std::sync::atomic::Ordering::Relaxed);
    ((v >> 32) as f64 / 1000.0, (v & 0xffff_ffff) as f64 / 1024.0)
}

/// >>> THE LARGEST SINGLE `queue.write_buffer` OF THE RUN IN BYTES, WHATEVER IT COST.
///
/// # Why a SIZE, when every other register here is a TIME
/// Because the time is the one thing this workstation cannot reproduce and the size is
/// identical on both machines. MEASURED on the user's phone (PowerVR D-series, Chrome
/// Android): a 2,315 KB write blocked **208 ms**, a 4-byte write into the same buffer
/// immediately after took **0.0 ms**, and THE SAME 2,315 KB AGAIN took **551.9 ms** with
/// **nothing in flight**. By [`report_write_stall_control`]'s own legend a slow repeat means
/// the cost is the SIZE - a staging path charging per megabyte every time - not the
/// destination and not queue drain. The same phase on this workstation reads 0.5 ms.
///
/// So the phone owns the RATE and the desktop owns the BYTES, and a budget on the bytes is
/// the only half of that pair a run without a phone can hold. Every stall the device dump
/// carried was a write of 367 KB or more; nothing at or below a couple of hundred KB has
/// ever been seen to block. That is what [`device_budget`] gates on, and it is a structural
/// rule rather than a timing one on purpose - a staging path that stops routing bulk
/// geometry through `write_buffer` satisfies it on this machine, visibly, with no device
/// round trip.
pub(super) static BUFFER_WRITE_LARGEST_BYTES: std::sync::atomic::AtomicU64 =
    std::sync::atomic::AtomicU64::new(0);

/// The largest single `queue.write_buffer` of the run, in bytes. See
/// [`BUFFER_WRITE_LARGEST_BYTES`].
pub fn buffer_write_largest_bytes() -> u64 {
    BUFFER_WRITE_LARGEST_BYTES.load(std::sync::atomic::Ordering::Relaxed)
}

/// Arena bytes uploaded over the run, and the largest single upload - WHATEVER TRANSPORT
/// carried them, `queue.write_buffer` and the staging belt alike.
///
/// The volume half of the pair. See the counting site for why it exists: the budget's other
/// row is deliberately specific to the blocking CALL, so on its own it would report zero for
/// a build that moved twice the geometry by a different route.
pub(super) static UPLOAD_BYTES_RUN: std::sync::atomic::AtomicU64 = std::sync::atomic::AtomicU64::new(0);
pub(super) static UPLOAD_LARGEST_BYTES: std::sync::atomic::AtomicU64 =
    std::sync::atomic::AtomicU64::new(0);

/// `(total arena bytes uploaded this run, largest single upload)`, both transports counted.
/// See [`UPLOAD_BYTES_RUN`].
pub fn upload_volume() -> (u64, u64) {
    use std::sync::atomic::Ordering::Relaxed;
    (UPLOAD_BYTES_RUN.load(Relaxed), UPLOAD_LARGEST_BYTES.load(Relaxed))
}

/// The largest `queue.write_buffer` a run may make before this is called a defect.
///
/// NOT a round number picked for tidiness: the smallest write that has ever been SEEN to
/// block on the user's device is **367 KB**, and the four in one device dump were 367 KB,
/// 576 KB, 953 KB and 2,315 KB. 256 KB sits below every one of them with room to spare and
/// above the per-pass uniform and index writes a healthy frame makes, so it separates the
/// two populations rather than splitting either. Raise it only with a device dump showing a
/// write of the new size that does not block.
pub const WRITE_BUFFER_BUDGET_BYTES: u64 = 256 * 1024;

/// One line of the device budget: what was measured, what it is allowed to be, and whether
/// this run is inside it.
///
/// `why` is not decoration. A budget with no reason attached is a number somebody will
/// raise the moment it fails, and the whole point of these rows is that each one stands on
/// a measurement taken on the device that cannot be taken here.
pub struct BudgetRow {
    pub name: &'static str,
    pub reading: String,
    pub budget: String,
    pub pass: bool,
    pub why: &'static str,
}

/// The GPU-side rows of the DEVICE BUDGET - the phone's limits, checked on this machine.
///
/// See [`BUFFER_WRITE_LARGEST_BYTES`] for the argument that makes a desktop run able to
/// answer a device question at all: the phone owns the rate, the desktop owns the bytes,
/// and the bytes are the half that is identical on both.
pub fn device_budget_rows() -> Vec<BudgetRow> {
    let largest = buffer_write_largest_bytes();
    let (total, biggest) = upload_volume();
    vec![
        BudgetRow {
            name: "largest queue.write_buffer",
            reading: format!("{:.0} KB", largest as f64 / 1024.0),
            budget: format!("{} KB", WRITE_BUFFER_BUDGET_BYTES / 1024),
            pass: largest <= WRITE_BUFFER_BUDGET_BYTES,
            why: "every write MEASURED blocking on the user's device was 367 KB or larger, \
                  and the control says the cost is the SIZE, not the destination - so bulk \
                  geometry belongs on a mapped staging buffer, not in queue.write_buffer",
        },
        // >>> AND THE VOLUME BESIDE IT, so the row above cannot be read as a traffic figure.
        // The belt changes the TRANSPORT and not one byte of the payload; this row is what
        // says so, and what would catch a build that quietly started sending more.
        // Reported rather than budgeted because nobody has yet established what this title's
        // geometry SHOULD cost - a budget invented here would be a number, not a limit.
        BudgetRow {
            name: "arena bytes uploaded (all paths)",
            reading: format!(
                "{:.1} MB total, largest {:.0} KB",
                total as f64 / (1024.0 * 1024.0),
                biggest as f64 / 1024.0
            ),
            budget: "(reported)".into(),
            pass: true,
            why: "",
        },
    ]
}

/// Which transport the arena upload took this run, so a CAPTURE NAMES ITS OWN ARM.
///
/// A device A/B here is two visits with a rebuild between them, and the belt's whole effect
/// on the budget's first row is to drive it to zero - which is also what a build that
/// uploaded nothing would report. The file has to say which arm produced it, or the pair
/// rests on somebody's memory of which visit was which. The `SHADER ARMS` line exists for
/// the same reason and covers a different seam.
pub fn upload_transport() -> &'static str {
    if crate::gpu::gxm_staging_on() {
        "staging belt (default)"
    } else {
        "queue.write_buffer (VITASLOP_GXM_STAGING=0, the arm back)"
    }
}

/// >>> A SINGLE `queue.write_buffer` THAT BLOCKED FOR A LONG TIME, SAID OUT LOUD.
///
/// MEASURED on the user's phone: **5,240 ms for 55 KB**, and on this workstation 890-1,550 ms
/// for 0.5-3.6 MB. Fifty-five kilobytes is not a bandwidth figure and no call costs seconds,
/// so this is the CPU stopped dead waiting on the GPU queue.
///
/// # Why a report and not just the maxima
/// The panel's `WORST write_buffer OF THE RUN` says it happened; it cannot say WHEN or what
/// else was in flight, and the device dump that carried it also showed a WORST FRAME of
/// 1,085 ms - a 5.2-second write cannot fit inside a 1.1-second frame, so it did not happen
/// inside a counted frame at all. That contradiction is the whole lead
/// [[vitaslop-contradiction-means-look-between]], and it needs a line at the moment it
/// happens, next to whatever else the log is saying, rather than a number read minutes later.
///
/// Every stall over the threshold is reported, on a doubling cadence so a pathological run
/// cannot bury the rest of the log.
/// How many submits the presenting layer believes are still in flight, published so a
/// stalled `queue.write_buffer` can say it. The platform layer cannot see the queue depth -
/// the counter is kept by whatever drives `present` - and the depth is the one fact that
/// separates "the queue is deep" from "the staging path is blocked on nothing".
/// Zero when nobody publishes it, which is honest: the desktop path submits and waits.
pub static GPU_SUBMITS_IN_FLIGHT: std::sync::atomic::AtomicU64 =
    std::sync::atomic::AtomicU64::new(0);

/// >>> EVERY STALL THE RUN HAD, ON A LINE THAT SURVIVES THE PANEL.
///
/// [`report_write_buffer_stall`] says a stall out loud, and that report is unreadable for
/// the two reasons this project keeps re-learning: it goes through the WARNINGS panel,
/// which keeps 96 distinct lines and silently drops the rest
/// [[vitaslop-a-diagnostic-can-bury-the-findings]], and it fires on a doubling cadence, so
/// stalls 5, 6 and 7 are not in the log at all. A device capture is then a downloaded file
/// that may carry the FIRST stall of a run and no count of the rest.
///
/// This is the census instead: how many there were, how much wall clock they cost together,
/// and the full context of the WORST one, so a single panel line answers the question the
/// per-stall report was written to answer. The context matters more than the total - see
/// [`BUFFER_WRITES_THIS_FRAME`] for why `nth` is the field that separates the two live
/// stories.
/// Which queue upload blocked. The two have OPPOSITE fixes - a texture burst wants
/// spreading over frames, a buffer write wants the staging ring flushed - so a census that
/// cannot name the kind cannot aim anything.
#[derive(Clone, Copy, PartialEq, Eq)]
pub(crate) enum UploadKind {
    Buffer,
    Texture,
}

impl UploadKind {
    pub(super) fn name(self) -> &'static str {
        match self {
            UploadKind::Buffer => "write_buffer",
            UploadKind::Texture => "write_texture",
        }
    }
}

#[derive(Default, Clone, Copy)]
pub(super) struct WriteStallCensus {
    /// Stalls at or over `STALL_US` on a BUFFER write, however many the log dropped.
    n_buffer: u64,
    /// The same for a TEXTURE upload.
    n_texture: u64,
    /// Their total, in microseconds. A run's rate is a lie beside a second of this.
    total_us: u64,
    /// The worst stall's own numbers, and what the frame had already put in the queue ahead
    /// of it: `(us, bytes, nth of frame, textures, texture KB, new buffers, in flight,
    /// pipelines built, resources destroyed, destroy ms)`.
    worst: Option<(u64, u64, u64, u64, u64, u64, u64, u64, u64, u64)>,
    /// Stalls that happened in a frame which BUILT A PIPELINE, against the total. If a
    /// shader compile is what the client waits behind, this is near every one of them; if
    /// it is near zero the compile theory is dead and the census says so without another
    /// run.
    n_with_pipeline_build: u64,
    /// Which call the worst one was.
    worst_kind: Option<UploadKind>,
}

pub(super) static WRITE_STALLS: std::sync::Mutex<WriteStallCensus> = std::sync::Mutex::new(
    WriteStallCensus {
        n_buffer: 0,
        n_texture: 0,
        total_us: 0,
        worst: None,
        worst_kind: None,
        n_with_pipeline_build: 0,
    },
);

/// The census as one panel-ready clause, or `None` when the run never stalled - which is
/// the reading a healthy run must be able to state, not merely omit.
pub fn write_stall_census() -> Option<String> {
    let c = *WRITE_STALLS.lock().unwrap_or_else(|e| e.into_inner());
    let (us, bytes, nth, texn, texkb, bufs, inflight, pipes, killed, kill_us) = c.worst?;
    Some(format!(
        "QUEUE UPLOAD STALLS over 100 ms: {} write_buffer + {} write_texture, {:.1} s of \
         wall clock together, {} of them in a frame that BUILT A PIPELINE; the worst was a \
         {} that blocked {:.0} ms for {} KB as upload #{nth} of its frame, behind {texn} \
         textures / {texkb} KB, {bufs} new buffers, {pipes} pipeline build(s) and {killed} \
         resource(s) destroyed in {:.0} ms, with {inflight} submit(s) in flight",
        c.n_buffer,
        c.n_texture,
        c.total_us as f64 / 1e6,
        c.n_with_pipeline_build,
        c.worst_kind.map_or("write", UploadKind::name),
        us as f64 / 1000.0,
        bytes / 1024,
        kill_us as f64 / 1000.0,
    ))
}

/// Say what the CONTROL write cost beside the one that looked slow.
///
/// Four bytes into an already-written buffer, issued immediately after a slow write. It
/// touches nothing any of the refuted theories name, so its time is the floor for "a
/// `write_buffer` call on this thread, right now". See the call site for the argument.
pub(super) fn report_write_stall_control(tiny_us: u64, same_us: u64, stalled_us: u64, bytes: u64) {
    static N: std::sync::atomic::AtomicU64 = std::sync::atomic::AtomicU64::new(0);
    let n = N.fetch_add(1, std::sync::atomic::Ordering::Relaxed) + 1;
    if !n.is_power_of_two() {
        return;
    }
    report_warn!(
        "gxm: write stall CONTROL - a {} KB write took {:.0} ms; immediately after it, 4 \
         bytes into the same buffer took {:.1} ms and THE SAME {} KB AGAIN took {:.1} ms. \
         The 4-byte number says whether this thread was running at all; the repeat says \
         whether the cost was the DESTINATION (fast repeat - first touch of a new buffer, \
         paid once, so create fewer) or the SIZE (slow repeat - a staging path charging per \
         megabyte every time). See `report_write_stall_control`.",
        bytes / 1024,
        stalled_us as f64 / 1000.0,
        tiny_us as f64 / 1000.0,
        bytes / 1024,
        same_us as f64 / 1000.0,
    );
}

pub(super) fn report_upload_stall(kind: UploadKind, us: u64, bytes: u64, nth_this_frame: u64) {
    /// A write is microseconds when it is healthy. 100 ms is far outside anything a copy
    /// explains, and low enough to catch the ones that are not yet seconds.
    const STALL_US: u64 = 100_000;
    if us < STALL_US {
        return;
    }
    static N: std::sync::atomic::AtomicU64 = std::sync::atomic::AtomicU64::new(0);
    let n = N.fetch_add(1, std::sync::atomic::Ordering::Relaxed) + 1;
    // >>> THE CENSUS TAKES EVERY STALL, including the ones the doubling cadence below
    // declines to print and the ones the warnings panel would drop. See
    // [`WriteStallCensus`].
    {
        let mut c = WRITE_STALLS.lock().unwrap_or_else(|e| e.into_inner());
        match kind {
            UploadKind::Buffer => c.n_buffer += 1,
            UploadKind::Texture => c.n_texture += 1,
        }
        c.total_us += us;
        let pipes = PIPELINES_BUILT_THIS_FRAME.load(std::sync::atomic::Ordering::Relaxed);
        if pipes > 0 {
            c.n_with_pipeline_build += 1;
        }
        if c.worst.is_none_or(|(w, ..)| us > w) {
            c.worst_kind = Some(kind);
            c.worst = Some((
                us,
                bytes,
                nth_this_frame,
                TEX_UPLOADS_THIS_FRAME.load(std::sync::atomic::Ordering::Relaxed),
                TEX_UPLOAD_BYTES_THIS_FRAME.load(std::sync::atomic::Ordering::Relaxed) / 1024,
                BUFFERS_MADE_THIS_FRAME.load(std::sync::atomic::Ordering::Relaxed),
                GPU_SUBMITS_IN_FLIGHT.load(std::sync::atomic::Ordering::Relaxed),
                pipes,
                RESOURCES_DESTROYED_THIS_FRAME.load(std::sync::atomic::Ordering::Relaxed),
                DESTROY_US_THIS_FRAME.load(std::sync::atomic::Ordering::Relaxed),
            ));
        }
    }
    // >>> EVERY ONE OF THE FIRST SIXTEEN, THEN A DOUBLING CADENCE.
    //
    // Pure doubling made every run of this rig report exactly three lines - stalls 1, 2 and
    // 4 - whatever the run actually suffered, so an A/B on stall COUNT read 3 against 3 by
    // construction and the census that could have counted them lives on a panel a short run
    // never dumps. Sixteen is enough to count a scene transition's worth and still cannot
    // bury a log [[vitaslop-a-diagnostic-can-bury-the-findings]].
    if n > 16 && !n.is_power_of_two() {
        return;
    }
    // >>> THE CONTEXT TRAVELS WITH THE REPORT. The first stall this caught happened to sit
    // one line below a `texture working set is 145 MB (new high)`, which reads like a cause;
    // the second sat among ordinary draw warnings and had no such neighbour. Log ADJACENCY
    // is not attribution - it is whatever else happened to be logging. These counters are
    // cumulative, so the DIFFERENCE between two consecutive stall lines is how much
    // expansion work went into the queue between them.
    let (submits, textures) = crate::texenc::raw_batch_counts();
    // >>> WHAT *THIS FRAME* PUT IN THE QUEUE AHEAD OF THE STALLED WRITE. The user's report is
    // that the stall happens on a NEW course/menu, i.e. exactly when a frame first uploads
    // textures it has never seen - see `TEX_UPLOADS_THIS_FRAME`.
    let frame_texn = TEX_UPLOADS_THIS_FRAME.load(std::sync::atomic::Ordering::Relaxed);
    let frame_texkb =
        TEX_UPLOAD_BYTES_THIS_FRAME.load(std::sync::atomic::Ordering::Relaxed) / 1024;
    let frame_bufs = BUFFERS_MADE_THIS_FRAME.load(std::sync::atomic::Ordering::Relaxed);
    // >>> AND THE DEPTH OF THE QUEUE THE WRITE IS SUPPOSEDLY WAITING ON. At ZERO this is not
    // a deep queue at all and the block is the staging path or the wire, which wants a
    // different fix from throttling presents. See `GPU_SUBMITS_IN_FLIGHT`.
    let inflight = GPU_SUBMITS_IN_FLIGHT.load(std::sync::atomic::Ordering::Relaxed);
    // HOW TO READ THIS LINE, kept here rather than re-transmitted on every stall:
    //
    //  * It is NOT a copy. No call costs milliseconds per kilobyte; the time is this thread
    //    waiting on the GPU queue to DRAIN, so the bytes in the line are not the cause.
    //  * `write #N of the frame` counts from 0 = the first write after the previous frame's
    //    submit. A large N with little uploaded ahead of it means this frame is draining a
    //    backlog left by EARLIER frames, not paying for its own work.
    //  * Compare consecutive stall lines: the DELTA between their cumulative counts is what
    //    actually entered the queue in between. Log ORDER is not evidence.
    report_warn!(
        "gxm: queue.{} BLOCKED {:.0} ms writing {} KB (stall {n} this run) - \
         waiting on the queue to drain, not copying. Upload #{nth_this_frame} of the frame, \
         behind {frame_texn} textures / {frame_texkb} KB, {frame_bufs} new buffers, {} \
         pipeline build(s), {} resource(s) destroyed in {:.0} ms, {:.0} ms inside \
         create_buffer and {} MB of WASM HEAP GROWTH this frame; run \
         total {textures} texture expansions in {submits} submits, with {inflight} submit(s)              still IN FLIGHT at the write. \
         See `report_upload_stall`, and the `SLOWEST FRAMES` panel line for the run's census.",
        kind.name(),
        us as f64 / 1000.0,
        bytes / 1024,
        PIPELINES_BUILT_THIS_FRAME.load(std::sync::atomic::Ordering::Relaxed),
        RESOURCES_DESTROYED_THIS_FRAME.load(std::sync::atomic::Ordering::Relaxed),
        DESTROY_US_THIS_FRAME.load(std::sync::atomic::Ordering::Relaxed) as f64 / 1000.0,
        CREATE_US_THIS_FRAME.load(std::sync::atomic::Ordering::Relaxed) as f64 / 1000.0,
        // >>> AND WHETHER THE RENDERER PROCESS JUST TOOK PAGES. See
        // `HEAP_PAGES_AT_FRAME_START`: growing wasm linear memory is synchronous in the
        // process the page itself runs in, and it would explain a browser-wide freeze that
        // an idle GPU cannot.
        heap_pages_now().saturating_sub(
            HEAP_PAGES_AT_FRAME_START.load(std::sync::atomic::Ordering::Relaxed),
        ) / 16,
    );
}

/// Bump one encode counter. Relaxed: these are a diagnostic, and no reader depends on
/// seeing two of them agree at an instant.
#[inline]
pub(super) fn enc(c: &std::sync::atomic::AtomicU64, n: u64) {
    c.fetch_add(n, std::sync::atomic::Ordering::Relaxed);
}
