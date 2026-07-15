//! Vita host-call implementations, grouped by module (one file per Sce* module,
//! mirroring the vita-headers layout). Each module holds the handler bodies; the
//! single [`dispatch`] match below routes a function NID straight to its handler.

pub mod at9;
pub mod audio;
pub mod cfmt;
pub mod ctrl;
pub mod display;
pub mod gxm;
pub mod iofilemgr;
pub mod libkernel;
pub mod lwsync;
pub mod ngs;
pub mod processmgr;
pub mod services;
pub mod sync;
pub mod sysmem;
pub mod threadmgr;
pub mod touch;

use crate::host::{GuestCtx, VitaState};
use crate::nid::{
    audio as audio_nid, ctrl as ctrl_nid, display as display_nid, gxm as gxm_nid,
    iofilemgr as io_nid, libkernel as lk_nid, lwsync as lw_nid, ngs as ngs_nid,
    processmgr as pm_nid, services as sv_nid, sync as sync_nid, sysmem as sm_nid,
    threadmgr as tm_nid,
};
use crate::{nid, SvcOutcome};

use std::collections::BTreeMap;
use std::sync::{LazyLock, Mutex};

/// Diagnostic call-site profiler (env `VITASLOP_DBG_CALLSITES`): counts host calls
/// keyed by (function NID, guest return address). A busy-wait spin shows up as one
/// (nid, lr) pair with an enormous count - the exact instruction to investigate.
static DBG_CALLSITES: LazyLock<bool> =
    LazyLock::new(|| std::env::var("VITASLOP_DBG_CALLSITES").is_ok());
/// Diagnostic (env `VITASLOP_TRACE_NGS`): log every NGS and sceAudioOut call with
/// its first four args and caller, to see exactly how a title feeds AT9 data to a
/// voice and where the final mix goes - the facts needed before HLE'ing the mixer.
static TRACE_NGS: LazyLock<bool> = LazyLock::new(|| std::env::var("VITASLOP_TRACE_NGS").is_ok());
static DBG_ERR: LazyLock<bool> = LazyLock::new(|| std::env::var("VITASLOP_DBG_ERR").is_ok());
static CALLSITE_HIST: Mutex<BTreeMap<(u32, u32), u64>> = Mutex::new(BTreeMap::new());

/// Print the hottest call sites (by count) gathered when `VITASLOP_DBG_CALLSITES` is
/// set. Call from a probe after the run to localize a spin.
pub fn dump_call_sites(top: usize) {
    let h = CALLSITE_HIST.lock().unwrap();
    let mut v: Vec<_> = h.iter().collect();
    v.sort_by(|a, b| b.1.cmp(a.1));
    eprintln!("--- hottest call sites (nid, caller lr): count ---");
    for ((nid_, lr), n) in v.into_iter().take(top) {
        eprintln!("  {n:>12}  {} @ lr={lr:#010x}", nid::name(*nid_));
    }
}

/// Route a NID call straight to its handler. Function NIDs are globally unique, so
/// one match over every implemented NID is unambiguous; `library_nid` is only for
/// logging an unimplemented call. This is a single decision - the compiler lowers
/// the match to one binary-decision tree / jump table - rather than the old chain
/// that re-probed each module's own match in turn (up to ~13 calls deep on a cold
/// NID). At tens of millions of host calls per frame that flat routing is the hot
/// path, so it lives in one place; the handler bodies stay in their per-module
/// files. An unhandled NID is recorded and returns 0 so the run continues and the
/// gap shows up in the capture.
pub fn dispatch(
    library_nid: u32,
    func_nid: u32,
    ctx: &mut GuestCtx,
    st: &mut VitaState,
) -> SvcOutcome {
    // A handler that returns `()` leaves the guest running; wrap its call so the arm
    // yields `Continue`. Handlers that can suspend a thread (blocking waits, the
    // frame flip, process/thread exit) return the `SvcOutcome` directly instead.
    macro_rules! cont {
        ($call:expr) => {{
            $call;
            SvcOutcome::Continue
        }};
    }

    // Diagnostic (env-gated): tally host calls by (nid, game-level caller), so a hot
    // busy-wait spin's exact site is visible without printing millions of lines. The
    // immediate LR is usually a thin libc lock wrapper, so scan the guest stack for
    // the first return address in the main module's code range - the game loop that
    // is actually spinning. Dumped by [`dump_call_sites`]. Zero cost when unset.
    if *DBG_CALLSITES {
        let mut caller = ctx.regs[14];
        let sp = ctx.regs[13];
        for i in 0..40u32 {
            let v = ctx.read_u32(sp.wrapping_add(i * 4));
            if (0x8130_0000..0x8150_0000).contains(&v) {
                caller = v;
                break;
            }
        }
        *CALLSITE_HIST.lock().unwrap().entry((func_nid, caller)).or_insert(0u64) += 1;
    }

    if *TRACE_NGS && (library_nid == nid::lib::SCE_NGS || library_nid == nid::lib::SCE_AUDIO) {
        eprintln!(
            "NGS {} a0={:#010x} a1={:#010x} a2={:#010x} a3={:#010x} lr={:#010x}",
            nid::name(func_nid),
            ctx.arg(0),
            ctx.arg(1),
            ctx.arg(2),
            ctx.arg(3),
            ctx.regs[14],
        );
    }

    let dbg_err = *DBG_ERR;
    let outcome = match func_nid {
        // --- lwsync: lightweight mutex / cond (the hottest surface) --------------
        lw_nid::CREATE_LW_MUTEX => cont!(lwsync::init_work(ctx, lwsync::LW_MUTEX_WORK_SIZE)),
        lw_nid::CREATE_LW_COND => cont!(lwsync::init_work(ctx, lwsync::LW_COND_WORK_SIZE)),
        lw_nid::WAIT_LW_COND | lw_nid::WAIT_LW_COND_CB => lwsync::wait_lw_cond(ctx, st),
        lw_nid::SIGNAL_LW_COND => cont!(lwsync::signal_lw_cond(ctx, st, false)),
        // SignalLwCondAll wakes every waiter; SignalLwCondTo targets one thread,
        // approximated by a broadcast (a spurious wake re-checks and re-waits).
        lw_nid::SIGNAL_LW_COND_ALL | lw_nid::SIGNAL_LW_COND_TO => {
            cont!(lwsync::signal_lw_cond(ctx, st, true))
        }
        lw_nid::LOCK_LW_MUTEX
        | lw_nid::LOCK_LW_MUTEX_CB
        | lw_nid::TRY_LOCK_LW_MUTEX
        | lw_nid::UNLOCK_LW_MUTEX
        | lw_nid::UNLOCK_LW_MUTEX2
        | lw_nid::DELETE_LW_MUTEX
        | lw_nid::DELETE_LW_COND => cont!(lwsync::succeed(ctx)),

        // --- sync: heavyweight mutex / sema / cond / event flag -----------------
        sync_nid::CREATE_MUTEX => cont!(sync::create_mutex(ctx, st)),
        // Lock and wait can block under the preemptive scheduler (Block parks).
        sync_nid::LOCK_MUTEX => sync::lock_mutex(ctx, st, false),
        sync_nid::TRY_LOCK_MUTEX => sync::lock_mutex(ctx, st, true),
        sync_nid::UNLOCK_MUTEX => cont!(sync::unlock_mutex(ctx, st)),
        sync_nid::DELETE_MUTEX => cont!(sync::delete_object(ctx, st)),
        sync_nid::CREATE_SEMA => cont!(sync::create_sema(ctx, st)),
        sync_nid::WAIT_SEMA => sync::wait_sema(ctx, st),
        sync_nid::SIGNAL_SEMA => cont!(sync::signal_sema(ctx, st)),
        sync_nid::DELETE_SEMA => cont!(sync::delete_object(ctx, st)),
        sync_nid::CREATE_COND => cont!(sync::create_cond(ctx, st)),
        sync_nid::WAIT_COND => sync::wait_cond(ctx, st),
        sync_nid::SIGNAL_COND => cont!(sync::signal_cond(ctx, st, false)),
        sync_nid::SIGNAL_COND_ALL => cont!(sync::signal_cond(ctx, st, true)),
        sync_nid::DELETE_COND => cont!(sync::delete_object(ctx, st)),
        sync_nid::CREATE_EVENT_FLAG => cont!(sync::create_event_flag(ctx, st)),
        sync_nid::SET_EVENT_FLAG => cont!(sync::set_event_flag(ctx, st)),
        sync_nid::WAIT_EVENT_FLAG => cont!(sync::wait_event_flag(ctx, st)),
        sync_nid::CLEAR_EVENT_FLAG => cont!(sync::clear_event_flag(ctx, st)),
        sync_nid::DELETE_EVENT_FLAG => cont!(sync::delete_object(ctx, st)),
        sync_nid::GET_SYSTEM_TIME_WIDE => cont!(sync::get_system_time_wide(ctx, st)),

        // --- libkernel: clib string/mem, threads, process ----------------------
        lk_nid::CLIB_PRINTF => cont!(libkernel::clib_printf(ctx, st)),
        lk_nid::CLIB_SNPRINTF => cont!(libkernel::clib_snprintf(ctx, st)),
        // memmove shares memcpy's read-then-write impl (tolerates overlap).
        lk_nid::CLIB_MEMCPY | lk_nid::CLIB_MEMMOVE => cont!(libkernel::clib_memcpy(ctx, st)),
        lk_nid::CLIB_MEMSET => cont!(libkernel::clib_memset(ctx, st)),
        lk_nid::CLIB_MEMCMP => cont!(libkernel::clib_memcmp(ctx, st)),
        lk_nid::CLIB_STRNLEN => cont!(libkernel::clib_strnlen(ctx, st)),
        lk_nid::CLIB_STRNCPY => cont!(libkernel::clib_strncpy(ctx, st)),
        lk_nid::CLIB_STRNCMP => cont!(libkernel::clib_strncmp(ctx, st)),
        lk_nid::CLIB_STRCMP => cont!(libkernel::clib_strcmp(ctx, st)),
        lk_nid::CREATE_THREAD => cont!(libkernel::create_thread(ctx, st)),
        lk_nid::START_THREAD => libkernel::start_thread(ctx, st),
        // Join can block under the preemptive scheduler.
        lk_nid::WAIT_THREAD_END => libkernel::wait_thread_end(ctx, st),
        lk_nid::GET_THREAD_ID => cont!(libkernel::get_thread_id(ctx, st)),
        lk_nid::GET_THREAD_EXIT_STATUS => cont!(libkernel::get_thread_exit_status(ctx, st)),
        lk_nid::GET_TLS_ADDR => cont!(libkernel::get_tls_addr(ctx, st)),
        lk_nid::GET_PROCESS_TIME_WIDE => libkernel::get_process_time_wide(ctx, st),
        lk_nid::EXIT_PROCESS => {
            // r0 (exit code) is left as the guest set it; any exit is a clean stop.
            libkernel::trace_exit(ctx, st);
            SvcOutcome::Halt
        }

        // --- threadmgr: delay, exit, process id --------------------------------
        tm_nid::DELAY_THREAD => cont!(threadmgr::delay_thread(ctx, st)),
        // A thread ending itself: just this thread under the preemptive scheduler;
        // a whole-run stop in single-thread-of-control bring-up (only main reaches
        // here there - workers return normally instead).
        tm_nid::EXIT_THREAD | tm_nid::EXIT_DELETE_THREAD => {
            if st.is_preemptive() {
                SvcOutcome::ThreadExit
            } else {
                SvcOutcome::Halt
            }
        }
        tm_nid::GET_PROCESS_ID => cont!(threadmgr::get_process_id(ctx, st)),

        // --- gxm: graphics ------------------------------------------------------
        gxm_nid::INITIALIZE => cont!(gxm::initialize(ctx, st)),
        gxm_nid::MAP_MEMORY
        | gxm_nid::FINISH
        | gxm_nid::PAD_HEARTBEAT
        | gxm_nid::DISPLAY_QUEUE_FINISH
        | gxm_nid::PROGRAM_CHECK
        | gxm_nid::DESTROY_CONTEXT
        | gxm_nid::DESTROY_RENDER_TARGET
        | gxm_nid::SHADER_PATCHER_DESTROY
        | gxm_nid::SHADER_PATCHER_UNREGISTER_PROGRAM
        | gxm_nid::SHADER_PATCHER_RELEASE_VERTEX_PROGRAM
        | gxm_nid::SHADER_PATCHER_RELEASE_FRAGMENT_PROGRAM
        | gxm_nid::DEPTH_STENCIL_SURFACE_INIT
        | gxm_nid::SET_FRAGMENT_PROGRAM => cont!(gxm::ok(ctx)),
        gxm_nid::TERMINATE => {
            ctx.ret(0);
            if st.halt_on_terminate {
                SvcOutcome::Halt
            } else {
                SvcOutcome::Continue
            }
        }
        gxm_nid::MAP_VERTEX_USSE_MEMORY | gxm_nid::MAP_FRAGMENT_USSE_MEMORY => {
            cont!(gxm::map_usse(ctx))
        }
        gxm_nid::CREATE_CONTEXT | gxm_nid::CREATE_RENDER_TARGET | gxm_nid::SHADER_PATCHER_CREATE => {
            cont!(gxm::out_handle(ctx, st, 1))
        }
        gxm_nid::SYNC_OBJECT_CREATE => cont!(gxm::out_handle(ctx, st, 0)),
        gxm_nid::SHADER_PATCHER_REGISTER_PROGRAM => cont!(gxm::register_program(ctx, st)),
        gxm_nid::SHADER_PATCHER_GET_PROGRAM_FROM_ID => cont!(gxm::get_program_from_id(ctx, st)),
        gxm_nid::PROGRAM_PARAMETER_GET_RESOURCE_INDEX => cont!(ctx.ret(0)),
        gxm_nid::PROGRAM_FIND_PARAMETER_BY_NAME => cont!(gxm::find_parameter(ctx, st)),
        gxm_nid::COLOR_SURFACE_INIT => cont!(gxm::color_surface_init(ctx, st)),
        gxm_nid::SHADER_PATCHER_CREATE_VERTEX_PROGRAM => cont!(gxm::create_vertex_program(ctx, st)),
        gxm_nid::SHADER_PATCHER_CREATE_FRAGMENT_PROGRAM => cont!(gxm::out_handle(ctx, st, 6)),
        gxm_nid::BEGIN_SCENE => cont!(gxm::begin_scene(ctx, st)),
        gxm_nid::END_SCENE => cont!(gxm::end_scene(ctx, st)),
        gxm_nid::SET_VERTEX_PROGRAM => cont!(gxm::set_vertex_program(ctx, st)),
        gxm_nid::RESERVE_VERTEX_DEFAULT_UNIFORM_BUFFER
        | gxm_nid::RESERVE_FRAGMENT_DEFAULT_UNIFORM_BUFFER => cont!(gxm::reserve_uniforms(ctx, st)),
        gxm_nid::SET_UNIFORM_DATA_F => cont!(gxm::set_uniform_data_f(ctx, st)),
        gxm_nid::SET_VERTEX_STREAM => cont!(gxm::set_vertex_stream(ctx, st)),
        gxm_nid::SET_FRAGMENT_TEXTURE => cont!(gxm::set_fragment_texture(ctx, st)),
        gxm_nid::TEXTURE_INIT_LINEAR => cont!(gxm::texture_init(ctx, st, gxm::TYPE_LINEAR)),
        gxm_nid::TEXTURE_INIT_LINEAR_STRIDED => {
            cont!(gxm::texture_init(ctx, st, gxm::TYPE_LINEAR_STRIDED))
        }
        gxm_nid::TEXTURE_INIT_SWIZZLED => cont!(gxm::texture_init(ctx, st, gxm::TYPE_SWIZZLED)),
        gxm_nid::TEXTURE_INIT_SWIZZLED_ARBITRARY => {
            cont!(gxm::texture_init(ctx, st, gxm::TYPE_SWIZZLED_ARBITRARY))
        }
        gxm_nid::TEXTURE_INIT_TILED => cont!(gxm::texture_init(ctx, st, gxm::TYPE_TILED)),
        gxm_nid::TEXTURE_SET_DATA => cont!(gxm::texture_set_data(ctx, st)),
        gxm_nid::TEXTURE_SET_FORMAT => cont!(gxm::texture_set_format(ctx, st)),
        gxm_nid::TEXTURE_GET_DATA => cont!(gxm::texture_get_data(ctx)),
        gxm_nid::TEXTURE_GET_WIDTH => cont!(gxm::texture_get_dim(ctx, 12)),
        gxm_nid::TEXTURE_GET_HEIGHT => cont!(gxm::texture_get_dim(ctx, 0)),
        gxm_nid::TEXTURE_GET_FORMAT => cont!(gxm::texture_get_format(ctx, st)),
        gxm_nid::TEXTURE_SET_MAG_FILTER
        | gxm_nid::TEXTURE_SET_MIN_FILTER
        | gxm_nid::TEXTURE_SET_MIP_FILTER
        | gxm_nid::TEXTURE_SET_U_ADDR_MODE
        | gxm_nid::TEXTURE_SET_V_ADDR_MODE
        | gxm_nid::SET_FRAGMENT_UNIFORM_BUFFER => cont!(gxm::ok(ctx)),
        gxm_nid::DRAW => cont!(gxm::draw(ctx, st)),
        gxm_nid::DISPLAY_QUEUE_ADD_ENTRY => {
            // The frame is complete and queued to flip; on hardware the caller waits
            // for the flip here, so this is the guest's per-frame yield point.
            gxm::display_queue_add_entry(ctx, st);
            SvcOutcome::Yield
        }

        // --- iofilemgr: file IO -------------------------------------------------
        io_nid::IO_OPEN => cont!(iofilemgr::io_open(ctx, st)),
        io_nid::IO_CLOSE => cont!(iofilemgr::io_close(ctx, st)),
        io_nid::IO_READ => cont!(iofilemgr::io_read(ctx, st)),
        io_nid::IO_WRITE => cont!(iofilemgr::io_write(ctx, st)),
        io_nid::IO_LSEEK32 => cont!(iofilemgr::io_lseek32(ctx, st)),
        io_nid::IO_LSEEK => cont!(iofilemgr::io_lseek(ctx, st)),
        io_nid::IO_PREAD => cont!(iofilemgr::io_pread(ctx, st)),
        io_nid::IO_PWRITE => cont!(iofilemgr::io_pwrite(ctx, st)),
        io_nid::IO_GETSTAT => cont!(iofilemgr::io_getstat(ctx, st)),
        io_nid::IO_MKDIR => cont!(iofilemgr::io_mkdir(ctx, st)),
        io_nid::IO_REMOVE => cont!(iofilemgr::io_remove(ctx, st)),

        // --- sysmem: memory blocks ---------------------------------------------
        sm_nid::ALLOC_MEM_BLOCK => cont!(sysmem::alloc_mem_block(ctx, st)),
        sm_nid::GET_MEM_BLOCK_BASE => cont!(sysmem::get_mem_block_base(ctx, st)),

        // --- display ------------------------------------------------------------
        display_nid::SET_FRAME_BUF => cont!(display::set_frame_buf(ctx, st)),

        // --- ctrl: input --------------------------------------------------------
        ctrl_nid::PEEK_BUFFER_POSITIVE => cont!(ctrl::peek_buffer_positive(ctx, st)),
        ctrl_nid::READ_BUFFER_POSITIVE => cont!(ctrl::read_buffer_positive(ctx, st)),
        ctrl_nid::PEEK_BUFFER_NEGATIVE => cont!(ctrl::peek_buffer_negative(ctx, st)),
        ctrl_nid::READ_BUFFER_NEGATIVE => cont!(ctrl::read_buffer_negative(ctx, st)),
        ctrl_nid::SET_SAMPLING_MODE => cont!(ctx.ret(0)),

        // --- ngs / audio --------------------------------------------------------
        ngs_nid::SYSTEM_GET_REQUIRED_MEMORY_SIZE => cont!(ngs::system_get_required_memory_size(ctx, st)),
        ngs_nid::SYSTEM_INIT => cont!(ngs::system_init(ctx, st)),
        ngs_nid::RACK_GET_REQUIRED_MEMORY_SIZE => cont!(ngs::rack_get_required_memory_size(ctx, st)),
        ngs_nid::RACK_INIT => cont!(ngs::rack_init(ctx, st)),
        ngs_nid::RACK_GET_VOICE_HANDLE => cont!(ngs::rack_get_voice_handle(ctx, st)),
        ngs_nid::VOICE_GET_STATE_DATA => cont!(ngs::voice_get_state_data(ctx, st)),
        ngs_nid::VOICE_LOCK_PARAMS => cont!(ngs::voice_lock_params(ctx, st)),
        ngs_nid::VOICE_DEF_GET_SIMPLE_ATRAC9
        | ngs_nid::VOICE_DEF_GET_MASTER_BUSS
        | ngs_nid::VOICE_DEF_GET_REVERB_BUSS
        | ngs_nid::VOICE_DEF_GET_EQ_BUSS => cont!(ngs::voice_def_get(ctx, st)),
        ngs_nid::PATCH_CREATE_ROUTING => cont!(ngs::patch_create_routing(ctx, st)),
        // The remaining NGS calls are state transitions / per-frame pumps that
        // succeed silently: update/flags/release, voice play/keyoff/kill/pause/
        // resume, param unlock, callbacks, bypass, patch info, AT9 details,
        // out-of-range query (0 = in range).
        ngs_nid::SYSTEM_UPDATE => cont!(ngs::system_update(ctx, st)),
        ngs_nid::VOICE_UNLOCK_PARAMS => cont!(ngs::voice_unlock_params(ctx, st)),
        ngs_nid::SYSTEM_SET_FLAGS
        | ngs_nid::SYSTEM_RELEASE
        | ngs_nid::VOICE_RESUME
        | ngs_nid::VOICE_SET_FINISHED_CALLBACK
        | ngs_nid::VOICE_SET_MODULE_CALLBACK
        | ngs_nid::VOICE_BYPASS_MODULE
        | ngs_nid::VOICE_GET_PARAMS_OUT_OF_RANGE
        | ngs_nid::VOICE_PATCH_SET_VOLUMES_MATRIX
        | ngs_nid::PATCH_GET_INFO
        | ngs_nid::AT9_GET_SECTION_DETAILS => cont!(ctx.ret(0)),
        ngs_nid::VOICE_PLAY => cont!(ngs::voice_play(ctx, st)),
        ngs_nid::VOICE_KEY_OFF | ngs_nid::VOICE_KILL | ngs_nid::VOICE_PAUSE => {
            cont!(ngs::voice_stop(ctx, st))
        }
        audio_nid::OUT_OPEN_PORT => cont!(audio::out_open_port(ctx, st)),
        audio_nid::OUT_OUTPUT => audio::out_output(ctx, st),
        audio_nid::OUT_SET_VOLUME => cont!(audio::out_set_volume(ctx, st)),
        audio_nid::OUT_RELEASE_PORT => cont!(audio::out_release_port(ctx, st)),

        // --- processmgr: process param, std streams, time ----------------------
        pm_nid::GET_PROCESS_PARAM => cont!(processmgr::get_process_param(ctx, st)),
        pm_nid::GET_STDIN => cont!(processmgr::get_stdin(ctx, st)),
        pm_nid::GET_STDOUT => cont!(processmgr::get_stdout(ctx, st)),
        pm_nid::GET_STDERR => cont!(processmgr::get_stderr(ctx, st)),
        pm_nid::LIBC_TIME => cont!(processmgr::libc_time(ctx, st)),

        // --- services: sysmodule / net / http / np / rtc / apputil / touch -----
        sv_nid::SYSMODULE_IS_LOADED => cont!(services::sysmodule_is_loaded(ctx, st)),
        sv_nid::NET_CTL_INET_GET_STATE => cont!(services::netctl_inet_get_state(ctx, st)),
        sv_nid::NET_CTL_INET_REGISTER_CALLBACK => cont!(services::netctl_register_callback(ctx, st)),
        sv_nid::RTC_GET_CURRENT_CLOCK_LOCAL_TIME => {
            cont!(services::rtc_get_current_clock_local_time(ctx, st))
        }
        sv_nid::APPUTIL_SYSTEM_PARAM_GET_INT => cont!(services::apputil_system_param_get_int(ctx, st)),
        sv_nid::TOUCH_READ => cont!(touch::read(ctx, st)),
        sv_nid::TOUCH_PEEK => cont!(touch::peek(ctx, st)),
        // No online account off-console: identity calls report signed-out so the
        // title takes its offline path instead of dereferencing a null identity.
        sv_nid::NP_MANAGER_GET_NP_ID | sv_nid::NP_SCORE_CREATE_TITLE_CTX => {
            cont!(ctx.ret(services::SCE_NP_ERROR_SIGNED_OUT as u32))
        }
        // Everything else here is an init/register that simply succeeds offline.
        sv_nid::NET_INIT
        | sv_nid::NET_CTL_INIT
        | sv_nid::HTTP_INIT
        | sv_nid::SSL_INIT
        | sv_nid::NP_INIT
        | sv_nid::NP_REGISTER_SERVICE_STATE_CALLBACK
        | sv_nid::NP_BASIC_INIT
        | sv_nid::NP_BASIC_REGISTER_HANDLER
        | sv_nid::FIOS_OVERLAY_GET_LIST
        | sv_nid::ULOBJ_REGISTER_PROTOCOL_REVISION
        | sv_nid::APPUTIL_INIT
        | sv_nid::NP_SCORE_INIT
        | sv_nid::TOUCH_SET_SAMPLING_STATE => cont!(ctx.ret(0)),

        _ => {
            st.capture.note_unimplemented(library_nid, func_nid, nid::name(func_nid));
            ctx.ret(0);
            SvcOutcome::Continue
        }
    };
    // Diagnostic (env `VITASLOP_DBG_ERR`): log any handler that returns an SCE error
    // code (top bit set) - the fastest way to find an HLE call whose failure sends the
    // guest down an unexpected (error/cleanup) path.
    if dbg_err {
        let r = ctx.regs[0];
        if r & 0x8000_0000 != 0 {
            eprintln!(
                "ERR_RET thid={} {} nid={func_nid:#010x} -> {r:#010x}",
                st.current_thread(),
                nid::name(func_nid)
            );
        }
    }
    outcome
}
