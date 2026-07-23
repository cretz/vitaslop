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
pub mod pvf;
pub mod services;
pub mod sync;
pub mod sysmem;
pub mod threadmgr;
pub mod touch;

use crate::host::{GuestCtx, VitaState};
use crate::nid::{
    audio as audio_nid, ctrl as ctrl_nid, display as display_nid, gxm as gxm_nid,
    iofilemgr as io_nid, libkernel as lk_nid, lwsync as lw_nid, ngs as ngs_nid,
    processmgr as pm_nid, pvf as pvf_nid, services as sv_nid, sync as sync_nid,
    sysmem as sm_nid, threadmgr as tm_nid,
};
use crate::{nid, SvcOutcome};

use std::collections::BTreeMap;
use std::sync::{LazyLock, Mutex};

/// Diagnostic call-site profiler (env `VITASLOP_DBG_CALLSITES`): counts host calls
/// keyed by (function NID, guest return address). A busy-wait spin shows up as one
/// (nid, lr) pair with an enormous count - the exact instruction to investigate.
static DBG_CALLSITES: LazyLock<bool> =
    LazyLock::new(|| std::env::var("VITASLOP_DBG_CALLSITES").is_ok());
static CALLSITE_HIST: Mutex<BTreeMap<(u32, u32), u64>> = Mutex::new(BTreeMap::new());

/// Ordered-timeline trace (env `VITASLOP_TRACE_ORDER`): print every *meaningful*
/// host call live, in global order, with a monotonic index and thread id. Unlike
/// the counting profiler this shows the boot NARRATIVE and the exact point it
/// flatlines into a pure lock/poll spin. The high-frequency lock/unlock and shader-
/// reflection calls are filtered so the interesting sequence is not drowned out.
static TRACE_ORDER: LazyLock<bool> =
    LazyLock::new(|| std::env::var("VITASLOP_TRACE_ORDER").is_ok());
static TRACE_ORDER_SEQ: std::sync::atomic::AtomicU64 = std::sync::atomic::AtomicU64::new(0);

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

    // Diagnostic (env `VITASLOP_TRACE_ORDER`): live, globally-ordered timeline of
    // meaningful calls. Filters the lock/unlock and shader-reflection storm so the
    // boot sequence and its flatline-into-spin are legible. Zero cost when unset.
    if *TRACE_ORDER {
        let nm = nid::name(func_nid);
        let noise = nm.contains("LwMutex")
            || nm.contains("LockMutex")
            || nm.contains("UnlockMutex")
            || nm.starts_with("sceGxmProgram")
            || nm == "sceKernelGetTLSAddr";
        if !noise {
            let seq = TRACE_ORDER_SEQ.fetch_add(1, std::sync::atomic::Ordering::Relaxed);
            let label = if nm == "<unknown>" {
                format!("nid:{func_nid:#010x}")
            } else {
                nm.to_string()
            };
            eprintln!(
                "[ord {seq:>7} t{:<3}] {label}({:#x}, {:#x}, {:#x}, {:#x}) lr={:#010x}",
                st.current_thread(),
                ctx.arg(0),
                ctx.arg(1),
                ctx.arg(2),
                ctx.arg(3),
                ctx.regs[14],
            );
        }
    }

    // Diagnostic (`RUST_LOG=vitaslop::ngs=trace`): log every NGS and sceAudioOut
    // call with its first four args and caller, to see exactly how a title feeds AT9
    // data to a voice and where the final mix goes.
    if library_nid == nid::lib::SCE_NGS || library_nid == nid::lib::SCE_AUDIO {
        tracing::trace!(
            target: "vitaslop::ngs",
            name = nid::name(func_nid),
            a0 = format_args!("{:#010x}", ctx.arg(0)),
            a1 = format_args!("{:#010x}", ctx.arg(1)),
            a2 = format_args!("{:#010x}", ctx.arg(2)),
            a3 = format_args!("{:#010x}", ctx.arg(3)),
            lr = format_args!("{:#010x}", ctx.regs[14]),
            "call"
        );
    }
    let outcome = match func_nid {
        // --- lwsync: lightweight mutex / cond (the hottest surface) --------------
        lw_nid::CREATE_LW_MUTEX => cont!(lwsync::create_lw_mutex(ctx, st)),
        lw_nid::CREATE_LW_COND => cont!(lwsync::create_lw_cond(ctx, st)),
        lw_nid::WAIT_LW_COND | lw_nid::WAIT_LW_COND_CB => lwsync::wait_lw_cond(ctx, st),
        lw_nid::SIGNAL_LW_COND => cont!(lwsync::signal_lw_cond(ctx, st, false)),
        // SignalLwCondAll wakes every waiter; SignalLwCondTo targets one thread,
        // approximated by a broadcast (a spurious wake re-checks and re-waits).
        lw_nid::SIGNAL_LW_COND_ALL | lw_nid::SIGNAL_LW_COND_TO => {
            cont!(lwsync::signal_lw_cond(ctx, st, true))
        }
        // A lightweight mutex genuinely blocks on contention and enforces mutual
        // exclusion (keyed by its guest work-area address). The `_CB` lock variant
        // additionally processes pending callbacks - none are queued in this model, so
        // it takes the same path.
        lw_nid::LOCK_LW_MUTEX | lw_nid::LOCK_LW_MUTEX_CB => lwsync::lock_lw_mutex(ctx, st, false),
        lw_nid::TRY_LOCK_LW_MUTEX => lwsync::lock_lw_mutex(ctx, st, true),
        lw_nid::UNLOCK_LW_MUTEX | lw_nid::UNLOCK_LW_MUTEX2 => {
            cont!(lwsync::unlock_lw_mutex(ctx, st))
        }
        lw_nid::DELETE_LW_MUTEX => cont!(lwsync::delete_lw_mutex(ctx, st)),
        // A lightweight cond has no persistent host record beyond its parked waiters
        // (keyed by work address in `wait_lw_cond`/`signal_lw_cond`), so delete is a
        // bare success.
        lw_nid::DELETE_LW_COND => cont!(lwsync::succeed(ctx)),

        // --- sync: heavyweight mutex / sema / cond / event flag -----------------
        sync_nid::CREATE_MUTEX => cont!(sync::create_mutex(ctx, st)),
        // Lock and wait can block under the preemptive scheduler (Block parks).
        sync_nid::LOCK_MUTEX => sync::lock_mutex(ctx, st, false),
        sync_nid::TRY_LOCK_MUTEX => sync::lock_mutex(ctx, st, true),
        sync_nid::UNLOCK_MUTEX => cont!(sync::unlock_mutex(ctx, st)),
        sync_nid::DELETE_MUTEX => cont!(sync::delete_object(ctx, st)),
        sync_nid::CREATE_SEMA | sync_nid::CREATE_SEMA_16XX => cont!(sync::create_sema(ctx, st)),
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
        // A real wait: parks under the preemptive scheduler until SetEventFlag
        // satisfies the pattern (or the timeout passes).
        sync_nid::WAIT_EVENT_FLAG | sync_nid::WAIT_EVENT_FLAG_CB => sync::wait_event_flag(ctx, st),
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
        lk_nid::GET_PROCESS_TIME => libkernel::get_process_time(ctx, st),
        lk_nid::GET_PROCESS_TIME_WIDE => libkernel::get_process_time_wide(ctx, st),
        lk_nid::EXIT_PROCESS => {
            // r0 (exit code) is left as the guest set it; any exit is a clean stop.
            libkernel::trace_exit(ctx, st);
            SvcOutcome::Halt
        }

        // --- threadmgr: delay, exit, process id --------------------------------
        // A real timed sleep: parks under the preemptive scheduler (see the handler).
        tm_nid::DELAY_THREAD => threadmgr::delay_thread(ctx, st),
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
        tm_nid::GET_THREAD_CURRENT_PRIORITY => {
            cont!(threadmgr::get_thread_current_priority(ctx, st))
        }
        // Closing a semaphore invalidates its id, same as deleting it in this model.
        tm_nid::CLOSE_SEMA => cont!(sync::delete_object(ctx, st)),
        tm_nid::CHANGE_THREAD_VFP_EXCEPTION => cont!(threadmgr::change_thread_vfp_exception(ctx, st)),

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
        gxm_nid::PROGRAM_PARAMETER_GET_RESOURCE_INDEX => cont!(gxm::param_get_resource_index(ctx)),
        gxm_nid::PROGRAM_FIND_PARAMETER_BY_NAME => cont!(gxm::find_parameter(ctx, st)),
        gxm_nid::PROGRAM_GET_PARAMETER_COUNT => cont!(gxm::program_get_parameter_count(ctx)),
        gxm_nid::PROGRAM_GET_PARAMETER => cont!(gxm::program_get_parameter(ctx)),
        gxm_nid::PROGRAM_PARAMETER_GET_CATEGORY => cont!(gxm::param_get_category(ctx)),
        gxm_nid::PROGRAM_PARAMETER_GET_TYPE => cont!(gxm::param_get_type(ctx)),
        gxm_nid::PROGRAM_PARAMETER_GET_COMPONENT_COUNT => cont!(gxm::param_get_component_count(ctx)),
        gxm_nid::PROGRAM_PARAMETER_GET_CONTAINER_INDEX => cont!(gxm::param_get_container_index(ctx)),
        gxm_nid::PROGRAM_PARAMETER_GET_ARRAY_SIZE => cont!(gxm::param_get_array_size(ctx)),
        gxm_nid::PROGRAM_PARAMETER_GET_NAME => cont!(gxm::param_get_name(ctx)),
        gxm_nid::COLOR_SURFACE_INIT => cont!(gxm::color_surface_init(ctx, st)),
        gxm_nid::SHADER_PATCHER_CREATE_VERTEX_PROGRAM => cont!(gxm::create_vertex_program(ctx, st)),
        gxm_nid::SHADER_PATCHER_CREATE_FRAGMENT_PROGRAM => cont!(gxm::create_fragment_program(ctx, st)),
        gxm_nid::BEGIN_SCENE => cont!(gxm::begin_scene(ctx, st)),
        gxm_nid::END_SCENE => cont!(gxm::end_scene(ctx, st)),
        gxm_nid::SET_VERTEX_PROGRAM => cont!(gxm::set_vertex_program(ctx, st)),
        gxm_nid::RESERVE_VERTEX_DEFAULT_UNIFORM_BUFFER => cont!(gxm::reserve_vertex_uniforms(ctx, st)),
        gxm_nid::RESERVE_FRAGMENT_DEFAULT_UNIFORM_BUFFER => cont!(gxm::reserve_fragment_uniforms(ctx, st)),
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
        // Texture filters: record the sticky min/mag/mip filter per texture so the
        // getters read them back (and a future renderer can sample faithfully).
        gxm_nid::TEXTURE_SET_MIN_FILTER => cont!(gxm::texture_set_min_filter(ctx, st)),
        gxm_nid::TEXTURE_SET_MAG_FILTER => cont!(gxm::texture_set_mag_filter(ctx, st)),
        gxm_nid::TEXTURE_SET_MIP_FILTER => cont!(gxm::texture_set_mip_filter(ctx, st)),
        gxm_nid::TEXTURE_SET_GAMMA_MODE => cont!(gxm::texture_set_gamma_mode(ctx, st)),
        gxm_nid::SET_FRAGMENT_UNIFORM_BUFFER => cont!(gxm::ok(ctx)),
        // Texture getters: read back the sticky sampler/format state a setter stored.
        gxm_nid::TEXTURE_GET_MIPMAP_COUNT_UNSAFE => cont!(gxm::texture_get_mipmap_count(ctx, st)),
        gxm_nid::TEXTURE_GET_STRIDE => cont!(gxm::texture_get_stride(ctx, st)),
        gxm_nid::TEXTURE_GET_LOD_BIAS => cont!(gxm::texture_get_lod_bias(ctx, st)),
        gxm_nid::TEXTURE_GET_U_ADDR_MODE_SAFE => cont!(gxm::texture_get_u_addr_mode(ctx, st)),
        gxm_nid::TEXTURE_GET_V_ADDR_MODE_SAFE => cont!(gxm::texture_get_v_addr_mode(ctx, st)),
        gxm_nid::TEXTURE_GET_MIN_FILTER => cont!(gxm::texture_get_min_filter(ctx, st)),
        gxm_nid::TEXTURE_GET_MAG_FILTER => cont!(gxm::texture_get_mag_filter(ctx, st)),
        gxm_nid::TEXTURE_GET_GAMMA_MODE => cont!(gxm::texture_get_gamma_mode(ctx, st)),
        gxm_nid::TEXTURE_INIT_CUBE => cont!(gxm::texture_init(ctx, st, gxm::TYPE_CUBE)),
        // Color-surface getters/setters beyond format.
        gxm_nid::COLOR_SURFACE_GET_DATA => cont!(gxm::color_surface_get_data(ctx, st)),
        gxm_nid::COLOR_SURFACE_GET_STRIDE_IN_PIXELS => {
            cont!(gxm::color_surface_get_stride_in_pixels(ctx, st))
        }
        gxm_nid::COLOR_SURFACE_SET_GAMMA_MODE => cont!(gxm::color_surface_set_gamma_mode(ctx, st)),
        // Render-target sizing + GPU notification region + program reflection.
        gxm_nid::GET_RENDER_TARGET_MEM_SIZE => cont!(gxm::get_render_target_mem_size(ctx, st)),
        gxm_nid::GET_NOTIFICATION_REGION => cont!(gxm::get_notification_region(ctx, st)),
        gxm_nid::PROGRAM_GET_DEFAULT_UNIFORM_BUFFER_SIZE => {
            cont!(gxm::program_get_default_uniform_buffer_size(ctx, st))
        }
        gxm_nid::FRAGMENT_PROGRAM_GET_PASS_TYPE => cont!(gxm::fragment_program_get_pass_type(ctx, st)),
        // Precomputed draws: record the bundle, replay it as a draw on DrawPrecomputed.
        gxm_nid::GET_PRECOMPUTED_DRAW_SIZE => cont!(gxm::get_precomputed_draw_size(ctx, st)),
        gxm_nid::PRECOMPUTED_DRAW_INIT => cont!(gxm::precomputed_draw_init(ctx, st)),
        gxm_nid::PRECOMPUTED_DRAW_SET_VERTEX_STREAM => {
            cont!(gxm::precomputed_draw_set_vertex_stream(ctx, st))
        }
        gxm_nid::PRECOMPUTED_DRAW_SET_PARAMS => cont!(gxm::precomputed_draw_set_params(ctx, st)),
        gxm_nid::DRAW_PRECOMPUTED => cont!(gxm::draw_precomputed(ctx, st)),
        gxm_nid::GET_PRECOMPUTED_VERTEX_STATE_SIZE => cont!(gxm::get_precomputed_vertex_state_size(ctx, st)),
        gxm_nid::GET_PRECOMPUTED_FRAGMENT_STATE_SIZE => cont!(gxm::get_precomputed_fragment_state_size(ctx, st)),
        gxm_nid::PRECOMPUTED_VERTEX_STATE_INIT => cont!(gxm::precomputed_vertex_state_init(ctx, st)),
        gxm_nid::PRECOMPUTED_FRAGMENT_STATE_INIT => cont!(gxm::precomputed_fragment_state_init(ctx, st)),
        gxm_nid::PRECOMPUTED_VERTEX_STATE_SET_DEFAULT_UNIFORM_BUFFER => {
            cont!(gxm::precomputed_vertex_state_set_default_uniform_buffer(ctx, st))
        }
        gxm_nid::PRECOMPUTED_FRAGMENT_STATE_SET_DEFAULT_UNIFORM_BUFFER => {
            cont!(gxm::precomputed_fragment_state_set_default_uniform_buffer(ctx, st))
        }
        gxm_nid::PRECOMPUTED_VERTEX_STATE_GET_DEFAULT_UNIFORM_BUFFER => {
            cont!(gxm::precomputed_vertex_state_get_default_uniform_buffer(ctx, st))
        }
        gxm_nid::PRECOMPUTED_FRAGMENT_STATE_GET_DEFAULT_UNIFORM_BUFFER => {
            cont!(gxm::precomputed_fragment_state_get_default_uniform_buffer(ctx, st))
        }
        gxm_nid::PRECOMPUTED_VERTEX_STATE_SET_TEXTURE => cont!(gxm::precomputed_vertex_state_set_texture(ctx, st)),
        gxm_nid::PRECOMPUTED_FRAGMENT_STATE_SET_TEXTURE => cont!(gxm::precomputed_fragment_state_set_texture(ctx, st)),
        gxm_nid::SET_PRECOMPUTED_VERTEX_STATE => cont!(gxm::set_precomputed_vertex_state(ctx, st)),
        gxm_nid::SET_PRECOMPUTED_FRAGMENT_STATE => cont!(gxm::set_precomputed_fragment_state(ctx, st)),
        // Fixed-function pipeline state: record into the sticky render state that is
        // snapshotted per draw (see `capture::RenderState`).
        gxm_nid::SET_CULL_MODE => cont!(gxm::set_cull_mode(ctx, st)),
        gxm_nid::SET_TWO_SIDED_ENABLE => cont!(gxm::set_two_sided_enable(ctx, st)),
        gxm_nid::SET_FRONT_DEPTH_FUNC => cont!(gxm::set_front_depth_func(ctx, st)),
        gxm_nid::SET_BACK_DEPTH_FUNC => cont!(gxm::set_back_depth_func(ctx, st)),
        gxm_nid::SET_FRONT_DEPTH_WRITE_ENABLE => cont!(gxm::set_front_depth_write_enable(ctx, st)),
        gxm_nid::SET_FRONT_FRAGMENT_PROGRAM_ENABLE => {
            cont!(gxm::set_front_fragment_program_enable(ctx, st))
        }
        gxm_nid::SET_BACK_FRAGMENT_PROGRAM_ENABLE => {
            cont!(gxm::set_back_fragment_program_enable(ctx, st))
        }
        gxm_nid::SET_FRONT_POINT_LINE_WIDTH => cont!(gxm::set_front_point_line_width(ctx, st)),
        gxm_nid::SET_FRONT_POLYGON_MODE => cont!(gxm::set_front_polygon_mode(ctx, st)),
        gxm_nid::SET_FRONT_STENCIL_REF => cont!(gxm::set_front_stencil_ref(ctx, st)),
        gxm_nid::SET_FRONT_STENCIL_FUNC => cont!(gxm::set_front_stencil_func(ctx, st)),
        gxm_nid::SET_VIEWPORT => cont!(gxm::set_viewport(ctx, st)),
        gxm_nid::SET_VIEWPORT_ENABLE => cont!(gxm::set_viewport_enable(ctx, st)),
        gxm_nid::SET_REGION_CLIP => cont!(gxm::set_region_clip(ctx, st)),
        gxm_nid::COLOR_SURFACE_GET_FORMAT => cont!(gxm::color_surface_get_format(ctx, st)),
        gxm_nid::COLOR_SURFACE_GET_TYPE => cont!(gxm::color_surface_get_type(ctx, st)),
        gxm_nid::COLOR_SURFACE_SET_CLIP => cont!(gxm::color_surface_set_clip(ctx, st)),
        gxm_nid::TEXTURE_GET_TYPE => cont!(gxm::texture_get_type(ctx, st)),
        gxm_nid::PROGRAM_PARAMETER_GET_SEMANTIC => cont!(gxm::param_get_semantic(ctx, st)),
        gxm_nid::PROGRAM_PARAMETER_GET_SEMANTIC_INDEX => {
            cont!(gxm::param_get_semantic_index(ctx, st))
        }
        // Texture sampler state: record wrap modes / LOD bias per texture (the plain
        // and "safe" variants set the same state; the safe one also validates on HW).
        gxm_nid::TEXTURE_SET_U_ADDR_MODE | gxm_nid::TEXTURE_SET_U_ADDR_MODE_SAFE => {
            cont!(gxm::texture_set_u_addr_mode(ctx, st))
        }
        gxm_nid::TEXTURE_SET_V_ADDR_MODE | gxm_nid::TEXTURE_SET_V_ADDR_MODE_SAFE => {
            cont!(gxm::texture_set_v_addr_mode(ctx, st))
        }
        gxm_nid::TEXTURE_SET_LOD_BIAS => cont!(gxm::texture_set_lod_bias(ctx, st)),
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
        io_nid::IO_GETSTAT_BY_FD => cont!(iofilemgr::io_getstat_by_fd(ctx, st)),
        io_nid::IO_MKDIR => cont!(iofilemgr::io_mkdir(ctx, st)),
        io_nid::IO_REMOVE => cont!(iofilemgr::io_remove(ctx, st)),
        io_nid::IO_DOPEN => cont!(iofilemgr::io_dopen(ctx, st)),
        io_nid::IO_DREAD => cont!(iofilemgr::io_dread(ctx, st)),
        io_nid::IO_DCLOSE => cont!(iofilemgr::io_dclose(ctx, st)),

        // --- sysmem: memory blocks ---------------------------------------------
        sm_nid::ALLOC_MEM_BLOCK => cont!(sysmem::alloc_mem_block(ctx, st)),
        sm_nid::GET_MEM_BLOCK_BASE => cont!(sysmem::get_mem_block_base(ctx, st)),
        sm_nid::FREE_MEM_BLOCK => cont!(sysmem::free_mem_block(ctx, st)),

        // --- display ------------------------------------------------------------
        display_nid::SET_FRAME_BUF => cont!(display::set_frame_buf(ctx, st)),
        // A real timed vblank wait (parks under the preemptive scheduler).
        display_nid::WAIT_VBLANK_START_MULTI => display::wait_vblank_start_multi(ctx, st),
        display_nid::WAIT_SET_FRAME_BUF => display::wait_set_frame_buf(ctx, st),

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
        | ngs_nid::VOICE_DEF_GET_EQ_BUSS
        | ngs_nid::VOICE_DEF_GET_SIMPLE_VOICE
        | ngs_nid::VOICE_DEF_GET_MIXER_BUSS => cont!(ngs::voice_def_get(ctx, st)),
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
        | ngs_nid::VOICE_PATCH_SET_VOLUME
        | ngs_nid::PATCH_GET_INFO
        | ngs_nid::PATCH_REMOVE_ROUTING
        // System lock/unlock guard the mix graph; single-thread-of-control here, so
        // there is no contention - both succeed immediately.
        | ngs_nid::SYSTEM_LOCK
        | ngs_nid::SYSTEM_UNLOCK
        | ngs_nid::AT9_GET_SECTION_DETAILS => cont!(ctx.ret(0)),
        ngs_nid::VOICE_PLAY => cont!(ngs::voice_play(ctx, st)),
        ngs_nid::VOICE_KEY_OFF | ngs_nid::VOICE_KILL | ngs_nid::VOICE_PAUSE => {
            cont!(ngs::voice_stop(ctx, st))
        }
        audio_nid::OUT_OPEN_PORT => cont!(audio::out_open_port(ctx, st)),
        audio_nid::OUT_OUTPUT => audio::out_output(ctx, st),
        audio_nid::OUT_SET_VOLUME => cont!(audio::out_set_volume(ctx, st)),
        audio_nid::OUT_RELEASE_PORT => cont!(audio::out_release_port(ctx, st)),
        audio_nid::OUT_GET_ADOPT => cont!(audio::out_get_adopt(ctx, st)),

        // --- pvf: font library --------------------------------------------------
        pvf_nid::NEW_LIB => cont!(pvf::new_lib(ctx, st)),
        pvf_nid::DONE_LIB => cont!(pvf::done_lib(ctx, st)),
        pvf_nid::OPEN => cont!(pvf::open(ctx, st)),
        pvf_nid::OPEN_USER_FILE => cont!(pvf::open_user_file(ctx, st)),
        pvf_nid::SET_EM => cont!(pvf::set_em(ctx, st)),
        pvf_nid::SET_RESOLUTION => cont!(pvf::set_resolution(ctx, st)),
        pvf_nid::SET_CHAR_SIZE => cont!(pvf::set_char_size(ctx, st)),
        pvf_nid::SET_SKEW_VALUE => cont!(pvf::set_skew_value(ctx, st)),
        pvf_nid::IS_ELEMENT => cont!(pvf::is_element(ctx, st)),
        pvf_nid::GET_FONT_INFO => cont!(pvf::get_font_info(ctx, st)),
        pvf_nid::GET_CHAR_INFO => cont!(pvf::get_char_info(ctx, st)),
        pvf_nid::GET_CHAR_IMAGE_RECT => cont!(pvf::get_char_image_rect(ctx, st)),
        pvf_nid::GET_CHAR_GLYPH_IMAGE => cont!(pvf::get_char_glyph_image(ctx, st)),
        pvf_nid::PIXEL_TO_POINT_H => cont!(pvf::pixel_to_point_h(ctx, st)),
        pvf_nid::PIXEL_TO_POINT_V => cont!(pvf::pixel_to_point_v(ctx, st)),

        // --- processmgr: process param, std streams, time ----------------------
        pm_nid::GET_PROCESS_PARAM => cont!(processmgr::get_process_param(ctx, st)),
        pm_nid::GET_STDIN => cont!(processmgr::get_stdin(ctx, st)),
        pm_nid::GET_STDOUT => cont!(processmgr::get_stdout(ctx, st)),
        pm_nid::GET_STDERR => cont!(processmgr::get_stderr(ctx, st)),
        pm_nid::LIBC_TIME => cont!(processmgr::libc_time(ctx, st)),
        pm_nid::LIBC_CLOCK => cont!(processmgr::libc_clock(ctx, st)),
        pm_nid::POWER_TICK => cont!(ctx.ret(0)),

        // --- services: sysmodule / net / http / np / rtc / apputil / touch -----
        sv_nid::SYSMODULE_IS_LOADED => cont!(services::sysmodule_is_loaded(ctx, st)),
        sv_nid::NET_CTL_INET_GET_STATE => cont!(services::netctl_inet_get_state(ctx, st)),
        sv_nid::NET_CTL_INET_REGISTER_CALLBACK => cont!(services::netctl_register_callback(ctx, st)),
        sv_nid::NET_CTL_CHECK_CALLBACK => cont!(services::net_check_callback(ctx, st)),
        sv_nid::NP_REGISTER_SERVICE_STATE_CALLBACK => {
            cont!(services::np_register_service_state_callback(ctx, st))
        }
        sv_nid::NP_CHECK_CALLBACK => cont!(services::np_check_callback(ctx, st)),
        sv_nid::RTC_GET_CURRENT_CLOCK => cont!(services::rtc_get_current_clock(ctx, st)),
        sv_nid::RTC_GET_CURRENT_CLOCK_LOCAL_TIME => {
            cont!(services::rtc_get_current_clock_local_time(ctx, st))
        }
        sv_nid::RTC_GET_CURRENT_TICK => cont!(services::rtc_get_current_tick(ctx, st)),
        sv_nid::RTC_GET_TICK => cont!(services::rtc_get_tick(ctx, st)),
        sv_nid::MOTION_GET_STATE => cont!(services::motion_get_state(ctx, st)),
        sv_nid::APPUTIL_SYSTEM_PARAM_GET_INT => cont!(services::apputil_system_param_get_int(ctx, st)),
        sv_nid::APPUTIL_APP_PARAM_GET_INT => cont!(services::apputil_app_param_get_int(ctx, st)),
        sv_nid::LIVE_AREA_GET_STATUS => cont!(services::live_area_get_status(ctx, st)),
        sv_nid::APPUTIL_SYSTEM_PARAM_GET_STRING => cont!(services::apputil_system_param_get_string(ctx, st)),
        sv_nid::APPUTIL_DRM_OPEN => cont!(services::apputil_drm_open(ctx, st)),
        sv_nid::APPUTIL_DRM_CLOSE => cont!(services::apputil_drm_close(ctx, st)),
        sv_nid::APPUTIL_SAVEDATA_SLOT_GET_PARAM => {
            cont!(services::apputil_savedata_slot_get_param(ctx, st))
        }
        sv_nid::APPUTIL_SAVEDATA_SLOT_CREATE => {
            cont!(services::apputil_savedata_slot_create(ctx, st))
        }
        sv_nid::APP_MGR_GET_APP_STATE => cont!(services::app_mgr_get_app_state(ctx, st)),
        // Offline services with an out-param handle to hand back.
        sv_nid::NETCTL_ADHOC_REGISTER_CALLBACK => {
            cont!(services::netctl_adhoc_register_callback(ctx, st))
        }
        sv_nid::NP_TROPHY_CREATE_CONTEXT => cont!(services::np_trophy_create_context(ctx, st)),
        sv_nid::NP_TROPHY_CREATE_HANDLE => cont!(services::np_trophy_create_handle(ctx, st)),
        sv_nid::NP_TROPHY_GET_GAME_INFO => cont!(services::np_trophy_get_game_info(ctx, st)),
        sv_nid::NP_TROPHY_GET_TROPHY_UNLOCK_STATE => cont!(services::np_trophy_get_trophy_unlock_state(ctx, st)),
        // The trophy-setup dialog's result read (zeroed result = OK), like the other
        // dialog GetResult calls.
        sv_nid::NP_TROPHY_SETUP_DIALOG_GET_RESULT => cont!(services::dialog_ok(ctx, st)),
        sv_nid::TOUCH_READ => cont!(touch::read(ctx, st)),
        sv_nid::TOUCH_PEEK => cont!(touch::peek(ctx, st)),
        sv_nid::TOUCH_GET_PANEL_INFO => cont!(touch::get_panel_info(ctx, st)),
        // No online account off-console: identity calls report signed-out so the
        // title takes its offline path instead of dereferencing a null identity.
        // sceNpManagerGetAccountRegion is an account-identity query (account
        // country + language); with no account off-console the faithful signal is
        // signed-out, same as GetNpId, not a fabricated region.
        sv_nid::NP_MANAGER_GET_NP_ID
        | sv_nid::NP_MANAGER_GET_ACCOUNT_REGION
        | sv_nid::NP_MANAGER_GET_CONTENT_RATING_FLAG
        | sv_nid::NP_MANAGER_GET_CHAT_RESTRICTION_FLAG
        | sv_nid::NP_SCORE_CREATE_TITLE_CTX => {
            cont!(ctx.ret(services::SCE_NP_ERROR_SIGNED_OUT as u32))
        }
        // Everything else here is an init/register that simply succeeds offline.
        sv_nid::NET_INIT
        | sv_nid::NET_CTL_INIT
        | sv_nid::HTTP_INIT
        | sv_nid::SSL_INIT
        | sv_nid::NP_INIT
        | sv_nid::NP_BASIC_INIT
        | sv_nid::NP_BASIC_REGISTER_HANDLER
        // NpBasic per-frame pump: no presence/friend events exist off-console.
        | sv_nid::NP_BASIC_CHECK_CALLBACK
        | sv_nid::FIOS_OVERLAY_GET_LIST
        | sv_nid::ULOBJ_REGISTER_PROTOCOL_REVISION
        | sv_nid::APPUTIL_INIT
        | sv_nid::NP_SCORE_INIT
        // The requested module is already linked into the image, so a load succeeds.
        | sv_nid::SYSMODULE_LOAD_MODULE
        | sv_nid::TOUCH_SET_SAMPLING_STATE
        | sv_nid::TOUCH_ENABLE_TOUCH_FORCE
        // SceScreenShot: nothing to capture off-console.
        | sv_nid::SCREENSHOT_DISABLE
        | sv_nid::SCREENSHOT_ENABLE
        | sv_nid::SCREENSHOT_SET_PARAM
        | sv_nid::SCREENSHOT_SET_OVERLAY_IMAGE
        // SceNpTrophy init + the Np subsystem inits: offline success.
        | sv_nid::NP_TROPHY_INIT
        | sv_nid::NP_ACTIVITY_INIT
        | sv_nid::NP_AUTH_INIT
        | sv_nid::NP_LOOKUP_INIT
        | sv_nid::NP_TUS_INIT
        | sv_nid::NP_SNS_FACEBOOK_INIT
        // Device services: location/motion sampling, ad-hoc power/config.
        | sv_nid::LOCATION_INIT
        | sv_nid::MOTION_START_SAMPLING
        | sv_nid::POWER_SET_CONFIGURATION_MODE
        // Shared dialog config accepted for every family.
        | sv_nid::COMMON_DIALOG_SET_CONFIG_PARAM
        // SceLiveArea: the app's home-screen tile. No home screen exists off-console,
        // so a frame update is an accepted no-op (the async variant has no completion
        // to deliver - there is no LiveArea state that changes).
        | sv_nid::LIVE_AREA_UPDATE_FRAME_ASYNC
        // Unnamed exports absent from every vita-headers revision, serviced as an
        // offline no-op success so they are handled rather than left as gaps.
        | sv_nid::NEAR_UTIL_UNKNOWN_A412E9CA
        | lk_nid::UNKNOWN_023EAA62 => cont!(ctx.ret(0)),

        // --- SceCommonDialog: system dialogs complete instantly offline ---------
        sv_nid::MSG_DIALOG_INIT => cont!(services::dialog_init(ctx, st, services::DialogFamily::Msg)),
        sv_nid::MSG_DIALOG_GET_STATUS => cont!(services::dialog_get_status(ctx, st, services::DialogFamily::Msg)),
        sv_nid::MSG_DIALOG_TERM => cont!(services::dialog_term(ctx, st, services::DialogFamily::Msg)),
        sv_nid::NET_CHECK_DIALOG_INIT => cont!(services::dialog_init(ctx, st, services::DialogFamily::NetCheck)),
        sv_nid::NET_CHECK_DIALOG_GET_STATUS => cont!(services::dialog_get_status(ctx, st, services::DialogFamily::NetCheck)),
        sv_nid::NET_CHECK_DIALOG_TERM => cont!(services::dialog_term(ctx, st, services::DialogFamily::NetCheck)),
        sv_nid::SAVEDATA_DIALOG_INIT => cont!(services::dialog_init(ctx, st, services::DialogFamily::SaveData)),
        sv_nid::SAVEDATA_DIALOG_GET_STATUS | sv_nid::SAVEDATA_DIALOG_GET_SUB_STATUS => {
            cont!(services::dialog_get_status(ctx, st, services::DialogFamily::SaveData))
        }
        sv_nid::SAVEDATA_DIALOG_TERM => cont!(services::dialog_term(ctx, st, services::DialogFamily::SaveData)),
        sv_nid::NP_MESSAGE_DIALOG_INIT => cont!(services::dialog_init(ctx, st, services::DialogFamily::NpMessage)),
        sv_nid::NP_MESSAGE_DIALOG_GET_STATUS => cont!(services::dialog_get_status(ctx, st, services::DialogFamily::NpMessage)),
        sv_nid::NP_MESSAGE_DIALOG_TERM | sv_nid::NP_MESSAGE_DIALOG_ABORT => {
            cont!(services::dialog_term(ctx, st, services::DialogFamily::NpMessage))
        }
        sv_nid::NP_TROPHY_SETUP_DIALOG_INIT => cont!(services::dialog_init(ctx, st, services::DialogFamily::NpTrophySetup)),
        sv_nid::NP_TROPHY_SETUP_DIALOG_GET_STATUS => cont!(services::dialog_get_status(ctx, st, services::DialogFamily::NpTrophySetup)),
        sv_nid::NP_TROPHY_SETUP_DIALOG_TERM => cont!(services::dialog_term(ctx, st, services::DialogFamily::NpTrophySetup)),
        sv_nid::STORE_CHECKOUT_DIALOG_INIT => cont!(services::dialog_init(ctx, st, services::DialogFamily::StoreCheckout)),
        sv_nid::STORE_CHECKOUT_DIALOG_GET_STATUS => cont!(services::dialog_get_status(ctx, st, services::DialogFamily::StoreCheckout)),
        sv_nid::STORE_CHECKOUT_DIALOG_TERM => cont!(services::dialog_term(ctx, st, services::DialogFamily::StoreCheckout)),
        sv_nid::NP_SNS_FACEBOOK_DIALOG_INIT => cont!(services::dialog_init(ctx, st, services::DialogFamily::NpSnsFacebook)),
        sv_nid::NP_SNS_FACEBOOK_DIALOG_GET_STATUS => cont!(services::dialog_get_status(ctx, st, services::DialogFamily::NpSnsFacebook)),
        // Result reads and per-frame pumping succeed with the caller's (zeroed)
        // result struct untouched; the update pump has no system UI to animate.
        sv_nid::COMMON_DIALOG_UPDATE
        | sv_nid::MSG_DIALOG_GET_RESULT
        | sv_nid::NET_CHECK_DIALOG_GET_RESULT
        | sv_nid::SAVEDATA_DIALOG_GET_RESULT
        | sv_nid::SAVEDATA_DIALOG_CONTINUE
        | sv_nid::SAVEDATA_DIALOG_FINISH
        | sv_nid::SAVEDATA_DIALOG_SUB_CLOSE
        | sv_nid::NP_MESSAGE_DIALOG_GET_RESULT
        | sv_nid::STORE_CHECKOUT_DIALOG_GET_RESULT
        | sv_nid::NP_SNS_FACEBOOK_DIALOG_GET_RESULT_LONG_TOKEN => {
            cont!(services::dialog_ok(ctx, st))
        }

        _ => {
            // No handler for this NID. Do NOT fake a success: a silent `ret(0)` lets
            // the guest continue on a false premise and desync into a spin or memory
            // corruption far from here (the exact failure mode this project keeps
            // hitting). Record it for the report and stop the run loudly, naming the
            // call so the fix is "implement this NID", pinpointed. Every legitimate
            // offline no-op has its own explicit arm above returning 0 deliberately;
            // reaching here means the NID is genuinely unhandled.
            st.capture.note_unimplemented(library_nid, func_nid, nid::name(func_nid));
            let name = nid::name(func_nid);
            return SvcOutcome::Fatal(format!(
                "unimplemented NID {name} (lib={library_nid:#010x} nid={func_nid:#010x}) \
                 called by thread {:#x}; implement it (no silent stub)",
                st.current_thread(),
            ));
        }
    };
    // Diagnostic (`RUST_LOG=vitaslop::err=debug`): log any handler that returns an
    // SCE error code (top bit set) - the fastest way to find an HLE call whose
    // failure sends the guest down an unexpected (error/cleanup) path.
    let r = ctx.regs[0];
    if r & 0x8000_0000 != 0 {
        tracing::debug!(
            target: "vitaslop::err",
            thid = st.current_thread(),
            name = nid::name(func_nid),
            nid = format_args!("{func_nid:#010x}"),
            ret = format_args!("{r:#010x}"),
            "error return"
        );
    }
    outcome
}
