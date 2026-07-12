//! NID constants and name resolution for the modules the cube uses. NIDs are
//! facts from the MIT vita-headers database. Grouped by module to mirror that
//! database and keep the handler files organized the same way.
//!
//! Dispatch matches on the function NID (globally unique), so a library-NID
//! mismatch cannot misroute a call; the library NID is carried only for logging.

/// Library NIDs (for logging and grouping).
pub mod lib {
    pub const SCE_GXM: u32 = 0xF76B_66BD;
    pub const SCE_DISPLAY_USER: u32 = 0x4FAA_CD11;
    pub const SCE_CTRL: u32 = 0xD197_E3C7;
    pub const SCE_SYSMEM: u32 = 0x37FE_725A;
    /// SceLibKernel: the user-facing libc/clib and process/thread wrappers
    /// (sceClib*, sceKernelExitProcess, sceKernelCreateThread, ...).
    pub const SCE_LIB_KERNEL: u32 = 0xCAE9_ACE6;
    /// SceThreadmgr: the thread-manager primitives (sceKernelDelayThread, ...).
    pub const SCE_THREADMGR: u32 = 0x859A_24B1;
    /// SceIofilemgr: the file IO library (sceIoRead/Write/Close/Lseek32). Note the
    /// user-facing sceIoOpen/Lseek/Getstat/Mkdir/Remove live under SceLibKernel;
    /// dispatch is by func NID, so the split is only cosmetic.
    pub const SCE_IO_FILEMGR: u32 = 0xF2FF_276E;
}

/// SceGxm function NIDs.
pub mod gxm {
    pub const INITIALIZE: u32 = 0xB0F1_E4EC;
    pub const TERMINATE: u32 = 0xB627_DE66;
    pub const MAP_MEMORY: u32 = 0xC61E_34FC;
    pub const MAP_VERTEX_USSE_MEMORY: u32 = 0xFA43_7510;
    pub const MAP_FRAGMENT_USSE_MEMORY: u32 = 0x0084_02C6;
    pub const CREATE_CONTEXT: u32 = 0xE84C_E5B4;
    pub const DESTROY_CONTEXT: u32 = 0xEDDC_5FB2;
    pub const CREATE_RENDER_TARGET: u32 = 0x207A_F96B;
    pub const DESTROY_RENDER_TARGET: u32 = 0x0B94_C50A;
    pub const COLOR_SURFACE_INIT: u32 = 0xED0F_6E25;
    pub const DEPTH_STENCIL_SURFACE_INIT: u32 = 0xCA9D_41D1;
    pub const SYNC_OBJECT_CREATE: u32 = 0x6A60_13E1;
    pub const SHADER_PATCHER_CREATE: u32 = 0x0503_2658;
    pub const SHADER_PATCHER_DESTROY: u32 = 0xEAA5_B100;
    pub const PROGRAM_CHECK: u32 = 0xED8B_6C69;
    pub const SHADER_PATCHER_REGISTER_PROGRAM: u32 = 0x2B52_8462;
    pub const SHADER_PATCHER_UNREGISTER_PROGRAM: u32 = 0xF103_AF8A;
    pub const SHADER_PATCHER_CREATE_VERTEX_PROGRAM: u32 = 0xB7BB_A6D5;
    pub const SHADER_PATCHER_CREATE_FRAGMENT_PROGRAM: u32 = 0x4ED2_E49D;
    pub const SHADER_PATCHER_RELEASE_VERTEX_PROGRAM: u32 = 0xAC1F_F2DA;
    pub const SHADER_PATCHER_RELEASE_FRAGMENT_PROGRAM: u32 = 0xBE27_43D1;
    pub const PROGRAM_FIND_PARAMETER_BY_NAME: u32 = 0x2777_94C4;
    pub const BEGIN_SCENE: u32 = 0x8734_FF4E;
    pub const END_SCENE: u32 = 0xFE30_0E2F;
    pub const SET_VERTEX_PROGRAM: u32 = 0x31FF_8ABD;
    pub const SET_FRAGMENT_PROGRAM: u32 = 0xAD2F_48D9;
    pub const RESERVE_VERTEX_DEFAULT_UNIFORM_BUFFER: u32 = 0x9711_8913;
    pub const SET_UNIFORM_DATA_F: u32 = 0x65DD_0C84;
    pub const SET_VERTEX_STREAM: u32 = 0x895D_F2E9;
    pub const DRAW: u32 = 0xBC05_9AFC;
    pub const PAD_HEARTBEAT: u32 = 0x3D25_FCE9;
    pub const DISPLAY_QUEUE_ADD_ENTRY: u32 = 0xEC5C_26B5;
    pub const DISPLAY_QUEUE_FINISH: u32 = 0xB98C_5B0D;
    pub const FINISH: u32 = 0x0733_D8AE;
}

/// SceDisplayUser function NIDs.
pub mod display {
    pub const SET_FRAME_BUF: u32 = 0x7A41_0B64;
}

/// SceCtrl function NIDs.
pub mod ctrl {
    pub const PEEK_BUFFER_POSITIVE: u32 = 0xA9C3_CED6;
}

/// SceSysmem (kernel memory) function NIDs.
pub mod sysmem {
    pub const ALLOC_MEM_BLOCK: u32 = 0xB9D5_EBDE;
    pub const GET_MEM_BLOCK_BASE: u32 = 0xB8EF_5818;
}

/// SceLibKernel function NIDs: the user-facing clib (string/memory/print) and the
/// process/thread wrappers. NIDs are facts from the MIT vita-headers database.
pub mod libkernel {
    // clib print family.
    pub const CLIB_PRINTF: u32 = 0xFA26_BC62;
    pub const CLIB_VPRINTF: u32 = 0x5EA3_B6CE;
    pub const CLIB_SNPRINTF: u32 = 0x8CBA_03D5;
    pub const CLIB_VSNPRINTF: u32 = 0xFA6B_E467;
    // clib memory/string family (pure, everything uses them).
    pub const CLIB_MEMCPY: u32 = 0x14E9_DBD7;
    pub const CLIB_MEMMOVE: u32 = 0x7367_53C8;
    pub const CLIB_MEMSET: u32 = 0x6329_80D7;
    pub const CLIB_MEMCMP: u32 = 0x9CC2_BFDF;
    pub const CLIB_STRNLEN: u32 = 0xAC59_5E68;
    pub const CLIB_STRNCPY: u32 = 0xC458_D60A;
    pub const CLIB_STRNCMP: u32 = 0x660D_1F6D;
    pub const CLIB_STRCMP: u32 = 0xA2FB_4D9D;
    // process control.
    pub const EXIT_PROCESS: u32 = 0x7595_D9AA;
    // thread wrappers (user-facing; the ThreadMgr primitives back them).
    pub const CREATE_THREAD: u32 = 0xC5C1_1EE7;
    pub const START_THREAD: u32 = 0xF08D_E149;
    pub const WAIT_THREAD_END: u32 = 0xDDB3_95A9;
    pub const GET_THREAD_ID: u32 = 0x0FB9_72F9;
}

/// SceThreadmgr function NIDs: thread-manager primitives not wrapped in
/// SceLibKernel.
pub mod threadmgr {
    pub const DELAY_THREAD: u32 = 0x4B67_5D05;
    pub const EXIT_DELETE_THREAD: u32 = 0x1D17_DECF;
    pub const EXIT_THREAD: u32 = 0x0C8A_38E1;
}

/// File IO (SceIoFilemgr). Function NIDs span SceIofilemgr (read/write/close/
/// lseek32) and SceLibKernel (open/lseek/getstat/mkdir/remove); grouped by concept
/// here since dispatch is by func NID.
pub mod iofilemgr {
    pub const IO_OPEN: u32 = 0x6C60_AC61;
    pub const IO_CLOSE: u32 = 0xC70B_8886;
    pub const IO_READ: u32 = 0xFDB3_2293;
    pub const IO_WRITE: u32 = 0x34EF_D876;
    pub const IO_LSEEK: u32 = 0x99BA_173E;
    pub const IO_LSEEK32: u32 = 0x4925_2B9B;
    pub const IO_GETSTAT: u32 = 0xBCA5_B623;
    pub const IO_MKDIR: u32 = 0x9670_D39F;
    pub const IO_REMOVE: u32 = 0xE20E_D0F3;
}

/// Synchronization primitives (mutex, semaphore, event flag) and system time.
/// Function NIDs span SceLibKernel (the create/lock/wait wrappers) and
/// SceThreadmgr (unlock/signal/set/clear/delete); dispatch is by func NID, so
/// they are grouped by concept here rather than by module.
pub mod sync {
    pub const CREATE_MUTEX: u32 = 0xED53_334A;
    pub const LOCK_MUTEX: u32 = 0x1D8D_7945;
    pub const TRY_LOCK_MUTEX: u32 = 0x72FC_1F54;
    pub const UNLOCK_MUTEX: u32 = 0x1A37_2EC8;
    pub const DELETE_MUTEX: u32 = 0xCB78_710D;
    pub const CREATE_SEMA: u32 = 0x1BD6_7366;
    pub const WAIT_SEMA: u32 = 0x0C7B_834B;
    pub const SIGNAL_SEMA: u32 = 0xE6B7_61D1;
    pub const DELETE_SEMA: u32 = 0xDB32_948A;
    pub const CREATE_EVENT_FLAG: u32 = 0x8516_D040;
    pub const SET_EVENT_FLAG: u32 = 0xEC94_DFF7;
    pub const WAIT_EVENT_FLAG: u32 = 0x83C0_E2AF;
    pub const CLEAR_EVENT_FLAG: u32 = 0x4CB8_7CA7;
    pub const DELETE_EVENT_FLAG: u32 = 0x5840_162C;
    pub const GET_SYSTEM_TIME_WIDE: u32 = 0xF4EE_4FA9;
    // Condition variables (SceLibKernel create/wait + SceThreadmgr signal/delete).
    pub const CREATE_COND: u32 = 0x5057_2FDA;
    pub const WAIT_COND: u32 = 0xC88D_44AD;
    pub const SIGNAL_COND: u32 = 0x6ED2_E2DC;
    pub const SIGNAL_COND_ALL: u32 = 0xC2E7_AC22;
    pub const DELETE_COND: u32 = 0x879E_6EBD;
}

/// A human-readable name for a `(library_nid, func_nid)` pair, for logging and
/// the unimplemented-call report. Falls back to the raw NIDs.
pub fn name(func_nid: u32) -> &'static str {
    use {
        ctrl as c, display as d, gxm as g, iofilemgr as io, libkernel as lk, sync as sy,
        sysmem as s, threadmgr as tm,
    };
    match func_nid {
        g::INITIALIZE => "sceGxmInitialize",
        g::TERMINATE => "sceGxmTerminate",
        g::MAP_MEMORY => "sceGxmMapMemory",
        g::MAP_VERTEX_USSE_MEMORY => "sceGxmMapVertexUsseMemory",
        g::MAP_FRAGMENT_USSE_MEMORY => "sceGxmMapFragmentUsseMemory",
        g::CREATE_CONTEXT => "sceGxmCreateContext",
        g::DESTROY_CONTEXT => "sceGxmDestroyContext",
        g::CREATE_RENDER_TARGET => "sceGxmCreateRenderTarget",
        g::DESTROY_RENDER_TARGET => "sceGxmDestroyRenderTarget",
        g::COLOR_SURFACE_INIT => "sceGxmColorSurfaceInit",
        g::DEPTH_STENCIL_SURFACE_INIT => "sceGxmDepthStencilSurfaceInit",
        g::SYNC_OBJECT_CREATE => "sceGxmSyncObjectCreate",
        g::SHADER_PATCHER_CREATE => "sceGxmShaderPatcherCreate",
        g::SHADER_PATCHER_DESTROY => "sceGxmShaderPatcherDestroy",
        g::PROGRAM_CHECK => "sceGxmProgramCheck",
        g::SHADER_PATCHER_REGISTER_PROGRAM => "sceGxmShaderPatcherRegisterProgram",
        g::SHADER_PATCHER_UNREGISTER_PROGRAM => "sceGxmShaderPatcherUnregisterProgram",
        g::SHADER_PATCHER_CREATE_VERTEX_PROGRAM => "sceGxmShaderPatcherCreateVertexProgram",
        g::SHADER_PATCHER_CREATE_FRAGMENT_PROGRAM => "sceGxmShaderPatcherCreateFragmentProgram",
        g::SHADER_PATCHER_RELEASE_VERTEX_PROGRAM => "sceGxmShaderPatcherReleaseVertexProgram",
        g::SHADER_PATCHER_RELEASE_FRAGMENT_PROGRAM => "sceGxmShaderPatcherReleaseFragmentProgram",
        g::PROGRAM_FIND_PARAMETER_BY_NAME => "sceGxmProgramFindParameterByName",
        g::BEGIN_SCENE => "sceGxmBeginScene",
        g::END_SCENE => "sceGxmEndScene",
        g::SET_VERTEX_PROGRAM => "sceGxmSetVertexProgram",
        g::SET_FRAGMENT_PROGRAM => "sceGxmSetFragmentProgram",
        g::RESERVE_VERTEX_DEFAULT_UNIFORM_BUFFER => "sceGxmReserveVertexDefaultUniformBuffer",
        g::SET_UNIFORM_DATA_F => "sceGxmSetUniformDataF",
        g::SET_VERTEX_STREAM => "sceGxmSetVertexStream",
        g::DRAW => "sceGxmDraw",
        g::PAD_HEARTBEAT => "sceGxmPadHeartbeat",
        g::DISPLAY_QUEUE_ADD_ENTRY => "sceGxmDisplayQueueAddEntry",
        g::DISPLAY_QUEUE_FINISH => "sceGxmDisplayQueueFinish",
        g::FINISH => "sceGxmFinish",
        d::SET_FRAME_BUF => "sceDisplaySetFrameBuf",
        c::PEEK_BUFFER_POSITIVE => "sceCtrlPeekBufferPositive",
        s::ALLOC_MEM_BLOCK => "sceKernelAllocMemBlock",
        s::GET_MEM_BLOCK_BASE => "sceKernelGetMemBlockBase",
        lk::CLIB_PRINTF => "sceClibPrintf",
        lk::CLIB_VPRINTF => "sceClibVprintf",
        lk::CLIB_SNPRINTF => "sceClibSnprintf",
        lk::CLIB_VSNPRINTF => "sceClibVsnprintf",
        lk::CLIB_MEMCPY => "sceClibMemcpy",
        lk::CLIB_MEMMOVE => "sceClibMemmove",
        lk::CLIB_MEMSET => "sceClibMemset",
        lk::CLIB_MEMCMP => "sceClibMemcmp",
        lk::CLIB_STRNLEN => "sceClibStrnlen",
        lk::CLIB_STRNCPY => "sceClibStrncpy",
        lk::CLIB_STRNCMP => "sceClibStrncmp",
        lk::CLIB_STRCMP => "sceClibStrcmp",
        lk::EXIT_PROCESS => "sceKernelExitProcess",
        lk::CREATE_THREAD => "sceKernelCreateThread",
        lk::START_THREAD => "sceKernelStartThread",
        lk::WAIT_THREAD_END => "sceKernelWaitThreadEnd",
        lk::GET_THREAD_ID => "sceKernelGetThreadId",
        io::IO_OPEN => "sceIoOpen",
        io::IO_CLOSE => "sceIoClose",
        io::IO_READ => "sceIoRead",
        io::IO_WRITE => "sceIoWrite",
        io::IO_LSEEK => "sceIoLseek",
        io::IO_LSEEK32 => "sceIoLseek32",
        io::IO_GETSTAT => "sceIoGetstat",
        io::IO_MKDIR => "sceIoMkdir",
        io::IO_REMOVE => "sceIoRemove",
        tm::DELAY_THREAD => "sceKernelDelayThread",
        tm::EXIT_DELETE_THREAD => "sceKernelExitDeleteThread",
        tm::EXIT_THREAD => "sceKernelExitThread",
        sy::CREATE_MUTEX => "sceKernelCreateMutex",
        sy::LOCK_MUTEX => "sceKernelLockMutex",
        sy::TRY_LOCK_MUTEX => "sceKernelTryLockMutex",
        sy::UNLOCK_MUTEX => "sceKernelUnlockMutex",
        sy::DELETE_MUTEX => "sceKernelDeleteMutex",
        sy::CREATE_SEMA => "sceKernelCreateSema",
        sy::WAIT_SEMA => "sceKernelWaitSema",
        sy::SIGNAL_SEMA => "sceKernelSignalSema",
        sy::DELETE_SEMA => "sceKernelDeleteSema",
        sy::CREATE_EVENT_FLAG => "sceKernelCreateEventFlag",
        sy::SET_EVENT_FLAG => "sceKernelSetEventFlag",
        sy::WAIT_EVENT_FLAG => "sceKernelWaitEventFlag",
        sy::CLEAR_EVENT_FLAG => "sceKernelClearEventFlag",
        sy::DELETE_EVENT_FLAG => "sceKernelDeleteEventFlag",
        sy::GET_SYSTEM_TIME_WIDE => "sceKernelGetSystemTimeWide",
        sy::CREATE_COND => "sceKernelCreateCond",
        sy::WAIT_COND => "sceKernelWaitCond",
        sy::SIGNAL_COND => "sceKernelSignalCond",
        sy::SIGNAL_COND_ALL => "sceKernelSignalCondAll",
        sy::DELETE_COND => "sceKernelDeleteCond",
        _ => "<unknown>",
    }
}
