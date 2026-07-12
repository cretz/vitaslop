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

/// A human-readable name for a `(library_nid, func_nid)` pair, for logging and
/// the unimplemented-call report. Falls back to the raw NIDs.
pub fn name(func_nid: u32) -> &'static str {
    use {ctrl as c, display as d, gxm as g, sysmem as s};
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
        _ => "<unknown>",
    }
}
