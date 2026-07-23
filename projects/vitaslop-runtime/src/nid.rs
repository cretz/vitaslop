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
    /// SceNgsUser: the NGS software synthesizer (voices, racks, patches).
    pub const SCE_NGS: u32 = 0xB015_98D9;
    /// SceAudio: the low-level PCM audio-output ports (sceAudioOut*).
    pub const SCE_AUDIO: u32 = 0x438B_B957;
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
    pub const SHADER_PATCHER_GET_PROGRAM_FROM_ID: u32 = 0xA949_A803;
    pub const PROGRAM_PARAMETER_GET_RESOURCE_INDEX: u32 = 0x5C79_D59A;
    pub const PROGRAM_GET_PARAMETER_COUNT: u32 = 0xD5D5_FCCD;
    pub const PROGRAM_GET_PARAMETER: u32 = 0x06FF_9151;
    pub const PROGRAM_PARAMETER_GET_CATEGORY: u32 = 0x1997_DC17;
    pub const PROGRAM_PARAMETER_GET_TYPE: u32 = 0x7B90_23C3;
    pub const PROGRAM_PARAMETER_GET_COMPONENT_COUNT: u32 = 0xBD29_98D1;
    pub const PROGRAM_PARAMETER_GET_CONTAINER_INDEX: u32 = 0xBB58_267D;
    pub const PROGRAM_PARAMETER_GET_ARRAY_SIZE: u32 = 0xDBA8_D061;
    pub const PROGRAM_PARAMETER_GET_NAME: u32 = 0x6AF8_8A5D;
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
    // Fragment textures.
    pub const SET_FRAGMENT_TEXTURE: u32 = 0x29C3_4DF5;
    pub const TEXTURE_INIT_LINEAR: u32 = 0x4811_AECB;
    pub const TEXTURE_INIT_LINEAR_STRIDED: u32 = 0x6679_BEF0;
    pub const TEXTURE_INIT_SWIZZLED: u32 = 0xD572_D547;
    pub const TEXTURE_INIT_SWIZZLED_ARBITRARY: u32 = 0x5DBF_BA2C;
    pub const TEXTURE_INIT_TILED: u32 = 0xE6F0_DB27;
    pub const TEXTURE_SET_DATA: u32 = 0x8558_14C4;
    pub const TEXTURE_SET_FORMAT: u32 = 0xFC94_3596;
    pub const TEXTURE_SET_MAG_FILTER: u32 = 0xFA69_5FD7;
    pub const TEXTURE_SET_MIN_FILTER: u32 = 0x4167_64E3;
    pub const TEXTURE_SET_MIP_FILTER: u32 = 0x1CA9_FE0B;
    pub const TEXTURE_SET_U_ADDR_MODE: u32 = 0x4281_763E;
    pub const TEXTURE_SET_V_ADDR_MODE: u32 = 0x126C_DAA3;
    pub const TEXTURE_GET_DATA: u32 = 0x5341_BD46;
    pub const TEXTURE_GET_WIDTH: u32 = 0x126A_3EB3;
    pub const TEXTURE_GET_HEIGHT: u32 = 0x5420_A086;
    pub const TEXTURE_GET_FORMAT: u32 = 0xE868_D2B3;
    pub const SET_FRAGMENT_UNIFORM_BUFFER: u32 = 0xEA0F_C310;
    pub const RESERVE_FRAGMENT_DEFAULT_UNIFORM_BUFFER: u32 = 0x7B1F_ABB6;
    // Fixed-function pipeline state setters (see `vita::gxm` render-state handlers).
    pub const SET_CULL_MODE: u32 = 0xE1CA_72AE;
    pub const SET_TWO_SIDED_ENABLE: u32 = 0x0DE9_AEB7;
    pub const SET_FRONT_DEPTH_FUNC: u32 = 0x14BD_831F;
    pub const SET_BACK_DEPTH_FUNC: u32 = 0xB042_A4D2;
    pub const SET_FRONT_DEPTH_WRITE_ENABLE: u32 = 0xF32C_BF34;
    pub const SET_FRONT_FRAGMENT_PROGRAM_ENABLE: u32 = 0x5759_58A8;
    pub const SET_BACK_FRAGMENT_PROGRAM_ENABLE: u32 = 0xE26B_4834;
    pub const SET_FRONT_POINT_LINE_WIDTH: u32 = 0x0675_2183;
    pub const SET_FRONT_POLYGON_MODE: u32 = 0xFD93_209D;
    pub const SET_FRONT_STENCIL_REF: u32 = 0x8FA6_FE44;
    pub const SET_FRONT_STENCIL_FUNC: u32 = 0xB864_5A9A;
    pub const SET_VIEWPORT: u32 = 0x3EB3_380B;
    pub const SET_VIEWPORT_ENABLE: u32 = 0x814F_61EB;
    pub const SET_REGION_CLIP: u32 = 0x70C8_6868;
    // Surface / texture / parameter getters and sampler-state setters.
    pub const COLOR_SURFACE_GET_FORMAT: u32 = 0xF3C1_C6C6;
    pub const COLOR_SURFACE_GET_TYPE: u32 = 0x52FD_E962;
    pub const COLOR_SURFACE_SET_CLIP: u32 = 0x8645_6F7B;
    pub const TEXTURE_GET_TYPE: u32 = 0xF65D_4917;
    /// `_sceGxmProgramParameterGetSemantic` (the exported user variant).
    pub const PROGRAM_PARAMETER_GET_SEMANTIC: u32 = 0xAAFD_61D5;
    pub const PROGRAM_PARAMETER_GET_SEMANTIC_INDEX: u32 = 0xB85C_C13E;
    pub const TEXTURE_INIT_CUBE: u32 = 0x11DC_8DC9;
    pub const TEXTURE_SET_U_ADDR_MODE_SAFE: u32 = 0x8699_ECF4;
    pub const TEXTURE_SET_V_ADDR_MODE_SAFE: u32 = 0xFA22_F6CC;
    pub const TEXTURE_SET_LOD_BIAS: u32 = 0xB65E_E6F7;
    // Color-surface getters/setters beyond format.
    pub const COLOR_SURFACE_GET_DATA: u32 = 0x2DB6_026C;
    pub const COLOR_SURFACE_GET_STRIDE_IN_PIXELS: u32 = 0xF33D_9980;
    pub const COLOR_SURFACE_SET_GAMMA_MODE: u32 = 0xF5C8_9643;
    // Render-target sizing + GPU notification region.
    pub const GET_RENDER_TARGET_MEM_SIZE: u32 = 0xB291_C959;
    pub const GET_NOTIFICATION_REGION: u32 = 0x8BDE_825A;
    // Program reflection: default uniform-buffer size + fragment-program pass type.
    pub const PROGRAM_GET_DEFAULT_UNIFORM_BUFFER_SIZE: u32 = 0x8FA3_F9C3;
    pub const FRAGMENT_PROGRAM_GET_PASS_TYPE: u32 = 0xCE0B_0A76;
    // Texture getters (read back the sticky sampler/format state).
    pub const TEXTURE_GET_MIPMAP_COUNT_UNSAFE: u32 = 0x4CC4_2929;
    pub const TEXTURE_GET_STRIDE: u32 = 0xB0BD_52F3;
    pub const TEXTURE_GET_LOD_BIAS: u32 = 0x2DE5_5DA5;
    pub const TEXTURE_GET_U_ADDR_MODE_SAFE: u32 = 0xC037_DA83;
    pub const TEXTURE_GET_V_ADDR_MODE_SAFE: u32 = 0xD2F0_D9C1;
    pub const TEXTURE_GET_MAG_FILTER: u32 = 0xAE7F_BB51;
    pub const TEXTURE_GET_MIN_FILTER: u32 = 0x9206_66C6;
    pub const TEXTURE_GET_GAMMA_MODE: u32 = 0xF23F_CE81;
    pub const TEXTURE_SET_GAMMA_MODE: u32 = 0xA6D9_F4DA;
    // Precomputed draw family: bundle a vertex program + streams + draw params into a
    // guest-owned block, then replay it via sceGxmDrawPrecomputed.
    pub const GET_PRECOMPUTED_DRAW_SIZE: u32 = 0x41BB_D792;
    pub const PRECOMPUTED_DRAW_INIT: u32 = 0xA197_F096;
    pub const PRECOMPUTED_DRAW_SET_PARAMS: u32 = 0x884D_0D08;
    pub const PRECOMPUTED_DRAW_SET_VERTEX_STREAM: u32 = 0x6C93_6214;
    pub const DRAW_PRECOMPUTED: u32 = 0xED3F_78B8;
    // Precomputed vertex/fragment state: build (size/init/setters) + bind.
    pub const GET_PRECOMPUTED_VERTEX_STATE_SIZE: u32 = 0x9D83_CA3B;
    pub const GET_PRECOMPUTED_FRAGMENT_STATE_SIZE: u32 = 0x85DE_8506;
    pub const PRECOMPUTED_VERTEX_STATE_INIT: u32 = 0xBE93_7F8D;
    pub const PRECOMPUTED_FRAGMENT_STATE_INIT: u32 = 0xE297_D7AF;
    pub const PRECOMPUTED_VERTEX_STATE_SET_DEFAULT_UNIFORM_BUFFER: u32 = 0x34BF_64E3;
    pub const PRECOMPUTED_FRAGMENT_STATE_SET_DEFAULT_UNIFORM_BUFFER: u32 = 0x9123_6858;
    pub const PRECOMPUTED_VERTEX_STATE_GET_DEFAULT_UNIFORM_BUFFER: u32 = 0xBE5A_68EF;
    pub const PRECOMPUTED_FRAGMENT_STATE_GET_DEFAULT_UNIFORM_BUFFER: u32 = 0xCECB_584A;
    pub const PRECOMPUTED_VERTEX_STATE_SET_TEXTURE: u32 = 0x6A29_EB06;
    pub const PRECOMPUTED_FRAGMENT_STATE_SET_TEXTURE: u32 = 0x2911_8BF1;
    pub const SET_PRECOMPUTED_VERTEX_STATE: u32 = 0xB762_6A93;
    pub const SET_PRECOMPUTED_FRAGMENT_STATE: u32 = 0xF895_2750;
}

/// SceDisplayUser / SceDisplay function NIDs. `SET_FRAME_BUF` is SceDisplayUser
/// (lib 0x4FAACD11); `WAIT_VBLANK_START_MULTI` is SceDisplay (lib 0x5ED8F994).
/// Dispatch is by func NID, so the two libraries are grouped by concept here.
pub mod display {
    pub const SET_FRAME_BUF: u32 = 0x7A41_0B64;
    pub const WAIT_VBLANK_START_MULTI: u32 = 0xDD0A_13B8;
    /// `sceDisplayWaitSetFrameBuf` (SceDisplay, lib 0x5ED8F994): block until the
    /// frame buffer queued by `sceDisplaySetFrameBuf` has been latched at vblank.
    pub const WAIT_SET_FRAME_BUF: u32 = 0x9423_560C;
}

/// SceCtrl function NIDs.
pub mod ctrl {
    pub const PEEK_BUFFER_POSITIVE: u32 = 0xA9C3_CED6;
    pub const READ_BUFFER_POSITIVE: u32 = 0x67E7_AB83;
    pub const PEEK_BUFFER_NEGATIVE: u32 = 0x104E_D1A7;
    pub const READ_BUFFER_NEGATIVE: u32 = 0x15F9_6FB0;
    pub const SET_SAMPLING_MODE: u32 = 0xA497_B150;
}

/// SceSysmem (kernel memory) function NIDs.
pub mod sysmem {
    pub const ALLOC_MEM_BLOCK: u32 = 0xB9D5_EBDE;
    pub const GET_MEM_BLOCK_BASE: u32 = 0xB8EF_5818;
    pub const FREE_MEM_BLOCK: u32 = 0xA91E_15EE;
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
    // Thread-local storage: a per-thread pointer slot indexed by key.
    pub const GET_TLS_ADDR: u32 = 0xB295_EB61;
    // Process/thread timing and status.
    pub const GET_PROCESS_TIME: u32 = 0x4C46_72BF;
    pub const GET_PROCESS_TIME_WIDE: u32 = 0xB110_C123;
    pub const GET_THREAD_EXIT_STATUS: u32 = 0xD5DC_26C4;
    // process control.
    pub const EXIT_PROCESS: u32 = 0x7595_D9AA;
    // thread wrappers (user-facing; the ThreadMgr primitives back them).
    pub const CREATE_THREAD: u32 = 0xC5C1_1EE7;
    pub const START_THREAD: u32 = 0xF08D_E149;
    pub const WAIT_THREAD_END: u32 = 0xDDB3_95A9;
    pub const GET_THREAD_ID: u32 = 0x0FB9_72F9;
    /// An unnamed SceLibKernel export the title imports (present in no vita-headers
    /// revision, any firmware). Serviced as a clib no-op success (return 0) so the
    /// call is handled rather than left as an unimplemented gap.
    pub const UNKNOWN_023EAA62: u32 = 0x023E_AA62;
}

/// SceProcessmgr function NIDs: process parameters and the standard IO handles the
/// libc crt fetches during startup.
pub mod processmgr {
    pub const GET_PROCESS_PARAM: u32 = 0x2BE3_E066;
    pub const GET_STDIN: u32 = 0xC172_7F59;
    pub const GET_STDOUT: u32 = 0xE5AA_625C;
    pub const GET_STDERR: u32 = 0xFA5E_3ADA;
    pub const LIBC_TIME: u32 = 0x0039_BE45;
    pub const LIBC_CLOCK: u32 = 0x9E45_DA09;
    /// sceKernelPowerTick: reset the idle/power-save countdown. No power management
    /// off-console, so it is an accepted no-op.
    pub const POWER_TICK: u32 = 0x2252_890C;
}

/// System and online-service NIDs the boot path touches (SceSysmodule, SceNet,
/// SceNetCtl, SceHttp, SceSsl, SceNpManager, SceNpBasic, SceRtc, SceFios2, and the
/// libult object manager). Off-console these have no backing service, so they are
/// serviced as "initialized but offline": init succeeds, connection state reads as
/// disconnected, and callback registration succeeds without ever firing.
pub mod services {
    // SceSysmodule.
    pub const SYSMODULE_IS_LOADED: u32 = 0x5309_9B7A;
    // SceNet / SceNetCtl.
    pub const NET_INIT: u32 = 0xEB03_E265;
    pub const NET_CTL_INIT: u32 = 0x495C_A1DB;
    pub const NET_CTL_INET_GET_STATE: u32 = 0x6D26_AC68;
    pub const NET_CTL_INET_REGISTER_CALLBACK: u32 = 0xEAEE_6185;
    pub const NET_CTL_CHECK_CALLBACK: u32 = 0xDFFC_3ED4;
    // SceHttp / SceSsl.
    pub const HTTP_INIT: u32 = 0x2149_26D9;
    pub const SSL_INIT: u32 = 0x3C73_3316;
    // SceNpManager / SceNpBasic.
    pub const NP_INIT: u32 = 0x04D9_F484;
    pub const NP_REGISTER_SERVICE_STATE_CALLBACK: u32 = 0x4423_9C35;
    pub const NP_CHECK_CALLBACK: u32 = 0x3B0A_E9A9;
    pub const NP_BASIC_INIT: u32 = 0xEFB9_1A99;
    pub const NP_BASIC_REGISTER_HANDLER: u32 = 0x26E6_E048;
    pub const NP_BASIC_CHECK_CALLBACK: u32 = 0x2014_6AEC;
    // SceRtc.
    pub const RTC_GET_CURRENT_CLOCK: u32 = 0x70FD_E8F1;
    pub const RTC_GET_CURRENT_CLOCK_LOCAL_TIME: u32 = 0x0572_EDDC;
    pub const RTC_GET_CURRENT_TICK: u32 = 0x23F7_9274;
    pub const RTC_GET_TICK: u32 = 0xF2B2_38E2;
    // SceMotion.
    pub const MOTION_GET_STATE: u32 = 0xBDB3_2767;
    // SceFios2 overlay + libult object manager.
    pub const FIOS_OVERLAY_GET_LIST: u32 = 0x1DD8_08D1;
    pub const ULOBJ_REGISTER_PROTOCOL_REVISION: u32 = 0x50F2_F2AA;
    // SceAppUtil: app utility init + system parameters (language, button assign, ...).
    pub const APPUTIL_INIT: u32 = 0xDAFF_E671;
    pub const APPUTIL_SYSTEM_PARAM_GET_INT: u32 = 0x5DFB_9CA0;
    pub const APPUTIL_APP_PARAM_GET_INT: u32 = 0xCD7F_D67A;
    pub const LIVE_AREA_GET_STATUS: u32 = 0x7FE5_B83F;
    pub const LIVE_AREA_UPDATE_FRAME_ASYNC: u32 = 0xD330_285D;
    pub const APPUTIL_DRM_OPEN: u32 = 0x2DB7_BE3B;
    pub const APPUTIL_DRM_CLOSE: u32 = 0x6A14_0498;
    pub const APPUTIL_SAVEDATA_SLOT_GET_PARAM: u32 = 0x93F0_D89F;
    pub const APPUTIL_SAVEDATA_SLOT_CREATE: u32 = 0x7E8F_E96A;
    // SceAppMgr: app-lifecycle state poll (system/app event counts, overlay flag).
    pub const APP_MGR_GET_APP_STATE: u32 = 0x5E86_319A;
    // SceNpScore / SceNpManager: online leaderboards and account identity.
    pub const NP_SCORE_INIT: u32 = 0x0433_069F;
    pub const NP_SCORE_CREATE_TITLE_CTX: u32 = 0x5685_F225;
    pub const NP_MANAGER_GET_NP_ID: u32 = 0x3C94_B4B4;
    pub const NP_MANAGER_GET_ACCOUNT_REGION: u32 = 0xFE83_5967;
    pub const NP_MANAGER_GET_CONTENT_RATING_FLAG: u32 = 0xAF00_73B2;
    pub const NP_MANAGER_GET_CHAT_RESTRICTION_FLAG: u32 = 0x60C5_75B1;
    // SceCommonDialog: the per-frame pump plus each dialog family's
    // Init/GetStatus/GetResult/Term lifecycle (see `services::dialog_*`).
    pub const COMMON_DIALOG_UPDATE: u32 = 0x9053_0F2F;
    pub const MSG_DIALOG_INIT: u32 = 0x755F_F270;
    pub const MSG_DIALOG_GET_STATUS: u32 = 0x4107_019E;
    pub const MSG_DIALOG_GET_RESULT: u32 = 0xBB3B_FC89;
    pub const MSG_DIALOG_TERM: u32 = 0x81AC_F695;
    pub const NET_CHECK_DIALOG_INIT: u32 = 0xA38A_4A0D;
    pub const NET_CHECK_DIALOG_GET_STATUS: u32 = 0x8027_292A;
    pub const NET_CHECK_DIALOG_GET_RESULT: u32 = 0xB05F_CE9E;
    pub const NET_CHECK_DIALOG_TERM: u32 = 0x8BE5_1C15;
    pub const SAVEDATA_DIALOG_INIT: u32 = 0xBF52_48FA;
    pub const SAVEDATA_DIALOG_GET_STATUS: u32 = 0x6E25_8046;
    pub const SAVEDATA_DIALOG_GET_SUB_STATUS: u32 = 0xBA05_42CA;
    pub const SAVEDATA_DIALOG_GET_RESULT: u32 = 0xB2FF_576E;
    pub const SAVEDATA_DIALOG_CONTINUE: u32 = 0x1919_2C8B;
    pub const SAVEDATA_DIALOG_FINISH: u32 = 0x6C49_924B;
    pub const SAVEDATA_DIALOG_SUB_CLOSE: u32 = 0x415D_6068;
    pub const SAVEDATA_DIALOG_TERM: u32 = 0x2192_A10A;
    pub const NP_MESSAGE_DIALOG_INIT: u32 = 0x4535_A358;
    pub const NP_MESSAGE_DIALOG_GET_STATUS: u32 = 0x2A0D_060F;
    pub const NP_MESSAGE_DIALOG_GET_RESULT: u32 = 0x7EC9_5C61;
    pub const NP_MESSAGE_DIALOG_ABORT: u32 = 0x47AB_6D04;
    pub const NP_MESSAGE_DIALOG_TERM: u32 = 0x7AB5_0F63;
    pub const NP_TROPHY_SETUP_DIALOG_INIT: u32 = 0x9E2C_02C9;
    pub const NP_TROPHY_SETUP_DIALOG_GET_STATUS: u32 = 0xC3A5_9547;
    pub const NP_TROPHY_SETUP_DIALOG_TERM: u32 = 0xA810_82DD;
    pub const STORE_CHECKOUT_DIALOG_INIT: u32 = 0x52EC_D8A5;
    pub const STORE_CHECKOUT_DIALOG_GET_STATUS: u32 = 0x7004_BB2E;
    pub const STORE_CHECKOUT_DIALOG_GET_RESULT: u32 = 0x07ED_1E26;
    pub const STORE_CHECKOUT_DIALOG_TERM: u32 = 0xB787_F4B0;
    pub const NP_SNS_FACEBOOK_DIALOG_INIT: u32 = 0x6821_F09B;
    pub const NP_SNS_FACEBOOK_DIALOG_GET_STATUS: u32 = 0x1476_50E8;
    pub const NP_SNS_FACEBOOK_DIALOG_GET_RESULT_LONG_TOKEN: u32 = 0xA868_2304;
    // SceTouch.
    pub const TOUCH_SET_SAMPLING_STATE: u32 = 0x1B9C_5D14;
    pub const TOUCH_READ: u32 = 0x169A_1D58;
    pub const TOUCH_PEEK: u32 = 0xFF08_2DF0;
    pub const TOUCH_GET_PANEL_INFO: u32 = 0x10A2_CA25;
    pub const TOUCH_ENABLE_TOUCH_FORCE: u32 = 0xB183_70C2;
    // SceSysmodule load: off-console the requested module is already linked in, so a
    // load request just reports success.
    pub const SYSMODULE_LOAD_MODULE: u32 = 0x79A0_160A;
    // SceAppUtil: fetch a system-parameter string (e.g. the account username).
    pub const APPUTIL_SYSTEM_PARAM_GET_STRING: u32 = 0x6E6A_A267;
    // SceScreenShot: the OS screenshot feature (enable/disable + overlay/param). Off
    // console there is nothing to capture, so each call succeeds with no effect.
    pub const SCREENSHOT_DISABLE: u32 = 0x50AE_9FF9;
    pub const SCREENSHOT_ENABLE: u32 = 0x76E6_74D1;
    pub const SCREENSHOT_SET_PARAM: u32 = 0x05DB_59C7;
    pub const SCREENSHOT_SET_OVERLAY_IMAGE: u32 = 0x7061_665B;
    // SceNpTrophy: trophies work offline (persisted locally, unlocked via the egress
    // ledger). Init + context/handle creation succeed with real non-zero out-handles.
    pub const NP_TROPHY_INIT: u32 = 0x3451_6838;
    pub const NP_TROPHY_CREATE_CONTEXT: u32 = 0xC49F_D33F;
    pub const NP_TROPHY_CREATE_HANDLE: u32 = 0x4EBC_6977;
    pub const NP_TROPHY_GET_GAME_INFO: u32 = 0xBA2B_7F2A;
    pub const NP_TROPHY_GET_TROPHY_UNLOCK_STATE: u32 = 0xC8D2_A4DE;
    // SceNp* subsystem inits with no backing service off-console: succeed so the
    // title proceeds (SceNpActivity/SceNpCommon-auth/SceNpUtility-lookup/SceNpTus).
    pub const NP_ACTIVITY_INIT: u32 = 0xE0FF_EE97;
    pub const NP_AUTH_INIT: u32 = 0x441D_8B4E;
    pub const NP_LOOKUP_INIT: u32 = 0x9246_A673;
    pub const NP_TUS_INIT: u32 = 0xB214_1F8D;
    // SceNpSnsFacebook: social-network integration; the library init succeeds offline
    // (no online SNS features are then available, and the title stays on its offline path).
    pub const NP_SNS_FACEBOOK_INIT: u32 = 0x8055_7AA0;
    // SceLibLocation / SceMotion / SceNetCtl(adhoc) / ScePower: device-service inits
    // and config that succeed with a neutral/offline result.
    pub const LOCATION_INIT: u32 = 0x09C4_F674;
    pub const MOTION_START_SAMPLING: u32 = 0x2803_4AC9;
    pub const NETCTL_ADHOC_REGISTER_CALLBACK: u32 = 0xFFA9_D594;
    pub const POWER_SET_CONFIGURATION_MODE: u32 = 0x3CE1_87B6;
    // SceCommonDialog: shared config for the dialog families, plus the trophy-setup
    // dialog's result read.
    pub const COMMON_DIALOG_SET_CONFIG_PARAM: u32 = 0xBECD_35C8;
    pub const NP_TROPHY_SETUP_DIALOG_GET_RESULT: u32 = 0xE370_69D5;
    /// An unnamed SceNearUtil export the title imports ("near" is the offline-social
    /// app; present in no vita-headers revision). Serviced as an offline success.
    pub const NEAR_UTIL_UNKNOWN_A412E9CA: u32 = 0xA412_E9CA;
}

/// Lightweight synchronization (SceLibKernel LwMutex/LwCond): mutexes and condition
/// variables whose state lives in a caller-provided work area rather than a kernel
/// object. Bring-up model matches the heavyweight primitives - uncontended success
/// in the single-thread-of-control mode (see [`crate::vita::sync`]).
pub mod lwsync {
    pub const CREATE_LW_MUTEX: u32 = 0xDA6E_C8EF;
    pub const DELETE_LW_MUTEX: u32 = 0x244E_76D2;
    pub const LOCK_LW_MUTEX: u32 = 0x46E7_BE7B;
    pub const LOCK_LW_MUTEX_CB: u32 = 0x3148_C6B6;
    pub const TRY_LOCK_LW_MUTEX: u32 = 0xA6A2_C915;
    pub const UNLOCK_LW_MUTEX: u32 = 0x91FA_6614;
    pub const UNLOCK_LW_MUTEX2: u32 = 0x120A_FC8C;
    pub const CREATE_LW_COND: u32 = 0x48C7_EAE6;
    pub const DELETE_LW_COND: u32 = 0x721F_6CB3;
    pub const WAIT_LW_COND: u32 = 0xE187_8282;
    pub const WAIT_LW_COND_CB: u32 = 0x8FA5_4B07;
    pub const SIGNAL_LW_COND: u32 = 0x3AC6_3B9A;
    pub const SIGNAL_LW_COND_ALL: u32 = 0xE524_1A0C;
    pub const SIGNAL_LW_COND_TO: u32 = 0xFC1A_48EB;
}

/// SceThreadmgr function NIDs: thread-manager primitives not wrapped in
/// SceLibKernel.
pub mod threadmgr {
    pub const DELAY_THREAD: u32 = 0x4B67_5D05;
    pub const EXIT_DELETE_THREAD: u32 = 0x1D17_DECF;
    pub const EXIT_THREAD: u32 = 0x0C8A_38E1;
    pub const GET_PROCESS_ID: u32 = 0x9DCB_4B7A;
    pub const GET_THREAD_CURRENT_PRIORITY: u32 = 0x0141_4F0B;
    /// `sceKernelCloseSema`: releases a semaphore id (same effect as DeleteSema in
    /// this model - the id becomes invalid). Routed to the shared delete handler.
    pub const CLOSE_SEMA: u32 = 0xA2D8_1F9E;
    /// `sceKernelChangeThreadVfpException`: sets which VFP/NEON floating-point
    /// exceptions trap for the calling thread. We compute IEEE arithmetic without
    /// trapping, so this only records intent; it never changes numeric results.
    pub const CHANGE_THREAD_VFP_EXCEPTION: u32 = 0xCC18_FBAE;
}

/// ScePvf: the Vita font library. A title creates a lib, configures em/resolution/
/// skew, and opens fonts. Handles are opaque; the surface is satisfied without a
/// glyph rasterizer (text is drawn through the captured GXM stream).
pub mod pvf {
    pub const NEW_LIB: u32 = 0x72E5_8672;
    pub const DONE_LIB: u32 = 0xE177_17EC;
    pub const SET_EM: u32 = 0xDFB6_77C5;
    pub const OPEN: u32 = 0xE354_34BB;
    pub const OPEN_USER_FILE: u32 = 0xD535_520F;
    pub const SET_RESOLUTION: u32 = 0xC444_4FB3;
    pub const SET_CHAR_SIZE: u32 = 0xF17A_DE4D;
    pub const SET_SKEW_VALUE: u32 = 0x3DD0_9BC9;
    pub const GET_FONT_INFO: u32 = 0xAB0C_7CF2;
    pub const GET_CHAR_INFO: u32 = 0xA88E_EDB0;
    pub const GET_CHAR_IMAGE_RECT: u32 = 0x6C1B_9CAF;
    pub const IS_ELEMENT: u32 = 0x9F01_8F25;
    pub const GET_CHAR_GLYPH_IMAGE: u32 = 0x37DA_496A;
    pub const PIXEL_TO_POINT_H: u32 = 0xF56B_5B9B;
    pub const PIXEL_TO_POINT_V: u32 = 0xCDA2_82D2;
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
    pub const IO_PREAD: u32 = 0x5231_5AD7;
    pub const IO_PWRITE: u32 = 0x8FFF_F5A8;
    pub const IO_GETSTAT: u32 = 0xBCA5_B623;
    pub const IO_GETSTAT_BY_FD: u32 = 0x57F8_CD25;
    pub const IO_MKDIR: u32 = 0x9670_D39F;
    pub const IO_REMOVE: u32 = 0xE20E_D0F3;
    pub const IO_DOPEN: u32 = 0xA928_3DD0;
    pub const IO_DREAD: u32 = 0x9C8B_6624;
    pub const IO_DCLOSE: u32 = 0x422A_221A;
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
    /// `sceKernelCreateSema_16XX`: the pre-3.60 firmware NID for the same call the
    /// SDK later re-exported as `CREATE_SEMA`. Titles built against an older SDK
    /// import this one; it dispatches to the identical handler.
    pub const CREATE_SEMA_16XX: u32 = 0x297A_A2AE;
    pub const WAIT_SEMA: u32 = 0x0C7B_834B;
    pub const SIGNAL_SEMA: u32 = 0xE6B7_61D1;
    pub const DELETE_SEMA: u32 = 0xDB32_948A;
    pub const CREATE_EVENT_FLAG: u32 = 0x8516_D040;
    pub const SET_EVENT_FLAG: u32 = 0xEC94_DFF7;
    pub const WAIT_EVENT_FLAG: u32 = 0x83C0_E2AF;
    /// `sceKernelWaitEventFlagCB`: identical to `WAIT_EVENT_FLAG` but also pumps the
    /// calling thread's async callbacks while blocked. Same 5-arg signature; we have
    /// no user-callback delivery to pump, so it shares the plain wait handler - the
    /// crucial behaviour is that it actually BLOCKS until the bits are set (a stub
    /// that returns success immediately lets a thread race past the worker it waits on).
    pub const WAIT_EVENT_FLAG_CB: u32 = 0xE737_B1DF;
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

/// SceNgsUser: the NGS software synthesizer. A title creates a system, then racks
/// of voices, plays AT9/PCM sources through them, and pumps the mix each frame.
pub mod ngs {
    pub const SYSTEM_GET_REQUIRED_MEMORY_SIZE: u32 = 0x6CE8_B36F;
    pub const SYSTEM_INIT: u32 = 0xED14_CF4A;
    pub const SYSTEM_UPDATE: u32 = 0x684F_080C;
    pub const SYSTEM_RELEASE: u32 = 0x4A25_BEBC;
    pub const SYSTEM_SET_FLAGS: u32 = 0x64D8_0013;
    pub const RACK_GET_REQUIRED_MEMORY_SIZE: u32 = 0x4773_18C0;
    pub const RACK_INIT: u32 = 0x0A92_E4EC;
    pub const RACK_GET_VOICE_HANDLE: u32 = 0xFE1A_98E9;
    pub const VOICE_GET_STATE_DATA: u32 = 0xC9B8_C0B4;
    pub const VOICE_LOCK_PARAMS: u32 = 0xAB6B_EF8F;
    pub const VOICE_UNLOCK_PARAMS: u32 = 0x3D46_D8A7;
    pub const VOICE_PLAY: u32 = 0xFA0A_0F34;
    pub const VOICE_KEY_OFF: u32 = 0xBB13_373D;
    pub const VOICE_KILL: u32 = 0x0E29_1AAD;
    pub const VOICE_PAUSE: u32 = 0xD778_6E99;
    pub const VOICE_RESUME: u32 = 0x54CF_B981;
    pub const VOICE_SET_FINISHED_CALLBACK: u32 = 0x17A6_F564;
    pub const VOICE_SET_MODULE_CALLBACK: u32 = 0x24E9_09A8;
    pub const VOICE_BYPASS_MODULE: u32 = 0x9AB8_7E71;
    pub const VOICE_GET_PARAMS_OUT_OF_RANGE: u32 = 0x4CBE_08F3;
    pub const VOICE_PATCH_SET_VOLUMES_MATRIX: u32 = 0xA0F5_402D;
    pub const VOICE_DEF_GET_SIMPLE_ATRAC9: u32 = 0x45CF_2A73;
    pub const VOICE_DEF_GET_MASTER_BUSS: u32 = 0x79A1_21D1;
    pub const VOICE_DEF_GET_REVERB_BUSS: u32 = 0x9DCF_50F5;
    pub const VOICE_DEF_GET_EQ_BUSS: u32 = 0xF964_120E;
    pub const PATCH_CREATE_ROUTING: u32 = 0xD668_B49C;
    pub const PATCH_GET_INFO: u32 = 0x9870_3DBC;
    pub const AT9_GET_SECTION_DETAILS: u32 = 0x2A9F_A501;
    // Additional voice-definition getters (return an opaque definition pointer), a
    // patch volume/routing edit, and the system lock/unlock (no-contention here).
    pub const VOICE_DEF_GET_SIMPLE_VOICE: u32 = 0x0D53_99CF;
    pub const VOICE_DEF_GET_MIXER_BUSS: u32 = 0xE0AC_8776;
    pub const VOICE_PATCH_SET_VOLUME: u32 = 0xA3C8_07BC;
    pub const PATCH_REMOVE_ROUTING: u32 = 0xD0C9_AE5A;
    pub const SYSTEM_LOCK: u32 = 0xB9D9_71F2;
    pub const SYSTEM_UNLOCK: u32 = 0x0A93_EA96;
}

/// SceAudio: low-level PCM audio-output ports.
pub mod audio {
    pub const OUT_OPEN_PORT: u32 = 0x5BC3_41E4;
    pub const OUT_OUTPUT: u32 = 0x02DB_3F5F;
    pub const OUT_SET_VOLUME: u32 = 0x6416_7F11;
    pub const OUT_RELEASE_PORT: u32 = 0x69E2_E6B5;
    pub const OUT_GET_ADOPT: u32 = 0x12FB_1767;
}

/// A human-readable name for a `(library_nid, func_nid)` pair, for logging and
/// the unimplemented-call report. Falls back to the raw NIDs.
pub fn name(func_nid: u32) -> &'static str {
    use {
        audio as au, ctrl as c, display as d, gxm as g, iofilemgr as io, libkernel as lk,
        lwsync as lw, ngs as ng, processmgr as pm, pvf as pv, services as sv, sync as sy,
        sysmem as s, threadmgr as tm,
    };
    match func_nid {
        ng::SYSTEM_GET_REQUIRED_MEMORY_SIZE => "sceNgsSystemGetRequiredMemorySize",
        ng::SYSTEM_INIT => "sceNgsSystemInit",
        ng::SYSTEM_UPDATE => "sceNgsSystemUpdate",
        ng::SYSTEM_RELEASE => "sceNgsSystemRelease",
        ng::SYSTEM_SET_FLAGS => "sceNgsSystemSetFlags",
        ng::RACK_GET_REQUIRED_MEMORY_SIZE => "sceNgsRackGetRequiredMemorySize",
        ng::RACK_INIT => "sceNgsRackInit",
        ng::RACK_GET_VOICE_HANDLE => "sceNgsRackGetVoiceHandle",
        ng::VOICE_GET_STATE_DATA => "sceNgsVoiceGetStateData",
        ng::VOICE_LOCK_PARAMS => "sceNgsVoiceLockParams",
        ng::VOICE_UNLOCK_PARAMS => "sceNgsVoiceUnlockParams",
        ng::VOICE_PLAY => "sceNgsVoicePlay",
        ng::VOICE_KEY_OFF => "sceNgsVoiceKeyOff",
        ng::VOICE_KILL => "sceNgsVoiceKill",
        ng::VOICE_PAUSE => "sceNgsVoicePause",
        ng::VOICE_RESUME => "sceNgsVoiceResume",
        ng::VOICE_SET_FINISHED_CALLBACK => "sceNgsVoiceSetFinishedCallback",
        ng::VOICE_SET_MODULE_CALLBACK => "sceNgsVoiceSetModuleCallback",
        ng::VOICE_BYPASS_MODULE => "sceNgsVoiceBypassModule",
        ng::VOICE_GET_PARAMS_OUT_OF_RANGE => "sceNgsVoiceGetParamsOutOfRange",
        ng::VOICE_PATCH_SET_VOLUMES_MATRIX => "sceNgsVoicePatchSetVolumesMatrix",
        ng::VOICE_DEF_GET_SIMPLE_ATRAC9 => "sceNgsVoiceDefGetSimpleAtrac9Voice",
        ng::VOICE_DEF_GET_MASTER_BUSS => "sceNgsVoiceDefGetMasterBuss",
        ng::VOICE_DEF_GET_REVERB_BUSS => "sceNgsVoiceDefGetReverbBuss",
        ng::VOICE_DEF_GET_EQ_BUSS => "sceNgsVoiceDefGetEqBuss",
        ng::PATCH_CREATE_ROUTING => "sceNgsPatchCreateRouting",
        ng::PATCH_GET_INFO => "sceNgsPatchGetInfo",
        ng::AT9_GET_SECTION_DETAILS => "sceNgsAT9GetSectionDetails",
        au::OUT_OPEN_PORT => "sceAudioOutOpenPort",
        au::OUT_OUTPUT => "sceAudioOutOutput",
        au::OUT_SET_VOLUME => "sceAudioOutSetVolume",
        au::OUT_RELEASE_PORT => "sceAudioOutReleasePort",
        au::OUT_GET_ADOPT => "sceAudioOutGetAdopt",
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
        g::SHADER_PATCHER_GET_PROGRAM_FROM_ID => "sceGxmShaderPatcherGetProgramFromId",
        g::PROGRAM_PARAMETER_GET_RESOURCE_INDEX => "sceGxmProgramParameterGetResourceIndex",
        g::PROGRAM_GET_PARAMETER_COUNT => "sceGxmProgramGetParameterCount",
        g::PROGRAM_GET_PARAMETER => "sceGxmProgramGetParameter",
        g::PROGRAM_PARAMETER_GET_CATEGORY => "sceGxmProgramParameterGetCategory",
        g::PROGRAM_PARAMETER_GET_TYPE => "sceGxmProgramParameterGetType",
        g::PROGRAM_PARAMETER_GET_COMPONENT_COUNT => "sceGxmProgramParameterGetComponentCount",
        g::PROGRAM_PARAMETER_GET_CONTAINER_INDEX => "sceGxmProgramParameterGetContainerIndex",
        g::PROGRAM_PARAMETER_GET_ARRAY_SIZE => "sceGxmProgramParameterGetArraySize",
        g::PROGRAM_PARAMETER_GET_NAME => "sceGxmProgramParameterGetName",
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
        g::SET_FRAGMENT_TEXTURE => "sceGxmSetFragmentTexture",
        g::TEXTURE_INIT_LINEAR => "sceGxmTextureInitLinear",
        g::TEXTURE_INIT_LINEAR_STRIDED => "sceGxmTextureInitLinearStrided",
        g::TEXTURE_INIT_SWIZZLED => "sceGxmTextureInitSwizzled",
        g::TEXTURE_INIT_SWIZZLED_ARBITRARY => "sceGxmTextureInitSwizzledArbitrary",
        g::TEXTURE_INIT_TILED => "sceGxmTextureInitTiled",
        g::TEXTURE_SET_DATA => "sceGxmTextureSetData",
        g::TEXTURE_SET_FORMAT => "sceGxmTextureSetFormat",
        g::TEXTURE_SET_MAG_FILTER => "sceGxmTextureSetMagFilter",
        g::TEXTURE_SET_MIN_FILTER => "sceGxmTextureSetMinFilter",
        g::TEXTURE_SET_MIP_FILTER => "sceGxmTextureSetMipFilter",
        g::TEXTURE_SET_U_ADDR_MODE => "sceGxmTextureSetUAddrMode",
        g::TEXTURE_SET_V_ADDR_MODE => "sceGxmTextureSetVAddrMode",
        g::TEXTURE_GET_DATA => "sceGxmTextureGetData",
        g::TEXTURE_GET_WIDTH => "sceGxmTextureGetWidth",
        g::TEXTURE_GET_HEIGHT => "sceGxmTextureGetHeight",
        g::TEXTURE_GET_FORMAT => "sceGxmTextureGetFormat",
        g::SET_FRAGMENT_UNIFORM_BUFFER => "sceGxmSetFragmentUniformBuffer",
        g::RESERVE_FRAGMENT_DEFAULT_UNIFORM_BUFFER => "sceGxmReserveFragmentDefaultUniformBuffer",
        d::SET_FRAME_BUF => "sceDisplaySetFrameBuf",
        c::PEEK_BUFFER_POSITIVE => "sceCtrlPeekBufferPositive",
        c::READ_BUFFER_POSITIVE => "sceCtrlReadBufferPositive",
        c::PEEK_BUFFER_NEGATIVE => "sceCtrlPeekBufferNegative",
        c::READ_BUFFER_NEGATIVE => "sceCtrlReadBufferNegative",
        c::SET_SAMPLING_MODE => "sceCtrlSetSamplingMode",
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
        lk::GET_TLS_ADDR => "sceKernelGetTLSAddr",
        lk::GET_PROCESS_TIME => "sceKernelGetProcessTime",
        lk::GET_PROCESS_TIME_WIDE => "sceKernelGetProcessTimeWide",
        lk::GET_THREAD_EXIT_STATUS => "sceKernelGetThreadExitStatus",
        lk::EXIT_PROCESS => "sceKernelExitProcess",
        pm::POWER_TICK => "sceKernelPowerTick",
        pm::GET_PROCESS_PARAM => "sceKernelGetProcessParam",
        pm::GET_STDIN => "sceKernelGetStdin",
        pm::GET_STDOUT => "sceKernelGetStdout",
        pm::GET_STDERR => "sceKernelGetStderr",
        pm::LIBC_TIME => "sceKernelLibcTime",
        pm::LIBC_CLOCK => "sceKernelLibcClock",
        tm::GET_PROCESS_ID => "sceKernelGetProcessId",
        sv::SYSMODULE_IS_LOADED => "sceSysmoduleIsLoaded",
        sv::NET_INIT => "sceNetInit",
        sv::NET_CTL_INIT => "sceNetCtlInit",
        sv::NET_CTL_INET_GET_STATE => "sceNetCtlInetGetState",
        sv::NET_CTL_INET_REGISTER_CALLBACK => "sceNetCtlInetRegisterCallback",
        sv::HTTP_INIT => "sceHttpInit",
        sv::SSL_INIT => "sceSslInit",
        sv::NP_INIT => "sceNpInit",
        sv::NP_REGISTER_SERVICE_STATE_CALLBACK => "sceNpRegisterServiceStateCallback",
        sv::NP_BASIC_INIT => "sceNpBasicInit",
        sv::NP_BASIC_REGISTER_HANDLER => "sceNpBasicRegisterHandler",
        sv::NP_BASIC_CHECK_CALLBACK => "sceNpBasicCheckCallback",
        sv::RTC_GET_CURRENT_CLOCK => "sceRtcGetCurrentClock",
        sv::RTC_GET_CURRENT_CLOCK_LOCAL_TIME => "sceRtcGetCurrentClockLocalTime",
        sv::RTC_GET_TICK => "sceRtcGetTick",
        sv::FIOS_OVERLAY_GET_LIST => "sceFiosOverlayGetList02",
        sv::ULOBJ_REGISTER_PROTOCOL_REVISION => "_sceUlobjMgrRegisterLibultProtocolRevision",
        sv::APPUTIL_INIT => "sceAppUtilInit",
        sv::APPUTIL_SYSTEM_PARAM_GET_INT => "sceAppUtilSystemParamGetInt",
        sv::APPUTIL_APP_PARAM_GET_INT => "sceAppUtilAppParamGetInt",
        sv::LIVE_AREA_GET_STATUS => "sceLiveAreaGetStatus",
        sv::LIVE_AREA_UPDATE_FRAME_ASYNC => "sceLiveAreaUpdateFrameAsync",
        sv::NP_SCORE_INIT => "sceNpScoreInit",
        sv::NP_SCORE_CREATE_TITLE_CTX => "sceNpScoreCreateTitleCtx",
        sv::NP_MANAGER_GET_NP_ID => "sceNpManagerGetNpId",
        sv::NP_MANAGER_GET_ACCOUNT_REGION => "sceNpManagerGetAccountRegion",
        sv::NP_MANAGER_GET_CONTENT_RATING_FLAG => "sceNpManagerGetContentRatingFlag",
        sv::NP_MANAGER_GET_CHAT_RESTRICTION_FLAG => "sceNpManagerGetChatRestrictionFlag",
        sv::APP_MGR_GET_APP_STATE => "_sceAppMgrGetAppState",
        sv::NET_CTL_CHECK_CALLBACK => "sceNetCtlCheckCallback",
        sv::APPUTIL_DRM_OPEN => "sceAppUtilDrmOpen",
        sv::APPUTIL_DRM_CLOSE => "sceAppUtilDrmClose",
        sv::APPUTIL_SAVEDATA_SLOT_GET_PARAM => "sceAppUtilSaveDataSlotGetParam",
        sv::APPUTIL_SAVEDATA_SLOT_CREATE => "sceAppUtilSaveDataSlotCreate",
        sv::NP_CHECK_CALLBACK => "sceNpCheckCallback",
        sv::TOUCH_SET_SAMPLING_STATE => "sceTouchSetSamplingState",
        sv::TOUCH_READ => "sceTouchRead",
        sv::TOUCH_PEEK => "sceTouchPeek",
        sv::TOUCH_GET_PANEL_INFO => "sceTouchGetPanelInfo",
        sv::TOUCH_ENABLE_TOUCH_FORCE => "sceTouchEnableTouchForce",
        sv::SYSMODULE_LOAD_MODULE => "sceSysmoduleLoadModule",
        sv::APPUTIL_SYSTEM_PARAM_GET_STRING => "sceAppUtilSystemParamGetString",
        lw::CREATE_LW_MUTEX => "sceKernelCreateLwMutex",
        lw::DELETE_LW_MUTEX => "sceKernelDeleteLwMutex",
        lw::LOCK_LW_MUTEX => "sceKernelLockLwMutex",
        lw::LOCK_LW_MUTEX_CB => "sceKernelLockLwMutexCB",
        lw::TRY_LOCK_LW_MUTEX => "sceKernelTryLockLwMutex",
        lw::UNLOCK_LW_MUTEX => "sceKernelUnlockLwMutex",
        lw::UNLOCK_LW_MUTEX2 => "sceKernelUnlockLwMutex2",
        lw::CREATE_LW_COND => "sceKernelCreateLwCond",
        lw::DELETE_LW_COND => "sceKernelDeleteLwCond",
        lw::WAIT_LW_COND => "sceKernelWaitLwCond",
        lw::WAIT_LW_COND_CB => "sceKernelWaitLwCondCB",
        lw::SIGNAL_LW_COND => "sceKernelSignalLwCond",
        lw::SIGNAL_LW_COND_ALL => "sceKernelSignalLwCondAll",
        lw::SIGNAL_LW_COND_TO => "sceKernelSignalLwCondTo",
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
        io::IO_PREAD => "sceIoPread",
        io::IO_PWRITE => "sceIoPwrite",
        io::IO_GETSTAT => "sceIoGetstat",
        io::IO_GETSTAT_BY_FD => "sceIoGetstatByFd",
        io::IO_MKDIR => "sceIoMkdir",
        io::IO_REMOVE => "sceIoRemove",
        io::IO_DOPEN => "sceIoDopen",
        io::IO_DREAD => "sceIoDread",
        io::IO_DCLOSE => "sceIoDclose",
        tm::DELAY_THREAD => "sceKernelDelayThread",
        tm::EXIT_DELETE_THREAD => "sceKernelExitDeleteThread",
        tm::EXIT_THREAD => "sceKernelExitThread",
        sy::CREATE_MUTEX => "sceKernelCreateMutex",
        sy::LOCK_MUTEX => "sceKernelLockMutex",
        sy::TRY_LOCK_MUTEX => "sceKernelTryLockMutex",
        sy::UNLOCK_MUTEX => "sceKernelUnlockMutex",
        sy::DELETE_MUTEX => "sceKernelDeleteMutex",
        sy::CREATE_SEMA | sy::CREATE_SEMA_16XX => "sceKernelCreateSema",
        sy::WAIT_SEMA => "sceKernelWaitSema",
        sy::SIGNAL_SEMA => "sceKernelSignalSema",
        sy::DELETE_SEMA => "sceKernelDeleteSema",
        sy::CREATE_EVENT_FLAG => "sceKernelCreateEventFlag",
        sy::SET_EVENT_FLAG => "sceKernelSetEventFlag",
        sy::WAIT_EVENT_FLAG => "sceKernelWaitEventFlag",
        sy::WAIT_EVENT_FLAG_CB => "sceKernelWaitEventFlagCB",
        sy::CLEAR_EVENT_FLAG => "sceKernelClearEventFlag",
        sy::DELETE_EVENT_FLAG => "sceKernelDeleteEventFlag",
        sy::GET_SYSTEM_TIME_WIDE => "sceKernelGetSystemTimeWide",
        sy::CREATE_COND => "sceKernelCreateCond",
        sy::WAIT_COND => "sceKernelWaitCond",
        sy::SIGNAL_COND => "sceKernelSignalCond",
        sy::SIGNAL_COND_ALL => "sceKernelSignalCondAll",
        sy::DELETE_COND => "sceKernelDeleteCond",
        // GXM render-state setters, getters, cube texture, sampler-state setters.
        g::SET_CULL_MODE => "sceGxmSetCullMode",
        g::SET_TWO_SIDED_ENABLE => "sceGxmSetTwoSidedEnable",
        g::SET_FRONT_DEPTH_FUNC => "sceGxmSetFrontDepthFunc",
        g::SET_BACK_DEPTH_FUNC => "sceGxmSetBackDepthFunc",
        g::SET_FRONT_DEPTH_WRITE_ENABLE => "sceGxmSetFrontDepthWriteEnable",
        g::SET_FRONT_FRAGMENT_PROGRAM_ENABLE => "sceGxmSetFrontFragmentProgramEnable",
        g::SET_BACK_FRAGMENT_PROGRAM_ENABLE => "sceGxmSetBackFragmentProgramEnable",
        g::SET_FRONT_POINT_LINE_WIDTH => "sceGxmSetFrontPointLineWidth",
        g::SET_FRONT_POLYGON_MODE => "sceGxmSetFrontPolygonMode",
        g::SET_FRONT_STENCIL_REF => "sceGxmSetFrontStencilRef",
        g::SET_FRONT_STENCIL_FUNC => "sceGxmSetFrontStencilFunc",
        g::SET_VIEWPORT => "sceGxmSetViewport",
        g::SET_VIEWPORT_ENABLE => "sceGxmSetViewportEnable",
        g::SET_REGION_CLIP => "sceGxmSetRegionClip",
        g::COLOR_SURFACE_GET_FORMAT => "sceGxmColorSurfaceGetFormat",
        g::COLOR_SURFACE_GET_TYPE => "sceGxmColorSurfaceGetType",
        g::COLOR_SURFACE_SET_CLIP => "sceGxmColorSurfaceSetClip",
        g::TEXTURE_GET_TYPE => "sceGxmTextureGetType",
        g::PROGRAM_PARAMETER_GET_SEMANTIC => "sceGxmProgramParameterGetSemantic",
        g::PROGRAM_PARAMETER_GET_SEMANTIC_INDEX => "sceGxmProgramParameterGetSemanticIndex",
        g::TEXTURE_INIT_CUBE => "sceGxmTextureInitCube",
        g::TEXTURE_SET_U_ADDR_MODE_SAFE => "sceGxmTextureSetUAddrModeSafe",
        g::TEXTURE_SET_V_ADDR_MODE_SAFE => "sceGxmTextureSetVAddrModeSafe",
        g::TEXTURE_SET_LOD_BIAS => "sceGxmTextureSetLodBias",
        g::COLOR_SURFACE_GET_DATA => "sceGxmColorSurfaceGetData",
        g::COLOR_SURFACE_GET_STRIDE_IN_PIXELS => "sceGxmColorSurfaceGetStrideInPixels",
        g::COLOR_SURFACE_SET_GAMMA_MODE => "sceGxmColorSurfaceSetGammaMode",
        g::GET_RENDER_TARGET_MEM_SIZE => "sceGxmGetRenderTargetMemSize",
        g::GET_NOTIFICATION_REGION => "sceGxmGetNotificationRegion",
        g::PROGRAM_GET_DEFAULT_UNIFORM_BUFFER_SIZE => "sceGxmProgramGetDefaultUniformBufferSize",
        g::FRAGMENT_PROGRAM_GET_PASS_TYPE => "sceGxmFragmentProgramGetPassType",
        g::TEXTURE_GET_MIPMAP_COUNT_UNSAFE => "sceGxmTextureGetMipmapCountUnsafe",
        g::TEXTURE_GET_STRIDE => "sceGxmTextureGetStride",
        g::TEXTURE_GET_LOD_BIAS => "sceGxmTextureGetLodBias",
        g::TEXTURE_GET_U_ADDR_MODE_SAFE => "sceGxmTextureGetUAddrModeSafe",
        g::TEXTURE_GET_V_ADDR_MODE_SAFE => "sceGxmTextureGetVAddrModeSafe",
        g::TEXTURE_GET_MAG_FILTER => "sceGxmTextureGetMagFilter",
        g::TEXTURE_GET_MIN_FILTER => "sceGxmTextureGetMinFilter",
        g::TEXTURE_GET_GAMMA_MODE => "sceGxmTextureGetGammaMode",
        g::TEXTURE_SET_GAMMA_MODE => "sceGxmTextureSetGammaMode",
        g::GET_PRECOMPUTED_DRAW_SIZE => "sceGxmGetPrecomputedDrawSize",
        g::PRECOMPUTED_DRAW_INIT => "sceGxmPrecomputedDrawInit",
        g::PRECOMPUTED_DRAW_SET_PARAMS => "sceGxmPrecomputedDrawSetParams",
        g::PRECOMPUTED_DRAW_SET_VERTEX_STREAM => "sceGxmPrecomputedDrawSetVertexStream",
        g::DRAW_PRECOMPUTED => "sceGxmDrawPrecomputed",
        g::GET_PRECOMPUTED_VERTEX_STATE_SIZE => "sceGxmGetPrecomputedVertexStateSize",
        g::GET_PRECOMPUTED_FRAGMENT_STATE_SIZE => "sceGxmGetPrecomputedFragmentStateSize",
        g::PRECOMPUTED_VERTEX_STATE_INIT => "sceGxmPrecomputedVertexStateInit",
        g::PRECOMPUTED_FRAGMENT_STATE_INIT => "sceGxmPrecomputedFragmentStateInit",
        g::PRECOMPUTED_VERTEX_STATE_SET_DEFAULT_UNIFORM_BUFFER => "sceGxmPrecomputedVertexStateSetDefaultUniformBuffer",
        g::PRECOMPUTED_FRAGMENT_STATE_SET_DEFAULT_UNIFORM_BUFFER => "sceGxmPrecomputedFragmentStateSetDefaultUniformBuffer",
        g::PRECOMPUTED_VERTEX_STATE_GET_DEFAULT_UNIFORM_BUFFER => "sceGxmPrecomputedVertexStateGetDefaultUniformBuffer",
        g::PRECOMPUTED_FRAGMENT_STATE_GET_DEFAULT_UNIFORM_BUFFER => "sceGxmPrecomputedFragmentStateGetDefaultUniformBuffer",
        g::PRECOMPUTED_VERTEX_STATE_SET_TEXTURE => "sceGxmPrecomputedVertexStateSetTexture",
        g::PRECOMPUTED_FRAGMENT_STATE_SET_TEXTURE => "sceGxmPrecomputedFragmentStateSetTexture",
        g::SET_PRECOMPUTED_VERTEX_STATE => "sceGxmSetPrecomputedVertexState",
        g::SET_PRECOMPUTED_FRAGMENT_STATE => "sceGxmSetPrecomputedFragmentState",
        // NGS.
        ng::VOICE_DEF_GET_SIMPLE_VOICE => "sceNgsVoiceDefGetSimpleVoice",
        ng::VOICE_DEF_GET_MIXER_BUSS => "sceNgsVoiceDefGetMixerBuss",
        ng::VOICE_PATCH_SET_VOLUME => "sceNgsVoicePatchSetVolume",
        ng::PATCH_REMOVE_ROUTING => "sceNgsPatchRemoveRouting",
        ng::SYSTEM_LOCK => "sceNgsSystemLock",
        ng::SYSTEM_UNLOCK => "sceNgsSystemUnlock",
        // ScePvf (font).
        pv::NEW_LIB => "scePvfNewLib",
        pv::DONE_LIB => "scePvfDoneLib",
        pv::SET_EM => "scePvfSetEM",
        pv::OPEN => "scePvfOpen",
        pv::OPEN_USER_FILE => "scePvfOpenUserFile",
        pv::SET_RESOLUTION => "scePvfSetResolution",
        pv::SET_CHAR_SIZE => "scePvfSetCharSize",
        pv::SET_SKEW_VALUE => "scePvfSetSkewValue",
        pv::GET_FONT_INFO => "scePvfGetFontInfo",
        pv::GET_CHAR_INFO => "scePvfGetCharInfo",
        pv::GET_CHAR_IMAGE_RECT => "scePvfGetCharImageRect",
        pv::IS_ELEMENT => "scePvfIsElement",
        pv::GET_CHAR_GLYPH_IMAGE => "scePvfGetCharGlyphImage",
        pv::PIXEL_TO_POINT_H => "scePvfPixelToPointH",
        pv::PIXEL_TO_POINT_V => "scePvfPixelToPointV",
        // Threadmgr, sysmem, display additions.
        tm::GET_THREAD_CURRENT_PRIORITY => "sceKernelGetThreadCurrentPriority",
        tm::CLOSE_SEMA => "sceKernelCloseSema",
        tm::CHANGE_THREAD_VFP_EXCEPTION => "sceKernelChangeThreadVfpException",
        s::FREE_MEM_BLOCK => "sceKernelFreeMemBlock",
        d::WAIT_VBLANK_START_MULTI => "sceDisplayWaitVblankStartMulti",
        d::WAIT_SET_FRAME_BUF => "sceDisplayWaitSetFrameBuf",
        lk::UNKNOWN_023EAA62 => "SceLibKernel_023EAA62",
        // Offline services: screenshot, trophy, Np inits, location/motion/power/net.
        sv::SCREENSHOT_DISABLE => "sceScreenShotDisable",
        sv::SCREENSHOT_ENABLE => "sceScreenShotEnable",
        sv::SCREENSHOT_SET_PARAM => "sceScreenShotSetParam",
        sv::SCREENSHOT_SET_OVERLAY_IMAGE => "sceScreenShotSetOverlayImage",
        sv::NP_TROPHY_INIT => "sceNpTrophyInit",
        sv::NP_TROPHY_CREATE_CONTEXT => "sceNpTrophyCreateContext",
        sv::NP_TROPHY_CREATE_HANDLE => "sceNpTrophyCreateHandle",
        sv::NP_TROPHY_GET_GAME_INFO => "sceNpTrophyGetGameInfo",
        sv::NP_TROPHY_GET_TROPHY_UNLOCK_STATE => "sceNpTrophyGetTrophyUnlockState",
        sv::NP_ACTIVITY_INIT => "sceNpActivityInit",
        sv::NP_AUTH_INIT => "sceNpAuthInit",
        sv::NP_LOOKUP_INIT => "sceNpLookupInit",
        sv::NP_TUS_INIT => "sceNpTusInit",
        sv::NP_SNS_FACEBOOK_INIT => "sceNpSnsFacebookInit",
        sv::LOCATION_INIT => "sceLocationInit",
        sv::MOTION_START_SAMPLING => "sceMotionStartSampling",
        sv::NETCTL_ADHOC_REGISTER_CALLBACK => "sceNetCtlAdhocRegisterCallback",
        sv::POWER_SET_CONFIGURATION_MODE => "scePowerSetConfigurationMode",
        sv::COMMON_DIALOG_SET_CONFIG_PARAM => "sceCommonDialogSetConfigParam",
        sv::NP_TROPHY_SETUP_DIALOG_GET_RESULT => "sceNpTrophySetupDialogGetResult",
        sv::NEAR_UTIL_UNKNOWN_A412E9CA => "SceNearUtil_A412E9CA",
        _ => "<unknown>",
    }
}
