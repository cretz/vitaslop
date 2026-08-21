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
    /// `sceGxmColorSurfaceInitDisabled`: a colour surface that writes nothing, which is
    /// what a depth-only pass (a shadow map, a z-prepass) binds.
    pub const COLOR_SURFACE_INIT_DISABLED: u32 = 0x6136_39FA;
    pub const DEPTH_STENCIL_SURFACE_INIT: u32 = 0xCA9D_41D1;
    pub const SYNC_OBJECT_CREATE: u32 = 0x6A60_13E1;
    pub const SYNC_OBJECT_DESTROY: u32 = 0x889A_E88C;
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
    pub const DRAW_INSTANCED: u32 = 0x14C4_E7D3;
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
    pub const SET_VERTEX_UNIFORM_BUFFER: u32 = 0xC680_15E4;
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
    /// `sceGxmSetBackStencilFunc`: the two-sided counterpart, applied when
    /// `sceGxmSetTwoSidedEnable` is on. Recorded unconditionally - see `vita::gxm`.
    pub const SET_BACK_STENCIL_FUNC: u32 = 0x1A68_C8D2;
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
    /// The checked variant of the above. GXM ships `Safe`/`Unsafe` pairs of several
    /// getters that differ only in whether the argument is validated, so both read the
    /// same field and share one handler here - but they are DIFFERENT NIDs, and a title
    /// linking the one we did not register gets a hard failure.
    pub const TEXTURE_GET_MIPMAP_COUNT: u32 = 0xF7B7_B1E4;
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
    // Depth/stencil surface: the whole struct layout is published (`SceGxmDepthStencilSurface`,
    // 0x14 bytes), so these setters write the real fields rather than a shadow.
    pub const DEPTH_STENCIL_SURFACE_SET_BACKGROUND_DEPTH: u32 = 0x32F2_80F0;
    pub const DEPTH_STENCIL_SURFACE_SET_BACKGROUND_STENCIL: u32 = 0xF5D3_F3E8;
    pub const DEPTH_STENCIL_SURFACE_SET_FORCE_LOAD_MODE: u32 = 0x0C44_ACD7;
    pub const DEPTH_STENCIL_SURFACE_SET_FORCE_STORE_MODE: u32 = 0x12AA_A7AF;
    // Back-face render state (the front-face counterparts are above).
    pub const SET_BACK_DEPTH_WRITE_ENABLE: u32 = 0xC18B_706B;
    pub const SET_BACK_POLYGON_MODE: u32 = 0xF66E_C6FE;
    // Occlusion queries.
    pub const SET_VISIBILITY_BUFFER: u32 = 0x7767_EC49;
    pub const SET_FRONT_VISIBILITY_TEST_ENABLE: u32 = 0x3045_9117;
    pub const SET_FRONT_VISIBILITY_TEST_INDEX: u32 = 0x1262_5C34;
    pub const SET_FRONT_VISIBILITY_TEST_OP: u32 = 0xD0E3_CD9A;
    // Unmapping: the inverse of the three map calls.
    pub const UNMAP_MEMORY: u32 = 0x828C_68E8;
    pub const UNMAP_VERTEX_USSE_MEMORY: u32 = 0x0991_34F5;
    pub const UNMAP_FRAGMENT_USSE_MEMORY: u32 = 0x80CC_EDBB;
    // Colour surface: the scale mode `Init` takes, and a data-pointer rebind.
    pub const COLOR_SURFACE_GET_SCALE_MODE: u32 = 0x6E3F_A74D;
    pub const COLOR_SURFACE_SET_DATA: u32 = 0x537C_A400;
    // Reflection + render-target/notification.
    pub const PROGRAM_GET_TYPE: u32 = 0x04BB_3C59;
    pub const PROGRAM_FIND_PARAMETER_BY_SEMANTIC: u32 = 0x7FFF_DD7A;
    pub const RENDER_TARGET_GET_DRIVER_MEM_BLOCK: u32 = 0x4955_3737;
    pub const NOTIFICATION_WAIT: u32 = 0x9F44_8E79;
    // Vertex-stage texture bind, cube-arbitrary init, paletted textures.
    pub const SET_VERTEX_TEXTURE: u32 = 0x16C9_D339;
    pub const TEXTURE_INIT_CUBE_ARBITRARY: u32 = 0xE3DF_5E3B;
    pub const TEXTURE_SET_PALETTE: u32 = 0xDD6A_ABFA;
    // Precomputed: the whole-array setters and the non-default uniform-buffer slots.
    pub const PRECOMPUTED_DRAW_SET_ALL_VERTEX_STREAMS: u32 = 0xB6C6_F571;
    pub const PRECOMPUTED_FRAGMENT_STATE_SET_ALL_TEXTURES: u32 = 0xC383_DE39;
    pub const PRECOMPUTED_VERTEX_STATE_SET_ALL_TEXTURES: u32 = 0xC40C_9127;
    pub const PRECOMPUTED_FRAGMENT_STATE_SET_ALL_UNIFORM_BUFFERS: u32 = 0x5A78_3DC3;
    pub const PRECOMPUTED_FRAGMENT_STATE_SET_UNIFORM_BUFFER: u32 = 0xB452_F1FB;
    pub const PRECOMPUTED_VERTEX_STATE_SET_ALL_UNIFORM_BUFFERS: u32 = 0x0389_861D;
    pub const PRECOMPUTED_VERTEX_STATE_SET_UNIFORM_BUFFER: u32 = 0xDBF9_7ED6;
}

/// SceDisplayUser / SceDisplay function NIDs. `SET_FRAME_BUF` is SceDisplayUser
/// (lib 0x4FAACD11); `WAIT_VBLANK_START_MULTI` is SceDisplay (lib 0x5ED8F994).
/// Dispatch is by func NID, so the two libraries are grouped by concept here.
pub mod display {
    pub const SET_FRAME_BUF: u32 = 0x7A41_0B64;
    pub const WAIT_VBLANK_START_MULTI: u32 = 0xDD0A_13B8;
    pub const WAIT_VBLANK_START: u32 = 0x5795_E898;
    /// `sceDisplayWaitSetFrameBuf` (SceDisplay, lib 0x5ED8F994): block until the
    /// frame buffer queued by `sceDisplaySetFrameBuf` has been latched at vblank.
    pub const WAIT_SET_FRAME_BUF: u32 = 0x9423_560C;
    pub const GET_VCOUNT: u32 = 0xB6FD_E0BA;
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
    /// SceDebugLed's user-mode export, declared in the same module as the kernel's
    /// `ksceKernelSetGPO` and sharing its NID.
    pub const SET_GPO: u32 = 0x78E7_02D3;
    /// Resolve an address back to the memory block that contains it.
    pub const FIND_MEM_BLOCK_BY_ADDR: u32 = 0xA33B_99D1;
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
    pub const CLIB_STRRCHR: u32 = 0x6E72_8AAE;
    pub const CLIB_STRNCASECMP: u32 = 0xB54C_0BE4;
    // clib memory spaces: a general allocator over a block of the title's own memory.
    pub const CLIB_MSPACE_CREATE: u32 = 0x3B9E_301A;
    pub const CLIB_MSPACE_DESTROY: u32 = 0xAE1A_21EC;
    pub const CLIB_MSPACE_MALLOC: u32 = 0x86EF_7680;
    pub const CLIB_MSPACE_FREE: u32 = 0x9C56_B4D1;
    pub const CLIB_MSPACE_MEMALIGN: u32 = 0x3C84_7D57;
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
    // Thread and semaphore introspection.
    pub const GET_THREAD_INFO: u32 = 0x8D9C_5461;
    pub const GET_SEMA_INFO: u32 = 0x595D_3FA6;
    // Per-thread signals (sceKernelSendSignal's counterpart).
    pub const WAIT_SIGNAL: u32 = 0xADCA_94E5;
    // Callback-processing variants of the blocking waits. Same wait, plus a point at
    // which the kernel delivers pending callbacks to the calling thread.
    pub const WAIT_SEMA_CB: u32 = 0x1746_92B4;
    pub const WAIT_THREAD_END_CB: u32 = 0xC549_41ED;
    // Process/module queries.
    pub const GET_PROCESS_TIME_LOW: u32 = 0xE9F9_73B1;
    pub const GET_OPEN_PS_ID: u32 = 0x261E_2C34;
    pub const GET_MODULE_INFO_BY_ADDR: u32 = 0xD11A_5103;
    pub const CALL_MODULE_EXIT: u32 = 0x15E2_A45D;
    /// The ARM EABI divide-by-zero hooks. Both return their argument unchanged.
    pub const AEABI_IDIV0: u32 = 0x4373_B548;
    pub const AEABI_LDIV0: u32 = 0xFB23_5848;
    // File operations SceLibKernel re-exports on top of SceIofilemgr.
    pub const IO_CHSTAT: u32 = 0x2948_2F7F;
    pub const IO_DEVCTL: u32 = 0x04B3_0CB2;
    pub const IO_IOCTL: u32 = 0x54AB_ACFA;
    pub const IO_RENAME: u32 = 0xF737_E369;
    pub const IO_RMDIR: u32 = 0xE9F9_1EC8;
    pub const IO_SYNC: u32 = 0x98AC_ED6D;
}

/// SceFios2Kernel: the kernel-side path OVERLAY layer beneath FIOS2. Titles ship
/// FIOS2's user half in their own package, so these are the floor that shipped
/// library stands on. See `vita::fios2`.
pub mod fios2 {
    pub const OVERLAY_ADD: u32 = 0x6DBC_F0B2;
    pub const OVERLAY_ADD_FOR_PROCESS: u32 = 0x2A38_1357;
    pub const OVERLAY_MODIFY: u32 = 0x6D6C_DE05;
    pub const OVERLAY_MODIFY_FOR_PROCESS: u32 = 0x6DF2_FC05;
    pub const OVERLAY_REMOVE: u32 = 0xB492_7173;
    pub const OVERLAY_REMOVE_FOR_PROCESS: u32 = 0xF827_7E07;
    pub const OVERLAY_GET_INFO: u32 = 0xF44F_3505;
    pub const OVERLAY_GET_INFO_FOR_PROCESS: u32 = 0xBC6B_3CC5;
    pub const OVERLAY_GET_LIST: u32 = 0x9379_E2D5;
    pub const OVERLAY_RESOLVE_SYNC: u32 = 0xE9AE_60FB;
    pub const OVERLAY_RESOLVE_WITH_RANGE_SYNC: u32 = 0x8CCA_471A;
    pub const OVERLAY_GET_RECOMMENDED_SCHEDULER: u32 = 0xB02E_0B26;
    pub const OVERLAY_THREAD_IS_DISABLED: u32 = 0x629F_4FE4;
    pub const OVERLAY_THREAD_SET_DISABLED: u32 = 0x3E91_72EA;
    pub const DH_OPEN_SYNC: u32 = 0x5D6A_1CCE;
    pub const DH_READ_SYNC: u32 = 0x2F06_ADC6;
    pub const DH_STAT_SYNC: u32 = 0x759E_BEE6;
    pub const DH_CHSTAT_SYNC: u32 = 0xF6A3_E335;
    pub const DH_SYNC_SYNC: u32 = 0x2A97_24C9;
    pub const DH_CLOSE_SYNC: u32 = 0x021B_4AF7;
}

/// SceFiber: cooperative user-level threads on guest-supplied stacks. See
/// [`crate::vita::fiber`] for the model and for the two calls whose prototypes are
/// unpublished.
pub mod fiber {
    pub const INITIALIZE_IMPL: u32 = 0xF24A_298C;
    pub const INITIALIZE_WITH_INTERNAL_OPTION_IMPL: u32 = 0xC6A3_F9BB;
    pub const ATTACH_CONTEXT_AND_SWITCH: u32 = 0xE00B_9AFE;
    pub const FINALIZE: u32 = 0xE160_F844;
    pub const GET_INFO: u32 = 0x1895_99B4;
    pub const GET_SELF: u32 = 0x414D_8CA5;
    pub const RETURN_TO_THREAD: u32 = 0x3B42_921F;
    pub const RUN: u32 = 0x7DF2_3243;
    pub const SWITCH: u32 = 0xE428_3144;
}

/// SceNet: the BSD-sockets surface, modelled OFFLINE (see [`crate::vita::net`]).
pub mod net {
    pub const SOCKET: u32 = 0xF084_FCE3;
    pub const SOCKET_CLOSE: u32 = 0x2982_2B4D;
    pub const BIND: u32 = 0x1296_A94B;
    pub const LISTEN: u32 = 0x7A8D_A094;
    pub const ACCEPT: u32 = 0x1ADF_9BB1;
    pub const CONNECT: u32 = 0x11E5_B6F6;
    pub const SEND: u32 = 0xE3DD_8CD9;
    pub const SENDTO: u32 = 0x52DB_31D5;
    pub const SENDMSG: u32 = 0x99C5_79AE;
    pub const RECV: u32 = 0x0236_43B7;
    pub const RECVFROM: u32 = 0xB226_138B;
    pub const SHUTDOWN: u32 = 0x69E5_0BB5;
    pub const GETSOCKNAME: u32 = 0x1C66_A6DB;
    pub const GETPEERNAME: u32 = 0x2348_D353;
    pub const SETSOCKOPT: u32 = 0x0655_05CA;
    pub const GETSOCKOPT: u32 = 0xBA65_2062;
    pub const GET_SOCK_INFO: u32 = 0xB1AF_6840;
    pub const SHOW_NETSTAT: u32 = 0x338E_DC2E;
    pub const HTONL: u32 = 0x4C30_B03C;
    pub const NTOHL: u32 = 0xD2EA_A645;
    pub const HTONS: u32 = 0x9FA3_207B;
    pub const NTOHS: u32 = 0x0784_5128;
    pub const INET_PTON: u32 = 0xD5EE_B048;
    pub const INET_NTOP: u32 = 0x9883_9B74;
    pub const ERRNO_LOC: u32 = 0xE37F_34AA;
    pub const RESOLVER_CREATE: u32 = 0x6DA2_9319;
    pub const RESOLVER_DESTROY: u32 = 0x3559_F098;
    pub const RESOLVER_START_NTOA: u32 = 0x1EB1_1857;
    pub const RESOLVER_START_ATON: u32 = 0x0424_AE26;
    pub const RESOLVER_GET_ERROR: u32 = 0x874E_F500;
    pub const EPOLL_CREATE: u32 = 0xF9D1_02AE;
    pub const EPOLL_DESTROY: u32 = 0x7915_CAF3;
    pub const EPOLL_CONTROL: u32 = 0x4C87_64AC;
    pub const EPOLL_WAIT: u32 = 0x45CE_337D;
}

/// SceLibDbg: the title's own assertion and logging handlers.
pub mod dbg {
    pub const ASSERTION_HANDLER: u32 = 0x1AF3_678B;
    pub const LOGGING_HANDLER: u32 = 0x6605_AB19;
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
    pub const LIBC_GETTIMEOFDAY: u32 = 0x4B87_9059;
    pub const CALL_ABORT_HANDLER: u32 = 0xEB6E_50BB;
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
    pub const NET_CTL_INET_GET_INFO: u32 = 0xB26D_07F3;
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
    pub const NP_BASIC_GET_FRIEND_LIST_ENTRY_COUNT: u32 = 0xDF41_F308;
    // SceRtc.
    pub const RTC_GET_CURRENT_CLOCK: u32 = 0x70FD_E8F1;
    pub const RTC_GET_CURRENT_CLOCK_LOCAL_TIME: u32 = 0x0572_EDDC;
    pub const RTC_GET_CURRENT_TICK: u32 = 0x23F7_9274;
    pub const RTC_GET_TICK: u32 = 0xF2B2_38E2;
    pub const RTC_GET_TIME64_T: u32 = 0xC995_DE02;
    /// `sceRtcGetTime_t`: the 32-bit `time_t` sibling of `RTC_GET_TIME64_T`.
    pub const RTC_GET_TIME_T: u32 = 0x8DE6_FEB7;
    pub const RTC_GET_CURRENT_NETWORK_TICK: u32 = 0xCDDD_25FE;
    pub const RTC_SET_TICK: u32 = 0xCD89_F464;
    pub const RTC_CONVERT_UTC_TO_LOCAL_TIME: u32 = 0x1282_C436;
    pub const RTC_CONVERT_LOCAL_TIME_TO_UTC: u32 = 0x0A05_E201;
    // The sceRtcTickAdd* family. The count is a 64-bit SceLong64 for the first four and a
    // plain int for the rest - see `rtc_tick_add_fixed`, where that distinction is read.
    pub const RTC_TICK_ADD_TICKS: u32 = 0x4559_E2DB;
    pub const RTC_TICK_ADD_MICROSECONDS: u32 = 0xAE26_D920;
    pub const RTC_TICK_ADD_SECONDS: u32 = 0x979A_FD79;
    pub const RTC_TICK_ADD_MINUTES: u32 = 0x4C35_8871;
    pub const RTC_TICK_ADD_HOURS: u32 = 0x6F19_3F55;
    pub const RTC_TICK_ADD_DAYS: u32 = 0x58DE_3C70;
    pub const RTC_TICK_ADD_WEEKS: u32 = 0xE713_C640;
    pub const RTC_TICK_ADD_MONTHS: u32 = 0x6321_B4AA;
    pub const RTC_TICK_ADD_YEARS: u32 = 0xDF6C_3E1B;
    // SceMotion.
    pub const MOTION_GET_STATE: u32 = 0xBDB3_2767;
    // SceFios2 overlay + libult object manager.
    pub const FIOS_OVERLAY_GET_LIST: u32 = 0x1DD8_08D1;
    pub const FIOS_OVERLAY_THREAD_SET_DISABLED: u32 = 0x7032_1220;
    pub const FIOS_OVERLAY_GET_RECOMMENDED_SCHEDULER: u32 = 0xF5C1_F928;
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
    pub const APPUTIL_SAVEDATA_SLOT_SET_PARAM: u32 = 0x9863_0136;
    pub const APPUTIL_SAVEDATA_SLOT_DELETE: u32 = 0x266A_7646;
    pub const APPUTIL_SAVEDATA_DATA_SAVE: u32 = 0x6076_47BA;
    pub const APPUTIL_SAVEDATA_DATA_REMOVE: u32 = 0xD1C6_AB8E;
    pub const APPUTIL_SAVEDATA_GET_QUOTA: u32 = 0xC560_E716;
    pub const APPUTIL_RECEIVE_APP_EVENT: u32 = 0xEE0D_BED9;
    pub const APPUTIL_APP_EVENT_PARSE_NEAR_GIFT: u32 = 0x7738_0601;
    pub const APPUTIL_APP_EVENT_PARSE_NP_BASIC_JOINABLE_PRESENCE: u32 = 0x28C7_D4F6;
    pub const APPUTIL_APP_EVENT_PARSE_NP_INVITE_MESSAGE: u32 = 0xA249_6814;
    // SceNetCtl async results, SceAppMgr, SceRtc, SceMotion, and the one-call services.
    pub const NET_CTL_INET_GET_RESULT: u32 = 0x6B20_EC02;
    pub const NET_CTL_ADHOC_GET_RESULT: u32 = 0x7AE0_ED19;
    pub const APPMGR_LOAD_EXEC: u32 = 0xE677_4ABC;
    pub const APPMGR_RECEIVE_SYSTEM_EVENT: u32 = 0x10B5_765F;
    pub const RTC_SET_TIME64_T: u32 = 0xA6C3_6B6A;
    pub const MOTION_SET_DEADBAND: u32 = 0x917E_A390;
    pub const MOTION_SET_TILT_CORRECTION: u32 = 0xAF09_FCDB;
    pub const SHUTTER_SOUND_PLAY: u32 = 0x7FFB_6D79;
    pub const PHOTO_EXPORT_FROM_DATA: u32 = 0x7051_2321;
    // SceNp: service state, the friend/presence surface, lookup requests and the
    // authentication (ticket / entitlement) surface. All modelled OFFLINE.
    pub const NP_GET_SERVICE_STATE: u32 = 0x5406_0DF6;
    pub const NP_ACTIVITY_POST_STATUS: u32 = 0xBC7F_DC77;
    pub const NP_BASIC_UNREGISTER_HANDLER: u32 = 0x050A_E072;
    pub const NP_BASIC_SET_IN_GAME_PRESENCE: u32 = 0x51D7_5562;
    pub const NP_BASIC_GET_GAME_JOINING_PRESENCE: u32 = 0x8632_49CB;
    pub const NP_BASIC_GET_FRIEND_LIST_ENTRIES: u32 = 0xFF07_E787;
    pub const NP_LOOKUP_CREATE_TITLE_CTX: u32 = 0x5110_E17E;
    pub const NP_LOOKUP_DELETE_REQUEST: u32 = 0x8B60_8BF6;
    pub const NP_LOOKUP_USER_PROFILE_ASYNC: u32 = 0xE528_5E0F;
    pub const NP_LOOKUP_POLL_ASYNC: u32 = 0xFCDB_A234;
    pub const NP_AUTH_CREATE_START_REQUEST: u32 = 0xED42_079F;
    pub const NP_AUTH_DESTROY_REQUEST: u32 = 0x14FC_18AF;
    pub const NP_AUTH_GET_TICKET: u32 = 0x5960_8D1C;
    pub const NP_AUTH_GET_TICKET_PARAM: u32 = 0xC1E2_3E01;
    pub const NP_AUTH_GET_ENTITLEMENT_BY_ID: u32 = 0xF938_42F0;
    pub const NP_AUTH_GET_ENTITLEMENT_ID_LIST: u32 = 0x3377_CD37;
    // SceAppMgr: app-lifecycle state poll (system/app event counts, overlay flag).
    pub const APP_MGR_GET_APP_STATE: u32 = 0x5E86_319A;
    pub const APP_MGR_IS_GAME_PROGRAM: u32 = 0xFFF8_F7F0;
    // SceNpScore / SceNpManager: online leaderboards and account identity.
    pub const NP_SCORE_INIT: u32 = 0x0433_069F;
    pub const NP_SCORE_TERM: u32 = 0x2050_F98F;
    pub const NP_SCORE_CREATE_TITLE_CTX: u32 = 0x5685_F225;
    pub const NP_MANAGER_GET_NP_ID: u32 = 0x3C94_B4B4;
    pub const NP_MANAGER_GET_ACCOUNT_REGION: u32 = 0xFE83_5967;
    pub const NP_MANAGER_GET_CONTENT_RATING_FLAG: u32 = 0xAF00_73B2;
    pub const NP_MANAGER_GET_CHAT_RESTRICTION_FLAG: u32 = 0x60C5_75B1;
    // SceNpUtility: online player lookup (NP id -> profile). No PSN session off-console.
    pub const NP_LOOKUP_CREATE_REQUEST: u32 = 0x9E42_E922;
    // SceNpMessage: server message sync - unreachable off-console (signed out).
    pub const NP_MESSAGE_SYNC_MESSAGE: u32 = 0x35BE_21C5;
    // SceNpTus: title user storage (online cloud stats) request - no session off-console.
    pub const NP_TUS_CREATE_REQUEST: u32 = 0x99DC_7420;
    // SceNpCommerce2: PS Store content check - store unreachable off-console (signed out).
    pub const NP_COMMERCE2_START_EMPTY_STORE_CHECK: u32 = 0x7132_EAA5;
    // The async store request's result poll - offline the request failed to reach the
    // server, so the poll reports the signed-out failure and the store check concludes.
    pub const NP_COMMERCE2_CREATE_SESSION_GET_RESULT: u32 = 0xAEE8_D3DF;
    pub const NP_COMMERCE2_CREATE_CTX: u32 = 0x123E_55F4;
    pub const NP_COMMERCE2_CREATE_SESSION_CREATE_REQ: u32 = 0xFDB3_9774;
    pub const NP_COMMERCE2_CREATE_SESSION_START: u32 = 0xBBDD_F866;
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
    pub const MSG_DIALOG_ABORT: u32 = 0x0CC6_6115;
    // SceIme: the on-screen keyboard dialog.
    pub const IME_DIALOG_INIT: u32 = 0x1E70_43BF;
    pub const IME_DIALOG_GET_STATUS: u32 = 0xCF04_31FD;
    pub const IME_DIALOG_GET_RESULT: u32 = 0x2EB3_D046;
    pub const IME_DIALOG_ABORT: u32 = 0x594A_220E;
    pub const IME_DIALOG_TERM: u32 = 0x838A_3AF4;
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
    // SceCamera: the two hardware cameras. Modelled as NOT PRESENT (see `vita::camera`);
    // the prototypes and this NID set are published in `psp2/camera.h`.
    pub const CAMERA_OPEN: u32 = 0xA462_F801;
    pub const CAMERA_CLOSE: u32 = 0xCD6E_1CFC;
    pub const CAMERA_START: u32 = 0xA8FE_AE35;
    pub const CAMERA_STOP: u32 = 0x1DD9_C9CE;
    pub const CAMERA_READ: u32 = 0x79B5_C2DE;
    pub const CAMERA_GET_REVERSE: u32 = 0x44F6_043F;
    pub const CAMERA_SET_REVERSE: u32 = 0x1175_F477;
    pub const CAMERA_SET_BACKLIGHT: u32 = 0xAE07_1044;
    pub const CAMERA_SET_WHITE_BALANCE: u32 = 0x4D45_14AC;
    // SceLibLocation: the positioning service, backed by the HOST's own location
    // provider (see `vita::location`) - real in the browser, absent on a bare desktop.
    // Prototypes: `psp2/location.h`; NIDs: `db/360/SceLibLocation.yml`.
    // The CALLBACK entry points (Start/StopLocationCallback, Start/StopHeadingCallback)
    // and SetGpsEmulationFile are deliberately absent - see the module docs.
    pub const LOCATION_OPEN: u32 = 0xDD27_1661;
    pub const LOCATION_CLOSE: u32 = 0x14FE_76E8;
    pub const LOCATION_REOPEN: u32 = 0xB1F5_5065;
    pub const LOCATION_GET_METHOD: u32 = 0x188C_E004;
    pub const LOCATION_CONFIRM: u32 = 0xC895_E567;
    pub const LOCATION_CONFIRM_GET_STATUS: u32 = 0x730F_F842;
    pub const LOCATION_CONFIRM_GET_RESULT: u32 = 0xFF01_6C13;
    pub const LOCATION_CONFIRM_ABORT: u32 = 0xE3CB_F875;
    pub const LOCATION_GET_LOCATION: u32 = 0x15BC_27C8;
    pub const LOCATION_GET_LOCATION_WITH_TIMEOUT: u32 = 0x16F4_1ED0;
    pub const LOCATION_CANCEL_GET_LOCATION: u32 = 0x7150_3251;
    pub const LOCATION_GET_HEADING: u32 = 0x4E9E_5ED9;
    pub const LOCATION_GET_PERMISSION: u32 = 0x4826_22C6;
    pub const LOCATION_DENY_APPLICATION: u32 = 0x8AAF_3FBD;
    // LOCATION_INIT is declared further down, with the device-service inits it was
    // first grouped with. It keeps its original home so this block stays additive.
    pub const LOCATION_TERM: u32 = 0x1E80_199A;
    pub const LOCATION_SET_THREAD_PARAMETER: u32 = 0xAA02_6B53;
    // SceJpegEnc: the hardware JPEG encoder. Setup only - see `vita::jpegenc` for why
    // Encode/Csc are deliberately left unimplemented. Prototypes: `psp2/jpegenc.h`.
    pub const JPEGENC_GET_CONTEXT_SIZE: u32 = 0x2B55_844D;
    pub const JPEGENC_INIT: u32 = 0x88DA_92B4;
    pub const JPEGENC_END: u32 = 0xC87A_A849;
    pub const JPEGENC_SET_OUTPUT_ADDR: u32 = 0x25D5_2D97;
    pub const JPEGENC_SET_COMPRESSION_RATIO: u32 = 0xB2B8_28EC;
    pub const JPEGENC_SET_VALID_REGION: u32 = 0x9511_F3BC;
    // SceJpeg: MJPEG decoder lifecycle only - see `vita::jpeg`.
    pub const JPEG_INIT_MJPEG: u32 = 0xB030_773B;
    pub const JPEG_FINISH_MJPEG: u32 = 0x6284_2598;
    // SceSystemGesture: gesture recognition layered on top of the touch panels. The NID
    // db names these; NO prototype or struct layout for the library is published
    // anywhere (vitasdk ships no `systemgesture.h`), so the argument shapes here are
    // read from the calling title's own code - see `vita::gesture`.
    pub const SYSTEM_GESTURE_INIT_PRIMITIVE_TOUCH_RECOGNIZER: u32 = 0x6078_A08B;
    pub const SYSTEM_GESTURE_UPDATE_PRIMITIVE_TOUCH_RECOGNIZER: u32 = 0xDF4C_665A;
    pub const SYSTEM_GESTURE_CREATE_TOUCH_RECOGNIZER: u32 = 0xC336_7370;
    pub const SYSTEM_GESTURE_UPDATE_TOUCH_RECOGNIZER: u32 = 0x851F_B144;
    pub const SYSTEM_GESTURE_GET_TOUCH_EVENTS_COUNT: u32 = 0x13AD_2218;
    pub const SYSTEM_GESTURE_GET_TOUCH_EVENT_BY_INDEX: u32 = 0x7472_4147;
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
    // SceNpTrophy: a title's trophy set is its OWN shipped data (`sce_sys/trophy/
    // <NPCOMMID>/TROPHY.TRP`), so every query reports it faithfully; only the unlock
    // ledger is console state, and off-console it starts empty and grows during the run.
    pub const NP_TROPHY_INIT: u32 = 0x3451_6838;
    pub const NP_TROPHY_TERM: u32 = 0xBFE0_F28F;
    pub const NP_TROPHY_CREATE_CONTEXT: u32 = 0xC49F_D33F;
    pub const NP_TROPHY_DESTROY_CONTEXT: u32 = 0x56F5_CBA5;
    pub const NP_TROPHY_CREATE_HANDLE: u32 = 0x4EBC_6977;
    pub const NP_TROPHY_DESTROY_HANDLE: u32 = 0xFF14_2071;
    pub const NP_TROPHY_ABORT_HANDLE: u32 = 0xD55C_6F4C;
    pub const NP_TROPHY_GET_GAME_INFO: u32 = 0xBA2B_7F2A;
    pub const NP_TROPHY_GET_GAME_ICON: u32 = 0xFE38_2529;
    pub const NP_TROPHY_GET_GROUP_INFO: u32 = 0x087B_0535;
    pub const NP_TROPHY_GET_GROUP_ICON: u32 = 0x1B8C_3192;
    pub const NP_TROPHY_GET_TROPHY_INFO: u32 = 0xA4AD_DD91;
    pub const NP_TROPHY_GET_TROPHY_ICON: u32 = 0x94BA_B8D0;
    pub const NP_TROPHY_GET_TROPHY_UNLOCK_STATE: u32 = 0xC8D2_A4DE;
    pub const NP_TROPHY_UNLOCK_TROPHY: u32 = 0xB397_AA24;
    // SceNp* subsystem inits with no backing service off-console: succeed so the
    // title proceeds (SceNpActivity/SceNpCommon-auth/SceNpUtility-lookup/SceNpTus).
    pub const NP_ACTIVITY_INIT: u32 = 0xE0FF_EE97;
    pub const NP_AUTH_INIT: u32 = 0x441D_8B4E;
    pub const NP_LOOKUP_INIT: u32 = 0x9246_A673;
    pub const NP_TUS_INIT: u32 = 0xB214_1F8D;
    // SceNpMessage: in-game messaging subsystem init (succeeds offline; no messages then).
    pub const NP_MESSAGE_INIT_WITH_PARAM: u32 = 0x26AF_5306;
    // SceNpMessage: subsystem teardown (the title tears messaging down after its offline
    // sync fails); a cleanup call that just succeeds.
    pub const NP_MESSAGE_TERM: u32 = 0x3802_30A1;
    // Subsystem TEARDOWN across the online stack. A title that finds itself offline
    // unwinds everything it brought up, so these arrive in a burst. Terminating a
    // subsystem that has no backing service genuinely succeeds - there is nothing to
    // fail - and the same goes for unregistering a callback that never fired and for
    // deleting a title context that only ever held local bookkeeping.
    pub const NP_TERM: u32 = 0x19E4_0AE1;
    pub const NP_UNREGISTER_SERVICE_STATE_CALLBACK: u32 = 0xD9E6_E56C;
    pub const NP_BASIC_TERM: u32 = 0x389B_CB3B;
    pub const NP_ACTIVITY_TERM: u32 = 0x9EA4_901F;
    pub const NP_AUTH_TERM: u32 = 0x6093_B689;
    pub const NP_LOOKUP_TERM: u32 = 0x0158_B61B;
    pub const NP_LOOKUP_DELETE_TITLE_CTX: u32 = 0x33B6_4699;
    pub const NP_TUS_TERM: u32 = 0x7EDC_33B3;
    pub const NP_TUS_DELETE_TITLE_CTX: u32 = 0xD53D_3692;
    pub const NP_SCORE_DELETE_TITLE_CTX: u32 = 0xF52E_A88A;
    pub const NP_MATCHING2_INIT: u32 = 0xEBB1_FE74;
    pub const NP_MATCHING2_TERM: u32 = 0x0124_641C;
    pub const HTTP_TERM: u32 = 0xC907_6666;
    pub const SSL_TERM: u32 = 0x03CE_6E3A;
    pub const NET_TERM: u32 = 0xEA3C_C286;
    pub const NET_CTL_TERM: u32 = 0xCD18_8648;
    pub const NET_CTL_INET_UNREGISTER_CALLBACK: u32 = 0xD0C3_BF3F;
    pub const NETCTL_ADHOC_UNREGISTER_CALLBACK: u32 = 0xA447_1E10;
    pub const SYSMODULE_UNLOAD_MODULE: u32 = 0x31D8_7805;
    pub const APPUTIL_SHUTDOWN: u32 = 0xB220_B00B;
    // SceNpCommerce2: PS Store commerce subsystem init (succeeds offline; no store then).
    pub const NP_COMMERCE2_INIT: u32 = 0xC73F_209A;
    // SceNpSnsFacebook: social-network integration; the library init succeeds offline
    // (no online SNS features are then available, and the title stays on its offline path).
    pub const NP_SNS_FACEBOOK_INIT: u32 = 0x8055_7AA0;
    // SceLibLocation / SceMotion / SceNetCtl(adhoc) / ScePower: device-service inits
    // and config that succeed with a neutral/offline result.
    pub const LOCATION_INIT: u32 = 0x09C4_F674;
    pub const MOTION_START_SAMPLING: u32 = 0x2803_4AC9;
    pub const NETCTL_ADHOC_REGISTER_CALLBACK: u32 = 0xFFA9_D594;
    pub const NETCTL_ADHOC_GET_IN_ADDR: u32 = 0x7118_C99D;
    pub const NETCTL_ADHOC_GET_STATE: u32 = 0x0961_A561;
    /// `sceNetCtlAdhocGetPeerList`: EMPTY offline, which is a real console state.
    pub const NETCTL_ADHOC_GET_PEER_LIST: u32 = 0x7758_6C59;
    /// `sceMotionMagnetometerOn`: enable magnetometer sampling. Accepted like
    /// `MOTION_START_SAMPLING` - see the group it dispatches with.
    pub const MOTION_MAGNETOMETER_ON: u32 = 0x122A_79F8;
    pub const NETCTL_ADHOC_DISCONNECT: u32 = 0xED43_B79A;
    pub const POWER_SET_CONFIGURATION_MODE: u32 = 0x3CE1_87B6;
    // SceCommonDialog: shared config for the dialog families, plus the trophy-setup
    // dialog's result read.
    pub const COMMON_DIALOG_SET_CONFIG_PARAM: u32 = 0xBECD_35C8;
    pub const NP_TROPHY_SETUP_DIALOG_GET_RESULT: u32 = 0xE370_69D5;
    /// SceMp4: opening a movie container. Names and NIDs are facts from the henkaku
    /// wiki's `SceMp4` page (the vitasdk NID db has no entry for this library).
    pub const MP4_OPEN_FILE: u32 = 0x0547_4AF0;
    /// SceMp4: begin streaming the opened file. A title that ignores a failed
    /// [`MP4_OPEN_FILE`] reaches this anyway, so it needs its own honest failure.
    pub const MP4_START_FILE_STREAMING: u32 = 0x30E4_9E4D;
    /// SceMp4: release the streaming session opened above.
    pub const MP4_CLOSE_FILE: u32 = 0x9206_23C8;
    /// SceMp4, unnamed on the henkaku wiki's 3.60 NID list. Its ROLE is recovered from
    /// the one call site: the title's movie teardown calls
    /// `sceMp4CloseFile(handle)` and then this, as `f(handle, &unit)` over the same unit
    /// struct its `GetNextUnit` family fills, before freeing its own two buffers. That is
    /// a buffer release - the 0.945 name list has a `sceMp4ReleaseBuffer` with no 3.60 NID
    /// beside it, which fits, but the mapping is inference so the constant is named by NID.
    pub const MP4_RELEASE_BUFFER_7B4832FE: u32 = 0x7B48_32FE;
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
    pub const DELETE_THREAD: u32 = 0x1BBD_E3D9;
    pub const EXIT_THREAD: u32 = 0x0C8A_38E1;
    pub const GET_PROCESS_ID: u32 = 0x9DCB_4B7A;
    pub const GET_THREAD_CURRENT_PRIORITY: u32 = 0x0141_4F0B;
    /// CPU affinity: RECORDED, not obeyed - this scheduler interleaves on one baton, so
    /// there is no placement to honour. Kept so the getter agrees with the setter and with
    /// `sceKernelGetThreadInfo`. See `vita::threadmgr`.
    pub const CHANGE_THREAD_CPU_AFFINITY_MASK: u32 = 0x1512_9174;
    pub const GET_THREAD_CPU_AFFINITY_MASK: u32 = 0xF1AE_5654;
    /// `sceKernelCloseSema`: releases a semaphore id (same effect as DeleteSema in
    /// this model - the id becomes invalid). Routed to the shared delete handler.
    pub const CLOSE_SEMA: u32 = 0xA2D8_1F9E;
    /// `sceKernelChangeThreadVfpException`: sets which VFP/NEON floating-point
    /// exceptions trap for the calling thread. We compute IEEE arithmetic without
    /// trapping, so this only records intent; it never changes numeric results.
    pub const CHANGE_THREAD_VFP_EXCEPTION: u32 = 0xCC18_FBAE;
    pub const CHANGE_THREAD_PRIORITY: u32 = 0xBD01_39F2;
    /// `sceKernelDelayThreadCB`: the same timed sleep as `DELAY_THREAD`, at which the
    /// kernel also delivers the calling thread's pending callbacks.
    pub const DELAY_THREAD_CB: u32 = 0x9C01_80E1;
    pub const SEND_SIGNAL: u32 = 0xD4C3_67B2;
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
    /// `scePvfOpenUserMemory`: the same open from bytes the title already holds, which a
    /// path-based open cannot reach (a font unpacked from the title's own archive).
    pub const OPEN_USER_MEMORY: u32 = 0x9E65_E4ED;
    /// `scePvfClose`: drop a font handle and its cached glyphs.
    pub const CLOSE: u32 = 0xD282_C23C;
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
    pub const IO_SYNC_BY_FD: u32 = 0x1651_2F59;
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
    /// `sceKernelCloseMutex`: release this thread's REFERENCE to a mutex, as opposed to
    /// `DELETE_MUTEX`'s destruction of the object. This engine's mutexes are lightweight
    /// handles with no per-thread reference count to drop, so both are the same no-op
    /// teardown - and a title that only ever closes (never deletes) must not hard-fail.
    pub const CLOSE_MUTEX: u32 = 0x03E2_3AF6;
    pub const CREATE_SEMA: u32 = 0x1BD6_7366;
    /// `sceKernelCreateSema_16XX`: the pre-3.60 firmware NID for the same call the
    /// SDK later re-exported as `CREATE_SEMA`. Titles built against an older SDK
    /// import this one; it dispatches to the identical handler.
    pub const CREATE_SEMA_16XX: u32 = 0x297A_A2AE;
    /// `sceKernelOpenSema`: resolve an EXISTING openable semaphore by name. Not a
    /// create - two modules share one semaphore this way.
    pub const OPEN_SEMA: u32 = 0xCBE2_35C7;
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
    /// `sceKernelPollEventFlag`: the non-blocking `WAIT_EVENT_FLAG`. Reports
    /// `SCE_KERNEL_ERROR_EVF_COND` instead of parking when the pattern does not satisfy.
    pub const POLL_EVENT_FLAG: u32 = 0x1FBB_0FE1;
    pub const CLEAR_EVENT_FLAG: u32 = 0x4CB8_7CA7;
    pub const DELETE_EVENT_FLAG: u32 = 0x5840_162C;

    // --- SIMPLE EVENTS: the NIDs are known, the SIGNATURES are not, and that is why
    // none of these is dispatched. ---
    //
    // A simple event is the kernel's other bit-pattern primitive: an object holding a
    // pattern that `SET_EVENT` ORs into and `WAIT_EVENT` blocks on. It looks like the
    // event FLAG with a smaller API, and it was implemented that way and then REVERTED,
    // because the argument positions are a guess:
    //
    // * No allowed source on this machine publishes a prototype. The vitasdk headers have
    //   no `SimpleEvent` anywhere; the NID db carries names and NIDs and nothing else.
    // * A Vita event carries 64-bit USER DATA that `sceKernelSetEvent` sets and the wait
    //   reports, which - if real - puts a `pUserData` pointer where the obvious reading
    //   puts the TIMEOUT. Reading a timeout from a pointer that is actually user data
    //   parks a thread on a nonsense deadline: a HANG, which is the one failure mode this
    //   runtime must never produce silently.
    // * Nothing exercises it. One title imports all four and never calls one - an
    //   unimplemented NID is FATAL, and the title drives a full race - so there is no
    //   evidence to check a guess against and no benefit to offset the risk.
    //
    // **Leaving them unhandled is the fix, not the omission.** The unimplemented-NID path
    // is fatal AND dumps r0-r3, the first stack words and what each pointer-looking
    // argument points at - which is precisely the evidence a signature needs. The first
    // title that really calls one will print its own prototype.
    pub const CREATE_SIMPLE_EVENT: u32 = 0xE6DB_2494;
    pub const DELETE_SIMPLE_EVENT: u32 = 0x208C_FE28;
    pub const OPEN_SIMPLE_EVENT: u32 = 0x4E1E_4DF8;
    pub const CLOSE_SIMPLE_EVENT: u32 = 0xFEF4_CA53;
    pub const SET_EVENT: u32 = 0x3242_18CD;
    pub const WAIT_EVENT: u32 = 0x120F_03AF;
    pub const WAIT_EVENT_CB: u32 = 0xA049_0795;
    pub const POLL_EVENT: u32 = 0x241F_3634;
    pub const CANCEL_EVENT: u32 = 0x603A_B770;

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
    pub const VOICE_INIT: u32 = 0x1DDB_EBEB;
    pub const VOICE_GET_INFO: u32 = 0x5551_410D;
    pub const RACK_RELEASE: u32 = 0xDD5C_A10B;
    pub const VOICE_DEF_GET_COMPRESSOR_BUSS: u32 = 0x0E0A_CB68;
    pub const VOICE_DEF_GET_DELAY_BUSS: u32 = 0x4D70_5E3E;
    pub const VOICE_DEF_GET_DISTORTION_BUSS: u32 = 0xAAD9_0DEB;
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
    pub const VOICE_DEF_GET_COMPRESSOR_SIDE_CHAIN_BUSS: u32 = 0x1AF8_3512;
    pub const VOICE_DEF_GET_SCREAM_ATRAC9_VOICE: u32 = 0xCD63_A2BF;
    pub const VOICE_DEF_GET_SCREAM_VOICE: u32 = 0xCE53_BC33;
    pub const VOICE_SET_PARAMS_BLOCK: u32 = 0xFB81_74B1;
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

/// What [`name`] returns for a NID it does not know. A NID gets its name in the same
/// change that gives it a handler, so link-time import coverage is checked against this
/// (see `link`), which turns "one missing NID revealed per boot" into one list.
pub const UNKNOWN_NAME: &str = "<unknown>";

/// A human-readable name for a `(library_nid, func_nid)` pair, for logging and
/// the unimplemented-call report. Falls back to the raw NIDs.
pub fn name(func_nid: u32) -> &'static str {
    use {
        audio as au, ctrl as c, display as d, fiber as fb, gxm as g, iofilemgr as io,
        libkernel as lk, lwsync as lw, net as nt, ngs as ng, processmgr as pm, pvf as pv, services as sv,
        sync as sy, sysmem as s, threadmgr as tm,
    };
    match func_nid {
        // SceNet, offline.
        nt::SOCKET => "sceNetSocket",
        nt::SOCKET_CLOSE => "sceNetSocketClose",
        nt::BIND => "sceNetBind",
        nt::LISTEN => "sceNetListen",
        nt::ACCEPT => "sceNetAccept",
        nt::CONNECT => "sceNetConnect",
        nt::SEND => "sceNetSend",
        nt::SENDTO => "sceNetSendto",
        nt::SENDMSG => "sceNetSendmsg",
        nt::RECV => "sceNetRecv",
        nt::RECVFROM => "sceNetRecvfrom",
        nt::SHUTDOWN => "sceNetShutdown",
        nt::GETSOCKNAME => "sceNetGetsockname",
        nt::GETPEERNAME => "sceNetGetpeername",
        nt::SETSOCKOPT => "sceNetSetsockopt",
        nt::GETSOCKOPT => "sceNetGetsockopt",
        nt::GET_SOCK_INFO => "sceNetGetSockInfo",
        nt::SHOW_NETSTAT => "sceNetShowNetstat",
        nt::HTONL => "sceNetHtonl",
        nt::NTOHL => "sceNetNtohl",
        nt::HTONS => "sceNetHtons",
        nt::NTOHS => "sceNetNtohs",
        nt::INET_PTON => "sceNetInetPton",
        nt::INET_NTOP => "sceNetInetNtop",
        nt::ERRNO_LOC => "sceNetErrnoLoc",
        nt::RESOLVER_CREATE => "sceNetResolverCreate",
        nt::RESOLVER_DESTROY => "sceNetResolverDestroy",
        nt::RESOLVER_START_NTOA => "sceNetResolverStartNtoa",
        nt::RESOLVER_START_ATON => "sceNetResolverStartAton",
        nt::RESOLVER_GET_ERROR => "sceNetResolverGetError",
        nt::EPOLL_CREATE => "sceNetEpollCreate",
        nt::EPOLL_DESTROY => "sceNetEpollDestroy",
        nt::EPOLL_CONTROL => "sceNetEpollControl",
        nt::EPOLL_WAIT => "sceNetEpollWait",
        // SceFiber.
        fb::INITIALIZE_IMPL => "_sceFiberInitializeImpl",
        fb::INITIALIZE_WITH_INTERNAL_OPTION_IMPL => "_sceFiberInitializeWithInternalOptionImpl",
        fb::ATTACH_CONTEXT_AND_SWITCH => "_sceFiberAttachContextAndSwitch",
        fb::FINALIZE => "sceFiberFinalize",
        fb::GET_INFO => "sceFiberGetInfo",
        fb::GET_SELF => "sceFiberGetSelf",
        fb::RETURN_TO_THREAD => "sceFiberReturnToThread",
        fb::RUN => "sceFiberRun",
        fb::SWITCH => "sceFiberSwitch",
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
        ng::VOICE_INIT => "sceNgsVoiceInit",
        ng::VOICE_GET_INFO => "sceNgsVoiceGetInfo",
        ng::RACK_RELEASE => "sceNgsRackRelease",
        ng::VOICE_DEF_GET_COMPRESSOR_BUSS => "sceNgsVoiceDefGetCompressorBuss",
        ng::VOICE_DEF_GET_DELAY_BUSS => "sceNgsVoiceDefGetDelayBuss",
        ng::VOICE_DEF_GET_DISTORTION_BUSS => "sceNgsVoiceDefGetDistortionBuss",
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
        g::COLOR_SURFACE_INIT_DISABLED => "sceGxmColorSurfaceInitDisabled",
        g::DEPTH_STENCIL_SURFACE_INIT => "sceGxmDepthStencilSurfaceInit",
        g::SYNC_OBJECT_CREATE => "sceGxmSyncObjectCreate",
        g::SYNC_OBJECT_DESTROY => "sceGxmSyncObjectDestroy",
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
        g::DRAW_INSTANCED => "sceGxmDrawInstanced",
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
        g::SET_VERTEX_UNIFORM_BUFFER => "sceGxmSetVertexUniformBuffer",
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
        lk::CLIB_STRRCHR => "sceClibStrrchr",
        lk::CLIB_STRNCASECMP => "sceClibStrncasecmp",
        lk::CLIB_MSPACE_CREATE => "sceClibMspaceCreate",
        lk::CLIB_MSPACE_DESTROY => "sceClibMspaceDestroy",
        lk::CLIB_MSPACE_MALLOC => "sceClibMspaceMalloc",
        lk::CLIB_MSPACE_FREE => "sceClibMspaceFree",
        lk::CLIB_MSPACE_MEMALIGN => "sceClibMspaceMemalign",
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
        sv::NET_CTL_INET_GET_INFO => "sceNetCtlInetGetInfo",
        sv::NET_CTL_INET_REGISTER_CALLBACK => "sceNetCtlInetRegisterCallback",
        sv::HTTP_INIT => "sceHttpInit",
        sv::SSL_INIT => "sceSslInit",
        sv::NP_INIT => "sceNpInit",
        sv::NP_REGISTER_SERVICE_STATE_CALLBACK => "sceNpRegisterServiceStateCallback",
        sv::NP_BASIC_INIT => "sceNpBasicInit",
        sv::NP_BASIC_REGISTER_HANDLER => "sceNpBasicRegisterHandler",
        sv::NP_BASIC_CHECK_CALLBACK => "sceNpBasicCheckCallback",
        sv::NP_BASIC_GET_FRIEND_LIST_ENTRY_COUNT => "sceNpBasicGetFriendListEntryCount",
        sv::RTC_GET_CURRENT_CLOCK => "sceRtcGetCurrentClock",
        sv::RTC_GET_CURRENT_TICK => "sceRtcGetCurrentTick",
        sv::RTC_SET_TIME64_T => "sceRtcSetTime64_t",
        sv::APPUTIL_SAVEDATA_DATA_REMOVE => "sceAppUtilSaveDataDataRemove",
        sv::APPUTIL_SAVEDATA_GET_QUOTA => "sceAppUtilSaveDataGetQuota",
        sv::APPUTIL_RECEIVE_APP_EVENT => "sceAppUtilReceiveAppEvent",
        sv::APPUTIL_APP_EVENT_PARSE_NEAR_GIFT => "sceAppUtilAppEventParseNearGift",
        sv::APPUTIL_APP_EVENT_PARSE_NP_BASIC_JOINABLE_PRESENCE => "sceAppUtilAppEventParseNpBasicJoinablePresence",
        sv::APPUTIL_APP_EVENT_PARSE_NP_INVITE_MESSAGE => "sceAppUtilAppEventParseNpInviteMessage",
        sv::NET_CTL_INET_GET_RESULT => "sceNetCtlInetGetResult",
        sv::NET_CTL_ADHOC_GET_RESULT => "sceNetCtlAdhocGetResult",
        sv::APPMGR_LOAD_EXEC => "sceAppMgrLoadExec",
        sv::APPMGR_RECEIVE_SYSTEM_EVENT => "sceAppMgrReceiveSystemEvent",
        sv::MOTION_SET_DEADBAND => "sceMotionSetDeadband",
        sv::MOTION_SET_TILT_CORRECTION => "sceMotionSetTiltCorrection",
        sv::SHUTTER_SOUND_PLAY => "sceShutterSoundPlay",
        sv::PHOTO_EXPORT_FROM_DATA => "scePhotoExportFromData",
        sv::NP_GET_SERVICE_STATE => "sceNpGetServiceState",
        sv::NP_ACTIVITY_POST_STATUS => "sceNpActivityPostStatus",
        sv::NP_BASIC_UNREGISTER_HANDLER => "sceNpBasicUnregisterHandler",
        sv::NP_BASIC_SET_IN_GAME_PRESENCE => "sceNpBasicSetInGamePresence",
        sv::NP_BASIC_GET_GAME_JOINING_PRESENCE => "sceNpBasicGetGameJoiningPresence",
        sv::NP_BASIC_GET_FRIEND_LIST_ENTRIES => "sceNpBasicGetFriendListEntries",
        sv::NP_LOOKUP_CREATE_TITLE_CTX => "sceNpLookupCreateTitleCtx",
        sv::NP_LOOKUP_DELETE_REQUEST => "sceNpLookupDeleteRequest",
        sv::NP_LOOKUP_USER_PROFILE_ASYNC => "sceNpLookupUserProfileAsync",
        sv::NP_LOOKUP_POLL_ASYNC => "sceNpLookupPollAsync",
        sv::NP_AUTH_CREATE_START_REQUEST => "sceNpAuthCreateStartRequest",
        sv::NP_AUTH_DESTROY_REQUEST => "sceNpAuthDestroyRequest",
        sv::NP_AUTH_GET_TICKET => "sceNpAuthGetTicket",
        sv::NP_AUTH_GET_TICKET_PARAM => "sceNpAuthGetTicketParam",
        sv::NP_AUTH_GET_ENTITLEMENT_BY_ID => "sceNpAuthGetEntitlementById",
        sv::NP_AUTH_GET_ENTITLEMENT_ID_LIST => "sceNpAuthGetEntitlementIdList",
        s::FIND_MEM_BLOCK_BY_ADDR => "sceKernelFindMemBlockByAddr",
        sv::MOTION_GET_STATE => "sceMotionGetState",
        sv::RTC_GET_CURRENT_CLOCK_LOCAL_TIME => "sceRtcGetCurrentClockLocalTime",
        sv::RTC_GET_TICK => "sceRtcGetTick",
        sv::RTC_GET_TIME64_T => "sceRtcGetTime64_t",
        sv::RTC_GET_TIME_T => "sceRtcGetTime_t",
        sv::RTC_GET_CURRENT_NETWORK_TICK => "sceRtcGetCurrentNetworkTick",
        sv::RTC_SET_TICK => "sceRtcSetTick",
        sv::RTC_CONVERT_UTC_TO_LOCAL_TIME => "sceRtcConvertUtcToLocalTime",
        sv::RTC_CONVERT_LOCAL_TIME_TO_UTC => "sceRtcConvertLocalTimeToUtc",
        sv::RTC_TICK_ADD_TICKS => "sceRtcTickAddTicks",
        sv::RTC_TICK_ADD_MICROSECONDS => "sceRtcTickAddMicroseconds",
        sv::RTC_TICK_ADD_SECONDS => "sceRtcTickAddSeconds",
        sv::RTC_TICK_ADD_MINUTES => "sceRtcTickAddMinutes",
        sv::RTC_TICK_ADD_HOURS => "sceRtcTickAddHours",
        sv::RTC_TICK_ADD_DAYS => "sceRtcTickAddDays",
        sv::RTC_TICK_ADD_WEEKS => "sceRtcTickAddWeeks",
        sv::RTC_TICK_ADD_MONTHS => "sceRtcTickAddMonths",
        sv::RTC_TICK_ADD_YEARS => "sceRtcTickAddYears",
        sv::FIOS_OVERLAY_GET_LIST => "sceFiosOverlayGetList02",
        sv::FIOS_OVERLAY_THREAD_SET_DISABLED => "sceFiosOverlayThreadSetDisabled02",
        sv::FIOS_OVERLAY_GET_RECOMMENDED_SCHEDULER => "sceFiosOverlayGetRecommendedScheduler02",
        sv::ULOBJ_REGISTER_PROTOCOL_REVISION => "_sceUlobjMgrRegisterLibultProtocolRevision",
        sv::APPUTIL_INIT => "sceAppUtilInit",
        sv::APPUTIL_SYSTEM_PARAM_GET_INT => "sceAppUtilSystemParamGetInt",
        sv::APPUTIL_APP_PARAM_GET_INT => "sceAppUtilAppParamGetInt",
        sv::LIVE_AREA_GET_STATUS => "sceLiveAreaGetStatus",
        sv::LIVE_AREA_UPDATE_FRAME_ASYNC => "sceLiveAreaUpdateFrameAsync",
        sv::NP_SCORE_INIT => "sceNpScoreInit",
        sv::NP_SCORE_TERM => "sceNpScoreTerm",
        sv::NP_SCORE_CREATE_TITLE_CTX => "sceNpScoreCreateTitleCtx",
        sv::NP_MANAGER_GET_NP_ID => "sceNpManagerGetNpId",
        sv::NP_MANAGER_GET_ACCOUNT_REGION => "sceNpManagerGetAccountRegion",
        sv::NP_MANAGER_GET_CONTENT_RATING_FLAG => "sceNpManagerGetContentRatingFlag",
        sv::NP_MANAGER_GET_CHAT_RESTRICTION_FLAG => "sceNpManagerGetChatRestrictionFlag",
        sv::NP_LOOKUP_CREATE_REQUEST => "sceNpLookupCreateRequest",
        sv::NP_MESSAGE_SYNC_MESSAGE => "sceNpMessageSyncMessage",
        sv::NP_TUS_CREATE_REQUEST => "sceNpTusCreateRequest",
        sv::NP_COMMERCE2_START_EMPTY_STORE_CHECK => "sceNpCommerce2StartEmptyStoreCheck",
        sv::NP_COMMERCE2_CREATE_SESSION_GET_RESULT => "sceNpCommerce2CreateSessionGetResult",
        sv::NP_COMMERCE2_CREATE_CTX => "sceNpCommerce2CreateCtx",
        sv::NP_COMMERCE2_CREATE_SESSION_CREATE_REQ => "sceNpCommerce2CreateSessionCreateReq",
        sv::NP_COMMERCE2_CREATE_SESSION_START => "sceNpCommerce2CreateSessionStart",
        sv::APP_MGR_GET_APP_STATE => "_sceAppMgrGetAppState",
        sv::APP_MGR_IS_GAME_PROGRAM => "sceAppMgrIsGameProgram",
        sv::NET_CTL_CHECK_CALLBACK => "sceNetCtlCheckCallback",
        sv::APPUTIL_DRM_OPEN => "sceAppUtilDrmOpen",
        sv::APPUTIL_DRM_CLOSE => "sceAppUtilDrmClose",
        sv::APPUTIL_SAVEDATA_SLOT_GET_PARAM => "sceAppUtilSaveDataSlotGetParam",
        sv::APPUTIL_SAVEDATA_SLOT_CREATE => "sceAppUtilSaveDataSlotCreate",
        sv::APPUTIL_SAVEDATA_SLOT_SET_PARAM => "sceAppUtilSaveDataSlotSetParam",
        sv::APPUTIL_SAVEDATA_SLOT_DELETE => "sceAppUtilSaveDataSlotDelete",
        sv::APPUTIL_SAVEDATA_DATA_SAVE => "sceAppUtilSaveDataDataSave",
        sv::NP_CHECK_CALLBACK => "sceNpCheckCallback",
        sv::TOUCH_SET_SAMPLING_STATE => "sceTouchSetSamplingState",
        sv::TOUCH_READ => "sceTouchRead",
        sv::TOUCH_PEEK => "sceTouchPeek",
        sv::TOUCH_GET_PANEL_INFO => "sceTouchGetPanelInfo",
        sv::CAMERA_OPEN => "sceCameraOpen",
        sv::CAMERA_CLOSE => "sceCameraClose",
        sv::CAMERA_START => "sceCameraStart",
        sv::CAMERA_STOP => "sceCameraStop",
        sv::CAMERA_READ => "sceCameraRead",
        sv::CAMERA_GET_REVERSE => "sceCameraGetReverse",
        sv::CAMERA_SET_REVERSE => "sceCameraSetReverse",
        sv::CAMERA_SET_BACKLIGHT => "sceCameraSetBacklight",
        sv::CAMERA_SET_WHITE_BALANCE => "sceCameraSetWhiteBalance",
        sv::LOCATION_OPEN => "sceLocationOpen",
        sv::LOCATION_CLOSE => "sceLocationClose",
        sv::LOCATION_REOPEN => "sceLocationReopen",
        sv::LOCATION_GET_METHOD => "sceLocationGetMethod",
        sv::LOCATION_CONFIRM => "sceLocationConfirm",
        sv::LOCATION_CONFIRM_GET_STATUS => "sceLocationConfirmGetStatus",
        sv::LOCATION_CONFIRM_GET_RESULT => "sceLocationConfirmGetResult",
        sv::LOCATION_CONFIRM_ABORT => "sceLocationConfirmAbort",
        sv::LOCATION_GET_LOCATION => "sceLocationGetLocation",
        sv::LOCATION_GET_LOCATION_WITH_TIMEOUT => "sceLocationGetLocationWithTimeout",
        sv::LOCATION_CANCEL_GET_LOCATION => "sceLocationCancelGetLocation",
        sv::LOCATION_GET_HEADING => "sceLocationGetHeading",
        sv::LOCATION_GET_PERMISSION => "sceLocationGetPermission",
        sv::LOCATION_DENY_APPLICATION => "sceLocationDenyApplication",
        sv::LOCATION_TERM => "sceLocationTerm",
        sv::LOCATION_SET_THREAD_PARAMETER => "sceLocationSetThreadParameter",
        sv::JPEGENC_GET_CONTEXT_SIZE => "sceJpegEncoderGetContextSize",
        sv::JPEGENC_INIT => "sceJpegEncoderInit",
        sv::JPEGENC_END => "sceJpegEncoderEnd",
        sv::JPEGENC_SET_OUTPUT_ADDR => "sceJpegEncoderSetOutputAddr",
        sv::JPEGENC_SET_COMPRESSION_RATIO => "sceJpegEncoderSetCompressionRatio",
        sv::JPEGENC_SET_VALID_REGION => "sceJpegEncoderSetValidRegion",
        sv::JPEG_INIT_MJPEG => "sceJpegInitMJpeg",
        sv::JPEG_FINISH_MJPEG => "sceJpegFinishMJpeg",
        sv::SYSTEM_GESTURE_INIT_PRIMITIVE_TOUCH_RECOGNIZER => {
            "sceSystemGestureInitializePrimitiveTouchRecognizer"
        }
        sv::SYSTEM_GESTURE_UPDATE_PRIMITIVE_TOUCH_RECOGNIZER => {
            "sceSystemGestureUpdatePrimitiveTouchRecognizer"
        }
        sv::SYSTEM_GESTURE_CREATE_TOUCH_RECOGNIZER => "sceSystemGestureCreateTouchRecognizer",
        sv::SYSTEM_GESTURE_UPDATE_TOUCH_RECOGNIZER => "sceSystemGestureUpdateTouchRecognizer",
        sv::SYSTEM_GESTURE_GET_TOUCH_EVENTS_COUNT => "sceSystemGestureGetTouchEventsCount",
        sv::SYSTEM_GESTURE_GET_TOUCH_EVENT_BY_INDEX => "sceSystemGestureGetTouchEventByIndex",
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
        tm::DELETE_THREAD => "sceKernelDeleteThread",
        tm::EXIT_THREAD => "sceKernelExitThread",
        sy::CREATE_MUTEX => "sceKernelCreateMutex",
        sy::LOCK_MUTEX => "sceKernelLockMutex",
        sy::TRY_LOCK_MUTEX => "sceKernelTryLockMutex",
        sy::UNLOCK_MUTEX => "sceKernelUnlockMutex",
        sy::DELETE_MUTEX => "sceKernelDeleteMutex",
        sy::CLOSE_MUTEX => "sceKernelCloseMutex",
        sy::CREATE_SEMA | sy::CREATE_SEMA_16XX => "sceKernelCreateSema",
        sy::WAIT_SEMA => "sceKernelWaitSema",
        sy::SIGNAL_SEMA => "sceKernelSignalSema",
        sy::DELETE_SEMA => "sceKernelDeleteSema",
        sy::CREATE_EVENT_FLAG => "sceKernelCreateEventFlag",
        sy::SET_EVENT_FLAG => "sceKernelSetEventFlag",
        sy::WAIT_EVENT_FLAG => "sceKernelWaitEventFlag",
        sy::WAIT_EVENT_FLAG_CB => "sceKernelWaitEventFlagCB",
        sy::POLL_EVENT_FLAG => "sceKernelPollEventFlag",
        sy::CLEAR_EVENT_FLAG => "sceKernelClearEventFlag",
        sy::DELETE_EVENT_FLAG => "sceKernelDeleteEventFlag",
        sy::CREATE_SIMPLE_EVENT => "sceKernelCreateSimpleEvent",
        sy::DELETE_SIMPLE_EVENT => "sceKernelDeleteSimpleEvent",
        sy::OPEN_SIMPLE_EVENT => "sceKernelOpenSimpleEvent",
        sy::CLOSE_SIMPLE_EVENT => "sceKernelCloseSimpleEvent",
        sy::SET_EVENT => "sceKernelSetEvent",
        sy::WAIT_EVENT => "sceKernelWaitEvent",
        sy::WAIT_EVENT_CB => "sceKernelWaitEventCB",
        sy::POLL_EVENT => "sceKernelPollEvent",
        sy::CANCEL_EVENT => "sceKernelCancelEvent",
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
        g::SET_BACK_STENCIL_FUNC => "sceGxmSetBackStencilFunc",
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
        g::TEXTURE_GET_MIPMAP_COUNT => "sceGxmTextureGetMipmapCount",
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
        g::DEPTH_STENCIL_SURFACE_SET_BACKGROUND_DEPTH => "sceGxmDepthStencilSurfaceSetBackgroundDepth",
        g::DEPTH_STENCIL_SURFACE_SET_BACKGROUND_STENCIL => "sceGxmDepthStencilSurfaceSetBackgroundStencil",
        g::DEPTH_STENCIL_SURFACE_SET_FORCE_LOAD_MODE => "sceGxmDepthStencilSurfaceSetForceLoadMode",
        g::DEPTH_STENCIL_SURFACE_SET_FORCE_STORE_MODE => "sceGxmDepthStencilSurfaceSetForceStoreMode",
        g::SET_BACK_DEPTH_WRITE_ENABLE => "sceGxmSetBackDepthWriteEnable",
        g::SET_BACK_POLYGON_MODE => "sceGxmSetBackPolygonMode",
        g::SET_VISIBILITY_BUFFER => "sceGxmSetVisibilityBuffer",
        g::SET_FRONT_VISIBILITY_TEST_ENABLE => "sceGxmSetFrontVisibilityTestEnable",
        g::SET_FRONT_VISIBILITY_TEST_INDEX => "sceGxmSetFrontVisibilityTestIndex",
        g::SET_FRONT_VISIBILITY_TEST_OP => "sceGxmSetFrontVisibilityTestOp",
        g::UNMAP_MEMORY => "sceGxmUnmapMemory",
        g::UNMAP_VERTEX_USSE_MEMORY => "sceGxmUnmapVertexUsseMemory",
        g::UNMAP_FRAGMENT_USSE_MEMORY => "sceGxmUnmapFragmentUsseMemory",
        g::COLOR_SURFACE_GET_SCALE_MODE => "sceGxmColorSurfaceGetScaleMode",
        g::COLOR_SURFACE_SET_DATA => "sceGxmColorSurfaceSetData",
        g::PROGRAM_GET_TYPE => "sceGxmProgramGetType",
        g::PROGRAM_FIND_PARAMETER_BY_SEMANTIC => "_sceGxmProgramFindParameterBySemantic",
        g::RENDER_TARGET_GET_DRIVER_MEM_BLOCK => "sceGxmRenderTargetGetDriverMemBlock",
        g::NOTIFICATION_WAIT => "sceGxmNotificationWait",
        g::SET_VERTEX_TEXTURE => "_sceGxmSetVertexTexture",
        g::TEXTURE_INIT_CUBE_ARBITRARY => "sceGxmTextureInitCubeArbitrary",
        g::TEXTURE_SET_PALETTE => "sceGxmTextureSetPalette",
        g::PRECOMPUTED_DRAW_SET_ALL_VERTEX_STREAMS => "sceGxmPrecomputedDrawSetAllVertexStreams",
        g::PRECOMPUTED_FRAGMENT_STATE_SET_ALL_TEXTURES => "sceGxmPrecomputedFragmentStateSetAllTextures",
        g::PRECOMPUTED_VERTEX_STATE_SET_ALL_TEXTURES => "sceGxmPrecomputedVertexStateSetAllTextures",
        g::PRECOMPUTED_FRAGMENT_STATE_SET_ALL_UNIFORM_BUFFERS => "sceGxmPrecomputedFragmentStateSetAllUniformBuffers",
        g::PRECOMPUTED_FRAGMENT_STATE_SET_UNIFORM_BUFFER => "sceGxmPrecomputedFragmentStateSetUniformBuffer",
        g::PRECOMPUTED_VERTEX_STATE_SET_ALL_UNIFORM_BUFFERS => "sceGxmPrecomputedVertexStateSetAllUniformBuffers",
        g::PRECOMPUTED_VERTEX_STATE_SET_UNIFORM_BUFFER => "sceGxmPrecomputedVertexStateSetUniformBuffer",
        // NGS.
        ng::VOICE_DEF_GET_SIMPLE_VOICE => "sceNgsVoiceDefGetSimpleVoice",
        ng::VOICE_DEF_GET_MIXER_BUSS => "sceNgsVoiceDefGetMixerBuss",
        ng::VOICE_DEF_GET_COMPRESSOR_SIDE_CHAIN_BUSS => "sceNgsVoiceDefGetCompressorSideChainBuss",
        ng::VOICE_DEF_GET_SCREAM_ATRAC9_VOICE => "sceNgsVoiceDefGetScreamAtrac9Voice",
        ng::VOICE_DEF_GET_SCREAM_VOICE => "sceNgsVoiceDefGetScreamVoice",
        ng::VOICE_SET_PARAMS_BLOCK => "sceNgsVoiceSetParamsBlock",
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
        pv::OPEN_USER_MEMORY => "scePvfOpenUserMemory",
        pv::CLOSE => "scePvfClose",
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
        tm::CHANGE_THREAD_CPU_AFFINITY_MASK => "sceKernelChangeThreadCpuAffinityMask",
        tm::GET_THREAD_CPU_AFFINITY_MASK => "sceKernelGetThreadCpuAffinityMask",
        tm::CLOSE_SEMA => "sceKernelCloseSema",
        tm::CHANGE_THREAD_VFP_EXCEPTION => "sceKernelChangeThreadVfpException",
        s::FREE_MEM_BLOCK => "sceKernelFreeMemBlock",
        d::WAIT_VBLANK_START_MULTI => "sceDisplayWaitVblankStartMulti",
        d::WAIT_VBLANK_START => "sceDisplayWaitVblankStart",
        d::WAIT_SET_FRAME_BUF => "sceDisplayWaitSetFrameBuf",
        lk::UNKNOWN_023EAA62 => "SceLibKernel_023EAA62",
        // Offline services: screenshot, trophy, Np inits, location/motion/power/net.
        sv::SCREENSHOT_DISABLE => "sceScreenShotDisable",
        sv::SCREENSHOT_ENABLE => "sceScreenShotEnable",
        sv::SCREENSHOT_SET_PARAM => "sceScreenShotSetParam",
        sv::SCREENSHOT_SET_OVERLAY_IMAGE => "sceScreenShotSetOverlayImage",
        sv::NP_TROPHY_INIT => "sceNpTrophyInit",
        sv::NP_TROPHY_TERM => "sceNpTrophyTerm",
        sv::NP_TROPHY_CREATE_CONTEXT => "sceNpTrophyCreateContext",
        sv::NP_TROPHY_DESTROY_CONTEXT => "sceNpTrophyDestroyContext",
        sv::NP_TROPHY_CREATE_HANDLE => "sceNpTrophyCreateHandle",
        sv::NP_TROPHY_DESTROY_HANDLE => "sceNpTrophyDestroyHandle",
        sv::NP_TROPHY_ABORT_HANDLE => "sceNpTrophyAbortHandle",
        sv::NP_TROPHY_GET_GAME_INFO => "sceNpTrophyGetGameInfo",
        sv::NP_TROPHY_GET_GAME_ICON => "sceNpTrophyGetGameIcon",
        sv::NP_TROPHY_GET_GROUP_INFO => "sceNpTrophyGetGroupInfo",
        sv::NP_TROPHY_GET_GROUP_ICON => "sceNpTrophyGetGroupIcon",
        sv::NP_TROPHY_GET_TROPHY_INFO => "sceNpTrophyGetTrophyInfo",
        sv::NP_TROPHY_GET_TROPHY_ICON => "sceNpTrophyGetTrophyIcon",
        sv::NP_TROPHY_GET_TROPHY_UNLOCK_STATE => "sceNpTrophyGetTrophyUnlockState",
        sv::NP_TROPHY_UNLOCK_TROPHY => "sceNpTrophyUnlockTrophy",
        sv::NP_ACTIVITY_INIT => "sceNpActivityInit",
        sv::NP_AUTH_INIT => "sceNpAuthInit",
        sv::NP_LOOKUP_INIT => "sceNpLookupInit",
        sv::NP_TUS_INIT => "sceNpTusInit",
        sv::NP_MESSAGE_INIT_WITH_PARAM => "sceNpMessageInitWithParam",
        sv::NP_MESSAGE_TERM => "sceNpMessageTerm",
        sv::MP4_OPEN_FILE => "sceMp4OpenFile",
        sv::MP4_START_FILE_STREAMING => "sceMp4StartFileStreaming",
        sv::MP4_CLOSE_FILE => "sceMp4CloseFile",
        sv::MP4_RELEASE_BUFFER_7B4832FE => "sceMp4(unnamed 0x7b4832fe, buffer release)",
        sv::NP_TERM => "sceNpTerm",
        sv::NP_UNREGISTER_SERVICE_STATE_CALLBACK => "sceNpUnregisterServiceStateCallback",
        sv::NP_BASIC_TERM => "sceNpBasicTerm",
        sv::NP_ACTIVITY_TERM => "sceNpActivityTerm",
        sv::NP_AUTH_TERM => "sceNpAuthTerm",
        sv::NP_LOOKUP_TERM => "sceNpLookupTerm",
        sv::NP_LOOKUP_DELETE_TITLE_CTX => "sceNpLookupDeleteTitleCtx",
        sv::NP_TUS_TERM => "sceNpTusTerm",
        sv::NP_TUS_DELETE_TITLE_CTX => "sceNpTusDeleteTitleCtx",
        sv::NP_SCORE_DELETE_TITLE_CTX => "sceNpScoreDeleteTitleCtx",
        sv::NP_MATCHING2_INIT => "sceNpMatching2Init",
        sv::NP_MATCHING2_TERM => "sceNpMatching2Term",
        sv::HTTP_TERM => "sceHttpTerm",
        sv::SSL_TERM => "sceSslTerm",
        sv::NET_TERM => "sceNetTerm",
        sv::NET_CTL_TERM => "sceNetCtlTerm",
        sv::NET_CTL_INET_UNREGISTER_CALLBACK => "sceNetCtlInetUnregisterCallback",
        sv::NETCTL_ADHOC_UNREGISTER_CALLBACK => "sceNetCtlAdhocUnregisterCallback",
        sv::SYSMODULE_UNLOAD_MODULE => "sceSysmoduleUnloadModule",
        sv::APPUTIL_SHUTDOWN => "sceAppUtilShutdown",
        sv::NP_COMMERCE2_INIT => "sceNpCommerce2Init",
        sv::NP_SNS_FACEBOOK_INIT => "sceNpSnsFacebookInit",
        sv::LOCATION_INIT => "sceLocationInit",
        sv::MOTION_START_SAMPLING => "sceMotionStartSampling",
        sv::MOTION_MAGNETOMETER_ON => "sceMotionMagnetometerOn",
        sv::NETCTL_ADHOC_REGISTER_CALLBACK => "sceNetCtlAdhocRegisterCallback",
        sv::NETCTL_ADHOC_GET_IN_ADDR => "sceNetCtlAdhocGetInAddr",
        sv::NETCTL_ADHOC_GET_STATE => "sceNetCtlAdhocGetState",
        sv::NETCTL_ADHOC_GET_PEER_LIST => "sceNetCtlAdhocGetPeerList",
        sv::NETCTL_ADHOC_DISCONNECT => "sceNetCtlAdhocDisconnect",
        sv::POWER_SET_CONFIGURATION_MODE => "scePowerSetConfigurationMode",
        sv::COMMON_DIALOG_SET_CONFIG_PARAM => "sceCommonDialogSetConfigParam",
        // SceCommonDialog lifecycle. These all had dispatch arms already; without a name
        // each the link-time coverage report counted them as MISSING, which is how a
        // "22 unhandled dialog calls" list came to include calls that were implemented.
        sv::COMMON_DIALOG_UPDATE => "sceCommonDialogUpdate",
        sv::MSG_DIALOG_INIT => "sceMsgDialogInit",
        sv::MSG_DIALOG_GET_STATUS => "sceMsgDialogGetStatus",
        sv::MSG_DIALOG_GET_RESULT => "sceMsgDialogGetResult",
        sv::MSG_DIALOG_ABORT => "sceMsgDialogAbort",
        sv::MSG_DIALOG_TERM => "sceMsgDialogTerm",
        sv::NET_CHECK_DIALOG_INIT => "sceNetCheckDialogInit",
        sv::NET_CHECK_DIALOG_GET_STATUS => "sceNetCheckDialogGetStatus",
        sv::NET_CHECK_DIALOG_GET_RESULT => "sceNetCheckDialogGetResult",
        sv::NET_CHECK_DIALOG_TERM => "sceNetCheckDialogTerm",
        sv::SAVEDATA_DIALOG_INIT => "sceSaveDataDialogInit",
        sv::SAVEDATA_DIALOG_GET_STATUS => "sceSaveDataDialogGetStatus",
        sv::SAVEDATA_DIALOG_GET_SUB_STATUS => "sceSaveDataDialogGetSubStatus",
        sv::SAVEDATA_DIALOG_GET_RESULT => "sceSaveDataDialogGetResult",
        sv::SAVEDATA_DIALOG_CONTINUE => "sceSaveDataDialogContinue",
        sv::SAVEDATA_DIALOG_FINISH => "sceSaveDataDialogFinish",
        sv::SAVEDATA_DIALOG_SUB_CLOSE => "sceSaveDataDialogSubClose",
        sv::SAVEDATA_DIALOG_TERM => "sceSaveDataDialogTerm",
        sv::NP_MESSAGE_DIALOG_INIT => "sceNpMessageDialogInit",
        sv::NP_MESSAGE_DIALOG_GET_STATUS => "sceNpMessageDialogGetStatus",
        sv::NP_MESSAGE_DIALOG_GET_RESULT => "sceNpMessageDialogGetResult",
        sv::NP_MESSAGE_DIALOG_ABORT => "sceNpMessageDialogAbort",
        sv::NP_MESSAGE_DIALOG_TERM => "sceNpMessageDialogTerm",
        sv::NP_TROPHY_SETUP_DIALOG_INIT => "sceNpTrophySetupDialogInit",
        sv::NP_TROPHY_SETUP_DIALOG_GET_STATUS => "sceNpTrophySetupDialogGetStatus",
        sv::NP_TROPHY_SETUP_DIALOG_TERM => "sceNpTrophySetupDialogTerm",
        sv::STORE_CHECKOUT_DIALOG_INIT => "sceStoreCheckoutDialogInit",
        sv::STORE_CHECKOUT_DIALOG_GET_STATUS => "sceStoreCheckoutDialogGetStatus",
        sv::STORE_CHECKOUT_DIALOG_GET_RESULT => "sceStoreCheckoutDialogGetResult",
        sv::STORE_CHECKOUT_DIALOG_TERM => "sceStoreCheckoutDialogTerm",
        sv::NP_SNS_FACEBOOK_DIALOG_INIT => "sceNpSnsFacebookDialogInit",
        sv::NP_SNS_FACEBOOK_DIALOG_GET_STATUS => "sceNpSnsFacebookDialogGetStatus",
        sv::NP_SNS_FACEBOOK_DIALOG_GET_RESULT_LONG_TOKEN => "sceNpSnsFacebookDialogGetResultLongToken",
        sv::IME_DIALOG_INIT => "sceImeDialogInit",
        sv::IME_DIALOG_GET_STATUS => "sceImeDialogGetStatus",
        sv::IME_DIALOG_GET_RESULT => "sceImeDialogGetResult",
        sv::IME_DIALOG_ABORT => "sceImeDialogAbort",
        sv::IME_DIALOG_TERM => "sceImeDialogTerm",
        sv::NP_TROPHY_SETUP_DIALOG_GET_RESULT => "sceNpTrophySetupDialogGetResult",
        sv::NEAR_UTIL_UNKNOWN_A412E9CA => "SceNearUtil_A412E9CA",

        // --- kernel core -----------------------------------------------------
        s::SET_GPO => "sceKernelSetGPO",
        lk::GET_THREAD_INFO => "sceKernelGetThreadInfo",
        lk::GET_SEMA_INFO => "sceKernelGetSemaInfo",
        lk::WAIT_SIGNAL => "sceKernelWaitSignal",
        lk::WAIT_SEMA_CB => "sceKernelWaitSemaCB",
        lk::WAIT_THREAD_END_CB => "sceKernelWaitThreadEndCB",
        lk::GET_PROCESS_TIME_LOW => "sceKernelGetProcessTimeLow",
        lk::GET_OPEN_PS_ID => "sceKernelGetOpenPsId",
        lk::GET_MODULE_INFO_BY_ADDR => "sceKernelGetModuleInfoByAddr",
        lk::CALL_MODULE_EXIT => "sceKernelCallModuleExit",
        lk::AEABI_IDIV0 => "__sce_aeabi_idiv0",
        lk::AEABI_LDIV0 => "__sce_aeabi_ldiv0",
        lk::IO_CHSTAT => "sceIoChstat",
        lk::IO_DEVCTL => "sceIoDevctl",
        lk::IO_IOCTL => "sceIoIoctl",
        lk::IO_RENAME => "sceIoRename",
        lk::IO_RMDIR => "sceIoRmdir",
        lk::IO_SYNC => "sceIoSync",
        io::IO_SYNC_BY_FD => "sceIoSyncByFd",
        sy::OPEN_SEMA => "sceKernelOpenSema",
        tm::CHANGE_THREAD_PRIORITY => "sceKernelChangeThreadPriority",
        tm::DELAY_THREAD_CB => "sceKernelDelayThreadCB",
        tm::SEND_SIGNAL => "sceKernelSendSignal",
        pm::LIBC_GETTIMEOFDAY => "sceKernelLibcGettimeofday",
        pm::CALL_ABORT_HANDLER => "sceKernelCallAbortHandler",
        d::GET_VCOUNT => "sceDisplayGetVcount",
        dbg::ASSERTION_HANDLER => "sceDbgAssertionHandler",
        dbg::LOGGING_HANDLER => "sceDbgLoggingHandler",

        // --- SceFios2Kernel ---------------------------------------------------
        fios2::OVERLAY_ADD => "_sceFiosKernelOverlayAdd",
        fios2::OVERLAY_ADD_FOR_PROCESS => "_sceFiosKernelOverlayAddForProcess",
        fios2::OVERLAY_MODIFY => "_sceFiosKernelOverlayModify",
        fios2::OVERLAY_MODIFY_FOR_PROCESS => "_sceFiosKernelOverlayModifyForProcess",
        fios2::OVERLAY_REMOVE => "_sceFiosKernelOverlayRemove",
        fios2::OVERLAY_REMOVE_FOR_PROCESS => "_sceFiosKernelOverlayRemoveForProcess",
        fios2::OVERLAY_GET_INFO => "_sceFiosKernelOverlayGetInfo",
        fios2::OVERLAY_GET_INFO_FOR_PROCESS => "_sceFiosKernelOverlayGetInfoForProcess",
        fios2::OVERLAY_GET_LIST => "_sceFiosKernelOverlayGetList",
        fios2::OVERLAY_RESOLVE_SYNC => "_sceFiosKernelOverlayResolveSync",
        fios2::OVERLAY_RESOLVE_WITH_RANGE_SYNC => "_sceFiosKernelOverlayResolveWithRangeSync",
        fios2::OVERLAY_GET_RECOMMENDED_SCHEDULER => "_sceFiosKernelOverlayGetRecommendedScheduler",
        fios2::OVERLAY_THREAD_IS_DISABLED => "_sceFiosKernelOverlayThreadIsDisabled",
        fios2::OVERLAY_THREAD_SET_DISABLED => "_sceFiosKernelOverlayThreadSetDisabled",
        fios2::DH_OPEN_SYNC => "_sceFiosKernelOverlayDHOpenSync",
        fios2::DH_READ_SYNC => "_sceFiosKernelOverlayDHReadSync",
        fios2::DH_STAT_SYNC => "_sceFiosKernelOverlayDHStatSync",
        fios2::DH_CHSTAT_SYNC => "_sceFiosKernelOverlayDHChstatSync",
        fios2::DH_SYNC_SYNC => "_sceFiosKernelOverlayDHSyncSync",
        fios2::DH_CLOSE_SYNC => "_sceFiosKernelOverlayDHCloseSync",

        _ => UNKNOWN_NAME,
    }
}
