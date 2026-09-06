/*
 * vitaslop conformance corpus: minimal GXM spinning cube (blob-free).
 *
 * This is the graphics north star, authored clean-room from the MIT
 * vita-headers API and the publicly documented GXM init sequence. It is NOT
 * derived from the (unlicensed) vitasdk gxm sample.
 *
 * Purpose in the emulator bring-up (work-backwards plan, see agent-notes):
 *   - It is a realistic ARM + NID-import corpus that drives the loader, the
 *     transpiler CFG buildout, and the host-module ABI demand-first.
 *   - Milestones 1-4 need NO GPU emulation: load it, resolve its NID imports,
 *     execute its CPU code, and capture the sequence of sceGxm* calls it makes.
 *
 * Shaders: the two SceGxmProgram blobs below are PLACEHOLDERS (milestone 5).
 * Compiling real GXM shaders needs Sony's libshacccg (a blob we refuse) or
 * hand-authored precompiled .gxp. Until we actually rasterize, the shader bytes
 * are opaque data the CPU path never interprets - our host stubs for
 * sceGxmProgramCheck / the shader patcher accept them and record their use. The
 * call structure around them is faithful.
 *
 * Built -nostdlib with a tiny self-contained runtime (bottom of file) so the
 * committed corpus binary stays license-clean and the loader has a small,
 * well-understood import surface (Sony NID stubs only, no newlib).
 */

#include <psp2/types.h>
#include <psp2/kernel/sysmem.h>
#include <psp2/gxm.h>
#include <psp2/display.h>
#include <psp2/ctrl.h>

#define DISPLAY_WIDTH        960
#define DISPLAY_HEIGHT       544
#define DISPLAY_STRIDE       1024
#define DISPLAY_BUFFER_COUNT 2
#define DISPLAY_PIXEL_BYTES  4

#define ALIGN(x, a) (((x) + ((a) - 1)) & ~((a) - 1))

/* ---- tiny runtime (declared here, defined at bottom) ------------------- */
static void  rt_memset(void *dst, int v, unsigned int n);
static float rt_sinf(float x);
static float rt_cosf(float x);

/* ---- placeholder shaders (milestone 5) --------------------------------- */
/* A valid SceGxmProgram begins with the four-byte magic "GXP\0". These carry
 * only the magic so the structure links and pointers flow; the USSE payload is
 * intentionally empty. Replace with real precompiled .gxp when we rasterize. */
__attribute__((aligned(64)))
static const unsigned char cube_vert_gxp[64] = { 'G', 'X', 'P', 0 };
__attribute__((aligned(64)))
static const unsigned char cube_frag_gxp[64] = { 'G', 'X', 'P', 0 };

/* ---- geometry ---------------------------------------------------------- */
typedef struct {
	float x, y, z;
	unsigned int color; /* 0xAABBGGRR */
} CubeVertex;

/* 8 corners of a unit cube, one color per corner. */
static const CubeVertex cube_vertices[8] = {
	{ -1.0f, -1.0f, -1.0f, 0xff0000ff },
	{  1.0f, -1.0f, -1.0f, 0xff00ff00 },
	{  1.0f,  1.0f, -1.0f, 0xffff0000 },
	{ -1.0f,  1.0f, -1.0f, 0xffffff00 },
	{ -1.0f, -1.0f,  1.0f, 0xffff00ff },
	{  1.0f, -1.0f,  1.0f, 0xff00ffff },
	{  1.0f,  1.0f,  1.0f, 0xffffffff },
	{ -1.0f,  1.0f,  1.0f, 0xff808080 },
};

/* 12 triangles, CCW winding. */
static const unsigned short cube_indices[36] = {
	0, 2, 1, 0, 3, 2, /* -Z */
	4, 5, 6, 4, 6, 7, /* +Z */
	0, 1, 5, 0, 5, 4, /* -Y */
	3, 6, 2, 3, 7, 6, /* +Y */
	0, 4, 7, 0, 7, 3, /* -X */
	1, 2, 6, 1, 6, 5, /* +X */
};

/* ---- GPU memory helpers ------------------------------------------------ */
/* Allocate a GPU-mapped LPDDR block and return its base (uid out via *uid). */
static void *gpu_alloc(SceKernelMemBlockType type, unsigned int size,
                       unsigned int alignment, SceGxmMemoryAttribFlags attr,
                       SceUID *uid) {
	/* LPDDR granularity is 4 KiB, CDRAM is 256 KiB. Round up per type. */
	if (type == SCE_KERNEL_MEMBLOCK_TYPE_USER_CDRAM_RW)
		size = ALIGN(size, 256 * 1024);
	else
		size = ALIGN(size, 4 * 1024);
	(void)alignment;

	SceUID memuid = sceKernelAllocMemBlock("vitaslop_gpu", type, size, NULL);
	if (memuid < 0)
		return NULL;

	void *base = NULL;
	if (sceKernelGetMemBlockBase(memuid, &base) < 0)
		return NULL;

	if (sceGxmMapMemory(base, size, attr) < 0)
		return NULL;

	*uid = memuid;
	return base;
}

/* Vertex USSE program memory: mapped, returns base and *usse_offset. */
static void *vertex_usse_alloc(unsigned int size, SceUID *uid,
                               unsigned int *usse_offset) {
	size = ALIGN(size, 4 * 1024);
	SceUID memuid = sceKernelAllocMemBlock("vitaslop_vert_usse",
		SCE_KERNEL_MEMBLOCK_TYPE_USER_RW, size, NULL);
	void *base = NULL;
	sceKernelGetMemBlockBase(memuid, &base);
	sceGxmMapVertexUsseMemory(base, size, usse_offset);
	*uid = memuid;
	return base;
}

static void *fragment_usse_alloc(unsigned int size, SceUID *uid,
                                 unsigned int *usse_offset) {
	size = ALIGN(size, 4 * 1024);
	SceUID memuid = sceKernelAllocMemBlock("vitaslop_frag_usse",
		SCE_KERNEL_MEMBLOCK_TYPE_USER_RW, size, NULL);
	void *base = NULL;
	sceKernelGetMemBlockBase(memuid, &base);
	sceGxmMapFragmentUsseMemory(base, size, usse_offset);
	*uid = memuid;
	return base;
}

/* ---- display swap ------------------------------------------------------ */
typedef struct {
	void *address;
} DisplayQueueData;

static void display_queue_callback(const void *callback_data) {
	const DisplayQueueData *d = (const DisplayQueueData *)callback_data;

	SceDisplayFrameBuf fb;
	rt_memset(&fb, 0, sizeof(fb));
	fb.size        = sizeof(fb);
	fb.base        = d->address;
	fb.pitch       = DISPLAY_STRIDE;
	fb.pixelformat = SCE_DISPLAY_PIXELFORMAT_A8B8G8R8;
	fb.width       = DISPLAY_WIDTH;
	fb.height      = DISPLAY_HEIGHT;

	sceDisplaySetFrameBuf(&fb, SCE_DISPLAY_SETBUF_NEXTFRAME);
}

/* ---- 4x4 matrix math (column-major, GL-style) -------------------------- */
static void mat_identity(float *m) {
	rt_memset(m, 0, 16 * sizeof(float));
	m[0] = m[5] = m[10] = m[15] = 1.0f;
}

static void mat_mul(float *out, const float *a, const float *b) {
	float r[16];
	for (int c = 0; c < 4; c++) {
		for (int row = 0; row < 4; row++) {
			r[c * 4 + row] =
				a[0 * 4 + row] * b[c * 4 + 0] +
				a[1 * 4 + row] * b[c * 4 + 1] +
				a[2 * 4 + row] * b[c * 4 + 2] +
				a[3 * 4 + row] * b[c * 4 + 3];
		}
	}
	for (int i = 0; i < 16; i++)
		out[i] = r[i];
}

/* Simple perspective projection. */
static void mat_perspective(float *m, float fov, float aspect,
                            float near_z, float far_z) {
	float f = 1.0f / rt_sinf(fov * 0.5f) * rt_cosf(fov * 0.5f); /* cot(fov/2) */
	rt_memset(m, 0, 16 * sizeof(float));
	m[0]  = f / aspect;
	m[5]  = f;
	m[10] = (far_z + near_z) / (near_z - far_z);
	m[11] = -1.0f;
	m[14] = (2.0f * far_z * near_z) / (near_z - far_z);
}

static void mat_rotate_y(float *m, float a) {
	mat_identity(m);
	float s = rt_sinf(a), c = rt_cosf(a);
	m[0] = c;  m[2] = s;
	m[8] = -s; m[10] = c;
}

static void mat_rotate_x(float *m, float a) {
	mat_identity(m);
	float s = rt_sinf(a), c = rt_cosf(a);
	m[5] = c;  m[6] = s;
	m[9] = -s; m[10] = c;
}

/* ---- static buffers for the shader patcher (no malloc) ----------------- */
#define PATCHER_BUFFER_SIZE       (64 * 1024)
#define PATCHER_VERTEX_USSE_SIZE  (64 * 1024)
#define PATCHER_FRAGMENT_USSE_SIZE (64 * 1024)

int main(void) {
	/* --- 1. initialize GXM --- */
	SceGxmInitializeParams init_params;
	rt_memset(&init_params, 0, sizeof(init_params));
	init_params.flags                       = 0;
	init_params.displayQueueMaxPendingCount = 2;
	init_params.displayQueueCallback        = display_queue_callback;
	init_params.displayQueueCallbackDataSize = sizeof(DisplayQueueData);
	init_params.parameterBufferSize         = 16 * 1024 * 1024;
	sceGxmInitialize(&init_params);

	/* --- 2. allocate ring buffers + create the rendering context --- */
	SceUID vdm_uid, vertex_uid, fragment_uid, fragment_usse_uid, host_dummy;
	unsigned int fragment_usse_offset;

	const unsigned int VDM_RING_SIZE      = 128 * 1024;
	const unsigned int VERTEX_RING_SIZE   = 128 * 1024;
	const unsigned int FRAGMENT_RING_SIZE = 128 * 1024;
	const unsigned int FRAGMENT_USSE_SIZE = 128 * 1024;

	void *vdm_ring = gpu_alloc(SCE_KERNEL_MEMBLOCK_TYPE_USER_RW, VDM_RING_SIZE,
		4, SCE_GXM_MEMORY_ATTRIB_READ, &vdm_uid);
	void *vertex_ring = gpu_alloc(SCE_KERNEL_MEMBLOCK_TYPE_USER_RW, VERTEX_RING_SIZE,
		4, SCE_GXM_MEMORY_ATTRIB_READ, &vertex_uid);
	void *fragment_ring = gpu_alloc(SCE_KERNEL_MEMBLOCK_TYPE_USER_RW, FRAGMENT_RING_SIZE,
		4, SCE_GXM_MEMORY_ATTRIB_READ, &fragment_uid);
	void *fragment_usse_ring = fragment_usse_alloc(FRAGMENT_USSE_SIZE,
		&fragment_usse_uid, &fragment_usse_offset);

	static unsigned char host_mem[16 * 1024] __attribute__((aligned(16)));
	(void)host_dummy;

	SceGxmContextParams ctx_params;
	rt_memset(&ctx_params, 0, sizeof(ctx_params));
	ctx_params.hostMem                       = host_mem;
	ctx_params.hostMemSize                   = sizeof(host_mem);
	ctx_params.vdmRingBufferMem              = vdm_ring;
	ctx_params.vdmRingBufferMemSize          = VDM_RING_SIZE;
	ctx_params.vertexRingBufferMem           = vertex_ring;
	ctx_params.vertexRingBufferMemSize       = VERTEX_RING_SIZE;
	ctx_params.fragmentRingBufferMem         = fragment_ring;
	ctx_params.fragmentRingBufferMemSize     = FRAGMENT_RING_SIZE;
	ctx_params.fragmentUsseRingBufferMem     = fragment_usse_ring;
	ctx_params.fragmentUsseRingBufferMemSize = FRAGMENT_USSE_SIZE;
	ctx_params.fragmentUsseRingBufferOffset  = fragment_usse_offset;

	SceGxmContext *context = NULL;
	sceGxmCreateContext(&ctx_params, &context);

	/* --- 3. create the render target --- */
	SceGxmRenderTargetParams rt_params;
	rt_memset(&rt_params, 0, sizeof(rt_params));
	rt_params.width           = DISPLAY_WIDTH;
	rt_params.height          = DISPLAY_HEIGHT;
	rt_params.scenesPerFrame  = 1;
	rt_params.multisampleMode = SCE_GXM_MULTISAMPLE_NONE;
	rt_params.driverMemBlock  = -1;

	SceGxmRenderTarget *render_target = NULL;
	sceGxmCreateRenderTarget(&rt_params, &render_target);

	/* --- 4. display buffers, color surfaces and sync objects --- */
	void *display_buffer[DISPLAY_BUFFER_COUNT];
	SceUID display_uid[DISPLAY_BUFFER_COUNT];
	SceGxmColorSurface display_surface[DISPLAY_BUFFER_COUNT];
	SceGxmSyncObject *display_sync[DISPLAY_BUFFER_COUNT];

	for (int i = 0; i < DISPLAY_BUFFER_COUNT; i++) {
		display_buffer[i] = gpu_alloc(SCE_KERNEL_MEMBLOCK_TYPE_USER_CDRAM_RW,
			DISPLAY_STRIDE * DISPLAY_HEIGHT * DISPLAY_PIXEL_BYTES,
			256 * 1024, SCE_GXM_MEMORY_ATTRIB_RW, &display_uid[i]);

		sceGxmColorSurfaceInit(&display_surface[i],
			SCE_GXM_COLOR_FORMAT_A8B8G8R8,
			SCE_GXM_COLOR_SURFACE_LINEAR,
			SCE_GXM_COLOR_SURFACE_SCALE_NONE,
			SCE_GXM_OUTPUT_REGISTER_SIZE_32BIT,
			DISPLAY_WIDTH, DISPLAY_HEIGHT, DISPLAY_STRIDE,
			display_buffer[i]);

		sceGxmSyncObjectCreate(&display_sync[i]);
	}

	/* --- 5. depth/stencil surface --- */
	unsigned int depth_width  = ALIGN(DISPLAY_WIDTH, SCE_GXM_TILE_SIZEX);
	unsigned int depth_height = ALIGN(DISPLAY_HEIGHT, SCE_GXM_TILE_SIZEY);
	unsigned int depth_samples = depth_width * depth_height;

	SceUID depth_uid;
	void *depth_buffer = gpu_alloc(SCE_KERNEL_MEMBLOCK_TYPE_USER_RW,
		depth_samples * 4, 4, SCE_GXM_MEMORY_ATTRIB_RW, &depth_uid);

	SceGxmDepthStencilSurface depth_surface;
	sceGxmDepthStencilSurfaceInit(&depth_surface,
		SCE_GXM_DEPTH_STENCIL_FORMAT_S8D24,
		SCE_GXM_DEPTH_STENCIL_SURFACE_TILED,
		depth_width, depth_buffer, NULL);

	/* --- 6. shader patcher --- */
	SceUID patcher_buffer_uid, patcher_vert_usse_uid, patcher_frag_usse_uid;
	unsigned int patcher_vert_usse_offset, patcher_frag_usse_offset;

	void *patcher_buffer = gpu_alloc(SCE_KERNEL_MEMBLOCK_TYPE_USER_RW,
		PATCHER_BUFFER_SIZE, 4, SCE_GXM_MEMORY_ATTRIB_RW, &patcher_buffer_uid);
	void *patcher_vert_usse = vertex_usse_alloc(PATCHER_VERTEX_USSE_SIZE,
		&patcher_vert_usse_uid, &patcher_vert_usse_offset);
	void *patcher_frag_usse = fragment_usse_alloc(PATCHER_FRAGMENT_USSE_SIZE,
		&patcher_frag_usse_uid, &patcher_frag_usse_offset);

	SceGxmShaderPatcherParams patcher_params;
	rt_memset(&patcher_params, 0, sizeof(patcher_params));
	patcher_params.bufferMem          = patcher_buffer;
	patcher_params.bufferMemSize      = PATCHER_BUFFER_SIZE;
	patcher_params.vertexUsseMem      = patcher_vert_usse;
	patcher_params.vertexUsseMemSize  = PATCHER_VERTEX_USSE_SIZE;
	patcher_params.vertexUsseOffset   = patcher_vert_usse_offset;
	patcher_params.fragmentUsseMem    = patcher_frag_usse;
	patcher_params.fragmentUsseMemSize = PATCHER_FRAGMENT_USSE_SIZE;
	patcher_params.fragmentUsseOffset = patcher_frag_usse_offset;

	SceGxmShaderPatcher *patcher = NULL;
	sceGxmShaderPatcherCreate(&patcher_params, &patcher);

	/* --- 7. register programs and build vertex/fragment programs --- */
	const SceGxmProgram *vert_program = (const SceGxmProgram *)cube_vert_gxp;
	const SceGxmProgram *frag_program = (const SceGxmProgram *)cube_frag_gxp;
	sceGxmProgramCheck(vert_program);
	sceGxmProgramCheck(frag_program);

	SceGxmShaderPatcherId vert_id, frag_id;
	sceGxmShaderPatcherRegisterProgram(patcher, vert_program, &vert_id);
	sceGxmShaderPatcherRegisterProgram(patcher, frag_program, &frag_id);

	SceGxmVertexAttribute attributes[2];
	rt_memset(attributes, 0, sizeof(attributes));
	/* aPosition: float3 at offset 0, register 0 */
	attributes[0].streamIndex    = 0;
	attributes[0].offset         = 0;
	attributes[0].format         = SCE_GXM_ATTRIBUTE_FORMAT_F32;
	attributes[0].componentCount = 3;
	attributes[0].regIndex       = 0;
	/* aColor: 4 unsigned bytes at offset 12, register 1 */
	attributes[1].streamIndex    = 0;
	attributes[1].offset         = 12;
	attributes[1].format         = SCE_GXM_ATTRIBUTE_FORMAT_U8N;
	attributes[1].componentCount = 4;
	attributes[1].regIndex       = 1;

	SceGxmVertexStream streams[1];
	rt_memset(streams, 0, sizeof(streams));
	streams[0].stride      = sizeof(CubeVertex);
	streams[0].indexSource = SCE_GXM_INDEX_SOURCE_INDEX_16BIT;

	SceGxmVertexProgram *cube_vertex_program = NULL;
	sceGxmShaderPatcherCreateVertexProgram(patcher, vert_id,
		attributes, 2, streams, 1, &cube_vertex_program);

	SceGxmFragmentProgram *cube_fragment_program = NULL;
	sceGxmShaderPatcherCreateFragmentProgram(patcher, frag_id,
		SCE_GXM_OUTPUT_REGISTER_FORMAT_UCHAR4,
		SCE_GXM_MULTISAMPLE_NONE, NULL, vert_program,
		&cube_fragment_program);

	/* --- 8. upload geometry to GPU memory --- */
	SceUID vbo_uid, ibo_uid;
	CubeVertex *vertices = gpu_alloc(SCE_KERNEL_MEMBLOCK_TYPE_USER_RW,
		sizeof(cube_vertices), 4, SCE_GXM_MEMORY_ATTRIB_READ, &vbo_uid);
	unsigned short *indices = gpu_alloc(SCE_KERNEL_MEMBLOCK_TYPE_USER_RW,
		sizeof(cube_indices), 4, SCE_GXM_MEMORY_ATTRIB_READ, &ibo_uid);

	for (int i = 0; i < 8; i++)
		vertices[i] = cube_vertices[i];
	for (int i = 0; i < 36; i++)
		indices[i] = cube_indices[i];

	/* Locate the wvp matrix uniform in the (placeholder) vertex program. */
	const SceGxmProgramParameter *wvp_param =
		sceGxmProgramFindParameterByName(vert_program, "wvp");

	/* --- 9. main loop: spin the cube --- */
	float projection[16], view[16], rot_x[16], rot_y[16], model[16], mvp[16];
	mat_perspective(projection, 0.9f,
		(float)DISPLAY_WIDTH / (float)DISPLAY_HEIGHT, 0.1f, 100.0f);

	/* view = translate cube back along -Z by 5 units */
	mat_identity(view);
	view[14] = -5.0f;

	SceCtrlData pad;
	float angle = 0.0f;
	unsigned int back_buffer = 0, front_buffer = 0;

	for (int frame = 0; frame < 600; frame++) {
		sceCtrlPeekBufferPositive(0, &pad, 1);
		if (pad.buttons & SCE_CTRL_START)
			break;

		angle += 0.02f;
		mat_rotate_x(rot_x, angle * 0.7f);
		mat_rotate_y(rot_y, angle);
		mat_mul(model, rot_y, rot_x);

		float vp[16];
		mat_mul(vp, projection, view);
		mat_mul(mvp, vp, model);

		sceGxmBeginScene(context, 0, render_target, NULL, NULL,
			display_sync[back_buffer],
			&display_surface[back_buffer], &depth_surface);

		sceGxmSetVertexProgram(context, cube_vertex_program);
		sceGxmSetFragmentProgram(context, cube_fragment_program);

		void *vertex_uniforms = NULL;
		sceGxmReserveVertexDefaultUniformBuffer(context, &vertex_uniforms);
		sceGxmSetUniformDataF(vertex_uniforms, wvp_param, 0, 16, mvp);

		sceGxmSetVertexStream(context, 0, vertices);
		sceGxmDraw(context, SCE_GXM_PRIMITIVE_TRIANGLES,
			SCE_GXM_INDEX_FORMAT_U16, indices, 36);

		sceGxmEndScene(context, NULL, NULL);

		sceGxmPadHeartbeat(&display_surface[back_buffer],
			display_sync[back_buffer]);

		/* queue the swap */
		DisplayQueueData queue_data;
		queue_data.address = display_buffer[back_buffer];
		sceGxmDisplayQueueAddEntry(display_sync[front_buffer],
			display_sync[back_buffer], &queue_data);

		front_buffer = back_buffer;
		back_buffer  = (back_buffer + 1) % DISPLAY_BUFFER_COUNT;
	}

	/* --- 10. teardown --- */
	sceGxmDisplayQueueFinish();
	sceGxmFinish(context);

	sceGxmShaderPatcherReleaseVertexProgram(patcher, cube_vertex_program);
	sceGxmShaderPatcherReleaseFragmentProgram(patcher, cube_fragment_program);
	sceGxmShaderPatcherUnregisterProgram(patcher, vert_id);
	sceGxmShaderPatcherUnregisterProgram(patcher, frag_id);
	sceGxmShaderPatcherDestroy(patcher);

	sceGxmDestroyContext(context);
	sceGxmDestroyRenderTarget(render_target);
	sceGxmTerminate();
	return 0;
}

/* ======================================================================= *
 *  Tiny freestanding runtime (-nostdlib).
 * ======================================================================= */

static void rt_memset(void *dst, int v, unsigned int n) {
	unsigned char *p = (unsigned char *)dst;
	for (unsigned int i = 0; i < n; i++)
		p[i] = (unsigned char)v;
}

/* memcpy: the compiler may emit calls to it for struct/array copies. */
void *memcpy(void *dst, const void *src, unsigned int n) {
	unsigned char *d = (unsigned char *)dst;
	const unsigned char *s = (const unsigned char *)src;
	for (unsigned int i = 0; i < n; i++)
		d[i] = s[i];
	return dst;
}

void *memset(void *dst, int v, unsigned int n) {
	rt_memset(dst, v, n);
	return dst;
}

/* Range-reduced Taylor sine/cosine. Precision is fine for a demo cube. */
#define RT_PI  3.14159265358979323846f
#define RT_TAU 6.28318530717958647692f

static float rt_sinf(float x) {
	/* reduce to [-pi, pi] */
	while (x > RT_PI)  x -= RT_TAU;
	while (x < -RT_PI) x += RT_TAU;
	float x2 = x * x;
	/* 7th-order Taylor: x - x^3/6 + x^5/120 - x^7/5040 */
	return x * (1.0f - x2 * (1.0f / 6.0f
		- x2 * (1.0f / 120.0f
		- x2 * (1.0f / 5040.0f))));
}

static float rt_cosf(float x) {
	return rt_sinf(x + RT_PI * 0.5f);
}

/* Entry point. The loader jumps here; set up nothing fancy, call main, and
 * spin forever on return (there is no OS to exit to in our bring-up yet). */
void _start(void) {
	main();
	for (;;) { }
}
