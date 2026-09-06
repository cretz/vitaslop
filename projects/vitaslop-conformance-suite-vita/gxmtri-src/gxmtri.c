/*
 * vitaslop conformance corpus: minimal GXM offscreen triangle (blob-free).
 *
 * The render north star's runnable milestone: a REAL Vita executable that drives
 * libgxm through init -> one offscreen scene -> one indexed draw -> finish, then
 * exits cleanly (so the plain single-thread Vm runs it to completion, unlike the
 * spinning display-loop cube). The emulator captures the GXM command stream and
 * both the software rasterizer and the wgpu renderer turn it into pixels - this is
 * the end-to-end proof that our GXM->render translation is faithful on a real
 * clean-room artifact, not just synthetic Rust scenes.
 *
 * Authored clean-room from the MIT vita-headers API. Built -nostdlib with a tiny
 * self-contained runtime so the corpus binary stays license-clean and the import
 * surface is Sony NID stubs only.
 *
 * Shaders are PLACEHOLDERS (magic-only SceGxmProgram) - the same blob-free stance
 * as cube.c. This draw deliberately needs NO shader reflection: the three vertices
 * are already in normalized device coordinates and there is no uniform buffer, so
 * the renderer recovers the draw purely from the captured vertex ATTRIBUTES (which
 * the app declares to sceGxmShaderPatcherCreateVertexProgram) and the vertex/index
 * streams - the parts that do not depend on the (absent) compiled shader.
 */

#include <psp2/types.h>
#include <psp2/kernel/sysmem.h>
#include <psp2/kernel/processmgr.h>
#include <psp2/gxm.h>

#define SURFACE_WIDTH  128
#define SURFACE_HEIGHT 128
#define SURFACE_STRIDE 128

#define ALIGN(x, a) (((x) + ((a) - 1)) & ~((a) - 1))

static void rt_memset(void *dst, int v, unsigned int n);

/* ---- placeholder shaders (magic only, USSE payload intentionally empty) ---- */
__attribute__((aligned(64)))
static const unsigned char tri_vert_gxp[64] = { 'G', 'X', 'P', 0 };
__attribute__((aligned(64)))
static const unsigned char tri_frag_gxp[64] = { 'G', 'X', 'P', 0 };

/* ---- geometry: an NDC triangle with one color per corner ---------------- */
typedef struct {
	float x, y, z;
	unsigned int color; /* 0xAABBGGRR */
} TriVertex;

static const TriVertex tri_vertices[3] = {
	{ -0.6f, -0.6f, 0.0f, 0xff0000ff }, /* red   */
	{  0.6f, -0.6f, 0.0f, 0xff00ff00 }, /* green */
	{  0.0f,  0.7f, 0.0f, 0xffff0000 }, /* blue  */
};
static const unsigned short tri_indices[3] = { 0, 1, 2 };

/* ---- GPU memory helpers ------------------------------------------------- */
static void *gpu_alloc(SceKernelMemBlockType type, unsigned int size,
                       SceGxmMemoryAttribFlags attr, SceUID *uid) {
	if (type == SCE_KERNEL_MEMBLOCK_TYPE_USER_CDRAM_RW)
		size = ALIGN(size, 256 * 1024);
	else
		size = ALIGN(size, 4 * 1024);

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

static void *vertex_usse_alloc(unsigned int size, SceUID *uid, unsigned int *usse_offset) {
	size = ALIGN(size, 4 * 1024);
	SceUID memuid = sceKernelAllocMemBlock("vitaslop_vert_usse",
		SCE_KERNEL_MEMBLOCK_TYPE_USER_RW, size, NULL);
	void *base = NULL;
	sceKernelGetMemBlockBase(memuid, &base);
	sceGxmMapVertexUsseMemory(base, size, usse_offset);
	*uid = memuid;
	return base;
}

static void *fragment_usse_alloc(unsigned int size, SceUID *uid, unsigned int *usse_offset) {
	size = ALIGN(size, 4 * 1024);
	SceUID memuid = sceKernelAllocMemBlock("vitaslop_frag_usse",
		SCE_KERNEL_MEMBLOCK_TYPE_USER_RW, size, NULL);
	void *base = NULL;
	sceKernelGetMemBlockBase(memuid, &base);
	sceGxmMapFragmentUsseMemory(base, size, usse_offset);
	*uid = memuid;
	return base;
}

#define PATCHER_BUFFER_SIZE        (64 * 1024)
#define PATCHER_VERTEX_USSE_SIZE   (64 * 1024)
#define PATCHER_FRAGMENT_USSE_SIZE (64 * 1024)

int main(void) {
	/* --- 1. initialize GXM (no display queue: this is offscreen) --- */
	SceGxmInitializeParams init_params;
	rt_memset(&init_params, 0, sizeof(init_params));
	init_params.flags                        = 0;
	init_params.displayQueueMaxPendingCount  = 0;
	init_params.displayQueueCallback         = NULL;
	init_params.displayQueueCallbackDataSize = 0;
	init_params.parameterBufferSize          = 16 * 1024 * 1024;
	sceGxmInitialize(&init_params);

	/* --- 2. ring buffers + rendering context --- */
	SceUID vdm_uid, vertex_uid, fragment_uid, fragment_usse_uid;
	unsigned int fragment_usse_offset;
	const unsigned int RING = 128 * 1024;

	void *vdm_ring = gpu_alloc(SCE_KERNEL_MEMBLOCK_TYPE_USER_RW, RING,
		SCE_GXM_MEMORY_ATTRIB_READ, &vdm_uid);
	void *vertex_ring = gpu_alloc(SCE_KERNEL_MEMBLOCK_TYPE_USER_RW, RING,
		SCE_GXM_MEMORY_ATTRIB_READ, &vertex_uid);
	void *fragment_ring = gpu_alloc(SCE_KERNEL_MEMBLOCK_TYPE_USER_RW, RING,
		SCE_GXM_MEMORY_ATTRIB_READ, &fragment_uid);
	void *fragment_usse_ring = fragment_usse_alloc(RING, &fragment_usse_uid, &fragment_usse_offset);

	static unsigned char host_mem[16 * 1024] __attribute__((aligned(16)));

	SceGxmContextParams ctx_params;
	rt_memset(&ctx_params, 0, sizeof(ctx_params));
	ctx_params.hostMem                       = host_mem;
	ctx_params.hostMemSize                   = sizeof(host_mem);
	ctx_params.vdmRingBufferMem              = vdm_ring;
	ctx_params.vdmRingBufferMemSize          = RING;
	ctx_params.vertexRingBufferMem           = vertex_ring;
	ctx_params.vertexRingBufferMemSize       = RING;
	ctx_params.fragmentRingBufferMem         = fragment_ring;
	ctx_params.fragmentRingBufferMemSize     = RING;
	ctx_params.fragmentUsseRingBufferMem     = fragment_usse_ring;
	ctx_params.fragmentUsseRingBufferMemSize = RING;
	ctx_params.fragmentUsseRingBufferOffset  = fragment_usse_offset;

	SceGxmContext *context = NULL;
	sceGxmCreateContext(&ctx_params, &context);

	/* --- 3. render target --- */
	SceGxmRenderTargetParams rt_params;
	rt_memset(&rt_params, 0, sizeof(rt_params));
	rt_params.width           = SURFACE_WIDTH;
	rt_params.height          = SURFACE_HEIGHT;
	rt_params.scenesPerFrame  = 1;
	rt_params.multisampleMode = SCE_GXM_MULTISAMPLE_NONE;
	rt_params.driverMemBlock  = -1;

	SceGxmRenderTarget *render_target = NULL;
	sceGxmCreateRenderTarget(&rt_params, &render_target);

	/* --- 4. one offscreen color surface + sync object --- */
	SceUID color_uid;
	void *color_buffer = gpu_alloc(SCE_KERNEL_MEMBLOCK_TYPE_USER_CDRAM_RW,
		SURFACE_STRIDE * SURFACE_HEIGHT * 4, SCE_GXM_MEMORY_ATTRIB_RW, &color_uid);

	SceGxmColorSurface color_surface;
	sceGxmColorSurfaceInit(&color_surface,
		SCE_GXM_COLOR_FORMAT_A8B8G8R8,
		SCE_GXM_COLOR_SURFACE_LINEAR,
		SCE_GXM_COLOR_SURFACE_SCALE_NONE,
		SCE_GXM_OUTPUT_REGISTER_SIZE_32BIT,
		SURFACE_WIDTH, SURFACE_HEIGHT, SURFACE_STRIDE, color_buffer);

	SceGxmSyncObject *sync = NULL;
	sceGxmSyncObjectCreate(&sync);

	/* --- 5. depth/stencil surface --- */
	unsigned int depth_width  = ALIGN(SURFACE_WIDTH, SCE_GXM_TILE_SIZEX);
	unsigned int depth_height = ALIGN(SURFACE_HEIGHT, SCE_GXM_TILE_SIZEY);
	SceUID depth_uid;
	void *depth_buffer = gpu_alloc(SCE_KERNEL_MEMBLOCK_TYPE_USER_RW,
		depth_width * depth_height * 4, SCE_GXM_MEMORY_ATTRIB_RW, &depth_uid);

	SceGxmDepthStencilSurface depth_surface;
	sceGxmDepthStencilSurfaceInit(&depth_surface,
		SCE_GXM_DEPTH_STENCIL_FORMAT_S8D24,
		SCE_GXM_DEPTH_STENCIL_SURFACE_TILED,
		depth_width, depth_buffer, NULL);

	/* --- 6. shader patcher --- */
	SceUID patcher_buffer_uid, patcher_vert_usse_uid, patcher_frag_usse_uid;
	unsigned int patcher_vert_usse_offset, patcher_frag_usse_offset;

	void *patcher_buffer = gpu_alloc(SCE_KERNEL_MEMBLOCK_TYPE_USER_RW,
		PATCHER_BUFFER_SIZE, SCE_GXM_MEMORY_ATTRIB_RW, &patcher_buffer_uid);
	void *patcher_vert_usse = vertex_usse_alloc(PATCHER_VERTEX_USSE_SIZE,
		&patcher_vert_usse_uid, &patcher_vert_usse_offset);
	void *patcher_frag_usse = fragment_usse_alloc(PATCHER_FRAGMENT_USSE_SIZE,
		&patcher_frag_usse_uid, &patcher_frag_usse_offset);

	SceGxmShaderPatcherParams patcher_params;
	rt_memset(&patcher_params, 0, sizeof(patcher_params));
	patcher_params.bufferMem           = patcher_buffer;
	patcher_params.bufferMemSize       = PATCHER_BUFFER_SIZE;
	patcher_params.vertexUsseMem       = patcher_vert_usse;
	patcher_params.vertexUsseMemSize   = PATCHER_VERTEX_USSE_SIZE;
	patcher_params.vertexUsseOffset    = patcher_vert_usse_offset;
	patcher_params.fragmentUsseMem     = patcher_frag_usse;
	patcher_params.fragmentUsseMemSize = PATCHER_FRAGMENT_USSE_SIZE;
	patcher_params.fragmentUsseOffset  = patcher_frag_usse_offset;

	SceGxmShaderPatcher *patcher = NULL;
	sceGxmShaderPatcherCreate(&patcher_params, &patcher);

	/* --- 7. register programs + build vertex/fragment programs --- */
	const SceGxmProgram *vert_program = (const SceGxmProgram *)tri_vert_gxp;
	const SceGxmProgram *frag_program = (const SceGxmProgram *)tri_frag_gxp;
	sceGxmProgramCheck(vert_program);
	sceGxmProgramCheck(frag_program);

	SceGxmShaderPatcherId vert_id, frag_id;
	sceGxmShaderPatcherRegisterProgram(patcher, vert_program, &vert_id);
	sceGxmShaderPatcherRegisterProgram(patcher, frag_program, &frag_id);

	/* Attribute layout: aPosition float3 @0 reg0, aColor u8n x4 @12 reg1. These are
	 * declared by the app (independent of the shader), so the capture recovers them. */
	SceGxmVertexAttribute attributes[2];
	rt_memset(attributes, 0, sizeof(attributes));
	attributes[0].streamIndex    = 0;
	attributes[0].offset         = 0;
	attributes[0].format         = SCE_GXM_ATTRIBUTE_FORMAT_F32;
	attributes[0].componentCount = 3;
	attributes[0].regIndex       = 0;
	attributes[1].streamIndex    = 0;
	attributes[1].offset         = 12;
	attributes[1].format         = SCE_GXM_ATTRIBUTE_FORMAT_U8N;
	attributes[1].componentCount = 4;
	attributes[1].regIndex       = 1;

	SceGxmVertexStream streams[1];
	rt_memset(streams, 0, sizeof(streams));
	streams[0].stride      = sizeof(TriVertex);
	streams[0].indexSource = SCE_GXM_INDEX_SOURCE_INDEX_16BIT;

	SceGxmVertexProgram *tri_vertex_program = NULL;
	sceGxmShaderPatcherCreateVertexProgram(patcher, vert_id,
		attributes, 2, streams, 1, &tri_vertex_program);

	SceGxmFragmentProgram *tri_fragment_program = NULL;
	sceGxmShaderPatcherCreateFragmentProgram(patcher, frag_id,
		SCE_GXM_OUTPUT_REGISTER_FORMAT_UCHAR4,
		SCE_GXM_MULTISAMPLE_NONE, NULL, vert_program,
		&tri_fragment_program);

	/* --- 8. upload geometry --- */
	SceUID vbo_uid, ibo_uid;
	TriVertex *vertices = gpu_alloc(SCE_KERNEL_MEMBLOCK_TYPE_USER_RW,
		sizeof(tri_vertices), SCE_GXM_MEMORY_ATTRIB_READ, &vbo_uid);
	unsigned short *indices = gpu_alloc(SCE_KERNEL_MEMBLOCK_TYPE_USER_RW,
		sizeof(tri_indices), SCE_GXM_MEMORY_ATTRIB_READ, &ibo_uid);
	for (int i = 0; i < 3; i++)
		vertices[i] = tri_vertices[i];
	for (int i = 0; i < 3; i++)
		indices[i] = tri_indices[i];

	/* --- 9. one offscreen scene, one indexed draw (no uniforms) --- */
	sceGxmBeginScene(context, 0, render_target, NULL, NULL,
		sync, &color_surface, &depth_surface);

	sceGxmSetVertexProgram(context, tri_vertex_program);
	sceGxmSetFragmentProgram(context, tri_fragment_program);
	sceGxmSetVertexStream(context, 0, vertices);
	sceGxmDraw(context, SCE_GXM_PRIMITIVE_TRIANGLES,
		SCE_GXM_INDEX_FORMAT_U16, indices, 3);

	sceGxmEndScene(context, NULL, NULL);
	sceGxmFinish(context);

	/* --- 10. teardown --- */
	sceGxmShaderPatcherReleaseVertexProgram(patcher, tri_vertex_program);
	sceGxmShaderPatcherReleaseFragmentProgram(patcher, tri_fragment_program);
	sceGxmShaderPatcherUnregisterProgram(patcher, vert_id);
	sceGxmShaderPatcherUnregisterProgram(patcher, frag_id);
	sceGxmShaderPatcherDestroy(patcher);
	sceGxmSyncObjectDestroy(sync);
	sceGxmDestroyContext(context);
	sceGxmDestroyRenderTarget(render_target);
	sceGxmTerminate();

	sceKernelExitProcess(0);
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

void _start(void) {
	main();
	for (;;) { }
}
