/*
 * vitaslop conformance corpus: the GXM PIPELINE-STATE conformance app.
 *
 * A real Vita executable that drives libgxm through one offscreen SCENE PER
 * FEATURE, each scene drawing geometry chosen so that what should appear is a
 * statement about a RECTANGLE - which the harness can assert with no reference
 * image, no golden capture and no tolerance.
 *
 *   scene 0  BASELINE          one full-viewport quad. Every pixel painted.
 *   scene 1  REGION CLIP       the same quad, clipped to the left half.
 *   scene 2  ABOVE THE VIEWPORT a quad entirely past y = +1. NOTHING painted.
 *   scene 3  DEPTH TEST        a far quad, then a near quad over its right half.
 *   scene 4  ALPHA BLEND       an opaque quad, then a half-alpha quad over its
 *                              right half, with src-alpha / inv-src-alpha.
 *   scene 5  VIEWPORT          the same full quad through a half-size viewport.
 *   scene 6  VIEWPORT + CLIP   both at once, over DIFFERENT rectangles: the
 *                              viewport is the left half, the clip the top
 *                              half, so only the top-left QUARTER may paint.
 *   scene 7  STENCIL MASK      a NEVER quad whose stencil FAIL op REPLACEs bit 0
 *                              over the bottom-right quarter (no colour), then
 *                              a full-viewport EQUAL ref=1 fill: only the
 *                              bottom-right QUARTER may paint, in the fill's
 *                              colour.
 *
 * ---------------------------------------------------------------------------
 * WHY THIS APP EXISTS, GIVEN THE SHADER CONFORMANCE SUITE ALREADY DOES
 *
 * They check different halves and neither substitutes for the other.
 *
 * `vitaslop-gxp-shader/tests/conformance.rs` authors USSE programs and checks
 * that the shader the GPU runs computes what the program MEANS. It says nothing
 * about the state machine around the shader: which pixels the draw is allowed to
 * touch, what the depth test does, how the result is blended into what is there.
 * Every one of those has cost this project a title-level debugging session:
 *
 *   * A fighting title's main screen rendered grey because a REGION CLIP set for
 *     one target was still in force for another, and it happened to FIT.
 *   * A football title's stadium crowd rasterised nothing for three sessions.
 *     Every downstream cause was measured and refuted one at a time - the
 *     sampler, the alpha test, the depth test, the geometry - before a position
 *     probe showed the vertices were simply ABOVE THE TOP OF THE VIEWPORT.
 *     Scene 2 is that defect, stated as a five-millisecond question.
 *
 * ---------------------------------------------------------------------------
 * WHY THE SHADERS ARE PLACEHOLDERS, AND WHY THAT IS THE RIGHT CHOICE HERE
 *
 * Same blob-free stance as `cube.c` and `gxmtri.c`: the `SceGxmProgram`s are
 * magic-only. These draws need NO shader reflection - the vertices are already in
 * normalized device coordinates and there is no uniform buffer, so the renderer
 * recovers each draw from the vertex ATTRIBUTES the app itself declares and from
 * the vertex/index streams.
 *
 * That is not a shortcut, it is what keeps this artifact honest. A container this
 * project WROTE would be read back through this project's own reading of the
 * format, so a field we write wrong we also read wrong and the test would agree
 * with itself. With no blob in the path, everything scene 0-5 asserts comes from
 * REAL vita-headers API calls and nothing else.
 *
 * Authored clean-room from the MIT vita-headers API. Built -nostdlib with a tiny
 * self-contained runtime, so the committed binary is licence-clean and its import
 * surface is Sony NID stubs only.
 */

#include <psp2/types.h>
#include <psp2/kernel/sysmem.h>
#include <psp2/kernel/processmgr.h>
#include <psp2/gxm.h>

#define SURFACE_WIDTH  128
#define SURFACE_HEIGHT 128
#define SURFACE_STRIDE 128

/* The left half, in pixels - what scene 1 clips to and scene 5's viewport covers. */
#define HALF_W (SURFACE_WIDTH / 2)

#define ALIGN(x, a) (((x) + ((a) - 1)) & ~((a) - 1))

static void rt_memset(void *dst, int v, unsigned int n);

/* ---- placeholder shaders (magic only, USSE payload intentionally empty) ---- */
__attribute__((aligned(64)))
static const unsigned char conf_vert_gxp[64] = { 'G', 'X', 'P', 0 };
__attribute__((aligned(64)))
static const unsigned char conf_frag_gxp[64] = { 'G', 'X', 'P', 0 };

/* ---- geometry ----------------------------------------------------------- */
typedef struct {
	float x, y, z;
	unsigned int color; /* 0xAABBGGRR */
} ConfVertex;

/* Every scene draws axis-aligned quads, so the invariant is always a rectangle.
 * Six indices per quad (two triangles), wound the same way throughout so no
 * scene's result depends on a cull mode nobody set. */
static const unsigned short quad_indices[6] = { 0, 1, 2, 2, 1, 3 };

#define RED   0xff0000ffu
#define GREEN 0xff00ff00u
#define BLUE  0xffff0000u
/* Half alpha, pure red - scene 4's blend source. */
#define RED_HALF 0x800000ffu

/* A quad covering [x0,x1] x [y0,y1] in NDC at depth z, one flat colour. Vertex
 * order matches `quad_indices`: (x0,y0), (x1,y0), (x0,y1), (x1,y1). */
static void fill_quad(ConfVertex *v, float x0, float y0, float x1, float y1,
                      float z, unsigned int color) {
	v[0].x = x0; v[0].y = y0; v[0].z = z; v[0].color = color;
	v[1].x = x1; v[1].y = y0; v[1].z = z; v[1].color = color;
	v[2].x = x0; v[2].y = y1; v[2].z = z; v[2].color = color;
	v[3].x = x1; v[3].y = y1; v[3].z = z; v[3].color = color;
}

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

/* >>> EVERY DRAW IN THE APP GETS ITS OWN FOUR VERTICES, AND NOTHING IS EVER REWRITTEN.
 *
 * The first version of this file used ONE four-vertex buffer and refilled it before each
 * scene. Every scene then rendered the LAST data written: the quad that should have sat
 * above the viewport covered the whole target, the depth scene's right half came back as
 * the final scene's red, and the blend scene's two halves were identical. Three of the six
 * scenes were reporting a defect that was entirely this file's.
 *
 * Which of "the capture snapshots a draw's vertices later than the draw" and "the guest may
 * not reuse a buffer inside a frame" is the real rule is a separate question, and a
 * conformance app is the wrong place to be asking it - so there is nothing to alias: eleven
 * draws, forty-four vertices, all written before the first BeginScene. */
#define QUADS        11
#define MAX_VERTICES (QUADS * 4)
/* First vertex of quad `q`, for `sceGxmSetVertexStream`. */
#define QUAD_AT(q) ((q) * 4)

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
	rt_params.scenesPerFrame  = 8;
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
	/* S8D24 carries the STENCIL byte scene 7 masks with, interleaved with depth in the one
	 * buffer (hence no separate stencil pointer). Every scene begins with the stencil
	 * cleared to this background, so scene 7's mark starts from a known all-zero mask. */
	sceGxmDepthStencilSurfaceSetBackgroundStencil(&depth_surface, 0);

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
	const SceGxmProgram *vert_program = (const SceGxmProgram *)conf_vert_gxp;
	const SceGxmProgram *frag_program = (const SceGxmProgram *)conf_frag_gxp;
	sceGxmProgramCheck(vert_program);
	sceGxmProgramCheck(frag_program);

	SceGxmShaderPatcherId vert_id, frag_id;
	sceGxmShaderPatcherRegisterProgram(patcher, vert_program, &vert_id);
	sceGxmShaderPatcherRegisterProgram(patcher, frag_program, &frag_id);

	/* Attribute layout: position float3 @0 reg0, colour u8n x4 @12 reg1. The app
	 * declares these, independently of the (absent) shader, so the capture
	 * recovers every draw from them. */
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
	streams[0].stride      = sizeof(ConfVertex);
	streams[0].indexSource = SCE_GXM_INDEX_SOURCE_INDEX_16BIT;

	SceGxmVertexProgram *vertex_program = NULL;
	sceGxmShaderPatcherCreateVertexProgram(patcher, vert_id,
		attributes, 2, streams, 1, &vertex_program);

	/* Two fragment programs: the ordinary opaque one, and one carrying a real
	 * `SceGxmBlendInfo` - src-alpha over inv-src-alpha, the commonest blend a
	 * title uses and the one scene 4 checks. The blend is a property of the
	 * FRAGMENT PROGRAM on this hardware, not of a context call, which is itself
	 * worth pinning: a renderer that looked for a blend state on the context
	 * would find none and draw every transparent thing opaque. */
	SceGxmFragmentProgram *fragment_program = NULL;
	sceGxmShaderPatcherCreateFragmentProgram(patcher, frag_id,
		SCE_GXM_OUTPUT_REGISTER_FORMAT_UCHAR4,
		SCE_GXM_MULTISAMPLE_NONE, NULL, vert_program,
		&fragment_program);

	SceGxmBlendInfo blend_info;
	rt_memset(&blend_info, 0, sizeof(blend_info));
	blend_info.colorMask = SCE_GXM_COLOR_MASK_ALL;
	blend_info.colorFunc = SCE_GXM_BLEND_FUNC_ADD;
	blend_info.alphaFunc = SCE_GXM_BLEND_FUNC_ADD;
	blend_info.colorSrc  = SCE_GXM_BLEND_FACTOR_SRC_ALPHA;
	blend_info.colorDst  = SCE_GXM_BLEND_FACTOR_ONE_MINUS_SRC_ALPHA;
	blend_info.alphaSrc  = SCE_GXM_BLEND_FACTOR_ONE;
	blend_info.alphaDst  = SCE_GXM_BLEND_FACTOR_ZERO;

	SceGxmFragmentProgram *blend_program = NULL;
	sceGxmShaderPatcherCreateFragmentProgram(patcher, frag_id,
		SCE_GXM_OUTPUT_REGISTER_FORMAT_UCHAR4,
		SCE_GXM_MULTISAMPLE_NONE, &blend_info, vert_program,
		&blend_program);

	/* --- 8. geometry buffers --- */
	SceUID vbo_uid, ibo_uid;
	ConfVertex *vertices = gpu_alloc(SCE_KERNEL_MEMBLOCK_TYPE_USER_RW,
		sizeof(ConfVertex) * MAX_VERTICES, SCE_GXM_MEMORY_ATTRIB_READ, &vbo_uid);
	unsigned short *indices = gpu_alloc(SCE_KERNEL_MEMBLOCK_TYPE_USER_RW,
		sizeof(unsigned short) * 6, SCE_GXM_MEMORY_ATTRIB_READ, &ibo_uid);
	for (int i = 0; i < 6; i++)
		indices[i] = quad_indices[i];

	/* ONE index block, used by every draw: each draw binds its own four vertices
	 * as the stream, so the indices are always 0..3. That also keeps every draw's
	 * captured vertex buffer exactly its own quad. */

	/* --- every quad, written once, before any scene begins --- */
	fill_quad(&vertices[QUAD_AT(0)], -1.0f, -1.0f, 1.0f, 1.0f, 0.5f, GREEN);   /* 0: baseline   */
	fill_quad(&vertices[QUAD_AT(1)], -1.0f, -1.0f, 1.0f, 1.0f, 0.5f, GREEN);   /* 1: clipped    */
	fill_quad(&vertices[QUAD_AT(2)], -1.0f,  1.5f, 1.0f, 2.5f, 0.5f, BLUE);    /* 2: off-screen */
	fill_quad(&vertices[QUAD_AT(3)], -1.0f, -1.0f, 1.0f, 1.0f, 0.9f, RED);     /* 3: depth far  */
	fill_quad(&vertices[QUAD_AT(4)],  0.0f, -1.0f, 1.0f, 1.0f, 0.1f, GREEN);   /* 4: depth near */
	fill_quad(&vertices[QUAD_AT(5)], -1.0f, -1.0f, 1.0f, 1.0f, 0.5f, BLUE);    /* 5: blend dst  */
	fill_quad(&vertices[QUAD_AT(6)],  0.0f, -1.0f, 1.0f, 1.0f, 0.4f, RED_HALF);/* 6: blend src  */
	fill_quad(&vertices[QUAD_AT(7)], -1.0f, -1.0f, 1.0f, 1.0f, 0.5f, RED);     /* 7: viewport   */
	fill_quad(&vertices[QUAD_AT(8)], -1.0f, -1.0f, 1.0f, 1.0f, 0.5f, GREEN);   /* 8: vp + clip  */
	fill_quad(&vertices[QUAD_AT(9)],  0.0f, -1.0f, 1.0f, 0.0f, 0.5f, RED);     /* 9: stencil mark (bottom-right; its red must NEVER appear) */
	fill_quad(&vertices[QUAD_AT(10)], -1.0f, -1.0f, 1.0f, 1.0f, 0.5f, BLUE);   /* 10: stencil fill */

	/* ================================================================== *
	 *  scene 0 - BASELINE: one quad over the whole viewport.
	 *
	 *  Everything below is read against this. If the full quad does not
	 *  cover the target, no later scene's coverage number means anything.
	 * ================================================================== */
	sceGxmBeginScene(context, 0, render_target, NULL, NULL,
		sync, &color_surface, &depth_surface);
	sceGxmSetVertexProgram(context, vertex_program);
	sceGxmSetFragmentProgram(context, fragment_program);
	sceGxmSetVertexStream(context, 0, &vertices[QUAD_AT(0)]);
	sceGxmDraw(context, SCE_GXM_PRIMITIVE_TRIANGLES,
		SCE_GXM_INDEX_FORMAT_U16, indices, 6);
	sceGxmEndScene(context, NULL, NULL);

	/* ================================================================== *
	 *  scene 1 - REGION CLIP: the same full quad, clipped to the left half.
	 *
	 *  A clip belongs to the target it was set for. A fighting title's main
	 *  screen rendered grey because one set for a 128x128 atlas was still in
	 *  force for the frame buffer - and it FIT, so nothing looked wrong
	 *  except the picture.
	 * ================================================================== */
	sceGxmBeginScene(context, 0, render_target, NULL, NULL,
		sync, &color_surface, &depth_surface);
	sceGxmSetVertexProgram(context, vertex_program);
	sceGxmSetFragmentProgram(context, fragment_program);
	sceGxmSetRegionClip(context, SCE_GXM_REGION_CLIP_OUTSIDE,
		0, 0, HALF_W - 1, SURFACE_HEIGHT - 1);
	sceGxmSetVertexStream(context, 0, &vertices[QUAD_AT(1)]);
	sceGxmDraw(context, SCE_GXM_PRIMITIVE_TRIANGLES,
		SCE_GXM_INDEX_FORMAT_U16, indices, 6);
	sceGxmSetRegionClip(context, SCE_GXM_REGION_CLIP_NONE,
		0, 0, SURFACE_WIDTH - 1, SURFACE_HEIGHT - 1);
	sceGxmEndScene(context, NULL, NULL);

	/* ================================================================== *
	 *  scene 2 - ABOVE THE VIEWPORT: a quad entirely past y = +1.
	 *
	 *  >>> THIS IS A FOOTBALL TITLE'S STADIUM CROWD, AS A TEST.
	 *
	 *  That defect took three sessions. Twenty-three draws and 59,202
	 *  indices rasterised NOTHING, every frame, and each candidate cause was
	 *  measured and refused one at a time - the sampler was reading the live
	 *  target, every texel passed the alpha test, depth was not it, the
	 *  geometry expanded correctly offline - before a position probe showed
	 *  the vertices were simply off the top of the screen.
	 *
	 *  The invariant needs no stadium: geometry outside the viewport paints
	 *  NOTHING, and a renderer that clamped it instead would paint a band
	 *  along the top edge.
	 * ================================================================== */
	sceGxmBeginScene(context, 0, render_target, NULL, NULL,
		sync, &color_surface, &depth_surface);
	sceGxmSetVertexProgram(context, vertex_program);
	sceGxmSetFragmentProgram(context, fragment_program);
	sceGxmSetVertexStream(context, 0, &vertices[QUAD_AT(2)]);
	sceGxmDraw(context, SCE_GXM_PRIMITIVE_TRIANGLES,
		SCE_GXM_INDEX_FORMAT_U16, indices, 6);
	sceGxmEndScene(context, NULL, NULL);

	/* ================================================================== *
	 *  scene 3 - DEPTH TEST: a FAR quad over everything, then a NEAR quad
	 *  over its right half. The right half must end up the near quad's
	 *  colour and the left half the far quad's.
	 *
	 *  Stated as two colours rather than one, so that a depth test which
	 *  REJECTED everything and one which accepted everything both fail: the
	 *  first leaves the right half far-coloured, the second is only
	 *  distinguishable because the near draw comes second - which is why the
	 *  scene also runs the pair the other way round would not do.
	 * ================================================================== */
	sceGxmBeginScene(context, 0, render_target, NULL, NULL,
		sync, &color_surface, &depth_surface);
	sceGxmSetVertexProgram(context, vertex_program);
	sceGxmSetFragmentProgram(context, fragment_program);
	sceGxmSetFrontDepthFunc(context, SCE_GXM_DEPTH_FUNC_LESS);
	sceGxmSetFrontDepthWriteEnable(context, SCE_GXM_DEPTH_WRITE_ENABLED);
	sceGxmSetVertexStream(context, 0, &vertices[QUAD_AT(3)]);
	sceGxmDraw(context, SCE_GXM_PRIMITIVE_TRIANGLES,
		SCE_GXM_INDEX_FORMAT_U16, indices, 6);
	sceGxmSetVertexStream(context, 0, &vertices[QUAD_AT(4)]);
	sceGxmDraw(context, SCE_GXM_PRIMITIVE_TRIANGLES,
		SCE_GXM_INDEX_FORMAT_U16, indices, 6);
	sceGxmEndScene(context, NULL, NULL);

	/* ================================================================== *
	 *  scene 4 - ALPHA BLEND: an opaque BLUE quad, then a half-alpha RED
	 *  quad over its right half through the blending fragment program.
	 *
	 *  The overlap must be neither colour: src-alpha over inv-src-alpha at
	 *  alpha 0.5 is the midpoint of the two. A renderer that ignored the
	 *  blend would paint pure red there, and a renderer that dropped the
	 *  second draw would leave pure blue - the two failures a single-colour
	 *  assertion could not tell apart.
	 * ================================================================== */
	sceGxmBeginScene(context, 0, render_target, NULL, NULL,
		sync, &color_surface, &depth_surface);
	sceGxmSetVertexProgram(context, vertex_program);
	sceGxmSetFrontDepthFunc(context, SCE_GXM_DEPTH_FUNC_ALWAYS);
	sceGxmSetFrontDepthWriteEnable(context, SCE_GXM_DEPTH_WRITE_DISABLED);
	sceGxmSetVertexStream(context, 0, &vertices[QUAD_AT(5)]);
	sceGxmSetFragmentProgram(context, fragment_program);
	sceGxmDraw(context, SCE_GXM_PRIMITIVE_TRIANGLES,
		SCE_GXM_INDEX_FORMAT_U16, indices, 6);
	sceGxmSetFragmentProgram(context, blend_program);
	sceGxmSetVertexStream(context, 0, &vertices[QUAD_AT(6)]);
	sceGxmDraw(context, SCE_GXM_PRIMITIVE_TRIANGLES,
		SCE_GXM_INDEX_FORMAT_U16, indices, 6);
	sceGxmEndScene(context, NULL, NULL);

	/* ================================================================== *
	 *  scene 5 - VIEWPORT: the full-screen quad through a viewport covering
	 *  the LEFT HALF of the target.
	 *
	 *  The same visible result as scene 1 by a completely different
	 *  mechanism - the clip discards fragments, the viewport transforms
	 *  coordinates - so the two together separate a renderer that
	 *  implements one and silently ignores the other. The offsets are in
	 *  PIXELS and the scales are half-extents, which is the convention a
	 *  transform read the other way round gets exactly wrong.
	 * ================================================================== */
	sceGxmBeginScene(context, 0, render_target, NULL, NULL,
		sync, &color_surface, &depth_surface);
	sceGxmSetVertexProgram(context, vertex_program);
	sceGxmSetFragmentProgram(context, fragment_program);
	sceGxmSetViewport(context,
		(float)HALF_W / 2.0f, (float)HALF_W / 2.0f,
		(float)SURFACE_HEIGHT / 2.0f, -(float)SURFACE_HEIGHT / 2.0f,
		0.5f, 0.5f);
	sceGxmSetVertexStream(context, 0, &vertices[QUAD_AT(7)]);
	sceGxmDraw(context, SCE_GXM_PRIMITIVE_TRIANGLES,
		SCE_GXM_INDEX_FORMAT_U16, indices, 6);
	sceGxmEndScene(context, NULL, NULL);

	/* ================================================================== *
	 *  scene 6 - VIEWPORT **AND** REGION CLIP, over DIFFERENT rectangles.
	 *
	 *  The viewport covers the LEFT half (as scene 5) and the clip keeps the
	 *  TOP half, so the full-screen quad may only paint the TOP-LEFT
	 *  QUARTER. The hardware applies both bounds, and each scene above
	 *  checks only one of them - which is exactly how a renderer that
	 *  implements one bound and drops the other on the same draw passes
	 *  every earlier scene.
	 *
	 *  It fails four ways and each says something different: the whole
	 *  target means neither bound applied, the left half means the clip was
	 *  lost, the top half means the viewport was lost, and nothing at all
	 *  means the two were composed as an empty rectangle rather than an
	 *  intersection.
	 *
	 *  >>> AND IT IS THE REGRESSION FOR THE VIEWPORT ARRIVING AT ALL. Both
	 *  backends ignored the guest viewport on their fixed-function path
	 *  until scene 5 said so; the fix maps clip space into the viewport
	 *  rectangle AND narrows the draw by it, and those are two claims. This
	 *  scene is the one that can tell them apart.
	 * ================================================================== */
	sceGxmBeginScene(context, 0, render_target, NULL, NULL,
		sync, &color_surface, &depth_surface);
	sceGxmSetVertexProgram(context, vertex_program);
	sceGxmSetFragmentProgram(context, fragment_program);
	sceGxmSetViewport(context,
		(float)HALF_W / 2.0f, (float)HALF_W / 2.0f,
		(float)SURFACE_HEIGHT / 2.0f, -(float)SURFACE_HEIGHT / 2.0f,
		0.5f, 0.5f);
	sceGxmSetRegionClip(context, SCE_GXM_REGION_CLIP_OUTSIDE,
		0, 0, SURFACE_WIDTH - 1, (SURFACE_HEIGHT / 2) - 1);
	sceGxmSetVertexStream(context, 0, &vertices[QUAD_AT(8)]);
	sceGxmDraw(context, SCE_GXM_PRIMITIVE_TRIANGLES,
		SCE_GXM_INDEX_FORMAT_U16, indices, 6);
	sceGxmSetRegionClip(context, SCE_GXM_REGION_CLIP_NONE,
		0, 0, SURFACE_WIDTH - 1, SURFACE_HEIGHT - 1);
	sceGxmEndScene(context, NULL, NULL);

	/* ================================================================== *
	 *  scene 7 - STENCIL MASK: mark a rectangle in the stencil with a draw
	 *  that paints NOTHING, then fill the whole viewport through a test
	 *  that only passes inside the mark.
	 *
	 *  >>> THIS IS A FOOTBALL TITLE'S HUD, AS A TEST.
	 *
	 *  It builds its scoreboard out of stencil masks: `NEVER` quads whose
	 *  stencil FAIL op REPLACEs a bit (colour never written - every
	 *  fragment fails), then `EQUAL ref=1 compare-mask=0x1` draws that may
	 *  only paint inside the cut-out. A renderer with no stencil painted
	 *  every one of those whole - white boxes over the team names.
	 *
	 *  The mark is the BOTTOM-RIGHT quarter, in RED; the fill is BLUE over
	 *  the whole target. Only the bottom-right quarter may end up blue and
	 *  nothing may end up red. Three failures, three meanings: the whole
	 *  target blue means the stencil was ignored entirely; red in the
	 *  quarter means the NEVER test did not reject the mark's colour; no
	 *  blue at all means the fail op never wrote the mask (or EQUAL read it
	 *  against the wrong value).
	 *
	 *  The fail op is the ONLY op that writes here, deliberately: a mark
	 *  that REPLACEd on pass would be the ordinary case and a renderer that
	 *  wired the three ops in the wrong order would still pass it.
	 * ================================================================== */
	sceGxmBeginScene(context, 0, render_target, NULL, NULL,
		sync, &color_surface, &depth_surface);
	sceGxmSetVertexProgram(context, vertex_program);
	sceGxmSetFragmentProgram(context, fragment_program);
	/* Scenes 5 and 6 left a half-width viewport in force on the context; this scene is
	 * about the stencil alone, so it states the full-target viewport and depth state it
	 * draws under rather than inheriting them. */
	sceGxmSetViewport(context,
		(float)SURFACE_WIDTH / 2.0f, (float)SURFACE_WIDTH / 2.0f,
		(float)SURFACE_HEIGHT / 2.0f, -(float)SURFACE_HEIGHT / 2.0f,
		0.5f, 0.5f);
	sceGxmSetFrontDepthFunc(context, SCE_GXM_DEPTH_FUNC_ALWAYS);
	sceGxmSetFrontDepthWriteEnable(context, SCE_GXM_DEPTH_WRITE_DISABLED);
	/* The MARK: every fragment fails, and the fail op writes ref 1 into bit 0. */
	sceGxmSetFrontStencilFunc(context, SCE_GXM_STENCIL_FUNC_NEVER,
		SCE_GXM_STENCIL_OP_REPLACE, SCE_GXM_STENCIL_OP_KEEP, SCE_GXM_STENCIL_OP_KEEP,
		0xff, 0x01);
	sceGxmSetFrontStencilRef(context, 1);
	sceGxmSetVertexStream(context, 0, &vertices[QUAD_AT(9)]);
	sceGxmDraw(context, SCE_GXM_PRIMITIVE_TRIANGLES,
		SCE_GXM_INDEX_FORMAT_U16, indices, 6);
	/* The FILL: passes only where bit 0 is 1, and writes no stencil. */
	sceGxmSetFrontStencilFunc(context, SCE_GXM_STENCIL_FUNC_EQUAL,
		SCE_GXM_STENCIL_OP_KEEP, SCE_GXM_STENCIL_OP_KEEP, SCE_GXM_STENCIL_OP_KEEP,
		0x01, 0x00);
	sceGxmSetFrontStencilRef(context, 1);
	sceGxmSetVertexStream(context, 0, &vertices[QUAD_AT(10)]);
	sceGxmDraw(context, SCE_GXM_PRIMITIVE_TRIANGLES,
		SCE_GXM_INDEX_FORMAT_U16, indices, 6);
	/* Back to the context default (ALWAYS, KEEP x3, masks 0xff, ref 0) so nothing leaks. */
	sceGxmSetFrontStencilFunc(context, SCE_GXM_STENCIL_FUNC_ALWAYS,
		SCE_GXM_STENCIL_OP_KEEP, SCE_GXM_STENCIL_OP_KEEP, SCE_GXM_STENCIL_OP_KEEP,
		0xff, 0xff);
	sceGxmSetFrontStencilRef(context, 0);
	sceGxmEndScene(context, NULL, NULL);

	sceGxmFinish(context);

	/* --- teardown --- */
	sceGxmShaderPatcherReleaseFragmentProgram(patcher, blend_program);
	sceGxmShaderPatcherReleaseFragmentProgram(patcher, fragment_program);
	sceGxmShaderPatcherReleaseVertexProgram(patcher, vertex_program);
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
