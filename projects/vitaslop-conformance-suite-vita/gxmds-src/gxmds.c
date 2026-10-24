/*
 * vitaslop conformance corpus: GXM DEPTH-STENCIL SURFACE (blob-free).
 *
 * A SceGxmDepthStencilSurface is 20 bytes a title copies freely, so every accessor must read
 * the struct itself. This proves:
 *   - init refuses a stride of 0 or one that is not a multiple of 32 (INVALID_VALUE);
 *   - a fresh surface reads back its format, its stride in samples, a background depth of
 *     1.0, background stencil 0, the background mask set, no forced load or store, enabled;
 *   - every setter is read back by its getter, on the original and on a by-value copy;
 *   - a disabled surface is not enabled, and keeps the same background defaults.
 * Expected output: see cases/gxmds.toml.
 *
 * Authored clean-room from the MIT vita-headers API, built -nostdlib.
 */

#include <psp2/gxm.h>
#include <psp2/kernel/clib.h>
#include <psp2/kernel/processmgr.h>

void *memcpy(void *dst, const void *src, unsigned int n);
/* Exported by SceGxm and named in the NID database, but not declared by the header. */
SceBool sceGxmDepthStencilSurfaceGetBackgroundMask(const SceGxmDepthStencilSurface *surface);
void sceGxmDepthStencilSurfaceSetBackgroundMask(SceGxmDepthStencilSurface *surface, SceBool mask);

static unsigned int depth[64];

static void show(const SceGxmDepthStencilSurface *s) {
	float d = sceGxmDepthStencilSurfaceGetBackgroundDepth(s);
	sceClibPrintf(" f%x st%u d%d s%u m%d l%x w%x e%d", sceGxmDepthStencilSurfaceGetFormat(s),
		sceGxmDepthStencilSurfaceGetStrideInSamples(s), (int)(d * 1000.0f),
		sceGxmDepthStencilSurfaceGetBackgroundStencil(s), sceGxmDepthStencilSurfaceGetBackgroundMask(s),
		sceGxmDepthStencilSurfaceGetForceLoadMode(s), sceGxmDepthStencilSurfaceGetForceStoreMode(s),
		sceGxmDepthStencilSurfaceIsEnabled(s));
}

int main(void) {
	SceGxmDepthStencilSurface a, copy, off;

	sceClibPrintf("init %x %x %x",
		sceGxmDepthStencilSurfaceInit(&a, SCE_GXM_DEPTH_STENCIL_FORMAT_S8D24,
			SCE_GXM_DEPTH_STENCIL_SURFACE_TILED, 0, depth, 0),
		sceGxmDepthStencilSurfaceInit(&a, SCE_GXM_DEPTH_STENCIL_FORMAT_S8D24,
			SCE_GXM_DEPTH_STENCIL_SURFACE_TILED, 48, depth, 0),
		sceGxmDepthStencilSurfaceInit(&a, SCE_GXM_DEPTH_STENCIL_FORMAT_S8D24,
			SCE_GXM_DEPTH_STENCIL_SURFACE_TILED, 960, depth, 0));

	sceClibPrintf("\nfresh");
	show(&a);

	sceGxmDepthStencilSurfaceSetBackgroundDepth(&a, 0.25f);
	sceGxmDepthStencilSurfaceSetBackgroundStencil(&a, 0x5a);
	sceGxmDepthStencilSurfaceSetBackgroundMask(&a, 0);
	sceGxmDepthStencilSurfaceSetForceLoadMode(&a, SCE_GXM_DEPTH_STENCIL_FORCE_LOAD_ENABLED);
	sceGxmDepthStencilSurfaceSetForceStoreMode(&a, SCE_GXM_DEPTH_STENCIL_FORCE_STORE_ENABLED);
	memcpy(&copy, &a, sizeof(copy));
	sceClibPrintf("\nset");
	show(&a);
	sceClibPrintf("\ncopy");
	show(&copy);

	sceGxmDepthStencilSurfaceInit(&a, SCE_GXM_DEPTH_STENCIL_FORMAT_DF32M,
		SCE_GXM_DEPTH_STENCIL_SURFACE_LINEAR, 32, depth, 0);
	sceClibPrintf("\nlinear");
	show(&a);

	sceGxmDepthStencilSurfaceInitDisabled(&off);
	sceClibPrintf("\noff");
	show(&off);

	sceKernelExitProcess(0);
	return 0;
}

/* ======================================================================= *
 *  Tiny freestanding runtime (-nostdlib).
 * ======================================================================= */

void *memcpy(void *dst, const void *src, unsigned int n) {
	unsigned char *d = (unsigned char *)dst;
	const unsigned char *s = (const unsigned char *)src;
	for (unsigned int i = 0; i < n; i++)
		d[i] = s[i];
	return dst;
}

void *memset(void *dst, int v, unsigned int n) {
	unsigned char *p = (unsigned char *)dst;
	for (unsigned int i = 0; i < n; i++)
		p[i] = (unsigned char)v;
	return dst;
}

void _start(void) {
	main();
	for (;;) { }
}
