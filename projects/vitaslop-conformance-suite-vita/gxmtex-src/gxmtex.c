/*
 * vitaslop conformance corpus: GXM TEXTURE CONTROL WORDS (blob-free).
 *
 * A SceGxmTexture is 16 bytes a title copies freely, so everything a sampler needs must
 * live in those bytes. This proves:
 *   - a freshly initialised texture has NO LOD bias: the field is excess-31, so the
 *     getter reads 31 ("b31"), and a set bias reads back ("b40");
 *   - a LINEAR_STRIDED texture's byte stride is in its words: a by-value copy reports the
 *     same stride as the original ("3968 3968");
 *   - two strided views of ONE surface (same format, size and data) at different pitches
 *     keep their own stride ("3840", and the first view's copy is still "3968");
 *   - a stride that is not a multiple of 4 is refused with INVALID_ALIGNMENT.
 * Expected output: "l0 b31 b40 t0 3968 3968 t0 3840 3968 e805b0005".
 *
 * Authored clean-room from the MIT vita-headers API, built -nostdlib.
 */

#include <psp2/gxm.h>
#include <psp2/kernel/clib.h>
#include <psp2/kernel/processmgr.h>

void *memcpy(void *dst, const void *src, unsigned int n);

#define FMT SCE_GXM_TEXTURE_FORMAT_U8U8U8U8_ABGR

static unsigned int pixels[64];

int main(void) {
	SceGxmTexture lin, a, copy, b, bad;

	sceClibPrintf("l%d", sceGxmTextureInitLinear(&lin, pixels, FMT, 4, 4, 1));
	sceClibPrintf(" b%u", sceGxmTextureGetLodBias(&lin));
	sceGxmTextureSetLodBias(&lin, 40);
	sceClibPrintf(" b%u", sceGxmTextureGetLodBias(&lin));

	sceClibPrintf(" t%d", sceGxmTextureInitLinearStrided(&a, pixels, FMT, 960, 544, 3968));
	memcpy(&copy, &a, sizeof(copy));
	sceClibPrintf(" %u %u", sceGxmTextureGetStride(&a), sceGxmTextureGetStride(&copy));

	sceClibPrintf(" t%d", sceGxmTextureInitLinearStrided(&b, pixels, FMT, 960, 544, 3840));
	sceClibPrintf(" %u %u", sceGxmTextureGetStride(&b), sceGxmTextureGetStride(&copy));

	sceClibPrintf(" e%x", sceGxmTextureInitLinearStrided(&bad, pixels, FMT, 960, 544, 3966));

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
