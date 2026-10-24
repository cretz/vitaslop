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
 *   - a stride that is not a multiple of 4 is refused with INVALID_ALIGNMENT;
 *   - every init boundary a console run recorded: strides up to 131072 round-trip, mip
 *     counts above 13, sizes of 0 or 4097, non-power-of-two swizzled/cube sizes and
 *     unaligned data are refused, NULL data is accepted and then fails Validate;
 *   - a fresh texture reads back CLAMP, normalized, its own mip count and no stride.
 * Expected output: see cases/gxmtex.toml.
 *
 * Authored clean-room from the MIT vita-headers API, built -nostdlib.
 */

#include <psp2/gxm.h>
#include <psp2/kernel/clib.h>
#include <psp2/kernel/processmgr.h>

void *memcpy(void *dst, const void *src, unsigned int n);
/* Exported by SceGxm and named in the NID database, but not declared by the header. */
SceGxmTextureNormalizeMode sceGxmTextureGetNormalizeMode(const SceGxmTexture *texture);

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

	/* The init's accept/reject boundaries, as a homebrew test recorded them on a console. */
	static const unsigned int strides[] = { 4, 512, 1028, 131072 };
	sceClibPrintf("\nstride");
	for (unsigned int i = 0; i < 4; i++) {
		int r = sceGxmTextureInitLinearStrided(&a, pixels, FMT, 16, 16, strides[i]);
		sceClibPrintf(" %x:%u", r, sceGxmTextureGetStride(&a));
	}
	static const unsigned int bad_strides[] = { 2, 6, 0, 131076 };
	for (unsigned int i = 0; i < 4; i++)
		sceClibPrintf(" %x", sceGxmTextureInitLinearStrided(&a, pixels, FMT, 16, 16, bad_strides[i]));

	static const unsigned int mips[] = { 0, 9, 10, 13, 14, 16 };
	sceClibPrintf("\nmips");
	for (unsigned int i = 0; i < 6; i++)
		sceClibPrintf(" %x", sceGxmTextureInitLinear(&a, pixels, FMT, 256, 256, mips[i]));

	sceClibPrintf("\nsize %x %x %x %x %x %x %x",
		sceGxmTextureInitLinear(&a, pixels, FMT, 0, 16, 1),
		sceGxmTextureInitLinear(&a, pixels, FMT, 16, 4097, 1),
		sceGxmTextureInitSwizzled(&a, pixels, FMT, 255, 256, 1),
		sceGxmTextureInitCube(&a, pixels, FMT, 256, 255, 1),
		sceGxmTextureInitSwizzled(&a, pixels, FMT, 4096, 4096, 1),
		sceGxmTextureInitLinear(&a, pixels, FMT, 1, 1, 1),
		sceGxmTextureInitLinear(&a, pixels, FMT, 3, 3, 1));
	sceClibPrintf("\nptr %x %x %x %x",
		sceGxmTextureInitLinear(&a, (char *)pixels + 1, FMT, 16, 16, 1),
		sceGxmTextureInitLinear(0, pixels, FMT, 16, 16, 1),
		sceGxmTextureInitLinear(&a, 0, FMT, 16, 16, 1),
		sceGxmTextureValidate(&a));

	/* What a fresh texture reads back: CLAMP addressing, normalized coordinates, the mip count
	 * passed in (0 included), no stride unless strided, and each layout's own size encoding. */
	sceGxmTextureInitLinear(&a, pixels, FMT, 64, 32, 5);
	sceClibPrintf("\nfresh u%u v%u n%x m%u s%u", sceGxmTextureGetUAddrMode(&a),
		sceGxmTextureGetVAddrMode(&a), sceGxmTextureGetNormalizeMode(&a),
		sceGxmTextureGetMipmapCount(&a), sceGxmTextureGetStride(&a));
	sceGxmTextureInitLinear(&a, pixels, FMT, 64, 32, 0);
	sceClibPrintf(" m%u", sceGxmTextureGetMipmapCount(&a));
	sceGxmTextureInitSwizzled(&a, pixels, FMT, 256, 64, 1);
	sceClibPrintf(" %ux%u", sceGxmTextureGetWidth(&a), sceGxmTextureGetHeight(&a));
	sceGxmTextureSetWidth(&a, 512);
	sceClibPrintf(" %ux%u", sceGxmTextureGetWidth(&a), sceGxmTextureGetHeight(&a));
	sceGxmTextureInitLinearStrided(&a, pixels, FMT, 16, 16, 64);
	sceClibPrintf(" %x", sceGxmTextureSetStride(&a, 65536));
	sceClibPrintf(" %u", sceGxmTextureGetStride(&a));

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
