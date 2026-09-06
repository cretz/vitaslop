/*
 * vitaslop conformance corpus: ScePvf vector-font engine (blob-free source).
 *
 * Drives the font host module end to end over the host virtual filesystem:
 *   - scePvfNewLib / scePvfSetResolution / scePvfSetEM,
 *   - scePvfOpenUserFile on a preloaded font file (the first case to read a real
 *     data file the way a retail title does),
 *   - scePvfSetCharSize, scePvfIsElement,
 *   - scePvfGetCharInfo / scePvfGetFontInfo (metrics),
 *   - scePvfGetCharGlyphImage (rasterize a glyph into a caller buffer),
 *   - scePvfDoneLib.
 *
 * The font is Ahem (public domain / CC0), chosen for its predictable geometry:
 * every glyph advances exactly 1 em and most glyphs are a solid box from 0.8 em
 * above the baseline to 0.2 em below. So at a 16 px em the horizontal advance is
 * exactly 16 px (1024 in 26.6 fixed point) and the 'X' glyph is a solid coverage
 * rectangle - values the golden transcript pins exactly.
 *
 * Every observable result is printed through sceClibPrintf, so the golden is a
 * deterministic transcript. Authored clean-room from the MIT vita-headers API,
 * built -nostdlib (self-contained runtime at the bottom).
 *
 * The harness preloads "app0:/ahem.ttf" before the run.
 */

#include <psp2/pvf.h>
#include <psp2/kernel/clib.h>
#include <psp2/kernel/processmgr.h>

int main(void) {
	ScePvfError err = 0;

	/* 1. Create the library. */
	ScePvfInitRec init;
	((unsigned char *)&init)[0] = 0; /* touch so the struct is materialized */
	for (unsigned int i = 0; i < sizeof(init); i++) ((unsigned char *)&init)[i] = 0;
	init.maxNumFonts = 4;
	ScePvfLibId lib = scePvfNewLib(&init, &err);
	sceClibPrintf("lib_ok=%d err=%d\n", lib != 0, (int)err);

	/* Resolution 72 dpi so pixel<->point is identity; em value exercised too. */
	scePvfSetResolution(lib, 72.0f, 72.0f);
	scePvfSetEM(lib, 72.0f);

	/* 2. Open the preloaded font file through the host filesystem. */
	err = 0;
	ScePvfFontId font = scePvfOpenUserFile(lib, "app0:/ahem.ttf", SCE_PVF_FILEBASEDSTREAM, &err);
	sceClibPrintf("font_ok=%d err=%d\n", font != 0, (int)err);

	/* 3. A 16 px em. */
	ScePvfError sz = scePvfSetCharSize(font, 16.0f, 16.0f);
	sceClibPrintf("setsize_ret=%d\n", (int)sz);

	/* 4. Element coverage: 'X' is present, a C0 control code (BEL) is not. */
	sceClibPrintf("isX=%d isBEL=%d\n",
		(int)scePvfIsElement(font, (ScePvfCharCode)'X'),
		(int)scePvfIsElement(font, (ScePvfCharCode)0x0007));

	/* 5. Per-glyph metrics for 'X'. */
	ScePvfCharInfo ci;
	for (unsigned int i = 0; i < sizeof(ci); i++) ((unsigned char *)&ci)[i] = 0;
	ScePvfError cr = scePvfGetCharInfo(font, (ScePvfCharCode)'X', &ci);
	sceClibPrintf("charinfo ret=%d w=%d h=%d adv64=%d left=%d top=%d\n",
		(int)cr, (int)ci.bitmapWidth, (int)ci.bitmapHeight,
		(int)ci.glyphMetrics.horizontalAdvance64, (int)ci.bitmapLeft, (int)ci.bitmapTop);

	/* 6. Face-wide metrics. */
	ScePvfFontInfo fi;
	for (unsigned int i = 0; i < sizeof(fi); i++) ((unsigned char *)&fi)[i] = 0;
	ScePvfError fr = scePvfGetFontInfo(font, &fi);
	sceClibPrintf("fontinfo ret=%d numchars=%d maxadv64=%d\n",
		(int)fr, (int)fi.numChars, (int)fi.maxIGlyphMetrics.horizontalAdvance64);

	/* 7. Rasterize 'X' into a 64x64 8-bit buffer, pen baseline at (0, 32). */
	static unsigned char glyphbuf[64 * 64];
	for (int i = 0; i < 64 * 64; i++) glyphbuf[i] = 0;
	ScePvfUserImageBufferRec img;
	for (unsigned int i = 0; i < sizeof(img); i++) ((unsigned char *)&img)[i] = 0;
	img.pixelFormat = SCE_PVF_USERIMAGE_DIRECT8;
	img.xPos64 = 0 << 6;
	img.yPos64 = 32 << 6;
	img.rect.width = 64;
	img.rect.height = 64;
	img.bytesPerLine = 64;
	img.buffer = glyphbuf;
	ScePvfError gr = scePvfGetCharGlyphImage(font, (ScePvfCharCode)'X', &img);

	/* Count solidly-covered pixels and sample a point inside the box. The box for
	 * 'X' at 16 px spans x 0..16 and y ~19..36 (baseline 32, 0.8 em up / 0.2 down),
	 * so (col 8, row 27) is well inside. */
	int filled = 0, near = 0;
	for (int i = 0; i < 64 * 64; i++) {
		if (glyphbuf[i] == 255) filled++;
		if (glyphbuf[i] >= 200) near++;
	}
	sceClibPrintf("glyph ret=%d filled=%d near=%d center=%d\n",
		(int)gr, filled, near, (int)glyphbuf[27 * 64 + 8]);

	/* 8. Teardown. */
	sceClibPrintf("donelib=%d\n", (int)scePvfDoneLib(lib));

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
