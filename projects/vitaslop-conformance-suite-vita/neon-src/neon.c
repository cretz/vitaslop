/*
 * vitaslop conformance corpus: NEON auto-vectorization coverage (blob-free).
 *
 * gcc -O2 auto-vectorizes array reductions into NEON data-processing (vmovl /
 * vaddw / vadd.i / vpadd / vabdl / vabal / vpadal / ...). This probe drives that
 * path end to end: each loop runs over a >= 16-element array so the vector body
 * (not just the scalar tail) actually executes, and every result is printed
 * through the trusted sceClibPrintf so the golden is a deterministic transcript.
 * Authored clean-room from the MIT vita-headers API, built -nostdlib. Built WITH
 * the tree vectorizer on (unlike the other probes, which pin -fno-tree-vectorize
 * to stay off the NEON path) - hitting NEON is the whole point.
 */

#include <psp2/kernel/clib.h>
#include <psp2/kernel/processmgr.h>

/* Plain (non-volatile) arrays so the reduction loops auto-vectorize, filled from a
 * volatile seed so the optimizer cannot constant-fold the contents (and thus the
 * results) away. seed == 1, so the comments below give the concrete values. */
static volatile int seed = 1;
static unsigned char ub[32];
static signed short ss[32];
static int ai[32];
static unsigned char pa[32];
static unsigned char pb[32];

/* The fill is deliberately NOT vectorized: narrowing an int product back into a
 * byte/short store would emit vmovn / vmul.i8 (a separate NEON corner from the
 * reduction family this probe targets). Keeping it scalar isolates the reductions. */
__attribute__((noinline, optimize("no-tree-vectorize")))
static void fill(int s) {
	for (int i = 0; i < 32; i++) {
		ub[i] = (unsigned char)((i + 1) * s);   /* 1..32 */
		ss[i] = (signed short)((i - 16) * s);   /* -16..15 */
		ai[i] = (i * 7 - 100) * s;              /* arithmetic spread */
		pa[i] = (unsigned char)(3 * i * s);     /* 0,3,..,93 */
		pb[i] = (unsigned char)((i + 5) * s);   /* 5..36 */
	}
}

int main(void) {
	fill(seed);

	/* 1. unsigned byte sum: vld1.8 -> vmovl.u8 -> vaddw.u16 -> vadd.i32 -> vpadd.i32. */
	int bsum = 0;
	for (int i = 0; i < 32; i++) bsum += ub[i];
	sceClibPrintf("bsum=%d\n", bsum);   /* 1+..+32 = 528 */

	/* 2. signed short sum: vmovl.s16 / vaddw.s16 (signed widen). */
	int ssum = 0;
	for (int i = 0; i < 32; i++) ssum += ss[i];
	sceClibPrintf("ssum=%d\n", ssum);   /* sum(-16..15) = -16 */

	/* 3. int element sum: vadd.i32 accumulation. */
	int isum = 0;
	for (int i = 0; i < 32; i++) isum += ai[i];
	sceClibPrintf("isum=%d\n", isum);   /* 7*(0+..+31) - 100*32 = 7*496 - 3200 = 272 */

	/* 4. sum of absolute differences: vabdl.u8 / vabal.u8 / vpadal.u16. */
	int sad = 0;
	for (int i = 0; i < 32; i++) {
		int d = (int)pa[i] - (int)pb[i];
		sad += d < 0 ? -d : d;
	}
	sceClibPrintf("sad=%d\n", sad);

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
