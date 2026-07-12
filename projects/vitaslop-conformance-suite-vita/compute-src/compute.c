/*
 * vitaslop conformance corpus: integer compute coverage (blob-free).
 *
 * A CPU-core probe: it drives the transpiler with arithmetic the earlier
 * artifacts did not - 64-bit widening multiply, count-leading-zeros, bit
 * shifts/rotates, and 64-bit add/sub - then prints every result so the golden is
 * a deterministic transcript. Its job is to surface (and, once lifted, certify)
 * the ARM instructions real code emits: UMULL/SMULL, CLZ, and the 64-bit
 * arithmetic helpers. Authored clean-room, built -nostdlib.
 */

#include <psp2/kernel/clib.h>
#include <psp2/kernel/processmgr.h>

/* Kept in separate non-const functions so the compiler cannot constant-fold the
 * whole thing away - the point is to emit real runtime instructions. */
static unsigned long long umul(unsigned int a, unsigned int b) {
	return (unsigned long long)a * (unsigned long long)b; /* UMULL */
}

static long long smul(int a, int b) {
	return (long long)a * (long long)b; /* SMULL */
}

static int clz(unsigned int x) {
	return __builtin_clz(x); /* CLZ */
}

int main(void) {
	/* `volatile` inputs so the optimizer cannot constant-fold these into stored
	 * literals - each operation must emit a real runtime instruction. */
	volatile unsigned int a = 0xFFFFFFFFu, b = 0xFFFFFFFFu;
	volatile int sa = -3, sb = 1000000;
	volatile unsigned int c1 = 1u, c2 = 0x00010000u, c3 = 0x80000000u;
	volatile unsigned int lo = 0xFFFFFFFFu, hi = 0x00000001u;
	volatile unsigned int x = 0x12345678u;

	/* 64-bit widening multiply: 0xFFFFFFFF * 0xFFFFFFFF = 0xFFFFFFFE00000001. */
	sceClibPrintf("umull: %llu\n", umul(a, b));

	/* Signed widening multiply: -3 * 1000000 = -3000000. */
	sceClibPrintf("smull: %lld\n", smul(sa, sb));

	/* Count leading zeros. */
	sceClibPrintf("clz: %d %d %d\n", clz(c1), clz(c2), clz(c3));

	/* 64-bit add and subtract (adc/sbc under the hood). Build the operands from
	 * volatile halves so this is genuine 64-bit arithmetic. */
	unsigned long long wide = ((unsigned long long)hi << 32) | lo; /* 0x1FFFFFFFF */
	sceClibPrintf("wide: sum=%llu dif=%llu\n", wide + 2ULL, wide - 3ULL);

	/* Shifts and a rotate (ror via the classic (x>>n)|(x<<(32-n))). */
	unsigned int rr = (x >> 8) | (x << 24);
	sceClibPrintf("shift: shl=%u shr=%u ror8=0x%x\n", x << 4, x >> 4, rr);

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
