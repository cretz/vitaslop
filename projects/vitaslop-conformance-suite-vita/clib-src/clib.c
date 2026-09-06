/*
 * vitaslop conformance corpus: SceLibKernel clib string/memory coverage
 * (blob-free).
 *
 * Where hello.c stresses the variadic printf path, this exercises the pure clib
 * memory and string host calls end to end: it runs each one on real guest memory
 * and prints the observable result through sceClibPrintf (already trusted), so
 * the golden is a deterministic transcript. Authored clean-room from the MIT
 * vita-headers API, built -nostdlib (self-contained runtime at the bottom).
 */

#include <psp2/kernel/clib.h>
#include <psp2/kernel/processmgr.h>

/* Normalize a comparison result to -1/0/1: the sign is defined by C, the exact
 * magnitude is not, so the golden compares only the sign. */
static int sgn(int x) {
	return x < 0 ? -1 : (x > 0 ? 1 : 0);
}

int main(void) {
	char buf[64];
	char dst[64];

	/* memset: fill 8 bytes with 'A', terminate, print. */
	sceClibMemset(buf, 'A', 8);
	buf[8] = 0;
	sceClibPrintf("memset: %s\n", buf);

	/* memcpy: copy a string including its NUL, print. */
	sceClibMemcpy(dst, "copied text", 12);
	sceClibPrintf("memcpy: %s\n", dst);

	/* memcmp: unequal (negative) and equal. */
	sceClibPrintf("memcmp: neg=%d eq=%d\n",
		sgn(sceClibMemcmp("abc", "abd", 3)),
		sgn(sceClibMemcmp("abc", "abc", 3)));

	/* strnlen: full and capped. */
	sceClibPrintf("strnlen: %d %d\n",
		(int)sceClibStrnlen("hello", 64),
		(int)sceClibStrnlen("hello", 3));

	/* strncpy: copies "hi" and zero-fills the rest of the 6-byte field. The pad
	 * byte at index 5 must be 0. */
	sceClibMemset(dst, '#', sizeof(dst));
	sceClibStrncpy(dst, "hi", 6);
	sceClibPrintf("strncpy: [%s] pad=%d\n", dst, (int)dst[5]);

	/* strcmp / strncmp: sign only. */
	sceClibPrintf("strcmp: %d %d\n",
		sgn(sceClibStrcmp("abc", "abc")),
		sgn(sceClibStrcmp("abc", "abd")));
	sceClibPrintf("strncmp: %d\n", sgn(sceClibStrncmp("abcXY", "abcZZ", 4)));

	/* snprintf: normal, then a truncating call (size 4 holds 3 chars + NUL) that
	 * still returns the full would-be length. */
	char sn[32];
	int n = sceClibSnprintf(sn, sizeof(sn), "n=%d s=%s", 7, "ok");
	sceClibPrintf("snprintf: [%s] ret=%d\n", sn, n);
	int n2 = sceClibSnprintf(sn, 4, "%d%d%d%d", 1, 2, 3, 4);
	sceClibPrintf("snprintf trunc: [%s] ret=%d\n", sn, n2);

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
