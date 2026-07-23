/*
 * vitaslop conformance corpus: COUNTING-semaphore accumulation over real Vita NID
 * imports (blob-free).
 *
 * A Vita semaphore is a counting semaphore, not a binary latch: signals accumulate,
 * a wait for N blocks until the count REACHES N (partial signals do not release it),
 * and a satisfied wait consumes exactly N, leaving the remainder for the next wait.
 * This proves all three.
 *
 * The taker parks needing 3; the giver posts 2 (not enough - the taker stays
 * blocked), then 1 (now 3 - the taker wakes, consuming all 3), then 2 more. The
 * taker's second wait needs only 2 and the leftover count is exactly 2, so it is
 * satisfied immediately without blocking.
 *   giver prints 'g'; taker wakes and prints 'X' (first wait, needed 3), then 'Y'
 *   (second wait, needed 2, taken from the leftover); main (joining) prints 'M'.
 *   -> "gXYM".
 * If a partial post (2 < 3) wrongly released the need-3 wait, or the count were not
 * consumed/left correctly, the order or the second wait would break.
 *
 * Authored clean-room from the MIT vita-headers API, built -nostdlib.
 */

#include <psp2/kernel/clib.h>
#include <psp2/kernel/threadmgr.h>
#include <psp2/kernel/processmgr.h>

static SceUID sema;

static int taker(SceSize args, void *argp) {
	sceKernelWaitSema(sema, 3, NULL);   /* blocks until the count reaches 3 */
	sceClibPrintf("X");
	sceKernelWaitSema(sema, 2, NULL);   /* satisfied by the leftover 2, no block */
	sceClibPrintf("Y");
	return 0;
}

static int giver(SceSize args, void *argp) {
	sceClibPrintf("g");
	sceKernelSignalSema(sema, 2);   /* count 2: NOT enough for the need-3 waiter */
	sceKernelSignalSema(sema, 1);   /* count 3: releases the waiter (consumes 3) */
	sceKernelSignalSema(sema, 2);   /* count 2: satisfies the taker's second wait */
	return 0;
}

int main(void) {
	sema = sceKernelCreateSema("s", 0, 0, 8, NULL);

	SceUID t = sceKernelCreateThread("t", taker, 0x10000100, 0x10000, 0, 0, NULL);
	SceUID g = sceKernelCreateThread("g", giver, 0x10000100, 0x10000, 0, 0, NULL);

	/* Start the taker first so it parks needing 3 before the giver posts. */
	sceKernelStartThread(t, 0, NULL);
	sceKernelStartThread(g, 0, NULL);

	sceKernelWaitThreadEnd(t, NULL, NULL);
	sceKernelWaitThreadEnd(g, NULL, NULL);
	sceClibPrintf("M");

	sceKernelExitProcess(0);
	return 0;
}

/* ======================================================================= *
 *  Tiny freestanding runtime (-nostdlib).
 * ======================================================================= */
void *memcpy(void *dst, const void *src, unsigned int n) {
	unsigned char *d = (unsigned char *)dst;
	const unsigned char *s = (const unsigned char *)src;
	for (unsigned int i = 0; i < n; i++) d[i] = s[i];
	return dst;
}
void *memset(void *dst, int v, unsigned int n) {
	unsigned char *p = (unsigned char *)dst;
	for (unsigned int i = 0; i < n; i++) p[i] = (unsigned char)v;
	return dst;
}
void _start(void) { main(); for (;;) { } }
