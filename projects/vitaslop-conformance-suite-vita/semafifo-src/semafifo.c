/*
 * vitaslop conformance corpus: multi-waiter SEMAPHORE release order over real Vita
 * NID imports (blob-free).
 *
 * Where mt.c proves a single waiter blocks and is woken by a sibling, this proves
 * the MULTI-waiter discipline: three workers all park in sceKernelWaitSema on an
 * empty counting semaphore, and one producer posts three single signals. The
 * kernel releases parked waiters in FIFO order (the order they blocked), and each
 * signal hands exactly one permit to exactly one waiter (the count is consumed, not
 * left to over-release).
 *
 * All four workers run at the default priority, so once main blocks (joining) the
 * scheduler runs them round-robin in start order: w1,w2,w3 each park in WaitSema,
 * then the producer prints 'p' and posts three permits, waking w1,w2,w3 in that
 * FIFO order; they resume and print '1','2','3'; main (joining) then prints 'M'.
 *   -> "p123M".
 * A broken release (LIFO, or one signal waking several) would reorder or drop a
 * digit; a leaked permit would let a fourth (absent) waiter through. So "p123M" is
 * only reachable if each post released exactly the next FIFO waiter.
 *
 * Authored clean-room from the MIT vita-headers API, built -nostdlib.
 */

#include <psp2/kernel/clib.h>
#include <psp2/kernel/threadmgr.h>
#include <psp2/kernel/processmgr.h>

static SceUID sema;

static int w1(SceSize args, void *argp) { sceKernelWaitSema(sema, 1, NULL); sceClibPrintf("1"); return 0; }
static int w2(SceSize args, void *argp) { sceKernelWaitSema(sema, 1, NULL); sceClibPrintf("2"); return 0; }
static int w3(SceSize args, void *argp) { sceKernelWaitSema(sema, 1, NULL); sceClibPrintf("3"); return 0; }

/* Posts three permits, one at a time, releasing the three parked waiters in turn. */
static int producer(SceSize args, void *argp) {
	sceClibPrintf("p");
	sceKernelSignalSema(sema, 1);
	sceKernelSignalSema(sema, 1);
	sceKernelSignalSema(sema, 1);
	return 0;
}

int main(void) {
	sema = sceKernelCreateSema("s", 0, 0, 8, NULL);

	SceUID a = sceKernelCreateThread("1", w1, 0x10000100, 0x10000, 0, 0, NULL);
	SceUID b = sceKernelCreateThread("2", w2, 0x10000100, 0x10000, 0, 0, NULL);
	SceUID c = sceKernelCreateThread("3", w3, 0x10000100, 0x10000, 0, 0, NULL);
	SceUID p = sceKernelCreateThread("p", producer, 0x10000100, 0x10000, 0, 0, NULL);

	/* Start the three waiters first (in order) so they park FIFO before the
	 * producer runs and posts. */
	sceKernelStartThread(a, 0, NULL);
	sceKernelStartThread(b, 0, NULL);
	sceKernelStartThread(c, 0, NULL);
	sceKernelStartThread(p, 0, NULL);

	sceKernelWaitThreadEnd(a, NULL, NULL);
	sceKernelWaitThreadEnd(b, NULL, NULL);
	sceKernelWaitThreadEnd(c, NULL, NULL);
	sceKernelWaitThreadEnd(p, NULL, NULL);
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
