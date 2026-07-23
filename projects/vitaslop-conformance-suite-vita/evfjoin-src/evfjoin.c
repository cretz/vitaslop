/*
 * vitaslop conformance corpus: an EVENT-FLAG rendezvous over real Vita NID imports
 * (blob-free) - many workers do work, one thread waits for ALL of them.
 *
 * main waits on an event flag for all three bits (0x1|0x2|0x4 = 0x7) under WAITAND:
 * the wait releases only when EVERY bit is set, so it is a barrier over the three
 * workers. Each worker prints its letter and sets its own bit; the third set
 * completes the pattern and wakes main, which reads 0x7 back through outBits and
 * prints 'D'.
 *   -> "abcD".
 * Under WAITAND the wait must not release on a partial pattern (0x1 or 0x3); a wait
 * that woke early would print 'D' before 'c'. So "abcD" proves the AND semantics and
 * that a set from one thread releases a waiter parked by another.
 *
 * Authored clean-room from the MIT vita-headers API, built -nostdlib.
 */

#include <psp2/kernel/clib.h>
#include <psp2/kernel/threadmgr.h>
#include <psp2/kernel/processmgr.h>

static SceUID evf;

static int wa(SceSize args, void *argp) { sceClibPrintf("a"); sceKernelSetEventFlag(evf, 0x1); return 0; }
static int wb(SceSize args, void *argp) { sceClibPrintf("b"); sceKernelSetEventFlag(evf, 0x2); return 0; }
static int wc(SceSize args, void *argp) { sceClibPrintf("c"); sceKernelSetEventFlag(evf, 0x4); return 0; }

int main(void) {
	evf = sceKernelCreateEventFlag("e", 0, 0, NULL);

	SceUID a = sceKernelCreateThread("a", wa, 0x10000100, 0x10000, 0, 0, NULL);
	SceUID b = sceKernelCreateThread("b", wb, 0x10000100, 0x10000, 0, 0, NULL);
	SceUID c = sceKernelCreateThread("c", wc, 0x10000100, 0x10000, 0, 0, NULL);

	sceKernelStartThread(a, 0, NULL);
	sceKernelStartThread(b, 0, NULL);
	sceKernelStartThread(c, 0, NULL);

	/* Barrier: block until all three worker bits are set. */
	unsigned int out = 0;
	sceKernelWaitEventFlag(evf, 0x7, SCE_EVENT_WAITAND, &out, NULL);
	sceClibPrintf(out == 0x7 ? "D" : "e");

	sceKernelWaitThreadEnd(a, NULL, NULL);
	sceKernelWaitThreadEnd(b, NULL, NULL);
	sceKernelWaitThreadEnd(c, NULL, NULL);

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
