/*
 * vitaslop conformance corpus: condition variables have NO MEMORY - a signal
 * delivered with no waiter parked is LOST - over real Vita NID imports (blob-free).
 *
 * This is the property that distinguishes a condition variable from a semaphore
 * (whose signals accumulate). A poster signals the condition BEFORE any thread is
 * waiting; that signal must be dropped, so the waiter that parks afterward is
 * released only by a LATER signal, never by the earlier lost one.
 *
 * Sequencing (deterministic under the strict-priority scheduler):
 *   - a gate semaphore holds the waiter until the poster has done its early signal;
 *   - the poster locks, signals the condition with no one waiting (LOST), unlocks,
 *     prints 'L', and posts the gate;
 *   - the waiter (higher priority than the releaser) then locks and parks in
 *     WaitCond - it MUST block, because the earlier signal was lost;
 *   - the releaser (lower priority, so it runs only once the waiter has parked)
 *     locks, signals, prints 'r', and unlocks - THIS is what releases the waiter,
 *     which prints 'w'; main (joining) prints 'M'.
 *   -> "LrwM".
 * If the early signal had been remembered, the waiter's WaitCond would return at
 * once and 'w' would print before 'r' ("Lw..."). So "LrwM" proves the signal was
 * lost and only the later one released the waiter.
 *
 * Authored clean-room from the MIT vita-headers API, built -nostdlib.
 */

#include <psp2/kernel/clib.h>
#include <psp2/kernel/threadmgr.h>
#include <psp2/kernel/processmgr.h>

static SceUID mutex;
static SceUID cond;
static SceUID gate;

static int waiter(SceSize args, void *argp) {
	sceKernelWaitSema(gate, 1, NULL);        /* until the poster's lost signal is done */
	sceKernelLockMutex(mutex, 1, NULL);
	sceKernelWaitCond(cond, NULL);           /* MUST block: earlier signal was lost */
	sceClibPrintf("w");
	sceKernelUnlockMutex(mutex, 1);
	return 0;
}

static int poster(SceSize args, void *argp) {
	sceKernelLockMutex(mutex, 1, NULL);
	sceKernelSignalCond(cond);               /* no waiter parked yet -> LOST */
	sceKernelUnlockMutex(mutex, 1);
	sceClibPrintf("L");
	sceKernelSignalSema(gate, 1);            /* let the waiter proceed to WaitCond */
	return 0;
}

static int releaser(SceSize args, void *argp) {
	sceKernelLockMutex(mutex, 1, NULL);
	sceKernelSignalCond(cond);               /* waiter is parked now -> releases it */
	sceClibPrintf("r");
	sceKernelUnlockMutex(mutex, 1);
	return 0;
}

int main(void) {
	mutex = sceKernelCreateMutex("m", 0, 0, NULL);
	cond = sceKernelCreateCond("c", 0, mutex, NULL);
	gate = sceKernelCreateSema("g", 0, 0, 4, NULL);

	SceUID w = sceKernelCreateThread("w", waiter,   0x10000100, 0x10000, 0, 0, NULL);
	SceUID p = sceKernelCreateThread("p", poster,   0x10000100, 0x10000, 0, 0, NULL);
	/* Lower priority (sentinel + 1) so the releaser runs only after the higher-
	 * priority waiter has parked in WaitCond. */
	SceUID r = sceKernelCreateThread("r", releaser, 0x10000101, 0x10000, 0, 0, NULL);

	sceKernelStartThread(w, 0, NULL);
	sceKernelStartThread(p, 0, NULL);
	sceKernelStartThread(r, 0, NULL);

	sceKernelWaitThreadEnd(w, NULL, NULL);
	sceKernelWaitThreadEnd(p, NULL, NULL);
	sceKernelWaitThreadEnd(r, NULL, NULL);
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
