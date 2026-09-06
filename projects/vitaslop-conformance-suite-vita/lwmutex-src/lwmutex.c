/*
 * vitaslop conformance corpus: LIGHTWEIGHT MUTEX mutual exclusion and blocking over
 * real Vita NID imports (blob-free).
 *
 * A lightweight mutex (SceKernelLwMutexWork, state in the caller's memory - no kernel
 * handle) must still block a contender: while thread A holds it, thread B's lock must
 * park until A unlocks. This is exactly what a stub that "always succeeds" gets wrong,
 * and the observable order tells the two apart:
 *
 *   A locks the lwmutex, prints 'A', then blocks on a semaphore WHILE HOLDING it. B
 *   tries to lock the same lwmutex and must block (A owns it). A releaser posts the
 *   semaphore; A wakes, unlocks (handing the lwmutex to B), and prints 'a'; B - now
 *   holding the lwmutex - prints 'B'; main (joining) prints 'M'.
 *     - real blocking:              "AaBM"  (B is held off until A unlocks)
 *     - "always succeeds" stub:     "ABaM"  (B never blocks, prints before A's 'a')
 * So "AaBM" is only reachable if the lightweight mutex genuinely blocked the
 * contender and handed ownership over on unlock.
 *
 * Authored clean-room from the MIT vita-headers API, built -nostdlib.
 */

#include <psp2/kernel/clib.h>
#include <psp2/kernel/threadmgr.h>
#include <psp2/kernel/processmgr.h>

static SceKernelLwMutexWork lwm;
static SceUID gate;

static int a_thread(SceSize args, void *argp) {
	sceKernelLockLwMutex(&lwm, 1, NULL);
	sceClibPrintf("A");
	sceKernelWaitSema(gate, 1, NULL);   /* hold the lwmutex across a block */
	sceKernelUnlockLwMutex(&lwm, 1);    /* release -> hands it to B */
	sceClibPrintf("a");
	return 0;
}

static int b_thread(SceSize args, void *argp) {
	sceKernelLockLwMutex(&lwm, 1, NULL);   /* must block until A unlocks */
	sceClibPrintf("B");
	sceKernelUnlockLwMutex(&lwm, 1);
	return 0;
}

/* Runs after B has parked on the lwmutex; posts the semaphore to release A. */
static int releaser(SceSize args, void *argp) {
	sceKernelSignalSema(gate, 1);
	return 0;
}

int main(void) {
	sceKernelCreateLwMutex(&lwm, "lm", 0, 0, NULL);
	gate = sceKernelCreateSema("g", 0, 0, 4, NULL);

	SceUID a = sceKernelCreateThread("A", a_thread, 0x10000100, 0x10000, 0, 0, NULL);
	SceUID b = sceKernelCreateThread("B", b_thread, 0x10000100, 0x10000, 0, 0, NULL);
	SceUID r = sceKernelCreateThread("r", releaser, 0x10000100, 0x10000, 0, 0, NULL);

	/* A acquires and blocks holding the lwmutex; B then parks on it; only then does
	 * the releaser (started last, so it runs after both have parked) post the gate. */
	sceKernelStartThread(a, 0, NULL);
	sceKernelStartThread(b, 0, NULL);
	sceKernelStartThread(r, 0, NULL);

	sceKernelWaitThreadEnd(a, NULL, NULL);
	sceKernelWaitThreadEnd(b, NULL, NULL);
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
