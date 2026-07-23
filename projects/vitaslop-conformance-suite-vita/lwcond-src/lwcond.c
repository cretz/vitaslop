/*
 * vitaslop conformance corpus: LIGHTWEIGHT CONDITION VARIABLE handshake over real
 * Vita NID imports (blob-free) - the lightweight twin of cond.c.
 *
 * A lightweight condition variable is created bound to a lightweight mutex; its
 * defining behaviour is identical to the heavyweight pair: sceKernelWaitLwCond must
 * RELEASE that mutex as it parks and RE-ACQUIRE it when woken. A waiter locks the
 * lwmutex and waits; a signaller locks the same lwmutex, signals, and unlocks,
 * handing the mutex back to the woken waiter.
 *
 *   - correct (wait releases the mutex):  "BAM"  (signaller runs, then the woken
 *     waiter, then main)
 *   - a wait that FAILS to release the mutex would hold it across the wait, so the
 *     signaller's own lock would deadlock the boot.
 * So "BAM" is only reachable if WaitLwCond released the lwmutex and the signal handed
 * it back.
 *
 * Authored clean-room from the MIT vita-headers API, built -nostdlib.
 */

#include <psp2/kernel/clib.h>
#include <psp2/kernel/threadmgr.h>
#include <psp2/kernel/processmgr.h>

static SceKernelLwMutexWork lwm;
static SceKernelLwCondWork lwc;

static int waiter(SceSize args, void *argp) {
	sceKernelLockLwMutex(&lwm, 1, NULL);
	sceKernelWaitLwCond(&lwc, NULL);   /* releases lwm, parks; re-acquires on wake */
	sceClibPrintf("A");
	sceKernelUnlockLwMutex(&lwm, 1);
	return 0;
}

static int signaller(SceSize args, void *argp) {
	sceClibPrintf("B");
	sceKernelLockLwMutex(&lwm, 1, NULL);   /* must succeed: the waiter released it */
	sceKernelSignalLwCond(&lwc);
	sceKernelUnlockLwMutex(&lwm, 1);
	return 0;
}

int main(void) {
	sceKernelCreateLwMutex(&lwm, "lm", 0, 0, NULL);
	sceKernelCreateLwCond(&lwc, "lc", 0, &lwm, NULL);

	SceUID a = sceKernelCreateThread("A", waiter, 0x10000100, 0x10000, 0, 0, NULL);
	SceUID b = sceKernelCreateThread("B", signaller, 0x10000100, 0x10000, 0, 0, NULL);

	/* Start the waiter first so it parks in WaitLwCond before the signaller runs. */
	sceKernelStartThread(a, 0, NULL);
	sceKernelStartThread(b, 0, NULL);

	sceKernelWaitThreadEnd(a, NULL, NULL);
	sceKernelWaitThreadEnd(b, NULL, NULL);
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
