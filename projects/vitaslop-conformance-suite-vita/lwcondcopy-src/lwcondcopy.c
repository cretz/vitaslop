/*
 * vitaslop conformance corpus: LIGHTWEIGHT CONDITION VARIABLE waited on through a
 * BYTE COPY of its work area - the clean-room reproduction of a retail 3D title's bug.
 *
 * A lightweight cond keeps its state in the caller's work area (no kernel handle).
 * The kernel identifies the cond by an id stored INSIDE that work area, so a caller
 * may legitimately stage a byte copy of the struct elsewhere (e.g. a C++ condvar
 * wrapper copying its embedded SceKernelLwCondWork onto the stack) and wait on the
 * copy - the wait must resolve to the SAME cond as the original: release its bound
 * lwmutex, park, and be woken by a signal delivered on the ORIGINAL work area.
 *
 * Here the waiter waits on `copy` (a byte copy of `lwc`) while the signaller signals
 * `lwc`. The observable order proves the copy resolved to the same cond:
 *   - correct (copy resolves): "BAM" - the wait on the copy released lwm (so the
 *     signaller could lock it), parked, and the signal on the ORIGINAL woke it.
 *   - copy NOT resolved: the wait on an "unknown" copy either fails to release lwm
 *     (the signaller's lock deadlocks) or parks on a phantom the signal never reaches
 *     (the waiter never wakes) - "BAM" is unreachable either way.
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
	/* Wait on a byte COPY of the cond work area, not the original. The kernel keeps
	 * the cond's identity inside the work area, so the copy must resolve to `lwc`:
	 * release lwm, park, and re-acquire lwm when a signal on the original wakes it. */
	SceKernelLwCondWork copy = lwc;
	sceKernelWaitLwCond(&copy, NULL);
	sceClibPrintf("A");
	sceKernelUnlockLwMutex(&lwm, 1);
	return 0;
}

static int signaller(SceSize args, void *argp) {
	sceClibPrintf("B");
	sceKernelLockLwMutex(&lwm, 1, NULL);   /* must succeed: the copy-wait released it */
	sceKernelSignalLwCond(&lwc);           /* signal the ORIGINAL; must wake the copy waiter */
	sceKernelUnlockLwMutex(&lwm, 1);
	return 0;
}

int main(void) {
	sceKernelCreateLwMutex(&lwm, "lm", 0, 0, NULL);
	sceKernelCreateLwCond(&lwc, "lc", 0, &lwm, NULL);

	SceUID a = sceKernelCreateThread("A", waiter, 0x10000100, 0x10000, 0, 0, NULL);
	SceUID b = sceKernelCreateThread("B", signaller, 0x10000100, 0x10000, 0, 0, NULL);

	/* Start the waiter first so it parks in WaitLwCond (via the copy) before the
	 * signaller runs, avoiding a lost wakeup. */
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
