/*
 * vitaslop conformance corpus: ZERO-TIMEOUT waits - a POLL, not a park - over real
 * Vita NID imports (blob-free).
 *
 * A NULL *timeout and a POINTER TO THE VALUE ZERO are different calls, and this case
 * exists because reading them as the same one was a real, shipped stall:
 *   - NULL           -> park until signalled;
 *   - &(SceUInt){0}  -> DO NOT WAIT. Answer now: SCE_KERNEL_ERROR_WAIT_TIMEOUT
 *                       (0x80028005) if the condition is not already met, 0 if it is.
 *
 * `sematimeout` does NOT cover this. It proves a NON-ZERO timeout expires ('T'), that
 * a satisfied timed wait returns 0 ('S') and that a satisfied NULL wait returns 0
 * ('i') - so it passes unchanged whether a zero timeout polls or parks for ever. That
 * is the exact shape this engine got wrong: a retail title's asset-load thread polled
 * a semaphore through EA::Thread's kTimeoutImmediate and was parked permanently, and
 * the whole suite stayed green.
 *
 * SINGLE-THREADED ON PURPOSE. A poll must not park, so one thread is enough to prove
 * it - and if a poll DOES park, the only thread is parked, the run deadlocks, and the
 * harness sees a non-Finished verdict rather than a wrong string. The failure is loud
 * either way: parked -> no output at all; wrongly satisfied -> a different letter.
 *
 * Covers both primitives whose poll needs no mutex held:
 *   'T' = an empty semaphore polled -> WAIT_TIMEOUT      'S' = then posted, polled -> 0
 *   'U' = a clear event flag polled -> WAIT_TIMEOUT      'V' = then set,    polled -> 0
 *   'M' = main reached the end, which is itself the "it did not park" assertion.
 *   -> "TUSVM".
 *
 * Authored clean-room from the MIT vita-headers API, built -nostdlib.
 */

#include <psp2/kernel/clib.h>
#include <psp2/kernel/threadmgr.h>
#include <psp2/kernel/processmgr.h>
#include <psp2/kernel/error.h>

static int timed_out(int r) {
	return (unsigned int)r == (unsigned int)SCE_KERNEL_ERROR_WAIT_TIMEOUT;
}

int main(void) {
	SceUID sem = sceKernelCreateSema("p", 0, 0, 8, NULL);
	SceUID evf = sceKernelCreateEventFlag("q", 0, 0, NULL);
	unsigned int out = 0;
	int r;

	/* A poll of an EMPTY semaphore. Must answer at once; must not park. */
	SceUInt z = 0;
	r = sceKernelWaitSema(sem, 1, &z);
	sceClibPrintf(timed_out(r) ? "T" : (r == 0 ? "0" : "E"));

	/* A poll of a CLEAR event flag. Same contract. */
	SceUInt z2 = 0;
	r = sceKernelWaitEventFlag(evf, 0x1, SCE_EVENT_WAITAND, &out, &z2);
	sceClibPrintf(timed_out(r) ? "U" : (r == 0 ? "1" : "F"));

	/* The same poll, now satisfied, must SUCCEED - a poll is not a refusal. */
	sceKernelSignalSema(sem, 1);
	SceUInt z3 = 0;
	r = sceKernelWaitSema(sem, 1, &z3);
	sceClibPrintf(r == 0 ? "S" : "e");

	sceKernelSetEventFlag(evf, 0x1);
	SceUInt z4 = 0;
	r = sceKernelWaitEventFlag(evf, 0x1, SCE_EVENT_WAITAND, &out, &z4);
	sceClibPrintf(r == 0 ? "V" : "f");

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
