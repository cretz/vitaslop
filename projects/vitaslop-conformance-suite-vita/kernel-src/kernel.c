/*
 * vitaslop conformance corpus: SceLibKernel synchronization + timing primitives
 * (blob-free).
 *
 * Rounds out the kernel-basics NID surface any real title needs after
 * print/mem/thread: a mutex (create/lock/unlock/delete), a semaphore
 * (create/wait/signal/delete), an event flag (create/set/wait/read/clear/delete),
 * and the wide system clock. Each call's observable result is printed through
 * sceClibPrintf, so the golden is a deterministic transcript.
 *
 * Bring-up model note: with one thread of control (workers run synchronously to
 * completion) nothing contends, so lock/wait always succeed immediately - correct
 * for single-thread use. The event-flag path (set a pattern, wait, read it back)
 * is fully observable. Authored clean-room from the MIT vita-headers API,
 * built -nostdlib.
 */

#include <psp2/kernel/clib.h>
#include <psp2/kernel/threadmgr.h>
#include <psp2/kernel/processmgr.h>

int main(void) {
	/* --- mutex --- */
	SceUID mtx = sceKernelCreateMutex("mtx", 0, 0, NULL);
	int lock = sceKernelLockMutex(mtx, 1, NULL);
	int unlock = sceKernelUnlockMutex(mtx, 1);
	sceClibPrintf("mutex: id_ok=%d lock=%d unlock=%d\n", mtx >= 0, lock, unlock);
	sceKernelDeleteMutex(mtx);

	/* --- semaphore --- */
	SceUID sema = sceKernelCreateSema("sema", 0, 2, 10, NULL);
	int waits = sceKernelWaitSema(sema, 1, NULL);
	int signals = sceKernelSignalSema(sema, 3);
	sceClibPrintf("sema: id_ok=%d wait=%d signal=%d\n", sema >= 0, waits, signals);
	sceKernelDeleteSema(sema);

	/* --- event flag: set a pattern, wait for it, read it back --- */
	SceUID evf = sceKernelCreateEventFlag("evf", 0, 0x0, NULL);
	sceKernelSetEventFlag(evf, 0x5);
	unsigned int out_bits = 0;
	sceKernelWaitEventFlag(evf, 0x5, 0, &out_bits, NULL);
	sceClibPrintf("eventflag: id_ok=%d pattern=0x%x\n", evf >= 0, out_bits);
	sceKernelDeleteEventFlag(evf);

	/* --- system time: monotonic (never goes backward) --- */
	SceUInt64 t1 = sceKernelGetSystemTimeWide();
	SceUInt64 t2 = sceKernelGetSystemTimeWide();
	sceClibPrintf("time: monotonic=%d\n", t2 >= t1);

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
