/*
 * vitaslop conformance corpus: condition-variable BROADCAST over real Vita NID
 * imports (blob-free).
 *
 * Where cond.c proves a single waiter is released by sceKernelSignalCond, this
 * proves sceKernelSignalCondAll wakes EVERY parked waiter, and that each woken
 * waiter re-acquires the shared mutex before it runs (so they serialize, not run
 * concurrently holding the same lock).
 *
 * Three workers each lock the mutex and wait on the condition (each wait releases
 * the mutex, so the next worker can lock and park too). The broadcaster prints 'B',
 * locks the mutex, signals ALL, and unlocks - handing the mutex to the first woken
 * waiter, which prints and unlocks to the next, and so on.
 *   -> "B123M"  (broadcaster, then all three woken in mutex-handoff order, then main).
 * If SignalCondAll woke only one, two workers would stay parked and the join would
 * deadlock; if the woken waiters did not re-acquire the mutex, the handoff order
 * would break.
 *
 * Authored clean-room from the MIT vita-headers API, built -nostdlib.
 */

#include <psp2/kernel/clib.h>
#include <psp2/kernel/threadmgr.h>
#include <psp2/kernel/processmgr.h>

static SceUID mutex;
static SceUID cond;

static int w1(SceSize args, void *argp) { sceKernelLockMutex(mutex, 1, NULL); sceKernelWaitCond(cond, NULL); sceClibPrintf("1"); sceKernelUnlockMutex(mutex, 1); return 0; }
static int w2(SceSize args, void *argp) { sceKernelLockMutex(mutex, 1, NULL); sceKernelWaitCond(cond, NULL); sceClibPrintf("2"); sceKernelUnlockMutex(mutex, 1); return 0; }
static int w3(SceSize args, void *argp) { sceKernelLockMutex(mutex, 1, NULL); sceKernelWaitCond(cond, NULL); sceClibPrintf("3"); sceKernelUnlockMutex(mutex, 1); return 0; }

static int broadcaster(SceSize args, void *argp) {
	sceClibPrintf("B");
	sceKernelLockMutex(mutex, 1, NULL);
	sceKernelSignalCondAll(cond);
	sceKernelUnlockMutex(mutex, 1);
	return 0;
}

int main(void) {
	mutex = sceKernelCreateMutex("m", 0, 0, NULL);
	cond = sceKernelCreateCond("c", 0, mutex, NULL);

	SceUID a = sceKernelCreateThread("1", w1, 0x10000100, 0x10000, 0, 0, NULL);
	SceUID b = sceKernelCreateThread("2", w2, 0x10000100, 0x10000, 0, 0, NULL);
	SceUID c = sceKernelCreateThread("3", w3, 0x10000100, 0x10000, 0, 0, NULL);
	SceUID d = sceKernelCreateThread("B", broadcaster, 0x10000100, 0x10000, 0, 0, NULL);

	/* Start the three waiters first so they all park in WaitCond before broadcast. */
	sceKernelStartThread(a, 0, NULL);
	sceKernelStartThread(b, 0, NULL);
	sceKernelStartThread(c, 0, NULL);
	sceKernelStartThread(d, 0, NULL);

	sceKernelWaitThreadEnd(a, NULL, NULL);
	sceKernelWaitThreadEnd(b, NULL, NULL);
	sceKernelWaitThreadEnd(c, NULL, NULL);
	sceKernelWaitThreadEnd(d, NULL, NULL);
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
