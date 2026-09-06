/*
 * vitaslop conformance corpus: PREEMPTIVE condition variables over real Vita NID
 * imports (blob-free).
 *
 * The sibling mt.c proves semaphore blocking; this proves the condition-variable
 * path, whose defining feature is that a wait RELEASES its mutex and a signal
 * hands the mutex back to the woken thread. A worker locks a mutex and waits on a
 * condition; the main thread later locks the same mutex, signals, and unlocks.
 *
 * A condition has no memory (a signal with no waiter is lost), so - exactly like
 * mt.c's semaphore handshake - the signal must come from a SEPARATE thread that
 * runs after the waiter has parked, not from main inline. The waiter thread starts
 * first, so it parks in sceKernelWaitCond before the signaller runs.
 *
 * The observable order distinguishes the two execution models:
 *   - preemptive scheduler: the waiter parks in sceKernelWaitCond (mutex released);
 *     the signaller runs, prints "B", locks the mutex, signals; the waiter
 *     re-acquires the mutex, wakes, and prints "A"; main (parked joining) then
 *     prints "M"  ->  "BAM".
 *   - synchronous run-to-completion would run the waiter fully at start, where its
 *     wait cannot block, so it prints "A" first  ->  "ABM".
 * So "BAM" is only reachable if the wait truly released the mutex and blocked, and
 * a sibling's signal handed the mutex back.
 *
 * Authored clean-room from the MIT vita-headers API, built -nostdlib.
 */

#include <psp2/kernel/clib.h>
#include <psp2/kernel/threadmgr.h>
#include <psp2/kernel/processmgr.h>

/* Shared across threads via the one guest address space. */
static SceUID mutex;
static SceUID cond;

/* Locks the mutex and waits on the condition; wakes only after the signaller. */
static int waiter(SceSize args, void *argp) {
	sceKernelLockMutex(mutex, 1, NULL);
	sceKernelWaitCond(cond, NULL);
	sceClibPrintf("A");
	sceKernelUnlockMutex(mutex, 1);
	return 0;
}

/* Prints, then signals the condition under the mutex, releasing the waiter. */
static int signaller(SceSize args, void *argp) {
	sceClibPrintf("B");
	sceKernelLockMutex(mutex, 1, NULL);
	sceKernelSignalCond(cond);
	sceKernelUnlockMutex(mutex, 1);
	return 0;
}

int main(void) {
	mutex = sceKernelCreateMutex("m", 0, 0, NULL);
	cond = sceKernelCreateCond("c", 0, mutex, NULL);

	SceUID a = sceKernelCreateThread("A", waiter, 0x10000100, 0x10000, 0, 0, NULL);
	SceUID b = sceKernelCreateThread("B", signaller, 0x10000100, 0x10000, 0, 0, NULL);

	/* Start the waiter first so it parks in WaitCond before the signaller runs. */
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
