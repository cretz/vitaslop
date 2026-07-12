/*
 * vitaslop conformance corpus: PREEMPTIVE multithreading over real Vita NID
 * imports (blob-free).
 *
 * Where thread-src proves the create/start/wait shape under the single-thread-of-
 * control bring-up (workers run synchronously at start), this proves REAL
 * concurrency: one thread blocks in sceKernelWaitSema on an empty semaphore and is
 * only released by ANOTHER thread's sceKernelSignalSema. The observable order
 * distinguishes the two execution models:
 *   - preemptive scheduler: the waiter parks; the signaller runs and prints "B",
 *     then signals; the waiter wakes and prints "A"; main (parked joining) then
 *     prints "M"  ->  "BAM".
 *   - synchronous run-to-completion would instead run the waiter fully at start
 *     (its wait cannot block, so it prints "A" first)  ->  "ABM".
 * So "BAM" is only reachable if the wait genuinely blocked and a sibling woke it.
 *
 * Also exercises: two address-taken thread entries discovered as code pointers, a
 * shared static (`sema`) written by main and read by both workers through the one
 * shared address space, and sceKernelWaitThreadEnd blocking until each ends.
 *
 * Authored clean-room from the MIT vita-headers API, built -nostdlib.
 */

#include <psp2/kernel/clib.h>
#include <psp2/kernel/threadmgr.h>
#include <psp2/kernel/processmgr.h>

/* Shared across all three threads via the one guest address space. */
static SceUID sema;

/* Blocks until the signaller posts the semaphore, then prints. */
static int waiter(SceSize args, void *argp) {
	sceKernelWaitSema(sema, 1, NULL);
	sceClibPrintf("A");
	return 0;
}

/* Prints, then posts the semaphore, releasing the waiter. */
static int signaller(SceSize args, void *argp) {
	sceClibPrintf("B");
	sceKernelSignalSema(sema, 1);
	return 0;
}

int main(void) {
	sema = sceKernelCreateSema("s", 0, 0, 1, NULL);

	SceUID a = sceKernelCreateThread("A", waiter, 0x10000100, 0x10000, 0, 0, NULL);
	SceUID b = sceKernelCreateThread("B", signaller, 0x10000100, 0x10000, 0, 0, NULL);

	sceKernelStartThread(a, 0, NULL);
	sceKernelStartThread(b, 0, NULL);

	/* Park here until each worker ends. NULL stat: the code is not needed, and a
	 * blocking join cannot write it back at wake time anyway. */
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
