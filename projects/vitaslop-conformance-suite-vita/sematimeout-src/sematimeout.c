/*
 * vitaslop conformance corpus: TIMED semaphore waits and their return codes over
 * real Vita NID imports (blob-free).
 *
 * sceKernelWaitSema takes an optional *timeout (microseconds). This proves the two
 * outcomes of a timed wait AND the untimed wait, all distinguished by the return
 * value:
 *   - a timed wait that no one satisfies must return SCE_KERNEL_ERROR_WAIT_TIMEOUT
 *     (0x80028005) when the deadline passes - NOT success, and NOT an infinite park;
 *   - a timed wait that IS satisfied returns 0;
 *   - an untimed (NULL timeout) wait that is satisfied returns 0.
 *
 * The timeout is measured on the emulator's virtual clock, which the scheduler jumps
 * forward instantly once every thread is parked - so the "1000 us" wait costs zero
 * wall-clock time (no real sleep).
 *
 * The worker first waits on a semaphore no one posts (times out -> 'T'), then on one
 * a poker already posted (satisfied, timed -> 'S'), then on one a poker already
 * posted with an untimed wait (satisfied -> 'i'). main (joining) prints 'M'.
 *   -> "TSiM".
 * Before timed-wait support, the first wait parks forever with no deadline and the
 * run deadlocks (never reaching 'T'); a wait that returns 0 on timeout would print
 * a different first letter. So "TSiM" pins the exact contract.
 *
 * Authored clean-room from the MIT vita-headers API, built -nostdlib.
 */

#include <psp2/kernel/clib.h>
#include <psp2/kernel/threadmgr.h>
#include <psp2/kernel/processmgr.h>
#include <psp2/kernel/error.h>

static SceUID semTo;   /* never posted -> the timed wait must time out */
static SceUID semB;    /* posted before the wait -> timed wait succeeds  */
static SceUID semC;    /* posted before the wait -> untimed wait succeeds */

static int worker(SceSize args, void *argp) {
	SceUInt t = 1000;
	int r = sceKernelWaitSema(semTo, 1, &t);
	sceClibPrintf((unsigned int)r == (unsigned int)SCE_KERNEL_ERROR_WAIT_TIMEOUT ? "T"
	              : (r == 0 ? "0" : "E"));

	SceUInt tbig = 1000000;
	int r2 = sceKernelWaitSema(semB, 1, &tbig);
	sceClibPrintf(r2 == 0 ? "S" : "e");

	int r3 = sceKernelWaitSema(semC, 1, NULL);
	sceClibPrintf(r3 == 0 ? "i" : "f");
	return 0;
}

static int pokerB(SceSize args, void *argp) { sceKernelSignalSema(semB, 1); return 0; }
static int pokerC(SceSize args, void *argp) { sceKernelSignalSema(semC, 1); return 0; }

int main(void) {
	semTo = sceKernelCreateSema("to", 0, 0, 8, NULL);
	semB  = sceKernelCreateSema("b",  0, 0, 8, NULL);
	semC  = sceKernelCreateSema("c",  0, 0, 8, NULL);

	SceUID w  = sceKernelCreateThread("w",  worker, 0x10000100, 0x10000, 0, 0, NULL);
	SceUID pb = sceKernelCreateThread("pb", pokerB, 0x10000100, 0x10000, 0, 0, NULL);
	SceUID pc = sceKernelCreateThread("pc", pokerC, 0x10000100, 0x10000, 0, 0, NULL);

	/* The pokers run and post semB/semC before the worker (parked on semTo) reaches
	 * those waits, so both are already satisfied when it gets there. */
	sceKernelStartThread(w, 0, NULL);
	sceKernelStartThread(pb, 0, NULL);
	sceKernelStartThread(pc, 0, NULL);

	sceKernelWaitThreadEnd(w, NULL, NULL);
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
