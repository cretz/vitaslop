/*
 * vitaslop conformance corpus: THREAD EVENT HANDLERS (blob-free).
 *
 * sceKernelRegisterThreadEventHandler asks the kernel to call a function when a
 * thread ends. A C library uses it to free each thread's TLS, so the handler has to
 * run ON the ending thread: it reads that thread's own state. This proves:
 *   - an EXIT handler registered for every thread of the process (0x10027) runs for a
 *     worker that RETURNS from its entry ("a" then "x5s") and for one that calls
 *     sceKernelExitThread ("b" then "x7s"): `x`, the exit code, and `s` when the
 *     handler is running on the very thread it was told about;
 *   - it runs before the thread counts as ended (main joins each worker, so the
 *     handler's output lands before the next worker's);
 *   - a thread that ended and is STARTED AGAIN under the same id runs it again at its second
 *     end ("ax5s" twice from one thread);
 *   - after sceKernelUnregisterThreadEventHandler it no longer runs ("c" alone);
 *   - `common` is passed through (the handler counts calls in it).
 * Expected output: "ax5sbx7sax5sax5scM4".
 *
 * Authored clean-room from the MIT vita-headers API, built -nostdlib.
 */

#include <psp2/kernel/clib.h>
#include <psp2/kernel/threadmgr.h>
#include <psp2/kernel/processmgr.h>

#define EVENT_EXIT 8
#define ALL_THREADS_IN_PROCESS 0x10027

/* The worker the handler is expected to be told about next. */
static SceUID expected;

static int on_event(SceInt32 type, SceUID thid, SceInt32 arg, void *common) {
	if (type != EVENT_EXIT)
		sceClibPrintf("?");
	sceClibPrintf("x%d", arg);
	if (thid == expected && sceKernelGetThreadId() == thid)
		sceClibPrintf("s");
	(*(int *)common)++;
	return 0;
}

static int returns(SceSize args, void *argp) {
	sceClibPrintf("a");
	return 5;
}

static int exits(SceSize args, void *argp) {
	sceClibPrintf("b");
	sceKernelExitThread(7);
	return 0;
}

static int after(SceSize args, void *argp) {
	sceClibPrintf("c");
	return 9;
}

static void run(SceKernelThreadEntry entry) {
	SceUID t = sceKernelCreateThread("w", entry, 0x10000100, 0x10000, 0, 0, NULL);
	expected = t;
	sceKernelStartThread(t, 0, NULL);
	sceKernelWaitThreadEnd(t, NULL, NULL);
}

/* One thread, started, ended, then started again under the same id. */
static void restart(SceKernelThreadEntry entry) {
	SceUID t = sceKernelCreateThread("r", entry, 0x10000100, 0x10000, 0, 0, NULL);
	expected = t;
	for (int i = 0; i < 2; i++) {
		sceKernelStartThread(t, 0, NULL);
		sceKernelWaitThreadEnd(t, NULL, NULL);
	}
}

int main(void) {
	static int calls;
	SceUID h = sceKernelRegisterThreadEventHandler("h", ALL_THREADS_IN_PROCESS, EVENT_EXIT, on_event, &calls);
	if (h < 0)
		sceClibPrintf("register failed %08x", h);

	run(returns);
	run(exits);
	restart(returns);
	sceKernelUnregisterThreadEventHandler(h);
	run(after);

	sceClibPrintf("M%d", calls);
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
