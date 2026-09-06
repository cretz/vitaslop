/*
 * vitaslop conformance corpus: minimal threading over real Vita NID imports
 * (blob-free).
 *
 * Exercises the create/start/wait pattern - the shape almost every threaded
 * homebrew uses - and, importantly, the two hard mechanisms behind it:
 *   - guest re-entry: the host must run the worker's own guest code (its
 *     sceClibPrintf must appear), so this drives the host's synchronous thread
 *     execution, and
 *   - an address-taken function: `worker` is passed to sceKernelCreateThread as a
 *     pointer, never called directly, so the transpiler must discover it as a
 *     code-pointer entry (not just via the direct-call closure from _start).
 *
 * Authored clean-room from the MIT vita-headers API, built -nostdlib.
 */

#include <psp2/kernel/clib.h>
#include <psp2/kernel/threadmgr.h>
#include <psp2/kernel/processmgr.h>

/* The worker thread: reads its argument, prints, and returns a derived value the
 * main thread collects through sceKernelWaitThreadEnd. */
static int worker(SceSize args, void *argp) {
	int v = *(const int *)argp;
	sceClibPrintf("worker: got %d\n", v);
	return v * 3;
}

int main(void) {
	sceClibPrintf("main: creating thread\n");

	SceUID th = sceKernelCreateThread("worker", worker, 0x10000100, 0x10000, 0, 0, NULL);
	sceClibPrintf("main: thid ok=%d\n", th >= 0);

	int arg = 14;
	sceKernelStartThread(th, sizeof(arg), &arg);

	int ret = 0;
	sceKernelWaitThreadEnd(th, &ret, NULL);
	sceClibPrintf("main: worker returned %d\n", ret);

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
