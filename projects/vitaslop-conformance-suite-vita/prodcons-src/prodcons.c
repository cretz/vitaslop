/*
 * vitaslop conformance corpus: a bounded-buffer PRODUCER/CONSUMER over real Vita
 * NID imports (blob-free) - the canonical "one thread produces, another waits on
 * it" handshake, driven by a mutex and TWO condition variables.
 *
 * A single-slot buffer forces strict alternation: the producer must wait on
 * `notFull` while the slot is occupied, and the consumer must wait on `notEmpty`
 * while it is empty. Each side re-checks its predicate in a `while` loop after every
 * wake (so a lost or spurious wake cannot corrupt the sequence). The producer feeds
 * 1, 2, 3; the consumer prints each in turn; main (joining) prints 'M'.
 *   -> "123M".
 * Because the buffer holds one item, value v+1 cannot be produced until v has been
 * consumed, so the digits are strictly ordered. A lost wakeup (a signal that fails
 * to release a parked peer) would stall the handshake and deadlock the join - which
 * is exactly the failure this exercises across the two threads.
 *
 * Authored clean-room from the MIT vita-headers API, built -nostdlib.
 */

#include <psp2/kernel/clib.h>
#include <psp2/kernel/threadmgr.h>
#include <psp2/kernel/processmgr.h>

static SceUID mutex;
static SceUID notEmpty;   /* signalled when the slot becomes occupied */
static SceUID notFull;    /* signalled when the slot becomes free     */
static volatile int slot; /* 0 = empty, else the item value           */

#define ITEMS 3

static int consumer(SceSize args, void *argp) {
	for (int i = 0; i < ITEMS; i++) {
		sceKernelLockMutex(mutex, 1, NULL);
		while (slot == 0)
			sceKernelWaitCond(notEmpty, NULL);
		int v = slot;
		slot = 0;
		sceKernelSignalCond(notFull);
		sceKernelUnlockMutex(mutex, 1);
		char b[2] = { (char)('0' + v), 0 };
		sceClibPrintf(b);
	}
	return 0;
}

static int producer(SceSize args, void *argp) {
	for (int v = 1; v <= ITEMS; v++) {
		sceKernelLockMutex(mutex, 1, NULL);
		while (slot != 0)
			sceKernelWaitCond(notFull, NULL);
		slot = v;
		sceKernelSignalCond(notEmpty);
		sceKernelUnlockMutex(mutex, 1);
	}
	return 0;
}

int main(void) {
	mutex = sceKernelCreateMutex("m", 0, 0, NULL);
	notEmpty = sceKernelCreateCond("ne", 0, mutex, NULL);
	notFull = sceKernelCreateCond("nf", 0, mutex, NULL);

	SceUID c = sceKernelCreateThread("c", consumer, 0x10000100, 0x10000, 0, 0, NULL);
	SceUID p = sceKernelCreateThread("p", producer, 0x10000100, 0x10000, 0, 0, NULL);

	/* Start the consumer first so it parks on notEmpty before the producer runs. */
	sceKernelStartThread(c, 0, NULL);
	sceKernelStartThread(p, 0, NULL);

	sceKernelWaitThreadEnd(p, NULL, NULL);
	sceKernelWaitThreadEnd(c, NULL, NULL);
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
