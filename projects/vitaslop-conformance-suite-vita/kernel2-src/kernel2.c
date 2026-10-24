/*
 * vitaslop conformance corpus: KERNEL ODDS AND ENDS (blob-free).
 *
 * Calls a title makes that each answer one question:
 *   - sceKernelChangeThreadPriority answers 0, its "2" spelling the PREVIOUS priority; a
 *     priority of 0 means the caller's own, one relative to the default (0x10000100 +/- 32)
 *     is resolved, and one outside 64..191 is ILLEGAL_PRIORITY;
 *   - sceKernelPollSema takes the count without waiting, SEMA_ZERO when it is short,
 *     ILLEGAL_COUNT for a signal below 1, UNKNOWN_SEMA_ID for a bad id;
 *   - sceKernelGetThreadCpuAffinityMask2 answers the user set for a default-mask thread;
 *   - sceKernelLibcGmtime_r / Localtime_r break a time down with the FULL year, and
 *     sceKernelLibcMktime normalises out-of-range fields back into the same struct.
 * Expected output: see cases/kernel2.toml.
 *
 * Authored clean-room from the MIT vita-headers API, built -nostdlib.
 */

#include <psp2/kernel/clib.h>
#include <psp2/kernel/processmgr.h>
#include <psp2/kernel/threadmgr.h>

/* Exported by the kernel libraries and named in the NID database, not declared by the headers. */
int sceKernelChangeThreadPriority2(SceUID thid, int priority);
int sceKernelGetThreadCpuAffinityMask2(SceUID thid);
int sceKernelGetThreadCurrentPriority(void);
typedef struct { int sec, min, hour, mday, mon, year, wday, yday, isdst; } KernelTm;
KernelTm *sceKernelLibcGmtime_r(const unsigned int *t, KernelTm *out);
KernelTm *sceKernelLibcLocaltime_r(const unsigned int *t, KernelTm *out);
int sceKernelLibcMktime(KernelTm *tm, unsigned int *out, unsigned long long *out64);

static void show_tm(const KernelTm *t) {
	sceClibPrintf(" %d-%d-%d %d:%d:%d w%d y%d d%d", t->year, t->mon, t->mday, t->hour, t->min,
		t->sec, t->wday, t->yday, t->isdst);
}

int main(void) {
	int base = sceKernelGetThreadCurrentPriority();
	sceClibPrintf("prio %d", base);
	sceClibPrintf(" %d", sceKernelChangeThreadPriority(0, 100));
	sceClibPrintf(" %d", sceKernelGetThreadCurrentPriority());
	sceClibPrintf(" %d", sceKernelChangeThreadPriority2(0, 0x10000100 + 5));
	sceClibPrintf(" %d", sceKernelGetThreadCurrentPriority());
	sceClibPrintf(" %d", sceKernelChangeThreadPriority2(0, 0));
	sceClibPrintf(" %x %x", sceKernelChangeThreadPriority(0, 16), sceKernelChangeThreadPriority(0, 192));
	sceClibPrintf(" %d", sceKernelChangeThreadPriority2(0, base));

	SceUID sema = sceKernelCreateSema("poll", 0, 2, 5, 0);
	sceClibPrintf("\nsema %x", sceKernelPollSema(sema, 1));
	sceClibPrintf(" %x", sceKernelPollSema(sema, 2));
	sceClibPrintf(" %x", sceKernelPollSema(sema, 1));
	sceKernelSignalSema(sema, 3);
	sceClibPrintf(" %x %x %x", sceKernelPollSema(sema, 3), sceKernelPollSema(sema, 0), sceKernelPollSema(-1, 1));

	sceClibPrintf("\naffinity %x", sceKernelGetThreadCpuAffinityMask2(0));

	/* 2012-02-29 13:45:30 UTC, a Wednesday, day 59 of a leap year */
	unsigned int t = 1330523130u;
	KernelTm tm;
	sceKernelLibcGmtime_r(&t, &tm);
	sceClibPrintf("\ngm");
	show_tm(&tm);
	sceKernelLibcLocaltime_r(&t, &tm);
	sceClibPrintf("\nlocal");
	show_tm(&tm);

	/* 2011-14-01 (month 13, 0-based) is 2012-02-01; 61 seconds rolls the minute */
	KernelTm m = { 61, 59, 23, 1, 13, 2011, 0, 0, 0 };
	unsigned int out = 0;
	unsigned long long out64 = 0;
	int r = sceKernelLibcMktime(&m, &out, &out64);
	sceClibPrintf("\nmk %d %u %u", r, out, (unsigned int)out64);
	show_tm(&m);

	sceKernelExitProcess(0);
	return 0;
}

/* ======================================================================= *
 *  Tiny freestanding runtime (-nostdlib).
 * ======================================================================= */

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
