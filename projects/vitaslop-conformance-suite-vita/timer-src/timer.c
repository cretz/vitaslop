/*
 * vitaslop conformance corpus: the SceLibKernel/SceThreadmgr virtual TIMER family
 * (blob-free).
 *
 * A Vita timer is a stopwatch, not an alarm: create it, start it, read it back,
 * stop it. A retail title brought this family in by creating one called "System
 * Debug Timer" and reading it for its own profiling, and an emulator that returns
 * a plausible-looking number here is indistinguishable from one that is right -
 * hence a case that asserts the SEMANTICS rather than the values.
 *
 * Every check prints a boolean or an exact return code, never a microsecond count,
 * so the golden is deterministic on any host:
 *   - a created timer reads ZERO until it is started (it is not running yet);
 *   - starting returns 0, and starting a second time is an ERROR, not a silent
 *     restart that would throw away what it had already counted;
 *   - a running timer never goes BACKWARD between two reads;
 *   - sceKernelOpenTimer resolves the SAME timer by the name it was created with;
 *   - a stopped timer is STABLE (two reads agree, so it is not still counting) and
 *     KEEPS what it counted rather than resetting;
 *   - stopping an already stopped timer is an error, like the double start;
 *   - a uid that is not a timer is an error, not a zero, and a deleted timer's uid
 *     stops working.
 *
 * NOT asserted here: that a running timer ADVANCES. The emulated clock is charged
 * by the scheduler at quantum boundaries and this corpus runs a guest with no
 * scheduler, so the clock is legitimately still and a strict `>` would be asserting
 * the harness rather than the timer. Advancement is covered where a clock actually
 * moves, in `VitaState`'s unit tests (`timer_counts_only_while_running`).
 *
 * vitasdk publishes these NIDs but no header for them, so the prototypes below are
 * declared here from the henkaku wiki's SceLibKernel/SceKernelThreadMgr pages and
 * linked against the SDK's own stubs. Authored clean-room, built -nostdlib.
 */

#include <psp2/kernel/clib.h>
#include <psp2/kernel/processmgr.h>
#include <psp2/kernel/threadmgr.h>

/* Not in vita-headers; NIDs are in db/360/{SceLibKernel,SceKernelThreadMgr}.yml. */
typedef struct SceKernelTimerOptParam {
	SceSize size;
} SceKernelTimerOptParam;

extern SceUID sceKernelCreateTimer(const char *name, SceUInt32 attr, const SceKernelTimerOptParam *opt);
extern SceUID sceKernelOpenTimer(const char *name);
extern int sceKernelStartTimer(SceUID timerId);
extern int sceKernelStopTimer(SceUID timerId);
extern int sceKernelGetTimerTime(SceUID timerId, SceUInt64 *time);
extern int sceKernelDeleteTimer(SceUID timerId);

/* Guest work with a data dependency, so the compiler cannot fold it away. Its result
 * is printed for the same reason: an unused result is a loop that need not run. */
static unsigned int spin(unsigned int n) {
	unsigned int acc = 1;
	for (unsigned int i = 1; i <= n; i++)
		acc = acc * 31 + i;
	return acc;
}

int main(void) {
	SceUInt64 t_new = 1, t_a = 0, t_b = 0, t_stop1 = 0, t_stop2 = 0, t_bad = 12345;

	SceUID tid = sceKernelCreateTimer("conformance timer", 0, NULL);
	int r_new = sceKernelGetTimerTime(tid, &t_new);
	sceClibPrintf("create: id_ok=%d get=%d zero_before_start=%d\n", tid >= 0, r_new, t_new == 0);

	int r_start = sceKernelStartTimer(tid);
	int r_restart = sceKernelStartTimer(tid);
	sceClibPrintf("start: first=%d second_is_error=%d\n", r_start, r_restart != 0);

	sceKernelGetTimerTime(tid, &t_a);
	unsigned int burn = spin(200000);
	sceKernelGetTimerTime(tid, &t_b);
	sceClibPrintf("running: never_backward=%d work=%d\n", t_b >= t_a, burn != 0);

	SceUID reopened = sceKernelOpenTimer("conformance timer");
	sceClibPrintf("open: same_id=%d\n", reopened == tid);

	int r_stop = sceKernelStopTimer(tid);
	sceKernelGetTimerTime(tid, &t_stop1);
	burn += spin(200000);
	sceKernelGetTimerTime(tid, &t_stop2);
	sceClibPrintf("stop: ret=%d stable=%d kept_count=%d\n", r_stop, t_stop1 == t_stop2, t_stop1 >= t_b);

	int r_restop = sceKernelStopTimer(tid);
	sceClibPrintf("restop: is_error=%d\n", r_restop != 0);

	int r_bad = sceKernelGetTimerTime(-1, &t_bad);
	sceClibPrintf("unknown: is_error=%d\n", r_bad != 0);

	int r_del = sceKernelDeleteTimer(tid);
	int r_after = sceKernelGetTimerTime(tid, &t_bad);
	sceClibPrintf("delete: ret=%d gone=%d work=%d\n", r_del, r_after != 0, burn != 0);

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
