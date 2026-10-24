/*
 * vitaslop conformance corpus: LIVE touch on both panels, as a person's pointer delivers it.
 *
 * Not a fixed-output program on its own: it reports what it is touched with, so the browser
 * test (vitaslop-web/e2e/touch-live.mjs) presses the player's canvas and checks the text. Each
 * new press prints its panel and a coarse 4x4 cell of the panel - `F` front, `B` back, then the
 * column and row - and each lift prints `f` / `b`. A front press in the top-left cell (`F00`)
 * ends the run with `E`, so the harness decides when it is over, not a timer the guest clock
 * could race. (A program with no presented frames has nothing holding its clock to the wall.) Clean-room against the MIT vita-headers API.
 */

#include <psp2/kernel/clib.h>
#include <psp2/kernel/processmgr.h>
#include <psp2/kernel/threadmgr.h>
#include <psp2/touch.h>

/* The panels' active areas (sceTouchGetPanelInfo): X 0..1919 on both, Y 0..1087 on the front
 * and 0..889 on the smaller back panel. */
static const int PANEL_H[2] = { 1088, 890 };

/* Which quarter of `span` the coordinate `v` is in - by multiplying, since a freestanding build
 * has no division helper. */
static int cell(int v, int span) {
	int c = 0;
	while (c < 3 && v * 4 >= span * (c + 1))
		c++;
	return c;
}

int main(void) {
	sceTouchSetSamplingState(SCE_TOUCH_PORT_FRONT, SCE_TOUCH_SAMPLING_STATE_START);
	sceTouchSetSamplingState(SCE_TOUCH_PORT_BACK, SCE_TOUCH_SAMPLING_STATE_START);
	int was[2] = { 0, 0 };
	/* No time limit: an idle guest's clock is not held to the wall (nothing is presented), so
	 * any limit here would be a race with the person pressing. The harness ends the program
	 * with its last press, and its own timeout ends a run that never gets one. */
	for (;;) {
		for (int port = 0; port < 2; port++) {
			SceTouchData d;
			if (sceTouchPeek(port, &d, 1) < 0) {
				sceClibPrintf("X");
				sceKernelExitProcess(0);
			}
			int down = d.reportNum > 0;
			if (down && !was[port]) {
				int col = cell(d.report[0].x, 1920);
				int row = cell(d.report[0].y, PANEL_H[port]);
				sceClibPrintf("%c%d%d", port ? 'B' : 'F', col, row);
				if (port == 0 && col == 0 && row == 0) {
					sceClibPrintf("E");
					sceKernelExitProcess(0);
				}
			} else if (!down && was[port]) {
				sceClibPrintf("%c", port ? 'b' : 'f');
			}
			was[port] = down;
		}
		sceKernelDelayThread(16666);
	}
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
