/*
 * vitaslop conformance corpus: SERVICE ODDS AND ENDS (blob-free).
 *
 * Calls a title makes that each answer one question:
 *   - sceCtrlSetSamplingMode answers the PREVIOUS mode (a process starts DIGITAL), refuses a
 *     mode above ANALOG_WIDE, and sceCtrlGetSamplingMode reads the mode back;
 *   - the console is the handheld: sceKernelGetModel is SCE_KERNEL_MODEL_VITA and
 *     sceKernelIsPSVitaTV is 0;
 *   - SceAppUtil safe memory is 64 KiB: a save answers its byte count, a load of a range never
 *     written reads zeroes, a save leaves the bytes around it alone, and a range past the end
 *     (or a NULL buffer) is refused.
 * Expected output: see cases/svc2.toml.
 *
 * Authored clean-room from the MIT vita-headers API, built -nostdlib.
 */

#include <psp2/apputil.h>
#include <psp2/ctrl.h>
#include <psp2/kernel/clib.h>
#include <psp2/kernel/processmgr.h>
#include <psp2/kernel/sysmem.h>

static unsigned char buf[16];

int main(void) {
	SceCtrlPadInputMode mode = 7;
	sceClibPrintf("ctrl %x", sceCtrlGetSamplingMode(&mode));
	sceClibPrintf(" %d", mode);
	sceClibPrintf(" %d", sceCtrlSetSamplingMode(SCE_CTRL_MODE_ANALOG_WIDE));
	sceClibPrintf(" %d", sceCtrlSetSamplingMode(SCE_CTRL_MODE_ANALOG));
	sceClibPrintf(" %x", sceCtrlSetSamplingMode(3));
	sceClibPrintf(" %d", sceCtrlGetSamplingMode(&mode));

	sceClibPrintf("\nmodel %x %d", sceKernelGetModel(), sceKernelIsPSVitaTV());

	SceAppUtilInitParam init;
	SceAppUtilBootParam boot;
	for (unsigned int i = 0; i < sizeof(init); i++) ((unsigned char *)&init)[i] = 0;
	for (unsigned int i = 0; i < sizeof(boot); i++) ((unsigned char *)&boot)[i] = 0;
	sceAppUtilInit(&init, &boot);

	for (unsigned int i = 0; i < 16; i++) buf[i] = (unsigned char)(0xA0 + i);
	sceClibPrintf("\nsafe %d", sceAppUtilSaveSafeMemory(buf, 16, 100));
	for (unsigned int i = 0; i < 16; i++) buf[i] = 0x11;
	sceClibPrintf(" %d", sceAppUtilSaveSafeMemory(buf, 4, 104));
	for (unsigned int i = 0; i < 16; i++) buf[i] = 0xFF;
	sceClibPrintf(" %d", sceAppUtilLoadSafeMemory(buf, 16, 98));
	for (unsigned int i = 0; i < 16; i++) sceClibPrintf("%s%02x", i ? "" : " ", buf[i]);
	sceClibPrintf(" %x %x %d", sceAppUtilSaveSafeMemory(buf, 16, 65530), sceAppUtilSaveSafeMemory(0, 4, 0),
		sceAppUtilLoadSafeMemory(buf, 16, 65520));

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
