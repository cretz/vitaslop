/*
 * vitaslop conformance corpus: minimal C "hello world" over real Vita NID
 * imports (blob-free).
 *
 * One rung up from arm/hello: instead of an ARM-level svc trap, this reaches the
 * host through the Vita's real module-import (NID) mechanism. It is authored
 * clean-room against the MIT vita-headers API - nothing here is copied from the
 * (unlicensed) vitasdk samples.
 *
 * Purpose in the bring-up: it drives the loader (SceLibKernel import resolution)
 * and the host-module ABI demand-first, and - unlike the cube - it exercises a
 * VARIADIC host call (sceClibPrintf). The format string is deliberately broad so
 * the host printf formatter and the AAPCS variadic argument walk (core registers
 * then stack, doubles promoted and 8-byte aligned) are both stressed.
 *
 * Built -nostdlib with a tiny self-contained runtime (bottom of file) so the
 * committed corpus binary stays license-clean and the import surface is only
 * Sony NID stubs (no newlib).
 */

#include <psp2/kernel/clib.h>
#include <psp2/kernel/processmgr.h>

int main(void) {
	/* The plain string: no conversions, the simplest path. */
	sceClibPrintf("Hello, world\n");

	/* Integer conversions: signed, unsigned, hex (both cases), octal. These all
	 * live in the core argument registers (r1..r3) after the format string. */
	sceClibPrintf("int=%d uint=%u hex=%x HEX=%X oct=%o\n",
		-42, 42u, 0xbeef, 0xBEEF, 64);

	/* Character, string, and a literal percent. %s dereferences a guest pointer. */
	sceClibPrintf("char=%c str=%s pct=%%\n", '!', "vitaslop");

	/* Width, zero-pad, left-justify, and forced sign flags. */
	sceClibPrintf("width=[%5d] zero=[%05d] left=[%-5d] plus=[%+d]\n",
		42, 42, 42, 42);

	/* A pointer, then six integers so the 4th..6th spill past r3 onto the stack -
	 * this is the variadic stack-argument path. */
	sceClibPrintf("ptr=%p six=%d,%d,%d,%d,%d,%d\n",
		(void *)0x81000000, 1, 2, 3, 4, 5, 6);

	/* Doubles: a variadic float is promoted to double and passed 8-byte aligned
	 * in the core registers/stack (NOT the VFP file). Exact binary values so the
	 * 6-decimal default output is unambiguous. */
	sceClibPrintf("float=%f half=%f neg=%f\n", 1.5, 0.25, -3.5);

	sceKernelExitProcess(0);
	return 0;
}

/* ======================================================================= *
 *  Tiny freestanding runtime (-nostdlib).
 * ======================================================================= */

/* memcpy/memset: the compiler may emit calls to these for aggregate copies and
 * initializations even in freestanding mode. */
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

/* Entry point. The loader jumps here; call main, then exit. sceKernelExitProcess
 * halts the run, so the spin is only a safety net if it ever returns. */
void _start(void) {
	main();
	for (;;) { }
}
