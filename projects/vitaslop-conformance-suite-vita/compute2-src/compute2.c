/*
 * vitaslop conformance corpus: integer compute coverage, part 2 (blob-free).
 *
 * A second CPU-core probe, targeting the bit- and byte-manipulation instructions
 * that ordinary C emits constantly but the earlier probes did not: byte reverse
 * (REV/REV16), sign/zero extension (SXTB/UXTB/SXTH), multiply-accumulate (MLA),
 * and bitfield extract (UBFX). Volatile inputs keep the optimizer from folding
 * any of it away. Authored clean-room, built -nostdlib.
 */

#include <psp2/kernel/clib.h>
#include <psp2/kernel/processmgr.h>

int main(void) {
	volatile unsigned int w = 0x11223344u;
	volatile unsigned int h = 0x0000ABCDu;
	volatile unsigned int val = 0xFFFFFF9Bu; /* low byte 0x9B, low half 0xFF9B */
	volatile unsigned int a = 7, b = 6, c = 100;
	volatile unsigned int bf = 0x00000AB0u;

	/* 32-bit byte reverse (REV): 0x11223344 -> 0x44332211. */
	sceClibPrintf("rev: 0x%x\n", __builtin_bswap32(w));

	/* 16-bit byte reverse (REV16): 0xABCD -> 0xCDAB. */
	sceClibPrintf("rev16: 0x%x\n", (unsigned)(unsigned short)__builtin_bswap16((unsigned short)h));

	/* Sign/zero extend byte and halfword (SXTB/UXTB/SXTH). 0x9B as signed byte is
	 * -101; as unsigned byte 155; 0xFF9B as signed halfword is -101. */
	sceClibPrintf("sxtb: %d\n", (int)(signed char)val);
	sceClibPrintf("uxtb: %u\n", (unsigned)(unsigned char)val);
	sceClibPrintf("sxth: %d\n", (int)(short)val);

	/* Multiply-accumulate (MLA): 7*6 + 100 = 142. */
	sceClibPrintf("mla: %u\n", a * b + c);

	/* Bitfield extract (UBFX): (0xAB0 >> 4) & 0x7f = 0xAB & 0x7f = 0x2B = 43. */
	sceClibPrintf("ubfx: %u\n", (bf >> 4) & 0x7fu);

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
