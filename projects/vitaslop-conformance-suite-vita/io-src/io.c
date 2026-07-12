/*
 * vitaslop conformance corpus: SceIoFilemgr file IO (blob-free).
 *
 * Drives the file-IO host module end to end over the host virtual filesystem:
 *   - sceIoWrite to fd 1 (the sink newlib's stdout resolves to),
 *   - sceIoOpen/Read/Getstat on a preloaded read-only asset,
 *   - sceIoOpen(create)/Write, then reopen + sceIoLseek + read-back on a file
 *     the guest itself produced.
 * Every observable result is printed through the trusted sceClibPrintf, so the
 * golden is a deterministic transcript. Authored clean-room from the MIT
 * vita-headers API, built -nostdlib (self-contained runtime at the bottom).
 *
 * The harness preloads "app0:/asset.bin" = bytes 1..=10 before the run.
 */

#include <psp2/io/fcntl.h>
#include <psp2/io/stat.h>
#include <psp2/kernel/clib.h>
#include <psp2/kernel/processmgr.h>

int main(void) {
	/* 1. Raw write to stdout (fd 1): the path newlib's printf ultimately takes. */
	const char *msg = "hello io\n";
	int len = 0;
	while (msg[len]) len++;
	sceIoWrite(1, msg, len);

	/* 2. Read a preloaded asset and sum its bytes. */
	SceUID fd = sceIoOpen("app0:/asset.bin", SCE_O_RDONLY, 0);
	sceClibPrintf("open asset: ok=%d\n", fd >= 0);
	unsigned char buf[16];
	int n = sceIoRead(fd, buf, sizeof(buf));
	int sum = 0;
	for (int i = 0; i < n; i++) sum += buf[i];
	sceClibPrintf("read: n=%d sum=%d\n", n, sum);
	sceIoClose(fd);

	/* 3. Stat the asset for its size. */
	SceIoStat st;
	int gs = sceIoGetstat("app0:/asset.bin", &st);
	sceClibPrintf("getstat: ret=%d size=%d\n", gs, (int)st.st_size);

	/* 4. Create + write a file. */
	SceUID w = sceIoOpen("ux0:/data/out.bin", SCE_O_WRONLY | SCE_O_CREAT | SCE_O_TRUNC, 0777);
	sceClibPrintf("open write: ok=%d\n", w >= 0);
	sceIoWrite(w, "vitaslop", 8);
	sceIoClose(w);

	/* 5. Reopen, seek to offset 4, read the rest back. */
	SceUID r = sceIoOpen("ux0:/data/out.bin", SCE_O_RDONLY, 0);
	int pos = (int)sceIoLseek(r, 4, SCE_SEEK_SET);
	char tail[8];
	int tn = sceIoRead(r, tail, sizeof(tail));
	tail[tn] = 0;
	sceClibPrintf("seek: pos=%d tail=[%s]\n", pos, tail);
	sceIoClose(r);

	/* 6. Reading a missing file fails. */
	SceUID bad = sceIoOpen("app0:/missing", SCE_O_RDONLY, 0);
	sceClibPrintf("missing: failed=%d\n", bad < 0);

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
