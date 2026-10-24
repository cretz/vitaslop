/*
 * vitaslop conformance corpus: APPUTIL SAVE SLOTS WRITTEN BY A DATA SAVE (blob-free).
 *
 * sceAppUtilSaveDataDataSave takes a slot `{id, slotParam*}` beside its file list. A save
 * that carries a param creates (or updates) that slot in the same call; one with a NULL
 * param writes only the files. A title that saves this way checks the slot on its next
 * boot before reading its files, and a file with no slot reads as a broken save. This
 * proves:
 *   - a slot that was never written is SLOT_NOT_FOUND ("n");
 *   - a data save with no param writes the file but no slot ("s0" then "n");
 *   - a data save with a param creates the slot, and its user param reads back ("p42");
 *   - a second save with a param updates it ("p43");
 *   - the file reads back through the savedata mount ("r3").
 * Expected output: "i0 n s0 n s0 p42 s0 p43 r3".
 *
 * Authored clean-room from the MIT vita-headers API, built -nostdlib.
 */

#include <psp2/apputil.h>
#include <psp2/io/fcntl.h>
#include <psp2/kernel/clib.h>
#include <psp2/kernel/processmgr.h>

void *memset(void *dst, int v, unsigned int n);

#define SLOT 7
#define SLOT_NOT_FOUND 0x80100641

static SceAppUtilSaveDataSlotParam param;

static void get(void) {
	SceAppUtilSaveDataSlotParam out;
	int r = sceAppUtilSaveDataSlotGetParam(SLOT, &out, NULL);
	if (r == (int)SLOT_NOT_FOUND)
		sceClibPrintf(" n");
	else if (r == 0)
		sceClibPrintf(" p%d", out.userParam);
	else
		sceClibPrintf(" g%08x", r);
}

static void save(SceAppUtilSaveDataSlotParam *p) {
	static char data[3] = { 'a', 'b', 'c' };
	SceAppUtilSaveDataFile file;
	SceAppUtilSaveDataFileSlot slot;
	SceSize need = 0;
	memset(&file, 0, sizeof(file));
	memset(&slot, 0, sizeof(slot));
	file.filePath = "a.txt";
	file.buf = data;
	file.bufSize = sizeof(data);
	file.mode = SCE_APPUTIL_SAVEDATA_DATA_SAVE_MODE_FILE;
	slot.id = SLOT;
	slot.slotParam = p;
	sceClibPrintf(" s%d", sceAppUtilSaveDataDataSave(&slot, &file, 1, NULL, &need));
}

int main(void) {
	SceAppUtilInitParam init;
	SceAppUtilBootParam boot;
	memset(&init, 0, sizeof(init));
	memset(&boot, 0, sizeof(boot));
	sceClibPrintf("i%d", sceAppUtilInit(&init, &boot));

	get();
	save(NULL);
	get();
	param.userParam = 42;
	save(&param);
	get();
	param.userParam = 43;
	save(&param);
	get();

	char buf[8];
	SceUID fd = sceIoOpen("savedata0:/a.txt", SCE_O_RDONLY, 0);
	int n = fd < 0 ? fd : sceIoRead(fd, buf, sizeof(buf));
	if (fd >= 0)
		sceIoClose(fd);
	sceClibPrintf(" r%d", n);

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
