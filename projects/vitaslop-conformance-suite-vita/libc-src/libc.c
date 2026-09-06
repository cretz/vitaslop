// libc conformance probe: the FIRST artifact that links real newlib (drops
// -nostdlib), so libc code (malloc, stdio, string, software-divide helpers) runs
// as guest ARM through the transpiler and the newlib syscall bottom (_write,
// _sbrk, exit) surfaces the real host-call demand for full-libc titles.
//
// Clean-room: our own MIT source. newlib is a permissive build-tool dependency
// linked in; we run the resulting binary, we do not derive from it. Deterministic
// output so a golden can assert it byte for byte.
//
// Deliberately exercises: printf/stdio (-> vfprintf -> _write), malloc/free (->
// _sbrk), integer division + modulo (-> __aeabi_uidiv/uidivmod on Cortex-A9 which
// has no hardware UDIV), qsort (indirect calls + comparisons), strtol.

#include <stdio.h>
#include <stdlib.h>
#include <string.h>

static int cmp_int(const void *a, const void *b) {
    int x = *(const int *)a, y = *(const int *)b;
    return (x > y) - (x < y);
}

int main(void) {
    // stdio: forces vfprintf + _write syscall bottom.
    printf("libc probe start\n");

    // malloc/free: forces _sbrk. Touch the memory so a bad heap traps.
    int *buf = malloc(16 * sizeof(int));
    for (int i = 0; i < 16; i++) {
        buf[i] = (i * 7 + 3) % 13;   // % forces __aeabi_uidivmod
    }

    // qsort: indirect calls through cmp_int, plenty of comparisons.
    qsort(buf, 16, sizeof(int), cmp_int);

    // Print sorted, and accumulate a checksum with division in it.
    long sum = 0;
    for (int i = 0; i < 16; i++) {
        sum += buf[i];
        printf("v[%d]=%d\n", i, buf[i]);
    }
    printf("sum=%ld avg=%ld\n", sum, sum / 16);

    // strtol: another common libc path.
    const char *s = "  -1234abc";
    char *end = NULL;
    long n = strtol(s, &end, 10);
    printf("strtol=%ld tail=%s\n", n, end);

    // string ops through newlib (not our sceClib*): memmove overlap + strcmp.
    char msg[32];
    strcpy(msg, "chocolate");
    memmove(msg + 4, msg, 5);   // overlap
    printf("msg=%s cmp=%d\n", msg, strcmp(msg, "chocolate"));

    free(buf);
    printf("libc probe done\n");
    return 0;
}
