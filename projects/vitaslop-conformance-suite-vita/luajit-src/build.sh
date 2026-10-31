#!/bin/bash
# Reproduce the LuaJIT probe artifacts (luajit.elf, luajit.velf) from luajit.c.
#
# Like libc-src, this links REAL libraries rather than -nostdlib: the vitasdk crt0 and newlib,
# and the vitasdk LuaJIT package (libluajit-5.1.a, LuaJIT 2.1, MIT - see COPYRIGHT here). The
# guest runs LuaJIT's own ARM interpreter; we run the output, not derive from it.
#
# Run under WSL:  VITASDK=$HOME/vitasdk bash build.sh
set -euo pipefail

: "${VITASDK:?set VITASDK to your Vita toolchain root}"
export PATH="$VITASDK/bin:$PATH"

HERE="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
cd "$HERE"

CC=arm-vita-eabi-gcc

# -Wl,-q keeps relocations (loader + vita-elf-create consume them).
"$CC" \
  -march=armv7-a -mtune=cortex-a9 -mfpu=neon -mfloat-abi=hard \
  -std=c11 -O2 \
  -Wall -Wextra \
  -I"$VITASDK/arm-vita-eabi/include/luajit-2.1" \
  -Wl,-q \
  luajit.c \
  -lluajit-5.1 -lm \
  -lSceSysmem_stub -lSceKernelThreadMgr_stub \
  -o luajit.elf

vita-elf-create luajit.elf luajit.velf

echo "OK: luajit.elf ($(stat -c%s luajit.elf) B), luajit.velf ($(stat -c%s luajit.velf) B)"
