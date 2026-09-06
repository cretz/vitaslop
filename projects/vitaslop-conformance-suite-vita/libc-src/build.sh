#!/bin/bash
# Reproduce the libc probe artifacts (libc.elf, libc.velf) from libc.c.
#
# UNLIKE the other suite-vita artifacts, this one links REAL newlib (no
# -nostdlib): it is the probe for full-libc titles (the newlib -> Doom arc). The
# vitasdk crt0 and libc/libm are linked, so libc runs as guest ARM and only the
# newlib syscall bottom (sceIoWrite, memory, exit) plus C init remain as NID
# imports. newlib is a permissive build-tool dependency; we run the output, not
# derive from it.
#
# Run under WSL:  VITASDK=$HOME/vitasdk bash build.sh
set -euo pipefail

: "${VITASDK:?set VITASDK to your Vita toolchain root}"
export PATH="$VITASDK/bin:$PATH"

HERE="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
cd "$HERE"

CC=arm-vita-eabi-gcc

# NOTE: no -nostdlib / -nostartfiles / -e here. We WANT the vita crt0 + newlib.
# -Wl,-q keeps relocations (loader + vita-elf-create consume them).
"$CC" \
  -march=armv7-a -mtune=cortex-a9 -mfpu=neon -mfloat-abi=hard \
  -std=c11 -O2 \
  -fno-tree-vectorize -fno-tree-slp-vectorize \
  -Wall -Wextra -Wno-unused-parameter \
  -Wl,-q \
  libc.c \
  -o libc.elf

vita-elf-create libc.elf libc.velf

echo "OK: libc.elf ($(stat -c%s libc.elf) B), libc.velf ($(stat -c%s libc.velf) B)"
