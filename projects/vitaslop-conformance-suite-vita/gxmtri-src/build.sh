#!/bin/bash
# Reproduce the gxmtri corpus artifacts (gxmtri.elf, gxmtri.velf) from gxmtri.c.
#
# Requires a Vita toolchain on PATH via $VITASDK (arm-vita-eabi-gcc,
# vita-elf-create). The toolchain is a build tool only - compiling our MIT,
# -nostdlib source with it does not attach any GPL/LGPL obligation to the output.
#
# Deterministic: run this, then `git diff` should be clean. Run under WSL:
#   VITASDK=$HOME/vitasdk bash build.sh
set -euo pipefail

: "${VITASDK:?set VITASDK to your Vita toolchain root}"
export PATH="$VITASDK/bin:$PATH"

HERE="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
cd "$HERE"

CC=arm-vita-eabi-gcc

# -Wl,-q keeps relocation sections (vita-elf-create + our loader consume them).
# -nostdlib + freestanding: self-contained runtime, so the only imports are Sony
# NID stubs (no newlib) - a small, clean loader surface and a license-clean binary.
"$CC" \
  -march=armv7-a -mtune=cortex-a9 -mfpu=neon -mfloat-abi=hard \
  -std=c11 -O2 -ffreestanding -fno-builtin \
  -Wall -Wextra -Wno-unused-parameter \
  -nostdlib -nostartfiles -e _start -Wl,-q \
  -I"$VITASDK/arm-vita-eabi/include" \
  gxmtri.c \
  -L"$VITASDK/arm-vita-eabi/lib" \
  -lSceGxm_stub -lSceSysmem_stub -lSceLibKernel_stub -lSceProcessmgr_stub \
  -o gxmtri.elf

vita-elf-create gxmtri.elf gxmtri.velf

echo "OK: gxmtri.elf ($(stat -c%s gxmtri.elf) B), gxmtri.velf ($(stat -c%s gxmtri.velf) B)"
