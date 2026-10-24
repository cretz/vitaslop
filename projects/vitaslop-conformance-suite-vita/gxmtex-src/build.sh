#!/bin/bash
# Reproduce the GXM texture control-word artifacts (gxmtex.elf, gxmtex.velf) from gxmtex.c.
#
# Requires a Vita toolchain on PATH via $VITASDK. The toolchain is a build tool
# only - compiling our MIT, -nostdlib source with it does not attach any GPL/LGPL
# obligation to the output. Run under WSL:
#   VITASDK=$HOME/vitasdk bash build.sh
set -euo pipefail

: "${VITASDK:?set VITASDK to your Vita toolchain root}"
export PATH="$VITASDK/bin:$PATH"

HERE="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
cd "$HERE"

CC=arm-vita-eabi-gcc

"$CC" \
  -march=armv7-a -mtune=cortex-a9 -mfpu=neon -mfloat-abi=hard \
  -std=c11 -O2 -ffreestanding -fno-builtin \
  -Wall -Wextra -Wno-unused-parameter \
  -nostdlib -nostartfiles -e _start -Wl,-q \
  -I"$VITASDK/arm-vita-eabi/include" \
  gxmtex.c \
  -L"$VITASDK/arm-vita-eabi/lib" \
  -lSceLibKernel_stub -lSceGxm_stub -lSceProcessmgr_stub \
  -o gxmtex.elf

vita-elf-create gxmtex.elf gxmtex.velf

echo "OK: gxmtex.elf ($(stat -c%s gxmtex.elf) B), gxmtex.velf ($(stat -c%s gxmtex.velf) B)"
