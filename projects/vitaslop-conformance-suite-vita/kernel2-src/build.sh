#!/bin/bash
# Reproduce the kernel odds-and-ends artifacts (kernel2.elf, kernel2.velf) from kernel2.c.
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
  kernel2.c \
  -L"$VITASDK/arm-vita-eabi/lib" \
  -lSceLibKernel_stub -lSceKernelThreadMgr_stub -lSceProcessmgr_stub \
  -o kernel2.elf

vita-elf-create kernel2.elf kernel2.velf

echo "OK: kernel2.elf ($(stat -c%s kernel2.elf) B), kernel2.velf ($(stat -c%s kernel2.velf) B)"
