#!/bin/bash
# Reproduce the service odds-and-ends artifacts (svc2.elf, svc2.velf) from svc2.c.
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
  svc2.c \
  -L"$VITASDK/arm-vita-eabi/lib" \
  -lSceLibKernel_stub -lSceCtrl_stub -lSceSysmem_stub -lSceAppUtil_stub -lSceProcessmgr_stub \
  -o svc2.elf

vita-elf-create svc2.elf svc2.velf

echo "OK: svc2.elf ($(stat -c%s svc2.elf) B), svc2.velf ($(stat -c%s svc2.velf) B)"
