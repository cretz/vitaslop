#!/bin/bash
# Reproduce the ScePvf font-engine coverage artifacts (pvf.elf, pvf.velf) from pvf.c.
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
  -std=c11 -O2 -ffreestanding -fno-builtin -fno-tree-vectorize \
  -Wall -Wextra -Wno-unused-parameter \
  -nostdlib -nostartfiles -e _start -Wl,-q \
  -I"$VITASDK/arm-vita-eabi/include" \
  pvf.c \
  -L"$VITASDK/arm-vita-eabi/lib" \
  -lScePvf_stub -lSceLibKernel_stub \
  -o pvf.elf

vita-elf-create pvf.elf pvf.velf

echo "OK: pvf.elf ($(stat -c%s pvf.elf) B), pvf.velf ($(stat -c%s pvf.velf) B)"
