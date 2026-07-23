#!/bin/bash
# Reproduce the condlost coverage artifacts (condlost.elf, condlost.velf) from condlost.c.
#
# Requires a Vita toolchain on PATH via $VITASDK. The toolchain is a build tool
# only - compiling our MIT, -nostdlib source with it attaches no GPL/LGPL
# obligation to the output. Run under WSL:
#   VITASDK=$HOME/vitasdk bash build.sh
set -euo pipefail

: "${VITASDK:?set VITASDK to your Vita toolchain root}"
export PATH="$VITASDK/bin:$PATH"

HERE="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
cd "$HERE"

CC=arm-vita-eabi-gcc

"$CC"   -march=armv7-a -mtune=cortex-a9 -mfpu=neon -mfloat-abi=hard   -std=c11 -O2 -ffreestanding -fno-builtin   -Wall -Wextra -Wno-unused-parameter   -nostdlib -nostartfiles -e _start -Wl,-q   -I"$VITASDK/arm-vita-eabi/include"   condlost.c   -L"$VITASDK/arm-vita-eabi/lib"   -lSceLibKernel_stub -lSceKernelThreadMgr_stub   -o condlost.elf

vita-elf-create condlost.elf condlost.velf

echo "OK: condlost.elf ($(stat -c%s condlost.elf) B), condlost.velf ($(stat -c%s condlost.velf) B)"
