#!/bin/bash
# Reproduce the compute artifacts (compute.elf, compute.velf) from compute.c.
# Requires a Vita toolchain via $VITASDK. Run under WSL: VITASDK=$HOME/vitasdk bash build.sh
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
  compute.c \
  -L"$VITASDK/arm-vita-eabi/lib" \
  -lSceLibKernel_stub \
  -o compute.elf
vita-elf-create compute.elf compute.velf
echo "OK: compute.elf ($(stat -c%s compute.elf) B), compute.velf ($(stat -c%s compute.velf) B)"
