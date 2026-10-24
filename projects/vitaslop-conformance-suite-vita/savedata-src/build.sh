#!/bin/bash
# Reproduce the AppUtil save-slot artifacts (savedata.elf, savedata.velf) from savedata.c.
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
  savedata.c \
  -L"$VITASDK/arm-vita-eabi/lib" \
  -lSceLibKernel_stub -lSceAppUtil_stub -lSceIofilemgr_stub -lSceProcessmgr_stub \
  -o savedata.elf

vita-elf-create savedata.elf savedata.velf

echo "OK: savedata.elf ($(stat -c%s savedata.elf) B), savedata.velf ($(stat -c%s savedata.velf) B)"
