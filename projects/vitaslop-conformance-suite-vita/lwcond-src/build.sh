#!/bin/bash
# Reproduce the lwcond coverage artifacts (lwcond.elf, lwcond.velf) from lwcond.c.
# Requires a Vita toolchain on PATH via $VITASDK (build tool only; MIT -nostdlib
# source attaches no GPL/LGPL obligation). Run under WSL: VITASDK=$HOME/vitasdk bash build.sh
set -euo pipefail
: "${VITASDK:?set VITASDK to your Vita toolchain root}"
export PATH="$VITASDK/bin:$PATH"
HERE="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"; cd "$HERE"
CC=arm-vita-eabi-gcc
"$CC" -march=armv7-a -mtune=cortex-a9 -mfpu=neon -mfloat-abi=hard   -std=c11 -O2 -ffreestanding -fno-builtin -Wall -Wextra -Wno-unused-parameter   -nostdlib -nostartfiles -e _start -Wl,-q -I"$VITASDK/arm-vita-eabi/include"   lwcond.c -L"$VITASDK/arm-vita-eabi/lib" -lSceLibKernel_stub -lSceKernelThreadMgr_stub -o lwcond.elf
vita-elf-create lwcond.elf lwcond.velf
echo "OK: lwcond.elf ($(stat -c%s lwcond.elf) B), lwcond.velf ($(stat -c%s lwcond.velf) B)"
