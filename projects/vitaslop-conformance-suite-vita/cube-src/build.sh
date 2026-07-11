#!/bin/bash
# Reproduce the cube corpus artifacts (cube.elf, cube.velf) from cube.c.
#
# Requires a Vita toolchain on PATH via $VITASDK (arm-vita-eabi-gcc,
# vita-elf-create). The toolchain is a build tool only - compiling our MIT,
# -nostdlib source with it does not attach any GPL/LGPL obligation to the
# output. See ../README.md and working-area/agent-notes.md for the plan.
#
# Deterministic: run this, then `git diff` should be clean (like the arm suite's
# regen). Run under WSL:
#   VITASDK=$HOME/vitasdk bash build.sh
set -euo pipefail

: "${VITASDK:?set VITASDK to your Vita toolchain root}"
export PATH="$VITASDK/bin:$PATH"

HERE="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
cd "$HERE"

CC=arm-vita-eabi-gcc

# -Wl,-q keeps relocation sections: vita-elf-create consumes them, and our
# loader wants them too. -nostdlib + freestanding: self-contained runtime, so
# the only imports are Sony NID stubs (no newlib) - a small, clean loader
# surface and a license-clean binary.
"$CC" \
  -march=armv7-a -mtune=cortex-a9 -mfpu=neon -mfloat-abi=hard \
  -std=c11 -O2 -ffreestanding -fno-builtin \
  -Wall -Wextra -Wno-unused-parameter \
  -nostdlib -nostartfiles -e _start -Wl,-q \
  -I"$VITASDK/arm-vita-eabi/include" \
  cube.c \
  -L"$VITASDK/arm-vita-eabi/lib" \
  -lSceGxm_stub -lSceDisplay_stub -lSceCtrl_stub -lSceSysmem_stub -lSceLibKernel_stub \
  -o cube.elf

# Convert the linked ELF into a Vita executable (velf): encodes the NID import
# tables the loader resolves. No crypto - a velf is the decrypted form a SELF
# wraps, and we own the loader so we skip the SELF/fself layer entirely.
vita-elf-create cube.elf cube.velf

echo "OK: cube.elf ($(stat -c%s cube.elf) B), cube.velf ($(stat -c%s cube.velf) B)"
