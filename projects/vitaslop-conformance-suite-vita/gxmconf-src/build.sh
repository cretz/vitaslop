#!/bin/bash
# Reproduce the gxmconf corpus artifacts (gxmconf.elf, gxmconf.velf) from gxmconf.c.
#
# Requires a Vita toolchain on PATH via $VITASDK (arm-vita-eabi-gcc,
# vita-elf-create). The toolchain is a build tool only - compiling our MIT,
# -nostdlib source with it does not attach any GPL/LGPL obligation to the output.
#
# Deterministic: run this, then `git diff` should be clean. Run under WSL:
#   VITASDK=$HOME/vitasdk bash build.sh
#
# NOTE for a Windows checkout: if git gave this file CRLF line endings, bash
# refuses `set -euo pipefail` with "invalid option name". `.gitattributes` now
# pins `*.sh` to `eol=lf` so a fresh checkout does not, but a file already on
# disk keeps whatever endings it has. Strip them into a copy AND tell the copy
# where the source lives - it cannot work that out from /tmp:
#   D=$PWD; sed 's/\r$//' build.sh > /tmp/b.sh
#   GXMCONF_SRC=$D VITASDK=$HOME/vitasdk bash /tmp/b.sh
#
# The old note here said only the first half, and the copy then compiled nothing
# and reported `gxmconf.c: No such file or directory` - a documented recipe that
# does not work is worse than none, because it is tried first.
set -euo pipefail

: "${VITASDK:?set VITASDK to your Vita toolchain root}"
export PATH="$VITASDK/bin:$PATH"

HERE="${GXMCONF_SRC:-$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)}"
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
  gxmconf.c \
  -L"$VITASDK/arm-vita-eabi/lib" \
  -lSceGxm_stub -lSceSysmem_stub -lSceLibKernel_stub -lSceProcessmgr_stub \
  -o gxmconf.elf

vita-elf-create gxmconf.elf gxmconf.velf

echo "OK: gxmconf.elf ($(stat -c%s gxmconf.elf) B), gxmconf.velf ($(stat -c%s gxmconf.velf) B)"
