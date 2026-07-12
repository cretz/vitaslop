#!/bin/bash
# Reproduce the NEON auto-vectorization artifacts (neon.elf, neon.velf) from neon.c.
#
# Requires a Vita toolchain on PATH via $VITASDK. The toolchain is a build tool
# only - compiling our MIT, -nostdlib source with it does not attach any GPL/LGPL
# obligation to the output. Run under WSL:
#   VITASDK=$HOME/vitasdk bash build.sh
#
# NOTE: unlike the other probes this is built WITH the tree vectorizer (the default
# at -O2) - the whole point is to emit NEON data-processing.
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
  neon.c \
  -L"$VITASDK/arm-vita-eabi/lib" \
  -lSceLibKernel_stub \
  -o neon.elf

vita-elf-create neon.elf neon.velf

echo "OK: neon.elf ($(stat -c%s neon.elf) B), neon.velf ($(stat -c%s neon.velf) B)"

# Show the NEON data-processing the vectorizer emitted (sanity: all must be lifted).
echo "--- NEON data-processing ops emitted ---"
arm-vita-eabi-objdump -d neon.elf | grep -oE 'v(add|sub|mul|mla|mls|mov|movl|addl|addw|subl|subw|abd|abal|abdl|padd|paddl|padal|max|min|mull|mlal|mlsl|neg|abs)[a-z0-9.]*' | sort | uniq -c
