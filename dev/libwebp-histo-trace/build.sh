#!/bin/sh
# #71: build an instrumented libwebp encoder from the vendored libwebp-sys
# source (fetched with `cargo read libwebp-sys`, never ~/.cargo/registry).
# Output: $OUT/libwebp_trace (default ~/tmp/libwebp-histo-trace/libwebp_trace).
set -eu
HERE=$(cd "$(dirname "$0")" && pwd)
OUT=${OUT:-$HOME/tmp/libwebp-histo-trace}
VENDOR=${VENDOR:-$(cargo read --path-only libwebp-sys | tail -1)/vendor}
[ -d "$VENDOR/src/enc" ] || { echo "vendor dir not found: $VENDOR" >&2; exit 1; }
rm -rf "$OUT/src" && mkdir -p "$OUT"
cp -R "$VENDOR" "$OUT/src"
python3 "$HERE/instrument.py" "$OUT/src/src/enc/histogram_enc.c"
CC=${CC:-cc}
ARCHFLAG=""
case "$(uname -m)" in
  arm64|aarch64) ARCHFLAG="-DWEBP_HAVE_NEON=1" ;;
  x86_64) ARCHFLAG="-DWEBP_HAVE_SSE2=1 -msse4.1 -DWEBP_HAVE_SSE41=1" ;;
esac
# shellcheck disable=SC2086
$CC -O2 -DNDEBUG=1 -D_THREAD_SAFE=1 $ARCHFLAG -I "$OUT/src" -w \
  "$OUT/src"/src/enc/*.c "$OUT/src"/src/dsp/*.c "$OUT/src"/src/utils/*.c \
  "$OUT/src"/sharpyuv/*.c "$HERE/main.c" -lm -o "$OUT/libwebp_trace"
echo "$OUT/libwebp_trace"
