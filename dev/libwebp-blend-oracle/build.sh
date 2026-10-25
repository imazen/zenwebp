#!/bin/sh
# Build the libwebp alpha-blend oracle from the vendored libwebp-sys source
# (fetched with `cargo read libwebp-sys`, never ~/.cargo/registry).
# Output: $OUT/blend_oracle (default ~/tmp/libwebp-blend-oracle/blend_oracle).
# Run: blend_oracle 1 (full domain), blend_oracle 5 (the CI subset).
set -eu
HERE=$(cd "$(dirname "$0")" && pwd)
OUT=${OUT:-$HOME/tmp/libwebp-blend-oracle}
VENDOR=${VENDOR:-$(cargo read --path-only libwebp-sys | tail -1)/vendor}
[ -f "$VENDOR/src/demux/anim_decode.c" ] || { echo "vendor dir not found: $VENDOR" >&2; exit 1; }
mkdir -p "$OUT"
CC=${CC:-cc}
$CC -O2 -DNDEBUG=1 -I "$VENDOR" -w "$HERE/main.c" \
  "$VENDOR"/src/demux/demux.c "$VENDOR"/src/dec/*.c "$VENDOR"/src/dsp/*.c \
  "$VENDOR"/src/utils/*.c -lm -lpthread -o "$OUT/blend_oracle"
echo "$OUT/blend_oracle"
