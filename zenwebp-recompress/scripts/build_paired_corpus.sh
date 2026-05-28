#!/usr/bin/env bash
#
# Build a paired-reference corpus for zwr-calibrate.
#
# Usage:
#   build_paired_corpus.sh <PNG_DIR> <OUT_DIR>
#
# Walks <PNG_DIR> for *.png files. For each PNG and each q ∈ {20, 22, …, 100}
# (41 q levels), emits:
#   <OUT_DIR>/refs/<sha8>.png   — original (deduplicated by sha)
#   <OUT_DIR>/lossy/<sha8>_q<NN>.webp  — libwebp-encoded variant
#
# Requires `cwebp` on PATH (libwebp's reference CLI encoder).

set -euo pipefail

if [[ $# -lt 2 ]]; then
  echo "usage: $0 <PNG_DIR> <OUT_DIR>" >&2
  exit 2
fi

PNG_DIR="$1"
OUT_DIR="$2"

mkdir -p "$OUT_DIR/refs" "$OUT_DIR/lossy"

if ! command -v cwebp >/dev/null 2>&1; then
  echo "cwebp not found on PATH — install libwebp" >&2
  exit 1
fi

# 41 q levels: 20, 22, …, 100.
Q_LEVELS=()
for q in $(seq 20 2 100); do
  Q_LEVELS+=("$q")
done

count=0
while IFS= read -r -d '' png; do
  sha="$(sha256sum "$png" | cut -c1-8)"
  ref="$OUT_DIR/refs/$sha.png"
  if [[ ! -f "$ref" ]]; then
    cp "$png" "$ref"
  fi
  for q in "${Q_LEVELS[@]}"; do
    out="$OUT_DIR/lossy/${sha}_q${q}.webp"
    if [[ ! -f "$out" ]]; then
      cwebp -m 4 -q "$q" "$png" -o "$out" -quiet || true
    fi
  done
  ((count+=1))
  if (( count % 25 == 0 )); then
    echo "encoded $count PNGs × ${#Q_LEVELS[@]} qualities" >&2
  fi
done < <(find "$PNG_DIR" -type f -iname '*.png' -print0)

echo "done: $count PNGs × ${#Q_LEVELS[@]} qualities = $((count * ${#Q_LEVELS[@]})) WebPs"
