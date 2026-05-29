#!/usr/bin/env bash
# Build the zenwebp-recompress calibration corpus: multi-class, multi-size,
# lossless-WebP references for `zwr-calibrate --refs`.
#
# Per the source-informing sweep discipline (CLAUDE.md): 4 content classes,
# size variants (tiny/small/medium + native-capped) via a principled kernel
# (Mitchell-Netravali), DOWNSCALE ONLY (no upscale — synthetic upscales mislead
# any size-conditioned fit), train/val split for held-out validation.
#
# Output layout (consumed by zwr-calibrate + fit_calibration.py):
#   $OUT/{train,val}/<class>/<stem>_<maxdim>.webp   (lossless VP8L)
#
# Usage: build_calib_corpus.sh [N_PER_CLASS] [OUT_DIR]
set -euo pipefail

N="${1:-50}"
OUT="${2:-/mnt/v/input/zenwebp-recompress-calib}"
SIZES=(64 256 1024)          # max-dim downscale targets (skip if >= native)
NATIVE_CAP=2048              # also emit native, capped to bound compute
VAL_EVERY=5                 # every 5th sampled ref -> validation (20%)

# class -> source dir (clean originals; PNG/JPG/WebP)
declare -A SRC=(
  [photo]="/mnt/v/work/corpus/CID22-512"
  [screen]="/mnt/v/input/imazen-26-screenshots-2026-05-28"
  [lineart]="/mnt/v/input/zensim/text-ui"
  [mixed]="/mnt/v/input/datasets/aic3/data/original"
)

emit() {  # emit <src> <class> <split> <stem>
  local src="$1" class="$2" split="$3" stem="$4"
  local nw nh nmax dims
  # NB: `identify -format` emits no trailing newline, so `read < <(...)` would
  # return non-zero on EOF and bail; capture then feed via here-string.
  dims=$(identify -format "%w %h" "$src[0]" 2>/dev/null) || return 0
  read -r nw nh <<< "$dims"
  [[ -z "${nw:-}" || -z "${nh:-}" ]] && return 0
  nmax=$(( nw > nh ? nw : nh ))
  local dst="$OUT/$split/$class"; mkdir -p "$dst"
  # downscale variants
  for s in "${SIZES[@]}"; do
    if (( s < nmax )); then
      convert "$src[0]" -filter Mitchell -resize "${s}x${s}>" \
        -define webp:lossless=true "$dst/${stem}_${s}.webp" 2>/dev/null || true
    fi
  done
  # native (capped, no upscale)
  local cap=$(( nmax < NATIVE_CAP ? nmax : NATIVE_CAP ))
  convert "$src[0]" -filter Mitchell -resize "${cap}x${cap}>" \
    -define webp:lossless=true "$dst/${stem}_n${cap}.webp" 2>/dev/null || true
}

total=0
for class in "${!SRC[@]}"; do
  dir="${SRC[$class]}"
  [[ -d "$dir" ]] || { echo "WARN: $class src missing: $dir" >&2; continue; }
  mapfile -t files < <(find "$dir" -maxdepth 6 -type f \
    \( -iname '*.png' -o -iname '*.jpg' -o -iname '*.jpeg' -o -iname '*.webp' \) | sort)
  cnt=${#files[@]}
  (( cnt == 0 )) && { echo "WARN: $class no files" >&2; continue; }
  # even stride to sample N spread across the corpus (diversity > head -N)
  stride=$(( cnt / N )); (( stride < 1 )) && stride=1
  picked=0; i=0
  while (( i < cnt && picked < N )); do
    src="${files[$i]}"
    stem="$(basename "$src")"; stem="${stem%.*}"
    if (( picked % VAL_EVERY == 0 )); then split=val; else split=train; fi
    emit "$src" "$class" "$split" "$stem"
    picked=$(( picked + 1 )); i=$(( i + stride ))
  done
  echo "$class: sampled $picked of $cnt"
  total=$(( total + picked ))
done
echo "== picked $total refs across classes =="
echo "== variant counts =="
for split in train val; do
  for class in "${!SRC[@]}"; do
    c=$(find "$OUT/$split/$class" -name '*.webp' 2>/dev/null | wc -l)
    printf "  %-5s %-8s %s\n" "$split" "$class" "$c"
  done
done
echo "OUT: $OUT"
