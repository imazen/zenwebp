#!/bin/bash
# Census phase-B driver (registration: benchmarks/zensim_instrument_census_2026-08-27.md):
# per-cell ZENWEBP_ZQ_START_Q from the fitted head's q0 table; same example,
# same instrument. usage: run_census_b.sh <corpus9.tsv> <q0_table.tsv> <k> <out-dir>
set -euo pipefail
C=$1; Q=$2; K=$3; O=$4
BIN=${ZW_CENSUS_BIN:-target/release/examples/zensim_census}
mkdir -p "$O/b_cells"
out=$O/census_b_k$K.tsv
: > "$out"
first=1
while IFS=$'\t' read -r path name class; do
  for t in 70 80 88; do
    q0=$(awk -F'\t' -v p="$path" -v t="$t" '$1==p && $2==t {print $3}' "$Q")
    [ -n "$q0" ] || { echo "NO q0 for $name t$t — arm void" >&2; exit 1; }
    one=$O/b_cells/one.tsv
    printf '%s\t%s\t%s\n' "$path" "$name" "$class" > "$one"
    ZENWEBP_ZQ_START_Q=$q0 nice -n19 ionice -c3 "$BIN" "$one" "$t" "$K" "$O/b_cells" 2>>"$O/b_k$K.log"
    if [ $first = 1 ]; then head -1 "$O/b_cells/census_k$K.tsv" | sed 's/$/\tseed_q0/' > "$out"; first=0; fi
    tail -n +2 "$O/b_cells/census_k$K.tsv" | sed "s/$/\t$q0/" >> "$out"
  done
done < "$C"
echo "B arm k$K -> $out ($(($(wc -l < "$out")-1)) cells)"
