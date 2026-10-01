#!/usr/bin/env bash
# Run decode_parity_dump.wasm under wasmtime in N parallel shards against a
# `decode_parity_sweep --save-gen DIR --hash-out DIR/lib.hashes` directory.
# usage: dev/decode_parity_wasm.sh <wasm> <gen-dir> <out-dir> [shards]
set -euo pipefail
wasm=$1 gen=$2 out=$3 n=${4:-14}
mkdir -p "$out"
pids=()
for i in $(seq 0 $((n - 1))); do
  nice -n19 wasmtime run --dir "$gen::/data" --dir "$out::/out" "$wasm" \
    --dir /data --expect /data/lib.hashes --shard "$i/$n" --out "/out/zen.$i.hashes" \
    > "$out/log.$i" 2>&1 &
  pids+=($!)
done
rc=0
for p in "${pids[@]}"; do wait "$p" || rc=1; done
cat "$out"/zen.*.hashes | sort > "$out/zen.hashes"
echo "shards done (any-shard-nonzero=$rc); $(wc -l < "$out/zen.hashes") digests"
cat "$out"/log.* | grep -E '^[a-z_0-9]+ +(match|diff|no_expect) +[0-9]+$' \
  | awk '{a[$1" "$2]+=$3} END{for(k in a) print k, a[k]}' | sort
