# libwebp histogram-clustering trace (#71)

Differential harness for zenwebp's VP8L histogram clustering against
libwebp's `histogram_enc.c`. Both sides print one line per clustering
phase and per pair evaluation; diff the two to find the first divergence.

```
# 1. instrumented libwebp (source from `cargo read libwebp-sys`)
dev/libwebp-histo-trace/build.sh            # -> ~/tmp/libwebp-histo-trace/libwebp_trace

# 2. raw RGBA of the image (the probe dumps it)
ISSUE71_DUMP_RGBA=~/tmp/issue71 cargo run --release --features mode_debug \
    --example issue71_probe -- --methods 5 <png>

# 3. traces
~/tmp/libwebp-histo-trace/libwebp_trace ~/tmp/issue71/<name>.rgba <w> <h> 5 2> lib.trace
HISTDBG=1 cargo run --release --features mode_debug --example issue71_probe -- \
    --methods 5 <png> 2> zen.trace

grep '^LHIST phase' lib.trace; grep '^ZHIST phase' zen.trace
```

`LHIST`/`ZHIST phase=...` lines carry the histogram count after each
phase (copy / bin / stochastic / greedy / remap); `push` lines are the
`HistoQueuePush` evaluations (stochastic + greedy), `bin` lines the
entropy-bin `HistogramAddEval` decisions. Indices are compact positions on
the libwebp side and original tile indices on zenwebp's, so compare the
cost columns and the phase counts, not the indices.

Everything here is committed — the previous instrumented tree lived in
`/tmp` and was lost (see `benchmarks/issue71_probe_2026-08-28.md`).
