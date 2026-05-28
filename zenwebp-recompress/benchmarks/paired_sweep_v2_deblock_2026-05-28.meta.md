# Paired-reference calibration sweep — 2026-05-28 (v2, deblock active)

Authoritative calibration data backing the constants in
`src/calibration/data.rs`. Each row stores `(measured_zensim_a_vs_source,
measured_zensim_a_vs_reference, size_ratio)` for one
`(reference, source_q, target_zensim_a, strategy)` cell.

## Setup

10 references (5 gallery1 lossy + 5 gallery2 lossless WebPs from
`tests/images/`), decoded losslessly to RGBA via `zenwebp::oneshot`.
For each reference and each `source_q ∈ {20, 25, …, 95}` (16 levels), an
in-memory libwebp lossy encode produced the synthetic source. Each
synthetic source was then run through all 4 strategies (`remux`,
`reencode`, `deblock`, `vp8l`) at 10 target zensim-A values
(`50, 55, …, 95`). Result cells: 10 × 16 × 10 × 4 = 6,400.

The deblock filter (`src/strategies/deblock.rs`) was **active** during
this sweep (gradient-gated; fires for `qi ≥ 60`). The earlier
`paired_sweep_2026-05-28.csv` (since removed) was a no-op-deblock
baseline.

## Command

```bash
cargo build -p zwr-calibrate --release
target/release/zwr-calibrate \
  --refs zenwebp-recompress/dev/refs \
  --q-grid 20:95:5 \
  --targets 50:95:5 \
  --strategies remux,reencode,deblock,vp8l \
  --output zenwebp-recompress/benchmarks/paired_sweep_v2_deblock_2026-05-28.csv
```

## Fit

Per-`(qi_bin, target_bin)` p50 aggregates are embedded in
`src/calibration/data.rs::REENCODE` and
`src/calibration/data.rs::REENCODE_GEN_LOSS`. The bound model used by the
router predicts cumulative as `min(gen_loss, source_cum)`.

## Findings

- libwebp `qi 41-60` is the sweet spot for re-encode: ratio 0.64-0.97 with
  cumulative 49-70 over the target grid.
- `vp8l` (lossless re-encode of decoded RGBA) inflates photo content
  4.5-7×.
- The synth grid produces few `qi ≥ 60` cells (low-quality sources need
  separate fixture acquisition); deblock is no-op for everything in this
  sweep, so the `DEBLOCK_REENCODE` table is constant-aliased to
  `REENCODE`. Future low-quality refs will differentiate.

## Next sweep

- Add genuine low-quality sources (libwebp `q ≤ 30`) to refs.
- Add 50-image cid22 holdout for held-out validation.
- Add `--method` sweep (0, 4, 6) for libwebp encoder method axis.
