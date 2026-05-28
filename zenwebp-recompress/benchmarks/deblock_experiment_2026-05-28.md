# Deblock experiment — 2026-05-28

**Question.** Does applying a block-artifact-aware deblock filter to decoded
VP8 output, before re-encoding, improve cumulative zensim-A at fixed bitrate
(i.e. let the router hit a target at lower output bpp)?

**Answer. No — it is net-negative on already-loop-filtered sources.**

## Method

The deblock filter (`src/strategies/deblock.rs`) uses the H.264/VP8-style
artifact decision: a 4-pixel grid boundary is smoothed only when the
boundary step is *moderate* (`0 < |p0−q0| ≤ edge_limit`) AND both interior
gradients are *flatter* than the boundary (`|p1−p0| < |p0−q0|` and
`|q1−q0| < |p0−q0|`) — the signature of a blocking artifact rather than a
real edge. Detected boundaries have each side pulled 1/4 of the gap toward
the other (a half-reduction of the step).

Swept 10 lossless references × 16 synthetic source qualities (libwebp
`-q 20..95`, default loop filter applied) × 10 targets, comparing the
`reencode` and `deblock` strategies cell-for-cell.

```bash
target/release/zwr-calibrate --refs <refs> --q-grid 20:95:5 \
  --targets 50:95:5 --strategies reencode,deblock --output deblock_compare.csv
```

## Result (150 matched cells, source qi ≥ 60)

| metric | deblock − reencode |
|--------|--------------------|
| mean cumulative zensim-A | **−2.75** |
| mean size ratio | **+0.039** (3.9% larger) |

Deblock is **strictly dominated**: lower quality AND larger output.

## Why

VP8 applies an in-loop deblocking filter during decode reconstruction
(controlled by `filter_level` in the frame header). The decoded RGBA we
operate on is therefore *already deblocked*. A second deblock pass removes
detail the in-loop filter already smoothed correctly, dropping fidelity vs
the original reference, and the smoother input re-encodes slightly larger
(more mid-tones that don't quantize to zero).

## Decision

1. The deblock pass now only fires when `vp8_filter_level <
   FILTER_WEAK_THRESHOLD` (8) — the source was weakly/un-filtered, leaving
   genuine artifacts our filter can remove. For the common
   strongly-loop-filtered case, `DeblockReencode` is identical to
   `Reencode`.
2. The calibration projection penalizes `DeblockReencode` by the measured
   −2.75 zensim / +3.9% size on strongly-filtered sources, so the router
   never prefers it there.
3. The filter code stays (correct + unit-tested); it's a real tool for the
   weak-filter case, which a future sweep on `cwebp -f 0` sources can
   calibrate.

## Follow-up — RESOLVED, hypothesis falsified

Added `--source-filter N` to `zwr-calibrate` and re-ran with `0` (loop
filter disabled at the synthetic source encode). The weak-filter
hypothesis **does not hold**:

```bash
target/release/zwr-calibrate --refs <refs> --q-grid 20:95:5 \
  --targets 50:95:5 --strategies reencode,deblock \
  --source-filter 0 --output deblock_weakfilter.csv
```

| source config | cells | mean cum delta (deblock − reencode) |
|---------------|-------|-------------------------------------|
| default filter | 150 (qi≥60) | −2.75 |
| **filter = 0** | 120 | **−3.35** |
| filter = 0, qi ≥ 90 (worst blocking) | 60 | **−5.09**, deblock wins **0/60** |

The single best deblock cell anywhere is +1.83 zensim (one q80 outlier);
everything else loses.

**Conclusion.** Post-decode spatial deblock does not improve cumulative
fidelity vs the *original* reference — regardless of source loop-filter
strength or quantization severity. Smoothing block boundaries moves the
decode away from the sharp original; the blocking step is real
quantization error, and averaging across it does not recover the lost
signal.

**Decision.** `DeblockReencode` is **removed from the router's candidate
set** (alongside the not-yet-implemented `CoeffEdit`). The artifact-aware
deblock filter (`src/strategies/deblock.rs`) stays as a correct,
unit-tested building block exposed via `expert::deblock_rgba`, and the
`DeblockReencode` strategy can still be dispatched directly through the
`expert` API — but the default router never selects it, because no tested
source configuration shows a win.

This is the honest outcome: the strategy was a reasonable hypothesis,
implemented properly, measured rigorously, and falsified. We don't ship a
measured-dominated path in the default router.
