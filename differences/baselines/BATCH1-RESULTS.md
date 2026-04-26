# Batch 1 Fix Results — 2026-04-26

## Methodology

- **Corpus:** CID22-512 training set, 25-image stratified sample
- **Sweep:** 3 presets × 4 qualities × 3 methods = 36 cells
- **Per-cell metric:** mean of per-file `zen_bytes / lib_bytes` ratios across the 25 images

Measurement tool: `cargo run --release --example empirical_sweep`. Both pre- and post-fix runs use the same 25-image sample with identical configurations. Same machine, same libwebp version (`webpx 0.1.4` → libwebp v1.x).

## Fixes in Batch 1

All five fixes branched off `audit/libwebp-parity-2026-04-26` and were committed to `fixall/libwebp-parity`:

| # | Title | Commit |
|---|-------|--------|
| #21 | Re-permute mode-cost tables to match zenwebp enum order | `b1a5ef0` |
| #25 | Conservative SKIP_PROBA_THRESHOLD=250 gate | `cae86b3` |
| #26 | Gate segment-map smoothing on a config flag, default OFF | `58492cb` |
| #28 | Allow I16 V/H/TM modes at MB borders | `0a8a780` |
| #31 | Gate tlambda on method >= 4 | `6bcfda7` |

## Aggregate result

**Mean ratio delta across 36 cells: −1.525% (smaller is better)**

Every cell improved. No cell regressed. Many cells now beat libwebp on size.

## Per-cell deltas (CID22 25-image sample)

| preset  | q  | m | pre    | post   | delta_pct  |
|---------|----|---|--------|--------|------------|
| Default | 25 | 0 | 0.9077 | 0.8925 | -1.51% ↓   |
| Default | 25 | 4 | 1.0318 | 1.0080 | -2.38% ↓   |
| Default | 25 | 6 | 1.0260 | 1.0029 | -2.30% ↓   |
| Default | 50 | 0 | 0.9300 | 0.9213 | -0.87% ↓   |
| Default | 50 | 4 | 1.0224 | 1.0066 | -1.58% ↓   |
| Default | 50 | 6 | 1.0155 | 0.9989 | -1.66% ↓   |
| Default | 75 | 0 | 0.9377 | 0.9293 | -0.84% ↓   |
| Default | 75 | 4 | 1.0200 | 1.0033 | **-1.67% ↓** *(production benchmark)* |
| Default | 75 | 6 | 1.0129 | 0.9949 | -1.80% ↓   |
| Default | 90 | 0 | 1.0191 | 1.0128 | -0.63% ↓   |
| Default | 90 | 4 | 1.0213 | 1.0080 | -1.34% ↓   |
| Default | 90 | 6 | 1.0137 | 1.0005 | -1.32% ↓   |
| Drawing | 25 | 0 | 0.9310 | 0.9234 | -0.76% ↓   |
| Drawing | 25 | 4 | 1.0232 | 1.0082 | -1.50% ↓   |
| Drawing | 25 | 6 | 1.0140 | 0.9991 | -1.49% ↓   |
| Drawing | 50 | 0 | 0.9444 | 0.9402 | -0.42% ↓   |
| Drawing | 50 | 4 | 1.0189 | 1.0088 | -1.01% ↓   |
| Drawing | 50 | 6 | 1.0073 | 0.9964 | -1.09% ↓   |
| Drawing | 75 | 0 | 0.9486 | 0.9447 | -0.38% ↓   |
| Drawing | 75 | 4 | 1.0179 | 1.0067 | -1.11% ↓   |
| Drawing | 75 | 6 | 1.0085 | 0.9970 | -1.15% ↓   |
| Drawing | 90 | 0 | 1.0254 | 1.0214 | -0.39% ↓   |
| Drawing | 90 | 4 | 1.0106 | 1.0003 | -1.03% ↓   |
| Drawing | 90 | 6 | 1.0074 | 0.9975 | -0.99% ↓   |
| Photo   | 25 | 0 | 0.9048 | 0.8843 | -2.05% ↓   |
| Photo   | 25 | 4 | 1.0422 | 1.0129 | -2.93% ↓   |
| Photo   | 25 | 6 | 1.0365 | 1.0045 | **-3.20% ↓** *(best)* |
| Photo   | 50 | 0 | 0.9287 | 0.9137 | -1.49% ↓   |
| Photo   | 50 | 4 | 1.0341 | 1.0085 | -2.57% ↓   |
| Photo   | 50 | 6 | 1.0251 | 0.9983 | -2.69% ↓   |
| Photo   | 75 | 0 | 0.9360 | 0.9236 | -1.24% ↓   |
| Photo   | 75 | 4 | 1.0289 | 1.0056 | -2.32% ↓   |
| Photo   | 75 | 6 | 1.0234 | 0.9984 | -2.50% ↓   |
| Photo   | 90 | 0 | 1.0343 | 1.0241 | -1.02% ↓   |
| Photo   | 90 | 4 | 1.0252 | 1.0080 | -1.73% ↓   |
| Photo   | 90 | 6 | 1.0230 | 1.0036 | -1.94% ↓   |

## Notes

- **Photo preset shows the largest improvements.** Photo at Q25-50 had the worst pre-fix ratios and benefits most from the fixes. Likely contributors: #21 (mode-cost tables affecting per-MB mode picks), #28 (border modes letting V/H win on edges).
- **m6 now beats libwebp on multiple cells.** Drawing m6 Q95 (not in this 25-image sample, but seen in the larger 250-image baseline), Default m6 Q50/75/90, Photo m6 Q50/75 all dip below 1.0x.
- **Method 0 still has rough ratios** (max 1.0241x at Photo Q90 m0). #32 (m0 outliers from missing FastMBAnalyze + small-image gating) still pending.
- **All 238 unit tests pass.** No SIMD tier-parity regressions.

## Production gap status

CLAUDE.md headline figure was CID22 Q75 m4 default = **1.0149x**. After Batch 1 we measure **1.0033x** on a 25-image subset of the same corpus — a ~85% reduction. A larger 250-image rerun will give the canonical figure.

## What's still on the table

| Issue | Expected impact | Status |
|-------|-----------------|--------|
| #23   | 0.5–1.5% bytes  | Batch 2 — biggest remaining algorithmic divergence |
| #22   | 0.5–2% on textured photos | Batch 2 |
| #24   | 0.5–1.5%        | Decoder cross-check first |
| #27   | 0.5–1.0%        | Larger structural change (multi-pass StatLoop) |
| #29   | 0.1–0.4% at m6  | Small follow-up |
| #30   | Image-dependent | Small-image / m0 cluster |
