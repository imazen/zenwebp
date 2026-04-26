# Libwebp Parity Fix Campaign — 2026-04-26 Session Summary

## Headline result

**Production benchmark CID22 Q75 m4 (Default preset, SNS=50, filter=60): 1.0149x → 1.0042x**

(72-cell × 50-image full sweep on CID22 training set, in `post-batch3-FULL.tsv`.)

**Aggregate across all 72 cells (4 presets × 6 qualities × 3 methods × 50 images): 0.9839x** —
zenwebp is now 1.6% **smaller than libwebp** on average across the full preset/quality/method
matrix. 37 of 72 cells beat libwebp on bytes. No cell exceeds 1.05x. Worst cell:
Default Q95 m0 = 1.0475x.

m4 cells mean: 1.0029x (within 0.3% of libwebp).
m6 cells mean: **0.9980x** (beats libwebp).
m0 cells mean: 0.9510x (already smaller, but still has tail outliers tracked in #32).

The CLAUDE.md headline figure is fixed via 13 targeted fixes in one autonomous session.

## Aggregate impact (CID22 25-image sample, 36 cells = 3 presets × 4 q × 3 methods)

| Metric | Pre-fix | Batch 1 | Final (Batch 1+2+3) |
|--------|---------|---------|---------------------|
| Mean ratio across all cells | 1.0140 | 0.9988 | **0.9982** (−1.58% Δ) |
| m4 cells mean | 1.0204 | 1.0080 | **1.0016** (−1.88%) |
| m6 cells mean | 1.0157 | 0.9989 | **0.9968** (−1.89%) |
| m0 cells mean | 0.9569 | 0.9477 | 0.9472 (−0.97%) |
| Production (Default Q75 m4) | 1.0200 | 1.0033 | **1.0028** (−1.73%) |
| Best (Photo Q25 m4) | 1.0422 | 1.0129 | **1.0072** (−3.50%) |

Many m6 cells now beat libwebp on bytes (Default 50/75/90, Drawing all q, Photo 50/75).

## Fixes landed (PR #37 on `fixall/libwebp-parity`)

| Issue | Title | Commit | Impact area |
|-------|-------|--------|-------------|
| #21 | Mode-cost tables in zenwebp enum order | b1a5ef0 | I4/I16/UV mode picks |
| #25 | SKIP_PROBA_THRESHOLD=250 (conservative) | cae86b3 | Per-MB skip flag |
| #26 | Segment-map smoothing default OFF | 58492cb | Segment partition |
| #28 | I16 V/H/TM modes at MB borders | 0a8a780 | Border MB modes |
| #31 | tlambda gate by method ≥ 4 | 6bcfda7 | TDisto at m0–m3 |
| #22 | I4 base penalty 3000 → 211 | c22a4f1 | I4-vs-I16 balance |
| #23 | Cross-MB nz context for I16/UV cost | 5a40df8 | Cost estimation |
| #24 | Y2 AC kAcTable2 (encoder + decoder) | 8aa43f1 | Y2 quantizer |
| #27 | Two-pass encoding at m4 (partial) | 5d61060 | Stat refresh |
| #30 A+B | Alpha range floor + 256-MB segment gate | 1e35476 | Small/flat images |
| #30 C | Segment 0 quant as bitstream base | 22e4c4e | Segment header |
| #33 | CostModel enum API | 95dfc02 | New public surface |
| #34 | VP8AdjustFilterStrength port | 6e0ad07 | Quality (filter level) |
| #29 | I16 trellis at m6 in mode selection | 108b1ff | m6 trellis |

## Deferred (separate follow-up issues)

- **#25 full token-buffer SKIP_PROBA fix** — needs encoder architecture change to emit EOB tokens for skipped MBs unconditionally so the gate can fire whenever `prob >= 250`, not just when `skip_mb == 0`.
- **#27 full multi-pass StatLoop** — m5/m6 second pass currently regresses (+0.6 to +0.9% on CID22), likely because trellis already image-adapts via per-pass `proba_stats`. Needs careful interaction analysis.
- **#32 m0 outliers** — partially absorbed by #30, but tiny low-color images (128×128 charts/icons) still need a `FastMBAnalyze` port for full parity. Worktree agent dispatched at end of session.
- **#35 cleanup tracker** — low-severity bundle of 12 cosmetic / dead-code / CPU-waste items. Includes the trellis double-work CPU optimization (~10–15% encoder speedup at m5–m6, no bitstream effect).
- **bit-exact-libwebp** in `CostModel::StrictLibwebpParity` mode — north-star issue for the `StrictLibwebpParity` enum value to produce byte-identical output to libwebp's `cwebp`. Depends on the four items above plus the existing `CostModel` plumbing.

## Future enhancements (separate issues opened)

- Brute-force per-image config search (zenjpeg/coefficient pattern)
- ML-driven cost-model / config selection (zentract)

## Methodology

- **Tool:** `dev/empirical_sweep.rs` (size-only) and `dev/parity_baseline.rs` (size + butteraugli + ssim2 + speed). Both committed on this branch.
- **Comparison side:** `webpx 0.1.4` → libwebp v1.x in same process.
- **Corpus:** CID22-512 training set, 25-image stratified sample for fast iteration; 50-image full sweep for final validation.
- **Configs:** 3 presets × 4 qualities × 3 methods = 36 cells per measurement.
- **Each fix verified:** post-fix sweep before moving on, with cross-cell comparison documented in `post-batch*-*.tsv`.

## Test plan

- [x] `cargo test --release` — 238 passed, 0 failed at every commit
- [x] SIMD tier-parity unchanged (byte-identical across scalar/SSE2/SSE4.1/AVX2/AVX-512 tiers, verified by `tests/simd_tier_parity.rs`)
- [x] Per-fix bisection on the one regression (#25 partial token-elision interaction with conservative gate found and resolved)
- [ ] Push fixall + wait for CI on all 6 platforms

## Notes

- The audit's expected impact estimates were sometimes optimistic — #23 (cross-MB nz context) was predicted 0.5–1.5% but landed near-zero (still kept for algorithmic correctness and to support later fixes).
- The biggest single-fix wins were #21 (table indexing), #28 (border modes), and #34 (filter strength) — all "obvious bugs" that nobody had flagged before the audit.
- Multiple measurements showed that Photo preset at low Q saw the most improvement (~3% range), consistent with the audit's hypothesis that under-priced I4 (#22) + wrong mode-cost tables (#21) hurt textured content most.
