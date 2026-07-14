# Bit-exact libwebp parity (#38) — feasibility + divergence inventory

**Date:** 2026-07-14
**Config under test:** CID22 382297 (512×512) and a 32×32 synthetic gradient,
q75, `CostModel::StrictLibwebpParity`, vs libwebp-sys 0.14.4 (`webpx`).
**Harness:** `dev/bitexact_diff.rs` (mode streams) + an instrumented libwebp
`cwebp-trace` build and a zen `MBDUMP` dump comparing per-MB coefficient
signatures `(signed-sum, abs-sum, nonzero-count)` over the quantized levels.

## Feasibility: YES — the forward transform + quantization is bit-identical

Where zen and libwebp pick the **same** intra mode, the quantized coefficient
levels are **identical**. On the 32×32 gradient at m0:

| MB | mode (both) | zen luma sig | libwebp luma sig |
|----|-------------|--------------|------------------|
| (0,0) | V  | −169 / 169 / 37 | −169 / 169 / 37 |
| (0,1) | H  | −78 / 102 / 37  | −78 / 102 / 37  |

MB(0,0) UV also matches (0 / 36 / 12 both). Three independent sums over the
272 luma coefficients agreeing exactly is a strong hash — the FTransform +
QuantizeBlock path reproduces libwebp bit-for-bit. **Bit-exact output is
reachable by aligning the DECISION logic; no transform/quant rewrite is
needed.**

## Remaining divergences (all decision-level), m0 segs1, 382297

1. **Luma mode selection ~95% agree** (was 0% — every MB collapsed to I16-DC
   until the single-segment FastMBAnalyze-hint fix, 331f386). I4-vs-I16 split
   now matches libwebp exactly (136 = 136 I4 MBs). Residual ~5% is
   SSE/cost tie-breaking in `pick_intra16_sse`.
2. **UV mode selection ~45% agree at m0.** libwebp m0 uses `refine_uv_mode=0`
   — the UV mode is the analysis pass's `MBAnalyzeBestUVMode` alpha/entropy
   pick, left untouched by RefineUsingDistortion. zen m0 uses an RD/SSE pick.
   Parity requires storing the analysis UV mode per-MB and using it at m0.
3. **Coefficient probabilities.** libwebp ships DEFAULT probas at m0-m2 with a
   single segment: `OneStatPass` returns `size_p0 = ΣH + segment_hdr.size = 0`
   (H is 0 at RD_OPT_NONE, segment header is 0 at 1 segment), StatLoop treats 0
   as failure and returns before `FinalizeTokenProbas`. zen updates probas
   (396 updates). This is the dominant byte gap at m0-m2 segs1 (zen 56 KB @
   29.38 dB vs libwebp 76 KB @ 29.38 dB — identical PSNR, libwebp just codes
   the same pixels with worse probabilities). Parity needs a
   StrictLibwebpParity gate that ships defaults under the same condition.
4. **I4 sub-mode agreement ~2%** (b4_same 3/136). Even where both pick I4, the
   16 per-sub-block modes rarely all match — `pick_intra4_sse` scoring detail.
5. **Skip decisions + segment analysis (alpha/beta/k-means)** at segs>1.

## What shipped from this investigation

- `fix(vp8): run FastMBAnalyze hints at m0/m1 with a single segment` (331f386)
  — a real quality bug (all-DC collapse) affecting every m0/m1 single-segment
  encode, not just parity. PSNR at m0 went from degraded to matching libwebp's
  exactly (29.38 dB).
- **Proba bail under `StrictLibwebpParity`** (item 3): at method ≤ 2 with one
  segment, ship default coefficient probabilities instead of the image-adapted
  update, matching libwebp's `StatLoop` early return. m0 size vs libwebp went
  **0.7393× → 0.9954×** (within 0.5%); m0/m1/m2 proba-update counts now both 0.
  The tuned default (`ZenwebpDefault`) is unaffected — it keeps the smaller,
  image-adapted proba coding. Remaining m2 gap (1.06×) is mode divergence, not
  probas.

## Byte-ratio snapshot vs libwebp (382297, q75, segs1, StrictLibwebpParity)

| method | before this work | after hint-fix + proba-bail |
|--------|------------------|-----------------------------|
| m0 | all-DC (broken) | **0.9954×** @ equal PSNR |
| m1 | all-DC (broken) | 1.0119× |
| m2 | 0.9193× (wrong probas) | 1.0645× (probas now match; modes differ) |
