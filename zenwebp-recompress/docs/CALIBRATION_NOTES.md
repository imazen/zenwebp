# Calibration notes — findings from sweep runs

A running log of insights from `zwr-calibrate` sweeps. Each entry is a
short hypothesis + measurement + actionable next step.

## 2026-05-28 — smoke run on `dev/smoke_corpus`

**Setup.** 5 files from `zenwebp/tests/images/gallery1/`, targets
60/70/80/90, strategies `remux,reencode,vp8l`.

**Findings.**

1. **`vp8l` always grows photo content 5-6×.** Confirms design intuition:
   the router should never propose lossless reencode for photo input. The
   eventual content classifier needs to gate VP8L on (`Screen` |
   `LineArt` | `Mixed-low-entropy`).

2. **`remux` is a true no-op when there is no metadata.** Gallery files
   ship without EXIF/XMP, so re-mux size == input size. This is correct
   behavior; we should still publish it because the *normalized chunk
   ordering* protects forward callers — but the marketing claim "remux
   shrinks by stripping metadata" is conditional on metadata existing.

3. **`reencode` at target_q ≈ source_q usually GROWS.** File 2 has
   `source_q ≈ 31` per `detect`. Re-encoding at target_q=70 (the
   default `Reencode` calibration anchor) **grows** the file by 1.01×
   with measured generation-loss zensim 65. This is the classic "round-
   trip inflation" — when a file was heavily quantized and you re-encode
   it at higher Q, you're adding bits to encode the quantization
   artifacts faithfully.

   **Action:** the calibration anchor for `target_q_for_target_zensim_a`
   must be conditioned on `source_q`. Re-encoding shrinks only when the
   gap `source_q - target_q` is meaningfully positive (heuristic: ≥10
   IJG points).

4. **`source_q = 1.0` for file 1.** `detect::quant_index_to_quality`
   bottomed out, likely because file 1 has a near-maximum quantizer
   (`qi ≈ 120+`). Two possibilities:
   - The file genuinely has very low quality (would explain why all
     re-encodes drop generation-loss zensim to 56-68).
   - The encoder used a non-libwebp q-mapping and the inversion is
     wrong.

   Either way, the *raw quantizer_index* is the encoder-independent
   ground truth. **Action:** when the calibration table lands, key on
   `quantizer_index` (or `(quantizer_index, encoder_family)` joint),
   NOT on `quality_estimate`. The derived `source_q` is fine for
   end-user reporting but is too lossy as a calibration key.

5. **Routing decisions hinge on the projection table — the placeholder
   table is too pessimistic.** Every target sample (60/70/80/90)
   produced `LosslessOnly: NoStrategyMeetsTarget` from the public
   `recompress()` API on the smoke set. The data shows real wins are
   available (file 1 reencode q60: 0.73× shrink), but the projection
   table's `predict_zensim_a_after_reencode` calc undershoots. Fitting
   the table from this smoke data + a larger sweep is the only fix.

## Next sweep

- **Inputs.** Paired-reference corpus: PNG sources + libwebp-encoded
  variants at q ∈ {20, 25, …, 95, 100} (17 q levels) for each PNG.
- **Sources.** 50 cid22 + 100 codec-corpus `sc` + 30 line-art + 20
  Sharp-encoded baselines from the wild = 200 reference PNGs × 17 q =
  3,400 input WebPs.
- **Sweep grid.** 21 target zensim-A (0:100:5) × 4 strategies = 84
  cells per input, ~285,600 cells total.
- **Output.** Per-cell `(size_ratio, zensim_a_vs_source,
  zensim_a_vs_reference, butter_pnorm3_vs_reference)`.
- **Fit.** Per-encoder per-content-class linear in (`source_qi`,
  `target_zensim_a`); ship p25/p50/p75 quantiles in
  `data.parquet`.
