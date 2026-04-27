# Issue #32 — Tiny low-color images at m0 produce ~4x libwebp size

## Status

**Open. Not absorbed by #30.** Needs a non-trivial port of libwebp's `FastMBAnalyze`
+ `RefineUsingDistortion` mode-selection flow. Diagnostic example landed:
`dev/issue_32_baseline.rs`.

## Reproduction

```
cargo run --release --features _profiling --example issue_32_baseline -- \
    /home/lilith/work/all-the-images/corpus/png/imagemagick-convert/d8/d88de4b9e6efe211.png
```

Worst offender after the #30 segment fix (commit `1e35476` on
`fixall/libwebp-parity`) — measured 2026-04-26 against current
`fixall/libwebp-parity` head (`0522ccb`):

| preset  | q  | m | zen   | lib | ratio |
|---------|----|---|-------|-----|-------|
| Photo   | 75 | 0 | 1454  | 318 | 4.572 |
| Photo   | 90 | 0 | 1772  | 434 | 4.083 |
| Photo   | 95 | 0 | 1882  | 424 | 4.439 |
| Photo   | 95 | 4 | 394   | -   | -     |
| Default | 95 | 0 | 1858  | -   | -     |

(Note: empirical sweep `differences/05-empirical-analysis.md` reported 1696/424 =
4.000x for this image at Photo Q95 m0. Current numbers are slightly higher;
intervening parity changes shifted absolute byte counts but the qualitative
4x blowup persists across presets and qualities.)

At m4+ this image encodes to ~250-394 bytes — the bug is **specific to m0/m1**.

## Root-cause hypothesis (not yet patched)

libwebp at m0/m1 takes a fundamentally different mode-selection path that
zenwebp does not currently model:

1. **Analysis pass (`analysis_enc.c:260` `FastMBAnalyze`)**: at `enc->method <= 1`,
   replaces full DCT-based `MBAnalyzeBestIntra16Mode` with a DC-variance test:
   ```c
   // threshold = 8 + (17-8)*q/100
   // m  = sum(dc[k]),  m2 = sum(dc[k]^2)
   if (kThreshold * m2 < m * m) {
       VP8SetIntra16Mode(it, 0);          // I16-DC
   } else {
       VP8SetIntra4Mode(it, all_DC);      // I4 with all sub-blocks DC
   }
   ```
   The mode decision is **persisted to `it->mb`** for the encode pass to read.

2. **Encode pass (`quant_enc.c:1419` `VP8Decimate` -> `RefineUsingDistortion`)**:
   at m0, called with `try_both_modes=method>=2=0` and
   `refine_uv_mode=method>=1=0`:
   - `is_i16 = it->mb->type == 1` — uses the mode set by `FastMBAnalyze`.
     Does NOT re-evaluate I16 vs I4.
   - For I16, only does SSE-based scoring across 4 modes (no DCT, no
     coefficient cost).
   - For I4 (when `FastMBAnalyze` chose intra4), tries 10 modes per sub-block
     by SSE only.
   - **Skips UV mode refinement entirely** — UV stays at default DC mode.

By contrast, zenwebp's m0/m1 path:
- `pick_intra16_fast_dc` (mode_selection.rs:961-981): always returns I16-DC,
  regardless of source variance. Never produces I4.
- `choose_macroblock_info` (mode_selection.rs:1851-1853): explicitly forces
  I16-only at m0/m1 (`if self.method <= 1 || self.partition_limit >= 100`).
- `pick_best_uv` (mode_selection.rs:1548): runs full RD UV mode evaluation
  at every method, including m0.

## Experiments attempted in this branch

- **Forcing UV mode to DC at m0/m1** — measured zero size change on the worst
  offender. Chroma mode wasn't the culprit on this image (UV plane is
  effectively flat for an 8-bit grayscale source). Reverted.

## Why this is non-trivial

The libwebp flow requires the **analysis pass to write per-MB mode decisions**
that the encode pass then consumes. zenwebp's `analyze_image`
(analysis/mod.rs:268) currently only collects alpha histograms and per-MB alpha
values for segmentation — it does not persist mode decisions per MB. Wiring
that up cleanly requires:

1. Extending `AnalysisResult` to carry per-MB mode info (intra16-DC vs
   intra4-all-DC) when `method <= 1`.
2. Threading that decision into `choose_macroblock_info` so m0/m1 can branch
   on it (DC-variance-low → I16-DC; DC-variance-high → I4-all-DC, which means
   `LumaMode::B` with a `[0u8; 16]` bpred array).
3. Skipping `pick_best_uv` at m0/m1 (force `ChromaMode::DC`), per
   `RefineUsingDistortion(refine_uv_mode=0)`.
4. Reproducing libwebp's `IsFlatSource16` border-resonance fix
   (quant_enc.c:1336-1342) that overrides best_mode on top-row/left-column MBs.

Risk surface: m0/m1 is currently rarely tested in CI (the parity sweep focuses
on m4-m6). Changes here could regress other low-method content that depends on
the current behavior (e.g., the corpus_test outputs for Default/Drawing m0 on
photo-grade content where zenwebp is currently ~10% smaller than libwebp at
low quality — see `05-empirical-analysis.md` rows where m0 has
agg ratio < 1.0).

A proper fix needs the segment-map and mode-persistence work before the
FastMBAnalyze port itself, and must be sweep-validated against the full
24,240-row empirical baseline so the low-q-m0-where-we-currently-win cells
don't regress.
