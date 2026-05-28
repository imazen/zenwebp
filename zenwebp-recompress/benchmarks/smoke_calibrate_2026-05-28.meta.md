# Smoke calibration sweep — 2026-05-28

First end-to-end run of `zwr-calibrate` against a 5-file `dev/smoke_corpus`
(copies of `zenwebp/tests/images/gallery1/{1..5}.webp`). Sanity-check only,
not a calibration fit.

## Command

```bash
cargo build -p zwr-calibrate --release
target/release/zwr-calibrate \
  --sources dev/smoke_corpus \
  --targets 60:90:10 \
  --strategies remux,reencode,vp8l \
  --output benchmarks/smoke_calibrate_2026-05-28.csv \
  --max-files 5
```

## Cells

5 files × 4 targets × 3 strategies = 60 rows.

## Findings

1. Pipeline runs clean end-to-end (no crashes, no NaN where data should be).
2. Per-strategy dispatch via `expert::run_*` gives proper data for every
   cell regardless of router opinion.
3. `vp8l` (lossless reencode) **always grows photo content by 5-6×** —
   confirms the design intuition that lossless reencode should *only* be
   considered for screen / line-art content.
4. `reencode` shrinks the file at target q60 on Q1 source by ~27%
   (zensim-A 56) but inflates by 29% at target q90.
5. `remux` is a true no-op on these inputs (no EXIF/XMP to strip): size
   ratio = 1.0 exactly.
6. **Detect quirk:** file 1's source quality probes as `1.0` — likely a
   detect.rs edge case (quantizer near 127). Worth investigating but does
   not block the calibration fit.

## Next sweep

For real calibration:

- Corpus: codec-corpus `sc` 100 files + cid22 holdout 50 + screen-content
  20.
- Targets: `0:100:5` (21 bins).
- Strategies: `remux,reencode,deblock,vp8l` (CoeffEdit not yet implemented).
- Add second pass with cumulative `zensim_a_vs_reference` once the
  PNG-source-paired corpus is staged under
  `/mnt/v/zen/zenwebp-recompress/corpus/`.

Expected cell count: ~200 files × 21 × 4 = ~16,800 rows. ~5 min wall time
at the rate observed (60 cells in ~5 s with rayon).
