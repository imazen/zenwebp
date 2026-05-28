# Source quality detection — findings (2026-05-28)

## The header-only base quantizer is unreliable

`zenwebp::detect::probe` reads the VP8 frame header's base quantizer
(`yac_abs`) and inverts libwebp's quality→quantizer curve to estimate the
source quality. **Measured against the true encode quality, this estimate
is essentially uncorrelated** for real (segmented) WebP files:

| true encode q | detect source_q | detect qi |
|---------------|-----------------|-----------|
| 40 | 1   | 124 |
| 50 | 4   | 97  |
| 55 | 2   | 97  |
| 75 | 25  | 60  |
| 90 | 52  | 39  |
| 95 | 42  | 44  |

(9 lossless references re-encoded at each q; values averaged.)

### Why

libwebp enables **segmentation** by default: the image is partitioned into
up to 4 segments, each with its own quantizer delta off the base
quantizer. The base `yac_abs` in the header is just the anchor — the
*effective* per-block quantizer is `base + segment_delta`, and the segment
map + deltas are what actually determine quality. A header-only reader sees
only the base, which can sit anywhere relative to the real working
quantizers. So the inverted "quality" is noise.

This is the WebP analog of the well-known difficulty estimating JPEG
quality from quantization tables when the encoder used custom/scaled
tables.

## Consequence for calibration

Any calibration keyed on the header quantizer is keyed on noise. The first
calibration attempt produced a **non-monotonic** `source_cum` table
(qi-bin 41-60 → 80, qi-bin 0-20 → 64) for exactly this reason: each qi bin
mixed unrelated images. Keying on `detect::source_q` is no better.

## The fix: decode-based quality estimation

`recompress()` decodes the source anyway (every pixel-domain strategy
needs the RGBA). The reliable quality signal is **recompression
self-consistency**: encode the decoded pixels across a probe quality sweep
and find the quality at which the re-encoded size matches the source size.
A source encoded at quality Q reproduces ~its own size when re-encoded near
Q; far above Q it inflates, far below it shrinks. The size-match crossing
is monotonic in true Q (with a roughly constant ~+10 offset from the
2nd-generation size shrink, which we calibrate out).

This is implemented in `src/estimate.rs::estimate_quality_by_recompression`
and used by the `recompress()` path as the calibration key. `plan()`
stays header-only and is therefore explicitly a **rough** preview — its
projected numbers carry the header-detection caveat.

## Safety net regardless of estimate accuracy

The router's **ground-truth size guard** (router.rs) re-checks the actual
output size after running any strategy and falls back to a lossless re-mux
if the output didn't shrink. So even when the quality estimate is off, the
crate never ships a larger-at-lower-quality file — it just may be more
conservative than optimal. Accuracy of the estimate affects *how good* the
pick is, never *correctness*.

## Re-fit / re-validate

The calibration constants in `src/calibration/data.rs` are fit from a
clean lossless-only sweep (9 refs; 3 photo refs drive the photo curve).
This is below the 50-images-per-class bar in the global sweep discipline —
treat the constants as **preliminary**. Re-fit with a larger corpus:

```bash
cargo build --release -p zwr-calibrate
# Lossless-only references (true originals — never mix lossy files in):
target/release/zwr-calibrate --refs <dir of LOSSLESS webp/png> \
  --q-grid 20:95:5 --targets 45:95:5 --strategies remux,reencode,vp8l \
  --output benchmarks/clean_sweep_<date>.csv
# Then re-derive: bin reencode rows by TRUE source q (the synth_q label,
# NOT detect's estimate), per content class.
```
