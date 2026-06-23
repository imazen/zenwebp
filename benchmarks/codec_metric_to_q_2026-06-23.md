# Per-codec metric → quality calibration (from R2 omni fleet sweeps)

**Goal:** replace the coarse linear `butteraugli→quality` / `ssim2→quality` placeholders
in the codec `with_fidelity` overrides (`zencodec::encode::LossyTarget::{ApproxButteraugli,
ApproxSsim2}`) with measured inverse tables.

**Source:** `s3://zentrain/{cvvdp-v15rc-2026-05-18, multi-codec-2026-05-18,
omni-multi-codec-2026-05-19}/omni/*.parquet` — **779,556 cells** with `score_ssim2_gpu`
+ `score_butteraugli_max_gpu` (+pnorm3/cvvdp/dssim/iwssim/zensim), keyed on
`(image_path, codec, q, knob_tuple_json)`. Pulled + aggregated 2026-06-23. Per-q value
= median across images × knob configs, monotonized (isotonic/PAVA on the forward
q→median curve) before inversion.

## ⚠ KEY FINDING: the sweep's `q` axis means a DIFFERENT thing per codec

The zenmetrics sweep encode arms (`zenmetrics-cli/src/sweep/encode.rs`) do NOT all drive a
raw codec quality dial. **Only zenwebp's `q` is a raw quality dial.** This was missed on the
first pass and produced two wrong wirings (both since corrected):

| codec | sweep call | what `q` IS |
|---|---|---|
| **zenwebp** | `LossyConfig::with_quality(q)` (native) | **raw VP8 quality dial** ✅ usable |
| **zenjpeg** | `JpegEncoderConfig::with_generic_quality(q)` | a **zensim Profile A target** (perceptual), NOT raw JPEG quality (`zenjpeg/src/codec.rs:322`) ✗ |
| zenavif | (`with_generic_quality`) | non-monotone in the data anyway ✗ |
| zenjxl | (`with_generic_quality`) | degenerate (every q→ssim2 ≈87.9) ✗ |

So a metric→q table fit from the zenjpeg cells maps `metric → zensim-A-target`, which does
**not** transfer to mozjpeg-rs's raw `quality()` dial. The jpeg/avif/jxl `q` axes are not
raw codec quality and cannot seed `with_fidelity` tables. A dedicated **raw-quality sweep**
(vary the codec's own `quality`/`distance`, measure butteraugli-max + ssim2) is the
prerequisite for those codecs.

## Norm: `ApproxButteraugli` is the **max-norm** (confirmed)

`zencodec/src/fidelity.rs:119` — "Aim for a butteraugli **max-norm** distance (worst-region
p-norm, p→∞)." So `score_butteraugli_max_gpu` is the correct column (NOT pnorm3). Lower is
better. The doc's "≈1.0 high quality" intuition is loose for max-norm — the sweep's top
(q95) sits at max-norm ≈2.41 for zenwebp, so sub-2.41 targets clamp to q95 (no data above
q95 swept; honest clamp, not extrapolated).

## Calibration gotcha (zenwebp): set the NATIVE dial, bypass the generic remap

zenwebp's zencodec wrapper `with_generic_quality(g)` applies `calibrated_webp_quality(g)`
(non-identity: generic 25→native 18, 15→native 5) before hitting the encoder. The sweep
drove the **native** dial (`LossyConfig::with_quality`), so the table is keyed on native
quality — it must reach the encoder un-recalibrated. zenwebp routes metric targets through
a `set_native_quality` helper (`self.inner.with_quality(q)` directly). **webpx**'s
`calibrated_webp_quality` is the identity, so `with_generic_quality` already feeds libwebp's
`quality` raw — no bypass needed there.

## LANDED (clean / defensible)

### zenwebp — `metric → native VP8 quality` (commits f18d0a48 + fix 7dc8c200)
600 imgs × 3 knobs × 15 q, clean monotone. The old maps were badly off: `100−12·d` put
d=2.41 at quality 71, measured **95**; ssim2 pass-through over-shot (quality 70 lands near
ssim2 81, not 70).
```
BUTTER_MAX_TO_Q: 2.41→95 2.57→90 2.86→85 2.93→80 3.39→75 3.51→65 3.72→60
                 3.80→55 4.00→45 4.90→35 5.57→25 6.55→15 7.47→10 9.07→5
SSIM2_TO_Q:      53→5 58→10 63→15 69→25 71→30 73→35 77→45 79→55 80→60
                 81→65 82→75 84→80 85→85 87→90 88→95
```

### webpx — same table via libwebp's identity quality dial (commit 5a052796)
libwebp tracks the swept zenwebp encoder to ~0.3 % bytes at matched quality, so the same
table applies; `with_generic_quality` feeds it un-remapped.

## NOT landed (data unusable — current maps kept)

- **mozjpeg-rs** — table built from zenjpeg cells was reverted (commit 45d36a71). Its `q`
  is a zensim-A target, not JPEG quality. Kept the linear `100−12·d`. Needs a raw-quality
  JPEG sweep.
- **zenavif** — sweep `q→metric` non-monotone (q90→ssim2 90.7 but q95→80.4); knob-confounded.
  Kept current map. Needs a clean per-q sweep with no knob mix.
- **zenjxl** — degenerate (every q→ssim2 ≈87.9; `q` not the real dial in that run) AND the
  repo was under an active concurrent session at wiring time. Kept current (native
  butteraugli `with_distance` for ApproxButteraugli; ssim2 pass-through). Note: a *pooled*
  universal `ssim2→butter-max` map exists (95→0.60 90→1.20 85→2.24 80→3.10 75→3.80 70→4.38
  …) but it's a codec-dependent centroid (max-norm↔ssim2 spreads by codec, unlike the
  3-norm) — low confidence, not landed.

## Reproduce
Scripts: `scratchpad/fit_tables3.py` (PAVA-monotonized fit + Rust const emit). Pull omni
parquets from the three R2 prefixes above; aggregate median-per-q; invert.
