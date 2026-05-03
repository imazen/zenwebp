# Picker v0.3 — held-out re-encode A/B (zenwebp)

* Date: 2026-05-03T23:56:17Z
* Corpus: `/home/lilith/work/zentrain-corpus/mlp-validate/cid22-val` (41 images)
* Picker bin: `/tmp/zenwebp_v0.3.patched.bin` (n_inputs=82, n_outputs=24, schema_hash=`0x139d73665fb030c7`)
* Targets: [75.0, 80.0, 85.0, 90.0]
* Encoder closed-loop: `target_zensim` with `max_passes=3`, `max_overshoot=1.5`
* Wall: 32.7 s

## Per-target table

| target | n | bytes_picker | bytes_bucket | Δ% (picker − bucket) | win_rate | achieved_picker | achieved_bucket |
|---:|---:|---:|---:|---:|---:|---:|---:|
| 75 | 41 | 1.20 MB | 1.28 MB | -5.86% | 90% (37/41) | 75.77 | 75.74 |
| 80 | 41 | 1.76 MB | 1.81 MB | -2.56% | 66% (27/41) | 80.60 | 80.36 |
| 85 | 41 | 2.37 MB | 2.45 MB | -3.25% | 90% (37/41) | 84.51 | 84.46 |
| 90 | 41 | 3.81 MB | 3.94 MB | -3.18% | 95% (39/41) | 88.01 | 88.06 |

## Total

* Picker total bytes: **9.15 MB** (mean achieved zensim 82.22)
* Bucket total bytes: **9.47 MB** (mean achieved zensim 82.15)
* Δ bytes: **-3.44%** (picker − bucket)
* Δ achieved zensim: **+0.068** pp

## Verdict

**SHIP**

* Threshold: SHIP if total bytes (picker) ≤ total bytes (bucket) within ±0.5pp achieved-zensim parity.

## Method notes

* Picker arm: extracted 36-feature zenanalyze vector in `FEAT_COLS` order, applied per-feature transforms (log/log1p/identity per the v0.3 `.bin`'s metadata; trainer applies these BEFORE engineering so cross terms consume transformed values), built the engineered 82-vec via `feats[36] || size_oh[4] || poly[5] || zq*feats[36] || icc[1]` (mirror of `src/encoder/picker/runtime.rs::engineered_features`), ran `Predictor::predict` against the externally-loaded v0.3 `.bin` (with its malformed `feature_transforms` metadata stripped — the original 36-entry list mismatched the 82-input model and would have hard-failed `parse_feature_transforms`), decoded the bytes_log argmin → cell index → `(method, segments)` from the lex-sorted (4,1)..(6,4) taxonomy, and read `sns_strength`, `filter_strength`, `filter_sharpness` from the per-cell scalar heads (clamped to [0,100], [0,100], [0,7] respectively).
* Bucket arm: classified RGB via `zenwebp::encoder::analysis::classify_image_type_rgb8` and called `content_type_to_tuning` for `(sns, filter_strength, filter_sharpness, segments)`. Method pinned to `4` (the bucket-table path's default).
* Both arms encode through `LossyConfig::with_target_zensim` so the iteration loop adapts global VP8 quality to land in the target band; reported bytes / achieved scores are the closed-loop best.
* FEAT_COLS source: hardcoded from `src/encoder/picker/spec.rs::FEAT_COLS` (36 entries, matches `zenwebp_picker_v0.3_2026-05-04.manifest.json::feat_cols` exactly). Engineered axes (46 = size_oh[4] + poly[5] + zq×feats[36] + icc[1]) match `manifest.json::extra_axes` order.
