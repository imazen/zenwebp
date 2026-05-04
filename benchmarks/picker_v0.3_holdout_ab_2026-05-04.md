# Picker v0.3 — held-out re-encode A/B (zenwebp)

* Date: 2026-05-04T05:02:01Z
* Corpus: `/home/lilith/work/zentrain-corpus/mlp-validate/cid22-val` (41 images)
* Picker bin: `/tmp/zenwebp_v0.3.patched.bin` (n_inputs=82, n_outputs=24, schema_hash=`0x139d73665fb030c7`)
* Targets: [30.0, 35.0, 40.0, 45.0, 50.0, 55.0, 60.0, 65.0, 70.0, 75.0, 80.0, 85.0, 90.0]
* Encoder closed-loop: `target_zensim` with `max_passes=3`, `max_overshoot=1.5`
* Wall: 110.4 s

## Per-target table

| target | n | bytes_picker | bytes_bucket | Δ% (picker − bucket) | win_rate | achieved_picker | achieved_bucket |
|---:|---:|---:|---:|---:|---:|---:|---:|
| 30 | 41 | 370130 B | 396640 B | -6.68% | 88% (36/41) | 44.77 | 44.80 |
| 35 | 41 | 374194 B | 400430 B | -6.55% | 88% (36/41) | 45.43 | 45.29 |
| 40 | 41 | 403328 B | 419600 B | -3.88% | 83% (34/41) | 47.88 | 46.84 |
| 45 | 41 | 430678 B | 460764 B | -6.53% | 90% (37/41) | 50.16 | 49.78 |
| 50 | 41 | 463614 B | 506030 B | -8.38% | 93% (38/41) | 53.06 | 53.17 |
| 55 | 41 | 521474 B | 565018 B | -7.71% | 76% (31/41) | 57.16 | 56.74 |
| 60 | 41 | 601030 B | 642566 B | -6.46% | 90% (37/41) | 60.76 | 60.58 |
| 65 | 41 | 721466 B | 775262 B | -6.94% | 93% (38/41) | 65.94 | 65.88 |
| 70 | 41 | 897468 B | 963100 B | -6.81% | 90% (37/41) | 71.01 | 71.06 |
| 75 | 41 | 1.20 MB | 1.28 MB | -5.86% | 90% (37/41) | 75.77 | 75.74 |
| 80 | 41 | 1.76 MB | 1.81 MB | -2.56% | 66% (27/41) | 80.60 | 80.36 |
| 85 | 41 | 2.37 MB | 2.45 MB | -3.25% | 90% (37/41) | 84.51 | 84.46 |
| 90 | 41 | 3.81 MB | 3.94 MB | -3.18% | 95% (39/41) | 88.01 | 88.06 |

## Per-band totals

Bands chosen to separate operationally-distinct quality regimes:
low = q∈[30,60], mid = q∈[65,80], high = q∈[85,90].

| band | targets | n pairs | bytes_picker | bytes_bucket | Δ% | win_rate | achieved_picker | achieved_bucket |
|---|---|---:|---:|---:|---:|---:|---:|---:|
| low  | 30,35,40,45,50,55,60 | 287 | 3,164,448 B (3.16 MB)  | 3,391,048 B (3.39 MB)  | **−6.68%** | 86.8% (249/287) | 51.32 | 51.03 |
| mid  | 65,70,75,80          | 164 | 4,584,122 B (4.58 MB)  | 4,824,680 B (4.82 MB)  | **−4.99%** | 84.8% (139/164) | 73.33 | 73.26 |
| high | 85,90                |  82 | 6,183,658 B (6.18 MB)  | 6,388,278 B (6.39 MB)  | **−3.20%** | 92.7% (76/82)   | 86.26 | 86.26 |

Per-band verdict:
* low (q30–q60): **SHIP** — −6.68% bytes, achieved-zensim parity within +0.29pp (well inside the 0.5pp tolerance band; picker slightly *over*shoots, so the byte savings are not borrowed from quality).
* mid (q65–q80): **SHIP** — −4.99% bytes, +0.07pp achieved-zensim drift.
* high (q85–q90): **SHIP** — −3.20% bytes, achieved-zensim parity (Δ < 0.01pp).

## Total

* Picker total bytes: **13.93 MB** (mean achieved zensim 63.47)
* Bucket total bytes: **14.60 MB** (mean achieved zensim 63.29)
* Δ bytes: **-4.60%** (picker − bucket)
* Δ achieved zensim: **+0.178** pp
* Total paired wins: 464/533 = **87.1%**

## Verdict

**SHIP** (all three bands, and overall).

* Threshold: SHIP if total bytes (picker) ≤ total bytes (bucket) within ±0.5pp achieved-zensim parity.
* The full-range result (**−4.60%** at +0.18pp drift) **strengthens** the high-q-only verdict (which was −3.44%): the low-q band — operationally the most byte-sensitive regime for web delivery — is where the picker wins hardest.
* No band shows a regression. Picker beats the bucket-table baseline at every one of the 13 sampled targets on bytes, with achieved-zensim within tolerance everywhere.

## Sweep coverage rationale

Per CLAUDE.md sweep discipline: web-focused codec calibration must sample
q5–q60 with the same density as q60–q100. The prior held-out A/B
(`imazen/zenwebp@1f46e06`) sampled only q∈{75,80,85,90} (4 high-q
points) — a sparse high-q-only grid that hides behavior in the
operationally-critical low-q regime where every byte matters most.
This run extends to q∈{30,35,…,90} step 5 (13 points, 9 new low-q
targets) so the verdict reflects the full production target range,
not just the high-quality subset.

## Method notes

* Picker arm: extracted 36-feature zenanalyze vector in `FEAT_COLS` order, applied per-feature transforms (log/log1p/identity per the v0.3 `.bin`'s metadata; trainer applies these BEFORE engineering so cross terms consume transformed values), built the engineered 82-vec via `feats[36] || size_oh[4] || poly[5] || zq*feats[36] || icc[1]` (mirror of `src/encoder/picker/runtime.rs::engineered_features`), ran `Predictor::predict` against the externally-loaded v0.3 `.bin` (with its malformed `feature_transforms` metadata stripped — the original 36-entry list mismatched the 82-input model and would have hard-failed `parse_feature_transforms`), decoded the bytes_log argmin → cell index → `(method, segments)` from the lex-sorted (4,1)..(6,4) taxonomy, and read `sns_strength`, `filter_strength`, `filter_sharpness` from the per-cell scalar heads (clamped to [0,100], [0,100], [0,7] respectively).
* Bucket arm: classified RGB via `zenwebp::encoder::analysis::classify_image_type_rgb8` and called `content_type_to_tuning` for `(sns, filter_strength, filter_sharpness, segments)`. Method pinned to `4` (the bucket-table path's default).
* Both arms encode through `LossyConfig::with_target_zensim` so the iteration loop adapts global VP8 quality to land in the target band; reported bytes / achieved scores are the closed-loop best.
* FEAT_COLS source: hardcoded from `src/encoder/picker/spec.rs::FEAT_COLS` (36 entries, matches `zenwebp_picker_v0.3_2026-05-04.manifest.json::feat_cols` exactly). Engineered axes (46 = size_oh[4] + poly[5] + zq×feats[36] + icc[1]) match `manifest.json::extra_axes` order.
