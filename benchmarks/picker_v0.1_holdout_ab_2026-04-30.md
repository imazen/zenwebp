# Picker v0.1 vs bucket-table A/B (held-out, 2026-04-30)

## Result: picker -1.20% bytes vs bucket-table at matched zensim

Held-out CID22 validation set (41 images, NOT in the trainer's training
corpus), four `target_zensim` values × `Preset::Auto`. Same outer
`target_zensim` convergence loop both runs — only difference is whether
the encoder picks `(sns, filter_strength, filter_sharpness, segments)`
via the bucket table or the baked `zenwebp_picker_v0.1.bin`.

| target | bucket bytes | picker bytes | Δbytes | Δ% | bucket ach | picker ach |
|--:|--:|--:|--:|--:|--:|--:|
| 75 | 1,199,472 | 1,194,168 | -5,304 | **-0.44%** | 75.50 | 75.62 |
| 80 | 1,684,446 | 1,669,636 | -14,810 | **-0.88%** | 80.28 | 80.26 |
| 85 | 2,439,218 | 2,442,906 | +3,688 | +0.15% | 84.51 | 84.52 |
| 90 | 3,967,114 | 3,871,884 | -95,230 | **-2.40%** | 88.11 | 87.98 |
| **total** | **9,290,250** | **9,178,594** | **-111,656** | **-1.20%** | — | — |

Per-(image, target) outcomes (n=164):

- picker wins: **85 (51.8%)**
- picker loses: 76 (46.3%)
- ties: 3 (1.8%)

Achieved zensim is essentially identical (within 0.13 pp on average per
target) — picker is not trading quality for bytes.

## Method

Both runs use `Preset::Auto` end-to-end with `target_zensim`
convergence (max passes default). The bucket-table path takes the
classifier's `(ImageContentType, ZenanalyzeDiag)` output and feeds the
hard-coded `content_type_to_tuning(bucket)` lookup; the picker path
calls `super::picker::pick_tuning(rgb, w, h, lossy_quality_proxy,
default_constraints)` and uses the picker's `(sns, filter_strength,
filter_sharpness, segments)` output verbatim. On `Err(_)` from the
picker, the codec falls through to the bucket table — no silent
miscalibration.

## Build invocations

End-to-end from a clean clone, given a Pareto sweep TSV + features
TSV under `benchmarks/`:

```bash
# Train (~30s with the pytorch student; pass --activation leakyrelu —
# default sklearn-relu is 10-20× slower at the same shape)
PYTHONPATH=$HOME/work/zen/zenanalyze/zentrain/examples:$HOME/work/zen/zenanalyze/zentrain/tools \
  python3 $HOME/work/zen/zenanalyze/zentrain/tools/train_hybrid.py \
    --codec-config zenwebp_picker_config \
    --hidden 128,128,128 \
    --activation leakyrelu

# Bake to v2 binary
cargo build --release -p zenpredict --bin zenpredict-bake \
  --manifest-path $HOME/work/zen/zenanalyze/Cargo.toml
python3 $HOME/work/zen/zenanalyze/tools/bake_picker.py \
  --model benchmarks/zenwebp_hybrid_2026-04-30.json \
  --out src/encoder/picker/zenwebp_picker_v0.1.bin \
  --dtype f16 \
  --bake-bin $HOME/work/zen/zenanalyze/target/release/zenpredict-bake

# A/B vs bucket-table baseline
cargo build --release --features "target-zensim analyzer" --example picker_ab_eval
target/release/examples/picker_ab_eval \
    --corpus validation=$HOME/work/codec-corpus/CID22/CID22-512/validation \
    --targets 75,80,85,90 --label bucket --out-tsv /tmp/ab/bucket.tsv

cargo build --release --features "target-zensim picker" --example picker_ab_eval
target/release/examples/picker_ab_eval \
    --corpus validation=$HOME/work/codec-corpus/CID22/CID22-512/validation \
    --targets 75,80,85,90 --label picker --out-tsv /tmp/ab/picker.tsv
```

Bake provenance: `benchmarks/zenwebp_hybrid_2026-04-30_v0.2_pruned36.json`
→ `tools/bake_picker.py --dtype f16 --allow-unsafe`
→ `src/encoder/picker/zenwebp_picker_v0.1.bin` (140KB,
schema_hash=0xb2aca28a2d7a34ec, n_inputs=82, n_outputs=24, n_layers=4).

`--allow-unsafe` was set because the trainer flagged
`DATA_STARVED_SIZE` (intrinsic to perceptual metrics on tiny images at
high zq — see imazen/zenanalyze#51 + #52) and `PER_ZQ_TAIL` at zq=5
(low-q noise). Both are known and addressed by the in-flight
ceiling-aware pipeline; neither affects the held-out comparison
because the codec falls through to the bucket table on any picker
error.

## Trainer provenance

- Pareto sweep TSV: `benchmarks/zenwebp_pareto_2026-04-30_combined.tsv`
  (1264 image-instances × 72 configs × 30 q values = 2.73M rows).
- Features TSV: `benchmarks/zenwebp_pareto_features_2026-04-30_combined.tsv`
  (zenanalyze main, post-#42, 103 columns).
- Schema: 36 features (Δ ≥ +0.05pp from the 2026-04-30 ablation pass);
  6 cells (method × segments) × 4 outputs (bytes_log + sns + filter +
  sharpness) = 24 outputs.
- Capacity: hidden 128x128x128 (47K params, ~92 KB f16; the 192x192x192
  variant overfit at this data scale).
- Trainer student metrics on the in-trainer holdout: 2.42% mean
  overhead, 50.9% argmin acc.

## Caveats

- 41-image held-out is small — the sign and magnitude are clear (-1.2%
  total, picker wins more often than it loses) but a wider held-out
  comparison would tighten confidence intervals. `clic2025/final-test`
  + `gb82-sc` were not used for training and could be added without
  re-running the trainer.
- `lossy_quality` is the picker's `target_zensim` proxy at the call
  site — the encoder's outer convergence loop adjusts `lossy_quality`
  per-pass, so this proxy tracks the iteration but isn't bit-identical
  to feeding the user's `target_zensim` directly. A follow-up should
  plumb the actual target_zensim through.
- Picker is gated behind `cargo --features picker` and stays
  off-by-default. The `analyzer` feature is also required.
