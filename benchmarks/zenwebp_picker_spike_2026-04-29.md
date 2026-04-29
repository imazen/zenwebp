# zenpicker spike A/B — 2026-04-29

Single-session research spike. Branch: `spike/zenpicker-knobs`.

## Question

Does replacing the 3-row `content_type_to_tuning` lookup table with a
zenpicker-baked MLP improve byte efficiency / convergence at matched
target perceptual quality on the existing 76-image corpus?

## Setup

- **Corpus** — CID22-512/validation (41) + gb82 (25) + gb82-sc (10) = **76 images**.
- **Targets** — `target_zensim ∈ {75, 80, 85}` → 228 decision rows per run.
- **Method** — fixed at 4 (production default).
- **Picker grid** — 16 cells: `sns ∈ {0,25,50,80} × filter ∈ {30,60} × segments ∈ {1,4}`,
  `filter_sharpness=0`, `method=4` fixed.
- **Features** — 14 zenanalyze raw signals + 4 size-onehot + log_pixels + target_zensim_norm
  + 5 polynomial cross terms + 14 (target × feat) cross terms + icc_bytes placeholder = 38 inputs.
- **Model** — sklearn HistGradientBoosting per-cell teacher → MLP `38 → 32 → 32 → 16` student
  trained on soft targets, baked to 6.2 KB f16 v1 binary.

## Distillation validation metrics (held-out images, n=45)

| | argmin acc | overhead mean | overhead p50 | overhead p90 |
|--|--|--|--|--|
| HistGB teacher | 11.1% | +5.4% | +2.7% | +12.6% |
| MLP student | 20.0% | +5.0% | +2.1% | +13.1% |
| **bucket(Drawing) baseline** | n/a | **+2.7%** | **+2.3%** | **+6.1%** |

Critical signal — the bucket table sits at +2.7% vs oracle, the
picker sits at +5.0%. The picker is ~2.3pp **WORSE** than the
bucket-table heuristic on held-out images.

## End-to-end A/B (full 76-image corpus)

Both runs use `Preset::Auto` and the same target_zensim convergence loop.

| metric | bucket | picker | delta |
|--|--|--|--|
| n encodes | 228 | 228 | — |
| avg passes | 1.697 | 1.689 | −0.5% |
| achieved zensim avg | 80.66 | 80.75 | +0.09 |
| target met % | 100% | 100% | — |
| total bytes | 12,799,726 | **13,001,116** | **+1.57%** |
| wall time | 39.5s | 65.6s | +66% |

The +1.57% bytes lift in the wrong direction matches what the offline
held-out metric predicted.

## Recommendation

**NO** — do not productionize. The bucket table is already well-tuned;
the picker spike costs more bytes at matched quality on this corpus.

## Why the spike fell short

1. **Bucket table is already near-optimal at this granularity.** It
   sits +2.7% from the oracle on held-out images. Beating a 3-cell
   hand-tuned heuristic with a 16-cell MLP needs more data than 76 images
   × 3 targets = 228 decision rows.
2. **Grid misses `filter_sharpness`.** The bucket table sends Photo
   content to `sharp=3` and Drawing to `sharp=0`; the picker grid forces
   `sharp=0`. This loses the sharpness lever the bucket table has.
3. **Tiny training set.** 228 decision rows × 16 cells = 3648 sweep
   cells, but with `target_zensim` convergence each cell only emits one
   data point per (image, target). The MLP overfit (loss converged in
   400 iters with no plateau).
4. **`lossy_quality` proxy at the wire-in site.** The picker is fed
   `lossy_quality` (the encoder's per-pass q knob), not the user's
   `target_zensim`. The encoder's outer convergence loop perturbs
   `lossy_quality` per pass, which the picker wasn't trained on.

## What WOULD make a picker worth productionizing

- 1000+ image corpus (web-representative, dense low-q coverage per
  CLAUDE.md sweep discipline).
- Hybrid heads (zenpicker v0.2) so `filter_sharpness` becomes a
  continuous output, not a forbidden axis.
- Target-zensim plumbed to the wire-in site (not `lossy_quality`
  proxy).
- The bake target: ≤ +1% mean overhead vs oracle on held-out images,
  to credibly beat the bucket table's +2.7%.

## Files committed in this spike

- `src/encoder/picker/spec.rs` — ConfigSpec, 16-cell grid, FEAT_COLS
- `src/encoder/picker/runtime.rs` — `pick_tuning(...)` calling zenpicker
- `src/encoder/picker/zenwebp_picker_v0.bin` — 6.2 KB f16 baked model
- `src/encoder/picker/zenwebp_picker_v0.manifest.json` — schema + cell names
- `dev/zenwebp_picker_sweep.rs` — sweep harness (16 cells × 76 images × 3 targets = 3648 encodes)
- `dev/picker_ab_eval.rs` — bucket vs picker A/B harness
- `scripts/zenwebp_picker_distill.py` — HistGB → MLP distill
- `benchmarks/zenwebp_picker_distill_2026-04-29.json` — student weights JSON
- `benchmarks/picker_ab_{bucket,picker}_2026-04-29.tsv` — per-encode A/B output

Sweep raw TSVs (large, kept in `/mnt/v/output/zenwebp/picker-sweep/` per CLAUDE.md):
- `pareto_1777490342.tsv` (3648 rows × 18 cols)
- `features_1777490342.tsv` (76 rows × 19 cols)

## zenpicker / zenanalyze gaps surfaced

- `tools/bake_picker.py::derive_extra_axes` is hard-coded for the
  zenjpeg layout (`size_oh + 5-poly + cross + icc`). Adding a new
  layout requires editing the bake tool. **Recommend** taking
  `extra_axes` from the model JSON when present (still hash-stable
  via SCHEMA_VERSION_TAG override).
- `SCHEMA_VERSION_TAG` is hard-coded to
  `"zenpicker.v1.shared-mlp.distill+icc"`; codec-side tag override
  helps version-skew safety. **Recommend** allowing a CLI flag
  `--schema-version-tag` so per-codec tags don't collide.
- The zenpicker examples/`load_baked_model.rs` has a dead-code
  warning on `AlignedBytes` (alignment is now handled internally by
  `Model::from_bytes` after the round-trip work).
- No way to plumb `target_zensim` from the LossyConfig down to the
  inner encoder's auto-preset resolver. The picker site has to use
  `lossy_quality` as a proxy. Not blocking for this spike, but
  productionization needs the real target as input.
