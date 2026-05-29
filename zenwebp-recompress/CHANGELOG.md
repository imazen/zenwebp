# Changelog

## [Unreleased]

### QUEUED BREAKING CHANGES
<!-- Breaking changes that will ship together before 0.1.0. -->
(none yet)

### Real per-content-class calibration (2026-05-28)
- **`src/calibration/calib_tables.rs`** (AUTO-GENERATED) replaces the
  preliminary photo-only 3-ref fit with **per-content-class** tables
  (photo/screen/line-art/mixed), each 7 source-q anchors × 16 reencode-q
  columns for `RATIO` + `CUM`, plus a linear `SOURCE_CUM`. `data.rs` functions
  (`source_cum`/`best_reencode`/`reencode_cum_at`/`max_reencode_cum`) now take
  `ContentClass`; the router threads `analysis.content_class` through.
- Fit from a disciplined sweep (`benchmarks/calibration_2026-05-28.md`):
  50 refs/class × size variants (Mitchell, downscale-only) × q20–100 step 2
  sources × raw reencode-q 20–95 step 5 = **248,501 cells, 0 errors**, 20%
  held-out per class. Validation cumulative-zensim MAE: screen 3.56, mixed
  3.98, photo 7.89, line-art 8.49 (photo/line-art tail = within-class content
  variance; the size guard + `MaxIterations` budget cover residual error).
- **`zwr-calibrate`** gains `--reencode-qs` (raw non-circular reencode-q
  sweep), `--refine` (opt-in decode-based estimate for estimator validation),
  and records `true_source_q` + `est_quality`. New `build_calib_corpus.sh`
  (multi-class/multi-size corpus) + `fit_calibration.py` (per-class fitter,
  size-dependence report, held-out validation, emits `calib_tables.rs`).
- Estimator finding: the decode-based estimate is biased ~−5 vs true q
  (MAE 6.5); tables are fit on true q (all anchors populated) and est-vs-true
  keying measured equivalent (<0.5 MAE) — interpolation absorbs the bias.

### CoeffEdit — VP8 coefficient transcoder built, edits RD-falsified for WebP (2026-05-28)
- **`src/vp8x/`** — a self-contained, validated VP8 keyframe coefficient
  transcoder: matched boolean coder (`bool.rs`), tables, `parse_vp8_keyframe`
  (→ quantized levels, no IDCT), `emit_vp8_keyframe`, and two edits in
  `edit.rs`: `drop_high_freq_ac` and `requantize` (level-grid coarsening).
  The verbatim transcode and no-op requantize are **bit-for-bit pixel-exact**
  with libwebp's decoder (MAD 0).
- **KEY FINDING:** every size-reducing coefficient edit is RD-dominated by
  `Reencode` at matched output size (AC-drop loses on 4/5 refs; requantize
  loses by 35–52 zensim-A points). Cause: **VP8 spatial intra-prediction
  drift** — editing coefficients breaks the prediction invariant and the
  error compounds across the frame; `Reencode` re-derives residuals from
  clean pixels with RD re-optimisation. Coefficient editing is the right
  tool for prediction-free codecs (baseline JPEG), not VP8/WebP. Full data +
  mechanism in `benchmarks/coeff_edit_experiment_2026-05-28.md`.
- **Drift compensation tried + falsified too.** `src/vp8x/compensate.rs` is a
  closed-loop DC drift compensator (decode source as target; iterate
  emit→decode→measure→nudge DC levels). It's Jacobi-unstable on the prediction
  chain (diverges above relax≈0.25), removes only ~10 % of the drift when
  stable, and never closes the gap to `Reencode` (e.g. ref-1 keep-10:
  uncompensated z=51 → compensated z=13, both ≪ Reencode z=69). Complete,
  stable compensation = sequential full-frequency reconstruction = re-encoding.
- The transcoder + edits + compensator stay reachable as research tools via
  `expert::run_coeff_edit{,_keep,_requant}` and `vp8x::compensate`. **The
  router does not select CoeffEdit** (`router::filter_candidates`); its only
  loss-free point (verbatim) is already `LosslessRemux`.

### Calibration overhaul — decode-based quality estimation (2026-05-28)
- **KEY FINDING:** header-only quality detection (`zenwebp::detect`'s base
  quantizer) is essentially uncorrelated with true encode quality for
  segmented WebP — libwebp's per-segment quantizers leave the base
  quantizer meaningless. trueq=40→detect 1, trueq=90→detect 52. This
  produced the earlier non-monotonic calibration. Documented in
  `docs/QUALITY_DETECTION.md`.
- **`src/estimate.rs`** — decode-based effective-quality estimator via
  recompression self-consistency (re-encode at 4 probe qualities, find the
  size-match crossing, de-bias the 2nd-generation offset). Reliable and
  monotonic where the header is not. `recompress()` uses it as the
  calibration key (it decodes anyway); `plan()` stays header-only/rough.
- **`src/calibration/data.rs` rebuilt** from a clean lossless-only sweep
  keyed on TRUE source quality (the earlier sweep had mixed lossy files in
  as references). Tables are now monotonic and sensible: `source_cum(q)`
  (q50→73, q75→79, q90→86, measured), and `best_reencode(eff_q, target)`
  which scans the target-quality grid and returns the cheapest q meeting
  the target. Real-photo result: a q90 photo recompresses 20% at target 60,
  13% at target 70, and correctly declines at target 75+.
- **Measured budget** (`MaxIterations`/`MaxTime`) replaced the
  generation-loss secant (which optimized the wrong quantity — cumulative
  is unmeasurable at runtime) with `minimize_size`: trust the calibration's
  `chosen_q` for the cumulative target, use the budget to minimize the
  ACTUAL output size (measurable) gated by the model's cumulative.
- `zwr --analyze` now reports the decode-based `estimated_quality`.
- `zwr-calibrate` gains `--source-filter` (used for the deblock
  experiment); the sweep harness keys aggregation on the true `synth_q`
  label, never the unreliable detect estimate.

### Added
- Initial workspace scaffold (zenwebp-recompress library + zwr-calibrate binary).
- Frozen public API: `recompress(...)`, `plan(...)`, `RecompressOptions`,
  `Budget`, `RecompressResult`, `Plan`, `StrategyKind`, `LosslessReason`,
  `NoOpReason`, `Error`.
- DESIGN.md ported from zenjpeg-recompress, adapted for VP8/VP8L taxonomy.
- README with usage example.
- 5 strategy implementations: `LosslessRemux` (real — re-mux via
  `zenwebp::mux::WebPMux`), `Reencode` (real — decode/encode via
  `zenwebp::oneshot::decode_rgba` + `EncodeRequest::lossy`),
  `DeblockReencode` (placeholder → falls through to `Reencode` until the
  content-aware deblock filter lands), `LosslessReencode` (real — decode
  + `EncodeRequest::lossless`), `CoeffEdit` (stub: `debug_assert!`s on
  dispatch — router never picks it yet).
- Hand-fit linear calibration table (`src/calibration/mod.rs`).
  Conservative placeholder until the real corpus sweep lands.
- `score_recompression` / `score_against_reference` / `score_rgba`
  wired to zensim Profile A (`ZensimProfile::A`).
- zwr-calibrate corpus sweep binary; emits per-cell CSV with
  `(size_ratio, measured_zensim_a_vs_source)`.
- `scripts/build_paired_corpus.sh` — paired-reference corpus builder
  (PNG sources + libwebp variants at q ∈ {20, 22, …, 100}).
- `docs/CALIBRATION_NOTES.md` — running log of sweep findings.

### Changed (post-scaffold hardening)
- **Calibration is now empirical**, fit from a 6,400-cell paired sweep
  (`benchmarks/paired_sweep_v2_deblock_2026-05-28.csv`): 10 lossless refs ×
  16 synthetic source qualities × 10 targets × 4 strategies. Keyed on VP8
  `quantizer_index` (encoder-independent), not the derived `source_q`
  estimate. Bound model: `cumulative = min(gen_loss, source_cum)`; bilinear
  interpolation between qi/target bins.
- **`DeblockReencode` — implemented, measured, FALSIFIED, removed from
  router.** The artifact-aware filter (`src/strategies/deblock.rs`,
  H.264/VP8-style boundary-vs-interior gate) is correct and unit-tested,
  but a paired sweep showed it is net-negative on every tested source
  config (default-filtered −2.75 zensim/+3.9% size; weak-filtered −3.35;
  worst-blocking qi≥90 −5.09, 0/60 wins) — VP8 already deblocks in-loop, so
  a second pass moves the image away from the sharp original. The router
  never selects it (with the unimplemented `CoeffEdit`); the filter stays
  as `expert::deblock_rgba`. See
  `benchmarks/deblock_experiment_2026-05-28.md`.
- **`Budget::MaxIterations(N)` runs real secant search** on libwebp_q
  (initial anchor → fixed-direction step → one-pair secant), shipping when
  measured generation-loss lands in `[target − tolerance, target + 2.0]`,
  capped at 8 passes, best-by-distance-to-target tracked. Demonstrated 30%
  shrink (vs 18% one-shot) on a qi-38 source at target 70.
- **Content classification wired into `recompress`**: a heuristic
  Photo/Screen/LineArt/Mixed classifier (`src/classify.rs`, 5-5-5 color
  signature density on a ≤4k-pixel subsample) refines `content_class` from
  the decoded pixels. Gates `LosslessReencode` size projection by content.
  `plan()` stays header-only (conservative `Mixed`).
- **Lossless sources now speculatively re-encode**: a VP8L source is
  re-encoded with our method-4 encoder and the result kept only if it
  actually shrinks (lossless → lossless, zero quality change); otherwise a
  clean re-mux ships. Fixed a projection bug where lossless sources were
  assigned a lossy qi-bin-0 cumulative (~64) instead of ~100.
- `RecompressOptions` gains `tolerance_below_target` (default 1.5): the
  router accepts a slight target undershoot in exchange for real size
  savings when no strategy projects strictly at-or-above target.

### API contract
- **Default features are minimal** (`std` only). The `expert` module is
  opt-in, per the crate's "minimal except when expert is active" contract.
  The `zwr` demo CLI declares `required-features = ["expert"]` (it uses the
  expert `--analyze` path), so building the bin pulls expert without
  forcing it on library consumers. CI exercises both: a default-feature
  build (minimal external-consumer view) and an `--all-features` build
  (expert API + bin + analyzer deps).

### Repo integration
- Ships as a **self-contained nested workspace** at
  `zenwebp/zenwebp-recompress/` (zwr-calibrate moved under it). zenwebp
  itself is intentionally NOT a Cargo workspace — the sibling path deps
  (`../zensim`, `../zenpixels`, `../zenanalyze`) would break zenwebp's core
  CI, which doesn't check them out. Verified `cargo metadata` for zenwebp
  shows only `zenwebp`.
- Isolated `.github/workflows/recompress.yml` (Linux/Windows-arm/macOS-x64)
  builds + tests + clippy(-D warnings) + fmt-checks the nested workspace,
  checking out the three siblings itself. Triggers only on
  `zenwebp-recompress/**` so sibling drift can't redden zenwebp core CI.

### Tests
- 23 passing: 4 locked-API + 8 end-to-end smoke + 9 lib unit (incl. 4
  deblock artifact-decision tests) + 2 quality-grid robustness (source
  quality 20..=100 step 2 × zensim targets, no-growth / valid-WebP /
  no-deselected-strategy invariants).
- Authoritative sweep: `benchmarks/paired_sweep_v2_deblock_2026-05-28.csv`
  (6,400 cells) + `deblock_weakfilter_2026-05-28.csv` (3,200 cells),
  pointer-filed to `/mnt/v` (see `sweeps-2026-05-28.pointer.md`).
