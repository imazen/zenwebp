# Changelog

## [Unreleased]

### QUEUED BREAKING CHANGES
<!-- Breaking changes that will ship together before 0.1.0. -->
(none yet)

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
- **`DeblockReencode` now runs a real filter**: gradient-gated 4-pixel
  edge-preserving blur on decoded RGBA, strength keyed on qi (off below
  qi 60, capped at qi 110+). Real edges preserved; block artifacts blended.
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

### Tests
- 20 passing: 4 locked-API + 8 end-to-end smoke + 8 lib unit.
- Sweep results: `benchmarks/paired_sweep_v2_deblock_2026-05-28.csv`
  (6,400 cells, authoritative), `test_corpus_2026-05-28.csv` (630 cells).
