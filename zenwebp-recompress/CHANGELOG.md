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
- 11 tests: 4 locked-API + 6 end-to-end smoke + 1 inline.
- First sweep results landed at
  `benchmarks/test_corpus_2026-05-28.csv` (630 cells across 21 files).
- `docs/CALIBRATION_NOTES.md` — running log of sweep findings.
