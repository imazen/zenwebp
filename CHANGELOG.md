# Changelog

All notable changes to zenwebp are documented here. (Started 2026-06-10;
earlier history lives in git log and LOG.md.)

## [Unreleased]

### QUEUED BREAKING CHANGES
<!-- Batch into the next 0.x minor. -->
- `mux::AnimationDecoder`'s fallible methods (`new`, `next_frame`, `decode_next`,
  `decode_all`, `reset`, `set_background_color`, `icc_profile`, `exif_metadata`,
  `xmp_metadata`) and its `Iterator::Item` now carry `whereat::At<DecodeError>`
  instead of bare `DecodeError`, so animation-decode errors keep their `file:line`
  source location (the one-shot decode path already returned `At<DecodeError>` —
  this brings the animation API in line). The trace was previously stripped via
  `From<At<DecodeError>> for DecodeError`. Get the inner error with `e.error()` /
  `e.decompose().0`.

### Changed
- Decoder `Limits::default()` raises `max_total_pixels` from 100 MP to 120 MP so
  common ~108 MP camera photos decode under the default policy.

### Added

- **SCALAR sweep-axis ladders** (`__expert` sweep planner; dense-sweep program
  / `zenpicker-train --scalar-axes`, zenmetrics `docs/PLAN_SWEEPS.md` §5
  gaps): sns_strength mid-ladder {25, 80} (Drawing/Photo preset constants;
  effective {0, 25, 50, 80, 100}), filter_strength mid-ladder {30, 10, 100}
  (effective {0, 10, 30, 60, 100}), and a NEW `filter_sharpness` axis
  {3, 6, 7} via the additive `-shp<v>` id token (effective {0, 3, 6, 7};
  fingerprint-hashed; ladder sheds it first). No-preset defaults stay
  unspelled (Some(default) would byte-alias None). Harness re-run ALL HARD
  CHECKS PASSED, every step live
  (`benchmarks/sweep_validate_webp_2026-06-12.tsv`).

- Re-export the cancellation types from `enough` (`Stop`, `StopReason`,
  `Unstoppable`) at the crate root so callers can reach them as
  `zenwebp::Stop` / `zenwebp::StopReason` / `zenwebp::Unstoppable` — the token
  named by `DecodeRequest::stop` / `EncodeRequest::with_stop` and the error in
  `DecodeError::Cancelled` / `EncodeError::Cancelled` — without adding `enough`
  to their own `Cargo.toml` (#65).

### Fixed

- **docs(readme): document the memory-limits + cancellation APIs (advertised
  but previously un-exampled) and `read_image` output format** — server-safety
  gaps found by an insulated-developer usability test. The feature tables
  advertised "Memory limits" and "Cancellation without thread killing" — the
  exact knobs a server needs for untrusted input — but the README showed no
  callable API for either. Added: a server-grade two-phase decode snippet using
  `WebPDecoder::set_limits`/`DecodeConfig::default().limits(Limits { … })` with
  real fields; a cancellation snippet wiring an `AtomicBool`-backed
  `enough::Stop` token through `DecodeRequest::stop` and
  `EncodeRequest::with_stop`; an explicit statement that `read_image` writes the
  native format (packed RGBA8 when `has_alpha`, else packed RGB8, exactly
  `output_buffer_size()` bytes); and the full list of `PixelLayout` variants
  lossy encode accepts. The two advertised table rows now link to their
  examples.

- **GRAY8/L8 lossless encode hit a literal-only compression cliff** (#57):
  the full VP8L pipeline (LZ77 / palette / meta-huffman / transforms) was
  gated on `is_color`, so `L8`/`La8` main-image lossless encodes fell to the
  weak literal-only encoder while the identical content fed as RGB took the
  full pipeline — honestly-declared grayscale was punished (a 128×96
  gradient+checkerboard measured 2844 B as L8 vs 140 B as gray→RGB, ~20×).
  The gate is now `!implicit_dimensions`; `convert_to_contiguous` already
  widens `L8`→RGB and `La8`→RGBA, so grayscale gets the same treatment as
  color. The literal-only fallback is now reached solely by the alpha plane
  of a lossy+alpha encode (`implicit_dimensions`), where the explicit-
  dimension VP8L header must be omitted. New `gray8_input` regression guards
  pin `L8`/`La8` lossless size at-or-below the gray-expanded equivalent.

- **Lossy one-shot decode ignored the stop token.** `do_decode_lossy` never
  wired the job's stop into the native `DecodeRequest`, the native lossy
  path (`decode_lossy_internal`) had no cancellation check, and `do_decode`
  swallowed any lossy-path error by retrying the general path — so a
  pre-stopped token decoded successfully instead of returning
  `Cancelled`. Now: entry checks in `do_decode` and
  `decode_lossy_internal`, stop wired through the lossy `DecodeRequest`,
  and cancellation from the lossy path propagates instead of falling
  through. Surfaced by zencodecs' `stop_decode_webp` (zenpipe#38 CI work).

### Added

- **Parity differential test for the local EXIF-orientation parser** (#58):
  `src/exif_orientation.rs` stays deliberately local — it runs in the core
  decoder path and delegating to `zencodec::helpers::parse_exif_orientation`
  would make zencodec (with its zenpixels/ICC-table weight) a required
  dependency of every build. Drift protection instead:
  `parity_with_zencodec_helper` pins the local parser against the canonical
  helper across every orientation value × byte order × prefixed/raw ×
  truncation, with zencodec added as a dev-dependency only. The module
  header documents the retention rationale (the old comment pointed at the
  long-closed zencodec#5); `hint_bakes` is annotated to delegate to
  `OrientationHint::bakes()` once zencodec ≥ 0.1.22 ships it.

- `sweep` module (`__expert`): variant-generation playbook adoption
  (patterns 1, 4, 5, 7) — mode-discriminated `SweepVariant`
  (lossy VP8 / lossless VP8L share no knob space structurally), labelled
  internal-params probe registry (def/parity/mpass/smooth/plim50),
  budgeted main-effects-first planner with fingerprint dedup and the
  one-value-at-a-time ladder, and the self-describing cell-id grammar +
  `variant_from_cell_id` parser with grammar-totality roundtrip test.
  Curated-axes exclusions documented in module docs (closed-loop
  targets, preset macro-knob, alpha/exact class-conditional,
  near_lossless metric-class — the lossless space stays 100% trial-class).
- `examples/sweep_validate.rs` (patterns 6/14/15): first run ALL HARD
  CHECKS PASSED across 41 dev≤1 cells × 7 images — every curated step
  live, every cell decode-verified, every lossless cell exact-roundtrip,
  partial-macroblock 509×381 corpus leg. Results committed at
  `benchmarks/sweep_validate_webp_2026-06-11.tsv`; adoption doc at
  `docs/VARIANT_GENERATION.md`.

### Added
- zencodec `OrientationHint` is now honored by the decode adapter. `Preserve`
  (default, unchanged) returns stored-orientation pixels + reports the stored
  (coded) dims and intrinsic EXIF `Orientation` tag; `Correct` /
  `CorrectAndTransform` / `ExactTransform` bake the resolved orientation into the
  decoded pixels (via `zenpixels-convert` `apply_orientation`) and report the
  display dims + `Orientation::Identity`. `probe`/`output_info` report
  consistently with `decode`. Native (non-zencodec) API unchanged; adapter-only,
  mirroring the heic fix. Requires `zenpixels-convert` 0.2.13 (unreleased).
  Tests: `tests/orientation.rs` (d7c54ec4).
- `cms` feature: ICC synthesis for the zencodec color-emit path via
  `zenpixels-convert/icc-db` (a bundled LZ4 profile blob + pure-Rust lz4_flex
  decoder — **no moxcms**), weak passthrough — takes effect with `zencodec`,
  covering the full ITU-T H.273 grid incl PQ/HLG. Failing to synthesize a
  needed (off-grid) ICC is an encode **error**
  (`EncodeError::IccSynthesisUnavailable`), not a silent skip: WebP has no
  CICP carrier, so an embedded ICC is the only way the color survives.
  Requires `zenpixels-convert` 0.2.13 (unreleased — adds the `icc-db`
  feature). CI tests `--features zencodec,cms`; tests
  `cicp_pq_without_cms_is_an_encode_error` / `cicp_pq_with_cms_synthesizes_icc`.
- zencodec 0.1.21 color-emit integration: the encode path reconciles ICC vs
  CICP via `resolve_color_emit` under the caller's `ColorEmitPolicy`. WebP has
  no CICP carrier, so a CICP-only source synthesizes an embedded ICC (via
  zenpixels-convert `synthesize_icc_for_cicp`, transfer-aware) instead of
  silently emitting an untagged sRGB-assumed file. Metadata retention now
  flows through `with_metadata_policy` / `Metadata::filtered`. Deps bumped to
  published zencodec 0.1.21 / zenpixels 0.2.11 / zenpixels-convert 0.2.12
  (8bc51dbe).

### Changed

- Public-API snapshots migrated to the `zenutils-apidoc` 0.1.0 runner package
  at `apidoc/` (self-contained, CI-free): three snapshot files under
  `docs/public-api/`, regenerated via `just api-doc`. Replaces the in-crate
  `tests/public_api_doc.rs` copy, its `serde_json` dev-dep, and every
  `ZEN_API_DOC` / cargo-public-api trace in CI. The nested
  `zenwebp-recompress/` workspace is unaffected.

### Known issues
- dev-dependency `webpx = "0.1.4"` is yanked on crates.io. Builds resolve via
  the committed `Cargo.lock`; a fresh `cargo update` fails until the webpx dep
  is migrated to a current release (0.3.4 at time of writing).
