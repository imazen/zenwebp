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
- `WebPDecoder::build` now returns `Result<Self, whereat::At<DecodeError>>`
  (previously `Result<Self, DecodeError>`), so two-phase-decode header-parse
  errors keep their `file:line` trace — matching `WebPDecoder::new`, which
  already returned `At<DecodeError>`. `build` no longer silently strips the trace
  through the removed `From<At<DecodeError>> for DecodeError` dropper. Get the
  inner error with `e.error()` / `e.decompose().0`.
- `EncodeError::LimitExceeded(String)` would become a kind-carrying variant
  (e.g. `LimitExceeded { kind: LimitKind, message: String }`) so its
  `CategorizedError::category()` can report the exact `LimitKind` instead of the
  representative `Memory` it returns today. Deferred — changing the tuple
  variant's shape is a breaking change; its construction sites
  (`src/codec.rs`) already hold the typed `Limits`-check error, so the rewire is
  mechanical once the break is approved.

### Changed
- **deps: migrate to published `zencodec 0.1.24` estimate API; drop the temporary
  git-rev patch.** Removed the `[patch.crates-io]` zencodec git-rev pin (0f71295)
  now that `zencodec 0.1.24` is on crates.io. Updated the
  `estimate_encode_resources` mapping for the refined `ResourceEstimate`:
  `new(peak, wall_ms: u64)` (was `f32`), `with_peak_max(max)` (the `min` arg is
  gone), and dropped the removed `with_output_bytes`.
- Decoder `Limits::default()` raises `max_total_pixels` from 100 MP to 120 MP so
  common ~108 MP camera photos decode under the default policy.
- **docs: README overhaul + split crates.io README.** Refreshed the README
  (focused Quick start, current decode/encode/limits/cancellation API, libwebp
  feature comparison + performance tables wrapped behind `crates.io:skip`),
  re-rendered the crosslink footer, and split a badge-free `README.crates.md`
  for crates.io (`readme = "README.crates.md"`; the full badge row stays on the
  GitHub `README.md`). Added `benchmarks/README.md` documenting the
  fair-benchmark methodology and exact repro. No code or public-API change.

### Added

- **Adopt the `zencodec` `CategorizedError` taxonomy (PR #103).** The public
  encode/decode error types — `DecodeError`, `EncodeError`, `mux::MuxError`,
  `ValidationError`, and `detect::ProbeError` — now `impl
  zencodec::CategorizedError` (gated on the `zencodec` feature) with
  `codec_name() = Some("zenwebp")` and a `category()` mapping every variant to one
  coarse `ErrorCategory`, so consumers route on the category (HTTP status,
  retry policy, logging) without naming the concrete enum. Bitstream errors map
  to `MalformedImage`; `NotEnoughInitData`/`NoMoreFrames`→`UnexpectedEof`;
  `UnsupportedFeature`→`UnsupportedImageFeature`;
  `IccSynthesisUnavailable`→`CmsRequired`; `InvalidBufferSize`→`InvalidBuffer`;
  validation/dimension/layout errors→`InvalidParameters`. Limit variants map to
  the closest `LimitKind` (`ImageTooLarge`→`Pixels`,
  `MemoryLimitExceeded`→`Memory`, `Partition0Overflow`→`OutputSize`); the
  `Cancelled`/`UnsupportedOperation` arms delegate to the zencodec cause types
  (`StopReason`/`UnsupportedOperation`), preserving timeout-vs-cancel; `MuxError`
  delegates its wrapped `EncodeError`/`DecodeError`. The `At<E>` blanket impl
  forwards both category and codec name. Additive (opt-in trait on
  `#[non_exhaustive]` enums; no public-API break). Behind a **temporary
  `[patch.crates-io]` pin** to the unreleased `cancellation-classification-99`
  branch — remove the patch and bump the `zencodec` dependency once
  `zencodec 0.1.26` ships. `EncodeError::LimitExceeded(String)` is stringly and
  can't carry the exact `LimitKind` (its construction sites collapse the
  dimensions/memory/output-size limit checks into a message), so it maps to a
  representative `Memory` kind (see QUEUED BREAKING CHANGES).

- **Honor `zencodec::AllocPreference` (3-mode, per-site) at untrusted decode
  allocations** (zencodec feature): the big buffers sized from decoded header
  dimensions — the final RGB/RGBA output, the VP8 row cache, the streaming
  strip buffer, the lossless ARGB plane / RGBA-expansion scratch, and the
  animation canvas — now route through a per-site fallibility policy
  (`src/decoder/alloc_util.rs`). The zencodec adapter sets it from
  `ResourceLimits::prefer_fallible_allocations` at the decode boundary:
  `CodecDefault` keeps each site's default (big header-sized sites fallible
  `try_reserve` → graceful `MemoryLimitExceeded`; bounded per-row/scratch stay
  infallible `vec!`); `Fallible`/`Infallible` force one path everywhere. The
  native decode API is unchanged (always `CodecDefault`). Added `checked_mul`
  guards on the streaming strip-buffer size. No public API change.

- **`estimate_decode_resources` override on `WebpDecoderConfig`** (zencodec
  feature): implements zencodec's unified
  `DecoderConfig::estimate_decode_resources` via the existing
  `heuristics::estimate_decode` model — peak = output buffer + VP8 working set
  (decoder state + ~12 B/px for the row cache, reconstruction buffer and
  loop-filter accumulator), reported as `ThreadingInformation::SERIAL`
  (zenwebp decode is single-threaded). Returns a
  `zencodec::estimate::ResourceEstimate`.

- **`estimate_encode_resources` override on `WebpEncoderConfig`** (zencodec
  feature): implements zencodec's unified `EncoderConfig::estimate_encode_resources`
  via the existing `heuristics::estimate_encode` model, returning a
  `zencodec::estimate::ResourceEstimate` (typical/max peak memory, wall time).
  WebP encode is single-threaded, so threading is reported as
  `ThreadingInformation::SERIAL`. Requires published zencodec `0.1.24`.

- **`scalar_dense` + compute-budget sweep controls** (`__expert` sweep
  planner; VARIANT_GENERATION patterns 17–18 — trained-scalar-head &
  compute-budget): `SweepAxes::scalar_dense()` (plus per-mode
  `LossyAxes`/`LosslessAxes::scalar_dense()`) pins every categorical axis to
  its production default and gives each CONTINUOUS knob a dense isolated
  ladder — the full `method` 0–6 ladder (filling the m0/m1/m3/m5 that
  `modes_full` skips), and 0,10,…,100 `sns_strength`/`filter_strength` plus
  0–7 `filter_sharpness` (no-preset defaults left unspelled to avoid
  byte-aliasing `None`). `compute_tier(&SweepVariant) -> u8` exposes the
  per-encode CPU ordinal (the `method` level, for both VP8 and VP8L).
  `SweepBuilder::with_compute_limit(max_tier)` drops cells above the tier
  (reported in the new `SweepPlan::compute_tier_skipped`, never silently),
  and `with_max_deviations(max)` keeps main-effects-only cells (1 = one
  isolated ladder per knob). All additive, behind `__expert`; 3 new tests.

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

- **Restore the `wasm32`/`i686`, `no_std`-test, and default-feature clippy
  builds** (pre-existing on `main`, unrelated to the taxonomy work; CI red since
  2026-06-23, run `28284349117`). Three independent breakages, fixed as one
  clearly-labeled fix-forward commit so this PR's CI can be green: (1)
  `predictor_avg_body_wide` in `src/decoder/lossless_transform_simd.rs` was
  missing the `#[cfg(target_arch = "x86_64")]` gate its sibling
  `predictor_add_body_wide` carries, so its x86_64-only `chunk32`/`chunk32_ref`
  helpers fell out of scope on `wasm32`/`i686` (`E0425 cannot find function
  chunk32_ref`); (2) the `src/decoder/extended.rs` `#[cfg(test)]` module used the
  `vec!` macro without `use alloc::vec;`, breaking
  `cargo test --no-default-features --lib`; (3) `AllocPreference::{Fallible,
  Infallible}` (`src/decoder/alloc_util.rs`) are only constructed via the
  `zencodec` adapter, so the default-feature clippy build flagged them
  `dead_code` under `-D warnings` — now `allow`ed only when `zencodec` is off.

- **Fuzz timeout in the `decode_v2` target on a decompression-bomb input**
  (closes #68) — a 104-byte WebP declaring a 12801×4097 (52 MP / 210 MB) VP8L
  canvas decoded correctly but, under libFuzzer's sanitizer-coverage
  instrumentation (~40× slower than release), the fully-bounded linear decode
  (~0.5s on bare metal) crossed the 25s libFuzzer timeout (measured 21.6s). The
  decoder is correct and bounded — the library's production `Limits::default()`
  (120 MP / 1 GB) is intentionally unchanged, and a 52 MP image still decodes —
  so the fix is in the harness, not the library: `decode_v2` now routes all three
  decode paths (`decode_rgba`, `decode_rgb`, animation) through a `DecodeRequest`
  /`AnimationDecoder::new_with_config` carrying restrictive fuzzing limits
  (4 MP / 64 MB, mirroring zengif's `fuzz_decode` convention), so oversized
  canvases are rejected at the dimension/memory check in ~26 ms under
  instrumentation instead of looping over millions of pixels. Regression seed
  `fuzz/regression/crash-…-issue68-bomb-52mp` + `bomb_seed_rejected_quickly`
  test gate added.

- **whereat error traces were silently dropped across zenwebp's internal
  decode/encode/mux boundaries** (0ea292fc) — five `From<At<X>> for Y` impls (in
  `decoder/api.rs`, `encoder/api.rs`, and `mux/error.rs`) plus several explicit
  `.decompose().0` calls discarded the `file:line` trace on every error that
  crossed an internal type boundary, so a decode/encode failure surfaced with an
  empty trace (`frame_count() == 0`). Removed all five droppers and converted the
  internal propagation functions to return `At<_>` so dependency traces propagate
  through a plain `?`: decode side (`WebpDecoder::do_decode` / `do_decode_lossy` /
  `do_decode_general` and the public `WebPDecoder::build`), encode side
  (`WebPEncoder::encode` / `encode_with_diagnostics`, `do_encode`,
  vp8l `encode_argb` / `encode_argb_single_config`, vp8 `encode_image` and the
  quality/psnr/partition-retry search helpers), the `target-zensim` metrics chain
  (`iteration::run`, `encode_at_with_diagnostics`, `measure_*`,
  `encode_single_pass`, `encode_pixels_with_metrics`), and mux
  (`encode_frame_data` / `push_encoded_frame` now use `.map_err_at(...)` at the
  `EncodeError`/`MuxError` boundary). Cross-type boundaries preserve the trace via
  `whereat::ResultAtExt::map_err_at`; variant matches inspect the inner error via
  `At::error()` instead of consuming the `At`. Regression test
  `tests/whereat_trace_propagation.rs` asserts a dependency-originated decode
  error keeps `frame_count() >= 1` through the converted path. See the
  `WebPDecoder::build` note under QUEUED BREAKING CHANGES for the one
  externally-visible signature change.

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
