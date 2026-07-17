# Changelog

All notable changes to zenwebp are documented here. (Started 2026-06-10;
earlier history lives in git log and LOG.md.)

## [Unreleased]

**Next release from main is 0.5.0** — the picker removal (5d6df59) and the
`At<zencodec::CodecError>` Pattern-B migration (#69) below are already on
main; the manifest is pre-bumped so a publish from head cannot ship them
under a reused/lower version. **0.4.5 was published 2026-05-02 (`6e337c2`)
and then yanked** (bundled a `zenpredict`-baked model via `include_bytes!`
for the picker feature — see `docs/RECOVERY_REGISTER_2026-05-08.md`); the
last currently-installable version is **0.4.4**. crates.io/cargo permanently
reserve a version number once published even if yanked, so 0.4.5 cannot be
reused — main has been accumulating unreleased changes on top of the yanked
0.4.5 point ever since, including the picker's full removal (this yank's
root cause) and multiple breaking changes. 0.5.0 covers all of it, matching
the re-release plan recorded in `docs/RECOVERY_REGISTER_2026-05-08.md`
("Tag as 0.5.0 (minor bump for API; 0.4.5 yanked)").

### QUEUED BREAKING CHANGES
<!-- Not yet landed. Batch into a future 0.x minor once approved. -->
(none currently — the `EncodeError::LimitExceeded` kind-carrying item queued
here has landed; see "Changed (BREAKING)" below.)

### Removed (BREAKING)
- **Removed the feature-gated v0.1 zenwebp picker** (5d6df59): the `picker`
  cargo feature and the `pub mod encoder::picker` surface
  (`PickError`/`TuningPick`/`pick_tuning`/`pick_tuning_from_features`/
  `encoder::picker::spec::{CellSpec, PickerConstraints, CELLS, FEAT_COLS,
  N_CELLS, N_FEAT_COLS, N_OUTPUTS, SCHEMA_HASH, SCHEMA_VERSION_TAG,
  cell_to_method_segments, ...}`). This was an off-by-default research spike —
  a zenpredict-baked MLP (ZNPR v2, `zenwebp_picker_v0.1.bin`) trained pre-A on
  the 2026-04-30 sweep that overrode the `Preset::Auto` (sns, filter,
  sharpness, segments) tuple. The A-era picker re-train refused zenwebp at the
  feature ceiling, so there is no A replacement; production encodes already
  fell back to the heuristic bucket table (`analysis::content_type_to_tuning`)
  on any picker error, and default builds never enabled the feature. Also
  dropped the now-unused `zenpredict` optional dependency and the 3 picker
  research dev examples (`zenwebp_picker_sweep`/`picker_ab_eval`/
  `picker_v0_3_holdout_ab`) plus their baked models. `analyzer`/`__expert` are
  unaffected. `docs/public-api/` snapshots regenerated to drop the removed
  symbols.

### Changed (BREAKING)
- **zencodec trait impls now return `At<zencodec::CodecError>` (the envelope,
  Pattern B) instead of the native `At<EncodeError>` / `At<DecodeError>`** (#69).
  Every `zencodec::{encode,decode}` trait `type Error` on the WebP adapters
  (`WebpEncoderConfig`/`WebpEncodeJob`/`WebpEncoder`/`WebpAnimationFrameEncoder`,
  `WebpDecoderConfig`/`WebpDecodeJob`/`WebpDecoder`/`WebpStreamingDecoder`/
  `WebpAnimationFrameDecoder`) plus the inherent `WebpDecoderConfig::probe_header`
  / `decode` convenience methods now surface `At<zencodec::CodecError>`. This lets
  a generic consumer recover the `ErrorCategory` **and** the codec name
  (`Some("zenwebp")`) **through `Dyn*` dispatch** — once a result is erased to a
  `BoxedError` the native error is no longer downcastable, so Pattern A lost both.
  The five native error types (`DecodeError`, `EncodeError`, `MuxError`,
  `ValidationError`, `ProbeError`) keep their `CategorizedError` impls and remain
  the envelope's *detail* + category source (reachable via
  `CodecErrorExt`/`find_cause::<DecodeError>()`); the native `EncodeRequest` /
  `DecodeRequest` / `oneshot` APIs are unchanged. Corrects a prior Pattern A
  surfacing (the envelope was always the intent).
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
- **`WebpEncoderConfig::with_sharp_yuv` is gated behind the `__expert` cargo
  feature** (ebb30bf, 2026-05-02 — part of the `expert`→`__expert` rename
  bundled into the never-published 0.4.5 manifest bump, `6e337c2`). It was
  unconditionally public in the last actually-published release (0.4.4).
  Confirmed via `cargo semver-checks check-release` against a locally-built
  0.4.4 rustdoc-JSON baseline (`inherent_method_missing`: `with_sharp_yuv` no
  longer reachable under the crate's default+all-additive-feature set — the
  one real failure out of 196 checks). Enable `--features __expert` to keep
  calling it directly, or use `LossyConfig`/`InternalParams`.
- **`EncodeError::LimitExceeded` now carries the real `zencodec::LimitKind`**:
  `LimitExceeded(String)` → `LimitExceeded(zencodec::LimitKind, String)`. Its
  `CategorizedError::category()` reports the actual kind exceeded
  (`Width`/`Height`/`Pixels` from `check_dimensions`, `Memory` from
  `check_memory`, `OutputSize` from `check_output_size`) instead of the
  hardcoded `Memory` stand-in it returned before — the decode side
  (`ImageTooLarge`→`Pixels`, `MemoryLimitExceeded`→`Memory`) already had this
  precision; the encode side now matches. The four construction sites
  (`src/codec.rs`) already held the typed `zencodec::LimitExceeded` cause, so
  the rewire reads `e.kind()` at each call site. Resolves the item that was
  queued in `QUEUED BREAKING CHANGES` since the PR #103 taxonomy adoption.

### Changed
- **VP8 profile bit matches libwebp** (#38): frame-tag version is now 2
  when the loop filter is disabled (was always 0), matching
  `webp_enc.c`'s profile deduction. Decode-identical.
- **Segment quantizer/filter values written as absolute** (#38): the
  segment header now uses `segment_feature_mode=1` with absolute
  per-segment values like libwebp ("we always use absolute values"),
  instead of deltas. Decode-identical; ~2 bytes/file larger, and the
  header field values now compare equal against libwebp's.
- **`zencodec` is now a required, always-on dependency — the codec-trait
  integration is no longer behind the optional `zencodec` cargo feature**
  (#69). `zencodec` is `#![no_std] + alloc`, so the `EncoderConfig` /
  `DecoderConfig` adapters, the `StreamingDecode` / animation-frame jobs, the
  color-emit (ICC-synthesis) path, and the `CategorizedError` impls on
  `DecodeError` / `EncodeError` / `mux::MuxError` / `ValidationError` /
  `detect::ProbeError` now build unconditionally. Removed the `zencodec` cargo
  feature; `self_cell` and `zenpixels-convert` (kept `default-features = false`)
  became unconditional dependencies. `cms` now enables
  `zenpixels-convert/icc-db` directly (was the weak `zenpixels-convert?/icc-db`
  passthrough). no_std and wasm32 builds are unaffected — the integration is
  no_std-clean; the only `std`-genuine arm (`DecodeError::IoError`) stays gated
  on `feature = "std"`. Also dropped the now-redundant `zencodec` dev-dependency
  (the EXIF-parity unit test reaches it through the regular dep).
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

### Changed

- **Tuned default: libwebp-exact chroma conversion at m1+** (#38 RD
  round 3, `6ef3df9b`): the gamma-corrected byte-rounded chroma measured
  as a matched-quality loss on real content at every method (m2 −0.30%,
  m4 −0.24% bytes at equal zensim; 15-image corpus). The tuned default
  now uses libwebp's exact YUV_FIX+2 chroma at method ≥ 1; m0 keeps the
  byte-rounded path (neutral there, and it holds the synthetic-gradient
  low-q floor). m0 output is bit-unchanged. With this, the tuned default
  sits at m0 0.908 / m2 1.005 / m4 0.993 / m6 0.984 vs libwebp at
  matched zensim. Also measured and rejected: truncate-exact segment
  quant (null) and an m2 UV-RD pick (null at +25% time). Analysis:
  `benchmarks/rd_pareto_review_2026-07-16.md` +
  `benchmarks/rd_round3_2026-07-16.md`.

- **Encoder speed session 2: m4 instruction AND wall parity with libwebp,
  m2 beats it — zero byte changes** (#38 speed, 2026-07-16, chunks
  `dd6236ec`/`dd352c55`/`7c33e0ae`/`f62983ab`/`479681c7`): the i16
  coefficient port is complete — DCT block carriers, the fused
  quantize+dequantize family on every SIMD tier (input packs + output
  sign-extend unpacks deleted), i16 IDCT/ftransform/add-residue encoder
  variants, and `trellis_quantize_block` i16 in/out, all matching
  libwebp's `int16_t` pipeline; the analysis TM/DC predictions lost
  hundreds of per-pixel bounds checks (fixed-window pattern); the I16 RD
  winner's reconstruction + levels now carry into the final pass at
  m3/m4 (completing the I4/UV/I16 winner-carry set); the segment
  quantization matrices are plain fields (11 hot-loop Option unwraps
  gone); and ~700 lines of unused legacy record paths were deleted.
  Result (512² q75 vs libwebp): m4 instructions 183.5M → **171.2M vs
  libwebp's 170.7M = 1.003x**, wall **1.015x diagnostic / ~1.16x
  default** (was 1.25x); m2 default **0.97x — faster than libwebp**;
  m6 392→374M (the residue is trellis-DP codegen density, measured and
  documented). Byte gate `958f376a6c8b118f` unchanged across every
  slice; suites green on x86-64 + i686 + wasm32.
  `benchmarks/speed_parity_2026-07-16.md` has the full breakdown.

- **Encoder speed: m2 at/above libwebp, m4 −14% instructions — zero
  byte changes** (#38 speed, 2026-07-16, six chunks `ec577bba` through
  `fdb928ab`): the final coding pass now uses the fused SIMD primitives
  (was per-coefficient scalar); reconstruction levels feed the token
  recorder directly (the raw DCT input was being re-quantized a second
  time); bordered prediction workspaces are shared across mode loops;
  the residual-cost walkers run one arcane region per MB instead of one
  dispatch per block; the trellis DP lost its Option-branches and
  per-position table indexing; and the I4 + UV RD winners' full
  reconstructions and levels carry into the final pass instead of being
  recomputed (I4: non-trellis SIMD paths; UV: diffusion-contract-gated,
  the final pass replays only the error store). Wall ratios vs libwebp
  (512² q75): m0 2.41→~1.9x, m2 1.24→**1.00x** (diagnostic **0.88x —
  faster than libwebp**), m4 1.47→**~1.26x** (instructions 1.075x),
  m6 1.57→~1.55x. Every slice verified byte-identical over the
  `dev/output_hash.rs` grid (both cost models, alpha, sharp, lossless)
  plus full test suites on x86-64 AND i686 (the 32-bit lane caught two
  real breaks in arch-gated arms). Analysis + next levers (i16
  coefficient migration for m6's trellis and the I1 footprint; I16
  winner carry): `benchmarks/speed_parity_2026-07-16.md`.

- **Lossy `exact` is now implemented: transparent areas are cleaned up by
  default** (#38, 2026-07-16): the long-documented `EncoderConfig::exact`
  flag (default `false`, matching libwebp) previously did nothing on the
  lossy path — RGB under fully-transparent pixels was encoded verbatim.
  The port of libwebp's `WebPCleanupTransparentArea` (YUV flavor) now
  runs after conversion for RGBA/BGRA/La8 input: mixed-alpha 8×8 blocks
  get their invisible lumas averaged from the visible ones, fully
  transparent blocks flatten to the run's first pixel — the checker-alpha
  probe's VP8 layer shrank 1554 → 1058 B with identical visible pixels.
  `exact(true)` preserves RGB under alpha=0 exactly as before.
- **VP8L Huffman ties now break exactly like libwebp** (#38, 2026-07-16):
  the length-limited Huffman builder is now an exact port of libwebp's
  `GenerateOptimalTree` sorted-array merge; the previous BinaryHeap left
  the order of equal-count internal nodes to heap sift order. Same-cost
  trees (identical length multisets), so sizes move only via tree-storage
  RLE by ±bytes; A/B corpus lossless ratio vs libwebp: 1.0011× (m4) /
  0.9999× (m6). Also: `red_and_blue_always_zero` (computed since the
  entropy-analysis port but never consumed) now skips the cross-color
  transform exactly like libwebp — images whose R/B channels are
  constant-zero for the chosen mode (alpha planes, pure-gray content in
  subtract-green modes) no longer spend a transform image on it.

- **`.sharp_yuv(true)` now runs the libwebp SharpYUV port** (#38,
  2026-07-16): measured +1.0..+1.8 zensim over standard conversion on the
  15-image A/B corpus (zenyuv's Newton converter: +0.18..+0.32) at the
  technique-inherent +2-5% bytes; the scalar port also benches 1.5× faster
  than libwebp's SSE2 build (44 vs 28 Mpix/s). Opt-in path only — default
  encodes (sharp off) are byte-unchanged, and `sharp_yuv_config(custom)`
  still selects zenyuv's converter. Below 4 px either dimension, sharp
  falls back to standard conversion (libwebp's
  `kMinDimensionIterativeConversion`). Analysis + data:
  `benchmarks/sharpyuv_port_2026-07-16.md`.

### Added

- **Alpha (RGBA) byte-parity axis complete: 192/192** (#38, 2026-07-16):
  whole lossy+alpha files under `StrictLibwebpParity` are byte-identical
  to libwebp across the sweep's alpha grid (opaque/gradient/checker ×
  64×64/33×17 × q5-90 × m0-6 × aq100/90) — the byteparity sweep is now
  100% on EVERY axis (base 4004/4004 + six permutation axes). Landed
  across this and the entries above: the Huffman tie-break port, the
  `red_and_blue_always_zero` cross-color skip, transparent-area cleanup
  (+ post-cleanup MB re-pad), alpha-WEIGHTED chroma averaging for
  mixed-alpha 2×2 blocks (`WebPAccumulateRGBA`, parity precision), and
  libwebp's hash-chain iteration accounting (heuristics consume
  iterations, pre-decrement walk, no stall budget) behind
  `Vp8lConfig::parity` — the tuned lossless keeps zenwebp's stall-budget
  variant (measured better at m0 on smooth gradients). New CI anchor
  `transparent_rgba_matches_libwebp`; diagnostics: `img=`-tagged TREE
  dumps, `ZBIT/LBIT STREAM` markers, `REFDBG` backward-ref dumps,
  `ALPHADUMP`, `corpus_test` lossless mode.

- **SharpYUV port — `sharp_yuv` byte-exact under `StrictLibwebpParity`**
  (#38, 2026-07-16): `src/encoder/sharpyuv.rs` is an exact Rust port of
  libwebp's SharpYUV converter (`sharpyuv/*.c`, 8-bit path): 10-bit
  fixed-point W/(R−W,G−W,B−W) iteration with linear-light targets (baked
  16-bit sRGB↔linear tables dumped from libwebp, immune to libm `pow`
  drift), the decoder-matched 9-3-3-1 upsampling filter, ≤4 gradient
  passes with the padded-dims `3·w·h` exit threshold, and the
  `kSharpYuvMatrixWebp` 16.16 final conversion. Verified byte-identical to
  `SharpYuvConvert` on 17 shapes (1×1…1023×7) and to the full
  webpx-flow encoder planes on real images; the byteparity sweep's
  sharp_yuv axis went 0/96 → **96/96 byte-identical** with the base grid
  unchanged at 4004/4004. Parity encodes route RGB/BGR input through the
  port when `sharp_yuv` is on (falling back to standard conversion below
  libwebp's 4-pixel `kMinDimensionIterativeConversion`, like libwebp);
  the tuned default keeps zenyuv's sharp converter. New anchor test
  `sharp_yuv_matches_libwebp`; diagnostics: `dev/sharpyuv_selftest.rs`,
  `__expert::sharpyuv_convert_rgb`, `bitexact_diff` arg 9, `ZYUVDUMP`
  plane dump (mode_debug).

- **Alpha-plane pipeline + full-VP8L payloads; tuned ALPH 2-3.5× smaller**
  (#38, 2026-07-16): `src/encoder/alpha.rs` ports libwebp's whole
  `EncodeAlpha` pipeline (QuantizeLevels k-means, h/v/gradient filters,
  `WebPEstimateBestFilter`, `GetFilterMap` trial loop, raw fallback, header
  byte). Root of the old 6-10× payload gap: zen's ALPH payloads used a
  literal-only VP8L fallback — never the full pipeline. New
  `Vp8lConfig::omit_headers` provides the ALPH stream framing (libwebp
  writes signature/dimensions/version only in `VP8LEncodeImage`); parity
  AND the tuned default now route alpha through the full VP8L coder
  (palette/predictors/LZ77/meta-huffman). Tuned ALPH payloads shrink
  2-3.5× (54→26, 109→32 B probes) with bit-identical decoded pixels
  (tuned keeps its level-quantizer mapping). Parity payloads land within
  0-3 bytes of libwebp with all pipeline decisions matching (`VP8LDBG`/
  `ZVP8LDBG` dumps agree on palette/histo-bits/entropy-mode/sorting);
  remaining bit-level divergence in the entropy-coded region is the next
  chunk. `dev/alphadiff.rs` (new) isolates the divergence layer.
- **Sharp-YUV dig: libwebp's SharpYUV substantially outperforms zenyuv's
  sharp** (2026-07-16): measured on the 15-image corpus
  (`dev/sharpyuv_compare.rs`, new), libwebp's SharpYUV delivers +0.95 to
  +1.78 zsim over standard conversion where zenyuv's sharp delivers only
  +0.23 to +0.32 — a genuine ~+0.6-1.0 zsim algorithmic advantage at
  mid-high quality even after discounting its 2-5% byte spend. zenyuv's
  redesign claim ("2 Newton iterations beat 4 gradient iterations") does
  not hold up end-to-end. Queued: exact SharpYUV port inside zenwebp
  (closes the sharp_yuv parity axis AND becomes a quality-mode candidate).
  Full analysis: `benchmarks/sharpyuv_dig_2026-07-16.md` + TSV.
- **`StrictLibwebpParity` validated across every shared setting axis; six
  more roots closed** (#38, 2026-07-16): `dev/byteparity_sweep.rs` gained a
  phase-2 permutation sweep (filter_sharpness 1-7, segments 3 + sns/filter
  extremes incl. sns>0/segs1, quality edges q0/q1/q99/q100, pinned
  partition_limit, sharp_yuv, RGBA alpha) — all knob axes now 100%:
  sharpness 1960/1960, extremes 1120/1120, q-edges 1456/1456,
  partition_limit 252/252. Roots: libwebp's `SetupFilterStrength`
  read-before-assign (strength derived with the PREVIOUS call's sharpness);
  UV-DC quant index unclipped at 117 in the single-segment init (**both
  models** — a real encoder/decoder mismatch at q0); error diffusion active
  above libwebp's q≤98 gate; `StoreMaxDelta` gating on raw instead of
  flat-latch-doubled D; a zenwebp-only pl²-scaled I4 seed plus plim100 path
  reroutes leaking into parity; missing uv_alpha-derived UV quant deltas on
  the single-segment path. **Opaque RGBA now encodes as a bare `VP8 ` file**
  (**both models**, like libwebp's `WebPPictureHasTransparency`) — drops a
  wasted VP8X+ALPH wrapper (~80 B on a 64×64). Follow-on documented:
  non-opaque alpha (needs libwebp's alpha-plane coder port; 128 sweep cells)
  and sharp_yuv (zenyuv's own algorithm, out of parity scope by design). CI
  gate extended with permutation + opaque-RGBA anchors; trace tools gained
  sharpness/partition_limit args and new dump hooks. Scope:
  `benchmarks/byteparity_scope_2026-07-14.md`.
- **Tuned default adopts the trellis-derived skip at m5/m6** (#38 follow-up,
  2026-07-16): the tuned trellis path decided the per-MB skip with a separate
  simple-quant test while emission recorded trellis levels — on borderline
  MBs the encoder reconstructed (and predicted from) a coefficient the
  decoder never received. The skip now derives from the recorded levels,
  matching the parity arm. Measured on the 15-image A/B corpus
  (`dev/tuned_ab_sweep.rs`, new): m5 +0.011% size / **+0.104 zsim**, m6
  +0.015% / **+0.097**, m4 control exactly 0; gain largest at low q (+0.156
  at q25). `check_all_coeffs_zero` removed. Provenance:
  `benchmarks/tuned_candidates_2026-07-16.md` + companion TSV.
- **`StrictLibwebpParity` byte-exact across the FULL 4004-cell grid —
  100%** (#38, 2026-07-16): 99.75% → **4004/4004**. The last 10 cells (all
  real photos, q80+) were one root: exact I4 RD-score ties broken toward
  different modes. zenwebp's `B_*` mode constants use the VP8 spec order
  (`LD=4, RD=5, VR=6`) while libwebp's internal enum permutes them
  (`RD=4, VR=5, LD=6`), and libwebp keeps the first minimum iterating ITS
  order — so an exact LD-vs-VR tie (382297 q80 m3 mb(26,16) blk3: both score
  145452) picked different winners and cascaded 264 MBs. Parity now iterates
  `LIBWEBP_I4_ORDER` (libwebp's visit order in zen indices) in all three I4
  evaluator paths. The CI gate gains regression anchors for the four
  2026-07-16 roots (skip-proba, trellis-skip, libm-pow, tie-break order:
  two SNS configs × q5/q20/q80 × m0-m6 + the 33x17 pow-boundary cell). The
  proven claim is byte-exactness across the committed grid — 13 images
  (CID22 photos + tiny/odd synthetics) × q5-95 × 4 (sns,flt,segs) configs ×
  m0-m6; unswept axes (sharpness≠0, partitions>1, alpha, target_size) need
  sweep extension before claim extension. Tuned default byte-unchanged
  throughout. Scope: `benchmarks/byteparity_scope_2026-07-14.md`.
- **`StrictLibwebpParity` byte-exactness at 99.75% of the grid; all
  synthetics closed** (#38, 2026-07-16): 99.6% → **99.75% (3994/4004)**, +5.
  The synth_33x17 q90 sns50/segs4 cluster (m2-m6) was one root:
  `compute_segment_quant` evaluated libwebp's
  `pow(QualityToCompression(Q), expn)` with the fast polynomial `pow`/`cbrt`
  approximations, and the ~1e-10 error flipped the truncated
  `(int)(127*(1-c))` quant index at an integer boundary (seg1 12 vs
  libwebp's 11, skewing seg_lf and downstream mode decisions). Parity now
  uses `compute_segment_quant_libm`/`quality_to_compression_libm` — the
  exact libm chain, including the base `pow(linear_c, 1./3.)` (libwebp never
  calls cbrt); the segs1-only `quality_to_quant_index_trunc` routes through
  it too. Tuned default keeps the fast path, bytes unchanged.
  `mbpixdiff`/`bitexact_diff` accept `synth:WxH:SEED` image specs
  reproducing the sweep's synthetic cells. Remaining 10 cells: real photos
  q80+ in 5 clusters. Scope: `benchmarks/byteparity_scope_2026-07-14.md`.
- **`StrictLibwebpParity` byte-exactness at 99.6% of the grid; m5/m6 trellis
  residue closed** (#38, 2026-07-16): 97.9% → **99.6% (3989/4004)**, +67
  cells (m5 39→4, m6 35→3). zen decided the per-MB skip with
  `check_all_coeffs_zero` — a SIMPLE re-quantization of the raw DCT — but at
  m5/m6 libwebp skips on `rd->nz == 0`, the nz of the FINAL trellis
  quantization; the trellis keeps borderline coefficients the simple bias
  drops (neutral-bias level0 + sharpen + RD), so zen skipped MBs libwebp
  coded. The parity trellis path now records the actual levels and derives
  the skip from them, and fires `StoreMaxDelta` before the skip test
  (libwebp stores it at the end of `PickBestIntra16`). Remaining 15 cells
  are all q80+. Also adds diffable diagnosis hooks (`REFRESHDBG2`,
  `LEVFINAL`, `TRELDBG`; `mode_debug`-gated) format-matched to the
  instrumented libwebp tree. Tuned default byte-unchanged (its simple-quant
  skip test is retained; switching is a measured-adoption candidate — it
  fixes an encoder-side reference mismatch where a skipped MB's encoder
  reconstruction keeps a coefficient the decoder never sees). Scope:
  `benchmarks/byteparity_scope_2026-07-14.md`.
- **`StrictLibwebpParity` byte-exactness at 97.9% of the grid; m0-m2 closed**
  (#38, 2026-07-15): the committed byteparity grid (13 images × q5-95 × 4
  configs × m0-6 = 4004 cells) went 89.4% → 94.1% (Cat5/Cat6 stat-node,
  `44ae3a0`) → 95.6% (StoreMaxDelta from the I16 candidate, `c9abe85`) →
  **97.9% (3922/4004)** with the **m0-m2 skip-proba fix** (+93 cells, closing
  the entire m0-m2 axis, 94 → 1). Parity had forced `use_skip_proba = 0` for
  every method, but libwebp's assert only covers the m3-m6 token loop; m0-m2
  run `VP8EncLoop`, whose StatLoop finalizes the flag. Reproduced exactly:
  `nb_skip` counted over the stats subset (m0 `fast_probe`: first
  `total>200 ? total>>2 : 50` MBs, frozen via `fast_probe_skip_count`;
  m1/m2 full frame), full-frame denominator with truncated unclamped
  `CalcSkipProba`, `use_skip = skip_proba < 250`, and the `size_p0 == 0`
  StatLoop bailout (single effective segment → finalize never runs → flag
  stays 0). Remaining 82: m5/m6 trellis residue (39/35), m3/m4 stragglers
  (4/3), one m2 synth cell. Parity-gated; tuned default byte-unchanged; full
  suite + `libwebp_byte_parity` gate + clippy (default & `__expert`) green.
  Scope + SKIPDBG evidence: `benchmarks/byteparity_scope_2026-07-14.md`.
- **`StrictLibwebpParity` byte-exactness generalized to 87.1% of the grid** (#38,
  2026-07-15): five parity-gated fixes took the broad byteparity grid (13 images ×
  q5–95 × 4 configs × m0–6 = 4004 cells) from **24% → 87.1%**. The latest two:
  (4) **skip-proba forced off** — instrumented libwebp always writes
  `use_skip_proba = 0` (unconditional `assert(use_skip_proba == 0)` at
  `VP8EncTokenLoop` entry), so parity forces `macroblock_no_skip_coeff = None`
  (+256 low-q cells). (5) **I16-AC-trellis nz-context seed** — at m6
  (RD_OPT_TRELLIS_ALL) zen's `pick_best_intra16` seeded the AC-trellis per-block nz
  context all-false, but libwebp's `ReconstructIntra16` calls `VP8IteratorNzToBytes`
  first so its trellis uses the REAL neighbour nz; for MBs whose neighbours carry
  coefficients this flipped an I16 sub-mode (H vs DC) and cascaded to I4-vs-I16.
  Seeding from `top_complexity`/`left_complexity` (parity-gated) fixed it (+81
  cells; first emitted divergence at mb(11,8) traced byte-identical). All
  parity-gated; tuned default byte-unchanged; q75 still 14/14; full test suite
  green (689 passed). Scope: `benchmarks/byteparity_scope_2026-07-14.md`.
- **`StrictLibwebpParity` byte-exactness generalized to 78.7% of the grid** (#38,
  2026-07-15, `52cf96f2`/`41923466`/`7acdd775`): three parity-gated fixes took the
  broad byteparity grid (13 images × q5–95 × 4 configs × m0–6) from **32% → 78.7%**.
  (1) **Base quant truncates** — `setup_encoding` rounded `127*(1-c)` where libwebp
  `(int)`-truncates, diverging at q10/30/50/80. (2) **Segmentation disabled on
  collapse** — libwebp writes `segmentation_enabled = (num_segments > 1)` after
  `SimplifySegments`; at sns=0 all segments are uniform and collapse to 1, so
  libwebp turns segmentation OFF, but zen emitted a full 4-segment header (the whole
  sns0/segs>1 config went 0% → 78% identical). (3) **Trailing segment slots** — the
  4-slot segment header's unused `[config..4]` slots use libwebp's base/seg0 values,
  not seg1's. All parity-gated; tuned default byte-unchanged; q75 still 14/14.
  Remaining ~21% is a common low-q (skip-proba/StatLoop #25+#27), high-q (token
  proba + I4 sub-mode), and m6 (mode-RD) tail. Scope + next steps:
  `benchmarks/byteparity_scope_2026-07-14.md`.
- **`StrictLibwebpParity` byte-exact at the traced operating point** (#38,
  `816aea5`/`6b4fa0c`/`c96f767`+`8b60a62`/`8256bec`): all 14 (method × config)
  cells are byte-identical to libwebp **at q75 on CID22 382297** — m0–m6 at
  (SNS=0, filter=0, segs1) and (SNS=50, filter=60, segs4), verified by
  `methodcmp`. Closes the residual m3-6 cascade at that point: I4 flatness
  penalty in the running total, I16 flat-source latch, I4 trellis static
  context, chroma-DC double-correction, StoreMaxDelta blocky-nz from the
  mode-selection quant state, and container even-padding inside the VP8 chunk.
  **Scope caveat — bit-exactness is NOT yet general:** a broad sweep (13 images,
  q5–95, 4 configs, m0–6 = 4004 cells) is only 972/4004 (24%) byte-identical;
  two configs (SNS=0/segs4, SNS=30/filter=20/segs2) never match and identity
  oscillates 12–34% across q. The fixes were traced at q75/382297 and don't
  generalize; #38 stays open. Full investigation:
  `benchmarks/bitexact_parity_2026-07-14.md`;
  scope: `benchmarks/byteparity_scope_2026-07-14.md`.
- **Tuned default adopts two parity-gated wins** (#38-D): measured on CID22 +
  imazen-26 and adopted into `ZenwebpDefault` — (1) `max_i4_header_bits` now
  uses libwebp's `partition_limit`-derived value (65536 at the default) instead
  of the historical 16384 I4-suppression band-aid, −0.16% size / +0.014 zsim at
  m4/m6 (`b9fd7cb`); (2) UV modes are RD-scored on the diffused reconstruction
  at m3-6 (matching what emission produces), −0.22% size / +0.107 zsim
  (`816aea5`). FastMBAnalyze-alpha (+4.25% bytes), the I4 flatness penalty
  (wash), and a mid-row level_costs refresh (regression) were measured and
  rejected. Provenance: `benchmarks/tuned_candidates_2026-07-14.md` +
  `benchmarks/maxi4_header_bits_2026-07-14.md`.
- **Byte-exact chroma downsampling under `StrictLibwebpParity`** (#38): the
  gamma-corrected 2×2 chroma downsampling now offers libwebp's YUV_FIX+2 (×4)
  precision path (`ChromaPrec::LibwebpExact`), byte-identical to
  `WebPPictureARGBToYUVA` (U/V: 0/65536 pixels differ on CID22 382297; UV mode
  agreement at m0 96.2% → 100%). The tuned default keeps the byte-rounded path
  (`ChromaPrec::TunedByteRound`), which scores measurably higher on synthetic
  low-q gradient/noise. Both share one SIMD kernel (a 16384-entry LUT selects
  the mode); the precision is chosen from `cost_model` in
  `prepare_input_for_encoding`. Also fixes a latent U/V cast-wrap at
  out-of-range colors (now clamped) and the odd×odd corner (now gamma-averaged
  like libwebp, was a bare per-pixel convert). Forward-DCT bit-exactness vs
  libwebp `FTransform_C` is locked by new differential tests in
  `src/common/transform.rs`. See `benchmarks/bitexact_parity_2026-07-14.md`.
- **m3-6 RD-path parity fixes under `StrictLibwebpParity`** (#38): four root
  causes of the m3+ RD divergence, closing much of the gap on 382297 q75 segs1
  (y_same 89.7→92.0%, uv_same 80.6→84.5%, I4 count 914→888 vs lib 874, bytes
  51632→51582 vs lib 51386; m0-2 remain byte-identical). (1) **Zigzag cost
  order** (3b90e27, unconditional): the I16/UV RD coefficient cost
  (`get_cost_luma16`/`get_cost_uv`) now walks coefficients in zigzag scan order
  like libwebp, instead of natural (raster) order — the natural-order walk used
  the wrong per-position bands and `last`, systematically under-costing and
  flipping mode picks (the I4 path already zigzagged). (2) **Chroma DC error
  diffusion in UV RD scoring** (0692ad6, parity-gated): `pick_best_uv` now
  applies `CorrectDCValues` per candidate mode like libwebp's `ReconstructUV`,
  via a shared `diffuse_chroma_dc_inplace`. (3) **I16/UV mode tie-break order**
  (0eeff05, parity-gated): ties now resolve to the lowest libwebp mode number
  (eval order DC,TM,V,H) so edge TM/H ties pick the same mode. (4) tlambda
  `CheckLambdaValue(>=1)` clamp documented (5488ef1) — correct parity value but
  left off pending an I4 sub-block ctx0 fix it regresses without. Residual gap
  is cascade-dominated (I4 sub-block context + downstream reconstruction).
- **Lossless m5/m6 predictor transform-bits search** (#70, 37cae10):
  VP8LResidualImage parity — methods above 4 search every predictor
  sampling in `[max_bits − 2·(m−4), max_bits]` by mode-usage + residual
  entropy, and the winning mode image is coarsened via OptimizeSampling
  (now also applied to the cross-color multiplier image). Photos vs
  libwebp: m5 1.0086× → 1.0011×, m6 1.0078× → 1.0003× (the m3–m6 ladder
  was byte-flat before). `benchmarks/msweep_post70_2026-07-14.tsv`.
- **Independent no-cache dual refs at m5+ q≥75** (#70, cd6714c):
  libwebp `do_no_cache` parity — the cache-free token stream gets its own
  LZ77-type selection, TraceBackwards, and full encode trial, replacing
  the strip-based cache trial (which measurably never won).
- **`dev/bitexact_diff.rs`** (#38): three-level bitstream diff harness vs
  libwebp (frame-header fields, per-MB keyframe mode streams, agreement
  percentages) with a self-contained RFC 6386 boolean decoder.
- **True low-effort lossless m0 tier** matching libwebp's `low_effort`
  shortcuts: skip entropy analysis (palette else SubtractGreen+Predictor),
  fixed Select predictor for all tiles, no cross-color, single plain-LZ77
  pass, 4-bin unconditional histogram merging without stochastic/greedy.
  Photo 512²: 71 ms → 29 ms (−59%), instructions −71%. m1+ output
  byte-identical. The old m0 operating point was a near-duplicate of m1
  (same pipeline, coarser clustering tiles) and remains reachable via m1.
  Measurements: `benchmarks/lossless_fast_tier_2026-07-14.{md,tsv}`.
- **m0 speed/size round 2**: dedicated fixed-Select residual pass (31M → ~3M
  instructions on photo 512²; forward-streaming, no per-pixel mode dispatch);
  hash-chain hot loop reworked to instruction parity with libwebp's
  VP8LHashChainFill (4-pixel-compare `vector_mismatch`, fold-checked quick
  reject, boxed `[i32; HASH_SIZE]` head table); row-above match heuristic
  kept ON at m0 (dice −7% bytes); fixed-bits color cache at m0 with a single
  A/B histogram-cost accept check instead of the 0..=10 search (photo −6.4%,
  dice −11%, frymire −9% bytes; declined automatically where it loses).
  Net: m0 is now smaller than libwebp m0 on the whole benchmark grid
  (−5% to −19%) at ~1.0–1.2× its wall time. An m0 chain-iteration cap was
  measured and rejected (+8.5% to +75% on smooth gradients).
- **m0 chain-walk no-progress pruning** (d8c9e3d): walks are abandoned after
  24 consecutive non-improving candidates instead of a flat iteration cap —
  productive deep walks keep their budget. Photo −0.7% bytes and ~20% less
  wall, frymire −30% wall (+0.45%), real-content sizes within ±0.5%; the
  synthetic diagonal-gradient stress case pays +5.3%. m1+ byte-identical.
- **Wire the `zencodec-testkit` `check_decode_truncation_series` EOF/truncation
  conformance check** into `tests/decode_truncation_series.rs` — truncates a
  known-good WebP at a deterministic prefix series and asserts every dyn-erased
  decode failure categorizes as incomplete-input (never panic/OOM/`Internal`).
  Bumps the `zencodec` git patch (root + `fuzz/`) to rev `c3220d51` (zencodec PR
  #112), which carries the testkit; the testkit dev-dep is pinned to the same rev.
- **Adopt the `zencodec` `CategorizedError` taxonomy (PR #103).** The public
  encode/decode error types — `DecodeError`, `EncodeError`, `mux::MuxError`,
  `ValidationError`, and `detect::ProbeError` — now `impl
  zencodec::CategorizedError` with
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
  `zencodec 0.1.26` ships. At this point `EncodeError::LimitExceeded(String)`
  was stringly and mapped to a representative `Memory` kind; see the reshape
  entry below for the follow-up that gives it the exact `LimitKind`.
- **Reshape `ErrorCategory` to the origin-first two-level taxonomy (zencodec
  PR #116)**: `ErrorCategory` becomes `Image(ImageError)` /
  `Request(RequestError)` / `Resource(ResourceError)` / `Policy(PolicyKind)` /
  `Lifecycle(enough::StopReason)` / `Io(CodecIoKind)` /
  `Internal(InternalKind)`, replacing the flat 17-variant enum above — the
  origin split (image-bytes vs caller-request) makes ambiguous cases
  structurally unambiguous instead of conflating them under one flat
  "unsupported"/"invalid" axis. `ErrorCategory` was never published, so this
  is not a break of any released API. Bumps the `zencodec`/`zencodec-testkit`
  git patch (root + `fuzz/`) from rev `c3220d51` to `2427387f`. All five
  `category()` maps (`DecodeError`, `EncodeError`, `MuxError`,
  `ValidationError`, `ProbeError`) rewritten to the new shape;
  `decode_truncation_series` conformance still passes. Two mapping bugs fixed
  in the same pass (found by a 13-codec ripgrep audit,
  `errorcategory-audit-2026-07-13.md`):
  - `TargetZensimUnsupportedLayout` now maps to
    `Request(Unsupported(PixelFormat))` (was `InvalidParameters`) — it's a
    well-formed request for a pixel layout the closed-loop iteration doesn't
    support yet, not a bad parameter value.
  - `NoMoreFrames` now maps to `Request(Invalid(State))` (was
    `Image(UnexpectedEof)`) — reading one frame past `num_frames()` via the
    low-level `WebPDecoder::read_frame` is a normal end-of-animation signal
    the caller could check for ahead of time (the high-level
    `mux::AnimationDecoder::next_frame`/`decode_next` wrapper already turns it
    into a clean `Ok(None)` before it ever reaches `category()`), not
    truncated/incomplete image bytes. The old mapping would also have made it
    silently pass the truncation-conformance check's incomplete-input
    tolerance (`is_incomplete_input_category` accepts the whole `Image(_)`
    arm), which was wrong — this has nothing to do with a truncated
    bitstream. See "Changed (BREAKING)" for the `EncodeError::LimitExceeded`
    kind-carrying follow-up landed in the same pass.
- **Honor `zencodec::AllocPreference` (3-mode, per-site) at untrusted decode
  allocations**: the big buffers sized from decoded header
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

- **`estimate_decode_resources` override on `WebpDecoderConfig`**: implements
  zencodec's unified
  `DecoderConfig::estimate_decode_resources` via the existing
  `heuristics::estimate_decode` model — peak = output buffer + VP8 working set
  (decoder state + ~12 B/px for the row cache, reconstruction buffer and
  loop-filter accumulator), reported as `ThreadingInformation::SERIAL`
  (zenwebp decode is single-threaded). Returns a
  `zencodec::estimate::ResourceEstimate`.

- **`estimate_encode_resources` override on `WebpEncoderConfig`**: implements
  zencodec's unified `EncoderConfig::estimate_encode_resources`
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

### Changed
- **Lossy m2 uses libwebp's full `RefineUsingDistortion` decimation**
  (try_both + refine_uv): SSE-scored I16/I4 picks with the accumulate-and-
  bail against the I16 score (`i4_penalty = 1000·q_i4²`, header bits capped
  by `mb_header_limit`), SSE-scored UV refine; m1 also adopts the SSE UV
  refine (libwebp `refine_uv_mode = m ≥ 1`). zenbench: **m2 9.6 → 4.1 ms
  (3.0× → 1.30× of libwebp)**, m1 ~1.45×; m2 bytes/PSNR shift to libwebp-
  m2-like RD points (photo 12,772 B/36.91 dB; +12.8% bytes vs libwebp m2,
  within all matrix gates). The SSE I4 pick loop runs in a single
  `#[arcane]` region (739bc14 recipe, bit-exact). m3+ RD paths unchanged.
- **Lossy m0/m1 mode decimation is SSE-scored** (libwebp
  `RefineUsingDistortion` shape): I16-hint macroblocks pick among all four
  modes by SSE + fixed cost (was DC-only), I4-hint macroblocks pick each
  sub-block's mode from all ten candidates by SSE with a single winner-only
  quantize (was a 3-mode full-RD loop). Wall-neutral on zenbench; photo-512
  bytes −2.7%, imazen-26 screenshot corpus −2.2% bytes at equal PSNR. Two
  synthetic zensim-floor cells re-baked for the new mode decisions
  (sub-point deltas). m2+ unchanged.

### Fixed

- **StrictLibwebpParity: ALL 14 of 14 method×segment cells now byte-identical
  to libwebp** (#38, 8256bec / 8b60a62 / c96f767 / 6b4fa0c). Four more
  parity-gated fixes (tuned default unchanged) closed the last three cells
  (m4-segs4, m5-segs1, m5-segs4): (1) **I16 flat-source penalty as a latch** —
  libwebp refines a single `is_flat` flag in DC→TM→V→H order and doubles a
  mode's D/SD only while it holds (`quant_enc.c:1044`); zen checked each mode's
  coeff-flatness independently and over-doubled (closed m4-segs4). (2)
  **I4 trellis static context at m5** — libwebp's `SimpleQuantize`/
  `ReconstructIntra4` reads the trellis rate context from the inter-MB neighbour
  nz and never updates it between sub-blocks (its m6 path updates per sub-block
  in `PickBestIntra4`); zen updated per sub-block, so at m5 it used the wrong
  context. (3) **m5 chroma DC double-correction** — m5 reconstructs chroma
  twice (`PickBestUV` then `SimpleQuantize`), and the second `CorrectDCValues`
  reads the errors `StoreDiffusionErrors` just wrote (this MB's own), so the
  final coded chroma DC uses the self errors, not the neighbour errors (closed
  m5-segs1). (4) **m5 blocky-I16 filter delta from mode-selection nz** —
  `StoreMaxDelta`'s all-Y1-AC-zero test uses the non-trellis PickBestIntra16 nz;
  zen used the m5 trellis quant (which zeros more AC), inflating `seg_lf`
  (closed m5-segs4). Byte-identical to libwebp on 382297 for **all** of
  segs1 m0-m6 and segs4 m0-m6. Full inventory:
  `benchmarks/bitexact_parity_2026-07-14.md` (part 14).
- **StrictLibwebpParity: 11 of 14 method×segment cells now byte-identical to
  libwebp** (#38, 44b6a38 / a899036 / 4d41a33 / defebc5 / e6ee888). Six
  parity-gated fixes (tuned default unchanged): (1) pad the VP8 bitstream to
  even *inside* the `VP8 ` chunk like libwebp's `VP8EncWrite` instead of a RIFF
  pad after it (differed on every odd-length stream); (2) finalize method-0
  probabilities from libwebp's `fast_probe` MB subset (`nb_mbs>>2`) via a
  `proba_stats` snapshot; (3) clamp `tlambda` to ≥1 (`CheckLambdaValue`) +
  (4) fold the I4 `FLATNESS_PENALTY` into the I4-vs-I16 running total + (5) use
  libwebp's `max_i4_header_bits = 256·16·16·(100−partition_limit)²/100²` (was
  hardcoded 16384) and rebuild `level_costs` mid-pass alongside the proba
  refresh (`VP8CalculateLevelCosts`); (6) break I4 sub-block mode ties toward
  the lower index (libwebp iterates 0..9; zen presorted by SSE). Byte-identical
  to libwebp on 382297: **segs1 m0/m1/m2/m3/m4/m6; segs4 m0/m1/m2/m3/m6** (all
  but m5 both + m4-segs4). The part-11 "coefficient-proba recording bug"
  hypothesis was **falsified** — the stat histogram is byte-exact (0/1056 cells
  differ where modes match); that count gap was `fast_probe` + the RD mode
  cascade, not recording. Remaining: m5 (trellis, Task 3) and m4-segs4 (a
  `VP8TDisto` spectral-distortion precision cascade, only visible at the large
  `tlambda` of m4/sns>0). Full inventory:
  `benchmarks/bitexact_parity_2026-07-14.md` (parts 11-13).
- **Lossy m0-m2 emitted the wrong Intra4 mode for LD/RD/VR sub-blocks**
  (cd5cc85, #38): the m0-m2 I4 mode pick indexed a mode-lookup array in
  libwebp's internal B-mode numbering while scoring in zenwebp's IntraMode
  order, so a winning LD/RD/VR *prediction* was emitted and reconstructed as a
  different mode. Every affected sub-block got a worse prediction and larger
  residual. Fix: derive the mode from the winning index via `IntraMode::from_i8`.
  Equal-PSNR lossy corpus: **m0 −2.14%, m1 −2.14%, m2 −7.88%** bytes; m3-m6
  unaffected (already correct). `benchmarks/bitexact_parity_2026-07-14.md`.
- **Forward Walsh-Hadamard transform rounded incorrectly** (4a154e1, #38):
  `wht4x4` (I16 Y2 DC transform) finalized with `(x + (x>0?1:0))/2` instead of
  the VP8/libwebp arithmetic `x >> 1` (floor), diverging on every odd
  intermediate and occasionally flipping a Y2 DC coefficient across a
  quantization boundary. Fixed to `>> 1`; quality- and size-neutral on the
  corpus (±0.02%), spec-correct.
- **VP8 lossy m0/m1 single-segment collapsed every macroblock to I16-DC**
  (331f386, #38): FastMBAnalyze mode hints were only populated when
  `num_segments > 1`; with one segment the encoder never explored V/H/TM/I4.
  Fix restores full mode selection — decode PSNR at m0 matches libwebp's.
- **VP8L encoder wrote Huffman tree groups for clusters unreferenced by the
  entropy image**: histogram remap can strand a cluster with zero tiles, and
  `build_final_histograms` still emitted its (empty) trees — but decoders
  size the group list from the entropy image's max symbol, so an
  unreferenced trailing cluster shifted the rest of the bitstream (silent
  whole-image corruption on decode). Latent for any method; easiest to hit
  with the new m0 clustering. Final groups are now compacted to referenced
  clusters with dense renumbering; regression coverage in
  `tests/lossless_fast_tier.rs` and `tests/lossless_stranded_cluster.rs`
  (#72). **Shipped broken in 0.4.4 at default settings** — 84% of pixels wrong
  on one 512x320 rendition, `BitStreamError` on others; the stream is
  well-formed so decoders returned garbage rather than erroring. The gate
  sweeps m4/m5/m6 across the quality axis on content verified to strand a
  cluster (a 511x320 crop, seed 42, fails at m4/q25 with the compaction
  reverted — matching the reported renditions), plus two unit tests pinning
  the compaction itself in `meta_huffman.rs`. All four were confirmed to fail
  against a deliberately reintroduced defect, so they gate rather than merely
  pass.
- **VP8L decoder rejected valid deep-Huffman-tree streams** (efddb6c3): the
  pixel-loop bit-window refills were under-budgeted — a literal's
  GREEN+RED+BLUE+ALPHA codes can need 60 bits but `fill()` guarantees 56 and
  the mid-literal refill only guarded 15; the backref path similarly lacked
  headroom for DIST symbol + 18 distance extra bits (33). Valid libwebp
  `-m 0 -lossless` files (literal-heavy, near-max-depth trees) failed with a
  spurious `BitStreamError` mid-stream. Both refills now use worst-case
  budgets; differential regression test vs libwebp in
  `tests/vp8l_deep_tree_decode.rs`.
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
