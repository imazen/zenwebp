# zenwebp-recompress — Design & Research Plan

**Goal.** Given an already-compressed WebP (lossy VP8 or lossless VP8L) and a
zensim Profile A target in `[0, 100]`, produce the smallest output WebP whose
true zensim-A vs the **original unknown reference** is at least the target.
Recompress only when it shrinks the file at the target quality; otherwise emit
a lossless re-mux. Never go below the target. Decisions must hold across source
quality `Q20`–`Q100` (step 2) for **any** encoder commonly seen in the wild
(libwebp, cwebp, zenwebp, Sharp, ImageMagick, Pillow).

This is harder than a normal RD encoder because:

1. **The reference is gone.** The user has the lossy source, not the original.
   Our zensim-A measurements are always against the source itself, so they
   describe the *recompression generation loss*, not the *cumulative* distance
   from the original. The product target is *cumulative* zensim-A vs original.
2. **The source is already a WebP.** Its DCT residuals carry block structure
   from VP8's 4×4 transform, ringing from the loop filter, and quantization
   error that any naïve pixel-domain re-encode will encode again. Generation
   loss compounds.
3. **Encoder identity matters less than for JPEG.** libwebp is dominant — Sharp,
   Pillow, ImageMagick, cwebp all share its core. zenwebp is the main Rust
   alternative and has known RD differences (XYB-aware AQ, trellis). Two
   families is enough; we don't need an 8-encoder taxonomy.
4. **Sometimes recompression cannot win.** A libwebp Q20 file is already past
   the point where any standard WebP encoder produces a smaller file at the
   same visual quality. The only correct action is a lossless container
   normalization (drop EXIF, normalize ICC, re-tag, drop unused chunks).
5. **Lossless reencoding is a real option.** For sufficiently quantized
   graphics/screen content, encoding the decoded pixels as VP8L can shrink the
   file *and* eliminate further generation loss.

## Public API contract (frozen)

```rust
pub struct RecompressOptions {
    pub target_zensim_a: f32,         // 0.0..=100.0
    pub budget: Budget,
}

pub enum Budget {
    OneShot,                          // default: no IQA loop, no measurements
    MaxIterations(u32),               // expert: bounded measure-and-iterate
    MaxTime(std::time::Duration),     // expert: wall-clock bounded
}

pub enum RecompressResult {
    Recompressed {
        bytes: Vec<u8>,
        strategy: StrategyKind,        // which path was chosen
        projected_zensim_a: f32,       // model-predicted, NOT measured (one-shot)
        measured_zensim_a: Option<f32>,// Some(_) only when Budget allows it
        source_to_output_ratio: f32,   // output_len / input_len
        better_handled_by_jxl: bool,   // hint: caller may want a JXL transcode
    },
    LosslessOnly {
        bytes: Vec<u8>,                // metadata-stripped re-mux
        reason: LosslessReason,        // WhyRecompressionLost
        better_handled_by_jxl: bool,
    },
    NoOp {
        reason: NoOpReason,            // source already meets target
    },
}

pub enum StrategyKind {
    /// Coefficient-domain VP8 tighten (no pixel round-trip).
    CoeffEdit,
    /// Decode + deblock + re-encode at calibrated Q.
    DeblockReencode,
    /// Decode + re-encode at calibrated Q (no deblock).
    Reencode,
    /// Decode + re-encode as VP8L.
    LosslessReencode,
    /// Container re-mux only (no recompression).
    LosslessRemux,
}

pub fn recompress(webp_bytes: &[u8], opts: &RecompressOptions)
    -> Result<RecompressResult, Error>;
```

All other types live under `#[cfg(feature = "expert")] pub mod expert`. The
default API has **one entry point, three enums, and one function** — that is
the "minimal" contract. Adding fields is allowed via `#[non_exhaustive]`.
Removing or renaming is a major-version break.

## Strategy taxonomy

Six strategies, each with a different point on the (generation-loss × bitrate
× CPU-cost) surface.

### `NoOp` — pass through
The source's projected zensim-A already meets the target with no margin to
shrink further. Return input bytes unchanged.

### `LosslessRemux` — container-only edit
Keep the VP8 / VP8L bitstream verbatim. Optionally:
- Drop `EXIF`, `XMP` chunks (unless the caller asks to preserve).
- Normalize `ICCP` chunk (re-encode profile to canonical form, drop if it
  matches sRGB exactly).
- Re-pack RIFF chunks in canonical order, drop trailing padding.
- Convert from extended VP8X to simple VP8/VP8L when no extended features are
  used.

Zero generation loss. Wins when the source is too aggressively compressed for
any standard re-encode to beat it.

### `CoeffEdit` — VP8 coefficient-domain tighten (built, RD-falsified for WebP)
Parse the VP8 keyframe to quantized coefficient **levels** (stop before IDCT),
edit the levels, and re-emit the token stream — **no IDCT/FDCT spatial
round-trip**. Implemented as a self-contained transcoder in `src/vp8x/`
(`parse_vp8_keyframe` → `edit` → `emit_vp8_keyframe`), whose verbatim path is
pixel-exact with libwebp. Two edits exist: `drop_high_freq_ac` (zero AC scan
positions ≥ keep) and `requantize` (coarsen the level grid).

**Measured 2026-05-28: this loses to `Reencode` for WebP and is not selected
by the router.** Every size-reducing coefficient edit is RD-dominated at
matched output size because VP8 predicts each block from its neighbours'
reconstructed pixels — editing coefficients breaks that prediction invariant
and the error drifts across the whole frame, while `Reencode` re-derives
residuals from clean pixels and re-runs RD optimisation. The "zero generation
loss" benefit is only real at the verbatim point (no size change), which
`LosslessRemux` already provides. Coefficient transcoding is the right tool for
prediction-free codecs (baseline JPEG) — VP8's intra prediction defeats it.
Full data + mechanism: `benchmarks/coeff_edit_experiment_2026-05-28.md`. The
transcoder + edits remain reachable via `expert::run_coeff_edit{,_keep,_requant}`
as research tools. **Lossless VP8L sources never participated in `CoeffEdit`.**

### `Reencode` — full decode + re-encode at calibrated Q
Decode through zenwebp's decoder to RGBA, re-encode at a calibrated quality.
One IDCT/FDCT pair of generation loss. Wins for medium-quality sources where
the lower-bound on the target is large enough that the recompression overshoots
even with the loss.

### `DeblockReencode` — full decode + perceptual deblock + re-encode
Decode, run a content-aware deblock filter to remove residual block boundary
artifacts and ringing that the original encoder's loop filter didn't fully
suppress, then re-encode at a calibrated quality. Higher CPU and the same
single IDCT/FDCT loss, but removes block artifacts the source baked in, so the
target zensim-A can be reached at a much lower output bpp.

The deblock filter is shared with zenwebp's existing `loop_filter` module,
extended for off-grid block sizes when the source was encoded with non-default
filter strength.

### `LosslessReencode` — decode + VP8L
Decode to RGBA, encode as VP8L. Wins for content with low entropy after
decoding (graphics, screen content, posterized photos where quantization has
already collapsed gradients into few values). Zero further generation loss
after the one decode; output is by construction a perfect reproduction of
the source-as-decoded pixels.

## Decision tree

```text
probe(source) -> (kind, encoder_family, source_q, has_alpha, has_animation, container)
analyze_content(source) -> (content_class, has_block_artifacts, posterization_score)

let source_zensim_a_estimate = encoder_q_to_zensim_a(encoder_family, source_q,
                                                     subsampling, content_class);

// 1. Source already meets target with no margin to recompress?
if source_zensim_a_estimate < target + ZENSIM_A_NOOP_BAND {
    return NoOp { SourceAlreadyMeetsTarget };
}

// 2. Project each strategy's (size, zensim-A).
let candidates = [
    calibration.coeff_edit(encoder_family, source_q, target, content_class),
    calibration.deblock(encoder_family, source_q, target, content_class),
    calibration.reencode(encoder_family, source_q, target, content_class),
    calibration.lossless_reencode(encoder_family, source_q, content_class),
    calibration.lossless_remux(encoder_family, source_q),
];

// 3. Pick the smallest at-or-above target zensim-A.
let best = candidates.iter()
    .filter(|c| c.projected_zensim_a >= target)
    .min_by(|a, b| a.projected_bytes.cmp(&b.projected_bytes));

// 4. If no recompression strategy meets target, or the smallest is larger than
//    source, ship LosslessRemux.
match best {
    Some(c) if c.projected_bytes < source.len() => Strategy::dispatch(c.kind, ...),
    _ => Strategy::dispatch(LosslessRemux, ...),
}
```

The output exposes `better_handled_by_jxl` when all of these are true:

- All non-remux strategies projected zensim-A < target, OR projected bytes >=
  source.
- Source was lossy (VP8) at Q ≤ 30, with `has_block_artifacts = true`.
- Predicted JXL recompression (via zenjxl's own table, when available) is
  projected to shrink the file by ≥10% at target zensim-A.

The flag is a *suggestion* — the recompressor still emits a valid WebP. The
caller decides whether to switch formats.

## Calibration data design

For each encoder family ∈ {`Libwebp`, `Zenwebp`}, for each source quality
(`q_in ∈ {20, 22, …, 100}`, 41 bins), for each content class ∈ {`Photo`,
`Screen`, `LineArt`, `Mixed`}, for each strategy ∈ {`CoeffEdit`,
`DeblockReencode`, `Reencode`, `LosslessReencode`, `LosslessRemux`}, for each
target zensim-A ∈ `{0, 5, 10, …, 100}` (21 bins), we need:

- The conditional distribution of `(output_zensim_a_vs_reference,
  output_size_ratio)` over ≥40 content samples per cell.

This is a 6-axis grid:

| axis | cardinality | source |
|------|---|---|
| encoder family | 2 | `zenwebp::detect::probe` |
| source quality bin | 41 | step-2 |
| content class | 4 | zenanalyze tier-2 features → k-means |
| target zensim-A | 21 | step-5 |
| strategy | 5 | enum |

For the **balanced** trail we ship `(p50, p25)` quantiles per cell over the
≥40 content samples. For the **compression** trail we ship the same plus a
`(p50_size_aggressive, p25_zensim_a)` estimator that maximizes shrinkage
subject to a zensim-A risk bound.

Table shipped as `include_bytes!()` parquet (zstd-compressed, ~50 KB target)
plus a generated `data.rs` with the schema constants. Loader is zero-alloc and
lazy. Validation holdout uses `cid22` references, never co-trained.

### Why two encoder families is enough

libwebp's `cwebp -q` is the dominant production encoder. Sharp, Pillow, NodeJS
`@squoosh/webp`, ImageMagick — all wrap libwebp via `libwebp-sys`. They produce
byte-identical output for the same `(quality, method)` inputs in the lossy
path. zenwebp's `target_zensim` path is the main divergent encoder in the wild.
A third family ("other" — handcrafted bitstreams, vp8enc, FFmpeg's webpenc)
falls back to the libwebp family table with a fixed risk-margin bump.

### Calibration corpus

- **Sources.** codec-corpus `sc` set + cid22 references. ≥800 unique source
  PNG/PPM references, mixed photo/screen/line/synthetic.
- **Pre-compressed inputs.** Each source compressed through `cwebp -q <q>`
  and zenwebp at every quality step. ~800 sources × 41 q × 2 encoders =
  ~65,000 input WebPs. Stored in block storage at
  `/mnt/v/zen/zenwebp-recompress/corpus/<sha256>.webp` with a manifest parquet
  at the root.
- **Per-cell recompression.** For each input WebP and each of {21 target
  zensim-A levels × 5 strategies}, produce a recompressed output. Measure:
  - `zensim_a_vs_source` (generation-loss signal)
  - `zensim_a_vs_reference` (cumulative — the product target)
  - `butter_pnorm3_vs_reference`
  - `output_size_ratio = output_len / input_len`
- Per-strategy regressions fit `zensim_a_vs_reference` as a function of
  `(source_q, target_q, encoder_family, content_class)`. Reported with
  bootstrap 95% CI over content sample.

## AQ block-decision experiment (CoeffEdit strategy)

Given an input lossy WebP at source quality `q_in`, a target zensim-A `T`, and
a calibrated per-segment quant scale `s = quant_scale(q_in, T)`:

- For each 4×4 luma block, classify activity (gradient magnitude, DC vs AC
  energy ratio). Use zenanalyze tier-2 features by default.
- Decide per block: `Preserve` (no change), `Tighten` (scale all AC by `s`),
  `ZeroBias` (zero AC > k threshold), `FullZero` (zero all AC).
- Train a small per-class lookup table (initial: hand-tuned thresholds; later:
  MLP) that picks the cheapest action keeping the per-block zensim-A above
  `T - δ_block_budget`.

The MLP becomes a final-stage refinement when the corpus is large enough.
Until then the table is hand-coded from a thresholded scan; **rigorous fit
only ships when bootstrap CI on the validation holdout is tighter than the
linear baseline.**

This experiment is gated behind `--features expert,analyzer`. The default
router does *not* call into AQ; it uses `coeff_edit_simple` which only
re-quantizes per-segment, no per-block work.

## Decision-router train/test split

- **Train.** `codec-corpus::sc` reencoded variants (excluding the cid22 50
  reference holdout that zensim training already uses).
- **Validation.** cid22 50 references, `alltheimages::small` mixed batch, and
  a 200-image LineArt/Screen mix from `gen-clothing` + `commerce-corpus`.
- **Test.** Held-out 100 references never seen by either train or val.

Train and val never share encoders × source-quality bins; bootstrap CI on the
held-out test set is the ship gate.

## Budget semantics

- `Budget::OneShot` — default. No measurements. Strategy is picked by table
  lookup and run exactly once. `projected_zensim_a` is the table's projection;
  `measured_zensim_a = None`.
- `Budget::MaxIterations(N)` — run the strategy, measure zensim-A vs source,
  and if the result undershoots target by more than `δ_iterate_band`, run a
  secant-step refinement with adjusted strategy params. Caps at `N`
  iterations. The "reference" for measurement is the source itself — the only
  signal we have. Iterations help when calibration is wrong about a particular
  content class, not when the *fundamental* target is unreachable.
- `Budget::MaxTime(d)` — same as MaxIterations(u32::MAX) but capped at `d`
  wall-clock.

One-shot is the recommended mode for production servers; multi-iteration is
for experimentation and offline batch optimization where ~5% extra zensim
accuracy is worth 3-5x CPU.

## Risks and known gaps

1. **VP8L coefficient editing is not yet defined.** Lossless WebP has a
   different transform stack (predictor + cross-color + huffman). `CoeffEdit`
   doesn't apply; the relevant lossless-only optimization is `LosslessRemux`
   (chunk normalization) or `LosslessReencode` (decode-then-encode with
   different transforms / better huffman tables).
2. **Animated WebP is not in scope for 0.1.x.** Animations are passed through
   to `LosslessRemux` (drop unused chunks; keep frames untouched).
3. **Alpha channel quality is hard to measure.** zensim Profile A doesn't
   currently score alpha; we score the RGB compositing against a deterministic
   noise floor (matches zenwebp's `target_zensim` design).
4. **Encoder identification is heuristic.** We compare the quantizer matrix
   shape, partition counts, and loop filter level against fingerprints for
   libwebp / zenwebp. False positives degrade to the libwebp table with a
   small risk margin.
5. **Calibration is a snapshot.** When zenwebp's encoder changes RD path, the
   table must be re-fit. Versioned tables ship with a CRC; loader rejects
   stale tables.

## Status (2026-05-28)

**Shipped and working:**
- `LosslessRemux`, `Reencode`, `LosslessReencode` strategies. All three are
  real and selected by the router.
- `OneShot`, `MaxIterations`, `MaxTime` budgets. `OneShot` uses the
  calibration's chosen quality; the measured budgets run `minimize_size`.
- Decode-based effective-quality estimation (`src/estimate.rs`) — the
  reliable calibration key, since header detection is unreliable (see
  `docs/QUALITY_DETECTION.md`).
- Monotonic, measured calibration (`src/calibration/data.rs`) from a clean
  lossless-only sweep. Real photos recompress 13–20% at mid targets.
- Heuristic content classifier (`src/classify.rs`); gates VP8L.
- Ground-truth size guard — never ships a file that didn't shrink.
- JXL handoff hint, `plan()` preview, full `expert` API.

**Measured and de-selected:**
- `DeblockReencode` — FALSIFIED (net-negative on every tested source
  config; VP8 already deblocks in-loop). Filter kept as `expert::deblock_rgba`;
  router never selects it. See `benchmarks/deblock_experiment_2026-05-28.md`.
- `CoeffEdit` — BUILT then RD-FALSIFIED. A complete, pixel-exact VP8
  coefficient transcoder (`src/vp8x/`) with two edits (AC-drop, requantize);
  the verbatim/no-op path is bit-exact (MAD 0). But every size-reducing edit
  is RD-dominated by `Reencode` at matched size because VP8 intra-prediction
  makes coefficient edits drift across the frame. Reachable via
  `expert::run_coeff_edit{,_keep,_requant}`; router never selects it. See
  `benchmarks/coeff_edit_experiment_2026-05-28.md`.

**Deferred (documented, not half-built):**
- Larger calibration corpus — current tables are fit from 3 photo refs
  (preliminary, below the 50-per-class bar). Re-fit recipe in
  `docs/QUALITY_DETECTION.md`. The size guard keeps the system correct
  regardless of fit precision.
- zenanalyze MLP content classifier (`--features analyzer`) — a final
  optimization step once the corpus is large enough; the heuristic
  classifier covers the common cases today.
- zenwebp encoder-family calibration (currently libwebp-family only).

## File layout

This crate lives inside the `zenwebp` repo as a **self-contained nested
workspace** (`zenwebp/zenwebp-recompress/`). zenwebp itself is deliberately
not a Cargo workspace — the sibling path deps below (`../../zensim`, …) would
break zenwebp's CI, whose runners don't check out the siblings. Build from
this directory.

```
zenwebp-recompress/                # nested workspace root (this dir)
├── Cargo.toml                      # [workspace] + library [package]
├── README.md
├── DESIGN.md                       # this file
├── CHANGELOG.md
├── src/
│   ├── lib.rs                      # re-exports the frozen API + expert mod
│   ├── api.rs                      # frozen public types (recompress, plan)
│   ├── error.rs
│   ├── source.rs                   # SourceAnalysis: probe + refine_content_class
│   ├── classify.rs                 # heuristic Photo/Screen/LineArt/Mixed
│   ├── budget.rs
│   ├── target.rs                   # target_zensim_a → libwebp q anchors
│   ├── router.rs                   # decide_strategy + dispatch + secant iter
│   ├── measure.rs                  # zensim Profile A score helpers
│   ├── aq.rs                       # CoeffEdit AQ mask (placeholder)
│   ├── bin/zwr.rs                  # demo CLI (--plan / --analyze / --iterations)
│   ├── strategies/
│   │   ├── mod.rs
│   │   ├── lossless_remux.rs
│   │   ├── coeff_edit.rs           # stub (router skips)
│   │   ├── reencode.rs             # run_reencode + run_reencode_at_q
│   │   ├── deblock.rs              # artifact-aware filter (expert building block)
│   │   ├── deblock_reencode.rs     # falsified strategy (router skips)
│   │   └── lossless_reencode.rs
│   └── calibration/
│       ├── mod.rs                  # CalibrationLookup (qi × target × strategy)
│       └── data.rs                 # empirical constants from the paired sweep
├── zwr-calibrate/                  # corpus-sweep binary (workspace member)
│   ├── Cargo.toml
│   └── src/main.rs
├── tests/
│   ├── locked_api.rs
│   └── recompress_smoke.rs
├── benchmarks/                     # meta docs + small CSVs (large → /mnt/v)
├── scripts/                        # build_paired_corpus.sh
└── docs/                           # CALIBRATION_NOTES.md
```

## References

- zenjpeg-recompress DESIGN.md — sibling design at
  `/home/lilith/work/zen/zenjpeg-recompress/DESIGN.md`. Direct ancestor.
- zenwebp `src/detect.rs` — already extracts source quality, quantizer index,
  filter level, sharpness, segment usage. We extend it; no parallel parser.
- zenwebp `src/encoder/zensim_target.rs` — closed-loop target-zensim encoder.
  Used directly under `Budget::MaxIterations`.
- zensim Profile A — `~/work/zen/zensim/zensim/src/profiles.rs` —
  `PreviewV0_5Balanced` is the default scoring trail for cross-codec
  comparison.
