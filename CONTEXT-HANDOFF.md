# zenwebp Context Handoff (2026-02-01)

Delete this file after loading into a new session.

## What Was Just Completed

### Quality/Size Improvement Plan - All 6 Phases Complete

1. **Phase 1: UV Quant Deltas** (commit 556f4ab)
   - Modified `analyze_image()` to return `AnalysisResult` with `uv_alpha_avg`
   - Added UV quant delta calculation matching libwebp's formula
   - Applied `dq_uv_dc` and `dq_uv_ac` to segment quantization tables

2. **Phase 2: Beta and Filter Modulation** (commit 1049cb8)
   - Added `compute_filter_level_with_beta()` to `src/encoder/cost.rs`
   - Per-segment beta: `beta = 255 * (center - min) / (max - min)`
   - Filter modulation: `f = base_strength * level0 / (256 + beta)`

3. **Phase 3: Segment Map Smoothing** (commit 7c442d8)
   - Added `smooth_segment_map()` with 3x3 majority filter (threshold 5)
   - Called after segment assignment when num_segments > 1

4. **Phase 4: Simplify Segments** (commit 650f3b5)
   - Added `simplify_segments()` to merge segments with identical quant/filter
   - Reduces effective segment count to save bits

5. **Phase 5: Butteraugli Corpus Tests** (commit d3d8aa5)
   - Added `butteraugli = "0.4.0"` to dev-dependencies
   - Updated `tests/auto_detection_tuning.rs` with both SSIM2 and butteraugli

6. **Phase 6: Classifier Analysis** (commits e7c0031, 3a4d7b1)
   - Concluded: classifier correctly detects Photo content
   - Quality gap is NOT in content detection - it's in I4 mode path

## Critical Finding: I4 Mode Path Issue

**The file size gap is in I4 mode, not I16:**

| Image | Method | zenwebp | libwebp | Ratio |
|-------|--------|---------|---------|-------|
| 792079 | 0 (I16 only) | 13294 | 14660 | **0.91x** |
| 792079 | 2 (I4, no trellis) | 12470 | 11318 | 1.10x |
| 792079 | 4 (I4 + trellis) | 12188 | 10962 | 1.11x |
| terminal | 0 (I16 only) | 77490 | 77510 | **1.00x** |
| terminal | 2 (I4, no trellis) | 70092 | 64098 | 1.09x |
| terminal | 4 (I4 + trellis) | 68888 | 61612 | 1.12x |

**Key insight:**
- I16 only (method 0): We're **SMALLER** than libwebp (0.91-1.00x)
- With I4 (methods 2-6): We're **LARGER** (1.09-1.14x)
- libwebp's I4 reduces size ~10% from I16; ours only ~2%

## Failed Experiment

Tried adding sharpening and zthresh to `quantize_coeff()`:
```rust
let coeff_with_sharpen = abs_coeff + u32::from(self.sharpen[pos]);
if coeff_with_sharpen <= self.zthresh[pos] { return 0; }
```
**Result: Made things WORSE.** Reverted. The issue is not in simple quantization.

## Commits (16 ahead of origin/main)

```
3a4d7b1 docs: add I4 mode path investigation findings
e7c0031 docs: update investigation notes with butteraugli corpus results
650f3b5 feat: add segment simplification (merge identical segments)
7c442d8 feat: add segment map smoothing with 3x3 majority filter
1049cb8 feat: add beta-based filter strength modulation per segment
d3d8aa5 test: add butteraugli metrics to corpus tests
556f4ab feat: add UV quant deltas based on uv_alpha analysis
abb116b docs: add context handoff and update CLAUDE.md with SSIMULACRA2 findings
ec75092 fix: update webpx to 0.1.3 (preset bug fix)
... (8 more from previous session)
```

## Next Steps: I4 Mode Investigation

1. **Compare I4 usage frequency** - Count how many MBs use I4 vs I16 in both encoders
2. **Trace I4 RD score calculation** - Verify `pick_best_intra4()` matches libwebp's `PickBestIntra4()`
3. **Check coefficient cost for I4** - `get_cost_luma4()` vs `VP8GetCostLuma4()`
4. **Compare bit emission** - Same coefficients should produce same bits

## Key Files

| File | Purpose |
|------|---------|
| `src/encoder/vp8.rs:1826-2070` | `pick_best_intra4()` - I4 mode selection |
| `src/encoder/vp8.rs:1539-1720` | `pick_best_intra16()` - I16 mode selection |
| `src/encoder/vp8.rs:2237-2275` | `choose_macroblock_info()` - I4 vs I16 decision |
| `src/encoder/vp8.rs:2368-2520` | `analyze_and_assign_segments()` - segment setup |
| `src/encoder/cost.rs:1968-1996` | `get_cost_luma4()` - I4 coefficient cost |
| `src/encoder/cost.rs:1731-1795` | `get_residual_cost_scalar()` - coefficient cost core |
| `src/encoder/analysis.rs` | `AnalysisResult`, `smooth_segment_map()`, classifier |
| `tests/auto_detection_tuning.rs` | Corpus tests with SSIM2 + butteraugli |

## libwebp Reference Files

| File | Purpose |
|------|---------|
| `~/work/libwebp/src/enc/quant_enc.c:1128-1220` | `PickBestIntra4()` |
| `~/work/libwebp/src/enc/quant_enc.c:1058-1106` | `PickBestIntra16()` |
| `~/work/libwebp/src/enc/quant_enc.c:681-705` | `QuantizeBlock_C()` - with sharpening |
| `~/work/libwebp/src/dsp/cost.c:238-287` | `GetResidualCost_C()` |

## Test Corpora

- `/mnt/v/work/codec-corpus/gb82-sc/` — 9 screenshots
- `/mnt/v/work/codec-corpus/CID22/CID22-512/validation/` — 41 diverse images
- `/mnt/v/work/codec-corpus/kodak/` — standard photo test set (files: 1.png - 24.png)

## Test Commands

```bash
# Corpus tests with quality metrics
cargo test --release --features _corpus_tests auto_detection_cid22 -- --nocapture
cargo test --release --features _corpus_tests auto_detection_screenshots -- --nocapture

# All unit tests
cargo test --release

# Clippy
cargo clippy --all-targets --all-features -- -D warnings
```
