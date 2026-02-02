# zenwebp Context Handoff (2026-02-01)

Delete this file after loading into a new session.

## What Was Just Completed

### Phase 1-3 of Preset Tuning Plan (all committed on main)

1. **Phase 1** (commit 1f8b488): Wired up 4 tuning parameters (sns_strength, filter_strength, filter_sharpness, num_segments) from presets through EncoderParams to Vp8Encoder. Per-segment loop filter deltas now computed and written to bitstream.

2. **Phase 2** (commit 9ada505): Preset comparison tests and webpx matched-config benchmarks. Kodak aggregate: 1.014x of libwebp at Default preset.

3. **Phase 3** (commits 84fffae, d2425f7, ec75092):
   - `Preset::Auto` with uniformity-based content detection
   - Classifier: uniformity ≥ 0.45 → Photo tuning, < 0.45 → Default tuning, ≤128x128 → Icon
   - Drawing/Text presets map to Default tuning (50,60,0,4) — original values were counterproductive
   - SSIMULACRA2 quality comparison added to corpus tests
   - webpx updated to 0.1.3 (we fixed a preset bug in webpx)

### Key Findings

**Auto-detection results (threshold 0.45):**
- CID22 (41 diverse images): 0.987x of Default, 2 regressions <0.5% each
- Screenshots (9 images): 0.954x of Default, 0 regressions

**SSIMULACRA2 quality-size tradeoff:**
| Corpus | Encoder | Size ratio | SSIM2 delta | SSIM2/% |
|--------|---------|------------|-------------|---------|
| CID22 | zenwebp Photo | 0.985x | -0.57 | 0.38 |
| CID22 | libwebp Photo | 0.993x | -0.14 | 0.20 |
| Screenshots | zenwebp Photo | 0.954x | -0.72 | 0.16 |
| Screenshots | libwebp Photo | 0.959x | -1.19 | 0.29 |

zenwebp SNS is more aggressive than libwebp on diverse images, but gentler on screenshots.

## Commits (unpushed, 8 ahead of origin/main)

```
ec75092 fix: update webpx to 0.1.3 (preset bug fix)
d2425f7 test: add SSIMULACRA2 quality comparison to corpus tests
46cabf3 style: cargo fmt
84fffae fix: tune auto-detection classifier for reliable compression gains
373789a docs: update CLAUDE.md with preset tuning and auto-detection status
e4a0494 feat: add Preset::Auto with content-type detection
9ada505 test: add preset comparison tests and webpx matched-config benchmarks
1f8b488 feat: wire up preset tuning parameters (sns, filter, sharpness, segments)
```

## Key Files Modified

| File | What Changed |
|------|-------------|
| `src/encoder/api.rs` | EncoderParams tuning fields, preset mapping, builder overrides, Auto variant |
| `src/encoder/vp8.rs` | Stores params on Vp8Encoder, threads through encode path, writes filter deltas |
| `src/common/types.rs` | `init_matrices(sns_strength: u8)` parameter added |
| `src/encoder/analysis.rs` | Uniformity classifier, `content_type_to_tuning()`, `ClassifierDiag` |
| `tests/auto_detection_tuning.rs` | Corpus tests with SSIMULACRA2 and webpx comparison |
| `Cargo.toml` | webpx 0.1.3, webpx in dev-deps |

## Open Optimization Opportunities

### Priority 1: SNS Tradeoff Investigation
Our Photo preset produces more aggressive size savings than libwebp's but with proportionally higher quality loss on CID22. The segment quantization spread likely differs.

**To investigate:**
1. Compare per-segment quant values between zenwebp and libwebp for same image+config
2. Trace `SetSegmentAlphas()` in libwebp vs our `analyze_and_assign_segments()` in `src/encoder/vp8.rs`
3. Check if libwebp clamps segment quant deltas differently

### Priority 2: Encoder Speed (instruction count)
From callgrind profiling (method 4, non-trellis comparison vs libwebp method 4):

| Function | Our instructions | libwebp | Ratio |
|----------|-----------------|---------|-------|
| get_residual_cost | 306M | 126M | 2.4x |
| idct4x4 | 150M | 64M | 2.3x |
| t_transform | 134M | 52M | 2.6x |
| dct4x4 | 79M | 41M | 1.9x |
| trellis_quantize_block | ~90M | ~12M | 7.8x |
| encode_coefficients | ~60M | ~11M | 5.3x |

### Priority 3: Decoder Speed
Loop filter overhead: ~10M extra instructions vs libwebp (2.5x more despite SIMD).
Per-pixel threshold checks (`should_filter_*`) are the main bottleneck.

## Test Corpora

- `/mnt/v/work/codec-corpus/gb82-sc/` — 10 screenshots
- `/mnt/v/work/codec-corpus/CID22/CID22-512/validation/` — 41 diverse images
- `/mnt/v/work/codec-corpus/kodak/` — standard photo test set

Run corpus tests: `cargo test --release --features "_corpus_tests" -- --nocapture`

## Build/Test Commands

```bash
cargo test --release                                    # basic tests
cargo test --release --features "_corpus_tests" -- --nocapture  # corpus tests
cargo clippy --all-targets --features "std,fast-yuv,simd" -- -D warnings
cargo build --no-default-features                       # no_std check
cargo doc --no-deps
```
