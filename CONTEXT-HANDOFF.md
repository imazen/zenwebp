# Context Handoff — zenwebp

**Date:** 2026-02-03
**Commit:** `a67f70f` on main (148 ahead of origin, not pushed)
**Working tree:** clean
**Tests:** 161 pass, clippy clean

## Session Summary

Implemented Zopfli-style cost interval manager in `src/encoder/vp8l/cost_model.rs`
(commit `f94a127`). Replaced O(n×4095) TraceBackwards DP with libwebp's interval-based
CostManager. Index-based doubly-linked list, no unsafe. Compression unchanged (0.997x
CID22 full corpus). All 250 images validate.

## Where zenwebp Stands

**VP8L lossless:** Feature-complete, beats libwebp (0.997x on 250-image CID22 corpus).
**VP8 lossy:** Near parity (1.015x Q75, 1.006x Q90 production).
**Decoder:** 1.4x slower than libwebp C.

## Actionable Next Steps (ordered by impact)

### 1. Fix the 3616956 outlier (1.105x — 29KB larger than libwebp)

This single image accounts for ~40% of our total aggregate byte deficit. libwebp
uses cache=9 for this image. Our try-both strategy picks the wrong cache size.

**Diagnosis:** Our entropy-based cache selection overestimates the value of large
cache. The global histogram says cache=8-10 is best, but per-tile meta-Huffman
codes pay extra alphabet overhead that isn't captured by global entropy estimation.

**Concrete fix to try:** In `backward_refs.rs:calculate_best_cache_size`, after
picking the best cache_bits via entropy, also try cache_bits-2 and cache_bits-4
(libwebp's `BackwardRefsWithLocalCache` does something similar). The try-both
strategy in `get_backward_references` currently only compares auto vs 0; expanding
to test 2-3 candidates would catch this. Cost: one extra encode per candidate,
negligible for method 4.

**Files:** `src/encoder/vp8l/backward_refs.rs:702-746` (cache selection + try-both)

### 2. VP8L encoding speed measurement

The Zopfli interval manager should have improved encoding speed significantly for
large images (O(n×20) vs O(n×4095)), but we never measured it. Worth benchmarking
to quantify for README/docs.

**How:** Add `std::time::Instant` timing to `lossless_benchmark.rs` around the
zenwebp encode call. Compare before (revert to simple DP) and after on a few
512×512 images. The improvement should be most visible on images with many long
matches (photographic content).

### 3. Publish 0.3.0

The VP8L encoder is a major feature addition since 0.2.1. Before publishing:
- Verify README accuracy (lossless encoding API examples)
- Run `cargo doc` and check public API surface
- Update version in Cargo.toml
- `cargo publish --dry-run`
- 148 unpushed commits need `git push`

### 4. VP8 lossy SIMD hot paths (bigger project)

The lossy encoder's remaining gap is in hot inner loops without SIMD:

| Function | vs libwebp | Savings potential |
|----------|-----------|------------------|
| `get_residual_cost` (cost.rs) | 2.4x slower | ~3M instructions/image |
| `trellis_quantize_block` (cost.rs) | 7.8x slower | ~2M instructions/image |
| `encode_coefficients` (residuals.rs) | 5.3x slower | ~1M instructions/image |

Use `archmage` + `wide` crate pattern (see CLAUDE.md SIMD section). Start with
`get_residual_cost` — it's the hottest and most straightforward to vectorize
(array operations on coefficient levels and context tracking).

## Key Paths

```
CID22 images:     /tmp/CID22/original/
libwebp cwebp:    /home/lilith/work/libwebp/examples/cwebp
libwebp source:   ~/work/libwebp/
Output dir:       /mnt/v/output/zenwebp/
```

```bash
cargo run --release --example lossless_benchmark -- /tmp/CID22/original 50
cargo run --release --example corpus_test -- /tmp/CID22/original
```

## Worst VP8L Outliers (CID22 Q75 M4)

| Image | zenwebp | libwebp | Ratio | libwebp cache | Issue |
|-------|---------|---------|-------|---------------|-------|
| 3616956 | 305904 | 276926 | 1.105x | 9 | Cache overestimation |
| 2079234 | 289998 | 283252 | 1.024x | 0 | LZ77 ref quality |
| 3779187 | 442914 | 437040 | 1.013x | 0 | LZ77 ref quality |
| 3330118 | 208612 | 205892 | 1.013x | 10 | Cache overestimation |

Read CLAUDE.md for detailed investigation notes, profiler data, and all diagnostic examples.
