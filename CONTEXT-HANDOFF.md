# Context Handoff: I4 Over-Selection Investigation

**Date:** 2026-02-03
**Session Focus:** Investigating why zenwebp selects I4 mode ~5% more often than libwebp

## Summary

zenwebp selects I4 (4x4 prediction blocks) for 73.4% of macroblocks vs libwebp's 68.3%
on the 792079.png test image (method 4, SNS=0, filter=0, segments=1). This results in
~1% larger files (12,398 vs 12,158 bytes).

Paradoxically, our I16-only encoding is 16-19% smaller than libwebp's! The issue is
specifically in the I4 path.

## Work Completed This Session

### 1. Added Spectral Distortion (TDisto) to I4 Mode Selection
**Commit:** dd06972 - "feat: add spectral distortion (TDisto) to I4 mode selection"

- Added `tdisto_4x4` import to mode_selection.rs
- Added `tlambda` parameter to I4 selection (for spectral distortion weighting)
- Compute spectral distortion for each I4 mode candidate:
  ```rust
  let spectral_disto = if tlambda > 0 {
      let td = tdisto_4x4(&src_block, &rec_block, 4, &VP8_WEIGHT_Y);
      (tlambda as i32 * td + 128) >> 8
  } else {
      0
  };
  ```
- Use `rd_score_full` instead of `rd_score_with_coeffs` to include SD in RD formula

**Result:** This change only affects encoding when SNS > 0 (since tlambda = SNS * q >> 5).
With SNS=0, tlambda=0 so spectral distortion has no effect. The I4 over-selection issue
persists.

### 2. Documentation Updates
**Commit:** 787b41e - "docs: add I4 over-selection investigation to CLAUDE.md"

Added detailed investigation notes to CLAUDE.md covering:
- Problem statement and metrics
- Method comparison showing I16-only vs I4-enabled behavior
- List of verified components that match libwebp
- Hypothesis about coefficient encoding efficiency

### 3. Added Diagnostic Examples
- `examples/test_sns.rs` - Compare I4 usage with/without SNS
- `examples/test_i16_only.rs` - Compare I16-only encoding
- `examples/test_bmode_cost.rs` - Track I4 vs libwebp mode decisions

## Key Findings

### Verified to Match libwebp:
- Lambda values: lambda_i4, lambda_i16, lambda_mode, tlambda
- BMODE_COST = 211 (I4 mode signaling penalty)
- VP8_FIXED_COSTS_I4 mode cost table
- VP8_LEVEL_FIXED_COSTS coefficient costs
- LevelCosts calculation from probability tables
- RD score formula: (R + H) * lambda + 256 * (D + SD)
- SSE calculation
- Token types (I16AC=0, I16DC=1, Chroma=2, I4=3)
- SIMD vs scalar produce identical output

### Root Cause Hypothesis:
The cost ESTIMATION is accurate, but actual I4 coefficient ENCODING produces more bits.

From diagnostic harness (same-mode I4 blocks compared to libwebp):
- Exact coefficient match: 57.9%
- Total |level| sum: zenwebp 2.7% higher
- Non-zero coefficient count: zenwebp 1.3% higher

This suggests our quantization/trellis produces slightly different coefficients that
encode less efficiently.

## Files Modified

- `src/encoder/vp8/mode_selection.rs` - Added TDisto to I4
- `CLAUDE.md` - Added investigation notes
- `examples/test_*.rs` - New diagnostic tools

## Next Steps to Investigate

1. **Compare trellis decisions for identical input blocks**
   - Feed same residual to both zenwebp and libwebp trellis
   - Compare output coefficient levels
   - Find where decisions diverge

2. **Check simple quantization (non-trellis path)**
   - Verify sharpen[] and zthresh[] arrays match libwebp
   - Compare quantize_coeff output for same input

3. **Verify probability tables during encoding**
   - Check if mode selection uses same tables as actual encoding
   - Look for any initialization differences

4. **Consider alternative approaches:**
   - Increase BMODE_COST to be more conservative about I4
   - Add additional penalty for I4 blocks with high coefficient counts
   - Compare libwebp's early-exit optimization in PickBestIntra4

## Test Commands

```bash
# Compare I4 usage
cargo run --release --features simd --example test_bmode_cost

# Compare I16-only performance
cargo run --release --features simd --example test_i16_only

# Compare method sizes
cargo run --release --features simd --example compare_all_methods

# Full corpus test
cargo run --release --features simd --example corpus_test /tmp/CID22/original
```

## Notes

- The user asked to "look at cost functions, consider simd and scalar math and fma
  affect numbers; make identical choices to libwebp"
- SIMD vs scalar has already been verified to produce identical output
- No FMA is used in cost calculations (all integer arithmetic)
- The difference is NOT in cost calculation but in actual coefficient production
