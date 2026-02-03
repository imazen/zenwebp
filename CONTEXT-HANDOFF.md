# Context Handoff: I4 Encoding Efficiency Investigation (2026-02-02)

## Summary

Investigating why zenwebp produces files ~3% larger than libwebp at method 6 (1.031x).
Root cause is MODE SELECTION, not entropy coding.

## Key Discovery

**Bits per coefficient is IDENTICAL (1.000x at m4)** - entropy coding is fine.
The gap comes from **more nonzero coefficients** due to different I4 vs I16 mode decisions.

### Mode Selection Difference

| Encoder | I16 | I4 | File Size |
|---------|-----|-----|-----------|
| zenwebp m4 | 708 | 316 | 12,174 bytes |
| libwebp m4 | 734 | 290 | 12,018 bytes |

We pick **26 more I4 macroblocks** than libwebp. This accounts for ~215 extra nonzero coefficients.

### Analysis of Disputed Blocks

For the 56 blocks where zenwebp picks I4 but libwebp picks I16:
- Our I4 produces 2.0 nonzero coefficients per MB (average)
- Their I16 produces 1.2 nonzero coefficients per MB
- Result: picking I4 when I16 would be better

## Files Modified This Session

1. `src/encoder/vp8/mode_selection.rs` - Fixed potential u16 overflow in I4 RD scoring
   - Changed `rd_score(sse, total_rate as u16, lambda)` to `rd_score_with_coeffs(sse, mode_cost, coeff_cost, lambda)`
   - The overflow wasn't actually occurring in tests, but fix is correct for safety

## What Was Tried (And Failed)

### Adding Trellis to I16 Mode Selection

Hypothesis: Mismatch between simple quantization in I16 mode selection vs trellis in final encoding.

**Result: Made files LARGER (+18 to +152 bytes)**

Reason: libwebp does NOT use trellis for Y2 (DC) coefficients, even at high methods. Our final encoding also doesn't use trellis for Y2. Adding trellis to Y1 during mode selection created a mismatch.

Code added and reverted:
- Added trellis for Y2 during I16 mode selection - WRONG (Y2 never uses trellis)
- Added trellis for Y1 during I16 mode selection - Made things worse

## Remaining Gap Analysis

The ~3% gap at m6 (1.031x) vs libwebp is due to:

1. **Mode selection threshold** - We favor I4 over I16 more than libwebp
2. **I4 produces more coefficients** when we pick it over I16 incorrectly

## Next Steps To Investigate

### 1. Compare I4 vs I16 RD Score Computation

The I4 vs I16 comparison happens in `choose_macroblock_info()`:
```rust
match self.pick_best_intra4(mbx, mby, i16_score) {
    Some((modes, _)) => (LumaMode::B, Some(modes)),  // I4 wins
    None => (luma_mode, None),  // I16 wins
}
```

Key questions:
- Why does I4 appear to have a lower RD score when it's actually worse?
- Is the BMODE_COST (211) penalty correct?
- Are the I4 sub-block mode costs (VP8_FIXED_COSTS_I4) correct?

### 2. Instrument Mode Selection

Add debug logging to a specific disputed macroblock (e.g., MB(14,1) where zen picks I4 but libwebp picks I16):
- Print I16 score components: mode_cost, coeff_cost, SSE, spectral_disto, lambda
- Print I4 running score at each sub-block
- Compare with libwebp's intermediate values

### 3. Check Lambda Values

The RD comparison uses `lambda_mode` for both I4 and I16. Verify:
- `lambda_mode` calculation matches libwebp's `dqm->lambda_mode`
- The formula `lambda_mode = (q² >> 7)` is correct

### 4. Check Coefficient Cost Estimation

The coefficient cost for I4 blocks comes from `get_cost_luma4()`. Verify:
- Cost tables match libwebp's
- Context tracking (nz_top, nz_left) is correct

## Diagnostic Test Files

These files are useful for investigation:
- `tests/i4_diagnostic_harness.rs` - Comprehensive comparison tests
- `examples/test_i4_vs_i16.rs` - I4 vs I16 decision analysis
- `examples/test_no_trellis.rs` - Method comparison

Run with:
```bash
cargo test --release --features _corpus_tests --test i4_diagnostic_harness -- --nocapture
cargo run --release --example test_i4_vs_i16
```

## Benchmark Results (792079.png, Q75, SNS=0, filter=0, segments=1)

| Method | zenwebp | libwebp | Ratio |
|--------|---------|---------|-------|
| 0 | 13,988 | 16,678 | **0.839x** |
| 2 | 12,616 | 13,440 | **0.939x** |
| 4 | 12,174 | 12,018 | 1.013x |
| 6 | 12,084 | 11,720 | 1.031x |

Key insight: At methods 0-2 (no I4 or limited I4), we're SMALLER than libwebp.
The gap opens up at methods 3+ when more I4 modes are enabled.

## Code Locations

- I4 mode selection: `src/encoder/vp8/mode_selection.rs:320-612` (`pick_best_intra4`)
- I16 mode selection: `src/encoder/vp8/mode_selection.rs:34-211` (`pick_best_intra16`)
- I4 vs I16 comparison: `src/encoder/vp8/mode_selection.rs:779-808` (`choose_macroblock_info`)
- Lambda calculations: `src/encoder/cost.rs:315-340` (`calc_lambda_*`)
- Fixed costs: `src/encoder/tables.rs:386-430` (FIXED_COSTS_I16, VP8_FIXED_COSTS_I4)

## Git State

Working tree has changes:
- `CLAUDE.md` - Updated with session notes
- `src/encoder/vp8/mode_selection.rs` - Safety fix for u16 overflow

Commit the safety fix before continuing investigation.
