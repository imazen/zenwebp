# File Size Bloat Investigation

**Date**: 2026-01-24
**Status**: In Progress
**Current Gap**: ~4.5% larger than libwebp (76.5KB vs 73KB on kodak/1.png at Q75)

## Executive Summary

Investigation into why zenwebp produces larger WebP files than libwebp. Started at 6.6% larger, now reduced to ~4.5% after fixing a TokenType enum bug.

## Fixes Applied

### 1. TokenType Enum Values Swapped (FIXED)

**Problem**: The `TokenType` enum in `src/encoder/cost.rs` had I16DC and I16AC values swapped compared to libwebp.

```rust
// BEFORE (wrong):
pub enum TokenType {
    I16DC = 0,  // Wrong! Should be 1
    I16AC = 1,  // Wrong! Should be 0
    Chroma = 2,
    I4 = 3,
}

// AFTER (correct):
pub enum TokenType {
    I16AC = 0,  // Matches libwebp TYPE_I16_AC
    I16DC = 1,  // Matches libwebp TYPE_I16_DC
    Chroma = 2,
    I4 = 3,
}
```

**libwebp reference** (`src/enc/cost_enc.h:28`):
```c
enum { TYPE_I16_AC = 0, TYPE_I16_DC = 1, TYPE_CHROMA_A = 2, TYPE_I4_AC = 3 };
```

**Impact**: This caused incorrect probability tables to be used for coefficient cost estimation. The I16 DC coefficients (Y2/WHT block) were using AC probability tables and vice versa.

**Result**: ~2% file size reduction (78KB → 76.5KB)

## Verified Correct Components

### Tables (all match libwebp exactly)

| Table | File | Status |
|-------|------|--------|
| `VP8_LEVEL_FIXED_COSTS` | `src/encoder/tables.rs` | ✅ Matches |
| `VP8_FREQ_SHARPENING` | `src/encoder/tables.rs` | ✅ Matches |
| `VP8_WEIGHT_TRELLIS` | `src/encoder/tables.rs` | ✅ Matches |
| `VP8_LEVEL_CODES` | `src/encoder/tables.rs` | ✅ Matches |

### Lambda Calculations (match libwebp formulas)

From `src/common/types.rs`, verified against `quant_enc.c`:

| Lambda | Formula | Status |
|--------|---------|--------|
| `lambda_i4` | `(3 * q * q) / 100` | ✅ Matches |
| `lambda_i16` | `(3 * q * q) / 100` | ✅ Matches |
| `lambda_mode` | `(1 * q * q) / 100` | ✅ Matches |
| `lambda_trellis_i4` | `(7 * q * q) / 100` | ✅ Matches |
| `lambda_trellis_i16` | `(5 * q * q) / 100` | ✅ Matches |

### RD Score Formula

```rust
// Our formula (cost.rs):
score = (R + H) * lambda + RD_DISTO_MULT * (D + SD)

// libwebp formula (quant_enc.c):
// Same structure - rate * lambda + distortion * RD_DISTO_MULT
```

### Sign Bit Cost Handling

Investigated whether sign bit was being double-counted. Attempted removal, which made files LARGER (78.5KB vs 76.5KB). Current approach is correct:

- `VP8_LEVEL_FIXED_COSTS[1] = 256` (1 bit for sign)
- We add `sign_cost = 256` for non-zero levels
- This is intentional for proper RD comparison during trellis optimization

## Failed Experiments

### 1. Removing Sign Bit Cost (REVERTED)

Attempted to remove sign bit cost from `level_cost_fast` and `level_cost_with_table`. Result was worse file sizes, so this was reverted.

## Remaining Investigation Areas

### High Priority

1. **Mode Selection Heuristics**
   - How do we decide between I16 and I4 modes?
   - What are the thresholds compared to libwebp?
   - File: `src/encoder/vp8.rs`, `choose_macroblock_info` function

2. **Probability Update Logic**
   - How do we decide when to update probability tables?
   - libwebp's `FinalizeTokenProbas` in `frame_enc.c:160-184`
   - Our `should_update` function
   - Threshold differences could affect compression

3. **Two-Pass Encoding**
   - Do we properly collect statistics in pass 1?
   - Do we properly apply them in pass 2?
   - libwebp: `VP8RecordCoeffs` → `FinalizeTokenProbas`

### Medium Priority

4. **Coefficient Encoding Order**
   - Zigzag scan order verification
   - Block processing order

5. **Quantization Matrix Differences**
   - Are Y2 (DC) quantization parameters identical?
   - UV quantization handling

6. **Residual Cost Estimation**
   - `get_residual_cost` is 2.4x slower than libwebp
   - May also produce different cost estimates

### Low Priority

7. **Segment Parameter Differences**
   - We use single segment (segment 0)
   - Quantization offsets per segment

## Benchmarks

**Test Image**: kodak/1.png (768x512)
**Quality**: 75
**Method**: 4 (with trellis)

| Encoder | File Size | Ratio |
|---------|-----------|-------|
| libwebp | 73KB | 1.0x |
| zenwebp (before fix) | 78KB | 1.068x |
| zenwebp (after TokenType fix) | 76.5KB | 1.048x |

## Code References

### Key Files

- `src/encoder/cost.rs` - Token types, trellis, cost calculation
- `src/encoder/vp8.rs` - Main encoder, mode selection
- `src/encoder/tables.rs` - Quantization and cost tables
- `src/common/types.rs` - Segment struct, lambda values

### libwebp Reference Files

- `src/enc/cost_enc.c` - Level cost calculation
- `src/enc/quant_enc.c` - Trellis quantization
- `src/enc/frame_enc.c` - Probability updates
- `src/dsp/cost.c` - Fixed cost tables

## Next Steps

1. **Compare probability update thresholds** - Read libwebp's `FinalizeTokenProbas` and compare with our `should_update`
2. **Compare mode selection RD scores** - Log and compare I16 vs I4 scores for specific macroblocks
3. **Verify coefficient encoding** - Dump actual encoded coefficients and compare bitstreams
4. **Profile trellis decisions** - Compare level choices made by trellis optimization

## Appendix: TokenType Usage Sites

After the fix, TokenType is used correctly:

1. `record_coeffs()` - Records coefficients with correct token type
2. `trellis_quantize_block()` - Uses ctype from token type for probability lookup
3. `get_residual_cost()` - Uses token type for cost estimation
4. `vp8.rs:1100` - Comment fixed to reflect correct values
