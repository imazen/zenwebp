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

## SSIMULACRA2 Quality Comparison (2026-01-24)

**Quality sweep across Q10-Q100** (average of kodak 1-6):

| Q Range | Size vs libwebp | SSIM2 Δ | Interpretation |
|---------|-----------------|---------|----------------|
| 10-70 | 100-105% | **+0.2 to +1.1** | zenwebp better quality |
| **75** | 101% | **-0.18** | **Crossover point** |
| 80-95 | 97-102% | **-0.3 to -0.9** | libwebp better quality |
| 100 | 102% | -0.07 | Nearly equal |

**Key insight**: Crossover at Q75
- **Below Q75**: zenwebp achieves +0.3 to +1.0 better SSIM2 at similar file size
- **Above Q75**: libwebp achieves +0.3 to +0.9 better SSIM2 at similar file size

This suggests our lambda/RD tradeoff favors perceptual quality at lower bitrates
but is less efficient at high quality settings.

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

## Numerical Reverse Engineering (2026-01-24)

### Verified Components (All Match libwebp)

| Component | Status | Notes |
|-----------|--------|-------|
| Lambda formulas | ✅ Match | λ_i4, λ_i16, λ_mode, λ_trellis all verified |
| Quantization tables | ✅ Match | DC_QUANT, AC_QUANT, AC_QUANT2 (×1.55) |
| Trellis weights | ✅ Match | VP8_WEIGHT_TRELLIS (USE_TDISTO=1) |
| Bias values | ✅ Match | Y1:(96,110), Y2:(96,108), UV:(110,115) |
| RD score formula | ✅ Match | rate×λ + RD_DISTO_MULT×(D+SD) |
| q_i16 calculation | ✅ Match | (y2_dc + 15×y2_ac + 8) >> 4 |

### Quality Crossover Analysis

The SSIM2/filesize crossover at Q75 is unexpected given all parameters match.

**At Q90 (high quality):**
- File size: 1.7% smaller than libwebp
- SSIM2: 0.82 worse than libwebp
- Interpretation: We're over-quantizing (trading quality for size)

**Possible root causes (not yet verified):**
1. Mode selection threshold differences
2. Coefficient cost estimation bias
3. Trellis node exploration differences
4. Probability table update timing

### Key Quantization Values

| Q | q_idx | y_dc | y_ac | λ_i4 | λ_i16 | λ_mode |
|---|-------|------|------|------|-------|--------|
| 75 | 59 | 54 | 68 | 81 | 10,443 | 27 |
| 90 | 28 | 25 | 32 | 18 | 2,352 | 6 |
| 95 | 15 | 17 | 19 | 5 | 675 | 1 |

At high quality (low λ), distortion dominates the RD score.
Our "smaller files, worse quality" pattern suggests rate bias.

## Next Steps

1. **Trace mode selection** - Log I4 vs I16 score comparisons to see if threshold differs
2. **Compare coefficient counts** - Count non-zeros to identify quantization differences
3. **Dump probability tables** - Compare token probabilities at Q90
4. **Profile trellis paths** - Log level choices to identify divergence

## Appendix: TokenType Usage Sites

After the fix, TokenType is used correctly:

1. `record_coeffs()` - Records coefficients with correct token type
2. `trellis_quantize_block()` - Uses ctype from token type for probability lookup
3. `get_residual_cost()` - Uses token type for cost estimation
4. `vp8.rs:1100` - Comment fixed to reflect correct values
