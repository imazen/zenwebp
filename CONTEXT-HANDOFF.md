# Context Handoff: Encoder SIMD Optimization

## Completed This Session

### SIMD Infrastructure Updates ✓
- Updated `multiversed` 0.1.0 → 0.2.0
- Updated `safe_unaligned_simd` 0.2 → 0.2.3
- Added `magetypes` 0.1.0 (token-gated SIMD types, replaces `wide` when tokens available)
- Changed all code to use `core::arch` instead of `std::arch`
- Replaced `Sse41Token` with `X64V3Token` throughout (AVX2+FMA baseline enables more compiler optimizations)

Commit: `0d23cb2` - refactor: update SIMD deps and use higher-level tokens

### Directory Reorganization (Previous Session) ✓
Restructured flat `src/` into modular layout:
- `src/decoder/` - VP8 decoder, lossless, loop filter, YUV
- `src/encoder/` - VP8 encoder, cost, analysis, tables
- `src/common/` - shared types, prediction, transforms, SIMD

Commit: `1740c51` - refactor: reorganize codebase into decoder/, encoder/, common/

## Callgrind Analysis (2026-01-23)

**Test: kodim07.png (768x512), Q75, method 4, 10 iterations**

### Instruction Counts
| Encoder | Total Instructions | Per Encode | Ratio |
|---------|-------------------|------------|-------|
| Ours | 9,662M | 966M | 2.78x more |
| libwebp | 3,470M | 347M | baseline |

### Top Hotspots Comparison

**Ours (per encode basis):**
| Function | Instructions | % |
|----------|-------------|---|
| choose_macroblock_info | 243M | 25.2% |
| get_residual_cost | 102M | 10.5% |
| idct4x4 | 76M | 7.8% |
| t_transform_sse2 | 74M | 7.6% |
| trellis_quantize_block | 68M | 7.0% |
| dct4x4 | 37M | 3.8% |
| forward_dct_4x4 (analysis) | 30M | 3.2% |
| add_residue | 27M | 2.8% |

**libwebp (per encode basis):**
| Function | Instructions | % |
|----------|-------------|---|
| ITransform_SSE2 | 35M | 10.0% |
| GetResidualCost_SSE2 | 34M | 9.8% |
| PickBestIntra4 | 30M | 8.6% |
| Disto4x4_SSE2 | 28M | 8.1% |
| FTransform_SSE2 | 22M | 6.3% |
| QuantizeBlock_SSE2 | 21M | 6.0% |

### Disparity Analysis

| Our Function | Ours | libwebp Equiv | libwebp | Ratio |
|--------------|------|---------------|---------|-------|
| get_residual_cost | 102M | GetResidualCost_SSE2 | 34M | **3.0x** |
| trellis_quantize_block | 68M | QuantizeBlock_SSE2 | 21M | **3.2x** |
| idct4x4 | 76M | ITransform_SSE2 | 35M | **2.2x** |
| t_transform | 74M | Disto4x4_SSE2 | 28M | **2.6x** |
| dct4x4 | 37M | FTransform_SSE2 | 22M | **1.7x** |

### Cachegrind Results
| Metric | Ours | libwebp |
|--------|------|---------|
| Instructions | 9.69B | 3.47B |
| Data refs | 2.86B | 1.03B |
| D1 miss rate | 0.1% | 0.2% |
| I1 miss rate | 0.28% | 0.04% |

**Key insight:** Cache behavior is similar/good. The 2.8x slowdown is pure instruction count.

## Priority SIMD Targets

Based on instruction savings potential:

1. **get_residual_cost** - 68M potential savings (3.0x gap)
   - libwebp uses `GetResidualCost_SSE2` with SIMD coefficient scanning
   - Our scalar version iterates through coefficients one-by-one
   - Location: `src/encoder/cost.rs:1311-1445`

2. **trellis_quantize_block** - 47M potential savings (3.2x gap)
   - libwebp uses `QuantizeBlock_SSE2` for SIMD quantization
   - Our version does per-coefficient trellis optimization
   - Location: `src/encoder/cost.rs:534-800`

3. **idct4x4** - 41M potential savings (2.2x gap)
   - We have SIMD but libwebp's `ITransform_SSE2` is faster
   - Consider porting libwebp's exact implementation
   - Location: `src/common/transform_simd_intrinsics.rs`

4. **t_transform** (distortion) - 46M potential savings (2.6x gap)
   - We have SIMD in `simd_sse.rs` but libwebp's Disto4x4_SSE2 is faster
   - Location: `src/common/simd_sse.rs`

5. **dct4x4** - 15M potential savings (1.7x gap)
   - Already have SIMD, but libwebp's FTransform_SSE2 more optimized
   - Location: `src/common/transform_simd_intrinsics.rs`

## SIMD Patterns (Updated)

### Token Usage
- Use `X64V3Token` (AVX2+FMA) instead of `Sse41Token` for better optimization
- Cache tokens when possible - pass through call chains
- Same token type parameter in nested `#[arcane]` calls offers performance benefits
- OK to summon from within `#[multiversed]` functions

### Crate Versions (2026-01)
```toml
archmage = "0.2.1"
multiversed = "0.2.0"
magetypes = "0.1.0"  # token-gated SIMD types (new 'wide' with tokens)
safe_unaligned_simd = "0.2.3"
```

### Pattern
```rust
use archmage::{arcane, X64V3Token, SimdToken};
use core::arch::x86_64::*;

#[multiversed::multiversed("x86-64-v4", "x86-64-v3", "x86-64-v2")]
pub fn my_simd_func(data: &[u8]) -> u32 {
    if let Some(token) = X64V3Token::summon() {
        my_simd_inner(token, data)
    } else {
        my_scalar_fallback(data)
    }
}

#[arcane]
fn my_simd_inner(token: X64V3Token, data: &[u8]) -> u32 {
    // SIMD code here - token enables intrinsics
}
```

## File Locations After Refactor

| Purpose | New Path |
|---------|----------|
| VP8 encoder | src/encoder/vp8.rs |
| Cost/trellis | src/encoder/cost.rs |
| Analysis | src/encoder/analysis.rs |
| Tables | src/encoder/tables.rs |
| Transforms | src/common/transform*.rs |
| SIMD SSE | src/common/simd_sse.rs |
| VP8 decoder | src/decoder/vp8.rs |
| Loop filter | src/decoder/loop_filter*.rs |
| YUV SIMD | src/decoder/yuv_simd.rs |

## libwebp Reference Files

For SIMD implementations, reference:
- `~/work/libwebp/src/dsp/enc_sse2.c` - FTransform, ITransform, quantization
- `~/work/libwebp/src/dsp/cost_sse2.c` - GetResidualCost, SetResidualCoeffs
- `~/work/libwebp/src/enc/quant_enc.c` - Trellis quantization logic

## Next Steps

1. Port `GetResidualCost_SSE2` to Rust with archmage
2. Add SIMD quantization path (bypass trellis for speed mode)
3. Optimize IDCT to match libwebp's ITransform_SSE2
4. Profile after each change to verify improvements
