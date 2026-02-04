# SIMD Optimization Handoff

## Project Overview

zenwebp is a Rust WebP encoder/decoder. The codebase has been refactored to isolate SIMD-amenable functions into focused modules. This handoff documents the optimization opportunities.

## SIMD Infrastructure

The project uses these crates (under `simd` feature):
- `multiversed = "0.2.0"` - Compile-time multi-target dispatch
- `archmage = "0.4.0"` - Token-based safe intrinsics
- `magetypes = "0.4.0"` - Token-gated SIMD types
- `wide = "1.1.1"` - Portable SIMD (autovectorizes well inside `#[multiversed]`)

Existing SIMD code in `src/common/simd_sse.rs` shows the pattern:
```rust
#[cfg(feature = "simd")]
#[multiversed::multiversed("x86-64-v4", "x86-64-v3", "x86-64-v2")]
pub fn t_transform(input: &[u8], stride: usize, w: &[u16; 16]) -> i32 {
    if let Some(token) = X64V3Token::summon() {
        t_transform_sse2(token, input, stride, w)
    } else {
        t_transform_scalar(input, stride, w)
    }
}

#[arcane]
fn t_transform_sse2(_token: impl Has128BitSimd + Copy, ...) -> i32 {
    // SSE2 intrinsics here
}
```

## Priority SIMD Targets

### 1. `src/encoder/analysis/histogram.rs` - forward_dct_4x4

**Current**: Scalar 4x4 DCT butterfly (lines 25-69)
```rust
pub fn forward_dct_4x4(src: &[u8], pred: &[u8], src_stride: usize, pred_stride: usize) -> [i16; 16]
```

**Operations**:
- Load 4 rows of 4 bytes, widen to i16
- Horizontal butterfly: adds/subs on 4 values
- Vertical butterfly: adds/subs on columns
- Multiply-accumulate with constants 2217, 5352

**SIMD approach**: Process all 4 rows in parallel with 128-bit registers. Similar to existing `t_transform_sse2` pattern.

### 2. `src/encoder/analysis/prediction.rs` - TrueMotion prediction

**Current**: Scalar per-pixel computation (lines 102-129, 198-223)
```rust
pub fn pred_luma16_tm(dst: &mut [u8], left_with_corner: Option<&[u8]>, top: Option<&[u8]>)
pub fn pred_chroma8_tm(dst: &mut [u8], left_with_corner: Option<&[u8]>, top: Option<&[u8]>)
```

**Operations**:
```rust
// For each row y, column x:
dst[y * BPS + x] = (left[1+y] + top[x] - top_left).clamp(0, 255) as u8
```

**SIMD approach**:
- Broadcast `left[1+y]` and `top_left`
- Add vector of `top[0..16]`
- Saturating subtract and pack to u8
- Process 16 pixels per iteration (AVX2) or 8 (SSE)

### 3. `src/encoder/cost/distortion.rs` - is_flat_source_16

**Current**: Scalar comparison loop (lines 158-169)
```rust
pub fn is_flat_source_16(src: &[u8], stride: usize) -> bool {
    let v = src[0];
    for y in 0..16 {
        for x in 0..16 {
            if src[y * stride + x] != v { return false; }
        }
    }
    true
}
```

**SIMD approach**:
- Broadcast `v` to 16/32-byte vector
- Compare 16 bytes at a time
- Use `movemask` + check all ones

### 4. `src/encoder/analysis/classifier.rs` - compute_edge_density

**Current**: Scalar horizontal diff scan (lines 131-155)
```rust
let diff = row[x].abs_diff(row[x - 1]);
if diff > threshold { edge_count += 1; }
```

**SIMD approach**:
- Load 16 bytes, shift by 1, compute abs diff
- Compare > threshold, popcount

## Secondary Targets

### `src/common/prediction.rs` - add_residue
```rust
pub fn add_residue(pblock: &mut [u8], rblock: &[i32; 16], y0, x0, stride)
```
Adds i32 residuals to u8 prediction with clamping. Good for packed add with saturation.

### `src/encoder/residual_cost.rs` - get_residual_cost
Already has partial SIMD (lines 207-241). Could extend AVX2 path.

## Build & Test Commands

```bash
cd /home/lilith/work/zenwebp

# Build with SIMD
cargo build --features simd

# Test
cargo test --lib

# Clippy (use stable to avoid nightly ICE)
cargo +stable clippy --lib --all-features -- -D warnings

# Benchmark (if criterion benches exist)
cargo bench --features simd
```

## File Structure After Refactoring

```
src/encoder/
├── analysis/
│   ├── mod.rs          # Main analysis types, analyze_image()
│   ├── classifier.rs   # Content type detection (SIMD: edge_density)
│   ├── histogram.rs    # DCT histogram (SIMD: forward_dct_4x4)
│   ├── iterator.rs     # MB iteration
│   ├── prediction.rs   # Analysis predictions (SIMD: TM modes)
│   └── segment.rs      # K-means segmentation
├── cost/
│   ├── mod.rs          # RD score functions
│   ├── distortion.rs   # Hadamard/TDisto (SIMD: is_flat_source_16)
│   ├── lambda.rs       # Lambda calculations (no SIMD needed)
│   ├── level_costs.rs  # Precomputed cost tables
│   └── stats.rs        # Token statistics
├── fast_math.rs        # quality_to_compression, cbrt, pow
├── psy.rs              # Perceptual model (CSF, JND)
├── quantize.rs         # Quantization matrices
├── residual_cost.rs    # SIMD-optimized cost estimation
├── trellis.rs          # Trellis quantization
└── vp8/
    ├── mod.rs          # VP8 encoder main
    ├── header.rs       # Bitstream header
    ├── mode_selection.rs
    ├── prediction.rs
    └── residuals.rs    # Token buffer
```

## Existing SIMD Reference

See `src/common/simd_sse.rs` for patterns:
- `t_transform` (line 432) - Hadamard with weights
- `sse_quantize` (line 563) - Coefficient quantization
- `idct4x4` (line 189) - Inverse DCT

## Profiling Results (2026-02-04)

Hot paths from perf profiling on method 4, 512x512 image:

| Function | % Time | Status |
|----------|--------|--------|
| `choose_macroblock_info` | 32.24% | Main mode selection loop |
| `idct4x4_intrinsics` | 17.71% | SIMD (already optimized) |
| `dct4x4_intrinsics` | 9.56% | SIMD (already optimized) |
| `get_residual_cost_sse2` | 8.00% | SIMD (already optimized) |
| `t_transform_sse2` | 6.80% | SIMD (already optimized) |
| `collect_histogram_with_offset` | 4.73% | **Scalar - optimization target** |
| `memmove` | 2.29% | Memory copies |

## Recent Optimizations

### Method 0-1 Fast DC Path (commit 8fd159e)
- Added `pick_intra16_fast_dc()` for method 0-1
- Uses DC mode only with simple SSE scoring (no full RD)
- Method 0 throughput improved ~33%

### Performance Gap Analysis

| Method | zenwebp | libwebp | Ratio |
|--------|---------|---------|-------|
| 0 | 9.75 ms | 2.79 ms | 3.5x |
| 4 | 27.0 ms | 10.5 ms | 2.6x |
| 6 | 33.3 ms | 16.1 ms | 2.1x |

### Limiting Factor: unsafe Code Forbidden

The crate has `#![forbid(unsafe_code)]`. This prevents using `forge_token_dangerously()`
to eliminate redundant CPU feature checking inside `#[multiversed]` functions. Each SIMD
function call does a runtime `summon()` check even when already inside a feature-specialized
code path.

If unsafe code were allowed, the dispatch overhead could be eliminated for ~10-15% improvement.

## Notes

- All 156 unit tests pass
- Pre-existing test failure in `preset_overrides_work` (unrelated to SIMD)
- Use `#[inline(always)]` for small SIMD helpers called in loops
- The `wide` crate autovectorizes well inside `#[multiversed]` functions - sometimes easier than manual intrinsics
- Criterion benchmarks added: `cargo bench` runs encode_benchmark and encode_vs_libwebp
