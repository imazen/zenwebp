# zenwebp Encoder Optimization Status

## Current Performance (2026-02-05, commit 92ac1cc)

| Metric | zenwebp | libwebp | Ratio |
|--------|---------|---------|-------|
| **Instructions** | **182.0M** | **203M** | **0.90x** |
| Output size | 12,198 | 12,018 | 1.015x |
| Wall-clock m4 | 15.0ms | 10.2ms | 1.47x |
| Wall-clock m6 | 22.2ms | 15.5ms | 1.43x |

*Test image: 792079.png 512x512, Q75, M4, SNS=0, filter=0, segments=1*

## Instruction Breakdown (callgrind, non-inclusive)

| Function | M instr | % | Notes |
|----------|---------|---|-------|
| evaluate_i4_modes_sse2 | 37.2 | 20.5% | I4 inner loop (inlines quantize/dequantize/SSE/DCT) |
| encode_image | 24.4 | 13.4% | Main encoding loop orchestration |
| get_residual_cost_sse2 | 21.6 | 11.9% | Coefficient cost estimation |
| choose_macroblock_info | 18.9 | 10.4% | I16/UV mode selection |
| convert_image_yuv | 9.5 | 5.2% | YUV conversion (in callgrind example) |
| ftransform2 | 7.8 | 4.3% | Fused 2-block DCT |
| record_coeff_tokens | 7.4 | 4.1% | Token recording to buffer |
| idct_add_residue_inplace | 6.8 | 3.7% | Fused IDCT+add_residue |
| pick_best_intra4 (outer) | 5.6 | 3.1% | I4 orchestration |
| write_bool | 4.6 | 2.5% | Arithmetic encoding |
| quantize_block (standalone) | 4.3 | 2.4% | Quantize (I16 path) |
| presort_i4_modes_sse2 | 3.1 | 1.7% | I4 mode SSE pre-sort |
| quantize_dequantize (standalone) | 2.4 | 1.3% | Fused Q+DQ (UV path) |

## vs libwebp Function Comparison

| zenwebp function | M instr | libwebp equivalent | M instr | Ratio | Root cause |
|-----------------|---------|-------------------|---------|-------|------------|
| get_residual_cost_sse2 | 21.6 | GetResidualCost_SSE2 | 9.7 | **2.2x** | Bounds checks in safe Rust |
| record_coeff_tokens | 7.4 | VP8RecordCoeffTokens | 3.4 | **2.2x** | Vec::push + stats recording overhead |
| evaluate_i4 + choose_mb + pick_i4 | 61.7 | PickBestIntra4 + quant | 18.3 | **3.4x** | Mode loop overhead + bounds checks |
| idct_add_residue + idct4x4 | 8.6 | ITransform_SSE2 | 15.9 | **0.54x** | We're faster! |
| ftransform2 + from_u8 | 9.6 | FTransform_SSE2 | 9.8 | **0.98x** | Parity |
| quantize_block (standalone) | 4.3 | QuantizeBlock_SSE41 | 6.0 | **0.72x** | We're faster! |

## Optimization History

| Date | Change | Before | After | Savings |
|------|--------|--------|-------|---------|
| Feb 5 | Fused SIMD primitives | 586M | 259M | -56% |
| Feb 5 | Hoist SIMD dispatch (arcane I4) | 259M | 259M | ~0% (prep for rite) |
| Feb 5 | archmage 0.5 #[rite] conversion | 259M | 183.3M | -29% |
| Feb 5 | Residual cost inner loop indexing | 183.3M | 182.0M | -0.7% |

## Key Discoveries

### Bounds Checks Are The Dominant Safe Rust Overhead

The 2.2x gap in `get_residual_cost_sse2` (21.6M vs 9.7M) is fundamentally from Rust's
bounds checking. The inner loop has ~22 instructions with 3 bounds checks (6 instructions,
27%). libwebp uses raw pointer arithmetic with zero bounds checks.

Assembly analysis shows the hot loop:
```asm
.LBB287_12:
    cmp rdi, 16              ; bounds check: n < 16
    jae panic                ; → panic_bounds_check
    movzx edx, [abs_levels]  ; load abs level
    cmp edx, 2047            ; min clamp
    cmovae edx, esi
    cmp rax, 2               ; bounds check: ctx < 3
    ja panic                 ; → panic_bounds_check
    movzx r8d, [levels]      ; load clamped level
    cmp r8b, 68              ; bounds check: level < 68
    jae panic                ; → panic_bounds_check
    ...                      ; actual work
```

Options to reduce: (1) use unchecked indexing (requires unsafe), (2) restructure to
fixed-iteration loops where compiler can prove bounds, (3) accept the gap as cost of safety.

### #[rite] Inlining Is Highly Effective

Converting 15 inner `_sse2` functions from `#[arcane]` to `#[rite]` achieved 29% instruction
reduction. The biggest single win: `quantize_dequantize_block` went from 21.1M (separate
function with dispatch wrapper) to ~2.4M (inlined into evaluate_i4_modes_sse2).

Key insight: `#[rite]` = `#[target_feature] + #[inline]`. From arcane/rite context, calls
inline fully. From non-arcane context, requires unsafe → use `#[arcane]` entry shim.

### Wall-Clock vs Instructions Gap

Instructions are 0.90x of libwebp but wall-clock is 1.47x. The gap is NOT from:
- L1 cache misses (our D1 miss rate is better: 0.1% vs 0.3%)
- SIMD dispatch overhead (eliminated by #[rite])

Likely causes:
- Memory access patterns (our stride = full image width vs libwebp's compact row cache)
- Branch misprediction from bounds checks
- Valgrind's instruction counting doesn't reflect real-world pipeline effects

### Pre-Indexing Cost Tables Helps Minimally

Matching libwebp's `CostArrayPtr` pattern (pre-index cost tables by type, then use
`t = costs_for_type[band][ctx]` pointer advancement) saved only 1.3M instructions.
The compiler already optimizes multi-dimensional array access well; the overhead is
in the bounds checks themselves, not the index computation.

## Remaining Optimization Opportunities

### High Impact (>5M savings potential)
1. **Mode selection overhead** (evaluate_i4 37.2M) — 3.4x vs libwebp. Mostly safe Rust
   overhead (bounds checks, slice operations). Would need algorithmic changes or unsafe.
2. **encode_image** (24.4M) — Main loop orchestration. Profile-guided optimization target.

### Medium Impact (2-5M savings potential)
3. **get_residual_cost_sse2** (21.6M, 2.2x) — Bounds checks dominate. Could use unsafe
   indexing but contradicts project safety goals.
4. **record_coeff_tokens** (7.4M, 2.2x) — Vec::push overhead + stats recording. Could
   pre-allocate exact capacity or use fixed buffer.

### Low Impact (<2M savings potential)
5. **ftransform2** (7.8M) — Already at parity with libwebp.
6. **presort_i4_modes** (3.1M) — Small function, limited optimization potential.

### Architectural Changes (high effort)
7. **Row cache for encoder** — Our encoder writes to full-image buffers (stride = width).
   libwebp uses a compact row cache. This affects cache behavior but not instruction count.
