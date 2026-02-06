# zenwebp Encoder Optimization Status

## Current Performance (2026-02-05, commit da96836)

| Metric | zenwebp | libwebp | Ratio |
|--------|---------|---------|-------|
| **Instructions (diag)** | **182.8M** | **203M** | **0.90x** |
| **Instructions (default)** | **226.3M** | - | - |
| Output size (diag) | 12,198 | 12,018 | 1.015x |
| Output size (default) | 11,864 | - | - |
| Wall-clock m4 | 16.3ms | 10.2ms | 1.60x |
| Wall-clock m6 | 23.8ms | 15.5ms | 1.53x |

*Test image: 792079.png 512x512, Q75, M4*
*Diagnostic: SNS=0, filter=0, segments=1. Default: SNS=50, filter=60, segments=4.*

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

## Cachegrind Analysis (2026-02-05)

### Summary

| Metric | zenwebp | libwebp | Ratio |
|--------|---------|---------|-------|
| D reads | 38.5M | 41.0M | 0.94x |
| D writes | 20.7M | 13.6M | **1.52x** |
| D1 read misses | 73,250 | 64,832 | 1.13x |
| D1 write misses | 91,511 | 41,243 | **2.22x** |
| D1 read miss rate | 0.2% | 0.2% | ~same |
| D1 write miss rate | 0.4% | 0.3% | ~same |

### D1 Read Cache Miss Hotspots

| zenwebp function | D1 rd misses | % | libwebp equivalent | D1 rd misses | % |
|-----------------|-------------|---|-------------------|-------------|---|
| get_residual_cost_sse2 | 20,354 | **27.8%** | GetResidualCost_SSE2 | 4,336 | 6.7% |
| ftransform2 | 14,476 | 19.8% | - | - | - |
| convert_image_yuv | 12,293 | 16.8% | ConvertRGBToY_SSE41 | 12,289 | 19.0% |
| encode_image | 6,983 | 9.5% | ??? (top fn) | 15,861 | 24.5% |
| record_coeff_tokens | 6,255 | 8.5% | VP8RecordCoeffTokens | 130 | 0.2% |

**Key finding:** `get_residual_cost_sse2` has **4.7x more D1 read cache misses** than
libwebp's equivalent. The `LevelCostTables` struct is ~52KB (4×8×3×68×2 bytes), which
exceeds the 32KB L1 data cache. Our access pattern jumps between different type/band/ctx
combinations, causing cache thrashing.

libwebp mitigates this by pre-indexing `CostArrayPtr` so the hot inner loop only touches
a small region of the cost table at a time. We pre-index by type but still index by
band×ctx in the inner loop, spreading accesses across more cache lines.

### Write Overhead (52% more writes)

Our 20.7M writes vs libwebp's 13.6M comes from:
- `choose_macroblock_info` + `evaluate_i4_modes`: 6.2M writes (mode selection scratch buffers)
- `pick_best_intra4`: 1.7M writes (I4 orchestration)
- `encode_image`: 1.7M writes (main loop)
- `__memcpy_avx`: 1.1M writes with 52.5% of all write cache misses (buffer copies)
- `__memset_avx2`: 0.9M writes with 17.9% of write cache misses (zeroing)

Write cache misses are dominated by memcpy (48K misses) and memset (16K misses).
These suggest unnecessary buffer copying or zeroing in the hot path.

## Hardware Counter Analysis (perf stat, 2026-02-05)

| Metric | zenwebp | libwebp | Notes |
|--------|---------|---------|-------|
| Cycles | 65.2M | 66.5M | We use fewer cycles! |
| Instructions | 179M | 195M | 0.92x (fewer instructions) |
| IPC | 2.75 | 2.93 | 6% lower throughput per cycle |
| Branch misses | 418K (2.17%) | 667K (3.88%) | **We miss fewer!** |
| Frontend stalls | 8.79% | 11.66% | We stall less |

**Important:** These numbers are for diagnostic mode (SNS=0, filter=0, segments=1).
The criterion benchmarks use default settings (SNS=50, filter=60, segments=4), which
adds segment analysis, SNS processing, and loop filtering — work not measured here.

**Key insight:** In diagnostic mode, we're at near-cycle-parity with libwebp. The
1.47x wall-clock gap in criterion benchmarks is likely from default-settings overhead
(segment analysis, SNS, filter level computation) that doesn't appear in callgrind.

**Counter-intuitive branch miss result:** Our bounds checks ADD branches but they're
highly predictable (almost never taken). libwebp has fewer total branches but more
data-dependent branches that mispredict. Net effect: our 2.17% miss rate beats their 3.88%.

## Heap Allocation Analysis (heaptrack, 2026-02-05)

| Metric | Value |
|--------|-------|
| Total allocations | 45 |
| Reallocs | ~10 (Vec growth) |
| Peak memory | 3.6M |
| Leaked | 544B (system) |

**Key allocations:**
- Token buffer: 1.54M (768 tokens/MB × 1024 MBs × 2 bytes) — properly pre-allocated
- Stored MB coefficients: 1.6M (1024 MBs × 1600 bytes each) — necessary storage
- YUV buffers: 393K (Y + U + V planes)
- ArithmeticEncoder partition: 65K

**No allocation bottleneck.** 45 total allocations for a 512×512 encode is excellent.
Token buffer pre-allocation was increased from 300 to 768 tokens/MB to match libwebp's
`TOKENS_COUNT_PER_MB = 2 * 16 * 24 = 768`.

## Optimization History

| Date | Change | Before | After | Savings |
|------|--------|--------|-------|---------|
| Feb 5 | Fused SIMD primitives | 586M | 259M | -56% |
| Feb 5 | Hoist SIMD dispatch (arcane I4) | 259M | 259M | ~0% (prep for rite) |
| Feb 5 | archmage 0.5 #[rite] conversion | 259M | 183.3M | -29% |
| Feb 5 | Residual cost inner loop indexing | 183.3M | 182.0M | -0.7% |
| Feb 5 | Token buffer pre-allocation fix | - | - | eliminates 1 realloc |
| Feb 5 | SIMD histogram + import_block opt | 247M* | 226.3M* | -8.4% |

*\* Default-mode instruction counts (SNS=50, filter=60, segments=4)*

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

### get_residual_cost Cache Thrashing

The cost table (`LevelCostTables`) is ~52KB, exceeding the 32KB L1 data cache. Our
`get_residual_cost_sse2` accounts for 27.8% of all D1 read cache misses despite being
only 11.9% of instructions. libwebp's equivalent accounts for only 6.7%.

The access pattern jumps between `costs_for_type[band][ctx][level]` combinations where
band and ctx change each iteration, touching different cache lines. Reducing the table
size or restructuring for sequential access could help.

**Possible mitigation:** The cost table is indexed as `[4 types][8 bands][3 ctx][68 levels]`.
Most accesses use only type 1 (I4_AC) or type 3 (UV_AC). Pre-extracting the active type's
sub-table (~13KB = 8×3×68×2) would fit in L1 cache.

### #[rite] Inlining Is Highly Effective

Converting 15 inner `_sse2` functions from `#[arcane]` to `#[rite]` achieved 29% instruction
reduction. The biggest single win: `quantize_dequantize_block` went from 21.1M (separate
function with dispatch wrapper) to ~2.4M (inlined into evaluate_i4_modes_sse2).

Key insight: `#[rite]` = `#[target_feature] + #[inline]`. From arcane/rite context, calls
inline fully. From non-arcane context, requires unsafe → use `#[arcane]` entry shim.

### Wall-Clock vs Instructions Gap

Instructions are 0.90x of libwebp but wall-clock is 1.47x with default settings.
Hardware counters in diagnostic mode show near-cycle-parity (65.2M vs 66.5M cycles).
The remaining wall-clock gap in benchmarks comes from:

1. **Default settings overhead** — criterion uses SNS=50, filter=60, segments=4.
   Segment analysis, SNS computation, and filter level estimation add significant work
   not captured in diagnostic-mode profiling.
2. **Lower IPC** (2.75 vs 2.93) — 6% less instruction-level parallelism, possibly from
   longer dependency chains in safe Rust code.
3. **Write overhead** — 52% more memory writes (20.7M vs 13.6M) with 2.2x more write
   cache misses, suggesting unnecessary buffer copying or zeroing.

**NOT the cause:**
- L1 cache misses: similar overall miss rate (0.2% both)
- Branch misprediction: we mispredict FEWER branches (2.17% vs 3.88%)
- SIMD dispatch overhead: eliminated by #[rite] conversion

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
3. **get_residual_cost_sse2** (21.6M, 2.2x) — Bounds checks + cache thrashing (27.8% of
   D1 read misses). Could use unsafe indexing or restructure cost table layout.
4. **record_coeff_tokens** (7.4M, 2.2x) — Vec::push overhead + stats recording.

### Low Impact (<2M savings potential)
5. **ftransform2** (7.8M) — Already at parity with libwebp. Has 19.8% of D1 read misses.
6. **presort_i4_modes** (3.1M) — Small function, limited optimization potential.

### Architectural Changes (high effort)
7. **Write overhead reduction** — Investigate unnecessary buffer copies/zeroing.
   52% more writes than libwebp, 2.2x more write cache misses. Biggest contributors
   are memcpy (52% of write misses) and memset (18%).

## Default-Mode Instruction Breakdown (226.3M, commit da96836)

| Function | M instr | % | Notes |
|----------|---------|---|-------|
| evaluate_i4_modes_sse2 | 42.7 | 18.9% | I4 inner loop |
| encode_image | 35.9 | 15.9% | Main encoding loop + default settings overhead |
| get_residual_cost_sse2 | 21.8 | 9.6% | Coefficient cost estimation |
| choose_macroblock_info | 20.5 | 9.1% | I16/UV mode selection |
| ftransform2_sse2 | 12.0 | 5.3% | DCT (mode selection + histogram) |
| tdisto_4x4_fused | 10.6 | 4.7% | Spectral distortion (SNS) |
| convert_image_yuv | 9.5 | 4.2% | YUV conversion |
| record_coeff_tokens | 7.3 | 3.2% | Token recording |
| idct_add_residue | 6.8 | 3.0% | Fused IDCT+add |
| pick_best_intra4 | 5.9 | 2.6% | I4 orchestration |
| collect_histogram_sse2 | 5.2 | 2.3% | Analysis histogram (SIMD) |
| memcpy | 4.7 | 2.1% | Buffer copies |
| write_bool | 4.5 | 2.0% | Arithmetic encoding |
| quantize_block | 4.3 | 1.9% | Quantize (I16 path) |

Default-mode overhead vs diagnostic: 226.3M - 182.8M = **43.5M** (was 65M before SIMD histogram).
Main sources: tdisto_4x4 (10.6M), collect_histogram (5.2M), encode_image loop overhead (11.5M).
