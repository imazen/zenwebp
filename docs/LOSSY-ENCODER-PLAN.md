# Lossy Encoder — Close 1.43-1.47x Gap to <1.10x

## Current State

512x512 Q75 m4: **1.47x** wall-clock, **1.12x** instructions vs libwebp C.
The wall-clock gap exceeds the instruction gap — memory access patterns dominate.

## Profile (512x512 Q75 m4, 179M total)

| Function | zenwebp (M) | libwebp (M) | Ratio |
|----------|-------------|-------------|-------|
| I4 mode eval + choose_mb | 61.7 | 18.3 | **3.4x** |
| Residual cost | 22.9 | 9.7 | **2.4x** |
| IDCT (recon) | 8.6 | 15.9 | **0.54x** (faster!) |
| Forward DCT | 9.6 | 9.8 | **0.98x** (parity) |
| Quantize | 4.3 | 6.0 | **0.72x** (faster!) |

The encoder is faster than C on IDCT, DCT, and quantize. The gap is entirely
in **mode selection** (3.4x) and **residual cost estimation** (2.4x).

## Plan

### Phase 1: Mode selection (3.4x — the whole gap)
- I4 mode evaluation tests all 10 modes × 16 sub-blocks = 160 predict+DCT+quant+cost cycles per MB
- libwebp's `PickBestIntra4` is a tight loop with minimal overhead
- Profile: where does the 3.4x come from?
  - Per-mode prediction overhead (10 match arms per sub-block)
  - Per-mode DCT+quant overhead (should be near parity given our SIMD)
  - Mode decision data structures (do we allocate per-mode?)
  - I16 mode evaluation duplicated work (predict all 4 modes, should skip obvious losers)
- Apply fixed-size array pattern to mode evaluation buffers
- Consider: batch all 10 mode predictions into one `#[arcane]` region

### Phase 2: Residual cost estimation (2.4x)
- `get_residual_cost_sse2` is already `#[rite]` SSE2
- Profile: is the 2.4x from the cost function itself or from call overhead?
- libwebp's `GetResidualCost_SSE2` uses a tight lookup table
- Check if our token probability table layout matches libwebp's for cache locality
- Apply fixed-size array pattern to coefficient cost tables

### Phase 3: Memory access patterns (wall-clock > instruction ratio)
- 1.47x wall-clock vs 1.12x instructions = **1.31x** from memory effects
- Profile with cachegrind: D1 miss rate, branch mispredictions
- The mode selection loop accesses prediction workspace, coefficient buffer, quantization matrices, and cost tables per sub-block — bad locality
- Consider: reorganize to process all sub-blocks of one mode together (better cache line reuse) instead of all modes of one sub-block

### Phase 4: Encoder-specific SIMD gaps
- Residual cost: only SSE2, missing NEON/WASM
- Mode selection: `#[arcane]` at X64V3 level, but inner work is SSE2
- Consider V3 (AVX2) for wider mode evaluation batches

### Phase 5: Algorithmic
- **Defer I16 reconstruction**: currently IDCT all 16 blocks for each I16 mode candidate. Only need IDCT for the winning mode.
- **Early exit from mode evaluation**: if a mode's partial cost already exceeds the best, skip remaining sub-blocks
- **RD cache**: cache prediction + DCT results across modes when the same sub-block borders are used

### Target: <1.10x wall-clock, maintaining compression parity

### Not planned
- Multi-threading (WebP encode is inherently serial per-MB due to prediction dependencies)
- Quality changes (compression ratio is already at parity)
