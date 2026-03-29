# zenwebp CLAUDE.md

See global ~/.claude/CLAUDE.md for general instructions.
Historical investigation notes and resolved bugs are in [LOG.md](LOG.md).

## Performance & Testing

See `docs/PERFORMANCE.md` for benchmarks, `docs/CALL-TREE.md` for SIMD tiers, `docs/ARCHITECTURE-CLEANUP.md` for code organization.

**Decoder v2** (`src/decoder/vp8v2/`): bit-exact with libwebp, 1.09-1.15x speed, streaming cache→RGB (~100KB peak). Use for all new decode work.

**Pixel-exact gate**: `tests/v2_pixel_perfect.rs` (tolerance 0 vs libwebp). Lossless: `examples/lossless_rt_check.rs` (24/24 exact).

**Benchmarks**: `benches/decode_compare.rs` (14 images), `benches/decode_lossless_compare.rs`. zenbench, no `-C target-cpu=native`.

## Key Files

**Encoder (lossy):**
- `src/encoder/vp8/mod.rs` - Main encoder orchestration, single-pass token loop
- `src/encoder/vp8/residuals.rs` - TokenBuffer, coefficient token recording/emission
- `src/encoder/vp8/header.rs` - Bitstream header encoding
- `src/encoder/vp8/mode_selection.rs` - I16/I4/UV mode selection
- `src/encoder/vp8/prediction.rs` - Block prediction + transform
- `src/encoder/api.rs` - Public API, EncoderConfig, EncoderParams, Preset enum
- `src/encoder/analysis.rs` - DCT analysis, k-means clustering, auto-detection classifier
- `src/encoder/cost.rs` - Cost estimation, trellis quantization, filter level computation
- `src/encoder/psy.rs` - Perceptual model (masking, JND thresholds)
- `src/common/types.rs` - Segment struct, init_matrices, quantization tables

**Encoder (lossless):**
- `src/encoder/vp8l/encode.rs` - Main pipeline, AnalyzeEntropy
- `src/encoder/vp8l/huffman.rs` - Huffman tree construction and encoding
- `src/encoder/vp8l/backward_refs.rs` - LZ77, cache selection, TraceBackwards
- `src/encoder/vp8l/hash_chain.rs` - Hash chain for match finding
- `src/encoder/vp8l/histogram.rs` - Symbol frequency histograms
- `src/encoder/vp8l/entropy.rs` - Entropy cost estimation, PopulationCost
- `src/encoder/vp8l/meta_huffman.rs` - Histogram clustering
- `src/encoder/vp8l/cost_model.rs` - TraceBackwards with Zopfli-style CostManager
- `src/encoder/vp8l/transforms.rs` - Image transforms
- `src/encoder/vp8l/near_lossless.rs` - Near-lossless pixel + residual quantization
**Mux/Demux/Animation:**
- `src/mux/demux.rs` - WebPDemuxer (zero-copy chunk parser)
- `src/mux/assemble.rs` - WebPMux (container assembler)
- `src/mux/anim.rs` - AnimationEncoder (high-level animation API)

## Current Status Summary

### Encoder (Lossy) vs libwebp

**Method mapping** (aligned with libwebp's RD optimization levels):
- m0-2: RD_OPT_NONE (fast, no RD optimization)
- m3-4: RD_OPT_BASIC (RD scoring, no trellis)
- m5: RD_OPT_TRELLIS (trellis during encoding)
- m6: RD_OPT_TRELLIS_ALL (trellis during I4 mode selection)

**Compression (CID22 corpus, 248 images, Q75, SNS=0, filter=0, segments=1):**
| Method | Ratio vs libwebp |
|--------|-----------------|
| 4 | 1.0099x |
| 5 | **1.0002x** |
| 6 | 1.0022x |

**Production settings (SNS=50, filter=60):**
- CID22 Q75: **1.0149x** | Q90: **1.0060x** (near parity)

**Speed (zenbench, 792079.png 512x512, Q75, 2026-03-28):**
| Method | zenwebp | libwebp | Ratio |
|--------|---------|---------|-------|
| 0 | 6.4ms | 2.8ms | 2.3x |
| 4 | 13.5ms | 9.9ms | **1.36x** |
| 6 | 21.0ms | 15.7ms | **1.34x** |

*Instruction ratio ~1.12x but wall-clock ~1.36x — gap is memory access patterns.*

### Encoder (Lossless) vs libwebp

CID22 50-image subset: **0.996x** (0.4% smaller). Screenshots: **0.995x** (0.5% smaller).

**Backward references parity (synthetic, 2026-03-25):**

| Size | m1 | m2 | m4 | m5 |
|------|-----|-----|-----|-----|
| 256x256 | — | — | **0.996x** | — |
| 512x512 | **1.000x** | **1.001x** | 1.014x | 1.004x |
| 640x480 | — | — | 1.017x | — |

Backward refs pipeline (hash chain, LZ77, TraceBackwards, color cache, 2D locality)
verified at parity. Remaining 1-2% gap at m3-m6 is from histogram clustering
differences (stochastic combining index compaction behavior), not backward refs.
128x128 blowup at m2 (4.4x) diagnosed as predictor transform_bits issue (large blocks
for small images), not backward refs.

**Histogram clustering optimization (2026-03-25):**

Two fixes: (1) entropy bin threshold bug — was using accumulator's cost instead
of incoming histogram's cost, making merges progressively harder. (2) cache trial
at m0-m4 — was trying both cache_bits=0 and cache_bits=N, but libwebp only does
this at m5+ q75+.

| Metric | Original | After queue rewrite | After bin+cache fix | After cache/RLE opts |
|--------|----------|---------------------|---------------------|---------------------|
| get_combined_histogram_cost | 7,341M | 1,717M | **307M** | **87M** |
| Total encoder (512x512 m4) | 9,350M | 3,709M | **2,211M** | **511M** |
| vs libwebp | 5.15x | 2.04x | **1.22x** | **1.01x** |

Note: instruction counts changed between sessions because the benchmark image
changed from real photo (792079.png) to synthetic gradient+noise 512x512
(matching libwebp's profiling binary). The synthetic image is more compressible,
giving lower absolute counts but valid relative ratios.

**Lossless instruction parity (synthetic 512x512, m4 q75, 2026-03-25):**

| Function | zenwebp (M) | libwebp (M) | Ratio |
|----------|-------------|-------------|-------|
| calculate_best_cache_size | 156 | 140 | 1.11x |
| get_combined_histogram_cost | 87 | 90 | **0.97x** |
| get_entropy_unrefined | 64 | 23 | 2.78x |
| encode_image_data | 37 | 37 | 1.00x |
| HashChain::new | 34 | 34 | 1.00x |
| backward_refs_rle | 10 | ~1 | — |
| Histogram::from_refs | 18 | 43 | **0.42x** |
| TraceBackwards (cost_model) | 15 | 28 | **0.54x** |
| VP8LBackwardRefsCursorAdd | 0 | 17 | **0.00x** |
| **Total** | **511** | **505** | **1.01x** |

**VP8L core is faster than C:** encode_argb_single_config inclusive 482M vs
libwebp VP8LEncodeStream 496M = **0.97x (3% faster)**. The 6M total excess
(511M vs 505M) is pixel format conversion and test harness overhead outside
the VP8L encoder core.

Functions faster than C: Histogram building (Vec vs linked-list), TraceBackwards
(tighter Rust iteration), no progress reporting overhead.
Functions slower than C: get_entropy_unrefined (bounds-check + codegen overhead),
calculate_best_cache_size (same), backward_refs_rle (quick-reject improved but
still has bounds checks).

ZENWEBP_TRACE=1 env var enables call count instrumentation.

### Decoder vs libwebp

**Wall-clock (x86-64-v3, codec_wiki 2560x1664, 200 iterations, 2026-03-25):**
zenwebp ~10.7ms vs libwebp ~7.9ms = **1.36x** (median)

**Instruction ratio (callgrind, codec_wiki 2560x1664 Q75 RGB, 10 decodes):**
zenwebp 182.0M vs libwebp 161.4M = **1.13x** per decode

**Decoder optimizations (2026-03-25):**
1. Per-block non_zero_blocks bitmap skips IDCT for zero blocks (24.5M -> 1.3M)
2. DC-only WHT fast path for Y2 block (iwht4x4 eliminated for DC-only case)
3. Fixed-length cat_probs iteration (no sentinel branch)
4. Reusable filter parameter buffer (eliminates per-row allocation)
5. Frame buffer FILTER_PADDING removal (saves 344KB per decode)
6. Bit reader sub-slice bounds check elimination (load_new_bytes hot path)
7. Inline state fields into ActivePartitionReader (eliminates pointer indirection;
   read_residual_data 45.4M -> 42.6M per decode, -6.2%)
8. Out-of-line read_coefficients + get_large_value (2026-03-27): Eliminates
   BTB aliasing from 25x inlining. Coeff mispredicts 3.66M -> 1.36M (-62.8%),
   matching libwebp's 1.39M. Adds ~100M instruction overhead from calls.

Total: 229.3M -> 182.0M -> ~179M per decode.
Wall-clock: ~2.3x -> ~1.25-1.28x vs libwebp on codec_wiki.

9. Multi-tier SIMD prediction+IDCT pipeline (2026-03-27): Single `#[arcane]`
   entry per MB puts prediction loops + IDCT in one target_feature region.
   Enables AVX2 autovectorization of prediction functions. Eliminates
   per-block `if let Some(token)` dispatch (24 checks/MB -> 1 dispatch/MB).
10. Bulk cache-to-frame copy (2026-03-27): Replace per-row copy loops in
    output_row_from_cache with single contiguous copy_from_slice calls.
    Enables wider vector stores for the ~40KB/row transfer.

**Remaining instruction gap breakdown (per decode, codec_wiki, Pillow-encoded):**

| Category | zenwebp (M) | libwebp (M) | Gap (M) |
|----------|-------------|-------------|---------|
| Coefficient parsing | 41.4 | ~38.3 | 3.1 |
| Loop filter SIMD | 41.7 | ~30.6 | 11.1 |
| YUV->RGB upsample | 50.4 | ~38.1 | 12.3 |
| Decode orchestration | 36.2 | ~24.9 | 11.3 |
| Predict+IDCT (SIMD) | 6.0 | — | — |
| IDCT (arcane entry) | 1.5 | ~17.1 | **-15.6** |
| memset (buf alloc) | 19.5 | ~0 | 19.5 |
| memcpy | 3.5 | 2.6 | 0.9 |
| Other | 14.3 | ~10 | 4.3 |
| **Total** | **~214** | **161.4** | **~53** |

Note: instruction count increased from ~180M to ~214M due to `#[arcane]`
boundary overhead in the predict_simd pipeline (call/return per MB for
prediction+IDCT dispatch). Wall-clock improved despite higher instruction
count because prediction code benefits from AVX2 autovectorization.

Main remaining opportunities:
- **memset 19.5M**: Frame buffer zero-init. Every byte overwritten before read.
  Could use uninitialized allocation but requires unsafe or refactoring.
- **Coeff parsing 4.3M excess**: Prob table bounds checks, zigzag/dequant indexing.
  Branch mispredicts now at parity with C (1.36M vs 1.39M per 10 decodes).
  Remaining excess is bounds checks on `probs[n][ctx]` array indexing.
- **Loop filter 7.3M excess**: SIMD dispatch overhead, bounds checks.
- **YUV->RGB 7.6M excess**: Scalar edge handling, bounds checks.

### V2 Decoder vs v1 and libwebp (2026-03-27)

**Architecture:** Streaming cache, per-MB predict+IDCT, per-row filter dispatch.
No full-frame Y/U/V allocation during decode (only for Frame output).
Single `#[arcane]` filter boundary per MB row.

**Wall-clock (x86-64-v3, zenbench, RGBA decode, 14 images, 2026-03-27):**

| Image | v1 (Mpix/s) | v2 (Mpix/s) | libwebp (Mpix/s) | v2/v1 | v2/libwebp |
|-------|-------------|-------------|-------------------|-------|------------|
| sc_4k_wiki (8.7M) | 224 | 221 | 487 | 0.99x | 0.45x |
| sc_3k_imac (5.6M) | 277 | 316 | 360 | 1.14x | 0.88x |
| sc_2k_wiki (4.3M) | 508 | 620 | 717 | 1.22x | 0.86x |
| sc_2k_ui (3.7M) | 573 | 731 | 820 | 1.28x | 0.89x |
| sc_1k_term (1.7M) | 354 | 402 | 442 | 1.14x | 0.91x |
| ph_2k_sq (4.2M) | 266 | 301 | 330 | 1.13x | 0.91x |
| ph_2k_43 (3.1M) | 69 | 72 | 78 | 1.04x | 0.93x |
| ph_576_baby | 253 | 280 | 297 | 1.11x | 0.94x |
| ph_512_cid | 256 | 274 | 293 | 1.07x | 0.94x |

**Summary:** v2 is 4-28% faster than v1 (geometric mean ~1.11x).
v2 vs libwebp: 6-12% slower for most images (geometric mean ~0.90x).
4k_wiki outlier (0.99x vs v1) dominated by full-frame allocation overhead.

Previous v1 vs libwebp was ~1.36x; v2 reduces the gap to ~1.11x.

**Correctness:** v2 produces byte-identical YUV planes to v1 on all 218
conformance files and all roundtrip/edge-case tests. The previous max_diff=7
on 49 files was caused by v1's default chroma dithering (strength=50) which
v2 does not implement. With dithering disabled, V2_V1_MAX_TOLERANCE=0.

### Decoder Threading Investigation (2026-03-24)

**Result: NOT WORTH IMPLEMENTING.** libwebp's 2-thread pipeline is a net negative.

Verified with `WEBP_USE_THREAD` patched into libwebp-sys and strace-confirmed
`clone3(CLONE_THREAD)` calls:

| Image              | lib 1T | lib 2T  | threading |
|--------------------|--------|---------|-----------|
| codec_wiki 2560w   | 8.6ms  | 12.1ms  | **-37%**  |
| terminal 1646w     | 4.1ms  |  5.6ms  | **-22%**  |
| imac 2940w         | 16.6ms | 16.9ms  | -2%       |
| windows 2560w      | 12.4ms | 12.1ms  | +3%       |

libwebp's `use_threads` defaults to OFF (simple API never enables it).
The `webpx` crate had a bug where `use_threads` was silently ignored
(fixed in 0.1.4). Coefficient parsing is ~10% of decode — too small
to pipeline effectively. Thread sync overhead dominates.

Our 1.13-1.41x gap vs libwebp is purely single-threaded instruction
count and memory access patterns.

### Lossless Decoder vs libwebp (2026-03-25)

**Wall-clock (x86-64-v3, codec_wiki 2560x1664 lossless, zenbench):**
zenwebp ~11.3ms vs libwebp ~8.5ms = **1.33x**

**Instruction ratio (callgrind, same file, 5 decodes):**
zenwebp 1072M vs libwebp 715M = **1.50x instruction count**

**Optimizations applied:**
1. Packed table: 6-bit single-lookup pixel decode (fires for images with short codes)
2. is_trivial_code: skip all bit reading when all trees are single-symbol
3. is_trivial_literal: pre-pack R/B/A when those channels are constant
4. Incremental col/row tracking (avoids div/mod at tile boundaries)
5. Infallible BitReader::fill() with slow-path split
6. Unchecked consume in Huffman fast path (read_symbol_fast)
7. SSE2 inverse transforms via archmage (2026-03-25):
   - TransformColorInverse: mulhi_epi16 trick for fixed-point multiply
   - AddGreenToBlueAndRed: shuffle+add for green channel broadcast
   - Predictor 1 (left): parallel prefix-sum of 4 pixels
   - Predictors 2,3,4 (top/TR/TL): batch 4-pixel byte-add
   - Predictors 8,9 (avg TL/T, avg T/TR): batch floor-average + add

**Instruction breakdown per 5 decodes (codec_wiki lossless):**

| Component | zenwebp (M) | libwebp (M) | Ratio |
|-----------|-------------|-------------|-------|
| Pixel decode loop | 551 | 241 | 2.29x |
| Inverse transforms | 430 | 254 | 1.69x |
| BGRA->RGBA convert | 0 | 51 | **0x** |
| memset (buffer init) | 69 | 0 | - |
| memcpy | 0 | 64 | **0x** |
| Huffman build | 7 | 7 | 1.0x |
| Other | 15 | 98 | 0.15x |
| **Total** | **1072** | **715** | **1.50x** |

**Inverse transform SIMD progress (per 5 decodes, codec_wiki):**
- Before SIMD: 778M (3.06x vs C)
- After SSE2: 430M (1.69x vs C) — **45% reduction**
- C target: 254M (SSE2+SSE4.1)
- Remaining gap: predictors 5-13 still scalar (serial data dependencies),
  bounds checks in scalar predictor fallbacks, Rust codegen overhead.

**Remaining opportunities:**
- Transforms: predictors 5-7 (average with left, serial), 10-13 (complex
  serial) still scalar. Could use per-pixel SSE2 for predictors 11-13
  (select, clamp) as C does. Predictors 5-7 can't be parallelized.
- memset 69M: buffer zero-init. Every byte overwritten before read.
- Decode loop 2.3x: Huffman match overhead, bounds checks on table lookups,
  Result error-path codegen, per-pixel color cache hash.

## Perceptual Encoder Features (method 3+)

- **Enhanced CSF tables** (method 3+) — best quality improvement alone
- **SATD-based masking** (method 4+) — texture, luminance, edge, uniformity
- **JND thresholds** (method 5+) — frequency-dependent coefficient zeroing
- **Psy-RD disabled** — hurts butteraugli (prefers smoother reconstructions)

Files: `src/encoder/psy.rs`, `src/encoder/trellis.rs`

## SIMD Architecture

### Encoder SIMD (archmage 0.9)

**Fused primitives (2026-02-05):**
- `ftransform_from_u8_4x4` — residual+DCT from flat u8[16] arrays
- `quantize_dequantize_block_simd` — quantize+dequantize in single SIMD pass
- `quantize_dequantize_ac_only_simd` — AC-only variant for I16 Y1 blocks
- `idct_add_residue_inplace` — fused IDCT+add_residue with DC-only fast path
- `tdisto_4x4` — port of libwebp's TTransform_SSE2, vertical-first Hadamard

**#[rite] conversion:** 15 inner `_sse2` functions use `#[rite]` (target_feature + inline)
instead of `#[arcane]`. Eliminates dispatch wrapper overhead. quantize_dequantize_block
went from 21.1M → 2.4M instructions (inlined into I4 inner loop).

**Instruction progression:** 586M → 259M (fused SIMD) → 183M (#[rite]) → 179M (token opt)

### Decoder SIMD

**Loop filters:** All V/H edge filters for luma+chroma use SSE2/SSE4.1 SIMD.
- 32-pixel AVX2 filters implemented but NOT integrated (filter order dependencies, see below)

**YUV→RGB:** 32-pixel SIMD conversion, 16-bit arithmetic via `_mm_mulhi_epu16`.

**Bit reader:** VP8GetBitAlt with 56-bit buffer, `leading_zeros()` → LZCNT.

### Bounds Check Elimination Strategy

**Fixed-region approach (decoder, 2026-02-04):**
```rust
const V_FILTER_REGION: usize = 3 * MAX_STRIDE + 16;
let region: &mut [u8; V_FILTER_REGION] =
    <&mut [u8; V_FILTER_REGION]>::try_from(&mut pixels[start..start + V_FILTER_REGION]).unwrap();
// All subsequent region[offset..] accesses have NO bounds checks
```
- FILTER_PADDING (57KB) added to pixel buffers → ~10% decode speedup
- Memory overhead: ~170KB per decode (negligible)
- Assembly confirmed: interior accesses use direct memory loads, no checks


**Key insight:** Rust asserts at function entry do NOT eliminate bounds checks on individual
slice accesses. Each `try_from(&slice[a..b]).unwrap()` generates 3 separate checks.
Fixed-size array conversion is the only way to eliminate interior checks without unsafe.

### AVX2 32-Pixel Filter Integration (BLOCKED)

32-pixel AVX2 loop filters are implemented and tested but NOT integrated due to
cross-MB filter order dependencies:

1. **Cross-MB write interference:** MB(x+1)'s left edge filter writes overlap MB(x)'s
   horizontal subblock filter reads (columns 12-15). Cannot batch across MBs.
2. **32-row horizontal batching** requires wider cache (extra_y_rows + 32 rows instead of +16).
   Needs restructuring: double cache allocation, paired MB row processing, new filter/output flow.

**Expected benefit:** ~5% total decode improvement. High complexity for modest gain.

### Cache/Memory Layout (Decoder)

**libwebp architecture:** Decode MB → 832-byte working buffer → row cache (tight stride)
→ loop filter on cache → delayed output.

**Our architecture:** Same row cache approach (commit c16995f). `extra_y_rows` = 8 for
normal filter, 2 for simple, 0 for none. Prediction writes to cache, filter operates
on cache, `rotate_extra_rows()` copies bottom rows for next iteration.

**Cache behavior:** Our D1 miss rate 0.1% vs libwebp 0.3% (better). Remaining gap is
instruction count, not cache efficiency.

## Profiler Hot Spots

### Encoder (method 4, 2026-03-28, plasma 512x512 Q75 SNS=0 seg=1, 200M total)
| Function | % | M instr | Notes |
|----------|---|---------|-------|
| evaluate_i4_modes_sse2 | 27.1% | 54.3M | I4 inner loop (inlines quant/dequant/SSE) |
| choose_macroblock_info | 13.6% | 27.2M | I16/UV mode selection |
| encode_image (incl. write_bool) | 12.3% | 24.7M | Main encoding loop + token emission |
| get_residual_cost_sse2 | 10.1% | 20.2M | Coefficient cost estimation |
| record_coeff_tokens | 4.1% | 8.2M | Token recording |

**get_residual_cost optimization history (2026-03-28):**
30.5M -> 24.1M (remapped cost table, padded LevelCostArray) -> 20.2M (direct
indexing with `& 7`/`& 0x7F` bounds proofs). Inner loop: 25 instr (5 branches)
-> 18 instr (0 branches, all cmov). -34% total, -7.6% whole encoder.

**vs libwebp function-level (2026-02-05):**
| zenwebp | M instr | libwebp | M instr | Ratio |
|---------|---------|---------|---------|-------|
| evaluate_i4 + choose_mb + pick_i4 | 61.7M | PickBestIntra4 + quant_enc | 18.3M | 3.4x |
| get_residual_cost_sse2 | 20.2M | GetResidualCost_SSE2 | 9.7M | **2.1x** |
| idct_add_residue + idct4x4 | 8.6M | ITransform_SSE2 | 15.9M | **0.54x** |
| ftransform2 + ftransform_from_u8 | 9.6M | FTransform_SSE2 | 9.8M | **0.98x** |
| quantize_block (standalone) | 4.3M | QuantizeBlock_SSE41 | 6.0M | **0.72x** |

### Decoder (2026-03-27, codec_wiki 2560x1664 Pillow-encoded Q75 RGB, 10 decodes)

zenwebp: 214.5M vs libwebp: 161.4M (**1.33x instruction ratio**)
Wall-clock (zenbench, zenwebp-encoded Q75 m4): 10.1ms vs 6.0ms = **1.68x**

| zenwebp function | M instr | libwebp equivalent | M instr | Ratio |
|-----------------|---------|-------------------|---------|-------|
| read_coefficients | 41.4M | GetCoeffsFast+GetLargeValue | 38.3M | **1.08x** |
| filter_row_simd | 41.7M | HFilter/VFilter/DoFilter SSE2 | 30.6M | **1.36x** |
| fancy_upsample + fill_row | 50.4M | YUV2RGB+Upsample SSE41 | 38.1M | **1.32x** |
| decode_frame_ (exclusive) | 36.2M | VP8DecodeMB+Reconstruct+ParseMode | 24.9M | 1.45x |
| predict+IDCT (#[arcane]) | 6.0M | (included above) | — | — |
| memset | 19.5M | — | ~0 | buffer zeroing |
| IDCT (arcane entry) | 1.5M | Transform_SSE2+DC prediction | 17.1M | **0.09x** |
| memcpy | 3.5M | memcpy | 2.6M | 1.35x |

decode_frame_ exclusive is higher (36.2M vs 22.9M) due to SIMD pipeline dispatch
overhead and scalar fallback code bloating the function body (36KB of machine code).
The predict_simd `#[arcane]` functions are separate (6.0M total).

IDCT is 11x faster than libwebp (zero-block skip + DC-only WHT).
Main remaining targets: memset (buffer alloc), decode_frame_ code bloat,
bounds checks in loop filter / YUV->RGB conversion.

## Remaining Optimization Opportunities

### Encoder
1. **Mode selection 3.4x vs libwebp** — I4 inner loop orchestration overhead
2. **Residual cost 2.1x (was 2.4x)** — inner loop now branchless, remaining
   gap is Rust overhead (cmov for ctx/level clamping, SIMD abs precompute
   setup, dispatch wrapper ~1.9M per encode)
3. **Wall-clock 1.36x (was 1.47x)** — memory access patterns still dominant
4. **Defer I16 reconstruction** — only IDCT winning mode (saves ~48 IDCT/MB)

### Decoder
1. **IDCT skip (DONE)** — Per-block non_zero_blocks bitmap eliminates IDCT for zero
   blocks. 24.5M -> 1.3M per decode. Matches libwebp's DoTransform case 0 / bits!=0.
2. **Multi-tier SIMD predict+IDCT (DONE)** — `#[arcane]` entry per MB puts prediction
   + IDCT in single target_feature region. Enables AVX2 autovectorization.
3. **Bulk cache-to-frame copy (DONE)** — Contiguous copy_from_slice replaces per-row
   loops in output_row_from_cache.
4. **Loop filter 41.7M vs 30.6M (1.36x)** — single `#[arcane]` entry with `#[rite]`
   inlining. Remaining gap from bounds checks and different code shape.
5. **Coefficient parsing 41.4M vs 38.3M (1.08x)** — near parity. Remaining excess
   from bounds checks on prob table lookups, zigzag/dequant indexing.
   Branch mispredicts at C parity (1.36M vs 1.39M/10 decodes).
6. **YUV->RGB 50.4M vs 38.1M (1.32x)** — scalar edge handling, bounds checks.
7. **memset 19.5M** — frame buffer zero-init. Every byte is overwritten before read.
8. **decode_frame_ code bloat (36KB)** — scalar prediction fallback code bloats
   decode_frame_ even though it's never executed on AVX2 hardware. Could be
   addressed by marking scalar fallbacks `#[cold]` or separating them.

## Profiling Commands

```bash
# Pre-convert image for callgrind (avoids PNG decoder AVX-512 issues)
convert image.png -depth 8 RGB:image_WxH.rgb

# Profile zenwebp
valgrind --tool=callgrind --callgrind-out-file=/tmp/callgrind.zen.out \
  target/release/examples/callgrind_encode image_WxH.rgb W H 75 4

# Profile libwebp
valgrind --tool=callgrind --callgrind-out-file=/tmp/callgrind.lib.out \
  target/release/examples/callgrind_libwebp image_WxH.rgb W H 75 4

# Criterion head-to-head
cargo bench --bench encode_vs_libwebp
```

## Quality Search (target_size)

```rust
let output = EncoderConfig::new()
    .quality(75.0)
    .target_size(10000)
    .encode_rgb(&pixels, width, height)?;
```
Secant method, convergence |dq| < 0.4, max passes = method + 3 or 6.

## Preset Tuning

| Preset | SNS | Filter | Sharp | Segs |
|--------|-----|--------|-------|------|
| Default | 50 | 60 | 0 | 4 |
| Photo | 80 | 30 | 3 | 4 |
| Drawing | 25 | 10 | 6 | 4 |
| Auto | detected | detected | detected | detected |

Auto: ≥0.45 uniformity → Photo, <0.45 → Default, ≤128px → Icon.

## no_std Support

`cargo build --no-default-features` for no_std+alloc. Both decoder and encoder work.
Dependencies: `thiserror`, `whereat`, `hashbrown`, `libm` (all no_std).

## Safety

`#![forbid(unsafe_code)]`. SIMD via `archmage` token-based safety (proc-macro generates
unsafe internally). No manual unsafe, transmute, get_unchecked, or raw pointer derefs.

## Examples and Dev Tools

**examples/** — Public API demonstrations:
- `api_guide.rs` — Comprehensive demo of 100% of zenwebp's public API

**dev/** — Internal diagnostic, benchmark, and comparison tools (48 files).
Not compiled by default. To use, move back to `examples/` or add `[[example]]`
entries to Cargo.toml. Key tools:

| Tool | Usage |
|------|-------|
| `corpus_test [dir]` | Batch file size comparison vs libwebp |
| `compare_all_methods` | Per-method size comparison |
| `callgrind_encode` | Minimal encoder for callgrind profiling |
| `decode_benchmark` | Decode speed comparison |
| `debug_mode_decision` | MB_DEBUG env for mode selection |
| `lossless_benchmark` | Lossless corpus benchmark |

## TODO: WebP Conformance Testing

**Status**: CI infrastructure in place, pending invalid/non-conformant files

**Phase 1 (DONE):**
- [x] Add conformance test file (`tests/webp_conformance.rs`)
- [x] Add conformance CI job (`.github/workflows/ci.yml`)
- [x] Integrate 225 valid WebP files from codec-corpus

**Phase 2 (Pending):**
- [ ] Create `invalid/` test files (corrupted/truncated WebP)
  - [ ] Truncated files (incomplete bitstream)
  - [ ] Malformed headers (bad chunk sizes, invalid FourCC)
  - [ ] Oversized dimensions (width/height > 16384)
  - [ ] Reserved field violations
- [ ] Create `non-conformant/` test files (gray-area edge cases)
  - [ ] Loop filter edge cases
  - [ ] Color space ambiguities (no ICC profile)
  - [ ] Alpha blending semantics
  - [ ] Rounding behavior differences

**Generation script:** Use `codec-corpus/webp-conformance/generate_corpus.py` to regenerate
synthetic valid files if needed. For invalid files, corrupt valid files programmatically:

```bash
# Truncate a valid file
truncate -s 500 ~/codec-corpus/webp-conformance/valid/file.webp > \
  ~/codec-corpus/webp-conformance/invalid/truncated/incomplete.webp

# Corrupt chunk size (Python)
python3 << 'EOF'
import struct
data = open('~/codec-corpus/webp-conformance/valid/file.webp', 'rb').read()
modified = bytearray(data)
modified[4:8] = struct.pack('<I', len(data) - 100)  # Wrong chunk size
open('~/codec-corpus/webp-conformance/invalid/malformed/bad_chunk_size.webp', 'wb').write(modified)
EOF
```

**Testing:** Run with `cargo test --release test_webp -- --ignored`

## Investigation Notes

### Branchless coefficient parsing (2026-03-27)

Attempted multiple approaches to reduce VP8 coefficient parsing branches.
Summary of findings (codec_wiki 2560x1664, cachegrind branch-sim):

**What worked:**
- `#[inline(never)]` on `read_coefficients` + `get_large_value` split:
  Coeff mispredicts 3.66M -> 1.36M (C's 1.39M). Eliminates BTB aliasing
  from 25x inlining. Wall-clock within noise (~1-2% improvement).

**What did NOT work:**
- **Branchless get_bit (multiply-select):** Eliminated 7.6M branches but
  added 37.4M instructions from mul/wrapping_sub codegen. Mispredicts
  unchanged (the `value > split` branch was already well-predicted). Net negative.
- **Branchless get_bit (cmov hint):** LLVM did generate 88 cmovs (up from 4),
  but the conditional arithmetic overhead cancelled the branch savings.
- **Flat prob table (u8 vs TreeNode):** 4x smaller memory, but changed code
  layout enough to INCREASE mispredicts by 0.3M. Cache was not the bottleneck.
- **Deferred refill (get_bit_fast):** Skipping `bits < 0` check works for
  normal-size images but fails at EOF for tiny images (2x2, 3x3). The
  `bits < 0` check is perfectly predicted anyway — zero mispredict savings.

**Key insight:** VP8's boolean decoder branches are highly biased and
predict well. The excess branch COUNT (69.5M vs 21.3M when inlined 25x)
came from BTB aliasing, not from inherently unpredictable branches.
Making the function out-of-line was the only approach that addressed
the root cause.

### Arithmetic encoder optimization (2026-03-28)

Rewrote `ArithmeticEncoder::write_bool` to match libwebp's `VP8PutBit`:

**What worked:**
- **Lookup-table normalization (kNorm/kNewRange):** Eliminates while-loop
  that iterated 1-7 times per bit. One table lookup gives shift count and
  final range. Matches libwebp's bit_writer_utils.c tables exactly.
- **Run-length carry handling:** Tracks pending 0xFF byte count instead of
  walking backwards through the output buffer on carry. O(1) vs O(n).
- **Batched bit shift:** Shifts value by full shift count at once, checks
  flush once per normalization instead of per-bit.
- **Result:** callgrind 203.1M -> 200.2M (-1.4%, -2.9M). write_bool fully
  inlined into encode_image (was 5.6M standalone at m4 SNS=0 seg=1).

**What did NOT work for record_coeff_tokens:**
- **Stack-local token buffer + extend_from_slice:** Eliminated 44 `grow_one`
  cold paths in assembly (1453->930 asm lines, -36%). But the 640-byte memset
  per call for the `[u16; 320]` buffer added 1.4M instructions to memset,
  cancelling the 0.55M savings on the function itself. Net regression.
- **Pre-reserve(320) per call:** Added 3M instructions from the reserve
  comparison on every call (all no-ops since buffer is pre-allocated).
  Vec::push capacity checks are effectively free — branch predictor handles
  the always-taken fast path with zero mispredicts.

**Key insight:** Vec::push in hot loops generates `grow_one` cold paths that
bloat the function's assembly (code size), but the branch predictor handles
the capacity check perfectly. The per-push overhead is ~1 compare + ~1 branch
(always predicted taken), which is negligible. Attempts to eliminate these
checks by staging in stack buffers add memset/memcpy overhead that exceeds
the savings. The `#![forbid(unsafe_code)]` ceiling for per-token recording
is essentially the current implementation.

**Production profile (default settings, 5x encode, 512x512 Q75 m4):**
| Function | M instr | % |
|----------|---------|---|
| get_residual_cost_sse2 | 564M | 25.4% |
| encode_image (incl. write_bool) | 472M | 21.2% |
| evaluate_i4_modes_sse2 | 296M | 13.3% |
| record_coeff_tokens | 291M | 13.1% |
| choose_macroblock_info | 119M | 5.4% |

## Known Bugs

(none currently)

## User Feedback Log

(none currently)

## API Design Conventions

**No backwards compatibility required** — no external users. Bump 0.x for breaking changes. Delete old APIs, no deprecation shims. One obvious way to do things — no duplicate entry points.

**Builder convention**: `with_` prefix for consuming builder setters, bare-name for getters.

**Licensing**: AGPL-3.0-or-later with commercial licensing (support@imazen.io). Versions 0.1.x-0.3.x were MIT OR Apache-2.0.

**Project standards**: `#![forbid(unsafe_code)]` with default features. no_std+alloc (minimum: wasm32). CI with codecov. Fuzz targets required. Safe for malicious input — no amplification, bound memory/CPU.

**Streaming encode** — `push_rows`/`finish` implemented. Lossy RGB8 converts to YUV420 during push (50% memory savings). Other formats accumulate raw bytes. WebP algorithms still need the full image at finish time, but callers can push strips without holding the full source.
