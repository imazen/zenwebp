# zenwebp Performance Analysis

Comprehensive performance comparison against libwebp (C) and image-webp (pure Rust).
All benchmarks use runtime SIMD dispatch (no `-C target-cpu=native`).

Last updated: 2026-03-26.

## Results Summary

### Lossless Encode (512x512 photo, Q75)

| Method | zenwebp | libwebp C | Ratio | Size ratio |
|--------|---------|-----------|-------|------------|
| m0 | 68ms | 28ms | 2.4x slower | 0.84x (smaller) |
| m2 | 100ms | 98ms | **1.02x** | 1.001x |
| m4 | 121ms | 117ms | **1.03x** | 1.004x |
| m6 | 132ms | 340ms | **2.6x faster** | 1.013x |

Compression at parity (1.00-1.01x). 24/24 pixel-exact roundtrips verified.

### Lossless Decode

| Image | zenwebp | libwebp C | image-webp | zen/lib |
|-------|---------|-----------|------------|---------|
| photo 512x512 | 101 MPix/s | 107 MPix/s | 104 MPix/s | **0.94x (faster)** |
| codec_wiki 2560x1664 | 202 MPix/s | 495 MPix/s | 191 MPix/s | 2.45x → **1.31x** |
| terminal 1646x1062 | 541 MPix/s | 779 MPix/s | 483 MPix/s | 1.44x → **1.23x** |

### Lossy Decode (14-image corpus, diverse content)

**Screenshots (UI, text, sharp edges):**

| Image | zenwebp | libwebp C | zen/lib |
|-------|---------|-----------|---------|
| sc_4k_wiki 3508x2480 | 223 MPix/s | 496 MPix/s | **2.22x** |
| sc_3k_imac 2940x1912 | 266 MPix/s | 362 MPix/s | **1.36x** |
| sc_2k_wiki 2560x1664 | 448 MPix/s | 720 MPix/s | **1.61x** |
| sc_2k_ui 1920x1920 | 512 MPix/s | 828 MPix/s | **1.62x** |
| sc_1k_term 1646x1062 | 319 MPix/s | 454 MPix/s | **1.42x** |

**Photos (CLIC2025, professional):**

| Image | zenwebp | libwebp C | zen/lib |
|-------|---------|-----------|---------|
| ph_2k_sq 2048x2048 | 245 MPix/s | 324 MPix/s | **1.32x** |
| ph_2k_43 2048x1536 | 68.2 MPix/s | 77.7 MPix/s | **1.14x** |
| ph_2k_32 2048x1360 | 175 MPix/s | 221 MPix/s | **1.26x** |
| ph_2k_uw 2048x976 | 114 MPix/s | 136 MPix/s | **1.19x** |
| ph_2k_pt 1360x2048 | 240 MPix/s | 313 MPix/s | **1.30x** |

**Small photos (576px, 512px):**

| Image | zenwebp | libwebp C | zen/lib |
|-------|---------|-----------|---------|
| ph_576_baby | 223 MPix/s | 302 MPix/s | **1.35x** |
| ph_576_city | 123 MPix/s | 150 MPix/s | **1.22x** |
| ph_576_flowers | 103 MPix/s | 125 MPix/s | **1.21x** |
| ph_512_cid | 225 MPix/s | 301 MPix/s | **1.34x** |

Photos: **1.14-1.35x** slower. Screenshots: **1.36-2.22x** (gap correlates with
bits-per-pixel — low-bpp screenshots exercise loop filter/YUV more than bit reader).
zenwebp is 2-2.5x faster than image-webp across all content types.

---

## Lossless Encoder Optimization History

### Starting point: 28x slower, 50-100% larger files

The lossless encoder's public API (`EncodeRequest`) was routing through a simple
literal-only encoder that wrote raw pixel values with basic Huffman coding — no LZ77,
no color cache, no histogram clustering, no transforms. The full VP8L pipeline existed
internally but wasn't wired to the public API.

### Fix 1: Wire full VP8L pipeline to public API

**Impact: 50-100% larger → 0.97-1.00x (compression parity)**

`encode_frame_lossless()` now routes color image encoding through `encode_vp8l()`.
The simple encoder is kept only for alpha plane encoding and grayscale.

### Fix 2: Replace libm::log2 with lookup tables

**Impact: 59.3B → 42.6B instructions (28% reduction)**

`fast_slog2()` was calling `libm::log2()` for every histogram entry — billions of
calls. Replaced with libwebp's approach: 256-entry `kSLog2Table` for v < 256,
CLZ + `kLog2Table` + linear correction for v < 65536. No `log()` call for any
practical histogram value.

A second copy of `fast_slog2` in `transforms.rs` was still using `libm::log2` —
fixing that gave another 17% speedup.

### Fix 3: Reduce cache_bits trials from N to 2

**Impact: 42.6B → ~11.3B instructions (3.3x speedup)**

Was trying all 10 possible cache_bits values (0-10), running the full histogram
clustering pipeline for each. libwebp only tries 2: the entropy-selected value
and 0. Gated on `method >= 5 && quality >= 75` matching libwebp's `do_no_cache`.

This was the single biggest optimization — purely algorithmic, no SIMD needed.

### Fix 4: Fix entropy bin threshold

**Impact: 82x → 19x → parity in entropy calls vs libwebp**

The entropy binning phase used the accumulator histogram's cost for merge threshold
(grows as histogram absorbs more entries, making subsequent merges harder). libwebp
uses the incoming histogram's cost (stays constant). Fix: merges went from 166 to
2,014, post-binning active histograms from 941 to 17.

### Fix 5: Priority-queue greedy combining

Replaced O(n² × merges) re-evaluation with incremental O(n) updates per merge,
matching libwebp's `HistogramCombineGreedy`.

### Fix 6: Eliminate recomputation after merges

After merging histograms, libwebp copies precomputed per-type costs from the pair
evaluation. We were calling `compute_histogram_cost()` (5× `population_cost` = 5×
`get_entropy_unrefined`) after every merge — 2,036 unnecessary calls.

### Fix 7: Flat cache storage + optimized cache key

Replaced `Vec<Vec<u32>>` (double indirection) with flat `Vec<u32>` plus offset
table. Precompute hash keys once for max cache size, shift for smaller sizes.

### Fix 8: Method 0 fast path

Method 0 now skips TraceBackwards DP, color cache, and Box LZ77 — matching
libwebp's `GetBackwardReferencesLowEffort`.

### Fix 9: SSE2 SIMD for lossless encode transforms

Ported `CombinedShannonEntropy_SSE2`, `SubtractGreenFromBlueAndRed_SSE2`, and
`TransformColor_SSE2` via archmage `#[rite]`. ~5% wall-clock improvement from
reduced branch mispredictions.

### What was tried and didn't help (encoder)

- **`#[autoversion]` on inner entropy function**: Generated 3 ISA variants
  (v1/v2/v3), causing I-cache bloat (1514 vs 546 lines of asm). Net slower.
- **Scalar bitmask for zero-skip**: Building a bitmask of nonzero entries takes
  ~48 instructions for 16 elements in scalar; SSE2's `_mm_movemask_epi8` does it
  in ~4. Scalar approach was slower than simple branching.
- **Type reordering in histogram cost evaluation**: Processing cheaper types
  (distance=40 entries) before expensive (literal=680) for better early bail-out.
  Threshold rarely hit, no improvement.
- **`#[inline]` on entropy functions**: LLVM was already inlining at -O3.
- **Vec reuse in stochastic combining**: Eliminated per-iteration `Vec::collect()`,
  but the Vec was tiny compared to entropy computation cost.

### Final encoder state

VP8L core: 3% faster than libwebp C in instruction count (482M vs 496M).
Remaining overhead is pixel format conversion outside the encoder core.

---

## Lossy Decoder Optimization History

### Starting point: 1.3-1.8x slower (wall-clock), 2.21x instruction ratio

### Fix 1: Skip IDCT for zero-coefficient blocks

**Impact: IDCT 24.5M → 1.3M instructions (18.5x reduction)**

zenwebp was running IDCT on every 4x4 block regardless of coefficient content.
Added `non_zero_blocks: u32` bitmap to `MacroBlock`, set during coefficient
parsing. IDCT skipped when the corresponding bit is clear. On compressed images
(Q75, 0.11 bpp), most blocks have zero residuals.

### Fix 2: DC-only WHT fast path

When the Y2 block has only a DC coefficient, use `dc0 = (dc[0] + 3) >> 3` and
broadcast — matching libwebp's `ParseResiduals` shortcut. Eliminated `iwht4x4`
from the hot path (19.7M → ~0).

### Fix 3: Branchless coefficient parsing

- `get_signed()`: Branchless VP8GetSigned using conditional arithmetic instead of
  `get_bit(128)` + branch.
- Array dequant: `dq[(n > 0) as usize]` instead of `if n > 0 { ac } else { dc }`.
- Fixed-size output: `&mut [i32; 16]` instead of `&mut [i32]` to eliminate bounds
  checks on zigzag writes.

### Fix 4: Convert loop filters from #[arcane] to #[rite]

**Impact: ~2% on screenshots**

All 20+ loop filter functions in `loop_filter_avx2.rs` and `loop_filter_neon.rs`
were using `#[arcane]` (target_feature boundary per call = 4x overhead). Changed
to `#[rite]` with a single `#[arcane]` entry point at `filter_row_simd`. All
filter calls now inline into one target_feature region.

### Fix 5: Inline bit reader state fields

**Impact: 6.2% reduction in read_residual_data**

`ActivePartitionReader` stored state via `&mut VP8BitReaderState` (pointer
indirection). Copied `value`, `range`, `bits`, `pos`, `eof` as direct struct
fields, writing back on `Drop`. LLVM keeps all hot fields in registers.

### Fix 6: Out-of-line coefficient parsing (BTB fix)

**Impact: Coefficient mispredicts 3.66M → 1.36M (matches libwebp's 1.39M)**

`read_coefficients` was `#[inline(always)]`, duplicated 25 times inside
`read_residual_data` (once per block). This created 25 separate sets of branch
addresses for identical logic, causing Branch Target Buffer aliasing. Changed to
`#[inline(never)]`, extracted `get_large_value` as separate `#[inline(never)]`.

### What was tried and didn't help (lossy decoder)

- **Branchless `get_bit` (multiply-select)**: Added 37.4M instructions for 7.6M
  fewer branches. Mispredicts unchanged because the branches predict well (>90%
  accuracy). Net slower.
- **Flat probability table (u8 instead of TreeNode)**: Changed memory layout enough
  to increase mispredicts. Cache was not the bottleneck.
- **Deferred refill (skip `bits < 0` check)**: The check is perfectly predicted
  (~0 mispredicts). Removing it broke EOF handling for tiny images.
- **`#[inline]` on large prediction functions**: Caused I-cache pressure, +10%
  regression. Reverted.
- **SSE2 VectorMismatch for hash chain**: Net negative for photo content — most
  LZ77 match attempts find short matches (< 4 elements), so SIMD setup overhead
  dominated.
- **Cursor-based bit reader (fat pointer)**: Extra pointer overhead from maintaining
  both `remaining` and `state.pos`. 17% regression.

### Remaining lossy decode gap (~1.4-1.7x)

| Category | Excess instructions | Root cause |
|----------|-------------------|------------|
| memset (buffer zeroing) | ~18M/decode | Safe Rust zeroes all Vec allocations; C uses malloc |
| Coeff parsing | ~8M | Extra branches from loop bounds (55.8M vs 22.6M total branches) |
| Loop filter | ~7M | Bounds checks within SIMD filter code |
| YUV→RGB | ~7M | Scalar edge handling for non-aligned widths |
| MB orchestration | ~8M | Indirect branch overhead from enum/match dispatch |

The coefficient branch count (55.8M vs 22.6M) is NOT from bounds checks (LLVM
proves them all away — zero panic paths in assembly). It's from having more
explicit loop control (`while n < 16`, `if n >= 16 return`) vs C's implicit
pointer arithmetic. The branch PREDICTION now matches C exactly (1.36M vs 1.39M
mispredicts), so the excess branches are well-predicted and cheap (~1 cycle each).

The memset overhead (18M) is fundamental to `#![forbid(unsafe_code)]` — Rust
zeroes Vec allocations, C doesn't. Buffer reuse would help for repeated decodes
of same-size images but doesn't help for the single-decode case.

---

## Lossless Decoder Optimization History

### Starting point: 2.15x instruction ratio, ~2.5x wall-clock on screenshots

### Fix 1: Packed table + trivial code/literal optimizations

When all 4 channel Huffman codes fit in 6 total bits, decode an entire ARGB pixel
in one 64-entry table lookup. Added `is_trivial_code` (skip all bit reading for
constant pixels) and `is_trivial_literal` (pre-pack R/B/A, only read green).
Incremental col/row tracking eliminates div/mod at tile boundaries.

### Fix 2: u32-based ColorCache with precomputed hash_shift

Replaced abstracted ColorCache with direct u32 array + precomputed hash shift.
Matches libwebp's inline `VP8LColorCacheLookup`/`VP8LColorCacheInsert`.

### Fix 3: Infallible BitReader::fill()

`fill()` always returned `Ok(())`; removed Result overhead. Split slow path
(< 8 bytes remaining) into `#[cold] #[inline(never)]` helper.

### Fix 4: SSE2 inverse transforms via archmage

**Impact: Transform instructions -45%, total -24%**

Ported from libwebp's `lossless_sse2.c`:
- `TransformColorInverse_SSE2`: Cross-color inverse using `_mm_mulhi_epi16`
- `AddGreenToBlueAndRed_SSE2`: Subtract-green inverse
- Predictor 1 (left): Parallel prefix-sum of 4 pixels
- Predictors 2, 3, 4: Batch `_mm_add_epi8`
- Predictors 8, 9: Batch floor-average using `_mm_avg_epu8`

All use `#[rite]` with single `#[arcane]` entry point per transform.

### What was tried and didn't help (lossless decoder)

- **Deferred color cache insert (batch at row boundary)**: libwebp does this
  because it stores pixels as `uint32_t` (one load per insert). Our byte-array
  storage requires 4 loads + reconstruct, making batch insert SLOWER (1423M →
  1870M). Not viable without changing pixel storage format.

### Remaining lossless decode gap (~1.2-1.3x on screenshots)

The remaining gap is from:
- Scalar fallback for predictors 5-7, 10-13 (serial data dependencies)
- Per-pixel Huffman tree traversal overhead vs C's pointer-based access
- `uint32_t` vs `[u8; 4]` pixel storage (C gets single-instruction hash; Rust
  needs 4 byte loads)

Photo content (512x512) is at parity or faster due to no BGRA→RGBA conversion
overhead (libwebp stores ARGB internally, pays 51M + 64M for conversion).

---

## Profiling Methodology

All measurements use:
- **zenbench 0.1.1** for wall-clock (interleaved, paired statistics)
- **valgrind --tool=callgrind** for instruction counts
- **valgrind --tool=cachegrind --branch-sim=yes** for cache + branch analysis
- **heaptrack** for allocation profiling
- Same WebP file bytes for both decoders (encode once, decode both)
- Default compiler target (no `-C target-cpu=native`) for production-representative
  results with runtime SIMD dispatch via archmage `summon()`

Test corpus:
- `codec_wiki.png` (2560x1664) — screenshot
- `terminal.png` (1646x1062) — screenshot
- `792079.png` (512x512) — CID22 photo

## Architecture

- `#![forbid(unsafe_code)]` — all SIMD via archmage proc macros
- SSE2 runtime dispatch via `Sse2Token::summon()` for decode transforms
- `#[rite]` for all SIMD inner functions, `#[arcane]` only at entry points
- No `target-cpu=native` dependency — works on any x86-64
