# zenwebp Session Context — 2026-03-27/28

This document captures everything learned in an intensive optimization session.
Read this BEFORE touching any zenwebp code. It will save days of rediscovery.

---

## 1. What Was Accomplished

### Lossless Encoder: 28x slower → parity with libwebp C
- **Root cause**: `encode_frame_lossless()` in `api.rs` routed color images through a simple literal-only encoder that wrote raw pixel values with basic Huffman coding. The full VP8L pipeline (LZ77, backward references, histogram clustering, color cache, predictor/cross-color/subtract-green transforms) existed internally in `encoder/vp8l/encode.rs` but was never wired to the public `EncodeRequest` API. The simple encoder was meant only for alpha plane encoding.
- **The fix** (commit `3c0972a`): added a `!implicit_dimensions && is_color` check that routes to `encode_vp8l()`. Simple encoder kept for alpha planes and grayscale.
- **Result**: 50-100% larger files → 0.97-1.00x compression parity. 28x slower → 1.03x at m2-m4, 2.6x FASTER at m6.

### Lossy Decoder v2: 1.3-2.2x → 1.09-1.15x of libwebp C
- Ground-up redesign as `src/decoder/vp8v2/` (9 files, 6169 lines)
- Streaming cache→RGB, no full-frame Y/U/V buffers (~100KB peak vs ~4.5MB)
- Bit-exact with libwebp (0 pixel diffs on 218+ conformance files)
- Animation support, chroma dithering, security hardening
- Merged as PR #2

### Architecture Cleanup
- 10 per-platform SIMD files → 3 consolidated (loop_filter.rs, transform.rs, yuv.rs)
- vp8.rs (2087 lines) → 5 modules
- 21 decode entry points → thin wrappers
- 4 bit readers → 2
- `simd` feature removed — archmage/magetypes always-on
- ~580 bounds checks eliminated via fixed-size array pattern

---

## 2. The VP8 Boolean Decoder — What We Learned The Hard Way

The VP8 boolean (arithmetic) decoder is the single hottest function in lossy decode.
Every optimization attempt taught us something:

### What worked:
- **`#[inline(never)]` on `read_coefficients`** (commit `be6666d`): The function was `#[inline(always)]`, duplicated 25 times inside `read_residual_data` (once per 4x4 block in a macroblock). This created 25 separate sets of branch addresses for identical logic, causing Branch Target Buffer (BTB) aliasing in the CPU. Mispredicts dropped from 3.66M to 1.36M — matching libwebp's 1.39M exactly. This is a general principle: when the same function is inlined into many call sites within a hot loop, the branch predictor sees the same branch pattern at many different addresses and can't learn it. Out-of-line gives one canonical address.

- **`#[inline(never)]` on `get_large_value`** (same commit): Separated the rare-path large coefficient categories (3-6) from the common path. Matches libwebp's `GetLargeValue`/`GetCoeffsFast` split.

- **Inlined state fields in `ActivePartitionReader`** (commit `1ff53b2`): Copied `value`, `range`, `bits`, `pos`, `eof` from `&mut VP8BitReaderState` into the struct as direct fields, writing back on `Drop`. Eliminated pointer indirection on every `get_bit` call. LLVM keeps all hot fields in registers (`r12`, `ebp`, `r15d`). 6.2% reduction in `read_residual_data`.

- **Sub-slice bounds check elimination** (commit `1f39f49`): In `load_new_bytes`, used `let tail = &self.data[self.pos..]; if tail.len() >= 8` so LLVM can see the length check and access are on the same slice variable. Eliminated 2 bounds-check branches from hot path (confirmed by assembly).

### What did NOT work:
- **Branchless `get_bit` (multiply-select)**: Replaced `if bit != 0 { range -= split } else { range = split + 1 }` with `range = bit * (range - split) + (1-bit) * (split + 1)`. Added 37.4M instructions for 7.6M fewer branches. Mispredicts unchanged because the `value > split` branch was already well-predicted (>92% accuracy). Net slower.

- **Branchless `get_bit` (cmov hint)**: LLVM generated 88 cmovs (up from 4), but conditional arithmetic overhead cancelled branch savings.

- **Flat prob table (u8 vs TreeNode)**: Changed probability table from `TreeNode { left, right, prob, index }` (4 bytes) to flat `u8` (1 byte). 4x smaller memory, but changed code layout enough to INCREASE mispredicts by 0.3M. Cache was not the bottleneck.

- **Deferred refill**: Skipping the `bits < 0` check in `get_bit` to avoid the branch. The check is perfectly predicted (~0 mispredicts). Removing it broke EOF handling for tiny images (2x2, 3x3).

- **Cursor-based bit reader (fat pointer)**: Stored `remaining: &[u8]` sub-slice instead of `data` + `pos`. Added pointer overhead. 17% regression.

### Key insight:
VP8's boolean decoder branches are highly biased and predict well. The excess branch COUNT (55.8M vs 22.6M) came from BTB aliasing when inlined 25x, not from inherently unpredictable control flow. The only fix was making the function out-of-line. This is the `#![forbid(unsafe_code)]` ceiling for per-bit codec inner loops.

---

## 3. The Fixed-Size Array Pattern — The #1 Safe Rust Optimization

This was the single most impactful discovery. It applies everywhere in hot loops.

### The problem:
Safe Rust bounds-checks every `slice[index]` access. In a loop that processes 16 pixels with stride-based indexing (`pixels[y * stride + x]`), that's 16 bounds checks per iteration. Each check is a compare + conditional branch to a panic path. LLVM can sometimes prove the check unnecessary, but NOT when the index depends on a runtime variable like `stride`.

### The pattern:
```rust
// BEFORE: 16 bounds checks (one per row)
fn process_block(pixels: &mut [u8], stride: usize) {
    for row in 0..16 {
        pixels[row * stride] = 128; // bounds check every time
    }
}

// AFTER: 1 bounds check at entry, 0 inside
const REGION_SIZE: usize = 15 * MAX_STRIDE + 1;
fn process_block(region: &mut [u8; REGION_SIZE], stride: usize) {
    assert!(stride <= MAX_STRIDE);
    for row in 0..16 {
        region[row * stride] = 128; // LLVM proves in-bounds: row*stride < 15*MAX_STRIDE+1
    }
}
```

The caller does ONE `try_into().unwrap()` to create the fixed-size reference:
```rust
let region: &mut [u8; REGION_SIZE] = pixels[base..base+REGION_SIZE].try_into().unwrap();
process_block(region, stride);
```

### Results where applied:
| Target | Instruction reduction | Code reduction |
|--------|----------------------|----------------|
| Loop filter (190 checks) | **-33.7%** filter instructions | -39% asm lines, -84% panic paths |
| Prediction (luma+chroma) | **-46% to -64%** per function | process_luma_mb: 1706→914 instr |
| Transform (IDCT) | **-37%** IDCT function | idct_add_residue: 214→135 lines |
| Encoder quantize | All 60 checks eliminated | 0 panic paths in hot functions |
| SIMD helpers | 118 of 147 checks eliminated | — |
| YUV fused kernel | ~800 checks/row eliminated | — |

### Gotchas:
- `stride` must be bounded by a constant (`assert!(stride <= MAX_STRIDE)`) for LLVM to prove interior accesses safe. A `debug_assert!` is NOT sufficient — it's stripped in release.
- The fixed-size array must be large enough for the worst case. `FILTER_PADDING_Y = 15 * MAX_STRIDE + 16` added ~240KB to cache buffers. Worth it.
- For SIMD functions using `#[rite]`, the fixed-size array pattern works perfectly — the array size is a compile-time constant, and all interior SIMD loads/stores compile to raw `movups`/`movaps` with no checks.
- `first_chunk::<N>()` (stable since Rust 1.77) is cleaner than `try_from(&slice[..N]).unwrap()` for sub-array extraction.

---

## 4. archmage SIMD — Correct Usage Patterns

### The token model:
A token (e.g., `X64V3Token`) is a zero-sized proof that the CPU has certain features. `summon()` reads a global bool (~1.3ns). The token itself costs nothing to pass.

### What's expensive:
The `#[target_feature]` boundary. When a function has `#[target_feature(enable = "avx2")]`, LLVM compiles it in a separate code region. Calling into it requires an indirect call that LLVM can't optimize across — no inlining, no interprocedural optimization. This is `#[arcane]`.

### Correct pattern:
```
#[arcane] fn entry(token: V3Token, ...) {      // ONE boundary crossing
    inner_work(token, ...);                     // #[rite] — inlines freely
    more_work(token, ...);                      // #[rite] — same target_feature region
}

#[rite] fn inner_work(token: V3Token, ...) { } // No boundary, inlines
#[rite] fn more_work(token: V3Token, ...) { }  // No boundary, inlines
```

### Wrong pattern (we had this, cost 4x):
```
#[arcane] fn filter_a(token, ...) { }  // Boundary per call
#[arcane] fn filter_b(token, ...) { }  // Another boundary
#[arcane] fn filter_c(token, ...) { }  // Another boundary
// Called 12 times per MB = 12 boundary crossings
```

### `#[autoversion]` — when to use:
For scalar loops that benefit from autovectorization. Generates copies at V3/NEON/WASM/scalar tiers with a dispatcher. Good for prediction loops, color conversion. NOT good for the inner coefficient parser (register pressure changes hurt it).

### `incant!` — multi-tier dispatch:
```rust
fn my_func(data: &[u8]) {
    incant!(my_func(data), [v3, neon, wasm128, scalar])
}
// Calls my_func_v3, my_func_neon, my_func_wasm128, or my_func_scalar
```

### Platform gating:
archmage's `#[arcane]` and `#[rite]` auto-apply `#[cfg(target_arch)]`. You do NOT need manual `#[cfg]` guards. `summon()` returns `None` on wrong platforms. `incant!` falls through to `_scalar`.

### The `simd` feature is gone:
archmage and magetypes are mandatory deps. No `#[cfg(feature = "simd")]` anywhere. The macros handle everything.

---

## 5. Lossless Encoder — Detailed Optimization Journey

### The 59.3B → 0.5B instruction reduction:

| Step | Before | After | Technique |
|------|--------|-------|-----------|
| Baseline (libm::log2) | 59.3B | — | `fast_slog2()` called libm for every value |
| LUT lookup tables | 59.3B | 42.6B | 256-entry `kSLog2Table` + CLZ+kLog2Table for v≥256 |
| Duplicate fast_slog2 in transforms.rs | 42.6B | ~36B | Second copy still used libm |
| Cache_bits trials N→2 | ~36B | ~11.3B | Was trying 10 cache sizes, libwebp tries 2 |
| Entropy bin threshold bug | 11.3B | ~3.7B | Accumulator cost → incoming cost |
| Priority-queue greedy combining | 3.7B | ~2.2B | O(n²×merges) → O(n) incremental |
| Eliminate compute_histogram_cost after merges | 2.2B | ~2.0B | Copy precomputed costs from pair |
| Cache key + flat storage | 2.0B | ~0.5B | Vec<Vec<u32>> → flat Vec<u32> |

### The cache_bits bug (biggest single fix):
libwebp's `EncodeImageInternal` only tries both cache_bits=0 and cache_bits=N when `do_no_cache=1` (method≥5, quality≥75). We unconditionally tried both, running TWO full histogram clustering passes at methods 0-4. This alone was a 3.3x speedup.

### The entropy bin threshold bug:
In histogram clustering, the binning phase merges histograms that are "close enough." The threshold was computed using the accumulator histogram's cost (which grows as it absorbs more histograms, making subsequent merges harder). libwebp uses the incoming histogram's cost (constant per histogram). Fix: merges went from 166 to 2,014, post-binning active histograms from 941 to 17.

### The 82x entropy overcall:
After the queue rewrite, `get_combined_entropy_unrefined` was still called 82x more than libwebp. Root cause: the stochastic combining loop evaluated too many pairs. Fix: correct the iteration count and queue size to match libwebp's `HistogramCombineStochastic`.

### Profiling methodology:
- **Always profile the SAME image** through both encoders. We used a synthetic 512x512 gradient+noise image (matching libwebp's profiling binary) and the real 792079.png photo.
- The instruction counts changed between images because the synthetic image is more compressible. Always compare ratios, not absolute counts.
- Build the C profiling binary from vendored source with `-g` for symbols.

---

## 6. Lossy Decoder v2 — Architecture Deep Dive

### Why a ground-up rewrite:
The v1 decoder had fundamental architectural issues that couldn't be fixed incrementally:
1. Full-frame Y/U/V buffers (4.5MB for 4K) allocated and zeroed per decode
2. Multiple `#[arcane]` boundaries per MB (predict, IDCT, filter — 3+ crossings)
3. `MacroBlock` structs stored for entire image (818KB at 4K)
4. YUV→RGB conversion as a separate pass after full decode
5. `DecodeError` with String variants used inside hot loops (stack bloat from drop guards)

### v2 module layout:
```
src/decoder/vp8v2/
    mod.rs              -- DecoderContext, decode_to_rgb, decode_to_frame, decode_mb_rows
    context.rs          -- Buffer management, ensure_capacity, reuse logic
    header.rs           -- Frame header parsing → FrameTables
    tables.rs           -- FrameTables, PrecomputedFilterParams, DequantPair
    coefficients.rs     -- Free-function coefficient parsing, flat u8 probs
    predict_fused.rs    -- Fused predict+IDCT per MB, #[arcane] per platform
    pipeline.rs         -- Per-row filter dispatch, #[arcane] entry
    yuv_exact.rs        -- Bit-exact fused fancy upsample + YUV→RGB
    animation.rs        -- Animated WebP via WebPDemuxer + frame reuse
```

### The streaming pipeline:
```
For each MB row:
  Phase A (scalar): Parse MB headers + read coefficients
  Phase B (#[arcane]): Predict + IDCT for each MB → write to cache
  Phase C (#[arcane]): Loop filter entire row in cache
  Phase D: Convert visible cache rows → RGB output (yuv_exact)
  Phase E: Rotate cache extra rows for next iteration
```

Coefficient parsing stays scalar because the boolean decoder's performance is register-pressure-sensitive — compiling under `#[target_feature(avx2)]` changes LLVM's register allocation and can HURT the bit reader. This was discovered when `#[autoversion]` on the entropy function caused 3x I-cache bloat from generating 3 ISA variants.

### Buffer reuse across frames:
`DecoderContext::ensure_capacity()` uses `Vec::resize()` not `vec![]`. On repeated decodes of same-size images, zero allocation happens. Only `fill()` is called on the small extra_y_rows region (~few KB) for filter context initialization. For animation, one context decodes all frames.

### The fancy upsample kernel:
The `yuv` crate's bilinear upsample produces max-diff-165 vs libwebp because its weights differ. Our kernel in `yuv_exact.rs` uses the EXACT 9:3:3:1 weights matching libwebp's `UpsampleRgbaLinePair`:
```
u_tl = (9*cur[cx] + 3*cur[cx-1] + 3*next[cx] + next[cx-1] + 8) >> 4
```
With exact integer `mulhi` color conversion:
```
R = clip((mulhi(Y,19077) + mulhi(V,26149) - 14234) >> 6)
G = clip((mulhi(Y,19077) - mulhi(U,6419) - mulhi(V,13320) + 8708) >> 6)
B = clip((mulhi(Y,19077) + mulhi(U,33050) - 17685) >> 6)
```
The V3 SIMD path processes 32 luma pixels per iteration using `_mm_mulhi_epu16`.

### The `yuv` crate is ONLY for encode:
`yuv::rgb_to_yuv420()` is used for RGB→YUV conversion during lossy encoding (2x faster than our scalar). It is NOT used for decode because its bilinear upsample differs from WebP's fancy upsample. This was verified: `fast-yuv=ON` gives max-diff-165 vs libwebp, `fast-yuv=OFF` gives 0 diffs.

---

## 7. What Failed — Optimization Dead Ends

### PGO (Profile-Guided Optimization):
Built with `-Cprofile-generate`, ran decode workloads, merged profiles, rebuilt with `-Cprofile-use`. Result: slightly WORSE on most images. The profile workload didn't represent the benchmark corpus well, and LLVM's layout changes hurt more than they helped. PGO needs a much more comprehensive training set to help codecs.

### AVX2 autovectorization of prediction loops:
Put prediction functions inside `#[arcane](X64V3Token)` to get AVX2 autovectorization. The inner loops are only 4-16 pixels wide — too short for 32-byte AVX2 registers. No improvement on x86, but the structural change helps NEON (16-byte, matches 16-pixel rows).

### Scalar bitmask for zero-skip in entropy:
Built a bitmask of nonzero histogram entries (processing 16 at a time), then iterate only nonzero positions via `trailing_zeros()`. The bitmask BUILDING costs ~48 scalar instructions for 16 elements. SSE2's `_mm_movemask_epi8` does it in ~4. The scalar approach was SLOWER than simple branching. Only real SIMD intrinsics help here.

### `#[autoversion]` on inner entropy function:
Generated 3 ISA variants (v1/v2/v3) that all inlined into the caller. I-cache bloat from 546→1514 lines of assembly. Actually slower.

### Batch zero-skip in entropy:
OR 8 elements, skip chunk if all zero. Only helps when all 8 are zero AND current streak is zero. Saved ~1.2B instructions (10.5B→9.35B) but was the limit of scalar approaches.

### SSE2 VectorMismatch for hash chain:
Ported libwebp's `VectorMismatch_SSE2` which compares 4 u32 values at a time. Net NEGATIVE for photo content — most LZ77 matches are short (<4 elements), so SIMD setup cost per call dominated.

---

## 8. Profiling Methodology

### Tools:
- **callgrind** for instruction counts (build WITHOUT -C target-cpu=native — valgrind can't handle AVX-512)
- **cachegrind --branch-sim=yes** for branch misprediction counts per function
- **heaptrack** for allocation profiling
- **zenbench** (NOT criterion) for wall-clock
- **cargo asm** to verify bounds checks disappeared

### Critical rules:
- **NEVER benchmark with `-C target-cpu=native`**. It bakes in AVX-512 at compile time, bypassing archmage's runtime dispatch. Production users get runtime dispatch.
- **Always profile BOTH sides** on the SAME image bytes. Encode once, save the WebP, decode with both.
- **Build libwebp C from vendored source** with `-g` for function names: `gcc -O2 -g -o /tmp/profile_lib $WEBP/src/enc/*.c $WEBP/src/dsp/*.c $WEBP/src/utils/*.c $WEBP/sharpyuv/*.c -lm -lpthread`
- **Count instructions, not just wall-clock**. WSL environment is noisy (20%+ variance). Instruction counts are deterministic.
- **Check cache/branch effects** separately with cachegrind when instruction ratio doesn't explain wall-clock ratio.

### The libwebp C source:
Vendored at `/home/lilith/.cargo/registry/src/index.crates.io-*/libwebp-sys-0.14.2/vendor/src/`
Key files: `enc/vp8l_enc.c`, `enc/histogram_enc.c`, `enc/backward_references_enc.c`, `dsp/lossless_enc.c`, `dsp/lossless_enc_sse2.c`, `dec/vp8_dec.c`, `dec/vp8l_dec.c`

---

## 9. Benchmark Corpus

All benchmarks use 14 images from codec-corpus spanning screenshots and photos:

**Screenshots** (UI, text, flat areas + sharp edges):
- sc_4k_wiki: 3508x2480, png-conformance (high detail, worst case for decoder)
- sc_3k_imac: 2940x1912, gb82-sc
- sc_2k_wiki: 2560x1664, gb82-sc
- sc_2k_ui: 1920x1920, png-conformance
- sc_1k_term: 1646x1062, gb82-sc

**Photos** (CLIC2025 professional):
- ph_2k_sq: 2048x2048 (square)
- ph_2k_43: 2048x1536 (4:3, highest bpp, closest to C)
- ph_2k_32: 2048x1360 (3:2)
- ph_2k_uw: 2048x976 (ultrawide)
- ph_2k_pt: 1360x2048 (portrait)

**Small photos** (gb82 576x576, CID22 512x512):
- ph_576_baby, ph_576_city, ph_576_flowers, ph_512_cid

The gap vs libwebp correlates with **bits per pixel**, not resolution. Low-bpp screenshots exercise the loop filter and YUV conversion proportionally more. High-bpp photos stress the bit reader. The 4K wiki screenshot (268KB, 3.5 bpp) is the worst case because the coefficient parsing bounds checks dominate.

---

## 10. The Remaining Gaps — Where Time Goes

### Lossy decode v2 (1.09-1.15x):
The 9-15% gap is from safe Rust bounds checking in the bit reader (`get_bit` has a `bits < 0` refill check every call, and the data slice access has a bounds check on refill). These can NOT be eliminated without `unsafe`. The fixed-size array pattern doesn't help here because the bit reader operates on a variable-length data stream, not a fixed-size region.

### Lossless decode (1.23-1.50x):
- Pixel decode loop 2.29x — Huffman table traversal bounds checks, color cache hash
- Inverse transforms 1.69x — missing NEON/WASM, predictors 5-13 still scalar
- memset 69M — buffer zeroing (streaming architecture would eliminate)

### Lossy encode (1.43-1.47x):
- Mode selection 3.4x — the entire gap. 160 predict+DCT+quant+cost cycles per MB.
- Residual cost 2.4x — token probability table lookup overhead
- Wall-clock 1.47x despite 1.12x instructions — memory access pattern mismatch

---

## 11. Security Status

### Hardened:
- Header parsing: dimension limits (0 < w,h ≤ 16383), overflow-safe MB computation
- `ensure_capacity`: all buffer sizes computed with `checked_mul`/`checked_add`, returns `Result` not `expect`
- Partition validation: sizes checked against remaining data
- Malformed input tests: 10 rejection tests (zero dims, huge dims, truncated, bad start code, non-keyframe)
- Fuzz target: `fuzz/fuzz_targets/decode_v2.rs`

### Not hardened:
- 899 potential panic paths from slice indexing in inner loops (fixed-size array pattern handles many, but some remain on dynamic-length data)
- The coefficient parser assumes partition data is well-formed after header validation — a carefully crafted partition could still trigger out-of-bounds
- No fuzzing campaign has been run on v2 yet

---

## 12. Test Commands

```bash
# Core tests
cargo test --release --lib                          # 227 tests
cargo test --release --test v2_pixel_perfect        # 0 diffs vs libwebp (218+ files)
cargo test --release --test v2_animation            # 10 animation tests
cargo test --release --test v2_decoder              # 6 roundtrip tests
cargo test --release --test dither                  # chroma dithering tests

# Benchmarks (NEVER use -C target-cpu=native)
cargo bench --bench decode_compare                  # 14-image lossy decode (4 decoders)
cargo bench --bench decode_lossless_compare         # lossless decode (3 decoders)
cargo bench --bench compare_zenbench                # lossless encode (3 encoders)

# Pixel-exact verification
cargo run --release --example lossless_rt_check     # 24/24 lossless roundtrip

# Profiling
cargo build --release --example profile_decode      # then valgrind --tool=callgrind
cargo build --release --example profile_lossless    # lossless encode profiling
```

---

## 13. File Map

### Decoder v2 (the active decoder):
```
src/decoder/vp8v2/mod.rs              -- DecoderContext, decode loop, public API
src/decoder/vp8v2/context.rs          -- Buffer management, ensure_capacity
src/decoder/vp8v2/header.rs           -- Frame header → FrameTables
src/decoder/vp8v2/tables.rs           -- Precomputed tables (filter, dequant, probs)
src/decoder/vp8v2/coefficients.rs     -- Boolean decoder, coefficient parsing
src/decoder/vp8v2/predict_fused.rs    -- Fused predict+IDCT, #[arcane] per platform
src/decoder/vp8v2/pipeline.rs         -- Per-row filter dispatch
src/decoder/vp8v2/yuv_exact.rs        -- Bit-exact fancy upsample + YUV→RGB SIMD
src/decoder/vp8v2/animation.rs        -- Animated WebP callback API
```

### Decoder v1 (kept for reference, still wired for lossless):
```
src/decoder/vp8/mod.rs                -- Original Vp8Decoder
src/decoder/vp8/coefficients.rs       -- v1 coefficient parsing
src/decoder/vp8/predict.rs            -- v1 prediction dispatch
src/decoder/vp8/cache.rs              -- v1 filter + cache management
src/decoder/vp8/header.rs             -- v1 header parsing
```

### Shared decoder infrastructure:
```
src/decoder/loop_filter.rs            -- Consolidated filter (scalar+SSE2+AVX2+NEON+WASM)
src/decoder/yuv.rs                    -- Consolidated YUV (scalar+SIMD, used by v1)
src/decoder/yuv_fused.rs              -- Fused YUV kernel (used by v1 fast-yuv path)
src/decoder/lossless.rs               -- VP8L lossless decoder
src/decoder/lossless_transform.rs     -- Lossless inverse transforms (scalar)
src/decoder/lossless_transform_simd.rs -- Lossless transforms (SSE2)
src/decoder/bit_reader.rs             -- VP8HeaderBitReader, ActivePartitionReader
src/decoder/api.rs                    -- Public DecodeRequest API
src/decoder/extended.rs               -- VP8X container, alpha, animation
src/decoder/dither.rs                 -- Chroma dithering
```

### Lossless encoder (heavily optimized this session):
```
src/encoder/vp8l/encode.rs            -- Main pipeline
src/encoder/vp8l/entropy.rs           -- LUT tables, histogram cost
src/encoder/vp8l/meta_huffman.rs      -- Histogram clustering
src/encoder/vp8l/backward_refs.rs     -- LZ77, cache selection
src/encoder/vp8l/hash_chain.rs        -- Hash chain
src/encoder/vp8l/cost_model.rs        -- TraceBackwards DP
src/encoder/vp8l/transforms.rs        -- Image transforms with SSE2
```

### Common (shared encoder+decoder):
```
src/common/transform.rs               -- Consolidated IDCT+DCT (all platforms)
src/common/prediction.rs              -- Prediction modes (fixed-size arrays)
src/common/simd_sse.rs                -- SSE2 helpers
src/common/simd_neon.rs               -- NEON helpers
src/common/simd_wasm.rs               -- WASM helpers
```

### Docs:
```
docs/PERFORMANCE.md                   -- Comprehensive benchmark data
docs/CALL-TREE.md                     -- SIMD tier map per function
docs/ARCHITECTURE-CLEANUP.md          -- Code organization plan
docs/LOSSLESS-DECODER-PLAN.md         -- Plan to reach <1.10x
docs/LOSSY-ENCODER-PLAN.md            -- Plan to reach <1.10x
```


## Encoder Optimization (2026-03-28)

Three fixes: I4 early exit, residual cost loop, encode loop cleanup.
Results: 1.47x → **1.32-1.39x** of libwebp C.


## API Surface Audit (needs cleanup)

1. `DecoderContext` is `pub` — make `pub(crate)`, it's internal
2. `decode_rgb_v2()` / `decode_rgba_v2()` on DecodeRequest — remove, v2 is the default via `decode_rgb()`
3. `pub use decoder::vp8` in lib.rs — only needed for TreeNode + diagnostics, not the whole module
4. `pub mod common` — leaks internal types. Only expose what's needed.
5. `pub mod heuristics` — likely internal
6. `test_helpers` gamma LUT accessors — gate behind `#[cfg(test)]`
7. MbRowEntry fields — `pub(super)` not `pub`
8. Remove `image-webp` from dev-dependencies if no benchmarks use it anymore
9. `AnimationFrame` from vp8v2 — should use zencodec's AnimationFrame type
10. Consider: should `pub mod decoder` be `pub(crate) mod decoder` with only re-exports at crate root?
