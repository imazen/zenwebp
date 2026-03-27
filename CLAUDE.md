# zenwebp CLAUDE.md

See global ~/.claude/CLAUDE.md for general instructions.
Historical investigation notes and resolved bugs are in [LOG.md](LOG.md).

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

**Speed (criterion, 792079.png 512x512, Q75, 2026-02-05):**
| Method | zenwebp | libwebp | Ratio |
|--------|---------|---------|-------|
| 0 | 7.0ms | 2.7ms | 2.6x |
| 4 | 15.0ms | 10.2ms | **1.47x** |
| 6 | 22.2ms | 15.5ms | **1.43x** |

*Instruction ratio is 1.12x (179M vs ~161M) but wall-clock is ~1.47x — gap is memory access patterns.*

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

Total: 229.3M -> 182.0M -> ~179M per decode.
Wall-clock: ~2.3x -> ~1.26-1.36x vs libwebp on codec_wiki.

**Remaining instruction gap breakdown (per decode, codec_wiki):**

| Category | zenwebp (M) | libwebp (M) | Gap (M) |
|----------|-------------|-------------|---------|
| Coefficient parsing | 42.6 | ~38.3 | 4.3 |
| Loop filter SIMD | 37.9 | ~30.6 | 7.3 |
| YUV->RGB upsample | 45.7 | ~38.1 | 7.6 |
| Decode orchestration | 22.9 | ~24.9 | **-2.0** |
| IDCT | 1.3 | ~17.1 | **-15.8** |
| memset (buf alloc) | 17.8 | ~0 | 17.8 |
| memcpy | 3.2 | 2.6 | 0.6 |
| Other | 9.4 | ~10 | -0.6 |
| **Total** | **~180** | **161.4** | **~19** |

Main remaining opportunities:
- **memset 17.8M**: Frame buffer zero-init. Every byte overwritten before read.
  Could use uninitialized allocation but requires unsafe or refactoring.
- **Coeff parsing 4.3M excess**: Prob table bounds checks, zigzag/dequant indexing.
- **Loop filter 7.3M excess**: SIMD dispatch overhead, bounds checks.
- **YUV->RGB 7.6M excess**: Scalar edge handling, bounds checks.

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

### Encoder (method 4, 2026-02-05, 179M total)
| Function | % | M instr | Notes |
|----------|---|---------|-------|
| evaluate_i4_modes_sse2 | 20.8% | 37.2M | I4 inner loop (inlines quant/dequant/SSE) |
| get_residual_cost_sse2 | 12.1% | 21.6M | Coefficient cost estimation |
| encode_image | 12.1% | 21.6M | Main encoding loop |
| choose_macroblock_info | 10.6% | 18.9M | I16/UV mode selection |

**vs libwebp function-level (2026-02-05):**
| zenwebp | M instr | libwebp | M instr | Ratio |
|---------|---------|---------|---------|-------|
| evaluate_i4 + choose_mb + pick_i4 | 61.7M | PickBestIntra4 + quant_enc | 18.3M | 3.4x |
| get_residual_cost_sse2 | 22.9M | GetResidualCost_SSE2 | 9.7M | 2.4x |
| idct_add_residue + idct4x4 | 8.6M | ITransform_SSE2 | 15.9M | **0.54x** |
| ftransform2 + ftransform_from_u8 | 9.6M | FTransform_SSE2 | 9.8M | **0.98x** |
| quantize_block (standalone) | 4.3M | QuantizeBlock_SSE41 | 6.0M | **0.72x** |

### Decoder (2026-03-25, codec_wiki 2560x1664 Q75 RGB, 10 decodes)

zenwebp: 182.0M vs libwebp: 161.4M (**1.13x instruction ratio**)

| zenwebp function | M instr | libwebp equivalent | M instr | Ratio |
|-----------------|---------|-------------------|---------|-------|
| read_residual_data | 42.6M | GetCoeffsFast+GetLargeValue | 38.3M | **1.11x** |
| filter_row_simd | 37.9M | HFilter/VFilter/DoFilter SSE2 | 30.6M | **1.24x** |
| fancy_upsample + fill_row | 45.7M | YUV2RGB+Upsample SSE41 | 38.1M | **1.20x** |
| decode_frame_ (exclusive) | 22.9M | VP8DecodeMB+Reconstruct+ParseMode | 24.9M | **0.92x** |
| memset | 17.6M | — | ~0 | buffer zeroing |
| IDCT | 1.3M | Transform_SSE2+DC prediction | 17.1M | **0.08x** |
| memcpy | 3.2M | memcpy | 2.6M | 1.23x |

IDCT is 18.5x faster than libwebp (zero-block skip + DC-only WHT).
Main remaining targets: memset (buffer alloc), bounds checks in coeff
parsing / loop filter / YUV->RGB conversion (~7-8M excess each).

## Remaining Optimization Opportunities

### Encoder
1. **Mode selection 3.4x vs libwebp** — I4 inner loop orchestration overhead
2. **Residual cost 2.4x** — tighter inner loop possible
3. **Wall-clock 1.47x despite 1.12x instructions** — memory access patterns
4. **Defer I16 reconstruction** — only IDCT winning mode (saves ~48 IDCT/MB)

### Decoder
1. **IDCT skip (DONE)** — Per-block non_zero_blocks bitmap eliminates IDCT for zero
   blocks. 24.5M -> 1.3M per decode. Matches libwebp's DoTransform case 0 / bits!=0.
2. **Loop filter 37.9M vs 30.6M (1.24x)** — single `#[arcane]` entry with `#[rite]`
   inlining. Remaining gap from bounds checks and different code shape.
3. **Coefficient parsing 42.6M vs 38.3M (1.11x)** — bounds checks on prob table
   lookups, zigzag indexing, dequant array access. Bit reader load_new_bytes bounds
   checks eliminated (sub-slice pattern), state pointer indirection eliminated
   (inline fields + Drop writeback). Was 47.0M, now 42.6M.
4. **YUV->RGB 45.7M vs 38.1M (1.20x)** — scalar edge handling, bounds checks.
5. **memset 17.8M** — frame buffer zero-init. Every byte is overwritten before read.
6. **decode_frame_ orchestration** — now at parity with libwebp (~22.9M vs ~24.9M).

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
