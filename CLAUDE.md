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
- `src/encoder/color_quantize.rs` - imagequant integration (`quantize` feature)

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

### Decoder vs libwebp

| Corpus | Ratio |
|--------|-------|
| CLIC2025 (10 images) | **0.93-1.25x** (avg ~1.15x slower) |
| 1024×1024 lossy | **1.32x slower** (267 vs 351 MPix/s) |

## Perceptual Encoder Features (method 3+)

- **Enhanced CSF tables** (method 3+) — best quality improvement alone
- **SATD-based masking** (method 4+) — texture, luminance, edge, uniformity
- **JND thresholds** (method 5+) — frequency-dependent coefficient zeroing
- **Psy-RD disabled** — hurts butteraugli (prefers smoother reconstructions)

Files: `src/encoder/psy.rs`, `src/encoder/trellis.rs`

## SIMD Architecture

### Encoder SIMD (archmage 0.5)

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

**`unchecked` feature:** Eliminates ALL bounds checks in loop filter hot paths via raw pointers.
Only for trusted input. ~1.5% additional wall time improvement.

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

### Decoder (2026-02-05)
| Category | % Time | Notes |
|----------|--------|-------|
| Coefficient reading | ~10% | read_coefficients_inline (optimized) |
| YUV→RGB | ~16% | fancy upsampling with SIMD |
| Loop filter | ~12% | 9 SIMD filter functions |
| memset | ~5% | Output buffer zero-init |

## Remaining Optimization Opportunities

### Encoder
1. **Mode selection 3.4x vs libwebp** — I4 inner loop orchestration overhead
2. **Residual cost 2.4x** — tighter inner loop possible
3. **Wall-clock 1.47x despite 1.12x instructions** — memory access patterns
4. **Defer I16 reconstruction** — only IDCT winning mode (saves ~48 IDCT/MB)

### Decoder
1. **Loop filter overhead** — still 2.5x more instructions than libwebp despite SIMD;
   per-pixel threshold checks are expensive. Consider batch threshold computation.
2. **YUV→RGB 1.69x slower** — `planar_to_24b` does 5 permutation passes; SSSE3 shuffle
   fails due to archmage function call overhead instead of inline pshufb.

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
Exception: `unchecked` feature for loop filter hot paths.

## Diagnostic Examples

| Example | Usage |
|---------|-------|
| `corpus_test [dir]` | Batch file size comparison vs libwebp |
| `compare_all_methods` | Per-method size comparison |
| `compare_coefficients` | Quantized level comparison |
| `compare_i4_modes` | Per-block I4 mode choice comparison |
| `compare_rd_costs` | Macroblock type agreement stats |
| `debug_mode_decision` | MB_DEBUG env for mode selection |
| `cache_test` | Lossless cache size testing |
| `lossless_benchmark` | Lossless corpus benchmark |

Run with: `cargo run --release --example <name> [args]`

## Known Bugs

(none currently)

## User Feedback Log

(none currently)

## API Convergence TODOs

See `/home/lilith/work/zendiff/API_COMPARISON.md` for full cross-codec comparison.

**Three-layer pattern: EncoderConfig → EncodeRequest<'a> → Encoder (streaming only)**

Done:
- [x] `EncodeError`/`DecodeError` naming ✓
- [x] `EncodeStats` naming ✓
- [x] `At<>` error wrapping ✓
- [x] `Limits` struct (decode side) ✓
- [x] `&dyn Stop` cancellation ✓
- [x] `#[non_exhaustive]` on errors ✓
- [x] `EncodeRequest<'a>` intermediate layer ✓
- [x] Metadata on request, not config ✓

**No backwards compatibility required** — we have no external users. Just bump the 0.x major version for breaking changes. No deprecation shims or legacy aliases — delete old APIs.

**Builder convention**: `with_` prefix for consuming builder setters, bare-name for getters. Config and Request setters use `with_foo(mut self, val) -> Self`. Getters use `foo(&self) -> T`.

**Project standards**: `#![forbid(unsafe_code)]` with default features. no_std+alloc (minimum: wasm32). CI with codecov. README with badges and usage examples. As of Rust 1.92, almost everything is in `core::` (including `Error`) — don't assume `std` is needed. Use `wasmtimer` crate for timing on wasm.

Remaining:
- [x] Rename `finish()` → `encode()` on `EncodeRequest` (one-shot, nothing was "started")
- [x] Rename `finish_into()` → `encode_into()`, `finish_to()` → `encode_to()` on request
- [x] Remove deprecated aliases (`encode()`, `encode_into()`, `encode_to_writer()`)
- [N/A] Add `EncodeRequest::build()` → streaming `Encoder` with `push()`/`finish()` (see "Why Streaming Encoding" section)
- [x] Add `Limits` on encode side (currently decode-only)
- [x] Replace `ColorType` with `PixelLayout` (or rename — same concept, just naming)
- [x] `Limits` fields: standardize to `Option<u64>`
- [x] Split `EncoderConfig` into `LossyConfig` / `LosslessConfig` (compile-time invalid state prevention) ✓ 2026-02-06
- [ ] Add `estimate_memory()` / `estimate_memory_ceiling()` on both config types
- [ ] Factor metadata into `ImageMetadata` struct (keep request clean)
- [ ] Adopt `with_` prefix convention for all builder setters on Config/Request
- [ ] Support `Rgba8` and `Bgra8` for both encode and decode (A=255 on decode for lossy, ignore A on encode)
- [ ] Add probing: `ImageInfo::from_bytes(&[u8])` static probe with `PROBE_BYTES` constant
- [ ] Two-phase decoder: `build()` parses header → `info()` inspects → `decode()` continues without re-parsing

## Why Streaming Encoding Doesn't Make Sense for WebP

**TL;DR:** WebP encoding algorithms require the entire image before encoding can start. A streaming/push-based API would use MORE memory, not less.

### Algorithmic Requirements

**VP8L (Lossless):**
- Backward references (LZ77) - needs to look back at all previous pixels
- Palette detection - must see all colors to build palette
- Predictor/color transforms - needs whole-image analysis to choose best transform
- Huffman coding - requires complete frequency statistics from entire image
- **Fundamentally cannot encode incrementally**

**VP8 (Lossy):**
- Segmentation - analyzes all macroblocks to cluster into 4 groups
- SNS (spatial noise shaping) - global texture analysis
- Filter strength - computed from whole-image statistics
- Rate-distortion optimization - benefits from seeing all data
- Could theoretically encode row-by-row, but quality would be significantly worse
- **Requires full image for good quality**

**Animation:**
- Each frame must be fully encoded before moving to next
- `AnimationEncoder::add_frame()` already provides frame-by-frame streaming
- **Already has the right API**

### Memory Analysis

**Current one-shot API:**
```rust
let img = vec![0u8; w * h * 4];  // User owns
EncodeRequest::new(&config, &img, ...).encode()?;  // We borrow
// Total: user buffer + working buffers
```

**Hypothetical streaming API:**
```rust
let mut encoder = EncodeRequest::build()?;
for row in rows {
    encoder.push(row)?;  // Must accumulate internally
}
encoder.finish()?;
// Total: our internal buffer (duplicate) + same working buffers
```

**Streaming would use MORE memory** because:
1. We'd need to buffer all rows internally (algorithms need complete image)
2. This duplicates the user's data
3. Same working buffers needed for actual encoding
4. Only saves memory if user generates procedurally and discards chunks
   - But user could just `Vec::extend()` themselves with same result

### Conclusion

The current one-shot `EncodeRequest::encode()` is the **optimal API** for WebP:
- User provides complete image (by reference, no copy)
- We encode it in one pass with necessary working buffers
- Minimal memory overhead

**Streaming encoder task marked as N/A** - doesn't fit WebP's algorithmic requirements.
