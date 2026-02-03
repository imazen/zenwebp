# VP8L Lossless Encoder - Context Handoff

**Date:** 2026-02-03
**Status:** Working VP8L encoder with 14 predictor modes. ~1.37x larger than libwebp.
**Branch:** main (clean working tree)
**Latest commits:**
- `0ff608a` docs: update CLAUDE.md with VP8L predictor status
- `5fff3d8` feat: implement full 14 VP8L predictor modes with encoder/decoder parity

## What Works

The VP8L lossless encoder produces **valid, decodable WebP files** verified by both dwebp and our decoder (0 pixel mismatches). All 163 tests pass.

**Implemented:**
- VP8L bitstream format (signature, header, transforms, Huffman-coded data)
- All 14 predictor modes with per-block mode selection (`predictor_bits` default=4 → 16x16 blocks)
- Subtract green transform
- Color indexing (palette) transform for low-color images
- LZ77 backward references with hash chain
- Huffman tree construction, length limiting (max 15), canonical codes
- Complex tree encoding with RLE (codes 16, 17, 18)
- Simple/two-symbol tree encoding
- Quality-dependent LZ77 search depth and window size
- Bitwriter with LSB-first ordering

**Disabled but partially implemented:**
- Color cache (`cache_bits` hardcoded to 0 in `encode.rs:145`)
- Meta-Huffman (infrastructure in `meta_huffman.rs`, not wired into encoder)
- Cross-color transform (enum variant defined, no encoder implementation)

## Remaining Work for Compression Parity

Five items, roughly ordered by expected compression impact:

### 1. Optimized LZ77 Match Finding

**Current state:** Basic hash chain with quality-dependent iteration limit.
**What libwebp does:** Multiple LZ77 strategies (prefix coding selection, backward reference scoring, cost-based reference optimization). libwebp evaluates references by their actual Huffman coding cost, not just match length.
**Expected impact:** HIGH - LZ77 is the primary compression mechanism.

**Key files:**
- `src/encoder/vp8l/backward_refs.rs` - `compute_backward_refs()` at line 74
- `src/encoder/vp8l/hash_chain.rs` - `HashChain::new()` at line 33
- `src/encoder/vp8l/types.rs` - `Vp8lQuality::max_iters()` at line 51

**What to do:**
- Implement cost-based reference selection (prefer references that minimize total Huffman-coded size, not just longest match)
- The `entropy.rs` module has basic entropy estimation functions ready to use
- Consider a two-pass approach: first pass builds histograms, second pass selects references based on estimated costs
- libwebp source: `src/enc/backward_references_enc.c` (BackwardReferencesTraceBackwards, BackwardReferencesHashChainDistanceOnly)

### 2. Color Cache Optimization

**Current state:** `ColorCache` struct fully implemented in `color_cache.rs` with insert/lookup/hash. `estimate_optimal_cache_bits()` exists but is unused. Color cache is disabled (`cache_bits = 0`).
**What libwebp does:** Estimates optimal cache_bits by testing each size and choosing the one that minimizes total coded size.
**Expected impact:** MEDIUM - helps images with repeated colors (icons, screenshots, graphics).

**Key files:**
- `src/encoder/vp8l/color_cache.rs` - Full implementation, `estimate_optimal_cache_bits()` at line 83
- `src/encoder/vp8l/encode.rs` - Disabled at lines 143-153 with TODO comment
- `src/encoder/vp8l/backward_refs.rs` - Already accepts `Option<&mut ColorCache>` at line 80

**What to do:**
- Enable `estimate_optimal_cache_bits()` call in `encode_argb()`
- Wire the cache into backward ref computation (already plumbed through)
- Write the cache_bits signal in the bitstream (code exists but is gated on `cache_bits > 0`)
- Test with icon-like and screenshot images where color cache matters most
- Verify decoder handles cache correctly (it should - our decoder already decodes libwebp files with color cache)

### 3. Cross-Color Transform

**Current state:** `TransformType::CrossColor = 1` enum variant defined. No encoder implementation.
**What libwebp does:** Signals per-block color correlations (green→red, green→blue, red→blue multipliers) in a sub-image, then decorrelates channels. Reduces entropy for photographic content.
**Expected impact:** MEDIUM - helps photos and natural images.

**Decoder reference:** `src/decoder/lossless_transform.rs:355` - `apply_color_transform()` shows the inverse transform. The encoder must produce a matching forward transform.

**Key files to create/modify:**
- `src/encoder/vp8l/transforms.rs` - Add `apply_cross_color_transform()`
- `src/encoder/vp8l/encode.rs` - Wire into transform pipeline (after predictor, before subtract green)

**What to do:**
- Study decoder's `apply_color_transform()` at `src/decoder/lossless_transform.rs:355-400`
- The decoder reads: `green_to_red`, `green_to_blue`, `red_to_blue` multipliers per block
- `color_transform_delta(t: i8, c: i8) -> u32` at `src/decoder/lossless_transform.rs:609` is the core operation
- Encoder needs: for each block, find optimal multipliers by minimizing entropy of residuals
- Write sub-image of multipliers (same encoding as predictor sub-image)
- libwebp source: `src/enc/vp8l_enc.c` (GetBestGreenToRed, GetBestGreenRedToBlue)

### 4. Color Indexing (Palette) Transform Improvements

**Current state:** `ColorIndexTransform` in `transforms.rs` handles basic palette detection and encoding. `try_build()` scans for unique colors (up to 256), `apply()` converts ARGB to indices. The palette is written differentially encoded.
**What libwebp does:** Better palette optimization (sorting, delta encoding, bit packing for small palettes).
**Expected impact:** LOW-MEDIUM - primarily helps low-color images.

**Key files:**
- `src/encoder/vp8l/transforms.rs` - `ColorIndexTransform` at line 359
- `src/encoder/vp8l/encode.rs` - Palette encoding at lines 90-94 and `write_palette()` at line 364

**What to do:**
- Verify palette sorting (currently insertion order; libwebp sorts by luminance)
- Test with actual low-color images (icons, pixel art)
- Verify bit packing: ≤2 colors → 1 bit/pixel, ≤4 → 2 bits, ≤16 → 4 bits (already in `bits_per_pixel()`)
- Verify decoder handles our palette encoding (test with various palette sizes)

### 5. Meta-Huffman (Spatially-Varying Codes)

**Current state:** Infrastructure in `meta_huffman.rs` with `MetaHuffmanConfig`, `build_meta_huffman()` (greedy histogram clustering), and `encode_meta_huffman_image()`. The `entropy.rs` module has `estimate_histogram_bits()` for cost estimation. Not wired into encoder.
**What libwebp does:** Divides image into blocks, builds per-block histograms, clusters similar histograms, uses different Huffman tables for different regions.
**Expected impact:** LOW-MEDIUM - helps heterogeneous images (mixed text/photo).

**Key files:**
- `src/encoder/vp8l/meta_huffman.rs` - Full clustering infrastructure
- `src/encoder/vp8l/entropy.rs` - Cost estimation
- `src/encoder/vp8l/encode.rs` - Disabled at line 156

**What to do:**
- Wire `build_meta_huffman()` into the encoder pipeline
- Generate backward refs per histogram group (or re-assign tokens after clustering)
- Write meta-Huffman sub-image (group indices per block)
- Write multiple Huffman tree sets (one per group)
- Emit image data with group-switching at block boundaries
- This is the most complex remaining feature - save for last

## Architecture Notes

### Transform Pipeline Order
```
Encoder applies:  Predictor → SubtractGreen → (data ready for LZ77+Huffman)
  or:             Palette → (data ready for LZ77+Huffman)
Decoder reverses: SubtractGreen⁻¹ → Predictor⁻¹
  or:             Palette⁻¹
```

Cross-color fits between Predictor and SubtractGreen when implemented.

### Pixel Format
- Encoder internally uses `u32` packed ARGB: `(a << 24) | (r << 16) | (g << 8) | b`
- Helper functions in `types.rs`: `argb_alpha()`, `argb_red()`, `argb_green()`, `argb_blue()`, `make_argb()`
- Decoder uses `u8` slices: `[R, G, B, A]` at `index * 4`

### Huffman Channel Layout
Five channels per Huffman group:
1. **Literal/length** (green channel + length prefix codes + cache codes): alphabet = 256 + 24 + cache_size
2. **Red**: alphabet = 256
3. **Blue**: alphabet = 256
4. **Alpha**: alphabet = 256
5. **Distance**: alphabet = 40

### Sub-Image Encoding Pattern
Predictor modes, cross-color multipliers, and meta-Huffman indices are all encoded as small images:
- Dimensions: `subsample_size(width, bits)` x `subsample_size(height, bits)`
- Each "pixel" stores transform parameters in ARGB channels
- Encoded with their own Huffman trees (5 trees per sub-image)
- Pattern is in `write_predictor_image()` at `encode.rs:317` - reuse for cross-color and meta-Huffman

### Testing Pattern
```bash
# Run all tests
cargo test --release

# Run VP8L-specific predictor test
cargo run --release --example test_pred_512

# Run noise compression test
cargo run --release --example test_noise
```

Validation approach: encode → decode with our decoder → compare pixels (0 mismatches required). Also validate with dwebp when available at `/home/lilith/work/libwebp/examples/dwebp`.

### Known Encoder/Decoder Parity Issues (FIXED)
1. **Select predictor tie-breaking:** Decoder chooses `top` on ties, encoder must match (fixed in `5fff3d8`)
2. **TopRight edge wrapping:** At x=width-1, decoder wraps to first pixel of current row (fixed in `5fff3d8`)
3. **Tree depth off-by-one:** Starts at 0, not 1 (fixed earlier)
4. **Kraft inequality:** `ensure_valid_code_lengths()` fixes after length clamping (fixed earlier)

## Quick Reference: File Locations

| File | Purpose | Key Functions |
|------|---------|---------------|
| `src/encoder/vp8l/encode.rs` | Main pipeline | `encode_vp8l()`, `encode_argb()`, `write_predictor_image()` |
| `src/encoder/vp8l/transforms.rs` | Transforms | `apply_predictor_transform()`, `choose_best_predictor()`, `apply_subtract_green()`, `ColorIndexTransform` |
| `src/encoder/vp8l/backward_refs.rs` | LZ77 | `compute_backward_refs()`, `compute_backward_refs_simple()` |
| `src/encoder/vp8l/hash_chain.rs` | Match finding | `HashChain::new()`, hash function |
| `src/encoder/vp8l/histogram.rs` | Frequency stats | `Histogram::from_refs()`, `length_to_code()`, `distance_code_to_prefix()` |
| `src/encoder/vp8l/huffman.rs` | Huffman coding | `build_huffman_lengths()`, `build_huffman_codes()`, `write_huffman_tree()` |
| `src/encoder/vp8l/bitwriter.rs` | Bit output | `BitWriter::write_bits()`, `write_bit()` |
| `src/encoder/vp8l/color_cache.rs` | Color cache | `ColorCache::insert()`, `lookup()`, `estimate_optimal_cache_bits()` |
| `src/encoder/vp8l/entropy.rs` | Cost estimation | `bits_entropy()`, `estimate_histogram_bits()` |
| `src/encoder/vp8l/meta_huffman.rs` | Multi-Huffman | `build_meta_huffman()`, `cluster_histograms()` |
| `src/encoder/vp8l/types.rs` | Data structures | `Vp8lConfig`, `PixOrCopy`, `BackwardRefs`, `Vp8lQuality` |
| `src/decoder/lossless_transform.rs` | Decoder transforms | `apply_color_transform()` (cross-color reference) |
| `src/decoder/lossless.rs` | Decoder pipeline | Sub-image reading, Huffman decoding |
