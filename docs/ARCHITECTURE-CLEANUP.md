# Architecture Cleanup Plan

## Problem: Optimization is hard because the code is scattered

### 1. Per-platform SIMD files (15 files, ~8000 lines of near-duplicates)

The same algorithm exists in 3-4 files with different intrinsics:

| Algorithm | Scalar | SSE2/AVX2 | NEON | WASM | Total lines |
|-----------|--------|-----------|------|------|-------------|
| Loop filter | loop_filter.rs | loop_filter_avx2.rs (3639!) | loop_filter_neon.rs (1321) | loop_filter_wasm.rs (745) | 5705+ |
| IDCT/transform | transform.rs | transform_simd_intrinsics.rs | transform_aarch64.rs | transform_wasm.rs | ~2500 |
| YUV→RGB | yuv.rs | yuv_simd.rs | yuv_neon.rs | yuv_wasm.rs | ~3300 |

**The problem**: When optimizing an algorithm, you have to understand and potentially modify 4 files. The scalar and SIMD versions can drift apart. Bug fixes need to be applied in all variants.

**The fix**: Each algorithm gets ONE file. Platform variants live in the same file using `#[rite(v3, neon, wasm128)]` multi-tier where possible, or `#[cfg(target_arch)]` blocks within the same file where intrinsics differ. The dispatch function and all its variants are visible together.

Example structure:
```
src/decoder/loop_filter.rs       # ALL filter code: scalar + dispatch + platform SIMD
src/decoder/idct.rs              # ALL IDCT: scalar + SSE2 + NEON + WASM
src/decoder/yuv_convert.rs       # ALL YUV: scalar + SIMD
```

### 2. Decoder state is a monolith (vp8.rs = 2087 lines)

`Vp8Decoder` struct has ~40 fields mixing:
- Parse state (bit reader, partitions)
- Frame metadata (width, height, filter settings)
- Working buffers (cache, borders, coeff_blocks)
- Output buffers (frame ybuf/ubuf/vbuf)

Everything is in one `impl` block making it hard to see which functions touch which state.

**The fix**: Split by concern:
```
src/decoder/vp8/mod.rs           # Vp8Decoder struct, decode_frame_ orchestration
src/decoder/vp8/coefficients.rs  # read_residual_data, read_coefficients, get_large_value
src/decoder/vp8/predict.rs       # intra_predict_luma/chroma, border updates
src/decoder/vp8/cache.rs         # filter_row_in_cache, output_row_from_cache, rotate
src/decoder/vp8/header.rs        # read_frame_header, read_segment_updates, read_partitions
```

### 3. Four bit reader structs

- `VP8BitReader<'a>` — generic boolean decoder
- `VP8HeaderBitReader` — header parsing (owns data)
- `PartitionReader<'a>` — wraps VP8BitReader with partition bounds
- `ActivePartitionReader<'a>` — inlined state fields, Drop writeback

**The problem**: The hot path uses `ActivePartitionReader` which was created as an optimization of `PartitionReader`. The original `VP8BitReader` is still used elsewhere. `VP8HeaderBitReader` is a separate implementation of the same algorithm with different ownership.

**The fix**: Keep `ActivePartitionReader` (the fast one) as THE bit reader. Make `VP8HeaderBitReader` use the same core. Remove `PartitionReader` wrapper — `ActivePartitionReader` already handles partition state directly.

### 4. 21 decode entry points (api.rs = 2219 lines)

```
decode_rgba, decode_rgb, decode_bgra, decode_bgr, decode_argb
decode_rgba_into, decode_rgb_into, decode_bgra_into, decode_bgr_into, decode_argb_into
decode_rgba_premultiplied, decode_bgra_premultiplied, decode_argb_premultiplied
decode_yuv420, decode_rgb565, decode_rgba4444
decode (auto-detect format)
```

Plus standalone functions that duplicate the `DecodeRequest` methods.

**The problem**: Every format variant has its own allocation + conversion code. The lossless decoder outputs RGBA internally, but many paths allocate a scratch buffer and convert. Any optimization to the decode path needs to be applied to all 21 functions.

**The fix**: One internal `decode_to_rgba()` that always returns RGBA. One `convert_rgba_to(format, src, dst)` function that handles all format conversions. Each public function becomes a 3-liner: decode + convert + return. The standalone functions just call the `DecodeRequest` methods.

### 5. Prediction code split across 3 files

- `common/prediction.rs` — scalar prediction modes (shared encoder/decoder)
- `decoder/predict_simd.rs` — SIMD prediction+IDCT dispatch (decoder only)
- `common/transform.rs` / `transform_simd_intrinsics.rs` — IDCT

The prediction+IDCT pipeline is logically one thing (predict → add residue) but the code is scattered across common/ and decoder/.

**The fix**:
```
src/decoder/vp8/predict.rs       # Prediction dispatch + SIMD pipeline
src/common/prediction.rs         # Shared prediction mode implementations
src/common/idct.rs               # IDCT (replaces transform*.rs for inverse)
src/encoder/vp8/transform.rs     # Forward DCT (encoder only)
```

## Priority Order

1. **Consolidate per-platform SIMD files** — biggest win for optimization clarity
2. **Split vp8.rs** — makes hot path analysis tractable
3. **Simplify decode API** — one decode path, format conversion at the end
4. **Unify bit readers** — one implementation, not four
5. **Separate forward/inverse transforms** — encoder DCT vs decoder IDCT

## What NOT to change

- The encoder vp8l/ directory is well-organized (entropy.rs, backward_refs.rs, etc.)
- The mux/ directory is clean
- The common/types.rs constants are fine where they are
- `#[inline(never)]` on read_coefficients is deliberate (BTB aliasing fix)
- `#[cold]` on scalar fallbacks is deliberate (code size)
