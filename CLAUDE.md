# image-webp CLAUDE.md

See global ~/.claude/CLAUDE.md for general instructions.

## Current Optimization Status (2026-01-22)

### Encoder Performance vs libwebp

**Method Parameter Added** - Speed/quality tradeoff (0-6):
| Method | Time | Throughput | File Size | Notes |
|--------|------|------------|-----------|-------|
| 0 | 55ms | 7.15 MPix/s | 101KB | I16-only, no trellis |
| 2 | 79ms | 4.97 MPix/s | 87KB | Limited I4 (3 modes), no trellis |
| 4 | 92ms | 4.27 MPix/s | 78KB | Balanced (4 modes), trellis |
| 6 | 122ms | 3.21 MPix/s | 76KB | Full search (10 modes), trellis |

*Benchmark: 768x512 Kodak image at Q75, 10 iterations, release mode*

### Recent SIMD Optimizations
- DCT/IDCT: SIMD i32/i16 conversion (13% speedup)
- t_transform: SIMD Hadamard for spectral distortion
- SSE4x4: SIMD distortion calculation

### Profiler Hot Spots (method 4)
| Function | % Runtime | Notes |
|----------|-----------|-------|
| choose_macroblock_info | 33.82% | Mode selection RD loop |
| get_residual_cost | 9.38% | Coefficient cost estimation |
| DCT | 9.29% | Forward transform |
| trellis_quantize_block | 7.43% | RD-optimized quantization |
| IDCT | 6.53% | Inverse transform (SIMD) |
| t_transform | 3.24% | Spectral distortion (SIMD) |

### Quality vs libwebp
- File sizes: 1.17-1.41x of libwebp (down from 2.5x before I4)
- PSNR gap: ~1.35 dB behind at equal BPP

### Key Files
- `src/vp8_encoder.rs` - Main encoder, mode selection
- `src/vp8_cost.rs` - Cost estimation, trellis quantization
- `src/simd_sse.rs` - SIMD implementations
- `src/encoder.rs` - Public API, EncoderParams

### Decoder Performance vs libwebp (2026-01-22)

| Test | Our Decoder | libwebp | Speed Ratio |
|------|-------------|---------|-------------|
| libwebp-encoded | 7.2ms (55 MPix/s) | 2.8ms (140 MPix/s) | 2.6x slower |
| our-encoded | 6.9ms (57 MPix/s) | 2.7ms (144 MPix/s) | 2.5x slower |

*Benchmark: 768x512 Kodak image, 100 iterations, release mode*

Our decoder is ~2.5x slower than libwebp. Key bottleneck is the boolean
arithmetic decoder (~24% of time). Recent optimizations:
- Lookup table for range normalization (replaces leading_zeros computation)
- SIMD fancy upsampling for YUV→RGB conversion
- AVX2 loop filter (16 pixels at once) - not yet fully integrated

### Decoder Profiler Hot Spots
| Function | % Time | Notes |
|----------|--------|-------|
| read_with_tree_with_first_node | 23.77% | Arithmetic decoder (hard to SIMD) |
| should_filter_vertical | 4.98% | Loop filter threshold check |
| fill_row_fancy_with_2_uv_rows | 4.13% | YUV upsampling + conversion |
| subblock_filter_horizontal | 4.10% | Loop filter application |
| idct4x4_simd | 3.73% | Already SIMD |
| Loop filter total | ~15% | Multiple functions |

### SIMD Decoder Optimizations
- `src/yuv_simd.rs` - SSE4.1 YUV→RGB with fancy upsampling
  - **Integrated** for both Simple and Fancy (bilinear) upsampling modes
  - Uses `_mm_avg_epu8` for efficient bilinear interpolation
  - Feature-gated: `unsafe-simd` feature + x86_64 + SSE4.1 detected
- `src/loop_filter_avx2.rs` - SSE4.1 loop filter (16 pixels at once)
  - Uses transpose technique for horizontal filtering
  - Integrated for simple filter path, not yet for normal filter
- `src/vp8_arithmetic_decoder.rs` - Lookup table for range normalization
  - VP8_SHIFT_TABLE replaces `leading_zeros()` computation
  - ~15% speedup in arithmetic decoding

### TODO
- [ ] Integrate SIMD normal/macroblock filter (DoFilter4/DoFilter6) - ~11% opportunity
- [ ] Consider SIMD for choose_macroblock_info inner loops (encoder)
- [ ] Profile get_residual_cost for optimization opportunities

## Known Bugs

(none currently)

## Investigation Notes

### VP8BitReader Investigation (2026-01-22)

Attempted to implement libwebp-style VP8BitReader to replace ArithmeticDecoder
for coefficient reading. **Result: New implementation was ~10% SLOWER**.

Key findings:
1. The existing `FastDecoder` in `vp8_arithmetic_decoder.rs` is already well-optimized:
   - Pre-chunked `[[u8; 4]]` storage enables fast 4-byte loads via `u32::from_be_bytes`
   - Speculative reading (reads past end with defaults, validates at commit)
   - State committed only once at the end of each tree read
   - Lookup table normalization already in place

2. The new VP8BitReader added overhead from:
   - Contiguous byte storage requiring slice operations for 7-byte loads
   - `try_into().unwrap()` pattern for array conversion
   - Reader creation/state save pattern on every coefficient read

3. Profile comparison:
   - Before: `read_with_tree_with_first_node` at 24% (both mode parsing + coefficients)
   - After: `read_coefficients` at 28.5%, `read_with_tree_with_first_node` at 3.5%
   - Total arithmetic decoding time INCREASED despite splitting the paths

**Conclusion**: Don't replace ArithmeticDecoder. Focus on other bottlenecks:
- Loop filter (~11%): SIMD normal filter could help significantly
- YUV conversion (~4%): Already has SIMD, may be near optimal

### Loop Filter Optimization Opportunity

The normal filter path (when `filter_type == false`) processes rows individually:
```rust
for y in 0..16 {
    loop_filter::subblock_filter_horizontal(...);
}
```

SIMD opportunity: Process 16 rows at once like the simple filter does.
Current SIMD exists in `loop_filter_avx2.rs` but only for simple filter.
Adding SIMD normal filter (DoFilter4/DoFilter6) could save ~5-8% of total time.

## User Feedback Log

(none currently)
