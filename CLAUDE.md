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
| libwebp-encoded | 8.40ms (46.8 MPix/s) | 3.26ms (120.6 MPix/s) | 2.58x slower |
| our-encoded | 8.45ms (46.5 MPix/s) | 3.36ms (117.0 MPix/s) | 2.51x slower |

*Benchmark: 768x512 Kodak image, 100 iterations, release mode*

Our decoder is ~2.5x slower than libwebp. This is expected for a pure Rust
implementation without SIMD. Key areas for optimization:
- IDCT reconstruction (SIMD)
- Boolean arithmetic decoding
- Loop filter application

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
- `src/yuv_simd.rs` - SSE4.1 YUV→RGB (8 pixels at once)
  - **Integrated** for Simple upsampling mode (non-default)
  - Uses exact same formula as scalar code (bit-exact)
  - Feature-gated: `unsafe-simd` feature + x86_64 + SSE4.1 detected
- `src/loop_filter_simd.rs` - SSE4.1 loop filter (4 edges at once)
  - **Not integrated** - requires restructuring decoder to batch operations

### TODO
- [ ] Integrate SIMD loop filter into decoder (requires batching)
- [ ] Add SIMD YUV for Bilinear (fancy) upsampling (default mode)
- [ ] Consider SIMD for choose_macroblock_info inner loops (encoder)
- [ ] Profile get_residual_cost for optimization opportunities

## Known Bugs

(none currently)

## Investigation Notes

(none currently)

## User Feedback Log

(none currently)
