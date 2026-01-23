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
| libwebp-encoded | 5.85ms (67 MPix/s) | 3.0ms (130 MPix/s) | 1.93x slower |
| our-encoded | 5.74ms (68 MPix/s) | 2.99ms (131 MPix/s) | 1.92x slower |

*Benchmark: 768x512 Kodak image, 100 iterations, release mode*

Our decoder is ~1.92x slower than libwebp (improved from 2.5x baseline). Recent optimizations:
- **copy_from_slice for macroblock output** (~1-2% speedup)
- **SIMD normal loop filter for vertical edges** (~10% speedup, commit 3bf30f1)
- **libwebp-rs style bit reader for coefficients** (16% speedup, commit 5588e44)
- **Inlined read_tree/is_eof** (~7% additional speedup)
- SIMD fancy upsampling for YUV→RGB conversion
- AVX2 loop filter (16 pixels at once) - simple filter only

### Decoder Profiler Hot Spots (after libwebp-rs bit reader)
| Function | % Time | Notes |
|----------|--------|-------|
| read_coefficients | 18.85% | Coefficient decoding (down from 24%) |
| idct4x4_avx512 | 6.05% | Already SIMD |
| should_filter_vertical | 5.56% | Loop filter threshold check |
| decode_frame_ | 5.20% | Frame processing overhead |
| Loop filter total | ~12% | Multiple functions |

### Cache/Branch Analysis (2026-01-22)
Per-decode comparison with libwebp:
| Metric | Ours | libwebp | Ratio |
|--------|------|---------|-------|
| Instructions | 449M | 99M | 4.5x more |
| Branch misses | 3.81% | 8.15% | Better! |
| Cache misses | 14.57% | 5.01% | **3x worse rate** |
| L1 misses | 4.65M | 256K | **18x more** |

**Main bottleneck is cache efficiency, not branch prediction.**

### SIMD Decoder Optimizations
- `src/yuv_simd.rs` - SSE4.1 YUV→RGB with fancy upsampling
  - **Integrated** for both Simple and Fancy (bilinear) upsampling modes
  - Uses `_mm_avg_epu8` for efficient bilinear interpolation
  - Feature-gated: `unsafe-simd` feature + x86_64 + SSE4.1 detected
- `src/loop_filter_avx2.rs` - SSE4.1 loop filter (16 pixels at once)
  - **Vertical normal filter SIMD integrated** for luma (DoFilter4/DoFilter6)
  - Uses transpose technique for horizontal filtering (simple filter only)
  - Horizontal normal filter still TODO
- `src/vp8_bit_reader.rs` - libwebp-rs style bit reader for coefficients
  - Uses VP8GetBitAlt algorithm with 56-bit buffer
  - `leading_zeros()` for normalization (single LZCNT instruction)
  - **16% speedup** in overall decode (commit 5588e44)

### TODO
- [x] ~~Replace byte-by-byte copy with `copy_from_slice()` (~1-2% gain)~~
- [ ] **High impact**: Add row cache for better loop filter cache locality
- [ ] Add SIMD horizontal normal filter (transpose technique like simple filter)
- [ ] Consider using libwebp-rs bit reader for mode parsing too (self.b field)
- [ ] Consider SIMD for choose_macroblock_info inner loops (encoder)
- [ ] Profile get_residual_cost for optimization opportunities

## Known Bugs

(none currently)

## Investigation Notes

### VP8BitReader Success (2026-01-22)

Initial attempt to replace ArithmeticDecoder was slower. Second attempt using
libwebp-rs's exact algorithm succeeded with **16% speedup**.

Key differences from initial failed attempt:
1. Used libwebp's VP8GetBitAlt algorithm exactly (not our modified version)
2. Stored `range - 1` internally (127-254 interval) matching libwebp
3. Used `leading_zeros()` which compiles to single LZCNT instruction
4. Simpler split calculation: `split = (range * prob) >> 8`

Verified algorithms produce identical bit streams (0 differences across all probabilities).

Results:
- Micro-benchmarks: 24-37% faster bit reading
- Full decoder: 55 → 62 MPix/s (16% speedup)
- Gap vs libwebp: 2.5x → 2.15x slower

The ArithmeticDecoder is still used for header/mode parsing (self.b field).
Only coefficient reading uses the new VP8Partitions/PartitionReader.

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

### libwebp-rs Mechanical Translation Comparison (2026-01-22)

Profiled `~/work/webp-porting/libwebp-rs` - a direct Rust port of libwebp:

| Decoder | Arith Decoder | Loop Filter | Overall |
|---------|--------------|-------------|---------|
| image-webp (ours) | 24% | ~11% | 2.5x slower |
| libwebp-rs (mechanical) | 11% | 24% | 2.6x slower |
| libwebp C | baseline | baseline | 1.0x |

Key finding: **Both Rust decoders have similar ~2.5x slowdown** despite different
bottleneck distributions. The mechanical translation uses rayon parallelism and
libwebp's exact bit reader algorithm, yet achieves similar performance to our
hand-written decoder.

libwebp-rs uses:
- libwebp's VP8GetBitAlt algorithm with `range - 1` representation
- `leading_zeros()` for normalization (not lookup table)
- SIMD for macroblock edge filters but scalar for inner edge (`filter_loop24`)
- rayon parallelism (adds ~18% overhead)

The inner edge filter `filter_loop24` is 24% of their time - scalar loop over
16 pixels. Same bottleneck pattern as our decoder.

Conclusion: The ~2.5x gap may be inherent to Rust vs C for this workload, or
requires deeper investigation into codegen/memory access patterns.

### Cache/Memory Layout Analysis (2026-01-22)

**Root cause of 18x more L1 cache misses identified.**

#### libwebp Architecture
```
1. Decode macroblock → 832-byte yuv_b working buffer (BPS=32, cache-aligned)
2. Copy via WEBP_UNSAFE_MEMCPY → row cache (cache_y/u/v)
   - cache_y_stride = 16 * mb_w (contiguous row of macroblocks)
3. Loop filter operates on row cache (contiguous data)
4. At row completion, FinishRow() outputs to final destination
```

Code from `frame_dec.c:195`:
```c
for (j = 0; j < 16; ++j) {
    WEBP_UNSAFE_MEMCPY(y_out + j * dec->cache_y_stride, y_dst + j * BPS, 16);
}
```

#### Our Architecture
```
1. Decode macroblock → 544-byte working buffer (same stride=32)
2. Copy byte-by-byte → final ybuf/ubuf/vbuf (full image size)
   - stride = full image width (e.g., 768 for 768x512 image)
3. Loop filter operates on final buffers (scattered access)
4. No row cache - immediate output
```

Code from `vp8.rs:889`:
```rust
for y in 0usize..16 {
    for (ybuf, &ws) in self.frame.ybuf[(mby * 16 + y) * mw * 16 + mbx * 16..][..16]
        .iter_mut()
        .zip(ws[(1 + y) * stride + 1..][..16].iter())
    {
        *ybuf = ws;
    }
}
```

#### Problems Identified

1. **No row cache** - We write directly to final buffers (MAIN ISSUE)
   - When filtering top edge of MB(x, y), we need data from row y-1
   - By then, we've processed 48 macroblocks (full row) since touching row y-1
   - With ~1KB touched per MB, that's 48KB of intervening data
   - Exceeds L1 cache (32KB), causing eviction and cache misses

2. **libwebp's clever "extra rows" solution**:
   - `cache_y - extra_rows` to `cache_y`: holds last N rows from PREVIOUS mb row
   - After each row: `memcpy(cache_y - ysize, bottom_of_current_row, ysize)`
   - Filter reads from extra_rows area (still hot) + current row (hot)
   - Only keeps what's needed: 2 rows for simple filter, 8 for normal filter

3. ~~Byte-by-byte copy~~ Fixed with `copy_from_slice()` (~1-2% gain)

#### Potential Solutions

1. ~~**Quick fix**: `copy_from_slice()`~~ Done, ~1-2% gain

2. **Row cache with extra rows** (HIGH IMPACT):
   - Allocate: `cache_y` = `mb_w * 16` bytes (one row of macroblocks)
   - Plus `extra_rows * mb_w` bytes before cache_y for filter context
   - After each MB row: copy bottom N rows to extra area
   - Filter reads from extra area (previous row) + current cache (current row)
   - Both are small, hot in L1 cache
   - `extra_rows` = 2 for simple filter, 8 for normal filter (from kFilterExtraRows)

## User Feedback Log

(none currently)
