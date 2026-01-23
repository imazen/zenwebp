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
| libwebp-encoded | 5.32ms (74 MPix/s) | 3.28ms (120 MPix/s) | 1.62x slower |
| our-encoded | 4.91ms (80 MPix/s) | 2.86ms (137 MPix/s) | 1.71x slower |

*Benchmark: 768x512 Kodak image, 100 iterations, release mode*

Our decoder is ~1.6-1.7x slower than libwebp (improved from 2.5x baseline). Recent optimizations:
- **SIMD horizontal loop filter** (DoFilter4/DoFilter6 for vertical edges, commit fc4c33f)
- **Inline coefficient tree** like libwebp's GetCoeffsFast (12% instruction reduction, commit ac47ba5)
- **16-bit SIMD YUV conversion** ported from libwebp (18% instruction reduction, commit b7ac4b9)
- **Row cache with extra rows** for loop filter cache locality (commit c16995f)
- **SIMD normal loop filter for vertical edges** (~10% speedup, commit 3bf30f1)
- **libwebp-rs style bit reader for coefficients** (16% speedup, commit 5588e44)
- AVX2 loop filter (16 pixels at once) - simple filter only

### Decoder Profiler Hot Spots (after SIMD horizontal filter)
| Function | % Time | Notes |
|----------|--------|-------|
| read_coefficients | 20.31% | Coefficient decoding |
| fancy_upsample_8_pairs | 4.62% | YUV SIMD |
| decode_frame_ | 4.49% | Frame processing overhead |
| should_filter_vertical | 3.99% | Loop filter threshold check |
| macroblock_filter_horizontal | 2.83% | Chroma only (scalar) |
| ArithmeticDecoder | 2.63% | Mode parsing |
| idct4x4_avx2 | 1.69% | Already SIMD |

### Detailed Callgrind/Cachegrind Analysis (2026-01-23)

**Per-decode instruction count (isolated runs):**
| Metric | Ours | libwebp | Ratio |
|--------|------|---------|-------|
| Instructions | 85.4M | 46.6M | **1.83x** |
| Memory reads | 12.9M | 6.7M | 1.92x |
| Memory writes | 10.2M | 4.9M | 2.09x |
| D1 read miss % | 0.33% | 0.70% | Better! |

**Instruction breakdown comparison:**
| Category | Ours | libwebp | Extra per decode |
|----------|------|---------|------------------|
| Coeff reading | 25.5M (30%) | 20.5M (44%) | +5M |
| Loop filter total | ~20M (23%) | ~4M (8%) | **+16M** |
| YUV conversion | ~6M (8%) | ~4M (9%) | +2M |
| memset zeroing | 2.5M (3%) | ~0 | **+2.5M** |
| ArithmeticDecoder (modes) | 3.3M (4%) | 0 | +3.3M |

**Root causes of the 1.83x instruction gap:**
1. **Loop filter scalar overhead** - `should_filter_vertical` called per-pixel (6%), chroma uses full scalar path
2. **Buffer zeroing** - memset for coefficient blocks costs 2.5M/decode
3. **Mode parsing** - ArithmeticDecoder still used (3.3M/decode) vs libwebp's inline bit reader
4. **Coefficient reading** - 24% more instructions than libwebp despite similar algorithm

Loop filter now uses SIMD for:
- Simple filter: both V and H edges (luma)
- Normal filter: V edges (luma), H edges (luma with SIMD, chroma scalar)

### SIMD Decoder Optimizations
- `src/yuv_simd.rs` - SSE2 YUV→RGB (ported from libwebp, commit b7ac4b9)
  - **16-bit arithmetic** using `_mm_mulhi_epu16` (8 values at once vs 4)
  - **SIMD RGB interleaving** via `planar_to_24b` (6 stores vs 48)
  - Processes 32 pixels at a time (simple) or 16 (fancy upsampling)
  - `unsafe-simd` feature now enabled by default
- `src/loop_filter_avx2.rs` - SSE4.1 loop filter (16 pixels at once)
  - **Normal filter SIMD for both V and H edges** (luma only, commit fc4c33f)
  - Uses transpose technique for horizontal filtering (16 rows × 8 cols)
  - Simple filter: both V and H edges have SIMD
  - Chroma filtering still scalar (8 rows, not 16)
- `src/vp8_bit_reader.rs` - libwebp-rs style bit reader for coefficients
  - Uses VP8GetBitAlt algorithm with 56-bit buffer
  - `leading_zeros()` for normalization (single LZCNT instruction)
  - **16% speedup** in overall decode (commit 5588e44)

### TODO - Remaining Optimization Opportunities
Priority ordered by instruction savings potential:

1. **Loop filter SIMD for chroma** (~8M savings) - 8 rows needs different approach than 16-row luma
   - `should_filter_vertical` called per-pixel is 6% of time
   - Chroma uses full scalar path (macroblock_filter_horizontal at 4.1%)

2. **Eliminate buffer zeroing** (~2.5M savings) - memset for coefficient blocks
   - Consider using uninitialized memory or lazy zeroing
   - `MaybeUninit` for coefficient blocks?

3. **Replace ArithmeticDecoder for mode parsing** (~3.3M savings)
   - Use libwebp-style bit reader for `self.b` field
   - Only coefficient reading uses fast path currently

4. **Reduce coefficient reading overhead** (~5M savings)
   - Still 24% more instructions than libwebp despite similar algorithm
   - May be Rust abstraction overhead or bounds checking

Completed:
- [x] ~~Row cache with extra rows for loop filter cache locality~~ (commit c16995f)
- [x] ~~Add SIMD horizontal normal filter~~ (commit fc4c33f)
- [x] ~~Inline coefficient tree like GetCoeffsFast~~ (commit ac47ba5)
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

#### Solutions Implemented

1. ~~**Quick fix**: `copy_from_slice()`~~ Done, ~1-2% gain

2. **Row cache with extra rows** - IMPLEMENTED (commit c16995f):
   - Added `cache_y/u/v` buffers: `(extra_rows + 16) * mb_w` bytes
   - `extra_y_rows` = 8 for normal filter, 2 for simple, 0 for none
   - Prediction writes to cache, not final buffer
   - Filter operates on cache (tighter stride, better cache locality)
   - Delayed output: filter modifies pixels above AND below edge
   - `rotate_extra_rows()` copies bottom rows for next iteration

   **Outcome**: Performance essentially unchanged (~1.92x). The extra copy
   overhead may be offsetting cache locality gains. The remaining 2x gap
   vs libwebp likely comes from instruction count (4.5x more) rather than
   cache efficiency alone.

## User Feedback Log

(none currently)
