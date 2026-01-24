# zenwebp CLAUDE.md

See global ~/.claude/CLAUDE.md for general instructions.

## Current Optimization Status (2026-01-23)

### Encoder Performance vs libwebp

**Method Parameter Added** - Speed/quality tradeoff (0-6):
| Method | Time | Throughput | File Size | Notes |
|--------|------|------------|-----------|-------|
| 0 | 55ms | 7.15 MPix/s | 101KB | I16-only, no trellis |
| 2 | 79ms | 4.97 MPix/s | 87KB | Limited I4 (3 modes), no trellis |
| 4 | 65ms | 6.05 MPix/s | 78KB | Balanced (4 modes), trellis |
| 6 | 122ms | 3.21 MPix/s | 76KB | Full search (10 modes), trellis |

*Benchmark: 768x512 Kodak image at Q75, 10 iterations, release mode*

### Recent SIMD Optimizations
- **GetResidualCost SIMD** - Precompute abs/ctx/levels with SSE2 (30% speedup, 2026-01-23)
- **FTransform2** - Fused residual+DCT for 2 blocks at once (2026-01-23)
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

### Detailed Encoder Callgrind Analysis (2026-01-23)

**Instruction counts (768x512 Kodak, Q75, method 4):**
| Encoder | Instructions | Ratio |
|---------|--------------|-------|
| Ours | 2,350M | 2.88x more |
| libwebp | 817M | baseline |

**Cache behavior (D1 miss rate):**
- Ours: 0.1% (better than libwebp)
- libwebp: 0.3%

The 2.8x slowdown is pure instruction count, not memory access patterns.

**Hotspot comparison (millions of instructions):**
| Our Function | Ours | libwebp Equivalent | libwebp | Ratio |
|--------------|------|-------------------|---------|-------|
| choose_macroblock_info | 1,468M (62%) | VP8Decimate + PickBest* | 580M | 2.5x |
| get_residual_cost | 349M (15%) | GetResidualCost_SSE2 | 135M | 2.6x |
| encode_coefficients | 304M (13%) | VP8EmitTokens | 57M | 5.3x |
| trellis_quantize_block | 304M (13%) | QuantizeBlock_SSE2 | 39M | **7.8x** |
| get_cost_luma16 | 182M (8%) | VP8GetCostLuma16 | 47M | 3.9x |
| get_cost_luma4 | 181M (8%) | VP8GetCostLuma4 | 115M | 1.6x |
| tdisto_16x16 | 154M (7%) | Disto4x4_SSE2 | 52M | 3.0x |
| idct4x4 | 138M (6%) | ITransform_SSE2 | 64M | 2.2x |
| dct4x4 | 119M (5%) | FTransform_SSE2 | 41M | 2.9x |
| write_with_tree | 121M (5%) | VP8PutBit | 44M | 2.8x |

**Biggest optimization opportunities (potential instruction savings):**
1. **trellis_quantize_block** - 7.8x slower, ~265M potential savings
2. **encode_coefficients** - 5.3x slower, ~247M potential savings
3. **get_residual_cost** - 2.6x slower, ~214M potential savings (libwebp uses SIMD)
4. **get_cost_luma16** - 3.9x slower, ~135M potential savings

**libwebp SIMD functions we lack:**
- `GetResidualCost_SSE2` - residual cost with SIMD
- `QuantizeBlock_SSE2` - quantization with SIMD
- `Disto4x4_SSE2` - distortion with SIMD
- `FTransform_SSE2` / `ITransform_SSE2` - faster than our SIMD

### Key Files
- `src/vp8_encoder.rs` - Main encoder, mode selection
- `src/vp8_cost.rs` - Cost estimation, trellis quantization
- `src/simd_sse.rs` - SIMD implementations
- `src/encoder.rs` - Public API, EncoderParams

### no_std Support (2026-01-23)

The crate supports `no_std` environments with the `alloc` crate:
- `cargo build --no-default-features` for no_std
- Both decoder and encoder work with no_std+alloc
- Decoder uses `&[u8]` slices instead of `Read` trait
- Encoder uses `Vec<u8>` instead of `Write` trait
- `Encoder.encode_to_writer()` requires `std` feature
- Dependencies: `thiserror` (no_std), `whereat` (no_std), `hashbrown` (no_std), `libm` (floating-point math)

### Decoder Performance vs libwebp (2026-01-23)

| Test | Our Decoder | libwebp | Speed Ratio |
|------|-------------|---------|-------------|
| libwebp-encoded | 4.24ms (93 MPix/s) | 3.05ms (129 MPix/s) | 1.39x slower |
| our-encoded | 4.14ms (95 MPix/s) | 2.91ms (135 MPix/s) | 1.42x slower |

*Benchmark: 768x512 Kodak image, 100 iterations, release mode*

Our decoder is ~1.4x slower than libwebp (improved from 2.5x baseline). Recent optimizations:
- **SIMD chroma vertical loop filter** - U+V processed together as 16 pixels (10% speedup, 2026-01-23)
- **Position-indexed probability table** - eliminates COEFF_BANDS lookup (10% instruction reduction, commit 15b3771)
- **VP8HeaderBitReader for mode parsing** - replaces ArithmeticDecoder (6-8% faster, commit 9b6f963)
- **SIMD chroma horizontal loop filter** - U+V processed together as 16 rows (7.8% instruction reduction, commit c3b9051)
- **DC-only IDCT fast path** and reusable coefficient buffer (commit 9d08433)
- **SIMD horizontal loop filter** (DoFilter4/DoFilter6 for vertical edges, commit fc4c33f)
- **Inline coefficient tree** like libwebp's GetCoeffsFast (12% instruction reduction, commit ac47ba5)
- **16-bit SIMD YUV conversion** ported from libwebp (18% instruction reduction, commit b7ac4b9)
- **Row cache with extra rows** for loop filter cache locality (commit c16995f)
- **SIMD normal loop filter for vertical edges** (~10% speedup, commit 3bf30f1)
- **libwebp-rs style bit reader for coefficients** (16% speedup, commit 5588e44)
- AVX2 loop filter (16 pixels at once) - simple filter only

### Decoder Profiler Hot Spots (after VP8HeaderBitReader)
| Function | % Time | Notes |
|----------|--------|-------|
| read_coefficients | ~22% | Coefficient decoding |
| decode_frame_ | ~12% | Frame processing + inlined mode parsing |
| fancy_upsample_8_pairs | ~5% | YUV SIMD |
| should_filter_vertical | ~4% | Loop filter threshold check |
| idct4x4_avx2 | ~2% | Already SIMD |
| normal_h_filter_uv_* | ~2% | Chroma SIMD |

### Detailed Callgrind/Cachegrind Analysis (2026-01-23)

**Per-decode instruction count (after chroma vertical SIMD):**
| Metric | Ours | libwebp | Ratio |
|--------|------|---------|-------|
| Coeff reading | 49.1M | 36.4M | 1.35x |
| Total program | 20.88B/200 | - | -10% vs previous |

*Total instructions reduced from 23.2B → 20.88B (-10%)*
| Memory reads | ~12M | 6.7M | 1.8x |
| Memory writes | ~9M | 4.9M | 1.8x |
| D1 read miss % | 0.33% | 0.70% | Better! |

*Note: 14443M / 200 decodes = 72.2M per decode. Improved from 78.7M (8% reduction since chroma SIMD).*

**Instruction breakdown comparison:**
| Category | Ours | libwebp | Extra per decode |
|----------|------|---------|------------------|
| Coeff reading | ~24M (33%) | 20.5M (44%) | +3.5M |
| Loop filter total | ~14M (19%) | ~4M (8%) | **+10M** |
| YUV conversion | ~6M (8%) | ~4M (9%) | +2M |
| Mode parsing (inlined) | ~0.2M | 0 | +0.2M |

**Root causes of the remaining 1.55x instruction gap:**
1. **Loop filter overhead** - still 2.5x more instructions than libwebp despite SIMD
2. **Coefficient reading** - ~17% more instructions than libwebp
3. **Function call overhead** - Rust abstractions vs libwebp's inline macros

Loop filter now uses SIMD for:
- Simple filter: both V and H edges (luma + chroma)
- Normal filter: both V and H edges (luma + chroma with SIMD)

### SIMD Decoder Optimizations
- `src/yuv_simd.rs` - SSE2 YUV→RGB (ported from libwebp, commit b7ac4b9)
  - **16-bit arithmetic** using `_mm_mulhi_epu16` (8 values at once vs 4)
  - **SIMD RGB interleaving** via `planar_to_24b` (6 stores vs 48)
  - Processes 32 pixels at a time (simple) or 16 (fancy upsampling)
  - `simd` feature now enabled by default
- `src/loop_filter_avx2.rs` - SSE4.1 loop filter (16 pixels at once)
  - **Normal filter SIMD for both V and H edges** (luma, commit fc4c33f)
  - **Chroma horizontal SIMD** - U+V processed together as 16 rows (commit c3b9051)
  - **Chroma vertical SIMD** - U+V packed as 16 pixels per row (2026-01-23)
  - Uses transpose technique for horizontal filtering (16 rows × 8 cols)
  - Simple filter: both V and H edges have SIMD (luma + chroma)
  - Normal filter: both V and H edges have SIMD (luma + chroma)
- `src/vp8_bit_reader.rs` - libwebp-rs style bit readers
  - **VP8HeaderBitReader** - header/mode parsing (commit 9b6f963)
  - **VP8Partitions/PartitionReader** - coefficient reading (commit 5588e44)
  - Uses VP8GetBitAlt algorithm with 56-bit buffer
  - `leading_zeros()` for normalization (single LZCNT instruction)
  - **16% speedup** in overall decode from coefficient reader

### TODO - Remaining Optimization Opportunities
Priority ordered by instruction savings potential:

1. **Loop filter overhead** (~10M savings)
   - Still 2.5x more instructions than libwebp despite SIMD
   - Per-pixel threshold checks (`should_filter_*`) are expensive
   - Consider batch threshold computation

Completed:
- [x] ~~Chroma vertical SIMD~~ (2026-01-23) - U+V packed as 16 pixels, 10% speedup
- [x] ~~Position-indexed probability table~~ (commit 15b3771) - 10% instruction reduction
- [x] ~~Replace ArithmeticDecoder for mode parsing~~ (commit 9b6f963) - 6-8% faster
- [x] ~~Loop filter SIMD for chroma horizontal~~ (commit c3b9051) - U+V together as 16 rows, 7.8% reduction
- [x] ~~DC-only IDCT and reusable coefficient buffer~~ (commit 9d08433)
- [x] ~~Row cache with extra rows for loop filter cache locality~~ (commit c16995f)
- [x] ~~Add SIMD horizontal normal filter~~ (commit fc4c33f)
- [x] ~~Inline coefficient tree like GetCoeffsFast~~ (commit ac47ba5)
**Encoder optimization opportunities (see Callgrind Analysis above):**
- [ ] trellis_quantize_block - 7.8x slower, SIMD quantization like QuantizeBlock_SSE2
- [ ] encode_coefficients - 5.3x slower, token emission overhead
- [ ] get_residual_cost - 2.6x slower, SIMD residual cost like GetResidualCost_SSE2
- [ ] get_cost_luma16 - 3.9x slower
- [ ] tdisto_16x16 - 3.0x slower (despite having some SIMD)

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

### Loop Filter Optimization (Resolved 2026-01-23)

Normal filter now uses SIMD for horizontal edges (both luma and chroma).
Chroma is handled by processing U and V planes together as 16 rows, reusing
the existing 16-row infrastructure. See commit c3b9051.

Remaining loop filter overhead (~6M extra instructions vs libwebp) appears
to come from per-pixel threshold checks rather than the filter math itself.

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
