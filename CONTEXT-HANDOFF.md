# Context Handoff - Decoder Optimization

## Summary
Optimizing the image-webp decoder to close the gap with libwebp C. Currently 2.15x slower (improved from 2.5x).

## Recent Work Completed

### 1. libwebp-rs Style Bit Reader (16% speedup)
- Created `src/vp8_bit_reader.rs` with VP8GetBitAlt algorithm from libwebp
- Replaced ArithmeticDecoder for coefficient reading with VP8Partitions/PartitionReader
- Throughput: 55 → 62 MPix/s
- Gap: 2.5x → 2.15x slower

Key commits:
- `5588e44` - perf: use libwebp-rs style bit reader for coefficient decoding
- `ebb0fbc` - perf: add branch/cache analysis, benchmark env vars

### 2. Branch/Cache Analysis
Compared our decoder vs libwebp using `perf stat`:

| Metric | Our Decoder | libwebp | Ratio |
|--------|-------------|---------|-------|
| Instructions | 449M | 99M | 4.5x more |
| Branch miss rate | 3.81% | 8.15% | **Better!** |
| Cache misses | 1.63M | 23.5K | **69x more** |
| L1 misses | 4.65M | 256K | **18x more** |
| Cache miss rate | 14.57% | 5.01% | 3x worse |

**Key Finding: Cache efficiency is the main bottleneck, NOT branch prediction.**

## Current Architecture

### Decoder Files
- `src/vp8.rs` - Main decoder, uses VP8Partitions for coefficients
- `src/vp8_bit_reader.rs` - New libwebp-style bit reader (VP8Partitions, PartitionReader)
- `src/vp8_arithmetic_decoder.rs` - Still used for header/mode parsing (self.b field)
- `src/loop_filter.rs` - Scalar loop filter
- `src/loop_filter_avx2.rs` - SIMD simple filter only

### Current Profile
| Function | % Time |
|----------|--------|
| read_coefficients | 18.85% |
| idct4x4_avx512 | 6.05% |
| should_filter_vertical | 5.56% |
| decode_frame_ | 5.20% |
| Loop filter total | ~12% |

## Next Optimization Opportunities

1. **Cache efficiency** (biggest opportunity)
   - Profile which data structures cause cache misses
   - Consider cache-line aligned allocations
   - Improve memory access patterns
   - Token probability tables may be cache-unfriendly

2. **SIMD normal loop filter** (~12% opportunity)
   - `src/loop_filter_avx2.rs` has simple filter only
   - Need DoFilter4/DoFilter6 for normal filter path

3. **Mode parsing bit reader**
   - `self.b: ArithmeticDecoder` still uses old code
   - Could use libwebp-rs style reader here too

## Test Commands
```bash
# Run tests
cargo test --release --features "unsafe-simd"

# Benchmark both decoders
cargo run --release --features "unsafe-simd,_profiling" --bin profile_decode

# Benchmark only our decoder (for perf stat)
BENCH_OURS=1 cargo run --release --features "unsafe-simd,_profiling" --bin profile_decode

# Benchmark only libwebp (for perf stat)
BENCH_LIBWEBP=1 cargo run --release --features "unsafe-simd,_profiling" --bin profile_decode

# Cache/branch analysis
BENCH_OURS=1 perf stat -e cycles,instructions,branches,branch-misses,cache-references,cache-misses,L1-dcache-load-misses cargo run --release --features "unsafe-simd,_profiling" --bin profile_decode
```

## Key Files to Read
- `CLAUDE.md` - Project documentation with performance tables
- `src/vp8_bit_reader.rs` - New bit reader implementation
- `src/vp8.rs:read_coefficients` - Coefficient decoding using new reader
- `benches/profile_decode.rs` - Benchmark with BENCH_OURS/BENCH_LIBWEBP env vars
