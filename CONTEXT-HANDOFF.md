# zenwebp Decoder Optimization Handoff

## Current Status (Updated 2026-02-04)

**Decoder performance:** 1.31x slower than libwebp (140 vs 184 MPix/s)

This is reasonable for safe Rust. The gap is smaller than it appears in profiles due to
valgrind amplification.

## Critical Constraint: `#![forbid(unsafe_code)]`

The crate explicitly forbids unsafe code (see `src/lib.rs:83`). This is a deliberate
design decision for safety. Any optimization must work within this constraint.

**Implication:** Raw pointer arithmetic is not allowed. All SIMD loads/stores must go
through `safe_unaligned_simd` which requires `&[u8; 16]` references, which in turn
requires bounds checks when created from slices.

## Key Discovery: Bounds Check Overhead (Constrained)

Used `cargo-asm` to inspect generated assembly. Found that **bounds checks add overhead**:

```
simple_v_filter16 breakdown:
- 10 conditional jumps: bounds checks (4 loads × 2 checks + 2 stores × 2 checks)
- ~60 instructions: actual SIMD work
```

**Root cause:** Each `<&[u8; 16]>::try_from(&pixels[offset..][..16]).unwrap()` generates 2 checks:
1. Slice offset bounds check
2. Slice length check

**Tested approaches that DON'T help:**
- `first_chunk()` - Still generates 2 checks per access
- `split_at_mut()` chain - Generates even MORE checks (one per split)
- Upfront asserts - Compiler can't propagate proof to subsequent accesses

**What WOULD help but is forbidden:**
- Raw pointer arithmetic after upfront assert (reduces to 2 total checks)
- This pattern is used by libwebp and is correct, but violates our safety policy

## Actual Profiler Hot Spots (callgrind 2026-02-04)

| Function | % Time | Notes |
|----------|--------|-------|
| read_coefficients_to_block | 13.71% | Arithmetic decoding |
| fill_row_fancy_with_2_uv_rows | 8.37% | YUV conversion |
| loop filter functions (total) | ~7% | Bounds checks are part of this |
| memset | 3.58% | Zero-init |
| update_token_probabilities | 2.76% | Probability updates |

**Key insight:** Loop filter bounds checks are only part of the 7% loop filter time.
Even complete elimination would save at most 2-3% of total decode time.

## Realistic Optimization Opportunities

### 1. Coefficient Reading (13.71%)

`read_coefficients_to_block` uses ~50% more instructions than libwebp's GetCoeffsFast.
This is from:
- More complex control flow (nested ifs vs. computed goto)
- Diagnostic code overhead (lines 1027-1036 in vp8.rs)
- Less optimal bit reader interface

### 2. YUV Conversion (8.37%)

`fill_row_fancy_with_2_uv_rows` does fancy upsampling. Potential improvements:
- Fuse more operations
- Better memory access patterns

### 3. Accept the Safety Tradeoff

The ~6% bounds check overhead is the cost of safe Rust. Options:
- Accept it as reasonable for a safe implementation
- Create a separate `unsafe_simd` feature that opts into raw pointers
- Fork `safe_unaligned_simd` to add pointer-based API

## `chunks_exact` Analysis

`chunks_exact` only helps when input sizes are **compile-time known**:

```rust
// This works (no bounds checks):
fn process(data: &[u8; 256]) {
    for chunk in data.chunks_exact(16) { ... }
}

// This doesn't help (runtime offset):
fn filter(pixels: &[u8], point: usize) {
    let p1 = &pixels[point..].first_chunk().unwrap();  // Still has checks
}
```

The loop filters have **runtime offsets** (point, stride are function parameters),
so `chunks_exact` cannot eliminate their bounds checks.

## Reference Materials in `~/work/helpful-info/`

### Key Articles

1. **state-of-simd-rust-2025.md** - Overview of SIMD approaches:
   - `pulp` for type-based polymorphism with built-in multiversioning
   - `multiversion` crate for function-level dispatch
   - `wide` for portable SIMD without multiversioning
   - Key insight: "pulp requires using its SIMD types"

2. **towards-fearless-simd-2025.md** - Linebender's approach:
   - `fearless_simd` prototype with fixed-size chunks
   - Token-based safety (like archmage)
   - AVX-512 predication eliminates tail loops
   - Runtime feature detection should happen once at startup

3. **safe_unaligned_simd.md** - Our current approach:
   - Safe wrappers for load/store
   - Requires `&[u8; 16]` references (hence the bounds checks)

### Crate Documentation

`~/work/helpful-info/crate-docs/`:
- `pulp/` - Safe SIMD with `WithSimd` trait
- `multiversion/` - Function multiversioning macros
- `pic-scale/` - Image scaler using 4-row batching
- `fast_image_resize/` - SIMD resize patterns

## Specific Files to Optimize

1. `src/decoder/vp8.rs` - `read_coefficients_to_block` (13.71% of time)
2. `src/decoder/yuv_simd.rs` - `fill_row_fancy_with_2_uv_rows` (8.37%)
3. `src/decoder/loop_filter_avx2.rs` - 10 filter functions (~7%)

## Questions Resolved

1. **Can `chunks_exact` help with non-contiguous access patterns?**
   - **NO.** Loop filters access rows at `stride` intervals with runtime offsets.
   - The compiler cannot prove bounds even with upfront asserts.
   - `first_chunk()` has the same limitation.

2. **Can we use raw pointers to eliminate checks?**
   - **NO.** The crate has `#![forbid(unsafe_code)]`.
   - This is a deliberate safety design decision.
   - Changing it would require project-wide discussion.

3. **How significant is the bounds check overhead?**
   - ~2-3% of total decode time (loop filters are 7%, checks are ~40% of that)
   - Not the biggest bottleneck. Coefficient reading (13.71%) is larger.

## Build & Test Commands

```bash
# Build with SIMD
cargo build --release --features simd

# Inspect assembly
cargo asm --lib --features simd "simple_v_filter16::__simd_inner"

# Run decode benchmark (needs a PNG file)
cargo run --release --features simd --example decode_benchmark /path/to/image.png

# Profile with callgrind
valgrind --tool=callgrind --callgrind-out-file=/tmp/decode.callgrind \
  ./target/release/examples/decode_benchmark /path/to/image.png
callgrind_annotate --auto=yes /tmp/decode.callgrind | head -60
```

## Next Steps

Given the `#![forbid(unsafe_code)]` constraint, focus on:

1. **Optimize coefficient reading** (13.71% of time)
   - Profile `read_coefficients_to_block` for micro-optimizations
   - Consider removing diagnostic code in hot path
   - Look at bit reader efficiency

2. **Optimize YUV conversion** (8.37% of time)
   - Profile `fill_row_fancy_with_2_uv_rows`
   - Look for redundant work or cache-unfriendly patterns

3. **Accept bounds check overhead** as cost of safe Rust
   - The 1.31x slowdown vs libwebp is reasonable for a safe implementation
   - Most of the gap is from higher-level algorithmic differences, not bounds checks

4. **Optional: Create `unsafe_simd` feature**
   - Allow opting into raw pointers for performance-critical uses
   - Would require careful safety documentation
   - Would need project owner approval
