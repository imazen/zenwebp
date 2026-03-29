# zenwebp Session Context — 2026-03-27/28

## What Was Accomplished

### Lossless Encoder: 28x slower → parity with libwebp C
- **Root cause**: public API bypassed full VP8L pipeline, used literal-only encoder
- **Key fixes**: LUT tables (eliminated libm::log2), cache_bits N→2, entropy bin threshold bug, priority-queue histogram clustering
- **Result**: 1.03x at m2-m4, 2.6x FASTER at m6. 24/24 pixel-exact roundtrips.

### Lossy Decoder v2: 1.3-2.2x → 1.09-1.15x of libwebp C
- **Architecture**: ground-up redesign as `src/decoder/vp8v2/` (6169 lines, 9 files)
- **Streaming**: cache→RGB directly, no full-frame Y/U/V buffers (~100KB peak vs ~4.5MB)
- **Buffer reuse**: `DecoderContext` persists across decodes, eliminates allocation
- **Bit-exact**: 0 pixel diffs vs libwebp on 218+ conformance files
- **Fused YUV→RGB**: exact 9:3:3:1 fancy upsample with archmage V3/NEON/WASM SIMD
- **Animation**: mixed lossy+lossless frames, callback API, frame buffer reuse
- **Chroma dithering**: ported from v1
- **Security**: header hardening (dimension limits, overflow checks, partition validation), fuzz target
- **Merged as PR #2** to main

### Architecture Cleanup
- 10 per-platform SIMD files → 3 consolidated per-algorithm files
- vp8.rs (2087 lines) → 5 focused modules (vp8/mod.rs, coefficients.rs, predict.rs, cache.rs, header.rs)
- 21 decode entry points → thin wrappers via decode_to_rgba + convert
- 4 bit readers → 2 (removed VP8BitReader, PartitionReader)
- `simd` feature removed — archmage/magetypes mandatory deps
- ~580 bounds checks eliminated via fixed-size array pattern

## Key Techniques Discovered

1. **Fixed-size array pattern**: `&[u8; N]` at boundary, zero interior bounds checks. Applied to loop filter (-33%), prediction (-46-64%), transform, quantize, SIMD helpers, YUV. The single most impactful safe-Rust optimization.

2. **BTB aliasing fix**: `#[inline(never)]` on `read_coefficients` — 25x inlining caused 2.6x branch mispredictions. Out-of-line matched C's mispredict rate exactly.

3. **archmage patterns**: `#[rite]` default, `#[arcane]` only at entry. Tokens cheap (1ns), target_feature boundary expensive. `incant!` for dispatch.

4. **Entropy bin threshold bug**: accumulator cost vs incoming cost — 82x overcall → parity.

5. **`yuv` crate integration**: 2x faster RGB→YUV for encode. NOT used for decode (bilinear differs from WebP's fancy upsample). Our own exact kernel is bit-identical to libwebp.

6. **Buffer reuse**: `DecoderContext` with `ensure_capacity` using `resize()` not `vec![]`. `fill()` only on regions that need specific values.

## Current Performance (no -C target-cpu=native)

### Lossy Decode (14 images, v2)
- Photos: **1.06-1.15x** of libwebp C
- Screenshots: **1.06-1.14x** (was 2.2x with v1)
- 4K wiki outlier: **1.12x** (was 2.2x)

### Lossless Encode
- m2-m4: **1.03x**, m6: **2.6x faster**
- Compression: 1.00-1.01x of libwebp

### Lossless Decode
- Photos: **at parity or faster**
- Screenshots: **1.23-1.31x** (plan filed to reach <1.10x)

### Lossy Encode
- **1.43-1.47x** (plan filed to reach <1.10x, gap is mode selection 3.4x)

## Outstanding Work

### Filed plans (docs/)
- `docs/LOSSLESS-DECODER-PLAN.md` — streaming arch, pixel decode loop, transforms NEON/WASM
- `docs/LOSSY-ENCODER-PLAN.md` — mode selection 3.4x, residual cost 2.4x, memory patterns

### Not done
- `incant!` dispatch in pipeline.rs (manual if-let-Some currently)
- Inner loop panic path audit (899 potential panics from slice indexing)
- Full fuzzing campaign on v2
- zencodec streaming integration (v2's row-by-row maps to `decode_next_into_buf`)
- Lossless decoder streaming architecture
- Lossy encoder optimization

## Branch State
- `main`: decoder-v2 merged, all optimizations landed
- `decoder-v2`: merged into main via PR #2
- All worktrees cleaned

## Key Files
- `src/decoder/vp8v2/` — v2 lossy decoder (9 files, 6169 lines)
- `src/decoder/vp8/` — v1 lossy decoder (5 modules, kept for reference)
- `src/decoder/lossless.rs` — lossless decoder (unchanged this session)
- `src/encoder/vp8l/` — lossless encoder (heavily optimized this session)
- `docs/PERFORMANCE.md` — comprehensive benchmark data
- `docs/CALL-TREE.md` — SIMD tier map per function
- `benches/decode_compare.rs` — 14-image 4-decoder benchmark
- `tests/v2_pixel_perfect.rs` — 218+ file pixel-exact verification
- `fuzz/fuzz_targets/decode_v2.rs` — v2 fuzz target

## Test Commands
```bash
cargo test --release --lib                          # 227 tests
cargo test --release --test v2_pixel_perfect        # 0 diffs vs libwebp
cargo test --release --test v2_animation            # 10 animation tests
cargo bench --bench decode_compare                  # 14 images, NO -C target-cpu=native
cargo bench --bench decode_lossless_compare         # lossless decode
examples/lossless_rt_check                          # 24/24 pixel-exact lossless
```
