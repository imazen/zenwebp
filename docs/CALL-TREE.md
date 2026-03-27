# zenwebp Call Tree — SIMD Tiers Per Function

Generated 2026-03-27. See `docs/PERFORMANCE.md` for benchmark data.

## Tier Legend

- **scalar**: No SIMD, always available
- **v1**: SSE2 (x86-64 baseline, always on)
- **v3**: AVX2+FMA (Haswell 2013+, Zen 1+)
- **neon**: AArch64 NEON (always on ARM64)
- **wasm128**: WebAssembly SIMD128

## Tier Availability Matrix

| Function | scalar | v1 | v3 | neon | wasm128 |
|----------|--------|----|----|------|---------|
| **Decoder IDCT** | ✓ | ✓ #[rite] | (via v1) | ✓ #[rite] | ✓ |
| **Decoder prediction** | ✓ | — | ✓ autovec | ✓ | — |
| **Loop filter** | ✓ | ✓ #[rite] | ✓ #[rite] | ✓ #[rite] | ✓ |
| **YUV→RGB** | ✓ | — | ✓ #[arcane] | ✓ #[rite] | ✓ |
| **Lossless inverse transforms** | ✓ | ✓ #[rite] | — | — | — |
| **Encoder forward DCT** | ✓ | ✓ #[rite] | — | ✓ #[rite] | ✓ |
| **Encoder quantize** | ✓ | ✓ #[rite] | — | — | — |
| **Encoder residual cost** | ✓ | ✓ #[rite] | — | — | — |
| **Encoder I4 mode eval** | ✓ | — | ✓ #[arcane] | — | — |
| **Lossless entropy** | ✓ | — | — | — | — |
| **Lossless hash chain** | ✓ | — | — | — | — |
| **Lossless histogram clustering** | ✓ | — | — | — | — |

## Gaps (no SIMD where libwebp has it)

- **Decoder prediction at v3**: Currently autovectorized under `#[arcane]` X64V3Token. 4-pixel-wide inner loops are too short for AVX2 to help. libwebp uses scalar here too.
- **Lossless inverse transforms**: Only SSE2. Missing NEON, WASM128. Missing predictors 5-7, 10-13 SIMD (serial dependencies).
- **Encoder quantize**: Only SSE2. Missing NEON, WASM128, V3.
- **Encoder residual cost**: Only SSE2. Missing NEON, WASM128.
- **YUV→RGB**: Missing V1 (SSE2) — jumps from scalar to V3. Could add SSE2 tier.

## Inline Strategy

| Attribute | Count | Purpose |
|-----------|-------|---------|
| `#[inline(always)]` | ~200 | Hot predicates, SIMD wrappers, per-pixel loops |
| `#[inline]` | ~150 | Dispatch, helpers |
| `#[inline(never)]` | ~15 | BTB aliasing fix (read_coefficients), cold paths |
| `#[cold]` | ~5 | Scalar prediction fallbacks (reduce code size) |
| `#[arcane]` | ~30 | SIMD entry points (target_feature boundary) |
| `#[rite]` | ~80 | SIMD inner functions (inline into arcane caller) |
