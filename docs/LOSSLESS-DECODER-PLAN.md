# Lossless Decoder — Close 1.23-1.50x Gap to <1.10x

## Current State

codec_wiki 2560x1664: **1.23-1.31x** wall-clock, **1.50x** instructions vs libwebp C.
Photos 512x512: at parity or faster (no BGRA→RGBA conversion overhead).

## Profile (per 5 decodes, codec_wiki)

| Component | zenwebp (M) | libwebp (M) | Ratio |
|-----------|-------------|-------------|-------|
| Pixel decode loop | 551 | 241 | 2.29x |
| Inverse transforms | 430 | 254 | 1.69x |
| memset | 69 | 0 | - |
| Huffman build | 7 | 7 | 1.0x |

## Plan

### Phase 1: Streaming architecture
- Eliminate full-frame ARGB buffer (~16MB for 4K)
- `LosslessDecoderContext` with buffer reuse across frames
- Target: eliminate 69M memset, ~200KB peak memory

### Phase 2: Pixel decode loop (2.29x — biggest target)
- Fixed-size Huffman tables (known max size, eliminates bounds checks)
- Color cache as `[u32; 2048]` with masked index (no bounds check)
- Profile `read_symbol_fast` 6-bit hit rate — should be >90%
- Apply fixed-size array pattern throughout

### Phase 3: Inverse transforms (1.69x)
- NEON/WASM128 for all transforms (currently SSE2 only)
- Predictors 5-13 SSE2 (select/clamp via `_mm_min/max_epu8`)
- Fixed-size array pattern on transform data buffers

### Phase 4: Huffman + color cache
- Compare table layout vs libwebp's `HuffmanCode`
- Flat array for short codes instead of tree-walk

### Target: <1.10x screenshots, <1.05x photos
