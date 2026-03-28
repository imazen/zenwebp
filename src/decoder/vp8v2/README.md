# VP8v2 Decoder

Redesigned VP8 lossy decoder targeting libwebp C parity or faster.

Key differences from v1:
- DecoderContext with buffer reuse (no memset per decode)
- Streaming row pipeline (no full-frame Y/U/V buffers)
- Single #[arcane] per MB row (all #[rite] inlines)
- Direct-to-cache prediction (no workspace copy)
- yuv crate for color conversion
- Precomputed filter/dequant tables
- Fixed-size arrays everywhere
