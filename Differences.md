# zenwebp vs libwebp and webpx

## vs libwebp (C reference implementation)

### Features

| Feature | zenwebp | libwebp |
|---------|---------|---------|
| Lossy encoding | Yes | Yes |
| Lossless encoding | Yes | Yes |
| Lossy decoding | Yes | Yes |
| Lossless decoding | Yes | Yes |
| Animation encode | Yes | Yes |
| Animation decode | Yes | Yes |
| Alpha encoding | Yes | Yes |
| ICC/EXIF/XMP metadata | Yes | Yes |
| Incremental decoding | No | Yes |
| Multi-threaded encoding | No | Yes |
| `#![forbid(unsafe_code)]` | Yes (except `unchecked` feature) | N/A (C) |
| no_std support | Yes | No |
| WASM SIMD128 | Yes | Via Emscripten |

### Compression (CID22 corpus, 248 images, Q75, SNS=0, filter=0, segments=1)

| Method | zenwebp / libwebp file size ratio |
|--------|-----------------------------------|
| 4 | 1.0099x (0.99% larger) |
| 5 | 1.0002x (essentially identical) |
| 6 | 1.0022x (0.22% larger) |

Production settings (SNS=50, filter=60): Q75 = 1.0149x, Q90 = 1.0060x.

### Lossless Compression

CID22 50-image subset: **0.996x** (0.4% smaller than libwebp).
Screenshots: **0.995x** (0.5% smaller).

### Speed (criterion, 512x512 Q75, 2026-02-05)

| Method | zenwebp | libwebp | Ratio |
|--------|---------|---------|-------|
| 0 | 7.0ms | 2.7ms | 2.6x slower |
| 4 | 15.0ms | 10.2ms | 1.47x slower |
| 6 | 22.2ms | 15.5ms | 1.43x slower |

Instruction ratio is 1.12x — remaining gap is memory access patterns.

### Decoding Speed

| Corpus | Ratio |
|--------|-------|
| CLIC2025 (10 images) | 0.93-1.25x (avg ~1.15x slower) |
| 1024x1024 lossy | 1.32x slower (267 vs 351 MPix/s) |

## vs webpx (Rust crate)

### API Differences

| Capability | zenwebp | webpx |
|------------|---------|-------|
| Animation encoder | `AnimationEncoder` (frame-by-frame, delta compression) | `AnimEncoder` |
| Animation decoder | `AnimationDecoder` (high-level iterator API) | `AnimDecoder` |
| Typed pixel API | `pixel::decode::<Rgba<u8>>()` via `rgb` crate (optional) | No |
| `imgref` integration | `pixel::decode_to_img()`, `EncoderConfig::encode_img()` | No |
| Encoder builder | `Encoder::new_rgba().quality(85.0).encode()` | Similar |
| Reusable config | `EncoderConfig` encodes multiple images | No |
| Stride-aware encode | `Encoder::new_rgba_stride()`, `new_rgb_stride()` | No |
| Demuxer | `WebPDemuxer` (zero-copy chunk parser) | No |
| Muxer | `WebPMux` (container assembler) | No |
| Standalone metadata | `get_icc_profile()`, `embed_exif()`, `remove_xmp()`, etc. | No |
| Presets | `Preset::{Photo, Drawing, Icon, Text, Auto}` | No |
| Target size search | Secant method via `EncoderConfig::target_size()` | No |
| Resource estimation | `heuristics::estimate_encode()`, `estimate_decode()` | No |
| Perceptual model | CSF tables, SATD masking, JND thresholds (method 3+) | No (wraps libwebp) |
| Bitstream format info | `ImageInfo::format` (`Lossy`/`Lossless`) | No |
| no_std | Yes | No |
| Pure Rust | Yes | No (FFI to libwebp C) |
| SIMD | archmage token-based (SSE2/SSE4.1/AVX2/NEON/WASM) | libwebp's C SIMD |
| License | MIT/Apache-2.0 | MIT (but links libwebp BSD) |

### Features webpx Has That zenwebp Doesn't

- Multi-threaded encoding (via libwebp's thread pool)
- Incremental/streaming decode
- Mature C codebase battle-tested across billions of images

### Features zenwebp Has That webpx Doesn't

- Pure Rust, no C toolchain needed
- no_std + alloc support
- WASM-native SIMD (not Emscripten)
- Perceptual quality optimization (CSF, masking, JND)
- Zero-copy demuxer
- Typed pixel API via `rgb` crate
- `imgref` integration for 2D image buffers
- Reusable encoder configuration
- Stride-aware encoder constructors
- Standalone metadata convenience functions (get/embed/remove ICC/EXIF/XMP)
- Resource estimation heuristics (memory, time, output size)
- Bitstream format detection (`ImageInfo::format`)
- Content-aware auto-detection (photo/drawing/icon)
- Sub-frame delta compression in animation encoder
