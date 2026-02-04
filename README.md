# zenwebp

[![crates.io](https://img.shields.io/crates/v/zenwebp.svg)](https://crates.io/crates/zenwebp)
[![Documentation](https://docs.rs/zenwebp/badge.svg)](https://docs.rs/zenwebp)
[![Build Status](https://github.com/imazen/zenwebp/workflows/Rust%20CI/badge.svg)](https://github.com/imazen/zenwebp/actions)
[![License](https://img.shields.io/crates/l/zenwebp.svg)](https://github.com/imazen/zenwebp#license)

Pure Rust WebP encoding and decoding. No C dependencies, no unsafe code.

## Getting Started

Add to your `Cargo.toml`:

```toml
[dependencies]
zenwebp = "0.2"
```

### Decode a WebP image

```rust
use zenwebp::WebPDecoder;

let webp_bytes: &[u8] = /* your WebP data */;
let mut decoder = WebPDecoder::new(webp_bytes)?;
let (width, height) = decoder.dimensions();

// Decode to RGBA
let mut rgba = vec![0u8; decoder.output_buffer_size().unwrap()];
decoder.read_image(&mut rgba)?;
```

### Encode to WebP

```rust
use zenwebp::{WebPEncoder, EncoderParams, ColorType};

let rgb_pixels: &[u8] = /* your RGB data */;
let (width, height) = (800, 600);

// Lossy encoding at quality 75
let mut webp_output = Vec::new();
let mut encoder = WebPEncoder::new(&mut webp_output);
encoder.set_params(EncoderParams::lossy(75));
encoder.encode(rgb_pixels, width, height, ColorType::Rgb8)?;
```

## Features

- **Pure Rust** - no C dependencies, builds anywhere Rust does
- **`#![forbid(unsafe_code)]`** - memory safety guaranteed
- **no_std compatible** - works with just `alloc`, no standard library needed
- **SIMD accelerated** - SSE2/SSE4.1/AVX2 on x86, SIMD128 on WASM
- **Full format support** - lossy, lossless, alpha, animation (decode), ICC/EXIF/XMP metadata

### Safe SIMD

We achieve both safety and performance through safe abstractions over CPU intrinsics:
- [`wide`](https://crates.io/crates/wide) - portable SIMD types that autovectorize well
- [`archmage`](https://crates.io/crates/archmage) and [`magetypes`](https://crates.io/crates/magetypes) - token-gated safe intrinsics
- [`safe_unaligned_simd`](https://crates.io/crates/safe_unaligned_simd) - safe unaligned load/store
- `core::arch` - newly stabilized as safe in Rust

These abstractions may not be perfect, but we trust them over hand-rolled unsafe code.

### Decoder

Supports all WebP features: lossy and lossless compression, alpha channel, animation, and extended format with ICC/EXIF/XMP chunks.

### Encoder

Supports lossy and lossless encoding with configurable quality (0-100) and speed/quality tradeoff (method 0-6).

```rust
// Lossless encoding
encoder.set_params(EncoderParams::lossless());

// Fast lossy encoding (larger files)
encoder.set_params(EncoderParams::lossy(75).method(0));

// High quality lossy (slower, smaller files)
encoder.set_params(EncoderParams::lossy(75).method(6));
```

## Feature Comparison with libwebp

zenwebp aims to be a drop-in replacement for libwebp in most use cases. Here's what's implemented and what's not.

### Decoder

| Feature | zenwebp | libwebp |
|---------|---------|---------|
| Lossy (VP8) | Yes | Yes |
| Lossless (VP8L) | Yes | Yes |
| Alpha channel | Yes | Yes |
| Animation (ANIM/ANMF) | Yes (read) | Yes (read + write) |
| Extended format (VP8X) | Yes | Yes |
| ICC/EXIF/XMP metadata | Yes (raw bytes) | Yes (raw bytes) |
| Output: RGB, RGBA | Yes | Yes |
| Output: BGR, BGRA, ARGB | No | Yes |
| Output: YUV 4:2:0 | No | Yes |
| Output: RGB565, RGBA4444 | No | Yes |
| Premultiplied alpha output | No | Yes |
| Fancy chroma upsampling | Yes | Yes |
| Simple (nearest) upsampling | Yes | Yes |
| Incremental/streaming decode | No | Yes |
| Crop during decode | No | Yes |
| Scale during decode | No | Yes |
| Threaded decoding | No | Yes |
| Dithering | No | Yes |

### Encoder (Lossy VP8)

| Feature | zenwebp | libwebp |
|---------|---------|---------|
| Quality (0-100) | Yes | Yes |
| Method (0-6) speed/quality | Yes | Yes |
| Presets (Photo, Drawing, etc.) | Yes (7 presets, including Auto) | Yes (6 presets) |
| Target file size | Yes (secant method) | Yes (multi-pass) |
| Target PSNR | Yes (secant method) | Yes |
| SNS (spatial noise shaping) | Yes | Yes |
| Filter strength/sharpness | Yes | Yes |
| Autofilter | Yes | Yes |
| Segments (1-4) | Yes | Yes |
| Token partitions | 1 partition | 1-8 partitions |
| Intra16 modes (DC/V/H/TM) | Yes | Yes |
| Intra4 modes (10 modes) | Yes | Yes |
| Trellis quantization | Yes (m5-6) | Yes (m5-6) |
| Alpha channel encoding | Yes (lossless) | Yes (lossless or lossy) |
| Sharp YUV conversion | Yes (via `yuv` crate, `fast-yuv` feature) | Yes (libsharpyuv) |
| Multi-pass encoding | Yes | Yes |
| Near-lossless | Yes | Yes |
| Input: RGB, RGBA | Yes | Yes |
| Input: L8 (grayscale) | Yes | No (requires conversion) |
| Input: BGR, BGRA | No | Yes |
| Input: YUV 4:2:0 | No | Yes |
| Encoding statistics | Yes (EncodingStats) | Yes (WebPAuxStats) |
| Progress callback | Yes (EncodeProgress + Stop) | Yes |
| Threaded encoding | No | Yes (alpha parallel) |

### Encoder (Lossless VP8L)

| Feature | zenwebp | libwebp |
|---------|---------|---------|
| Predictor transform (14 modes) | Yes | Yes |
| Cross-color transform | Yes | Yes |
| Subtract green transform | Yes | Yes |
| Color indexing (palette) | Yes | Yes |
| Palette sorting strategies | Yes (2) | Yes |
| Pixel bundling (2/4/8 per pixel) | Yes | Yes |
| Color cache (auto-sized) | Yes | Yes |
| LZ77 Standard | Yes | Yes |
| LZ77 RLE | Yes | Yes |
| LZ77 Box | Yes (palette images) | Yes (palette images) |
| TraceBackwards DP | Yes (Zopfli-style) | Yes (Zopfli-style) |
| Meta-Huffman (spatial codes) | Yes | Yes |
| Multi-config testing | Yes (m5-6) | Yes (m5-6) |
| Near-lossless | Yes (pixel + residual) | Yes (pixel + residual) |
| AnalyzeEntropy selection | Yes (5 modes) | Yes (5 modes) |

### Container / Metadata

| Feature | zenwebp | libwebp |
|---------|---------|---------|
| RIFF container read | Yes | Yes |
| RIFF container write | Yes | Yes |
| VP8X extended format | Yes | Yes |
| ICC profile read | Yes | Yes |
| EXIF metadata read | Yes | Yes |
| XMP metadata read | Yes | Yes |
| Metadata write (ICC/EXIF/XMP) | Yes | Yes (via libwebpmux) |
| Animation write | No | Yes (WebPAnimEncoder) |
| Mux API (add/remove chunks) | No | Yes (libwebpmux) |
| Demux API (frame iteration) | No | Yes (libwebpdemux) |

### Platform / Build

| Feature | zenwebp | libwebp |
|---------|---------|---------|
| Language | Pure Rust | C |
| Unsafe code | `#![forbid(unsafe_code)]` | N/A (C) |
| no_std + alloc | Yes | No |
| WASM | Yes (SIMD128) | Yes (via Emscripten/SIMDe) |
| SSE2 | Yes | Yes |
| SSE4.1 | Yes | Yes |
| AVX2 | Yes | No |
| NEON (ARM64) | Yes | Yes |
| MIPS DSP | No | Yes |
| Runtime CPU detection | Yes | Yes |
| Custom allocator | No (uses alloc) | No (compile-time only) |

### What zenwebp has that libwebp doesn't

- **no_std support** - use in embedded, WASM, or kernel contexts with just `alloc`
- **Memory safety** - `#![forbid(unsafe_code)]`, no buffer overflows by construction
- **AVX2 SIMD** - wider SIMD for loop filter and YUV conversion
- **Auto preset** - content-aware preset selection based on image analysis
- **Grayscale input** - direct L8/LA8 encoding without manual conversion
- **Cooperative cancellation** - separate `Stop` token for external cancellation (via `enough` crate)

## Performance

Benchmarks on a 768x512 image (Kodak test suite):

| Operation | zenwebp | libwebp | Ratio |
|-----------|---------|---------|-------|
| Decode | 4.2ms | 3.0ms | 1.4x slower |
| Encode (method 4) | 65ms | 75ms* | 1.15x faster |

*libwebp method 6 with trellis, comparable quality settings

### Quality

At the same quality setting, zenwebp produces files within 1-5% of libwebp's size with comparable visual quality. Quality is slightly better than libwebp below Q75 and slightly worse above Q75.

## no_std Support

```toml
[dependencies]
zenwebp = { version = "0.2", default-features = false }
```

Both encoder and decoder work without std. The decoder takes `&[u8]` slices and the encoder writes to `Vec<u8>`. Only `encode_to_writer()` requires the `std` feature.

## Origin

Forked from [`image-webp`](https://github.com/image-rs/image-webp) with significant enhancements:
- Lossy encoder (original only supported lossless)
- ~2x faster decoding through SIMD and algorithmic improvements
- Full no_std support for both encoder and decoder
- WASM SIMD128 support

## License

Licensed under either [Apache License, Version 2.0](LICENSE-APACHE) or [MIT license](LICENSE-MIT) at your option.

## Contributing

Contributions welcome! Please feel free to open issues or pull requests.

## Credits

- **[image-rs/image-webp](https://github.com/image-rs/image-webp)** - Original crate this was forked from
- **[libwebp](https://chromium.googlesource.com/webm/libwebp)** (Google) - Reference implementation and algorithm source
- **[wide](https://crates.io/crates/wide)** (Lokathor) - Portable SIMD types
- **[archmage](https://crates.io/crates/archmage)** & **[magetypes](https://crates.io/crates/magetypes)** (Lilith) - Safe intrinsics
- **[safe_unaligned_simd](https://crates.io/crates/safe_unaligned_simd)** (Lilith) - Safe unaligned SIMD operations
- **Claude** (Anthropic) - AI development assistance

Code review recommended for production use.
