# zenwebp

[![crates.io](https://img.shields.io/crates/v/zenwebp.svg)](https://crates.io/crates/zenwebp)
[![Documentation](https://docs.rs/zenwebp/badge.svg)](https://docs.rs/zenwebp)
[![Build Status](https://github.com/imazen/zenwebp/workflows/Rust%20CI/badge.svg)](https://github.com/imazen/zenwebp/actions)
[![License: AGPL/Commercial](https://img.shields.io/badge/License-AGPL%2FCommercial-blue.svg)](https://github.com/imazen/zenwebp#license)

Pure Rust WebP encoding and decoding. No C dependencies, no unsafe code.

## Getting Started

Add to your `Cargo.toml`:

```toml
[dependencies]
zenwebp = "0.3"
```

**📚 New to zenwebp?** Check out the comprehensive [API guide](examples/api_guide.rs) that demonstrates 100% of the public API with runnable examples.

### Decode a WebP image

```rust
use zenwebp::WebPDecoder;

let webp_bytes: &[u8] = /* your WebP data */;

// Two-phase decoding: parse headers first
let mut decoder = WebPDecoder::build(webp_bytes)?;
let info = decoder.info();
println!("{}x{}, alpha={}", info.width, info.height, info.has_alpha);

// Then decode into a pre-allocated buffer
let mut output = vec![0u8; decoder.output_buffer_size().unwrap()];
decoder.read_image(&mut output)?;
```

### Encode to WebP (Lossy)

```rust
use zenwebp::{LossyConfig, EncodeRequest, PixelLayout};

let rgb_pixels: &[u8] = /* your RGB data */;
let (width, height) = (800, 600);

// Create reusable config
let config = LossyConfig::new()
    .with_quality(85.0)
    .with_method(4);  // 0=fast, 6=best

// Encode
let webp = EncodeRequest::lossy(&config, rgb_pixels, PixelLayout::Rgb8, width, height)
    .encode()?;
```

### Encode to WebP (Lossless)

```rust
use zenwebp::{LosslessConfig, EncodeRequest, PixelLayout};

let rgba_pixels: &[u8] = /* your RGBA data */;
let (width, height) = (800, 600);

let config = LosslessConfig::new()
    .with_quality(90.0)
    .with_method(6);

let webp = EncodeRequest::lossless(&config, rgba_pixels, PixelLayout::Rgba8, width, height)
    .encode()?;
```

## Features

- **Pure Rust** - no C dependencies, builds anywhere Rust does
- **`#![forbid(unsafe_code)]`** - memory safety guaranteed
- **no_std compatible** - works with just `alloc`, no standard library needed
- **SIMD accelerated** - SSE2/SSE4.1/AVX2 on x86, SIMD128 on WASM
- **Full format support** - lossy, lossless, alpha, animation (encode + decode), ICC/EXIF/XMP metadata, mux/demux
- **Metadata module** - `zenwebp::metadata` for extracting/embedding ICC, EXIF, and XMP in encoded WebP bytes without decoding pixels

### Safe SIMD

We achieve both safety and performance through safe abstractions over CPU intrinsics:
- [`archmage`](https://crates.io/crates/archmage) and [`magetypes`](https://crates.io/crates/magetypes) - token-gated safe intrinsics with runtime CPU detection
- [`safe_unaligned_simd`](https://crates.io/crates/safe_unaligned_simd) - safe unaligned load/store

These abstractions may not be perfect, but we trust them over hand-rolled unsafe code.

### Decoder

Supports all WebP features: lossy and lossless compression, alpha channel, animation, and extended format with ICC/EXIF/XMP chunks.

### Encoder

Supports lossy and lossless encoding with configurable quality (0-100) and speed/quality tradeoff (method 0-6).

```rust
use zenwebp::{LossyConfig, LosslessConfig, EncodeRequest, PixelLayout};

// Lossless encoding
let config = LosslessConfig::new().with_quality(100.0);
let webp = EncodeRequest::lossless(&config, pixels, PixelLayout::Rgba8, w, h).encode()?;

// Fast lossy encoding (larger files)
let config = LossyConfig::new().with_quality(75.0).with_method(0);
let webp = EncodeRequest::lossy(&config, pixels, PixelLayout::Rgb8, w, h).encode()?;

// High quality lossy (slower, smaller files)
let config = LossyConfig::new().with_quality(75.0).with_method(6);
let webp = EncodeRequest::lossy(&config, pixels, PixelLayout::Rgb8, w, h).encode()?;
```

## Feature Comparison with libwebp

### Decoder

| Feature | zenwebp | libwebp |
|---------|:-------:|:-------:|
| Lossy (VP8) | :white_check_mark: | :white_check_mark: |
| Lossless (VP8L) | :white_check_mark: | :white_check_mark: |
| Alpha channel | :white_check_mark: | :white_check_mark: |
| Animation decode + encode (ANIM/ANMF) | :white_check_mark: | :white_check_mark: |
| Extended format (VP8X) | :white_check_mark: | :white_check_mark: |
| ICC/EXIF/XMP metadata | :white_check_mark: | :white_check_mark: |
| Metadata without pixel decode | :white_check_mark: | :white_check_mark: |
| Output: RGB, RGBA | :white_check_mark: | :white_check_mark: |
| Output: BGR, BGRA | :white_check_mark: | :white_check_mark: |
| Output: ARGB | :x: | :white_check_mark: |
| Output: YUV 4:2:0 | :white_check_mark: | :white_check_mark: |
| Output: RGB565, RGBA4444 | :x: | :white_check_mark: |
| Premultiplied alpha output | :x: | :white_check_mark: |
| Fancy chroma upsampling | :white_check_mark: | :white_check_mark: |
| Bilinear chroma upsampling | :white_check_mark: | :white_check_mark: |
| Nearest-neighbor upsampling | :white_check_mark: | :white_check_mark: |
| Incremental decode (partial bytes in, rows out) | :x: | :white_check_mark: |
| Crop during decode | :x: | :white_check_mark: |
| Scale during decode | :x: | :white_check_mark: |
| Pipelined MB row decode + filter | :x: | :white_check_mark: width >= 512 |
| Chroma dithering (hides banding at low Q) | :x: | :white_check_mark: |
| Memory limits | :white_check_mark: | :x: |

### Encoder (Lossy VP8)

| Feature | zenwebp | libwebp |
|---------|:-------:|:-------:|
| Quality (0-100) | :white_check_mark: | :white_check_mark: |
| Method (0-6) speed/quality | :white_check_mark: | :white_check_mark: |
| Presets (Photo, Drawing, etc.) | :white_check_mark: 6 | :white_check_mark: 6 |
| Auto preset (content-aware selection) | :white_check_mark: | :x: |
| Target file size (secant method) | :white_check_mark: | :x: |
| Target file size (multi-pass) | :x: | :white_check_mark: |
| Target PSNR | :white_check_mark: | :white_check_mark: |
| SNS (spatial noise shaping) | :white_check_mark: | :white_check_mark: |
| Filter strength/sharpness | :white_check_mark: | :white_check_mark: |
| Autofilter | :white_check_mark: | :white_check_mark: |
| Segments (1-4) | :white_check_mark: | :white_check_mark: |
| Token partitions | 1 | 1-8 |
| Intra16 modes (DC/V/H/TM) | :white_check_mark: | :white_check_mark: |
| Intra4 modes (10 modes) | :white_check_mark: | :white_check_mark: |
| Trellis quantization (m5-6) | :white_check_mark: | :white_check_mark: |
| Alpha channel (lossless + lossy quant) | :white_check_mark: | :white_check_mark: |
| Sharp YUV conversion | :white_check_mark: | :white_check_mark: |
| Multi-pass encoding | :white_check_mark: | :white_check_mark: |
| Near-lossless | :white_check_mark: | :white_check_mark: |
| Encoding statistics | :white_check_mark: | :white_check_mark: |
| Progress callback | :white_check_mark: | :white_check_mark: |
| Cancellation without thread killing (key for untrusted input) | :white_check_mark: | :x: |
| Alpha encoded on 2nd thread | :x: | :white_check_mark: |

### Encoder (Lossless VP8L)

| Feature | zenwebp | libwebp |
|---------|:-------:|:-------:|
| Predictor transform (14 modes) | :white_check_mark: | :white_check_mark: |
| Cross-color transform | :white_check_mark: | :white_check_mark: |
| Subtract green transform | :white_check_mark: | :white_check_mark: |
| Color indexing (palette) | :white_check_mark: | :white_check_mark: |
| Palette sorting strategies | :white_check_mark: 2 | :white_check_mark: |
| Pixel bundling (2/4/8 per pixel) | :white_check_mark: | :white_check_mark: |
| Color cache (auto-sized) | :white_check_mark: | :white_check_mark: |
| LZ77 (standard + RLE + box) | :white_check_mark: | :white_check_mark: |
| TraceBackwards DP (Zopfli-style) | :white_check_mark: | :white_check_mark: |
| Meta-Huffman (spatial codes) | :white_check_mark: | :white_check_mark: |
| Multi-config testing (m5-6) | :white_check_mark: | :white_check_mark: |
| Near-lossless (pixel + residual) | :white_check_mark: | :white_check_mark: |
| AnalyzeEntropy (5 modes) | :white_check_mark: | :white_check_mark: |

### Encoder Input Formats

| Format | zenwebp | libwebp |
|--------|:-------:|:-------:|
| RGB, RGBA | :white_check_mark: | :white_check_mark: |
| BGR, BGRA | :white_check_mark: | :white_check_mark: |
| ARGB | :x: | :white_check_mark: |
| YUV 4:2:0 | :white_check_mark: | :white_check_mark: |
| L8 (grayscale) | :white_check_mark: | :x: requires conversion |
| LA8 (grayscale + alpha) | :white_check_mark: | :x: requires conversion |
| Streaming input (push_rows) | :white_check_mark: | :x: |

### Container / Metadata

| Feature | zenwebp | libwebp |
|---------|:-------:|:-------:|
| RIFF container read/write | :white_check_mark: | :white_check_mark: |
| VP8X extended format | :white_check_mark: | :white_check_mark: |
| ICC/EXIF/XMP read | :white_check_mark: | :white_check_mark: |
| ICC/EXIF/XMP write | :white_check_mark: | :white_check_mark: via libwebpmux |
| Animation encode | :white_check_mark: | :white_check_mark: |
| Mux API (assemble chunks) | :white_check_mark: | :white_check_mark: via libwebpmux |
| Demux API (frame iteration) | :white_check_mark: | :white_check_mark: via libwebpdemux |

### Platform / Build

| Feature | zenwebp | libwebp |
|---------|:-------:|:-------:|
| Language | Pure Rust | C |
| Memory safety | :white_check_mark: `#![forbid(unsafe_code)]` | :x: manual C memory management |
| no_std + alloc | :white_check_mark: | :x: |
| WASM | :white_check_mark: | :white_check_mark: via Emscripten |
| WASM SIMD128 acceleration | :white_check_mark: | :white_check_mark: via SIMDe |
| SSE2 / SSE4.1 | :white_check_mark: | :white_check_mark: |
| AVX2 | :white_check_mark: | :x: |
| NEON (ARM64) | :white_check_mark: | :white_check_mark: |
| MIPS DSP | :x: | :white_check_mark: |
| Runtime CPU detection | :white_check_mark: | :white_check_mark: |

## Performance

### Decoder benchmarks

Tested across three corpora with varying image sizes and content:

| Corpus | Images | Megapixels | zenwebp | image-webp | zenwebp speedup |
|--------|--------|------------|---------|------------|-----------------|
| CLIC2025 | 15 | 42.6 | 1.28x slower | 2.78x slower | **2.2x faster** |
| Screenshots | 9 | 64.4 | 1.22x slower | 2.46x slower | **2.0x faster** |
| CID22 | 15 | 3.9 | 1.67x slower | 2.98x slower | **1.8x faster** |

*All ratios vs libwebp (C). Lower is better. zenwebp consistently ~2x faster than image-webp.*

### Encoder benchmarks

| Corpus | zenwebp m5 | libwebp m5 | Size ratio |
|--------|------------|------------|------------|
| CID22 (248 images) | - | - | 1.0002x |
| Screenshots (13 images) | - | - | 1.0022x |

Encoding speed is ~1.5-1.7x slower than libwebp. File sizes within 0.2% of libwebp at method 5.

### Quality

At the same quality setting, zenwebp produces files within 1-5% of libwebp's size with comparable visual quality. Quality is slightly better than libwebp below Q75 and slightly worse above Q75.

## no_std Support

```toml
[dependencies]
zenwebp = { version = "0.3", default-features = false }
```

Both encoder and decoder work without std. The decoder takes `&[u8]` slices and the encoder writes to `Vec<u8>`. Only `encode_to_writer()` requires the `std` feature.

## Comparison with image-webp

Forked from [`image-webp`](https://github.com/image-rs/image-webp), the official pure-Rust
WebP crate from the image-rs project.

### Decoder

| Feature | image-webp | zenwebp |
|---------|:----------:|:-------:|
| Lossy (VP8) decode | :white_check_mark: | :white_check_mark: |
| Lossless (VP8L) decode | :white_check_mark: | :white_check_mark: |
| Alpha channel decode | :white_check_mark: | :white_check_mark: |
| Animation decode (ANIM/ANMF) | :white_check_mark: | :white_check_mark: |
| Extended format (VP8X) | :white_check_mark: | :white_check_mark: |
| ICC/EXIF/XMP extraction | :white_check_mark: | :white_check_mark: |
| Output: RGB, RGBA | :white_check_mark: | :white_check_mark: |
| Output: BGR, BGRA | :x: | :white_check_mark: |
| Output: YUV 4:2:0 | :x: | :white_check_mark: |
| Fancy chroma upsampling | :x: | :white_check_mark: |
| Bilinear chroma upsampling | :white_check_mark: | :white_check_mark: |
| Nearest-neighbor upsampling | :white_check_mark: | :white_check_mark: |
| SIMD loop filter | :x: | :white_check_mark: |
| SIMD YUV-to-RGB | :x: | :white_check_mark: |
| Memory limits | :white_check_mark: | :white_check_mark: |
| Decode speed vs libwebp | ~2.5-3x slower | ~1.3-1.7x slower |

### Encoder (Lossy)

| Feature | image-webp | zenwebp |
|---------|:----------:|:-------:|
| Lossy VP8 encoding | :white_check_mark: basic | :white_check_mark: full |
| Quality (0-100) | :white_check_mark: | :white_check_mark: |
| Method (0-6) speed/quality tradeoff | :x: | :white_check_mark: |
| I16 prediction (DC/V/H/TM) | DC only | all 4 modes |
| I4 prediction (10 modes) | :x: | :white_check_mark: |
| RD optimization | :x: | :white_check_mark: |
| Trellis quantization | :x: | :white_check_mark: |
| Segments (1-4) | :x: | :white_check_mark: |
| SNS (spatial noise shaping) | :x: | :white_check_mark: |
| Filter strength/sharpness | :x: | :white_check_mark: |
| Autofilter | :x: | :white_check_mark: |
| Presets (Photo, Drawing, etc.) | :x: | :white_check_mark: 6 |
| Auto preset (content-aware selection) | :x: | :white_check_mark: |
| Target file size (secant method) | :x: | :white_check_mark: |
| Target PSNR | :x: | :white_check_mark: |
| Multi-pass encoding | :x: | :white_check_mark: |
| Alpha channel (lossless+lossy quant) | :white_check_mark: lossless | :white_check_mark: lossless+lossy |
| Sharp YUV conversion | :x: | :white_check_mark: |
| Progress callback | :x: | :white_check_mark: |
| Cancellation without thread killing (key for untrusted input) | :x: | :white_check_mark: |
| Encoding statistics | :x: | :white_check_mark: |
| Compression vs libwebp | not measured | within 0.2% at m5 |

### Encoder (Lossless)

| Feature | image-webp | zenwebp |
|---------|:----------:|:-------:|
| Predictor transform | :white_check_mark: | :white_check_mark: |
| Cross-color transform | :white_check_mark: | :white_check_mark: |
| Subtract green transform | :white_check_mark: | :white_check_mark: |
| Color indexing (palette) | :white_check_mark: | :white_check_mark: |
| Pixel bundling | :white_check_mark: | :white_check_mark: |
| Color cache | :white_check_mark: | :white_check_mark: |
| LZ77 (standard + RLE) | :white_check_mark: | :white_check_mark: |
| TraceBackwards DP (Zopfli-style) | :x: | :white_check_mark: |
| Meta-Huffman (spatial codes) | :x: | :white_check_mark: |
| Multi-config testing (m5-6) | :x: | :white_check_mark: |
| Near-lossless (pixel + residual) | :x: | :white_check_mark: |
| AnalyzeEntropy (5 modes) | :x: | :white_check_mark: |
| Compression vs libwebp | larger than libwebp | 0.4% smaller |

### Container / Animation

| Feature | image-webp | zenwebp |
|---------|:----------:|:-------:|
| RIFF read/write | :white_check_mark: | :white_check_mark: |
| ICC/EXIF/XMP write | :white_check_mark: | :white_check_mark: |
| Animation decode | :white_check_mark: | :white_check_mark: |
| Animation encode | :x: | :white_check_mark: |
| Mux API (assemble chunks) | :x: | :white_check_mark: |
| Demux API (frame iteration) | :x: | :white_check_mark: |
| Metadata module (no pixel decode) | :x: | :white_check_mark: |

### Encoder Input Formats

| Format | image-webp | zenwebp |
|--------|:----------:|:-------:|
| RGB8 | :white_check_mark: | :white_check_mark: |
| RGBA8 | :white_check_mark: | :white_check_mark: |
| L8 (grayscale) | :white_check_mark: | :white_check_mark: |
| LA8 (grayscale+alpha) | :white_check_mark: | :white_check_mark: |
| BGR8, BGRA8 | :x: | :white_check_mark: |
| YUV 4:2:0 | :x: | :white_check_mark: |
| Streaming input (push_rows) | :x: | :white_check_mark: |

### Platform / Safety

| Feature | image-webp | zenwebp |
|---------|:----------:|:-------:|
| `#![forbid(unsafe_code)]` | :white_check_mark: | :white_check_mark: |
| Zero unsafe in all deps | :white_check_mark: | :x: archmage generates unsafe for SIMD |
| no_std + alloc | :x: requires std | :white_check_mark: |
| WASM | :x: | :white_check_mark: |
| WASM SIMD128 acceleration | :x: | :white_check_mark: |
| SSE2 / SSE4.1 SIMD | :x: | :white_check_mark: |
| AVX2 SIMD | :x: | :white_check_mark: |
| NEON (ARM64) SIMD | :x: | :white_check_mark: |
| Runtime CPU detection | :x: | :white_check_mark: |
| Dependencies | 2 | 14 (7 optional) |
| Codebase size | ~10,600 lines | ~41,000 lines |
| License | MIT OR Apache-2.0 | AGPL-3.0 / Commercial |

### Tradeoffs

**Choose image-webp if** you need the smallest possible dependency tree, zero unsafe
anywhere in the build, or only need lossless encoding. Its ~10k line codebase is easy to audit.

**Choose zenwebp if** you need lossy encoding competitive with libwebp, faster decoding,
animation encoding, near-lossless, target file size, no_std, or SIMD acceleration.

**Safety model:** Both crates use `#![forbid(unsafe_code)]`. image-webp extends this to all
dependencies — truly zero unsafe anywhere. zenwebp's SIMD acceleration depends on `archmage`,
whose `#[arcane]` proc macro generates unsafe blocks internally to call CPU intrinsics. We
consider archmage's token-based safety model sound, but it is not the same as zero-unsafe.
Without the `simd` feature, zenwebp also contains no unsafe at all.

## License

Sustainable, large-scale open source work requires a funding model, and I have been
doing this full-time for 15 years. If you are using this for closed-source development
AND make over $1 million per year, you'll need to buy a commercial license at
https://www.imazen.io/pricing

Commercial licenses are similar to the Apache 2 license but company-specific, and on
a sliding scale. You can also use this under the AGPL v3.

### Previous Versions

Versions 0.1.x - 0.2.x were dual-licensed under MIT OR Apache-2.0. See the git history for those license files.


## Contributing

Contributions welcome! Please feel free to open issues or pull requests.

## Credits

This project builds on excellent work by others:

- **[image-rs/image-webp](https://github.com/image-rs/image-webp)** - The foundation of this crate.
  The image-rs team built a complete, correct, truly-safe WebP decoder and lossless encoder.
  We forked their work and added SIMD acceleration, lossy encoding with full RD optimization,
  animation encoding, and more. If you don't need those features, consider using their crate
  directly for a smaller, simpler dependency.

- **[libwebp](https://chromium.googlesource.com/webm/libwebp)** (Google) - Reference implementation.
  Our lossy encoder closely follows libwebp's algorithms for RD optimization, trellis quantization,
  and mode selection. The WebP format itself is Google's creation.

- **[archmage](https://crates.io/crates/archmage)** & **[magetypes](https://crates.io/crates/magetypes)** - Safe SIMD abstractions
- **[safe_unaligned_simd](https://crates.io/crates/safe_unaligned_simd)** - Safe unaligned SIMD operations
- **Claude** (Anthropic) - AI-assisted development

Code review recommended for production use.
