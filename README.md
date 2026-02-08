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

zenwebp aims to be a drop-in replacement for libwebp in most use cases. Here's what's implemented and what's not.

### Decoder

| Feature | zenwebp | libwebp |
|---------|---------|---------|
| Lossy (VP8) | Yes | Yes |
| Lossless (VP8L) | Yes | Yes |
| Alpha channel | Yes | Yes |
| Animation (ANIM/ANMF) | Yes (read + write) | Yes (read + write) |
| Extended format (VP8X) | Yes | Yes |
| ICC/EXIF/XMP metadata | Yes (raw bytes) | Yes (raw bytes) |
| Output: RGB, RGBA | Yes | Yes |
| Output: BGR, BGRA | Yes | Yes |
| Output: ARGB | No | Yes |
| Output: YUV 4:2:0 | Yes | Yes |
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
| Alpha channel encoding | Yes (lossless + lossy quantization) | Yes (lossless + lossy quantization) |
| Sharp YUV conversion | Yes (via `yuv` crate, `fast-yuv` feature) | Yes (libsharpyuv) |
| Multi-pass encoding | Yes | Yes |
| Near-lossless | Yes | Yes |
| Input: RGB, RGBA | Yes | Yes |
| Input: L8 (grayscale) | Yes | No (requires conversion) |
| Input: BGR, BGRA | Yes | Yes |
| Input: YUV 4:2:0 | Yes | Yes |
| Encoding statistics | Yes (EncodeStats) | Yes (WebPAuxStats) |
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
| Animation write | Yes (AnimationEncoder) | Yes (WebPAnimEncoder) |
| Mux API (assemble chunks) | Yes (WebPMux) | Yes (libwebpmux) |
| Demux API (frame iteration) | Yes (WebPDemuxer) | Yes (libwebpdemux) |

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

This crate is forked from [`image-webp`](https://github.com/image-rs/image-webp), the official pure-Rust
WebP crate from the image-rs project. Both are excellent choices depending on your needs.

### When to use image-webp (upstream)

| Aspect | image-webp |
|--------|------------|
| **Safety** | Zero unsafe code in crate AND all dependencies |
| **Codebase** | ~10,600 lines - small, auditable |
| **Dependencies** | 2 (byteorder-lite, quick-error) |
| **Decoder speed** | ~2.5-3x slower than libwebp |
| **Encoder** | Lossless only, basic but fast |
| **Best for** | Security-critical contexts, minimal attack surface |

**Choose image-webp if:** You need maximum assurance of memory safety, minimal dependencies,
or only need lossless encoding. The smaller codebase is easier to audit.

### When to use zenwebp

| Aspect | zenwebp |
|--------|---------|
| **Safety** | `#![forbid(unsafe_code)]` but relies on archmage for SIMD |
| **Codebase** | ~41,000 lines (+30k for lossy encoder, same decoder base) |
| **Dependencies** | 14 (7 optional) |
| **Decoder speed** | ~1.4-1.7x slower than libwebp (~2x faster than image-webp) |
| **Encoder** | Lossy + lossless, matches libwebp compression |
| **Best for** | Full WebP support, lossy encoding, libwebp replacement |

**Choose zenwebp if:** You need lossy encoding, faster decoding, libwebp-compatible compression,
or features like animation encoding, near-lossless, or target file size.

### Honest tradeoffs

**Code size:** We added **~30,000 lines** to implement lossy encoding matching libwebp. This is
a significant increase in attack surface and audit burden compared to image-webp's compact codebase.

**Safety model:** We use `#![forbid(unsafe_code)]` which prevents any direct unsafe in our source.
However, our SIMD acceleration depends on the `archmage` crate, whose `#[arcane]` proc macro
generates `unsafe` blocks internally (to call CPU intrinsics). The generated unsafe bypasses
Rust's `forbid` lint due to proc-macro span handling. We consider archmage's token-based safety
model reasonable - tokens are only created after runtime CPU feature detection, and we don't
forge tokens - but this is not the same as image-webp's truly zero-unsafe guarantee where both
the crate and all dependencies contain no unsafe whatsoever.

**Without the `simd` feature**, zenwebp contains no unsafe code at all, but decoding will be slower.

### Feature additions over image-webp

- **Lossy VP8 encoder** - full RD optimization, trellis quantization, all I4/I16 modes
- **~2x faster decoder** - SIMD loop filter, YUV conversion, coefficient decoding
- **Animation encoding** - AnimationEncoder with frame timing
- **Near-lossless** - pixel and residual quantization
- **Target file size** - secant method convergence
- **SIMD** - SSE2/SSE4.1/AVX2 via archmage (but at cost of indirect unsafe)
- **no_std** - both encoder and decoder work with just `alloc`

## License

Sustainable, large-scale open source work requires a funding model, and I have been
doing this full-time for 15 years. If you are using this for closed-source development
AND make over $1 million per year, you'll need to buy a commercial license at
https://www.imazen.io/pricing

Commercial licenses are similar to the Apache 2 license but company-specific, and on
a sliding scale. You can also use this under the AGPL v3.

### Previous Versions

Versions 0.1.x - 0.3.x were dual-licensed under MIT OR Apache-2.0. See the git history for those license files.

### Color Quantization Features

The optional color quantization features provide two backend choices:

- **`quantize`** or **`quantize-quantizr`** (MIT-licensed): Uses the `quantizr` crate. Compatible with AGPL.

- **`quantize-imagequant`** (GPL-3.0-or-later): Uses the [`imagequant`](https://github.com/ImageOptim/libimagequant) crate, which **produces dramatically better
  file sizes** via more compressible dithering and quantization patterns. GPL is compatible with AGPL. [Commercial license available from upstream](https://supso.org/projects/pngquant) if you purchase both.

```toml
# Default quantizer (MIT, compatible with AGPL)
zenwebp = { version = "0.3", features = ["quantize"] }

# Better quality GPL quantizer (also compatible with AGPL)
zenwebp = { version = "0.3", features = ["quantize-imagequant"] }
```

## Contributing

Contributions welcome! Please feel free to open issues or pull requests.

## Credits

This project builds on excellent work by others:

- **[image-rs/image-webp](https://github.com/image-rs/image-webp)** - The foundation of this crate.
  The image-rs team built a complete, correct, truly-safe WebP decoder and lossless encoder.
  We forked their work and added lossy encoding on top. If you don't need lossy encoding,
  consider using their crate directly for a smaller, simpler dependency.

- **[libwebp](https://chromium.googlesource.com/webm/libwebp)** (Google) - Reference implementation.
  Our lossy encoder closely follows libwebp's algorithms for RD optimization, trellis quantization,
  and mode selection. The WebP format itself is Google's creation.

- **[archmage](https://crates.io/crates/archmage)** & **[magetypes](https://crates.io/crates/magetypes)** - Safe SIMD abstractions
- **[safe_unaligned_simd](https://crates.io/crates/safe_unaligned_simd)** - Safe unaligned SIMD operations
- **Claude** (Anthropic) - AI-assisted development

Code review recommended for production use.
