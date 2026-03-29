# zenwebp [![CI](https://img.shields.io/github/actions/workflow/status/imazen/zenwebp/ci.yml?style=flat-square&label=CI)](https://github.com/imazen/zenwebp/actions/workflows/ci.yml) [![crates.io](https://img.shields.io/crates/v/zenwebp?style=flat-square)](https://crates.io/crates/zenwebp) [![docs.rs](https://img.shields.io/docsrs/zenwebp?style=flat-square)](https://docs.rs/zenwebp) [![MSRV](https://img.shields.io/badge/MSRV-1.93-blue?style=flat-square)](https://doc.rust-lang.org/cargo/reference/manifest.html#the-rust-version-field) [![license](https://img.shields.io/badge/License-AGPL%2FCommercial-blue?style=flat-square)](https://github.com/imazen/zenwebp#license) [![codecov](https://img.shields.io/codecov/c/github/imazen/zenwebp?style=flat-square)](https://codecov.io/gh/imazen/zenwebp)

Pure Rust WebP encoding and decoding. No C dependencies, no unsafe code.

## Getting Started

Add to your `Cargo.toml`:

```toml
[dependencies]
zenwebp = "0.4"
```

**New to zenwebp?** Check out the [API guide](examples/api_guide.rs) that demonstrates 100% of the public API with runnable examples.

### Decode a WebP image

```rust
// One-shot decode
let (pixels, width, height) = zenwebp::oneshot::decode_rgba(webp_bytes)?;
```

Or use [`WebPDecoder`] for two-phase decoding (inspect headers before allocating):

```rust
use zenwebp::WebPDecoder;

let webp_bytes: &[u8] = /* your WebP data */;
let mut decoder = WebPDecoder::build(webp_bytes)?;
let info = decoder.info();
println!("{}x{}, alpha={}, orientation={:?}",
    info.width, info.height, info.has_alpha, info.orientation);

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
- **SIMD accelerated** - SSE2/SSE4.1/AVX2 on x86, NEON on ARM64, SIMD128 on WASM
- **Full format support** - lossy, lossless, alpha, animation (encode + decode), ICC/EXIF/XMP metadata, EXIF orientation parsing, mux/demux, chroma dithering
- **Metadata module** - `zenwebp::metadata` for extracting/embedding ICC, EXIF, and XMP in encoded WebP bytes without decoding pixels
- **zencodec integration** - optional `zencodec` feature for unified codec trait implementations

### Safe SIMD

We achieve both safety and performance through safe abstractions over CPU intrinsics:
- [`archmage`](https://crates.io/crates/archmage) and [`magetypes`](https://crates.io/crates/magetypes) - token-gated safe intrinsics with runtime CPU detection
- [`safe_unaligned_simd`](https://crates.io/crates/safe_unaligned_simd) - safe unaligned load/store

These abstractions may not be perfect, but we trust them over hand-rolled unsafe code.

### Decoder

Supports all WebP features: lossy and lossless compression, alpha channel, animation, and extended format with ICC/EXIF/XMP chunks. Output formats: RGB, RGBA, BGR, BGRA, ARGB, YUV 4:2:0, RGB565, RGBA4444, premultiplied RGBA/BGRA/ARGB. Chroma dithering matches libwebp pixel-for-pixel.

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
| Output: ARGB | :white_check_mark: | :white_check_mark: |
| Output: YUV 4:2:0 | :white_check_mark: | :white_check_mark: |
| Output: RGB565, RGBA4444 | :white_check_mark: | :white_check_mark: |
| Premultiplied alpha output (RGBA, BGRA, ARGB) | :white_check_mark: | :white_check_mark: |
| Fancy chroma upsampling | :white_check_mark: | :white_check_mark: |
| Bilinear chroma upsampling | :white_check_mark: | :white_check_mark: |
| Nearest-neighbor upsampling | :white_check_mark: | :white_check_mark: |
| Incremental decode (partial bytes in, rows out) | :x: | :white_check_mark: |
| Crop during decode | :x: | :white_check_mark: |
| Scale during decode | :x: | :white_check_mark: |
| 2-thread decode pipeline (reconstruct + filter overlap) | :x: | :white_check_mark: width >= 512 |
| Chroma dithering (hides banding at high Q) | :white_check_mark: default off | :white_check_mark: default off |
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
| ARGB | :white_check_mark: | :white_check_mark: |
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

### Lossy decoder benchmarks

**Bit-exact with libwebp** — 0 pixel diffs on 12,825 scraped WebP files and 218 conformance files.

Tested across 14 images (CLIC2025 photos, screenshots, CID22) without `-C target-cpu=native`:

| Content | vs libwebp (C) |
|---------|---------------|
| Photos (CLIC2025, 2K) | **1.09-1.12x** |
| Screenshots (1K-4K) | **1.06-1.14x** |
| Small photos (512-576px) | **1.10-1.15x** |

Streaming architecture via zencodec's `StreamingDecode` trait (feature `zencodec`). The full decoded image never needs to exist in memory:

| Decode mode | Peak memory (2940×1912) |
|------------|------------------------|
| `StreamingDecode::next_batch()` | **1.5 MB** |
| `decode_rgb()` (full frame) | 35 MB |
| libwebp `WebPDecodeRGB` | 34 MB |

The streaming decoder yields 16-row RGB strips via zencodec's `StreamingDecode` trait, enabling strip-based pipelines (decode → resize → encode) with constant memory regardless of image size.

### Lossless decoder benchmarks

| Content | vs libwebp (C) |
|---------|---------------|
| Photos (512px) | **at parity or faster** |
| Screenshots (2K) | **1.23-1.31x** |

### Lossy encoder benchmarks

| Method | Speed vs libwebp | Compression |
|--------|-----------------|-------------|
| m4 (default) | **1.35x** | 1.01x |
| m5 | **1.34x** | **1.0002x** |
| m6 (best) | **1.32x** | 1.002x |

File sizes within 0.02% of libwebp at method 5.

### Lossless encoder benchmarks

| Method | Speed vs libwebp | Compression |
|--------|-----------------|-------------|
| m2-m4 | **1.03x** (near parity) | 1.00-1.01x |
| m6 | **2.6x faster** | 1.01x |

24/24 pixel-exact lossless roundtrips verified.

### Quality

At the same quality setting, zenwebp produces files within 1-5% of libwebp's size with comparable visual quality.

## no_std Support

```toml
[dependencies]
zenwebp = { version = "0.4", default-features = false }
```

Both encoder and decoder work without std. The decoder takes `&[u8]` slices and the encoder writes to `Vec<u8>`. Only `encode_to_writer()` requires the `std` feature.

## Credits

zenwebp started as a fork of [`image-webp`](https://github.com/image-rs/image-webp), the
pure-Rust WebP crate from the [image-rs](https://github.com/image-rs) project. The original
decoder and lossless encoder formed the foundation on which zenwebp was built. We're grateful
to the image-rs maintainers for their well-structured, battle-tested codebase.

From that foundation, zenwebp was substantially rewritten to achieve libwebp feature and
performance parity: a ground-up lossy encoder, a redesigned streaming decoder, SIMD
acceleration via [archmage](https://crates.io/crates/archmage), and extensive optimization
work across all pipelines. The lossless decoder retains the most shared DNA with image-webp.

## Image tech I maintain

| | |
|:--|:--|
| State of the art codecs* | [zenjpeg] · [zenpng] · **zenwebp** · [zengif] · [zenavif] ([rav1d-safe] · [zenrav1e] · [zenavif-parse] · [zenavif-serialize]) · [zenjxl] ([jxl-encoder] · [zenjxl-decoder]) · [zentiff] · [zenbitmaps] · [heic] · [zenraw] · [zenpdf] · [ultrahdr] · [mozjpeg-rs] · [webpx] |
| Compression | [zenflate] · [zenzop] |
| Processing | [zenresize] · [zenfilters] · [zenquant] · [zenblend] |
| Metrics | [zensim] · [fast-ssim2] · [butteraugli] · [resamplescope-rs] · [codec-eval] · [codec-corpus] |
| Pixel types & color | [zenpixels] · [zenpixels-convert] · [linear-srgb] · [garb] |
| Pipeline | [zenpipe] · [zencodec] · [zencodecs] · [zenlayout] · [zennode] |
| ImageResizer | [ImageResizer] (C#) — 24M+ NuGet downloads across all packages |
| [Imageflow][] | Image optimization engine (Rust) — [.NET][imageflow-dotnet] · [node][imageflow-node] · [go][imageflow-go] — 9M+ NuGet downloads across all packages |
| [Imageflow Server][] | [The fast, safe image server](https://www.imazen.io/) (Rust+C#) — 552K+ NuGet downloads, deployed by Fortune 500s and major brands |

<sub>* as of 2026</sub>

### General Rust awesomeness

[archmage] · [magetypes] · [enough] · [whereat] · [zenbench] · [cargo-copter]

[And other projects](https://www.imazen.io/open-source) · [GitHub @imazen](https://github.com/imazen) · [GitHub @lilith](https://github.com/lilith) · [lib.rs/~lilith](https://lib.rs/~lilith) · [NuGet](https://www.nuget.org/profiles/imazen) (over 30 million downloads / 87 packages)

## License

Dual-licensed: [AGPL-3.0](LICENSE-AGPL3) or [commercial](LICENSE-COMMERCIAL).

I've maintained and developed open-source image server software — and the 40+
library ecosystem it depends on — full-time since 2011. Fifteen years of
continual maintenance, backwards compatibility, support, and the (very rare)
security patch. That kind of stability requires sustainable funding, and
dual-licensing is how we make it work without venture capital or rug-pulls.
Support sustainable and secure software; swap patch tuesday for patch leap-year.

[Our open-source products](https://www.imazen.io/open-source)

**Your options:**

- **Startup license** — $1 if your company has under $1M revenue and fewer
  than 5 employees. [Get a key →](https://www.imazen.io/pricing)
- **Commercial subscription** — Governed by the Imazen Site-wide Subscription
  License v1.1 or later. Apache 2.0-like terms, no source-sharing requirement.
  Sliding scale by company size.
  [Pricing & 60-day free trial →](https://www.imazen.io/pricing)
- **AGPL v3** — Free and open. Share your source if you distribute.

See [LICENSE-COMMERCIAL](LICENSE-COMMERCIAL) for details.

Upstream code from [image-rs/image-webp](https://github.com/image-rs/image-webp) is licensed under MIT OR Apache-2.0.
Our additions and improvements are dual-licensed (AGPL-3.0 or commercial) as above.

### Upstream Contribution

We are willing to release our improvements under the original MIT OR Apache-2.0
license if upstream takes over maintenance of those improvements. We'd rather
contribute back than maintain a parallel codebase. Open an issue or reach out.
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

[zenjpeg]: https://github.com/imazen/zenjpeg
[zenpng]: https://github.com/imazen/zenpng
[zengif]: https://github.com/imazen/zengif
[zenavif]: https://github.com/imazen/zenavif
[zenjxl]: https://github.com/imazen/zenjxl
[zentiff]: https://github.com/imazen/zentiff
[zenbitmaps]: https://github.com/imazen/zenbitmaps
[heic]: https://github.com/imazen/heic-decoder-rs
[zenraw]: https://github.com/imazen/zenraw
[zenpdf]: https://github.com/imazen/zenpdf
[ultrahdr]: https://github.com/imazen/ultrahdr
[jxl-encoder]: https://github.com/imazen/jxl-encoder
[zenjxl-decoder]: https://github.com/imazen/zenjxl-decoder
[rav1d-safe]: https://github.com/imazen/rav1d-safe
[zenrav1e]: https://github.com/imazen/zenrav1e
[mozjpeg-rs]: https://github.com/imazen/mozjpeg-rs
[zenavif-parse]: https://github.com/imazen/zenavif-parse
[zenavif-serialize]: https://github.com/imazen/zenavif-serialize
[webpx]: https://github.com/imazen/webpx
[zenflate]: https://github.com/imazen/zenflate
[zenzop]: https://github.com/imazen/zenzop
[zenresize]: https://github.com/imazen/zenresize
[zenfilters]: https://github.com/imazen/zenfilters
[zenquant]: https://github.com/imazen/zenquant
[zenblend]: https://github.com/imazen/zenblend
[zensim]: https://github.com/imazen/zensim
[fast-ssim2]: https://github.com/imazen/fast-ssim2
[butteraugli]: https://github.com/imazen/butteraugli
[zenpixels]: https://github.com/imazen/zenpixels
[zenpixels-convert]: https://github.com/imazen/zenpixels
[linear-srgb]: https://github.com/imazen/linear-srgb
[garb]: https://github.com/imazen/garb
[zenpipe]: https://github.com/imazen/zenpipe
[zencodec]: https://github.com/imazen/zencodec
[zencodecs]: https://github.com/imazen/zencodecs
[zenlayout]: https://github.com/imazen/zenlayout
[zennode]: https://github.com/imazen/zennode
[Imageflow]: https://github.com/imazen/imageflow
[Imageflow Server]: https://github.com/imazen/imageflow-server
[imageflow-dotnet]: https://github.com/imazen/imageflow-dotnet
[imageflow-node]: https://github.com/imazen/imageflow-node
[imageflow-go]: https://github.com/imazen/imageflow-go
[ImageResizer]: https://github.com/imazen/resizer
[archmage]: https://github.com/imazen/archmage
[magetypes]: https://github.com/imazen/archmage
[enough]: https://github.com/imazen/enough
[whereat]: https://github.com/lilith/whereat
[zenbench]: https://github.com/imazen/zenbench
[cargo-copter]: https://github.com/imazen/cargo-copter
[resamplescope-rs]: https://github.com/imazen/resamplescope-rs
[codec-eval]: https://github.com/imazen/codec-eval
[codec-corpus]: https://github.com/imazen/codec-corpus
