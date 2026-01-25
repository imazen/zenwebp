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
let mut rgba = vec![0u8; decoder.output_buffer_size()?];
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

---

*Developed with assistance from Claude (Anthropic). Code review recommended for production use.*
