# zenwebp

[![crates.io](https://img.shields.io/crates/v/zenwebp.svg)](https://crates.io/crates/zenwebp)
[![Documentation](https://docs.rs/zenwebp/badge.svg)](https://docs.rs/zenwebp)
[![Build Status](https://github.com/imazen/zenwebp/workflows/Rust%20CI/badge.svg)](https://github.com/imazen/zenwebp/actions)

High-performance WebP encoding and decoding in pure Rust, forked from `image-webp`.

## Current Status

* **Decoder:** Supports all WebP format features including both lossless and
  lossy compression, alpha channel, and animation. Both the "simple" and
  "extended" formats are handled, and it exposes methods to extract ICC, EXIF,
  and XMP chunks. Decoding speed is approximately **70%** of libwebp.

* **Encoder:** Supports both **lossy and lossless** encoding. The lossy encoder
  includes RD-optimized mode selection, trellis quantization, and SIMD
  acceleration. Encoding speed is approximately **40%** of libwebp with
  comparable quality.

## Features

- Pure Rust implementation (no C dependencies)
- SIMD acceleration via `archmage` (SSE2/SSE4.1/AVX2)
- Lossy encoding with full mode search (I16, I4, UV modes)
- Lossless encoding
- Animation support (decode)
- Alpha channel support
- ICC, EXIF, XMP metadata extraction

## Usage

```rust
use zenwebp::{WebPDecoder, WebPEncoder, EncoderParams};

// Decode
let decoder = WebPDecoder::new(reader)?;
let image = decoder.decode()?;

// Encode lossy
let encoder = WebPEncoder::new_with_params(writer, EncoderParams::lossy(75));
encoder.encode(width, height, color_type, &data)?;

// Encode lossless
let encoder = WebPEncoder::new_with_params(writer, EncoderParams::lossless());
encoder.encode(width, height, color_type, &data)?;
```

## Performance

Benchmarks on 768x512 Kodak image at Q75:

| Encoder | Time | Throughput |
|---------|------|------------|
| zenwebp | 66ms | 5.9 MPix/s |
| libwebp | 25ms | 15.6 MPix/s |

| Decoder | Time | Throughput |
|---------|------|------------|
| zenwebp | 4.2ms | 93 MPix/s |
| libwebp | 3.0ms | 129 MPix/s |

## License

Licensed under either of Apache License, Version 2.0 or MIT license at your option.
