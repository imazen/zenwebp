<!-- GENERATED FROM README.md by zenutils gen-readme-crates.sh — DO NOT EDIT. -->

# zenwebp

zenwebp is a pure-Rust WebP codec — lossy (VP8) and lossless (VP8L) encode and
decode, plus alpha, animation, and ICC/EXIF/XMP metadata. No C dependencies,
`#![forbid(unsafe_code)]`, and `no_std`-compatible (just `alloc`). It is
SIMD-accelerated on x86 (SSE2/SSE4.1/AVX2), ARM64 (NEON), and WASM (SIMD128) via
runtime CPU detection, and tracks libwebp on both features and performance.

## Quick start

```toml
[dependencies]
zenwebp = "0.4.5"
```

```rust
use zenwebp::{EncodeRequest, LossyConfig, PixelLayout};

// Decode any WebP to packed RGBA8 (width * height * 4 bytes) in one call:
let (rgba, width, height) = zenwebp::oneshot::decode_rgba(webp_bytes)?;

// Encode 8-bit pixels to lossy WebP at quality 85 (RGB shown; RGBA/BGRA/L8/… also accepted):
let rgb: &[u8] = /* width * height * 3 bytes */;
let webp = EncodeRequest::lossy(&LossyConfig::new().with_quality(85.0), rgb, PixelLayout::Rgb8, width, height)
    .encode()?;
```

**New to zenwebp?** The [API guide](https://github.com/imazen/zenwebp/blob/main/examples/api_guide.rs)
demonstrates 100% of the public API with runnable examples.

## Decoding

```rust
// One-shot decode — always 4-channel RGBA8 (alpha = 255 for opaque images):
let (pixels, width, height) = zenwebp::oneshot::decode_rgba(webp_bytes)?;
```

The `oneshot` module also has `decode_rgb`, `decode_bgra`, `decode_argb`,
`decode_yuv420`, `decode_rgb565`, `decode_rgba4444`, premultiplied variants, and
`*_into` versions that decode into a caller-provided buffer.

For control over headers, dithering, upsampling, and limits, use `WebPDecoder`
for two-phase decoding (inspect headers before allocating):

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

**`read_image` output format.** `read_image` writes the image's *native* format:
packed **RGBA8** (4 bytes/pixel, `R,G,B,A`) when `info.has_alpha`, otherwise packed
**RGB8** (3 bytes/pixel, `R,G,B`). It does not take a format parameter — the channel
count follows the bitstream. The buffer must be **exactly** `output_buffer_size()`
bytes (`width * height * {3 or 4}`); any other length returns
`DecodeError::ImageTooLarge`. Branch on `info.has_alpha` before interpreting the
bytes. If you always want 4-channel output regardless of the source, use
`oneshot::decode_rgba` / `DecodeRequest::decode_rgba` instead (alpha is set to 255
for opaque images).

## Encoding

Build a reusable `LossyConfig` or `LosslessConfig` and drive it with
`EncodeRequest`. Lossy encode accepts any interleaved or planar `PixelLayout`:
`Rgb8`, `Rgba8`, `Bgr8`, `Bgra8`, `Argb8`, `L8`, `La8`, and `Yuv420` — pass the
variant that matches your buffer (e.g. `PixelLayout::Rgba8` for 4-channel RGBA, no
pre-conversion needed). Alpha-bearing layouts encode an alpha plane; the rest are
opaque. (See the [feature comparison](https://github.com/imazen/zenwebp#feature-comparison-with-libwebp)
for the full input-format matrix.)

```rust
use zenwebp::{LossyConfig, EncodeRequest, PixelLayout};

let rgb_pixels: &[u8] = /* your RGB data */;
let (width, height) = (800, 600);

let config = LossyConfig::new()
    .with_quality(85.0)
    .with_method(4);  // 0 = fast, 6 = best

let webp = EncodeRequest::lossy(&config, rgb_pixels, PixelLayout::Rgb8, width, height)
    .encode()?;
```

Lossless takes the same shape:

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

ICC/EXIF/XMP metadata can be embedded during encode with
`EncodeRequest::with_icc_profile` / `with_metadata`, or added to, read from, and
removed from already-encoded bytes via the `zenwebp::metadata` module (no pixel
re-encode). Animation is built with `mux::AnimationEncoder`.

## Transcode (decode → re-encode)

```rust
use zenwebp::{LossyConfig, EncodeRequest, PixelLayout};

let (rgba, w, h) = zenwebp::oneshot::decode_rgba(input_webp)?;
let cfg = LossyConfig::new().with_quality(75.0);
// Arg order: lossy(config, pixels, LAYOUT, width, height) — the pixel layout
// comes BEFORE the dimensions. `decode_rgba` always yields `PixelLayout::Rgba8`.
let out = EncodeRequest::lossy(&cfg, &rgba, PixelLayout::Rgba8, w, h).encode()?;
```

## Untrusted input: error handling & limits

Decode/encode return `Result<_, whereat::At<E>>` (`At<DecodeError>` /
`At<EncodeError>`). The `At<…>` wrapper records a build-time source location for
your logs; get the underlying error with `err.error()` (borrow) or
`err.into_inner()` (owned), then match the variant:

```rust
use zenwebp::DecodeError;

match zenwebp::oneshot::decode_rgba(webp_bytes) {
    Ok((pixels, w, h)) => { /* `pixels` is packed RGBA8: w*h*4 bytes, R,G,B,A */ }
    Err(e) => {
        // `e.location()` is the whereat capture site (file:line) — log it for triage:
        if let Some(loc) = e.location() {
            eprintln!("decode failed at {}:{}", loc.file(), loc.line());
        }
        match e.error() {
            DecodeError::Cancelled(_)          => eprintln!("cancelled / timed out"), // 499
            // pixel/dimension caps surface here; the message says which limit:
            DecodeError::InvalidParameter(msg) => eprintln!("rejected: {msg}"),  // 400/413
            DecodeError::MemoryLimitExceeded   => eprintln!("too large"),        // 413
            other => eprintln!("decode failed: {other:?}"),                      // 400/500
        }
    }
}
```

The one-shot helpers decode with `DecodeConfig::default()`, which already
enforces a server-safe `Limits` (≤120 MP, 16384×16384, 1 GB memory). To tighten
or relax the caps, build a `DecodeConfig` with your own `Limits` and drive the
decode explicitly with `DecodeRequest`:

```rust
use zenwebp::{DecodeConfig, DecodeRequest, Limits};

// `Limits` is `#[non_exhaustive]`; spread `..Limits::default()` to keep the other
// server-safe caps, then override the fields you care about. Fields are
// `Option<…>`; `None` means "no cap on this dimension".
let config = DecodeConfig::default().limits(Limits {
    max_total_pixels: Some(40_000_000),       // 40 MP hard cap
    max_memory: Some(256 * 1024 * 1024),      // 256 MB during decode
    max_width: Some(8192),
    max_height: Some(8192),
    ..Limits::default()
});

let (rgba, w, h) = DecodeRequest::new(&config, webp_bytes).decode_rgba()?;
```

`DecodeConfig` also has builder shortcuts for the common caps —
`DecodeConfig::default().max_dimensions(8192, 8192).max_memory(256 << 20)` — and
`Limits::none()` removes every cap (only for fully trusted input).

The two-phase `WebPDecoder` carries the same knobs — set them before `read_image`:

```rust
use zenwebp::{WebPDecoder, Limits};

let webp_bytes: &[u8] = /* untrusted WebP data */;
let mut decoder = WebPDecoder::build(webp_bytes)?;

// Reject anything bigger than your budget *before* allocating the output buffer.
decoder.set_limits(Limits {
    max_total_pixels: Some(40_000_000),       // 40 MP
    max_memory: Some(256 * 1024 * 1024),      // 256 MB
    ..Limits::default()                        // keeps the other server-safe caps
});

let mut output = vec![0u8; decoder.output_buffer_size().unwrap()];
decoder.read_image(&mut output)?; // errs if a limit is exceeded
```

## Untrusted input: cancellation (no thread killing)

Both decode and encode accept a cooperative `enough::Stop` token, so a
long-running operation on a hostile input can be aborted from another thread
without `kill`-ing it (which would leak the work-in-progress allocations). The
token types are re-exported at the crate root (`zenwebp::Stop` / `StopReason` /
`Unstoppable`), so you don't need to add `enough` to your own `Cargo.toml`. Wrap
your own cancel flag — an `AtomicBool`, a deadline, a request-aborted handle — in
a tiny `impl Stop`:

```rust
use core::sync::atomic::{AtomicBool, Ordering};
use zenwebp::{Stop, StopReason};
use zenwebp::{DecodeConfig, DecodeRequest, EncodeRequest, LossyConfig, PixelLayout};

struct CancelFlag<'a>(&'a AtomicBool);
impl Stop for CancelFlag<'_> {
    fn check(&self) -> Result<(), StopReason> {
        if self.0.load(Ordering::Relaxed) {
            Err(StopReason::Cancelled)   // also StopReason::TimedOut for deadlines
        } else {
            Ok(())
        }
    }
}

let cancelled = AtomicBool::new(false);
let stop = CancelFlag(&cancelled);
// (another thread / a timeout sets `cancelled` to true to abort)

// Cancellable decode: pass the token via `.stop(...)`.
let config = DecodeConfig::default();
let decoded = DecodeRequest::new(&config, webp_bytes).stop(&stop).decode_rgba();

// Cancellable encode: pass the token via `.with_stop(...)`.
let cfg = LossyConfig::new().with_quality(75.0);
let encoded = EncodeRequest::lossy(&cfg, &rgb_pixels, PixelLayout::Rgb8, w, h)
    .with_stop(&stop)
    .encode();
```

A cancelled operation returns `DecodeError::Cancelled(StopReason)` /
`EncodeError::Cancelled(StopReason)`. The token is checked periodically inside the
hot loops, so cancellation latency is bounded by a chunk of work, not the whole
image. The default (`Unstoppable`, used when you don't call `.stop(...)`) is
zero-cost. For a ready-made, thread-safe cancel/deadline token see
[`almost_enough::Stopper`](https://crates.io/crates/almost-enough).

## Features

- **Pure Rust** — no C dependencies, builds anywhere Rust does
- **`#![forbid(unsafe_code)]`** — memory safety guaranteed; SIMD via safe abstractions
- **`no_std` compatible** — works with just `alloc`, no standard library needed
- **SIMD accelerated** — SSE2/SSE4.1/AVX2 on x86, NEON on ARM64, SIMD128 on WASM
- **Full format support** — lossy, lossless, alpha, animation (encode + decode),
  ICC/EXIF/XMP metadata, EXIF orientation parsing, mux/demux, chroma dithering
- **Metadata module** — `zenwebp::metadata` extracts/embeds/removes ICC, EXIF, and
  XMP in encoded WebP bytes without decoding pixels
- **Server-safe** — memory/pixel limits and cooperative cancellation for untrusted input
- **zencodec integration** — optional `zencodec` feature for unified codec traits,
  streaming decode, color/metadata policy, and resource estimation

### Safe SIMD

We achieve both safety and performance through safe abstractions over CPU intrinsics:

- [`archmage`](https://crates.io/crates/archmage) and [`magetypes`](https://crates.io/crates/magetypes)
  — token-gated safe intrinsics with runtime CPU detection
- [`safe_unaligned_simd`](https://crates.io/crates/safe_unaligned_simd) — safe unaligned load/store

These abstractions may not be perfect, but we trust them over hand-rolled unsafe code.

### Decoder

Supports all WebP features: lossy and lossless compression, alpha channel,
animation, and the extended format with ICC/EXIF/XMP chunks. Output formats: RGB,
RGBA, BGR, BGRA, ARGB, YUV 4:2:0, RGB565, RGBA4444, and premultiplied
RGBA/BGRA/ARGB. Chroma dithering matches libwebp pixel-for-pixel.

### Encoder

Supports lossy and lossless encoding with configurable quality (0–100) and a
speed/quality tradeoff (method 0–6). Lossy adds content-aware presets, target file
size / target PSNR search, SNS, trellis quantization, and near-lossless; lossless
adds the full predictor / cross-color / palette / meta-Huffman transform pipeline.

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


## no_std support

```toml
[dependencies]
zenwebp = { version = "0.4.5", default-features = false }
```

Both encoder and decoder work without std. The decoder takes `&[u8]` slices and the
encoder writes to `Vec<u8>`. Only `EncodeRequest::encode_to` (writing to a
`std::io::Write`) requires the `std` feature.

## Credits

zenwebp started as a fork of [`image-webp`](https://github.com/image-rs/image-webp),
the pure-Rust WebP crate from the [image-rs](https://github.com/image-rs) project.
The original decoder and lossless encoder formed the foundation on which zenwebp was
built, and the lossless decoder still shares the most DNA with image-webp. We're
grateful to the image-rs maintainers for their well-structured, battle-tested
codebase — if you don't need lossy encoding, animation, or SIMD, their crate is a
smaller, simpler dependency.

From that foundation, zenwebp added a ground-up lossy encoder with full RD
optimization, a redesigned streaming decoder, SIMD acceleration via
[archmage](https://crates.io/crates/archmage) /
[magetypes](https://crates.io/crates/magetypes) (and
[safe_unaligned_simd](https://crates.io/crates/safe_unaligned_simd)), animation
encoding, and extensive optimization across all pipelines to reach libwebp feature
and performance parity.

- **[libwebp](https://chromium.googlesource.com/webm/libwebp)** (Google) —
  reference implementation. Our lossy encoder follows libwebp's algorithms for RD
  optimization, trellis quantization, and mode selection. The WebP format itself
  is Google's creation.
- **Claude** (Anthropic) — AI-assisted development.

Contributions are welcome — please open an issue or pull request. Code review is
recommended for production use.

## License

Dual-licensed:
[AGPL-3.0](https://github.com/imazen/zenwebp/blob/main/LICENSE-AGPL3) or
[commercial](https://github.com/imazen/zenwebp/blob/main/LICENSE-COMMERCIAL).

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

See [LICENSE-COMMERCIAL](https://github.com/imazen/zenwebp/blob/main/LICENSE-COMMERCIAL)
for details.

Upstream code from [image-rs/image-webp](https://github.com/image-rs/image-webp) is
licensed under MIT OR Apache-2.0. Our additions and improvements are dual-licensed
(AGPL-3.0 or commercial) as above. Versions 0.1.x–0.3.x of zenwebp were MIT OR
Apache-2.0.

## Image tech I maintain

| | |
|:--|:--|
| **Codecs** ¹ | [zenjpeg] · [zenpng] · **zenwebp** · [zengif] · [zenavif] · [zenjxl] · [zenbitmaps] · [heic] · [zentiff] · [zenpdf] · [zensvg] · [zenjp2] · [zenraw] · [ultrahdr] |
| Codec internals | [zenjxl-decoder] · [jxl-encoder] · [zenrav1e] · [rav1d-safe] · [zenavif-parse] · [zenavif-serialize] |
| Compression | [zenflate] · [zenzop] · [zenzstd] |
| Processing | [zenresize] · [zenquant] · [zenblend] · [zenfilters] · [zensally] · [zentone] |
| Pixels & color | [zenpixels] · [zenpixels-convert] · [linear-srgb] · [garb] |
| Pipeline & framework | [zenpipe] · [zencodec] · [zencodecs] · [zenlayout] · [zennode] · [zenwasm] · [zentract] |
| Metrics | [zensim] · [fast-ssim2] · [butteraugli] · [zenmetrics] · [resamplescope-rs] |
| Pickers & ML | [zenanalyze] · [zenpredict] · [zenpicker] |
| Products | [Imageflow] image engine ([.NET][imageflow-dotnet] · [Node][imageflow-node] · [Go][imageflow-go]) · [Imageflow Server] · [ImageResizer] (C#) |

<sub>¹ pure-Rust, `#![forbid(unsafe_code)]` codecs, as of 2026</sub>

### General Rust awesomeness

[zenbench] · [archmage] · [magetypes] · [enough] · [whereat] · [cargo-copter]

[Open source](https://www.imazen.io/open-source) · [@imazen](https://github.com/imazen) · [@lilith](https://github.com/lilith) · [lib.rs/~lilith](https://lib.rs/~lilith)

[zenjpeg]: https://github.com/imazen/zenjpeg
[zenpng]: https://github.com/imazen/zenpng
[zengif]: https://github.com/imazen/zengif
[zenavif]: https://github.com/imazen/zenavif
[zenjxl]: https://github.com/imazen/zenjxl
[zenbitmaps]: https://github.com/imazen/zenbitmaps
[heic]: https://github.com/imazen/heic
[zentiff]: https://github.com/imazen/zentiff
[zenpdf]: https://github.com/imazen/zenpdf
[zensvg]: https://github.com/imazen/zenextras
[zenjp2]: https://github.com/imazen/zenextras
[zenraw]: https://github.com/imazen/zenraw
[ultrahdr]: https://github.com/imazen/ultrahdr
[zenjxl-decoder]: https://github.com/imazen/zenjxl-decoder
[jxl-encoder]: https://github.com/imazen/jxl-encoder
[zenrav1e]: https://github.com/imazen/zenrav1e
[rav1d-safe]: https://github.com/imazen/rav1d-safe
[zenavif-parse]: https://github.com/imazen/zenavif-parse
[zenavif-serialize]: https://github.com/imazen/zenavif-serialize
[zenflate]: https://github.com/imazen/zenflate
[zenzop]: https://github.com/imazen/zenzop
[zenzstd]: https://github.com/imazen/zenzstd
[zenresize]: https://github.com/imazen/zenresize
[zenquant]: https://github.com/imazen/zenquant
[zenblend]: https://github.com/imazen/zenblend
[zenfilters]: https://github.com/imazen/zenfilters
[zensally]: https://github.com/imazen/zensally
[zentone]: https://github.com/imazen/zentone
[zenpixels]: https://github.com/imazen/zenpixels
[zenpixels-convert]: https://github.com/imazen/zenpixels
[linear-srgb]: https://github.com/imazen/linear-srgb
[garb]: https://github.com/imazen/garb
[zenpipe]: https://github.com/imazen/zenpipe
[zencodec]: https://github.com/imazen/zencodec
[zencodecs]: https://github.com/imazen/zencodecs
[zenlayout]: https://github.com/imazen/zenlayout
[zennode]: https://github.com/imazen/zennode
[zenwasm]: https://github.com/imazen/zenwasm
[zentract]: https://github.com/imazen/zentract
[zensim]: https://github.com/imazen/zensim
[fast-ssim2]: https://github.com/imazen/fast-ssim2
[butteraugli]: https://github.com/imazen/butteraugli
[zenmetrics]: https://github.com/imazen/zenmetrics
[resamplescope-rs]: https://github.com/imazen/resamplescope-rs
[zenanalyze]: https://github.com/imazen/zenanalyze
[zenpredict]: https://github.com/imazen/zenanalyze
[zenpicker]: https://github.com/imazen/zenanalyze
[zenbench]: https://github.com/imazen/zenbench
[archmage]: https://github.com/imazen/archmage
[magetypes]: https://github.com/imazen/archmage
[enough]: https://github.com/imazen/enough
[whereat]: https://github.com/lilith/whereat
[cargo-copter]: https://github.com/imazen/cargo-copter
[Imageflow]: https://github.com/imazen/imageflow
[Imageflow Server]: https://github.com/imazen/imageflow-dotnet-server
[ImageResizer]: https://github.com/imazen/resizer
[imageflow-dotnet]: https://github.com/imazen/imageflow-dotnet
[imageflow-node]: https://github.com/imazen/imageflow-node
[imageflow-go]: https://github.com/imazen/imageflow-go
