# Release Notes

### [Unreleased]

**Sharp YUV re-enabled, zenyuv integration, streaming push decoder, security fix**

#### Added
- Row-streaming push decoder for lossy VP8 — `WebPPushDecoder` emits rows as they decode (f4993a1)
- NEON and WASM128 SIMD for lossless color transforms, both decoder and encoder paths (cff6c14)
- Sharp YUV re-enabled with gamma-corrected chroma seeding and Y refinement, configurable via `Option<SharpYuvConfig>` (5e1cf9f, 558c41f, 0990023)
- `RGBX8` / `BGRX8` pixel descriptors accepted in encode dispatch (23b89b8)
- `SHARP-YUV.md` documentation covering the three sharp YUV modes and evaluation data (920e344)

#### Changed
- Swapped the `yuv` crate for `zenyuv` with `garb` strided preprocessing; `zenyuv` is now an unconditional dependency and the `fast-yuv` feature has been dropped (6e248aa, 29c0b79)
- Single-frame animation inputs now downgrade to a static WebP to match libwebp behavior (3f1f08a)
- In lossless encode, RGB channels are zeroed where alpha is 0 by default, matching libwebp (f5744d1)

#### Fixed
- `WebpAnimationFrameEncoder::push_frame` now honors `PixelSlice` stride instead of assuming tightly-packed rows (d8cbbec)
- Eliminated green shift that appeared when sharp YUV was enabled (7a23ffa)
- Restored `#[rite]` attributes on SSE2 encoder functions and wrapped tests with `#[arcane]` (1f066f1)

#### Performance
- Non-sharp YUV path accelerated via `zenyuv`, including a Y-only kernel for luma-first passes; sharp path allocation reuse (4f6960f, 91b84af)
- Gamma-corrected chroma seeding uses the magetypes scalar-LUT-bookend SIMD pattern (4089ddf)
- Removed double SIMD dispatch in SSE encoder hot paths and dropped dead SSE wrapper functions (85a14c5, ee31003)
- Removed `#[arcane]` entry shims on WASM and unnecessary `#[rite]` wrappers for color transforms, letting archmage conventions handle dispatch directly (93506eb, 170831a, 1a15fa5, 93431cf, bc8358c)
- `is_lossy_streaming_candidate` reuses `detect::probe()` instead of re-parsing headers (7284ba9)

#### Security
- `rand` updated 0.9.2 → 0.9.3 to pick up GHSA-cq8v-f236-94qc; added RGBA lossless roundtrip regression tests alongside the update (cafaee6)
- Also bumped `rand` 0.10.0 → 0.10.1 (06ed69e)

#### Tests
- libwebp-golden RGBA lossless regression suite (#15) (6e3e5f6)
- `zenyuv` parity and `probe` parity tests ported from #9 (09c219d)
- Streaming push decoder parity tests (8c13c95)
- Trimmed `large_encode_roundtrip` from 18 to 5 cases to keep CI responsive (d187459)
- Reproducers for imageflow RGBA-vs-BGRA lossless routing and byte-exact-vs-libwebp rose decode (62e9400, 0262745, b248b61)

### Version 0.4.2

**Security fixes, SIMD improvements, i686 correctness**

#### Security Fixes (PR #4)
- Default `WebPDecoder` to `Limits::default()` instead of `Limits::none()` — prevents unbounded memory use on adversarial input
- Wire up frame count checking during ANMF chunk parsing
- Cap `num_huff_groups` to prevent memory amplification from malformed lossless streams
- Replace `assert!` panics with error returns in lossless transforms and Huffman table build
- Bounds-check VP8 chunk range before slicing

#### SIMD Additions
- NEON and WASM128 dispatch for lossless predictor transforms and `add_green_to_blue_and_red`
- NEON and WASM128 dispatch for encoder lossless transforms
- Lossless inverse transforms upgraded to AVX2 (v3) via `u8x32` wide SIMD
- Monolithic V3 `#[arcane]` entry with `#[rite]` wrappers for all predictors — enables LLVM to autovectorize scalar fallback loops with AVX2 target_feature
- Fused chroma-upsample + YUV→RGB kernel wired into v2 decoder

#### Bug Fixes
- Fix chroma interpolation in scalar fallback for `fused_row_2uv` — 4-tap interpolation was collapsed to 2-tap for even luma positions, producing wrong pixel values on i686 (all YUV bit-exact tests now pass on 32-bit targets)

### Version 0.4.0

**BREAKING CHANGES — API reorganization, new features**

#### API Reorganization

The crate root is now lean — types live in their natural namespaces.

**Moved to `zenwebp::oneshot::`:**
- All 16 `decode_*` convenience functions (`decode_rgba`, `decode_rgb`, etc.)

**Moved to `zenwebp::mux::`** (no longer re-exported at root):
- `AnimationConfig`, `AnimationEncoder`, `AnimationDecoder`, `WebPDemuxer`,
  `WebPMux`, `BlendMethod`, `DisposeMethod`, `MuxError`, `MuxResult`,
  `AnimFrame`, `DemuxFrame`, `MuxFrame`, `AnimationInfo`

**Moved to `zenwebp::encoder::`** (no longer re-exported at root):
- `EncodeStats`, `EncodeProgress`, `NoProgress`, `ClassifierDiag`,
  `ImageContentType` (renamed from `ContentType`)

**Moved to `zenwebp::decoder::`** (no longer re-exported at root):
- `BitstreamFormat`, `LoopCount`, `StreamStatus`, `UpsamplingMethod`,
  `YuvPlanes`, `AnimationFrame`

**Moved to `zenwebp::zencodec::`** (feature-gated):
- All `Webp*` zencodec trait adapters

**Removed from root** (import from `enough` directly):
- `Stop`, `StopReason`, `Unstoppable`

**Removed from public API:**
- `DecoderContext` — now `pub(crate)`, buffer reuse happens internally

**Renamed:**
- `ContentType` → `ImageContentType`

**Stays at root:**
- `DecodeConfig`, `DecodeError`, `DecodeRequest`, `DecodeResult`, `ImageInfo`,
  `Limits`, `StreamingDecoder`, `WebPDecoder`
- `EncodeError`, `EncodeRequest`, `EncodeResult`, `EncoderConfig`,
  `LossyConfig`, `LosslessConfig`, `ImageMetadata`, `PixelLayout`, `Preset`
- `Orientation` (re-exported from `zenpixels`)

#### Migration

```rust
// 0.3.x
use zenwebp::{decode_rgba, AnimationDecoder, MuxError, Stop};

// 0.4.0
use zenwebp::oneshot::decode_rgba;
use zenwebp::mux::{AnimationDecoder, MuxError};
use enough::Stop;
```

#### New Features

- **EXIF orientation parsing** — `ImageInfo.orientation` is automatically
  extracted from the EXIF chunk. Returns `Option<zenpixels::Orientation>`.
- **`zenwebp::Orientation`** re-exported from `zenpixels` (canonical D4
  dihedral group type for the zen ecosystem).
- **ICC profile documentation** — crate-level docs explain the full ICC
  pipeline (extract, embed, post-hoc via `metadata` module).
- **12,825-file corpus validation** — pixel-exact match against libwebp
  on the full scraped WebP corpus.

#### NEON (ARM64) SIMD

- **NEON YUV→RGB conversion** — fused upsample + color conversion for decode,
  supporting both RGB and RGBA output. Uses `vst3q_u8`/`vst4q_u8` for
  hardware-accelerated pixel interleaving.
- **Fixed NEON loop filter transpose bug** — horizontal filter load functions
  (`load_4x16_neon`, `load_4x8x2_neon`) used a `vtrnq_u8`/`vtrnq_u16`
  byte-level transpose that produced scrambled element ordering, while the
  corresponding stores wrote sequentially. This caused incorrect loop filter
  results on aarch64 (max_diff=138 vs libwebp).
- Decoder is now **bit-exact with libwebp on aarch64** (verified via QEMU
  cross-testing against webpx).

#### Bug Fixes

- **Default dithering strength changed from 50 to 0** — libwebp defaults to
  0 in both simple and advanced APIs. Our default of 50 caused spurious
  chroma diffs (max=4) in every comparison against libwebp/webpx, masked by
  pixel-perfect tests explicitly overriding to 0.

#### Internal Improvements

- `common`, `vp8`, `vp8v2` modules now `pub(crate)`
- Encoder sub-modules `#[doc(hidden)]` (still accessible, hidden from docs)
- All v1/v2 naming artifacts removed (the v1 decoder was deleted previously)
- `MbRowEntry` fields narrowed to `pub(super)`
- `zenpixels` is now a required (non-optional) dependency

### Version 0.3.0

**BREAKING CHANGES - Complete API overhaul for type safety and convergence**

This release completes the API convergence initiative, bringing zenwebp in line with modern Rust
codec design patterns. **No backward compatibility** - this is a clean break from 0.2.x.

#### Type-Safe Encoder Configuration

The unified `EncoderConfig` struct has been split into separate lossy and lossless types:

**Old API (0.2.x):**
```rust
let config = EncoderConfig::new().quality(85.0).lossless(false);
```

**New API (0.3.0):**
```rust
// Compile-time mode selection
let config = LossyConfig::new().with_quality(85.0);
let webp = EncodeRequest::lossy(&config, pixels, layout, w, h).encode()?;

// Or runtime selection
let config = EncoderConfig::new_lossy().with_quality(85.0);
let webp = EncodeRequest::new(&config, pixels, layout, w, h).encode()?;
```

This prevents setting invalid parameter combinations (e.g., `sns_strength` on lossless) at compile time.

#### Builder Convention - with_ Prefix

All builder methods now use `with_` prefix following Rust conventions:
- `quality()` → `with_quality()`
- `method()` → `with_method()`
- `sns_strength()` → `with_sns_strength()`
- `stop()` → `with_stop()`
- `progress()` → `with_progress()`
- 25+ other method renames

#### ImageMetadata Struct

Metadata is now grouped in a dedicated struct instead of individual request methods:
```rust
let metadata = ImageMetadata::new()
    .with_icc_profile(&icc_data)
    .with_exif(&exif_data);

EncodeRequest::lossy(&config, pixels, layout, w, h)
    .with_metadata(metadata)
    .encode()?;
```

#### Memory Estimation

New methods on all config types:
```rust
let peak_mem = config.estimate_memory(width, height, bpp);
let worst_case = config.estimate_memory_ceiling(width, height, bpp);
```

#### Image Probing

Fast header-only parsing without decoding:
```rust
let info = ImageInfo::from_bytes(webp_data)?;  // NEW: Probing API
println!("{}x{}", info.width, info.height);

// Minimum bytes needed
ImageInfo::PROBE_BYTES  // 64 bytes
```

#### Two-Phase Decoder

Explicit build → inspect → decode pattern:
```rust
// Parse headers
let decoder = WebPDecoder::build(data)?;

// Inspect metadata (zero-cost)
let info = decoder.info();

// Decode (no re-parsing)
let (pixels, w, h) = decoder.decode_rgba()?;
```

#### Migration Guide

**Encoder:**
- Replace `EncoderConfig::new()` with `LossyConfig::new()` or `LosslessConfig::new()`
- Add `with_` prefix to all builder methods
- Use `EncodeRequest::lossy()` or `EncodeRequest::lossless()` instead of `::new()`

**Decoder:**
- Use `WebPDecoder::build()` instead of `::new()` for clarity (though `new()` still works)
- Use `decoder.info()` to get `ImageInfo` instead of calling individual methods

**Full API guide:** See `examples/api_guide.rs` for 100% coverage of the new API.

> **Note:** The quantization features described in v0.3.0 were later removed.

#### Color Quantization Backend Choice

The `quantize` feature now supports two backends with different licensing:

```toml
# MIT/Apache-2.0 (default) - uses quantizr
zenwebp = { version = "0.3", features = ["quantize"] }

# GPL-3.0-or-later - uses imagequant (better quality)
zenwebp = { version = "0.3", features = ["quantize-imagequant"] }
```

**Backend comparison:**
- **`quantize-quantizr`** (default): MIT-licensed, decent quality, no licensing restrictions
- **`quantize-imagequant`**: GPL-3.0-or-later, **dramatically better file sizes** via more compressible
  dithering and quantization patterns, requires GPL licensing

Choose based on your licensing requirements. The API is identical regardless of backend.

#### Other Changes

- Added `ColorType` → `PixelLayout` rename (ColorType deprecated)
- Added `Limits` support on encoder side
- Renamed `finish()` → `encode()` on `EncodeRequest`
- All examples and benchmarks updated

### Version 0.2.4

Changes:
 - Changed default upscaling to bilinear interpolation to match libwebp (#147)

Bug fixes:
 - Fixed all remaining divergences against libwebp in loop filtering (#148, #149)

Optimizations:
 - Optimized predictors in lossless_transform (#152)
 - Improved performance of horizontal loop filtering (#151, #156)


### Version 0.2.3

Changes:
 - Do not reject images with ICC profile bit set but missing ICCP chunk (#143)

Bug Fixes:
 - Fixed a bug that caused the last chroma macroblock in the image to be sometimes decoded incorrectly (#144)

### Version 0.2.2

Changes:
 - Do not apply background color to animated images by default to better match libwebp behavior (#135)

Bug Fixes:
 - Fixed a bug in the loop filter causing subtly but noticeably incorrect decoding of some lossy images (#140)

Optimizations:
 - Remove bounds checks from color transform hot loop (#133)
 - Optimize resolving indexed images into RGB colors (#132, #134)

### Version 0.2.1

Changes:
 - Increased the required Rust compiler version to v1.80

Optimizations:
 - Removed bounds checks from hot loops in `read_coefficients()` (#121)
 - Faster YUV -> RGBA conversion for a 7% speedup on lossy RGBA images (#122)
 - Faster alpha blending for up to 20% speedup on animated images (#123)
 - Much faster arithmetic decoding for up to 30% speedup on lossy images (#124)
 - Avoid unnecessarily cloning image data for a 4% speedup (#126)

### Version 0.2.0

Breaking Changes:
- `WebPDecoder` now requires the passed reader implement `BufRead`.

Changes:
- Add `EncoderParams` to make predictor transform optional.

Bug Fixes:
- Several bug fixes in animation compositing.
- Fix indexing for filling image regions with tivial huffman codes.
- Properly update the color cache when trivial huffman codes are used.

Optimizations:
- Substantially faster decoding of lossless images, by switching to a
  table-based Huffman decoder and a variety of smaller optimizations.

### Version 0.1.3

Changes:
- Accept files with out-of-order "unknown" chunks.
- Switched to `quick-error` crate for faster compliation.

Bug Fixes:
- Fixed decoding of animations with ALPH chunks.
- Fixed encoding bug for extended WebP files.
- Resolved compliation problem with fuzz targets.

Optimizations:
- Faster YUV to RGB conversion.
- Improved `BitReader` logic.
- In-place decoding of lossless RGBA images.

### Version 0.1.2

- Export `decoder::LoopCount`.
- Fix decode bug in `read_quantization_indices` that caused some lossy images to
  appear washed out.
- Switch to `byteorder-lite` crate for byte order conversions.

### Version 0.1.1

- Fix RIFF size calculation in encoder.

### Version 0.1.0

- Initial release
