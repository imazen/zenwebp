# Release Notes

### Version 0.4.0

**BREAKING CHANGE - License Change**

zenwebp is now licensed under **AGPL-3.0-or-later** (was MIT OR Apache-2.0).

**Why AGPL?**
- Ensures improvements benefit the community, especially for server-side/SaaS deployments
- Aligns with the project's goal of providing professional-grade WebP processing
- Commercial licenses available for closed-source use: support@imazen.io

**Migration:**
- If you're using zenwebp in open-source software: No action needed (AGPL compatible)
- If you're using zenwebp in closed-source/proprietary software: Contact us for commercial licensing
- If you need to stay on MIT/Apache-2.0: Use version 0.3.x (maintenance only)

No API changes in this release - purely a licensing update.

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
