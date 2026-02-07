# API Feedback - zencodecs Integration

**Date:** 2026-02-06
**Context:** Implementing WebP codec adapter in zencodecs (src/codecs/webp.rs)

## Issues Encountered

### 1. Method name: frame_count() vs num_frames()
**Issue:** Expected `demuxer.frame_count()` but method is actually `num_frames()`
**Solution:** Changed to use `num_frames()`
**Feedback:** Minor naming inconsistency. Other codecs (GIF, AVIF) commonly use "frame_count" terminology. Consider adding `frame_count()` as an alias to `num_frames()` for consistency.

### 2. Redundant animation check
**Issue:** Initially wrote `demuxer.num_frames() > 1` to check for animation
**Solution:** Discovered `is_animated()` method exists
**Feedback:** Not really an issue - just didn't notice `is_animated()` at first glance. The API is actually well-designed here.

## Current Implementation

```rust
// Probe
let decoder = zenwebp::WebPDecoder::new(data)?;
let (width, height) = decoder.dimensions();
let has_alpha = decoder.has_alpha();

let demuxer = zenwebp::WebPDemuxer::new(data)?;
let has_animation = demuxer.is_animated();
let frame_count = demuxer.num_frames();
let icc_profile = demuxer.icc_profile().map(|p| p.to_vec());

// Decode
let (pixels, width, height) = match output_layout {
    PixelLayout::Rgb8 => zenwebp::decode_rgb(data)?,
    PixelLayout::Rgba8 => zenwebp::decode_rgba(data)?,
    PixelLayout::Bgr8 => zenwebp::decode_bgr(data)?,
    PixelLayout::Bgra8 => zenwebp::decode_bgra(data)?,
};

// Encode
let webp_data = if lossless {
    let config = zenwebp::LosslessConfig::new();
    zenwebp::EncodeRequest::lossless(&config, pixels, webp_layout, width, height)
        .encode()?
} else {
    let quality = quality.unwrap_or(85.0).clamp(0.0, 100.0);
    let config = zenwebp::LossyConfig::new().with_quality(quality);
    zenwebp::EncodeRequest::lossy(&config, pixels, webp_layout, width, height)
        .encode()?
};
```

## What Worked Well

- **Convenience functions:** `decode_rgb()`, `decode_rgba()`, etc. are perfect for simple use cases
- **Clear config separation:** `LossyConfig` vs `LosslessConfig` makes the API intent clear
- **Comprehensive demuxer:** `WebPDemuxer` provides all the metadata needed (ICC, EXIF, XMP, animation info)
- **Builder pattern:** `EncodeRequest::lossy()` and `.with_quality()` is intuitive
- **Layout conversion:** `PixelLayout` enum matches zencodecs perfectly, minimal mapping needed

## Minor Suggestions

1. **Naming consistency:** Consider aliasing `num_frames()` → `frame_count()` to match common codec terminology
2. **Single-struct decode:** Currently need both `WebPDecoder` and `WebPDemuxer` for full metadata. Could `WebPDemuxer` provide dimensions/alpha too?

## Overall Assessment

**Very smooth integration.** zenwebp has the best API ergonomics of the three codecs integrated so far. The convenience functions and clear config types made the adapter implementation straightforward. Only one minor method name issue, and that's mostly about cross-codec consistency rather than a real problem with zenwebp itself.
