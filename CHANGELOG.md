# Changelog

This file covers the 0.4.x maintenance line (`release/0.4.x`). Development of
0.5 and later happens on `main`, which keeps its own changelog. Format: Keep a
Changelog; versions follow Cargo semver.

## [Unreleased]

### Fixed

- Container walk (#107, #108, #109), backported from main and checked against libwebp in `tests/container_walk.rs` (all 4 tests fail without the fix). Trailing data after a VP8X file no longer fails the decode, and a chunk after the RIFF end is no longer read as metadata (the chunk walk ran ten bytes past the RIFF end). An odd-sized ANMF, or any other chunk between ANMF frames, no longer ends the animation: `read_frame` steps by the padded ANMF size, skips non-ANMF chunks, and bounds sub-chunks by the padded ANMF size as libwebp does.

### QUEUED BREAKING CHANGES

(none — 0.4.x takes only semver-compatible fixes)

## [0.4.6] - 2026-10-24

Bug-fix release on top of 0.4.4 (0.4.5 was yanked). Every fix is backported
from `main` unless marked *new*, and each landed with a regression test that
was watched to fail without it.

### Fixed

- **Lossless encoder emitted corrupt streams (#72, #118).** A histogram
  cluster stranded by the remap still got a Huffman group, so decoders parsed
  the surplus trees as pixel data and every pixel decoded wrong, at default
  settings (c0f7fbf). A debug-build emit-time invariant now pins it (ad3a6bb,
  dee3ec2).
- **Demuxer panics and hangs on untrusted input.** Truncated or oversized
  ICCP/ANIM/ANMF/ALPH/VP8/VP8L/EXIF/XMP chunks no longer slice out of bounds;
  32-bit chunk walks no longer wrap and spin; malformed ANMF frames are
  rejected at parse; VP8L simple-format dimensions and ANMF VP8L `has_alpha`
  are read correctly. All 36,051 fuzz-farm `demux_container` crash inputs now
  replay clean (0902b88).
- **Valid lossless files failed to decode** with a spurious `BitStreamError`
  when all four channel trees were deep (88ac2d8), and 16384-px lossless
  images wrapped to 0 width (0ccdaa3).
- **Decode limits were ignored.** `DecodeConfig::limits` dimension / pixel /
  frame caps were applied at default values during header parse (fd3eae6),
  lossless transform and ALPH allocations bypassed `max_memory` (9cefec9),
  `DecodeRequest::decode_yuv420` ignored the request's limits (c9513a9), and
  `AnimationDecoder::decode_all` did not bound the retained total (499b714).
- **Animated decode panicked** on an ALPH frame whose VP8 dimensions differ
  from its ANMF frame (a31b91b).
- **Animation compositing matches libwebp's `WebPAnimDecoder`.** Keyframes
  are copied instead of blended, disposal applies even when the next frame
  has no alpha (stale pixels survived), and the blend uses libwebp's
  arithmetic. `has_alpha()` is true for animations whose compositing can
  leave transparent pixels (092d570).
- **`UpsamplingMethod::Simple` (no-fancy upsampling) was ignored**; it now
  matches libwebp's point sampling (092d570).
- **wasm32 `+simd128` decoded every lossy image wrong** (YUV→RGB kernel
  discarded every sample; loop filter base delta wrong) (6995404).
- **aarch64 NEON and wasm SIMD128 encoders used wrong distortion values**
  (tdisto transpose, chroma SSE stride), so they encoded worse and differently
  from x86 (a8b8871). The palette color-cache ceiling differed between 32-
  and 64-bit builds (f6c5cef).
- **Cancellation ignored on lossy decode.** A pre-stopped token decoded lossy
  content successfully through the zencodec adapter, `DecodeRequest` and
  `WebPDecoder::read_frame` (f3dd09d; the native-path checks are *new*).
- **Encoder options silently dropped.** `with_near_lossless` never reached
  the VP8L encoder (db48a1a), which in turn needed the palette-index guard so
  near-lossless cannot corrupt palette images (2cbee57);
  `EncoderConfig::with_lossless` dropped `alpha_quality` and `target_size`
  (f0e0650).
- **Out-of-range quality panicked the encoder** instead of clamping (dc41fb1).
- **Streaming `push_rows` encoded different pixels than one-shot `encode`**
  for lossy RGB8 (37eba3e).
- `reset_animation` composited onto a stale canvas; `max_frame_count(n)`
  rejected n-frame animations; `WebPMux` offset/canvas arithmetic could
  overflow or underflow; `histo_bits` was re-clamped to 8 (2cbee57).
  `StreamingDecoder::append` re-parsed headers on every append (499b714).

### Changed (behaviour)

- `Limits::max_file_size` (default 100 MB, as documented) is now enforced on
  native decodes and the `StreamingDecoder` buffer; decodes above it return
  `DecodeError::InvalidParameter` (499b714).
- `WebPMux::push_frame` rejects durations above the 24-bit ANMF field with
  the new `MuxError::FrameDurationTooLarge` (`MuxError` is
  `#[non_exhaustive]`) instead of truncating (499b714).
- The zencodec streaming decoder rejects `UpsamplingMethod::Simple` so
  callers fall back to the full decode (092d570).

### Not backported (deliberately)

- RGB565 / RGBA4444 byte order and premultiplied rounding changes (main
  31b69a3): 0.4.4 documents little-endian 16-bit output.
- `record_coeffs()` removal (public function), fallible-allocation policy
  (#63, new API), memory-estimate recalibration (fitted to main's encoder),
  L8 lossless routing (#57, size only), whole-file streaming admission.
