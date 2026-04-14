//! zencodec trait implementations for zenwebp.
//!
//! Provides [`WebpEncoderConfig`] and [`WebpDecoderConfig`] types that implement
//! the 4-layer trait hierarchy from zencodec, wrapping the native zenwebp API.
//!
//! The native API remains untouched — this is a thin adapter layer.
//!
//! # Trait mapping
//!
//! | zencodec | zenwebp adapter |
//! |----------------|-----------------|
//! | `EncoderConfig` | [`WebpEncoderConfig`] |
//! | `EncodeJob<'a>` | [`WebpEncodeJob`] |
//! | `Encoder` | [`WebpEncoder`] |
//! | `AnimationFrameEncoder` | [`WebpAnimationFrameEncoder`] |
//! | `DecoderConfig` | [`WebpDecoderConfig`] |
//! | `DecodeJob<'a>` | [`WebpDecodeJob`] |
//! | `Decode` | [`WebpDecoder`] |
//! | `AnimationFrameDecoder` | [`WebpAnimationFrameDecoder`] |

use alloc::borrow::Cow;
use alloc::sync::Arc;
use alloc::vec::Vec;

use whereat::{At, ResultAtExt, at};
use zencodec::decode::{AnimationFrame, DecodeOutput, OutputInfo, OwnedAnimationFrame, SinkError};
use zencodec::encode::EncodeOutput;
use zencodec::{
    ImageFormat, ImageInfo, ImageSequence, Metadata, Orientation, ResourceLimits,
    UnsupportedOperation,
};
use zenpixels::{PixelBuffer, PixelDescriptor, PixelSlice};

use crate::encoder::config::EncoderConfig;
use crate::mux::{AnimationConfig, AnimationDecoder, AnimationEncoder, MuxError};
use crate::{DecodeConfig, DecodeError, DecodeRequest, EncodeError, EncodeRequest, PixelLayout};

// ── Encoding ────────────────────────────────────────────────────────────────

/// WebP encoder configuration implementing [`zencodec::encode::EncoderConfig`].
///
/// Wraps the native [`EncoderConfig`] (lossy/lossless enum) and tracks
/// universal quality/effort settings for the trait interface.
///
/// # Examples
///
/// ```rust,ignore
/// use zencodec::encode::EncoderConfig;
/// use zenwebp::zencodec::WebpEncoderConfig;
///
/// let enc = WebpEncoderConfig::lossy()
///     .with_quality(85.0)
///     .with_sharp_yuv(true);
/// ```
#[derive(Clone, Debug)]
pub struct WebpEncoderConfig {
    inner: EncoderConfig,
    /// Trait-level effort (0-10 signed scale).
    trait_effort: Option<i32>,
    /// Trait-level calibrated quality (0.0-100.0).
    trait_quality: Option<f32>,
}

impl WebpEncoderConfig {
    /// Create a lossy encoder config with defaults.
    #[must_use]
    pub fn lossy() -> Self {
        Self {
            inner: EncoderConfig::new_lossy(),
            trait_effort: None,
            trait_quality: None,
        }
    }

    /// Create a lossless encoder config with defaults.
    #[must_use]
    pub fn lossless() -> Self {
        Self {
            inner: EncoderConfig::new_lossless(),
            trait_effort: None,
            trait_quality: None,
        }
    }

    /// Create from a preset with the given quality.
    #[must_use]
    pub fn with_preset(preset: crate::Preset, quality: f32) -> Self {
        Self {
            inner: EncoderConfig::with_preset(preset, quality),
            trait_effort: None,
            trait_quality: None,
        }
    }

    /// Set encoding quality (0.0 = smallest, 100.0 = best).
    #[must_use]
    pub fn with_quality(mut self, quality: f32) -> Self {
        self.inner = self.inner.with_quality(quality);
        self
    }

    /// Set quality/speed tradeoff (0-10, mapped to WebP method 0-6).
    #[must_use]
    pub fn with_effort_u32(mut self, effort: u32) -> Self {
        let method = ((effort as u64 * 6) / 10).min(6) as u8;
        self.inner = self.inner.with_method(method);
        self
    }

    /// Enable or disable lossless encoding (inherent method).
    #[must_use]
    pub fn with_lossless_mode(mut self, lossless: bool) -> Self {
        self.inner = self.inner.with_lossless(lossless);
        self
    }

    /// Set alpha plane quality (0.0-100.0).
    #[must_use]
    pub fn with_alpha_quality_value(mut self, quality: f32) -> Self {
        let aq = quality.clamp(0.0, 100.0) as u8;
        match &mut self.inner {
            EncoderConfig::Lossy(cfg) => cfg.alpha_quality = aq,
            EncoderConfig::Lossless(cfg) => cfg.alpha_quality = aq,
        }
        self
    }

    /// Enable sharp YUV conversion (lossy only).
    #[must_use]
    pub fn with_sharp_yuv(mut self, enable: bool) -> Self {
        self.inner = self.inner.with_sharp_yuv(enable);
        self
    }

    /// Set SNS strength (lossy only, 0-100).
    #[must_use]
    pub fn with_sns_strength(mut self, strength: u8) -> Self {
        self.inner = self.inner.with_sns_strength(strength);
        self
    }

    /// Set near-lossless preprocessing (lossless only, 0-100, 100=off).
    #[must_use]
    pub fn with_near_lossless(mut self, value: u8) -> Self {
        self.inner = self.inner.with_near_lossless(value);
        self
    }

    /// Preserve exact RGB values under fully transparent pixels (lossless only).
    ///
    /// No-op if the config is lossy.
    #[must_use]
    pub fn with_exact(mut self, exact: bool) -> Self {
        if let crate::encoder::config::EncoderConfig::Lossless(ref mut cfg) = self.inner {
            *cfg = cfg.clone().with_exact(exact);
        }
        self
    }

    /// Set filter strength (lossy only, 0-100).
    #[must_use]
    pub fn with_filter_strength(mut self, strength: u8) -> Self {
        self.inner = self.inner.with_filter_strength(strength);
        self
    }

    /// Set filter sharpness (lossy only, 0-7).
    #[must_use]
    pub fn with_filter_sharpness(mut self, sharpness: u8) -> Self {
        self.inner = self.inner.with_filter_sharpness(sharpness);
        self
    }

    /// Set target file size in bytes (lossy/lossless). 0 = disabled.
    #[must_use]
    pub fn with_target_size(mut self, bytes: u32) -> Self {
        self.inner = self.inner.with_target_size(bytes);
        self
    }

    /// Set target PSNR in dB (lossy only). 0.0 = disabled.
    #[must_use]
    pub fn with_target_psnr(mut self, psnr: f32) -> Self {
        self.inner = self.inner.with_target_psnr(psnr);
        self
    }

    /// Set number of segments (lossy only, 1-4).
    #[must_use]
    pub fn with_segments(mut self, segments: u8) -> Self {
        self.inner = self.inner.with_segments(segments);
        self
    }

    /// Set content-aware preset (lossy only).
    #[must_use]
    pub fn with_preset_value(mut self, preset: crate::Preset) -> Self {
        if let crate::encoder::config::EncoderConfig::Lossy(ref mut cfg) = self.inner {
            *cfg = cfg.clone().with_preset_value(preset);
        }
        self
    }

    /// Set encoding method directly (0-6). Prefer `with_effort_u32` for normalized 0-100 input.
    #[must_use]
    pub fn with_method(mut self, method: u8) -> Self {
        self.inner = self.inner.with_method(method);
        self
    }

    /// Access the underlying [`EncoderConfig`].
    #[must_use]
    pub fn inner(&self) -> &EncoderConfig {
        &self.inner
    }

    /// Mutably access the underlying [`EncoderConfig`].
    pub fn inner_mut(&mut self) -> &mut EncoderConfig {
        &mut self.inner
    }

    /// Set calibrated quality (inherent convenience, delegates to trait).
    #[must_use]
    pub fn with_calibrated_quality(self, quality: f32) -> Self {
        <Self as zencodec::encode::EncoderConfig>::with_generic_quality(self, quality)
    }

    /// Get calibrated quality (inherent convenience, delegates to trait).
    pub fn calibrated_quality(&self) -> Option<f32> {
        <Self as zencodec::encode::EncoderConfig>::generic_quality(self)
    }

    /// Set effort (inherent convenience, delegates to trait).
    #[must_use]
    pub fn with_effort(self, effort: i32) -> Self {
        <Self as zencodec::encode::EncoderConfig>::with_generic_effort(self, effort)
    }

    /// Get effort (inherent convenience, delegates to trait).
    pub fn effort(&self) -> Option<i32> {
        <Self as zencodec::encode::EncoderConfig>::generic_effort(self)
    }
}

static ENCODE_DESCRIPTORS: &[PixelDescriptor] = &[
    PixelDescriptor::RGB8_SRGB,
    PixelDescriptor::RGBA8_SRGB,
    PixelDescriptor::BGRA8_SRGB,
];

static ENCODE_CAPABILITIES: zencodec::encode::EncodeCapabilities =
    zencodec::encode::EncodeCapabilities::new()
        .with_icc(true)
        .with_exif(true)
        .with_xmp(true)
        .with_stop(true)
        .with_lossy(true)
        .with_lossless(true)
        .with_animation(true)
        .with_push_rows(true)
        .with_native_alpha(true)
        .with_effort_range(0, 10)
        .with_quality_range(0.0, 100.0)
        .with_enforces_max_pixels(true)
        .with_enforces_max_memory(true);

/// Map generic quality (libjpeg-turbo scale) to WebP native quality.
///
/// Calibrated on CID22-512 corpus (209 images) to produce the same median
/// SSIMULACRA2 as libjpeg-turbo at each quality level.
fn calibrated_webp_quality(generic_q: f32) -> f32 {
    const TABLE: &[(f32, f32)] = &[
        (5.0, 5.0),
        (10.0, 5.0),
        (15.0, 5.0),
        (20.0, 10.4),
        (25.0, 18.0),
        (30.0, 25.4),
        (35.0, 32.3),
        (40.0, 37.8),
        (45.0, 43.4),
        (50.0, 49.2),
        (55.0, 54.3),
        (60.0, 59.5),
        (65.0, 65.8),
        (70.0, 73.4),
        (72.0, 76.0),
        (75.0, 78.1),
        (78.0, 80.3),
        (80.0, 81.8),
        (82.0, 83.4),
        (85.0, 85.9),
        (87.0, 87.5),
        (90.0, 90.5),
        (92.0, 92.2),
        (95.0, 97.4),
        (97.0, 99.0),
        (99.0, 99.0),
    ];
    interp_quality(TABLE, generic_q)
}

/// Piecewise linear interpolation with clamping at table bounds.
fn interp_quality(table: &[(f32, f32)], x: f32) -> f32 {
    if x <= table[0].0 {
        return table[0].1;
    }
    if x >= table[table.len() - 1].0 {
        return table[table.len() - 1].1;
    }
    for i in 1..table.len() {
        if x <= table[i].0 {
            let (x0, y0) = table[i - 1];
            let (x1, y1) = table[i];
            let t = (x - x0) / (x1 - x0);
            return y0 + t * (y1 - y0);
        }
    }
    table[table.len() - 1].1
}

impl zencodec::encode::EncoderConfig for WebpEncoderConfig {
    type Error = At<EncodeError>;
    type Job = WebpEncodeJob;

    fn format() -> ImageFormat {
        ImageFormat::WebP
    }

    fn supported_descriptors() -> &'static [PixelDescriptor] {
        ENCODE_DESCRIPTORS
    }

    fn capabilities() -> &'static zencodec::encode::EncodeCapabilities {
        &ENCODE_CAPABILITIES
    }

    fn with_generic_effort(mut self, effort: i32) -> Self {
        let clamped = effort.clamp(0, 10);
        self.trait_effort = Some(clamped);
        let method = ((clamped as u64 * 6) / 10).min(6) as u8;
        self.inner = self.inner.with_method(method);
        self
    }

    fn generic_effort(&self) -> Option<i32> {
        self.trait_effort
    }

    fn with_generic_quality(mut self, quality: f32) -> Self {
        let clamped = quality.clamp(0.0, 100.0);
        self.trait_quality = Some(clamped);
        let native = calibrated_webp_quality(clamped);
        self.inner = self.inner.with_quality(native);
        self
    }

    fn generic_quality(&self) -> Option<f32> {
        self.trait_quality
    }

    fn with_lossless(mut self, lossless: bool) -> Self {
        self.inner = self.inner.with_lossless(lossless);
        self
    }

    fn is_lossless(&self) -> Option<bool> {
        Some(matches!(self.inner, EncoderConfig::Lossless(_)))
    }

    fn with_alpha_quality(self, quality: f32) -> Self {
        self.with_alpha_quality_value(quality)
    }

    fn alpha_quality(&self) -> Option<f32> {
        let aq = match &self.inner {
            EncoderConfig::Lossy(cfg) => cfg.alpha_quality,
            EncoderConfig::Lossless(cfg) => cfg.alpha_quality,
        };
        Some(aq as f32)
    }

    fn job(self) -> WebpEncodeJob {
        WebpEncodeJob {
            config: self,
            stop: None,
            icc: None,
            exif: None,
            xmp: None,
            limits: ResourceLimits::none(),
            canvas_size: None,
            loop_count: None,
            policy: None,
        }
    }
}

// ── Encode Job ──────────────────────────────────────────────────────────────

/// Per-operation WebP encode job.
pub struct WebpEncodeJob {
    config: WebpEncoderConfig,
    stop: Option<zencodec::StopToken>,
    icc: Option<Arc<[u8]>>,
    exif: Option<Arc<[u8]>>,
    xmp: Option<Arc<[u8]>>,
    limits: ResourceLimits,
    canvas_size: Option<(u32, u32)>,
    loop_count: Option<Option<u32>>,
    policy: Option<zencodec::encode::EncodePolicy>,
}

impl WebpEncodeJob {
    fn build_inner_config(&self) -> EncoderConfig {
        let mut inner = self.config.inner.clone();
        let mut limits = crate::Limits::none();
        if let Some(px) = self.limits.max_pixels {
            limits = limits.max_total_pixels(px);
        }
        if let Some(mem) = self.limits.max_memory_bytes {
            limits = limits.max_memory(mem);
        }
        if self.limits.max_width.is_some() || self.limits.max_height.is_some() {
            limits = limits.max_dimensions(
                self.limits.max_width.unwrap_or(u32::MAX),
                self.limits.max_height.unwrap_or(u32::MAX),
            );
        }
        inner = inner.limits(limits);
        inner
    }

    fn _build_metadata(&self) -> crate::ImageMetadata<'_> {
        let mut meta = crate::ImageMetadata::new();
        if let Some(ref icc) = self.icc {
            meta = meta.with_icc_profile(icc.as_ref());
        }
        if let Some(ref exif) = self.exif {
            meta = meta.with_exif(exif.as_ref());
        }
        if let Some(ref xmp) = self.xmp {
            meta = meta.with_xmp(xmp.as_ref());
        }
        meta
    }

    fn _has_metadata(&self) -> bool {
        self.icc.is_some() || self.exif.is_some() || self.xmp.is_some()
    }
}

impl zencodec::encode::EncodeJob for WebpEncodeJob {
    type Error = At<EncodeError>;
    type Enc = WebpEncoder;
    type AnimationFrameEnc = WebpAnimationFrameEncoder;

    fn with_stop(mut self, stop: zencodec::StopToken) -> Self {
        self.stop = Some(stop);
        self
    }

    fn with_policy(mut self, policy: zencodec::encode::EncodePolicy) -> Self {
        self.policy = Some(policy);
        self
    }

    fn with_metadata(mut self, meta: Metadata) -> Self {
        self.icc = meta.icc_profile;
        self.exif = meta.exif;
        self.xmp = meta.xmp;
        self
    }

    fn with_limits(mut self, limits: ResourceLimits) -> Self {
        self.limits = limits;
        self
    }

    fn with_canvas_size(mut self, width: u32, height: u32) -> Self {
        self.canvas_size = Some((width, height));
        self
    }

    fn with_loop_count(mut self, count: Option<u32>) -> Self {
        self.loop_count = Some(count);
        self
    }

    fn encoder(self) -> Result<WebpEncoder, At<EncodeError>> {
        let inner_config = self.build_inner_config();
        let policy = self.policy.unwrap_or_default();
        Ok(WebpEncoder {
            inner_config,
            stop: self.stop,
            icc: if policy.resolve_icc(true) {
                self.icc
            } else {
                None
            },
            exif: if policy.resolve_exif(true) {
                self.exif
            } else {
                None
            },
            xmp: if policy.resolve_xmp(true) {
                self.xmp
            } else {
                None
            },
            limits: self.limits,
            canvas_size: self.canvas_size,
            stream: None,
        })
    }

    fn animation_frame_encoder(self) -> Result<WebpAnimationFrameEncoder, At<EncodeError>> {
        let inner_config = self.build_inner_config();
        let loop_count = match self.loop_count {
            Some(Some(0)) | None => crate::decoder::LoopCount::Forever,
            Some(None) => crate::decoder::LoopCount::Forever,
            Some(Some(n)) => {
                let n16 = (n.min(u16::MAX as u32)) as u16;
                crate::decoder::LoopCount::Times(
                    core::num::NonZeroU16::new(n16)
                        .unwrap_or(core::num::NonZeroU16::new(1).unwrap()),
                )
            }
        };
        Ok(WebpAnimationFrameEncoder {
            inner_config,
            anim_enc: None,
            cumulative_ms: 0,
            last_frame_duration_ms: 100,
            canvas_size: self.canvas_size,
            loop_count,
            limits: self.limits,
        })
    }
}

// ── Encoder ─────────────────────────────────────────────────────────────────

/// Streaming accumulation state for row-level encoding.
///
/// Initialized on the first `push_rows` call based on encoder config and input format.
enum StreamAccum {
    /// Lossy opaque RGB8: convert to YUV420 during push for lower peak memory
    /// (1.5 bytes/pixel vs 3 bytes/pixel for RGB8 input).
    Yuv {
        y_plane: Vec<u8>,
        u_plane: Vec<u8>,
        v_plane: Vec<u8>,
        /// Held-back RGB row for chroma pairing when odd rows received.
        pending_row: Vec<u8>,
        width: u32,
        total_rows: u32,
    },
    /// Generic path: accumulate converted pixel bytes.
    Raw {
        pixels: Vec<u8>,
        layout: PixelLayout,
        width: u32,
        total_rows: u32,
    },
}

/// Single-image WebP encoder.
pub struct WebpEncoder {
    inner_config: EncoderConfig,
    stop: Option<zencodec::StopToken>,
    icc: Option<Arc<[u8]>>,
    exif: Option<Arc<[u8]>>,
    xmp: Option<Arc<[u8]>>,
    limits: ResourceLimits,
    canvas_size: Option<(u32, u32)>,
    stream: Option<StreamAccum>,
}

impl WebpEncoder {
    fn do_encode(
        self,
        pixels: &[u8],
        layout: PixelLayout,
        w: u32,
        h: u32,
        stride_pixels: usize,
    ) -> Result<EncodeOutput, EncodeError> {
        self.limits
            .check_dimensions(w, h)
            .map_err(|e| EncodeError::LimitExceeded(alloc::format!("{e}")))?;
        let bpp = layout.bytes_per_pixel() as u64;
        let estimated_mem = w as u64 * h as u64 * bpp;
        self.limits
            .check_memory(estimated_mem)
            .map_err(|e| EncodeError::LimitExceeded(alloc::format!("{e}")))?;

        let mut req =
            EncodeRequest::new(&self.inner_config, pixels, layout, w, h).with_stride(stride_pixels);
        if let Some(ref stop) = self.stop {
            req = req.with_stop(stop);
        }
        {
            let mut meta = crate::ImageMetadata::new();
            if let Some(ref icc) = self.icc {
                meta = meta.with_icc_profile(icc.as_ref());
            }
            if let Some(ref exif) = self.exif {
                meta = meta.with_exif(exif.as_ref());
            }
            if let Some(ref xmp) = self.xmp {
                meta = meta.with_xmp(xmp.as_ref());
            }
            req = req.with_metadata(meta);
        }
        let data = req.encode().map_err(|e| e.decompose().0)?;
        self.limits
            .check_output_size(data.len() as u64)
            .map_err(|e| EncodeError::LimitExceeded(alloc::format!("{e}")))?;
        Ok(EncodeOutput::new(data, ImageFormat::WebP))
    }
}

/// Convert a type-erased PixelSlice to raw bytes + PixelLayout for the native WebP API.
///
/// Returns `(bytes, layout, width, height, stride_pixels)`.
/// For passthrough formats (RGB8, RGBA8), bytes may be borrowed zero-copy.
/// For converted formats, bytes are always owned and contiguous (stride = width).
/// Cast `&[u8]` to `&[f32]`, copying to an aligned buffer only if needed.
fn as_f32_slice(data: &[u8]) -> Cow<'_, [f32]> {
    match bytemuck::try_cast_slice(data) {
        Ok(s) => Cow::Borrowed(s),
        Err(bytemuck::PodCastError::TargetAlignmentGreaterAndInputNotAligned) => Cow::Owned(
            data.chunks_exact(4)
                .map(|c| f32::from_ne_bytes([c[0], c[1], c[2], c[3]]))
                .collect(),
        ),
        Err(e) => panic!("cannot cast &[u8] to &[f32]: {e:?}"),
    }
}

#[allow(clippy::type_complexity)]
fn pixels_to_webp_input<'a>(
    pixels: &'a PixelSlice<'a>,
) -> Result<(alloc::borrow::Cow<'a, [u8]>, PixelLayout, u32, u32, usize), At<EncodeError>> {
    use alloc::borrow::Cow;

    let desc = pixels.descriptor();
    let w = pixels.width();
    let h = pixels.rows();

    let stride_pixels = pixels.stride() / desc.bytes_per_pixel();

    if desc == PixelDescriptor::RGB8_SRGB {
        Ok((
            Cow::Borrowed(pixels.as_strided_bytes()),
            PixelLayout::Rgb8,
            w,
            h,
            stride_pixels,
        ))
    } else if desc == PixelDescriptor::RGBA8_SRGB {
        Ok((
            Cow::Borrowed(pixels.as_strided_bytes()),
            PixelLayout::Rgba8,
            w,
            h,
            stride_pixels,
        ))
    } else if desc == PixelDescriptor::BGRA8_SRGB {
        let raw = pixels.contiguous_bytes();
        let mut rgba = alloc::vec![0u8; raw.len()];
        garb::bytes::bgra_to_rgba(&raw, &mut rgba)
            .map_err(|e| EncodeError::InvalidBufferSize(alloc::format!("pixel conversion: {e}")))?;
        Ok((Cow::Owned(rgba), PixelLayout::Rgba8, w, h, w as usize))
    } else if desc == PixelDescriptor::GRAY8_SRGB {
        let raw = pixels.contiguous_bytes();
        let rgb: Vec<u8> = raw.iter().flat_map(|&g| [g, g, g]).collect();
        Ok((Cow::Owned(rgb), PixelLayout::Rgb8, w, h, w as usize))
    } else if desc == PixelDescriptor::RGBF32_LINEAR {
        let raw = pixels.contiguous_bytes();
        let floats = as_f32_slice(&raw);
        let mut rgb = alloc::vec![0u8; floats.len()];
        linear_srgb::default::linear_to_srgb_u8_slice(&floats, &mut rgb);
        Ok((Cow::Owned(rgb), PixelLayout::Rgb8, w, h, w as usize))
    } else if desc == PixelDescriptor::RGBAF32_LINEAR {
        let raw = pixels.contiguous_bytes();
        let floats = as_f32_slice(&raw);
        let mut rgba = alloc::vec![0u8; floats.len()];
        linear_srgb::default::linear_to_srgb_u8_rgba_slice(&floats, &mut rgba);
        Ok((Cow::Owned(rgba), PixelLayout::Rgba8, w, h, w as usize))
    } else if desc == PixelDescriptor::GRAYF32_LINEAR {
        let raw = pixels.contiguous_bytes();
        let floats = as_f32_slice(&raw);
        let mut gray_u8 = alloc::vec![0u8; floats.len()];
        linear_srgb::default::linear_to_srgb_u8_slice(&floats, &mut gray_u8);
        // gray→rgb: no garb::gray_to_rgb without experimental feature, expand manually
        let mut rgb = alloc::vec![0u8; floats.len() * 3];
        for (i, &g) in gray_u8.iter().enumerate() {
            rgb[i * 3] = g;
            rgb[i * 3 + 1] = g;
            rgb[i * 3 + 2] = g;
        }
        Ok((Cow::Owned(rgb), PixelLayout::Rgb8, w, h, w as usize))
    } else {
        Err(at!(EncodeError::InvalidBufferSize(alloc::format!(
            "unsupported pixel format for WebP encode: {:?}",
            desc
        ))))
    }
}

/// Convert a pair of RGB8 rows to Y/U/V planes.
///
/// Appends Y samples for both rows and one row of U/V chroma samples
/// (2:1 subsampling in both directions).
fn convert_row_pair_to_yuv(
    row0: &[u8],
    row1: &[u8],
    width: usize,
    y_plane: &mut Vec<u8>,
    u_plane: &mut Vec<u8>,
    v_plane: &mut Vec<u8>,
) {
    use crate::decoder::yuv::{rgb_to_u_avg, rgb_to_v_avg, rgb_to_y};

    // Y for both rows
    for x in 0..width {
        y_plane.push(rgb_to_y(&row0[x * 3..x * 3 + 3]));
    }
    for x in 0..width {
        y_plane.push(rgb_to_y(&row1[x * 3..x * 3 + 3]));
    }

    // U/V: average of 2×2 blocks
    let uv_w = width.div_ceil(2);
    for cx in 0..uv_w {
        let x0 = cx * 2;
        let x1 = (cx * 2 + 1).min(width - 1);
        let p00 = &row0[x0 * 3..x0 * 3 + 3];
        let p01 = &row0[x1 * 3..x1 * 3 + 3];
        let p10 = &row1[x0 * 3..x0 * 3 + 3];
        let p11 = &row1[x1 * 3..x1 * 3 + 3];
        u_plane.push(rgb_to_u_avg(p00, p01, p10, p11));
        v_plane.push(rgb_to_v_avg(p00, p01, p10, p11));
    }
}

/// Convert a single RGB8 row to Y plane + U/V chroma (edge replication).
///
/// Used for the last row when image height is odd.
fn convert_single_row_to_yuv(
    row: &[u8],
    width: usize,
    y_plane: &mut Vec<u8>,
    u_plane: &mut Vec<u8>,
    v_plane: &mut Vec<u8>,
) {
    use crate::decoder::yuv::{rgb_to_u_avg, rgb_to_v_avg, rgb_to_y};

    for x in 0..width {
        y_plane.push(rgb_to_y(&row[x * 3..x * 3 + 3]));
    }
    let uv_w = width.div_ceil(2);
    for cx in 0..uv_w {
        let x0 = cx * 2;
        let x1 = (cx * 2 + 1).min(width - 1);
        let p0 = &row[x0 * 3..x0 * 3 + 3];
        let p1 = &row[x1 * 3..x1 * 3 + 3];
        u_plane.push(rgb_to_u_avg(p0, p1, p0, p1));
        v_plane.push(rgb_to_v_avg(p0, p1, p0, p1));
    }
}

impl zencodec::encode::Encoder for WebpEncoder {
    type Error = At<EncodeError>;

    fn reject(op: UnsupportedOperation) -> At<EncodeError> {
        At::from(EncodeError::from(op))
    }

    fn preferred_strip_height(&self) -> u32 {
        match &self.inner_config {
            // Row pair for chroma subsampling (YUV420 processes 2 rows at a time)
            EncoderConfig::Lossy(_) => 2,
            // Lossless has no row-level preference
            EncoderConfig::Lossless(_) => 1,
        }
    }

    fn encode(self, pixels: PixelSlice<'_>) -> Result<EncodeOutput, At<EncodeError>> {
        let (buf, layout, w, h, stride) = pixels_to_webp_input(&pixels)?;
        self.do_encode(&buf, layout, w, h, stride)
            .map_err(|e| at!(e))
    }

    fn push_rows(&mut self, rows: PixelSlice<'_>) -> Result<(), At<EncodeError>> {
        let desc = rows.descriptor();
        let strip_w = rows.width();
        let strip_h = rows.rows();

        if strip_h == 0 {
            return Ok(());
        }

        // Initialize streaming state on first push
        if self.stream.is_none() {
            let is_lossy = matches!(self.inner_config, EncoderConfig::Lossy(_));
            let is_rgb8 = desc == PixelDescriptor::RGB8_SRGB;
            let sharp_yuv = match &self.inner_config {
                EncoderConfig::Lossy(cfg) => cfg.sharp_yuv,
                EncoderConfig::Lossless(_) => false,
            };

            if is_lossy && is_rgb8 && !sharp_yuv {
                // YUV conversion path: lower peak memory
                let (cw, ch) = self.canvas_size.unwrap_or((strip_w, 0));
                let y_cap = cw as usize * ch as usize;
                let uv_w = (cw as usize).div_ceil(2);
                let uv_h = (ch as usize).div_ceil(2);
                self.stream = Some(StreamAccum::Yuv {
                    y_plane: Vec::with_capacity(y_cap),
                    u_plane: Vec::with_capacity(uv_w * uv_h),
                    v_plane: Vec::with_capacity(uv_w * uv_h),
                    pending_row: Vec::new(),
                    width: cw,
                    total_rows: 0,
                });
            } else {
                // Generic path: convert per-strip, accumulate bytes
                let (_, layout, _, _, _) = pixels_to_webp_input(&rows)?;
                let (cw, ch) = self.canvas_size.unwrap_or((strip_w, 0));
                let bpp = layout.bytes_per_pixel();
                let cap = cw as usize * ch as usize * bpp;
                self.stream = Some(StreamAccum::Raw {
                    pixels: Vec::with_capacity(cap),
                    layout,
                    width: cw,
                    total_rows: 0,
                });
                // First strip already converted; store it
                let stream = self.stream.as_mut().unwrap();
                if let StreamAccum::Raw {
                    pixels, total_rows, ..
                } = stream
                {
                    let (buf, _, _, _, _) = pixels_to_webp_input(&rows)?;
                    pixels.extend_from_slice(&buf);
                    *total_rows += strip_h;
                }
                return Ok(());
            }
        }

        match self.stream.as_mut().unwrap() {
            StreamAccum::Yuv {
                y_plane,
                u_plane,
                v_plane,
                pending_row,
                width,
                total_rows,
            } => {
                let w = *width as usize;
                let data = rows.contiguous_bytes();
                let row_bytes = w * 3;
                let input_row_count = data.len() / row_bytes;
                let mut i = 0;

                // Pair pending row from previous push with first new row
                if !pending_row.is_empty() && input_row_count > 0 {
                    let mut pr = core::mem::take(pending_row);
                    let row1 = &data[0..row_bytes];
                    convert_row_pair_to_yuv(&pr, row1, w, y_plane, u_plane, v_plane);
                    pr.clear();
                    *pending_row = pr; // return allocation
                    i = 1;
                }

                // Process complete row pairs
                while i + 1 < input_row_count {
                    let r0 = &data[i * row_bytes..(i + 1) * row_bytes];
                    let r1 = &data[(i + 1) * row_bytes..(i + 2) * row_bytes];
                    convert_row_pair_to_yuv(r0, r1, w, y_plane, u_plane, v_plane);
                    i += 2;
                }

                // Leftover row → pending
                if i < input_row_count {
                    pending_row.clear();
                    pending_row.extend_from_slice(&data[i * row_bytes..(i + 1) * row_bytes]);
                }

                *total_rows += input_row_count as u32;
            }
            StreamAccum::Raw {
                pixels, total_rows, ..
            } => {
                let (buf, _, _, _, _) = pixels_to_webp_input(&rows)?;
                pixels.extend_from_slice(&buf);
                *total_rows += strip_h;
            }
        }

        Ok(())
    }

    fn finish(mut self) -> Result<EncodeOutput, At<EncodeError>> {
        let stream = self
            .stream
            .take()
            .ok_or_else(|| at!(EncodeError::InvalidBufferSize("no rows pushed".into())))?;

        match stream {
            StreamAccum::Yuv {
                mut y_plane,
                mut u_plane,
                mut v_plane,
                pending_row,
                width,
                total_rows,
            } => {
                // Process pending row (odd image height)
                if !pending_row.is_empty() {
                    convert_single_row_to_yuv(
                        &pending_row,
                        width as usize,
                        &mut y_plane,
                        &mut u_plane,
                        &mut v_plane,
                    );
                }

                // Pack Y+U+V into a single buffer for the Yuv420 encode path
                let mut yuv_buf = Vec::with_capacity(y_plane.len() + u_plane.len() + v_plane.len());
                yuv_buf.append(&mut y_plane);
                yuv_buf.append(&mut u_plane);
                yuv_buf.append(&mut v_plane);

                self.do_encode(
                    &yuv_buf,
                    PixelLayout::Yuv420,
                    width,
                    total_rows,
                    width as usize,
                )
                .map_err(|e| at!(e))
            }
            StreamAccum::Raw {
                pixels,
                layout,
                width,
                total_rows,
            } => self
                .do_encode(&pixels, layout, width, total_rows, width as usize)
                .map_err(|e| at!(e)),
        }
    }
}

// ── Full Frame Encoder ──────────────────────────────────────────────────────

/// Animation WebP full-frame encoder.
///
/// The animation encoder is created lazily on the first frame push
/// (to determine canvas dimensions).
pub struct WebpAnimationFrameEncoder {
    inner_config: EncoderConfig,
    anim_enc: Option<AnimationEncoder>,
    cumulative_ms: u32,
    last_frame_duration_ms: u32,
    canvas_size: Option<(u32, u32)>,
    loop_count: crate::decoder::LoopCount,
    limits: ResourceLimits,
}

/// Convert a [`MuxError`] to an [`EncodeError`].
fn mux_to_encode_err(e: MuxError) -> EncodeError {
    use alloc::string::ToString;
    match e {
        MuxError::EncodeError(e) => e,
        MuxError::InvalidDimensions { .. } => EncodeError::InvalidDimensions,
        other => EncodeError::InvalidBufferSize(other.to_string()),
    }
}

impl WebpAnimationFrameEncoder {
    fn ensure_encoder(&mut self, frame_w: u32, frame_h: u32) -> Result<(), At<EncodeError>> {
        if self.anim_enc.is_none() {
            let (cw, ch) = self.canvas_size.unwrap_or((frame_w, frame_h));
            let config = AnimationConfig {
                loop_count: self.loop_count,
                ..AnimationConfig::default()
            };
            let enc = AnimationEncoder::new(cw, ch, config)
                .map_err(|e| at!(mux_to_encode_err(e.decompose().0)))?;
            self.anim_enc = Some(enc);
        }
        Ok(())
    }
}

impl zencodec::encode::AnimationFrameEncoder for WebpAnimationFrameEncoder {
    type Error = At<EncodeError>;

    fn reject(op: UnsupportedOperation) -> At<EncodeError> {
        At::from(EncodeError::from(op))
    }

    fn push_frame(
        &mut self,
        pixels: PixelSlice<'_>,
        duration_ms: u32,
        stop: Option<&dyn enough::Stop>,
    ) -> Result<(), At<EncodeError>> {
        if let Some(s) = stop {
            s.check().map_err(|e| at!(EncodeError::from(e)))?;
        }
        let (buf, layout, w, h, _stride) = pixels_to_webp_input(&pixels)?;
        self.ensure_encoder(w, h)?;
        let timestamp_ms = self.cumulative_ms;
        let enc = self.anim_enc.as_mut().unwrap();
        enc.add_frame(&buf, layout, timestamp_ms, &self.inner_config)
            .map_err(|e| at!(mux_to_encode_err(e.decompose().0)))?;
        self.cumulative_ms = self.cumulative_ms.saturating_add(duration_ms);
        self.last_frame_duration_ms = duration_ms;
        Ok(())
    }

    fn finish(self, stop: Option<&dyn enough::Stop>) -> Result<EncodeOutput, At<EncodeError>> {
        if let Some(s) = stop {
            s.check().map_err(|e| at!(EncodeError::from(e)))?;
        }
        let enc = self
            .anim_enc
            .ok_or_else(|| EncodeError::InvalidBufferSize("no frames added".into()))
            .map_err(|e| at!(e))?;
        let data = enc
            .finalize(self.last_frame_duration_ms)
            .map_err(|e| at!(mux_to_encode_err(e.decompose().0)))?;
        self.limits
            .check_output_size(data.len() as u64)
            .map_err(|e| at!(EncodeError::LimitExceeded(alloc::format!("{e}"))))?;
        Ok(EncodeOutput::new(data, ImageFormat::WebP))
    }
}

// ── Decoding ────────────────────────────────────────────────────────────────

/// WebP decoder configuration implementing [`zencodec::decode::DecoderConfig`].
///
/// Wraps [`DecodeConfig`] for the trait interface.
#[derive(Clone, Debug)]
pub struct WebpDecoderConfig {
    inner: DecodeConfig,
}

impl WebpDecoderConfig {
    /// Create a new decoder config with defaults.
    #[must_use]
    pub fn new() -> Self {
        Self {
            inner: DecodeConfig::default(),
        }
    }

    /// Set the chroma upsampling method.
    #[must_use]
    pub fn with_upsampling(mut self, method: crate::decoder::UpsamplingMethod) -> Self {
        self.inner = self.inner.upsampling(method);
        self
    }

    /// Set chroma dithering strength (0=off, 100=max). Default: 50.
    #[must_use]
    pub fn with_dithering_strength(mut self, strength: u8) -> Self {
        self.inner = self.inner.with_dithering_strength(strength);
        self
    }

    /// Set resource limits on the inner decode config.
    #[must_use]
    pub fn with_limits(mut self, limits: ResourceLimits) -> Self {
        if let Some(px) = limits.max_pixels {
            self.inner.limits = self.inner.limits.max_total_pixels(px);
        }
        if let Some(mem) = limits.max_memory_bytes {
            self.inner.limits = self.inner.limits.max_memory(mem);
        }
        if limits.max_width.is_some() || limits.max_height.is_some() {
            self.inner.limits = self.inner.limits.max_dimensions(
                limits.max_width.unwrap_or(u32::MAX),
                limits.max_height.unwrap_or(u32::MAX),
            );
        }
        self
    }

    /// Access the underlying [`DecodeConfig`].
    #[must_use]
    pub fn inner(&self) -> &DecodeConfig {
        &self.inner
    }

    /// Mutably access the underlying [`DecodeConfig`].
    pub fn inner_mut(&mut self) -> &mut DecodeConfig {
        &mut self.inner
    }

    /// Convenience: probe image header.
    pub fn probe_header(&self, data: &[u8]) -> Result<ImageInfo, At<DecodeError>> {
        use zencodec::decode::{DecodeJob, DecoderConfig};
        <Self as DecoderConfig>::job(self.clone()).probe(data)
    }

    /// Convenience: decode image with this config.
    pub fn decode(&self, data: &[u8]) -> Result<DecodeOutput, At<DecodeError>> {
        use zencodec::decode::{Decode, DecodeJob, DecoderConfig};
        <Self as DecoderConfig>::job(self.clone())
            .decoder(Cow::Borrowed(data), &[])
            .at()?
            .decode()
    }
}

impl Default for WebpDecoderConfig {
    fn default() -> Self {
        Self::new()
    }
}

static DECODE_DESCRIPTORS: &[PixelDescriptor] = &[
    PixelDescriptor::RGB8_SRGB,
    PixelDescriptor::RGBA8_SRGB,
    PixelDescriptor::BGRA8_SRGB,
];

static DECODE_CAPABILITIES: zencodec::decode::DecodeCapabilities =
    zencodec::decode::DecodeCapabilities::new()
        .with_icc(true)
        .with_exif(true)
        .with_xmp(true)
        .with_stop(true)
        .with_animation(true)
        .with_cheap_probe(true)
        .with_native_alpha(true)
        .with_streaming(true)
        .with_enforces_max_pixels(true)
        .with_enforces_max_memory(true)
        .with_enforces_max_input_bytes(true);

impl zencodec::decode::DecoderConfig for WebpDecoderConfig {
    type Error = At<DecodeError>;
    type Job<'a> = WebpDecodeJob;

    fn formats() -> &'static [ImageFormat] {
        &[ImageFormat::WebP]
    }

    fn supported_descriptors() -> &'static [PixelDescriptor] {
        DECODE_DESCRIPTORS
    }

    fn capabilities() -> &'static zencodec::decode::DecodeCapabilities {
        &DECODE_CAPABILITIES
    }

    fn job<'a>(self) -> Self::Job<'a> {
        WebpDecodeJob {
            config: self,
            stop: None,
            limits: ResourceLimits::none(),
            start_frame_index: 0,
            policy: None,
            preferred: Vec::new(),
        }
    }
}

// ── Decode Job ──────────────────────────────────────────────────────────────

/// Per-operation WebP decode job.
#[derive(Clone)]
pub struct WebpDecodeJob {
    config: WebpDecoderConfig,
    stop: Option<zencodec::StopToken>,
    limits: ResourceLimits,
    start_frame_index: u32,
    policy: Option<zencodec::decode::DecodePolicy>,
    preferred: Vec<PixelDescriptor>,
}

impl WebpDecodeJob {
    /// Set preferred pixel formats for `output_info()` prediction.
    ///
    /// This is an inherent method (not part of the trait) because
    /// `output_info()` does not receive a `preferred` list.
    /// Call this before `output_info()` to predict the actual decode format.
    #[must_use]
    pub fn with_preferred(mut self, preferred: &[PixelDescriptor]) -> Self {
        self.preferred = preferred.to_vec();
        self
    }

    /// Apply decode policy to strip metadata from an `ImageInfo`.
    fn apply_policy_to_info(&self, mut info: ImageInfo) -> ImageInfo {
        if let Some(ref policy) = self.policy {
            if !policy.resolve_icc(true) {
                info.source_color.icc_profile = None;
            }
            if !policy.resolve_exif(true) {
                info.embedded_metadata.exif = None;
            }
            if !policy.resolve_xmp(true) {
                info.embedded_metadata.xmp = None;
            }
        }
        info
    }

    fn build_config(&self) -> DecodeConfig {
        let mut cfg = self.config.inner.clone();
        if let Some(px) = self.limits.max_pixels {
            cfg.limits = cfg.limits.max_total_pixels(px);
        }
        if let Some(mem) = self.limits.max_memory_bytes {
            cfg.limits = cfg.limits.max_memory(mem);
        }
        if self.limits.max_width.is_some() || self.limits.max_height.is_some() {
            cfg.limits = cfg.limits.max_dimensions(
                self.limits.max_width.unwrap_or(u32::MAX),
                self.limits.max_height.unwrap_or(u32::MAX),
            );
        }
        if let Some(max_frames) = self.limits.max_frames {
            cfg.limits = cfg.limits.max_frame_count(max_frames as u64);
        }
        cfg
    }

    fn effective_input_size_limit(&self) -> Option<u64> {
        self.limits
            .max_input_bytes
            .or(self.config.inner.limits.max_file_size)
    }
}

impl<'a> zencodec::decode::DecodeJob<'a> for WebpDecodeJob {
    type Error = At<DecodeError>;
    type Dec = WebpDecoder<'a>;
    type StreamDec = WebpStreamingDecoder;
    type AnimationFrameDec = WebpAnimationFrameDecoder;

    fn with_stop(mut self, stop: zencodec::StopToken) -> Self {
        self.stop = Some(stop);
        self
    }

    fn with_limits(mut self, limits: ResourceLimits) -> Self {
        self.limits = limits;
        self
    }

    fn with_policy(mut self, policy: zencodec::decode::DecodePolicy) -> Self {
        self.policy = Some(policy);
        self
    }

    fn with_start_frame_index(mut self, index: u32) -> Self {
        self.start_frame_index = index;
        self
    }

    fn probe(&self, data: &[u8]) -> Result<ImageInfo, At<DecodeError>> {
        let native = crate::ImageInfo::from_webp(data)?;
        let mut info = to_image_info(&native, None);
        if let Ok(probe) = crate::detect::probe(data) {
            info = info.with_source_encoding_details(probe);
        }
        Ok(self.apply_policy_to_info(info))
    }

    fn output_info(&self, data: &[u8]) -> Result<OutputInfo, At<DecodeError>> {
        let native = crate::ImageInfo::from_webp(data)?;
        let mut desc = if native.has_alpha {
            PixelDescriptor::RGBA8_SRGB
        } else {
            PixelDescriptor::RGB8_SRGB
        };
        // Apply RGBA→BGRA negotiation if the stored preferred list requests it.
        if !self.preferred.is_empty()
            && self.preferred.contains(&PixelDescriptor::BGRA8_SRGB)
            && desc == PixelDescriptor::RGBA8_SRGB
        {
            desc = PixelDescriptor::BGRA8_SRGB;
        }
        Ok(OutputInfo::full_decode(native.width, native.height, desc))
    }

    fn decoder(
        mut self,
        data: Cow<'a, [u8]>,
        preferred: &[PixelDescriptor],
    ) -> Result<WebpDecoder<'a>, At<DecodeError>> {
        let cfg = self.build_config();
        let stop = self.stop.take();
        Ok(WebpDecoder {
            config: cfg,
            stop,
            input_size_limit: self.effective_input_size_limit(),
            limits: self.limits,
            data,
            preferred: preferred.to_vec(),
            policy: self.policy,
        })
    }

    fn streaming_decoder(
        self,
        data: Cow<'a, [u8]>,
        preferred: &[PixelDescriptor],
    ) -> Result<WebpStreamingDecoder, At<DecodeError>> {
        if let Some(max) = self.effective_input_size_limit()
            && data.len() as u64 > max
        {
            return Err(at!(DecodeError::InvalidParameter(alloc::format!(
                "input size {} exceeds limit {}",
                data.len(),
                max
            ))));
        }

        let cfg = self.build_config();
        let dither_strength = cfg.dithering_strength;

        // Determine bpp from preferred list
        let has_alpha_preferred = preferred.contains(&PixelDescriptor::RGBA8_SRGB)
            || preferred.contains(&PixelDescriptor::BGRA8_SRGB);
        let default_bpp: usize = if has_alpha_preferred { 4 } else { 3 };

        // Parse RIFF/WebP container
        let data_ref: &[u8] = &data;
        if data_ref.len() < 20 {
            return Err(at!(DecodeError::NotEnoughInitData));
        }
        if &data_ref[..4] != b"RIFF" {
            let mut sig = [0u8; 4];
            sig.copy_from_slice(&data_ref[..4]);
            return Err(at!(DecodeError::RiffSignatureInvalid(sig)));
        }
        if &data_ref[8..12] != b"WEBP" {
            let mut sig = [0u8; 4];
            sig.copy_from_slice(&data_ref[8..12]);
            return Err(at!(DecodeError::WebpSignatureInvalid(sig)));
        }

        let first_chunk = &data_ref[12..16];

        match first_chunk {
            b"VP8 " => {
                let chunk_size =
                    u32::from_le_bytes([data_ref[16], data_ref[17], data_ref[18], data_ref[19]])
                        as usize;
                let vp8_start = 20;
                let vp8_end = (vp8_start + chunk_size).min(data_ref.len());
                let vp8_data = &data_ref[vp8_start..vp8_end];

                // Probe info for the trait
                let native_info = crate::ImageInfo::from_webp(data_ref).ok();
                let info = if let Some(ref ni) = native_info {
                    to_image_info(ni, None)
                } else {
                    // Fallback: will be populated after header parse
                    ImageInfo::new(0, 0, ImageFormat::WebP)
                };

                WebpStreamingDecoder::new(
                    vp8_data,
                    None,
                    default_bpp,
                    preferred,
                    info,
                    dither_strength,
                )
            }
            b"VP8X" => {
                use crate::mux::WebPDemuxer;
                let demuxer = WebPDemuxer::new(data_ref).map_err(|e| {
                    at!(DecodeError::InvalidParameter(alloc::format!(
                        "demux error: {e}"
                    )))
                })?;

                if demuxer.is_animated() {
                    return Err(at!(DecodeError::UnsupportedFeature(
                        "streaming decode does not support animation".into()
                    )));
                }

                let frame = demuxer
                    .frame(1)
                    .ok_or_else(|| at!(DecodeError::ChunkMissing))?;

                if !frame.is_lossy {
                    return Err(at!(DecodeError::UnsupportedFeature(
                        "streaming decode only supports lossy VP8, got VP8L".into()
                    )));
                }

                // Decode alpha plane up front (it's small: width*height bytes)
                let alpha_plane = if let Some(alpha_data) = frame.alpha_data {
                    let native_info = crate::ImageInfo::from_webp(data_ref)?;
                    let w = native_info.width as u16;
                    let h = native_info.height as u16;
                    let alpha_chunk = crate::decoder::extended::read_alpha_chunk(alpha_data, w, h)?;

                    // Apply alpha filtering to produce final alpha plane
                    let fw = usize::from(w);
                    let fh = usize::from(h);
                    let mut alpha_out = alloc::vec![0u8; fw * fh];
                    for y in 0..fh {
                        for x in 0..fw {
                            let predictor =
                                crate::decoder::extended::get_alpha_predictor_from_alpha(
                                    x,
                                    y,
                                    fw,
                                    alpha_chunk.filtering_method,
                                    &alpha_out,
                                );
                            let idx = y * fw + x;
                            alpha_out[idx] = predictor.wrapping_add(alpha_chunk.data[idx]);
                        }
                    }
                    Some(alpha_out)
                } else {
                    None
                };

                let native_info = crate::ImageInfo::from_webp(data_ref).ok();
                let info = if let Some(ref ni) = native_info {
                    to_image_info(ni, None)
                } else {
                    ImageInfo::new(0, 0, ImageFormat::WebP)
                };

                let bpp = if alpha_plane.is_some() {
                    4
                } else {
                    default_bpp
                };

                WebpStreamingDecoder::new(
                    frame.bitstream,
                    alpha_plane,
                    bpp,
                    preferred,
                    info,
                    dither_strength,
                )
            }
            b"VP8L" => Err(at!(DecodeError::UnsupportedFeature(
                "streaming decode does not support lossless VP8L".into()
            ))),
            _ => Err(at!(DecodeError::UnsupportedFeature(alloc::format!(
                "streaming decode: unsupported chunk type {:?}",
                first_chunk
            )))),
        }
    }

    fn push_decoder(
        self,
        data: Cow<'a, [u8]>,
        sink: &mut dyn zencodec::decode::DecodeRowSink,
        preferred: &[PixelDescriptor],
    ) -> Result<OutputInfo, Self::Error> {
        push_decoder_impl(self, data, sink, preferred)
    }

    fn animation_frame_decoder(
        self,
        data: Cow<'a, [u8]>,
        preferred: &[PixelDescriptor],
    ) -> Result<WebpAnimationFrameDecoder, At<DecodeError>> {
        // Block animation if policy denies it.
        if let Some(ref policy) = self.policy
            && !policy.resolve_animation(true)
        {
            return Err(At::from(DecodeError::from(
                UnsupportedOperation::AnimationDecode,
            )));
        }

        if let Some(max) = self.effective_input_size_limit()
            && data.len() as u64 > max
        {
            return Err(at!(DecodeError::InvalidParameter(alloc::format!(
                "input size {} exceeds limit {}",
                data.len(),
                max
            ))));
        }

        let cfg = self.build_config();

        // Parse container info before moving data into the self-referential struct.
        let native_info = crate::ImageInfo::from_webp(&data).ok();

        // Probe animation metadata with a temporary decoder.
        let probe_anim = AnimationDecoder::new_with_config(&data, &cfg)?;
        let anim_info = probe_anim.info();
        let total_frames = anim_info.frame_count;
        let anim_loop_count = match anim_info.loop_count {
            crate::decoder::LoopCount::Forever => Some(0),
            crate::decoder::LoopCount::Times(n) => Some(n.get() as u32),
        };
        let base_info = if let Some(ref ni) = native_info {
            to_image_info(ni, Some(anim_loop_count))
        } else {
            ImageInfo::new(
                anim_info.canvas_width,
                anim_info.canvas_height,
                ImageFormat::WebP,
            )
            .with_alpha(anim_info.has_alpha)
            .with_bit_depth(8)
            .with_channel_count(if anim_info.has_alpha { 4 } else { 3 })
            .with_sequence(ImageSequence::Animation {
                frame_count: Some(anim_info.frame_count),
                loop_count: anim_loop_count,
                random_access: false,
            })
        };
        let base_info = self.apply_policy_to_info(base_info);
        let shared_info = Arc::new(base_info);
        drop(probe_anim);

        // Build the self-referential struct: owned data + borrowing AnimationDecoder.
        // The AnimationDecoder borrows &[u8] from the owned Vec, so we use self_cell
        // to safely express this relationship without unsafe code.
        let owned_data = data.into_owned();
        let decoder = OwnedAnimDecoder::try_new(
            AnimDecoderOwner {
                data: owned_data,
                config: cfg,
            },
            |owner| AnimationDecoder::new_with_config(&owner.data, &owner.config),
        )?;

        let has_alpha = anim_info.has_alpha;
        let canvas_width = anim_info.canvas_width;
        let canvas_height = anim_info.canvas_height;
        let bpp: usize = if has_alpha { 4 } else { 3 };
        let buf_size = canvas_width as usize * canvas_height as usize * bpp;
        let source_desc = if has_alpha {
            PixelDescriptor::RGBA8_SRGB
        } else {
            PixelDescriptor::RGB8_SRGB
        };

        Ok(WebpAnimationFrameDecoder {
            decoder,
            preferred: preferred.to_vec(),
            frame_buf: alloc::vec![0u8; buf_size],
            has_alpha,
            canvas_width,
            canvas_height,
            current_duration_ms: 0,
            current_descriptor: source_desc,
            next_frame_index: 0,
            start_frame_index: self.start_frame_index,
            frames_skipped: false,
            info: shared_info,
            total_frames,
            anim_loop_count,
            limits: self.limits,
            accumulated_duration_ms: 0,
        })
    }
}

// ── Decoder ─────────────────────────────────────────────────────────────────

/// Single-image WebP decoder.
pub struct WebpDecoder<'a> {
    config: DecodeConfig,
    stop: Option<zencodec::StopToken>,
    input_size_limit: Option<u64>,
    limits: ResourceLimits,
    data: Cow<'a, [u8]>,
    preferred: Vec<PixelDescriptor>,
    policy: Option<zencodec::decode::DecodePolicy>,
}

impl WebpDecoder<'_> {
    /// Apply decode policy to strip metadata from an `ImageInfo`.
    fn apply_policy_to_info(&self, mut info: ImageInfo) -> ImageInfo {
        if let Some(ref policy) = self.policy {
            if !policy.resolve_icc(true) {
                info.source_color.icc_profile = None;
            }
            if !policy.resolve_exif(true) {
                info.embedded_metadata.exif = None;
            }
            if !policy.resolve_xmp(true) {
                info.embedded_metadata.xmp = None;
            }
        }
        info
    }

    fn check_input_size(&self, data: &[u8]) -> Result<(), At<DecodeError>> {
        if let Some(max) = self.input_size_limit
            && data.len() as u64 > max
        {
            return Err(at!(DecodeError::InvalidParameter(alloc::format!(
                "input size {} exceeds limit {}",
                data.len(),
                max
            ))));
        }
        Ok(())
    }

    fn do_decode(&self, data: &[u8]) -> Result<DecodeOutput, DecodeError> {
        self.check_input_size(data)?;

        if let Ok(info) = crate::ImageInfo::from_webp(data) {
            self.limits
                .check_dimensions(info.width, info.height)
                .map_err(|e| DecodeError::InvalidParameter(alloc::format!("{e}")))?;
        }

        // Try lossy VP8 direct path (faster, streaming cache)
        if let Ok(result) = self.do_decode_lossy(data) {
            return Ok(result);
        }

        // General path (lossless, animation, etc)
        self.do_decode_general(data)
    }

    /// Decode lossy VP8 direct path (with alpha support).
    fn do_decode_lossy(&self, data: &[u8]) -> Result<DecodeOutput, DecodeError> {
        let dither_strength = self.config.dithering_strength;

        // Use DecodeRequest's lossy path which handles container parsing
        let req = DecodeRequest::new(&self.config, data);
        let (pixels, w, h) = req.decode_rgba_lossy().map_err(|e| e.decompose().0)?;

        let w = u32::from(w);
        let h = u32::from(h);
        let _ = dither_strength; // already applied inside decode_rgba_lossy

        let has_alpha = {
            let native_info = crate::ImageInfo::from_webp(data).ok();
            native_info.as_ref().is_some_and(|i| i.has_alpha)
        };

        let buf = if has_alpha {
            PixelBuffer::from_vec(pixels, w, h, PixelDescriptor::RGBA8_SRGB)
                .map_err(|_| DecodeError::InvalidParameter("pixel count mismatch".into()))?
        } else {
            // Strip alpha channel — lossy path always decodes to RGBA when alpha is present,
            // but for no-alpha images decode_rgba_lossy returns RGBA with alpha=255.
            // Convert to RGB to save memory.
            let pixel_count = (w as usize) * (h as usize);
            let mut rgb = alloc::vec![0u8; pixel_count * 3];
            garb::bytes::rgba_to_rgb(&pixels, &mut rgb)
                .map_err(|e| DecodeError::InvalidParameter(alloc::format!("{e}")))?;
            PixelBuffer::from_vec(rgb, w, h, PixelDescriptor::RGB8_SRGB)
                .map_err(|_| DecodeError::InvalidParameter("pixel count mismatch".into()))?
        };

        let native_info = crate::ImageInfo::from_webp(data).ok();
        let info = if let Some(ref ni) = native_info {
            to_image_info(ni, None)
        } else {
            ImageInfo::new(w, h, ImageFormat::WebP)
                .with_alpha(has_alpha)
                .with_bit_depth(8)
                .with_channel_count(if has_alpha { 4 } else { 3 })
        };
        let info = self.apply_policy_to_info(info);

        let mut output = DecodeOutput::new(buf, info);
        if let Ok(probe) = crate::detect::probe(data) {
            output = output.with_source_encoding_details(probe);
        }
        Ok(output)
    }

    /// General decode path (handles lossless, animation frames, etc.).
    fn do_decode_general(&self, data: &[u8]) -> Result<DecodeOutput, DecodeError> {
        let mut req = DecodeRequest::new(&self.config, data);
        if let Some(ref stop) = self.stop {
            req = req.stop(stop);
        }

        let (pixels, w, h, layout) = req.decode().map_err(|e| e.decompose().0)?;

        let buf: PixelBuffer = match layout {
            PixelLayout::Rgb8 => PixelBuffer::from_vec(pixels, w, h, PixelDescriptor::RGB8_SRGB)
                .map_err(|_| DecodeError::InvalidParameter("pixel count mismatch".into()))?,
            PixelLayout::Rgba8 => PixelBuffer::from_vec(pixels, w, h, PixelDescriptor::RGBA8_SRGB)
                .map_err(|_| DecodeError::InvalidParameter("pixel count mismatch".into()))?,
            _ => {
                // Fallback: decode as RGBA
                let rgba_req = DecodeRequest::new(&self.config, data);
                let (rgba_pixels, rw, rh) = if let Some(ref stop) = self.stop {
                    rgba_req
                        .stop(stop)
                        .decode_rgba()
                        .map_err(|e| e.decompose().0)?
                } else {
                    rgba_req.decode_rgba().map_err(|e| e.decompose().0)?
                };
                PixelBuffer::from_vec(rgba_pixels, rw, rh, PixelDescriptor::RGBA8_SRGB)
                    .map_err(|_| DecodeError::InvalidParameter("pixel count mismatch".into()))?
            }
        };

        let has_alpha = buf.has_alpha();
        let native_info = crate::ImageInfo::from_webp(data).ok();
        let info = if let Some(ref ni) = native_info {
            to_image_info(ni, None)
        } else {
            ImageInfo::new(w, h, ImageFormat::WebP)
                .with_alpha(has_alpha)
                .with_bit_depth(8)
                .with_channel_count(if has_alpha { 4 } else { 3 })
        };
        let info = self.apply_policy_to_info(info);

        let mut output = DecodeOutput::new(buf, info);
        if let Ok(probe) = crate::detect::probe(data) {
            output = output.with_source_encoding_details(probe);
        }
        Ok(output)
    }
}

/// Apply preferred format negotiation to decoded output.
fn negotiate_format(pixels: PixelBuffer, preferred: &[PixelDescriptor]) -> PixelBuffer {
    if preferred.is_empty() {
        return pixels;
    }
    let desc = pixels.descriptor();
    // Check if BGRA is preferred and we have RGBA
    if preferred.contains(&PixelDescriptor::BGRA8_SRGB) && desc == PixelDescriptor::RGBA8_SRGB {
        let w = pixels.width();
        let h = pixels.height();
        let mut raw = pixels.into_vec();
        garb::bytes::rgba_to_bgra_inplace(&mut raw)
            .expect("negotiate_format: validated 4bpp buffer");
        return PixelBuffer::from_vec(raw, w, h, PixelDescriptor::BGRA8_SRGB)
            .expect("negotiate_format: dimensions unchanged");
    }
    pixels
}

impl zencodec::decode::Decode for WebpDecoder<'_> {
    type Error = At<DecodeError>;

    fn decode(self) -> Result<DecodeOutput, At<DecodeError>> {
        let output = self.do_decode(&self.data).map_err(|e| at!(e))?;
        if self.preferred.is_empty() {
            return Ok(output);
        }
        let info = output.info().clone();
        let pixels = negotiate_format(output.into_buffer(), &self.preferred);
        Ok(DecodeOutput::new(pixels, info))
    }
}

// ── push_decoder dispatch ─────────────────────────────────────────────────

/// Return `true` iff the input is a lossy VP8 single image eligible for
/// zero-copy streaming decode.
///
/// The check is intentionally quick — just enough to decide between the
/// native streaming path and the full-decode fallback:
///
/// * `VP8 ` chunk after the RIFF/WEBP magic → lossy bitstream, eligible.
/// * `VP8X` chunk → inspect VP8X flags. Animations, VP8L bitstreams, and
///   truly unknown encodings disqualify; plain lossy (optionally with ALPH)
///   qualifies.
/// * Anything else (VP8L as the first chunk, truncated, unknown) is not
///   eligible.
fn is_lossy_streaming_candidate(data: &[u8]) -> bool {
    let Ok(info) = crate::detect::probe(data) else {
        return false;
    };
    !info.has_animation && matches!(info.bitstream, crate::detect::BitstreamType::Lossy { .. })
}

/// Native row-streaming `push_decoder` for lossy VP8.
///
/// Feeds each macroblock-row strip directly to the sink, skipping the
/// full-frame RGB/RGBA allocation that [`zencodec::helpers::copy_decode_to_sink`]
/// requires. For lossless (VP8L), animations, or lossy streams where the
/// internal streaming decoder cannot be constructed, transparently falls
/// back to the full-decode helper so the caller still gets a correct image.
fn push_decoder_impl<'a>(
    job: WebpDecodeJob,
    data: Cow<'a, [u8]>,
    sink: &mut dyn zencodec::decode::DecodeRowSink,
    preferred: &[PixelDescriptor],
) -> Result<OutputInfo, At<DecodeError>> {
    let wrap_sink = |e: SinkError| at!(DecodeError::InvalidParameter(alloc::format!("{e}")));

    // Fast path only applies to lossy single-image VP8. For VP8L, animation,
    // or anything the container scanner can't classify, fall back to the
    // helper-driven full decode — it handles all the same cases `decode()` does.
    if !is_lossy_streaming_candidate(&data) {
        return zencodec::helpers::copy_decode_to_sink(job, data, sink, preferred, |e| {
            at!(DecodeError::InvalidParameter(alloc::format!("{e}")))
        });
    }

    // Dimension limit enforcement (matches `WebpDecoder::do_decode`).
    if let Ok(info) = crate::ImageInfo::from_webp(&data) {
        job.limits
            .check_dimensions(info.width, info.height)
            .map_err(|e| at!(DecodeError::InvalidParameter(alloc::format!("{e}"))))?;
    }

    use zencodec::decode::{DecodeJob, StreamingDecode};

    // Clone stop out of the job so we can still check cancellation after
    // `streaming_decoder` consumes `job` below.
    let stop = job.stop.clone();
    let fallback_job = job.clone();

    // Build the streaming decoder. If this rejects the input (e.g.
    // extra_y_rows < 2 — filter disabled — or any other edge case we
    // haven't pre-filtered), retry with the full-decode helper using a
    // cloned job so `self` is not consumed.
    let mut stream = match job.streaming_decoder(Cow::Borrowed(&data), preferred) {
        Ok(s) => s,
        Err(_) => {
            return zencodec::helpers::copy_decode_to_sink(
                fallback_job,
                data,
                sink,
                preferred,
                |e| at!(DecodeError::InvalidParameter(alloc::format!("{e}"))),
            );
        }
    };

    let width = stream.width;
    let height = stream.height;
    let descriptor = stream.descriptor;
    let bpp = stream.bpp;

    // Tell the sink what's coming. After this, any error path must not
    // call `sink.finish()`.
    sink.begin(width, height, descriptor).map_err(wrap_sink)?;

    let row_bytes = width as usize * bpp;

    while let Some((y_start, pixel_slice)) = stream.next_batch()? {
        // Cooperative cancellation between strips.
        if let Some(ref s) = stop {
            use enough::Stop;
            s.check().map_err(|e| at!(DecodeError::from(e)))?;
        }

        let num_rows: u32 = pixel_slice.rows();
        if num_rows == 0 {
            continue;
        }

        let mut dst = sink
            .provide_next_buffer(y_start, num_rows, width, descriptor)
            .map_err(wrap_sink)?;

        for row in 0..num_rows {
            let src = pixel_slice.row(row);
            let dst_row = dst.row_mut(row);
            // PixelSlice rows are width*bpp; sink buffers may be wider
            // (stride padding). Copy only the pixel bytes.
            dst_row[..row_bytes].copy_from_slice(&src[..row_bytes]);
        }
        drop(dst);
    }

    sink.finish().map_err(wrap_sink)?;

    Ok(OutputInfo::full_decode(width, height, descriptor))
}

// ── Streaming Decoder ─────────────────────────────────────────────────────

/// Streaming WebP decoder yielding RGB/RGBA strips per MB row.
///
/// Each [`next_batch()`](zencodec::decode::StreamingDecode::next_batch) call
/// decodes one macroblock row (up to 16 Y rows) and returns the visible
/// pixel rows as a [`PixelSlice`]. Peak memory is bounded by the input data
/// plus the row cache (~100 KB for 4K) plus the strip buffer (~150 KB for
/// 4K RGBA), instead of the full-frame RGB allocation.
///
/// Only supports lossy VP8 (simple and VP8X extended, with alpha). Lossless
/// VP8L and animation are not supported via streaming decode.
pub struct WebpStreamingDecoder {
    /// VP8 decoder context (owns row cache and decode state).
    ctx: crate::decoder::vp8v2::DecoderContext,
    /// Reusable strip buffer for RGB/RGBA output.
    strip_buf: Vec<u8>,
    /// Current MB row index (0-based, incremented after each next_batch).
    current_mby: usize,
    /// Total MB row count.
    mbheight: usize,
    /// Visible pixel width.
    width: u32,
    /// Visible pixel height.
    height: u32,
    /// Bytes per pixel (3 for RGB, 4 for RGBA).
    bpp: usize,
    /// Pixel descriptor for the output.
    descriptor: PixelDescriptor,
    /// Image info for the trait.
    info: Arc<ImageInfo>,
    /// Decoded alpha plane (if VP8X with alpha). Applied per-strip.
    alpha_plane: Option<Vec<u8>>,
}

impl WebpStreamingDecoder {
    /// Construct a streaming decoder from parsed VP8 data.
    ///
    /// `vp8_data` is the raw VP8 bitstream (after container parsing).
    /// `alpha_data` is the optional alpha chunk (already decompressed into a plane).
    fn new(
        vp8_data: &[u8],
        alpha_plane: Option<Vec<u8>>,
        bpp: usize,
        preferred: &[PixelDescriptor],
        info: ImageInfo,
        dither_strength: u8,
    ) -> Result<Self, At<DecodeError>> {
        let mut ctx =
            crate::decoder::vp8v2::DecoderContext::new().with_dithering_strength(dither_strength);
        ctx.read_frame_header(vp8_data)?;

        let width = ctx.width();
        let height = ctx.height();
        let mbheight = usize::from(ctx.mbheight());
        let extra_y_rows = ctx.extra_y_rows();

        // Check that extra_y_rows >= 2 for streaming (fancy upsampling)
        if extra_y_rows < 2 {
            // Fallback: for no-filter case, streaming is not supported.
            // This is extremely rare (very high quality / filter disabled).
            return Err(at!(DecodeError::UnsupportedFeature(
                "streaming decode requires filter (extra_y_rows >= 2)".into(),
            )));
        }

        ctx.init_streaming_uv_buffers();

        // Determine output format
        let has_alpha = alpha_plane.is_some();
        let effective_bpp = if has_alpha { 4 } else { bpp };
        let mut descriptor = if has_alpha {
            PixelDescriptor::RGBA8_SRGB
        } else if effective_bpp == 4 {
            PixelDescriptor::RGBA8_SRGB
        } else {
            PixelDescriptor::RGB8_SRGB
        };

        // Apply BGRA negotiation
        if !preferred.is_empty()
            && preferred.contains(&PixelDescriptor::BGRA8_SRGB)
            && descriptor == PixelDescriptor::RGBA8_SRGB
        {
            descriptor = PixelDescriptor::BGRA8_SRGB;
        }

        // Max strip rows: last MB row may emit extra_y_rows + 16 rows.
        // For the first row of a single-row image, it's 16.
        let max_strip_rows = if mbheight == 1 { 16 } else { extra_y_rows + 16 };
        let strip_buf_size = usize::from(width) * max_strip_rows * effective_bpp;
        let strip_buf = alloc::vec![0u8; strip_buf_size];

        Ok(Self {
            ctx,
            strip_buf,
            current_mby: 0,
            mbheight,
            width: u32::from(width),
            height: u32::from(height),
            bpp: effective_bpp,
            descriptor,
            info: Arc::new(info),
            alpha_plane,
        })
    }
}

impl zencodec::decode::StreamingDecode for WebpStreamingDecoder {
    type Error = At<DecodeError>;

    fn next_batch(&mut self) -> Result<Option<(u32, PixelSlice<'_>)>, At<DecodeError>> {
        if self.current_mby >= self.mbheight {
            return Ok(None);
        }

        let mby = self.current_mby;
        let (y_start, num_rows) = self
            .ctx
            .decode_strip_mb_row(mby, &mut self.strip_buf, self.bpp)
            .map_err(|e| at!(DecodeError::from(e)))?;

        self.current_mby += 1;

        if num_rows == 0 {
            return Ok(None);
        }

        let width = self.width as usize;
        let row_bytes = width * self.bpp;
        let strip_bytes = num_rows * row_bytes;

        // Apply alpha plane if present
        if let Some(ref alpha) = self.alpha_plane {
            if self.bpp == 4 {
                let fw = width;
                for row in 0..num_rows {
                    let img_y = y_start + row;
                    if img_y >= self.height as usize {
                        break;
                    }
                    for x in 0..fw {
                        let alpha_index = img_y * fw + x;
                        let strip_index = row * row_bytes + x * 4 + 3;
                        if alpha_index < alpha.len() && strip_index < self.strip_buf.len() {
                            self.strip_buf[strip_index] = alpha[alpha_index];
                        }
                    }
                }
            }
        }

        // Apply RGBA→BGRA swizzle if negotiated
        if self.descriptor == PixelDescriptor::BGRA8_SRGB {
            let strip_data = &mut self.strip_buf[..strip_bytes];
            let _ = garb::bytes::rgba_to_bgra_inplace(strip_data);
        }

        let slice = PixelSlice::new(
            &self.strip_buf[..strip_bytes],
            self.width,
            num_rows as u32,
            row_bytes,
            self.descriptor,
        )
        .map_err(|_| at!(DecodeError::InvalidParameter("strip slice mismatch".into())))?;

        Ok(Some((y_start as u32, slice)))
    }

    fn info(&self) -> &ImageInfo {
        &self.info
    }
}

// ── Full Frame Decoder ──────────────────────────────────────────────────────

/// Owned data for the self-referential animation decoder.
struct AnimDecoderOwner {
    data: Vec<u8>,
    config: DecodeConfig,
}

// Safety: self_cell uses only safe Rust internally. AnimationDecoder<'a> is
// covariant in 'a, so the #[covariant] annotation is correct.
self_cell::self_cell! {
    struct OwnedAnimDecoder {
        owner: AnimDecoderOwner,
        #[covariant]
        dependent: AnimationDecoder,
    }
}

/// Animation WebP full-frame decoder.
///
/// Decodes frames lazily — each `render_next_frame` call decodes exactly one
/// frame via the underlying [`AnimationDecoder`]. Zero per-frame allocations
/// on the borrowing path (`render_next_frame`); one allocation on the owned
/// path (`render_next_frame_owned`).
///
/// Memory usage is O(canvas_size) instead of O(N_frames × canvas_size).
///
/// The [`AnimationDecoder`] borrows `&[u8]` from the input data. A self-referential
/// struct ([`OwnedAnimDecoder`] via `self_cell`) stores the owned data alongside
/// the borrowing decoder, satisfying the `'static` requirement on `AnimationFrameDec`.
pub struct WebpAnimationFrameDecoder {
    decoder: OwnedAnimDecoder,
    preferred: Vec<PixelDescriptor>,
    /// Reusable buffer for composited frame data. Pre-allocated to canvas size;
    /// reused across frames without reallocating.
    frame_buf: Vec<u8>,
    /// Whether the animation has alpha (RGBA vs RGB output).
    has_alpha: bool,
    /// Canvas dimensions.
    canvas_width: u32,
    canvas_height: u32,
    /// Duration of the most recently decoded frame.
    current_duration_ms: u32,
    /// Descriptor after format negotiation for the current frame.
    current_descriptor: PixelDescriptor,
    /// Frame index of the next frame to yield.
    next_frame_index: u32,
    /// Frames before this index are decoded (for compositing) but not returned.
    start_frame_index: u32,
    /// Whether we've already skipped past start_frame_index.
    frames_skipped: bool,
    info: Arc<ImageInfo>,
    total_frames: u32,
    anim_loop_count: Option<u32>,
    /// Resource limits for frame count and animation duration enforcement.
    limits: ResourceLimits,
    /// Accumulated duration of all yielded frames in milliseconds.
    accumulated_duration_ms: u64,
}

impl WebpAnimationFrameDecoder {
    /// Decode and discard frames before `start_frame_index` (needed for
    /// correct compositing state). Called lazily on first `render_next_frame`.
    fn skip_to_start(&mut self) -> Result<(), At<DecodeError>> {
        if self.frames_skipped {
            return Ok(());
        }
        self.frames_skipped = true;
        let skip_count = self.start_frame_index as usize;
        for _ in 0..skip_count {
            let done = self
                .decoder
                .with_dependent_mut(|_, anim| match anim.decode_next() {
                    Ok(Some(_)) => Ok(false),
                    Ok(None) => Ok(true),
                    Err(e) => Err(e),
                })
                .map_err(|e| at!(e))?;
            if done {
                break;
            }
        }
        self.next_frame_index = self.start_frame_index;
        Ok(())
    }

    /// Decode next frame into `self.frame_buf` (zero-alloc: reuses buffer).
    /// Returns `true` if a frame was decoded, `false` if no more frames.
    fn decode_next_into_buf(&mut self) -> Result<bool, At<DecodeError>> {
        // Re-allocate if frame_buf was taken by render_next_frame_owned.
        let expected_size = self.stride() * self.canvas_height as usize;
        if self.frame_buf.len() < expected_size {
            self.frame_buf.resize(expected_size, 0);
        }

        let frame_buf = &mut self.frame_buf;
        let result = self
            .decoder
            .with_dependent_mut(|_, anim| {
                match anim.decode_next() {
                    Ok(Some(info)) => {
                        let data = anim.current_frame_data();
                        // frame_buf is pre-allocated to canvas size; this is a memcpy, no alloc.
                        frame_buf[..data.len()].copy_from_slice(data);
                        Ok(Some(info.duration_ms))
                    }
                    Ok(None) => Ok(None),
                    Err(e) => Err(e),
                }
            })
            .map_err(|e| at!(e))?;

        match result {
            Some(duration_ms) => {
                self.current_duration_ms = duration_ms;
                // Apply RGBA→BGRA swizzle in-place if preferred.
                let source = if self.has_alpha {
                    PixelDescriptor::RGBA8_SRGB
                } else {
                    PixelDescriptor::RGB8_SRGB
                };
                self.current_descriptor =
                    negotiate_format_inplace(&mut self.frame_buf, source, &self.preferred);
                Ok(true)
            }
            None => Ok(false),
        }
    }

    /// Byte stride for the current frame format.
    fn stride(&self) -> usize {
        let bpp: usize = if self.has_alpha { 4 } else { 3 };
        self.canvas_width as usize * bpp
    }
}

/// Apply RGBA→BGRA swizzle in-place if the preferred list requests BGRA.
/// Returns the descriptor that matches the buffer contents after the call.
fn negotiate_format_inplace(
    data: &mut [u8],
    source: PixelDescriptor,
    preferred: &[PixelDescriptor],
) -> PixelDescriptor {
    if !preferred.is_empty()
        && preferred.contains(&PixelDescriptor::BGRA8_SRGB)
        && source == PixelDescriptor::RGBA8_SRGB
    {
        garb::bytes::rgba_to_bgra_inplace(data)
            .expect("negotiate_format_inplace: validated 4bpp buffer");
        return PixelDescriptor::BGRA8_SRGB;
    }
    source
}

impl zencodec::decode::AnimationFrameDecoder for WebpAnimationFrameDecoder {
    type Error = At<DecodeError>;

    fn wrap_sink_error(err: SinkError) -> At<DecodeError> {
        at!(DecodeError::InvalidParameter(alloc::format!("{err}")))
    }

    fn info(&self) -> &ImageInfo {
        &self.info
    }

    fn frame_count(&self) -> Option<u32> {
        Some(self.total_frames)
    }

    fn loop_count(&self) -> Option<u32> {
        self.anim_loop_count
    }

    fn render_next_frame(
        &mut self,
        stop: Option<&dyn enough::Stop>,
    ) -> Result<Option<AnimationFrame<'_>>, At<DecodeError>> {
        if let Some(s) = stop {
            s.check().map_err(|e| at!(DecodeError::from(e)))?;
        }
        self.skip_to_start()?;

        if !self.decode_next_into_buf()? {
            return Ok(None);
        }
        let idx = self.next_frame_index;
        self.next_frame_index += 1;

        // Enforce max_frames limit.
        self.limits
            .check_frames(self.next_frame_index)
            .map_err(|e| at!(DecodeError::InvalidParameter(alloc::format!("{e}"))))?;

        // Enforce max_animation_ms limit.
        self.accumulated_duration_ms = self
            .accumulated_duration_ms
            .saturating_add(self.current_duration_ms as u64);
        self.limits
            .check_animation_ms(self.accumulated_duration_ms)
            .map_err(|e| at!(DecodeError::InvalidParameter(alloc::format!("{e}"))))?;

        // Zero-alloc: create PixelSlice directly from the reusable frame_buf.
        let stride = self.stride();
        let slice = PixelSlice::new(
            &self.frame_buf,
            self.canvas_width,
            self.canvas_height,
            stride,
            self.current_descriptor,
        )
        .map_err(|_| {
            at!(DecodeError::InvalidParameter(
                "frame buffer mismatch".into(),
            ))
        })?;
        Ok(Some(AnimationFrame::new(
            slice,
            self.current_duration_ms,
            idx,
        )))
    }

    fn render_next_frame_owned(
        &mut self,
        stop: Option<&dyn enough::Stop>,
    ) -> Result<Option<OwnedAnimationFrame>, At<DecodeError>> {
        if let Some(s) = stop {
            s.check().map_err(|e| at!(DecodeError::from(e)))?;
        }
        self.skip_to_start()?;

        if !self.decode_next_into_buf()? {
            return Ok(None);
        }
        let idx = self.next_frame_index;
        self.next_frame_index += 1;

        // Enforce max_frames limit.
        self.limits
            .check_frames(self.next_frame_index)
            .map_err(|e| at!(DecodeError::InvalidParameter(alloc::format!("{e}"))))?;

        // Enforce max_animation_ms limit.
        self.accumulated_duration_ms = self
            .accumulated_duration_ms
            .saturating_add(self.current_duration_ms as u64);
        self.limits
            .check_animation_ms(self.accumulated_duration_ms)
            .map_err(|e| at!(DecodeError::InvalidParameter(alloc::format!("{e}"))))?;

        // Take the frame_buf for the owned PixelBuffer. A fresh buffer will
        // be allocated on the next call (unavoidable for owned output).
        let data = core::mem::take(&mut self.frame_buf);
        let buf = PixelBuffer::from_vec(
            data,
            self.canvas_width,
            self.canvas_height,
            self.current_descriptor,
        )
        .map_err(|_| at!(DecodeError::InvalidParameter("frame size mismatch".into())))?;
        Ok(Some(OwnedAnimationFrame::new(
            buf,
            self.current_duration_ms,
            idx,
        )))
    }

    fn render_next_frame_to_sink(
        &mut self,
        stop: Option<&dyn enough::Stop>,
        sink: &mut dyn zencodec::decode::DecodeRowSink,
    ) -> Result<Option<OutputInfo>, Self::Error> {
        zencodec::helpers::copy_frame_to_sink(self, stop, sink)
    }
}

// ── Helpers ─────────────────────────────────────────────────────────────────

/// Convert a native `crate::ImageInfo` to a `zencodec::ImageInfo`.
///
/// `loop_count` is optional because lightweight probing paths may not have
/// parsed the ANIM chunk. When available (e.g. from the animation decoder),
/// pass `Some(count)` where `0` means infinite looping.
fn to_image_info(native: &crate::ImageInfo, loop_count: Option<Option<u32>>) -> ImageInfo {
    let channel_count: u8 = if native.has_alpha { 4 } else { 3 };
    let mut info = ImageInfo::new(native.width, native.height, ImageFormat::WebP)
        .with_alpha(native.has_alpha)
        .with_bit_depth(8)
        .with_channel_count(channel_count)
        .with_sequence(if native.has_animation {
            ImageSequence::Animation {
                frame_count: Some(native.frame_count),
                loop_count: loop_count.unwrap_or(None),
                random_access: false,
            }
        } else {
            ImageSequence::Single
        });
    if let Some(ref icc) = native.icc_profile {
        info = info.with_icc_profile(icc.clone());
    }
    if let Some(ref exif) = native.exif {
        // Extract orientation from EXIF before storing the raw blob.
        // WebP EXIF is raw TIFF bytes (no Exif\0\0 prefix).
        if let Some(orient_val) = crate::exif_orientation::parse_orientation(exif) {
            info = info.with_orientation(Orientation::from_exif(orient_val).unwrap_or_default());
        }
        info = info.with_exif(exif.clone());
    }
    if let Some(ref xmp) = native.xmp {
        info = info.with_xmp(xmp.clone());
    }
    info
}

#[cfg(test)]
mod tests {
    use super::*;
    use zencodec::decode::{Decode, DecodeJob, DecoderConfig};
    use zencodec::encode::{EncodeJob, Encoder, EncoderConfig};

    fn make_rgb8_pixels(w: u32, h: u32) -> PixelBuffer {
        let mut buf = PixelBuffer::new(w, h, PixelDescriptor::RGB8_SRGB);
        let mut s = buf.as_slice_mut();
        for y in 0..h {
            let row = s.row_mut(y);
            for (i, b) in row.iter_mut().enumerate() {
                *b = ((y * w * 3 + i as u32) % 256) as u8;
            }
        }
        buf
    }

    fn make_rgba8_pixels(w: u32, h: u32) -> PixelBuffer {
        let mut buf = PixelBuffer::new(w, h, PixelDescriptor::RGBA8_SRGB);
        let mut s = buf.as_slice_mut();
        for y in 0..h {
            let row = s.row_mut(y);
            for chunk in row.chunks_exact_mut(4) {
                chunk[0] = 100;
                chunk[1] = 150;
                chunk[2] = 200;
                chunk[3] = 255;
            }
        }
        buf
    }

    #[test]
    fn roundtrip_rgb8_lossy() {
        let buf = make_rgb8_pixels(64, 64);
        let enc = WebpEncoderConfig::lossy().with_quality(90.0);
        let output = enc.job().encoder().unwrap().encode(buf.as_slice()).unwrap();
        assert!(!output.is_empty());
        assert_eq!(output.format(), ImageFormat::WebP);

        let dec = WebpDecoderConfig::new();
        let decoded = dec.decode(output.data()).unwrap();
        assert_eq!(decoded.width(), 64);
        assert_eq!(decoded.height(), 64);
    }

    #[test]
    fn roundtrip_rgba8_lossless() {
        let buf = make_rgba8_pixels(32, 32);
        let enc = WebpEncoderConfig::lossless();
        let output = enc.job().encoder().unwrap().encode(buf.as_slice()).unwrap();
        assert!(!output.is_empty());

        let dec = WebpDecoderConfig::new();
        let decoded = dec.decode(output.data()).unwrap();
        assert_eq!(decoded.width(), 32);
        assert_eq!(decoded.height(), 32);
    }

    #[test]
    fn probe_header() {
        let buf = make_rgb8_pixels(16, 16);
        let enc = WebpEncoderConfig::lossy();
        let output = enc.job().encoder().unwrap().encode(buf.as_slice()).unwrap();

        let dec = WebpDecoderConfig::new();
        let info = dec.probe_header(output.data()).unwrap();
        assert_eq!(info.width, 16);
        assert_eq!(info.height, 16);
        assert_eq!(info.format, ImageFormat::WebP);
    }

    #[test]
    fn job_with_stop() {
        let buf = make_rgb8_pixels(8, 8);
        let enc = WebpEncoderConfig::lossy();
        let output = enc
            .job()
            .with_stop(zencodec::StopToken::new(enough::Unstoppable))
            .encoder()
            .unwrap()
            .encode(buf.as_slice())
            .unwrap();
        assert!(!output.is_empty());
    }

    #[test]
    fn four_layer_encode_flow() {
        let buf = make_rgb8_pixels(8, 8);
        let config = WebpEncoderConfig::lossy().with_quality(80.0);
        let output = config
            .job()
            .encoder()
            .unwrap()
            .encode(buf.as_slice())
            .unwrap();
        assert!(!output.is_empty());
        assert_eq!(output.format(), ImageFormat::WebP);
    }

    #[test]
    fn four_layer_decode_flow() {
        let buf = make_rgb8_pixels(8, 8);
        let encoded = WebpEncoderConfig::lossy()
            .job()
            .encoder()
            .unwrap()
            .encode(buf.as_slice())
            .unwrap();

        let config = WebpDecoderConfig::new();
        let job = config.clone().job();
        let info = job.output_info(encoded.data()).unwrap();
        assert_eq!(info.width, 8);
        assert_eq!(info.height, 8);

        let decoded = config
            .job()
            .decoder(Cow::Borrowed(encoded.data()), &[])
            .unwrap()
            .decode()
            .unwrap();
        assert_eq!(decoded.width(), 8);
        assert_eq!(decoded.height(), 8);
    }

    #[test]
    fn effort_and_quality_getters() {
        let config = WebpEncoderConfig::lossy()
            .with_calibrated_quality(75.0)
            .with_effort(5);

        assert_eq!(config.calibrated_quality(), Some(75.0));
        assert_eq!(config.effort(), Some(5));
        assert_eq!(
            <WebpEncoderConfig as EncoderConfig>::is_lossless(&config),
            Some(false)
        );

        let lossless = WebpEncoderConfig::lossless();
        assert_eq!(
            <WebpEncoderConfig as EncoderConfig>::is_lossless(&lossless),
            Some(true)
        );
    }

    #[test]
    fn output_info_minimal() {
        let buf = make_rgb8_pixels(4, 4);
        let encoded = WebpEncoderConfig::lossy()
            .job()
            .encoder()
            .unwrap()
            .encode(buf.as_slice())
            .unwrap();

        let dec = WebpDecoderConfig::new();
        let info = dec.job().output_info(encoded.data()).unwrap();
        assert_eq!(info.width, 4);
        assert_eq!(info.height, 4);
    }

    #[test]
    fn encoder_trait_roundtrip() {
        let buf = make_rgb8_pixels(16, 16);
        let config = WebpEncoderConfig::lossy().with_quality(80.0);
        let encoder = config.job().encoder().unwrap();
        let output = encoder.encode(buf.as_slice()).unwrap();
        assert!(!output.is_empty());
        assert_eq!(output.format(), ImageFormat::WebP);

        let dec = WebpDecoderConfig::new();
        let decoded = dec.decode(output.data()).unwrap();
        assert!(decoded.width() > 0);
        assert!(decoded.height() > 0);
    }

    #[test]
    fn encoder_trait_gray8() {
        let mut buf = PixelBuffer::new(16, 16, PixelDescriptor::GRAY8_SRGB);
        {
            let mut s = buf.as_slice_mut();
            for y in 0..16u32 {
                let row = s.row_mut(y);
                for (i, b) in row.iter_mut().enumerate() {
                    *b = ((y as usize * 16 + i) % 256) as u8;
                }
            }
        }
        let config = WebpEncoderConfig::lossy().with_quality(80.0);
        let output = config
            .job()
            .encoder()
            .unwrap()
            .encode(buf.as_slice())
            .unwrap();
        assert!(!output.is_empty());
    }

    #[test]
    fn dyn_encoder_path() {
        let buf = make_rgb8_pixels(32, 32);
        let config = WebpEncoderConfig::lossy().with_quality(75.0);
        let dyn_enc = config.job().dyn_encoder().unwrap();
        let output = dyn_enc.encode(buf.as_slice()).unwrap();
        assert!(!output.is_empty());
        assert_eq!(output.format(), ImageFormat::WebP);
    }

    #[test]
    fn encode_srgba8() {
        let mut data = vec![0u8; 16 * 16 * 4];
        for chunk in data.chunks_exact_mut(4) {
            chunk[0] = 100;
            chunk[1] = 150;
            chunk[2] = 200;
            chunk[3] = 255;
        }
        let config = WebpEncoderConfig::lossy().with_quality(75.0);
        let output = config
            .job()
            .encoder()
            .unwrap()
            .encode_srgba8(&mut data, false, 16, 16, 16)
            .unwrap();
        assert!(!output.is_empty());
    }

    #[test]
    fn capabilities_encode() {
        let caps = WebpEncoderConfig::capabilities();
        assert!(caps.icc());
        assert!(caps.exif());
        assert!(caps.xmp());
        assert!(caps.stop());
        assert!(caps.lossy());
        assert!(caps.lossless());
        assert!(caps.animation());
        assert!(caps.push_rows());
        assert!(caps.native_alpha());
        assert!(caps.enforces_max_pixels());
        assert!(caps.enforces_max_memory());
        assert_eq!(caps.effort_range(), Some([0, 10]));
        assert_eq!(caps.quality_range(), Some([0.0, 100.0]));
        // WebP doesn't natively support these
        assert!(!caps.cicp());
        assert!(!caps.hdr());
        assert!(!caps.native_16bit());
        assert!(!caps.encode_from());
    }

    #[test]
    fn capabilities_decode() {
        let caps = WebpDecoderConfig::capabilities();
        assert!(caps.icc());
        assert!(caps.exif());
        assert!(caps.xmp());
        assert!(caps.stop());
        assert!(caps.animation());
        assert!(caps.cheap_probe());
        assert!(caps.native_alpha());
        assert!(caps.enforces_max_pixels());
        assert!(caps.enforces_max_memory());
        // WebP doesn't natively support these
        assert!(!caps.cicp());
        assert!(!caps.hdr());
        assert!(!caps.native_16bit());
        assert!(!caps.decode_into());
        assert!(caps.streaming());
    }

    #[test]
    fn animation_roundtrip_lazy_decode() {
        use zencodec::decode::AnimationFrameDecoder;
        use zencodec::encode::AnimationFrameEncoder;

        // Encode a 3-frame animation.
        let frame1 = make_rgba8_pixels(16, 16);
        let frame2 = make_rgba8_pixels(16, 16);
        let frame3 = make_rgba8_pixels(16, 16);
        let enc_config = WebpEncoderConfig::lossy().with_quality(90.0);
        let mut enc = enc_config.job().animation_frame_encoder().unwrap();
        enc.push_frame(frame1.as_slice(), 100, None).unwrap();
        enc.push_frame(frame2.as_slice(), 100, None).unwrap();
        enc.push_frame(frame3.as_slice(), 100, None).unwrap();
        let output = enc.finish(None).unwrap();
        assert!(!output.is_empty());

        // Decode lazily through AnimationFrameDecoder.
        let dec_config = WebpDecoderConfig::new();
        let mut dec = dec_config
            .job()
            .animation_frame_decoder(Cow::Owned(output.data().to_vec()), &[])
            .unwrap();

        assert_eq!(dec.frame_count(), Some(3));

        // Each render_next_frame decodes one frame lazily.
        let f1 = dec.render_next_frame(None).unwrap().expect("frame 1");
        assert_eq!(f1.frame_index(), 0);
        assert_eq!(f1.duration_ms(), 100);

        let f2 = dec.render_next_frame_owned(None).unwrap().expect("frame 2");
        assert_eq!(f2.frame_index(), 1);

        let f3 = dec.render_next_frame(None).unwrap().expect("frame 3");
        assert_eq!(f3.frame_index(), 2);

        // No more frames.
        assert!(dec.render_next_frame(None).unwrap().is_none());
    }

    #[test]
    fn animation_lazy_decode_with_start_frame() {
        use zencodec::decode::AnimationFrameDecoder;
        use zencodec::encode::AnimationFrameEncoder;

        // Encode 4 frames.
        let enc_config = WebpEncoderConfig::lossy().with_quality(90.0);
        let mut enc = enc_config.job().animation_frame_encoder().unwrap();
        for _ in 0..4 {
            let frame = make_rgba8_pixels(8, 8);
            enc.push_frame(frame.as_slice(), 50, None).unwrap();
        }
        let output = enc.finish(None).unwrap();

        // Decode starting from frame 2.
        let dec_config = WebpDecoderConfig::new();
        let mut dec = dec_config
            .job()
            .with_start_frame_index(2)
            .animation_frame_decoder(Cow::Owned(output.data().to_vec()), &[])
            .unwrap();

        // First yielded frame should be index 2.
        let f = dec.render_next_frame(None).unwrap().expect("frame 2");
        assert_eq!(f.frame_index(), 2);

        let f = dec.render_next_frame(None).unwrap().expect("frame 3");
        assert_eq!(f.frame_index(), 3);

        assert!(dec.render_next_frame(None).unwrap().is_none());
    }

    #[test]
    fn streaming_encode_lossy_rgb8() {
        // Push rows in strips of 2, verify output decodes correctly.
        let buf = make_rgb8_pixels(64, 64);
        let config = WebpEncoderConfig::lossy().with_quality(90.0);
        let mut encoder = config.job().with_canvas_size(64, 64).encoder().unwrap();
        assert_eq!(encoder.preferred_strip_height(), 2);

        // Push 32 strips of 2 rows each
        let raw = buf.as_slice().contiguous_bytes();
        let raw = raw.as_ref();
        let row_bytes = 64 * 3;
        for y in (0..64).step_by(2) {
            let strip = PixelSlice::new(
                &raw[y * row_bytes..(y + 2) * row_bytes],
                64,
                2,
                row_bytes,
                PixelDescriptor::RGB8_SRGB,
            )
            .unwrap();
            encoder.push_rows(strip).unwrap();
        }

        let output = encoder.finish().unwrap();
        assert!(!output.is_empty());
        assert_eq!(output.format(), ImageFormat::WebP);

        // Verify decodable
        let dec = WebpDecoderConfig::new();
        let decoded = dec.decode(output.data()).unwrap();
        assert_eq!(decoded.width(), 64);
        assert_eq!(decoded.height(), 64);
    }

    #[test]
    fn streaming_encode_lossy_rgb8_odd_height() {
        // Image with odd height: tests the pending row chroma handling.
        let buf = make_rgb8_pixels(32, 33);
        let config = WebpEncoderConfig::lossy().with_quality(80.0);
        let mut encoder = config.job().with_canvas_size(32, 33).encoder().unwrap();

        // Push one row at a time: 16 pairs + 1 leftover
        let raw = buf.as_slice().contiguous_bytes();
        let raw = raw.as_ref();
        let row_bytes = 32 * 3;
        for y in 0..33usize {
            let strip = PixelSlice::new(
                &raw[y * row_bytes..(y + 1) * row_bytes],
                32,
                1,
                row_bytes,
                PixelDescriptor::RGB8_SRGB,
            )
            .unwrap();
            encoder.push_rows(strip).unwrap();
        }

        let output = encoder.finish().unwrap();
        let dec = WebpDecoderConfig::new();
        let decoded = dec.decode(output.data()).unwrap();
        assert_eq!(decoded.width(), 32);
        assert_eq!(decoded.height(), 33);
    }

    #[test]
    fn streaming_encode_lossless_rgba8() {
        // Lossless RGBA: uses the Raw accumulation path.
        let buf = make_rgba8_pixels(16, 16);
        let config = WebpEncoderConfig::lossless();
        let mut encoder = config.job().with_canvas_size(16, 16).encoder().unwrap();
        assert_eq!(encoder.preferred_strip_height(), 1);

        let raw = buf.as_slice().contiguous_bytes();
        let raw = raw.as_ref();
        let row_bytes = 16 * 4;
        for y in 0..16usize {
            let strip = PixelSlice::new(
                &raw[y * row_bytes..(y + 1) * row_bytes],
                16,
                1,
                row_bytes,
                PixelDescriptor::RGBA8_SRGB,
            )
            .unwrap();
            encoder.push_rows(strip).unwrap();
        }

        let output = encoder.finish().unwrap();
        assert!(!output.is_empty());

        let dec = WebpDecoderConfig::new();
        let decoded = dec.decode(output.data()).unwrap();
        assert_eq!(decoded.width(), 16);
        assert_eq!(decoded.height(), 16);
    }

    #[test]
    fn streaming_encode_matches_oneshot() {
        // Verify streaming produces output of similar size to one-shot.
        let buf = make_rgb8_pixels(64, 64);
        let config = WebpEncoderConfig::lossy().with_quality(75.0);

        // One-shot
        let oneshot_output = config
            .clone()
            .job()
            .encoder()
            .unwrap()
            .encode(buf.as_slice())
            .unwrap();

        // Streaming (full image in one push)
        let mut encoder = config.job().with_canvas_size(64, 64).encoder().unwrap();
        encoder.push_rows(buf.as_slice()).unwrap();
        let stream_output = encoder.finish().unwrap();

        // Streaming via YUV path may differ slightly from one-shot RGB path
        // due to different conversion rounding, but sizes should be similar.
        let oneshot_len = oneshot_output.data().len();
        let stream_len = stream_output.data().len();
        let ratio = stream_len as f64 / oneshot_len as f64;
        assert!(
            (0.8..1.25).contains(&ratio),
            "streaming output size {stream_len} too different from one-shot {oneshot_len} (ratio {ratio:.3})"
        );
    }

    #[test]
    fn streaming_encode_capabilities() {
        let caps = WebpEncoderConfig::capabilities();
        assert!(caps.push_rows());
    }

    #[test]
    fn last_frame_duration_preserved() {
        use zencodec::decode::AnimationFrameDecoder;
        use zencodec::encode::AnimationFrameEncoder;

        // Encode 3 frames with distinct durations: 200, 50, 300.
        let enc_config = WebpEncoderConfig::lossy().with_quality(90.0);
        let mut enc = enc_config.job().animation_frame_encoder().unwrap();
        let frame = make_rgba8_pixels(16, 16);
        enc.push_frame(frame.as_slice(), 200, None).unwrap();
        enc.push_frame(frame.as_slice(), 50, None).unwrap();
        enc.push_frame(frame.as_slice(), 300, None).unwrap();
        let output = enc.finish(None).unwrap();

        // Decode and verify each frame's duration.
        let dec_config = WebpDecoderConfig::new();
        let mut dec = dec_config
            .job()
            .animation_frame_decoder(Cow::Owned(output.data().to_vec()), &[])
            .unwrap();

        let f1 = dec.render_next_frame(None).unwrap().expect("frame 1");
        assert_eq!(
            f1.duration_ms(),
            200,
            "first frame duration should be 200ms"
        );

        let f2 = dec.render_next_frame(None).unwrap().expect("frame 2");
        assert_eq!(f2.duration_ms(), 50, "second frame duration should be 50ms");

        let f3 = dec.render_next_frame(None).unwrap().expect("frame 3");
        assert_eq!(
            f3.duration_ms(),
            300,
            "last frame duration should be 300ms, not hardcoded 100ms"
        );

        assert!(dec.render_next_frame(None).unwrap().is_none());
    }

    #[test]
    fn decode_capabilities_enforces_max_input_bytes() {
        let caps = WebpDecoderConfig::capabilities();
        assert!(
            caps.enforces_max_input_bytes(),
            "decode capabilities should report enforces_max_input_bytes since effective_input_size_limit is checked"
        );
    }

    #[test]
    fn streaming_decode_basic() {
        use zencodec::decode::StreamingDecode;

        // Encode a small test image
        let buf = make_rgb8_pixels(64, 64);
        let enc = WebpEncoderConfig::lossy().with_quality(90.0);
        let output = enc.job().encoder().unwrap().encode(buf.as_slice()).unwrap();

        // Stream decode
        let dec = WebpDecoderConfig::new();
        let mut stream = dec
            .job()
            .streaming_decoder(Cow::Borrowed(output.data()), &[])
            .unwrap();

        let info = stream.info();
        assert_eq!(info.width, 64);
        assert_eq!(info.height, 64);

        let mut total_rows = 0u32;
        let mut last_y = 0u32;
        while let Some((y, strip)) = stream.next_batch().unwrap() {
            assert!(strip.rows() > 0, "strip should have rows");
            assert_eq!(strip.width(), 64);
            if total_rows > 0 {
                assert!(
                    y > last_y,
                    "y offsets should increase: got {y} after {last_y}"
                );
            }
            last_y = y;
            total_rows += strip.rows();
        }
        assert_eq!(total_rows, 64, "should have decoded all 64 rows");

        // Should return None when done
        assert!(stream.next_batch().unwrap().is_none());
    }

    #[test]
    fn streaming_decode_matches_oneshot() {
        use zencodec::decode::{Decode, StreamingDecode};

        // Encode a test image
        let buf = make_rgb8_pixels(48, 48);
        let enc = WebpEncoderConfig::lossy().with_quality(75.0);
        let output = enc.job().encoder().unwrap().encode(buf.as_slice()).unwrap();
        let data = output.data();

        // One-shot decode (reference)
        let dec = WebpDecoderConfig::new();
        let oneshot = dec
            .clone()
            .job()
            .decoder(Cow::Borrowed(data), &[])
            .unwrap()
            .decode()
            .unwrap();
        let oneshot_buf = oneshot.into_buffer();
        let oneshot_rows = oneshot_buf.height();
        let oneshot_width = oneshot_buf.width();
        let bpp = oneshot_buf.descriptor().bytes_per_pixel();

        // Streaming decode — request the same format as oneshot
        let mut stream = dec
            .job()
            .streaming_decoder(Cow::Borrowed(data), &[])
            .unwrap();

        let mut assembled = alloc::vec![0u8; oneshot_width as usize * oneshot_rows as usize * bpp];
        while let Some((y, strip)) = stream.next_batch().unwrap() {
            let y = y as usize;
            let row_bytes = oneshot_width as usize * bpp;
            for row in 0..strip.rows() as usize {
                let src = strip.row(row as u32);
                let dst_start = (y + row) * row_bytes;
                assembled[dst_start..dst_start + row_bytes].copy_from_slice(src);
            }
        }

        // Compare pixel-by-pixel
        let reference = oneshot_buf.into_vec();
        assert_eq!(
            assembled.len(),
            reference.len(),
            "buffer sizes should match"
        );
        assert_eq!(
            assembled, reference,
            "streaming and oneshot should produce identical pixels"
        );
    }

    #[test]
    fn streaming_decode_large_image() {
        use zencodec::decode::StreamingDecode;

        // Encode a larger image to exercise multiple MB rows
        let buf = make_rgb8_pixels(128, 96);
        let enc = WebpEncoderConfig::lossy().with_quality(80.0);
        let output = enc.job().encoder().unwrap().encode(buf.as_slice()).unwrap();

        let dec = WebpDecoderConfig::new();
        let mut stream = dec
            .job()
            .streaming_decoder(Cow::Borrowed(output.data()), &[])
            .unwrap();

        let mut total_rows = 0u32;
        let mut batch_count = 0u32;
        while let Some((_y, strip)) = stream.next_batch().unwrap() {
            total_rows += strip.rows();
            batch_count += 1;
        }
        assert_eq!(total_rows, 96);
        // 96 / 16 = 6 MB rows, but delayed output means different batch count
        assert!(batch_count >= 1, "should have at least one batch");
    }

    #[test]
    fn streaming_decode_rejects_lossless() {
        // Encode a lossless image
        let buf = make_rgba8_pixels(8, 8);
        let enc = WebpEncoderConfig::lossless();
        let output = enc.job().encoder().unwrap().encode(buf.as_slice()).unwrap();

        let dec = WebpDecoderConfig::new();
        let result = dec
            .job()
            .streaming_decoder(Cow::Borrowed(output.data()), &[]);
        assert!(result.is_err(), "lossless should be rejected for streaming");
    }

    #[test]
    fn streaming_decode_with_alpha() {
        use zencodec::decode::StreamingDecode;

        // Encode RGBA image
        let buf = make_rgba8_pixels(32, 32);
        let enc = WebpEncoderConfig::lossy().with_quality(90.0);
        let output = enc.job().encoder().unwrap().encode(buf.as_slice()).unwrap();

        // Check if it has alpha
        let native_info = crate::ImageInfo::from_webp(output.data()).unwrap();

        if native_info.has_alpha {
            let dec = WebpDecoderConfig::new();
            let mut stream = dec
                .job()
                .streaming_decoder(Cow::Borrowed(output.data()), &[PixelDescriptor::RGBA8_SRGB])
                .unwrap();

            let info = stream.info();
            assert_eq!(info.width, 32);

            let mut total_rows = 0u32;
            while let Some((_y, strip)) = stream.next_batch().unwrap() {
                // RGBA: 4 bytes per pixel
                let expected_row_bytes = 32 * 4;
                assert_eq!(
                    strip.row(0).len(),
                    expected_row_bytes,
                    "should be 4bpp RGBA"
                );
                total_rows += strip.rows();
            }
            assert_eq!(total_rows, 32);
        }
    }

    // ── push_decoder tests ─────────────────────────────────────────────

    /// A minimal sink that collects every strip into a single flat buffer,
    /// then lets tests compare it byte-for-byte against a reference.
    struct CollectSink {
        buf: alloc::vec::Vec<u8>,
        width: u32,
        height: u32,
        descriptor: Option<PixelDescriptor>,
        began: bool,
        finished: bool,
        strips_received: u32,
    }

    impl CollectSink {
        fn new() -> Self {
            Self {
                buf: alloc::vec::Vec::new(),
                width: 0,
                height: 0,
                descriptor: None,
                began: false,
                finished: false,
                strips_received: 0,
            }
        }
    }

    impl zencodec::decode::DecodeRowSink for CollectSink {
        fn begin(
            &mut self,
            width: u32,
            height: u32,
            descriptor: PixelDescriptor,
        ) -> Result<(), SinkError> {
            self.width = width;
            self.height = height;
            self.descriptor = Some(descriptor);
            let bpp = descriptor.bytes_per_pixel();
            let total = (width as usize) * (height as usize) * bpp;
            self.buf.clear();
            self.buf.resize(total, 0);
            self.began = true;
            Ok(())
        }

        fn provide_next_buffer(
            &mut self,
            y: u32,
            height: u32,
            width: u32,
            descriptor: PixelDescriptor,
        ) -> Result<zenpixels::PixelSliceMut<'_>, SinkError> {
            assert!(self.began, "provide_next_buffer before begin");
            assert!(!self.finished, "provide_next_buffer after finish");
            assert_eq!(width, self.width);
            assert_eq!(Some(descriptor), self.descriptor);
            let bpp = descriptor.bytes_per_pixel();
            let row_bytes = (width as usize) * bpp;
            let start = (y as usize) * row_bytes;
            let end = start + (height as usize) * row_bytes;
            assert!(end <= self.buf.len(), "strip exceeds buffer");
            self.strips_received += 1;
            let slice = zenpixels::PixelSliceMut::new(
                &mut self.buf[start..end],
                width,
                height,
                row_bytes,
                descriptor,
            )
            .expect("valid slice");
            Ok(slice)
        }

        fn finish(&mut self) -> Result<(), SinkError> {
            self.finished = true;
            Ok(())
        }
    }

    fn push_decode(data: &[u8], preferred: &[PixelDescriptor]) -> (CollectSink, OutputInfo) {
        use zencodec::decode::DecodeJob;
        let mut sink = CollectSink::new();
        let dec = WebpDecoderConfig::new();
        let info = dec
            .job()
            .push_decoder(Cow::Borrowed(data), &mut sink, preferred)
            .expect("push_decoder");
        (sink, info)
    }

    fn full_decode(
        data: &[u8],
        preferred: &[PixelDescriptor],
    ) -> (alloc::vec::Vec<u8>, u32, u32, PixelDescriptor) {
        use zencodec::decode::DecodeJob;
        let dec = WebpDecoderConfig::new();
        let out = dec
            .job()
            .decoder(Cow::Borrowed(data), preferred)
            .unwrap()
            .decode()
            .unwrap();
        let buf = out.into_buffer();
        let width = buf.width();
        let height = buf.height();
        let desc = buf.descriptor();
        (buf.into_vec(), width, height, desc)
    }

    /// Exact byte-for-byte parity between `push_decoder` and the full-decode
    /// path on a lossy-no-alpha VP8 stream (hits the native streaming path).
    #[test]
    fn push_decoder_parity_lossy_rgb() {
        let rgb = make_rgb8_pixels(128, 96);
        let enc = WebpEncoderConfig::lossy().with_quality(85.0);
        let output = enc.job().encoder().unwrap().encode(rgb.as_slice()).unwrap();
        let data = output.data();

        let (sink, info) = push_decode(data, &[]);
        let (reference, rw, rh, rdesc) = full_decode(data, &[]);

        assert_eq!(sink.width, rw);
        assert_eq!(sink.height, rh);
        assert_eq!(sink.descriptor, Some(rdesc));
        assert_eq!(info.native_format, rdesc);
        assert!(sink.began && sink.finished);
        assert!(sink.strips_received >= 1);
        assert_eq!(sink.buf, reference, "streaming bytes must match oneshot");
    }

    /// Parity on a lossy VP8 + VP8X container with ALPH chunk. This uses
    /// the same streaming path but `alpha_plane` is applied per-strip.
    #[test]
    fn push_decoder_parity_lossy_with_alpha() {
        let rgba = make_rgba8_pixels(48, 48);
        let enc = WebpEncoderConfig::lossy().with_quality(90.0);
        let output = enc
            .job()
            .encoder()
            .unwrap()
            .encode(rgba.as_slice())
            .unwrap();
        let data = output.data();

        // Only run the parity assertion if the encoder actually produced alpha.
        let native_info = crate::ImageInfo::from_webp(data).unwrap();
        if !native_info.has_alpha {
            return;
        }

        let (sink, _info) = push_decode(data, &[PixelDescriptor::RGBA8_SRGB]);
        let (reference, _, _, rdesc) = full_decode(data, &[PixelDescriptor::RGBA8_SRGB]);
        assert_eq!(rdesc, PixelDescriptor::RGBA8_SRGB);
        assert_eq!(sink.descriptor, Some(PixelDescriptor::RGBA8_SRGB));
        assert_eq!(sink.buf, reference, "alpha strip must match full decode");
    }

    /// Lossless VP8L is **not** eligible for the native streaming path —
    /// push_decoder must fall back to the helper and still produce a
    /// correct image that matches the full-decode output byte-for-byte.
    #[test]
    fn push_decoder_fallback_lossless() {
        let rgba = make_rgba8_pixels(32, 32);
        let enc = WebpEncoderConfig::lossless();
        let output = enc
            .job()
            .encoder()
            .unwrap()
            .encode(rgba.as_slice())
            .unwrap();
        let data = output.data();

        // Sanity: the container must actually be VP8L or VP8X+VP8L for the
        // fallback path to be exercised. If the encoder produced plain VP8
        // (unlikely for lossless), this test degenerates into a lossy parity
        // check, which is still valid.
        let (sink, _info) = push_decode(data, &[]);
        let (reference, _, _, _) = full_decode(data, &[]);
        assert_eq!(sink.buf, reference, "lossless fallback must be exact");
    }

    /// `Cow::Owned` must work as well as `Cow::Borrowed` (the function
    /// body owns `data` throughout, so both lifetimes are equivalent).
    #[test]
    fn push_decoder_cow_owned_works() {
        use zencodec::decode::DecodeJob;
        let rgb = make_rgb8_pixels(32, 32);
        let enc = WebpEncoderConfig::lossy().with_quality(80.0);
        let output = enc.job().encoder().unwrap().encode(rgb.as_slice()).unwrap();

        let borrowed = {
            let mut sink = CollectSink::new();
            WebpDecoderConfig::new()
                .job()
                .push_decoder(Cow::Borrowed(output.data()), &mut sink, &[])
                .unwrap();
            sink.buf
        };
        let owned_bytes = output.data().to_vec();
        let owned = {
            let mut sink = CollectSink::new();
            WebpDecoderConfig::new()
                .job()
                .push_decoder(Cow::Owned(owned_bytes), &mut sink, &[])
                .unwrap();
            sink.buf
        };

        assert_eq!(owned, borrowed, "Cow::Owned must match Cow::Borrowed");
    }

    /// BGRA negotiation: when the sink asks for BGRA, the streaming path
    /// must swizzle and deliver BGRA, and the result must still match the
    /// full-decode BGRA output.
    #[test]
    fn push_decoder_bgra_negotiation() {
        let rgba = make_rgba8_pixels(16, 16);
        let enc = WebpEncoderConfig::lossy().with_quality(90.0);
        let output = enc
            .job()
            .encoder()
            .unwrap()
            .encode(rgba.as_slice())
            .unwrap();
        let data = output.data();

        let native_info = crate::ImageInfo::from_webp(data).unwrap();
        if !native_info.has_alpha {
            return;
        }

        let (sink, _info) = push_decode(data, &[PixelDescriptor::BGRA8_SRGB]);
        assert_eq!(sink.descriptor, Some(PixelDescriptor::BGRA8_SRGB));
        let (reference, _, _, rdesc) = full_decode(data, &[PixelDescriptor::BGRA8_SRGB]);
        assert_eq!(rdesc, PixelDescriptor::BGRA8_SRGB);
        assert_eq!(sink.buf, reference);
    }

    /// Real corpus image: gallery1/1.webp is a plain lossy VP8 at 550x368.
    /// This exercises a larger input than the synthetic gradients and
    /// verifies parity on a real-world encoder's output.
    #[test]
    fn push_decoder_parity_corpus_gallery1() {
        let data = include_bytes!("../tests/images/gallery1/1.webp");
        let (sink, info) = push_decode(data, &[]);
        let (reference, rw, rh, _) = full_decode(data, &[]);
        assert_eq!(sink.width, rw);
        assert_eq!(sink.height, rh);
        assert_eq!(info.native_format, sink.descriptor.unwrap());
        assert_eq!(sink.buf, reference, "gallery1/1.webp parity");
    }

    /// Real corpus image with alpha (VP8X + VP8 + ALPH).
    #[test]
    fn push_decoder_parity_corpus_alpha() {
        let data = include_bytes!("../tests/images/gallery2/1_webp_a.webp");
        let native = crate::ImageInfo::from_webp(data).unwrap();
        if !native.has_alpha {
            // Corpus layout guarantees _a files have alpha; assert for safety.
            panic!("expected 1_webp_a.webp to have alpha");
        }
        let (sink, _info) = push_decode(data, &[PixelDescriptor::RGBA8_SRGB]);
        let (reference, _, _, rdesc) = full_decode(data, &[PixelDescriptor::RGBA8_SRGB]);
        assert_eq!(rdesc, PixelDescriptor::RGBA8_SRGB);
        assert_eq!(sink.descriptor, Some(PixelDescriptor::RGBA8_SRGB));
        assert_eq!(sink.buf, reference, "gallery2/1_webp_a.webp parity");
    }

    /// Real corpus lossless image — must go through the fallback path.
    #[test]
    fn push_decoder_parity_corpus_lossless() {
        let data = include_bytes!("../tests/images/gallery2/1_webp_ll.webp");
        let (sink, _info) = push_decode(data, &[]);
        let (reference, _, _, _) = full_decode(data, &[]);
        assert_eq!(sink.buf, reference, "lossless fallback parity");
    }
}
