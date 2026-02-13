//! zencodec-types trait implementations for zenwebp.
//!
//! Provides [`WebpEncoding`] and [`WebpDecoding`] types that implement the
//! [`Encoding`] / [`Decoding`] traits from zencodec-types, wrapping the native
//! zenwebp API.
//!
//! The native API remains untouched — this is a thin adapter layer.

use alloc::vec::Vec;

use zencodec_types::{
    CodecCapabilities, DecodeOutput, Decoding, DecodingJob, EncodeOutput, Encoding, EncodingJob,
    ImageFormat, ImageInfo, ImageMetadata, ImgRef, PixelData, ResourceLimits, Stop,
};
use zencodec_types::rgb::alt::BGRA;

use crate::encoder::config::EncoderConfig;
use crate::{DecodeConfig, DecodeError, DecodeRequest, EncodeError, EncodeRequest, PixelLayout};

// ── Encoding ────────────────────────────────────────────────────────────────

/// WebP encoder configuration implementing [`Encoding`].
///
/// Wraps [`EncoderConfig`] (lossy/lossless enum) with resource limits for the
/// trait interface.
///
/// # Examples
///
/// ```rust
/// use zencodec_types::Encoding;
/// use zenwebp::WebpEncoding;
///
/// let enc = WebpEncoding::lossy()
///     .with_quality(85.0)
///     .with_sharp_yuv(true);
/// ```
#[derive(Clone, Debug)]
pub struct WebpEncoding {
    inner: EncoderConfig,
    limits: ResourceLimits,
}

impl WebpEncoding {
    /// Create a lossy encoder config with defaults.
    #[must_use]
    pub fn lossy() -> Self {
        Self {
            inner: EncoderConfig::new_lossy(),
            limits: ResourceLimits::none(),
        }
    }

    /// Create a lossless encoder config with defaults.
    #[must_use]
    pub fn lossless() -> Self {
        Self {
            inner: EncoderConfig::new_lossless(),
            limits: ResourceLimits::none(),
        }
    }

    /// Create from a preset with the given quality.
    #[must_use]
    pub fn with_preset(preset: crate::Preset, quality: f32) -> Self {
        Self {
            inner: EncoderConfig::with_preset(preset, quality),
            limits: ResourceLimits::none(),
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
    pub fn with_effort(mut self, effort: u32) -> Self {
        let method = ((effort as u64 * 6) / 10).min(6) as u8;
        self.inner = self.inner.with_method(method);
        self
    }

    /// Enable or disable lossless encoding.
    #[must_use]
    pub fn with_lossless(mut self, lossless: bool) -> Self {
        self.inner = self.inner.with_lossless(lossless);
        self
    }

    /// Set alpha plane quality (0.0-100.0).
    #[must_use]
    pub fn with_alpha_quality(mut self, quality: f32) -> Self {
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

    /// Access the underlying [`EncoderConfig`].
    #[must_use]
    pub fn inner(&self) -> &EncoderConfig {
        &self.inner
    }

    /// Mutably access the underlying [`EncoderConfig`].
    pub fn inner_mut(&mut self) -> &mut EncoderConfig {
        &mut self.inner
    }
}

static ENCODE_CAPS: CodecCapabilities = CodecCapabilities::new()
    .with_encode_icc(true)
    .with_encode_exif(true)
    .with_encode_xmp(true)
    .with_encode_cancel(true);

impl Encoding for WebpEncoding {
    type Error = EncodeError;
    type Job<'a> = WebpEncodeJob<'a>;

    fn capabilities() -> &'static CodecCapabilities {
        &ENCODE_CAPS
    }

    fn with_limits(mut self, limits: ResourceLimits) -> Self {
        self.limits = limits;
        self
    }

    fn job(&self) -> WebpEncodeJob<'_> {
        WebpEncodeJob {
            config: self,
            stop: None,
            icc: None,
            exif: None,
            xmp: None,
            limits: ResourceLimits::none(),
        }
    }
}

/// Per-operation WebP encode job.
pub struct WebpEncodeJob<'a> {
    config: &'a WebpEncoding,
    stop: Option<&'a dyn Stop>,
    icc: Option<&'a [u8]>,
    exif: Option<&'a [u8]>,
    xmp: Option<&'a [u8]>,
    limits: ResourceLimits,
}

impl<'a> WebpEncodeJob<'a> {
    /// Set ICC profile data.
    #[must_use]
    pub fn with_icc(mut self, icc: &'a [u8]) -> Self {
        self.icc = Some(icc);
        self
    }

    /// Set EXIF data.
    #[must_use]
    pub fn with_exif(mut self, exif: &'a [u8]) -> Self {
        self.exif = Some(exif);
        self
    }

    /// Set XMP data.
    #[must_use]
    pub fn with_xmp(mut self, xmp: &'a [u8]) -> Self {
        self.xmp = Some(xmp);
        self
    }

    fn effective_limit_pixels(&self) -> Option<u64> {
        self.limits.max_pixels.or(self.config.limits.max_pixels)
    }

    fn effective_limit_memory(&self) -> Option<u64> {
        self.limits
            .max_memory_bytes
            .or(self.config.limits.max_memory_bytes)
    }

    fn build_inner_config(&self) -> EncoderConfig {
        let mut inner = self.config.inner.clone();
        // Apply limits to the inner config
        let mut limits = crate::Limits::none();
        if let Some(px) = self.effective_limit_pixels() {
            limits = limits.max_total_pixels(px);
        }
        if let Some(mem) = self.effective_limit_memory() {
            limits = limits.max_memory(mem);
        }
        inner = inner.limits(limits);
        inner
    }

    fn build_metadata(&self) -> crate::ImageMetadata<'a> {
        let mut meta = crate::ImageMetadata::new();
        if let Some(icc) = self.icc {
            meta = meta.with_icc_profile(icc);
        }
        if let Some(exif) = self.exif {
            meta = meta.with_exif(exif);
        }
        if let Some(xmp) = self.xmp {
            meta = meta.with_xmp(xmp);
        }
        meta
    }

    fn has_metadata(&self) -> bool {
        self.icc.is_some() || self.exif.is_some() || self.xmp.is_some()
    }

    fn do_encode(
        self,
        pixels: &[u8],
        layout: PixelLayout,
        w: u32,
        h: u32,
    ) -> Result<EncodeOutput, EncodeError> {
        let inner = self.build_inner_config();
        let mut req = EncodeRequest::new(&inner, pixels, layout, w, h);

        if let Some(stop) = self.stop {
            req = req.with_stop(stop);
        }

        if self.has_metadata() {
            req = req.with_metadata(self.build_metadata());
        }

        let data = req.encode()?;
        Ok(EncodeOutput::new(data, ImageFormat::WebP))
    }
}

impl<'a> EncodingJob<'a> for WebpEncodeJob<'a> {
    type Error = EncodeError;

    fn with_stop(mut self, stop: &'a dyn Stop) -> Self {
        self.stop = Some(stop);
        self
    }

    fn with_metadata(mut self, meta: &'a ImageMetadata<'a>) -> Self {
        if let Some(icc) = meta.icc_profile {
            self.icc = Some(icc);
        }
        if let Some(exif) = meta.exif {
            self.exif = Some(exif);
        }
        if let Some(xmp) = meta.xmp {
            self.xmp = Some(xmp);
        }
        self
    }

    fn with_limits(mut self, limits: ResourceLimits) -> Self {
        self.limits = limits;
        self
    }

    fn encode_rgb8(
        self,
        img: ImgRef<'_, zencodec_types::Rgb<u8>>,
    ) -> Result<EncodeOutput, Self::Error> {
        let (buf, w, h) = img_rgb_to_bytes(img);
        self.do_encode(&buf, PixelLayout::Rgb8, w, h)
    }

    fn encode_rgba8(
        self,
        img: ImgRef<'_, zencodec_types::Rgba<u8>>,
    ) -> Result<EncodeOutput, Self::Error> {
        let (buf, w, h) = img_rgba_to_bytes(img);
        self.do_encode(&buf, PixelLayout::Rgba8, w, h)
    }

    fn encode_gray8(
        self,
        img: ImgRef<'_, zencodec_types::Gray<u8>>,
    ) -> Result<EncodeOutput, Self::Error> {
        // WebP doesn't support grayscale natively — expand to RGB
        let (buf, _, _) = img.to_contiguous_buf();
        let w = img.width() as u32;
        let h = img.height() as u32;
        let rgb: Vec<u8> = buf
            .iter()
            .flat_map(|g| {
                let v = g.value();
                [v, v, v]
            })
            .collect();
        self.do_encode(&rgb, PixelLayout::Rgb8, w, h)
    }

    fn encode_bgra8(
        self,
        img: ImgRef<'_, BGRA<u8>>,
    ) -> Result<EncodeOutput, Self::Error> {
        // Swizzle BGRA → RGBA
        let (buf, _, _) = img.to_contiguous_buf();
        let w = img.width() as u32;
        let h = img.height() as u32;
        let rgba: Vec<u8> = buf
            .iter()
            .flat_map(|p| [p.r, p.g, p.b, p.a])
            .collect();
        self.do_encode(&rgba, PixelLayout::Rgba8, w, h)
    }

    fn encode_bgrx8(
        self,
        img: ImgRef<'_, BGRA<u8>>,
    ) -> Result<EncodeOutput, Self::Error> {
        // Swizzle BGRA → RGB (drop alpha/padding)
        let (buf, _, _) = img.to_contiguous_buf();
        let w = img.width() as u32;
        let h = img.height() as u32;
        let rgb: Vec<u8> = buf
            .iter()
            .flat_map(|p| [p.r, p.g, p.b])
            .collect();
        self.do_encode(&rgb, PixelLayout::Rgb8, w, h)
    }

    fn encode_rgb_f32(
        self,
        img: ImgRef<'_, zencodec_types::Rgb<f32>>,
    ) -> Result<EncodeOutput, Self::Error> {
        use linear_srgb::default::linear_to_srgb_u8;
        let (buf, _, _) = img.to_contiguous_buf();
        let w = img.width() as u32;
        let h = img.height() as u32;
        let rgb: Vec<u8> = buf
            .iter()
            .flat_map(|p| {
                [
                    linear_to_srgb_u8(p.r.clamp(0.0, 1.0)),
                    linear_to_srgb_u8(p.g.clamp(0.0, 1.0)),
                    linear_to_srgb_u8(p.b.clamp(0.0, 1.0)),
                ]
            })
            .collect();
        self.do_encode(&rgb, PixelLayout::Rgb8, w, h)
    }

    fn encode_rgba_f32(
        self,
        img: ImgRef<'_, zencodec_types::Rgba<f32>>,
    ) -> Result<EncodeOutput, Self::Error> {
        use linear_srgb::default::linear_to_srgb_u8;
        let (buf, _, _) = img.to_contiguous_buf();
        let w = img.width() as u32;
        let h = img.height() as u32;
        let rgba: Vec<u8> = buf
            .iter()
            .flat_map(|p| {
                [
                    linear_to_srgb_u8(p.r.clamp(0.0, 1.0)),
                    linear_to_srgb_u8(p.g.clamp(0.0, 1.0)),
                    linear_to_srgb_u8(p.b.clamp(0.0, 1.0)),
                    (p.a.clamp(0.0, 1.0) * 255.0 + 0.5) as u8,
                ]
            })
            .collect();
        self.do_encode(&rgba, PixelLayout::Rgba8, w, h)
    }

    fn encode_gray_f32(
        self,
        img: ImgRef<'_, zencodec_types::Gray<f32>>,
    ) -> Result<EncodeOutput, Self::Error> {
        use linear_srgb::default::linear_to_srgb_u8;
        let (buf, _, _) = img.to_contiguous_buf();
        let w = img.width() as u32;
        let h = img.height() as u32;
        let rgb: Vec<u8> = buf
            .iter()
            .flat_map(|g| {
                let v = linear_to_srgb_u8(g.value().clamp(0.0, 1.0));
                [v, v, v]
            })
            .collect();
        self.do_encode(&rgb, PixelLayout::Rgb8, w, h)
    }
}

// ── Decoding ────────────────────────────────────────────────────────────────

/// WebP decoder configuration implementing [`Decoding`].
///
/// Wraps [`DecodeConfig`] with resource limits for the trait interface.
#[derive(Clone, Debug)]
pub struct WebpDecoding {
    inner: DecodeConfig,
    limits: ResourceLimits,
}

impl WebpDecoding {
    /// Create a new decoder config with defaults.
    #[must_use]
    pub fn new() -> Self {
        Self {
            inner: DecodeConfig::default(),
            limits: ResourceLimits::none(),
        }
    }

    /// Set the chroma upsampling method.
    #[must_use]
    pub fn with_upsampling(mut self, method: crate::UpsamplingMethod) -> Self {
        self.inner = self.inner.upsampling(method);
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
}

impl Default for WebpDecoding {
    fn default() -> Self {
        Self::new()
    }
}

static DECODE_CAPS: CodecCapabilities = CodecCapabilities::new()
    .with_decode_icc(true)
    .with_decode_exif(true)
    .with_decode_xmp(true)
    .with_decode_cancel(true)
    .with_cheap_probe(true);

impl Decoding for WebpDecoding {
    type Error = DecodeError;
    type Job<'a> = WebpDecodeJob<'a>;

    fn capabilities() -> &'static CodecCapabilities {
        &DECODE_CAPS
    }

    fn with_limits(mut self, limits: ResourceLimits) -> Self {
        // Also propagate supported limits to the inner DecodeConfig
        if let Some(px) = limits.max_pixels {
            self.inner.limits = self.inner.limits.max_total_pixels(px);
        }
        if let Some(mem) = limits.max_memory_bytes {
            self.inner.limits = self.inner.limits.max_memory(mem);
        }
        if let Some(w) = limits.max_width {
            if let Some(h) = limits.max_height {
                self.inner.limits = self.inner.limits.max_dimensions(w, h);
            }
        }
        self.limits = limits;
        self
    }

    fn job(&self) -> WebpDecodeJob<'_> {
        WebpDecodeJob {
            config: self,
            stop: None,
            limits: ResourceLimits::none(),
        }
    }

    fn probe_header(&self, data: &[u8]) -> Result<ImageInfo, Self::Error> {
        if let Some(max) = self.limits.max_file_size {
            if data.len() as u64 > max {
                return Err(DecodeError::InvalidParameter(alloc::format!(
                    "file size {} exceeds limit {}",
                    data.len(),
                    max
                )));
            }
        }
        let native = crate::ImageInfo::from_webp(data)?;
        Ok(to_image_info(&native))
    }
}

/// Per-operation WebP decode job.
pub struct WebpDecodeJob<'a> {
    config: &'a WebpDecoding,
    stop: Option<&'a dyn Stop>,
    limits: ResourceLimits,
}

impl<'a> WebpDecodeJob<'a> {
    fn build_config(&self) -> DecodeConfig {
        let mut cfg = self.config.inner.clone();
        if let Some(px) = self.limits.max_pixels {
            cfg.limits = cfg.limits.max_total_pixels(px);
        }
        if let Some(mem) = self.limits.max_memory_bytes {
            cfg.limits = cfg.limits.max_memory(mem);
        }
        if let Some(w) = self.limits.max_width {
            if let Some(h) = self.limits.max_height {
                cfg.limits = cfg.limits.max_dimensions(w, h);
            }
        }
        cfg
    }

    fn effective_file_size_limit(&self) -> Option<u64> {
        self.limits
            .max_file_size
            .or(self.config.limits.max_file_size)
    }
}

impl<'a> DecodingJob<'a> for WebpDecodeJob<'a> {
    type Error = DecodeError;

    fn with_stop(mut self, stop: &'a dyn Stop) -> Self {
        self.stop = Some(stop);
        self
    }

    fn with_limits(mut self, limits: ResourceLimits) -> Self {
        self.limits = limits;
        self
    }

    fn decode(self, data: &[u8]) -> Result<DecodeOutput, Self::Error> {
        // Check file size limit
        if let Some(max) = self.effective_file_size_limit() {
            if data.len() as u64 > max {
                return Err(DecodeError::InvalidParameter(alloc::format!(
                    "file size {} exceeds limit {}",
                    data.len(),
                    max
                )));
            }
        }

        let cfg = self.build_config();
        let mut req = DecodeRequest::new(&cfg, data);

        if let Some(stop) = self.stop {
            req = req.stop(stop);
        }

        let (pixels, w, h, layout) = req.decode()?;
        let w_usize = w as usize;
        let h_usize = h as usize;

        let pixel_data = match layout {
            PixelLayout::Rgb8 => {
                let rgb = bytes_to_rgb(&pixels);
                PixelData::Rgb8(zencodec_types::ImgVec::new(rgb, w_usize, h_usize))
            }
            PixelLayout::Rgba8 => {
                let rgba = bytes_to_rgba(&pixels);
                PixelData::Rgba8(zencodec_types::ImgVec::new(rgba, w_usize, h_usize))
            }
            _ => {
                // Fallback: decode as RGBA
                let rgba_req = DecodeRequest::new(&cfg, data);
                let (rgba_pixels, rw, rh) = if let Some(stop) = self.stop {
                    rgba_req.stop(stop).decode_rgba()?
                } else {
                    rgba_req.decode_rgba()?
                };
                let rgba = bytes_to_rgba(&rgba_pixels);
                PixelData::Rgba8(zencodec_types::ImgVec::new(rgba, rw as usize, rh as usize))
            }
        };

        // Probe for metadata
        let native_info = crate::ImageInfo::from_webp(data).ok();

        let info = if let Some(ref ni) = native_info {
            to_image_info(ni)
        } else {
            ImageInfo::new(w, h, ImageFormat::WebP).with_alpha(pixel_data.has_alpha())
        };

        Ok(DecodeOutput::new(pixel_data, info))
    }

    fn decode_into_rgb8(
        self,
        data: &[u8],
        dst: zencodec_types::ImgRefMut<'_, zencodec_types::Rgb<u8>>,
    ) -> Result<ImageInfo, Self::Error> {
        if let Some(max) = self.effective_file_size_limit() {
            if data.len() as u64 > max {
                return Err(DecodeError::InvalidParameter(alloc::format!(
                    "file size {} exceeds limit {}",
                    data.len(),
                    max
                )));
            }
        }

        let cfg = self.build_config();
        let mut req = DecodeRequest::new(&cfg, data);
        if let Some(stop) = self.stop {
            req = req.stop(stop);
        }

        let width = dst.width();
        let stride = dst.stride();
        if stride != width {
            req = req.stride(stride as u32);
        }

        use zencodec_types::rgb::ComponentBytes;
        let bytes: &mut [u8] = dst.into_buf().as_bytes_mut();
        let (w, h) = req.decode_rgb_into(bytes)?;

        let native_info = crate::ImageInfo::from_webp(data).ok();
        let info = if let Some(ref ni) = native_info {
            to_image_info(ni)
        } else {
            ImageInfo::new(w, h, ImageFormat::WebP)
        };

        Ok(info)
    }

    fn decode_into_rgba8(
        self,
        data: &[u8],
        dst: zencodec_types::ImgRefMut<'_, zencodec_types::Rgba<u8>>,
    ) -> Result<ImageInfo, Self::Error> {
        // Check file size limit
        if let Some(max) = self.effective_file_size_limit() {
            if data.len() as u64 > max {
                return Err(DecodeError::InvalidParameter(alloc::format!(
                    "file size {} exceeds limit {}",
                    data.len(),
                    max
                )));
            }
        }

        let cfg = self.build_config();
        let mut req = DecodeRequest::new(&cfg, data);
        if let Some(stop) = self.stop {
            req = req.stop(stop);
        }

        // Use stride if dst has padding between rows
        let width = dst.width();
        let stride = dst.stride();
        if stride != width {
            req = req.stride(stride as u32);
        }

        // Cast the caller's Rgba<u8> buffer to raw bytes for zero-copy decode
        use zencodec_types::rgb::ComponentBytes;
        let bytes: &mut [u8] = dst.into_buf().as_bytes_mut();
        let (w, h) = req.decode_rgba_into(bytes)?;

        // Build metadata
        let native_info = crate::ImageInfo::from_webp(data).ok();
        let info = if let Some(ref ni) = native_info {
            to_image_info(ni)
        } else {
            ImageInfo::new(w, h, ImageFormat::WebP).with_alpha(true)
        };

        Ok(info)
    }

    fn decode_into_gray8(
        self,
        data: &[u8],
        mut dst: zencodec_types::ImgRefMut<'_, zencodec_types::Gray<u8>>,
    ) -> Result<ImageInfo, Self::Error> {
        // WebP has no native grayscale — decode RGB and convert
        let output = self.decode(data)?;
        let info = output.info().clone();
        let src = output.into_rgb8();
        for (src_row, dst_row) in src.as_ref().rows().zip(dst.rows_mut()) {
            for (s, d) in src_row.iter().zip(dst_row.iter_mut()) {
                // ITU-R BT.601 luma
                let luma = ((s.r as u16 * 77 + s.g as u16 * 150 + s.b as u16 * 29) >> 8) as u8;
                *d = zencodec_types::Gray(luma);
            }
        }
        Ok(info)
    }

    fn decode_into_bgra8(
        self,
        data: &[u8],
        mut dst: zencodec_types::ImgRefMut<'_, BGRA<u8>>,
    ) -> Result<ImageInfo, Self::Error> {
        let output = self.decode(data)?;
        let info = output.info().clone();
        let src = output.into_rgba8();
        for (src_row, dst_row) in src.as_ref().rows().zip(dst.rows_mut()) {
            for (s, d) in src_row.iter().zip(dst_row.iter_mut()) {
                *d = BGRA { b: s.b, g: s.g, r: s.r, a: s.a };
            }
        }
        Ok(info)
    }

    fn decode_into_bgrx8(
        self,
        data: &[u8],
        mut dst: zencodec_types::ImgRefMut<'_, BGRA<u8>>,
    ) -> Result<ImageInfo, Self::Error> {
        let output = self.decode(data)?;
        let info = output.info().clone();
        let src = output.into_rgb8();
        for (src_row, dst_row) in src.as_ref().rows().zip(dst.rows_mut()) {
            for (s, d) in src_row.iter().zip(dst_row.iter_mut()) {
                *d = BGRA { b: s.b, g: s.g, r: s.r, a: 255 };
            }
        }
        Ok(info)
    }

    fn decode_into_rgb_f32(
        self,
        data: &[u8],
        mut dst: zencodec_types::ImgRefMut<'_, zencodec_types::Rgb<f32>>,
    ) -> Result<ImageInfo, Self::Error> {
        use linear_srgb::default::srgb_u8_to_linear;
        let output = self.decode(data)?;
        let info = output.info().clone();
        let src = output.into_rgb8();
        for (src_row, dst_row) in src.as_ref().rows().zip(dst.rows_mut()) {
            for (s, d) in src_row.iter().zip(dst_row.iter_mut()) {
                *d = zencodec_types::Rgb {
                    r: srgb_u8_to_linear(s.r),
                    g: srgb_u8_to_linear(s.g),
                    b: srgb_u8_to_linear(s.b),
                };
            }
        }
        Ok(info)
    }

    fn decode_into_rgba_f32(
        self,
        data: &[u8],
        mut dst: zencodec_types::ImgRefMut<'_, zencodec_types::Rgba<f32>>,
    ) -> Result<ImageInfo, Self::Error> {
        use linear_srgb::default::srgb_u8_to_linear;
        let output = self.decode(data)?;
        let info = output.info().clone();
        let src = output.into_rgba8();
        for (src_row, dst_row) in src.as_ref().rows().zip(dst.rows_mut()) {
            for (s, d) in src_row.iter().zip(dst_row.iter_mut()) {
                *d = zencodec_types::Rgba {
                    r: srgb_u8_to_linear(s.r),
                    g: srgb_u8_to_linear(s.g),
                    b: srgb_u8_to_linear(s.b),
                    a: s.a as f32 / 255.0,
                };
            }
        }
        Ok(info)
    }

    fn decode_into_gray_f32(
        self,
        data: &[u8],
        mut dst: zencodec_types::ImgRefMut<'_, zencodec_types::Gray<f32>>,
    ) -> Result<ImageInfo, Self::Error> {
        use linear_srgb::default::srgb_u8_to_linear;
        let output = self.decode(data)?;
        let info = output.info().clone();
        let src = output.into_rgb8();
        for (src_row, dst_row) in src.as_ref().rows().zip(dst.rows_mut()) {
            for (s, d) in src_row.iter().zip(dst_row.iter_mut()) {
                // Convert to linear, then luma via linear-light coefficients
                let r = srgb_u8_to_linear(s.r);
                let g = srgb_u8_to_linear(s.g);
                let b = srgb_u8_to_linear(s.b);
                *d = zencodec_types::Gray(0.2126 * r + 0.7152 * g + 0.0722 * b);
            }
        }
        Ok(info)
    }
}

// ── Helpers ─────────────────────────────────────────────────────────────────

/// Convert a native `crate::ImageInfo` to a `zencodec_types::ImageInfo`.
fn to_image_info(native: &crate::ImageInfo) -> ImageInfo {
    let mut info = ImageInfo::new(native.width, native.height, ImageFormat::WebP)
        .with_alpha(native.has_alpha)
        .with_animation(native.has_animation)
        .with_frame_count(native.frame_count);
    if let Some(ref icc) = native.icc_profile {
        info = info.with_icc_profile(icc.clone());
    }
    if let Some(ref exif) = native.exif {
        info = info.with_exif(exif.clone());
    }
    if let Some(ref xmp) = native.xmp {
        info = info.with_xmp(xmp.clone());
    }
    info
}

/// Convert an ImgRef<Rgb<u8>> to contiguous bytes.
fn img_rgb_to_bytes(img: ImgRef<'_, zencodec_types::Rgb<u8>>) -> (Vec<u8>, u32, u32) {
    let (buf, _, _) = img.to_contiguous_buf();
    let w = img.width() as u32;
    let h = img.height() as u32;
    let bytes: Vec<u8> = buf.iter().flat_map(|p| [p.r, p.g, p.b]).collect();
    (bytes, w, h)
}

/// Convert an ImgRef<Rgba<u8>> to contiguous bytes.
fn img_rgba_to_bytes(img: ImgRef<'_, zencodec_types::Rgba<u8>>) -> (Vec<u8>, u32, u32) {
    let (buf, _, _) = img.to_contiguous_buf();
    let w = img.width() as u32;
    let h = img.height() as u32;
    let bytes: Vec<u8> = buf.iter().flat_map(|p| [p.r, p.g, p.b, p.a]).collect();
    (bytes, w, h)
}

/// Convert raw RGB bytes to Vec<Rgb<u8>>.
fn bytes_to_rgb(data: &[u8]) -> Vec<zencodec_types::Rgb<u8>> {
    data.chunks_exact(3)
        .map(|c| zencodec_types::Rgb {
            r: c[0],
            g: c[1],
            b: c[2],
        })
        .collect()
}

/// Convert raw RGBA bytes to Vec<Rgba<u8>>.
fn bytes_to_rgba(data: &[u8]) -> Vec<zencodec_types::Rgba<u8>> {
    data.chunks_exact(4)
        .map(|c| zencodec_types::Rgba {
            r: c[0],
            g: c[1],
            b: c[2],
            a: c[3],
        })
        .collect()
}

#[cfg(test)]
mod tests {
    use super::*;
    use zencodec_types::{Decoding, Encoding, ImgVec, Rgb, Rgba};

    #[test]
    fn roundtrip_rgb8_lossy() {
        let pixels: Vec<Rgb<u8>> = (0..64 * 64)
            .map(|i| Rgb {
                r: (i % 256) as u8,
                g: ((i * 7) % 256) as u8,
                b: ((i * 13) % 256) as u8,
            })
            .collect();
        let img = ImgVec::new(pixels, 64, 64);

        let enc = WebpEncoding::lossy().with_quality(90.0);
        let output = enc.encode_rgb8(img.as_ref()).unwrap();
        assert!(!output.is_empty());
        assert_eq!(output.format(), ImageFormat::WebP);

        let dec = WebpDecoding::new();
        let decoded = dec.decode(output.bytes()).unwrap();
        assert_eq!(decoded.width(), 64);
        assert_eq!(decoded.height(), 64);
    }

    #[test]
    fn roundtrip_rgba8_lossless() {
        let pixels: Vec<Rgba<u8>> = (0..32 * 32)
            .map(|i| Rgba {
                r: (i % 256) as u8,
                g: ((i * 3) % 256) as u8,
                b: ((i * 7) % 256) as u8,
                a: 255,
            })
            .collect();
        let img = ImgVec::new(pixels, 32, 32);

        let enc = WebpEncoding::lossless();
        let output = enc.encode_rgba8(img.as_ref()).unwrap();
        assert!(!output.is_empty());

        let dec = WebpDecoding::new();
        let decoded = dec.decode(output.bytes()).unwrap();
        assert_eq!(decoded.width(), 32);
        assert_eq!(decoded.height(), 32);
    }

    #[test]
    fn probe_header() {
        let pixels: Vec<Rgb<u8>> = vec![
            Rgb {
                r: 128,
                g: 128,
                b: 128,
            };
            16 * 16
        ];
        let img = ImgVec::new(pixels, 16, 16);
        let output = WebpEncoding::lossy().encode_rgb8(img.as_ref()).unwrap();

        let dec = WebpDecoding::new();
        let info = dec.probe_header(output.bytes()).unwrap();
        assert_eq!(info.width, 16);
        assert_eq!(info.height, 16);
        assert_eq!(info.format, ImageFormat::WebP);
    }

    #[test]
    fn encode_gray8() {
        use zencodec_types::Gray;
        let pixels: Vec<Gray<u8>> = (0..16 * 16).map(|i| Gray((i % 256) as u8)).collect();
        let img = ImgVec::new(pixels, 16, 16);

        let enc = WebpEncoding::lossy().with_quality(80.0);
        let output = enc.encode_gray8(img.as_ref()).unwrap();
        assert!(!output.is_empty());
    }

    #[test]
    fn job_with_stop() {
        use zencodec_types::Unstoppable;
        let pixels: Vec<Rgb<u8>> = vec![Rgb { r: 0, g: 0, b: 0 }; 8 * 8];
        let img = ImgVec::new(pixels, 8, 8);
        let enc = WebpEncoding::lossy();
        let output = enc
            .job()
            .with_stop(&Unstoppable)
            .encode_rgb8(img.as_ref())
            .unwrap();
        assert!(!output.is_empty());
    }

    #[test]
    fn f32_roundtrip_all_simd_tiers() {
        use archmage::testing::{for_each_token_permutation, CompileTimePolicy};
        use zencodec_types::Gray;

        let report = for_each_token_permutation(CompileTimePolicy::Warn, |_perm| {
            // Encode linear f32 → WebP → decode back to f32
            let pixels: Vec<Rgb<f32>> = (0..16 * 16)
                .map(|i| {
                    let t = i as f32 / 255.0;
                    Rgb {
                        r: t,
                        g: (t * 0.7),
                        b: (t * 0.3),
                    }
                })
                .collect();
            let img = ImgVec::new(pixels, 16, 16);

            let enc = WebpEncoding::lossless();
            let output = enc.encode_rgb_f32(img.as_ref()).unwrap();
            assert!(!output.is_empty());

            let dec = WebpDecoding::new();
            let dst = vec![
                Rgb {
                    r: 0.0f32,
                    g: 0.0,
                    b: 0.0,
                };
                16 * 16
            ];
            let mut dst_img = ImgVec::new(dst, 16, 16);
            let _info = dec
                .decode_into_rgb_f32(output.bytes(), dst_img.as_mut())
                .unwrap();

            // Verify values are in valid range
            for p in dst_img.buf().iter() {
                assert!(p.r >= 0.0 && p.r <= 1.0, "r out of range: {}", p.r);
                assert!(p.g >= 0.0 && p.g <= 1.0, "g out of range: {}", p.g);
                assert!(p.b >= 0.0 && p.b <= 1.0, "b out of range: {}", p.b);
            }

            // Also test gray f32 path
            let gray_pixels: Vec<Gray<f32>> =
                (0..8 * 8).map(|i| Gray(i as f32 / 63.0)).collect();
            let gray_img = ImgVec::new(gray_pixels, 8, 8);
            let gray_out = WebpEncoding::lossless()
                .encode_gray_f32(gray_img.as_ref())
                .unwrap();
            assert!(!gray_out.is_empty());
        });
        assert!(report.permutations_run >= 1);
    }

    #[test]
    fn f32_rgba_roundtrip_lossless() {
        let pixels: Vec<Rgba<f32>> = (0..16 * 16)
            .map(|i| {
                let t = i as f32 / 255.0;
                Rgba {
                    r: t,
                    g: (t * 0.7),
                    b: (t * 0.3),
                    a: 1.0 - t * 0.5,
                }
            })
            .collect();
        let img = ImgVec::new(pixels, 16, 16);

        let enc = WebpEncoding::lossless();
        let output = enc.encode_rgba_f32(img.as_ref()).unwrap();
        assert!(!output.is_empty());

        let dec = WebpDecoding::new();
        let mut dst_img = ImgVec::new(
            vec![Rgba { r: 0.0f32, g: 0.0, b: 0.0, a: 0.0 }; 16 * 16],
            16,
            16,
        );
        dec.decode_into_rgba_f32(output.bytes(), dst_img.as_mut())
            .unwrap();

        for p in dst_img.buf().iter() {
            assert!(p.r >= 0.0 && p.r <= 1.0, "r out of range: {}", p.r);
            assert!(p.g >= 0.0 && p.g <= 1.0, "g out of range: {}", p.g);
            assert!(p.b >= 0.0 && p.b <= 1.0, "b out of range: {}", p.b);
            assert!(p.a >= 0.0 && p.a <= 1.0, "a out of range: {}", p.a);
        }
    }

    #[test]
    fn f32_gray_decode_values() {
        use linear_srgb::default::srgb_u8_to_linear;
        use zencodec_types::Gray;

        // Encode known sRGB gray values as RGB, decode to gray f32
        let pixels = vec![
            Rgb { r: 0u8, g: 0, b: 0 },
            Rgb { r: 128, g: 128, b: 128 },
            Rgb { r: 255, g: 255, b: 255 },
            Rgb { r: 128, g: 128, b: 128 },
        ];
        let img = ImgVec::new(pixels, 2, 2);
        let enc = WebpEncoding::lossless();
        let output = enc.encode_rgb8(img.as_ref()).unwrap();

        let dec = WebpDecoding::new();
        let mut dst = ImgVec::new(vec![Gray(0.0f32); 4], 2, 2);
        dec.decode_into_gray_f32(output.bytes(), dst.as_mut())
            .unwrap();

        let buf = dst.buf();
        // Black → 0.0
        assert!(buf[0].0.abs() < 0.02, "black: {}", buf[0].0);
        // White → 1.0
        assert!((buf[2].0 - 1.0).abs() < 0.02, "white: {}", buf[2].0);
        // Mid-gray → should be close to srgb_u8_to_linear(128)
        let expected = srgb_u8_to_linear(128);
        assert!((buf[1].0 - expected).abs() < 0.05, "mid: {} vs {}", buf[1].0, expected);
    }
}
