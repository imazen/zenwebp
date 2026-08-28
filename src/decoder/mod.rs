//! WebP decoder implementation

pub(crate) mod alloc_util;
mod alpha_blending;
mod api;
pub(crate) mod arithmetic;
mod bit_reader;
mod dither;
pub(crate) mod extended;
mod huffman;
mod internal_error;
mod limits;
mod loop_filter;
mod lossless;
/// Dev-only public exposure for per-tier benchmarking. NOT public API.
///
/// `missing_docs` is allowed here because the gate surfaces pre-existing
/// internal items that were never part of the documented surface; documenting
/// them would imply a stability they do not have.
#[cfg(feature = "_dev")]
#[allow(missing_docs)]
pub mod lossless_transform;
#[cfg(not(feature = "_dev"))]
mod lossless_transform;
#[cfg(any(
    target_arch = "x86_64",
    target_arch = "x86",
    target_arch = "aarch64",
    target_arch = "wasm32"
))]
mod lossless_transform_simd;
mod streaming;
pub(crate) mod yuv;
mod yuv_fused;

// VP8 lossy decoder (streaming cache→RGB architecture)
#[cfg(feature = "_dev")]
pub mod vp8v2;
#[cfg(not(feature = "_dev"))]
pub(crate) mod vp8v2;

// Re-export public API
pub use api::{
    BitstreamFormat, DecodeConfig, DecodeError, DecodeRequest, DecodeResult, ImageInfo, LoopCount,
    UpsamplingMethod, WebPDecoder, YuvPlanes, decode_argb, decode_argb_into,
    decode_argb_premultiplied, decode_bgr, decode_bgr_into, decode_bgra, decode_bgra_into,
    decode_bgra_premultiplied, decode_rgb, decode_rgb_into, decode_rgb565, decode_rgba,
    decode_rgba_into, decode_rgba_premultiplied, decode_rgba4444, decode_yuv420,
};
#[allow(deprecated)]
pub use limits::Limits;
pub use streaming::{StreamStatus, StreamingDecoder};

// Re-export common types used in diagnostics
#[doc(hidden)]
pub use crate::common::types::{ChromaMode, IntraMode, LumaMode};

/// Level-0 (main) image coding parameters of a VP8L stream (#71 diagnostic).
#[cfg(feature = "mode_debug")]
#[derive(Debug, Clone, Default)]
pub struct Vp8lMainImageInfo {
    /// Color cache bits (`None` = no cache).
    pub cache_bits: Option<u8>,
    /// Meta-Huffman tile bits (0 = single group).
    pub histo_bits: u8,
    /// Number of Huffman groups.
    pub num_groups: u32,
    /// Bits spent on the meta-Huffman flag + entropy image.
    pub entropy_image_bits: u64,
    /// Bits spent on the Huffman code tables (all groups).
    pub huffman_tables_bits: u64,
    /// Bits spent on the pixel data itself.
    pub pixel_data_bits: u64,
    /// Main-image token counts: `[literals, cache_hits, copies, copied_pixels]`.
    pub tokens: [u64; 4],
}

/// `mode_debug` diagnostic for #71: decode a simple (non-animated)
/// lossless WebP and return, per transform,
/// `(kind, size_bits, bits_on_wire, decoded_data)`, the bit offset where
/// the main image starts, the VP8L payload length in bits, and the main
/// image's coding parameters. `kind`: 0 predictor, 1 color, 2
/// subtract-green, 3 color-indexing; predictor modes are the green byte of
/// each 4-byte tile entry.
#[cfg(feature = "mode_debug")]
#[allow(clippy::type_complexity)]
pub fn vp8l_transform_dump(
    webp: &[u8],
) -> Result<
    (
        alloc::vec::Vec<(u8, u8, u64, alloc::vec::Vec<u8>)>,
        u64,
        u64,
        Vp8lMainImageInfo,
    ),
    alloc::string::String,
> {
    let demux = crate::mux::WebPDemuxer::new(webp).map_err(|e| alloc::format!("{e:?}"))?;
    let frame = demux.frame(1).ok_or("no frame")?;
    if frame.is_lossy {
        return Err("not a lossless frame".into());
    }
    let payload = frame.bitstream;
    let (dumps, main_start, main) =
        lossless::LosslessDecoder::debug_transforms(payload, frame.width, frame.height)
            .map_err(|e| alloc::format!("{e:?}"))?;
    let total_bits = payload.len() as u64 * 8;
    let info = Vp8lMainImageInfo {
        cache_bits: main.cache_bits,
        histo_bits: main.histo_bits,
        num_groups: main.num_groups,
        entropy_image_bits: main.entropy_image_bits,
        huffman_tables_bits: main.pixel_data_start - main_start - main.entropy_image_bits,
        pixel_data_bits: total_bits - main.pixel_data_start,
        tokens: main.tokens,
    };
    Ok((
        dumps
            .into_iter()
            .map(|d| (d.kind, d.size_bits, d.bits, d.data))
            .collect(),
        main_start,
        total_bits,
        info,
    ))
}
