//! WebP decoder implementation

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
mod lossless_transform;
#[cfg(any(target_arch = "x86_64", target_arch = "x86"))]
mod lossless_transform_simd;
mod streaming;
pub(crate) mod yuv;
mod yuv_fused;

// VP8 diagnostic types and tree nodes (used by lossless decoder and tests)
pub(crate) mod vp8;

// VP8 lossy decoder (streaming cache→RGB architecture)
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

// Re-export DecoderContext for animation and buffer-reuse workflows
pub use vp8v2::{AnimationFrame, DecoderContext};

// Re-export diagnostic types for tests (hidden from public docs)
#[doc(hidden)]
pub use vp8::{BlockDiagnostic, DiagnosticFrame, MacroblockDiagnostic, TreeNode};

// Re-export common types used in diagnostics
#[doc(hidden)]
pub use crate::common::types::{ChromaMode, IntraMode, LumaMode};
