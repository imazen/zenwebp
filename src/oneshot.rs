//! One-shot decode convenience functions.
//!
//! Each function takes raw WebP bytes and returns decoded pixels + dimensions.
//! For more control (dithering, upsampling, limits), use [`DecodeRequest`](crate::DecodeRequest).
//!
//! # Example
//!
//! ```rust,no_run
//! let webp_data: &[u8] = &[]; // your WebP data
//! let (pixels, width, height) = zenwebp::oneshot::decode_rgba(webp_data)?;
//! # Ok::<(), whereat::At<zenwebp::DecodeError>>(())
//! ```

pub use crate::decoder::{
    decode_argb, decode_argb_into, decode_argb_premultiplied, decode_bgr, decode_bgr_into,
    decode_bgra, decode_bgra_into, decode_bgra_premultiplied, decode_rgb, decode_rgb_into,
    decode_rgb565, decode_rgba, decode_rgba_into, decode_rgba_premultiplied, decode_rgba4444,
    decode_yuv420,
};
