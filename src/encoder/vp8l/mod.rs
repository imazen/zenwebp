//! VP8L (Lossless WebP) encoder module.
//!
//! Implements the VP8L lossless compression format as specified in:
//! <https://developers.google.com/speed/webp/docs/webp_lossless_bitstream_specification>

// Many infrastructure components are in place for future optimization work
#![allow(dead_code)]

mod backward_refs;
mod bitwriter;
mod color_cache;
mod encode;
mod entropy;
mod hash_chain;
mod histogram;
mod huffman;
mod meta_huffman;
mod transforms;
mod types;

pub use encode::encode_vp8l;
pub use types::{Vp8lQuality, Vp8lConfig};
