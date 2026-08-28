//! VP8L (Lossless WebP) encoder module.
//!
//! Implements the VP8L lossless compression format as specified in:
//! <https://developers.google.com/speed/webp/docs/webp_lossless_bitstream_specification>

// Many infrastructure components are in place for future optimization work
#![allow(dead_code)]

mod backward_refs;
mod bitwriter;
mod color_cache;
mod cost_model;
mod encode;
mod entropy;
mod hash_chain;
mod histogram;
mod huffman;
mod meta_huffman;
mod near_lossless;
/// Public under `_dev` so `benches/kernel_tiers.rs` can measure the VP8L
/// transforms per-tier. Not public API, not semver-covered.
#[cfg(feature = "_dev")]
pub mod transforms;
#[cfg(not(feature = "_dev"))]
mod transforms;
mod types;

pub use encode::encode_vp8l;
pub(crate) use encode::encode_vp8l_with_alloc;
pub use entropy::print_entropy_stats;
pub use meta_huffman::print_clustering_stats;
pub use types::{Vp8lConfig, Vp8lQuality};
