//! Common types and utilities shared between encoder and decoder

pub mod prediction;
/// DCT/IDCT transform functions
pub mod transform;
pub mod types;

#[cfg(all(feature = "simd", any(target_arch = "x86_64", target_arch = "x86")))]
pub mod simd_sse;

#[cfg(all(feature = "simd", target_arch = "aarch64"))]
pub mod simd_neon;

#[cfg(all(feature = "simd", target_arch = "aarch64"))]
pub mod transform_aarch64;

#[cfg(feature = "simd")]
pub mod transform_simd_intrinsics;

#[cfg(all(feature = "simd", target_arch = "wasm32"))]
pub mod simd_wasm;

#[cfg(all(feature = "simd", target_arch = "wasm32"))]
pub mod transform_wasm;
