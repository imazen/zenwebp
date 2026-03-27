//! Common types and utilities shared between encoder and decoder

pub mod prediction;
/// DCT/IDCT transform functions
pub mod transform;
pub mod types;

#[cfg(any(target_arch = "x86_64", target_arch = "x86"))]
pub mod simd_sse;

#[cfg(target_arch = "aarch64")]
pub mod simd_neon;

#[cfg(target_arch = "wasm32")]
pub mod simd_wasm;
