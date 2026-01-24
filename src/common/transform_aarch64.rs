//! AArch64 NEON intrinsics for DCT/IDCT transforms.
//!
//! Note: NEON SIMD implementation has known bugs in transpose/shuffle logic.
//! Currently falls back to scalar. TODO: Fix NEON shuffle order to match x86.
//!
//! Safe intrinsics via archmage's #[arcane] and safe_unaligned_simd.

#[cfg(target_arch = "aarch64")]
use archmage::NeonToken;

// =============================================================================
// Public dispatch functions
// =============================================================================

/// Forward DCT using NEON (currently uses scalar fallback due to NEON bugs)
#[cfg(target_arch = "aarch64")]
pub(crate) fn dct4x4_neon(_token: NeonToken, block: &mut [i32; 16]) {
    // TODO: Fix NEON SIMD implementation - transpose/shuffle logic differs from x86
    crate::transform::dct4x4_scalar(block);
}

/// Inverse DCT using NEON (currently uses scalar fallback due to NEON bugs)
#[cfg(target_arch = "aarch64")]
pub(crate) fn idct4x4_neon(_token: NeonToken, block: &mut [i32]) {
    debug_assert!(block.len() >= 16);
    // TODO: Fix NEON SIMD implementation - transpose/shuffle logic differs from x86
    crate::transform::idct4x4_scalar(block);
}
