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

// ---------------------------------------------------------------------------
// Compile-time array splitting helpers
//
// These split fixed-size arrays into sub-array references without any runtime
// bounds checks. LLVM proves the lengths at compile time via split_first_chunk,
// eliminating the branch + panic path that `try_from(&slice[a..b]).unwrap()`
// would otherwise emit in complex SIMD functions.
// ---------------------------------------------------------------------------

/// Split `&[T; 16]` into four `&[T; 4]` references.
#[inline(always)]
pub(crate) fn q16<T>(a: &[T; 16]) -> (&[T; 4], &[T; 4], &[T; 4], &[T; 4]) {
    let (a0, rest) = a.split_first_chunk::<4>().unwrap();
    let (a1, rest) = rest.split_first_chunk::<4>().unwrap();
    let (a2, a3) = rest.split_first_chunk::<4>().unwrap();
    let a3: &[T; 4] = a3.try_into().unwrap();
    (a0, a1, a2, a3)
}

/// Split `&mut [T; 16]` into four `&mut [T; 4]` references.
#[inline(always)]
#[allow(dead_code)]
pub(crate) fn q16_mut<T>(a: &mut [T; 16]) -> (&mut [T; 4], &mut [T; 4], &mut [T; 4], &mut [T; 4]) {
    let (a0, rest) = a.split_first_chunk_mut::<4>().unwrap();
    let (a1, rest) = rest.split_first_chunk_mut::<4>().unwrap();
    let (a2, a3) = rest.split_first_chunk_mut::<4>().unwrap();
    let a3: &mut [T; 4] = a3.try_into().unwrap();
    (a0, a1, a2, a3)
}

/// Split `&[T; 16]` into two `&[T; 8]` references.
#[inline(always)]
pub(crate) fn h16<T>(a: &[T; 16]) -> (&[T; 8], &[T; 8]) {
    let (lo, hi) = a.split_first_chunk::<8>().unwrap();
    let hi: &[T; 8] = hi.try_into().unwrap();
    (lo, hi)
}

/// Split `&mut [T; 16]` into two `&mut [T; 8]` references.
#[inline(always)]
#[allow(dead_code)]
pub(crate) fn h16_mut<T>(a: &mut [T; 16]) -> (&mut [T; 8], &mut [T; 8]) {
    let (lo, hi) = a.split_first_chunk_mut::<8>().unwrap();
    let hi: &mut [T; 8] = hi.try_into().unwrap();
    (lo, hi)
}
