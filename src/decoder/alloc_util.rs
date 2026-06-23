//! Allocation helpers honoring an [`AllocPreference`] policy per call site.
//!
//! A WebP decode mixes two allocation regimes:
//!
//! * **Big, untrusted-sized buffers** — the full-image RGB/RGBA output buffer,
//!   the VP8 row cache (sized from the header macroblock dimensions), the
//!   lossless ARGB plane, the animation canvas. A malicious header can demand
//!   gigabytes, so we want a graceful `MemoryLimitExceeded` rather than an
//!   abort. These sites default to the *fallible* `try_reserve` path.
//! * **Small, bounded scratch** — per-row workspaces and fixed tables, bounded
//!   by the image width (not unbounded attacker control). A single `vec!`
//!   `calloc` is faster, so these sites default to the *infallible* path.
//!
//! [`AllocPreference`] is a **3-mode, per-site override** of that default:
//! [`Fallible`](AllocPreference::Fallible) / [`Infallible`](AllocPreference::Infallible)
//! force one path everywhere; [`CodecDefault`](AllocPreference::CodecDefault)
//! (and any future variant) keeps each site's own default. The helper
//! signatures therefore take the caller's preference *and* the site default,
//! and resolve them together.
//!
//! This module is deliberately decoupled from `zencodec` (an optional
//! dependency): it defines its own [`AllocPreference`] mirror that is always
//! available, and the `zencodec` adapter layer (`crate::codec`, gated on the
//! `zencodec` feature) converts `zencodec::AllocPreference` into this enum at
//! the decode boundary. The native decode API leaves the preference at
//! [`CodecDefault`](AllocPreference::CodecDefault), so behavior is unchanged.
//!
//! The helpers return `Result<_, AllocFailed>` — a pure allocation-failure
//! signal with no codec-error coupling — and each call site maps `AllocFailed`
//! to the error type appropriate for its layer (`InternalDecodeError` on the
//! hot decode paths, `DecodeError` at the API boundary).

use alloc::vec;
use alloc::vec::Vec;

/// Per-site allocation fallibility policy for the decoder.
///
/// A crate-local mirror of `zencodec::AllocPreference` so the decoder compiles
/// without the optional `zencodec` dependency. The `zencodec` adapter converts
/// the public enum into this one at the decode boundary.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Default)]
pub(crate) enum AllocPreference {
    /// Keep each allocation site's own default fallibility.
    #[default]
    CodecDefault,
    /// Force every site onto the fallible (`try_reserve`) path.
    Fallible,
    /// Force every site onto the infallible (`vec!`) path.
    Infallible,
}

/// Zero-sized marker returned when a fallible allocation fails. Each call site
/// maps this to the error type appropriate for its layer.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub(crate) struct AllocFailed;

/// Resolve the 3-mode [`AllocPreference`] against THIS site's default
/// fallibility.
///
/// * [`Fallible`](AllocPreference::Fallible) → always `true`.
/// * [`Infallible`](AllocPreference::Infallible) → always `false`.
/// * [`CodecDefault`](AllocPreference::CodecDefault) → the site default,
///   unchanged.
#[inline]
#[must_use]
pub(crate) fn resolve_fallible(pref: AllocPreference, site_default_fallible: bool) -> bool {
    match pref {
        AllocPreference::Fallible => true,
        AllocPreference::Infallible => false,
        AllocPreference::CodecDefault => site_default_fallible,
    }
}

/// Allocate `n` zeroed bytes, honoring the per-site fallibility.
///
/// * fallible → `try_reserve_exact` then zero-fill, returning [`AllocFailed`]
///   on allocation failure.
/// * infallible → `vec![0u8; n]` (single `calloc`, aborts on OOM).
#[inline]
pub(crate) fn alloc_zeroed(
    pref: AllocPreference,
    site_default_fallible: bool,
    n: usize,
) -> Result<Vec<u8>, AllocFailed> {
    if resolve_fallible(pref, site_default_fallible) {
        let mut v = Vec::new();
        v.try_reserve_exact(n).map_err(|_| AllocFailed)?;
        v.resize(n, 0);
        Ok(v)
    } else {
        Ok(vec![0u8; n])
    }
}

/// Resize an existing `Vec<u8>` to `n` bytes (zero-filling any growth),
/// honoring the per-site fallibility.
///
/// On the fallible path, capacity for any growth is reserved with
/// `try_reserve_exact` first (returning [`AllocFailed`] on failure) before the
/// infallible `resize` runs. On the infallible path this is a plain `resize`
/// (aborts on OOM). Reused buffers that already have enough capacity allocate
/// nothing on either path.
#[inline]
pub(crate) fn try_resize_zeroed(
    pref: AllocPreference,
    site_default_fallible: bool,
    buf: &mut Vec<u8>,
    n: usize,
) -> Result<(), AllocFailed> {
    if resolve_fallible(pref, site_default_fallible) && n > buf.len() {
        buf.try_reserve_exact(n - buf.len())
            .map_err(|_| AllocFailed)?;
    }
    buf.resize(n, 0);
    Ok(())
}

#[cfg(test)]
mod tests {
    use super::*;

    // `CodecDefault` keeps each site's own default fallibility.

    #[test]
    fn codec_default_keeps_site_default_true() {
        // Big-buffer site (default fallible): CodecDefault stays fallible.
        assert!(resolve_fallible(AllocPreference::CodecDefault, true));
    }

    #[test]
    fn codec_default_keeps_site_default_false() {
        // Small-scratch site (default infallible): CodecDefault stays infallible.
        assert!(!resolve_fallible(AllocPreference::CodecDefault, false));
    }

    #[test]
    fn explicit_fallible_overrides_any_site_default() {
        assert!(resolve_fallible(AllocPreference::Fallible, false));
        assert!(resolve_fallible(AllocPreference::Fallible, true));
    }

    #[test]
    fn explicit_infallible_overrides_any_site_default() {
        assert!(!resolve_fallible(AllocPreference::Infallible, true));
        assert!(!resolve_fallible(AllocPreference::Infallible, false));
    }

    #[test]
    fn default_is_codec_default() {
        assert_eq!(AllocPreference::default(), AllocPreference::CodecDefault);
    }

    #[test]
    fn alloc_zeroed_all_modes_equal_bytes() {
        let a = alloc_zeroed(AllocPreference::CodecDefault, true, 4096).unwrap();
        let b = alloc_zeroed(AllocPreference::Infallible, true, 4096).unwrap();
        let c = alloc_zeroed(AllocPreference::Fallible, false, 4096).unwrap();
        assert_eq!(a.len(), 4096);
        assert_eq!(a, b);
        assert_eq!(a, c);
        assert!(a.iter().all(|&x| x == 0));
    }

    #[test]
    fn try_resize_grows_and_zero_fills() {
        let mut v = vec![1u8, 2, 3];
        try_resize_zeroed(AllocPreference::Fallible, true, &mut v, 6).unwrap();
        assert_eq!(v, [1, 2, 3, 0, 0, 0]);
        // Shrinking never allocates and stays correct on every mode.
        try_resize_zeroed(AllocPreference::Infallible, true, &mut v, 2).unwrap();
        assert_eq!(v, [1, 2]);
    }

    #[test]
    fn try_resize_all_modes_equal_bytes() {
        let mut a = Vec::new();
        let mut b = Vec::new();
        let mut c = Vec::new();
        try_resize_zeroed(AllocPreference::CodecDefault, true, &mut a, 1000).unwrap();
        try_resize_zeroed(AllocPreference::Infallible, true, &mut b, 1000).unwrap();
        try_resize_zeroed(AllocPreference::Fallible, false, &mut c, 1000).unwrap();
        assert_eq!(a, b);
        assert_eq!(a, c);
        assert_eq!(a.len(), 1000);
        assert!(a.iter().all(|&x| x == 0));
    }

    #[test]
    fn alloc_zeroed_fallible_oom_returns_err() {
        // Request an impossibly large allocation; the fallible path must
        // return Err rather than abort.
        let r = alloc_zeroed(AllocPreference::Fallible, true, usize::MAX / 2);
        assert!(r.is_err());
    }

    #[test]
    fn try_resize_fallible_oom_returns_err() {
        let mut v = Vec::new();
        let r = try_resize_zeroed(AllocPreference::Fallible, true, &mut v, usize::MAX / 2);
        assert!(r.is_err());
    }
}
