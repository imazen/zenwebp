//! Stable garb byte operations used by the native codec and its adapter.
//!
//! The pinned analyzer's legacy converter enables garb's experimental feature.
//! At this garb revision that marks the whole crate deprecated. Keep the
//! allowance on these stable byte calls until the analyzer migrates.

#[allow(deprecated)]
pub(crate) type SizeError = garb::SizeError;

macro_rules! swizzles {
    ($signature:tt; $($name:ident),+ $(,)?) => {
        $(swizzles!(@one $name $signature);)+
    };
    (@one $name:ident ($($arg:ident : $ty:ty),+)) => {
        #[inline]
        #[allow(deprecated)]
        pub(crate) fn $name($($arg: $ty),+) -> Result<(), SizeError> {
            garb::bytes::$name($($arg),+)
        }
    };
}

swizzles!((buf: &mut [u8]); rgba_to_argb_inplace, rgba_to_bgra_inplace);
swizzles!(
    (src: &[u8], dst: &mut [u8]);
    argb_to_rgba,
    bgr_to_rgba,
    bgra_to_rgb,
    bgra_to_rgba,
    gray_alpha_to_rgba,
    gray_to_rgba,
    rgb_to_bgra,
    rgb_to_rgba,
    rgba_to_bgr,
    rgba_to_bgra,
    rgba_to_rgb,
);
swizzles!(
    (src: &[u8], dst: &mut [u8], width: usize, height: usize, src_stride: usize, dst_stride: usize);
    bgr_to_rgb_strided,
    bgra_to_rgb_strided,
    rgba_to_argb_strided,
    rgba_to_bgr_strided,
    rgba_to_bgra_strided,
    rgba_to_rgb_strided,
);
