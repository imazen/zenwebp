//! Bridge byte-strided views to the native encoder's whole-pixel stride.
use alloc::borrow::Cow;
use zenpixels::PixelSlice;

pub(super) fn whole_pixel_rows<'a>(pixels: &'a PixelSlice<'_>) -> (Cow<'a, [u8]>, usize) {
    let pixel_size = pixels.descriptor().bytes_per_pixel();
    if pixels.stride().is_multiple_of(pixel_size) {
        (
            Cow::Borrowed(pixels.as_strided_bytes()),
            pixels.stride() / pixel_size,
        )
    } else {
        (pixels.contiguous_bytes(), pixels.width() as usize)
    }
}
