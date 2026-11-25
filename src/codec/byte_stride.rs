//! Bridge byte-strided views to the native encoder's whole-pixel stride.
use alloc::borrow::Cow;
use zenpixels::PixelView;

pub(super) fn whole_pixel_rows<'a>(
    pixels: &PixelView<'a>,
) -> Result<(Cow<'a, [u8]>, usize), whereat::At<zenpixels::BufferError>> {
    let pixel_size = pixels.row_bytes() / pixels.width() as usize;
    if pixels.stride().is_multiple_of(pixel_size) {
        Ok((
            Cow::Borrowed(pixels.as_strided_bytes()),
            pixels.stride() / pixel_size,
        ))
    } else {
        Ok((pixels.contiguous_bytes()?, pixels.width() as usize))
    }
}
