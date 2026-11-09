#![forbid(unsafe_code)]
extern crate alloc;
#[cfg(test)]
#[path = "../../../src/codec/byte_stride.rs"]
mod input;

#[cfg(test)]
mod tests {
    use super::input::whole_pixel_rows;
    use alloc::{borrow::Cow, vec};
    use zenpixels::{PixelDescriptor as D, PixelSlice};
    #[test]
    fn native_rows_match_visible_samples_and_keep_representable_views_borrowed() {
        for desc in [D::RGB8_SRGB, D::RGBA8_SRGB, D::GRAY8_SRGB, D::GRAYA8_SRGB] {
            let row = 2 * desc.bytes_per_pixel();
            for padding in [0, 1, 2, 3, 4, 7] {
                let stride = row + padding;
                let mut data = vec![0xee; 2 * stride + row];
                for y in 0..3 {
                    for x in 0..row {
                        data[y * stride + x] = (y * row + x) as u8;
                    }
                }
                let view = PixelSlice::new(&data, 2, 3, stride, desc).unwrap();
                let (bytes, pixel_stride) = whole_pixel_rows(&view);
                for y in 0..3 {
                    let start = y * pixel_stride * desc.bytes_per_pixel();
                    assert_eq!(&bytes[start..start + row], view.row(y as u32));
                }
                if stride.is_multiple_of(desc.bytes_per_pixel()) {
                    assert!(matches!(bytes, Cow::Borrowed(_)));
                    assert_eq!(bytes.as_ptr(), data.as_ptr());
                } else {
                    assert!(matches!(bytes, Cow::Owned(_)));
                    assert_eq!(pixel_stride, 2);
                }
            }
        }
    }
}
