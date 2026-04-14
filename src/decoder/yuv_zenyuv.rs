//! Drop-in replacement for `convert_image_yuv` and `convert_image_sharp_yuv`
//! using the `zenyuv` crate instead of scalar Rust or the `yuv` crate.
//!
//! Produces MB-aligned (16×16 Y, 8×8 UV) planes with Limited range BT.601,
//! matching the VP8 encoder's expected input format.

use alloc::vec;
use alloc::vec::Vec;

/// Convert RGB8 image to YUV420 via zenyuv (Limited range BT.601).
/// Returns MB-aligned (Y, U, V) planes.
pub(crate) fn convert_rgb_yuv420(
    image_data: &[u8],
    width: u16,
    height: u16,
    stride: usize,
) -> (Vec<u8>, Vec<u8>, Vec<u8>) {
    convert_yuv420_impl::<3>(image_data, width, height, stride, false)
}

/// Convert RGBA8 image to YUV420 via zenyuv.
pub(crate) fn convert_rgba_yuv420(
    image_data: &[u8],
    width: u16,
    height: u16,
    stride: usize,
) -> (Vec<u8>, Vec<u8>, Vec<u8>) {
    convert_yuv420_impl::<4>(image_data, width, height, stride, false)
}

/// Convert RGB8 image to YUV420 with sharp YUV via zenyuv.
pub(crate) fn convert_rgb_sharp_yuv420(
    image_data: &[u8],
    width: u16,
    height: u16,
    stride: usize,
) -> (Vec<u8>, Vec<u8>, Vec<u8>) {
    convert_yuv420_impl::<3>(image_data, width, height, stride, true)
}

/// Convert RGBA8 image to YUV420 with sharp YUV via zenyuv.
pub(crate) fn convert_rgba_sharp_yuv420(
    image_data: &[u8],
    width: u16,
    height: u16,
    stride: usize,
) -> (Vec<u8>, Vec<u8>, Vec<u8>) {
    convert_yuv420_impl::<4>(image_data, width, height, stride, true)
}

fn convert_yuv420_impl<const BPP: usize>(
    image_data: &[u8],
    width: u16,
    height: u16,
    stride: usize,
    sharp: bool,
) -> (Vec<u8>, Vec<u8>, Vec<u8>) {
    let w = usize::from(width);
    let h = usize::from(height);
    let mb_w = w.div_ceil(16);
    let mb_h = h.div_ceil(16);
    let luma_w = 16 * mb_w;
    let luma_h = 16 * mb_h;
    let chroma_w = 8 * mb_w;
    let chroma_h = 8 * mb_h;

    // Strip to contiguous RGB if stride != width or BPP != 3.
    let rgb_contiguous: Vec<u8>;
    let rgb: &[u8] = if BPP == 3 && stride == w {
        &image_data[..w * h * 3]
    } else {
        rgb_contiguous = (0..h)
            .flat_map(|y| {
                let row_start = y * stride * BPP;
                (0..w).flat_map(move |x| {
                    let i = row_start + x * BPP;
                    [image_data[i], image_data[i + 1], image_data[i + 2]]
                })
            })
            .collect();
        &rgb_contiguous
    };

    // Produce tight-stride YUV via zenyuv.
    let cw = w.div_ceil(2);
    let ch = h.div_ceil(2);
    let mut y_tight = vec![0u8; w * h];
    let mut u_tight = vec![0u8; cw * ch];
    let mut v_tight = vec![0u8; cw * ch];

    if sharp {
        let mut ctx = zenyuv::YuvContext::new(zenyuv::Range::Limited, zenyuv::Matrix::Bt601);
        let config = zenyuv::SharpYuvConfig {
            ..Default::default()
        };
        ctx.encode_sharp_420_u8(
            rgb, &mut y_tight, &mut u_tight, &mut v_tight, w, h, &config,
        );
    } else {
        let mut ctx = zenyuv::YuvContext::new(zenyuv::Range::Limited, zenyuv::Matrix::Bt601);
        ctx.encode_420_u8(rgb, &mut y_tight, &mut u_tight, &mut v_tight, w, h);
    }

    // Copy into MB-aligned planes (zero-padded beyond image area).
    let mut y_out = vec![0u8; luma_w * luma_h];
    let mut u_out = vec![0u8; chroma_w * chroma_h];
    let mut v_out = vec![0u8; chroma_w * chroma_h];

    for row in 0..h {
        y_out[row * luma_w..row * luma_w + w].copy_from_slice(&y_tight[row * w..row * w + w]);
    }
    for row in 0..ch {
        u_out[row * chroma_w..row * chroma_w + cw]
            .copy_from_slice(&u_tight[row * cw..row * cw + cw]);
        v_out[row * chroma_w..row * chroma_w + cw]
            .copy_from_slice(&v_tight[row * cw..row * cw + cw]);
    }

    (y_out, u_out, v_out)
}
