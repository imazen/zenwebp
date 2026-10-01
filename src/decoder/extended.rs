use alloc::vec::Vec;

#[allow(unused_imports)]
use whereat::at;

use super::alpha_blending::do_alpha_blending;
use super::api::DecodeError;
use super::lossless::LosslessDecoder;
use crate::slice_reader::SliceReader;

#[derive(Debug, Clone)]
pub(crate) struct WebPExtendedInfo {
    pub(crate) alpha: bool,

    pub(crate) canvas_width: u32,
    pub(crate) canvas_height: u32,

    #[allow(unused)]
    pub(crate) icc_profile: bool,
    pub(crate) exif_metadata: bool,
    pub(crate) xmp_metadata: bool,
    pub(crate) animation: bool,

    pub(crate) background_color: Option<[u8; 4]>,
    pub(crate) background_color_hint: [u8; 4],
}

/// Rectangle and disposal state of the previously composited animation frame.
#[derive(Debug, Clone, Copy)]
pub(crate) struct PrevFrame {
    pub(crate) x: u32,
    pub(crate) y: u32,
    pub(crate) width: u32,
    pub(crate) height: u32,
    /// The previous frame asked to be disposed to the background.
    pub(crate) dispose_to_background: bool,
    /// The previous frame was composited as a keyframe.
    pub(crate) was_keyframe: bool,
}

/// Composites a frame onto a canvas, mirroring libwebp's `WebPAnimDecoderGetNext`
/// (`src/demux/anim_decode.c`) so animated output is byte-identical to libwebp.
///
/// The model, in libwebp's terms:
/// 1. A frame is a *keyframe* when it is the first frame, when it is full-canvas
///    and either declares no alpha or does not blend, or when the previous frame
///    was disposed to the background and was itself full-canvas or a keyframe.
///    A keyframe starts from a canvas filled with `clear_color`.
/// 2. Otherwise the canvas is the previous canvas with the previous frame's
///    rectangle filled with `clear_color` if that frame was disposed.
/// 3. The frame's pixels are written raw. When blending (and not a keyframe),
///    pixels with alpha < 255 are then blended against the pre-frame canvas —
///    except inside a disposed previous rectangle, where libwebp treats the
///    blend against transparent as a no-op and keeps the raw pixel.
///
/// `clear_color` is transparent black unless the caller overrode the background
/// (libwebp always uses transparent). With an opaque/colored override, the
/// "blend against cleared pixel is a no-op" shortcut no longer holds, so those
/// pixels are blended against the clear color instead.
///
/// `frame` is RGB (`frame_bpp == 3`, implicitly opaque) or RGBA (`4`).
/// `frame_declares_alpha` is the container/bitstream alpha flag libwebp's demuxer
/// reports (ALPH chunk for VP8, the header bit for VP8L), used only for the
/// keyframe decision. Returns whether the frame was a keyframe.
#[allow(clippy::too_many_arguments)]
pub(crate) fn composite_frame(
    canvas: &mut [u8],
    canvas_width: u32,
    canvas_height: u32,
    clear_color: [u8; 4],
    frame: &[u8],
    frame_bpp: usize,
    frame_offset_x: u32,
    frame_offset_y: u32,
    frame_width: u32,
    frame_height: u32,
    frame_declares_alpha: bool,
    frame_use_alpha_blending: bool,
    prev: Option<PrevFrame>,
) -> Result<bool, whereat::At<DecodeError>> {
    // Validate canvas size with checked arithmetic
    let canvas_stride = (canvas_width as usize)
        .checked_mul(4)
        .ok_or_else(|| at!(DecodeError::ImageTooLarge))?;
    let expected_canvas_size = canvas_stride
        .checked_mul(canvas_height as usize)
        .ok_or_else(|| at!(DecodeError::ImageTooLarge))?;
    if canvas.len() < expected_canvas_size
        || frame_offset_x.saturating_add(frame_width) > canvas_width
        || frame_offset_y.saturating_add(frame_height) > canvas_height
    {
        return Err(at!(DecodeError::ImageTooLarge));
    }
    let frame_stride = (frame_width as usize) * frame_bpp;
    if (frame_bpp != 3 && frame_bpp != 4) || frame.len() < frame_stride * frame_height as usize {
        return Err(at!(DecodeError::ImageTooLarge));
    }

    let is_full = |w: u32, h: u32| w == canvas_width && h == canvas_height;
    let is_keyframe = match prev {
        None => true,
        Some(p) => {
            ((!frame_declares_alpha || !frame_use_alpha_blending)
                && is_full(frame_width, frame_height))
                || (p.dispose_to_background && (is_full(p.width, p.height) || p.was_keyframe))
        }
    };

    if is_keyframe {
        canvas[..expected_canvas_size]
            .as_chunks_mut::<4>()
            .0
            .fill(clear_color);
    } else if let Some(p) = prev
        && p.dispose_to_background
    {
        for y in p.y..p.y + p.height {
            let row = y as usize * canvas_stride + p.x as usize * 4;
            canvas[row..row + p.width as usize * 4]
                .as_chunks_mut::<4>()
                .0
                .fill(clear_color);
        }
    }

    let clear_is_transparent = clear_color == [0, 0, 0, 0];
    // Region whose pre-frame pixels are known to be `clear_color`.
    let cleared = |x: u32, y: u32| -> bool {
        if is_keyframe {
            return true;
        }
        match prev {
            Some(p) if p.dispose_to_background => {
                x >= p.x && x < p.x + p.width && y >= p.y && y < p.y + p.height
            }
            _ => false,
        }
    };

    for fy in 0..frame_height {
        let cy = frame_offset_y + fy;
        let src_row = &frame[fy as usize * frame_stride..][..frame_stride];
        let dst_row = &mut canvas[cy as usize * canvas_stride + frame_offset_x as usize * 4..]
            [..frame_width as usize * 4];
        for (fx, dst) in dst_row.as_chunks_mut::<4>().0.iter_mut().enumerate() {
            let s = &src_row[fx * frame_bpp..];
            let src = [s[0], s[1], s[2], if frame_bpp == 4 { s[3] } else { 255 }];
            *dst = if !frame_use_alpha_blending || src[3] == 255 {
                src
            } else if cleared(frame_offset_x + fx as u32, cy) {
                if clear_is_transparent {
                    src
                } else {
                    do_alpha_blending(src, clear_color)
                }
            } else {
                do_alpha_blending(src, *dst)
            };
        }
    }

    Ok(is_keyframe)
}

pub(crate) fn get_alpha_predictor(
    x: usize,
    y: usize,
    width: usize,
    filtering_method: FilteringMethod,
    image_slice: &[u8],
) -> u8 {
    match filtering_method {
        FilteringMethod::None => 0,
        FilteringMethod::Horizontal => {
            if x == 0 && y == 0 {
                0
            } else if x == 0 {
                let index = (y - 1) * width + x;
                image_slice[index * 4 + 3]
            } else {
                let index = y * width + x - 1;
                image_slice[index * 4 + 3]
            }
        }
        FilteringMethod::Vertical => {
            if x == 0 && y == 0 {
                0
            } else if y == 0 {
                let index = y * width + x - 1;
                image_slice[index * 4 + 3]
            } else {
                let index = (y - 1) * width + x;
                image_slice[index * 4 + 3]
            }
        }
        FilteringMethod::Gradient => {
            let (left, top, top_left) = match (x, y) {
                (0, 0) => (0, 0, 0),
                (0, y) => {
                    let above_index = (y - 1) * width + x;
                    let val = image_slice[above_index * 4 + 3];
                    (val, val, val)
                }
                (x, 0) => {
                    let before_index = y * width + x - 1;
                    let val = image_slice[before_index * 4 + 3];
                    (val, val, val)
                }
                (x, y) => {
                    let left_index = y * width + x - 1;
                    let left = image_slice[left_index * 4 + 3];
                    let top_index = (y - 1) * width + x;
                    let top = image_slice[top_index * 4 + 3];
                    let top_left_index = (y - 1) * width + x - 1;
                    let top_left = image_slice[top_left_index * 4 + 3];

                    (left, top, top_left)
                }
            };

            let combination = i16::from(left) + i16::from(top) - i16::from(top_left);
            i16::clamp(combination, 0, 255).try_into().unwrap()
        }
    }
}

/// Same as [`get_alpha_predictor`] but reads from a standalone alpha plane
/// (one byte per pixel) instead of an RGBA interleaved buffer.
#[allow(dead_code)]
pub(crate) fn get_alpha_predictor_from_alpha(
    x: usize,
    y: usize,
    width: usize,
    filtering_method: FilteringMethod,
    alpha_plane: &[u8],
) -> u8 {
    match filtering_method {
        FilteringMethod::None => 0,
        FilteringMethod::Horizontal => {
            if x == 0 && y == 0 {
                0
            } else if x == 0 {
                alpha_plane[(y - 1) * width + x]
            } else {
                alpha_plane[y * width + x - 1]
            }
        }
        FilteringMethod::Vertical => {
            if x == 0 && y == 0 {
                0
            } else if y == 0 {
                alpha_plane[y * width + x - 1]
            } else {
                alpha_plane[(y - 1) * width + x]
            }
        }
        FilteringMethod::Gradient => {
            let (left, top, top_left) = match (x, y) {
                (0, 0) => (0, 0, 0),
                (0, y) => {
                    let val = alpha_plane[(y - 1) * width + x];
                    (val, val, val)
                }
                (x, 0) => {
                    let val = alpha_plane[y * width + x - 1];
                    (val, val, val)
                }
                (x, y) => {
                    let left = alpha_plane[y * width + x - 1];
                    let top = alpha_plane[(y - 1) * width + x];
                    let top_left = alpha_plane[(y - 1) * width + x - 1];
                    (left, top, top_left)
                }
            };

            let combination = i16::from(left) + i16::from(top) - i16::from(top_left);
            i16::clamp(combination, 0, 255).try_into().unwrap()
        }
    }
}

pub(crate) fn read_extended_header(
    reader: &mut SliceReader,
) -> Result<WebPExtendedInfo, whereat::At<DecodeError>> {
    let chunk_flags = reader.read_u8()?;

    let icc_profile = chunk_flags & 0b00100000 != 0;
    let alpha = chunk_flags & 0b00010000 != 0;
    let exif_metadata = chunk_flags & 0b00001000 != 0;
    let xmp_metadata = chunk_flags & 0b00000100 != 0;
    let animation = chunk_flags & 0b00000010 != 0;

    // reserved bytes are ignored
    let _reserved_bytes = read_3_bytes(reader)?;

    let canvas_width = read_3_bytes(reader)? + 1;
    let canvas_height = read_3_bytes(reader)? + 1;

    //product of canvas dimensions cannot be larger than u32 max
    if u32::checked_mul(canvas_width, canvas_height).is_none() {
        return Err(at!(DecodeError::ImageTooLarge));
    }

    let info = WebPExtendedInfo {
        icc_profile,
        alpha,
        exif_metadata,
        xmp_metadata,
        animation,
        canvas_width,
        canvas_height,
        background_color_hint: [0; 4],
        background_color: None,
    };

    Ok(info)
}

pub(crate) fn read_3_bytes(reader: &mut SliceReader) -> Result<u32, DecodeError> {
    let mut buffer: [u8; 3] = [0; 3];
    reader.read_exact(&mut buffer)?;
    let value: u32 =
        (u32::from(buffer[2]) << 16) | (u32::from(buffer[1]) << 8) | u32::from(buffer[0]);
    Ok(value)
}

#[derive(Debug)]
pub(crate) struct AlphaChunk {
    _preprocessing: bool,
    pub(crate) filtering_method: FilteringMethod,
    pub(crate) data: Vec<u8>,
}

#[derive(Debug, Copy, Clone)]
pub(crate) enum FilteringMethod {
    None,
    Horizontal,
    Vertical,
    Gradient,
}

pub(crate) fn read_alpha_chunk(
    data: &[u8],
    width: u16,
    height: u16,
    limits: &super::limits::Limits,
) -> Result<AlphaChunk, whereat::At<DecodeError>> {
    if data.is_empty() {
        return Err(at!(DecodeError::BitStreamError));
    }
    let info_byte = data[0];

    let preprocessing = (info_byte & 0b00110000) >> 4;
    let filtering = (info_byte & 0b00001100) >> 2;
    let compression = info_byte & 0b00000011;

    let preprocessing = match preprocessing {
        0 => false,
        1 => true,
        _ => return Err(at!(DecodeError::InvalidAlphaPreprocessing)),
    };

    let filtering_method = match filtering {
        0 => FilteringMethod::None,
        1 => FilteringMethod::Horizontal,
        2 => FilteringMethod::Vertical,
        3 => FilteringMethod::Gradient,
        _ => unreachable!(),
    };

    let lossless_compression = match compression {
        0 => false,
        1 => true,
        _ => return Err(at!(DecodeError::InvalidCompressionMethod)),
    };

    let alpha_data = &data[1..];
    let decoded_data = if lossless_compression {
        let rgba_size = usize::from(width) * usize::from(height) * 4;
        limits.check_memory(rgba_size)?;

        let mut decoder = LosslessDecoder::new(alpha_data);
        decoder.set_limits(Some(limits));

        // Alpha-as-VP8L scratch + extracted green plane are sized from the
        // (untrusted) header dimensions → fallible by default.
        let mut rgba_data =
            crate::decoder::alloc_util::alloc_zeroed(limits.alloc_pref, true, rgba_size)
                .map_err(|_| at!(DecodeError::MemoryLimitExceeded))?;
        decoder.decode_frame(u32::from(width), u32::from(height), true, &mut rgba_data)?;

        let green_size = usize::from(width) * usize::from(height);
        limits.check_memory(green_size)?;
        let mut green =
            crate::decoder::alloc_util::alloc_zeroed(limits.alloc_pref, true, green_size)
                .map_err(|_| at!(DecodeError::MemoryLimitExceeded))?;
        for (rgba_val, green_val) in rgba_data.as_chunks::<4>().0.iter().zip(green.iter_mut()) {
            *green_val = rgba_val[1];
        }
        green
    } else {
        let required = width as usize * height as usize;
        if alpha_data.len() < required {
            return Err(at!(DecodeError::BitStreamError));
        }
        alpha_data[..required].to_vec()
    };

    let chunk = AlphaChunk {
        _preprocessing: preprocessing,
        filtering_method,
        data: decoded_data,
    };

    Ok(chunk)
}

#[cfg(test)]
mod tests {
    use super::*;
    // `vec!` is not in the `no_std` prelude; the glob above re-imports `Vec`
    // (a type) but not the macro, so bring it in explicitly for the test bodies.
    use alloc::vec;

    /// Before the fix, clearing the canvas for a non-alpha frame used 3-byte
    /// stride on the always-RGBA canvas, corrupting pixel alignment.
    #[test]
    fn dispose_clear_fullsize_rgb_frame() {
        let w = 4u32;
        let h = 4u32;
        // Canvas pre-filled with a recognizable pattern (0xAA in every byte).
        let mut canvas = vec![0xAA_u8; (w * h * 4) as usize];

        // RGB frame (no alpha) — white pixels.
        let frame = vec![0xFF_u8; (w * h * 3) as usize];

        // Dispose previous full-size frame to transparent black, then composite
        // the RGB frame.  `frame_use_alpha_blending = true` prevents the
        // fast-path early return so the clear code actually runs.
        composite_frame(
            &mut canvas,
            w,
            h,
            [0, 0, 0, 0],
            &frame,
            3,
            0,
            0,
            w,
            h,
            false, // frame_declares_alpha
            true,  // frame_use_alpha_blending
            Some(PrevFrame {
                x: 0,
                y: 0,
                width: w,
                height: h,
                dispose_to_background: true,
                was_keyframe: false,
            }),
        )
        .unwrap();

        // After clear + RGB→RGBA composite, every pixel should be opaque white.
        for (i, pixel) in canvas.as_chunks::<4>().0.iter().enumerate() {
            assert_eq!(
                *pixel,
                [0xFF, 0xFF, 0xFF, 0xFF],
                "pixel {i} corrupted: {pixel:?}"
            );
        }
    }

    /// Same bug for sub-frame clear: 3-byte stride wrote to wrong canvas offsets.
    #[test]
    fn dispose_clear_subframe_rgb_frame() {
        let canvas_w = 8u32;
        let canvas_h = 8u32;
        let mut canvas = vec![0xAA_u8; (canvas_w * canvas_h * 4) as usize];

        // Previous frame occupied (2,2)-(6,6) — a 4×4 sub-frame.
        let prev_x = 2u32;
        let prev_y = 2u32;
        let prev_w = 4u32;
        let prev_h = 4u32;

        // New frame: 4×4 RGB at (0,0).
        let frame_w = 4u32;
        let frame_h = 4u32;
        let frame = vec![0xFF_u8; (frame_w * frame_h * 3) as usize];

        composite_frame(
            &mut canvas,
            canvas_w,
            canvas_h,
            [0, 0, 0, 0],
            &frame,
            3,
            0,
            0,
            frame_w,
            frame_h,
            false,
            true,
            Some(PrevFrame {
                x: prev_x,
                y: prev_y,
                width: prev_w,
                height: prev_h,
                dispose_to_background: true,
                was_keyframe: false,
            }),
        )
        .unwrap();

        let stride = canvas_w as usize * 4;

        // The previous-frame rectangle (2,2)-(6,6) should be cleared to [0,0,0,0].
        for y in prev_y as usize..(prev_y + prev_h) as usize {
            for x in prev_x as usize..(prev_x + prev_w) as usize {
                let idx = y * stride + x * 4;
                let pixel = &canvas[idx..idx + 4];
                // Pixels inside new frame (0..4, 0..4) get overwritten by composite,
                // so only check the region that is cleared but NOT overwritten.
                if x >= frame_w as usize || y >= frame_h as usize {
                    assert_eq!(
                        pixel,
                        [0, 0, 0, 0],
                        "prev-frame pixel ({x},{y}) not cleared: {pixel:?}"
                    );
                }
            }
        }

        // New frame region (0,0)-(4,4) should be opaque white from RGB→RGBA.
        for y in 0..frame_h as usize {
            for x in 0..frame_w as usize {
                let idx = y * stride + x * 4;
                let pixel = &canvas[idx..idx + 4];
                assert_eq!(
                    pixel,
                    [0xFF, 0xFF, 0xFF, 0xFF],
                    "new-frame pixel ({x},{y}) wrong: {pixel:?}"
                );
            }
        }

        // Pixels outside both rectangles should be untouched (0xAA).
        let pixel = &canvas[7 * 4..7 * 4 + 4]; // (7,0) — outside both
        assert_eq!(pixel, [0xAA, 0xAA, 0xAA, 0xAA], "untouched pixel modified");
    }
}
