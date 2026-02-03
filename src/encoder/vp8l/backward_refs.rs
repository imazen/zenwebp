//! Backward reference finding using hash chains.
//!
//! Converts image pixels into a stream of literals and LZ77 backward references.

use super::color_cache::ColorCache;
use super::hash_chain::HashChain;
use super::types::{BackwardRefs, PixOrCopy, MIN_LENGTH, WINDOW_SIZE};

/// Distance code lookup table for 2D neighborhood.
/// Maps (xoffset, yoffset) pairs to distance codes 1-120.
#[rustfmt::skip]
const DISTANCE_MAP: [(i8, i8); 120] = [
    (0, 1),  (1, 0),  (1, 1),  (-1, 1), (0, 2),  (2, 0),  (1, 2),  (-1, 2),
    (2, 1),  (-2, 1), (2, 2),  (-2, 2), (0, 3),  (3, 0),  (1, 3),  (-1, 3),
    (3, 1),  (-3, 1), (2, 3),  (-2, 3), (3, 2),  (-3, 2), (0, 4),  (4, 0),
    (1, 4),  (-1, 4), (4, 1),  (-4, 1), (3, 3),  (-3, 3), (2, 4),  (-2, 4),
    (4, 2),  (-4, 2), (0, 5),  (3, 4),  (-3, 4), (4, 3),  (-4, 3), (5, 0),
    (1, 5),  (-1, 5), (5, 1),  (-5, 1), (2, 5),  (-2, 5), (5, 2),  (-5, 2),
    (4, 4),  (-4, 4), (3, 5),  (-3, 5), (5, 3),  (-5, 3), (0, 6),  (6, 0),
    (1, 6),  (-1, 6), (6, 1),  (-6, 1), (2, 6),  (-2, 6), (6, 2),  (-6, 2),
    (4, 5),  (-4, 5), (5, 4),  (-5, 4), (3, 6),  (-3, 6), (6, 3),  (-6, 3),
    (0, 7),  (7, 0),  (1, 7),  (-1, 7), (5, 5),  (-5, 5), (7, 1),  (-7, 1),
    (4, 6),  (-4, 6), (6, 4),  (-6, 4), (2, 7),  (-2, 7), (7, 2),  (-7, 2),
    (3, 7),  (-3, 7), (7, 3),  (-7, 3), (5, 6),  (-5, 6), (6, 5),  (-6, 5),
    (8, 0),  (4, 7),  (-4, 7), (7, 4),  (-7, 4), (8, 1),  (8, 2),  (6, 6),
    (-6, 6), (8, 3),  (5, 7),  (-5, 7), (7, 5),  (-7, 5), (8, 4),  (6, 7),
    (-6, 7), (7, 6),  (-7, 6), (8, 5),  (7, 7),  (-7, 7), (8, 6),  (8, 7)
];

/// Reverse lookup table: given (yoffset * 16 + 8 - xoffset), get distance code.
/// Only valid for xoffset in [0, 8] and yoffset in [0, 7].
#[rustfmt::skip]
const PLANE_TO_CODE_LUT: [u8; 128] = [
    96,  73,  55,  39,  23, 13, 5,  1,  255, 255, 255, 255, 255, 255, 255, 255,
    101, 78,  58,  42,  26, 16, 8,  2,  0,   3,   9,   17,  27,  43,  59,  79,
    102, 86,  62,  46,  32, 20, 10, 6,  4,   7,   11,  21,  33,  47,  63,  87,
    105, 90,  70,  52,  37, 28, 18, 14, 12,  15,  19,  29,  38,  53,  71,  91,
    110, 99,  82,  66,  48, 35, 30, 24, 22,  25,  31,  36,  49,  67,  83,  100,
    115, 108, 94,  76,  64, 50, 44, 40, 34,  41,  45,  51,  65,  77,  95,  109,
    118, 113, 103, 92,  80, 68, 60, 56, 54,  57,  61,  69,  81,  93,  104, 114,
    119, 116, 111, 106, 97, 88, 84, 74, 72,  75,  85,  89,  98,  107, 112, 117
];

/// Convert linear distance to distance code.
pub fn distance_to_plane_code(xsize: usize, dist: usize) -> u32 {
    let yoffset = dist / xsize;
    let xoffset = dist - yoffset * xsize;

    if xoffset <= 8 && yoffset < 8 {
        u32::from(PLANE_TO_CODE_LUT[yoffset * 16 + 8 - xoffset]) + 1
    } else if xoffset > xsize - 8 && yoffset < 7 {
        u32::from(PLANE_TO_CODE_LUT[(yoffset + 1) * 16 + 8 + (xsize - xoffset)]) + 1
    } else {
        (dist + 120) as u32
    }
}

/// Convert distance code back to linear distance.
pub fn plane_code_to_distance(xsize: usize, code: u32) -> usize {
    if code > 120 {
        (code - 120) as usize
    } else {
        let (xoff, yoff) = DISTANCE_MAP[(code - 1) as usize];
        let dist = xoff as i32 + yoff as i32 * xsize as i32;
        if dist < 1 {
            1
        } else {
            dist as usize
        }
    }
}

/// Find backward references for the given ARGB pixels.
pub fn compute_backward_refs(
    argb: &[u32],
    width: usize,
    height: usize,
    quality: u8,
    mut cache: Option<&mut ColorCache>,
) -> BackwardRefs {
    let size = width * height;
    let mut refs = BackwardRefs::with_capacity(size);

    if size == 0 {
        return refs;
    }

    // Build hash chain
    let chain = HashChain::new(argb, quality, width);

    let use_cache = cache.is_some();
    let mut pos = 0;

    while pos < size {
        let len = chain.length(pos);
        let dist = chain.offset(pos);

        // Check if we should use a backward reference
        if len >= MIN_LENGTH && dist > 0 && dist <= WINDOW_SIZE {
            // Emit backward reference
            let dist_code = distance_to_plane_code(width, dist);
            refs.push(PixOrCopy::copy(len as u16, dist_code));

            // Update color cache with copied pixels
            if let Some(ref mut c) = cache {
                for i in 0..len {
                    c.insert(argb[pos + i]);
                }
            }

            pos += len;
        } else {
            // Emit literal or cache index
            let argb_val = argb[pos];

            if use_cache {
                if let Some(idx) = cache.as_ref().unwrap().lookup(argb_val) {
                    refs.push(PixOrCopy::cache_idx(idx));
                } else {
                    refs.push(PixOrCopy::literal(argb_val));
                }
                cache.as_mut().unwrap().insert(argb_val);
            } else {
                refs.push(PixOrCopy::literal(argb_val));
            }

            pos += 1;
        }
    }

    refs
}

/// Simpler backward reference finder that only uses literals and run-length encoding.
/// Used for low quality or small images.
pub fn compute_backward_refs_simple(argb: &[u32], _width: usize, _height: usize) -> BackwardRefs {
    let size = argb.len();
    let mut refs = BackwardRefs::with_capacity(size);

    if size == 0 {
        return refs;
    }

    let mut pos = 0;
    while pos < size {
        let argb_val = argb[pos];

        // Check for run of identical pixels (distance = 1)
        let mut run_len = 1;
        while pos + run_len < size
            && argb[pos + run_len] == argb_val
            && run_len < super::types::MAX_LENGTH
        {
            run_len += 1;
        }

        if run_len >= MIN_LENGTH {
            // Emit first pixel as literal, then backward reference
            refs.push(PixOrCopy::literal(argb_val));
            if run_len > 1 {
                // Distance code 1 = distance 1 (previous pixel)
                refs.push(PixOrCopy::copy((run_len - 1) as u16, 1));
            }
            pos += run_len;
        } else {
            // Just emit literal
            refs.push(PixOrCopy::literal(argb_val));
            pos += 1;
        }
    }

    refs
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_distance_code_roundtrip() {
        let xsize = 100;
        for dist in 1..=200 {
            let code = distance_to_plane_code(xsize, dist);
            let back = plane_code_to_distance(xsize, code);
            assert_eq!(back, dist, "dist={}, code={}", dist, code);
        }
    }

    #[test]
    fn test_2d_neighborhood_codes() {
        let xsize = 100;
        // Distance 1 (previous pixel) maps to code 1
        // According to DISTANCE_MAP: (0,1) = code 1, (1,0) = code 2
        // dist=1 means xoffset=1, yoffset=0 which is code 2
        assert_eq!(distance_to_plane_code(xsize, 1), 2);
        // Distance = xsize (pixel above) means xoffset=0, yoffset=1 which is code 1
        assert_eq!(distance_to_plane_code(xsize, xsize), 1);
    }

    #[test]
    fn test_backward_refs_simple() {
        let pixels = vec![0xFF000000u32; 100];
        let refs = compute_backward_refs_simple(&pixels, 10, 10);

        // Should produce: 1 literal + 1 copy(99, 1)
        assert_eq!(refs.len(), 2);
        assert!(refs.tokens[0].is_literal());
        assert!(refs.tokens[1].is_copy());
    }
}
