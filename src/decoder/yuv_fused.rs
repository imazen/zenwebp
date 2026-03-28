//! Fused chroma-upsample + YUV→RGB conversion kernel.
//!
//! Eliminates the intermediate buffer between upsampling and color conversion
//! by performing both operations in-register. A single `#[arcane]` entry point
//! per architecture tier processes the entire row, avoiding repeated
//! target_feature boundary crossings.
//!
//! The math matches libwebp exactly:
//! - Fancy upsample: `(9*near_h*near_v + 3*far_h*near_v + 3*near_h*far_v + far_h*far_v + 8) / 16`
//!   implemented as chained `_mm_avg_epu8` (see `fancy_upsample_16`).
//! - YUV→RGB (BT.601):
//!   ```text
//!   R = clip((mulhi(Y,19077) + mulhi(V,26149) - 14234) >> 6)
//!   G = clip((mulhi(Y,19077) - mulhi(U,6419) - mulhi(V,13320) + 8708) >> 6)
//!   B = clip((mulhi(Y,19077) + mulhi(U,33050) - 17685) >> 6)
//!   ```

use super::yuv::{get_fancy_chroma_value, set_pixel};

// ============================================================================
// x86_64 V3 (AVX2+FMA) — single #[arcane] entry, #[rite] helpers
// ============================================================================

#[cfg(target_arch = "x86_64")]
mod x86_fused {
    use archmage::intrinsics::x86_64 as simd_mem;
    use archmage::{X64V3Token, arcane, rite};
    use core::arch::x86_64::*;

    use super::{get_fancy_chroma_value, set_pixel};

    // ---- #[rite] helpers (inline into the #[arcane] caller) ----

    /// Fancy upsample 8 (or 16) chroma samples using the libwebp avg-based method.
    /// Produces two __m128i: diag1 and diag2. Each chroma sample becomes two luma-aligned values.
    /// a = near[0..N], b = near[1..N+1], c = far[0..N], d = far[1..N+1]
    #[rite(v3, import_intrinsics)]
    fn fancy_upsample_16(
        a: __m128i,
        b: __m128i,
        c: __m128i,
        d: __m128i,
    ) -> (__m128i, __m128i) {
        let one = _mm_set1_epi8(1);

        // s = (a + d + 1) / 2
        let s = _mm_avg_epu8(a, d);
        // t = (b + c + 1) / 2
        let t = _mm_avg_epu8(b, c);
        let st = _mm_xor_si128(s, t);
        let ad = _mm_xor_si128(a, d);
        let bc = _mm_xor_si128(b, c);

        // k = (a + b + c + d) / 4 with proper rounding
        let t1 = _mm_or_si128(ad, bc);
        let t2 = _mm_or_si128(t1, st);
        let t3 = _mm_and_si128(t2, one);
        let t4 = _mm_avg_epu8(s, t);
        let k = _mm_sub_epi8(t4, t3);

        // m1 = (k + t + 1) / 2 - (((b^c) & (s^t)) | (k^t)) & 1
        let tmp1 = _mm_avg_epu8(k, t);
        let tmp2 = _mm_and_si128(bc, st);
        let tmp3 = _mm_xor_si128(k, t);
        let tmp4 = _mm_or_si128(tmp2, tmp3);
        let tmp5 = _mm_and_si128(tmp4, one);
        let m1 = _mm_sub_epi8(tmp1, tmp5);

        // m2 = (k + s + 1) / 2 - (((a^d) & (s^t)) | (k^s)) & 1
        let tmp1 = _mm_avg_epu8(k, s);
        let tmp2 = _mm_and_si128(ad, st);
        let tmp3 = _mm_xor_si128(k, s);
        let tmp4 = _mm_or_si128(tmp2, tmp3);
        let tmp5 = _mm_and_si128(tmp4, one);
        let m2 = _mm_sub_epi8(tmp1, tmp5);

        let diag1 = _mm_avg_epu8(a, m1); // (9*a + 3*b + 3*c + d + 8) / 16
        let diag2 = _mm_avg_epu8(b, m2); // (3*a + 9*b + c + 3*d + 8) / 16

        (diag1, diag2)
    }

    /// Convert 8 YUV444 pixels (16-bit, values in upper 8 bits) to R, G, B (16-bit).
    #[rite(v3, import_intrinsics)]
    fn convert_yuv444_to_rgb(
        y: __m128i,
        u: __m128i,
        v: __m128i,
    ) -> (__m128i, __m128i, __m128i) {
        let k19077 = _mm_set1_epi16(19077);
        let k26149 = _mm_set1_epi16(26149);
        let k14234 = _mm_set1_epi16(14234);
        let k33050 = _mm_set1_epi16(33050u16 as i16);
        let k17685 = _mm_set1_epi16(17685);
        let k6419 = _mm_set1_epi16(6419);
        let k13320 = _mm_set1_epi16(13320);
        let k8708 = _mm_set1_epi16(8708);

        let y1 = _mm_mulhi_epu16(y, k19077);

        // R = Y1 + V*26149 - 14234
        let r0 = _mm_mulhi_epu16(v, k26149);
        let r1 = _mm_sub_epi16(y1, k14234);
        let r2 = _mm_add_epi16(r1, r0);

        // G = Y1 - U*6419 - V*13320 + 8708
        let g0 = _mm_mulhi_epu16(u, k6419);
        let g1 = _mm_mulhi_epu16(v, k13320);
        let g2 = _mm_add_epi16(y1, k8708);
        let g3 = _mm_add_epi16(g0, g1);
        let g4 = _mm_sub_epi16(g2, g3);

        // B = Y1 + U*33050 - 17685
        let b0 = _mm_mulhi_epu16(u, k33050);
        let b1 = _mm_adds_epu16(b0, y1);
        let b2 = _mm_subs_epu16(b1, k17685);

        let r = _mm_srai_epi16(r2, 6);
        let g = _mm_srai_epi16(g4, 6);
        let b = _mm_srli_epi16(b2, 6);

        (r, g, b)
    }

    /// VP8PlanarTo24b: convert planar RRRR...GGGG...BBBB... to interleaved RGBRGB...
    macro_rules! planar_to_24b_helper {
        ($in0:expr, $in1:expr, $in2:expr, $in3:expr, $in4:expr, $in5:expr,
         $out0:expr, $out1:expr, $out2:expr, $out3:expr, $out4:expr, $out5:expr) => {
            let v_mask = _mm_set1_epi16(0x00ff);
            $out0 = _mm_packus_epi16(_mm_and_si128($in0, v_mask), _mm_and_si128($in1, v_mask));
            $out1 = _mm_packus_epi16(_mm_and_si128($in2, v_mask), _mm_and_si128($in3, v_mask));
            $out2 = _mm_packus_epi16(_mm_and_si128($in4, v_mask), _mm_and_si128($in5, v_mask));
            $out3 = _mm_packus_epi16(_mm_srli_epi16($in0, 8), _mm_srli_epi16($in1, 8));
            $out4 = _mm_packus_epi16(_mm_srli_epi16($in2, 8), _mm_srli_epi16($in3, 8));
            $out5 = _mm_packus_epi16(_mm_srli_epi16($in4, 8), _mm_srli_epi16($in5, 8));
        };
    }

    /// Planar R,G,B (32 pixels each) → interleaved 96-byte RGB.
    #[rite(v3, import_intrinsics)]
    fn planar_to_24b(
        in0: __m128i,
        in1: __m128i,
        in2: __m128i,
        in3: __m128i,
        in4: __m128i,
        in5: __m128i,
    ) -> (__m128i, __m128i, __m128i, __m128i, __m128i, __m128i) {
        let (mut t0, mut t1, mut t2, mut t3, mut t4, mut t5);
        let (mut o0, mut o1, mut o2, mut o3, mut o4, mut o5);

        planar_to_24b_helper!(in0, in1, in2, in3, in4, in5, t0, t1, t2, t3, t4, t5);
        planar_to_24b_helper!(t0, t1, t2, t3, t4, t5, o0, o1, o2, o3, o4, o5);
        planar_to_24b_helper!(o0, o1, o2, o3, o4, o5, t0, t1, t2, t3, t4, t5);
        planar_to_24b_helper!(t0, t1, t2, t3, t4, t5, o0, o1, o2, o3, o4, o5);
        planar_to_24b_helper!(o0, o1, o2, o3, o4, o5, t0, t1, t2, t3, t4, t5);

        (t0, t1, t2, t3, t4, t5)
    }

    /// Process 32 Y pixels (16 chroma pairs) → 96 bytes RGB.
    /// All work happens in-register: upsample chroma, convert YUV→RGB, interleave, store.
    #[rite(v3, import_intrinsics)]
    fn process_32_pixels(
        y_row: &[u8],
        y_offset: usize,
        u_row_1: &[u8],
        u_row_2: &[u8],
        v_row_1: &[u8],
        v_row_2: &[u8],
        uv_offset: usize,
        rgb: &mut [u8],
        rgb_offset: usize,
    ) {
        // Load 16+1 chroma samples for overlapping window
        let u_a = simd_mem::_mm_loadu_si128(
            <&[u8; 16]>::try_from(&u_row_1[uv_offset..uv_offset + 16]).unwrap(),
        );
        let u_b = simd_mem::_mm_loadu_si128(
            <&[u8; 16]>::try_from(&u_row_1[uv_offset + 1..uv_offset + 17]).unwrap(),
        );
        let u_c = simd_mem::_mm_loadu_si128(
            <&[u8; 16]>::try_from(&u_row_2[uv_offset..uv_offset + 16]).unwrap(),
        );
        let u_d = simd_mem::_mm_loadu_si128(
            <&[u8; 16]>::try_from(&u_row_2[uv_offset + 1..uv_offset + 17]).unwrap(),
        );

        let v_a = simd_mem::_mm_loadu_si128(
            <&[u8; 16]>::try_from(&v_row_1[uv_offset..uv_offset + 16]).unwrap(),
        );
        let v_b = simd_mem::_mm_loadu_si128(
            <&[u8; 16]>::try_from(&v_row_1[uv_offset + 1..uv_offset + 17]).unwrap(),
        );
        let v_c = simd_mem::_mm_loadu_si128(
            <&[u8; 16]>::try_from(&v_row_2[uv_offset..uv_offset + 16]).unwrap(),
        );
        let v_d = simd_mem::_mm_loadu_si128(
            <&[u8; 16]>::try_from(&v_row_2[uv_offset + 1..uv_offset + 17]).unwrap(),
        );

        // Upsample: 16 chroma → 32 luma-aligned values
        let (u_diag1, u_diag2) = fancy_upsample_16(u_a, u_b, u_c, u_d);
        let (v_diag1, v_diag2) = fancy_upsample_16(v_a, v_b, v_c, v_d);

        let u_lo = _mm_unpacklo_epi8(u_diag1, u_diag2);
        let u_hi = _mm_unpackhi_epi8(u_diag1, u_diag2);
        let v_lo = _mm_unpacklo_epi8(v_diag1, v_diag2);
        let v_hi = _mm_unpackhi_epi8(v_diag1, v_diag2);

        // Load 32 Y values
        let y_0 = simd_mem::_mm_loadu_si128(
            <&[u8; 16]>::try_from(&y_row[y_offset..y_offset + 16]).unwrap(),
        );
        let y_1 = simd_mem::_mm_loadu_si128(
            <&[u8; 16]>::try_from(&y_row[y_offset + 16..y_offset + 32]).unwrap(),
        );

        let zero = _mm_setzero_si128();

        // Group 0: Y[0..8]
        let y_0_lo = _mm_unpacklo_epi8(zero, y_0);
        let u_0_lo = _mm_unpacklo_epi8(zero, u_lo);
        let v_0_lo = _mm_unpacklo_epi8(zero, v_lo);
        let (r0, g0, b0) = convert_yuv444_to_rgb(y_0_lo, u_0_lo, v_0_lo);

        // Group 1: Y[8..16]
        let y_0_hi = _mm_unpackhi_epi8(zero, y_0);
        let u_0_hi = _mm_unpackhi_epi8(zero, u_lo);
        let v_0_hi = _mm_unpackhi_epi8(zero, v_lo);
        let (r1, g1, b1) = convert_yuv444_to_rgb(y_0_hi, u_0_hi, v_0_hi);

        // Group 2: Y[16..24]
        let y_1_lo = _mm_unpacklo_epi8(zero, y_1);
        let u_1_lo = _mm_unpacklo_epi8(zero, u_hi);
        let v_1_lo = _mm_unpacklo_epi8(zero, v_hi);
        let (r2, g2, b2) = convert_yuv444_to_rgb(y_1_lo, u_1_lo, v_1_lo);

        // Group 3: Y[24..32]
        let y_1_hi = _mm_unpackhi_epi8(zero, y_1);
        let u_1_hi = _mm_unpackhi_epi8(zero, u_hi);
        let v_1_hi = _mm_unpackhi_epi8(zero, v_hi);
        let (r3, g3, b3) = convert_yuv444_to_rgb(y_1_hi, u_1_hi, v_1_hi);

        // Pack to 8-bit
        let r_0 = _mm_packus_epi16(r0, r1);
        let r_1 = _mm_packus_epi16(r2, r3);
        let g_0 = _mm_packus_epi16(g0, g1);
        let g_1 = _mm_packus_epi16(g2, g3);
        let b_0 = _mm_packus_epi16(b0, b1);
        let b_1 = _mm_packus_epi16(b2, b3);

        // Interleave RGB
        let (out0, out1, out2, out3, out4, out5) =
            planar_to_24b(r_0, r_1, g_0, g_1, b_0, b_1);

        // Store 96 bytes
        let o = rgb_offset;
        simd_mem::_mm_storeu_si128(
            <&mut [u8; 16]>::try_from(&mut rgb[o..o + 16]).unwrap(),
            out0,
        );
        simd_mem::_mm_storeu_si128(
            <&mut [u8; 16]>::try_from(&mut rgb[o + 16..o + 32]).unwrap(),
            out1,
        );
        simd_mem::_mm_storeu_si128(
            <&mut [u8; 16]>::try_from(&mut rgb[o + 32..o + 48]).unwrap(),
            out2,
        );
        simd_mem::_mm_storeu_si128(
            <&mut [u8; 16]>::try_from(&mut rgb[o + 48..o + 64]).unwrap(),
            out3,
        );
        simd_mem::_mm_storeu_si128(
            <&mut [u8; 16]>::try_from(&mut rgb[o + 64..o + 80]).unwrap(),
            out4,
        );
        simd_mem::_mm_storeu_si128(
            <&mut [u8; 16]>::try_from(&mut rgb[o + 80..o + 96]).unwrap(),
            out5,
        );
    }

    /// Process 16 Y pixels (8 chroma pairs) → 48 bytes RGB.
    #[rite(v3, import_intrinsics)]
    fn process_16_pixels(
        y_row: &[u8],
        y_offset: usize,
        u_row_1: &[u8],
        u_row_2: &[u8],
        v_row_1: &[u8],
        v_row_2: &[u8],
        uv_offset: usize,
        rgb: &mut [u8],
        rgb_offset: usize,
    ) {
        // Load 8 chroma samples from each row (need 9 for overlapping window)
        macro_rules! load_8_from_slice {
            ($slice:expr, $off:expr) => {{
                let bytes: [u8; 8] = [
                    $slice[$off],
                    $slice[$off + 1],
                    $slice[$off + 2],
                    $slice[$off + 3],
                    $slice[$off + 4],
                    $slice[$off + 5],
                    $slice[$off + 6],
                    $slice[$off + 7],
                ];
                let val = i64::from_le_bytes(bytes);
                _mm_cvtsi64_si128(val)
            }};
        }

        let u_a = load_8_from_slice!(u_row_1, uv_offset);
        let u_b = load_8_from_slice!(u_row_1, uv_offset + 1);
        let u_c = load_8_from_slice!(u_row_2, uv_offset);
        let u_d = load_8_from_slice!(u_row_2, uv_offset + 1);

        let v_a = load_8_from_slice!(v_row_1, uv_offset);
        let v_b = load_8_from_slice!(v_row_1, uv_offset + 1);
        let v_c = load_8_from_slice!(v_row_2, uv_offset);
        let v_d = load_8_from_slice!(v_row_2, uv_offset + 1);

        // Upsample
        let (u_diag1, u_diag2) = fancy_upsample_16(u_a, u_b, u_c, u_d);
        let (v_diag1, v_diag2) = fancy_upsample_16(v_a, v_b, v_c, v_d);

        let u_interleaved = _mm_unpacklo_epi8(u_diag1, u_diag2);
        let v_interleaved = _mm_unpacklo_epi8(v_diag1, v_diag2);

        // Load 16 Y
        let y_vec = simd_mem::_mm_loadu_si128(
            <&[u8; 16]>::try_from(&y_row[y_offset..y_offset + 16]).unwrap(),
        );
        let zero = _mm_setzero_si128();

        // Process 2 groups of 8
        let y_lo = _mm_unpacklo_epi8(zero, y_vec);
        let u_lo = _mm_unpacklo_epi8(zero, u_interleaved);
        let v_lo = _mm_unpacklo_epi8(zero, v_interleaved);
        let (r0, g0, b0) = convert_yuv444_to_rgb(y_lo, u_lo, v_lo);

        let y_hi = _mm_unpackhi_epi8(zero, y_vec);
        let u_hi = _mm_unpackhi_epi8(zero, u_interleaved);
        let v_hi = _mm_unpackhi_epi8(zero, v_interleaved);
        let (r1, g1, b1) = convert_yuv444_to_rgb(y_hi, u_hi, v_hi);

        let r8 = _mm_packus_epi16(r0, r1);
        let g8 = _mm_packus_epi16(g0, g1);
        let b8 = _mm_packus_epi16(b0, b1);

        let (out0, out1, out2, _, _, _) = planar_to_24b(
            r8,
            _mm_setzero_si128(),
            g8,
            _mm_setzero_si128(),
            b8,
            _mm_setzero_si128(),
        );

        let o = rgb_offset;
        simd_mem::_mm_storeu_si128(
            <&mut [u8; 16]>::try_from(&mut rgb[o..o + 16]).unwrap(),
            out0,
        );
        simd_mem::_mm_storeu_si128(
            <&mut [u8; 16]>::try_from(&mut rgb[o + 16..o + 32]).unwrap(),
            out1,
        );
        simd_mem::_mm_storeu_si128(
            <&mut [u8; 16]>::try_from(&mut rgb[o + 32..o + 48]).unwrap(),
            out2,
        );
    }

    // ---- Single #[arcane] entry point for the entire row ----

    /// Fused fancy-upsample + YUV→RGB for one row, with two chroma rows (interior rows).
    /// Single target_feature boundary for the entire row.
    #[arcane]
    pub(crate) fn fused_row_2uv_x86(
        _token: X64V3Token,
        rgb: &mut [u8],
        y_row: &[u8],
        u_row_1: &[u8],
        u_row_2: &[u8],
        v_row_1: &[u8],
        v_row_2: &[u8],
    ) {
        let width = y_row.len();
        debug_assert!(rgb.len() >= width * 3);

        // Handle first pixel (edge: no left chroma neighbor)
        {
            let y_value = y_row[0];
            let u_value =
                get_fancy_chroma_value(u_row_1[0], u_row_1[0], u_row_2[0], u_row_2[0]);
            let v_value =
                get_fancy_chroma_value(v_row_1[0], v_row_1[0], v_row_2[0], v_row_2[0]);
            set_pixel(&mut rgb[0..3], y_value, u_value, v_value);
        }

        let mut y_offset: usize = 1;
        let mut uv_offset: usize = 0;
        let mut rgb_offset: usize = 3; // BPP=3

        // Process 32 Y pixels (16 chroma pairs) per iteration
        while y_offset + 32 <= width && uv_offset + 17 <= u_row_1.len() {
            process_32_pixels(
                y_row, y_offset, u_row_1, u_row_2, v_row_1, v_row_2, uv_offset, rgb,
                rgb_offset,
            );
            y_offset += 32;
            uv_offset += 16;
            rgb_offset += 96;
        }

        // Process 16 Y pixels (8 chroma pairs)
        while y_offset + 16 <= width && uv_offset + 9 <= u_row_1.len() {
            process_16_pixels(
                y_row, y_offset, u_row_1, u_row_2, v_row_1, v_row_2, uv_offset, rgb,
                rgb_offset,
            );
            y_offset += 16;
            uv_offset += 8;
            rgb_offset += 48;
        }

        // Scalar remainder: process pairs
        while y_offset + 2 <= width && uv_offset + 2 <= u_row_1.len() {
            {
                let y_value = y_row[y_offset];
                let u_value = get_fancy_chroma_value(
                    u_row_1[uv_offset],
                    u_row_1[uv_offset + 1],
                    u_row_2[uv_offset],
                    u_row_2[uv_offset + 1],
                );
                let v_value = get_fancy_chroma_value(
                    v_row_1[uv_offset],
                    v_row_1[uv_offset + 1],
                    v_row_2[uv_offset],
                    v_row_2[uv_offset + 1],
                );
                set_pixel(&mut rgb[rgb_offset..rgb_offset + 3], y_value, u_value, v_value);
            }
            {
                let y_value = y_row[y_offset + 1];
                let u_value = get_fancy_chroma_value(
                    u_row_1[uv_offset + 1],
                    u_row_1[uv_offset],
                    u_row_2[uv_offset + 1],
                    u_row_2[uv_offset],
                );
                let v_value = get_fancy_chroma_value(
                    v_row_1[uv_offset + 1],
                    v_row_1[uv_offset],
                    v_row_2[uv_offset + 1],
                    v_row_2[uv_offset],
                );
                set_pixel(
                    &mut rgb[rgb_offset + 3..rgb_offset + 6],
                    y_value,
                    u_value,
                    v_value,
                );
            }

            y_offset += 2;
            uv_offset += 1;
            rgb_offset += 6;
        }

        // Handle final odd pixel
        if y_offset < width {
            let final_u_1 = *u_row_1.last().unwrap();
            let final_u_2 = *u_row_2.last().unwrap();
            let final_v_1 = *v_row_1.last().unwrap();
            let final_v_2 = *v_row_2.last().unwrap();

            let u_value = get_fancy_chroma_value(final_u_1, final_u_1, final_u_2, final_u_2);
            let v_value = get_fancy_chroma_value(final_v_1, final_v_1, final_v_2, final_v_2);
            set_pixel(
                &mut rgb[rgb_offset..rgb_offset + 3],
                y_row[y_offset],
                u_value,
                v_value,
            );
        }
    }
}

#[cfg(target_arch = "x86_64")]
pub(crate) use x86_fused::fused_row_2uv_x86;

// ============================================================================
// aarch64 NEON — single #[arcane] entry, #[rite] helpers
// ============================================================================

#[cfg(target_arch = "aarch64")]
mod neon_fused {
    use archmage::intrinsics::aarch64 as simd_mem;
    use archmage::{NeonToken, arcane, rite};
    use core::arch::aarch64::*;

    use super::{get_fancy_chroma_value, set_pixel};

    // YUV→RGB constants matching libwebp's upsampling_neon.c
    const K_COEFFS1: [i16; 4] = [19077, 26149, 6419, 13320];
    const R_ROUNDER: i16 = -14234;
    const G_ROUNDER: i16 = 8708;
    const B_ROUNDER: i16 = -17685;
    const B_MULT_EXTRA: i16 = 282;

    #[rite]
    fn upsample_16pixels_neon(
        _token: NeonToken,
        a: uint8x8_t,
        b: uint8x8_t,
        c: uint8x8_t,
        d: uint8x8_t,
    ) -> uint8x16_t {
        let one = vdup_n_u8(1);

        let s = vrhadd_u8(a, d);
        let t = vrhadd_u8(b, c);
        let st = veor_u8(s, t);
        let ad = veor_u8(a, d);
        let bc = veor_u8(b, c);

        let t1 = vorr_u8(ad, bc);
        let t2 = vorr_u8(t1, st);
        let t3 = vand_u8(t2, one);
        let t4 = vrhadd_u8(s, t);
        let k = vsub_u8(t4, t3);

        let tmp1 = vrhadd_u8(k, t);
        let tmp2 = vand_u8(bc, st);
        let tmp3 = veor_u8(k, t);
        let tmp4 = vorr_u8(tmp2, tmp3);
        let tmp5 = vand_u8(tmp4, one);
        let m1 = vsub_u8(tmp1, tmp5);

        let tmp1 = vrhadd_u8(k, s);
        let tmp2 = vand_u8(ad, st);
        let tmp3 = veor_u8(k, s);
        let tmp4 = vorr_u8(tmp2, tmp3);
        let tmp5 = vand_u8(tmp4, one);
        let m2 = vsub_u8(tmp1, tmp5);

        let diag1 = vrhadd_u8(a, m1);
        let diag2 = vrhadd_u8(b, m2);

        let zip = vzip_u8(diag1, diag2);
        vcombine_u8(zip.0, zip.1)
    }

    #[rite]
    fn convert_and_store_rgb16_neon(
        _token: NeonToken,
        y_vals: uint8x16_t,
        u_vals: uint8x16_t,
        v_vals: uint8x16_t,
        rgb: &mut [u8; 48],
    ) {
        let coeffs1 = simd_mem::vld1_s16(&K_COEFFS1);

        let y_lo = vget_low_u8(y_vals);
        let y_hi = vget_high_u8(y_vals);
        let u_lo = vget_low_u8(u_vals);
        let u_hi = vget_high_u8(u_vals);
        let v_lo = vget_low_u8(v_vals);
        let v_hi = vget_high_u8(v_vals);

        // Widen to i16 and shift left by 7 (multiply by 128)
        let y_lo16 = vreinterpretq_s16_u16(vshll_n_u8::<7>(y_lo));
        let y_hi16 = vreinterpretq_s16_u16(vshll_n_u8::<7>(y_hi));
        let u_lo16 = vreinterpretq_s16_u16(vshll_n_u8::<7>(u_lo));
        let u_hi16 = vreinterpretq_s16_u16(vshll_n_u8::<7>(u_hi));
        let v_lo16 = vreinterpretq_s16_u16(vshll_n_u8::<7>(v_lo));
        let v_hi16 = vreinterpretq_s16_u16(vshll_n_u8::<7>(v_hi));

        // Y * 19077
        let y1_lo = vqdmulhq_lane_s16::<0>(y_lo16, coeffs1);
        let y1_hi = vqdmulhq_lane_s16::<0>(y_hi16, coeffs1);

        // R = Y1 + V*26149 - 14234
        let r_rounder = vdupq_n_s16(R_ROUNDER);
        let r0_lo = vqdmulhq_lane_s16::<1>(v_lo16, coeffs1);
        let r0_hi = vqdmulhq_lane_s16::<1>(v_hi16, coeffs1);
        let r1_lo = vaddq_s16(y1_lo, r_rounder);
        let r1_hi = vaddq_s16(y1_hi, r_rounder);
        let r2_lo = vaddq_s16(r1_lo, r0_lo);
        let r2_hi = vaddq_s16(r1_hi, r0_hi);

        // G = Y1 - U*6419 - V*13320 + 8708
        let g_rounder = vdupq_n_s16(G_ROUNDER);
        let g0_lo = vqdmulhq_lane_s16::<2>(u_lo16, coeffs1);
        let g0_hi = vqdmulhq_lane_s16::<2>(u_hi16, coeffs1);
        let g1_lo = vqdmulhq_lane_s16::<3>(v_lo16, coeffs1);
        let g1_hi = vqdmulhq_lane_s16::<3>(v_hi16, coeffs1);
        let g2_lo = vaddq_s16(y1_lo, g_rounder);
        let g2_hi = vaddq_s16(y1_hi, g_rounder);
        let g3_lo = vaddq_s16(g0_lo, g1_lo);
        let g3_hi = vaddq_s16(g0_hi, g1_hi);
        let g4_lo = vsubq_s16(g2_lo, g3_lo);
        let g4_hi = vsubq_s16(g2_hi, g3_hi);

        // B = Y1 + U*33050 - 17685
        // 33050 = 32768 + 282, split: vqdmulhq_n(U, 282) + U
        let b_rounder = vdupq_n_s16(B_ROUNDER);
        let b0_lo = vqdmulhq_n_s16(u_lo16, B_MULT_EXTRA);
        let b0_hi = vqdmulhq_n_s16(u_hi16, B_MULT_EXTRA);
        let b1_lo = vaddq_s16(b0_lo, vreinterpretq_s16_u16(vshll_n_u8::<7>(u_lo)));
        let b1_hi = vaddq_s16(b0_hi, vreinterpretq_s16_u16(vshll_n_u8::<7>(u_hi)));
        let b2_lo = vaddq_s16(y1_lo, b_rounder);
        let b2_hi = vaddq_s16(y1_hi, b_rounder);
        let b3_lo = vaddq_s16(b2_lo, b1_lo);
        let b3_hi = vaddq_s16(b2_hi, b1_hi);

        // Shift right by 6, clamp to 0..255
        let r_lo = vqshrun_n_s16::<6>(r2_lo);
        let r_hi = vqshrun_n_s16::<6>(r2_hi);
        let g_lo = vqshrun_n_s16::<6>(g4_lo);
        let g_hi = vqshrun_n_s16::<6>(g4_hi);
        let b_lo = vqshrun_n_s16::<6>(b3_lo);
        let b_hi = vqshrun_n_s16::<6>(b3_hi);

        let r = vcombine_u8(r_lo, r_hi);
        let g = vcombine_u8(g_lo, g_hi);
        let b = vcombine_u8(b_lo, b_hi);

        // Store as interleaved RGB using hardware vst3q_u8
        let rgb_array = uint8x16x3_t(r, g, b);
        simd_mem::vst3q_u8(rgb, rgb_array);
    }

    /// Single #[arcane] entry: process entire row with 2 UV rows (interior).
    #[arcane]
    pub(crate) fn fused_row_2uv_neon(
        _token: NeonToken,
        rgb: &mut [u8],
        y_row: &[u8],
        u_row_1: &[u8],
        u_row_2: &[u8],
        v_row_1: &[u8],
        v_row_2: &[u8],
    ) {
        let width = y_row.len();
        debug_assert!(rgb.len() >= width * 3);

        // Handle first pixel (edge)
        {
            let y_value = y_row[0];
            let u_value =
                get_fancy_chroma_value(u_row_1[0], u_row_1[0], u_row_2[0], u_row_2[0]);
            let v_value =
                get_fancy_chroma_value(v_row_1[0], v_row_1[0], v_row_2[0], v_row_2[0]);
            set_pixel(&mut rgb[0..3], y_value, u_value, v_value);
        }

        let mut y_offset: usize = 1;
        let mut uv_offset: usize = 0;
        let mut rgb_offset: usize = 3;

        // 32 Y pixels (16 chroma pairs) per iteration
        while y_offset + 32 <= width && uv_offset + 17 <= u_row_1.len() {
            let u_a0 = simd_mem::vld1_u8(
                <&[u8; 8]>::try_from(&u_row_1[uv_offset..uv_offset + 8]).unwrap(),
            );
            let u_b0 = simd_mem::vld1_u8(
                <&[u8; 8]>::try_from(&u_row_1[uv_offset + 1..uv_offset + 9]).unwrap(),
            );
            let u_c0 = simd_mem::vld1_u8(
                <&[u8; 8]>::try_from(&u_row_2[uv_offset..uv_offset + 8]).unwrap(),
            );
            let u_d0 = simd_mem::vld1_u8(
                <&[u8; 8]>::try_from(&u_row_2[uv_offset + 1..uv_offset + 9]).unwrap(),
            );
            let u_a1 = simd_mem::vld1_u8(
                <&[u8; 8]>::try_from(&u_row_1[uv_offset + 8..uv_offset + 16]).unwrap(),
            );
            let u_b1 = simd_mem::vld1_u8(
                <&[u8; 8]>::try_from(&u_row_1[uv_offset + 9..uv_offset + 17]).unwrap(),
            );
            let u_c1 = simd_mem::vld1_u8(
                <&[u8; 8]>::try_from(&u_row_2[uv_offset + 8..uv_offset + 16]).unwrap(),
            );
            let u_d1 = simd_mem::vld1_u8(
                <&[u8; 8]>::try_from(&u_row_2[uv_offset + 9..uv_offset + 17]).unwrap(),
            );

            let v_a0 = simd_mem::vld1_u8(
                <&[u8; 8]>::try_from(&v_row_1[uv_offset..uv_offset + 8]).unwrap(),
            );
            let v_b0 = simd_mem::vld1_u8(
                <&[u8; 8]>::try_from(&v_row_1[uv_offset + 1..uv_offset + 9]).unwrap(),
            );
            let v_c0 = simd_mem::vld1_u8(
                <&[u8; 8]>::try_from(&v_row_2[uv_offset..uv_offset + 8]).unwrap(),
            );
            let v_d0 = simd_mem::vld1_u8(
                <&[u8; 8]>::try_from(&v_row_2[uv_offset + 1..uv_offset + 9]).unwrap(),
            );
            let v_a1 = simd_mem::vld1_u8(
                <&[u8; 8]>::try_from(&v_row_1[uv_offset + 8..uv_offset + 16]).unwrap(),
            );
            let v_b1 = simd_mem::vld1_u8(
                <&[u8; 8]>::try_from(&v_row_1[uv_offset + 9..uv_offset + 17]).unwrap(),
            );
            let v_c1 = simd_mem::vld1_u8(
                <&[u8; 8]>::try_from(&v_row_2[uv_offset + 8..uv_offset + 16]).unwrap(),
            );
            let v_d1 = simd_mem::vld1_u8(
                <&[u8; 8]>::try_from(&v_row_2[uv_offset + 9..uv_offset + 17]).unwrap(),
            );

            let u_up0 = upsample_16pixels_neon(_token, u_a0, u_b0, u_c0, u_d0);
            let u_up1 = upsample_16pixels_neon(_token, u_a1, u_b1, u_c1, u_d1);
            let v_up0 = upsample_16pixels_neon(_token, v_a0, v_b0, v_c0, v_d0);
            let v_up1 = upsample_16pixels_neon(_token, v_a1, v_b1, v_c1, v_d1);

            let y0 = simd_mem::vld1q_u8(
                <&[u8; 16]>::try_from(&y_row[y_offset..y_offset + 16]).unwrap(),
            );
            let y1 = simd_mem::vld1q_u8(
                <&[u8; 16]>::try_from(&y_row[y_offset + 16..y_offset + 32]).unwrap(),
            );

            convert_and_store_rgb16_neon(
                _token,
                y0,
                u_up0,
                v_up0,
                <&mut [u8; 48]>::try_from(&mut rgb[rgb_offset..rgb_offset + 48]).unwrap(),
            );
            convert_and_store_rgb16_neon(
                _token,
                y1,
                u_up1,
                v_up1,
                <&mut [u8; 48]>::try_from(&mut rgb[rgb_offset + 48..rgb_offset + 96]).unwrap(),
            );

            y_offset += 32;
            uv_offset += 16;
            rgb_offset += 96;
        }

        // 16 Y pixels (8 chroma pairs)
        while y_offset + 16 <= width && uv_offset + 9 <= u_row_1.len() {
            let u_a = simd_mem::vld1_u8(
                <&[u8; 8]>::try_from(&u_row_1[uv_offset..uv_offset + 8]).unwrap(),
            );
            let u_b = simd_mem::vld1_u8(
                <&[u8; 8]>::try_from(&u_row_1[uv_offset + 1..uv_offset + 9]).unwrap(),
            );
            let u_c = simd_mem::vld1_u8(
                <&[u8; 8]>::try_from(&u_row_2[uv_offset..uv_offset + 8]).unwrap(),
            );
            let u_d = simd_mem::vld1_u8(
                <&[u8; 8]>::try_from(&u_row_2[uv_offset + 1..uv_offset + 9]).unwrap(),
            );
            let v_a = simd_mem::vld1_u8(
                <&[u8; 8]>::try_from(&v_row_1[uv_offset..uv_offset + 8]).unwrap(),
            );
            let v_b = simd_mem::vld1_u8(
                <&[u8; 8]>::try_from(&v_row_1[uv_offset + 1..uv_offset + 9]).unwrap(),
            );
            let v_c = simd_mem::vld1_u8(
                <&[u8; 8]>::try_from(&v_row_2[uv_offset..uv_offset + 8]).unwrap(),
            );
            let v_d = simd_mem::vld1_u8(
                <&[u8; 8]>::try_from(&v_row_2[uv_offset + 1..uv_offset + 9]).unwrap(),
            );

            let u_up = upsample_16pixels_neon(_token, u_a, u_b, u_c, u_d);
            let v_up = upsample_16pixels_neon(_token, v_a, v_b, v_c, v_d);

            let y_vec = simd_mem::vld1q_u8(
                <&[u8; 16]>::try_from(&y_row[y_offset..y_offset + 16]).unwrap(),
            );

            convert_and_store_rgb16_neon(
                _token,
                y_vec,
                u_up,
                v_up,
                <&mut [u8; 48]>::try_from(&mut rgb[rgb_offset..rgb_offset + 48]).unwrap(),
            );

            y_offset += 16;
            uv_offset += 8;
            rgb_offset += 48;
        }

        // Scalar remainder
        while y_offset + 2 <= width && uv_offset + 2 <= u_row_1.len() {
            {
                let y_value = y_row[y_offset];
                let u_value = get_fancy_chroma_value(
                    u_row_1[uv_offset],
                    u_row_1[uv_offset + 1],
                    u_row_2[uv_offset],
                    u_row_2[uv_offset + 1],
                );
                let v_value = get_fancy_chroma_value(
                    v_row_1[uv_offset],
                    v_row_1[uv_offset + 1],
                    v_row_2[uv_offset],
                    v_row_2[uv_offset + 1],
                );
                set_pixel(&mut rgb[rgb_offset..rgb_offset + 3], y_value, u_value, v_value);
            }
            {
                let y_value = y_row[y_offset + 1];
                let u_value = get_fancy_chroma_value(
                    u_row_1[uv_offset + 1],
                    u_row_1[uv_offset],
                    u_row_2[uv_offset + 1],
                    u_row_2[uv_offset],
                );
                let v_value = get_fancy_chroma_value(
                    v_row_1[uv_offset + 1],
                    v_row_1[uv_offset],
                    v_row_2[uv_offset + 1],
                    v_row_2[uv_offset],
                );
                set_pixel(
                    &mut rgb[rgb_offset + 3..rgb_offset + 6],
                    y_value,
                    u_value,
                    v_value,
                );
            }
            y_offset += 2;
            uv_offset += 1;
            rgb_offset += 6;
        }

        // Final odd pixel
        if y_offset < width {
            let final_u_1 = *u_row_1.last().unwrap();
            let final_u_2 = *u_row_2.last().unwrap();
            let final_v_1 = *v_row_1.last().unwrap();
            let final_v_2 = *v_row_2.last().unwrap();

            let u_value = get_fancy_chroma_value(final_u_1, final_u_1, final_u_2, final_u_2);
            let v_value = get_fancy_chroma_value(final_v_1, final_v_1, final_v_2, final_v_2);
            set_pixel(
                &mut rgb[rgb_offset..rgb_offset + 3],
                y_row[y_offset],
                u_value,
                v_value,
            );
        }
    }
}

#[cfg(target_arch = "aarch64")]
pub(crate) use neon_fused::fused_row_2uv_neon;

// ============================================================================
// wasm32 SIMD128 — single #[arcane] entry
// ============================================================================

#[cfg(target_arch = "wasm32")]
mod wasm_fused {
    use archmage::{Wasm128Token, arcane};
    use core::arch::wasm32::*;

    use super::{get_fancy_chroma_value, set_pixel};

    // Re-use the wasm helper functions from yuv.rs via import
    // For now, delegate to the existing per-chunk functions
    // The key win on wasm is the same: single #[arcane] boundary per row

    #[arcane]
    pub(crate) fn fused_row_2uv_wasm(
        _token: Wasm128Token,
        rgb: &mut [u8],
        y_row: &[u8],
        u_row_1: &[u8],
        u_row_2: &[u8],
        v_row_1: &[u8],
        v_row_2: &[u8],
    ) {
        let width = y_row.len();
        debug_assert!(rgb.len() >= width * 3);

        // Handle first pixel (edge)
        {
            let y_value = y_row[0];
            let u_value =
                get_fancy_chroma_value(u_row_1[0], u_row_1[0], u_row_2[0], u_row_2[0]);
            let v_value =
                get_fancy_chroma_value(v_row_1[0], v_row_1[0], v_row_2[0], v_row_2[0]);
            set_pixel(&mut rgb[0..3], y_value, u_value, v_value);
        }

        let mut y_offset: usize = 1;
        let mut uv_offset: usize = 0;
        let mut rgb_offset: usize = 3;

        // Helper: load 8 bytes into low half of v128
        fn load_u8x8_low(src: &[u8; 8]) -> v128 {
            let val = u64::from_le_bytes(*src);
            u64x2_replace_lane::<0>(i64x2_splat(0), val)
        }

        fn load_u8x16(src: &[u8; 16]) -> v128 {
            v128_load(src.as_ptr() as *const v128)
        }

        fn store_u8x16(dst: &mut [u8; 16], v: v128) {
            v128_store(dst.as_mut_ptr() as *mut v128, v);
        }

        // Upsample 8 chroma → 16 pixels
        fn upsample_16pixels(a: v128, b: v128, c: v128, d: v128) -> v128 {
            let one = u8x16_splat(1);
            let s = u8x16_avgr(a, d);
            let t = u8x16_avgr(b, c);
            let st = v128_xor(s, t);
            let ad = v128_xor(a, d);
            let bc = v128_xor(b, c);

            let t1 = v128_or(ad, bc);
            let t2 = v128_or(t1, st);
            let t3 = v128_and(t2, one);
            let t4 = u8x16_avgr(s, t);
            let k = u8x16_sub(t4, t3);

            let tmp1 = u8x16_avgr(k, t);
            let tmp2 = v128_and(bc, st);
            let tmp3 = v128_xor(k, t);
            let tmp4 = v128_or(tmp2, tmp3);
            let tmp5 = v128_and(tmp4, one);
            let m1 = u8x16_sub(tmp1, tmp5);

            let tmp1 = u8x16_avgr(k, s);
            let tmp2 = v128_and(ad, st);
            let tmp3 = v128_xor(k, s);
            let tmp4 = v128_or(tmp2, tmp3);
            let tmp5 = v128_and(tmp4, one);
            let m2 = u8x16_sub(tmp1, tmp5);

            let diag1 = u8x16_avgr(a, m1);
            let diag2 = u8x16_avgr(b, m2);

            i8x16_shuffle::<0, 16, 1, 17, 2, 18, 3, 19, 4, 20, 5, 21, 6, 22, 7, 23>(
                diag1, diag2,
            )
        }

        // Convert 16 YUV444 → 48 bytes RGB
        fn convert_and_store_rgb16(y: v128, u: v128, v: v128, rgb: &mut [u8; 48]) {
            fn process_half(y_half: v128, u_half: v128, v_half: v128) -> (v128, v128, v128) {
                let zero = u16x8_splat(0);
                let k19077 = u16x8_splat(19077);
                let k26149 = u16x8_splat(26149);
                let k14234 = u16x8_splat(14234);
                let k33050 = u16x8_splat(33050);
                let k17685 = u16x8_splat(17685);
                let k6419 = u16x8_splat(6419);
                let k13320 = u16x8_splat(13320);
                let k8708 = u16x8_splat(8708);

                // Widen to u16 in high byte position
                let y16 = u16x8_extend_high_u8x16(i8x16_shuffle::<
                    8, 0, 9, 1, 10, 2, 11, 3, 12, 4, 13, 5, 14, 6, 15, 7,
                >(zero, y_half));
                let u16v = u16x8_extend_high_u8x16(i8x16_shuffle::<
                    8, 0, 9, 1, 10, 2, 11, 3, 12, 4, 13, 5, 14, 6, 15, 7,
                >(zero, u_half));
                let v16v = u16x8_extend_high_u8x16(i8x16_shuffle::<
                    8, 0, 9, 1, 10, 2, 11, 3, 12, 4, 13, 5, 14, 6, 15, 7,
                >(zero, v_half));

                // mulhi_epu16 emulation: multiply wide, take high 16 bits
                fn mulhi_epu16(a: v128, b: v128) -> v128 {
                    // Compute (a[i] * b[i]) >> 16 for unsigned u16 lanes
                    // Split into even/odd lanes and multiply as u32
                    let mask = u32x4_splat(0x0000_FFFF);
                    let a_even = v128_and(a, mask);
                    let b_even = v128_and(b, mask);
                    let a_odd = u32x4_shr(a, 16);
                    let b_odd = u32x4_shr(b, 16);

                    let prod_even = i32x4_mul(a_even, b_even);
                    let prod_odd = i32x4_mul(a_odd, b_odd);

                    let hi_even = u32x4_shr(prod_even, 16);
                    let hi_odd = u32x4_shr(prod_odd, 16);

                    // Recombine: even into low 16 bits, odd into high 16 bits
                    v128_or(v128_and(hi_even, mask), i32x4_shl(hi_odd, 16))
                }

                let y1 = mulhi_epu16(y16, k19077);

                // R
                let r0 = mulhi_epu16(v16v, k26149);
                let r1 = i16x8_sub(y1, k14234);
                let r2 = i16x8_add(r1, r0);
                let r = i16x8_shr(r2, 6);

                // G
                let g0 = mulhi_epu16(u16v, k6419);
                let g1 = mulhi_epu16(v16v, k13320);
                let g2 = i16x8_add(y1, k8708);
                let g3 = i16x8_add(g0, g1);
                let g4 = i16x8_sub(g2, g3);
                let g = i16x8_shr(g4, 6);

                // B
                let b0 = mulhi_epu16(u16v, k33050);
                let b1 = u16x8_add_sat(b0, y1);
                let b2 = u16x8_sub_sat(b1, k17685);
                let b_val = u16x8_shr(b2, 6);

                (r, g, b_val)
            }

            // Split into low/high halves
            let y_lo = y;
            let u_lo = u;
            let v_lo = v;

            // Process low 8 pixels
            let y_lo_half =
                i8x16_shuffle::<0, 1, 2, 3, 4, 5, 6, 7, 16, 16, 16, 16, 16, 16, 16, 16>(
                    y_lo,
                    u8x16_splat(0),
                );
            let u_lo_half =
                i8x16_shuffle::<0, 1, 2, 3, 4, 5, 6, 7, 16, 16, 16, 16, 16, 16, 16, 16>(
                    u_lo,
                    u8x16_splat(0),
                );
            let v_lo_half =
                i8x16_shuffle::<0, 1, 2, 3, 4, 5, 6, 7, 16, 16, 16, 16, 16, 16, 16, 16>(
                    v_lo,
                    u8x16_splat(0),
                );
            let (r0, g0, b0) = process_half(y_lo_half, u_lo_half, v_lo_half);

            // Process high 8 pixels
            let y_hi_half =
                i8x16_shuffle::<8, 9, 10, 11, 12, 13, 14, 15, 16, 16, 16, 16, 16, 16, 16, 16>(
                    y_lo,
                    u8x16_splat(0),
                );
            let u_hi_half =
                i8x16_shuffle::<8, 9, 10, 11, 12, 13, 14, 15, 16, 16, 16, 16, 16, 16, 16, 16>(
                    u_lo,
                    u8x16_splat(0),
                );
            let v_hi_half =
                i8x16_shuffle::<8, 9, 10, 11, 12, 13, 14, 15, 16, 16, 16, 16, 16, 16, 16, 16>(
                    v_lo,
                    u8x16_splat(0),
                );
            let (r1, g1, b1) = process_half(y_hi_half, u_hi_half, v_hi_half);

            // Pack to u8
            let r8 = u8x16_narrow_i16x8(r0, r1);
            let g8 = u8x16_narrow_i16x8(g0, g1);
            let b8 = u8x16_narrow_i16x8(b0, b1);

            // Interleave RGB manually (no vst3 on wasm)
            for i in 0..16 {
                rgb[i * 3] = u8x16_extract_lane::<0>(u8x16_shr(
                    i8x16_swizzle(r8, u8x16_splat(i as u8)),
                    0,
                ));
                rgb[i * 3 + 1] = u8x16_extract_lane::<0>(i8x16_swizzle(g8, u8x16_splat(i as u8)));
                rgb[i * 3 + 2] = u8x16_extract_lane::<0>(i8x16_swizzle(b8, u8x16_splat(i as u8)));
            }
        }

        // Main SIMD loops — same structure as x86, but using wasm128 intrinsics
        // For wasm, delegate to the existing per-chunk approach but inside single #[arcane]
        // The 32-pixel and 16-pixel loops inline the full upsample+convert pipeline

        while y_offset + 32 <= width && uv_offset + 17 <= u_row_1.len() {
            let u_a0 = load_u8x8_low(
                <&[u8; 8]>::try_from(&u_row_1[uv_offset..uv_offset + 8]).unwrap(),
            );
            let u_b0 = load_u8x8_low(
                <&[u8; 8]>::try_from(&u_row_1[uv_offset + 1..uv_offset + 9]).unwrap(),
            );
            let u_c0 = load_u8x8_low(
                <&[u8; 8]>::try_from(&u_row_2[uv_offset..uv_offset + 8]).unwrap(),
            );
            let u_d0 = load_u8x8_low(
                <&[u8; 8]>::try_from(&u_row_2[uv_offset + 1..uv_offset + 9]).unwrap(),
            );
            let u_a1 = load_u8x8_low(
                <&[u8; 8]>::try_from(&u_row_1[uv_offset + 8..uv_offset + 16]).unwrap(),
            );
            let u_b1 = load_u8x8_low(
                <&[u8; 8]>::try_from(&u_row_1[uv_offset + 9..uv_offset + 17]).unwrap(),
            );
            let u_c1 = load_u8x8_low(
                <&[u8; 8]>::try_from(&u_row_2[uv_offset + 8..uv_offset + 16]).unwrap(),
            );
            let u_d1 = load_u8x8_low(
                <&[u8; 8]>::try_from(&u_row_2[uv_offset + 9..uv_offset + 17]).unwrap(),
            );
            let v_a0 = load_u8x8_low(
                <&[u8; 8]>::try_from(&v_row_1[uv_offset..uv_offset + 8]).unwrap(),
            );
            let v_b0 = load_u8x8_low(
                <&[u8; 8]>::try_from(&v_row_1[uv_offset + 1..uv_offset + 9]).unwrap(),
            );
            let v_c0 = load_u8x8_low(
                <&[u8; 8]>::try_from(&v_row_2[uv_offset..uv_offset + 8]).unwrap(),
            );
            let v_d0 = load_u8x8_low(
                <&[u8; 8]>::try_from(&v_row_2[uv_offset + 1..uv_offset + 9]).unwrap(),
            );
            let v_a1 = load_u8x8_low(
                <&[u8; 8]>::try_from(&v_row_1[uv_offset + 8..uv_offset + 16]).unwrap(),
            );
            let v_b1 = load_u8x8_low(
                <&[u8; 8]>::try_from(&v_row_1[uv_offset + 9..uv_offset + 17]).unwrap(),
            );
            let v_c1 = load_u8x8_low(
                <&[u8; 8]>::try_from(&v_row_2[uv_offset + 8..uv_offset + 16]).unwrap(),
            );
            let v_d1 = load_u8x8_low(
                <&[u8; 8]>::try_from(&v_row_2[uv_offset + 9..uv_offset + 17]).unwrap(),
            );

            let u_up0 = upsample_16pixels(u_a0, u_b0, u_c0, u_d0);
            let u_up1 = upsample_16pixels(u_a1, u_b1, u_c1, u_d1);
            let v_up0 = upsample_16pixels(v_a0, v_b0, v_c0, v_d0);
            let v_up1 = upsample_16pixels(v_a1, v_b1, v_c1, v_d1);

            let y0 = load_u8x16(
                <&[u8; 16]>::try_from(&y_row[y_offset..y_offset + 16]).unwrap(),
            );
            let y1 = load_u8x16(
                <&[u8; 16]>::try_from(&y_row[y_offset + 16..y_offset + 32]).unwrap(),
            );

            convert_and_store_rgb16(
                y0,
                u_up0,
                v_up0,
                <&mut [u8; 48]>::try_from(&mut rgb[rgb_offset..rgb_offset + 48]).unwrap(),
            );
            convert_and_store_rgb16(
                y1,
                u_up1,
                v_up1,
                <&mut [u8; 48]>::try_from(&mut rgb[rgb_offset + 48..rgb_offset + 96]).unwrap(),
            );

            y_offset += 32;
            uv_offset += 16;
            rgb_offset += 96;
        }

        while y_offset + 16 <= width && uv_offset + 9 <= u_row_1.len() {
            let u_a = load_u8x8_low(
                <&[u8; 8]>::try_from(&u_row_1[uv_offset..uv_offset + 8]).unwrap(),
            );
            let u_b = load_u8x8_low(
                <&[u8; 8]>::try_from(&u_row_1[uv_offset + 1..uv_offset + 9]).unwrap(),
            );
            let u_c = load_u8x8_low(
                <&[u8; 8]>::try_from(&u_row_2[uv_offset..uv_offset + 8]).unwrap(),
            );
            let u_d = load_u8x8_low(
                <&[u8; 8]>::try_from(&u_row_2[uv_offset + 1..uv_offset + 9]).unwrap(),
            );
            let v_a = load_u8x8_low(
                <&[u8; 8]>::try_from(&v_row_1[uv_offset..uv_offset + 8]).unwrap(),
            );
            let v_b = load_u8x8_low(
                <&[u8; 8]>::try_from(&v_row_1[uv_offset + 1..uv_offset + 9]).unwrap(),
            );
            let v_c = load_u8x8_low(
                <&[u8; 8]>::try_from(&v_row_2[uv_offset..uv_offset + 8]).unwrap(),
            );
            let v_d = load_u8x8_low(
                <&[u8; 8]>::try_from(&v_row_2[uv_offset + 1..uv_offset + 9]).unwrap(),
            );

            let u_up = upsample_16pixels(u_a, u_b, u_c, u_d);
            let v_up = upsample_16pixels(v_a, v_b, v_c, v_d);
            let y_vec = load_u8x16(
                <&[u8; 16]>::try_from(&y_row[y_offset..y_offset + 16]).unwrap(),
            );

            convert_and_store_rgb16(
                y_vec,
                u_up,
                v_up,
                <&mut [u8; 48]>::try_from(&mut rgb[rgb_offset..rgb_offset + 48]).unwrap(),
            );

            y_offset += 16;
            uv_offset += 8;
            rgb_offset += 48;
        }

        // Scalar remainder
        while y_offset + 2 <= width && uv_offset + 2 <= u_row_1.len() {
            {
                let y_value = y_row[y_offset];
                let u_value = get_fancy_chroma_value(
                    u_row_1[uv_offset],
                    u_row_1[uv_offset + 1],
                    u_row_2[uv_offset],
                    u_row_2[uv_offset + 1],
                );
                let v_value = get_fancy_chroma_value(
                    v_row_1[uv_offset],
                    v_row_1[uv_offset + 1],
                    v_row_2[uv_offset],
                    v_row_2[uv_offset + 1],
                );
                set_pixel(&mut rgb[rgb_offset..rgb_offset + 3], y_value, u_value, v_value);
            }
            {
                let y_value = y_row[y_offset + 1];
                let u_value = get_fancy_chroma_value(
                    u_row_1[uv_offset + 1],
                    u_row_1[uv_offset],
                    u_row_2[uv_offset + 1],
                    u_row_2[uv_offset],
                );
                let v_value = get_fancy_chroma_value(
                    v_row_1[uv_offset + 1],
                    v_row_1[uv_offset],
                    v_row_2[uv_offset + 1],
                    v_row_2[uv_offset],
                );
                set_pixel(
                    &mut rgb[rgb_offset + 3..rgb_offset + 6],
                    y_value,
                    u_value,
                    v_value,
                );
            }
            y_offset += 2;
            uv_offset += 1;
            rgb_offset += 6;
        }

        if y_offset < width {
            let final_u_1 = *u_row_1.last().unwrap();
            let final_u_2 = *u_row_2.last().unwrap();
            let final_v_1 = *v_row_1.last().unwrap();
            let final_v_2 = *v_row_2.last().unwrap();

            let u_value = get_fancy_chroma_value(final_u_1, final_u_1, final_u_2, final_u_2);
            let v_value = get_fancy_chroma_value(final_v_1, final_v_1, final_v_2, final_v_2);
            set_pixel(
                &mut rgb[rgb_offset..rgb_offset + 3],
                y_row[y_offset],
                u_value,
                v_value,
            );
        }
    }
}

#[cfg(target_arch = "wasm32")]
pub(crate) use wasm_fused::fused_row_2uv_wasm;

// ============================================================================
// Dispatch: pick the best tier at runtime
// ============================================================================

use crate::common::prediction::SimdTokenType;

/// Fused fancy-upsample + YUV→RGB for a row with two chroma rows (interior rows).
/// Dispatches to the best available SIMD tier; falls back to scalar.
/// BPP must be 3 (RGB). For RGBA (BPP=4), falls back to the old path.
#[allow(unused_variables)]
pub(crate) fn fused_fill_row_fancy_with_2_uv_rows(
    row_buffer: &mut [u8],
    y_row: &[u8],
    u_row_1: &[u8],
    u_row_2: &[u8],
    v_row_1: &[u8],
    v_row_2: &[u8],
    simd_token: SimdTokenType,
) -> bool {
    // Only for RGB (BPP=3) and rows wide enough for SIMD
    if y_row.len() < 17 {
        return false;
    }

    #[cfg(target_arch = "x86_64")]
    if let Some(token) = simd_token {
        fused_row_2uv_x86(token, row_buffer, y_row, u_row_1, u_row_2, v_row_1, v_row_2);
        return true;
    }

    #[cfg(target_arch = "aarch64")]
    if let Some(token) = simd_token {
        fused_row_2uv_neon(token, row_buffer, y_row, u_row_1, u_row_2, v_row_1, v_row_2);
        return true;
    }

    #[cfg(target_arch = "wasm32")]
    if let Some(token) = simd_token {
        fused_row_2uv_wasm(token, row_buffer, y_row, u_row_1, u_row_2, v_row_1, v_row_2);
        return true;
    }

    false
}
