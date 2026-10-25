//! Optimized alpha blending routines based on libwebp
//!
//! <https://github.com/webmproject/libwebp/blob/e4f7a9f0c7c9fbfae1568bc7fa5c94b989b50872/src/demux/anim_decode.c#L215-L267>

const fn channel_shift(i: u32) -> u32 {
    i * 8
}

/// Blend a single channel of `src` over `dst`, given their alpha channel values.
/// `src` and `dst` are assumed to be NOT pre-multiplied by alpha.
fn blend_channel_nonpremult(
    src: u32,
    src_a: u8,
    dst: u32,
    dst_a: u8,
    scale: u32,
    shift: u32,
) -> u8 {
    let src_channel = ((src >> shift) & 0xff) as u8;
    let dst_channel = ((dst >> shift) & 0xff) as u8;
    let blend_unscaled =
        (u32::from(src_channel) * u32::from(src_a)) + (u32::from(dst_channel) * u32::from(dst_a));
    debug_assert!(u64::from(blend_unscaled) < (1u64 << 32) / u64::from(scale));
    ((blend_unscaled * scale) >> channel_shift(3)) as u8
}

/// Blend `src` over `dst` assuming they are NOT pre-multiplied by alpha.
fn blend_pixel_nonpremult(src: u32, dst: u32) -> u32 {
    let src_a = ((src >> channel_shift(3)) & 0xff) as u8;

    if src_a == 0 {
        dst
    } else {
        let dst_a = ((dst >> channel_shift(3)) & 0xff) as u8;
        // Approximate integer arithmetic for: dst_factor_a = (dst_a * (255 - src_a)) / 255,
        // exactly as libwebp's BlendPixelNonPremult computes it. A rounded
        // divide-by-255 is closer to the real-valued formula, but it makes
        // animated output diverge from libwebp's WebPAnimDecoder by 1 level
        // wherever a translucent pixel lands on a translucent canvas.
        let dst_factor_a = (u32::from(dst_a) * (256 - u32::from(src_a))) >> 8;
        let blend_a = u32::from(src_a) + dst_factor_a;
        let scale = (1u32 << 24) / blend_a;

        let blend_r =
            blend_channel_nonpremult(src, src_a, dst, dst_factor_a as u8, scale, channel_shift(0));
        let blend_g =
            blend_channel_nonpremult(src, src_a, dst, dst_factor_a as u8, scale, channel_shift(1));
        let blend_b =
            blend_channel_nonpremult(src, src_a, dst, dst_factor_a as u8, scale, channel_shift(2));
        debug_assert!(u32::from(src_a) + dst_factor_a < 256);

        (u32::from(blend_r) << channel_shift(0))
            | (u32::from(blend_g) << channel_shift(1))
            | (u32::from(blend_b) << channel_shift(2))
            | (blend_a << channel_shift(3))
    }
}

pub(crate) fn do_alpha_blending(buffer: [u8; 4], canvas: [u8; 4]) -> [u8; 4] {
    // The original C code contained different shift functions for different endianness,
    // but they didn't work when ported to Rust directly (and probably didn't work in C either).
    // So instead we reverse the order of bytes on big-endian here, at the interface.
    // `from_le_bytes` is a no-op on little endian (most systems) and a cheap shuffle on big endian.
    blend_pixel_nonpremult(u32::from_le_bytes(buffer), u32::from_le_bytes(canvas)).to_le_bytes()
}

#[cfg(test)]
mod tests {
    use super::*;

    /// FNV-1a over `do_alpha_blending` for every (src, dst) red/alpha pair,
    /// sampled every `step` values per axis. G and B take the same
    /// per-channel path as R.
    fn blend_hash(step: usize) -> u64 {
        let mut h: u64 = 1_469_598_103_934_665_603;
        for r1 in (0..=255u8).step_by(step) {
            for a1 in (0..=255u8).step_by(step) {
                for r2 in (0..=255u8).step_by(step) {
                    for a2 in (0..=255u8).step_by(step) {
                        let out = do_alpha_blending([r1, 0, 0, a1], [r2, 0, 0, a2]);
                        h = (h ^ u64::from(u32::from_le_bytes(out)))
                            .wrapping_mul(1_099_511_628_211);
                    }
                }
            }
        }
        h
    }

    // Expected hashes come from libwebp's own BlendPixelNonPremult
    // (src/demux/anim_decode.c, libwebp-sys 0.14.4 vendor), built by
    // `dev/libwebp-blend-oracle/build.sh`: `blend_oracle 5` and `blend_oracle 1`.

    #[test]
    fn alpha_blending_matches_libwebp_sampled() {
        assert_eq!(blend_hash(5), 0x2ebaec7cf153c178);
    }

    #[test]
    #[ignore] // 4.3e9 blends; run locally with --release when changing the function.
    fn alpha_blending_matches_libwebp_exhaustive() {
        assert_eq!(blend_hash(1), 0x348e86d5f637711b);
    }
}
