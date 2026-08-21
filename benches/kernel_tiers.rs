//! Per-kernel NEON-vs-forced-scalar for zenwebp's VP8 transforms.
//!
//! zenwebp has 287 dispatch sites and its tier bench measures whole-image
//! decode/encode (1.53-1.69x). A whole-pipeline ratio cannot show an individual
//! kernel running BELOW its scalar tier — that gap has hidden six real
//! regressions elsewhere in this sweep (zenquant 0.58x, zenav1-svt inverse
//! transforms 0.59x, iwssim 0.87x, linear-srgb 0.93x, zenresize 0.94x, and
//! zenjpeg's DCT which had no ARM arm at all).
//!
//! NEON is BASELINE on aarch64, so the "scalar" arm is autovectorized too:
//! ~1.00x means LLVM already matched it; BELOW 1.00 is a bug.
//!
//! Run: `cargo bench --features _dev --bench kernel_tiers`

use zenbench::prelude::*;
use zenwebp::common::transform::__bench_kernels as transform;

#[cfg(target_arch = "aarch64")]
type TierToken = archmage::NeonToken;
#[cfg(target_arch = "x86_64")]
type TierToken = archmage::X64V3Token;

#[cfg(any(target_arch = "aarch64", target_arch = "x86_64"))]
const TIER_NAME: &str = if cfg!(target_arch = "aarch64") {
    "neon"
} else {
    "v3(avx2)"
};

#[cfg(any(target_arch = "aarch64", target_arch = "x86_64"))]
fn set_simd(on: bool) -> bool {
    use archmage::SimdToken;
    TierToken::dangerously_disable_token_process_wide(!on).is_ok()
}
#[cfg(not(any(target_arch = "aarch64", target_arch = "x86_64")))]
fn set_simd(_on: bool) -> bool {
    false
}

fn coeffs(seed: u32) -> [i32; 16] {
    let mut s = seed | 1;
    let mut b = [0i32; 16];
    for v in b.iter_mut() {
        s = s.wrapping_mul(1_664_525).wrapping_add(1_013_904_223);
        // Coefficient-shaped: mostly small, occasional large.
        *v = ((s >> 20) as i32 % 512) - 256;
    }
    b
}

fn bench(suite: &mut Suite) {
    if !set_simd(true) || !set_simd(false) {
        eprintln!("[kernel_tiers] SIMD tier not toggleable here. Skipping.");
        return;
    }
    set_simd(true);
    eprintln!("[kernel_tiers] comparing {TIER_NAME} vs forced scalar");

    macro_rules! t {
        ($name:expr, $call:path) => {
            suite.compare($name, |g| {
                for (arm, simd) in [(TIER_NAME, true), ("scalar", false)] {
                    g.bench(arm, move |b| {
                        b.with_input(move || {
                            set_simd(simd);
                            coeffs(7)
                        })
                        .run(move |mut blk| {
                            $call(&mut blk);
                            blk
                        })
                    });
                }
            });
        };
    }

    t!("idct4x4", transform::idct4x4);
    t!("dct4x4", transform::dct4x4);
    t!("iwht4x4", transform::iwht4x4);
    t!("wht4x4", transform::wht4x4);

    // ---- encoder distortion kernels ----
    // The transform group above found 2 regressions in 4 kernels, so the next
    // hottest module is worth sweeping too rather than assuming it is clean.
    // These are `pub` + `incant!`-dispatched, so they need no shim.
    {
        use zenwebp::encoder::cost::distortion;
        const STRIDE: usize = 32;
        let plane_a: &'static [u8] = Box::leak(
            (0..STRIDE * 32)
                .map(|i| ((i * 7919) % 251) as u8)
                .collect::<Vec<u8>>()
                .into_boxed_slice(),
        );
        let plane_b: &'static [u8] = Box::leak(
            (0..STRIDE * 32)
                .map(|i| ((i * 5779) % 241) as u8)
                .collect::<Vec<u8>>()
                .into_boxed_slice(),
        );
        // The VP8 "weights" table shape; values are irrelevant to timing.
        let w: &'static [u16; 16] = Box::leak(Box::new([
            38, 32, 20, 9, 32, 28, 17, 7, 20, 17, 10, 4, 9, 7, 4, 2,
        ]));

        macro_rules! d {
            ($name:expr, $call:expr) => {
                suite.compare($name, |g| {
                    for (arm, simd) in [(TIER_NAME, true), ("scalar", false)] {
                        g.bench(arm, move |b| {
                            b.iter(move || {
                                set_simd(simd);
                                $call
                            })
                        });
                    }
                });
            };
        }
        d!(
            "tdisto_4x4",
            distortion::tdisto_4x4(plane_a, plane_b, STRIDE, w)
        );
        d!(
            "tdisto_8x8",
            distortion::tdisto_8x8(plane_a, plane_b, STRIDE, w)
        );
        d!(
            "tdisto_16x16",
            distortion::tdisto_16x16(plane_a, plane_b, STRIDE, w)
        );
        // Measured BOTH ways on purpose. `is_flat_source_16_scalar` returns at
        // the first differing pixel, so on non-flat input it exits after ~1
        // comparison while any vector path loads a whole 16-byte row. Timing
        // only that case makes SIMD look broken for a reason that has nothing
        // to do with the kernel — the same artifact that produced zenpng's
        // apparent `is_opaque` loss. The flat case forces full traversal and is
        // the one that actually exercises the vector work.
        let flat: &'static [u8] = Box::leak(vec![128u8; STRIDE * 32].into_boxed_slice());
        d!(
            "is_flat_source_16/flat",
            distortion::is_flat_source_16(flat, STRIDE)
        );
        d!(
            "is_flat_source_16/early-exit",
            distortion::is_flat_source_16(plane_a, STRIDE)
        );
    }

    // ---- VP8L lossless: subtract-green ----
    // Per-pixel transform applied across the whole image on the lossless path.
    // `pub` + `incant!`-dispatched, so no shim needed.
    {
        const N: usize = 1 << 20;
        let px: &'static [u32] = Box::leak(
            (0..N)
                .map(|i| ((i as u32).wrapping_mul(2_654_435_761)) | 0xFF00_0000)
                .collect::<Vec<u32>>()
                .into_boxed_slice(),
        );
        suite.compare("apply_subtract_green/1MP", move |g| {
            g.throughput(Throughput::Bytes((N * 4) as u64));
            for (arm, simd) in [(TIER_NAME, true), ("scalar", false)] {
                g.bench(arm, move |b| {
                    b.with_input(move || {
                        set_simd(simd);
                        px.to_vec()
                    })
                    .run(move |mut v| {
                        zenwebp::encoder::vp8l::transforms::apply_subtract_green(&mut v);
                        v
                    })
                });
            }
        });
    }

    // ---- WebP decode: exact YUV 4:2:0 -> RGB ----
    // The biggest uncovered module by dispatch count. This is the lossy decode
    // output path, run once per frame over every pixel.
    {
        const W: usize = 1280;
        const H: usize = 720;
        let cw = W.div_ceil(2);
        let chh = H.div_ceil(2);
        let yb: &'static [u8] = Box::leak(
            (0..W * H)
                .map(|i| ((i * 7919) % 251) as u8)
                .collect::<Vec<u8>>()
                .into_boxed_slice(),
        );
        let ub: &'static [u8] = Box::leak(
            (0..cw * chh)
                .map(|i| ((i * 5779) % 241) as u8)
                .collect::<Vec<u8>>()
                .into_boxed_slice(),
        );
        let vb: &'static [u8] = Box::leak(
            (0..cw * chh)
                .map(|i| ((i * 3571) % 239) as u8)
                .collect::<Vec<u8>>()
                .into_boxed_slice(),
        );

        for (label, bpp) in [("rgb", 3usize), ("rgba", 4usize)] {
            suite.compare(&format!("yuv420_to_rgb_exact/{label}"), move |g| {
                g.throughput(Throughput::Bytes((W * H * bpp) as u64));
                for (arm, simd) in [(TIER_NAME, true), ("scalar", false)] {
                    g.bench(arm, move |b| {
                        b.with_input(move || { set_simd(simd); Vec::<u8>::new() })
                            .run(move |mut out| {
                                zenwebp::decoder::vp8v2::yuv_exact::__bench_kernels::yuv420_to_rgb_exact(
                                    yb, ub, vb, W, H, W, cw, &mut out, bpp,
                                );
                                out
                            })
                    });
                }
            });
        }
    }

    // ---- residue add + lossless inverse transform ----
    {
        let r16: &'static [i16; 16] = Box::leak(Box::new([
            12, -34, 56, -7, 89, -21, 3, -45, 67, -8, 90, -12, 34, -56, 7, -89,
        ]));
        suite.compare("add_residue_i16", |g| {
            for (arm, simd) in [(TIER_NAME, true), ("scalar", false)] {
                g.bench(arm, move |b| {
                    b.with_input(move || {
                        set_simd(simd);
                        [128u8; 256]
                    })
                    .run(move |mut p| {
                        zenwebp::common::prediction::__bench_kernels::add_residue_i16_256(
                            &mut p, r16, 0, 0, 16,
                        );
                        p
                    })
                });
            }
        });

        // Decode-side inverse subtract-green over a 1 MP ARGB image.
        const N: usize = (1 << 20) * 4;
        let img: &'static [u8] = Box::leak(
            (0..N)
                .map(|i| ((i * 7919) % 251) as u8)
                .collect::<Vec<u8>>()
                .into_boxed_slice(),
        );
        suite.compare("inverse_subtract_green/1MP", move |g| {
            g.throughput(Throughput::Bytes(N as u64));
            for (arm, simd) in [(TIER_NAME, true), ("scalar", false)] {
                g.bench(arm, move |b| {
                    b.with_input(move || { set_simd(simd); img.to_vec() })
                        .run(move |mut v| {
                            zenwebp::decoder::lossless_transform::__bench_kernels::apply_subtract_green_transform(&mut v);
                            v
                        })
                });
            }
        });
    }

    // ---- encoder quantization ----
    // Built through `VP8Matrix::new` rather than a hand-filled struct: its
    // fields are interdependent (iq = (1<<QFIX)/q, bias, zthresh), so a
    // literal would be self-inconsistent and the timings meaningless.
    {
        use zenwebp::encoder::quantize::{
            MatrixType, VP8Matrix, quantize_block_simd, quantize_dequantize_block_simd,
        };
        let m: &'static VP8Matrix = Box::leak(Box::new(VP8Matrix::new(8, 12, MatrixType::Y1)));
        let src16: &'static [i16; 16] = Box::leak(Box::new([
            420, -180, 96, -44, 210, -130, 60, -22, 88, -300, 150, -70, 33, -18, 9, -400,
        ]));
        let src32: &'static [i32; 16] = Box::leak(Box::new([
            420, -180, 96, -44, 210, -130, 60, -22, 88, -300, 150, -70, 33, -18, 9, -400,
        ]));

        suite.compare("quantize_block_simd", |g| {
            for (arm, simd) in [(TIER_NAME, true), ("scalar", false)] {
                g.bench(arm, move |b| {
                    b.with_input(move || {
                        set_simd(simd);
                        *src32
                    })
                    .run(move |mut c| {
                        let r = quantize_block_simd(&mut c, m, false);
                        (c, r)
                    })
                });
            }
        });
        suite.compare("quantize_dequantize_block", |g| {
            for (arm, simd) in [(TIER_NAME, true), ("scalar", false)] {
                g.bench(arm, move |b| {
                    b.with_input(move || set_simd(simd)).run(move |_| {
                        let (mut q, mut d) = ([0i16; 16], [0i16; 16]);
                        let r = quantize_dequantize_block_simd(src16, m, false, &mut q, &mut d);
                        (q, d, r)
                    })
                });
            }
        });
        suite.compare("dequantize_block", |g| {
            for (arm, simd) in [(TIER_NAME, true), ("scalar", false)] {
                g.bench(arm, move |b| {
                    b.with_input(move || {
                        set_simd(simd);
                        *src32
                    })
                    .run(move |mut c| {
                        m.dequantize_block(&mut c);
                        c
                    })
                });
            }
        });
    }

    set_simd(true);
}

zenbench::main!(bench);
