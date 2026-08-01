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
const TIER_NAME: &str = if cfg!(target_arch = "aarch64") { "neon" } else { "v3(avx2)" };

#[cfg(any(target_arch = "aarch64", target_arch = "x86_64"))]
fn set_simd(on: bool) -> bool {
    use archmage::SimdToken;
    TierToken::dangerously_disable_token_process_wide(!on).is_ok()
}
#[cfg(not(any(target_arch = "aarch64", target_arch = "x86_64")))]
fn set_simd(_on: bool) -> bool { false }

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
                        b.with_input(move || { set_simd(simd); coeffs(7) })
                            .run(move |mut blk| { $call(&mut blk); blk })
                    });
                }
            });
        };
    }

    t!("idct4x4", transform::idct4x4);
    t!("dct4x4", transform::dct4x4);
    t!("iwht4x4", transform::iwht4x4);
    t!("wht4x4", transform::wht4x4);

    set_simd(true);
}

zenbench::main!(bench);
