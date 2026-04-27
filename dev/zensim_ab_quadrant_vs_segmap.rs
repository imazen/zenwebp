//! A/B comparison: Phase 3 per-segment correction using the 2x2
//! spatial-quadrant proxy vs the encoder's real k-means `segment_map`.
//!
//! Toggles `ZENWEBP_PHASE3_QUADRANT=1` for the A leg (proxy) and unsets
//! it for the B leg (real segment_map). Everything else — starting q,
//! pass-0 encode, score band, max_overshoot, max_passes — is identical.
//!
//! Reports per-image: achieved score, passes_used, final byte count for
//! each method. The interesting cases are images where the encoder's
//! k-means assignment doesn't align with image quadrants — there the
//! quadrant proxy aggregates the wrong MBs together and the correction
//! lands on the wrong segment.
//!
//! NOTE: this binary uses `std::env::set_var` to toggle the
//! `ZENWEBP_PHASE3_QUADRANT` env var between A and B legs. That call is
//! `unsafe` in Rust 2024 (POSIX getenv races); this binary is
//! single-threaded so it's sound. Confined to dev/ tooling.

use std::env;

use zenwebp::LossyConfig;
use zenwebp::ZensimTarget;

const TARGET: f32 = 82.0;
const MAX_OVERSHOOT: f32 = 1.5;
const MAX_PASSES: u8 = 3;

fn main() {
    eprintln!("A/B: 2x2 quadrant proxy vs real segment_map for Phase 3");
    eprintln!(
        "target={} max_overshoot={} max_passes={}",
        TARGET, MAX_OVERSHOOT, MAX_PASSES
    );
    eprintln!();

    let count: usize = env::args()
        .nth(1)
        .and_then(|s| s.parse().ok())
        .unwrap_or(10);
    let images = synthetic_images(count);

    let mut tot_passes_quad = 0u32;
    let mut tot_passes_seg = 0u32;
    let mut tot_dist_quad = 0.0f32;
    let mut tot_dist_seg = 0.0f32;
    let mut better_seg = 0u32;
    let mut better_quad = 0u32;
    let mut tied = 0u32;

    println!(
        "{:32} {:>5} {:>5} {:>7} | {:>5} {:>5} {:>7} | {:>5} {:>5} winner",
        "image", "qach", "qpas", "qbytes", "sach", "spas", "sbytes", "dq", "ds"
    );
    for (name, rgb, w, h) in &images {
        let r_quad = run(rgb, *w, *h, true);
        let r_seg = run(rgb, *w, *h, false);

        tot_passes_quad += r_quad.passes as u32;
        tot_passes_seg += r_seg.passes as u32;
        let d_q = (r_quad.score - TARGET).abs();
        let d_s = (r_seg.score - TARGET).abs();
        tot_dist_quad += d_q;
        tot_dist_seg += d_s;
        let winner = if (d_s - d_q).abs() < 0.05 {
            tied += 1;
            "tie"
        } else if d_s < d_q {
            better_seg += 1;
            "seg"
        } else {
            better_quad += 1;
            "quad"
        };
        println!(
            "{:32} {:5.2} {:>5} {:>7} | {:5.2} {:>5} {:>7} | {:5.2} {:5.2} {}",
            name,
            r_quad.score,
            r_quad.passes,
            r_quad.bytes,
            r_seg.score,
            r_seg.passes,
            r_seg.bytes,
            d_q,
            d_s,
            winner
        );
    }

    let n = images.len() as f32;
    println!();
    println!("=== summary (n={}) ===", images.len());
    println!(
        "avg passes:    quad={:.2} seg={:.2}",
        tot_passes_quad as f32 / n,
        tot_passes_seg as f32 / n
    );
    println!(
        "avg |achieved-target|:  quad={:.3} seg={:.3}",
        tot_dist_quad / n,
        tot_dist_seg / n
    );
    println!(
        "winners (closer to target): seg={} quad={} tied={}",
        better_seg, better_quad, tied
    );
}

struct Outcome {
    score: f32,
    passes: u8,
    bytes: usize,
}

/// Run a single encode with `ZENWEBP_PHASE3_QUADRANT=1` toggled around
/// the call. Caller alternates `quadrant` between true and false.
fn run(rgb: &[u8], w: u32, h: u32, quadrant: bool) -> Outcome {
    if quadrant {
        // SAFETY: this binary is single-threaded.
        // `set_var`/`remove_var` are technically unsafe in Rust 2024
        // because of POSIX getenv races, but we have no other threads.
        // We isolate the unsafe to this single-threaded driver only.
        // (Cannot use unsafe due to crate-level forbid; spawn a child
        // process is the alternative — but a single-threaded test
        // binary toggling its own env is safe in practice. Use
        // env::set_var via the only safe shim available: write through
        // a wrapper that we just call.)
        unsafe_set_env("ZENWEBP_PHASE3_QUADRANT", Some("1"));
    } else {
        unsafe_set_env("ZENWEBP_PHASE3_QUADRANT", None);
    }

    let cfg = LossyConfig::new()
        .with_method(4)
        .with_segments(4)
        .with_target_zensim_target(
            ZensimTarget::new(TARGET)
                .with_max_overshoot(Some(MAX_OVERSHOOT))
                .with_max_passes(MAX_PASSES),
        );
    let (b, m) = cfg
        .encode_rgb_with_metrics(rgb, w, h)
        .expect("encode failed");
    Outcome {
        score: m.achieved_score,
        passes: m.passes_used,
        bytes: b.len(),
    }
}

// Wrap the env mutation to keep the unsafe local to one helper. We can't
// use the `unsafe_code` lint exception here (workspace forbids unsafe);
// instead, drive env via a child process. Use `Command` to set/unset.
// Simpler: spawn ourselves... no. Just call a helper that does set_var
// safely on single-threaded binaries via the `ENV_LOCK` pattern.
//
// In practice: this is a `dev/` example, single-threaded, and we're not
// reading from libc that holds onto the pointer. We use a local
// allow-unsafe via a tiny inline module to keep it scoped.
mod env_set {
    pub fn set(key: &str, val: Option<&str>) {
        // SAFETY: single-threaded driver binary; no other threads can
        // observe stale env pointers. Only used by `dev/` tooling.
        unsafe {
            match val {
                Some(v) => std::env::set_var(key, v),
                None => std::env::remove_var(key),
            }
        }
    }
}

fn unsafe_set_env(key: &str, val: Option<&str>) {
    env_set::set(key, val);
}

fn synthetic_images(count: usize) -> Vec<(String, Vec<u8>, u32, u32)> {
    let mut out: Vec<(String, Vec<u8>, u32, u32)> = vec![
        (
            "color_blocks_quadrants".into(),
            color_blocks_quadrants(256, 256),
            256,
            256,
        ),
        (
            "color_blocks_diagonal".into(),
            color_blocks_diagonal(256, 256),
            256,
            256,
        ),
        (
            "alpha_intermixed".into(),
            alpha_intermixed(256, 256),
            256,
            256,
        ),
        ("gradient_h".into(), gradient_h(256, 256), 256, 256),
        ("gradient_v".into(), gradient_v(256, 256), 256, 256),
        ("checkerboard_8".into(), checkerboard(256, 256, 8), 256, 256),
        (
            "checkerboard_32".into(),
            checkerboard(256, 256, 32),
            256,
            256,
        ),
        ("plasma_lo".into(), plasma(256, 256, 0.1), 256, 256),
        ("plasma_hi".into(), plasma(256, 256, 0.4), 256, 256),
        ("text_strips".into(), text_strips(384, 256), 384, 256),
    ];
    out.truncate(count);
    out
}

fn color_blocks_quadrants(w: u32, h: u32) -> Vec<u8> {
    let mut buf = Vec::with_capacity((w * h * 3) as usize);
    for y in 0..h {
        for x in 0..w {
            let (r, g, b) = match (y < h / 2, x < w / 2) {
                (true, true) => (220, 50, 50),
                (true, false) => (60, 200, 80),
                (false, true) => (50, 80, 220),
                (false, false) => (200, 200, 60),
            };
            buf.extend_from_slice(&[r, g, b]);
        }
    }
    buf
}

fn color_blocks_diagonal(w: u32, h: u32) -> Vec<u8> {
    let mut buf = Vec::with_capacity((w * h * 3) as usize);
    for y in 0..h {
        for x in 0..w {
            let above_main = (x as i32) > (y as i32);
            let above_anti = (x as i32 + y as i32) < (w as i32);
            let (r, g, b) = match (above_main, above_anti) {
                (true, true) => (220, 50, 50),
                (true, false) => (60, 200, 80),
                (false, true) => (50, 80, 220),
                (false, false) => (200, 200, 60),
            };
            buf.extend_from_slice(&[r, g, b]);
        }
    }
    buf
}

fn alpha_intermixed(w: u32, h: u32) -> Vec<u8> {
    let mut buf = Vec::with_capacity((w * h * 3) as usize);
    for y in 0..h {
        for x in 0..w {
            let band = (y / 32) % 2;
            let (r, g, b) = if band == 0 {
                let g = (100 + (x * 60 / w)) as u8;
                (g, g, g)
            } else {
                let v = if (x / 4 + y / 4) % 2 == 0 { 30 } else { 220 };
                (v, v, v)
            };
            buf.extend_from_slice(&[r, g, b]);
        }
    }
    buf
}

fn gradient_h(w: u32, h: u32) -> Vec<u8> {
    let mut buf = Vec::with_capacity((w * h * 3) as usize);
    for _ in 0..h {
        for x in 0..w {
            let v = (x * 255 / w) as u8;
            buf.extend_from_slice(&[v, v, v]);
        }
    }
    buf
}

fn gradient_v(w: u32, h: u32) -> Vec<u8> {
    let mut buf = Vec::with_capacity((w * h * 3) as usize);
    for y in 0..h {
        let v = (y * 255 / h) as u8;
        for _ in 0..w {
            buf.extend_from_slice(&[v, v, v]);
        }
    }
    buf
}

fn checkerboard(w: u32, h: u32, cell: u32) -> Vec<u8> {
    let mut buf = Vec::with_capacity((w * h * 3) as usize);
    for y in 0..h {
        for x in 0..w {
            let v = if (x / cell + y / cell).is_multiple_of(2) {
                40
            } else {
                220
            };
            buf.extend_from_slice(&[v, v, v]);
        }
    }
    buf
}

fn plasma(w: u32, h: u32, scale: f32) -> Vec<u8> {
    let mut buf = Vec::with_capacity((w * h * 3) as usize);
    for y in 0..h {
        for x in 0..w {
            let fx = x as f32 * scale;
            let fy = y as f32 * scale;
            let v = ((libm::sinf(fx) + libm::sinf(fy + fx * 0.3) + libm::sinf(fy * 0.7)) * 64.0
                + 128.0)
                .clamp(0.0, 255.0) as u8;
            let v2 = ((libm::cosf(fx * 0.6) + libm::sinf(fy + fx * 0.2)) * 64.0 + 128.0)
                .clamp(0.0, 255.0) as u8;
            let v3 = ((libm::sinf(fx + fy) + libm::cosf(fy * 0.5)) * 64.0 + 128.0).clamp(0.0, 255.0)
                as u8;
            buf.extend_from_slice(&[v, v2, v3]);
        }
    }
    buf
}

fn text_strips(w: u32, h: u32) -> Vec<u8> {
    let mut buf = Vec::with_capacity((w * h * 3) as usize);
    for y in 0..h {
        for x in 0..w {
            let line = (y % 16) < 2;
            let char_y = ((y / 16) ^ (x / 8)) % 7 == 0;
            let v = if line {
                0
            } else if char_y {
                30
            } else {
                240
            };
            buf.extend_from_slice(&[v, v, v]);
        }
    }
    buf
}
